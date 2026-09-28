"""Worker protocol, transport and lifecycle tests.

These tests never build a simulator: the pool is driven against a synthetic
worker process that speaks the real protocol, which is what makes the failure
paths testable at all.  A worker that dies, hangs, or answers the wrong request
must fail the iteration instead of leaving the parent blocked or holding a
partial payload, and those are exactly the cases a happy-path test misses.
"""

from __future__ import annotations

import functools
import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from mjlab.tasks.tracking.distillation import worker as worker_module
from mjlab.tasks.tracking.distillation.cohort_setup import (
  CohortSetup,
  worker_evaluation_seed,
  worker_generator_seed,
)
from mjlab.tasks.tracking.distillation.observations import PackedObservationBatch
from mjlab.tasks.tracking.distillation.reset_policy import make_reset_policy
from mjlab.tasks.tracking.distillation.storage import LabeledReplayBatch
from mjlab.tasks.tracking.distillation.vae_config import DEFAULT_SCHEMA
from mjlab.tasks.tracking.distillation.worker import (
  CollectReply,
  CollectRequest,
  DescribeReply,
  DescribeWorker,
  EvaluateReply,
  EvaluateRequest,
  ShutdownWorker,
  WorkerEnvironmentDescription,
  WorkerError,
  WorkerFailure,
  WorkerPool,
  WorkerPoolError,
  WorkerSpec,
  _stage_batch,
  _Worker,
  derive_reset_rng,
  watch_parent,
  worker_specs,
)


def _setup(tmp_path) -> CohortSetup:
  return CohortSetup(
    manifest=tmp_path / "manifest.yaml",
    repo_root=tmp_path,
    teacher_ids=("tennis_000",),
    task_id=None,
    phase_policy="uniform",
    reset_policy=make_reset_policy(),
    base_seed=5,
  )


def _specs(
  tmp_path, devices=("cpu", "cpu"), *, num_envs: int = 4
) -> tuple[WorkerSpec, ...]:
  return worker_specs(_setup(tmp_path), devices, num_envs=num_envs)


def _batch(rows: int, worker_index: int) -> LabeledReplayBatch:
  """One synthetic collected batch, tagged by the worker that produced it."""
  return LabeledReplayBatch(
    observations=PackedObservationBatch(
      reference=torch.full((rows, 4), float(worker_index)),
      conditioning=torch.zeros(rows, 3),
      schema=DEFAULT_SCHEMA,
    ),
    teacher_action=torch.full((rows, 2), float(worker_index)),
    motion_id=torch.zeros(rows, dtype=torch.long),
    teacher_id=torch.zeros(rows, dtype=torch.long),
    reference_frame=torch.arange(rows, dtype=torch.long),
    episode_id=torch.zeros(rows, dtype=torch.long),
    collector_iteration=torch.zeros(rows, dtype=torch.long),
  )


def _synthetic_worker(spec, commands, replies, mode: str = "normal") -> None:
  """A protocol-faithful worker that owns no environment.

  Kept module level and spawn-safe (called through ``functools.partial``) so the
  pool under test is the real one, including its process and queue handling.
  """
  while True:
    request = commands.get(timeout=60.0)
    if isinstance(request, ShutdownWorker):
      return
    if isinstance(request, CollectRequest):
      if mode == "die":
        os._exit(3)
      if mode == "hang":
        time.sleep(120.0)
        continue
      if mode == "slow_then_hang":
        if spec.worker_index == 0:
          # Answers inside the timeout, so the only worker the pool can be
          # waiting on is the one that never answers.
          time.sleep(3.0)
        else:
          # Never answers: the point is that the pool must still bound *this*
          # worker from the moment it was asked, not from when it is awaited.
          time.sleep(120.0)
          continue
      if mode == "slow":
        marker = Path(spec.setup.repo_root)
        (marker / f"worker-{spec.worker_index}-start").write_text(str(time.monotonic()))
        time.sleep(2.0)
        (marker / f"worker-{spec.worker_index}-end").write_text(str(time.monotonic()))
    call_id = request.call_id
    if mode == "wrong_id" and isinstance(request, CollectRequest):
      call_id += 1000
    if mode == "fail" and isinstance(request, CollectRequest):
      replies.put(
        WorkerFailure(
          call_id=request.call_id, message="ValueError: synthetic failure", traceback=""
        )
      )
      continue
    if isinstance(request, DescribeWorker):
      replies.put(
        DescribeReply(
          call_id=call_id,
          description=WorkerEnvironmentDescription(
            worker_index=spec.worker_index,
            device=spec.device,
            num_envs=spec.num_envs,
            schema=DEFAULT_SCHEMA,
            reset_policy_enabled=False,
            reset_provenance=None,
            sampling_mode="cyclic",
          ),
        )
      )
    elif isinstance(request, CollectRequest):
      rows = spec.num_envs * request.steps
      replies.put(
        CollectReply(
          call_id=call_id,
          iteration=request.iteration,
          ticks=request.steps,
          samples=rows,
          teacher_steps=rows,
          student_steps=0,
          disagreement_mean=0.0,
          diagnostics=(),
          motion_stats=(),
          boundaries=(),
          rows_per_tick=spec.num_envs,
          batch=_batch(rows, spec.worker_index),
        )
      )
    elif isinstance(request, EvaluateRequest):
      raise AssertionError("the synthetic worker does not evaluate")
    else:
      raise AssertionError(f"unexpected request {type(request).__name__}")


def _pool(tmp_path, mode: str = "normal", **kwargs) -> WorkerPool:
  return WorkerPool(
    _specs(tmp_path),
    target=functools.partial(_synthetic_worker, mode=mode),
    request_timeout_s=kwargs.pop("request_timeout_s", 30.0),
    **kwargs,
  )


def test_pool_collects_the_shards_concurrently(tmp_path) -> None:
  """A slow shard must not delay its neighbour's start.

  Asking worker 1 only after worker 0's reply has arrived makes collection wall
  time the sum of the shards rather than the maximum.  Every counter, report and
  checkpoint would look identical, so only a timing property catches it -- and
  without it the whole point of the multi-GPU run is gone.

  The property is asserted as interval overlap rather than a wall-clock bound,
  because spawning a process and importing torch costs more than the sleep that
  makes serialization visible.
  """
  pool = _pool(tmp_path, mode="slow")
  pool.start()
  try:
    pool.collect(
      iteration=0,
      steps=1,
      teacher_probability=0.0,
      reset=True,
      rollout_latent="mean",
      weights={},
    )
  finally:
    pool.close()

  started = {
    index: float((tmp_path / f"worker-{index}-start").read_text()) for index in (0, 1)
  }
  ended = {
    index: float((tmp_path / f"worker-{index}-end").read_text()) for index in (0, 1)
  }
  assert started[0] < ended[1] and started[1] < ended[0], (started, ended)


def test_pool_collects_every_shard_in_worker_order(tmp_path) -> None:
  """Both shards answer, and the rows stay attributable to their worker.

  The merge depends on receiver order, so the pool must return replies in
  worker order regardless of which process answered first.
  """
  with _pool(tmp_path) as pool:
    descriptions = pool.describe()
    assert [item.worker_index for item in descriptions] == [0, 1]
    assert [item.num_envs for item in descriptions] == [2, 2]
    replies = pool.collect(
      iteration=3,
      steps=2,
      teacher_probability=0.0,
      reset=True,
      rollout_latent="mean",
      weights={},
    )
  assert [reply.iteration for reply in replies] == [3, 3]
  assert [reply.rows_per_tick for reply in replies] == [2, 2]
  assert [reply.ticks for reply in replies] == [2, 2]
  for index, reply in enumerate(replies):
    values = reply.batch.observations.reference
    assert values.shape == (4, 4)
    assert torch.all(values == float(index))
    assert reply.batch.teacher_action.shape == (4, 2)


def test_pool_rejects_a_reply_that_does_not_match_the_outstanding_call(
  tmp_path,
) -> None:
  """A mismatched call id is a protocol failure, not a usable payload.

  Accepting it would attach one request's rows to another request's identity,
  which is silent corruption rather than a crash.
  """
  pool = _pool(tmp_path, mode="wrong_id")
  pool.start()
  with pytest.raises(WorkerPoolError) as error:
    pool.collect(
      iteration=0,
      steps=1,
      teacher_probability=0.0,
      reset=True,
      rollout_latent="mean",
      weights={},
    )
  assert "answered call" in str(error.value)
  assert pool._processes == []
  pool.close()


def test_pool_aborts_when_a_worker_exits_mid_request(tmp_path) -> None:
  """A dead worker fails the iteration instead of blocking the parent."""
  pool = _pool(tmp_path, mode="die")
  pool.start()
  with pytest.raises(WorkerPoolError) as error:
    pool.collect(
      iteration=0,
      steps=1,
      teacher_probability=0.0,
      reset=True,
      rollout_latent="mean",
      weights={},
    )
  assert "exited with code" in str(error.value)
  assert pool._processes == []
  pool.close()


def test_pool_aborts_on_a_worker_that_stops_answering(tmp_path) -> None:
  """A hung worker is bounded by the request timeout, then torn down."""
  pool = _pool(tmp_path, mode="hang", request_timeout_s=2.0)
  pool.start()
  started = time.monotonic()
  with pytest.raises(WorkerPoolError) as error:
    pool.collect(
      iteration=0,
      steps=1,
      teacher_probability=0.0,
      reset=True,
      rollout_latent="mean",
      weights={},
    )
  elapsed = time.monotonic() - started
  assert "did not answer" in str(error.value)
  assert elapsed < 30.0, elapsed
  assert pool._processes == []
  pool.close()


def test_pool_reports_a_worker_side_exception_with_its_request(tmp_path) -> None:
  """A worker-side failure is attributed to the request that caused it."""
  pool = _pool(tmp_path, mode="fail")
  pool.start()
  with pytest.raises(WorkerPoolError) as error:
    pool.collect(
      iteration=0,
      steps=1,
      teacher_probability=0.0,
      reset=True,
      rollout_latent="mean",
      weights={},
    )
  message = str(error.value)
  assert "synthetic failure" in message
  assert "CollectRequest" in message
  pool.close()


def test_pool_close_leaves_no_running_worker(tmp_path) -> None:
  """Teardown must not leave a child, or a GPU context, behind."""
  pool = _pool(tmp_path)
  pool.start()
  processes = list(pool._processes)
  assert all(process.is_alive() for process in processes)
  pool.close()
  assert all(not process.is_alive() for process in processes)
  # Closing twice is a no-op rather than an error: cleanup runs on every path.
  pool.close()


def test_pool_refuses_a_reused_gpu_device_and_allows_repeated_cpu(tmp_path) -> None:
  """Two shards may not claim one GPU; repeated CPU shards are a test affordance.

  A repeated GPU device would put two environments and two collections on one
  card while the resolved identity claimed two shards.  Repeating ``'cpu'`` is
  how the N-worker merge path is exercised without GPUs, and a CPU shard
  contends for no physical device.
  """
  with pytest.raises(WorkerError):
    WorkerPool(_specs(tmp_path, devices=("cuda:1", "cuda:1")))
  pool = WorkerPool(_specs(tmp_path, devices=("cpu", "cpu", "cpu"), num_envs=6))
  assert [spec.num_envs for spec in pool.specs] == [2, 2, 2]
  assert [spec.worker_index for spec in pool.specs] == [0, 1, 2]


def test_worker_specs_require_an_even_split_and_contiguous_indices(tmp_path) -> None:
  """The split and the worker numbering are validated where they are built."""
  specs = worker_specs(_setup(tmp_path), ("cpu", "cpu"), num_envs=6)
  assert [spec.num_envs for spec in specs] == [3, 3]
  assert [spec.worker_index for spec in specs] == [0, 1]
  with pytest.raises(WorkerError):
    worker_specs(_setup(tmp_path), ("cpu", "cpu"), num_envs=5)
  with pytest.raises(WorkerError):
    worker_specs(_setup(tmp_path), (), num_envs=4)
  with pytest.raises(WorkerError):
    WorkerPool(
      [
        WorkerSpec(worker_index=1, device="cpu", num_envs=2, setup=_setup(tmp_path)),
        WorkerSpec(worker_index=3, device="cpu", num_envs=2, setup=_setup(tmp_path)),
      ]
    )


def test_staging_preserves_values_and_provenance_without_sharing_storage(
  tmp_path,
) -> None:
  """A staged batch must be equal, independent, and complete.

  The parent merges what the worker staged, so a missing provenance field or a
  view that aliases the worker's live tensors would corrupt the merged rows.
  """
  source = _batch(3, worker_index=2)
  staged = _stage_batch(source)

  assert torch.equal(staged.observations.reference, source.observations.reference)
  assert torch.equal(staged.teacher_action, source.teacher_action)
  assert torch.equal(staged.reference_frame, source.reference_frame)
  assert (
    staged.observations.reference.data_ptr() != source.observations.reference.data_ptr()
  )
  assert staged.teacher_action.data_ptr() != source.teacher_action.data_ptr()
  assert staged.observations.schema is source.observations.schema
  assert staged.reference_frame.tolist() == [0, 1, 2]
  assert staged.has_provenance == source.has_provenance


def test_staging_preserves_provenance_when_the_rows_carry_it(tmp_path) -> None:
  """A provenance-carrying batch stages every extra field.

  Dropping the standing-start provenance on the way out of a worker would make
  the parent's merged rows claim a longer reference history than they have.
  """
  rows = 2
  source = replace(
    _batch(rows, worker_index=0),
    initialization_kind=torch.full((rows,), 2, dtype=torch.long),
    segment_initial_reference_frame=torch.zeros(rows, dtype=torch.long),
    segment_age=torch.arange(rows, dtype=torch.long),
  )
  staged = _stage_batch(source)

  assert staged.has_provenance
  assert staged.initialization_kind is not None
  assert staged.initialization_kind.tolist() == [2, 2]
  assert staged.segment_initial_reference_frame is not None
  assert staged.segment_age is not None
  assert staged.segment_age.tolist() == [0, 1]


def test_exchange_deadline_is_measured_from_dispatch(tmp_path) -> None:
  """A slow reply must not hand another worker a fresh timeout budget.

  Requests are dispatched together and awaited in worker order, so a deadline
  anchored at the start of the wait would give worker 1 its full timeout only
  after worker 0 answered, and an N-worker exchange could take N timeouts.
  Here worker 0 answers after 5s and worker 1 never answers: from dispatch the
  pool fails at about 5s, while a receive-anchored deadline would wait 8s.
  """
  pool = _pool(
    tmp_path, mode="slow_then_hang", request_timeout_s=6.0, close_grace_s=0.5
  )
  pool.start()
  started = time.monotonic()
  try:
    # Warm the shards first: process spawn and import cost more than the
    # timeout, and they are not part of the deadline property under test.
    pool.describe()
    started = time.monotonic()
    with pytest.raises(WorkerPoolError) as error:
      pool.collect(
        iteration=0,
        steps=1,
        teacher_probability=0.0,
        reset=True,
        rollout_latent="mean",
        weights={},
      )
    elapsed = time.monotonic() - started
  finally:
    pool.close()
  message = str(error.value)
  # Worker 0 answers inside the budget, so only worker 1 can be the timeout.
  assert "worker 1 did not answer" in message, message
  assert "of dispatch" in message, message
  # Dispatch-anchored: about one timeout (plus at most one poll interval).
  # Receive-anchored: worker 0's 3s plus a full timeout, i.e. 9s and up.
  assert elapsed < 8.5, f"pool waited {elapsed:.1f}s, a receive-anchored deadline"


def test_parent_watchdog_exits_without_waiting_for_a_request() -> None:
  """A child must notice a dead parent even while a request is running.

  The loop's own check only runs between requests, and a collection can take
  minutes; a killed parent would leave that device busy for a rollout nobody
  will read.  The watcher lives in its own thread and fires on reparenting.
  """
  lost = threading.Event()
  state = {"pid": 4321}
  stop = watch_parent(
    4321,
    interval_s=0.05,
    getppid=lambda: state["pid"],
    on_parent_lost=lost.set,
  )
  try:
    assert not lost.wait(0.3), "the watcher fired while the parent was alive"
    state["pid"] = 1  # reparented: the parent is gone
    assert lost.wait(2.0), "the watcher did not notice reparenting"
  finally:
    stop()


def _fake_command(*, enabled: bool, calls: list[str] | None = None) -> Any:
  """A multi-motion command stand-in, recording when it is re-seeded."""

  class _Command:
    reset_policy = SimpleNamespace(enabled=enabled)

    def __init__(self) -> None:
      self.state: torch.Tensor | None = None

    def set_reset_rng_state(self, state: torch.Tensor) -> None:
      self.state = state.clone()
      if calls is not None:
        calls.append("seed")

  return _Command()


def test_derive_reset_rng_is_a_pure_function_of_the_seed() -> None:
  """A derived reset stream is reproducible from the call that draws it.

  That is the whole reason a sharded standing run needs no per-shard RNG state
  in its checkpoint: iteration k derives the same resets on a resume as it drew
  the first time.
  """
  first = _fake_command(enabled=True)
  second = _fake_command(enabled=True)
  derive_reset_rng(first, 1234)
  derive_reset_rng(second, 1234)
  expected = torch.Generator(device="cpu").manual_seed(1234).get_state()
  assert first.state is not None
  assert torch.equal(first.state, expected)
  assert torch.equal(first.state, second.state)

  derive_reset_rng(second, 1235)
  assert not torch.equal(first.state, second.state)


def test_derive_reset_rng_leaves_a_disabled_policy_at_its_construction_seed() -> None:
  """A run that draws no standing decision is not force-seeded."""
  calls: list[str] = []
  command = _fake_command(enabled=False, calls=calls)
  derive_reset_rng(command, 7)
  assert calls == []
  assert command.state is None


def test_derive_reset_rng_refuses_an_enabled_policy_it_cannot_seed() -> None:
  """Resets that are neither derived nor recorded must not run silently."""

  class _Command:
    reset_policy = SimpleNamespace(enabled=True)

  with pytest.raises(WorkerError, match="set_reset_rng_state"):
    derive_reset_rng(_Command(), 7)


def test_worker_seeds_its_reset_rng_before_it_collects(tmp_path) -> None:
  """The derivation happens before the call whose resets it governs.

  Seeding after collection would be a silent no-op: the call would draw from
  whatever stream the shard happened to hold, which no checkpoint records.
  """
  order: list[str] = []
  command = _fake_command(enabled=True, calls=order)
  adapter = SimpleNamespace(
    env=SimpleNamespace(
      command_manager=SimpleNamespace(get_term=lambda name: command),
    )
  )
  batch = _batch(2, 0)

  class _Collector:
    def collect(self, config, *, reset: bool):
      order.append("collect")
      return SimpleNamespace(
        ticks=1,
        samples=2,
        teacher_steps=1,
        student_steps=1,
        disagreement_mean=0.0,
        diagnostics=(),
        motion_stats=(),
        boundaries=(),
        eligible_resets={},
        initialization_resets={},
        fresh_data=SimpleNamespace(batch=batch),
      )

  spec = _specs(tmp_path, devices=("cpu",))[0]
  worker = _Worker(spec)
  worker._adapter = adapter
  worker._collector = _Collector()
  worker._student = SimpleNamespace(
    load_state_dict=lambda state: None,
    reference_normalizer=SimpleNamespace(freeze=lambda: None),
    conditioning_normalizer=SimpleNamespace(freeze=lambda: None),
  )

  reply = worker.collect(
    CollectRequest(
      call_id=1,
      iteration=3,
      steps=1,
      teacher_probability=0.0,
      reset=True,
      rollout_latent="mean",
      weights={},
    )
  )

  assert isinstance(reply, CollectReply)
  assert order == ["seed", "collect"]
  derived = worker_generator_seed(spec.setup.base_seed, 3, spec.worker_index)
  assert command.state is not None
  assert torch.equal(
    command.state, torch.Generator(device="cpu").manual_seed(derived).get_state()
  )


def test_worker_derives_its_reset_rng_before_it_evaluates(
  tmp_path, monkeypatch
) -> None:
  """Evaluation draws standing resets too, from its own derived stream.

  A shard evaluates after collecting the same iteration, and evaluation resets
  draw standing/reference decisions from the same command generator.  Deriving
  evaluation's stream as well is what keeps an evaluated iteration reproducible
  after a resume; the separate offset is what stops the policy from being scored
  on the very resets it was just trained against.
  """
  order: list[str] = []
  command = _fake_command(enabled=True, calls=order)
  adapter = SimpleNamespace(
    env=SimpleNamespace(
      command_manager=SimpleNamespace(get_term=lambda name: command),
    ),
    bank=SimpleNamespace(),
  )
  spec = _specs(tmp_path, devices=("cpu",))[0]
  collection_seed = worker_generator_seed(spec.setup.base_seed, 4, spec.worker_index)

  def fake_evaluate(adapter_arg, bank, student, **kwargs):
    order.append("evaluate")
    assert kwargs["seed"] == collection_seed
    return "evaluated"

  monkeypatch.setattr(worker_module, "evaluate_distillation", fake_evaluate)

  worker = _Worker(spec)
  worker._adapter = adapter
  worker._student = SimpleNamespace(
    load_state_dict=lambda state: None,
    reference_normalizer=SimpleNamespace(freeze=lambda: None),
    conditioning_normalizer=SimpleNamespace(freeze=lambda: None),
  )
  reply = worker.evaluate(
    EvaluateRequest(
      call_id=1,
      iteration=4,
      steps=2,
      mode="student",
      rollout_latent="mean",
      weights={},
    )
  )

  assert isinstance(reply, EvaluateReply)
  assert order == ["seed", "evaluate"]
  derived = worker_evaluation_seed(spec.setup.base_seed, 4, spec.worker_index)
  assert derived != collection_seed
  assert command.state is not None
  assert torch.equal(
    command.state, torch.Generator(device="cpu").manual_seed(derived).get_state()
  )


def test_worker_derives_the_process_global_rng_for_each_call(tmp_path) -> None:
  """The environment's own sampling must be derived too, not carried.

  Reference resampling and the inherited command's integer draws use the
  process-global torch RNG.  A worker that reseeded only the streams it owns
  would carry that generator across calls, and a sharded checkpoint records no
  per-worker global state, so a resumed iteration would draw different
  environment randomness while every counter still looked right.
  """
  draws: list[float] = []

  class _Collector:
    def collect(self, config, *, reset: bool):
      draws.append(float(torch.rand(1)))
      return SimpleNamespace(
        ticks=1,
        samples=2,
        teacher_steps=1,
        student_steps=1,
        disagreement_mean=0.0,
        diagnostics=(),
        motion_stats=(),
        boundaries=(),
        eligible_resets={},
        initialization_resets={},
        fresh_data=SimpleNamespace(batch=_batch(2, 0)),
      )

  spec = _specs(tmp_path, devices=("cpu",))[0]
  worker = _Worker(spec)
  worker._adapter = SimpleNamespace(
    env=SimpleNamespace(
      command_manager=SimpleNamespace(get_term=lambda name: _fake_command(enabled=True))
    )
  )
  worker._collector = _Collector()
  worker._student = SimpleNamespace(
    load_state_dict=lambda state: None,
    reference_normalizer=SimpleNamespace(freeze=lambda: None),
    conditioning_normalizer=SimpleNamespace(freeze=lambda: None),
  )

  def collect(iteration: int) -> None:
    worker.collect(
      CollectRequest(
        call_id=1,
        iteration=iteration,
        steps=1,
        teacher_probability=0.0,
        reset=True,
        rollout_latent="mean",
        weights={},
      )
    )

  collect(4)
  collect(4)
  collect(5)

  # Same call seed, same first draw: the generator is re-derived, not carried.
  assert draws[0] == draws[1]
  # A different iteration derives a different stream.
  assert draws[0] != draws[2]
  expected = torch.rand(
    1,
    generator=torch.Generator(device="cpu").manual_seed(
      worker_generator_seed(spec.setup.base_seed, 4, spec.worker_index)
    ),
  )
  assert draws[0] == float(expected)
