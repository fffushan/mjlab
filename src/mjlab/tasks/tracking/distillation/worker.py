"""Sharded collection workers and the parent-side pool that drives them.

A worker owns one environment shard on its own device and answers requests: it
describes the environment it built, it collects a bounded number of ticks with
the weights the parent sends, and it evaluates its shard with those same
weights.  Workers never insert into a replay, never update normalizers, never
write checkpoints and never report progress: the parent owns every durable
state, and a worker's reply is the only thing that crosses the boundary.

Transport is deliberately boring.  Payloads are staged into host memory and
passed through ``multiprocessing`` queues, so CPU tests exercise the same code
path a GPU run uses and no new dependency (NCCL, CUDA IPC) is required.  The
plan records the measured trigger for replacing it with something faster.

Failure handling is the point of this module.  Every request carries a
monotonic ``call_id`` and a reply that does not match it is rejected instead of
being used; every wait is bounded and also watches the child process, so a dead
or hung worker aborts the whole pool rather than letting the parent block.  A
failed worker kills the pool: the parent then has no partially updated replay,
no torn payload, and no orphan process holding a GPU context.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as queue_module
import threading
import time
import traceback
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

import torch

from mjlab.tasks.tracking.distillation.adapter import (
  InitializationKind,
  MultiMotionDistillationAdapter,
)
from mjlab.tasks.tracking.distillation.cohort_contract import (
  reset_provenance_from_adapter,
)
from mjlab.tasks.tracking.distillation.cohort_setup import (
  CohortSetup,
  worker_evaluation_seed,
  worker_generator_seed,
)
from mjlab.tasks.tracking.distillation.collector import (
  CollectionConfig,
  CollectionResult,
  DAggerCollector,
  EvaluationMode,
  EvaluationResult,
  RolloutLatent,
  evaluate_distillation,
)
from mjlab.tasks.tracking.distillation.multi_motion import PhasePolicy
from mjlab.tasks.tracking.distillation.reset_policy import ResetPolicy
from mjlab.tasks.tracking.distillation.storage import LabeledReplayBatch
from mjlab.tasks.tracking.distillation.teachers import TeacherBank

DEFAULT_REQUEST_TIMEOUT_S = 900.0
"""Upper bound on one request before the pool aborts.

Bounded on purpose: a worker that stops answering must fail the iteration, not
silently stall a multi-hour run.  The bound is generous enough that a large
collection on a busy GPU cannot trip it.
"""

PARENT_WATCH_INTERVAL_S = 5.0
"""How often an idle worker checks that its parent still exists.

Short on purpose.  A ``SIGKILL``ed parent cannot run cleanup, so this interval
is the only thing standing between a killed run and a child that keeps a GPU
context for as long as its queue wait allows.
"""

_DEATH_POLL_INTERVAL_S = 1.0
"""How often a bounded wait re-checks that the child is still alive."""

DEFAULT_CLOSE_GRACE_S = 10.0
"""How long ``close`` waits for a worker to exit before terminating it.

A worker that is inside a long collection will not read a shutdown request, so
this is the difference between a prompt teardown and a stalled one; a caller
that would rather kill quickly (an operator's signal handler, a test) can lower
it."""


class WorkerError(RuntimeError):
  """A worker failed, timed out, or answered a request it should not have."""


def derive_reset_rng(command: Any, seed: int) -> None:
  """Re-seed one multi-motion command's standing-reset RNG from a call seed.

  The derived state is a pure function of the call seed, so a resumed iteration
  reproduces its standing-reset draws without any per-shard RNG state in the
  checkpoint.  A command whose reset policy is disabled draws no decision from
  this generator and is left at its construction seed, so a non-standing or
  evaluation-only configuration is never force-seeded.

  Raises:
    WorkerError: if the policy is enabled but the command cannot accept a
      derived state, because those resets would then be neither derived nor
      recorded anywhere.
  """
  policy = getattr(command, "reset_policy", None)
  if not getattr(policy, "enabled", False):
    return
  setter = getattr(command, "set_reset_rng_state", None)
  if not callable(setter):
    raise WorkerError(
      "the motion command enables a reset policy but exposes no "
      "set_reset_rng_state setter, so its standing resets cannot be derived"
    )
  setter(torch.Generator(device="cpu").manual_seed(int(seed)).get_state())


class WorkerPoolError(RuntimeError):
  """The pool is unusable: a worker failed, so no reply can be trusted."""


@dataclass(frozen=True, slots=True)
class WorkerSpec:
  """Everything one worker process needs to build and drive its shard.

  The cohort recipe travels by value (``setup``) rather than being
  reconstructed from a command line, so a worker cannot build a different
  environment than the parent planned.
  """

  worker_index: int
  device: str
  num_envs: int
  setup: CohortSetup


@dataclass(frozen=True, slots=True)
class WorkerEnvironmentDescription:
  """What a worker's live environment says about itself.

  The parent records reset provenance and the sampling mode from here instead
  of from an adapter of its own: in the sharded layout the parent owns no
  environment, and these fields still describe the run.
  """

  worker_index: int
  device: str
  num_envs: int
  schema: Any
  reset_policy_enabled: bool
  reset_provenance: Any | None
  sampling_mode: str
  library: Any = None
  """Reference library the shard built, for the parent's replay and identity.

  A sharded parent owns no environment, so the clip extents, the audited slot
  allocation and the motion-to-teacher codes it needs to construct the replay
  and the cohort identity come from a worker that did build one.
  """
  audit: Any = None
  """Live multi-motion audit the shard produced during construction."""
  motion_teacher_codes: Any = None
  """Per-motion teacher codes, the replay's routing table."""


@dataclass(frozen=True, slots=True)
class DescribeWorker:
  """Ask a worker to build its shard and describe the environment."""

  call_id: int


@dataclass(frozen=True, slots=True)
class ShutdownWorker:
  """Ask a worker to release its environment and exit."""

  call_id: int


@dataclass(frozen=True, slots=True)
class CollectRequest:
  """One bounded collection call, with the weights to collect under."""

  call_id: int
  iteration: int
  steps: int
  teacher_probability: float
  reset: bool
  rollout_latent: RolloutLatent
  weights: Mapping[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class EvaluateRequest:
  """One bounded evaluation call on this worker's own shard."""

  call_id: int
  iteration: int
  steps: int
  mode: EvaluationMode
  rollout_latent: RolloutLatent
  weights: Mapping[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class DescribeReply:
  call_id: int
  description: WorkerEnvironmentDescription


@dataclass(frozen=True, slots=True)
class CollectReply:
  """One worker's rows for one collection call.

  ``batch`` holds rows concatenated tick by tick, ``rows_per_tick`` says how to
  slice them back into ticks, and every tensor is a host tensor so the parent
  can merge before any device copy.  ``motion_frames`` carries the distinct
  reference frames this shard observed per motion, which is what lets the
  parent report the exact union of shard coverage instead of a lower bound.
  """

  call_id: int
  iteration: int
  ticks: int
  samples: int
  teacher_steps: int
  student_steps: int
  disagreement_mean: float
  diagnostics: tuple[str, ...]
  motion_stats: tuple[Any, ...]
  boundaries: tuple[Any, ...]
  rows_per_tick: int
  batch: LabeledReplayBatch
  eligible_resets: dict[int, int] = field(default_factory=dict)
  initialization_resets: dict[int, dict[str, int]] = field(default_factory=dict)
  motion_frames: tuple[tuple[int, tuple[int, ...]], ...] = ()


@dataclass(frozen=True, slots=True)
class EvaluateReply:
  call_id: int
  iteration: int
  result: EvaluationResult


@dataclass(frozen=True, slots=True)
class WorkerFailure:
  """A worker-side exception, reported with the request it belongs to."""

  call_id: int
  message: str
  traceback: str


WorkerReply = DescribeReply | CollectReply | EvaluateReply | WorkerFailure
WorkerRequest = DescribeWorker | CollectRequest | EvaluateRequest | ShutdownWorker


def _stage(tensor: torch.Tensor) -> torch.Tensor:
  """Return a host copy of ``tensor`` suitable for a queue.

  A CUDA tensor is copied into page-locked memory so the device-to-host copy can
  be asynchronous; that copy is the only device traffic this module performs.
  Waiting for it here is deliberate: a reply that has not finished copying is a
  reply the parent would race against.
  """
  detached = tensor.detach()
  if not detached.is_cuda:
    return detached.clone().contiguous()
  pinned = torch.empty_like(detached, device="cpu", pin_memory=True)
  pinned.copy_(detached, non_blocking=True)
  torch.cuda.current_stream(detached.device).synchronize()
  return pinned


def _unstage(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
  """Move one received host tensor onto ``device`` without touching its values."""
  if device.type == "cpu":
    return tensor.detach().clone()
  return tensor.detach().to(device, non_blocking=False, copy=True)


def _stage_weights(weights: Mapping[str, torch.Tensor]) -> dict[str, Any]:
  """Stage one model state dict for the queue in a stable key order.

  Non-tensor entries (a module's ``_extra_state``) travel unchanged rather than
  being dropped: they are part of the model's own state contract, so a shard
  that received only the tensors would hold different metadata than the parent.
  """
  return {
    name: _stage(value) if isinstance(value, torch.Tensor) else value
    for name, value in weights.items()
  }


def _stage_batch(batch: LabeledReplayBatch) -> LabeledReplayBatch:
  """Stage every tensor of one collected batch into host memory."""
  observations = batch.observations
  staged = replace(
    batch,
    observations=replace(
      observations,
      reference=_stage(observations.reference),
      conditioning=_stage(observations.conditioning),
    ),
    teacher_action=_stage(batch.teacher_action),
    motion_id=_stage(batch.motion_id),
    teacher_id=_stage(batch.teacher_id),
    reference_frame=_stage(batch.reference_frame),
    episode_id=_stage(batch.episode_id),
    collector_iteration=_stage(batch.collector_iteration),
  )
  if not batch.has_provenance:
    return staged
  assert batch.initialization_kind is not None
  assert batch.segment_initial_reference_frame is not None
  assert batch.segment_age is not None
  return replace(
    staged,
    initialization_kind=_stage(batch.initialization_kind),
    segment_initial_reference_frame=_stage(batch.segment_initial_reference_frame),
    segment_age=_stage(batch.segment_age),
  )


def _observed_frames(
  batch: LabeledReplayBatch,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
  """Return the distinct reference frames this shard observed, per motion.

  Read from the collected rows rather than from the collector's per-motion
  builder, so the parent can take an exact union of shard coverage.  The rows
  carry the same motion and frame the statistics attribute them to: both come
  from the pre-step snapshot.
  """
  result: list[tuple[int, tuple[int, ...]]] = []
  motion_id = batch.motion_id
  reference_frame = batch.reference_frame
  for motion in torch.unique(motion_id).tolist():
    frames = reference_frame[motion_id == motion]
    result.append(
      (int(motion), tuple(int(value) for value in torch.unique(frames).tolist()))
    )
  return tuple(result)


def freeze_student_normalizers(student: Any) -> None:
  """Freeze a shard's normalizers so collection cannot move them.

  The trainer owns the only legal normalizer update — once per fresh row — and
  freezes the same buffers before every collection.  A worker that skipped this
  would let its own forward pass drift the running statistics it was just
  handed, giving each shard a slightly different observation normalization than
  the parent holds, and violating the once-per-fresh-row invariant from a
  process the parent cannot see.  Because ``_frozen`` is a registered buffer,
  this must be re-applied after every weight load as well as at build time.
  """
  student.reference_normalizer.freeze()
  student.conditioning_normalizer.freeze()


def describe_environment(
  adapter: MultiMotionDistillationAdapter, spec: WorkerSpec
) -> WorkerEnvironmentDescription:
  """Read the live environment's provenance fields for the parent to record.

  The parent has no environment of its own in the sharded layout, so these
  values come from a worker.  They are read from the live command rather than
  from the configuration the parent supplied, which is what makes the record
  describe what was actually built.
  """
  command = getattr(getattr(adapter, "env", None), "command_manager", None)
  get_term = getattr(command, "get_term", None)
  motion = get_term("motion") if callable(get_term) else None
  policy = getattr(motion, "reset_policy", None)
  enabled = bool(getattr(policy, "enabled", False))
  provenance = reset_provenance_from_adapter(adapter) if enabled else None
  sampling_mode = motion.cfg.sampling_mode
  return WorkerEnvironmentDescription(
    worker_index=spec.worker_index,
    device=spec.device,
    num_envs=spec.num_envs,
    schema=adapter.schema,
    reset_policy_enabled=enabled,
    reset_provenance=provenance,
    sampling_mode=sampling_mode,
    # The library owns reference arrays on the shard's device, so the copy the
    # parent receives must already be host storage; the parent builds its
    # replay and cohort identity from it and must not move device tensors
    # across a process boundary.
    library=adapter.library.cpu_copy(),
    audit=adapter.audit,
    motion_teacher_codes=dict(adapter.motion_teacher_codes),
  )


class _Worker:
  """The child-side shard: one environment, one collector, no durable state."""

  def __init__(self, spec: WorkerSpec) -> None:
    self.spec = spec
    self._adapter: MultiMotionDistillationAdapter | None = None
    self._collector: DAggerCollector | None = None
    self._student = None

  def build(self) -> None:
    """Construct the shard lazily, so a construction failure is a reply.

    Building inside the first request (rather than before serving) is what lets
    a missing teacher artifact or a bad device reach the parent as an
    attributable error instead of an unexplained dead child.
    """
    if self._adapter is not None:
      return
    adapter = self.spec.setup.build_adapter(
      device=self.spec.device,
      num_envs=self.spec.num_envs,
      worker_index=self.spec.worker_index,
    )
    try:
      student = self.spec.setup.build_student(
        schema=adapter.schema,
        device=self.spec.device,
        worker_index=self.spec.worker_index,
      )
    except BaseException:
      adapter.close()
      raise
    # No replay: a worker must never write to the trainer's replay, whose only
    # writer is the parent.
    freeze_student_normalizers(student)
    self._student = student
    self._collector = DAggerCollector(adapter, adapter.bank, student, None)
    self._adapter = adapter

  @property
  def adapter(self) -> MultiMotionDistillationAdapter:
    assert self._adapter is not None
    return self._adapter

  @property
  def collector(self) -> DAggerCollector:
    assert self._collector is not None
    return self._collector

  def student(self):
    assert self._student is not None
    return self._student

  def apply_weights(self, weights: Mapping[str, torch.Tensor]) -> None:
    """Load the parent's student state onto this shard's device.

    The model must be exactly the one the trainer holds before collection: a
    worker that collected under its own initialization would be collecting for
    a different policy while every counter still looked right.
    """
    device = torch.device(self.spec.device)
    state = {
      name: _unstage(value, device) if isinstance(value, torch.Tensor) else value
      for name, value in weights.items()
    }
    self.student().load_state_dict(state)
    # ``_frozen`` is a registered buffer, so loading the parent's state can
    # overwrite this shard's freeze.  Re-applying it here is what makes the
    # worker's own invariant true regardless of what the parent happened to
    # send, instead of relying on the parent having frozen first.
    freeze_student_normalizers(self.student())

  def describe(self, request: DescribeWorker) -> DescribeReply:
    self.build()
    return DescribeReply(
      call_id=request.call_id,
      description=describe_environment(self.adapter, self.spec),
    )

  def seed_reset_rng(self, seed: int) -> None:
    """Derive this shard's standing-reset RNG for one collection call.

    The motion command owns a CPU generator that decides standing-versus-
    reference for every full-reset row, seeded from this shard's environment
    seed.  A shard re-derives it per call for the same reason its rollout stream
    is derived per call: a resumed iteration reproduces the resets it draws
    without per-shard RNG state in the checkpoint, and each shard's own
    environment seed never stands in for a shared record.  Re-seeding per call
    also means an evaluation or a bootstrap cannot shift what collection draws
    next.
    """
    derive_reset_rng(self.adapter.env.command_manager.get_term("motion"), seed)

  def collect(self, request: CollectRequest) -> CollectReply:
    self.build()
    self.apply_weights(request.weights)
    generator_seed = worker_generator_seed(
      self.spec.setup.base_seed, request.iteration, self.spec.worker_index
    )
    self.seed_reset_rng(generator_seed)
    result: CollectionResult = self.collector.collect(
      CollectionConfig(
        steps=request.steps,
        teacher_probability=request.teacher_probability,
        rollout_latent=request.rollout_latent,
        seed=generator_seed,
        collector_iteration=request.iteration,
        # A worker derives its randomness from the iteration instead of
        # carrying a persistent stream: that is what makes a resumed iteration
        # reproduce collection without per-worker RNG state in the checkpoint.
        rng_mode="per_call",
      ),
      reset=request.reset,
    )
    if result.fresh_data is None:
      raise WorkerError("collection returned no fresh data")
    batch = result.fresh_data.batch
    if result.ticks <= 0 or batch.batch_size % result.ticks != 0:
      raise WorkerError("collection returned an unsliceable fresh batch")
    return CollectReply(
      call_id=request.call_id,
      iteration=request.iteration,
      ticks=result.ticks,
      samples=result.samples,
      teacher_steps=result.teacher_steps,
      student_steps=result.student_steps,
      disagreement_mean=result.disagreement_mean,
      diagnostics=tuple(result.diagnostics),
      motion_stats=tuple(result.motion_stats),
      boundaries=tuple(result.boundaries),
      rows_per_tick=batch.batch_size // result.ticks,
      batch=_stage_batch(batch),
      eligible_resets=dict(result.eligible_resets),
      initialization_resets={
        motion: dict(counts) for motion, counts in result.initialization_resets.items()
      },
      motion_frames=_observed_frames(batch),
    )

  def evaluate(self, request: EvaluateRequest) -> EvaluateReply:
    self.build()
    self.apply_weights(request.weights)
    # Evaluation resets draw standing/reference decisions too, from the same
    # command generator.  Deriving its stream as well is what keeps an evaluated
    # iteration reproducible after a resume; the offset keeps it disjoint from the
    # collection of the same iteration, so a policy is never scored on the resets
    # it was just trained against.
    self.seed_reset_rng(
      worker_evaluation_seed(
        self.spec.setup.base_seed, request.iteration, self.spec.worker_index
      )
    )
    result = evaluate_distillation(
      self.adapter,
      self.adapter.bank,
      self.student(),
      mode=request.mode,
      steps=request.steps,
      rollout_latent=request.rollout_latent,
      seed=worker_generator_seed(
        self.spec.setup.base_seed, request.iteration, self.spec.worker_index
      ),
    )
    return EvaluateReply(
      call_id=request.call_id, iteration=request.iteration, result=result
    )

  def close(self) -> None:
    """Release the environment exactly once."""
    if self._adapter is not None:
      self._adapter.close()
      self._adapter = None
      self._collector = None
      self._student = None


def _exit_now() -> None:
  """Leave immediately, with no teardown, because the owner is already gone."""
  os._exit(0)


def watch_parent(
  parent_pid: int,
  *,
  interval_s: float = PARENT_WATCH_INTERVAL_S,
  getppid: Callable[[], int] = os.getppid,
  on_parent_lost: Callable[[], None] | None = None,
) -> Callable[[], None]:
  """Exit when this process is reparented, even in the middle of a request.

  A ``SIGKILL``ed parent cannot ask its children to stop, and a child that only
  checked between requests would keep a device busy for the rest of a long
  collection nobody will ever read.  The watcher therefore runs in its own
  thread and does not wait for the current request to finish.  Reparenting is
  the signal: when the parent dies, this process's parent id changes.

  Returns a callable that stops the watcher.
  """
  stop = threading.Event()

  def loop() -> None:
    while not stop.wait(interval_s):
      if getppid() != parent_pid:
        (on_parent_lost or _exit_now)()
        return

  threading.Thread(target=loop, name="distill-parent-watch", daemon=True).start()
  return stop.set


def worker_main(
  spec: WorkerSpec,
  commands: Any,
  replies: Any,
  *,
  parent_watch_interval_s: float = PARENT_WATCH_INTERVAL_S,
) -> None:
  """Child entry point: build lazily, answer requests, exit on shutdown.

  The loop wakes on ``parent_watch_interval_s`` rather than on a long request
  timeout, so a parent killed without running any cleanup is noticed within
  seconds.  Reparenting (``os.getppid()`` changing) is the signal, because a
  ``SIGKILL``ed parent cannot ask its children to stop.
  """
  parent_pid = os.getppid()
  shard = _Worker(spec)
  stop_watch = watch_parent(parent_pid, interval_s=parent_watch_interval_s)
  try:
    while True:
      try:
        request = commands.get(timeout=parent_watch_interval_s)
      except queue_module.Empty:
        if os.getppid() != parent_pid:
          return
        continue
      if os.getppid() != parent_pid:
        return
      if isinstance(request, ShutdownWorker):
        return
      try:
        if isinstance(request, DescribeWorker):
          reply: WorkerReply = shard.describe(request)
        elif isinstance(request, CollectRequest):
          reply = shard.collect(request)
        elif isinstance(request, EvaluateRequest):
          reply = shard.evaluate(request)
        else:
          raise WorkerError(f"unknown worker request {type(request).__name__}")
      except BaseException as exc:  # reported, never swallowed
        reply = WorkerFailure(
          call_id=getattr(request, "call_id", -1),
          message=f"{type(exc).__name__}: {exc}",
          traceback=traceback.format_exc(),
        )
      replies.put(reply)
  finally:
    stop_watch()
    shard.close()


class WorkerPool:
  """The parent-side owner of N collection workers.

  The pool is fail-closed in one direction only: any worker failure, timeout,
  or protocol mismatch makes the whole pool unusable.  Continuing with the
  surviving workers would silently change the data mix of an iteration, so the
  pool refuses instead and lets the caller decide.
  """

  def __init__(
    self,
    specs: Sequence[WorkerSpec],
    *,
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    close_grace_s: float = DEFAULT_CLOSE_GRACE_S,
    context: Any | None = None,
    target: Callable[..., None] = worker_main,
  ) -> None:
    if not specs:
      raise WorkerError("a worker pool needs at least one worker")
    indexed = sorted(specs, key=lambda spec: spec.worker_index)
    if [spec.worker_index for spec in indexed] != list(range(len(indexed))):
      raise WorkerError("worker indices must be 0..N-1 without gaps")
    if len(
      {spec.device for spec in indexed if not spec.device.startswith("cpu")}
    ) != len([spec for spec in indexed if not spec.device.startswith("cpu")]):
      raise WorkerError("two workers cannot share one device")
    self.specs = tuple(indexed)
    self.request_timeout_s = float(request_timeout_s)
    self.close_grace_s = float(close_grace_s)
    self._target = target
    self._context = mp.get_context("spawn") if context is None else context
    self._processes: list[Any] = []
    self._commands: list[Any] = []
    self._replies: list[Any] = []
    self._call_ids = [0] * len(self.specs)
    self._sent_at = [0.0] * len(self.specs)
    self._closed = False
    self._failure: str | None = None

  # -- lifecycle -----------------------------------------------------------

  def start(self) -> None:
    """Spawn every worker process."""
    if self._processes or self._closed:
      raise WorkerError("the pool was already started or closed")
    try:
      for spec in self.specs:
        commands = self._context.Queue()
        replies = self._context.Queue()
        process = self._context.Process(
          target=self._target,
          args=(spec, commands, replies),
          name=f"distill-worker-{spec.worker_index}",
        )
        process.daemon = False
        process.start()
        self._processes.append(process)
        self._commands.append(commands)
        self._replies.append(replies)
    except BaseException:
      self.close()
      raise

  def __enter__(self) -> WorkerPool:
    self.start()
    return self

  def __exit__(self, *_exc: object) -> None:
    self.close()

  def close(self) -> None:
    """Stop every worker, then release its queues.  Safe to call twice."""
    if self._closed:
      return
    self._closed = True
    for index, process in enumerate(self._processes):
      if process.is_alive():
        try:
          self._commands[index].put(ShutdownWorker(call_id=-1), timeout=1.0)
        except Exception:
          pass
    for process in self._processes:
      process.join(timeout=self.close_grace_s)
      if process.is_alive():
        process.terminate()
        process.join(timeout=self.close_grace_s)
      if process.is_alive():
        process.kill()
        process.join(timeout=self.close_grace_s)
    for channel in (*self._commands, *self._replies):
      try:
        channel.close()
        channel.cancel_join_thread()
      except Exception:
        pass
    self._processes = []
    self._commands = []
    self._replies = []

  def _abort(self, reason: str) -> WorkerPoolError:
    """Make the pool unusable and release its processes."""
    if self._failure is None:
      self._failure = reason
    self.close()
    return WorkerPoolError(
      f"worker pool aborted: {reason}; the iteration was not completed and no "
      "partial data was kept"
    )

  # -- requests ------------------------------------------------------------

  def _new_call_id(self, worker_index: int) -> int:
    self._call_ids[worker_index] += 1
    return self._call_ids[worker_index]

  def _require_usable(self) -> None:
    """Fail closed if the pool cannot be used for a new request."""
    if self._closed or self._failure is not None:
      raise WorkerPoolError(f"the pool is not usable ({self._failure or 'closed'})")
    if not self._processes:
      raise WorkerError(
        "the worker pool was not started; call start() or use it as a context "
        "manager before sending requests"
      )

  def _send(self, worker_index: int, request: WorkerRequest) -> None:
    """Enqueue one request without waiting for its reply."""
    self._require_usable()
    dispatched = time.monotonic()
    try:
      self._commands[worker_index].put(request, timeout=self.request_timeout_s)
    except Exception as exc:
      raise self._abort(
        f"worker {worker_index} could not accept {type(request).__name__}: {exc}"
      ) from exc
    # The deadline belongs to the dispatch, not to the moment the caller starts
    # waiting: a request sent alongside its neighbours must not collect a fresh
    # timeout budget because an earlier worker was slow to answer.
    self._sent_at[worker_index] = dispatched

  def _receive(self, worker_index: int, request: WorkerRequest) -> WorkerReply:
    """Wait for the reply to one already-sent request, or fail the pool.

    The deadline is measured from dispatch, so one worker's reply cannot hand
    another worker extra time, and a whole exchange cannot take N timeouts.
    """
    self._require_usable()
    process = self._processes[worker_index]
    deadline = self._sent_at[worker_index] + self.request_timeout_s
    while True:
      remaining = deadline - time.monotonic()
      if remaining <= 0:
        raise self._abort(
          f"worker {worker_index} did not answer {type(request).__name__} "
          f"within {self.request_timeout_s:.0f}s of dispatch"
        )
      try:
        reply = self._replies[worker_index].get(
          timeout=min(_DEATH_POLL_INTERVAL_S, remaining)
        )
        break
      except queue_module.Empty:
        if not process.is_alive():
          raise self._abort(
            f"worker {worker_index} exited with code {process.exitcode} while "
            f"answering {type(request).__name__}"
          ) from None
    if isinstance(reply, WorkerFailure):
      raise self._abort(
        f"worker {worker_index} failed on {type(request).__name__}: "
        f"{reply.message}\n{reply.traceback}"
      )
    expected_id = getattr(request, "call_id", None)
    if getattr(reply, "call_id", None) != expected_id:
      raise self._abort(
        f"worker {worker_index} answered call {getattr(reply, 'call_id', None)} "
        f"while {expected_id} was outstanding"
      )
    return reply

  def _exchange(self, requests: Sequence[WorkerRequest]) -> tuple[WorkerReply, ...]:
    """Send every request first, then collect the replies in worker order.

    Sending first is the entire point of sharded collection.  Asking worker 1
    only after worker 0's reply has arrived makes collection wall time the sum
    of the shards instead of the maximum, which looks identical in every
    counter and silently removes the parallelism the run exists for.
    """
    if len(requests) != len(self.specs):
      raise WorkerError(
        f"an exchange needs one request per worker: got {len(requests)} for "
        f"{len(self.specs)} workers"
      )
    for index, request in enumerate(requests):
      self._send(index, request)
    return tuple(
      self._receive(index, request) for index, request in enumerate(requests)
    )

  def _request(self, worker_index: int, request: WorkerRequest) -> WorkerReply:
    """Send one request and wait for its reply (no overlap with other workers)."""
    self._send(worker_index, request)
    return self._receive(worker_index, request)

  def describe(self) -> tuple[WorkerEnvironmentDescription, ...]:
    """Build every shard and return the environments they described.

    The builds are dispatched together: at eight shards, constructing them one
    at a time would multiply the run's startup by the worker count for no
    reason.
    """
    requests = [
      DescribeWorker(call_id=self._new_call_id(index))
      for index in range(len(self.specs))
    ]
    replies = self._exchange(requests)
    descriptions = []
    for index, (spec, reply) in enumerate(zip(self.specs, replies, strict=True)):
      assert isinstance(reply, DescribeReply)
      if reply.description.num_envs != spec.num_envs:
        raise self._abort(
          f"worker {index} built {reply.description.num_envs} environments, "
          f"the plan said {spec.num_envs}"
        )
      descriptions.append(reply.description)
    return tuple(descriptions)

  def collect(
    self,
    *,
    iteration: int,
    steps: int,
    teacher_probability: float,
    reset: bool,
    rollout_latent: RolloutLatent,
    weights: Mapping[str, torch.Tensor],
  ) -> tuple[CollectReply, ...]:
    """Collect one iteration's rows from every worker, in worker order.

    Every worker is asked before any reply is consumed, so the shards collect
    concurrently and a slow shard cannot hold up its neighbour's start.
    """
    staged = _stage_weights(weights)
    requests = [
      CollectRequest(
        call_id=self._new_call_id(index),
        iteration=iteration,
        steps=steps,
        teacher_probability=teacher_probability,
        reset=reset,
        rollout_latent=rollout_latent,
        weights=staged,
      )
      for index in range(len(self.specs))
    ]
    replies = list(self._exchange(requests))
    if any(
      reply.rows_per_tick != spec.num_envs
      for reply, spec in zip(replies, self.specs, strict=True)
    ):
      raise self._abort(
        "workers returned row blocks that do not match their shard size"
      )
    ticks = {reply.ticks for reply in replies}
    if len(ticks) != 1:
      raise self._abort(f"workers collected different tick counts: {sorted(ticks)}")
    assert ticks.pop() >= 0
    return tuple(replies)

  def evaluate(
    self,
    *,
    iteration: int,
    mode: EvaluationMode,
    steps: int,
    rollout_latent: RolloutLatent,
    weights: Mapping[str, torch.Tensor],
  ) -> tuple[EvaluateReply, ...]:
    """Evaluate every shard with the same policy state, concurrently."""
    staged = _stage_weights(weights)
    requests = [
      EvaluateRequest(
        call_id=self._new_call_id(index),
        iteration=iteration,
        steps=steps,
        mode=mode,
        rollout_latent=rollout_latent,
        weights=staged,
      )
      for index in range(len(self.specs))
    ]
    replies = self._exchange(requests)
    for reply in replies:
      assert isinstance(reply, EvaluateReply)
    return tuple(replies)


def worker_specs(
  setup: CohortSetup,
  worker_devices: Sequence[str],
  *,
  num_envs: int,
) -> tuple[WorkerSpec, ...]:
  """Build the per-worker specs for one sharded run.

  The environment count is split evenly, which the CLI validates before any
  environment exists; this function re-checks it so a caller that constructs
  specs directly cannot silently give one shard fewer environments.
  """
  devices = tuple(worker_devices)
  if not devices:
    raise WorkerError("a sharded run needs at least one worker device")
  if num_envs % len(devices) != 0:
    raise WorkerError(
      f"{num_envs} environments do not split evenly over {len(devices)} workers"
    )
  per_worker = num_envs // len(devices)
  if per_worker < 1:
    raise WorkerError("each worker needs at least one environment")
  return tuple(
    WorkerSpec(
      worker_index=index,
      device=device,
      num_envs=per_worker,
      setup=setup,
    )
    for index, device in enumerate(devices)
  )


__all__ = [
  "CollectReply",
  "CollectRequest",
  "DEFAULT_REQUEST_TIMEOUT_S",
  "PARENT_WATCH_INTERVAL_S",
  "DescribeReply",
  "DescribeWorker",
  "EvaluateReply",
  "EvaluateRequest",
  "InitializationKind",
  "MultiMotionDistillationAdapter",
  "PhasePolicy",
  "ResetPolicy",
  "ShutdownWorker",
  "TeacherBank",
  "WorkerEnvironmentDescription",
  "WorkerError",
  "WorkerFailure",
  "WorkerPool",
  "WorkerPoolError",
  "WorkerSpec",
  "describe_environment",
  "freeze_student_normalizers",
  "worker_main",
  "worker_specs",
  "watch_parent",
]
