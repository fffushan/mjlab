"""Focused CPU tests for the asynchronous D5 latent runtime facade."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch
from tracking_diffusion_fixtures import make_state

from mjlab.tasks.tracking.diffusion.async_latent_runtime import (
  AsyncLatentRuntime,
  RuntimeStatus,
  RuntimeStep,
)
from mjlab.tasks.tracking.diffusion.handover import ReferenceFreeContinuation
from mjlab.tasks.tracking.diffusion.latent_planning import (
  LATENT_PLANNING_SCHEMA,
  RAW_LATENT_REPRESENTATION,
  LatentPlanner,
  LatentPlanningPacket,
  build_planning_packet,
)
from mjlab.tasks.tracking.diffusion.planner_executor import BoundedPlannerExecutor
from mjlab.tasks.tracking.diffusion.projection import (
  FeatureStats,
  ProjectionBundle,
  fit_feature_stats,
)
from mjlab.tasks.tracking.diffusion.schedule import DiffusionSchedule, InferenceGrid
from mjlab.tasks.tracking.diffusion.scheduler import (
  Plan,
  PlannerRequest,
  PlannerResult,
)


def _stats(width: int, *, seed: int) -> FeatureStats:
  rng = np.random.default_rng(seed)
  return fit_feature_stats([rng.normal(size=(64, width)) + 3.0], std_floor=1e-6)


def _projection() -> ProjectionBundle:
  return ProjectionBundle.create(_stats(135, seed=1), _stats(32, seed=2))


class _Clock:
  def __init__(self, value: float = 0.0) -> None:
    self.value = value

  def __call__(self) -> float:
    return self.value


class _RealDenoiser:
  def __call__(self, sample: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
    del step_ids
    return 0.2 * sample


class _Decoder:
  def __init__(self, *, fail: bool = False) -> None:
    self.fail = fail
    self.latents: list[np.ndarray] = []
    self.conditionings: list[np.ndarray] = []

  def __call__(self, latent: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
    self.latents.append(latent[0].detach().cpu().numpy().copy())
    self.conditionings.append(conditioning[0].detach().cpu().numpy().copy())
    if self.fail:
      return torch.full((1, 31), float("nan"))
    return latent[:, :31] + 0.001 * conditioning[:, :31]


def _plan_for_request(
  request: PlannerRequest,
  *,
  source_timestamp: float | None = None,
  source_frame: int | None = None,
  source_tick: int | None = None,
  action_width: int = 32,
) -> Plan:
  packet = LatentPlanningPacket.from_request(request)
  timestamp = packet.source_timestamp if source_timestamp is None else source_timestamp
  frame = packet.source_frame if source_frame is None else source_frame
  start_tick = packet.source_tick if source_tick is None else source_tick
  actions = np.stack(
    [np.full(action_width, start_tick + row, dtype=np.float64) for row in range(33)]
  )
  task_revision = "none" if packet.task is None else packet.task.revision
  task_mode = "unguided" if packet.task is None else packet.task.mode
  metadata = {
    "schema": LATENT_PLANNING_SCHEMA,
    "contract": packet.contract_identity,
    "representation": RAW_LATENT_REPRESENTATION,
    "kind": "latent",
    "semantic": "raw",
    "latent_dimension": 32,
    "latent_space": "raw",
    "source_tick": start_tick,
    "valid_from_tick": start_tick,
    "valid_until_tick": start_tick + 32,
    "source_timestamp": timestamp,
    "source_frame": frame,
    "root_frame": packet.root_frame,
    "period_seconds": packet.period_seconds,
    "vae_identity": packet.vae_identity,
    "projection_identity": packet.projection_identity,
    "model_identity": packet.model_identity,
    "task_revision": task_revision,
    "task_mode": task_mode,
  }
  return Plan(
    plan_sequence_id=request.request_sequence_id,
    source_timestamp=timestamp,
    valid_from_tick=start_tick,
    actions=actions,
    schema_metadata=metadata,
  )


class _ImmediateExecutor:
  def __init__(self, callback: Callable[[PlannerRequest], Plan]) -> None:
    self.callback = callback
    self.pending: PlannerRequest | None = None
    self.closed = False

  def submit(self, request: PlannerRequest) -> None:
    if self.pending is not None:
      raise RuntimeError("executor already has a request")
    self.pending = request

  def poll(self) -> tuple[PlannerResult, ...]:
    if self.pending is None:
      return ()
    request = self.pending
    self.pending = None
    return (
      PlannerResult(
        request_sequence_id=request.request_sequence_id,
        policy_generation=request.policy_generation,
        plan=self.callback(request),
      ),
    )

  def shutdown(self) -> None:
    self.closed = True
    self.pending = None


class _DelayedExecutor(_ImmediateExecutor):
  def __init__(self, callback: Callable[[PlannerRequest], Plan]) -> None:
    super().__init__(callback)
    self._delay_once = True

  def poll(self) -> tuple[PlannerResult, ...]:
    if self.pending is None:
      return ()
    if self._delay_once:
      self._delay_once = False
      return ()
    return super().poll()


class _FailureExecutor(_ImmediateExecutor):
  def poll(self) -> tuple[PlannerResult, ...]:
    if self.pending is None:
      return ()
    request = self.pending
    self.pending = None
    return (
      PlannerResult(
        request_sequence_id=request.request_sequence_id,
        policy_generation=request.policy_generation,
        error=RuntimeError("synthetic worker failure"),
      ),
    )


class _NoInspectionExecutor:
  def submit(self, request: PlannerRequest) -> None:
    del request

  def poll(self) -> tuple[PlannerResult, ...]:
    return ()

  def shutdown(self) -> None:
    return None


class _AliveOnlyExecutor(_NoInspectionExecutor):
  worker_alive = False


class _ContradictoryInspectionExecutor(_NoInspectionExecutor):
  closed = True
  worker_alive = True


class _ErrorInspectionExecutor(_NoInspectionExecutor):
  @property
  def closed(self) -> bool:
    raise RuntimeError("inspection unavailable")

  @property
  def worker_alive(self) -> bool:
    return False


def _worker_factory():
  return _worker_callback


def _worker_callback(request: PlannerRequest) -> Plan:
  return _plan_for_request(request)


def _real_planner(projection: ProjectionBundle) -> LatentPlanner:
  schedule = DiffusionSchedule.build()
  return LatentPlanner(
    denoiser=_RealDenoiser(),
    schedule=schedule,
    grid=InferenceGrid.uniform(schedule),
    projection=projection,
    vae_identity="vae-test",
    model_identity="model-test",
  )


def _real_worker_factory():
  return _real_planner(_projection()).process


def _runtime(
  executor: Any,
  clock: _Clock,
  *,
  decoder: _Decoder | None = None,
  generator: torch.Generator | None = None,
  max_source_age_seconds: float = 0.5,
  max_future_skew_seconds: float = 0.0,
  expected_source_frame: int | None = None,
) -> AsyncLatentRuntime:
  random = generator or torch.Generator(device="cpu").manual_seed(9)
  return AsyncLatentRuntime(
    executor,
    projection=_projection(),
    decode=decoder or _Decoder(),
    generator=random,
    clock=clock,
    vae_identity="vae-test",
    model_identity="model-test",
    max_source_age_seconds=max_source_age_seconds,
    max_future_skew_seconds=max_future_skew_seconds,
    expected_source_frame=expected_source_frame,
  )


def _bootstrap(runtime: AsyncLatentRuntime) -> None:
  for index in range(8):
    assert runtime.acknowledge_bootstrap(
      make_state(index), np.full(32, 0.05 * index), success=True
    )
  assert runtime.history_ready


def test_bootstrap_ack_and_failure_do_not_fabricate_history() -> None:
  clock = _Clock()
  runtime = _runtime(_ImmediateExecutor(_plan_for_request), clock)
  assert not runtime.acknowledge_bootstrap(make_state(0), np.zeros(32), success=False)
  assert runtime.history_size == 0
  waiting = runtime.step(make_state(8), np.zeros(99), tick=8)
  assert waiting.status is RuntimeStatus.BOOTSTRAP_REQUIRED
  _bootstrap(runtime)
  action = runtime.step(make_state(8), np.zeros(99), tick=8)
  assert action.status is RuntimeStatus.ACTION
  assert runtime.history_size == 8
  assert not runtime.acknowledge(action, success=False)
  assert runtime.history_size == 8
  runtime.release()


def test_late_plan_selects_execution_row_and_decodes_current_state() -> None:
  clock = _Clock(10.0)
  decoder = _Decoder()
  runtime = _runtime(
    _DelayedExecutor(_plan_for_request),
    clock,
    decoder=decoder,
    expected_source_frame=5,
  )
  _bootstrap(runtime)
  first = runtime.step(make_state(8), np.full(99, 1.0), tick=10, source_frame=5)
  assert first.status is RuntimeStatus.WAITING_FOR_PLAN
  second_state = make_state(9)
  second = runtime.step(second_state, np.full(99, 2.0), tick=11, source_frame=5)
  assert second.status is RuntimeStatus.ACTION
  assert second.executed_latent is not None
  np.testing.assert_array_equal(second.executed_latent, np.full(32, 11.0))
  assert len(decoder.conditionings) == 1
  np.testing.assert_array_equal(decoder.conditionings[0], np.full(99, 2.0))
  assert second.provenance is not None
  assert second.provenance.source_tick == 10
  assert second.provenance.execution_tick == 11
  assert runtime.acknowledge(second, success=True)
  assert runtime.history.pairs[-1].state.root_position[0] == pytest.approx(
    second_state.root_position[0]
  )
  runtime.release()


def test_d5_plan_origin_binding_rejects_self_consistent_spoof() -> None:
  decoder = _Decoder()
  runtime = _runtime(
    _DelayedExecutor(
      lambda request: _plan_for_request(request, source_tick=10, source_timestamp=0.25)
    ),
    _Clock(0.0),
    decoder=decoder,
  )
  _bootstrap(runtime)
  first = runtime.step(make_state(8), np.zeros(99), tick=0)
  assert first.status is RuntimeStatus.WAITING_FOR_PLAN
  spoofed = runtime.step(make_state(9), np.zeros(99), tick=1)
  assert spoofed.status is RuntimeStatus.INVALID_PLAN
  assert "request" in spoofed.detail
  assert not decoder.latents
  assert runtime.history_size == 8
  assert runtime.scheduler.accepted_plan is None
  assert runtime.scheduler.accepted_request_provenance is None
  runtime.release()


def test_task_change_invalidates_old_plan_without_clearing_history() -> None:
  clock = _Clock()
  runtime = _runtime(_ImmediateExecutor(_plan_for_request), clock)
  _bootstrap(runtime)
  first = runtime.step(
    make_state(8), np.zeros(99), tick=8, task={"mode": "unguided", "revision": "task-a"}
  )
  assert first.status is RuntimeStatus.ACTION
  assert runtime.acknowledge(first, success=True)
  assert runtime.history_size == 8
  changed = runtime.step(
    make_state(9), np.zeros(99), tick=9, task={"mode": "unguided", "revision": "task-b"}
  )
  assert changed.status is RuntimeStatus.ACTION
  assert changed.provenance is not None
  assert changed.provenance.task_revision == "task-b"
  assert runtime.history_size == 8
  runtime.release()


def test_source_age_future_and_frame_guards_fail_closed() -> None:
  clock = _Clock(0.0)
  stale_runtime = _runtime(
    _DelayedExecutor(_plan_for_request), clock, max_source_age_seconds=0.5
  )
  _bootstrap(stale_runtime)
  assert (
    stale_runtime.step(make_state(8), np.zeros(99), tick=0).status
    is RuntimeStatus.WAITING_FOR_PLAN
  )
  clock.value = 1.0
  stale = stale_runtime.step(make_state(9), np.zeros(99), tick=1)
  assert stale.status is RuntimeStatus.SOURCE_TOO_OLD
  stale_runtime.release()

  future_clock = _Clock(2.0)
  future_runtime = _runtime(
    _DelayedExecutor(_plan_for_request),
    future_clock,
    max_future_skew_seconds=0.0,
  )
  _bootstrap(future_runtime)
  first_future = future_runtime.step(make_state(8), np.zeros(99), tick=0)
  assert first_future.status is RuntimeStatus.WAITING_FOR_PLAN
  future_clock.value = 1.0
  future = future_runtime.step(make_state(9), np.zeros(99), tick=1)
  assert future.status is RuntimeStatus.SOURCE_IN_FUTURE
  future_runtime.release()

  wrong_frame_runtime = _runtime(
    _ImmediateExecutor(lambda request: _plan_for_request(request, source_frame=8)),
    _Clock(),
    expected_source_frame=7,
  )
  _bootstrap(wrong_frame_runtime)
  wrong_frame = wrong_frame_runtime.step(
    make_state(8), np.zeros(99), tick=0, source_frame=7
  )
  assert wrong_frame.status is RuntimeStatus.INVALID_PLAN
  wrong_frame_runtime.release()


def test_decoder_failure_and_worker_failure_have_no_action_or_history() -> None:
  decoder_runtime = _runtime(
    _ImmediateExecutor(_plan_for_request), _Clock(), decoder=_Decoder(fail=True)
  )
  _bootstrap(decoder_runtime)
  decoded = decoder_runtime.step(make_state(8), np.zeros(99), tick=0)
  assert decoded.status is RuntimeStatus.DECODE_FAILURE
  assert decoded.action is None
  assert decoder_runtime.history_size == 8
  decoder_runtime.release()

  worker_runtime = _runtime(_FailureExecutor(_plan_for_request), _Clock())
  _bootstrap(worker_runtime)
  failed = worker_runtime.step(make_state(8), np.zeros(99), tick=0)
  assert failed.status is RuntimeStatus.PLANNER_FAILURE
  assert failed.action is None
  assert worker_runtime.history_size == 8
  worker_runtime.release()


def test_malformed_plan_is_rejected_before_decoder() -> None:
  decoder = _Decoder()
  runtime = _runtime(
    _ImmediateExecutor(lambda request: _plan_for_request(request, action_width=31)),
    _Clock(),
    decoder=decoder,
  )
  _bootstrap(runtime)
  result = runtime.step(make_state(8), np.zeros(99), tick=0)
  assert result.status is RuntimeStatus.INVALID_PLAN
  assert not decoder.latents
  runtime.release()


def test_d3_synchronous_fixed_noise_parity_uses_current_latent_only() -> None:
  projection = _projection()
  schedule = DiffusionSchedule.build()
  grid = InferenceGrid.uniform(schedule)

  class _Denoiser:
    def __call__(self, sample: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
      del step_ids
      return 0.2 * sample

  denoiser = _Denoiser()
  planner = LatentPlanner(
    denoiser=denoiser,
    schedule=schedule,
    grid=grid,
    projection=projection,
    vae_identity="vae-test",
    model_identity="model-test",
  )
  clock = _Clock(2.0)
  runtime = AsyncLatentRuntime(
    _ImmediateExecutor(planner.process),
    projection=projection,
    decode=_Decoder(),
    generator=torch.Generator(device="cpu").manual_seed(31),
    clock=clock,
    vae_identity="vae-test",
    model_identity="model-test",
  )
  synchronous = ReferenceFreeContinuation(
    denoiser=denoiser,
    decode=_Decoder(),
    schedule=schedule,
    grid=grid,
    projection=projection,
    generator=torch.Generator(device="cpu").manual_seed(31),
  )
  for index in range(8):
    state = make_state(index)
    latent = np.full(32, 0.05 * index)
    assert runtime.acknowledge_bootstrap(state, latent, success=True)
    synchronous.record_executed(state, latent)
  conditioning = np.linspace(-0.5, 0.5, 99)
  runtime_step = runtime.step(make_state(8), conditioning, tick=8)
  synchronous_step = synchronous.step(make_state(8), conditioning, tick=8)
  assert runtime_step.status is RuntimeStatus.ACTION
  assert runtime_step.executed_latent is not None
  assert runtime_step.action is not None
  np.testing.assert_array_equal(
    runtime_step.executed_latent, synchronous_step.executed_latent
  )
  np.testing.assert_array_equal(runtime_step.action, synchronous_step.action)
  assert runtime.acknowledge(runtime_step, success=True)
  runtime.release()


def test_real_cpu_worker_round_trip_uses_assembled_facade() -> None:
  executor = BoundedPlannerExecutor(_worker_factory)
  runtime = _runtime(executor, _Clock())
  _bootstrap(runtime)
  first = runtime.step(make_state(8), np.full(99, 0.25), tick=0)
  assert first.status is RuntimeStatus.WAITING_FOR_PLAN
  receiver = executor._result_receiver
  assert receiver is not None
  assert receiver.poll(3.0)
  second = runtime.step(make_state(9), np.full(99, 0.75), tick=1)
  assert second.status is RuntimeStatus.ACTION
  assert second.executed_latent is not None
  np.testing.assert_array_equal(second.executed_latent, np.ones(32))
  assert runtime.acknowledge(second, success=True)
  runtime.release()
  assert executor.closed
  assert not executor.worker_alive
  assert runtime.cleanup_confirmed


def test_real_cpu_worker_round_trip_uses_actual_latent_planner() -> None:
  projection = _projection()
  generator = torch.Generator(device="cpu").manual_seed(41)
  expected_generator = torch.Generator(device="cpu").manual_seed(41)
  decoder = _Decoder()
  executor = BoundedPlannerExecutor(_real_worker_factory)
  runtime = AsyncLatentRuntime(
    executor,
    projection=projection,
    decode=decoder,
    generator=generator,
    clock=_Clock(2.0),
    vae_identity="vae-test",
    model_identity="model-test",
  )
  _bootstrap(runtime)
  current = make_state(8)
  first = runtime.step(current, np.full(99, 0.25), tick=20, source_frame=3)
  assert first.status is RuntimeStatus.WAITING_FOR_PLAN
  expected_packet = build_planning_packet(
    runtime.history,
    current,
    projection=projection,
    source_timestamp=2.0,
    source_tick=20,
    source_frame=3,
    task={"mode": "unguided", "revision": "none"},
    task_revision="none",
    generator=expected_generator,
    vae_identity="vae-test",
    model_identity="model-test",
  )
  expected = _real_planner(projection)(expected_packet)
  receiver = executor._result_receiver
  assert receiver is not None
  assert receiver.poll(5.0)
  current_next = make_state(9)
  second = runtime.step(current_next, np.full(99, 0.75), tick=21, source_frame=3)
  assert second.status is RuntimeStatus.ACTION
  assert second.executed_latent is not None
  np.testing.assert_array_equal(second.executed_latent, expected.raw_latents[1])
  assert decoder.conditionings
  np.testing.assert_array_equal(decoder.conditionings[-1], np.full(99, 0.75))
  assert runtime.acknowledge(second, success=True)
  assert runtime.history.pairs[-1].state.root_position[0] == pytest.approx(
    current_next.root_position[0]
  )
  runtime.release()
  assert executor.closed
  assert not executor.worker_alive
  assert runtime.cleanup_confirmed


def test_reset_switch_and_release_clear_history_and_cleanup_exactly() -> None:
  executor = BoundedPlannerExecutor(_worker_factory)
  runtime = _runtime(executor, _Clock())
  _bootstrap(runtime)
  action = runtime.step(make_state(8), np.zeros(99), tick=0)
  assert action.status is RuntimeStatus.WAITING_FOR_PLAN
  assert runtime.reset() is RuntimeStatus.RESET
  assert runtime.history_size == 0
  assert (
    runtime.step(make_state(8), np.zeros(99), tick=0).status
    is RuntimeStatus.BOOTSTRAP_REQUIRED
  )
  _bootstrap(runtime)
  assert runtime.switch_policy("policy-next") is RuntimeStatus.RESET
  assert runtime.history_size == 0
  assert runtime.release() is RuntimeStatus.RELEASED
  assert runtime.release() is RuntimeStatus.RELEASED
  assert executor.closed
  assert not executor.worker_alive
  closed = runtime.step(make_state(8), np.zeros(99), tick=0)
  assert closed.status is RuntimeStatus.RELEASED
  assert closed.action is None


def test_execution_id_is_strict_at_construction_and_ack_boundary() -> None:
  runtime = _runtime(_ImmediateExecutor(_plan_for_request), _Clock())
  _bootstrap(runtime)
  step = runtime.step(make_state(8), np.zeros(99), tick=0)
  assert step.status is RuntimeStatus.ACTION
  assert step.execution_id is not None
  for invalid in (True, 1.0, np.float64(1.0)):
    with pytest.raises(ValueError, match="execution_id"):
      replace(step, execution_id=invalid)
  forged = object.__new__(RuntimeStep)
  object.__setattr__(forged, "status", RuntimeStatus.ACTION)
  object.__setattr__(forged, "execution_id", True)
  with pytest.raises(ValueError, match="execution_id"):
    runtime.acknowledge(forged, success=True)
  assert runtime.pending_execution_id == step.execution_id
  assert runtime.acknowledge(step, success=True)
  runtime.release()


def test_cleanup_confirmed_requires_both_exact_inspection_properties() -> None:
  executors = (
    _NoInspectionExecutor(),
    _ImmediateExecutor(_plan_for_request),
    _AliveOnlyExecutor(),
    _ContradictoryInspectionExecutor(),
    _ErrorInspectionExecutor(),
  )
  for executor in executors:
    runtime = _runtime(executor, _Clock())
    runtime.release()
    assert runtime.cleanup_confirmed is False


def test_acknowledgement_token_is_required_for_history_pairing() -> None:
  runtime = _runtime(_ImmediateExecutor(_plan_for_request), _Clock())
  _bootstrap(runtime)
  step = runtime.step(make_state(20), np.full(99, 0.4), tick=0)
  assert step.status is RuntimeStatus.ACTION
  waiting = runtime.step(make_state(21), np.full(99, 0.4), tick=1)
  assert waiting.status is RuntimeStatus.AWAITING_EXECUTION_ACK
  with pytest.raises(ValueError, match="execution acknowledgement"):
    runtime.acknowledge(
      RuntimeStep(
        RuntimeStatus.ACTION, 0, step.action, step.executed_latent, step.provenance, 99
      ),
      success=True,
    )
  assert runtime.history.pairs[-1].state.root_position[0] == pytest.approx(
    make_state(7).root_position[0]
  )
  assert runtime.acknowledge(step, success=True)
  assert runtime.history.pairs[-1].state.root_position[0] == pytest.approx(
    make_state(20).root_position[0]
  )
  runtime.release()
