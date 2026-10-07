"""Focused CPU tests for the D5 controller lifecycle owner."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch
from tracking_diffusion_fixtures import make_state

from mjlab.tasks.tracking.diffusion.async_latent_runtime import (
  AsyncLatentRuntime,
  ExecutionProvenance,
  RuntimeStatus,
  RuntimeStep,
  WarmupReport,
)
from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT
from mjlab.tasks.tracking.diffusion.controller import (
  ControllerError,
  ControllerObservation,
  ControllerState,
  D5Controller,
)
from mjlab.tasks.tracking.diffusion.latent_planning import (
  LATENT_PLANNING_SCHEMA,
  RAW_LATENT_REPRESENTATION,
  ROOT_FRAME,
  LatentPlanningPacket,
)
from mjlab.tasks.tracking.diffusion.projection import (
  FeatureStats,
  ProjectionBundle,
  fit_feature_stats,
)
from mjlab.tasks.tracking.diffusion.scheduler import (
  Plan,
  PlannerRequest,
  PlannerResult,
)


class _Clock:
  def __init__(self, value: float = 0.0) -> None:
    self.value = value

  def __call__(self) -> float:
    return self.value


def _observation(
  *,
  sample_id: int = 0,
  tick: int = 0,
  timestamp: float = 0.0,
  source_frame: int = 3,
  segment_id: str = "segment-a",
  state_index: int = 0,
) -> ControllerObservation:
  return ControllerObservation(
    make_state(state_index),
    np.full(99, float(sample_id)),
    tick,
    sample_id,
    source_frame,
    segment_id,
    timestamp,
  )


class _FakeRuntime:
  def __init__(self, clock: Callable[[], float]) -> None:
    self.clock = clock
    self.history_ready = False
    self.history_size = 0
    self.closed = False
    self.worker_alive = True
    self.bootstrap_calls: list[bool] = []
    self.warmup_calls: list[tuple[int, int]] = []
    self.step_calls: list[tuple[int, int]] = []
    self.ack_calls: list[tuple[int, bool]] = []
    self.pending: RuntimeStep | None = None
    self.next_execution_id = 1
    self.on_step: Callable[[], None] | None = None

  @property
  def cleanup_confirmed(self) -> bool:
    return self.closed is True and self.worker_alive is False

  def acknowledge_bootstrap(self, state: Any, latent: Any, *, success: bool) -> bool:
    del state, latent
    self.bootstrap_calls.append(success)
    if success:
      self.history_size += 1
      self.history_ready = self.history_size >= 8
    return success

  def warmup(
    self,
    state: Any,
    conditioning: Any,
    *,
    tick: int,
    source_frame: int,
    task: Any,
    task_revision: str | None,
    decoder_calls: int,
    timeout_seconds: float,
  ) -> WarmupReport:
    del state, conditioning, task, task_revision, decoder_calls, timeout_seconds
    self.warmup_calls.append((tick, source_frame))
    return WarmupReport(1, 1, 1, 0.0)

  def step(
    self,
    state: Any,
    conditioning: Any,
    *,
    tick: int,
    source_frame: int,
    task: Any,
    task_revision: str | None,
  ) -> RuntimeStep:
    del state, conditioning, source_frame, task
    if self.on_step is not None:
      self.on_step()
    self.step_calls.append((tick, int(task_revision is not None)))
    provenance = ExecutionProvenance(
      plan_sequence_id=self.next_execution_id,
      source_timestamp=0.0,
      source_tick=tick,
      execution_tick=tick,
      valid_until_tick=tick + 32,
      source_frame=3,
      contract_identity=DEFAULT_CONTRACT.identity_hash(),
      vae_identity="fake-vae",
      projection_identity="fake-projection",
      model_identity="fake-model",
      task_revision=task_revision or "none",
      task_mode="unguided",
    )
    result = RuntimeStep(
      RuntimeStatus.ACTION,
      tick,
      np.arange(31, dtype=np.float64),
      np.zeros(32),
      provenance,
      self.next_execution_id,
    )
    self.next_execution_id += 1
    self.pending = result
    return result

  def acknowledge(self, step: RuntimeStep, *, success: bool) -> bool:
    assert step is self.pending
    self.ack_calls.append((step.execution_id or 0, success))
    self.pending = None
    if success:
      self.history_size += 1
    return success

  def reset(self) -> RuntimeStatus:
    self.pending = None
    self.history_size = 0
    self.history_ready = False
    return RuntimeStatus.RESET

  def switch_policy(self, policy_id: str) -> RuntimeStatus:
    assert policy_id
    return self.reset()

  def release(self) -> RuntimeStatus:
    self.pending = None
    self.closed = True
    self.worker_alive = False
    return RuntimeStatus.RELEASED

  def shutdown(self) -> RuntimeStatus:
    self.pending = None
    self.closed = True
    self.worker_alive = False
    return RuntimeStatus.SHUTDOWN


def _loaded_controller(
  clock: _Clock, runtime: _FakeRuntime | None = None
) -> tuple[D5Controller, _FakeRuntime]:
  owned = runtime or _FakeRuntime(clock)
  created = [owned]

  def factory(*, clock: Callable[[], float]) -> _FakeRuntime:
    if len(created) == 1:
      created.append(_FakeRuntime(clock))
    return created.pop(0)

  controller = D5Controller(factory, clock=clock)
  controller.load()
  for index in range(8):
    assert controller.acknowledge_bootstrap(
      make_state(index), np.full(32, index, dtype=np.float64), success=True
    )
  return controller, owned


def test_observation_owns_measured_inputs_and_validates_dimensions() -> None:
  state = make_state()
  conditioning = np.zeros(99)
  observation = ControllerObservation(state, conditioning, 0, 0, 0, "segment", 0.0)
  state.root_position[0] = 99.0
  conditioning[0] = 99.0
  assert observation.state.root_position[0] != 99.0
  assert observation.conditioning[0] == 0.0
  with pytest.raises(ControllerError, match="conditioning"):
    ControllerObservation(make_state(), np.zeros(98), 0, 0, 0, "segment", 0.0)


def test_warmup_is_discarded_and_activation_requires_fresh_measured_input() -> None:
  clock = _Clock()
  controller, runtime = _loaded_controller(clock)
  warmup_observation = _observation(sample_id=10, tick=20)
  report = controller.warmup(warmup_observation)
  assert report.planner_plan_sequence_id == 1
  assert controller.state is ControllerState.READY
  assert runtime.step_calls == []
  controller.request_activation()
  with pytest.raises(ControllerError, match="sample_id"):
    controller.step(warmup_observation)
  fresh = _observation(sample_id=11, tick=20, state_index=1)
  step = controller.step(fresh)
  assert step.status is RuntimeStatus.ACTION
  assert controller.state is ControllerState.ACTIVE
  assert controller.acknowledge(step, success=True)
  assert runtime.ack_calls == [(1, True)]
  controller.release()
  assert controller.cleanup_confirmed


def test_duplicate_load_preserves_owned_runtime_and_reset_requires_new_load() -> None:
  clock = _Clock()
  controller, runtime = _loaded_controller(clock)
  with pytest.raises(ControllerError, match="load"):
    controller.load()
  assert controller.runtime is runtime
  controller.reset()
  assert controller.state is ControllerState.LOADING
  assert controller.runtime is None
  assert controller.cleanup_confirmed
  controller.load()
  assert controller.state is ControllerState.BOOTSTRAPPING
  assert controller.runtime is not runtime
  controller.shutdown()


def test_failed_acknowledgement_faults_and_cannot_leave_pending_action() -> None:
  clock = _Clock()
  controller, runtime = _loaded_controller(clock)
  controller.warmup(_observation(sample_id=1))
  controller.request_activation()
  step = controller.step(_observation(sample_id=2, state_index=1))
  history_before = runtime.history_size
  assert not controller.acknowledge(step, success=False)
  # A failed execution must never add a measured pair.  Fault cleanup may
  # clear history, but the runtime must not have pushed this pending pair.
  assert runtime.history_size == history_before
  assert controller.state is ControllerState.FAULT
  assert controller.runtime is None
  assert runtime.pending is None
  with pytest.raises(ControllerError, match="state"):
    controller.step(_observation(sample_id=3, tick=1))


def test_task_revision_change_discards_old_pending_action_before_new_step() -> None:
  clock = _Clock()
  controller, runtime = _loaded_controller(clock)
  controller.warmup(
    _observation(sample_id=1), task={"mode": "unguided", "revision": "a"}
  )
  controller.request_activation()
  first = controller.step(
    _observation(sample_id=2, tick=1, state_index=1),
    task={"mode": "unguided", "revision": "a"},
  )
  second = controller.step(
    _observation(sample_id=3, tick=2, state_index=2),
    task={"mode": "unguided", "revision": "b"},
  )
  assert second.status is RuntimeStatus.ACTION
  assert runtime.ack_calls == [(first.execution_id or 0, False)]
  assert controller.acknowledge(second, success=True)
  controller.release()


def test_segment_frame_tick_and_age_guards_fail_closed() -> None:
  def activated() -> tuple[D5Controller, _FakeRuntime]:
    clock = _Clock(1.0)
    controller, runtime = _loaded_controller(clock)
    controller.warmup(_observation(sample_id=1, timestamp=1.0))
    controller.request_activation()
    return controller, runtime

  controller, runtime = activated()
  with pytest.raises(ControllerError, match="source_frame"):
    controller.step(_observation(sample_id=2, source_frame=4, timestamp=1.0))
  assert controller.state is ControllerState.FAULT
  assert controller.cleanup_confirmed
  assert runtime.closed

  controller, _runtime = activated()
  with pytest.raises(ControllerError, match="age"):
    controller.step(_observation(sample_id=2, timestamp=0.0))
  assert controller.state is ControllerState.READY

  controller, _runtime = activated()
  with pytest.raises(ControllerError, match="future"):
    controller.step(_observation(sample_id=2, timestamp=2.0))
  assert controller.state is ControllerState.READY

  controller, _runtime = activated()
  first = controller.step(_observation(sample_id=2, tick=5, timestamp=1.0))
  assert first.status is RuntimeStatus.ACTION
  with pytest.raises(ControllerError, match="increasing"):
    controller.step(_observation(sample_id=3, tick=5, timestamp=1.0))
  assert controller.state is ControllerState.ACTIVE


def test_runtime_callback_cannot_mutate_lifecycle() -> None:
  clock = _Clock()
  controller, runtime = _loaded_controller(clock)
  controller.warmup(_observation(sample_id=1))
  controller.request_activation()
  runtime.on_step = controller.reset
  result = controller.step(_observation(sample_id=2, tick=1))
  assert result.status is RuntimeStatus.PLANNER_FAILURE
  assert controller.state is ControllerState.FAULT
  assert controller.cleanup_confirmed


class _ImmediateExecutor:
  def __init__(self, callback: Callable[[PlannerRequest], Plan]) -> None:
    self.callback = callback
    self.pending: PlannerRequest | None = None
    self.closed = False
    self.worker_alive = True

  def submit(self, request: PlannerRequest) -> None:
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

  def wait_for_completion(self, timeout_seconds: float) -> bool:
    del timeout_seconds
    return self.pending is not None

  def shutdown(self) -> None:
    self.closed = True
    self.worker_alive = False
    self.pending = None


def _stats(width: int, seed: int) -> FeatureStats:
  rng = np.random.default_rng(seed)
  return fit_feature_stats([rng.normal(size=(32, width))], std_floor=1e-6)


def _projection() -> ProjectionBundle:
  return ProjectionBundle.create(_stats(135, 1), _stats(32, 2))


def _plan(request: PlannerRequest) -> Plan:
  packet = LatentPlanningPacket.from_request(request)
  task_revision = "none" if packet.task is None else packet.task.revision
  task_mode = "unguided" if packet.task is None else packet.task.mode
  metadata = {
    "schema": LATENT_PLANNING_SCHEMA,
    "contract": DEFAULT_CONTRACT.identity_hash(),
    "representation": RAW_LATENT_REPRESENTATION,
    "kind": "latent",
    "semantic": "raw",
    "latent_dimension": 32,
    "latent_space": "raw",
    "source_tick": request.source_tick,
    "valid_from_tick": request.source_tick,
    "valid_until_tick": request.source_tick + 32,
    "source_timestamp": request.source_timestamp,
    "source_frame": packet.source_frame,
    "root_frame": ROOT_FRAME,
    "period_seconds": DEFAULT_CONTRACT.period_seconds,
    "vae_identity": "test-vae",
    "projection_identity": packet.projection_identity,
    "model_identity": "test-model",
    "task_revision": task_revision,
    "task_mode": task_mode,
  }
  actions = np.zeros((33, 32), dtype=np.float64)
  return Plan(
    plan_sequence_id=request.request_sequence_id,
    source_timestamp=request.source_timestamp,
    valid_from_tick=request.source_tick,
    actions=actions,
    schema_metadata=metadata,
  )


def _immediate_runtime_controller() -> D5Controller:
  """Build a controller over the real runtime with an immediate executor."""

  clock = _Clock()
  executor = _ImmediateExecutor(_plan)
  projection = _projection()

  def decode(latent: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
    del conditioning
    return torch.cat((latent[:, :31], torch.zeros((latent.shape[0], 0))), dim=1)

  def factory(*, clock: Callable[[], float]) -> AsyncLatentRuntime:
    return AsyncLatentRuntime(
      executor,
      projection=projection,
      decode=decode,
      generator=torch.Generator(device="cpu").manual_seed(7),
      clock=clock,
      vae_identity="test-vae",
      model_identity="test-model",
    )

  controller = D5Controller(factory, clock=clock)
  controller.load()
  for index in range(8):
    assert controller.acknowledge_bootstrap(
      make_state(index), np.full(32, index, dtype=np.float64), success=True
    )
  return controller


def _poseguided_task(strength: float) -> dict[str, object]:
  """Return one pose-guided task mapping with a fixed revision string."""

  return {
    "mode": "poseguided",
    "goals": ({"tick": 5, "state": make_state(3), "provenance": "test-goal"},),
    "weights": {
      "root_position": 1.0,
      "root_orientation": 1.0,
      "body_positions": 1.0,
    },
    "strength": strength,
    "revision": "same-revision",
  }


def test_real_async_runtime_warmup_and_ack_use_controller_boundaries() -> None:
  controller = _immediate_runtime_controller()
  report = controller.warmup(_observation(sample_id=10, tick=20))
  assert report.decoder_calls == 3
  controller.request_activation()
  action = controller.step(_observation(sample_id=11, tick=21, state_index=9))
  assert action.status is RuntimeStatus.ACTION
  assert action.executed_latent is not None
  assert controller.acknowledge(action, success=True)
  runtime = controller.runtime
  assert runtime is not None
  assert runtime.history_size == 8
  # The bounded eight-pair deque alone cannot prove the acknowledged commit, so
  # assert the exact newest state and raw latent pair.
  np.testing.assert_allclose(
    runtime.history.pairs[-1].latent,
    action.executed_latent,
    rtol=1.0e-6,
    atol=1.0e-6,
  )
  np.testing.assert_array_equal(
    runtime.history.pairs[-1].state.root_position,
    make_state(9).root_position,
  )
  controller.release()
  assert controller.cleanup_confirmed


def test_same_revision_task_content_change_discards_pending_action() -> None:
  """A content-only task change must not fault or leave two pending actions."""

  controller = _immediate_runtime_controller()
  controller.warmup(
    _observation(sample_id=10, tick=20),
    task=_poseguided_task(0.1),
  )
  controller.request_activation()
  first = controller.step(
    _observation(sample_id=11, tick=21, state_index=9),
    task=_poseguided_task(0.1),
  )
  assert first.status is RuntimeStatus.ACTION
  runtime = controller.runtime
  assert runtime is not None
  assert runtime.pending_execution_id == first.execution_id

  # The revision string is unchanged; only the guidance strength differs.
  second = controller.step(
    _observation(sample_id=12, tick=22, state_index=10),
    task=_poseguided_task(0.2),
  )
  assert second.status is RuntimeStatus.ACTION
  assert second is not first
  assert controller.state is ControllerState.ACTIVE
  assert controller.failure_detail is None
  assert runtime.pending_execution_id == second.execution_id

  with pytest.raises(ControllerError, match="pending"):
    controller.acknowledge(first, success=True)
  assert controller.state is ControllerState.ACTIVE
  assert runtime.pending_execution_id == second.execution_id

  assert controller.acknowledge(second, success=True)
  assert second.executed_latent is not None
  assert runtime.history_size == 8
  np.testing.assert_allclose(
    runtime.history.pairs[-1].latent,
    second.executed_latent,
    rtol=1.0e-6,
    atol=1.0e-6,
  )
  np.testing.assert_array_equal(
    runtime.history.pairs[-1].state.root_position,
    make_state(10).root_position,
  )
  controller.release()
  assert controller.cleanup_confirmed


def test_policy_switch_discards_pending_action_and_requires_fresh_bootstrap() -> None:
  clock = _Clock()
  controller, runtime = _loaded_controller(clock)
  controller.warmup(
    _observation(sample_id=1), task={"mode": "unguided", "revision": "a"}
  )
  controller.request_activation()
  pending = controller.step(_observation(sample_id=2, tick=1, state_index=1))
  assert pending.status is RuntimeStatus.ACTION
  assert runtime.pending is pending

  controller.switch_policy("policy-next")
  assert controller.state is ControllerState.LOADING
  assert controller.runtime is None
  assert runtime.closed
  assert controller.cleanup_confirmed
  with pytest.raises(ControllerError, match="state"):
    controller.acknowledge(pending, success=True)
  with pytest.raises(ControllerError, match="state"):
    controller.step(_observation(sample_id=3))

  controller.load()
  assert controller.state is ControllerState.BOOTSTRAPPING
  fresh = controller.runtime
  assert fresh is not None
  assert fresh is not runtime
  with pytest.raises(ControllerError, match="warmup"):
    controller.warmup(_observation(sample_id=1, tick=1))
  for index in range(8):
    assert controller.acknowledge_bootstrap(
      make_state(index), np.full(32, float(index)), success=True
    )
  controller.warmup(_observation(sample_id=1, tick=1))
  assert controller.state is ControllerState.READY
  with pytest.raises(ControllerError, match="activation"):
    controller.step(_observation(sample_id=2, tick=2))
  controller.request_activation()
  resumed = controller.step(_observation(sample_id=2, tick=2, state_index=1))
  assert resumed.status is RuntimeStatus.ACTION
  assert controller.acknowledge(resumed, success=True)
  controller.release()
  assert controller.cleanup_confirmed
