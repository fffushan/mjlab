"""Real CPU-spawn integration checks for the D5 controller bridge."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch
from tracking_diffusion_fixtures import make_state

from mjlab.tasks.tracking.diffusion.async_latent_runtime import (
  AsyncLatentRuntime,
  RuntimeStatus,
)
from mjlab.tasks.tracking.diffusion.controller import (
  ControllerError,
  ControllerObservation,
  ControllerState,
  D5Controller,
)
from mjlab.tasks.tracking.diffusion.controller_driver import (
  ActionCommand,
  D5ControllerDriver,
  DispatchOutcome,
  ExecutionOutcome,
)
from mjlab.tasks.tracking.diffusion.latent_planning import LatentPlanner
from mjlab.tasks.tracking.diffusion.planner_executor import (
  BoundedPlannerExecutor,
  PlannerExecutorError,
)
from mjlab.tasks.tracking.diffusion.projection import (
  ProjectionBundle,
  fit_feature_stats,
)
from mjlab.tasks.tracking.diffusion.schedule import DiffusionSchedule, InferenceGrid
from mjlab.tasks.tracking.diffusion.scheduler import PlannerRequest, SchedulerStatus

_INTEGRATION_FRAME = 7
_INTEGRATION_SEGMENT = "cpu-spawn-integration"
_INTEGRATION_VAE = "synthetic-integration-vae"
_INTEGRATION_MODEL = "synthetic-integration-model"


class _Clock:
  def __init__(self, value: float = 0.0) -> None:
    self.value = value

  def __call__(self) -> float:
    return self.value


def _projection() -> ProjectionBundle:
  rng = np.random.default_rng(20261007)
  state_stats = fit_feature_stats(
    [rng.normal(size=(64, 135))],
    std_floor=1.0e-6,
  )
  latent_stats = fit_feature_stats(
    [rng.normal(size=(64, 32))],
    std_floor=1.0e-6,
  )
  return ProjectionBundle.create(state_stats, latent_stats)


class _SyntheticDenoiser:
  """Small finite denoiser created inside the real spawned planner worker."""

  def __call__(self, sample: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
    del step_ids
    return 0.2 * sample


def _latent_planner_factory() -> Callable[..., Any]:
  """Build the actual ``LatentPlanner.process`` callback in the worker."""

  schedule = DiffusionSchedule.build()
  planner = LatentPlanner(
    denoiser=_SyntheticDenoiser(),
    schedule=schedule,
    grid=InferenceGrid.uniform(schedule),
    projection=_projection(),
    vae_identity=_INTEGRATION_VAE,
    model_identity=_INTEGRATION_MODEL,
  )
  return planner.process


def _failed_planner(_request: object) -> object:
  raise RuntimeError("synthetic planner callback failed")


def _failed_worker_factory() -> Callable[..., Any]:
  return _failed_planner


def _blocked_planner(_request: object) -> object:
  time.sleep(60.0)
  raise RuntimeError("synthetic blocked planner callback was not stopped")


def _blocked_worker_factory() -> Callable[..., Any]:
  return _blocked_planner


class _RecordingDecoder:
  """Owner-side synthetic VAE decoder with observable inputs."""

  def __init__(self) -> None:
    self.latents: list[np.ndarray] = []
    self.conditionings: list[np.ndarray] = []

  def __call__(self, latent: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
    self.latents.append(latent[0].detach().cpu().numpy().copy())
    self.conditionings.append(conditioning[0].detach().cpu().numpy().copy())
    return latent[:, :31] + 0.01 * conditioning[:, :31]


def _observation(
  *,
  sample_id: int,
  tick: int,
  state_index: int,
  conditioning_value: float,
  timestamp: float = 0.0,
) -> ControllerObservation:
  return ControllerObservation(
    make_state(state_index),
    np.full(99, conditioning_value, dtype=np.float64),
    tick,
    sample_id,
    _INTEGRATION_FRAME,
    _INTEGRATION_SEGMENT,
    timestamp,
  )


def _runtime_factory_records(
  runtimes: list[AsyncLatentRuntime],
  executors: list[BoundedPlannerExecutor],
  decoders: list[_RecordingDecoder],
  *,
  planner_factory: Callable[..., Any] = _latent_planner_factory,
  shutdown_timeout_seconds: float = 2.0,
) -> Callable[[Callable[[], float]], AsyncLatentRuntime]:
  def factory(clock: Callable[[], float]) -> AsyncLatentRuntime:
    executor = BoundedPlannerExecutor(
      planner_factory,
      startup_timeout_seconds=5.0,
      shutdown_timeout_seconds=shutdown_timeout_seconds,
    )
    decoder = _RecordingDecoder()
    runtime = AsyncLatentRuntime(
      executor,
      projection=_projection(),
      decode=decoder,
      generator=torch.Generator(device="cpu").manual_seed(2026),
      clock=clock,
      max_fallback_ticks=0,
      vae_identity=_INTEGRATION_VAE,
      model_identity=_INTEGRATION_MODEL,
      expected_source_frame=_INTEGRATION_FRAME,
    )
    executors.append(executor)
    decoders.append(decoder)
    runtimes.append(runtime)
    return runtime

  return factory


def _acknowledge_bootstrap(controller: D5Controller, *, state_start: int) -> None:
  for offset in range(8):
    assert controller.acknowledge_bootstrap(
      make_state(state_start + offset),
      np.full(32, 0.05 * offset, dtype=np.float64),
      success=True,
    )
  assert controller.runtime is not None
  assert controller.runtime.history_size == 8
  assert controller.runtime.history_ready


def test_spawned_controller_driver_owns_warmup_action_ack_reset_and_release(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Exercise the assembled lifecycle through one real spawned CPU worker."""

  clock = _Clock()
  runtimes: list[AsyncLatentRuntime] = []
  executors: list[BoundedPlannerExecutor] = []
  decoders: list[_RecordingDecoder] = []
  controller = D5Controller(
    _runtime_factory_records(runtimes, executors, decoders),
    clock=clock,
    max_observation_age_seconds=0.5,
  )
  observations = [
    _observation(
      sample_id=100,
      tick=40,
      state_index=8,
      conditioning_value=100.0,
    ),
    _observation(
      sample_id=101,
      tick=41,
      state_index=9,
      conditioning_value=101.0,
    ),
    _observation(
      sample_id=102,
      tick=42,
      state_index=10,
      conditioning_value=202.0,
    ),
    _observation(
      sample_id=103,
      tick=43,
      state_index=11,
      conditioning_value=203.0,
    ),
    _observation(
      sample_id=104,
      tick=44,
      state_index=12,
      conditioning_value=204.0,
    ),
    _observation(
      sample_id=105,
      tick=45,
      state_index=13,
      conditioning_value=205.0,
    ),
    _observation(
      sample_id=106,
      tick=46,
      state_index=14,
      conditioning_value=206.0,
    ),
  ]
  commands: list[ActionCommand] = []

  def read_observation() -> ControllerObservation:
    if not observations:
      raise AssertionError("integration observation queue was exhausted")
    return observations.pop(0)

  def execute_action(command: ActionCommand) -> bool:
    commands.append(command)
    return True

  driver = D5ControllerDriver(controller, read_observation, execute_action)
  try:
    controller.load()
    assert controller.state is ControllerState.BOOTSTRAPPING
    _acknowledge_bootstrap(controller, state_start=0)
    assert len(commands) == 0

    report = driver.warmup(decoder_calls=2, timeout_seconds=5.0)
    assert report.decoder_calls == 2
    assert controller.state is ControllerState.READY
    assert len(commands) == 0
    assert len(decoders[0].latents) == 2

    driver.request_activation()
    cold_step = driver.tick()
    assert cold_step.runtime_status in {
      RuntimeStatus.WAITING_FOR_PLAN,
      RuntimeStatus.ACTION,
    }
    assert cold_step.dispatch is DispatchOutcome.NOT_DISPATCHED or (
      cold_step.execution is ExecutionOutcome.ACCEPTED
    )
    if cold_step.execution is ExecutionOutcome.ACCEPTED:
      active_dispatch = cold_step
      active_conditioning_value = 101.0
      active_state_index = 9
    else:
      assert cold_step.runtime_status is RuntimeStatus.WAITING_FOR_PLAN
      assert executors[0].wait_for_completion(5.0)
      active_dispatch = driver.tick()
      active_conditioning_value = 202.0
      active_state_index = 10

    assert active_dispatch.execution is ExecutionOutcome.ACCEPTED
    assert active_dispatch.dispatch is DispatchOutcome.DISPATCHED
    assert active_dispatch.runtime_step is not None
    assert active_dispatch.command is not None
    assert active_dispatch.runtime_step.status is RuntimeStatus.ACTION
    assert controller.state is ControllerState.ACTIVE
    assert len(commands) == 1

    step = active_dispatch.runtime_step
    command = active_dispatch.command
    assert step.executed_latent is not None
    assert step.action is not None
    assert step.executed_latent.shape == (32,)
    assert step.action.shape == (31,)
    assert command.action.shape == (31,)
    assert command.provenance is step.provenance
    np.testing.assert_allclose(
      command.action,
      np.asarray(
        np.asarray(step.executed_latent[:31], dtype=np.float32)
        + np.float32(0.01) * np.float32(active_conditioning_value),
        dtype=np.float32,
      ),
      rtol=1.0e-6,
      atol=1.0e-6,
    )
    np.testing.assert_array_equal(
      decoders[0].conditionings[-1],
      np.full(99, active_conditioning_value),
    )
    np.testing.assert_allclose(
      decoders[0].latents[-1], step.executed_latent, rtol=1.0e-6, atol=1.0e-6
    )
    assert controller.runtime is not None
    assert controller.runtime.pending_execution_id is None
    assert controller.runtime.history_size == 8
    np.testing.assert_allclose(
      controller.runtime.history.pairs[-1].latent,
      step.executed_latent,
      rtol=1.0e-6,
      atol=1.0e-6,
    )
    np.testing.assert_array_equal(
      controller.runtime.history.pairs[-1].state.root_position,
      make_state(active_state_index).root_position,
    )

    # Deterministically force one fallback tick: replace only this executor's
    # poll with an empty result for this tick, then restore it immediately.  The
    # real worker completion stays in its pipe; no plan is synthesized,
    # consumed or discarded, and the real worker, planner, action, ack and
    # cleanup all stay exercised.
    with monkeypatch.context() as patch:
      patch.setattr(executors[0], "poll", lambda: ())
      warning_dispatch = driver.tick()
    assert warning_dispatch.runtime_step is not None
    assert warning_dispatch.runtime_step.status is RuntimeStatus.ACTION
    assert warning_dispatch.scheduler_events
    assert any(
      event.status is SchedulerStatus.FALLBACK_WARNING
      for event in warning_dispatch.scheduler_events
    )
    assert warning_dispatch.execution is ExecutionOutcome.ACCEPTED
    assert len(commands) == 2

    old_runtime = runtimes[0]
    old_executor = executors[0]
    driver.reset()
    assert controller.state is ControllerState.LOADING
    assert controller.runtime is None
    assert controller.cleanup_confirmed
    assert old_runtime.cleanup_confirmed
    assert old_executor.closed is True
    assert old_executor.worker_alive is False
    with pytest.raises(ControllerError, match="state"):
      controller.acknowledge(step, success=True)

    controller.load()
    assert controller.state is ControllerState.BOOTSTRAPPING
    _acknowledge_bootstrap(controller, state_start=20)
    assert controller.runtime is runtimes[1]
    assert controller.runtime.history_size == 8
    report = driver.warmup(decoder_calls=1, timeout_seconds=5.0)
    assert report.decoder_calls == 1
    assert controller.state is ControllerState.READY
    assert len(commands) == 2
    driver.request_activation()

    driver.release()
    assert controller.state is ControllerState.RELEASED
    assert controller.runtime is None
    assert controller.cleanup_confirmed
    assert executors[1].closed is True
    assert executors[1].worker_alive is False
    after_release = driver.tick()
    assert after_release.dispatch is DispatchOutcome.NOT_DISPATCHED
    assert after_release.execution is ExecutionOutcome.NOT_ATTEMPTED
    assert after_release.controller_state is ControllerState.RELEASED
    assert len(commands) == 2
  finally:
    if controller.state not in {ControllerState.RELEASED, ControllerState.SHUTDOWN}:
      controller.shutdown()


def test_spawned_worker_failure_does_not_report_ready_and_confirms_cleanup() -> None:
  """A real worker callback failure faults the controller and closes its worker."""

  clock = _Clock()
  runtimes: list[AsyncLatentRuntime] = []
  executors: list[BoundedPlannerExecutor] = []
  decoders: list[_RecordingDecoder] = []
  controller = D5Controller(
    _runtime_factory_records(
      runtimes,
      executors,
      decoders,
      planner_factory=_failed_worker_factory,
    ),
    clock=clock,
  )
  driver = D5ControllerDriver(
    controller,
    lambda: _observation(
      sample_id=100,
      tick=40,
      state_index=8,
      conditioning_value=1.0,
    ),
    lambda _command: pytest.fail("failed warmup must not call the action sink"),
  )
  controller.load()
  _acknowledge_bootstrap(controller, state_start=0)
  with pytest.raises(ControllerError, match="warmup"):
    driver.warmup(decoder_calls=1, timeout_seconds=5.0)
  assert controller.state is ControllerState.FAULT
  assert controller.runtime is None
  assert controller.cleanup_confirmed
  assert runtimes[0].cleanup_confirmed
  assert executors[0].closed is True
  assert executors[0].worker_alive is False
  with pytest.raises(ControllerError, match="state"):
    driver.request_activation()


def test_busy_spawned_worker_does_not_claim_ready_or_hide_cleanup() -> None:
  """A busy real worker causes fail-closed warmup and exact cleanup."""

  clock = _Clock()
  runtimes: list[AsyncLatentRuntime] = []
  executors: list[BoundedPlannerExecutor] = []
  decoders: list[_RecordingDecoder] = []
  controller = D5Controller(
    _runtime_factory_records(
      runtimes,
      executors,
      decoders,
      planner_factory=_blocked_worker_factory,
      shutdown_timeout_seconds=0.05,
    ),
    clock=clock,
  )
  driver = D5ControllerDriver(
    controller,
    lambda: _observation(
      sample_id=100,
      tick=40,
      state_index=8,
      conditioning_value=1.0,
    ),
    lambda _command: pytest.fail("busy warmup must not call the action sink"),
  )
  controller.load()
  _acknowledge_bootstrap(controller, state_start=0)
  executor = executors[0]
  executor.submit(
    PlannerRequest(
      input_snapshot=np.ones(1, dtype=np.float64),
      source_timestamp=0.0,
      source_tick=0,
      request_sequence_id=1,
      policy_generation=0,
    )
  )

  with pytest.raises(ControllerError, match="warmup"):
    driver.warmup(decoder_calls=1, timeout_seconds=0.1)
  assert controller.state is ControllerState.FAULT
  assert controller.runtime is None
  assert controller.cleanup_confirmed
  assert runtimes[0].cleanup_confirmed
  assert executor.closed is True
  assert executor.worker_alive is False
  assert controller.failure_detail is not None
  assert any(
    marker in controller.failure_detail.lower() for marker in ("busy", "already active")
  )


def test_runtime_factory_failure_does_not_claim_readiness() -> None:
  """A constructor failure leaves the lifecycle in an explicit fault state."""

  clock = _Clock()

  def factory(_clock: Callable[[], float]) -> AsyncLatentRuntime:
    raise PlannerExecutorError("synthetic worker factory construction failed")

  controller = D5Controller(factory, clock=clock)
  with pytest.raises(ControllerError, match="runtime factory failed"):
    controller.load()
  assert controller.state is ControllerState.FAULT
  assert controller.runtime is None
  assert controller.failure_detail is not None
  assert "factory construction failed" in controller.failure_detail
  assert controller.cleanup_confirmed is False
  with pytest.raises(ControllerError, match="state"):
    controller.request_activation()
