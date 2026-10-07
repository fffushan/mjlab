"""Focused CPU tests for the D5 execution bridge."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from tracking_diffusion_fixtures import make_state

from mjlab.tasks.tracking.diffusion import controller_driver
from mjlab.tasks.tracking.diffusion.async_latent_runtime import (
  AsyncLatentRuntime,
  ExecutionProvenance,
  RuntimeStatus,
  RuntimeStep,
  WarmupReport,
)
from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT
from mjlab.tasks.tracking.diffusion.controller import (
  ControllerObservation,
  ControllerState,
  D5Controller,
)
from mjlab.tasks.tracking.diffusion.controller_driver import (
  ActionCommand,
  ControllerDispatch,
  D5ControllerDriver,
  DispatchOutcome,
  ExecutionOutcome,
  build_model_backed_controller_driver,
)
from mjlab.tasks.tracking.diffusion.latent_planning import (
  LATENT_PLANNING_SCHEMA,
  RAW_LATENT_REPRESENTATION,
  ROOT_FRAME,
  LatentPlanningPacket,
)
from mjlab.tasks.tracking.diffusion.model_backend import ModelBackendSpec
from mjlab.tasks.tracking.diffusion.projection import (
  ProjectionBundle,
  fit_feature_stats,
)
from mjlab.tasks.tracking.diffusion.scheduler import (
  Plan,
  PlannerRequest,
  PlannerResult,
  SchedulerEvent,
  SchedulerStatus,
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
    self.pending: RuntimeStep | None = None
    self.bootstrap_calls = 0
    self.warmup_calls = 0
    self.step_calls = 0
    self.ack_calls: list[tuple[RuntimeStep, bool]] = []
    self.on_warmup: Callable[[], None] | None = None

  @property
  def cleanup_confirmed(self) -> bool:
    return self.closed is True and self.worker_alive is False

  def acknowledge_bootstrap(self, state: Any, latent: Any, *, success: bool) -> bool:
    del state, latent
    if success:
      self.bootstrap_calls += 1
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
    del (
      state,
      conditioning,
      task,
      task_revision,
      decoder_calls,
      timeout_seconds,
    )
    assert source_frame == 3
    self.warmup_calls += 1
    if self.on_warmup is not None:
      self.on_warmup()
    return WarmupReport(1, tick + 1, tick + 1, 0.0)

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
    del state, conditioning, source_frame, task, task_revision
    self.step_calls += 1
    provenance = ExecutionProvenance(
      plan_sequence_id=self.step_calls,
      source_timestamp=0.0,
      source_tick=tick,
      execution_tick=tick,
      valid_until_tick=tick + 32,
      source_frame=3,
      contract_identity=DEFAULT_CONTRACT.identity_hash(),
      vae_identity="fake-vae",
      projection_identity="fake-projection",
      model_identity="fake-model",
      task_revision="none",
      task_mode="unguided",
    )
    result = RuntimeStep(
      RuntimeStatus.ACTION,
      tick,
      np.full(31, float(tick)),
      np.full(32, float(tick)),
      provenance,
      self.step_calls,
    )
    self.pending = result
    return result

  def acknowledge(self, step: RuntimeStep, *, success: bool) -> bool:
    assert self.pending is step
    self.ack_calls.append((step, success))
    self.pending = None
    if success:
      self.history_size = min(self.history_size + 1, 8)
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


def _loaded_driver(
  *,
  clock: _Clock | None = None,
  runtime: _FakeRuntime | None = None,
  sink: Callable[[ActionCommand], object] | None = None,
) -> tuple[D5ControllerDriver, D5Controller, _FakeRuntime, list[ActionCommand]]:
  owned_clock = clock or _Clock()
  owned_runtime = runtime or _FakeRuntime(owned_clock)
  controller = D5Controller(
    lambda *, clock: owned_runtime,
    clock=owned_clock,
  )
  controller.load()
  for index in range(8):
    assert controller.acknowledge_bootstrap(
      make_state(index), np.full(32, float(index)), success=True
    )
  commands: list[ActionCommand] = []
  execute: Callable[[ActionCommand], object]
  if sink is None:

    def default_execute(command: ActionCommand) -> bool:
      commands.append(command)
      return True

    execute = default_execute
  else:
    execute = sink
  observations = [
    _observation(sample_id=1, tick=10),
    _observation(sample_id=2, tick=10, state_index=1),
  ]

  def read_observation() -> ControllerObservation:
    if not observations:
      raise RuntimeError("observation queue is empty")
    return observations.pop(0)

  driver = D5ControllerDriver(controller, read_observation, execute)
  return driver, controller, owned_runtime, commands


def _prepare_active(
  *,
  sink: Callable[[ActionCommand], object] | None = None,
) -> tuple[D5ControllerDriver, D5Controller, _FakeRuntime, list[ActionCommand]]:
  driver, controller, runtime, commands = _loaded_driver(sink=sink)
  driver.warmup()
  driver.request_activation()
  return driver, controller, runtime, commands


def test_no_sink_before_activation_or_during_warmup_and_fresh_action_boundary() -> None:
  driver, controller, runtime, commands = _loaded_driver()
  driver.warmup()
  assert runtime.warmup_calls == 1
  assert commands == []

  before_activation = driver.tick()
  assert before_activation.dispatch is DispatchOutcome.NOT_DISPATCHED
  assert before_activation.execution is ExecutionOutcome.NOT_ATTEMPTED
  assert commands == []

  driver.request_activation()
  # The fake observation reader needs a new sample after the failed call.
  driver._read_observation = lambda: _observation(sample_id=2, tick=10)
  result = driver.tick()
  assert result.dispatch is DispatchOutcome.DISPATCHED
  assert result.execution is ExecutionOutcome.ACCEPTED
  assert result.command is not None
  assert result.command.action.shape == (31,)
  assert not hasattr(result.command, "latent")
  assert commands == [result.command]
  assert controller.state is ControllerState.ACTIVE


def test_no_sink_publication_during_bootstrapping_and_warming() -> None:
  driver, controller, runtime, commands = _loaded_driver()
  assert controller.state is ControllerState.BOOTSTRAPPING
  bootstrap_tick = driver.tick()
  assert bootstrap_tick.dispatch is DispatchOutcome.NOT_DISPATCHED
  assert bootstrap_tick.execution is ExecutionOutcome.NOT_ATTEMPTED
  assert commands == []

  warming: list[tuple[ControllerState, ControllerDispatch, str]] = []

  def during_warmup() -> None:
    result = driver.tick()
    warming.append((controller.state, result, result.detail))

  runtime.on_warmup = during_warmup
  controller.warmup(_observation(sample_id=1, tick=10))
  assert len(warming) == 1
  state, dispatch, detail = warming[0]
  assert state is ControllerState.WARMING
  assert dispatch.dispatch is DispatchOutcome.NOT_DISPATCHED
  assert dispatch.execution is ExecutionOutcome.NOT_ATTEMPTED
  # The bridge refuses the tick while warmup owns the runtime callback and the
  # lifecycle remains WARMING; the execution sink is never reached.
  assert dispatch.command is None
  assert dispatch.runtime_step is None
  assert detail
  assert commands == []
  assert controller.state is ControllerState.READY


def test_success_acknowledges_original_step_once_and_owns_action_copy() -> None:
  driver, controller, runtime, commands = _prepare_active()
  result = driver.tick()
  assert result.execution is ExecutionOutcome.ACCEPTED
  assert result.runtime_step is not None
  assert result.command is not None
  step = result.runtime_step
  command = result.command
  assert command is commands[0]
  assert command.execution_id == step.execution_id
  assert command.action is not step.action
  assert runtime.ack_calls == [(step, True)]
  assert controller.state is ControllerState.ACTIVE
  assert command.action.flags.writeable is True
  before = np.array(step.action, copy=True)
  command.action[0] = 123.0
  np.testing.assert_array_equal(step.action, before)


def _receipt_cases() -> tuple[
  tuple[object, type[BaseException] | None, ExecutionOutcome], ...
]:
  return (
    (False, None, ExecutionOutcome.REJECTED),
    (None, None, ExecutionOutcome.FAILED),
    (np.bool_(True), None, ExecutionOutcome.FAILED),
    (1, None, ExecutionOutcome.FAILED),
    (None, RuntimeError, ExecutionOutcome.FAILED),
  )


@pytest.mark.parametrize("receipt,raised,expected", _receipt_cases())
def test_rejection_exception_and_nonboolean_receipts_fault_without_ack(
  receipt: object,
  raised: type[BaseException] | None,
  expected: ExecutionOutcome,
) -> None:
  def sink(_command: ActionCommand) -> object:
    if raised is not None:
      raise raised("sink failed")
    return receipt

  driver, controller, runtime, _commands = _prepare_active(sink=sink)
  result = driver.tick()
  assert result.dispatch is DispatchOutcome.DISPATCHED
  assert result.execution is expected
  assert controller.state is ControllerState.FAULT
  assert controller.cleanup_confirmed
  assert runtime.ack_calls == []
  assert runtime.pending is None


def test_reentrant_tick_and_lifecycle_reset_suppress_stale_acknowledgement() -> None:
  nested: list[ControllerDispatch] = []
  driver_ref: list[D5ControllerDriver] = []

  def sink(_command: ActionCommand) -> bool:
    nested.append(driver_ref[0].tick())
    return True

  driver, controller, runtime, _commands = _prepare_active(sink=sink)
  driver_ref.append(driver)
  result = driver.tick()
  assert result.execution is ExecutionOutcome.ACCEPTED
  assert nested[0].dispatch is DispatchOutcome.NOT_DISPATCHED
  assert runtime.ack_calls == [(result.runtime_step, True)]

  def reset_sink(_command: ActionCommand) -> bool:
    controller.reset()
    return True

  driver, controller, runtime, _commands = _prepare_active(sink=reset_sink)
  result = driver.tick()
  assert result.dispatch is DispatchOutcome.DISPATCHED
  assert result.execution is ExecutionOutcome.SUPPRESSED
  assert "suppressed" in result.detail
  assert result.runtime_step is not None
  assert runtime.ack_calls == []
  assert controller.state is ControllerState.LOADING
  assert controller.runtime is None
  assert controller.cleanup_confirmed


def test_release_and_shutdown_are_terminal_after_dispatch() -> None:
  def release_sink(_command: ActionCommand) -> bool:
    controller.release()
    return True

  driver, controller, runtime, _commands = _prepare_active(sink=release_sink)
  released = driver.tick()
  assert released.execution is ExecutionOutcome.SUPPRESSED
  assert controller.state is ControllerState.RELEASED
  assert controller.cleanup_confirmed
  assert runtime.ack_calls == []
  assert driver.tick().dispatch is DispatchOutcome.NOT_DISPATCHED

  def shutdown_sink(_command: ActionCommand) -> bool:
    controller.shutdown()
    return True

  driver, controller, runtime, _commands = _prepare_active(sink=shutdown_sink)
  shut_down = driver.tick()
  assert shut_down.execution is ExecutionOutcome.SUPPRESSED
  assert controller.state is ControllerState.SHUTDOWN
  assert controller.cleanup_confirmed
  assert runtime.ack_calls == []


def test_scheduler_warning_telemetry_is_forwarded_without_global_warning() -> None:
  driver, _controller, runtime, _commands = _prepare_active()
  original_step = runtime.step

  def warning_step(
    state: Any,
    conditioning: Any,
    *,
    tick: int,
    source_frame: int,
    task: Any,
    task_revision: str | None,
  ) -> RuntimeStep:
    step = original_step(
      state,
      conditioning,
      tick=tick,
      source_frame=source_frame,
      task=task,
      task_revision=task_revision,
    )
    result = RuntimeStep(
      step.status,
      step.control_tick,
      step.action,
      step.executed_latent,
      step.provenance,
      step.execution_id,
      step.detail,
      (
        SchedulerEvent(
          SchedulerStatus.FALLBACK_WARNING,
          "fallback threshold reached",
          plan_sequence_id=1,
        ),
      ),
    )
    runtime.pending = result
    return result

  runtime.step = warning_step  # type: ignore[method-assign]
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    result = driver.tick()
  assert result.execution is ExecutionOutcome.ACCEPTED
  assert len(result.warning_events) == 1
  assert result.warning_events[0].status is SchedulerStatus.FALLBACK_WARNING


def _projection() -> ProjectionBundle:
  rng = np.random.default_rng(3)
  state_stats = fit_feature_stats([rng.normal(size=(32, 135))], std_floor=1e-6)
  latent_stats = fit_feature_stats([rng.normal(size=(32, 32))], std_floor=1e-6)
  return ProjectionBundle.create(state_stats, latent_stats)


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
  return Plan(
    plan_sequence_id=request.request_sequence_id,
    source_timestamp=request.source_timestamp,
    valid_from_tick=request.source_tick,
    actions=np.zeros((33, 32), dtype=np.float64),
    schema_metadata=metadata,
  )


def test_actual_lifecycle_and_runtime_path_acknowledges_through_driver() -> None:
  clock = _Clock()
  executor = _ImmediateExecutor(_plan)
  projection = _projection()

  def decode(latent: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
    del conditioning
    return latent[:, :31]

  def factory(*, clock: Callable[[], float]) -> AsyncLatentRuntime:
    return AsyncLatentRuntime(
      executor,
      projection=projection,
      decode=decode,
      generator=torch.Generator(device="cpu").manual_seed(4),
      clock=clock,
      vae_identity="test-vae",
      model_identity="test-model",
    )

  controller = D5Controller(factory, clock=clock)
  controller.load()
  for index in range(8):
    assert controller.acknowledge_bootstrap(
      make_state(index), np.full(32, float(index)), success=True
    )
  observations = [
    _observation(sample_id=10, tick=20),
    _observation(sample_id=11, tick=21, state_index=9),
  ]
  commands: list[ActionCommand] = []

  def read_observation() -> ControllerObservation:
    return observations.pop(0)

  def execute(command: ActionCommand) -> bool:
    commands.append(command)
    return True

  driver = D5ControllerDriver(controller, read_observation, execute)
  driver.warmup()
  driver.request_activation()
  result = driver.tick()
  assert result.execution is ExecutionOutcome.ACCEPTED
  assert len(commands) == 1
  runtime = controller.runtime
  assert runtime is not None
  assert runtime.history_size == 8
  step = result.runtime_step
  assert step is not None
  assert step.executed_latent is not None
  # The bounded eight-pair deque alone cannot prove the acknowledged commit.
  np.testing.assert_allclose(
    runtime.history.pairs[-1].latent,
    step.executed_latent,
    rtol=1.0e-6,
    atol=1.0e-6,
  )
  driver.release()
  assert controller.cleanup_confirmed


def test_model_backed_assembly_passes_spec_generator_and_exact_clock(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  clock = _Clock()
  generator = torch.Generator(device="cpu").manual_seed(8)
  captured: dict[str, object] = {}
  runtime = _FakeRuntime(clock)

  def fake_build(
    spec: ModelBackendSpec,
    *,
    generator: torch.Generator,
    clock: Callable[[], float],
    **kwargs: object,
  ) -> _FakeRuntime:
    captured.update({"spec": spec, "generator": generator, "clock": clock, **kwargs})
    return runtime

  monkeypatch.setattr(controller_driver, "build_model_backed_runtime", fake_build)
  spec = ModelBackendSpec(
    contract_path=Path("contract.yaml"),
    diffusion_checkpoint=Path("diffusion.pt"),
    vae_checkpoint=Path("vae.pt"),
    vae_manifest=Path("vae.yaml"),
    normalization_bundle=Path("normalizer.npz"),
    repo_root=Path("."),
    vae_member="member-0",
  )
  driver = build_model_backed_controller_driver(
    spec,
    generator=generator,
    read_observation=lambda: _observation(sample_id=1),
    execute_action=lambda _command: True,
    clock=clock,
    max_fallback_ticks=9,
    max_source_age_seconds=0.25,
    max_future_skew_seconds=0.125,
    expected_source_frame=4,
  )
  assembled_controller = cast(D5Controller, driver.controller)
  assembled_controller.load()
  assert captured["spec"] is spec
  assert captured["generator"] is generator
  assert captured["clock"] is clock
  assert captured["max_fallback_ticks"] == 9
  assert captured["max_source_age_seconds"] == 0.25
  assert captured["max_future_skew_seconds"] == 0.125
  assert captured["expected_source_frame"] == 4
  assert runtime.clock is clock
  assembled_controller.release()
