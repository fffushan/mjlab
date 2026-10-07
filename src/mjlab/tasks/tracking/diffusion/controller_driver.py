"""Synchronous execution bridge for the D5 controller lifecycle.

The bridge owns no planner, decoder or pacing loop.  It converts one decoded
31-D runtime action into an owned execution command, calls the injected
execution boundary, and acknowledges the exact originating runtime step only
when that boundary returns the literal boolean ``True``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np
import torch

from .async_latent_runtime import (
  AsyncLatentRuntime,
  ExecutionProvenance,
  RuntimeStatus,
  RuntimeStep,
  WarmupReport,
)
from .controller import (
  ControllerObservation,
  ControllerState,
  D5Controller,
)
from .model_backend import ModelBackendSpec, build_model_backed_runtime
from .scheduler import SchedulerEvent

_ACTION_DIMENSION = 31
_MISSING = object()


class ControllerDriverError(ValueError):
  """An execution bridge input or callback contract is invalid."""


class DispatchOutcome(str, Enum):
  """Whether the execution sink was called for a runtime action."""

  NOT_DISPATCHED = "not_dispatched"
  DISPATCHED = "dispatched"


class ExecutionOutcome(str, Enum):
  """The result of the execution boundary for one dispatched command."""

  NOT_ATTEMPTED = "not_attempted"
  ACCEPTED = "accepted"
  REJECTED = "rejected"
  FAILED = "failed"
  SUPPRESSED = "suppressed"


def _nonnegative_int(value: object, name: str) -> int:
  if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
    raise ControllerDriverError(f"{name} must be a non-negative integer")
  result = int(value)
  if result < 0:
    raise ControllerDriverError(f"{name} must be a non-negative integer")
  return result


def _copy_action(value: object) -> np.ndarray:
  try:
    action = np.asarray(value, dtype=np.float64)
  except (TypeError, ValueError, OverflowError) as exc:
    raise ControllerDriverError("action must be numeric") from exc
  if action.shape != (_ACTION_DIMENSION,) or not np.isfinite(action).all():
    raise ControllerDriverError("action must be finite with shape [31]")
  owned = np.array(action, dtype=np.float64, copy=True)
  return owned


def _events(step: RuntimeStep | None) -> tuple[SchedulerEvent, ...]:
  if step is None:
    return ()
  values = getattr(step, "scheduler_events", ())
  try:
    result = tuple(values)
  except TypeError as exc:
    raise ControllerDriverError("runtime scheduler events are not iterable") from exc
  if not all(isinstance(event, SchedulerEvent) for event in result):
    raise ControllerDriverError("runtime scheduler events are malformed")
  return result


def _cleanup_confirmed(controller: object) -> bool:
  """Read exact cleanup evidence without turning telemetry into a fault."""

  try:
    marker = getattr(controller, "cleanup_confirmed", _MISSING)
  except BaseException:
    return False
  return marker is True


@dataclass(frozen=True, slots=True)
class ActionCommand:
  """One owned decoded action sent to the injected execution boundary.

  The raw latent is intentionally absent.  ``provenance`` remains attached so
  the receiver can bind the command to its source plan and execution tick.
  """

  action: np.ndarray
  execution_id: int
  tick: int
  provenance: ExecutionProvenance

  def __post_init__(self) -> None:
    object.__setattr__(self, "action", _copy_action(self.action))
    execution_id = _nonnegative_int(self.execution_id, "execution_id")
    if execution_id == 0:
      raise ControllerDriverError("execution_id must be positive")
    tick = _nonnegative_int(self.tick, "tick")
    if not isinstance(self.provenance, ExecutionProvenance):
      raise ControllerDriverError("provenance must be ExecutionProvenance")
    if self.provenance.execution_tick != tick:
      raise ControllerDriverError("provenance execution tick differs from command")
    object.__setattr__(self, "execution_id", execution_id)
    object.__setattr__(self, "tick", tick)


@dataclass(frozen=True, slots=True)
class ControllerDispatch:
  """Evidence for one driver operation.

  ``runtime_step`` is the exact step returned by the controller.  For an
  action, ``command`` owns a separate 31-D array while the step is retained
  unchanged for acknowledgement.  A dispatched command remains marked as
  dispatched even when its callback fails or changes lifecycle state.
  """

  controller_state: ControllerState
  runtime_step: RuntimeStep | None
  command: ActionCommand | None
  dispatch_outcome: DispatchOutcome
  execution_outcome: ExecutionOutcome
  detail: str = ""
  cleanup_confirmed: bool = False
  scheduler_events: tuple[SchedulerEvent, ...] = ()

  def __post_init__(self) -> None:
    if not isinstance(self.controller_state, ControllerState):
      raise ControllerDriverError("controller_state must be ControllerState")
    if self.runtime_step is not None and not isinstance(self.runtime_step, RuntimeStep):
      raise ControllerDriverError("runtime_step must be RuntimeStep or None")
    if self.command is not None and not isinstance(self.command, ActionCommand):
      raise ControllerDriverError("command must be ActionCommand or None")
    if not isinstance(self.dispatch_outcome, DispatchOutcome):
      raise ControllerDriverError("dispatch_outcome must be DispatchOutcome")
    if not isinstance(self.execution_outcome, ExecutionOutcome):
      raise ControllerDriverError("execution_outcome must be ExecutionOutcome")
    events = tuple(self.scheduler_events)
    if not all(isinstance(event, SchedulerEvent) for event in events):
      raise ControllerDriverError("scheduler_events must contain SchedulerEvent values")
    object.__setattr__(self, "scheduler_events", events)
    if self.dispatch_outcome is DispatchOutcome.NOT_DISPATCHED:
      if self.command is not None:
        raise ControllerDriverError(
          "not-dispatched outcome cannot contain an action command"
        )
      if self.execution_outcome is not ExecutionOutcome.NOT_ATTEMPTED:
        raise ControllerDriverError(
          "not-dispatched outcome cannot contain an execution result"
        )
    elif self.command is None:
      raise ControllerDriverError("dispatched outcome requires an action command")
    if not isinstance(self.cleanup_confirmed, bool):
      raise ControllerDriverError("cleanup_confirmed must be boolean")

  @property
  def state(self) -> ControllerState:
    """Compatibility alias for the controller state snapshot."""

    return self.controller_state

  @property
  def dispatch(self) -> DispatchOutcome:
    """Return the dispatch outcome."""

    return self.dispatch_outcome

  @property
  def execution(self) -> ExecutionOutcome:
    """Return the execution outcome."""

    return self.execution_outcome

  @property
  def dispatched(self) -> bool:
    """Whether the sink was called."""

    return self.dispatch_outcome is DispatchOutcome.DISPATCHED

  @property
  def accepted(self) -> bool | None:
    """Return acceptance, or ``None`` when the sink was not called."""

    if self.execution_outcome is ExecutionOutcome.ACCEPTED:
      return True
    if self.execution_outcome is ExecutionOutcome.NOT_ATTEMPTED:
      return None
    return False

  @property
  def runtime_status(self) -> RuntimeStatus | None:
    """Return the runtime status when the controller produced a step."""

    return None if self.runtime_step is None else self.runtime_step.status

  @property
  def warning_events(self) -> tuple[SchedulerEvent, ...]:
    """Return scheduler events without using process-global warnings."""

    return tuple(
      event for event in self.scheduler_events if event.status.value.endswith("warning")
    )


class ControllerProtocol(Protocol):
  """Minimum controller surface consumed by the execution bridge."""

  @property
  def state(self) -> ControllerState: ...

  @property
  def runtime(self) -> object | None: ...

  @property
  def cleanup_confirmed(self) -> bool: ...

  def warmup(
    self,
    observation: ControllerObservation,
    *,
    task: object | None = None,
    task_revision: str | None = None,
    decoder_calls: int = 3,
    timeout_seconds: float = 10.0,
  ) -> WarmupReport: ...

  def request_activation(self) -> None: ...

  def step(
    self,
    observation: ControllerObservation,
    *,
    task: object | None = None,
    task_revision: str | None = None,
  ) -> RuntimeStep: ...

  def acknowledge(self, step: RuntimeStep, *, success: bool) -> bool: ...

  def fault(self, detail: str) -> None: ...

  def reset(self) -> None: ...

  def switch_policy(self, policy_id: str) -> None: ...

  def release(self) -> None: ...

  def shutdown(self) -> None: ...


ReadObservation = Callable[[], ControllerObservation]
ExecuteAction = Callable[[ActionCommand], object]


class D5ControllerDriver:
  """Bridge decoded runtime actions to one injected execution boundary."""

  def __init__(
    self,
    controller: ControllerProtocol,
    read_observation: ReadObservation,
    execute_action: ExecuteAction,
  ) -> None:
    if not callable(getattr(controller, "step", None)):
      raise ControllerDriverError("controller must expose step")
    if not callable(read_observation):
      raise ControllerDriverError("read_observation must be callable")
    if not callable(execute_action):
      raise ControllerDriverError("execute_action must be callable")
    self.controller = controller
    self._read_observation = read_observation
    self._execute_action = execute_action
    self._operation_active = False
    self._dispatch_generation = 0

  def warmup(
    self,
    *,
    task: object | None = None,
    task_revision: str | None = None,
    decoder_calls: int = 3,
    timeout_seconds: float = 10.0,
  ) -> WarmupReport:
    """Read one observation and warm the controller without sending an action."""

    self._enter_operation("warmup")
    try:
      observation = self._read_observation_checked()
      return self.controller.warmup(
        observation,
        task=task,
        task_revision=task_revision,
        decoder_calls=decoder_calls,
        timeout_seconds=timeout_seconds,
      )
    finally:
      self._operation_active = False

  def request_activation(self) -> None:
    """Delegate explicit activation intent to the lifecycle owner."""

    self._enter_operation("request_activation")
    try:
      self.controller.request_activation()
    finally:
      self._operation_active = False

  def tick(
    self,
    *,
    task: object | None = None,
    task_revision: str | None = None,
  ) -> ControllerDispatch:
    """Read, step and synchronously dispatch at most one current action."""

    if self._operation_active:
      return self._suppressed_dispatch("tick reentered during an active callback")
    self._operation_active = True
    try:
      return self._tick_impl(task=task, task_revision=task_revision)
    finally:
      self._operation_active = False

  def reset(self) -> None:
    """Delegate reset and invalidate any driver-side dispatch context."""

    self._dispatch_generation += 1
    self.controller.reset()

  def switch_policy(self, policy_id: str) -> None:
    """Delegate policy switching and invalidate driver-side dispatch context."""

    self._dispatch_generation += 1
    self.controller.switch_policy(policy_id)

  def fault(self, detail: str) -> None:
    """Delegate fail-closed fault handling to the lifecycle owner."""

    self._dispatch_generation += 1
    self.controller.fault(detail)

  def release(self) -> None:
    """Delegate terminal release and invalidate driver-side dispatch context."""

    self._dispatch_generation += 1
    self.controller.release()

  def shutdown(self) -> None:
    """Delegate terminal shutdown and invalidate driver-side dispatch context."""

    self._dispatch_generation += 1
    self.controller.shutdown()

  def _tick_impl(
    self, *, task: object | None, task_revision: str | None
  ) -> ControllerDispatch:
    try:
      observation = self._read_observation_checked()
    except BaseException as exc:
      detail = f"observation callback failed: {type(exc).__name__}: {exc}"
      self._fault_safely(detail)
      return self._dispatch_without_action(detail)

    try:
      step = self.controller.step(
        observation,
        task=task,
        task_revision=task_revision,
      )
    except BaseException as exc:
      return self._dispatch_without_action(
        f"controller step failed: {type(exc).__name__}: {exc}"
      )
    if not isinstance(step, RuntimeStep):
      detail = "controller step returned an invalid RuntimeStep"
      self._fault_safely(detail)
      return self._dispatch_without_action(detail)
    try:
      scheduler_events = _events(step)
    except ControllerDriverError as exc:
      self._fault_safely(str(exc))
      return self._dispatch_without_action(str(exc), step=step)

    state = self._controller_state()
    if step.status is not RuntimeStatus.ACTION:
      return self._dispatch_without_action(
        step.detail or f"runtime returned {step.status.value}",
        step=step,
        scheduler_events=scheduler_events,
        state=state,
      )
    if state is not ControllerState.ACTIVE:
      return self._dispatch_without_action(
        "runtime returned ACTION while controller was not ACTIVE",
        step=step,
        scheduler_events=scheduler_events,
        state=state,
      )

    try:
      if step.action is None or step.execution_id is None or step.provenance is None:
        raise ControllerDriverError("ACTION RuntimeStep is missing command fields")
      command = ActionCommand(
        action=step.action,
        execution_id=step.execution_id,
        tick=step.control_tick,
        provenance=step.provenance,
      )
    except (ControllerDriverError, TypeError, ValueError) as exc:
      detail = f"runtime action cannot be dispatched: {exc}"
      self._fault_safely(detail)
      return self._dispatch_without_action(
        detail,
        step=step,
        scheduler_events=scheduler_events,
      )

    runtime = self._controller_runtime()
    generation = self._dispatch_generation
    try:
      receipt = self._execute_action(command)
    except BaseException as exc:
      live = self._context_is_live(generation, runtime)
      if live:
        self._fault_safely(f"execution callback failed: {type(exc).__name__}: {exc}")
      state = self._controller_state()
      return self._dispatch_sent(
        command,
        step,
        ExecutionOutcome.FAILED if live else ExecutionOutcome.SUPPRESSED,
        f"execution callback failed: {type(exc).__name__}: {exc}",
        scheduler_events,
        state,
      )

    if not self._context_is_live(generation, runtime):
      return self._dispatch_sent(
        command,
        step,
        ExecutionOutcome.SUPPRESSED,
        "execution callback changed controller lifecycle; acknowledgement suppressed",
        scheduler_events,
        self._controller_state(),
      )
    if type(receipt) is not bool:
      detail = "execution callback must return the literal bool type"
      self._fault_safely(detail)
      return self._dispatch_sent(
        command,
        step,
        ExecutionOutcome.FAILED,
        detail,
        scheduler_events,
        self._controller_state(),
      )
    if not receipt:
      detail = "execution callback rejected the action"
      self._fault_safely(detail)
      return self._dispatch_sent(
        command,
        step,
        ExecutionOutcome.REJECTED,
        detail,
        scheduler_events,
        self._controller_state(),
      )

    # Check the lifecycle boundary immediately before acknowledgement.  The
    # exact RuntimeStep is retained; no reconstructed step can enter history.
    if not self._context_is_live(generation, runtime):
      return self._dispatch_sent(
        command,
        step,
        ExecutionOutcome.SUPPRESSED,
        "execution callback changed controller lifecycle; acknowledgement suppressed",
        scheduler_events,
        self._controller_state(),
      )
    try:
      acknowledged = self.controller.acknowledge(step, success=True)
    except BaseException as exc:
      self._fault_safely(
        f"execution acknowledgement failed: {type(exc).__name__}: {exc}"
      )
      return self._dispatch_sent(
        command,
        step,
        ExecutionOutcome.FAILED,
        f"execution acknowledgement failed: {type(exc).__name__}: {exc}",
        scheduler_events,
        self._controller_state(),
      )
    if type(acknowledged) is not bool:
      detail = "controller acknowledgement must return the literal bool type"
      self._fault_safely(detail)
      return self._dispatch_sent(
        command,
        step,
        ExecutionOutcome.FAILED,
        detail,
        scheduler_events,
        self._controller_state(),
      )
    if not acknowledged:
      detail = "controller rejected the successful execution acknowledgement"
      self._fault_safely(detail)
      return self._dispatch_sent(
        command,
        step,
        ExecutionOutcome.FAILED,
        detail,
        scheduler_events,
        self._controller_state(),
      )
    return self._dispatch_sent(
      command,
      step,
      ExecutionOutcome.ACCEPTED,
      "execution accepted and runtime history acknowledged",
      scheduler_events,
      self._controller_state(),
    )

  def _read_observation_checked(self) -> ControllerObservation:
    try:
      observation = self._read_observation()
    except BaseException:
      raise
    if not isinstance(observation, ControllerObservation):
      raise ControllerDriverError("read_observation must return ControllerObservation")
    return observation

  def _controller_state(self) -> ControllerState:
    try:
      state = self.controller.state
    except BaseException:
      return ControllerState.FAULT
    return state if isinstance(state, ControllerState) else ControllerState.FAULT

  def _controller_runtime(self) -> object:
    try:
      return getattr(self.controller, "runtime", _MISSING)
    except BaseException:
      return _MISSING

  def _context_is_live(self, generation: int, runtime: object) -> bool:
    if generation != self._dispatch_generation:
      return False
    if self._controller_state() is not ControllerState.ACTIVE:
      return False
    return self._controller_runtime() is runtime

  def _fault_safely(self, detail: str) -> None:
    self._dispatch_generation += 1
    try:
      self.controller.fault(detail)
    except BaseException:
      # The original dispatch failure remains the authoritative detail.  A
      # cleanup marker is reported from the controller snapshot instead.
      return

  def _dispatch_without_action(
    self,
    detail: str,
    *,
    step: RuntimeStep | None = None,
    scheduler_events: tuple[SchedulerEvent, ...] = (),
    state: ControllerState | None = None,
  ) -> ControllerDispatch:
    return ControllerDispatch(
      controller_state=self._controller_state() if state is None else state,
      runtime_step=step,
      command=None,
      dispatch_outcome=DispatchOutcome.NOT_DISPATCHED,
      execution_outcome=ExecutionOutcome.NOT_ATTEMPTED,
      detail=detail,
      cleanup_confirmed=_cleanup_confirmed(self.controller),
      scheduler_events=scheduler_events,
    )

  def _dispatch_sent(
    self,
    command: ActionCommand,
    step: RuntimeStep,
    outcome: ExecutionOutcome,
    detail: str,
    scheduler_events: tuple[SchedulerEvent, ...],
    state: ControllerState,
  ) -> ControllerDispatch:
    return ControllerDispatch(
      controller_state=state,
      runtime_step=step,
      command=command,
      dispatch_outcome=DispatchOutcome.DISPATCHED,
      execution_outcome=outcome,
      detail=detail,
      cleanup_confirmed=_cleanup_confirmed(self.controller),
      scheduler_events=scheduler_events,
    )

  def _suppressed_dispatch(self, detail: str) -> ControllerDispatch:
    return self._dispatch_without_action(detail)

  def _enter_operation(self, operation: str) -> None:
    if self._operation_active:
      raise ControllerDriverError(f"{operation} is not reentrant")
    self._operation_active = True


def _model_runtime_factory(
  spec: ModelBackendSpec,
  generator: torch.Generator,
  *,
  max_fallback_ticks: int,
  max_source_age_seconds: float,
  max_future_skew_seconds: float,
  expected_source_frame: int | None,
) -> Callable[[Callable[[], float]], AsyncLatentRuntime]:
  def factory(clock: Callable[[], float]) -> AsyncLatentRuntime:
    return build_model_backed_runtime(
      spec,
      generator=generator,
      clock=clock,
      max_fallback_ticks=max_fallback_ticks,
      max_source_age_seconds=max_source_age_seconds,
      max_future_skew_seconds=max_future_skew_seconds,
      expected_source_frame=expected_source_frame,
    )

  return factory


def build_model_backed_controller_driver(
  spec: ModelBackendSpec,
  *,
  generator: torch.Generator,
  read_observation: ReadObservation,
  execute_action: ExecuteAction,
  clock: Callable[[], float] = time.monotonic,
  max_observation_age_seconds: float = 0.5,
  max_future_skew_seconds: float = 0.0,
  max_fallback_ticks: int = 4,
  max_source_age_seconds: float = 0.5,
  expected_source_frame: int | None = None,
) -> D5ControllerDriver:
  """Assemble a controller driver around explicit model-backed runtime paths.

  The runtime is built only when the returned controller is loaded.  The
  supplied generator is passed to :func:`build_model_backed_runtime`; that
  helper loads the decoder in the owner and creates the denoiser worker from
  the explicit specification.  The lifecycle controller supplies the exact
  same clock object to both sides of that factory boundary.
  """

  if not isinstance(spec, ModelBackendSpec):
    raise ControllerDriverError("spec must be a ModelBackendSpec")
  if not isinstance(generator, torch.Generator):
    raise ControllerDriverError("generator must be a torch.Generator")
  runtime_factory = _model_runtime_factory(
    spec,
    generator,
    max_fallback_ticks=max_fallback_ticks,
    max_source_age_seconds=max_source_age_seconds,
    max_future_skew_seconds=max_future_skew_seconds,
    expected_source_frame=expected_source_frame,
  )
  controller = D5Controller(
    runtime_factory,
    clock=clock,
    max_observation_age_seconds=max_observation_age_seconds,
    max_future_skew_seconds=max_future_skew_seconds,
  )
  return D5ControllerDriver(controller, read_observation, execute_action)


# The shorter spelling is convenient for callers while the longer spelling
# makes the assembled lifecycle object explicit in source searches.
build_model_backed_driver = build_model_backed_controller_driver


__all__ = [
  "ActionCommand",
  "ControllerDispatch",
  "ControllerDriverError",
  "ControllerProtocol",
  "DispatchOutcome",
  "D5ControllerDriver",
  "ExecuteAction",
  "ExecutionOutcome",
  "ReadObservation",
  "build_model_backed_controller_driver",
  "build_model_backed_driver",
]
