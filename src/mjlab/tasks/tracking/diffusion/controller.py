"""Synchronous lifecycle owner for the D5 asynchronous latent runtime.

The controller owns one already-assembled :class:`AsyncLatentRuntime`.  It
accepts only measured observations and exposes runtime steps to a later
execution bridge.  It never publishes an action, creates a standing fallback,
or invents a safe hold.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import numpy as np

from .async_latent_runtime import (
  AsyncLatentRuntime,
  RuntimeStatus,
  RuntimeStep,
  WarmupReport,
)
from .latent_planning import LatentPlanningError, TaskSnapshot
from .state import WorldState

_CONDITIONING_DIM = 99
_MISSING = object()


class ControllerState(str, Enum):
  """Explicit owner-side D5 lifecycle state."""

  LOADING = "loading"
  BOOTSTRAPPING = "bootstrapping"
  WARMING = "warming"
  READY = "ready"
  ACTIVE = "active"
  FAULT = "fault"
  RELEASED = "released"
  SHUTDOWN = "shutdown"


class ControllerError(ValueError):
  """A controller lifecycle, measurement, or runtime ownership violation."""


def _finite(value: object, name: str) -> float:
  if isinstance(value, (bool, np.bool_)):
    raise ControllerError(f"{name} must be a finite scalar")
  try:
    scalar = float(cast(Any, value))
  except (TypeError, ValueError, OverflowError) as exc:
    raise ControllerError(f"{name} must be a finite scalar") from exc
  if not np.isfinite(scalar):
    raise ControllerError(f"{name} must be finite")
  return scalar


def _nonnegative_int(value: object, name: str) -> int:
  if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
    raise ControllerError(f"{name} must be a non-negative integer")
  result = int(value)
  if result < 0:
    raise ControllerError(f"{name} must be a non-negative integer")
  return result


def _identity(value: object, name: str) -> str:
  if not isinstance(value, str) or not value.strip():
    raise ControllerError(f"{name} must be a non-empty string")
  return value


def _copy_state(state: object) -> WorldState:
  if not isinstance(state, WorldState) or state.root_position.ndim != 1:
    raise ControllerError("state must be one unbatched WorldState")
  return WorldState(
    **{
      name: np.array(getattr(state, name), dtype=np.float64, copy=True)
      for name in (
        "root_position",
        "root_quaternion_wxyz",
        "root_linear_velocity",
        "root_angular_velocity",
        "body_positions",
        "body_linear_velocities",
      )
    }
  )


def _copy_conditioning(value: object) -> np.ndarray:
  try:
    conditioning = np.asarray(value, dtype=np.float64)
  except (TypeError, ValueError, OverflowError) as exc:
    raise ControllerError("conditioning must be numeric") from exc
  if conditioning.shape != (_CONDITIONING_DIM,) or not np.isfinite(conditioning).all():
    raise ControllerError("conditioning must be finite with shape [99]")
  return np.array(conditioning, dtype=np.float64, copy=True)


@dataclass(frozen=True, slots=True)
class ControllerObservation:
  """One owned, measured observation in the controller clock domain.

  ``sample_id`` identifies the captured measurement.  It is separate from
  ``tick`` because a caller can capture a new sample without advancing a
  physical control step.  The controller still requires increasing execution
  ticks once normal stepping starts.
  """

  state: WorldState
  conditioning: Sequence[float] | np.ndarray
  tick: int
  sample_id: int
  source_frame: int
  segment_id: str
  timestamp: float

  def __post_init__(self) -> None:
    object.__setattr__(self, "state", _copy_state(self.state))
    conditioning = _copy_conditioning(self.conditioning)
    conditioning.setflags(write=False)
    object.__setattr__(self, "conditioning", conditioning)
    object.__setattr__(self, "tick", _nonnegative_int(self.tick, "tick"))
    object.__setattr__(self, "sample_id", _nonnegative_int(self.sample_id, "sample_id"))
    object.__setattr__(
      self, "source_frame", _nonnegative_int(self.source_frame, "source_frame")
    )
    object.__setattr__(self, "segment_id", _identity(self.segment_id, "segment_id"))
    timestamp = _finite(self.timestamp, "timestamp")
    if timestamp < 0.0:
      raise ControllerError("timestamp must be non-negative")
    object.__setattr__(self, "timestamp", timestamp)


class D5Controller:
  """Own one asynchronous D5 runtime through an explicit lifecycle.

  The factory receives the controller's injected monotonic clock as its only
  argument.  A zero-argument factory is also accepted when the returned
  runtime exposes that exact clock object.  This supports test factories while
  preserving one clock domain as an enforced ownership invariant.
  """

  def __init__(
    self,
    runtime_factory: Callable[..., Any],
    *,
    clock: Callable[[], float] = time.monotonic,
    max_observation_age_seconds: float = 0.5,
    max_future_skew_seconds: float = 0.0,
  ) -> None:
    if not callable(runtime_factory):
      raise ControllerError("runtime_factory must be callable")
    if not callable(clock):
      raise ControllerError("clock must be callable")
    max_age = _finite(max_observation_age_seconds, "max_observation_age_seconds")
    max_skew = _finite(max_future_skew_seconds, "max_future_skew_seconds")
    if max_age < 0.0 or max_skew < 0.0:
      raise ControllerError("observation-age limits must be non-negative")

    self._runtime_factory = runtime_factory
    self._clock = clock
    self._max_observation_age_seconds = max_age
    self._max_future_skew_seconds = max_skew
    self._state = ControllerState.LOADING
    self._runtime: AsyncLatentRuntime | None = None
    self._cleanup_confirmed = False
    self._ever_owned_runtime = False
    self._failure_detail: str | None = None
    self._callback_active = False
    self._activation_requested = False
    self._warmup_sample_id: int | None = None
    self._warmup_end_timestamp: float | None = None
    self._last_sample_id: int | None = None
    self._last_execution_tick: int | None = None
    self._warmup_tick: int | None = None
    self._segment_id: str | None = None
    self._source_frame: int | None = None
    self._task_snapshot: TaskSnapshot | None = None
    self._pending_step: RuntimeStep | None = None
    self._clock_last: float | None = None

  @property
  def state(self) -> ControllerState:
    """Return the current controller lifecycle state."""

    return self._state

  @property
  def runtime(self) -> AsyncLatentRuntime | None:
    """Return the owned runtime, or ``None`` before successful load/after cleanup."""

    return self._runtime

  @property
  def cleanup_confirmed(self) -> bool:
    """Return exact cleanup evidence for the owned runtime.

    Cleanup is true only when the runtime reports true and its executor exposes
    the exact markers ``closed is True`` and ``worker_alive is False``.  Missing
    or truthy-but-not-exact markers are not accepted.
    """

    runtime = self._runtime
    if runtime is None:
      return self._cleanup_confirmed
    return self._runtime_cleanup_confirmed(runtime)

  @property
  def failure_detail(self) -> str | None:
    """Return the most recent fail-closed detail, if any."""

    return self._failure_detail

  def load(self) -> None:
    """Construct and own one runtime, entering ``BOOTSTRAPPING``."""

    self._ensure_not_reentrant()
    if self._state is not ControllerState.LOADING:
      raise ControllerError(f"load is invalid in state {self._state.value}")
    if self._runtime is not None:
      raise ControllerError("load cannot replace an owned runtime")
    if self._ever_owned_runtime and not self._cleanup_confirmed:
      raise ControllerError("runtime cleanup is not confirmed; replacement is blocked")

    self._cleanup_confirmed = False
    runtime: Any = None
    try:
      runtime = self._invoke_factory()
      if runtime is not None:
        self._runtime = runtime
        self._ever_owned_runtime = True
      self._validate_runtime(runtime)
    except BaseException as exc:
      if self._runtime is not None:
        self._close_owned_runtime("shutdown")
      self._state = ControllerState.FAULT
      self._failure_detail = f"runtime factory failed: {exc}"
      raise ControllerError(self._failure_detail) from exc
    self._ever_owned_runtime = True
    self._failure_detail = None
    self._state = ControllerState.BOOTSTRAPPING

  def acknowledge_bootstrap(
    self,
    state: WorldState,
    latent: Sequence[float] | np.ndarray,
    *,
    success: bool,
  ) -> bool:
    """Delegate one actual bootstrap execution acknowledgement."""

    self._ensure_not_reentrant()
    self._require_state(ControllerState.BOOTSTRAPPING, "acknowledge_bootstrap")
    if not isinstance(success, bool):
      raise ControllerError("bootstrap success must be boolean")
    checked_state = _copy_state(state)
    try:
      checked_latent = np.asarray(latent, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
      raise ControllerError("bootstrap latent must be numeric") from exc
    if checked_latent.shape != (32,) or not np.isfinite(checked_latent).all():
      raise ControllerError("bootstrap latent must be finite with shape [32]")
    try:
      result = self._runtime_call(
        "acknowledge_bootstrap", checked_state, checked_latent, success=success
      )
    except BaseException as exc:
      self._fault_internal(f"bootstrap acknowledgement failed: {exc}")
      raise ControllerError(self._failure_detail or str(exc)) from exc
    if not isinstance(result, bool):
      self._fault_internal("bootstrap acknowledgement returned a non-boolean result")
      raise ControllerError(self._failure_detail or "invalid bootstrap acknowledgement")
    return result

  def warmup(
    self,
    observation: ControllerObservation,
    *,
    task: object | None = None,
    task_revision: str | None = None,
    decoder_calls: int = 3,
    timeout_seconds: float = 10.0,
  ) -> WarmupReport:
    """Warm the real runtime and discard all warmup output."""

    self._ensure_not_reentrant()
    self._require_state(ControllerState.BOOTSTRAPPING, "warmup")
    runtime = self._require_runtime()
    if not self._runtime_history_ready(runtime):
      raise ControllerError("warmup requires eight acknowledged bootstrap pairs")
    checked = self._checked_observation(observation, require_new=False)
    snapshot = self._resolve_task_snapshot(task, task_revision)

    self._state = ControllerState.WARMING
    try:
      report = self._runtime_call(
        "warmup",
        checked.state,
        checked.conditioning,
        tick=checked.tick,
        source_frame=checked.source_frame,
        task=task,
        task_revision=task_revision,
        decoder_calls=decoder_calls,
        timeout_seconds=timeout_seconds,
      )
      if not isinstance(report, WarmupReport):
        raise ControllerError("runtime warmup returned an invalid report")
      end_timestamp = self._clock_value()
    except BaseException as exc:
      self._fault_internal(f"runtime warmup failed: {exc}")
      raise ControllerError(self._failure_detail or str(exc)) from exc

    self._warmup_sample_id = checked.sample_id
    self._warmup_end_timestamp = end_timestamp
    self._last_sample_id = checked.sample_id
    self._warmup_tick = checked.tick
    self._last_execution_tick = None
    self._segment_id = checked.segment_id
    self._source_frame = checked.source_frame
    self._task_snapshot = snapshot
    self._activation_requested = False
    self._pending_step = None
    self._failure_detail = None
    self._state = ControllerState.READY
    return report

  def request_activation(self) -> None:
    """Arm explicit activation without publishing or creating an action."""

    self._ensure_not_reentrant()
    self._require_state(ControllerState.READY, "request_activation")
    if self._warmup_sample_id is None or self._warmup_end_timestamp is None:
      raise ControllerError("activation requires successful warmup")
    self._activation_requested = True

  def step(
    self,
    observation: ControllerObservation,
    *,
    task: object | None = None,
    task_revision: str | None = None,
  ) -> RuntimeStep:
    """Process one fresh observation without waiting or publishing."""

    self._ensure_not_reentrant()
    if self._state not in {ControllerState.READY, ControllerState.ACTIVE}:
      raise ControllerError(f"step is invalid in state {self._state.value}")
    if not self._activation_requested:
      raise ControllerError("explicit activation is required before step")
    try:
      checked = self._checked_observation(observation, require_new=True)
      snapshot = self._resolve_task_snapshot(task, task_revision)
    except ControllerError as exc:
      if "segment_id changed" in str(exc) or "source_frame changed" in str(exc):
        self._fault_internal(f"measurement continuity failed: {exc}")
      raise

    # The runtime compares the full immutable TaskSnapshot, not only its
    # revision string.  Match that boundary so a changed mode, target, strength
    # or metadata with an unchanged revision still discards the pending action
    # here before the runtime invalidates its own pending execution.
    if self._task_snapshot is not None and snapshot != self._task_snapshot:
      self._invalidate_pending_task_work()

    # Consume the identity only after all input checks pass.  Even a waiting
    # runtime result therefore cannot be retried with the same measurement.
    self._last_sample_id = checked.sample_id
    self._last_execution_tick = checked.tick
    self._task_snapshot = snapshot
    try:
      result = self._runtime_call(
        "step",
        checked.state,
        checked.conditioning,
        tick=checked.tick,
        source_frame=checked.source_frame,
        task=task,
        task_revision=task_revision,
      )
    except BaseException as exc:
      self._fault_internal(f"runtime step failed: {exc}")
      return RuntimeStep(
        status=RuntimeStatus.PLANNER_FAILURE,
        control_tick=checked.tick,
        detail=self._failure_detail or str(exc),
      )

    if not isinstance(result, RuntimeStep):
      self._fault_internal("runtime step returned an invalid result")
      return RuntimeStep(
        status=RuntimeStatus.INVALID_INPUT,
        control_tick=checked.tick,
        detail=self._failure_detail or "invalid runtime step",
      )

    if not isinstance(result.status, RuntimeStatus):
      self._fault_internal("runtime step returned an unknown status")
      return RuntimeStep(
        status=RuntimeStatus.INVALID_INPUT,
        control_tick=checked.tick,
        detail=self._failure_detail or "unknown runtime status",
      )

    if result.status is RuntimeStatus.ACTION:
      provenance = result.provenance
      if provenance is None or provenance.execution_tick != checked.tick:
        self._fault_internal("runtime action execution tick does not match observation")
        return RuntimeStep(
          status=RuntimeStatus.INVALID_PLAN,
          control_tick=checked.tick,
          detail=self._failure_detail or "action tick mismatch",
        )
      if provenance.source_frame != checked.source_frame:
        self._fault_internal("runtime action source frame does not match observation")
        return RuntimeStep(
          status=RuntimeStatus.INVALID_PLAN,
          control_tick=checked.tick,
          detail=self._failure_detail or "action frame mismatch",
        )
      if result.control_tick != checked.tick:
        self._fault_internal("runtime action control tick does not match observation")
        return RuntimeStep(
          status=RuntimeStatus.INVALID_PLAN,
          control_tick=checked.tick,
          detail=self._failure_detail or "action control tick mismatch",
        )
      if self._pending_step is not None:
        self._fault_internal(
          "runtime returned an action while another action was pending"
        )
        return RuntimeStep(
          status=RuntimeStatus.INVALID_PLAN,
          control_tick=checked.tick,
          detail=self._failure_detail or "multiple pending actions",
        )
      self._pending_step = result
      self._state = ControllerState.ACTIVE
      return result

    if result.status in {
      RuntimeStatus.PLANNER_FAILURE,
      RuntimeStatus.DECODE_FAILURE,
      RuntimeStatus.INVALID_PLAN,
      RuntimeStatus.INVALID_INPUT,
    }:
      self._fault_internal(
        f"runtime step failed: {result.detail or result.status.value}"
      )
    return result

  def acknowledge(self, step: RuntimeStep, *, success: bool) -> bool:
    """Acknowledge exactly the pending action and commit only on success."""

    self._ensure_not_reentrant()
    if self._state is not ControllerState.ACTIVE:
      raise ControllerError(f"acknowledge is invalid in state {self._state.value}")
    pending = self._pending_step
    if pending is None or step is not pending:
      raise ControllerError("acknowledgement must match the exact pending RuntimeStep")
    if not isinstance(success, bool):
      raise ControllerError("execution success must be boolean")

    try:
      result = self._runtime_call("acknowledge", pending, success=success)
    except BaseException as exc:
      self._pending_step = None
      self._fault_internal(f"execution acknowledgement failed: {exc}")
      raise ControllerError(self._failure_detail or str(exc)) from exc
    self._pending_step = None
    if not isinstance(result, bool):
      self._fault_internal("execution acknowledgement returned a non-boolean result")
      return False
    if not result:
      self._fault_internal("execution acknowledgement was rejected")
      return False
    return True

  def fault(self, detail: str) -> None:
    """Fail closed and clean the exact owned runtime."""

    self._ensure_not_reentrant()
    if self._state in {ControllerState.RELEASED, ControllerState.SHUTDOWN}:
      return
    self._fault_internal(_identity(detail, "fault detail"))

  def reset(self) -> None:
    """Close the current runtime and require a new explicit load."""

    self._ensure_not_reentrant()
    if self._state in {ControllerState.RELEASED, ControllerState.SHUTDOWN}:
      raise ControllerError(f"reset is invalid in state {self._state.value}")
    runtime = self._runtime
    if runtime is None:
      if self._ever_owned_runtime and not self._cleanup_confirmed:
        raise ControllerError("runtime cleanup is not confirmed; reset is blocked")
      self._clear_session()
      self._state = ControllerState.LOADING
      return
    lifecycle_error: BaseException | None = None
    try:
      self._runtime_call("reset")
    except BaseException as exc:
      lifecycle_error = exc
    closed = self._close_owned_runtime("shutdown")
    if lifecycle_error is not None or not closed:
      detail = (
        "runtime reset failed"
        if lifecycle_error is not None
        else "runtime cleanup is not confirmed"
      )
      if lifecycle_error is not None:
        detail = f"{detail}: {lifecycle_error}"
      self._state = ControllerState.FAULT
      self._failure_detail = detail
      return
    self._clear_session()
    self._state = ControllerState.LOADING

  def switch_policy(self, policy_id: str) -> None:
    """Invalidate policy work, close the old runtime, and require a new load."""

    self._ensure_not_reentrant()
    policy = _identity(policy_id, "policy_id")
    if self._state in {ControllerState.RELEASED, ControllerState.SHUTDOWN}:
      raise ControllerError(f"switch_policy is invalid in state {self._state.value}")
    runtime = self._runtime
    if runtime is None:
      if self._ever_owned_runtime and not self._cleanup_confirmed:
        raise ControllerError(
          "runtime cleanup is not confirmed; policy switch is blocked"
        )
      self._clear_session()
      self._state = ControllerState.LOADING
      return
    lifecycle_error: BaseException | None = None
    try:
      self._runtime_call("switch_policy", policy)
    except BaseException as exc:
      lifecycle_error = exc
    closed = self._close_owned_runtime("shutdown")
    if lifecycle_error is not None or not closed:
      detail = (
        "runtime policy switch failed"
        if lifecycle_error is not None
        else "runtime cleanup is not confirmed"
      )
      if lifecycle_error is not None:
        detail = f"{detail}: {lifecycle_error}"
      self._state = ControllerState.FAULT
      self._failure_detail = detail
      return
    self._clear_session()
    self._state = ControllerState.LOADING

  def release(self) -> None:
    """Permanently release the owned runtime and enter ``RELEASED``."""

    self._ensure_not_reentrant()
    if self._state is ControllerState.RELEASED:
      return
    if self._state is ControllerState.SHUTDOWN:
      return
    self._pending_step = None
    self._activation_requested = False
    closed = self._close_owned_runtime("release")
    self._state = ControllerState.RELEASED
    if not closed:
      self._failure_detail = "runtime release cleanup is not confirmed"

  def shutdown(self) -> None:
    """Permanently shut down the owned runtime and enter ``SHUTDOWN``."""

    self._ensure_not_reentrant()
    if self._state is ControllerState.SHUTDOWN:
      return
    if self._state is ControllerState.RELEASED:
      return
    self._pending_step = None
    self._activation_requested = False
    closed = self._close_owned_runtime("shutdown")
    self._state = ControllerState.SHUTDOWN
    if not closed:
      self._failure_detail = "runtime shutdown cleanup is not confirmed"

  def _ensure_not_reentrant(self) -> None:
    if self._callback_active:
      raise ControllerError(
        "lifecycle mutation is not allowed during a runtime callback"
      )

  def _require_state(self, expected: ControllerState, operation: str) -> None:
    if self._state is not expected:
      raise ControllerError(f"{operation} is invalid in state {self._state.value}")

  def _require_runtime(self) -> AsyncLatentRuntime:
    runtime = self._runtime
    if runtime is None:
      raise ControllerError("no runtime is loaded")
    return runtime

  def _invoke_factory(self) -> Any:
    self._callback_active = True
    try:
      factory = self._runtime_factory
      try:
        signature = inspect.signature(factory)
      except (TypeError, ValueError):
        return factory(self._clock)
      parameters = tuple(signature.parameters.values())
      accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
      )
      clock_parameter = signature.parameters.get("clock")
      if accepts_kwargs or (
        clock_parameter is not None
        and clock_parameter.kind
        in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
      ):
        return factory(clock=self._clock)
      positional = tuple(
        parameter
        for parameter in parameters
        if parameter.kind
        in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
      )
      if positional or any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters
      ):
        return factory(self._clock)
      return factory()
    finally:
      self._callback_active = False

  def _validate_runtime(self, runtime: object) -> None:
    if runtime is None:
      raise ControllerError("runtime factory returned None")
    required_methods = (
      "acknowledge_bootstrap",
      "warmup",
      "step",
      "acknowledge",
      "reset",
      "switch_policy",
      "release",
      "shutdown",
    )
    for name in required_methods:
      if not callable(getattr(runtime, name, None)):
        raise ControllerError(f"runtime is missing callable {name}")
    runtime_clock = getattr(runtime, "clock", _MISSING)
    if runtime_clock is _MISSING or runtime_clock is not self._clock:
      raise ControllerError("runtime must use the controller's exact monotonic clock")
    if not hasattr(runtime, "history_ready"):
      raise ControllerError("runtime must expose history_ready")

  def _runtime_call(self, name: str, *args: object, **kwargs: object) -> Any:
    runtime = self._require_runtime()
    if self._callback_active:
      raise ControllerError("runtime callback reentrancy is not allowed")
    method = getattr(runtime, name, None)
    if not callable(method):
      raise ControllerError(f"runtime is missing callable {name}")
    self._callback_active = True
    try:
      return method(*args, **kwargs)
    finally:
      self._callback_active = False

  def _runtime_history_ready(self, runtime: AsyncLatentRuntime) -> bool:
    try:
      value = runtime.history_ready
    except BaseException as exc:
      raise ControllerError(
        f"could not inspect runtime bootstrap state: {exc}"
      ) from exc
    if not isinstance(value, bool):
      raise ControllerError("runtime history_ready must be boolean")
    return value

  def _runtime_cleanup_confirmed(self, runtime: AsyncLatentRuntime) -> bool:
    try:
      marker = getattr(runtime, "cleanup_confirmed", _MISSING)
      if marker is not True:
        return False
      closed = getattr(runtime, "closed", _MISSING)
      alive = getattr(runtime, "worker_alive", _MISSING)
      if closed is _MISSING or alive is _MISSING:
        executor = getattr(runtime, "executor", _MISSING)
        if executor is _MISSING:
          return False
        closed = getattr(executor, "closed", _MISSING)
        alive = getattr(executor, "worker_alive", _MISSING)
      return closed is True and alive is False
    except BaseException:
      return False

  def _close_owned_runtime(self, method_name: str) -> bool:
    runtime = self._runtime
    if runtime is None:
      return self._cleanup_confirmed
    self._pending_step = None
    try:
      self._runtime_call(method_name)
    except BaseException as exc:
      self._failure_detail = f"runtime {method_name} failed: {exc}"
    confirmed = self._runtime_cleanup_confirmed(runtime)
    self._cleanup_confirmed = confirmed
    if confirmed:
      self._runtime = None
    return confirmed

  def _clear_session(self) -> None:
    self._activation_requested = False
    self._warmup_sample_id = None
    self._warmup_end_timestamp = None
    self._last_sample_id = None
    self._last_execution_tick = None
    self._warmup_tick = None
    self._segment_id = None
    self._source_frame = None
    self._task_snapshot = None
    self._pending_step = None
    self._failure_detail = None

  def _fault_internal(self, detail: str) -> None:
    if self._state in {ControllerState.RELEASED, ControllerState.SHUTDOWN}:
      return
    self._state = ControllerState.FAULT
    self._activation_requested = False
    self._pending_step = None
    self._failure_detail = detail
    if self._runtime is not None:
      confirmed = self._close_owned_runtime("shutdown")
      if not confirmed:
        self._failure_detail = f"{detail}; runtime cleanup is not confirmed"

  def _clock_value(self) -> float:
    value = _finite(self._clock(), "monotonic clock value")
    if value < 0.0:
      raise ControllerError("monotonic clock value must be non-negative")
    if self._clock_last is not None and value < self._clock_last:
      raise ControllerError("monotonic clock regressed")
    self._clock_last = value
    return value

  def _checked_observation(
    self, observation: ControllerObservation, *, require_new: bool
  ) -> ControllerObservation:
    if not isinstance(observation, ControllerObservation):
      raise ControllerError("observation must be a ControllerObservation")
    now = self._clock_value()
    age = now - observation.timestamp
    if age > self._max_observation_age_seconds:
      raise ControllerError("observation is older than the configured age limit")
    if age < -self._max_future_skew_seconds:
      raise ControllerError("observation timestamp is in the future")
    if self._segment_id is not None and observation.segment_id != self._segment_id:
      raise ControllerError("observation segment_id changed without reset")
    if (
      self._source_frame is not None and observation.source_frame != self._source_frame
    ):
      raise ControllerError("observation source_frame changed without reset")
    if require_new:
      if self._last_sample_id is None or observation.sample_id <= self._last_sample_id:
        raise ControllerError("observation sample_id must be strictly new")
      if self._last_execution_tick is not None:
        if observation.tick <= self._last_execution_tick:
          raise ControllerError("execution tick must be strictly increasing")
      elif self._warmup_tick is not None and observation.tick < self._warmup_tick:
        raise ControllerError("execution tick regressed relative to warmup")
      if (
        self._warmup_end_timestamp is not None
        and observation.timestamp < self._warmup_end_timestamp
      ):
        raise ControllerError("observation was captured before warmup ended")
    return observation

  def _resolve_task_snapshot(
    self, task: object | None, revision: str | None
  ) -> TaskSnapshot:
    try:
      return TaskSnapshot.from_inputs(task, revision=revision)
    except (LatentPlanningError, TypeError, ValueError) as exc:
      raise ControllerError(f"invalid task revision or task input: {exc}") from exc

  def _invalidate_pending_task_work(self) -> None:
    pending = self._pending_step
    if pending is None:
      return
    try:
      result = self._runtime_call("acknowledge", pending, success=False)
    except BaseException as exc:
      self._fault_internal(f"task change invalidation failed: {exc}")
      raise ControllerError(self._failure_detail or str(exc)) from exc
    self._pending_step = None
    if result is not False:
      self._fault_internal(
        "task change invalidation did not discard the pending action"
      )
      raise ControllerError(
        self._failure_detail or "pending task work was not discarded"
      )


__all__ = [
  "ControllerError",
  "ControllerObservation",
  "ControllerState",
  "D5Controller",
]
