"""Asynchronous raw-latent D5 runtime facade.

This module owns the narrow boundary between the D5 planner scheduler and the
X2 decoder.  Planning is asynchronous and uses the bounded executor supplied by
the caller.  The owner thread selects one raw 32-D latent for the current
control tick, decodes it with the current finite 99-D proprioception, and
returns one finite 31-D action.

The facade never decodes future rows, never treats raw latent vectors as robot
commands, and never records history before an execution acknowledgement.  The
clock is an injected monotonic clock.  Source timestamps and source-age checks
therefore stay in one explicit domain and do not use wall time.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from .contract import DEFAULT_CONTRACT, DiffusionContract
from .handover import HandoverHistory
from .latent_planning import (
  LATENT_PLANNING_SCHEMA,
  RAW_LATENT_REPRESENTATION,
  ROOT_FRAME,
  LatentPlanningError,
  TaskSnapshot,
  build_planning_packet,
)
from .projection import ProjectionBundle
from .scheduler import (
  AcceptedRequestProvenance,
  Plan,
  PlannerExecutor,
  RuntimeScheduler,
  SchedulerError,
  SchedulerEvent,
  SchedulerStatus,
)
from .state import WorldState

_CONDITIONING_DIM = 99


class AsyncLatentRuntimeError(ValueError):
  """An asynchronous latent runtime input or lifecycle value is invalid."""


class RuntimeStatus(str, Enum):
  """Explicit action or fail-closed status from one runtime tick."""

  ACTION = "action"
  WAITING_FOR_PLAN = "waiting_for_plan"
  BOOTSTRAP_REQUIRED = "bootstrap_required"
  AWAITING_EXECUTION_ACK = "awaiting_execution_ack"
  PLAN_NOT_YET_VALID = "plan_not_yet_valid"
  PLAN_EXHAUSTED = "plan_exhausted"
  SOURCE_TOO_OLD = "source_too_old"
  SOURCE_IN_FUTURE = "source_in_future"
  INVALID_PLAN = "invalid_plan"
  PLANNER_FAILURE = "planner_failure"
  DECODE_FAILURE = "decode_failure"
  INVALID_INPUT = "invalid_input"
  RESET = "reset"
  RELEASED = "released"
  SHUTDOWN = "shutdown"


def _finite(value: object, name: str) -> float:
  if isinstance(value, (bool, np.bool_)):
    raise AsyncLatentRuntimeError(f"{name} must be a finite scalar")
  try:
    scalar = float(cast(Any, value))
  except (TypeError, ValueError, OverflowError) as exc:
    raise AsyncLatentRuntimeError(f"{name} must be a finite scalar") from exc
  if not np.isfinite(scalar):
    raise AsyncLatentRuntimeError(f"{name} must be finite")
  return scalar


def _nonnegative_int(value: object, name: str) -> int:
  if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
    raise AsyncLatentRuntimeError(f"{name} must be a non-negative integer")
  result = int(value)
  if result < 0:
    raise AsyncLatentRuntimeError(f"{name} must be a non-negative integer")
  return result


def _identity(value: object, name: str) -> str:
  if not isinstance(value, str) or not value.strip():
    raise AsyncLatentRuntimeError(f"{name} must be a non-empty string")
  return value


def _copy_state(state: WorldState) -> WorldState:
  if not isinstance(state, WorldState) or state.root_position.ndim != 1:
    raise AsyncLatentRuntimeError("current state must be one unbatched WorldState")
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


def _copy_vector(value: object, name: str, width: int) -> np.ndarray:
  try:
    array = np.asarray(value, dtype=np.float64)
  except (TypeError, ValueError, OverflowError) as exc:
    raise AsyncLatentRuntimeError(f"{name} must be numeric") from exc
  if array.shape != (width,) or not np.isfinite(array).all():
    raise AsyncLatentRuntimeError(f"{name} must be finite with shape [{width}]")
  return np.array(array, dtype=np.float64, copy=True)


def _released_decode(_latent: Tensor, _conditioning: Tensor) -> Tensor:
  raise AsyncLatentRuntimeError("decoder is released")


def _metadata_int(metadata: Mapping[str, object], name: str) -> int:
  value = metadata.get(name)
  return _nonnegative_int(value, f"plan metadata {name}")


def _metadata_float(metadata: Mapping[str, object], name: str) -> float:
  return _finite(metadata.get(name), f"plan metadata {name}")


@dataclass(frozen=True, slots=True)
class ExecutionProvenance:
  """Identity and timing facts for the one latent decoded at execution."""

  plan_sequence_id: int
  source_timestamp: float
  source_tick: int
  execution_tick: int
  valid_until_tick: int
  source_frame: int
  contract_identity: str
  vae_identity: str
  projection_identity: str
  model_identity: str
  task_revision: str
  task_mode: str
  representation: str = RAW_LATENT_REPRESENTATION

  def __post_init__(self) -> None:
    object.__setattr__(
      self,
      "plan_sequence_id",
      _nonnegative_int(self.plan_sequence_id, "plan_sequence_id"),
    )
    if self.plan_sequence_id == 0:
      raise AsyncLatentRuntimeError("plan_sequence_id must be positive")
    object.__setattr__(
      self, "source_timestamp", _finite(self.source_timestamp, "source_timestamp")
    )
    for name in ("source_tick", "execution_tick", "valid_until_tick", "source_frame"):
      object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
    if self.valid_until_tick < self.source_tick:
      raise AsyncLatentRuntimeError("plan valid_until_tick precedes source_tick")
    for name in (
      "contract_identity",
      "vae_identity",
      "projection_identity",
      "model_identity",
      "task_revision",
      "representation",
      "task_mode",
    ):
      object.__setattr__(self, name, _identity(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class RuntimeStep:
  """One decoded action or an explicit fail-closed no-action result."""

  status: RuntimeStatus
  control_tick: int
  action: np.ndarray | None = None
  executed_latent: np.ndarray | None = None
  provenance: ExecutionProvenance | None = None
  execution_id: int | None = None
  detail: str = ""
  scheduler_events: tuple[SchedulerEvent, ...] = ()

  def __post_init__(self) -> None:
    object.__setattr__(
      self, "control_tick", _nonnegative_int(self.control_tick, "control_tick")
    )
    events = tuple(self.scheduler_events)
    if not all(isinstance(event, SchedulerEvent) for event in events):
      raise AsyncLatentRuntimeError(
        "scheduler_events must contain SchedulerEvent values"
      )
    object.__setattr__(self, "scheduler_events", events)
    if self.status is RuntimeStatus.ACTION:
      if self.action is None or self.executed_latent is None or self.provenance is None:
        raise AsyncLatentRuntimeError("action status requires action provenance")
      if self.execution_id is None:
        raise AsyncLatentRuntimeError("action status requires a positive execution id")
      execution_id = _nonnegative_int(self.execution_id, "execution_id")
      if execution_id == 0:
        raise AsyncLatentRuntimeError("action status requires a positive execution id")
      object.__setattr__(self, "execution_id", execution_id)
      action = _copy_vector(self.action, "action", DEFAULT_CONTRACT.action_dimension)
      latent = _copy_vector(
        self.executed_latent, "executed_latent", DEFAULT_CONTRACT.latent_dimension
      )
      action.setflags(write=False)
      latent.setflags(write=False)
      object.__setattr__(self, "action", action)
      object.__setattr__(self, "executed_latent", latent)
    else:
      if any(
        value is not None
        for value in (
          self.action,
          self.executed_latent,
          self.provenance,
          self.execution_id,
        )
      ):
        raise AsyncLatentRuntimeError("fail-closed status cannot return an action")


@dataclass(frozen=True, slots=True)
class _PendingExecution:
  execution_id: int
  state: WorldState
  latent: np.ndarray
  provenance: ExecutionProvenance


class AsyncLatentRuntime:
  """Nonblocking D5 runtime adapter for raw latent plans.

  ``step`` snapshots the current measured state and task, queues a planner
  request, polls the bounded scheduler, selects only the row for the current
  execution tick, and decodes that row with the current 99-D conditioning.
  It never waits for inference.  The caller must call :meth:`acknowledge`
  after the action is accepted by the execution boundary before that pair can
  enter the measured history.

  ``source_timestamp`` and the injected ``clock`` use the same monotonic clock
  domain.  A plan is rejected if it is older than ``max_source_age_seconds`` or
  newer than ``max_future_skew_seconds``.  No standing or robot fallback is
  emitted for any failure.
  """

  def __init__(
    self,
    executor: PlannerExecutor,
    *,
    projection: ProjectionBundle,
    decode: Callable[[Tensor, Tensor], Tensor],
    generator: torch.Generator,
    contract: DiffusionContract = DEFAULT_CONTRACT,
    period_seconds: float = DEFAULT_CONTRACT.period_seconds,
    max_fallback_ticks: int = 4,
    max_source_age_seconds: float = 0.5,
    max_future_skew_seconds: float = 0.0,
    clock: Callable[[], float] = time.monotonic,
    device: str | torch.device = "cpu",
    planner_device: str | torch.device = "cpu",
    vae_identity: str = "unspecified-vae",
    model_identity: str = "unspecified-model",
    projection_identity: str | None = None,
    expected_source_frame: int | None = None,
  ) -> None:
    if not isinstance(contract, DiffusionContract):
      raise AsyncLatentRuntimeError("contract must be a DiffusionContract")
    if contract.identity_hash() != DEFAULT_CONTRACT.identity_hash():
      raise AsyncLatentRuntimeError("runtime requires the active no-mixture contract")
    if not isinstance(projection, ProjectionBundle):
      raise AsyncLatentRuntimeError("projection must be a ProjectionBundle")
    if projection.contract.identity_hash() != contract.identity_hash():
      raise AsyncLatentRuntimeError("projection contract identity differs from runtime")
    if not callable(decode):
      raise AsyncLatentRuntimeError("decode must be callable")
    if not callable(clock):
      raise AsyncLatentRuntimeError("clock must be callable")
    if not isinstance(generator, torch.Generator):
      raise AsyncLatentRuntimeError("generator must be a torch.Generator")
    try:
      resolved_device = torch.device(device)
      torch.empty(0, device=resolved_device)
      resolved_planner_device = torch.device(planner_device)
      torch.empty(0, device=resolved_planner_device)
    except (RuntimeError, ValueError) as exc:
      raise AsyncLatentRuntimeError(
        f"invalid decoder or planner device {device!r}/{planner_device!r}"
      ) from exc
    if torch.device(generator.device) != resolved_planner_device:
      raise AsyncLatentRuntimeError(
        "request generator device must match planner_device"
      )
    period = _finite(period_seconds, "period_seconds")
    if abs(period - contract.period_seconds) > 1.0e-12:
      raise AsyncLatentRuntimeError("period_seconds differs from the active contract")
    max_age = _finite(max_source_age_seconds, "max_source_age_seconds")
    max_skew = _finite(max_future_skew_seconds, "max_future_skew_seconds")
    if max_age < 0.0 or max_skew < 0.0:
      raise AsyncLatentRuntimeError("source-age limits must be non-negative")
    if (
      not isinstance(max_fallback_ticks, (int, np.integer))
      or isinstance(max_fallback_ticks, (bool, np.bool_))
      or max_fallback_ticks < 0
    ):
      raise AsyncLatentRuntimeError("max_fallback_ticks must be a non-negative integer")
    frame = (
      None
      if expected_source_frame is None
      else _nonnegative_int(expected_source_frame, "expected_source_frame")
    )
    self.executor = executor
    self.projection = projection
    self.contract = contract
    self.decode = decode
    self.device = resolved_device
    self.planner_device = resolved_planner_device
    self.period_seconds = period
    self.max_source_age_seconds = max_age
    self.max_future_skew_seconds = max_skew
    self.clock = clock
    self.vae_identity = _identity(vae_identity, "vae_identity")
    self.model_identity = _identity(model_identity, "model_identity")
    self._projection_identity = projection_identity
    if projection_identity is not None:
      self._projection_identity = _identity(projection_identity, "projection_identity")
    self.expected_source_frame = frame
    self.history = HandoverHistory(contract=contract)
    self.scheduler = RuntimeScheduler(
      executor,
      period_seconds=period,
      max_fallback_ticks=int(max_fallback_ticks),
      expected_action_dim=contract.latent_dimension,
      expected_contract_identity=contract.identity_hash(),
    )
    self._generator = torch.Generator(device=resolved_planner_device)
    self._generator.set_state(generator.get_state().detach().clone())
    self._task_snapshot: TaskSnapshot | None = None
    self._pending_execution: _PendingExecution | None = None
    self._next_execution_id = 1
    self._released = False
    self._shutdown = False

  @property
  def history_ready(self) -> bool:
    """Whether the eight measured VAE bootstrap pairs are present."""

    return self.history.ready

  @property
  def history_size(self) -> int:
    """Number of currently retained measured state/latent pairs."""

    return len(self.history.pairs)

  @property
  def pending_execution_id(self) -> int | None:
    """The acknowledgement token awaiting execution, if any."""

    return (
      None if self._pending_execution is None else self._pending_execution.execution_id
    )

  @property
  def released(self) -> bool:
    return self._released

  @property
  def shutdown_complete(self) -> bool:
    return self._shutdown

  @property
  def cleanup_confirmed(self) -> bool:
    """Return exact cleanup status when the supplied executor exposes it."""

    closed_marker = object()
    try:
      closed = getattr(self.executor, "closed", closed_marker)
      alive = getattr(self.executor, "worker_alive", closed_marker)
    except BaseException:
      return False
    if closed is closed_marker or alive is closed_marker:
      return False
    return bool(self._released or self._shutdown) and closed is True and alive is False

  def acknowledge_bootstrap(
    self,
    state: WorldState,
    latent: Sequence[float] | np.ndarray,
    *,
    success: bool,
  ) -> bool:
    """Record one VAE bootstrap pair only after an explicit execution ack.

    A failed bootstrap acknowledgement does not enter history.  The caller
    must supply the measured pre-action state and the latent decoded on it.
    """

    self._require_open()
    if not isinstance(success, bool):
      raise AsyncLatentRuntimeError("bootstrap success must be boolean")
    if not success:
      return False
    checked_state = _copy_state(state)
    checked_latent = _copy_vector(
      latent, "bootstrap latent", self.contract.latent_dimension
    )
    self.history.push(checked_state, checked_latent)
    return True

  def poll(self) -> tuple[SchedulerEvent, ...]:
    """Poll the planner without selecting or decoding an action."""

    if self._released:
      return (SchedulerEvent(SchedulerStatus.RELEASED, "runtime is released"),)
    if self._shutdown:
      return (SchedulerEvent(SchedulerStatus.SHUTDOWN, "runtime is shut down"),)
    return self.scheduler.poll()

  def step(
    self,
    current_state: WorldState,
    conditioning: Sequence[float] | np.ndarray,
    *,
    tick: int,
    source_frame: int = 0,
    task: object | None = None,
    task_revision: str | None = None,
  ) -> RuntimeStep:
    """Queue planning and decode only the raw latent for ``tick``.

    The method performs no sleep or inference wait.  A missing or late plan is
    an explicit no-action status.  The returned action is pending until the
    caller invokes :meth:`acknowledge`.
    """

    tick_value = _nonnegative_int(tick, "control tick")
    if self._released:
      return self._closed_step(
        tick_value, RuntimeStatus.RELEASED, "runtime is released"
      )
    if self._shutdown:
      return self._closed_step(
        tick_value, RuntimeStatus.SHUTDOWN, "runtime is shut down"
      )
    try:
      frame_value = _nonnegative_int(source_frame, "source_frame")
      state = _copy_state(current_state)
      _copy_vector(conditioning, "decoder conditioning", _CONDITIONING_DIM)
      now = self._now()
      snapshot_task = TaskSnapshot.from_inputs(task, revision=task_revision)
    except (AsyncLatentRuntimeError, LatentPlanningError) as exc:
      return self._closed_step(tick_value, RuntimeStatus.INVALID_INPUT, str(exc))

    if (
      self.expected_source_frame is not None
      and frame_value != self.expected_source_frame
    ):
      return self._closed_step(
        tick_value,
        RuntimeStatus.INVALID_INPUT,
        "source_frame differs from expected_source_frame",
      )

    if self._task_snapshot != snapshot_task:
      if self._task_snapshot is not None:
        self._invalidate_task_work()
      self._task_snapshot = snapshot_task

    if self._pending_execution is not None:
      return self._closed_step(
        tick_value,
        RuntimeStatus.AWAITING_EXECUTION_ACK,
        "previous action is awaiting execution acknowledgement",
      )
    if not self.history.ready:
      return self._closed_step(
        tick_value,
        RuntimeStatus.BOOTSTRAP_REQUIRED,
        f"eight measured bootstrap pairs are required; got {self.history_size}",
      )

    try:
      packet = build_planning_packet(
        self.history,
        state,
        projection=self.projection,
        source_timestamp=now,
        source_tick=tick_value,
        source_frame=frame_value,
        period_seconds=self.period_seconds,
        task=snapshot_task,
        task_revision=snapshot_task.revision,
        generator=self._generator,
        contract=self.contract,
        vae_identity=self.vae_identity,
        model_identity=self.model_identity,
        projection_identity=self._projection_identity,
      )
      self._projection_identity = packet.projection_identity
      self._advance_generator()
      self.scheduler.request(
        np.asarray((packet.source_tick, packet.source_timestamp), dtype=np.float64),
        source_timestamp=packet.source_timestamp,
        source_tick=packet.source_tick,
        context=packet.as_context(),
      )
      scheduler_step = self.scheduler.consume(control_tick=tick_value, timestamp=now)
    except (AsyncLatentRuntimeError, LatentPlanningError, SchedulerError) as exc:
      return self._closed_step(tick_value, RuntimeStatus.INVALID_INPUT, str(exc))
    except BaseException as exc:
      return self._closed_step(tick_value, RuntimeStatus.PLANNER_FAILURE, str(exc))

    accepted = self.scheduler.accepted_plan
    accepted_origin = self.scheduler.accepted_request_provenance
    if accepted is not None:
      invalid = self._validate_plan(
        accepted, accepted_origin, now, frame_value, snapshot_task
      )
      if invalid is not None:
        self.scheduler.reset()
        return self._closed_step(
          tick_value, invalid[0], invalid[1], scheduler_step.event
        )

    if scheduler_step.status in (
      SchedulerStatus.ACTION,
      SchedulerStatus.FALLBACK_ACTION,
    ):
      if accepted is None or scheduler_step.action is None:
        self.scheduler.reset()
        return self._closed_step(
          tick_value,
          RuntimeStatus.INVALID_PLAN,
          "scheduler returned an action without an accepted latent plan",
          scheduler_step.event,
        )
      try:
        latent = _copy_vector(
          scheduler_step.action, "selected raw latent", self.contract.latent_dimension
        )
        action = self._decode_action(latent, conditioning)
      except (AsyncLatentRuntimeError, RuntimeError, ValueError, TypeError) as exc:
        self.scheduler.reset()
        return self._closed_step(
          tick_value, RuntimeStatus.DECODE_FAILURE, str(exc), scheduler_step.event
        )
      provenance = self._provenance(accepted, tick_value)
      execution_id = self._next_execution_id
      self._next_execution_id += 1
      self._pending_execution = _PendingExecution(
        execution_id,
        state,
        latent.copy(),
        provenance,
      )
      return RuntimeStep(
        RuntimeStatus.ACTION,
        tick_value,
        action,
        latent,
        provenance,
        execution_id,
        "raw latent decoded at the execution tick",
        (scheduler_step.event,) if scheduler_step.event is not None else (),
      )

    status, detail = self._status_from_scheduler(scheduler_step.status)
    return self._closed_step(tick_value, status, detail, scheduler_step.event)

  def acknowledge(self, step: RuntimeStep, *, success: bool) -> bool:
    """Commit or discard one returned action after execution acknowledgement."""

    self._require_open()
    if not isinstance(step, RuntimeStep) or step.status is not RuntimeStatus.ACTION:
      raise AsyncLatentRuntimeError("only an action RuntimeStep can be acknowledged")
    if step.execution_id is None:
      raise AsyncLatentRuntimeError(
        "execution acknowledgement requires an execution id"
      )
    execution_id = _nonnegative_int(step.execution_id, "execution_id")
    if execution_id == 0:
      raise AsyncLatentRuntimeError(
        "execution acknowledgement requires a positive execution id"
      )
    if not isinstance(success, bool):
      raise AsyncLatentRuntimeError("execution success must be boolean")
    pending = self._pending_execution
    if pending is None or pending.execution_id != execution_id:
      raise AsyncLatentRuntimeError(
        "execution acknowledgement does not match pending action"
      )
    self._pending_execution = None
    if not success:
      self.scheduler.reset()
      return False
    self.history.push(pending.state, pending.latent)
    return True

  def reset(self) -> RuntimeStatus:
    """Clear history and all unsafe planner/action work."""

    if self._released:
      return RuntimeStatus.RELEASED
    if self._shutdown:
      return RuntimeStatus.SHUTDOWN
    self._pending_execution = None
    self.history.reset()
    self._task_snapshot = None
    self.scheduler.reset()
    return RuntimeStatus.RESET

  def switch_policy(self, policy_id: str) -> RuntimeStatus:
    """Invalidate history and work for a policy-generation change."""

    _identity(policy_id, "policy_id")
    if self._released:
      return RuntimeStatus.RELEASED
    if self._shutdown:
      return RuntimeStatus.SHUTDOWN
    self._pending_execution = None
    self.history.reset()
    self._task_snapshot = None
    self.scheduler.switch_policy(policy_id)
    return RuntimeStatus.RESET

  def release(self) -> RuntimeStatus:
    """Permanently fail closed and release the exact executor resources."""

    if self._released:
      return RuntimeStatus.RELEASED
    self._pending_execution = None
    self.history.reset()
    self._released = True
    self.scheduler.release()
    self.decode = _released_decode
    return RuntimeStatus.RELEASED

  def shutdown(self) -> RuntimeStatus:
    """Idempotently shut down without permitting later actions."""

    if self._released:
      return RuntimeStatus.RELEASED
    if self._shutdown:
      return RuntimeStatus.SHUTDOWN
    self._pending_execution = None
    self.history.reset()
    self._shutdown = True
    self.scheduler.shutdown()
    self.decode = _released_decode
    return RuntimeStatus.SHUTDOWN

  def _require_open(self) -> None:
    if self._released:
      raise AsyncLatentRuntimeError("runtime is released")
    if self._shutdown:
      raise AsyncLatentRuntimeError("runtime is shut down")

  def _now(self) -> float:
    value = _finite(self.clock(), "monotonic clock value")
    if value < 0.0:
      raise AsyncLatentRuntimeError("monotonic clock value must be non-negative")
    return value

  def _advance_generator(self) -> None:
    try:
      torch.rand(1, generator=self._generator, device=self.planner_device)
    except (RuntimeError, TypeError) as exc:
      raise AsyncLatentRuntimeError(
        "could not advance request-local planner-device RNG"
      ) from exc

  def _invalidate_task_work(self) -> None:
    self._pending_execution = None
    self.scheduler.reset()

  def _validate_plan(
    self,
    plan: Plan,
    origin: AcceptedRequestProvenance | None,
    now: float,
    source_frame: int,
    task: TaskSnapshot,
  ) -> tuple[RuntimeStatus, str] | None:
    metadata = plan.schema_metadata
    if origin is None:
      return RuntimeStatus.INVALID_PLAN, "accepted plan has no request provenance"
    required = {
      "schema",
      "contract",
      "representation",
      "kind",
      "semantic",
      "latent_dimension",
      "latent_space",
      "source_tick",
      "valid_from_tick",
      "valid_until_tick",
      "source_timestamp",
      "source_frame",
      "root_frame",
      "period_seconds",
      "vae_identity",
      "projection_identity",
      "model_identity",
      "task_revision",
      "task_mode",
    }
    if not required.issubset(metadata):
      return RuntimeStatus.INVALID_PLAN, "latent plan metadata is incomplete"
    if metadata.get("schema") != LATENT_PLANNING_SCHEMA:
      return RuntimeStatus.INVALID_PLAN, "latent plan schema is incompatible"
    if metadata.get("contract") != self.contract.identity_hash():
      return RuntimeStatus.INVALID_PLAN, "latent plan contract identity is incompatible"
    if metadata.get("representation") != RAW_LATENT_REPRESENTATION:
      return RuntimeStatus.INVALID_PLAN, "latent plan is not raw 32-D latent data"
    if metadata.get("kind") != "latent" or metadata.get("semantic") != "raw":
      return RuntimeStatus.INVALID_PLAN, "latent plan semantic is not raw latent"
    if metadata.get("latent_space") != "raw" or metadata.get("latent_dimension") != 32:
      return RuntimeStatus.INVALID_PLAN, "latent plan dimension or space is invalid"
    if metadata.get("root_frame") != ROOT_FRAME:
      return RuntimeStatus.INVALID_PLAN, "latent plan root frame is invalid"
    if metadata.get("vae_identity") != self.vae_identity:
      return RuntimeStatus.INVALID_PLAN, "latent plan VAE identity is incompatible"
    if metadata.get("model_identity") != self.model_identity:
      return RuntimeStatus.INVALID_PLAN, "latent plan model identity is incompatible"
    if (
      self._projection_identity is None
      or metadata.get("projection_identity") != self._projection_identity
    ):
      return (
        RuntimeStatus.INVALID_PLAN,
        "latent plan projection identity is incompatible",
      )
    if (
      metadata.get("task_revision") != task.revision
      or metadata.get("task_mode") != task.mode
    ):
      return (
        RuntimeStatus.INVALID_PLAN,
        "latent plan task revision or mode is incompatible",
      )
    try:
      source_tick = _metadata_int(metadata, "source_tick")
      valid_from = _metadata_int(metadata, "valid_from_tick")
      valid_until = _metadata_int(metadata, "valid_until_tick")
      metadata_frame = _metadata_int(metadata, "source_frame")
      metadata_timestamp = _metadata_float(metadata, "source_timestamp")
      metadata_period = _metadata_float(metadata, "period_seconds")
    except AsyncLatentRuntimeError as exc:
      return RuntimeStatus.INVALID_PLAN, str(exc)
    if source_tick != origin.source_tick:
      return (
        RuntimeStatus.INVALID_PLAN,
        "latent plan source tick differs from its request",
      )
    if metadata_timestamp != origin.source_timestamp:
      return (
        RuntimeStatus.INVALID_PLAN,
        "latent plan source timestamp differs from its request",
      )
    if plan.source_timestamp != origin.source_timestamp:
      return (
        RuntimeStatus.INVALID_PLAN,
        "latent plan timestamp differs from its request",
      )
    if source_tick != valid_from or plan.valid_from_tick != source_tick:
      return RuntimeStatus.INVALID_PLAN, "latent plan source and valid ticks disagree"
    if valid_until != source_tick + self.contract.future_steps:
      return (
        RuntimeStatus.INVALID_PLAN,
        "latent plan horizon is not the fixed 33-row suffix",
      )
    if plan.actions.shape != (
      self.contract.future_steps + 1,
      self.contract.latent_dimension,
    ):
      return RuntimeStatus.INVALID_PLAN, "latent plan action matrix is not [33, 32]"
    if not np.isfinite(plan.actions).all():
      return RuntimeStatus.INVALID_PLAN, "latent plan contains non-finite values"
    if metadata_frame != source_frame:
      return RuntimeStatus.INVALID_PLAN, "latent plan source frame is incompatible"
    if metadata_timestamp != plan.source_timestamp:
      return (
        RuntimeStatus.INVALID_PLAN,
        "latent plan source timestamp disagrees with metadata",
      )
    if plan.source_timestamp < 0.0:
      return RuntimeStatus.INVALID_PLAN, "latent plan source timestamp is negative"
    if abs(metadata_period - self.period_seconds) > 1.0e-12:
      return RuntimeStatus.INVALID_PLAN, "latent plan period differs from runtime"
    age = now - plan.source_timestamp
    if age > self.max_source_age_seconds:
      return (
        RuntimeStatus.SOURCE_TOO_OLD,
        "latent plan source timestamp exceeded its age limit",
      )
    if age < -self.max_future_skew_seconds:
      return (
        RuntimeStatus.SOURCE_IN_FUTURE,
        "latent plan source timestamp is in the future",
      )
    return None

  def _provenance(self, plan: Plan, execution_tick: int) -> ExecutionProvenance:
    metadata = plan.schema_metadata
    return ExecutionProvenance(
      plan_sequence_id=plan.plan_sequence_id,
      source_timestamp=plan.source_timestamp,
      source_tick=_metadata_int(metadata, "source_tick"),
      execution_tick=execution_tick,
      valid_until_tick=_metadata_int(metadata, "valid_until_tick"),
      source_frame=_metadata_int(metadata, "source_frame"),
      contract_identity=cast(str, metadata["contract"]),
      vae_identity=cast(str, metadata["vae_identity"]),
      projection_identity=cast(str, metadata["projection_identity"]),
      model_identity=cast(str, metadata["model_identity"]),
      task_revision=cast(str, metadata["task_revision"]),
      task_mode=cast(str, metadata["task_mode"]),
    )

  def _decode_action(
    self, latent: np.ndarray, conditioning: Sequence[float] | np.ndarray
  ) -> np.ndarray:
    latent_values = _copy_vector(latent, "raw latent", self.contract.latent_dimension)
    conditioning_values = _copy_vector(
      conditioning, "decoder conditioning", _CONDITIONING_DIM
    )
    latent_tensor = torch.from_numpy(
      np.ascontiguousarray(latent_values, dtype=np.float32)
    ).to(device=self.device)[None]
    conditioning_tensor = torch.from_numpy(
      np.ascontiguousarray(conditioning_values, dtype=np.float32)
    ).to(device=self.device)[None]
    try:
      with torch.no_grad():
        action = self.decode(latent_tensor, conditioning_tensor)
    except BaseException as exc:
      raise AsyncLatentRuntimeError(f"decoder failed: {exc}") from exc
    if not isinstance(action, Tensor):
      raise AsyncLatentRuntimeError("decoder must return a torch tensor")
    try:
      decoded = (
        action.detach().to(device="cpu", dtype=torch.float32).numpy().reshape(-1)
      )
    except (RuntimeError, TypeError, ValueError) as exc:
      raise AsyncLatentRuntimeError("decoder output could not be converted") from exc
    if (
      decoded.shape != (self.contract.action_dimension,)
      or not np.isfinite(decoded).all()
    ):
      raise AsyncLatentRuntimeError("decoder action must be finite with shape [31]")
    return np.array(decoded, dtype=np.float64, copy=True)

  @staticmethod
  def _status_from_scheduler(status: SchedulerStatus) -> tuple[RuntimeStatus, str]:
    mapping = {
      SchedulerStatus.NO_PLAN: (
        RuntimeStatus.WAITING_FOR_PLAN,
        "no accepted latent plan is ready",
      ),
      SchedulerStatus.PLAN_NOT_YET_VALID: (
        RuntimeStatus.PLAN_NOT_YET_VALID,
        "accepted latent plan starts after the execution tick",
      ),
      SchedulerStatus.PLAN_EXHAUSTED: (
        RuntimeStatus.PLAN_EXHAUSTED,
        "accepted latent plan has no execution-tick suffix",
      ),
      SchedulerStatus.INVALID_OUTPUT: (
        RuntimeStatus.INVALID_PLAN,
        "scheduler rejected planner output",
      ),
      SchedulerStatus.PLANNER_EXCEPTION: (
        RuntimeStatus.PLANNER_FAILURE,
        "planner or executor failed",
      ),
      SchedulerStatus.RELEASED: (RuntimeStatus.RELEASED, "runtime is released"),
      SchedulerStatus.SHUTDOWN: (RuntimeStatus.SHUTDOWN, "runtime is shut down"),
    }
    return mapping.get(status, (RuntimeStatus.WAITING_FOR_PLAN, status.value))

  @staticmethod
  def _closed_step(
    tick: int,
    status: RuntimeStatus,
    detail: str,
    event: SchedulerEvent | None = None,
  ) -> RuntimeStep:
    return RuntimeStep(
      status=status,
      control_tick=tick,
      detail=detail,
      scheduler_events=() if event is None else (event,),
    )


D5AsyncLatentRuntime = AsyncLatentRuntime


__all__ = [
  "AsyncLatentRuntime",
  "AsyncLatentRuntimeError",
  "D5AsyncLatentRuntime",
  "ExecutionProvenance",
  "RuntimeStatus",
  "RuntimeStep",
]
