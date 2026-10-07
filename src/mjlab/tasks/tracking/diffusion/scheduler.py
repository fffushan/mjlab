"""Simulator-independent D5 planner scheduling primitives.

The scheduler owns request and plan snapshots, but it does not create worker
threads.  A caller supplies an executor that implements :class:`PlannerExecutor`;
tests and runtime adapters can therefore drive submission and completion with a
deterministic polling loop.

The control consumer is synchronous. It selects the row for the execution tick
from the newest accepted plan. A result belongs to the newest submitted request
in the current policy generation; a newer pending request does not invalidate
that active request. When a planner is late, the consumer can use later rows
from that plan while the trajectory remains valid. The ``max_fallback_ticks``
value is a warning threshold only; it does not reject an action before the
accepted trajectory ends. Reset and policy switch invalidate queued and
accepted work but retain active ownership until its matching completion arrives.
Release and shutdown stop the executor. This is an action-trajectory primitive,
not a latent planner or current-proprioception decoder integration.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, cast

import numpy as np


class SchedulerError(ValueError):
  """A scheduler API call has invalid arguments or lifecycle state."""


class FrozenContext(Mapping[str, object]):
  """An immutable, pickleable mapping used for request-local context.

  ``MappingProxyType`` is intentionally not used here because the executor may
  serialize a request.  Nested mappings, sequences and NumPy arrays are
  converted to immutable plain values by :func:`_freeze_context`.
  """

  __slots__ = ("_items",)

  def __init__(
    self, items: Mapping[str, object] | Iterable[tuple[str, object]]
  ) -> None:
    values = dict(items)
    if any(not isinstance(key, str) for key in values):
      raise SchedulerError("planner context mapping keys must be strings")
    self._items = tuple(
      sorted(
        ((key, _freeze_context(value)) for key, value in values.items()),
        key=lambda item: item[0],
      )
    )

  def __getitem__(self, key: str) -> object:
    for item_key, value in self._items:
      if item_key == key:
        return value
    raise KeyError(key)

  def __iter__(self) -> Iterator[str]:
    return (key for key, _value in self._items)

  def __len__(self) -> int:
    return len(self._items)

  def __repr__(self) -> str:
    return f"FrozenContext({dict(self._items)!r})"

  def __eq__(self, other: object) -> bool:
    if isinstance(other, Mapping):
      return dict(self.items()) == dict(other.items())
    return NotImplemented

  def __reduce__(self) -> tuple[type["FrozenContext"], tuple[dict[str, object]]]:
    return type(self), (dict(self._items),)


class SchedulerStatus(str, Enum):
  """Explicit status returned by the synchronous consumer."""

  ACTION = "action"
  FALLBACK_ACTION = "fallback_action"
  FALLBACK_WARNING = "fallback_warning"
  NO_PLAN = "no_plan"
  PLAN_NOT_YET_VALID = "plan_not_yet_valid"
  PLAN_EXHAUSTED = "plan_exhausted"
  INVALID_OUTPUT = "invalid_output"
  PLANNER_EXCEPTION = "planner_exception"
  RESET = "reset"
  POLICY_SWITCH = "policy_switch"
  RELEASED = "released"
  SHUTDOWN = "shutdown"


@dataclass(frozen=True, slots=True)
class PlannerRequest:
  """An owned planner input snapshot and its source identity.

  ``request_sequence_id`` is assigned by :class:`RuntimeScheduler` and is
  strictly increasing for the scheduler lifetime.  ``policy_generation``
  changes on reset and policy switch, so a late result from an earlier
  generation cannot be accepted.  ``context`` is an optional immutable,
  pickleable snapshot.  It never contains a live simulator, model or decoder.
  """

  input_snapshot: np.ndarray
  source_timestamp: float
  source_tick: int
  request_sequence_id: int
  policy_generation: int
  context: object | None = None

  def __post_init__(self) -> None:
    _validate_timestamp(self.source_timestamp, "source_timestamp")
    _validate_tick(self.source_tick, "source_tick")
    _validate_integer(self.request_sequence_id, "request_sequence_id", minimum=1)
    _validate_integer(self.policy_generation, "policy_generation")
    try:
      snapshot = np.array(self.input_snapshot, dtype=np.float64, copy=True)
      finite = bool(np.isfinite(snapshot).all())
    except (TypeError, ValueError, OverflowError) as exc:
      raise SchedulerError("input_snapshot must be numeric and finite") from exc
    if snapshot.size == 0 or not finite:
      raise SchedulerError("input_snapshot must be nonempty and finite")
    snapshot.setflags(write=False)
    object.__setattr__(self, "input_snapshot", snapshot)
    if self.context is not None:
      object.__setattr__(self, "context", _freeze_context(self.context))


@dataclass(frozen=True, slots=True)
class AcceptedRequestProvenance:
  """Immutable source identity for the currently accepted plan."""

  request_sequence_id: int
  policy_generation: int
  source_timestamp: float
  source_tick: int

  def __post_init__(self) -> None:
    _validate_integer(
      self.request_sequence_id, "accepted request_sequence_id", minimum=1
    )
    _validate_integer(self.policy_generation, "accepted policy_generation")
    _validate_timestamp(self.source_timestamp, "accepted source_timestamp")
    _validate_tick(self.source_tick, "accepted source_tick")
    object.__setattr__(self, "request_sequence_id", int(self.request_sequence_id))
    object.__setattr__(self, "policy_generation", int(self.policy_generation))
    object.__setattr__(self, "source_timestamp", float(self.source_timestamp))
    object.__setattr__(self, "source_tick", int(self.source_tick))


@dataclass(frozen=True, slots=True)
class Plan:
  """A candidate action trajectory returned by a planner.

  The first row is valid at ``valid_from_tick``.  The final row is valid at
  ``valid_until_tick``.  The planner supplies a monotonically increasing plan
  sequence; the scheduler rejects a result that is not newer than its
  accepted plan.  The metadata mapping is copied at the top level. Metadata
  values are opaque and remain caller-owned. Finite actions are checked when
  the scheduler accepts the result, so malformed planner output can produce
  an explicit fail-closed event.
  """

  plan_sequence_id: int
  source_timestamp: float
  valid_from_tick: int
  actions: np.ndarray
  schema_metadata: Mapping[str, object]

  def __post_init__(self) -> None:
    _validate_integer(self.plan_sequence_id, "plan_sequence_id", minimum=1)
    _validate_timestamp(self.source_timestamp, "source_timestamp")
    _validate_tick(self.valid_from_tick, "valid_from_tick")
    try:
      actions = np.array(self.actions, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
      raise SchedulerError("actions must be numeric") from exc
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] == 0:
      raise SchedulerError("actions must have shape [horizon, action_dim]")
    actions.setflags(write=False)
    object.__setattr__(self, "actions", actions)
    metadata = dict(self.schema_metadata)
    if any(not isinstance(key, str) for key in metadata):
      raise SchedulerError("schema_metadata keys must be strings")
    object.__setattr__(self, "schema_metadata", MappingProxyType(metadata))

  @property
  def valid_until_tick(self) -> int:
    """Last control tick for which this trajectory has an action."""

    return self.valid_from_tick + self.actions.shape[0] - 1


@dataclass(frozen=True, slots=True)
class PlannerResult:
  """One executor completion, including a planner exception when present."""

  request_sequence_id: int
  policy_generation: int
  plan: Plan | None = None
  error: BaseException | None = None

  def __post_init__(self) -> None:
    _validate_integer(self.request_sequence_id, "result request_sequence_id", minimum=1)
    _validate_integer(self.policy_generation, "result policy_generation")
    if (self.plan is None) == (self.error is None):
      raise SchedulerError("result must contain exactly one plan or error")


class PlannerExecutor(Protocol):
  """Polling boundary for a planner worker.

  ``submit`` must retain or process the owned request without mutating it.
  ``poll`` must return completed results without blocking.  ``shutdown`` must
  release worker resources exactly once; the scheduler tolerates idempotent
  executor shutdown for lifecycle adapters.
  """

  def submit(self, request: PlannerRequest) -> None:
    """Submit one owned request for asynchronous planning."""
    ...

  def poll(self) -> Iterable[PlannerResult]:
    """Return completed results without waiting."""
    ...

  def shutdown(self) -> None:
    """Stop the executor and release its resources."""
    ...


@dataclass(frozen=True, slots=True)
class SchedulerEvent:
  """A lifecycle or planner event observed by the caller."""

  status: SchedulerStatus
  detail: str
  request_sequence_id: int | None = None
  plan_sequence_id: int | None = None


@dataclass(frozen=True, slots=True)
class SchedulerStep:
  """The action selected for one synchronous control tick.

  ``action`` is an owned writable copy when status is ``ACTION`` or
  ``FALLBACK_ACTION``.  All other statuses are fail-closed and return
  ``action=None``.
  """

  status: SchedulerStatus
  action: np.ndarray | None
  control_tick: int
  event: SchedulerEvent | None = None
  plan_sequence_id: int | None = None

  def __post_init__(self) -> None:
    _validate_tick(self.control_tick, "control_tick")
    if self.action is not None:
      action = np.array(self.action, dtype=np.float64, copy=True)
      if action.ndim != 1 or action.size == 0 or not np.isfinite(action).all():
        raise SchedulerError("selected action must be a finite nonempty vector")
      object.__setattr__(self, "action", action)
    if self.status in (SchedulerStatus.ACTION, SchedulerStatus.FALLBACK_ACTION):
      if self.action is None:
        raise SchedulerError("action status requires an action")
    elif self.action is not None:
      raise SchedulerError("fail-closed status cannot return an action")


class RuntimeScheduler:
  """Bounded latest-request-wins scheduler for a synchronous control loop.

  The default control period is 0.020 seconds.  The period is metadata for
  runtime adapters; scheduler decisions use only caller-provided ticks and
  timestamps, which makes tests deterministic and avoids wall-clock sleeps.

  At most one request is pending and at most one request is in flight.
  Calling :meth:`request` replaces an unsubmitted request. A newer pending
  request does not invalidate the active completion. After first selecting a
  new plan at tick T, its fallback suffix is used until a newer plan replaces
  it or the trajectory ends. ``max_fallback_ticks`` emits one warning event
  when the fallback count reaches the configured threshold; it is not a hard
  action limit. Skipped ticks count toward this warning threshold. Plan
  replacement is immediate: future-valid plans return no action until their
  start tick, and expired plans return no action. The previous plan is never
  restored after replacement.
  """

  DEFAULT_PERIOD_SECONDS = 0.020
  DEFAULT_CONTROL_HZ = 50.0

  def __init__(
    self,
    executor: PlannerExecutor,
    *,
    period_seconds: float = DEFAULT_PERIOD_SECONDS,
    max_fallback_ticks: int = 4,
    expected_action_dim: int | None = None,
    expected_contract_identity: str | None = None,
  ) -> None:
    _validate_timestamp(period_seconds, "period_seconds")
    if float(period_seconds) <= 0:
      raise SchedulerError("period_seconds must be finite and positive")
    if (
      not isinstance(max_fallback_ticks, (int, np.integer))
      or isinstance(max_fallback_ticks, (bool, np.bool_))
      or max_fallback_ticks < 0
    ):
      raise SchedulerError("max_fallback_ticks must be a nonnegative integer")
    if expected_action_dim is not None and (
      isinstance(expected_action_dim, (bool, np.bool_))
      or not isinstance(expected_action_dim, (int, np.integer))
      or expected_action_dim <= 0
    ):
      raise SchedulerError("expected_action_dim must be a positive integer or None")
    if expected_contract_identity is not None and (
      not isinstance(expected_contract_identity, str) or not expected_contract_identity
    ):
      raise SchedulerError(
        "expected_contract_identity must be a nonempty string or None"
      )
    self.executor = executor
    self.period_seconds = float(period_seconds)
    self.max_fallback_ticks = max_fallback_ticks
    self.expected_action_dim = expected_action_dim
    self.expected_contract_identity = expected_contract_identity
    self._policy_generation = 0
    self._next_request_sequence = 1
    self._latest_request_sequence: int | None = None
    self._pending_request: PlannerRequest | None = None
    self._submitted_request_sequence: int | None = None
    self._submitted_policy_generation: int | None = None
    self._submitted_source_timestamp: float | None = None
    self._submitted_source_tick: int | None = None
    self._accepted_plan: Plan | None = None
    self._accepted_request_sequence: int | None = None
    self._accepted_request_provenance: AcceptedRequestProvenance | None = None
    self._accepted_plan_sequence = 0
    self._new_plan_since_consume = False
    self._fallback_ticks = 0
    self._plan_first_use_tick: int | None = None
    self._fallback_warning_emitted = False
    self._last_control_tick: int | None = None
    self._fault: SchedulerEvent | None = None
    self._released = False
    self._shutdown = False

  @property
  def control_frequency_hz(self) -> float:
    """Control frequency implied by :attr:`period_seconds`."""

    return 1.0 / self.period_seconds

  @property
  def policy_generation(self) -> int:
    """Current generation. Results from another generation are ignored."""

    return self._policy_generation

  @property
  def pending_request(self) -> PlannerRequest | None:
    """The one request waiting for executor submission, if any."""

    return self._pending_request

  @property
  def accepted_plan(self) -> Plan | None:
    """The currently accepted plan, or ``None`` after invalidation."""

    return self._accepted_plan

  @property
  def accepted_request_provenance(self) -> AcceptedRequestProvenance | None:
    """Source identity of the currently accepted plan, if any."""

    return self._accepted_request_provenance

  def request(
    self,
    input_snapshot: np.ndarray,
    *,
    source_timestamp: float,
    source_tick: int,
    context: object | None = None,
  ) -> PlannerRequest:
    """Queue a latest-wins planner request and return its owned snapshot.

    ``context`` is copied and frozen before the request can reach an executor.
    The default ``None`` preserves the original scheduler protocol.
    """

    self._require_open()
    request = PlannerRequest(
      input_snapshot=input_snapshot,
      source_timestamp=source_timestamp,
      source_tick=source_tick,
      request_sequence_id=self._next_request_sequence,
      policy_generation=self._policy_generation,
      context=context,
    )
    self._next_request_sequence += 1
    self._latest_request_sequence = request.request_sequence_id
    self._pending_request = request
    return request

  def poll(self) -> tuple[SchedulerEvent, ...]:
    """Submit the newest pending request and drain completed executor work."""

    if self._released:
      return (SchedulerEvent(SchedulerStatus.RELEASED, "scheduler is released"),)
    if self._shutdown:
      return (SchedulerEvent(SchedulerStatus.SHUTDOWN, "scheduler is shut down"),)
    events: list[SchedulerEvent] = []
    self._submit_pending_if_idle(events)
    try:
      results = tuple(self.executor.poll())
    except BaseException as exc:
      events.append(
        self._set_fault(
          SchedulerStatus.PLANNER_EXCEPTION,
          f"executor poll failed: {exc}",
        )
      )
      return tuple(events)
    for result in results:
      matches_active = False
      result_sequence_id: int | None = None
      try:
        if not isinstance(result, PlannerResult):
          raise SchedulerError("executor returned a malformed planner result")
        _validate_integer(result.request_sequence_id, "request_sequence_id", minimum=1)
        _validate_integer(result.policy_generation, "policy_generation")
        result_sequence_id = int(result.request_sequence_id)
        matches_active = (
          result_sequence_id == self._submitted_request_sequence
          and result.policy_generation == self._submitted_policy_generation
        )
        event = self._handle_result(result)
      except (AttributeError, TypeError, ValueError, IndexError, OverflowError) as exc:
        event = self._set_fault(
          SchedulerStatus.INVALID_OUTPUT,
          f"malformed planner result: {exc}",
          request_sequence_id=result_sequence_id,
        )
      if event is not None:
        events.append(event)
      # Only a validated completion identity proves that active work ended.
      # Unattributed faults must retain the tombstone, even across reset.
      if matches_active:
        self._submitted_request_sequence = None
        self._submitted_policy_generation = None
        self._submitted_source_timestamp = None
        self._submitted_source_tick = None
    self._submit_pending_if_idle(events)
    return tuple(events)

  def consume(self, *, control_tick: int, timestamp: float) -> SchedulerStep:
    """Select one action for ``control_tick`` without sleeping.

    The timestamp is validated for runtime traceability but does not drive
    scheduling.  A control tick must increase strictly until reset or release.
    """

    _validate_timestamp(timestamp, "timestamp")
    _validate_tick(control_tick, "control_tick")
    if self._released:
      return self._closed_step(
        control_tick, SchedulerStatus.RELEASED, "scheduler is released"
      )
    if self._shutdown:
      return self._closed_step(
        control_tick, SchedulerStatus.SHUTDOWN, "scheduler is shut down"
      )
    if self._last_control_tick is not None and control_tick <= self._last_control_tick:
      raise SchedulerError("control_tick must increase strictly")
    self._last_control_tick = control_tick
    events = self.poll()
    event = events[-1] if events else self._fault
    if self._fault is not None:
      return self._closed_step(
        control_tick, self._fault.status, self._fault.detail, self._fault
      )
    plan = self._accepted_plan
    if plan is None:
      return self._closed_step(
        control_tick, SchedulerStatus.NO_PLAN, "no accepted plan", event
      )
    if control_tick < plan.valid_from_tick:
      return self._closed_step(
        control_tick,
        SchedulerStatus.PLAN_NOT_YET_VALID,
        "accepted plan is not valid for this control tick",
        event,
        plan.plan_sequence_id,
      )
    if control_tick > plan.valid_until_tick:
      self._accepted_plan = None
      self._accepted_request_sequence = None
      self._accepted_request_provenance = None
      return self._closed_step(
        control_tick,
        SchedulerStatus.PLAN_EXHAUSTED,
        "accepted plan has no valid action suffix",
        event,
        plan.plan_sequence_id,
      )
    warning_event: SchedulerEvent | None = None
    if self._new_plan_since_consume:
      self._fallback_ticks = 0
      self._plan_first_use_tick = control_tick
      self._new_plan_since_consume = False
      status = SchedulerStatus.ACTION
    else:
      assert self._plan_first_use_tick is not None
      self._fallback_ticks = control_tick - self._plan_first_use_tick
      if (
        self._fallback_ticks >= self.max_fallback_ticks
        and not self._fallback_warning_emitted
      ):
        self._fallback_warning_emitted = True
        warning_event = SchedulerEvent(
          SchedulerStatus.FALLBACK_WARNING,
          f"fallback warning: plan {plan.plan_sequence_id} has been used for "
          f"{self._fallback_ticks} fallback ticks; threshold "
          f"{self.max_fallback_ticks} reached; continuing until trajectory end",
          plan_sequence_id=plan.plan_sequence_id,
        )
      status = SchedulerStatus.FALLBACK_ACTION
    action = plan.actions[control_tick - plan.valid_from_tick]
    if not np.isfinite(action).all():
      fault = self._set_fault(
        SchedulerStatus.INVALID_OUTPUT,
        "accepted plan produced a non-finite action",
        plan_sequence_id=plan.plan_sequence_id,
      )
      return self._closed_step(
        control_tick, fault.status, fault.detail, fault, plan.plan_sequence_id
      )
    return SchedulerStep(
      status,
      action,
      control_tick,
      warning_event if warning_event is not None else event,
      plan.plan_sequence_id,
    )

  def reset(self) -> SchedulerEvent:
    """Invalidate pending requests, accepted plans, and planner faults."""

    if self._released or self._shutdown:
      return SchedulerEvent(
        SchedulerStatus.RELEASED if self._released else SchedulerStatus.SHUTDOWN,
        "scheduler cannot reset after release or shutdown",
      )
    self._policy_generation += 1
    self._pending_request = None
    self._accepted_plan = None
    self._accepted_request_sequence = None
    self._accepted_request_provenance = None
    self._latest_request_sequence = None
    self._new_plan_since_consume = False
    self._fallback_ticks = 0
    self._plan_first_use_tick = None
    self._last_control_tick = None
    self._fault = None
    return SchedulerEvent(SchedulerStatus.RESET, "queued and accepted work invalidated")

  def switch_policy(self, policy_id: str) -> SchedulerEvent:
    """Invalidate all old results when the caller changes policy identity."""

    if not isinstance(policy_id, str) or not policy_id:
      raise SchedulerError("policy_id must be a nonempty string")
    if self._released or self._shutdown:
      return SchedulerEvent(
        SchedulerStatus.RELEASED if self._released else SchedulerStatus.SHUTDOWN,
        "scheduler cannot switch policy after release or shutdown",
      )
    self._policy_generation += 1
    self._pending_request = None
    self._accepted_plan = None
    self._accepted_request_sequence = None
    self._accepted_request_provenance = None
    self._latest_request_sequence = None
    self._new_plan_since_consume = False
    self._fallback_ticks = 0
    self._plan_first_use_tick = None
    self._last_control_tick = None
    self._fault = None
    return SchedulerEvent(
      SchedulerStatus.POLICY_SWITCH, f"policy switched to {policy_id!r}"
    )

  def release(self) -> SchedulerEvent:
    """Fail closed permanently and shut down the executor."""

    if self._released:
      return SchedulerEvent(SchedulerStatus.RELEASED, "scheduler is already released")
    self._released = True
    self._pending_request = None
    self._submitted_request_sequence = None
    self._submitted_policy_generation = None
    self._submitted_source_timestamp = None
    self._submitted_source_tick = None
    self._latest_request_sequence = None
    self._accepted_plan = None
    self._accepted_request_sequence = None
    self._accepted_request_provenance = None
    self._new_plan_since_consume = False
    self._fallback_ticks = 0
    self._plan_first_use_tick = None
    self._last_control_tick = None
    self._fault = None
    self._shutdown_executor()

    return SchedulerEvent(
      SchedulerStatus.RELEASED, "scheduler released; no action may be emitted"
    )

  def shutdown(self) -> None:
    """Invalidate all work and shut down the executor exactly once."""

    if self._released or self._shutdown:
      return
    self._pending_request = None
    self._submitted_request_sequence = None
    self._submitted_policy_generation = None
    self._submitted_source_timestamp = None
    self._submitted_source_tick = None
    self._latest_request_sequence = None
    self._accepted_plan = None
    self._accepted_request_sequence = None
    self._accepted_request_provenance = None
    self._new_plan_since_consume = False
    self._fallback_ticks = 0
    self._plan_first_use_tick = None
    self._last_control_tick = None
    self._fault = None
    self._shutdown_executor()

  def _submit_pending_if_idle(self, events: list[SchedulerEvent]) -> None:
    if self._pending_request is None or self._submitted_request_sequence is not None:
      return
    if self._fault is not None:
      return
    request = self._pending_request
    self._pending_request = None
    # Submit may start work before raising. Retain ownership conservatively
    # until a matching completion or explicit executor shutdown proves cleanup.
    self._submitted_request_sequence = request.request_sequence_id
    self._submitted_policy_generation = request.policy_generation
    self._submitted_source_timestamp = request.source_timestamp
    self._submitted_source_tick = request.source_tick
    try:
      self.executor.submit(request)
    except BaseException as exc:
      events.append(
        self._set_fault(
          SchedulerStatus.PLANNER_EXCEPTION,
          f"executor submit failed: {exc}",
          request_sequence_id=request.request_sequence_id,
        )
      )
      return

  def _handle_result(self, result: PlannerResult) -> SchedulerEvent | None:
    if result.request_sequence_id != self._submitted_request_sequence:
      return None
    if result.policy_generation != self._submitted_policy_generation:
      return None
    if result.policy_generation != self._policy_generation:
      return None
    if (result.plan is None) == (result.error is None):
      raise SchedulerError("result must contain exactly one plan or error")
    if result.error is not None:
      if not isinstance(result.error, BaseException):
        raise SchedulerError("result error must be an exception")
      return self._set_fault(
        SchedulerStatus.PLANNER_EXCEPTION,
        f"planner failed: {result.error}",
        request_sequence_id=result.request_sequence_id,
      )
    plan = result.plan
    if not isinstance(plan, Plan):
      return self._set_fault(
        SchedulerStatus.INVALID_OUTPUT,
        "planner result contains a malformed plan",
        result.request_sequence_id,
      )
    # Revalidate identifiers at the external executor boundary as well as
    # construction: Python annotations and frozen dataclasses are not a guard.
    _validate_integer(plan.plan_sequence_id, "plan_sequence_id", minimum=1)
    _validate_tick(plan.valid_from_tick, "valid_from_tick")
    _validate_timestamp(plan.source_timestamp, "source_timestamp")
    if plan.actions.ndim != 2 or 0 in plan.actions.shape:
      raise SchedulerError("actions must have shape [horizon, action_dim]")
    if plan.plan_sequence_id <= self._accepted_plan_sequence:
      return self._set_fault(
        SchedulerStatus.INVALID_OUTPUT,
        "planner result has a non-newer plan sequence",
        result.request_sequence_id,
        plan.plan_sequence_id,
      )
    if self.expected_contract_identity is not None and (
      plan.schema_metadata.get("contract") != self.expected_contract_identity
    ):
      return self._set_fault(
        SchedulerStatus.INVALID_OUTPUT,
        "planner result has an unexpected contract identity",
        result.request_sequence_id,
        plan.plan_sequence_id,
      )
    if self.expected_action_dim is not None and (
      plan.actions.shape[1] != self.expected_action_dim
    ):
      return self._set_fault(
        SchedulerStatus.INVALID_OUTPUT,
        "planner result has an unexpected action dimension",
        result.request_sequence_id,
        plan.plan_sequence_id,
      )
    try:
      if not np.isfinite(plan.actions).all():
        raise SchedulerError("plan action trajectory is non-finite")
    except (TypeError, ValueError) as exc:
      return self._set_fault(
        SchedulerStatus.INVALID_OUTPUT,
        f"invalid planner output: {exc}",
        result.request_sequence_id,
        plan.plan_sequence_id,
      )
    if self._submitted_source_timestamp is None or self._submitted_source_tick is None:
      return self._set_fault(
        SchedulerStatus.INVALID_OUTPUT,
        "accepted request provenance is unavailable",
        result.request_sequence_id,
        plan.plan_sequence_id,
      )
    accepted_provenance = AcceptedRequestProvenance(
      request_sequence_id=result.request_sequence_id,
      policy_generation=result.policy_generation,
      source_timestamp=self._submitted_source_timestamp,
      source_tick=self._submitted_source_tick,
    )
    self._accepted_plan = plan
    self._accepted_plan_sequence = plan.plan_sequence_id
    self._accepted_request_sequence = result.request_sequence_id
    self._accepted_request_provenance = accepted_provenance
    self._new_plan_since_consume = True
    self._fallback_ticks = 0
    self._plan_first_use_tick = None
    self._fallback_warning_emitted = False
    return SchedulerEvent(
      SchedulerStatus.ACTION,
      "plan accepted",
      result.request_sequence_id,
      plan.plan_sequence_id,
    )

  def _set_fault(
    self,
    status: SchedulerStatus,
    detail: str,
    request_sequence_id: int | None = None,
    plan_sequence_id: int | None = None,
  ) -> SchedulerEvent:
    event = SchedulerEvent(status, detail, request_sequence_id, plan_sequence_id)
    self._fault = event
    return event

  def _closed_step(
    self,
    control_tick: int,
    status: SchedulerStatus,
    detail: str,
    event: SchedulerEvent | None = None,
    plan_sequence_id: int | None = None,
  ) -> SchedulerStep:
    if event is None:
      event = SchedulerEvent(status, detail, plan_sequence_id=plan_sequence_id)
    return SchedulerStep(status, None, control_tick, event, plan_sequence_id)

  def _shutdown_executor(self) -> None:
    if not self._shutdown:
      try:
        self.executor.shutdown()
      finally:
        self._shutdown = True

  def _require_open(self) -> None:
    if self._released:
      raise SchedulerError("scheduler is released")
    if self._shutdown:
      raise SchedulerError("scheduler is shut down")


def _freeze_context(value: object) -> object:
  """Copy a bounded serializable context into immutable plain values."""
  if value is None or isinstance(value, (bool, int, str, bytes)):
    return value
  if isinstance(value, float):
    if not np.isfinite(value):
      raise SchedulerError("planner context floats must be finite")
    return value
  if isinstance(value, np.generic):
    return _freeze_context(value.item())
  if isinstance(value, np.ndarray):
    if value.dtype.kind in "fiu" and not np.isfinite(value).all():
      raise SchedulerError("planner context arrays must be finite")
    array = cast(Any, value)
    return _freeze_context(array.tolist())
  if isinstance(value, Mapping):
    return FrozenContext(cast(Mapping[str, object], value))
  if isinstance(value, (list, tuple)):
    return tuple(_freeze_context(item) for item in value)
  if isinstance(value, (set, frozenset)):
    raise SchedulerError("planner context sets are not deterministically serializable")
  raise SchedulerError(
    "planner context must contain only plainly serializable scalar, mapping, "
    "sequence or NumPy values"
  )


def _validate_integer(value: int, name: str, *, minimum: int = 0) -> None:
  if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
    raise SchedulerError(f"{name} must be an integer, not boolean")
  if value < minimum:
    raise SchedulerError(f"{name} must be an integer >= {minimum}")


def _validate_tick(value: int, name: str) -> None:
  _validate_integer(value, name)


def _validate_timestamp(value: float, name: str) -> None:
  try:
    scalar = np.asarray(value)
    if scalar.ndim != 0 or scalar.dtype.kind not in "iuf":
      raise SchedulerError(f"{name} must be a finite scalar number")
    finite = bool(np.isfinite(float(scalar)))
  except (TypeError, ValueError, OverflowError) as exc:
    raise SchedulerError(f"{name} must be a finite scalar number") from exc
  if not finite:
    raise SchedulerError(f"{name} must be finite")


__all__ = [
  "AcceptedRequestProvenance",
  "FrozenContext",
  "Plan",
  "PlannerExecutor",
  "PlannerRequest",
  "PlannerResult",
  "RuntimeScheduler",
  "SchedulerError",
  "SchedulerEvent",
  "SchedulerStatus",
  "SchedulerStep",
]
