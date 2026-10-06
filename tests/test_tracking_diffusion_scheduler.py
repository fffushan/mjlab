"""Focused tests for the simulator-independent D5 scheduler."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import cast

import numpy as np
import pytest

from mjlab.tasks.tracking.diffusion.scheduler import (
  Plan,
  PlannerRequest,
  PlannerResult,
  RuntimeScheduler,
  SchedulerError,
  SchedulerStatus,
)


class FakeExecutor:
  """Deterministic non-threaded executor used by all scheduler tests."""

  def __init__(self) -> None:
    self.submitted: list[PlannerRequest] = []
    self.completed: deque[PlannerResult] = deque()
    self.shutdown_count = 0

  def submit(self, request: PlannerRequest) -> None:
    self.submitted.append(request)

  def poll(self) -> tuple[PlannerResult, ...]:
    results = tuple(self.completed)
    self.completed.clear()
    return results

  def shutdown(self) -> None:
    self.shutdown_count += 1

  def complete(
    self,
    request: PlannerRequest,
    plan: Plan | None = None,
    error: BaseException | None = None,
  ) -> None:
    self.completed.append(
      PlannerResult(
        request_sequence_id=request.request_sequence_id,
        policy_generation=request.policy_generation,
        plan=plan,
        error=error,
      )
    )


def make_plan(
  sequence: int,
  *,
  start: int = 0,
  values: tuple[float, ...] = (1.0, 2.0, 3.0),
  action_dim: int = 1,
  contract: str = "active-167",
) -> Plan:
  actions = np.asarray(values, dtype=np.float64)[:, None]
  if action_dim != 1:
    actions = np.repeat(actions, action_dim, axis=1)
  return Plan(
    plan_sequence_id=sequence,
    source_timestamp=float(start),
    valid_from_tick=start,
    actions=actions,
    schema_metadata={"contract": contract, "guidance": "generic"},
  )


def make_scheduler(
  *,
  max_fallback_ticks: int = 2,
  expected_action_dim: int | None = None,
  expected_contract_identity: str | None = None,
) -> tuple[RuntimeScheduler, FakeExecutor]:
  executor = FakeExecutor()
  return (
    RuntimeScheduler(
      executor,
      max_fallback_ticks=max_fallback_ticks,
      expected_action_dim=expected_action_dim,
      expected_contract_identity=expected_contract_identity,
    ),
    executor,
  )


def submit_and_accept(
  scheduler: RuntimeScheduler,
  executor: FakeExecutor,
  *,
  sequence: int,
  source_tick: int = 0,
  plan: Plan | None = None,
) -> PlannerRequest:
  request = scheduler.request(
    np.asarray([source_tick, 9.0]),
    source_timestamp=float(source_tick),
    source_tick=source_tick,
  )
  scheduler.poll()
  executor.complete(request, plan or make_plan(sequence, start=source_tick))
  scheduler.poll()
  return request


def test_accepted_request_provenance_tracks_origin_and_clears_with_plan() -> None:
  scheduler, executor = make_scheduler()
  request = scheduler.request(np.ones(1), source_timestamp=4.5, source_tick=4)
  scheduler.poll()
  executor.complete(request, make_plan(1, start=9))
  assert scheduler.poll()[0].status is SchedulerStatus.ACTION
  provenance = scheduler.accepted_request_provenance
  assert provenance is not None
  assert provenance.request_sequence_id == request.request_sequence_id
  assert provenance.policy_generation == request.policy_generation
  assert provenance.source_timestamp == 4.5
  assert provenance.source_tick == 4
  scheduler.reset()
  assert scheduler.accepted_plan is None
  assert scheduler.accepted_request_provenance is None


def test_accepted_request_provenance_clears_on_release() -> None:
  scheduler, executor = make_scheduler()
  request = scheduler.request(np.ones(1), source_timestamp=0.0, source_tick=0)
  scheduler.poll()
  executor.complete(request, make_plan(1))
  scheduler.poll()
  assert scheduler.accepted_request_provenance is not None
  scheduler.release()
  assert scheduler.accepted_plan is None
  assert scheduler.accepted_request_provenance is None


def test_accepted_request_provenance_clears_on_shutdown() -> None:
  scheduler, executor = make_scheduler()
  request = scheduler.request(np.ones(1), source_timestamp=0.0, source_tick=0)
  scheduler.poll()
  executor.complete(request, make_plan(1))
  scheduler.poll()
  assert scheduler.accepted_request_provenance is not None
  scheduler.shutdown()
  assert scheduler.accepted_plan is None
  assert scheduler.accepted_request_provenance is None


def test_latest_request_replaces_bounded_pending_request() -> None:
  scheduler, executor = make_scheduler()
  first = scheduler.request(np.array([1.0]), source_timestamp=0.0, source_tick=0)
  second = scheduler.request(np.array([2.0]), source_timestamp=0.02, source_tick=1)

  assert scheduler.pending_request is second
  scheduler.poll()
  assert executor.submitted == [second]
  assert first.request_sequence_id < second.request_sequence_id
  assert np.array_equal(second.input_snapshot, [2.0])


def test_newer_requests_wait_for_the_single_in_flight_request() -> None:
  scheduler, executor = make_scheduler()
  first = scheduler.request(np.array([0.0]), source_timestamp=0.0, source_tick=0)
  scheduler.poll()
  newest = first
  for tick in range(1, 5):
    newest = scheduler.request(
      np.array([float(tick)]), source_timestamp=tick * 0.02, source_tick=tick
    )
    scheduler.poll()
    assert executor.submitted == [first]

  executor.complete(first, make_plan(1))
  events = scheduler.poll()
  assert events[0].status is SchedulerStatus.ACTION
  assert executor.submitted == [first, newest]
  assert scheduler.pending_request is None

  executor.complete(newest, make_plan(2))
  events = scheduler.poll()
  assert events[0].status is SchedulerStatus.ACTION


def test_old_result_cannot_overwrite_newer_accepted_plan() -> None:
  scheduler, executor = make_scheduler()
  first = submit_and_accept(scheduler, executor, sequence=1)
  first_step = scheduler.consume(control_tick=0, timestamp=0.0)
  assert first_step.action is not None
  assert first_step.action.tolist() == [1.0]

  second = scheduler.request(np.array([2.0]), source_timestamp=0.02, source_tick=1)
  scheduler.poll()
  executor.complete(second, make_plan(2))
  scheduler.poll()
  second_step = scheduler.consume(control_tick=1, timestamp=0.02)
  assert second_step.action is not None
  assert second_step.action.tolist() == [2.0]

  executor.complete(first, make_plan(99, values=(9.0,)))
  events = scheduler.poll()
  assert events == ()
  assert scheduler.accepted_plan is not None
  assert scheduler.accepted_plan.plan_sequence_id == 2


def test_plan_actions_are_selected_at_consecutive_ticks() -> None:
  scheduler, executor = make_scheduler()
  submit_and_accept(scheduler, executor, sequence=1)

  first = scheduler.consume(control_tick=0, timestamp=0.0)
  second = scheduler.consume(control_tick=1, timestamp=0.02)
  third = scheduler.consume(control_tick=2, timestamp=0.04)
  assert first.status is SchedulerStatus.ACTION
  assert second.status is SchedulerStatus.FALLBACK_ACTION
  assert third.status is SchedulerStatus.FALLBACK_ACTION
  np.testing.assert_array_equal(first.action, [1.0])
  np.testing.assert_array_equal(second.action, [2.0])
  np.testing.assert_array_equal(third.action, [3.0])


def test_late_result_uses_only_bounded_previous_plan_suffix() -> None:
  scheduler, executor = make_scheduler(max_fallback_ticks=1)
  submit_and_accept(scheduler, executor, sequence=1)
  assert (
    scheduler.consume(control_tick=0, timestamp=0.0).status is SchedulerStatus.ACTION
  )
  assert (
    scheduler.consume(control_tick=1, timestamp=0.02).status
    is SchedulerStatus.FALLBACK_ACTION
  )
  exhausted = scheduler.consume(control_tick=2, timestamp=0.04)
  assert exhausted.status is SchedulerStatus.PLAN_EXHAUSTED
  assert exhausted.action is None


def test_plan_exhaustion_fails_closed_even_with_long_trajectory() -> None:
  scheduler, executor = make_scheduler(max_fallback_ticks=0)
  submit_and_accept(scheduler, executor, sequence=1)
  assert scheduler.consume(control_tick=0, timestamp=0.0).action is not None
  exhausted = scheduler.consume(control_tick=1, timestamp=0.02)
  assert exhausted.status is SchedulerStatus.PLAN_EXHAUSTED
  assert exhausted.event is not None


def test_reset_invalidates_accepted_plan_and_drains_old_in_flight_request() -> None:
  scheduler, executor = make_scheduler()
  accepted = submit_and_accept(scheduler, executor, sequence=1)
  assert scheduler.accepted_plan is not None

  old = scheduler.request(np.array([2.0]), source_timestamp=0.02, source_tick=1)
  scheduler.poll()
  event = scheduler.reset()
  assert event.status is SchedulerStatus.RESET
  assert scheduler.accepted_plan is None

  new = scheduler.request(np.array([3.0]), source_timestamp=0.04, source_tick=2)
  scheduler.poll()
  assert executor.submitted == [accepted, old]
  executor.complete(old, make_plan(2))
  assert scheduler.poll() == ()
  assert executor.submitted == [accepted, old, new]
  executor.complete(new, make_plan(3))
  assert scheduler.poll()[0].status is SchedulerStatus.ACTION


def test_policy_switch_drains_old_in_flight_request_before_new_submission() -> None:
  scheduler, executor = make_scheduler()
  old = scheduler.request(np.array([1.0]), source_timestamp=0.0, source_tick=0)
  scheduler.poll()
  event = scheduler.switch_policy("new-policy")
  assert event.status is SchedulerStatus.POLICY_SWITCH

  new = scheduler.request(np.array([2.0]), source_timestamp=0.02, source_tick=1)
  scheduler.poll()
  assert executor.submitted == [old]
  executor.complete(old, make_plan(1))
  assert scheduler.poll() == ()
  assert executor.submitted == [old, new]
  executor.complete(new, make_plan(2))
  assert scheduler.poll()[0].status is SchedulerStatus.ACTION


def test_release_prevents_every_later_action_and_shuts_down_once() -> None:
  scheduler, executor = make_scheduler()
  request = submit_and_accept(scheduler, executor, sequence=1)
  release = scheduler.release()
  assert release.status is SchedulerStatus.RELEASED
  executor.complete(request, make_plan(2))
  step = scheduler.consume(control_tick=0, timestamp=0.0)
  assert step.status is SchedulerStatus.RELEASED
  assert step.action is None
  repeated = scheduler.consume(control_tick=0, timestamp=0.0)
  assert repeated.status is SchedulerStatus.RELEASED
  assert repeated.action is None
  assert executor.shutdown_count == 1
  scheduler.shutdown()
  assert executor.shutdown_count == 1


def test_policy_switch_ignores_late_old_result() -> None:
  scheduler, executor = make_scheduler()
  old = scheduler.request(np.array([1.0]), source_timestamp=0.0, source_tick=0)
  scheduler.poll()
  event = scheduler.switch_policy("new-policy")
  assert event.status is SchedulerStatus.POLICY_SWITCH
  executor.complete(old, make_plan(1))
  assert scheduler.poll() == ()
  assert (
    scheduler.consume(control_tick=0, timestamp=0.0).status is SchedulerStatus.NO_PLAN
  )


def test_duplicate_plan_sequence_fails_closed() -> None:
  scheduler, executor = make_scheduler()
  request = submit_and_accept(scheduler, executor, sequence=1)
  assert scheduler.consume(control_tick=0, timestamp=0.0).action is not None
  newer_request = scheduler.request(
    np.array([2.0]), source_timestamp=0.02, source_tick=1
  )
  scheduler.poll()
  executor.complete(newer_request, make_plan(1))
  events = scheduler.poll()
  assert events[0].status is SchedulerStatus.INVALID_OUTPUT
  step = scheduler.consume(control_tick=1, timestamp=0.02)
  assert step.status is SchedulerStatus.INVALID_OUTPUT
  assert step.action is None
  assert request.request_sequence_id != newer_request.request_sequence_id


def test_planner_exception_fails_closed_until_reset() -> None:
  scheduler, executor = make_scheduler()
  request = scheduler.request(np.array([1.0]), source_timestamp=0.0, source_tick=0)
  scheduler.poll()
  executor.complete(request, error=RuntimeError("planner failed"))
  events = scheduler.poll()
  assert events[0].status is SchedulerStatus.PLANNER_EXCEPTION
  step = scheduler.consume(control_tick=0, timestamp=0.0)
  assert step.status is SchedulerStatus.PLANNER_EXCEPTION
  assert step.action is None


def test_non_finite_planner_output_fails_closed_with_event() -> None:
  scheduler, executor = make_scheduler()
  request = scheduler.request(np.array([1.0]), source_timestamp=0.0, source_tick=0)
  scheduler.poll()
  executor.complete(request, make_plan(1, values=(np.nan,)))
  events = scheduler.poll()
  assert events[0].status is SchedulerStatus.INVALID_OUTPUT
  step = scheduler.consume(control_tick=0, timestamp=0.0)
  assert step.status is SchedulerStatus.INVALID_OUTPUT
  assert step.action is None


def test_snapshots_and_returned_actions_are_owned_copies() -> None:
  scheduler, executor = make_scheduler()
  snapshot = np.array([1.0, 2.0])
  request = scheduler.request(snapshot, source_timestamp=0.0, source_tick=0)
  snapshot[0] = 99.0
  assert request.input_snapshot.flags.writeable is False
  assert request.input_snapshot[0] == 1.0

  submit_and_accept(scheduler, executor, sequence=1)
  first = scheduler.consume(control_tick=0, timestamp=0.0)
  assert first.action is not None
  first.action[0] = 88.0
  second = scheduler.consume(control_tick=1, timestamp=0.02)
  assert second.action is not None
  assert second.action[0] == 2.0
  assert not np.shares_memory(first.action, second.action)


def test_shutdown_invalidates_pending_and_accepted_work() -> None:
  scheduler, executor = make_scheduler()
  accepted = submit_and_accept(scheduler, executor, sequence=1)
  active = scheduler.request(np.array([2.0]), source_timestamp=0.02, source_tick=1)
  scheduler.poll()
  pending = scheduler.request(np.array([3.0]), source_timestamp=0.04, source_tick=2)
  scheduler.shutdown()

  assert scheduler.pending_request is None
  assert scheduler.accepted_plan is None
  assert executor.shutdown_count == 1
  executor.complete(active, make_plan(2))
  executor.complete(pending, make_plan(3))
  assert scheduler.poll()[0].status is SchedulerStatus.SHUTDOWN
  assert (
    scheduler.consume(control_tick=0, timestamp=0.0).status is SchedulerStatus.SHUTDOWN
  )
  with pytest.raises(SchedulerError, match="shut down"):
    scheduler.request(np.array([4.0]), source_timestamp=0.06, source_tick=3)
  scheduler.shutdown()
  assert executor.shutdown_count == 1
  assert accepted.request_sequence_id < active.request_sequence_id


def test_executor_shutdown_and_lifecycle_are_deterministic() -> None:
  scheduler, executor = make_scheduler()
  scheduler.shutdown()
  scheduler.shutdown()
  assert executor.shutdown_count == 1
  with pytest.raises(SchedulerError, match="shut down"):
    scheduler.request(np.array([1.0]), source_timestamp=0.0, source_tick=0)


def test_wrong_action_width_fails_closed_at_scheduler_boundary() -> None:
  scheduler, executor = make_scheduler(expected_action_dim=2)
  request = scheduler.request(np.array([1.0]), source_timestamp=0.0, source_tick=0)
  scheduler.poll()
  executor.complete(request, make_plan(1, action_dim=1))
  events = scheduler.poll()
  assert events[0].status is SchedulerStatus.INVALID_OUTPUT
  step = scheduler.consume(control_tick=0, timestamp=0.0)
  assert step.status is SchedulerStatus.INVALID_OUTPUT
  assert step.action is None


def test_wrong_contract_identity_fails_closed_at_scheduler_boundary() -> None:
  scheduler, executor = make_scheduler(expected_contract_identity="active-167")
  request = scheduler.request(np.array([1.0]), source_timestamp=0.0, source_tick=0)
  scheduler.poll()
  executor.complete(request, make_plan(1, contract="other-contract"))
  events = scheduler.poll()
  assert events[0].status is SchedulerStatus.INVALID_OUTPUT
  assert "contract" in events[0].detail


@pytest.mark.parametrize("value", [np.nan, np.inf, -1, 1.5, True])
def test_max_fallback_ticks_requires_nonnegative_integer(value: object) -> None:
  with pytest.raises(SchedulerError, match="nonnegative integer"):
    RuntimeScheduler(FakeExecutor(), max_fallback_ticks=cast(int, value))


def test_malformed_planner_plan_fails_closed_and_does_not_stick() -> None:
  scheduler, executor = make_scheduler()
  request = scheduler.request(np.array([1.0]), source_timestamp=0.0, source_tick=0)
  scheduler.poll()
  executor.completed.append(
    PlannerResult(
      request_sequence_id=request.request_sequence_id,
      policy_generation=request.policy_generation,
      plan=cast(Plan, object()),
    )
  )
  events = scheduler.poll()
  assert events[0].status is SchedulerStatus.INVALID_OUTPUT

  scheduler.reset()
  replacement = scheduler.request(np.array([2.0]), source_timestamp=0.02, source_tick=1)
  scheduler.poll()
  assert executor.submitted == [request, replacement]


def test_malformed_request_and_plan_arrays_raise_scheduler_error() -> None:
  scheduler, _executor = make_scheduler()
  with pytest.raises(SchedulerError, match="numeric"):
    scheduler.request(
      cast(np.ndarray, [[1.0], [1.0, 2.0]]),
      source_timestamp=0.0,
      source_tick=0,
    )
  with pytest.raises(SchedulerError, match="numeric"):
    scheduler.request(
      cast(np.ndarray, ["not-a-number"]), source_timestamp=0.0, source_tick=0
    )
  with pytest.raises(SchedulerError, match="numeric"):
    Plan(
      plan_sequence_id=1,
      source_timestamp=0.0,
      valid_from_tick=0,
      actions=cast(np.ndarray, [[1.0], [1.0, 2.0]]),
      schema_metadata={},
    )
  with pytest.raises(SchedulerError, match="numeric"):
    Plan(
      plan_sequence_id=1,
      source_timestamp=0.0,
      valid_from_tick=0,
      actions=cast(np.ndarray, [["not-a-number"]]),
      schema_metadata={},
    )


def test_stale_and_current_results_do_not_reorder_pending_submission() -> None:
  for stale_first in (True, False):
    scheduler, executor = make_scheduler()
    old = scheduler.request(np.array([1.0]), source_timestamp=0.0, source_tick=0)
    scheduler.poll()
    scheduler.reset()
    current = scheduler.request(np.array([2.0]), source_timestamp=0.02, source_tick=1)
    scheduler.poll()
    executor.complete(old, make_plan(1))
    scheduler.poll()
    pending = scheduler.request(np.array([3.0]), source_timestamp=0.04, source_tick=2)
    scheduler.poll()

    stale = PlannerResult(
      request_sequence_id=old.request_sequence_id,
      policy_generation=old.policy_generation,
      plan=make_plan(99),
    )
    current_result = PlannerResult(
      request_sequence_id=current.request_sequence_id,
      policy_generation=current.policy_generation,
      plan=make_plan(2),
    )
    results = (stale, current_result) if stale_first else (current_result, stale)
    executor.completed.extend(results)
    events = scheduler.poll()

    assert any(event.status is SchedulerStatus.ACTION for event in events)
    assert scheduler.accepted_plan is not None
    assert scheduler.accepted_plan.plan_sequence_id == 2
    assert executor.submitted == [old, current, pending]
    assert scheduler.pending_request is None


def test_default_period_exposes_consistent_fifty_hz_metadata() -> None:
  scheduler, _executor = make_scheduler()
  assert RuntimeScheduler.DEFAULT_CONTROL_HZ == 50
  assert scheduler.period_seconds == pytest.approx(0.020)
  assert scheduler.control_frequency_hz == pytest.approx(50.0)
  with pytest.raises(SchedulerError, match="finite"):
    scheduler.consume(control_tick=0, timestamp=np.inf)
  with pytest.raises(SchedulerError, match="finite scalar"):
    scheduler.consume(control_tick=0, timestamp=cast(float, np.array([0.0, 0.02])))


@pytest.mark.parametrize("attributed", [False, True])
def test_malformed_completion_preserves_active_work_across_reset(
  attributed: bool,
) -> None:
  scheduler, executor = make_scheduler()
  active = scheduler.request(np.ones(1), source_timestamp=0.0, source_tick=0)
  scheduler.poll()
  pending = scheduler.request(np.ones(1), source_timestamp=0.02, source_tick=1)
  if attributed:
    malformed = type(
      "Impostor", (), {"request_sequence_id": active.request_sequence_id}
    )()
  else:
    malformed = object()
  executor.completed.append(cast(PlannerResult, malformed))
  assert scheduler.poll()[0].status is SchedulerStatus.INVALID_OUTPUT
  assert executor.submitted == [active]
  assert scheduler.pending_request is pending
  assert scheduler.consume(control_tick=0, timestamp=0.0).action is None

  scheduler.reset()
  replacement = scheduler.request(np.ones(1), source_timestamp=0.04, source_tick=2)
  scheduler.poll()
  assert executor.submitted == [active]
  executor.complete(active, make_plan(1))
  assert scheduler.poll() == ()
  assert executor.submitted == [active, replacement]
  executor.complete(replacement, make_plan(2, start=2))
  assert (
    scheduler.consume(control_tick=2, timestamp=0.04).status is SchedulerStatus.ACTION
  )


@pytest.mark.parametrize(
  "value",
  [np.nan, np.inf, 1.0, 1.5, True, False, np.bool_(True), -1, "1", np.array([1, 2])],
)
def test_sequence_and_generation_constructors_reject_invalid_ids(value: object) -> None:
  invalid = cast(int, value)
  with pytest.raises(SchedulerError, match="plan_sequence_id"):
    make_plan(invalid)
  for sequence, generation, name in (
    (invalid, 0, "request_sequence_id"),
    (1, invalid, "policy_generation"),
  ):
    with pytest.raises(SchedulerError, match=name):
      PlannerRequest(np.ones(1), 0.0, 0, sequence, generation)
    with pytest.raises(SchedulerError, match=name):
      PlannerResult(sequence, generation, plan=make_plan(1))


def test_zero_sequence_ids_are_rejected_but_zero_generation_is_valid() -> None:
  with pytest.raises(SchedulerError, match="plan_sequence_id"):
    make_plan(0)
  with pytest.raises(SchedulerError, match="request_sequence_id"):
    PlannerRequest(np.ones(1), 0.0, 0, 0, 0)
  with pytest.raises(SchedulerError, match="request_sequence_id"):
    PlannerResult(0, 0, plan=make_plan(1))
  request = PlannerRequest(
    np.ones(1), 0.0, 0, cast(int, np.int64(1)), cast(int, np.int64(0))
  )
  assert request.request_sequence_id == 1
  assert make_plan(cast(int, np.int64(1))).plan_sequence_id == 1


@pytest.mark.parametrize("field", ["request_sequence_id", "policy_generation"])
def test_malformed_completion_ids_do_not_release_active_slot(field: str) -> None:
  scheduler, executor = make_scheduler()
  active = scheduler.request(np.ones(1), source_timestamp=0.0, source_tick=0)
  scheduler.poll()
  result = PlannerResult(
    active.request_sequence_id, active.policy_generation, plan=make_plan(1)
  )
  object.__setattr__(result, field, float(getattr(result, field)))
  executor.completed.append(result)
  assert scheduler.poll()[0].status is SchedulerStatus.INVALID_OUTPUT
  scheduler.reset()
  replacement = scheduler.request(np.ones(1), source_timestamp=0.02, source_tick=1)
  scheduler.poll()
  assert executor.submitted == [active]
  executor.complete(active, make_plan(1))
  scheduler.poll()
  assert executor.submitted == [active, replacement]


@pytest.mark.parametrize("value", [np.nan, np.inf, True, 1.0])
def test_plan_sequence_is_revalidated_at_executor_boundary(value: object) -> None:
  scheduler, executor = make_scheduler()
  active = scheduler.request(np.ones(1), source_timestamp=0.0, source_tick=0)
  scheduler.poll()
  plan = make_plan(1)
  object.__setattr__(plan, "plan_sequence_id", value)
  executor.complete(active, plan)
  assert scheduler.poll()[0].status is SchedulerStatus.INVALID_OUTPUT
  assert scheduler.consume(control_tick=0, timestamp=0.0).action is None


@pytest.mark.parametrize(
  "value",
  [
    np.array([0.02, 0.03]),
    np.array(True),
    True,
    "0.02",
    object(),
    np.nan,
    np.inf,
    0.0,
    -0.02,
    0.02j,
  ],
)
def test_period_requires_a_finite_positive_real_scalar(value: object) -> None:
  with pytest.raises(SchedulerError, match="period_seconds"):
    RuntimeScheduler(FakeExecutor(), period_seconds=cast(float, value))


def test_fallback_expires_on_elapsed_ticks_not_consume_calls() -> None:
  scheduler, executor = make_scheduler(max_fallback_ticks=1)
  submit_and_accept(
    scheduler,
    executor,
    sequence=1,
    plan=make_plan(1, values=tuple(float(i) for i in range(8))),
  )
  assert (
    scheduler.consume(control_tick=0, timestamp=0.0).status is SchedulerStatus.ACTION
  )
  expired = scheduler.consume(control_tick=5, timestamp=0.10)
  assert expired.status is SchedulerStatus.PLAN_EXHAUSTED
  assert expired.action is None


def test_tick_jump_within_fallback_bound_selects_execution_tick_row() -> None:
  scheduler, executor = make_scheduler(max_fallback_ticks=2)
  submit_and_accept(
    scheduler, executor, sequence=1, plan=make_plan(1, values=(10.0, 11.0, 12.0, 13.0))
  )
  assert scheduler.consume(control_tick=0, timestamp=0.0).action is not None
  step = scheduler.consume(control_tick=2, timestamp=0.04)
  assert step.status is SchedulerStatus.FALLBACK_ACTION
  np.testing.assert_array_equal(step.action, [12.0])
  assert scheduler.consume(control_tick=3, timestamp=0.06).action is None


def test_future_plan_replacement_fails_closed_until_valid() -> None:
  scheduler, executor = make_scheduler(max_fallback_ticks=10)
  submit_and_accept(
    scheduler, executor, sequence=1, plan=make_plan(1, values=(10.0,) * 8)
  )
  assert scheduler.consume(control_tick=0, timestamp=0.0).action is not None
  submit_and_accept(
    scheduler,
    executor,
    sequence=2,
    source_tick=1,
    plan=make_plan(2, start=3, values=(20.0, 21.0)),
  )
  for tick in (1, 2):
    step = scheduler.consume(control_tick=tick, timestamp=tick * 0.02)
    assert step.status is SchedulerStatus.PLAN_NOT_YET_VALID
    assert step.action is None
    assert step.plan_sequence_id == 2
  first = scheduler.consume(control_tick=3, timestamp=0.06)
  assert first.status is SchedulerStatus.ACTION
  np.testing.assert_array_equal(first.action, [20.0])
  last = scheduler.consume(control_tick=4, timestamp=0.08)
  np.testing.assert_array_equal(last.action, [21.0])
  assert (
    scheduler.consume(control_tick=5, timestamp=0.10).status
    is SchedulerStatus.PLAN_EXHAUSTED
  )


def test_expired_replacement_does_not_resurrect_old_plan() -> None:
  scheduler, executor = make_scheduler(max_fallback_ticks=10)
  submit_and_accept(
    scheduler, executor, sequence=1, plan=make_plan(1, values=(10.0,) * 8)
  )
  assert scheduler.consume(control_tick=0, timestamp=0.0).action is not None
  submit_and_accept(
    scheduler, executor, sequence=2, source_tick=1, plan=make_plan(2, values=(20.0,))
  )
  expired = scheduler.consume(control_tick=1, timestamp=0.02)
  assert expired.status is SchedulerStatus.PLAN_EXHAUSTED
  assert expired.action is None
  assert scheduler.accepted_plan is None
  assert (
    scheduler.consume(control_tick=2, timestamp=0.04).status is SchedulerStatus.NO_PLAN
  )


def test_wrong_generation_cannot_clear_current_active_slot() -> None:
  scheduler, executor = make_scheduler()
  active = scheduler.request(np.ones(1), source_timestamp=0.0, source_tick=0)
  scheduler.poll()
  pending = scheduler.request(np.ones(1), source_timestamp=0.02, source_tick=1)
  executor.completed.append(
    PlannerResult(
      active.request_sequence_id, active.policy_generation + 1, plan=make_plan(99)
    )
  )
  assert scheduler.poll() == ()
  assert executor.submitted == [active]
  assert scheduler.accepted_plan is None
  executor.complete(active, make_plan(1))
  scheduler.poll()
  assert executor.submitted == [active, pending]


@pytest.mark.parametrize("operation", ["submit", "poll"])
def test_executor_exception_preserves_active_ownership_during_reset(
  operation: str,
) -> None:
  class FaultyExecutor(FakeExecutor):
    fault_enabled = True

    def submit(self, request: PlannerRequest) -> None:
      super().submit(request)
      if operation == "submit" and self.fault_enabled:
        self.fault_enabled = False
        raise RuntimeError("work started before submit raised")

    def poll(self) -> tuple[PlannerResult, ...]:
      if operation == "poll" and self.fault_enabled:
        self.fault_enabled = False
        raise RuntimeError("poll failed while work may still run")
      return super().poll()

  executor = FaultyExecutor()
  scheduler = RuntimeScheduler(executor)
  active = scheduler.request(np.ones(1), source_timestamp=0.0, source_tick=0)
  assert scheduler.poll()[0].status is SchedulerStatus.PLANNER_EXCEPTION
  assert scheduler.consume(control_tick=0, timestamp=0.0).action is None
  scheduler.reset()
  replacement = scheduler.request(np.ones(1), source_timestamp=0.02, source_tick=1)
  scheduler.poll()
  assert executor.submitted == [active]
  executor.complete(active, make_plan(1))
  scheduler.poll()
  assert executor.submitted == [active, replacement]


def test_malformed_active_envelope_fails_closed_without_assertion() -> None:
  scheduler, executor = make_scheduler()
  active = scheduler.request(np.ones(1), source_timestamp=0.0, source_tick=0)
  scheduler.poll()
  result = PlannerResult(
    active.request_sequence_id, active.policy_generation, plan=make_plan(1)
  )
  object.__setattr__(result, "plan", None)
  executor.completed.append(result)
  assert scheduler.poll()[0].status is SchedulerStatus.INVALID_OUTPUT
  assert scheduler.consume(control_tick=0, timestamp=0.0).action is None


def test_wrong_plan_shape_is_revalidated_at_executor_boundary() -> None:
  scheduler, executor = make_scheduler()
  active = scheduler.request(np.ones(1), source_timestamp=0.0, source_tick=0)
  scheduler.poll()
  plan = make_plan(1)
  object.__setattr__(plan, "actions", np.ones(3))
  executor.complete(active, plan)
  assert scheduler.poll()[0].status is SchedulerStatus.INVALID_OUTPUT
  assert scheduler.consume(control_tick=0, timestamp=0.0).action is None


def test_optional_request_context_is_owned_frozen_and_pickleable() -> None:
  import pickle

  scheduler, executor = make_scheduler()
  targets = [8, 12]
  mask = np.array([1, 0])
  source: dict[str, object] = {
    "task_revision": "rev-a",
    "targets": targets,
    "mask": mask,
  }
  request = scheduler.request(
    np.ones(2), source_timestamp=0.0, source_tick=0, context=source
  )
  source["task_revision"] = "changed"
  targets.append(99)
  mask[0] = 0
  assert request.context is not None
  context = cast(Mapping[str, object], request.context)
  assert context["task_revision"] == "rev-a"
  assert context["targets"] == (8, 12)
  assert context["mask"] == (1, 0)
  restored = pickle.loads(pickle.dumps(request.context))
  assert restored == request.context
  scheduler.poll()
  assert executor.submitted == [request]


def test_request_context_defaults_to_none_for_legacy_callers() -> None:
  request = PlannerRequest(np.ones(1), 0.0, 0, 1, 0)
  assert request.context is None


def test_request_context_rejects_non_serializable_or_nonfinite_values() -> None:
  scheduler, _executor = make_scheduler()
  with pytest.raises(SchedulerError, match="plainly serializable"):
    scheduler.request(np.ones(1), source_timestamp=0.0, source_tick=0, context=object())
  with pytest.raises(SchedulerError, match="finite"):
    scheduler.request(
      np.ones(1), source_timestamp=0.0, source_tick=0, context={"value": np.inf}
    )
