"""Focused CPU tests for the bounded diffusion planner executor."""

from __future__ import annotations

import os
import pickle
import time
from typing import Any, cast

import numpy as np
import pytest

from mjlab.tasks.tracking.diffusion.planner_executor import (
  BoundedPlannerExecutor,
  PlannerExecutorBusyError,
  PlannerExecutorError,
  PlannerWorkerError,
  _ActiveRequest,
)
from mjlab.tasks.tracking.diffusion.scheduler import Plan, PlannerRequest


def _request(*, sequence: int = 7, generation: int = 3) -> PlannerRequest:
  return PlannerRequest(
    input_snapshot=np.array([1.5, -2.0], dtype=np.float64),
    source_timestamp=1.25,
    source_tick=11,
    request_sequence_id=sequence,
    policy_generation=generation,
  )


def _round_trip_factory():
  return _round_trip_planner


def _round_trip_planner(request: PlannerRequest) -> Plan:
  actions = np.arange(32 * 32, dtype=np.float64).reshape(32, 32)
  return Plan(
    plan_sequence_id=101,
    source_timestamp=request.source_timestamp,
    valid_from_tick=request.source_tick,
    actions=actions,
    schema_metadata={
      "contract": "generic-test",
      "source_first": float(request.input_snapshot[0]),
      "vector_width": 32,
    },
  )


def _failing_factory():
  raise RuntimeError("factory failed before callback creation")


def _raising_factory():
  return _raising_planner


def _raising_planner(_request: PlannerRequest) -> Plan:
  raise ValueError("callback failed")


class _UnpickleableError(RuntimeError):
  def __reduce__(self) -> Any:
    raise TypeError("this exception is intentionally not pickleable")


def _unpickleable_error_factory():
  return _unpickleable_error_planner


def _unpickleable_error_planner(_request: PlannerRequest) -> Plan:
  raise _UnpickleableError("opaque callback failure")


def _malformed_factory():
  return _malformed_planner


def _malformed_planner(_request: PlannerRequest) -> Plan:
  return object()  # type: ignore[return-value]


def _blocked_factory():
  return _blocked_planner


def _blocked_planner(_request: PlannerRequest) -> Plan:
  time.sleep(60.0)
  raise AssertionError("the bounded shutdown should terminate this callback")


def _exit_factory():
  return _exit_planner


def _exit_planner(_request: PlannerRequest) -> Plan:
  os._exit(17)


class _Unpickleable:
  def __reduce__(self) -> Any:
    raise TypeError("request payload is intentionally not pickleable")


def _wait_for_wire(executor: BoundedPlannerExecutor, timeout: float = 3.0):
  """Wait on the real result connection, then use the nonblocking API."""

  receiver = executor._result_receiver
  assert receiver is not None
  assert receiver.poll(timeout), "the worker did not notify within the test bound"
  results = tuple(executor.poll())
  assert len(results) == 1
  return results[0]


def _wire_bytes(wire: tuple[object, ...]) -> bytes:
  return b"W" + pickle.dumps(wire, protocol=pickle.HIGHEST_PROTOCOL)


def _plan_wire(
  request_sequence_id: object, policy_generation: object
) -> tuple[object, ...]:
  return (
    "plan",
    request_sequence_id,
    policy_generation,
    101,
    1.25,
    11,
    ((1.0,),),
    pickle.dumps({}),
  )


def _error_wire(
  request_sequence_id: object, policy_generation: object
) -> tuple[object, ...]:
  return (
    "error",
    request_sequence_id,
    policy_generation,
    "ValueError",
    "malformed",
    "trace",
  )


def _in_memory_decoder() -> BoundedPlannerExecutor:
  """Build a fresh executor decoder without starting a worker process."""

  decoder = object.__new__(BoundedPlannerExecutor)
  decoder._active = _ActiveRequest(request_sequence_id=1, policy_generation=0)
  decoder._fatal_error = None
  decoder._max_result_bytes = 8 * 1024 * 1024
  decoder._max_metadata_bytes = 2 * 1024 * 1024
  decoder._max_action_elements = 262_144
  return decoder


@pytest.mark.parametrize(
  "value",
  [
    1.0,
    1.5,
    np.float32(1.0),
    np.float64(1.0),
    True,
    np.bool_(True),
    0,
    -1,
  ],
)
def test_wire_request_ids_are_strict_before_plan_or_error_attribution(
  value: object,
) -> None:
  decoder = _in_memory_decoder()
  for wire_builder in (_plan_wire, _error_wire):
    result = decoder._decode_result(_wire_bytes(wire_builder(value, 0)))
    assert result is None
    assert decoder.active_request_sequence_id == 1


@pytest.mark.parametrize(
  "value",
  [
    0.0,
    -0.5,
    np.float32(0.0),
    np.float64(0.0),
    False,
    np.bool_(False),
    -1,
  ],
)
def test_wire_policy_generations_are_strict_before_plan_or_error_attribution(
  value: object,
) -> None:
  decoder = _in_memory_decoder()
  for wire_builder in (_plan_wire, _error_wire):
    result = decoder._decode_result(_wire_bytes(wire_builder(1, value)))
    assert result is None
    assert decoder.active_request_sequence_id == 1


def test_in_memory_decoder_control_accepts_valid_plan_and_error_ids() -> None:
  plan_decoder = _in_memory_decoder()
  plan_result = plan_decoder._decode_result(_wire_bytes(_plan_wire(1, 0)))
  assert plan_result is not None
  assert plan_result.plan is not None
  assert plan_decoder.active_request_sequence_id is None

  error_decoder = _in_memory_decoder()
  error_result = error_decoder._decode_result(_wire_bytes(_error_wire(1, 0)))
  assert error_result is not None
  assert error_result.error is not None
  assert error_decoder.active_request_sequence_id is None


def test_real_blocked_worker_rejects_malformed_wire_and_cleans_exactly() -> None:
  executor = BoundedPlannerExecutor(_blocked_factory, shutdown_timeout_seconds=0.05)
  pid = executor.worker_pid
  assert pid is not None
  try:
    executor.submit(_request(sequence=1, generation=0))
    result = executor._decode_result(_wire_bytes(_plan_wire(True, 0)))
    assert result is None
    assert executor.active_request_sequence_id == 1
    assert executor.last_error is not None
    receiver = executor._result_receiver
    assert receiver is not None
    assert not receiver.poll(0.1)
  finally:
    executor.shutdown()
  assert executor.closed
  assert not executor.worker_alive
  with pytest.raises(ProcessLookupError):
    os.kill(pid, 0)


@pytest.mark.parametrize(
  "value",
  [
    True,
    False,
    np.bool_(True),
    np.bool_(False),
    np.array(0.1),
    np.array([0.1]),
    "0.1",
    None,
    1 + 0j,
    np.nan,
    np.inf,
  ],
)
def test_timeout_constructor_rejects_non_boolean_free_finite_real_scalars(
  value: object,
) -> None:
  for name in ("startup_timeout_seconds", "shutdown_timeout_seconds"):
    with pytest.raises(ValueError):
      if name == "startup_timeout_seconds":
        BoundedPlannerExecutor(
          _round_trip_factory, startup_timeout_seconds=cast(float, value)
        )
      else:
        BoundedPlannerExecutor(
          _round_trip_factory, shutdown_timeout_seconds=cast(float, value)
        )


def test_real_worker_round_trip_owns_request_and_transports_32_vectors() -> None:
  executor = BoundedPlannerExecutor(_round_trip_factory)
  try:
    request = _request()
    executor.submit(request)
    # The executor serialized the request before this caller-side mutation.
    request.input_snapshot.setflags(write=True)
    request.input_snapshot[0] = 99.0

    result = _wait_for_wire(executor)
    assert result.request_sequence_id == 7
    assert result.policy_generation == 3
    assert result.error is None
    assert result.plan is not None
    np.testing.assert_array_equal(
      result.plan.actions, np.arange(32 * 32).reshape(32, 32)
    )
    assert result.plan.actions.shape == (32, 32)
    assert result.plan.schema_metadata["source_first"] == 1.5
    assert result.plan.schema_metadata["vector_width"] == 32
    assert result.plan.actions.flags.writeable is False
    assert result.plan.schema_metadata.__class__.__name__ == "mappingproxy"
    with pytest.raises(TypeError):
      pickle.dumps(result.plan)
  finally:
    executor.shutdown()


def test_active_slot_and_completed_transport_remain_bounded() -> None:
  executor = BoundedPlannerExecutor(_round_trip_factory)
  try:
    first = _request(sequence=1)
    second = _request(sequence=2)
    executor.submit(first)
    with pytest.raises(PlannerExecutorBusyError, match="already active"):
      executor.submit(second)

    receiver = executor._result_receiver
    assert receiver is not None
    assert receiver.poll(3.0)
    # The result is notified but not yet drained.  It still occupies the one
    # active slot, so no implicit completed-result FIFO is available.
    assert executor.active_request_sequence_id == 1
    with pytest.raises(PlannerExecutorBusyError, match="already active"):
      executor.submit(second)
    result = tuple(executor.poll())
    assert len(result) == 1
    assert executor.active_request_sequence_id is None
    executor.submit(second)
    assert _wait_for_wire(executor).request_sequence_id == 2
  finally:
    executor.shutdown()


def test_poll_is_immediate_while_callback_is_incomplete() -> None:
  executor = BoundedPlannerExecutor(_blocked_factory, shutdown_timeout_seconds=0.05)
  try:
    executor.submit(_request())
    started = time.monotonic()
    assert tuple(executor.poll()) == ()
    assert time.monotonic() - started < 0.2
  finally:
    executor.shutdown()


def test_factory_initialization_failure_is_reported_and_cleaned() -> None:
  with pytest.raises(PlannerExecutorError, match="initialization failed"):
    BoundedPlannerExecutor(_failing_factory)


def test_callback_failure_retains_request_identity_without_cancellation_claim() -> None:
  executor = BoundedPlannerExecutor(_raising_factory)
  try:
    executor.submit(_request(sequence=13, generation=8))
    result = _wait_for_wire(executor)
    assert result.request_sequence_id == 13
    assert result.policy_generation == 8
    assert isinstance(result.error, PlannerWorkerError)
    assert "callback failed" in str(result.error)
    assert executor.worker_alive
  finally:
    executor.shutdown()


def test_unpickleable_callback_exception_uses_safe_error_transport() -> None:
  executor = BoundedPlannerExecutor(_unpickleable_error_factory)
  try:
    executor.submit(_request())
    result = _wait_for_wire(executor)
    assert result.error is not None
    assert "_UnpickleableError" in str(result.error)
    assert "opaque callback failure" in str(result.error)
  finally:
    executor.shutdown()


def test_malformed_callback_output_becomes_explicit_failure() -> None:
  executor = BoundedPlannerExecutor(_malformed_factory)
  try:
    executor.submit(_request(sequence=21, generation=4))
    result = _wait_for_wire(executor)
    assert result.request_sequence_id == 21
    assert result.policy_generation == 4
    assert isinstance(result.error, PlannerWorkerError)
    assert "expected Plan" in str(result.error)
  finally:
    executor.shutdown()


def test_request_serialization_failure_does_not_retain_unpickleable_input() -> None:
  executor = BoundedPlannerExecutor(_round_trip_factory)
  try:
    request = _request()
    object.__setattr__(request, "input_snapshot", _Unpickleable())
    with pytest.raises(PlannerExecutorError, match="request serialization failed"):
      executor.submit(request)
    assert executor.active_request_sequence_id is None
  finally:
    executor.shutdown()


def test_worker_exit_is_reported_with_the_original_identity() -> None:
  executor = BoundedPlannerExecutor(_exit_factory)
  try:
    executor.submit(_request(sequence=31, generation=9))
    process = executor._process
    assert process is not None
    process.join(timeout=3.0)
    result = tuple(executor.poll())
    assert len(result) == 1
    assert result[0].request_sequence_id == 31
    assert result[0].policy_generation == 9
    assert result[0].error is not None
    assert "exited before completion" in str(result[0].error)
    assert not executor.worker_alive
  finally:
    executor.shutdown()


def test_blocked_callback_has_bounded_exact_process_cleanup() -> None:
  executor = BoundedPlannerExecutor(_blocked_factory, shutdown_timeout_seconds=0.05)
  executor.submit(_request())
  receiver = executor._result_receiver
  assert receiver is not None
  assert not receiver.poll(0.1)
  pid = executor.worker_pid
  assert pid is not None
  started = time.monotonic()
  executor.shutdown()
  elapsed = time.monotonic() - started
  assert elapsed < 1.0
  assert executor.closed
  assert not executor.worker_alive
  # The executor keeps only the Process handle for inspection.  Its IPC
  # endpoints are closed and the exact owned process has exited.
  assert executor._result_receiver is None
  assert executor._command_sender is None
  with pytest.raises(ProcessLookupError):
    os.kill(pid, 0)


def test_shutdown_is_idempotent_and_rejects_later_submission() -> None:
  executor = BoundedPlannerExecutor(_round_trip_factory)
  executor.shutdown()
  executor.shutdown()
  with pytest.raises(PlannerExecutorError, match="shut down"):
    executor.submit(_request())
  assert tuple(executor.poll()) == ()
