"""Bounded spawn-process execution for simulator-independent diffusion planners.

The executor owns one persistent process.  It serializes a complete
:class:`PlannerRequest` before it sends the request.  The worker constructs the
planner by calling the supplied factory and returns an explicit wire record;
it never sends a :class:`Plan` directly because ``Plan.schema_metadata`` is a
``MappingProxyType``.

Factories must be pickleable.  In practice, use a top-level function or a
pickleable callable object.  The factory is called inside the spawned worker,
so model and callback state is created there.  This slice does not initialize
CUDA.  A planner callback must accept one ``PlannerRequest`` and return one
base ``Plan``.  The request and its optional context must contain only
pickleable, finite, owned data.

Only one request can be active.  A completed result stays in the operating
system pipe until :meth:`BoundedPlannerExecutor.poll` drains it.  A second
submission is rejected instead of entering a hidden queue.  Polling never
waits for inference.  Shutdown first stops submissions, asks the exact owned
worker to exit, and then uses bounded join/terminate/kill escalation.
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.connection import wait as wait_connections
from numbers import Real
from typing import Any, cast

import numpy as np

from mjlab.tasks.tracking.diffusion.scheduler import (
  Plan,
  PlannerExecutor,
  PlannerRequest,
  PlannerResult,
)

PlannerCallback = Callable[[PlannerRequest], Plan]
PlannerFactory = Callable[[], PlannerCallback]

DEFAULT_MAX_FACTORY_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_REQUEST_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_RESULT_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_METADATA_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_ACTION_ELEMENTS = 262_144
DEFAULT_STARTUP_TIMEOUT_SECONDS = 10.0
DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 1.0
_MAX_ERROR_TEXT = 4_096
_MAX_TRACEBACK_TEXT = 16_384


class PlannerExecutorError(RuntimeError):
  """The executor cannot accept or transport a planner request."""


class PlannerExecutorBusyError(PlannerExecutorError):
  """A request is already active or its completion is not yet drained."""


class PlannerWorkerError(PlannerExecutorError):
  """A worker callback or worker transport failed for one request."""


class PlannerWorkerExitedError(PlannerWorkerError):
  """The owned worker exited before it returned the active request."""


@dataclass(frozen=True, slots=True)
class _ActiveRequest:
  request_sequence_id: int
  policy_generation: int


def _validate_positive_limit(value: int, name: str) -> int:
  if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
    raise ValueError(f"{name} must be a positive integer")
  if int(value) <= 0:
    raise ValueError(f"{name} must be a positive integer")
  return int(value)


def _validate_wire_integer(value: object, name: str, *, minimum: int) -> None:
  if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
    raise PlannerExecutorError(f"{name} must be a non-boolean integer")
  if value < minimum:
    raise PlannerExecutorError(f"{name} must be >= {minimum}")


def _validate_timeout(value: float, name: str, *, allow_zero: bool = False) -> float:
  if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
    raise ValueError(f"{name} must be a finite real scalar")
  try:
    scalar = float(value)
  except (TypeError, ValueError, OverflowError) as exc:
    raise ValueError(f"{name} must be a finite real scalar") from exc
  if not np.isfinite(scalar) or (scalar < 0 if allow_zero else scalar <= 0):
    qualifier = "nonnegative" if allow_zero else "positive"
    raise ValueError(f"{name} must be finite and {qualifier}")
  return scalar


def _safe_text(value: object, *, limit: int = _MAX_ERROR_TEXT) -> str:
  try:
    text = str(value)
  except BaseException:
    text = f"<{type(value).__name__} could not be rendered>"
  return text[:limit]


def _safe_traceback() -> str:
  try:
    return traceback.format_exc(limit=32)[:_MAX_TRACEBACK_TEXT]
  except BaseException:
    return "<traceback unavailable>"


def _exception_wire(exc: BaseException) -> tuple[str, str, str]:
  """Return bounded strings without transporting the exception object."""

  return type(exc).__name__[:256], _safe_text(exc), _safe_traceback()


def _pickle(value: object, *, name: str, limit: int) -> bytes:
  try:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
  except Exception as exc:
    raise PlannerExecutorError(
      f"{name} serialization failed: {type(exc).__name__}: {_safe_text(exc)}"
    ) from exc
  if len(payload) > limit:
    raise PlannerExecutorError(f"{name} serialization exceeded the {limit}-byte bound")
  return payload


def _wire_message(value: object, *, limit: int) -> bytes:
  payload = _pickle(value, name="planner result", limit=limit - 1)
  return b"W" + payload


def _send_wire(connection: Connection, value: object, *, limit: int) -> bool:
  try:
    connection.send_bytes(_wire_message(value, limit=limit))
  except BaseException:
    return False
  return True


def _plan_to_wire(
  request: PlannerRequest,
  plan: object,
  *,
  max_action_elements: int,
  max_metadata_bytes: int,
  max_result_bytes: int,
) -> tuple[object, ...]:
  """Convert a base plan into bounded, plain transport fields."""

  if not isinstance(plan, Plan):
    raise TypeError(f"planner returned {type(plan).__name__}, expected {Plan.__name__}")
  try:
    actions = np.array(plan.actions, dtype=np.float64, copy=True)
  except (TypeError, ValueError, OverflowError) as exc:
    raise ValueError(f"planner actions are not numeric: {_safe_text(exc)}") from exc
  if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] == 0:
    raise ValueError("planner actions must have shape [horizon, action_dim]")
  if actions.size > max_action_elements:
    raise ValueError(f"planner actions exceed the {max_action_elements}-element bound")
  if not bool(np.isfinite(actions).all()):
    raise ValueError("planner actions must be finite")
  metadata = dict(plan.schema_metadata)
  if any(not isinstance(key, str) for key in metadata):
    raise ValueError("planner metadata keys must be strings")
  metadata_payload = _pickle(
    metadata, name="planner metadata", limit=max_metadata_bytes
  )
  action_values = tuple(
    tuple(float(value) for value in row) for row in actions.tolist()
  )
  wire: tuple[object, ...] = (
    "plan",
    request.request_sequence_id,
    request.policy_generation,
    plan.plan_sequence_id,
    plan.source_timestamp,
    plan.valid_from_tick,
    action_values,
    metadata_payload,
  )
  # This is an explicit upper bound on the complete result, not just on its
  # matrix.  The parent repeats the bound before unpickling a result.
  _pickle(wire, name="planner result", limit=max_result_bytes - 1)
  return wire


def _worker_main(
  command_receiver: Connection,
  result_sender: Connection,
  factory_payload: bytes,
  max_request_bytes: int,
  max_result_bytes: int,
  max_action_elements: int,
  max_metadata_bytes: int,
) -> None:
  """Initialize and run the one owned planner worker."""

  try:
    try:
      factory = pickle.loads(factory_payload)
      if not callable(factory):
        raise TypeError("planner factory is not callable")
      planner = factory()
      if not callable(planner):
        raise TypeError("planner factory did not return a callable")
    except BaseException as exc:
      _send_wire(
        result_sender,
        ("initialization_error", *_exception_wire(exc)),
        limit=max_result_bytes,
      )
      return

    if not _send_wire(result_sender, ("ready",), limit=max_result_bytes):
      return

    while True:
      try:
        command = command_receiver.recv_bytes()
      except (EOFError, OSError):
        return
      if command == b"S":
        return
      if not command.startswith(b"Q"):
        if not _send_wire(
          result_sender,
          ("protocol_error", "worker received an unknown command"),
          limit=max_result_bytes,
        ):
          return
        continue
      request_payload = command[1:]
      if len(request_payload) > max_request_bytes:
        if not _send_wire(
          result_sender,
          ("protocol_error", "request payload exceeded its transport bound"),
          limit=max_result_bytes,
        ):
          return
        continue
      try:
        request = pickle.loads(request_payload)
        if not isinstance(request, PlannerRequest):
          raise TypeError("request payload is not a PlannerRequest")
        request_sequence_id = request.request_sequence_id
        policy_generation = request.policy_generation
      except BaseException as exc:
        if not _send_wire(
          result_sender,
          ("protocol_error", *_exception_wire(exc)),
          limit=max_result_bytes,
        ):
          return
        continue

      try:
        plan = planner(request)
      except BaseException as exc:
        if not _send_wire(
          result_sender,
          (
            "error",
            request_sequence_id,
            policy_generation,
            *_exception_wire(exc),
          ),
          limit=max_result_bytes,
        ):
          return
        continue

      try:
        wire = _plan_to_wire(
          request,
          plan,
          max_action_elements=max_action_elements,
          max_metadata_bytes=max_metadata_bytes,
          max_result_bytes=max_result_bytes,
        )
      except BaseException as exc:
        if not _send_wire(
          result_sender,
          (
            "error",
            request_sequence_id,
            policy_generation,
            *_exception_wire(exc),
          ),
          limit=max_result_bytes,
        ):
          return
        continue
      if not _send_wire(result_sender, wire, limit=max_result_bytes):
        return
  finally:
    try:
      command_receiver.close()
    except OSError:
      pass
    try:
      result_sender.close()
    except OSError:
      pass


class BoundedPlannerExecutor(PlannerExecutor):
  """One bounded planner callback running in an explicitly owned process.

  The executor starts the spawned worker during construction and waits only for
  the bounded initialization handshake.  A factory failure raises
  :class:`PlannerExecutorError` after the child is closed.  Callback failures
  become ``PlannerResult.error`` values that retain the original request
  sequence and policy generation; they do not imply cancellation.

  ``poll`` is nonblocking.  It returns zero or one result.  ``shutdown`` is
  idempotent and confirms worker exit before it closes both one-way pipes.
  """

  def __init__(
    self,
    factory: PlannerFactory,
    *,
    context: Any | None = None,
    startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
    shutdown_timeout_seconds: float = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
    max_factory_bytes: int = DEFAULT_MAX_FACTORY_BYTES,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    max_result_bytes: int = DEFAULT_MAX_RESULT_BYTES,
    max_metadata_bytes: int = DEFAULT_MAX_METADATA_BYTES,
    max_action_elements: int = DEFAULT_MAX_ACTION_ELEMENTS,
  ) -> None:
    if not callable(factory):
      raise TypeError("factory must be callable")
    self._startup_timeout_seconds = _validate_timeout(
      startup_timeout_seconds, "startup_timeout_seconds"
    )
    self._shutdown_timeout_seconds = _validate_timeout(
      shutdown_timeout_seconds,
      "shutdown_timeout_seconds",
      allow_zero=True,
    )
    self._max_factory_bytes = _validate_positive_limit(
      max_factory_bytes, "max_factory_bytes"
    )
    self._max_request_bytes = _validate_positive_limit(
      max_request_bytes, "max_request_bytes"
    )
    self._max_result_bytes = _validate_positive_limit(
      max_result_bytes, "max_result_bytes"
    )
    self._max_metadata_bytes = _validate_positive_limit(
      max_metadata_bytes, "max_metadata_bytes"
    )
    self._max_action_elements = _validate_positive_limit(
      max_action_elements, "max_action_elements"
    )
    if self._max_result_bytes <= 1 or self._max_request_bytes <= 0:
      raise ValueError("transport bounds are too small for a wire message")

    factory_payload = _pickle(
      factory, name="planner factory", limit=self._max_factory_bytes
    )
    mp_context = mp.get_context("spawn") if context is None else context
    self._closed = False
    self._ready = False
    self._fatal_error: str | None = None
    self._active: _ActiveRequest | None = None
    self._process: Any | None = None
    self._command_sender: Connection | None = None
    self._command_receiver: Connection | None = None
    self._result_sender: Connection | None = None
    self._result_receiver: Connection | None = None

    command_receiver, command_sender = mp_context.Pipe(duplex=False)
    result_receiver, result_sender = mp_context.Pipe(duplex=False)
    self._command_sender = cast(Connection, command_sender)
    self._command_receiver = cast(Connection, command_receiver)
    self._result_receiver = cast(Connection, result_receiver)
    self._result_sender = cast(Connection, result_sender)
    process = mp_context.Process(
      target=_worker_main,
      args=(
        self._command_receiver,
        self._result_sender,
        factory_payload,
        self._max_request_bytes,
        self._max_result_bytes,
        self._max_action_elements,
        self._max_metadata_bytes,
      ),
      name="mjlab-diffusion-planner",
    )
    process.daemon = False
    self._process = process
    try:
      process.start()
    except BaseException as exc:
      self._close_channels()
      self._process = None
      raise PlannerExecutorError(
        f"planner worker could not start: {type(exc).__name__}: {_safe_text(exc)}"
      ) from exc
    self._close_child_channels_in_parent()
    self._await_ready()

  @property
  def active_request_sequence_id(self) -> int | None:
    """The one active request sequence, or ``None`` when the slot is free."""

    return None if self._active is None else self._active.request_sequence_id

  @property
  def worker_pid(self) -> int | None:
    """The exact owned worker PID, retained for cleanup inspection."""

    return None if self._process is None else self._process.pid

  @property
  def worker_alive(self) -> bool:
    """Whether the exact owned process is currently alive."""

    return bool(self._process is not None and self._process.is_alive())

  @property
  def closed(self) -> bool:
    """Whether submission has been closed and IPC has been released."""

    return self._closed

  @property
  def last_error(self) -> str | None:
    """The first worker or transport failure observed by the owner."""

    return self._fatal_error

  def submit(self, request: PlannerRequest) -> None:
    """Serialize and submit one request without creating hidden backlog."""

    if self._closed:
      raise PlannerExecutorError("planner executor is shut down")
    if self._fatal_error is not None:
      raise PlannerExecutorError(f"planner executor is unusable: {self._fatal_error}")
    if not isinstance(request, PlannerRequest):
      raise TypeError("request must be a PlannerRequest")
    if self._active is not None:
      raise PlannerExecutorBusyError(
        "one planner request is already active; poll its completion first"
      )
    if not self._ready:
      raise PlannerExecutorError("planner worker is not initialized")
    if self._process is None or not self._process.is_alive():
      self._record_fatal("planner worker exited before submit")
      raise PlannerExecutorError(f"planner executor is unusable: {self._fatal_error}")

    # This dump is the ownership boundary.  No caller-owned request object is
    # retained by the process or by the IPC layer after this method returns.
    request_payload = _pickle(
      request, name="planner request", limit=self._max_request_bytes
    )
    self._active = _ActiveRequest(
      request_sequence_id=request.request_sequence_id,
      policy_generation=request.policy_generation,
    )
    try:
      assert self._command_sender is not None
      self._command_sender.send_bytes(b"Q" + request_payload)
    except BaseException as exc:
      self._record_fatal(
        f"planner request transport failed: {type(exc).__name__}: {_safe_text(exc)}"
      )
      # Keep the active tombstone until poll emits an attributed failure.  A
      # send exception does not prove that the worker did not start the work.
      raise PlannerExecutorError(
        f"planner request transport failed: {type(exc).__name__}: {_safe_text(exc)}"
      ) from exc

  def poll(self) -> Iterable[PlannerResult]:
    """Return completed results immediately without waiting for inference."""

    if self._closed:
      return ()
    results: list[PlannerResult] = []
    receiver = self._result_receiver
    if receiver is not None:
      # There can be at most one normal result because there can be at most one
      # active request.  Read at most two frames to detect a protocol breach
      # without introducing an unbounded drain loop.
      for _ in range(2):
        if not receiver.poll(0.0):
          break
        try:
          raw = receiver.recv_bytes()
        except (EOFError, OSError) as exc:
          self._record_fatal(
            f"planner result transport failed: {type(exc).__name__}: {_safe_text(exc)}"
          )
          break
        result = self._decode_result(raw)
        if result is not None:
          results.append(result)
        if self._active is None:
          break
      if receiver.poll(0.0):
        self._record_fatal("planner worker produced more than one completed result")

    process = self._process
    if process is not None and not process.is_alive() and self._active is not None:
      results.append(self._active_failure_from_exit(process.exitcode))
    elif process is not None and not process.is_alive() and self._fatal_error is None:
      self._record_fatal(f"planner worker exited with code {process.exitcode}")
    return tuple(results)

  def shutdown(self) -> None:
    """Close submissions and stop the exact worker within a bounded deadline."""

    if self._closed:
      return
    self._closed = True
    sender = self._command_sender
    process = self._process
    if sender is not None and process is not None and process.is_alive():
      try:
        sender.send_bytes(b"S")
      except (EOFError, OSError, ValueError):
        pass

    if process is not None:
      deadline = time.monotonic() + self._shutdown_timeout_seconds
      while process.is_alive():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
          break
        # Drain one completed result so a large but bounded response cannot
        # prevent a worker that already returned from reading its shutdown.
        self._drain_one_raw_result()
        process.join(timeout=min(remaining, 0.02))
      if process.is_alive():
        process.terminate()
        process.join(timeout=self._shutdown_timeout_seconds)
      if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(timeout=self._shutdown_timeout_seconds)
      if process.is_alive():
        self._record_fatal("planner worker did not exit after kill")

    self._active = None
    self._close_channels()

  def _await_ready(self) -> None:
    receiver = self._result_receiver
    process = self._process
    if receiver is None or process is None:
      raise PlannerExecutorError("planner worker IPC was not initialized")
    ready_sources = wait_connections(
      (receiver, process.sentinel), timeout=self._startup_timeout_seconds
    )
    if receiver not in ready_sources:
      if process.sentinel in ready_sources:
        detail = (
          f"planner worker exited during initialization (exit code {process.exitcode})"
        )
      else:
        detail = (
          f"planner worker initialization timed out after "
          f"{self._startup_timeout_seconds:g}s"
        )
      self._record_fatal(detail)
      self.shutdown()
      raise PlannerExecutorError(detail)
    try:
      raw = receiver.recv_bytes()
    except (EOFError, OSError) as exc:
      detail = f"planner worker initialization transport failed: {type(exc).__name__}"
      self._record_fatal(detail)
      self.shutdown()
      raise PlannerExecutorError(detail) from exc
    try:
      wire = self._decode_wire(raw)
      kind = wire[0]
    except BaseException as exc:
      detail = f"malformed planner initialization message: {_safe_text(exc)}"
      self._record_fatal(detail)
      self.shutdown()
      raise PlannerExecutorError(detail) from exc
    if kind == "ready" and len(wire) == 1:
      self._ready = True
      return
    if kind == "initialization_error" and len(wire) == 4:
      detail = f"planner factory initialization failed: {wire[1]}: {wire[2]}"
      self._record_fatal(detail)
      self.shutdown()
      raise PlannerExecutorError(detail)
    detail = "planner worker returned an invalid initialization status"
    self._record_fatal(detail)
    self.shutdown()
    raise PlannerExecutorError(detail)

  def _decode_wire(self, raw: bytes) -> tuple[Any, ...]:
    if len(raw) > self._max_result_bytes:
      raise PlannerExecutorError("planner result exceeded its transport bound")
    if not raw.startswith(b"W"):
      raise PlannerExecutorError("planner result is missing its wire marker")
    value = pickle.loads(raw[1:])
    if not isinstance(value, tuple) or not value or not isinstance(value[0], str):
      raise PlannerExecutorError("planner result wire record is not a tagged tuple")
    return cast(tuple[Any, ...], value)

  def _decode_result(self, raw: bytes) -> PlannerResult | None:
    if self._active is None:
      self._record_fatal("planner worker returned a result with no active request")
      return None
    try:
      wire = self._decode_wire(raw)
    except BaseException as exc:
      # There is no trusted identity to attribute this frame to.  Keep the
      # active tombstone until the worker exits or the owner shuts it down.
      self._record_fatal(
        f"malformed planner result: {type(exc).__name__}: {_safe_text(exc)}"
      )
      return None

    kind = wire[0]
    if kind not in ("plan", "error") or len(wire) < 3:
      self._record_fatal("planner result has no attributable request identity")
      return None
    try:
      # Validate the wire IDs before any equality comparison.  A malformed
      # frame must not become a trusted completion for the active request.
      self._require_active_identity(wire[1], wire[2])
    except BaseException as exc:
      self._record_fatal(
        f"unattributed planner result: {type(exc).__name__}: {_safe_text(exc)}"
      )
      return None

    try:
      if kind == "plan":
        result = self._decode_plan_result(wire)
      else:
        result = self._decode_error_result(wire)
    except BaseException as exc:
      # Matching, validated IDs are completion evidence.  Malformed plan/error
      # fields therefore become an attributed failure, not an accepted plan.
      result = self._failure_result(
        f"malformed planner result: {type(exc).__name__}: {_safe_text(exc)}"
      )
    self._active = None
    return result

  def _decode_plan_result(self, wire: tuple[Any, ...]) -> PlannerResult:
    if len(wire) != 8:
      raise PlannerExecutorError("plan wire record has the wrong field count")
    (
      _kind,
      request_sequence_id,
      policy_generation,
      plan_sequence_id,
      source_timestamp,
      valid_from_tick,
      action_values,
      metadata_payload,
    ) = wire
    active = self._require_active_identity(request_sequence_id, policy_generation)
    if not isinstance(metadata_payload, bytes):
      raise PlannerExecutorError("plan metadata payload is not bytes")
    if len(metadata_payload) > self._max_metadata_bytes:
      raise PlannerExecutorError("plan metadata exceeded its transport bound")
    metadata = pickle.loads(metadata_payload)
    if not isinstance(metadata, dict):
      raise PlannerExecutorError("plan metadata is not a dictionary")
    if not isinstance(action_values, tuple):
      raise PlannerExecutorError("plan actions are not a tuple matrix")
    try:
      actions = np.array(action_values, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
      raise PlannerExecutorError(
        f"plan actions are malformed: {_safe_text(exc)}"
      ) from exc
    if actions.ndim != 2 or actions.size == 0:
      raise PlannerExecutorError("plan actions must have shape [horizon, action_dim]")
    if actions.size > self._max_action_elements:
      raise PlannerExecutorError("plan actions exceeded their element bound")
    if not bool(np.isfinite(actions).all()):
      raise PlannerExecutorError("plan actions are non-finite")
    try:
      plan = Plan(
        plan_sequence_id=plan_sequence_id,
        source_timestamp=source_timestamp,
        valid_from_tick=valid_from_tick,
        actions=actions,
        schema_metadata=metadata,
      )
    except BaseException as exc:
      raise PlannerExecutorError(
        f"plan fields failed validation: {type(exc).__name__}: {_safe_text(exc)}"
      ) from exc
    return PlannerResult(
      request_sequence_id=active.request_sequence_id,
      policy_generation=active.policy_generation,
      plan=plan,
    )

  def _decode_error_result(self, wire: tuple[Any, ...]) -> PlannerResult:
    if len(wire) != 6:
      raise PlannerExecutorError("error wire record has the wrong field count")
    _kind, request_sequence_id, policy_generation, error_type, detail, trace = wire
    active = self._require_active_identity(request_sequence_id, policy_generation)
    if not isinstance(error_type, str) or not isinstance(detail, str):
      raise PlannerExecutorError("worker error fields are not strings")
    trace_text = trace if isinstance(trace, str) else "<traceback unavailable>"
    message = f"{error_type}: {detail}"
    if trace_text:
      message = f"{message}\n{trace_text[:_MAX_TRACEBACK_TEXT]}"
    return PlannerResult(
      request_sequence_id=active.request_sequence_id,
      policy_generation=active.policy_generation,
      error=PlannerWorkerError(message),
    )

  def _require_active_identity(
    self, request_sequence_id: object, policy_generation: object
  ) -> _ActiveRequest:
    _validate_wire_integer(request_sequence_id, "wire request_sequence_id", minimum=1)
    _validate_wire_integer(policy_generation, "wire policy_generation", minimum=0)
    active = self._active
    if active is None:
      raise PlannerExecutorError("planner result has no active request")
    if (
      request_sequence_id != active.request_sequence_id
      or policy_generation != active.policy_generation
    ):
      raise PlannerExecutorError(
        "planner result identity does not match the active request"
      )
    return active

  def _failure_result(self, detail: str) -> PlannerResult:
    active = self._active
    if active is None:
      raise PlannerExecutorError(detail)
    return PlannerResult(
      request_sequence_id=active.request_sequence_id,
      policy_generation=active.policy_generation,
      error=PlannerWorkerError(detail),
    )

  def _active_failure_from_exit(self, exitcode: int | None) -> PlannerResult:
    active = self._active
    assert active is not None
    detail = f"planner worker exited before completion (exit code {exitcode})"
    self._record_fatal(detail)
    self._active = None
    return PlannerResult(
      request_sequence_id=active.request_sequence_id,
      policy_generation=active.policy_generation,
      error=PlannerWorkerExitedError(detail),
    )

  def _drain_one_raw_result(self) -> None:
    receiver = self._result_receiver
    if receiver is None or not receiver.poll(0.0):
      return
    try:
      receiver.recv_bytes()
    except (EOFError, OSError):
      return

  def _record_fatal(self, detail: str) -> None:
    if self._fatal_error is None:
      self._fatal_error = detail[:_MAX_ERROR_TEXT]

  def _close_child_channels_in_parent(self) -> None:
    for connection in (self._command_receiver, self._result_sender):
      if connection is not None:
        try:
          connection.close()
        except OSError:
          pass
    self._command_receiver = None
    self._result_sender = None

  def _close_channels(self) -> None:
    for attribute in ("_command_sender", "_result_receiver"):
      connection = getattr(self, attribute)
      if connection is not None:
        try:
          connection.close()
        except OSError:
          pass
        setattr(self, attribute, None)
    self._close_child_channels_in_parent()


# A descriptive alias keeps the process boundary visible to callers that use
# the concrete name in integration code.
PlannerProcessExecutor = BoundedPlannerExecutor


__all__ = [
  "BoundedPlannerExecutor",
  "PlannerCallback",
  "PlannerExecutorBusyError",
  "PlannerExecutorError",
  "PlannerFactory",
  "PlannerProcessExecutor",
  "PlannerWorkerError",
  "PlannerWorkerExitedError",
]
