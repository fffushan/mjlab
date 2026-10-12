"""Opt-in streaming publication, progress and limit controls for D1 collection.

``DiffusionCollector.collect_batched`` keeps its original behaviour when no
controls are supplied: every finished trial is retained as full rows and the
store receives one append after the schedule.  These controls let a caller
coalesce completed-trial rows into bounded synchronous appends, keep only
compact per-trial accounting, receive real progress records, and stop a run
before more physics work when a wall-clock, inactivity or stored-output budget
is spent.

Row semantics do not change.  Published rows keep the same identities,
timestamps, accepted-segment flags and seeded noise as the full-result mode.
Limits and faults are reported truthfully: a stop records a reason and leaves
the rows that were already validated in the store, and a storage or callback
fault propagates instead of being reported as a completed collection.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from .collector import TrialSpec
  from .qualification import QualificationResult
  from .storage import TrajectoryRow

# Stop reasons share the vocabulary of the scalar collection report.
STOP_WALL_CLOCK = "wall_clock_limit"
STOP_INACTIVITY = "inactivity_limit"
STOP_OUTPUT_SIZE = "output_size_limit"
STOP_TRANSITION_BUDGET = "control_transition_budget"

# Rows buffered before one publication append when the caller names no bound.
DEFAULT_PUBLICATION_CHUNK = 4096

_TENSOR_DISPATCH_METHODS = (
  ("env", "observe_vae_tensor_batch"),
  ("vae", "batch_action_tensor"),
  ("env", "step_tensor_batch"),
)


class CollectionControlError(ValueError):
  """A streaming control, resource limit or progress report is invalid."""


@dataclass(frozen=True, slots=True)
class CollectionLimits:
  """Thresholds that stop a collection before more scheduled work runs."""

  wall_seconds: float | None = None
  inactivity_seconds: float | None = None
  max_output_bytes: int | None = None

  def __post_init__(self) -> None:
    for name in ("wall_seconds", "inactivity_seconds"):
      value = getattr(self, name)
      if value is None:
        continue
      number = float(value)
      if not math.isfinite(number) or number <= 0.0:
        raise CollectionControlError(f"{name} must be positive and finite")
    if self.max_output_bytes is not None and self.max_output_bytes <= 0:
      raise CollectionControlError("max_output_bytes must be positive")


def limits_from_request(request: Any) -> CollectionLimits:
  """Translate CLI collection budgets (minutes, mebibytes) into limits.

  ``CollectionRequest`` reports wall-clock time in minutes and stored output in
  mebibytes; the collector compares seconds and bytes.
  """
  return CollectionLimits(
    wall_seconds=float(request.max_wall_minutes) * 60.0,
    inactivity_seconds=float(request.monitor_inactivity_seconds),
    max_output_bytes=int(request.max_output_mib) * 1024 * 1024,
  )


@dataclass(frozen=True, slots=True)
class CollectionProgress:
  """Published-work counters observed at one point of a collection run.

  ``stored_rows`` counts rows that a store append already accepted, and
  ``stored_bytes`` is the measured stored output (``None`` when no measurement
  was supplied).  ``store_files`` names the shards written while this trial's
  rows were buffered, or every shard of the run in the ``final`` record.  Most
  events name no shard because publication coalesces rows across trials.
  """

  event: str
  final: bool
  trials_completed: int
  transitions: int
  stored_rows: int
  accepted_rows: int
  stored_bytes: int | None = None
  store_files: tuple[str, ...] = ()
  stop_reason: str | None = None
  episode_id: str | None = None

  def __post_init__(self) -> None:
    if self.event not in {"trial", "final"}:
      raise CollectionControlError(f"unknown progress event {self.event!r}")
    if self.final != (self.event == "final"):
      raise CollectionControlError("only the final progress event is final")
    for name in ("trials_completed", "transitions", "stored_rows", "accepted_rows"):
      if getattr(self, name) < 0:
        raise CollectionControlError(f"{name} must be non-negative")


ProgressCallback = Callable[[CollectionProgress], None]


@dataclass(frozen=True, slots=True)
class CollectionSummary:
  """Per-trial accounting that survives after its rows are released.

  Compact mode returns these instead of :class:`CollectionResult`.  The counts
  are explicit because the full mode returns the rows themselves.
  """

  spec: TrialSpec
  qualification: QualificationResult
  row_count: int
  accepted_row_count: int
  verification_row_count: int
  store_files: tuple[str, ...] = ()
  stopped_reason: str | None = None

  def __post_init__(self) -> None:
    for name in ("row_count", "accepted_row_count", "verification_row_count"):
      if getattr(self, name) < 0:
        raise CollectionControlError(f"{name} must be non-negative")
    if self.accepted_row_count + self.verification_row_count != self.row_count:
      raise CollectionControlError("summary row counts must account for every row")

  @property
  def failed_attempt(self) -> bool:
    return not self.qualification.qualified


@dataclass(frozen=True, slots=True)
class CollectionControls:
  """Optional controls for one batched collection call.

  Defaults reproduce the legacy behaviour: full result rows, one store append
  after the schedule, no limits and no progress callbacks.  ``output_bytes``
  overrides the stored-byte measurement and ``clock`` overrides the monotonic
  clock; both exist for hosts that measure differently and for deterministic
  tests.

  ``max_rows_per_append`` is an upper bound on one store append, not one append
  per trial: the publisher buffers whole trials and writes a full chunk once
  ``max_rows_per_append`` rows accumulate (or :data:`DEFAULT_PUBLICATION_CHUNK`
  rows when it is ``None``).
  """

  limits: CollectionLimits | None = None
  stream_rows: bool = False
  compact_results: bool = False
  max_rows_per_append: int | None = None
  tensor_dispatch: bool = False
  progress: ProgressCallback | None = None
  output_bytes: Callable[[], int] | None = None
  clock: Callable[[], float] | None = None

  def __post_init__(self) -> None:
    if self.max_rows_per_append is not None and self.max_rows_per_append <= 0:
      raise CollectionControlError("max_rows_per_append must be positive")
    if self.max_rows_per_append is not None and not self.stream_rows:
      raise CollectionControlError("max_rows_per_append requires stream_rows")
    if self.compact_results and not self.stream_rows:
      raise CollectionControlError("compact_results requires stream_rows")
    if (
      self.limits is not None
      and self.limits.max_output_bytes is not None
      and not self.stream_rows
    ):
      # Without streaming publication the stored output can only be measured
      # after the schedule, so the limit could never stop the run.
      raise CollectionControlError("max_output_bytes requires stream_rows")
    for name in ("progress", "output_bytes", "clock"):
      value = getattr(self, name)
      if value is not None and not callable(value):
        raise CollectionControlError(f"{name} must be callable")


class CollectionLimitWatcher:
  """Track wall-clock, inactivity and stored-output stops for one run.

  All three limits are sampled before every physics batch.  Wall-clock and
  inactivity are cheap to sample; the stored-output measurement walks the store
  directory, so it is re-measured only after a publication marked it dirty.
  """

  def __init__(
    self,
    limits: CollectionLimits | None = None,
    *,
    output_bytes: Callable[[], int] | None = None,
    clock: Callable[[], float] | None = None,
  ) -> None:
    self.limits = limits or CollectionLimits()
    if self.limits.max_output_bytes is not None and output_bytes is None:
      raise CollectionControlError("output-size limit needs a measured stored output")
    self._output_bytes = output_bytes
    self._clock = clock or time.monotonic
    self._started = self._clock()
    self._last_completion = self._started
    self._measured_output: int | None = None
    self._output_dirty = True

  @property
  def started(self) -> float:
    return self._started

  @property
  def last_completion(self) -> float:
    return self._last_completion

  def note_trial_completed(self) -> None:
    """Restart the inactivity deadline after one trial finishes."""
    self._last_completion = self._clock()

  def note_publication(self) -> None:
    """Mark that stored output grew, so the next check re-measures it."""
    self._output_dirty = True

  def measured_output_bytes(self) -> int | None:
    """Return the stored byte count, re-measuring only after a publication."""
    if self._output_bytes is None:
      return None
    if self._output_dirty or self._measured_output is None:
      self._measured_output = int(self._output_bytes())
      self._output_dirty = False
    return self._measured_output

  def clock_stop(self) -> str | None:
    """Return the wall-clock or inactivity stop reason, if one applies."""
    now = self._clock()
    if (
      self.limits.wall_seconds is not None
      and now - self._started >= self.limits.wall_seconds
    ):
      return STOP_WALL_CLOCK
    if (
      self.limits.inactivity_seconds is not None
      and now - self._last_completion >= self.limits.inactivity_seconds
    ):
      return STOP_INACTIVITY
    return None

  def output_stop(self) -> str | None:
    """Return the stored-output stop reason when the store already exceeds it."""
    limit = self.limits.max_output_bytes
    if limit is None:
      return None
    measured = self.measured_output_bytes()
    if measured is not None and measured > limit:
      return STOP_OUTPUT_SIZE
    return None

  def stop_reason(self) -> str | None:
    """Return any stop reason in the scalar report's check order."""
    return self.clock_stop() or self.output_stop()


def stored_output_bytes(store: Any) -> Callable[[], int] | None:
  """Return a stored-byte counter for a store, or ``None`` when unmeasurable.

  The measurement matches the scalar collection report: every file below the
  store root contributes, including the manifest.
  """
  root = getattr(store, "root", None)
  if not isinstance(root, Path):
    return None

  def measure() -> int:
    return sum(path.stat().st_size for path in root.glob("**/*") if path.is_file())

  return measure


def missing_tensor_methods(env: Any, vae: Any) -> tuple[str, ...]:
  """Return the tensor-dispatch methods that the runtime does not provide."""
  return tuple(
    f"{owner}.{name}"
    for owner, name in _TENSOR_DISPATCH_METHODS
    if not callable(getattr(env if owner == "env" else vae, name, None))
  )


class BoundedRowPublisher:
  """Coalesce completed-trial rows into bounded, ordered store appends.

  Rows are buffered and written when the buffer reaches the publication chunk
  (``max_rows_per_append`` rows, or :data:`DEFAULT_PUBLICATION_CHUNK` when it is
  ``None``).  One append therefore carries up to a full chunk of rows instead of
  one finished trial, so the file count scales with published row chunks, not
  with the trial count.  Row order is the completion order.  A trial whose rows
  cross a chunk boundary is split between two appends.  :meth:`flush` writes the
  remaining buffered rows at the end of the run, so no accepted row is dropped.

  Appends are synchronous.  A store fault propagates unchanged and nothing
  later is written; the buffer stays bounded by one chunk plus one trial.
  """

  def __init__(self, store: Any, *, max_rows_per_append: int | None = None) -> None:
    if not callable(getattr(store, "append", None)):
      raise CollectionControlError("streaming publication needs a store append method")
    if max_rows_per_append is not None and max_rows_per_append <= 0:
      raise CollectionControlError("max_rows_per_append must be positive")
    self._store = store
    self._chunk = (
      DEFAULT_PUBLICATION_CHUNK
      if max_rows_per_append is None
      else int(max_rows_per_append)
    )
    self._pending: list[TrajectoryRow] = []
    self.files: list[str] = []
    self.stored_rows = 0

  @property
  def pending_rows(self) -> int:
    """Rows buffered but not yet published."""
    return len(self._pending)

  def add(self, rows: Sequence[TrajectoryRow]) -> tuple[str, ...]:
    """Buffer one finished trial's rows and write every full chunk.

    Returns only the shard files this call created; rows still buffered are
    published by a later :meth:`add` or by :meth:`flush`.
    """
    self._pending.extend(rows)
    written: list[str] = []
    while len(self._pending) >= self._chunk:
      chunk = self._pending[: self._chunk]
      del self._pending[: self._chunk]
      written.extend(self._write(chunk))
    return tuple(written)

  def flush(self) -> tuple[str, ...]:
    """Write the remaining buffered rows; the buffer is empty afterwards."""
    if not self._pending:
      return ()
    chunk = tuple(self._pending)
    self._pending.clear()
    return self._write(chunk)

  def _write(self, chunk: Sequence[TrajectoryRow]) -> tuple[str, ...]:
    if not chunk:
      return ()
    files = tuple(self._store.append(chunk))
    if not files:
      raise CollectionControlError(
        f"store wrote no shard for {len(chunk)} completed rows"
      )
    self.files.extend(files)
    self.stored_rows += len(chunk)
    return files


class CollectionProgressTracker:
  """Count published work and emit :class:`CollectionProgress` records.

  Only observed events move the counters: a trial counts after it finished and
  its rows were published, and stored rows count successful store appends.  A
  raising callback propagates to the caller; the collector never swallows it.
  """

  def __init__(
    self,
    callback: ProgressCallback | None = None,
    *,
    output_bytes: Callable[[], int | None] | None = None,
  ) -> None:
    self.callback = callback
    self._output_bytes = output_bytes
    self.trials_completed = 0
    self.transitions = 0
    self.accepted_rows = 0
    self.stored_rows = 0

  def note_trial(self, *, steps: int, accepted_rows: int) -> None:
    if steps < 0 or accepted_rows < 0:
      raise CollectionControlError("trial counters must be non-negative")
    self.trials_completed += 1
    self.transitions += steps
    self.accepted_rows += accepted_rows

  def note_append(self, *, rows: int) -> None:
    if rows < 0:
      raise CollectionControlError("appended row count must be non-negative")
    self.stored_rows += rows

  def emit(
    self,
    *,
    event: str,
    stop_reason: str | None = None,
    episode_id: str | None = None,
    files: Sequence[str] = (),
  ) -> CollectionProgress:
    """Build one progress record, pass it to the callback and return it."""
    measured = None if self._output_bytes is None else self._output_bytes()
    record = CollectionProgress(
      event=event,
      final=event == "final",
      trials_completed=self.trials_completed,
      transitions=self.transitions,
      stored_rows=self.stored_rows,
      accepted_rows=self.accepted_rows,
      stored_bytes=None if measured is None else int(measured),
      store_files=tuple(files),
      stop_reason=stop_reason,
      episode_id=episode_id,
    )
    if self.callback is not None:
      self.callback(record)
    return record


__all__ = [
  "BoundedRowPublisher",
  "CollectionControlError",
  "DEFAULT_PUBLICATION_CHUNK",
  "CollectionControls",
  "CollectionLimitWatcher",
  "CollectionLimits",
  "CollectionProgress",
  "CollectionProgressTracker",
  "CollectionSummary",
  "ProgressCallback",
  "STOP_INACTIVITY",
  "STOP_OUTPUT_SIZE",
  "STOP_TRANSITION_BUDGET",
  "STOP_WALL_CLOCK",
  "limits_from_request",
  "missing_tensor_methods",
  "stored_output_bytes",
]
