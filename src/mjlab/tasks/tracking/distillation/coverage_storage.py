"""Protected difficult-state replay composed around the legacy batch contract.

The enabled archive is intentionally separate from :mod:`balanced_storage`: the
legacy FIFO and its ``phase_bins`` option remain untouched.  This module owns the
new policy metadata, protected cell quotas, family reservoirs, and a persisted
sampling stream while exposing ``LabeledReplayBatch`` at the trainer boundary.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch

from mjlab.tasks.tracking.distillation.balanced_storage import (
  allocate_motion_capacity_quotas,
)
from mjlab.tasks.tracking.distillation.coverage_metadata import (
  CoverageBatchMetadata,
  CoverageMetadataError,
  CoverageRowMetadata,
)
from mjlab.tasks.tracking.distillation.coverage_policy import CoveragePolicy
from mjlab.tasks.tracking.distillation.observations import PackedObservationBatch
from mjlab.tasks.tracking.distillation.storage import (
  LabeledReplayBatch,
  LabeledReplayBuffer,
  ReplayValidationError,
  _canonical_device,
  validate_replay_batch,
)
from mjlab.tasks.tracking.distillation.vae_config import VaeSchema

_ARCHIVE_FRACTION = 0.20
_DRAW_FRACTION = 0.20
_CATEGORIES = ("student_difficult", "teacher_continuation", "finite_post_failure")
_CATEGORY_SHARES = (0.50, 0.40, 0.10)
_SEVERITIES = ("moderate", "strong")
_PHASE_BINS = 5
_MAX_ROWS_PER_FAMILY_CELL = 32
_MAX_ROWS_PER_FAMILY_MOTION = 256
_STATE_KIND = "x2-protected-coverage-replay"
_STATE_VERSION = 1


class CoverageReplayNotReadyError(ReplayValidationError):
  """A no-replacement draw cannot satisfy the protected replay contract."""


@dataclass(frozen=True, slots=True)
class CoverageReplayBatch:
  """A legacy-compatible tensor batch plus its owned enabled metadata."""

  batch: LabeledReplayBatch
  metadata: CoverageBatchMetadata

  @property
  def observations(self):
    return self.batch.observations

  @property
  def teacher_action(self):
    return self.batch.teacher_action

  @property
  def batch_size(self) -> int:
    return self.batch.batch_size


@dataclass(frozen=True, slots=True)
class CoverageInsertReport:
  """Admission result; ``fresh_row_ids`` is the normalizer integration seam."""

  fresh_row_ids: tuple[str, ...]
  retained_row_ids: tuple[str, ...]
  duplicate_row_ids: tuple[str, ...]
  rejected_row_ids: tuple[str, ...]
  archive_row_ids: tuple[str, ...]


@dataclass(slots=True)
class _StoredRow:
  fields: dict[str, torch.Tensor]
  metadata: CoverageRowMetadata

  def clone(self) -> _StoredRow:
    return _StoredRow(
      {name: value.detach().clone() for name, value in self.fields.items()},
      self.metadata,
    )


@dataclass(frozen=True, slots=True)
class _RecentSelection:
  motion: int
  index: int
  metadata: CoverageRowMetadata


@dataclass(frozen=True, slots=True)
class _RecentRowRef:
  motion: int
  index: int
  metadata: CoverageRowMetadata


@dataclass(slots=True)
class _FamilyBucket:
  rows: list[_StoredRow]
  seen: int = 0
  priority: int = 0


class _RecentPartition:
  """A legacy FIFO ring with an aligned bounded metadata ring.

  The tensor storage and its layout/cursor validation deliberately belong to
  :class:`LabeledReplayBuffer`.  This wrapper only owns the row provenance
  ring, so the enabled provenance layout cannot drift from the legacy loader.
  """

  def __init__(self, capacity: int, schema: VaeSchema) -> None:
    if capacity <= 0:
      raise ReplayValidationError("recent capacity must be positive")
    self.capacity = capacity
    self.schema = schema
    self.buffer = LabeledReplayBuffer(capacity, schema)
    self.metadata: list[CoverageRowMetadata | None] = [None] * capacity
    self._next = 0
    self._size = 0

  def append_batch(
    self, batch: LabeledReplayBatch, metadata: Sequence[CoverageRowMetadata]
  ) -> int:
    """Append one batched tensor payload and its aligned metadata.

    The legacy ring has a special ``count >= capacity`` tail path.  Mirror its
    exact physical cursor rather than deriving a cursor from the incoming
    count; this is what keeps metadata paired with the retained tensor rows on
    both oversized and wrapped inserts.
    """
    if batch.batch_size != len(metadata):
      raise ReplayValidationError("recent batch and metadata lengths differ")
    if not batch.batch_size:
      return 0
    old_next = self.buffer._next
    old_size = self.buffer._size
    self.buffer.insert(batch)
    if batch.batch_size >= self.capacity:
      retained = metadata[-self.capacity :]
      for index, row_metadata in enumerate(retained):
        self.metadata[index] = row_metadata
    else:
      for offset, row_metadata in enumerate(metadata):
        self.metadata[(old_next + offset) % self.capacity] = row_metadata
    self._next = self.buffer._next
    self._size = self.buffer._size
    return max(0, old_size + batch.batch_size - self.capacity)

  def __len__(self) -> int:
    return self._size

  def active_rows(self) -> list[tuple[int, CoverageRowMetadata]]:
    if self._size == 0:
      return []
    start = (self._next - self._size) % self.capacity
    result: list[tuple[int, CoverageRowMetadata]] = []
    for offset in range(self._size):
      index = (start + offset) % self.capacity
      row_metadata = self.metadata[index]
      if row_metadata is None:
        raise ReplayValidationError("recent metadata ring has an empty active slot")
      result.append((index, row_metadata))
    return result

  def max_segment_id(self) -> int | None:
    storage = self.buffer._storage
    if storage is None or self._size == 0:
      return None
    active = self._active_index_tensor(storage["episode_id"].device)
    return int(storage["episode_id"][active].max().item())

  def _active_index_tensor(self, device: torch.device) -> torch.Tensor:
    start = (self._next - self._size) % self.capacity
    return (torch.arange(self._size, device=device) + start) % self.capacity

  def rebase_segment_ids(self, collector_iteration: int, base: int) -> int:
    storage = self.buffer._storage
    if storage is None or self._size == 0 or base == 0:
      return 0
    active = self._active_index_tensor(storage["episode_id"].device)
    selected_iterations = storage["collector_iteration"][active]
    mask = selected_iterations == collector_iteration
    if not bool(mask.any().item()):
      return 0
    storage["episode_id"][active[mask]] += base
    return int(mask.sum().item())

  def state_dict(self) -> dict[str, Any]:
    # Keep the legacy state keys/layout flag intact and add only the aligned
    # provenance ring.  In particular, no tensor is serialized per row.
    state = self.buffer.state_dict()
    state["metadata"] = [
      None if row_metadata is None else row_metadata.as_dict()
      for row_metadata in self.metadata
    ]
    return state

  def _candidate_state(
    self,
    state: object,
    device: torch.device | None,
    dtype: torch.dtype | None,
  ) -> tuple[LabeledReplayBuffer, list[CoverageRowMetadata | None]]:
    if not isinstance(state, Mapping):
      raise ReplayValidationError("recent partition state is invalid")
    if "metadata" not in state:
      raise ReplayValidationError("recent partition metadata is missing")
    state_mapping = cast(Mapping[str, Any], state)
    raw_metadata = state_mapping["metadata"]
    if not isinstance(raw_metadata, Sequence) or isinstance(raw_metadata, (str, bytes)):
      raise ReplayValidationError("recent partition metadata is invalid")
    if len(raw_metadata) != self.capacity:
      raise ReplayValidationError("recent partition metadata has invalid capacity")
    parsed_metadata = [
      None if value is None else CoverageRowMetadata.from_dict(value)
      for value in raw_metadata
    ]
    buffer_state = dict(state)
    buffer_state.pop("metadata")
    candidate = LabeledReplayBuffer(self.capacity, self.schema)
    candidate.validate_state(buffer_state)
    candidate.load_state_dict(buffer_state)
    if candidate.size == 0:
      # Empty motion partitions have no tensor evidence from which to infer a
      # device/dtype.  Adopt the established replay policy without inventing
      # storage or rows; a later correctly-routed insertion remains validated
      # against this policy.
      if (
        candidate.device is not None
        and device is not None
        and candidate.device != device
      ):
        raise ReplayValidationError("recent partition device does not match replay")
      if candidate.dtype is not None and dtype is not None and candidate.dtype != dtype:
        raise ReplayValidationError("recent partition dtype does not match replay")
      candidate._device = device
      candidate._dtype = dtype
    else:
      if device is not None and candidate.device != device:
        raise ReplayValidationError("recent partition device does not match replay")
      if dtype is not None and candidate.dtype != dtype:
        raise ReplayValidationError("recent partition dtype does not match replay")
    if (
      sum(row_metadata is not None for row_metadata in parsed_metadata)
      != candidate.size
    ):
      raise ReplayValidationError("recent metadata occupancy is invalid")
    return candidate, parsed_metadata

  def validate_state(
    self, state: object, device: torch.device | None, dtype: torch.dtype | None
  ) -> None:
    self._candidate_state(state, device, dtype)

  def load_state(
    self,
    state: Mapping[str, Any],
    device: torch.device | None,
    dtype: torch.dtype | None,
  ) -> None:
    candidate, parsed_metadata = self._candidate_state(state, device, dtype)
    # Candidate validation/loading is complete before replacing any live field.
    self.buffer = candidate
    self.metadata = parsed_metadata
    self._size = candidate.size
    self._next = int(candidate._next)

  def row_at(self, index: int, metadata: CoverageRowMetadata) -> _StoredRow:
    storage = self.buffer._storage
    if storage is None:
      raise CoverageReplayNotReadyError("recent partition is empty")
    return _StoredRow(
      {name: value[index].detach().clone() for name, value in storage.items()},
      metadata,
    )


@dataclass(frozen=True, slots=True)
class CoverageReplayReport:
  capacity: int
  recent_capacity: int
  archive_capacity: int
  recent_retained: int
  archive_retained: int
  fresh_rows: int
  drawn_rows: int
  counters: dict[str, int]

  @property
  def retained(self) -> int:
    return self.recent_retained + self.archive_retained

  @property
  def ready(self) -> bool:
    return self.recent_retained > 0


# Imported lazily in a few downstream environments; this alias keeps the
# exception name discoverable without modifying legacy storage.
ReplayNotReadyError = CoverageReplayNotReadyError


def _largest_remainder(total: int, shares: Sequence[float]) -> tuple[int, ...]:
  if (
    total < 0
    or not shares
    or any(value < 0 or not math.isfinite(value) for value in shares)
  ):
    raise ReplayValidationError("invalid largest-remainder allocation")
  weight = math.fsum(shares)
  if total and weight <= 0:
    raise ReplayValidationError("positive capacity needs positive allocation weights")
  exact = (
    [total * value / weight for value in shares] if weight else [0.0] * len(shares)
  )
  counts = [math.floor(value) for value in exact]
  order = sorted(
    range(len(shares)),
    key=lambda index: (-(exact[index] - counts[index]), index),
  )
  for index in order[: total - sum(counts)]:
    counts[index] += 1
  return tuple(counts)


def _row_view(
  batch: LabeledReplayBatch, index: int, metadata: CoverageRowMetadata
) -> _StoredRow:
  """Return a zero-copy row view for bounded archive admission.

  Fresh batches stay batched through the recent ring.  A row wrapper is made
  only when the bounded archive actually needs to inspect/retain that row;
  admission never materializes one tensor dictionary per incoming row.
  """
  fields = {
    "reference": batch.reference[index],
    "conditioning": batch.conditioning[index],
    "teacher_action": batch.teacher_action[index],
    "motion_id": batch.motion_id[index],
    "teacher_id": batch.teacher_id[index],
    "reference_frame": batch.reference_frame[index],
    "episode_id": batch.episode_id[index],
    "collector_iteration": batch.collector_iteration[index],
  }
  provenance = batch.provenance()
  if provenance is not None:
    for name, value in zip(
      ("initialization_kind", "segment_initial_reference_frame", "segment_age"),
      provenance,
      strict=True,
    ):
      fields[name] = value[index]
  return _StoredRow(fields, metadata)


def _row_fields(batch: LabeledReplayBatch, index: int) -> dict[str, torch.Tensor]:
  """Compatibility helper returning an owned single-row field mapping.

  Production insertion uses :func:`_row_view`; this helper remains private and
  is intentionally limited to callers that explicitly need an owned row.
  """
  return (
    _row_view(
      batch,
      index,
      CoverageRowMetadata(
        row_id="coverage-internal-row",
        family_id="coverage-internal-family",
        source_kind="student_difficult",
        motion_id=str(int(batch.motion_id[index].item())),
        source_frame=int(batch.reference_frame[index].item()),
      ),
    )
    .clone()
    .fields
  )


def _metadata_tuple(
  metadata: CoverageBatchMetadata | Sequence[CoverageRowMetadata],
) -> CoverageBatchMetadata:
  if isinstance(metadata, CoverageBatchMetadata):
    return metadata
  try:
    return CoverageBatchMetadata(tuple(metadata))
  except (TypeError, CoverageMetadataError) as exc:
    raise ReplayValidationError("metadata must be an owned coverage batch") from exc


def _effective_source_kind(row: CoverageRowMetadata) -> str:
  """Apply P0 source precedence without inspecting teacher outcomes."""
  if row.terminal or row.first_failure:
    return "finite_post_failure"
  if row.source_branch == "teacher_branch":
    return "teacher_continuation"
  return row.source_kind


def _validate_metadata_types(row: CoverageRowMetadata) -> None:
  for name in ("source_frame", "phase_bin", "branch_age_steps"):
    value = getattr(row, name)
    if isinstance(value, bool) or not isinstance(value, int):
      raise ReplayValidationError(f"metadata {name} must be an integer")
  if row.event_frame is not None and (
    isinstance(row.event_frame, bool) or not isinstance(row.event_frame, int)
  ):
    raise ReplayValidationError("metadata event_frame must be an integer or None")
  for name in ("terminal", "first_failure"):
    if not isinstance(getattr(row, name), bool):
      raise ReplayValidationError(f"metadata {name} must be a bool")


def _cell_key(row: CoverageRowMetadata) -> tuple[str, int, str]:
  # P0 stores nominal teacher/terminal rows in the moderate cell while keeping
  # the actual score in metadata.  The old global phase_bins feature is never
  # consulted here.
  _validate_metadata_types(row)
  if row.phase_bin < 0 or row.phase_bin >= _PHASE_BINS:
    raise ReplayValidationError("metadata phase_bin must be in [0, 5)")
  category = _effective_source_kind(row)
  severity = row.severity_band
  if (
    category in ("teacher_continuation", "finite_post_failure")
    and severity == "nominal"
  ):
    severity = "moderate"
  return (category, row.phase_bin, severity)


def _motion_id(row: _StoredRow) -> int:
  return int(row.fields["motion_id"].item())


def _batch_from_rows(
  rows: Sequence[_StoredRow], schema: VaeSchema
) -> LabeledReplayBatch:
  if not rows:
    raise CoverageReplayNotReadyError("cannot construct an empty replay batch")
  names = tuple(rows[0].fields)
  values = {name: torch.stack([row.fields[name] for row in rows]) for name in names}
  return LabeledReplayBatch(
    observations=PackedObservationBatch(
      values["reference"], values["conditioning"], schema
    ),
    teacher_action=values["teacher_action"],
    motion_id=values["motion_id"],
    teacher_id=values["teacher_id"],
    reference_frame=values["reference_frame"],
    episode_id=values["episode_id"],
    collector_iteration=values["collector_iteration"],
    **{
      name: values[name]
      for name in (
        "initialization_kind",
        "segment_initial_reference_frame",
        "segment_age",
      )
      if name in values
    },
  )


def _batch_from_field_tensors(
  fields: Mapping[str, torch.Tensor], schema: VaeSchema
) -> LabeledReplayBatch:
  return LabeledReplayBatch(
    observations=PackedObservationBatch(
      fields["reference"], fields["conditioning"], schema
    ),
    teacher_action=fields["teacher_action"],
    motion_id=fields["motion_id"],
    teacher_id=fields["teacher_id"],
    reference_frame=fields["reference_frame"],
    episode_id=fields["episode_id"],
    collector_iteration=fields["collector_iteration"],
    **{
      name: fields[name]
      for name in (
        "initialization_kind",
        "segment_initial_reference_frame",
        "segment_age",
      )
      if name in fields
    },
  )


class ProtectedCoverageReplayBuffer:
  """Opt-in protected archive plus a recent per-motion FIFO.

  ``insert`` accepts the legacy ``LabeledReplayBatch`` and a separate owned
  ``CoverageBatchMetadata``.  ``sample`` returns the legacy batch expected by
  trainers; ``sample_with_metadata`` is the enabled integration seam for
  provenance-aware workers and holdouts.
  """

  def __init__(
    self,
    capacity: int,
    schema: VaeSchema,
    motion_weights: Mapping[int, float],
    *,
    policy: CoveragePolicy | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    seed: int = 0,
    row_namespace: str = "default",
    teacher_codes: Mapping[int, int],
    frame_counts: Mapping[int, int],
  ) -> None:
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
      raise ReplayValidationError("capacity must be a positive integer")
    if not isinstance(schema, VaeSchema):
      raise ReplayValidationError("schema must be a VaeSchema")
    if not isinstance(motion_weights, Mapping) or not motion_weights:
      raise ReplayValidationError("motion_weights must be a non-empty mapping")
    if any(
      isinstance(key, bool) or not isinstance(key, int) or key < 0
      for key in motion_weights
    ):
      raise ReplayValidationError("motion weights must use nonnegative integer IDs")
    if dtype is not None and not dtype.is_floating_point:
      raise ReplayValidationError("dtype must be a floating-point torch dtype")
    self.capacity = capacity
    self.schema = schema
    self.policy = (
      policy if policy is not None else CoveragePolicy.difficult_state_coverage()
    )
    if not self.policy.enabled:
      raise ReplayValidationError(
        "protected coverage replay requires an enabled policy"
      )
    self.motion_weights = {
      int(key): float(value) for key, value in motion_weights.items()
    }
    if any(
      key < 0 or not math.isfinite(value) or value <= 0
      for key, value in self.motion_weights.items()
    ):
      raise ReplayValidationError("motion weights must be finite and positive")
    motion_ids = tuple(sorted(self.motion_weights))
    quotas = allocate_motion_capacity_quotas(self.motion_weights, capacity)
    self.motion_quotas = dict(zip(quotas.motion_ids, quotas.quotas, strict=True))
    self.teacher_codes = self._routing_table(
      "teacher_codes", teacher_codes, motion_ids, allow_none=False
    )
    self.frame_counts = self._routing_table(
      "frame_counts", frame_counts, motion_ids, allow_none=False
    )
    self.archive_quotas = {
      motion: math.floor(quota * _ARCHIVE_FRACTION)
      for motion, quota in self.motion_quotas.items()
    }
    self.recent_quotas = {
      motion: quota - self.archive_quotas[motion]
      for motion, quota in self.motion_quotas.items()
    }
    self.archive_capacity = sum(self.archive_quotas.values())
    self.recent_capacity = sum(self.recent_quotas.values())
    if not isinstance(row_namespace, str) or not row_namespace:
      raise ReplayValidationError("row_namespace must be non-empty")
    self._row_namespace = row_namespace
    self._device: torch.device | None = (
      _canonical_device(device) if device is not None else None
    )
    self._dtype: torch.dtype | None = dtype
    self._provenance_enabled: bool | None = None
    self._recent: dict[int, _RecentPartition] = {
      motion: _RecentPartition(self.recent_quotas[motion], schema)
      for motion in motion_ids
    }
    self._cells: dict[tuple[int, tuple[str, int, str]], dict[str, _FamilyBucket]] = {}
    # Family admission is a reservoir over distinct family identities.  The
    # identity cache is deliberately bounded per cell; it prevents repeated
    # rows from a rejected family being mistaken for new families without
    # retaining an unbounded lifetime set.
    self._family_salt = hashlib.sha256(
      f"x2-coverage-family-salt:{seed}".encode()
    ).hexdigest()
    self._last_collection_call: int | None = None
    self._family_motion_counts: dict[tuple[int, str], int] = defaultdict(int)
    self._dedup_capacity = max(1024, capacity * 4)
    self._dedup_ids: dict[str, None] = {}
    self._cell_quotas = self._make_cell_quotas()
    self._fresh_count: int = 0
    self._draw_count: int = 0
    self._counters: dict[str, int] = defaultdict(int)
    self._rng = torch.Generator(device="cpu")
    self._rng.manual_seed(seed)
    self._draw_generator: torch.Generator | None = None

  def coverage_replay_policy(self) -> dict[str, Any]:
    """Return the v4 partition policy descriptor a coverage checkpoint records.

    The checkpoint reads this hook and passes the result to
    ``ReplayPolicy.from_dict``, so the return value is a mapping.

    The legacy ``ReplayPolicy.from_buffer`` records only the balanced and the FIFO
    layouts, so the enabled buffer answers for itself through this hook.  The
    checkpoint reads it through ``_coverage_replay_policy``.  Every tuple is driven
    from one motion order, so a disagreement between the stored tables raises
    instead of producing a silently misaligned policy.
    """
    from mjlab.tasks.tracking.distillation.cohort_contract import ReplayPolicy

    motion_ids = tuple(int(motion) for motion in self.motion_weights)
    return ReplayPolicy(
      # The kind the checkpoint accepts for an enabled run (see _live_replay_policy).
      kind="coverage-protected-replay",
      capacity=self.capacity,
      motion_ids=motion_ids,
      weights=tuple(float(self.motion_weights[motion]) for motion in motion_ids),
      quotas=tuple(int(self.motion_quotas[motion]) for motion in motion_ids),
      teacher_codes=tuple(int(self.teacher_codes[motion]) for motion in motion_ids),
      frame_counts=tuple(int(self.frame_counts[motion]) for motion in motion_ids),
    ).as_dict()

  @staticmethod
  def _routing_table(
    name: str,
    values: Mapping[int, int] | None,
    motion_ids: tuple[int, ...],
    *,
    allow_none: bool,
  ) -> dict[int, int] | None:
    if values is None:
      if allow_none:
        return None
      raise ReplayValidationError(f"{name} must be supplied")
    if not isinstance(values, Mapping) or set(values) != set(motion_ids):
      raise ReplayValidationError(
        f"{name} keys must exactly match selected motions {list(motion_ids)}"
      )
    result: dict[int, int] = {}
    for motion in motion_ids:
      value = values[motion]
      if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReplayValidationError(f"{name}[{motion}] must be a nonnegative integer")
      if name == "frame_counts" and value == 0:
        raise ReplayValidationError(f"frame_counts[{motion}] must be positive")
      result[motion] = value
    if name == "teacher_codes" and len(set(result.values())) != len(result):
      raise ReplayValidationError("teacher_codes must be distinct per motion")
    return result

  def _active_rng(self) -> torch.Generator:
    return self._draw_generator if self._draw_generator is not None else self._rng

  def _make_cell_quotas(self) -> dict[tuple[int, tuple[str, int, str]], int]:
    result: dict[tuple[int, tuple[str, int, str]], int] = {}
    for motion, total in self.archive_quotas.items():
      category_counts = _largest_remainder(total, _CATEGORY_SHARES)
      for category, category_count in zip(_CATEGORIES, category_counts, strict=True):
        phase_counts = _largest_remainder(category_count, (1.0,) * _PHASE_BINS)
        for phase, phase_count in enumerate(phase_counts):
          severity_counts = _largest_remainder(phase_count, (1.0, 1.0))
          for severity, cell_count in zip(_SEVERITIES, severity_counts, strict=True):
            result[(motion, (category, phase, severity))] = cell_count
    return result

  @property
  def size(self) -> int:
    return sum(len(rows) for rows in self._recent.values()) + self._archive_size()

  @property
  def device(self) -> torch.device | None:
    return self._device

  @property
  def dtype(self) -> torch.dtype | None:
    return self._dtype

  @property
  def is_empty(self) -> bool:
    return self.size == 0

  def __len__(self) -> int:
    return self.size

  def _validate_metadata(
    self, batch: LabeledReplayBatch, metadata: CoverageBatchMetadata
  ) -> None:
    validate_replay_batch(batch, self.schema, device=self._device, dtype=self._dtype)
    has_provenance = batch.has_provenance
    if (
      self._provenance_enabled is not None
      and has_provenance != self._provenance_enabled
    ):
      expected = "enabled" if self._provenance_enabled else "legacy"
      raise ReplayValidationError(
        f"coverage replay layout is {expected}; provenance layout cannot be mixed"
      )
    if len(metadata) != batch.batch_size:
      raise ReplayValidationError("coverage metadata and replay batch lengths differ")
    if len(set(row.row_id for row in metadata)) != len(metadata):
      raise ReplayValidationError("coverage row IDs must be unique in an insertion")
    if batch.batch_size == 0:
      return
    for index, row in enumerate(metadata):
      _validate_metadata_types(row)
      tensor_motion = int(batch.motion_id[index].item())
      if row.source_frame != int(batch.reference_frame[index].item()):
        raise ReplayValidationError("metadata source_frame does not match replay frame")
      if not row.motion_id.isdecimal() or int(row.motion_id) != tensor_motion:
        raise ReplayValidationError(
          "metadata motion_id must exactly match tensor routing"
        )
      if tensor_motion not in self._recent:
        raise ReplayValidationError(f"unknown replay motion {tensor_motion}")
      teacher_id = int(batch.teacher_id[index].item())
      if teacher_id < 0:
        raise ReplayValidationError("teacher_id must be nonnegative")
      if (
        self.teacher_codes is not None
        and teacher_id != self.teacher_codes[tensor_motion]
      ):
        raise ReplayValidationError(
          f"teacher_id {teacher_id} does not match motion {tensor_motion} routing"
        )
      source_frame = int(batch.reference_frame[index].item())
      if (
        self.frame_counts is not None
        and not 0 <= source_frame < self.frame_counts[tensor_motion]
      ):
        raise ReplayValidationError(
          f"reference frame {source_frame} is outside motion {tensor_motion}"
        )
      _cell_key(row)

  def validate_batch(
    self,
    batch: LabeledReplayBatch,
    metadata: CoverageBatchMetadata | Sequence[CoverageRowMetadata] | None = None,
  ) -> None:
    validate_replay_batch(batch, self.schema, device=self._device, dtype=self._dtype)
    has_provenance = batch.has_provenance
    if (
      self._provenance_enabled is not None
      and has_provenance != self._provenance_enabled
    ):
      raise ReplayValidationError("coverage replay layout does not match buffer")
    if metadata is not None:
      self._validate_metadata(batch, _metadata_tuple(metadata))

  def _archive_size(self) -> int:
    return sum(
      len(bucket.rows)
      for families in self._cells.values()
      for bucket in families.values()
    )

  def _family_count(self, motion: int) -> int:
    return len(
      {
        family
        for (row_motion, _), families in self._cells.items()
        if row_motion == motion
        for family, bucket in families.items()
        if bucket.rows
      }
    )

  def _family_priority(
    self, family_id: str, motion: int, cell: tuple[str, int, str]
  ) -> int:
    return int(
      hashlib.sha256(
        f"{self._family_salt}:{motion}:{cell}:{family_id}".encode()
      ).hexdigest(),
      16,
    )

  def _archive_insert(
    self,
    batch: LabeledReplayBatch,
    index: int,
    row_metadata: CoverageRowMetadata,
    microbatch_family_counts: dict[str, int],
  ) -> bool:
    """Admit one fresh row without allocating until archive retention is likely."""
    motion = int(batch.motion_id[index].item())
    cell = _cell_key(row_metadata)
    quota = self._cell_quotas.get((motion, cell), 0)
    if (
      row_metadata.source_kind == "student_difficult"
      and row_metadata.severity_band == "nominal"
      and not row_metadata.terminal
      and not row_metadata.first_failure
    ):
      self._counters["nominal_not_archived"] += 1
      return False
    if quota <= 0:
      self._counters["unused_cell_rejections"] += 1
      return False
    family_id = row_metadata.family_id
    family_key = (motion, family_id)
    if microbatch_family_counts.get(family_id, 0) >= _MAX_ROWS_PER_FAMILY_CELL:
      self._counters["microbatch_family_rejections"] += 1
      return False
    key = (motion, cell)
    families = self._cells.get(key)
    max_families = max(1, quota // _MAX_ROWS_PER_FAMILY_CELL)
    priority = self._family_priority(family_id, motion, cell)
    family = families.get(family_id) if families is not None else None
    if family is None:
      motion_family_rows = self._family_motion_counts.get(family_key, 0)
      if motion_family_rows >= _MAX_ROWS_PER_FAMILY_MOTION:
        self._counters["family_motion_cap_rejections"] += 1
        return False
      if families is None:
        families = {}
      if len(families) < max_families:
        family = _FamilyBucket([], priority=priority)
        families[family_id] = family
        self._cells[key] = families
      else:
        worst_family, worst_bucket = max(
          families.items(), key=lambda item: item[1].priority
        )
        if priority >= worst_bucket.priority:
          self._counters["family_reservoir_rejections"] += 1
          return False
        families.pop(worst_family)
        remaining = self._family_motion_counts.get((motion, worst_family), 0) - len(
          worst_bucket.rows
        )
        if remaining:
          self._family_motion_counts[(motion, worst_family)] = remaining
        else:
          self._family_motion_counts.pop((motion, worst_family), None)
        self._counters["family_replacements"] += 1
        family = _FamilyBucket([], priority=priority)
        families[family_id] = family

    motion_family_rows = self._family_motion_counts.get(family_key, 0)
    row_limit = min(_MAX_ROWS_PER_FAMILY_CELL, quota)
    if len(family.rows) < row_limit and motion_family_rows >= (
      _MAX_ROWS_PER_FAMILY_MOTION
    ):
      self._counters["family_motion_cap_rejections"] += 1
      return False
    family.seen += 1
    if len(family.rows) < row_limit:
      family.rows.append(_row_view(batch, index, row_metadata).clone())
      self._family_motion_counts[family_key] = motion_family_rows + 1
    else:
      draw = int(torch.randint(family.seen, (), generator=self._active_rng()).item())
      if draw < row_limit:
        family.rows[draw] = _row_view(batch, index, row_metadata).clone()
      else:
        self._counters["row_reservoir_rejections"] += 1
        return False
    microbatch_family_counts[family_id] = microbatch_family_counts.get(family_id, 0) + 1
    return True

  def _classify_row_ids(
    self, metadata: CoverageBatchMetadata
  ) -> tuple[list[int], list[str], list[str]]:
    fresh_indices: list[int] = []
    fresh_ids: list[str] = []
    duplicate_ids: list[str] = []
    for index, row_metadata in enumerate(metadata):
      if row_metadata.row_id in self._dedup_ids:
        duplicate_ids.append(row_metadata.row_id)
      else:
        fresh_indices.append(index)
        fresh_ids.append(row_metadata.row_id)
    return fresh_indices, fresh_ids, duplicate_ids

  def _commit_row_ids(
    self, fresh_ids: Sequence[str], duplicate_ids: Sequence[str]
  ) -> None:
    for _row_id in duplicate_ids:
      self._counters["duplicate_rows"] += 1
    for row_id in fresh_ids:
      self._dedup_ids[row_id] = None
      if len(self._dedup_ids) > self._dedup_capacity:
        self._dedup_ids.pop(next(iter(self._dedup_ids)))
        self._counters["dedup_evictions"] += 1
    self._fresh_count += len(fresh_ids)

  @staticmethod
  def _slice_batch(
    batch: LabeledReplayBatch, indices: torch.Tensor
  ) -> LabeledReplayBatch:
    provenance = batch.provenance()
    return LabeledReplayBatch(
      observations=PackedObservationBatch(
        batch.reference.index_select(0, indices),
        batch.conditioning.index_select(0, indices),
        batch.schema,
      ),
      teacher_action=batch.teacher_action.index_select(0, indices),
      motion_id=batch.motion_id.index_select(0, indices),
      teacher_id=batch.teacher_id.index_select(0, indices),
      reference_frame=batch.reference_frame.index_select(0, indices),
      episode_id=batch.episode_id.index_select(0, indices),
      collector_iteration=batch.collector_iteration.index_select(0, indices),
      **(
        {
          name: value.index_select(0, indices)
          for name, value in zip(
            ("initialization_kind", "segment_initial_reference_frame", "segment_age"),
            provenance or (),
            strict=True,
          )
        }
        if provenance is not None
        else {}
      ),
    )

  def _validate_collection_call(
    self, collection_call_id: int | None, row_namespace: str | None
  ) -> None:
    if collection_call_id is not None and (
      isinstance(collection_call_id, bool)
      or not isinstance(collection_call_id, int)
      or collection_call_id < 0
    ):
      raise ReplayValidationError("collection_call_id must be nonnegative")
    if row_namespace is not None and row_namespace != self._row_namespace:
      raise ReplayValidationError("row namespace does not match replay buffer")
    if collection_call_id is not None and (
      self._last_collection_call is not None
      and collection_call_id <= self._last_collection_call
    ):
      raise ReplayValidationError("stale or out-of-order collection call")

  def _insert_validated(
    self,
    batch: LabeledReplayBatch,
    metadata: CoverageBatchMetadata,
    *,
    archive: bool,
  ) -> CoverageInsertReport:
    """Commit one already-validated batched collection payload."""
    fresh_indices, fresh_ids, duplicate_ids = self._classify_row_ids(metadata)
    if not batch.batch_size:
      return CoverageInsertReport((), (), (), (), ())
    if not fresh_indices:
      self._commit_row_ids((), duplicate_ids)
      return CoverageInsertReport((), (), tuple(duplicate_ids), (), ())

    # Validate each grouped legacy insertion before touching any partition.  A
    # later per-motion layout/device failure therefore cannot leave a partial
    # collection call in the recent ring.
    indices_by_motion: dict[int, list[int]] = defaultdict(list)
    for index in fresh_indices:
      indices_by_motion[int(batch.motion_id[index].item())].append(index)
    grouped: list[tuple[int, LabeledReplayBatch, tuple[CoverageRowMetadata, ...]]] = []
    for motion, indices in indices_by_motion.items():
      index_tensor = torch.as_tensor(
        indices, dtype=torch.int64, device=batch.reference.device
      )
      group_batch = self._slice_batch(batch, index_tensor)
      group_metadata = tuple(metadata.rows[index] for index in indices)
      self._recent[motion].buffer.validate_batch(group_batch)
      grouped.append((motion, group_batch, group_metadata))

    if self._device is None:
      self._device = _canonical_device(batch.device)
    if self._dtype is None:
      self._dtype = batch.dtype
    if self._provenance_enabled is None:
      self._provenance_enabled = batch.has_provenance

    for motion, group_batch, group_metadata in grouped:
      evicted = self._recent[motion].append_batch(group_batch, group_metadata)
      self._counters["recent_fifo_evictions"] += evicted

    archived: list[str] = []
    rejected: list[str] = []
    if archive:
      family_counts: dict[str, int] = {}
      for index in fresh_indices:
        row_metadata = metadata.rows[index]
        if self._archive_insert(batch, index, row_metadata, family_counts):
          archived.append(row_metadata.row_id)
        else:
          rejected.append(row_metadata.row_id)
    self._commit_row_ids(fresh_ids, duplicate_ids)
    return CoverageInsertReport(
      tuple(fresh_ids),
      tuple(fresh_ids),
      tuple(duplicate_ids),
      tuple(rejected),
      tuple(archived),
    )

  def insert_recent(
    self,
    batch: LabeledReplayBatch,
    metadata: CoverageBatchMetadata | Sequence[CoverageRowMetadata] | None = None,
    collection_call_id: int | None = None,
    row_namespace: str | None = None,
  ) -> CoverageInsertReport:
    """Insert ordinary rows into recent replay without archive admission."""
    self._validate_collection_call(collection_call_id, row_namespace)
    self.validate_batch(batch, metadata)
    if metadata is None:
      if collection_call_id is None or row_namespace is None:
        raise ReplayValidationError(
          "metadata-free recent insertion requires collection_call_id and row_namespace"
        )
      metadata = CoverageBatchMetadata(
        tuple(
          CoverageRowMetadata(
            row_id=(
              f"{row_namespace}:{collection_call_id}:recent-"
              f"{int(batch.motion_id[i])}-{int(batch.episode_id[i])}-"
              f"{int(batch.reference_frame[i])}-{i}"
            ),
            family_id=f"recent-{int(batch.episode_id[i])}",
            source_kind="student_difficult",
            motion_id=str(int(batch.motion_id[i])),
            source_frame=int(batch.reference_frame[i]),
          )
          for i in range(batch.batch_size)
        )
      )
    owned_metadata = _metadata_tuple(metadata)
    self._validate_metadata(batch, owned_metadata)
    report = self._insert_validated(batch, owned_metadata, archive=False)
    if collection_call_id is not None:
      self._last_collection_call = collection_call_id
    return report

  def insert(
    self,
    batch: LabeledReplayBatch | CoverageReplayBatch,
    metadata: CoverageBatchMetadata | Sequence[CoverageRowMetadata] | None = None,
    collection_call_id: int | None = None,
    row_namespace: str | None = None,
  ) -> CoverageInsertReport:
    if isinstance(batch, CoverageReplayBatch):
      if metadata is not None:
        raise ReplayValidationError("metadata must not be passed twice")
      metadata = batch.metadata
      batch = batch.batch
    self._validate_collection_call(collection_call_id, row_namespace)
    if metadata is None:
      return self.insert_recent(
        batch,
        collection_call_id=collection_call_id,
        row_namespace=row_namespace,
      )
    owned_metadata = _metadata_tuple(metadata)
    self._validate_metadata(batch, owned_metadata)
    report = self._insert_validated(batch, owned_metadata, archive=True)
    if collection_call_id is not None:
      self._last_collection_call = collection_call_id
    return report

  def insert_committed(
    self,
    batch: LabeledReplayBatch | CoverageReplayBatch,
    metadata: CoverageBatchMetadata | Sequence[CoverageRowMetadata],
    *,
    collection_call_id: int,
    row_namespace: str,
  ) -> CoverageInsertReport:
    """Insert one immutable main/branch union under a monotonic call identity."""
    if not isinstance(row_namespace, str) or not row_namespace:
      raise ReplayValidationError("row_namespace must be non-empty")
    if isinstance(collection_call_id, bool) or not isinstance(collection_call_id, int):
      raise ReplayValidationError("collection_call_id must be nonnegative")
    owned = _metadata_tuple(metadata)
    prefix = f"{row_namespace}:{collection_call_id}:"
    if any(not row.row_id.startswith(prefix) for row in owned):
      raise ReplayValidationError(
        "committed row IDs must carry the supplied namespace and collection call"
      )
    return self.insert(
      batch,
      owned,
      collection_call_id=collection_call_id,
      row_namespace=row_namespace,
    )

  add = insert

  def _draw_indices(self, count: int) -> list[int]:
    if count <= 0:
      return []
    return [
      int(value)
      for value in torch.randperm(count, generator=self._active_rng()).tolist()
    ]

  def _archive_rows(self, motion: int) -> list[_StoredRow]:
    return [
      row
      for (row_motion, _), families in self._cells.items()
      if row_motion == motion
      for bucket in families.values()
      for row in bucket.rows
    ]

  def _archive_families_ready(self, motion: int) -> bool:
    return self._family_count(motion) >= 8

  def _sample_archive(
    self,
    motion: int,
    count: int,
    used: set[str],
    family_counts: dict[str, int],
  ) -> list[_StoredRow]:
    rows = [
      row for row in self._archive_rows(motion) if row.metadata.row_id not in used
    ]
    if not rows or count <= 0:
      return []
    # Allocate by non-empty category targets, then evenly across their cells.
    by_category: dict[str, list[_StoredRow]] = defaultdict(list)
    for row in rows:
      by_category[_effective_source_kind(row.metadata)].append(row)
    category_counts = _largest_remainder(count, _CATEGORY_SHARES)
    selected: list[_StoredRow] = []
    for category, target in zip(_CATEGORIES, category_counts, strict=True):
      candidates = by_category.get(category, [])
      cells: dict[tuple[int, str], list[_StoredRow]] = defaultdict(list)
      for row in candidates:
        key = (row.metadata.phase_bin, _cell_key(row.metadata)[2])
        cells[key].append(row)
      phase_counts = _largest_remainder(target, (1.0,) * _PHASE_BINS)
      for phase, phase_target in enumerate(phase_counts):
        severity_counts = _largest_remainder(phase_target, (1.0, 1.0))
        for severity, cell_target in zip(_SEVERITIES, severity_counts, strict=True):
          key = (phase, severity)
          choices = [
            row for row in cells.get(key, ()) if row.metadata.row_id not in used
          ]
          order = self._draw_indices(len(choices))
          for index in order[:cell_target]:
            row = choices[index]
            if (
              row.metadata.row_id not in used
              and family_counts.get(row.metadata.family_id, 0)
              < _MAX_ROWS_PER_FAMILY_CELL
            ):
              selected.append(row)
              used.add(row.metadata.row_id)
              family_counts[row.metadata.family_id] = (
                family_counts.get(row.metadata.family_id, 0) + 1
              )

    return selected

  def _sample_recent(
    self,
    motion: int,
    count: int,
    used: set[str],
    family_counts: dict[str, int],
  ) -> list[_RecentSelection]:
    partition = self._recent[motion]
    candidates = [
      (index, metadata)
      for index, metadata in partition.active_rows()
      if metadata.row_id not in used
    ]
    if len(candidates) < count:
      raise CoverageReplayNotReadyError(
        f"recent replay motion {motion} has {len(candidates)} unique rows, needs {count}"
      )
    order = self._draw_indices(len(candidates))
    selected: list[_RecentSelection] = []
    for candidate_index in order:
      physical_index, row_metadata = candidates[candidate_index]
      if family_counts.get(row_metadata.family_id, 0) >= _MAX_ROWS_PER_FAMILY_CELL:
        continue
      selected.append(_RecentSelection(motion, physical_index, row_metadata))
      used.add(row_metadata.row_id)
      family_counts[row_metadata.family_id] = (
        family_counts.get(row_metadata.family_id, 0) + 1
      )

      if len(selected) == count:
        break
    if len(selected) < count:
      raise CoverageReplayNotReadyError(
        f"recent replay motion {motion} family cap leaves fewer than {count} rows"
      )
    return selected

  def _assemble_selected(
    self,
    selected: Sequence[_StoredRow | _RecentSelection],
  ) -> CoverageReplayBatch:
    """Gather recent rows by raw ring indices and stack only bounded archive rows."""
    recent_groups: dict[int, list[tuple[int, _RecentSelection]]] = defaultdict(list)
    archive_rows: list[tuple[int, _StoredRow]] = []
    for position, selection in enumerate(selected):
      if isinstance(selection, _RecentSelection):
        recent_groups[selection.motion].append((position, selection))
      else:
        archive_rows.append((position, selection))

    chunks: list[
      tuple[list[int], dict[str, torch.Tensor], tuple[CoverageRowMetadata, ...]]
    ] = []
    field_names: tuple[str, ...] | None = None
    for motion in sorted(recent_groups):
      partition = self._recent[motion]
      storage = partition.buffer._storage
      if storage is None:
        raise CoverageReplayNotReadyError("recent replay partition lost its storage")
      entries = recent_groups[motion]
      indices = torch.tensor(
        [selection.index for _, selection in entries],
        dtype=torch.int64,
        device=storage["reference"].device,
      )
      fields = {name: storage[name][indices] for name in storage}
      field_names = tuple(storage)
      chunks.append(
        (
          [position for position, _ in entries],
          fields,
          tuple(selection.metadata for _, selection in entries),
        )
      )
    if archive_rows:
      if field_names is None:
        field_names = tuple(archive_rows[0][1].fields)
      fields = {
        name: torch.stack([row.fields[name] for _, row in archive_rows])
        for name in field_names
      }
      chunks.append(
        (
          [position for position, _ in archive_rows],
          fields,
          tuple(row.metadata for _, row in archive_rows),
        )
      )
    if not chunks or field_names is None:
      raise CoverageReplayNotReadyError("cannot assemble an empty replay batch")
    base_fields = {
      name: torch.cat([fields[name] for _, fields, _ in chunks], dim=0)
      for name in field_names
    }
    base_metadata = tuple(
      row_metadata for _, _, metadata in chunks for row_metadata in metadata
    )
    order = self._draw_indices(len(selected))
    order_tensor = torch.tensor(
      order, dtype=torch.int64, device=base_fields["reference"].device
    )
    output_fields = {
      name: value.index_select(0, order_tensor) for name, value in base_fields.items()
    }
    output_metadata = CoverageBatchMetadata(
      tuple(base_metadata[index] for index in order)
    )
    return CoverageReplayBatch(
      _batch_from_field_tensors(output_fields, self.schema), output_metadata
    )

  def _motion_draw_counts(self, batch_size: int) -> dict[int, int]:
    motions = sorted(self.motion_weights)
    values = [self.motion_weights[motion] for motion in motions]
    total = math.fsum(values)
    exact = [batch_size * value / total for value in values]
    counts = [math.floor(value) for value in exact]
    leftover = batch_size - sum(counts)
    if leftover:
      remainders = torch.tensor(
        [value - math.floor(value) for value in exact], dtype=torch.float64
      )
      if float(remainders.sum()) == 0.0:
        remainders.fill_(1.0)
      picks = torch.multinomial(
        remainders / remainders.sum(),
        leftover,
        replacement=False,
        generator=self._active_rng(),
      )
      for index in picks.tolist():
        counts[index] += 1
    return dict(zip(motions, counts, strict=True))

  def sample_with_metadata(
    self,
    batch_size: int,
    *,
    replacement: bool = False,
    generator: torch.Generator | None = None,
  ) -> CoverageReplayBatch:
    if generator is not None:
      if not isinstance(generator, torch.Generator):
        raise ReplayValidationError("generator must be a torch.Generator")
      if torch.device(generator.device).type != "cpu":
        raise ReplayValidationError(
          "coverage replay currently requires a CPU generator"
        )
    old_generator = self._draw_generator
    old_state = (generator or self._rng).get_state().clone()
    old_draw_count = self._draw_count
    old_counters = dict(self._counters)
    self._draw_generator = generator
    try:
      return self._sample_with_metadata_impl(batch_size, replacement=replacement)
    except Exception:
      (generator or self._rng).set_state(old_state)
      self._draw_count = old_draw_count
      self._counters = defaultdict(int, old_counters)
      raise
    finally:
      self._draw_generator = old_generator

  def _sample_with_metadata_impl(
    self, batch_size: int, *, replacement: bool = False
  ) -> CoverageReplayBatch:
    if (
      isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0
    ):
      raise ReplayValidationError("sample batch_size must be a positive integer")
    if not isinstance(replacement, bool):
      raise ReplayValidationError("replacement must be a bool")
    if replacement:
      raise ReplayValidationError("protected coverage draws are without replacement")
    if self.is_empty:
      raise CoverageReplayNotReadyError("cannot sample an empty protected replay")
    selected: list[_StoredRow | _RecentSelection] = []
    used: set[str] = set()
    family_counts: dict[str, int] = {}
    for motion, motion_count in self._motion_draw_counts(batch_size).items():
      if motion_count == 0:
        continue
      archive_count = (
        math.floor(_DRAW_FRACTION * motion_count)
        if self._archive_families_ready(motion)
        else 0
      )
      archive = self._sample_archive(motion, archive_count, used, family_counts)
      selected.extend(archive)
      recent_need = motion_count - len(archive)
      try:
        recent = self._sample_recent(motion, recent_need, used, family_counts)
      except CoverageReplayNotReadyError:
        self._counters["not_ready_recent_shortage"] += 1
        raise
      selected.extend(recent)
      if len(archive) < archive_count:
        self._counters["archive_shortfall_backfilled_recent"] += archive_count - len(
          archive
        )
    if len(selected) != batch_size:
      raise CoverageReplayNotReadyError(
        "protected replay could not fill a unique microbatch"
      )
    result = self._assemble_selected(selected)
    self._draw_count += len(selected)
    return result

  def sample(
    self,
    batch_size: int,
    *,
    replacement: bool = False,
    generator: torch.Generator | None = None,
  ) -> LabeledReplayBatch:
    return self.sample_with_metadata(
      batch_size, replacement=replacement, generator=generator
    ).batch

  def fresh_row_ids(self, report: CoverageInsertReport) -> tuple[str, ...]:
    """Explicit normalizer seam: archive copies are not fresh rows."""
    return report.fresh_row_ids

  def report(self) -> CoverageReplayReport:
    return CoverageReplayReport(
      self.capacity,
      self.recent_capacity,
      self.archive_capacity,
      sum(len(rows) for rows in self._recent.values()),
      self._archive_size(),
      self._fresh_count,
      self._draw_count,
      dict(self._counters),
    )

  def _serialize_row(self, row: _StoredRow) -> dict[str, Any]:
    return {
      "fields": {name: value.detach().clone() for name, value in row.fields.items()},
      "metadata": row.metadata.as_dict(),
    }

  def state_dict(self) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for (motion, cell), families in sorted(self._cells.items()):
      if not families:
        continue
      cells.append(
        {
          "motion": motion,
          "cell": list(cell),
          "families": {
            family: {
              "seen": bucket.seen,
              "priority": bucket.priority,
              "rows": [self._serialize_row(row) for row in bucket.rows],
            }
            for family, bucket in sorted(families.items())
          },
        }
      )
    return {
      "kind": _STATE_KIND,
      "version": _STATE_VERSION,
      "capacity": self.capacity,
      "schema": self.schema.compatibility_metadata(),
      "policy_identity": self.policy.identity,
      "motion_weights": dict(self.motion_weights),
      "motion_quotas": dict(self.motion_quotas),
      "archive_quotas": dict(self.archive_quotas),
      "recent_quotas": dict(self.recent_quotas),
      "teacher_codes": None if self.teacher_codes is None else dict(self.teacher_codes),
      "frame_counts": None if self.frame_counts is None else dict(self.frame_counts),
      "device": None if self._device is None else str(self._device),
      "dtype": None if self._dtype is None else str(self._dtype).replace("torch.", ""),
      "recent": self._recent_state_dicts(),
      "cells": cells,
      "dedup_capacity": self._dedup_capacity,
      "dedup_ids": list(self._dedup_ids),
      "row_namespace": self._row_namespace,
      "family_salt": self._family_salt,
      "last_collection_call": self._last_collection_call,
      "family_motion_counts": [
        {"motion": motion, "family": family, "count": count}
        for (motion, family), count in sorted(self._family_motion_counts.items())
        if count
      ],
      "fresh_count": self._fresh_count,
      "draw_count": self._draw_count,
      "counters": dict(self._counters),
      "rng_state": self._rng.get_state().clone(),
      **({"layout": "provenance-v1"} if self._provenance_enabled else {}),
    }

  def _recent_state_dicts(self) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for motion, partition in self._recent.items():
      partition_state = partition.state_dict()
      if self._provenance_enabled:
        partition_state["layout"] = "provenance-v1"
      result[motion] = partition_state
    return result

  def _decode_state(
    self, state: Mapping[str, Any]
  ) -> tuple[
    dict[int, _RecentPartition],
    dict[tuple[int, tuple[str, int, str]], dict[str, _FamilyBucket]],
    dict[str, None],
    dict[tuple[int, str], int],
    int | None,
    torch.device | None,
    torch.dtype | None,
    int,
    int,
    dict[str, int],
    torch.Tensor,
    bool,
  ]:
    base_required = {
      "kind",
      "version",
      "capacity",
      "schema",
      "policy_identity",
      "motion_weights",
      "motion_quotas",
      "archive_quotas",
      "recent_quotas",
      "teacher_codes",
      "frame_counts",
      "device",
      "dtype",
      "recent",
      "cells",
      "dedup_capacity",
      "dedup_ids",
      "row_namespace",
      "family_salt",
      "last_collection_call",
      "family_motion_counts",
      "fresh_count",
      "draw_count",
      "counters",
      "rng_state",
    }
    raw_layout = state.get("layout")
    if raw_layout is not None and raw_layout != "provenance-v1":
      raise ReplayValidationError("coverage replay layout is invalid")
    required = base_required | ({"layout"} if raw_layout is not None else set())
    if (
      set(state) != required
      or state["kind"] != _STATE_KIND
      or state["version"] != _STATE_VERSION
    ):
      raise ReplayValidationError("coverage replay state fields are invalid")
    state_provenance = raw_layout == "provenance-v1"
    if (
      self._provenance_enabled is not None
      and state_provenance != self._provenance_enabled
    ):
      raise ReplayValidationError("coverage replay layout does not match buffer")
    if (
      state["capacity"] != self.capacity
      or state["schema"] != self.schema.compatibility_metadata()
      or state["policy_identity"] != self.policy.identity
    ):
      raise ReplayValidationError("coverage replay state identity does not match")
    if (
      state["motion_weights"] != self.motion_weights
      or state["motion_quotas"] != self.motion_quotas
      or state["archive_quotas"] != self.archive_quotas
      or state["recent_quotas"] != self.recent_quotas
    ):
      raise ReplayValidationError("coverage replay capacity policy does not match")
    if state["teacher_codes"] != self.teacher_codes:
      raise ReplayValidationError("coverage replay teacher routing does not match")
    if state["frame_counts"] != self.frame_counts:
      raise ReplayValidationError("coverage replay frame routing does not match")
    raw_device = state["device"]
    if raw_device is not None and not isinstance(raw_device, str):
      raise ReplayValidationError("coverage replay device is invalid")
    try:
      device = _canonical_device(raw_device) if raw_device is not None else None
    except (RuntimeError, TypeError, ValueError) as exc:
      raise ReplayValidationError("coverage replay device is invalid") from exc
    raw_dtype = state["dtype"]
    if raw_dtype is not None and not isinstance(raw_dtype, str):
      raise ReplayValidationError("coverage replay dtype is invalid")
    try:
      dtype = getattr(torch, raw_dtype) if raw_dtype is not None else None
    except AttributeError as exc:
      raise ReplayValidationError("coverage replay dtype is invalid") from exc
    if dtype is not None and not dtype.is_floating_point:
      raise ReplayValidationError("coverage replay dtype is not floating point")
    if self._device is not None and device is not None and self._device != device:
      raise ReplayValidationError("coverage replay device does not match")
    if self._dtype is not None and dtype is not None and self._dtype != dtype:
      raise ReplayValidationError("coverage replay dtype does not match")
    if (
      not isinstance(state["rng_state"], torch.Tensor)
      or state["rng_state"].dtype != torch.uint8
    ):
      raise ReplayValidationError("coverage replay RNG state is invalid")
    try:
      rng_candidate = torch.Generator(device="cpu")
      rng_candidate.set_state(state["rng_state"])
      validated_rng_state = rng_candidate.get_state().clone()
    except (RuntimeError, TypeError, ValueError) as exc:
      raise ReplayValidationError("coverage replay RNG state is invalid") from exc
    recent: dict[int, _RecentPartition] = {
      motion: _RecentPartition(self.recent_quotas[motion], self.schema)
      for motion in self.motion_quotas
    }
    recent_row_ids: dict[str, _RecentRowRef] = {}
    archive_row_ids: dict[str, _StoredRow] = {}
    if not isinstance(state["recent"], Mapping):
      raise ReplayValidationError("coverage replay recent state is invalid")
    if set(state["recent"]) != set(recent):
      raise ReplayValidationError("coverage replay recent motion layout is invalid")
    for raw_motion, raw_partition in state["recent"].items():
      if (
        isinstance(raw_motion, bool)
        or not isinstance(raw_motion, int)
        or raw_motion not in recent
        or not isinstance(raw_partition, Mapping)
      ):
        raise ReplayValidationError("coverage replay recent partition is invalid")
      nested_provenance = raw_partition.get("layout") == "provenance-v1"
      if nested_provenance != state_provenance:
        raise ReplayValidationError("coverage replay nested layout is invalid")
      recent[raw_motion].load_state(raw_partition, device, dtype)
      partition = recent[raw_motion]
      if (partition.buffer._provenance_enabled is True) != state_provenance:
        raise ReplayValidationError("coverage replay nested layout does not match")
      storage = partition.buffer._storage
      for index, metadata in partition.active_rows():
        if storage is None:
          raise ReplayValidationError("nonempty recent partition has no storage")
        self._validate_recent_storage_row(storage, index, metadata, raw_motion)
        if metadata.row_id in recent_row_ids:
          raise ReplayValidationError("duplicate recent replay row ID")
        recent_row_ids[metadata.row_id] = _RecentRowRef(raw_motion, index, metadata)
    cells: dict[tuple[int, tuple[str, int, str]], dict[str, _FamilyBucket]] = {}
    if not isinstance(state["cells"], Sequence):
      raise ReplayValidationError("coverage replay archive state is invalid")
    for raw_cell in state["cells"]:
      if not isinstance(raw_cell, Mapping) or set(raw_cell) != {
        "motion",
        "cell",
        "families",
      }:
        raise ReplayValidationError("coverage replay cell state is invalid")
      raw_motion = raw_cell["motion"]
      if isinstance(raw_motion, bool) or not isinstance(raw_motion, int):
        raise ReplayValidationError("coverage replay cell motion is invalid")
      motion = raw_motion
      raw_key = raw_cell["cell"]
      if (
        not isinstance(raw_key, Sequence)
        or isinstance(raw_key, (str, bytes))
        or len(raw_key) != 3
        or not isinstance(raw_key[0], str)
        or isinstance(raw_key[1], bool)
        or not isinstance(raw_key[1], int)
        or not isinstance(raw_key[2], str)
      ):
        raise ReplayValidationError("coverage replay cell key is invalid")
      cell = (raw_key[0], raw_key[1], raw_key[2])
      if motion not in self.motion_quotas:
        raise ReplayValidationError("coverage replay cell motion is invalid")
      key = (motion, cell)
      if (
        self._cell_quotas.get(key, 0) <= 0
        or not isinstance(raw_cell["families"], Mapping)
        or not raw_cell["families"]
      ):
        raise ReplayValidationError("coverage replay sparse cell is invalid")
      if key in cells:
        raise ReplayValidationError("duplicate archive cell key")
      families: dict[str, _FamilyBucket] = {}
      max_families = max(1, self._cell_quotas[key] // _MAX_ROWS_PER_FAMILY_CELL)
      if len(raw_cell["families"]) > max_families:
        raise ReplayValidationError("coverage replay family reservoir is oversized")
      for family, raw_bucket in raw_cell["families"].items():
        if (
          not isinstance(family, str)
          or not isinstance(raw_bucket, Mapping)
          or set(raw_bucket) != {"seen", "priority", "rows"}
        ):
          raise ReplayValidationError("coverage replay family bucket is invalid")
        rows = []
        if not isinstance(raw_bucket["rows"], Sequence) or isinstance(
          raw_bucket["rows"], (str, bytes)
        ):
          raise ReplayValidationError("coverage replay row reservoir is invalid")
        for raw in raw_bucket["rows"]:
          row = self._decode_row(
            raw, device, dtype, provenance_enabled=state_provenance
          )
          self._validate_decoded_row(row, motion)
          if row.metadata.family_id != family or _cell_key(row.metadata) != cell:
            raise ReplayValidationError("archive row family or cell is inconsistent")
          if row.metadata.row_id in archive_row_ids:
            raise ReplayValidationError("duplicate archive replay row ID")
          archive_row_ids[row.metadata.row_id] = row
          rows.append(row)
        if len(rows) > min(_MAX_ROWS_PER_FAMILY_CELL, self._cell_quotas[key]):
          raise ReplayValidationError("coverage replay row reservoir is oversized")
        if not rows:
          raise ReplayValidationError("empty archive family bucket is invalid")
        if (
          isinstance(raw_bucket["seen"], bool)
          or not isinstance(raw_bucket["seen"], int)
          or raw_bucket["seen"] < len(rows)
        ):
          raise ReplayValidationError("coverage replay family metadata is invalid")
        priority = raw_bucket["priority"]
        if isinstance(priority, bool) or not isinstance(priority, int):
          raise ReplayValidationError("coverage replay family priority is invalid")
        if priority != self._family_priority(family, motion, cell):
          raise ReplayValidationError("coverage replay family priority is invalid")
        families[family] = _FamilyBucket(rows, int(raw_bucket["seen"]), priority)
      cells[key] = families
    for row_id in recent_row_ids.keys() & archive_row_ids.keys():
      recent_ref = recent_row_ids[row_id]
      archive_row = archive_row_ids[row_id]
      if recent_ref.metadata.as_dict() != archive_row.metadata.as_dict():
        raise ReplayValidationError("recent/archive row metadata conflicts")
      recent_storage = recent[recent_ref.motion].buffer._storage
      if recent_storage is None or set(recent_storage) != set(archive_row.fields):
        raise ReplayValidationError("recent/archive row fields conflict")
      if any(
        not torch.equal(recent_storage[name][recent_ref.index], value)
        for name, value in archive_row.fields.items()
      ):
        raise ReplayValidationError("recent/archive row tensors conflict")
    if any(
      sum(
        len(bucket.rows)
        for (cell_motion, _), families in cells.items()
        if cell_motion == motion
        for bucket in families.values()
      )
      > self.archive_quotas[motion]
      for motion in self.motion_quotas
    ):
      raise ReplayValidationError("coverage replay archive capacity is invalid")
    if state["dedup_capacity"] != self._dedup_capacity:
      raise ReplayValidationError("coverage replay dedup capacity does not match")
    raw_dedup = state["dedup_ids"]
    if (
      not isinstance(raw_dedup, Sequence)
      or len(raw_dedup) > self._dedup_capacity
      or len(set(raw_dedup)) != len(raw_dedup)
      or any(not isinstance(value, str) for value in raw_dedup)
    ):
      raise ReplayValidationError("coverage replay dedup IDs are invalid")
    if state["row_namespace"] != self._row_namespace:
      raise ReplayValidationError("coverage replay row namespace does not match")
    if state["family_salt"] != self._family_salt:
      raise ReplayValidationError("coverage replay family salt does not match")
    last_call = state["last_collection_call"]
    if last_call is not None and (
      isinstance(last_call, bool) or not isinstance(last_call, int) or last_call < 0
    ):
      raise ReplayValidationError("coverage replay collection call is invalid")
    raw_family_counts = state["family_motion_counts"]
    if not isinstance(raw_family_counts, Sequence):
      raise ReplayValidationError("coverage replay family counts are invalid")
    family_motion_counts: dict[tuple[int, str], int] = defaultdict(int)
    for record in raw_family_counts:
      if not isinstance(record, Mapping) or set(record) != {
        "motion",
        "family",
        "count",
      }:
        raise ReplayValidationError("coverage replay family counts are invalid")
      motion, family, count = record["motion"], record["family"], record["count"]
      if (
        isinstance(motion, bool)
        or not isinstance(motion, int)
        or not isinstance(family, str)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
      ):
        raise ReplayValidationError("coverage replay family counts are invalid")
      if not count:
        raise ReplayValidationError(
          "coverage replay family counts must omit zero entries"
        )
      if count > _MAX_ROWS_PER_FAMILY_MOTION:
        raise ReplayValidationError("coverage replay family count exceeds motion cap")
      family_motion_counts[(motion, family)] = count
    expected_family_counts: dict[tuple[int, str], int] = defaultdict(int)
    for (cell_motion, _), families in cells.items():
      for family, bucket in families.items():
        expected_family_counts[(cell_motion, family)] += len(bucket.rows)
    if dict(expected_family_counts) != dict(family_motion_counts):
      raise ReplayValidationError("coverage replay family counts do not match rows")
    counters = dict(state["counters"])
    if any(
      not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, int)
      for key, value in counters.items()
    ):
      raise ReplayValidationError("coverage replay counters are invalid")
    fresh_count, draw_count = state["fresh_count"], state["draw_count"]
    if (
      isinstance(fresh_count, bool)
      or not isinstance(fresh_count, int)
      or fresh_count < 0
      or isinstance(draw_count, bool)
      or not isinstance(draw_count, int)
      or draw_count < 0
    ):
      raise ReplayValidationError("coverage replay counters are invalid")
    return (
      recent,
      cells,
      dict.fromkeys(raw_dedup),
      family_motion_counts,
      last_call,
      device,
      dtype,
      fresh_count,
      draw_count,
      counters,
      validated_rng_state,
      state_provenance,
    )

  def _decode_row(
    self,
    raw: object,
    device: torch.device | None,
    dtype: torch.dtype | None,
    *,
    provenance_enabled: bool,
  ) -> _StoredRow:
    if not isinstance(raw, Mapping):
      raise ReplayValidationError("coverage replay row is invalid")
    raw_mapping = cast(Mapping[str, Any], raw)
    raw_fields = raw_mapping.get("fields")
    if set(raw_mapping) != {"fields", "metadata"} or not isinstance(
      raw_fields, Mapping
    ):
      raise ReplayValidationError("coverage replay row is invalid")
    metadata = CoverageRowMetadata.from_dict(raw_mapping["metadata"])
    if any(
      isinstance(value, bool)
      for value in (
        metadata.source_frame,
        metadata.phase_bin,
        metadata.branch_age_steps,
        metadata.event_frame,
      )
      if value is not None
    ):
      raise ReplayValidationError("coverage metadata integer fields are invalid")
    expected = {
      "reference",
      "conditioning",
      "teacher_action",
      "motion_id",
      "teacher_id",
      "reference_frame",
      "episode_id",
      "collector_iteration",
    }
    if provenance_enabled:
      expected.update(
        {"initialization_kind", "segment_initial_reference_frame", "segment_age"}
      )
    fields = dict(cast(Mapping[str, Any], raw_fields))
    if set(fields) != expected:
      raise ReplayValidationError("coverage replay row fields do not match layout")
    expected_widths = {
      "reference": self.schema.reference_dim,
      "conditioning": self.schema.conditioning_dim,
      "teacher_action": self.schema.action_dim,
    }
    result: dict[str, torch.Tensor] = {}
    row_device = device
    floating_dtype: torch.dtype | None = None
    for name, value in fields.items():
      if not isinstance(value, torch.Tensor):
        raise ReplayValidationError("coverage replay row tensor is invalid")
      if name in expected_widths:
        if (
          value.ndim != 1
          or value.shape[0] != expected_widths[name]
          or not value.is_floating_point()
          or not torch.isfinite(value).all().item()
        ):
          raise ReplayValidationError("coverage replay row floating tensor is invalid")
        if floating_dtype is None:
          floating_dtype = value.dtype
        elif value.dtype != floating_dtype:
          raise ReplayValidationError("coverage replay row floating dtypes differ")
        if dtype is not None and value.dtype != dtype:
          raise ReplayValidationError("coverage replay row dtype is invalid")
      elif value.ndim != 0 or value.dtype != torch.int64:
        raise ReplayValidationError("coverage replay row metadata tensor is invalid")
      if row_device is None:
        row_device = value.device
      elif value.device != row_device:
        raise ReplayValidationError("coverage replay row device fields differ")
      result[name] = value.detach().clone()
    row = _StoredRow(result, metadata)
    self._validate_decoded_row(row, int(result["motion_id"].item()))
    return row

  def _validate_recent_storage_row(
    self,
    storage: Mapping[str, torch.Tensor],
    index: int,
    metadata: CoverageRowMetadata,
    expected_motion: int,
  ) -> None:
    _validate_metadata_types(metadata)
    tensor_motion = int(storage["motion_id"][index].item())
    if tensor_motion != expected_motion or tensor_motion not in self._recent:
      raise ReplayValidationError("decoded replay row motion routing is invalid")
    if not metadata.motion_id.isdecimal() or int(metadata.motion_id) != tensor_motion:
      raise ReplayValidationError("decoded metadata motion routing is invalid")
    teacher_id = int(storage["teacher_id"][index].item())
    if teacher_id < 0:
      raise ReplayValidationError("decoded teacher ID is invalid")
    if (
      self.teacher_codes is not None and teacher_id != self.teacher_codes[tensor_motion]
    ):
      raise ReplayValidationError("decoded teacher routing is invalid")
    source_frame = int(storage["reference_frame"][index].item())
    if metadata.source_frame != source_frame:
      raise ReplayValidationError("decoded metadata frame is invalid")
    if (
      self.frame_counts is not None
      and not 0 <= source_frame < self.frame_counts[tensor_motion]
    ):
      raise ReplayValidationError("decoded reference frame is invalid")
    _cell_key(metadata)

  def _validate_decoded_row(self, row: _StoredRow, expected_motion: int) -> None:
    _validate_metadata_types(row.metadata)
    fields = row.fields
    tensor_motion = int(fields["motion_id"].item())
    if tensor_motion != expected_motion or tensor_motion not in self._recent:
      raise ReplayValidationError("decoded replay row motion routing is invalid")
    if (
      not row.metadata.motion_id.isdecimal()
      or int(row.metadata.motion_id) != tensor_motion
    ):
      raise ReplayValidationError("decoded metadata motion routing is invalid")
    teacher_id = int(fields["teacher_id"].item())
    if teacher_id < 0:
      raise ReplayValidationError("decoded teacher ID is invalid")
    if (
      self.teacher_codes is not None and teacher_id != self.teacher_codes[tensor_motion]
    ):
      raise ReplayValidationError("decoded teacher routing is invalid")
    source_frame = int(fields["reference_frame"].item())
    if row.metadata.source_frame != source_frame:
      raise ReplayValidationError("decoded metadata frame is invalid")
    if (
      self.frame_counts is not None
      and not 0 <= source_frame < self.frame_counts[tensor_motion]
    ):
      raise ReplayValidationError("decoded reference frame is invalid")
    _cell_key(row.metadata)

  def validate_state(self, state: Mapping[str, Any]) -> None:
    try:
      self._decode_state(state)
    except ReplayValidationError:
      raise
    except (
      AttributeError,
      CoverageMetadataError,
      RuntimeError,
      TypeError,
      ValueError,
    ) as exc:
      raise ReplayValidationError("coverage replay state is invalid") from exc

  def load_state_dict(self, state: Mapping[str, Any]) -> None:
    if not isinstance(state, Mapping):
      raise ReplayValidationError("coverage replay state must be a mapping")
    try:
      decoded = self._decode_state(state)
    except ReplayValidationError:
      raise
    except (
      AttributeError,
      CoverageMetadataError,
      RuntimeError,
      TypeError,
      ValueError,
    ) as exc:
      raise ReplayValidationError("coverage replay state is invalid") from exc
    (
      recent,
      cells,
      dedup_ids,
      family_motion_counts,
      last_call,
      device,
      dtype,
      fresh_count,
      draw_count,
      counters,
      rng_state,
      provenance_enabled,
    ) = decoded
    self._recent, self._cells = recent, cells
    self._dedup_ids = dedup_ids
    self._family_motion_counts = defaultdict(int, family_motion_counts)
    self._last_collection_call = last_call
    self._device, self._dtype = device, dtype
    self._provenance_enabled = provenance_enabled
    self._fresh_count, self._draw_count = fresh_count, draw_count
    self._counters = defaultdict(int, counters)
    self._rng.set_state(rng_state)

  def max_valid_segment_id(self) -> int | None:
    values = [
      value
      for partition in self._recent.values()
      if (value := partition.max_segment_id()) is not None
    ]
    values.extend(
      int(row.fields["episode_id"].item())
      for families in self._cells.values()
      for bucket in families.values()
      for row in bucket.rows
    )
    return max(values, default=None)

  def rebase_segment_ids(self, collector_iteration: int, base: int) -> int:
    if (
      not isinstance(collector_iteration, int) or not isinstance(base, int) or base < 0
    ):
      raise ReplayValidationError("segment rebase arguments are invalid")
    changed = sum(
      partition.rebase_segment_ids(collector_iteration, base)
      for partition in self._recent.values()
    )
    for families in self._cells.values():
      for bucket in families.values():
        for row in bucket.rows:
          if int(row.fields["collector_iteration"].item()) == collector_iteration:
            row.fields["episode_id"] += base
            changed += 1
    return changed


# Alias with a shorter integration name and a protocol-oriented name.
CoverageReplayBuffer = ProtectedCoverageReplayBuffer

__all__ = [
  "CoverageInsertReport",
  "CoverageReplayBatch",
  "CoverageReplayBuffer",
  "CoverageReplayNotReadyError",
  "CoverageReplayReport",
  "ProtectedCoverageReplayBuffer",
  "ReplayNotReadyError",
]
