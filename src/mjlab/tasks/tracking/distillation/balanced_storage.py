"""Balanced per-motion raw replay partitions for M4 multi-teacher distillation.

The M3 :class:`~mjlab.tasks.tracking.distillation.storage.LabeledReplayBuffer`
stays the default single-partition FIFO.  This module adds the opt-in M4 buffer:
one total capacity split into per-motion integer quotas, so a longer or easier
clip can never evict another clip's records, and minibatches are drawn in the
configured motion proportions instead of in whatever mixture one global ring
happens to hold.

Storage layout: one flat ring per field (``reference``, ``conditioning``,
``teacher_action``, and the int64 metadata), each of ``capacity`` rows, plus one
per-motion metadata block.  A motion owns the contiguous segment
``[start, start + quota)`` of every field and its own cursor wraps inside that
segment, so a draw still returns each motion's own rows in its own FIFO order.
One flat ring per field, instead of one tensor per motion per field, keeps the
number of gathers per draw independent of the motion count: a 3261-motion cohort
draws with the same handful of kernels as a two-motion cohort.

Contract:

* Quotas come from the configured motion weights, sum exactly to the requested
  capacity, and give every selected motion at least one slot.
* An insertion validates the whole incoming batch -- schema, device, dtype,
  motion routing, teacher mapping, and reference-frame ranges -- before any row
  is written.  Each motion is an independent FIFO ring, so an oversized
  insertion retains its own newest rows and never evicts another motion.
* A draw converts the configured weights into expected counts, assigns the
  residual rows by unbiased randomized selection among the largest fractional
  remainders, samples uniformly inside each motion's segment (with replacement
  by default), and then shuffles the mixture.  A batch of size one is therefore
  not pinned to the first motion.
* A draw raises :class:`ReplayNotReadyError` instead of silently omitting a
  motion whose segment is still empty.
* Inserts clone the caller's rows into owned storage and draws return fresh
  tensors that never alias the buffer.
* Segments, quotas, weights, ordered routing, counters, and the flat field
  rings round-trip through ``state_dict``/``load_state_dict``, which validates
  one complete candidate before mutating this buffer.  The buffer owns no
  sampler RNG: every draw uses the caller's explicit ``torch.Generator``, whose
  state the existing checkpoint RNG contract persists.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from mjlab.tasks.tracking.distillation.observations import PackedObservationBatch
from mjlab.tasks.tracking.distillation.storage import (
  LabeledReplayBatch,
  ReplayValidationError,
  _canonical_device,
  validate_replay_batch,
)
from mjlab.tasks.tracking.distillation.vae_config import VaeSchema

_STATE_KIND = "balanced-motion-replay"
_STATE_VERSION = 2
_FLOAT_FIELDS = frozenset({"reference", "conditioning", "teacher_action"})
_INITIALIZATION_NAMES = {0: "unknown", 1: "reference", 2: "standing"}
_INITIALIZATION_KINDS = len(_INITIALIZATION_NAMES)
"""Kind values accepted by ``validate_replay_batch``, and their report labels."""
_STATE_OPTIONAL_KEYS = frozenset(
  {"inserted_by_initialization", "drawn_by_initialization", "drawn_by_phase_bin"}
)
"""Checkpoint tables a writer may omit; an absent table restores as zeros."""


class ReplayNotReadyError(ReplayValidationError):
  """A requested draw cannot cover every selected motion yet."""


def _phase_cells(
  reference_frame: torch.Tensor, frame_count: int, phase_bins: int
) -> torch.Tensor:
  """Phase cell of each row: ``floor(frame * bins / frames)``.

  Uses exact integer arithmetic whenever ``frame * bins`` cannot overflow
  int64 (every physically realizable clip), and a clamped float64 fallback
  beyond that, so the result is provably inside ``[0, phase_bins)`` for any
  representable validated frame (float rounding can never push the last frame
  into cell ``bins``, and int64 wraparound is avoided).  Only the *non-empty*
  cells participate in a balanced draw.
  """
  if frame_count <= (2**63 - 1) // phase_bins:
    return (reference_frame * int(phase_bins)).div(
      int(frame_count), rounding_mode="floor"
    )
  scaled = reference_frame.to(dtype=torch.float64) * (
    float(phase_bins) / float(frame_count)
  )
  return scaled.floor().clamp(max=phase_bins - 1).to(dtype=torch.int64)


def _split_count_over_cells(
  count: int,
  cell_sizes: torch.Tensor,
  *,
  generator: torch.Generator | None,
  random_device: torch.device,
) -> torch.Tensor:
  """Allocate ``count`` rows over phase cells without replacement.

  The residual goes to ``leftover`` *distinct* cells drawn from a permutation,
  so a cell receives at most one extra row; combined with the exact preflight in
  :meth:`BalancedReplayBuffer._validate_without_replacement_cells` this makes
  every per-cell request satisfiable by construction.  The with-replacement
  allocation needs no such helper: it is one per-row draw over the non-empty
  cells and stays vectorised in
  :meth:`BalancedReplayBuffer._balanced_phase_rows`.
  """
  cell_count = int(cell_sizes.numel())
  counts = torch.full((cell_count,), count // cell_count, dtype=torch.int64)
  leftover = count - (count // cell_count) * cell_count
  if leftover:
    order = torch.randperm(cell_count, device=random_device, generator=generator)
    extra = [0] * cell_count
    for index in order[:leftover].tolist():
      extra[int(index)] = 1
    counts = counts + torch.tensor(extra, dtype=torch.int64)
  return counts


def _storage_layout(
  schema: VaeSchema, provenance_enabled: bool = False
) -> tuple[tuple[str, int], ...]:
  """Ordered ``(field, row width)`` pairs; width ``0`` marks a 1-D field."""
  layout = (
    ("reference", schema.reference_dim),
    ("conditioning", schema.conditioning_dim),
    ("teacher_action", schema.action_dim),
    ("motion_id", 0),
    ("teacher_id", 0),
    ("reference_frame", 0),
    ("episode_id", 0),
    ("collector_iteration", 0),
  )
  if provenance_enabled:
    layout += (
      ("initialization_kind", 0),
      ("segment_initial_reference_frame", 0),
      ("segment_age", 0),
    )
  return layout


def _random_device(generator: torch.Generator | None) -> torch.device:
  """Device used for index draws, matching the single-partition replay buffer."""
  return (
    torch.device(generator.device) if generator is not None else torch.device("cpu")
  )


def _uniform_offsets(
  size: torch.Tensor,
  *,
  generator: torch.Generator | None,
  random_device: torch.device,
) -> torch.Tensor:
  """One uniform draw in ``[0, size)`` per row, clamped into range.

  ``size`` holds the row count available to each output row, so one draw serves
  every motion in a batch.  A float64 uniform is exact for every representable
  segment size (``size < 2**53``) and the result is clamped defensively, so a
  rounding error can never index past a segment's last row.
  """
  uniform = torch.rand(
    size.numel(), dtype=torch.float64, generator=generator, device=random_device
  )
  scaled = uniform.to(size.device) * size.to(torch.float64)
  return torch.minimum(scaled.floor().to(torch.int64), size - 1)


def _require_int(name: str, value: object, *, minimum: int | None = None) -> int:
  if isinstance(value, bool) or not isinstance(value, int):
    raise ReplayValidationError(f"{name} must be an integer")
  if minimum is not None and value < minimum:
    raise ReplayValidationError(f"{name} must be at least {minimum}")
  return value


@dataclass(frozen=True, slots=True)
class MotionCapacityQuota:
  """Requested motion weights and the integer capacity quota each received."""

  motion_ids: tuple[int, ...]
  weights: tuple[float, ...]
  quotas: tuple[int, ...]
  total_capacity: int

  def __post_init__(self) -> None:
    size = len(self.motion_ids)
    if size == 0:
      raise ReplayValidationError("a capacity quota needs at least one motion")
    if len(self.weights) != size or len(self.quotas) != size:
      raise ReplayValidationError("quota weights and allocations must align")
    if len(set(self.motion_ids)) != size:
      raise ReplayValidationError("quota motion ids must be unique")
    for motion_id in self.motion_ids:
      if isinstance(motion_id, bool) or not isinstance(motion_id, int) or motion_id < 0:
        raise ReplayValidationError(
          f"quota motion ids must be non-negative integers, got {motion_id!r}"
        )
    for weight in self.weights:
      if not math.isfinite(weight) or weight <= 0.0:
        raise ReplayValidationError(
          f"quota weights must be finite and positive, got {weight!r}"
        )
    if any(quota < 1 for quota in self.quotas):
      raise ReplayValidationError(
        f"every selected motion needs at least one replay slot, got {self.quotas}"
      )
    if sum(self.quotas) != self.total_capacity:
      raise ReplayValidationError(
        f"quotas {self.quotas} do not fill capacity {self.total_capacity}"
      )

  @property
  def motion_count(self) -> int:
    return len(self.motion_ids)

  @property
  def fractions(self) -> tuple[float, ...]:
    """Realized capacity fraction per motion, aligned with ``motion_ids``."""
    return tuple(quota / self.total_capacity for quota in self.quotas)

  def quota_for(self, motion_id: int) -> int:
    try:
      return self.quotas[self.motion_ids.index(motion_id)]
    except ValueError as exc:
      raise ReplayValidationError(
        f"motion {motion_id} has no replay capacity quota"
      ) from exc

  def as_dict(self) -> dict[str, Any]:
    """Plain-data form for provenance records and run reports."""
    return {
      "motion_ids": list(self.motion_ids),
      "weights": list(self.weights),
      "quotas": list(self.quotas),
      "fractions": list(self.fractions),
      "total_capacity": self.total_capacity,
    }


def allocate_motion_capacity_quotas(
  weights: Mapping[int, float], capacity: int
) -> MotionCapacityQuota:
  """Split one total capacity into per-motion integer replay quotas.

  One slot is reserved for every motion and the remainder is distributed by
  deterministic largest-remainder allocation, so quotas sum exactly to
  ``capacity``, every selected motion receives at least one, and ties break
  toward the larger weight and then the lower motion id.  A capacity smaller
  than the number of selected motions is impossible and is refused rather than
  silently reweighted.
  """
  if not isinstance(weights, Mapping) or not weights:
    raise ReplayValidationError("motion weights must be a non-empty mapping")
  motion_ids = tuple(sorted(weights))
  for motion_id in motion_ids:
    if isinstance(motion_id, bool) or not isinstance(motion_id, int) or motion_id < 0:
      raise ReplayValidationError(
        f"motion ids must be non-negative integers, got {motion_id!r}"
      )
  values = tuple(float(weights[motion_id]) for motion_id in motion_ids)
  for motion_id, weight in zip(motion_ids, values, strict=True):
    if not math.isfinite(weight) or weight <= 0.0:
      raise ReplayValidationError(
        f"weight for motion {motion_id} must be finite and positive, got {weight!r}"
      )
  _require_int("capacity", capacity, minimum=1)
  size = len(motion_ids)
  if capacity < size:
    raise ReplayValidationError(
      f"capacity {capacity} cannot give each of {size} selected motions a replay slot"
    )

  remaining = capacity - size
  quotas = [1] * size
  if remaining:
    total_weight = math.fsum(values)
    exact = [remaining * weight / total_weight for weight in values]
    extra = [int(math.floor(value)) for value in exact]
    leftover = remaining - sum(extra)
    remainders = [value - base for value, base in zip(exact, extra, strict=True)]
    order = sorted(
      range(size),
      key=lambda index: (-remainders[index], -values[index], motion_ids[index]),
    )
    for index in order[:leftover]:
      extra[index] += 1
    quotas = [1 + value for value in extra]
  return MotionCapacityQuota(
    motion_ids=motion_ids,
    weights=values,
    quotas=tuple(quotas),
    total_capacity=capacity,
  )


@dataclass(frozen=True, slots=True)
class MotionReplayStats:
  """Per-motion capacity, occupancy, and sample counters."""

  motion_id: int
  weight: float
  quota: int
  retained: int
  inserted: int
  drawn: int
  inserted_by_initialization: dict[str, int] = field(default_factory=dict)
  drawn_by_initialization: dict[str, int] = field(default_factory=dict)
  retained_by_initialization: dict[str, int] = field(default_factory=dict)
  retained_by_phase_bin: tuple[int, ...] | None = None
  drawn_by_phase_bin: tuple[int, ...] | None = None

  @property
  def coverage(self) -> float:
    """Retained fraction of this motion's capacity quota."""
    return self.retained / self.quota

  @property
  def ready(self) -> bool:
    return self.retained > 0

  def as_dict(self) -> dict[str, Any]:
    result = {
      "motion_id": self.motion_id,
      "weight": self.weight,
      "quota": self.quota,
      "retained": self.retained,
      "inserted": self.inserted,
      "drawn": self.drawn,
      "inserted_by_initialization": dict(self.inserted_by_initialization),
      "drawn_by_initialization": dict(self.drawn_by_initialization),
      "retained_by_initialization": dict(self.retained_by_initialization),
      "coverage": self.coverage,
    }
    if self.retained_by_phase_bin is not None:
      result["retained_by_phase_bin"] = list(self.retained_by_phase_bin)
    if self.drawn_by_phase_bin is not None:
      result["drawn_by_phase_bin"] = list(self.drawn_by_phase_bin)
    return result


@dataclass(frozen=True, slots=True)
class BalancedReplayReport:
  """Capacity/occupancy and per-motion coverage for one balanced buffer."""

  capacity: int
  retained: int
  inserted: int
  drawn: int
  ready: bool
  unready_motion_ids: tuple[int, ...]
  motions: tuple[MotionReplayStats, ...]
  coverage_sampling: bool = False
  coverage_drawn: int = 0

  @property
  def occupancy(self) -> float:
    """Retained fraction of the whole capacity."""
    return self.retained / self.capacity

  def as_dict(self) -> dict[str, Any]:
    result: dict[str, Any] = {
      "capacity": self.capacity,
      "retained": self.retained,
      "inserted": self.inserted,
      "drawn": self.drawn,
      "occupancy": self.occupancy,
      "ready": self.ready,
      "unready_motion_ids": list(self.unready_motion_ids),
      "motions": [motion.as_dict() for motion in self.motions],
    }
    if self.coverage_sampling:
      result.update(
        {
          "coverage_sampling": True,
          "coverage_drawn": self.coverage_drawn,
        }
      )
    return result


@dataclass(slots=True)
class _MotionSegments:
  """Per-motion segment metadata, held as tensors indexed by motion slot.

  ``start[m]`` is the first global row of motion ``m``'s segment, so the motion
  occupies ``start[m] + local`` for the local ring position ``local`` in
  ``[0, quota[m])`` and its cursor ``next[m]`` wraps inside that segment only.
  Every table is device-side so insertion and sampling can convert a draw into
  global row indices without a Python loop over motions.
  """

  start: torch.Tensor
  quota: torch.Tensor
  size: torch.Tensor
  next: torch.Tensor
  inserted: torch.Tensor
  drawn: torch.Tensor
  inserted_by_initialization: torch.Tensor | None = None
  drawn_by_initialization: torch.Tensor | None = None
  drawn_by_phase_bin: torch.Tensor | None = None

  @property
  def device(self) -> torch.device:
    return self.start.device

  def to(self, device: torch.device) -> _MotionSegments:
    """Copy every table to ``device``."""

    def moved(value: torch.Tensor | None) -> torch.Tensor | None:
      return None if value is None else value.to(device)

    return _MotionSegments(
      start=self.start.to(device),
      quota=self.quota.to(device),
      size=self.size.to(device),
      next=self.next.to(device),
      inserted=self.inserted.to(device),
      drawn=self.drawn.to(device),
      inserted_by_initialization=moved(self.inserted_by_initialization),
      drawn_by_initialization=moved(self.drawn_by_initialization),
      drawn_by_phase_bin=moved(self.drawn_by_phase_bin),
    )


@dataclass(slots=True)
class _RowGrouping:
  """Vectorised split of one validated batch into per-motion row runs.

  ``source`` and ``destination`` are aligned and unique, so a single index
  assignment per field commits the whole insertion: ``source`` selects rows of
  the caller's batch (in write order) and ``destination`` is the global replay
  row each one lands in.  ``counts`` is the incoming row count per motion and
  ``kinds`` the per-motion initialization-kind histogram, both for the counters.
  """

  counts: torch.Tensor
  source: torch.Tensor
  destination: torch.Tensor
  kinds: torch.Tensor | None = None


@dataclass(slots=True)
class _PhaseGroups:
  """Retained rows regrouped by ``(motion slot, phase cell)``.

  Built from the flat rings after an insertion and reused by every draw, so a
  phase-balanced sample never sorts or partitions per motion.  ``group_order``
  is a permutation of live-row positions ordered by key, and the members of key
  ``slot * phase_bins + cell`` are the slice that starts at ``group_start`` and
  is ``cell_sizes[slot, cell]`` long.
  """

  live_rows: torch.Tensor
  slots: torch.Tensor
  cells: torch.Tensor
  cell_sizes: torch.Tensor
  nonempty_cells: torch.Tensor
  nonempty_counts: torch.Tensor
  group_order: torch.Tensor
  group_start: torch.Tensor


@dataclass(slots=True)
class _RowIndex:
  """Rows of one retained subset, grouped by motion slot in ascending order."""

  rows: torch.Tensor
  first: torch.Tensor
  counts: torch.Tensor


def _assemble_batch(
  fields: Mapping[str, torch.Tensor], schema: VaeSchema
) -> LabeledReplayBatch:
  """Assemble one labeled batch from already-selected owned field tensors."""
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
    **(
      {
        name: fields[name]
        for name in (
          "initialization_kind",
          "segment_initial_reference_frame",
          "segment_age",
        )
      }
      if "initialization_kind" in fields
      else {}
    ),
  )


def _kind_counts(names: list[str], counts: list[int]) -> dict[str, int]:
  """Report one motion's nonzero initialization-kind counts by label."""
  return {
    names[kind]: counts[kind] for kind in range(_INITIALIZATION_KINDS) if counts[kind]
  }


class BalancedReplayBuffer:
  """Capacity-bounded replay split into per-motion FIFO segments.

  ``weights`` are the configured motion proportions; they determine both the
  per-motion capacity quotas and the minibatch mixture.  ``teacher_codes`` and
  ``frame_counts`` are the authoritative motion-to-teacher-code mapping and
  per-motion reference length used to validate every insertion, so a row can
  never be stored under the wrong teacher or an impossible reference frame.
  Motion ids are ordered ascending for deterministic allocation and tie breaks.

  ``phase_bins`` (0, the default, disables phase balancing) splits each clip's
  ``[0, 1)`` reference-phase axis into that many cells and equalizes a draw's
  within-motion exposure over the non-empty cells; the default reproduces the
  historical density-proportional draw exactly.
  """

  def __init__(
    self,
    capacity: int,
    schema: VaeSchema,
    weights: Mapping[int, float],
    *,
    teacher_codes: Mapping[int, int],
    frame_counts: Mapping[int, int],
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    provenance_enabled: bool | None = None,
    phase_bins: int = 0,
    coverage_sampling: bool = False,
  ) -> None:
    if not isinstance(schema, VaeSchema):
      raise ReplayValidationError("schema must be a VaeSchema")
    if dtype is not None and not dtype.is_floating_point:
      raise ReplayValidationError("dtype must be a floating-point torch dtype")
    quota = allocate_motion_capacity_quotas(weights, capacity)
    motion_ids = quota.motion_ids
    self._quota = quota
    self._motion_ids = motion_ids
    self._weights = quota.weights
    self._teacher_codes = self._routing_values(
      "teacher code", teacher_codes, motion_ids
    )
    self._frame_counts = self._routing_values("frame count", frame_counts, motion_ids)
    for motion_id, code in zip(motion_ids, self._teacher_codes, strict=True):
      if code < 0:
        raise ReplayValidationError(
          f"teacher code for motion {motion_id} must be non-negative, got {code}"
        )
    if len(set(self._teacher_codes)) != len(self._teacher_codes):
      raise ReplayValidationError(
        f"teacher codes must be distinct per motion, got {self._teacher_codes}"
      )
    for motion_id, frames in zip(motion_ids, self._frame_counts, strict=True):
      if frames < 1:
        raise ReplayValidationError(
          f"frame count for motion {motion_id} must be positive, got {frames}"
        )
    if (
      isinstance(phase_bins, bool) or not isinstance(phase_bins, int) or phase_bins < 0
    ):
      raise ReplayValidationError(
        f"phase_bins must be a non-negative integer, got {phase_bins!r}"
      )
    self._phase_bins = int(phase_bins)
    self._exact_cell_arithmetic = max(self._frame_counts) <= (2**63 - 1) // max(
      1, self._phase_bins
    )
    self._schema = schema
    self._capacity = quota.total_capacity
    self._device = _canonical_device(device) if device is not None else None
    self._dtype = dtype
    self._size = 0
    self._provenance_enabled = provenance_enabled
    self._coverage_sampling = bool(coverage_sampling)
    self._coverage_drawn = 0
    self._layout = _storage_layout(schema, bool(provenance_enabled))
    self._fields: dict[str, torch.Tensor] | None = None
    self._phase_groups_cache: _PhaseGroups | None = None
    self._unknown_rows_cache: _RowIndex | None = None
    self._meta = self._new_segments()
    self._bind_device_tables()

  def _new_segments(self) -> _MotionSegments:
    """Segment metadata for the configured quotas, on the current device."""
    device = self._device if self._device is not None else torch.device("cpu")
    quota = torch.tensor(self._quota.quotas, dtype=torch.int64, device=device)
    start = torch.cat(
      [torch.zeros(1, dtype=torch.int64, device=device), quota.cumsum(0)]
    )
    motions = len(self._motion_ids)
    zero = torch.zeros(motions, dtype=torch.int64, device=device)
    provenance = bool(self._provenance_enabled)
    return _MotionSegments(
      start=start[:-1],
      quota=quota,
      size=zero.clone(),
      next=zero.clone(),
      inserted=zero.clone(),
      drawn=zero.clone(),
      inserted_by_initialization=(
        torch.zeros((motions, _INITIALIZATION_KINDS), dtype=torch.int64, device=device)
        if provenance
        else None
      ),
      drawn_by_initialization=(
        torch.zeros((motions, _INITIALIZATION_KINDS), dtype=torch.int64, device=device)
        if provenance
        else None
      ),
      drawn_by_phase_bin=(
        torch.zeros((motions, self._phase_bins), dtype=torch.int64, device=device)
        if self._phase_bins
        else None
      ),
    )

  def _bind_device_tables(self) -> None:
    """Index tables used by the vectorised paths, on the segment device."""
    device = self._meta.device
    self._slot_ids = torch.arange(
      len(self._motion_ids), dtype=torch.int64, device=device
    )
    self._frame_count_tensor = torch.tensor(
      self._frame_counts, dtype=torch.int64, device=device
    )

  def _invalidate_derived(self) -> None:
    """Drop the cached regroupings after any change to the retained rows."""
    self._phase_groups_cache = None
    self._unknown_rows_cache = None

  @staticmethod
  def _routing_values(
    name: str, values: Mapping[int, int], motion_ids: tuple[int, ...]
  ) -> tuple[int, ...]:
    """Validate one motion-id-keyed routing table against the motion set."""
    if not isinstance(values, Mapping):
      raise ReplayValidationError(f"{name} table must be a mapping")
    if set(values) != set(motion_ids):
      raise ReplayValidationError(
        f"{name} table keys {sorted(values)} do not cover the selected motions "
        f"{list(motion_ids)}"
      )
    resolved = []
    for motion_id in motion_ids:
      value = values[motion_id]
      if isinstance(value, bool) or not isinstance(value, int):
        raise ReplayValidationError(
          f"{name} for motion {motion_id} must be an integer, got {value!r}"
        )
      resolved.append(value)
    return tuple(resolved)

  # Reporting and identity.

  @property
  def schema(self) -> VaeSchema:
    return self._schema

  @property
  def capacity(self) -> int:
    return self._capacity

  @property
  def size(self) -> int:
    return self._size

  @property
  def device(self) -> torch.device | None:
    return self._device

  @property
  def dtype(self) -> torch.dtype | None:
    return self._dtype

  @property
  def is_empty(self) -> bool:
    return self._size == 0

  def __len__(self) -> int:
    return self._size

  @property
  def quota(self) -> MotionCapacityQuota:
    return self._quota

  @property
  def motion_ids(self) -> tuple[int, ...]:
    return self._motion_ids

  @property
  def weights(self) -> tuple[float, ...]:
    return self._weights

  @property
  def teacher_codes(self) -> tuple[int, ...]:
    """Ordered motion-to-teacher-code mapping, aligned with ``motion_ids``."""
    return self._teacher_codes

  @property
  def frame_counts(self) -> tuple[int, ...]:
    """Ordered per-motion reference length, aligned with ``motion_ids``."""
    return self._frame_counts

  @property
  def phase_bins(self) -> int:
    """Configured within-motion phase-balancing cells; 0 is disabled."""
    return self._phase_bins

  @property
  def coverage_sampling(self) -> bool:
    """Whether one unknown/protected MVP row is forced into each draw."""
    return self._coverage_sampling

  @property
  def coverage_drawn(self) -> int:
    """Count MVP coverage rows included in replay draws."""
    return self._coverage_drawn

  @property
  def unready_motion_ids(self) -> tuple[int, ...]:
    """Selected motions whose segment holds no retained record yet."""
    if self._size == 0:
      return self._motion_ids
    empty = self._meta.size == 0
    if not bool(empty.any().item()):
      return ()
    return tuple(self._motion_ids[slot] for slot in empty.nonzero().flatten().tolist())

  @property
  def ready(self) -> bool:
    """Whether every selected motion has at least one retained record."""
    return not self.unready_motion_ids

  def retained(self, motion_id: int) -> int:
    """Retained record count for one selected motion."""
    return int(self._meta.size[self._slot(motion_id)].item())

  def _slot(self, motion_id: int) -> int:
    try:
      return self._motion_ids.index(motion_id)
    except ValueError as exc:
      raise ReplayValidationError(
        f"motion {motion_id} has no replay partition"
      ) from exc

  def _field(self, name: str) -> torch.Tensor:
    """One flat field ring; established by the first non-empty insertion."""
    assert self._fields is not None
    return self._fields[name]

  def _live_rows(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Global row index and motion slot of every retained record.

    Rows are ordered motion-major and oldest-first inside a motion, which is
    exactly the order a segment's ring defines.  Every derived table and the
    persistence snapshot build on this one place, so they agree by construction.
    """
    meta = self._meta
    device = meta.device
    slots = torch.repeat_interleave(self._slot_ids, meta.size)
    rank = (
      torch.arange(slots.numel(), dtype=torch.int64, device=device)
      - (meta.size.cumsum(0) - meta.size)[slots]
    )
    local = (meta.next[slots] - meta.size[slots] + rank) % meta.quota[slots]
    return meta.start[slots] + local, slots

  def _cells_of(self, frames: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Phase cell per row for a possibly heterogeneous set of clips.

    :func:`_phase_cells` takes one clip length, so a mixed cohort needs the same
    floor arithmetic per clip.  The integer form is used whenever every clip is
    below the int64 product bound (``max_frames * bins <= int64 max``), which is
    decided once at construction; the float64 fallback then matches
    :func:`_phase_cells`.
    """
    bins = self._phase_bins
    limits = self._frame_count_tensor[slots]
    if self._exact_cell_arithmetic:
      return (frames * int(bins)).div(limits, rounding_mode="floor")
    scaled = frames.to(dtype=torch.float64) * (
      float(bins) / limits.to(dtype=torch.float64)
    )
    return scaled.floor().clamp(max=bins - 1).to(dtype=torch.int64)

  def _retained_kind_counts(self) -> torch.Tensor | None:
    """Retained rows per motion and initialization kind, ``[motions, kinds]``."""
    if self._meta.inserted_by_initialization is None or self._size == 0:
      return None
    rows, slots = self._live_rows()
    kinds = self._field("initialization_kind")[rows]
    keys = slots * _INITIALIZATION_KINDS + kinds
    return torch.bincount(
      keys, minlength=len(self._motion_ids) * _INITIALIZATION_KINDS
    ).view(len(self._motion_ids), _INITIALIZATION_KINDS)

  def _phase_groups(self) -> _PhaseGroups:
    """Regroup the retained rows by ``(motion, phase cell)``, cached per insert.

    One stable sort over the retained rows builds the grouping, so a
    phase-balanced draw needs no per-motion loop and no per-draw sort.
    """
    if self._phase_groups_cache is not None:
      return self._phase_groups_cache
    motions = len(self._motion_ids)
    bins = self._phase_bins
    live_rows, slots = self._live_rows()
    cells = (
      self._cells_of(self._field("reference_frame")[live_rows], slots)
      if live_rows.numel()
      else live_rows
    )
    keys = slots * bins + cells
    cell_sizes = torch.bincount(keys, minlength=motions * bins).view(motions, bins)
    flat_sizes = cell_sizes.reshape(-1)
    group_start = flat_sizes.cumsum(0) - flat_sizes
    order = torch.argsort(keys, stable=True)
    nonempty = cell_sizes > 0
    # Non-empty cells first, in ascending cell order, so column ``k`` of
    # ``nonempty_cells`` is the k-th cell that can supply a row.
    ranked = torch.argsort((~nonempty).to(torch.int8), dim=1, stable=True)
    nonempty_cells = torch.where(
      torch.gather(nonempty, 1, ranked), ranked, torch.zeros_like(ranked)
    )
    groups = _PhaseGroups(
      live_rows=live_rows,
      slots=slots,
      cells=cells,
      cell_sizes=cell_sizes,
      nonempty_cells=nonempty_cells,
      nonempty_counts=nonempty.sum(1),
      group_order=order,
      group_start=group_start,
    )
    self._phase_groups_cache = groups
    return groups

  def _unknown_rows(self) -> _RowIndex:
    """Retained unknown-initialization rows, grouped by motion slot."""
    if self._unknown_rows_cache is not None:
      return self._unknown_rows_cache
    rows, slots = self._live_rows()
    unknown = self._field("initialization_kind")[rows] == 0
    selected = rows[unknown]
    selected_slots = slots[unknown]
    counts = torch.bincount(selected_slots, minlength=len(self._motion_ids))
    index = _RowIndex(
      rows=selected,
      first=counts.cumsum(0) - counts,
      counts=counts,
    )
    self._unknown_rows_cache = index
    return index

  def _read_meta(self) -> dict[str, list[Any]]:
    """Read every per-motion table with ONE host read.

    One ``tolist`` call carries the sizes, quotas, counters, kind tallies, and
    phase counters, so reporting a fully populated buffer costs one host
    synchronisation whatever the motion count is.
    """
    meta = self._meta
    motions = len(self._motion_ids)
    blocks: list[tuple[str, torch.Tensor, int]] = [
      ("size", meta.size, 1),
      ("quota", meta.quota, 1),
      ("inserted", meta.inserted, 1),
      ("drawn", meta.drawn, 1),
    ]
    if meta.inserted_by_initialization is not None:
      inserted_by_kind = meta.inserted_by_initialization
      drawn_by_kind = meta.drawn_by_initialization
      assert drawn_by_kind is not None
      retained = self._retained_kind_counts()
      if retained is None:
        retained = torch.zeros(
          (motions, _INITIALIZATION_KINDS), dtype=torch.int64, device=meta.device
        )
      blocks += [
        ("inserted_by_initialization", inserted_by_kind, 2),
        ("drawn_by_initialization", drawn_by_kind, 2),
        ("retained_by_initialization", retained, 2),
      ]
    if meta.drawn_by_phase_bin is not None:
      blocks += [
        ("retained_by_phase_bin", self._phase_groups().cell_sizes, 2),
        ("drawn_by_phase_bin", meta.drawn_by_phase_bin, 2),
      ]
    hosted = torch.cat([value.reshape(-1) for _, value, _ in blocks]).tolist()
    tables: dict[str, list[Any]] = {}
    offset = 0
    for name, value, rank in blocks:
      count = value.numel()
      flat = hosted[offset : offset + count]
      offset += count
      if rank == 1:
        tables[name] = flat
      else:
        width = count // motions
        tables[name] = [flat[row * width : (row + 1) * width] for row in range(motions)]
    return tables

  def stats(self) -> tuple[MotionReplayStats, ...]:
    """Per-motion capacity, occupancy, and sample counters."""
    return self._report(self._read_meta(), self._size).motions

  def report(self) -> BalancedReplayReport:
    """Capacity, occupancy, and per-motion coverage/counters."""
    return self._report(self._read_meta(), self._size)

  def _report(
    self, tables: dict[str, list[Any]], retained: int
  ) -> BalancedReplayReport:
    """Assemble one report from a single per-motion readout."""
    unready = tuple(
      self._motion_ids[slot] for slot, size in enumerate(tables["size"]) if size == 0
    )
    result: list[MotionReplayStats] = []
    for slot, motion_id in enumerate(self._motion_ids):
      inserted_by_initialization = (
        _kind_counts(
          list(_INITIALIZATION_NAMES.values()),
          tables["inserted_by_initialization"][slot],
        )
        if "inserted_by_initialization" in tables
        else {}
      )
      drawn_by_initialization = (
        _kind_counts(
          list(_INITIALIZATION_NAMES.values()),
          tables["drawn_by_initialization"][slot],
        )
        if "drawn_by_initialization" in tables
        else {}
      )
      retained_by_initialization = (
        _kind_counts(
          list(_INITIALIZATION_NAMES.values()),
          tables["retained_by_initialization"][slot],
        )
        if "retained_by_initialization" in tables
        else {}
      )
      result.append(
        MotionReplayStats(
          motion_id=motion_id,
          weight=self._weights[slot],
          quota=tables["quota"][slot],
          retained=tables["size"][slot],
          inserted=tables["inserted"][slot],
          drawn=tables["drawn"][slot],
          inserted_by_initialization=inserted_by_initialization,
          drawn_by_initialization=drawn_by_initialization,
          retained_by_initialization=retained_by_initialization,
          retained_by_phase_bin=(
            tuple(tables["retained_by_phase_bin"][slot])
            if "retained_by_phase_bin" in tables
            else None
          ),
          drawn_by_phase_bin=(
            tuple(tables["drawn_by_phase_bin"][slot])
            if "drawn_by_phase_bin" in tables
            else None
          ),
        )
      )
    motions = tuple(result)
    return BalancedReplayReport(
      capacity=self.capacity,
      retained=retained,
      inserted=sum(motion.inserted for motion in motions),
      drawn=sum(motion.drawn for motion in motions),
      ready=not unready,
      unready_motion_ids=unready,
      motions=motions,
      coverage_sampling=self._coverage_sampling,
      coverage_drawn=self._coverage_drawn,
    )

  # Insertion.

  def validate_batch(self, batch: LabeledReplayBatch) -> None:
    """Validate routing, provenance, and tensors without mutating storage."""
    has_provenance = batch.has_provenance
    if (
      self._provenance_enabled is not None
      and has_provenance != self._provenance_enabled
    ):
      raise ReplayValidationError(
        "replay layout does not match the enabled provenance contract"
      )
    validate_replay_batch(batch, self.schema, device=self._device, dtype=self._dtype)
    if batch.batch_size == 0:
      return
    self._validate_routing(batch.motion_id, batch.teacher_id, batch.reference_frame)

  def _validate_routing(
    self,
    motion_id: torch.Tensor,
    teacher_id: torch.Tensor,
    reference_frame: torch.Tensor,
  ) -> None:
    """Reject unknown motions, wrong teacher mapping, or impossible frames."""
    ids = torch.tensor(self._motion_ids, dtype=torch.int64, device=motion_id.device)
    position = torch.searchsorted(ids, motion_id)
    in_range = position < ids.numel()
    position = position.clamp(max=max(ids.numel() - 1, 0))
    known = in_range & (ids[position] == motion_id)
    if not bool(known.all().item()):
      unknown = torch.unique(motion_id[~known]).tolist()
      raise ReplayValidationError(
        f"batched motion ids {unknown} have no replay partition; selected "
        f"motions are {list(self._motion_ids)}"
      )
    codes = torch.tensor(
      self._teacher_codes, dtype=torch.int64, device=motion_id.device
    )[position]
    wrong = teacher_id != codes
    if bool(wrong.any().item()):
      rows = wrong.nonzero().flatten()[:4].tolist()
      raise ReplayValidationError(
        "batched teacher ids disagree with the motion-to-teacher mapping at rows "
        f"{rows}: expected {codes[wrong][:4].tolist()}, got "
        f"{teacher_id[wrong][:4].tolist()}"
      )
    limits = torch.tensor(
      self._frame_counts, dtype=torch.int64, device=motion_id.device
    )[position]
    impossible = (reference_frame < 0) | (reference_frame >= limits)
    if bool(impossible.any().item()):
      rows = impossible.nonzero().flatten()[:4].tolist()
      raise ReplayValidationError(
        "batched reference frames must be inside each row's own clip at rows "
        f"{rows}: frames {reference_frame[impossible][:4].tolist()} against "
        f"limits {limits[impossible][:4].tolist()}"
      )

  def insert(self, batch: LabeledReplayBatch) -> None:
    """Validate and append one batch into the per-motion flat ring segments.

    The complete batch is validated before any row is written.  A valid empty
    batch is a no-op.  An insertion larger than a segment's quota retains that
    motion's newest rows and leaves every other segment untouched.
    """
    self.validate_batch(batch)
    if batch.batch_size == 0:
      return
    if self._provenance_enabled is None:
      self._provenance_enabled = batch.has_provenance
      self._layout = _storage_layout(self.schema, self._provenance_enabled)
    if self._device is None:
      self._establish(batch.device, batch.dtype)
    if self._provenance_enabled and self._meta.inserted_by_initialization is None:
      self._meta.inserted_by_initialization = torch.zeros(
        (len(self._motion_ids), _INITIALIZATION_KINDS),
        dtype=torch.int64,
        device=self._meta.device,
      )
      self._meta.drawn_by_initialization = torch.zeros(
        (len(self._motion_ids), _INITIALIZATION_KINDS),
        dtype=torch.int64,
        device=self._meta.device,
      )
    fields = {name: getattr(batch, name) for name, _ in self._layout}
    if self._fields is None:
      self._fields = {
        name: torch.empty(
          (self._capacity, value.shape[1]) if value.ndim == 2 else (self._capacity,),
          dtype=value.dtype,
          device=value.device,
        )
        for name, value in fields.items()
      }
    self._commit(self._group_rows(batch.motion_id, fields), fields)
    self._size = int(self._meta.size.sum().item())
    self._invalidate_derived()

  def _establish(self, device: torch.device, dtype: torch.dtype) -> None:
    """Adopt the first non-empty batch's storage device and dtype."""
    self._device = _canonical_device(device)
    self._dtype = dtype
    self._meta = self._meta.to(self._device)
    self._bind_device_tables()
    self._invalidate_derived()

  def _group_rows(
    self, motion_id: torch.Tensor, fields: Mapping[str, torch.Tensor]
  ) -> _RowGrouping:
    """Split one validated batch into per-motion row runs, on the device.

    A single stable sort orders the rows by motion id, so each motion's rows
    become one contiguous run of the ordered row array.  The run boundaries come
    from ``bincount``/``cumsum`` over the routed slots, and the write positions
    are one arithmetic expression over the segment cursors: the local ring
    position of the ``r``-th incoming row of a motion is
    ``(next + r) % quota``, and an oversized insertion keeps only its newest
    ``quota`` rows, which land in local slots ``0..quota-1``.  Nothing here
    touches the host, and no per-motion tensor is materialised.
    """
    device = motion_id.device
    meta = self._meta
    ids = torch.tensor(self._motion_ids, dtype=torch.int64, device=device)
    _, order = torch.sort(motion_id, stable=True)
    # Routing was validated before this call, so every row's motion is selected
    # and its slot is in range.  ``self._motion_ids`` is ascending, so the slots
    # are non-decreasing along ``order`` and each motion occupies one run.
    slots = torch.searchsorted(ids, motion_id[order])
    counts = torch.bincount(slots, minlength=len(self._motion_ids))
    rank = (
      torch.arange(order.numel(), dtype=torch.int64, device=device)
      - (counts.cumsum(0) - counts)[slots]
    )
    dropped = (counts - torch.minimum(counts, meta.quota)).clamp(min=0)[slots]
    write_base = torch.where(counts >= meta.quota, torch.zeros_like(counts), meta.next)[
      slots
    ]
    local = (rank - dropped + write_base) % meta.quota[slots]
    destination = meta.start[slots] + local
    keep = rank >= dropped
    kinds = fields.get("initialization_kind")
    histogram = None
    if kinds is not None:
      keys = slots * _INITIALIZATION_KINDS + kinds[order]
      histogram = torch.bincount(
        keys, minlength=len(self._motion_ids) * _INITIALIZATION_KINDS
      ).view(len(self._motion_ids), _INITIALIZATION_KINDS)
    return _RowGrouping(
      counts=counts,
      source=order[keep],
      destination=destination[keep],
      kinds=histogram,
    )

  add = insert

  def _commit(self, grouping: _RowGrouping, fields: Mapping[str, torch.Tensor]) -> None:
    """Write one grouping's retained rows and advance the segment cursors.

    ``grouping.destination`` is unique by construction (distinct segments and,
    inside a segment, distinct local ring slots), so one index assignment per
    field commits the whole insertion as a single device kernel per field.
    """
    meta = self._meta
    if grouping.destination.numel():
      for name, value in fields.items():
        # Detach the gathered rows so a caller's autograd graph can never mark
        # the flat ring as requiring grad.
        self._field(name)[grouping.destination] = value[grouping.source].detach()
    counts = grouping.counts
    meta.inserted.add_(counts)
    meta.size.add_(counts).clamp_(max=meta.quota)
    meta.next.add_(counts).remainder_(meta.quota).masked_fill_(counts >= meta.quota, 0)
    if grouping.kinds is not None:
      assert meta.inserted_by_initialization is not None
      meta.inserted_by_initialization.add_(grouping.kinds)

  # Sampling.

  def _draw_counts(
    self,
    batch_size: int,
    *,
    generator: torch.Generator | None,
    device: torch.device,
  ) -> torch.Tensor:
    """Expected counts from the weights plus unbiased residual allocation.

    Floors come from the configured proportions; the residual rows are assigned
    by random selection among the largest fractional remainders, so odd and
    one-row batches keep the requested mixture in expectation instead of always
    preferring the first motion.  The counts stay on ``device``: the residual
    draw is one ``multinomial`` and one ``bincount``, so no host read is needed.
    """
    total_weight = math.fsum(self._weights)
    exact = [batch_size * weight / total_weight for weight in self._weights]
    counts = [int(math.floor(value)) for value in exact]
    base = torch.tensor(counts, dtype=torch.int64, device=device)
    leftover = batch_size - sum(counts)
    if not leftover:
      return base
    remainders = [
      max(0.0, value - count) for value, count in zip(exact, counts, strict=True)
    ]
    total_remainder = math.fsum(remainders)
    if total_remainder > 0.0:
      probabilities = [value / total_remainder for value in remainders]
    else:
      # Degenerate floating-point residual: stay unbiased instead of biasing
      # the leftover rows toward one motion.
      probabilities = [1.0 / len(counts)] * len(counts)
    picks = torch.multinomial(
      torch.tensor(
        probabilities, dtype=torch.float64, device=_random_device(generator)
      ),
      leftover,
      replacement=True,
      generator=generator,
    )
    return base + torch.bincount(picks.to(device), minlength=len(counts))

  def _refuse_oversized_draw(self, counts: torch.Tensor) -> None:
    """Refuse a without-replacement draw a motion cannot satisfy."""
    too_many = counts > self._meta.size
    if not bool(too_many.any().item()):
      return
    slot = int(too_many.nonzero().flatten()[0].item())
    raise ReplayValidationError(
      f"without-replacement draw asks for {int(counts[slot].item())} rows from "
      f"motion {self._motion_ids[slot]}, which retains "
      f"{int(self._meta.size[slot].item())}"
    )

  def _draw_rows(
    self,
    counts: torch.Tensor,
    *,
    generator: torch.Generator | None,
    random_device: torch.device,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """With-replacement draw: global rows, motion slots, and phase cells.

    Every step is one vectorised expression over the output rows, so the kernel
    count is independent of the motion count: repeat the drawn slots, one
    uniform offset per row inside its own segment, then the global row.  With
    phase balancing the cell of each row is chosen by the same floor-plus-
    residual allocation the per-motion draw used, expressed per row: the first
    ``quota * count // cells`` rows fill every non-empty cell evenly and the
    residual rows draw a uniform non-empty cell each.
    """
    meta = self._meta
    device = meta.device
    slots = torch.repeat_interleave(self._slot_ids, counts)
    if not self._phase_bins:
      size = meta.size[slots]
      offset = _uniform_offsets(size, generator=generator, random_device=random_device)
      base = (meta.next - meta.size) % meta.quota
      rows = meta.start[slots] + (base[slots] + offset) % meta.quota[slots]
      return rows, slots, None

    groups = self._phase_groups()
    rows_per_motion = counts.cumsum(0) - counts
    rank = (
      torch.arange(slots.numel(), dtype=torch.int64, device=device)
      - rows_per_motion[slots]
    )
    cells_of_motion = groups.nonempty_counts[slots]
    even = counts // groups.nonempty_counts
    quotient = even.clamp(min=1)[slots]
    floor_rows = (even * groups.nonempty_counts)[slots]
    in_floor = rank < floor_rows
    even_cell = torch.minimum(rank // quotient, cells_of_motion - 1)
    residual_cell = _uniform_offsets(
      cells_of_motion, generator=generator, random_device=random_device
    )
    cell = groups.nonempty_cells[slots, torch.where(in_floor, even_cell, residual_cell)]
    within = _uniform_offsets(
      groups.cell_sizes[slots, cell],
      generator=generator,
      random_device=random_device,
    )
    member = groups.group_start[slots * self._phase_bins + cell] + within
    return groups.live_rows[groups.group_order[member]], slots, cell

  def _draw_distinct_rows(
    self,
    counts: torch.Tensor,
    *,
    generator: torch.Generator | None,
    random_device: torch.device,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Without-replacement draw: distinct rows inside every motion.

    Distinctness inside a motion needs one permutation of that motion's own
    segment, which cannot be folded into a single kernel over independently
    sized segments, so this path keeps one permutation per motion that receives
    rows (plus one per non-empty phase cell when phase balancing is on).  The
    training path draws with replacement and never enters here.
    """
    meta = self._meta
    device = meta.device
    sizes = meta.size.tolist()
    quotas = meta.quota.tolist()
    starts = meta.start.tolist()
    nexts = meta.next.tolist()
    per_motion = counts.tolist()
    groups = self._phase_groups() if self._phase_bins else None
    group_start = [] if groups is None else groups.group_start.tolist()
    cell_sizes = [] if groups is None else groups.cell_sizes.tolist()
    rows_parts: list[torch.Tensor] = []
    cell_parts: list[torch.Tensor] = []
    for slot, count in enumerate(per_motion):
      if count == 0:
        continue
      base = (nexts[slot] - sizes[slot]) % quotas[slot]
      if groups is None:
        picks = torch.randperm(sizes[slot], device=random_device, generator=generator)[
          :count
        ]
        rows_parts.append(starts[slot] + (base + picks.to(device)) % quotas[slot])
        continue
      nonempty = [cell for cell in range(self._phase_bins) if cell_sizes[slot][cell]]
      available = torch.tensor(
        [cell_sizes[slot][cell] for cell in nonempty], dtype=torch.int64
      )
      allocation = _split_count_over_cells(
        count,
        available,
        generator=generator,
        random_device=random_device,
      )
      for cell, per_cell in zip(nonempty, allocation.tolist(), strict=True):
        if not per_cell:
          continue
        begin = group_start[slot * self._phase_bins + cell]
        picks = torch.randperm(
          cell_sizes[slot][cell], device=random_device, generator=generator
        )[:per_cell]
        member = groups.group_order[begin + picks.to(groups.group_order.device)]
        rows_parts.append(groups.live_rows[member])
        cell_parts.append(
          torch.full((per_cell,), cell, dtype=torch.int64, device=device)
        )
    rows = (
      torch.cat(rows_parts)
      if rows_parts
      else torch.empty(0, dtype=torch.int64, device=device)
    )
    slots = torch.repeat_interleave(self._slot_ids, counts)
    cells = None if groups is None else torch.cat(cell_parts)
    return rows, slots, cells

  def _force_coverage_rows(
    self,
    rows: torch.Tensor,
    slots: torch.Tensor,
    counts: torch.Tensor,
    *,
    generator: torch.Generator | None,
    random_device: torch.device,
  ) -> torch.Tensor:
    """Force one unknown-initialization row into every motion that retains one.

    A motion that already received an unknown row is left alone, and a motion
    that retains none is not invented.  The replacement lands on the motion's
    first drawn row, which the final shuffle then scatters, exactly as the
    per-motion draw did; doing it for all motions at once keeps the host
    synchronisation count at one for the whole draw.
    """
    kinds = self._field("initialization_kind")
    index = self._unknown_rows()
    drawn_unknown = torch.bincount(
      slots[kinds[rows] == 0], minlength=len(self._motion_ids)
    )
    needy = (index.counts > 0) & (drawn_unknown == 0) & (counts > 0)
    needy_slots = self._slot_ids[needy]
    if needy_slots.numel():
      forced = index.rows[
        index.first[needy_slots]
        + _uniform_offsets(
          index.counts[needy_slots],
          generator=generator,
          random_device=random_device,
        )
      ]
      rows = rows.clone()
      rows[(counts.cumsum(0) - counts)[needy_slots]] = forced
    self._coverage_drawn += int((kinds[rows] == 0).sum().item())
    return rows

  def _count_draw(
    self,
    counts: torch.Tensor,
    rows: torch.Tensor,
    slots: torch.Tensor,
    cells: torch.Tensor | None,
  ) -> None:
    """Accumulate the draw's counters on the device, without a host read."""
    meta = self._meta
    motions = len(self._motion_ids)
    meta.drawn.add_(counts)
    if meta.drawn_by_initialization is not None:
      kinds = self._field("initialization_kind")[rows]
      keys = slots * _INITIALIZATION_KINDS + kinds
      meta.drawn_by_initialization.add_(
        torch.bincount(keys, minlength=motions * _INITIALIZATION_KINDS).view(
          motions, _INITIALIZATION_KINDS
        )
      )
    if meta.drawn_by_phase_bin is not None and cells is not None:
      meta.drawn_by_phase_bin.add_(
        torch.bincount(
          slots * self._phase_bins + cells, minlength=motions * self._phase_bins
        ).view(motions, self._phase_bins)
      )

  def _validate_without_replacement_cells(
    self, motion_id: int, slot: int, count: int
  ) -> None:
    """Refuse a without-replacement draw whose cells cannot supply their rows.

    The check is exact for the allocation :func:`_split_count_over_cells`
    performs without replacement — floors of ``count // cells`` plus at most
    one residual row per cell — so a draw is possible exactly when
    ``ceil(count / cells) <= smallest cell``.  It runs before any per-motion
    randomness beyond the motion-mixture counts is consumed and before any
    counter is mutated, mirroring the whole-batch refusal rule for segments:
    never silently reweight an impossible per-cell request.
    """
    sizes = self._phase_groups().cell_sizes[slot]
    nonempty_sizes = sizes[sizes > 0]
    cell_count = int(nonempty_sizes.numel())
    per_cell_floor = count // cell_count
    residual = count - per_cell_floor * cell_count
    largest_possible = per_cell_floor + (1 if residual else 0)
    smallest = int(nonempty_sizes.min().item())
    if largest_possible > smallest:
      raise ReplayValidationError(
        f"without-replacement draw asks for up to {largest_possible} rows from a "
        f"phase cell of motion {motion_id} that retains {smallest}; collect more "
        f"phases or lower --phase-bins"
      )

  def sample(
    self,
    batch_size: int,
    *,
    replacement: bool = True,
    generator: torch.Generator | None = None,
  ) -> LabeledReplayBatch:
    """Draw a balanced mixture of owned rows from every motion segment.

    The draw is deterministic for a given generator state: residual counts are
    drawn first, then one index draw for all output rows at once, then the row
    shuffle.  Sampling never updates model statistics.  With
    ``replacement=False`` rows are distinct inside each motion and a motion
    whose randomized count exceeds its occupancy is refused rather than silently
    reweighted.  When ``phase_bins`` is positive, each motion's rows are
    additionally spread over that clip's non-empty phase cells before the
    per-cell uniform draws.
    """
    if (
      not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0
    ):
      raise ReplayValidationError("sample batch_size must be a positive integer")
    if not isinstance(replacement, bool):
      raise ReplayValidationError("replacement must be a bool")
    if self._size == 0:
      raise ReplayValidationError("cannot sample an empty replay buffer")
    if not replacement and batch_size > self._size:
      raise ReplayValidationError(
        "without-replacement sample size cannot exceed buffer size"
      )
    unready = self.unready_motion_ids
    if unready:
      raise ReplayNotReadyError(
        f"balanced replay partitions {list(unready)} hold no records; collect "
        "until every selected motion is populated before sampling"
      )

    device = self._meta.device
    random_device = _random_device(generator)
    counts = self._draw_counts(batch_size, generator=generator, device=device)
    if not replacement:
      # Refuse before any per-motion randomness or counter mutation so a
      # rejected draw leaves the buffer exactly as it was.  The motion-mixture
      # count draw itself precedes the segment-level refusal identically on the
      # historical unbalanced path, so that ordering is the convention.
      self._refuse_oversized_draw(counts)
      if self._phase_bins:
        for slot, count in enumerate(counts.tolist()):
          if count > 0:
            self._validate_without_replacement_cells(
              self._motion_ids[slot], slot, count
            )
      rows, slots, cells = self._draw_distinct_rows(
        counts, generator=generator, random_device=random_device
      )
    else:
      rows, slots, cells = self._draw_rows(
        counts, generator=generator, random_device=random_device
      )
    if (
      self._coverage_sampling
      and self._fields is not None
      and "initialization_kind" in self._fields
    ):
      rows = self._force_coverage_rows(
        rows, slots, counts, generator=generator, random_device=random_device
      )
    self._count_draw(counts, rows, slots, cells)

    order = torch.randperm(batch_size, device=random_device, generator=generator).to(
      device
    )
    selected = torch.index_select(rows, 0, order)
    fields = {
      name: torch.index_select(self._field(name), 0, selected)
      for name, _ in self._layout
    }
    return _assemble_batch(fields, self.schema)

  # Persistence.

  def state_dict(self) -> dict[str, Any]:
    """Return a detached snapshot of the flat rings, segments, and mappings.

    The segment cursors, counters, ordered routing, and configured
    weights/quotas are retained so a strict resume reproduces both insertion
    and sampling behavior.  Rows outside every segment's active window are
    zeroed, because an unwritten ring slot holds undefined memory and two
    snapshots of the same logical state must be identical.  Callers must treat
    the returned tensors as owned copies.
    """
    state = {
      "kind": _STATE_KIND,
      "version": _STATE_VERSION,
      "capacity": self.capacity,
      "schema": self.schema.compatibility_metadata(),
      "device": None if self._device is None else str(self._device),
      "dtype": None if self._dtype is None else str(self._dtype).replace("torch.", ""),
      "motion_ids": list(self._motion_ids),
      "weights": list(self._weights),
      "quotas": list(self._quota.quotas),
      "teacher_codes": list(self._teacher_codes),
      "frame_counts": list(self._frame_counts),
      "size": self._size,
      "partitions": self._segment_states(),
      "fields": self._field_snapshot(),
    }
    if self._provenance_enabled:
      state["layout"] = "provenance-v1"
    if self._coverage_sampling:
      state["coverage_sampling"] = True
      state["coverage_drawn"] = self._coverage_drawn
    meta = self._meta
    if meta.inserted_by_initialization is not None:
      inserted_by_kind = meta.inserted_by_initialization
      drawn_by_kind = meta.drawn_by_initialization
      assert drawn_by_kind is not None
      state["inserted_by_initialization"] = inserted_by_kind.detach().clone()
      state["drawn_by_initialization"] = drawn_by_kind.detach().clone()
    if meta.drawn_by_phase_bin is not None:
      state["drawn_by_phase_bin"] = meta.drawn_by_phase_bin.detach().clone()
    return state

  def _segment_states(self) -> list[dict[str, Any]]:
    """One metadata record per motion, without any ring contents."""
    meta = self._meta
    motions = len(self._motion_ids)
    hosted = torch.cat([meta.size, meta.quota, meta.next, meta.inserted, meta.drawn])
    sizes, quotas, nexts, inserted, drawn = (
      hosted[block * motions : (block + 1) * motions].tolist() for block in range(5)
    )
    return [
      {
        "motion_id": motion_id,
        "quota": quotas[slot],
        "size": sizes[slot],
        "next": nexts[slot],
        "inserted": inserted[slot],
        "drawn": drawn[slot],
      }
      for slot, motion_id in enumerate(self._motion_ids)
    ]

  def _field_snapshot(self) -> dict[str, torch.Tensor] | None:
    """Copy every flat field to host memory with dead rows zeroed.

    The copy is made on the host on purpose.  Building it on the device clones
    the whole replay while the replay is still resident: at 5 000 000 rows the
    largest field alone is 1.62 GiB, and the clone left the allocator with
    3.93 GiB reserved-but-split and no contiguous block.  The first checkpoint
    of the 3261-clip run died there, after the file had been written, on the
    next iteration's allocation.  A host copy costs one device-to-host transfer
    and adds nothing to peak device memory.
    """
    if self._fields is None:
      return None
    live, _ = self._live_rows()
    keep = torch.zeros(self._capacity, dtype=torch.bool, device="cpu")
    keep[live.detach().to("cpu", dtype=torch.int64)] = True
    snapshot: dict[str, torch.Tensor] = {}
    for name, value in self._fields.items():
      copy = value.detach().to("cpu", copy=True)
      copy[~keep] = 0
      snapshot[name] = copy
    return snapshot

  def validate_state(self, state: Mapping[str, Any]) -> None:
    """Validate a candidate checkpoint state without mutating this buffer."""
    if not isinstance(state, Mapping):
      raise ReplayValidationError("replay state must be a mapping")
    self._validated_state(state)

  def _validated_state(
    self, state: Mapping[str, Any]
  ) -> tuple[
    dict[str, torch.Tensor] | None,
    _MotionSegments,
    torch.device | None,
    torch.dtype | None,
    bool,
    int,
    int,
  ]:
    """Validate one complete checkpoint state into a candidate replacement."""
    # The kind and version are checked first so a checkpoint written by the
    # pre-flat per-motion layout reports its real problem: version 1 stored one
    # ring per motion per field and cannot be read as a flat-ring state.
    if state.get("kind") != _STATE_KIND:
      raise ReplayValidationError("replay state is not a balanced motion replay")
    if state.get("version") != _STATE_VERSION:
      raise ReplayValidationError("replay state version is not supported")
    enabled = state.get("layout") == "provenance-v1"
    coverage_enabled = state.get("coverage_sampling") is True
    required = (
      {
        "kind",
        "version",
        "capacity",
        "schema",
        "device",
        "dtype",
        "motion_ids",
        "weights",
        "quotas",
        "teacher_codes",
        "frame_counts",
        "size",
        "partitions",
        "fields",
      }
      | ({"layout"} if enabled else set())
      | ({"coverage_sampling", "coverage_drawn"} if coverage_enabled else set())
    )
    if not required <= set(state) or set(state) - required - _STATE_OPTIONAL_KEYS:
      raise ReplayValidationError("replay state has missing or unknown fields")
    if self._provenance_enabled is not None and enabled != self._provenance_enabled:
      raise ReplayValidationError("replay checkpoint layout does not match buffer")
    if coverage_enabled != self._coverage_sampling:
      raise ReplayValidationError("replay coverage sampling does not match buffer")
    if coverage_enabled:
      coverage_drawn = state["coverage_drawn"]
      if (
        isinstance(coverage_drawn, bool)
        or not isinstance(coverage_drawn, int)
        or coverage_drawn < 0
      ):
        raise ReplayValidationError("replay coverage draw counter is invalid")
    else:
      coverage_drawn = 0
    if state["capacity"] != self.capacity:
      raise ReplayValidationError("replay capacity does not match checkpoint")
    if state["schema"] != self.schema.compatibility_metadata():
      raise ReplayValidationError("replay schema does not match checkpoint")
    motion_ids = state["motion_ids"]
    if (
      not isinstance(motion_ids, (list, tuple)) or tuple(motion_ids) != self._motion_ids
    ):
      raise ReplayValidationError(
        "replay partition mapping does not match the checkpoint motion ids"
      )
    weights = state["weights"]
    if not isinstance(weights, (list, tuple)) or len(weights) != len(self._weights):
      raise ReplayValidationError("replay partition weights do not align with motions")
    for value, expected in zip(weights, self._weights, strict=True):
      if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayValidationError("replay partition weights are invalid")
      if not math.isclose(float(value), expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ReplayValidationError(
          "replay partition weights do not match the checkpoint configuration"
        )
    quotas = state["quotas"]
    if not isinstance(quotas, (list, tuple)) or tuple(quotas) != self._quota.quotas:
      raise ReplayValidationError(
        "replay partition quotas do not match the checkpoint configuration"
      )
    self._same_routing("teacher codes", state["teacher_codes"], self._teacher_codes)
    self._same_routing("frame counts", state["frame_counts"], self._frame_counts)

    raw_device = state["device"]
    raw_dtype = state["dtype"]
    if raw_device is not None and not isinstance(raw_device, str):
      raise ReplayValidationError("replay device metadata is invalid")
    if raw_dtype is not None and not isinstance(raw_dtype, str):
      raise ReplayValidationError("replay dtype metadata is invalid")
    device = _canonical_device(raw_device) if raw_device is not None else None
    try:
      dtype = getattr(torch, raw_dtype) if raw_dtype is not None else None
    except AttributeError as exc:
      raise ReplayValidationError("replay dtype metadata is invalid") from exc
    if dtype is not None and not dtype.is_floating_point:
      raise ReplayValidationError("replay dtype metadata is not floating point")

    storage_device = self._checkpoint_storage_device(state["fields"], device)
    meta = self._validated_segments(state, device=storage_device, enabled=enabled)
    fields = self._validated_fields(
      state["fields"], enabled=enabled, meta=meta, device=device, dtype=dtype
    )
    if meta.inserted_by_initialization is not None:
      meta.inserted_by_initialization = self._validated_tally(
        state.get("inserted_by_initialization"),
        name="inserted_by_initialization",
        shape=(len(self._motion_ids), _INITIALIZATION_KINDS),
        device=storage_device,
      )
      meta.drawn_by_initialization = self._validated_tally(
        state.get("drawn_by_initialization"),
        name="drawn_by_initialization",
        shape=(len(self._motion_ids), _INITIALIZATION_KINDS),
        device=storage_device,
      )
    if meta.drawn_by_phase_bin is not None:
      meta.drawn_by_phase_bin = self._validated_tally(
        state.get("drawn_by_phase_bin"),
        name="drawn_by_phase_bin",
        shape=(len(self._motion_ids), self._phase_bins),
        device=storage_device,
      )
    size = state["size"]
    if (
      not isinstance(size, int)
      or isinstance(size, bool)
      or not 0 <= size <= self.capacity
    ):
      raise ReplayValidationError("replay size is invalid")
    total = int(meta.size.sum().item())
    if size != total:
      raise ReplayValidationError(
        f"replay size {size} disagrees with the {total} retained partition rows"
      )
    if fields is not None:
      device = fields["reference"].device
      dtype = fields["reference"].dtype
    return fields, meta, device, dtype, enabled, coverage_drawn, size

  @staticmethod
  def _checkpoint_storage_device(
    raw_fields: object, device: torch.device | None
  ) -> torch.device:
    """Device the checkpoint's flat rings live on, before they are validated."""
    if raw_fields is None:
      return device if device is not None else torch.device("cpu")
    if not isinstance(raw_fields, Mapping):
      raise ReplayValidationError("replay field table is invalid")
    table: dict[str, Any] = {str(name): value for name, value in raw_fields.items()}
    reference = table.get("reference")
    if not isinstance(reference, torch.Tensor):
      raise ReplayValidationError("replay field table is invalid")
    return reference.device

  @staticmethod
  def _same_routing(name: str, values: object, expected: tuple[int, ...]) -> None:
    if not isinstance(values, (list, tuple)) or len(values) != len(expected):
      raise ReplayValidationError(
        f"replay state {name} do not align with the checkpoint motions"
      )
    for value, reference in zip(values, expected, strict=True):
      if isinstance(value, bool) or not isinstance(value, int) or value != reference:
        raise ReplayValidationError(
          f"replay state {name} do not match the live routing"
        )

  def _validated_segments(
    self,
    state: Mapping[str, Any],
    *,
    device: torch.device,
    enabled: bool,
  ) -> _MotionSegments:
    """Validate the per-motion metadata records into a candidate."""
    raw_partitions = state["partitions"]
    if not isinstance(raw_partitions, (list, tuple)) or len(raw_partitions) != len(
      self._motion_ids
    ):
      raise ReplayValidationError(
        "replay state must describe one partition per selected motion"
      )
    required = {"motion_id", "quota", "size", "next", "inserted", "drawn"}
    motions = len(self._motion_ids)
    size = torch.zeros(motions, dtype=torch.int64)
    next_index = torch.zeros(motions, dtype=torch.int64)
    inserted = torch.zeros(motions, dtype=torch.int64)
    drawn = torch.zeros(motions, dtype=torch.int64)
    for slot, motion_id in enumerate(self._motion_ids):
      raw = raw_partitions[slot]
      if not isinstance(raw, Mapping) or set(raw) != required:
        raise ReplayValidationError("replay partition fields are invalid")
      fields: dict[str, Any] = {str(name): value for name, value in raw.items()}
      quota = self._quota.quotas[slot]
      if fields["motion_id"] != motion_id or fields["quota"] != quota:
        raise ReplayValidationError(
          f"replay partition identity for motion {motion_id} is invalid"
        )
      retained = _require_int("replay partition size", fields["size"], minimum=0)
      cursor = _require_int("replay partition cursor", fields["next"], minimum=0)
      inserted_rows = _require_int(
        "replay partition inserted", fields["inserted"], minimum=0
      )
      drawn_rows = _require_int("replay partition drawn", fields["drawn"], minimum=0)
      if retained > quota or cursor >= quota:
        raise ReplayValidationError(
          f"replay partition for motion {motion_id} has an invalid size or cursor"
        )
      if inserted_rows < retained:
        raise ReplayValidationError(
          f"replay partition for motion {motion_id} retained more rows than inserted"
        )
      size[slot] = retained
      next_index[slot] = cursor
      inserted[slot] = inserted_rows
      drawn[slot] = drawn_rows
    quota = torch.tensor(self._quota.quotas, dtype=torch.int64)
    start = torch.cat([torch.zeros(1, dtype=torch.int64), quota.cumsum(0)])
    return _MotionSegments(
      start=start[:-1].to(device),
      quota=quota.to(device),
      size=size.to(device),
      next=next_index.to(device),
      inserted=inserted.to(device),
      drawn=drawn.to(device),
      inserted_by_initialization=(
        torch.zeros((motions, _INITIALIZATION_KINDS), dtype=torch.int64, device=device)
        if enabled
        else None
      ),
      drawn_by_initialization=(
        torch.zeros((motions, _INITIALIZATION_KINDS), dtype=torch.int64, device=device)
        if enabled
        else None
      ),
      drawn_by_phase_bin=(
        torch.zeros((motions, self._phase_bins), dtype=torch.int64, device=device)
        if self._phase_bins
        else None
      ),
    )

  def _validated_fields(
    self,
    raw_fields: object,
    *,
    enabled: bool,
    meta: _MotionSegments,
    device: torch.device | None,
    dtype: torch.dtype | None,
  ) -> dict[str, torch.Tensor] | None:
    """Validate the flat field rings and return detached owned copies."""
    widths = dict(_storage_layout(self.schema, enabled))
    if raw_fields is None:
      if int(meta.size.sum().item()) != 0:
        raise ReplayValidationError("non-empty replay state has no fields")
      return None
    if not isinstance(raw_fields, Mapping) or set(raw_fields) != set(widths):
      raise ReplayValidationError("replay field table is invalid")
    table: dict[str, Any] = {str(name): value for name, value in raw_fields.items()}
    live, _ = self._validated_live_rows(meta)
    validated: dict[str, torch.Tensor] = {}
    for name, width in widths.items():
      value = table[name]
      if not isinstance(value, torch.Tensor):
        raise ReplayValidationError(f"replay field {name!r} is not a tensor")
      expected_shape = (self.capacity, width) if width else (self.capacity,)
      if tuple(value.shape) != expected_shape:
        raise ReplayValidationError(f"replay field {name!r} has invalid shape")
      if name in _FLOAT_FIELDS:
        if not value.is_floating_point():
          raise ReplayValidationError(f"replay field {name!r} is invalid")
        if not torch.isfinite(value[live]).all().item():
          raise ReplayValidationError(f"replay field {name!r} is invalid")
        if dtype is not None and value.dtype != dtype:
          raise ReplayValidationError("replay storage dtype does not match metadata")
      elif value.dtype != torch.int64:
        raise ReplayValidationError(f"replay field {name!r} must be int64")
      if device is not None and value.device != device:
        raise ReplayValidationError("replay storage device does not match metadata")
      validated[name] = value.detach().clone()
    return validated

  @staticmethod
  def _validated_live_rows(
    meta: _MotionSegments,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Live global rows and motion slots for a candidate metadata block."""
    device = meta.device
    slots = torch.repeat_interleave(
      torch.arange(meta.size.numel(), dtype=torch.int64, device=device), meta.size
    )
    rank = (
      torch.arange(slots.numel(), dtype=torch.int64, device=device)
      - (meta.size.cumsum(0) - meta.size)[slots]
    )
    local = (meta.next[slots] - meta.size[slots] + rank) % meta.quota[slots]
    return meta.start[slots] + local, slots

  @staticmethod
  def _validated_tally(
    raw: object,
    *,
    name: str,
    shape: tuple[int, int],
    device: torch.device,
  ) -> torch.Tensor:
    """Validate one optional checkpoint counter table, defaulting to zeros."""
    if raw is None:
      return torch.zeros(shape, dtype=torch.int64, device=device)
    if not isinstance(raw, torch.Tensor):
      raise ReplayValidationError(f"replay state {name} is not a tensor")
    if tuple(raw.shape) != shape or raw.dtype != torch.int64:
      raise ReplayValidationError(f"replay state {name} has invalid shape or dtype")
    if bool((raw < 0).any().item()):
      raise ReplayValidationError(f"replay state {name} is invalid")
    return raw.detach().clone().to(device)

  def load_state_dict(self, state: Mapping[str, Any]) -> None:
    """Restore a validated balanced replay state atomically.

    Every segment, quota, weight, mapping, and field ring is validated first;
    this buffer keeps its current contents when any part of the candidate is
    invalid.  Restored segments keep this buffer's configured ``phase_bins``
    policy: the policy is replay-construction state, not ring state, so it is
    validated by the resolved-configuration comparison rather than read from
    the ring.
    """
    if not isinstance(state, Mapping):
      raise ReplayValidationError("replay state must be a mapping")
    fields, meta, device, dtype, enabled, coverage_drawn, size = self._validated_state(
      state
    )
    if self._device is not None and device is not None and self._device != device:
      raise ReplayValidationError("checkpoint replay device does not match buffer")
    if self._dtype is not None and dtype is not None and self._dtype != dtype:
      raise ReplayValidationError("checkpoint replay dtype does not match buffer")
    # All checks above complete before any live field is changed.
    self._device = device
    self._dtype = dtype
    self._provenance_enabled = enabled
    self._coverage_drawn = coverage_drawn
    self._layout = _storage_layout(self.schema, enabled)
    self._fields = fields
    self._meta = meta
    self._size = size
    self._bind_device_tables()
    self._invalidate_derived()

  # Resume seams shared with the single-partition replay buffer.

  def max_valid_segment_id(self) -> int | None:
    """Largest segment ID among retained records, ``None`` when the buffer is empty."""
    if self._size == 0:
      return None
    live, _ = self._live_rows()
    return int(self._field("episode_id")[live].max().item())

  def rebase_segment_ids(self, collector_iteration: int, base: int) -> int:
    """Add ``base`` to segment IDs from one collection iteration.

    This narrow resume API gives a restarted simulator a fresh segment-ID
    namespace without changing insertion or sampling semantics.  It returns the
    number of records changed.
    """
    if (
      not isinstance(collector_iteration, int)
      or isinstance(collector_iteration, bool)
      or not isinstance(base, int)
      or isinstance(base, bool)
      or base < 0
    ):
      raise ReplayValidationError("segment rebase arguments are invalid")
    if self._size == 0 or base == 0:
      return 0
    live, _ = self._live_rows()
    mask = self._field("collector_iteration")[live] == collector_iteration
    self._field("episode_id")[live[mask]] += base
    # One host read for the changed-row count, so a rebase does not cost a
    # synchronisation per motion.
    return int(mask.sum().item())


__all__ = [
  "BalancedReplayBuffer",
  "BalancedReplayReport",
  "MotionCapacityQuota",
  "MotionReplayStats",
  "ReplayNotReadyError",
  "allocate_motion_capacity_quotas",
]
