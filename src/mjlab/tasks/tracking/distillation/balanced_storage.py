"""Balanced per-motion raw replay partitions for M4 multi-teacher distillation.

The M3 :class:`~mjlab.tasks.tracking.distillation.storage.LabeledReplayBuffer`
stays the default single-partition FIFO.  This module adds the opt-in M4 buffer:
one total capacity split into per-motion integer quotas, so a longer or easier
clip can never evict another clip's records, and minibatches are drawn in the
configured motion proportions instead of in whatever mixture one global ring
happens to hold.

Contract:

* Quotas come from the configured motion weights, sum exactly to the requested
  capacity, and give every selected motion at least one slot.
* An insertion validates the whole incoming batch -- schema, device, dtype,
  motion routing, teacher mapping, and reference-frame ranges -- before any
  partition is mutated.  Each partition is an independent FIFO ring, so an
  oversized insertion retains its own newest rows and never evicts another
  motion.
* A draw converts the configured weights into expected counts, assigns the
  residual rows by unbiased randomized selection among the largest fractional
  remainders, samples uniformly inside each partition (with replacement by
  default), and then shuffles the mixture.  A batch of size one is therefore
  not pinned to the first motion.
* A draw raises :class:`ReplayNotReadyError` instead of silently omitting a
  motion whose partition is still empty.
* Inserts clone the caller's tensors and draws return fresh tensors that never
  alias a partition.
* Partitions, quotas, weights, ordered routing, and counters round-trip through
  ``state_dict``/``load_state_dict``, which validates one complete candidate
  before mutating this buffer.  The buffer owns no sampler RNG: every draw uses
  the caller's explicit ``torch.Generator``, whose state the existing checkpoint
  RNG contract persists.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
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
_STATE_VERSION = 1
_FLOAT_FIELDS = frozenset({"reference", "conditioning", "teacher_action"})


class ReplayNotReadyError(ReplayValidationError):
  """A requested draw cannot cover every selected motion yet."""


def _storage_layout(schema: VaeSchema) -> tuple[tuple[str, int], ...]:
  """Ordered ``(field, row width)`` pairs; width ``0`` marks a 1-D field."""
  return (
    ("reference", schema.reference_dim),
    ("conditioning", schema.conditioning_dim),
    ("teacher_action", schema.action_dim),
    ("motion_id", 0),
    ("teacher_id", 0),
    ("reference_frame", 0),
    ("episode_id", 0),
    ("collector_iteration", 0),
  )


def _active_indices(partition: _MotionPartition, device: torch.device) -> torch.Tensor:
  """Physical ring slots holding this partition's records, oldest first.

  The single place that converts a partition's cursor/size into ring positions,
  so insertion, sampling, persistence, and resume agree by construction.
  """
  start = (partition.next - partition.size) % partition.quota
  return (torch.arange(partition.size, device=device) + start) % partition.quota


def _random_device(generator: torch.Generator | None) -> torch.device:
  """Device used for index draws, matching the single-partition replay buffer."""
  return (
    torch.device(generator.device) if generator is not None else torch.device("cpu")
  )


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

  @property
  def coverage(self) -> float:
    """Retained fraction of this motion's capacity quota."""
    return self.retained / self.quota

  @property
  def ready(self) -> bool:
    return self.retained > 0

  def as_dict(self) -> dict[str, Any]:
    return {
      "motion_id": self.motion_id,
      "weight": self.weight,
      "quota": self.quota,
      "retained": self.retained,
      "inserted": self.inserted,
      "drawn": self.drawn,
      "coverage": self.coverage,
      "ready": self.ready,
    }


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

  @property
  def occupancy(self) -> float:
    """Retained fraction of the whole capacity."""
    return self.retained / self.capacity

  def as_dict(self) -> dict[str, Any]:
    return {
      "capacity": self.capacity,
      "retained": self.retained,
      "inserted": self.inserted,
      "drawn": self.drawn,
      "occupancy": self.occupancy,
      "ready": self.ready,
      "unready_motion_ids": list(self.unready_motion_ids),
      "motions": [motion.as_dict() for motion in self.motions],
    }


@dataclass(slots=True)
class _MotionPartition:
  """One motion's independent FIFO ring of retained raw replay records."""

  motion_id: int
  quota: int
  size: int = 0
  next: int = 0
  inserted: int = 0
  drawn: int = 0
  storage: dict[str, torch.Tensor] | None = None


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
  )


class BalancedReplayBuffer:
  """Capacity-bounded replay split into per-motion FIFO partitions.

  ``weights`` are the configured motion proportions; they determine both the
  per-motion capacity quotas and the minibatch mixture.  ``teacher_codes`` and
  ``frame_counts`` are the authoritative motion-to-teacher-code mapping and
  per-motion reference length used to validate every insertion, so a row can
  never be stored under the wrong teacher or an impossible reference frame.
  Motion ids are ordered ascending for deterministic allocation and tie breaks.
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
    self._schema = schema
    self._capacity = quota.total_capacity
    self._device = _canonical_device(device) if device is not None else None
    self._dtype = dtype
    self._size = 0
    self._layout = _storage_layout(schema)
    self._partitions = {
      motion_id: _MotionPartition(motion_id=motion_id, quota=quota.quota_for(motion_id))
      for motion_id in motion_ids
    }

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
  def unready_motion_ids(self) -> tuple[int, ...]:
    """Selected motions whose partition holds no retained record yet."""
    return tuple(
      motion_id
      for motion_id in self._motion_ids
      if self._partitions[motion_id].size == 0
    )

  @property
  def ready(self) -> bool:
    """Whether every selected motion has at least one retained record."""
    return not self.unready_motion_ids

  def retained(self, motion_id: int) -> int:
    """Retained record count for one selected motion."""
    return self._partition(motion_id).size

  def stats(self) -> tuple[MotionReplayStats, ...]:
    return tuple(
      MotionReplayStats(
        motion_id=motion_id,
        weight=self._weights[index],
        quota=self._quota.quotas[index],
        retained=self._partitions[motion_id].size,
        inserted=self._partitions[motion_id].inserted,
        drawn=self._partitions[motion_id].drawn,
      )
      for index, motion_id in enumerate(self._motion_ids)
    )

  def report(self) -> BalancedReplayReport:
    """Capacity, occupancy, and per-motion coverage/counters."""
    motions = self.stats()
    return BalancedReplayReport(
      capacity=self.capacity,
      retained=self._size,
      inserted=sum(motion.inserted for motion in motions),
      drawn=sum(motion.drawn for motion in motions),
      ready=self.ready,
      unready_motion_ids=self.unready_motion_ids,
      motions=motions,
    )

  def _partition(self, motion_id: int) -> _MotionPartition:
    try:
      return self._partitions[motion_id]
    except KeyError as exc:
      raise ReplayValidationError(
        f"motion {motion_id} has no replay partition"
      ) from exc

  # Insertion.

  def validate_batch(self, batch: LabeledReplayBatch) -> None:
    """Validate routing, mapping, frames, and tensors without mutating storage."""
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
    """Validate and append one batch, partitioned by motion.

    The complete batch is validated before any partition is mutated.  A valid
    empty batch is a no-op.  An insertion larger than a partition's quota
    retains that partition's newest rows and leaves every other partition
    untouched.
    """
    self.validate_batch(batch)
    if batch.batch_size == 0:
      return
    fields = {name: getattr(batch, name) for name, _ in self._layout}
    # Row selection happens before any write, so a failure can never leave part
    # of an insertion committed.
    groups: list[tuple[int, dict[str, torch.Tensor]]] = []
    for motion_id in self._motion_ids:
      rows = batch.motion_id == motion_id
      count = int(rows.sum().item())
      if count == 0:
        continue
      groups.append(
        (
          motion_id,
          {name: value[rows].detach().clone() for name, value in fields.items()},
        )
      )
    if self._device is None:
      self._device = _canonical_device(batch.device)
    if self._dtype is None:
      self._dtype = batch.dtype
    for motion_id, owned in groups:
      self._commit(motion_id, owned)
    self._size = sum(partition.size for partition in self._partitions.values())

  add = insert

  def _commit(self, motion_id: int, owned: Mapping[str, torch.Tensor]) -> None:
    """Append one motion's already-validated rows to its own FIFO ring."""
    partition = self._partitions[motion_id]
    quota = partition.quota
    count = owned["reference"].shape[0]
    if count >= quota:
      partition.storage = {
        name: value[-quota:].detach().clone() for name, value in owned.items()
      }
      partition.size = quota
      partition.next = 0
    else:
      if partition.storage is None:
        partition.storage = {
          name: torch.empty(
            (quota, value.shape[1]) if value.ndim == 2 else (quota,),
            dtype=value.dtype,
            device=value.device,
          )
          for name, value in owned.items()
        }
      device = partition.storage["reference"].device
      indices = (torch.arange(count, device=device) + partition.next) % quota
      for name, value in owned.items():
        partition.storage[name][indices] = value
      partition.size = min(quota, partition.size + count)
      partition.next = (partition.next + count) % quota
    partition.inserted += count

  # Sampling.

  def _draw_counts(
    self, batch_size: int, *, generator: torch.Generator | None
  ) -> tuple[int, ...]:
    """Expected counts from the weights plus unbiased residual allocation.

    Floors come from the configured proportions; the residual rows are assigned
    by random selection among the largest fractional remainders, so odd and
    one-row batches keep the requested mixture in expectation instead of always
    preferring the first motion.
    """
    total_weight = math.fsum(self._weights)
    exact = [batch_size * weight / total_weight for weight in self._weights]
    counts = [int(math.floor(value)) for value in exact]
    leftover = batch_size - sum(counts)
    if leftover:
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
      extra = torch.bincount(picks, minlength=len(counts)).tolist()
      counts = [count + int(value) for count, value in zip(counts, extra, strict=True)]
    return tuple(counts)

  def _partition_indices(
    self,
    partition: _MotionPartition,
    count: int,
    *,
    replacement: bool,
    generator: torch.Generator | None,
    random_device: torch.device,
  ) -> torch.Tensor:
    if replacement:
      logical = torch.randint(
        partition.size, (count,), device=random_device, generator=generator
      )
    else:
      logical = torch.randperm(
        partition.size, device=random_device, generator=generator
      )[:count]
    storage = partition.storage
    if storage is None:  # pragma: no cover - readiness is checked before drawing
      raise ReplayNotReadyError(
        f"motion {partition.motion_id} has no retained replay records"
      )
    logical = logical.to(device=storage["reference"].device)
    return _active_indices(partition, storage["reference"].device)[logical]

  def sample(
    self,
    batch_size: int,
    *,
    replacement: bool = True,
    generator: torch.Generator | None = None,
  ) -> LabeledReplayBatch:
    """Draw a balanced mixture of owned rows from every motion partition.

    The draw is deterministic for a given generator state: residual counts are
    drawn first, then one index draw per motion in ascending motion order, then
    the row shuffle.  Sampling never updates model statistics.  With
    ``replacement=False`` rows are distinct inside each partition and a
    partition whose randomized count exceeds its occupancy is refused rather
    than silently reweighted.
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

    random_device = _random_device(generator)
    counts = self._draw_counts(batch_size, generator=generator)
    if not replacement:
      # Refuse before any partition is touched so a rejected draw leaves the
      # buffer (and its counters) exactly as it was.
      for motion_id, count in zip(self._motion_ids, counts, strict=True):
        available = self._partitions[motion_id].size
        if count > available:
          raise ReplayValidationError(
            f"without-replacement draw asks for {count} rows from motion "
            f"{motion_id}, which retains {available}"
          )
    parts: dict[str, list[torch.Tensor]] = {}
    for motion_id, count in zip(self._motion_ids, counts, strict=True):
      if count == 0:
        continue
      partition = self._partitions[motion_id]
      indices = self._partition_indices(
        partition,
        count,
        replacement=replacement,
        generator=generator,
        random_device=random_device,
      )
      storage = partition.storage
      assert storage is not None
      for name, _ in self._layout:
        parts.setdefault(name, []).append(storage[name][indices])
      partition.drawn += count

    order = torch.randperm(batch_size, device=random_device, generator=generator).to(
      device=self._device
    )
    fields = {name: torch.cat(values, dim=0)[order] for name, values in parts.items()}
    return _assemble_batch(fields, self.schema)

  # Persistence.

  def state_dict(self) -> dict[str, Any]:
    """Return a detached snapshot of every partition, quota, and mapping.

    The partition rings, cursors, counters, ordered routing, and configured
    weights/quotas are retained so a strict resume reproduces both insertion
    and sampling behavior.  Callers must treat the returned tensors as owned
    copies.
    """
    return {
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
      "partitions": [
        self._partition_state(self._partitions[motion_id])
        for motion_id in self._motion_ids
      ],
    }

  @staticmethod
  def _partition_state(partition: _MotionPartition) -> dict[str, Any]:
    """Detached snapshot of one partition ring, cursor, and counters.

    Rows outside the partition's active window are zeroed instead of copied: a
    ring slot that was never written holds undefined memory, so cloning it
    would make two snapshots of the same logical state differ.  Restoring still
    accepts a snapshot that carries such unwritten slots, because only the
    active window is read.
    """
    storage = partition.storage
    snapshot: dict[str, torch.Tensor] | None = None
    if storage is not None:
      snapshot = {name: value.detach().clone() for name, value in storage.items()}
      active = _active_indices(partition, storage["reference"].device)
      if active.numel() < partition.quota:
        keep = torch.zeros(
          partition.quota, dtype=torch.bool, device=storage["reference"].device
        )
        keep[active] = True
        for value in snapshot.values():
          value[~keep] = 0
    return {
      "motion_id": partition.motion_id,
      "quota": partition.quota,
      "size": partition.size,
      "next": partition.next,
      "inserted": partition.inserted,
      "drawn": partition.drawn,
      "storage": snapshot,
    }

  def validate_state(self, state: Mapping[str, Any]) -> None:
    """Validate a candidate checkpoint state without mutating this buffer."""
    if not isinstance(state, Mapping):
      raise ReplayValidationError("replay state must be a mapping")
    self._validated_state(state)

  def _validated_state(
    self, state: Mapping[str, Any]
  ) -> tuple[dict[int, _MotionPartition], torch.device | None, torch.dtype | None]:
    """Validate one complete checkpoint state into a candidate replacement."""
    required = {
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
    }
    if set(state) != required:
      raise ReplayValidationError("replay state has missing or unknown fields")
    if state["kind"] != _STATE_KIND:
      raise ReplayValidationError("replay state is not a balanced motion replay")
    if state["version"] != _STATE_VERSION:
      raise ReplayValidationError("replay state version is not supported")
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

    raw_partitions = state["partitions"]
    if not isinstance(raw_partitions, (list, tuple)) or len(raw_partitions) != len(
      self._motion_ids
    ):
      raise ReplayValidationError(
        "replay state must describe one partition per selected motion"
      )
    widths = dict(self._layout)
    candidate: dict[int, _MotionPartition] = {}
    for index, motion_id in enumerate(self._motion_ids):
      candidate[motion_id] = self._validated_partition(
        raw_partitions[index],
        motion_id=motion_id,
        quota=self._quota.quotas[index],
        widths=widths,
        device=device,
        dtype=dtype,
      )
    resolved_device, resolved_dtype = self._resolved_storage_policy(
      candidate, device, dtype
    )
    device, dtype = resolved_device, resolved_dtype
    size = state["size"]
    if (
      not isinstance(size, int)
      or isinstance(size, bool)
      or not 0 <= size <= self.capacity
    ):
      raise ReplayValidationError("replay size is invalid")
    total = sum(partition.size for partition in candidate.values())
    if size != total:
      raise ReplayValidationError(
        f"replay size {size} disagrees with the {total} retained partition rows"
      )
    return candidate, device, dtype

  @staticmethod
  def _resolved_storage_policy(
    partitions: Mapping[int, _MotionPartition],
    device: torch.device | None,
    dtype: torch.dtype | None,
  ) -> tuple[torch.device | None, torch.dtype | None]:
    """Resolve missing device/dtype metadata and require one value per partition."""
    for partition in partitions.values():
      if partition.storage is None:
        continue
      tensor = partition.storage["reference"]
      if device is None:
        device = tensor.device
      elif tensor.device != device:
        raise ReplayValidationError("replay partitions must share one device")
      if dtype is None:
        dtype = tensor.dtype
      elif tensor.dtype != dtype:
        raise ReplayValidationError("replay partitions must share one dtype")
    return device, dtype

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

  @staticmethod
  def _validated_partition(
    raw: object,
    *,
    motion_id: int,
    quota: int,
    widths: Mapping[str, int],
    device: torch.device | None,
    dtype: torch.dtype | None,
  ) -> _MotionPartition:
    """Validate one partition's ring and return a detached owned candidate."""
    required = {
      "motion_id",
      "quota",
      "size",
      "next",
      "inserted",
      "drawn",
      "storage",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
      raise ReplayValidationError("replay partition fields are invalid")
    fields: dict[str, Any] = {str(name): value for name, value in raw.items()}
    if fields["motion_id"] != motion_id or fields["quota"] != quota:
      raise ReplayValidationError(
        f"replay partition identity for motion {motion_id} is invalid"
      )
    size = _require_int("replay partition size", fields["size"], minimum=0)
    next_index = _require_int("replay partition cursor", fields["next"], minimum=0)
    inserted = _require_int("replay partition inserted", fields["inserted"], minimum=0)
    drawn = _require_int("replay partition drawn", fields["drawn"], minimum=0)
    if size > quota or next_index >= quota:
      raise ReplayValidationError(
        f"replay partition for motion {motion_id} has an invalid size or cursor"
      )
    if inserted < size:
      raise ReplayValidationError(
        f"replay partition for motion {motion_id} retained more rows than inserted"
      )
    raw_storage = fields["storage"]
    if raw_storage is None:
      if size != 0:
        raise ReplayValidationError("non-empty replay partition has no storage")
      return _MotionPartition(
        motion_id=motion_id,
        quota=quota,
        size=0,
        next=next_index,
        inserted=inserted,
        drawn=drawn,
        storage=None,
      )
    if not isinstance(raw_storage, Mapping) or set(raw_storage) != set(widths):
      raise ReplayValidationError("replay partition storage fields are invalid")
    storage_fields: dict[str, Any] = {
      str(name): value for name, value in raw_storage.items()
    }
    validated: dict[str, torch.Tensor] = {}
    for name, width in widths.items():
      value = storage_fields[name]
      if not isinstance(value, torch.Tensor):
        raise ReplayValidationError(f"replay partition field {name!r} is not a tensor")
      expected_shape = (quota, width) if width else (quota,)
      if tuple(value.shape) != expected_shape:
        raise ReplayValidationError(
          f"replay partition field {name!r} has invalid shape"
        )
      if name in _FLOAT_FIELDS:
        start = (next_index - size) % quota
        active = (torch.arange(size, device=value.device) + start) % quota
        finite = torch.isfinite(value[active]).all().item()
        if not value.is_floating_point() or not finite:
          raise ReplayValidationError(f"replay partition field {name!r} is invalid")
        if dtype is not None and value.dtype != dtype:
          raise ReplayValidationError("replay storage dtype does not match metadata")
      elif value.dtype != torch.int64:
        raise ReplayValidationError(f"replay partition field {name!r} must be int64")
      if device is not None and value.device != device:
        raise ReplayValidationError("replay storage device does not match metadata")
      validated[name] = value.detach().clone()
    return _MotionPartition(
      motion_id=motion_id,
      quota=quota,
      size=size,
      next=next_index,
      inserted=inserted,
      drawn=drawn,
      storage=validated,
    )

  def load_state_dict(self, state: Mapping[str, Any]) -> None:
    """Restore a validated balanced replay state atomically.

    Every partition, quota, weight, and mapping is validated first; this buffer
    keeps its current contents when any part of the candidate is invalid.
    """
    if not isinstance(state, Mapping):
      raise ReplayValidationError("replay state must be a mapping")
    partitions, device, dtype = self._validated_state(state)
    if self._device is not None and device is not None and self._device != device:
      raise ReplayValidationError("checkpoint replay device does not match buffer")
    if self._dtype is not None and dtype is not None and self._dtype != dtype:
      raise ReplayValidationError("checkpoint replay dtype does not match buffer")
    # All checks above complete before any live field is changed.
    self._device = device
    self._dtype = dtype
    self._partitions = partitions
    self._size = sum(partition.size for partition in partitions.values())

  # Resume seams shared with the single-partition replay buffer.

  def max_valid_segment_id(self) -> int | None:
    """Largest segment ID among retained records, ``None`` when the buffer is empty."""
    highest: int | None = None
    if self._size == 0:
      return None
    for motion_id in self._motion_ids:
      partition = self._partitions[motion_id]
      if partition.size == 0 or partition.storage is None:
        continue
      storage = partition.storage
      active = _active_indices(partition, storage["episode_id"].device)
      value = int(storage["episode_id"][active].max().item())
      highest = value if highest is None else max(highest, value)
    return highest

  def rebase_segment_ids(self, collector_iteration: int, base: int) -> int:
    """Add ``base`` to segment IDs from one collection iteration in every partition.

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
    changed = 0
    for motion_id in self._motion_ids:
      partition = self._partitions[motion_id]
      if partition.size == 0 or partition.storage is None:
        continue
      storage = partition.storage
      active = _active_indices(partition, storage["episode_id"].device)
      mask = storage["collector_iteration"][active] == collector_iteration
      if not bool(mask.any().item()):
        continue
      storage["episode_id"][active[mask]] += base
      changed += int(mask.sum().item())
    return changed


__all__ = [
  "BalancedReplayBuffer",
  "BalancedReplayReport",
  "MotionCapacityQuota",
  "MotionReplayStats",
  "ReplayNotReadyError",
  "allocate_motion_capacity_quotas",
]
