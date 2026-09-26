"""Capacity-bounded raw replay for labeled VAE observations.

The buffer stores only raw packed observations, teacher action labels, and
integer routing/collection metadata.  It has no model, optimizer, simulator,
or latent cache dependency.  Floating-point observations keep their input
(dtype and device); the first non-empty insertion establishes those values
unless they were supplied to the constructor.  All metadata uses ``int64`` and
must be on the same device as the observations.

Insertion is FIFO: when an insertion is larger than capacity, only its latest
``capacity`` rows are retained.  Sampling is uniform with replacement by
default.  A caller may pass a ``torch.Generator`` to make the draw
reproducible; the generator's device is used for random-index generation.
Empty insertion is a validated no-op.  Sampling an empty buffer raises
``ReplayValidationError``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from mjlab.tasks.tracking.distillation.observations import (
  ObservationValidationError,
  PackedObservationBatch,
)
from mjlab.tasks.tracking.distillation.vae_config import VaeSchema


class ReplayValidationError(ValueError):
  """An insertion or sampling request violates the replay contract."""


@dataclass(frozen=True, slots=True)
class LabeledReplayBatch:
  """Raw observations and labels for aligned replay records.

  ``episode_id`` is the segment/episode routing identifier.  All five
  metadata tensors are one-dimensional ``int64`` tensors of length ``B``.
  ``observations`` may have ``B == 0`` for an explicit empty insertion; the
  regular observation validator intentionally rejects empty batches, so the
  replay boundary validates that case separately.
  """

  observations: PackedObservationBatch
  teacher_action: torch.Tensor
  motion_id: torch.Tensor
  teacher_id: torch.Tensor
  reference_frame: torch.Tensor
  episode_id: torch.Tensor
  collector_iteration: torch.Tensor

  @property
  def reference(self) -> torch.Tensor:
    """Return the raw reference tensor without copying it."""
    return self.observations.reference

  @property
  def conditioning(self) -> torch.Tensor:
    """Return the raw decoder-conditioning tensor without copying it."""
    return self.observations.conditioning

  @property
  def schema(self) -> VaeSchema:
    return self.observations.schema

  @property
  def batch_size(self) -> int:
    return self.reference.shape[0]

  @property
  def device(self) -> torch.device:
    return self.reference.device

  @property
  def dtype(self) -> torch.dtype:
    return self.reference.dtype

  @property
  def segment_id(self) -> torch.Tensor:
    """Alias for callers that call episode/segment IDs ``segment_id``."""
    return self.episode_id


# A shorter name is useful at call sites while retaining the descriptive API.
ReplayBatch = LabeledReplayBatch


def _require_tensor(name: str, value: object) -> torch.Tensor:
  if not isinstance(value, torch.Tensor):
    raise ReplayValidationError(f"{name} must be a torch.Tensor")
  return value


def _require_matrix(
  name: str,
  value: object,
  batch_size: int,
  width: int,
  *,
  floating: bool = True,
) -> torch.Tensor:
  tensor = _require_tensor(name, value)
  if tensor.ndim != 2 or tensor.shape != (batch_size, width):
    raise ReplayValidationError(f"{name} must have shape [{batch_size}, {width}]")
  if floating and not tensor.is_floating_point():
    raise ReplayValidationError(f"{name} must use a floating-point dtype")
  if not floating and tensor.dtype != torch.int64:
    raise ReplayValidationError(f"{name} must use torch.int64")
  if floating and not torch.isfinite(tensor).all().item():
    raise ReplayValidationError(f"{name} contains non-finite values")
  return tensor


def _require_metadata(name: str, value: object, batch_size: int) -> torch.Tensor:
  tensor = _require_tensor(name, value)
  if tensor.ndim != 1 or tensor.shape[0] != batch_size:
    raise ReplayValidationError(f"{name} must have shape [{batch_size}]")
  if tensor.dtype != torch.int64:
    raise ReplayValidationError(f"{name} must use torch.int64")
  return tensor


def _canonical_device(device: torch.device | str) -> torch.device:
  """Resolve aliases so policy comparisons match tensor allocation devices."""
  resolved = torch.device(device)
  if resolved.type == "cpu":
    return torch.device("cpu")
  if resolved.type == "cuda" and resolved.index is None:
    if torch.cuda.is_available():
      return torch.device("cuda", torch.cuda.current_device())
  return resolved


class LabeledReplayBuffer:
  """A schema-checked, capacity-bounded FIFO tensor replay buffer.

  The buffer does not transfer tensors between devices or cast dtypes.  Raw
  floating tensors must share one device and dtype within a batch and with
  earlier insertions.  Metadata is always ``torch.int64`` and must use the
  same device as raw tensors.  If ``device``/``dtype`` are omitted, the first
  non-empty insertion establishes them; empty insertions do not establish
  either value.
  """

  def __init__(
    self,
    capacity: int,
    schema: VaeSchema,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
  ) -> None:
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
      raise ReplayValidationError("capacity must be a positive integer")
    if not isinstance(schema, VaeSchema):
      raise ReplayValidationError("schema must be a VaeSchema")
    if dtype is not None and not dtype.is_floating_point:
      raise ReplayValidationError("dtype must be a floating-point torch dtype")
    self.capacity = capacity
    self.schema = schema
    self._device = _canonical_device(device) if device is not None else None
    self._dtype = dtype
    self._size = 0
    self._next = 0
    self._storage: dict[str, torch.Tensor] | None = None

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

  def _validate_batch(self, batch: LabeledReplayBatch) -> None:
    if not isinstance(batch, LabeledReplayBatch):
      raise ReplayValidationError("batch must be a LabeledReplayBatch")
    if batch.schema != self.schema:
      raise ReplayValidationError("batch schema does not match replay schema")

    batch_size = batch.batch_size
    reference = _require_matrix(
      "reference", batch.reference, batch_size, self.schema.reference_dim
    )
    conditioning = _require_matrix(
      "conditioning", batch.conditioning, batch_size, self.schema.conditioning_dim
    )
    teacher_action = _require_matrix(
      "teacher_action", batch.teacher_action, batch_size, self.schema.action_dim
    )
    metadata = (
      ("motion_id", batch.motion_id),
      ("teacher_id", batch.teacher_id),
      ("reference_frame", batch.reference_frame),
      ("episode_id", batch.episode_id),
      ("collector_iteration", batch.collector_iteration),
    )
    metadata_tensors = [
      _require_metadata(name, value, batch_size) for name, value in metadata
    ]

    tensors = [reference, conditioning, teacher_action, *metadata_tensors]
    expected_device = tensors[0].device
    expected_dtype = tensors[0].dtype
    if any(tensor.device != expected_device for tensor in tensors[1:]):
      raise ReplayValidationError("all replay fields must share one device")
    if any(tensor.dtype != expected_dtype for tensor in (conditioning, teacher_action)):
      raise ReplayValidationError(
        "reference, conditioning and teacher_action must share one dtype"
      )
    if self._device is not None and expected_device != self._device:
      raise ReplayValidationError(
        f"replay tensors use {expected_device}, expected {self._device}"
      )
    if self._dtype is not None and expected_dtype != self._dtype:
      raise ReplayValidationError(
        f"replay tensors use {expected_dtype}, expected {self._dtype}"
      )
    # PackedObservationBatch.validate deliberately requires a non-empty batch.
    # The replay boundary has already checked the complete empty shape/device/
    # dtype contract above, so only invoke it where its precondition holds.
    if batch_size > 0:
      try:
        batch.observations.validate()
      except ObservationValidationError as exc:
        raise ReplayValidationError(str(exc)) from exc

  def _owned_batch(self, batch: LabeledReplayBatch) -> dict[str, torch.Tensor]:
    """Clone all fields before the first ring slot is mutated."""
    return {
      "reference": batch.reference.detach().clone(),
      "conditioning": batch.conditioning.detach().clone(),
      "teacher_action": batch.teacher_action.detach().clone(),
      "motion_id": batch.motion_id.detach().clone(),
      "teacher_id": batch.teacher_id.detach().clone(),
      "reference_frame": batch.reference_frame.detach().clone(),
      "episode_id": batch.episode_id.detach().clone(),
      "collector_iteration": batch.collector_iteration.detach().clone(),
    }

  def insert(self, batch: LabeledReplayBatch) -> None:
    """Validate and append a batch, retaining the newest records on overflow.

    Every field is validated before any ring storage is changed.  A valid empty
    batch is a no-op, including when the buffer has no established device or
    dtype.
    """
    self._validate_batch(batch)
    if batch.batch_size == 0:
      return
    if self._device is None:
      self._device = _canonical_device(batch.device)
    if self._dtype is None:
      self._dtype = batch.dtype
    owned = self._owned_batch(batch)
    if batch.batch_size >= self.capacity:
      owned = {name: value[-self.capacity :] for name, value in owned.items()}
      self._storage = {name: value.clone() for name, value in owned.items()}
      self._size = self.capacity
      self._next = 0
      return

    if self._storage is None:
      self._storage = {
        name: torch.empty(
          (self.capacity, value.shape[1]), dtype=value.dtype, device=value.device
        )
        if value.ndim == 2
        else torch.empty((self.capacity,), dtype=value.dtype, device=value.device)
        for name, value in owned.items()
      }

    indices = (
      torch.arange(batch.batch_size, device=self._device) + self._next
    ) % self.capacity
    for name, value in owned.items():
      self._storage[name][indices] = value
    self._size = min(self.capacity, self._size + batch.batch_size)
    self._next = (self._next + batch.batch_size) % self.capacity

  add = insert

  def state_dict(self) -> dict[str, Any]:
    """Return a detached, version-neutral snapshot of the ring storage.

    The ring layout and write cursor are retained so restoring a checkpoint
    preserves FIFO order and subsequent sampling/insertion behavior.  Callers
    must treat the returned tensors as owned copies.
    """
    storage = None
    if self._storage is not None:
      storage = {name: value.detach().clone() for name, value in self._storage.items()}
    return {
      "capacity": self.capacity,
      "schema": self.schema.compatibility_metadata(),
      "device": None if self._device is None else str(self._device),
      "dtype": None if self._dtype is None else str(self._dtype).replace("torch.", ""),
      "size": self._size,
      "next": self._next,
      "storage": storage,
    }

  def _validated_state_storage(
    self, state: Mapping[str, Any]
  ) -> tuple[
    int, int, dict[str, torch.Tensor] | None, torch.device | None, torch.dtype | None
  ]:
    """Validate checkpoint replay state without changing this buffer."""
    required = {"capacity", "schema", "device", "dtype", "size", "next", "storage"}
    if set(state) != required:
      raise ReplayValidationError("replay state has missing or unknown fields")
    if state["capacity"] != self.capacity:
      raise ReplayValidationError("replay capacity does not match checkpoint")
    if state["schema"] != self.schema.compatibility_metadata():
      raise ReplayValidationError("replay schema does not match checkpoint")
    size, next_index = state["size"], state["next"]
    if (
      not isinstance(size, int)
      or isinstance(size, bool)
      or not 0 <= size <= self.capacity
      or not isinstance(next_index, int)
      or isinstance(next_index, bool)
      or not 0 <= next_index < self.capacity
    ):
      raise ReplayValidationError("replay size or ring cursor is invalid")
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
    raw_storage = state["storage"]
    if raw_storage is None:
      if size != 0:
        raise ReplayValidationError("non-empty replay state has no storage")
      return size, next_index, None, device, dtype
    if not isinstance(raw_storage, Mapping) or set(raw_storage) != set(
      (
        "reference",
        "conditioning",
        "teacher_action",
        "motion_id",
        "teacher_id",
        "reference_frame",
        "episode_id",
        "collector_iteration",
      )
    ):
      raise ReplayValidationError("replay storage fields are invalid")
    validated: dict[str, torch.Tensor] = {}
    widths = {
      "reference": self.schema.reference_dim,
      "conditioning": self.schema.conditioning_dim,
      "teacher_action": self.schema.action_dim,
    }
    for name, value in raw_storage.items():
      if not isinstance(value, torch.Tensor):
        raise ReplayValidationError(f"replay storage field {name!r} is not a tensor")
      expected_shape = (
        (self.capacity, widths[name]) if name in widths else (self.capacity,)
      )
      if tuple(value.shape) != expected_shape:
        raise ReplayValidationError(f"replay storage field {name!r} has invalid shape")
      if name in widths:
        active = torch.arange(size, device=value.device)
        start = (next_index - size) % self.capacity
        active = (active + start) % self.capacity
        if (
          not value.is_floating_point()
          or not torch.isfinite(value[active]).all().item()
        ):
          raise ReplayValidationError(f"replay storage field {name!r} is invalid")
        if dtype is not None and value.dtype != dtype:
          raise ReplayValidationError("replay storage dtype does not match metadata")
      elif value.dtype != torch.int64:
        raise ReplayValidationError(f"replay storage field {name!r} must be int64")
      if device is not None and value.device != device:
        raise ReplayValidationError("replay storage device does not match metadata")
      validated[name] = value.detach().clone()
    if device is None:
      device = validated["reference"].device
    if dtype is None:
      dtype = validated["reference"].dtype
    return size, next_index, validated, device, dtype

  def load_state_dict(self, state: Mapping[str, Any]) -> None:
    """Restore a validated ring snapshot atomically."""
    if not isinstance(state, Mapping):
      raise ReplayValidationError("replay state must be a mapping")
    size, next_index, storage, device, dtype = self._validated_state_storage(state)
    if self._device is not None and device is not None and self._device != device:
      raise ReplayValidationError("checkpoint replay device does not match buffer")
    if self._dtype is not None and dtype is not None and self._dtype != dtype:
      raise ReplayValidationError("checkpoint replay dtype does not match buffer")
    # All checks above complete before any live field is changed.
    self._device = device
    self._dtype = dtype
    self._size = size
    self._next = next_index
    self._storage = storage

  def rebase_segment_ids(self, collector_iteration: int, base: int) -> int:
    """Add ``base`` to segment IDs from one owned collection iteration.

    This narrow resume API lets a restarted simulator use a fresh segment-ID
    namespace without changing insertion or sampling semantics.  It returns
    the number of records changed and performs no operation on an empty ring.
    """
    if (
      not isinstance(collector_iteration, int)
      or isinstance(collector_iteration, bool)
      or not isinstance(base, int)
      or isinstance(base, bool)
      or base < 0
    ):
      raise ReplayValidationError("segment rebase arguments are invalid")
    if self._storage is None or self._size == 0 or base == 0:
      return 0
    start = (self._next - self._size) % self.capacity
    logical = torch.arange(self._size, device=self._device)
    physical = (logical + start) % self.capacity
    selected_iterations = self._storage["collector_iteration"][physical]
    mask = selected_iterations == collector_iteration
    if not bool(mask.any().item()):
      return 0
    targets = physical[mask]
    self._storage["episode_id"][targets] += base
    return int(mask.sum().item())

  def sample(
    self,
    batch_size: int,
    *,
    replacement: bool = True,
    generator: torch.Generator | None = None,
  ) -> LabeledReplayBatch:
    """Draw a uniform batch and return detached, non-aliasing tensors.

    Sampling is with replacement by default.  Set ``replacement=False`` to
    draw distinct records; that request must not exceed the current size.
    """
    if (
      not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0
    ):
      raise ReplayValidationError("sample batch_size must be a positive integer")
    if not isinstance(replacement, bool):
      raise ReplayValidationError("replacement must be a bool")
    if self._size == 0 or self._storage is None:
      raise ReplayValidationError("cannot sample an empty replay buffer")
    if not replacement and batch_size > self._size:
      raise ReplayValidationError(
        "without-replacement sample size cannot exceed buffer size"
      )

    random_device = (
      torch.device(generator.device) if generator is not None else torch.device("cpu")
    )
    if replacement:
      logical_indices = torch.randint(
        self._size, (batch_size,), device=random_device, generator=generator
      )
    else:
      logical_indices = torch.randperm(
        self._size, device=random_device, generator=generator
      )[:batch_size]
    logical_indices = logical_indices.to(device=self._device)
    start = (self._next - self._size) % self.capacity
    physical_indices = (logical_indices + start) % self.capacity

    assert self._storage is not None
    selected = {
      name: value[physical_indices].detach().clone()
      for name, value in self._storage.items()
    }
    observations = PackedObservationBatch(
      selected["reference"], selected["conditioning"], self.schema
    )
    return LabeledReplayBatch(
      observations=observations,
      teacher_action=selected["teacher_action"],
      motion_id=selected["motion_id"],
      teacher_id=selected["teacher_id"],
      reference_frame=selected["reference_frame"],
      episode_id=selected["episode_id"],
      collector_iteration=selected["collector_iteration"],
    )


# Both names are intentionally public: the descriptive name is preferred in
# new code, while ReplayBuffer keeps the common short spelling ergonomic.
ReplayBuffer = LabeledReplayBuffer

__all__ = [
  "LabeledReplayBatch",
  "LabeledReplayBuffer",
  "ReplayBatch",
  "ReplayBuffer",
  "ReplayValidationError",
]
