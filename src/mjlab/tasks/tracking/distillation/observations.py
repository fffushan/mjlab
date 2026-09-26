"""Pure named observation snapshots and M2 tensor packing."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.tasks.tracking.distillation.vae_config import (
  ACTION_DIM,
  DEFAULT_SCHEMA,
  JOINT_DIM,
  DecoderMode,
  VaeSchema,
)


class ObservationValidationError(ValueError):
  """A named student feature snapshot cannot satisfy its schema."""


def _validate_tensor(name: str, value: torch.Tensor, batch: int | None = None) -> None:
  if not isinstance(value, torch.Tensor):
    raise ObservationValidationError(f"{name} must be a torch.Tensor")
  if not value.is_floating_point():
    raise ObservationValidationError(f"{name} must use a floating-point dtype")
  if value.ndim not in (2, 3):
    raise ObservationValidationError(f"{name} must be a rank-2 batch tensor")
  if batch is not None and value.shape[0] != batch:
    raise ObservationValidationError(
      "all snapshot fields must have the same batch size"
    )
  if value.shape[0] <= 0:
    raise ObservationValidationError("snapshot batches must be non-empty")
  if not torch.isfinite(value).all().item():
    raise ObservationValidationError(f"{name} contains non-finite values")


def _validate_vector(
  name: str, value: torch.Tensor, width: int, batch: int | None
) -> None:
  _validate_tensor(name, value, batch)
  if value.ndim != 2 or value.shape[1] != width:
    raise ObservationValidationError(f"{name} must have shape [B, {width}]")


def _validate_rot6d_or_matrix(
  name: str, value: torch.Tensor, batch: int | None
) -> None:
  _validate_tensor(name, value, batch)
  if value.ndim == 2 and value.shape[1] == 6:
    return
  if value.ndim == 3 and value.shape[1:] == (3, 3):
    return
  raise ObservationValidationError(f"{name} must have shape [B, 6] or [B, 3, 3]")


def _pack_rot6d(value: torch.Tensor) -> torch.Tensor:
  """Pack the first two rotation-matrix columns in existing reshape order."""
  if value.ndim == 2:
    return value
  return value[..., :2].reshape(value.shape[0], -1)


@dataclass(frozen=True, slots=True)
class ObservationSnapshot:
  """One raw, aligned observation snapshot before learned normalization.

  All fields must come from the same observation time.  The caller owns that
  alignment, including the previous action being the actually executed,
  normalized action.  This class does not query an environment, resample
  noise/history, or mutate any supplied tensor.
  """

  reference_q: torch.Tensor
  reference_dq: torch.Tensor
  anchor_orientation_error: torch.Tensor
  projected_gravity: torch.Tensor
  gyro: torch.Tensor
  relative_joint_q: torch.Tensor
  joint_dq: torch.Tensor
  previous_action: torch.Tensor

  def __post_init__(self) -> None:
    vector_fields = (
      ("reference_q", self.reference_q, JOINT_DIM),
      ("reference_dq", self.reference_dq, JOINT_DIM),
      ("projected_gravity", self.projected_gravity, 3),
      ("gyro", self.gyro, 3),
      ("relative_joint_q", self.relative_joint_q, JOINT_DIM),
      ("joint_dq", self.joint_dq, JOINT_DIM),
      ("previous_action", self.previous_action, ACTION_DIM),
    )
    batch: int | None = None
    device: torch.device | None = None
    dtype: torch.dtype | None = None
    for name, value, width in vector_fields:
      _validate_vector(name, value, width, batch)
      batch = value.shape[0]
      if device is None:
        device, dtype = value.device, value.dtype
      elif value.device != device or value.dtype != dtype:
        raise ObservationValidationError(
          "all snapshot fields must share device and dtype"
        )
    _validate_rot6d_or_matrix(
      "anchor_orientation_error", self.anchor_orientation_error, batch
    )
    if self.anchor_orientation_error.device != device:
      raise ObservationValidationError(
        "all snapshot fields must share device and dtype"
      )
    if self.anchor_orientation_error.dtype != dtype:
      raise ObservationValidationError(
        "all snapshot fields must share device and dtype"
      )

  @property
  def batch_size(self) -> int:
    return self.reference_q.shape[0]

  @property
  def device(self) -> torch.device:
    return self.reference_q.device

  @property
  def dtype(self) -> torch.dtype:
    return self.reference_q.dtype

  def named_features(self) -> dict[str, torch.Tensor]:
    """Return named raw features without creating or caching transformed data."""
    return {
      "reference_q": self.reference_q,
      "reference_dq": self.reference_dq,
      "anchor_orientation_error": self.anchor_orientation_error,
      "projected_gravity": self.projected_gravity,
      "gyro": self.gyro,
      "relative_joint_q": self.relative_joint_q,
      "joint_dq": self.joint_dq,
      "previous_action": self.previous_action,
    }


# A descriptive alias for callers that use the architecture document's name.
FeatureSnapshot = ObservationSnapshot


@dataclass(frozen=True, slots=True)
class PackedObservationBatch:
  """Packed encoder reference and decoder conditioning tensors."""

  reference: torch.Tensor
  conditioning: torch.Tensor
  schema: VaeSchema

  @property
  def batch_size(self) -> int:
    return self.reference.shape[0]

  @property
  def device(self) -> torch.device:
    return self.reference.device

  @property
  def dtype(self) -> torch.dtype:
    return self.reference.dtype

  def validate(self) -> None:
    """Validate this packed batch without changing it."""
    _validate_vector("reference", self.reference, self.schema.reference_dim, None)
    _validate_vector(
      "conditioning", self.conditioning, self.schema.conditioning_dim, self.batch_size
    )
    if self.reference.device != self.conditioning.device:
      raise ObservationValidationError("packed tensors must share device")
    if self.reference.dtype != self.conditioning.dtype:
      raise ObservationValidationError("packed tensors must share dtype")


def pack_observations(
  snapshot: ObservationSnapshot,
  schema: VaeSchema = DEFAULT_SCHEMA,
) -> PackedObservationBatch:
  """Pack a validated named snapshot according to an immutable schema.

  Reference fields are always ``q, dq, anchor_orientation_error``.  Decoder
  fields are gravity-first for the combined mode and otherwise follow the
  mode's declared ordered field tuple.  ``torch.cat`` creates owned outputs;
  neither the snapshot nor any of its tensors is modified.
  """
  if not isinstance(snapshot, ObservationSnapshot):
    raise ObservationValidationError("snapshot must be an ObservationSnapshot")
  values = snapshot.named_features()
  packed: dict[str, torch.Tensor] = {}
  for field in (*schema.reference_fields, *schema.conditioning_fields):
    value = values.get(field.name)
    if value is None:
      raise ObservationValidationError(f"snapshot is missing feature {field.name!r}")
    packed[field.name] = _pack_rot6d(value)
    if packed[field.name].shape[1] != field.dimension:
      raise ObservationValidationError(
        f"feature {field.name!r} has width {packed[field.name].shape[1]}, "
        f"expected {field.dimension}"
      )
  reference = torch.cat(
    [packed[field.name] for field in schema.reference_fields], dim=1
  )
  conditioning = torch.cat(
    [packed[field.name] for field in schema.conditioning_fields], dim=1
  )
  result = PackedObservationBatch(reference, conditioning, schema)
  result.validate()
  return result


def pack_feature_snapshot(
  snapshot: ObservationSnapshot,
  mode: DecoderMode | str = DecoderMode.GRAVITY,
) -> PackedObservationBatch:
  """Convenience wrapper selecting a schema by explicit decoder mode."""
  from mjlab.tasks.tracking.distillation.vae_config import schema_for_mode

  return pack_observations(snapshot, schema_for_mode(mode))


__all__ = [
  "FeatureSnapshot",
  "ObservationSnapshot",
  "ObservationValidationError",
  "PackedObservationBatch",
  "pack_feature_snapshot",
  "pack_observations",
]
