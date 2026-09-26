"""CPU tests for the versioned M2 schemas and pure packing boundary."""

from __future__ import annotations

import pytest
import torch

from mjlab.tasks.tracking.distillation.observations import (
  ObservationSnapshot,
  ObservationValidationError,
  pack_observations,
)
from mjlab.tasks.tracking.distillation.vae_config import (
  DEFAULT_SCHEMA,
  DecoderMode,
  ModelSettings,
  schema_for_mode,
)


def make_snapshot(batch: int = 2) -> ObservationSnapshot:
  values = torch.arange(batch * 31, dtype=torch.float32).reshape(batch, 31)
  return ObservationSnapshot(
    reference_q=values,
    reference_dq=values + 100,
    anchor_orientation_error=torch.arange(batch * 6, dtype=torch.float32).reshape(
      batch, 6
    ),
    projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]).expand(batch, -1).clone(),
    gyro=torch.full((batch, 3), 2.0),
    relative_joint_q=values + 200,
    joint_dq=values + 300,
    previous_action=values + 400,
  )


@pytest.mark.parametrize(
  ("mode", "names", "width"),
  [
    (
      DecoderMode.GRAVITY,
      (
        "projected_gravity",
        "gyro",
        "relative_joint_q",
        "joint_dq",
        "previous_action",
      ),
      99,
    ),
    (
      DecoderMode.ANCHOR,
      (
        "anchor_orientation_error",
        "gyro",
        "relative_joint_q",
        "joint_dq",
        "previous_action",
      ),
      102,
    ),
    (
      DecoderMode.GRAVITY_ANCHOR,
      (
        "projected_gravity",
        "anchor_orientation_error",
        "gyro",
        "relative_joint_q",
        "joint_dq",
        "previous_action",
      ),
      105,
    ),
  ],
)
def test_all_modes_have_exact_named_order_and_width(
  mode: DecoderMode, names: tuple[str, ...], width: int
) -> None:
  schema = schema_for_mode(mode)
  packed = pack_observations(make_snapshot(), schema)

  assert schema.reference_names == (
    "reference_q",
    "reference_dq",
    "anchor_orientation_error",
  )
  assert schema.conditioning_names == names
  assert (packed.reference.shape, packed.conditioning.shape) == ((2, 68), (2, width))
  assert schema.decoder_reference_conditioned is (mode is not DecoderMode.GRAVITY)


def test_rot6d_uses_existing_first_two_column_reshape() -> None:
  snapshot = make_snapshot(batch=1)
  # Row-major matrix values make an accidental first-six-elements flatten visible.
  matrix = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3)
  snapshot = ObservationSnapshot(
    reference_q=snapshot.reference_q,
    reference_dq=snapshot.reference_dq,
    anchor_orientation_error=matrix,
    projected_gravity=snapshot.projected_gravity,
    gyro=snapshot.gyro,
    relative_joint_q=snapshot.relative_joint_q,
    joint_dq=snapshot.joint_dq,
    previous_action=snapshot.previous_action,
  )

  packed = pack_observations(snapshot, schema_for_mode(DecoderMode.ANCHOR))
  assert packed.reference[0, 62:68].tolist() == [0.0, 1.0, 3.0, 4.0, 6.0, 7.0]
  assert packed.conditioning[0, :6].tolist() == [0.0, 1.0, 3.0, 4.0, 6.0, 7.0]


def test_gravity_default_keeps_reference_isolated_at_packing_boundary() -> None:
  first = make_snapshot()
  second = make_snapshot()
  second = ObservationSnapshot(
    reference_q=second.reference_q + 10000,
    reference_dq=second.reference_dq - 10000,
    anchor_orientation_error=second.anchor_orientation_error + 10000,
    projected_gravity=first.projected_gravity,
    gyro=first.gyro,
    relative_joint_q=first.relative_joint_q,
    joint_dq=first.joint_dq,
    previous_action=first.previous_action,
  )

  first_packed = pack_observations(first, DEFAULT_SCHEMA)
  second_packed = pack_observations(second, DEFAULT_SCHEMA)
  torch.testing.assert_close(first_packed.conditioning, second_packed.conditioning)
  assert not torch.equal(first_packed.reference, second_packed.reference)
  # The default has no reference-feature slice or zero placeholder in conditioning.
  assert DEFAULT_SCHEMA.conditioning_names == (
    "projected_gravity",
    "gyro",
    "relative_joint_q",
    "joint_dq",
    "previous_action",
  )


def test_snapshot_rejects_mixed_batch_device_dtype_and_nonfinite_values() -> None:
  values = make_snapshot()
  with pytest.raises(ObservationValidationError, match="same batch size"):
    ObservationSnapshot(
      reference_q=values.reference_q[:1],
      reference_dq=values.reference_dq,
      anchor_orientation_error=values.anchor_orientation_error,
      projected_gravity=values.projected_gravity,
      gyro=values.gyro,
      relative_joint_q=values.relative_joint_q,
      joint_dq=values.joint_dq,
      previous_action=values.previous_action,
    )
  with pytest.raises(ObservationValidationError, match="non-finite"):
    ObservationSnapshot(
      reference_q=values.reference_q,
      reference_dq=values.reference_dq,
      anchor_orientation_error=values.anchor_orientation_error,
      projected_gravity=values.projected_gravity,
      gyro=values.gyro,
      relative_joint_q=values.relative_joint_q,
      joint_dq=values.joint_dq,
      previous_action=values.previous_action.clone().fill_(float("nan")),
    )


def test_schema_metadata_roundtrip_and_model_defaults_are_serializable() -> None:
  schema = schema_for_mode(DecoderMode.GRAVITY_ANCHOR)
  metadata = schema.compatibility_metadata()
  restored = type(schema).from_metadata(metadata)
  assert restored == schema
  assert metadata["decoder_reference_conditioned"] is True
  assert all(
    field.frame.verification == "declared_unverified"
    for field in (*schema.reference_fields, *schema.conditioning_fields)
  )
  assert ModelSettings().to_metadata() == {
    "latent_dim": 32,
    "hidden_dims": [2048, 1024, 512],
    "activation": "ELU",
    "beta": 0.01,
  }
  assert ModelSettings(hidden_dims=(8, 4), beta=0.2).to_metadata() == {
    "latent_dim": 32,
    "hidden_dims": [8, 4],
    "activation": "ELU",
    "beta": 0.2,
  }
