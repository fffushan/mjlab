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
  PREVIEW_DIM,
  PREVIEW_SCHEMA_VERSION,
  REFERENCE_DIM,
  REFERENCE_DIM_WITH_PREVIEW,
  REFERENCE_PREVIEW_FIELD,
  SCHEMA_VERSION,
  DecoderMode,
  ModelSettings,
  VaeSchema,
  make_schema,
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


def add_preview(
  snapshot: ObservationSnapshot, preview: torch.Tensor | None = None
) -> ObservationSnapshot:
  """Copy a snapshot with a marked preview, so a mis-ordered pack is visible."""
  batch = snapshot.batch_size
  if preview is None:
    preview = (
      torch.arange(batch * PREVIEW_DIM, dtype=torch.float32).reshape(batch, PREVIEW_DIM)
      + 500
    )
  return ObservationSnapshot(
    reference_q=snapshot.reference_q,
    reference_dq=snapshot.reference_dq,
    anchor_orientation_error=snapshot.anchor_orientation_error,
    projected_gravity=snapshot.projected_gravity,
    gyro=snapshot.gyro,
    relative_joint_q=snapshot.relative_joint_q,
    joint_dq=snapshot.joint_dq,
    previous_action=snapshot.previous_action,
    motion_lookahead=preview,
  )


def test_the_frozen_version_one_layout_is_unchanged() -> None:
  """Existing checkpoints, statistics and bundles must keep loading."""
  assert SCHEMA_VERSION == 1
  assert DEFAULT_SCHEMA.version == SCHEMA_VERSION
  assert (DEFAULT_SCHEMA.reference_dim, DEFAULT_SCHEMA.conditioning_dim) == (68, 99)
  assert REFERENCE_DIM == 68
  assert REFERENCE_DIM_WITH_PREVIEW == 130
  assert PREVIEW_DIM == 62
  # A default schema is version 1 even though a preview layout now exists.
  assert make_schema().version == SCHEMA_VERSION
  assert make_schema().reference_names == (
    "reference_q",
    "reference_dq",
    "anchor_orientation_error",
  )


def test_the_preview_layout_adds_one_reference_frame_only() -> None:
  schema = make_schema(preview=True)
  assert schema.version == PREVIEW_SCHEMA_VERSION
  assert schema.reference_dim == REFERENCE_DIM_WITH_PREVIEW == 130
  # The decoder stays proprioception-only: the preview reaches it through the latent.
  assert schema.conditioning_dim == DEFAULT_SCHEMA.conditioning_dim == 99
  assert schema.conditioning_names == DEFAULT_SCHEMA.conditioning_names
  assert schema.reference_names == (
    "reference_q",
    "reference_dq",
    "anchor_orientation_error",
    REFERENCE_PREVIEW_FIELD,
  )
  assert schema.mode is DecoderMode.GRAVITY


def test_the_preview_pack_keeps_the_version_one_prefix_and_appends_the_preview() -> (
  None
):
  """The first 68 values must not move, so the version-1 layout stays the prefix."""
  snapshot = make_snapshot(2)
  marked = add_preview(snapshot)
  version_one = pack_observations(snapshot, make_schema())
  version_two = pack_observations(marked, make_schema(preview=True))

  assert version_two.reference.shape == (2, 130)
  assert torch.equal(version_two.reference[:, :REFERENCE_DIM], version_one.reference)
  assert torch.equal(version_two.reference[:, REFERENCE_DIM:], marked.motion_lookahead)
  # The preview carries the future joint positions and then the future velocities.
  assert version_two.reference[0, 68].item() == 500.0
  assert version_two.reference[0, 68 + 31].item() == 500.0 + 31.0
  assert torch.equal(version_two.conditioning, version_one.conditioning)


def test_the_preview_pack_refuses_a_snapshot_that_omits_the_preview() -> None:
  with pytest.raises(ObservationValidationError, match=REFERENCE_PREVIEW_FIELD):
    pack_observations(make_snapshot(2), make_schema(preview=True))


def test_a_version_one_pack_ignores_a_supplied_preview() -> None:
  packed = pack_observations(add_preview(make_snapshot(2)), make_schema())
  assert packed.reference.shape == (2, 68)


def test_a_preview_of_the_wrong_width_is_refused() -> None:
  with pytest.raises(ObservationValidationError, match="motion_lookahead"):
    add_preview(make_snapshot(2), torch.zeros(2, 31))


def test_a_version_two_schema_needs_the_preview_field_in_the_last_position() -> None:
  base = make_schema()
  with pytest.raises(ValueError, match="must end with"):
    VaeSchema(
      mode=base.mode,
      version=PREVIEW_SCHEMA_VERSION,
      reference_fields=base.reference_fields,
      conditioning_fields=base.conditioning_fields,
    )


def test_a_version_two_schema_rejects_a_mis_sized_preview_field() -> None:
  base = make_schema()
  wrong = tuple(base.reference_fields) + (
    base.reference_fields[0].__class__(
      REFERENCE_PREVIEW_FIELD, 30, base.reference_fields[0].frame
    ),
  )
  with pytest.raises(ValueError, match="must be 62 wide"):
    VaeSchema(
      mode=base.mode,
      version=PREVIEW_SCHEMA_VERSION,
      reference_fields=wrong,
      conditioning_fields=base.conditioning_fields,
    )


def test_an_unsupported_schema_version_is_refused() -> None:
  base = make_schema()
  with pytest.raises(ValueError, match="unsupported schema version"):
    VaeSchema(
      mode=base.mode,
      version=3,
      reference_fields=base.reference_fields,
      conditioning_fields=base.conditioning_fields,
    )


def test_preview_metadata_roundtrips_and_crosses_no_version_boundary() -> None:
  version_one = make_schema()
  version_two = make_schema(preview=True)
  assert VaeSchema.from_metadata(version_two.compatibility_metadata()) == version_two
  assert VaeSchema.from_metadata(version_one.compatibility_metadata()) == version_one
  # A version-1 checkpoint must not load into a version-2 run, and the reverse.
  with pytest.raises(ValueError, match="does not match"):
    version_one.assert_compatible(version_two.compatibility_metadata())
  with pytest.raises(ValueError, match="does not match"):
    version_two.assert_compatible(version_one.compatibility_metadata())
  assert version_two.compatibility_metadata()["reference_dim"] == 130
  assert version_two.compatibility_metadata()["schema_version"] == 2


def test_the_preview_layout_is_available_for_every_decoder_mode() -> None:
  for mode in (
    DecoderMode.GRAVITY,
    DecoderMode.ANCHOR,
    DecoderMode.GRAVITY_ANCHOR,
    DecoderMode.GRAVITY_TWIST,
  ):
    base = make_schema(mode)
    with_preview = make_schema(mode, preview=True)
    assert with_preview.reference_dim == base.reference_dim + PREVIEW_DIM
    assert with_preview.conditioning_names == base.conditioning_names
    assert VaeSchema.from_metadata(with_preview.compatibility_metadata()) == (
      with_preview
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
    (
      DecoderMode.GRAVITY_TWIST,
      (
        "projected_gravity",
        "root_linear_velocity",
        "gyro",
        "relative_joint_q",
        "joint_dq",
        "previous_action",
      ),
      102,
    ),
  ],
)
def test_all_modes_have_exact_named_order_and_width(
  mode: DecoderMode, names: tuple[str, ...], width: int
) -> None:
  schema = schema_for_mode(mode)
  if mode is DecoderMode.GRAVITY_TWIST:
    base = make_snapshot()
    base = ObservationSnapshot(
      reference_q=base.reference_q,
      reference_dq=base.reference_dq,
      anchor_position_error=torch.full((2, 3), 7.0),
      anchor_orientation_error=base.anchor_orientation_error,
      projected_gravity=base.projected_gravity,
      root_linear_velocity=torch.full((2, 3), 8.0),
      gyro=base.gyro,
      relative_joint_q=base.relative_joint_q,
      joint_dq=base.joint_dq,
      previous_action=base.previous_action,
    )
  else:
    base = make_snapshot()
  packed = pack_observations(base, schema)

  if mode is DecoderMode.GRAVITY_TWIST:
    assert schema.reference_names == (
      "reference_q",
      "reference_dq",
      "anchor_position_error",
      "anchor_orientation_error",
    )
    assert packed.reference.shape == (2, 71)
    assert packed.reference[0, 62:65].tolist() == [7.0, 7.0, 7.0]
    assert packed.conditioning[0, 3:6].tolist() == [8.0, 8.0, 8.0]
  else:
    assert schema.reference_names == (
      "reference_q",
      "reference_dq",
      "anchor_orientation_error",
    )
  assert schema.conditioning_names == names
  assert packed.conditioning.shape == (2, width)
  assert schema.decoder_reference_conditioned is (
    mode in (DecoderMode.ANCHOR, DecoderMode.GRAVITY_ANCHOR)
  )


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
