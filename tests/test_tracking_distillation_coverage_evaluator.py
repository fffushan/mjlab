from __future__ import annotations

import pytest
import torch

from mjlab.tasks.tracking.distillation.coverage_evaluator import (
  evaluate_fixed_states,
  make_family_split,
)
from mjlab.tasks.tracking.distillation.coverage_metadata import (
  CoverageBatchMetadata,
  make_row_metadata,
)
from mjlab.tasks.tracking.distillation.model import ConditionalVAE
from mjlab.tasks.tracking.distillation.observations import PackedObservationBatch
from mjlab.tasks.tracking.distillation.storage import LabeledReplayBatch
from mjlab.tasks.tracking.distillation.vae_config import (
  DEFAULT_MODEL_SETTINGS,
  DEFAULT_SCHEMA,
)


def _batch(rows: int = 6) -> LabeledReplayBatch:
  reference = torch.zeros(rows, DEFAULT_SCHEMA.reference_dim)
  conditioning = torch.zeros(rows, DEFAULT_SCHEMA.conditioning_dim)
  return LabeledReplayBatch(
    PackedObservationBatch(reference, conditioning, DEFAULT_SCHEMA),
    torch.zeros(rows, 31),
    torch.zeros(rows, dtype=torch.long),
    torch.zeros(rows, dtype=torch.long),
    torch.arange(rows, dtype=torch.long),
    torch.arange(rows, dtype=torch.long),
    torch.zeros(rows, dtype=torch.long),
  )


def _metadata(rows: int = 6) -> CoverageBatchMetadata:
  return CoverageBatchMetadata(
    tuple(
      make_row_metadata(
        motion_id=str(index % 2),
        trajectory_id=f"family-{index // 2}",
        source_kind="student_difficult",
        frame=index,
        ordinal=index,
      )
      for index in range(rows)
    )
  )


def test_family_split_keeps_related_rows_together() -> None:
  metadata = _metadata()
  assignment = make_family_split(metadata, seed=9)
  assert set(assignment.split_by_family) == {row.family_id for row in metadata}
  for _family, split in assignment.split_by_family.items():
    indices = assignment.row_indices[split]
    families_in_split = {
      item
      for item, item_split in assignment.split_by_family.items()
      if item_split == split
    }
    assert all(metadata.rows[index].family_id in families_in_split for index in indices)


def test_fixed_state_evaluator_is_frozen_and_reports_teacher_fidelity() -> None:
  batch = _batch()
  metadata = _metadata()
  model = ConditionalVAE(DEFAULT_SCHEMA, DEFAULT_MODEL_SETTINGS)
  model.reference_normalizer.freeze()
  model.conditioning_normalizer.freeze()
  assignment = make_family_split(metadata, seed=2, fractions=(0.0, 0.0, 1.0))
  report = evaluate_fixed_states(model, batch, metadata, assignment=assignment)
  assert report.provenance == "teacher_fidelity_on_fixed_states_only_cpu"
  assert report.normalizers_frozen
  assert report.metrics.rows == len(metadata)
  assert report.metrics.families == 6
  assert report.metrics.rms is not None


def test_family_split_is_stable_to_row_order_and_unrelated_families() -> None:
  metadata = _metadata()
  baseline = make_family_split(metadata, seed=17)
  reordered = CoverageBatchMetadata(tuple(reversed(metadata.rows)))
  reordered_assignment = make_family_split(reordered, seed=17)
  assert reordered_assignment.split_by_family == baseline.split_by_family

  extra = make_row_metadata(
    motion_id="unrelated",
    trajectory_id="new-family",
    source_kind="student_difficult",
    frame=0,
    ordinal=0,
  )
  extended = CoverageBatchMetadata(metadata.rows + (extra,))
  extended_assignment = make_family_split(extended, seed=17)
  assert {
    family: extended_assignment.split_by_family[family]
    for family in baseline.split_by_family
  } == baseline.split_by_family


def test_fixed_state_evaluator_preserves_mode_normalizers_and_rng() -> None:
  batch = _batch()
  metadata = _metadata()
  model = ConditionalVAE(DEFAULT_SCHEMA, DEFAULT_MODEL_SETTINGS)
  model.reference_normalizer.freeze()
  model.conditioning_normalizer.freeze()
  model.train()
  normalizer_before = [
    {name: value.clone() for name, value in normalizer.state_dict().items()}
    for normalizer in (model.reference_normalizer, model.conditioning_normalizer)
  ]
  torch.manual_seed(1234)
  rng_before = torch.random.get_rng_state()

  evaluate_fixed_states(
    model,
    batch,
    metadata,
    assignment=make_family_split(metadata, fractions=(0.0, 0.0, 1.0)),
  )

  assert model.training
  for normalizer, expected in zip(
    (model.reference_normalizer, model.conditioning_normalizer),
    normalizer_before,
    strict=True,
  ):
    assert all(
      torch.equal(normalizer.state_dict()[name], value)
      for name, value in expected.items()
    )
  assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_fixed_state_evaluator_rejects_misaligned_assignment() -> None:
  model = ConditionalVAE(DEFAULT_SCHEMA, DEFAULT_MODEL_SETTINGS)
  model.reference_normalizer.freeze()
  model.conditioning_normalizer.freeze()
  metadata = _metadata()
  assignment = make_family_split(metadata, fractions=(0.0, 0.0, 1.0))
  reordered = CoverageBatchMetadata(tuple(reversed(metadata.rows)))
  with pytest.raises(ValueError, match="misaligned"):
    evaluate_fixed_states(model, _batch(), reordered, assignment=assignment)


def test_fixed_state_evaluator_rejects_hardware_fixture() -> None:
  model = ConditionalVAE(DEFAULT_SCHEMA, DEFAULT_MODEL_SETTINGS)
  model.reference_normalizer.freeze()
  model.conditioning_normalizer.freeze()
  try:
    evaluate_fixed_states(model, _batch(), _metadata(), hardware_fixture=True)
  except ValueError as exc:
    assert "hardware fixture" in str(exc)
  else:
    raise AssertionError("hardware fixture must remain held out")
