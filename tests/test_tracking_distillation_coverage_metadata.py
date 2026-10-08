import pytest

from mjlab.tasks.tracking.distillation.coverage_metadata import (
  CoverageBatchMetadata,
  CoverageMetadataError,
  CoverageRowMetadata,
  make_row_metadata,
  own_metadata,
  stable_family_id,
)


def test_owned_row_metadata_has_stable_family_and_batch_identity() -> None:
  first = make_row_metadata(
    motion_id="tennis_001",
    trajectory_id="trajectory-7",
    source_kind="student_difficult",
    frame=12,
    ordinal=0,
    trigger="threshold_crossing",
    severity_score=1.5,
    severity_band="moderate",
    trigger_reason="anchor_gravity",
  )
  second = make_row_metadata(
    motion_id="tennis_001",
    trajectory_id="trajectory-7",
    source_kind="teacher_continuation",
    frame=13,
    ordinal=1,
    source_branch="teacher_branch",
    branch_id="branch-2",
    branch_age_steps=1,
  )
  batch = own_metadata([first, second])
  assert (
    first.family_id
    == second.family_id
    == stable_family_id(motion_id="tennis_001", trajectory_id="trajectory-7")
  )
  assert CoverageBatchMetadata.from_dict(batch.as_dict()).digest == batch.digest
  assert batch.batch_id


def test_metadata_rejects_outcome_like_or_malformed_provenance() -> None:
  with pytest.raises(CoverageMetadataError):
    CoverageRowMetadata("r", "f", "teacher_continuation", "m")
  with pytest.raises(CoverageMetadataError):
    CoverageRowMetadata("r", "f", "student_difficult", "m", severity_score=float("nan"))
  row = make_row_metadata(
    motion_id="m",
    trajectory_id="t",
    source_kind="student_difficult",
    frame=0,
    ordinal=0,
  )
  payload = row.as_dict()
  payload["teacher_outcome"] = "success"
  with pytest.raises(CoverageMetadataError):
    CoverageRowMetadata.from_dict(payload)


def test_metadata_batch_owns_rows_and_rejects_duplicate_ids() -> None:
  row = make_row_metadata(
    motion_id="m",
    trajectory_id="t",
    source_kind="student_difficult",
    frame=0,
    ordinal=0,
  )
  with pytest.raises(CoverageMetadataError):
    CoverageBatchMetadata((row, row))
  assert len(own_metadata(iter((row,)))) == 1
