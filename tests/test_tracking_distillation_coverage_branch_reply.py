"""Slice 2a of the coverage execution path: branch rows in the reply, and namespacing.

Main rows stay tick major in ``CollectReply.batch``.  Branch rows are variable size,
so they travel in their own channel with their own metadata.  Two shards can derive
the same local identifier from the same clip, so the parent namespaces every reply
before it inserts anything.
"""

from __future__ import annotations

import pytest
from test_tracking_distillation_coverage_replay import make_batch

from mjlab.tasks.tracking.distillation.coverage_metadata import (
  CoverageBatchMetadata,
  CoverageMetadataError,
  CoverageRowMetadata,
  namespace_coverage_batch,
  namespace_coverage_rows,
)
from mjlab.tasks.tracking.distillation.coverage_storage import CoverageReplayBatch
from mjlab.tasks.tracking.distillation.worker import (
  CollectReply,
  WorkerError,
  validate_collect_reply,
)


def row(
  row_id: str = "row-0", family_id: str = "family-0", **kwargs
) -> CoverageRowMetadata:
  return CoverageRowMetadata(
    row_id=row_id,
    family_id=family_id,
    source_kind="student_difficult",
    motion_id="tennis_001",
    **kwargs,
  )


def reply(
  *, ticks: int = 2, rows_per_tick: int = 3, branch: CoverageReplayBatch | None = None
) -> CollectReply:
  return CollectReply(
    call_id=1,
    iteration=1,
    ticks=ticks,
    samples=ticks * rows_per_tick,
    teacher_steps=0,
    student_steps=0,
    disagreement_mean=0.0,
    diagnostics=(),
    motion_stats=(),
    boundaries=(),
    rows_per_tick=rows_per_tick,
    batch=make_batch(ticks * rows_per_tick),
    branch_batch=branch,
  )


def test_branch_channel_defaults_to_absent() -> None:
  assert reply().branch_batch is None


def test_namespace_prefixes_row_family_and_branch_ids() -> None:
  original = row(
    branch_id="branch-7", source_branch="teacher_branch", trigger="threshold_crossing"
  )
  [namespaced] = namespace_coverage_rows((original,), namespace="w0")
  assert namespaced.row_id == "w0/row-0"
  assert namespaced.family_id == "w0/family-0"
  assert namespaced.branch_id == "w0/branch-7"
  # everything else is preserved
  assert namespaced.source_branch == "teacher_branch"
  assert namespaced.trigger == "threshold_crossing"
  assert namespaced.motion_id == original.motion_id


def test_namespace_leaves_a_missing_branch_id_as_none() -> None:
  [namespaced] = namespace_coverage_rows((row(),), namespace="w0")
  assert namespaced.branch_id is None


def test_two_workers_cannot_collide_on_the_same_local_ids() -> None:
  local = (row("row-3", "family-3"), row("row-4", "family-4"))
  first = namespace_coverage_rows(local, namespace="w0")
  second = namespace_coverage_rows(local, namespace="w1")
  assert {r.row_id for r in first}.isdisjoint({r.row_id for r in second})
  assert {r.family_id for r in first}.isdisjoint({r.family_id for r in second})


def test_namespace_rejects_a_slash_or_surrounding_space() -> None:
  with pytest.raises(CoverageMetadataError):
    namespace_coverage_rows((row(),), namespace="w/0")
  with pytest.raises(CoverageMetadataError):
    namespace_coverage_rows((row(),), namespace=" w0")


def test_valid_main_only_reply_passes_validation() -> None:
  validate_collect_reply(reply())


def test_valid_reply_with_branch_rows_passes_validation() -> None:
  branch = CoverageReplayBatch(make_batch(1), CoverageBatchMetadata(rows=(row(),)))
  validate_collect_reply(reply(branch=branch))


def test_main_rows_that_do_not_match_ticks_times_rows_per_tick_are_rejected() -> None:
  broken = reply()
  object.__setattr__(broken, "batch", make_batch(5))  # 5 != 2 * 3
  with pytest.raises(WorkerError, match="ticks times rows_per_tick"):
    validate_collect_reply(broken)


def test_branch_rows_that_do_not_match_their_metadata_are_rejected() -> None:
  branch = CoverageReplayBatch(
    make_batch(4), CoverageBatchMetadata(rows=(row(),), batch_id="b1")
  )
  with pytest.raises(WorkerError, match="branch rows do not match their metadata"):
    validate_collect_reply(reply(branch=branch))


def test_namespacing_leaves_a_usable_batch_id_and_row_ids_agree() -> None:
  # The metadata owns its batch id.  This slice namespaces rows only, and it must
  # not disturb the batch id or the row count that the id covers.
  batch = CoverageBatchMetadata(rows=(row(),))
  assert batch.batch_id
  namespaced = namespace_coverage_batch(batch, namespace="w2")
  assert namespaced.batch_id
  assert len(namespaced.rows) == len(batch.rows)
  assert namespaced.rows[0].row_id == "w2/row-0"
