"""Slice 2b: the merge accepts variable-size branch rows and orders them deliberately."""

from __future__ import annotations

import pytest

from mjlab.tasks.tracking.distillation.coverage_metadata import CoverageBatchMetadata, CoverageRowMetadata
from mjlab.tasks.tracking.distillation.coverage_storage import CoverageReplayBatch
from mjlab.tasks.tracking.distillation.sharded_collection import (
  ShardedCollectionError,
  _require_mergeable,
  merge_branch_rows,
)
from mjlab.tasks.tracking.distillation.worker import CollectReply

from test_tracking_distillation_coverage_replay import make_batch


def row(row_id: str, family_id: str = "family-0") -> CoverageRowMetadata:
  return CoverageRowMetadata(
    row_id=row_id,
    family_id=family_id,
    source_kind="teacher_continuation",
    motion_id="tennis_001",
    source_branch="teacher_branch",
    branch_id="branch-1",
    trigger="threshold_crossing",
  )


def branch_batch(count: int) -> CoverageReplayBatch:
  rows = tuple(row(f"row-{i}", f"family-{i}") for i in range(count))
  return CoverageReplayBatch(make_batch(count), CoverageBatchMetadata(rows=rows, batch_id="b"))


def reply(
  *, ticks: int = 1, rows_per_tick: int = 2, branch: CoverageReplayBatch | None = None, branch_rows: int | None = None
) -> CollectReply:
  payload = branch if branch is not None else (branch_batch(branch_rows) if branch_rows else None)
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
    branch_batch=payload,
  )


def test_merge_returns_none_when_no_shard_has_branch_rows() -> None:
  assert merge_branch_rows([reply(), reply()]) is None


def test_merge_returns_none_for_no_replies() -> None:
  assert merge_branch_rows([]) is None


def test_merge_keeps_row_count_and_namespaces_the_rows() -> None:
  merged = merge_branch_rows([reply(), reply(branch_rows=2)])
  assert merged is not None
  assert merged.batch.batch_size == 2
  assert len(merged.metadata.rows) == 2
  # worker index 1 produced them, so the namespace names worker 1
  assert all(r.row_id.startswith("branch-w1/") for r in merged.metadata.rows)


def test_merge_orders_workers_ascending_and_keeps_local_order() -> None:
  merged = merge_branch_rows([
    reply(branch_rows=2),
    reply(branch_rows=2),
  ])
  assert merged is not None
  ids = [r.row_id for r in merged.metadata.rows]
  assert ids == [
    "branch-w0/row-0", "branch-w0/row-1", "branch-w1/row-0", "branch-w1/row-1",
  ]


def test_two_shards_with_identical_local_ids_stay_distinct() -> None:
  merged = merge_branch_rows([reply(branch_rows=3), reply(branch_rows=3)])
  assert merged is not None
  ids = [r.row_id for r in merged.metadata.rows]
  assert len(ids) == len(set(ids)) == 6


def test_merge_rejects_a_namespace_with_a_slash() -> None:
  with pytest.raises(ShardedCollectionError, match="without a slash"):
    merge_branch_rows([reply(branch_rows=1)], namespace_prefix="a/b")


def test_require_mergeable_accepts_a_shard_without_branch_rows() -> None:
  ticks, rows_per_tick = _require_mergeable([reply(ticks=2, rows_per_tick=3), reply(ticks=2, rows_per_tick=3, branch_rows=1)])
  assert (ticks, rows_per_tick) == (2, 3)


def test_require_mergeable_rejects_branch_rows_that_do_not_match_their_metadata() -> None:
  broken = CoverageReplayBatch(make_batch(4), CoverageBatchMetadata(rows=(row("row-0"),), batch_id="b"))
  with pytest.raises(ShardedCollectionError, match="unusable collect reply"):
    _require_mergeable([reply(branch=broken)])


def test_require_mergeable_still_rejects_bad_main_framing() -> None:
  bad = reply(ticks=2, rows_per_tick=3)
  object.__setattr__(bad, "batch", make_batch(5))
  with pytest.raises(ShardedCollectionError, match="ticks of"):
    _require_mergeable([bad])
