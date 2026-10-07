"""Pure CPU coverage replay tests."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Literal

import pytest
import torch

import mjlab.tasks.tracking.distillation.coverage_storage as coverage_storage
from mjlab.tasks.tracking.distillation.coverage_metadata import (
  CoverageBatchMetadata,
  CoverageRowMetadata,
  CoverageSource,
)
from mjlab.tasks.tracking.distillation.coverage_policy import CoveragePolicy
from mjlab.tasks.tracking.distillation.coverage_storage import (
  CoverageReplayNotReadyError,
  ProtectedCoverageReplayBuffer,
)
from mjlab.tasks.tracking.distillation.observations import PackedObservationBatch
from mjlab.tasks.tracking.distillation.storage import (
  LabeledReplayBatch,
  ReplayValidationError,
)
from mjlab.tasks.tracking.distillation.vae_config import DEFAULT_SCHEMA


def make_batch(count: int, *, start: int = 0) -> LabeledReplayBatch:
  reference = torch.arange(
    count * DEFAULT_SCHEMA.reference_dim, dtype=torch.float32
  ).reshape(count, DEFAULT_SCHEMA.reference_dim)
  conditioning = torch.zeros(count, DEFAULT_SCHEMA.conditioning_dim)
  action = torch.zeros(count, DEFAULT_SCHEMA.action_dim)
  return LabeledReplayBatch(
    PackedObservationBatch(reference, conditioning, DEFAULT_SCHEMA),
    action,
    torch.zeros(count, dtype=torch.int64),
    torch.zeros(count, dtype=torch.int64),
    torch.arange(start, start + count, dtype=torch.int64),
    torch.arange(start, start + count, dtype=torch.int64),
    torch.zeros(count, dtype=torch.int64),
  )


def make_metadata(
  count: int,
  *,
  start: int = 0,
  family_offset: int = 0,
  severity: str = "strong",
  source: str = "student_difficult",
  phase_offset: int = 0,
) -> CoverageBatchMetadata:
  rows = tuple(
    CoverageRowMetadata(
      row_id=f"row-{start + i}",
      family_id=f"family-{family_offset + i}",
      source_kind=source,  # type: ignore[arg-type]
      source_branch="teacher_branch" if source == "teacher_continuation" else "main",
      branch_id=f"branch-{family_offset + i}"
      if source == "teacher_continuation"
      else None,
      motion_id="0",
      source_frame=start + i,
      phase_bin=(phase_offset + i) % 5,
      severity_score=2.0 if severity == "strong" else 1.1,
      severity_band=severity,  # type: ignore[arg-type]
    )
    for i in range(count)
  )
  return CoverageBatchMetadata(rows)


def add_rows(
  buffer: ProtectedCoverageReplayBuffer,
  count: int,
  *,
  start: int = 0,
  family_offset: int = 0,
  severity: str = "strong",
  source: str = "student_difficult",
  phase_offset: int = 0,
):
  return buffer.insert(
    make_batch(count, start=start),
    make_metadata(
      count,
      start=start,
      family_offset=family_offset,
      severity=severity,
      source=source,
      phase_offset=phase_offset,
    ),
  )


def make_buffer(capacity: int = 400, *, seed: int = 7) -> ProtectedCoverageReplayBuffer:
  return ProtectedCoverageReplayBuffer(
    capacity,
    DEFAULT_SCHEMA,
    {0: 1.0},
    seed=seed,
    teacher_codes={0: 0},
    frame_counts={0: 1_000_000},
  )


def test_total_capacity_split_and_sparse_cell_rounding() -> None:
  buffer = ProtectedCoverageReplayBuffer(
    7,
    DEFAULT_SCHEMA,
    {0: 1.0, 1: 2.0},
    policy=CoveragePolicy.balance_recovery(),
    teacher_codes={0: 0, 1: 1},
    frame_counts={0: 1_000_000, 1: 1_000_000},
  )
  assert buffer.capacity == 7
  assert buffer.archive_capacity == 0
  assert buffer.recent_capacity == 7
  assert sum(buffer.motion_quotas.values()) == 7
  assert sum(buffer._cell_quotas.values()) == 0


def test_nominal_flooding_cannot_fifo_evict_strong_protected_cell() -> None:
  buffer = make_buffer(400)
  strong = add_rows(buffer, 5, family_offset=0, severity="strong")
  assert len(strong.archive_row_ids) == 5
  strong_ids = set(strong.archive_row_ids)
  add_rows(buffer, 200, start=1000, family_offset=1000, severity="moderate")
  archive_ids = {
    row.metadata.row_id
    for families in buffer._cells.values()
    for bucket in families.values()
    for row in bucket.rows
  }
  assert strong_ids <= archive_ids


def test_family_replacement_and_deepcopy_ownership() -> None:
  buffer = make_buffer(160)
  first = add_rows(buffer, 1, family_offset=1, severity="strong")
  assert first.archive_row_ids
  report = buffer.insert(
    make_batch(1, start=2),
    make_metadata(1, start=2, family_offset=2, severity="strong"),
  )
  assert report.archive_row_ids or report.rejected_row_ids
  assert (
    buffer.report().counters.get("family_replacements", 0)
    + buffer.report().counters.get("family_reservoir_rejections", 0)
    >= 1
  )
  second_batch = make_batch(1, start=3)
  buffer.insert(
    second_batch, make_metadata(1, start=3, family_offset=3, severity="strong")
  )
  second_batch.reference[0, 0] = 9999.0
  assert all(
    row.fields["reference"][0].item() != 9999.0 for row in buffer._archive_rows(0)
  )


def test_same_row_dedup_and_reproducible_next_draws() -> None:
  buffer = make_buffer(400)
  add_rows(buffer, 5, family_offset=0)
  duplicate = buffer.insert(make_batch(1), make_metadata(1, family_offset=0))
  assert duplicate.duplicate_row_ids == ("row-0",)
  add_rows(
    buffer,
    5,
    start=100,
    family_offset=100,
    source="teacher_continuation",
    severity="moderate",
  )
  add_rows(
    buffer,
    5,
    start=200,
    family_offset=200,
    source="finite_post_failure",
    severity="moderate",
  )
  state = copy.deepcopy(buffer.state_dict())
  restored = make_buffer(400)
  restored.load_state_dict(state)
  one = buffer.sample_with_metadata(10)
  two = restored.sample_with_metadata(10)
  assert one.metadata.canonical_json() == two.metadata.canonical_json()
  assert torch.equal(one.batch.reference, two.batch.reference)
  assert len({row.row_id for row in one.metadata}) == 10


def test_invalid_late_restore_is_transactional_and_shortage_is_fail_closed() -> None:
  buffer = make_buffer(400)
  add_rows(buffer, 5, family_offset=0)
  before = copy.deepcopy(buffer.state_dict())
  invalid = copy.deepcopy(before)
  invalid["recent"][0]["storage"]["reference"][0, 0] = float("nan")
  with pytest.raises(ReplayValidationError):
    buffer.load_state_dict(invalid)
  after = buffer.state_dict()
  torch.testing.assert_close(
    before["recent"][0]["storage"]["reference"],
    after["recent"][0]["storage"]["reference"],
    equal_nan=True,
  )
  tiny = make_buffer(20)
  add_rows(tiny, 1)
  with pytest.raises(CoverageReplayNotReadyError):
    tiny.sample_with_metadata(2)


def test_family_priority_reservoir_rejects_repeated_loser() -> None:
  buffer = make_buffer(160)
  add_rows(buffer, 1, family_offset=0, severity="strong")
  rejected_family = None
  for family in range(1, 64):
    report = add_rows(buffer, 1, start=family, family_offset=family, severity="strong")
    if report.rejected_row_ids:
      rejected_family = family
      break
  assert rejected_family is not None
  repeated = add_rows(
    buffer, 1, start=1000, family_offset=rejected_family, severity="strong"
  )
  assert repeated.rejected_row_ids == ("row-1000",)


def test_bounded_dedup_and_call_watermark() -> None:
  buffer = make_buffer(20)
  for call in range(32):
    add_rows(buffer, 1, start=call, family_offset=call)
  assert len(buffer._dedup_ids) <= buffer._dedup_capacity
  batch = make_batch(1, start=500)
  metadata = make_metadata(1, start=500, family_offset=500)
  buffer.insert(batch, metadata, collection_call_id=10)
  with pytest.raises(ReplayValidationError):
    buffer.insert(batch, metadata, collection_call_id=10)
  with pytest.raises(ReplayValidationError):
    buffer.insert(batch, metadata, collection_call_id=9)


def test_optional_protocol_validation_and_metadata_routing_check() -> None:
  buffer = make_buffer(100)
  batch = make_batch(1)
  buffer.validate_batch(batch)
  row = make_metadata(1).rows[0]
  bad_row = CoverageRowMetadata(
    row_id=row.row_id,
    family_id=row.family_id,
    source_kind="student_difficult",
    motion_id="9",
    source_frame=0,
  )
  with pytest.raises(ReplayValidationError):
    buffer.validate_batch(batch, CoverageBatchMetadata((bad_row,)))
  assert buffer.insert_recent(
    batch, collection_call_id=1, row_namespace="default"
  ).fresh_row_ids


def test_explicit_generator_and_global_family_cap() -> None:
  buffer = make_buffer(400)
  batch = make_batch(40)
  base = make_metadata(40, severity="strong")
  same_family = CoverageBatchMetadata(
    tuple(
      CoverageRowMetadata(
        row_id=row.row_id,
        family_id="one-family",
        source_kind="student_difficult",
        motion_id="0",
        source_frame=row.source_frame,
        phase_bin=row.phase_bin,
        severity_band="strong",
        severity_score=2.0,
      )
      for row in base
    )
  )
  buffer.insert(batch, same_family)
  generator = torch.Generator(device="cpu").manual_seed(123)
  with pytest.raises(CoverageReplayNotReadyError):
    buffer.sample(40, generator=generator)
  other = make_buffer(400)
  other.insert(batch, same_family)
  generator2 = torch.Generator(device="cpu").manual_seed(123)
  with pytest.raises(CoverageReplayNotReadyError):
    other.sample(40, generator=generator2)
  assert torch.equal(generator.get_state(), generator2.get_state())


def test_unknown_motion_and_bad_phase_reject_before_mutation() -> None:
  buffer = make_buffer(100)
  batch = make_batch(2)
  valid = make_metadata(2)
  bad = CoverageRowMetadata(
    row_id=valid.rows[1].row_id,
    family_id=valid.rows[1].family_id,
    source_kind="student_difficult",
    motion_id="99",
    source_frame=1,
    phase_bin=5,
    severity_band="strong",
  )
  with pytest.raises(ReplayValidationError):
    buffer.insert(batch, CoverageBatchMetadata((valid.rows[0], bad)))
  assert buffer.size == 0
  assert buffer._last_collection_call is None


def test_recent_sample_does_not_snapshot_full_ring_and_bulk_insert_is_one_call(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  buffer = make_buffer(100)
  batch = make_batch(8)
  metadata = make_metadata(8)
  partition = buffer._recent[0]
  calls = 0
  original_insert = partition.buffer.insert

  def counted_insert(batch: LabeledReplayBatch) -> None:
    nonlocal calls
    calls += 1
    return original_insert(batch)

  monkeypatch.setattr(partition.buffer, "insert", counted_insert)
  report = buffer.insert(batch, metadata)
  assert report.fresh_row_ids
  assert calls == 1

  def fail_state_dict():
    raise AssertionError("sample must not snapshot the full recent ring")

  monkeypatch.setattr(partition.buffer, "state_dict", fail_state_dict)
  assert buffer.sample(1).batch_size == 1


def test_recent_bulk_insert_does_not_allocate_row_wrappers(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  buffer = make_buffer(100)

  def fail_row_view(*args, **kwargs):
    raise AssertionError("recent bulk insertion must stay batched")

  monkeypatch.setattr(coverage_storage, "_row_view", fail_row_view)
  report = buffer.insert_recent(make_batch(32), make_metadata(32))
  assert len(report.fresh_row_ids) == 32
  assert len(buffer._recent[0]) == 32


def test_recent_metadata_tracks_legacy_ring_on_wrap_and_oversize() -> None:
  buffer = make_buffer(20)
  buffer.insert_recent(make_batch(10), make_metadata(10))
  buffer.insert_recent(make_batch(8, start=10), make_metadata(8, start=10))
  sampled = buffer.sample_with_metadata(16)
  for index, row in enumerate(sampled.metadata):
    assert row.source_frame == int(sampled.batch.reference_frame[index].item())
    assert (
      sampled.batch.reference[index, 0].item()
      == (row.source_frame % 10) * DEFAULT_SCHEMA.reference_dim
    )
  buffer.insert_recent(make_batch(25, start=25), make_metadata(25, start=25))
  sampled = buffer.sample_with_metadata(16)
  for index, row in enumerate(sampled.metadata):
    assert row.source_frame == int(sampled.batch.reference_frame[index].item())
    assert (
      sampled.batch.reference[index, 0].item()
      == (row.source_frame - 25) * DEFAULT_SCHEMA.reference_dim
    )


def test_partially_empty_motion_partitions_restore_and_accept_later_rows() -> None:
  buffer = ProtectedCoverageReplayBuffer(
    20,
    DEFAULT_SCHEMA,
    {0: 1.0, 1: 1.0},
    policy=CoveragePolicy.balance_recovery(),
    teacher_codes={0: 0, 1: 1},
    frame_counts={0: 100, 1: 100},
  )
  first = make_batch(1)
  buffer.insert_recent(first, make_metadata(1))
  state = copy.deepcopy(buffer.state_dict())

  restored = ProtectedCoverageReplayBuffer(
    20,
    DEFAULT_SCHEMA,
    {0: 1.0, 1: 1.0},
    policy=CoveragePolicy.balance_recovery(),
    teacher_codes={0: 0, 1: 1},
    frame_counts={0: 100, 1: 100},
  )
  restored.load_state_dict(state)
  assert restored._recent[0].buffer.device == torch.device("cpu")
  assert restored._recent[1].buffer.device == torch.device("cpu")
  assert len(restored._recent[1]) == 0

  second = replace(
    make_batch(1, start=1),
    motion_id=torch.ones(1, dtype=torch.int64),
    teacher_id=torch.ones(1, dtype=torch.int64),
  )
  second_metadata = CoverageBatchMetadata(
    (replace(make_metadata(1, start=1).rows[0], motion_id="1"),)
  )
  restored.insert_recent(second, second_metadata)
  sampled = restored.sample(2, generator=torch.Generator().manual_seed(3))
  assert sorted(sampled.motion_id.tolist()) == [0, 1]


def test_archive_draw_uses_hierarchical_category_phase_severity_rounding() -> None:
  buffer = make_buffer(4000)
  batch = make_batch(10)
  rows: list[CoverageRowMetadata] = []
  source_specs: tuple[
    tuple[CoverageSource, Literal["main", "teacher_branch"], str | None, int], ...
  ] = (
    ("student_difficult", "main", None, 5),
    ("teacher_continuation", "teacher_branch", "branch", 4),
    ("finite_post_failure", "main", None, 1),
  )
  ordinal = 0
  for source_kind, source_branch, branch_prefix, count in source_specs:
    for phase in range(count):
      rows.append(
        CoverageRowMetadata(
          row_id=f"hier-{ordinal}",
          family_id=f"hier-family-{ordinal}",
          source_kind=source_kind,
          motion_id="0",
          source_branch=source_branch,
          branch_id=None if branch_prefix is None else f"{branch_prefix}-{ordinal}",
          source_frame=ordinal,
          phase_bin=phase,
          severity_score=1.1,
          severity_band="moderate",
        )
      )
      ordinal += 1
  buffer.insert(batch, CoverageBatchMetadata(tuple(rows)))
  selected = buffer._sample_archive(0, 10, set(), {})
  assert len(selected) == 10
  by_source = {
    source: sum(_row.metadata.source_kind == source for _row in selected)
    for source in (
      "student_difficult",
      "teacher_continuation",
      "finite_post_failure",
    )
  }
  assert by_source == {
    "student_difficult": 5,
    "teacher_continuation": 4,
    "finite_post_failure": 1,
  }
  assert sum(row.metadata.phase_bin == 4 for row in selected) == 1


def test_fractional_and_nonboolean_metadata_reject_before_any_mutation() -> None:
  buffer = make_buffer(100)
  baseline_rng = buffer._rng.get_state().clone()
  source = make_metadata(1).rows[0]
  invalid_rows = (
    replace(source, phase_bin=0.5, row_id="bad-phase"),
    replace(source, terminal=1, row_id="bad-terminal"),
    replace(source, event_frame=True, row_id="bad-event"),
  )
  for invalid_row in invalid_rows:
    with pytest.raises(ReplayValidationError):
      buffer.insert_committed(
        make_batch(1),
        CoverageBatchMetadata((invalid_row,)),
        collection_call_id=1,
        row_namespace="default",
      )
    assert buffer.size == 0
    assert buffer._last_collection_call is None
    assert torch.equal(buffer._rng.get_state(), baseline_rng)


def test_rebase_updates_recent_tensor_ring_and_archive() -> None:
  buffer = make_buffer(400)
  batch = make_batch(8)
  metadata = make_metadata(8)
  buffer.insert(batch, metadata)
  changed = buffer.rebase_segment_ids(0, 100)
  assert changed == 13
  active = buffer._recent[0].active_rows()
  storage = buffer._recent[0].buffer._storage
  assert storage is not None
  indices = [index for index, _ in active]
  assert torch.equal(
    storage["episode_id"][indices], torch.arange(100, 108, dtype=torch.int64)
  )
  assert buffer.max_valid_segment_id() == 107


def test_invalid_rng_restore_is_fully_transactional() -> None:
  buffer = make_buffer(100)
  add_rows(buffer, 8)
  before = buffer.state_dict()
  invalid = copy.deepcopy(before)
  invalid["rng_state"] = torch.zeros(1, dtype=torch.uint8)
  with pytest.raises(ReplayValidationError):
    buffer.load_state_dict(invalid)
  after = buffer.state_dict()
  assert before["fresh_count"] == after["fresh_count"]
  assert before["draw_count"] == after["draw_count"]
  assert torch.equal(before["rng_state"], after["rng_state"])


def test_metadata_free_unknown_motion_does_not_consume_call_watermark() -> None:
  buffer = make_buffer(100)
  batch = make_batch(2)
  batch.motion_id[1] = 99
  with pytest.raises(ReplayValidationError):
    buffer.insert_recent(batch, collection_call_id=7, row_namespace="default")
  assert buffer._last_collection_call is None
  assert buffer.size == 0


def test_committed_union_admits_nominal_recent_and_difficult_archive_once() -> None:
  buffer = ProtectedCoverageReplayBuffer(
    400,
    DEFAULT_SCHEMA,
    {0: 1.0},
    policy=CoveragePolicy.balance_recovery(),
    row_namespace="ns",
    teacher_codes={0: 0},
    frame_counts={0: 1_000_000},
  )
  batch = make_batch(2)
  nominal = CoverageRowMetadata(
    row_id="ns:1:nominal",
    family_id="family-nominal",
    source_kind="student_difficult",
    motion_id="0",
    source_frame=0,
    severity_band="nominal",
  )
  difficult = CoverageRowMetadata(
    row_id="ns:1:difficult",
    family_id="family-difficult",
    source_kind="student_difficult",
    motion_id="0",
    source_frame=1,
    phase_bin=1,
    severity_band="strong",
    severity_score=2.0,
  )
  report = buffer.insert_committed(
    batch,
    CoverageBatchMetadata((nominal, difficult)),
    collection_call_id=1,
    row_namespace="ns",
  )
  assert report.fresh_row_ids == ("ns:1:nominal", "ns:1:difficult")
  assert "ns:1:nominal" not in report.archive_row_ids


def test_provenance_v1_restore_preserves_nested_ring_layout() -> None:
  buffer = make_buffer(100)
  source = make_batch(5)
  provenance = replace(
    source,
    initialization_kind=torch.ones(5, dtype=torch.int64),
    segment_initial_reference_frame=torch.arange(5, dtype=torch.int64),
    segment_age=torch.arange(5, dtype=torch.int64) + 10,
  )
  buffer.insert(provenance, make_metadata(5))
  state = copy.deepcopy(buffer.state_dict())
  assert state["layout"] == "provenance-v1"
  assert state["recent"][0]["layout"] == "provenance-v1"

  restored = make_buffer(100)
  restored.load_state_dict(state)
  assert restored._provenance_enabled is True
  assert restored._recent[0].buffer._provenance_enabled is True
  assert restored.state_dict()["recent"][0]["layout"] == "provenance-v1"
  sampled = restored.sample_with_metadata(5)
  provenance_batch = sampled.batch.provenance()
  assert provenance_batch is not None
  assert torch.equal(
    torch.sort(provenance_batch[1]).values,
    torch.arange(5, dtype=torch.int64),
  )


def test_recent_restore_does_not_construct_row_wrappers_for_entire_ring(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  buffer = make_buffer(100)
  buffer.insert_recent(make_batch(32), make_metadata(32))
  state = copy.deepcopy(buffer.state_dict())
  original_row = coverage_storage._StoredRow

  def fail_row_wrapper(*args, **kwargs):
    raise AssertionError("recent restore must not construct one wrapper per row")

  monkeypatch.setattr(coverage_storage, "_StoredRow", fail_row_wrapper)
  restored = make_buffer(100)
  restored.load_state_dict(state)
  monkeypatch.setattr(coverage_storage, "_StoredRow", original_row)


def test_nonfinite_conditioning_in_recent_checkpoint_is_rejected_transactionally() -> (
  None
):
  buffer = make_buffer(100)
  add_rows(buffer, 5)
  before = copy.deepcopy(buffer.state_dict())
  invalid = copy.deepcopy(before)
  invalid["recent"][0]["storage"]["conditioning"][0, 0] = float("nan")
  with pytest.raises(ReplayValidationError, match="conditioning"):
    buffer.load_state_dict(invalid)
  after = buffer.state_dict()
  recent_state = after["recent"][0]
  size, next_index = recent_state["size"], recent_state["next"]
  active = (torch.arange(size) + (next_index - size)) % buffer.recent_quotas[0]
  torch.testing.assert_close(
    before["recent"][0]["storage"]["conditioning"][active],
    recent_state["storage"]["conditioning"][active],
  )
  assert before["fresh_count"] == after["fresh_count"]


def test_nested_provenance_layout_mismatch_rejects_without_mutation() -> None:
  buffer = make_buffer(100)
  source = make_batch(3)
  buffer.insert(
    replace(
      source,
      initialization_kind=torch.ones(3, dtype=torch.int64),
      segment_initial_reference_frame=torch.zeros(3, dtype=torch.int64),
      segment_age=torch.zeros(3, dtype=torch.int64),
    ),
    make_metadata(3),
  )
  before = copy.deepcopy(buffer.state_dict())
  invalid = copy.deepcopy(before)
  invalid["recent"][0].pop("layout")
  with pytest.raises(ReplayValidationError, match="layout"):
    buffer.load_state_dict(invalid)
  assert buffer.state_dict()["recent"][0]["layout"] == "provenance-v1"


def test_routing_maps_reject_wrong_teacher_and_out_of_range_frame_before_watermark() -> (
  None
):
  buffer = ProtectedCoverageReplayBuffer(
    100,
    DEFAULT_SCHEMA,
    {0: 1.0},
    policy=CoveragePolicy.balance_recovery(),
    row_namespace="route",
    teacher_codes={0: 7},
    frame_counts={0: 4},
  )
  batch = replace(
    make_batch(1),
    teacher_id=torch.tensor([0], dtype=torch.int64),
    reference_frame=torch.tensor([4], dtype=torch.int64),
  )
  metadata = CoverageBatchMetadata(
    (
      CoverageRowMetadata(
        row_id="route:1:bad",
        family_id="family",
        source_kind="student_difficult",
        motion_id="0",
        source_frame=4,
        severity_band="strong",
      ),
    )
  )
  with pytest.raises(ReplayValidationError):
    buffer.insert_committed(
      batch, metadata, collection_call_id=1, row_namespace="route"
    )
  assert buffer._last_collection_call is None
  assert buffer.size == 0


def test_committed_old_source_call_id_cannot_be_replayed_as_new_call() -> None:
  buffer = ProtectedCoverageReplayBuffer(
    100,
    DEFAULT_SCHEMA,
    {0: 1.0},
    policy=CoveragePolicy.balance_recovery(),
    row_namespace="route",
    teacher_codes={0: 0},
    frame_counts={0: 100},
  )
  batch = make_batch(1)
  metadata = CoverageBatchMetadata(
    (
      CoverageRowMetadata(
        row_id="route:1:row",
        family_id="family",
        source_kind="student_difficult",
        motion_id="0",
        source_frame=0,
        severity_band="strong",
      ),
    )
  )
  buffer.insert_committed(batch, metadata, collection_call_id=1, row_namespace="route")
  with pytest.raises(ReplayValidationError, match="namespace and collection"):
    buffer.insert_committed(
      batch, metadata, collection_call_id=2, row_namespace="route"
    )
  assert buffer._last_collection_call == 1
  assert buffer.report().fresh_rows == 1

# --- independent specification tests for the defects the replay lane recorded ---
# Written by the integration owner on 2026-10-07.  Each test states the required
# behaviour.  Defect record: MAX-HANDOFF.md "Replay max owner".


def test_replay_defect_nan_conditioning_is_rejected() -> None:
  """A non-finite recent conditioning tensor must not load."""
  buffer = make_buffer(100)
  buffer.insert_recent(make_batch(5), make_metadata(5))
  before = copy.deepcopy(buffer.state_dict())
  invalid = copy.deepcopy(before)
  invalid["recent"][0]["storage"]["conditioning"][0, 0] = float("nan")
  with pytest.raises(ReplayValidationError):
    buffer.load_state_dict(invalid)
  after = buffer.state_dict()
  assert after["recent"][0]["size"] == before["recent"][0]["size"]
  # The free slots of the ring hold NaN as an empty-slot sentinel, so a plain
  # torch.equal comparison always reports a difference.  Compare NaN-aware, and
  # require the same NaN count so that a written NaN cannot pass unnoticed.
  for name, value in before["recent"][0]["storage"].items():
    other = after["recent"][0]["storage"][name]
    if value.dtype.is_floating_point:
      assert torch.equal(torch.isnan(value), torch.isnan(other))
      assert torch.allclose(value, other, equal_nan=True, rtol=0, atol=0)
    else:
      assert torch.equal(value, other)


def test_replay_defect_infinite_conditioning_is_rejected() -> None:
  """A positive infinity in recent conditioning must not load."""
  buffer = make_buffer(100)
  buffer.insert_recent(make_batch(5), make_metadata(5))
  invalid = copy.deepcopy(buffer.state_dict())
  invalid["recent"][0]["storage"]["conditioning"][0, 0] = float("inf")
  with pytest.raises(ReplayValidationError):
    buffer.load_state_dict(invalid)


def test_replay_defect_valid_provenance_state_round_trips() -> None:
  """A valid provenance-v1 state loads, and restore sets every identity field."""
  buffer = make_buffer(100)
  source = make_batch(5)
  provenance = replace(
    source,
    initialization_kind=torch.ones(5, dtype=torch.int64),
    segment_initial_reference_frame=torch.arange(5, dtype=torch.int64),
    segment_age=torch.arange(5, dtype=torch.int64) + 3,
  )
  buffer.insert_recent(provenance, make_metadata(5))
  state = copy.deepcopy(buffer.state_dict())
  assert state["recent"][0]["layout"] == "provenance-v1"

  restored = make_buffer(100)
  restored.load_state_dict(state)

  # Defect: the nested ring restore left device, dtype and provenance as None.
  assert restored._device is not None
  assert restored._dtype is not None
  assert restored._provenance_enabled is not None
  nested = restored._recent[0].buffer
  assert nested._device is not None
  assert nested._dtype is not None
  assert nested._provenance_enabled is not None

  # Defect: the nested ring flags were absent from the saved state.
  again = restored.state_dict()
  assert again["recent"][0]["layout"] == "provenance-v1"
  assert again["recent"][0]["dtype"] == state["recent"][0]["dtype"]
  assert again["recent"][0]["device"] == state["recent"][0]["device"]
  assert again["recent"][0]["size"] == state["recent"][0]["size"]

  # The restored buffer must still sample provenance rows.
  sampled = restored.sample_with_metadata(5)
  provenance_batch = sampled.batch.provenance()
  assert provenance_batch is not None


def test_replay_defect_legacy_state_round_trips_without_provenance() -> None:
  """A legacy state keeps provenance disabled and still restores."""
  buffer = make_buffer(100)
  buffer.insert_recent(make_batch(5), make_metadata(5))
  state = copy.deepcopy(buffer.state_dict())
  # A legacy state carries no layout key at all; the key appears for provenance only.
  assert "layout" not in state["recent"][0]

  restored = make_buffer(100)
  restored.load_state_dict(state)
  assert restored._provenance_enabled is False
  assert restored._recent[0].buffer._provenance_enabled is False
  assert "layout" not in restored.state_dict()["recent"][0]

def test_v4_partition_policy_seam_records_the_buffer_it_belongs_to() -> None:
  """The seam a v4 coverage checkpoint reads through _coverage_replay_policy.

  The legacy ``ReplayPolicy.from_buffer`` records only the balanced and the FIFO
  layouts, so the enabled buffer must answer for itself.  The checkpoint compares
  the result with the stored payload, so a wrong order or a missing motion here
  would silently accept a mismatched resume.
  """
  from mjlab.tasks.tracking.distillation.checkpoint import _coverage_replay_policy

  from mjlab.tasks.tracking.distillation.cohort_contract import ReplayPolicy

  buffer = make_buffer(capacity=400)
  policy = ReplayPolicy.from_dict(
    buffer.coverage_replay_policy(), "live coverage replay policy"
  )
  assert policy.kind == "coverage-protected-replay"
  assert policy.capacity == buffer.capacity
  assert policy.motion_ids == tuple(int(motion) for motion in buffer.motion_weights)
  assert policy.weights == tuple(
    float(buffer.motion_weights[motion]) for motion in policy.motion_ids
  )
  assert policy.quotas == tuple(
    int(buffer.motion_quotas[motion]) for motion in policy.motion_ids
  )
  assert policy.teacher_codes == tuple(
    int(buffer.teacher_codes[motion]) for motion in policy.motion_ids
  )
  assert policy.frame_counts == tuple(
    int(buffer.frame_counts[motion]) for motion in policy.motion_ids
  )
  assert sum(policy.quotas) == policy.capacity
  # The round trip and the checkpoint's own hook must both accept it.
  assert type(policy).from_dict(policy.as_dict(), "round trip") == policy
  assert _coverage_replay_policy(buffer) == policy
