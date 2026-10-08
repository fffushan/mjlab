"""Owned, deferred per-motion diagnostics; no simulator or model allocation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mjlab.tasks.tracking.distillation.collector import (
  TeacherRoutingError,
  _MotionRecord,
  _observe_motion_tick,
)


def snapshot(device="cpu", kinds=True):
  return SimpleNamespace(
    motion_id=torch.tensor([1, 0, 1, 0], device=device),
    reference_frame=torch.tensor([4, 0, 3, 1], device=device),
    teacher_id="teacher0",
    teacher_code=0,
    initialization_kind=(torch.tensor([2, 1, 9, 0], device=device) if kinds else None),
    segment_age=torch.tensor([0, 25, 24, 26], device=device),
  )


def test_deferred_statistics_are_exact_owned_and_repeatable():
  records = {}
  snap = snapshot()
  codes = torch.tensor([1, 0, 1, 0])
  teacher_rows = torch.tensor([True, False, False, True])
  disagreement = torch.tensor([0.5, 1.0, 1.5, 2.0])
  for _ in range(2):
    _observe_motion_tick(
      records,
      snap,
      teacher_codes=codes,
      teacher_rows=teacher_rows,
      disagreement=disagreement,
      bank=None,
      early_window_frames=25,
    )
  # Original snapshot/action rows may be reused or modified by the environment.
  snap.reference_frame.fill_(99)
  snap.initialization_kind.fill_(1)
  snap.segment_age.fill_(99)
  disagreement.zero_()
  teacher_rows.zero_()
  stats = {motion: record.build(motion) for motion, record in records.items()}
  assert stats[0].samples == stats[1].samples == 4
  assert stats[0].teacher_steps == stats[1].teacher_steps == 2
  assert stats[0].student_steps == stats[1].student_steps == 2
  assert stats[0].disagreement_mean == 1.5
  assert stats[1].disagreement_mean == 1.0
  assert stats[0].reference_frames_observed == stats[1].reference_frames_observed == 2
  assert (stats[0].reference_frame_min, stats[0].reference_frame_max) == (0, 1)
  assert (stats[1].reference_frame_min, stats[1].reference_frame_max) == (3, 4)
  assert stats[0].initialization_samples == {"reference": 2, "unknown": 2}
  assert stats[1].initialization_samples == {"standing": 2, "unknown": 2}
  assert stats[0].early_transition_samples == {"reference": 0, "unknown": 0}
  assert stats[1].early_transition_samples == {"standing": 2, "unknown": 2}
  assert records[0].build(0) == stats[0]  # Flushing twice does not double count.
  assert not records[0]._pending_frames
  assert not records[0]._pending_statistics


def test_mapping_change_is_rejected_before_report_flush():
  records = {}
  snap = snapshot()
  teacher_rows = torch.zeros(4, dtype=torch.bool)
  disagreement = torch.ones(4)
  _observe_motion_tick(
    records,
    snap,
    teacher_codes=torch.tensor([1, 0, 1, 0]),
    teacher_rows=teacher_rows,
    disagreement=disagreement,
    bank=None,
  )
  with pytest.raises(TeacherRoutingError, match="mapping changed"):
    _observe_motion_tick(
      records,
      snap,
      teacher_codes=torch.tensor([2, 0, 2, 0]),
      teacher_rows=teacher_rows,
      disagreement=disagreement,
      bank=None,
    )


def test_mixed_optional_metadata_and_unknown_kinds_keep_legacy_totals():
  record = _MotionRecord(teacher_id=None, teacher_code=0)
  record.observe_tick(torch.tensor([0, 1]))
  record.observe_tick(
    torch.tensor([2, 3]),
    initialization_kind=torch.tensor([7, 8]),
    segment_age=None,
  )
  record.observe_tick(
    torch.tensor([1, 4]),
    initialization_kind=torch.tensor([2, 1]),
    segment_age=torch.tensor([9, 10]),
    early_window_frames=10,
  )
  stats = record.build(0)
  assert stats.reference_frames_observed == 5
  assert stats.initialization_samples == {"unknown": 2, "standing": 1, "reference": 1}
  assert stats.early_transition_samples == {"standing": 1, "reference": 0}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_metadata_recording_defers_cuda_scalar_conversion():
  record = _MotionRecord(teacher_id="a", teacher_code=0)
  frames = torch.arange(1024, device="cuda:0") % 20
  kinds = torch.ones_like(frames)
  ages = torch.zeros_like(frames)
  record.observe_tick(frames, initialization_kind=kinds, segment_age=ages)
  record.build(0)
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
    record.observe_tick(frames, initialization_kind=kinds, segment_age=ages)
  assert sum(e.count for e in prof.key_averages() if e.key == "aten::item") == 0
  stats = record.build(0)
  assert stats.initialization_samples == {"reference": 2048}
  assert stats.early_transition_samples == {"reference": 2048}
