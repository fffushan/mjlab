"""Vectorised per-tick motion accounting equals the per-motion reference.

The collector attributes one tick's rows to the motions their PRE-step snapshot
named.  The reference below is the element-wise form: for each tick, for each
motion, one mask and one set of counters.  `_observe_motion_tick` sorts the
batch into contiguous per-motion segments instead and must report the same
`MotionCollectionStats` values.

Every test here runs on CPU without a simulator, a teacher or a student.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from mjlab.tasks.tracking.distillation import collector
from mjlab.tasks.tracking.distillation.collector import (
  MotionCollectionStats,
  TeacherRoutingError,
  _build_motion_stats,
  _observe_motion_tick,
)
from mjlab.tasks.tracking.distillation.teachers import TeacherBank

KIND_NAMES = {0: "unknown", 1: "reference", 2: "standing"}
EARLY_WINDOW = 25


def bank() -> TeacherBank:
  """The member `_motion_teacher_label` reads from a real TeacherBank."""
  return cast(
    TeacherBank,
    SimpleNamespace(teacher_ids=("teacher0", "teacher1", "teacher2")),
  )


def make_tick(motion: torch.Tensor, seed: int) -> dict:
  """One PRE-step snapshot plus the per-row values the collector observes."""
  generator = torch.Generator().manual_seed(seed)
  rows = motion.numel()
  return dict(
    snapshot=SimpleNamespace(
      motion_id=motion,
      reference_frame=torch.randint(0, 40, (rows,), generator=generator),
      teacher_id="teacher0",
      teacher_code=0,
      initialization_kind=torch.randint(0, 4, (rows,), generator=generator).masked_fill(
        torch.rand(rows, generator=generator) < 0.2, 9
      ),
      segment_age=torch.randint(0, 40, (rows,), generator=generator),
    ),
    teacher_codes=(motion % 3).to(torch.int64),
    teacher_rows=torch.rand(rows, generator=generator) < 0.5,
    disagreement=torch.rand(rows, generator=generator) * 1e-3,
  )


def reference_stats(ticks, teacher_ids) -> tuple[MotionCollectionStats, ...]:
  """Element-wise per-motion reference: one mask per motion per tick."""
  totals: dict[int, dict] = {}
  for tick in ticks:
    snapshot = tick["snapshot"]
    motion_id = snapshot.motion_id
    for motion in torch.unique(motion_id).tolist():
      rows = motion_id == motion
      code = int(tick["teacher_codes"][rows][0])
      entry = totals.setdefault(
        motion,
        {
          "teacher_id": teacher_ids[code],
          "teacher_code": code,
          "samples": 0,
          "teacher_steps": 0,
          "disagreement_sum": 0.0,
          "disagreement_rows": 0,
          "frames": set(),
          "frame_min": None,
          "frame_max": None,
          "initialization_samples": {},
          "early_transition_samples": {},
        },
      )
      count = int(rows.sum())
      entry["samples"] += count
      entry["disagreement_rows"] += count
      entry["teacher_steps"] += int(tick["teacher_rows"][rows].sum())
      # Each tick's float32 row sum widens to a float64 total in tick order.
      entry["disagreement_sum"] += float(
        tick["disagreement"][rows].sum().to(torch.float64)
      )
      frames = snapshot.reference_frame[rows]
      entry["frames"].update(int(value) for value in torch.unique(frames).tolist())
      lowest, highest = int(frames.min()), int(frames.max())
      entry["frame_min"] = (
        lowest if entry["frame_min"] is None else min(entry["frame_min"], lowest)
      )
      entry["frame_max"] = (
        highest if entry["frame_max"] is None else max(entry["frame_max"], highest)
      )
      kinds = snapshot.initialization_kind[rows]
      ages = snapshot.segment_age[rows]
      for kind in torch.unique(kinds).tolist():
        label = KIND_NAMES.get(int(kind), "unknown")
        selected = kinds == kind
        entry["initialization_samples"][label] = entry["initialization_samples"].get(
          label, 0
        ) + int(selected.sum())
        entry["early_transition_samples"][label] = entry[
          "early_transition_samples"
        ].get(label, 0) + int((selected & (ages < EARLY_WINDOW)).sum())
  return tuple(
    MotionCollectionStats(
      motion_id=motion,
      teacher_id=entry["teacher_id"],
      teacher_code=entry["teacher_code"],
      samples=entry["samples"],
      teacher_steps=entry["teacher_steps"],
      student_steps=entry["samples"] - entry["teacher_steps"],
      boundaries=0,
      disagreement_mean=(
        None
        if entry["disagreement_rows"] == 0
        else entry["disagreement_sum"] / entry["disagreement_rows"]
      ),
      disagreement_rows=entry["disagreement_rows"],
      reference_frames_observed=len(entry["frames"]),
      reference_frame_min=entry["frame_min"],
      reference_frame_max=entry["frame_max"],
      initialization_samples=dict(entry["initialization_samples"]),
      early_transition_samples=dict(entry["early_transition_samples"]),
    )
    for motion, entry in sorted(totals.items())
  )


def observe(ticks) -> dict[int, collector._MotionRecord]:
  records: dict[int, collector._MotionRecord] = {}
  for tick in ticks:
    _observe_motion_tick(
      records,
      tick["snapshot"],
      teacher_codes=tick["teacher_codes"],
      teacher_rows=tick["teacher_rows"],
      disagreement=tick["disagreement"],
      bank=bank(),
      early_window_frames=EARLY_WINDOW,
    )
  return records


def grid_ticks(motions: int = 5, rows_per_motion: int = 8) -> list:
  """5 motions x 8 rows x 3 ticks, each tick a different row order."""
  base = torch.arange(motions, dtype=torch.int64).repeat_interleave(rows_per_motion)
  generator = torch.Generator().manual_seed(11)
  return [
    make_tick(base[torch.randperm(base.numel(), generator=generator)], seed)
    for seed in (1, 2, 3)
  ]


def test_observed_totals_equal_the_per_motion_reference():
  ticks = grid_ticks()
  records = observe(ticks)
  expected = reference_stats(ticks, bank().teacher_ids)
  assert len(expected) == 5
  assert _build_motion_stats(records) == expected


def test_record_ownership_survives_snapshot_reuse():
  ticks = grid_ticks()
  records = observe(ticks)
  expected = reference_stats(ticks, bank().teacher_ids)
  # The environment may rewrite the captured tensors right after the tick.
  for tick in ticks:
    snapshot = tick["snapshot"]
    snapshot.reference_frame.fill_(99)
    snapshot.initialization_kind.fill_(1)
    snapshot.segment_age.fill_(99)
    tick["disagreement"].zero_()
    tick["teacher_rows"].zero_()
    tick["teacher_codes"].zero_()
  assert _build_motion_stats(records) == expected


def test_ragged_and_missing_motions_match_the_reference():
  generator = torch.Generator().manual_seed(5)
  ticks = []
  for seed in (21, 22, 23):
    # Uneven row counts leave ragged segments and drop a motion from a tick.
    motion = torch.randint(0, 5, (40,), generator=generator).to(torch.int64)
    ticks.append(make_tick(motion, seed))
  records = observe(ticks)
  assert _build_motion_stats(records) == reference_stats(ticks, bank().teacher_ids)


def test_teacher_label_lookup_runs_once_per_motion():
  ticks = grid_ticks()
  calls: list[int] = []
  original = collector._motion_teacher_label

  def counting_label(teacher_bank, snapshot, code):
    calls.append(code)
    return original(teacher_bank, snapshot, code)

  collector._motion_teacher_label = counting_label
  try:
    records = observe(ticks)
  finally:
    collector._motion_teacher_label = original
  assert sorted(calls) == [0, 0, 1, 1, 2]  # one lookup per motion, not per tick
  assert len(_build_motion_stats(records)) == 5


def test_two_codes_for_one_motion_in_one_tick_are_rejected():
  motion = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
  tick = make_tick(motion, 31)
  tick["teacher_codes"] = torch.tensor([1, 2, 1, 1], dtype=torch.int64)
  with pytest.raises(TeacherRoutingError) as error:
    observe([tick])
  assert str(error.value) == (
    "motion 0 was labeled by teacher codes 1 and 2 in one collection call; the "
    "motion-to-teacher mapping changed"
  )


def test_changed_motion_to_teacher_mapping_is_rejected_on_the_next_tick():
  motion = torch.tensor([0, 0], dtype=torch.int64)
  first = make_tick(motion, 41)
  first["teacher_codes"] = torch.tensor([1, 1], dtype=torch.int64)
  second = make_tick(motion, 42)
  second["teacher_codes"] = torch.tensor([2, 2], dtype=torch.int64)
  records: dict[int, collector._MotionRecord] = {}
  _observe_motion_tick(
    records,
    first["snapshot"],
    teacher_codes=first["teacher_codes"],
    teacher_rows=first["teacher_rows"],
    disagreement=first["disagreement"],
    bank=bank(),
  )
  with pytest.raises(TeacherRoutingError, match="teacher codes 1 and 2"):
    _observe_motion_tick(
      records,
      second["snapshot"],
      teacher_codes=second["teacher_codes"],
      teacher_rows=second["teacher_rows"],
      disagreement=second["disagreement"],
      bank=bank(),
    )


class HostReadCounter:
  """Count the calls that can wait for the device: item, tolist and cpu."""

  def __init__(self) -> None:
    self.calls = 0
    self._patched: list[tuple[str, object]] = []

  def __enter__(self) -> HostReadCounter:
    for name in ("item", "tolist", "cpu"):
      original = getattr(torch.Tensor, name)

      def counted(tensor, *args, _original=original, **kwargs):
        self.calls += 1
        return _original(tensor, *args, **kwargs)

      self._patched.append((name, original))
      setattr(torch.Tensor, name, counted)
    return self

  def __exit__(self, *exception) -> None:
    for name, original in self._patched:
      setattr(torch.Tensor, name, original)
    self._patched.clear()


def host_reads(motions: int, ticks: int = 3) -> int:
  base = torch.arange(motions, dtype=torch.int64).repeat_interleave(8)
  generator = torch.Generator().manual_seed(3)
  snapshots = [
    make_tick(base[torch.randperm(base.numel(), generator=generator)], seed)
    for seed in range(1, ticks + 1)
  ]
  records: dict[int, collector._MotionRecord] = {}
  with HostReadCounter() as counter:
    for tick in snapshots:
      _observe_motion_tick(
        records,
        tick["snapshot"],
        teacher_codes=tick["teacher_codes"],
        teacher_rows=tick["teacher_rows"],
        disagreement=tick["disagreement"],
        bank=bank(),
        early_window_frames=EARLY_WINDOW,
      )
  return counter.calls


def test_host_reads_per_tick_do_not_grow_with_the_motion_count():
  # The constant is three per tick: `.cpu()` and `.tolist()` for the segment
  # table (motion, lowest code, highest code, row count) and `.cpu()` for the
  # float64 pair (teacher rows, disagreement sum).  No `.item()` remains.
  assert host_reads(20) == host_reads(200) == 3 * 3
