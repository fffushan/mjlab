"""Scaling of the balanced replay insertion and sampling paths.

A real 3261-clip run never reported its first iteration because
``BalancedReplayBuffer.insert`` built one full-batch mask, took one host round
trip, and copied every field once PER MOTION.  Later, on the same cohort, the
draw dominated the iteration: one iteration performed about a hundred samples,
and each sample gathered per motion from per-motion tensors, so the number of
gathers grew with the clip count and the workers idled.

These tests pin the replacements.  Insertion: one stable sort, a constant number
of tensor-to-host reads whatever the motion count is.  Sampling: one flat ring
per field and one index draw for all output rows, so both the host round trips
and the advanced-index gathers per sample are constants of the draw mode, not of
the motion count.  Partition and segment semantics are pinned unchanged.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace

import torch
from test_tracking_distillation_balanced_replay import (
  assert_state_equal,
  make_motion_batch,
  retained_tags,
)

from mjlab.tasks.tracking.distillation.balanced_storage import BalancedReplayBuffer
from mjlab.tasks.tracking.distillation.storage import LabeledReplayBatch
from mjlab.tasks.tracking.distillation.vae_config import DEFAULT_SCHEMA

INSERT_HOST_SYNCS = 13
"""Host round trips one provenance-layout insertion performs, at any count.

Nine are batch-validation reads, three more are the provenance-field validation
reads, and the last one is the retained-size sum.  The same insertion on the
legacy layout takes 9 plus that sum.  ``HOST_SYNC_BUDGET`` is the assertion, so
an unrelated change to the validation reads does not fail these tests, while any
return to per-motion host traffic does.
"""

LEGACY_INSERT_HOST_SYNCS = 9

HOST_SYNC_BUDGET = 16

SAMPLE_HOST_SYNCS = 1
"""Host round trips one with-replacement draw performs, at any motion count.

The readiness check reads one device Boolean; the count draw, the offset draw,
the counter updates, and the field gathers all stay on the device, so the host
never waits per motion.
"""

SAMPLE_COVERAGE_HOST_SYNCS = 2
"""One more, for the coverage draw counter ``coverage_drawn``."""

SAMPLE_GATHERS = 13
"""Advanced-index reads one plain draw performs, at any motion count.

Nine are the one-per-field reads that build the batch (``torch.index_select``)
and four are the small per-row index tables.  A draw is one uniform per output
row plus those reads, so the count does not grow with the motion count, which is
the property these tests pin.
"""

SAMPLE_PHASE_GATHERS = 25
"""A phase-balanced draw adds twelve reads of the cached cell grouping."""

SAMPLE_COVERAGE_GATHERS = 33
"""A coverage draw re-reads the drawn kinds before and after forcing a row."""

FRAMES = 12


def multi_motion_buffer(motions: int, quota: int) -> BalancedReplayBuffer:
  """One balanced buffer with ``motions`` equally weighted partitions."""
  return BalancedReplayBuffer(
    capacity=motions * quota,
    schema=DEFAULT_SCHEMA,
    weights={motion: 1.0 for motion in range(motions)},
    teacher_codes={motion: motion for motion in range(motions)},
    frame_counts={motion: FRAMES for motion in range(motions)},
  )


@contextmanager
def counted_host_syncs() -> Iterator[Counter[str]]:
  """Count tensor-to-host round trips: ``.item()``, ``.tolist()``, ``.cpu()``.

  These are the calls that make a device wait for the host.  Patching the
  tensor methods themselves, rather than one helper, also catches a sync that a
  future edit introduces some other way.
  """
  counts: Counter[str] = Counter()
  originals = {name: getattr(torch.Tensor, name) for name in ("item", "tolist", "cpu")}
  try:
    for name, original in originals.items():

      def counted(value, *args, __original=original, __name=name, **kwargs):
        counts[__name] += 1
        return __original(value, *args, **kwargs)

      setattr(torch.Tensor, name, counted)
    yield counts
  finally:
    for name in originals:
      delattr(torch.Tensor, name)


def count_host_syncs(action) -> int:
  with counted_host_syncs() as counts:
    action()
  return sum(counts.values())


def provenance_batch(
  motion_ids: Sequence[int],
  kinds: Sequence[int],
  *,
  start: int = 0,
  frames: Sequence[int] | None = None,
) -> LabeledReplayBatch:
  """One batch carrying standing-start provenance with explicit kind values."""
  batch = make_motion_batch(motion_ids, start=start, frames=frames)
  size = batch.batch_size
  return replace(
    batch,
    initialization_kind=torch.tensor(list(kinds), dtype=torch.int64),
    segment_initial_reference_frame=torch.zeros(size, dtype=torch.int64),
    segment_age=torch.zeros(size, dtype=torch.int64),
  )


def test_insert_host_round_trips_do_not_scale_with_the_motion_count() -> None:
  measured = {}
  for provenance in (False, True):
    for motions in (1, 40, 400):
      buffer = multi_motion_buffer(motions, 4)
      motion_ids = list(range(motions))
      batch = (
        provenance_batch(motion_ids, [1] * motions)
        if provenance
        else make_motion_batch(motion_ids, start=0)
      )
      measured[provenance, motions] = count_host_syncs(
        lambda target=buffer, rows=batch: target.insert(rows)
      )
      assert buffer.size == motions
  motion_counts = (1, 40, 400)
  assert [measured[False, count] for count in motion_counts] == [
    LEGACY_INSERT_HOST_SYNCS
  ] * 3
  assert [measured[True, count] for count in motion_counts] == [INSERT_HOST_SYNCS] * 3
  assert max(measured.values()) <= HOST_SYNC_BUDGET


def test_report_host_round_trips_do_not_scale_with_the_motion_count() -> None:
  measured = {}
  for motions in (40, 400):
    buffer = multi_motion_buffer(motions, 4)
    buffer.insert(provenance_batch(list(range(motions)), [1] * motions))
    measured[motions] = count_host_syncs(buffer.report)
  # One read for the whole retained-kind table, whatever the motion count is.
  assert set(measured.values()) == {1}


def test_resume_rebase_host_round_trips_do_not_scale_with_the_motion_count() -> None:
  measured = {}
  for motions in (40, 400):
    buffer = multi_motion_buffer(motions, 4)
    buffer.insert(
      make_motion_batch(list(range(motions)), start=0, iterations=[1] * motions)
    )
    with counted_host_syncs() as counts:
      changed = buffer.rebase_segment_ids(1, 1000)
    measured[motions] = (sum(counts.values()), changed)
  assert measured == {40: (1, 40), 400: (1, 400)}


def test_multi_motion_batch_lands_in_its_own_partitions_in_batch_order() -> None:
  buffer = multi_motion_buffer(4, quota=4)
  # Interleaved motions: every partition must keep the batch's own row order.
  buffer.insert(make_motion_batch([2, 0, 3, 2, 1, 0, 2, 3, 1, 1], start=0))
  assert retained_tags(buffer, 0) == [1.0, 5.0]
  assert retained_tags(buffer, 1) == [1004.0, 1008.0, 1009.0]
  assert retained_tags(buffer, 2) == [2000.0, 2003.0, 2006.0]
  assert retained_tags(buffer, 3) == [3002.0, 3007.0]
  assert buffer.size == 10
  report = buffer.report()
  assert [motion.retained for motion in report.motions] == [2, 3, 3, 2]
  assert [motion.inserted for motion in report.motions] == [2, 3, 3, 2]
  assert report.ready


def test_oversized_partition_keeps_its_newest_rows_and_leaves_others_alone() -> None:
  buffer = multi_motion_buffer(3, quota=2)
  buffer.insert(make_motion_batch([0, 1, 2, 0, 1, 2], start=0))
  before = buffer.state_dict()
  # Four rows for a two-slot partition: the newest two survive.
  buffer.insert(make_motion_batch([1, 1, 1, 1], start=10))
  assert retained_tags(buffer, 1) == [1012.0, 1013.0]
  after = buffer.state_dict()
  for position in (0, 2):
    assert_state_equal(after["partitions"][position], before["partitions"][position])
  assert [motion.inserted for motion in buffer.stats()] == [2, 6, 2]


def test_single_motion_batch_writes_only_that_partition() -> None:
  buffer = multi_motion_buffer(3, quota=2)
  buffer.insert(make_motion_batch([1, 1], start=0))
  assert retained_tags(buffer, 1) == [1000.0, 1001.0]
  assert retained_tags(buffer, 0) == []
  assert retained_tags(buffer, 2) == []
  assert buffer.size == 2
  assert buffer.unready_motion_ids == (0, 2)
  assert not buffer.ready


def test_all_distinct_batch_writes_one_row_per_partition() -> None:
  buffer = multi_motion_buffer(5, quota=1)
  buffer.insert(make_motion_batch([3, 0, 4, 1, 2], start=0))
  assert [retained_tags(buffer, motion) for motion in range(5)] == [
    [1.0],
    [1003.0],
    [2004.0],
    [3000.0],
    [4002.0],
  ]
  assert buffer.size == 5
  assert buffer.ready


def test_empty_batch_is_a_no_op_for_a_populated_buffer() -> None:
  buffer = multi_motion_buffer(3, quota=2)
  buffer.insert(make_motion_batch([0, 1, 2], start=0))
  before = buffer.state_dict()
  buffer.insert(make_motion_batch([], start=5))
  assert_state_equal(buffer.state_dict(), before)


def test_insert_counts_initialization_kinds_per_motion() -> None:
  buffer = multi_motion_buffer(3, quota=4)
  buffer.insert(provenance_batch([0, 1, 2, 0, 2, 2], [0, 1, 2, 2, 1, 1]))
  by_motion = {item["motion_id"]: item for item in buffer.report().as_dict()["motions"]}
  assert by_motion[0]["inserted_by_initialization"] == {"unknown": 1, "standing": 1}
  assert by_motion[1]["inserted_by_initialization"] == {"reference": 1}
  assert by_motion[2]["inserted_by_initialization"] == {
    "reference": 2,
    "standing": 1,
  }
  assert by_motion[0]["retained_by_initialization"] == {"unknown": 1, "standing": 1}
  assert by_motion[2]["retained_by_initialization"] == {"reference": 2, "standing": 1}
  # Retained kinds follow eviction: two more rows for a three-row partition of
  # quota four evict that partition's oldest row, which was a standing start.
  buffer.insert(provenance_batch([2, 2], [0, 0], start=6))
  by_motion = {item["motion_id"]: item for item in buffer.report().as_dict()["motions"]}
  assert by_motion[2]["inserted_by_initialization"] == {
    "reference": 2,
    "standing": 1,
    "unknown": 2,
  }
  assert by_motion[2]["retained_by_initialization"] == {
    "reference": 2,
    "unknown": 2,
  }
  assert buffer.size == 7


@contextmanager
def counted_tensor_ops() -> Iterator[Counter[str]]:
  """Count host round trips and advanced-index reads during one action.

  ``item``/``tolist``/``cpu`` are the calls that make a device wait for the
  host.  ``__getitem__`` and ``torch.index_select`` cover every advanced-index
  read, which is what a per-motion draw pays once per field per motion.
  """
  counts: Counter[str] = Counter()
  names = ("item", "tolist", "cpu")
  originals = {name: getattr(torch.Tensor, name) for name in names}
  original_getitem = torch.Tensor.__getitem__
  original_index_select = torch.index_select
  try:
    for name, original in originals.items():

      def counted(value, *args, __original=original, __name=name, **kwargs):
        counts[__name] += 1
        return __original(value, *args, **kwargs)

      setattr(torch.Tensor, name, counted)

    def counted_getitem(value, key, __original=original_getitem):
      counts["getitem"] += 1
      return __original(value, key)

    def counted_index_select(value, dim, index, *args, **kwargs):
      counts["index_select"] += 1
      return original_index_select(value, dim, index, *args, **kwargs)

    torch.Tensor.__getitem__ = counted_getitem  # type: ignore[invalid-assignment]
    torch.index_select = counted_index_select  # type: ignore[invalid-assignment]
    yield counts
  finally:
    torch.Tensor.__getitem__ = original_getitem
    torch.index_select = original_index_select
    for name in originals:
      delattr(torch.Tensor, name)


def gathered_reads(counts: Counter[str]) -> int:
  """Advanced-index reads of one draw: ``getitem`` plus ``index_select``."""
  return counts["getitem"] + counts["index_select"]


def sample_buffer(
  motions: int,
  *,
  quota: int = 4,
  phase_bins: int = 0,
  provenance: bool = False,
) -> BalancedReplayBuffer:
  """A ready balanced buffer with ``quota`` retained rows per motion."""
  buffer = BalancedReplayBuffer(
    capacity=motions * quota,
    schema=DEFAULT_SCHEMA,
    weights={motion: 1.0 for motion in range(motions)},
    teacher_codes={motion: motion for motion in range(motions)},
    frame_counts={motion: 100 for motion in range(motions)},
    provenance_enabled=True if provenance else None,
    phase_bins=phase_bins,
    coverage_sampling=provenance,
  )
  rows = list(range(motions))
  for start in range(quota):
    frames = [(motion * 7 + start) % 100 for motion in rows]
    kinds = [(motion + start) % 3 for motion in rows]
    if provenance:
      buffer.insert(provenance_batch(rows, kinds, start=start, frames=frames))
    else:
      buffer.insert(make_motion_batch(rows, start=start, frames=frames))
  assert buffer.ready
  return buffer


def sampled_ops(
  motions: int, *, phase_bins: int = 0, provenance: bool = False
) -> tuple[int, int]:
  """Host round trips and advanced-index reads one draw performs."""
  buffer = sample_buffer(motions, phase_bins=phase_bins, provenance=provenance)
  with counted_tensor_ops() as counts:
    batch = buffer.sample(256, generator=torch.Generator().manual_seed(7))
  assert batch.batch_size == 256
  host = sum(counts[name] for name in ("item", "tolist", "cpu"))
  return host, gathered_reads(counts)


def test_sample_host_round_trips_and_gathers_do_not_scale_with_motion_count() -> None:
  # The drawing path never visits motions one at a time, so growing the cohort
  # by 100x must not add a single host wait or read per draw.
  for motions in (40, 400, 4000):
    assert sampled_ops(motions) == (SAMPLE_HOST_SYNCS, SAMPLE_GATHERS)
    assert sampled_ops(motions, phase_bins=8) == (
      SAMPLE_HOST_SYNCS,
      SAMPLE_PHASE_GATHERS,
    )
    assert sampled_ops(motions, provenance=True) == (
      SAMPLE_COVERAGE_HOST_SYNCS,
      SAMPLE_COVERAGE_GATHERS,
    )
