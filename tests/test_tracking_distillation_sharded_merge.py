"""Merging sharded collection replies, tested as exact arithmetic.

The merge is where a sharded run can be subtly wrong while every counter still
looks plausible: row order decides which records a full replay forgets first,
and a mean over rows has to be weighted by rows rather than averaged across
shards.  Both are pinned here exactly, without any process or simulator.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from mjlab.tasks.tracking.distillation.collector import (
  CollectionConfig,
  EvaluationSegment,
  MotionCollectionStats,
  SegmentBoundary,
)
from mjlab.tasks.tracking.distillation.observations import PackedObservationBatch
from mjlab.tasks.tracking.distillation.sharded_collection import (
  WORKER_SEGMENT_STRIDE,
  ShardedCollection,
  ShardedCollectionError,
  merge_collection_replies,
  merge_evaluations,
  merge_motion_stats,
  union_motion_frames,
)
from mjlab.tasks.tracking.distillation.storage import LabeledReplayBatch
from mjlab.tasks.tracking.distillation.vae_config import DEFAULT_SCHEMA
from mjlab.tasks.tracking.distillation.worker import CollectReply, EvaluateReply


def _batch(rows: int, worker_index: int) -> LabeledReplayBatch:
  """Rows tagged so the merge order is readable: value = worker*1000 + row."""
  values = torch.arange(rows, dtype=torch.float32) + worker_index * 1000
  batch = LabeledReplayBatch(
    observations=PackedObservationBatch(
      reference=values.unsqueeze(1),
      conditioning=torch.zeros(rows, 2),
      schema=DEFAULT_SCHEMA,
    ),
    teacher_action=values.unsqueeze(1),
    motion_id=torch.zeros(rows, dtype=torch.long),
    teacher_id=torch.zeros(rows, dtype=torch.long),
    reference_frame=torch.arange(rows, dtype=torch.long),
    episode_id=torch.zeros(rows, dtype=torch.long),
    collector_iteration=torch.zeros(rows, dtype=torch.long),
  )
  return batch


def _stats(
  motion_id: int,
  *,
  samples: int,
  disagreement_mean: float | None,
  disagreement_rows: int,
  frames_observed: int = 0,
  frame_min: int | None = None,
  frame_max: int | None = None,
  teacher_code: int = 0,
  teacher_id: str | None = "tennis_000",
) -> MotionCollectionStats:
  return MotionCollectionStats(
    motion_id=motion_id,
    teacher_id=teacher_id,
    teacher_code=teacher_code,
    samples=samples,
    teacher_steps=samples,
    student_steps=0,
    boundaries=0,
    disagreement_mean=disagreement_mean,
    disagreement_rows=disagreement_rows,
    reference_frames_observed=frames_observed,
    reference_frame_min=frame_min,
    reference_frame_max=frame_max,
  )


def _reply(
  *,
  worker_index: int,
  ticks: int = 2,
  rows_per_tick: int = 2,
  samples: int | None = None,
  disagreement_mean: float = 0.0,
  motion_stats: tuple[MotionCollectionStats, ...] = (),
  boundaries: tuple[SegmentBoundary, ...] = (),
  diagnostics: tuple[str, ...] = (),
  batch: LabeledReplayBatch | None = None,
) -> CollectReply:
  rows = ticks * rows_per_tick
  return CollectReply(
    call_id=worker_index + 1,
    iteration=4,
    ticks=ticks,
    samples=rows if samples is None else samples,
    teacher_steps=rows,
    student_steps=0,
    disagreement_mean=disagreement_mean,
    diagnostics=diagnostics,
    motion_stats=motion_stats,
    boundaries=boundaries,
    rows_per_tick=rows_per_tick,
    batch=_batch(rows, worker_index) if batch is None else batch,
  )


def test_rows_merge_tick_major_with_workers_ascending_within_a_tick() -> None:
  """Tick k from every worker precedes tick k+1.

  This is the FIFO ordering a full replay evicts by.  Sorting by worker
  instead would pass every counter check and quietly change which rows the
  replay forgets first.
  """
  reply0 = _reply(worker_index=0)
  reply1 = _reply(worker_index=1)

  result = merge_collection_replies(
    [reply0, reply1], iteration=4, device=torch.device("cpu")
  )

  assert result.fresh_data is not None
  merged = result.fresh_data.batch
  # worker 0 rows 0-1, worker 1 rows 0-1, then the second tick of each.
  assert merged.observations.reference.flatten().tolist() == [
    0.0,
    1.0,
    1000.0,
    1001.0,
    2.0,
    3.0,
    1002.0,
    1003.0,
  ]
  assert result.fresh_data.update_id == 4
  assert merged.batch_size == 8


def test_counters_are_a_union_of_ticks_and_sums_of_rows() -> None:
  """Ticks stay a union so samples-per-tick keeps its meaning.

  Summing ticks would inflate the reported simulation length by the worker
  count, and mean-of-means disagreement would weight a five-row shard the same
  as a five-thousand-row one.
  """
  reply0 = _reply(worker_index=0, samples=2, disagreement_mean=0.1)
  reply1 = _reply(worker_index=1, samples=6, disagreement_mean=0.7)

  result = merge_collection_replies(
    [reply0, reply1], iteration=4, device=torch.device("cpu")
  )

  assert result.ticks == 2
  assert result.samples == 8
  assert result.teacher_steps == 8
  assert result.student_steps == 0
  assert result.disagreement_mean == pytest.approx((0.1 * 2 + 0.7 * 6) / 8)


def test_per_motion_statistics_are_summed_and_row_weighted() -> None:
  """Per-motion counters add, disagreement weights by rows, frames take bounds."""
  merged = merge_motion_stats(
    [
      [
        _stats(
          3,
          samples=2,
          disagreement_mean=0.2,
          disagreement_rows=2,
          frames_observed=10,
          frame_min=5,
          frame_max=40,
        ),
      ],
      [
        _stats(
          3,
          samples=6,
          disagreement_mean=0.6,
          disagreement_rows=6,
          frames_observed=60,
          frame_min=1,
          frame_max=90,
        ),
      ],
    ]
  )

  assert len(merged) == 1
  stats = merged[0]
  assert stats.motion_id == 3
  assert stats.samples == 8
  assert stats.teacher_steps == 8
  assert stats.disagreement_rows == 8
  assert stats.disagreement_mean == pytest.approx((0.2 * 2 + 0.6 * 6) / 8)
  # A distinct-frame count cannot be summed: two shards observing frame 40
  # would be counted twice and could exceed the clip length.
  assert stats.reference_frames_observed == 60
  assert (stats.reference_frame_min, stats.reference_frame_max) == (1, 90)


def test_per_motion_statistics_refuse_two_teachers_for_one_motion() -> None:
  """One motion cannot be attributed to two teachers by different shards."""
  with pytest.raises(ShardedCollectionError) as error:
    merge_motion_stats(
      [
        [_stats(1, samples=2, disagreement_mean=0.0, disagreement_rows=2)],
        [
          _stats(
            1,
            samples=2,
            disagreement_mean=0.0,
            disagreement_rows=2,
            teacher_code=1,
            teacher_id="tennis_001",
          )
        ],
      ]
    )
  assert "two different teachers" in str(error.value)


def test_boundaries_are_worker_ascending_and_tagged_with_their_shard() -> None:
  """Merged boundaries name their shard, because indices are per-shard."""
  boundary0 = SegmentBoundary((0,), "reset", (7,), (8,), (0,), (1,), (2,), (0,))
  boundary1 = SegmentBoundary((0,), "timeout", (3,), (4,), (0,), (0,), (2,), (0,))

  result = merge_collection_replies(
    [
      _reply(worker_index=0, boundaries=(boundary0,)),
      _reply(worker_index=1, boundaries=(boundary1,)),
    ],
    iteration=4,
    device=torch.device("cpu"),
  )

  assert [item.worker_index for item in result.boundaries] == [0, 1]
  assert [item.reason for item in result.boundaries] == ["reset", "timeout"]
  assert all(item.worker_index is not None for item in result.boundaries)


def test_diagnostics_are_prefixed_with_the_worker_that_reported_them() -> None:
  """Two shards reporting the same text must stay distinguishable."""
  result = merge_collection_replies(
    [
      _reply(worker_index=0, diagnostics=("standing reset fraction",)),
      _reply(worker_index=1, diagnostics=("standing reset fraction",)),
    ],
    iteration=4,
    device=torch.device("cpu"),
  )

  assert result.diagnostics == (
    "worker 0: standing reset fraction",
    "worker 1: standing reset fraction",
  )


def test_mismatched_framing_is_refused_instead_of_reordered() -> None:
  """A shard that stepped a different number of ticks is an error, not data."""
  with pytest.raises(ShardedCollectionError) as error:
    merge_collection_replies(
      [_reply(worker_index=0), _reply(worker_index=1, ticks=3)],
      iteration=4,
      device=torch.device("cpu"),
    )
  assert "ticks=" in str(error.value)


def test_replies_that_disagree_about_provenance_are_refused() -> None:
  """Provenance is all-or-nothing: a partial merge would invent history.

  Standing-start provenance says how long a segment has been running and where
  it started.  Filling it in for the shards that omitted it would make the
  merged rows claim a reference history they do not have.
  """
  from dataclasses import replace

  reply0 = _reply(worker_index=0)
  reply1 = _reply(worker_index=1)
  rows = 4
  reply1 = replace(
    reply1,
    batch=replace(
      reply1.batch,
      initialization_kind=torch.zeros(rows, dtype=torch.long),
      segment_initial_reference_frame=torch.zeros(rows, dtype=torch.long),
      segment_age=torch.arange(rows, dtype=torch.long),
    ),
  )

  with pytest.raises(ShardedCollectionError) as error:
    merge_collection_replies([reply0, reply1], iteration=4, device=torch.device("cpu"))
  assert "provenance" in str(error.value)


def test_segment_ids_are_namespaced_per_worker_before_merge() -> None:
  """Two simulators' local segment ids must not collide in one replay.

  Each worker's simulator numbers its own segments from zero, so merging them
  unchanged would make worker 0's segment 7 and worker 1's segment 7 a single
  identity — and the resume rebase, which adds one namespace to an iteration's
  ids, could never tell them apart afterwards.
  """
  rows = 4
  reply0 = _reply(worker_index=0)
  reply1 = _reply(worker_index=1)
  reply0 = replace(reply0, batch=replace(reply0.batch, episode_id=torch.arange(rows)))
  reply1 = replace(reply1, batch=replace(reply1.batch, episode_id=torch.arange(rows)))

  result = merge_collection_replies(
    [reply0, reply1], iteration=4, device=torch.device("cpu")
  )
  assert result.fresh_data is not None
  merged_ids = result.fresh_data.batch.episode_id
  assert len(set(merged_ids.tolist())) == rows * 2
  # Worker 1's block sits exactly one stride above worker 0's, so a later
  # namespace rebase keeps the two shards' ids disjoint.
  stride = WORKER_SEGMENT_STRIDE
  assert merged_ids.tolist() == [
    0,
    1,
    stride,
    stride + 1,
    2,
    3,
    stride + 2,
    stride + 3,
  ]


def test_segment_ids_that_would_overflow_their_worker_block_are_refused() -> None:
  """A shard whose local ids exceed the block width is refused, not overlapped."""
  rows = 2
  reply = _reply(worker_index=0)
  reply = replace(
    reply,
    batch=replace(reply.batch, episode_id=torch.full((rows,), WORKER_SEGMENT_STRIDE)),
  )
  with pytest.raises(ShardedCollectionError) as error:
    merge_collection_replies([reply], iteration=4, device=torch.device("cpu"))
  assert "do not fit this run's per-worker block" in str(error.value)


def test_reset_telemetry_from_every_shard_reaches_the_merged_result() -> None:
  """Standing-reset counters must survive the merge.

  They are reported per motion and per reset kind; dropping them (which
  happens silently, because the merged fields default to empty) would make a
  sharded run's standing-reset report under-report what actually happened.
  """
  reply0 = _reply(worker_index=0)
  reply1 = _reply(worker_index=1)
  reply0 = replace(
    reply0,
    eligible_resets={2: 3},
    initialization_resets={2: {"reference": 1, "standing": 2}},
  )
  reply1 = replace(
    reply1,
    eligible_resets={2: 4, 5: 1},
    initialization_resets={2: {"standing": 4}},
  )

  result = merge_collection_replies(
    [reply0, reply1], iteration=4, device=torch.device("cpu")
  )

  assert result.eligible_resets == {2: 7, 5: 1}
  assert result.initialization_resets == {2: {"reference": 1, "standing": 6}}


def test_frame_coverage_is_the_exact_union_of_shard_frame_sets() -> None:
  """Disjoint shard coverage must add up, not collapse to the largest shard.

  Worker 0 observing frames 0-9 and worker 1 observing 10-19 covers twenty
  frames; reporting the larger shard count would report ten and understate
  coverage against the clip length.
  """
  first = [
    _stats(1, samples=2, disagreement_mean=0.0, disagreement_rows=2, frames_observed=10)
  ]
  second = [
    _stats(1, samples=2, disagreement_mean=0.0, disagreement_rows=2, frames_observed=10)
  ]

  stats = merge_motion_stats(
    [first, second],
    frames_by_motion={1: tuple(range(20))},
  )

  assert stats[0].reference_frames_observed == 20
  assert stats[0].reference_frame_min == 0
  assert stats[0].reference_frame_max == 19


def test_union_motion_frames_combines_overlapping_shards_exactly() -> None:
  """Overlap must be counted once, which is what makes the count exact."""
  reply0 = replace(_reply(worker_index=0), motion_frames=((1, (0, 1, 2)),))
  reply1 = replace(_reply(worker_index=1), motion_frames=((1, (2, 3)),))

  assert union_motion_frames([reply0, reply1]) == {1: (0, 1, 2, 3)}


class _StubPool:
  """A pool that hands back prepared replies, for source-level tests only."""

  def __init__(self, replies=(), *, failure: Exception | None = None) -> None:
    self._replies = tuple(replies)
    self._failure = failure
    self.closed = False
    self.calls = 0

  def collect(self, **_kwargs):
    self.calls += 1
    if self._failure is not None:
      raise self._failure
    return self._replies

  def close(self) -> None:
    self.closed = True


def _schema_batch(rows: int, worker_index: int) -> LabeledReplayBatch:
  """A schema-correct batch, for tests that insert into a real replay."""
  return LabeledReplayBatch(
    observations=PackedObservationBatch(
      reference=torch.full((rows, DEFAULT_SCHEMA.reference_dim), float(worker_index)),
      conditioning=torch.zeros(rows, DEFAULT_SCHEMA.conditioning_dim),
      schema=DEFAULT_SCHEMA,
    ),
    teacher_action=torch.zeros(rows, DEFAULT_SCHEMA.action_dim),
    motion_id=torch.zeros(rows, dtype=torch.long),
    teacher_id=torch.zeros(rows, dtype=torch.long),
    reference_frame=torch.zeros(rows, dtype=torch.long),
    episode_id=torch.zeros(rows, dtype=torch.long),
    collector_iteration=torch.zeros(rows, dtype=torch.long),
  )


def _source(replies, *, replay, failure=None) -> ShardedCollection:
  from mjlab.tasks.tracking.distillation.model import ConditionalVAE
  from mjlab.tasks.tracking.distillation.vae_config import ModelSettings

  model = ConditionalVAE(DEFAULT_SCHEMA, ModelSettings(hidden_dims=(8, 8)))
  return ShardedCollection(
    _StubPool(replies, failure=failure), replay=replay, model=model
  )


def test_sharded_source_inserts_merged_rows_into_the_one_replay() -> None:
  """One iteration's merged rows land in the parent's replay exactly once."""
  from mjlab.tasks.tracking.distillation.storage import LabeledReplayBuffer

  replay = LabeledReplayBuffer(64, DEFAULT_SCHEMA)
  reply0 = replace(_reply(worker_index=0), batch=_schema_batch(4, 0))
  reply1 = replace(_reply(worker_index=1), batch=_schema_batch(4, 1))
  source = _source([reply0, reply1], replay=replay)

  result = source.collect(CollectionConfig(steps=2, collector_iteration=0), reset=True)

  assert len(replay) == 8
  assert result.samples == 8
  assert result.fresh_data is not None
  assert result.fresh_data.update_id == 0
  assert source._requires_reset is False


def test_sharded_source_invalidates_and_closes_on_a_post_reply_failure() -> None:
  """A failure after the shards answered must not look resumable.

  Their environments have advanced while their rows were dropped, so claiming
  no reset is required would let the next collection continue across a gap
  that nothing recorded.
  """
  from mjlab.tasks.tracking.distillation.storage import LabeledReplayBuffer

  replay = LabeledReplayBuffer(64, DEFAULT_SCHEMA)
  # Mismatched framing fails inside the merge, i.e. after every shard replied.
  replies = [
    replace(_reply(worker_index=0), batch=_schema_batch(4, 0)),
    replace(_reply(worker_index=1), ticks=3, batch=_schema_batch(6, 1)),
  ]
  source = _source(replies, replay=replay)

  with pytest.raises(ShardedCollectionError):
    source.collect(CollectionConfig(steps=2, collector_iteration=0), reset=True)

  assert len(replay) == 0
  assert source._requires_reset is True
  assert source.pool.closed is True
  # The source refuses to continue without an explicit reset.
  with pytest.raises(ShardedCollectionError) as error:
    source.collect(CollectionConfig(steps=2, collector_iteration=1), reset=False)
  assert "reset=True is required" in str(error.value)


def test_sharded_source_invalidates_on_a_pool_failure() -> None:
  """A worker-pool failure leaves the replay untouched and needs a reset."""
  from mjlab.tasks.tracking.distillation.storage import LabeledReplayBuffer
  from mjlab.tasks.tracking.distillation.worker import WorkerPoolError

  replay = LabeledReplayBuffer(64, DEFAULT_SCHEMA)
  source = _source([], replay=replay, failure=WorkerPoolError("worker 1 died"))

  with pytest.raises(WorkerPoolError):
    source.collect(CollectionConfig(steps=2, collector_iteration=0), reset=True)

  assert len(replay) == 0
  assert source._requires_reset is True


def _segment(
  env_index: int,
  *,
  outcome: str,
  steps: int = 10,
  metrics: dict | None = None,
  motion_id: int = 1,
) -> EvaluationSegment:
  return EvaluationSegment(
    env_index=env_index,
    segment_id=env_index,
    generation_id=0,
    steps=steps,
    completed=outcome == "reference_complete",
    failed=outcome == "failure",
    capped=outcome == "step_cap",
    metrics=dict(metrics or {}),
    outcome=outcome,
    motion_id=motion_id,
  )


def _evaluation(segments, *, steps: int = 10, num_envs: int = 2, settings=None):
  from mjlab.tasks.tracking.distillation.collector import EvaluationResult

  return EvaluationResult(
    mode="student",
    rollout_latent="mean",
    steps=steps,
    segments=tuple(segments),
    metrics={},
    settings={"num_envs": num_envs, "step_cap": steps, **(settings or {})},
  )


def _evaluate_reply(index: int, segments, **kwargs) -> EvaluateReply:
  return EvaluateReply(
    call_id=index + 1, iteration=0, result=_evaluation(segments, **kwargs)
  )


def test_merged_evaluation_recomputes_rates_over_the_union_of_segments() -> None:
  """Rates must use the union denominator, not an average of shard rates.

  Shard 0 has one completion out of one known segment; shard 1 has one failure
  out of three known segments.  The true completion rate over the union is
  1/4.  Averaging the shard rates would report 2/3, and weighting by shard
  rather than by segment count would report the same wrong number.
  """
  shard0 = [_segment(0, outcome="reference_complete")]
  shard1 = [
    _segment(0, outcome="failure"),
    _segment(1, outcome="timeout"),
    _segment(1, outcome="step_cap"),
    _segment(0, outcome="reference_complete"),
  ]

  merged = merge_evaluations([_evaluate_reply(0, shard0), _evaluate_reply(1, shard1)])

  # Hand-computed from the concatenated records: 2 of 5 segments complete, 1 of
  # 5 known segments failed, 1 of 5 timed out, 1 of 5 capped.
  assert len(merged.segments) == 5
  assert merged.metrics["completion_known_segments"] == 3.0
  assert merged.metrics["completion_rate"] == pytest.approx(2 / 3)
  assert merged.metrics["failure_rate"] == pytest.approx(1 / 3)
  assert merged.metrics["timeout_rate"] == pytest.approx(1 / 5)
  assert merged.metrics["step_cap_rate"] == pytest.approx(1 / 5)
  assert merged.metrics["reset_rate"] == pytest.approx(0.0)
  # The derived properties recompute from the merged segments too.
  assert merged.completion_rate == pytest.approx(2 / 3)
  assert merged.failure_rate == pytest.approx(1 / 3)


def test_merged_evaluation_means_a_metric_over_segments_that_carry_it() -> None:
  """A per-segment metric is averaged over the segments that reported it."""
  shard0 = [
    _segment(0, outcome="timeout", metrics={"tracking_pose_error": 0.1}),
    _segment(1, outcome="timeout", metrics={"tracking_pose_error": 0.2}),
  ]
  shard1 = [
    _segment(0, outcome="timeout", metrics={"tracking_pose_error": 0.6}),
    # This segment reports no such metric and must not be counted as a zero.
    _segment(1, outcome="timeout", metrics={}),
  ]

  merged = merge_evaluations([_evaluate_reply(0, shard0), _evaluate_reply(1, shard1)])

  assert merged.metrics["tracking_pose_error"] == pytest.approx((0.1 + 0.2 + 0.6) / 3)


def test_merged_evaluation_tags_segments_and_sums_the_environment_count() -> None:
  """Merged segments name their shard, and the cohort size is the union's."""
  merged = merge_evaluations(
    [
      _evaluate_reply(0, [_segment(0, outcome="timeout")], num_envs=2),
      _evaluate_reply(1, [_segment(1, outcome="timeout")], num_envs=2),
    ]
  )

  assert [segment.worker_index for segment in merged.segments] == [0, 1]
  # Both shards number their own environments from zero, which is exactly why
  # the shard tag is required to read a merged segment.
  assert [segment.env_index for segment in merged.segments] == [0, 1]
  assert merged.settings["num_envs"] == 4
  assert merged.settings["sharded_workers"] == 2
  assert merged.trial_window_steps is None


def test_merged_evaluation_accepts_an_empty_shard() -> None:
  """A shard that produced no segment must not invent one or skew a rate.

  An empty shard is a real outcome (its environments never finished a
  segment), and the union denominators have to reflect the shards that did
  report rather than the number of shards.
  """
  merged = merge_evaluations(
    [
      _evaluate_reply(0, []),
      _evaluate_reply(1, [_segment(0, outcome="reference_complete")]),
    ]
  )

  assert len(merged.segments) == 1
  assert merged.metrics["completion_rate"] == pytest.approx(1.0)
  assert merged.metrics["timeout_rate"] == pytest.approx(0.0)


def test_merged_evaluation_refuses_shards_that_evaluated_differently() -> None:
  """Segments from different evaluations cannot be pooled into one metric."""
  with pytest.raises(ShardedCollectionError) as error:
    merge_evaluations(
      [
        _evaluate_reply(0, [_segment(0, outcome="timeout")], steps=10),
        _evaluate_reply(1, [_segment(0, outcome="timeout")], steps=25),
      ]
    )
  assert "do not describe one evaluation" in str(error.value)


def test_merged_evaluation_reports_a_standing_profile_from_shard_settings() -> None:
  """Trial accounting is applied when the shards ran a standing profile."""
  merged = merge_evaluations(
    [
      _evaluate_reply(
        0, [_segment(0, outcome="timeout")], settings={"trial_window_steps": 25}
      ),
      _evaluate_reply(
        1, [_segment(0, outcome="failure")], settings={"trial_window_steps": 25}
      ),
    ]
  )

  assert "trial_failure_rate" in merged.metrics
  assert "trial_window_steps" in merged.settings


def test_merged_evaluation_refuses_shards_from_different_iterations() -> None:
  """Merging two iterations would report one iteration's metrics as the other's."""
  with pytest.raises(ShardedCollectionError) as error:
    merge_evaluations(
      [
        _evaluate_reply(0, [_segment(0, outcome="timeout")]),
        EvaluateReply(
          call_id=2,
          iteration=1,
          result=_evaluation([_segment(0, outcome="timeout")]),
        ),
      ]
    )
  assert "do not describe one evaluation" in str(error.value)


def test_merged_evaluation_refuses_a_mixed_standing_profile() -> None:
  """A standing shard and a non-standing one do not describe one evaluation.

  Their trial accounting differs, so pooling the segments would apply standing
  trial buckets to rows that never ran under that profile.
  """
  with pytest.raises(ShardedCollectionError) as error:
    merge_evaluations(
      [
        _evaluate_reply(
          0, [_segment(0, outcome="timeout")], settings={"trial_window_steps": 25}
        ),
        _evaluate_reply(1, [_segment(0, outcome="timeout")]),
      ]
    )
  assert "trial accounting cannot be merged" in str(error.value)


def test_merged_evaluation_refuses_two_different_trial_windows() -> None:
  """The trial window decides the buckets, so every shard must share it."""
  with pytest.raises(ShardedCollectionError) as error:
    merge_evaluations(
      [
        _evaluate_reply(
          0, [_segment(0, outcome="timeout")], settings={"trial_window_steps": 25}
        ),
        _evaluate_reply(
          1, [_segment(0, outcome="timeout")], settings={"trial_window_steps": 50}
        ),
      ]
    )
  assert "window 50 while shard 0 reported" in str(error.value)
  assert "window 25" in str(error.value)
