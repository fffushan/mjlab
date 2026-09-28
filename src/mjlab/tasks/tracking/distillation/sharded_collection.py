"""Merging sharded collection replies into the one replay the parent owns.

A sharded iteration produces one reply per worker.  This module turns those
replies into the single :class:`CollectionResult` the lifecycle runner already
understands, so nothing downstream — the trainer, the normalizer update, the
segment rebase, the report — needs to know how many processes collected the
rows.

Two rules are worth stating because they are the ones a plausible-looking
implementation gets silently wrong:

* **Insertion order is tick-major, worker-ascending.**  Tick ``k`` from every
  worker is inserted before tick ``k+1``, which keeps the FIFO/eviction
  semantics of the single-process insert-per-tick loop.  A batch sorted by
  worker instead would look right in every counter and quietly change which
  rows a full replay forgets first.
* **A mean over rows is merged by row counts.**  Disagreement is an average
  over rows, so weighting each shard's mean by the rows it contributed
  reproduces the union's mean exactly.  Averaging the shard means instead would
  weight a shard with five rows the same as one with five thousand.

Nothing is inserted until every reply has arrived and been validated: a failed
shard leaves the replay untouched rather than half-written.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from mjlab.tasks.tracking.distillation.collector import (
  CollectionConfig,
  CollectionResult,
  EvaluationMode,
  EvaluationResult,
  MotionCollectionStats,
  RolloutLatent,
  SegmentBoundary,
  aggregate_evaluation_metrics,
)
from mjlab.tasks.tracking.distillation.storage import (
  LabeledReplayBatch,
  ReplayBufferProtocol,
)
from mjlab.tasks.tracking.distillation.trainer import FreshTrainingData
from mjlab.tasks.tracking.distillation.worker import (
  CollectReply,
  EvaluateReply,
  WorkerEnvironmentDescription,
  WorkerPool,
  WorkerPoolError,
)


class ShardedCollectionError(RuntimeError):
  """A set of worker replies cannot be merged into one collection result."""


WORKER_SEGMENT_STRIDE = 1 << 40
"""Width of one worker's block of segment ids.

A merged segment id is ``namespace + worker * stride + local``, where the local
part belongs to one simulator and the namespace is added by the runner's resume
rebase.  Each worker's simulator numbers its own segments from zero, so without
the worker offset two shards' segment 7 would be one identity in the replay and
the resume rebase — which adds a single namespace to an iteration's ids — could
never tell them apart.  The stride is far wider than any local id a bounded run
can produce, and a shard whose local ids would reach it is refused below rather
than silently overlapping its neighbour.
"""


def _weighted_mean(
  values: Iterable[tuple[float | None, int]], *, empty: float | None
) -> float | None:
  """Average ``(value, weight)`` pairs by weight, or ``empty`` with no weight.

  ``empty`` distinguishes the two callers: the top-level counter has to stay a
  float because the single-process result always carries one, while a
  per-motion entry reports ``None`` when no rows were attributed to it.
  """
  total_weight = 0
  total = 0.0
  for value, weight in values:
    if value is None or weight <= 0:
      continue
    total += value * weight
    total_weight += weight
  if total_weight == 0:
    return empty
  return total / total_weight


def merge_motion_stats(
  groups: Sequence[Sequence[MotionCollectionStats]],
  *,
  frames_by_motion: Mapping[int, Sequence[int]] | None = None,
) -> tuple[MotionCollectionStats, ...]:
  """Combine per-motion statistics from every shard, ordered by motion id.

  Additive counters are summed, disagreement is weighted by the rows each shard
  contributed, frame bounds take the union's bounds, and
  ``reference_frames_observed`` is the size of the union of the shards' frame
  sets when those sets are supplied here (the merge always has them, because a
  worker reports the frames it observed).

  Without the sets, the largest shard count is only a lower bound: two shards
  that observed disjoint frames would each report their own count, and the sum
  would be wrong in the other direction by exceeding the clip length.  A caller
  that cannot supply frame identity is therefore given the safe bound rather
  than a number that looks exact.
  """
  merged: dict[int, dict[str, Any]] = {}
  for group in groups:
    for stats in group:
      entry = merged.setdefault(
        stats.motion_id,
        {
          "teacher_id": stats.teacher_id,
          "teacher_code": stats.teacher_code,
          "samples": 0,
          "teacher_steps": 0,
          "student_steps": 0,
          "boundaries": 0,
          "disagreement_rows": 0,
          "disagreement": [],
          "reference_frames_observed": 0,
          "frame_min": None,
          "frame_max": None,
          "initialization_samples": {},
          "early_transition_samples": {},
        },
      )
      if (entry["teacher_id"], entry["teacher_code"]) != (
        stats.teacher_id,
        stats.teacher_code,
      ):
        raise ShardedCollectionError(
          f"motion {stats.motion_id} was attributed to two different teachers "
          f"across shards: {entry['teacher_id']!r}/{entry['teacher_code']} and "
          f"{stats.teacher_id!r}/{stats.teacher_code}"
        )
      entry["samples"] += stats.samples
      entry["teacher_steps"] += stats.teacher_steps
      entry["student_steps"] += stats.student_steps
      entry["boundaries"] += stats.boundaries
      entry["disagreement_rows"] += stats.disagreement_rows
      entry["disagreement"].append((stats.disagreement_mean, stats.disagreement_rows))
      entry["reference_frames_observed"] = max(
        entry["reference_frames_observed"], stats.reference_frames_observed
      )
      if stats.reference_frame_min is not None:
        current = entry["frame_min"]
        entry["frame_min"] = (
          stats.reference_frame_min
          if current is None
          else min(current, stats.reference_frame_min)
        )
      if stats.reference_frame_max is not None:
        current = entry["frame_max"]
        entry["frame_max"] = (
          stats.reference_frame_max
          if current is None
          else max(current, stats.reference_frame_max)
        )
      for key, value in stats.initialization_samples.items():
        entry["initialization_samples"][key] = (
          entry["initialization_samples"].get(key, 0) + value
        )
      for key, value in stats.early_transition_samples.items():
        entry["early_transition_samples"][key] = (
          entry["early_transition_samples"].get(key, 0) + value
        )
  return tuple(
    MotionCollectionStats(
      motion_id=motion_id,
      teacher_id=entry["teacher_id"],
      teacher_code=entry["teacher_code"],
      samples=entry["samples"],
      teacher_steps=entry["teacher_steps"],
      student_steps=entry["student_steps"],
      boundaries=entry["boundaries"],
      disagreement_mean=_weighted_mean(entry["disagreement"], empty=None),
      disagreement_rows=entry["disagreement_rows"],
      reference_frames_observed=(
        entry["reference_frames_observed"]
        if frames_by_motion is None or motion_id not in frames_by_motion
        else len(set(frames_by_motion[motion_id]))
      ),
      reference_frame_min=(
        entry["frame_min"]
        if frames_by_motion is None or motion_id not in frames_by_motion
        else min(frames_by_motion[motion_id])
      ),
      reference_frame_max=(
        entry["frame_max"]
        if frames_by_motion is None or motion_id not in frames_by_motion
        else max(frames_by_motion[motion_id])
      ),
      initialization_samples=dict(entry["initialization_samples"]),
      early_transition_samples=dict(entry["early_transition_samples"]),
    )
    for motion_id, entry in sorted(merged.items())
  )


def union_motion_frames(
  replies: Sequence[CollectReply],
) -> dict[int, tuple[int, ...]]:
  """Union every shard's observed reference frames, per motion."""
  frames: dict[int, set[int]] = {}
  for reply in replies:
    for motion, observed in reply.motion_frames:
      frames.setdefault(int(motion), set()).update(int(frame) for frame in observed)
  return {motion: tuple(sorted(values)) for motion, values in frames.items()}


def _require_mergeable(replies: Sequence[CollectReply]) -> tuple[int, int]:
  """Validate the reply set's framing and return ``(ticks, rows_per_tick)``."""
  if not replies:
    raise ShardedCollectionError("a merged collection needs at least one worker reply")
  ticks = replies[0].ticks
  rows_per_tick = replies[0].rows_per_tick
  if ticks <= 0 or rows_per_tick <= 0:
    raise ShardedCollectionError(
      f"a merged collection needs positive framing, got ticks={ticks} "
      f"rows_per_tick={rows_per_tick}"
    )
  for index, reply in enumerate(replies):
    if reply.ticks != ticks or reply.rows_per_tick != rows_per_tick:
      raise ShardedCollectionError(
        f"worker {index} returned ticks={reply.ticks} rows_per_tick="
        f"{reply.rows_per_tick} while worker 0 returned ticks={ticks} "
        f"rows_per_tick={rows_per_tick}"
      )
    if reply.batch.batch_size != ticks * rows_per_tick:
      raise ShardedCollectionError(
        f"worker {index} returned {reply.batch.batch_size} rows, which is not "
        f"{ticks} ticks of {rows_per_tick} rows"
      )
  return ticks, rows_per_tick


def merge_batches_tick_major(
  replies: Sequence[CollectReply], *, ticks: int, rows_per_tick: int
) -> LabeledReplayBatch:
  """Concatenate shard rows tick by tick, workers ascending within a tick."""

  def accessor(name: str) -> Callable[[LabeledReplayBatch], torch.Tensor]:
    if name.startswith("observations."):
      field = name.split(".", 1)[1]
      return lambda batch: getattr(batch.observations, field)
    return lambda batch: getattr(batch, name)

  def merged(name: str) -> torch.Tensor:
    blocks = [
      accessor(name)(reply.batch)[tick * rows_per_tick : (tick + 1) * rows_per_tick]
      for tick in range(ticks)
      for reply in replies
    ]
    return torch.cat([block.contiguous() for block in blocks], dim=0)

  first = replies[0].batch
  provenance = [reply.batch.has_provenance for reply in replies]
  if any(provenance) and not all(provenance):
    raise ShardedCollectionError(
      "workers disagree about whether collected rows carry standing-start "
      "provenance; merging them would invent a reference history"
    )
  observations = replace(
    first.observations,
    reference=merged("observations.reference"),
    conditioning=merged("observations.conditioning"),
  )
  batch = replace(
    first,
    observations=observations,
    teacher_action=merged("teacher_action"),
    motion_id=merged("motion_id"),
    teacher_id=merged("teacher_id"),
    reference_frame=merged("reference_frame"),
    episode_id=_merged_segment_ids(replies, ticks=ticks, rows_per_tick=rows_per_tick),
    collector_iteration=merged("collector_iteration"),
  )
  if not all(provenance):
    return batch
  return replace(
    batch,
    initialization_kind=merged("initialization_kind"),
    segment_initial_reference_frame=merged("segment_initial_reference_frame"),
    segment_age=merged("segment_age"),
  )


def _merged_segment_ids(
  replies: Sequence[CollectReply], *, ticks: int, rows_per_tick: int
) -> torch.Tensor:
  """Offset each shard's segment ids into its own namespace block.

  The offset is applied here, in the parent, because segment identity is a
  parent-side concern: a worker's reply stays a faithful copy of what its own
  collector produced, and this is the one place where several simulators'
  numbering is combined into one identity space.
  """
  blocks = []
  for index, reply in enumerate(replies):
    local = reply.batch.episode_id
    if local.numel():
      local_max = int(local.max().item())
      local_min = int(local.min().item())
      if local_min < 0 or local_max >= WORKER_SEGMENT_STRIDE:
        raise ShardedCollectionError(
          f"worker {index} produced segment ids in [{local_min}, {local_max}], "
          f"which do not fit this run's per-worker block of "
          f"{WORKER_SEGMENT_STRIDE}; refusing rather than overlapping another "
          f"worker's ids"
        )
  # Tick-major, workers ascending: the same nesting the rows themselves use, so
  # a segment id stays aligned with the row block it belongs to.
  for tick in range(ticks):
    for index, reply in enumerate(replies):
      block = reply.batch.episode_id[tick * rows_per_tick : (tick + 1) * rows_per_tick]
      blocks.append(block + index * WORKER_SEGMENT_STRIDE)
  return torch.cat([block.contiguous() for block in blocks], dim=0)


def _move_batch(batch: LabeledReplayBatch, device: torch.device) -> LabeledReplayBatch:
  """Move a merged batch onto the replay's device in one hop."""
  if device.type == "cpu":
    return batch

  def moved(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.to(device)

  return replace(
    batch,
    observations=replace(
      batch.observations,
      reference=moved(batch.observations.reference),
      conditioning=moved(batch.observations.conditioning),
    ),
    teacher_action=moved(batch.teacher_action),
    motion_id=moved(batch.motion_id),
    teacher_id=moved(batch.teacher_id),
    reference_frame=moved(batch.reference_frame),
    episode_id=moved(batch.episode_id),
    collector_iteration=moved(batch.collector_iteration),
    initialization_kind=moved(batch.initialization_kind),
    segment_initial_reference_frame=moved(batch.segment_initial_reference_frame),
    segment_age=moved(batch.segment_age),
  )


def merge_collection_replies(
  replies: Sequence[CollectReply],
  *,
  iteration: int,
  device: torch.device,
) -> CollectionResult:
  """Merge worker replies into the single result the lifecycle runner expects.

  ``ticks`` is the union (every shard steps the same ticks, which the framing
  check enforces) while the row counters are sums, so the reported
  samples-per-tick ratio keeps its meaning.  Boundaries are concatenated
  worker-ascending and tagged with the worker that produced them, because
  environment indices are per-shard and must never be read as one global index.
  """
  ticks, rows_per_tick = _require_mergeable(replies)
  batch = _move_batch(
    merge_batches_tick_major(replies, ticks=ticks, rows_per_tick=rows_per_tick),
    device,
  )
  samples = sum(reply.samples for reply in replies)
  boundaries: list[SegmentBoundary] = []
  diagnostics: list[str] = []
  eligible_resets: dict[int, int] = {}
  initialization_resets: dict[int, dict[str, int]] = {}
  for index, reply in enumerate(replies):
    boundaries.extend(replace(item, worker_index=index) for item in reply.boundaries)
    diagnostics.extend(f"worker {index}: {text}" for text in reply.diagnostics)
    for motion, count in reply.eligible_resets.items():
      eligible_resets[motion] = eligible_resets.get(motion, 0) + count
    for motion, counts in reply.initialization_resets.items():
      entry = initialization_resets.setdefault(motion, {})
      for kind, count in counts.items():
        entry[kind] = entry.get(kind, 0) + count
  return CollectionResult(
    ticks=ticks,
    samples=samples,
    teacher_steps=sum(reply.teacher_steps for reply in replies),
    student_steps=sum(reply.student_steps for reply in replies),
    boundaries=tuple(boundaries),
    disagreement_mean=float(
      _weighted_mean(
        [(reply.disagreement_mean, reply.samples) for reply in replies], empty=0.0
      )
    ),
    diagnostics=tuple(diagnostics),
    fresh_data=FreshTrainingData(batch, iteration),
    motion_stats=merge_motion_stats(
      [list(reply.motion_stats) for reply in replies],
      frames_by_motion=union_motion_frames(replies),
    ),
    eligible_resets=eligible_resets,
    initialization_resets=initialization_resets,
  )


def _evaluation_profile(result: EvaluationResult) -> tuple[bool, int]:
  """The trial-accounting profile one shard's evaluation ran under.

  A standing-trial run records its window in the result settings; that presence
  is exactly the flag the single-process aggregation gated on.  A shard that
  ran no standing trials reports no window, and its unused window field is
  deliberately not compared.
  """
  if "trial_window_steps" not in result.settings:
    return False, 0
  return True, int(result.settings["trial_window_steps"])


def merge_evaluations(replies: Sequence[EvaluateReply]) -> EvaluationResult:
  """Merge shard evaluations into one result by recomputing from segments.

  Segments are concatenated worker-ascending and tagged with their shard, then
  the same aggregation the single-process path uses recomputes every rate, mean
  and trial bucket over the union.  That is what makes the merge exact instead
  of an average of averages: the aggregation is a pure function of the segment
  list, so rates keep their true denominators across shards and the trial
  buckets add up.  Averaging shard-level results would weight a shard with two
  segments like one with two hundred.

  Whether the profile is a standing-trial run is read from the shards'
  settings, because that is exactly the flag that gated the trial accounting
  when each shard produced its own result -- and every shard must report the
  same profile, since a standing shard and a non-standing one do not describe
  one evaluation however well their call arguments agree.
  """
  if not replies:
    raise ShardedCollectionError("a merged evaluation needs at least one shard")
  first = replies[0]
  expected_call = (
    first.iteration,
    first.result.mode,
    first.result.rollout_latent,
    first.result.steps,
  )
  expected_profile = _evaluation_profile(first.result)
  for index, reply in enumerate(replies):
    result = reply.result
    call = (reply.iteration, result.mode, result.rollout_latent, result.steps)
    if call != expected_call:
      raise ShardedCollectionError(
        f"shard {index} evaluated iteration={reply.iteration} mode={result.mode!r} "
        f"rollout_latent={result.rollout_latent!r} steps={result.steps} while "
        f"shard 0 evaluated iteration={first.iteration} mode={first.result.mode!r} "
        f"rollout_latent={first.result.rollout_latent!r} steps={first.result.steps}; "
        "their segments do not describe one evaluation"
      )
    profile = _evaluation_profile(result)
    if profile != expected_profile:
      theirs = "a standing-trial" if profile[0] else "no standing-trial"
      ours = "a standing-trial" if expected_profile[0] else "no standing-trial"
      raise ShardedCollectionError(
        f"shard {index} reported {theirs} profile with window {profile[1]} while "
        f"shard 0 reported {ours} profile with window {expected_profile[1]}; "
        "their trial accounting cannot be merged"
      )
  segments = tuple(
    replace(segment, worker_index=index)
    for index, reply in enumerate(replies)
    for segment in reply.result.segments
  )
  standing, trial_window = expected_profile
  metrics = aggregate_evaluation_metrics(
    segments,
    standing_trials=standing,
    trial_window_steps=trial_window,
  )
  settings = dict(first.result.settings)
  settings["num_envs"] = sum(
    int(reply.result.settings.get("num_envs", 0)) for reply in replies
  )
  settings["sharded_workers"] = len(replies)
  settings["sharded_metrics"] = (
    "recomputed from every shard's segments, not averaged from shard results"
  )
  return EvaluationResult(
    mode=first.result.mode,
    rollout_latent=first.result.rollout_latent,
    steps=first.result.steps,
    segments=segments,
    metrics=metrics,
    settings=settings,
    trial_window_steps=first.result.trial_window_steps,
  )


class ShardedCollection:
  """Parent-side collection source backed by a worker pool.

  It exposes the same narrow surface the lifecycle runner uses on a local
  collector — ``collect``, ``invalidate_snapshot`` and ``teacher`` — so the
  runner does not branch on how rows were produced.  The replay handed in stays
  the single writer's target: this object never lets a worker touch it.
  """

  def __init__(
    self,
    pool: WorkerPool,
    *,
    replay: ReplayBufferProtocol,
    model: Any,
    descriptions: Sequence[WorkerEnvironmentDescription] | None = None,
  ) -> None:
    self.pool = pool
    self.replay = replay
    self.model = model
    self.descriptions = tuple(descriptions or ())
    self.teacher = None
    self._requires_reset = False
    self._invalidation_cause: str | None = None

  @classmethod
  def start(
    cls,
    specs: Sequence[Any],
    *,
    replay: ReplayBufferProtocol,
    model: Any,
    request_timeout_s: float | None = None,
  ) -> ShardedCollection:
    """Spawn the pool, verify every shard built its environment, and cache it.

    The description exchange is the first real work a worker does, so a shard
    that cannot build its environment fails here rather than one iteration into
    a multi-hour run.
    """
    pool = (
      WorkerPool(specs)
      if request_timeout_s is None
      else WorkerPool(specs, request_timeout_s=request_timeout_s)
    )
    pool.start()
    try:
      descriptions = pool.describe()
    except BaseException:
      pool.close()
      raise
    return cls(pool, replay=replay, model=model, descriptions=descriptions)

  @property
  def reset_provenance(self) -> Any | None:
    """The reset provenance a resume and a checkpoint record.

    Read from a worker's live command, because in this layout the parent owns
    no environment; every shard applies the same reset policy, so the first
    worker's record describes the run.
    """
    if not self.descriptions:
      return None
    description = self.descriptions[0]
    return description.reset_provenance if description.reset_policy_enabled else None

  @property
  def reset_policy_enabled(self) -> bool:
    return bool(self.descriptions and self.descriptions[0].reset_policy_enabled)

  @property
  def sampling_mode(self) -> str | None:
    if not self.descriptions:
      return None
    return self.descriptions[0].sampling_mode

  def invalidate_snapshot(self, *, requires_reset: bool = True) -> None:
    """Mark every shard as needing a reset before the next collection.

    A worker keeps its environment between requests, exactly like the adapter
    the single-process path keeps, so an external rollout on those environments
    invalidates the continuation the same way it does there.
    """
    self._requires_reset = requires_reset
    self._invalidation_cause = None

  def collect(
    self, config: CollectionConfig, *, reset: bool = True
  ) -> CollectionResult:
    """Collect one iteration across every shard and insert it atomically."""
    if self._requires_reset and not reset:
      cause = self._invalidation_cause or "an external reset or rollout"
      raise ShardedCollectionError(
        f"collection state was invalidated by {cause}; reset=True is required"
      )
    # A model state dict is not only tensors: a module may publish plain-data
    # ``_extra_state``, which is part of its own state contract and must travel
    # unchanged.  Only tensors need staging.
    state = self.model.state_dict()
    weights = {
      name: value.detach() if isinstance(value, torch.Tensor) else value
      for name, value in state.items()
    }
    try:
      replies = self.pool.collect(
        iteration=config.collector_iteration,
        steps=config.steps,
        teacher_probability=config.teacher_probability,
        reset=reset,
        rollout_latent=config.rollout_latent,
        weights=weights,
      )
    except WorkerPoolError as exc:
      self._requires_reset = True
      self._invalidation_cause = f"{type(exc).__name__}: {exc}"
      raise
    # Every failure after the shards have answered leaves those environments
    # advanced but their rows unrecorded, so the source must not claim to be
    # resumable and the pool is closed: continuing would collect a gap.
    try:
      result = merge_collection_replies(
        replies, iteration=config.collector_iteration, device=self.replay_device
      )
      assert result.fresh_data is not None
      # The insert is the first durable effect of the iteration and it happens
      # only here, after every shard has answered and the merged rows validate.
      self.replay.insert(result.fresh_data.batch)
    except BaseException as exc:
      self._requires_reset = True
      self._invalidation_cause = f"{type(exc).__name__}: {exc}"
      self.pool.close()
      raise
    self._requires_reset = False
    return result

  @property
  def replay_device(self) -> torch.device:
    """Device the merged rows must arrive on before they can be inserted."""
    return self.replay.device or torch.device("cpu")

  def evaluate(
    self,
    *,
    iteration: int,
    mode: EvaluationMode,
    steps: int,
    rollout_latent: RolloutLatent,
  ) -> EvaluationResult:
    """Evaluate every shard with one policy state and merge their segments.

    Each shard evaluates its own environments under the weights the trainer
    holds, and the parent merges the retained segment records, so the evaluated
    environment count stays the whole cohort rather than one shard's share.
    """
    # A model state dict is not only tensors: a module may publish plain-data
    # ``_extra_state``, which must travel unchanged.
    state = self.model.state_dict()
    weights = {
      name: value.detach() if isinstance(value, torch.Tensor) else value
      for name, value in state.items()
    }
    replies = self.pool.evaluate(
      iteration=iteration,
      mode=mode,
      steps=steps,
      rollout_latent=rollout_latent,
      weights=weights,
    )
    return merge_evaluations(replies)

  def close(self) -> None:
    self.pool.close()
