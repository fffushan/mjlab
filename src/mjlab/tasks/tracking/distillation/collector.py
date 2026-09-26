"""Synchronous single-teacher DAgger collection and bounded evaluation.

The collector operates on :class:`DistillationEnvironmentAdapter` snapshots.  A
snapshot is labeled and copied before its action is stepped; consequently a
record always describes the state whose teacher action it contains.  This
module intentionally contains no optimizer, normalizer update, checkpoint, or
CLI lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import torch

from mjlab.tasks.tracking.distillation.adapter import (
  DistillationSnapshot,
  DistillationStep,
)
from mjlab.tasks.tracking.distillation.model import ConditionalVAE
from mjlab.tasks.tracking.distillation.observations import PackedObservationBatch
from mjlab.tasks.tracking.distillation.storage import (
  LabeledReplayBatch,
  LabeledReplayBuffer,
)
from mjlab.tasks.tracking.distillation.teachers import FrozenTeacher
from mjlab.tasks.tracking.distillation.trainer import FreshTrainingData


class CollectionNumericalError(RuntimeError):
  """A non-finite snapshot, label, or command prevented a simulator step."""


class _AdapterLike(Protocol):
  def reset(self, seed: int | None = None) -> DistillationSnapshot: ...

  def step(self, action: torch.Tensor) -> DistillationStep: ...


class _TeacherLike(Protocol):
  action_dim: int

  def label(self, observations: torch.Tensor) -> torch.Tensor: ...


RolloutLatent = Literal["mean", "sampled"]
EvaluationMode = Literal["teacher", "student"]
RngMode = Literal["per_call", "persistent"]


@dataclass(frozen=True, slots=True)
class CollectionConfig:
  """Bounded collection settings.

  ``teacher_probability`` selects a complete command vector independently for
  each environment.  It is not a blend coefficient: at 0 and 1 no random draw
  is made and the corresponding endpoint is exact.
  """

  steps: int
  teacher_probability: float = 1.0
  rollout_latent: RolloutLatent = "mean"
  seed: int = 0
  collector_iteration: int = 0
  rng_mode: RngMode = "per_call"

  def __post_init__(self) -> None:
    if (
      not isinstance(self.steps, int) or isinstance(self.steps, bool) or self.steps < 0
    ):
      raise ValueError("steps must be a non-negative integer")
    if not 0.0 <= self.teacher_probability <= 1.0:
      raise ValueError("teacher_probability must be in [0, 1]")
    if self.rollout_latent not in ("mean", "sampled"):
      raise ValueError("rollout_latent must be 'mean' or 'sampled'")
    if self.rng_mode not in ("per_call", "persistent"):
      raise ValueError("rng_mode must be 'per_call' or 'persistent'")


@dataclass(frozen=True, slots=True)
class SegmentBoundary:
  """A reset, termination, timeout, or motion-generation boundary."""

  env_indices: tuple[int, ...]
  reason: str
  before_segment: tuple[int, ...]
  after_segment: tuple[int, ...]
  before_generation: tuple[int, ...]
  after_generation: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CollectionResult:
  """Counters and explicit boundaries from one bounded collection call."""

  ticks: int
  samples: int
  teacher_steps: int
  student_steps: int
  boundaries: tuple[SegmentBoundary, ...]
  disagreement_mean: float
  diagnostics: tuple[str, ...] = ()
  fresh_data: FreshTrainingData | None = None


@dataclass(frozen=True, slots=True)
class EvaluationSegment:
  """Truthful result for one continuous reference segment."""

  env_index: int
  segment_id: int
  generation_id: int
  steps: int
  completed: bool
  failed: bool
  capped: bool
  metrics: dict[str, float]
  outcome: str = "step_cap"
  reference_start_frame: int | None = None
  reference_end_frame: int | None = None
  reference_coverage: int = 0
  metric_valid_steps: int = 0
  valid_action_transitions: int = 0
  completion_available: bool | None = None
  completion_reason: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
  """Bounded evaluation output, retaining per-segment outcomes."""

  mode: EvaluationMode
  rollout_latent: RolloutLatent
  steps: int
  segments: tuple[EvaluationSegment, ...]
  metrics: dict[str, float]
  settings: dict[str, Any] = field(default_factory=dict)

  @property
  def completion_rate(self) -> float | None:
    known = [
      segment
      for segment in self.segments
      if segment.outcome in ("reference_complete", "failure")
    ]
    if not known:
      return None
    return sum(segment.outcome == "reference_complete" for segment in known) / len(
      known
    )

  @property
  def failure_rate(self) -> float | None:
    known = [
      segment
      for segment in self.segments
      if segment.outcome in ("reference_complete", "failure")
    ]
    if not known:
      return None
    return sum(segment.failed for segment in known) / len(known)


def _finite(name: str, value: torch.Tensor) -> None:
  if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all().item():
    raise CollectionNumericalError(f"non-finite {name}; simulator step was skipped")


def _as_bool_vector(name: str, value: torch.Tensor, batch: int) -> torch.Tensor:
  if not isinstance(value, torch.Tensor) or value.shape != (batch,):
    raise ValueError(f"{name} must have shape [{batch}]")
  return value.to(dtype=torch.bool)


def _event_reason_summary(events: Any, batch: int) -> str | None:
  if events is None:
    return None
  available = getattr(events, "available", None)
  reasons = getattr(events, "reasons", None)
  if (
    not isinstance(available, torch.Tensor)
    or available.shape != (batch,)
    or not isinstance(reasons, tuple)
    or len(reasons) != batch
  ):
    raise ValueError("malformed reference boundary events")
  values = sorted(
    {reasons[index] for index in torch.nonzero(available).flatten().tolist()}
  )
  return "+".join(values) if values else None


def _require_auto_reset(adapter: _AdapterLike) -> None:
  """Reject adapters that return terminal states requiring manual reset.

  The live distillation adapter wraps ManagerBasedRlEnv's auto-reset path.  A
  custom adapter may opt out explicitly with ``auto_reset = False``; silently
  stepping such an adapter would label a stale terminal observation.
  """
  auto_reset = getattr(adapter, "auto_reset", True)
  if not isinstance(auto_reset, bool):
    raise ValueError("adapter.auto_reset must be a bool when provided")
  if not auto_reset:
    raise ValueError("DAggerCollector requires an auto-reset adapter")


def _concat_batches(batches: list[LabeledReplayBatch]) -> LabeledReplayBatch | None:
  if not batches:
    return None
  schema = batches[0].schema
  observations = PackedObservationBatch(
    torch.cat([batch.reference for batch in batches], dim=0),
    torch.cat([batch.conditioning for batch in batches], dim=0),
    schema,
  )
  return LabeledReplayBatch(
    observations=observations,
    teacher_action=torch.cat([batch.teacher_action for batch in batches], dim=0),
    motion_id=torch.cat([batch.motion_id for batch in batches], dim=0),
    teacher_id=torch.cat([batch.teacher_id for batch in batches], dim=0),
    reference_frame=torch.cat([batch.reference_frame for batch in batches], dim=0),
    episode_id=torch.cat([batch.episode_id for batch in batches], dim=0),
    collector_iteration=torch.cat(
      [batch.collector_iteration for batch in batches], dim=0
    ),
  )


def _ids(snapshot: DistillationSnapshot) -> tuple[torch.Tensor, torch.Tensor]:
  segment = snapshot.segment_id.detach().clone().to(dtype=torch.int64)
  generation = snapshot.generation_id.detach().clone().to(dtype=torch.int64)
  return segment, generation


def _boundary(
  before: DistillationSnapshot,
  after: DistillationSnapshot,
  terminated: torch.Tensor,
  time_outs: torch.Tensor,
  events: Any = None,
) -> SegmentBoundary | None:
  before_segment, before_generation = _ids(before)
  after_segment, after_generation = _ids(after)
  changed = (before_segment != after_segment) | (before_generation != after_generation)
  changed |= terminated | time_outs
  reasons: list[str] = []
  if events is not None:
    available = getattr(events, "available", None)
    event_reasons = getattr(events, "reasons", None)
    if (
      not isinstance(available, torch.Tensor)
      or available.shape != changed.shape
      or not isinstance(event_reasons, tuple)
      or len(event_reasons) != changed.shape[0]
    ):
      raise ValueError("malformed reference boundary events")
    changed |= available.to(dtype=torch.bool)
    reasons.extend(
      sorted(
        {event_reasons[index] for index in torch.nonzero(available).flatten().tolist()}
      )
    )
  indices = torch.nonzero(changed, as_tuple=False).flatten().tolist()
  if not indices:
    return None
  if bool(terminated.any()):
    reasons.append("terminated")
  if bool(time_outs.any()):
    reasons.append("timeout")
  if bool((before_generation != after_generation).any()):
    reasons.append("generation")
  elif bool((before_segment != after_segment).any()):
    reasons.append("segment")
  return SegmentBoundary(
    env_indices=tuple(int(index) for index in indices),
    reason="+".join(reasons) or "boundary",
    before_segment=tuple(int(value) for value in before_segment.tolist()),
    after_segment=tuple(int(value) for value in after_segment.tolist()),
    before_generation=tuple(int(value) for value in before_generation.tolist()),
    after_generation=tuple(int(value) for value in after_generation.tolist()),
  )


def _student_action(
  student: ConditionalVAE,
  snapshot: DistillationSnapshot,
  rollout_latent: RolloutLatent,
  generator: torch.Generator,
) -> torch.Tensor:
  reference, conditioning = snapshot.packed.reference, snapshot.packed.conditioning
  with torch.no_grad():
    if rollout_latent == "mean":
      action = student.mean_inference(reference, conditioning)
    else:
      mu, logvar = student.encode(reference)
      noise = torch.randn(
        mu.shape, dtype=mu.dtype, device=generator.device, generator=generator
      ).to(device=mu.device)
      action = student.decode(student.sample_latent(mu, logvar, noise), conditioning)
  _finite("student action", action)
  return action.detach().clone()


def _teacher_action(
  teacher: _TeacherLike, snapshot: DistillationSnapshot
) -> torch.Tensor:
  _finite("teacher observation", snapshot.teacher_observation)
  with torch.no_grad():
    action = teacher.label(snapshot.teacher_observation)
  _finite("teacher label", action)
  if action.shape != (snapshot.teacher_observation.shape[0], teacher.action_dim):
    raise ValueError("teacher label has an unexpected shape")
  return action.detach().clone()


def _select_action(
  teacher_action: torch.Tensor,
  student_action: torch.Tensor,
  probability: float,
  generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
  if probability == 1.0:
    use_teacher = torch.ones(teacher_action.shape[0], dtype=torch.bool)
  elif probability == 0.0:
    use_teacher = torch.zeros(teacher_action.shape[0], dtype=torch.bool)
  else:
    use_teacher = (
      torch.rand(teacher_action.shape[0], generator=generator, device="cpu")
      < probability
    )
  use_teacher = use_teacher.to(device=teacher_action.device)
  # where() chooses complete rows, never an arithmetic average of commands.
  selected = torch.where(use_teacher[:, None], teacher_action, student_action)
  return selected, use_teacher


def _replay_batch(
  snapshot: DistillationSnapshot,
  teacher_action: torch.Tensor,
  collector_iteration: int,
) -> LabeledReplayBatch:
  batch = snapshot.packed.batch_size

  def metadata(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone().to(dtype=torch.int64)

  return LabeledReplayBatch(
    observations=snapshot.packed,
    teacher_action=teacher_action.detach().clone(),
    motion_id=metadata(snapshot.motion_id),
    teacher_id=torch.full_like(
      snapshot.motion_id, snapshot.teacher_code, dtype=torch.int64
    ),
    reference_frame=metadata(snapshot.reference_frame),
    episode_id=metadata(snapshot.segment_id),
    collector_iteration=torch.full(
      (batch,), collector_iteration, dtype=torch.int64, device=teacher_action.device
    ),
  )


class DAggerCollector:
  """Collect valid pre-step labels from one adapter into raw replay."""

  def __init__(
    self,
    adapter: _AdapterLike,
    teacher: FrozenTeacher | _TeacherLike,
    student: ConditionalVAE,
    replay: LabeledReplayBuffer,
  ) -> None:
    self.adapter = adapter
    _require_auto_reset(adapter)
    self.teacher = teacher
    self.student = student
    self.replay = replay
    self._generator = torch.Generator(device="cpu")
    self._snapshot: DistillationSnapshot | None = None
    self._requires_reset = False
    self._rng_initialized = False
    self._invalidation_cause: str | None = None

  def generator_state(self) -> torch.Tensor:
    """Return the persistent collection RNG state for lifecycle checkpoints."""
    return self._generator.get_state().clone()

  def set_generator_state(self, state: torch.Tensor) -> None:
    """Validate and restore the collection RNG state atomically."""
    if (
      not isinstance(state, torch.Tensor)
      or state.dtype != torch.uint8
      or state.ndim != 1
    ):
      raise ValueError("collection generator state must be a uint8 vector")
    self._generator.set_state(state.detach().clone().cpu())
    self._rng_initialized = True

  def invalidate_snapshot(self, *, requires_reset: bool = True) -> None:
    """Discard a snapshot after an external reset/rollout.

    Evaluation uses the same adapter as collection and may advance it outside
    the collector.  Keeping the old snapshot would label a stale state on the
    next collection call, so the runner explicitly invalidates this cache.
    """
    self._snapshot = None
    self._requires_reset = requires_reset
    self._invalidation_cause = None

  def _invalidate_on_failure(self, error: Exception) -> None:
    """Invalidate the continuation and keep the original diagnostic.

    A failed reset, step, capture, or snapshot validation leaves the adapter at
    an unknown instant, so the next collection must reset rather than label the
    retained snapshot.  The caller re-raises ``error`` unchanged.
    """
    self.invalidate_snapshot()
    self._invalidation_cause = f"{type(error).__name__}: {error}"

  def collect(
    self, config: CollectionConfig, *, reset: bool = True
  ) -> CollectionResult:
    """Collect at most ``config.steps`` ticks, including auto-reset ticks."""
    _require_auto_reset(self.adapter)
    if self._requires_reset and not reset:
      cause = (
        "an external reset or rollout"
        if self._invalidation_cause is None
        else self._invalidation_cause
      )
      raise CollectionNumericalError(
        f"collection state was invalidated by {cause}; reset=True is required"
      )
    if config.rng_mode == "per_call":
      self._generator.manual_seed(config.seed)
      self._rng_initialized = True
    elif not self._rng_initialized:
      self._generator.manual_seed(config.seed)
      self._rng_initialized = True
    boundaries: list[SegmentBoundary] = []
    diagnostics: list[str] = []
    did_reset = reset or self._snapshot is None
    if did_reset:
      try:
        self._snapshot = self.adapter.reset(seed=config.seed)
      except Exception as exc:
        self._invalidate_on_failure(exc)
        raise
      self._requires_reset = False
      self._invalidation_cause = None
      diagnostics.append("explicit_reset")
    snapshot = self._snapshot
    assert snapshot is not None
    ticks = samples = teacher_steps = student_steps = 0
    if did_reset:
      try:
        segment, generation = _ids(snapshot)
        reset_reason = _event_reason_summary(
          snapshot.boundary_events, snapshot.packed.batch_size
        )
      except Exception as exc:
        self._invalidate_on_failure(exc)
        raise
      boundaries.append(
        SegmentBoundary(
          tuple(range(snapshot.packed.batch_size)),
          "explicit_reset" if reset_reason is None else reset_reason,
          tuple(int(value) for value in segment.tolist()),
          tuple(int(value) for value in segment.tolist()),
          tuple(int(value) for value in generation.tolist()),
          tuple(int(value) for value in generation.tolist()),
        )
      )
    disagreements: list[float] = []
    fresh_batches: list[LabeledReplayBatch] = []
    if config.steps == 0:
      return CollectionResult(
        0, 0, 0, 0, tuple(boundaries), 0.0, tuple(diagnostics), None
      )
    for _ in range(config.steps):
      try:
        teacher_action = _teacher_action(self.teacher, snapshot)
        _finite("packed reference", snapshot.packed.reference)
        _finite("packed conditioning", snapshot.packed.conditioning)
        # A valid pre-failure label is retained even if student inference below
        # produces a non-finite command.  No such command can reach ``step``.
        fresh_batch = _replay_batch(
          snapshot, teacher_action, config.collector_iteration
        )
      except Exception as exc:
        self._invalidate_on_failure(exc)
        raise
      self.replay.insert(fresh_batch)
      fresh_batches.append(fresh_batch)
      samples += snapshot.packed.batch_size
      try:
        student_action = _student_action(
          self.student, snapshot, config.rollout_latent, self._generator
        )
        # The selected command is checked immediately before the unchanged
        # adapter action path; no NaN can reach the simulator.
        selected, use_teacher = _select_action(
          teacher_action, student_action, config.teacher_probability, self._generator
        )
        _finite("selected action", selected)
      except Exception as exc:
        self._invalidate_on_failure(exc)
        raise
      disagreement = (teacher_action - student_action).abs().mean().item()
      disagreements.append(float(disagreement))
      teacher_steps += int(use_teacher.sum().item())
      student_steps += int((~use_teacher).sum().item())
      try:
        step = self.adapter.step(selected)
      except Exception as exc:
        self._invalidate_on_failure(exc)
        raise
      try:
        next_snapshot = step.snapshot
        terminated = _as_bool_vector(
          "terminated", step.terminated, snapshot.packed.batch_size
        )
        time_outs = _as_bool_vector(
          "time_outs", step.time_outs, snapshot.packed.batch_size
        )
        boundary = _boundary(
          snapshot,
          next_snapshot,
          terminated,
          time_outs,
          step.events,
        )
      except Exception as exc:
        self._invalidate_on_failure(exc)
        raise
      if boundary is not None:
        boundaries.append(boundary)
      snapshot = next_snapshot
      self._snapshot = snapshot
      self._requires_reset = False
      ticks += 1
    self._snapshot = snapshot
    fresh_batch = _concat_batches(fresh_batches)
    fresh_data = (
      None
      if fresh_batch is None
      else FreshTrainingData(fresh_batch, config.collector_iteration)
    )
    return CollectionResult(
      ticks,
      samples,
      teacher_steps,
      student_steps,
      tuple(boundaries),
      sum(disagreements) / len(disagreements) if disagreements else 0.0,
      tuple(diagnostics),
      fresh_data,
    )


# An explicit factory is convenient for callers that do not need to retain a
# collector object between bounded calls.
def collect_dagger(
  adapter: _AdapterLike,
  teacher: FrozenTeacher | _TeacherLike,
  student: ConditionalVAE,
  replay: LabeledReplayBuffer,
  config: CollectionConfig,
  *,
  reset: bool = True,
) -> CollectionResult:
  return DAggerCollector(adapter, teacher, student, replay).collect(config, reset=reset)


def _metric_vector(value: Any, batch: int) -> torch.Tensor | None:
  if not isinstance(value, torch.Tensor):
    return None
  if value.shape != (batch,) or not value.is_floating_point():
    return None
  if not torch.isfinite(value).all().item():
    return None
  return value.detach().clone()


def _flatten_metric_vectors(
  value: Any, batch: int, prefix: str = ""
) -> dict[str, torch.Tensor]:
  result: dict[str, torch.Tensor] = {}
  if isinstance(value, dict):
    for key, child in value.items():
      name = f"{prefix}_{key}" if prefix else str(key)
      result.update(_flatten_metric_vectors(child, batch, name))
    return result
  vector = _metric_vector(value, batch)
  if vector is not None and prefix:
    result[prefix.replace("/", "_").replace(".", "_")] = vector
  return result


def _tracking_metric_vectors(
  snapshot: DistillationSnapshot, extras: dict[str, Any]
) -> dict[str, torch.Tensor]:
  """Return per-environment metrics at one explicit aligned snapshot time."""
  batch = snapshot.packed.batch_size
  metrics = snapshot.metrics
  if metrics is not None:
    result = {
      "tracking_global_body_pose_error": metrics.global_body_pose_error,
      "tracking_pose_error": metrics.global_body_pose_error,
      "tracking_root_relative_pose_error": metrics.root_relative_pose_error,
      # This is the declared root-relative yaw, not full anchor rotation error.
      "tracking_heading_yaw_error": metrics.root_relative_yaw_error,
      "tracking_heading_error": metrics.root_relative_yaw_error,
      "tracking_anchor_position_error": metrics.anchor_position_error,
    }
    if any(value.shape != (batch,) for value in result.values()):
      raise CollectionNumericalError(
        "physical tracking metrics must be per-environment vectors"
      )
    if any(not torch.isfinite(value).all().item() for value in result.values()):
      raise CollectionNumericalError(
        "physical tracking metrics contain non-finite values"
      )
    return {name: value.detach().clone() for name, value in result.items()}

  result = _flatten_metric_vectors(extras, batch)
  # Synthetic adapters may provide a true yaw metric under either explicit name.
  for name in ("heading_error", "yaw_error"):
    if name in result:
      result["tracking_heading_yaw_error"] = result[name]
      result["tracking_heading_error"] = result[name]
      break
  if "error_body_pos" in result:
    result["tracking_pose_error"] = result["error_body_pos"]
  return result


def _adapter_control_period(
  adapter: _AdapterLike, override: float | None
) -> float | None:
  value = override
  if value is None:
    audit = getattr(adapter, "audit", None)
    value = getattr(audit, "control_period_s", None)
  if value is None:
    return None
  if not isinstance(value, (float, int)) or not torch.isfinite(
    torch.tensor(float(value))
  ):
    raise ValueError("control_period_s must be finite when provided")
  if float(value) <= 0.0:
    raise ValueError("control_period_s must be positive when provided")
  return float(value)


def _finish_outcome(
  terminated: bool,
  timed_out: bool,
  changed: bool,
  complete: bool,
  interrupted: bool,
  interrupted_reason: str | None,
  completion_available: bool | None,
) -> tuple[str | None, bool | None, str | None]:
  if terminated:
    return "failure", completion_available, "terminated"
  if completion_available is True and complete:
    return "reference_complete", True, "reference_completed"
  if timed_out:
    return "timeout", completion_available, "timeout"
  if completion_available is True and interrupted:
    if interrupted_reason is not None and "timer" in interrupted_reason:
      return "timer_resampled", False, interrupted_reason
    if interrupted_reason is not None and "reset" in interrupted_reason:
      return "reset", False, interrupted_reason
    return "teleport", False, interrupted_reason
  if changed:
    # A missing event object is an unavailable completion determination, not a
    # license to infer completion from consecutive frame indices.
    return "teleport", None, "boundary_unavailable"
  return None, None, None


@dataclass(frozen=True, slots=True)
class _BoundaryEvent:
  """One environment's producer attribution for a single step."""

  available: bool
  completed: bool
  interrupted: bool
  completed_generation: int
  reason: str | None


def _event_at(events: Any, index: int, batch: int) -> _BoundaryEvent | None:
  if events is None:
    return None
  available = getattr(events, "available", None)
  completed = getattr(events, "completed", None)
  interrupted = getattr(events, "interrupted", None)
  completed_generation = getattr(events, "completed_generation", None)
  reasons = getattr(events, "reasons", None)
  if not isinstance(available, torch.Tensor) or available.shape != (batch,):
    raise ValueError("malformed reference boundary events")
  if not isinstance(completed, torch.Tensor) or completed.shape != (batch,):
    raise ValueError("malformed reference boundary events")
  if not isinstance(interrupted, torch.Tensor) or interrupted.shape != (batch,):
    raise ValueError("malformed reference boundary events")
  if not isinstance(
    completed_generation, torch.Tensor
  ) or completed_generation.shape != (batch,):
    raise ValueError("malformed reference boundary events")
  if not isinstance(reasons, tuple) or len(reasons) != batch:
    raise ValueError("malformed reference boundary events")
  return _BoundaryEvent(
    bool(available[index]),
    bool(completed[index]),
    bool(interrupted[index]),
    int(completed_generation[index].item()),
    reasons[index],
  )


def evaluate_distillation(
  adapter: _AdapterLike,
  teacher: FrozenTeacher | _TeacherLike,
  student: ConditionalVAE | None = None,
  *,
  mode: EvaluationMode = "student",
  steps: int,
  rollout_latent: RolloutLatent = "mean",
  seed: int = 0,
  control_period_s: float | None = None,
) -> EvaluationResult:
  """Run bounded evaluation with explicit per-segment censoring semantics.

  Metrics are attributed to the pre-step snapshot.  Real adapters provide
  per-environment ``PhysicalTrackingMetrics`` there; synthetic adapters may
  provide per-environment vectors in ``extras``.  A timeout is not claimed as
  reference completion unless the adapter explicitly supplies a completion
  flag *for the generation this segment was stepped at*: a completion the
  producer attributes to a newer generation is a boundary recorded inside the
  step, not evidence about the pre-step segment.  Teleports/timer resamples and
  step caps are censored outcomes.
  """
  _require_auto_reset(adapter)
  if mode not in ("teacher", "student"):
    raise ValueError("mode must be 'teacher' or 'student'")
  if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
    raise ValueError("steps must be a non-negative integer")
  if rollout_latent not in ("mean", "sampled"):
    raise ValueError("rollout_latent must be 'mean' or 'sampled'")
  if mode == "student" and student is None:
    raise ValueError("student mode requires a student model")
  period = _adapter_control_period(adapter, control_period_s)
  generator = torch.Generator(device="cpu").manual_seed(seed)
  was_training = None if student is None else student.training
  if student is not None:
    student.eval()
  try:
    snapshot = adapter.reset(seed=seed)
    batch = snapshot.packed.batch_size
    records: dict[tuple[int, int, int], dict[str, Any]] = {}
    outcomes: dict[tuple[int, int, int], str] = {}
    completion_availability: dict[tuple[int, int, int], bool | None] = {}
    completion_reasons: dict[tuple[int, int, int], str | None] = {}

    def start_segments(value: DistillationSnapshot) -> None:
      for index in range(batch):
        key = (
          index,
          int(value.segment_id[index].item()),
          int(value.generation_id[index].item()),
        )
        records.setdefault(
          key,
          {
            "steps": 0,
            "frames": [],
            "metrics": {},
            "metric_counts": {},
            "last_action": None,
            "action_delta": 0.0,
            "action_transitions": 0,
            "start_frame": None,
            "end_frame": None,
          },
        )

    start_segments(snapshot)
    total_steps = 0
    while total_steps < steps:
      teacher_action = _teacher_action(teacher, snapshot)
      if mode == "teacher":
        action = teacher_action
      else:
        assert student is not None
        action = _student_action(student, snapshot, rollout_latent, generator)
      _finite("evaluation action", action)

      pre_metric_vectors = _tracking_metric_vectors(snapshot, {})
      for index in range(batch):
        key = (
          index,
          int(snapshot.segment_id[index].item()),
          int(snapshot.generation_id[index].item()),
        )
        record = records[key]
        frame = int(snapshot.reference_frame[index].item())
        record["steps"] += 1
        record["frames"].append(frame)
        record["start_frame"] = (
          frame if record["start_frame"] is None else record["start_frame"]
        )
        record["end_frame"] = frame
        record["metrics"]["action_magnitude"] = record["metrics"].get(
          "action_magnitude", 0.0
        ) + float(action[index].abs().mean().item())
        record["metrics"]["teacher_student_disagreement"] = record["metrics"].get(
          "teacher_student_disagreement", 0.0
        ) + float((teacher_action[index] - action[index]).abs().mean().item())
        previous = record["last_action"]
        if previous is not None:
          delta = float((action[index] - previous).abs().mean().item())
          record["action_delta"] += delta
          record["action_transitions"] += 1
        record["last_action"] = action[index].detach().clone()
        for name, vector in pre_metric_vectors.items():
          record["metrics"][name] = record["metrics"].get(name, 0.0) + float(
            vector[index].item()
          )
          record["metric_counts"][name] = record["metric_counts"].get(name, 0) + 1

      step = adapter.step(action)
      next_snapshot = step.snapshot
      post_metric_vectors = (
        {}
        if pre_metric_vectors
        else _tracking_metric_vectors(next_snapshot, step.extras)
      )
      if post_metric_vectors:
        for index in range(batch):
          key = (
            index,
            int(snapshot.segment_id[index].item()),
            int(snapshot.generation_id[index].item()),
          )
          record = records[key]
          for name, vector in post_metric_vectors.items():
            record["metrics"][name] = record["metrics"].get(name, 0.0) + float(
              vector[index].item()
            )
            record["metric_counts"][name] = record["metric_counts"].get(name, 0) + 1
      terminated = _as_bool_vector("terminated", step.terminated, batch)
      time_outs = _as_bool_vector("time_outs", step.time_outs, batch)
      for index in range(batch):
        old_key = (
          index,
          int(snapshot.segment_id[index].item()),
          int(snapshot.generation_id[index].item()),
        )
        new_key = (
          index,
          int(next_snapshot.segment_id[index].item()),
          int(next_snapshot.generation_id[index].item()),
        )
        changed = old_key != new_key
        event = _event_at(step.events, index, batch)
        event_available = None if event is None else event.available
        # A completion attributed to a newer generation belongs to a boundary
        # after the pre-step segment (a reset/timer resample whose freshly
        # sampled frame wrapped in the same step), so it cannot complete the
        # segment this snapshot names.
        event_completed = (
          event is not None
          and event.completed
          and event.completed_generation == int(snapshot.generation_id[index].item())
        )
        event_interrupted = False if event is None else event.interrupted
        event_reason = None if event is None else event.reason
        outcome, completion_known, outcome_reason = _finish_outcome(
          bool(terminated[index]),
          bool(time_outs[index]),
          changed,
          event_completed,
          event_interrupted,
          event_reason,
          event_available,
        )
        if outcome is not None:
          outcomes.setdefault(old_key, outcome)
          completion_availability.setdefault(old_key, completion_known)
          completion_reasons.setdefault(old_key, outcome_reason)
      snapshot = next_snapshot
      start_segments(snapshot)
      total_steps += 1

    for key, record in records.items():
      if record["steps"] > 0 and key not in outcomes:
        outcomes[key] = "step_cap"
        completion_availability[key] = None
        completion_reasons[key] = "step_cap"

    segments: list[EvaluationSegment] = []
    for (index, segment_id, generation_id), record in records.items():
      count = int(record["steps"])
      if count == 0:
        continue
      metrics = {
        name: value / record["metric_counts"].get(name, count)
        for name, value in record["metrics"].items()
      }
      transitions = int(record["action_transitions"])
      if transitions:
        delta = record["action_delta"] / transitions
        metrics["action_delta_normalized"] = delta
        metrics["action_delta_valid_transitions"] = float(transitions)
        if period is not None:
          metrics["action_rate_normalized_per_s"] = delta / period
      outcome = outcomes.get((index, segment_id, generation_id), "step_cap")
      segments.append(
        EvaluationSegment(
          index,
          segment_id,
          generation_id,
          count,
          outcome == "reference_complete",
          outcome == "failure",
          outcome == "step_cap",
          metrics,
          outcome,
          record["start_frame"],
          record["end_frame"],
          len(set(record["frames"])),
          max(record["metric_counts"].values(), default=0),
          transitions,
          completion_availability.get((index, segment_id, generation_id)),
          completion_reasons.get((index, segment_id, generation_id)),
        )
      )

    denominator = len(segments)
    known = [
      item for item in segments if item.outcome in ("reference_complete", "failure")
    ]
    all_metrics: dict[str, float] = {
      "completion_known_segments": float(len(known)),
      "timeout_rate": (
        sum(item.outcome == "timeout" for item in segments) / denominator
        if denominator
        else 0.0
      ),
      "teleport_rate": (
        sum(item.outcome == "teleport" for item in segments) / denominator
        if denominator
        else 0.0
      ),
      "timer_resampled_rate": (
        sum(item.outcome == "timer_resampled" for item in segments) / denominator
        if denominator
        else 0.0
      ),
      "reset_rate": (
        sum(item.outcome == "reset" for item in segments) / denominator
        if denominator
        else 0.0
      ),
      "step_cap_rate": (
        sum(item.outcome == "step_cap" for item in segments) / denominator
        if denominator
        else 0.0
      ),
      "reference_coverage_mean": (
        sum(item.reference_coverage for item in segments) / denominator
        if denominator
        else 0.0
      ),
    }
    if known:
      all_metrics["completion_rate"] = sum(
        item.outcome == "reference_complete" for item in known
      ) / len(known)
      all_metrics["failure_rate"] = sum(
        item.outcome == "failure" for item in known
      ) / len(known)
    for name in sorted({name for item in segments for name in item.metrics}):
      values = [item.metrics[name] for item in segments if name in item.metrics]
      if values:
        all_metrics[name] = sum(values) / len(values)
    aligned_time = (
      snapshot.metrics.aligned_time if snapshot.metrics is not None else None
    )
    return EvaluationResult(
      mode,
      rollout_latent,
      total_steps,
      tuple(segments),
      all_metrics,
      {
        "seed": seed,
        "step_cap": steps,
        "num_envs": batch,
        "control_period_s": period,
        "action_units": "normalized_action_units",
        "action_rate_units": (
          "normalized_action_units_per_second" if period is not None else None
        ),
        "action_rate_denominator": "valid_same_segment_transitions",
        "metric_aligned_time": aligned_time,
        "manual_reset_policy": "rejected",
      },
    )
  finally:
    if student is not None and was_training is not None:
      student.train(was_training)


__all__ = [
  "CollectionConfig",
  "CollectionNumericalError",
  "CollectionResult",
  "DAggerCollector",
  "EvaluationMode",
  "EvaluationResult",
  "EvaluationSegment",
  "RolloutLatent",
  "SegmentBoundary",
  "collect_dagger",
  "evaluate_distillation",
]
