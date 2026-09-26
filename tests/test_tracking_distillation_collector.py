"""Adversarial CPU tests for synchronous collection and evaluation semantics."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
import torch

from mjlab.tasks.tracking.distillation.adapter import (
  DistillationSnapshot,
  DistillationStep,
)
from mjlab.tasks.tracking.distillation.collector import (
  CollectionConfig,
  CollectionNumericalError,
  DAggerCollector,
  evaluate_distillation,
)
from mjlab.tasks.tracking.distillation.environment import ReferenceBoundaryEvents
from mjlab.tasks.tracking.distillation.model import ConditionalVAE
from mjlab.tasks.tracking.distillation.observations import (
  ObservationSnapshot,
  pack_observations,
)
from mjlab.tasks.tracking.distillation.storage import LabeledReplayBuffer
from mjlab.tasks.tracking.distillation.vae_config import DEFAULT_SCHEMA, ModelSettings


class FakeTeacher:
  action_dim = 31

  def __init__(self, value: float = 2.0, nonfinite: bool = False) -> None:
    self.value = value
    self.nonfinite = nonfinite

  def label(self, observations: torch.Tensor) -> torch.Tensor:
    action = torch.full((observations.shape[0], self.action_dim), self.value)
    if self.nonfinite:
      action[0, 0] = float("nan")
    return action


def _snapshot(step: int, segment: int = 0, generation: int = 0) -> DistillationSnapshot:
  batch = 2
  values = torch.full((batch, 31), float(step))
  features = ObservationSnapshot(
    reference_q=values,
    reference_dq=values + 1,
    anchor_orientation_error=torch.zeros(batch, 6),
    projected_gravity=torch.zeros(batch, 3),
    gyro=torch.zeros(batch, 3),
    relative_joint_q=values + 2,
    joint_dq=values + 3,
    previous_action=torch.full((batch, 31), float(step - 1)),
  )
  packed = pack_observations(features)
  return DistillationSnapshot(
    teacher_observation=torch.cat((values, values, torch.zeros(batch, 102)), 1),
    features=features,
    packed=packed,
    teacher_id="fake",
    teacher_code=0,
    motion_id=torch.tensor([4, 4]),
    reference_frame=torch.tensor([step, step]),
    segment_id=torch.tensor([segment, segment]),
    generation_id=torch.tensor([generation, generation]),
  )


@dataclass
class FakeAdapter:
  reset_calls: int = 0
  step_calls: int = 0
  done_on_step: int | None = None
  timeout_on_step: int | None = None
  auto_reset: bool = True
  metric_pose: torch.Tensor | None = None
  metric_heading: torch.Tensor | None = None
  fail_reset: bool = False
  fail_step_after_state: bool = False
  malformed_done: bool = False
  step_events: ReferenceBoundaryEvents | None = None
  reset_events: ReferenceBoundaryEvents | None = None
  snapshots: list[DistillationSnapshot] | None = None

  def __post_init__(self) -> None:
    self.snapshots = []
    self.current = _snapshot(0)
    self.actions: list[torch.Tensor] = []

  def reset(self, seed: int | None = None) -> DistillationSnapshot:
    del seed
    self.reset_calls += 1
    if self.fail_reset:
      raise RuntimeError("reset failed after reset attempt")
    self.current = replace(_snapshot(0), boundary_events=self.reset_events)
    return self.current

  def step(self, action: torch.Tensor) -> DistillationStep:
    self.actions.append(action.detach().clone())
    self.step_calls += 1
    # A generation change is a wrap/resample boundary without done flags.
    generation = 1 if self.step_calls == 2 else 0
    self.current = _snapshot(self.step_calls, generation=generation)
    if self.fail_step_after_state:
      raise RuntimeError("step failed after advancing state")
    terminated = torch.full(
      (2,), self.done_on_step == self.step_calls, dtype=torch.bool
    )
    time_outs = torch.full(
      (2,), self.timeout_on_step == self.step_calls, dtype=torch.bool
    )
    return DistillationStep(
      self.current,
      torch.zeros(2),
      torch.zeros(1, dtype=torch.bool) if self.malformed_done else terminated,
      time_outs,
      {
        "error_body_pos": self.metric_pose
        if self.metric_pose is not None
        else torch.ones(2),
        "heading_error": self.metric_heading
        if self.metric_heading is not None
        else torch.ones(2) * 2,
      },
      events=self.step_events,
    )


def _student() -> ConditionalVAE:
  return ConditionalVAE(DEFAULT_SCHEMA, ModelSettings(hidden_dims=(8, 8)))


def test_collector_labels_pre_step_and_records_actual_previous_action() -> None:
  adapter = FakeAdapter()
  replay = LabeledReplayBuffer(16, DEFAULT_SCHEMA)
  collector = DAggerCollector(adapter, FakeTeacher(), _student(), replay)
  result = collector.collect(
    CollectionConfig(steps=3, teacher_probability=0.0), reset=True
  )

  assert result.samples == 6
  assert adapter.step_calls == 3
  assert result.fresh_data is not None
  assert result.fresh_data.batch.batch_size == 6
  assert result.fresh_data.update_id == 0
  assert any(boundary.reason == "generation" for boundary in result.boundaries)
  assert torch.equal(adapter.actions[0], adapter.actions[0])
  sampled = replay.sample(6, replacement=False)
  # Replay labels are fixed teacher targets, while commands passed to the env
  # are student commands, never an average or a teacher target.
  torch.testing.assert_close(sampled.teacher_action, torch.full((6, 31), 2.0))
  assert sorted(sampled.reference_frame.tolist()) == [0, 0, 1, 1, 2, 2]
  assert not torch.allclose(adapter.actions[0], sampled.teacher_action[:2])


def test_auto_reset_and_manual_reset_are_explicit_boundaries() -> None:
  adapter = FakeAdapter(done_on_step=1)
  replay = LabeledReplayBuffer(8, DEFAULT_SCHEMA)
  collector = DAggerCollector(adapter, FakeTeacher(), _student(), replay)
  first = collector.collect(CollectionConfig(steps=2))
  assert adapter.reset_calls == 1  # the environment auto-resets; collector does not.
  assert any("terminated" in boundary.reason for boundary in first.boundaries)
  second = collector.collect(CollectionConfig(steps=1), reset=True)
  assert adapter.reset_calls == 2
  assert any(boundary.reason == "explicit_reset" for boundary in second.boundaries)
  empty = collector.collect(CollectionConfig(steps=0), reset=True)
  assert empty.ticks == 0
  assert [boundary.reason for boundary in empty.boundaries] == ["explicit_reset"]


def test_timeout_is_completion_but_wrap_is_not_done() -> None:
  adapter = FakeAdapter(timeout_on_step=1)
  replay = LabeledReplayBuffer(8, DEFAULT_SCHEMA)
  collector = DAggerCollector(adapter, FakeTeacher(), _student(), replay)
  result = collector.collect(CollectionConfig(steps=2))
  assert any("timeout" in boundary.reason for boundary in result.boundaries)
  assert any(boundary.reason == "generation" for boundary in result.boundaries)


def test_manual_reset_adapter_is_rejected() -> None:
  adapter = FakeAdapter(auto_reset=False)
  with pytest.raises(ValueError, match="auto-reset"):
    DAggerCollector(
      adapter, FakeTeacher(), _student(), LabeledReplayBuffer(4, DEFAULT_SCHEMA)
    )


def test_numerical_failure_invalidates_persistent_snapshot(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  adapter = FakeAdapter()
  student = _student()
  original = student.mean_inference
  calls = 0

  def fail_on_second(
    reference: torch.Tensor, conditioning: torch.Tensor
  ) -> torch.Tensor:
    nonlocal calls
    calls += 1
    action = original(reference, conditioning)
    if calls == 2:
      action = torch.full_like(action, float("nan"))
    return action

  monkeypatch.setattr(student, "mean_inference", fail_on_second)
  replay = LabeledReplayBuffer(8, DEFAULT_SCHEMA)
  collector = DAggerCollector(adapter, FakeTeacher(), student, replay)
  with pytest.raises(CollectionNumericalError, match="student action"):
    collector.collect(CollectionConfig(steps=3))
  assert adapter.step_calls == 1
  assert replay.size == 4
  with pytest.raises(CollectionNumericalError, match="reset=True"):
    collector.collect(CollectionConfig(steps=1), reset=False)
  collector.collect(CollectionConfig(steps=1), reset=True)


def test_reset_and_step_failures_invalidate_continuation_after_state_change() -> None:
  reset_adapter = FakeAdapter(fail_reset=True)
  collector = DAggerCollector(
    reset_adapter, FakeTeacher(), _student(), LabeledReplayBuffer(4, DEFAULT_SCHEMA)
  )
  with pytest.raises(RuntimeError, match="reset failed"):
    collector.collect(CollectionConfig(steps=1))
  reset_adapter.fail_reset = False
  collector.collect(CollectionConfig(steps=1), reset=True)

  step_adapter = FakeAdapter(fail_step_after_state=True)
  step_collector = DAggerCollector(
    step_adapter, FakeTeacher(), _student(), LabeledReplayBuffer(4, DEFAULT_SCHEMA)
  )
  with pytest.raises(RuntimeError, match="step failed"):
    step_collector.collect(CollectionConfig(steps=1))
  with pytest.raises(CollectionNumericalError, match="reset=True"):
    step_collector.collect(CollectionConfig(steps=1), reset=False)


def test_ordinary_exceptions_invalidate_continuation(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  adapter = FakeAdapter()
  student = _student()
  original = student.mean_inference
  calls = 0

  def fail_on_second(
    reference: torch.Tensor, conditioning: torch.Tensor
  ) -> torch.Tensor:
    nonlocal calls
    calls += 1
    if calls == 2:
      raise ValueError("student inference rejected the batch")
    return original(reference, conditioning)

  monkeypatch.setattr(student, "mean_inference", fail_on_second)
  replay = LabeledReplayBuffer(8, DEFAULT_SCHEMA)
  collector = DAggerCollector(adapter, FakeTeacher(), student, replay)
  # The failing tick keeps its valid teacher label; no simulator step happened.
  with pytest.raises(ValueError, match="student inference rejected the batch"):
    collector.collect(CollectionConfig(steps=3))
  assert adapter.step_calls == 1
  assert replay.size == 4
  with pytest.raises(CollectionNumericalError, match="ValueError: student inference"):
    collector.collect(CollectionConfig(steps=1), reset=False)
  collector.collect(CollectionConfig(steps=1), reset=True)

  class FailingTeacher(FakeTeacher):
    def __init__(self) -> None:
      super().__init__()
      self.failing = True

    def label(self, observations: torch.Tensor) -> torch.Tensor:
      if self.failing:
        self.failing = False
        raise RuntimeError("teacher label failed")
      return super().label(observations)

  failing = FakeAdapter()
  failing_replay = LabeledReplayBuffer(4, DEFAULT_SCHEMA)
  failing_collector = DAggerCollector(
    failing, FailingTeacher(), _student(), failing_replay
  )
  with pytest.raises(RuntimeError, match="teacher label failed"):
    failing_collector.collect(CollectionConfig(steps=1))
  assert failing.step_calls == 0
  assert failing_replay.size == 0
  with pytest.raises(CollectionNumericalError, match="reset=True"):
    failing_collector.collect(CollectionConfig(steps=1), reset=False)
  failing_collector.collect(CollectionConfig(steps=1), reset=True)


def _malformed_events() -> ReferenceBoundaryEvents:
  """Valid at construction, but for the wrong environment batch."""
  return ReferenceBoundaryEvents(
    available=torch.ones(3, dtype=torch.bool),
    completed=torch.zeros(3, dtype=torch.bool),
    interrupted=torch.zeros(3, dtype=torch.bool),
    pre_generation=torch.zeros(3, dtype=torch.long),
    post_generation=torch.zeros(3, dtype=torch.long),
    completed_generation=torch.full((3,), -1, dtype=torch.long),
    interrupted_generation=torch.full((3,), -1, dtype=torch.long),
    reasons=("unavailable", "unavailable", "unavailable"),
  )


def test_malformed_boundary_events_invalidate_continuation() -> None:
  reset_adapter = FakeAdapter(reset_events=_malformed_events())
  collector = DAggerCollector(
    reset_adapter, FakeTeacher(), _student(), LabeledReplayBuffer(4, DEFAULT_SCHEMA)
  )
  with pytest.raises(ValueError, match="malformed reference boundary events"):
    collector.collect(CollectionConfig(steps=1))
  assert reset_adapter.step_calls == 0
  reset_adapter.reset_events = None
  with pytest.raises(CollectionNumericalError, match="reset=True"):
    collector.collect(CollectionConfig(steps=1), reset=False)
  collector.collect(CollectionConfig(steps=1), reset=True)

  step_adapter = FakeAdapter(step_events=_malformed_events())
  step_collector = DAggerCollector(
    step_adapter, FakeTeacher(), _student(), LabeledReplayBuffer(8, DEFAULT_SCHEMA)
  )
  with pytest.raises(ValueError, match="malformed reference boundary events"):
    step_collector.collect(CollectionConfig(steps=1))
  assert step_adapter.step_calls == 1
  step_adapter.step_events = None
  with pytest.raises(CollectionNumericalError, match="reset=True"):
    step_collector.collect(CollectionConfig(steps=1), reset=False)
  step_collector.collect(CollectionConfig(steps=1), reset=True)


def test_malformed_done_vector_after_step_requires_reset() -> None:
  adapter = FakeAdapter(malformed_done=True)
  collector = DAggerCollector(
    adapter, FakeTeacher(), _student(), LabeledReplayBuffer(4, DEFAULT_SCHEMA)
  )
  with pytest.raises(ValueError, match="terminated"):
    collector.collect(CollectionConfig(steps=1))
  assert adapter.step_calls == 1
  with pytest.raises(CollectionNumericalError, match="reset=True"):
    collector.collect(CollectionConfig(steps=1), reset=False)

  adapter = FakeAdapter()
  collector = DAggerCollector(
    adapter, FakeTeacher(), _student(), LabeledReplayBuffer(8, DEFAULT_SCHEMA)
  )
  config = CollectionConfig(
    steps=1, teacher_probability=0.5, rng_mode="persistent", seed=17
  )
  collector.collect(config)
  first = collector.generator_state()
  collector.collect(config, reset=True)
  second = collector.generator_state()
  assert not torch.equal(first, second)
  collector.set_generator_state(first)
  torch.testing.assert_close(collector.generator_state(), first)
  per_call = CollectionConfig(steps=1, rng_mode="per_call", seed=17)
  collector.collect(per_call, reset=True)
  per_call_first = collector.generator_state()
  collector.collect(per_call, reset=True)
  torch.testing.assert_close(collector.generator_state(), per_call_first)

  for probability, expected in ((0.0, 0.0), (1.0, 2.0)):
    adapter = FakeAdapter()
    replay = LabeledReplayBuffer(4, DEFAULT_SCHEMA)
    collector = DAggerCollector(adapter, FakeTeacher(), _student(), replay)
    collector.collect(CollectionConfig(steps=1, teacher_probability=probability))
    assert (
      torch.all(adapter.actions[0] == expected)
      if probability == 1.0
      else torch.any(adapter.actions[0] != expected)
    )


def test_collector_consumes_mixed_pre_step_boundary_events() -> None:
  events = ReferenceBoundaryEvents(
    available=torch.ones(2, dtype=torch.bool),
    completed=torch.tensor([True, False]),
    interrupted=torch.tensor([False, True]),
    pre_generation=torch.tensor([0, 0]),
    post_generation=torch.tensor([0, 1]),
    completed_generation=torch.tensor([0, -1]),
    interrupted_generation=torch.tensor([-1, 0]),
    reasons=("reference_completed", "timer_resampled"),
  )
  adapter = FakeAdapter(step_events=events)
  result = DAggerCollector(
    adapter, FakeTeacher(), _student(), LabeledReplayBuffer(4, DEFAULT_SCHEMA)
  ).collect(CollectionConfig(steps=1))
  assert any(
    "reference_completed" in boundary.reason and "timer_resampled" in boundary.reason
    for boundary in result.boundaries
  )


def _completed_events() -> ReferenceBoundaryEvents:
  return ReferenceBoundaryEvents(
    available=torch.ones(2, dtype=torch.bool),
    completed=torch.ones(2, dtype=torch.bool),
    interrupted=torch.zeros(2, dtype=torch.bool),
    pre_generation=torch.zeros(2, dtype=torch.long),
    post_generation=torch.zeros(2, dtype=torch.long),
    completed_generation=torch.zeros(2, dtype=torch.long),
    interrupted_generation=torch.full((2,), -1, dtype=torch.long),
    reasons=("reference_completed", "reference_completed"),
  )


def test_evaluation_uses_explicit_completion_availability() -> None:
  unavailable = evaluate_distillation(
    FakeAdapter(), FakeTeacher(), _student(), mode="student", steps=1, seed=2
  )
  assert unavailable.completion_rate is None
  assert "completion_rate" not in unavailable.metrics
  completed = evaluate_distillation(
    FakeAdapter(step_events=_completed_events()),
    FakeTeacher(),
    _student(),
    mode="student",
    steps=1,
    seed=2,
  )
  assert completed.completion_rate == pytest.approx(1.0)
  assert all(segment.outcome == "reference_complete" for segment in completed.segments)

  adapter = FakeAdapter(
    metric_pose=torch.tensor([1.0, 3.0]),
    metric_heading=torch.tensor([2.0, 4.0]),
  )
  result = evaluate_distillation(
    adapter, FakeTeacher(), _student(), mode="student", steps=1, seed=4
  )
  by_env = {segment.env_index: segment for segment in result.segments}
  assert by_env[0].metrics["tracking_pose_error"] == pytest.approx(1.0)
  assert by_env[1].metrics["tracking_pose_error"] == pytest.approx(3.0)
  assert by_env[0].metrics["tracking_heading_error"] == pytest.approx(2.0)
  assert by_env[1].metrics["tracking_heading_error"] == pytest.approx(4.0)

  adapter = FakeAdapter()
  replay = LabeledReplayBuffer(4, DEFAULT_SCHEMA)
  collector = DAggerCollector(adapter, FakeTeacher(nonfinite=True), _student(), replay)
  with pytest.raises(CollectionNumericalError, match="teacher label"):
    collector.collect(CollectionConfig(steps=1))
  assert adapter.step_calls == 0
  assert replay.size == 0


class _ResampleOnStepAdapter(FakeAdapter):
  """A step that auto-resets into a new generation, like the real environment."""

  def step(self, action: torch.Tensor) -> DistillationStep:
    step = super().step(action)
    return replace(
      step,
      snapshot=replace(
        step.snapshot,
        segment_id=torch.ones(2, dtype=torch.long),
        generation_id=torch.ones(2, dtype=torch.long),
      ),
    )


def _same_step_resample_and_wrap_events(
  *, completed_generation: int, reason: str
) -> ReferenceBoundaryEvents:
  """Events for a resample whose freshly sampled last frame wrapped at once.

  ``SegmentMotionCommand`` records the reset/timer interruption for the
  pre-step generation and the wrap completion for the generation the resample
  produced, so the completion belongs to a generation after the pre-step one.
  """
  return ReferenceBoundaryEvents(
    available=torch.ones(2, dtype=torch.bool),
    completed=torch.ones(2, dtype=torch.bool),
    interrupted=torch.ones(2, dtype=torch.bool),
    pre_generation=torch.zeros(2, dtype=torch.long),
    post_generation=torch.full((2,), completed_generation + 1, dtype=torch.long),
    completed_generation=torch.full((2,), completed_generation, dtype=torch.long),
    interrupted_generation=torch.zeros(2, dtype=torch.long),
    reasons=(reason, reason),
  )


def test_newer_generation_completion_does_not_complete_pre_step_segment() -> None:
  aligned = evaluate_distillation(
    FakeAdapter(step_events=_completed_events()),
    FakeTeacher(),
    _student(),
    mode="student",
    steps=1,
    seed=3,
  )
  assert all(segment.outcome == "reference_complete" for segment in aligned.segments)
  assert aligned.completion_rate == pytest.approx(1.0)

  # A timed-out env auto-resets inside the step, and that reset resample landed
  # on the clip's last frame, so the same step also wrapped it: the producer
  # reports the interruption for the pre-step generation and a completion for
  # the next one. The pre-step segment timed out.
  timed_out = evaluate_distillation(
    _ResampleOnStepAdapter(
      timeout_on_step=1,
      step_events=_same_step_resample_and_wrap_events(
        completed_generation=1, reason="reset+reference_completed"
      ),
    ),
    FakeTeacher(),
    _student(),
    mode="student",
    steps=1,
    seed=3,
  )
  assert all(segment.generation_id == 0 for segment in timed_out.segments)
  assert all(segment.outcome == "timeout" for segment in timed_out.segments)
  assert not any(segment.completed for segment in timed_out.segments)
  assert all(segment.completion_available is True for segment in timed_out.segments)
  assert timed_out.metrics["timeout_rate"] == pytest.approx(1.0)
  assert timed_out.metrics["completion_known_segments"] == 0.0
  assert timed_out.completion_rate is None
  assert "completion_rate" not in timed_out.metrics

  # The same producer shape from a timer resample is censored as a timer
  # resample instead of completing the pre-step segment.
  timer = evaluate_distillation(
    _ResampleOnStepAdapter(
      step_events=_same_step_resample_and_wrap_events(
        completed_generation=1, reason="timer_resampled+reference_completed"
      )
    ),
    FakeTeacher(),
    _student(),
    mode="student",
    steps=1,
    seed=3,
  )
  assert all(segment.outcome == "timer_resampled" for segment in timer.segments)
  assert timer.metrics["timer_resampled_rate"] == pytest.approx(1.0)
  assert "completion_rate" not in timer.metrics


def test_evaluation_is_bounded_and_does_not_mutate_replay_or_normalizers() -> None:
  adapter = FakeAdapter()
  student = _student()
  replay = LabeledReplayBuffer(4, DEFAULT_SCHEMA)
  collector = DAggerCollector(adapter, FakeTeacher(), student, replay)
  collector.collect(CollectionConfig(steps=1))
  size = replay.size
  state = {
    key: value.clone()
    for key, value in student.state_dict().items()
    if isinstance(value, torch.Tensor)
  }
  result = evaluate_distillation(
    adapter, FakeTeacher(), student, mode="student", steps=2, seed=9
  )
  assert result.steps == 2
  assert result.settings["step_cap"] == 2
  assert replay.size == size
  assert result.metrics["tracking_pose_error"] == pytest.approx(1.0)
  assert result.metrics["tracking_heading_error"] == pytest.approx(2.0)
  assert all(
    torch.equal(state[key], value)
    for key, value in student.state_dict().items()
    if isinstance(value, torch.Tensor)
  )
  assert any(
    not segment.completed and not segment.failed and not segment.capped
    for segment in result.segments
  )
