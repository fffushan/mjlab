"""Standing-profile trial accounting for bounded evaluation.

A standing evaluation asks one question per environment: starting from the
standing pose, how long does the row survive?  A reference-wrap teleport inside
that rollout is a continuation of the same trial, so it must never be counted as
another standing trial.  These tests pin that contract, the mutual exclusivity of
the early-window buckets, and the reset-perturbation report, all on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pytest
import torch

from mjlab.tasks.tracking.distillation.adapter import (
  DistillationSnapshot,
  DistillationStep,
  InitializationKind,
)
from mjlab.tasks.tracking.distillation.collector import (
  EvaluationSegment,
  evaluate_distillation,
  standing_trial_buckets,
)
from mjlab.tasks.tracking.distillation.environment import ReferenceBoundaryEvents
from mjlab.tasks.tracking.distillation.model import ConditionalVAE
from mjlab.tasks.tracking.distillation.observations import (
  ObservationSnapshot,
  pack_observations,
)
from mjlab.tasks.tracking.distillation.vae_config import (
  DEFAULT_SCHEMA,
  ModelSettings,
)

BATCH = 1
WINDOW = 4


def _snapshot(
  step: int, segment: int, generation: int, kind: int
) -> DistillationSnapshot:
  values = torch.full((BATCH, 31), float(step))
  features = ObservationSnapshot(
    reference_q=values,
    reference_dq=values + 1,
    anchor_orientation_error=torch.zeros(BATCH, 6),
    projected_gravity=torch.zeros(BATCH, 3),
    gyro=torch.zeros(BATCH, 3),
    relative_joint_q=values + 2,
    joint_dq=values + 3,
    previous_action=torch.full((BATCH, 31), float(step - 1)),
  )
  return DistillationSnapshot(
    teacher_observation=torch.cat((values, values, torch.zeros(BATCH, 102)), 1),
    features=features,
    packed=pack_observations(features),
    teacher_id="multiple",
    teacher_code=-1,
    motion_id=torch.tensor([7] * BATCH),
    reference_frame=torch.full((BATCH,), step),
    segment_id=torch.full((BATCH,), segment),
    generation_id=torch.full((BATCH,), generation),
    teacher_codes=torch.zeros(BATCH, dtype=torch.int64),
    initialization_kind=torch.full((BATCH,), kind, dtype=torch.int64),
    segment_initial_reference_frame=torch.zeros(BATCH, dtype=torch.int64),
    segment_age=torch.zeros(BATCH, dtype=torch.int64),
  )


def _events(reason: str, generation: int) -> ReferenceBoundaryEvents:
  available = torch.ones(BATCH, dtype=torch.bool)
  completed = torch.tensor([reason == "reference_completed"] * BATCH)
  interrupted = torch.tensor([reason != "reference_completed"] * BATCH)
  return ReferenceBoundaryEvents(
    available=available,
    completed=completed,
    interrupted=interrupted,
    pre_generation=torch.full((BATCH,), generation - 1, dtype=torch.long),
    post_generation=torch.full((BATCH,), generation, dtype=torch.long),
    completed_generation=torch.full(
      (BATCH,), generation if reason == "reference_completed" else -1, dtype=torch.long
    ),
    interrupted_generation=torch.full(
      (BATCH,), -1 if reason == "reference_completed" else generation, dtype=torch.long
    ),
    reasons=tuple(reason for _ in range(BATCH)),
  )


@dataclass
class ScriptedAdapter:
  """Deterministic adapter whose segment boundaries are fully scripted.

  ``script`` holds one entry per step: the reason that boundary carried, so a
  test can place a full reset, a wrap, or a timer resample exactly where it
  matters instead of relying on an incidental generation change.
  """

  script: tuple[str, ...] = ()
  auto_reset: bool = True

  def __post_init__(self) -> None:
    self.step_calls = 0
    self.segment = 0
    self.generation = 0
    self.reasons: list[str] = []
    self.current = _snapshot(0, 0, 0, int(InitializationKind.STANDING))

  def reset(self, seed: int | None = None) -> DistillationSnapshot:
    del seed
    self.segment = 0
    self.generation = 0
    self.current = _snapshot(0, 0, 0, int(InitializationKind.STANDING))
    return self.current

  def step(self, action: torch.Tensor) -> DistillationStep:
    del action
    self.step_calls += 1
    reason = (
      self.script[self.step_calls - 1]
      if self.step_calls <= len(self.script)
      else "none"
    )
    self.reasons.append(reason)
    if reason != "none":
      # Every scripted boundary starts a new segment for the whole batch, which
      # is what makes the trial/continuation split observable.
      self.segment += 1
      self.generation += 1
    kind = (
      int(InitializationKind.STANDING)
      if reason == "reset"
      else int(InitializationKind.REFERENCE)
    )
    self.current = _snapshot(self.step_calls, self.segment, self.generation, kind)
    return DistillationStep(
      snapshot=self.current,
      reward=torch.zeros(BATCH),
      terminated=torch.zeros(BATCH, dtype=torch.bool),
      time_outs=torch.zeros(BATCH, dtype=torch.bool),
      extras={
        "error_body_pos": torch.ones(BATCH),
        "heading_error": torch.ones(BATCH) * 2,
      },
      events=None if reason == "none" else _events(reason, self.generation),
    )


def _teacher() -> Any:
  class _Teacher:
    action_dim = 31

    def label(self, observations: torch.Tensor) -> torch.Tensor:
      return torch.full((observations.shape[0], 31), 2.0)

  return _Teacher()


def _student() -> ConditionalVAE:
  return ConditionalVAE(DEFAULT_SCHEMA, ModelSettings(hidden_dims=(8, 8)))


def _run(adapter: ScriptedAdapter, *, steps: int, standing: bool):
  return evaluate_distillation(
    adapter,
    _teacher(),
    None,
    mode="teacher",
    steps=steps,
    seed=0,
    standing_trials=standing,
    trial_window_steps=WINDOW,
  )


def test_wrap_is_a_continuation_not_a_new_standing_trial() -> None:
  # One full reset at the start, a reference wrap mid-rollout, and a full reset
  # near the end: two trials, and the wrap must not add a third.
  adapter = ScriptedAdapter(script=("reference_completed", "none", "reset", "none"))
  result = _run(adapter, steps=6, standing=True)

  trials = [segment for segment in result.segments if segment.is_trial]
  continuations = [segment for segment in result.segments if not segment.is_trial]
  assert len(trials) == 2
  assert len(continuations) == 1
  assert result.metrics["trials"] == 2.0
  assert result.metrics["continuation_segments"] == 1.0
  # The continuation came from a wrap and was recorded as such.
  assert continuations[0].start_reason == "reference_completed"
  # The mid-rollout reset is labelled with its boundary; the first trial began
  # before the first step, so no boundary event exists for it and the row's
  # first segment is a trial by construction instead of by a fabricated reason.
  assert trials[0].start_reason is None
  assert trials[1].start_reason == "reset"


def test_timer_resample_is_also_a_continuation() -> None:
  adapter = ScriptedAdapter(script=("timer_resampled", "none", "none"))
  result = _run(adapter, steps=4, standing=True)

  assert result.metrics["trials"] == 1.0
  assert result.metrics["continuation_segments"] == 1.0


def test_reference_profile_keeps_every_segment_accounting() -> None:
  adapter = ScriptedAdapter(script=("reference_completed", "none"))
  result = _run(adapter, steps=3, standing=False)

  # No trial semantics, no trial keys, and no segment claims to be a trial.
  assert all(not segment.is_trial for segment in result.segments)
  assert result.trial_window_steps is None
  assert not any(name.startswith("trial") for name in result.metrics)
  assert "trials" not in result.metrics
  assert "trial_semantics" not in result.settings


def test_trial_buckets_are_mutually_exclusive_and_cover_every_trial() -> None:
  def segment(steps: int, failed: bool, outcome: str) -> EvaluationSegment:
    return EvaluationSegment(
      env_index=0,
      segment_id=0,
      generation_id=0,
      steps=steps,
      completed=outcome == "reference_complete",
      failed=failed,
      capped=outcome == "step_cap",
      metrics={},
      outcome=outcome,
      is_trial=True,
    )

  segments = [
    segment(WINDOW, True, "failure"),  # failure on the last window step
    segment(WINDOW + 1, True, "failure"),  # failure after the window
    segment(WINDOW, False, "step_cap"),  # survived at least the window
    segment(WINDOW - 1, False, "reference_complete"),  # short clip
    segment(WINDOW - 1, False, "timeout"),  # censored before the window
    replace(segment(WINDOW - 1, True, "failure"), is_trial=False),  # continuation
  ]
  counts = standing_trial_buckets(segments, WINDOW)

  assert counts["trials"] == 5
  assert counts["failures_within_window"] == 1
  assert counts["failures_after_window"] == 1
  assert counts["survived_window"] == 1
  assert counts["short_clip_completions"] == 1
  assert counts["censored"] == 1
  buckets = sum(
    counts[name]
    for name in (
      "failures_within_window",
      "failures_after_window",
      "survived_window",
      "short_clip_completions",
      "censored",
    )
  )
  assert buckets == counts["trials"]


def test_trial_buckets_reject_a_non_positive_window() -> None:
  with pytest.raises(ValueError, match="window_steps must be positive"):
    standing_trial_buckets([], 0)


def test_standing_trials_reject_a_non_positive_window() -> None:
  with pytest.raises(ValueError, match="trial_window_steps must be positive"):
    evaluate_distillation(
      ScriptedAdapter(),
      _teacher(),
      None,
      mode="teacher",
      steps=1,
      standing_trials=True,
      trial_window_steps=0,
    )


def test_per_motion_report_separates_trials_from_continuations() -> None:
  adapter = ScriptedAdapter(script=("reference_completed", "none", "reset", "none"))
  result = _run(adapter, steps=6, standing=True)
  per_motion = result.per_motion

  assert len(per_motion) == 1
  entry = per_motion[0]
  assert entry.trial_counts is not None
  assert entry.trial_counts["trials"] == 2
  assert entry.continuation_segments == 1
  # Trial denominators live in their own field, so the aggregate segment count
  # can still include the continuation without hiding it.
  assert entry.segments == len(result.segments)
  assert entry.as_dict()["trial_counts"]["trials"] == 2
