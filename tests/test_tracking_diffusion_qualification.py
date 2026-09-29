"""CPU qualification precedence and fail-closed evidence tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from tracking_diffusion_fixtures import make_state

from mjlab.tasks.tracking.diffusion import (
  PpoObservation,
  QualificationConfig,
  QualificationTracker,
  StepEvidence,
  VaeObservation,
  parse_step_evidence,
)


def _safe_observation() -> VaeObservation:
  return VaeObservation(
    make_state(),
    np.zeros(68),
    np.zeros(99),
    "motion",
    0,
    0.0,
    anchor_z_error=0.0,
    gravity_z_error=0.0,
    end_effector_z_error=0.0,
  )


def _safe_ppo_observation() -> PpoObservation:
  return PpoObservation(make_state(), np.zeros(102), "motion", 0, 0.0)


def _complete(
  tracker: QualificationTracker, *, final: StepEvidence | None = None
) -> None:
  for index in range(tracker.config.control_steps):
    tracker.observe(
      "vae",
      final
      if index == tracker.config.control_steps - 1 and final
      else StepEvidence(True),
      _safe_observation(),
    )


def test_final_step_physical_fall_precedes_success() -> None:
  tracker = QualificationTracker()
  _complete(tracker, final=StepEvidence(True, physical_fall=True))
  result = tracker.finish()
  assert not result.qualified
  assert result.reason == "physical_fall"


def test_early_timeout_is_incomplete() -> None:
  tracker = QualificationTracker(QualificationConfig(control_steps=2, retained_steps=2))
  tracker.observe("vae", StepEvidence(True, truncated=True), _safe_observation())
  result = tracker.finish()
  assert result.label == "incomplete"
  assert result.reason == "early_termination"


def test_missing_terminal_evidence_is_not_survival() -> None:
  tracker = QualificationTracker(QualificationConfig(control_steps=2, retained_steps=2))
  tracker.observe("vae", None)
  result = tracker.finish()
  assert not result.qualified
  assert result.reason == "missing_terminal_evidence"


def test_reference_guards_stop_after_handoff_and_label_hybrid() -> None:
  tracker = QualificationTracker(QualificationConfig(control_steps=3, retained_steps=2))
  tracker.observe(
    "vae",
    StepEvidence(True, reference_boundary=True, clip_ended=True),
    _safe_observation(),
  )
  tracker.observe(
    "ppo", StepEvidence(True, tracking_rejection=True), _safe_ppo_observation()
  )
  tracker.observe("ppo", StepEvidence(True), _safe_ppo_observation())
  result = tracker.finish()
  assert result.qualified
  assert result.label == "vae_then_standing"


def test_missing_post_step_vae_observation_is_rejected() -> None:
  tracker = QualificationTracker(QualificationConfig(control_steps=1, retained_steps=1))
  tracker.observe("vae", StepEvidence(True))
  assert tracker.finish().reason == "vae_tracking_rejection"


def test_post_step_quaternion_tilt_is_rejected_without_explicit_metric() -> None:
  tracker = QualificationTracker(QualificationConfig(control_steps=1, retained_steps=1))
  tilted = replace(
    make_state(),
    root_quaternion_wxyz=np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]),
  )
  observation = VaeObservation(
    tilted,
    np.zeros(68),
    np.zeros(99),
    "motion",
    0,
    0.0,
    anchor_z_error=0.0,
    gravity_z_error=0.0,
    end_effector_z_error=0.0,
  )
  tracker.observe("vae", StepEvidence(True), observation)
  assert tracker.finish().reason == "physical_fall"


def test_missing_vae_reference_metrics_reject_control() -> None:
  tracker = QualificationTracker(QualificationConfig(control_steps=1, retained_steps=1))
  observation = VaeObservation(
    make_state(),
    np.zeros(68),
    np.zeros(99),
    "motion",
    0,
    0.0,
  )
  tracker.observe("vae", StepEvidence(True), observation)
  assert tracker.finish().reason == "vae_tracking_rejection"


def test_missing_post_step_ppo_observation_is_rejected() -> None:
  tracker = QualificationTracker(QualificationConfig(control_steps=1, retained_steps=1))
  tracker.observe("ppo", StepEvidence(True))
  result = tracker.finish()
  assert not result.qualified
  assert result.reason == "physical_fall"


def test_ppo_phase_post_step_quaternion_tilt_is_rejected() -> None:
  tracker = QualificationTracker(QualificationConfig(control_steps=1, retained_steps=1))
  tilted = replace(
    make_state(),
    root_quaternion_wxyz=np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]),
  )
  observation = PpoObservation(tilted, np.zeros(102), "motion", 0, 0.0)
  tracker.observe("ppo", StepEvidence(True), observation)
  result = tracker.finish()
  assert not result.qualified
  assert result.reason == "physical_fall"


def test_parse_step_evidence_preserves_boundary_and_reset() -> None:
  evidence = parse_step_evidence(
    {
      "available": True,
      "terminated": False,
      "truncated": False,
      "reference_boundary": True,
      "reset": True,
    }
  )
  assert evidence.reference_boundary
  assert evidence.reset
  assert evidence.terminal.available
