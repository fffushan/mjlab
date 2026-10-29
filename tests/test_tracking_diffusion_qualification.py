"""CPU qualification tests for the D1 endpoint verdict.

The collection is a stability test, so integrity checks run on every row and the
four stability limits run once, on the last row of a rollout.  These tests cover
that split, the reason precedence, and the per-rollout summary.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from tracking_diffusion_fixtures import make_state

from mjlab.tasks.tracking.diffusion import (
  QualificationConfig,
  QualificationError,
  QualificationTracker,
  StepEvidence,
  VaeObservation,
  parse_step_evidence,
)


def _tilted_state(tilt_degrees: float):
  """A state whose root is rolled by ``tilt_degrees`` from upright."""
  half = np.radians(tilt_degrees) / 2.0
  return replace(
    make_state(), root_quaternion_wxyz=np.array([np.cos(half), np.sin(half), 0.0, 0.0])
  )


def _observation(
  *,
  tilt_degrees: float = 0.0,
  anchor_z_error: float = 0.0,
  gravity_z_error: float = 0.0,
  end_effector_z_error: float = 0.0,
  state: object | None = None,
) -> VaeObservation:
  return VaeObservation(
    _tilted_state(tilt_degrees) if state is None else state,  # type: ignore[arg-type]
    np.zeros(68),
    np.zeros(99),
    "motion",
    0,
    0.0,
    anchor_z_error=anchor_z_error,
    gravity_z_error=gravity_z_error,
    end_effector_z_error=end_effector_z_error,
  )


def _endpoint(
  tracker: QualificationTracker, observation: VaeObservation | None
) -> None:
  tracker.observe(
    StepEvidence(True, reference_boundary=True, clip_ended=True), observation
  )


def test_a_clean_endpoint_row_qualifies() -> None:
  tracker = QualificationTracker()
  tracker.observe(StepEvidence(True), _observation())
  _endpoint(tracker, _observation())
  result = tracker.finish()
  assert result.qualified
  assert result.label == "endpoint_pass"
  assert result.reason is None
  assert result.steps == 2


def test_a_rollout_without_an_endpoint_row_is_incomplete() -> None:
  tracker = QualificationTracker()
  tracker.observe(StepEvidence(True), _observation())
  result = tracker.finish()
  assert not result.qualified
  assert result.label == "incomplete"
  assert result.reason == "missing_endpoint_evidence"


def test_a_mid_rollout_deviation_is_kept_when_the_endpoint_is_clean() -> None:
  """The paper's rule: only the last row decides stability."""
  tracker = QualificationTracker()
  tracker.observe(StepEvidence(True), _observation(tilt_degrees=65.0))
  tracker.observe(StepEvidence(True), _observation(anchor_z_error=0.9))
  _endpoint(tracker, _observation())
  result = tracker.finish()
  assert result.qualified
  assert result.fall_frames == 0
  assert result.max_anchor_z_error_m == pytest.approx(0.9)


def test_a_prone_endpoint_row_fails_and_counts_its_fall_frames() -> None:
  tracker = QualificationTracker()
  tracker.observe(StepEvidence(True), _observation(tilt_degrees=75.0))
  _endpoint(tracker, _observation(tilt_degrees=120.0))
  result = tracker.finish()
  assert not result.qualified
  assert result.reason == "endpoint_tilt"
  assert result.fall_frames == 2
  assert result.min_tilt_degrees == pytest.approx(75.0)


@pytest.mark.parametrize(
  ("metric", "value", "reason"),
  (
    ("anchor_z_error", -0.3, "endpoint_anchor_z_error"),
    ("gravity_z_error", 0.85, "endpoint_gravity_z_error"),
    ("end_effector_z_error", 0.3, "endpoint_end_effector_z_error"),
  ),
)
def test_each_endpoint_limit_has_its_own_reason(metric, value, reason) -> None:
  tracker = QualificationTracker()
  _endpoint(tracker, _observation(**{metric: value}))
  result = tracker.finish()
  assert not result.qualified
  assert result.reason == reason


def test_the_endpoint_limits_come_from_the_config() -> None:
  config = QualificationConfig(anchor_z_error_m=0.01)
  tracker = QualificationTracker(config)
  _endpoint(tracker, _observation(anchor_z_error=0.02))
  assert tracker.finish().reason == "endpoint_anchor_z_error"


def test_integrity_failures_end_the_rollout_immediately() -> None:
  for evidence, reason in (
    (StepEvidence(False), "missing_terminal_evidence"),
    (StepEvidence(True, finite=False), "nonfinite_post_step"),
    (StepEvidence(True, reset=True), "unexpected_reset"),
    (StepEvidence(True, teleport=True), "reference_teleport"),
  ):
    tracker = QualificationTracker()
    tracker.observe(evidence, _observation())
    assert tracker.done
    result = tracker.finish()
    assert not result.qualified
    assert result.label == "failed"
    assert result.reason == reason


def test_a_truncated_rollout_is_incomplete_not_survival() -> None:
  tracker = QualificationTracker()
  tracker.observe(StepEvidence(True, truncated=True), _observation())
  result = tracker.finish()
  assert result.label == "incomplete"
  assert result.reason == "early_termination"


def test_a_missing_observation_leaves_the_verdict_unmeasured() -> None:
  tracker = QualificationTracker()
  _endpoint(tracker, None)
  result = tracker.finish()
  assert not result.qualified
  assert result.label == "incomplete"
  assert result.reason == "missing_stability_metrics"


def test_a_missing_metric_on_the_endpoint_row_is_refused() -> None:
  tracker = QualificationTracker()
  partial = VaeObservation(
    make_state(),
    np.zeros(68),
    np.zeros(99),
    "motion",
    0,
    0.0,
  )
  _endpoint(tracker, partial)
  assert tracker.finish().reason == "missing_stability_metrics"


def test_an_observation_without_a_quaternion_leaves_the_endpoint_unmeasured() -> None:
  # No root quaternion means no tilt, and a tilt the tracker cannot read must
  # refuse the rollout instead of assuming the robot stood still.
  broken = SimpleNamespace(
    anchor_z_error=0.0, gravity_z_error=0.0, end_effector_z_error=0.0
  )
  tracker = QualificationTracker()
  _endpoint(tracker, broken)  # type: ignore[arg-type]
  result = tracker.finish()
  assert not result.qualified
  assert result.reason == "missing_stability_metrics"


def test_padding_steps_must_be_non_negative() -> None:
  with pytest.raises(QualificationError, match="padding_steps"):
    QualificationConfig(padding_steps=-1)


def test_the_step_bound_closes_a_rollout_that_never_reaches_its_endpoint() -> None:
  from mjlab.tasks.tracking.diffusion.qualification import MAX_ROLLOUT_STEPS

  tracker = QualificationTracker()
  for _ in range(MAX_ROLLOUT_STEPS):
    tracker.observe(StepEvidence(True), _observation())
  assert tracker.done
  with pytest.raises(QualificationError, match="step bound"):
    tracker.observe(StepEvidence(True), _observation())
  result = tracker.finish()
  assert result.label == "incomplete"
  assert result.reason == "rollout_step_cap"


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
