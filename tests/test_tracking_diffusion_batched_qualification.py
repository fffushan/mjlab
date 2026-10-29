"""Regression tests for vectorized endpoint qualification row identity."""

from __future__ import annotations

import numpy as np
import pytest
from tracking_diffusion_fixtures import make_state

from mjlab.tasks.tracking.diffusion import (
  StepEvidence,
  StepEvidenceBatch,
  VaeObservationBatch,
  WorldState,
)
from mjlab.tasks.tracking.diffusion.qualification import BatchedQualificationTracker


def _observation(
  rows: int, *, anchor_z_error: float = 0.0, tilt_rows: int = 0
) -> VaeObservationBatch:
  states = []
  for row in range(rows):
    state = make_state()
    if row < tilt_rows:
      state = WorldState(
        state.root_position,
        np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]),
        state.root_linear_velocity,
        state.root_angular_velocity,
        state.body_positions,
        state.body_linear_velocities,
      )
    states.append(state)
  return VaeObservationBatch(
    WorldState.stack(states),
    np.zeros((rows, 68)),
    np.zeros((rows, 99)),
    ("motion",) * rows,
    np.zeros(rows, dtype=np.int64),
    np.zeros(rows),
    anchor_z_error=np.full(rows, anchor_z_error),
    gravity_z_error=np.zeros(rows),
    end_effector_z_error=np.zeros(rows),
  )


def _endpoint(rows: int) -> StepEvidenceBatch:
  return StepEvidenceBatch.from_values(
    [StepEvidence(True, reference_boundary=True, clip_ended=True)] * rows
  )


@pytest.mark.parametrize("ids", [[1], [2], [3, 1], [2, 0]])
@pytest.mark.parametrize("terminal_field", ["terminated", "truncated"])
def test_early_terminal_marks_exact_global_slot(ids, terminal_field) -> None:
  tracker = BatchedQualificationTracker(4)
  evidence = StepEvidenceBatch.from_values(
    [StepEvidence(True, **{terminal_field: True})]
    + [StepEvidence(True)] * (len(ids) - 1)
  )
  tracker.observe(ids, evidence, _observation(len(ids)))
  expected = np.zeros(4, dtype=bool)
  expected[ids[0]] = True
  np.testing.assert_array_equal(tracker.interrupted, expected)
  np.testing.assert_array_equal(tracker.done, expected)
  assert tracker.finish(ids[0]).reason == "early_termination"


def test_the_endpoint_row_decides_that_row_only() -> None:
  tracker = BatchedQualificationTracker(2)
  tracker.observe([0, 1], _endpoint(2), _observation(2, anchor_z_error=0.0))
  tracker.last_metrics[1] = np.array([0.0, 0.5, 0.0, 0.0])
  assert tracker.finish(0).qualified
  failed = tracker.finish(1)
  assert not failed.qualified
  assert failed.reason == "endpoint_anchor_z_error"


def test_failure_precedence_and_reset_preserve_other_rows() -> None:
  tracker = BatchedQualificationTracker(4)
  tracker.observe(
    [3, 1],
    StepEvidenceBatch.from_values([StepEvidence(True, reset=True), StepEvidence(True)]),
    _observation(2),
  )
  assert tracker.finish(3).reason == "unexpected_reset"
  assert not tracker.interrupted.any()
  tracker.reset([3])
  assert tracker.steps[1] == 1
  assert not tracker.done.any()


@pytest.mark.parametrize("terminal_field", ["terminated", "truncated"])
def test_the_endpoint_verdict_wins_over_a_same_step_time_out(terminal_field) -> None:
  tracker = BatchedQualificationTracker(2)
  tracker.observe(
    [1],
    StepEvidenceBatch.from_values(
      [StepEvidence(True, reference_boundary=True, **{terminal_field: True})]
    ),
    _observation(1),
  )
  assert tracker.done[1]
  assert not tracker.interrupted[1]
  assert tracker.finish(1).qualified


def test_a_mid_rollout_deviation_does_not_end_the_rollout() -> None:
  tracker = BatchedQualificationTracker(2)
  tracker.observe(
    [1],
    StepEvidenceBatch.from_values([StepEvidence(True)]),
    _observation(1, tilt_rows=1),
  )
  assert not tracker.done[1]
  assert tracker.failure_codes[1] == 0
  assert tracker.fall_frames[1] == 1
  tracker.observe([1], _endpoint(1), _observation(1))
  result = tracker.finish(1)
  assert result.qualified
  assert result.fall_frames == 1
  assert result.min_tilt_degrees == 0.0
