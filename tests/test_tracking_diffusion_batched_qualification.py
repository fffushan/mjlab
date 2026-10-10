"""Regression tests for vectorized qualification row identity."""

from __future__ import annotations

import numpy as np
import pytest
from tracking_diffusion_fixtures import make_state

from mjlab.tasks.tracking.diffusion import (
  QualificationConfig,
  StepEvidence,
  StepEvidenceBatch,
  VaeObservationBatch,
  WorldState,
)
from mjlab.tasks.tracking.diffusion.qualification import BatchedQualificationTracker


def _observation(rows: int, *, bad_reference: bool = False) -> VaeObservationBatch:
  return VaeObservationBatch(
    WorldState.stack([make_state()] * rows),
    np.zeros((rows, 68)),
    np.zeros((rows, 99)),
    ("motion",) * rows,
    np.zeros(rows, dtype=np.int64),
    np.zeros(rows),
    anchor_z_error=np.full(rows, 1.0 if bad_reference else 0.0),
    gravity_z_error=np.zeros(rows),
    end_effector_z_error=np.zeros(rows),
  )


@pytest.mark.parametrize("ids", [[1], [2], [3, 1], [2, 0]])
@pytest.mark.parametrize("terminal_field", ["terminated", "truncated"])
def test_early_terminal_marks_exact_global_slot(ids, terminal_field) -> None:
  tracker = BatchedQualificationTracker(4)
  evidence = StepEvidenceBatch.from_values(
    [StepEvidence(True, **{terminal_field: True})]
    + [StepEvidence(True)] * (len(ids) - 1)
  )
  tracker.observe(ids, np.zeros(len(ids), dtype=bool), evidence, _observation(len(ids)))
  expected = np.zeros(4, dtype=bool)
  expected[ids[0]] = True
  np.testing.assert_array_equal(tracker.interrupted, expected)
  np.testing.assert_array_equal(tracker.done, expected)
  assert tracker.finish(ids[0]).reason == "early_termination"


def test_mixed_vae_ppo_batch_keeps_vae_reference_guards() -> None:
  tracker = BatchedQualificationTracker(2)
  tracker.observe(
    [0, 1],
    [False, True],
    StepEvidenceBatch.from_values([StepEvidence(True)] * 2),
    _observation(2, bad_reference=True),
  )
  assert tracker.finish(0).reason == "vae_tracking_rejection"
  assert tracker.failure_codes[1] == 0


def test_failure_precedence_and_reset_preserve_other_rows() -> None:
  tracker = BatchedQualificationTracker(4)
  tracker.observe(
    [3, 1],
    [False, False],
    StepEvidenceBatch.from_values(
      [StepEvidence(True, terminated=True, physical_fall=True), StepEvidence(True)]
    ),
    _observation(2),
  )
  assert tracker.finish(3).reason == "physical_fall"
  assert not tracker.interrupted.any()
  tracker.reset([3])
  assert tracker.steps[1] == 1
  assert not tracker.done.any()


@pytest.mark.parametrize("terminal_field", ["terminated", "truncated"])
def test_terminal_at_boundary_requires_reset_not_handoff(terminal_field) -> None:
  tracker = BatchedQualificationTracker(2)
  tracker.observe(
    [1],
    [False],
    StepEvidenceBatch.from_values(
      [StepEvidence(True, reference_boundary=True, **{terminal_field: True})]
    ),
    _observation(1),
  )
  assert tracker.done[1]
  assert not tracker.saw_handoff[1]
  assert tracker.finish(1).reason == "early_termination"


def test_nonterminal_boundary_continues_into_recovery() -> None:
  tracker = BatchedQualificationTracker(
    2, QualificationConfig(control_steps=2, retained_steps=1)
  )
  tracker.observe(
    [1],
    [False],
    StepEvidenceBatch.from_values([StepEvidence(True, reference_boundary=True)]),
    _observation(1),
  )
  assert not tracker.done[1]
  tracker.observe(
    [1], [True], StepEvidenceBatch.from_values([StepEvidence(True)]), _observation(1)
  )
  assert tracker.finish(1).label == "vae_then_standing"
