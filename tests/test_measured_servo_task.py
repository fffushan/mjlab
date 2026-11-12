"""Tests for the measured-servo single-clip tracking task.

The task exists so the measured-servo distillation cohort has a ``base_task`` whose
actuator damping matches the exported teacher ONNX: the collection environment builds
from the base task, while the gains the teacher declares come from its own training run.
A base task with vendor damping would therefore feed the teacher an observation the
teacher never saw, so these tests pin the two properties the cohort depends on -- the
per-group damping, and the wrist effort event after the split moved its index.
"""

import pytest

from mjlab.asset_zoo.robots.agibot_x2.measured_servo import (
  WRIST_GROUP_PATTERNS,
  group_index_of,
)
from mjlab.tasks.registry import list_tasks, load_env_cfg

TASK = (
  "Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Increated-Perturbations-"
  "MeasuredServo"
)
VENDOR_TASK = (
  "Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Increated-Perturbations-"
  "Vendor-Gains"
)
MULTI_MOTION_TASK = "Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion-MeasuredServo"


def _articulation(task_id: str, play: bool = False):
  return load_env_cfg(task_id, play=play).scene.entities["robot"].articulation


def test_measured_servo_task_is_registered() -> None:
  assert TASK in list_tasks()


@pytest.mark.parametrize("play", [False, True])
def test_damping_matches_the_multi_motion_measured_servo_task(play: bool) -> None:
  """The single-clip task must carry the same damping as the teacher's own task."""
  single = [group.damping for group in _articulation(TASK, play=play).actuators]
  multi = [group.damping for group in _articulation(MULTI_MOTION_TASK, play=play).actuators]
  assert single == multi
  assert max(single) == pytest.approx(34.71)


def test_damping_differs_from_the_vendor_task() -> None:
  """Without this the task would be a copy of the vendor one and the tests toothless."""
  measured = [group.damping for group in _articulation(TASK).actuators]
  vendor = [group.damping for group in _articulation(VENDOR_TASK).actuators]
  assert measured != vendor


def test_wrist_effort_event_targets_the_wrist_group() -> None:
  """The measured-servo split moves group 13, so the event must re-resolve by patterns."""
  cfg = load_env_cfg(TASK)
  robot_cfg = cfg.scene.entities["robot"]
  articulation = robot_cfg.articulation
  assert articulation is not None

  event = cfg.events["randomize_effort_limits_wrist"]
  ids = list(event.params["asset_cfg"].actuator_ids)
  expected = group_index_of(robot_cfg, WRIST_GROUP_PATTERNS)
  assert ids == [expected]
  assert tuple(articulation.actuators[ids[0]].target_names_expr) == WRIST_GROUP_PATTERNS
