"""Tests for the g1_29dof_mode_15 robot constants."""

import re

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.unitree_g1 import g1_constants
from mjlab.entity import Entity


@pytest.fixture(scope="module")
def mode15_entity() -> Entity:
  return Entity(g1_constants.get_g1_29dof_mode_15_robot_cfg())


@pytest.fixture(scope="module")
def mode15_model(mode15_entity: Entity) -> mujoco.MjModel:
  return mode15_entity.spec.compile()


def test_mode15_entity_structure(mode15_entity) -> None:
  assert mode15_entity.num_actuators == 29
  assert mode15_entity.num_joints == 29
  assert mode15_entity.is_actuated
  assert not mode15_entity.is_fixed_base


def test_mode15_hip_pitch_uses_7520_22(mode15_model) -> None:
  """Mode-15 hip pitch/roll use the 7520-22 actuator (139 Nm @ 20 rad/s)."""
  for side in ("left", "right"):
    for joint in ("hip_pitch", "hip_roll"):
      name = f"{side}_{joint}_joint"
      ai = mujoco.mj_name2id(mode15_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
      assert ai >= 0, name
      assert mode15_model.actuator_forcerange[ai][1] == pytest.approx(
        g1_constants.ACTUATOR_7520_22.effort_limit
      ), name
      assert mode15_model.actuator_gainprm[ai][0] == pytest.approx(
        g1_constants.STIFFNESS_7520_22
      ), name


def test_mode15_hip_yaw_keeps_7520_14(mode15_model) -> None:
  """Mode-15 hip yaw keeps the 7520-14 actuator (88 Nm @ 32 rad/s)."""
  for side in ("left", "right"):
    name = f"{side}_hip_yaw_joint"
    ai = mujoco.mj_name2id(mode15_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    assert ai >= 0, name
    assert mode15_model.actuator_forcerange[ai][1] == pytest.approx(
      g1_constants.ACTUATOR_7520_14.effort_limit
    ), name
    assert mode15_model.actuator_gainprm[ai][0] == pytest.approx(
      g1_constants.STIFFNESS_7520_14
    ), name


def test_mode15_wrist_pitch_mass(mode15_model) -> None:
  """Mode-15 wrist pitch links are heavier (0.684 kg vs 0.48405 kg base)."""
  for side in ("left", "right"):
    name = f"{side}_wrist_pitch_link"
    bid = mujoco.mj_name2id(mode15_model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert bid >= 0, name
    assert mode15_model.body_mass[bid] == pytest.approx(0.684, abs=1e-6), name


def test_mode15_hip_effort_differs_from_base() -> None:
  """Hip pitch/roll use the 7520-22 rating (139 Nm), not the base 7520-14 (88 Nm)."""
  assert g1_constants.ACTUATOR_7520_22.effort_limit != pytest.approx(
    g1_constants.ACTUATOR_7520_14.effort_limit
  )
  for name in (".*_hip_pitch_joint", ".*_hip_roll_joint"):
    assert name in g1_constants.G1_29DOF_MODE_15_ACTION_SCALE
    assert (
      name in g1_constants.G1_29DOF_MODE_15_ARTICULATION.actuators[2].target_names_expr
    )
  # Knee is 7520-22 in both variants -> same action scale.
  assert g1_constants.G1_29DOF_MODE_15_ACTION_SCALE[".*_knee_joint"] == pytest.approx(
    g1_constants.G1_ACTION_SCALE[".*_knee_joint"]
  )


def test_mode15_total_mass(mode15_model) -> None:
  """Mode-15 total mass must match the URDF sum (33.74 kg) up to merged extras."""
  total = float(np.sum(mode15_model.body_mass))
  # mjlab merges the head/logo/waist_support into the torso (+0.001 kg).
  assert total == pytest.approx(33.7400429, abs=0.002)


# fmt: off
@pytest.mark.parametrize(
  "actuator_config,stiffness,damping",
  [
    (g1_constants.G1_ACTUATOR_5020, g1_constants.STIFFNESS_5020, g1_constants.DAMPING_5020),
    (g1_constants.G1_29DOF_MODE_15_ACTUATOR_7520_14, g1_constants.STIFFNESS_7520_14, g1_constants.DAMPING_7520_14),
    (g1_constants.G1_29DOF_MODE_15_ACTUATOR_7520_22, g1_constants.STIFFNESS_7520_22, g1_constants.DAMPING_7520_22),
    (g1_constants.G1_ACTUATOR_4010, g1_constants.STIFFNESS_4010, g1_constants.DAMPING_4010),
    (g1_constants.G1_ACTUATOR_WAIST, g1_constants.STIFFNESS_5020 * 2, g1_constants.DAMPING_5020 * 2),
    (g1_constants.G1_ACTUATOR_ANKLE, g1_constants.STIFFNESS_5020 * 2, g1_constants.DAMPING_5020 * 2),
  ],
)
# fmt: on
def test_mode15_actuator_parameters(mode15_model, actuator_config, stiffness, damping):
  """Each mode-15 actuator class must match its gain/bias/effort config."""
  for i in range(mode15_model.nu):
    actuator = mode15_model.actuator(i)
    actuator_name = actuator.name
    matches = any(
      re.match(pattern, actuator_name) for pattern in actuator_config.target_names_expr
    )
    if matches:
      assert actuator.gainprm[0] == stiffness
      assert actuator.biasprm[1] == -stiffness
      assert actuator.biasprm[2] == -damping
      assert actuator.forcerange[0] == -actuator_config.effort_limit
      assert actuator.forcerange[1] == actuator_config.effort_limit
