"""Tests for x2_constants.py."""

import re

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.agibot_x2 import x2_constants
from mjlab.entity import Entity
from mjlab.utils.string import resolve_expr

FOOT_PATTERN = r"^(left|right)_foot[0-9]+_collision$"


@pytest.fixture(scope="module")
def x2_entity() -> Entity:
  return Entity(x2_constants.get_x2_robot_cfg())


@pytest.fixture(scope="module")
def x2_model(x2_entity: Entity) -> mujoco.MjModel:
  return x2_entity.spec.compile()


# fmt: off
@pytest.mark.parametrize(
  "actuator_config",
  [
    x2_constants.X2_ACTUATOR_HIP_KNEE,
    x2_constants.X2_ACTUATOR_WAIST,
    x2_constants.X2_ACTUATOR_36NM,
    x2_constants.X2_ACTUATOR_24NM,
    x2_constants.X2_ACTUATOR_WRIST,
    x2_constants.X2_ACTUATOR_HEAD_YAW,
    x2_constants.X2_ACTUATOR_HEAD_PITCH,
  ],
)
# fmt: on
def test_actuator_parameters(x2_model, actuator_config):
  matched = 0
  for i in range(x2_model.nu):
    actuator = x2_model.actuator(i)
    if not any(
      re.match(pattern, actuator.name) for pattern in actuator_config.target_names_expr
    ):
      continue
    matched += 1
    stiffness = actuator_config.stiffness
    damping = actuator_config.damping
    assert actuator.gainprm[0] == stiffness
    assert actuator.biasprm[1] == -stiffness
    assert actuator.biasprm[2] == -damping
    assert actuator.forcerange[0] == -actuator_config.effort_limit
    assert actuator.forcerange[1] == actuator_config.effort_limit
    dof = x2_model.jnt_dofadr[actuator.trnid[0]]
    assert x2_model.dof_armature[dof] == actuator_config.armature
  assert matched > 0


def test_every_joint_is_actuated_exactly_once(x2_entity: Entity) -> None:
  """The actuator groups must cover all 31 joints without overlapping."""
  matched = [False] * x2_entity.num_joints
  for config in x2_constants.X2_ARTICULATION.actuators:
    patterns = config.target_names_expr
    for i, name in enumerate(x2_entity.joint_names):
      if any(re.match(pattern, name) for pattern in patterns):
        assert not matched[i], f"joint {name} matched by more than one actuator group"
        matched[i] = True
  unactuated = [n for n, m in zip(x2_entity.joint_names, matched, strict=True) if not m]
  assert all(matched), f"unactuated joints: {unactuated}"


def test_x2_entity_creation(x2_entity: Entity) -> None:
  assert x2_entity.num_actuators == 31
  assert x2_entity.num_joints == 31
  assert x2_entity.is_actuated
  assert not x2_entity.is_fixed_base


def test_total_mass(x2_model: mujoco.MjModel) -> None:
  """The MJCF must carry the URDF inertials, not mesh-density mass.

  The flagship X2 Ultra weighs ~42 kg; the vendor xml only states inertials for
  some bodies and leaves the rest to mesh density, which lands ~3.6% high.
  """
  assert sum(x2_model.body_mass) == pytest.approx(41.967, abs=1e-3)


def test_keyframe_base_position(x2_model) -> None:
  data = mujoco.MjData(x2_model)
  mujoco.mj_resetDataKeyframe(x2_model, data, 0)
  mujoco.mj_forward(x2_model, data)
  np.testing.assert_array_equal(
    data.qpos[:3], x2_constants.STAND_KEYFRAME.pos
  )
  np.testing.assert_array_equal(
    data.qpos[3:7], x2_constants.STAND_KEYFRAME.rot
  )


def test_keyframe_joint_positions(x2_entity, x2_model) -> None:
  """Test that keyframe joint positions match the configuration."""
  key = x2_model.key("init_state")
  expected_joint_pos = x2_constants.STAND_KEYFRAME.joint_pos
  assert expected_joint_pos is not None
  expected_values = resolve_expr(expected_joint_pos, x2_entity.joint_names, 0.0)
  for joint_name, expected_value in zip(
    x2_entity.joint_names, expected_values, strict=True
  ):
    joint = x2_model.joint(joint_name)
    qpos_idx = joint.qposadr[0]
    actual_value = key.qpos[qpos_idx]
    np.testing.assert_allclose(
      actual_value,
      expected_value,
      rtol=1e-5,
      err_msg=f"Joint {joint_name} position mismatch: "
      f"expected {expected_value}, got {actual_value}",
    )


def test_keyframe_is_inside_soft_joint_limits(x2_entity, x2_model) -> None:
  """The stance keyframe must not start episodes at a joint-limit penalty.

  Knee, elbow, hip roll and shoulder roll sit on (or outside) their soft limits
  in the vendored xml's all-zero pose, which is why the keyframe is bent.
  """
  factor = x2_constants.X2_ARTICULATION.soft_joint_pos_limit_factor
  key = x2_model.key("init_state")
  expected_joint_pos = x2_constants.STAND_KEYFRAME.joint_pos
  assert expected_joint_pos is not None
  values = resolve_expr(expected_joint_pos, x2_entity.joint_names, 0.0)
  for joint_name, _value in zip(x2_entity.joint_names, values, strict=True):
    joint = x2_model.joint(joint_name)
    low, high = joint.range
    span = high - low
    soft_low = low + 0.5 * (1.0 - factor) * span
    soft_high = high - 0.5 * (1.0 - factor) * span
    actual = key.qpos[joint.qposadr[0]]
    assert soft_low <= actual <= soft_high, (
      f"{joint_name} = {actual:.4f} outside soft limits "
      f"[{soft_low:.4f}, {soft_high:.4f}]"
    )


def test_keyframe_feet_on_ground(x2_model) -> None:
  """The stance keyframe should place the foot collision spheres at z=0."""
  data = mujoco.MjData(x2_model)
  mujoco.mj_resetDataKeyframe(x2_model, data, 0)
  mujoco.mj_forward(x2_model, data)
  sole_z = [
    data.geom_xpos[i][2] - x2_model.geom_size[i][0]
    for i in range(x2_model.ngeom)
    if re.match(FOOT_PATTERN, x2_model.geom(i).name)
  ]
  assert len(sole_z) == 24
  assert min(sole_z) == pytest.approx(0.0, abs=5e-4)


def test_foot_collision_geoms(x2_model) -> None:
  for i in range(x2_model.ngeom):
    geom = x2_model.geom(i)
    if re.match(FOOT_PATTERN, geom.name):
      assert geom.condim == 3
      assert geom.priority == 1
      assert geom.friction[0] == 0.6


def test_non_foot_collision_geoms(x2_model) -> None:
  excluded = {"pelvis_collision", "head_yaw_link_collision", "head_pitch_link_collision"}
  for i in range(x2_model.ngeom):
    geom = x2_model.geom(i)
    if "_collision" not in geom.name or geom.name in excluded:
      continue
    if not re.match(FOOT_PATTERN, geom.name):
      assert geom.condim == 1


def test_collision_geom_count(x2_model) -> None:
  names = [x2_model.geom(i).name for i in range(x2_model.ngeom)]
  collision_geoms = [n for n in names if "_collision" in n]
  assert len(collision_geoms) == 49
  assert len([n for n in collision_geoms if re.match(FOOT_PATTERN, n)]) == 24


def test_excluded_hulls_do_not_collide(x2_model) -> None:
  """Pelvis and head hulls are kept in the xml but disabled for collision."""
  excluded = {"pelvis_collision", "head_yaw_link_collision", "head_pitch_link_collision"}
  for i in range(x2_model.ngeom):
    geom = x2_model.geom(i)
    if geom.name in excluded:
      assert geom.contype == 0
      assert geom.conaffinity == 0
    elif "_collision" in geom.name:
      assert geom.contype == 1, f"{geom.name} should collide"
      assert geom.conaffinity == 1


def test_imu_sensors(x2_model) -> None:
  """The tracking task reads base velocity from these named sensors."""
  names = {
    mujoco.mj_id2name(x2_model, mujoco.mjtObj.mjOBJ_SENSOR, i)
    for i in range(x2_model.nsensor)
  }
  assert {"imu_ang_vel", "imu_lin_vel", "imu_lin_acc", "root_angmom"} <= names
