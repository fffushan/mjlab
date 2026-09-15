"""Tests for x2_constants.py."""

import re
from pathlib import Path

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
  "group_key",
  list(x2_constants.X2_ACTUATOR_GROUPS),
)
# fmt: on
def test_actuator_parameters(x2_model, group_key):
  actuator_config = x2_constants.X2_ACTUATOR_GROUPS[group_key]
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
    effort_limit = actuator_config.effort_limit
    assert effort_limit is not None
    assert actuator.gainprm[0] == stiffness
    assert actuator.biasprm[1] == -stiffness
    assert actuator.biasprm[2] == -damping
    assert actuator.forcerange[0] == -effort_limit
    assert actuator.forcerange[1] == effort_limit
    dof = x2_model.jnt_dofadr[actuator.trnid[0]]
    assert x2_model.dof_armature[dof] == actuator_config.armature
    assert x2_model.dof_frictionloss[dof] == actuator_config.frictionloss
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


##
# Gain provenance and torque-budget invariants.
#
# The gain table below is the *evidence* the robot config is pinned to, so it is
# written as literals here rather than derived from the config. It is the
# measured native stable-stand profile of the real X2, read off the
# /aima/hal/joint/*/command JointCommandArray messages of two captures that
# reproduce it (20260915T130441Z on firmware v1.1.0.95 and 20260915T141139Z on
# v1.1.4). Every clean mode window in those recordings holds exactly one
# (kp, kd) per joint, so these are not medians over a ramp. See
# docs/source/x2_gain_provenance.md.
# fmt: off
MEASURED_NATIVE_STAND_GAINS: dict[str, tuple[float, float]] = {
  "hip_pitch": (100.0, 4.0),
  "hip_roll": (100.0, 3.0),
  "hip_yaw": (100.0, 3.0),
  "knee": (150.0, 5.0),
  "ankle_pitch": (40.0, 3.0),
  "ankle_roll": (30.0, 2.0),
  "shoulder_pitch": (30.0, 1.0),
  "shoulder_roll": (20.0, 1.0),
  "shoulder_yaw": (20.0, 1.0),
  "elbow": (50.0, 1.0),
  "wrist_yaw": (50.0, 1.0),
  "wrist_pitch_roll": (20.0, 1.0),
  "head_yaw": (3.4, 0.114),
  # 40/8 is the measured stand value, corroborated by the vendor's sample policy
  # (40.1792/2.5579 for this joint) after the PFP-96 armature inference was
  # replaced by the vendor's own. See docs/source/x2_gain_provenance.md.
  "waist_yaw": (40.0, 8.0),
}
# Groups that keep the reflected-inertia template instead: the vendor's own two
# tables disagree too much for a nominal (waist pitch/roll), or the measured gain
# contradicts the joint's effort limit (head pitch).
TEMPLATE_GROUPS = ("waist_pitch_roll", "head_pitch")
# The flagship revision's joint torque ratings, as released in
# X2_URDF-v1.3.0/x2_ultra.xml (actuatorfrcrange): 9x120, 8x24, 6x36, 4x4.8,
# 2x48, plus 2.6 and 0.6 on the head. This model is v1.3.0 by explicit choice,
# and the newer revision is a different joint set (waist 36, wrist 6, ankle pitch
# and shoulders 60, the former 24 group 36), so the two must not be mixed.
VENDOR_V130_EFFORT_LIMITS: dict[str, float] = {
  "hip_pitch": 120.0,
  "hip_roll": 120.0,
  "hip_yaw": 120.0,
  "knee": 120.0,
  "waist_yaw": 120.0,
  "waist_pitch_roll": 48.0,
  "ankle_pitch": 36.0,
  "ankle_roll": 24.0,
  "shoulder_pitch": 36.0,
  "shoulder_roll": 36.0,
  "shoulder_yaw": 24.0,
  "elbow": 24.0,
  "wrist_yaw": 24.0,
  "wrist_pitch_roll": 4.8,
  "head_yaw": 2.6,
  "head_pitch": 0.6,
}
# fmt: on


def test_effort_limits_are_the_flagship_revisions() -> None:
  for key, limit in VENDOR_V130_EFFORT_LIMITS.items():
    assert x2_constants.X2_ACTUATOR_GROUPS[key].effort_limit == limit, key


def test_measured_stand_gains_are_what_the_robot_is_configured_with() -> None:
  for key, (kp, kd) in MEASURED_NATIVE_STAND_GAINS.items():
    group = x2_constants.X2_ACTUATOR_GROUPS[key]
    assert (group.stiffness, group.damping) == (kp, kd), key
  assert set(TEMPLATE_GROUPS).isdisjoint(MEASURED_NATIVE_STAND_GAINS)
  assert set(MEASURED_NATIVE_STAND_GAINS) | set(TEMPLATE_GROUPS) == set(
    x2_constants.X2_ACTUATOR_GROUPS
  )


@pytest.mark.parametrize("group_key", TEMPLATE_GROUPS)
def test_template_gains_follow_the_reflected_inertia_template(group_key) -> None:
  """Kp = I * (20*pi)^2 and Kd = 4 * I * (20*pi) for the template groups.

  Pins the convention the template came from, so an edit here cannot silently
  stop the group from being a 10 Hz / zeta = 2 closed loop.
  """
  group = x2_constants.X2_ACTUATOR_GROUPS[group_key]
  armature = group.armature
  assert armature is not None
  omega = x2_constants.NATURAL_FREQ
  assert group.stiffness == pytest.approx(armature * omega**2, rel=1e-4)
  assert group.damping == pytest.approx(
    2.0 * x2_constants.DAMPING_RATIO * armature * omega, rel=1e-4
  )


def _scale_of(group_key: str) -> float:
  """The action scale the config gives a group's joints."""
  pattern = x2_constants.X2_ACTUATOR_GROUPS[group_key].target_names_expr[0]
  return x2_constants.X2_ACTION_SCALE[pattern]


def test_action_scale_keeps_a_unit_action_inside_the_torque_budget() -> None:
  """The torque floor of the action scale: 0.25 * effort / Kp.

  Below it a full-unit action would command more than the joint can deliver,
  which is why the action scale is not simply 0.25 * effort / Kp any more: the
  tracking floor may raise it, never lower it.
  """
  for key, group in x2_constants.X2_ACTUATOR_GROUPS.items():
    effort = group.effort_limit
    assert effort is not None
    assert _scale_of(key) >= 0.25 * effort / group.stiffness - 1e-9, key


def test_frictionloss_stays_a_small_fraction_of_the_torque_budget(x2_model) -> None:
  """No joint may carry dry friction anywhere near its effort limit.

  The vendored xml sets a blanket 0.3 N.m for all 31 joints, which is 25x
  head_pitch's 2% budget and 50% of its 0.6 N.m effort limit; combined with the
  friction and effort-limit randomization that can lock a joint outright.
  """
  for i in range(x2_model.nu):
    actuator = x2_model.actuator(i)
    dof = x2_model.jnt_dofadr[actuator.trnid[0]]
    effort = float(actuator.forcerange[1])
    friction = float(x2_model.dof_frictionloss[dof])
    assert friction <= 0.02 * effort, (
      f"{actuator.name}: friction {friction} is more than 2% of {effort}"
    )


# The tracking floor of the action scale, and the p95 reference deviation each was
# measured from. Literals for the same reason as the gains above: the motion file
# they come from is gitignored, so the exact re-derivation is skipped where it is
# absent and this table is what pins the numbers on every checkout.
# fmt: off
TRACKING_ACTION_SCALE_FLOOR: dict[str, tuple[float, float]] = {
  "knee": (0.261, 1.301),
  "shoulder_pitch": (0.326, 1.629),
  "shoulder_yaw": (0.345, 1.721),
  "elbow": (0.241, 1.200),
  "wrist_yaw": (0.156, 0.777),
  "wrist_pitch_roll": (0.145, 0.724),
}
# fmt: on


def test_tracking_action_scale_floors_are_the_documented_ones() -> None:
  """Each tracking floor equals the reference deviation it was measured from / 5.

  Runs without the motion file, so the rule keeps a guard on a clean checkout.
  """
  for key, (scale, deviation) in TRACKING_ACTION_SCALE_FLOOR.items():
    assert _scale_of(key) == pytest.approx(scale)
    assert scale == pytest.approx(deviation / 5.0, abs=2e-3), key
    # The floor only exists to raise the scale, never to lower it.
    group = x2_constants.X2_ACTUATOR_GROUPS[key]
    effort = group.effort_limit
    assert effort is not None
    assert scale >= 0.25 * effort / group.stiffness, key


def _reference_motion() -> np.ndarray:
  path = Path("data/qianghuo_smplx_agibot_x2_tracking.npz")
  if not path.exists():
    pytest.skip(f"{path} not present (generated motion data)")
  joint_pos = np.load(path)["joint_pos"]
  assert joint_pos.shape[1] == 31
  return joint_pos


def test_action_scale_covers_the_reference_motion(x2_entity) -> None:
  """Every group's action scale reaches its worst reference deviation / 5.

  This is the tracking floor of the action-scale rule. The floor is a property of
  the *group*, so it has to cover the worst joint in it, and it only has a
  meaning while the keyframe is the pose the action is an offset from, so the
  deviation is re-derived from the current keyframe.
  """
  joint_pos = _reference_motion()
  keyframe = x2_constants.STAND_KEYFRAME.joint_pos
  assert keyframe is not None
  default = resolve_expr(keyframe, x2_entity.joint_names, 0.0)

  worst: dict[str, float] = {}
  for index, (joint_name, default_pos) in enumerate(
    zip(x2_entity.joint_names, default, strict=True)
  ):
    deviation = float(np.percentile(np.abs(joint_pos[:, index] - default_pos), 95))
    group_key = next(
      key
      for key, group in x2_constants.X2_ACTUATOR_GROUPS.items()
      if any(re.fullmatch(p, joint_name) for p in group.target_names_expr)
    )
    worst[group_key] = max(worst.get(group_key, 0.0), deviation)

  for key, deviation in worst.items():
    assert _scale_of(key) >= deviation / 5.0 - 1e-9, (
      f"{key}: action scale {_scale_of(key)} cannot cover a {deviation:.3f} rad "
      f"reference deviation inside 5 action units"
    )
    # Where the tracking floor is the one in force, it has to be current: a
    # keyframe change that shifts the deviation must not leave a stale floor.
    if key in TRACKING_ACTION_SCALE_FLOOR:
      recorded = TRACKING_ACTION_SCALE_FLOOR[key][1]
      assert recorded == pytest.approx(deviation, abs=2e-3), (
        f"{key}: tracking floor was measured at {recorded} rad, the keyframe "
        f"now gives {deviation:.3f} rad"
      )
