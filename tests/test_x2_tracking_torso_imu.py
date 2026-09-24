"""Tests for the X2 torso-anchor + torso-IMU tracking ablation.

The shipped X2 tracking contract tracks the reference in the ``torso_link``
frame but reads its base velocity from the ``imu_0`` site on the pelvis. This
ablation keeps the anchor on ``torso_link`` and moves the base angular/linear
velocity observations onto the ``imu_1`` site, so a single change isolates the
IMU source. These tests pin the config wiring, the numerical sensor frame, the
privileged critic routing, and the exported metadata.
"""

import io
import tempfile
import warnings
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import cast

import mujoco
import numpy as np
import onnx
import pytest
import torch

from mjlab.asset_zoo.robots.agibot_x2 import x2_constants
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
  resolve_site_sensor_frame,
)
from mjlab.scene import Scene
from mjlab.sensor import BuiltinSensorCfg, ObjRef
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.config.agibot_x2.env_cfgs import (
  X2TrackingImuSource,
  agibot_x2_flat_tracking_correlated_dr_env_cfg,
)
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg

PREFIX = "Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR-"
REDUCED = PREFIX + "Reduced-Perturbations"
TORSO_IMU = REDUCED + "-Torso-IMU"
MOTION_FILE = "data/qianghuo_smplx_agibot_x2_tracking.npz"


@pytest.fixture(autouse=True)
def clear_axis_selection(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.delenv("MJLAB_DR_AXES", raising=False)


@pytest.fixture(scope="module")
def x2_entity() -> Entity:
  return Entity(x2_constants.get_x2_robot_cfg())


@pytest.fixture(scope="module")
def x2_model(x2_entity: Entity) -> mujoco.MjModel:
  return x2_entity.spec.compile()


@pytest.fixture(scope="module")
def torso_scene_model() -> mujoco.MjModel:
  cfg = agibot_x2_flat_tracking_correlated_dr_env_cfg(
    reduced_perturbations=True, imu_source="torso"
  )
  return Scene(cfg.scene, device="cpu").compile()


def motion_cfg(cfg) -> MotionCommandCfg:
  motion = cfg.commands["motion"]
  assert isinstance(motion, MotionCommandCfg)
  return motion


##
# Configuration contracts.
##


def test_task_is_registered_with_its_own_experiment_directory() -> None:
  assert TORSO_IMU in list_tasks()
  assert load_rl_cfg(TORSO_IMU).experiment_name == (
    "agibot_x2_tracking_correlated_dr_reduced_perturbations_torso_imu"
  )
  assert load_rl_cfg(TORSO_IMU).experiment_name != load_rl_cfg(REDUCED).experiment_name


def test_task_changes_only_the_imu_source() -> None:
  parent = agibot_x2_flat_tracking_correlated_dr_env_cfg(reduced_perturbations=True)
  torso = agibot_x2_flat_tracking_correlated_dr_env_cfg(
    reduced_perturbations=True, imu_source="torso"
  )

  # The anchor is torso_link in both: this ablation does not touch it.
  assert motion_cfg(parent).anchor_body_name == "torso_link"
  assert motion_cfg(torso).anchor_body_name == "torso_link"
  assert motion_cfg(torso) == motion_cfg(parent)

  # Only the base velocity sensor source moves.
  actor_parent = parent.observations["actor"].terms
  actor_torso = torso.observations["actor"].terms
  assert list(actor_torso) == list(actor_parent)
  for name, term in actor_parent.items():
    if name == "base_ang_vel":
      assert (
        replace(term, params={"sensor_name": "robot/imu_1_ang_vel"})
        == (actor_torso[name])
      )
    else:
      assert actor_torso[name] == term

  critic_parent = parent.observations["critic"].terms
  critic_torso = torso.observations["critic"].terms
  assert list(critic_torso) == list(critic_parent)
  for name, term in critic_parent.items():
    if name == "base_ang_vel":
      assert (
        replace(term, params={"sensor_name": "robot/imu_1_ang_vel"})
        == (critic_torso[name])
      )
    elif name == "base_lin_vel":
      assert (
        replace(term, params={"sensor_name": "robot/imu_1_lin_vel"})
        == (critic_torso[name])
      )
    else:
      assert critic_torso[name] == term

  assert torso.rewards == parent.rewards
  assert torso.terminations == parent.terminations
  assert torso.actions == parent.actions
  assert torso.events == parent.events
  assert torso.sim == parent.sim
  assert torso.scene.entities == parent.scene.entities

  # The scene keeps self_collision and adds exactly the two torso sensors.
  assert len(parent.scene.sensors) == 1
  assert parent.scene.sensors[0].name == "self_collision"
  assert len(torso.scene.sensors) == 3
  assert torso.scene.sensors[0] == parent.scene.sensors[0]
  gyro, velocimeter = torso.scene.sensors[1:]
  assert isinstance(gyro, BuiltinSensorCfg)
  assert isinstance(velocimeter, BuiltinSensorCfg)
  assert (gyro.sensor_type, velocimeter.sensor_type) == ("gyro", "velocimeter")
  for sensor in (gyro, velocimeter):
    assert sensor.obj == ObjRef(type="site", name="imu_1", entity="robot")
    assert sensor.prefixed_name.startswith("robot/imu_1_")


def test_registered_play_config_keeps_the_torso_imu_and_play_overrides() -> None:
  cfg = load_env_cfg(TORSO_IMU, play=True)
  motion = motion_cfg(cfg)

  assert motion.anchor_body_name == "torso_link"
  assert cfg.observations["actor"].terms["base_ang_vel"].params["sensor_name"] == (
    "robot/imu_1_ang_vel"
  )
  assert cfg.observations["critic"].terms["base_lin_vel"].params["sensor_name"] == (
    "robot/imu_1_lin_vel"
  )
  assert [s.prefixed_name for s in cfg.scene.sensors] == [
    "self_collision",
    "robot/imu_1_ang_vel",
    "robot/imu_1_lin_vel",
  ]
  assert cfg.observations["actor"].enable_corruption is False
  assert "push_robot" not in cfg.events
  assert motion.sampling_mode == "start"
  assert motion.pose_range == {}
  assert motion.velocity_range == {}
  assert motion.joint_position_range == (-0.05, 0.05)


def test_default_and_existing_tasks_keep_the_pelvis_imu() -> None:
  default = agibot_x2_flat_tracking_correlated_dr_env_cfg(reduced_perturbations=True)
  explicit = agibot_x2_flat_tracking_correlated_dr_env_cfg(
    reduced_perturbations=True, imu_source="pelvis"
  )
  assert default.observations == explicit.observations
  assert default.scene.sensors == explicit.scene.sensors
  assert len(default.scene.sensors) == 1
  assert (
    default.observations["actor"].terms["base_ang_vel"].params["sensor_name"]
    == "robot/imu_ang_vel"
  )
  assert (
    default.observations["critic"].terms["base_lin_vel"].params["sensor_name"]
    == "robot/imu_lin_vel"
  )

  registered = load_env_cfg(REDUCED)
  assert (
    registered.observations["actor"].terms["base_ang_vel"].params["sensor_name"]
    == "robot/imu_ang_vel"
  )
  assert len(registered.scene.sensors) == 1


def test_observation_settings_are_retained() -> None:
  parent = agibot_x2_flat_tracking_correlated_dr_env_cfg(reduced_perturbations=True)
  torso = agibot_x2_flat_tracking_correlated_dr_env_cfg(
    reduced_perturbations=True, imu_source="torso"
  )
  for group in ("actor", "critic"):
    terms = parent.observations[group].terms
    child_terms = torso.observations[group].terms
    assert list(child_terms) == list(terms)
    for name, term in terms.items():
      child = child_terms[name]
      assert child.noise == term.noise
      assert child.scale == term.scale
      assert child.clip == term.clip
      assert (child.delay_min_lag, child.delay_max_lag) == (
        term.delay_min_lag,
        term.delay_max_lag,
      )
      assert child.delay_group == term.delay_group
      assert child.delay_hold_prob == term.delay_hold_prob
      assert child.delay_update_period == term.delay_update_period
      assert child.history_length == term.history_length


def test_building_the_torso_variant_does_not_mutate_the_parent() -> None:
  before = agibot_x2_flat_tracking_correlated_dr_env_cfg(reduced_perturbations=True)
  motion_before = deepcopy(motion_cfg(before))

  _ = agibot_x2_flat_tracking_correlated_dr_env_cfg(
    reduced_perturbations=True, imu_source="torso"
  )
  _ = load_env_cfg(TORSO_IMU)
  _ = load_env_cfg(TORSO_IMU, play=True)

  after = agibot_x2_flat_tracking_correlated_dr_env_cfg(reduced_perturbations=True)
  assert motion_cfg(after) == motion_before
  assert len(after.scene.sensors) == 1
  assert (
    after.observations["actor"].terms["base_ang_vel"].params["sensor_name"]
    == "robot/imu_ang_vel"
  )


def test_unknown_imu_source_is_rejected_at_config_time() -> None:
  with pytest.raises(ValueError, match="imu_source"):
    agibot_x2_flat_tracking_correlated_dr_env_cfg(
      imu_source=cast(X2TrackingImuSource, "waist")
    )


##
# Model-level wiring and kinematics.
##


def test_root_and_tracked_body_order_are_preserved(
  x2_model: mujoco.MjModel,
) -> None:
  cfg = agibot_x2_flat_tracking_correlated_dr_env_cfg(
    reduced_perturbations=True, imu_source="torso"
  )
  motion = motion_cfg(cfg)

  assert motion.body_names[0] == "pelvis"
  assert motion.body_names.index("torso_link") > motion.body_names.index("pelvis")
  assert motion.anchor_body_name == "torso_link"

  free_joints = [
    j
    for j in range(x2_model.njnt)
    if x2_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
  ]
  assert len(free_joints) == 1
  root_body = int(x2_model.jnt_bodyid[free_joints[0]])
  assert mujoco.mj_id2name(x2_model, mujoco.mjtObj.mjOBJ_BODY, root_body) == "pelvis"


def test_torso_imu_sensors_target_the_imu_1_site_on_torso_link(
  torso_scene_model: mujoco.MjModel,
) -> None:
  model = torso_scene_model
  gyro = _sensor_id(model, "robot/imu_1_ang_vel")
  velocimeter = _sensor_id(model, "robot/imu_1_lin_vel")
  pelvis_gyro = _sensor_id(model, "robot/imu_ang_vel")
  imu_1 = _site_id(model, "robot/imu_1")
  imu_0 = _site_id(model, "robot/imu_0")
  torso_body = _body_id(model, "robot/torso_link")
  pelvis_body = _body_id(model, "robot/pelvis")

  for sensor in (gyro, velocimeter):
    assert model.sensor_objtype[sensor] == mujoco.mjtObj.mjOBJ_SITE
    assert int(model.sensor_objid[sensor]) == imu_1
  assert int(model.sensor_objid[pelvis_gyro]) == imu_0

  assert int(model.site_bodyid[imu_1]) == torso_body
  assert int(model.site_bodyid[imu_0]) == pelvis_body
  # The site carries no rotation of its own, so the gyro/velocimeter frame is
  # the torso_link body frame rather than a rotated or offset frame.
  np.testing.assert_allclose(model.site_quat[imu_1], [1.0, 0.0, 0.0, 0.0], atol=1e-9)


def test_torso_gyro_matches_independent_kinematics_and_differs_from_pelvis(
  torso_scene_model: mujoco.MjModel,
) -> None:
  """Torso gyro is the torso frame's angular velocity, waist rates included.

  A static neutral pose cannot tell the two sensors apart: at rest both read
  zero. The robot is posed with a nonzero waist bend and moved with nonzero
  waist joint rates, so the torso reading can only match if the sensor is on the
  torso and the frame includes the waist joints.
  """
  model = torso_scene_model
  imu_1 = _site_id(model, "robot/imu_1")
  torso_body = _body_id(model, "robot/torso_link")
  pelvis_body = _body_id(model, "robot/pelvis")
  torso_gyro_id = _sensor_id(model, "robot/imu_1_ang_vel")
  pelvis_gyro_id = _sensor_id(model, "robot/imu_ang_vel")
  torso_vel_id = _sensor_id(model, "robot/imu_1_lin_vel")

  waist = ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint")
  angles = (0.30, -0.20, 0.25)
  rates = (0.50, -0.40, 0.60)

  data = mujoco.MjData(model)
  _pose_waist(model, data, waist, angles, rates)

  torso_gyro = _sensor_value(model, data, torso_gyro_id)
  pelvis_gyro = _sensor_value(model, data, pelvis_gyro_id)
  torso_vel = _sensor_value(model, data, torso_vel_id)

  # Independent kinematics: MuJoCo's object velocity in the site frame (the
  # result is angular-then-linear).
  site_vel = np.zeros(6)
  mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, imu_1, site_vel, 1)
  np.testing.assert_allclose(torso_gyro, site_vel[0:3], atol=1e-9)
  np.testing.assert_allclose(torso_vel, site_vel[3:6], atol=1e-9)

  # Rotating the pelvis angular velocity into the torso frame does NOT reproduce
  # the torso reading while the waist joints are moving.
  torso_from_pelvis = data.xmat[torso_body].reshape(3, 3).T @ data.xmat[
    pelvis_body
  ].reshape(3, 3)
  rotated_pelvis = torso_from_pelvis @ pelvis_gyro
  assert np.max(np.abs(torso_gyro - rotated_pelvis)) > 1e-2

  # With the waist joints held still, the two frames agree up to the constant
  # waist orientation: the extra term is exactly the joint-rate contribution.
  _pose_waist(model, data, waist, angles, rates=(0.0, 0.0, 0.0))
  still_torso = _sensor_value(model, data, torso_gyro_id)
  still_pelvis = _sensor_value(model, data, pelvis_gyro_id)
  still_rotated = torso_from_pelvis @ still_pelvis
  np.testing.assert_allclose(still_torso, still_rotated, atol=1e-9)


def test_resolve_site_sensor_frame_reads_the_compiled_model(
  torso_scene_model: mujoco.MjModel,
) -> None:
  model = torso_scene_model
  assert resolve_site_sensor_frame(model, "robot/imu_1_ang_vel") == (
    "robot/imu_1",
    "robot/torso_link",
  )
  assert resolve_site_sensor_frame(model, "robot/imu_ang_vel") == (
    "robot/imu_0",
    "robot/pelvis",
  )
  # A non-site sensor (subtree angular momentum) and a missing name resolve to
  # no frame rather than a guessed one.
  assert resolve_site_sensor_frame(model, "robot/root_angmom") == ("", "")
  assert resolve_site_sensor_frame(model, "robot/nope") == ("", "")


##
# Runtime integration (slow): real environment and production metadata path.
##


@pytest.fixture(scope="module")
def torso_env():
  motion_path = Path(MOTION_FILE)
  if not motion_path.exists():
    pytest.skip(f"{motion_path} not present (generated motion data)")
  cfg = load_env_cfg(TORSO_IMU)
  cfg.scene.num_envs = 2
  motion = motion_cfg(cfg)
  motion.motion_file = str(motion_path)
  motion.sampling_mode = "start"
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
      env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    yield env
  finally:
    env.close()


@pytest.mark.slow
def test_actor_base_ang_vel_reads_the_torso_gyro(torso_env) -> None:
  env = torso_env
  term = env.observation_manager.get_term_cfg("actor", "base_ang_vel")
  assert term.params["sensor_name"] == "robot/imu_1_ang_vel"

  value = term.func(env, **term.params)
  torso_data = env.scene["robot/imu_1_ang_vel"].data
  torch.testing.assert_close(value, torso_data)

  # Stepping exercises the live sensor buffer, not just the config object, and
  # the torso reading must not be a relabelled pelvis reading: the trajectory is
  # a real one, so the waist articulates and the two gyros must differ.
  torch.manual_seed(0)
  action = torch.randn(env.num_envs, env.action_manager.total_action_dim)
  pelvis_sensor = env.scene["robot/imu_ang_vel"]
  max_gap = 0.0
  for _ in range(5):
    env.step(action)
    value = term.func(env, **term.params)
    torch.testing.assert_close(value, env.scene["robot/imu_1_ang_vel"].data)
    gap = (value - pelvis_sensor.data).abs().max().item()
    max_gap = max(max_gap, gap)
  assert max_gap > 0.05, (
    "the torso and pelvis gyro readings never differed, so the torso IMU may be "
    f"wired to the pelvis site (max gap {max_gap:.4f} rad/s)"
  )


@pytest.mark.slow
def test_torso_anchor_orientation_is_the_torso_link_frame(torso_env) -> None:
  env = torso_env
  cmd = env.command_manager.get_term("motion")
  assert isinstance(cmd, MotionCommand)
  robot = env.scene["robot"]

  torso_index = robot.body_names.index("torso_link")
  pelvis_index = robot.body_names.index("pelvis")
  assert cmd.robot_anchor_body_index == torso_index
  assert cmd.robot_anchor_body_index != pelvis_index
  assert cmd.motion_anchor_body_index == motion_cfg(env.cfg).body_names.index(
    "torso_link"
  )
  torch.testing.assert_close(
    cmd.robot_anchor_quat_w, robot.data.body_link_quat_w[:, torso_index]
  )


@pytest.mark.slow
def test_env_steps_with_finite_observations_and_expected_dimensions(
  torso_env,
) -> None:
  env = torso_env
  obs, _ = env.reset()
  assert obs["actor"].shape == (env.num_envs, 164)
  assert obs["critic"].shape == (env.num_envs, 350)

  torch.manual_seed(0)
  for _ in range(20):
    action = torch.randn(env.num_envs, env.action_manager.total_action_dim)
    obs, reward, _terminated, _truncated, _extras = env.step(action)
    assert torch.isfinite(obs["actor"]).all()
    assert torch.isfinite(reward).all()


@pytest.mark.slow
def test_export_metadata_records_the_torso_imu_frame_and_serializes(torso_env) -> None:
  metadata = get_base_metadata(torso_env, run_path="test/run")
  names = _metadata_list(metadata, "observation_names")
  sensor_names = _metadata_list(metadata, "observation_terms_sensor_name")
  sites = _metadata_list(metadata, "observation_terms_sensor_site")
  bodies = _metadata_list(metadata, "observation_terms_sensor_body")
  assert all(len(entries) == len(names) for entries in (sensor_names, sites, bodies))

  index = names.index("base_ang_vel")
  assert sensor_names[index] == "robot/imu_1_ang_vel"
  assert sites[index] == "imu_1"
  assert bodies[index] == "torso_link"

  # Terms that do not read a builtin sensor carry no frame.
  anchor_index = names.index("motion_anchor_ori_b")
  assert sites[anchor_index] == ""
  assert bodies[anchor_index] == ""

  with tempfile.TemporaryDirectory() as tmpdir:
    onnx_path = str(Path(tmpdir) / "policy.onnx")
    _write_identity_onnx(onnx_path)
    attach_metadata_to_onnx(onnx_path, metadata)

    props = {prop.key: prop.value for prop in onnx.load(onnx_path).metadata_props}
    assert props["observation_terms_sensor_site"].split(",")[index] == "imu_1"
    assert props["observation_terms_sensor_body"].split(",")[index] == "torso_link"


##
# Helpers.
##


def _metadata_list(metadata: dict, key: str) -> list:
  value = metadata[key]
  assert isinstance(value, list), key
  return value


def _sensor_id(model: mujoco.MjModel, name: str) -> int:
  sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
  assert sensor_id >= 0, name
  return int(sensor_id)


def _site_id(model: mujoco.MjModel, name: str) -> int:
  site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
  assert site_id >= 0, name
  return int(site_id)


def _body_id(model: mujoco.MjModel, name: str) -> int:
  body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
  assert body_id >= 0, name
  return int(body_id)


def _sensor_value(
  model: mujoco.MjModel, data: mujoco.MjData, sensor_id: int
) -> np.ndarray:
  start = int(model.sensor_adr[sensor_id])
  dim = int(model.sensor_dim[sensor_id])
  return data.sensordata[start : start + dim].copy()


def _pose_waist(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  joint_names: tuple[str, ...],
  angles: tuple[float, ...],
  rates: tuple[float, ...],
) -> None:
  """Pose the floating base and waist, then refresh derived quantities."""
  data.qpos[:] = 0.0
  data.qpos[0:3] = [0.0, 0.0, 0.8]
  data.qpos[3:7] = [np.cos(0.15), np.sin(0.15), 0.0, 0.0]
  data.qvel[:] = 0.0
  data.qvel[3:6] = [0.2, -0.1, 0.3]
  for name, angle, rate in zip(joint_names, angles, rates, strict=True):
    joint_id = _joint_id(model, f"robot/{name}")
    data.qpos[int(model.jnt_qposadr[joint_id])] = angle
    data.qvel[int(model.jnt_dofadr[joint_id])] = rate
  mujoco.mj_forward(model, data)


def _joint_id(model: mujoco.MjModel, name: str) -> int:
  joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
  assert joint_id >= 0, name
  return int(joint_id)


def _write_identity_onnx(path: str) -> None:
  input_tensor = onnx.helper.make_tensor_value_info(
    "input", onnx.TensorProto.FLOAT, [1, 2]
  )
  output_tensor = onnx.helper.make_tensor_value_info(
    "output", onnx.TensorProto.FLOAT, [1, 2]
  )
  node = onnx.helper.make_node("Identity", ["input"], ["output"])
  graph = onnx.helper.make_graph([node], "test_graph", [input_tensor], [output_tensor])
  onnx.save(onnx.helper.make_model(graph), path)
