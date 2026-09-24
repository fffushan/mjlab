"""Tests for the X2 velocity torso-IMU ablation grid.

The shipped X2 velocity task is no-state-estimation: the actor reads the pelvis
``imu_0`` gyro and up-vector, and the critic's privileged velocity/gravity signals
come from the pelvis too. These variants move the *actor* onto the torso
``imu_1`` and enumerate the two frames the *critic* can legitimately see:

  Critic-Pelvis-Root       critic gyro/velocimeter pelvis, gravity root-link
  Critic-Pelvis-Upvector   critic gyro/velocimeter pelvis, gravity torso IMU
  Critic-Torso-Root        critic gyro/velocimeter torso,  gravity root-link
  Critic-Torso-Upvector    critic gyro/velocimeter torso,  gravity torso IMU

The velocity config builds its critic terms as ``{**actor_terms}``, which copies
*references*, so the tests below also pin that the actor and critic never share a
base-velocity term object -- mutating one shared object is exactly how this
two-axis grid would silently collapse into one axis.
"""

import io
import tempfile
import warnings
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path
from typing import cast

import mujoco
import numpy as np
import onnx
import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.scene import Scene
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.config.agibot_x2.env_cfgs import (
  X2VelocityCriticGravitySource,
  X2VelocityImuSource,
  agibot_x2_flat_velocity_env_cfg,
)

PREFIX = "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation"
BASELINE = PREFIX
TASK_PELVIS_ROOT = PREFIX + "-Torso-IMU-Critic-Pelvis-Root"
TASK_PELVIS_UPVECTOR = PREFIX + "-Torso-IMU-Critic-Pelvis-Upvector"
TASK_TORSO_ROOT = PREFIX + "-Torso-IMU-Critic-Torso-Root"
TASK_TORSO_UPVECTOR = PREFIX + "-Torso-IMU-Critic-Torso-Upvector"

# task id -> (experiment slug, critic imu source, critic gravity source)
VARIANTS = {
  TASK_PELVIS_ROOT: ("critic_pelvis_root", "pelvis", "root"),
  TASK_PELVIS_UPVECTOR: ("critic_pelvis_upvector", "pelvis", "upvector"),
  TASK_TORSO_ROOT: ("critic_torso_root", "torso", "root"),
  TASK_TORSO_UPVECTOR: ("critic_torso_upvector", "torso", "upvector"),
}

PELVIS_GYRO = "robot/imu_ang_vel"
PELVIS_VELOCIMETER = "robot/imu_lin_vel"
PELVIS_UPVECTOR = "robot/imu_upvector"
TORSO_GYRO = "robot/imu_1_ang_vel"
TORSO_VELOCIMETER = "robot/imu_1_lin_vel"
TORSO_UPVECTOR = "imu_1_upvector"


@pytest.fixture(autouse=True)
def clear_axis_selection(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.delenv("MJLAB_DR_AXES", raising=False)


@pytest.fixture(scope="module")
def torso_scene_model() -> mujoco.MjModel:
  cfg = agibot_x2_flat_velocity_env_cfg(
    imu_source="torso", critic_imu_source="torso", critic_gravity_source="upvector"
  )
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    return Scene(cfg.scene, device="cpu").compile()


def terms(cfg, group: str) -> dict:
  return cfg.observations[group].terms


def sensor_of(cfg, group: str, term: str) -> str:
  return terms(cfg, group)[term].params["sensor_name"]


##
# Registration and configuration contracts.
##


def test_variants_are_registered_with_their_own_experiment_dirs() -> None:
  registered = set(list_tasks())
  names = set()
  for task_id, (slug, _, _) in VARIANTS.items():
    assert task_id in registered
    expected = f"agibot_x2_velocity_torso_imu_{slug}"
    assert load_rl_cfg(task_id).experiment_name == expected
    names.add(expected)

  assert len(names) == len(VARIANTS)
  assert load_rl_cfg(BASELINE).experiment_name == "agibot_x2_velocity"
  assert "agibot_x2_velocity" not in names


def test_baseline_task_is_unchanged() -> None:
  cfg = load_env_cfg(BASELINE)

  assert sensor_of(cfg, "actor", "base_ang_vel") == PELVIS_GYRO
  assert sensor_of(cfg, "actor", "projected_gravity") == PELVIS_UPVECTOR
  assert sensor_of(cfg, "critic", "base_ang_vel") == PELVIS_GYRO
  assert sensor_of(cfg, "critic", "base_lin_vel") == PELVIS_VELOCIMETER

  # The default is a strict no-op, so the terms keep the base config's identity:
  # the velocity config shares one object between the actor and the critic, and
  # that must still be true when nothing is re-pointed.
  assert terms(cfg, "actor")["base_ang_vel"] is terms(cfg, "critic")["base_ang_vel"]
  assert cfg.observations["critic"].terms["projected_gravity"].func.__name__ == (
    "projected_gravity"
  )
  assert "base_lin_vel" not in terms(cfg, "actor")
  assert not [s for s in cfg.scene.sensors if "imu_1" in s.name]


def test_each_variant_moves_only_the_intended_terms() -> None:
  baseline = load_env_cfg(BASELINE)

  for task_id, (_, critic_imu, critic_gravity) in VARIANTS.items():
    cfg = load_env_cfg(task_id)
    assert cfg.scene.entities == baseline.scene.entities
    assert cfg.rewards == baseline.rewards
    assert cfg.terminations == baseline.terminations
    assert cfg.actions == baseline.actions
    assert cfg.events == baseline.events
    assert cfg.commands == baseline.commands
    assert cfg.sim == baseline.sim
    assert cfg.viewer == baseline.viewer
    assert cfg.decimation == baseline.decimation
    assert cfg.episode_length_s == baseline.episode_length_s

    # The actor is on the torso IMU in every variant.
    assert sensor_of(cfg, "actor", "base_ang_vel") == TORSO_GYRO
    assert sensor_of(cfg, "actor", "projected_gravity") == TORSO_UPVECTOR
    assert "base_lin_vel" not in terms(cfg, "actor")

    # The critic follows its two axes.
    if critic_imu == "torso":
      assert sensor_of(cfg, "critic", "base_ang_vel") == TORSO_GYRO
      assert sensor_of(cfg, "critic", "base_lin_vel") == TORSO_VELOCIMETER
    else:
      assert sensor_of(cfg, "critic", "base_ang_vel") == PELVIS_GYRO
      assert sensor_of(cfg, "critic", "base_lin_vel") == PELVIS_VELOCIMETER

    gravity = terms(cfg, "critic")["projected_gravity"]
    if critic_gravity == "upvector":
      assert gravity.func.__name__ == "projected_gravity_from_sensor"
      assert gravity.params["sensor_name"] == TORSO_UPVECTOR
    else:
      assert gravity.func.__name__ == "projected_gravity"
      assert gravity.params == {}


def test_actor_and_critic_never_share_a_base_velocity_term() -> None:
  """The two critic axes are only real if the term objects are distinct.

  ``{**actor_terms}`` copies references, so a mutation-based implementation would
  give the actor and the critic the same source and make the grid degenerate.
  """
  for task_id, (_, critic_imu, _) in VARIANTS.items():
    cfg = load_env_cfg(task_id)
    actor_term = terms(cfg, "actor")["base_ang_vel"]
    critic_term = terms(cfg, "critic")["base_ang_vel"]
    assert actor_term is not critic_term, task_id
    assert actor_term.params["sensor_name"] == TORSO_GYRO
    assert critic_term.params["sensor_name"] == (
      TORSO_GYRO if critic_imu == "torso" else PELVIS_GYRO
    )


def test_grid_covers_both_critic_axes() -> None:
  observed = set()
  for task_id in VARIANTS:
    cfg = load_env_cfg(task_id)
    observed.add(
      (
        sensor_of(cfg, "actor", "base_ang_vel"),
        sensor_of(cfg, "critic", "base_ang_vel"),
        terms(cfg, "critic")["projected_gravity"].func.__name__,
      )
    )

  assert observed == {
    (TORSO_GYRO, PELVIS_GYRO, "projected_gravity"),
    (TORSO_GYRO, PELVIS_GYRO, "projected_gravity_from_sensor"),
    (TORSO_GYRO, TORSO_GYRO, "projected_gravity"),
    (TORSO_GYRO, TORSO_GYRO, "projected_gravity_from_sensor"),
  }


def test_registered_play_config_keeps_the_variant() -> None:
  cfg = load_env_cfg(TASK_TORSO_UPVECTOR, play=True)

  assert sensor_of(cfg, "actor", "base_ang_vel") == TORSO_GYRO
  assert sensor_of(cfg, "critic", "base_lin_vel") == TORSO_VELOCIMETER
  assert cfg.observations["actor"].enable_corruption is False
  assert "push_robot" not in cfg.events
  assert cfg.curriculum == {}
  assert cfg.episode_length_s == int(1e9)


def test_building_a_variant_does_not_mutate_the_baseline() -> None:
  before = load_env_cfg(BASELINE)
  before_motion = deepcopy(terms(before, "actor")["base_ang_vel"])

  for task_id in VARIANTS:
    _ = load_env_cfg(task_id)
    _ = load_env_cfg(task_id, play=True)

  after = load_env_cfg(BASELINE)
  assert sensor_of(after, "actor", "base_ang_vel") == PELVIS_GYRO
  assert sensor_of(after, "critic", "base_ang_vel") == PELVIS_GYRO
  assert sensor_of(after, "critic", "base_lin_vel") == PELVIS_VELOCIMETER
  assert terms(after, "actor")["base_ang_vel"] == before_motion
  assert "base_ang_vel" in terms(after, "critic")


def test_unknown_imu_source_is_rejected() -> None:
  bad = cast(X2VelocityImuSource, "waist")
  with pytest.raises(ValueError, match="imu_source"):
    agibot_x2_flat_velocity_env_cfg(imu_source=bad)
  with pytest.raises(ValueError, match="critic_imu_source"):
    agibot_x2_flat_velocity_env_cfg(critic_imu_source=bad)


def test_unknown_critic_gravity_source_is_rejected() -> None:
  bad = cast(X2VelocityCriticGravitySource, "pelvis")
  with pytest.raises(ValueError, match="critic_gravity_source"):
    agibot_x2_flat_velocity_env_cfg(critic_gravity_source=bad)


##
# Model-level wiring and kinematics.
##


def test_torso_sensors_target_imu_1_and_the_world_up_vector(
  torso_scene_model: mujoco.MjModel,
) -> None:
  model = torso_scene_model
  imu_1 = _site_id(model, "robot/imu_1")
  torso_body = _body_id(model, "robot/torso_link")
  assert int(model.site_bodyid[imu_1]) == torso_body
  np.testing.assert_allclose(model.site_quat[imu_1], [1.0, 0.0, 0.0, 0.0], atol=1e-9)

  for name in (TORSO_GYRO, TORSO_VELOCIMETER):
    sensor = _sensor_id(model, name)
    assert model.sensor_objtype[sensor] == mujoco.mjtObj.mjOBJ_SITE
    assert int(model.sensor_objid[sensor]) == imu_1

  upvector = _sensor_id(model, TORSO_UPVECTOR)
  assert model.sensor_type[upvector] == mujoco.mjtSensor.mjSENS_FRAMEZAXIS
  assert model.sensor_objtype[upvector] == mujoco.mjtObj.mjOBJ_BODY
  assert int(model.sensor_objid[upvector]) == 0
  assert model.sensor_reftype[upvector] == mujoco.mjtObj.mjOBJ_SITE
  assert int(model.sensor_refid[upvector]) == imu_1

  # The world-referenced framezaxis has no entity to take a prefix from, so it
  # compiles under its bare name; the pelvis sensors are entity-scoped.
  assert (
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "robot/" + TORSO_UPVECTOR)
    == -1
  )
  assert _sensor_id(model, PELVIS_UPVECTOR) >= 0


def test_torso_up_vector_is_the_torso_frame_world_z(
  torso_scene_model: mujoco.MjModel,
) -> None:
  """A neutral pose cannot distinguish the two IMUs, so pose the waist."""
  model = torso_scene_model
  data = mujoco.MjData(model)
  data.qpos[3:7] = [np.cos(0.35), 0.0, np.sin(0.35), 0.0]
  for joint, angle in (("waist_pitch_joint", 0.4), ("waist_yaw_joint", 0.3)):
    joint_id = _joint_id(model, f"robot/{joint}")
    data.qpos[int(model.jnt_qposadr[joint_id])] = angle
  mujoco.mj_forward(model, data)

  torso = _body_id(model, "robot/torso_link")
  value = _sensor_value(model, data, _sensor_id(model, TORSO_UPVECTOR))
  np.testing.assert_allclose(value, data.xmat[torso].reshape(3, 3)[2, :], atol=1e-9)

  pelvis_value = _sensor_value(model, data, _sensor_id(model, PELVIS_UPVECTOR))
  assert np.max(np.abs(value - pelvis_value)) > 1e-2


def test_torso_gyro_includes_the_waist_joint_rates(
  torso_scene_model: mujoco.MjModel,
) -> None:
  """The torso gyro is the torso frame's angular velocity, waist rates included."""
  model = torso_scene_model
  imu_1 = _site_id(model, "robot/imu_1")
  torso = _body_id(model, "robot/torso_link")
  pelvis = _body_id(model, "robot/pelvis")
  torso_gyro_id = _sensor_id(model, TORSO_GYRO)
  pelvis_gyro_id = _sensor_id(model, PELVIS_GYRO)

  data = mujoco.MjData(model)
  data.qpos[0:3] = [0.0, 0.0, 0.8]
  data.qpos[3:7] = [np.cos(0.15), np.sin(0.15), 0.0, 0.0]
  data.qvel[3:6] = [0.2, -0.1, 0.3]
  waist = (
    ("waist_yaw_joint", 0.30, 0.50),
    ("waist_pitch_joint", -0.20, -0.40),
    ("waist_roll_joint", 0.25, 0.60),
  )
  for joint, angle, rate in waist:
    joint_id = _joint_id(model, f"robot/{joint}")
    data.qpos[int(model.jnt_qposadr[joint_id])] = angle
    data.qvel[int(model.jnt_dofadr[joint_id])] = rate
  mujoco.mj_forward(model, data)

  torso_gyro = _sensor_value(model, data, torso_gyro_id)
  pelvis_gyro = _sensor_value(model, data, pelvis_gyro_id)

  # Independent ground truth: MuJoCo's own site velocity.
  site_velocity = np.zeros(6)
  mujoco.mj_objectVelocity(
    model, data, mujoco.mjtObj.mjOBJ_SITE, imu_1, site_velocity, 1
  )
  np.testing.assert_allclose(torso_gyro, site_velocity[0:3], atol=1e-9)

  torso_from_pelvis = data.xmat[torso].reshape(3, 3).T @ data.xmat[pelvis].reshape(3, 3)
  rotated = torso_from_pelvis @ pelvis_gyro
  assert np.max(np.abs(torso_gyro - rotated)) > 1e-2

  # With the waist joints held still the two agree up to that constant rotation.
  for joint, angle, _ in waist:
    joint_id = _joint_id(model, f"robot/{joint}")
    data.qpos[int(model.jnt_qposadr[joint_id])] = angle
    data.qvel[int(model.jnt_dofadr[joint_id])] = 0.0
  mujoco.mj_forward(model, data)
  still_torso = _sensor_value(model, data, torso_gyro_id)
  still_pelvis = _sensor_value(model, data, pelvis_gyro_id)
  np.testing.assert_allclose(still_torso, torso_from_pelvis @ still_pelvis, atol=1e-9)


##
# Runtime integration (slow): real environment and production metadata path.
##


def build_env(task_id: str):
  cfg = load_env_cfg(task_id)
  cfg.scene.num_envs = 2
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
      return ManagerBasedRlEnv(cfg, device="cpu")


@pytest.fixture(scope="module")
def torso_env():
  env = build_env(TASK_TORSO_UPVECTOR)
  try:
    yield env
  finally:
    env.close()


@pytest.mark.slow
def test_actor_observations_read_the_torso_imu(torso_env) -> None:
  env = torso_env
  gyro_term = env.observation_manager.get_term_cfg("actor", "base_ang_vel")
  gravity_term = env.observation_manager.get_term_cfg("actor", "projected_gravity")
  assert gyro_term.params["sensor_name"] == TORSO_GYRO
  assert gravity_term.params["sensor_name"] == TORSO_UPVECTOR

  pelvis_gyro = env.scene[PELVIS_GYRO]
  max_gap = 0.0
  torch.manual_seed(0)
  for _ in range(5):
    env.step(torch.randn(env.num_envs, env.action_manager.total_action_dim))
    value = gyro_term.func(env, **gyro_term.params)
    torch.testing.assert_close(value, env.scene[TORSO_GYRO].data)
    max_gap = max(max_gap, (value - pelvis_gyro.data).abs().max().item())

  assert max_gap > 0.05, (
    "the torso and pelvis gyro readings never differed, so the actor may still "
    f"be wired to the pelvis site (max gap {max_gap:.4f} rad/s)"
  )


@pytest.mark.slow
def test_runtime_observation_shape_matches_the_baseline(torso_env) -> None:
  baseline_env = build_env(BASELINE)
  try:
    obs, _ = torso_env.reset()
    baseline_obs, _ = baseline_env.reset()
    assert obs["actor"].shape == baseline_obs["actor"].shape
    assert obs["critic"].shape == baseline_obs["critic"].shape

    torch.manual_seed(0)
    for _ in range(10):
      action = torch.randn(
        torso_env.num_envs, torso_env.action_manager.total_action_dim
      )
      obs, reward, _terminated, _truncated, _extras = torso_env.step(action)
      assert torch.isfinite(obs["actor"]).all()
      assert torch.isfinite(obs["critic"]).all()
      assert torch.isfinite(reward).all()
  finally:
    baseline_env.close()


@pytest.mark.slow
def test_export_metadata_records_the_torso_imu(torso_env) -> None:
  metadata = get_base_metadata(torso_env, run_path="test/run")
  names = metadata["observation_names"]
  assert isinstance(names, list)
  sites = metadata["observation_terms_sensor_site"]
  bodies = metadata["observation_terms_sensor_body"]
  sensor_names = metadata["observation_terms_sensor_name"]
  assert isinstance(sites, list) and isinstance(bodies, list)
  assert isinstance(sensor_names, list)
  assert all(len(entry) == len(names) for entry in (sites, bodies, sensor_names))

  index = names.index("base_ang_vel")
  assert sensor_names[index] == TORSO_GYRO
  assert sites[index] == "imu_1"
  assert bodies[index] == "torso_link"

  with tempfile.TemporaryDirectory() as tmpdir:
    path = str(Path(tmpdir) / "policy.onnx")
    _write_identity_onnx(path)
    attach_metadata_to_onnx(path, metadata)
    props = {prop.key: prop.value for prop in onnx.load(path).metadata_props}
    assert props["observation_terms_sensor_body"].split(",")[index] == "torso_link"


##
# Helpers.
##


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


def _joint_id(model: mujoco.MjModel, name: str) -> int:
  joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
  assert joint_id >= 0, name
  return int(joint_id)


def _sensor_value(
  model: mujoco.MjModel, data: mujoco.MjData, sensor_id: int
) -> np.ndarray:
  start = int(model.sensor_adr[sensor_id])
  dim = int(model.sensor_dim[sensor_id])
  return data.sensordata[start : start + dim].copy()


def _write_identity_onnx(path: str) -> None:
  inp = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 2])
  out = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 2])
  node = onnx.helper.make_node("Identity", ["input"], ["output"])
  graph = onnx.helper.make_graph([node], "test_graph", [inp], [out])
  onnx.save(onnx.helper.make_model(graph), path)
