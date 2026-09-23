"""Tests for the X2 pelvis-anchor tracking ablation.

The X2 simulated root is the pelvis, while the shipped tracking contract tracks
the reference in the torso frame. These tests pin both halves of that split: the
task differs from its reduced-perturbation parent *only* in
``commands["motion"].anchor_body_name``, and ``pelvis`` really is the free-joint
root body (and the body that carries the IMU the policy reads).
"""

from copy import deepcopy
from dataclasses import replace

import mujoco
import pytest

from mjlab.asset_zoo.robots.agibot_x2 import x2_constants
from mjlab.entity import Entity
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.config.agibot_x2.env_cfgs import (
  agibot_x2_flat_tracking_correlated_dr_env_cfg,
)
from mjlab.tasks.tracking.mdp import MotionCommandCfg

PREFIX = "Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR-"
REDUCED = PREFIX + "Reduced-Perturbations"
PELVIS = REDUCED + "-Pelvis-Anchor"


@pytest.fixture(autouse=True)
def clear_axis_selection(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.delenv("MJLAB_DR_AXES", raising=False)


@pytest.fixture(scope="module")
def x2_entity() -> Entity:
  return Entity(x2_constants.get_x2_robot_cfg())


@pytest.fixture(scope="module")
def x2_model(x2_entity: Entity) -> mujoco.MjModel:
  return x2_entity.spec.compile()


def motion_cfg(cfg) -> MotionCommandCfg:
  motion = cfg.commands["motion"]
  assert isinstance(motion, MotionCommandCfg)
  return motion


def test_task_is_registered_with_its_own_experiment_directory() -> None:
  assert PELVIS in list_tasks()
  assert load_rl_cfg(PELVIS).experiment_name == (
    "agibot_x2_tracking_correlated_dr_reduced_perturbations_pelvis_anchor"
  )
  assert load_rl_cfg(PELVIS).experiment_name != load_rl_cfg(REDUCED).experiment_name


def test_task_changes_only_the_anchor() -> None:
  parent = agibot_x2_flat_tracking_correlated_dr_env_cfg(reduced_perturbations=True)
  pelvis = agibot_x2_flat_tracking_correlated_dr_env_cfg(
    reduced_perturbations=True, anchor_body_name="pelvis"
  )

  assert motion_cfg(parent).anchor_body_name == "torso_link"
  assert motion_cfg(pelvis).anchor_body_name == "pelvis"
  assert replace(motion_cfg(pelvis), anchor_body_name="torso_link") == motion_cfg(
    parent
  )

  assert pelvis.observations == parent.observations
  assert pelvis.rewards == parent.rewards
  assert pelvis.terminations == parent.terminations
  assert pelvis.actions == parent.actions
  assert pelvis.events == parent.events
  assert pelvis.scene.entities == parent.scene.entities
  assert pelvis.sim == parent.sim

  # Anchor-sensitive terms keep their identities; only the frame they resolve
  # in moves from the torso to the pelvis.
  assert {"motion_global_root_pos", "motion_global_root_ori"} <= set(pelvis.rewards)
  assert {"anchor_pos", "anchor_ori"} <= set(pelvis.terminations)
  assert {
    "motion_anchor_pos_b",
    "motion_anchor_ori_b",
  } <= set(pelvis.observations["critic"].terms)


def test_registered_play_config_keeps_the_pelvis_anchor() -> None:
  cfg = load_env_cfg(PELVIS, play=True)
  motion = motion_cfg(cfg)

  assert motion.anchor_body_name == "pelvis"
  assert cfg.observations["actor"].enable_corruption is False
  assert "push_robot" not in cfg.events
  assert motion.sampling_mode == "start"
  assert motion.pose_range == {}
  assert motion.velocity_range == {}
  assert motion.joint_position_range == (-0.05, 0.05)


def test_building_the_pelvis_variant_does_not_mutate_the_parent() -> None:
  before = agibot_x2_flat_tracking_correlated_dr_env_cfg(reduced_perturbations=True)
  motion_before = deepcopy(motion_cfg(before))

  _ = agibot_x2_flat_tracking_correlated_dr_env_cfg(
    reduced_perturbations=True, anchor_body_name="pelvis"
  )
  _ = load_env_cfg(PELVIS)
  _ = load_env_cfg(PELVIS, play=True)

  after = agibot_x2_flat_tracking_correlated_dr_env_cfg(reduced_perturbations=True)
  assert motion_cfg(after) == motion_before
  assert motion_cfg(after).anchor_body_name == "torso_link"


def test_unknown_anchor_is_rejected_at_config_time() -> None:
  with pytest.raises(ValueError, match="must be one of the tracked"):
    agibot_x2_flat_tracking_correlated_dr_env_cfg(anchor_body_name="torso")


def test_pelvis_is_the_free_joint_root_that_carries_the_imu(
  x2_model: mujoco.MjModel,
) -> None:
  free_joints = [
    j
    for j in range(x2_model.njnt)
    if x2_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
  ]
  assert len(free_joints) == 1
  root_body = int(x2_model.jnt_bodyid[free_joints[0]])

  def body_name(body_id: int) -> str:
    return mujoco.mj_id2name(x2_model, mujoco.mjtObj.mjOBJ_BODY, body_id)

  def site_body_name(site: str) -> str:
    site_id = mujoco.mj_name2id(x2_model, mujoco.mjtObj.mjOBJ_SITE, site)
    assert site_id >= 0
    return body_name(int(x2_model.site_bodyid[site_id]))

  assert body_name(root_body) == "pelvis"
  # The observations the policy reads from the robot itself are pelvis-mounted:
  # base_ang_vel/base_lin_vel come from these sensors, and projected gravity is
  # the root-link frame, which is the same body.
  assert site_body_name("imu_0") == "pelvis"
  assert site_body_name("imu_1") == "torso_link"
  # torso_link is neither the root nor the directly measured frame, which is
  # why the deployment reconstructs it from the pelvis IMU plus waist joints.
  assert site_body_name("imu_0") != site_body_name("imu_1")


def test_pelvis_is_the_tracked_anchor_index(x2_entity: Entity) -> None:
  cfg = agibot_x2_flat_tracking_correlated_dr_env_cfg(anchor_body_name="pelvis")
  motion = motion_cfg(cfg)

  # MotionCommand resolves the anchor by name against the tracked body list and
  # against the robot model's bodies; both must find the pelvis.
  assert motion.body_names.index(motion.anchor_body_name) == 0
  assert x2_entity.body_names.index(motion.anchor_body_name) == 0
  # Index 0 is also the RSI reference: ``_resample_command`` writes
  # ``body_pos_w[:, 0]`` to the simulated root, so the first tracked body has to
  # be the root body for either anchor choice.
  assert motion.body_names.index("torso_link") > motion.body_names.index("pelvis")
