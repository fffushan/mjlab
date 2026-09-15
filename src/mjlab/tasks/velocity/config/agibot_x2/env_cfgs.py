"""AgiBot X2 Ultra velocity environment configurations."""

import mujoco

from mjlab.asset_zoo.robots import X2_ACTION_SCALE, get_x2_robot_cfg
from mjlab.asset_zoo.robots.agibot_x2.x2_constants import get_spec
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

# The 12 collision spheres per foot live in the ankle-roll body.
_FOOT_GEOMS = r"^(left|right)_foot[0-9]+_collision$"

# Foot-site pose in the ankle-roll body frame: the centroid of the 12 foot
# spheres (0.0298, 0, -0.0638). The vendored XML carries no foot sites, so the
# velocity config injects them at spec build time; the XML file stays
# byte-identical.
_FOOT_SITE_POS = (0.03, 0.0, -0.064)


def _x2_velocity_spec_fn() -> mujoco.MjSpec:
  """Load the X2 spec and add the foot sites the velocity task needs."""
  spec = get_spec()
  for side in ("left", "right"):
    body = spec.body(f"{side}_ankle_roll_link")
    body.add_site(
      name=f"{side}_foot",
      pos=_FOOT_SITE_POS,
      size=(0.01,),
      type=mujoco.mjtGeom.mjGEOM_SPHERE,
    )
  return spec


def agibot_x2_flat_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create AgiBot X2 Ultra flat terrain velocity configuration.

  This is the no-state-estimation variant: the actor observes only IMU and
  joint measurements (gyro, IMU up-vector, encoders). ``base_lin_vel`` is
  dropped because linear velocity requires an estimator on hardware, and
  ``projected_gravity`` is read from the IMU up-vector sensor instead of the
  root-link orientation.
  """
  cfg = make_velocity_env_cfg()

  cfg.sim.njmax = 500
  cfg.sim.nconmax = 70
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64

  robot_cfg = get_x2_robot_cfg()
  robot_cfg.spec_fn = _x2_velocity_spec_fn
  cfg.scene.entities = {"robot": robot_cfg}

  # Flat terrain: no height scan, no terrain curriculum.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  site_names = ("left_foot", "right_foot")

  # Wire the foot height scan to the injected per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  # Feet contact: the 12 collision spheres per foot are geoms of the ankle-roll
  # body, so the subtree of the ankle-roll links matches exactly the feet.
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = X2_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.2

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = _FOOT_GEOMS
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Matched kp/kd scaling (see the G1 bandwidth analysis in the DR doc): wider
  # or kp-only ranges leave the heavy joints oscillatory.
  cfg.events["randomize_pd_gains"] = EventTermCfg(
    mode="reset",
    func=dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "kp_range": (0.7, 1.3),
      "kd_range": (0.7, 1.3),
      "operation": "scale",
    },
  )

  # No-state-estimation observation set. The IMU up-vector sensor
  # (``framezaxis`` on imu_0) gives the body-frame gravity direction directly.
  del cfg.observations["actor"].terms["base_lin_vel"]
  cfg.observations["actor"].terms["projected_gravity"] = ObservationTermCfg(
    func=mdp.projected_gravity_from_sensor,
    params={"sensor_name": "robot/imu_upvector"},
    noise=Unoise(n_min=-0.05, n_max=0.05),
  )
  # Critic keeps privileged base velocity (teacher signal only).
  cfg.observations["actor"] = ObservationGroupCfg(
    terms=cfg.observations["actor"].terms,
    concatenate_terms=True,
    enable_corruption=True,
  )

  # Per-joint posture tolerance by speed regime. Head joints are loose (they do
  # not affect balance); wrists loose as in the tracking task; ankles tight for
  # balance; hips/knees the loosest for stride.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*_hip_pitch.*": 0.3,
    r".*_hip_roll.*": 0.15,
    r".*_hip_yaw.*": 0.15,
    r".*_knee.*": 0.35,
    r".*_ankle_pitch.*": 0.25,
    r".*_ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*_shoulder_pitch.*": 0.15,
    r".*_shoulder_roll.*": 0.15,
    r".*_shoulder_yaw.*": 0.1,
    r".*_elbow.*": 0.15,
    r".*_wrist.*": 0.3,
    # Head.
    r".*head.*": 0.3,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*_hip_pitch.*": 0.5,
    r".*_hip_roll.*": 0.2,
    r".*_hip_yaw.*": 0.2,
    r".*_knee.*": 0.6,
    r".*_ankle_pitch.*": 0.35,
    r".*_ankle_roll.*": 0.15,
    # Waist.
    r".*waist_yaw.*": 0.3,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.2,
    # Arms.
    r".*_shoulder_pitch.*": 0.5,
    r".*_shoulder_roll.*": 0.2,
    r".*_shoulder_yaw.*": 0.15,
    r".*_elbow.*": 0.35,
    r".*_wrist.*": 0.3,
    # Head.
    r".*head.*": 0.3,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum.pop("terrain_levels", None)

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}

    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
