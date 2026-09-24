"""AgiBot X2 Ultra velocity environment configurations."""

from dataclasses import replace
from typing import Literal

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
  BuiltinSensorCfg,
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

# Base-observation IMU sources. The XML declares the gyro, velocimeter and
# up-vector framezaxis on site ``imu_0`` (pelvis); ``imu_1`` carries an
# equivalent IMU body on ``torso_link`` but no sensors, so the torso variants
# declare them here rather than editing the shared XML.
_PELVIS_GYRO = "robot/imu_ang_vel"
_PELVIS_VELOCIMETER = "robot/imu_lin_vel"
_PELVIS_UPVECTOR = "robot/imu_upvector"

_TORSO_GYRO = "robot/imu_1_ang_vel"
_TORSO_VELOCIMETER = "robot/imu_1_lin_vel"
# A ``framezaxis`` whose object is the *world* body has no entity to take a
# prefix from, so ``BuiltinSensorCfg.prefixed_name`` cannot add ``robot/`` and
# the sensor compiles under this bare name. Pinned by
# ``tests/test_x2_velocity_torso_imu.py``.
_TORSO_UPVECTOR = "imu_1_upvector"

X2VelocityImuSource = Literal["pelvis", "torso"]
"""IMU the base angular/linear velocity observations are read from."""

X2VelocityCriticGravitySource = Literal["root", "upvector"]
"""Frame the critic's projected gravity is read in.

``root`` is the privileged root-link (pelvis) orientation the critic inherits
from the base config. ``upvector`` is the torso IMU's world-Z reading, i.e. the
same source the actor uses, but without its noise.
"""


def _torso_imu_sensors(
  *, gyro: bool, velocimeter: bool, upvector: bool
) -> tuple[BuiltinSensorCfg, ...]:
  """Declare only the torso IMU sensors the selected variant references."""
  sensors: list[BuiltinSensorCfg] = []
  if gyro:
    sensors.append(
      BuiltinSensorCfg(
        name="imu_1_ang_vel",
        sensor_type="gyro",
        obj=ObjRef(type="site", name="imu_1", entity="robot"),
      )
    )
  if velocimeter:
    sensors.append(
      BuiltinSensorCfg(
        name="imu_1_lin_vel",
        sensor_type="velocimeter",
        obj=ObjRef(type="site", name="imu_1", entity="robot"),
      )
    )
  if upvector:
    sensors.append(
      BuiltinSensorCfg(
        name="imu_1_upvector",
        sensor_type="framezaxis",
        obj=ObjRef(type="body", name="world"),
        ref=ObjRef(type="site", name="imu_1", entity="robot"),
      )
    )
  return tuple(sensors)


def _apply_imu_source(
  cfg: ManagerBasedRlEnvCfg,
  imu_source: X2VelocityImuSource,
  critic_imu_source: X2VelocityImuSource,
  critic_gravity_source: X2VelocityCriticGravitySource,
) -> None:
  """Point the base velocity and gravity observations at the selected IMUs.

  Terms are *replaced* rather than mutated. The base velocity config builds its
  critic terms as ``{**actor_terms}``, which copies references, so
  ``base_ang_vel`` is one shared object between the actor and the critic group:
  writing ``params["sensor_name"]`` once would silently point both at the same
  source and collapse the two ablation axes into one.
  """
  if (imu_source, critic_imu_source, critic_gravity_source) == (
    "pelvis",
    "pelvis",
    "root",
  ):
    # The shipped configuration: leave the terms exactly as the base config
    # built them, so this task is byte-identical rather than merely equivalent.
    return

  need_torso_gyro = imu_source == "torso" or critic_imu_source == "torso"
  need_torso_velocimeter = critic_imu_source == "torso"
  need_torso_upvector = imu_source == "torso" or critic_gravity_source == "upvector"
  if need_torso_gyro or need_torso_velocimeter or need_torso_upvector:
    cfg.scene.sensors = (cfg.scene.sensors or ()) + _torso_imu_sensors(
      gyro=need_torso_gyro,
      velocimeter=need_torso_velocimeter,
      upvector=need_torso_upvector,
    )

  def reread(term, sensor_name: str):
    return replace(term, params={**term.params, "sensor_name": sensor_name})

  actor = cfg.observations["actor"].terms
  critic = cfg.observations["critic"].terms

  actor["base_ang_vel"] = reread(
    actor["base_ang_vel"], _TORSO_GYRO if imu_source == "torso" else _PELVIS_GYRO
  )
  critic["base_ang_vel"] = reread(
    critic["base_ang_vel"],
    _TORSO_GYRO if critic_imu_source == "torso" else _PELVIS_GYRO,
  )
  # The no-state-estimation actor has no base_lin_vel; only the critic's
  # privileged copy is re-pointed.
  critic["base_lin_vel"] = reread(
    critic["base_lin_vel"],
    _TORSO_VELOCIMETER if critic_imu_source == "torso" else _PELVIS_VELOCIMETER,
  )
  # The actor's projected_gravity is already the IMU up-vector term, so it only
  # changes source; its noise is preserved.
  actor["projected_gravity"] = reread(
    actor["projected_gravity"],
    _TORSO_UPVECTOR if imu_source == "torso" else _PELVIS_UPVECTOR,
  )
  if critic_gravity_source == "upvector":
    critic["projected_gravity"] = ObservationTermCfg(
      func=mdp.projected_gravity_from_sensor,
      params={"sensor_name": _TORSO_UPVECTOR},
    )


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


def agibot_x2_flat_velocity_env_cfg(
  play: bool = False,
  imu_source: X2VelocityImuSource = "pelvis",
  critic_imu_source: X2VelocityImuSource = "pelvis",
  critic_gravity_source: X2VelocityCriticGravitySource = "root",
) -> ManagerBasedRlEnvCfg:
  """Create AgiBot X2 Ultra flat terrain velocity configuration.

  This is the no-state-estimation variant: the actor observes only IMU and
  joint measurements (gyro, IMU up-vector, encoders). ``base_lin_vel`` is
  dropped because linear velocity requires an estimator on hardware, and
  ``projected_gravity`` is read from the IMU up-vector sensor instead of the
  root-link orientation.

  ``imu_source`` selects which physical IMU the *actor's* gyro and up-vector
  come from; ``critic_imu_source`` and ``critic_gravity_source`` do the same for
  the critic's privileged velocity and gravity terms. All three default to the
  shipped behavior (pelvis ``imu_0`` gyro, velocimeter and up-vector, with the
  critic's gravity from the root-link orientation).
  """
  if imu_source not in ("pelvis", "torso"):
    raise ValueError(
      f"imu_source {imu_source!r} must be 'pelvis' (imu_0 on the pelvis) or "
      f"'torso' (imu_1 on torso_link)"
    )
  if critic_imu_source not in ("pelvis", "torso"):
    raise ValueError(
      f"critic_imu_source {critic_imu_source!r} must be 'pelvis' or 'torso'"
    )
  if critic_gravity_source not in ("root", "upvector"):
    raise ValueError(
      f"critic_gravity_source {critic_gravity_source!r} must be 'root' (the "
      f"root-link orientation) or 'upvector' (the torso IMU up-vector)"
    )

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

  # Applied after the no-state-estimation observation edits, so the actor term
  # it re-points is the final up-vector term, and before the play overrides,
  # which do not touch sensor names.
  _apply_imu_source(cfg, imu_source, critic_imu_source, critic_gravity_source)

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
