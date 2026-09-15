"""Motion mimic task configuration.

This module defines the base configuration for motion mimic tasks.
Robot-specific configurations are located in the config/ directory.

This is a re-implementation of BeyondMimic (https://beyondmimic.github.io/).

Based on https://github.com/HybridRobotics/whole_body_tracking
Commit: f8e20c880d9c8ec7172a13d3a88a65e3a5a88448
"""

import math
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

VELOCITY_RANGE = {
  "x": (-0.5, 0.5),
  "y": (-0.5, 0.5),
  "z": (-0.2, 0.2),
  "roll": (-0.52, 0.52),
  "pitch": (-0.52, 0.52),
  "yaw": (-0.78, 0.78),
}

DR_AXES: tuple[str, ...] = (
  "inertia",
  "armature",
  "effort_limits",
  "joint_friction",
  "joint_damping",
  "foot_size",
  "pd_gains",
  "obs_delay",
)
"""Sim-to-real randomization axes wired into the tracking tasks.

Ranges and rationale are in ``docs/source/sim2real_domain_randomization.md``.
Every axis is on by default. ``MJLAB_DR_AXES`` selects a subset at build time so
a trained policy's sensitivity can be swept one axis at a time, e.g.

  MJLAB_DR_AXES=armature uv run python scripts/evaluate_tracking_policy.py ...
  MJLAB_DR_AXES=none     # only the pre-existing set, for a baseline
"""


def selected_dr_axes() -> set[str]:
  """Return the randomization axes enabled for this process."""
  raw = os.environ.get("MJLAB_DR_AXES", "all").strip().lower()
  if raw in ("", "all"):
    return set(DR_AXES)
  if raw == "none":
    return set()
  selected = {axis.strip() for axis in raw.split(",") if axis.strip()}
  unknown = selected - set(DR_AXES)
  if unknown:
    raise ValueError(
      f"MJLAB_DR_AXES has unknown axes {sorted(unknown)}; choose from {list(DR_AXES)}"
    )
  return selected


def make_tracking_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base tracking task configuration."""

  dr_axes = selected_dr_axes()

  ##
  # Observations
  ##

  actor_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    "motion_lookahead": ObservationTermCfg(
      func=mdp.motion_lookahead,
      params={"command_name": "motion"},
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.25, n_max=0.25),
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      params={"biased": True},
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  if "obs_delay" in dr_axes:
    # Sensor-pipeline latency: encoder/IMU transport and filtering delay the
    # measurements the policy acts on (1 lag = 20 ms at the 50 Hz policy rate).
    # This is distinct from the command delay, which is set on the actuators.
    # Terms read from the reference motion are not delayed. A whole policy step,
    # or none: a lag has no nominal to take a percentage of, and one step is
    # already the smallest non-zero latency there is.
    for term_name in ("joint_pos", "joint_vel", "base_ang_vel"):
      actor_terms[term_name].delay_min_lag = 0
      actor_terms[term_name].delay_max_lag = 1

  critic_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}
    ),
    "body_pos": ObservationTermCfg(
      func=mdp.robot_body_pos_b, params={"command_name": "motion"}
    ),
    "body_ori": ObservationTermCfg(
      func=mdp.robot_body_ori_b, params={"command_name": "motion"}
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}
    ),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "motion": MotionCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      pose_range={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
      },
      velocity_range=VELOCITY_RANGE,
      joint_position_range=(-0.1, 0.1),
      # Override in robot cfg.
      motion_file="",
      anchor_body_name="",
      body_names=(),
    )
  }

  ##
  # Events
  ##

  events: dict[str, EventTermCfg] = {
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={"velocity_range": VELOCITY_RANGE},
    ),
  }

  if "inertia" in dr_axes:
    # Physically consistent mass/inertia/COM randomization (Rucker & Wensing
    # 2022): alpha scales mass and inertia by exp(2*alpha), t shifts the COM in
    # the body frame. Placed before base_com so the torso keeps its own payload
    # offset from the term below.
    events["randomize_inertia"] = EventTermCfg(
      mode="startup",
      func=dr.pseudo_inertia,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        # 0.95-1.05x mass, with inertia following by the same factor.
        "alpha_range": (math.log(0.95) / 2.0, math.log(1.05) / 2.0),
        # COM shift in the body frame, m. A translation has no nominal to take
        # 5% of, so it is halved along with the mass range it accompanies.
        "t_range": (-0.01, 0.01),
      },
    )

  events |= {
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set in robot cfg.
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.01, 0.01),
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,  # All foot geoms share the same friction.
      },
    ),
  }

  ##
  # Domain randomization (sim-to-real)
  ##
  # Ranges follow docs/source/sim2real_domain_randomization.md; each axis is
  # disabled by removing it from MJLAB_DR_AXES. Order matters for terms that
  # write the same model field: every dr.* function samples from the compiled
  # defaults, so the last writer wins.
  #
  # These axes exist only here: the velocity tasks randomize nothing but the
  # actuator PD gains (see their configs). The tracking task has to reproduce a
  # measured hardware behaviour, so its sim-to-real ranges are held to 5% of
  # nominal - wide enough that a policy cannot over-fit the exact model, narrow
  # enough that it trains against roughly the machine it will be deployed on.

  if "armature" in dr_axes:
    # Reflected rotor inertia, from the PFP module table. The nominal PD gains
    # used to be derived from it, so this axis also decoupled a fixed
    # gain/inertia relationship; where the gains are measured hardware values
    # instead (the X2) it probes the transmission inertia alone.
    events["randomize_armature"] = EventTermCfg(
      mode="startup",
      func=dr.joint_armature,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "ranges": (0.95, 1.05),
        "operation": "scale",
      },
    )

  if "effort_limits" in dr_axes:
    # Weaker-only: a policy that needs more torque than the hardware can deliver
    # is the failure mode that matters, and the vendor simulator itself derates
    # its motors below the URDF peaks.
    events["randomize_effort_limits"] = EventTermCfg(
      mode="startup",
      func=dr.effort_limits,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "effort_limit_range": (0.95, 1.0),
        "operation": "scale",
      },
    )

  if "joint_friction" in dr_axes:
    events["randomize_joint_friction"] = EventTermCfg(
      mode="startup",
      func=dr.joint_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "ranges": (0.95, 1.05),
        "operation": "scale",
      },
    )

  if "joint_damping" in dr_axes:
    # The shipped models have damping=0.0, i.e. no viscous transmission loss, so
    # there is no nominal to take a percentage of: the range stays absolute and
    # is one tenth of what it was, which leaves it all but inert against the
    # active derivative gains (0.03 against Kd 4 on an X2 hip). Robot configs
    # should scale it with their own torque class; see X2_DAMPING_RANGES.
    events["randomize_joint_damping"] = EventTermCfg(
      mode="startup",
      func=dr.joint_damping,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "ranges": (0.0, 0.03),
        "operation": "abs",
      },
    )

  if "foot_size" in dr_axes:
    events["randomize_foot_size"] = EventTermCfg(
      mode="startup",
      func=dr.geom_size,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        # Already inside the 5% cap; left as it is.
        "ranges": (0.97, 1.03),
        "operation": "scale",
        "shared_random": True,  # All foot geoms share the same scale.
      },
    )

  if "pd_gains" in dr_axes:
    # Matched kp/kd scaling (see the G1 bandwidth analysis in the DR doc): wider
    # or kp-only ranges leave the heavy joints oscillatory.
    events["randomize_pd_gains"] = EventTermCfg(
      mode="reset",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "kp_range": (0.7, 1.3),
        "kd_range": (0.7, 1.3),
        "operation": "scale",
      },
    )

  ##
  # Rewards
  ##

  rewards: dict[str, RewardTermCfg] = {
    "motion_global_root_pos": RewardTermCfg(
      func=mdp.motion_global_anchor_position_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_global_root_ori": RewardTermCfg(
      func=mdp.motion_global_anchor_orientation_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_pos": RewardTermCfg(
      func=mdp.motion_relative_body_position_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_body_ori": RewardTermCfg(
      func=mdp.motion_relative_body_orientation_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_lin_vel": RewardTermCfg(
      func=mdp.motion_global_body_linear_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 1.0},
    ),
    "motion_body_ang_vel": RewardTermCfg(
      func=mdp.motion_global_body_angular_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 3.14},
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-1),
    "joint_limit": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-10.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
  }

  ##
  # Terminations
  ##

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "anchor_pos": TerminationTermCfg(
      func=mdp.bad_anchor_pos_z_only,
      params={"command_name": "motion", "threshold": 0.25},
    ),
    "anchor_ori": TerminationTermCfg(
      func=mdp.bad_anchor_ori,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "command_name": "motion",
        "threshold": 0.8,
      },
    ),
    "ee_body_pos": TerminationTermCfg(
      func=mdp.bad_motion_body_pos_z_only,
      params={
        "command_name": "motion",
        "threshold": 0.25,
        "body_names": (),  # Set per-robot.
      },
    ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=2.8,
      fovy=55.0,
      elevation=-5.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=250,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=10.0,
  )
