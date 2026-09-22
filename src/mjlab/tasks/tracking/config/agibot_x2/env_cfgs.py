"""AgiBot X2 Ultra flat tracking environment configurations."""

from copy import deepcopy
from typing import Literal

from mjlab.asset_zoo.robots import (
  X2_ACTION_SCALE,
  get_x2_robot_cfg,
)
from mjlab.asset_zoo.robots.agibot_x2.x2_constants import (
  X2_ACTUATOR_GROUPS,
  actuator_group_index,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

# The 12 collision spheres per foot in the ankle-roll body.
_FOOT_GEOMS = r"^(left|right)_foot[0-9]+_collision$"

# Index of the wrist pitch/roll group in ``X2_ARTICULATION``, resolved by name so
# a change to the group layout cannot silently point this at another joint.
_WRIST_ACTUATOR_ID = actuator_group_index("wrist_pitch_roll")

# The generic joint-damping range is an absolute value, not a scale, and it is
# calibrated on a 120 N.m leg joint: even at the tracking tasks' narrowed bound
# (0.03 N.m.s/rad) that is far too much passive drag for a 0.6 N.m head joint.
# The X2 therefore replaces the range for *every* group, scaled to its torque
# class; ``effort_limit / 4000`` reproduces the generic 0.03 for the legs.
X2_DAMPING_RANGES: dict[str, tuple[float, float]] = {
  pattern: (0.0, group.effort_limit / 4000.0)
  for group in X2_ACTUATOR_GROUPS.values()
  for pattern in group.target_names_expr
  if group.effort_limit is not None
}

X2TrackingObservationAblation = Literal[
  "projected_gravity",
  "projected_gravity_anchor",
  "vendor_velocity_scaling",
]
"""Actor-observation changes evaluated from the reduced-perturbation baseline."""


def agibot_x2_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create AgiBot X2 Ultra flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_x2_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = X2_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "left_wrist_pitch_link",
    "left_wrist_roll_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
    "right_wrist_pitch_link",
    "right_wrist_roll_link",
    "head_yaw_link",
    "head_pitch_link",
  )

  # Wrist flexion is secondary: keep it tracked, but weight it below the main
  # body chain so the policy prioritizes core pose and velocity tracking.
  # The tuple aligns with ``motion_cmd.body_names``; only the wrist pitch/roll
  # links (the newly tracked end effectors) are down-weighted.
  secondary_weights = {
    "left_wrist_pitch_link": 0.3,
    "left_wrist_roll_link": 0.3,
    "right_wrist_pitch_link": 0.3,
    "right_wrist_roll_link": 0.3,
  }
  body_weights = tuple(
    secondary_weights.get(name, 1.0) for name in motion_cmd.body_names
  )
  for reward_name in (
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
  ):
    cfg.rewards[reward_name].params["body_weights"] = body_weights

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = _FOOT_GEOMS
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  if "randomize_foot_size" in cfg.events:
    cfg.events["randomize_foot_size"].params["asset_cfg"].geom_names = _FOOT_GEOMS

  # Replace the robot-agnostic absolute joint-damping range with one scaled to
  # each group's torque class (see X2_DAMPING_RANGES). The ranges are replaced on
  # the generic event rather than added as a second one, so there is no ordering
  # dependency between the two.
  if "randomize_joint_damping" in cfg.events:
    cfg.events["randomize_joint_damping"].params["ranges"] = X2_DAMPING_RANGES

  # Wrist-specific effort derating. The vendor simulator clamps the wrist
  # pitch/roll motors to 2.2 N.m where the vendor URDF allows 4.8, and the wrists
  # are end-effector bodies in the termination set, so this term used to train
  # against a 0.45-1.0x weaker wrist. The tracking task's sim-to-real ranges are
  # now capped at 5%, which leaves it a near no-op: modelling the clamp now means
  # changing the wrist's *nominal* effort limit to 2.2 N.m and dropping this term
  # (open item 5.6 of docs/source/x2_gain_provenance.md). Kept as the 5% residual
  # so the axis wiring and its ordering assertion stay in place.
  # Must come after the whole-robot effort term: both write the same fields and
  # only the last writer takes effect.
  wrist_group = X2_ACTUATOR_GROUPS["wrist_pitch_roll"]
  assert wrist_group.effort_limit == 4.8, wrist_group.effort_limit
  assert wrist_group.target_names_expr == (
    ".*_wrist_pitch_joint",
    ".*_wrist_roll_joint",
  ), wrist_group.target_names_expr
  if "randomize_effort_limits" in cfg.events:
    cfg.events["randomize_effort_limits_wrist"] = EventTermCfg(
      mode="startup",
      func=dr.effort_limits,
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_ids=[_WRIST_ACTUATOR_ID]),
        "effort_limit_range": (0.95, 1.0),
        "operation": "scale",
      },
    )
    event_order = list(cfg.events)
    assert event_order.index("randomize_effort_limits_wrist") > event_order.index(
      "randomize_effort_limits"
    )

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  cfg.viewer.body_name = "torso_link"

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg


def agibot_x2_flat_tracking_correlated_dr_env_cfg(
  reduced_perturbations: bool = False,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the X2 no-state-estimation correlated-DR ablation configuration."""
  cfg = agibot_x2_flat_tracking_env_cfg(has_state_estimation=False, play=play)

  if "randomize_pd_gains" in cfg.events:
    cfg.events["randomize_pd_gains"].params["shared_gain_scale"] = True

  actor_terms = cfg.observations["actor"].terms
  if actor_terms["joint_pos"].delay_max_lag > 0:
    for term_name in ("joint_pos", "joint_vel"):
      term = actor_terms[term_name]
      term.delay_group = "encoder_packet"
      term.delay_hold_prob = 0.9
      term.delay_update_period = 1
    base_ang_vel = actor_terms["base_ang_vel"]
    base_ang_vel.delay_hold_prob = 0.9
    base_ang_vel.delay_update_period = 1

  if reduced_perturbations:
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.pose_range = {
      axis: (lower * 0.5, upper * 0.5)
      for axis, (lower, upper) in motion_cmd.pose_range.items()
    }
    motion_cmd.velocity_range = {
      axis: (lower * 0.5, upper * 0.5)
      for axis, (lower, upper) in motion_cmd.velocity_range.items()
    }
    lower, upper = motion_cmd.joint_position_range
    motion_cmd.joint_position_range = (lower * 0.5, upper * 0.5)
    if "push_robot" in cfg.events:
      push = cfg.events["push_robot"]
      push.interval_range_s = (4.0, 8.0)
      push.params["velocity_range"] = {
        axis: (lower * 0.5, upper * 0.5)
        for axis, (lower, upper) in push.params["velocity_range"].items()
      }

  return cfg


def agibot_x2_flat_tracking_observation_ablation_env_cfg(
  ablation: X2TrackingObservationAblation,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a fresh actor-observation ablation from the reduced-DR baseline."""
  cfg = agibot_x2_flat_tracking_correlated_dr_env_cfg(
    reduced_perturbations=True, play=play
  )
  actor_observations = cfg.observations["actor"]
  actor_terms = actor_observations.terms

  if ablation in ("projected_gravity", "projected_gravity_anchor"):
    anchor_orientation = actor_terms["motion_anchor_ori_b"]
    # Deep-copy the reference term's corruption configuration before replacing
    # its motion-dependent callable. Projected gravity is a root-body measurement.
    projected_gravity = deepcopy(anchor_orientation)
    projected_gravity.func = env_mdp.projected_gravity
    projected_gravity.params = {}
    projected_gravity.scale = 1.0

    new_actor_terms = {}
    for name, term in actor_terms.items():
      if name == "motion_anchor_ori_b":
        new_actor_terms["projected_gravity"] = projected_gravity
        if ablation == "projected_gravity_anchor":
          new_actor_terms[name] = term
      else:
        new_actor_terms[name] = term
    actor_observations.terms = new_actor_terms
  elif ablation == "vendor_velocity_scaling":
    actor_terms["base_ang_vel"].scale = 0.25
    actor_terms["joint_vel"].scale = 0.05
  else:
    raise ValueError(f"Unknown X2 tracking observation ablation: {ablation}")

  return cfg
