"""AgiBot X2 Ultra flat tracking environment configurations."""

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots import (
  X2_ACTION_SCALE,
  get_x2_robot_cfg,
)
from mjlab.asset_zoo.robots.agibot_x2.x2_constants import X2_ARTICULATION
from mjlab.envs import ManagerBasedRlEnvCfg
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

# Index of the wrist pitch/roll actuator group in ``X2_ARTICULATION``, checked
# against the model below so a layout change cannot silently miss it.
_WRIST_ACTUATOR_ID = 4


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

  # Wrist-specific derating. The vendor simulator clamps the wrist pitch/roll
  # motors to 2.2 N.m where the vendor URDF allows 4.8, and the wrists are
  # end-effector bodies in the termination set, so train against a much weaker
  # wrist than nominal. This term must come after the whole-robot effort term:
  # both write the same fields and only the last writer takes effect.
  wrist_group = X2_ARTICULATION.actuators[_WRIST_ACTUATOR_ID]
  assert isinstance(wrist_group, BuiltinPositionActuatorCfg)
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
        "effort_limit_range": (0.45, 1.0),
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
