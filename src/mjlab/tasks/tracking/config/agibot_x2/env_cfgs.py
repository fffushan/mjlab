"""AgiBot X2 Ultra flat tracking environment configurations."""

from copy import deepcopy
from dataclasses import fields
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
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import BuiltinSensorCfg, ContactMatch, ContactSensorCfg, ObjRef
from mjlab.tasks.tracking.mdp import MotionCommandCfg, motion_lookahead
from mjlab.tasks.tracking.mdp.multi_motion_ppo import (
  MultiMotionPpoCommandCfg,
  clip_complete,
)
from mjlab.tasks.tracking.tracking_env_cfg import VELOCITY_RANGE, make_tracking_env_cfg

from .vendor_gains import apply_vendor_rl_gains

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

X2TrackingImuSource = Literal["pelvis", "torso"]
"""IMU frame the actor's base velocity observations are read from.

``pelvis`` keeps the entity XML's gyro/velocimeter on site ``imu_0`` (the
historical behavior); ``torso`` adds task-local gyro/velocimeter sensors on
site ``imu_1`` (a child of ``torso_link``) and re-points the base angular and
linear velocity observations at them. Site ``imu_1`` is rigidly attached to
``torso_link``, so its gyro reads the torso frame's angular velocity, which
already contains the waist joint rates rather than a rotation of the pelvis
angular velocity.
"""

_TORSO_IMU_SITE = "imu_1"
_TORSO_IMU_GYRO = "robot/imu_1_ang_vel"
_TORSO_IMU_VELOCIMETER = "robot/imu_1_lin_vel"

_MULTI_MOTION_PREVIEW_S = 0.2
"""Reference preview offset of the multi-motion teacher, in seconds.

Ten reference steps at the task's 50 Hz control rate, which is one future
frame of joint positions and velocities.
"""


def _torso_imu_sensors() -> tuple[BuiltinSensorCfg, BuiltinSensorCfg]:
  """Task-local torso IMU sensors, named to avoid the entity's ``imu_0`` ones."""
  return (
    BuiltinSensorCfg(
      name="imu_1_ang_vel",
      sensor_type="gyro",
      obj=ObjRef(type="site", name=_TORSO_IMU_SITE, entity="robot"),
    ),
    BuiltinSensorCfg(
      name="imu_1_lin_vel",
      sensor_type="velocimeter",
      obj=ObjRef(type="site", name=_TORSO_IMU_SITE, entity="robot"),
    ),
  )


def _select_imu_source(
  cfg: ManagerBasedRlEnvCfg, imu_source: X2TrackingImuSource
) -> None:
  """Point the base velocity observations at the selected IMU and add its sensors.

  ``pelvis`` is a no-op: the observation params already name the entity XML's
  ``robot/imu_ang_vel``/``robot/imu_lin_vel`` sensors on site ``imu_0``. ``torso``
  adds the task-local ``imu_1`` sensors to the scene and re-points every present
  ``base_ang_vel``/``base_lin_vel`` term in the actor and critic groups.
  """
  if imu_source == "pelvis":
    return
  assert imu_source == "torso"
  cfg.scene.sensors = cfg.scene.sensors + _torso_imu_sensors()
  sensor_names = {
    "base_ang_vel": _TORSO_IMU_GYRO,
    "base_lin_vel": _TORSO_IMU_VELOCIMETER,
  }
  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    for term_name, sensor_name in sensor_names.items():
      if term_name in terms:
        terms[term_name].params["sensor_name"] = sensor_name


def apply_x2_tracking_robot_randomization(cfg: ManagerBasedRlEnvCfg) -> None:
  """Point the tracking sim-to-real DR terms at the X2's own geometry.

  The generic tracking events are robot-agnostic, so three of them need X2
  values, and the wrist needs a group-specific effort term besides the
  whole-robot one:

  * ``foot_friction`` and ``randomize_foot_size`` act on the 12 collision
    spheres per foot, not on the mesh hull of the ankle-roll link.
  * ``base_com`` acts on ``torso_link``.
  * ``randomize_joint_damping`` gets :data:`X2_DAMPING_RANGES`, i.e. ranges
    scaled to each group's torque class instead of one absolute range calibrated
    on a 120 N.m leg joint.
  * ``randomize_effort_limits_wrist`` re-derates the wrist group after the
    whole-robot pass. Both write the same fields and only the last writer takes
    effect, so this term must come later in ``events``; the ordering is asserted.

  Every step is conditional on the corresponding event existing, so a caller that
  selects a subset of axes (``MJLAB_DR_AXES``) or adds these terms itself gets a
  no-op for the missing ones. Called from the tracking task and from the tennis
  ball task, which runs the same sim-to-real set.
  """
  if "foot_friction" in cfg.events:
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = _FOOT_GEOMS
  if "base_com" in cfg.events:
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
  if "randomize_effort_limits" in cfg.events:
    wrist_group = X2_ACTUATOR_GROUPS["wrist_pitch_roll"]
    assert wrist_group.effort_limit == 4.8, wrist_group.effort_limit
    assert wrist_group.target_names_expr == (
      ".*_wrist_pitch_joint",
      ".*_wrist_roll_joint",
    ), wrist_group.target_names_expr
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


def agibot_x2_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
  anchor_body_name: str = "torso_link",
  imu_source: X2TrackingImuSource = "pelvis",
) -> ManagerBasedRlEnvCfg:
  """Create AgiBot X2 Ultra flat terrain tracking configuration.

  ``anchor_body_name`` selects the body frame the reference is tracked in. It
  is *not* the simulated root: the root link is the pelvis (the free-joint body,
  which also carries ``imu_0``). The anchor is the frame of the
  ``motion_anchor_*`` observations, the ``motion_relative_body_*`` rewards, the
  ``anchor_pos``/``anchor_ori`` terminations and the ``motion_global_root_*``
  rewards, which are named after the anchor rather than the simulated root.

  ``torso_link`` is the deployment contract: the ONNX metadata records it, the
  X2 controller (``x2_docker``) reconstructs it from the pelvis IMU plus
  measured waist joints, and its policy loader rejects any other value.
  ``pelvis`` is the one frame the robot measures directly.

  ``imu_source`` selects which physical IMU the ``base_ang_vel``/``base_lin_vel``
  observations are read from independently of the anchor; see
  :data:`X2TrackingImuSource`.
  """
  if imu_source not in ("pelvis", "torso"):
    raise ValueError(
      f"imu_source {imu_source!r} must be 'pelvis' (imu_0 on the pelvis) or "
      f"'torso' (imu_1 on torso_link)"
    )

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
  motion_cmd.anchor_body_name = anchor_body_name
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
  if anchor_body_name not in motion_cmd.body_names:
    raise ValueError(
      f"anchor_body_name {anchor_body_name!r} must be one of the tracked "
      f"bodies {motion_cmd.body_names}"
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

  apply_x2_tracking_robot_randomization(cfg)

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

  # Applied after the NSE actor rebuild so the terms it re-points are the final
  # ones, and before the play overrides, which do not touch sensor names.
  _select_imu_source(cfg, imu_source)

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
  anchor_body_name: str = "torso_link",
  imu_source: X2TrackingImuSource = "pelvis",
) -> ManagerBasedRlEnvCfg:
  """Create the X2 no-state-estimation correlated-DR ablation configuration."""
  cfg = agibot_x2_flat_tracking_env_cfg(
    has_state_estimation=False,
    play=play,
    anchor_body_name=anchor_body_name,
    imu_source=imu_source,
  )

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


def agibot_x2_flat_tracking_standing_start_env_cfg(
  play: bool = False,
  anchor_body_name: str = "torso_link",
  imu_source: X2TrackingImuSource = "pelvis",
) -> ManagerBasedRlEnvCfg:
  """Reduced-perturbation tracking cfg with standing-pose episode starts.

  Fork of the reduced-perturbation correlated-DR configuration that
  additionally initializes a fraction of leading-window episode starts in the
  entity's default standing pose
  (:data:`MotionCommandCfg.standing_start_prob`). The robot's motion starts
  from the standing pose the deployed controller holds before engaging the
  policy (``JOINT_DEFAULT`` with ``pose_source: policy_metadata``), so the
  policy must learn the standing -> motion-first-frame transition; training
  against perturbed reference frames alone never shows it. The play
  configuration keeps the insert disabled: ``sampling_mode == "start"`` is
  the deterministic reference-frame initialization contract used by play and
  evaluation.
  """
  cfg = agibot_x2_flat_tracking_correlated_dr_env_cfg(
    reduced_perturbations=True,
    play=play,
    anchor_body_name=anchor_body_name,
    imu_source=imu_source,
  )
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.standing_start_prob = 0.0 if play else 0.3
  return cfg


def agibot_x2_flat_tracking_increated_perturbations_env_cfg(
  play: bool = False,
  anchor_body_name: str = "torso_link",
  imu_source: X2TrackingImuSource = "pelvis",
) -> ManagerBasedRlEnvCfg:
  """X2 standing-start tracking with the G1-sized motion perturbations.

  This keeps the X2 robot, observations, rewards, domain randomization and
  standing-start behavior, while replacing only the reduced RSI and push
  ranges with the corresponding magnitudes from the Unitree G1 tracking task.
  Play mode retains the standing-start task's deterministic no-RSI behavior.
  """
  cfg = agibot_x2_flat_tracking_standing_start_env_cfg(
    play=play,
    anchor_body_name=anchor_body_name,
    imu_source=imu_source,
  )
  if play:
    return cfg

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.pose_range = {
    "x": (-0.05, 0.05),
    "y": (-0.05, 0.05),
    "z": (-0.01, 0.01),
    "roll": (-0.1, 0.1),
    "pitch": (-0.1, 0.1),
    "yaw": (-0.2, 0.2),
  }
  motion_cmd.velocity_range = dict(VELOCITY_RANGE)
  motion_cmd.joint_position_range = (-0.1, 0.1)

  push = cfg.events.get("push_robot")
  if push is not None:
    push.interval_range_s = (1.0, 3.0)
    push.params["velocity_range"] = dict(VELOCITY_RANGE)
  return cfg


def agibot_x2_flat_tracking_increated_perturbations_vendor_gains_env_cfg(
  play: bool = False,
  anchor_body_name: str = "torso_link",
  imu_source: X2TrackingImuSource = "pelvis",
) -> ManagerBasedRlEnvCfg:
  """Increased-perturbation tracking with the vendor reference-policy PD gains.

  Identical to
  :func:`agibot_x2_flat_tracking_increated_perturbations_env_cfg` except that
  the robot's servo gains are the vendor's own reference-policy values
  (:data:`~.vendor_gains.X2_VENDOR_RL_GAINS`) instead of mjlab's measured native
  stable-stand profile. Every other quantity stays as it is in the base task --
  observations, rewards, terminations, domain randomization, the motion
  command's perturbation ranges, effort limits, armature, friction loss and the
  action scale -- so a run of this task isolates the gain change.

  The action scale is deliberately *not* recomputed from the vendor gains. It is
  reused exactly as the base task has it, because the base scale is a fixed
  interface (its per-group floors were chosen for reachability and for the 25%
  torque budget of the *mjlab* gains). Recomputing it would change a second
  quantity at the same time and make the gain change unattributable. The
  consequence is that a unit action now reaches a different fraction of the
  torque budget on the groups whose vendor Kp is higher (hip pitch 30.0% against
  25%, ankle roll 33.3%, waist pitch/roll 41.7%); the effort limit itself is
  unchanged, so those joints saturate sooner rather than acting out of range.
  """
  cfg = agibot_x2_flat_tracking_increated_perturbations_env_cfg(
    play=play,
    anchor_body_name=anchor_body_name,
    imu_source=imu_source,
  )
  apply_vendor_rl_gains(cfg.scene.entities["robot"])
  return cfg


def agibot_x2_flat_tracking_increated_perturbations_vendor_gains_no_pd_gain_dr_env_cfg(
  play: bool = False,
  anchor_body_name: str = "torso_link",
  imu_source: X2TrackingImuSource = "pelvis",
) -> ManagerBasedRlEnvCfg:
  """Vendor-gain tracking with no randomization of the servo PD gains.

  Identical to
  :func:`agibot_x2_flat_tracking_increated_perturbations_vendor_gains_env_cfg`
  except that the ``randomize_pd_gains`` event is removed. That event is the
  only term in these tasks that randomizes the servo gains kp and kd: the parent
  draws one scale per environment from ``(0.7, 1.3)`` and applies it to both,
  because it sets ``shared_gain_scale``. Without it, every episode runs the
  nominal vendor gains. The event writes only the solver's actuator
  ``gainprm``/``biasprm`` fields, so removing it is the whole change.

  The passive ``randomize_joint_damping`` axis is deliberately kept: it
  randomizes the model's ``<joint damping>`` property (about 0.03 on an X2 hip
  against a Kd of 4), which is a transmission property and not a servo gain. So
  is the 0-2 physics-step per-actuator command delay, which models the 50 Hz
  transport to the motors.

  One caveat worth knowing: the axes come from ``MJLAB_DR_AXES`` at build time.
  If that variable excludes ``pd_gains`` to begin with, the parent has no such
  event and this task builds exactly like its parent. The variable is unset by
  default (all axes on) and the job queue does not set it, so under the queue
  this task differs from its parent by the removed event only.
  """
  cfg = agibot_x2_flat_tracking_increated_perturbations_vendor_gains_env_cfg(
    play=play,
    anchor_body_name=anchor_body_name,
    imu_source=imu_source,
  )
  cfg.events.pop("randomize_pd_gains", None)
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


def agibot_x2_multi_motion_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the X2 multi-motion PPO teacher configuration.

  The teacher follows an explicit clip list instead of one reference file. The
  base is
  :func:`agibot_x2_flat_tracking_increated_perturbations_vendor_gains_env_cfg`,
  so the robot, vendor gains, action scale, torso anchor, pelvis IMU, actor
  observations, rewards, and domain randomization stay as they are. Three
  things differ:

  * The motion command becomes the plain-data
    :class:`~mjlab.tasks.tracking.mdp.multi_motion_ppo.MultiMotionPpoCommandCfg`.
    It resolves its YAML manifest into a loaded clip library when the
    environment builds, not in the serializable configuration. Training samples
    each clip's phase uniformly; play uses ``start``. Both keep
    ``standing_start_prob`` at zero, because an episode starts from a perturbed
    reference frame and not from a standing insert.
  * The actor and the critic both receive the existing ``motion_lookahead``
    sample at :data:`_MULTI_MOTION_PREVIEW_S`. The sample carries no reference
    noise, no reference delay, and no clip identity.
  * ``clip_complete`` joins the terminations with ``time_out=False``, so the
    clip endpoint is a reference-task boundary. The physical failures and the
    episode time limit keep their own terms and their own precedence.

  ``scene.num_envs`` defaults to the paper's 4096 environments for training;
  play keeps the base configuration's environment count for the viewer.
  """
  cfg = agibot_x2_flat_tracking_increated_perturbations_vendor_gains_env_cfg(play=play)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_values = {
    field_.name: getattr(motion_cmd, field_.name) for field_ in fields(MotionCommandCfg)
  }
  motion_values.update(
    motion_file="",  # A clip list, not one clip, bounds the mixed rows.
    motion_manifest="",  # Selected by the training or playback entry point.
    sampling_mode="start" if play else "uniform",
    standing_start_prob=0.0,
    lookahead_s=_MULTI_MOTION_PREVIEW_S,
  )
  cfg.commands["motion"] = MultiMotionPpoCommandCfg(**motion_values)

  # The future reference is part of the value estimate too: the return depends
  # on the requested future motion, so the critic reads the same 62-wide
  # sample the actor does. Placed after ``command`` to mirror the actor order.
  critic_group = cfg.observations["critic"]
  critic_terms: dict[str, ObservationTermCfg] = {}
  for name, term in critic_group.terms.items():
    critic_terms[name] = term
    if name == "command":
      critic_terms["motion_lookahead"] = ObservationTermCfg(
        func=motion_lookahead,
        params={"command_name": "motion"},
      )
  critic_group.terms = critic_terms

  # A clip endpoint finishes the reference task for its row; it is not a time
  # limit, and the termination manager reports a simultaneous physical failure
  # through its own term.
  cfg.terminations["clip_complete"] = TerminationTermCfg(
    func=clip_complete, time_out=False
  )

  if not play:
    cfg.scene.num_envs = 4096

  return cfg
