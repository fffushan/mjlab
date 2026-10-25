"""AgiBot X2 tennis-ball juggling velocity environment configuration factory.

The juggling variant of :mod:`~mjlab.tasks.velocity.config.agibot_x2.
tennis_ball_env_cfg`. This module is a **copy** of the carrying task rather than a
wrapper, so the two can be read side by side and diffed by intent. Scene, assets,
1 ms physics, observations, resets, the ``ball_dropped`` termination, the servo
gains and the physics domain randomization are the carrying task's, unchanged. The
ball is meant to **bounce on the plate and return to an apex**, repeatedly:
juggling, not carrying. Version 1 juggles while standing still.

**What differs from the carrying task, and nothing else does.**

*Reward set.* ``ball_height`` is removed: it is flat out to a 100 mm gap, so it
pays for a ball parked on the strings, which is the behaviour juggling must not
reward. Two terms replace it. ``ball_bounce_apex`` pays **once per bounce**, when
the apex of the bounce is at :data:`JUGGLE_APEX_TARGET_M`; a per-step height
reward cannot express a bounce, because a ball parked at the target height would
collect it on every step. ``ball_at_rest`` charges for a ball that has stayed low
and slow on the plate for :data:`JUGGLE_REST_GRACE_S`, which closes the "park it
and stand still" solution the reward set would otherwise leave open.
``ball_centred`` keeps its function and parameters, but its weight drops to
:data:`JUGGLE_CENTRE_WEIGHT`: the ball is in the air for most of the cycle and
cannot be steered there, so the apex payment has to carry the task.
``ball_in_fov`` and ``head_joint_motion`` are unchanged.

*Command.* The twist command is pinned to a standstill band: ``lin_vel_x`` and
``lin_vel_y`` within :data:`JUGGLE_COMMAND_LIN_VEL_MPS`, ``ang_vel_z`` within
:data:`JUGGLE_COMMAND_ANG_VEL_RPS`. Everything else about the command term --
resampling period, standing and heading fractions, heading stiffness -- is the
carrying task's. The band is small rather than exactly zero, so the command still
resamples inside it and the policy has to hold its ground against a moving target
instead of one fixed target.

*No command curriculum.* ``ball_command_ramp`` is not installed, because the ramp
advances through command stages and one narrow band has no stage to advance to.
The base task's ``command_vel`` term is removed here as well: it would write the
base task's wider ranges over the band. With no term left, ``cfg.curriculum`` is
empty in both train and play mode.

*One metric.* ``ball_bounce_count`` is registered with ``reduce="sum"``, so
``Episode_Metrics/ball_bounce_count`` reports **bounces per episode**. That is
this task's acceptance signal: a reward can be collected in unintended ways, a
counted bounce is behaviour. The adapter reads the apex reward term's per-step
flag, and ``RewardManager.compute`` skips terms whose weight is 0, so the apex
weight must stay non-zero or the count goes stale.

The unchanged parts carry the carrying task's reasons, documented in
:mod:`~mjlab.tasks.velocity.config.agibot_x2.tennis_ball_env_cfg`: the 1 ms step is
a string-bed contact-calibration requirement (the same ``solref`` injects energy at
5 ms), the robot and ball resets are pinned so the ball starts on the bed in every
environment, the servo gains are the vendor reference-policy table, and the
physics randomization is the tracking family's. Observations are the actor's
delayed and held ball position (the real 10 Hz perception chain) and the critic's
privileged position and velocity, so the actor width stays 105 and the critic 123.

See ``docs/source/x2_tennis_ball_juggle.rst``.
"""

from __future__ import annotations

from copy import deepcopy

from mjlab.asset_zoo.robots.agibot_x2.tennis_constants import (
  BALL_CENTRE_PLATEAU,
  BALL_CENTRE_SIGMA,
  BALL_DROP_GRACE_S,
  BALL_DROPOUT_HOLD_STEPS,
  BALL_DROPOUT_PROB,
  BALL_FALL_THRESHOLD,
  BALL_FOV_PLATEAU,
  BALL_FOV_SIGMA,
  BALL_FOV_WEIGHT,
  BALL_OBS_DELAY_MAX_STEPS,
  BALL_OBS_DELAY_MIN_STEPS,
  BALL_OBS_UPDATE_PERIOD_STEPS,
  BALL_POSITION_NOISE_M,
  BALL_RADIUS,
  CAMERA_HALF_ANGLE_X,
  CAMERA_HALF_ANGLE_Y,
  CAMERA_MOUNT_QUAT,
  CAMERA_MOUNT_XYZ,
  HEAD_JOINT_NAMES,
  HEAD_MOTION_WEIGHT,
  JUGGLE_APEX_MIN_CLEARANCE_M,
  JUGGLE_APEX_MIN_SPEED_MPS,
  JUGGLE_APEX_SIGMA_M,
  JUGGLE_APEX_TARGET_M,
  JUGGLE_APEX_WEIGHT,
  JUGGLE_CENTRE_WEIGHT,
  JUGGLE_COMMAND_ANG_VEL_RPS,
  JUGGLE_COMMAND_LIN_VEL_MPS,
  JUGGLE_REST_CLEARANCE_M,
  JUGGLE_REST_GRACE_S,
  JUGGLE_REST_SPEED_MPS,
  JUGGLE_REST_WEIGHT,
  RACKET_BODY,
  get_tennis_ball_cfg,
  get_x2_tennis_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.config.agibot_x2.env_cfgs import (
  apply_x2_tracking_robot_randomization,
)
from mjlab.tasks.tracking.config.agibot_x2.vendor_gains import (
  apply_vendor_rl_gains,
)
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.agibot_x2.env_cfgs import (
  add_velocity_foot_sites,
  agibot_x2_flat_velocity_env_cfg,
)
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

TENNIS_PHYSICS_TIMESTEP = 0.001
"""Physics step tied to the string-bed contact calibration. Do not raise."""

TENNIS_SUBSTEPS = 20
"""Substeps per control step: 1 ms x 20 = 50 Hz, as the reference controller."""

BALL_ENTITY = "ball"
BAT_BALL_CONTACT_SENSOR = "bat_ball_contact"

_ZERO_POSE_RANGE = {
  "x": (0.0, 0.0),
  "y": (0.0, 0.0),
  "z": (0.0, 0.0),
  "yaw": (0.0, 0.0),
}

SIM_TO_REAL_EVENT_NAMES = (
  "randomize_inertia",
  "randomize_armature",
  "randomize_effort_limits",
  "randomize_joint_friction",
  "randomize_joint_damping",
  "randomize_foot_size",
)
"""Sim-to-real DR terms copied from the tracking family, in install order.

``randomize_effort_limits_wrist`` is not listed: it is added by
``apply_x2_tracking_robot_randomization``, which must run after
``randomize_effort_limits`` because both write the same actuator fields.

The tracking family's seventh axis, observation delay, is deliberately absent. It
changes the actor's input timing contract, which is a different decision from
randomizing physics.
"""


def _sim_to_real_events() -> dict[str, EventTermCfg]:
  """Copy the tracking family's physics DR terms, definitions and all.

  The terms are taken from ``make_tracking_env_cfg()`` so there is one definition
  of every range and operation. Reading them through that factory also means
  ``MJLAB_DR_AXES`` selects the axes here in the same way it does for the
  tracking tasks; an axis switched off simply does not appear.
  """
  source = make_tracking_env_cfg().events
  copied: dict[str, EventTermCfg] = {}
  for name in SIM_TO_REAL_EVENT_NAMES:
    term = source.get(name)
    if term is None:
      continue
    # The source config is rebuilt per call, but its term objects are shared with
    # whatever else holds that config, so copy before editing.
    copied[name] = deepcopy(term)
  return copied


def _tennis_robot_cfg() -> EntityCfg:
  """Tennis robot with the vendor servo gains and the velocity foot sites.

  Two layers on top of :func:`~mjlab.asset_zoo.robots.agibot_x2.
  tennis_constants.get_x2_tennis_robot_cfg`:

  * ``apply_vendor_rl_gains`` installs the vendor reference-policy kp/kd, which is
    what ``Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion`` runs and what has the
    hardware trials behind it. It rewrites only stiffness/damping, and it
    deep-copies the articulation first, so the module-level ``X2_ARTICULATION``
    shared with the velocity and tracking tasks is not mutated.
  * The base X2 velocity task injects ``left_foot``/``right_foot`` sites in its own
    ``spec_fn``; the tennis spec supplies a different ``spec_fn``, so the sites are
    re-applied here rather than duplicated. The foot height scan and the foot
    clearance/slip rewards both resolve them by name.
  """
  robot_cfg = get_x2_tennis_robot_cfg()
  apply_vendor_rl_gains(robot_cfg)
  tennis_spec_fn = robot_cfg.spec_fn
  robot_cfg.spec_fn = lambda: add_velocity_foot_sites(tennis_spec_fn())
  return robot_cfg


def _make_ball_sensor() -> ContactSensorCfg:
  """Net normal force between the racket (any geom) and the ball.

  ``primary`` is the racket *body*, so this covers the string bed and any other
  racket geom that gains collision in future. The bed carries ``priority=1``, so
  the reported force is the calibrated bed contact, not a mixed pair.
  """
  return ContactSensorCfg(
    name=BAT_BALL_CONTACT_SENSOR,
    primary=ContactMatch(mode="body", pattern=RACKET_BODY, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="ball", entity=BALL_ENTITY),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )


def agibot_x2_tennis_juggle_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the AgiBot X2 tennis-ball juggling velocity environment config.

  Args:
    play: If True, applies the base task's play-mode overrides (effectively
      infinite episode, no observation corruption, no push_robot, no
      curriculum) plus the tennis resets below. The standstill command band and
      the juggling reward set are applied in both modes: the base task's play
      ranges describe a walking policy, which is a different task here.

  Returns:
    A ``ManagerBasedRlEnvCfg`` with the tennis equipment in the scene, the
    bouncing-ball reward set, the bounce-count metric and a
    ``bat_ball_contact`` sensor. The base task's observations, terminations,
    actions and domain randomization are unchanged.
  """
  cfg = agibot_x2_flat_velocity_env_cfg(play=play)

  ##
  # Scene: the bat lives inside the robot entity (rigid attachment), the ball is
  # its own floating entity.
  ##

  cfg.scene.entities = {
    "robot": _tennis_robot_cfg(),
    BALL_ENTITY: get_tennis_ball_cfg(),
  }

  ##
  # Simulation. 1 ms is a contact-calibration requirement, not a preference. The
  # contact and constraint budgets are headroom for the ball's extra pairs, not
  # a measured requirement: no overflow was observed at the base task's 70/500.
  ##

  cfg.sim.mujoco.timestep = TENNIS_PHYSICS_TIMESTEP
  cfg.decimation = TENNIS_SUBSTEPS
  cfg.sim.nconmax = 120
  cfg.sim.njmax = 800

  ##
  # Sensors.
  ##

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (_make_ball_sensor(),)

  ##
  # Sim-to-real randomization. The tracking family's physics axes, plus one shared
  # PD-gain scale per environment instead of a per-joint one. The velocity task's
  # own terms (push, foot friction, encoder bias, base COM) are untouched, and
  # ``apply_x2_tracking_robot_randomization`` points the new terms at the X2's
  # foot spheres, torso COM and torque classes.
  ##

  cfg.events.update(_sim_to_real_events())
  if "randomize_pd_gains" in cfg.events:
    cfg.events["randomize_pd_gains"].params["shared_gain_scale"] = True
  apply_x2_tracking_robot_randomization(cfg)

  ##
  # Resets. Both entities are pinned so the ball starts on the bed in every
  # environment. Everything else in the base reset sequence is preserved.
  ##

  reset_base = cfg.events["reset_base"]
  reset_base.params["pose_range"] = dict(_ZERO_POSE_RANGE)

  ball_cfg = SceneEntityCfg(BALL_ENTITY)
  cfg.events["reset_ball"] = EventTermCfg(
    mode="reset",
    func=mdp.reset_root_state_uniform,
    params={
      "asset_cfg": ball_cfg,
      "pose_range": dict(_ZERO_POSE_RANGE),
      # Empty: the ball starts at rest on the bed. A non-empty range would
      # immediately tip it off, which is a separate scenario (ball reception),
      # not this one.
      "velocity_range": {},
    },
  )

  ##
  # Ball observation. The policy cannot balance a ball it cannot see, so both
  # groups get the ball's position relative to the plate, and the critic alone also
  # gets its velocity. The critic reads both terms **without** corruption, like the
  # rest of its privileged set.
  #
  # Corruption is the perception budget, not physics: on hardware the ball
  # position comes from an external perception program. Both terms are in the
  # plate frame, so a ball carried along with the bat reads zero velocity. Each
  # dimension of each term is randomized independently; see the constants file for
  # the magnitudes and how they were derived.
  ##

  plate_cfg = SceneEntityCfg("robot", body_names=(RACKET_BODY,))
  ball_state_params = {"plate_cfg": plate_cfg, "ball_cfg": ball_cfg}

  # Six independent randomization magnitudes, one per dimension. ``Unoise`` takes
  # per-dimension bounds (`NoiseParam = float | tuple[float, ...]`), so one term
  # can be randomized differently along the plate's shaft, width and normal.
  # Each entry is a half-width: the distribution is symmetric uniform.
  #
  # The actor's position is also delayed, because the real camera publishes at
  # 10 Hz and a published frame is 110 ms old at the median and 141 ms at P95.
  # At the 50 Hz control rate that is a lag of 5 to 9 steps, refreshed once per
  # sensor frame. The critic is deliberately left clean: it is privileged and may
  # read the simulator's ball state directly.
  #
  # The camera also rejects frames, and the stream can break outright, so the actor
  # additionally sees the last valid position repeated while no new measurement
  # arrives. The hold is applied first, then the existing noise and delay are
  # unchanged; the shape is still (num_envs, 3), so the actor width does not change.
  cfg.observations["actor"].terms["ball_pos"] = ObservationTermCfg(
    func=mdp.ball_position_hold,
    params=dict(
      ball_state_params,
      rejection_prob=BALL_DROPOUT_PROB,
      hold_steps=BALL_DROPOUT_HOLD_STEPS,
    ),
    noise=Unoise(
      n_min=tuple(-m for m in BALL_POSITION_NOISE_M),
      n_max=BALL_POSITION_NOISE_M,
    ),
    delay_min_lag=BALL_OBS_DELAY_MIN_STEPS,
    delay_max_lag=BALL_OBS_DELAY_MAX_STEPS,
    delay_update_period=BALL_OBS_UPDATE_PERIOD_STEPS,
    delay_per_env=True,
    delay_per_env_phase=True,
  )
  # No ball velocity for the actor. Position is the only quantity the perception
  # chain provides on the real robot: the best available velocity is a finite
  # difference of a 10 Hz noisy position, which is unusable. Velocity is therefore
  # critic-only privileged information.
  cfg.observations["critic"].terms["ball_pos"] = ObservationTermCfg(
    func=mdp.ball_position_in_plate,
    params=dict(ball_state_params),
  )
  cfg.observations["critic"].terms["ball_vel"] = ObservationTermCfg(
    func=mdp.ball_velocity_in_plate,
    params=dict(ball_state_params),
  )

  ##
  # Juggling reward set, and the fatal drop.
  #
  # ``ball_height`` is deliberately absent. It is flat out to a 100 mm gap and would
  # pay for a ball parked on the strings, which is the behaviour this task has to
  # rule out. ``ball_bounce_apex`` pays once per bounce at the target apex instead,
  # and ``ball_at_rest`` charges for parking. Centring stays -- the ball has to come
  # back to the plate, not merely be hit -- but at a quarter of the carrying weight,
  # because the ball is in the air for most of the cycle and cannot be steered there.
  # The two ball SceneEntityCfgs are shared: all four ball terms measure the same
  # plate frame and the same ball.
  ##

  cfg.rewards["ball_centred"] = RewardTermCfg(
    func=mdp.ball_centred,
    weight=JUGGLE_CENTRE_WEIGHT,
    params={
      "plateau": BALL_CENTRE_PLATEAU,
      "sigma": BALL_CENTRE_SIGMA,
      **ball_state_params,
    },
  )
  cfg.rewards["ball_bounce_apex"] = RewardTermCfg(
    func=mdp.ball_bounce_apex,
    weight=JUGGLE_APEX_WEIGHT,
    params={
      "target_height": JUGGLE_APEX_TARGET_M,
      "sigma": JUGGLE_APEX_SIGMA_M,
      "min_clearance": JUGGLE_APEX_MIN_CLEARANCE_M,
      "min_speed": JUGGLE_APEX_MIN_SPEED_MPS,
      **ball_state_params,
    },
  )
  cfg.rewards["ball_at_rest"] = RewardTermCfg(
    func=mdp.ball_at_rest,
    weight=JUGGLE_REST_WEIGHT,
    params={
      "max_clearance": JUGGLE_REST_CLEARANCE_M,
      "max_speed": JUGGLE_REST_SPEED_MPS,
      "grace_s": JUGGLE_REST_GRACE_S,
      **ball_state_params,
    },
  )

  # Grace, not a weaker threshold: the ball is lost at the same height as before,
  # but the episode only ends once it has stayed lost. Without this an untrained
  # policy gets 0.4 s episodes out of a 20 s allowance.
  step_dt = cfg.sim.mujoco.timestep * cfg.decimation
  grace_steps = round(BALL_DROP_GRACE_S / step_dt)
  # The head camera, as a soft constraint: the ball has to stay where perception can
  # see it. Geometry is approximate on purpose, see the constants file.
  head_cfg = SceneEntityCfg("robot", body_names=("head_pitch_link",))
  cfg.rewards["ball_in_fov"] = RewardTermCfg(
    func=mdp.ball_in_camera_fov,
    weight=BALL_FOV_WEIGHT,
    params={
      "plateau": BALL_FOV_PLATEAU,
      "sigma": BALL_FOV_SIGMA,
      "mount_xyz": CAMERA_MOUNT_XYZ,
      "mount_quat": CAMERA_MOUNT_QUAT,
      "half_angle_x": CAMERA_HALF_ANGLE_X,
      "half_angle_y": CAMERA_HALF_ANGLE_Y,
      "head_cfg": head_cfg,
      "ball_cfg": ball_cfg,
    },
  )
  # And the head must not hunt. The field-of-view term would otherwise pay for
  # dithering the two head joints.
  cfg.rewards["head_joint_motion"] = RewardTermCfg(
    func=mdp.head_joint_motion,
    weight=HEAD_MOTION_WEIGHT,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=HEAD_JOINT_NAMES),
    },
  )

  cfg.terminations["ball_dropped"] = TerminationTermCfg(
    func=mdp.ball_lost,
    params={
      "threshold": BALL_FALL_THRESHOLD,
      "ball_radius": BALL_RADIUS,
      "grace_steps": grace_steps,
      **ball_state_params,
    },
  )

  ##
  # Acceptance metric: bounces per episode.
  #
  # ``reduce="sum"`` accumulates the per-step value over the episode, so a term that
  # is 1.0 on a counted apex and 0.0 otherwise becomes a bounce count. The adapter
  # reads the apex reward term's state, so the apex weight must stay non-zero:
  # ``RewardManager.compute`` skips terms whose weight is 0 and the flag would then go
  # stale, freezing the count at its last value.
  ##

  cfg.metrics["ball_bounce_count"] = MetricsTermCfg(
    func=mdp.ball_bounce_count,
    reduce="sum",
  )

  ##
  # Command: a standstill band, and no curriculum.
  #
  # Version 1 juggles while standing still, so the twist command is pinned to a narrow
  # band around zero. There is no ramp because one band has no stage to advance to.
  #
  # The base task's ``command_vel`` term must still be removed: it writes the base
  # task's wider ranges to this same command term, which would undo the pin. With no
  # term left the curriculum is empty in both train and play mode.
  #
  # The band is written in play mode too. The base task's play overrides widen the
  # linear and yaw ranges for evaluating a walking policy, which is a different task
  # here.
  ##

  cfg.curriculum.pop("command_vel", None)
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (-JUGGLE_COMMAND_LIN_VEL_MPS, JUGGLE_COMMAND_LIN_VEL_MPS)
  twist_cmd.ranges.lin_vel_y = (-JUGGLE_COMMAND_LIN_VEL_MPS, JUGGLE_COMMAND_LIN_VEL_MPS)
  twist_cmd.ranges.ang_vel_z = (-JUGGLE_COMMAND_ANG_VEL_RPS, JUGGLE_COMMAND_ANG_VEL_RPS)

  return cfg


__all__ = [
  "BAT_BALL_CONTACT_SENSOR",
  "BALL_ENTITY",
  "SIM_TO_REAL_EVENT_NAMES",
  "TENNIS_PHYSICS_TIMESTEP",
  "TENNIS_SUBSTEPS",
  "agibot_x2_tennis_juggle_env_cfg",
]
