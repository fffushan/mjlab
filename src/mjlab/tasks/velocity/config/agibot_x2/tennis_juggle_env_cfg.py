"""AgiBot X2 tennis-ball juggling velocity environment configuration factory.

The juggling variant of :mod:`~mjlab.tasks.velocity.config.agibot_x2.
tennis_ball_env_cfg`. This module is a **copy** of the carrying task rather than a
wrapper, so the two can be read side by side and diffed by intent. Scene, assets,
1 ms physics, observations, the servo gains and the physics domain randomization are
the carrying task's, unchanged. The ball is meant to **bounce on the plate and
return to an apex**, repeatedly: juggling, not carrying. Version 1 juggles while
standing still.

**What differs from the carrying task, and nothing else does.**

*Reward set.* ``ball_height`` is removed: it is flat out to a 100 mm gap, so it
pays for a ball parked on the strings, which is the behaviour juggling must not
reward. A dense ``ball_energy`` term replaces it. The term compares the ball's
vertical energy, with the datum at the ball's fixed world rest height, against the
energy of a bounce to :data:`JUGGLE_ENERGY_TARGET_HEIGHT_M`, and pays every control
step::

    h   = z_ball - JUGGLE_ENERGY_BALL_REST_Z
    E   = 0.5 * v_ball_z**2 + g * h,  E* = g * target_height
    r_E = exp(-((E - E*) / sigma)**2)

The racket is the control input and must not pay itself, so no plate velocity is
subtracted from the ball's: a descending racket would otherwise manufacture energy
with the ball untouched. Both datums are **fixed world heights**, not the pelvis:
with a pelvis datum, lowering the pelvis 0.20 m while the ball stays put raises
``h`` by 0.20 m and adds ``g * 0.20 = 1.962`` J/kg, half of the 3.924 J/kg target,
with no work done on the ball. That was free energy, and the lift term could not see
it because the plate and the pelvis dropped together. A world datum costs gait bob
in ``E`` (3 cm is 0.29 J/kg, 0.19 sigma), which is the smaller error. The absolute
heights are valid only because the tennis tasks pin the base reset at the carrying
task's measured base height with zero yaw; see
:data:`JUGGLE_ENERGY_BALL_REST_Z`.

A ball parked on the plate has ``h = 0`` and ``E ~ 0``, so parking earns almost
nothing. **Why this replaced the per-bounce apex bell:** measured on the
4000-iteration sweep, the bell was worth about 2e-5 of per-step reward at the
apexes the policies actually reached (0.16-0.24 m against a 0.30 m target with an
80 mm sigma) - flat, and therefore no gradient - and it paid on one step per bounce
instead of on every step.

A second dense term, ``racket_lift``, is installed with **negative** weight
:data:`JUGGLE_RACKET_LIFT_WEIGHT`. ``h`` is measured from the ball's resting state
on the plate, so a ball resting on a racket raised to the 0.40 m target reads
``E*``; the lift term charges the plate top's world height above its carry height
(world, not pelvis-relative, so a lift by the legs is charged like a lift by the
arm). Its weight follows the **maximum-slope rule**: it is at least as steep as the
energy Gaussian's steepest slope in lift, 11.2196 per metre at sigma 1.5, so no
partial lift can farm the pair. The old rule that cancelled the reward at the
target is not sufficient under the wider sigma. ``ball_centred`` keeps its function
and parameters at :data:`JUGGLE_CENTRE_WEIGHT`; ``ball_in_fov`` and
``head_joint_motion`` are unchanged.

The spawn is **not** the target. :data:`BALL_SPAWN_DROP_HEIGHT` is still 0.20 m, so
the ball starts 0.20 m below the 0.40 m target. At spawn ``E = 1.962`` J/kg against
``E* = 3.924`` J/kg. At the original sigma of 0.5 J/kg the energy reward there was
about 2.1e-7, i.e. no usable signal. The operator widened
:data:`JUGGLE_ENERGY_SIGMA_J_PER_KG` to 1.5 J/kg, so the spawn scores about 0.1807
and the 0.20 -> 0.40 m climb has a gradient. The passive bounce peak of about
0.21 m scores about 0.21.

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

*Eight ball metrics.* ``ball_energy`` and ``ball_energy_error`` are registered with
``reduce="mean"``, so ``Episode_Metrics/ball_energy`` reports the episode mean of
``E`` (target ``E*``) and ``Episode_Metrics/ball_energy_error`` the episode mean of
``|E - E*|`` in J/kg. Both are recomputed from the current simulation state, so
they cannot read a stale value: unlike the removed ``ball_bounce_count``, which read
a reward term's per-step flag and would freeze if that term were not computed, there
is no term state to go stale.

``ball_start_height_m``, ``ball_start_plate`` and ``ball_first_lift_s`` are the
stage-1 release-state metrics, also **logging only**. The first is the ball's height
above its resting state at reset, 0 for a plate start and the sampled drop release
height otherwise. The second is 1.0 for a plate start and 0.0 for a drop start. The
third is the first control time at which the ball energy reaches
``JUGGLE_LIFT_ENERGY_THRESHOLD_J_PER_KG``, or -1.0 if the episode ends first; a drop
start reads 0.0 because its release energy is already above the threshold. All three
use ``reduce="last"``: the value is the episode's release fact, not a per-step
average.

``ball_impact_count``, ``ball_flight_time`` and ``ball_sustained_contact`` are read
from the ``bat_ball_contact`` sensor and are **logging only**. They do not enter a
reward, an observation, a termination or a curriculum. They are the only evidence
that can separate juggling from a passive bounce. A racket-ball contact lasts about
9-10 ms against a 20 ms control step, and the sensor's ``found`` field is one
instantaneous snapshot per step, so an instantaneous metric misses about half the
contacts. The sensor therefore keeps one control step of substeps in
``force_history`` (``history_length`` = ``cfg.decimation``), and the metrics derive
contact from it. The first counts rising edges of the per-step contact flag with
``reduce="sum"``; a contact inside one step counts once, and a continuous contact
spanning several steps also counts once. The second reports a running mean airborne
time between impacts in seconds with ``reduce="last"``, accumulated in substeps. The
third reports the longest continuous contact seen in the control step, in seconds,
with ``reduce="max"``.

The unchanged parts carry the carrying task's reasons, documented in
:mod:`~mjlab.tasks.velocity.config.agibot_x2.tennis_ball_env_cfg`: the 1 ms step is
a string-bed contact-calibration requirement (the same ``solref`` injects energy at
5 ms), the robot base reset is pinned so the measured world heights stay valid, the
servo gains are the vendor reference-policy table, and the physics randomization is
the tracking family's. The robot joint reset keeps the generic zero-offset term and
adds a right-arm/waist perturbation; the ball reset samples a plate or drop release
instead of a fixed drop. Observations are the actor's delayed and held ball position
(the real 10 Hz perception chain) and the critic's privileged position and velocity,
so the actor width stays 105 and the critic 123.

See ``docs/source/x2_tennis_ball_juggle.rst``.
"""

from __future__ import annotations

import os
from copy import deepcopy

from mjlab.asset_zoo.robots.agibot_x2.x2_constants import X2_ACTION_SCALE
from mjlab.envs.mdp.actions import JointPositionActionCfg

from mjlab.asset_zoo.robots.agibot_x2.tennis_constants import (
  BALL_CENTRE_PLATEAU,
  BALL_CENTRE_SIGMA,
  BALL_POSITION_BIAS_M,
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
  BALL_SPAWN_DROP_HEIGHT,
  CAMERA_HALF_ANGLE_X,
  CAMERA_HALF_ANGLE_Y,
  CAMERA_MOUNT_QUAT,
  CAMERA_MOUNT_XYZ,
  FACE_SITE,
  HEAD_JOINT_NAMES,
  HEAD_MOTION_WEIGHT,
  JUGGLE_HEAD_PITCH_RANGE_RAD,
  JUGGLE_HEAD_YAW_RANGE_RAD,
  JUGGLE_ARM_RESET_RANGE_RAD,
  JUGGLE_CENTRE_WEIGHT,
  JUGGLE_COMMAND_ANG_VEL_RPS,
  JUGGLE_COMMAND_LIN_VEL_MPS,
  JUGGLE_ENERGY_BALL_REST_Z,
  JUGGLE_ENERGY_SIGMA_J_PER_KG,
  JUGGLE_ENERGY_TARGET_HEIGHT_M,
  JUGGLE_ENERGY_WEIGHT,
  JUGGLE_LIFT_ENERGY_THRESHOLD_J_PER_KG,
  JUGGLE_NO_LIFT_TIMEOUT_S,
  JUGGLE_RACKET_LIFT_DEADBAND_M,
  JUGGLE_RACKET_LIFT_PLATE_TOP_REST_Z,
  JUGGLE_RACKET_LIFT_WEIGHT,
  JUGGLE_RESET_JOINT_NAMES,
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

_ACTION_PATTERN_WITHOUT_HEAD = r"^(?!head_yaw_joint$|head_pitch_joint$).*$"
"""Actuator pattern for the 29-DOF family: every joint except the two head axes.

``resolve_matching_names`` applies ``re.fullmatch`` per key, so this negative
lookahead matches the 29 names that are not exactly a head joint. The excluded
joints stay in the model, stay in the observation and stay actuated by their own PD;
they simply stop being something the policy can command.
"""

_ABLATION_ENV_PREFIX = "MJLAB_JUGGLE_"


def _ablation_value(name: str, default: float | None = None) -> float | None:
  """Read one per-arm reward-weight override from the environment.

  The sweep arms differ only in reward weights, and a run reads this module once,
  so without this one code tree could launch one arm only. An unset variable
  returns *default*, so the shipped configuration is unchanged and cannot drift
  from the constants in ``tennis_constants``.

  Only the three reward weights and the four ball-perception magnitudes are
  readable here. Nothing in the observation **layout**, the action, the
  termination or the physics path may be overridden this way: the actor's width and
  every non-perception input have to stay identical across arms or the comparison
  means nothing. A perception magnitude is a distribution parameter, not a layout
  parameter, so ablating it is the point of an arm and the shape it feeds is
  unchanged.
  """
  raw = os.environ.get(_ABLATION_ENV_PREFIX + name)
  if raw in (None, ""):
    return default
  return float(raw)


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


def _make_ball_sensor(history_length: int) -> ContactSensorCfg:
  """Net normal force between the racket (any geom) and the ball.

  ``primary`` is the racket *body*, so this covers the string bed and any other
  racket geom that gains collision in future. The bed carries ``priority=1``, so
  the reported force is the calibrated bed contact, not a mixed pair.

  Args:
    history_length: Physics substeps kept in ``force_history``. The juggling
      task passes its decimation, so the buffer covers exactly one control step;
      ``0`` disables the buffer.
  """
  return ContactSensorCfg(
    name=BAT_BALL_CONTACT_SENSOR,
    primary=ContactMatch(mode="body", pattern=RACKET_BODY, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="ball", entity=BALL_ENTITY),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=history_length,
  )


def agibot_x2_tennis_juggle_env_cfg(
  play: bool = False,
  head_actuated: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Create the AgiBot X2 tennis-ball juggling velocity environment config.

  Args:
    play: If True, applies the base task's play-mode overrides (effectively
      infinite episode, no observation corruption, no push_robot, no
      curriculum) plus the tennis resets below. The standstill command band and
      the juggling reward set are applied in both modes: the base task's play
      ranges describe a walking policy, which is a different task here.
    head_actuated: If False, the two head joints leave the action space, giving a
      29-DOF policy. They stay in the ``joint_pos``/``joint_vel`` observations
      because the camera rides on the head, so its measured pose is needed for the
      camera-to-plate conversion. Their pose is then randomized per episode instead
      of being commanded, which is what lets one policy cover robots that park the
      head differently. ``head_joint_motion`` is dropped in that case: it would
      charge the policy for motion it cannot cause.

  Returns:
    A ``ManagerBasedRlEnvCfg`` with the tennis equipment in the scene, the
    dense ball-energy reward set, the energy and contact metrics and a
    ``bat_ball_contact`` sensor. The base task's observations, terminations,
    actions and domain randomization are unchanged.
  """
  cfg = agibot_x2_flat_velocity_env_cfg(play=play)

  if not head_actuated:
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.actuator_names = _ACTION_PATTERN_WITHOUT_HEAD
    # The scale map is keyed by joint-name pattern and every key must match at least
    # one target, so the two head entries have to go with the joints. Leaving them in
    # raises "Not all regular expressions are matched" at build time, which is the
    # framework refusing a scale for a joint that is not being commanded.
    joint_pos_action.scale = {
      pattern: value
      for pattern, value in X2_ACTION_SCALE.items()
      if pattern not in HEAD_JOINT_NAMES
    }

  ##
  # Scene: the bat lives inside the robot entity (rigid attachment), the ball is
  # its own floating entity.
  ##

  # Dropped in, not placed: a bounce exists from the first control step, so the
  # policy has something to sustain instead of having to create a bounce from a
  # ball at rest. Measured at 4000 iterations, the sweep policies never initiated
  # a bounce on their own.
  cfg.scene.entities = {
    "robot": _tennis_robot_cfg(),
    BALL_ENTITY: get_tennis_ball_cfg(BALL_SPAWN_DROP_HEIGHT),
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
  #
  # One control step of substeps in the rolling buffer. The racket-ball contact
  # lasts about 9-10 ms while the control step is 20 ms, so the sensor's single
  # instantaneous ``found`` sample misses about half the contacts. ``decimation``
  # is the control step over the sim step (0.02 s / 0.001 s = 20), which is
  # exactly the window ``ball_contact_metric`` reads. Only the juggling task
  # enables it; the carrying task's sensor keeps the instantaneous field.
  ##

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (_make_ball_sensor(cfg.decimation),)

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
  # Resets. The base is pinned, so the measured world heights and the ball's drop
  # point stay valid. The ball draws one of two release states per environment:
  # 20 % plate start, 80 % drop start. The generic joint reset stays at zero; a
  # second term perturbs only the right arm and the waist. Both robot joint terms
  # are inserted before `reset_ball`: the plate release reads the post-reset plate
  # pose, so it must see the perturbed joint state.
  ##

  reset_base = cfg.events["reset_base"]
  reset_base.params["pose_range"] = dict(_ZERO_POSE_RANGE)

  # Right arm and waist only, +-JUGGLE_ARM_RESET_RANGE_RAD. ``reset_robot_joints``
  # above writes the default pose to every joint with a zero offset; this term is
  # appended after it, so its sampled offset survives. Legs, the left arm and the
  # head are deliberately absent: perturbing them would move the base, the energy
  # datum or the camera rather than the strike geometry.
  # The head cannot be commanded in the 29-DOF family, so its pose is drawn per
  # episode and held by its own PD for the whole episode. Installed before the ball
  # resets so any forward-kinematics read inside them already sees the final pose.
  if not head_actuated:
    cfg.events["reset_head_pose"] = EventTermCfg(
      mode="reset",
      func=mdp.reset_head_pose,
      params={
        "position_range": (
          -max(JUGGLE_HEAD_YAW_RANGE_RAD, JUGGLE_HEAD_PITCH_RANGE_RAD),
          max(JUGGLE_HEAD_YAW_RANGE_RAD, JUGGLE_HEAD_PITCH_RANGE_RAD),
        ),
        "asset_cfg": SceneEntityCfg("robot", joint_names=HEAD_JOINT_NAMES),
      },
    )

  cfg.events["reset_robot_juggle_joints"] = EventTermCfg(
    mode="reset",
    func=mdp.reset_joints_by_offset,
    params={
      "position_range": (-JUGGLE_ARM_RESET_RANGE_RAD, JUGGLE_ARM_RESET_RANGE_RAD),
      "velocity_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=JUGGLE_RESET_JOINT_NAMES),
    },
  )

  # ``plate_cfg`` resolves the plate's top face; the event reads its live pose
  # after a forward-kinematics refresh to seat a plate start on the perturbed
  # plate. It must stay after both robot joint resets.
  ball_cfg = SceneEntityCfg(BALL_ENTITY)
  cfg.events["reset_ball"] = EventTermCfg(
    mode="reset",
    func=mdp.reset_ball_release,
    params={
      "asset_cfg": ball_cfg,
      "plate_cfg": SceneEntityCfg("robot", site_names=(FACE_SITE,)),
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
  # Perception ablation arms. The defaults reproduce the shipped chain exactly; an
  # arm overrides one magnitude at a time through MJLAB_JUGGLE_BALL_*.
  #   BALL_DELAY_MIN_STEPS / MAX_STEPS / PERIOD_STEPS: latency model, in control steps.
  #     18.44 Hz was measured on the real estimator against the 10 Hz camera the
  #     shipped numbers model, so the "rate18" arm uses period 3 and keeps the lag in
  #     the same physical range (100-140 ms = 5-7 steps).
  #   BALL_BIAS_M: constant per-episode offset half-width, for the hand-eye bias.
  ball_delay_min = int(
    _ablation_value("BALL_DELAY_MIN_STEPS", BALL_OBS_DELAY_MIN_STEPS)
  )
  ball_delay_max = int(
    _ablation_value("BALL_DELAY_MAX_STEPS", BALL_OBS_DELAY_MAX_STEPS)
  )
  ball_delay_period = int(
    _ablation_value("BALL_DELAY_PERIOD_STEPS", BALL_OBS_UPDATE_PERIOD_STEPS)
  )
  ball_bias_half_m = float(_ablation_value("BALL_BIAS_M", BALL_POSITION_BIAS_M) or 0.0)
  cfg.observations["actor"].terms["ball_pos"] = ObservationTermCfg(
    func=mdp.ball_position_hold,
    params=dict(
      ball_state_params,
      rejection_prob=BALL_DROPOUT_PROB,
      hold_steps=BALL_DROPOUT_HOLD_STEPS,
      bias_half_width_m=ball_bias_half_m,
    ),
    noise=Unoise(
      n_min=tuple(-m for m in BALL_POSITION_NOISE_M),
      n_max=BALL_POSITION_NOISE_M,
    ),
    delay_min_lag=ball_delay_min,
    delay_max_lag=ball_delay_max,
    delay_update_period=ball_delay_period,
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
  # rule out. ``ball_energy`` pays for the ball's energy relative to its resting
  # state on the plate, on every step instead; a parked ball has h ~ 0 and so earns
  # almost nothing. Both heights are world z, not pelvis-relative: a crouch must not
  # move the energy datum. Solving the one remaining park -- holding the resting ball
  # at the 0.40 m target lift, where h = 0.40 reads E* -- is ``racket_lift``'s job,
  # with a negative weight, and it is world-relative so the legs cannot do the lift
  # for free either. Centring stays -- the ball has to come back to the plate, not
  # merely be hit -- at a quarter of the carrying weight. The two ball
  # SceneEntityCfgs are shared: all the ball terms measure the same plate frame and
  # the same ball.
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
  # Sweep arms vary the two weights below. The lift weight is linear in the energy
  # weight by construction (see ``juggle_racket_lift_weight``), so an arm that
  # changes the energy weight must scale the lift weight with it; otherwise the arm
  # silently reopens the parking farm it was derived to close.
  energy_override = _ablation_value("ENERGY_WEIGHT")
  energy_weight = JUGGLE_ENERGY_WEIGHT if energy_override is None else energy_override
  lift_override = _ablation_value("LIFT_WEIGHT")
  lift_weight = (
    JUGGLE_RACKET_LIFT_WEIGHT * energy_weight / JUGGLE_ENERGY_WEIGHT
    if lift_override is None
    else lift_override
  )
  cfg.rewards["ball_energy"] = RewardTermCfg(
    func=mdp.ball_energy,
    weight=energy_weight,
    params={
      "target_height": JUGGLE_ENERGY_TARGET_HEIGHT_M,
      "sigma": JUGGLE_ENERGY_SIGMA_J_PER_KG,
      "ball_rest_z": JUGGLE_ENERGY_BALL_REST_Z,
      **ball_state_params,
    },
  )
  # Lifting the plate above its carry height and parking a ball on it reads
  # h = lift in world, so a lift by the legs is charged like a lift by the arm.
  # Negative weight: the term returns metres of excess lift.
  cfg.rewards["racket_lift"] = RewardTermCfg(
    func=mdp.racket_lift,
    weight=-lift_weight,
    params={
      "plate_top_rest_z": JUGGLE_RACKET_LIFT_PLATE_TOP_REST_Z,
      "deadband": JUGGLE_RACKET_LIFT_DEADBAND_M,
      "plate_cfg": plate_cfg,
    },
  )

  # A standing robot with a zero command already earns the ceiling of both tracking
  # terms, 0.040 per step each, while a perfect juggle earns 0.040 from
  # ``ball_energy``. The objective is therefore worth half of doing nothing, which
  # is exactly what one sweep arm tests. Unset leaves the base value alone.
  track_override = _ablation_value("TRACK_WEIGHT")
  if track_override is not None:
    for term in ("track_linear_velocity", "track_angular_velocity"):
      if term in cfg.rewards:
        cfg.rewards[term].weight = track_override

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
  if head_actuated:
    cfg.rewards["head_joint_motion"] = RewardTermCfg(
      func=mdp.head_joint_motion,
      weight=HEAD_MOTION_WEIGHT,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=HEAD_JOINT_NAMES),
      },
    )
  # With the head outside the action space the term is dropped rather than zeroed:
  # the joint follows a fixed target, so any motion it reports is PD tracking of the
  # sampled pose, which the policy cannot influence in either direction.

  cfg.terminations["ball_dropped"] = TerminationTermCfg(
    func=mdp.ball_lost,
    params={
      "threshold": BALL_FALL_THRESHOLD,
      "ball_radius": BALL_RADIUS,
      "grace_steps": grace_steps,
      **ball_state_params,
    },
  )

  # A plate start in which the policy never lifts the ball has no bounce to
  # sustain and nothing else ends it before the 20 s horizon, so it spends the
  # episode on a state that teaches nothing. The term fires only after the
  # control-time timeout AND only if the peak energy since reset stayed below the
  # lift threshold; a drop start has E = g * h >= 1.4715 J/kg at release, so it
  # is inert for drop episodes by construction.
  cfg.terminations["ball_not_lifted"] = TerminationTermCfg(
    func=mdp.ball_not_lifted,
    params={
      "threshold": JUGGLE_LIFT_ENERGY_THRESHOLD_J_PER_KG,
      "timeout_s": JUGGLE_NO_LIFT_TIMEOUT_S,
      "ball_cfg": ball_cfg,
      "ball_rest_z": JUGGLE_ENERGY_BALL_REST_Z,
    },
  )

  ##
  # Metrics: two energy means, three release-state facts and three contact
  # metrics. All are logging only: they change no reward, observation, termination
  # or curriculum.
  #
  # ``reduce="mean"`` reports the mean of a per-step value over the episode. Both
  # energy functions recompute E from the current simulation state, so neither can
  # read a stale reward-term flag. The old ``ball_bounce_count`` was removed with the
  # apex term it read: ``RewardManager.compute`` skips zero-weight terms, so a
  # counter that reads another term's state freezes when that term stops being
  # computed.
  #
  # The contact metrics read the ``bat_ball_contact`` sensor, which was already in
  # the scene and unused. The energy metrics still start below the target: the ball
  # spawns 0.20 m above rest against a 0.40 m target, so at reset they read about
  # 1.962 J/kg and 1.962 J/kg of error, not 3.924 and 0.
  ##

  cfg.metrics["ball_energy"] = MetricsTermCfg(
    func=mdp.ball_energy_value,
    params={
      "target_height": JUGGLE_ENERGY_TARGET_HEIGHT_M,
      "ball_rest_z": JUGGLE_ENERGY_BALL_REST_Z,
      **ball_state_params,
    },
    reduce="mean",
  )
  cfg.metrics["ball_energy_error"] = MetricsTermCfg(
    func=mdp.ball_energy_error,
    params={
      "target_height": JUGGLE_ENERGY_TARGET_HEIGHT_M,
      "ball_rest_z": JUGGLE_ENERGY_BALL_REST_Z,
      **ball_state_params,
    },
    reduce="mean",
  )

  # Release-state metrics: logging only, like the energy and contact metrics.
  # Each value is constant or first-crossing per episode, so ``reduce="last"``
  # reports the episode value instead of diluting it over the steps. The start
  # height and start type are recorded by ``reset_ball`` when it samples the
  # episode, so they report the sampled release and not a value reconstructed
  # from a perturbed plate pose. The first-lift time uses the same energy
  # threshold as ``ball_not_lifted``.
  cfg.metrics["ball_start_height_m"] = MetricsTermCfg(
    func=mdp.ball_start_height,
    params={"release_event": "reset_ball"},
    reduce="last",
  )
  cfg.metrics["ball_start_plate"] = MetricsTermCfg(
    func=mdp.ball_start_plate,
    params={"release_event": "reset_ball"},
    reduce="last",
  )
  cfg.metrics["ball_first_lift_s"] = MetricsTermCfg(
    func=mdp.ball_first_lift_time,
    params={
      "threshold": JUGGLE_LIFT_ENERGY_THRESHOLD_J_PER_KG,
      "ball_cfg": ball_cfg,
      "ball_rest_z": JUGGLE_ENERGY_BALL_REST_Z,
    },
    reduce="last",
  )

  # Contact metrics: logging only. The sensor's history buffer holds one control
  # step of substeps, so these resolve a 9-10 ms contact that a once-per-step
  # ``found`` sample would miss. ``impact_count`` sums rising edges of the per-step
  # contact flag, so a continuous contact counts once but an in-step contact counts
  # at all. ``flight_time`` reports the running mean airborne time between impacts;
  # ``reduce="last"`` keeps the final mean because the manager's ``mean`` would
  # divide by every step, not by the number of flights. ``sustained_contact``
  # reports the longest continuous contact in the step, and ``reduce="max"`` keeps
  # the longest one of the episode.
  cfg.metrics["ball_impact_count"] = MetricsTermCfg(
    func=mdp.ball_contact_metric,
    params={"sensor_name": BAT_BALL_CONTACT_SENSOR, "mode": "impact_count"},
    reduce="sum",
  )
  cfg.metrics["ball_flight_time"] = MetricsTermCfg(
    func=mdp.ball_contact_metric,
    params={"sensor_name": BAT_BALL_CONTACT_SENSOR, "mode": "flight_time"},
    reduce="last",
  )
  cfg.metrics["ball_sustained_contact"] = MetricsTermCfg(
    func=mdp.ball_contact_metric,
    params={"sensor_name": BAT_BALL_CONTACT_SENSOR, "mode": "sustained_contact"},
    reduce="max",
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
