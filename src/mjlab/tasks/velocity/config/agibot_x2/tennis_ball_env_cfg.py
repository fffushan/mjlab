"""AgiBot X2 tennis-ball-on-the-bat velocity environment configuration factory.

Builds on :func:`~mjlab.tasks.velocity.config.agibot_x2.env_cfgs.
agibot_x2_flat_velocity_env_cfg` -- the no-state-estimation velocity task -- and adds
the tennis equipment: the racket assembly ("bat") rigidly attached to the robot's
right wrist, and a free-floating tennis ball spawned on the string bed. Nothing
else about the base task changes: same 31 actuators, same observation terms, same
rewards, same command distribution, same domain randomization.

Two deliberate differences from the base task exist, and both are physical
requirements rather than tuning choices:

*The physics step is 1 ms with 20 substeps per control step (50 Hz).* The string
bed uses a negative ``solref`` -- MuJoCo's direct stiffness/damping form -- fitted
to drop measurements at 1 ms. At mjlab's default 5 ms step the same numbers inject
energy: a 0.30 m drop rebounds to 1.19 m instead of 0.23 m. See
``tests/test_x2_tennis_ball_task.py``, which re-runs the drop test.

*The robot and ball resets are pinned.* The ball's spawn pose is a fixed offset
from the environment origin, computed from the bat-carry keyframe. Sampling a base
yaw or height would move the bat and leave the ball behind, spawning it inside the
bed or in mid-air. The base task's reset jitter is therefore zeroed here; push
disturbances, encoder bias, COM offset, PD-gain scaling and foot friction
randomization are all untouched.

The ball is currently **not observed by the policy and not rewarded**: this task
wires the assets into a trainable environment. Observations and the ball-holding
reward are the next step, and they change the actor input width.

Servo gains are the **vendor reference-policy values**, not mjlab's measured
native-stand profile. Everything else on the robot is unchanged:
``apply_vendor_rl_gains`` only rewrites ``stiffness``/``damping``, and the joint
ranges, effort limits, armature, friction loss and action scale are identical
between the velocity and multi-motion tracking tasks (checked in
``tests/test_x2_tennis_ball_task.py``). The table is imported from the tracking
task rather than copied, so the two tasks cannot drift apart.

The sim-to-real randomization is also the tracking task's: armature, effort
limits (plus the wrist-only residual derating), body inertia, joint damping,
joint friction and foot size, with one shared PD-gain scale per environment.
Those terms are pulled from ``make_tracking_env_cfg()`` and finished with
``apply_x2_tracking_robot_randomization``, so ``MJLAB_DR_AXES`` still selects the
axes at build time. Observation delay is the tracking family's seventh axis and
is **not** adopted, because it changes the actor's input contract.

See ``docs/source/x2_tennis_ball_balance.rst``.
"""

from __future__ import annotations

from copy import deepcopy

from mjlab.asset_zoo.robots.agibot_x2.tennis_constants import (
  RACKET_BODY,
  get_tennis_ball_cfg,
  get_x2_tennis_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
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


def agibot_x2_tennis_ball_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the AgiBot X2 tennis-ball-on-the-bat velocity environment config.

  Args:
    play: If True, applies the base task's play-mode overrides (effectively
      infinite episode, no observation corruption, no push_robot, no
      curriculum, wider command ranges) plus the tennis resets below.

  Returns:
    A ``ManagerBasedRlEnvCfg`` with the tennis equipment in the scene and a
    ``bat_ball_contact`` sensor. The base task's observations, rewards,
    terminations, commands, actions and domain randomization are unchanged.
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

  return cfg


__all__ = [
  "BAT_BALL_CONTACT_SENSOR",
  "BALL_ENTITY",
  "SIM_TO_REAL_EVENT_NAMES",
  "TENNIS_PHYSICS_TIMESTEP",
  "TENNIS_SUBSTEPS",
  "agibot_x2_tennis_ball_env_cfg",
]
