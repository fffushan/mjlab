"""AgiBot X2 Ultra constants.

Robot: AgiBot Lingxi X2 Ultra, flagship revision (vendor URDF release
``X2_URDF-v1.3.0``). ``xmls/x2_ultra.xml`` is generated from that release by
``scripts/convert_agibot_x2_to_mjcf.py``.

Actuator data comes from AgiBot's PFP joint-module table (peak output torque,
reducer ratio, rotor inertia). The flagship and the newer "X2 Ultra (new
version)" revision share that module family but not their peak torques: the
new version upgraded the ankle-pitch and shoulder motors from 36 Nm to 60 Nm
and the waist from 48 Nm to 36 Nm. We model the flagship, so effort limits are
taken from its URDF and the joint armature is the module's reflected rotor
inertia.

The nominal PD gains are *measured*, not derived: they are the gains the real
robot commands in its native stable stand, read off the
``/aima/hal/joint/*/command`` messages of two independent hardware captures.
Two groups keep the reflected-inertia 10 Hz template because no measurement pins
them. ``docs/source/x2_gain_provenance.md`` documents the evidence, the other
gain tables in circulation, and every number chosen here.

Effort limits are the *flagship* revision's (X2_URDF-v1.3.0 / x2_ultra.xml) joint
ratings, byte for byte. The newer revision is a different joint set -- waist 48 to
36 N.m, wrist 4.8 to 6, ankle pitch and the shoulders 36 to 60, the former 24 N.m
group to 36 -- so its numbers must not be mixed in here.
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import reflected_inertia
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

X2_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "agibot_x2" / "xmls" / "x2_ultra.xml"
)
assert X2_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(X2_XML))


##
# Actuator config.
##

# AgiBot's PFP joint modules. Rotor inertias are quoted in kg mm^2 for the
# PFP-41/59/74/96 modules and in g cm^2 for PFP-25/12; both are converted to
# SI here. ``peak_torque``/``peak_velocity`` are the catalog ratings at the
# module output, which the flagship reaches for PFP-96 only. Only the rotor
# inertia and gear ratio are used below, for the joint armature; the PD gains
# no longer come from this table.
PFP_96 = dict(
  rotor_inertia=90.19890664e-6, gear_ratio=21.9, peak_torque=120.0, peak_rpm=136.0
)
PFP_74 = dict(
  rotor_inertia=30.5020502e-6, gear_ratio=20.03, peak_torque=60.0, peak_rpm=240.0
)
PFP_59 = dict(
  rotor_inertia=10.13746228e-6, gear_ratio=22.1, peak_torque=36.0, peak_rpm=170.0
)
PFP_41 = dict(
  rotor_inertia=1.3590987e-6, gear_ratio=24.4, peak_torque=6.0, peak_rpm=200.0
)
PFP_25 = dict(rotor_inertia=5.1e-7, gear_ratio=27.56, peak_torque=2.0, peak_rpm=150.0)
PFP_12 = dict(rotor_inertia=1.76e-8, gear_ratio=107.2, peak_torque=0.5, peak_rpm=300.0)

# The 10 Hz / damping-ratio 2 closed loop on the reflected rotor inertia that
# the gains were originally built from: Kp = armature * NATURAL_FREQ**2 and
# Kd = 2 * DAMPING_RATIO * armature * NATURAL_FREQ. Kept for the two groups
# marked `template` below, where the measurement cannot pin a nominal value.
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

# Reflected rotor inertia of the waist yaw joint, as AgiBot's own Isaac actuator
# configuration carries it. That source does not separate rotor inertia from gear
# ratio, so it is given as a value rather than as a PFP module; PFP_96 is left in
# the table for the hip and knee, where its row is still the right module.
WAIST_YAW_ARMATURE = 0.010177520


def _template_gains(module: dict) -> tuple[float, float]:
  """The 10 Hz / zeta = 2 loop on a module's reflected rotor inertia."""
  armature = reflected_inertia(module["rotor_inertia"], module["gear_ratio"])
  return (
    armature * NATURAL_FREQ**2,
    2.0 * DAMPING_RATIO * armature * NATURAL_FREQ,
  )


def _actuator(
  module: dict | None,
  target_names_expr: tuple[str, ...],
  effort_limit: float,
  stiffness: float,
  damping: float,
  frictionloss: float,
  *,
  armature: float | None = None,
) -> BuiltinPositionActuatorCfg:
  """Build one position actuator group.

  ``stiffness``/``damping`` are the nominal servo gains (see the table below),
  ``frictionloss`` the dry friction of that joint group. The armature is the
  module's reflected rotor inertia and is a physical property of the
  transmission, independent of the gains.

  Every group carries a command-channel delay of 0-2 physics steps (0-10 ms at
  the 5 ms training timestep): the deployment sends position targets over ROS 2
  at 50 Hz, so the torque the servo loop applies is derived from a target that is
  one or two steps stale by the time it arrives. The decision period already
  contributes a 0-20 ms hold, so this is the transport and firmware part on top.
  """
  if armature is None:
    assert module is not None, "pass a PFP module or an explicit armature"
    armature = reflected_inertia(module["rotor_inertia"], module["gear_ratio"])
  else:
    assert module is None, "pass a PFP module or an explicit armature, not both"
  return BuiltinPositionActuatorCfg(
    target_names_expr=target_names_expr,
    stiffness=stiffness,
    damping=damping,
    effort_limit=effort_limit,
    armature=armature,
    frictionloss=frictionloss,
    delay_min_lag=0,
    delay_max_lag=2,
    delay_hold_prob=0.9,  # Latency is mostly stable, with occasional jitter.
    delay_update_period=1,
  )


# One group per joint role, because the gains are no longer uniform across a
# torque class: the measured hardware gains differ per joint and left/right are
# symmetric. Armature still follows the PFP module of each joint's torque class.
#
#   gains        measured native stable-stand profile of the real robot, from
#                the recorded `/aima/hal/joint/*/command` messages of captures
#                20260915T130441Z (firmware v1.1.0.95) and 20260915T141139Z
#                (v1.1.4). Reproduced across both, so it is what the robot
#                actually runs, but note these are *commanded* gains: the
#                achieved motor-side loop is still unverified.
#   frictionloss dry friction by joint size. The vendored XML sets a blanket
#                0.3 N.m for all 31 joints, which is 50% of head_pitch's 0.6 N.m
#                effort limit; these are order-of-magnitude values from the
#                measured torque residual in the same captures, scaled down
#                with the joint's torque class.
#
# ``X2_ACTUATOR_GROUPS`` is ordered: ``actuator_group_index`` and
# ``SceneEntityCfg(actuator_ids=...)`` depend on it.
X2_ACTUATOR_GROUPS: dict[str, BuiltinPositionActuatorCfg] = {
  # Legs: the measured stand gains, 100/4 hip and 150/5 knee.
  "hip_pitch": _actuator(PFP_96, (".*_hip_pitch_joint",), 120.0, 100.0, 4.0, 0.30),
  "hip_roll": _actuator(PFP_96, (".*_hip_roll_joint",), 120.0, 100.0, 3.0, 0.30),
  "hip_yaw": _actuator(PFP_96, (".*_hip_yaw_joint",), 120.0, 100.0, 3.0, 0.30),
  "knee": _actuator(PFP_96, (".*_knee_joint",), 120.0, 150.0, 5.0, 0.30),
  # Waist yaw used to keep the template value because the vendor's own two tables
  # disagree by 3.75x on its gain (40/8 standing, 150/3 posture hold). The
  # measured 40/8 is now corroborated: the vendor's sample policy ships
  # 40.1792/2.5579 for this joint, which is exactly the template applied to the
  # armature in AgiBot's own Isaac actuator config. So the measurement wins, and
  # that armature replaces the PFP-96 torque-class inference -- the waist yaw
  # module is 120 N.m in both URDF revisions, so this joint's armature is the
  # same in each and it was the PFP-96 assignment that was wrong.
  "waist_yaw": _actuator(
    None, ("waist_yaw_joint",), 120.0, 40.0, 8.0, 0.15, armature=WAIST_YAW_ARMATURE
  ),
  # Waist pitch/roll: the same disagreement (40/5 vs 300/3); the template value
  # sits between them and is 21% above the measured stand value.
  "waist_pitch_roll": _actuator(
    PFP_74,
    ("waist_pitch_joint", "waist_roll_joint"),
    48.0,
    *_template_gains(PFP_74),
    0.15,
  ),
  # Ankles. The measured stand gains are 2x stiffer than the template on ankle
  # pitch, which is what made the simulated ankle more compliant than the real
  # one.
  "ankle_pitch": _actuator(PFP_59, (".*_ankle_pitch_joint",), 36.0, 40.0, 3.0, 0.30),
  "ankle_roll": _actuator(PFP_59, (".*_ankle_roll_joint",), 24.0, 30.0, 2.0, 0.30),
  # Arms.
  "shoulder_pitch": _actuator(
    PFP_59, (".*_shoulder_pitch_joint",), 36.0, 30.0, 1.0, 0.12
  ),
  "shoulder_roll": _actuator(
    PFP_59, (".*_shoulder_roll_joint",), 36.0, 20.0, 1.0, 0.12
  ),
  "shoulder_yaw": _actuator(PFP_59, (".*_shoulder_yaw_joint",), 24.0, 20.0, 1.0, 0.08),
  "elbow": _actuator(PFP_59, (".*_elbow_joint",), 24.0, 50.0, 1.0, 0.08),
  "wrist_yaw": _actuator(PFP_59, (".*_wrist_yaw_joint",), 24.0, 50.0, 1.0, 0.08),
  "wrist_pitch_roll": _actuator(
    PFP_41, (".*_wrist_pitch_joint", ".*_wrist_roll_joint"), 4.8, 20.0, 1.0, 0.04
  ),
  # Head. The flagship drives head yaw with the stronger module and head pitch
  # with the weaker one, the opposite of the new revision's layout.
  "head_yaw": _actuator(PFP_25, ("head_yaw_joint",), 2.6, 3.4, 0.114, 0.02),
  # The measured head pitch gain (10/0.8) contradicts this joint's 0.6 N.m effort
  # limit: it would saturate at 0.06 rad of error, and the capture shows a 0.2 rad
  # tracking error at ~0 N.m of torque, so the limit, the torque estimate or the
  # gain is wrong. Keep the torque-consistent template value until a hardware
  # torque measurement settles it.
  "head_pitch": _actuator(
    PFP_12, ("head_pitch_joint",), 0.6, *_template_gains(PFP_12), 0.01
  ),
}


##
# Keyframe config.
##

# The vendored xml stands the robot with straight legs (every joint at its zero
# position, pelvis at z=0.68). Knees, elbows, hip roll and shoulder roll would
# then sit on or outside their soft limits, and the arms would hang straight.
# The keyframe is instead the pose the real robot is measured standing in: the
# commanded native stable-stand profile of captures 20260915T130441Z /
# 20260915T141139Z, symmetrized (left/right hip and knee differ by up to 0.05
# rad there because it is a dynamic equilibrium, not a rest pose) and with
# ankle_pitch = -(hip_pitch + knee) so the soles stay flat. The pelvis height
# puts the lowest sole sphere exactly on z = 0.
STAND_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.66003),
  joint_pos={
    ".*_hip_pitch_joint": -0.19,
    "left_hip_roll_joint": 0.045,
    "right_hip_roll_joint": -0.045,
    ".*_knee_joint": 0.457,
    ".*_ankle_pitch_joint": -0.267,  # -(hip_pitch + knee).
    ".*_shoulder_pitch_joint": 0.4,
    # The robot holds shoulder roll at ~0 (both vendor tables command 0), but
    # the shoulder_roll joint's range starts at -0.061, so the 0.9 soft limit
    # sits at 0.092 and a straight-arm keyframe would start episodes at a limit
    # penalty.
    "left_shoulder_roll_joint": 0.25,
    "right_shoulder_roll_joint": -0.25,
    ".*_elbow_joint": -1.2,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# Feet are the 12 collision spheres per foot in the ankle-roll body; everything
# else is a mesh collision geom. Foot contacts get condim=3 and a friction
# range, self-collisions condim=1.
_FOOT_GEOMS = r"^(left|right)_foot[0-9]+_collision$"

# The vendored xml gives every body a mesh collision geom, and MuJoCo collides
# their convex hulls. The pelvis and head hulls overlap their neighbours in
# most poses of a reference motion (head pitch 1-5 mm deep in every frame, hip
# roll over 5 mm in a third of them), which would preload the neck and hips and
# feed a constant self-collision penalty. Those three geoms stay out of
# collision; limb, torso and foot hulls are kept.
_EXCLUDED_HULLS = "pelvis|head_yaw_link|head_pitch_link"
_BODY_GEOMS = rf"^(?!(?:{_EXCLUDED_HULLS})_collision$).*_collision$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(_BODY_GEOMS,),
  contype=1,
  conaffinity=1,
  condim={_FOOT_GEOMS: 3, ".*_collision": 1},
  priority={_FOOT_GEOMS: 1, ".*": 0},
  friction={_FOOT_GEOMS: (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(_BODY_GEOMS,),
  contype=0,
  conaffinity=1,
  condim={_FOOT_GEOMS: 3, ".*_collision": 1},
  priority={_FOOT_GEOMS: 1, ".*": 0},
  friction={_FOOT_GEOMS: (0.6,)},
)

# Disables all collisions except the feet.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_FOOT_GEOMS,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

X2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=tuple(X2_ACTUATOR_GROUPS.values()),
  soft_joint_pos_limit_factor=0.9,
)


def actuator_group_index(key: str) -> int:
  """Index of a named group in ``X2_ARTICULATION.actuators``.

  Use this instead of a hard-coded position: the group layout changes whenever a
  joint's gains stop matching its neighbours, and ``SceneEntityCfg(actuator_ids=``
  ``...)`` needs the index, not the name.
  """
  keys = list(X2_ACTUATOR_GROUPS)
  if key not in X2_ACTUATOR_GROUPS:
    raise KeyError(f"unknown X2 actuator group {key!r}; known: {keys}")
  return keys.index(key)


def get_x2_robot_cfg() -> EntityCfg:
  """Get a fresh X2 Ultra robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=STAND_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=X2_ARTICULATION,
  )


# Action scale per group, in radians of position target per unit of action. It is
# decoupled from the gains on purpose: deriving it as 0.25 * effort / Kp silently
# pinned the gain of every low-torque joint, because keeping a sane action scale
# on the wrist meant keeping its Kp 6x below what the robot runs.
#
# The scale is the larger of two floors:
#
#   torque floor    0.25 * effort_limit / stiffness. A unit action stays inside
#                   25% of the torque budget, so the joint does not saturate.
#   tracking floor  _TRACKING_ACTION_SCALE[key]. The reference motion's p95
#                   deviation from the keyframe divided by 5, so the motion stays
#                   reachable inside a 5-unit action. Measured against
#                   data/qianghuo_smplx_agibot_x2_tracking.npz; revisit it if the
#                   keyframe or the reference motion changes
#                   (tests/test_x2_constants.py checks it while that file exists).
#
# The tracking floor wins on six groups, where a unit action consequently reaches
# 27% (knee) to 60% (wrist pitch/roll) of the torque budget: knee 32.6%, shoulder
# pitch 27.2%, shoulder yaw 28.7%, elbow 50.2%, wrist yaw 32.5%, wrist pitch/roll
# 60.4%. Everywhere else it is the torque floor and exactly 25%. That is the
# honest place to be: those six joints really are torque-limited at their
# reference amplitudes, and a real joint saturates there too.
_TRACKING_ACTION_SCALE: dict[str, float] = {
  "knee": 0.261,  # Reference bends the knee 1.30 rad.
  "shoulder_pitch": 0.326,  # 1.63 rad.
  "shoulder_yaw": 0.345,  # 1.72 rad.
  "elbow": 0.241,  # 1.20 rad.
  "wrist_yaw": 0.156,  # 0.78 rad.
  "wrist_pitch_roll": 0.145,  # 0.72 rad.
}

X2_ACTION_SCALE: dict[str, float] = {}
for _key, _group in X2_ACTUATOR_GROUPS.items():
  _effort = _group.effort_limit
  assert _effort is not None
  _scale = max(
    0.25 * _effort / _group.stiffness,
    _TRACKING_ACTION_SCALE.get(_key, 0.0),
  )
  for _pattern in _group.target_names_expr:
    X2_ACTION_SCALE[_pattern] = _scale


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_x2_robot_cfg())

  viewer.launch(robot.spec.compile())
