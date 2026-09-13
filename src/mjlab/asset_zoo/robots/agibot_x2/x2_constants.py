"""AgiBot X2 Ultra constants.

Robot: AgiBot Lingxi X2 Ultra, flagship revision (vendor URDF release
``X2_URDF-v1.3.0``). ``xmls/x2_ultra.xml`` is generated from that release by
``scripts/convert_agibot_x2_to_mjcf.py``.

Actuator data comes from AgiBot's PFP joint-module table (peak output torque,
reducer ratio, rotor inertia). The flagship and the newer "X2 Ultra (new
version)" revision share that module family but not their peak torques: the
new version upgraded the ankle-pitch and shoulder motors from 36 Nm to 60 Nm
and the waist from 48 Nm to 36 Nm. We model the flagship, so effort limits are
taken from its URDF while armature and PD gains are derived from the joint
module closest to each joint's torque class (see ``X2_ACTUATOR_*`` below).
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
# module output, which the flagship reaches for PFP-96 only.
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

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0


def _actuator(
  module: dict,
  target_names_expr: tuple[str, ...],
  effort_limit: float,
) -> BuiltinPositionActuatorCfg:
  """Build a position actuator group from a PFP module's reflected inertia."""
  armature = reflected_inertia(module["rotor_inertia"], module["gear_ratio"])
  return BuiltinPositionActuatorCfg(
    target_names_expr=target_names_expr,
    stiffness=armature * NATURAL_FREQ**2,
    damping=2.0 * DAMPING_RATIO * armature * NATURAL_FREQ,
    effort_limit=effort_limit,
    armature=armature,
  )


# Legs and waist yaw use PFP-96; the flagship reaches its 120 Nm peak.
X2_ACTUATOR_HIP_KNEE = _actuator(
  PFP_96,
  (
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_knee_joint",
    "waist_yaw_joint",
  ),
  effort_limit=120.0,
)
# Waist pitch/roll run at 48 Nm in the flagship; the closest module is PFP-74.
X2_ACTUATOR_WAIST = _actuator(
  PFP_74,
  ("waist_pitch_joint", "waist_roll_joint"),
  effort_limit=48.0,
)
# Ankle pitch and shoulder pitch/roll run at 36 Nm, the PFP-59 catalog peak.
X2_ACTUATOR_36NM = _actuator(
  PFP_59,
  (
    ".*_ankle_pitch_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
  ),
  effort_limit=36.0,
)
# Ankle roll, shoulder yaw, elbow and wrist yaw run at 24 Nm, below the
# PFP-59 peak; they keep PFP-59 armature at the flagship's lower limit.
X2_ACTUATOR_24NM = _actuator(
  PFP_59,
  (
    ".*_ankle_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_yaw_joint",
  ),
  effort_limit=24.0,
)
X2_ACTUATOR_WRIST = _actuator(
  PFP_41,
  (".*_wrist_pitch_joint", ".*_wrist_roll_joint"),
  effort_limit=4.8,
)
# The flagship drives head yaw with the stronger module and head pitch with
# the weaker one, the opposite of the new revision's layout.
X2_ACTUATOR_HEAD_YAW = _actuator(PFP_25, ("head_yaw_joint",), effort_limit=2.6)
X2_ACTUATOR_HEAD_PITCH = _actuator(PFP_12, ("head_pitch_joint",), effort_limit=0.6)

##
# Keyframe config.
##

# The vendored xml stands the robot with straight legs (every joint at its zero
# position, pelvis at z=0.68). Knees, elbows, hip roll and shoulder roll would
# then sit on or outside their soft limits, so the keyframe adds a mild stance
# bend with the soles flat on the ground.
STAND_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.6706),
  joint_pos={
    ".*_hip_pitch_joint": -0.182,
    "left_hip_roll_joint": 0.10,
    "right_hip_roll_joint": -0.10,
    ".*_knee_joint": 0.30,
    ".*_ankle_pitch_joint": -0.118,
    "left_shoulder_roll_joint": 0.25,
    "right_shoulder_roll_joint": -0.25,
    ".*_elbow_joint": -0.25,
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
  actuators=(
    X2_ACTUATOR_HIP_KNEE,
    X2_ACTUATOR_WAIST,
    X2_ACTUATOR_36NM,
    X2_ACTUATOR_24NM,
    X2_ACTUATOR_WRIST,
    X2_ACTUATOR_HEAD_YAW,
    X2_ACTUATOR_HEAD_PITCH,
  ),
  soft_joint_pos_limit_factor=0.9,
)


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


X2_ACTION_SCALE: dict[str, float] = {}
for a in X2_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    X2_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_x2_robot_cfg())

  viewer.launch(robot.spec.compile())
