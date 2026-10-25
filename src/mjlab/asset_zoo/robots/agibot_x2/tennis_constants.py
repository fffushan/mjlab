"""X2 Ultra + tennis racket assembly ("bat") and tennis ball.

The racket is **rigid tooling on the robot**, not a separate scene entity: the
white wrist connector replaces the right hand, and the clamp plus racket hang
off ``right_wrist_roll_link`` as fixed bodies. Rigid attachment is modelled by
the kinematic tree, so the bat's 0.37 kg adds to the arm's reflected inertia
through ``mj_step`` with no weld equality and no soft constraint. The ball is a
separate floating entity (see :func:`get_tennis_ball_spec`).

Provenance: the geometry, masses and contact parameters are copied from the
X2 tennis asset snapshot in ``x2_references/x2_tennis`` (source commit
``a0239f49ccf81e511e46bcbf8b5104b0c7478562``, branch ``codex/tennis-keep-up``).
The four STL files under ``tennis_meshes/`` are byte-identical to that snapshot
(SHA-256 pinned in :data:`TENNIS_MESH_SHA256`). The numbers themselves are
geometric estimates, not measurements: the white connector mass is an estimate
and the clamp/racket inertia come from CAD geometry. Only the ball rebound was
fitted, to user drop measurements (see :data:`BED_SOLREF`).

Two properties of this file are load-bearing and easy to break:

* The string bed is named ``string_bed``, **not** ``*_collision``. The shipped
  :data:`~.x2_constants.FULL_COLLISION` disables every geom that its
  ``geom_names_expr`` does not match, so the bed would otherwise be switched
  off, and a ``*_collision`` name would route it through the ``condim=1``,
  ``priority=0`` catch-all. Either mistake silently replaces the calibrated
  contact with the default one.
* :data:`BED_SOLREF` is a *negative* ``solref``, i.e. MuJoCo's direct
  stiffness/damping parameterisation. It was fitted at a 1 ms physics step.
  ``tests/test_x2_tennis_ball_task.py`` re-runs the drop test and shows that at
  mjlab's default 5 ms step the same parameters inject energy (a 0.3 m drop
  rebounds to 1.19 m). The tennis task therefore runs at 1 ms.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco
import numpy as np

from mjlab import MJLAB_SRC_PATH
from mjlab.asset_zoo.robots.agibot_x2.x2_constants import (
  FULL_COLLISION,
  X2_ARTICULATION,
  get_spec,
)
from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# Paths.
##

TENNIS_MESH_DIR: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "agibot_x2" / "tennis_meshes"
)
assert TENNIS_MESH_DIR.is_dir()

TENNIS_MESH_SHA256: dict[str, str] = {
  "clamp_a.stl": "10ddc321eed1cad60f080fedb0216c76f4b571b4fb3f43d0fe0109ab294e7da6",
  "clamp_b.stl": "65f8a4d4f2dda56e439fd38f65471cceb535dbe70521a1d6b32f469ec6d7de7d",
  "racket_visual.stl": (
    "ecf7a4f52abb144a77d629656b98ff20e3faa112cc000fae8c14b53a086bf935"
  ),
  "right_hand_connector.stl": (
    "48a56bcec4c68406e9299d1632035381231ff6f5621d1f8eba5233b5cebbb7f2"
  ),
}


def mesh_sha256(name: str) -> str:
  """SHA-256 of a shipped tennis mesh, so a test can pin provenance."""
  return hashlib.sha256((TENNIS_MESH_DIR / name).read_bytes()).hexdigest()


##
# Names. Referenced by contact sensors, rewards and tests.
##

CONNECTOR_GEOM = "tennis_white_connector_visual"
CLAMP_A_GEOM = "racket_clamp_a_visual"
CLAMP_B_GEOM = "racket_clamp_b_visual"
RACKET_GEOM = "tennis_racket_visual"
BED_GEOM = "string_bed"
FACE_SITE = "face_center"

CLAMP_BODY = "racket_clamp"
RACKET_BODY = "tennis_racket"
WRIST_BODY = "right_wrist_roll_link"

##
# Racket assembly geometry and inertia.
#
# Transforms are from ``x2_references/x2_tennis/model/scene.xml`` (the exported
# training scene), which is the compiled form of the assembly URDF. They are
# *not* recomputed from the URDF here: the URDF pass through the MJCF converter
# permutes the racket's principal axes (URDF ixx/iyy/izz 0.00168/0.00933/0.01027
# becomes diaginertia 0.01027/0.00933/0.00168 plus a 90 deg quat), and the
# exported scene is what the reference played back.
##

# The white connector *is* the right wrist's distal link: it replaces the hand,
# so the wrist body's own inertia is replaced rather than extended.
CONNECTOR_MASS = 0.08
CONNECTOR_IPOS = (0.0, 0.0, -0.02)
CONNECTOR_DIAGINERTIA = (4e-05, 4e-05, 4e-05)

CLAMP_POS = (-0.0143531, -0.0252336, -0.0375)
CLAMP_QUAT = (0.0, 0.707107, 0.707107, 0.0)  # rpy 180, 0, 90 deg.
CLAMP_MASS = 0.09
CLAMP_IPOS = (0.0252336, 0.0143531, 0.03)
CLAMP_DIAGINERTIA = (6e-05, 6e-05, 6e-05)

RACKET_POS = (0.0252336, -0.0756469, 0.05)
RACKET_QUAT = (0.5, -0.5, -0.5, 0.5)
RACKET_MASS = 0.28
RACKET_IPOS = (0.4, 0.0, 0.0)
RACKET_QUAT_INERTIAL = (0.0, 0.707107, 0.0, 0.707107)
RACKET_DIAGINERTIA = (0.0102667, 0.00933333, 0.00168)

# The clamp meshes are exported in millimetres; the connector and the racket in
# metres. ``x2_references`` encodes this as a mesh ``scale`` attribute.
CLAMP_MESH_SCALE = (0.001, 0.001, 0.001)

# String bed: a 32-sided, 3 mm thick elliptical plate. Deliberately NOT the
# convex hull of the racket, which would be a solid slab ~1 cm thick.
BED_CENTER_X = 0.515
BED_HALF_LENGTH = 0.157  # Local x (along the shaft).
BED_HALF_WIDTH = 0.117  # Local y.
BED_HALF_THICKNESS = 0.0015
BED_SEGMENTS = 32
FACE_SITE_POS = (BED_CENTER_X, 0.0, BED_HALF_THICKNESS)

##
# Ball.
##

BALL_MASS = 0.057
BALL_RADIUS = 0.0335
BALL_FRICTION = (0.6, 0.005, 0.0001)
# Positive solref here (normalised time constant / damping ratio): the ball is
# the *softer* of the pair, but ``priority`` below makes the bed's parameters
# win on every bed contact, so this only governs ball-vs-robot/ground contacts.
BALL_SOLREF = (0.004, 0.35)
BALL_RGBA = (0.75, 0.95, 0.08, 1.0)

##
# String-bed contact. Fitted to the user's drop measurements (rebound 0.17, 0.23,
# 0.31, 0.42 m for drops of 0.20, 0.30, 0.40, 0.50 m). Negative solref is
# MuJoCo's direct stiffness/damping form, not physical N/m or N*s/m, and it is
# only valid near the 1 ms step it was fitted at.
##

BED_FRICTION = (0.6, 0.005, 0.0001)
BED_SOLREF = (-100000.0, -22.0)
BED_SOLIMP = (0.95, 0.95, 0.001, 0.5, 2.0)
BED_CONDIM = 3
BED_PRIORITY = 1

##
# Stable contact. The bed must survive the shipped collision policy, which
# zeroes ``contype``/``conaffinity`` of every geom its patterns do not match.
#
# This config is applied *after* ``FULL_COLLISION`` and therefore repairs what
# that pass does to the bed. It sets ``disable_other_geoms=False`` so it cannot
# undo the rest of the robot's collision story. Fields are scalars because the
# pattern matches exactly one geom.
##

TENNIS_BED_COLLISION = CollisionCfg(
  geom_names_expr=(rf"^{BED_GEOM}$",),
  contype=1,
  conaffinity=1,
  condim=BED_CONDIM,
  priority=BED_PRIORITY,
  friction=BED_FRICTION,
  solref=BED_SOLREF,
  solimp=BED_SOLIMP,
  disable_other_geoms=False,
)


##
# Spec builders.
##


def _add_mesh(
  spec: mujoco.MjSpec,
  name: str,
  file: str,
  scale: tuple[float, float, float] | None = None,
) -> None:
  """Register a mesh from ``tennis_meshes/`` under an explicit name."""
  path = str(TENNIS_MESH_DIR / file)
  if scale is None:
    spec.add_mesh(name=name, file=path)
  else:
    spec.add_mesh(name=name, file=path, scale=list(scale))


def _set_explicit_inertial(
  body: mujoco.MjsBody,
  *,
  mass: float,
  ipos: tuple[float, float, float],
  iquat: tuple[float, float, float, float],
  inertia: tuple[float, float, float],
  full_form: bool = False,
) -> None:
  """Write a body's inertial element explicitly, in SI units.

  Explicit inertia is required: without it MuJoCo would derive the bat's mass
  from the visual meshes (which carry ``density=0``) and the certified numbers
  would silently become the visuals'. ``MjsBody``'s inertial attributes are data
  descriptors with custom setters, so the type checker rejects plain assignment
  even though the bindings accept it.

  ``full_form`` is for bodies whose source XML already holds this inertia as a
  six-element ``fullinertia``: MuJoCo refuses a model that carries both forms,
  and zeroing the unused one does not unset it.
  """
  body.mass = mass
  body.ipos = list(ipos)  # type: ignore[invalid-assignment]
  body.iquat = list(iquat)  # type: ignore[invalid-assignment]
  if full_form:
    body.fullinertia = [*inertia, 0.0, 0.0, 0.0]  # type: ignore[invalid-assignment]
  else:
    body.inertia = list(inertia)  # type: ignore[invalid-assignment]
  body.explicitinertial = True


def attach_tennis_assembly(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Replace the X2's right hand with the connector + clamp + racket.

  Mutates and returns ``spec`` so it can be used inline in a ``spec_fn``.
  ``fusestatic`` is disabled because the bat bodies carry no joints: with
  fusion on, MuJoCo would fold them into ``right_wrist_roll_link`` and the
  named bodies, the ``face_center`` site and the bed geom's parent would stop
  existing as separate names.
  """
  spec.compiler.fusestatic = False

  wrist = spec.body(WRIST_BODY)

  # Remove the hand outright: geometry and inertia both belong to the connector
  # now. Keeping the hand's 0.30 kg beside the connector's 0.08 kg would double
  # count the wrist mass.
  for geom in tuple(wrist.geoms):
    spec.delete(geom)
  _set_explicit_inertial(
    wrist,
    mass=CONNECTOR_MASS,
    ipos=CONNECTOR_IPOS,
    iquat=(1.0, 0.0, 0.0, 0.0),
    inertia=CONNECTOR_DIAGINERTIA,
    full_form=True,
  )

  _add_mesh(spec, "tennis_white_connector", "right_hand_connector.stl")
  wrist.add_geom(
    name=CONNECTOR_GEOM,
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="tennis_white_connector",
    contype=0,
    conaffinity=0,
    group=1,
    rgba=[0.7, 0.72, 0.75, 1.0],
  )

  clamp = wrist.add_body(name=CLAMP_BODY, pos=list(CLAMP_POS), quat=list(CLAMP_QUAT))
  _set_explicit_inertial(
    clamp,
    mass=CLAMP_MASS,
    ipos=CLAMP_IPOS,
    iquat=(1.0, 0.0, 0.0, 0.0),
    inertia=CLAMP_DIAGINERTIA,
  )

  # The clamp STLs are exported in millimetres, so the scale lives on the mesh
  # element (as the reference's MJCF does); ``add_geom`` has no scale argument.
  _add_mesh(spec, "tennis_assembly_clamp_a", "clamp_a.stl", CLAMP_MESH_SCALE)
  _add_mesh(spec, "tennis_assembly_clamp_b", "clamp_b.stl", CLAMP_MESH_SCALE)
  for geom_name, mesh_name, rgba in (
    (CLAMP_A_GEOM, "tennis_assembly_clamp_a", (0.95, 0.55, 0.12, 1.0)),
    (CLAMP_B_GEOM, "tennis_assembly_clamp_b", (0.15, 0.45, 0.65, 1.0)),
  ):
    clamp.add_geom(
      name=geom_name,
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=mesh_name,
      contype=0,
      conaffinity=0,
      group=1,
      density=0.0,
      rgba=list(rgba),
    )

  racket = clamp.add_body(
    name=RACKET_BODY, pos=list(RACKET_POS), quat=list(RACKET_QUAT)
  )
  _set_explicit_inertial(
    racket,
    mass=RACKET_MASS,
    ipos=RACKET_IPOS,
    iquat=RACKET_QUAT_INERTIAL,
    inertia=RACKET_DIAGINERTIA,
  )

  _add_mesh(spec, "tennis_assembly_racket_visual", "racket_visual.stl")
  racket.add_geom(
    name=RACKET_GEOM,
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="tennis_assembly_racket_visual",
    contype=0,
    conaffinity=0,
    group=1,
    density=0.0,
    rgba=[0.2, 0.4, 0.8, 1.0],
  )

  # Convex body for the bed. ``mass=0`` keeps every kilogram accounted for by
  # the body's explicit inertial.
  angle = np.linspace(0.0, 2.0 * np.pi, BED_SEGMENTS, endpoint=False)
  vertices = np.concatenate(
    [
      np.c_[
        BED_CENTER_X + BED_HALF_LENGTH * np.cos(angle),
        BED_HALF_WIDTH * np.sin(angle),
        np.full(BED_SEGMENTS, z),
      ]
      for z in (-BED_HALF_THICKNESS, BED_HALF_THICKNESS)
    ]
  )
  spec.add_mesh(name="string_bed_mesh", uservert=vertices.ravel().tolist())
  racket.add_geom(
    name=BED_GEOM,
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="string_bed_mesh",
    contype=1,
    conaffinity=1,
    condim=BED_CONDIM,
    priority=BED_PRIORITY,
    mass=0.0,
    group=3,
    friction=list(BED_FRICTION),
    solref=list(BED_SOLREF),
    solimp=list(BED_SOLIMP),
    rgba=[0.8, 0.8, 0.8, 0.15],
  )

  racket.add_site(name=FACE_SITE, pos=list(FACE_SITE_POS), size=[0.005] * 3)
  return spec


def get_tennis_spec() -> mujoco.MjSpec:
  """X2 Ultra spec with the tennis racket assembly attached."""
  return attach_tennis_assembly(get_spec())


def get_tennis_ball_spec() -> mujoco.MjSpec:
  """Free-floating tennis ball.

  ``condim=3`` with the bed's ``priority=1`` means the bed's friction and
  solver parameters govern bed contacts; this geom's own values only apply to
  ball-vs-robot and ball-vs-ground pairs.
  """
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="ball")
  body.add_freejoint(name="ball_free")
  body.add_geom(
    name="ball_collision",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=[BALL_RADIUS, 0.0, 0.0],
    mass=BALL_MASS,
    contype=1,
    conaffinity=1,
    condim=3,
    friction=list(BALL_FRICTION),
    solref=list(BALL_SOLREF),
    rgba=list(BALL_RGBA),
  )
  return spec


##
# Bat-carry pose and the ball's spawn offset.
#
# ``BAT_CARRY_POSE`` is the reference's approved ''raised prepare'' keep-up
# pose, read from ``x2_references/x2_tennis/parameters/keep_up_pose.json``. The
# lower body differs from ``STAND_KEYFRAME`` (hip -0.15 / knee 0.30 / ankle
# -0.15 instead of -0.19 / 0.457 / -0.267), so ``BAT_CARRY_BASE_HEIGHT`` and
# ``BALL_SPAWN_LOCAL_POS`` are *not* the standing task's numbers. Both are
# recomputed and asserted by ``tests/test_x2_tennis_ball_task.py`` via
# :func:`measure_bat_carry_state`, so a spec change cannot leave them stale.
##

BAT_CARRY_POSE: dict[str, float] = {
  "left_hip_pitch_joint": -0.15,
  "left_hip_roll_joint": 0.0,
  "left_hip_yaw_joint": 0.0,
  "left_knee_joint": 0.3,
  "left_ankle_pitch_joint": -0.15,
  "left_ankle_roll_joint": 0.0,
  "right_hip_pitch_joint": -0.15,
  "right_hip_roll_joint": 0.0,
  "right_hip_yaw_joint": 0.0,
  "right_knee_joint": 0.3,
  "right_ankle_pitch_joint": -0.15,
  "right_ankle_roll_joint": 0.0,
  "waist_yaw_joint": 0.0,
  "waist_roll_joint": 0.0,
  "waist_pitch_joint": 0.0,
  "left_shoulder_pitch_joint": 0.0,
  "left_shoulder_roll_joint": 0.0,
  "left_shoulder_yaw_joint": 0.0,
  "left_elbow_joint": -0.35,
  "left_wrist_yaw_joint": 0.0,
  "left_wrist_pitch_joint": 0.0,
  "left_wrist_roll_joint": 0.0,
  "right_shoulder_pitch_joint": -1.0181433860990814,
  "right_shoulder_roll_joint": -0.38188969469963957,
  "right_shoulder_yaw_joint": -0.46367365027320445,
  "right_elbow_joint": -1.0509471627810472,
  "right_wrist_yaw_joint": 1.5071232960137622,
  "right_wrist_pitch_joint": -0.1323629921327199,
  "right_wrist_roll_joint": 0.646498393112228,
  "head_yaw_joint": 0.27599528646882693,
  "head_pitch_joint": 0.061799387799999994,
}

# Bat carry pose, pelvis root height (m). Measured with the shipped spec by
# :func:`measure_bat_carry_state`; reproduces the reference's recorded
# ``keep_up_pose.json`` face-centre position to 1e-16 m.
BAT_CARRY_BASE_HEIGHT = 0.6698410651336125

# The bat-carry pose is lifted so the lowest sole point starts this far above
# z=0. The reference did the same, and it keeps the first physics step from
# resolving a deep foot-ground penetration.
BAT_CARRY_GROUND_CLEARANCE = 0.002

# Ball centre in the environment frame when the robot is in ``BAT_CARRY_POSE``
# at ``BAT_CARRY_BASE_HEIGHT`` with zero yaw: the face centre plus the face
# normal times (radius + 1 mm). Measured.
BALL_SPAWN_LOCAL_POS = (
  0.5600012287607482,
  -0.024057310128549662,
  1.15388480278694,
)

# Clearance between the ball's lowest point and the bed's top surface at spawn.
BALL_SPAWN_CLEARANCE = 0.001

# Lowest foot contact point may be this far below z=0 before the pose counts as
# penetrating the ground.
_FOOT_LOW_Z_TOLERANCE = 1e-6


def measure_bat_carry_state(
  base_height: float | None = None,
) -> tuple[float, np.ndarray]:
  """Forward-kinematics probe for the bat-carry pose.

  Returns ``(base_height, ball_centre)``: the pelvis height at which the lowest
  foot contact point sits :data:`BAT_CARRY_GROUND_CLEARANCE` above z=0, and the
  ball centre that rests on the face centre along the face normal with
  :data:`BALL_SPAWN_CLEARANCE` of slack.

  ``base_height`` is measured, not assumed, when it is ``None``: the leg is
  rigid at fixed joint angles, so the lowest sole point translates with the
  root height exactly and one probe suffices.
  """
  spec = get_tennis_spec()
  model = spec.compile()
  data = mujoco.MjData(model)

  def place(height: float) -> None:
    data.qpos[:] = 0.0
    data.qpos[3] = 1.0
    data.qpos[2] = height
    for name, value in BAT_CARRY_POSE.items():
      data.qpos[model.joint(name).qposadr] = value
    mujoco.mj_forward(model, data)

  def lowest_foot_point() -> float:
    lows = []
    for geom_id in range(model.ngeom):
      body_name = model.body(model.geom_bodyid[geom_id]).name
      if not body_name.endswith("ankle_roll_link"):
        continue
      if model.geom_contype[geom_id] == 0 and model.geom_conaffinity[geom_id] == 0:
        continue
      if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_SPHERE:
        continue
      lows.append(data.geom_xpos[geom_id, 2] - model.geom_size[geom_id, 0])
    if not lows:
      raise RuntimeError("no foot collision spheres found on the ankle-roll links")
    return float(min(lows))

  if base_height is None:
    place(0.0)
    base_height = BAT_CARRY_GROUND_CLEARANCE - lowest_foot_point()

  place(base_height)
  if lowest_foot_point() < -_FOOT_LOW_Z_TOLERANCE:
    raise RuntimeError(
      f"bat-carry pose penetrates the ground by {lowest_foot_point():.6f} m at "
      f"base height {base_height:.6f} m"
    )

  site_id = model.site(FACE_SITE).id
  face_pos = data.site_xpos[site_id]
  face_normal = data.site_xmat[site_id].reshape(3, 3)[:, 2]
  ball_center = face_pos + face_normal * (BALL_RADIUS + BALL_SPAWN_CLEARANCE)
  return float(base_height), np.asarray(ball_center)


##
# Entity configurations.
##


def get_x2_tennis_robot_cfg() -> EntityCfg:
  """X2 Ultra with the tennis assembly, in the bat-carry pose.

  Collisions are the shipped ``FULL_COLLISION`` plus
  :data:`TENNIS_BED_COLLISION`, in that order: the second config restores the
  calibrated bed contact that the first one's ``disable_other_geoms`` pass
  switches off.
  """
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, BAT_CARRY_BASE_HEIGHT),
      joint_pos=dict(BAT_CARRY_POSE),
      joint_vel={".*": 0.0},
    ),
    collisions=(FULL_COLLISION, TENNIS_BED_COLLISION),
    spec_fn=get_tennis_spec,
    articulation=X2_ARTICULATION,
  )


def get_tennis_ball_cfg() -> EntityCfg:
  """Ball entity, spawned on the bat in the bat-carry pose.

  ``init_state.pos`` is in the environment frame (it is added to
  ``env_origins``), so it only lines up with the bed when the robot's own reset
  places the root at :data:`BAT_CARRY_BASE_HEIGHT` with zero yaw. The tennis
  task therefore pins the base reset — see
  ``agibot_x2_tennis_ball_env_cfg``.
  """
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=BALL_SPAWN_LOCAL_POS,
      rot=(1.0, 0.0, 0.0, 0.0),
      lin_vel=(0.0, 0.0, 0.0),
      ang_vel=(0.0, 0.0, 0.0),
    ),
    spec_fn=get_tennis_ball_spec,
  )


__all__ = [
  "BALL_RADIUS",
  "BALL_SPAWN_CLEARANCE",
  "BALL_SPAWN_LOCAL_POS",
  "BAT_CARRY_BASE_HEIGHT",
  "BAT_CARRY_POSE",
  "BED_GEOM",
  "BED_SOLREF",
  "CLAMP_BODY",
  "FACE_SITE",
  "RACKET_BODY",
  "TENNIS_BED_COLLISION",
  "WRIST_BODY",
  "attach_tennis_assembly",
  "get_tennis_ball_cfg",
  "get_tennis_ball_spec",
  "get_tennis_spec",
  "get_x2_tennis_robot_cfg",
  "measure_bat_carry_state",
]
