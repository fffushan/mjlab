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

The tangential contact parameters are the exception: they no longer come from
that snapshot. :data:`SLIDING_FRICTION` is measured on a tennis string bed,
:data:`BALL_DIAGINERTIA` is measured on a tennis ball, and
:data:`ROLLING_FRICTION` is fitted in simulation to a coast distance derived
from a measured slope run
because no source gives a usable rolling-resistance coefficient at these
speeds. See ``docs/reports/x2_tennis_ball_bat_contact_models.md``.

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
  rebounds to 0.635 m). Above that the fit stops discriminating: 1.0, 2.0 and
  2.5 ms are all within the test tolerance, so 1 ms is kept for calibration
  provenance and margin, not because the fit demands it.
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

# The ITF lower limits of the Type 2 regulation ball: 65.4 mm diameter and 56.0 g.
#
# Both are **deliberately conservative estimates, not measurements of the ball in
# hand.** The operator measured roughly 65 mm with a ruler against blocks and has
# no caliper, scale or inertia rig, so the lower limit of the published range is
# used instead of the midpoint that was here before (67 mm, 57 g). The real ball
# is somewhere in 65.4-68.6 mm and 56.0-59.4 g, so the shipped values are at most
# 5% small in diameter and 6% small in mass.
#
# The conservative choice costs nothing in the contact fit: with MuJoCo's direct
# ``solref``/``solimp`` form the contact response is mass-normalised, so the four
# calibrated drop rebounds are identical for 40 g and 57 g at either radius
# (measured: 0.159/0.237/0.335/0.394 m in all cases). Mass and radius therefore
# only enter through the rolling calibration below, and only through I/(m R^2).
BALL_MASS = 0.0560
BALL_RADIUS = 0.0327

# Moment of inertia about a diameter through the centre of mass. Cross ("Oblique
# impact on strings") measured 3.41e-05 kg m^2 for a 57 g ball at 33 mm, i.e.
# I/(m R^2) = 0.549. That ratio is a **shape** property -- how hollow the ball is
# -- so it is carried over unchanged and the inertia is re-derived at the shipped
# mass and radius: I = 0.549 * 0.0560 * 0.0327^2 = 3.2874e-05 kg m^2.
#
# This is the weakest number in this file. The ball's real wall thickness is
# unknown, and I/(m R^2) runs from 0.400 (solid) through 0.549 (Cross) to 0.667
# (infinitely thin shell). Unlike mass and radius it is **not** free: it enters
# the rolling-resistance conversion and does not cancel, so if the ball is ever
# measured properly this constant and ``ROLLING_FRICTION`` must move together.
# See the note on that constant for the size of the effect.
BALL_DIAGINERTIA = (3.2874e-05, 3.2874e-05, 3.2874e-05)

# Positive solref here (normalised time constant / damping ratio): the ball is
# the *softer* of the pair, but ``priority`` below makes the bed's parameters
# win on every bed contact, so this only governs ball-vs-robot/ground contacts.
BALL_SOLREF = (0.004, 0.35)
BALL_RGBA = (0.75, 0.95, 0.08, 1.0)

##
# Contact parameters, one row shared by the bed and the ball.
#
# ``CONTACT_CONDIM = 6`` gives each contact three dimensions carrying the normal
# force plus two sliding directions, a fourth for torsional (spin) friction
# about the contact normal, and two more for rolling friction about the two
# tangent axes. ``condim=3``, used before, has **no rotation of any kind**: a
# ball could not lose angular momentum to the bed, so a ball rolling on the
# level bed kept exactly the same speed forever.
#
# MuJoCo geom friction is exactly three numbers, in the order
# (sliding, spin, rolling), and they are not the same kind of quantity: the
# sliding one is dimensionless and the other two carry length units. With
# ``priority=1`` on the bed geom, the bed's row governs every bed-ball contact;
# the ball's identical row only applies to ball-vs-robot and ball-vs-ground
# pairs, which keeps the two rows deliberately equal so no contact pair in the
# scene silently falls back to a different tangential law.
##

# Measured for a tennis ball on a string bed: 0.43 +- 0.02 (Cross, "Oblique
# impact on strings"). The previous 0.6 is the ball-on-court value (0.6 on a
# fast court to 0.8 on a slow one) and does not apply to strings.
SLIDING_FRICTION = 0.44

# Provisional, and only active at ``CONTACT_CONDIM >= 4``. The previous 0.005
# sat in this slot but was inert: at ``condim=3`` MuJoCo never reads it. No
# measurement exists for the spin coefficient; this keeps the snapshot's value
# as a placeholder that is at least now exercised.
SPIN_FRICTION = 0.005

# Derived from the operator's slope measurement, 2026-10-25. The physical
# quantity is the dimensionless rolling resistance
#
#   mu_r = (sin(theta) - (1 + k) a / g) / cos(theta)
#
# with theta = 0.129 rad (7.39 deg), a = 2 L / t^2 = 0.705 m/s^2 from a 0.245 m
# run in 0.834 s (a repeat run gave 0.833 s, so the timing spread is negligible),
# and k = I/(m R^2) = 0.549 for this ball. That gives mu_r = 0.0174. On a level
# bed the same ball then decelerates by ``mu_r g / (1 + k) = 0.110 m/s^2``, so a
# 0.10 m/s release coasts 0.045 m -- 54% of the ball's usable travel across the
# plate, so a nudge still settles well inside the bed.
#
# 0.045 m is the **target** handed to
# ``scripts/experiments/x2_tennis_roll_friction_calibrate.py --target-coast-m
# 0.0453 --release-speed 0.10``, which returns 1.7185e-03. Fit against the ball
# this file now describes: the same target returns 1.7515e-03 against the 67 mm
# ball that was here before, so refitting after a ball change is not optional. MuJoCo does not derive
# this coefficient from a dimensionless rolling-resistance ratio, so it is not
# mu_r and cannot be read off literature; the translation goes through the coast
# distance, which is a quantity both the measurement and the fixture can express.
#
# The run is 25 frames of 30 fps video, so t carries about +-0.017 s (2 frames of
# start/end ambiguity), worth +-0.0004 in mu_r and +-0.001 m of coast.
#
# **The largest remaining uncertainty is k, not the measurement.** k enters twice
# and does not cancel, so the same slope run implies different coasts:
#
#   k = 0.400 (uniform solid sphere)   -> coast 0.025 m
#   k = 0.549 (this ball, Cross)       -> coast 0.045 m
#   k = 0.600 (60 mm ball, 3.5 mm wall)-> coast 0.059 m
#   k = 0.667 (infinitely thin shell)  -> coast 0.096 m
#
# The slope measurement alone cannot separate k from mu_r: every value above
# reproduces the measured 0.705 m/s^2 equally well. 0.549 is the only one backed
# by a real measurement of a real tennis ball, so it is the choice; if the ball is
# ever weighed and rolled in a bowl, re-derive both constants together.
#
# The slope run has one known bias: the ball rolls into the raised racket frame
# at the end of the measured travel, and the last two frames show it already
# deflected. ``a = 2 L / t^2`` then slightly over-states the free acceleration,
# which over-states mu_r, which under-states the coast. Fitting the per-frame
# positions instead gives a = 0.73 m/s^2 over the same span, i.e. a 0.039 m
# coast for k = 0.549, so the true value is a little above 0.045 m.
#
# Any re-fit must use this script. The coast distance moves by up to 12% with the
# release convention alone (exact contact versus a millimetre of clearance), so a
# freshly written fixture will not reproduce this number; changing ``SETTLE_TIME``
# without re-fitting is the same mistake.
#
# ``CONTACT_CONDIM`` must stay 6. ``condim=4`` adds torsional friction only, which
# a rolling ball never exercises: measured, it coasts exactly like ``condim=3``.
#
# Residual bias that is deliberately not removed: a ball this soft sinks into the
# string bed and has to climb out of its own crater, which the slope test counts
# as rolling resistance. MuJoCo's mesh-sphere contact reproduces part of that
# geometrically, so the lumped value is what we want to match, not the "pure"
# rolling resistance of a rigid ball on a rigid plane.
ROLLING_FRICTION = 1.7185e-03

CONTACT_FRICTION = (SLIDING_FRICTION, SPIN_FRICTION, ROLLING_FRICTION)
CONTACT_CONDIM = 6

##
# String-bed contact. Fitted to the user's drop measurements (rebound 0.17, 0.23,
# 0.31, 0.42 m for drops of 0.20, 0.30, 0.40, 0.50 m). Negative solref is
# MuJoCo's direct stiffness/damping form, not physical N/m or N*s/m, and it is
# only valid near the 1 ms step it was fitted at.
##

BED_SOLREF = (-100000.0, -22.0)
BED_SOLIMP = (0.95, 0.95, 0.001, 0.5, 2.0)
# The bed's ``priority=1`` is what makes the bed's contact row -- including
# :data:`CONTACT_FRICTION` -- govern every bed-ball pair.
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
  condim=CONTACT_CONDIM,
  priority=BED_PRIORITY,
  friction=CONTACT_FRICTION,
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
    condim=CONTACT_CONDIM,
    priority=BED_PRIORITY,
    mass=0.0,
    group=3,
    friction=list(CONTACT_FRICTION),
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

  ``CONTACT_CONDIM=6`` with the bed's ``priority=1`` means the bed's friction
  and solver parameters govern bed contacts; this geom's own values only apply
  to ball-vs-robot and ball-vs-ground pairs. The ball body carries its inertia
  explicitly (a tennis ball is a shell, not a solid sphere).
  """
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="ball")
  body.add_freejoint(name="ball_free")
  _set_explicit_inertial(
    body,
    mass=BALL_MASS,
    ipos=(0.0, 0.0, 0.0),
    iquat=(1.0, 0.0, 0.0, 0.0),
    inertia=BALL_DIAGINERTIA,
  )
  body.add_geom(
    name="ball_collision",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=[BALL_RADIUS, 0.0, 0.0],
    # The body's explicit inertial already carries every kilogram, so the geom
    # must contribute none or the mass would be counted twice.
    mass=0.0,
    contype=1,
    conaffinity=1,
    condim=CONTACT_CONDIM,
    friction=list(CONTACT_FRICTION),
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
  0.560001221763507,
  -0.02405731523163448,
  1.1530848027869869,
)

# Clearance between the ball's lowest point and the bed's top surface at spawn.
BALL_SPAWN_CLEARANCE = 0.001

##
# Ball-holding task definition: what the policy observes, what it is paid for, and
# when the episode ends.
#
# The policy has to hold the ball near the middle of the plate while the robot
# walks. All distances here are measured in the **plate frame**, whose origin is
# the centre of the plate's top surface (``FACE_SITE_POS``), x along the shaft, y
# across it, z along the plate normal.
##

# Centring reward: full credit inside a plateau, then a Gaussian falloff.
#
#   reward = exp(-((d - plateau) / sigma)^2)   for d > plateau, else 1
#
# where d is the ball's horizontal distance from the plate centre. The plateau is
# there because of the observation noise: with +-5 mm of ball-position error, a
# plain kernel has a slope right where the ball spends most of its time, so
# measurement error would become reward noise. Inside the plateau the gradient is
# exactly zero and noise cannot reach the reward. 20 mm is about a quarter of the
# ball's usable travel towards the plate's short edge.
BALL_CENTRE_PLATEAU = 0.020
BALL_CENTRE_SIGMA = 0.040

# Height band. The ball is wanted on or just above the strings and not thrown up:
# anything from the string plane to 10 cm above it earns full credit, and a ball
# higher than that decays with the same sigma as the centring term. A ball below
# the plate is not penalised but **fatal**, see ``BALL_FALL_THRESHOLD``.
BALL_HEIGHT_LIMIT = 0.10
BALL_HEIGHT_SIGMA = 0.040

# The ball's centre below this height above the plate surface counts as lost.
# Negative, so the small penetration a compliant contact allows does not fire it;
# a ball that rolls off the edge reaches it about 65 ms after leaving the plate.
BALL_FALL_THRESHOLD = -0.020

# How long the ball may stay lost before the episode actually ends.
#
# This is a curriculum for patience, not a physics choice: an untrained policy
# loses the ball within 20 control steps, so terminating on the first lost step
# gives episodes of 0.4 s out of a 20 s allowance and almost nothing to learn from.
# The reward already reads zero while the ball is gone, so the extra time is spent
# with an honest signal, and a policy that recovers gets to continue.
BALL_DROP_GRACE_S = 0.5

# Reward weights. Centring is the task, so it carries the same order of magnitude
# as the base task's velocity terms (2.0 each). The height term exists only to
# stop "throw the ball straight up and stay centred" from being a cheap strategy;
# it is small on purpose.
BALL_CENTRE_WEIGHT = 1.0
BALL_HEIGHT_WEIGHT = 0.1

# Ball-state observation randomization. The real ball position comes from an
# external perception program, not from the simulator, so the observation must be
# as wrong as that program is. These are **three independent magnitudes**, one per
# dimension, because the errors are not alike: a camera measures the two lateral
# axes better than the depth axis.
#
# Units and frame: the plate frame, so x is along the shaft, y across it, z along
# the plate normal. Each entry is a half-width of a symmetric **uniform**
# distribution, matching the rest of mjlab: every other observation term in the
# velocity task uses ``Unoise``, not Gaussian. A uniform half-width ``m`` has a
# standard deviation of ``m/sqrt(3)``.
#
# How the numbers were chosen, so they can be replaced with the real ones:
#
# * Position: +-5 mm along the two lateral axes and +-8 mm along the normal. The
#   normal axis is the one least well observed, which is why it is the odd one out.
#
# **These are placeholders for the perception program's real error, not
# measurements.** They are also the reason the centring reward has a plateau:
# +-5 mm of position error on a plain Gaussian reward would show up as reward noise
# exactly where the ball spends most of its time.
BALL_POSITION_NOISE_M = (0.005, 0.005, 0.008)

# Ball-position latency of the real perception chain, measured on the robot. The
# head camera publishes at 10 Hz, so one frame serves five 50 Hz control steps
# (``env.step_dt = 0.02 s``: 20 substeps of 1 ms), and a published position is
# P50 110 ms and P95 141 ms old. In control steps that is 110 / 20 = 5.5 and
# 141 / 20 = 7.05. The age a frame can have when the policy consumes it spans one
# sensor period on top of that latency, so lags of 5 to 9 steps cover the usable
# range, with the P95 age inside it. These are sensor facts, not tuning knobs.
BALL_OBS_DELAY_MIN_STEPS = 5  # floor(110 ms / 20 ms).
BALL_OBS_DELAY_MAX_STEPS = 9  # Floor(P50) + one 5-step sensor period - 1 step.
BALL_OBS_UPDATE_PERIOD_STEPS = 5  # 10 Hz frame against the 50 Hz control rate.

# Frame rejection: the perception chain also publishes **nothing** on some steps.
# The detector drops a frame when it finds no ball, finds more than one candidate,
# cannot fit a consistent depth, or when the live calibration differs; on a stream
# break it publishes nothing at all. A deployed controller therefore repeats the
# last valid position until a new one arrives, and the policy must see that same
# behaviour or it will be fitted to a measurement that is always fresh.
#
# Chosen rate: one rejection every 50 control steps on average (1 s at 50 Hz),
# each holding for 5-15 steps (100-300 ms at 50 Hz), so about 20 % of steps are
# served from a stale value. One draw per environment per step, so rejections are
# independent across the batch.
#
# **This is a placeholder for an unmeasured rejection rate.** The reference
# perception implementation was only ever measured on a clean static scene
# (293/293 frames valid), so nothing is known about rejection under motion or
# occlusion. Replace these with the measured rate when there is one.
BALL_DROPOUT_PROB = 0.02  # Probability per control step of a rejected frame.
BALL_DROPOUT_HOLD_STEPS = (5, 15)  # Hold length in control steps: 100-300 ms.

##
# Command curriculum.
#
# Balancing a ball and walking are two separate skills, and the base velocity task
# asks for the second one from iteration 0 with commands of up to 1 m/s. This task
# widens the commands in stages instead.
#
# **Why the stages advance on a step-count schedule and not on performance.**
# Three designs were tried on this task:
#
# 1. A performance gate on typical episode length, starting with the robot standing
#    still. Rejected: with the robot commanded to stand, the ball simply sits on
#    the plate, nothing terminates for the full 20 s episode, and the gate read
#    that as mastery. In a 4096-environment run it promoted straight to the
#    hardest stage in one call.
# 2. The same gate without the stand-still stage. Rejected for the same reason one
#    level down: an untrained policy already keeps the ball for 9.1 s at 0.3 m/s
#    and 15.7 s at 0.6 m/s, so any survivability threshold placed below those
#    values is cleared with no skill at all, while at 1.0 m/s the same policy
#    manages 1.05 s. Every survivability observable saturates in the easy regime,
#    so no absolute threshold there can separate "doing nothing" from "doing it
#    well".
# 3. A step-count schedule carrying the difficulty, with performance only able to
#    **block** an advance it cannot justify. This is the design below, and it is
#    also what the base task does for the same reason.
#
# The schedule is expressed in environment steps, matching
# ``env.common_step_counter``. With 24 control steps per iteration,
# ``iterations * 24`` converts. The two points below are iteration 2000 and
# iteration 5000; the second is the base task's own first widening point.
#
# The floor is a **permission check, not a trigger**: the stage advances on the
# schedule only if the typical episode at the current stage is at least
# ``BALL_COMMAND_MASTERY_FLOOR_S``. Measured on this task, a failing policy runs
# about 1.05 s episodes and a comfortable one several seconds, so 8 s means "the
# ball is not being lost immediately". Without it a policy that is failing at the
# current stage would still be pushed faster by the clock.

BALL_COMMAND_STAGES: tuple[dict, ...] = (
  {
    "min_step": 0,
    "description": "creep",
    "lin_vel_x": (-0.3, 0.3),
    "lin_vel_y": (-0.3, 0.3),
    "ang_vel_z": (-0.15, 0.15),
  },
  {
    "min_step": 2000 * 24,
    "description": "walk",
    "lin_vel_x": (-0.6, 0.6),
    "lin_vel_y": (-0.6, 0.6),
    "ang_vel_z": (-0.3, 0.3),
  },
  {
    "min_step": 5000 * 24,
    "description": "the base task's stage 0",
    "lin_vel_x": (-1.0, 1.0),
    "lin_vel_y": (-1.0, 1.0),
    "ang_vel_z": (-0.5, 0.5),
  },
)

BALL_COMMAND_MASTERY_FLOOR_S = 8.0
"""Typical episode length the current stage must sustain before advancing."""

# Tracking gate. A long episode is not evidence of walking: a policy that stands
# still, or shuffles with 6 mm of foot clearance, keeps the ball parked on the plate,
# so its episodes run the full 20 s and clear any length floor. Measured, such a
# policy misses a +-0.3 m/s command by 0.586 m/s, i.e. about twice the command it was
# given, while the floor above passes it.
#
# The limit is expressed as a **fraction of the fastest linear command of the active
# stage** rather than in m/s, so one constant covers every stage: 0.075 m/s at the
# 0.3 m/s stage, 0.15 m/s at 0.6 m/s and 0.25 m/s at 1.0 m/s. An absolute value would
# have to be re-tuned at each widening, and a value that is reasonable at 0.3 m/s is
# trivially satisfied at 1.0 m/s.
#
# 0.25 is a judgement call, not a measurement: no measurement of a good policy's
# tracking error on this task exists yet. It rejects the observed failing case with a
# margin of about 8 (0.075 m/s allowed against 0.586 m/s measured), and it asks a
# promoted policy to spend no more than a quarter of its top commanded speed on
# tracking error. Being too strict costs training time only, because this gate can
# withhold an advance and never force one.
BALL_COMMAND_MAX_TRACK_ERR_FRAC = 0.25
"""Typical tracking error allowed before advancing, as a fraction of stage speed."""

##
# Camera field of view.
#
# The policy has to keep the ball where perception can see it: on hardware the ball
# position comes from the head stereo pair, and a ball outside the frame is a lost
# ball however well it is balanced.
#
# **This is deliberately approximate.** The operator's plan is that perception and
# ball estimation are handled by a separate process, and that the camera calibration
# is refined later, so the geometry here only has to be good enough to shape
# behaviour, not to predict pixels.
#
# Geometry:
#
# * The mount is the reference snapshot's ``camera_mount_162.json``
#   (``x2_references/x2_tennis/parameters``). The reference README warns that this
#   is robot 162's vendor transform, not a calibration of another robot. The live
#   robot's TF agrees with it in the same ballpark, which is all this term needs.
# * That mount produces a **ROS optical frame**: +z forward along the optical axis,
#   +x right, +y down. MuJoCo's own camera convention is the opposite along z, so
#   the sign matters: using -z here would put the ball behind the camera.
# * Intrinsics are the live stereo left eye, 2048x1536 with the **equidistant**
#   model, so the angle to an image edge is ``r / f`` and the half-angles are
#   ``cx / fx`` and ``cy / fy`` rather than ``atan(cx / fx)``.
#
# Measured at the ball-carry pose, the ball sits 24 deg off the optical axis, i.e.
# 27 % of the horizontal half-frame. So the term is a **safety term**: it sits near
# 1.0 while the ball is on the racket and only bites when the ball leaves the hand
# or the arm swings out of view.
##

CAMERA_MOUNT_XYZ = (0.05058, 0.03013, 0.032504)
"""Camera position in the ``head_pitch_link`` frame, from the reference snapshot."""

CAMERA_MOUNT_QUAT = (0.497273, -0.5026607, 0.5070336, -0.4929186)
"""Camera orientation in ``head_pitch_link``, wxyz, from the same reference.

Pre-computed from ``Rotation.from_euler('xyz', (-1.5903, 0.0087267, -1.5709))`` so
the runtime needs no scipy. The local axes are +x right, +y down, +z forward.
"""

CAMERA_HALF_ANGLE_X = 1.4907
CAMERA_HALF_ANGLE_Y = 1.1168
"""Angle from the optical axis to the image edge, radians.

``cx / fx`` and ``cy / fy`` from the live stereo-left calibration
(1024.451 / 687.245 and 767.497 / 687.245), i.e. 85.4 and 64.0 deg. The fisheye's
true corner angle is larger; the per-axis edge angle is what a rectangular frame
test needs.
"""

BALL_FOV_PLATEAU = 0.60
"""Normalised image radius that still earns full credit.

0.6 of the half-frame both ways. The ball at rest is at 0.27, so normal carrying is
free and the term only bites when the ball drifts towards the frame edge.
"""

BALL_FOV_SIGMA = 0.25
"""Falloff beyond the plateau, in the same normalised units. Zero at the edge."""

BALL_FOV_WEIGHT = 0.2
"""Soft: this is a perception constraint, not the task objective."""

##
# Head steadying.
#
# The head joints are policy outputs, so the field-of-view reward gives the policy a
# reason to hunt with them. Dithering two joints to chase a ball that perception
# measures with 5 mm of noise is not useful motion: on hardware it costs actuator
# travel, it smears the very frames perception needs, and it puts the ball's image
# in motion. This penalises head joint **velocity**, not position, so the head can
# still be aimed deliberately and then held.
##

HEAD_JOINT_NAMES = ("head_yaw_joint", "head_pitch_joint")

HEAD_MOTION_WEIGHT = -0.05
"""Penalty weight on squared head joint velocity, summed over the two joints."""

BALL_COMMAND_WINDOW = 100
"""Finished episodes the promotion gate waits for before deciding."""

##
# Juggling task definition: the ball bounces on the plate and comes back to an
# apex, instead of being carried.
#
# The apex reward is paid **once per bounce**, so what matters is the height of a
# bounce, not the height of the ball on some arbitrary step. All heights below are
# the **gap under the ball's lowest point**, measured along the world vertical from
# the centre of the plate's top surface: a ball resting on the strings reads about
# zero. The world vertical, and not the plate normal, because the plate tilts while
# juggling and gravity is what turns the ball around.
##

# Apex gap the bounce reward targets: 300 mm under the ball. The plate top surface at
# the carry pose is world z 1.1194, so the ball centre reaches 1.1194 + 0.0327 + 0.30
# = 1.452 m. Measured on the built task at that height, the ball sits at 0.434 of the
# camera half-frame (27.8 deg against the 64.0 deg vertical half-angle) and the
# field-of-view reward is still exactly 1.000, because 0.434 is inside the 0.60
# plateau. At rest the same ball reads 0.271, so the apex costs almost nothing in
# field of view and the target is inside the frame by construction, not by luck. The
# falloff begins just above 0.41 m of gap (0.5950 at 0.41, 0.6081 at 0.42).
#
# An earlier comment here claimed 0.889. That figure was wrong: it came from a
# measurement script that built the camera pose from the robot's root link instead of
# the head body. Do not restore it.
JUGGLE_APEX_TARGET_M = 0.30

# Gaussian falloff of the apex reward in the apex gap. 80 mm puts 1/e at an apex of
# 0.22 or 0.38 m, i.e. a bounce 8 cm out is still worth 0.37.
JUGGLE_APEX_SIGMA_M = 0.08

# Smallest apex gap that can pay. A ball resting on the plate has a gap of about zero
# and its vertical speed wobbles around zero, so it produces sign crossings of its
# own; this gate and the speed gate below are what reject them.
JUGGLE_APEX_MIN_CLEARANCE_M = 0.03

# Smallest peak upward speed of the rise that can pay: 30 cm of apex needs a launch of
# sqrt(2 * 9.81 * 0.30) = 2.43 m/s, so a real bounce clears this by a factor of eight
# while contact jitter on a parked ball, a few cm/s, does not. It is applied to the
# **peak** of the rise: gravity removes 0.20 m/s per 0.02 s control step, so the last
# sample before the apex is small whatever the launch was.
JUGGLE_APEX_MIN_SPEED_MPS = 0.30

# Resting test for the parking penalty: a gap below 30 mm and a world vertical speed
# below 0.20 m/s. The grace period is what keeps the bottom of a real bounce, which
# passes through this state for about two control steps, from being charged.
JUGGLE_REST_CLEARANCE_M = 0.03
JUGGLE_REST_SPEED_MPS = 0.20
JUGGLE_REST_GRACE_S = 0.50

# Weights. The reward manager scales every term by the same step duration, so the
# comparison below is between weight values.
#
# Centring is charged at a quarter of the ball-holding task's 1.0 because the ball is
# in the air for most of the cycle and cannot be steered there; the apex payment has
# to carry the task instead. Over one 0.5 s bounce cycle at 50 Hz, i.e. 25 control
# steps, carrying the ball earns about 0.25 x 25 = 6 from centring alone. A juggle
# earns about 6.0 per apex, the same 6 per cycle, plus the same centring while the
# ball is near the plate, so juggling is worth roughly twice as much as parking the
# ball at the target height. The rest penalty tips that balance rather than dominating
# it: a parked ball still collects centring, so parking earns about 6.25 - 2.5 = 3.75
# per cycle against roughly 12 for juggling.
#
# **These are first guesses to be tuned against the bounce count, not measurements.**
# The ratio is the claim; the absolute values only set the scale of the reward.
JUGGLE_APEX_WEIGHT = 6.0
JUGGLE_CENTRE_WEIGHT = 0.25
JUGGLE_REST_WEIGHT = -0.10

# Command for the first juggling version: the robot juggles standing still, so the
# twist command is a small jitter around zero instead of the holding task's walking
# curriculum. Walking while juggling is a later step.
JUGGLE_COMMAND_LIN_VEL_MPS = 0.10
JUGGLE_COMMAND_ANG_VEL_RPS = 0.05


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
