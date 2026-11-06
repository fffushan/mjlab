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
import math
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
# String-bed contact. Fitted to the drop measurements recorded in the ``x2_tennis``
# reference project (rebound 0.17, 0.23, 0.31, 0.42 m for drops of 0.20, 0.30, 0.40,
# 0.50 m). Those are a colleague's numbers, not operator measurements of the shipped
# ball and racket.
#
# An operator video measurement (30 fps, racket on a board) gives rebound 0.15, 0.23,
# 0.30, 0.40 m, i.e. energy ratio 0.767 against the reference 0.808 -- a 5% difference,
# well inside the spread of a hand release, and identical at the 0.30 m drop. The
# reference fit is therefore kept, not re-fitted. Negative solref is
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

# Clearance between the ball's lowest point and the bed's top surface at the
# on-bed reference point.  The ball does not actually start here: see
# ``BALL_SPAWN_DROP_HEIGHT``.
BALL_SPAWN_CLEARANCE = 0.001

# Unit normal of the string bed in the carry pose, measured from the compiled
# model's ``FACE_SITE`` orientation (``site_xmat[:, 2]``).  It is world +z to
# 6e-6, i.e. the bed is horizontal to 0.0006 deg at the kinematic carry pose, so a
# drop along the normal is a vertical drop.  Kept as a measured value rather than
# assumed, so that changing the carry pose cannot silently tilt the drop.
BALL_SPAWN_FACE_NORMAL = (8.747e-06, 6.379e-06, 1.0)

# How far above the bed the ball starts, measured along the bed normal.
#
# Rationale, not tuning: with the ball resting on the bed the policy never learns
# to *start* a bounce.  Measured on the 4000-iteration sweep, the ball's apex was
# p50 47 mm / p90 71 mm and the bouncing collapsed entirely when the observation
# noise was removed - i.e. the policy was reacting to disturbances, not driving a
# bounce cycle, because nothing ever taught it to strike a ball upward from rest.
# Dropping it in gives a real bounce from step zero, which the policy must then
# sustain and raise.
#
# A 0.20 m fall arrives at about 1.98 m/s; with a COR near 0.88 the first passive
# rebound is about 0.15 m, and the measured rebound on the built environment is
# about 0.205 m.  Both are below the current 0.40 m energy target, so the spawn no
# longer matches the target.  The policy still starts with a bounce it must improve
# rather than one it must create; see JUGGLE_ENERGY_TARGET_HEIGHT_M for the reward
# consequence of the broken spawn-target pairing.
BALL_SPAWN_DROP_HEIGHT = 0.20

# Ball centre at reset for a ball dropped in from ``BALL_SPAWN_DROP_HEIGHT``:
# the on-bed reference raised along the bed normal.  This is the *juggle* start;
# the carry task keeps spawning on the bed and passes ``drop_height=0.0``.
BALL_SPAWN_INIT_POS = tuple(
  p + n * BALL_SPAWN_DROP_HEIGHT
  for p, n in zip(BALL_SPAWN_LOCAL_POS, BALL_SPAWN_FACE_NORMAL, strict=True)
)


def ball_spawn_init_pos(
  drop_height: float = 0.0,
) -> tuple[float, float, float]:
  """Ball centre at reset, ``drop_height`` above the bed along the bed normal.

  ``drop_height=0.0`` is the on-bed reference and reproduces
  :data:`BALL_SPAWN_LOCAL_POS`.
  """
  return tuple(
    p + n * drop_height
    for p, n in zip(BALL_SPAWN_LOCAL_POS, BALL_SPAWN_FACE_NORMAL, strict=True)
  )


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
# higher than that decays with the same sigma as the centring term. A ball that sinks
# more than ``BALL_SINK_M`` below the string plane decays the same way: it is gone, not
# resting (see ``BALL_SINK_M`` and ``BALL_LOST_MIN_WORLD_Z_M``).
BALL_HEIGHT_LIMIT = 0.10
BALL_HEIGHT_SIGMA = 0.040

# How far the ball may sink below the string plane and still count as on the plate.
#
# A compliant contact lets the ball penetrate a fraction of a millimetre, so a hard
# zero would make a carried ball pay less than 1. Without a floor at all, both
# plate-frame rewards paid full credit for a ball that had left the plate downward:
# ``ball_centred`` measures its horizontal distance only, and ``ball_height`` clamps
# its excursion from above, so a ball half a metre under the strings scored 1.0 on
# both. 5 mm separates "resting, slightly sunk" from "gone", and it is below the
# smallest height a real bounce reaches in one control step.
BALL_SINK_M = 0.005

# The ball's centre at or below this height above the ground counts as lost.
#
# Absolute, in the world frame, and the plate is not consulted. A plate-frame test is
# defeated by the policy: "below the plate" rotates with the racket, so tilting the
# racket to near vertical and dropping the ball on the floor keeps the ball on the
# positive side of that plane and the episode runs to the 20 s limit with no ball at
# all. The 30 k Ball3 baseline shows it: 3.6 racket contacts in a 19.2 s episode, and
# 94 % of its episodes ended by truncation instead of by losing the ball.
#
# 0.5 m leaves room for a strike from below the plate, which is carried at about
# BAT_CARRY_BASE_HEIGHT = 0.67 m, while a ball on the floor rests BALL_RADIUS = 32.7 mm
# above the ground. No pose of the robot changes this number, so the only way to stay
# above it is to keep the ball up.
BALL_LOST_MIN_WORLD_Z_M = 0.5

# How long the ball may stay lost before the episode actually ends.
#
# This is a curriculum for patience, not a physics choice: an untrained policy
# loses the ball within 20 control steps, so terminating on the first lost step
# gives episodes of 0.4 s out of a 20 s allowance and almost nothing to learn from.
# The reward already reads zero while the ball is gone, so the extra time is spent
# with an honest signal, and a policy that recovers gets to continue.
BALL_DROP_GRACE_S = 0.5

# The lost-ball cost that replaces the drop termination (S9a).
#
# Episode reward is ``sum(raw * weight * dt) / max_episode_length_s``, so a weight
# applied every control step is worth ``weight / 20`` per second of episode. The
# measured positive intake of an S8 episode is +0.351 per second (probe: 16 envs,
# 300 steps, reward manager's own per-term buffer), so -4.0 while lost costs
# 0.20 per second, about 57 % of everything the policy earns. -0.15 would have cost
# 0.0075 per second, which is nothing: the factor is 20, not 1.
JUGGLE_BALL_LOST_PENALTY_WEIGHT = -4.0
"""Reward weight while the ball counts as lost, per second of episode."""

# Torso angular-velocity weight for the drop-tail rung.
#
# The measured contribution at the shipped -0.05 was -0.0017 per step, which is
# 1/550th of ``upright``: a 17.6 rad/s torso rotation, the motion that broke a
# connector, cost half a percent of a single velocity-tracking step. At -0.5 a
# 1 rad/s twist costs 0.5 per step, half an ``upright`` step, and 2 rad/s costs 2.0.
# The unusual magnitude is the point: the term has to be able to outvote the ball
# reward, or it cannot forbid anything.
JUGGLE_TORSO_ANG_VEL_WEIGHT = -0.5
"""Weight of ``body_ang_vel`` on the torso for the drop-tail rung."""

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

# Constant ball-position offset per episode, half-width in metres on each plate axis.
#
# Zero by default: the shipped configuration assumes an unbiased measurement chain.
# The robot does **not** have one. The measured camera-to-pelvis disagreement is a
# constant 24.69 mm and 1.24 deg, which is a bias, not noise, and zero-mean noise
# cannot represent it: a policy trained on noise alone averages the offset away and
# then meets it on the robot. Set this to about 0.025 for the ablation arm that
# trains against the real hand-eye error. One draw per environment per episode, so
# a policy cannot average it out within an episode either.
BALL_POSITION_BIAS_M = 0.0

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
# Ball-stream history observation (``ball_hist``).
#
# The real estimator does not publish at a fixed rate, so the actor cannot be
# trained against the fixed delay model above. Measured on the robots: it published
# 18.44 Hz on .147 and only 5.40 Hz on .132, and .132 also stalled for 10.4 s once,
# hence the age clip below. .132's published estimates were 34% wrong, with 2-3 m
# ghosts, which three samples can reject with a parabola consistency check.
# Timestamps make the latency an input instead of a fixed delay model, so one
# policy covers both robots. Newest first, so the freshest sample keeps a fixed
# offset, and three slots are the minimum that constrains curvature.
##
BALL_HISTORY_DEPTH = 3
BALL_HISTORY_AGE_CLIP_S = 1.0
BALL_STREAM_RATE_RANGE_HZ = (5.0, 20.0)
BALL_STREAM_LATENCY_RANGE_S = (0.080, 0.250)
BALL_STREAM_STALL_PROB = 0.02
BALL_STREAM_STALL_RANGE_S = (0.1, 0.6)
BALL_TRUE_HISTORY_S = 1.5

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
# stage** rather than in m/s, so one constant covers every stage: 0.15 m/s at the
# 0.3 m/s stage, 0.30 m/s at 0.6 m/s and 0.50 m/s at 1.0 m/s. An absolute value would
# have to be re-tuned at each widening, and a value that is reasonable at 0.3 m/s is
# trivially satisfied at 1.0 m/s.
#
# 0.5 is set from measurement (2026-10-26), not from taste. The first value, 0.25, was
# a judgement call and proved unreachable: it sat below the tracking error of the
# robot's own walking skill, so the gate could never open. A run held at stage 0 for
# 10 790 iterations moved its tracking error only from 0.2512 to 0.2297 m/s, against a
# 0.075 m/s gate. Measurements at the 1.0 m/s command range:
#   - converged 30k-iteration base velocity task, no ball at all: 0.58 m/s (58 %);
#   - this carry task at 12 939 iterations, same command range: 0.556 m/s (56 %).
# The ball therefore costs no tracking, and a robot that stood still would score
# 0.765 of command. So the whole discriminating range is roughly 0.76 down to the
# skill's own floor near 0.56. 0.5 rejects a policy that does not walk while
# permitting one that walks about as well as the base skill.
#
# Caveat: 0.5 does not ask for good walking, only for walking no worse than the
# existing base skill. Tighten it only together with a better base policy, and
# re-measure before changing it. Being too strict costs training time only, because
# this gate can withhold an advance and never force one.
BALL_COMMAND_MAX_TRACK_ERR_FRAC = 0.5
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

# Head pose randomization half-width, for the 29-DOF family where the policy cannot
# command the head at all.
#
# The head is fragile on the real robots and regularly breaks, so it is removed from
# the action space and becomes an **exogenous condition** instead. That matters twice
# over: the two joints are still observation inputs, and the head carries the camera,
# so its pose also sets the camera-to-plate conversion. A policy trained on one head
# pose is therefore fitted to one robot.
#
# Measured offsets from the trained default (yaw 0.27599528646882693, pitch
# 0.061799387799999994), radians:
#   .132  yaw -0.349, pitch +0.450  (yaw axis delivers no torque, pitch is free)
#   .236  yaw  0.000 to +0.217
#   .147  yaw +2.866  (frozen at +3.141590118)
#
# The model's head joint range is +-0.3 rad (x2_ultra.xml), which the operator chose
# and will enforce on the robots. Note what that means: the last two configurations
# above are **outside** it, and the sampler cannot reach them. A robot parked beyond
# +-0.3 rad is therefore an operator problem, not something this policy can absorb.
# These constants are the widest offset worth asking for; the sample is intersected
# with the joint's soft limits, so the reachable band is what the model allows.
JUGGLE_HEAD_YAW_RANGE_RAD = 0.5
JUGGLE_HEAD_PITCH_RANGE_RAD = 0.5

HEAD_MOTION_WEIGHT = -0.05
"""Penalty weight on squared head joint velocity, summed over the two joints."""

BALL_COMMAND_WINDOW = 100
"""Finished episodes the promotion gate waits for before deciding."""

##
# Juggling task definition: the ball bounces on the plate and comes back, instead
# of being carried.
#
# The task is paid by the ball's **energy per unit mass**, on every control step:
#
#   h     = z_ball - JUGGLE_ENERGY_BALL_REST_Z
#   E     = 0.5 * v_ball_z**2 + g * h   vertical only, no plate velocity
#   E*    = g * target_height
#   r_E   = exp(-((E - E*) / sigma)**2)
#   r_arm = -JUGGLE_RACKET_LIFT_WEIGHT
#           * max(0, z_plate_top - JUGGLE_RACKET_LIFT_PLATE_TOP_REST_Z - deadband)
#
# Both reference heights are **fixed world z**, not offsets from the pelvis. The
# pelvis datum is exploitable: with h = z_ball - (z_pelvis + K), lowering the pelvis
# 0.20 m while the ball stays put raises h by 0.20 m and adds g * 0.20 = 1.962
# J/kg, half of the current 3.924 J/kg target, with no work done on the ball. The
# lift term cannot see it, because the plate and the pelvis drop together, so its
# excess stays zero. That is free energy, and leg motion is not a juggling skill.
# The price of the world datum is that gait bob moves E: 3 cm of bob is g * 0.03 =
# 0.29 J/kg, 0.19 sigma. Accepted: the bob is small, and the exploit is free
# energy under any target.
#
# **Measured why, not taste.** The previous design paid a Gaussian bell once per
# bounce, at an apex gap of 0.30 m with an 80 mm sigma. On the 4000-iteration sweep
# the policies reached apexes of about 0.16-0.24 m, where the bell contributed about
# 2e-5 of per-step reward: indistinguishable from zero and flat, so it supplied no
# gradient. It also paid on one step per bounce and nothing on the others, i.e. on
# about one of every 25 control steps of a 0.5 s bounce cycle. The energy term is
# dense, so every step carries the gradient that teaches the strike.
#
# The ball velocity is **vertical only**: with the full speed, walking at 2.80 m/s
# with the ball parked on the racket would reach E* without a bounce, and so would a
# ball rolling across the plate.
#
# The racket is the **control input and must not pay itself**, so no plate velocity
# is subtracted from the ball's. Subtracting it would let a descending racket
# manufacture energy with the ball untouched: 1 m/s of arm drop is 0.5 J/kg, 13 % of
# E* for free, and about 2.8 m/s reaches E* with the ball never leaving the strings.
# For the same reason the energy target is paired with the lift penalty below: a
# ball resting on the plate and raised by L has h = L in world, so a lift to the
# 0.40 m target alone would earn r_E = 1.0 forever.
##

# Ball centre's absolute world z when the ball rests on the plate in the carry pose.
# Measured, not guessed: BALL_SPAWN_LOCAL_POS[2] = 1.1530848... m.
#
# At reset the ball is dropped from BALL_SPAWN_DROP_HEIGHT above the bed, so it
# starts at h = 0.20 m with zero vertical speed. That is 0.20 m BELOW the current
# 0.40 m target, so the reset state is not E = E*. A test pins the spawn height and
# the resulting energy deficit, not a target identity. See
# JUGGLE_ENERGY_TARGET_HEIGHT_M for the consequence.
#
# Valid as an absolute height only because the tennis tasks pin the base reset at
# BAT_CARRY_BASE_HEIGHT with zero yaw (``reset_base`` pose_range is zeroed in
# ``agibot_x2_tennis_ball_env_cfg``). If that pin were ever removed, the pelvis
# would move away from this reference and both heights below would be wrong.
JUGGLE_ENERGY_BALL_REST_Z = BALL_SPAWN_LOCAL_POS[2]

# Apex height the energy reward targets above the ball's resting state: 400 mm.
# This is the operator's choice, to match the height a human juggles a tennis ball.
# At the model gravity (9.81 m/s^2) E* = g * 0.40 = 3.924 J/kg. A ball resting on
# the plate has h = 0 and E ~ 0, so the target needs the ball to be moving up, and
# a carried or rolling ball scores nothing because only the vertical ball velocity
# is used.
#
# **The spawn is below the target.** BALL_SPAWN_DROP_HEIGHT is still 0.20 m, so
# the ball starts 0.20 m above its resting height, i.e. 0.20 m BELOW this target.
# At spawn E = g * 0.20 = 1.962 J/kg against E* = 3.924 J/kg, a deficit of 1.962
# J/kg. At the original sigma of 0.5 J/kg the reward there was
# exp(-(1.962 / 0.5)**2) = 2.1e-7, i.e. no usable signal. The operator has now set
# JUGGLE_ENERGY_SIGMA_J_PER_KG to 1.5 J/kg, so the spawn scores
# exp(-(1.962 / 1.5)**2) = 0.1807 and the whole 0.20 -> 0.40 m climb has a
# gradient. The passive bounce peak at about 0.21 m scores about 0.2 under the
# wider sigma.
#
# At 0.40 m the ball is at the edge of the camera's 0.60 field-of-view plateau: the
# falloff begins just above 0.41 m of gap (0.595 at 0.41, 0.608 at 0.42). The target
# therefore does not fight the field-of-view term, but it has little margin. A
# 0.40 m bounce needs a launch of sqrt(2 * 9.81 * 0.40) = 2.80 m/s.
JUGGLE_ENERGY_TARGET_HEIGHT_M = 0.40

# Gaussian falloff in energy, in J/kg, the same units as E. 1.5 J/kg puts 1/e at
# E* +/- 1.5 J/kg: at the top of a bounce that is 153 mm of height (1.5 / 9.81 m),
# and at rest it is a vertical ball speed of sqrt(2 * 1.5) = 1.73 m/s. The reward is
# therefore neither flat over the bounce nor narrower than the control rate can
# resolve.
#
# 1.5, not the original 0.5: the target is 0.40 m while the ball spawns at 0.20 m,
# so at sigma 0.5 the spawn scores exp(-(1.962 / 0.5)**2) = 2.1e-7 and the run has
# no learning signal. At 1.5 the spawn scores exp(-(1.962 / 1.5)**2) = 0.1807, so
# the whole 0.20 -> 0.40 m climb has a gradient. The operator authorised this
# change. The wider sigma is also why the lift weight uses the maximum-slope rule
# below instead of cancelling the reward at the target.
JUGGLE_ENERGY_SIGMA_J_PER_KG = 1.5

# Weights. The reward manager scales every term by the same step duration, so the
# comparison below is between weight values.
#
# 2.0 is chosen so this term's per-step maximum, weight * 1.0 * dt = 2.0 * 1.0 *
# 0.02 = 0.040 at the 0.02 s control step, equals ``track_linear_velocity``'s
# ceiling (2.0 * 1.0 * 0.02). At a smaller weight the velocity-tracking term would
# outvote the ball, and the policy would buy base-velocity credit instead of the
# energy it is supposed to learn.
#
# Centring stays at a quarter of the ball-holding task's 1.0 because the ball is in
# the air for most of the cycle and cannot be steered there. The energy term has to
# carry the task: over one 0.5 s bounce cycle at 50 Hz, i.e. 25 control steps, it
# can earn at most 2.0 x 25 x 0.02 = 1.0 s of credit, against 0.25 x 25 x 0.02 =
# 0.125 from centring.
JUGGLE_ENERGY_WEIGHT = 2.0
JUGGLE_CENTRE_WEIGHT = 0.25

# Plate top-surface absolute world z in the carry pose: the ball centre minus one
# radius and the spawn clearance, because the on-bed ball centre is one radius plus
# the clearance above the face along the bed normal, and that normal is world +z
# (BALL_SPAWN_FACE_NORMAL z = 1.0). Measured through the ball rest height, so it
# shares that measurement: 1.1193848... m. The same reset pin applies.
JUGGLE_RACKET_LIFT_PLATE_TOP_REST_Z = (
  JUGGLE_ENERGY_BALL_REST_Z - BALL_RADIUS - BALL_SPAWN_CLEARANCE
)

# Lift that is not charged, so normal jostling is free: 30 mm.
JUGGLE_RACKET_LIFT_DEADBAND_M = 0.03

# Gravity of the compiled tennis model, in m/s^2. The energy reward reads it from
# ``env.sim.mj_model.opt.gravity``; the lift-weight derivation needs it at import
# time, so it is repeated here. Change both together.
_MODEL_GRAVITY_MPS2 = 9.81


def juggle_racket_lift_weight(
  sigma: float = JUGGLE_ENERGY_SIGMA_J_PER_KG,
  energy_weight: float = JUGGLE_ENERGY_WEIGHT,
  gravity: float = _MODEL_GRAVITY_MPS2,
) -> float:
  r"""Lift weight whose slope blocks every partial-lift farm.

  The energy reward is a Gaussian in lifted height ``L``,
  ``r_E(L) = exp(-((g * L - g * T) / sigma)**2)``. Its steepest slope in ``L`` is

  .. math:: \sqrt{2} \, (g / \sigma) \, e^{-1/2}

  per metre, attained at ``|g * L - g * T| = sigma / sqrt(2)``. A lift penalty at
  least that steep makes ``W_E * r_E(L) - W_lift * max(0, L - deadband)``
  non-increasing above the deadband, so no partial lift can turn the pair
  positive.

  Args:
    sigma: Gaussian falloff in energy, in J/kg.
    energy_weight: Weight of the energy reward.
    gravity: Gravity of the compiled model, in m/s^2.

  Returns:
    Weight per metre of excess lift.
  """
  return energy_weight * math.sqrt(2.0) * (gravity / sigma) * math.exp(-0.5)


# Weight of the lift penalty, per metre of excess lift above the deadband.
#
# **Maximum-slope rule.** The energy reward is a Gaussian in lift, so its steepest
# slope is sqrt(2) * (g / sigma) * exp(-1/2) per metre, attained at
# |g * L - g * T| = sigma / sqrt(2); :func:`juggle_racket_lift_weight` derives the
# weight from that. A penalty at least that steep makes the net reward
# W_E * r_E(L) - W_lift * max(0, L - deadband) non-increasing above the deadband, so
# a partial lift cannot farm it. With W_E = 2.0, g = 9.81 and sigma = 1.5 the weight
# is 2.0 * sqrt(2) * (9.81 / 1.5) * exp(-0.5) = 11.2196 per metre.
#
# The old rule W_E / (T - deadband) cancelled the parked reward exactly at L = T.
# It is not sufficient at sigma 1.5: the wider Gaussian makes partial lifts near the
# target profitable, about +0.09 at L = 0.36 m, which is the farm this term exists
# to prevent. At the maximum-slope weight the same point is negative.
#
# Derived, not hardcoded: a typed 11.2196 would silently stop satisfying the rule if
# the energy weight, sigma or gravity changed.
#
# **The penalty charges sustained lifting, not arm motion.** It is per step, and so
# is the reward. A 0.10 m strike held for 5 steps costs about
# 11.2196 * (0.10 - 0.03) * 5 = 3.9 in weight-value units, while one 20-step bounce
# cycle at the target earns about 2.0 * 20 = 40. The strike is affordable; only
# parking the ball at the target height is ruled out.
#
# **Default off since 2026-11-03, on two batches of evidence.** The argument above
# models the reward at a single point; what it does not model is what the policy does
# with the rest of the episode. Both batches agree that the penalty suppresses
# bouncing rather than parking:
#   A0-A4 sweep   the lift-0.0 arm (A3) was the only arm that juggled, and is the
#                 policy deployed on the robots.
#   stage-1 30k   racket_lift -11.22 held the ball on the plate: 1.95 impacts per
#                 20 s episode, ball_energy 0.080 against a 3.92 target, and a reward
#                 of 82 earned by standing. At 0.0 the same task gave 16.7 impacts and
#                 ball_energy 0.423 for a tenth of the reward.
# So the default installs no penalty. The derived weight stays available through
# `juggle_racket_lift_weight()` and the MJLAB_JUGGLE_LIFT_WEIGHT override, so the
# charged arm can be re-run without editing this file. Nothing here claims the
# parking problem is solved: neither setting reaches the energy target.
JUGGLE_RACKET_LIFT_WEIGHT = 0.0

# The two blocks below parameterise the earlier per-bounce design:
# ``ball_bounce_apex`` and ``ball_at_rest`` in ``rewards.py``. **The juggling task no
# longer installs those terms.** The constants are kept because
# ``tests/test_x2_tennis_ball_task.py`` still exercises both library terms with
# these values; deleting them would break that module, which this change does not
# touch. Nothing in the juggling task reads them.

# Apex gap the library apex term targets: 300 mm under the ball. The plate top surface at
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

# Gaussian falloff of the apex term in the apex gap. 80 mm puts 1/e at an apex of
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

# Resting test for the library parking penalty: a gap below 30 mm and a world
# vertical speed below 0.20 m/s. The grace period is what keeps the bottom of a real
# bounce, which passes through this state for about two control steps, from being
# charged.
JUGGLE_REST_CLEARANCE_M = 0.03
JUGGLE_REST_SPEED_MPS = 0.20
JUGGLE_REST_GRACE_S = 0.50

# Weights of the library per-bounce terms, unused by the juggling task (see above).
JUGGLE_APEX_WEIGHT = 6.0
JUGGLE_REST_WEIGHT = -0.10

# Command for the first juggling version: the robot juggles standing still, so the
# twist command is a small jitter around zero instead of the holding task's walking
# curriculum. Walking while juggling is a later step.
JUGGLE_COMMAND_LIN_VEL_MPS = 0.10
JUGGLE_COMMAND_ANG_VEL_RPS = 0.05

##
# Stage-1 release-state randomization.
#
# Every episode draws one release type per environment. A **drop** start releases
# the ball above the bed, so a bounce exists from the first step. A **plate** start
# places the ball on the strings at rest, so the policy must strike it up itself.
# The mixture is the curriculum: the easier drop still dominates, but the harder
# plate start occurs in every batch. Both branches share the bed reference and bed
# normal that the carrying task already measured.
#
# The bed normal is world +z to 6e-6 (see BALL_SPAWN_FACE_NORMAL), so sampling the
# lateral offset and lateral velocity on the world x/y axes is the same as sampling
# them in the bed plane to that tolerance. No rotation is built, and the axial drop
# term uses the measured normal so a future carry-pose change cannot silently tilt
# it.
##

JUGGLE_PLATE_START_FRACTION = 0.20
"""Probability that an episode starts with the ball resting on the plate.

0.20 keeps the drop start, which already supplies a bounce, as the common case
while every 4096-environment batch still contains about 819 plate starts. A
plate start has no bounce to sustain, so it is the harder half of the stage and
must not be rare enough to learn only from a few episodes.
"""

JUGGLE_PLATE_START_LATERAL_RANGE_M = 0.005
"""Half-width of the in-plane offset of a plate start, in metres.

5 mm is a quarter of BALL_CENTRE_PLATEAU, so the offset stays inside the flat
part of the centring reward: it changes the strike point without changing the
score the reset state deserves. A larger offset would also let the ball start
near the plate edge, which is a reception scenario, not a strike scenario.
"""

JUGGLE_DROP_HEIGHT_RANGE_M = (0.15, 0.25)
"""Release height above the bed for a drop start, in metres, uniform.

The range brackets BALL_SPAWN_DROP_HEIGHT = 0.20 m, so the new drop starts sit
on both sides of the old fixed release. At 0.15 m the ball arrives at
sqrt(2 g 0.15) = 1.72 m/s and its reset energy is g * 0.15 = 1.4715 J/kg. That
minimum is what makes ``ball_not_lifted`` inert for drop episodes: it is 0.4715
J/kg above JUGGLE_LIFT_ENERGY_THRESHOLD_J_PER_KG. At 0.25 m the arrival speed is
2.21 m/s.
"""

JUGGLE_DROP_EXTRA_TIME_S = (0.0, 0.10)
"""Extra fall time already applied to a drop release, in seconds, uniform.

The position and velocity are advanced together by this time, so the sample is
exactly the state of a ball released at height h, t seconds earlier::

    z = h - 0.5 g t^2,    v_z = -g t

The extra time changes the arrival speed without changing the height
distribution: at t = 0.10 s the ball has already fallen 49 mm and is moving at
0.981 m/s. The range ends at 0.10 s so the lowest effective release height stays
above 0.10 m and the two release types remain separated by more than the ball
radius.
"""

JUGGLE_DROP_LATERAL_RANGE_M = 0.03
"""Half-width of the in-plane offset of a drop start, in metres.

30 mm is about a quarter of the plate's 117 mm short half-width and close to the
ball radius. The ball therefore meets a different string point every episode but
still lands well inside the strings, not on the frame or the edge.
"""

JUGGLE_DROP_LATERAL_SPEED_RANGE_MPS = 0.02
"""Half-width of the in-plane release speed of a drop start, in m/s.

20 mm/s is 1 % of the 1.98 m/s arrival speed of a 0.20 m drop. It tilts the
bounce enough to decorrelate the episodes without turning the release into a
throw at the plate.
"""

JUGGLE_ARM_RESET_RANGE_RAD = 0.05
"""Half-width of the right-arm and waist joint offsets at reset, in radians.

0.05 rad is 2.9 deg, which moves the plate centre by a few centimetres. That is
enough to sample a different strike geometry every episode, and small enough
that the reset state is still the measured bat-carry pose. The right arm and the
waist are the joints that aim the plate; perturbing the legs would move the base
and change the energy datum, and perturbing the left arm or head would change the
camera view without changing the strike.
"""

JUGGLE_RESET_JOINT_NAMES = (
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_yaw_joint",
  "right_wrist_pitch_joint",
  "right_wrist_roll_joint",
  "waist_yaw_joint",
  "waist_pitch_joint",
  "waist_roll_joint",
)
"""Joints that receive the reset offset; every other joint stays at its default."""

JUGGLE_NO_LIFT_TIMEOUT_S = 1.5
"""Control time after reset that must pass before ``ball_not_lifted`` can fire.

1.5 s is 75 control steps at the 50 Hz control rate. A 0.25 m drop reaches the
strings in sqrt(2 h / g) = 0.226 s, so 1.5 s leaves ample time for a strike or a
first passive bounce to lift the ball. The limit is short enough that a policy
which never lifts the ball does not spend the whole 20 s episode on it.
"""

JUGGLE_NO_BOUNCE_TIMEOUT_S = 2.5
"""Control time without a racket-ball impact before the juggle episode ends.

The deadline is measured from the last impact, not from reset. 2.5 s is 4.4
flight times of a 0.40 m bounce, and the window only expires on sustained
contact: a missed bounce ends the episode through ``ball_dropped`` 0.5 s later,
so 2.5 s of airborne misses cannot accrue. The timeout is greater than
:data:`JUGGLE_NO_LIFT_TIMEOUT_S` (1.5 s), so this term cannot pre-empt
``ball_not_lifted`` on a plate start that is never lifted. The cap is the
point: in the last ablation the parked arms collected about 4.5 reward per
second for 19.4 s with 2.5 impacts per episode, and the episode reward is
divided by the 20 s limit, so duration multiplied their score.
"""

JUGGLE_LIFT_ENERGY_THRESHOLD_J_PER_KG = 1.0
"""Ball energy that counts as lifted, in J/kg, shared by the termination and metric.

The energy is ``E = 0.5 * v_z^2 + g * (z - JUGGLE_ENERGY_BALL_REST_Z)``, the same
formula the reward pays. 1.0 J/kg is 1.41 m/s of vertical speed at the resting
height, or 102 mm of height with zero speed: a real strike, not contact jitter.
A drop start has ``E = g * h >= g * 0.15 = 1.4715`` J/kg at release by
construction, a margin of 0.4715 J/kg above this threshold, so ``ball_not_lifted``
cannot fire on a drop episode. The term only rejects a plate episode in which the
policy never lifted the ball.

A plate start is not exactly at ``E = 0``, because the energy datum is a fixed
world height: the perturbed arm can raise the plate, and ``E = g (z - rest_z)``
counts that height. Measured over 612 plate samples at
``JUGGLE_ARM_RESET_RANGE_RAD = 0.05``, first-step plate energy reached 0.741 J/kg,
only 0.259 J/kg below this threshold. Re-measure that margin whenever
``JUGGLE_ARM_RESET_RANGE_RAD``, the carry-pose height or
``JUGGLE_ENERGY_BALL_REST_Z`` changes: this threshold is coupled to the arm range.
"""


# Lowest foot contact point may be this far below z=0 before the pose counts as
# penetrating the ground.
_FOOT_LOW_Z_TOLERANCE = 1e-6


def measure_bat_carry_state(
  base_height: float | None = None,
  drop_height: float = 0.0,
) -> tuple[float, np.ndarray]:
  """Forward-kinematics probe for the bat-carry pose.

  Returns ``(base_height, ball_centre)``: the pelvis height at which the lowest
  foot contact point sits :data:`BAT_CARRY_GROUND_CLEARANCE` above z=0, and the
  ball centre that rests on the face centre along the face normal with
  :data:`BALL_SPAWN_CLEARANCE` of slack, raised a further ``drop_height`` along
  that normal.  ``drop_height=0`` reproduces :data:`BALL_SPAWN_LOCAL_POS`, the
  on-bed reference; :data:`BALL_SPAWN_DROP_HEIGHT` reproduces
  :data:`BALL_SPAWN_INIT_POS`, where the ball actually starts.

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
  ball_center = face_pos + face_normal * (
    BALL_RADIUS + BALL_SPAWN_CLEARANCE + drop_height
  )
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


def get_tennis_ball_cfg(drop_height: float = 0.0) -> EntityCfg:
  """Ball entity, spawned on or above the bat in the bat-carry pose.

  ``drop_height=0.0`` (the default) puts the ball on the bed, which is what the
  carry task needs.  The juggle task passes :data:`BALL_SPAWN_DROP_HEIGHT` so the
  ball falls in and a bounce exists from the first step; see that constant.

  ``init_state.pos`` is in the environment frame (it is added to
  ``env_origins``), so it only lines up with the bed when the robot's own reset
  places the root at :data:`BAT_CARRY_BASE_HEIGHT` with zero yaw. The tennis
  task therefore pins the base reset — see
  ``agibot_x2_tennis_ball_env_cfg``.
  """
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=ball_spawn_init_pos(drop_height),
      rot=(1.0, 0.0, 0.0, 0.0),
      lin_vel=(0.0, 0.0, 0.0),
      ang_vel=(0.0, 0.0, 0.0),
    ),
    spec_fn=get_tennis_ball_spec,
  )


__all__ = [
  "BALL_RADIUS",
  "BALL_SPAWN_CLEARANCE",
  "BALL_SPAWN_DROP_HEIGHT",
  "BALL_SPAWN_FACE_NORMAL",
  "BALL_SPAWN_INIT_POS",
  "BALL_SPAWN_LOCAL_POS",
  "ball_spawn_init_pos",
  "BAT_CARRY_BASE_HEIGHT",
  "BAT_CARRY_POSE",
  "BED_GEOM",
  "BED_SOLREF",
  "CLAMP_BODY",
  "FACE_SITE",
  "JUGGLE_ARM_RESET_RANGE_RAD",
  "JUGGLE_DROP_EXTRA_TIME_S",
  "JUGGLE_DROP_HEIGHT_RANGE_M",
  "JUGGLE_DROP_LATERAL_RANGE_M",
  "JUGGLE_DROP_LATERAL_SPEED_RANGE_MPS",
  "JUGGLE_ENERGY_BALL_REST_Z",
  "JUGGLE_ENERGY_SIGMA_J_PER_KG",
  "JUGGLE_ENERGY_TARGET_HEIGHT_M",
  "JUGGLE_ENERGY_WEIGHT",
  "JUGGLE_LIFT_ENERGY_THRESHOLD_J_PER_KG",
  "JUGGLE_NO_BOUNCE_TIMEOUT_S",
  "JUGGLE_NO_LIFT_TIMEOUT_S",
  "JUGGLE_PLATE_START_FRACTION",
  "JUGGLE_PLATE_START_LATERAL_RANGE_M",
  "JUGGLE_RACKET_LIFT_DEADBAND_M",
  "JUGGLE_RACKET_LIFT_PLATE_TOP_REST_Z",
  "JUGGLE_RACKET_LIFT_WEIGHT",
  "JUGGLE_RESET_JOINT_NAMES",
  "RACKET_BODY",
  "TENNIS_BED_COLLISION",
  "WRIST_BODY",
  "attach_tennis_assembly",
  "get_tennis_ball_cfg",
  "get_tennis_ball_spec",
  "get_tennis_spec",
  "get_x2_tennis_robot_cfg",
  "juggle_racket_lift_weight",
  "measure_bat_carry_state",
]


# --- S7 delayed release -------------------------------------------------------
#
# The operator holds the ball over the racket and lets go after a wait. These are
# the first delayed-release rung's values: a fixed 0.40 m hold, so the release energy
# equals the 0.40 m energy target and any change is attributable to waiting rather
# than to a different drop energy. Widening the hold height is a separate rung.
JUGGLE_HOLD_HEIGHT_M = 0.40
"""Ball-centre height above its resting state while held, in metres."""

JUGGLE_HOLD_DELAY_S = (0.0, 2.0)
"""Uniform hold duration before release, in seconds. Includes 0.0 on purpose: an
immediate release must stay in the distribution, or stillness becomes a
precondition and the policy can learn a countdown instead of watching the ball."""

JUGGLE_HOLD_OFFSET_SPEED_MPS = 0.05
"""Metres of world x/y displacement per second of hold, for the offset rung.

The ball is placed ``offset_speed * hold_duration`` away from the nominal hold
point along one uniformly drawn direction, held there, then released. Coupling the
distance to the duration keeps the required reach speed constant at 5 cm/s: the
target is further away exactly when more time exists to reach it. Over a uniform
0-6 s hold the displacement is uniform on 0-0.30 m, mean 0.15 m, against a plate
half-extent of 0.157 x 0.117 m, so most offsets put the ball clear of the plate and
the plate has to travel.
"""

JUGGLE_HOLD_LONG_DELAY_S = (0.0, 6.0)
"""Uniform hold duration for the long-wait rung, in seconds.

An episode is 20 s, and episode reward is ``sum(raw * weight * dt) /
max_episode_length_s``, so every second spent holding is a second that earns
nothing and cannot be recovered. A 6 s upper bound gives a mean wait of 3 s, which
still leaves 14 s of juggling and 70 % of the reachable return. The hold phases are
the only place a permanently parked policy could hide, so the bound is a reward
argument as much as a curriculum one: at 10 s the ceiling is half, and past that
the rung measures endurance rather than waiting."""

JUGGLE_HOLD_LATERAL_RANGE_M = 0.02
"""Half-width of the world x/y hold offset, in metres. The same +-20 mm spread the
S6 release uses, centred on the nominal bed point."""

JUGGLE_WAIT_ALIGN_WEIGHT = 0.4
"""Weight of the waiting-alignment reward ``racket_under_ball``. Paid only while
the ball is held. Kept below the juggling terms: waiting is a means, not the task."""

JUGGLE_WAIT_PLATEAU_M = 0.010
"""Plate-frame offset inside which waiting alignment is exactly 1, in metres."""

JUGGLE_WAIT_SIGMA_M = 0.030
"""Falloff of the waiting alignment beyond the plateau, in metres."""

JUGGLE_WAIT_LEVEL_POWER = 2.0
"""Exponent on the plate normal's world-z component in the waiting alignment."""

JUGGLE_WAIT_SPEED_SIGMA_MPS = 0.5
"""Scale of the quiet factor in the waiting alignment, in metres per second."""
