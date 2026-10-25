"""Tests for the X2 tennis-ball-on-the-bat velocity task.

The task adds two things to the shipped X2 no-state-estimation velocity task: the
racket assembly ("bat"), rigidly attached to the right wrist, and a
free-floating tennis ball spawned on the string bed. Three claims need evidence,
and they are tested separately here:

1. **Asset fidelity.** The bat's masses, inertias and transforms are the
   reference snapshot's, and the string bed keeps the calibrated contact
   parameters through mjlab's collision-config pipeline. The shipped
   ``FULL_COLLISION`` zeroes ``contype``/``conaffinity`` on every geom its
   patterns do not match, so a silent failure mode is a bed that no longer
   collides at all.
2. **Contact calibration.** The negative-``solref`` bed reproduces the user's
   measured ball rebounds at a 1 ms physics step, with no lateral drift. The
   same parameters inject energy at mjlab's default 5 ms step, which is why the
   task changes the timestep.
3. **Unchanged contract.** The task keeps the base task's action width,
   observation widths, joints and sensors; only the reset poses are pinned.

What these tests do **not** show: that a ball stays on the bat. Open-loop PD
cannot hold the bat-carry pose (the reference README says the same about its own
default pose), so in-environment retention needs a trained policy.
"""

from __future__ import annotations

import re

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.agibot_x2.tennis_constants import (
  BALL_COMMAND_STAGES,
  BALL_COMMAND_WINDOW,
  BALL_DIAGINERTIA,
  BALL_DROP_GRACE_S,
  BALL_FALL_THRESHOLD,
  BALL_MASS,
  BALL_POSITION_NOISE_M,
  BALL_RADIUS,
  BALL_SPAWN_CLEARANCE,
  BALL_SPAWN_LOCAL_POS,
  BALL_VELOCITY_NOISE_M,
  BAT_CARRY_BASE_HEIGHT,
  BAT_CARRY_POSE,
  BED_CENTER_X,
  BED_GEOM,
  BED_HALF_LENGTH,
  BED_HALF_THICKNESS,
  BED_HALF_WIDTH,
  BED_PRIORITY,
  BED_SOLIMP,
  BED_SOLREF,
  CLAMP_BODY,
  CLAMP_DIAGINERTIA,
  CLAMP_MASS,
  CONNECTOR_DIAGINERTIA,
  CONNECTOR_MASS,
  CONTACT_CONDIM,
  CONTACT_FRICTION,
  FACE_SITE,
  FACE_SITE_POS,
  RACKET_BODY,
  RACKET_DIAGINERTIA,
  RACKET_MASS,
  ROLLING_FRICTION,
  SLIDING_FRICTION,
  SPIN_FRICTION,
  TENNIS_BED_COLLISION,
  TENNIS_MESH_DIR,
  TENNIS_MESH_SHA256,
  WRIST_BODY,
  get_tennis_ball_cfg,
  get_tennis_ball_spec,
  get_tennis_spec,
  get_x2_tennis_robot_cfg,
  measure_bat_carry_state,
  mesh_sha256,
)
from mjlab.asset_zoo.robots.agibot_x2.x2_constants import (
  actuator_group_index,
  get_spec,
)
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.agibot_x2.env_cfgs import (
  agibot_x2_flat_velocity_env_cfg,
)
from mjlab.tasks.velocity.config.agibot_x2.tennis_ball_env_cfg import (
  BALL_ENTITY,
  BAT_BALL_CONTACT_SENSOR,
  SIM_TO_REAL_EVENT_NAMES,
  TENNIS_PHYSICS_TIMESTEP,
  TENNIS_SUBSTEPS,
)

TASK_ID = "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Balance"
BASE_TASK_ID = "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation"

# Reference rebound measurements (ball bottom above the string bed), captured
# against a stationary horizontal bed and recorded in the asset snapshot's
# ``contact.json``. Tolerance is 30 mm: the fit is approximate by construction
# and the snapshot's own note calls the friction and racket inertia provisional.
MEASURED_REBOUND = {0.20: 0.17, 0.30: 0.23, 0.40: 0.31, 0.50: 0.42}
REBOUND_TOLERANCE = 0.03


##
# Isolated fixtures: one stationary horizontal string bed plus one ball.
# Built from the shipped constants (and the shipped ball spec) so a parameter
# change that is not re-fitted fails the test instead of silently changing
# contact behaviour.
##


def _bed_fixture_spec(
  timestep: float, rolling_friction: float | None = None
) -> mujoco.MjSpec:
  """Isolated level bed plus the shipped ball.

  ``rolling_friction`` overrides the third friction number on **both** geoms so
  the behavioural test can run the identical setup with rolling friction off.
  Both rows are changed because the shipped rows are deliberately identical;
  only the bed's ``priority=1`` makes it the row that governs. ``None`` keeps
  the shipped :data:`CONTACT_FRICTION`.
  """
  spec = get_tennis_ball_spec()
  spec.option.timestep = timestep
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
  spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
  spec.option.iterations = 100
  spec.option.tolerance = 1e-8
  spec.option.ls_iterations = 50
  spec.option.ls_tolerance = 0.01

  friction = list(CONTACT_FRICTION)
  if rolling_friction is not None:
    friction[2] = rolling_friction
    spec.geom("ball_collision").friction = friction

  angle = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
  vertices = np.concatenate(
    [
      np.c_[
        BED_HALF_LENGTH * np.cos(angle),
        BED_HALF_WIDTH * np.sin(angle),
        np.full(32, z),
      ]
      for z in (-BED_HALF_THICKNESS, BED_HALF_THICKNESS)
    ]
  )
  spec.add_mesh(name="bed_mesh", uservert=vertices.ravel().tolist())
  bed = spec.worldbody.add_body(name="bed")
  bed.add_geom(
    name=BED_GEOM,
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="bed_mesh",
    contype=1,
    conaffinity=1,
    condim=CONTACT_CONDIM,
    priority=BED_PRIORITY,
    mass=0.0,
    friction=friction,
    solref=list(BED_SOLREF),
    solimp=list(BED_SOLIMP),
  )
  return spec


def _ball_qpos_and_dof(model: mujoco.MjModel) -> tuple[int, int]:
  joint = model.joint("ball_free")
  return int(np.ravel(joint.qposadr)[0]), int(np.ravel(joint.dofadr)[0])


def _settle_ball(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  """Put the ball on the bed and let the compliant contact take its weight."""
  qadr, _ = _ball_qpos_and_dof(model)
  data.qpos[:] = 0.0
  data.qpos[qadr + 3] = 1.0
  data.qpos[qadr + 2] = BED_HALF_THICKNESS + BALL_RADIUS
  mujoco.mj_forward(model, data)
  for _ in range(round(0.1 / model.opt.timestep)):
    mujoco.mj_step(model, data)
  assert data.ncon, "the ball never settled on the bed"


def _drop_and_measure(timestep: float, drop: float) -> tuple[float, float, float]:
  """Apex height, maximum lateral speed, and lowest point for a dropped ball."""
  model = _bed_fixture_spec(timestep).compile()
  data = mujoco.MjData(model)
  qadr, vadr = _ball_qpos_and_dof(model)
  data.qpos[:] = 0.0
  data.qpos[qadr + 3] = 1.0
  # Bed top surface is at +BED_HALF_THICKNESS; the ball starts resting on it plus
  # ``drop`` metres.
  data.qpos[qadr + 2] = BED_HALF_THICKNESS + BALL_RADIUS + drop
  mujoco.mj_forward(model, data)

  trace: list[float] = []
  min_bottom = np.inf
  max_lateral = 0.0
  first_contact = None
  for step in range(int(1.0 / timestep)):
    mujoco.mj_step(model, data)
    trace.append(float(data.qpos[qadr + 2]))
    if data.ncon and first_contact is None:
      first_contact = step
    min_bottom = min(min_bottom, float(data.qpos[qadr + 2]) - BALL_RADIUS)
    max_lateral = max(
      max_lateral,
      float(abs(data.qvel[vadr + 0])),
      float(abs(data.qvel[vadr + 1])),
    )
  assert first_contact is not None, "the ball never touched the bed"
  # The rebound apex is reached in free flight, after the contact has ended, so
  # it cannot be sampled only on contact steps.
  apex = max(trace[first_contact:])
  return apex - BED_HALF_THICKNESS - BALL_RADIUS, max_lateral, min_bottom


def _roll_and_measure(
  timestep: float,
  rolling_friction: float | None,
  release_speed: float,
  duration: float = 1.0,
) -> tuple[float, float, float]:
  """Release the ball rolling without slipping; return speeds and travel.

  ``omega_y = v / R`` is the zero-slip relation for travel along +x: the
  contact point sits at ``-R z`` from the centre, so ``v - omega R = 0``. The
  first returned speed is sampled immediately after the release, the second one
  ``duration`` seconds later, and the third is the distance travelled.
  """
  model = _bed_fixture_spec(timestep, rolling_friction=rolling_friction).compile()
  data = mujoco.MjData(model)
  qadr, vadr = _ball_qpos_and_dof(model)
  _settle_ball(model, data)
  release_x = float(data.qpos[qadr])

  data.qvel[vadr + 0] = release_speed
  data.qvel[vadr + 4] = release_speed / BALL_RADIUS
  mujoco.mj_forward(model, data)
  start = float(data.qvel[vadr + 0])
  for _ in range(round(duration / timestep)):
    mujoco.mj_step(model, data)
  return start, float(data.qvel[vadr + 0]), float(data.qpos[qadr]) - release_x


##
# 1. Asset fidelity.
##


def test_shipped_meshes_match_the_pinned_provenance() -> None:
  assert set(TENNIS_MESH_SHA256) == {
    "clamp_a.stl",
    "clamp_b.stl",
    "racket_visual.stl",
    "right_hand_connector.stl",
  }
  for name, expected in TENNIS_MESH_SHA256.items():
    assert (TENNIS_MESH_DIR / name).is_file()
    assert mesh_sha256(name) == expected, f"{name} bytes changed"


def test_bat_masses_and_inertias_match_the_reference() -> None:
  model = get_tennis_spec().compile()

  def body(name: str):
    body_id = int(np.ravel(model.body(name).id)[0])
    return (
      float(model.body_mass[body_id]),
      tuple(np.round(model.body_ipos[body_id], 9)),
      tuple(np.round(model.body_inertia[body_id], 9)),
    )

  wrist_mass, wrist_ipos, wrist_inertia = body(WRIST_BODY)
  # The white connector *replaces* the hand, so the wrist's own mass and
  # inertia are the connector's, not the hand's 0.303847 kg.
  assert wrist_mass == pytest.approx(CONNECTOR_MASS)
  assert wrist_ipos == pytest.approx((0.0, 0.0, -0.02))
  assert wrist_inertia == pytest.approx(CONNECTOR_DIAGINERTIA, rel=1e-6)

  clamp = body(CLAMP_BODY)
  assert clamp[0] == pytest.approx(CLAMP_MASS)
  assert clamp[1] == pytest.approx((0.0252336, 0.0143531, 0.03))
  assert clamp[2] == pytest.approx(CLAMP_DIAGINERTIA, rel=1e-6)

  racket = body(RACKET_BODY)
  assert racket[0] == pytest.approx(RACKET_MASS)
  assert racket[1] == pytest.approx((0.4, 0.0, 0.0))
  assert racket[2] == pytest.approx(RACKET_DIAGINERTIA, rel=1e-6)


def test_bat_adds_no_joints_and_no_actuators() -> None:
  plain = get_spec().compile()
  bat = get_tennis_spec().compile()
  assert bat.nu == plain.nu
  assert bat.njnt == plain.njnt
  assert bat.nq == plain.nq
  assert bat.nv == plain.nv
  assert bat.nbody == plain.nbody + 2  # racket_clamp + tennis_racket


def test_face_site_sits_on_the_racket_at_the_bed_top() -> None:
  model = get_tennis_spec().compile()
  site_id = int(np.ravel(model.site(FACE_SITE).id)[0])
  assert model.body(int(np.ravel(model.site_bodyid[site_id])[0])).name == RACKET_BODY
  assert tuple(np.round(model.site_pos[site_id], 9)) == pytest.approx(FACE_SITE_POS)
  assert FACE_SITE_POS[0] == pytest.approx(BED_CENTER_X)
  assert FACE_SITE_POS[2] == pytest.approx(BED_HALF_THICKNESS)


def test_only_the_string_bed_collides() -> None:
  """Every other tennis geom must stay visual-only, or the fit is invalid."""
  model = get_tennis_spec().compile()
  colliding = []
  for geom_id in range(model.ngeom):
    name = model.geom(geom_id).name
    if not any(key in name for key in ("tennis_", "racket_", "string_")):
      continue
    if model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]:
      colliding.append(name)
  assert colliding == [BED_GEOM]


def test_bed_survives_the_shipped_collision_policy() -> None:
  """FULL_COLLISION disables unmatched geoms; the bed must be re-enabled after."""
  entity = Entity(get_x2_tennis_robot_cfg())
  model = entity.spec.compile()
  geom_id = int(np.ravel(model.geom(BED_GEOM).id)[0])
  assert model.geom_contype[geom_id] == 1
  assert model.geom_conaffinity[geom_id] == 1
  assert model.geom_condim[geom_id] == CONTACT_CONDIM
  assert model.geom_priority[geom_id] == BED_PRIORITY
  assert tuple(model.geom_friction[geom_id]) == pytest.approx(CONTACT_FRICTION)
  assert tuple(model.geom_solref[geom_id]) == pytest.approx(BED_SOLREF)
  assert tuple(model.geom_solimp[geom_id]) == pytest.approx(BED_SOLIMP)
  # The repair pass must not disturb the robot's own collision story: the
  # original bed-less geoms are still off (visual) and the feet still collide.
  feet = [
    geom_id
    for geom_id in range(model.ngeom)
    if model.geom(geom_id).name.startswith(("left_foot", "right_foot"))
  ]
  assert feet, "no foot collision geoms found"
  assert all(model.geom_contype[geom_id] == 1 for geom_id in feet)


def test_bed_collision_config_is_the_last_word() -> None:
  """The repair config must not disable geoms the shipped policy enabled."""
  assert TENNIS_BED_COLLISION.disable_other_geoms is False
  assert TENNIS_BED_COLLISION.geom_names_expr == (rf"^{BED_GEOM}$",)


def test_contact_is_six_dimensional_with_the_declared_friction() -> None:
  """``condim=6`` is what lets a rolling ball lose energy to the bed.

  ``condim=3`` carries a normal direction and two sliding directions and no
  rotation at all, so spin and rolling friction cannot act: a ball rolling on
  the level bed kept exactly the same speed forever. The bed's ``priority=1``
  makes its row the one that governs every bed contact, which is why both rows
  are declared identical rather than merely similar.
  """
  assert CONTACT_CONDIM == 6
  assert CONTACT_FRICTION == (SLIDING_FRICTION, SPIN_FRICTION, ROLLING_FRICTION)
  # MuJoCo geom friction is exactly three numbers: sliding, spin, rolling.
  assert len(CONTACT_FRICTION) == 3
  assert SLIDING_FRICTION == pytest.approx(0.44)

  robot = Entity(get_x2_tennis_robot_cfg()).spec.compile()
  bed = int(np.ravel(robot.geom(BED_GEOM).id)[0])
  assert robot.geom_condim[bed] == 6
  assert tuple(robot.geom_friction[bed]) == pytest.approx(CONTACT_FRICTION)
  assert robot.geom_priority[bed] == 1

  ball = get_tennis_ball_cfg().spec_fn().compile()
  ball_geom = int(np.ravel(ball.geom("ball_collision").id)[0])
  assert ball.geom_condim[ball_geom] == 6
  assert tuple(ball.geom_friction[ball_geom]) == pytest.approx(CONTACT_FRICTION)
  # Lower priority than the bed, so the bed's row is the one that acts.
  assert ball.geom_priority[ball_geom] == 0


def test_ball_inertia_is_the_measured_shell_value() -> None:
  """A tennis ball is a thin shell; MuJoCo's sphere default is a solid lump."""
  model = get_tennis_ball_cfg().spec_fn().compile()
  body = int(np.ravel(model.body("ball").id)[0])
  mass = float(model.body_mass[body])
  inertia = np.asarray(model.body_inertia[body], dtype=float)

  # One kilogram, once: the geom carries no mass of its own.
  assert tuple(inertia) == pytest.approx(BALL_DIAGINERTIA, rel=1e-9)
  assert mass == pytest.approx(BALL_MASS)

  # The ball is the ITF lower limit, a conservative estimate rather than a
  # measurement of the ball in hand. If someone substitutes a regulation midpoint
  # ball this test is meant to stop and make them re-check the rolling fit, which
  # depends on I/(m R^2).
  assert mass == pytest.approx(0.0560, abs=1e-9)
  assert 2 * BALL_RADIUS == pytest.approx(0.0654, abs=1e-6)

  ratio = inertia[0] / (mass * BALL_RADIUS**2)
  # 0.400 would be a uniform solid sphere, which is what MuJoCo would compute.
  # Tolerance matches the five significant figures the constant is stored at.
  assert ratio == pytest.approx(0.549, rel=1e-4), ratio
  assert ratio > 1.3 * 0.400
  # Cross measured 3.41e-05 kg m^2 for a 57 g ball at 33 mm, i.e. this ratio.
  # The inertia is re-derived at the shipped mass and radius, so the ratio is
  # preserved and the absolute value is not.
  assert inertia[0] == pytest.approx(0.549 * 0.0560 * 0.0327**2, rel=1e-4)


def test_carry_pose_constants_are_current() -> None:
  """``BAT_CARRY_BASE_HEIGHT`` and the spawn point are measured, not assumed."""
  height, ball_centre = measure_bat_carry_state()
  assert height == pytest.approx(BAT_CARRY_BASE_HEIGHT, abs=1e-9)
  assert tuple(ball_centre) == pytest.approx(BALL_SPAWN_LOCAL_POS, abs=1e-9)


def test_ball_spawns_one_millimetre_above_the_bed() -> None:
  """Geometric check against the compiled scene geometry, not the constants."""
  model = Entity(get_x2_tennis_robot_cfg()).spec.compile()
  data = mujoco.MjData(model)
  data.qpos[:] = model.qpos0
  free = int(np.ravel(model.joint("floating_base_joint").qposadr)[0])
  data.qpos[free : free + 3] = [0.0, 0.0, BAT_CARRY_BASE_HEIGHT]
  data.qpos[free + 3 : free + 7] = [1.0, 0.0, 0.0, 0.0]
  for name, value in BAT_CARRY_POSE.items():
    data.qpos[int(np.ravel(model.joint(name).qposadr)[0])] = value
  mujoco.mj_forward(model, data)

  geom_id = int(np.ravel(model.geom(BED_GEOM).id)[0])
  mesh_id = model.geom_dataid[geom_id]
  vert_adr = int(np.ravel(model.mesh_vertadr[mesh_id])[0])
  vert_num = int(np.ravel(model.mesh_vertnum[mesh_id])[0])
  verts = model.mesh_vert[vert_adr : vert_adr + vert_num].copy()
  rotation = data.geom_xmat[geom_id].reshape(3, 3)
  world = verts @ rotation.T + data.geom_xpos[geom_id]
  extent = world.max(axis=0) - world.min(axis=0)
  bed_centre = (world.max(axis=0) + world.min(axis=0)) / 2.0
  # The bed must be a thin horizontal plate: the bat is level in the carry pose,
  # so the ball has no slope to roll down. The plate's thinnest world extent must
  # be the vertical one, and the surface normal from an SVD of the vertex cloud
  # must be vertical too (the mesh's own axes are permuted by the racket frame,
  # so the geom's local axes cannot be read off positionally).
  assert np.argmin(extent) == 2
  assert extent[2] == pytest.approx(2.0 * BED_HALF_THICKNESS, abs=1e-4)
  _, _, vh = np.linalg.svd(world - world.mean(axis=0))
  normal = vh[2]
  assert abs(normal[2]) == pytest.approx(1.0, abs=1e-4)

  ball_bottom = BALL_SPAWN_LOCAL_POS[2] - BALL_RADIUS
  bed_top = world[:, 2].max()
  assert ball_bottom - bed_top == pytest.approx(BALL_SPAWN_CLEARANCE, abs=1e-5)
  # And it must be over the plate, not hanging off the edge.
  assert abs(BALL_SPAWN_LOCAL_POS[0] - bed_centre[0]) < 0.01
  assert abs(BALL_SPAWN_LOCAL_POS[1] - bed_centre[1]) < 0.01
  assert extent[0] > 2.0 * BALL_RADIUS
  assert extent[1] > 2.0 * BALL_RADIUS


##
# 2. Contact calibration.
##


@pytest.mark.parametrize("drop", sorted(MEASURED_REBOUND))
def test_bed_reproduces_the_measured_rebound_at_one_millisecond(
  drop: float,
) -> None:
  apex, max_lateral, _ = _drop_and_measure(1e-3, drop)
  assert apex == pytest.approx(MEASURED_REBOUND[drop], abs=REBOUND_TOLERANCE)
  # A calibrated normal-impact fixture must not create tangential motion.
  assert max_lateral < 1e-3


def test_a_rolling_ball_slows_down_and_does_not_stop_instantly() -> None:
  """A level-bed roll must decay, and must not be arrested on the spot.

  The bounds are deliberately loose in absolute terms, because the coefficient
  itself is provisional and will change when the real racket is measured. What
  the test pins is the *shape*: rotation must let the bed take energy, and the
  ball must not be brought to a halt within a tenth of a second, which a badly
  over-large coefficient would do.

  ``condim=3`` fails the decay bound: the speed never changed at all.
  """
  start, after, travel = _roll_and_measure(1e-3, None, 0.10)
  assert start == pytest.approx(0.10, abs=1e-6)
  ratio = after / start
  assert 0.05 <= ratio <= 0.95, (
    f"speed after 1 s is {ratio:.3f} of the release (travel {travel:.3f} m)"
  )
  early, _, _ = _roll_and_measure(1e-3, None, 0.10, duration=0.1)
  assert early / start > 0.5, f"arrested within 0.1 s: {early / start:.3f} of release"


def test_zero_rolling_friction_does_not_slow_the_ball() -> None:
  """The same setup with the rolling coefficient removed must coast.

  This is the control: it shows the slowdown above is caused by the rolling
  dimension of the friction triple, not by the sliding coefficient or by the
  solver. MuJoCo clamps the coefficient to ``mjMINMU`` = 1e-5, so a little decay
  remains -- far less than the shipped value produces.
  """
  start, after, travel = _roll_and_measure(1e-3, 0.0, 0.10)
  assert after / start > 0.99
  assert travel > 0.09


def test_bed_parameters_are_unstable_at_the_default_timestep() -> None:
  """Documents why the tennis task runs at 1 ms instead of mjlab's 5 ms.

  ``BED_SOLREF`` is MuJoCo's direct stiffness/damping form, fitted at 1 ms. At
  5 ms the same numbers feed energy into the contact: a 0.30 m drop rebounds far
  higher than it fell from. This is a property of the parameters, not a bug in
  the task, and it is why ``TENNIS_PHYSICS_TIMESTEP`` must not be raised.
  """
  apex, _, _ = _drop_and_measure(5e-3, 0.30)
  assert apex > 0.6, f"expected gross over-rebound at 5 ms, got {apex:.3f} m"


##
# 3. Task wiring.
##


def test_task_is_registered() -> None:
  assert TASK_ID in list_tasks()


def test_task_scene_and_simulation_contract() -> None:
  cfg = load_env_cfg(TASK_ID)
  assert set(cfg.scene.entities) == {"robot", BALL_ENTITY}
  assert cfg.sim.mujoco.timestep == pytest.approx(TENNIS_PHYSICS_TIMESTEP)
  assert cfg.decimation == TENNIS_SUBSTEPS
  # 50 Hz control, as the reference controller.
  assert cfg.sim.mujoco.timestep * cfg.decimation == pytest.approx(0.02)
  assert cfg.scene.sensors is not None
  assert BAT_BALL_CONTACT_SENSOR in {s.name for s in cfg.scene.sensors}


##
# 4. Servo configuration: the multi-motion tracking task's, not the velocity task's.
#
# The tennis task's arm carries 0.37 kg of tooling, so the servo stiffness decides
# how far the bat sags under that load and how well it damps a ball impact. The
# multi-motion tracking task's gains are the ones with hardware trials behind
# them, so the tennis task adopts them wholesale and the two must not drift apart.
##

TRACKING_TASK_ID = "Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion"


def _compiled_robot(task_id: str) -> tuple[mujoco.MjModel, dict[str, float]]:
  """Compiled robot model plus the task's per-joint action scale map."""
  cfg = load_env_cfg(task_id)
  model = Entity(cfg.scene.entities["robot"]).spec.compile()
  return model, cfg.actions["joint_pos"].scale


def _per_joint(model: mujoco.MjModel, scale: dict[str, float]) -> dict[str, dict]:
  """Servo, limit and action-scale values keyed by joint name."""
  rows: dict[str, dict] = {}
  for actuator_id in range(model.nu):
    joint_id = int(model.actuator_trnid[actuator_id][0])
    name = model.joint(joint_id).name
    dof = int(np.ravel(model.jnt_dofadr[joint_id])[0])
    rows[name] = {
      "kp": float(model.actuator_gainprm[actuator_id][0]),
      "kd": -float(model.actuator_biasprm[actuator_id][2]),
      "effort": float(np.max(np.abs(model.actuator_forcerange[actuator_id])))
      or float(model.actuator_ctrlrange[actuator_id][1]),
      "armature": float(model.dof_armature[dof]),
      "frictionloss": float(model.dof_frictionloss[dof]),
      "lower": float(model.jnt_range[joint_id][0]),
      "upper": float(model.jnt_range[joint_id][1]),
      "scale": float(next(v for k, v in scale.items() if re.fullmatch(k, name))),
    }
  return rows


def test_servo_gains_are_the_multi_motion_tracking_tasks() -> None:
  tennis_model, tennis_scale = _compiled_robot(TASK_ID)
  track_model, track_scale = _compiled_robot(TRACKING_TASK_ID)
  tennis = _per_joint(tennis_model, tennis_scale)
  tracking = _per_joint(track_model, track_scale)
  assert tennis.keys() == tracking.keys()
  mismatched = {
    name: (
      tennis[name]["kp"],
      tennis[name]["kd"],
      tracking[name]["kp"],
      tracking[name]["kd"],
    )
    for name in tennis
    if (tennis[name]["kp"], tennis[name]["kd"])
    != (tracking[name]["kp"], tracking[name]["kd"])
  }
  assert not mismatched, mismatched


def test_joint_and_torque_limits_are_the_multi_motion_tracking_tasks() -> None:
  """The limits were already identical between the two tasks; pin that."""
  tennis_model, tennis_scale = _compiled_robot(TASK_ID)
  track_model, track_scale = _compiled_robot(TRACKING_TASK_ID)
  tennis = _per_joint(tennis_model, tennis_scale)
  tracking = _per_joint(track_model, track_scale)
  for field in ("effort", "lower", "upper", "armature", "frictionloss", "scale"):
    differing = {
      name: (tennis[name][field], tracking[name][field])
      for name in tennis
      if tennis[name][field] != tracking[name][field]
    }
    assert not differing, f"{field} differs: {differing}"


def test_tennis_gains_do_not_leak_into_other_tasks() -> None:
  """The vendor table deep-copies the articulation; guard that.

  ``get_x2_robot_cfg`` shares one module-level articulation between every task,
  so an in-place edit would silently change the velocity, tracking and recovery
  tasks too.
  """
  base_model, base_scale = _compiled_robot(BASE_TASK_ID)
  tennis_model, tennis_scale = _compiled_robot(TASK_ID)
  base = _per_joint(base_model, base_scale)
  tennis = _per_joint(tennis_model, tennis_scale)
  assert base["right_shoulder_pitch_joint"]["kp"] == 30.0
  assert tennis["right_shoulder_pitch_joint"]["kp"] == 50.0
  assert base["waist_pitch_joint"]["kp"] != tennis["waist_pitch_joint"]["kp"]


def test_wrist_damping_is_at_least_critical_with_the_new_gains() -> None:
  """The ball-balancing reason for adopting the table, stated as a test.

  Wrist pitch/roll keep kp = 20 but kd doubles to 2.0, so the servo is no longer
  underdamped against the bat's reflected inertia. An underdamped wrist
  overshoots every ball impact, and the bed's tilt is what rolls the ball away.
  """
  model, _ = _compiled_robot(TASK_ID)
  data = mujoco.MjData(model)
  data.qpos[:] = model.qpos0
  free = int(np.ravel(model.joint("floating_base_joint").qposadr)[0])
  data.qpos[free : free + 3] = [0.0, 0.0, BAT_CARRY_BASE_HEIGHT]
  data.qpos[free + 3 : free + 7] = [1.0, 0.0, 0.0, 0.0]
  for name, value in BAT_CARRY_POSE.items():
    data.qpos[int(np.ravel(model.joint(name).qposadr)[0])] = value
  data.qvel[:] = 0.0
  mujoco.mj_forward(model, data)
  mass = np.zeros((model.nv, model.nv))
  mujoco.mj_fullM(model, data, mass)

  for joint in ("right_wrist_pitch_joint", "right_wrist_roll_joint"):
    joint_id = int(np.ravel(model.joint(joint).id)[0])
    dof = int(np.ravel(model.jnt_dofadr[joint_id])[0])
    kp = kd = None
    for actuator_id in range(model.nu):
      if int(model.actuator_trnid[actuator_id][0]) == joint_id:
        kp = float(model.actuator_gainprm[actuator_id][0])
        kd = -float(model.actuator_biasprm[actuator_id][2])
    assert kp is not None and kd is not None
    zeta = kd / (2.0 * np.sqrt(kp * float(mass[dof, dof])))
    assert zeta >= 1.0, f"{joint}: zeta={zeta:.2f} is still underdamped"


def _normalise_params(value):
  """Comparable form of an event's params, including nested SceneEntityCfg."""
  if isinstance(value, dict):
    return {k: _normalise_params(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return tuple(_normalise_params(v) for v in value)
  if value.__class__.__name__ == "SceneEntityCfg":
    return (
      value.name,
      tuple(value.joint_names or ()),
      tuple(value.body_names or ()),
      tuple(value.geom_names or ()),
      tuple(value.actuator_names or ()),
      tuple(value.actuator_ids) if not isinstance(value.actuator_ids, slice) else "all",
    )
  return value


def test_sim_to_real_randomization_is_the_multi_motion_tracking_tasks() -> None:
  """The physics DR axes must carry the tracking task's exact definitions."""
  tennis = load_env_cfg(TASK_ID).events
  tracking = load_env_cfg(TRACKING_TASK_ID).events
  names = (
    *SIM_TO_REAL_EVENT_NAMES,
    "randomize_effort_limits_wrist",
    "randomize_pd_gains",
  )
  for name in names:
    assert name in tennis, f"{name} missing"
    assert name in tracking, f"{name} missing from the tracking task"
    a, b = tennis[name], tracking[name]
    assert a.mode == b.mode, name
    assert a.func is b.func, name
    assert _normalise_params(a.params) == _normalise_params(b.params), name


def test_wrist_effort_derating_runs_after_the_whole_robot_term() -> None:
  """Both terms write the same actuator fields; only the last one takes effect."""
  events = list(load_env_cfg(TASK_ID).events)
  assert events.index("randomize_effort_limits_wrist") > events.index(
    "randomize_effort_limits"
  )
  wrist = load_env_cfg(TASK_ID).events["randomize_effort_limits_wrist"]
  assert wrist.params["asset_cfg"].actuator_ids == [
    actuator_group_index("wrist_pitch_roll")
  ]


def test_pd_gain_randomization_shares_one_scale_per_environment() -> None:
  """Correlated noise, as the tracking task runs it.

  A per-joint scale lets every joint drift independently, which contradicts the
  measured kp/kd pairing; one scale per environment keeps an episode's servo
  calibration internally consistent.
  """
  term = load_env_cfg(TASK_ID).events["randomize_pd_gains"]
  assert term.params["shared_gain_scale"] is True
  assert term.params["kp_range"] == term.params["kd_range"]


@pytest.mark.parametrize("name", SIM_TO_REAL_EVENT_NAMES)
def test_sim_to_real_terms_act_on_the_x2_geometry(name: str) -> None:
  """Foot terms must hit the 12 sole spheres, not the ankle-roll hull."""
  term = load_env_cfg(TASK_ID).events[name]
  asset = term.params["asset_cfg"]
  if name == "randomize_foot_size":
    assert asset.geom_names == r"^(left|right)_foot[0-9]+_collision$", asset.geom_names
  if name == "randomize_joint_damping":
    # Per-group ranges, not the generic absolute (0.0, 0.03).
    assert isinstance(term.params["ranges"], dict)
    assert term.params["ranges"][".*_wrist_pitch_joint"] == (0.0, 0.0012)
    assert term.params["ranges"][".*_hip_pitch_joint"] == (0.0, 0.03)


def test_task_adds_exactly_the_ball_terms_to_the_base_task() -> None:
  """The base terms must be untouched; only the ball may be added."""
  base = agibot_x2_flat_velocity_env_cfg()
  tennis = load_env_cfg(TASK_ID)
  assert tennis.actions.keys() == base.actions.keys()
  for group in ("actor", "critic"):
    added = set(tennis.observations[group].terms) - set(base.observations[group].terms)
    assert added == {"ball_pos", "ball_vel"}, group
    assert set(base.observations[group].terms) <= set(tennis.observations[group].terms)
  # Rewards and terminations keep every base term and add the ball's.
  assert set(base.rewards.keys()) <= set(tennis.rewards.keys())
  assert set(tennis.rewards.keys()) - set(base.rewards.keys()) == {
    "ball_centred",
    "ball_height",
    "ball_in_fov",
    "head_joint_motion",
  }
  assert set(tennis.terminations.keys()) - set(base.terminations.keys()) == {
    "ball_dropped"
  }
  assert set(base.terminations.keys()) <= set(tennis.terminations.keys())
  assert tennis.commands.keys() == base.commands.keys()


def test_ball_observation_widens_the_actor_by_six() -> None:
  """The ball adds a 3-vector position and a 3-vector velocity to both groups.

  Widths are read off a built environment, because the term shape depends on the
  asset, not on the config text. A checkpoint of the base task cannot be resumed
  here: the observation normalizer has the wrong width. That is why the ball task
  has its own experiment name.
  """
  env = _build_env(num_envs=1)
  try:
    om = env.observation_manager
    for group in ("actor", "critic"):
      names = om._group_obs_term_names[group]
      dims = om._group_obs_term_dim[group]
      widths = dict(zip(names, (int(np.prod(d)) for d in dims), strict=True))
      assert widths["ball_pos"] == 3
      assert widths["ball_vel"] == 3
      assert sum(widths.values()) == int(np.prod(om._group_obs_dim[group]))
  finally:
    env.close()


def test_resets_are_pinned_so_the_ball_lands_on_the_bed() -> None:
  """A sampled yaw or height would move the bat and leave the ball behind."""
  cfg = load_env_cfg(TASK_ID)
  base = agibot_x2_flat_velocity_env_cfg()
  pose_range = cfg.events["reset_base"].params["pose_range"]
  for key in ("x", "y", "z", "yaw"):
    assert pose_range[key] == (0.0, 0.0), key
  # The base task did jitter these; this task must not.
  assert base.events["reset_base"].params["pose_range"] != pose_range

  ball_reset = cfg.events["reset_ball"]
  assert ball_reset.mode == "reset"
  assert ball_reset.params["asset_cfg"].name == BALL_ENTITY
  assert ball_reset.params["pose_range"] == pose_range
  assert ball_reset.params["velocity_range"] == {}


def test_ball_entity_is_floating_and_starts_on_the_bat() -> None:
  cfg = get_tennis_ball_cfg()
  assert cfg.init_state.pos == pytest.approx(BALL_SPAWN_LOCAL_POS)
  model = cfg.spec_fn().compile()
  assert model.nq == 7
  assert model.nv == 6
  assert model.njnt == 1
  assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
  assert model.nu == 0


def test_play_config_also_loads() -> None:
  play_cfg = load_env_cfg(TASK_ID, play=True)
  assert play_cfg.events["reset_base"].params["pose_range"]["yaw"] == (0.0, 0.0)
  assert set(play_cfg.scene.entities) == {"robot", BALL_ENTITY}


def _build_env(num_envs: int = 1, steps: int = 1):
  """A live environment with the ball settled on the strings.

  ``steps`` control steps are run so the contact and the entity states are the
  stepped ones, which is what the reward and termination terms read at runtime.
  ``steps=0`` reads the pure reset state: the ball is exactly where the resetter
  put it. That matters because the reset also draws a random PD-gain scale, so
  after a step the bat has already begun to sag by a seed-dependent amount.
  The ball-drop termination ignores the first few steps after a reset, so tests
  that expect it to fire need ``steps >= 4``.
  """
  import torch

  cfg = load_env_cfg(TASK_ID)
  env = ManagerBasedRlEnv(cfg, device="cpu", num_envs=num_envs)
  env.reset()
  action = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
  for _ in range(steps):
    env.step(action)
  return env


def _place_ball(env, offset_plate_frame) -> None:
  """Put the ball at ``offset`` from the plate's top-surface centre.

  ``offset_plate_frame`` is ``(x along shaft, y across it, z along the normal)``,
  in the plate frame. A resting ball is ``(0, 0, BALL_RADIUS)``.
  """
  import torch

  from mjlab.utils.lab_api.math import quat_apply

  racket = env.scene["robot"]
  body_id = _racket_body_id(env)
  plate_pos = racket.data.body_link_pos_w[:, body_id]
  plate_quat = racket.data.body_link_quat_w[:, body_id]
  local = torch.tensor(
    [offset_plate_frame], dtype=plate_pos.dtype, device=plate_pos.device
  )
  local = local + torch.tensor(FACE_SITE_POS, dtype=local.dtype, device=local.device)
  world = plate_pos + quat_apply(plate_quat, local)
  ball = env.scene[BALL_ENTITY]
  ball.write_root_link_pose_to_sim(
    torch.cat([world, ball.data.root_link_quat_w], dim=-1)
  )
  # Kill any residual velocity so the measurement is of the placed state only.
  ball.write_root_link_velocity_to_sim(torch.zeros_like(ball.data.root_link_vel_w))
  env.sim.forward()


def _racket_body_id(env) -> int:
  for name, cfglist in zip(
    env.reward_manager._term_names, env.reward_manager._term_cfgs, strict=True
  ):
    if name == "ball_centred":
      cfg = cfglist if not isinstance(cfglist, list) else cfglist[0]
      ids = cfg.params["plate_cfg"].body_ids
      assert len(ids) == 1
      return ids[0]
  raise AssertionError("ball_centred reward is not configured")


def _ball_term(env, kind: str, name: str):
  """The resolved reward, termination or curriculum config for a ball term.

  For class-based terms ``func`` has been replaced by the instance, so the returned
  config's ``func`` is callable as ``func(env, env_ids, **params)`` for curricula and
  ``func(env, **params)`` for the others.
  """
  manager = {
    "reward": env.reward_manager,
    "termination": env.termination_manager,
    "curriculum": env.curriculum_manager,
  }[kind]
  for term_name, cfglist in zip(manager._term_names, manager._term_cfgs, strict=True):
    if term_name == name:
      return cfglist if not isinstance(cfglist, list) else cfglist[0]
  raise AssertionError(f"{name} is not configured")


@pytest.mark.slow
def test_environment_builds_and_steps() -> None:
  """One real build: the scene must compile and step without overflow."""
  import torch

  cfg = load_env_cfg(TASK_ID)
  env = ManagerBasedRlEnv(cfg, device="cpu", num_envs=1)
  env.reset()
  model = env.sim.mj_model
  assert model.opt.timestep == pytest.approx(TENNIS_PHYSICS_TIMESTEP)
  ball = env.scene[BALL_ENTITY]
  before = ball.data.root_link_pos_w[0].cpu().numpy().copy()
  for _ in range(3):
    env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim))
  after = ball.data.root_link_pos_w[0].cpu().numpy()
  assert np.all(np.isfinite(after))
  # The ball must move, but nothing like a launch: three 20 ms control steps of
  # an unsupported ball is at most ~18 mm of free fall.
  assert np.linalg.norm(after - before) < 0.05
  env.close()


##
# Ball-holding task: observation, reward and termination.
##


@pytest.mark.slow
def test_the_ball_observation_is_the_plate_relative_pose() -> None:
  """Position is the ball centre from the plate's top-surface centre, and a ball
  moving with the plate reads zero velocity.

  The velocity check uses MuJoCo's own site velocity as the oracle, so it does not
  re-use the formula under test. The wrist is given a real joint velocity first,
  because the face centre is 0.515 m from the racket body origin: a ball carried
  by a turning wrist must still read zero, which is only true if the plate-point
  velocity (``v + omega x r``) is subtracted, not just the body origin's velocity.
  """
  import torch

  # No steps: the pose assertions are about the reset state, and one step would
  # already include a seed-dependent PD sag.
  env = _build_env(steps=0)
  try:
    plate_cfg = _ball_term(env, "reward", "ball_centred").params["plate_cfg"]
    ball_cfg = _ball_term(env, "reward", "ball_centred").params["ball_cfg"]
    robot = env.scene["robot"]
    ball = env.scene[BALL_ENTITY]

    ## Position, at rest on the strings.
    position, _ = mdp.plate_relative_ball_state(env, plate_cfg, ball_cfg)
    pos = position[0].cpu().numpy()
    assert pos[0] == pytest.approx(0.0, abs=2e-3)
    assert pos[1] == pytest.approx(0.0, abs=2e-3)
    assert pos[2] == pytest.approx(BALL_RADIUS + BALL_SPAWN_CLEARANCE, abs=3e-3), (
      "a ball resting on the strings must read z = radius"
    )

    ## Velocity, with the arm turning.
    #
    # Step 1 validates the plate-point formula against MuJoCo itself: at the face
    # site, ``v + omega x r`` must reproduce mjlab's ``site_vel_w``, which comes
    # from MuJoCo's own subtrees. Step 2 then applies that validated formula at the
    # ball's position. Dropping the ``omega x r`` term changes the answer by
    # ``omega x r``, which is ~0.23 m/s at 3 rad/s, far above the tolerance.
    joint_id = robot.find_joints("right_elbow_joint")
    joints = joint_id[0] if isinstance(joint_id, tuple) else joint_id
    joint_vel = torch.zeros(env.num_envs, len(joints))
    joint_vel[:, 0] = 3.0  # rad/s
    robot.write_joint_velocity_to_sim(joint_vel, joint_ids=joints)
    env.sim.forward()

    body_id = _racket_body_id(env)
    body_pos = robot.data.body_link_pos_w[:, body_id]
    body_vel = robot.data.body_link_lin_vel_w[:, body_id]
    body_omega = robot.data.body_link_ang_vel_w[:, body_id]
    site_ids = robot.find_sites(FACE_SITE)[0]
    site_pos = robot.data.site_pos_w[:, site_ids].squeeze(1)
    site_vel = robot.data.site_vel_w[:, site_ids, :3].squeeze(1)
    formula_at_site = body_vel + torch.cross(body_omega, site_pos - body_pos, dim=1)
    assert (formula_at_site - site_vel).abs().max() < 1e-5, "formula vs MuJoCo"

    # Ball at rest in the world, plate turning.
    ball.write_root_link_velocity_to_sim(torch.zeros(env.num_envs, 6))
    env.sim.forward()
    ball_pos = ball.data.root_link_pos_w
    point_vel = body_vel + torch.cross(body_omega, ball_pos - body_pos, dim=1)
    assert (point_vel - body_vel).norm() > 0.2, "the rotation must matter"

    plate_quat = robot.data.body_link_quat_w[:, body_id]
    from mjlab.utils.lab_api.math import quat_apply_inverse

    expected = quat_apply_inverse(plate_quat, -point_vel)
    _, relative = mdp.plate_relative_ball_state(env, plate_cfg, ball_cfg)
    assert (relative - expected).abs().max() < 5e-3, (relative[0], expected[0])

    # And a ball moving exactly with the plate reads zero: "held" means zero here.
    ball.write_root_link_velocity_to_sim(
      torch.cat([point_vel, torch.zeros_like(point_vel)], dim=-1)
    )
    env.sim.forward()
    _, relative = mdp.plate_relative_ball_state(env, plate_cfg, ball_cfg)
    assert relative[0].abs().max() < 2e-3, relative[0].tolist()
  finally:
    env.close()


@pytest.mark.slow
def test_the_centring_reward_follows_the_plateau_then_decays() -> None:
  env = _build_env()
  try:
    term = _ball_term(env, "reward", "ball_centred")
    for dy, expected in (
      (0.000, 1.0),  # on the centre
      (0.010, 1.0),  # inside the plateau: no gradient, so no reward noise
      (0.020, 1.0),  # the plateau edge is still full credit
      (0.050, np.exp(-(((0.050 - 0.020) / 0.040) ** 2))),
      (0.084, np.exp(-(((0.084 - 0.020) / 0.040) ** 2))),
    ):
      _place_ball(env, (0.0, dy, BALL_RADIUS))
      got = float(term.func(env, **term.params)[0])
      assert got == pytest.approx(expected, abs=3e-3), (dy, got, expected)
    # The long axis reaches further, so the same offset costs less there.
    _place_ball(env, (0.084, 0.0, BALL_RADIUS))
    along = float(term.func(env, **term.params)[0])
    _place_ball(env, (0.0, 0.084, BALL_RADIUS))
    across = float(term.func(env, **term.params)[0])
    assert along == pytest.approx(across, abs=1e-6), "in-plane distance only"
  finally:
    env.close()


@pytest.mark.slow
def test_the_height_reward_is_full_on_the_strings_and_decays_above() -> None:
  env = _build_env()
  try:
    term = _ball_term(env, "reward", "ball_height")
    for dz, expected in (
      (BALL_RADIUS, 1.0),  # resting on the strings
      (BALL_RADIUS - 0.005, 1.0),  # sunk in: compliant contact
      (BALL_RADIUS + 0.05, 1.0),  # inside the 100 mm band
      (BALL_RADIUS + 0.10, 1.0),  # the band's edge
      (BALL_RADIUS + 0.20, np.exp(-((0.10 / 0.040) ** 2))),
    ):
      _place_ball(env, (0.0, 0.0, dz))
      got = float(term.func(env, **term.params)[0])
      assert got == pytest.approx(expected, abs=3e-3), (dz, got, expected)
  finally:
    env.close()


@pytest.mark.slow
def test_losing_the_ball_ends_the_episode_and_a_bounce_does_not() -> None:
  # Four steps: the drop term deliberately ignores the first three after a reset,
  # so that a stale ball state cannot end a fresh episode.
  env = _build_env(steps=4)
  try:
    term = _ball_term(env, "termination", "ball_dropped")
    # A ball on the strings, and one thrown a hand's width up, are both fine.
    for dz in (BALL_RADIUS, BALL_RADIUS + 0.15):
      _place_ball(env, (0.0, 0.0, dz))
      assert not bool(term.func(env, **term.params)[0]), dz
    # A ball below the plate is fatal once the grace has elapsed.
    term.func.reset(env_ids=slice(None))
    _place_ball(env, (0.0, 0.0, BALL_RADIUS - 0.05))
    grace = int(term.params["grace_steps"])
    for _ in range(grace):
      fired = bool(term.func(env, **term.params)[0])
    assert fired
    # Just above the threshold it must not fire at all: a sinking contact is not
    # a drop, however long it lasts.
    term.func.reset(env_ids=slice(None))
    _place_ball(env, (0.0, 0.0, BALL_RADIUS - 0.010))
    for _ in range(grace + 5):
      assert not bool(term.func(env, **term.params)[0])
  finally:
    env.close()


@pytest.mark.slow
def test_each_ball_observation_dimension_is_randomized_independently() -> None:
  """Six independent magnitudes, not one shared scale.

  The actor must see a randomized ball with a **different** spread on each of the
  three axes of each term; the critic must see the true ball. A single shared
  magnitude would fail the "different on each axis" assertions.
  """
  import numpy as np

  env = _build_env()
  try:
    om = env.observation_manager
    wanted = {"ball_pos": BALL_POSITION_NOISE_M, "ball_vel": BALL_VELOCITY_NOISE_M}
    spread = {}
    for group in ("actor", "critic"):
      samples = {name: [] for name in wanted}
      for _ in range(400):
        obs = om.compute_group(group)
        names = om._group_obs_term_names[group]
        dims = om._group_obs_term_dim[group]
        idx = 0
        for name, dim in zip(names, dims, strict=True):
          n = int(np.prod(dim))
          if name in samples:
            samples[name].append(obs[0, idx : idx + n].cpu().numpy().copy())
          idx += n
      spread[group] = {k: np.asarray(v) for k, v in samples.items()}

    for name, magnitudes in wanted.items():
      actor = spread["actor"][name]
      std = actor.std(axis=0)
      # Uniform half-width m has standard deviation m / sqrt(3).
      expected = np.asarray(magnitudes) / np.sqrt(3.0)
      assert std == pytest.approx(expected, rel=0.12), (name, std, expected)
      # The axes must not share one magnitude: the constants differ, so the
      # measured spread must too.
      assert not np.allclose(std, std[0], rtol=0.05), (name, std)
      # And the critic must be clean.
      assert np.abs(spread["critic"][name].std(axis=0)).max() < 1e-6, name
  finally:
    env.close()


##
# Training-procedure curriculum: patience on a lost ball, and a command ramp.
##


def test_the_task_starts_at_the_easiest_real_difficulty() -> None:
  """Stage 0 is slow walking, the base task's step-based ramp is gone, and there is
  no stand-still stage.

  A stand-still stage was tried and removed: with the robot commanded to stand the
  ball simply sits on the plate, nothing terminates for the full 20 s episode, and
  the performance gate read that as mastery and promoted to the hardest stage at
  once. Episode length only says something about ball handling when the robot is
  asked to move.
  """
  cfg = load_env_cfg(TASK_ID)
  ranges = cfg.commands["twist"].ranges
  assert ranges.lin_vel_x == BALL_COMMAND_STAGES[0]["lin_vel_x"]
  assert ranges.lin_vel_x == (-0.3, 0.3)
  assert ranges.lin_vel_x != (0.0, 0.0), "no stand-still stage"
  assert "command_vel" not in cfg.curriculum
  assert list(cfg.curriculum) == ["ball_command_ramp"]


def test_play_mode_uses_the_final_stage_and_does_not_ramp() -> None:
  """An evaluation must be commanded to walk, and must not start at a standstill."""
  cfg = load_env_cfg(TASK_ID, play=True)
  ranges = cfg.commands["twist"].ranges
  assert ranges.lin_vel_x == BALL_COMMAND_STAGES[-1]["lin_vel_x"]
  assert ranges.ang_vel_z == BALL_COMMAND_STAGES[-1]["ang_vel_z"]
  assert list(cfg.curriculum) == []


def test_the_drop_termination_is_a_grace_not_a_weaker_threshold() -> None:
  """The lost height is unchanged; only the time before the episode ends is new."""
  cfg = load_env_cfg(TASK_ID)
  term = cfg.terminations["ball_dropped"]
  assert term.func is mdp.ball_lost
  assert term.params["threshold"] == BALL_FALL_THRESHOLD
  assert term.params["grace_steps"] == round(
    BALL_DROP_GRACE_S / (cfg.sim.mujoco.timestep * cfg.decimation)
  )
  assert term.params["grace_steps"] > 1


@pytest.mark.slow
def test_a_lost_ball_ends_the_episode_only_after_the_grace() -> None:
  """Count consecutive lost steps; a recovering ball restarts the count."""
  env = _build_env(steps=0)
  try:
    term = _ball_term(env, "termination", "ball_dropped")
    grace = int(term.params["grace_steps"])
    term.func.reset(env_ids=slice(None))

    # Ball well below the plate: the first grace-1 steps must not fire.
    _place_ball(env, (0.0, 0.0, BALL_RADIUS - 0.05))
    for step in range(grace - 1):
      assert not bool(term.func(env, **term.params)[0]), step
    assert bool(term.func(env, **term.params)[0]), "must fire when the grace runs out"

    # Back above the plate: the count restarts, so it does not fire.
    _place_ball(env, (0.0, 0.0, BALL_RADIUS))
    assert not bool(term.func(env, **term.params)[0])
    _place_ball(env, (0.0, 0.0, BALL_RADIUS - 0.05))
    assert not bool(term.func(env, **term.params)[0]), "count must have restarted"

    # A reset clears it too.
    for _ in range(grace):
      term.func(env, **term.params)
    assert bool(term.func(env, **term.params)[0])
    term.func.reset(env_ids=slice(None))
    assert not bool(term.func(env, **term.params)[0])
  finally:
    env.close()


@pytest.mark.slow
def test_the_command_ramp_needs_both_the_schedule_and_the_floor() -> None:
  """The clock carries the difficulty; performance can only withhold an advance.

  Three failure modes this pins, all observed on real runs before the redesign:
  promoting without any skill because the easy stage saturates, promoting several
  stages on one batch, and demoting after a bad batch.

  The gate reads a window of finished episodes, so each phase feeds enough
  episodes to flush the previous ones out before it means anything.
  """
  env = _build_env(steps=0)
  try:
    term = _ball_term(env, "curriculum", "ball_command_ramp")
    ranges = env.command_manager.get_term("twist").cfg.ranges
    step_dt = env.step_dt
    floor = float(term.params["mastery_floor_s"])
    window = BALL_COMMAND_WINDOW
    last_step = BALL_COMMAND_STAGES[-1]["min_step"]

    def feed(episode_seconds: float, times: int) -> dict:
      state = None
      for _ in range(times):
        env.episode_length_buf[:] = round(episode_seconds / step_dt)
        state = term.func(env, slice(None), **term.params)
      return state

    # Phase A: skill, but the clock has not come round. Nothing advances.
    state = feed(floor + 5.0, window + 5)
    assert int(state["stage"]) == 0, "a good episode alone must not promote"
    assert ranges.lin_vel_x == BALL_COMMAND_STAGES[0]["lin_vel_x"]

    # Phase B: open the schedule only *after* flushing phase A's long episodes
    # out of the window, so the floor is judged on short ones alone.
    env.common_step_counter = 0
    state = feed(1.0, window + 5)
    assert int(state["stage"]) == 0
    env.common_step_counter = last_step + 1
    state = feed(1.0, 1)
    assert int(state["stage"]) == 0, "the floor must withhold the advance"
    assert float(state["typical_episode_s"]) < floor

    # Phase C: schedule open and skill present. The median flips after half the
    # window, and that single call may advance exactly one stage.
    flip = window // 2 + 1
    state = feed(floor + 5.0, flip)
    assert int(state["stage"]) == 1, "exactly one stage per call"
    assert ranges.lin_vel_x == BALL_COMMAND_STAGES[1]["lin_vel_x"]

    # Phase D: a collapse must not demote. It may not advance either once the
    # window reflects the collapse, but during the flush the stale long episodes
    # can still justify a promotion, which is fine: the property under test is that
    # the stage never goes down.
    stage_before = int(state["stage"])
    state = feed(0.5, window + 5)
    assert int(state["stage"]) >= stage_before, "promotion must be one-way"
    assert ranges.lin_vel_x == BALL_COMMAND_STAGES[int(state["stage"])]["lin_vel_x"]

    # Phase E: keep it up and the last stage is reached.
    state = feed(floor + 5.0, window + 5)
    assert int(state["stage"]) == len(BALL_COMMAND_STAGES) - 1
    assert ranges.lin_vel_x == (-1.0, 1.0)
    assert ranges.ang_vel_z == (-0.5, 0.5)
  finally:
    env.close()


@pytest.mark.slow
def test_a_few_lucky_episodes_cannot_promote_the_command_ramp() -> None:
  """The gate is a median, so outliers cannot outvote the typical episode.

  Regression: with a mean gate this task promoted a stage while the typical
  episode was still about 1 s long, because a handful of environments whose ball
  happened to settle survived for many seconds and dragged the mean up.
  """
  env = _build_env(steps=0)
  try:
    term = _ball_term(env, "curriculum", "ball_command_ramp")
    step_dt = env.step_dt

    def feed(episode_seconds: float) -> dict:
      env.episode_length_buf[:] = round(episode_seconds / step_dt)
      return term.func(env, slice(None), **term.params)

    # Open the schedule so that only the median protects the gate.
    env.common_step_counter = BALL_COMMAND_STAGES[-1]["min_step"] + 1
    # A typical 1 s episode for most environments.
    for _ in range(2):
      feed(1.0)
    # Now a minority of very long ones fill the rest of the window. With a mean
    # gate this alone would cross a 6 s threshold.
    state = feed(19.0)
    assert float(state["mean_episode_s"]) > float(state["typical_episode_s"])
    assert int(state["stage"]) == 0, "the median must protect the gate"
  finally:
    env.close()


##
# Head camera field of view, and head steadying.
##


def _place_ball_world(env, world_xyz) -> None:
  """Put the ball at an explicit world position, velocities zeroed."""
  import torch

  ball = env.scene[BALL_ENTITY]
  p = torch.tensor(
    [world_xyz],
    dtype=ball.data.root_link_pos_w.dtype,
    device=ball.data.root_link_pos_w.device,
  )
  ball.write_root_link_pose_to_sim(torch.cat([p, ball.data.root_link_quat_w], dim=-1))
  ball.write_root_link_velocity_to_sim(torch.zeros_like(ball.data.root_link_vel_w))
  env.sim.forward()


@pytest.mark.slow
def test_the_ball_is_in_view_where_it_is_carried() -> None:
  """At the carry pose the ball must score full credit on the camera term.

  This is the premise of the whole perception chain: if the ball is not in frame
  while it is being carried, no FOV reward can help.
  """
  env = _build_env(steps=0)
  try:
    term = _ball_term(env, "reward", "ball_in_fov")
    score = float(term.func(env, **term.params)[0])
    assert score == pytest.approx(1.0, abs=1e-3), score
  finally:
    env.close()


@pytest.mark.slow
def test_the_camera_reward_falls_with_height_and_vanishes_behind_the_head() -> None:
  """Monotone in the vertical angle, and zero for a ball behind the camera plane."""
  env = _build_env(steps=0)
  try:
    term = _ball_term(env, "reward", "ball_in_fov")
    ball = env.scene[BALL_ENTITY]
    base = ball.data.root_link_pos_w.clone()

    scores = []
    for dz in (0.0, 0.3, 0.6, 0.9):
      target = base.clone()
      target[0, 2] += dz
      _place_ball_world(env, target[0].tolist())
      scores.append(float(term.func(env, **term.params)[0]))
    assert scores[0] == pytest.approx(1.0, abs=1e-3)
    assert scores == sorted(scores, reverse=True), scores
    assert scores[-1] < 0.2, scores

    # A ball behind the head is behind the camera plane: no credit at all.
    robot = env.scene["robot"]
    head_id = robot.find_bodies("head_pitch_link")[0][0]
    head_pos = robot.data.body_link_pos_w[:, head_id]
    behind = head_pos.clone()
    behind[0, 0] -= 0.6
    _place_ball_world(env, behind[0].tolist())
    assert float(term.func(env, **term.params)[0]) == 0.0
  finally:
    env.close()


@pytest.mark.slow
def test_the_head_motion_penalty_is_zero_at_rest_and_square_in_velocity() -> None:
  """The penalty must be a velocity cost, not a position cost.

  A position cost would fight the field-of-view reward, which needs the head to be
  able to aim. Velocity is what "the head must not hunt" means.
  """
  import torch

  env = _build_env(steps=0)
  try:
    term = _ball_term(env, "reward", "head_joint_motion")
    robot = env.scene["robot"]
    joints = term.params["asset_cfg"].joint_ids

    assert float(term.func(env, **term.params)[0]) == pytest.approx(0.0, abs=1e-9)

    # Move only the head joints and check the quadratic form.
    vel = torch.zeros(env.num_envs, len(joints))
    vel[:, 0] = 0.5
    vel[:, 1] = -1.0
    robot.write_joint_velocity_to_sim(vel, joint_ids=joints)
    env.sim.forward()
    got = float(term.func(env, **term.params)[0])
    assert got == pytest.approx(0.5**2 + 1.0**2, rel=1e-3), got
  finally:
    env.close()


@pytest.mark.slow
def test_the_head_terms_are_wired_with_the_expected_signs() -> None:
  """FOV pays, head motion costs, and the FOV term is soft."""
  cfg = load_env_cfg(TASK_ID)
  fov = cfg.rewards["ball_in_fov"]
  head = cfg.rewards["head_joint_motion"]
  assert fov.func is mdp.ball_in_camera_fov
  assert head.func is mdp.head_joint_motion
  assert fov.weight > 0
  assert head.weight < 0
  # Soft: the FOV term must not outbid the centring term, which is the task.
  assert fov.weight < cfg.rewards["ball_centred"].weight
