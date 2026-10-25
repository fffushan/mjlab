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
  BALL_RADIUS,
  BALL_SPAWN_CLEARANCE,
  BALL_SPAWN_LOCAL_POS,
  BAT_CARRY_BASE_HEIGHT,
  BAT_CARRY_POSE,
  BED_CENTER_X,
  BED_CONDIM,
  BED_FRICTION,
  BED_GEOM,
  BED_HALF_THICKNESS,
  BED_PRIORITY,
  BED_SOLIMP,
  BED_SOLREF,
  CLAMP_BODY,
  CLAMP_DIAGINERTIA,
  CLAMP_MASS,
  CONNECTOR_DIAGINERTIA,
  CONNECTOR_MASS,
  FACE_SITE,
  FACE_SITE_POS,
  RACKET_BODY,
  RACKET_DIAGINERTIA,
  RACKET_MASS,
  TENNIS_BED_COLLISION,
  TENNIS_MESH_DIR,
  TENNIS_MESH_SHA256,
  WRIST_BODY,
  get_tennis_ball_cfg,
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
# Isolated drop fixture: one stationary horizontal string bed plus one ball.
# Built from the shipped constants so a parameter change that is not re-fitted
# fails the test instead of silently changing contact behaviour.
##


def _bed_fixture_spec(timestep: float) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  spec.option.timestep = timestep
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
  spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
  spec.option.iterations = 100
  spec.option.tolerance = 1e-8
  spec.option.ls_iterations = 50
  spec.option.ls_tolerance = 0.01

  angle = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
  vertices = np.concatenate(
    [
      np.c_[
        0.157 * np.cos(angle),
        0.117 * np.sin(angle),
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
    condim=BED_CONDIM,
    priority=BED_PRIORITY,
    mass=0.0,
    friction=list(BED_FRICTION),
    solref=list(BED_SOLREF),
    solimp=list(BED_SOLIMP),
  )

  ball = spec.worldbody.add_body(name="ball")
  ball.add_freejoint(name="ball_free")
  ball.add_geom(
    name="ball_collision",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=[BALL_RADIUS, 0.0, 0.0],
    mass=0.057,
    contype=1,
    conaffinity=1,
    condim=3,
  )
  return spec


def _drop_and_measure(timestep: float, drop: float) -> tuple[float, float, float]:
  """Apex height, maximum lateral speed, and lowest point for a dropped ball."""
  model = _bed_fixture_spec(timestep).compile()
  data = mujoco.MjData(model)
  data.qpos[:] = 0.0
  # Bed top surface is at +BED_HALF_THICKNESS; the ball starts resting on it plus
  # ``drop`` metres.
  data.qpos[2] = BED_HALF_THICKNESS + BALL_RADIUS + drop
  mujoco.mj_forward(model, data)

  trace: list[float] = []
  min_bottom = np.inf
  max_lateral = 0.0
  first_contact = None
  for step in range(int(1.0 / timestep)):
    mujoco.mj_step(model, data)
    trace.append(float(data.qpos[2]))
    if data.ncon and first_contact is None:
      first_contact = step
    min_bottom = min(min_bottom, data.qpos[2] - BALL_RADIUS)
    max_lateral = max(max_lateral, float(abs(data.qvel[0])), float(abs(data.qvel[1])))
  assert first_contact is not None, "the ball never touched the bed"
  # The rebound apex is reached in free flight, after the contact has ended, so
  # it cannot be sampled only on contact steps.
  apex = max(trace[first_contact:])
  return apex - BED_HALF_THICKNESS - BALL_RADIUS, max_lateral, min_bottom


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
  assert model.geom_condim[geom_id] == BED_CONDIM
  assert model.geom_priority[geom_id] == BED_PRIORITY
  assert tuple(model.geom_friction[geom_id]) == pytest.approx(BED_FRICTION)
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


def test_task_keeps_the_base_action_and_observation_widths() -> None:
  base = agibot_x2_flat_velocity_env_cfg()
  tennis = load_env_cfg(TASK_ID)
  assert tennis.actions.keys() == base.actions.keys()
  for group in ("actor", "critic"):
    assert tennis.observations[group].terms.keys() == (
      base.observations[group].terms.keys()
    )
  assert tennis.rewards.keys() == base.rewards.keys()
  assert tennis.terminations.keys() == base.terminations.keys()
  assert tennis.commands.keys() == base.commands.keys()


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
