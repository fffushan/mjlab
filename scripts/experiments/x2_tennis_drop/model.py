"""Tennis robot (base fixed, task actuators) + the shipped ball as a free body."""
import mujoco

from mjlab.entity import Entity
from mjlab.tasks.registry import load_env_cfg

TASK = "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF-Ball3-Pelvis-Nominal"


def ship_ball(ball_task=TASK):
  """Read the shipped ball's compiled inertial and contact parameters."""
  cfg = load_env_cfg(ball_task, play=True)
  ball = Entity(cfg.scene.entities["ball"])
  m = ball.spec.compile()
  bid = m.body("ball").id
  gid = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid][0]
  return dict(
    mass=float(m.body_mass[bid]),
    inertia=tuple(float(v) for v in m.body_inertia[bid]),
    ipos=tuple(float(v) for v in m.body_ipos[bid]),
    iquat=tuple(float(v) for v in m.body_iquat[bid]),
    radius=float(m.geom_size[gid][0]),
    priority=int(m.geom_priority[gid]),
    condim=int(m.geom_condim[gid]),
    solref=tuple(float(v) for v in m.geom_solref[gid]),
    solimp=tuple(float(v) for v in m.geom_solimp[gid]),
    friction=tuple(float(v) for v in m.geom_friction[gid]),
  )


def build(task=TASK, ball_task=TASK, timestep=0.001):
  cfg = load_env_cfg(task, play=True)
  rob = Entity(cfg.scene.entities["robot"])
  sim_cfg = cfg.sim.mujoco
  spec = rob.spec
  free = [j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_FREE]
  assert len(free) == 1, free
  spec.delete(free[0])

  floor = spec.worldbody.add_geom(
    name="drop_floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[0.0, 0.0, 0.05],
    condim=3,
  )
  floor.friction = [1.0, 0.005, 0.0001]
  floor.priority = 0

  b = ship_ball(ball_task)
  body = spec.worldbody.add_body(name="drop_ball", pos=[0.0, 0.0, 2.0])
  body.add_joint(name="drop_ball_free", type=mujoco.mjtJoint.mjJNT_FREE)
  body.explicitinertial = True
  body.mass = b["mass"]
  body.inertia = list(b["inertia"])
  body.ipos = list(b["ipos"])
  body.iquat = list(b["iquat"])
  geom = body.add_geom(
    name="drop_ball_geom",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=[b["radius"], 0.0, 0.0],
    priority=b["priority"],
    condim=b["condim"],
  )
  geom.solref = list(b["solref"])
  geom.solimp = list(b["solimp"])
  geom.friction = list(b["friction"])
  model = spec.compile()
  # Use the task's own solver settings. The compiled defaults (Euler, 100 solver
  # iterations) are not what a training run integrates.
  sim_cfg.apply(model)
  # The task's own step, unless the caller pins one. The bed contact was fitted at 1 ms;
  # the velocity tasks run at 5 ms, where the same parameters inject energy.
  model.opt.timestep = timestep
  data = mujoco.MjData(model)
  return model, data
