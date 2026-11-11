"""Ball drop with the carry-velocity robot standing on the floor, base free.

No weld: the base is free and the robot stands on its own feet. After one second of settling
the ball is released.

**Status: not usable yet.** A constant-target PD hold is not a balancing controller, and this
fixture falls over during the first second: base height 0.6698 -> 0.0327 m. The real robot
stands because the carry-velocity *policy* closes the loop; a static gravity-compensated target
does not. To use this fixture, load that policy's checkpoint and drive the arm and legs from
it, rather than holding targets. The welded fixture in ``drop.py`` is the one the restitution
report uses, and its limitation is stated there.
"""
import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from drop import (  # noqa: E402
  MEASURED,
  first_bounce,
  first_contact_run,
  qadr,
  set_arm,
)
from model import build  # noqa: E402

from mjlab.tasks.registry import load_env_cfg  # noqa: E402

CARRY = "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-CarryVelocity"
JUGGLE = "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF-Ball3-Pelvis-Nominal"


def set_ball(model, data, pos):
  j = model.joint("drop_ball_free")
  q, v = int(j.qposadr[0]), int(j.dofadr[0])
  data.qpos[q : q + 3] = pos
  data.qpos[q + 3 : q + 7] = [1.0, 0.0, 0.0, 0.0]
  data.qvel[v : v + 6] = 0.0
  mujoco.mj_forward(model, data)


def park_ball(model, data):
  set_ball(model, data, [8.0, 8.0, 2.0])


def gravity_targets(model, data, pose):
  mujoco.mj_forward(model, data)
  rne = np.zeros(model.nv)
  mujoco.mj_rne(model, data, 0, rne)
  for i in range(model.nu):
    jid = int(model.actuator_trnid[i, 0])
    if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
      continue
    jname = model.joint(jid).name
    dof = int(model.jnt_dofadr[jid])
    q = float(data.qpos[int(model.jnt_qposadr[jid])])
    kp = float(model.actuator_gainprm[i, 0])
    base = pose.get(jname, q)
    data.ctrl[i] = base + (float(rne[dof]) / kp if kp > 1e-9 else 0.0)


def step(model, data, seconds):
  for _ in range(int(round(seconds / model.opt.timestep))):
    mujoco.mj_step(model, data)


def main():
  drops = [float(v) for v in (sys.argv[1:] or ["0.9"])]
  # The carry task integrates at 5 ms; the bed contact was fitted at 1 ms and injects energy
  # at 5 ms, so the ball drop runs at 1 ms.
  model, data = build(task=CARRY, ball_task=JUGGLE, timestep=0.001)
  print(f"task step {load_env_cfg(CARRY, play=True).sim.mujoco.timestep * 1000:.1f} ms -> "
        f"fixture step {model.opt.timestep * 1000:.1f} ms")

  cfg = load_env_cfg(CARRY, play=True)
  robot = cfg.scene.entities["robot"]
  base0 = tuple(float(v) for v in robot.init_state.pos)

  # free base at the task's standing height, joints from the task, arm from the robot
  data.qpos[:] = 0.0
  free_id = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
  assert len(free_id) == 1, free_id
  adr = int(model.jnt_qposadr[free_id[0]])
  base_adr = adr
  data.qpos[adr : adr + 3] = base0
  data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
  for name, val in robot.init_state.joint_pos.items():
    try:
      data.qpos[qadr(model, name)] = float(np.asarray(val).ravel()[0])
    except KeyError:
      continue
  set_arm(model, data, MEASURED)
  mujoco.mj_forward(model, data)

  # stand: compensate gravity, settle, repeat. One second of settling before the release.
  z_start = float(data.qpos[base_adr + 2])
  for _ in range(3):
    gravity_targets(model, data, MEASURED)
    park_ball(model, data)
    step(model, data, 0.35)
  park_ball(model, data)
  gravity_targets(model, data, MEASURED)
  step(model, data, 1.0)
  z_stand = float(data.qpos[base_adr + 2])
  mujoco.mj_forward(model, data)
  print(
    f"stand: base z {z_start:.4f} -> {z_stand:.4f} m ({z_stand - z_start:+.4f}), "
    f"contacts {data.ncon}"
  )

  fc = model.site("face_center").id
  radius = float(model.geom("drop_ball_geom").size[0])
  face = data.site_xpos[fc].copy()
  print(f"face_center {face[0]:.3f} {face[1]:.3f} {face[2]:.4f}")

  # bed surface from a slow drop, then the requested releases
  set_ball(model, data, [face[0], face[1], face[2] + radius + 0.02])
  zs_trace = []
  for _ in range(400):
    mujoco.mj_step(model, data)
    zs_trace.append(float(data.qpos[int(model.joint("drop_ball_free").qposadr[0]) + 2]))
  z_surface = float(min(zs_trace)) - radius

  for d in drops:
    set_ball(model, data, [face[0], face[1], z_surface + radius + d])
    q = int(model.joint("drop_ball_free").qposadr[0])
    bed = [
      gi for gi in range(model.ngeom) if "string_bed" in model.geom(gi).name
    ][0]
    ball_body = model.body("drop_ball").id
    z, contact, racket_q, base = [], [], [], []
    for _ in range(1600):
      mujoco.mj_step(model, data)
      z.append(float(data.qpos[q + 2]))
      hit = 0
      for k in range(data.ncon):
        g1, g2 = int(data.contact[k].geom1), int(data.contact[k].geom2)
        if bed in (g1, g2):
          other = g2 if g1 == bed else g1
          if model.geom_bodyid[other] == ball_body:
            hit = 1
      contact.append(hit)
      racket_q.append(data.xquat[model.body("tennis_racket").id].copy())
      base.append(float(data.qpos[base_adr + 2]))
    z = np.array(z)
    contact = np.array(contact)
    fb = first_bounce(z, contact, z_surface + radius + d, z_surface)
    run = first_contact_run(contact)
    if fb is None or run is None:
      print(f"release {d:.2f}: no first bounce")
      continue
    i0, i1 = run
    q0 = racket_q[max(i0 - 1, 0)]
    q1 = racket_q[i1]
    dot = min(1.0, abs(float(np.dot(q0, q1))))
    rot = float(2.0 * np.degrees(np.arccos(dot)))
    base_drift = max(base[i0 : i1 + 3]) - min(base[i0 : i1 + 3])
    print(
      f"release {d:.2f} | contact {i1 - i0 + 1} ms, racket {rot:.2f} deg, "
      f"base {base_drift * 1000:.3f} mm | e_h raw {fb['raw']['e']:.4f} "
      f"pipeline {fb['pipeline']['e']:.4f} | h_drop {fb['raw']['h_drop']:.3f} "
      f"apex above proxy {fb['raw']['apex_above_surface']:.3f}"
    )


if __name__ == "__main__":
  main()
