"""Static-racket ball drop on the tennis juggle model.

Protocol from the JugglingSim2Real session (2026-11-11, +08):
  * hold the pose the standing lane held on the robot, targets gravity-compensated,
  * drop the ball vertically from a height above the bed,
  * resample to 100 Hz, 3-sample moving average, exclude the first and last three
    samples, locate local extrema,
  * e_h = sqrt(h_reb / h_drop), first apex after the impact.

The model is built by ``model.py``, which applies the task's own MuJoCo options.
"""
import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from model import build  # noqa: E402

from mjlab.tasks.registry import load_env_cfg  # noqa: E402

TASK = (
  "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-"
  "Tennis-Ball-Juggle-29DOF-Ball3-Pelvis-Nominal"
)

# Median pose the standing lane held on the robot during the drop session.
MEASURED = {
  "waist_yaw_joint": -0.0041,
  "waist_pitch_joint": 0.0034,
  "waist_roll_joint": 0.0006,
  "right_shoulder_pitch_joint": -1.0108,
  "right_shoulder_roll_joint": -0.3753,
  "right_shoulder_yaw_joint": -0.4718,
  "right_elbow_joint": -1.0363,
  "right_wrist_yaw_joint": 1.5191,
  "right_wrist_pitch_joint": -0.1271,
  "right_wrist_roll_joint": 0.6441,
}


def qadr(model, name):
  return int(model.joint(name).qposadr[0])


def set_arm(model, data, pose):
  for name, angle in pose.items():
    data.qpos[qadr(model, name)] = angle
  mujoco.mj_forward(model, data)


def set_ball(model, data, pos):
  j = model.joint("drop_ball_free")
  q, v = int(j.qposadr[0]), int(j.dofadr[0])
  data.qpos[q : q + 3] = pos
  data.qpos[q + 3 : q + 7] = [1.0, 0.0, 0.0, 0.0]
  data.qvel[v : v + 6] = 0.0
  mujoco.mj_forward(model, data)


def settle(model, data, seconds):
  for _ in range(int(round(seconds / model.opt.timestep))):
    mujoco.mj_step(model, data)


def park_ball(model, data):
  """Put the ball far away so it cannot touch the racket while the arm settles."""
  set_ball(model, data, [8.0, 8.0, 2.0])


def gravity_targets(model, data, pose):
  """Set ctrl so the settled pose is ``pose``: target = pose + tau_gravity / kp.

  The real arm holds its pose with the same PD law, so its target also sits ahead of the
  achieved angle. Compensating by inverse dynamics gives the same equilibrium in one shot.
  """
  summon = dict(pose)
  set_arm(model, data, summon)
  data.qvel[:] = 0.0
  data.qacc[:] = 0.0
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
    base = summon.get(jname, q)
    data.ctrl[i] = base + (float(rne[dof]) / kp if kp > 1e-9 else 0.0)
  park_ball(model, data)
  settle(model, data, 1.0)


def set_wrist(model, gains):
  """Diagnostic gain intervention on both wrists; not a rigidity claim."""
  if gains is None:
    return
  kp, kd = gains
  for i in range(model.nu):
    if "wrist" in model.actuator(i).name:
      model.actuator_gainprm[i, 0] = kp
      model.actuator_biasprm[i, 1] = -kp
      model.actuator_biasprm[i, 2] = -kd


def first_contact_run(contact):
  """Index range of the first contiguous contact episode."""
  idx = np.flatnonzero(contact > 0)
  if not len(idx):
    return None
  i = int(idx[0])
  j = i
  while j + 1 < len(contact) and contact[j + 1] > 0:
    j += 1
  return i, j


def drop(model, data, xy, z0, seconds=1.6):
  set_ball(model, data, [xy[0], xy[1], z0])
  j = model.joint("drop_ball_free")
  q = int(j.qposadr[0])
  bed = [g for g in range(model.ngeom) if "string_bed" in model.geom(g).name][0]
  ball_body = model.body("drop_ball").id
  racket = model.body("tennis_racket").id
  z, contact, quat, centre = [], [], [], []
  for _ in range(int(round(seconds / model.opt.timestep))):
    mujoco.mj_step(model, data)
    z.append(float(data.qpos[q + 2]))
    centre.append(data.qpos[q : q + 3].copy())
    hit = 0
    for k in range(data.ncon):
      g1, g2 = int(data.contact[k].geom1), int(data.contact[k].geom2)
      if bed in (g1, g2):
        other = g2 if g1 == bed else g1
        if model.geom_bodyid[other] == ball_body:
          hit = 1
    contact.append(hit)
    quat.append(data.xquat[racket].copy())
  return (np.array(z), np.array(contact), np.array(quat), np.array(centre))


def first_bounce(z, contact, z0, z_surface, sample=10, smooth=3, band=0.06,
                 min_drop=0.08):
  """First bounce from the known release height, reported twice.

  The release is a held ball in both experiments, so its height is known: ``z0``. There is
  no pre-release arc, so the extremum rule in :func:`analyse` cannot find a maximum before
  the first impact.

  Two estimates are returned because the two sides hold different information:

  * ``raw``: minimum and apex from the 1 ms trace. This is simulation truth.
  * ``pipeline``: minimum and apex from the 100 Hz, 3-sample-smoothed trace. This is the
    only estimate comparable with a 100 Hz motion-capture analysis.
  """
  zz = z[::sample].copy()
  if smooth > 1:
    zz = np.convolve(zz, np.ones(smooth) / smooth, mode="same")
  run = first_contact_run(contact)
  if run is None:
    return None
  i0, i1 = run
  i0s, i1s = i0 // sample, i1 // sample
  out = {}
  for label, z_min, lo, span in (
    ("raw", float(z[max(i0 - 2, 0) : i1 + 3].min()), max(i0 - 2, 0), z),
    ("pipeline", float(zz[max(i0s - 1, 0) : i1s + 2].min()), max(i0s - 1, 0), zz),
  ):
    apex = None
    for i in range(lo + 1, len(span) - 1):
      if span[i] > span[i - 1] and span[i] >= span[i + 1] and i > (i1s if label == "pipeline" else i1):
        apex = float(span[i])
        break
    if apex is None:
      return None
    h_drop = z0 - z_min
    h_reb = apex - z_min
    if abs(z0 - z_surface) > 1e9:  # placeholder, gates below
      pass
    accepted = (
      abs(z_min - z_surface) <= band and h_drop >= min_drop and h_reb > 0
    )
    out[label] = dict(
      h_drop=h_drop,
      h_reb=h_reb,
      apex_above_surface=h_reb + (z_min - z_surface),
      e=float(np.sqrt(h_reb / h_drop)) if h_drop > 0 else float("nan"),
      accepted=accepted,
    )
  return out


def analyse(z, z_surface, sample=10, smooth=3, band=0.06, min_drop=0.08, pad=3):
  """The measurement owner's pipeline, with their first/last-three-sample exclusion."""
  zz = z[::sample].copy()
  if smooth > 1:
    zz = np.convolve(zz, np.ones(smooth) / smooth, mode="same")
  lo, hi = pad, len(zz) - pad
  events = []
  for i in range(lo, hi):
    if zz[i] > zz[i - 1] and zz[i] >= zz[i + 1]:
      events.append(("max", i, float(zz[i])))
    elif zz[i] < zz[i - 1] and zz[i] <= zz[i + 1]:
      events.append(("min", i, float(zz[i])))
  out = []
  for k, (kind, i, val) in enumerate(events):
    if kind != "min" or k == 0 or k == len(events) - 1:
      continue
    if events[k - 1][0] != "max" or events[k + 1][0] != "max":
      continue
    h_drop = events[k - 1][2] - val
    h_reb = events[k + 1][2] - val
    if abs(val - z_surface) > band or h_drop < min_drop or h_reb <= 0:
      continue
    out.append(
      dict(
        t=i * sample * 1e-3,
        h_drop=h_drop,
        h_reb=h_reb,
        apex_above_bed=h_reb + (val - z_surface),
        e=float(np.sqrt(h_reb / h_drop)),
      )
    )
  return out


def main():
  drops = [float(v) for v in (sys.argv[1:] or ["0.9"])]
  model, data = build()
  cfg = load_env_cfg(TASK, play=True)
  jp = cfg.scene.entities["robot"].init_state.joint_pos
  data.qpos[:] = 0.0
  for name, val in jp.items():
    try:
      data.qpos[qadr(model, name)] = float(np.asarray(val).ravel()[0])
    except KeyError:
      continue
  set_arm(model, data, MEASURED)

  fc = model.site("face_center").id
  radius = float(model.geom("drop_ball_geom").size[0])
  bed_geom = [g for g in range(model.ngeom) if "string_bed" in model.geom(g).name][0]
  m = np.asarray(data.geom_xmat[bed_geom]).reshape(3, 3)
  normal = m[:, 0] * np.sign(m[2, 0])
  tilt = float(np.degrees(np.arccos(min(1.0, normal[2]))))

  # Bed surface: the lowest ball centre of a slow drop, minus the radius.
  gravity_targets(model, data, MEASURED)
  face = data.site_xpos[fc].copy()
  z_slow, _, _, _ = drop(model, data, (face[0], face[1]), face[2] + radius + 0.02, 0.4)
  z_surface = float(z_slow.min()) - radius
  residual = max(abs(MEASURED[n] - data.qpos[qadr(model, n)]) for n in MEASURED)
  print(f"pose residual {residual:.4f} rad | plate tilt {tilt:.2f} deg | "
        f"bed surface z {z_surface:.4f} (site {face[2]:.4f})")

  arms = [
    ("shipped wrist 20/2", None),
    ("wrist gain intervention 5000/50", (5000.0, 50.0)),
  ]
  for label, gains in arms:
    set_wrist(model, gains)
    for d in drops:
      gravity_targets(model, data, MEASURED)
      face = data.site_xpos[fc].copy()
      z, contact, quat, centre = drop(
        model, data, (face[0], face[1]), z_surface + radius + d
      )
      bounces = analyse(z, z_surface)
      first = first_bounce(z, contact, z_surface + radius + d, z_surface)
      run = first_contact_run(contact)
      if run:
        i0, i1 = max(run[0] - 1, 0), run[1]
        q0, q1 = quat[i0], quat[i1]
        dot = abs(float(np.dot(q0, q1)))
        rot = float(2 * np.degrees(np.arccos(min(1.0, dot))))
        dur_ms = (run[1] - run[0] + 1)
      else:
        rot = 0.0
        dur_ms = 0
      line = f"[{label}] release {d:.2f} | first contact {dur_ms} ms, racket {rot:.2f} deg"
      if first:
        r, p = first["raw"], first["pipeline"]
        line += (
          f" | e_h raw {r['e']:.4f} / pipeline {p['e']:.4f}"
          f" | h_drop {r['h_drop']:.3f}"
          f" | apex above proxy {r['apex_above_surface']:.3f} m"
          f" | gates {'pass' if p['accepted'] else 'FAIL'}"
        )
      else:
        line += " | no first bounce found"
      print(line)
      for n, b in enumerate(bounces[:3], start=2):
        print(
          f"      release {d:.2f} bounce {n}: drop {b['h_drop']:.3f}"
          f"  apex {b['h_reb']:.3f}  e_h {b['e']:.3f}"
        )


if __name__ == "__main__":
  main()
