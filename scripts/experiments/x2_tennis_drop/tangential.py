"""Vertical-drop baseline plus hand-like releases with incoming tangential motion."""
import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from drop import MEASURED, gravity_targets, qadr, set_ball
from model import build

from mjlab.tasks.registry import load_env_cfg

model, data = build()
cfg = load_env_cfg("Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF-Ball3-Pelvis-Nominal", play=True)
jp = cfg.scene.entities["robot"].init_state.joint_pos
data.qpos[:] = 0.0
for name, val in jp.items():
    try:
        data.qpos[qadr(model, name)] = float(np.asarray(val).ravel()[0])
    except KeyError:
        continue
gravity_targets(model, data, MEASURED)
fc = model.site("face_center").id
radius = float(model.geom("drop_ball_geom").size[0])
face = data.site_xpos[fc].copy()
bed = [g for g in range(model.ngeom) if "string_bed" in model.geom(g).name][0]
rng = np.random.default_rng(7)


def run(vxy=None, drop=0.9):
    """Release so the ball lands on the plate centre, with a chosen incoming v_xy."""
    if vxy is None:
        vxy = np.zeros(2)
    # Re-establish the arm equilibrium for every trial: the previous impact leaves a
    # residual oscillation, and every trial must start from the same state.
    gravity_targets(model, data, MEASURED)
    z0 = face[2] + radius + drop
    t_fall = np.sqrt(2 * drop / 9.81)
    x0 = face[:2] - vxy * t_fall
    set_ball(model, data, [x0[0], x0[1], z0])
    _j = model.joint("drop_ball_free")
    q, v = int(_j.qposadr[0]), int(_j.dofadr[0])
    data.qvel[v] = vxy[0]
    data.qvel[v + 1] = vxy[1]
    mujoco.mj_forward(model, data)
    C, V, W, Z = [], [], [], []
    for _ in range(700):
        mujoco.mj_step(model, data)
        C.append(1 if any(bed in (data.contact[j].geom1, data.contact[j].geom2)
                          for j in range(data.ncon)) else 0)
        V.append(data.qvel[v:v + 3].copy())
        W.append(data.qvel[v + 3:v + 6].copy())
        Z.append(float(data.qpos[q + 2]))
    C = np.array(C)
    if not C.any():
        return None
    i = int(np.argmax(C > 0))
    j = i
    while j < len(C) - 1 and C[j] > 0:
        j += 1
    z = np.array(Z)
    z_min = z[i : j + 1].min()
    h_drop = Z[0] - z_min
    h_reb = z[j:].max() - z_min
    vin, vout = np.array(V[i - 1]), np.array(V[j])
    dv = vout - vin
    return dict(e=float(np.sqrt(max(h_reb, 0) / h_drop)),
                vin_xy=float(np.hypot(*vin[:2])), vout_xy=float(np.hypot(*vout[:2])),
                dv_xy=float(np.hypot(dv[0], dv[1])),
                dh=float((np.degrees(np.arctan2(vout[1], vout[0]) - np.arctan2(vin[1], vin[0])) + 180) % 360 - 180),
                spin=float(np.linalg.norm(W[j])))


base = [run() for _ in range(4)]
print("perfectly vertical release (sim):")
print("  e_h %.3f   out v_xy %.3f   |dv_xy| %.3f   |spin| %.2f rad/s" % (
  np.median([b["e"] for b in base]), np.median([b["vout_xy"] for b in base]),
  np.median([b["dv_xy"] for b in base]), np.median([b["spin"] for b in base])))

for lo, hi, label in ((0.02, 0.15, "incoming 0.02-0.15 m/s"), (0.02, 0.36, "incoming 0.02-0.36 m/s")):
    rows = []
    for _ in range(25):
        mag = rng.uniform(lo, hi)
        th = rng.uniform(0.0, 2.0 * np.pi)
        r = run(np.array([mag * np.cos(th), mag * np.sin(th)]))
        if r:
            rows.append(r)
    dv = np.array([r["dv_xy"] for r in rows])
    dh = np.array([r["dh"] for r in rows])
    spin = np.array([r["spin"] for r in rows])
    print(f"{label} (n={len(rows)}):")
    print("  |dv_xy| median %.3f  IQR %.3f-%.3f   |spin| median %.2f   heading %+d..%+d deg   e_h median %.3f"
          % (np.median(dv), np.percentile(dv, 25), np.percentile(dv, 75),
             np.median(spin), int(dh.min()), int(dh.max()), np.median([r["e"] for r in rows])))
