"""Distribution analysis of the prepared X2 joint-pose dataset.

Reports, for each motion mode, three distributions: the reference the tracker was
asked to follow, the achieved state that was recorded, and the failed subset from
the trackability ledger.  Everything is expressed in the robot's own root frame,
so the numbers do not depend on where the robot was facing.

Two kinds of mode are kept apart on purpose:

* behaviour modes (turning, arm elevation, crouch, speed, ...) answer "does the
  dataset contain this behaviour";
* artifact modes (segment length, start phase, lane, split) answer "is the
  dataset biased by how we collected it".

Nothing here writes to the dataset.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

DATA = Path("/home/fushan/data/mjlab-latent-free-144/data")
TENNIS = Path("/home/fushan/mjlab/data/tennis")
LEDGER = Path("/home/fushan/data/mjlab-latent-free-144/failure_ledger.json")
WORKTREE = Path(
    os.environ.get("MJLAB_WORKTREE", Path(__file__).resolve().parents[3])
)
XML = WORKTREE / "src/mjlab/asset_zoo/robots/agibot_x2/xmls/x2_ultra.xml"

# Store column order is the robot's joint order, which is the model's joint order.
JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
    "head_yaw_joint", "head_pitch_joint",
)
IDX = {name: i for i, name in enumerate(JOINTS)}
ARM_L = [IDX["left_shoulder_pitch_joint"], IDX["left_shoulder_roll_joint"],
         IDX["left_elbow_joint"], IDX["left_wrist_yaw_joint"]]
ARM_R = [IDX["right_shoulder_pitch_joint"], IDX["right_shoulder_roll_joint"],
         IDX["right_elbow_joint"], IDX["right_wrist_yaw_joint"]]
LEG_L = [IDX["left_hip_pitch_joint"], IDX["left_knee_joint"], IDX["left_ankle_pitch_joint"]]
LEG_R = [IDX["right_hip_pitch_joint"], IDX["right_knee_joint"], IDX["right_ankle_pitch_joint"]]
BODIES = ("pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link",
          "left_wrist_yaw_link", "right_wrist_yaw_link")


def build_fk() -> tuple[mujoco.MjModel, mujoco.MjData, np.ndarray, np.ndarray]:
  model = mujoco.MjModel.from_xml_path(str(XML))
  data = mujoco.MjData(model)
  addrs = np.array(
      [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
       for name in JOINTS],
      dtype=np.int64,
  )
  body_ids = np.array(
      [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in BODIES],
      dtype=np.int64,
  )
  if (body_ids < 0).any():
    missing = [n for n, i in zip(BODIES, body_ids, strict=True) if i < 0]
    raise RuntimeError(f"bodies missing from the model: {missing}")
  return model, data, addrs, body_ids


def fk_positions(
    model, data, addrs, body_ids, q: np.ndarray, torso_id: int
) -> tuple[np.ndarray, np.ndarray]:
  """Return body positions in the root frame, plus the torso tilt in degrees.

  The free joint is pinned at the origin with an identity orientation, so the
  result is independent of where the robot was facing or standing.

  Tilt comes from the torso body's own z axis, not from a position difference:
  the torso link origin sits on the waist axis, so positions cannot see rotation.
  """
  data.qpos[:] = 0.0
  data.qpos[3] = 1.0
  out = np.empty((q.shape[0], body_ids.shape[0], 3), dtype=np.float64)
  tilt = np.empty(q.shape[0], dtype=np.float64)
  for row in range(q.shape[0]):
    data.qpos[addrs] = q[row]
    mujoco.mj_forward(model, data)
    out[row] = data.xpos[body_ids]
    rot = data.xmat[torso_id].reshape(3, 3)
    tilt[row] = np.rad2deg(np.arccos(np.clip(rot[2, 2], -1.0, 1.0)))
  return out, tilt


def canonical_quat(quat_wxyz: np.ndarray) -> np.ndarray:
  """Remove the quaternion double-cover sign flip before differencing.

  ``q`` and ``-q`` are the same rotation, but a yaw series taken across a flip
  jumps by 2*pi and turns a smooth turn into a false 100 deg/s spike.
  """
  out = np.array(quat_wxyz, dtype=np.float64, copy=True)
  for row in range(1, out.shape[0]):
    if float(np.dot(out[row], out[row - 1])) < 0.0:
      out[row] = -out[row]
  if out.shape[0] and out[0, 0] < 0.0:
    out[:] = -out
  return out


def yaw_of(quat_wxyz: np.ndarray) -> np.ndarray:
  w, x, y, z = quat_wxyz[..., 0], quat_wxyz[..., 1], quat_wxyz[..., 2], quat_wxyz[..., 3]
  return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def dynamics(q: np.ndarray, dq: np.ndarray, dt: float) -> dict[str, float]:
  """Per-segment dynamics summaries from the sampled rows."""
  speed = float(np.abs(dq).mean())
  accel = float(np.abs(np.diff(dq, axis=0)).mean() / dt) if q.shape[0] > 1 else 0.0
  sign = np.sign(dq)
  flips = (np.diff(sign, axis=0) != 0) & (np.diff(sign, axis=0) != 0)
  reversals = float(flips.sum(axis=1).mean()) if q.shape[0] > 1 else 0.0
  return {"speed_rad_s": speed, "accel_rad_s2": accel, "reversals": reversals}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--stride-segments", type=int, default=20)
  ap.add_argument("--stride-rows", type=int, default=8)
  ap.add_argument("--out-json", type=Path, default=Path("/tmp/modes/report.json"))
  ap.add_argument("--out-md", type=Path, default=Path("/tmp/modes/report.md"))
  args = ap.parse_args()

  table = json.loads((DATA / "store" / "manifest.json").read_text())["table"]
  q_mm = np.load(DATA / "store" / "joint_pos.npy", mmap_mode="r")
  dq_mm = np.load(DATA / "store" / "joint_vel.npy", mmap_mode="r")
  quat_mm = np.load(DATA / "store" / "root_quat_wxyz.npy", mmap_mode="r")

  ledger = json.loads(LEDGER.read_text())["segments"] if LEDGER.is_file() else {}
  split_of: dict[str, str] = {}
  assign_path = DATA / "assignments.json"
  if assign_path.is_file():
    payload = json.loads(assign_path.read_text())
    for motion, value in payload.get("motions", {}).items():
      split_of[motion] = value if isinstance(value, str) else value.get("split", "?")

  model, data, addrs, body_ids = build_fk()
  torso_id = int(body_ids[1])
  dt = 0.02 * args.stride_rows
  clip_cache: dict[str, dict[str, np.ndarray]] = {}
  src_cache: dict[str, np.ndarray] = {}

  def clip(name: str) -> dict[str, np.ndarray]:
    if name not in clip_cache:
      with np.load(TENNIS / name, allow_pickle=False) as z:
        clip_cache.clear()
        clip_cache[name] = {
            k: np.asarray(z[k]) for k in
            ("joint_pos", "joint_vel", "body_quat_w", "body_pos_w")
        }
    return clip_cache[name]

  def inits(path: str) -> np.ndarray:
    if path not in src_cache:
      with np.load(path, allow_pickle=False) as z:
        src_cache.clear()
        src_cache[path] = np.asarray(z["trajectory_initial_frame"], dtype=np.int64)
    return src_cache[path]

  rows: list[dict] = []
  order_check: list[float] = []
  for position in range(0, len(table), args.stride_segments):
    entry = table[position]
    seg = int(entry["segment"])
    lo = int(entry["row_start"])
    hi = lo + int(entry["rows"])
    n = hi - lo
    if n < 3:
      continue
    ticks = np.arange(0, n, args.stride_rows)
    if ticks.size < 2:
      continue
    c = clip(entry["motion"])
    init = int(inits(entry["source"])[int(entry["trajectory_index"])])
    total = c["joint_pos"].shape[0]
    ref_rows = np.minimum(init + ticks, total - 1)

    q = np.asarray(q_mm[lo + ticks], dtype=np.float64)
    dq = np.asarray(dq_mm[lo + ticks], dtype=np.float64)
    quat = np.asarray(quat_mm[lo + ticks], dtype=np.float64)
    rq = np.asarray(c["joint_pos"][ref_rows], dtype=np.float64)
    rdq = np.asarray(c["joint_vel"][ref_rows], dtype=np.float64)

    if len(order_check) < 40:
      order_check.append(float(np.abs(q[0] - rq[0]).mean()))

    pos, tilt = fk_positions(model, data, addrs, body_ids, q, torso_id)
    rpos, rtilt = fk_positions(model, data, addrs, body_ids, rq, torso_id)
    pelvis = pos[:, 0]
    rpelvis = rpos[:, 0]

    dyaw_a = np.diff(yaw_of(canonical_quat(quat)))
    ref_root_quat = np.asarray(c["body_quat_w"][ref_rows])[:, 0, :]
    dyaw_r = np.diff(yaw_of(canonical_quat(ref_root_quat)))
    arm_act = float(np.abs(dq[:, ARM_L]).mean())
    arm_act_r = float(np.abs(dq[:, ARM_R]).mean())

    record = {
        "segment": seg,
        "group": entry["group"],
        "motion": entry["motion"],
        "split": split_of.get(entry["motion"], "?"),
        "rows": n,
        "windows": int(entry["windows"]),
        "phase": round(init / max(total - 1, 1), 3),
        "failures": int(ledger.get(str(seg), {}).get("failures", 0)),
        "runs": int(ledger.get(str(seg), {}).get("runs", 0)),
        # rotation
        "yaw_rate_deg_s": float(np.abs(dyaw_a).mean() / dt * 180 / np.pi) if dyaw_a.size else 0.0,
        "yaw_rate_ref_deg_s": float(np.abs(dyaw_r).mean() / dt * 180 / np.pi) if dyaw_r.size else 0.0,
        "yaw_signed_deg": float(dyaw_a.sum() * 180 / np.pi) if dyaw_a.size else 0.0,
        "yaw_path_deg": float(np.abs(dyaw_a).sum() * 180 / np.pi) if dyaw_a.size else 0.0,
        # arms
        "shoulder_pitch_L_deg": float(np.rad2deg(q[:, IDX["left_shoulder_pitch_joint"]]).mean()),
        "shoulder_pitch_R_deg": float(np.rad2deg(q[:, IDX["right_shoulder_pitch_joint"]]).mean()),
        "shoulder_pitch_L_ref_deg": float(np.rad2deg(rq[:, IDX["left_shoulder_pitch_joint"]]).mean()),
        "shoulder_pitch_R_ref_deg": float(np.rad2deg(rq[:, IDX["right_shoulder_pitch_joint"]]).mean()),
        "wrist_z_max": float((pos[:, 4, 2] - pelvis[:, 2]).max()),
        "wrist_z_max_ref": float((rpos[:, 4, 2] - rpelvis[:, 2]).max()),
        "arm_asym": float((arm_act - arm_act_r) / (arm_act + arm_act_r + 1e-9)),
        # legs
        "knee_mean_deg": float(np.rad2deg(q[:, [IDX["left_knee_joint"], IDX["right_knee_joint"]]]).mean()),
        "knee_mean_ref_deg": float(np.rad2deg(rq[:, [IDX["left_knee_joint"], IDX["right_knee_joint"]]]).mean()),
        "ankle_z_min": float(min((pos[:, 2, 2] - pelvis[:, 2]).min(), (pos[:, 3, 2] - pelvis[:, 2]).min())),
        "ankle_z_min_ref": float(min((rpos[:, 2, 2] - rpelvis[:, 2]).min(), (rpos[:, 3, 2] - rpelvis[:, 2]).min())),
        "foot_height_diff": float(np.abs((pos[:, 2, 2] - pelvis[:, 2]) - (pos[:, 3, 2] - pelvis[:, 2])).mean()),
        # torso
        "waist_pitch_deg": float(np.rad2deg(q[:, IDX["waist_pitch_joint"]]).mean()),
        "waist_yaw_range_deg": float(np.rad2deg(np.ptp(q[:, IDX["waist_yaw_joint"]]))),
        "torso_lean_deg": float(tilt.mean()),
        "torso_lean_ref_deg": float(rtilt.mean()),
        # dynamics
        **dynamics(q, dq, dt),
        "speed_ref_rad_s": float(np.abs(rdq).mean()),
        "arm_speed_rad_s": max(arm_act, arm_act_r),
    }
    rows.append(record)

  print(f"[INFO] sampled segments={len(rows)} rows_per_segment~{len(ticks)}", flush=True)
  print(f"[INFO] joint-order check: mean |q_achieved - q_reference| at first sampled row "
        f"= {np.mean(order_check):.4f} rad over {len(order_check)} segments", flush=True)

  # Reference the tracker was asked to follow, per axis, alongside achieved.
  pairs = [
      ("yaw_rate_deg_s", "yaw_rate_ref_deg_s", "root yaw rate", "deg/s"),
      ("shoulder_pitch_L_deg", "shoulder_pitch_L_ref_deg", "left shoulder pitch", "deg"),
      ("shoulder_pitch_R_deg", "shoulder_pitch_R_ref_deg", "right shoulder pitch", "deg"),
      ("wrist_z_max", "wrist_z_max_ref", "highest wrist above pelvis", "m"),
      ("knee_mean_deg", "knee_mean_ref_deg", "mean knee flexion", "deg"),
      ("ankle_z_min", "ankle_z_min_ref", "lowest ankle below pelvis", "m"),
      ("speed_rad_s", "speed_ref_rad_s", "mean joint speed", "rad/s"),
      ("yaw_signed_deg", None, "net heading change (left +, right -)", "deg"),
      ("yaw_path_deg", None, "heading path length", "deg"),
      ("arm_asym", None, "arm activity asymmetry (right +)", "-"),
      ("foot_height_diff", None, "mean foot height difference", "m"),
      ("torso_lean_deg", "torso_lean_ref_deg", "torso tilt from vertical", "deg"),
      ("reversals", None, "joint direction reversals", "count"),
  ]

  def hist(values: np.ndarray, lo: float, hi: float, bins: int = 16) -> list[int]:
    counts, _ = np.histogram(np.clip(values, lo, hi), bins=bins, range=(lo, hi))
    return counts.tolist()

  report: dict = {"sampled_segments": len(rows), "axes": {}}
  for achieved_key, ref_key, label, unit in pairs:
    a = np.array([r[achieved_key] for r in rows], dtype=np.float64)
    b = (
        np.array([r[ref_key] for r in rows], dtype=np.float64)
        if ref_key is not None
        else np.empty(0, dtype=np.float64)
    )
    finite = np.isfinite(a) & (np.isfinite(b) if b.size else True)
    a = a[finite]
    b = b[finite] if b.size else b
    lo = float(min(a.min(), b.min())) if b.size else float(a.min())
    hi = float(max(a.max(), b.max())) if b.size else float(a.max())
    bin_edges = np.linspace(lo, hi, 17).tolist()
    bin_ids = np.clip(np.digitize(a, bin_edges[1:-1]), 0, 15)
    fail = np.array([r["failures"] > 0 for r in rows if np.isfinite(r[achieved_key])])
    per_bin = [
        {
            "lo": bin_edges[i], "hi": bin_edges[i + 1],
            "achieved": int((bin_ids == i).sum()),
            "reference": int(hist(b, lo, hi)[i]) if b.size else None,
            "fail_rate": float(fail[bin_ids == i].mean()) if (bin_ids == i).any() else None,
        }
        for i in range(16)
    ]
    share = np.array([p["achieved"] for p in per_bin], dtype=np.float64)
    share = share / max(share.sum(), 1)
    entropy = float(-(share[share > 0] * np.log(share[share > 0])).sum() / np.log(16))
    report["axes"][label] = {
        "unit": unit,
        "quantiles": {
            "achieved": np.percentile(a, [5, 50, 95]).round(4).tolist(),
            "reference": (
                np.percentile(b, [5, 50, 95]).round(4).tolist() if b.size else None
            ),
        },
        "entropy_norm": round(entropy, 4),
        "max_bin_share": round(float(share.max()), 4),
        "bins": per_bin,
    }

  artifact = {
      "group": dict(Counter(r["group"] for r in rows)),
      "split": dict(Counter(r["split"] for r in rows)),
      "rows": {
          "p5": float(np.percentile([r["rows"] for r in rows], 5)),
          "p50": float(np.percentile([r["rows"] for r in rows], 50)),
          "p95": float(np.percentile([r["rows"] for r in rows], 95)),
      },
      "phase": {
          "p5": round(float(np.percentile([r["phase"] for r in rows], 5)), 3),
          "p50": round(float(np.percentile([r["phase"] for r in rows], 50)), 3),
          "p95": round(float(np.percentile([r["phase"] for r in rows], 95)), 3),
      },
      "arm_asym": {
          "p5": round(float(np.percentile([r["arm_asym"] for r in rows], 5)), 3),
          "p50": round(float(np.percentile([r["arm_asym"] for r in rows], 50)), 3),
          "p95": round(float(np.percentile([r["arm_asym"] for r in rows], 95)), 3),
      },
      "torso_lean_deg_p50": round(float(np.percentile([r["torso_lean_deg"] for r in rows], 50)), 2),
      "foot_height_diff_p50": round(float(np.percentile([r["foot_height_diff"] for r in rows], 50)), 4),
      "reversals_p50": round(float(np.percentile([r["reversals"] for r in rows], 50)), 1),
  }
  report["artifacts"] = artifact
  report["joint_order_check_rad"] = round(float(np.mean(order_check)), 4)

  args.out_json.parent.mkdir(parents=True, exist_ok=True)
  args.out_json.write_text(json.dumps(report, indent=1))

  lines = ["# X2 joint-pose dataset: mode distribution", "",
           f"Sampled {len(rows)} segments (every {args.stride_segments}th), "
           f"every {args.stride_rows}th row.", "",
           f"Joint-order check (achieved vs reference at the first row): "
           f"{report['joint_order_check_rad']:.4f} rad.", ""]
  for label, payload in report["axes"].items():
    qa, qr = payload["quantiles"]["achieved"], payload["quantiles"]["reference"]
    lines += [f"## {label} ({payload['unit']})", "", "| | p5 | p50 | p95 |", "|---|---:|---:|---:|"]
    if qr is not None:
      lines.append(f"| reference | {qr[0]:.3f} | {qr[1]:.3f} | {qr[2]:.3f} |")
    lines += [
        f"| achieved | {qa[0]:.3f} | {qa[1]:.3f} | {qa[2]:.3f} |", "",
        f"normalized entropy {payload['entropy_norm']:.3f}, "
        f"largest bin {payload['max_bin_share']:.1%}", "",
        "| bin | reference | achieved | fail rate |", "|---|---:|---:|---:|",
    ]
    for item in payload["bins"]:
      rate = "-" if item["fail_rate"] is None else f"{item['fail_rate']:.0%}"
      ref = "-" if item["reference"] is None else item["reference"]
      lines.append(
          f"| {item['lo']:.3f} .. {item['hi']:.3f} | {ref} | {item['achieved']} | {rate} |"
      )
    lines.append("")
  lines += ["## Artifacts", "", "```", json.dumps(artifact, indent=1), "```", ""]
  args.out_md.write_text("\n".join(lines))
  print(f"[INFO] wrote {args.out_json} and {args.out_md}", flush=True)


if __name__ == "__main__":
  main()
