"""Build one joint-overlay reference per prepared segment.

For each segment the reference keeps the original clip's body kinematics at the
frames the episode actually visited, and replaces the joint channels with the
achieved values.  This is the joint-only substitution used by the replay pilot,
now applied to the whole prepared dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DATA = Path("/home/fushan/data/mjlab-latent-free-144/data")
TENNIS = Path("/home/fushan/mjlab/data/tennis")
OUT = Path("/tmp/wd-refs")


def load_clip_arrays(
    name: str, cache: dict[str, dict[str, np.ndarray]]
) -> dict[str, np.ndarray]:
  """Return the clip's body arrays, keeping only the current clip resident."""
  if name not in cache:
    with np.load(TENNIS / name, allow_pickle=False) as z:
      cache.clear()
      cache[name] = {k: np.asarray(z[k]) for k in (
          "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "fps")}
  return cache[name]


def initial_frame_table(path: str, cache: dict[str, np.ndarray]) -> np.ndarray:
  """Return one source file's per-trajectory start frames."""
  if path not in cache:
    with np.load(path, allow_pickle=False) as z:
      cache.clear()
      cache[path] = np.asarray(z["trajectory_initial_frame"], dtype=np.int64)
  return cache[path]


def write_reference(
    *,
    out_dir: Path,
    entry: dict,
    q: np.ndarray,
    dq: np.ndarray,
    clip: dict[str, np.ndarray],
    init: int,
) -> tuple[Path, int]:
  """Write one reference and return its path and usable row count.

  The loader clamps the reference row at the last frame, so an achieved
  trajectory of n rows maps onto min(init + t, total - 1).
  """
  seg = int(entry["segment"])
  n = int(entry["rows"])
  lo = int(entry["row_start"])
  total = clip["body_pos_w"].shape[0]
  rows_idx = np.minimum(init + np.arange(n), total - 1)
  path = out_dir / f"s{seg:06d}.npz"
  np.savez(
      path,
      joint_pos=np.ascontiguousarray(q[lo : lo + n]),
      joint_vel=np.ascontiguousarray(dq[lo : lo + n]),
      body_pos_w=clip["body_pos_w"][rows_idx],
      body_quat_w=clip["body_quat_w"][rows_idx],
      body_lin_vel_w=clip["body_lin_vel_w"][rows_idx],
      body_ang_vel_w=clip["body_ang_vel_w"][rows_idx],
      fps=np.array([float(np.asarray(clip["fps"]).ravel()[0])], np.float32),
  )
  return path, n


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--limit", type=int, default=0)
  ap.add_argument("--stride", type=int, default=1)
  ap.add_argument("--out", type=Path, default=OUT)
  args = ap.parse_args()

  store = DATA / "store"
  table = json.loads((store / "manifest.json").read_text())["table"]
  if args.stride > 1:
    table = table[:: args.stride]
  if args.limit:
    table = table[: args.limit]

  q = np.load(store / "joint_pos.npy", mmap_mode="r")
  dq = np.load(store / "joint_vel.npy", mmap_mode="r")

  args.out.mkdir(parents=True, exist_ok=True)
  clip_cache: dict[str, dict[str, np.ndarray]] = {}
  src_cache: dict[str, np.ndarray] = {}
  index = []
  problems = []
  clamp_rows = 0

  def load_clip(name: str) -> dict[str, np.ndarray]:
    return load_clip_arrays(name, clip_cache)

  def initial_frames(path: str) -> np.ndarray:
    return initial_frame_table(path, src_cache)

  for entry in table:
    seg = int(entry["segment"])
    n = int(entry["rows"])
    lo = int(entry["row_start"])
    clip_name = entry["motion"]
    clip = load_clip(clip_name)
    init = int(initial_frames(entry["source"])[int(entry["trajectory_index"])])
    fps = float(np.asarray(clip["fps"]).ravel()[0])
    total = clip["body_pos_w"].shape[0]
    # The loader clamps the reference row at the last frame
    # (multi_motion_ppo.py: time_steps = min(t + 1, row_length - 1)), so an
    # achieved trajectory of n rows maps onto min(init + t, total - 1).
    rows_idx = np.minimum(init + np.arange(n), total - 1)
    clamped = int(np.count_nonzero(init + np.arange(n) > total - 1))
    if clamped:
      clamp_rows += clamped
    name = f"s{seg:06d}.npz"
    np.savez(
        args.out / name,
        joint_pos=np.ascontiguousarray(q[lo : lo + n]),
        joint_vel=np.ascontiguousarray(dq[lo : lo + n]),
        body_pos_w=clip["body_pos_w"][rows_idx],
        body_quat_w=clip["body_quat_w"][rows_idx],
        body_lin_vel_w=clip["body_lin_vel_w"][rows_idx],
        body_ang_vel_w=clip["body_ang_vel_w"][rows_idx],
        fps=np.array([fps], np.float32),
    )
    index.append({"file": name, "segment": seg, "motion": clip_name,
                  "group": entry["group"], "seed": int(entry["seed"]),
                  "env_index": int(entry["env_index"]),
                  "trajectory_index": int(entry["trajectory_index"]),
                  "initial_frame": init, "rows": n,
                  "windows": int(entry["windows"]), "source": entry["source"]})

  (args.out / "index.json").write_text(json.dumps(index, indent=0))
  summary = {"refs": len(index), "problems": len(problems),
             "clamped_tail_rows": clamp_rows,
             "rows": sum(e["rows"] for e in index),
             "windows": sum(e["windows"] for e in index),
             "groups": {g: sum(1 for e in index if e["group"] == g)
                        for g in sorted({e["group"] for e in index})}}
  print(json.dumps(summary))
  if problems:
    (args.out / "problems.json").write_text(json.dumps(problems[:200], indent=1))
    print("PROBLEM_SAMPLE", json.dumps(problems[:3]))


if __name__ == "__main__":
  main()
