"""Add the reference clips as a second, re-weightable window source.

The prepared dataset holds achieved motion: what the tracker did. The reference
clips hold what it was asked to do. They differ systematically, and a generative
model trained only on achieved motion inherits the tracker's bias.

This tool packs the same 144 clips into the same 68-channel window contract, in a
separate directory, together with per-source sampling weights. It writes no
training code and changes no existing artifact. The split follows each clip's
motion assignment, so a motion never appears in one split as reference and in
another as achieved.

Weights are sampling weights, not loss weights: a window from the reference
source is drawn `weight` times as often as a window from the achieved source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare  # noqa: E402  (window packing is reused, not reimplemented)

TENNIS = Path("/home/fushan/mjlab/data/tennis")
DATA = Path("/home/fushan/data/mjlab-latent-free-144/data")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--data-dir", type=Path, default=DATA)
  ap.add_argument("--tennis", type=Path, default=TENNIS)
  ap.add_argument("--name", default="reference")
  ap.add_argument("--weight-achieved", type=float, default=1.0)
  ap.add_argument(
      "--weight-reference",
      type=float,
      default=0.0,
      help="sampling weight; 0 means solve it from --reference-share",
  )
  ap.add_argument(
      "--reference-share",
      type=float,
      default=0.20,
      help="target share of sampled windows that come from the reference source",
  )
  args = ap.parse_args()

  assignments = json.loads((args.data_dir / "assignments.json").read_text())
  split_of = {
      motion: (value if isinstance(value, str) else value["split"])
      for motion, value in assignments["motions"].items()
  }
  stats = np.load(args.data_dir / "projection.npz")
  mean, std = stats["mean"].astype(np.float32), stats["std"].astype(np.float32)

  out_dir = args.data_dir / args.name
  out_dir.mkdir(exist_ok=True)
  per_split: dict[str, list[np.ndarray]] = {"train": [], "validation": [], "test": []}
  per_motion: dict[str, int] = {}
  sources: list[dict] = []

  for motion in sorted(split_of):
    clip_path = args.tennis / motion
    if not clip_path.is_file():
      raise SystemExit(f"reference clip missing: {clip_path}")
    with np.load(clip_path, allow_pickle=False) as z:
      quat = np.asarray(z["body_quat_w"])[:, 0, :].astype(np.float64)
      q = np.asarray(z["joint_pos"]).astype(np.float64)
      dq = np.asarray(z["joint_vel"]).astype(np.float64)
      fps = float(np.asarray(z["fps"]).ravel()[0])
    frames = quat.shape[0]
    if frames < prepare.WINDOW:
      continue
    if not np.isclose(fps, 50.0):
      raise SystemExit(f"{motion}: {fps:g} Hz is not the 50 Hz contract")
    starts = np.arange(frames - prepare.WINDOW + 1, dtype=np.int64)
    windows = prepare.segment_windows(quat, q, dq, starts)
    finite = np.isfinite(windows).all(axis=(1, 2))
    windows = windows[finite]
    normalized = ((windows - mean) / std).astype(np.float32)
    split = split_of[motion]
    per_split[split].append(normalized)
    per_motion[motion] = int(windows.shape[0])
    sources.append({
        "motion": motion,
        "split": split,
        "frames": int(frames),
        "windows": int(windows.shape[0]),
        "sha256": hashlib.sha256(clip_path.read_bytes()).hexdigest(),
    })

  counts: dict[str, int] = {}
  for split, parts in per_split.items():
    if not parts:
      counts[split] = 0
      continue
    block = np.concatenate(parts, axis=0)
    np.save(out_dir / f"{split}.npy", block)
    counts[split] = int(block.shape[0])

  # Implied mixture share, using the achieved window counts of the same dataset.
  achieved = {}
  for split in ("train", "validation", "test"):
    meta = args.data_dir / "offline" / f"tokens-{split}.json"
    if meta.is_file():
      achieved[split] = int(json.loads(meta.read_text())["shape"][0])
    else:
      achieved[split] = 0
  n_achieved = achieved.get("train", 0)
  n_reference = counts.get("train", 0)
  weight_reference = args.weight_reference
  if weight_reference <= 0.0:
    if n_reference == 0 or n_achieved == 0:
      raise SystemExit("cannot solve a reference weight without both window counts")
    share = args.reference_share
    weight_reference = share * n_achieved / ((1.0 - share) * n_reference)

  def share_of(w_ref: float) -> float:
    num = w_ref * n_reference
    return num / (num + args.weight_achieved * n_achieved) if num else 0.0

  manifest = {
      "format": "mjlab-x2-joint-pose-reference-source-v1",
      "contract": {"window": prepare.WINDOW, "channels": prepare.CHANNELS,
                   "target": "reference clip, achieved-free"},
      "normalization": "the dataset's train-only mean and std",
      "split_policy": "follows each motion's assignment; no motion crosses splits",
      "sampling": {
          "weight_achieved": args.weight_achieved,
          "weight_reference": round(weight_reference, 6),
          "train_share_reference": round(share_of(weight_reference), 6),
          "train_share_reference_at_weight_1": round(
              share_of(1.0), 6),
      },
      "windows": counts,
      "achieved_windows": achieved,
      "motions": len(sources),
      "sources": sources,
  }
  (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))

  print(json.dumps({
      "reference_windows": counts,
      "achieved_windows": achieved,
      "weight_reference": round(weight_reference, 4),
      "train_share_reference": round(share_of(weight_reference), 4),
      "share_if_weight_1": round(share_of(1.0), 5),
      "share_if_weight_10": round(share_of(10.0), 4),
      "share_if_weight_50": round(share_of(50.0), 4),
      "normalized_reference_mean_abs": float(
          np.abs(np.concatenate([np.load(out_dir / f"{s}.npy", mmap_mode="r")
                                 for s in ("train",)])).mean()),
  }, indent=1))


if __name__ == "__main__":
  main()
