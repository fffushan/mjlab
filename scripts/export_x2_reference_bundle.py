"""Write an X2 reference clip bundle for a reference-free tracking policy.

Each clip becomes ``motions/<name>.json`` in the ``mjlab-vae-motion`` v2 format that the X2
controller already reads, plus a ``manifest.json`` that records the source npz, the frame
count and the digests. One policy ONNX can then replay any clip in the bundle.

The joint order and the body order come from the live environment, so they cannot drift
from the policy's own metadata. The reference tables themselves come from the motion npz.

Usage:
  uv run python scripts/export_x2_reference_bundle.py \\
    --motion-dir data/tennis --pattern 'single_00*.npz' \\
    --out <bundle-dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.play import select_multi_motion_playback_clip
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.rl.reference_bundle import (
  ReferenceContract,
  clip_paths,
  write_reference_bundle,
)

DEFAULT_TASK = "Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion"


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--motion-dir", required=True, type=Path)
  parser.add_argument("--pattern", default="*.npz")
  parser.add_argument(
    "--name-pattern",
    default=None,
    help="Regex whose first group is the clip name; defaults to the file stem.",
  )
  parser.add_argument("--out", required=True, type=Path)
  parser.add_argument("--task", default=DEFAULT_TASK)
  parser.add_argument("--device", default="cpu")
  parser.add_argument(
    "--anchor-body-name",
    default=None,
    help="Override the anchor body; defaults to the task's motion command anchor.",
  )
  return parser


def main() -> None:
  args = build_parser().parse_args()
  paths = sorted(args.motion_dir.glob(args.pattern))
  if not paths:
    raise SystemExit(f"no motion files matched {args.pattern} in {args.motion_dir}")
  clips = clip_paths(paths, name_pattern=args.name_pattern)

  env_cfg = load_env_cfg(args.task, play=True)
  motion_cfg = env_cfg.commands["motion"]
  select_multi_motion_playback_clip(motion_cfg, str(paths[0]))
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
  robot = env.scene["robot"]

  anchor = args.anchor_body_name or str(motion_cfg.anchor_body_name)
  contract = ReferenceContract(
    joint_names=tuple(robot.joint_names),
    body_names=tuple(robot.body_names),
    anchor_body_name=anchor,
  )
  manifest = write_reference_bundle(args.out, clips, contract)
  print(json.dumps(manifest["reference_contract"], indent=2))
  print(f"[OK] wrote {len(manifest['clips'])} clips to {args.out / 'motions'}")
  for entry in manifest["clips"]:
    print(
      f"  {entry['name']:12} frames={entry['frames']:5} {entry['duration_s']:6.2f}s"
    )


if __name__ == "__main__":
  main()
