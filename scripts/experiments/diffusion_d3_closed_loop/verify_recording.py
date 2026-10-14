"""Verify that a recorded state file rebuilds the body pose exactly.

A recording from ``evaluate_parallel.py --record-states DIR`` holds the root
pose and (with ``--record-bodies``) the simulator's body positions.  This tool
writes the recorded root pose and joint positions back into a fresh
environment, forwards the kinematics, and compares the rebuilt body positions
against the recorded ones.

A small difference, at the float32 rounding level, proves that the recorded
channels rebuild the pose and that a playback tool can render the motion.  A
large difference means the recording does not describe the robot.

Usage:
  verify_recording.py RECORDING.npz [--member vgk000] [--device cuda:0]
                                     [--ticks 0,10,50] [--tolerance 1e-4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mjlab.tasks.tracking.diffusion.parallel_closed_loop_x2 import (  # noqa: E402
  build_parallel_closed_loop_runtime,
)

_MANIFEST = "configs/distillation/x2_tennis_vendor_gains_dr_waist_kd4_v1.yaml"


def _to_numpy(value: object) -> np.ndarray:
  if isinstance(value, torch.Tensor):
    return value.detach().cpu().numpy()
  return np.asarray(value)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("recording", type=Path)
  parser.add_argument("--repo-root", default=".")
  parser.add_argument("--manifest", default=_MANIFEST)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--ticks", default="", help="comma list of slots, default: a spread")
  parser.add_argument("--tolerance", type=float, default=1.0e-4)
  args = parser.parse_args(argv)

  with np.load(args.recording, allow_pickle=False) as payload:
    if "body_positions" not in payload.files:
      raise SystemExit(
        "this recording holds no body positions; record again with --record-bodies"
      )
    ticks = np.array(payload["tick"])
    root_state = np.concatenate(
      [
        payload["root_position"],
        payload["root_quaternion_wxyz"],
        payload["root_linear_velocity"],
        payload["root_angular_velocity"],
      ],
      axis=2,
    )
    joint_position = np.array(payload["joint_position"])
    joint_velocity = np.array(payload["joint_velocity"])
    truth = np.array(payload["body_positions"])
    member = str(payload["member"])
    rows = int(payload["rows"])
    recorded_rows = int(payload["recorded_rows"])
  if args.ticks:
    slots = [int(item) for item in args.ticks.split(",")]
  else:
    slots = sorted({0, len(ticks) // 2, len(ticks) - 1})
  rows_in_file = root_state.shape[1]

  runtime = build_parallel_closed_loop_runtime(
    manifest_path=args.manifest,
    repo_root=args.repo_root,
    member=member,
    device=args.device,
    seed=0,
    num_envs=rows_in_file,
  )
  try:
    robot = runtime.env.scene["robot"]
    device = runtime.env.device
    env_ids = torch.arange(rows_in_file, dtype=torch.int64, device=device)
    body_index = list(runtime._body_indexes)
    print(
      f"recording {args.recording}: member={member} rows={rows} "
      f"recorded_rows={recorded_rows} ticks={len(ticks)}"
    )
    worst = 0.0
    for slot in slots:
      if not 0 <= slot < len(ticks):
        raise SystemExit(f"slot {slot} is outside the {len(ticks)} recorded ticks")
      robot.write_joint_state_to_sim(
        torch.as_tensor(joint_position[slot], dtype=torch.float32, device=device),
        torch.as_tensor(joint_velocity[slot], dtype=torch.float32, device=device),
        env_ids=env_ids,
      )
      robot.write_root_state_to_sim(
        torch.as_tensor(root_state[slot], dtype=torch.float32, device=device),
        env_ids=env_ids,
      )
      runtime.env.scene.write_data_to_sim()
      runtime.env.sim.forward()
      runtime.env.sim.sense()
      rebuilt = _to_numpy(robot.data.body_link_pos_w)[:, body_index, :]
      difference = float(np.max(np.abs(rebuilt - truth[slot].reshape(rebuilt.shape))))
      worst = max(worst, difference)
      print(f"  tick {int(ticks[slot]):4d}  max_body_difference_m {difference:.3e}")
    verdict = "PASS" if worst <= args.tolerance else "FAIL"
    print(
      f"worst {worst:.3e} m against tolerance {args.tolerance:.0e} m: {verdict}"
    )
    return 0 if verdict == "PASS" else 1
  finally:
    runtime.close()


if __name__ == "__main__":
  raise SystemExit(main())
