"""Generate deploy.yaml for the unitree_rl_mjlab g1_ctrl mimic FSM from our run's ONNX metadata.

Writes a deploy.yaml compatible with deploy/robots/g1/config/policy/mimic/<clip>/params/deploy.yaml
using the observation/action layout of the NSE tracking task (matches the reference
dance1_subject2 deploy.yaml shipped in unitree_rl_mjlab).

Usage:
  uv run python scripts/gen_deploy_yaml.py \
    --onnx logs/rsl_rl/g1_tracking/<run>/<run>.onnx \
    --out /path/to/deploy/config/policy/mimic/<clip>/params/deploy.yaml
"""

import argparse
from pathlib import Path
from typing import Any

import onnx
import yaml


def _csv_to_list(s: str) -> list[float]:
  return [float(x) for x in s.split(",")]


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--onnx",
    required=True,
    type=Path,
    help="Auto-exported tracking ONNX (has metadata).",
  )
  parser.add_argument(
    "--out", required=True, type=Path, help="Output deploy.yaml path."
  )
  parser.add_argument(
    "--lookahead-s",
    type=float,
    default=0.5,
    help="Lookahead duration used by the trained policy (default: 0.5).",
  )
  args = parser.parse_args()

  model = onnx.load(str(args.onnx))
  meta = {p.key: p.value for p in model.metadata_props}

  n_joints = 29
  assert len(meta["joint_names"].split(",")) == n_joints

  deploy_yaml: dict[str, Any] = {
    "joint_ids_map": list(range(n_joints)),
    "step_dt": 0.02,  # 50 Hz control (0.005 s physics x decimation 4)
    "stiffness": _csv_to_list(meta["joint_stiffness"]),
    "damping": _csv_to_list(meta["joint_damping"]),
    "default_joint_pos": _csv_to_list(meta["default_joint_pos"]),
    "commands": {},
    "actions": {
      "JointPositionAction": {
        "clip": None,
        "joint_names": [".*"],
        "scale": _csv_to_list(meta["action_scale"]),
        "offset": _csv_to_list(meta["default_joint_pos"]),
        "joint_ids": None,
      }
    },
    # NSE tracking obs layout: order and dims must match the trained policy.
    "observations": {
      "motion_command": {
        "params": {"command_name": "motion"},
        "clip": None,
        "scale": [1.0] * 58,
        "history_length": 1,
      },
      "motion_lookahead": {
        "params": {"command_name": "motion", "lookahead_s": args.lookahead_s},
        "clip": None,
        "scale": [1.0] * 58,
        "history_length": 1,
      },
      "motion_anchor_ori_b": {
        "params": {"command_name": "motion"},
        "clip": None,
        "scale": [1.0] * 6,
        "history_length": 1,
      },
      "base_ang_vel": {
        "params": {},
        "clip": None,
        "scale": [1.0] * 3,
        "history_length": 1,
      },
      "joint_pos_rel": {
        "params": {},
        "clip": None,
        "scale": [1.0] * n_joints,
        "history_length": 1,
      },
      "joint_vel_rel": {
        "params": {},
        "clip": None,
        "scale": [1.0] * n_joints,
        "history_length": 1,
      },
      "last_action": {
        "params": {},
        "clip": None,
        "scale": [1.0] * n_joints,
        "history_length": 1,
      },
    },
  }

  args.out.parent.mkdir(parents=True, exist_ok=True)
  with open(args.out, "w") as f:
    yaml.dump(deploy_yaml, f, default_flow_style=None, sort_keys=False, width=200)
  print(f"Wrote deploy.yaml -> {args.out}")

  # Sanity: obs dims sum
  total = sum(len(v["scale"]) for v in deploy_yaml["observations"].values())
  print(f"  obs dims total: {total} (expect 212)")
  assert total == 212, f"Expected 212-dim lookahead NSE obs, got {total}"


if __name__ == "__main__":
  main()
