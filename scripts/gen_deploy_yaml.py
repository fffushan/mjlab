"""Generate deploy.yaml for the unitree_rl_mjlab g1_ctrl mimic FSM from our run's ONNX metadata.

Writes a deploy.yaml compatible with deploy/robots/g1/config/policy/mimic/<clip>/params/deploy.yaml
using the observation/action layout of the NSE tracking task (matches the reference
dance1_subject2 deploy.yaml shipped in unitree_rl_mjlab).

The observation dimension depends on the policy's lookahead setting:
  - lookahead_s > 0 : adds the 58-dim motion_lookahead observation -> 212-dim obs
  - lookahead_s <= 0: no motion_lookahead term                     -> 154-dim obs
lookahead_s is taken from --lookahead-s if given, otherwise inferred from the
run's params/env.yaml (0.0 if the field is absent).

Usage:
  uv run python scripts/gen_deploy_yaml.py \
    --onnx logs/rsl_rl/g1_tracking/<run>/<run>.onnx \
    --out /path/to/deploy/config/policy/mimic/<clip>/params/deploy.yaml
"""

import argparse
import re
from pathlib import Path
from typing import Any

import onnx
import yaml


def _csv_to_list(s: str) -> list[float]:
  return [float(x) for x in s.split(",")]


def _load_lookahead_s(onnx_path: Path) -> float | None:
  """Infer lookahead_s from the run's params/env.yaml (None if not found)."""
  env_yaml = onnx_path.parent / "params" / "env.yaml"
  if not env_yaml.exists():
    return None
  m = re.search(r"^\s*lookahead_s:\s*([0-9.]+)\s*$", env_yaml.read_text(), re.M)
  return float(m.group(1)) if m else None


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
    default=None,
    help=(
      "Lookahead duration used by the trained policy. Defaults to the "
      "lookahead_s field of the run's params/env.yaml, or 0.0 if absent. "
      ">0 adds the 58-dim motion_lookahead observation (212-dim obs); "
      "0.0 omits it (154-dim obs)."
    ),
  )
  args = parser.parse_args()

  lookahead_s = args.lookahead_s
  if lookahead_s is None:
    lookahead_s = _load_lookahead_s(args.onnx)
    if lookahead_s is None:
      lookahead_s = 0.0

  model = onnx.load(str(args.onnx))
  meta = {p.key: p.value for p in model.metadata_props}

  n_joints = 29
  assert len(meta["joint_names"].split(",")) == n_joints

  # NSE tracking obs layout: order and dims must match the trained policy.
  # The 58-dim motion_lookahead term exists only for lookahead policies.
  obs_terms: list[tuple[str, dict[str, Any]]] = [
    (
      "motion_command",
      {
        "params": {"command_name": "motion"},
        "clip": None,
        "scale": [1.0] * 58,
        "history_length": 1,
      },
    ),
  ]
  if lookahead_s > 0:
    obs_terms.append(
      (
        "motion_lookahead",
        {
          "params": {"command_name": "motion", "lookahead_s": lookahead_s},
          "clip": None,
          "scale": [1.0] * 58,
          "history_length": 1,
        },
      )
    )
  obs_terms += [
    (
      "motion_anchor_ori_b",
      {
        "params": {"command_name": "motion"},
        "clip": None,
        "scale": [1.0] * 6,
        "history_length": 1,
      },
    ),
    (
      "base_ang_vel",
      {
        "params": {},
        "clip": None,
        "scale": [1.0] * 3,
        "history_length": 1,
      },
    ),
    (
      "joint_pos_rel",
      {
        "params": {},
        "clip": None,
        "scale": [1.0] * n_joints,
        "history_length": 1,
      },
    ),
    (
      "joint_vel_rel",
      {
        "params": {},
        "clip": None,
        "scale": [1.0] * n_joints,
        "history_length": 1,
      },
    ),
    (
      "last_action",
      {
        "params": {},
        "clip": None,
        "scale": [1.0] * n_joints,
        "history_length": 1,
      },
    ),
  ]

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
    "observations": dict(obs_terms),
  }

  args.out.parent.mkdir(parents=True, exist_ok=True)
  with open(args.out, "w") as f:
    yaml.dump(deploy_yaml, f, default_flow_style=None, sort_keys=False, width=200)
  print(f"Wrote deploy.yaml -> {args.out}")

  # Sanity: obs dims sum
  total = sum(len(v["scale"]) for v in deploy_yaml["observations"].values())
  expected = 212 if lookahead_s > 0 else 154
  print(f"  obs dims total: {total} (expect {expected}, lookahead_s={lookahead_s})")
  assert total == expected, (
    f"Expected {expected}-dim NSE obs (lookahead_s={lookahead_s}), got {total}"
  )


if __name__ == "__main__":
  main()
