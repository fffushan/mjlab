"""Export the X2 tennis-juggle velocity policy for the hardware controller.

The controller reads the actor layout, joint order, gains and action scales from the
ONNX metadata, so this script attaches ``get_base_metadata`` and then refuses to
report success unless the written file matches the controller's contract: the six
standing terms followed by ``ball_pos`` last, all observation scales 1.0, ``twist``
commands, and the 31 X2 hinge names in table order.

Usage:
  uv run python scripts/export_x2_tennis_juggle_onnx.py --checkpoint <model.pt> \\
    --out /home/agiuser/projects/x2_docker/x2_controller/x2_rl_deploy_controller/config/rl_model/x2_velocity_tennis.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

DEFAULT_TASK = "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle"
OBS_WIDTH, ACTION_WIDTH = 105, 31
# fmt: off
EXPECTED_OBSERVATION_NAMES = [
  "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions",
  "command", "ball_pos",
]
# Exact order and spelling of x2::kinematics::table::kJointNames in
# x2_rl_deploy_controller/include/x2_rl_deploy_controller/x2_body_table_generated.h.
EXPECTED_JOINT_NAMES = [
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
  "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_yaw_joint",
  "left_wrist_pitch_joint", "left_wrist_roll_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_yaw_joint",
  "right_wrist_pitch_joint", "right_wrist_roll_joint",
  "head_yaw_joint", "head_pitch_joint",
]
REQUIRED_METADATA_KEYS = (
  "joint_names", "default_joint_pos", "joint_stiffness", "joint_damping",
  "action_scale", "observation_names", "observation_terms_scale", "command_names",
)
# fmt: on


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", required=True, type=Path, help="Checkpoint.")
  parser.add_argument("--out", required=True, type=Path, help="Output ONNX path.")
  parser.add_argument("--task", default=DEFAULT_TASK)
  parser.add_argument("--device", default="cpu")
  parser.add_argument(
    "--verify-only",
    action="store_true",
    help="Re-check an existing ONNX without building an env or loading a checkpoint.",
  )
  return parser


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1 << 20), b""):
      digest.update(block)
  return digest.hexdigest()


def check_metadata(metadata: dict[str, str]) -> list[str]:
  """Apply the controller's metadata contract to a written ONNX file."""
  missing = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
  if missing:
    return [f"metadata is missing required key {key!r}" for key in missing]

  problems: list[str] = []
  names = metadata["observation_names"].split(",")
  if names != EXPECTED_OBSERVATION_NAMES:
    problems.append(
      f"observation_names must be {EXPECTED_OBSERVATION_NAMES}, got {names}"
    )
  scales = [float(value) for value in metadata["observation_terms_scale"].split(",")]
  if len(scales) != len(EXPECTED_OBSERVATION_NAMES):
    problems.append(
      f"observation_terms_scale needs {len(EXPECTED_OBSERVATION_NAMES)} values, got {len(scales)}"
    )
  if any(abs(scale - 1.0) > 1e-9 for scale in scales):
    problems.append(f"every observation_terms_scale must be 1.0, got {scales}")
  if metadata["command_names"].split(",") != ["twist"]:
    problems.append(
      f"command_names must be ['twist'], got {metadata['command_names']!r}"
    )

  joints = metadata["joint_names"].split(",")
  if joints != EXPECTED_JOINT_NAMES:
    problems.append(f"joint_names do not match the X2 hinge order: {joints}")
  for key in ("default_joint_pos", "joint_stiffness", "joint_damping", "action_scale"):
    values = [float(value) for value in metadata[key].split(",")]
    if len(values) != ACTION_WIDTH:
      problems.append(f"{key} needs {ACTION_WIDTH} values, got {len(values)}")
    if not all(math.isfinite(value) for value in values):
      problems.append(f"{key} contains a non-finite value")
  return problems


def verify_onnx(path: Path) -> tuple[int, int, dict[str, str], list[str]]:
  """Re-open the written file and test it the way the controller will."""
  if not path.is_file():
    raise SystemExit(f"ONNX file not found: {path}")
  session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
  inputs = {value.name: value.shape for value in session.get_inputs()}
  outputs = {value.name: value.shape for value in session.get_outputs()}
  metadata = session.get_modelmeta().custom_metadata_map
  obs_width = int(inputs.get("obs", [0, 0])[1])
  action_width = int(outputs.get("actions", [0, 0])[1])

  problems = check_metadata(metadata)
  if inputs != {"obs": [1, OBS_WIDTH]}:
    problems.append(f"input must be exactly obs [1, {OBS_WIDTH}], got {inputs}")
  if outputs != {"actions": [1, ACTION_WIDTH]}:
    problems.append(
      f"output must be exactly actions [1, {ACTION_WIDTH}], got {outputs}"
    )
  values = session.run(
    ["actions"], {"obs": np.zeros((1, OBS_WIDTH), dtype=np.float32)}
  )[0]
  if values.shape != (1, ACTION_WIDTH):
    problems.append(f"zero-observation run returned shape {values.shape}")
  if not np.isfinite(values).all():
    problems.append("zero-observation run returned a non-finite action")
  return obs_width, action_width, metadata, problems


def export(args: argparse.Namespace) -> None:
  checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  checkpoint_iter = checkpoint["iter"]
  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.seed = 42
  agent_cfg = load_rl_cfg(args.task)
  runner_cls = load_runner_cls(args.task)
  if runner_cls is None:
    raise SystemExit(f"task {args.task!r} registers no runner class")

  args.out.parent.mkdir(parents=True, exist_ok=True)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
  try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = runner_cls(
      wrapped, asdict(agent_cfg), log_dir=str(args.out.parent), device=args.device
    )
    runner.load(
      str(args.checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=args.device,
    )
    runner.export_policy_to_onnx(str(args.out.parent), args.out.name)
    metadata = get_base_metadata(env, run_path=str(args.checkpoint.parent))
    attach_metadata_to_onnx(str(args.out), metadata)
  finally:
    env.close()
  print(f"checkpoint: {args.checkpoint} (iter {checkpoint_iter})")


def main() -> None:
  args = build_parser().parse_args()
  configure_torch_backends()
  if not args.verify_only:
    export(args)
  obs_width, action_width, metadata, problems = verify_onnx(args.out)
  print(f"onnx: {args.out}")
  print(f"sha256: {sha256_file(args.out)}")
  print(f"observation width: {obs_width}")
  print(f"action width: {action_width}")
  print("metadata:")
  print(json.dumps(metadata, indent=2, sort_keys=True))
  if problems:
    raise SystemExit("export validation failed:\n  " + "\n  ".join(problems))
  print("[OK] controller metadata contract satisfied")


if __name__ == "__main__":
  main()
