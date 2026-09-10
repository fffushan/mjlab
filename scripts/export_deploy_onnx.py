"""Export a deploy-ready plain ONNX (obs -> actions) for the unitree_rl_mjlab g1_ctrl runtime.

The auto-exported tracking ONNX (from MotionTrackingOnPolicyRunner) wraps the policy with
a time_step input and motion reference buffers. The C++ OrtRunner in
deploy/include/isaaclab/algorithms/algorithms.h builds one input tensor per ONNX input
name and requires every name to exist in the obs map; a "time_step" input therefore
fails with "Input name time_step not found in observations".

This script instead exports the raw rsl_rl as_onnx() model: input ["obs"],
output ["actions"], observation normalizer baked in, deterministic mean output.
Motion reference data is NOT embedded - the C++ State_Mimic reads it from the
motion npz directly (MotionLoader_).

Usage:
  uv run python scripts/export_deploy_onnx.py \
    --checkpoint logs/rsl_rl/g1_tracking/<run>/model_XXXX.pt \
    --motion-file ./qianghuo.npz \
    --out /path/to/deploy/config/policy/mimic/<clip>/exported/policy.onnx
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--checkpoint",
    required=True,
    type=Path,
    help="Path to the trained checkpoint (e.g. model_2999.pt).",
  )
  parser.add_argument(
    "--motion-file",
    required=True,
    type=Path,
    help="Motion npz used for training (needed to build the env).",
  )
  parser.add_argument(
    "--out",
    required=True,
    type=Path,
    help="Output path for policy.onnx (deploy expects 'exported/policy.onnx').",
  )
  parser.add_argument("--task", default="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation")
  parser.add_argument("--num-envs", type=int, default=1)
  args = parser.parse_args()

  device = "cpu"

  env_cfg = load_env_cfg(args.task, play=True)
  agent_cfg = load_rl_cfg(args.task)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.commands["motion"].motion_file = str(args.motion_file)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(args.task)
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
  )

  policy = runner.alg.get_policy()
  model = policy.as_onnx(verbose=False)
  model.to("cpu")
  model.eval()

  args.out.parent.mkdir(parents=True, exist_ok=True)
  obs = torch.zeros(1, model.input_size)
  torch.onnx.export(
    model,
    obs,
    str(args.out),
    export_params=True,
    opset_version=18,
    input_names=model.input_names,
    output_names=model.output_names,
    dynamic_axes={},
    dynamo=False,
  )
  print(f"Exported deploy ONNX -> {args.out}")
  print(f"  inputs:  {model.input_names}  ({model.input_size} dims)")
  print(f"  outputs: {model.output_names}")


if __name__ == "__main__":
  main()
