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

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg


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
  parser.add_argument(
    "--task", default="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"
  )
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument(
    "--lookahead-s",
    type=float,
    default=None,
    help="Motion lookahead used during training. Defaults to params/env.yaml.",
  )
  args = parser.parse_args()

  device = "cpu"

  env_cfg = load_env_cfg(args.task, play=True)
  agent_cfg = load_rl_cfg(args.task)
  env_cfg.scene.num_envs = args.num_envs
  motion_cfg = env_cfg.commands["motion"]
  assert isinstance(motion_cfg, MotionCommandCfg)
  motion_cfg.motion_file = str(args.motion_file)
  if args.lookahead_s is None:
    from mjlab.tasks.tracking.mdp.commands import load_saved_lookahead_s

    saved_lookahead_s = load_saved_lookahead_s(args.checkpoint)
    if saved_lookahead_s is None:
      raise RuntimeError(
        "Could not infer lookahead_s: pass --lookahead-s or keep params/env.yaml "
        "next to the checkpoint."
      )
    motion_cfg.lookahead_s = saved_lookahead_s
  else:
    motion_cfg.lookahead_s = args.lookahead_s

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(args.task)
  assert runner_cls is not None
  runner = runner_cls(env, asdict(agent_cfg), device=device)

  checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
  actor_state_dict = checkpoint["actor_state_dict"]
  target_dim = sum(
    int(np.prod(dim))
    for dim in env.unwrapped.observation_manager.group_obs_term_dim["actor"]
  )
  source_dim = actor_state_dict["mlp.0.weight"].shape[1]
  if source_dim == target_dim:
    runner.alg.get_policy().load_state_dict(actor_state_dict, strict=True)
    print(f"Loaded {source_dim}-input no-state-estimation actor for deployment.")
  elif source_dim == target_dim + 6:
    # A state-estimation tracking actor adds motion_anchor_pos_b (3) after
    # command/lookahead and base_lin_vel (3) before base_ang_vel. Drop those
    # columns so it can run with the deploy controller's NSE observations.
    term_names = env.unwrapped.observation_manager.active_terms["actor"]
    term_dims = [
      int(np.prod(dim))
      for dim in env.unwrapped.observation_manager.group_obs_term_dim["actor"]
    ]
    insert_anchor = sum(
      dim
      for name, dim in zip(term_names, term_dims, strict=True)
      if name in {"command", "motion_lookahead"}
    )
    insert_base_lin_vel = sum(
      dim
      for name, dim in zip(term_names, term_dims, strict=True)
      if name in {"command", "motion_lookahead", "motion_anchor_ori_b"}
    )
    source_drop = {
      *range(insert_anchor, insert_anchor + 3),
      *range(insert_base_lin_vel + 3, insert_base_lin_vel + 6),
    }
    keep = torch.tensor(
      [idx for idx in range(source_dim) if idx not in source_drop], dtype=torch.long
    )
    for key, value in list(actor_state_dict.items()):
      if key == "mlp.0.weight":
        actor_state_dict[key] = value[:, keep]
      elif key.startswith("obs_normalizer._") and key != "obs_normalizer.count":
        actor_state_dict[key] = value[:, keep]
    runner.alg.get_policy().load_state_dict(actor_state_dict, strict=True)
    print(
      f"Converted state-estimation actor from {source_dim} to {target_dim} "
      "deployment observations."
    )
  else:
    raise ValueError(
      f"Checkpoint actor input is {source_dim}, but deployment task expects "
      f"{target_dim} (NSE) or {target_dim + 6} (state estimation)."
    )

  policy = runner.alg.get_policy()
  model = policy.as_onnx(verbose=False)
  model.to("cpu")
  model.eval()

  args.out.parent.mkdir(parents=True, exist_ok=True)
  obs = torch.zeros(1, model.input_size)
  torch.onnx.export(
    model,
    (obs,),
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
