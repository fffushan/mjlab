"""Convert a State-Estimation tracking checkpoint to the No-State-Estimation variant.

The NSE task drops ``motion_anchor_pos_b`` (3) and ``base_lin_vel`` (3) from the
actor observations (160 -> 154 dims). The critic observation group is unchanged,
so the critic transfers 1:1. The optimizer state (Adam buffers for the first
layer) has the old shapes, so it is deliberately NOT transferred - the converted
checkpoint carries a fresh optimizer and ``iter=0``, i.e. it is a fine-tune start.

Usage:
  uv run python scripts/convert_checkpoint_nse.py \
    --source logs/rsl_rl/g1_tracking/2026-09-09_08-29-20/model_29999.pt \
    --motion-file ./qianghuo.npz \
    --output-dir logs/rsl_rl/g1_tracking/nse_from_29999

Then resume training:
  uv run train Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation \
    --agent.resume \
    --agent.load-run nse_from_29999 \
    --agent.load-checkpoint model_29999.pt \
    --env.commands.motion.motion-file ./qianghuo.npz
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.utils.torch import configure_torch_backends

SE_TASK = "Mjlab-Tracking-Flat-Unitree-G1"
NSE_TASK = "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"


def build_actor_layout(
  task_id: str, motion_file: Path, device: str
) -> tuple[ManagerBasedRlEnv, list[tuple[str, int]]]:
  """Build the env and return the ordered [(term_name, dim)] actor obs layout."""
  env_cfg = load_env_cfg(task_id, play=False)
  motion = env_cfg.commands.get("motion")
  if not isinstance(motion, MotionCommandCfg):
    raise ValueError(f"Task {task_id} is not a tracking task.")
  motion.motion_file = str(motion_file)
  env_cfg.scene.num_envs = 1

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  obs_mgr = env.observation_manager
  names = obs_mgr.active_terms["actor"]
  term_dims = obs_mgr.group_obs_term_dim["actor"]
  layout = [
    (name, int(np.prod(dims))) for name, dims in zip(names, term_dims, strict=True)
  ]
  return env, layout


def compute_keep_columns(
  old_layout: list[tuple[str, int]], new_layout: list[tuple[str, int]]
) -> list[int]:
  """Column indices (in the old concatenated obs) that survive in the new layout.

  Verifies that no terms are added and that the dropped terms exactly account
  for the dimension difference.
  """
  old_names = [n for n, _ in old_layout]
  new_names = [n for n, _ in new_layout]
  extra = [n for n in new_names if n not in old_names]
  if extra:
    raise ValueError(f"Layout mismatch: new layout adds terms {extra}")

  old_dim = sum(d for _, d in old_layout)
  new_dim = sum(d for _, d in new_layout)
  dropped_dim = sum(d for n, d in old_layout if n not in new_names)
  if old_dim - dropped_dim != new_dim:
    raise ValueError(
      f"Layout mismatch: old_dim={old_dim}, dropped={dropped_dim}, new_dim={new_dim}"
    )

  offsets: dict[str, tuple[int, int]] = {}
  cursor = 0
  for name, dim in old_layout:
    offsets[name] = (cursor, cursor + dim)
    cursor += dim

  keep: list[int] = []
  for name, _ in new_layout:
    start, end = offsets[name]
    keep.extend(range(start, end))
  return keep


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--source", type=Path, required=True, help="Source model_*.pt checkpoint."
  )
  parser.add_argument("--motion-file", type=Path, default=Path("./qianghuo.npz"))
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("logs/rsl_rl/g1_tracking/nse_from_29999"),
    help="Run directory for the converted checkpoint (must live under the log root).",
  )
  parser.add_argument(
    "--output-name",
    type=str,
    default=None,
    help="Checkpoint file name (default: source stem).",
  )
  args = parser.parse_args()

  configure_torch_backends()
  device = "cpu"

  # 1. Build both envs and read the true obs layouts.
  print(f"[INFO] Building {SE_TASK} env (obs layout)...")
  old_env, old_layout = build_actor_layout(SE_TASK, args.motion_file, device)
  old_dim = sum(d for _, d in old_layout)
  print(f"[INFO] Old actor obs terms: {old_layout} (total {old_dim})")

  print(f"[INFO] Building {NSE_TASK} env (obs layout)...")
  new_env, new_layout = build_actor_layout(NSE_TASK, args.motion_file, device)
  new_dim = sum(d for _, d in new_layout)
  print(f"[INFO] New actor obs terms: {new_layout} (total {new_dim})")

  keep = compute_keep_columns(old_layout, new_layout)
  print(f"[INFO] Keeping old obs columns: {keep} ({len(keep)} dims)")

  # 2. Load source checkpoint and trim the actor.
  ckpt = torch.load(args.source, map_location="cpu", weights_only=False)
  actor_sd = ckpt["actor_state_dict"]
  if actor_sd["mlp.0.weight"].shape[1] != old_dim:
    raise ValueError(
      f"Checkpoint actor input {actor_sd['mlp.0.weight'].shape[1]} != env obs dim {old_dim}"
    )

  keep_t = torch.tensor(keep, dtype=torch.long)
  new_actor_sd: dict[str, torch.Tensor] = {}
  for key, value in actor_sd.items():
    if key == "mlp.0.weight":
      new_actor_sd[key] = value[:, keep_t]
    elif key.startswith("obs_normalizer._") and key != "obs_normalizer.count":
      new_actor_sd[key] = value[:, keep_t]
    else:
      new_actor_sd[key] = value

  # 3. Build the NSE runner and load converted weights into it.
  args.output_dir.mkdir(parents=True, exist_ok=True)
  new_env = RslRlVecEnvWrapper(new_env, clip_actions=None)
  agent_cfg = asdict(load_rl_cfg(NSE_TASK))
  runner = MotionTrackingOnPolicyRunner(
    new_env, agent_cfg, str(args.output_dir), device
  )

  actor = runner.alg.get_policy()
  actor.load_state_dict(new_actor_sd, strict=True)
  print(
    f"[INFO] Actor loaded: mlp.0.weight {actor.mlp[0].weight.shape}, "
    f"normalizer mean {actor.obs_normalizer._mean.shape}"
  )
  # Sanity: the new actor must not reference the dropped channels anymore.
  assert actor.mlp[0].weight.shape == (512, new_dim), actor.mlp[0].weight.shape
  critic = runner.alg._raw_critic  # type: ignore[attr-defined]
  critic.load_state_dict(ckpt["critic_state_dict"], strict=True)
  print("[INFO] Critic transferred 1:1 (obs unchanged).")

  # 4. Save through the real runner: fresh optimizer + converted weights,
  #    plus a fresh NSE ONNX export as a sanity artifact.
  output_name = args.output_name or args.source.name
  checkpoint_path = args.output_dir / output_name
  runner.save(str(checkpoint_path))
  print(f"[INFO] Saved converted checkpoint: {checkpoint_path}")

  # 5. Verify the artifact on disk.
  reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  assert reloaded["actor_state_dict"]["mlp.0.weight"].shape == (512, new_dim)
  assert reloaded["actor_state_dict"]["obs_normalizer._mean"].shape == (1, new_dim)
  # Critic obs group is unchanged between SE/NSE, so its input dim must equal
  # the source checkpoint's critic input dim.
  old_critic_dim = ckpt["critic_state_dict"]["mlp.0.weight"].shape[1]
  new_critic_dim = reloaded["critic_state_dict"]["mlp.0.weight"].shape[1]
  assert old_critic_dim == new_critic_dim, (old_critic_dim, new_critic_dim)
  print(f"[VERIFY] mlp.0.weight {reloaded['actor_state_dict']['mlp.0.weight'].shape}")
  print(
    f"[VERIFY] critic mlp.0.weight {reloaded['critic_state_dict']['mlp.0.weight'].shape}"
  )
  print(f"[VERIFY] iter={reloaded['iter']} (fresh run)")

  old_env.close()
  new_env.close()


if __name__ == "__main__":
  main()
