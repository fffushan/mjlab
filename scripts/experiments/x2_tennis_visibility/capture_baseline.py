"""STEP 0 baseline probe for the ball-visibility gate.

Builds the S9c juggle task at 8 envs, seed 11, zero actions for 40 control steps,
and records the actor's 12 ball-history numbers (obs[:, 100:112]) at reset and
after every step. Saved to /tmp/vis_baseline_before.pt as a (41, 8, 12) tensor.
"""

from __future__ import annotations

import sys

import torch

import mjlab.tasks  # noqa: F401 - registers tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

TASK_ID = (
  "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
  "-Ball3-Pelvis-Nominal-S9c-PoseQuad"
)
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vis_baseline_before.pt"
DEVICE = sys.argv[2] if len(sys.argv) > 2 else None
NUM_ENVS = 8
SEED = 11
STEPS = 40


def main() -> int:
  configure_torch_backends()
  device = DEVICE or ("cuda:0" if torch.cuda.is_available() else "cpu")
  cfg = load_env_cfg(TASK_ID)
  cfg.scene.num_envs = NUM_ENVS
  cfg.seed = SEED
  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
  try:
    obs, _ = env.reset()
    records = [obs["actor"][:, 100:112].to("cpu").clone()]
    action = torch.zeros(
      (env.num_envs, env.action_manager.total_action_dim),
      device=env.device,
      dtype=torch.float32,
    )
    for _ in range(STEPS):
      obs, _, _, _, _ = env.step(action)
      records.append(obs["actor"][:, 100:112].to("cpu").clone())
  finally:
    env.close()
  stack = torch.stack(records)
  torch.save(stack, OUT)
  print(f"device: {device}")
  print(f"actor width: {records[0].shape[1]} (slice rows x 12)")
  print(f"saved: {OUT} shape {tuple(stack.shape)} dtype {stack.dtype}")
  print("final:", stack[-1].tolist())
  return 0


if __name__ == "__main__":
  sys.exit(main())
