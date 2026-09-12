"""Evaluate a trained tracking policy against a local motion file (wandb-free).

The in-repo evaluator (``mjlab.tasks.tracking.scripts.evaluate``) resolves both
the checkpoint and the motion artifact through Weights & Biases, which requires
an account. This script is the local-file alternative: point it at a checkpoint
and a tracking-format npz and it runs the same metrics:

  - MPKPE / root-relative MPKPE (per-body breakdown)
  - joint velocity error
  - end-effector (and ankle-only) position / orientation error
  - success rate over the evaluated horizon, with a histogram of
    termination times when episodes fail

By default the episode length covers the *entire* motion (``episode_length_s``
unset), so the reported success rate is "survived the whole clip without
terminating". Pass ``--episode-length-s`` to cap it (e.g. 10.0 to match the
training horizon).

Example:

  uv run python scripts/evaluate_tracking_policy.py \
    --checkpoint-file logs/rsl_rl/g1_29dof_mode_15_tracking/2026-09-11_20-36-01/model_15998.pt \
    --motion-file data/qianghuo_smplx_unitree_g1_29dof_mode_15_tracking.npz
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand, load_saved_lookahead_s
from mjlab.tasks.tracking.mdp.metrics import (
  compute_ee_orientation_error,
  compute_ee_position_error,
  compute_joint_velocity_error,
  compute_mpkpe,
  compute_root_relative_mpkpe,
)
from mjlab.utils.torch import configure_torch_backends

DEFAULT_TASK = "Mjlab-Tracking-Flat-Unitree-G1-29DOF-Mode-15-No-State-Estimation"


def main(
  checkpoint_file: str,
  motion_file: str,
  task: str = DEFAULT_TASK,
  num_envs: int = 1024,
  episode_length_s: float | None = None,
  device: str | None = None,
) -> None:
  """Evaluate a trained tracking policy and print metrics.

  Args:
    checkpoint_file: Path to a trained ``model_*.pt`` checkpoint.
    motion_file: Path to a tracking-format motion npz (joint_pos, joint_vel,
      body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, fps).
    task: Task id used to build the environment.
    num_envs: Number of parallel environments (= number of evaluation episodes).
    episode_length_s: Episode length in seconds; None = full motion trajectory.
    device: Device to run on; defaults to CUDA if available.
  """
  configure_torch_backends()
  device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  if episode_length_s is None:
    data = np.load(motion_file, allow_pickle=True)
    frames = data["body_pos_w"].shape[0]
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    eval_length_s = frames / fps
    print(
      f"[INFO] Full-trajectory eval: {frames} frames @ {fps:.0f} Hz "
      f"= {eval_length_s:.2f} s"
    )
  else:
    eval_length_s = episode_length_s
  env_cfg = load_env_cfg(task, play=False)
  agent_cfg = load_rl_cfg(task)
  motion_cmd = env_cfg.commands.get("motion")
  if not isinstance(motion_cmd, MotionCommandCfg):
    raise ValueError(f"Task {task} is not a tracking task.")
  motion_cmd.motion_file = motion_file
  saved_lookahead_s = load_saved_lookahead_s(Path(checkpoint_file))
  if saved_lookahead_s is None:
    print("[WARN] No saved params/env.yaml; leaving lookahead_s at default.")
  else:
    motion_cmd.lookahead_s = saved_lookahead_s
  motion_cmd.sampling_mode = "start"
  env_cfg.observations["actor"].enable_corruption = True
  env_cfg.events.pop("push_robot", None)
  env_cfg.scene.num_envs = num_envs
  env_cfg.episode_length_s = eval_length_s

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(checkpoint_file), map_location=device)
  policy = runner.get_inference_policy(device=device)

  command = cast(MotionCommand, env.unwrapped.command_manager.get_term("motion"))
  ee_bodies = env_cfg.terminations["ee_body_pos"].params["body_names"]
  ankle_bodies = ("left_ankle_roll_link", "right_ankle_roll_link")
  body_names = command.cfg.body_names
  n_bodies = len(body_names)

  accum = {
    k: []
    for k in ("mpkpe", "r_mpkpe", "jvel", "ee_pos", "ee_ori", "ee_pos_ankles", "active")
  }
  per_body_sum = torch.zeros(n_bodies, device=device)
  per_body_cnt = torch.zeros(1, device=device)

  done_envs = torch.zeros(num_envs, dtype=torch.bool, device=device)
  success = torch.zeros(num_envs, dtype=torch.bool, device=device)
  fail_steps: list[tuple[int, int]] = []

  obs = env.get_observations()
  step = 0
  while not done_envs.all():
    # Snapshot the reference frame the upcoming step is scored against: the
    # command advances its motion frame *after* the step, so reading the
    # reference afterwards would pair the robot with the next frame.
    ref = SimpleNamespace(
      num_envs=command.num_envs,
      device=command.device,
      cfg=command.cfg,
      body_pos_w=command.body_pos_w.clone(),
      body_pos_relative_w=command.body_pos_relative_w.clone(),
      body_quat_relative_w=command.body_quat_relative_w.clone(),
      joint_vel=command.joint_vel.clone(),
    )
    with torch.no_grad():
      actions = policy(obs)
    obs, _, dones, _ = env.step(actions)
    ref.robot_body_pos_w = command.robot_body_pos_w
    ref.robot_body_quat_w = command.robot_body_quat_w
    ref.robot_joint_vel = command.robot_joint_vel
    ref_command = cast(MotionCommand, ref)

    active = ~done_envs
    accum["active"].append(active.float())
    accum["mpkpe"].append(torch.where(active, compute_mpkpe(ref_command), 0.0))
    accum["r_mpkpe"].append(
      torch.where(active, compute_root_relative_mpkpe(ref_command), 0.0)
    )
    accum["jvel"].append(
      torch.where(active, compute_joint_velocity_error(ref_command), 0.0)
    )
    accum["ee_pos"].append(
      torch.where(active, compute_ee_position_error(ref_command, ee_bodies), 0.0)
    )
    accum["ee_ori"].append(
      torch.where(active, compute_ee_orientation_error(ref_command, ee_bodies), 0.0)
    )
    accum["ee_pos_ankles"].append(
      torch.where(active, compute_ee_position_error(ref_command, ankle_bodies), 0.0)
    )
    per_body = torch.norm(
      ref_command.body_pos_relative_w - ref_command.robot_body_pos_w, dim=-1
    )
    per_body_sum += (per_body * active.unsqueeze(1)).sum(dim=0)
    per_body_cnt += active.sum()

    terminated = env.unwrapped.termination_manager.terminated
    truncated = env.unwrapped.termination_manager.time_outs
    newly_done = dones.bool() & ~done_envs
    if newly_done.any():
      success = success | (newly_done & truncated & ~terminated)
      done_envs = done_envs | newly_done
      failed_now = newly_done & terminated & ~truncated
      fail_steps.append((step, int(failed_now.sum().item())))
    step += 1

  stacks = {k: torch.stack(v, dim=0) for k, v in accum.items()}
  active_steps = stacks["active"].sum(dim=0).clamp(min=1)
  mean = {
    k: (stacks[k].sum(dim=0) / active_steps).mean().item()
    for k in stacks
    if k != "active"
  }

  print("=" * 58)
  print(f"checkpoint: {Path(checkpoint_file).name}")
  print("=" * 58)
  for name, value in mean.items():
    print(f"  {name:14s}: {value:.4f}")
  print(f"  {'success_rate':14s}: {success.float().mean().item():.4f}")
  print(f"  {'steps_taken':14s}: {step}")
  per_body = per_body_sum / per_body_cnt.clamp(min=1)
  print("  root-relative per-body pos error:")
  for i, name in enumerate(body_names):
    tag = " <-- ankle" if "ankle" in name else ""
    print(f"    {name:28s}: {per_body[i].item():.4f} m{tag}")
  if fail_steps:
    weighted = np.repeat([t for t, _ in fail_steps], [c for _, c in fail_steps]) / 50.0
    print(f"  failed envs: {len(weighted)}")
    print(
      f"  failure times (s): min={weighted.min():.1f} "
      f"p25={np.percentile(weighted, 25):.1f} median={np.median(weighted):.1f} "
      f"p75={np.percentile(weighted, 75):.1f} max={weighted.max():.1f}"
    )
    hist, edges = np.histogram(weighted, bins=[0, 10, 20, 30, 40, 46])
    labels = [f"{e:.0f}-{edges[i + 1]:.0f}s" for i, e in enumerate(edges[:-1])]
    print("  failures per 10s bin:", dict(zip(labels, hist.tolist(), strict=True)))
  print("=" * 58)
  env.close()


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
