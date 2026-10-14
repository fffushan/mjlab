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

The multi-motion PPO task (``Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion``) takes
a different path: its episodes end at a clip endpoint instead of a reference
wrap, so it is evaluated one explicit clip at a time with a manual-reset loop.
The environment does not auto-reset, every environment plays the same clip from
its first frame, and a finished row is reset only after its terminal metrics are
read. The policy then reads the observation buffer that reset produced. A clean
``clip_complete`` is a full-clip success, an early time-limit exit is not, and a
simultaneous physical failure wins. Add ``--output-json`` to save the result
record:

  uv run python scripts/evaluate_tracking_policy.py \
    --task Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion \
    --checkpoint-file logs/rsl_rl/agibot_x2_multi_motion_tracking/<run>/model_500.pt \
    --motion-file data/tennis/clip_001_agibot_x2_tracking.npz \
    --output-json logs/diagnostics/clip_001_eval.json
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, cast

import numpy as np
import torch
import tyro
from tensordict import TensorDict

import mjlab
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.termination_manager import TerminationManager
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
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
from mjlab.tasks.tracking.mdp.multi_motion_ppo import (
  MultiMotionPpoCommand,
  MultiMotionPpoCommandCfg,
)
from mjlab.utils.torch import configure_torch_backends

DEFAULT_TASK = "Mjlab-Tracking-Flat-Unitree-G1-29DOF-Mode-15-No-State-Estimation"

OUTCOME_COMPLETE = "complete"
OUTCOME_FAILURE = "failure"
OUTCOME_TIMEOUT = "timeout"

CLIP_COMPLETE_TERM = "clip_complete"
"""Registered name of the reference-completion termination term."""


class MultiMotionEvaluationEnv(Protocol):
  """The environment surface the multi-motion evaluation loop uses.

  Production passes the rsl_rl vector-env wrapper; tests pass a fake with the
  same shape.  ``unwrapped`` stays the manager-based environment, whose command,
  termination, and step-rate state the loop reads.
  """

  num_envs: int

  @property
  def unwrapped(self) -> Any: ...

  def step(
    self, actions: torch.Tensor
  ) -> tuple[Any, torch.Tensor, torch.Tensor, dict[str, Any]]: ...

  def get_observations(self) -> Any: ...


def physical_failure_mask(termination: TerminationManager) -> torch.Tensor:
  """Return the rows whose registered physical termination terms fired.

  ``TerminationManager.terminated`` also contains the reference completion, so
  ``terminated & ~complete`` would drop a physical failure that happens on the
  same step as a completion.  The registered terms are the authority: a term is
  a physical failure exactly when it is neither a time limit nor the reference
  completion.
  """
  mask = torch.zeros(termination.num_envs, dtype=torch.bool, device=termination.device)
  for name in termination.active_terms:
    if name == CLIP_COMPLETE_TERM or termination.get_term_cfg(name).time_out:
      continue
    mask |= termination.get_term(name)
  return mask


def classify_finished_rows(
  dones: torch.Tensor,
  complete: torch.Tensor,
  failed: torch.Tensor,
  timed_out: torch.Tensor,
) -> list[str | None]:
  """Attribute one step's finished rows to completion, failure, or timeout.

  A physical failure takes precedence over a simultaneous clip completion, and a
  clip completion over a simultaneous time-limit exit, so a full clip that ends
  exactly on its budgeted step is still a success. A row that did not finish has
  no outcome. A finished row that no registered termination term explains
  raises, because that would silently misattribute an episode.
  """
  outcomes: list[str | None] = [None] * int(dones.shape[0])
  for index in dones.nonzero(as_tuple=False).squeeze(-1).tolist():
    if bool(failed[index]):
      outcomes[index] = OUTCOME_FAILURE
    elif bool(complete[index]):
      outcomes[index] = OUTCOME_COMPLETE
    elif bool(timed_out[index]):
      outcomes[index] = OUTCOME_TIMEOUT
    else:
      raise RuntimeError(
        f"environment {index} finished without a completion, physical-failure, "
        "or time-limit termination term"
      )
  return outcomes


def configure_multi_motion_eval_env(
  env_cfg: ManagerBasedRlEnvCfg,
  motion_file: Path,
  checkpoint_file: Path,
  *,
  num_envs: int,
  episode_length_s: float,
  seed: int,
) -> dict[str, Any]:
  """Configure a multi-motion evaluation environment and report its settings.

  The reference is one explicit clip played from its first frame, with the
  preview offset the checkpoint was trained with. Auto-reset is turned off so
  the terminal metrics of a finished row are read before that row is reset; the
  caller resets the finished rows itself. The other settings match the
  single-motion path: actor corruption on, ``push_robot`` removed, and the
  training randomization left as it is.
  """
  motion_cmd = env_cfg.commands.get("motion")
  if not isinstance(motion_cmd, MultiMotionPpoCommandCfg):
    raise ValueError(
      "configure_multi_motion_eval_env needs the multi-motion tracking task."
    )
  motion_cmd.motion_manifest = ""
  motion_cmd.motion_file = str(motion_file)
  motion_cmd.sampling_mode = "start"
  saved_lookahead_s = load_saved_lookahead_s(checkpoint_file)
  if saved_lookahead_s is None:
    print("[WARN] No saved params/env.yaml; leaving lookahead_s at default.")
  else:
    motion_cmd.lookahead_s = saved_lookahead_s
  env_cfg.observations["actor"].enable_corruption = True
  env_cfg.events.pop("push_robot", None)
  env_cfg.scene.num_envs = num_envs
  env_cfg.episode_length_s = episode_length_s
  env_cfg.seed = seed
  env_cfg.auto_reset = False
  return {
    "motion_file": str(motion_file),
    "num_envs": num_envs,
    "episode_length_s": episode_length_s,
    "seed": seed,
    "lookahead_s": float(motion_cmd.lookahead_s),
    "sampling_mode": str(motion_cmd.sampling_mode),
    "auto_reset": bool(env_cfg.auto_reset),
    "actor_corruption": bool(env_cfg.observations["actor"].enable_corruption),
    "push_robot_event": "push_robot" in env_cfg.events,
  }


def _clip_extent(motion_file: Path) -> tuple[int, float]:
  """Return the frame count and rate of one reference clip."""
  data = np.load(motion_file, allow_pickle=True)
  frames = int(data["body_pos_w"].shape[0])
  fps = float(np.asarray(data["fps"]).reshape(-1)[0])
  return frames, fps


def _distribution(values: list[float]) -> dict[str, float] | None:
  """Return a small min/quartile/max summary, or ``None`` when empty."""
  if not values:
    return None
  array = np.asarray(values, dtype=np.float64)
  return {
    "min": float(array.min()),
    "p25": float(np.percentile(array, 25)),
    "median": float(np.median(array)),
    "p75": float(np.percentile(array, 75)),
    "max": float(array.max()),
  }


def evaluate_multi_motion_ppo(
  env_cfg: ManagerBasedRlEnvCfg,
  agent_cfg: RslRlBaseRunnerCfg,
  *,
  checkpoint_file: Path,
  motion_file: Path,
  task: str,
  num_envs: int,
  episode_length_s: float | None,
  device: str,
  seed: int,
  output_json: Path | None,
) -> dict[str, Any]:
  """Evaluate one checkpoint on one clip of the multi-motion PPO teacher.

  Every environment plays the same clip from its first frame. An episode ends at
  the clip endpoint, at a physical failure, or at the time limit; a clip of ``N``
  frames takes ``N`` steps, so a surviving row reaches its endpoint on the last
  budgeted step. The environment does not auto-reset, so the terminal step is
  scored before its row is reset, and every row the environment marks done is
  reset to keep the unfinished rows stepping.
  """
  frames, fps = _clip_extent(motion_file)
  budget_s = episode_length_s if episode_length_s is not None else frames / fps
  settings = configure_multi_motion_eval_env(
    env_cfg,
    motion_file,
    checkpoint_file,
    num_envs=num_envs,
    episode_length_s=budget_s,
    seed=seed,
  )
  print(f"[INFO] Full-clip budget: {frames} frames @ {fps:.0f} Hz = {budget_s:.2f} s")
  print(f"[INFO] environment seed: {env_cfg.seed}")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(checkpoint_file), map_location=device)
  policy = runner.get_inference_policy(device=device)

  command = env.unwrapped.command_manager.get_term("motion")
  if not isinstance(command, MultiMotionPpoCommand):
    raise RuntimeError(
      f"Task {task} built a {type(command).__name__} instead of the "
      "multi-motion PPO command."
    )
  outcome = run_multi_motion_evaluation(env, command, policy)
  env.close()

  result: dict[str, Any] = {
    "task": task,
    "checkpoint": str(checkpoint_file),
    "clip": {"file": str(motion_file), "frames": frames, "fps": fps},
    "episodes": num_envs,
    "settings": settings,
    **outcome,
  }
  failure_time = outcome["failure_time_s"]
  failure_phase = outcome["failure_phase_fraction"]

  print("=" * 58)
  print(f"checkpoint: {checkpoint_file.name}")
  print(f"clip: {motion_file.name}")
  print("=" * 58)
  print(f"  {'episodes':14s}: {num_envs}")
  print(f"  {'completed':14s}: {result['completed']}")
  print(f"  {'failed':14s}: {result['failed']}")
  print(f"  {'timed_out':14s}: {result['timed_out']}")
  print(f"  {'completion':14s}: {result['completion_fraction']:.4f}")
  print(f"  {'r_mpkpe':14s}: {result['root_relative_mpkpe_m']:.4f}")
  print(f"  {'jvel':14s}: {result['joint_velocity_error']:.4f}")
  print(f"  {'steps_taken':14s}: {result['steps_taken']}")
  if failure_time is not None and failure_phase is not None:
    print(
      f"  failure times (s): min={failure_time['min']:.1f} "
      f"median={failure_time['median']:.1f} max={failure_time['max']:.1f}"
    )
    print(
      f"  failure phase: min={failure_phase['min']:.3f} "
      f"median={failure_phase['median']:.3f} max={failure_phase['max']:.3f}"
    )
  print("=" * 58)

  if output_json is not None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[INFO] Wrote the evaluation result to {output_json}")
  return result


def run_multi_motion_evaluation(
  env: MultiMotionEvaluationEnv,
  command: MultiMotionPpoCommand,
  policy: Callable[[Any], torch.Tensor],
) -> dict[str, Any]:
  """Run the manual-reset evaluation loop and return the episode record.

  Every active row is scored on every step, including the row that finishes on
  that step, because the environment does not auto-reset: its post-step robot
  state is still the terminal one.  A row that finishes leaves the accounting
  permanently, but every row the environment marks done is reset before the next
  step, including rows whose first episode already ended: the environment
  requires it while auto-reset is off, and an inactive row can fail again after
  its reset.  The policy always reads the observation buffer the last reset
  produced.
  """
  num_envs = int(env.num_envs)
  termination = env.unwrapped.termination_manager
  step_dt = env.unwrapped.step_dt
  device = command.device

  accum: dict[str, list[torch.Tensor]] = {"r_mpkpe": [], "jvel": [], "active": []}
  outcomes: list[dict[str, Any]] = []
  active = torch.ones(num_envs, dtype=torch.bool, device=device)
  obs = env.get_observations()
  step = 0
  while bool(active.any()):
    # Snapshot the reference frame the upcoming step is scored against: the
    # command advances its motion frame *after* the step, so reading the
    # reference afterwards would pair the robot with the next frame.
    ref = SimpleNamespace(
      num_envs=command.num_envs,
      device=command.device,
      cfg=command.cfg,
      body_pos_relative_w=command.body_pos_relative_w.clone(),
      joint_vel=command.joint_vel.clone(),
      frame=command.time_steps.clone(),
      row_length=command.row_lengths.clone(),
    )
    with torch.no_grad():
      actions = policy(obs)
    obs, _, dones, _ = env.step(actions)
    ref.robot_body_pos_w = command.robot_body_pos_w
    ref.robot_joint_vel = command.robot_joint_vel
    ref_command = cast(MotionCommand, ref)

    # Every active row is scored, including the row that finishes on this step:
    # auto-reset is off, so its post-step robot state is still the terminal one.
    accum["active"].append(active.float())
    accum["r_mpkpe"].append(
      torch.where(active, compute_root_relative_mpkpe(ref_command), 0.0)
    )
    accum["jvel"].append(
      torch.where(active, compute_joint_velocity_error(ref_command), 0.0)
    )

    done_now = dones.bool()
    classified = classify_finished_rows(
      done_now & active,
      termination.get_term(CLIP_COMPLETE_TERM),
      physical_failure_mask(termination),
      termination.time_outs,
    )
    first_episode_done = torch.tensor(
      [outcome is not None for outcome in classified],
      dtype=torch.bool,
      device=device,
    )
    for index, outcome in enumerate(classified):
      if outcome is None:
        continue
      clip_frames = int(ref.row_length[index])
      outcomes.append(
        {
          "env": index,
          "outcome": outcome,
          # Elapsed control periods after this step, mirroring the episode
          # length the environment counted.
          "time_s": (step + 1) * step_dt,
          "phase_frame": int(ref.frame[index]),
          "clip_frames": clip_frames,
          "phase_fraction": float(ref.frame[index]) / max(clip_frames - 1, 1),
        }
      )
    active = active & ~first_episode_done
    if done_now.any():
      # The environment requires every row it marks done to be reset before the
      # next step while auto-reset is off.  The reset recomputes the observation
      # buffer, and wrapping its return value hands the policy exactly that
      # buffer instead of the pre-reset one, without a second observation
      # compute.
      obs_dict, _ = env.unwrapped.reset(
        env_ids=done_now.nonzero(as_tuple=False).squeeze(-1)
      )
      obs = TensorDict(obs_dict, batch_size=[num_envs])
    step += 1

  stacks = {name: torch.stack(values, dim=0) for name, values in accum.items()}
  active_steps = stacks["active"].sum(dim=0).clamp(min=1)
  mean = {
    name: (stacks[name].sum(dim=0) / active_steps).mean().item()
    for name in stacks
    if name != "active"
  }
  counts = {
    outcome: sum(1 for row in outcomes if row["outcome"] == outcome)
    for outcome in (OUTCOME_COMPLETE, OUTCOME_FAILURE, OUTCOME_TIMEOUT)
  }
  return {
    "completed": counts[OUTCOME_COMPLETE],
    "failed": counts[OUTCOME_FAILURE],
    "timed_out": counts[OUTCOME_TIMEOUT],
    "completion_fraction": counts[OUTCOME_COMPLETE] / num_envs,
    "failure_time_s": _distribution(
      [row["time_s"] for row in outcomes if row["outcome"] == OUTCOME_FAILURE]
    ),
    "failure_phase_fraction": _distribution(
      [row["phase_fraction"] for row in outcomes if row["outcome"] == OUTCOME_FAILURE]
    ),
    "root_relative_mpkpe_m": mean["r_mpkpe"],
    "joint_velocity_error": mean["jvel"],
    "steps_taken": step,
  }


def main(
  checkpoint_file: str,
  motion_file: str,
  task: str = DEFAULT_TASK,
  num_envs: int = 1024,
  episode_length_s: float | None = None,
  device: str | None = None,
  seed: int = 0,
  output_json: str | None = None,
) -> None:
  """Evaluate a trained tracking policy and print metrics.

  Args:
    checkpoint_file: Path to a trained ``model_*.pt`` checkpoint.
    motion_file: Path to a tracking-format motion npz (joint_pos, joint_vel,
      body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, fps).
    task: Task id used to build the environment. The multi-motion PPO task
      selects the manual-reset, endpoint-completion evaluation below.
    num_envs: Number of parallel environments (= number of evaluation episodes).
    episode_length_s: Episode length in seconds; None = full motion trajectory.
    device: Device to run on; defaults to CUDA if available.
    seed: Environment seed. A fixed default makes repeat runs and comparisons
      between checkpoints of the same policy reproducible; with ``None`` the
      environment draws a random seed and identical runs disagree.
    output_json: Optional path for a small JSON result record. The multi-motion
      path writes it; the single-clip path prints its report.
  """
  configure_torch_backends()
  device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task, play=False)
  if isinstance(env_cfg.commands.get("motion"), MultiMotionPpoCommandCfg):
    evaluate_multi_motion_ppo(
      env_cfg,
      load_rl_cfg(task),
      checkpoint_file=Path(checkpoint_file),
      motion_file=Path(motion_file),
      task=task,
      num_envs=num_envs,
      episode_length_s=episode_length_s,
      device=device,
      seed=seed,
      output_json=Path(output_json) if output_json is not None else None,
    )
    return

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
  env_cfg.seed = seed
  print(f"[INFO] environment seed: {env_cfg.seed}")

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
  weighted = np.repeat([t for t, _ in fail_steps], [c for _, c in fail_steps]) / 50.0
  print(f"  failed envs: {weighted.size}")
  if weighted.size:
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
