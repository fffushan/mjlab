"""Collect raw achieved X2 poses from the frozen multi-motion PPO teacher.

This is a research data collector for the latent-free joint-space pipeline.  It
rolls out the existing X2 multi-motion tracking teacher on one or more explicit
reference clips and writes the achieved simulator state:

* root-link quaternion in the world/simulator frame, ``wxyz``;
* measured joint positions in the robot joint order;
* measured joint velocities in the robot joint order.

No heading alignment, 6D rotation packing, reference adapter, VAE, flow model,
or training pipeline is applied here.  Those belong to later preprocessing.
Reference command values and policy observations are deliberately not saved as
achieved state.

Example:

  PYTHONPATH="$PWD/src" uv run --no-sync python \\
    scripts/experiments/joint_pose_collection/collect.py \\
    --checkpoint-file <teacher-run>/model_19999.pt \\
    --motion-file data/tennis/single_000_zhanghongyu_agibot_x2_tracking.npz \\
    --output-dir logs/joint_pose_collection/run_000 \\
    --num-envs 64 --seed 0
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
import tyro
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.termination_manager import TerminationManager
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp.commands import load_saved_lookahead_s
from mjlab.tasks.tracking.mdp.multi_motion_ppo import (
  MultiMotionPpoCommandCfg,
  repository_root,
)
from mjlab.utils.torch import configure_torch_backends

DEFAULT_TASK = "Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion"
SCHEMA_VERSION = 1
OUTCOME_COMPLETE = "complete"
OUTCOME_FAILURE = "failure"
OUTCOME_BUDGET = "budget"
CLIP_COMPLETE_TERM = "clip_complete"
"""Registered termination name of the multi-motion reference endpoint."""


class CollectionEnv(Protocol):
  """The vector-env surface the collection loop uses."""

  num_envs: int

  @property
  def unwrapped(self) -> Any: ...

  def get_observations(self) -> Any: ...

  def step(
    self, actions: torch.Tensor
  ) -> tuple[Any, torch.Tensor, torch.Tensor, dict[str, Any]]: ...

  def close(self) -> None: ...


@dataclass(frozen=True)
class CollectionConfig:
  """One bounded raw joint-pose collection run."""

  checkpoint_file: Path
  """Frozen tracking-policy checkpoint (model_*.pt)."""

  motion_file: list[Path]
  """One or more tracking-format reference clips."""

  output_dir: Path
  """New output directory. The collector never overwrites an existing path."""

  task: str = DEFAULT_TASK
  """Registered multi-motion tracking task."""

  num_envs: int = 64
  """Environment rows, and therefore trajectories, per reference clip."""

  seed: int = 0
  """Environment seed, applied to every per-clip environment."""

  max_steps: int | None = None
  """Control-step cap per trajectory. None, or a value at least the clip length,
  uses the full clip. A value below the clip length stops and labels the
  unfinished trajectory budget."""

  device: str | None = None
  """Torch device. Defaults to CUDA when available, otherwise CPU."""


@dataclass(frozen=True)
class ClipRollout:
  """Ragged trajectories from one batched clip rollout."""

  joint_names: tuple[str, ...]
  control_dt_s: float
  root_quat_wxyz: np.ndarray
  joint_pos: np.ndarray
  joint_vel: np.ndarray
  tick: np.ndarray
  time_s: np.ndarray
  env_index: np.ndarray
  trajectory_offsets: np.ndarray
  trajectory_env_index: np.ndarray
  trajectory_outcome: np.ndarray
  trajectory_steps: np.ndarray

  def arrays(self) -> dict[str, np.ndarray]:
    """Return the NPZ payload, one flat row per recorded control tick."""
    return {
      "root_quat_wxyz": self.root_quat_wxyz,
      "joint_pos": self.joint_pos,
      "joint_vel": self.joint_vel,
      "tick": self.tick,
      "time_s": self.time_s,
      "env_index": self.env_index,
      "trajectory_offsets": self.trajectory_offsets,
      "trajectory_env_index": self.trajectory_env_index,
      "trajectory_outcome": self.trajectory_outcome,
      "trajectory_steps": self.trajectory_steps,
    }

  def outcome_counts(self) -> dict[str, int]:
    """Count first-episode outcomes by label."""
    counts = {OUTCOME_COMPLETE: 0, OUTCOME_FAILURE: 0, OUTCOME_BUDGET: 0}
    for value in self.trajectory_outcome:
      counts[str(value)] += 1
    return counts


def parse_config(argv: Sequence[str] | None = None) -> CollectionConfig:
  """Parse the public CLI into a collection configuration."""
  return tyro.cli(
    CollectionConfig,
    args=None if argv is None else list(argv),
    description="Collect raw achieved X2 root orientation, q and dq.",
  )


def _resolve_checkpoint(path: Path) -> Path:
  return path.expanduser().resolve()


def _resolve_clip(path: Path) -> Path:
  expanded = path.expanduser()
  if expanded.is_absolute():
    return expanded.resolve()
  return (repository_root() / expanded).resolve()


def _resolve_output_dir(path: Path) -> Path:
  return path.expanduser().resolve()


def validate_config(config: CollectionConfig) -> None:
  """Refuse a run that cannot produce bounded, non-overwriting data."""
  if config.num_envs < 1:
    raise ValueError(f"num_envs must be positive, got {config.num_envs}")
  if config.seed < 0:
    raise ValueError(f"seed must be non-negative, got {config.seed}")
  if config.max_steps is not None and config.max_steps < 1:
    raise ValueError(f"max_steps must be positive when set, got {config.max_steps}")
  if not config.motion_file:
    raise ValueError("at least one --motion-file is required")
  checkpoint = _resolve_checkpoint(config.checkpoint_file)
  if not checkpoint.is_file():
    raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
  if load_saved_lookahead_s(checkpoint) is None:
    raise RuntimeError(
      "checkpoint has no saved params/env.yaml; cannot preserve its reference lookahead"
    )
  for clip in config.motion_file:
    resolved = _resolve_clip(clip)
    if not resolved.is_file():
      raise FileNotFoundError(f"reference clip not found: {resolved}")


def prepare_output_dir(path: Path) -> Path:
  """Create a new run directory, refusing to reuse an existing one."""
  output_dir = _resolve_output_dir(path)
  if output_dir.exists():
    raise FileExistsError(f"output directory already exists: {output_dir}")
  output_dir.mkdir(parents=True)
  (output_dir / "trajectories").mkdir()
  return output_dir


def read_clip_extent(clip_file: Path) -> tuple[int, float]:
  """Return the frame count and rate of one tracking-format reference clip."""
  with np.load(clip_file, allow_pickle=True) as data:
    if "body_pos_w" not in data or "fps" not in data:
      raise ValueError(f"reference clip {clip_file} needs body_pos_w and fps")
    frames = int(data["body_pos_w"].shape[0])
    fps_values = np.asarray(data["fps"]).reshape(-1)
  if frames < 1:
    raise ValueError(f"reference clip {clip_file} has no frames")
  if fps_values.size != 1:
    raise ValueError(f"reference clip {clip_file} must store one fps value")
  fps = float(fps_values[0])
  if not np.isfinite(fps) or fps <= 0.0:
    raise ValueError(f"reference clip {clip_file} has invalid fps {fps!r}")
  return frames, fps


def configure_collection_env(
  env_cfg: ManagerBasedRlEnvCfg,
  motion_file: Path,
  checkpoint_file: Path,
  *,
  num_envs: int,
  seed: int,
  episode_length_s: float,
) -> dict[str, Any]:
  """Apply the evaluator's manual-reset playback settings to one clip.

  The saved ``lookahead_s`` is mandatory.  A checkpoint without
  ``params/env.yaml`` would otherwise silently run a different preview than the
  one the teacher was trained with.
  """
  motion_cmd = env_cfg.commands.get("motion")
  if not isinstance(motion_cmd, MultiMotionPpoCommandCfg):
    raise ValueError("collection needs the multi-motion tracking task")
  saved_lookahead_s = load_saved_lookahead_s(checkpoint_file)
  if saved_lookahead_s is None:
    raise RuntimeError(
      "checkpoint has no saved params/env.yaml; cannot preserve its reference lookahead"
    )
  motion_cmd.motion_manifest = ""
  motion_cmd.motion_file = str(motion_file)
  motion_cmd.sampling_mode = "start"
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
    "seed": seed,
    "episode_length_s": episode_length_s,
    "lookahead_s": float(saved_lookahead_s),
    "sampling_mode": "start",
    "auto_reset": False,
    "actor_corruption": True,
    "push_robot_event": False,
  }


def _to_numpy(value: torch.Tensor) -> np.ndarray:
  return value.detach().cpu().numpy().astype(np.float32, copy=True)


def _read_robot_state(
  robot: Any, num_envs: int, joint_dim: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Read achieved simulator state, never the reference command."""
  data = robot.data
  quat = data.root_link_quat_w
  joint_pos = data.joint_pos
  joint_vel = data.joint_vel
  if tuple(quat.shape) != (num_envs, 4):
    raise RuntimeError(
      f"root_link_quat_w has shape {tuple(quat.shape)}, expected {(num_envs, 4)}"
    )
  if tuple(joint_pos.shape) != (num_envs, joint_dim):
    raise RuntimeError(
      f"joint_pos has shape {tuple(joint_pos.shape)}, expected {(num_envs, joint_dim)}"
    )
  if tuple(joint_vel.shape) != (num_envs, joint_dim):
    raise RuntimeError(
      f"joint_vel has shape {tuple(joint_vel.shape)}, expected {(num_envs, joint_dim)}"
    )
  return _to_numpy(quat), _to_numpy(joint_pos), _to_numpy(joint_vel)


def physical_failure_mask(termination: TerminationManager) -> torch.Tensor:
  """Return rows whose registered physical termination terms fired.

  The reference completion is not a physical failure, and neither is a
  registered time limit.  Reading the registered terms keeps a simultaneous
  failure on the completion step from being lost.
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
  """Attribute this step's finished rows to completion, failure, or budget.

  A physical failure wins over a simultaneous completion; a completion wins over
  a simultaneous time limit.  A time-limit exit is labelled ``budget`` because
  the collector sets it from the explicit clip/step cap.
  """
  outcomes: list[str | None] = [None] * int(dones.shape[0])
  for index in dones.nonzero(as_tuple=False).squeeze(-1).tolist():
    if bool(failed[index]):
      outcomes[index] = OUTCOME_FAILURE
    elif bool(complete[index]):
      outcomes[index] = OUTCOME_COMPLETE
    elif bool(timed_out[index]):
      outcomes[index] = OUTCOME_BUDGET
    else:
      raise RuntimeError(
        f"environment {index} finished without a completion, physical-failure, "
        "or time-limit termination term"
      )
  return outcomes


def collect_clip(
  env: CollectionEnv,
  command: Any,
  policy: Callable[[Any], torch.Tensor],
  *,
  step_budget: int,
) -> ClipRollout:
  """Roll out one clip on all environment rows and record achieved state.

  Tick 0 is the state after the environment reset and before the first action.
  Tick ``k`` is the state after ``k`` control steps.  A finished row's terminal
  state is recorded before that row is reset; the reset state is never appended.
  A row ends at its first outcome, so a fresh episode on the same environment
  row is not a continuation of the recorded trajectory.
  """
  if step_budget < 1:
    raise ValueError(f"step_budget must be positive, got {step_budget}")

  unwrapped = env.unwrapped
  termination = unwrapped.termination_manager
  num_envs = int(env.num_envs)
  step_dt = float(unwrapped.step_dt)
  entity_name = command.cfg.entity_name
  robot = unwrapped.scene[entity_name]
  joint_names = tuple(str(name) for name in robot.joint_names)
  joint_dim = len(joint_names)
  if joint_dim < 1:
    raise RuntimeError("robot exposes no joints")

  quat_rows: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
  pos_rows: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
  vel_rows: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
  tick_rows: list[list[int]] = [[] for _ in range(num_envs)]
  initial_quat, initial_pos, initial_vel = _read_robot_state(robot, num_envs, joint_dim)
  for index in range(num_envs):
    quat_rows[index].append(initial_quat[index])
    pos_rows[index].append(initial_pos[index])
    vel_rows[index].append(initial_vel[index])
    tick_rows[index].append(0)

  active = torch.ones(num_envs, dtype=torch.bool, device=unwrapped.device)
  outcomes: list[str | None] = [None] * num_envs
  obs = env.get_observations()
  step = 0
  while bool(active.any()) and step < step_budget:
    with torch.no_grad():
      actions = policy(obs)
    obs, _, dones, _ = env.step(actions)
    step += 1

    quat, joint_pos, joint_vel = _read_robot_state(robot, num_envs, joint_dim)
    for index in active.nonzero(as_tuple=False).squeeze(-1).tolist():
      quat_rows[index].append(quat[index])
      pos_rows[index].append(joint_pos[index])
      vel_rows[index].append(joint_vel[index])
      tick_rows[index].append(step)

    done_now = dones.bool()
    classified = classify_finished_rows(
      done_now & active,
      termination.get_term(CLIP_COMPLETE_TERM),
      physical_failure_mask(termination),
      termination.time_outs,
    )
    for index, outcome in enumerate(classified):
      if outcome is None:
        continue
      outcomes[index] = outcome
      active[index] = False

    if step >= step_budget:
      for index in active.nonzero(as_tuple=False).squeeze(-1).tolist():
        outcomes[index] = OUTCOME_BUDGET
        active[index] = False

    if bool(done_now.any()):
      # Every row the environment marks done must be reset before the next
      # step while auto-reset is off.  This includes inactive rows, whose fresh
      # episodes are not recorded.  The returned buffer becomes the policy input.
      reset_ids = done_now.nonzero(as_tuple=False).squeeze(-1)
      obs_dict, _ = unwrapped.reset(env_ids=reset_ids)
      obs = TensorDict(obs_dict, batch_size=[num_envs])

  if any(outcome is None for outcome in outcomes):
    raise RuntimeError("collection ended without an outcome for every environment")

  quats = [np.stack(quat_rows[index], axis=0) for index in range(num_envs)]
  positions = [np.stack(pos_rows[index], axis=0) for index in range(num_envs)]
  velocities = [np.stack(vel_rows[index], axis=0) for index in range(num_envs)]
  ticks = [np.asarray(tick_rows[index], dtype=np.int32) for index in range(num_envs)]
  offsets = np.zeros(num_envs + 1, dtype=np.int64)
  for index in range(num_envs):
    offsets[index + 1] = offsets[index] + ticks[index].shape[0]
  flat_ticks = np.concatenate(ticks, axis=0)
  flat_env = np.concatenate(
    [
      np.full(ticks[index].shape[0], index, dtype=np.int32) for index in range(num_envs)
    ],
    axis=0,
  )
  outcomes_array = np.asarray(
    [outcome for outcome in outcomes if outcome is not None], dtype="U8"
  )
  steps_array = np.asarray(
    [int(ticks[index][-1]) for index in range(num_envs)], dtype=np.int32
  )
  return ClipRollout(
    joint_names=joint_names,
    control_dt_s=step_dt,
    root_quat_wxyz=np.concatenate(quats, axis=0).astype(np.float32, copy=False),
    joint_pos=np.concatenate(positions, axis=0).astype(np.float32, copy=False),
    joint_vel=np.concatenate(velocities, axis=0).astype(np.float32, copy=False),
    tick=flat_ticks,
    time_s=flat_ticks.astype(np.float32) * np.float32(step_dt),
    env_index=flat_env,
    trajectory_offsets=offsets,
    trajectory_env_index=np.arange(num_envs, dtype=np.int32),
    trajectory_outcome=outcomes_array,
    trajectory_steps=steps_array,
  )


def write_clip_npz(path: Path, rollout: ClipRollout) -> None:
  """Write one clip's finite achieved-state trajectories."""
  for name in ("root_quat_wxyz", "joint_pos", "joint_vel"):
    if not np.isfinite(getattr(rollout, name)).all():
      raise ValueError(f"{name} contains non-finite achieved state")
  path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(path, **cast(Any, rollout.arrays()))


def _array_documentation() -> dict[str, str]:
  return {
    "root_quat_wxyz": (
      "float32 [rows, 4]; achieved root-link orientation in the world/simulator "
      "frame, wxyz order"
    ),
    "joint_pos": (
      "float32 [rows, joints]; achieved joint positions in robot joint order, rad"
    ),
    "joint_vel": (
      "float32 [rows, joints]; achieved joint velocities in robot joint order, rad/s"
    ),
    "tick": (
      "int32 [rows]; control tick. 0 is the post-reset state before the first "
      "action; k is after k control steps"
    ),
    "time_s": "float32 [rows]; tick * control_dt_s",
    "env_index": "int32 [rows]; environment row that produced the state",
    "trajectory_offsets": (
      "int64 [trajectories + 1]; row slice for one environment row's first episode"
    ),
    "trajectory_env_index": "int32 [trajectories]; environment row of that episode",
    "trajectory_outcome": ("unicode [trajectories]; complete, failure, or budget"),
    "trajectory_steps": "int32 [trajectories]; control steps after tick 0",
  }


def _build_manifest(
  config: CollectionConfig,
  checkpoint_file: Path,
  clips: list[dict[str, Any]],
  *,
  created_utc: str,
  complete: bool,
  error: str | None = None,
) -> dict[str, Any]:
  manifest: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "kind": "x2_joint_pose_collection",
    "complete": complete,
    "created_utc": created_utc,
    "task": config.task,
    "checkpoint": {
      "file": str(checkpoint_file),
    },
    "collection": {
      "num_envs": config.num_envs,
      "seed": config.seed,
      "max_steps": config.max_steps,
      "device": config.device,
      "lookahead_s": None,
      "actor_corruption": True,
      "push_robot_event": False,
      "sampling_mode": "start",
      "auto_reset": False,
    },
    "frame": {
      "root_quaternion": "world/simulator frame",
      "quaternion_order": "wxyz",
      "heading_alignment": "none; raw world-frame orientation",
    },
    "joint_names": None,
    "control_dt_s": None,
    "units": {
      "root_quaternion": "unit quaternion, dimensionless",
      "joint_pos": "rad",
      "joint_vel": "rad/s",
      "time_s": "s",
    },
    "sample_timing": (
      "tick 0 is the state after the environment reset and before the first "
      "action; tick k is the achieved state after k control steps. The last "
      "recorded tick of a trajectory is its terminal or budget-stop state. "
      "Post-reset state is never appended to a trajectory."
    ),
    "arrays": _array_documentation(),
    "clips": clips,
  }
  if clips:
    manifest["joint_names"] = list(clips[0]["joint_names"])
    manifest["control_dt_s"] = clips[0]["control_dt_s"]
  if error is not None:
    manifest["error"] = error
  return manifest


def _write_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
  (output_dir / "manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
  )


def _clip_entry(
  clip_index: int,
  clip_file: Path,
  *,
  frames: int,
  fps: float,
  step_budget: int,
  budget_source: str,
  output_relative: Path,
  rollout: ClipRollout,
) -> dict[str, Any]:
  return {
    "index": clip_index,
    "file": str(clip_file),
    "file_name": clip_file.name,
    "frames": frames,
    "fps": fps,
    "step_budget": step_budget,
    "budget_source": budget_source,
    "output": str(output_relative),
    "control_dt_s": rollout.control_dt_s,
    "joint_names": list(rollout.joint_names),
    "rows": int(rollout.tick.shape[0]),
    "trajectories": int(rollout.trajectory_env_index.shape[0]),
    "outcomes": rollout.outcome_counts(),
  }


def collect(config: CollectionConfig) -> dict[str, Any]:
  """Run the bounded per-clip collection and return its manifest.

  Each clip is one batched environment.  Its NPZ is written before the next clip
  is built, so memory is bounded by one clip's explicit rollout.  The policy is
  loaded once, kept in inference mode, and evaluated under ``torch.no_grad``.
  """
  validate_config(config)
  checkpoint_file = _resolve_checkpoint(config.checkpoint_file)
  clip_files = tuple(_resolve_clip(path) for path in config.motion_file)
  device = config.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  configure_torch_backends()
  agent_cfg = load_rl_cfg(config.task)
  output_dir = prepare_output_dir(config.output_dir)
  created_utc = datetime.now(timezone.utc).isoformat()
  manifest = _build_manifest(
    config, checkpoint_file, [], created_utc=created_utc, complete=False
  )
  manifest["collection"]["device"] = device
  _write_manifest(output_dir, manifest)
  policy: Callable[[Any], torch.Tensor] | None = None

  print(f"[INFO] device: {device}")
  print(f"[INFO] checkpoint: {checkpoint_file}")
  try:
    for clip_index, clip_file in enumerate(clip_files):
      frames, fps = read_clip_extent(clip_file)
      if config.max_steps is None or config.max_steps >= frames:
        step_budget = frames
        budget_source = "clip_length"
      else:
        step_budget = config.max_steps
        budget_source = "max_steps"
      env_cfg = load_env_cfg(config.task, play=False)
      settings = configure_collection_env(
        env_cfg,
        clip_file,
        checkpoint_file,
        num_envs=config.num_envs,
        seed=config.seed,
        episode_length_s=frames / fps,
      )
      manifest["collection"]["lookahead_s"] = settings["lookahead_s"]
      print(
        f"[INFO] clip {clip_index + 1}/{len(clip_files)}: {clip_file.name} "
        f"frames={frames} fps={fps:g} step_budget={step_budget} "
        f"lookahead_s={settings['lookahead_s']:g} "
        f"actor_corruption={settings['actor_corruption']} "
        f"push_robot={settings['push_robot_event']}"
      )
      raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
      env: Any = raw_env
      try:
        env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        if policy is None:
          runner_cls = load_runner_cls(config.task) or MjlabOnPolicyRunner
          runner = runner_cls(env, asdict(agent_cfg), device=device)
          runner.load(str(checkpoint_file), map_location=device)
          policy = runner.get_inference_policy(device=device)
          del runner
        command = env.unwrapped.command_manager.get_term("motion")
        rollout = collect_clip(env, command, policy, step_budget=step_budget)
        output_relative = (
          Path("trajectories") / f"{clip_index:03d}_{clip_file.stem}.npz"
        )
        write_clip_npz(output_dir / output_relative, rollout)
        manifest["clips"].append(
          _clip_entry(
            clip_index,
            clip_file,
            frames=frames,
            fps=fps,
            step_budget=step_budget,
            budget_source=budget_source,
            output_relative=output_relative,
            rollout=rollout,
          )
        )
        manifest["joint_names"] = list(rollout.joint_names)
        manifest["control_dt_s"] = rollout.control_dt_s
        _write_manifest(output_dir, manifest)
        print(
          f"[INFO] wrote {output_relative}: rows={rollout.tick.shape[0]} "
          f"outcomes={rollout.outcome_counts()}"
        )
      finally:
        env.close()
  except Exception as exc:
    manifest["error"] = f"{type(exc).__name__}: {exc}"
    _write_manifest(output_dir, manifest)
    raise

  manifest["complete"] = True
  _write_manifest(output_dir, manifest)
  return manifest


def main(config: CollectionConfig | None = None) -> None:
  """Run the collector from the parsed command line."""
  if config is None:
    config = parse_config()
  manifest = collect(config)
  print(
    f"[INFO] collection complete: {len(manifest['clips'])} clip(s), "
    f"manifest={Path(config.output_dir).expanduser().resolve() / 'manifest.json'}"
  )


if __name__ == "__main__":
  main()
