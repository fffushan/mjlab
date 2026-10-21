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
from typing import Any, Literal, Protocol, cast

import numpy as np
import torch
import tyro
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.events import push_by_setting_velocity, resolve_env_ids
from mjlab.managers.event_manager import EventTermCfg
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
SCHEMA_VERSION = 2
OUTCOME_COMPLETE = "complete"
OUTCOME_FAILURE = "failure"
OUTCOME_BUDGET = "budget"
SCENARIO_BASELINE = "baseline"
SCENARIO_DISTURBED = "disturbed"
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


class PushRecorder:
  """Count and tick the pushes applied to each first-episode trajectory.

  The interval event fires inside ``env.step`` after the episode step counter is
  incremented. Rows that already finished their first episode are excluded, so
  a later episode on the same environment row cannot add push evidence.
  """

  def __init__(self, num_envs: int) -> None:
    if num_envs < 1:
      raise ValueError(f"num_envs must be positive, got {num_envs}")
    self.num_envs = num_envs
    self.counts = [0] * num_envs
    self.ticks: list[list[int]] = [[] for _ in range(num_envs)]
    self._active_mask: torch.Tensor | None = None

  def bind_active_mask(self, active: torch.Tensor) -> None:
    """Share the collection loop's live first-episode row mask."""
    if tuple(active.shape) != (self.num_envs,):
      raise ValueError(
        f"active mask has shape {tuple(active.shape)}, expected {(self.num_envs,)}"
      )
    self._active_mask = active

  def record(self, env: Any, env_ids: torch.Tensor | None) -> None:
    """Record one interval-event firing for the active requested rows."""
    ids = resolve_env_ids(env, env_ids).detach().cpu().tolist()
    ticks = env.episode_length_buf
    for env_index in ids:
      if self._active_mask is not None and not bool(self._active_mask[env_index]):
        continue
      self.counts[env_index] += 1
      self.ticks[env_index].append(int(ticks[env_index]))


class RecordedVelocityPush:
  """Apply the task's root-velocity push and record the event.

  ``push_by_setting_velocity`` samples and adds a world-frame root velocity
  delta. This is not a force or torque. ``__deepcopy__`` keeps the recorder
  shared: the event manager deep-copies its configuration, and the collector
  reads the same recorder after the rollout.
  """

  def __init__(
    self, recorder: PushRecorder, velocity_range: dict[str, tuple[float, float]]
  ) -> None:
    self.recorder = recorder
    self.velocity_range = velocity_range

  def __call__(self, env: Any, env_ids: torch.Tensor | None) -> None:
    push_by_setting_velocity(env, env_ids, velocity_range=self.velocity_range)
    self.recorder.record(env, env_ids)

  def __deepcopy__(self, memo: dict[int, Any]) -> "RecordedVelocityPush":
    del memo
    return self


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
  """First environment seed. Each seed run reuses it for every clip."""

  num_seeds: int = 1
  """Finite number of seed runs, from ``seed`` through ``seed + num_seeds - 1``."""

  sampling_mode: Literal["start", "uniform"] = "start"
  """Reference phase at reset: ``start`` or uniform anywhere in the clip."""

  push_interval_s: tuple[float, float] | None = None
  """Opt-in disturbed scenario: uniformly sampled interval range for root pushes.

  ``None`` keeps the baseline contract and removes ``push_robot`` entirely.
  """

  push_linear_velocity_range: tuple[float, float] = (-0.5, 0.5)
  """World-frame root linear-velocity delta range per push, m/s."""

  push_angular_velocity_range: tuple[float, float] = (-0.52, 0.52)
  """World-frame root angular-velocity delta range per push, rad/s."""

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
  trajectory_initial_frame: np.ndarray
  trajectory_seed: np.ndarray
  trajectory_run_index: np.ndarray
  trajectory_scenario: np.ndarray
  trajectory_push_count: np.ndarray
  push_offsets: np.ndarray
  push_tick: np.ndarray
  push_env_index: np.ndarray
  scenario: str = SCENARIO_BASELINE

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
      "trajectory_initial_frame": self.trajectory_initial_frame,
      "trajectory_seed": self.trajectory_seed,
      "trajectory_run_index": self.trajectory_run_index,
      "trajectory_scenario": self.trajectory_scenario,
      "trajectory_push_count": self.trajectory_push_count,
      "push_offsets": self.push_offsets,
      "push_tick": self.push_tick,
      "push_env_index": self.push_env_index,
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


def collection_scenario(config: CollectionConfig) -> str:
  """Configured treatment label. It does not claim that a push happened."""
  if config.push_interval_s is None:
    return SCENARIO_BASELINE
  return SCENARIO_DISTURBED


def resolved_push_velocity_range(
  config: CollectionConfig,
) -> dict[str, tuple[float, float]]:
  """Map the CLI ranges onto the task's six root-velocity axes."""
  linear = (
    float(config.push_linear_velocity_range[0]),
    float(config.push_linear_velocity_range[1]),
  )
  angular = (
    float(config.push_angular_velocity_range[0]),
    float(config.push_angular_velocity_range[1]),
  )
  return {
    "x": linear,
    "y": linear,
    "z": linear,
    "roll": angular,
    "pitch": angular,
    "yaw": angular,
  }


def _disturbance_document(config: CollectionConfig) -> dict[str, Any] | None:
  if config.push_interval_s is None:
    return None
  return {
    "kind": "simulated_root_velocity_delta",
    "interval_s": [float(value) for value in config.push_interval_s],
    "velocity_range": {
      axis: [float(value) for value in values]
      for axis, values in resolved_push_velocity_range(config).items()
    },
    "units": {
      "interval_s": "s",
      "x_y_z": "m/s root linear-velocity delta",
      "roll_pitch_yaw": "rad/s root angular-velocity delta",
    },
    "note": (
      "Each push adds a uniformly sampled world-frame root-velocity delta. "
      "It is not a force or a torque. Counts and tick indices, not magnitudes, "
      "are recorded per trajectory."
    ),
  }


def _validate_range(
  name: str, value: tuple[float, float], *, positive: bool = False
) -> None:
  try:
    lower = float(value[0])
    upper = float(value[1])
  except (IndexError, TypeError, ValueError) as exc:
    raise ValueError(f"{name} must contain two finite numbers, got {value!r}") from exc
  if not np.isfinite(lower) or not np.isfinite(upper):
    raise ValueError(f"{name} must be finite, got {value!r}")
  if lower > upper:
    raise ValueError(f"{name} lower bound exceeds upper bound: {value!r}")
  if positive and lower <= 0.0:
    raise ValueError(f"{name} must be positive, got {value!r}")


def validate_config(config: CollectionConfig) -> None:
  """Refuse a run that cannot produce bounded, non-overwriting data."""
  if config.num_envs < 1:
    raise ValueError(f"num_envs must be positive, got {config.num_envs}")
  if config.seed < 0:
    raise ValueError(f"seed must be non-negative, got {config.seed}")
  if config.num_seeds < 1:
    raise ValueError(f"num_seeds must be positive, got {config.num_seeds}")
  if config.sampling_mode not in ("start", "uniform"):
    raise ValueError(
      f"sampling_mode must be 'start' or 'uniform', got {config.sampling_mode!r}"
    )
  if config.max_steps is not None and config.max_steps < 1:
    raise ValueError(f"max_steps must be positive when set, got {config.max_steps}")
  if config.push_interval_s is not None:
    _validate_range("push_interval_s", config.push_interval_s, positive=True)
  _validate_range("push_linear_velocity_range", config.push_linear_velocity_range)
  _validate_range("push_angular_velocity_range", config.push_angular_velocity_range)
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
  sampling_mode: Literal["start", "uniform"] = "start",
  push_interval_s: tuple[float, float] | None = None,
  push_velocity_range: dict[str, tuple[float, float]] | None = None,
  push_recorder: PushRecorder | None = None,
) -> dict[str, Any]:
  """Apply the evaluator's manual-reset playback settings to one clip.

  The saved ``lookahead_s`` is mandatory.  A checkpoint without
  ``params/env.yaml`` would otherwise silently run a different preview than the
  one the teacher was trained with.

  ``sampling_mode`` selects the existing multi-motion command's phase policy.
  Inherited reset pose, velocity, and joint perturbations stay on. The baseline
  removes ``push_robot``; the disturbed scenario installs the same interval
  root-velocity push term behind a recorder.
  """
  motion_cmd = env_cfg.commands.get("motion")
  if not isinstance(motion_cmd, MultiMotionPpoCommandCfg):
    raise ValueError("collection needs the multi-motion tracking task")
  if sampling_mode not in ("start", "uniform"):
    raise ValueError(
      f"sampling_mode must be 'start' or 'uniform', got {sampling_mode!r}"
    )
  saved_lookahead_s = load_saved_lookahead_s(checkpoint_file)
  if saved_lookahead_s is None:
    raise RuntimeError(
      "checkpoint has no saved params/env.yaml; cannot preserve its reference lookahead"
    )
  motion_cmd.motion_manifest = ""
  motion_cmd.motion_file = str(motion_file)
  motion_cmd.sampling_mode = sampling_mode
  motion_cmd.standing_start_prob = 0.0
  motion_cmd.lookahead_s = saved_lookahead_s
  env_cfg.observations["actor"].enable_corruption = True
  if push_interval_s is None:
    env_cfg.events.pop("push_robot", None)
  else:
    if push_velocity_range is None or push_recorder is None:
      raise ValueError(
        "a disturbed collection needs a push velocity range and recorder"
      )
    env_cfg.events["push_robot"] = EventTermCfg(
      func=RecordedVelocityPush(push_recorder, push_velocity_range),
      mode="interval",
      interval_range_s=push_interval_s,
    )
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
    "sampling_mode": sampling_mode,
    "standing_start_prob": 0.0,
    "auto_reset": False,
    "actor_corruption": True,
    "push_robot_event": push_interval_s is not None,
    "push_interval_s": (
      None if push_interval_s is None else [float(value) for value in push_interval_s]
    ),
    "push_velocity_range": (
      None
      if push_velocity_range is None
      else {
        axis: [float(value) for value in values]
        for axis, values in push_velocity_range.items()
      }
    ),
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


def _push_evidence(
  recorder: PushRecorder | None, num_envs: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Flatten first-episode push counts and ticks by trajectory."""
  if recorder is None:
    counts = np.zeros(num_envs, dtype=np.int32)
    ticks: list[int] = []
    env_ids: list[int] = []
  else:
    if recorder.num_envs != num_envs:
      raise ValueError(
        f"push recorder has {recorder.num_envs} rows, expected {num_envs}"
      )
    counts = np.asarray(recorder.counts, dtype=np.int32)
    ticks = []
    env_ids = []
    for index in range(num_envs):
      if recorder.counts[index] != len(recorder.ticks[index]):
        raise RuntimeError(
          f"push count and tick history disagree for environment {index}"
        )
      ticks.extend(recorder.ticks[index])
      env_ids.extend([index] * recorder.counts[index])
  offsets = np.zeros(num_envs + 1, dtype=np.int64)
  for index in range(num_envs):
    offsets[index + 1] = offsets[index] + int(counts[index])
  return (
    counts,
    offsets,
    np.asarray(ticks, dtype=np.int32),
    np.asarray(env_ids, dtype=np.int32),
  )


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
  seed: int = 0,
  run_index: int = 0,
  scenario: str = SCENARIO_BASELINE,
  push_recorder: PushRecorder | None = None,
) -> ClipRollout:
  """Roll out one clip on all environment rows and record achieved state.

  Tick 0 is the state after the environment reset and before the first action.
  Tick ``k`` is the state after ``k`` control steps.  A finished row's terminal
  state is recorded before that row is reset; the reset state is never appended.
  A row ends at its first outcome, so a fresh episode on the same environment
  row is not a continuation of the recorded trajectory.

  ``command.time_steps`` at entry is the actual reference frame the reset wrote
  for each row.  It is recorded separately from the elapsed trajectory tick.
  """
  if step_budget < 1:
    raise ValueError(f"step_budget must be positive, got {step_budget}")
  if scenario not in (SCENARIO_BASELINE, SCENARIO_DISTURBED):
    raise ValueError(f"unknown scenario label: {scenario!r}")

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
  initial_frames = command.time_steps.detach().cpu().numpy()
  if initial_frames.shape != (num_envs,):
    raise RuntimeError(
      f"command.time_steps has shape {initial_frames.shape}, expected {(num_envs,)}"
    )
  initial_frames = initial_frames.astype(np.int32, copy=True)

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
  if push_recorder is not None:
    push_recorder.bind_active_mask(active)
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
  push_counts, push_offsets, push_ticks, push_env_index = _push_evidence(
    push_recorder, num_envs
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
    trajectory_initial_frame=initial_frames,
    trajectory_seed=np.full(num_envs, int(seed), dtype=np.int64),
    trajectory_run_index=np.full(num_envs, int(run_index), dtype=np.int32),
    trajectory_scenario=np.full(num_envs, scenario, dtype="U9"),
    trajectory_push_count=push_counts,
    push_offsets=push_offsets,
    push_tick=push_ticks,
    push_env_index=push_env_index,
    scenario=scenario,
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
    "trajectory_initial_frame": (
      "int32 [trajectories]; actual reference frame written by the reset"
    ),
    "trajectory_seed": "int64 [trajectories]; seed of the seed run",
    "trajectory_run_index": ("int32 [trajectories]; index in the finite seed schedule"),
    "trajectory_scenario": (
      "unicode [trajectories]; configured treatment: baseline or disturbed"
    ),
    "trajectory_push_count": (
      "int32 [trajectories]; interval pushes observed during the first episode, "
      "including terminal/budget-tick events without a recorded response"
    ),
    "push_offsets": ("int64 [trajectories + 1]; push slice for one trajectory"),
    "push_tick": (
      "int32 [pushes]; event tick k, after physics integration. A response in "
      "saved root orientation/q/dq can first appear at tick k+1; a terminal "
      "or budget-tick event has no subsequent recorded sample. Empty when none fired"
    ),
    "push_env_index": "int32 [pushes]; environment row of each observed push",
  }


def _build_manifest(
  config: CollectionConfig,
  checkpoint_file: Path,
  clips: list[dict[str, Any]],
  *,
  num_clips: int,
  created_utc: str,
  complete: bool,
  error: str | None = None,
) -> dict[str, Any]:
  scenario = collection_scenario(config)
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
      "num_seeds": config.num_seeds,
      "seeds": [config.seed + index for index in range(config.num_seeds)],
      "num_clips": num_clips,
      "num_seed_runs": config.num_seeds,
      "num_clip_rollouts": num_clips * config.num_seeds,
      "total_trajectories": num_clips * config.num_seeds * config.num_envs,
      "max_steps": config.max_steps,
      "device": config.device,
      "lookahead_s": None,
      "actor_corruption": True,
      "push_robot_event": config.push_interval_s is not None,
      "sampling_mode": config.sampling_mode,
      "standing_start_prob": 0.0,
      "auto_reset": False,
      "scenario": scenario,
      "disturbance": _disturbance_document(config),
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
  run_index: int,
  seed: int,
  scenario: str,
  disturbance: dict[str, Any] | None,
  frames: int,
  fps: float,
  step_budget: int,
  budget_source: str,
  sampling_mode: str,
  output_relative: Path,
  rollout: ClipRollout,
) -> dict[str, Any]:
  return {
    "index": clip_index,
    "run_index": run_index,
    "seed": seed,
    "scenario": scenario,
    "disturbance": disturbance,
    "file": str(clip_file),
    "file_name": clip_file.name,
    "frames": frames,
    "fps": fps,
    "step_budget": step_budget,
    "budget_source": budget_source,
    "sampling_mode": sampling_mode,
    "output": str(output_relative),
    "control_dt_s": rollout.control_dt_s,
    "joint_names": list(rollout.joint_names),
    "rows": int(rollout.tick.shape[0]),
    "trajectories": int(rollout.trajectory_env_index.shape[0]),
    "pushes_observed": int(rollout.trajectory_push_count.sum()),
    "outcomes": rollout.outcome_counts(),
  }


def _trajectory_output_path(
  *,
  num_seeds: int,
  run_index: int,
  seed: int,
  clip_index: int,
  clip_file: Path,
  scenario: str,
) -> Path:
  """Keep the legacy single-seed baseline name; add identity otherwise."""
  if num_seeds == 1 and scenario == SCENARIO_BASELINE:
    name = f"{clip_index:03d}_{clip_file.stem}.npz"
  else:
    name = (
      f"run{run_index:03d}_seed{seed:03d}_{clip_index:03d}_"
      f"{clip_file.stem}_{scenario}.npz"
    )
  return Path("trajectories") / name


def collect(config: CollectionConfig) -> dict[str, Any]:
  """Run the bounded per-clip collection and return its manifest.

  Each clip is one batched environment.  Its NPZ is written before the next clip
  is built, so memory is bounded by one clip's explicit rollout.  The policy is
  loaded once, kept in inference mode, and evaluated under ``torch.no_grad``.
  """
  validate_config(config)
  checkpoint_file = _resolve_checkpoint(config.checkpoint_file)
  clip_files = tuple(_resolve_clip(path) for path in config.motion_file)
  seeds = tuple(config.seed + index for index in range(config.num_seeds))
  scenario = collection_scenario(config)
  disturbance = _disturbance_document(config)
  push_velocity_range = (
    resolved_push_velocity_range(config) if config.push_interval_s is not None else None
  )
  device = config.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  configure_torch_backends()
  agent_cfg = load_rl_cfg(config.task)
  output_dir = prepare_output_dir(config.output_dir)
  created_utc = datetime.now(timezone.utc).isoformat()
  manifest = _build_manifest(
    config,
    checkpoint_file,
    [],
    num_clips=len(clip_files),
    created_utc=created_utc,
    complete=False,
  )
  manifest["collection"]["device"] = device
  _write_manifest(output_dir, manifest)
  policy: Callable[[Any], torch.Tensor] | None = None

  print(f"[INFO] device: {device}")
  print(f"[INFO] checkpoint: {checkpoint_file}")
  print(
    f"[INFO] plan: clips={len(clip_files)} runs={len(seeds)} "
    f"envs={config.num_envs} "
    f"rollouts={len(clip_files) * len(seeds)} "
    f"trajectories={len(clip_files) * len(seeds) * config.num_envs} "
    f"scenario={scenario} sampling_mode={config.sampling_mode}"
  )
  try:
    for run_index, seed in enumerate(seeds):
      for clip_index, clip_file in enumerate(clip_files):
        frames, fps = read_clip_extent(clip_file)
        if config.max_steps is None or config.max_steps >= frames:
          step_budget = frames
          budget_source = "clip_length"
        else:
          step_budget = config.max_steps
          budget_source = "max_steps"
        push_recorder = (
          PushRecorder(config.num_envs) if config.push_interval_s is not None else None
        )
        env_cfg = load_env_cfg(config.task, play=False)
        settings = configure_collection_env(
          env_cfg,
          clip_file,
          checkpoint_file,
          num_envs=config.num_envs,
          seed=seed,
          episode_length_s=frames / fps,
          sampling_mode=config.sampling_mode,
          push_interval_s=config.push_interval_s,
          push_velocity_range=push_velocity_range,
          push_recorder=push_recorder,
        )
        manifest["collection"]["lookahead_s"] = settings["lookahead_s"]
        print(
          f"[INFO] run {run_index + 1}/{len(seeds)} seed={seed} "
          f"clip {clip_index + 1}/{len(clip_files)}: {clip_file.name} "
          f"frames={frames} fps={fps:g} step_budget={step_budget} "
          f"lookahead_s={settings['lookahead_s']:g} "
          f"sampling_mode={settings['sampling_mode']} "
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
            policy = cast(
              Callable[[Any], torch.Tensor],
              runner.get_inference_policy(device=device),
            )
            del runner
          assert policy is not None
          command = env.unwrapped.command_manager.get_term("motion")
          rollout = collect_clip(
            env,
            command,
            policy,
            step_budget=step_budget,
            seed=seed,
            run_index=run_index,
            scenario=scenario,
            push_recorder=push_recorder,
          )
          output_relative = _trajectory_output_path(
            num_seeds=config.num_seeds,
            run_index=run_index,
            seed=seed,
            clip_index=clip_index,
            clip_file=clip_file,
            scenario=scenario,
          )
          write_clip_npz(output_dir / output_relative, rollout)
          manifest["clips"].append(
            _clip_entry(
              clip_index,
              clip_file,
              run_index=run_index,
              seed=seed,
              scenario=scenario,
              disturbance=disturbance,
              frames=frames,
              fps=fps,
              step_budget=step_budget,
              budget_source=budget_source,
              sampling_mode=config.sampling_mode,
              output_relative=output_relative,
              rollout=rollout,
            )
          )
          manifest["joint_names"] = list(rollout.joint_names)
          manifest["control_dt_s"] = rollout.control_dt_s
          _write_manifest(output_dir, manifest)
          print(
            f"[INFO] wrote {output_relative}: rows={rollout.tick.shape[0]} "
            f"outcomes={rollout.outcome_counts()} "
            f"pushes={int(rollout.trajectory_push_count.sum())}"
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
  collection = manifest["collection"]
  print(
    f"[INFO] collection complete: {collection['num_clip_rollouts']} clip rollout(s), "
    f"{collection['total_trajectories']} trajectory(ies), "
    f"manifest={Path(config.output_dir).expanduser().resolve() / 'manifest.json'}"
  )


if __name__ == "__main__":
  main()
