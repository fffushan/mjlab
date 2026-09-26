"""Bounded evaluation for X2 tennis-end recovery.

This module provides:

* :class:`EvaluationSchedule` — deterministic held-out sample/seed schedules
  for source vs candidate policies.
* :func:`run_evaluation` — sequential one-env-per-episode evaluation that
  builds a single-env config with ``auto_reset=False``, loads a checkpoint,
  records a true t=0 reset snapshot, steps until termination or horizon,
  captures terminal state before any reset, and computes per-episode metrics.
* :func:`main` — tyro CLI entry point with ``mjlab.TYRO_FLAGS``.

Design (per parent review):

Each episode runs in its own single-env environment with ``auto_reset=False``.
This avoids vector partial-termination contamination: the terminal state is
read before any reset, and the env is closed in ``finally``. Recovery episodes
force the recovery group and assign a validated explicit eval row before the
first reset. Retention episodes use the original velocity task config with
its native command manager and curriculum.

All recorded arrays are deep-copied snapshots (no aliasing of live simulator
buffers). Torso tilt is computed from the ``torso_link`` body quaternion, not
the root. Real termination cause is preserved (``fell_over`` vs ``time_out``).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

__all__ = [
  "EvaluationSchedule",
  "EvaluationCliArgs",
  "build_evaluation_schedule",
  "format_group_summary",
  "DistributionFn",
  "run_evaluation",
  "EpisodeDescriptor",
  "prepare_episode_descriptors",
]

DistributionFn = Callable[[np.random.Generator, int], np.ndarray]


# --------------------------------------------------------------------------- #
# Schedule primitives.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EvaluationSchedule:
  """Deterministic held-out evaluation schedule for source vs candidate.

  Produces identical held-out sample/seed schedules for both source and
  candidate policies. The original command distribution remains *required*
  for the retention evaluation; recovery always uses zero command.

  Bounds: ``num_episodes``, ``episode_length`` (frames), ``duration_s`` and
  ``fps`` must all be positive and finite. The schedule is seeded so the
  trajectory/frame selection and per-episode seeds are reproducible.
  """

  num_episodes: int
  episode_length: int  # frames per episode
  duration_s: float  # seconds per episode
  fps: float  # sampling rate
  seed: int = 42
  split: str = "validation"  # train | validation | all
  recovery_fraction: float = 0.8

  def __post_init__(self) -> None:
    for name, val in [
      ("num_episodes", self.num_episodes),
      ("episode_length", self.episode_length),
    ]:
      if not isinstance(val, (int, np.integer)) or int(val) <= 0:
        raise ValueError(f"{name} must be a positive int, got {val!r}")
    if not np.isfinite(self.duration_s) or self.duration_s <= 0:
      raise ValueError(f"duration_s must be positive finite, got {self.duration_s!r}")
    if not np.isfinite(self.fps) or self.fps <= 0:
      raise ValueError(f"fps must be positive finite, got {self.fps!r}")
    if self.split not in ("train", "validation", "all"):
      raise ValueError(f"split must be train|validation|all, got {self.split!r}")
    if not (0.0 <= self.recovery_fraction <= 1.0):
      raise ValueError(
        f"recovery_fraction must be in [0,1], got {self.recovery_fraction!r}"
      )
    if self.duration_s * self.fps < self.episode_length:
      raise ValueError(
        f"episode_length={self.episode_length} exceeds "
        f"duration_s*fps={self.duration_s * self.fps:.1f} frames"
      )

  def episode_timestamps(self) -> np.ndarray:
    """Per-episode timestamps (seconds since reset origin), ``[episode_length]``."""
    dt = 1.0 / self.fps
    return np.arange(self.episode_length, dtype=np.float64) * dt

  def num_recovery_episodes(self) -> int:
    return int(round(self.recovery_fraction * self.num_episodes))

  def num_retention_episodes(self) -> int:
    return self.num_episodes - self.num_recovery_episodes()

  def sample_indices(self, pool_size: int) -> np.ndarray:
    """Seeded row indices into a pool for the held-out episodes.

    Identical for source and candidate (same seed). Bounds: ``pool_size``
    must be a positive int. If ``num_episodes > pool_size`` sampling uses
    replacement and that is documented here (rather than failing silently).
    """
    if not isinstance(pool_size, (int, np.integer)) or int(pool_size) <= 0:
      raise ValueError(f"pool_size must be a positive int, got {pool_size!r}")
    rng = np.random.default_rng(self.seed)
    n = self.num_episodes
    ps = int(pool_size)
    if ps >= n:
      return rng.choice(ps, size=n, replace=False)
    # Pool smaller than requested episodes: sample with replacement.
    return rng.integers(0, ps, size=n)

  def group_assignment(self) -> np.ndarray:
    """Fixed per-episode group labels (``recovery``/``retention``).

    Deterministic for the schedule lifetime. The first
    ``round(recovery_fraction * num_episodes)`` episodes are recovery, the
    rest retention (matching the 80/20 mix).
    """
    labels = np.where(
      np.arange(self.num_episodes) < self.num_recovery_episodes(),
      "recovery",
      "retention",
    )
    return labels.astype(object)

  def recovery_commands(self) -> np.ndarray:
    """Zero recovery commands for every recovery episode.

    Shape ``[num_recovery_episodes, 3]`` (lin_x, lin_y, ang_z), all zeros.
    Recovery commands remain exactly zero forever.
    """
    return np.zeros((self.num_recovery_episodes(), 3), dtype=np.float64)

  def retention_commands(self, distribution_fn: DistributionFn) -> np.ndarray:
    """Draw retention commands from the *original* command distribution.

    ``distribution_fn`` is a callable ``(rng, n) -> [n,3]`` returning commanded
    body-frame linear/angular velocity. The original command distribution
    remains required for the retention evaluation; this helper does not
    fabricate a substitute. The callback output is validated for shape and
    finiteness.
    """
    rng = np.random.default_rng(self.seed + 1)
    n_ret = self.num_retention_episodes()
    out = np.asarray(distribution_fn(rng, n_ret), dtype=np.float64)
    if out.ndim != 2 or out.shape[0] != n_ret or out.shape[1] != 3:
      raise ValueError(
        f"retention distribution_fn must return shape [{n_ret},3], got {out.shape}"
      )
    if not np.all(np.isfinite(out)):
      raise ValueError("retention distribution_fn returned non-finite values")
    return out

  def episode_seeds(self) -> np.ndarray:
    """Per-episode deterministic seeds (identical for source and candidate)."""
    rng = np.random.default_rng(self.seed + 2)
    return rng.integers(0, 2**31 - 1, size=self.num_episodes, dtype=np.int64)


@dataclass
class EvaluationCliArgs:
  """CLI argument namespace for the bounded evaluation entry point.

  Integration builds an :class:`EvaluationSchedule` from these arguments and
  wires them to a real simulator/runner. Defaults follow the plan (20 s
  episodes at 50 Hz, 10 held-out validation episodes).
  """

  num_episodes: int = 10
  episode_length: int = 1000  # 20 s @ 50 Hz
  duration_s: float = 20.0
  fps: float = 50.0
  seed: int = 42
  split: str = "validation"
  recovery_fraction: float = 0.8
  early_window_s: float = 3.0
  # Settling thresholds (passed through to the metrics module).
  settling_speed_xy: float = 0.1
  settling_yaw_rate: float = 0.2
  settling_torso_tilt_deg: float = 10.0
  settling_sustained_s: float = 0.5
  # Source/candidate run identifiers (resolved by integration to real paths).
  source_run: str = ""
  candidate_run: str = ""
  # Output directory for per-trajectory results; defaults to a managed artifact.
  output_dir: str = ""
  # Task base for the retention evaluation with the original command distribution.
  base_task: str = "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation"


def build_evaluation_schedule(args: EvaluationCliArgs) -> EvaluationSchedule:
  """Build a validated :class:`EvaluationSchedule` from CLI arguments."""
  return EvaluationSchedule(
    num_episodes=args.num_episodes,
    episode_length=args.episode_length,
    duration_s=args.duration_s,
    fps=args.fps,
    seed=args.seed,
    split=args.split,
    recovery_fraction=args.recovery_fraction,
  )


def format_group_summary(stats_list: list) -> str:
  """Format per-group :class:`RecoveryGroupStats` into a human-readable summary.

  Accepts the ``RecoveryGroupStats`` dataclass from the metrics module (imported
  lazily to avoid a hard import dependency at module load). Handles ``None``
  unavailable aggregates gracefully.
  """
  lines: list[str] = []
  for s in stats_list:
    med = (
      f"{s.median_time_to_settle_s:.3f}s"
      if s.median_time_to_settle_s is not None
      else "n/a (none settled)"
    )
    mean_tts = (
      f"{s.mean_time_to_settle_s:.3f}s"
      if s.mean_time_to_settle_s is not None
      else "n/a"
    )
    mean_peak = (
      f"{s.mean_peak_tilt_deg:.2f}deg" if s.mean_peak_tilt_deg is not None else "n/a"
    )
    mean_bwd = (
      f"{s.mean_signed_backward_displacement_m:.3f}m"
      if s.mean_signed_backward_displacement_m is not None
      else "n/a"
    )
    mean_travel = (
      f"{s.mean_total_planar_travel_m:.3f}m"
      if s.mean_total_planar_travel_m is not None
      else "n/a"
    )
    if s.mean_command_lin_error_mps is not None:
      cmd_str = (
        f" cmd_lin_err={s.mean_command_lin_error_mps:.3f}m/s"
        f" cmd_yaw_err={s.mean_command_yaw_error_radps:.3f}rad/s"
        f" (n={s.num_command_tracked})"
      )
    else:
      cmd_str = ""
    lines.append(
      f"[{s.group}] n={s.num_trajectories} settled={s.num_settled} "
      f"({s.settle_rate:.1%}) falls={s.num_falls} orient={s.num_orientation_terminations} "
      f"timeout={s.num_timeouts} not_settled={s.num_not_settled} "
      f"censored={s.num_censored} invalid={s.num_invalid} "
      f"coverage={s.coverage:.1%} | "
      f"median_tts={med} mean_tts={mean_tts} "
      f"mean_peak_tilt={mean_peak} "
      f"mean_bwd={mean_bwd} "
      f"mean_travel={mean_travel}{cmd_str}"
    )
  return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Episode descriptors.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EpisodeDescriptor:
  """Planned episode descriptor for deterministic evaluation.

  Each descriptor specifies the group, pool row index (for recovery), episode
  seed, and trajectory/frame provenance. Both source and candidate use the
  same descriptors so the comparison is apples-to-apples.
  """

  episode_index: int
  group: str  # "recovery" | "retention"
  pool_row_index: int | None  # None for retention
  trajectory_id: int | None  # None for retention
  frame_index: int | None  # None for retention
  episode_seed: int


def prepare_episode_descriptors(
  schedule: EvaluationSchedule,
  pool_size: int,
  pool_trajectory_ids: np.ndarray | None = None,
  pool_frame_indices: np.ndarray | None = None,
) -> list[EpisodeDescriptor]:
  """Prepare deterministic episode descriptors from the schedule.

  Returns one descriptor per episode. Recovery episodes get explicit pool row
  indices; retention episodes get ``None`` (they use the original reset).

  ``pool_trajectory_ids`` and ``pool_frame_indices`` are the pool's provenance
  arrays, used to record the planned trajectory/frame for each recovery episode.
  """
  row_indices = schedule.sample_indices(pool_size)
  group_labels = schedule.group_assignment()
  episode_seeds = schedule.episode_seeds()

  descriptors: list[EpisodeDescriptor] = []
  rec_idx = 0
  for i in range(schedule.num_episodes):
    group = str(group_labels[i])
    if group == "recovery":
      row = int(row_indices[rec_idx])
      traj_id = (
        int(pool_trajectory_ids[row]) if pool_trajectory_ids is not None else row
      )
      frame_id = (
        int(pool_frame_indices[row]) if pool_frame_indices is not None else None
      )
      descriptors.append(
        EpisodeDescriptor(
          episode_index=i,
          group="recovery",
          pool_row_index=row,
          trajectory_id=traj_id,
          frame_index=frame_id,
          episode_seed=int(episode_seeds[i]),
        )
      )
      rec_idx += 1
    else:
      descriptors.append(
        EpisodeDescriptor(
          episode_index=i,
          group="retention",
          pool_row_index=None,
          trajectory_id=None,
          frame_index=None,
          episode_seed=int(episode_seeds[i]),
        )
      )
  return descriptors


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #


def _sha256_file(path: str | Path) -> str:
  """Return the hex SHA-256 digest of a file."""
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def _yaw_from_quat_wxyz(q: np.ndarray) -> float:
  """Extract yaw (rad) from a single wxyz quaternion."""
  w, x, y, z = q
  return float(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _safe_float(v: Any) -> float | None:
  """Convert to float, returning None for NaN/inf."""
  if v is None:
    return None
  f = float(v)
  if not np.isfinite(f):
    return None
  return f


# --------------------------------------------------------------------------- #
# Single-episode evaluation.
# --------------------------------------------------------------------------- #


def recovery_reset_event_overrides(
  schedule: EvaluationSchedule,
  pool_directory: str,
  last_n_frames: int = 10,
) -> dict[str, Any]:
  """Deterministic recovery-reset event overrides for one evaluation episode.

  The endpoint window is passed explicitly: a policy trained on a wider window
  must not be silently scored on a narrower (or wider) one, and the window also
  decides the pool the descriptor row indices refer to.

  Raises:
    ValueError: If ``last_n_frames`` is not a positive int.
  """
  if isinstance(last_n_frames, bool) or not isinstance(last_n_frames, int):
    raise ValueError("last_n_frames must be an int")
  if last_n_frames < 1:
    raise ValueError("last_n_frames must be positive")
  return {
    "pool_directory": pool_directory,
    "split": schedule.split,
    "seed": schedule.seed,
    "recovery_fraction": 1.0,
    "force_mode": "recovery",
    "last_n_frames": last_n_frames,
  }


def _run_single_episode(
  descriptor: EpisodeDescriptor,
  schedule: EvaluationSchedule,
  checkpoint_path: str,
  pool_directory: str,
  device: str,
  recovery_task_id: str,
  base_task_id: str,
  last_n_frames: int = 10,
) -> dict[str, Any]:
  """Run one episode in a single-env config with auto_reset=False.

  Builds a 1-env config, loads the checkpoint, records a true t=0 reset
  snapshot, steps until termination or horizon, captures terminal state
  before any reset, and returns the trace + metrics.

  Recovery episodes force the recovery group and assign a validated explicit
  eval row before the first reset. Retention episodes use the original
  velocity task config with its native command manager.
  """
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import (
    load_env_cfg,
    load_rl_cfg,
    load_runner_cls,
  )
  from mjlab.tasks.velocity.mdp.tennis_recovery_metrics import (
    RecoveryTrace,
    RetentionTrace,
    SettlingThresholds,
    compute_recovery_metrics,
    torso_tilt_from_quat,
  )

  is_recovery = descriptor.group == "recovery"
  task_id = recovery_task_id if is_recovery else base_task_id

  # Build a single-env config with finite horizon and auto_reset=False.
  env_cfg = load_env_cfg(task_id, play=False)
  env_cfg.scene.num_envs = 1
  env_cfg.auto_reset = False
  env_cfg.seed = descriptor.episode_seed
  configured_dt = env_cfg.decimation * env_cfg.sim.mujoco.timestep
  if not np.isclose(1.0 / schedule.fps, configured_dt, rtol=1e-6, atol=1e-9):
    raise ValueError(
      f"Evaluation fps={schedule.fps} does not match task step_dt={configured_dt}"
    )
  # A shorter step budget is itself an explicit finite evaluation horizon.
  env_cfg.episode_length_s = min(
    schedule.duration_s, schedule.episode_length * configured_dt
  )

  # For recovery: set pool directory, split, seed, force recovery group.
  if is_recovery:
    event_params = env_cfg.events["tennis_recovery_reset"].params
    event_params.update(
      recovery_reset_event_overrides(schedule, pool_directory, last_n_frames)
    )

  env: ManagerBasedRlEnv | None = None
  try:
    # cfg.seed is applied before startup randomization by the environment.
    env = ManagerBasedRlEnv(env_cfg, device=device)
    step_dt = env.step_dt
    robot = env.scene["robot"]

    # For recovery: set eval_row_indices BEFORE the wrapper's first reset.
    if is_recovery:
      event_term_cfg = env.event_manager.get_term_cfg("tennis_recovery_reset")
      event_term = event_term_cfg.func
      row = descriptor.pool_row_index
      assert row is not None and row >= 0, (
        f"Recovery episode {descriptor.episode_index} has invalid row {row}"
      )
      # The pool is loaded lazily on first reset; we set the override now
      # and it will be used when _ensure_pool runs.
      event_term.eval_row_indices = torch.tensor([row], dtype=torch.long, device=device)

    # Wrap and load runner.
    wrapped = RslRlVecEnvWrapper(env)
    runner_cls = load_runner_cls(task_id)
    assert runner_cls is not None
    rl_cfg = load_rl_cfg(task_id)
    runner = runner_cls(wrapped, asdict(rl_cfg), None, device)
    runner.load(checkpoint_path, map_location=device)
    policy = runner.get_inference_policy(device=device)

    # Reset with the episode seed.
    obs, _ = wrapped.reset()

    # Record true t=0 reset snapshot (deep copy, no aliasing).
    snapshots: list[dict[str, Any]] = []

    def _snapshot(t: float) -> dict[str, Any]:
      """Capture a deep-copied state snapshot at time t."""
      root_pos = robot.data.root_link_pos_w[0].cpu().numpy().copy()
      root_quat = robot.data.root_link_quat_w[0].cpu().numpy().copy()
      root_lin = robot.data.root_link_lin_vel_w[0].cpu().numpy().copy()
      root_ang = robot.data.root_link_ang_vel_w[0].cpu().numpy().copy()
      # Torso tilt from torso_link body quaternion, not root.
      torso_idx = robot.body_names.index("torso_link")
      torso_quat = robot.data.body_link_quat_w[0, torso_idx].cpu().numpy().copy()
      tilt = float(torso_tilt_from_quat(torso_quat[None, :])[0])
      # Command (body-frame velocity) paired with this state.
      cmd = env.command_manager.get_command("twist")
      cmd_b = cmd[0].cpu().numpy().copy() if cmd is not None else None
      if is_recovery and (cmd_b is None or np.count_nonzero(cmd_b)):
        raise RuntimeError("Recovery evaluation received a nonzero or missing command")
      return {
        "t": t,
        "root_pos_w": root_pos,
        "root_quat_w": root_quat,
        "root_lin_vel_w": root_lin,
        "root_ang_vel_w": root_ang,
        "torso_tilt_deg": tilt,
        "vel_command_b": cmd_b,
      }

    entry_yaw = _yaw_from_quat_wxyz(robot.data.root_link_quat_w[0].cpu().numpy())
    snapshots.append(_snapshot(0.0))

    # Get actual trajectory/frame provenance after reset.
    actual_traj_id = -1
    actual_frame_id = -1
    if is_recovery:
      event_term_cfg = env.event_manager.get_term_cfg("tennis_recovery_reset")
      event_term = event_term_cfg.func
      actual_traj_id = int(event_term.last_trajectory_ids[0].item())
      actual_frame_id = int(event_term.last_frame_indices[0].item())
      if (actual_traj_id, actual_frame_id) != (
        descriptor.trajectory_id,
        descriptor.frame_index,
      ):
        raise ValueError(
          "Actual recovery reset does not match the planned source frame"
        )

    # Step until termination or horizon.
    max_steps = schedule.episode_length
    termination_cause = "timeout"
    terminated = False
    timed_out = False

    for step in range(max_steps):
      with torch.inference_mode():
        actions = policy(obs)
      obs, reward, dones, extras = wrapped.step(actions)

      # Check termination from the underlying env.
      # wrapped.step returns dones as terminated|truncated.
      # We need to distinguish: check the env's termination manager.
      term_dones = env.termination_manager._term_dones
      if term_dones["fell_over"][0]:
        termination_cause = "orientation"
        terminated = True
      elif term_dones["time_out"][0]:
        # A failure on the final step must not be overwritten by timeout.
        termination_cause = "timeout"
        timed_out = True

      # Capture post-step snapshot (before any auto-reset, which is off).
      snapshots.append(_snapshot((step + 1) * step_dt))

      if terminated or timed_out:
        break

    # Build trace arrays from snapshots.
    ts = np.array([s["t"] for s in snapshots], dtype=np.float64)
    pos = np.array([s["root_pos_w"] for s in snapshots], dtype=np.float64)
    quat = np.array([s["root_quat_w"] for s in snapshots], dtype=np.float64)
    lin = np.array([s["root_lin_vel_w"] for s in snapshots], dtype=np.float64)
    ang = np.array([s["root_ang_vel_w"] for s in snapshots], dtype=np.float64)
    tilt = np.array([s["torso_tilt_deg"] for s in snapshots], dtype=np.float64)
    cmd_arr = np.array(
      [s["vel_command_b"] for s in snapshots if s["vel_command_b"] is not None],
      dtype=np.float64,
    )
    vel_command_b = cmd_arr if cmd_arr.shape[0] == len(snapshots) else None

    thresholds = SettlingThresholds(max_sample_gap_s=2 * step_dt)

    if is_recovery:
      trace = RecoveryTrace(
        timestamps=ts,
        root_pos_w=pos,
        root_quat_w=quat,
        root_lin_vel_w=lin,
        root_ang_vel_w=ang,
        torso_tilt_deg=tilt,
        entry_yaw=entry_yaw,
        trajectory_id=actual_traj_id,
        group="recovery",
        termination=termination_cause,
      )
    else:
      trace = RetentionTrace(
        timestamps=ts,
        root_pos_w=pos,
        root_quat_w=quat,
        root_lin_vel_w=lin,
        root_ang_vel_w=ang,
        torso_tilt_deg=tilt,
        entry_yaw=entry_yaw,
        trajectory_id=descriptor.episode_index,
        termination=termination_cause,
        vel_command_b=vel_command_b,
      )

    metrics = compute_recovery_metrics(trace, thresholds=thresholds)
    trace_digest = hashlib.sha256()
    for array in (ts, pos, quat, lin, ang, tilt, cmd_arr):
      trace_digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes())

    return {
      "descriptor": descriptor,
      "trace_sha256": trace_digest.hexdigest(),
      "metrics": metrics,
      "actual_trajectory_id": actual_traj_id if is_recovery else None,
      "actual_frame_index": actual_frame_id if is_recovery else None,
      "planned_trajectory_id": descriptor.trajectory_id,
      "planned_frame_index": descriptor.frame_index,
      "termination_cause": termination_cause,
      "entry_yaw": entry_yaw,
      "step_dt": step_dt,
      "num_steps": len(snapshots) - 1,
    }
  finally:
    if env is not None:
      env.close()


# --------------------------------------------------------------------------- #
# Full evaluation run.
# --------------------------------------------------------------------------- #


def run_evaluation(
  schedule: EvaluationSchedule,
  pool_directory: str,
  checkpoint_path: str,
  recovery_task_id: str = (
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Recovery"
  ),
  base_task_id: str = "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation",
  device: str = "cpu",
  output_path: str | None = None,
  candidate_checkpoint_path: str | None = None,
  last_n_frames: int = 10,
) -> dict[str, Any]:
  """Run bounded evaluation and optionally compare source vs candidate.

  Runs each episode sequentially in a single-env config with
  ``auto_reset=False``. Recovery episodes force the recovery group and assign
  a validated explicit eval row. Retention episodes use the original velocity
  task config with its native command manager.

  When ``candidate_checkpoint_path`` is provided, runs the same episode
  descriptors for the candidate and compares states/commands/metrics.

  Args:
    schedule: Evaluation schedule with episode count, length, seed, split.
    pool_directory: Path to the tennis endpoint dataset directory.
    checkpoint_path: Path to the source policy checkpoint (.pt file).
    recovery_task_id: Registered task ID for the recovery environment.
    base_task_id: Registered task ID for the retention (original velocity) env.
    device: Device to run on.
    output_path: If provided, write JSON results to this path.
    candidate_checkpoint_path: If provided, also run the candidate checkpoint
      with the same descriptors and compare.
    last_n_frames: Endpoint window length in frames, in [1, trajectory length].
      Must match the window the evaluated checkpoint was trained on. It selects
      both the pool used for descriptor row indices and the reset event window,
      so both stay consistent. Recorded in ``pool.manifest``.

  Returns:
    A dict with per-episode results, group summaries, manifest, and optional
    candidate comparison.
  """
  from mjlab.tasks.velocity.mdp.tennis_endpoint_pool import EndpointPool
  from mjlab.tasks.velocity.mdp.tennis_recovery_metrics import (
    aggregate_recovery_metrics,
  )

  # Load pool to get provenance and size. The window must match the reset
  # event window, otherwise row indices would refer to different states.
  overrides = recovery_reset_event_overrides(schedule, pool_directory, last_n_frames)
  pool = EndpointPool.from_directory(
    pool_directory,
    last_n_frames=overrides["last_n_frames"],
    split=schedule.split,
    seed=schedule.seed,
  )

  descriptors = prepare_episode_descriptors(
    schedule,
    pool_size=len(pool),
    pool_trajectory_ids=pool.trajectory_ids,
    pool_frame_indices=pool.frame_indices,
  )

  # Run source episodes.
  source_results: list[dict[str, Any]] = []
  for desc in descriptors:
    result = _run_single_episode(
      descriptor=desc,
      schedule=schedule,
      checkpoint_path=checkpoint_path,
      pool_directory=pool_directory,
      device=device,
      recovery_task_id=recovery_task_id,
      base_task_id=base_task_id,
      last_n_frames=last_n_frames,
    )
    source_results.append(result)

  source_metrics = [r["metrics"] for r in source_results]
  source_group_stats = aggregate_recovery_metrics(source_metrics)

  # Run candidate episodes if requested.
  candidate_results: list[dict[str, Any]] | None = None
  candidate_group_stats = None
  if candidate_checkpoint_path is not None:
    candidate_results = []
    for desc in descriptors:
      result = _run_single_episode(
        descriptor=desc,
        schedule=schedule,
        checkpoint_path=candidate_checkpoint_path,
        pool_directory=pool_directory,
        device=device,
        recovery_task_id=recovery_task_id,
        base_task_id=base_task_id,
        last_n_frames=last_n_frames,
      )
      candidate_results.append(result)
    candidate_metrics = [r["metrics"] for r in candidate_results]
    candidate_group_stats = aggregate_recovery_metrics(candidate_metrics)

  # Build output dict.
  def _metrics_to_dict(m) -> dict:
    return {
      "trajectory_id": m.trajectory_id,
      "group": m.group,
      "termination": m.termination,
      "settled": m.settled,
      "time_to_settle": _safe_float(m.time_to_settle),
      "earliest_calm_window_s": _safe_float(m.earliest_calm_window_s),
      "censored": m.censored,
      "not_settled": m.not_settled,
      "peak_tilt_deg": m.peak_tilt_deg,
      "signed_backward_displacement_m": m.signed_backward_displacement_m,
      "total_planar_travel_m": m.total_planar_travel_m,
      "early_peak_tilt_deg": m.early_peak_tilt_deg,
      "early_signed_backward_displacement_m": m.early_signed_backward_displacement_m,
      "early_total_planar_travel_m": m.early_total_planar_travel_m,
      "early_window_s": m.early_window_s,
      "duration_s": m.duration_s,
      "num_frames": m.num_frames,
      "torque_requested_vs_applied_ratio": _safe_float(
        m.torque_requested_vs_applied_ratio
      ),
      "mean_command_lin_error_mps": _safe_float(m.mean_command_lin_error_mps),
      "mean_command_yaw_error_radps": _safe_float(m.mean_command_yaw_error_radps),
      "peak_command_lin_error_mps": _safe_float(m.peak_command_lin_error_mps),
      "contributes_settled": m.contributes_settled,
    }

  def _group_stats_to_dict(s) -> dict:
    return {
      "group": s.group,
      "num_trajectories": s.num_trajectories,
      "num_settled": s.num_settled,
      "num_falls": s.num_falls,
      "num_orientation_terminations": s.num_orientation_terminations,
      "num_timeouts": s.num_timeouts,
      "num_invalid": s.num_invalid,
      "num_not_settled": s.num_not_settled,
      "num_censored": s.num_censored,
      "settle_rate": s.settle_rate,
      "median_time_to_settle_s": _safe_float(s.median_time_to_settle_s),
      "mean_time_to_settle_s": _safe_float(s.mean_time_to_settle_s),
      "mean_peak_tilt_deg": _safe_float(s.mean_peak_tilt_deg),
      "mean_signed_backward_displacement_m": _safe_float(
        s.mean_signed_backward_displacement_m
      ),
      "mean_total_planar_travel_m": _safe_float(s.mean_total_planar_travel_m),
      "coverage": s.coverage,
      "mean_command_lin_error_mps": _safe_float(s.mean_command_lin_error_mps),
      "mean_command_yaw_error_radps": _safe_float(s.mean_command_yaw_error_radps),
      "num_command_tracked": s.num_command_tracked,
    }

  result: dict[str, Any] = {
    "schedule": {
      "num_episodes": schedule.num_episodes,
      "episode_length": schedule.episode_length,
      "duration_s": schedule.duration_s,
      "fps": schedule.fps,
      "seed": schedule.seed,
      "split": schedule.split,
      "recovery_fraction": schedule.recovery_fraction,
    },
    "checkpoint": {
      "path": checkpoint_path,
      "sha256": _sha256_file(checkpoint_path),
    },
    "pool": {
      "directory": str(Path(pool_directory).resolve()),
      "manifest": pool.manifest,
    },
    "source": {
      "per_episode": [
        {
          **_metrics_to_dict(r["metrics"]),
          "planned_trajectory_id": r["planned_trajectory_id"],
          "planned_frame_index": r["planned_frame_index"],
          "actual_trajectory_id": r["actual_trajectory_id"],
          "actual_frame_index": r["actual_frame_index"],
          "termination_cause": r["termination_cause"],
          "entry_yaw": r["entry_yaw"],
          "step_dt": r["step_dt"],
          "num_steps": r["num_steps"],
          "episode_seed": r["descriptor"].episode_seed,
          "trace_sha256": r["trace_sha256"],
        }
        for r in source_results
      ],
      "group_summary": [_group_stats_to_dict(s) for s in source_group_stats],
    },
  }

  if candidate_results is not None and candidate_group_stats is not None:
    result["candidate"] = {
      "checkpoint": {
        "path": candidate_checkpoint_path,
        "sha256": _sha256_file(candidate_checkpoint_path)
        if candidate_checkpoint_path is not None
        else None,
      },
      "per_episode": [
        {
          **_metrics_to_dict(r["metrics"]),
          "planned_trajectory_id": r["planned_trajectory_id"],
          "planned_frame_index": r["planned_frame_index"],
          "actual_trajectory_id": r["actual_trajectory_id"],
          "actual_frame_index": r["actual_frame_index"],
          "termination_cause": r["termination_cause"],
          "entry_yaw": r["entry_yaw"],
          "step_dt": r["step_dt"],
          "num_steps": r["num_steps"],
          "episode_seed": r["descriptor"].episode_seed,
          "trace_sha256": r["trace_sha256"],
        }
        for r in candidate_results
      ],
      "group_summary": [_group_stats_to_dict(s) for s in candidate_group_stats],
    }

  if output_path is not None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
      json.dump(result, f, indent=2, allow_nan=False)

  return result


# --------------------------------------------------------------------------- #
# CLI entry point.
# --------------------------------------------------------------------------- #


def main() -> None:
  """Bounded evaluation CLI entry point.

  Uses ``mjlab.TYRO_FLAGS`` for consistent CLI parsing. Builds the recovery
  environment from the registered task, loads a checkpoint, and runs
  deterministic held-out evaluation episodes.
  """
  import tyro

  import mjlab

  tyro.cli(_run_evaluation_cli, config=mjlab.TYRO_FLAGS)


def _run_evaluation_cli(
  checkpoint: str,
  pool_directory: str = "data/tennis",
  recovery_task_id: str = (
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Recovery"
  ),
  base_task_id: str = "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation",
  num_episodes: int = 10,
  episode_length: int = 1000,
  duration_s: float = 20.0,
  fps: float = 50.0,
  seed: int = 42,
  split: str = "validation",
  recovery_fraction: float = 0.8,
  last_n_frames: int = 10,
  device: str = "cpu",
  output: str = "tennis_recovery_eval.json",
  candidate_checkpoint: str | None = None,
) -> None:
  """Run bounded tennis-end recovery evaluation.

  Each episode runs sequentially in a single-env config with
  ``auto_reset=False``. Recovery episodes force the recovery group and assign
  a validated explicit eval row. Retention episodes use the original velocity
  task config with its native command manager.

  Args:
    checkpoint: Path to the source policy checkpoint (.pt file).
    pool_directory: Path to the tennis endpoint dataset directory.
    recovery_task_id: Registered task ID for the recovery environment.
    base_task_id: Registered task ID for the retention (original velocity) env.
    num_episodes: Number of evaluation episodes.
    episode_length: Frames per episode.
    duration_s: Episode duration in seconds.
    fps: Sampling rate.
    seed: Seed for deterministic held-out row selection.
    split: Pool split to evaluate on (train/validation/all).
    recovery_fraction: Fraction of recovery episodes.
    last_n_frames: Endpoint window length in frames; must match the evaluated
      checkpoint's training window (default 10).
    device: Device to run on.
    output: Output JSON path for per-episode metrics.
    candidate_checkpoint: Optional candidate checkpoint for comparison.
  """
  schedule = EvaluationSchedule(
    num_episodes=num_episodes,
    episode_length=episode_length,
    duration_s=duration_s,
    fps=fps,
    seed=seed,
    split=split,
    recovery_fraction=recovery_fraction,
  )
  result = run_evaluation(
    schedule=schedule,
    pool_directory=pool_directory,
    checkpoint_path=checkpoint,
    recovery_task_id=recovery_task_id,
    base_task_id=base_task_id,
    device=device,
    output_path=output,
    candidate_checkpoint_path=candidate_checkpoint,
    last_n_frames=last_n_frames,
  )
  for s in result["source"]["group_summary"]:
    print(
      f"[source/{s['group']}] n={s['num_trajectories']} "
      f"settled={s['num_settled']} ({s['settle_rate']:.1%}) "
      f"falls={s['num_falls']} orient={s['num_orientation_terminations']} "
      f"timeout={s['num_timeouts']}"
    )
  if "candidate" in result:
    for s in result["candidate"]["group_summary"]:
      print(
        f"[candidate/{s['group']}] n={s['num_trajectories']} "
        f"settled={s['num_settled']} ({s['settle_rate']:.1%}) "
        f"falls={s['num_falls']} orient={s['num_orientation_terminations']} "
        f"timeout={s['num_timeouts']}"
      )


if __name__ == "__main__":
  main()
