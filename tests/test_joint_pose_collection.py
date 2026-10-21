"""Focused CPU tests for the raw X2 joint-pose research collector.

The fake environment mirrors the manual-reset contract these tests need:
terminations are computed before the command frame advances, every done row must
be reset before the next step while auto-reset is off, and a reset writes a
poison state that must never reach a recorded trajectory.  No MuJoCo model is
compiled.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from tracking_distillation_fixtures import write_reference_clip

from mjlab.managers.termination_manager import TerminationManager, TerminationTermCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.multi_motion_ppo import (
  MultiMotionPpoCommand,
  MultiMotionPpoCommandCfg,
  clip_complete,
)

TASK = "Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion"
COLLECTOR_SCRIPT = (
  pathlib.Path(__file__).resolve().parents[1]
  / "scripts"
  / "experiments"
  / "joint_pose_collection"
  / "collect.py"
)
JOINT_NAMES = ("joint_a", "joint_b", "joint_c")
JOINT_DIM = len(JOINT_NAMES)
STEP_DT = 0.02
START_MARKER = 1.0
STEP_MARKER = 2.0
RESET_MARKER = 3.0


def _collector_module():
  """Load the collector script, which is not an importable package."""
  spec = importlib.util.spec_from_file_location(
    "joint_pose_collector_under_test", COLLECTOR_SCRIPT
  )
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def _checkpoint(tmp_path: Path, *, lookahead_s: float = 0.2) -> Path:
  checkpoint = tmp_path / "run" / "model_0.pt"
  checkpoint.parent.mkdir(parents=True)
  checkpoint.write_bytes(b"")
  params = checkpoint.parent / "params"
  params.mkdir()
  (params / "env.yaml").write_text(
    f"commands:\n  motion:\n    lookahead_s: {lookahead_s}\n",
    encoding="utf-8",
  )
  return checkpoint


def _clip(tmp_path: Path, *, frames: int = 4) -> Path:
  return write_reference_clip(
    tmp_path / "data" / "clip_000_agibot_x2_tracking.npz",
    frames=frames,
    joint_dim=JOINT_DIM,
  )


class _FakeCommand(MultiMotionPpoCommand):
  """Scripted command state; reference fields are deliberately not robot state."""

  def __init__(self, rows: int, frames: int) -> None:
    self._num_envs = rows
    self._device = "cpu"
    self.time_steps = torch.zeros(rows, dtype=torch.long)
    self._row_lengths = torch.full((rows,), frames, dtype=torch.long)
    self.reference_joint_pos = torch.full((rows, JOINT_DIM), -777.0)
    self.reference_joint_vel = torch.full((rows, JOINT_DIM), -888.0)
    self.cfg = MotionCommandCfg(
      motion_file="",
      anchor_body_name="torso_link",
      body_names=("pelvis",),
      entity_name="robot",
      lookahead_s=0.2,
      resampling_time_range=(1.0e9, 1.0e9),
    )

  @property
  def num_envs(self) -> int:
    return self._num_envs

  @property
  def device(self) -> str:
    return self._device

  @property
  def row_lengths(self) -> torch.Tensor:
    return self._row_lengths


def _failure_term(env: Any) -> torch.Tensor:
  return env.failure_script[env.step_index].clone()


def _timeout_term(env: Any) -> torch.Tensor:
  return env.timeout_script[env.step_index].clone()


class _FakeEnv:
  """Minimal vector env with the real manual-reset and termination contract."""

  def __init__(
    self,
    *,
    rows: int,
    frames: int,
    failures: dict[int, set[int]] | None = None,
    steps: int | None = None,
    push_events: dict[int, list[int]] | None = None,
    push_recorder: Any = None,
  ) -> None:
    self.num_envs = rows
    self.device = "cpu"
    self.step_dt = STEP_DT
    self.command = _FakeCommand(rows, frames)
    self.unwrapped = self
    self.command_manager = SimpleNamespace(get_term=lambda name: self.command)
    self.scene = {"robot": SimpleNamespace(joint_names=JOINT_NAMES, data=None)}
    self.robot = self.scene["robot"]
    self.robot.data = SimpleNamespace(
      root_link_quat_w=torch.zeros(rows, 4),
      joint_pos=torch.zeros(rows, JOINT_DIM),
      joint_vel=torch.zeros(rows, JOINT_DIM),
    )
    # Initial achieved state is distinct per row and per channel.
    for index in range(rows):
      self.robot.data.root_link_quat_w[index] = torch.tensor(
        [1.0, 0.01 * (index + 1), 0.02, 0.03]
      )
      base = torch.arange(JOINT_DIM, dtype=torch.float32)
      self.robot.data.joint_pos[index] = 1000.0 * (index + 1) + base
      self.robot.data.joint_vel[index] = 2000.0 * (index + 1) + base

    failures = failures or {}
    script_steps = max(frames, steps or 0, 1) + 1
    failure_script = torch.zeros(script_steps, rows, dtype=torch.bool)
    for row, row_failures in failures.items():
      for step in row_failures:
        failure_script[step, row] = True
    self.failure_script = tuple(failure_script[step] for step in range(script_steps))
    self.timeout_script = tuple(
      torch.zeros(rows, dtype=torch.bool) for _ in range(script_steps)
    )
    self.termination_manager = TerminationManager(
      {
        "time_out": TerminationTermCfg(func=_timeout_term, time_out=True),
        "anchor_pos": TerminationTermCfg(func=_failure_term, time_out=False),
        "clip_complete": TerminationTermCfg(func=clip_complete, time_out=False),
      },
      cast(Any, self),
    )
    self.step_index = 0
    self.episode_length_buf = torch.zeros(rows, dtype=torch.long)
    self.push_events = push_events or {}
    self.push_recorder = push_recorder
    self.reset_step = 0
    self.pending = torch.zeros(rows, dtype=torch.bool)
    self.resets: list[list[int]] = []
    self.obs = {"marker": torch.full((rows,), START_MARKER)}
    self.closed = False
    self.actions: list[torch.Tensor] = []

  def get_observations(self) -> dict[str, torch.Tensor]:
    return self.obs

  def reset(self, *, env_ids: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict]:
    self.resets.append([int(index) for index in env_ids.tolist()])
    self.pending[env_ids] = False
    self.command.time_steps[env_ids] = 0
    self.episode_length_buf[env_ids] = 0
    self.reset_step += 1
    # A recorded post-reset row would carry these values.
    self.robot.data.root_link_quat_w[env_ids] = torch.tensor([0.5, 0.5, 0.5, 0.5])
    self.robot.data.joint_pos[env_ids] = -9999.0
    self.robot.data.joint_vel[env_ids] = -9999.0
    self.obs = {"marker": torch.full((self.num_envs,), RESET_MARKER)}
    return self.obs, {}

  def step(
    self, actions: torch.Tensor
  ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
    if bool(self.pending.any()):
      raise RuntimeError(
        "a done row was stepped without the manual reset the environment requires"
      )
    self.actions.append(actions.detach().clone())
    # Distinct achieved state for every row and control step.  Inactive rows
    # after a reset also receive fresh-episode values, which must not be
    # recorded as a continuation of their first trajectory.
    index = torch.arange(self.num_envs, dtype=torch.float32)[:, None]
    base = torch.arange(JOINT_DIM, dtype=torch.float32)[None, :]
    tick = float(self.step_index + 1)
    self.robot.data.root_link_quat_w[:, 0] = 1.0 + 0.001 * tick
    self.robot.data.joint_pos[:] = 1000.0 * (index + 1.0) + base + 100.0 * tick
    self.robot.data.joint_vel[:] = 2000.0 * (index + 1.0) + base - 10.0 * tick
    self.episode_length_buf += 1
    dones = self.termination_manager.compute()
    pushed_rows = self.push_events.get(self.step_index)
    if pushed_rows:
      if self.push_recorder is None:
        raise RuntimeError("scripted push has no recorder")
      self.push_recorder.record(self, torch.tensor(pushed_rows, dtype=torch.long))
    self.command.time_steps = torch.minimum(
      self.command.time_steps + 1, self.command.row_lengths - 1
    )
    self.pending |= dones
    self.step_index += 1
    self.obs = {"marker": torch.full((self.num_envs,), STEP_MARKER)}
    return self.obs, torch.zeros(self.num_envs), dones.to(torch.long), {}

  def close(self) -> None:
    self.closed = True


def test_public_parser_validation_and_overwrite_refusal(tmp_path: Path) -> None:
  module = _collector_module()
  checkpoint = _checkpoint(tmp_path)
  clip = _clip(tmp_path)
  output = tmp_path / "run_output"

  config = module.parse_config(
    [
      "--checkpoint-file",
      str(checkpoint),
      "--motion-file",
      str(clip),
      "--output-dir",
      str(output),
      "--num-envs",
      "3",
      "--seed",
      "11",
      "--max-steps",
      "4",
    ]
  )
  assert config.checkpoint_file == checkpoint
  assert config.motion_file == [clip]
  assert config.output_dir == output
  assert config.num_envs == 3
  assert config.seed == 11
  assert config.num_seeds == 1
  assert config.sampling_mode == "start"
  assert config.push_interval_s is None
  assert config.max_steps == 4
  module.validate_config(config)

  with pytest.raises(ValueError, match="num_envs"):
    module.validate_config(replace(config, num_envs=0))
  with pytest.raises(ValueError, match="seed"):
    module.validate_config(replace(config, seed=-1))
  with pytest.raises(ValueError, match="num_seeds"):
    module.validate_config(replace(config, num_seeds=0))
  with pytest.raises(ValueError, match="sampling_mode"):
    module.validate_config(replace(config, sampling_mode=cast(Any, "adaptive")))
  with pytest.raises(ValueError, match="max_steps"):
    module.validate_config(replace(config, max_steps=0))
  with pytest.raises(ValueError, match="push_interval_s"):
    module.validate_config(replace(config, push_interval_s=(-1.0, 2.0)))
  with pytest.raises(ValueError, match="push_interval_s"):
    module.validate_config(replace(config, push_interval_s=(1.0, float("nan"))))
  with pytest.raises(ValueError, match="push_linear_velocity_range"):
    module.validate_config(replace(config, push_linear_velocity_range=(0.5, -0.5)))
  with pytest.raises(ValueError, match="push_angular_velocity_range"):
    module.validate_config(
      replace(config, push_angular_velocity_range=(0.0, float("inf")))
    )
  with pytest.raises(FileNotFoundError, match="reference clip"):
    module.validate_config(replace(config, motion_file=[tmp_path / "missing.npz"]))
  no_params = tmp_path / "no_params" / "model_0.pt"
  no_params.parent.mkdir(parents=True)
  no_params.write_bytes(b"")
  with pytest.raises(RuntimeError, match="lookahead"):
    module.validate_config(replace(config, checkpoint_file=no_params))

  created = module.prepare_output_dir(output)
  assert created == output.resolve()
  assert (created / "trajectories").is_dir()
  with pytest.raises(FileExistsError, match="already exists"):
    module.prepare_output_dir(output)


def test_public_parser_reads_seed_phase_and_push_options(tmp_path: Path) -> None:
  module = _collector_module()
  checkpoint = _checkpoint(tmp_path)
  clip = _clip(tmp_path)

  config = module.parse_config(
    [
      "--checkpoint-file",
      str(checkpoint),
      "--motion-file",
      str(clip),
      "--output-dir",
      str(tmp_path / "run_output"),
      "--num-seeds",
      "3",
      "--sampling-mode",
      "uniform",
      "--push-interval-s",
      "1.0",
      "2.0",
      "--push-linear-velocity-range",
      "-0.3",
      "0.4",
      "--push-angular-velocity-range",
      "-0.6",
      "0.7",
    ]
  )

  assert config.num_seeds == 3
  assert config.sampling_mode == "uniform"
  assert config.push_interval_s == (1.0, 2.0)
  assert config.push_linear_velocity_range == (-0.3, 0.4)
  assert config.push_angular_velocity_range == (-0.6, 0.7)
  assert module.collection_scenario(config) == "disturbed"
  module.validate_config(config)


def test_configure_preserves_saved_preview_and_reports_defaults(tmp_path: Path) -> None:
  module = _collector_module()
  clip = _clip(tmp_path)
  checkpoint = _checkpoint(tmp_path, lookahead_s=0.2)
  env_cfg = load_env_cfg(TASK, play=False)
  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MultiMotionPpoCommandCfg)
  pose_range = dict(motion_cmd.pose_range)
  velocity_range = dict(motion_cmd.velocity_range)
  joint_position_range = motion_cmd.joint_position_range

  settings = module.configure_collection_env(
    env_cfg,
    clip,
    checkpoint,
    num_envs=4,
    seed=7,
    episode_length_s=0.08,
  )

  assert env_cfg.auto_reset is False
  assert env_cfg.episode_length_s == pytest.approx(0.08)
  assert env_cfg.scene.num_envs == 4
  assert env_cfg.seed == 7
  assert env_cfg.observations["actor"].enable_corruption is True
  assert "push_robot" not in env_cfg.events
  assert motion_cmd.motion_manifest == ""
  assert motion_cmd.motion_file == str(clip)
  assert motion_cmd.sampling_mode == "start"
  assert motion_cmd.lookahead_s == pytest.approx(0.2)
  assert settings == {
    "motion_file": str(clip),
    "num_envs": 4,
    "seed": 7,
    "episode_length_s": 0.08,
    "lookahead_s": 0.2,
    "sampling_mode": "start",
    "standing_start_prob": 0.0,
    "auto_reset": False,
    "actor_corruption": True,
    "push_robot_event": False,
    "push_interval_s": None,
    "push_velocity_range": None,
  }
  assert dict(motion_cmd.pose_range) == pose_range
  assert dict(motion_cmd.velocity_range) == velocity_range
  assert motion_cmd.joint_position_range == joint_position_range

  random_phase_cfg = load_env_cfg(TASK, play=False)
  random_motion_cmd = random_phase_cfg.commands["motion"]
  assert isinstance(random_motion_cmd, MultiMotionPpoCommandCfg)
  random_pose_range = dict(random_motion_cmd.pose_range)
  random_velocity_range = dict(random_motion_cmd.velocity_range)
  random_joint_range = random_motion_cmd.joint_position_range
  random_settings = module.configure_collection_env(
    random_phase_cfg,
    clip,
    checkpoint,
    num_envs=4,
    seed=7,
    episode_length_s=0.08,
    sampling_mode="uniform",
  )
  assert random_motion_cmd.sampling_mode == "uniform"
  assert random_motion_cmd.lookahead_s == pytest.approx(0.2)
  assert dict(random_motion_cmd.pose_range) == random_pose_range
  assert dict(random_motion_cmd.velocity_range) == random_velocity_range
  assert random_motion_cmd.joint_position_range == random_joint_range
  assert random_settings["sampling_mode"] == "uniform"
  assert random_settings["standing_start_prob"] == 0.0
  assert "push_robot" not in random_phase_cfg.events

  missing = tmp_path / "no_params" / "model_0.pt"
  missing.parent.mkdir(parents=True)
  missing.write_bytes(b"")
  with pytest.raises(RuntimeError, match="lookahead"):
    module.configure_collection_env(
      load_env_cfg(TASK, play=False),
      clip,
      missing,
      num_envs=1,
      seed=0,
      episode_length_s=0.08,
    )


def test_configure_installs_recorded_disturbed_push(tmp_path: Path) -> None:
  module = _collector_module()
  clip = _clip(tmp_path)
  checkpoint = _checkpoint(tmp_path)
  recorder = module.PushRecorder(2)
  velocity_range = module.resolved_push_velocity_range(
    module.CollectionConfig(
      checkpoint_file=checkpoint,
      motion_file=[clip],
      output_dir=tmp_path / "run",
      push_linear_velocity_range=(-0.3, 0.4),
      push_angular_velocity_range=(-0.6, 0.7),
      push_interval_s=(1.0, 2.0),
    )
  )
  env_cfg = load_env_cfg(TASK, play=False)

  settings = module.configure_collection_env(
    env_cfg,
    clip,
    checkpoint,
    num_envs=2,
    seed=9,
    episode_length_s=0.08,
    push_interval_s=(1.0, 2.0),
    push_velocity_range=velocity_range,
    push_recorder=recorder,
  )

  event = env_cfg.events["push_robot"]
  assert event.mode == "interval"
  assert event.interval_range_s == (1.0, 2.0)
  assert event.params == {}
  assert event.func.recorder is recorder
  assert event.func.velocity_range == velocity_range
  assert copy.deepcopy(event.func) is event.func
  assert copy.deepcopy(event).func.recorder is recorder
  assert settings["push_robot_event"] is True
  assert settings["push_interval_s"] == [1.0, 2.0]
  assert settings["push_velocity_range"] == {
    "x": [-0.3, 0.4],
    "y": [-0.3, 0.4],
    "z": [-0.3, 0.4],
    "roll": [-0.6, 0.7],
    "pitch": [-0.6, 0.7],
    "yaw": [-0.6, 0.7],
  }

  with pytest.raises(ValueError, match="disturbed collection"):
    module.configure_collection_env(
      load_env_cfg(TASK, play=False),
      clip,
      checkpoint,
      num_envs=2,
      seed=9,
      episode_length_s=0.08,
      push_interval_s=(1.0, 2.0),
    )


def test_collect_clip_records_actual_state_and_async_trajectories() -> None:
  module = _collector_module()
  env = _FakeEnv(
    rows=3,
    frames=4,
    failures={0: {0, 2}, 2: {2}},
    steps=4,
  )
  grad_modes: list[bool] = []
  seen_markers: list[float] = []

  def policy(obs: Any) -> torch.Tensor:
    grad_modes.append(torch.is_grad_enabled())
    seen_markers.append(float(obs["marker"][0]))
    return torch.zeros(env.num_envs, 1)

  rollout = module.collect_clip(env, env.command, policy, step_budget=4)

  assert grad_modes == [False, False, False, False]
  assert seen_markers == [START_MARKER, RESET_MARKER, STEP_MARKER, RESET_MARKER]
  assert rollout.trajectory_outcome.tolist() == [
    "failure",
    "complete",
    "failure",
  ]
  assert rollout.trajectory_steps.tolist() == [1, 4, 3]
  assert rollout.trajectory_offsets.tolist() == [0, 2, 7, 11]
  assert rollout.env_index.tolist() == [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2]
  assert rollout.tick.tolist() == [0, 1, 0, 1, 2, 3, 4, 0, 1, 2, 3]
  assert rollout.joint_names == JOINT_NAMES
  assert rollout.control_dt_s == pytest.approx(STEP_DT)
  assert rollout.root_quat_wxyz.shape == (11, 4)
  assert rollout.joint_pos.shape == (11, JOINT_DIM)
  assert rollout.joint_vel.shape == (11, JOINT_DIM)
  assert np.allclose(
    rollout.time_s, rollout.tick.astype(np.float32) * np.float32(STEP_DT)
  )

  # The initial row is the genuine start state, not the reference command.
  assert rollout.root_quat_wxyz[0].tolist() == pytest.approx([1.0, 0.01, 0.02, 0.03])
  assert rollout.root_quat_wxyz[1].tolist() == pytest.approx([1.001, 0.01, 0.02, 0.03])
  assert rollout.joint_vel[0].tolist() == pytest.approx([2000.0, 2001.0, 2002.0])
  assert rollout.joint_vel[1].tolist() == pytest.approx([1990.0, 1991.0, 1992.0])
  assert rollout.joint_pos[0, 0] == pytest.approx(1000.0)
  assert rollout.joint_pos[2, 0] == pytest.approx(2000.0)
  assert rollout.joint_pos[7, 0] == pytest.approx(3000.0)
  # The terminal row of row 0 is the failure state of its first episode.
  assert rollout.joint_pos[1, 0] == pytest.approx(1100.0)
  # Row 1 completes on its fourth step.
  assert rollout.joint_pos[6, 0] == pytest.approx(2400.0)
  # Row 2 fails on its third step.
  assert rollout.joint_pos[10, 0] == pytest.approx(3300.0)
  # No reset poison and no reference-command value reached the arrays.
  assert not np.any(rollout.joint_pos == -9999.0)
  assert not np.any(rollout.joint_pos == -777.0)
  assert not np.any(rollout.joint_vel == -888.0)
  assert not np.any(rollout.root_quat_wxyz == 0.5)

  # Every row the environment marked done was reset before the next step.
  assert env.resets == [[0], [0, 2], [1]]


def test_collect_clip_records_initial_frame_and_late_completion() -> None:
  module = _collector_module()
  env = _FakeEnv(rows=3, frames=6)
  env.command.time_steps = torch.tensor([5, 0, 3], dtype=torch.long)

  rollout = module.collect_clip(
    env,
    env.command,
    lambda obs: torch.zeros(3, 1),
    step_budget=6,
    seed=11,
    run_index=2,
  )

  assert rollout.trajectory_initial_frame.tolist() == [5, 0, 3]
  assert rollout.trajectory_steps.tolist() == [1, 6, 3]
  assert rollout.trajectory_outcome.tolist() == ["complete"] * 3
  assert rollout.trajectory_offsets.tolist() == [0, 2, 9, 13]
  assert rollout.tick[0:2].tolist() == [0, 1]
  assert rollout.tick[2:9].tolist() == [0, 1, 2, 3, 4, 5, 6]
  assert rollout.tick[9:13].tolist() == [0, 1, 2, 3]
  assert rollout.trajectory_seed.tolist() == [11, 11, 11]
  assert rollout.trajectory_run_index.tolist() == [2, 2, 2]
  assert rollout.trajectory_scenario.tolist() == ["baseline"] * 3
  assert rollout.trajectory_push_count.tolist() == [0, 0, 0]
  assert rollout.push_offsets.tolist() == [0, 0, 0, 0]
  assert rollout.push_tick.size == 0


def test_collect_clip_push_evidence_uses_first_episode_only() -> None:
  module = _collector_module()
  recorder = module.PushRecorder(2)
  env = _FakeEnv(
    rows=2,
    frames=8,
    failures={0: {1}},
    push_events={
      0: [0, 1],
      1: [0, 1],
      2: [1],
      3: [0],
      4: [1],
    },
    push_recorder=recorder,
  )

  rollout = module.collect_clip(
    env,
    env.command,
    lambda obs: torch.zeros(2, 1),
    step_budget=8,
    scenario="disturbed",
    push_recorder=recorder,
  )

  assert rollout.trajectory_outcome.tolist() == ["failure", "complete"]
  assert rollout.trajectory_steps.tolist() == [2, 8]
  assert rollout.trajectory_scenario.tolist() == ["disturbed", "disturbed"]
  assert rollout.trajectory_push_count.tolist() == [2, 4]
  assert rollout.push_offsets.tolist() == [0, 2, 6]
  # Row 0 pushes on tick 1 and on its terminal tick 2. The push at step index
  # 3 is in a later episode on the reset row and must not be recorded.
  assert rollout.push_tick.tolist() == [1, 2, 1, 2, 3, 5]
  assert rollout.push_env_index.tolist() == [0, 0, 1, 1, 1, 1]

  quiet_recorder = module.PushRecorder(2)
  quiet_env = _FakeEnv(rows=2, frames=2, push_recorder=quiet_recorder)
  quiet = module.collect_clip(
    quiet_env,
    quiet_env.command,
    lambda obs: torch.zeros(2, 1),
    step_budget=2,
    scenario="disturbed",
    push_recorder=quiet_recorder,
  )
  assert quiet.trajectory_outcome.tolist() == ["complete", "complete"]
  assert quiet.trajectory_push_count.tolist() == [0, 0]
  assert quiet.push_offsets.tolist() == [0, 0, 0]
  assert quiet.push_tick.size == 0


def test_collect_clip_budget_labels_and_refuses_zero_budget() -> None:
  module = _collector_module()
  env = _FakeEnv(rows=2, frames=6)
  policy = lambda obs: torch.zeros(env.num_envs, 1)  # noqa: E731

  rollout = module.collect_clip(env, env.command, policy, step_budget=2)

  assert rollout.trajectory_outcome.tolist() == ["budget", "budget"]
  assert rollout.trajectory_steps.tolist() == [2, 2]
  assert rollout.trajectory_offsets.tolist() == [0, 3, 6]
  assert rollout.tick.tolist() == [0, 1, 2, 0, 1, 2]
  assert env.resets == []
  assert not np.any(rollout.joint_pos == -9999.0)

  with pytest.raises(ValueError, match="step_budget"):
    module.collect_clip(env, env.command, policy, step_budget=0)


@pytest.mark.parametrize("field", ["root_quat_wxyz", "joint_pos", "joint_vel"])
def test_write_refuses_nonfinite_achieved_state(tmp_path: Path, field: str) -> None:
  module = _collector_module()
  env = _FakeEnv(rows=1, frames=2)
  rollout = module.collect_clip(
    env, env.command, lambda obs: torch.zeros(1, 1), step_budget=1
  )
  getattr(rollout, field)[0, 0] = np.nan
  output = tmp_path / "bad.npz"
  with pytest.raises(ValueError, match=f"{field} contains non-finite"):
    module.write_clip_npz(output, rollout)
  assert not output.exists()


class _FakeRunner:
  """Runner stand-in that records the load and returns one frozen policy."""

  loaded: list[str] = []
  learn_called = False

  def __init__(self, env: Any, train_cfg: dict, device: str) -> None:
    self.env = env
    self.train_cfg = train_cfg
    self.device = device
    self.policy = lambda obs: torch.zeros(env.num_envs, 1)  # noqa: E731

  def load(self, path: str, map_location: str | None = None) -> None:
    self.loaded.append(path)

  def get_inference_policy(self, device: str | None = None) -> Any:
    return self.policy

  def learn(self, *_args: Any, **_kwargs: Any) -> None:
    self.learn_called = True


def test_collect_writes_npz_and_manifest_without_simulator(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  module = _collector_module()
  checkpoint = _checkpoint(tmp_path)
  clip = _clip(tmp_path, frames=4)
  second_clip = write_reference_clip(
    tmp_path / "data" / "clip_001_agibot_x2_tracking.npz",
    frames=3,
    joint_dim=JOINT_DIM,
  )
  output = tmp_path / "run_output"
  envs: list[_FakeEnv] = []
  clip_frames = iter((4, 3))

  def build_env(cfg: Any, device: str) -> _FakeEnv:
    env = _FakeEnv(rows=cfg.scene.num_envs, frames=next(clip_frames))
    envs.append(env)
    return env

  monkeypatch.setattr(module, "ManagerBasedRlEnv", build_env)
  monkeypatch.setattr(module, "RslRlVecEnvWrapper", lambda env, clip_actions=None: env)
  monkeypatch.setattr(module, "load_runner_cls", lambda task: _FakeRunner)
  monkeypatch.setattr(module, "configure_torch_backends", lambda: None)
  _FakeRunner.loaded = []
  _FakeRunner.learn_called = False

  config = module.CollectionConfig(
    checkpoint_file=checkpoint,
    motion_file=[clip, second_clip],
    output_dir=output,
    num_envs=3,
    seed=5,
  )
  manifest = module.collect(config)

  assert manifest["complete"] is True
  assert manifest["checkpoint"]["file"] == str(checkpoint.resolve())
  assert manifest["collection"]["lookahead_s"] == pytest.approx(0.2)
  assert manifest["collection"]["actor_corruption"] is True
  assert manifest["collection"]["push_robot_event"] is False
  assert manifest["collection"]["auto_reset"] is False
  assert manifest["collection"]["num_seeds"] == 1
  assert manifest["collection"]["seeds"] == [5]
  assert manifest["collection"]["num_clips"] == 2
  assert manifest["collection"]["num_seed_runs"] == 1
  assert manifest["collection"]["num_clip_rollouts"] == 2
  assert manifest["collection"]["total_trajectories"] == 6
  assert manifest["collection"]["scenario"] == "baseline"
  assert manifest["collection"]["disturbance"] is None
  assert manifest["collection"]["sampling_mode"] == "start"
  assert manifest["collection"]["standing_start_prob"] == 0.0
  assert manifest["schema_version"] == 2
  assert manifest["joint_names"] == list(JOINT_NAMES)
  assert manifest["control_dt_s"] == pytest.approx(STEP_DT)
  assert manifest["clips"][0]["file"] == str(clip.resolve())
  assert manifest["clips"][0]["frames"] == 4
  assert manifest["clips"][0]["run_index"] == 0
  assert manifest["clips"][0]["seed"] == 5
  assert manifest["clips"][0]["scenario"] == "baseline"
  assert manifest["clips"][0]["disturbance"] is None
  assert manifest["clips"][0]["sampling_mode"] == "start"
  assert manifest["clips"][0]["pushes_observed"] == 0
  assert manifest["clips"][0]["step_budget"] == 4
  assert manifest["clips"][0]["budget_source"] == "clip_length"
  assert manifest["clips"][0]["outcomes"] == {
    "complete": 3,
    "failure": 0,
    "budget": 0,
  }
  assert manifest["clips"][1]["file"] == str(second_clip.resolve())
  assert manifest["clips"][1]["frames"] == 3
  assert manifest["clips"][1]["run_index"] == 0
  assert manifest["clips"][1]["seed"] == 5
  assert manifest["clips"][1]["step_budget"] == 3
  assert manifest["clips"][1]["budget_source"] == "clip_length"
  assert manifest["clips"][1]["outcomes"] == {
    "complete": 3,
    "failure": 0,
    "budget": 0,
  }
  assert _FakeRunner.loaded == [str(checkpoint.resolve())]
  assert _FakeRunner.learn_called is False
  assert len(envs) == 2 and all(env.closed for env in envs)

  written = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
  assert written == manifest

  payload = np.load(output / "trajectories" / "000_clip_000_agibot_x2_tracking.npz")
  try:
    assert set(payload.files) == {
      "root_quat_wxyz",
      "joint_pos",
      "joint_vel",
      "tick",
      "time_s",
      "env_index",
      "trajectory_offsets",
      "trajectory_env_index",
      "trajectory_outcome",
      "trajectory_steps",
      "trajectory_initial_frame",
      "trajectory_seed",
      "trajectory_run_index",
      "trajectory_scenario",
      "trajectory_push_count",
      "push_offsets",
      "push_tick",
      "push_env_index",
    }
    assert payload["root_quat_wxyz"].shape == (15, 4)
    assert payload["joint_pos"].shape == (15, JOINT_DIM)
    assert payload["joint_vel"].shape == (15, JOINT_DIM)
    assert payload["trajectory_offsets"].tolist() == [0, 5, 10, 15]
    assert payload["trajectory_outcome"].tolist() == ["complete"] * 3
    assert payload["trajectory_initial_frame"].tolist() == [0, 0, 0]
    assert payload["trajectory_seed"].tolist() == [5, 5, 5]
    assert payload["trajectory_run_index"].tolist() == [0, 0, 0]
    assert payload["trajectory_scenario"].tolist() == ["baseline"] * 3
    assert payload["trajectory_push_count"].tolist() == [0, 0, 0]
    assert payload["push_offsets"].tolist() == [0, 0, 0, 0]
    assert payload["push_tick"].size == 0
    assert payload["push_env_index"].size == 0
    assert payload["tick"].tolist() == [0, 1, 2, 3, 4] * 3
    # Joint order is the robot's order: the first three initial rows are
    # 1000, 2000 and 3000, each with the joint index added in column order.
    assert payload["joint_pos"][0].tolist() == [1000.0, 1001.0, 1002.0]
    assert payload["joint_pos"][5].tolist() == [2000.0, 2001.0, 2002.0]
    assert payload["joint_pos"][10].tolist() == [3000.0, 3001.0, 3002.0]
  finally:
    payload.close()

  second_payload = np.load(
    output / "trajectories" / "001_clip_001_agibot_x2_tracking.npz"
  )
  try:
    assert second_payload["trajectory_offsets"].tolist() == [0, 4, 8, 12]
    assert second_payload["trajectory_outcome"].tolist() == ["complete"] * 3
    assert second_payload["tick"].tolist() == [0, 1, 2, 3] * 3
  finally:
    second_payload.close()

  with pytest.raises(FileExistsError, match="already exists"):
    module.collect(config)
  assert len(envs) == 2


def test_collect_multi_seed_uses_unique_outputs_and_provenance(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  module = _collector_module()
  checkpoint = _checkpoint(tmp_path)
  clip = _clip(tmp_path, frames=2)
  output = tmp_path / "multi_seed"
  envs: list[_FakeEnv] = []
  configs: list[Any] = []

  def build_env(cfg: Any, device: str) -> _FakeEnv:
    configs.append(cfg)
    env = _FakeEnv(rows=cfg.scene.num_envs, frames=2)
    envs.append(env)
    return env

  monkeypatch.setattr(module, "ManagerBasedRlEnv", build_env)
  monkeypatch.setattr(module, "RslRlVecEnvWrapper", lambda env, clip_actions=None: env)
  monkeypatch.setattr(module, "load_runner_cls", lambda task: _FakeRunner)
  monkeypatch.setattr(module, "configure_torch_backends", lambda: None)
  _FakeRunner.loaded = []

  config = module.CollectionConfig(
    checkpoint_file=checkpoint,
    motion_file=[clip],
    output_dir=output,
    num_envs=2,
    seed=7,
    num_seeds=3,
  )
  manifest = module.collect(config)

  assert [cfg.seed for cfg in configs] == [7, 8, 9]
  assert manifest["collection"]["num_seeds"] == 3
  assert manifest["collection"]["seeds"] == [7, 8, 9]
  assert manifest["collection"]["num_seed_runs"] == 3
  assert manifest["collection"]["num_clip_rollouts"] == 3
  assert manifest["collection"]["total_trajectories"] == 6
  assert [entry["run_index"] for entry in manifest["clips"]] == [0, 1, 2]
  assert [entry["seed"] for entry in manifest["clips"]] == [7, 8, 9]
  outputs = [entry["output"] for entry in manifest["clips"]]
  assert len(set(outputs)) == 3
  assert outputs == [
    "trajectories/run000_seed007_000_clip_000_agibot_x2_tracking_baseline.npz",
    "trajectories/run001_seed008_000_clip_000_agibot_x2_tracking_baseline.npz",
    "trajectories/run002_seed009_000_clip_000_agibot_x2_tracking_baseline.npz",
  ]
  assert len(list((output / "trajectories").glob("*.npz"))) == 3
  for entry in manifest["clips"]:
    payload = np.load(output / entry["output"])
    try:
      assert payload["trajectory_seed"].tolist() == [entry["seed"]] * 2
      assert payload["trajectory_run_index"].tolist() == [entry["run_index"]] * 2
      assert payload["trajectory_scenario"].tolist() == ["baseline"] * 2
      assert payload["trajectory_initial_frame"].tolist() == [0, 0]
    finally:
      payload.close()
  assert len(envs) == 3 and all(env.closed for env in envs)


def test_collect_disturbed_labels_zero_pushes_truthfully(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  module = _collector_module()
  checkpoint = _checkpoint(tmp_path)
  clip = _clip(tmp_path, frames=2)
  output = tmp_path / "disturbed"

  monkeypatch.setattr(
    module,
    "ManagerBasedRlEnv",
    lambda cfg, device: _FakeEnv(rows=cfg.scene.num_envs, frames=2),
  )
  monkeypatch.setattr(module, "RslRlVecEnvWrapper", lambda env, clip_actions=None: env)
  monkeypatch.setattr(module, "load_runner_cls", lambda task: _FakeRunner)
  monkeypatch.setattr(module, "configure_torch_backends", lambda: None)
  _FakeRunner.loaded = []

  config = module.CollectionConfig(
    checkpoint_file=checkpoint,
    motion_file=[clip],
    output_dir=output,
    num_envs=2,
    seed=3,
    sampling_mode="uniform",
    push_interval_s=(1.0, 2.0),
    push_linear_velocity_range=(-0.3, 0.4),
    push_angular_velocity_range=(-0.6, 0.7),
  )
  manifest = module.collect(config)

  assert manifest["collection"]["scenario"] == "disturbed"
  assert manifest["collection"]["push_robot_event"] is True
  assert manifest["collection"]["sampling_mode"] == "uniform"
  disturbance = manifest["collection"]["disturbance"]
  assert disturbance["kind"] == "simulated_root_velocity_delta"
  assert disturbance["interval_s"] == [1.0, 2.0]
  assert disturbance["velocity_range"]["x"] == [-0.3, 0.4]
  assert disturbance["velocity_range"]["yaw"] == [-0.6, 0.7]
  entry = manifest["clips"][0]
  assert entry["scenario"] == "disturbed"
  assert entry["sampling_mode"] == "uniform"
  assert entry["pushes_observed"] == 0
  assert entry["output"] == (
    "trajectories/run000_seed003_000_clip_000_agibot_x2_tracking_disturbed.npz"
  )
  payload = np.load(output / entry["output"])
  try:
    assert payload["trajectory_scenario"].tolist() == ["disturbed"] * 2
    assert payload["trajectory_push_count"].tolist() == [0, 0]
    assert payload["push_offsets"].tolist() == [0, 0, 0]
    assert payload["push_tick"].size == 0
  finally:
    payload.close()
