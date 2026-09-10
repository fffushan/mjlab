from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp.commands import (
  MotionCommand,
  MotionLoader,
  load_saved_lookahead_s,
)


def _write_motion(path: Path, fps: float | None = 50.0, frames: int = 5) -> None:
  data: dict[str, np.ndarray] = {
    "joint_pos": np.arange(frames * 2, dtype=np.float32).reshape(frames, 2),
    "joint_vel": (100 + np.arange(frames * 2, dtype=np.float32)).reshape(frames, 2),
    "body_pos_w": np.zeros((frames, 1, 3), dtype=np.float32),
    "body_quat_w": np.tile(
      np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (frames, 1, 1)
    ),
    "body_lin_vel_w": np.zeros((frames, 1, 3), dtype=np.float32),
    "body_ang_vel_w": np.zeros((frames, 1, 3), dtype=np.float32),
  }
  if fps is None:
    np.savez(path, **data)  # type: ignore[invalid-argument-type]
  else:
    np.savez(path, **data, fps=np.array([fps], dtype=np.float32))  # type: ignore[invalid-argument-type]


def test_motion_loader_reads_fps(tmp_path: Path) -> None:
  path = tmp_path / "motion.npz"
  _write_motion(path, fps=60.0)

  loader = MotionLoader(str(path), torch.tensor([0]))

  assert loader.fps == 60.0


def test_motion_loader_legacy_fps_fallback_warns(tmp_path: Path) -> None:
  path = tmp_path / "motion.npz"
  _write_motion(path, fps=None)

  with pytest.warns(UserWarning, match="assuming 50 Hz"):
    loader = MotionLoader(str(path), torch.tensor([0]))

  assert loader.fps == 50.0


def test_motion_loader_rejects_invalid_fps(tmp_path: Path) -> None:
  path = tmp_path / "motion.npz"
  _write_motion(path, fps=0.0)

  with pytest.raises(ValueError, match="non-positive fps"):
    MotionLoader(str(path), torch.tensor([0]))


def _make_lookahead_command(
  path: Path, lookahead_s: float, time_steps: list[int]
) -> MotionCommand:
  command = object.__new__(MotionCommand)
  command_any = cast(Any, command)
  command_any._env = SimpleNamespace(
    device="cpu", num_envs=len(time_steps), step_dt=0.1
  )
  command_any.motion = MotionLoader(str(path), torch.tensor([0]))
  command_any.time_steps = torch.tensor(time_steps, dtype=torch.long)
  command_any.cfg = SimpleNamespace(lookahead_s=lookahead_s)
  command_any.lookahead_steps = max(
    1, int(np.ceil(lookahead_s * command_any.motion.fps))
  )
  return command


def test_motion_lookahead_uses_ceil_and_clamps_to_last_frame(tmp_path: Path) -> None:
  path = tmp_path / "motion.npz"
  _write_motion(path, fps=10.0, frames=5)
  command = _make_lookahead_command(path, lookahead_s=0.11, time_steps=[0, 4])

  # ceil(0.11 * 10) = 2; the second env clamps at frame 4.
  expected = torch.tensor([[4.0, 5.0, 104.0, 105.0], [8.0, 9.0, 108.0, 109.0]])

  torch.testing.assert_close(command.lookahead_command, expected)


def test_motion_lookahead_disabled_is_empty(tmp_path: Path) -> None:
  path = tmp_path / "motion.npz"
  _write_motion(path)
  command = _make_lookahead_command(path, lookahead_s=0.0, time_steps=[0, 1])
  command.lookahead_steps = 0

  assert command.lookahead_command.shape == (2, 0)


def test_tracking_lookahead_is_actor_only() -> None:
  for task_id in (
    "Mjlab-Tracking-Flat-Unitree-G1",
    "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
  ):
    cfg = load_env_cfg(task_id)

    assert "motion_lookahead" in cfg.observations["actor"].terms
    assert "motion_lookahead" not in cfg.observations["critic"].terms


def _write_saved_env_yaml(path: Path, lookahead_s: object = 0.5) -> None:
  """Write a params/env.yaml artifact like the one train.py saves."""
  path.parent.mkdir(parents=True, exist_ok=True)
  text = (
    """commands:
  motion:
    lookahead_s: %s
    pose_range:
      x: !!python/tuple [0.0, 1.0]
observations:
  actor:
    terms:
      motion_lookahead:
        func: !!python/name:mjlab.tasks.tracking.mdp.observations.motion_lookahead ''
"""
    % lookahead_s
  )
  path.write_text(text)


def test_load_saved_lookahead_s(tmp_path: Path) -> None:
  env_yaml = tmp_path / "params" / "env.yaml"
  _write_saved_env_yaml(env_yaml, lookahead_s=0.5)

  # Simulate a checkpoint path whose parent contains params/env.yaml.
  checkpoint_path = env_yaml.parent.parent / "model_2999.pt"

  assert load_saved_lookahead_s(checkpoint_path) == 0.5


def test_load_saved_lookahead_s_missing_returns_none(tmp_path: Path) -> None:
  checkpoint_path = tmp_path / "run" / "model_0.pt"

  assert load_saved_lookahead_s(checkpoint_path) is None


def test_load_saved_lookahead_s_rejects_non_finite(tmp_path: Path) -> None:
  env_yaml = tmp_path / "params" / "env.yaml"
  _write_saved_env_yaml(env_yaml, lookahead_s="nan")
  checkpoint_path = env_yaml.parent.parent / "model_2999.pt"

  with pytest.raises(RuntimeError, match="Non-finite"):
    load_saved_lookahead_s(checkpoint_path)
