"""Tests for motion-tracking body reward functions."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from mjlab.tasks.tracking.mdp.rewards import (
  motion_global_body_angular_velocity_error_exp,
  motion_global_body_linear_velocity_error_exp,
  motion_relative_body_orientation_error_exp,
  motion_relative_body_position_error_exp,
)

BODY_NAMES = (
  "pelvis",
  "torso_link",
  "left_elbow_link",
  "left_wrist_pitch_link",
  "right_wrist_pitch_link",
)


@pytest.fixture
def mock_env():
  """Mock env whose motion command holds per-body reference/robot states."""
  num_envs, num_bodies = 2, len(BODY_NAMES)
  command = Mock()
  command.cfg.body_names = BODY_NAMES
  command.body_pos_relative_w = torch.zeros(num_envs, num_bodies, 3)
  command.robot_body_pos_w = torch.zeros(num_envs, num_bodies, 3)
  command.body_quat_relative_w = torch.zeros(num_envs, num_bodies, 4)
  command.robot_body_quat_w = torch.zeros(num_envs, num_bodies, 4)
  command.body_quat_relative_w[:, :, 0] = 1.0
  command.robot_body_quat_w[:, :, 0] = 1.0
  command.body_lin_vel_w = torch.zeros(num_envs, num_bodies, 3)
  command.robot_body_lin_vel_w = torch.zeros(num_envs, num_bodies, 3)
  command.body_ang_vel_w = torch.zeros(num_envs, num_bodies, 3)
  command.robot_body_ang_vel_w = torch.zeros(num_envs, num_bodies, 3)
  env = Mock()
  env.command_manager.get_term.return_value = command
  return env, command


def _set_error_on_body(command, body_name: str) -> int:
  """Perturb exactly one tracked body in every reference channel."""
  num_envs, num_bodies = command.body_pos_relative_w.shape[:2]
  idx = BODY_NAMES.index(body_name)
  command.body_pos_relative_w[:, idx, 0] = 1.0
  command.robot_body_pos_w[:, idx, 0] = 0.0
  command.body_lin_vel_w[:, idx, 0] = 1.0
  command.robot_body_lin_vel_w[:, idx, 0] = 0.0
  command.body_ang_vel_w[:, idx, 0] = 1.0
  command.robot_body_ang_vel_w[:, idx, 0] = 0.0
  command.body_quat_relative_w[:, idx] = torch.tensor([0.0, 1.0, 0.0, 0.0])
  command.robot_body_quat_w[:, idx] = torch.tensor([1.0, 0.0, 0.0, 0.0])
  return idx


@pytest.mark.parametrize(
  "reward_fn",
  [
    motion_relative_body_position_error_exp,
    motion_relative_body_orientation_error_exp,
    motion_global_body_linear_velocity_error_exp,
    motion_global_body_angular_velocity_error_exp,
  ],
)
def test_body_weights_scale_error_contribution(mock_env, reward_fn):
  """Down-weighting the only erroneous body scales the reward to base**w.

  With a single erroneous body the reward is exp(-w*E/(B*std**2)) where E is
  that body's error and w its weight, so the weighted reward equals the
  unweighted reward raised to the body's weight: r_w = r_1 ** w.
  """
  env, command = mock_env
  num_bodies = len(BODY_NAMES)
  weight_idx = _set_error_on_body(command, "left_wrist_pitch_link")
  secondary = 0.3

  weighted = [1.0] * num_bodies
  weighted[weight_idx] = secondary

  kwargs = {"env": env, "command_name": "motion", "std": 0.3}
  base = reward_fn(body_names=None, body_weights=None, **kwargs)
  secondary_only = reward_fn(body_names=None, body_weights=tuple(weighted), **kwargs)

  assert torch.allclose(secondary_only, base**secondary, atol=1e-5)
  # Scaling every body uniformly is equivalent to only scaling the one that
  # actually carries the error.
  assert torch.allclose(
    reward_fn(
      body_names=None,
      body_weights=(secondary,) * num_bodies,
      **kwargs,
    ),
    secondary_only,
    atol=1e-5,
  )


def test_body_weights_must_align_with_body_names(mock_env):
  """Mismatched weight length must fail loudly instead of silently misaligning."""
  env, _ = mock_env
  with pytest.raises(ValueError, match="align with command.body_names"):
    motion_relative_body_position_error_exp(
      env=env,
      command_name="motion",
      std=0.3,
      body_weights=(1.0, 1.0),
    )
