"""Standing-start initialization for X2 tracking: configuration and mechanics.

The standing-start insert replaces a fraction of leading-window episode
initializations with the entity's default (standing) pose, so the policy
learns the standing -> motion-first-frame transition that real deployment
performs (the controller holds ``JOINT_DEFAULT`` — the exported policy default
pose — before engaging the policy). See :data:`MotionCommandCfg.standing_start_prob`.
"""

from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.config.agibot_x2.env_cfgs import (
  agibot_x2_flat_tracking_correlated_dr_env_cfg,
  agibot_x2_flat_tracking_standing_start_env_cfg,
)
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.utils.lab_api.math import quat_apply_inverse

TASK = (
  "Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR-"
  "Reduced-Perturbations-Standing-Start"
)
REDUCED_TASK = (
  "Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR-"
  "Reduced-Perturbations"
)
# Any real tennis motion: 359 frames @ 50 fps, bin 0 = frames 0-44 > 25.
MOTION_FILE = Path("data/tennis/single_093_zhanghongyu_agibot_x2_tracking.npz")


def motion_cfg(cfg) -> MotionCommandCfg:
  motion = cfg.commands["motion"]
  assert isinstance(motion, MotionCommandCfg)
  return motion


##
# Configuration contracts.
##


def test_task_is_registered_with_isolated_experiment_directory() -> None:
  assert TASK in list_tasks()
  assert (
    load_rl_cfg(TASK).experiment_name
    == "agibot_x2_tracking_correlated_dr_reduced_perturbations_standing_start"
  )


def test_factory_forks_reduced_perturbations_plus_standing_start() -> None:
  reduced = agibot_x2_flat_tracking_correlated_dr_env_cfg(reduced_perturbations=True)
  standing = agibot_x2_flat_tracking_standing_start_env_cfg()

  assert motion_cfg(standing).standing_start_prob == 0.3
  assert motion_cfg(standing).standing_start_window_frames == 25
  # The fork changes only the insert knob: everything else stays identical.
  assert motion_cfg(reduced).standing_start_prob == 0.0
  assert standing.observations == reduced.observations
  assert standing.actions == reduced.actions
  assert standing.events == reduced.events
  assert standing.rewards == reduced.rewards
  assert standing.terminations == reduced.terminations
  assert standing.sim == reduced.sim
  assert standing.decimation == reduced.decimation
  assert standing.episode_length_s == reduced.episode_length_s


def test_registered_sibling_task_is_untouched() -> None:
  assert motion_cfg(load_env_cfg(REDUCED_TASK)).standing_start_prob == 0.0


def test_play_configuration_never_inserts() -> None:
  play = agibot_x2_flat_tracking_standing_start_env_cfg(play=True)
  assert motion_cfg(play).standing_start_prob == 0.0
  assert motion_cfg(play).sampling_mode == "start"
  assert motion_cfg(load_env_cfg(TASK, play=True)).standing_start_prob == 0.0


##
# Mechanics: real CPU environment with a real tennis motion.
##


def _build_env(prob: float, sampling_mode: str, num_envs: int = 1):
  motion_path = Path(MOTION_FILE)
  if not motion_path.exists():
    pytest.skip(f"{motion_path} not present (generated motion data)")
  cfg = agibot_x2_flat_tracking_standing_start_env_cfg()
  cfg.scene.num_envs = num_envs
  motion = motion_cfg(cfg)
  motion.motion_file = str(motion_path)
  motion.sampling_mode = sampling_mode
  motion.standing_start_prob = prob
  with (
    contextlib.redirect_stdout(io.StringIO()),
    contextlib.redirect_stderr(io.StringIO()),
  ):
    from mjlab.envs import ManagerBasedRlEnv

    env = ManagerBasedRlEnv(cfg, device="cpu")
  return env


@pytest.fixture(scope="module")
def uniform_env():
  env = _build_env(prob=1.0, sampling_mode="uniform", num_envs=4)
  try:
    yield env
  finally:
    env.close()


def _reset_quietly(env) -> None:
  with (
    contextlib.redirect_stdout(io.StringIO()),
    contextlib.redirect_stderr(io.StringIO()),
  ):
    env.reset()


def _soft_clipped(robot, joint_pos: torch.Tensor) -> torch.Tensor:
  """Clip joint positions to the robot's soft limits (what write-to-sim does).

  The tennis reference drives some joints (e.g. shoulder roll) to or below
  their soft limits, so the written state is the clipped reference plus the
  perturbation; comparisons against the raw reference would over-count.
  """
  limits = robot.data.soft_joint_pos_limits[0]
  return torch.clip(joint_pos, limits[:, 0], limits[:, 1])


def _pin_frames_and_resample(env, command, frame: int) -> None:
  """Resample with the frame pinned: neutralize the mode's frame sampler.

  ``_resample_command`` re-samples ``time_steps`` per ``sampling_mode``;
  replacing the sampler on this instance keeps the pinned frame so the
  standing-start eligibility window is exercised deterministically.
  """
  command._uniform_sampling = lambda env_ids: None  # type: ignore[method-assign]
  command.time_steps[:] = frame
  with contextlib.redirect_stdout(io.StringIO()):
    command._resample_command(torch.arange(env.num_envs, device=env.device))
  env.sim.forward()
  command.update_relative_body_poses()


def test_prob_one_inserts_standing_pose_for_window_frames(uniform_env) -> None:
  env = uniform_env
  command = env.command_manager.get_term("motion")
  assert isinstance(command, MotionCommand)

  # All < 25-frame window (bin 0 = frames 0-44 for this motion).
  for frame in (0, 5, 24):
    _reset_quietly(env)
    _pin_frames_and_resample(env, command, frame)

    robot = env.scene["robot"]
    frame_joint_pos = command.motion.joint_pos[frame]
    default_joint_pos = robot.data.default_joint_pos[0]
    written = robot.data.joint_pos[0]

    # Joints are the default (standing) pose + perturbation within the
    # reduced joint_position_range (-0.05, 0.05), not the reference frame.
    delta = (written - default_joint_pos).abs().max().item()
    assert delta <= 0.05 + 1e-5, (frame, delta)
    ref_gap = (written - frame_joint_pos).abs().max().item()
    assert ref_gap > 0.05, (frame, ref_gap)  # not the reference pose

    # Root: upright at default height (allowing z perturbation <= 0.005).
    root = robot.data.default_root_state[0]
    robot_root = robot.data.root_link_pose_w[0]
    assert abs(robot_root[2].item() - root[2].item()) <= 0.005 + 1e-5
    roll, pitch, _ = _quat_roll_pitch(robot.data.root_link_quat_w[0])
    assert abs(roll) <= 0.05 + 1e-5  # roll perturbation bound
    assert abs(pitch) <= 0.05 + 1e-5  # pitch perturbation bound

    # Velocities: perturbation only (reduced ranges xy/z).
    lin_vel = robot.data.root_link_lin_vel_w[0]
    assert lin_vel[0].abs().item() <= 0.25 + 1e-5
    assert lin_vel[2].abs().item() <= 0.1 + 1e-5

    # The motion clock and reference are untouched.
    assert command.time_steps.tolist() == [frame] * env.num_envs
    assert command.metrics["standing_start"].max().item() == 1.0


def test_prob_one_outside_window_keeps_reference_initialization(uniform_env) -> None:
  env = uniform_env
  command = env.command_manager.get_term("motion")
  assert isinstance(command, MotionCommand)

  _reset_quietly(env)
  # Frame 30 lies beyond the 25-frame window (bin 0 is frames 0-44).
  _pin_frames_and_resample(env, command, 30)

  robot = env.scene["robot"]
  written = robot.data.joint_pos[0]
  frame_joint_pos = command.motion.joint_pos[30]
  # Reference pose + perturbation (soft-limit clipped): stays near the
  # reference frame within the perturbation band.
  clipped = _soft_clipped(robot, frame_joint_pos)
  assert (written - clipped).abs().max().item() <= 0.05 + 1e-5
  assert command.metrics["standing_start"].max().item() == 0.0


def test_start_sampling_mode_is_exempt(uniform_env) -> None:
  env = uniform_env
  command = env.command_manager.get_term("motion")
  assert isinstance(command, MotionCommand)

  _reset_quietly(env)
  command.cfg.sampling_mode = "start"
  try:
    _pin_frames_and_resample(env, command, 0)
    # Frame 0, prob 1.0, but "start" mode: reference-frame init, no insert.
    robot = env.scene["robot"]
    written = robot.data.joint_pos[0]
    frame_joint_pos = command.motion.joint_pos[0]
    default_joint_pos = robot.data.default_joint_pos[0]
    clipped = _soft_clipped(robot, frame_joint_pos)
    assert (written - clipped).abs().max().item() <= 0.05 + 1e-5
    assert (written - default_joint_pos).abs().max().item() > 0.05
    assert command.metrics["standing_start"].max().item() == 0.0
  finally:
    command.cfg.sampling_mode = "uniform"


def test_prob_zero_is_reference_initialization(uniform_env) -> None:
  env = uniform_env
  command = env.command_manager.get_term("motion")
  assert isinstance(command, MotionCommand)

  _reset_quietly(env)
  command.cfg.standing_start_prob = 0.0
  try:
    _pin_frames_and_resample(env, command, 0)
    robot = env.scene["robot"]
    written = robot.data.joint_pos[0]
    frame_joint_pos = command.motion.joint_pos[0]
    clipped = _soft_clipped(robot, frame_joint_pos)
    assert (written - clipped).abs().max().item() <= 0.05 + 1e-5
    assert command.metrics["standing_start"].max().item() == 0.0
  finally:
    command.cfg.standing_start_prob = 1.0


def _quat_roll_pitch(quat: torch.Tensor) -> tuple[float, float, float]:
  """Roll/pitch/yaw Euler angles of a single wxyz quaternion."""
  quat = quat / quat.norm(dim=-1, keepdim=True)
  w, x, y, z = quat.unbind(dim=-1)
  roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
  pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
  yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  return roll.item(), pitch.item(), yaw.item()


##
# Window capping: motions whose bin 0 is smaller than the 25-frame window.
##


def test_window_is_capped_at_bin_0_extent(tmp_path: Path) -> None:
  # bin_count = frames // env_steps_per_second + 1, so bin 0 spans at least
  # one second of frames and exceeds the 25-frame window for any motion >= 1 s.
  # The cap binds only for sub-second motions (bin_count = 1, bin 0 = all
  # frames): a 20-frame motion caps the window at 20; a 60-frame motion does
  # not cap (bin_count = 2, bin 0 = 30 >= 25).
  for frames, expected_window in ((20, 20), (60, 25)):
    data = {
      "joint_pos": np.zeros((frames, 2), dtype=np.float32),
      "joint_vel": np.zeros((frames, 2), dtype=np.float32),
      "body_pos_w": np.zeros((frames, 1, 3), dtype=np.float32),
      "body_quat_w": np.tile(
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (frames, 1, 1)
      ),
      "body_lin_vel_w": np.zeros((frames, 1, 3), dtype=np.float32),
      "body_ang_vel_w": np.zeros((frames, 1, 3), dtype=np.float32),
      "fps": np.array([50.0], dtype=np.float32),
    }
    path = tmp_path / f"motion_{frames}.npz"
    np.savez(path, **data)  # type: ignore[invalid-argument-type]

    command = object.__new__(MotionCommand)
    from types import SimpleNamespace
    from typing import Any, cast

    command_any = cast(Any, command)
    command_any._env = SimpleNamespace(device="cpu", num_envs=1, step_dt=0.02)
    from mjlab.tasks.tracking.mdp.commands import MotionLoader

    command_any.motion = MotionLoader(str(path), torch.tensor([0]))
    command_any.cfg = SimpleNamespace(standing_start_window_frames=25)
    bin_count = int(command_any.motion.time_step_total // (1 / 0.02)) + 1
    bin_0 = math.ceil(command_any.motion.time_step_total / bin_count)
    assert min(25, bin_0) == expected_window, (frames, bin_0)


##
# Termination safety: the standing-start initial state must not insta-terminate
# against the reference (integration, real model + real tennis motion).
##


@pytest.mark.slow
def test_standing_start_state_passes_termination_margins(uniform_env) -> None:
  env = uniform_env
  command = env.command_manager.get_term("motion")
  assert isinstance(command, MotionCommand)

  _reset_quietly(env)
  _pin_frames_and_resample(env, command, 0)

  robot = env.scene["robot"]
  body_names = list(robot.body_names)

  # ee_body_pos termination: z of robot bodies vs the anchor-relative
  # reference (threshold 0.25 m), exactly what bad_motion_body_pos_z_only
  # compares. The motion file's world z is in the retarget frame and is not
  # directly comparable; the relative table is the physical quantity.
  ee_names = env.cfg.terminations["ee_body_pos"].params["body_names"]
  ref_relative = command.body_pos_relative_w[0]
  robot_pos = robot.data.body_link_pos_w[0]
  for name in ee_names:
    i = body_names.index(name)
    j = command.cfg.body_names.index(name)
    dz = abs(robot_pos[i, 2].item() - ref_relative[j, 2].item())
    assert dz < 0.25, (name, dz)

  # anchor_pos termination (z-only, 0.25 m): command.anchor_pos_w vs
  # robot_anchor_pos_w (both include env_origins/relative handling).
  dz = abs(
    command.anchor_pos_w[0, -1].item() - command.robot_anchor_pos_w[0, -1].item()
  )
  assert dz < 0.25, dz

  # anchor_ori termination (0.8 rad): projected-gravity z difference, the
  # exact quantity bad_anchor_ori compares.
  gravity_w = env.scene["robot"].data.gravity_vec_w[0]
  motion_pg_z = quat_apply_inverse(command.anchor_quat_w[0], gravity_w)[2].item()
  robot_pg_z = quat_apply_inverse(command.robot_anchor_quat_w[0], gravity_w)[2].item()
  assert abs(motion_pg_z - robot_pg_z) < 0.8, (motion_pg_z, robot_pg_z)
