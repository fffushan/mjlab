"""CPU standing-start mechanics for the mixed distillation command."""

from __future__ import annotations

import math
from typing import Any, cast

import pytest
import torch
from test_tracking_distillation_multi_motion import (
  _FakeEnv,
  make_command,
  make_library,
  make_slots,
  stabilize,
)

from mjlab.tasks.tracking.distillation.environment import SegmentMotionCommand
from mjlab.tasks.tracking.distillation.multi_motion import (
  MultiMotionCommand,
  MultiMotionError,
)
from mjlab.tasks.tracking.distillation.reset_policy import ResetPolicy


def test_standing_reset_uses_anchor_yaw_origin_and_default_pose(tmp_path) -> None:
  library = make_library(tmp_path, clip_frames=(3,))
  slots = make_slots((0,), teacher_ids=("tiny_000",))
  policy = ResetPolicy(
    kind="standing-mixture",
    standing_start_fraction=1.0,
    standing_start_window_frames=2,
    standing_start_frame_zero_fraction=1.0,
  )
  command, env = make_command(
    library,
    slots,
    num_envs=1,
    reset_policy=policy,
  )
  env.scene.env_origins[0] = torch.tensor((4.0, -2.0, 0.3))
  yaw = math.pi / 2
  command.library._body_quat_w[:, 1] = torch.tensor(
    (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
  )

  env.reset()

  assert command.time_steps.tolist() == [0]
  assert command.metrics["standing_start"].tolist() == [1.0]
  assert torch.equal(
    command.robot.data.joint_pos, torch.zeros_like(command.robot.data.joint_pos)
  )
  root = command.robot.data.body_link_pos_w[0, 0]
  assert torch.allclose(root, torch.tensor((4.0, -1.0, 1.0)))
  assert torch.allclose(
    command.robot.data.body_link_quat_w[0, 0],
    torch.tensor((math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))),
    atol=1e-6,
  )


def test_standing_mixture_resets_only_selected_rows(tmp_path) -> None:
  library = make_library(tmp_path, clip_frames=(3,))
  slots = make_slots((0, 0), teacher_ids=("tiny_000",))
  policy = ResetPolicy(
    kind="standing-mixture",
    standing_start_fraction=1.0,
    standing_start_frame_zero_fraction=1.0,
  )
  command, env = make_command(library, slots, num_envs=2, reset_policy=policy)
  command.time_steps[:] = torch.tensor((2, 1))
  command.motion_ids[:] = 0
  before_frame = command.time_steps[0].clone()
  before_position = command.robot.data.body_link_pos_w[0].clone()
  before_generation = command.generation_ids[0].clone()

  env.reset(env_ids=torch.tensor((1,), dtype=torch.long))

  assert command.time_steps[0] == before_frame
  assert torch.equal(command.robot.data.body_link_pos_w[0], before_position)
  assert command.generation_ids[0] == before_generation
  assert command.generation_ids[1] == before_generation + 1
  assert command.time_steps[1] == 0
  assert command.metrics["standing_start"][1] == 1.0


def test_reference_reset_does_not_consume_owned_reset_rng(tmp_path) -> None:
  library = make_library(tmp_path, clip_frames=(2,))
  command, env = make_command(
    library,
    make_slots((0,), teacher_ids=("tiny_000",)),
    num_envs=1,
  )
  before = command.reset_rng_state()

  env.reset()

  assert torch.equal(command.reset_rng_state(), before)


def test_reference_wrap_does_not_apply_standing_mixture(tmp_path) -> None:
  library = make_library(tmp_path, clip_frames=(2,))
  slots = make_slots((0,), teacher_ids=("tiny_000",))
  policy = ResetPolicy(
    kind="standing-mixture",
    standing_start_fraction=1.0,
    standing_start_frame_zero_fraction=1.0,
  )
  command, _ = make_command(library, slots, num_envs=1, reset_policy=policy)
  command.time_steps[:] = 1
  command._wrap_resample_ids = torch.tensor((0,))

  command._resample_command(torch.tensor((0,), dtype=torch.long))

  assert command.metrics["standing_start"].tolist() == [0.0]
  command._wrap_resample_ids = torch.empty(0, dtype=torch.long)


def test_full_reset_pending_frame_skips_once_then_normally_advances(tmp_path) -> None:
  library = make_library(tmp_path, clip_frames=(5,))
  policy = ResetPolicy(
    kind="standing-mixture",
    standing_start_fraction=1.0,
    standing_start_frame_zero_fraction=1.0,
  )
  command, _ = make_command(
    library,
    make_slots((0, 0), teacher_ids=("tiny_000",)),
    num_envs=2,
    reset_policy=policy,
  )
  command.time_steps[:] = torch.tensor((3, 2))
  stabilize(command)
  reset_rows = torch.tensor((0,), dtype=torch.long)
  command.reset(reset_rows)
  assert command.time_steps.tolist() == [0, 2]

  command.compute(dt=0.0, env_ids=torch.arange(2))
  assert command.time_steps.tolist() == [0, 3]

  command.compute(dt=0.0, env_ids=torch.arange(2))
  assert command.time_steps.tolist() == [1, 4]


def test_mixed_reset_fraction_realizes_both_kinds_and_clip_local_frames(
  tmp_path,
) -> None:
  library = make_library(tmp_path, clip_frames=(5,))
  policy = ResetPolicy(
    kind="standing-mixture",
    standing_start_fraction=0.5,
    standing_start_window_frames=2,
    standing_start_frame_zero_fraction=0.0,
  )
  command, env = make_command(
    library,
    make_slots((0,) * 8, teacher_ids=("tiny_000",)),
    num_envs=8,
    reset_policy=policy,
  )
  env.reset()
  standing = command.metrics["standing_start"].to(torch.bool)
  frames = command.time_steps
  assert bool(standing.any()) and bool((~standing).any())
  assert bool(((frames >= 0) & (frames < 5)).all())
  assert bool((frames[standing] < 2).all())
  assert bool((frames[~standing] < 5).all())


def test_disabled_reset_matches_the_pre_standing_reference_path(tmp_path) -> None:
  library = make_library(tmp_path, clip_frames=(5,))
  slots = make_slots((0,), teacher_ids=("tiny_000",))
  current, current_env = make_command(library, slots, num_envs=1)

  # The pre-standing behavior is the inherited SegmentMotionCommand path, so
  # the baseline subclasses the real command (a statically known base, unlike
  # ``type(current)``) and routes to that method explicitly.
  class _BaselineCommand(MultiMotionCommand):
    def _resample_command(self, env_ids: torch.Tensor) -> None:
      SegmentMotionCommand._resample_command(self, env_ids)

  baseline_env = _FakeEnv(1)
  baseline = _BaselineCommand(cast(Any, current.cfg), cast(Any, baseline_env))
  baseline_env.command = baseline
  torch.manual_seed(1234)
  current_env.reset()
  current_frame = current.time_steps.clone()
  current_joint = current.robot.data.joint_pos.clone()
  current_root = current.robot.data.body_link_pos_w.clone()
  current_global_after = torch.get_rng_state()

  torch.manual_seed(1234)
  baseline_env.reset()
  baseline_global_after = torch.get_rng_state()
  assert torch.equal(current_frame, baseline.time_steps)
  assert torch.equal(current_joint, baseline.robot.data.joint_pos)
  assert torch.equal(current_root, baseline.robot.data.body_link_pos_w)
  assert torch.equal(current_global_after, baseline_global_after)


def test_reset_rng_state_round_trip_and_rejection(tmp_path) -> None:
  command, _ = make_command(
    make_library(tmp_path, clip_frames=(3,)),
    make_slots((0,), teacher_ids=("tiny_000",)),
    num_envs=1,
    reset_policy=ResetPolicy(kind="standing-mixture"),
  )
  saved = command.reset_rng_state()
  command._reset_generator.manual_seed(99)
  command.set_reset_rng_state(saved)
  assert torch.equal(command.reset_rng_state(), saved)

  for invalid in (
    saved.to(dtype=torch.int64),
    saved[:-1],
    saved.to(device="meta"),
  ):
    before = command.reset_rng_state()
    with pytest.raises(MultiMotionError, match="reset RNG state"):
      command.set_reset_rng_state(invalid)
    assert torch.equal(command.reset_rng_state(), before)
