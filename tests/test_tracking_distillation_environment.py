"""CPU tests for segment-aware motion command bookkeeping."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
import pytest
import torch

from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.distillation.adapter import validate_live_contract
from mjlab.tasks.tracking.distillation.config import load_manifest, resolve_cohort
from mjlab.tasks.tracking.distillation.environment import (
  SegmentMotionCommand,
  build_distillation_environment,
)


def make_command(num_envs: int = 3) -> SegmentMotionCommand:
  command = object.__new__(SegmentMotionCommand)
  command.generation_ids = torch.zeros(num_envs, dtype=torch.long)
  command.segment_ids = torch.zeros(num_envs, dtype=torch.long)
  return command


def test_every_explicit_boundary_advances_generation_and_segment() -> None:
  command = make_command()
  command._mark_boundary(torch.tensor([0, 2]))
  assert command.generation_ids.tolist() == [1, 0, 1]
  assert command.segment_ids.tolist() == [1, 0, 1]

  command._mark_boundary(torch.tensor([2]))
  assert command.generation_ids.tolist() == [1, 0, 2]
  assert command.segment_ids.tolist() == [1, 0, 2]


@pytest.fixture(scope="module")
def real_cpu_environment():
  cohort = resolve_cohort(
    load_manifest("configs/distillation/x2_tennis.yaml", repo_root=Path("."))
  )
  output = io.StringIO()
  with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
    env = build_distillation_environment(cohort, "tennis_000", num_envs=1, device="cpu")
  try:
    yield env
  finally:
    env.close()


def test_real_routes_mark_reset_timer_wrap_and_frame_teleport(
  real_cpu_environment,
) -> None:
  fresh_cfg = load_env_cfg(
    "Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR-Reduced-Perturbations"
  )
  assert fresh_cfg.observations["actor"].terms["joint_pos"].delay_update_period == 1
  assert fresh_cfg.observations["actor"].terms["joint_pos"].history_length == 0
  env = real_cpu_environment
  assert (
    env.observation_manager.get_term_cfg("actor", "joint_pos").delay_update_period == 1
  )
  assert env.observation_manager.get_term_cfg("actor", "joint_pos").history_length == 0
  command = env.command_manager.get_term("motion")
  assert isinstance(command, SegmentMotionCommand)

  with (
    contextlib.redirect_stdout(io.StringIO()),
    contextlib.redirect_stderr(io.StringIO()),
  ):
    env.reset(seed=7)
  initial = command.generation_ids.clone()

  with (
    contextlib.redirect_stdout(io.StringIO()),
    contextlib.redirect_stderr(io.StringIO()),
  ):
    env.reset(env_ids=torch.tensor([0], device=env.device))
  assert command.generation_ids[0] > initial[0]
  reset_events = command.consume_boundary_events()
  assert reset_events.interrupted.tolist() == [True]
  assert "reset" in reset_events.reasons[0]

  before_timer = command.generation_ids.clone()
  command.time_left[:] = -1.0
  with (
    contextlib.redirect_stdout(io.StringIO()),
    contextlib.redirect_stderr(io.StringIO()),
  ):
    env.command_manager.compute(dt=env.step_dt)
  assert command.generation_ids[0] > before_timer[0]
  timer_events = command.consume_boundary_events()
  assert timer_events.interrupted.tolist() == [True]
  assert "timer_resampled" in timer_events.reasons[0]

  before_wrap = command.generation_ids.clone()
  command.time_steps[:] = command.motion.time_step_total - 1
  with (
    contextlib.redirect_stdout(io.StringIO()),
    contextlib.redirect_stderr(io.StringIO()),
  ):
    command._update_command(None)
  assert command.generation_ids[0] > before_wrap[0]
  wrap_events = command.consume_boundary_events()
  assert wrap_events.completed.tolist() == [True]
  assert wrap_events.completed_generation.tolist() == [before_wrap[0].item()]

  before_frame = command.generation_ids.clone()
  command.reset_to_frame(torch.tensor([0], device=env.device), 0)
  assert command.generation_ids[0] > before_frame[0]

  events = command.consume_boundary_events()
  assert events.available.tolist() == [True]
  assert events.interrupted.tolist() == [True]
  assert events.reasons[0].startswith("reset_to_frame")


def test_timer_then_new_generation_wrap_only_completes_new_generation(
  real_cpu_environment,
) -> None:
  env = real_cpu_environment
  command = env.command_manager.get_term("motion")
  assert isinstance(command, SegmentMotionCommand)
  env.reset(seed=8)
  command.consume_boundary_events()
  pre_generation = command.generation_ids.clone()
  command.time_left[:] = -1.0
  command.time_steps[:] = command.motion.time_step_total - 1
  command.compute(dt=env.step_dt)
  timer_events = command.consume_boundary_events(pre_generation)
  assert timer_events.interrupted.tolist() == [True]
  assert timer_events.completed.tolist() == [False]
  new_generation = command.generation_ids.clone()
  command.time_steps[:] = command.motion.time_step_total - 1
  command._update_command(None)
  events = command.consume_boundary_events(new_generation)
  assert events.interrupted.tolist() == [False]
  assert events.completed.tolist() == [True]
  assert events.interrupted_generation.tolist() == [-1]
  assert events.completed_generation.tolist() == new_generation.tolist()
  assert "reference_completed" in events.reasons[0]


def test_same_compute_timer_resample_and_wrap_classifies_each_resample_once(
  real_cpu_environment, monkeypatch: pytest.MonkeyPatch
) -> None:
  env = real_cpu_environment
  command = env.command_manager.get_term("motion")
  assert isinstance(command, SegmentMotionCommand)
  env.reset(seed=11)
  command.consume_boundary_events()

  def sample_last_frame(env_ids: torch.Tensor) -> None:
    command.time_steps[env_ids] = command.motion.time_step_total - 1

  # A uniform resample can land on the clip's last frame; force that outcome so
  # the same compute also wraps the frame it just sampled.
  monkeypatch.setattr(command.cfg, "sampling_mode", "uniform")
  monkeypatch.setattr(command, "_uniform_sampling", sample_last_frame)

  pre_generation = command.generation_ids.clone()
  command.time_left[:] = -1.0
  command.compute(dt=env.step_dt)

  # Two resamples really happened (timer expiry, then the wrap of the frame it
  # sampled), so exactly two generations advance; the wrap must not be re-tagged
  # as a second timer resample of a generation no env sampled.
  assert command.generation_ids.tolist() == (pre_generation + 2).tolist()
  assert command.segment_ids.tolist() == (pre_generation + 2).tolist()
  events = command.consume_boundary_events(pre_generation)
  assert events.available.tolist() == [True]
  assert events.interrupted.tolist() == [True]
  assert events.completed.tolist() == [True]
  assert events.interrupted_generation.tolist() == pre_generation.tolist()
  assert events.completed_generation.tolist() == (pre_generation + 1).tolist()
  assert events.post_generation.tolist() == (pre_generation + 2).tolist()
  assert events.reasons == ("timer_resampled+reference_completed",)


def test_factory_seed_is_applied_before_construction_and_recorded() -> None:
  cohort = resolve_cohort(
    load_manifest("configs/distillation/x2_tennis.yaml", repo_root=Path("."))
  )
  output = io.StringIO()
  with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
    env = build_distillation_environment(
      cohort, "tennis_000", num_envs=1, device="cpu", seed=123
    )
  try:
    assert env.cfg.seed == 123
    provenance = env.cfg.__dict__["_distillation_seed_provenance"]
    assert provenance.requested_seed == 123
    assert provenance.effective_seed == 123
    assert provenance.applied_before_construction is True
    audit = validate_live_contract(env, cohort)
    assert audit.seed_provenance == provenance
    first_body_mass = env.sim.mj_model.body_mass.copy()
  finally:
    env.close()

  with (
    contextlib.redirect_stdout(io.StringIO()),
    contextlib.redirect_stderr(io.StringIO()),
  ):
    same_seed_env = build_distillation_environment(
      cohort, "tennis_000", num_envs=1, device="cpu", seed=123
    )
  try:
    np.testing.assert_allclose(first_body_mass, same_seed_env.sim.mj_model.body_mass)
  finally:
    same_seed_env.close()

  fresh = load_env_cfg(
    "Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR-Reduced-Perturbations"
  )
  assert fresh.seed is None
  with pytest.raises(ValueError, match="seed"):
    build_distillation_environment(
      cohort, "tennis_000", num_envs=1, device="cpu", seed=True
    )


def test_empty_boundary_does_not_mutate_bookkeeping() -> None:
  command = make_command()
  command._mark_boundary(torch.empty(0, dtype=torch.long))
  assert torch.equal(command.generation_ids, torch.zeros(3, dtype=torch.long))
  assert torch.equal(command.segment_ids, torch.zeros(3, dtype=torch.long))
