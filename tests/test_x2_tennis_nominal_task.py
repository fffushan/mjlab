"""Tests for the nominal juggling pilot task (``...-Ball3-Pelvis-Nominal``).

The nominal task is the pelvis-history juggling task with the randomization axes
the pilot fixes switched off. It is built by wrapping that task, so the risk is
that the wrapper misses an axis or that it changes the shipped task. Four claims
are tested:

1. **The resolved nominal settings.** No randomization event survives, the actor
   is uncorrupted, the history term is noise-free, the resets are fixed, the
   command is zero and every term the pilot does not touch is the pelvis task's.
2. **The old defaults.** The pelvis task still has its randomization, its
   sampled resets and the shipped stream magnitudes, and the history term still
   defaults to :data:`BALL_POSITION_NOISE_M`.
3. **A zero-noise history serves exact positions.** Driven on a scripted state,
   so the claim is about the term and not about the physics.
4. **The nominal reset places a centred 0.20 m drop.** Read off a built
   environment.

What these tests do **not** show: that a policy learns anything under these
settings. No training run is made here.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from test_x2_tennis_pelvis_ball_history import _StubPelvisEnv

from mjlab.asset_zoo.robots.agibot_x2.tennis_constants import (
  BALL_POSITION_BIAS_M,
  BALL_POSITION_NOISE_M,
  BALL_STREAM_LATENCY_RANGE_S,
  BALL_STREAM_RATE_RANGE_HZ,
  BALL_STREAM_STALL_PROB,
  HEAD_JOINT_NAMES,
  JUGGLE_ARM_RESET_RANGE_RAD,
  ball_spawn_init_pos,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.agibot_x2.tennis_juggle_env_cfg import BALL_ENTITY
from mjlab.tasks.velocity.config.agibot_x2.tennis_juggle_nominal_env_cfg import (
  NOMINAL_REMOVED_EVENTS,
  agibot_x2_tennis_juggle_nominal_env_cfg,
)
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg

TASK_ID = "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle"
PELVIS_TASK_ID = f"{TASK_ID}-29DOF-Ball3-Pelvis"
NOMINAL_TASK_ID = f"{PELVIS_TASK_ID}-Nominal"

NOMINAL_RESET_EVENTS = {
  "reset_base",
  "reset_robot_joints",
  "reset_head_pose",
  "reset_robot_juggle_joints",
  "reset_ball",
}
"""Every event the nominal task has left: robot, head, arm and ball resets."""

PELVIS_BODY = 0


##
# 1. The resolved nominal settings.
##


def test_the_nominal_task_keeps_only_the_resets_and_the_pelvis_tasks_terms() -> None:
  assert NOMINAL_TASK_ID in list_tasks()
  cfg = load_env_cfg(NOMINAL_TASK_ID)
  pelvis = load_env_cfg(PELVIS_TASK_ID)

  # Randomization is removed, not zeroed: the terms write sampled values, so an
  # installed term with a zero range would still run the writes. The set equality
  # also catches a randomization term added under a name this config does not list.
  assert set(cfg.events) == NOMINAL_RESET_EVENTS
  assert set(NOMINAL_REMOVED_EVENTS).isdisjoint(cfg.events)

  # Clean actor input. The history term's own noise is a separate setting, below.
  assert cfg.observations["actor"].enable_corruption is False
  assert cfg.observations["critic"] == pelvis.observations["critic"]

  assert cfg.events["reset_head_pose"].params["position_range"] == (0.0, 0.0)
  assert cfg.events["reset_robot_juggle_joints"].params["position_range"] == (
    0.0,
    0.0,
  )
  ball_params = cfg.events["reset_ball"].params
  assert ball_params["plate_start_fraction"] == 0.0
  assert ball_params["drop_height_range"] == (0.20, 0.20)
  assert ball_params["drop_extra_time_s"] == (0.0, 0.0)
  assert ball_params["drop_lateral_range_m"] == 0.005
  assert ball_params["drop_lateral_speed_range_mps"] == 0.0

  # Zero command. ``rel_forward_envs`` has to go too: the command term lifts a
  # forward environment's lin_vel_x to at least 0.3 m/s after sampling the ranges.
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  assert twist.rel_forward_envs == 0.0
  assert twist.ranges.lin_vel_x == (0.0, 0.0)
  assert twist.ranges.lin_vel_y == (0.0, 0.0)
  assert twist.ranges.ang_vel_z == (0.0, 0.0)

  # Everything the pilot does not touch is the pelvis task's, term for term.
  assert cfg.rewards == pelvis.rewards
  assert cfg.terminations == pelvis.terminations
  assert cfg.metrics == pelvis.metrics
  assert cfg.actions == pelvis.actions
  actor = cfg.observations["actor"]
  pelvis_actor = pelvis.observations["actor"]
  history = actor.terms["ball_hist_pelvis"]
  assert history.func is mdp.ball_history_samples
  assert history.params["reference_frame"] == "pelvis"
  assert history.params["pelvis_cfg"].body_names == ("pelvis",)
  assert set(actor.terms) == set(pelvis_actor.terms)
  for name in set(actor.terms) - {"ball_hist_pelvis"}:
    assert actor.terms[name] == pelvis_actor.terms[name], name


def test_the_nominal_history_is_noise_free_and_keeps_three_samples() -> None:
  cfg = load_env_cfg(NOMINAL_TASK_ID)
  term = cfg.observations["actor"].terms["ball_hist_pelvis"]
  assert term.params["bias_half_width_m"] == 0.0
  assert term.params["position_noise_m"] == (0.0, 0.0, 0.0)
  # One acquisition per 20 ms control step, no latency and no stalls, so the
  # three samples are three consecutive control steps. The term applies its own
  # delay, so the manager must still add none.
  assert term.params["stream_rate_hz"] == (50.0, 50.0)
  assert term.params["stream_latency_s"] == (0.0, 0.0)
  assert term.params["stall_prob"] == 0.0
  assert term.params["stall_range_s"] == (0.0, 0.0)
  assert term.noise is None
  assert term.delay_max_lag == 0


def test_the_energy_target_override_reaches_the_reward_and_both_metrics(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """The two pilot arms share one target value, and nothing else selects it."""
  for target in (0.20, 0.40):
    monkeypatch.setenv("MJLAB_JUGGLE_ENERGY_TARGET_HEIGHT_M", str(target))
    cfg = agibot_x2_tennis_juggle_nominal_env_cfg()
    assert cfg.rewards["ball_energy"].params["target_height"] == target
    assert cfg.metrics["ball_energy"].params["target_height"] == target
    assert cfg.metrics["ball_energy_error"].params["target_height"] == target


##
# 2. The old defaults.
##


def test_the_pelvis_task_keeps_its_randomization_and_stream() -> None:
  """Wrapping must not have moved the shipped task's defaults."""
  cfg = load_env_cfg(PELVIS_TASK_ID)
  for name in NOMINAL_REMOVED_EVENTS:
    assert name in cfg.events, name
  assert cfg.observations["actor"].enable_corruption is True

  params = cfg.observations["actor"].terms["ball_hist_pelvis"].params
  # The new noise parameter is not recorded here, so the term's own default is
  # what it uses. Same for the reset sampler's new arguments.
  assert "position_noise_m" not in params
  assert params["bias_half_width_m"] == BALL_POSITION_BIAS_M
  assert params["stream_rate_hz"] == BALL_STREAM_RATE_RANGE_HZ
  assert params["stream_latency_s"] == BALL_STREAM_LATENCY_RANGE_S
  assert params["stall_prob"] == BALL_STREAM_STALL_PROB

  assert cfg.events["reset_robot_juggle_joints"].params["position_range"] == (
    -JUGGLE_ARM_RESET_RANGE_RAD,
    JUGGLE_ARM_RESET_RANGE_RAD,
  )
  assert cfg.events["reset_head_pose"].params["position_range"] != (0.0, 0.0)
  ball_params = cfg.events["reset_ball"].params
  for name in (
    "plate_start_fraction",
    "drop_height_range",
    "drop_extra_time_s",
    "drop_lateral_range_m",
    "drop_lateral_speed_range_mps",
  ):
    assert name not in ball_params, name


def test_the_history_term_defaults_to_the_shipped_position_noise() -> None:
  env = cast(Any, _StubPelvisEnv())
  params = {
    "pelvis_cfg": SceneEntityCfg("robot", body_ids=[PELVIS_BODY]),
    "ball_cfg": SceneEntityCfg(BALL_ENTITY),
    "reference_frame": "pelvis",
  }
  term = mdp.ball_history_samples(
    ObservationTermCfg(func=mdp.ball_history_samples, params=params), env
  )
  assert torch.equal(
    term._noise_half, torch.tensor(BALL_POSITION_NOISE_M, dtype=torch.float32)
  )


##
# 3. A zero-noise history serves exact positions.
##


def test_a_zero_noise_history_serves_the_exact_position() -> None:
  """With noise, bias and latency off, the served sample is the true position.

  The bound is float32 equality: a shipped 5-8 mm noise model, or a bias, or a
  latency, would move the sample well outside it.
  """
  env = cast(Any, _StubPelvisEnv())
  params: dict[str, Any] = {
    "pelvis_cfg": SceneEntityCfg("robot", body_ids=[PELVIS_BODY]),
    "ball_cfg": SceneEntityCfg(BALL_ENTITY),
    "reference_frame": "pelvis",
    "bias_half_width_m": 0.0,
    "position_noise_m": (0.0, 0.0, 0.0),
    "stream_rate_hz": (50.0, 50.0),
    "stream_latency_s": (0.0, 0.0),
    "stall_prob": 0.0,
    "stall_range_s": (0.0, 0.0),
  }
  term = mdp.ball_history_samples(
    ObservationTermCfg(func=mdp.ball_history_samples, params=params), env
  )
  env.place(
    pelvis_pos=torch.tensor([[0.0, 0.0, 0.8]]),
    ball_pos=torch.tensor([[0.4, 0.2, 1.05]]),
  )
  out = None
  for _ in range(3):
    env.episode_length_buf += 1
    out = term(env, **params)
  assert out is not None
  # Newest sample: R^T (ball - pelvis), no offset and no draw. Float dust only: the
  # shipped noise, bias and latency are millimetre and millisecond scale.
  assert torch.allclose(out[:, :3], torch.tensor([[0.4, 0.2, 0.25]]), atol=1e-6)
  # Three consecutive control steps, newest first, so the ages are 0, 1 and 2 dt.
  assert torch.allclose(out[0, 3:12:4], torch.tensor([0.0, 0.02, 0.04]), atol=1e-6)


##
# 4. The nominal reset on a live environment.
##


def test_the_nominal_reset_places_a_centred_drop_at_rest() -> None:
  cfg = load_env_cfg(NOMINAL_TASK_ID)
  num_envs = 16
  cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg, device="cpu", num_envs=num_envs)
  try:
    env.reset()
    term = env.event_manager.get_term_cfg("reset_ball").func
    # No plate starts, and every drop is released from 0.20 m with no fall time.
    assert float(term.start_plate.max()) == 0.0
    assert torch.allclose(
      term.start_height, torch.full_like(term.start_height, 0.20), atol=1e-6
    )

    ball = env.scene[BALL_ENTITY]
    anchor = torch.tensor(ball_spawn_init_pos(0.20)).expand(num_envs, 3)
    offset = ball.data.root_link_pos_w - env.scene.env_origins - anchor
    assert float(offset[:, :2].abs().max()) <= 0.005 + 1e-6, "lateral jitter bound"
    assert float(offset[:, 2].abs().max()) < 1e-4, "release height"
    # At rest: the release imparts no speed at all.
    assert float(ball.data.root_link_lin_vel_w.abs().max()) == 0.0

    # Fixed robot pose: head, right arm and waist sit at the default pose.
    robot = env.scene["robot"]
    defaults = robot.data.default_joint_pos
    assert defaults is not None
    names = list(robot.joint_names)
    ids = [names.index(name) for name in HEAD_JOINT_NAMES]
    ids += [
      names.index(name)
      for name in (
        "right_shoulder_pitch_joint",
        "right_elbow_joint",
        "waist_yaw_joint",
        "waist_pitch_joint",
        "waist_roll_joint",
      )
    ]
    assert float((robot.data.joint_pos[:, ids] - defaults[:, ids]).abs().max()) < 1e-6

    # The zero command survives resampling, and the layout is the pelvis task's.
    observations = cast("dict[str, torch.Tensor]", env.obs_buf)
    assert observations["actor"].shape[-1] == 112
    assert observations["critic"].shape[-1] == 121
    assert env.action_manager.total_action_dim == 29
    action = torch.zeros(num_envs, env.action_manager.total_action_dim)
    for _ in range(2):
      env.step(action)
    command = env.command_manager.get_command("twist")
    assert command is not None
    assert float(command.abs().max()) == 0.0
  finally:
    env.close()
