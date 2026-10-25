"""Tests for the X2 tennis-ball juggling velocity task.

The juggling task is a copy of the carrying task
(``Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Balance``) with a
different ball reward set, a standstill command band and no command curriculum.
Because it is a copy, the risk is not that a new mechanism is wrong; it is that the
two files drift, so most of these tests compare the juggle config against the
carrying config term by term.

Four claims are tested:

1. **Registration and build.** The task is registered and produces a working
   environment.
2. **The reward diff, and only the reward diff.** ``ball_height`` is gone,
   ``ball_bounce_apex`` and ``ball_at_rest`` are in, ``ball_centred`` keeps its
   function and parameters at the juggle weight, and every other reward term is
   byte-for-byte the carrying task's.
3. **The unchanged contract.** The observation widths are still 105/123 on a built
   environment, and the delayed critic-clean ball terms are still the carrying
   task's.
4. **Juggle behaviour.** On a scripted bounce the apex term pays on exactly one
   step and the rest term reads 0 throughout.

What these tests do **not** show: that a policy can juggle. The behaviour test
drives the reward terms with a scripted ball state, so it measures the wiring and
the detector, not whether a real ball bounces as scripted.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest
import torch

from mjlab.asset_zoo.robots.agibot_x2.tennis_constants import (
  BALL_CENTRE_PLATEAU,
  BALL_CENTRE_SIGMA,
  BALL_RADIUS,
  FACE_SITE_POS,
  JUGGLE_APEX_MIN_CLEARANCE_M,
  JUGGLE_APEX_MIN_SPEED_MPS,
  JUGGLE_APEX_SIGMA_M,
  JUGGLE_APEX_TARGET_M,
  JUGGLE_APEX_WEIGHT,
  JUGGLE_CENTRE_WEIGHT,
  JUGGLE_COMMAND_ANG_VEL_RPS,
  JUGGLE_COMMAND_LIN_VEL_MPS,
  JUGGLE_REST_CLEARANCE_M,
  JUGGLE_REST_GRACE_S,
  JUGGLE_REST_SPEED_MPS,
  JUGGLE_REST_WEIGHT,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.agibot_x2.tennis_juggle_env_cfg import (
  BALL_ENTITY,
)
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import quat_apply

TASK_ID = "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle"
CARRY_TASK_ID = "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Balance"

GRAVITY = 9.81

NEW_BALL_REWARDS = {"ball_bounce_apex", "ball_at_rest"}


def test_task_is_registered() -> None:
  assert TASK_ID in list_tasks()
  # Registering the juggle task must not disturb the carrying task.
  assert CARRY_TASK_ID in list_tasks()


##
# 1. The reward diff, and only the reward diff.
##


def test_the_reward_set_is_the_carry_tasks_with_the_ball_terms_spliced() -> None:
  """One term out, two in; every surviving term is untouched."""
  carry = load_env_cfg(CARRY_TASK_ID)
  juggle = load_env_cfg(TASK_ID)

  assert set(juggle.rewards) == (set(carry.rewards) - {"ball_height"}) | (
    NEW_BALL_REWARDS
  )
  # ``ball_height`` is flat out to a 100 mm gap and pays for a parked ball, so
  # leaving it in would reward exactly the behaviour juggling must rule out.
  assert "ball_height" not in juggle.rewards
  assert juggle.rewards["ball_centred"].weight != carry.rewards["ball_centred"].weight

  for name in set(carry.rewards) - {"ball_height", "ball_centred"}:
    assert juggle.rewards[name].func is carry.rewards[name].func, name
    assert juggle.rewards[name].weight == carry.rewards[name].weight, name
    assert juggle.rewards[name].params == carry.rewards[name].params, name


def test_the_two_new_ball_terms_are_wired_with_the_juggle_constants() -> None:
  juggle = load_env_cfg(TASK_ID)
  centred = juggle.rewards["ball_centred"]

  assert centred.func is mdp.ball_centred
  assert centred.weight == JUGGLE_CENTRE_WEIGHT
  assert centred.params["plateau"] == BALL_CENTRE_PLATEAU
  assert centred.params["sigma"] == BALL_CENTRE_SIGMA

  apex = juggle.rewards["ball_bounce_apex"]
  assert apex.func is mdp.ball_bounce_apex
  assert apex.weight == JUGGLE_APEX_WEIGHT
  # The bounce-count metric reads this term's per-step flag, and RewardManager
  # skips zero-weight terms, so a zero weight here would freeze the count.
  assert apex.weight > 0
  assert apex.params["target_height"] == JUGGLE_APEX_TARGET_M
  assert apex.params["sigma"] == JUGGLE_APEX_SIGMA_M
  assert apex.params["min_clearance"] == JUGGLE_APEX_MIN_CLEARANCE_M
  assert apex.params["min_speed"] == JUGGLE_APEX_MIN_SPEED_MPS

  rest = juggle.rewards["ball_at_rest"]
  assert rest.func is mdp.ball_at_rest
  assert rest.weight == JUGGLE_REST_WEIGHT
  # Negative: this term is the parking penalty, so it must cost.
  assert rest.weight < 0
  assert rest.params["max_clearance"] == JUGGLE_REST_CLEARANCE_M
  assert rest.params["max_speed"] == JUGGLE_REST_SPEED_MPS
  assert rest.params["grace_s"] == JUGGLE_REST_GRACE_S

  # The new terms measure the same plate frame and the same ball as centring.
  for term in (apex, rest):
    assert term.params["plate_cfg"] is centred.params["plate_cfg"]
    assert term.params["ball_cfg"] is centred.params["ball_cfg"]


def test_the_observations_are_the_carry_tasks() -> None:
  """Juggling changes rewards, not the actor's input contract."""
  carry = load_env_cfg(CARRY_TASK_ID)
  juggle = load_env_cfg(TASK_ID)
  for group in ("actor", "critic"):
    terms = juggle.observations[group].terms
    other = carry.observations[group].terms
    assert set(terms) == set(other), group
    for name, term in terms.items():
      assert term.func is other[name].func, (group, name)
      assert term.noise == other[name].noise, (group, name)
      assert (
        term.delay_min_lag,
        term.delay_max_lag,
        term.delay_update_period,
        term.delay_per_env,
        term.delay_per_env_phase,
      ) == (
        other[name].delay_min_lag,
        other[name].delay_max_lag,
        other[name].delay_update_period,
        other[name].delay_per_env,
        other[name].delay_per_env_phase,
      ), (group, name)


##
# 2. Command: one standstill band, and nothing to ramp.
##


def _twist_cfg(cfg) -> UniformVelocityCommandCfg:
  """The ``twist`` command term, narrowed from the generic ``CommandTermCfg``."""
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  return twist


def test_the_twist_command_is_a_standstill_band() -> None:
  cfg = load_env_cfg(TASK_ID)
  ranges = _twist_cfg(cfg).ranges
  assert ranges.lin_vel_x == (-JUGGLE_COMMAND_LIN_VEL_MPS, JUGGLE_COMMAND_LIN_VEL_MPS)
  assert ranges.lin_vel_y == (-JUGGLE_COMMAND_LIN_VEL_MPS, JUGGLE_COMMAND_LIN_VEL_MPS)
  assert ranges.ang_vel_z == (-JUGGLE_COMMAND_ANG_VEL_RPS, JUGGLE_COMMAND_ANG_VEL_RPS)

  # Everything else about the command term is the carrying task's.
  carry = _twist_cfg(load_env_cfg(CARRY_TASK_ID))
  for field in (
    "resampling_time_range",
    "rel_standing_envs",
    "rel_heading_envs",
    "rel_forward_envs",
    "heading_command",
    "heading_control_stiffness",
  ):
    assert getattr(_twist_cfg(cfg), field) == getattr(carry, field), field


def test_the_command_curriculum_is_gone() -> None:
  """A single narrow band has no stage to advance to."""
  cfg = load_env_cfg(TASK_ID)
  assert "ball_command_ramp" not in cfg.curriculum
  # The base task's step-count term would write the base ranges over the band.
  assert "command_vel" not in cfg.curriculum
  assert cfg.curriculum == {}

  # Play mode must not widen the band either: the base play overrides describe a
  # walking policy, which is a different task here.
  play = load_env_cfg(TASK_ID, play=True)
  assert _twist_cfg(play).ranges == _twist_cfg(cfg).ranges
  assert play.curriculum == {}


def test_the_bounce_count_is_a_summed_episode_metric() -> None:
  """``reduce="sum"`` is what turns a per-step flag into bounces per episode."""
  cfg = load_env_cfg(TASK_ID)
  assert "ball_bounce_count" in cfg.metrics
  metric = cfg.metrics["ball_bounce_count"]
  assert metric.func is mdp.ball_bounce_count
  assert metric.reduce == "sum"
  # The base task's metric is kept.
  assert "mean_action_acc" in cfg.metrics


##
# 3. A live environment: build, widths, metrics manager.
##


def _build_env(
  task_id: str = TASK_ID, num_envs: int = 1, steps: int = 1
) -> ManagerBasedRlEnv:
  """A live environment, reset and optionally stepped.

  ``steps=0`` is the pure reset state, which is what the ball-placement helpers
  below expect: after a step the PD gains have begun to sag the arm by a
  seed-dependent amount.
  """
  env = ManagerBasedRlEnv(load_env_cfg(task_id), device="cpu", num_envs=num_envs)
  env.reset()
  action = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
  for _ in range(steps):
    env.step(action)
  return env


@pytest.mark.slow
def test_the_environment_builds_and_steps() -> None:
  env = _build_env(steps=0)
  try:
    ball = env.scene[BALL_ENTITY]
    before = ball.data.root_link_pos_w[0].cpu().numpy().copy()
    for _ in range(3):
      env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim))
    after = ball.data.root_link_pos_w[0].cpu().numpy()
    assert np.all(np.isfinite(after))
    # Nothing should launch the ball: three 20 ms control steps of an unsupported
    # ball is at most ~18 mm of free fall.
    assert np.linalg.norm(after - before) < 0.05
    assert "ball_bounce_count" in env.metrics_manager.active_terms
    assert "ball_bounce_apex" in env.reward_manager.active_terms
    assert "ball_height" not in env.reward_manager.active_terms
  finally:
    env.close()


@pytest.mark.slow
def test_the_observation_widths_are_the_carry_tasks() -> None:
  """Read off a built environment: the term shapes come from the assets."""
  env = _build_env()
  try:
    om = env.observation_manager
    wanted = {"actor": {"ball_pos": 3}, "critic": {"ball_pos": 3, "ball_vel": 3}}
    for group, ball_widths in wanted.items():
      names = om._group_obs_term_names[group]
      dims = om._group_obs_term_dim[group]
      widths = dict(zip(names, (int(np.prod(d)) for d in dims), strict=True))
      assert {name: widths[name] for name in ball_widths} == ball_widths, group
      assert sum(widths.values()) == int(np.prod(om._group_obs_dim[group]))
    # The carrying task's widths: 102 + 3 for the actor, 117 + 6 for the critic.
    observations = cast("dict[str, torch.Tensor]", env.obs_buf)
    assert observations["actor"].shape[-1] == 105
    assert observations["critic"].shape[-1] == 123
  finally:
    env.close()


##
# 4. Behaviour: a scripted bounce, read through the built environment's terms.
##


def _reward_term(env: ManagerBasedRlEnv, name: str):
  """The resolved reward term config, whose ``func`` is the term instance.

  For class-based terms the reward manager has replaced ``func`` by the instance, so
  it is callable as ``func(env, **params)``.
  """
  for term_name, cfg in zip(
    env.reward_manager._term_names,
    env.reward_manager._term_cfgs,
    strict=True,
  ):
    if term_name == name:
      return cfg if not isinstance(cfg, list) else cfg[0]
  raise AssertionError(f"{name} is not configured")


def _racket_body_id(env: ManagerBasedRlEnv) -> int:
  plate_cfg = _reward_term(env, "ball_centred").params["plate_cfg"]
  ids = plate_cfg.body_ids
  assert len(ids) == 1
  return ids[0]


def _place_ball_at_gap(
  env: ManagerBasedRlEnv, gap: float, vertical_speed: float
) -> None:
  """Put the ball ``gap`` above the plate face, moving at ``vertical_speed``.

  Both quantities are what the juggle terms read: the gap under the ball's lowest
  point, measured along the world vertical from the plate's top surface, and the
  ball's world vertical velocity.
  """
  robot = env.scene["robot"]
  body_id = _racket_body_id(env)
  plate_pos = robot.data.body_link_pos_w[:, body_id]
  plate_quat = robot.data.body_link_quat_w[:, body_id]
  local = torch.tensor([FACE_SITE_POS], dtype=plate_pos.dtype, device=plate_pos.device)
  face = plate_pos + quat_apply(plate_quat, local.expand_as(plate_pos))

  ball = env.scene[BALL_ENTITY]
  position = face.clone()
  position[:, 2] += BALL_RADIUS + gap
  ball.write_root_link_pose_to_sim(
    torch.cat([position, ball.data.root_link_quat_w], dim=-1)
  )
  velocity = torch.zeros_like(ball.data.root_link_vel_w)
  velocity[:, 2] = vertical_speed
  ball.write_root_link_velocity_to_sim(velocity)
  env.sim.forward()


def _toss_profile(launch_speed: float, steps: int, step_dt: float = 0.02):
  """``(gap, vertical speed)`` of a ballistic ball, one entry per control step."""
  return [
    (
      launch_speed * step_dt * step - 0.5 * GRAVITY * (step_dt * step) ** 2,
      launch_speed - GRAVITY * step_dt * step,
    )
    for step in range(steps)
  ]


@pytest.mark.slow
def test_a_bounce_pays_on_exactly_one_step_and_parking_is_not_charged() -> None:
  """The behaviour the whole reward set exists for, on the built environment.

  The ball is placed on a ballistic rise and fall, so there is exactly one sign
  change of its world vertical speed and the apex is the top of that arc. The apex
  term must pay there and nowhere else, and the rest term must stay at 0: a ball in
  flight is not a parked ball, even at the bottom of the arc.
  """
  env = _build_env(steps=0)
  try:
    apex = _reward_term(env, "ball_bounce_apex")
    rest = _reward_term(env, "ball_at_rest")
    launch = math.sqrt(2.0 * GRAVITY * JUGGLE_APEX_TARGET_M)

    rewards: list[float] = []
    resting: list[float] = []
    for gap, speed in _toss_profile(launch, steps=40):
      _place_ball_at_gap(env, max(gap, 0.0), speed)
      rewards.append(float(apex.func(env, **apex.params)[0]))
      resting.append(float(rest.func(env, **rest.params)[0]))

    paid = [step for step, reward in enumerate(rewards) if reward > 0.0]
    assert len(paid) == 1, paid
    # The target apex earns full credit; the sampled apex is one 20 ms step past the
    # top of the arc, which is worth 0.999 of it.
    assert rewards[paid[0]] == pytest.approx(1.0, abs=1e-2)
    assert sum(rewards[: paid[0]]) == 0.0, "nothing on the way up"
    assert sum(rewards[paid[0] + 1 :]) == 0.0, "nothing on the way down"

    assert sum(resting) == 0.0, "a ball in flight is not charged as parked"
  finally:
    env.close()


@pytest.mark.slow
def test_a_parked_ball_is_charged_and_earns_no_apex_payment() -> None:
  """The converse: a ball left on the plate must not collect the bounce reward."""
  env = _build_env(steps=0)
  try:
    apex = _reward_term(env, "ball_bounce_apex")
    rest = _reward_term(env, "ball_at_rest")
    grace_steps = round(JUGGLE_REST_GRACE_S / env.step_dt)

    for step in range(grace_steps + 5):
      _place_ball_at_gap(env, 0.0, 0.0)
      assert float(apex.func(env, **apex.params)[0]) == 0.0, step
      charged = float(rest.func(env, **rest.params)[0])
      # The penalty starts only after the grace period, which is what separates a
      # park from the bottom of a real bounce. One call is one control step, so the
      # ``grace_steps``-th consecutive resting step is the first one charged.
      assert charged == (1.0 if step >= grace_steps - 1 else 0.0), step
  finally:
    env.close()


##
# 5. Regression: the carrying task is untouched.
##


@pytest.mark.slow
def test_the_carry_task_still_builds_and_still_has_the_height_reward() -> None:
  cfg = load_env_cfg(CARRY_TASK_ID)
  assert "ball_height" in cfg.rewards
  assert NEW_BALL_REWARDS.isdisjoint(cfg.rewards)
  assert "ball_command_ramp" in cfg.curriculum

  env = _build_env(CARRY_TASK_ID, steps=1)
  try:
    assert "ball_height" in env.reward_manager.active_terms
    ball = env.scene[BALL_ENTITY]
    assert np.all(np.isfinite(ball.data.root_link_pos_w[0].cpu().numpy()))
    observations = cast("dict[str, torch.Tensor]", env.obs_buf)
    assert observations["actor"].shape[-1] == 105
  finally:
    env.close()
