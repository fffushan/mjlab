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
   ``ball_energy`` and the negative-weight ``racket_lift`` are in, the per-bounce
   ``ball_bounce_apex`` and ``ball_at_rest`` are gone, ``ball_centred`` keeps its
   function and parameters at the juggle weight, and every other reward term is
   byte-for-byte the carrying task's.
3. **The unchanged contract.** The observation widths are still 105/123 on a built
   environment, and the delayed critic-clean ball terms are still the carrying
   task's.
4. **The ball energy and the lift penalty.** On synthetic states: the spawn is
   0.20 m below the target and scores 0.1807; a settled ball has ``E ~ 0``; the top
   of a bounce of height h has ``E == g * h``, so 0.40 m is exactly the target; the
   Gaussian is symmetric about the target; a crouch cannot change ``E`` or engage
   the lift term; raising the whole robot 0.40 m engages the lift term and
   outweighs the energy reward; the plate's own velocity does **not** change ``E``
   (the racket cannot pay itself); and no partial lift turns the energy-plus-lift
   pair positive. On the built environment: a parked ball earns ~0 and a 0.40 m
   apex earns 1.
5. **The contact metrics.** Synthetic substep contact windows drive the three
   logging-only metrics (impact count, flight time, sustained contact). They show
   that a contact inside one 20 ms control step is counted, that a contact
   spanning several steps is counted once, that flight and contact time resolve
   1 ms substeps, that their per-episode state resets, and that the ``found``
   fallback still works without a force history.

What these tests do **not** show: that a policy can juggle. The formula tests drive
one term with a scripted state and the environment tests read one settled state, so
they measure the wiring and the formula, not whether a trained policy keeps a ball
bouncing.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from mjlab.asset_zoo.robots.agibot_x2.tennis_constants import (
  BALL_CENTRE_PLATEAU,
  BALL_CENTRE_SIGMA,
  BALL_RADIUS,
  BALL_SPAWN_DROP_HEIGHT,
  BALL_SPAWN_INIT_POS,
  BAT_CARRY_BASE_HEIGHT,
  FACE_SITE_POS,
  JUGGLE_CENTRE_WEIGHT,
  JUGGLE_COMMAND_ANG_VEL_RPS,
  JUGGLE_COMMAND_LIN_VEL_MPS,
  JUGGLE_ENERGY_BALL_REST_Z,
  JUGGLE_ENERGY_SIGMA_J_PER_KG,
  JUGGLE_ENERGY_TARGET_HEIGHT_M,
  JUGGLE_ENERGY_WEIGHT,
  JUGGLE_RACKET_LIFT_DEADBAND_M,
  JUGGLE_RACKET_LIFT_PLATE_TOP_REST_Z,
  JUGGLE_RACKET_LIFT_WEIGHT,
  juggle_racket_lift_weight,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.agibot_x2.tennis_juggle_env_cfg import (
  BALL_ENTITY,
  BAT_BALL_CONTACT_SENSOR,
)
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import quat_apply

TASK_ID = "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle"
CARRY_TASK_ID = "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Balance"

GRAVITY = 9.81

PLATE_ENTITY = "robot"

NEW_BALL_REWARDS = {"ball_energy", "racket_lift"}
REMOVED_BALL_REWARDS = {"ball_bounce_apex", "ball_at_rest"}


def test_task_is_registered() -> None:
  assert TASK_ID in list_tasks()
  # Registering the juggle task must not disturb the carrying task.
  assert CARRY_TASK_ID in list_tasks()


##
# 1. The reward diff, and only the reward diff.
##


def test_the_reward_set_is_the_carry_tasks_with_the_ball_terms_spliced() -> None:
  """One term out, one in; every surviving term is untouched."""
  carry = load_env_cfg(CARRY_TASK_ID)
  juggle = load_env_cfg(TASK_ID)

  assert set(juggle.rewards) == (set(carry.rewards) - {"ball_height"}) | (
    NEW_BALL_REWARDS
  )
  # ``ball_height`` is flat out to a 100 mm gap and pays for a parked ball, so
  # leaving it in would reward exactly the behaviour juggling must rule out.
  assert "ball_height" not in juggle.rewards
  # The per-bounce design is replaced by the dense energy term, not extended.
  assert REMOVED_BALL_REWARDS.isdisjoint(juggle.rewards)
  assert juggle.rewards["ball_centred"].weight != carry.rewards["ball_centred"].weight

  for name in set(carry.rewards) - {"ball_height", "ball_centred"}:
    assert juggle.rewards[name].func is carry.rewards[name].func, name
    assert juggle.rewards[name].weight == carry.rewards[name].weight, name
    assert juggle.rewards[name].params == carry.rewards[name].params, name


def test_the_ball_terms_are_wired_with_the_juggle_constants() -> None:
  juggle = load_env_cfg(TASK_ID)
  centred = juggle.rewards["ball_centred"]

  assert centred.func is mdp.ball_centred
  assert centred.weight == JUGGLE_CENTRE_WEIGHT
  assert centred.params["plateau"] == BALL_CENTRE_PLATEAU
  assert centred.params["sigma"] == BALL_CENTRE_SIGMA

  energy = juggle.rewards["ball_energy"]
  assert energy.func is mdp.ball_energy
  assert energy.weight == JUGGLE_ENERGY_WEIGHT
  assert energy.params["target_height"] == JUGGLE_ENERGY_TARGET_HEIGHT_M
  assert energy.params["sigma"] == JUGGLE_ENERGY_SIGMA_J_PER_KG
  assert energy.params["ball_rest_z"] == JUGGLE_ENERGY_BALL_REST_Z

  lift = juggle.rewards["racket_lift"]
  assert lift.func is mdp.racket_lift
  # Negative: the term returns metres of excess lift, so it has to cost.
  assert lift.weight == -JUGGLE_RACKET_LIFT_WEIGHT
  assert lift.weight < 0
  assert lift.params["plate_top_rest_z"] == (JUGGLE_RACKET_LIFT_PLATE_TOP_REST_Z)
  assert lift.params["deadband"] == JUGGLE_RACKET_LIFT_DEADBAND_M

  # The terms measure the same plate frame and the same ball as centring.
  for term in (energy, lift):
    assert term.params["plate_cfg"] is centred.params["plate_cfg"]
  assert energy.params["ball_cfg"] is centred.params["ball_cfg"]


def test_the_lift_weight_is_derived_from_sigma() -> None:
  """The weight follows the maximum-slope rule, so a sigma change moves it."""
  assert JUGGLE_ENERGY_TARGET_HEIGHT_M == pytest.approx(0.40)
  assert JUGGLE_ENERGY_SIGMA_J_PER_KG == 1.5
  assert JUGGLE_RACKET_LIFT_WEIGHT == pytest.approx(
    juggle_racket_lift_weight(sigma=JUGGLE_ENERGY_SIGMA_J_PER_KG)
  )
  # Halving sigma doubles the steepest slope, and so the published weight.
  assert juggle_racket_lift_weight(sigma=0.75) == pytest.approx(
    2.0 * JUGGLE_RACKET_LIFT_WEIGHT
  )
  # 2.0 * sqrt(2) * (9.81 / 1.5) * exp(-0.5). The old rule gave 5.405405.
  assert JUGGLE_RACKET_LIFT_WEIGHT == pytest.approx(11.219551615286045, rel=1e-12)


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


def test_the_energy_metrics_are_means() -> None:
  """``reduce="mean"`` is what turns a per-step quantity into an episode mean."""
  cfg = load_env_cfg(TASK_ID)
  assert set(cfg.metrics) - {"mean_action_acc"} == {
    "ball_energy",
    "ball_energy_error",
    "ball_start_height_m",
    "ball_start_plate",
    "ball_first_lift_s",
    "ball_impact_count",
    "ball_flight_time",
    "ball_sustained_contact",
  }
  energy = cfg.metrics["ball_energy"]
  assert energy.func is mdp.ball_energy_value
  assert energy.reduce == "mean"
  assert energy.params["target_height"] == JUGGLE_ENERGY_TARGET_HEIGHT_M
  assert energy.params["ball_rest_z"] == JUGGLE_ENERGY_BALL_REST_Z
  error = cfg.metrics["ball_energy_error"]
  assert error.func is mdp.ball_energy_error
  assert error.reduce == "mean"
  assert error.params["target_height"] == JUGGLE_ENERGY_TARGET_HEIGHT_M
  assert error.params["ball_rest_z"] == JUGGLE_ENERGY_BALL_REST_Z
  release_metrics = {
    "ball_start_height_m": mdp.ball_start_height,
    "ball_start_plate": mdp.ball_start_plate,
    "ball_first_lift_s": mdp.ball_first_lift_time,
  }
  for name, func in release_metrics.items():
    assert cfg.metrics[name].func is func, name
    assert cfg.metrics[name].reduce == "last", name
  # It read the removed apex term's per-step flag and would freeze without it.
  assert "ball_bounce_count" not in cfg.metrics
  # The base task's metric is kept.
  assert "mean_action_acc" in cfg.metrics


def test_the_contact_metrics_are_logging_only() -> None:
  """The three sensor metrics are read-only signals: nothing else consumes them."""
  cfg = load_env_cfg(TASK_ID)
  expected = {
    "ball_impact_count": ("impact_count", "sum"),
    "ball_flight_time": ("flight_time", "last"),
    "ball_sustained_contact": ("sustained_contact", "max"),
  }
  for name, (mode, reduce) in expected.items():
    term = cfg.metrics[name]
    assert term.func is mdp.ball_contact_metric, name
    assert term.params["sensor_name"] == BAT_BALL_CONTACT_SENSOR, name
    assert term.params["mode"] == mode, name
    assert term.reduce == reduce, name
    # Logging only: no reward, observation, termination or curriculum term.
    assert name not in cfg.rewards, name
    assert name not in cfg.terminations, name
    assert name not in cfg.curriculum, name
    for group in ("actor", "critic"):
      assert name not in cfg.observations[group].terms, (name, group)
  assert BAT_BALL_CONTACT_SENSOR in {s.name for s in cfg.scene.sensors or ()}


def test_only_the_juggle_sensor_keeps_substep_history() -> None:
  """The history buffer is one control step, and the carry task keeps its cost."""
  juggle = load_env_cfg(TASK_ID)
  sensor = next(
    s for s in juggle.scene.sensors or () if s.name == BAT_BALL_CONTACT_SENSOR
  )
  assert sensor.history_length == juggle.decimation
  # The buffer length is the control step over the sim step: 0.02 s / 0.001 s.
  assert sensor.history_length == round(
    juggle.sim.mujoco.timestep * juggle.decimation / juggle.sim.mujoco.timestep
  )

  carry = load_env_cfg(CARRY_TASK_ID)
  carry_sensor = next(
    s for s in carry.scene.sensors or () if s.name == BAT_BALL_CONTACT_SENSOR
  )
  assert carry_sensor.history_length == 0


##
# 3. A live environment: build, widths, metrics manager.
##


def _build_env(
  task_id: str = TASK_ID, num_envs: int = 1, steps: int = 1, perturb_arm: bool = False
) -> ManagerBasedRlEnv:
  """A live environment, reset and optionally stepped.

  ``steps=0`` is the pure reset state, which is what the ball-placement helpers
  below expect: after a step the PD gains have begun to sag the arm by a
  seed-dependent amount.

  ``perturb_arm`` defaults to False, so ``reset_robot_juggle_joints`` keeps the
  carry pose.  The ball energy is measured against a *fixed world* rest height,
  not against the plate, so a perturbed arm moves the plate by up to 16 mm and
  shifts a target-gap energy by about 2 %.  Tests that assert an exact energy
  therefore pin the arm; the release-mixture test asks for it.
  """
  cfg = load_env_cfg(task_id)
  cfg.scene.num_envs = num_envs
  if not perturb_arm and "reset_robot_juggle_joints" in cfg.events:
    cfg.events["reset_robot_juggle_joints"].params["position_range"] = (0.0, 0.0)
  env = ManagerBasedRlEnv(cfg, device="cpu", num_envs=num_envs)
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
    # Nothing should launch the ball. A sampled drop release can start at
    # -g * 0.10 = -0.981 m/s, which is 0.0765 m over 3 x 0.02 s, so 0.09 m leaves
    # margin. The bound still catches a penetration ejection: 22 J/kg is about
    # 6.6 m/s, roughly 0.4 m over the same window.
    assert np.linalg.norm(after - before) < 0.09
    assert {
      "ball_energy",
      "ball_energy_error",
      "ball_impact_count",
      "ball_flight_time",
      "ball_sustained_contact",
    }.issubset(env.metrics_manager.active_terms)
    # The contact metrics need one control step of substeps, not one snapshot:
    # [num_envs, one primary x one slot, history, xyz].
    contact = env.scene[BAT_BALL_CONTACT_SENSOR].data
    assert contact.force_history is not None
    assert contact.force_history.shape == (env.num_envs, 1, env.cfg.decimation, 3)
    assert "ball_bounce_count" not in env.metrics_manager.active_terms
    assert "ball_energy" in env.reward_manager.active_terms
    assert "racket_lift" in env.reward_manager.active_terms
    assert REMOVED_BALL_REWARDS.isdisjoint(env.reward_manager.active_terms)
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
# 4. The ball terms: the formulas on scripted states, then the built environment.
##


def _reward_term(env: ManagerBasedRlEnv, name: str):
  """The resolved reward term config.

  For class-based terms the reward manager has replaced ``func`` by the instance;
  for the plain functions here it is the function itself. Either way it is callable
  as ``func(env, **params)``.
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

  The gap is measured under the ball's lowest point, along the world vertical from
  the plate's top surface, so at the carry pose it is the ball's energy height
  ``h`` plus :data:`BALL_SPAWN_CLEARANCE`.
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


class _StubBallEnv:
  """The pelvis, plate, ball and gravity model the ball terms read.

  Driving the terms directly keeps the formula tests off the physics path: ``h`` is
  the ball's world height above its rest height and ``E`` adds the ball's world
  vertical speed, so a scripted state sets each input exactly. A MuJoCo step cannot
  hold the pelvis and the racket at chosen heights while the ball moves at a chosen
  speed.
  """

  def __init__(self, num_envs: int = 1) -> None:
    self.num_envs = num_envs
    self.device = "cpu"
    self.plate = SimpleNamespace(
      data=SimpleNamespace(
        root_link_pos_w=torch.zeros(num_envs, 3),
        body_link_pos_w=torch.zeros(num_envs, 1, 3),
        body_link_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0])
        .repeat(num_envs, 1, 1)
        .clone(),
        body_link_lin_vel_w=torch.zeros(num_envs, 1, 3),
      )
    )
    self.ball = SimpleNamespace(
      data=SimpleNamespace(
        root_link_pos_w=torch.zeros(num_envs, 3),
        root_link_lin_vel_w=torch.zeros(num_envs, 3),
      )
    )
    self.scene = {PLATE_ENTITY: self.plate, BALL_ENTITY: self.ball}
    # The same storage as a compiled MuJoCo model, negative z included, so the
    # absolute value in the term is exercised rather than assumed.
    self.sim = SimpleNamespace(
      mj_model=SimpleNamespace(
        opt=SimpleNamespace(gravity=np.array([0.0, 0.0, -GRAVITY]))
      )
    )

  def drive(
    self,
    height: float,
    ball_vertical_speed: float = 0.0,
    *,
    plate_height: float | None = None,
    plate_vertical_speed: float = 0.0,
    pelvis_height: float = BAT_CARRY_BASE_HEIGHT,
  ) -> None:
    """Set the ball's energy height, its speed and the plate's world lift.

    ``height`` is ``h``: the ball centre's **world** height above its resting state
    on the plate. The plate top is placed ``plate_height`` above its carry height in
    world, which defaults to ``height``, i.e. the ball rests on the plate. Passing
    ``plate_height`` separately puts the ball in flight relative to the racket.

    ``pelvis_height`` is recorded but read by neither term: that is the point of
    the world datum, and the leg-farm test drives it to prove so.
    """
    self.plate.data.root_link_pos_w[:, 2] = pelvis_height
    self.ball.data.root_link_pos_w[:, 2] = JUGGLE_ENERGY_BALL_REST_Z + height
    self.ball.data.root_link_lin_vel_w[:, 2] = ball_vertical_speed
    lift = height if plate_height is None else plate_height
    plate_top = JUGGLE_RACKET_LIFT_PLATE_TOP_REST_Z + lift
    self.plate.data.body_link_pos_w[:, 0, 2] = plate_top - FACE_SITE_POS[2]
    self.plate.data.body_link_lin_vel_w[:, 0, 2] = plate_vertical_speed


def _energy_state_params() -> dict[str, Any]:
  """The parameters ``ball_energy_value`` and ``ball_energy_error`` take."""
  return dict(
    plate_cfg=SceneEntityCfg(PLATE_ENTITY, body_ids=[0]),
    ball_cfg=SceneEntityCfg(BALL_ENTITY),
    target_height=JUGGLE_ENERGY_TARGET_HEIGHT_M,
    ball_rest_z=JUGGLE_ENERGY_BALL_REST_Z,
  )


def _energy_reward_params() -> dict[str, Any]:
  return dict(_energy_state_params(), sigma=JUGGLE_ENERGY_SIGMA_J_PER_KG)


def _lift_params() -> dict[str, Any]:
  return dict(
    plate_cfg=SceneEntityCfg(PLATE_ENTITY, body_ids=[0]),
    plate_top_rest_z=JUGGLE_RACKET_LIFT_PLATE_TOP_REST_Z,
    deadband=JUGGLE_RACKET_LIFT_DEADBAND_M,
  )


def _model_gravity(env: _StubBallEnv) -> float:
  return abs(float(env.sim.mj_model.opt.gravity[2]))


def test_the_spawn_is_below_the_energy_target() -> None:
  """The drop-in spawn is 0.20 m below the 0.40 m target and scores 0.1807.

  The ball rests at the measured world height ``JUGGLE_ENERGY_BALL_REST_Z``, so the
  spawn's world height minus that reference is the distance the ball was dropped.
  That must equal ``BALL_SPAWN_DROP_HEIGHT`` exactly. It is half the target, so the
  reset state is not ``E*``. At sigma 1.5 J/kg the energy reward at spawn is about
  0.1807, which gives the 0.20 -> 0.40 m climb a gradient.
  """
  assert BALL_SPAWN_INIT_POS[2] - JUGGLE_ENERGY_BALL_REST_Z == pytest.approx(
    BALL_SPAWN_DROP_HEIGHT, abs=1e-9
  )
  assert BALL_SPAWN_DROP_HEIGHT < JUGGLE_ENERGY_TARGET_HEIGHT_M
  assert JUGGLE_ENERGY_SIGMA_J_PER_KG == 1.5
  assert BALL_SPAWN_DROP_HEIGHT == 0.20

  env = cast(Any, _StubBallEnv())
  env.drive(height=BALL_SPAWN_DROP_HEIGHT, plate_height=0.0)
  energy = float(mdp.ball_energy_value(env, **_energy_state_params())[0])
  gravity = _model_gravity(env)
  target = gravity * JUGGLE_ENERGY_TARGET_HEIGHT_M
  assert energy == pytest.approx(gravity * BALL_SPAWN_DROP_HEIGHT)
  deficit = target - energy
  assert deficit == pytest.approx(
    gravity * (JUGGLE_ENERGY_TARGET_HEIGHT_M - BALL_SPAWN_DROP_HEIGHT)
  )
  # At sigma 1.5 the deficit is 1.3 sigma, so the spawn keeps a usable gradient.
  reward = float(mdp.ball_energy(env, **_energy_reward_params())[0])
  assert reward == pytest.approx(
    float(np.exp(-((deficit / JUGGLE_ENERGY_SIGMA_J_PER_KG) ** 2)))
  )
  assert reward == pytest.approx(0.1807, abs=1e-3)


def test_a_settled_ball_has_zero_energy_and_earns_almost_nothing() -> None:
  """A ball resting on a still plate has ``E = 0`` and pays no lift penalty.

  This is the parked-ball case that the removed ``ball_at_rest`` penalty used to
  handle. The energy formula enforces it: no height and no vertical speed means no
  energy, so parking at the carry pose earns nothing.
  """
  env = cast(Any, _StubBallEnv())
  env.drive(height=0.0)
  state = _energy_state_params()

  assert float(mdp.ball_energy_value(env, **state)[0]) == pytest.approx(0.0)
  # exp(-(E* / sigma)**2) = exp(-6.84) = 0.0011 at sigma 1.5; the target is still
  # far from a parked ball.
  assert float(mdp.ball_energy(env, **_energy_reward_params())[0]) < 2e-3
  assert float(mdp.racket_lift(env, **_lift_params())[0]) == 0.0


def test_the_top_of_a_bounce_to_height_h_has_energy_g_times_h() -> None:
  """``E == g * h`` at the top of a bounce, so 0.40 m is exactly the target."""
  env = cast(Any, _StubBallEnv())
  state = _energy_state_params()
  reward_params = _energy_reward_params()
  gravity = _model_gravity(env)

  for height in (0.10, JUGGLE_ENERGY_TARGET_HEIGHT_M, 0.50):
    env.drive(height=height, plate_height=0.0)  # Plate at carry, ball in flight.
    energy = float(mdp.ball_energy_value(env, **state)[0])
    assert energy == pytest.approx(gravity * height, rel=1e-6), height

  env.drive(height=JUGGLE_ENERGY_TARGET_HEIGHT_M, plate_height=0.0)
  assert float(mdp.ball_energy(env, **reward_params)[0]) == pytest.approx(1.0)
  # A bounce 30 cm lower or higher is a different energy and is not full credit.
  for height in (
    JUGGLE_ENERGY_TARGET_HEIGHT_M - 0.30,
    JUGGLE_ENERGY_TARGET_HEIGHT_M + 0.30,
  ):
    env.drive(height=height, plate_height=0.0)
    assert float(mdp.ball_energy(env, **reward_params)[0]) < 0.05, height


def test_the_energy_reward_is_symmetric_about_the_target() -> None:
  """``E* - sigma`` and ``E* + sigma`` J/kg are equally wrong and pay the same."""
  env = cast(Any, _StubBallEnv())
  params = _energy_reward_params()
  gravity = _model_gravity(env)
  target = gravity * JUGGLE_ENERGY_TARGET_HEIGHT_M
  sigma = JUGGLE_ENERGY_SIGMA_J_PER_KG

  env.drive(height=(target - sigma) / gravity, plate_height=0.0)
  below = float(mdp.ball_energy(env, **params)[0])
  env.drive(height=(target + sigma) / gravity, plate_height=0.0)
  above = float(mdp.ball_energy(env, **params)[0])

  # The stub is float32 like the simulator, and reconstructing h subtracts two
  # ~1.15 m world heights, so the two sides can differ by a couple of 1e-6.
  assert below == pytest.approx(above, abs=1e-5)
  # sigma is 1.5 J/kg, so exactly one sigma out is exp(-1) of full credit.
  assert below == pytest.approx(float(np.exp(-1.0)), abs=1e-6)
  assert below < 1.0


def test_a_crouch_cannot_buy_energy() -> None:
  """The leg farm: dropping the pelvis must not move either datum.

  This is the regression the world datum exists for. With the pelvis datum, ``h``
  was ``z_ball - (z_pelvis + K)``, so lowering the pelvis 0.20 m while the ball
  stayed put raised ``h`` by 0.20 m and added ``g * 0.20 = 1.962`` J/kg: the whole
  old 0.20 m target, and half of the current 3.924 J/kg target, with no work done on
  the ball. The lift term could not see it, because the plate and the pelvis dropped
  together and its excess stayed zero. That paid for leg motion.

  Held fixed in world: the ball. The pelvis drops 0.20 m and the arm holds its
  carry pose, so the plate drops with it. ``E``, ``r_E`` and the lift term must all
  stay where they were.
  """
  env = cast(Any, _StubBallEnv())
  state = _energy_state_params()
  reward_params = _energy_reward_params()
  lift_params = _lift_params()
  crouch = 0.20

  env.drive(height=0.0, plate_height=0.0, pelvis_height=BAT_CARRY_BASE_HEIGHT)
  standing_energy = float(mdp.ball_energy_value(env, **state)[0])
  standing_reward = float(mdp.ball_energy(env, **reward_params)[0])
  standing_lift = float(mdp.racket_lift(env, **lift_params)[0])
  # The pelvis drops; the arm holds its carry pose, so the plate drops with it; the
  # ball does not move in world.
  env.drive(
    height=0.0,
    plate_height=-crouch,
    pelvis_height=BAT_CARRY_BASE_HEIGHT - crouch,
  )
  crouched_energy = float(mdp.ball_energy_value(env, **state)[0])
  crouched_reward = float(mdp.ball_energy(env, **reward_params)[0])
  crouched_lift = float(mdp.racket_lift(env, **lift_params)[0])

  assert crouched_energy == pytest.approx(standing_energy, abs=1e-9)
  assert crouched_reward == pytest.approx(standing_reward, abs=1e-9)
  assert crouched_lift == standing_lift == 0.0
  # The old pelvis datum would have read h = 0.20 m, the old target, and paid
  # JUGGLE_ENERGY_WEIGHT with the lift term still at zero.
  assert crouched_energy < 0.05
  assert crouched_reward < 2e-3


def test_the_plate_vertical_velocity_does_not_enter_the_energy() -> None:
  """The anti-cheat test: the racket must not be able to pay itself.

  A racket dropping away under a resting ball used to add
  ``0.5 * (0 - (-1.0))**2 = 0.5`` J/kg to the plate-relative formula, 13 % of the
  target for free. The energy term reads the ball's world vertical velocity only, so
  the plate's velocity cannot change it, and the racket's motion only counts once it
  has actually moved the ball. The +1.0 case is ball and plate moving together with
  the same ``v_z`` at an unchanged ball world height.
  """
  env = cast(Any, _StubBallEnv())
  state = _energy_state_params()
  reward_params = _energy_reward_params()

  env.drive(
    height=0.10,
    plate_height=0.0,
    ball_vertical_speed=1.0,
    plate_vertical_speed=0.0,
  )
  still = float(mdp.ball_energy_value(env, **state)[0])
  still_reward = float(mdp.ball_energy(env, **reward_params)[0])
  for plate_speed in (-1.0, 1.0):
    env.drive(
      height=0.10,
      plate_height=0.0,
      ball_vertical_speed=1.0,
      plate_vertical_speed=plate_speed,
    )
    assert float(mdp.ball_energy_value(env, **state)[0]) == pytest.approx(
      still, rel=1e-6
    ), plate_speed
    assert float(mdp.ball_energy(env, **reward_params)[0]) == pytest.approx(
      still_reward, rel=1e-6
    ), plate_speed


def test_lifting_the_resting_ball_to_the_target_is_not_profitable() -> None:
  """At the target the maximum-slope penalty outweighs the energy reward.

  A ball resting on the plate has ``h`` equal to the plate's lift, so a 0.40 m lift
  makes ``r_E = 1.0`` on every step with no bounce. The lift weight is at least as
  steep as the energy Gaussian, so the pair goes negative well before the target and
  is strongly negative there: parking cannot be farmed.
  """
  env = cast(Any, _StubBallEnv())
  env.drive(
    height=JUGGLE_ENERGY_TARGET_HEIGHT_M,
    plate_height=JUGGLE_ENERGY_TARGET_HEIGHT_M,
  )
  energy_reward = float(mdp.ball_energy(env, **_energy_reward_params())[0])
  lift = float(mdp.racket_lift(env, **_lift_params())[0])

  assert energy_reward == pytest.approx(1.0)
  assert lift == pytest.approx(
    JUGGLE_ENERGY_TARGET_HEIGHT_M - JUGGLE_RACKET_LIFT_DEADBAND_M
  )
  net = JUGGLE_ENERGY_WEIGHT * energy_reward - JUGGLE_RACKET_LIFT_WEIGHT * lift
  assert net == pytest.approx(-2.15, abs=0.01)
  assert net < 0.0


def test_no_partial_lift_turns_the_energy_pair_positive() -> None:
  """Sample the lift range: the best net is the deadband edge, at most +0.006."""
  env = cast(Any, _StubBallEnv())
  lifts = np.linspace(0.0, 0.45, 4501)
  net = np.zeros(len(lifts), dtype=float)
  for index, lift in enumerate(lifts):
    env.drive(height=lift, plate_height=lift)
    energy_reward = float(mdp.ball_energy(env, **_energy_reward_params())[0])
    penalty = float(mdp.racket_lift(env, **_lift_params())[0])
    net[index] = (
      JUGGLE_ENERGY_WEIGHT * energy_reward - JUGGLE_RACKET_LIFT_WEIGHT * penalty
    )

  # The best lift is the free edge of the deadband, not a partial lift near the
  # target. The residual is the deadband's free energy, about +0.0057.
  step = lifts[1] - lifts[0]
  assert lifts[int(np.argmax(net))] == pytest.approx(
    JUGGLE_RACKET_LIFT_DEADBAND_M, abs=step
  )
  assert net.max() > 0.0
  assert net.max() <= 0.006

  # At the target, where the old rule cancelled exactly, the pair is negative.
  env.drive(
    height=JUGGLE_ENERGY_TARGET_HEIGHT_M,
    plate_height=JUGGLE_ENERGY_TARGET_HEIGHT_M,
  )
  at_target = JUGGLE_ENERGY_WEIGHT * float(
    mdp.ball_energy(env, **_energy_reward_params())[0]
  ) - JUGGLE_RACKET_LIFT_WEIGHT * float(mdp.racket_lift(env, **_lift_params())[0])
  assert at_target < -1.0


def test_raising_the_whole_robot_is_charged_like_an_arm_lift() -> None:
  """A 0.40 m world lift is charged whether the legs or the arm do it.

  Raising the whole robot 0.40 m with the arm at the carry pose lifts the plate
  0.40 m in world and carries the resting ball with it, so ``h = 0.40`` and the
  energy term pays exactly 1.0. The lift term measures the same world height, so it
  charges the same 0.37 m of excess: the pair is strongly negative either way.
  """
  env = cast(Any, _StubBallEnv())
  rise = JUGGLE_ENERGY_TARGET_HEIGHT_M
  env.drive(
    height=rise,
    plate_height=rise,
    pelvis_height=BAT_CARRY_BASE_HEIGHT + rise,
  )
  plate_top_z = float(env.plate.data.body_link_pos_w[0, 0, 2] + FACE_SITE_POS[2])
  assert plate_top_z - JUGGLE_RACKET_LIFT_PLATE_TOP_REST_Z == pytest.approx(rise)

  energy_reward = float(mdp.ball_energy(env, **_energy_reward_params())[0])
  lift = float(mdp.racket_lift(env, **_lift_params())[0])
  assert energy_reward == pytest.approx(1.0)
  assert lift == pytest.approx(rise - JUGGLE_RACKET_LIFT_DEADBAND_M)
  net = JUGGLE_ENERGY_WEIGHT * energy_reward - JUGGLE_RACKET_LIFT_WEIGHT * lift
  assert net < 0.0


def test_a_higher_lift_is_punished_rather_than_merely_neutralised() -> None:
  """Above the target the penalty wins, so lifting the ball higher loses reward."""
  env = cast(Any, _StubBallEnv())
  above = JUGGLE_ENERGY_TARGET_HEIGHT_M + 0.10
  env.drive(height=above, plate_height=above)
  energy_reward = float(mdp.ball_energy(env, **_energy_reward_params())[0])
  lift = float(mdp.racket_lift(env, **_lift_params())[0])

  net = JUGGLE_ENERGY_WEIGHT * energy_reward - JUGGLE_RACKET_LIFT_WEIGHT * lift
  assert net < 0.0


def test_the_lift_penalty_is_zero_at_or_below_the_carry_height() -> None:
  """The deadband leaves normal jostling free, and a downward move is not a lift."""
  env = cast(Any, _StubBallEnv())
  params = _lift_params()

  for lift in (0.0, -0.10, JUGGLE_RACKET_LIFT_DEADBAND_M * 0.5):
    env.drive(height=lift, plate_height=lift)
    assert float(mdp.racket_lift(env, **params)[0]) == 0.0, lift


@pytest.mark.slow
def test_a_parked_ball_earns_almost_no_energy_reward() -> None:
  """A ball left on the plate must not collect the energy reward.

  The parked-ball scenario the removed ``ball_at_rest`` penalty existed for. The
  formula now enforces it on the built environment: a settled ball has ``h ~ 0`` and
  no vertical speed, so ``E ~ 0`` and the reward is ~0.
  """
  env = _build_env(steps=0)
  try:
    term = _reward_term(env, "ball_energy")
    _place_ball_at_gap(env, 0.0, 0.0)
    energy = float(
      mdp.ball_energy_value(
        env,
        plate_cfg=term.params["plate_cfg"],
        ball_cfg=term.params["ball_cfg"],
        target_height=term.params["target_height"],
        ball_rest_z=term.params["ball_rest_z"],
      )[0]
    )
    assert energy < 0.05, (
      f"E = {energy} J/kg against a {GRAVITY * JUGGLE_ENERGY_TARGET_HEIGHT_M:.2f} "
      "target"
    )
    assert float(term.func(env, **term.params)[0]) < 2e-3
    # The plate is at its carry height at reset, so the lift penalty is zero.
    assert float(mdp.racket_lift(env, **_lift_params())[0]) == 0.0
  finally:
    env.close()


@pytest.mark.slow
def test_a_bounce_to_the_target_height_earns_full_credit() -> None:
  """The top of a target-height bounce is the state the reward is designed to pay."""
  env = _build_env(steps=0)
  try:
    term = _reward_term(env, "ball_energy")
    # A target gap is h = 0.40 minus the 1 mm spawn clearance. At the peak the
    # reward is flat, so that is worth 1.0.
    _place_ball_at_gap(env, JUGGLE_ENERGY_TARGET_HEIGHT_M, 0.0)
    assert float(term.func(env, **term.params)[0]) == pytest.approx(1.0, abs=1e-2)
    # Thirty centimetres lower is a different energy and is not full credit.
    _place_ball_at_gap(env, JUGGLE_ENERGY_TARGET_HEIGHT_M - 0.30, 0.0)
    assert float(term.func(env, **term.params)[0]) < 0.05
  finally:
    env.close()


##
# 5. The contact metrics: synthetic contact sequences, no physics.
##


_HISTORY_LENGTH = 20
_SUBSTEP_DT = 0.001
_STEP_DT = 0.02


class _StubContactEnv:
  """The sensor fields and durations ``ball_contact_metric`` reads.

  With ``history_length > 0`` the stub holds one control step of substeps, the
  shape the real sensor produces when it is configured with the decimation.
  With ``history_length == 0`` ``force_history`` is None and the metric must use
  the instantaneous ``found`` field.
  """

  def __init__(self, history_length: int = _HISTORY_LENGTH) -> None:
    self.num_envs = 1
    self.device = "cpu"
    self.step_dt = _STEP_DT
    self.physics_dt = _SUBSTEP_DT
    force_history = torch.zeros(1, 1, history_length, 3) if history_length > 0 else None
    self.sensor = SimpleNamespace(
      cfg=SimpleNamespace(history_length=history_length),
      data=SimpleNamespace(
        found=torch.zeros(1, 1, dtype=torch.long),
        force_history=force_history,
      ),
    )
    self.scene = {BAT_BALL_CONTACT_SENSOR: self.sensor}

  @property
  def history_length(self) -> int:
    return self.sensor.cfg.history_length

  def set_found(self, found: int) -> None:
    """Write the instantaneous fallback field."""
    self.sensor.data.found = torch.tensor([[found]], dtype=torch.long)

  def set_contact(self, contact: list[bool]) -> None:
    """Write one control step of substeps, oldest substep first."""
    force_history = self.sensor.data.force_history
    assert force_history is not None
    assert len(contact) == force_history.shape[2]
    force_history.zero_()
    for i, in_contact in enumerate(contact):
      if in_contact:
        # Index 0 of force_history is the most recent substep.
        force_history[0, 0, force_history.shape[2] - 1 - i, 0] = 1.0


def _contact_term(env: _StubContactEnv, mode: str) -> tuple[Any, MetricsTermCfg]:
  """A resolved ``ball_contact_metric`` instance and the config that drives it."""
  cfg = MetricsTermCfg(
    func=mdp.ball_contact_metric,
    params={"sensor_name": BAT_BALL_CONTACT_SENSOR, "mode": mode},
  )
  term = mdp.ball_contact_metric(cfg=cfg, env=cast(Any, env))
  return term, cfg


def _run_contact_windows(
  term: Any,
  env: _StubContactEnv,
  cfg: MetricsTermCfg,
  windows: list[list[bool]],
) -> list[float]:
  """Drive one control step per window, each window oldest substep first."""
  values = []
  for window in windows:
    env.set_contact(window)
    values.append(float(term(env, **cfg.params)[0]))
  return values


def _run_contact_sequence(
  term: Any, env: _StubContactEnv, cfg: MetricsTermCfg, contact: list[int]
) -> list[float]:
  """Drive whole control steps: a 1 is contact on every substep of the step."""
  windows = [[bool(flag)] * env.history_length for flag in contact]
  return _run_contact_windows(term, env, cfg, windows)


def test_the_impact_count_counts_distinct_contacts() -> None:
  """One rising edge per impact: ``0,0,1,1,1,0,0,1,1`` is two impacts."""
  env = _StubContactEnv()
  term, cfg = _contact_term(env, "impact_count")
  values = _run_contact_sequence(term, env, cfg, [0, 0, 1, 1, 1, 0, 0, 1, 1])

  assert values == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
  assert sum(values) == 2.0


def test_a_contact_inside_one_control_step_is_one_impact() -> None:
  """The regression: a 9 ms contact is invisible to one ``found`` sample."""
  env = _StubContactEnv()
  term, cfg = _contact_term(env, "impact_count")

  # Nine substeps of contact in the middle of a 20-substep control step.
  window = [False] * _HISTORY_LENGTH
  for i in range(5, 14):
    window[i] = True
  values = _run_contact_windows(
    term,
    env,
    cfg,
    [[False] * _HISTORY_LENGTH, window, [False] * _HISTORY_LENGTH],
  )

  assert values == [0.0, 1.0, 0.0]


def test_a_contact_spanning_three_control_steps_is_one_impact() -> None:
  """A continuous contact that crosses two step boundaries counts once."""
  env = _StubContactEnv()
  term, cfg = _contact_term(env, "impact_count")
  first = [False] * 15 + [True] * 5
  middle = [True] * _HISTORY_LENGTH
  last = [True] * 5 + [False] * 15
  values = _run_contact_windows(
    term,
    env,
    cfg,
    [
      [False] * _HISTORY_LENGTH,
      first,
      middle,
      last,
      [False] * _HISTORY_LENGTH,
    ],
  )

  assert values == [0.0, 1.0, 0.0, 0.0, 0.0]
  assert sum(values) == 1.0


def test_two_contacts_inside_one_control_step_are_one_impact() -> None:
  """Documented limit: the impact flag resolves control steps, not substeps.

  A bounce wholly inside one 20 ms step is one impact in ``impact_count``,
  because the count is a rising edge of the per-step flag. ``flight_time``
  still measures the airborne gap between the two contacts in substeps.
  """
  env = _StubContactEnv()
  impact, impact_cfg = _contact_term(env, "impact_count")
  flight, flight_cfg = _contact_term(env, "flight_time")

  window = [False] * _HISTORY_LENGTH
  for i in (2, 3, 4, 8, 9, 10):
    window[i] = True
  contact_window = [True] * _HISTORY_LENGTH
  windows = [contact_window, window]

  assert sum(_run_contact_windows(impact, env, impact_cfg, windows)) == 1.0
  # Two flights close: 2 substeps after the first contact, then the 3 substeps
  # between the two contacts in the final window. The mean is 2.5 ms.
  flight_values = _run_contact_windows(flight, env, flight_cfg, windows)
  assert flight_values[-1] == pytest.approx((2 + 3) / 2 * _SUBSTEP_DT)


def test_the_flight_time_excludes_contact_substeps() -> None:
  """Flights of 40 and 60 substeps give 0.04 s and a 0.05 s mean."""
  env = _StubContactEnv()
  term, cfg = _contact_term(env, "flight_time")
  values = _run_contact_sequence(term, env, cfg, [0, 0, 1, 0, 0, 0, 1])

  # The first flight closed on the contact step after two airborne steps.
  assert values[2] == pytest.approx(0.04)
  # The second flight is three steps, so the running mean is (0.04 + 0.06) / 2.
  assert values[-1] == pytest.approx(0.05)
  # If contact substeps were counted as airborne, the flights would be longer.
  assert values[-1] != pytest.approx(0.07)


def test_the_sustained_contact_reports_the_longest_contact() -> None:
  """Two contacts of two and four steps give a maximum of four steps."""
  env = _StubContactEnv()
  term, cfg = _contact_term(env, "sustained_contact")
  values = _run_contact_sequence(term, env, cfg, [1, 1, 0, 1, 1, 1, 1])

  assert values[1] == pytest.approx(2 * _STEP_DT)
  assert max(values) == pytest.approx(4 * _STEP_DT)


def test_flight_and_sustained_contact_resolve_substeps() -> None:
  """Both metrics count 1 ms substeps, including across a step boundary."""
  env = _StubContactEnv()
  sustained, sustained_cfg = _contact_term(env, "sustained_contact")
  # A contact of 15 substeps: the last 5 of one step and the first 10 of the
  # next. The step-based answer would be one step (20 ms); the truth is 15 ms.
  sustained_values = _run_contact_windows(
    sustained,
    env,
    sustained_cfg,
    [
      [False] * _HISTORY_LENGTH,
      [False] * 15 + [True] * 5,
      [True] * 10 + [False] * 10,
    ],
  )
  assert max(sustained_values) == pytest.approx(15 * _SUBSTEP_DT)

  # A flight of 20 substeps that spans two control steps: contact at the start
  # of one step, then 15 airborne substeps, then 5 more before the next contact.
  # The step-based flight would be zero, because both steps contain a contact.
  flight, flight_cfg = _contact_term(env, "flight_time")
  flight_values = _run_contact_windows(
    flight,
    env,
    flight_cfg,
    [
      [True] * _HISTORY_LENGTH,
      [True] * 5 + [False] * 15,
      [False] * 5 + [True] * 10 + [False] * 5,
    ],
  )
  assert flight_values[-1] == pytest.approx(20 * _SUBSTEP_DT)


def test_the_found_field_is_the_fallback_without_force_history() -> None:
  """With ``history_length == 0`` the pre-history per-step behaviour is kept."""
  env = _StubContactEnv(history_length=0)
  assert env.sensor.data.force_history is None
  impact, impact_cfg = _contact_term(env, "impact_count")
  flight, flight_cfg = _contact_term(env, "flight_time")
  sustained, sustained_cfg = _contact_term(env, "sustained_contact")

  def step(term: Any, cfg: MetricsTermCfg, found: int) -> float:
    env.set_found(found)
    return float(term(env, **cfg.params)[0])

  assert [step(impact, impact_cfg, f) for f in (0, 0, 1, 1, 0, 1)] == [
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    1.0,
  ]
  # Two airborne control steps close a 40 ms flight at the step resolution.
  assert step(flight, flight_cfg, 0) == 0.0
  assert step(flight, flight_cfg, 0) == 0.0
  assert step(flight, flight_cfg, 1) == pytest.approx(2 * _STEP_DT)
  # Contact time is counted in control steps when there is no history buffer.
  assert step(sustained, sustained_cfg, 1) == pytest.approx(_STEP_DT)
  assert step(sustained, sustained_cfg, 1) == pytest.approx(2 * _STEP_DT)


def test_the_contact_metrics_reset_their_episode_state() -> None:
  """A new episode starts with no carried-over count, flight or contact time."""
  env = _StubContactEnv()
  impact, impact_cfg = _contact_term(env, "impact_count")
  flight, flight_cfg = _contact_term(env, "flight_time")
  sustained, sustained_cfg = _contact_term(env, "sustained_contact")

  contact = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1]
  assert sum(_run_contact_sequence(impact, env, impact_cfg, contact)) == 2.0
  assert _run_contact_sequence(flight, env, flight_cfg, contact)[-1] == pytest.approx(
    0.06
  )
  assert max(_run_contact_sequence(sustained, env, sustained_cfg, contact)) == (
    pytest.approx(4 * _STEP_DT)
  )

  for term in (impact, flight, sustained):
    term.reset(env_ids=torch.tensor([0]))

  # A contact on the first substep of the new episode is a new impact, not
  # carried state; it has no flight before it; and its contact time is one step.
  env.set_contact([True] * _HISTORY_LENGTH)
  assert float(impact(env, **impact_cfg.params)[0]) == 1.0
  assert float(flight(env, **flight_cfg.params)[0]) == 0.0
  assert float(sustained(env, **sustained_cfg.params)[0]) == pytest.approx(_STEP_DT)

  # A fresh flight is measured against the new episode only.
  for window in (
    [False] * _HISTORY_LENGTH,
    [False] * _HISTORY_LENGTH,
    [True] * _HISTORY_LENGTH,
  ):
    env.set_contact(window)
    flight_value = float(flight(env, **flight_cfg.params)[0])
  assert flight_value == pytest.approx(2 * _STEP_DT)


##
# 6. Regression: the carrying task is untouched.
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


def test_the_release_mixture_survives_a_partial_reset() -> None:
  """Both release branches are reachable, near the configured fraction, and a
  subset reset works.

  The subset case is the one that broke training: the plate branch read the site
  pose for every environment while writing only the reset subset, so the first
  step where any episode ended raised a size mismatch.  Resetting an explicit
  subset reproduces that without stepping.
  """
  env = _build_env(num_envs=256, steps=0, perturb_arm=True)
  try:
    term = env.event_manager.get_term_cfg("reset_ball").func
    plate = (term.start_plate > 0.5).cpu()
    heights = term.start_height.cpu()
    share = float(plate.float().mean())
    assert 0.12 < share < 0.28, f"plate share {share:.3f} is not near the 0.20 fraction"
    # Flags and sampled heights agree: a plate start is at rest, a drop start is not.
    assert float(heights[plate].abs().max()) == 0.0
    assert float(heights[~plate].min()) > 0.0

    subset = torch.tensor([0, 2, 5, 7, 11], dtype=torch.long)
    env.reset(env_ids=subset)  # Must not raise.
    assert torch.equal(term.start_plate[subset].cpu() >= 0.0, torch.ones(5, dtype=torch.bool))
  finally:
    env.close()
