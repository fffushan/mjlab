"""Probe of the target-height juggle family: S1 and S1a.

Section 7 of ``docs/design/x2_tennis_target_height_plan.md`` asks for seven checks.
This script runs them on CPU and prints the measured numbers:

1. widths: actor 113, critic 122, actions 29, and nothing but the target machinery
   differs from the shipped ``s1-15hz`` stage;
2. the command is per environment, in the observation and in the reward;
3. at a 0.40 m target the reward equals the **old constant-0.40 code path** at the
   same ball state;
4. the reward peaks at the commanded apex. Checked from physics: a ball at rest at
   the commanded height scores 1.0, and two states with the same mechanical energy
   (one at rest, one moving) score the same. No production formula is recomputed;
5. the sigma and the lift threshold scale with the target. Measured behaviourally:
   the reward's 1/e half-width, and the energy at which ``ball_not_lifted`` and
   ``ball_first_lift_s`` change verdict, are bisected rather than read from code;
6. the spawn scales: 256 resets give a mean release height of ``0.5 * h``;
7. the target survives a reset: resampled on reset, constant inside an episode, and
   the fresh value is the one that episode's spawn was built from.

Checks 3, 4 and 5 drive a stub environment, because a MuJoCo step cannot hold the
ball at a chosen height and vertical speed while the reward reads it. The stub
carries the task's own parameters, taken from the built config, so the values under
test are the ones the task runs. It stores state in float64 so a bisection is not
limited by the float32 resolution of the simulator's positions.

Run it with no arguments::

  uv run python scripts/experiments/x2_target_height/probe_target_height.py

The script exits non-zero on the first failed check. It does not train, does not
write a run directory and does not need a GPU.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from mjlab.asset_zoo.robots.agibot_x2.tennis_constants import (
  JUGGLE_ENERGY_BALL_REST_Z,
  JUGGLE_ENERGY_SIGMA_J_PER_M,
  JUGGLE_ENERGY_TARGET_HEIGHT_M,
  JUGGLE_LIFT_ENERGY_J_PER_M,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.agibot_x2.tennis_juggle_stage_env_cfg import (
  agibot_x2_tennis_juggle_stage_env_cfg,
)
from mjlab.tasks.velocity.config.agibot_x2.tennis_juggle_target_height_env_cfg import (
  S1_TARGET_HEIGHT_RANGE_M,
  S1A_TARGET_HEIGHT_RANGE_M,
  STAGE_TARGET_HEIGHT_RANGE_M,
  TARGET_HEIGHT_RANGE_ENV,
  agibot_x2_tennis_juggle_target_height_env_cfg,
)
from mjlab.tasks.velocity.mdp.target_height_command import (
  TARGET_HEIGHT_COMMAND,
  commanded_target_height,
)

S1_TASK = (
  "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
  "-Ball3-Pelvis-Nominal-S1-TargetHeight"
)
S1A_TASK = (
  "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
  "-Ball3-Pelvis-Nominal-S1a-TargetHeight"
)

NUM_ENVS = 8
"""Environments per build. CPU only, so small; every check is per-environment."""

SPAWN_SAMPLES = 256
"""Reset samples per spawn-height check, as section 7 requires."""

SPAWN_RESETS = SPAWN_SAMPLES // NUM_ENVS

DRIFT_STEPS = 20
"""Control steps (0.4 s) an episode is advanced to test that the target holds."""

PROBE_SEED = 0
"""Environment seed, so a report can be compared with an earlier one.

The checks are per-environment relations, not statistics, so the seed only fixes
which target each environment draws; every assertion holds for any seed.
"""

CHECKED_HEIGHTS_M = (0.05, 0.20, 0.40)
"""The three targets section 7 checks the sigma and the threshold at."""

GRAVITY = 9.81
"""Gravity of the stub model, in m/s^2. The real model's value is read by the terms."""

EXPECTED_ACTOR_WIDTH = 113
EXPECTED_CRITIC_WIDTH = 122
EXPECTED_ACTIONS = 29
LADDER_ACTOR_WIDTH = 112
LADDER_CRITIC_WIDTH = 121

STUB_DTYPE = torch.float64
"""Working precision of the stub, unless a check asks for the shipped float32.

The environment's command and ball state are float32. The stub defaults to float64
so a bisection is not limited by the position resolution of the simulator; check 3
runs both, because the no-op at 0.40 m is a statement about float arithmetic."""


class ProbeFailure(AssertionError):
  """A check measured something other than the plan requires."""


def _require(condition: Any, message: str) -> None:
  if not bool(condition):
    raise ProbeFailure(message)


class _StubCommandManager:
  """The command-manager surface the terms use: one named target tensor."""

  def __init__(self, command: torch.Tensor) -> None:
    self._command = command

  def get_command(self, name: str) -> torch.Tensor | None:
    if name != TARGET_HEIGHT_COMMAND:
      return None
    return self._command

  def set_heights(self, heights: torch.Tensor) -> None:
    self._command[:, 0] = heights


class _StubBallEnv:
  """The ball, the gravity model and the command the ball terms read.

  A scripted state sets the ball's height above its resting state and its vertical
  speed exactly, which is what checks 3, 4 and 5 need: the reward peaks at the
  commanded apex only if height and speed are independent inputs.
  """

  def __init__(self, heights: torch.Tensor) -> None:
    num_envs = len(heights)
    self.num_envs = num_envs
    self.device = "cpu"
    self.step_dt = 0.02
    self.dtype = heights.dtype
    self.command_manager = _StubCommandManager(heights.reshape(num_envs, 1).clone())
    self.ball = SimpleNamespace(
      data=SimpleNamespace(
        root_link_pos_w=torch.zeros(num_envs, 3, dtype=heights.dtype),
        root_link_lin_vel_w=torch.zeros(num_envs, 3, dtype=heights.dtype),
      )
    )
    self.scene = {"ball": self.ball}
    # The same storage as a compiled model, negative z included, so the absolute
    # value in the terms is exercised rather than assumed.
    self.sim = SimpleNamespace(
      mj_model=SimpleNamespace(
        opt=SimpleNamespace(gravity=np.array([0.0, 0.0, -GRAVITY]))
      )
    )

  def drive(self, height: float, vertical_speed: float = 0.0) -> None:
    """Put every environment's ball at ``height`` above rest, moving at ``speed``."""
    self.ball.data.root_link_pos_w[:, 2] = JUGGLE_ENERGY_BALL_REST_Z + height
    self.ball.data.root_link_lin_vel_w[:, 2] = vertical_speed


def _stub(heights: list[float], dtype: torch.dtype = STUB_DTYPE) -> _StubBallEnv:
  return _StubBallEnv(torch.tensor(heights, dtype=dtype))


def _build_env(stage: str) -> tuple[ManagerBasedRlEnv, Any]:
  """Build and reset one environment of the target-height family.

  The factory is called directly rather than through the registry so the
  ``MJLAB_JUGGLE_TARGET_HEIGHT_RANGE_M`` override is read on every call; the
  registered tasks read it when the task module is imported, as every other
  ``MJLAB_JUGGLE_*`` override does.

  The environment prints its manager tables while it is constructed; that output is
  captured so the probe's own report is readable. Warnings and errors still reach
  stderr.
  """
  cfg = agibot_x2_tennis_juggle_target_height_env_cfg(stage=stage)
  cfg.scene.num_envs = NUM_ENVS
  cfg.seed = PROBE_SEED
  with contextlib.redirect_stdout(io.StringIO()):
    env = ManagerBasedRlEnv(cfg, device="cpu")
  env.reset()
  return env, cfg


def _ball_heights(env: ManagerBasedRlEnv, rest_z: float) -> torch.Tensor:
  """Ball height above its resting state, per environment, in metres."""
  return env.scene["ball"].data.root_link_pos_w[:, 2] - rest_z


def _write_heights(env: ManagerBasedRlEnv, heights: list[float]) -> None:
  """Give each environment its own target, in the order it appears in the batch."""
  command = env.command_manager.get_command(TARGET_HEIGHT_COMMAND)
  _require(command is not None, "the target-height command is missing")
  command[:, 0] = torch.tensor(heights, dtype=command.dtype)


def _refresh_observations(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
  """Recompute the observations after a command or state write.

  ``update_history=True`` bypasses the observation manager's per-step cache, which
  a plain ``compute()`` returns without calling the terms again.
  """
  return env.observation_manager.compute(update_history=True)


def _last_column(obs: dict[str, torch.Tensor], group: str) -> torch.Tensor:
  return obs[group][:, -1]


##
# Check 1. Widths, and nothing but the target machinery differs.
##


def check_1_widths() -> str:
  tasks = list_tasks()
  _require(S1_TASK in tasks, f"{S1_TASK} is not registered")
  _require(S1A_TASK in tasks, f"{S1A_TASK} is not registered")

  cfg = load_env_cfg(S1_TASK)
  cfg.scene.num_envs = NUM_ENVS
  cfg.seed = PROBE_SEED
  with contextlib.redirect_stdout(io.StringIO()):
    env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    env.reset()
    obs = _refresh_observations(env)
    actor = obs["actor"].shape[-1]
    critic = obs["critic"].shape[-1]
    actions = env.action_manager.total_action_dim
    _require(
      actor == EXPECTED_ACTOR_WIDTH,
      f"actor width is {actor}, expected {EXPECTED_ACTOR_WIDTH} (the shipped "
      f"ladder is {LADDER_ACTOR_WIDTH})",
    )
    _require(
      critic == EXPECTED_CRITIC_WIDTH,
      f"critic width is {critic}, expected {EXPECTED_CRITIC_WIDTH} (the shipped "
      f"ladder is {LADDER_CRITIC_WIDTH})",
    )
    _require(
      actions == EXPECTED_ACTIONS,
      f"action dim is {actions}, expected {EXPECTED_ACTIONS}",
    )
  finally:
    env.close()

  # Both registered tasks must carry the stage's own target range and a resampling
  # timer that cannot expire inside an episode. Without this the probe would prove
  # the machinery works and not that the two launched conditions ask for it.
  for stage, task_id in (("s1", S1_TASK), ("s1a", S1A_TASK)):
    registered = load_env_cfg(task_id)
    command = registered.commands.get(TARGET_HEIGHT_COMMAND)
    _require(command is not None, f"{task_id} has no target-height command")
    expected_range = STAGE_TARGET_HEIGHT_RANGE_M[stage]
    _require(
      command.height_range_m == expected_range,
      f"{task_id} draws {command.height_range_m}, expected the {stage} range "
      f"{expected_range}",
    )
    _require(
      command.resampling_time_range[0] >= registered.episode_length_s,
      f"{task_id} can resample its target {command.resampling_time_range[0]} s into "
      f"a {registered.episode_length_s} s episode",
    )

  # The family is the shipped rung plus the target machinery: remove exactly the
  # declared additions and the two configs must be identical field for field.
  ladder = agibot_x2_tennis_juggle_stage_env_cfg(stage="s1-15hz")
  stripped = deepcopy(cfg)
  stripped.commands.pop(TARGET_HEIGHT_COMMAND)
  for group in ("actor", "critic"):
    _require(
      list(stripped.observations[group].terms)[-1] == TARGET_HEIGHT_COMMAND,
      f"the target-height observation is not last in the {group} group: "
      f"{list(stripped.observations[group].terms)}",
    )
    stripped.observations[group].terms.pop(TARGET_HEIGHT_COMMAND)
  stripped.rewards["ball_energy"].params.pop("command_name")
  stripped.terminations["ball_not_lifted"].params.pop("command_name")
  for name in ("ball_energy", "ball_energy_error", "ball_first_lift_s"):
    stripped.metrics[name].params.pop("command_name")
  stripped.events["reset_ball"].params.pop("target_height_command_name")

  # ``spec_fn`` is a fresh closure on every factory call, so two identical scenes
  # never compare equal on the callable; neutralise it and the environment counts on
  # both sides and every other scene field is compared field for field.
  for side in (stripped, ladder):
    side.scene.num_envs = 1
    if side.scene.terrain is not None:
      side.scene.terrain.spec_fn = None
      # The environment constructor mirrors the scene's environment count into the
      # terrain, so the built copy carries 8 here and the ladder factory default 1.
      side.scene.terrain.num_envs = 1
    for entity in side.scene.entities.values():
      entity.spec_fn = None

  sections = (
    "observations",
    "rewards",
    "terminations",
    "metrics",
    "events",
    "commands",
    "actions",
    "curriculum",
    "scene",
    "episode_length_s",
    "decimation",
    "sim",
  )
  for section in sections:
    _require(
      getattr(stripped, section) == getattr(ladder, section),
      f"{section} differs from the shipped s1-15hz stage beyond the target machinery",
    )
  _require(
    stripped.observations["actor"].enable_corruption
    == ladder.observations["actor"].enable_corruption,
    "the family changes the actor's corruption setting",
  )
  ranges = " and ".join(
    f"{stage} {load_env_cfg(task_id).commands[TARGET_HEIGHT_COMMAND].height_range_m}"
    for stage, task_id in (("s1", S1_TASK), ("s1a", S1A_TASK))
  )
  return (
    f"both task ids registered with their stage range ({ranges}); actor {actor} "
    f"(ladder {LADDER_ACTOR_WIDTH}), critic {critic} "
    f"(ladder {LADDER_CRITIC_WIDTH}), actions {actions}; the config stripped of the "
    f"declared target additions equals s1-15hz in {', '.join(sections)}"
  )


##
# Check 2. The command is per environment, in the observation and in the reward.
##


def check_2_per_environment() -> str:
  env, cfg = _build_env("s1a")
  try:
    rest_z = cfg.rewards["ball_energy"].params["ball_rest_z"]
    params = cfg.rewards["ball_energy"].params
    scale = 1.0 / JUGGLE_ENERGY_TARGET_HEIGHT_M

    # Each environment is given its own ball height as its target, with the ball at
    # rest there: the reward must be 1.0 everywhere, which cannot happen if the
    # command or the reward were shared across the batch.
    heights = _ball_heights(env, rest_z)
    _write_heights(env, heights.tolist())
    obs = _refresh_observations(env)
    _require(
      torch.allclose(_last_column(obs, "actor"), heights * scale, rtol=0, atol=1e-6),
      "the actor's last column is not the per-environment command / 0.40",
    )
    _require(
      torch.allclose(_last_column(obs, "critic"), heights * scale, rtol=0, atol=1e-6),
      "the critic's last column is not the per-environment command / 0.40",
    )
    on_target = mdp.ball_energy(env, **params)
    _require(
      torch.allclose(on_target, torch.ones_like(on_target), rtol=0, atol=1e-5),
      f"reward at each environment's own commanded apex is {on_target.tolist()}, "
      "expected 1.0 everywhere",
    )

    # Two environments of one batch are then given two different targets, and the
    # same call scores them differently: the lowest ball is given the highest target
    # while the highest keeps its own. A shared command cannot do this.
    order = torch.argsort(heights)
    low_index, high_index = int(order[0]), int(order[-1])
    crossed = heights.clone()
    crossed[low_index] = heights[high_index]
    _write_heights(env, crossed.tolist())
    off_target = mdp.ball_energy(env, **params)
    _require(
      math.isclose(float(off_target[high_index]), 1.0, rel_tol=0, abs_tol=1e-5),
      f"the highest environment lost its own target: {float(off_target[high_index])!r}",
    )
    _require(
      float(off_target[low_index]) < 0.5,
      f"the lowest ball ({float(heights[low_index]):.4f} m) given the highest "
      f"target ({float(heights[high_index]):.4f} m) still scores "
      f"{float(off_target[low_index])!r}",
    )

    # The two values the plan names explicitly.
    _write_heights(env, [0.05, JUGGLE_ENERGY_TARGET_HEIGHT_M] + [0.20] * (NUM_ENVS - 2))
    obs = _refresh_observations(env)
    first_two = _last_column(obs, "actor")[:2]
    _require(
      torch.allclose(first_two, torch.tensor([0.125, 1.0]), rtol=0, atol=1e-6),
      f"0.05 / 0.40 targets read {first_two.tolist()} in the observation, expected "
      f"[0.125, 1.0]",
    )
    reward_two = mdp.ball_energy(env, **params)[:2]
    _require(
      float(reward_two[0]) != float(reward_two[1]),
      f"0.05 and 0.40 targets scored the same reward {reward_two.tolist()}",
    )

    # Not corrupted: a second computation returns exactly the same column.
    again = _last_column(_refresh_observations(env), "actor")
    _require(
      torch.equal(again, _last_column(obs, "actor")),
      "the target-height observation changed between two computations",
    )

    # The two energy metrics read the same command as the reward. A metric left on
    # the constant would keep reporting a distance from 0.40 m while the reward
    # scored a 0.05 m target, and every check above would still pass.
    gravity = abs(float(env.sim.mj_model.opt.gravity[2]))
    ball_data = env.scene["ball"].data
    energy = 0.5 * torch.square(ball_data.root_link_lin_vel_w[:, 2]) + gravity * (
      ball_data.root_link_pos_w[:, 2] - rest_z
    )
    _write_heights(env, [0.05] * NUM_ENVS)
    error_005 = mdp.ball_energy_error(env, **cfg.metrics["ball_energy_error"].params)
    _require(
      torch.allclose(error_005, (energy - gravity * 0.05).abs(), rtol=0, atol=1e-6),
      f"ball_energy_error at a 0.05 m target is {error_005.tolist()}, expected the "
      f"distance from the 0.05 m apex energy",
    )
    _write_heights(env, [JUGGLE_ENERGY_TARGET_HEIGHT_M] * NUM_ENVS)
    error_040 = mdp.ball_energy_error(env, **cfg.metrics["ball_energy_error"].params)
    _require(
      torch.allclose(
        error_040,
        (energy - gravity * JUGGLE_ENERGY_TARGET_HEIGHT_M).abs(),
        rtol=0,
        atol=1e-6,
      ),
      f"ball_energy_error at a 0.40 m target is {error_040.tolist()}, expected the "
      f"distance from the 0.40 m apex energy",
    )
    # ``ball_energy_value`` reports the ball's energy alone, so the target cannot
    # move it; it is wired to the command only so a missing term fails there too.
    value = mdp.ball_energy_value(env, **cfg.metrics["ball_energy"].params)
    _require(
      torch.allclose(value, energy, rtol=0, atol=1e-6),
      f"ball_energy_value is {value.tolist()}, expected the ball's own energy",
    )
    return (
      f"targets 0.05 / 0.40 read {first_two.tolist()} in the observation and score "
      f"{reward_two.tolist()}; all {env.num_envs} environments score "
      f"{float(on_target.min()):.6f}-{float(on_target.max()):.6f} at their own "
      f"apex; in one call the highest ball keeps its target "
      f"({float(off_target[high_index]):.6f}) while the lowest ball given that "
      f"target scores {float(off_target[low_index]):.3e}; ball_energy_error tracks "
      f"the target ({float(error_005[0]):.4f} -> {float(error_040[0]):.4f} J/kg for "
      f"one state); repeated computations agree exactly"
    )
  finally:
    env.close()


##
# Check 3. At 0.40 m the reward equals the shipped constant-0.40 code path.
##


def check_3_matches_the_old_path() -> str:
  cfg = load_env_cfg(S1_TASK)
  params = cfg.rewards["ball_energy"].params
  _require("command_name" in params, "the family's ball_energy does not read a command")
  # The shipped path: the same parameters with the command key removed.
  old_params = {name: value for name, value in params.items() if name != "command_name"}
  _require(
    old_params["target_height"] == JUGGLE_ENERGY_TARGET_HEIGHT_M,
    "the shipped target is not 0.40 m",
  )

  states = [
    (height, speed)
    for height in (-0.02, 0.0, 0.05, 0.20, 0.21, 0.40, 0.55)
    for speed in (-2.5, -1.0, 0.0, 0.9, 2.8)
  ]
  # Both precisions. Float64 isolates the formula; float32 is the dtype the
  # environment's command and ball state actually have, so the no-op at 0.40 m is
  # also shown to survive the shipped arithmetic.
  measured = []
  for dtype, tolerance in ((torch.float64, 1e-12), (torch.float32, 1e-6)):
    stub = _stub([JUGGLE_ENERGY_TARGET_HEIGHT_M], dtype)
    worst = 0.0
    for height, speed in states:
      stub.drive(height, speed)
      # The same call with and without the command: the second is literally the
      # code path the shipped 112-wide task runs.
      new_value = float(mdp.ball_energy(stub, **params)[0])
      old_value = float(mdp.ball_energy(stub, **old_params)[0])
      worst = max(worst, abs(new_value - old_value))
      _require(
        math.isclose(new_value, old_value, rel_tol=tolerance, abs_tol=tolerance),
        f"in {dtype} at height {height} m and vertical speed {speed} m/s the command "
        f"path gives {new_value!r} and the constant-0.40 path {old_value!r}",
      )
    measured.append((str(dtype).removeprefix("torch."), worst))
  return (
    f"{len(states)} ball states at a 0.40 m target, max |command path - "
    f"constant-0.40 path| = "
    + ", ".join(f"{worst:.1e} in {name}" for name, worst in measured)
  )


##
# Check 4. The reward peaks at the commanded apex, from physics.
##


def check_4_peaks_at_the_command() -> str:
  cfg = load_env_cfg(S1A_TASK)
  params = cfg.rewards["ball_energy"].params
  heights = torch.tensor(
    [0.05, 0.10, 0.20, 0.30, 0.35, JUGGLE_ENERGY_TARGET_HEIGHT_M], dtype=STUB_DTYPE
  )
  stub = _StubBallEnv(heights)

  # A ball at rest at the commanded height has all of its energy in height, so its
  # mechanical energy is exactly the commanded apex energy and the reward must be
  # its maximum, 1.0, at every target in the range.
  at_rest = []
  for index, height in enumerate(heights):
    stub.drive(float(height))
    value = float(mdp.ball_energy(stub, **params)[index])
    at_rest.append(value)
    _require(
      math.isclose(value, 1.0, rel_tol=0, abs_tol=1e-9),
      f"a ball at rest at the commanded {float(height):.2f} m scores {value!r}, "
      f"expected 1.0",
    )

  # Physics, not the production formula: a ball at 0.10 m moving up at the speed
  # that gives it the same mechanical energy as a ball at rest at the commanded
  # height must score exactly the same. 0.5 v^2 = g (h_cmd - 0.10) is the whole
  # statement, and it uses only the ball's height and vertical speed.
  lower = 0.10
  equal_energy = []
  for index, height in enumerate(heights):
    if float(height) <= lower:
      continue
    speed = math.sqrt(2.0 * GRAVITY * (float(height) - lower))
    stub.drive(lower, speed)
    moving = float(mdp.ball_energy(stub, **params)[index])
    stub.drive(float(height), 0.0)
    resting = float(mdp.ball_energy(stub, **params)[index])
    equal_energy.append(abs(moving - resting))
    _require(
      math.isclose(moving, resting, rel_tol=0, abs_tol=1e-9),
      f"a ball at {lower} m moving {speed:.4f} m/s and a ball at rest at "
      f"{float(height):.2f} m have the same energy but score {moving!r} and "
      f"{resting!r}",
    )

  # The same ball state scored against two targets gives two values: the command is
  # what moves the peak.
  stub.drive(0.20)
  stub.command_manager.set_heights(torch.full_like(heights, 0.05))
  low_target = float(mdp.ball_energy(stub, **params)[0])
  stub.command_manager.set_heights(torch.full_like(heights, 0.40))
  high_target = float(mdp.ball_energy(stub, **params)[0])
  _require(
    low_target != high_target,
    f"a ball at rest at 0.20 m scores {low_target!r} against both a 0.05 m and a "
    f"0.40 m target",
  )
  _require(
    low_target < 0.01,
    f"a 0.05 m target pays {low_target!r} for a ball parked at 0.20 m; expected "
    f"almost nothing",
  )
  return (
    f"rest-at-command scores {min(at_rest):.9f}-{max(at_rest):.9f} over "
    f"{[round(float(h), 2) for h in heights]}; {len(equal_energy)} equal-energy "
    f"pairs agree to {max(equal_energy):.3e}; a ball at rest at 0.20 m scores "
    f"{low_target:.3e} (0.05 m target) vs {high_target:.6f} (0.40 m)"
  )


##
# Check 5. The sigma and the lift threshold scale with the target.
##


def _reward_half_width(
  stub: _StubBallEnv,
  params: dict[str, Any],
  target: float,
) -> float:
  """Energy offset from the target at which the reward falls to 1/e, in J/kg.

  Bisected on the ball's **energy**, which the stub sets from height alone, so the
  number is measured from the reward's values rather than read from a constant.
  """
  stub.command_manager.set_heights(
    torch.full((stub.num_envs,), target, dtype=stub.dtype)
  )
  one_over_e = math.exp(-1.0)
  low, high = 0.0, 2.0 * JUGGLE_ENERGY_SIGMA_J_PER_M * JUGGLE_ENERGY_TARGET_HEIGHT_M
  for _ in range(80):
    middle = 0.5 * (low + high)
    stub.drive(target + middle / GRAVITY)
    if float(mdp.ball_energy(stub, **params)[0]) > one_over_e:
      low = middle
    else:
      high = middle
  return 0.5 * (low + high)


def _not_lifted_term(cfg: Any, stub: _StubBallEnv) -> Any:
  """A live ``ball_not_lifted`` term, built as the manager builds it."""
  term_cfg = cfg.terminations["ball_not_lifted"]
  return mdp.ball_not_lifted(cfg=term_cfg, env=stub)


def _first_lift_term(cfg: Any, stub: _StubBallEnv) -> Any:
  """A live ``ball_first_lift_s`` term, built as the manager builds it."""
  term_cfg = cfg.metrics["ball_first_lift_s"]
  return mdp.ball_first_lift_time(cfg=term_cfg, env=stub)


def _bisect_verdict(
  term: Any,
  params: dict[str, Any],
  stub: _StubBallEnv,
  reports_no_lift: Any,
) -> float:
  """Ball energy at which a lift term changes verdict, in J/kg, by bisection."""
  low, high = 0.0, 2.0 * JUGGLE_LIFT_ENERGY_J_PER_M * JUGGLE_ENERGY_TARGET_HEIGHT_M
  for _ in range(50):
    middle = 0.5 * (low + high)
    stub.drive(middle / GRAVITY)
    if reports_no_lift(term, params, stub):
      low = middle
    else:
      high = middle
  return 0.5 * (low + high)


def _termination_fires(term: Any, params: dict[str, Any], stub: _StubBallEnv) -> bool:
  """Whether ``ball_not_lifted`` ends the episode for a still ball."""
  # Per-episode state: the peak energy and the elapsed steps both persist, so a
  # candidate must be scored from a fresh episode or an earlier, higher energy
  # would keep the peak above the threshold.
  term.reset(None)
  fired = False
  for _ in range(80):  # 1.6 s, past the 1.5 s timeout, at the 0.02 s step.
    fired = bool(term(stub, **params)[0])
  return fired


def _metric_reports_no_lift(
  term: Any, params: dict[str, Any], stub: _StubBallEnv
) -> bool:
  """Whether ``ball_first_lift_s`` stays at -1.0 for a still ball."""
  term.reset(None)
  return float(term(stub, **params)[0]) < 0.0


def check_5_sigma_and_threshold_scale() -> str:
  cfg = load_env_cfg(S1A_TASK)
  reward_params = cfg.rewards["ball_energy"].params
  stub = _stub([CHECKED_HEIGHTS_M[0]])
  termination = _not_lifted_term(cfg, stub)
  metric = _first_lift_term(cfg, stub)

  measured_sigma = []
  measured_threshold = []
  measured_metric = []
  for height in CHECKED_HEIGHTS_M:
    measured_sigma.append(_reward_half_width(stub, reward_params, height))
    measured_threshold.append(
      _bisect_verdict(
        termination,
        cfg.terminations["ball_not_lifted"].params,
        stub,
        _termination_fires,
      )
    )
    measured_metric.append(
      _bisect_verdict(
        metric, cfg.metrics["ball_first_lift_s"].params, stub, _metric_reports_no_lift
      )
    )

  for index, height in enumerate(CHECKED_HEIGHTS_M):
    expected_sigma = JUGGLE_ENERGY_SIGMA_J_PER_M * height
    _require(
      math.isclose(measured_sigma[index], expected_sigma, rel_tol=1e-6),
      f"sigma at a {height:.2f} m target is {measured_sigma[index]!r} J/kg, "
      f"expected 3.75 * {height} = {expected_sigma!r}",
    )
    expected_threshold = JUGGLE_LIFT_ENERGY_J_PER_M * height
    _require(
      math.isclose(measured_threshold[index], expected_threshold, rel_tol=1e-6),
      f"the ball_not_lifted threshold at {height:.2f} m is "
      f"{measured_threshold[index]!r} J/kg, expected 2.5 * {height} = "
      f"{expected_threshold!r}",
    )
    _require(
      math.isclose(measured_metric[index], expected_threshold, rel_tol=1e-6),
      f"the ball_first_lift_s threshold at {height:.2f} m is "
      f"{measured_metric[index]!r} J/kg, expected {expected_threshold!r}",
    )

  # The no-op at the reference target, stated as the shipped numbers.
  _require(
    CHECKED_HEIGHTS_M[-1] == JUGGLE_ENERGY_TARGET_HEIGHT_M,
    "0.40 is not the last checked target",
  )
  _require(
    math.isclose(measured_sigma[-1], 1.5, rel_tol=1e-6),
    f"sigma at 0.40 m is {measured_sigma[-1]!r}, not the shipped 1.5 J/kg",
  )
  _require(
    math.isclose(measured_threshold[-1], 1.0, rel_tol=1e-6),
    f"the lift threshold at 0.40 m is {measured_threshold[-1]!r}, not the shipped "
    f"1.0 J/kg",
  )

  def table(values: list[float]) -> str:
    return ", ".join(
      f"{height:.2f} m -> {value:.4f} J/kg"
      for height, value in zip(CHECKED_HEIGHTS_M, values, strict=True)
    )

  return (
    f"sigma {table(measured_sigma)}; ball_not_lifted {table(measured_threshold)}; "
    f"ball_first_lift_s {table(measured_metric)}"
  )


##
# Check 6. The spawn scales with the target.
##


def _spawn_heights(
  env: ManagerBasedRlEnv, cfg: Any, resets: int
) -> tuple[torch.Tensor, torch.Tensor]:
  rest_z = cfg.rewards["ball_energy"].params["ball_rest_z"]
  samples = []
  targets = []
  for _ in range(resets):
    env.reset()
    samples.append(_ball_heights(env, rest_z).clone())
    targets.append(commanded_target_height(env, TARGET_HEIGHT_COMMAND).clone())
  return torch.cat(samples), torch.cat(targets)


def check_6_spawn_scales() -> str:
  lines = []
  for label, stage, height, override in (
    ("s1", "s1", S1_TARGET_HEIGHT_RANGE_M[0], None),
    ("s1a-pinned", "s1a", 0.05, "0.05"),
  ):
    if override is not None:
      # The pinned S1a build goes through MJLAB_JUGGLE_TARGET_HEIGHT_RANGE_M, which
      # is how an acceptance cell fixes one height. That also proves the override
      # reaches the command term.
      os.environ[TARGET_HEIGHT_RANGE_ENV] = override
    try:
      env, cfg = _build_env(stage)
    finally:
      os.environ.pop(TARGET_HEIGHT_RANGE_ENV, None)
    try:
      spawn, targets = _spawn_heights(env, cfg, SPAWN_RESETS)
    finally:
      env.close()
    _require(
      torch.equal(targets, torch.full_like(targets, height)),
      f"{label}: the command drew {sorted(set(targets.tolist()))}, expected {height}",
    )
    expected = 0.5 * height
    mean = float(spawn.mean())
    worst = float((spawn - expected).abs().max())
    _require(
      math.isclose(mean, expected, rel_tol=0, abs_tol=1e-6),
      f"{label}: mean release height over {SPAWN_SAMPLES} resets is {mean!r} m, "
      f"expected 0.5 * {height} = {expected!r}",
    )
    _require(
      worst <= 1e-6,
      f"{label}: a release sample is {worst!r} m away from 0.5 * {height}",
    )
    lines.append(
      f"{label} target {height:.2f} m: mean {mean:.6f} m, max deviation "
      f"{worst:.2e} m over {SPAWN_SAMPLES} resets"
    )
  return "; ".join(lines)


##
# Check 7. The target survives a reset and does not drift.
##


def check_7_target_survives_a_reset() -> str:
  env, cfg = _build_env("s1a")
  try:
    rest_z = cfg.rewards["ball_energy"].params["ball_rest_z"]
    scale = 1.0 / JUGGLE_ENERGY_TARGET_HEIGHT_M
    before = commanded_target_height(env, TARGET_HEIGHT_COMMAND).clone()

    # Inside an episode the target must not move. The wide range matters here: an
    # in-episode resample on the fixed 0.40 m stage would be invisible.
    action = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
    for _ in range(DRIFT_STEPS):
      env.step(action)
    after_steps = commanded_target_height(env, TARGET_HEIGHT_COMMAND).clone()
    _require(
      torch.equal(after_steps, before),
      f"the target moved inside an episode: {before.tolist()} -> "
      f"{after_steps.tolist()} after {DRIFT_STEPS} steps",
    )
    _require(
      torch.allclose(
        _last_column(env.obs_buf, "actor"), after_steps * scale, rtol=0, atol=1e-6
      ),
      "the observation does not follow the unchanged command",
    )

    # The command metric must carry the target, or a later achieved-apex measurement
    # cannot say which height was asked for. The value reported at a reset is the
    # per-episode mean of the buffer, i.e. the target the episode just had.
    logged_key = f"Metrics/{TARGET_HEIGHT_COMMAND}/{TARGET_HEIGHT_COMMAND}"
    _require(
      torch.equal(
        env.command_manager.get_term(TARGET_HEIGHT_COMMAND).metrics[
          TARGET_HEIGHT_COMMAND
        ],
        before,
      ),
      "the command does not report its target as a metric",
    )

    # A reset resamples it, and the same reset places the ball from the new value.
    _, extras = env.reset()
    _require(
      logged_key in extras["log"],
      f"the target metric is not logged at reset: {sorted(extras['log'])[:4]}",
    )
    _require(
      math.isclose(
        float(extras["log"][logged_key]), float(before.mean()), rel_tol=0, abs_tol=1e-6
      ),
      f"the logged target is {extras['log'][logged_key]!r}, expected the episode's "
      f"mean {float(before.mean()):.6f}",
    )
    after_reset = commanded_target_height(env, TARGET_HEIGHT_COMMAND).clone()
    changed = int((after_reset != before).sum())
    _require(
      changed > 0,
      f"a reset did not resample the target: {before.tolist()} -> "
      f"{after_reset.tolist()}",
    )
    spawn = _ball_heights(env, rest_z)
    _require(
      torch.allclose(spawn, 0.5 * after_reset, rtol=0, atol=1e-6),
      f"the release heights {spawn.tolist()} are not 0.5 * the post-reset targets "
      f"{after_reset.tolist()}; the reset ran before the command was resampled",
    )
    _require(
      torch.allclose(
        _last_column(env.obs_buf, "actor"), after_reset * scale, rtol=0, atol=1e-6
      ),
      "the observation does not follow the resampled command",
    )
    return (
      f"target constant over {DRIFT_STEPS} steps ({before.tolist()}); "
      f"{changed}/{env.num_envs} environments resampled on reset "
      f"({after_reset.tolist()}), each release height is 0.5 * its own target, and "
      f"the target is logged as {logged_key} = {extras['log'][logged_key]:.4f} m"
    )
  finally:
    env.close()


CHECKS = (
  ("1 widths", check_1_widths),
  ("2 per-environment command", check_2_per_environment),
  ("3 equals the constant-0.40 path", check_3_matches_the_old_path),
  ("4 peaks at the commanded apex", check_4_peaks_at_the_command),
  ("5 sigma and threshold scale", check_5_sigma_and_threshold_scale),
  ("6 spawn scales", check_6_spawn_scales),
  ("7 target survives a reset", check_7_target_survives_a_reset),
)


def main() -> int:
  # The checks measure the committed stages, and the registry builds each config when
  # the task module is imported, so an override in the caller's shell would silently
  # replace the ranges under test. Check 6 sets the override itself for the pinned
  # case, inside this process and only around its own factory call.
  override = os.environ.get(TARGET_HEIGHT_RANGE_ENV, "")
  if override not in ("", None):
    print(
      f"FAIL {TARGET_HEIGHT_RANGE_ENV}={override!r} is set, so the registered tasks "
      "are not the committed stages. Unset it and rerun."
    )
    return 1
  print("target-height juggle probe: S1 and S1a")
  print(f"tasks {S1_TASK}")
  print(f"      {S1A_TASK}")
  print(
    "effective ranges from the registered configs: "
    + ", ".join(
      f"{stage} {load_env_cfg(task_id).commands[TARGET_HEIGHT_COMMAND].height_range_m}"
      for stage, task_id in (("s1", S1_TASK), ("s1a", S1A_TASK))
    )
    + f" (module defaults s1 {S1_TARGET_HEIGHT_RANGE_M} s1a "
    f"{S1A_TARGET_HEIGHT_RANGE_M}); {NUM_ENVS} environments per build; CPU"
  )
  # The engine prints manager tables and one-time kernel-load lines to stdout. They
  # are captured per check so the report below is the probe's own output; warnings
  # and tracebacks still reach stderr.
  for name, check in CHECKS:
    chatter = io.StringIO()
    try:
      with contextlib.redirect_stdout(chatter):
        detail = check()
    except ProbeFailure as error:
      print(f"\ncheck {name}\n  FAIL {error}")
      return 1
    print(f"\ncheck {name}\n  ok   {detail}")
  print("\nall seven checks passed")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
