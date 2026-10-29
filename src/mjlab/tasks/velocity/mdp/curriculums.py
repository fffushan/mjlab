from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, TypedDict, cast

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
    dim=1,
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to
  # simpler terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # On the initial reset (before any env step) the robot is still at its spawn
  # pose rather than a walked-to position, so ``distance`` is meaningless and
  # would spuriously promote every env from level 0 to 1, ignoring
  # ``max_init_terrain_level``. Freeze levels on that first reset.
  if env.common_step_counter == 0:
    move_up = torch.zeros_like(move_up)
    move_down = torch.zeros_like(move_down)

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  # Compute per-terrain-type mean levels.
  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
  }

  # In curriculum mode num_cols == num_terrains (one column per type),
  # so the column index directly maps to the sub-terrain name.
  sub_terrain_names = list(terrain_generator.sub_terrains.keys())
  terrain_origins = terrain.terrain_origins
  assert terrain_origins is not None
  num_cols = terrain_origins.shape[1]
  if num_cols == len(sub_terrain_names):
    types = terrain.terrain_types
    for i, name in enumerate(sub_terrain_names):
      mask = types == i
      if mask.any():
        result[name] = torch.mean(levels[mask])

  return result


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter >= stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


##
# Tennis ball on the bat.
##


class _FinishedEpisode(NamedTuple):
  """One episode that has ended, as the promotion gate sees it.

  ``length_steps`` is the episode's control-step count. ``track_err_mps`` is the
  episode's **mean** planar velocity-tracking error, not the accumulated sum the
  command term keeps.
  """

  length_steps: int
  track_err_mps: float


class ball_command_ramp(ManagerTermBase):
  """Widen the walking command on a schedule, blocked when the policy is failing.

  The difficulty advances on a step count, not on a measurement. Measured on this
  task, every survivability observable saturates while the robot is barely moving:
  an untrained policy keeps the ball for 9 s at 0.3 m/s and 15 s at 0.6 m/s, and
  1 s at 1.0 m/s. A performance threshold placed in the easy regime is therefore
  cleared with no skill at all, and a greedy gate promoted this task to its hardest
  stage within two iterations, twice, by two different routes.

  So the clock carries the difficulty, and performance can only **withhold** an
  advance: a stage is entered on its scheduled step only if the typical episode at
  the current stage is at least ``mastery_floor_s`` **and** its typical planar
  velocity-tracking error is at most ``max_track_err_frac`` times the stage's
  fastest linear command. That keeps a failing policy from being pushed faster
  while leaving the progression predictable.

  The tracking condition exists because length alone is not evidence of walking. A
  policy that stands still keeps the ball parked on the plate, so its episodes run
  the full 20 s and clear every length floor while it misses the command by about
  twice the commanded speed. Tracking is the skill the walk stages exist to teach,
  so it is measured rather than inferred from survival.

  Advancement is one stage per call and one-way. The stage, the typical episode and
  the next stage's scheduled step are returned for the training log.

  Args:
    command_name: The velocity command term whose ranges are written.
    stages: Sequence of dicts, each with ``min_step`` and any of ``lin_vel_x``,
      ``lin_vel_y``, ``ang_vel_z``. ``min_step`` is compared against
      ``env.common_step_counter``.
    mastery_floor_s: Typical episode length, in seconds, the current stage must
      sustain before the next one may be entered.
    max_track_err_frac: Largest typical tracking error the current stage may show,
      as a fraction of its fastest linear command. A fraction rather than a speed
      so that one constant covers every stage.
    window: Number of finished episodes the typical length and the typical tracking
      error are measured over.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    self._stage = 0
    self._episodes: deque[_FinishedEpisode] = deque(maxlen=int(cfg.params["window"]))
    # Calls served by this term. The gate refuses a window that fewer calls than
    # entries could have built; see the priming rule in ``__call__``.
    self._calls = 0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    command_name: str,
    stages: Sequence[Mapping[str, Any]],
    mastery_floor_s: float,
    max_track_err_frac: float,
    window: int,
  ) -> dict[str, torch.Tensor]:
    window = int(window)
    if env_ids is None:
      env_ids = slice(None)

    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None
    command_cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
    ranges = command_cfg.ranges

    # ``error_vel_xy`` accumulates the per-step planar error, each step divided by
    # ``max_command_step`` (see ``UniformVelocityCommand._update_metrics``), and is
    # zeroed only when the env resets. The episode mean is therefore the
    # accumulator times that divisor over the episode's own step count.
    max_command_step = command_cfg.resampling_time_range[1] / env.step_dt

    # A window is not evidence when one reset batch could have filled it, and exactly
    # that happens at the start of a run. ``scripts/train.py`` passes
    # ``init_at_random_ep_len=True``, so rsl_rl randomises ``episode_length_buf`` to
    # desynchronise episode endings; the construction reset then reports those staggers
    # as finished episodes while the tracking accumulator is still zero, and the
    # staggered envs time out over the next few steps. The window therefore claims long
    # episodes with perfect tracking. Measured 2026-10-26: it read 16.67 s while the
    # mean episode length in the same log block was 17 steps, and the value decayed
    # 16.67 -> 15.16 -> 9.92 -> 1.42 s as real episodes replaced the fabrications.
    #
    # A fresh run is shielded by the schedule, which is closed for the first 2000
    # iterations. A resumed run is not: ``rl/runner.py`` restores ``common_step_counter``
    # from the checkpoint, so the schedule is open on the very first call, and on
    # 2026-10-26 a resumed run was promoted through both stages within two calls. An
    # elapsed-time rule cannot survive that restore, because restoring the counter *is*
    # a jump. This term's own call count does survive it, so the rule is: the window
    # must span at least as many calls as it holds episodes. One batch can never satisfy
    # that, and real episodes satisfy it within about one window of steps.
    self._calls += 1
    first_call = self._calls == 1

    # ``env.episode_length_buf`` still holds the length of the episode that just
    # ended: the environment zeroes it only after every manager's reset.
    steps = env.episode_length_buf[env_ids]
    if steps.numel() and not first_call:
      err_sum = command_term.metrics["error_vel_xy"][env_ids]
      track_err = err_sum * max_command_step / steps.clamp(min=1)
      self._episodes.extend(
        _FinishedEpisode(int(n), float(err))
        for n, err in zip(
          steps.detach().cpu().tolist(),
          track_err.detach().cpu().tolist(),
          strict=True,
        )
      )

    # One episode length of calls must have passed before the window can only hold
    # episodes this run observed. The staggering above lasts at most one episode length,
    # a term call happens at most once per env step, so by then every staggered episode
    # has ended and the window has been refreshed many times over. Counting this term's
    # own calls rather than env steps is deliberate: it cannot be jumped.
    primed = self._calls > int(env.max_episode_length)

    # A partial window is not evidence either.
    complete = primed and len(self._episodes) >= window
    typical_seconds = 0.0
    typical_track_err = 0.0
    mean_seconds = 0.0
    if self._episodes:
      lengths = [episode.length_steps for episode in self._episodes]
      mean_seconds = float(np.mean(lengths)) * env.step_dt
      if complete:
        typical_seconds = float(np.median(lengths)) * env.step_dt
        typical_track_err = float(
          np.median([episode.track_err_mps for episode in self._episodes])
        )

    # Scale-free threshold: a fraction of the fastest linear command the active
    # stage allows, so one constant holds at every stage. Read before the ranges are
    # written below, so the gate and the log both report the stage that was judged.
    max_lin_speed = max(abs(v) for v in (*ranges.lin_vel_x, *ranges.lin_vel_y))
    track_err_limit = float(max_track_err_frac) * max_lin_speed

    promoted = False
    if complete and self._stage + 1 < len(stages):
      next_stage = stages[self._stage + 1]
      scheduled = env.common_step_counter >= int(next_stage["min_step"])
      mastered = (
        typical_seconds >= mastery_floor_s and typical_track_err <= track_err_limit
      )
      if scheduled and mastered:
        self._stage += 1
        promoted = True

    active = stages[self._stage]
    for field_name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
      if field_name in active:
        setattr(ranges, field_name, tuple(active[field_name]))

    next_index = min(self._stage + 1, len(stages) - 1)
    return {
      "stage": torch.tensor(self._stage),
      "promoted": torch.tensor(float(promoted)),
      "typical_episode_s": torch.tensor(typical_seconds),
      "mean_episode_s": torch.tensor(mean_seconds),
      "typical_track_err_mps": torch.tensor(typical_track_err),
      "track_err_limit_mps": torch.tensor(track_err_limit),
      "mastery_floor_s": torch.tensor(float(mastery_floor_s)),
      "next_stage_step": torch.tensor(float(stages[next_index]["min_step"])),
      "lin_vel_x_max": torch.tensor(float(ranges.lin_vel_x[1])),
      "ang_vel_z_max": torch.tensor(float(ranges.ang_vel_z[1])),
    }
