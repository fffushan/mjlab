"""A per-environment target apex height, as a command term.

The target apex height is an **input** to the task, not a reward constant. It enters
as a command so that the value a reward scores against is the value the episode was
given, and so that two environments in one batch can be asked for two different
heights.

Two properties follow from "goal, not measurement":

* the command is **not corrupted**. Corrupting a sensor reading teaches a policy to
  hedge against measurement error; the height an operator asks for has no such error.
  The observation term :func:`target_height_observation` therefore has no noise model
  on the actor, and the critic reads the same uncorrupted value.
* the command is resampled **only** on an episode reset, through
  :meth:`CommandTerm.reset`. The timed path is disabled by a resampling range beyond
  any horizon, so a target cannot change under a reward mid-episode.

See ``docs/design/x2_tennis_target_height_plan.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

TARGET_HEIGHT_COMMAND = "target_height"
"""Default key of the target-height command, and of the observation that reads it."""


def commanded_target_height(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """The target apex height the named command currently holds, in metres.

  This is the single accessor for every consumer of the target: the energy reward's
  target, its sigma, the lift threshold and the spawn scaling all read the same
  tensor here, so they cannot disagree about which height is being scored.

  Args:
    env: The environment.
    command_name: Key of the target-height command term in ``cfg.commands``.

  Returns:
    Shape ``(num_envs,)``, in metres.

  Raises:
    KeyError: If the task has no command term under that name. ``CommandManager``
      indexes its term dict, so a misconfigured task fails on the lookup itself.
    ValueError: If the manager has no terms at all (``NullCommandManager`` returns
      ``None`` from ``get_command``). A silent zero would be worse: it would make
      the target, sigma, lift threshold and spawn height all zero together, which
      is a whole task scored against nothing instead of one loud failure.
  """
  command = env.command_manager.get_command(command_name)
  if command is None:
    raise ValueError(
      f"no command term named {command_name!r}: a target-height consumer needs a "
      "TargetHeightCommandCfg installed under that name"
    )
  return command.reshape(env.num_envs)


def draw_commanded_target_height(
  env: ManagerBasedRlEnv,
  command_name: str,
  env_ids: torch.Tensor,
) -> torch.Tensor:
  """Draw the named command's target for a reset, and report it in metres.

  A reset event that has to build the episode from the target -- the ball release
  does, because the spawn scales with it -- calls this instead of reading the
  command, because the environment applies its reset events **before** it resets
  the command manager. The draw is marked, and the command manager's own reset in
  the same episode keeps it rather than drawing a second value, so an episode has
  exactly one target and the spawn is built from it.

  Args:
    env: The environment.
    command_name: Key of the target-height command term in ``cfg.commands``.
    env_ids: Environments being reset.

  Returns:
    Shape ``(len(env_ids),)``, in metres.

  Raises:
    KeyError: If the task has no command term under that name.
    ValueError: If the term exists but is not a ``TargetHeightCommand``.
  """
  term = env.command_manager.get_term(command_name)
  if not isinstance(term, TargetHeightCommand):
    raise ValueError(
      f"command {command_name!r} is not a TargetHeightCommand, so a reset event "
      "cannot draw a target from it"
    )
  return term.draw_for_reset(env_ids)


def target_height_observation(
  env: ManagerBasedRlEnv,
  reference_height_m: float,
  command_name: str = TARGET_HEIGHT_COMMAND,
) -> torch.Tensor:
  """The commanded target apex height, divided by a reference height.

  The division is what keeps the input in the same magnitude band as the rest of the
  vector. At the shipped 0.40 m reference a 0.05-0.40 m target reads 0.125-1.0, and
  the value is 1.0 for the fixed 0.40 m condition. The reference is a constant of the
  observation contract, not of the target range: normalising by the episode's own
  target would erase the number the policy has to read.

  Args:
    env: The environment.
    reference_height_m: Height the input is divided by, in metres.
    command_name: Key of the target-height command term in ``cfg.commands``.

  Returns:
    Shape ``(num_envs, 1)``, dimensionless.
  """
  height = commanded_target_height(env, command_name)
  return (height / reference_height_m).unsqueeze(-1)


class TargetHeightCommand(CommandTerm):
  """One target apex height per environment, in metres.

  The value is drawn uniformly inside :attr:`TargetHeightCommandCfg.height_range_m` on
  every episode reset and is constant for the rest of the episode, so a reward that
  scores against it scores against a target the episode was given from its first step.

  The term samples once at construction as well as on reset. The environment resets
  before the first reward, but a consumer that divides by the target (the energy
  sigma does) must not be able to see a zero-length episode, and a draw is cheaper to
  reason about than an unwritten zero that happens to be safe.

  **One draw per episode, and the release event may take it.** The environment
  applies its reset events before it resets the command manager, so a reset event
  that must build the episode from the target -- the ball release does, because the
  spawn scales with it -- calls :meth:`draw_for_reset`. That draws the episode's
  target and marks it; the command manager's own reset then keeps it instead of
  drawing a second value. Without the mark the two would draw independently and the
  episode would be spawned for one target and scored against another.

  The term also reports the target it holds as a command metric, so a run's log
  records the heights it was asked for.
  """

  cfg: TargetHeightCommandCfg

  def __init__(self, cfg: TargetHeightCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.height_command = torch.zeros(self.num_envs, 1, device=self.device)
    self._pre_drawn = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self._resample_command(torch.arange(self.num_envs, device=self.device))

  @property
  def command(self) -> torch.Tensor:
    return self.height_command

  def draw_for_reset(self, env_ids: torch.Tensor) -> torch.Tensor:
    """Draw these environments' target now, and report it in metres.

    Called by a reset event that needs the target before the command manager's own
    reset runs. See the class docstring for why the mark is required.

    Args:
      env_ids: Environments being reset.

    Returns:
      Shape ``(len(env_ids),)``, in metres.
    """
    self._resample_command(env_ids)
    self._pre_drawn[env_ids] = True
    return self.height_command[env_ids].reshape(len(env_ids))

  def _update_metrics(self) -> None:
    # The batch's commanded apex, so a run's log shows which heights it asked for:
    # ``CommandTerm.reset`` reports the per-episode mean of this buffer and zeroes
    # it. It is a copy, not a view -- a view would be zeroed in place and would erase
    # the command. A measurement BINNED by commanded target needs a per-episode
    # recorder for the achieved apex as well, which is separate work.
    self.metrics[TARGET_HEIGHT_COMMAND] = self.height_command[:, 0].clone()

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Environments drawn by this episode's reset event keep the value they were
    # drawn; the flag is consumed here so it cannot leak into another episode.
    undrawn = env_ids[~self._pre_drawn[env_ids]]
    self._pre_drawn[env_ids] = False
    if len(undrawn) == 0:
      return
    low, high = self.cfg.height_range_m
    self.height_command[undrawn, 0] = torch.empty(
      len(undrawn), device=self.device
    ).uniform_(low, high)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    # Nothing in the state moves the target inside an episode; see the module
    # docstring. The reset path is the only writer.
    del env_ids


@dataclass(kw_only=True)
class TargetHeightCommandCfg(CommandTermCfg):
  """Configuration of the target apex height command.

  Attributes:
    height_range_m: ``(low, high)`` range one target is drawn from per episode, in
      metres. ``(h, h)`` fixes the target at ``h``, which is how the reference 0.40 m
      condition reproduces the shipped constant target exactly.
    resampling_time_range: The command base class requires a timed resampling range.
      The default is 1e9 s, i.e. never within a horizon: an episode reset is the only
      resample point. A shorter range would move the target mid-episode and make the
      reward score a target the episode was not given.
  """

  height_range_m: tuple[float, float]
  resampling_time_range: tuple[float, float] = (1e9, 1e9)

  def build(self, env: ManagerBasedRlEnv) -> TargetHeightCommand:
    return TargetHeightCommand(self, env)

  def __post_init__(self) -> None:
    low, high = self.height_range_m
    if low <= 0.0 or high < low:
      raise ValueError(
        "height_range_m must satisfy 0 < low <= high: the target divides the energy "
        f"sigma and the lift threshold, and scales the spawn, got {(low, high)}"
      )
