"""Per-episode ball cases for the mixed safety stage.

One integer per environment is drawn at reset and stored on the ``reset_ball``
event instance. Every consumer - the hold, the rewards, the terminations and the
new step event - reads that one tensor through :func:`case_of`, so no two terms can
disagree about the case an environment is in.

The case ids are ordered as the declared shares are: live, ground, never-seen and
induced. The shares themselves and the placement values live in
:mod:`mjlab.asset_zoo.robots.agibot_x2.tennis_constants`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

LIVE = 0
"""Normal juggling episode: a live ball, held then released."""

GROUND = 1
"""Ball already resting on the floor in front of the robot."""

NEVER_SEEN = 2
"""Ball resting on the floor behind the robot, outside every camera view."""

INDUCED = 3
"""Live juggle, then a horizontal velocity impulse at the sampled loss time."""

CASES = (LIVE, GROUND, NEVER_SEEN, INDUCED)
"""Every case id, in declared-share order."""


def case_of(env: ManagerBasedRlEnv, event_name: str) -> torch.Tensor | None:
  """The per-environment case tensor of a ``reset_ball`` event instance.

  Returns ``None`` when the stage does not declare a mixture: the reset term then
  never allocated the tensor, and every consumer treats the batch as all-live, so
  stages before S10 keep their exact behaviour.

  Args:
    env: The environment.
    event_name: Name of the class-based reset event that owns the case draw.

  Returns:
    Shape ``(num_envs,)`` long tensor of case ids, or ``None``.
  """
  term = env.event_manager.get_term_cfg(event_name).func
  return getattr(term, "case", None)


def live_mask(cases: torch.Tensor) -> torch.Tensor:
  """True where the case is playable: ``LIVE`` or ``INDUCED``.

  A ground or never-seen ball is dead from reset, so a term that rewards the ball
  must pay nothing there. An induced episode is live until its impulse; the ball
  rewards stay enabled for it, and the impulse itself is what makes the loss real.
  """
  return (cases == LIVE) | (cases == INDUCED)
