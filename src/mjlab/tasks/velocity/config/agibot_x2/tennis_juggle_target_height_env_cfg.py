"""AgiBot X2 target-height juggling conditions (S1 and S1a).

A second juggling family that takes the target apex height as an **input**. The
shipped ladder keeps a constant 0.40 m target in the reward, so every rung trains
the same apex at a different perception condition. Here the apex is a command: S1
repeats the shipped ``s1-15hz`` condition with the target fixed at 0.40 m, and S1a
draws the target uniformly from 0.05 to 0.40 m.

The condition itself is :func:`~mjlab.tasks.velocity.config.agibot_x2.
tennis_juggle_stage_env_cfg.agibot_x2_tennis_juggle_stage_env_cfg`'s ``s1-15hz``,
called first and edited afterwards, so the stream rate, the latency band, the
scene, the physics, the reward weights, the terminations and the resets are that
stage's by construction. What is added here is the target machinery and nothing
else:

* a ``target_height`` command term: one height per environment, in metres,
  resampled on every episode reset (:class:`~mjlab.tasks.velocity.mdp.
  target_height_command.TargetHeightCommand`). It is not corrupted, because it is a
  goal rather than a measurement;
* a ``target_height`` observation appended **last** to both the actor and the
  critic groups, divided by 0.40 so a 0.05-0.40 m target reads 0.125-1.0. Appending
  it last is what keeps every existing column at its index;
* five consumers that read the command instead of a constant: the ``ball_energy``
  target and sigma, the ``ball_not_lifted`` and ``ball_first_lift_s`` thresholds,
  and the two energy metrics. Each rule is a no-op at 0.40 m, so S1 is the shipped
  S1 condition;
* the ball's spawn drop height, scaled by ``h / 0.40``: a 0.05 m target releases the
  ball 25 mm above the bed. The lateral jitter stays absolute.

The observation widens the actor from 112 to 113 and the critic from 121 to 122, so
no checkpoint of the existing ladder resumes with ``strict=True``. That is why this
is a family and not a stage: the controller's 112-wide path does not apply to this
policy either, and a 113-wide / 29-action export is separate work.

**The target comes from one place: the command.** ``MJLAB_JUGGLE_ENERGY_TARGET_HEIGHT_M``
has no effect on this family, because every consumer here reads the command and the
spawn is scaled from it, so moving that variable alone would change nothing. Use
``MJLAB_JUGGLE_TARGET_HEIGHT_RANGE_M`` to set the height for an evaluation cell. The
ladder and nominal tasks keep honouring ``MJLAB_JUGGLE_ENERGY_TARGET_HEIGHT_M``,
because they have no command to read.

See ``docs/design/x2_tennis_target_height_plan.md``.
"""

from __future__ import annotations

import math
import os
from copy import deepcopy

from mjlab.asset_zoo.robots.agibot_x2.tennis_constants import (
  JUGGLE_ENERGY_TARGET_HEIGHT_M,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.agibot_x2.tennis_juggle_stage_env_cfg import (
  agibot_x2_tennis_juggle_stage_env_cfg,
)
from mjlab.tasks.velocity.mdp.target_height_command import (
  TARGET_HEIGHT_COMMAND,
  TargetHeightCommandCfg,
)

TARGET_HEIGHT_RANGE_ENV = "MJLAB_JUGGLE_TARGET_HEIGHT_RANGE_M"

S1_TARGET_HEIGHT_RANGE_M = (
  JUGGLE_ENERGY_TARGET_HEIGHT_M,
  JUGGLE_ENERGY_TARGET_HEIGHT_M,
)
"""S1's fixed 0.40 m target: the shipped condition, as a degenerate range."""

S1A_TARGET_HEIGHT_RANGE_M = (0.05, JUGGLE_ENERGY_TARGET_HEIGHT_M)
"""S1a's range: uniform 0.05-0.40 m.

The floor is 0.05 m rather than 0: the target divides the energy sigma and the lift
threshold and scales the spawn, so a zero target would put a zero-width Gaussian and
a zero-height release into the task. 0.05 m is 25 mm of release height and a
0.1875 J/kg sigma, small but non-degenerate.
"""

STAGE_TARGET_HEIGHT_RANGE_M: dict[str, tuple[float, float]] = {
  "s1": S1_TARGET_HEIGHT_RANGE_M,
  "s1a": S1A_TARGET_HEIGHT_RANGE_M,
  "s2": S1A_TARGET_HEIGHT_RANGE_M,
  "s3": S1A_TARGET_HEIGHT_RANGE_M,
}
"""The target range each stage runs.

S1 is fixed at the reference height, so it reproduces the shipped reward, sigma,
threshold and spawn exactly. S1a resumes S1 and draws the range instead, which is
the only difference between the two. S2 resumes S1a and carries its range unchanged,
which is the stage discipline: a rung carries the complete condition and adds one
thing, so S2 differs from S1a only in the perception band. S3 carries the range the
same way and widens the band to 40-140 ms, redrawn per acquisition, which is the rung
that made the shipped ladder flat across 0-140 ms.
"""

STAGE_BASE_STAGE: dict[str, str] = {
  "s1": "s1-15hz",
  "s1a": "s1-15hz",
  "s2": "s2-80-100ms",
  "s3": "s3-40-140ms",
}
"""The ladder stage each target-height stage is built on.

The latency, stall and noise conditions are not redefined here: they come from the
stage table in :mod:`~mjlab.tasks.velocity.config.agibot_x2.
tennis_juggle_stage_env_cfg`, so there is one definition of every range and the two
families cannot drift apart. ``s2-80-100ms`` is the shipped second rung: 15 Hz, an
80-100 ms transport band.
"""


def _target_height_range(default: tuple[float, float]) -> tuple[float, float]:
  """The target range for this run, in metres, from the environment.

  Defaults to the stage's own range, so a run that does not set
  ``MJLAB_JUGGLE_TARGET_HEIGHT_RANGE_M`` keeps the committed condition. The
  override exists because an acceptance cell has to pin the target instead of
  sampling it: a height sweep sets one value per cell, and no ``--env.*`` flag can
  reach a command term. It takes ``H`` or ``LO,HI``.

  This is an evaluation knob, not a training condition. A rung that changes the
  range belongs in the stage table.
  """
  raw = os.environ.get(TARGET_HEIGHT_RANGE_ENV, "")
  if raw in ("", None):
    return default
  parts = [p.strip() for p in str(raw).split(",") if p.strip()]
  if len(parts) == 1:
    parts = [parts[0], parts[0]]
  if len(parts) != 2:
    raise ValueError(f"{TARGET_HEIGHT_RANGE_ENV} must be H or LO,HI, got {raw!r}")
  low, high = float(parts[0]), float(parts[1])
  if not math.isfinite(low) or not math.isfinite(high) or low <= 0.0 or high < low:
    raise ValueError(
      f"{TARGET_HEIGHT_RANGE_ENV} must satisfy 0 < lo <= hi, got ({low}, {high})"
    )
  return (low, high)


def _apply_target_height(
  cfg: ManagerBasedRlEnvCfg,
  height_range_m: tuple[float, float],
) -> None:
  """Install the target command, its observation and its consumers.

  One command name is written to every consumer, so the reward, the termination, the
  lift-time metric and the spawn cannot read different heights. Nothing else about
  the config moves: the reward weights, the scene, the physics and the resets are the
  S1 stage's.
  """
  cfg.commands[TARGET_HEIGHT_COMMAND] = TargetHeightCommandCfg(
    height_range_m=height_range_m
  )

  # Appended last in both groups: a dict preserves insertion order and the
  # observation manager concatenates in that order, so every existing column keeps
  # its index and the new value is the final one. The reference height is the
  # shipped target, not the episode's own target: normalising by the sample would
  # erase the number the policy has to read.
  observation = ObservationTermCfg(
    func=mdp.target_height_observation,
    params={
      "command_name": TARGET_HEIGHT_COMMAND,
      "reference_height_m": JUGGLE_ENERGY_TARGET_HEIGHT_M,
    },
  )
  cfg.observations["actor"].terms[TARGET_HEIGHT_COMMAND] = observation
  # The critic gets the same uncorrupted value, like its other privileged terms; the
  # manager deep-copies a term config per group, and the copy here keeps the two
  # groups from sharing one mutable object through later edits.
  cfg.observations["critic"].terms[TARGET_HEIGHT_COMMAND] = deepcopy(observation)

  cfg.rewards["ball_energy"].params["command_name"] = TARGET_HEIGHT_COMMAND
  # ``ball_energy_error`` is not one of the four consumers the plan lists, but it
  # reads the same target: left on the constant, it would report the episode mean
  # distance from 0.40 m while the reward scored a 0.05 m target.
  for name in ("ball_energy", "ball_energy_error"):
    cfg.metrics[name].params["command_name"] = TARGET_HEIGHT_COMMAND
  cfg.terminations["ball_not_lifted"].params["command_name"] = TARGET_HEIGHT_COMMAND
  cfg.metrics["ball_first_lift_s"].params["command_name"] = TARGET_HEIGHT_COMMAND

  # The spawn scales with the target, so the whole episode is the reference one
  # scaled by h / 0.40: impact speed and flight time fall as sqrt(h / 0.40), which is
  # what a smaller juggle is. A 0.05 m target then releases the ball 25 mm above the
  # bed instead of 200 mm, which would ask the policy to remove energy at contact.
  cfg.events["reset_ball"].params["target_height_command_name"] = TARGET_HEIGHT_COMMAND


def agibot_x2_tennis_juggle_target_height_env_cfg(
  stage: str = "s1",
  play: bool = False,
  right_arm_measured: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a target-height juggling environment config.

  Args:
    stage: The condition to build. Must be a key of
      :data:`STAGE_TARGET_HEIGHT_RANGE_M`.
    play: If True, applies the staged factory's play-mode overrides first.
    right_arm_measured: If True, build the same stage on the measured right-arm
      servo, as the fixed-height ladder's measured twins do.

  Returns:
    A ``ManagerBasedRlEnvCfg`` for the 29-DOF pelvis-history juggling task, equal to
    the ``s1-15hz`` stage's config except for the target command, the appended
    observation and the consumers that read the command.

  Raises:
    ValueError: If ``stage`` is not a known stage.
  """
  if stage not in STAGE_TARGET_HEIGHT_RANGE_M:
    raise ValueError(
      f"unknown target-height stage {stage!r}, known stages are "
      f"{sorted(STAGE_TARGET_HEIGHT_RANGE_M)}"
    )
  # The stage factory builds the 29-DOF family: its nominal parent passes
  # ``head_actuated=False``, so the actions stay 29 and the head stays an
  # observation. The base stage carries the perception condition; this family adds
  # the target machinery on top of it.
  cfg = agibot_x2_tennis_juggle_stage_env_cfg(
    stage=STAGE_BASE_STAGE[stage], play=play, right_arm_measured=right_arm_measured
  )
  _apply_target_height(cfg, _target_height_range(STAGE_TARGET_HEIGHT_RANGE_M[stage]))
  return cfg


__all__ = [
  "S1A_TARGET_HEIGHT_RANGE_M",
  "S1_TARGET_HEIGHT_RANGE_M",
  "STAGE_BASE_STAGE",
  "STAGE_TARGET_HEIGHT_RANGE_M",
  "TARGET_HEIGHT_RANGE_ENV",
  "agibot_x2_tennis_juggle_target_height_env_cfg",
]
