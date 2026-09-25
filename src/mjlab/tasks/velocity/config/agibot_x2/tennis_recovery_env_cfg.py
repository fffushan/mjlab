"""AgiBot X2 tennis-end recovery environment configuration factory.

Builds on the existing X2 no-state-estimation velocity configuration,
replacing the twist command with the recovery-specialized
:class:`TennisRecoveryVelocityCommand` and adding the
:class:`TennisRecoveryResetEvent` paired reset event. Recovery environments
restore a full reference state from an endpoint pool and receive zero velocity
commands; retention environments keep the original reset sequence and command
distribution.

The pool is loaded lazily when the environment is actually instantiated; the
factory itself does no dataset I/O.

This module does **not** register a task. The integration owner wires shared
registrations/exports after all component handoffs are approved.
"""

from __future__ import annotations

from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.config.agibot_x2.env_cfgs import (
  agibot_x2_flat_velocity_env_cfg,
)
from mjlab.tasks.velocity.mdp.tennis_recovery import (
  TennisRecoveryResetEvent,
  TennisRecoveryVelocityCommandCfg,
)
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg

RecoveryForceMode = Literal["none", "recovery", "retention"]
"""Override the 80/20 group split for tests/evaluation.

``"none"`` uses the seeded recovery-fraction assignment; ``"recovery"`` and
``"retention"`` force every environment into that group.
"""


def agibot_x2_tennis_recovery_env_cfg(
  play: bool = False,
  *,
  pool_directory: str | None = "data/tennis",
  last_n_frames: int = 10,
  split: str = "train",
  validation_fraction: float = 0.2,
  seed: int = 42,
  recovery_fraction: float = 0.8,
  force_mode: RecoveryForceMode = "none",
  yaw_aug_range: tuple[float, float] | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create the AgiBot X2 tennis-end recovery velocity environment config.

  Builds from the no-state-estimation flat-velocity X2 config, replacing the
  twist command with :class:`TennisRecoveryVelocityCommand` and adding the
  :class:`TennisRecoveryResetEvent` paired reset event.

  Args:
    play: If True, applies play-mode overrides (infinite episode, no corruption,
      no push_robot, no curriculum) from the base config.
    pool_directory: Path to the tennis dataset directory (production). Defaults
      to ``"data/tennis"`` so the field is a string and overrideable via tyro
      CLI. Tests may pass ``None`` to inject a pool at runtime.
    last_n_frames: Number of final frames per trajectory to use as candidate
      recovery states.
    split: Which trajectory split to sample from (``"train"``, ``"validation"``,
      ``"all"``).
    validation_fraction: Fraction of trajectories held out for validation.
    seed: Seed for the deterministic train/validation split and group assignment.
    recovery_fraction: Fraction of environments assigned to the recovery group.
      The count is ``round(recovery_fraction * num_envs)``.
    force_mode: Override the group split. ``"recovery"`` forces all envs to
      recovery; ``"retention"`` forces all envs to retention; ``"none"`` uses
      the seeded assignment.
    yaw_aug_range: Optional ``(min, max)`` yaw augmentation range in radians.
      If provided, each reset samples a yaw offset and rotates the reference
      orientation and world velocities consistently, preserving roll/pitch.

  Returns:
    A ``ManagerBasedRlEnvCfg`` with the recovery command and reset event wired
    in. The base task's rewards, noise, gains, action scale, termination logic,
    actor/critic dimensions and 20-second horizon are unchanged.
  """
  cfg = agibot_x2_flat_velocity_env_cfg(play=play)

  # Replace the twist command with the recovery-specialized command.
  base_twist = cfg.commands["twist"]
  assert isinstance(base_twist, UniformVelocityCommandCfg)
  # Copy the base UniformVelocityCommandCfg fields into the recovery cfg.
  recovery_cmd_cfg = TennisRecoveryVelocityCommandCfg(
    entity_name=base_twist.entity_name,
    resampling_time_range=base_twist.resampling_time_range,
    heading_command=base_twist.heading_command,
    heading_control_stiffness=base_twist.heading_control_stiffness,
    rel_standing_envs=base_twist.rel_standing_envs,
    rel_heading_envs=base_twist.rel_heading_envs,
    rel_world_envs=base_twist.rel_world_envs,
    rel_forward_envs=base_twist.rel_forward_envs,
    init_velocity_prob=base_twist.init_velocity_prob,
    ranges=base_twist.ranges,
    viz=base_twist.viz,
    debug_vis=base_twist.debug_vis,
  )
  cfg.commands["twist"] = recovery_cmd_cfg

  # Add the recovery reset event after the existing reset events.
  cfg.events["tennis_recovery_reset"] = EventTermCfg(
    mode="reset",
    func=TennisRecoveryResetEvent,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "pool_directory": pool_directory,
      "last_n_frames": last_n_frames,
      "split": split,
      "validation_fraction": validation_fraction,
      "seed": seed,
      "recovery_fraction": recovery_fraction,
      "force_mode": force_mode,
      "yaw_aug_range": yaw_aug_range,
    },
  )

  return cfg
