"""AgiBot X2 nominal juggling pilot configuration factory.

One opt-in task for the nominal juggling pilot described in
``docs/design/x2_tennis_nominal_pilot.md``: the pelvis-history juggling task with
every randomization axis the pilot holds fixed switched off. It is built by
calling :func:`~mjlab.tasks.velocity.config.agibot_x2.tennis_juggle_env_cfg.
agibot_x2_tennis_juggle_env_cfg` and editing the returned config, so the scene,
the 1 ms contact physics, the vendor gains, the actuator limits, the reward
weights, the loss and no-bounce terminations and the observation layout are that
task's by construction and cannot drift from it.

**What is switched off, and why.**

*Randomization.* The pushes and every physics/encoder DR term are removed, so
gains, inertias, armature, effort limits, joint friction and damping, foot size
and foot friction, the torso COM offset and the encoder bias all keep their
nominal values. The pilot asks whether the policy can learn juggling at all; a
randomized plant adds a second question.

*Actor input corruption.* ``enable_corruption`` is False, so the manager drops
every term-level noise model on the actor. That is not sufficient on its own: the
ball-history term applies its own noise and its own latency, so this config also
sets ``position_noise_m`` to zero, removes the bias, and pins the stream to a
50 Hz one-sample-per-control-step schedule with zero latency and no stalls. Three
positions and their ages remain, so the actor width is unchanged.

*Resets.* Head and the right arm/waist are reset to their default pose instead of
a sampled one, and the ball is released from a centred 0.20 m drop with only
+-5 mm of lateral jitter: no plate starts, no extra fall time, no lateral speed.

*Command.* The twist command is zero. ``rel_forward_envs`` has to be cleared as
well as the ranges: the command term clamps a forward environment's ``lin_vel_x``
to at least 0.3 m/s after sampling the ranges, so zero ranges alone still give
20 % of the batch a forward command.

**What is left alone.** The target apex height is not a parameter here. The
existing ``MJLAB_JUGGLE_ENERGY_TARGET_HEIGHT_M`` override
(``_ablation_value("ENERGY_TARGET_HEIGHT_M")`` in the juggle config) is the only
way to move it, and it feeds the reward and both energy metrics with one value,
so the 0.20 m and 0.40 m arms cannot score against one target and be measured
against another. Unset, the shipped 0.40 m is used.

See ``docs/source/x2_tennis_ball_juggle.rst``.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity.config.agibot_x2.tennis_juggle_env_cfg import (
  agibot_x2_tennis_juggle_env_cfg,
)
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg

NOMINAL_REMOVED_EVENTS = (
  "push_robot",
  "foot_friction",
  "encoder_bias",
  "base_com",
  "randomize_pd_gains",
  "randomize_inertia",
  "randomize_armature",
  "randomize_effort_limits",
  "randomize_joint_friction",
  "randomize_joint_damping",
  "randomize_foot_size",
  "randomize_effort_limits_wrist",
)
"""Randomization events the nominal pilot removes, by name.

A name that is absent is skipped rather than refused: the tracking DR axes are
configurable through ``MJLAB_DR_AXES``, so an axis can be missing before this
config ever sees it. ``randomize_effort_limits_wrist`` is added by
``apply_x2_tracking_robot_randomization``, which is why it is listed separately
from the six copied tracking axes.
"""


def agibot_x2_tennis_juggle_nominal_env_cfg(
  play: bool = False,
  right_arm_measured: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the nominal pilot's AgiBot X2 juggling environment config.

  Args:
    play: If True, applies the base task's play-mode overrides first (infinite
      episode, no corruption, no push, no curriculum). The nominal settings below
      are applied in both modes.
    right_arm_measured: If True, use the right shoulder damping measured on the
      robot instead of the vendor value. Everything else about the robot is
      unchanged.

  Returns:
    A ``ManagerBasedRlEnvCfg`` for the 29-DOF pelvis-history juggling task with no
    pushes, no physics or encoder randomization, clean actor observations, fixed
    head/arm/waist resets, centred 0.20 m drop starts and a zero command.
  """
  cfg = agibot_x2_tennis_juggle_env_cfg(
    play=play,
    head_actuated=False,
    ball_history=True,
    ball_history_frame="pelvis",
    right_arm_measured=right_arm_measured,
  )

  # Randomization. Removal, not a zero range: the terms write sampled values at
  # startup or reset, so leaving them installed with a zero range would still run
  # the writes and report them as active.
  for name in NOMINAL_REMOVED_EVENTS:
    cfg.events.pop(name, None)

  # Actor input. Corruption off removes every term-level noise model; the history
  # term's own noise, bias, latency, stalls and rate need the explicit values below.
  cfg.observations["actor"].enable_corruption = False
  cfg.observations["actor"].terms["ball_hist_pelvis"].params.update(
    bias_half_width_m=0.0,
    position_noise_m=(0.0, 0.0, 0.0),
    stream_rate_hz=(50.0, 50.0),
    stream_latency_s=(0.0, 0.0),
    stall_prob=0.0,
    stall_range_s=(0.0, 0.0),
  )

  # Resets. A zero range is the default pose: both reset terms sample uniformly
  # inside the band, and a zero-width band is one point.
  cfg.events["reset_head_pose"].params["position_range"] = (0.0, 0.0)
  cfg.events["reset_robot_juggle_joints"].params["position_range"] = (0.0, 0.0)
  cfg.events["reset_ball"].params.update(
    plate_start_fraction=0.0,
    drop_height_range=(0.20, 0.20),
    drop_extra_time_s=(0.0, 0.0),
    drop_lateral_range_m=0.005,
    drop_lateral_speed_range_mps=0.0,
  )

  # Command. See the module docstring for why ``rel_forward_envs`` matters.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.rel_forward_envs = 0.0
  twist_cmd.ranges.lin_vel_x = (0.0, 0.0)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

  return cfg


__all__ = [
  "NOMINAL_REMOVED_EVENTS",
  "agibot_x2_tennis_juggle_nominal_env_cfg",
]
