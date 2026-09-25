from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import (
  TennisRecoveryOnPolicyRunner,
  VelocityOnPolicyRunner,
)

from .env_cfgs import agibot_x2_flat_velocity_env_cfg
from .rl_cfg import agibot_x2_velocity_ppo_runner_cfg
from .tennis_recovery_env_cfg import agibot_x2_tennis_recovery_env_cfg
from .tennis_recovery_rl_cfg import agibot_x2_tennis_recovery_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation",
  env_cfg=agibot_x2_flat_velocity_env_cfg(),
  play_env_cfg=agibot_x2_flat_velocity_env_cfg(play=True),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Tennis-end recovery fine-tuning task. Builds on the no-state-estimation
# velocity config, replacing the twist command with the recovery-specialized
# command and adding the paired recovery reset event. Recovery environments
# restore a full reference state from the tennis endpoint pool and receive
# zero velocity commands; retention environments keep the original reset
# sequence and command distribution. The source velocity checkpoint loads
# strictly via the narrow TennisRecoveryOnPolicyRunner (same actor/critic
# dimensions, sensor frames, rewards, observations, actions, PD/DR).
# See docs/source/x2_tennis_recovery.rst.
register_mjlab_task(
  task_id=("Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Recovery"),
  env_cfg=agibot_x2_tennis_recovery_env_cfg(),
  play_env_cfg=agibot_x2_tennis_recovery_env_cfg(play=True),
  rl_cfg=agibot_x2_tennis_recovery_ppo_runner_cfg(),
  runner_cls=TennisRecoveryOnPolicyRunner,
)

# Torso-IMU ablation grid. The actor reads the torso IMU (``imu_1``) in all four
# variants; the two axes describe the *critic*, which is the only part that can
# legitimately see a different frame from the actor:
#
#   suffix            critic velocity signals   critic projected_gravity
#   Critic-Pelvis-Root      pelvis imu_0            root-link orientation
#   Critic-Pelvis-Upvector  pelvis imu_0            torso imu_1 up-vector
#   Critic-Torso-Root       torso imu_1             root-link orientation
#   Critic-Torso-Upvector   torso imu_1             torso imu_1 up-vector
#
# ``Critic-Pelvis-Root`` differs from the shipped task in the actor IMU alone, so
# it is the clean actor contrast; the other three measure the critic axes under
# the torso actor. See docs/source/sim2real_domain_randomization.md.

register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Torso-IMU-Critic-Pelvis-Root"
  ),
  env_cfg=agibot_x2_flat_velocity_env_cfg(
    imu_source="torso", critic_imu_source="pelvis", critic_gravity_source="root"
  ),
  play_env_cfg=agibot_x2_flat_velocity_env_cfg(
    play=True,
    imu_source="torso",
    critic_imu_source="pelvis",
    critic_gravity_source="root",
  ),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_torso_imu_critic_pelvis_root"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Torso-IMU-Critic-Pelvis-Upvector"
  ),
  env_cfg=agibot_x2_flat_velocity_env_cfg(
    imu_source="torso", critic_imu_source="pelvis", critic_gravity_source="upvector"
  ),
  play_env_cfg=agibot_x2_flat_velocity_env_cfg(
    play=True,
    imu_source="torso",
    critic_imu_source="pelvis",
    critic_gravity_source="upvector",
  ),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_torso_imu_critic_pelvis_upvector"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Torso-IMU-Critic-Torso-Root"
  ),
  env_cfg=agibot_x2_flat_velocity_env_cfg(
    imu_source="torso", critic_imu_source="torso", critic_gravity_source="root"
  ),
  play_env_cfg=agibot_x2_flat_velocity_env_cfg(
    play=True,
    imu_source="torso",
    critic_imu_source="torso",
    critic_gravity_source="root",
  ),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_torso_imu_critic_torso_root"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Torso-IMU-Critic-Torso-Upvector"
  ),
  env_cfg=agibot_x2_flat_velocity_env_cfg(
    imu_source="torso", critic_imu_source="torso", critic_gravity_source="upvector"
  ),
  play_env_cfg=agibot_x2_flat_velocity_env_cfg(
    play=True,
    imu_source="torso",
    critic_imu_source="torso",
    critic_gravity_source="upvector",
  ),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_torso_imu_critic_torso_upvector"
  ),
  runner_cls=VelocityOnPolicyRunner,
)
