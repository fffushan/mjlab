from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import (
  TennisRecoveryOnPolicyRunner,
  VelocityOnPolicyRunner,
)

from .env_cfgs import agibot_x2_flat_velocity_env_cfg
from .rl_cfg import agibot_x2_velocity_ppo_runner_cfg
from .tennis_ball_env_cfg import agibot_x2_tennis_ball_env_cfg
from .tennis_juggle_env_cfg import agibot_x2_tennis_juggle_env_cfg
from .tennis_recovery_env_cfg import agibot_x2_tennis_recovery_env_cfg
from .tennis_recovery_rl_cfg import agibot_x2_tennis_recovery_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation",
  env_cfg=agibot_x2_flat_velocity_env_cfg(),
  play_env_cfg=agibot_x2_flat_velocity_env_cfg(play=True),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Tennis-ball-on-the-bat velocity task. The base no-state-estimation velocity
# task plus the tennis equipment: the racket assembly rigidly attached to the
# right wrist and a floating tennis ball spawned on the string bed. Physics runs
# at 1 ms so the calibrated negative-solref bed contact stays stable, and the
# robot/ball resets are pinned so the ball starts on the bed in every env. The
# ball is not yet observed or rewarded; that is the next step and it changes the
# actor input width. See docs/source/x2_tennis_ball_balance.rst.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Balance",
  env_cfg=agibot_x2_tennis_ball_env_cfg(),
  play_env_cfg=agibot_x2_tennis_ball_env_cfg(play=True),
  # The ball observation widens the actor from 102 to 105 (position only) and the
  # critic from 117 to 123 (position and velocity), so the observation normalizer
  # has a different width. Runs made before the ball was observed, or with the
  # ball velocity in the actor, cannot be resumed, hence a separate experiment
  # name.
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_tennis_ball_centre_pos_only"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# Juggling variant of the tennis-ball task: the ball bounces on the plate and
# returns to an apex about 30 cm above it, instead of being carried. Same scene,
# observations, resets, termination and domain randomization as the balancing
# task above; the reward set pays per bounce and charges for parking the ball, the
# twist command is pinned to a standstill band, and there is no command
# curriculum. The observation contract is unchanged, so the widths are the
# balancing task's 105/123, but the reward set is not, hence a separate experiment
# name. See tennis_juggle_env_cfg.py and docs/source/x2_tennis_ball_juggle.rst.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle",
  env_cfg=agibot_x2_tennis_juggle_env_cfg(),
  play_env_cfg=agibot_x2_tennis_juggle_env_cfg(play=True),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_tennis_ball_juggle"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# The same juggling task with the two head joints removed from the action space, so a
# 29-DOF policy. The head regularly breaks on the real robots, and two of the three
# machines we can reach already have a head axis that does not follow its command. The
# joints stay in the observation because the camera rides on the head and its measured
# pose is needed for the camera-to-plate conversion, and their pose is randomized per
# episode so one policy covers robots that park the head differently. Widths become
# 103/121 (105/123 minus the two action channels) with 29 actions.
register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
  ),
  env_cfg=agibot_x2_tennis_juggle_env_cfg(head_actuated=False),
  play_env_cfg=agibot_x2_tennis_juggle_env_cfg(play=True, head_actuated=False),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_tennis_ball_juggle_29dof"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# The 29-DOF juggling task with the actor's ball observation widened from one
# delayed plate-frame position (3 numbers) to three timestamped samples (12
# numbers): the estimator's rate, latency, stalls, bias and noise are modelled
# inside the term instead of by a fixed delay buffer. Actor 103 -> 112, critic stays
# 121, actions stay 29. The observation contract and therefore the normalizer width
# changed, so this is a separate task and experiment.
register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF-Ball3"
  ),
  env_cfg=agibot_x2_tennis_juggle_env_cfg(head_actuated=False, ball_history=True),
  play_env_cfg=agibot_x2_tennis_juggle_env_cfg(
    play=True, head_actuated=False, ball_history=True
  ),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_tennis_ball_juggle_29dof_ball3"
  ),
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
