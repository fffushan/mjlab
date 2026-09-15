from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import agibot_x2_flat_velocity_env_cfg
from .rl_cfg import agibot_x2_velocity_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation",
  env_cfg=agibot_x2_flat_velocity_env_cfg(),
  play_env_cfg=agibot_x2_flat_velocity_env_cfg(play=True),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
