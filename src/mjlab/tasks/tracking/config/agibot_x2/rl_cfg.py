"""RL configuration for AgiBot X2 Ultra tracking task."""

from dataclasses import replace

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

_MULTI_MOTION_HIDDEN_DIMS: tuple[int, ...] = (1024, 1024, 1024, 1024, 512, 128)
"""Actor and critic hidden layers of the multi-motion teacher (SCDP II-D)."""


def agibot_x2_tracking_ppo_runner_cfg(
  experiment_name: str = "agibot_x2_tracking",
) -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for AgiBot X2 Ultra tracking task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


def agibot_x2_multi_motion_ppo_runner_cfg(
  experiment_name: str = "agibot_x2_multi_motion_tracking",
) -> RslRlOnPolicyRunnerCfg:
  """Create the RL runner configuration for the X2 multi-motion teacher.

  Fork of :func:`agibot_x2_tracking_ppo_runner_cfg` that changes only the
  network shape, the initial learning rate, and the iteration cap. Both networks
  get the six specified hidden layers; the initial rate is ``3e-4`` with the
  inherited adaptive schedule, and the cap is the plan's reference setting, not
  a launch authorization. The optimizer, ELU activation, actor and critic
  observation normalization, scalar Gaussian std, and every other PPO setting
  stay as the single-motion X2 task has them.
  """
  cfg = agibot_x2_tracking_ppo_runner_cfg(experiment_name=experiment_name)
  cfg.actor = replace(cfg.actor, hidden_dims=_MULTI_MOTION_HIDDEN_DIMS)
  cfg.critic = replace(cfg.critic, hidden_dims=_MULTI_MOTION_HIDDEN_DIMS)
  cfg.algorithm = replace(cfg.algorithm, learning_rate=3.0e-4)
  cfg.max_iterations = 100_000
  return cfg
