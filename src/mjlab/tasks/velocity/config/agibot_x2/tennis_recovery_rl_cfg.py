"""RL configuration for AgiBot X2 tennis-end recovery fine-tuning.

Inherits the shipped X2 velocity PPO config exactly — same actor/critic
architecture, distribution, algorithm hyperparameters and adaptive-KL
schedule — so the source checkpoint (``model_19999.pt`` from
``2026-09-16_01-53-44_x2-velocity-measured-gains``) loads strictly.

The only addition is ``initial_finetune_lr``: an optional learning-rate
override applied *after* checkpoint load (see
:class:`~mjlab.tasks.velocity.rl.tennis_recovery_runner.TennisRecoveryOnPolicyRunner`).

rsl-rl 5.5.0 restores the saved optimizer LR on load, so merely setting
``algorithm.learning_rate`` in YAML has no post-load effect.  The runner reads
this field after load and, when set, overrides both
``self.alg.learning_rate`` and every optimizer param-group ``lr``.  When
``None`` (normal resume of a recovery checkpoint) the saved optimizer LR is
preserved.

Recommended first-finetune value: ``1e-4``.  The original adaptive-KL schedule
remains active and may move the rate away from this initial value; it is an
initial rate, not a hard cap.
"""

from dataclasses import dataclass

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class TennisRecoveryRlRunnerCfg(RslRlOnPolicyRunnerCfg):
  """X2 velocity PPO config with an optional post-load LR override.

  All inherited fields (actor/critic dims, distribution, algorithm, etc.)
  match the shipped :func:`agibot_x2_velocity_ppo_runner_cfg` so the source
  checkpoint loads strictly and ordinary task behavior is unchanged.
  """

  initial_finetune_lr: float | None = None
  """Initial fine-tuning learning rate applied after checkpoint load.

  When set, overrides ``algorithm.learning_rate`` and all optimizer
  param-group ``lr`` values *after* loading the checkpoint, so rsl-rl's
  default behaviour of restoring the saved optimizer LR is superseded.
  When ``None`` (ordinary resume), the saved optimizer LR is preserved.

  The adaptive-KL schedule stays active and may move the effective rate
  away from this initial value during training.
  """


def agibot_x2_tennis_recovery_ppo_runner_cfg() -> TennisRecoveryRlRunnerCfg:
  """Create RL runner config for X2 tennis-end recovery fine-tuning.

  Inherits actor/critic dimensions, distribution, algorithm hyperparameters
  and experiment root from the shipped velocity config, adding only the
  optional ``initial_finetune_lr`` override.

  ``experiment_name`` stays ``agibot_x2_velocity`` so the existing
  regex-only checkpoint resolver can find the source run; ``run_name`` is
  unique so new logs/checkpoints go to a new timestamped directory and never
  overwrite the source.

  ``logger`` defaults to ``tensorboard`` (matching the source run) and
  ``upload_model`` to ``False`` (no unauthorized uploads during local
  fine-tuning).  ``max_iterations`` defaults to ``20_000`` (matching the
  source run's configured horizon).
  """
  return TennisRecoveryRlRunnerCfg(
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
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="agibot_x2_velocity",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=20_000,
    run_name="x2-tennis-end-recovery",
    logger="tensorboard",
    upload_model=False,
    initial_finetune_lr=None,
  )
