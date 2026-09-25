"""Narrow recovery runner for X2 tennis-end fine-tuning.

Inherits :class:`~mjlab.tasks.velocity.rl.runner.VelocityOnPolicyRunner` (which
inherits :class:`~mjlab.rl.MjlabOnPolicyRunner`) and adds one behaviour: an
optional explicit initial-fine-tune learning-rate override applied *after*
checkpoint load.

rsl-rl 5.5.0's ``PPO.load`` restores ``self.learning_rate`` from the saved
optimizer param-group ``lr`` when ``load_cfg["optimizer"]`` is ``True`` (the
default).  This means setting ``algorithm.learning_rate`` in the config has no
effect on a resumed run — the saved optimizer LR wins.  For a first fine-tune
from the velocity source checkpoint we want to *start* at a lower rate
(recommended ``1e-4``) while preserving optimizer moments, normalizers and
iteration counter.

When ``initial_finetune_lr`` is set on the runner config, this runner overrides
``self.alg.learning_rate`` and every ``self.alg.optimizer.param_groups[*]["lr"]``
after the base ``load`` completes.  When it is ``None`` (ordinary resume of a
recovery checkpoint) the saved optimizer LR is preserved unchanged.

The adaptive-KL schedule stays active: after the first ``update`` the rate may
move away from ``initial_finetune_lr``.
"""

import math

from mjlab.tasks.velocity.rl.runner import VelocityOnPolicyRunner


class TennisRecoveryOnPolicyRunner(VelocityOnPolicyRunner):
  """Velocity runner with optional post-load LR override."""

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Load checkpoint, then optionally override the learning rate.

    Delegates to the base implementation (which restores actor, critic,
    optimizer state + moments, normalizers, iteration and
    ``common_step_counter``), then — when ``initial_finetune_lr`` is set on
    the runner config — overrides both ``alg.learning_rate`` and all
    optimizer param-group ``lr`` values.

    When ``initial_finetune_lr`` is ``None`` the saved optimizer LR is
    preserved (ordinary resume).

    When ``map_location`` is ``None`` it defaults to ``self.device`` so
    CUDA-saved checkpoints load correctly when CUDA is hidden
    (``CUDA_VISIBLE_DEVICES=''``). An explicit ``map_location`` is
    respected as given.
    """
    effective_map_location = map_location if map_location is not None else self.device
    infos = super().load(
      path,
      load_cfg=load_cfg,
      strict=strict,
      map_location=effective_map_location,
    )
    lr_override = self.cfg.get("initial_finetune_lr")
    if lr_override is not None:
      self._apply_learning_rate_override(lr_override)
    return infos

  @staticmethod
  def _validate_finetune_lr(lr: float) -> None:
    """Reject non-finite, zero, or negative learning-rate overrides."""
    if not isinstance(lr, (int, float)):
      raise ValueError(f"initial_finetune_lr must be a float, got {type(lr).__name__}")
    if not math.isfinite(lr):
      raise ValueError(f"initial_finetune_lr must be finite, got {lr}")
    if lr <= 0.0:
      raise ValueError(f"initial_finetune_lr must be positive, got {lr}")

  def _apply_learning_rate_override(self, lr: float) -> None:
    """Override algorithm LR and all optimizer param-group LRs.

    Validates the override is finite and positive *before* mutating any
    algorithm or optimizer state, so an invalid value leaves the loaded
    checkpoint state intact.
    """
    self._validate_finetune_lr(lr)
    self.alg.learning_rate = lr
    for param_group in self.alg.optimizer.param_groups:
      param_group["lr"] = lr
