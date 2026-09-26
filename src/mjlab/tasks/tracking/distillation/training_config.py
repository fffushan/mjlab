"""Validated configuration for pure VAE distillation updates.

The trainer deliberately owns no simulator or collection schedule.  This
configuration only describes replay minibatches and the accepted M2 supervised
objective.  Production defaults retain the paper-sized model elsewhere; the
model settings remain independently configurable for CPU tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

TrainingLatentMode = Literal["sampled"]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
  """Bounded settings for one raw-replay supervised training stream.

  ``accumulation_steps`` is the number of replay minibatches in one optimizer
  update.  Each microbatch loss is divided by this value before backward, so
  the resulting update is the mean of the requested microbatch gradients.
  The accepted training path always uses sampled posterior latents with an
  explicit RNG generator; deterministic mean inference belongs to evaluation.
  """

  learning_rate: float = 5e-4
  beta: float = 0.01
  accumulation_steps: int = 15
  minibatch_size: int = 256
  latent_mode: TrainingLatentMode = "sampled"

  def __post_init__(self) -> None:
    for name, value in (
      ("learning_rate", self.learning_rate),
      ("beta", self.beta),
    ):
      if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
      if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive")
    for name, value in (
      ("accumulation_steps", self.accumulation_steps),
      ("minibatch_size", self.minibatch_size),
    ):
      if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if self.latent_mode != "sampled":
      raise ValueError("latent_mode must be 'sampled'")


# The descriptive alias makes the lifecycle API readable without breaking the
# shorter name used by configuration callers.
DistillationTrainingConfig = TrainingConfig


__all__ = ["DistillationTrainingConfig", "TrainingConfig", "TrainingLatentMode"]
