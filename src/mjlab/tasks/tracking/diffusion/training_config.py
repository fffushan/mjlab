"""Validated configuration and analytic schedules for diffusion training."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


class TrainingConfigError(ValueError):
  """A diffusion training configuration is invalid or disagrees with D0."""


_FROZEN_MODEL = {
  "layers": 6,
  "width": 512,
  "attention_heads": 8,
  "ffn_width": 2048,
  "dropout": 0.0,
}
_FROZEN_NOISE = {
  "training_k": 1000,
  "cosine_offset": 0.008,
  "beta_min": 1.0e-5,
  "beta_max": 0.999,
  "updates": 20,
}
_FROZEN_EMA = {
  "power": 0.75,
  "max_decay": 0.9999,
}


@dataclass(frozen=True, slots=True)
class TrainingConfig:
  """The bounded trainer configuration from D2 section 5.6."""

  epochs: int = 1000
  max_updates: int | None = None
  effective_batch_size: int = 512
  microbatch_size: int = 128
  gradient_accumulation_steps: int = 4
  learning_rate: float = 1.0e-4
  weight_decay: float = 0.001
  scheduler: str = "cosine"
  warmup_updates: int = 10000
  max_grad_norm: float = 1.0
  mixed_precision: str = "bf16"
  loss_reduction: str = "token_mean"
  seed: int = 0
  epoch_sample_budget: int | None = None
  clean_weight: float = 1.0
  perturbed_weight: float = 1.0
  eval_split: str = "validation"
  eval_seed: int = 0
  ema_power: float = 0.75
  ema_max_decay: float = 0.9999

  def validate(
    self, *, explicit_budget: bool = False, allow_long_run: bool = False
  ) -> None:
    """Validate arithmetic, precision and bounded-run constraints."""
    if (
      isinstance(self.epochs, bool)
      or not isinstance(self.epochs, int)
      or self.epochs <= 0
    ):
      raise TrainingConfigError("epochs must be a positive integer")
    for name, value in (
      ("effective_batch_size", self.effective_batch_size),
      ("microbatch_size", self.microbatch_size),
      ("gradient_accumulation_steps", self.gradient_accumulation_steps),
      ("warmup_updates", self.warmup_updates),
    ):
      if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrainingConfigError(f"{name} must be a non-negative integer")
    if self.max_updates is not None and (
      isinstance(self.max_updates, bool)
      or not isinstance(self.max_updates, int)
      or self.max_updates <= 0
    ):
      raise TrainingConfigError("max_updates must be a positive integer when set")
    if explicit_budget and self.max_updates is None:
      raise TrainingConfigError(
        "an explicit max_updates budget is required for this invocation"
      )
    if (
      self.max_updates is not None and self.max_updates > 200_000 and not allow_long_run
    ):
      raise TrainingConfigError(
        "max_updates above 200000 requires an explicit allow-long-run permission"
      )
    if self.effective_batch_size <= 0 or self.microbatch_size <= 0:
      raise TrainingConfigError("batch sizes must be positive")
    if (
      self.microbatch_size * self.gradient_accumulation_steps
      != self.effective_batch_size
    ):
      raise TrainingConfigError(
        "microbatch_size * gradient_accumulation_steps must equal effective_batch_size"
      )
    if self.warmup_updates < 0:
      raise TrainingConfigError("warmup_updates must be non-negative")
    for name, value in (
      ("learning_rate", self.learning_rate),
      ("weight_decay", self.weight_decay),
      ("max_grad_norm", self.max_grad_norm),
      ("clean_weight", self.clean_weight),
      ("perturbed_weight", self.perturbed_weight),
      ("ema_power", self.ema_power),
      ("ema_max_decay", self.ema_max_decay),
    ):
      if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise TrainingConfigError(f"{name} must be finite")
      if name in {"clean_weight", "perturbed_weight"} and float(value) < 0.0:
        raise TrainingConfigError(f"{name} must be non-negative")
      if name not in {"clean_weight", "perturbed_weight"} and float(value) < 0.0:
        raise TrainingConfigError(f"{name} must be non-negative")
    if self.learning_rate <= 0.0:
      raise TrainingConfigError("learning_rate must be positive")
    if self.max_grad_norm <= 0.0:
      raise TrainingConfigError("max_grad_norm must be positive")
    if self.clean_weight == 0.0 and self.perturbed_weight == 0.0:
      raise TrainingConfigError("at least one sampling weight must be positive")
    if self.ema_power != _FROZEN_EMA["power"]:
      raise TrainingConfigError("ema_power disagrees with frozen D0 value 0.75")
    if self.ema_max_decay != _FROZEN_EMA["max_decay"]:
      raise TrainingConfigError("ema_max_decay disagrees with frozen D0 value 0.9999")
    if self.scheduler != "cosine":
      raise TrainingConfigError("only the frozen cosine scheduler is supported")
    if self.loss_reduction != "token_mean":
      raise TrainingConfigError(
        "loss_reduction must be token_mean; D2 requires plain x0 loss"
      )
    if self.mixed_precision not in {"fp32", "bf16"}:
      raise TrainingConfigError("mixed_precision must be fp32 or bf16")
    if self.epoch_sample_budget is not None and (
      isinstance(self.epoch_sample_budget, bool)
      or not isinstance(self.epoch_sample_budget, int)
      or self.epoch_sample_budget <= 0
    ):
      raise TrainingConfigError("epoch_sample_budget must be positive when set")
    if self.eval_split not in {"train", "validation", "test"}:
      raise TrainingConfigError("eval_split must be train, validation or test")
    if self.eval_split == "test":
      raise TrainingConfigError(
        "the test split is reserved for the final report and must never drive "
        "model selection"
      )

  def as_dict(self) -> dict[str, object]:
    """Return the resolved scalar fields in stable serialization order."""
    self.validate()
    return {key: value for key, value in asdict(self).items()}

  def sha256(self) -> str:
    """Hash the canonical resolved configuration."""
    payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode(
      "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()

  def resolved_updates(self, updates_per_epoch: int) -> int:
    """Return the smaller of the epoch and explicit update budgets."""
    if updates_per_epoch <= 0:
      raise TrainingConfigError("updates_per_epoch must be positive")
    candidates: list[int] = [int(self.epochs) * int(updates_per_epoch)]
    if self.max_updates is not None:
      candidates.append(int(self.max_updates))
    if not candidates:
      raise TrainingConfigError("no training update budget is configured")
    return min(candidates)

  def validate_resolved_updates(
    self, updates_per_epoch: int, *, allow_long_run: bool = False
  ) -> int:
    """Resolve the epoch/update cap and enforce the CLI long-run guard."""
    resolved = self.resolved_updates(updates_per_epoch)
    if resolved > 200_000 and not allow_long_run:
      raise TrainingConfigError(
        "resolved updates above 200000 require an explicit allow-long-run permission"
      )
    return resolved

  @classmethod
  def from_mapping(cls, payload: Mapping[str, Any]) -> "TrainingConfig":
    """Read flattened fields or the nested D2 YAML layout.

    Model, noise and EMA sections are checked against the D0 constants so a
    config cannot silently train a different artifact.
    """
    if not isinstance(payload, Mapping):
      raise TrainingConfigError("training config must be a mapping")
    model = payload.get("model")
    if model is not None:
      _check_frozen_section("model", model, _FROZEN_MODEL)
    noise = payload.get("noise")
    if noise is not None:
      if not isinstance(noise, Mapping):
        raise TrainingConfigError("noise section must be a mapping")
      normalized_noise = dict(noise)
      if "training_K" in normalized_noise and "training_k" not in normalized_noise:
        normalized_noise["training_k"] = normalized_noise.pop("training_K")
      _check_frozen_section("noise", normalized_noise, _FROZEN_NOISE)

    training = payload.get("training", payload)
    if not isinstance(training, Mapping):
      raise TrainingConfigError("training section must be a mapping")
    ema = payload.get("ema", {})
    if not isinstance(ema, Mapping):
      raise TrainingConfigError("ema section must be a mapping")
    _check_frozen_section("ema", ema, _FROZEN_EMA)
    sampling = payload.get("sampling", {})
    if not isinstance(sampling, Mapping):
      raise TrainingConfigError("sampling section must be a mapping")
    evaluation = payload.get("eval", {})
    if not isinstance(evaluation, Mapping):
      raise TrainingConfigError("eval section must be a mapping")

    values: dict[str, Any] = {}
    field_names = set(cls.__dataclass_fields__)
    for name in field_names:
      if name in training:
        values[name] = training[name]
    if "clean_weight" in sampling:
      values["clean_weight"] = sampling["clean_weight"]
    if "perturbed_weight" in sampling:
      values["perturbed_weight"] = sampling["perturbed_weight"]
    if "split" in evaluation:
      values["eval_split"] = evaluation["split"]
    if "seed" in evaluation:
      values["eval_seed"] = evaluation["seed"]
    if "power" in ema:
      values["ema_power"] = ema["power"]
    if "max_decay" in ema:
      values["ema_max_decay"] = ema["max_decay"]
    try:
      config = cls(**values)
    except (TypeError, ValueError) as exc:
      raise TrainingConfigError(f"invalid training fields: {exc}") from exc
    config.validate()
    return config

  @classmethod
  def from_yaml(cls, path: str | Path) -> "TrainingConfig":
    """Load and validate a YAML training record without launching training."""
    try:
      import yaml

      payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
      raise TrainingConfigError(f"could not read training config {path}") from exc
    except ImportError as exc:  # pragma: no cover - project environments include yaml
      raise TrainingConfigError("PyYAML is required for diffusion configs") from exc
    if not isinstance(payload, Mapping):
      raise TrainingConfigError("training YAML must contain a mapping")
    return cls.from_mapping(payload)


def _check_frozen_section(
  name: str, value: Any, expected: Mapping[str, int | float]
) -> None:
  if not isinstance(value, Mapping):
    raise TrainingConfigError(f"{name} section must be a mapping")
  for key, expected_value in expected.items():
    if key not in value:
      continue
    actual = value[key]
    try:
      if isinstance(expected_value, float):
        matches = isinstance(actual, (int, float)) and not isinstance(actual, bool)
        matches = (
          matches and math.isfinite(float(actual)) and float(actual) == expected_value
        )
      else:
        matches = isinstance(actual, int) and not isinstance(actual, bool)
        matches = matches and actual == expected_value
    except (TypeError, ValueError):
      matches = False
    if not matches:
      raise TrainingConfigError(
        f"{name}.{key}={actual!r} disagrees with frozen D0 value {expected_value!r}"
      )


def warmup_cosine_learning_rate(
  update_index: int,
  *,
  total_updates: int,
  learning_rate: float,
  warmup_updates: int,
) -> float:
  """Return the LR used by one zero-based optimizer update.

  Warmup is linear over completed updates; the cosine reaches zero on the final
  resolved update.  Keeping this arithmetic in a pure helper makes it directly
  testable and records the exact schedule used by the trainer.
  """
  if update_index < 0 or total_updates <= 0:
    raise TrainingConfigError("update index and total_updates are invalid")
  if learning_rate < 0.0 or warmup_updates < 0:
    raise TrainingConfigError("learning_rate and warmup_updates must be non-negative")
  completed = min(update_index + 1, total_updates)
  if warmup_updates > 0 and completed <= warmup_updates:
    return float(learning_rate) * completed / warmup_updates
  if total_updates <= warmup_updates:
    return float(learning_rate) * completed / max(1, warmup_updates)
  progress = (completed - warmup_updates) / (total_updates - warmup_updates)
  progress = min(1.0, max(0.0, progress))
  import math

  return float(learning_rate) * 0.5 * (1.0 + math.cos(math.pi * progress))


def ema_decay(
  update_number: int, *, power: float = 0.75, max_decay: float = 0.9999
) -> float:
  """Return the bias-corrected power-law EMA decay for update ``n``."""
  if update_number <= 0:
    raise ValueError("EMA update_number must be positive")
  if power <= 0.0 or not 0.0 < max_decay < 1.0:
    raise ValueError("EMA power and max_decay are invalid")
  return min(float(max_decay), 1.0 - (1.0 + update_number) ** (-float(power)))


__all__ = [
  "TrainingConfig",
  "TrainingConfigError",
  "ema_decay",
  "warmup_cosine_learning_rate",
]
