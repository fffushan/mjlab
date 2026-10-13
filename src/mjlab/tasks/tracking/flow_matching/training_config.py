"""Validated configuration for the X2 x0 flow-matching family.

The bounded-run, precision, EMA and accumulation surface is shared with the
diffusion family through :class:`~mjlab.tasks.tracking.diffusion.training_config.TrainingConfig`.
The only flow-specific field is ``max_level``, which fixes the linear Gaussian
process identity that flow checkpoints record.

The flow family is stricter than the shared surface because its mathematical
contract is fixed: history dropout, non-uniform level sampling and block-weighted
losses are rejected instead of silently ignored.  The ``scheduler="cosine"``
field inherited from the shared surface is a learning-rate schedule, not a noise
schedule; this module never constructs or records a diffusion noise schedule.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..diffusion.training_config import TrainingConfig, TrainingConfigError
from .process import DEFAULT_MAX_LEVEL, FlowMatchingError, LinearNoiseProcess


class FlowTrainingConfigError(ValueError):
  """A flow-matching training configuration is invalid or violates the contract."""


# Shared diffusion sections that a nested ``training`` section would hide.  The
# flow loader passes only the ``training`` sub-mapping to the shared reader, so
# accepting these siblings would silently ignore a nonzero history dropout or a
# non-default level-sampling setting instead of rejecting it.
_SHARED_SECTION_NAMES = ("ablation", "ema", "eval", "model", "noise", "sampling")


@dataclass(frozen=True, slots=True)
class FlowTrainingConfig:
  """Bounded flow training: shared training fields plus the linear process.

  ``training`` carries the same fields as the diffusion trainer, so both
  families reuse one validated training surface and the same optimizer,
  precision, EMA and accumulation behavior.  ``max_level`` selects the exact
  linear process ``s = k / max_level`` used for noising and recorded in flow
  checkpoints.
  """

  training: TrainingConfig = TrainingConfig()
  max_level: int = DEFAULT_MAX_LEVEL

  def __post_init__(self) -> None:
    if not isinstance(self.training, TrainingConfig):
      raise FlowTrainingConfigError("training must be a TrainingConfig")
    if isinstance(self.max_level, bool) or not isinstance(self.max_level, int):
      raise FlowTrainingConfigError("max_level must be an integer")
    # Constructing the process validates the level range without mutating state.
    try:
      LinearNoiseProcess(self.max_level)
    except FlowMatchingError as exc:
      raise FlowTrainingConfigError(str(exc)) from exc

  @property
  def process(self) -> LinearNoiseProcess:
    """The exact linear Gaussian process identified by ``max_level``."""
    return LinearNoiseProcess(self.max_level)

  def validate(
    self, *, explicit_budget: bool = False, allow_long_run: bool = False
  ) -> None:
    """Validate the shared training surface and the flow-only restrictions."""
    try:
      self.training.validate(
        explicit_budget=explicit_budget, allow_long_run=allow_long_run
      )
    except TrainingConfigError as exc:
      raise FlowTrainingConfigError(str(exc)) from exc
    if float(self.training.history_state_dropout_probability) != 0.0:
      raise FlowTrainingConfigError(
        "the flow family forbids history dropout; "
        "history_state_dropout_probability must be exactly 0.0, got "
        f"{self.training.history_state_dropout_probability}"
      )
    if self.training.level_sampling != "uniform":
      raise FlowTrainingConfigError(
        "the flow family samples independent uniform integer levels; "
        f"level_sampling must be uniform, got {self.training.level_sampling!r}"
      )
    if (
      float(self.training.state_loss_weight) != 1.0
      or float(self.training.latent_loss_weight) != 1.0
    ):
      raise FlowTrainingConfigError(
        "the flow family uses plain channel-mean x0 MSE; state_loss_weight and "
        "latent_loss_weight must both be 1.0"
      )

  def as_dict(self) -> dict[str, object]:
    """Serialize the resolved configuration for identity hashing."""
    self.validate(allow_long_run=True)
    return {"training": self.training.as_dict(), "max_level": self.max_level}

  def sha256(self) -> str:
    """Hash the canonical resolved configuration."""
    payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode(
      "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()

  @classmethod
  def from_mapping(cls, payload: Mapping[str, Any]) -> "FlowTrainingConfig":
    """Read a flat or nested flow configuration without launching training.

    The shared training fields may be nested under ``training`` or supplied
    flat.  ``max_level`` may be nested under an optional ``process`` mapping or
    supplied as a flat key; :meth:`as_dict` writes the flat form, so the
    round-trip preserves a non-default level.
    """
    if not isinstance(payload, Mapping):
      raise FlowTrainingConfigError("flow training config must be a mapping")
    process_payload = payload.get("process", {})
    if process_payload is None:
      process_payload = {}
    if not isinstance(process_payload, Mapping):
      raise FlowTrainingConfigError("process section must be a mapping")
    nested_level = process_payload.get("max_level")
    flat_level = payload.get("max_level")
    if (
      nested_level is not None and flat_level is not None and nested_level != flat_level
    ):
      raise FlowTrainingConfigError(
        f"max_level disagrees between the flat key ({flat_level!r}) and the "
        f"process section ({nested_level!r})"
      )
    max_level = nested_level if nested_level is not None else flat_level
    if max_level is None:
      max_level = DEFAULT_MAX_LEVEL
    training_payload = payload.get("training", payload)
    if not isinstance(training_payload, Mapping):
      raise FlowTrainingConfigError("training section must be a mapping")
    if "training" in payload:
      hidden = sorted(name for name in _SHARED_SECTION_NAMES if name in payload)
      if hidden:
        raise FlowTrainingConfigError(
          "a nested training section hides the top-level shared "
          f"section(s) {', '.join(hidden)}; move those fields under 'training' "
          "or remove the section, because the flow loader reads training only"
        )
    try:
      training = TrainingConfig.from_mapping(training_payload)
    except TrainingConfigError as exc:
      raise FlowTrainingConfigError(str(exc)) from exc
    config = cls(training=training, max_level=max_level)
    # Reading configuration data is not execution authorization.  The trainer
    # enforces the explicit long-run permission before starting work.
    config.validate(allow_long_run=True)
    return config

  @classmethod
  def from_yaml(cls, path: str | Path) -> "FlowTrainingConfig":
    """Load a YAML flow configuration without launching training."""
    try:
      import yaml

      payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
      raise FlowTrainingConfigError(f"could not read flow config {path}") from exc
    except ImportError as exc:  # pragma: no cover - environments include yaml
      raise FlowTrainingConfigError("PyYAML is required for flow configs") from exc
    if not isinstance(payload, Mapping):
      raise FlowTrainingConfigError("flow training YAML must contain a mapping")
    return cls.from_mapping(payload)


__all__ = ["FlowTrainingConfig", "FlowTrainingConfigError"]
