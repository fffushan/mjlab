"""Pure raw-replay supervised updates for the M2 conditional VAE.

This module intentionally does not collect data, step a simulator, persist
state, or own a CLI.  A caller supplies a bounded raw replay and explicit RNG
generators.  Each update samples ``accumulation_steps`` minibatches, divides
each loss by that count before backward, and performs exactly one Adam step.

Student normalizers are updated only through :meth:`update_normalizers_from_new_data`.
That method requires an explicit update identity and rejects reuse, so a runner
must pass the newly inserted raw batch rather than resampling the accumulated
replay.  Collection and evaluation should call :meth:`freeze_normalizers` (or
the corresponding boundary helpers).
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from mjlab.tasks.tracking.distillation.model import (
  ConditionalVAE,
  ModelValidationError,
  VaeLoss,
  vae_loss,
)
from mjlab.tasks.tracking.distillation.storage import (
  LabeledReplayBatch,
  LabeledReplayBuffer,
  ReplayValidationError,
)
from mjlab.tasks.tracking.distillation.training_config import TrainingConfig


class TrainerValidationError(ValueError):
  """Raised when a raw batch or lifecycle request is invalid."""


class TrainerNumericalError(RuntimeError):
  """Raised when a finite optimizer step cannot be safely completed."""


class TrainerPoisonedError(TrainerNumericalError):
  """A post-optimizer failure made further trainer use unsafe.

  A poisoned trainer must be restored from a separately validated known-good
  state before another update or checkpoint save is permitted.
  """


@dataclass(frozen=True, slots=True)
class TrainerHealth:
  """Small checkpoint-facing health record for a trainer instance."""

  healthy: bool
  reason: str | None = None


@dataclass(frozen=True, slots=True)
class FreshTrainingData:
  """One newly inserted raw batch and its unique normalization update identity.

  ``update_id`` is normally a collector iteration or replay insertion sequence.
  It is deliberately separate from replay metadata: the lifecycle runner owns
  the identity and must not use a random replay sample as fresh data.
  """

  batch: LabeledReplayBatch
  update_id: Hashable

  def __post_init__(self) -> None:
    if not isinstance(self.batch, LabeledReplayBatch):
      raise TrainerValidationError("fresh training data must contain a replay batch")
    try:
      hash(self.update_id)
    except TypeError as exc:
      raise TrainerValidationError("update_id must be hashable") from exc


@dataclass(frozen=True, slots=True)
class TrainingUpdate:
  """Detached diagnostics for one completed optimizer update."""

  optimizer_step: int
  microbatches: int
  samples: int
  total_loss: float
  reconstruction: float
  kl: float
  gradient_norm: float


# A descriptive alias is useful to lifecycle code without creating a second API.
TrainerState = TrainingUpdate


def _finite(name: str, value: torch.Tensor, *, skipped: bool = True) -> None:
  if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all().item():
    cause = "optimizer step was skipped" if skipped else "optimizer step already ran"
    raise TrainerNumericalError(f"non-finite {name}; {cause}")


def _generator_device(generator: torch.Generator) -> torch.device:
  try:
    return torch.device(generator.device)
  except (AttributeError, RuntimeError) as exc:
    raise TrainerValidationError("generator must expose a valid torch device") from exc


class VaeDistillationTrainer:
  """Adam trainer for raw labeled replay and the accepted M2 loss."""

  def __init__(
    self,
    model: ConditionalVAE,
    replay: LabeledReplayBuffer,
    config: TrainingConfig | None = None,
    *,
    replay_generator: torch.Generator | None = None,
    latent_generator: torch.Generator | None = None,
    seed: int = 0,
    teacher: nn.Module | None = None,
  ) -> None:
    if not isinstance(model, ConditionalVAE):
      raise TrainerValidationError("model must be a ConditionalVAE")
    if not isinstance(replay, LabeledReplayBuffer):
      raise TrainerValidationError("replay must be a LabeledReplayBuffer")
    if config is None:
      config = TrainingConfig()
    if not isinstance(config, TrainingConfig):
      raise TrainerValidationError("config must be a TrainingConfig")
    if replay.schema != model.schema:
      raise TrainerValidationError("replay schema does not match student model schema")
    if not isinstance(seed, int) or isinstance(seed, bool):
      raise TrainerValidationError("seed must be an integer")
    self.model = model
    self.replay = replay
    self.config = config
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
    self.replay_generator = replay_generator or torch.Generator(device="cpu")
    self.latent_generator = latent_generator or torch.Generator(device="cpu")
    if replay_generator is None:
      self.replay_generator.manual_seed(seed)
    if latent_generator is None:
      self.latent_generator.manual_seed(seed + 1)
    _generator_device(self.replay_generator)
    _generator_device(self.latent_generator)

    # A teacher is not needed for pure replay optimization.  If a caller keeps
    # one beside this trainer, defensively preserve its frozen/eval contract.
    self.teacher = teacher
    if teacher is not None:
      teacher.eval()
      teacher.requires_grad_(False)

    self.optimizer_steps = 0
    self.microbatches_seen = 0
    self.samples_seen = 0
    self._normalization_update_ids: set[Hashable] = set()
    self._poison_reason: str | None = None

  @property
  def poisoned(self) -> bool:
    """Whether a post-step failure forbids updates and checkpoint saves."""
    return self._poison_reason is not None

  @property
  def is_healthy(self) -> bool:
    """Whether this trainer may safely update or be checkpointed."""
    return not self.poisoned

  def health_check(self) -> TrainerHealth:
    """Return the current health without changing trainer state."""
    return TrainerHealth(not self.poisoned, self._poison_reason)

  def assert_healthy(self) -> None:
    """Reject updates/saves until a validated restore clears post-step poison."""
    if self.poisoned:
      raise TrainerPoisonedError(
        f"trainer is poisoned and requires a validated restore: {self._poison_reason}"
      )

  def _mark_poisoned(self, reason: str) -> None:
    self._poison_reason = reason

  def _validate_known_good_state(self) -> None:
    """Validate model and optimizer tensors before clearing post-step poison."""
    for name, parameter in self.model.named_parameters():
      if not torch.isfinite(parameter).all().item():
        raise TrainerValidationError(f"restored model parameter {name!r} is non-finite")
    for group_index, group in enumerate(self.optimizer.param_groups):
      for parameter in group["params"]:
        state = self.optimizer.state.get(parameter, {})
        for state_name, value in state.items():
          if isinstance(value, torch.Tensor):
            if not torch.isfinite(value).all().item():
              raise TrainerValidationError(
                f"restored optimizer state {group_index}:{state_name} is non-finite"
              )
            if value.ndim > 0 and value.shape != parameter.shape:
              raise TrainerValidationError(
                f"restored optimizer state {group_index}:{state_name} has wrong shape"
              )

  def clear_poison_after_validated_restore(self) -> None:
    """Clear poison only after checkpoint code has restored finite state.

    Checkpoint loading should restore model, optimizer, counters, and RNGs,
    then call this method.  It performs cheap finite/shape checks itself and
    refuses to clear poison if the candidate state is not known good.
    """
    self._validate_known_good_state()
    self._poison_reason = None

  @property
  def normalizers_frozen(self) -> bool:
    return (
      self.model.reference_normalizer.frozen
      and self.model.conditioning_normalizer.frozen
    )

  @property
  def normalization_update_ids(self) -> frozenset[Hashable]:
    """Identities already consumed by the explicit normalizer lifecycle."""
    return frozenset(self._normalization_update_ids)

  def freeze_normalizers(self) -> None:
    """Freeze both student statistics at a collection/evaluation boundary."""
    self.model.reference_normalizer.freeze()
    self.model.conditioning_normalizer.freeze()

  def unfreeze_normalizers(self) -> None:
    """Open the explicit training boundary for fresh-data statistics."""
    self.model.reference_normalizer.unfreeze()
    self.model.conditioning_normalizer.unfreeze()

  # Boundary names are intentionally explicit for runner integration.
  begin_collection = freeze_normalizers
  begin_evaluation = freeze_normalizers

  def begin_training(self, fresh_data: FreshTrainingData | None = None) -> None:
    """Open training and optionally consume one fresh raw batch exactly once."""
    self.unfreeze_normalizers()
    if fresh_data is not None:
      self.update_normalizers_from_new_data(fresh_data)

  def update_normalizers_from_new_data(
    self,
    fresh_data: FreshTrainingData | LabeledReplayBatch,
    *,
    update_id: Hashable | None = None,
  ) -> None:
    """Update normalizers once from a newly inserted raw batch.

    ``FreshTrainingData`` is preferred.  For convenience a batch plus explicit
    ``update_id`` is also accepted.  Omitting the identity is rejected rather
    than allowing an accidental repeated full-replay update.
    """
    if isinstance(fresh_data, FreshTrainingData):
      if update_id is not None:
        raise TrainerValidationError("update_id must not be supplied twice")
      batch = fresh_data.batch
      identity = fresh_data.update_id
    elif isinstance(fresh_data, LabeledReplayBatch):
      if update_id is None:
        raise TrainerValidationError("fresh replay data requires an explicit update_id")
      batch = fresh_data
      identity = update_id
    else:
      raise TrainerValidationError(
        "fresh_data must be FreshTrainingData or replay batch"
      )
    try:
      hash(identity)
    except TypeError as exc:
      raise TrainerValidationError("update_id must be hashable") from exc
    if (
      self.model.reference_normalizer.frozen
      or self.model.conditioning_normalizer.frozen
    ):
      raise TrainerValidationError(
        "normalizers are frozen; call begin_training before fresh-data updates"
      )
    if identity in self._normalization_update_ids:
      raise TrainerValidationError(
        f"normalization update_id {identity!r} was already used"
      )
    self._validate_batch(batch)
    self.model.reference_normalizer.update(batch.reference.detach())
    self.model.conditioning_normalizer.update(batch.conditioning.detach())
    self._normalization_update_ids.add(identity)

  def generator_states(self) -> dict[str, torch.Tensor]:
    """Return cloned explicit RNG states for a later lifecycle checkpoint."""
    return {
      "replay": self.replay_generator.get_state().clone(),
      "latent": self.latent_generator.get_state().clone(),
    }

  def set_generator_states(self, states: dict[str, torch.Tensor]) -> None:
    """Atomically restore both explicit RNG streams after full validation.

    ``torch.Generator.set_state`` is exercised on temporary generators first,
    including its device-specific state-length validation.  Original streams
    are therefore untouched when either supplied state is malformed.
    """
    if not isinstance(states, dict) or set(states) != {"replay", "latent"}:
      raise TrainerValidationError("generator states must contain replay and latent")
    candidates: dict[str, torch.Generator] = {}
    for name, generator in (
      ("replay", self.replay_generator),
      ("latent", self.latent_generator),
    ):
      state = states[name]
      if (
        not isinstance(state, torch.Tensor)
        or state.dtype != torch.uint8
        or state.ndim != 1
      ):
        raise TrainerValidationError(f"{name} generator state must be a uint8 vector")
      candidate = torch.Generator(device=_generator_device(generator))
      try:
        candidate.set_state(state.detach().clone().cpu())
      except (RuntimeError, TypeError) as exc:
        raise TrainerValidationError(f"invalid {name} generator state: {exc}") from exc
      candidates[name] = candidate

    # Both candidates accepted their states, so the commit should not reject.
    # Keep rollback for an unusual backend failure anyway.
    old = self.generator_states()
    try:
      self.replay_generator.set_state(candidates["replay"].get_state())
      self.latent_generator.set_state(candidates["latent"].get_state())
    except (RuntimeError, TypeError) as exc:
      try:
        self.replay_generator.set_state(old["replay"])
        self.latent_generator.set_state(old["latent"])
      except (RuntimeError, TypeError) as rollback_exc:
        self._mark_poisoned(f"RNG restore rollback failed: {rollback_exc}")
        raise TrainerPoisonedError(
          f"RNG restore failed and rollback failed: {rollback_exc}"
        ) from exc
      raise TrainerValidationError(f"RNG restore failed: {exc}") from exc

  def _validate_batch(self, batch: LabeledReplayBatch) -> None:
    if not isinstance(batch, LabeledReplayBatch):
      raise TrainerValidationError("batch must be a LabeledReplayBatch")
    if batch.schema != self.model.schema:
      raise TrainerValidationError("replay batch schema does not match student model")
    if batch.batch_size <= 0:
      raise TrainerValidationError("training batch must be non-empty")
    try:
      self.replay._validate_batch(batch)  # noqa: SLF001 - same owned contract
    except ReplayValidationError as exc:
      raise TrainerValidationError(str(exc)) from exc

  def _noise(
    self, shape: torch.Size, dtype: torch.dtype, device: torch.device
  ) -> torch.Tensor:
    generator_device = _generator_device(self.latent_generator)
    noise = torch.randn(
      shape,
      dtype=dtype,
      device=generator_device,
      generator=self.latent_generator,
    )
    return noise.to(device=device)

  def _loss_for_batch(self, batch: LabeledReplayBatch) -> VaeLoss:
    reference, conditioning = batch.reference, batch.conditioning
    if reference.device != next(self.model.parameters()).device:
      raise TrainerValidationError(
        f"replay batch uses {reference.device}, model uses "
        f"{next(self.model.parameters()).device}"
      )
    if reference.dtype != next(self.model.parameters()).dtype:
      raise TrainerValidationError("replay and model dtypes must match")
    mu, logvar = self.model.encode(reference)
    latent = self.model.sample_latent(
      mu, logvar, self._noise(mu.shape, mu.dtype, mu.device)
    )
    prediction = self.model.decode(latent, conditioning)
    return vae_loss(
      prediction,
      batch.teacher_action,
      mu,
      logvar,
      beta=self.config.beta,
    )

  def _check_gradients(self) -> float:
    squared_norm = torch.zeros((), device=next(self.model.parameters()).device)
    for parameter in self.model.parameters():
      if parameter.grad is None:
        continue
      _finite("gradient", parameter.grad)
      squared_norm = squared_norm + parameter.grad.detach().square().sum()
    _finite("gradient norm", squared_norm)
    norm = torch.sqrt(squared_norm)
    _finite("gradient norm", norm)
    return float(norm.item())

  def train_update(
    self, microbatches: Sequence[LabeledReplayBatch] | None = None
  ) -> TrainingUpdate:
    """Run one accumulated Adam update from raw replay minibatches.

    ``microbatches`` is an optional deterministic test/lifecycle seam.  When
    supplied it must contain exactly ``accumulation_steps`` validated batches;
    otherwise batches are sampled from replay with the explicit replay
    generator.
    """
    self.assert_healthy()
    if len(self.replay) == 0 and microbatches is None:
      raise TrainerValidationError("cannot train from an empty replay")
    if microbatches is not None:
      if len(microbatches) != self.config.accumulation_steps:
        raise TrainerValidationError(
          "explicit microbatches must match accumulation_steps"
        )
    self.model.train(True)
    self.optimizer.zero_grad(set_to_none=True)
    total = reconstruction = kl = 0.0
    samples = 0
    optimizer_called = False
    try:
      for index in range(self.config.accumulation_steps):
        batch = (
          microbatches[index]
          if microbatches is not None
          else self.replay.sample(
            self.config.minibatch_size,
            generator=self.replay_generator,
          )
        )
        self._validate_batch(batch)
        result = self._loss_for_batch(batch)
        _finite("loss", result.total)
        scaled = result.total / self.config.accumulation_steps
        _finite("scaled loss", scaled)
        scaled.backward()
        total += float(result.total.detach().item())
        reconstruction += float(result.reconstruction.detach().item())
        kl += float(result.kl.detach().item())
        samples += batch.batch_size
      gradient_norm = self._check_gradients()
      optimizer_called = True
      self.optimizer.step()
      for parameter in self.model.parameters():
        _finite("updated parameter", parameter, skipped=False)
    except Exception as exc:
      # This covers schema/device/dtype/runtime failures after any prior
      # microbatch, while preserving the original exception and traceback.
      self.optimizer.zero_grad(set_to_none=True)
      if optimizer_called:
        reason = f"post-step failure: {exc}"
        self._mark_poisoned(reason)
        raise TrainerPoisonedError(
          f"{reason}; trainer is poisoned and requires a validated restore"
        ) from exc
      if isinstance(exc, ModelValidationError):
        raise TrainerNumericalError(f"{exc}; optimizer step was skipped") from exc
      raise
    self.optimizer_steps += 1
    self.microbatches_seen += self.config.accumulation_steps
    self.samples_seen += samples
    count = float(self.config.accumulation_steps)
    return TrainingUpdate(
      optimizer_step=self.optimizer_steps,
      microbatches=self.config.accumulation_steps,
      samples=samples,
      total_loss=total / count,
      reconstruction=reconstruction / count,
      kl=kl / count,
      gradient_norm=gradient_norm,
    )


# Concise names for callers choosing either architecture terminology.
DistillationTrainer = VaeDistillationTrainer
SupervisedVAETrainer = VaeDistillationTrainer

__all__ = [
  "DistillationTrainer",
  "FreshTrainingData",
  "SupervisedVAETrainer",
  "TrainerHealth",
  "TrainerNumericalError",
  "TrainerPoisonedError",
  "TrainerState",
  "TrainerValidationError",
  "TrainingUpdate",
  "VaeDistillationTrainer",
]
