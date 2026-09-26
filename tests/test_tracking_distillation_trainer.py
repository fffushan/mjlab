"""Focused CPU tests for the pure M3 supervised trainer."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from mjlab.tasks.tracking.distillation.model import ConditionalVAE
from mjlab.tasks.tracking.distillation.observations import PackedObservationBatch
from mjlab.tasks.tracking.distillation.storage import (
  LabeledReplayBatch,
  LabeledReplayBuffer,
)
from mjlab.tasks.tracking.distillation.trainer import (
  FreshTrainingData,
  TrainerNumericalError,
  TrainerPoisonedError,
  TrainerValidationError,
  VaeDistillationTrainer,
)
from mjlab.tasks.tracking.distillation.training_config import TrainingConfig
from mjlab.tasks.tracking.distillation.vae_config import (
  DEFAULT_SCHEMA,
  ModelSettings,
  make_schema,
)


def make_batch(size: int = 16, seed: int = 0) -> LabeledReplayBatch:
  generator = torch.Generator().manual_seed(seed)
  reference = torch.randn(size, 68, generator=generator)
  conditioning = torch.randn(size, 99, generator=generator)
  # A bounded, deterministic target keeps this a supervised CPU-only test.
  teacher_action = torch.tanh(reference[:, :31])
  ids = torch.arange(size, dtype=torch.int64)
  return LabeledReplayBatch(
    observations=PackedObservationBatch(reference, conditioning, DEFAULT_SCHEMA),
    teacher_action=teacher_action,
    motion_id=ids,
    teacher_id=torch.zeros_like(ids),
    reference_frame=ids,
    episode_id=torch.zeros_like(ids),
    collector_iteration=torch.zeros_like(ids),
  )


def make_trainer(
  batch: LabeledReplayBatch,
  *,
  accumulation_steps: int = 2,
  minibatch_size: int = 8,
  seed: int = 4,
) -> VaeDistillationTrainer:
  model = ConditionalVAE(settings=ModelSettings(hidden_dims=(16, 8)))
  replay = LabeledReplayBuffer(capacity=64, schema=DEFAULT_SCHEMA)
  replay.insert(batch)
  config = TrainingConfig(
    accumulation_steps=accumulation_steps,
    minibatch_size=minibatch_size,
  )
  return VaeDistillationTrainer(model, replay, config, seed=seed)


def test_config_preserves_accepted_defaults_and_rejects_invalid_values() -> None:
  config = TrainingConfig()
  assert config.learning_rate == pytest.approx(5e-4)
  assert config.beta == pytest.approx(0.01)
  assert config.accumulation_steps == 15
  with pytest.raises(ValueError, match="positive"):
    TrainingConfig(learning_rate=0.0)
  with pytest.raises(ValueError, match="finite"):
    TrainingConfig(beta=float("nan"))
  with pytest.raises(ValueError, match="positive integer"):
    TrainingConfig(accumulation_steps=0)


def test_fresh_data_normalization_is_explicit_once_and_boundary_freezes() -> None:
  batch = make_batch()
  trainer = make_trainer(batch)
  fresh = FreshTrainingData(batch, "collector-7")
  trainer.begin_training(fresh)
  assert (
    float(trainer.model.reference_normalizer.state_dict()["count"]) == batch.batch_size
  )
  assert (
    float(trainer.model.conditioning_normalizer.state_dict()["count"])
    == batch.batch_size
  )
  trainer.begin_collection()
  assert trainer.normalizers_frozen
  with pytest.raises(TrainerValidationError, match="frozen"):
    trainer.update_normalizers_from_new_data(fresh)
  trainer.begin_training()
  with pytest.raises(TrainerValidationError, match="already used"):
    trainer.update_normalizers_from_new_data(fresh)


def test_inference_does_not_update_normalizer_statistics() -> None:
  batch = make_batch(4)
  trainer = make_trainer(batch)
  before = trainer.model.reference_normalizer.state_dict()["count"].clone()
  trainer.model.mean_inference(batch.reference, batch.conditioning)
  assert trainer.model.reference_normalizer.count == before
  trainer.begin_collection()
  assert trainer.normalizers_frozen


def test_sampled_training_loss_decreases_and_counters_are_exact() -> None:
  batch = make_batch(24)
  trainer = make_trainer(batch, accumulation_steps=2, minibatch_size=8)
  trainer.begin_training(FreshTrainingData(batch, 0))
  trainer.freeze_normalizers()
  losses = [trainer.train_update().total_loss for _ in range(24)]
  assert all(torch.isfinite(torch.tensor(losses)))
  assert losses[-1] < losses[0]
  assert trainer.optimizer_steps == 24
  assert trainer.microbatches_seen == 48
  assert trainer.samples_seen == 24 * 2 * 8


def test_accumulation_matches_one_equivalent_batch_with_controlled_rng() -> None:
  batch = make_batch(4, seed=9)
  first = replace(
    batch,
    observations=PackedObservationBatch(
      batch.reference[:2], batch.conditioning[:2], DEFAULT_SCHEMA
    ),
    teacher_action=batch.teacher_action[:2],
    motion_id=batch.motion_id[:2],
    teacher_id=batch.teacher_id[:2],
    reference_frame=batch.reference_frame[:2],
    episode_id=batch.episode_id[:2],
    collector_iteration=batch.collector_iteration[:2],
  )
  second = replace(
    batch,
    observations=PackedObservationBatch(
      batch.reference[2:], batch.conditioning[2:], DEFAULT_SCHEMA
    ),
    teacher_action=batch.teacher_action[2:],
    motion_id=batch.motion_id[2:],
    teacher_id=batch.teacher_id[2:],
    reference_frame=batch.reference_frame[2:],
    episode_id=batch.episode_id[2:],
    collector_iteration=batch.collector_iteration[2:],
  )
  accumulated = make_trainer(batch, accumulation_steps=2, minibatch_size=2, seed=1)
  equivalent = make_trainer(batch, accumulation_steps=1, minibatch_size=4, seed=1)
  equivalent.model.load_state_dict(accumulated.model.state_dict())
  accumulated.config = TrainingConfig(accumulation_steps=2, minibatch_size=2)
  equivalent.config = TrainingConfig(accumulation_steps=1, minibatch_size=4)
  accumulated.latent_generator.set_state(equivalent.latent_generator.get_state())
  accumulated.train_update([first, second])
  equivalent.train_update([batch])
  for left, right in zip(
    accumulated.model.parameters(), equivalent.model.parameters(), strict=True
  ):
    torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)


def test_nonfinite_loss_fails_before_step_and_clears_gradients() -> None:
  batch = make_batch(4)
  trainer = make_trainer(batch, accumulation_steps=1, minibatch_size=4)
  with torch.no_grad():
    next(trainer.model.encoder.parameters()).fill_(float("inf"))
  with pytest.raises(TrainerNumericalError, match="optimizer step was skipped"):
    trainer.train_update()
  assert trainer.optimizer_steps == 0
  assert all(parameter.grad is None for parameter in trainer.model.parameters())


def test_explicit_microbatch_count_is_validated() -> None:
  batch = make_batch(4)
  trainer = make_trainer(batch, accumulation_steps=2)
  with pytest.raises(TrainerValidationError, match="match accumulation_steps"):
    trainer.train_update([batch])


def test_partial_microbatch_validation_clears_prior_gradients() -> None:
  batch = make_batch(4)
  trainer = make_trainer(batch, accumulation_steps=2, minibatch_size=2)
  alternate_schema = make_schema("gravity", tuple(f"other_{i}" for i in range(31)))
  invalid = replace(
    batch,
    observations=PackedObservationBatch(
      batch.reference, batch.conditioning, alternate_schema
    ),
  )
  with pytest.raises(TrainerValidationError, match="schema"):
    trainer.train_update([batch, invalid])
  assert trainer.optimizer_steps == 0
  assert trainer.microbatches_seen == 0
  assert all(parameter.grad is None for parameter in trainer.model.parameters())


def test_two_generator_restore_validates_atomically() -> None:
  trainer = make_trainer(make_batch(4))
  before = trainer.generator_states()
  replacement = torch.Generator().manual_seed(123).get_state()
  invalid_latent = torch.zeros(1, dtype=torch.uint8)
  with pytest.raises(TrainerValidationError, match="latent generator state"):
    trainer.set_generator_states({"replay": replacement, "latent": invalid_latent})
  after = trainer.generator_states()
  torch.testing.assert_close(before["replay"], after["replay"])
  torch.testing.assert_close(before["latent"], after["latent"])


def test_post_step_failure_poison_requires_validated_restore() -> None:
  trainer = make_trainer(make_batch(4), accumulation_steps=1, minibatch_size=4)
  original_step = trainer.optimizer.step

  def poison_step(*args: object, **kwargs: object) -> None:
    original_step(*args, **kwargs)
    with torch.no_grad():
      next(trainer.model.parameters()).fill_(float("nan"))

  trainer.optimizer.step = poison_step
  with pytest.raises(TrainerPoisonedError, match="poisoned") as poisoned:
    trainer.train_update()
  message = str(poisoned.value)
  assert "non-finite updated parameter" in message
  assert "optimizer step already ran" in message
  assert "optimizer step was skipped" not in message
  assert trainer.optimizer_steps == 0
  assert trainer.poisoned
  assert not trainer.health_check().healthy
  with pytest.raises(TrainerPoisonedError, match="validated restore"):
    trainer.assert_healthy()
  with torch.no_grad():
    next(trainer.model.parameters()).zero_()
  trainer.clear_poison_after_validated_restore()
  assert trainer.is_healthy
  trainer.assert_healthy()
