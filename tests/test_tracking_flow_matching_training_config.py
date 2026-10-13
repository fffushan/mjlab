"""Flow training configuration gates for the fixed x0 linear-flow contract."""

from __future__ import annotations

import pytest

from mjlab.tasks.tracking.diffusion.training_config import TrainingConfig
from mjlab.tasks.tracking.flow_matching import (
  FlowTrainingConfig,
  FlowTrainingConfigError,
  LinearNoiseProcess,
)


def test_default_config_identifies_the_linear_process() -> None:
  config = FlowTrainingConfig()
  config.validate()
  process = config.process
  assert process == LinearNoiseProcess()
  assert process.as_dict() == {
    "family": "x0_linear_flow",
    "prediction_type": "x0",
    "path": "linear_gaussian",
    "loss": "x0_mse",
    "time_encoding": "integer_noise_fraction",
    "max_level": 1000,
    "embedding_count": 1001,
  }
  assert config.sha256() == FlowTrainingConfig().sha256()
  assert len(config.sha256()) == 64


def test_nonzero_history_dropout_is_rejected() -> None:
  config = FlowTrainingConfig(
    training=TrainingConfig(history_state_dropout_probability=0.5)
  )
  with pytest.raises(FlowTrainingConfigError, match="history dropout"):
    config.validate()


def test_nonzero_history_dropout_is_rejected_from_mapping() -> None:
  with pytest.raises(FlowTrainingConfigError, match="history dropout"):
    FlowTrainingConfig.from_mapping(
      {"ablation": {"history_state_dropout_probability": 0.25}}
    )
  with pytest.raises(FlowTrainingConfigError, match="history dropout"):
    FlowTrainingConfig(
      training=TrainingConfig(history_state_dropout_probability=1.0)
    ).as_dict()
  # A top-level ablation section is never read once training is nested, so the
  # mixed schema is rejected instead of silently ignoring the dropout.
  with pytest.raises(FlowTrainingConfigError, match="ablation"):
    FlowTrainingConfig.from_mapping(
      {
        "training": {"epochs": 1},
        "ablation": {"history_state_dropout_probability": 0.25},
      }
    )


def test_non_uniform_level_sampling_and_block_loss_are_rejected() -> None:
  with pytest.raises(FlowTrainingConfigError, match="uniform integer levels"):
    FlowTrainingConfig(
      training=TrainingConfig(level_sampling="logit_normal")
    ).validate()
  with pytest.raises(FlowTrainingConfigError, match="plain channel-mean"):
    FlowTrainingConfig(training=TrainingConfig(state_loss_weight=0.5)).validate()
  with pytest.raises(FlowTrainingConfigError, match="plain channel-mean"):
    FlowTrainingConfig(training=TrainingConfig(latent_loss_weight=2.0)).validate()


def test_max_level_selects_the_process_identity() -> None:
  small = FlowTrainingConfig(max_level=100)
  assert small.process.max_level == 100
  assert small.process.embedding_count == 101
  assert small.as_dict()["max_level"] == 100
  assert small.sha256() != FlowTrainingConfig().sha256()
  with pytest.raises(FlowTrainingConfigError, match="max_level"):
    FlowTrainingConfig(max_level=0)


def test_flat_and_nested_mappings_agree() -> None:
  shared = {
    "epochs": 2,
    "max_updates": 4,
    "microbatch_size": 2,
    "gradient_accumulation_steps": 2,
    "effective_batch_size": 4,
    "mixed_precision": "fp32",
  }
  flat = FlowTrainingConfig.from_mapping(dict(shared))
  nested = FlowTrainingConfig.from_mapping(
    {"training": dict(shared), "process": {"max_level": 1000}}
  )
  assert flat.sha256() == nested.sha256()
  assert nested.training.epochs == 2


def test_mapping_round_trip_preserves_a_non_default_max_level() -> None:
  config = FlowTrainingConfig(max_level=100)
  restored = FlowTrainingConfig.from_mapping(config.as_dict())
  assert restored.max_level == 100
  assert restored.process.max_level == 100
  assert restored.sha256() == config.sha256()

  flat = FlowTrainingConfig.from_mapping({"max_level": 100})
  assert flat.max_level == 100
  with pytest.raises(FlowTrainingConfigError, match="max_level disagrees"):
    FlowTrainingConfig.from_mapping({"max_level": 100, "process": {"max_level": 1000}})
