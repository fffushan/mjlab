"""CPU-only trainer mechanics and bounded-resume regression gates."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from mjlab.tasks.tracking.diffusion import DiffusionContract, DiffusionSchedule
from mjlab.tasks.tracking.diffusion.trainer import (
  DiffusionTrainer,
  ExponentialMovingAverage,
  TrainerError,
)
from mjlab.tasks.tracking.diffusion.training_config import (
  TrainingConfig,
  TrainingConfigError,
  ema_decay,
  warmup_cosine_learning_rate,
)


class SyntheticWindows(Dataset[tuple[Tensor, Tensor, int]]):
  """Small well-formed token fixtures with explicit phase sampling weights."""

  def __init__(self, count: int = 5, *, identity: str = "synthetic-v1") -> None:
    generator = torch.Generator().manual_seed(31)
    self.tokens = torch.randn((count, 41, 231), generator=generator)
    self.weights = np.asarray([1.0, 1.0, 2.0, 2.0, 1.0][:count])
    self.dataset_identity = {
      "directory": identity,
      "assignments_hash": "a" * 64,
      "split_coverage": {"train": count, "validation": 0, "test": 0},
      "projection_hashes": {
        "matrix": "b" * 64,
        "statistics": "c" * 64,
      },
    }

  def __len__(self) -> int:
    return len(self.tokens)

  def __getitem__(self, index: int) -> tuple[Tensor, Tensor, int]:
    return (
      self.tokens[index],
      torch.tensor(self.weights[index], dtype=torch.float32),
      index,
    )


class TinyDenoiser(nn.Module):
  """Fast bidirectional pointwise model for CPU trainer tests."""

  def __init__(self) -> None:
    super().__init__()
    self.linear = nn.Linear(231, 231)

  def forward(self, tokens: Tensor, step_ids: Tensor) -> Tensor:
    del step_ids
    return self.linear(tokens)


class ConstantDenoiser(nn.Module):
  """Learnable constant used for a deterministic one-window overfit gate."""

  def __init__(self) -> None:
    super().__init__()
    self.value = nn.Parameter(torch.zeros((1, 41, 231)))

  def forward(self, tokens: Tensor, step_ids: Tensor) -> Tensor:
    del step_ids
    return self.value.expand(tokens.shape[0], -1, -1)


class NaNDenoiser(TinyDenoiser):
  """Model used to prove divergence aborts before checkpoint writes."""

  def forward(self, tokens: Tensor, step_ids: Tensor) -> Tensor:
    del step_ids
    return torch.full_like(tokens, float("nan"))


def _config(**overrides: object) -> TrainingConfig:
  values: dict[str, Any] = {
    "epochs": 2,
    "max_updates": 2,
    "effective_batch_size": 4,
    "microbatch_size": 2,
    "gradient_accumulation_steps": 2,
    "mixed_precision": "fp32",
    "warmup_updates": 1,
    "eval_split": "validation",
  }
  values.update(overrides)
  return TrainingConfig(**values)


def _trainer(
  output: Path,
  *,
  model: nn.Module | None = None,
  dataset: Any | None = None,
  config: TrainingConfig | None = None,
  contract: DiffusionContract | None = None,
  resume: Path | None = None,
) -> DiffusionTrainer:
  train = dataset or SyntheticWindows()
  return DiffusionTrainer(
    config=config or _config(),
    contract=contract or DiffusionContract(),
    schedule=DiffusionSchedule.build(),
    model=model or TinyDenoiser(),
    train_dataset=train,
    eval_datasets={"validation": SyntheticWindows(2)},
    output_dir=output,
    device="cpu",
    resume=resume,
  )


def test_config_defaults_to_validation_and_rejects_test_selection() -> None:
  assert TrainingConfig().eval_split == "validation"
  with pytest.raises(TrainingConfigError, match="test split.*never drive"):
    TrainingConfig(eval_split="test").validate()


def test_best_checkpoint_records_the_validation_selection_split(tmp_path) -> None:
  trainer = _trainer(
    tmp_path,
    config=_config(epochs=1, max_updates=3, eval_split="validation"),
  )
  result = trainer.train()
  checkpoint = result.checkpoint_paths["best"]
  payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
  assert payload["selection_split"] == "validation"


def test_config_rejects_unbounded_invocation_and_contract_drift() -> None:
  with pytest.raises(TrainingConfigError, match="explicit max_updates"):
    TrainingConfig().validate(explicit_budget=True)
  with pytest.raises(TrainingConfigError, match="model.layers"):
    TrainingConfig.from_mapping({"model": {"layers": 5}})
  with pytest.raises(TrainingConfigError, match="loss_reduction"):
    TrainingConfig(loss_reduction="epsilon").validate()
  config = TrainingConfig.from_mapping({"ema": {"power": 0.75, "max_decay": 0.9999}})
  assert config.as_dict()["ema_power"] == 0.75
  with pytest.raises(TrainingConfigError, match="ema.power"):
    TrainingConfig.from_mapping({"ema": {"power": 0.5}})
  with pytest.raises(TrainingConfigError, match="resolved updates"):
    TrainingConfig().validate_resolved_updates(201)
  assert TrainingConfig().validate_resolved_updates(201, allow_long_run=True) == 201_000

  assert warmup_cosine_learning_rate(
    0, total_updates=10, learning_rate=1.0, warmup_updates=2
  ) == pytest.approx(0.5)
  assert warmup_cosine_learning_rate(
    1, total_updates=10, learning_rate=1.0, warmup_updates=2
  ) == pytest.approx(1.0)
  assert warmup_cosine_learning_rate(
    9, total_updates=10, learning_rate=1.0, warmup_updates=2
  ) == pytest.approx(0.0)
  for update in (1, 2, 17):
    assert ema_decay(update) == pytest.approx(
      min(0.9999, 1.0 - (1.0 + update) ** -0.75)
    )


def test_tiny_fixed_window_overfits_x0_objective(tmp_path) -> None:
  target = torch.full((1, 41, 231), 0.25)
  config = TrainingConfig(
    epochs=200,
    max_updates=100,
    effective_batch_size=1,
    microbatch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=0.05,
    warmup_updates=0,
    mixed_precision="fp32",
    eval_split="validation",
  )
  trainer = _trainer(
    tmp_path,
    model=ConstantDenoiser(),
    dataset=torch.utils.data.TensorDataset(target),
    config=config,
  )
  result = trainer.train()
  assert result.final_train_loss < 1.0e-3


def test_accumulation_partial_group_and_epoch_sample_budget(tmp_path) -> None:
  config = _config(
    epochs=1,
    max_updates=None,
    epoch_sample_budget=5,
  )
  trainer = _trainer(tmp_path, config=config)
  sampler = trainer._sampler(0)
  assert sampler is not None
  assert getattr(sampler, "num_samples", None) == 5
  result = trainer.train()
  assert result.global_step == 2
  updates = [
    json.loads(line)
    for line in result.metrics_path.read_text().splitlines()
    if json.loads(line)["kind"] == "update"
  ]
  assert [item["accumulated_microbatches"] for item in updates] == [2.0, 1.0]
  assert all("learning_rate" in item for item in updates)
  report = json.loads((tmp_path / "train-report.json").read_text())
  assert report["effective_batch_size"] == 4
  assert report["resume_reproducibility"]["measured"] is False


def test_empty_evaluation_split_is_not_a_best_checkpoint(tmp_path) -> None:
  trainer = _trainer(
    tmp_path,
    config=_config(epochs=1, max_updates=1, eval_split="validation"),
  )
  trainer.eval_datasets = {
    "validation": torch.utils.data.TensorDataset(
      torch.empty((0, 41, 231), dtype=torch.float32)
    )
  }
  result = trainer.train()
  assert result.best_validation_loss is None
  assert "best" not in result.checkpoint_paths
  report = json.loads((tmp_path / "train-report.json").read_text())
  assert report["evaluation_split"] == "validation"
  assert report["best_evaluation_split"] is None
  assert report["best_evaluation_loss"] is None


def test_resume_restores_fp32_check_state(tmp_path) -> None:
  first = _trainer(tmp_path / "first").train()
  resumed = _trainer(
    tmp_path / "resume",
    resume=first.checkpoint_paths["last"],
  )
  assert resumed._fp32_checked is True


def test_nan_or_inf_aborts_without_writing_checkpoint(tmp_path) -> None:
  trainer = _trainer(tmp_path, model=NaNDenoiser())
  with pytest.raises(TrainerError, match="finite|NaN|Inf"):
    trainer.train()
  assert not (tmp_path / "checkpoint-last.pt").exists()
  assert not (tmp_path / "checkpoint-best.pt").exists()


def test_resume_rejects_mismatched_dataset_identity(tmp_path) -> None:
  first_dir = tmp_path / "first"
  result = _trainer(first_dir).train()
  changed = SyntheticWindows(identity="different-dataset")
  with pytest.raises(TrainerError, match="dataset identity"):
    _trainer(
      tmp_path / "resume",
      dataset=changed,
      resume=result.checkpoint_paths["last"],
    )


def test_resume_rejects_mismatched_contract_and_projection_identity(tmp_path) -> None:
  first_dir = tmp_path / "first"
  result = _trainer(first_dir).train()
  changed_contract = replace(DiffusionContract(), contract_sha256="d" * 64)
  with pytest.raises(TrainerError, match="contract identity"):
    _trainer(
      tmp_path / "contract-resume",
      contract=changed_contract,
      resume=result.checkpoint_paths["last"],
    )
  changed_projection = SyntheticWindows()
  changed_projection.dataset_identity["projection_hashes"] = {
    "matrix": "e" * 64,
    "statistics": "f" * 64,
  }
  with pytest.raises(TrainerError, match="dataset identity"):
    _trainer(
      tmp_path / "projection-resume",
      dataset=changed_projection,
      resume=result.checkpoint_paths["last"],
    )


def test_ema_shadow_updates_with_analytic_decay() -> None:
  model = TinyDenoiser()
  ema = ExponentialMovingAverage(model)
  before = {name: value.clone() for name, value in model.state_dict().items()}
  with torch.no_grad():
    for value in model.parameters():
      value.add_(1.0)
  decay = ema.update(model)
  assert decay == pytest.approx(ema_decay(1))
  for name, value in ema.shadow.items():
    if torch.is_floating_point(value):
      expected = decay * before[name] + (1.0 - decay) * model.state_dict()[name]
      assert torch.allclose(value, expected)
