"""CPU-only regression for explicit long-run authority at trainer construction."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from mjlab.tasks.tracking.diffusion.contract import DiffusionContract
from mjlab.tasks.tracking.diffusion.schedule import DiffusionSchedule
from mjlab.tasks.tracking.diffusion.trainer import DiffusionTrainer
from mjlab.tasks.tracking.diffusion.training_config import (
  TrainingConfig,
  TrainingConfigError,
)


class _TinyModel(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.linear = nn.Linear(167, 167)

  def forward(self, tokens: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
    del step_ids
    return self.linear(tokens)


def _trainer(output: Path, *, allow_long_run: bool) -> DiffusionTrainer:
  dataset = TensorDataset(torch.zeros((5, 41, 167)))
  config = TrainingConfig(
    epochs=100000,
    max_updates=400000,
    effective_batch_size=1,
    microbatch_size=1,
    gradient_accumulation_steps=1,
    mixed_precision="fp32",
  )
  return DiffusionTrainer(
    config=config,
    contract=DiffusionContract(),
    schedule=DiffusionSchedule.build(),
    model=_TinyModel(),
    train_dataset=dataset,
    eval_datasets={},
    output_dir=output,
    device="cpu",
    allow_long_run=allow_long_run,
  )


def test_trainer_preserves_explicit_long_run_permission(tmp_path: Path) -> None:
  trainer = _trainer(tmp_path / "allowed", allow_long_run=True)
  assert trainer._resolved_updates == 400000
  assert trainer.global_step == 0
  assert trainer.config.max_updates == 400000
  assert not list((tmp_path / "allowed").glob("*.pt"))


def test_trainer_refuses_long_run_without_permission(tmp_path: Path) -> None:
  with pytest.raises(TrainingConfigError, match="allow-long-run"):
    _trainer(tmp_path / "refused", allow_long_run=False)
  assert not (tmp_path / "refused").exists()


def test_config_roundtrip_does_not_grant_execution_permission() -> None:
  config = TrainingConfig(max_updates=400000)
  serialized = config.as_dict()
  loaded = TrainingConfig.from_mapping(serialized)
  assert loaded.max_updates == 400000
  assert loaded.sha256() == config.sha256()
  with pytest.raises(TrainingConfigError, match="allow-long-run"):
    loaded.validate()
  loaded.validate(allow_long_run=True)


def test_cli_passes_permission_to_real_trainer() -> None:
  import mjlab.scripts.diffusion as cli

  tree = ast.parse(Path(cli.__file__).read_text())
  calls = [
    node
    for node in ast.walk(tree)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Name)
    and node.func.id == "DiffusionTrainer"
  ]
  assert len(calls) == 1
  keywords = {item.arg: item.value for item in calls[0].keywords}
  value = keywords["allow_long_run"]
  assert isinstance(value, ast.Attribute)
  assert isinstance(value.value, ast.Name)
  assert value.value.id == "args"
  assert value.attr == "allow_long_run"
