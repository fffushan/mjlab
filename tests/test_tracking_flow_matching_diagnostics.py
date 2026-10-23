"""CPU-only gates for the optional flow-matching sidecar recorder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

from mjlab.tasks.tracking.diffusion.contract import DiffusionContract
from mjlab.tasks.tracking.diffusion.state import HYBRID_STATE
from mjlab.tasks.tracking.diffusion.training_config import TrainingConfig
from mjlab.tasks.tracking.flow_matching import FlowTrainer, FlowTrainingConfig
from mjlab.tasks.tracking.flow_matching.diagnostics import (
  ENV_DIRECTORY,
  ENV_RING_SIZE,
  ENV_TRIGGER_FACTOR,
  FlowDiagnostics,
  FlowDiagnosticsError,
  _bin_indices,
  channel_groups,
  gradient_norms_by_block,
)


class SyntheticWindows(torch.utils.data.Dataset[tuple[Tensor, Tensor, int]]):
  def __init__(self, count: int = 8) -> None:
    generator = torch.Generator().manual_seed(17)
    self.tokens = torch.randn((count, 41, 167), generator=generator)
    self.dataset_identity = {
      "directory": "synthetic-diagnostics-v1",
      "assignments_hash": "a" * 64,
      "split_coverage": {"train": count, "validation": 0, "test": 0},
      "projection_hashes": {"matrix": "b" * 64, "statistics": "c" * 64},
    }

  def __len__(self) -> int:
    return int(self.tokens.shape[0])

  def __getitem__(self, index: int) -> tuple[Tensor, Tensor, int]:
    return self.tokens[index], torch.tensor(1.0), int(index)


class TinyDenoiser(torch.nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.linear = torch.nn.Linear(167, 167)

  def forward(self, tokens: Tensor, step_ids: Tensor) -> Tensor:
    del step_ids
    return self.linear(tokens)


def _trainer(output: Path, **overrides: Any) -> FlowTrainer:
  values: dict[str, Any] = {
    "epochs": 1,
    "max_updates": 2,
    "effective_batch_size": 4,
    "microbatch_size": 2,
    "gradient_accumulation_steps": 2,
    "mixed_precision": "fp32",
    "warmup_updates": 1,
    "eval_split": "validation",
  }
  values.update(overrides)
  config = FlowTrainingConfig(training=TrainingConfig(**values))
  torch.manual_seed(config.training.seed)
  return FlowTrainer(
    config=config,
    contract=DiffusionContract(),
    model=TinyDenoiser(),
    train_dataset=SyntheticWindows(),
    eval_datasets={"validation": SyntheticWindows(2)},
    output_dir=output,
    device="cpu",
  )


def test_recorder_is_off_without_the_environment_variable(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  monkeypatch.delenv(ENV_DIRECTORY, raising=False)
  trainer = _trainer(tmp_path / "run")
  assert trainer.diagnostics is None
  trainer.train()
  assert not (tmp_path / "run" / "updates.jsonl").exists()


def test_recorder_requires_a_valid_ring_size(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  monkeypatch.setenv(ENV_DIRECTORY, str(tmp_path / "diag"))
  monkeypatch.setenv(ENV_RING_SIZE, "0")
  with pytest.raises(FlowDiagnosticsError, match="DIAGNOSTICS_RING"):
    _trainer(tmp_path / "run")


def test_recorder_writes_one_row_per_update_with_blocks_and_levels(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  monkeypatch.setenv(ENV_DIRECTORY, str(tmp_path / "diag"))
  trainer = _trainer(tmp_path / "run")
  assert isinstance(trainer.diagnostics, FlowDiagnostics)
  trainer.train()
  rows = [
    json.loads(line)
    for line in (tmp_path / "diag" / "updates.jsonl").read_text().splitlines()
  ]
  assert len(rows) == 2
  row = rows[0]
  assert row["microbatches"] == 2
  assert row["gradient_norm"] > 0.0
  assert row["gradient_norm_block/linear"] > 0.0
  assert "loss_channel_group/root_position" in row
  assert "loss_channel_group/latent" in row
  assert "loss_by_level/state_level_0" in row
  assert "loss_by_row/40" in row
  assert row["microbatch_loss"] == row["loss"]


def test_trigger_writes_a_capture_and_a_checkpoint(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  # A permissive factor plus a one-update baseline triggers on the third
  # update, which is the first one the baseline window allows.
  monkeypatch.setenv(ENV_DIRECTORY, str(tmp_path / "diag"))
  monkeypatch.setenv(ENV_TRIGGER_FACTOR, "1e-9")
  trainer = _trainer(tmp_path / "run", max_updates=3)
  trainer.diagnostics.baseline_window = 1  # noqa: SLF001
  trainer.train()
  captures = sorted((tmp_path / "diag" / "captures").glob("capture-*.npz"))
  # Every update after the first qualified baseline crosses a 1e-9 factor, so
  # the recorder dumps once per such update.
  assert len(captures) >= 1
  payload = np.load(captures[0])
  assert payload["window_indices"].size > 0
  assert payload["state_levels"].shape[1] == 41
  assert payload["generator_state"].ndim == 2
  assert payload["generator_state"].shape[0] == payload["microbatch_rows"].size
  assert int(payload["microbatch_rows"].sum()) == payload["window_indices"].size
  assert payload["window_loss"].size == payload["window_indices"].size
  triggers = sorted((tmp_path / "run").glob("checkpoint-trigger-*.pt"))
  assert triggers
  # The checkpoint budget bounds a runaway arm: it keeps writing captures but
  # stops spending 307 MB per update.
  assert len(triggers) <= trainer.diagnostics.max_checkpoints
  sidecar = json.loads(captures[0].with_suffix(".json").read_text())
  assert sidecar["reason"] == "loss_trigger"
  # The trigger checkpoint must hold the weights that produced the triggering
  # batch, which is why it is written before the optimizer step.  Its step is
  # therefore the update index the row records, not that index plus one.
  checkpoint = torch.load(triggers[0], map_location="cpu", weights_only=False)
  assert checkpoint["global_step"] == sidecar["update_index"]


def test_channel_groups_follow_the_names_and_fall_back_on_a_mismatch() -> None:
  groups = channel_groups(HYBRID_STATE.slices, state_dimension=135, token_dimension=167)
  assert groups[0].name == "root_position"
  assert groups[-1].name == "latent"
  assert groups[-1].start == 135 and groups[-1].stop == 167

  # A layout that does not cover the state dimension must fall back, so a
  # contract change cannot silently mislabel the groups.
  fallback = channel_groups(
    {"root_rot6d": slice(0, 6)}, state_dimension=135, token_dimension=167
  )
  assert [group.name for group in fallback] == ["state", "latent"]


def test_bin_indices_isolate_the_clean_row() -> None:
  levels = torch.tensor([[0, 1, 500, 1000]], dtype=torch.long)
  bins = _bin_indices(levels, level_max=1000, bins=10)
  assert bins[0, 0].item() == 0
  assert bins[0, 1].item() == 1
  # Bucket 0 is clean and the noised levels fill buckets 1..bins.
  assert bins[0, 3].item() == 10
  assert bins.max().item() == 10


def test_gradient_norms_by_block_splits_encoder_layers() -> None:
  class Wrapper(torch.nn.Module):
    def __init__(self) -> None:
      super().__init__()
      self.input_projection = torch.nn.Linear(4, 4)
      self.encoder = torch.nn.Module()
      self.encoder.layers = torch.nn.ModuleList(
        [torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)]
      )

  model = Wrapper()
  model.input_projection.weight.grad = torch.ones_like(model.input_projection.weight)
  model.encoder.layers[1].weight.grad = torch.full((4, 4), 2.0, dtype=torch.float32)
  norms = gradient_norms_by_block(model)
  # A 4x4 all-ones gradient has L2 norm 4; an all-twos gradient has 8.
  assert norms == pytest.approx({"encoder.layers.1": 8.0, "input_projection": 4.0})
