"""Selection provenance must fail closed: the test split may never select a model.

The F1 fix made the trainer structurally unable to select on the test split.
These tests pin the two remaining routes by which malformed or legacy
provenance could still reach offline evaluation: an unknown split name, a
contradictory pair of split fields, and a legacy checkpoint whose best metric
came from the test split.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mjlab.scripts import diffusion
from mjlab.tasks.tracking.diffusion import (
  DiffusionContract,
  DiffusionSchedule,
  load_checkpoint,
  save_checkpoint,
)
from mjlab.tasks.tracking.diffusion.checkpoint import CheckpointError
from mjlab.tasks.tracking.diffusion.training_config import TrainingConfig


def _identity() -> dict[str, object]:
  return {
    "directory": "synthetic",
    "assignments_hash": "a" * 64,
    "split_coverage": {"train": 2, "validation": 0, "test": 0},
    "projection_hashes": {"matrix": "b" * 64, "statistics": "c" * 64},
  }


def _config() -> TrainingConfig:
  return TrainingConfig(
    epochs=1,
    max_updates=1,
    effective_batch_size=1,
    microbatch_size=1,
    gradient_accumulation_steps=1,
    mixed_precision="fp32",
  )


def _save(
  path: Path,
  *,
  best_metric_split: str | None = None,
  selection_split: str | None = None,
) -> None:
  save_checkpoint(
    path,
    model=nn.Linear(3, 2),
    ema=None,
    optimizer=None,
    scaler=None,
    config=_config(),
    contract=DiffusionContract(),
    schedule=DiffusionSchedule.build(),
    dataset_identity=_identity(),
    global_step=1,
    epoch=1,
    best_validation_loss=None,
    rng_state={"torch": torch.get_rng_state()},
    best_metric_split=best_metric_split,
    selection_split=selection_split,
  )


def test_save_rejects_unknown_selection_split(tmp_path: Path) -> None:
  with pytest.raises(CheckpointError):
    _save(tmp_path / "banana.pt", selection_split="banana")
  with pytest.raises(CheckpointError):
    _save(tmp_path / "whitespace.pt", selection_split="test ")
  with pytest.raises(CheckpointError):
    _save(tmp_path / "case.pt", selection_split="Test")


def test_save_rejects_contradictory_selection_splits(tmp_path: Path) -> None:
  with pytest.raises(CheckpointError):
    _save(
      tmp_path / "contradiction.pt",
      best_metric_split="validation",
      selection_split="test",
    )


def test_load_rejects_unknown_selection_split(tmp_path: Path) -> None:
  path = tmp_path / "tampered.pt"
  _save(path, best_metric_split="validation")
  payload = torch.load(path, map_location="cpu", weights_only=False)
  payload["selection_split"] = "banana"
  torch.save(payload, path)
  with pytest.raises(CheckpointError):
    load_checkpoint(path, model=nn.Linear(3, 2))


def test_load_rejects_contradictory_selection_splits(tmp_path: Path) -> None:
  path = tmp_path / "tampered-contradiction.pt"
  _save(path, best_metric_split="validation")
  payload = torch.load(path, map_location="cpu", weights_only=False)
  payload["selection_split"] = "test"
  torch.save(payload, path)
  with pytest.raises(CheckpointError):
    load_checkpoint(path, model=nn.Linear(3, 2))


def test_evaluate_offline_refuses_legacy_best_selected_on_test(
  tmp_path: Path, monkeypatch, capsys
) -> None:
  """A pre-F1 best checkpoint records the test split only in best_metric_split."""
  from mjlab.tasks.tracking.diffusion import checkpoint as checkpoint_module
  from mjlab.tasks.tracking.diffusion import model as model_module

  run = tmp_path / "legacy-best"
  run.mkdir()
  (run / "checkpoint-best.pt").write_bytes(b"placeholder")

  class TinyModel(torch.nn.Module):
    def __init__(self, _: object) -> None:
      super().__init__()
      self.value = torch.nn.Parameter(torch.zeros(1))

  monkeypatch.setattr(model_module, "StateLatentTransformer", TinyModel)
  monkeypatch.setattr(
    checkpoint_module,
    "load_checkpoint",
    lambda *args, **kwargs: SimpleNamespace(
      selection_split=None, best_metric_split="test"
    ),
  )
  code = diffusion.main(["evaluate-offline", "--run", str(run), "--device", "cpu"])
  assert code == 1
  error = capsys.readouterr().err
  assert "best_metric_split" in error
  assert "checkpoint-last.pt" in error


def test_evaluate_offline_allows_legacy_last_but_flags_test_provenance(
  tmp_path: Path, monkeypatch, capsys
) -> None:
  """`last` was never selected on anything, so it stays evaluable -- but flagged."""
  from mjlab.tasks.tracking.diffusion import checkpoint as checkpoint_module
  from mjlab.tasks.tracking.diffusion import evaluation as evaluation_module
  from mjlab.tasks.tracking.diffusion import model as model_module
  from mjlab.tasks.tracking.diffusion import window_dataset as window_module

  contract = DiffusionContract.from_yaml(
    Path("docs/plans/beyondmimic_diffusion_d0_contract.yaml")
  )
  schedule = DiffusionSchedule.from_contract(contract)
  dataset_dir = tmp_path / "dataset"
  dataset_dir.mkdir()
  source_hash = "d" * 64
  projection_hashes = {
    "matrix": "m" * 64,
    "pseudoinverse": "p" * 64,
    "statistics": "s" * 64,
  }
  loaded = SimpleNamespace(
    projection=SimpleNamespace(
      contract=contract,
      matrix_sha256=projection_hashes["matrix"],
      pseudoinverse_sha256=projection_hashes["pseudoinverse"],
      statistics_sha256=projection_hashes["statistics"],
    ),
    assignments=SimpleNamespace(sha256=lambda: "a" * 64),
    index=SimpleNamespace(coverage=lambda: {"train": 0, "validation": 0, "test": 0}),
  )

  class TinyModel(torch.nn.Module):
    def __init__(self, _: object) -> None:
      super().__init__()
      self.value = torch.nn.Parameter(torch.zeros(1))

  class FakeSource:
    directory = dataset_dir

    def load(self) -> object:
      return loaded

  state = SimpleNamespace(
    selection_split=None,
    best_metric_split="test",
    contract_identity=contract.identity_hash(),
    schedule_identity=schedule.identity_hash(),
    dataset_identity={
      "directory": str(dataset_dir),
      "source_dataset_hash": source_hash,
      "assignments_hash": "a" * 64,
      "split_coverage": {"train": 0, "validation": 0, "test": 0},
      "contract_hash": contract.identity_hash(),
      "projection_hashes": projection_hashes,
    },
    config=TrainingConfig().as_dict(),
    ema_state=None,
  )
  monkeypatch.setattr(model_module, "StateLatentTransformer", TinyModel)
  monkeypatch.setattr(checkpoint_module, "load_checkpoint", lambda *a, **k: state)
  monkeypatch.setattr(
    window_module.DatasetSource, "resolve", staticmethod(lambda *a, **k: FakeSource())
  )
  monkeypatch.setattr(window_module, "_source_dataset_hash", lambda *a: source_hash)
  monkeypatch.setattr(window_module, "load_window_records", lambda *a, **k: [])
  monkeypatch.setattr(
    evaluation_module,
    "evaluate_generation",
    lambda *a, **k: SimpleNamespace(
      windows=0,
      as_dict=lambda: {"split": k["split"], "windows": 0},
    ),
  )

  run = tmp_path / "legacy-last"
  run.mkdir()
  (run / "checkpoint-last.pt").write_bytes(b"placeholder")
  code = diffusion.main(
    [
      "evaluate-offline",
      "--run",
      str(run),
      "--checkpoint",
      str(run / "checkpoint-last.pt"),
      "--device",
      "cpu",
      "--split",
      "validation",
      "--max-windows",
      "0",
    ]
  )
  out = capsys.readouterr().out
  assert code == 0
  assert '"selection_split": null' in out
  assert '"test_selection_provenance": true' in out
