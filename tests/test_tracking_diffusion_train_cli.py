"""Tests for the bounded D2 training/evaluation CLI verbs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from mjlab.scripts import diffusion
from mjlab.tasks.tracking.diffusion.training_config import TrainingConfig

CONFIG = Path("configs/diffusion/x2_50hz_train.yaml")
PILOT = Path("logs/diffusion/d1-pilot-20260930b")


def test_evaluate_offline_missing_best_names_last_checkpoint(
  tmp_path: Path, capsys
) -> None:
  code = diffusion.main(
    ["evaluate-offline", "--run", str(tmp_path / "run"), "--device", "cpu"]
  )
  assert code == 1
  error = capsys.readouterr().err
  assert "--checkpoint" in error
  assert "checkpoint-last.pt" in error


def test_evaluate_offline_refuses_test_selected_checkpoint(
  tmp_path: Path, monkeypatch, capsys
) -> None:
  from mjlab.tasks.tracking.diffusion import checkpoint as checkpoint_module
  from mjlab.tasks.tracking.diffusion import model as model_module

  run = tmp_path / "run"
  run.mkdir()
  checkpoint = run / "checkpoint-best.pt"
  checkpoint.write_bytes(b"placeholder")

  class TinyModel(torch.nn.Module):
    def __init__(self, _: object) -> None:
      super().__init__()
      self.value = torch.nn.Parameter(torch.zeros(1))

  monkeypatch.setattr(model_module, "StateLatentTransformer", TinyModel)
  monkeypatch.setattr(
    checkpoint_module,
    "load_checkpoint",
    lambda *args, **kwargs: SimpleNamespace(selection_split="test"),
  )
  code = diffusion.main(["evaluate-offline", "--run", str(run), "--device", "cpu"])
  assert code == 1
  error = capsys.readouterr().err
  assert "selection_split" in error
  assert "checkpoint-last.pt" in error

  def constructor(_: object) -> torch.nn.Module:
    return torch.nn.Linear(3, 2)

  first = diffusion._build_seeded_model(object(), 17, constructor)
  second = diffusion._build_seeded_model(object(), 17, constructor)
  assert all(
    torch.equal(left, right)
    for left, right in zip(first.parameters(), second.parameters(), strict=True)
  )


def test_evaluate_offline_empty_split_requires_nonzero_unless_zero_bound(
  tmp_path: Path, monkeypatch, capsys
) -> None:
  from mjlab.tasks.tracking.diffusion import checkpoint as checkpoint_module
  from mjlab.tasks.tracking.diffusion import evaluation as evaluation_module
  from mjlab.tasks.tracking.diffusion import model as model_module
  from mjlab.tasks.tracking.diffusion import window_dataset as window_module
  from mjlab.tasks.tracking.diffusion.contract import DiffusionContract
  from mjlab.tasks.tracking.diffusion.schedule import DiffusionSchedule

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

  empty_run = tmp_path / "empty"
  empty_run.mkdir()
  (empty_run / "checkpoint-best.pt").write_bytes(b"placeholder")
  code = diffusion.main(
    [
      "evaluate-offline",
      "--run",
      str(empty_run),
      "--device",
      "cpu",
      "--split",
      "validation",
    ]
  )
  assert code == 1
  error = capsys.readouterr().err
  assert "validation" in error
  assert "no windows were selected" in error

  zero_run = tmp_path / "zero"
  zero_run.mkdir()
  (zero_run / "checkpoint-best.pt").write_bytes(b"placeholder")
  code = diffusion.main(
    [
      "evaluate-offline",
      "--run",
      str(zero_run),
      "--device",
      "cpu",
      "--split",
      "validation",
      "--max-windows",
      "0",
    ]
  )
  assert code == 0

  base = {
    "directory": "/dataset",
    "source_dataset_hash": "a" * 64,
    "contract_hash": "b" * 64,
    "projection_hashes": {"matrix": "c" * 64},
  }
  dataset = diffusion._attach_dataset_identity(
    torch.utils.data.TensorDataset(torch.zeros((1, 41, 231))),
    base_identity=base,
    split="train",
    motion_ids=("tennis_001",),
  )
  assert dataset.dataset_identity["source_dataset_hash"] == "a" * 64
  assert dataset.dataset_identity["motion_ids"] == ["tennis_001"]


def test_checkpoint_motion_filter_is_normalized() -> None:
  assert diffusion._checkpoint_motion_ids({"motion_ids": ["b", "a", "a"]}) == (
    "a",
    "b",
  )


def test_train_parser_requires_the_owned_arguments() -> None:
  args = diffusion.build_parser().parse_args(
    [
      "train",
      "--dataset-dir",
      str(PILOT),
      "--config",
      str(CONFIG),
      "--out",
      "run",
      "--max-updates",
      "5",
      "--no-cache-tokens",
    ]
  )
  assert args.command == "train"
  assert args.max_updates == 5
  assert args.cache_tokens is False


def test_train_refuses_the_non_authorizing_default_budget(
  tmp_path: Path, capsys
) -> None:
  code = diffusion.main(
    [
      "train",
      "--dataset-dir",
      str(PILOT),
      "--config",
      str(CONFIG),
      "--out",
      str(tmp_path / "run"),
      "--device",
      "cpu",
    ]
  )
  assert code == 1
  assert "explicit" in capsys.readouterr().err


def test_training_yaml_matches_the_frozen_config() -> None:
  config = TrainingConfig.from_yaml(CONFIG)
  assert config.epochs == 1000
  assert config.effective_batch_size == 512
  assert config.microbatch_size == 128
  assert config.gradient_accumulation_steps == 4
  assert config.clean_weight == 1.0
  assert config.perturbed_weight == 1.0
  assert config.ema_power == 0.75
  assert config.ema_max_decay == 0.9999
  assert config.eval_split == "validation"
  assert (
    config.sha256()
    == TrainingConfig.from_mapping(
      {
        **config.as_dict(),
        "ema_power": 0.75,
        "ema_max_decay": 0.9999,
      }
    ).sha256()
  )


def test_audit_parser_is_offline_only() -> None:
  args = diffusion.build_parser().parse_args(["audit", "--dataset-dir", str(PILOT)])
  assert args.command == "audit"
  assert args.dataset_dir == PILOT
