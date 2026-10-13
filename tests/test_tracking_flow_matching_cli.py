"""CPU-only CLI gates and a synthetic train-save-load-generate smoke test.

The smoke test substitutes a D1-shaped synthetic source, read-only cache and
model, so it needs no collected artifact, simulator or controller.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from mjlab.scripts import flow_matching
from mjlab.tasks.tracking.diffusion import model as model_module
from mjlab.tasks.tracking.diffusion import token_cache_lifecycle as lifecycle_module
from mjlab.tasks.tracking.diffusion import window_dataset as window_module
from mjlab.tasks.tracking.diffusion.contract import DiffusionContract
from mjlab.tasks.tracking.diffusion.window_dataset import WindowRecord
from mjlab.tasks.tracking.flow_matching import (
  FLOW_CHECKPOINT_FORMAT,
  FlowTrainingConfig,
)

CONTRACT_PATH = Path("docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml")
TRAIN_CONFIG = Path("configs/flow_matching/x2_50hz_train.yaml")
CONTRACT = DiffusionContract()
STEPS = CONTRACT.window_steps
TOKENS = CONTRACT.token_dimension


class _TinyDenoiser(nn.Module):
  """Pointwise model that accepts the frozen denoiser calling convention."""

  def __init__(self, settings: Any) -> None:
    super().__init__()
    self.settings = settings
    self.linear = nn.Linear(settings.token_dimension, settings.token_dimension)

  def forward(self, tokens: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
    del step_ids
    return self.linear(tokens)


class _SyntheticData:
  """Synthetic D1 source, read-only caches, records and token windows."""

  def __init__(self, root: Path, counts: dict[str, int]) -> None:
    self.directory = root
    self.coverage = {
      name: int(counts.get(name, 0)) for name in ("train", "validation", "test")
    }
    self.source_hash = "e" * 64
    self.assignments_hash = "a" * 64
    self.projection_hashes = {
      "matrix": "b" * 64,
      "pseudoinverse": "c" * 64,
      "statistics": "d" * 64,
    }
    self.read_only_calls: list[str] = []
    self.tokens = {
      split: torch.randn(
        (count, STEPS, TOKENS), generator=torch.Generator().manual_seed(index)
      )
      for index, (split, count) in enumerate(self.coverage.items())
    }
    self.identity = {
      split: {
        "directory": str(root),
        "assignments_hash": self.assignments_hash,
        "split_coverage": dict(self.coverage),
        "split": split,
        "motion_ids": None,
        "limit": None,
        "source_dataset_hash": self.source_hash,
        "contract_hash": CONTRACT.identity_hash(),
        "projection_hash": "f" * 64,
        "projection_hashes": dict(self.projection_hashes),
      }
      for split in self.coverage
    }
    self.loaded = SimpleNamespace(
      contract=CONTRACT,
      projection=SimpleNamespace(
        matrix_sha256=self.projection_hashes["matrix"],
        pseudoinverse_sha256=self.projection_hashes["pseudoinverse"],
        statistics_sha256=self.projection_hashes["statistics"],
      ),
      assignments=SimpleNamespace(sha256=lambda: self.assignments_hash),
      index=SimpleNamespace(coverage=lambda: dict(self.coverage)),
    )

  def load(self) -> Any:
    return self.loaded

  def records(self, split: str) -> list[WindowRecord]:
    return [
      WindowRecord(
        tokens=row.numpy(),
        split=split,
        group_key="group",
        motion_id="tennis_000",
        phase="clean",
        ou_noise_norm=0.0,
        pair_id="pair",
        start_tick=index,
      )
      for index, row in enumerate(self.tokens[split])
    ]

  def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
    harness = self

    def validate_read_only(source: Any, loaded: Any, split: str, **kwargs: Any) -> Any:
      del source, loaded, kwargs
      harness.read_only_calls.append(split)
      return SimpleNamespace(
        path=harness.directory / "offline" / f"tokens-{split}.npy",
        provenance=(),
        cache=SimpleNamespace(),
      )

    def load_records(
      source: Any, split: str, *, limit: int | None = None, **kwargs: Any
    ) -> list[WindowRecord]:
      del source, kwargs
      records = harness.records(split)
      return records[:limit] if limit is not None else records

    class SyntheticTokenDataset(Dataset[torch.Tensor]):
      def __init__(self, path: Any, records: Any = None, **kwargs: Any) -> None:
        del path, records
        split = str(kwargs["split"])
        self.split = split
        self.tokens = harness.tokens[split]
        self.dataset_identity = dict(harness.identity[split])
        self.records = None
        self.weights = np.ones(len(self.tokens))

      def __len__(self) -> int:
        return len(self.tokens)

      def __getitem__(self, index: int) -> torch.Tensor:
        return self.tokens[index]

    monkeypatch.setattr(
      window_module.DatasetSource,
      "resolve",
      staticmethod(lambda directory, *, contract_path: harness),
    )
    monkeypatch.setattr(
      window_module, "_source_dataset_hash", lambda *args, **kwargs: harness.source_hash
    )
    monkeypatch.setattr(window_module, "TokenWindowDataset", SyntheticTokenDataset)
    monkeypatch.setattr(window_module, "load_window_records", load_records)
    monkeypatch.setattr(
      lifecycle_module, "validate_token_cache_read_only", validate_read_only
    )
    monkeypatch.setattr(model_module, "StateLatentTransformer", _TinyDenoiser)


def _write_config(tmp_path: Path, *, max_updates: int = 2) -> Path:
  path = tmp_path / "smoke-train.yaml"
  path.write_text(
    "\n".join(
      (
        "schema_version: mjlab-x2-flow-matching-train-v1",
        f"contract: {CONTRACT_PATH.as_posix()}",
        "process: {max_level: 1000}",
        "training:",
        "  epochs: 1",
        f"  max_updates: {max_updates}",
        "  effective_batch_size: 2",
        "  microbatch_size: 2",
        "  gradient_accumulation_steps: 1",
        "  learning_rate: 0.01",
        "  warmup_updates: 0",
        "  mixed_precision: fp32",
        "  eval_split: validation",
        "  eval_seed: 7",
        "",
      )
    ),
    encoding="utf-8",
  )
  return path


def test_bare_cli_and_parser_defaults(capsys: pytest.CaptureFixture[str]) -> None:
  assert flow_matching.main([]) == 0
  output = capsys.readouterr().out
  assert "train" in output
  args = flow_matching.build_parser().parse_args(
    ["train", "--dataset-dir", "d", "--config", "c", "--out", "o"]
  )
  assert args.contract == CONTRACT_PATH
  assert args.device == "cpu"
  assert args.max_updates is None


def test_training_config_matches_the_frozen_flow_family() -> None:
  config = FlowTrainingConfig.from_yaml(TRAIN_CONFIG)
  config.validate()
  assert config.training.epochs == 1000
  assert config.training.effective_batch_size == 512
  assert config.training.microbatch_size == 128
  assert config.training.gradient_accumulation_steps == 4
  assert config.training.ema_power == 0.75
  assert config.training.history_state_dropout_probability == 0.0
  assert config.training.eval_split == "validation"
  assert config.max_level == 1000
  assert config.process.as_dict()["family"] == "x0_linear_flow"
  assert config.process.as_dict()["prediction_type"] == "x0"
  import yaml

  raw = yaml.safe_load(TRAIN_CONFIG.read_text(encoding="utf-8"))
  assert "noise" not in raw


def test_train_requires_a_budget_long_run_permission_and_matching_contract(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  base = [
    "train",
    "--dataset-dir",
    str(tmp_path / "missing-dataset"),
    "--config",
    str(TRAIN_CONFIG),
    "--out",
    str(tmp_path / "run"),
    "--device",
    "cpu",
  ]
  assert flow_matching.main(base) == 1
  assert "explicit" in capsys.readouterr().err
  assert flow_matching.main([*base, "--max-updates", "200001"]) == 1
  assert "allow-long-run" in capsys.readouterr().err
  assert not (tmp_path / "run").exists()

  config = _write_config(tmp_path)
  config.write_text(
    config.read_text().replace(
      CONTRACT_PATH.as_posix(), "docs/plans/beyondmimic_diffusion_d0_contract.yaml"
    ),
    encoding="utf-8",
  )
  code = flow_matching.main(
    [
      "train",
      "--dataset-dir",
      str(tmp_path / "missing-dataset"),
      "--config",
      str(config),
      "--out",
      str(tmp_path / "run"),
      "--max-updates",
      "1",
    ]
  )
  assert code == 1
  assert "disagrees with --contract" in capsys.readouterr().err


def test_evaluate_offline_refuses_missing_and_non_flow_checkpoints(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  run = tmp_path / "run"
  assert flow_matching.main(["evaluate-offline", "--run", str(run)]) == 1
  error = capsys.readouterr().err
  assert "--checkpoint" in error and "checkpoint-last.pt" in error

  run.mkdir()
  checkpoint = run / "checkpoint-last.pt"
  # The diffusion on-disk format must never load as a flow checkpoint.
  torch.save({"format": "mjlab-x2-diffusion-checkpoint-v1"}, checkpoint)
  assert (
    flow_matching.main(
      ["evaluate-offline", "--run", str(run), "--checkpoint", str(checkpoint)]
    )
    == 1
  )
  assert "flow checkpoint" in capsys.readouterr().err


def test_synthetic_cli_train_save_load_generate(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
) -> None:
  harness = _SyntheticData(tmp_path / "dataset", {"train": 3, "validation": 2})
  harness.install(monkeypatch)
  config = _write_config(tmp_path)
  out = tmp_path / "run"
  train_json = tmp_path / "train.json"
  code = flow_matching.main(
    [
      "train",
      "--dataset-dir",
      str(harness.directory),
      "--config",
      str(config),
      "--out",
      str(out),
      "--contract",
      str(CONTRACT_PATH),
      "--json",
      str(train_json),
    ]
  )
  assert code == 0, capsys.readouterr().err
  # Normal training only reads the published caches.
  assert harness.read_only_calls == ["train", "validation"]
  metadata = json.loads((out / "flow-run.json").read_text())
  assert metadata["schema"] == "mjlab-flow-matching-run-v1"
  assert metadata["family"] == "x0_linear_flow"
  assert metadata["process"]["prediction_type"] == "x0"
  assert metadata["process"]["loss"] == "x0_mse"
  assert metadata["process"]["max_level"] == 1000
  assert metadata["history_state_dropout_probability"] == 0.0
  assert metadata["seed"] == 0
  assert metadata["windows"] == {"train": 3, "validation": 2}
  assert sorted(metadata["cache_paths"]) == ["train", "validation"]
  assert (
    metadata["dataset_identity"]["projection_hashes"]["matrix"]
    == harness.projection_hashes["matrix"]
  )
  payload = json.loads(train_json.read_text())
  assert payload["family"] == "x0_linear_flow"
  assert Path(payload["checkpoint_paths"]["last"]).is_file()
  assert Path(payload["checkpoint_paths"]["best"]).is_file()
  checkpoint = torch.load(
    payload["checkpoint_paths"]["last"], map_location="cpu", weights_only=False
  )
  assert checkpoint["format"] == FLOW_CHECKPOINT_FORMAT
  assert checkpoint["process"]["time_encoding"] == "integer_noise_fraction"
  assert "schedule" not in checkpoint

  evaluate_json = tmp_path / "evaluate.json"
  code = flow_matching.main(
    [
      "evaluate-offline",
      "--run",
      str(out),
      "--split",
      "validation",
      "--contract",
      str(CONTRACT_PATH),
      "--json",
      str(evaluate_json),
    ]
  )
  assert code == 0, capsys.readouterr().err
  report = json.loads(evaluate_json.read_text())
  assert report["windows"] == 2
  assert report["split"] == "validation"
  # Without an explicit --seed the recorded evaluation seed is used.
  assert report["seed"] == 7
  assert report["sampler_calls"] == 20
  assert report["grid"] == {"levels": list(range(1000, -1, -50)), "calls": 20}
  assert report["process_identity"] == metadata["process_identity"]
  assert report["parameters"] == "ema"
  assert report["metrics"]["model_unknown_token_mse"] > 0.0
  assert sorted(report["baselines"]) == [
    "copy_current_latent",
    "copy_current_state",
    "prior_only_latent",
    "prior_only_state",
    "zero_latent",
  ]
  assert len(report["sample_sha256"]) == 64
  assert (out / "flow-generation-validation.json").is_file()

  # A changed dataset projection must fail closed instead of scoring silently.
  harness.loaded.projection = SimpleNamespace(
    matrix_sha256="0" * 64,
    pseudoinverse_sha256=harness.projection_hashes["pseudoinverse"],
    statistics_sha256=harness.projection_hashes["statistics"],
  )
  code = flow_matching.main(
    ["evaluate-offline", "--run", str(out), "--split", "validation"]
  )
  assert code == 1
  assert "projection identity mismatch" in capsys.readouterr().err
