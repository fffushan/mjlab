"""CPU-only gate: a capture must replay exactly, and the hooks must report.

The analyzer is the only consumer of a capture, so this test drives the whole
path: train two updates with the recorder on, force a trigger, then replay the
capture through that checkpoint and require the same per-window loss.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from mjlab.scripts.analyze_flow_capture import main as analyze_main
from mjlab.tasks.tracking.diffusion.contract import DiffusionContract
from mjlab.tasks.tracking.diffusion.model import (
  DenoiserSettings,
  StateLatentTransformer,
)
from mjlab.tasks.tracking.diffusion.training_config import TrainingConfig
from mjlab.tasks.tracking.flow_matching import FlowTrainer, FlowTrainingConfig
from mjlab.tasks.tracking.flow_matching.checkpoint import (
  FLOW_CHECKPOINT_FORMAT,
  model_identity,
)
from mjlab.tasks.tracking.flow_matching.diagnostics import (
  ENV_DIRECTORY,
  ENV_TRIGGER_FACTOR,
)

COUNT = 6


class SyntheticWindows(torch.utils.data.Dataset[tuple[Tensor, Tensor, int]]):
  def __init__(self) -> None:
    generator = torch.Generator().manual_seed(23)
    self.tokens = torch.randn((COUNT, 41, 167), generator=generator)
    self.dataset_identity = {
      "directory": "synthetic-capture-v1",
      "assignments_hash": "a" * 64,
      "split_coverage": {"train": COUNT, "validation": 0, "test": 0},
      "projection_hashes": {"matrix": "b" * 64, "statistics": "c" * 64},
    }

  def __len__(self) -> int:
    return COUNT

  def __getitem__(self, index: int) -> tuple[Tensor, Tensor, int]:
    return self.tokens[index], torch.tensor(1.0), int(index)


def _settings() -> DenoiserSettings:
  # A small stand-in keeps the frozen default-shape count guard out of the way.
  return DenoiserSettings(width=32, layers=2, attention_heads=2, ffn_width=64)


def _train(tmp_path: Path) -> Path:
  config = FlowTrainingConfig(
    training=TrainingConfig(
      epochs=1,
      max_updates=2,
      effective_batch_size=2,
      microbatch_size=2,
      gradient_accumulation_steps=1,
      mixed_precision="fp32",
      warmup_updates=1,
      eval_split="validation",
    )
  )
  torch.manual_seed(config.training.seed)
  output = tmp_path / "run"
  trainer = FlowTrainer(
    config=config,
    contract=DiffusionContract(),
    model=StateLatentTransformer(_settings()),
    train_dataset=SyntheticWindows(),
    eval_datasets={"validation": SyntheticWindows()},
    output_dir=output,
    device="cpu",
  )
  trainer.diagnostics.baseline_window = 1  # noqa: SLF001
  trainer.train()
  return output


def test_a_capture_replays_to_the_recorded_loss(tmp_path: Path, monkeypatch) -> None:
  monkeypatch.setenv(ENV_DIRECTORY, str(tmp_path / "diag"))
  # A 1e-9 factor triggers on every update after the first baseline sample.
  monkeypatch.setenv(ENV_TRIGGER_FACTOR, "1e-9")
  output = _train(tmp_path)

  captures = sorted((tmp_path / "diag" / "captures").glob("capture-*.npz"))
  assert captures, "the recorder wrote no capture"
  capture = captures[-1]

  # The analyzer reads clean tokens from a plain memmap, so write the same
  # fixture window stack in that shape.
  cache = tmp_path / "tokens.npy"
  np.save(cache, SyntheticWindows().tokens.numpy())

  checkpoints = sorted(output.glob("checkpoint-trigger-*.pt"))
  assert checkpoints, "a trigger must also save a checkpoint"
  report_path = tmp_path / "analysis.json"

  code = analyze_main(
    [
      "--capture",
      str(capture),
      "--token-cache",
      str(cache),
      "--config",
      str(output / "resolved-config.json"),
      "--checkpoint",
      f"trigger={checkpoints[0]}",
      "--checkpoint",
      f"last={output / 'checkpoint-last.pt'}",
      "--attention-windows",
      "2",
      "--out",
      str(report_path),
    ]
  )
  assert code == 0
  report = json.loads(report_path.read_text())
  assert len(report["checkpoints"]) == 2
  first = report["checkpoints"][0]
  # The replay must reproduce the loss the trainer recorded for those windows,
  # otherwise the noise draw was rebuilt from the wrong generator state.
  assert first["replay_matched_rows"] >= 1
  assert first["replay_abs_error"] < 1e-4
  assert first["windows"] > 0
  assert "layer0_attention_entropy_mean" in first
  assert first["layer0_input_var"] > 0.0
  assert first["layer0_ffn_input_var"] > 0.0
  assert first["layer0_activation_norm"] > 0.0
  assert any(key.endswith("_adamw_first_norm") for key in first)
  assert "last_vs_trigger" in report["comparison"]
  stats = report["comparison"]["last_vs_trigger"]
  assert 0.0 <= stats["windows_worse"] <= 1.0
  assert (report_path.with_suffix(".md")).exists()


def test_the_loader_builds_the_recorded_model_class(tmp_path: Path) -> None:
  """A final-norm checkpoint must load, because the artifact names its class.

  The closed-loop loader and this analyzer share one registry.  Before that
  registry existed, both assumed the frozen base class and rejected every
  final-norm checkpoint, which is how two captured arms became unanalyzable.
  """
  from mjlab.scripts.analyze_flow_capture import _load_checkpoint
  from mjlab.tasks.tracking.diffusion.model_final_norm import (
    StateLatentTransformerFinalNorm,
  )

  settings = _settings()
  model = StateLatentTransformerFinalNorm(settings)
  payload = {
    "format": FLOW_CHECKPOINT_FORMAT,
    "model": model.state_dict(),
    "model_identity": {
      "type": StateLatentTransformerFinalNorm.__qualname__,
      "settings": model_identity(StateLatentTransformerFinalNorm(settings))["settings"],
    },
    "process": {"max_level": 100},
    "global_step": 7,
  }
  path = tmp_path / "checkpoint-final-norm.pt"
  torch.save(payload, path)
  loaded, process, restored = _load_checkpoint(path, torch.device("cpu"))
  assert isinstance(loaded, StateLatentTransformerFinalNorm)
  assert restored["global_step"] == 7
  assert process.max_level == 100

  payload["model_identity"]["type"] = "NotARecordedDenoiser"
  torch.save(payload, path)
  try:
    _load_checkpoint(path, torch.device("cpu"))
  except SystemExit as error:
    assert "unsupported model type" in str(error)
  else:
    raise AssertionError("an unknown recorded type must fail closed")
