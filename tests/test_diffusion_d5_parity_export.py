"""Focused checks for the D5 denoiser export wrapper.

The D5 gate is a script, so these tests load it by path and exercise the two
behaviours that decide whether an exported graph is the checkpoint it claims to
be: the final-normalisation ablation must be copied into the wrapper, and a
recorded tensor the wrapper would drop must fail loudly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from mjlab.tasks.tracking.diffusion.model import (
  DenoiserSettings,
  StateLatentTransformer,
)
from mjlab.tasks.tracking.diffusion.model_final_norm import (
  StateLatentTransformerFinalNorm,
)

_ROOT = Path(__file__).resolve().parents[1]
_VALIDATE = _ROOT / "scripts" / "experiments" / "diffusion_d5_parity" / "validate.py"


@pytest.fixture(scope="module")
def gate():
  spec = importlib.util.spec_from_file_location("d5_parity_gate", _VALIDATE)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def _settings() -> DenoiserSettings:
  return DenoiserSettings(
    token_dimension=16,
    sequence_length=4,
    embedding_count=11,
    width=8,
    layers=1,
    attention_heads=2,
    ffn_width=16,
  )


def _probe() -> tuple[torch.Tensor, torch.Tensor]:
  torch.manual_seed(0)
  tokens = torch.randn(2, 4, 16)
  step_ids = torch.tensor([[[10, 10], [0, 0], [10, 5], [10, 10]]] * 2, dtype=torch.long)
  return tokens, step_ids


def test_denoiser_export_matches_the_final_norm_model(gate) -> None:
  tokens, step_ids = _probe()
  base = StateLatentTransformer(_settings()).eval()
  variant = StateLatentTransformerFinalNorm(_settings()).eval()
  assert gate.DenoiserExport(base).final_norm is None
  assert gate.DenoiserExport(variant).final_norm is not None
  with torch.no_grad():
    for model in (base, variant):
      expected = model(tokens, step_ids)
      exported = gate.DenoiserExport(model)(tokens, step_ids)
      assert torch.allclose(exported, expected, atol=1e-6, rtol=0.0)


def test_export_refuses_a_recorded_tensor_the_wrapper_would_drop(gate) -> None:
  base = StateLatentTransformer(_settings()).eval()
  recorded: dict[str, Any] = dict(base.named_parameters())
  recorded["final_norm.weight"] = torch.ones(8)
  recorded["final_norm.bias"] = torch.zeros(8)
  with pytest.raises(gate.ParityError, match="final_norm"):
    gate._require_export_covers_recorded_state(
      gate.DenoiserExport(base), recorded, checkpoint=Path("checkpoint.pt")
    )
  variant = StateLatentTransformerFinalNorm(_settings()).eval()
  gate._require_export_covers_recorded_state(
    gate.DenoiserExport(variant),
    dict(variant.named_parameters()),
    checkpoint=Path("checkpoint.pt"),
  )
