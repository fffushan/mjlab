"""Tests for the frozen bidirectional Transformer denoiser."""

from __future__ import annotations

import pytest
import torch

from mjlab.tasks.tracking.diffusion.contract import DiffusionContract
from mjlab.tasks.tracking.diffusion.model import (
  DenoiserSettings,
  StateLatentTransformer,
)


def test_frozen_model_shape_parameter_count_and_finite_output() -> None:
  model = StateLatentTransformer(DenoiserSettings())
  assert model.parameter_count() == 20_197_607
  model.train()
  tokens = torch.randn((1, 41, 231), dtype=torch.float32)
  ids = torch.zeros((1, 41, 2), dtype=torch.long)
  first = model(tokens, ids)
  second = model(tokens, ids)
  assert first.shape == tokens.shape
  assert torch.isfinite(first).all()
  assert torch.equal(first, second)


def test_settings_from_contract_uses_frozen_layout() -> None:
  settings = DenoiserSettings.from_contract(DiffusionContract())
  assert settings == DenoiserSettings()


def test_invalid_step_ids_are_rejected() -> None:
  model = StateLatentTransformer(
    DenoiserSettings(width=32, layers=1, attention_heads=4, ffn_width=64)
  )
  tokens = torch.randn((1, 41, 231))
  valid = torch.zeros((1, 41, 2), dtype=torch.long)
  invalid_low = valid.clone()
  invalid_low[0, 0, 0] = -1
  invalid_high = valid.clone()
  invalid_high[0, 0, 1] = 1001
  with pytest.raises(ValueError, match="step_ids"):
    model(tokens, invalid_low)
  with pytest.raises(ValueError, match="step_ids"):
    model(tokens, invalid_high)


def test_token_40_influences_token_0_without_causal_mask() -> None:
  torch.manual_seed(19)
  model = StateLatentTransformer(
    DenoiserSettings(width=32, layers=1, attention_heads=4, ffn_width=64)
  )
  model.eval()
  tokens = torch.randn((1, 41, 231))
  ids = torch.zeros((1, 41, 2), dtype=torch.long)
  with torch.no_grad():
    baseline = model(tokens, ids)
    perturbed_tokens = tokens.clone()
    perturbed_tokens[:, 40, 0] += 1.0
    perturbed = model(perturbed_tokens, ids)
  assert not torch.equal(baseline[:, 0], perturbed[:, 0])
  assert float((baseline[:, 0] - perturbed[:, 0]).abs().max()) > 1.0e-7


def test_gradients_reach_every_parameter() -> None:
  model = StateLatentTransformer(
    DenoiserSettings(width=32, layers=1, attention_heads=4, ffn_width=64)
  )
  tokens = torch.randn((2, 41, 231), requires_grad=False)
  ids = torch.randint(0, 1001, (2, 41, 2), dtype=torch.long)
  model(tokens, ids).square().mean().backward()
  assert all(parameter.grad is not None for parameter in model.parameters())
