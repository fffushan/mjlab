"""Focused tests for the diffusion level-sampling and loss-weighting ablations.

Every added surface must default to the frozen D2 behaviour, so the first thing
these tests pin is that an unmodified config reproduces the uniform sampler's
exact draw sequence and the plain ``x0`` mean loss.
"""

from __future__ import annotations

import math

import pytest
import torch

from mjlab.tasks.tracking.diffusion.noising import sample_levels
from mjlab.tasks.tracking.diffusion.training_config import (
  TrainingConfig,
  TrainingConfigError,
)


def _config(**overrides: object) -> TrainingConfig:
  values: dict[str, object] = {
    "epochs": 10,
    "max_updates": 100,
    "effective_batch_size": 8,
    "microbatch_size": 4,
    "gradient_accumulation_steps": 2,
  }
  values.update(overrides)
  return TrainingConfig(**values)  # type: ignore[arg-type]


def test_default_config_is_uniform_with_plain_loss() -> None:
  config = _config()
  assert config.level_sampling == "uniform"
  assert config.state_loss_weight == 1.0
  assert config.latent_loss_weight == 1.0
  assert config.checkpoint_every is None
  config.validate()


def test_uniform_mode_matches_randint_draw_for_draw() -> None:
  """The default path must not consume extra or different randomness."""
  shape = (4, 41)
  reference_generator = torch.Generator().manual_seed(7)
  reference = sample_levels(shape, training_k=1000, generator=reference_generator)
  explicit_generator = torch.Generator().manual_seed(7)
  explicit = sample_levels(
    shape, training_k=1000, generator=explicit_generator, mode="uniform"
  )
  assert torch.equal(reference, explicit)
  # And the generator is left in the same place for the next draw.
  assert torch.equal(
    sample_levels(shape, training_k=1000, generator=reference_generator),
    sample_levels(shape, training_k=1000, generator=explicit_generator),
  )


@pytest.mark.parametrize("mode", ["logit_normal", "low_noise", "high_noise"])
def test_modes_stay_in_range_and_use_the_generator(mode: str) -> None:
  levels = sample_levels(
    (32, 41), training_k=1000, generator=torch.Generator().manual_seed(3), mode=mode
  )
  assert levels.dtype == torch.long
  assert int(levels.min()) >= 0
  assert int(levels.max()) <= 1000
  repeat = sample_levels(
    (32, 41), training_k=1000, generator=torch.Generator().manual_seed(3), mode=mode
  )
  assert torch.equal(levels, repeat)


def test_low_noise_shifts_mass_below_high_noise() -> None:
  shape = (64, 41)
  low = sample_levels(
    shape,
    training_k=1000,
    generator=torch.Generator().manual_seed(11),
    mode="low_noise",
    shift_gamma=3.0,
  )
  high = sample_levels(
    shape,
    training_k=1000,
    generator=torch.Generator().manual_seed(11),
    mode="high_noise",
    shift_gamma=3.0,
  )
  assert float(low.float().mean()) < 250.0
  assert float(high.float().mean()) > 750.0
  # u**3 gives P(t < K/2) = 2**-1/3 ~= 0.79.
  assert abs(float((low < 500).float().mean()) - 0.794) < 0.05
  assert abs(float((high > 500).float().mean()) - 0.794) < 0.05


def test_logit_normal_concentrates_on_mid_noise() -> None:
  levels = sample_levels(
    (256, 41),
    training_k=1000,
    generator=torch.Generator().manual_seed(5),
    mode="logit_normal",
  )
  middle = float(((levels >= 250) & (levels <= 750)).float().mean())
  assert middle > 0.5
  assert float(levels.float().mean()) == pytest.approx(500.0, abs=40.0)


def test_unknown_mode_is_rejected() -> None:
  with pytest.raises(ValueError, match="unsupported level sampling mode"):
    sample_levels((2, 3), training_k=10, generator=torch.Generator(), mode="quadratic")


@pytest.mark.parametrize(
  ("field", "value"),
  [
    ("level_sampling", "cosine"),
    ("level_shift_gamma", 0.0),
    ("level_logit_std", -1.0),
    ("state_loss_weight", -0.5),
    ("latent_loss_weight", float("nan")),
    ("checkpoint_every", 0),
  ],
)
def test_invalid_ablation_values_are_rejected(field: str, value: object) -> None:
  with pytest.raises(TrainingConfigError):
    _config(**{field: value}).validate()


def test_both_loss_weights_zero_is_rejected() -> None:
  with pytest.raises(TrainingConfigError, match="at least one loss weight"):
    _config(state_loss_weight=0.0, latent_loss_weight=0.0).validate()


def test_ablation_block_is_read_and_hashed(tmp_path) -> None:
  """An ablation block must change the config hash, defaults must not."""
  base = """schema_version: mjlab-x2-state-latent-train-v1
contract: docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml
model: {layers: 6, width: 512, attention_heads: 8, ffn_width: 2048, dropout: 0.0}
noise: {training_k: 1000, cosine_offset: 0.008, beta_min: 1.0e-5, beta_max: 0.999, updates: 20}
training:
  epochs: 10
  effective_batch_size: 512
  microbatch_size: 128
  gradient_accumulation_steps: 4
  learning_rate: 1.0e-4
  weight_decay: 0.001
  scheduler: cosine
  warmup_updates: 10000
  max_grad_norm: 1.0
  mixed_precision: bf16
  loss_reduction: token_mean
  seed: 0
ema: {power: 0.75, max_decay: 0.9999}
sampling: {clean_weight: 1.0, perturbed_weight: 1.0}
eval: {split: validation, seed: 0}
"""
  plain = tmp_path / "plain.yaml"
  plain.write_text(base, encoding="utf-8")
  plain_config = TrainingConfig.from_yaml(plain)
  assert plain_config.level_sampling == "uniform"
  assert plain_config.checkpoint_every is None

  ablated = tmp_path / "ablated.yaml"
  ablated.write_text(
    base + "ablation: {level_sampling: low_noise, latent_loss_weight: 0.5, "
    "checkpoint_every: 4000}\n",
    encoding="utf-8",
  )
  ablated_config = TrainingConfig.from_yaml(ablated)
  assert ablated_config.level_sampling == "low_noise"
  assert ablated_config.latent_loss_weight == 0.5
  assert ablated_config.checkpoint_every == 4000
  assert ablated_config.sha256() != plain_config.sha256()


def test_weighted_loss_reduces_to_the_plain_mean() -> None:
  """Recomputing the weighted form at unit weights must match F.mse_loss."""
  torch.manual_seed(0)
  prediction = torch.randn(3, 41, 167)
  target = torch.randn(3, 41, 167)
  difference = (prediction - target).square()
  split = 135
  state = difference[..., :split]
  latent = difference[..., split:]
  weighted = (state.sum() + latent.sum()) / (state.numel() + latent.numel())
  plain = torch.nn.functional.mse_loss(prediction, target, reduction="mean")
  assert math.isclose(float(weighted), float(plain), rel_tol=1e-6, abs_tol=1e-9)
