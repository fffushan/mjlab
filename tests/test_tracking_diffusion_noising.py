"""Tests for independent state/latent forward noising and conditioning."""

from __future__ import annotations

import math

import torch

from mjlab.tasks.tracking.diffusion.noising import (
  add_independent_noise,
  apply_clean_mask,
  clean_mask,
  sample_levels,
)
from mjlab.tasks.tracking.diffusion.schedule import DiffusionSchedule


def test_levels_are_inclusive_and_nearly_uniform() -> None:
  generator = torch.Generator().manual_seed(123)
  levels = sample_levels((20_000, 41), training_k=1000, generator=generator)
  assert levels.dtype == torch.long
  assert int(levels.min()) >= 0
  assert int(levels.max()) <= 1000
  assert abs(float(levels.to(torch.float64).mean()) - 500.0) < 3.0


def test_clean_level_is_exact_and_noise_draws_do_not_alias() -> None:
  schedule = DiffusionSchedule.build()
  state = torch.randn((2, 41, 199), dtype=torch.float64)
  latent = torch.randn((2, 41, 32), dtype=torch.float64)
  levels = torch.zeros((2, 41), dtype=torch.long)
  noised = add_independent_noise(
    state,
    latent,
    levels,
    levels,
    schedule.alpha_bars,
    generator=torch.Generator().manual_seed(7),
  )
  assert torch.equal(noised.state, state)
  assert torch.equal(noised.latent, latent)
  assert noised.noise_state.data_ptr() != noised.noise_latent.data_ptr()
  assert noised.tokens.shape == (2, 41, 231)


def test_noising_marginal_matches_signal_and_noise_scale() -> None:
  schedule = DiffusionSchedule.build()
  count = 20_000
  clean_value = 1.25
  state = torch.full((count, 1, 1), clean_value, dtype=torch.float64)
  latent = torch.zeros((count, 1, 1), dtype=torch.float64)
  levels = torch.full((count, 1), 500, dtype=torch.long)
  noised = add_independent_noise(
    state,
    latent,
    levels,
    levels,
    schedule.alpha_bars,
    generator=torch.Generator().manual_seed(11),
  )
  alpha = schedule.alpha_bar_at(500)
  expected_mean = math.sqrt(alpha) * clean_value
  expected_std = math.sqrt(1.0 - alpha)
  observed = noised.state[:, 0, 0]
  assert abs(float(observed.mean()) - expected_mean) < 0.02
  assert abs(float(observed.std(unbiased=False)) - expected_std) < 0.02


def test_d0_clean_mask_and_reapplication() -> None:
  mask = clean_mask(
    current_index=8,
    state_dimension=199,
    token_dimension=231,
    device="cpu",
  )
  assert mask.shape == (41, 231)
  assert mask.dtype == torch.bool
  assert bool(mask[:8].all())
  assert bool(mask[8, :199].all())
  assert not bool(mask[8, 199:].any())
  assert not bool(mask[9:].any())

  clean = torch.arange(41 * 231, dtype=torch.float64).reshape(41, 231)
  sample = torch.full_like(clean, -1.0)
  reapplied = apply_clean_mask(sample, clean, mask)
  assert torch.equal(reapplied[mask], clean[mask])
  assert bool((reapplied[~mask] == -1.0).all())
