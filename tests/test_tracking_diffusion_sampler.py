"""Tests for the frozen 20-call unguided DDIM sampler."""

from __future__ import annotations

import pytest
import torch

from mjlab.tasks.tracking.diffusion.noising import apply_clean_mask, clean_mask
from mjlab.tasks.tracking.diffusion.sampler import (
  SamplerConditions,
  SamplerConfig,
  sample_trajectory,
)
from mjlab.tasks.tracking.diffusion.schedule import DiffusionSchedule, InferenceGrid


class _OracleDenoiser:
  def __init__(self, x0: torch.Tensor) -> None:
    self.x0 = x0
    self.inputs: list[torch.Tensor] = []
    self.step_ids: list[torch.Tensor] = []

  def __call__(self, sample: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
    self.inputs.append(sample.detach().clone())
    self.step_ids.append(step_ids.detach().clone())
    return self.x0.expand_as(sample)


class _InputDependentDenoiser:
  def __call__(self, sample: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
    del step_ids
    return 0.25 * sample


def _conditions() -> SamplerConditions:
  clean = torch.randn((1, 41, 231), dtype=torch.float64)
  mask = clean_mask(
    current_index=8,
    state_dimension=199,
    token_dimension=231,
    device=clean.device,
  )
  return SamplerConditions(clean, mask)


def test_ddim_oracle_has_exact_frozen_calls_ids_and_jumps() -> None:
  schedule = DiffusionSchedule.build()
  grid = InferenceGrid.uniform(schedule)
  conditions = _conditions()
  denoiser = _OracleDenoiser(conditions.clean_tokens)
  result, diagnostics = sample_trajectory(
    denoiser,
    schedule=schedule,
    conditions=conditions,
    config=SamplerConfig(grid),
    generator=torch.Generator().manual_seed(5),
  )
  assert result.shape == conditions.clean_tokens.shape
  assert diagnostics.denoiser_calls == 20
  assert diagnostics.jumps == 20
  assert diagnostics.source_ids_returned == tuple(range(1000, 0, -50))
  assert diagnostics.step_ids_seen == tuple(range(1000, 0, -50))
  assert diagnostics.final_index == 0

  alpha = schedule.alpha_bars
  for index, (source, destination) in enumerate(
    zip(grid.source_ids, grid.destination_ids, strict=True)
  ):
    current = denoiser.inputs[index]
    if destination == 0:
      expected = conditions.clean_tokens
    else:
      eps_hat = (
        current - torch.sqrt(torch.tensor(alpha[source])) * conditions.clean_tokens
      ) / torch.sqrt(torch.tensor(1.0 - alpha[source]))
      expected = torch.sqrt(torch.tensor(alpha[destination])) * conditions.clean_tokens
      expected = expected + torch.sqrt(torch.tensor(1.0 - alpha[destination])) * eps_hat
    expected = apply_clean_mask(
      expected, conditions.clean_tokens, conditions.clean_mask
    )
    if index + 1 < len(denoiser.inputs):
      assert torch.allclose(
        denoiser.inputs[index + 1], expected, atol=1.0e-12, rtol=0.0
      )

  assert torch.equal(
    result[0][conditions.clean_mask], conditions.clean_tokens[0][conditions.clean_mask]
  )
  assert len(denoiser.step_ids) == 20
  for source, ids in zip(grid.source_ids, denoiser.step_ids, strict=True):
    assert torch.equal(ids[0, :8], torch.zeros_like(ids[0, :8]))
    assert torch.equal(ids[0, 8, 0], torch.tensor(0))
    assert torch.equal(ids[0, 8, 1], torch.tensor(source))
    assert torch.equal(ids[0, 9:, :], torch.full_like(ids[0, 9:, :], source))


def test_ddim_seed_reproducibility_and_clean_reapplication() -> None:
  schedule = DiffusionSchedule.build()
  grid = InferenceGrid.uniform(schedule)
  conditions = _conditions()
  first, first_diagnostics = sample_trajectory(
    _InputDependentDenoiser(),
    schedule=schedule,
    conditions=conditions,
    config=SamplerConfig(grid),
    generator=torch.Generator().manual_seed(17),
  )
  second, second_diagnostics = sample_trajectory(
    _InputDependentDenoiser(),
    schedule=schedule,
    conditions=conditions,
    config=SamplerConfig(grid),
    generator=torch.Generator().manual_seed(17),
  )
  third, _ = sample_trajectory(
    _InputDependentDenoiser(),
    schedule=schedule,
    conditions=conditions,
    config=SamplerConfig(grid),
    generator=torch.Generator().manual_seed(18),
  )
  assert torch.equal(first, second)
  assert first_diagnostics == second_diagnostics
  assert not torch.equal(first, third)
  assert torch.equal(
    first[0][conditions.clean_mask], conditions.clean_tokens[0][conditions.clean_mask]
  )


def test_non_frozen_grid_is_rejected_before_sampling() -> None:
  schedule = DiffusionSchedule.build()
  source_ids = tuple(range(1000, 0, -50))
  destination_ids = tuple(range(949, 48, -50)) + (0,)
  grid = InferenceGrid(source_ids, destination_ids)
  with pytest.raises(ValueError, match="frozen inference grid"):
    sample_trajectory(
      _InputDependentDenoiser(),
      schedule=schedule,
      conditions=_conditions(),
      config=SamplerConfig(grid),
      generator=torch.Generator().manual_seed(1),
    )


def test_nonzero_eta_and_guidance_are_rejected() -> None:
  grid = InferenceGrid.uniform(DiffusionSchedule.build())
  with pytest.raises(ValueError, match="eta"):
    SamplerConfig(grid, eta=0.1)
  with pytest.raises(ValueError, match="guidance"):
    SamplerConfig(grid, guidance_strength=0.1)
