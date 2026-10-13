"""Oracle tests for the linear Gaussian flow-matching process primitives."""

from __future__ import annotations

import pytest
import torch

from mjlab.tasks.tracking.flow_matching.process import (
  DEFAULT_MAX_LEVEL,
  LinearNoiseProcess,
  add_linear_noise,
  sample_independent_levels,
)

_INTEGER_VIEW = {2: torch.int16, 4: torch.int32, 8: torch.int64}


def _bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
  """Compare value and bit pattern, so exact endpoint claims stay strict."""
  if left.shape != right.shape or left.dtype != right.dtype:
    return False
  view = _INTEGER_VIEW[left.element_size()]
  return bool(torch.equal(left.contiguous().view(view), right.contiguous().view(view)))


def test_process_identity_records_the_fixed_linear_contract() -> None:
  process = LinearNoiseProcess()
  metadata = process.as_dict()
  assert metadata["prediction_type"] == "x0"
  assert metadata["path"] == "linear_gaussian"
  assert metadata["loss"] == "x0_mse"
  assert metadata["time_encoding"] == "integer_noise_fraction"
  assert metadata["max_level"] == DEFAULT_MAX_LEVEL == 1000
  assert metadata["embedding_count"] == 1001
  assert process.identity_hash() == LinearNoiseProcess().identity_hash()
  digest = process.identity_hash()
  assert len(digest) == 64
  assert all(character in "0123456789abcdef" for character in digest)
  assert LinearNoiseProcess(max_level=999).identity_hash() != digest

  with pytest.raises(ValueError):
    LinearNoiseProcess(max_level=0)
  with pytest.raises(ValueError):
    LinearNoiseProcess(max_level=True)


def test_noise_fraction_is_exactly_level_over_max_level() -> None:
  process = LinearNoiseProcess()
  assert process.noise_fraction_at(0) == 0.0
  assert process.noise_fraction_at(250) == 0.25
  assert process.noise_fraction_at(500) == 0.5
  assert process.noise_fraction_at(1000) == 1.0

  levels = torch.tensor([0, 250, 1000], dtype=torch.long)
  fractions = process.noise_fractions_torch(levels, dtype=torch.float64)
  assert torch.equal(fractions, torch.tensor([0.0, 0.25, 1.0], dtype=torch.float64))

  for invalid in (-1, 1001):
    with pytest.raises(ValueError):
      process.noise_fraction_at(invalid)
  with pytest.raises(ValueError):
    process.noise_fraction_at(1.5)  # type: ignore[invalid-argument-type]
  with pytest.raises(ValueError):
    process.noise_fractions_torch(
      torch.tensor([0.0, 1.0], dtype=torch.float32), dtype=torch.float32
    )
  with pytest.raises(ValueError):
    process.noise_fractions_torch(
      torch.tensor([0, 1001], dtype=torch.long), dtype=torch.float32
    )


def test_levels_are_independent_state_latent_and_reproducible() -> None:
  shape = (10_000, 2)
  first = sample_independent_levels(
    shape, generator=torch.Generator().manual_seed(2026)
  )
  second = sample_independent_levels(
    shape, generator=torch.Generator().manual_seed(2026)
  )
  other = sample_independent_levels(
    shape, generator=torch.Generator().manual_seed(2027)
  )
  assert torch.equal(first.state, second.state)
  assert torch.equal(first.latent, second.latent)
  assert not torch.equal(first.state, other.state)

  for levels in (first.state, first.latent):
    assert levels.dtype == torch.long
    assert levels.shape == shape
    assert int(levels.min()) == 0
    assert int(levels.max()) == DEFAULT_MAX_LEVEL
    assert abs(float(levels.to(torch.float64).mean()) - 500.0) < 8.0

  # Independent draws agree only by coincidence, at the uniform rate.
  agreement = float((first.state == first.latent).to(torch.float64).mean())
  assert abs(agreement - 1.0 / 1001.0) < 0.001

  with pytest.raises(ValueError):
    sample_independent_levels((), generator=torch.Generator().manual_seed(1))
  with pytest.raises(ValueError):
    sample_independent_levels((0, 3), generator=torch.Generator().manual_seed(1))
  with pytest.raises(ValueError):
    sample_independent_levels((-2, 3), generator=torch.Generator().manual_seed(1))


def test_noising_endpoints_are_exact_clean_and_exact_noise() -> None:
  process = LinearNoiseProcess()
  state = torch.randn((3, 41, 135), dtype=torch.float64)
  latent = torch.randn((3, 41, 32), dtype=torch.float64)
  clean_levels = torch.zeros((3, 41), dtype=torch.long)
  clean = add_linear_noise(
    state,
    latent,
    state_levels=clean_levels,
    latent_levels=clean_levels,
    process=process,
    generator=torch.Generator().manual_seed(4),
  )
  assert _bitwise_equal(clean.state, state)
  assert _bitwise_equal(clean.latent, latent)
  assert torch.equal(clean.tokens, torch.cat((state, latent), dim=-1))

  full_levels = torch.full((3, 41), DEFAULT_MAX_LEVEL, dtype=torch.long)
  noised = add_linear_noise(
    state,
    latent,
    state_levels=full_levels,
    latent_levels=full_levels,
    process=process,
    generator=torch.Generator().manual_seed(4),
  )
  reference = torch.Generator().manual_seed(4)
  expected_state = torch.randn(state.shape, generator=reference, dtype=torch.float64)
  expected_latent = torch.randn(latent.shape, generator=reference, dtype=torch.float64)
  assert torch.equal(noised.noise_state, expected_state)
  assert torch.equal(noised.noise_latent, expected_latent)
  assert torch.equal(noised.state, expected_state)
  assert torch.equal(noised.latent, expected_latent)
  assert torch.equal(
    noised.tokens, torch.cat((expected_state, expected_latent), dim=-1)
  )
  assert not torch.equal(noised.state, state)


def test_noising_matches_the_analytic_linear_marginal_at_level_500() -> None:
  process = LinearNoiseProcess()
  count = 8_000
  clean_state_value = 1.25
  state = torch.full((count, 1, 135), clean_state_value, dtype=torch.float64)
  latent = torch.zeros((count, 1, 32), dtype=torch.float64)
  levels = torch.full((count, 1), 500, dtype=torch.long)
  noised = add_linear_noise(
    state,
    latent,
    state_levels=levels,
    latent_levels=levels,
    process=process,
    generator=torch.Generator().manual_seed(9),
  )
  fraction = 0.5
  observed = noised.state[:, 0, :]
  assert abs(float(observed.mean()) - (1.0 - fraction) * clean_state_value) < 0.01
  assert abs(float(observed.std(unbiased=False)) - fraction) < 0.01
  observed_latent = noised.latent[:, 0, :]
  assert abs(float(observed_latent.mean())) < 0.01
  assert abs(float(observed_latent.std(unbiased=False)) - fraction) < 0.01


def test_noising_uses_one_level_per_row_and_independent_draws() -> None:
  state = torch.randn((2, 3, 4), dtype=torch.float32)
  latent = torch.randn((2, 3, 2), dtype=torch.float32)
  state_levels = torch.tensor([[0, 1000, 500], [500, 0, 1000]], dtype=torch.long)
  latent_levels = torch.tensor([[1000, 0, 500], [500, 1000, 0]], dtype=torch.long)
  noised = add_linear_noise(
    state,
    latent,
    state_levels=state_levels,
    latent_levels=latent_levels,
    generator=torch.Generator().manual_seed(3),
  )

  zero_state = state_levels == 0
  assert torch.equal(noised.state[zero_state], state[zero_state])
  full_state = state_levels == 1000
  assert torch.equal(noised.state[full_state], noised.noise_state[full_state])
  half_state = state_levels == 500
  assert torch.equal(
    noised.state[half_state],
    0.5 * state[half_state] + 0.5 * noised.noise_state[half_state],
  )
  zero_latent = latent_levels == 0
  assert torch.equal(noised.latent[zero_latent], latent[zero_latent])

  assert noised.noise_state.data_ptr() != noised.noise_latent.data_ptr()
  assert noised.tokens.shape == (2, 3, 6)
  assert torch.equal(noised.tokens, torch.cat((noised.state, noised.latent), dim=-1))
  assert torch.equal(noised.state_levels, state_levels)
  assert torch.equal(noised.latent_levels, latent_levels)


def test_noising_rejects_invalid_inputs() -> None:
  state = torch.randn((2, 3, 4), dtype=torch.float32)
  latent = torch.randn((2, 3, 2), dtype=torch.float32)
  levels = torch.zeros((2, 3), dtype=torch.long)
  generator = torch.Generator().manual_seed(1)

  def build(**overrides: torch.Tensor) -> None:
    add_linear_noise(
      overrides.get("state", state),
      overrides.get("latent", latent),
      state_levels=overrides.get("state_levels", levels),
      latent_levels=overrides.get("latent_levels", levels),
      generator=generator,
    )

  with pytest.raises(ValueError):
    build(state_levels=torch.zeros((3,), dtype=torch.long))
  with pytest.raises(ValueError):
    build(latent_levels=torch.zeros((2, 2), dtype=torch.long))
  with pytest.raises(ValueError):
    build(state_levels=torch.full((2, 3), 1001, dtype=torch.long))
  with pytest.raises(ValueError):
    build(state_levels=torch.full((2, 3), -1, dtype=torch.long))
  with pytest.raises(ValueError):
    build(state_levels=torch.zeros((2, 3), dtype=torch.float32))
  with pytest.raises(ValueError):
    build(state_levels=torch.zeros((2, 3), dtype=torch.bool))
  with pytest.raises(ValueError):
    build(latent=latent[:1])
  with pytest.raises(ValueError):
    build(state=state.to(torch.float64))
  with pytest.raises(ValueError):
    build(state=state.to(torch.int64))

  non_finite = state.clone()
  non_finite[0, 0, 0] = float("nan")
  with pytest.raises(ValueError):
    build(state=non_finite)
  non_finite_latent = latent.clone()
  non_finite_latent[1, 2, 1] = float("inf")
  with pytest.raises(ValueError):
    build(latent=non_finite_latent)
