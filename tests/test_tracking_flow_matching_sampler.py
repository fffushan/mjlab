"""Oracle tests for the x0 Euler flow sampler and its level grid."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor

from mjlab.tasks.tracking.diffusion.noising import clean_mask
from mjlab.tasks.tracking.flow_matching.process import (
  DEFAULT_MAX_LEVEL,
  LinearNoiseProcess,
)
from mjlab.tasks.tracking.flow_matching.sampler import (
  DEFAULT_CALLS,
  FlowConditions,
  FlowGrid,
  FlowSamplerConfig,
  euler_x0_update,
  sample_trajectory,
)

_INTEGER_VIEW = {2: torch.int16, 4: torch.int32, 8: torch.int64}
_STATE_DIMENSION = 135
_TOKEN_DIMENSION = 167
_WINDOW = 41
_CURRENT_INDEX = 8


def _bitwise_equal(left: Tensor, right: Tensor) -> bool:
  """Compare value and bit pattern, so exact conditioning stays strict."""
  if left.shape != right.shape or left.dtype != right.dtype:
    return False
  view = _INTEGER_VIEW[left.element_size()]
  return bool(torch.equal(left.contiguous().view(view), right.contiguous().view(view)))


class _OracleDenoiser:
  """Return the clean tokens and record every call."""

  def __init__(self, clean: Tensor) -> None:
    self.clean = clean
    self.inputs: list[Tensor] = []
    self.step_ids: list[Tensor] = []

  def __call__(self, sample: Tensor, step_ids: Tensor) -> Tensor:
    self.inputs.append(sample.detach().clone())
    self.step_ids.append(step_ids.detach().clone())
    return self.clean.expand_as(sample)


class _InputDependentDenoiser:
  """Return a fixed fraction of the input so updates accumulate instead."""

  def __call__(self, sample: Tensor, step_ids: Tensor) -> Tensor:
    del step_ids
    return 0.25 * sample


class _WrongShapeDenoiser:
  def __call__(self, sample: Tensor, step_ids: Tensor) -> Tensor:
    del step_ids
    return sample[..., :1]


class _NonFiniteDenoiser:
  def __call__(self, sample: Tensor, step_ids: Tensor) -> Tensor:
    del step_ids
    prediction = sample.clone()
    prediction[..., 0] = float("nan")
    return prediction


class _NonTensorDenoiser:
  def __call__(self, sample: Tensor, step_ids: Tensor) -> Any:
    del step_ids
    del sample
    return "not a tensor"


def _conditions(batch: int = 1) -> FlowConditions:
  clean = torch.randn((batch, _WINDOW, _TOKEN_DIMENSION), dtype=torch.float64)
  mask = clean_mask(
    current_index=_CURRENT_INDEX,
    state_dimension=_STATE_DIMENSION,
    token_dimension=_TOKEN_DIMENSION,
    device="cpu",
  )
  return FlowConditions(clean, mask)


def test_default_grid_is_twenty_strictly_descending_calls() -> None:
  grid = FlowGrid.uniform()
  assert grid.calls == DEFAULT_CALLS == 20
  assert grid.levels == tuple(range(1000, -1, -50))
  assert grid.source_levels == tuple(range(1000, 0, -50))
  assert grid.destination_levels == tuple(range(950, -1, -50))
  assert grid.as_dict() == {
    "levels": list(range(1000, -1, -50)),
    "calls": 20,
  }

  fine = FlowGrid.uniform(calls=40, max_level=1000)
  assert fine.calls == 40
  assert fine.levels[-1] == 0
  assert all(
    source > destination
    for source, destination in zip(
      fine.source_levels, fine.destination_levels, strict=True
    )
  )
  assert FlowGrid.uniform(calls=5).levels == (1000, 800, 600, 400, 200, 0)

  with pytest.raises(ValueError):
    FlowGrid.uniform(calls=0)
  with pytest.raises(ValueError):
    FlowGrid.uniform(calls=1001, max_level=1000)
  with pytest.raises(ValueError):
    FlowGrid.uniform(calls=20, max_level=True)
  with pytest.raises(ValueError):
    FlowGrid((1000, 900))
  with pytest.raises(ValueError):
    FlowGrid((1000, 900, 900, 0))
  with pytest.raises(ValueError):
    FlowGrid((1000, 1000, 0))
  with pytest.raises(ValueError):
    FlowGrid((0,))
  with pytest.raises(ValueError):
    FlowGrid((1000, 950.0, 0))  # type: ignore[invalid-argument-type]
  with pytest.raises(ValueError):
    FlowGrid((True, 0))


def test_sampler_config_requires_the_grid_to_start_at_max_level() -> None:
  assert FlowSamplerConfig().grid.levels[0] == DEFAULT_MAX_LEVEL
  assert FlowSamplerConfig().process == LinearNoiseProcess()

  with pytest.raises(ValueError, match="max_level"):
    FlowSamplerConfig(grid=FlowGrid((500, 250, 0)))
  with pytest.raises(ValueError, match="max_level"):
    FlowSamplerConfig(
      grid=FlowGrid.uniform(calls=20, max_level=500),
      process=LinearNoiseProcess(max_level=1000),
    )


def test_euler_update_is_exact_at_the_terminal_clean_level() -> None:
  sample = torch.randn((2, 3, _TOKEN_DIMENSION), dtype=torch.float64)
  x0_hat = torch.randn((2, 3, _TOKEN_DIMENSION), dtype=torch.float64)
  terminal = euler_x0_update(sample, x0_hat, source_levels=50, destination_levels=0)
  assert _bitwise_equal(terminal, x0_hat)

  sources = torch.tensor([[1000, 500, 100], [250, 40, 1000]], dtype=torch.long)
  destinations = torch.tensor([[500, 100, 0], [50, 0, 750]], dtype=torch.long)
  stepped = euler_x0_update(
    sample, x0_hat, source_levels=sources, destination_levels=destinations
  )
  ratio = destinations.to(torch.float64) / sources.to(torch.float64)
  expected = ratio.unsqueeze(-1) * sample + (1.0 - ratio).unsqueeze(-1) * x0_hat
  assert torch.allclose(stepped, expected, atol=1.0e-12, rtol=0.0)

  rows, slots = sources.shape
  for row in range(rows):
    for slot in range(slots):
      single = euler_x0_update(
        sample[row, slot],
        x0_hat[row, slot],
        source_levels=int(sources[row, slot]),
        destination_levels=int(destinations[row, slot]),
      )
      assert torch.allclose(single, stepped[row, slot], atol=1.0e-12, rtol=0.0)
  steady = euler_x0_update(
    sample, sample, source_levels=sources, destination_levels=destinations
  )
  assert torch.allclose(steady, sample, atol=1.0e-12, rtol=0.0)


def test_euler_update_rejects_invalid_inputs() -> None:
  sample = torch.randn((2, 2, 5), dtype=torch.float32)
  x0_hat = torch.randn((2, 2, 5), dtype=torch.float32)

  with pytest.raises(ValueError, match="source level 0"):
    euler_x0_update(sample, x0_hat, source_levels=0, destination_levels=0)
  with pytest.raises(ValueError, match="source level 0"):
    euler_x0_update(
      sample,
      x0_hat,
      source_levels=torch.tensor([[0, 10], [10, 10]], dtype=torch.long),
      destination_levels=0,
    )
  with pytest.raises(ValueError, match="descend"):
    euler_x0_update(sample, x0_hat, source_levels=100, destination_levels=100)
  with pytest.raises(ValueError, match="descend"):
    euler_x0_update(sample, x0_hat, source_levels=100, destination_levels=200)
  with pytest.raises(ValueError, match="destination_levels"):
    euler_x0_update(sample, x0_hat, source_levels=100, destination_levels=-5)
  with pytest.raises(ValueError):
    euler_x0_update(sample, x0_hat, source_levels=1001, destination_levels=0)
  with pytest.raises(ValueError):
    euler_x0_update(
      sample,
      x0_hat,
      source_levels=torch.tensor([100.0]),
      destination_levels=0,
    )
  with pytest.raises(ValueError):
    euler_x0_update(
      sample,
      x0_hat,
      source_levels=torch.tensor([[[100]]], dtype=torch.long),
      destination_levels=0,
    )
  with pytest.raises(ValueError):
    euler_x0_update(sample, x0_hat[:, :1], source_levels=100, destination_levels=0)
  with pytest.raises(ValueError):
    euler_x0_update(
      sample, x0_hat.to(torch.int64), source_levels=100, destination_levels=0
    )
  with pytest.raises(ValueError):
    euler_x0_update(
      sample, x0_hat.to(torch.float64), source_levels=100, destination_levels=0
    )

  non_finite = sample.clone()
  non_finite[0, 0, 0] = float("inf")
  with pytest.raises(ValueError):
    euler_x0_update(non_finite, x0_hat, source_levels=100, destination_levels=0)
  non_finite_prediction = x0_hat.clone()
  non_finite_prediction[1, 1, 4] = float("nan")
  with pytest.raises(ValueError):
    euler_x0_update(
      sample, non_finite_prediction, source_levels=100, destination_levels=0
    )


def test_oracle_sampler_reproduces_the_euler_recurrence_and_conditions() -> None:
  conditions = _conditions()
  config = FlowSamplerConfig()
  denoiser = _OracleDenoiser(conditions.clean_tokens)
  result, diagnostics = sample_trajectory(
    denoiser,
    conditions=conditions,
    config=config,
    generator=torch.Generator().manual_seed(11),
  )
  assert result.shape == conditions.clean_tokens.shape
  assert diagnostics.denoiser_calls == 20
  assert diagnostics.source_levels == tuple(range(1000, 0, -50))
  assert diagnostics.destination_levels == tuple(range(950, -1, -50))
  assert diagnostics.final_level == 0
  assert min(diagnostics.source_levels) > 0

  # The first input is the level-1000 Gaussian draw outside the known slices.
  reference = torch.Generator().manual_seed(11)
  noise = torch.randn(
    conditions.clean_tokens.shape, generator=reference, dtype=torch.float64
  )
  initial = torch.where(conditions.clean_mask, conditions.clean_tokens, noise)
  assert torch.equal(denoiser.inputs[0], initial)

  grid = config.grid
  clean = conditions.clean_tokens
  for index, (source, destination) in enumerate(
    zip(grid.source_levels, grid.destination_levels, strict=True)
  ):
    current = denoiser.inputs[index]
    if destination == 0:
      expected = clean.clone()
    else:
      ratio = destination / source
      expected = ratio * current + (1.0 - ratio) * clean
    expected = torch.where(conditions.clean_mask, clean, expected)
    if index + 1 < len(denoiser.inputs):
      assert torch.allclose(
        denoiser.inputs[index + 1], expected, atol=1.0e-12, rtol=0.0
      )

  # A perfect x0 oracle ends exactly on the clean tokens for unknown slices.
  unknown = ~conditions.clean_mask
  assert torch.equal(result[0][unknown], clean[0][unknown])
  for tensor in (*denoiser.inputs, result):
    assert _bitwise_equal(
      tensor[0][conditions.clean_mask], clean[0][conditions.clean_mask]
    )


def test_step_ids_use_zero_on_known_slices_and_never_zero_on_unknown() -> None:
  conditions = _conditions()
  denoiser = _OracleDenoiser(conditions.clean_tokens)
  _, diagnostics = sample_trajectory(
    denoiser,
    conditions=conditions,
    config=FlowSamplerConfig(),
    generator=torch.Generator().manual_seed(12),
  )
  history = _CURRENT_INDEX
  mask = conditions.clean_mask
  fixed_state = mask[..., :_STATE_DIMENSION].all(dim=-1)
  fixed_latent = mask[..., _STATE_DIMENSION:].all(dim=-1)
  assert bool(fixed_state[: history + 1].all())
  assert bool(fixed_latent[:history].all())
  assert bool((~fixed_state[history + 1 :]).all())
  assert bool((~fixed_latent[history:]).all())
  for source, step_ids in zip(
    diagnostics.source_levels, denoiser.step_ids, strict=True
  ):
    assert step_ids.shape == (1, _WINDOW, 2)
    assert step_ids.dtype == torch.long
    assert torch.equal(
      step_ids[0, :history, :],
      torch.zeros((history, 2), dtype=torch.long),
    )
    assert torch.equal(step_ids[0, history], torch.tensor([0, source]))
    assert torch.equal(
      step_ids[0, history + 1 :, 0],
      torch.full((_WINDOW - history - 1,), source, dtype=torch.long),
    )
    assert torch.equal(
      step_ids[0, history + 1 :, 1],
      torch.full((_WINDOW - history - 1,), source, dtype=torch.long),
    )
    # The network is never conditioned at level 0 on an unknown slice.
    assert int(step_ids[0][~fixed_state, 0].min()) == source
    assert int(step_ids[0][~fixed_latent, 1].min()) == source


def test_fully_known_mask_returns_clean_without_calling_the_denoiser() -> None:
  conditions = _conditions()
  known = FlowConditions(
    conditions.clean_tokens, torch.ones_like(conditions.clean_mask)
  )
  denoiser = _OracleDenoiser(conditions.clean_tokens)
  result, diagnostics = sample_trajectory(
    denoiser,
    conditions=known,
    config=FlowSamplerConfig(),
    generator=torch.Generator().manual_seed(13),
  )
  assert denoiser.inputs == []
  assert diagnostics.denoiser_calls == 0
  assert diagnostics.source_levels == ()
  assert _bitwise_equal(result, conditions.clean_tokens)


def test_sampler_noise_and_levels_are_reproducible() -> None:
  conditions = _conditions()
  config = FlowSamplerConfig()
  denoiser = _InputDependentDenoiser()
  first, first_diagnostics = sample_trajectory(
    denoiser,
    conditions=conditions,
    config=config,
    generator=torch.Generator().manual_seed(31),
  )
  second, second_diagnostics = sample_trajectory(
    denoiser,
    conditions=conditions,
    config=config,
    generator=torch.Generator().manual_seed(31),
  )
  third, _ = sample_trajectory(
    denoiser,
    conditions=conditions,
    config=config,
    generator=torch.Generator().manual_seed(32),
  )
  assert torch.equal(first, second)
  assert first_diagnostics == second_diagnostics
  unknown = ~conditions.clean_mask
  assert not torch.equal(first[0][unknown], third[0][unknown])
  assert torch.equal(first[0][~unknown], third[0][~unknown])
  assert _bitwise_equal(first[0][~unknown], conditions.clean_tokens[0][~unknown])

  # The flow sampler consumes exactly one Gaussian draw, before the grid, so
  # the generator state after a run is reproducible for resume.
  generator = torch.Generator().manual_seed(31)
  sample_trajectory(
    _InputDependentDenoiser(),
    conditions=conditions,
    config=config,
    generator=generator,
  )
  reference = torch.Generator().manual_seed(31)
  torch.randn(conditions.clean_tokens.shape, generator=reference, dtype=torch.float64)
  assert torch.equal(generator.get_state(), reference.get_state())


def test_sampler_supports_batched_conditions_and_row_masks() -> None:
  conditions = _conditions(batch=2)
  denoiser = _OracleDenoiser(conditions.clean_tokens)
  result, diagnostics = sample_trajectory(
    denoiser,
    conditions=conditions,
    config=FlowSamplerConfig(),
    generator=torch.Generator().manual_seed(15),
  )
  assert result.shape == (2, _WINDOW, _TOKEN_DIMENSION)
  assert diagnostics.denoiser_calls == 20
  for index, source in enumerate(diagnostics.source_levels):
    assert torch.equal(
      denoiser.step_ids[index][1, _CURRENT_INDEX],
      torch.tensor([0, source]),
    )
  for row in range(2):
    assert _bitwise_equal(
      result[row][conditions.clean_mask],
      conditions.clean_tokens[row][conditions.clean_mask],
    )


def test_sampler_accepts_one_unbatched_window() -> None:
  clean = torch.randn((_WINDOW, _TOKEN_DIMENSION), dtype=torch.float64)
  mask = clean_mask(
    current_index=_CURRENT_INDEX,
    state_dimension=_STATE_DIMENSION,
    token_dimension=_TOKEN_DIMENSION,
    device="cpu",
  )
  denoiser = _OracleDenoiser(clean)
  result, diagnostics = sample_trajectory(
    denoiser,
    conditions=FlowConditions(clean, mask),
    config=FlowSamplerConfig(),
    generator=torch.Generator().manual_seed(16),
  )
  assert result.shape == (_WINDOW, _TOKEN_DIMENSION)
  assert diagnostics.denoiser_calls == 20
  assert _bitwise_equal(result[mask], clean[mask])


def test_explicit_state_dimension_controls_the_known_slice_split() -> None:
  clean = torch.randn((1, _WINDOW, _TOKEN_DIMENSION), dtype=torch.float64)
  mask = torch.zeros((_WINDOW, _TOKEN_DIMENSION), dtype=torch.bool)
  mask[:, _STATE_DIMENSION:] = True
  conditions = FlowConditions(clean, mask, state_dimension=_STATE_DIMENSION)
  denoiser = _OracleDenoiser(clean)
  result, diagnostics = sample_trajectory(
    denoiser,
    conditions=conditions,
    config=FlowSamplerConfig(),
    generator=torch.Generator().manual_seed(17),
  )
  assert diagnostics.denoiser_calls == 20
  step_ids = denoiser.step_ids[0]
  assert torch.equal(step_ids[..., 0], torch.full_like(step_ids[..., 0], 1000))
  assert torch.equal(step_ids[..., 1], torch.zeros_like(step_ids[..., 1]))
  assert _bitwise_equal(result[..., _STATE_DIMENSION:], clean[..., _STATE_DIMENSION:])
  assert torch.equal(result[..., :_STATE_DIMENSION], clean[..., :_STATE_DIMENSION])

  with pytest.raises(ValueError, match="state_dimension"):
    sample_trajectory(
      _OracleDenoiser(clean),
      conditions=FlowConditions(clean, mask, state_dimension=200),
      config=FlowSamplerConfig(),
      generator=torch.Generator().manual_seed(18),
    )


def test_sampler_rejects_invalid_conditions_and_predictions() -> None:
  conditions = _conditions()
  config = FlowSamplerConfig()
  generator = torch.Generator().manual_seed(19)

  for denoiser in (
    _WrongShapeDenoiser(),
    _NonFiniteDenoiser(),
    _NonTensorDenoiser(),
  ):
    with pytest.raises(ValueError):
      sample_trajectory(
        denoiser, conditions=conditions, config=config, generator=generator
      )

  with pytest.raises(ValueError, match="non-finite"):
    sample_trajectory(
      _OracleDenoiser(conditions.clean_tokens),
      conditions=FlowConditions(
        torch.full((1, _WINDOW, _TOKEN_DIMENSION), float("nan")),
        conditions.clean_mask,
      ),
      config=config,
      generator=generator,
    )
  with pytest.raises(ValueError, match="bool"):
    sample_trajectory(
      _OracleDenoiser(conditions.clean_tokens),
      conditions=FlowConditions(
        conditions.clean_tokens, conditions.clean_mask.to(torch.int64)
      ),
      config=config,
      generator=generator,
    )
  with pytest.raises(ValueError, match="clean_mask"):
    sample_trajectory(
      _OracleDenoiser(conditions.clean_tokens),
      conditions=FlowConditions(conditions.clean_tokens, conditions.clean_mask[:, :-1]),
      config=config,
      generator=generator,
    )
  with pytest.raises(ValueError, match="clean_tokens"):
    sample_trajectory(
      _OracleDenoiser(conditions.clean_tokens),
      conditions=FlowConditions(
        conditions.clean_tokens.unsqueeze(0), conditions.clean_mask
      ),
      config=config,
      generator=generator,
    )
  with pytest.raises(ValueError, match="state_dimension"):
    sample_trajectory(
      _OracleDenoiser(conditions.clean_tokens),
      conditions=FlowConditions(
        conditions.clean_tokens, conditions.clean_mask, state_dimension=167
      ),
      config=config,
      generator=generator,
    )
  mapping: dict[str, Any] = {
    "clean_tokens": conditions.clean_tokens,
    "clean_mask": conditions.clean_mask,
    "state_dimension": _STATE_DIMENSION,
  }
  denoiser = _OracleDenoiser(conditions.clean_tokens)
  _, diagnostics = sample_trajectory(
    denoiser, conditions=mapping, config=config, generator=generator
  )
  assert diagnostics.denoiser_calls == 20


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_euler_update_follows_the_sample_device() -> None:
  """The level ratio must not drag a CUDA sample back to the CPU level device."""
  sample = torch.full((2, 3, 4), 0.5, device="cuda")
  levels = torch.tensor([[1000], [1000]])  # integer levels stay on the CPU
  stepped = euler_x0_update(
    sample, torch.ones_like(sample), source_levels=levels, destination_levels=500
  )
  assert stepped.device == sample.device
  assert stepped.dtype == sample.dtype
  assert torch.allclose(stepped, torch.full_like(sample, 0.75))
