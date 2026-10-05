"""Focused tests for the generated-diversity decomposition.

The arithmetic decides whether a low normalised MSE means skill or regression to
the mean, so it is pinned here rather than trusted.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

SCRIPT = (
  pathlib.Path(__file__).resolve().parents[1]
  / "scripts"
  / "experiments"
  / "diffusion_generated_diversity"
  / "analyze.py"
)


def _module():
  spec = importlib.util.spec_from_file_location(
    "generated_diversity_under_test", SCRIPT
  )
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class _Contract:
  def __init__(self, state: int, total: int) -> None:
    self.projected_state_dimension = state
    self.token_dimension = total


def test_block_slices_splits_state_then_latent() -> None:
  module = _module()
  slices = module.block_slices(_Contract(135, 167))
  assert slices["state"] == slice(0, 135)
  assert slices["latent"] == slice(135, 167)


@pytest.mark.parametrize("state,total", [(0, 167), (167, 167), (200, 167)])
def test_block_slices_rejects_a_bad_layout(state: int, total: int) -> None:
  module = _module()
  with pytest.raises(ValueError, match="state-then-latent"):
    module.block_slices(_Contract(state, total))


def test_spread_recovers_the_sampling_noise_and_bias_is_zero() -> None:
  module = _module()
  rng = np.random.default_rng(0)
  sigma = 0.37
  target = rng.normal(size=(5, 3))
  samples = target[None] + rng.normal(scale=sigma, size=(8, 5, 3))
  stats = module.dispersion_decomposition(samples, target, np.ones_like(target, bool))
  assert stats["spread"] == pytest.approx(sigma**2, rel=0.25)
  # The mean of the samples is within sampling error of the target, so bias2 is
  # small compared with the spread.
  assert stats["bias2"] < 0.2 * sigma**2
  assert stats["elements"] == target.size


def test_identical_samples_have_zero_spread() -> None:
  module = _module()
  target = np.zeros((4, 2))
  samples = np.full((6, 4, 2), 3.0)
  stats = module.dispersion_decomposition(samples, target, np.ones_like(target, bool))
  assert stats["spread"] == 0.0
  assert stats["bias2"] == pytest.approx(9.0)


def test_error_identity_holds_with_the_unbiased_correction() -> None:
  """error == spread * (K-1)/K + bias2 for the unbiased spread."""
  module = _module()
  rng = np.random.default_rng(1)
  target = rng.normal(size=(6, 4))
  samples = target[None] + rng.normal(scale=0.5, size=(5, 6, 4)) + 0.3
  unknown = np.ones_like(target, bool)
  stats = module.dispersion_decomposition(samples, target, unknown)
  k = samples.shape[0]
  error = float(((samples - target[None]) ** 2).mean())
  assert stats["spread"] * (k - 1) / k + stats["bias2"] == pytest.approx(
    error, rel=1e-9
  )


def test_single_sample_reports_no_spread() -> None:
  module = _module()
  target = np.zeros((3, 2))
  samples = np.ones((1, 3, 2))
  stats = module.dispersion_decomposition(samples, target, np.ones_like(target, bool))
  assert stats["spread"] == 0.0
  assert stats["bias2"] == pytest.approx(1.0)
  assert stats["elements"] == 6


def test_spread_uses_only_unknown_elements() -> None:
  """Spread must not be diluted by the (clamped, seed-identical) known entries.

  The samples vary only at the single unknown element, so a denominator that
  counted all four entries would report 2.0 instead of 8.0.
  """
  module = _module()
  target = np.zeros((2, 2))
  unknown = np.array([[True, False], [False, False]])
  samples = np.stack(
    [np.array([[2.0, 0.0], [0.0, 0.0]]), np.array([[-2.0, 0.0], [0.0, 0.0]])]
  )
  stats = module.dispersion_decomposition(samples, target, unknown)
  assert stats["elements"] == 1
  assert stats["bias2"] == pytest.approx(0.0)
  assert stats["spread"] == pytest.approx(8.0)


def test_decomposition_ignores_wrong_known_entries_in_bias() -> None:
  module = _module()
  target = np.zeros((2, 2))
  unknown = np.array([[True, False], [False, False]])
  # Known entries are wildly off but seed-identical: they must not enter bias2.
  samples = np.stack([np.full((2, 2), 5.0), np.full((2, 2), 1.0)])
  stats = module.dispersion_decomposition(samples, target, unknown)
  assert stats["elements"] == 1
  assert stats["bias2"] == pytest.approx(9.0)  # mean of {5, 1} is 3, squared
  assert stats["spread"] == pytest.approx(8.0)


def test_empty_unknown_region_is_an_error() -> None:
  module = _module()
  with pytest.raises(ValueError, match="no unknown elements"):
    module.dispersion_decomposition(
      np.zeros((2, 3)), np.zeros(3), np.zeros(3, dtype=bool)
    )
