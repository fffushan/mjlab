"""Synthetic verification of the future-predictability measurement.

The measurement claims to estimate how much of the future is *not* determined by
the clean prefix, in the same units as the generation metrics (where 1.0 equals
the unconditional future variance).  These tests pin down that behaviour on data
whose answer is known by construction:

* identical prefixes with identical futures across *different* trials must give
  ratio ~ 0 (the estimator and its Bessel correction are right);
* futures drawn independently of the prefix must give ratio ~ 1 (so 1.0 really
  is the "predict nothing" reference the evaluation uses);
* windows of one trial, which the sliding-window dataset makes near duplicates,
  must be excluded -- measured on one-hot prefixes so the neighbour distances are
  exactly 0 within a trial and exactly sqrt(2) across trials, with no reliance on
  coincidental similarity.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

MODULE_PATH = (
  Path(__file__).resolve().parents[1]
  / "scripts"
  / "experiments"
  / "diffusion_future_predictability"
  / "analyze.py"
)


def _module():
  spec = importlib.util.spec_from_file_location("future_predictability", MODULE_PATH)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _report(prefix, future, trial, phase, *, neighbours: int = 4, exclude: bool = True):
  return _module().neighbourhood_report(
    np.asarray(prefix, dtype=np.float64),
    np.asarray(future, dtype=np.float64),
    np.asarray(trial),
    np.asarray(phase),
    np.arange(len(future)),
    neighbours=neighbours,
    blocks={"all": slice(0, np.shape(future)[1])},
    exclude_same_trial=exclude,
  )


def test_exact_function_of_prefix_gives_zero_residual() -> None:
  """Duplicate (prefix, future) pairs in distinct trials are fully predictable."""
  rng = np.random.default_rng(0)
  trials, dim, reps = 40, 5, 6
  base = rng.normal(size=(trials, dim))
  mapped = np.concatenate([base[:, :2] * 3.0, base[:, 2:] * -0.5], axis=1)
  prefix = np.tile(base, (reps, 1))
  future = np.tile(mapped, (reps, 1))
  report = _report(prefix, future, np.arange(len(prefix)), ["clean"] * len(prefix))
  assert report["ratio_all"] < 0.05, report["ratio_all"]


def test_independent_future_gives_full_residual() -> None:
  """An unpredictable future must land on 1.0, the evaluation's mean reference."""
  rng = np.random.default_rng(1)
  trials, dim = 300, 5
  prefix = rng.normal(size=(trials, dim))
  future = rng.normal(size=(trials, dim))
  report = _report(prefix, future, np.arange(trials), ["clean"] * trials)
  assert 0.85 < report["ratio_all"] < 1.15, report["ratio_all"]


def test_same_trial_neighbours_are_excluded() -> None:
  """One-hot prefixes: within-trial distance is exactly 0, across it is sqrt(2).

  The invariant is reported directly rather than inferred from a ratio, because
  with ties on the cross-trial distance the identity of the picked rows depends
  on tie-breaking order and is not a reliable statistical signal.
  """
  trials, dim, reps = 50, 50, 3
  rng = np.random.default_rng(2)
  prefix = np.repeat(np.eye(trials, dim), reps, axis=0)
  future = np.repeat(rng.normal(size=(trials, 4)), reps, axis=0)
  trial = np.repeat(np.arange(trials), reps)
  phase = ["clean"] * len(prefix)
  allowed = _report(prefix, future, trial, phase, exclude=False)
  excluded = _report(prefix, future, trial, phase, exclude=True)
  assert allowed["picked_same_trial_fraction"] > 0.0
  assert excluded["picked_same_trial_fraction"] == 0.0
  # With the exclusion every pick is in another trial, i.e. at distance sqrt(2).
  assert excluded["neighbour_distance_median"] > 1.0
  assert allowed["neighbour_distance_median"] == 0.0


def test_state_and_latent_blocks_are_distinguished() -> None:
  """Blocks must slice channels, not steps: a state-only signal must not leak.

  The state channels are an exact function of the prefix while the latent
  channels are pure noise, so a correct implementation reports ~0 for the state
  block and ~1 for the latent block.  A channel/step indexing mistake would
  report the same value for both.
  """
  rng = np.random.default_rng(4)
  windows, steps, state_dim, latent_dim = 300, 8, 8, 4
  channels = state_dim + latent_dim
  base = rng.normal(size=(windows, state_dim))
  prefix = np.tile(base[:, :3], (3, 1))
  future = np.zeros((windows * 3, steps, channels))
  for replica in range(3):
    sl = slice(replica * windows, (replica + 1) * windows)
    future[sl, :, :state_dim] = base[:, None, :] * 2.0
    future[sl, :, state_dim:] = rng.normal(size=(windows, steps, latent_dim))
  trial = np.arange(len(prefix))
  phase = np.array(["clean"] * len(prefix))
  report = _module().neighbourhood_report(
    prefix,
    future,
    trial,
    phase,
    np.arange(len(prefix)),
    neighbours=2,
    blocks={"state": slice(0, state_dim), "latent": slice(state_dim, channels)},
  )
  assert report["ratio_state"] < 0.05, report["ratio_state"]
  assert report["ratio_latent"] > 0.85, report["ratio_latent"]


def test_split_conditioning_shapes_match_the_contract_convention() -> None:
  module = _module()
  windows, steps, channels = 7, 41, 167
  array = np.arange(windows * steps * channels, dtype=np.float32).reshape(
    windows, steps, channels
  )
  prefix, future = module.split_conditioning(
    array, current_index=8, state_dimension=135
  )
  assert prefix.shape == (windows, 8 * channels + 135)
  assert future.shape == (windows, steps - 9, channels)
  # The prefix must contain the past tokens and the current state only.
  assert np.array_equal(prefix[:, : 8 * channels], array[:, :8, :].reshape(windows, -1))
  assert np.array_equal(prefix[:, 8 * channels :], array[:, 8, :135])
  # The future must start one step after the current index.
  assert np.array_equal(future, array[:, 9:, :])


class _FakeView:
  """Minimal stand-in for a loaded dataset's per-split view."""

  def __init__(self, tokens):
    self._tokens = tokens

  def __getitem__(self, index):
    class _Sample:
      tokens = None

    sample = _Sample()
    sample.tokens = self._tokens[index]
    return sample


class _FakeLoaded:
  def __init__(self, tokens):
    self._view = _FakeView(tokens)

  def dataset(self, split):  # noqa: ARG002
    return self._view


def test_verify_cache_alignment_measures_row_identity() -> None:
  module = _module()
  tokens = np.arange(4 * 41 * 3, dtype=np.float32).reshape(4, 41, 3)
  message = module.verify_cache_alignment(
    loaded=_FakeLoaded(tokens), tokens=tokens, split="validation", count=4
  )
  assert "4 rows compared" in message
  assert "0.000e+00" in message


def test_verify_cache_alignment_detects_a_permuted_cache() -> None:
  """A reordered cache must be reported, not silently accepted."""
  module = _module()
  tokens = np.arange(4 * 41 * 3, dtype=np.float32).reshape(4, 41, 3)
  shuffled = tokens[[1, 0, 3, 2]]
  message = module.verify_cache_alignment(
    loaded=_FakeLoaded(tokens), tokens=shuffled, split="validation", count=4
  )
  assert "0.000e+00" not in message
