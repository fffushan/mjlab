"""CPU-only generation-evaluation semantics for the x0 flow family."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from mjlab.tasks.tracking.diffusion.contract import DiffusionContract
from mjlab.tasks.tracking.diffusion.noising import clean_mask
from mjlab.tasks.tracking.diffusion.window_dataset import WindowRecord
from mjlab.tasks.tracking.flow_matching import (
  FlowGrid,
  GenerationReport,
  LinearNoiseProcess,
  evaluate_generation,
)
from mjlab.tasks.tracking.flow_matching.evaluation import _baseline_tokens

CONTRACT = DiffusionContract()
STATE = CONTRACT.projected_state_dimension
TOKENS = CONTRACT.token_dimension
CURRENT = CONTRACT.current_index


def _tokens(count: int = 2, seed: int = 0) -> torch.Tensor:
  return torch.randn(
    (count, CONTRACT.window_steps, TOKENS),
    generator=torch.Generator().manual_seed(seed),
  )


def _records(tokens: torch.Tensor, split: str = "validation") -> list[WindowRecord]:
  return [
    WindowRecord(
      tokens=row.numpy(),
      split=split,
      group_key="group",
      motion_id="tennis_000",
      phase="clean",
      ou_noise_norm=0.0,
      pair_id="pair",
      start_tick=index,
    )
    for index, row in enumerate(tokens)
  ]


def _unknown(count: int) -> np.ndarray:
  mask = clean_mask(
    current_index=CURRENT, state_dimension=STATE, token_dimension=TOKENS, device="cpu"
  )
  return np.broadcast_to((~mask).numpy(), (count, CONTRACT.window_steps, TOKENS)).copy()


class ZeroDenoiser(nn.Module):
  """Predicts x0 = 0, so the terminal Euler update returns zeros exactly."""

  def forward(self, tokens: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
    del step_ids
    return torch.zeros_like(tokens)


class TinyDenoiser(nn.Module):
  """Pointwise model whose weights come from the ambient torch seed."""

  def __init__(self) -> None:
    super().__init__()
    self.linear = nn.Linear(TOKENS, TOKENS)

  def forward(self, tokens: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
    del step_ids
    return self.linear(tokens)


def _run(
  model: nn.Module,
  tokens: torch.Tensor,
  *,
  seed: int = 3,
  grid: FlowGrid | None = None,
  process: LinearNoiseProcess | None = None,
  max_windows: int | None = None,
  diagnostics_path: Path | None = None,
):
  return evaluate_generation(
    model,
    process=LinearNoiseProcess() if process is None else process,
    records=_records(tokens),
    contract=CONTRACT,
    seed=seed,
    device="cpu",
    grid=grid,
    max_windows=max_windows,
    diagnostics_path=diagnostics_path,
    split="validation",
  )


def test_unknown_entries_are_scored_and_known_entries_are_excluded() -> None:
  tokens = _tokens(count=2, seed=1)
  report = _run(ZeroDenoiser(), tokens)
  target = tokens.numpy().astype(np.float64)
  unknown = _unknown(2)
  squared = np.square(target)
  assert report.windows == 2
  assert report.metrics["model_unknown_state_mse"] == pytest.approx(
    squared[..., :STATE][unknown[..., :STATE]].mean(), rel=1e-12
  )
  assert report.metrics["model_unknown_latent_mse"] == pytest.approx(
    squared[..., STATE:][unknown[..., STATE:]].mean(), rel=1e-12
  )
  assert report.metrics["model_unknown_token_mse"] == pytest.approx(
    squared[unknown].mean(), rel=1e-12
  )
  # Known entries are restored from clean and would contribute zero error, so a
  # score that included them would be strictly lower.
  including_known = np.concatenate([squared[unknown], np.zeros(int((~unknown).sum()))])
  assert report.metrics["model_unknown_token_mse"] > including_known.mean()


def test_baselines_are_scored_on_their_own_slice_only() -> None:
  tokens = _tokens(count=2, seed=2)
  report = _run(ZeroDenoiser(), tokens)
  target = tokens.numpy().astype(np.float64)
  unknown = _unknown(2)
  # copy_current_state repeats the known current state over future state rows.
  predicted = target.copy()
  predicted[:, CURRENT + 1 :, :STATE] = target[:, CURRENT, :STATE][:, None, :]
  assert report.baselines["copy_current_state"] == pytest.approx(
    np.square(predicted - target)[..., :STATE][unknown[..., :STATE]].mean(),
    rel=1e-12,
  )
  # copy_current_latent repeats the last known history latent.
  predicted = target.copy()
  predicted[:, CURRENT:, STATE:] = target[:, CURRENT - 1, STATE:][:, None, :]
  assert report.baselines["copy_current_latent"] == pytest.approx(
    np.square(predicted - target)[..., STATE:][unknown[..., STATE:]].mean(),
    rel=1e-12,
  )
  # zero_latent writes zero latent; no baseline is scored on the other slice.
  assert report.baselines["zero_latent"] == pytest.approx(
    np.square(target[..., STATE:])[unknown[..., STATE:]].mean(), rel=1e-12
  )
  assert set(report.gates) == {
    "beats_prior_only",
    "beats_copy_current_state",
    "beats_copy_current_latent",
    "beats_zero_latent",
  }
  # Every baseline reads conditioning entries only: changing the unknown targets
  # cannot change the value a baseline predicts there.
  changed = tokens.clone()
  changed[:, CURRENT:, STATE:] += 5.0
  changed[:, CURRENT + 1 :, :STATE] -= 7.0
  first = _baseline_tokens(tokens, current_index=CURRENT, state_dimension=STATE)
  second = _baseline_tokens(changed, current_index=CURRENT, state_dimension=STATE)
  assert torch.equal(
    first["copy_current_state"][:, CURRENT + 1 :, :STATE],
    second["copy_current_state"][:, CURRENT + 1 :, :STATE],
  )
  assert torch.equal(
    first["copy_current_latent"][:, CURRENT:, STATE:],
    second["copy_current_latent"][:, CURRENT:, STATE:],
  )
  assert torch.equal(
    second["zero_latent"][:, CURRENT:, STATE:],
    torch.zeros_like(second["zero_latent"][:, CURRENT:, STATE:]),
  )


def test_seed_grid_and_no_future_leak() -> None:
  torch.manual_seed(23)
  model = TinyDenoiser()
  tokens = _tokens(count=3, seed=6)
  grid = FlowGrid.uniform(calls=4, max_level=1000)
  first = _run(model, tokens, seed=13, grid=grid)
  second = _run(model, tokens, seed=13, grid=grid)
  third = _run(model, tokens, seed=14, grid=grid)
  assert first.sampler_calls == 4
  assert first.grid == {"levels": [1000, 750, 500, 250, 0], "calls": 4}
  assert first.process == LinearNoiseProcess().as_dict()
  assert first.process_identity == LinearNoiseProcess().identity_hash()
  assert first.sample_sha256 == second.sample_sha256
  assert first.metrics == second.metrics
  assert third.sample_sha256 != first.sample_sha256
  # Changing every unknown target changes the score but not the generated
  # sample: future values never reach the network through the conditions.
  changed = tokens.clone()
  changed[:, CURRENT:, STATE:] += 4.0
  changed[:, CURRENT + 1 :, :STATE] -= 3.0
  perturbed = _run(model, changed, seed=13, grid=grid)
  assert perturbed.sample_sha256 == first.sample_sha256
  assert perturbed.metrics["model_unknown_token_mse"] != pytest.approx(
    first.metrics["model_unknown_token_mse"]
  )


def test_diagnostics_restore_known_entries_and_bound_invalid_requests(
  tmp_path: Path,
) -> None:
  class RecordingDenoiser(nn.Module):
    def __init__(self) -> None:
      super().__init__()
      self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, tokens: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
      self.calls.append((tokens.detach().clone(), step_ids.detach().clone()))
      return torch.zeros_like(tokens)

  model = RecordingDenoiser()
  tokens = _tokens(count=2, seed=8)
  path = tmp_path / "generation.json"
  report = _run(model, tokens, diagnostics_path=path)
  assert isinstance(report, GenerationReport)
  mask = clean_mask(
    current_index=CURRENT, state_dimension=STATE, token_dimension=TOKENS, device="cpu"
  )
  assert len(model.calls) == 20
  for value, step_ids in model.calls:
    # Every network input carries the clean conditioning values on known entries.
    assert torch.equal(value[:, mask], tokens[:, mask])
    # The network is never evaluated at clean level 0 on an unknown entry.
    assert bool((step_ids[:, ~mask[:, :STATE].all(dim=-1), 0] > 0).all())
    assert bool((step_ids[:, ~mask[:, STATE:].all(dim=-1), 1] > 0).all())
  payload = json.loads(path.read_text())
  assert payload["windows"] == 2
  assert payload["sampler_calls"] == 20
  assert payload["split"] == "validation"
  assert payload["sample_sha256"] == report.sample_sha256
  assert payload["grid"] == {"levels": list(range(1000, -1, -50)), "calls": 20}
  assert _run(ZeroDenoiser(), tokens, max_windows=0).metrics == {}
  with pytest.raises(ValueError, match="non-negative"):
    _run(ZeroDenoiser(), tokens, max_windows=-1)
  # A process with max_level < 20 still reaches level 0: the default grid uses
  # min(20, max_level) calls, so such a checkpoint stays evaluable.
  small = _run(ZeroDenoiser(), tokens, process=LinearNoiseProcess(max_level=7))
  assert small.sampler_calls == 7
  assert small.grid == {"levels": [7, 6, 5, 4, 3, 2, 1, 0], "calls": 7}
