"""Offline generated-token evaluation for the opt-in X2 x0 flow family.

The evaluator conditions the Stage 1 sampler on the fixed D1 entries and
descends the flow grid from Gaussian noise to clean level 0.  Every score is a
token-space error on generated *unknown* entries; a training loss, a velocity
error or any other objective used during fitting is never reported here.  The
baselines use conditioning information only, so a model score can be compared
with a trivial predictor.

Held-out future values never reach the network.  The sampler starts from one
Gaussian draw, restores only fixed entries with :func:`apply_clean_mask`, and
leaves every unknown entry at its sample.  Changing an unknown target therefore
changes the scores without changing the generated sample, which the tests assert
directly.  The initial noise comes from a CPU generator, and windows are
processed in fixed 32-window batches, so the noise stream depends only on the
seed and the evaluated window order.

Read-only reuse from the frozen diffusion package: ``_MetricAccumulator``
(bounded per-element means) and the token-mask semantics of ``diffusion.noising``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from ..diffusion.contract import DiffusionContract
from ..diffusion.evaluation import _MetricAccumulator
from ..diffusion.noising import apply_clean_mask, clean_mask
from ..diffusion.window_dataset import WindowRecord
from .process import LinearNoiseProcess, draw_gaussian
from .sampler import (
  DEFAULT_CALLS,
  FlowConditions,
  FlowGrid,
  FlowSamplerConfig,
  sample_trajectory,
)

_BATCH_WINDOWS = 32


@dataclass(frozen=True, slots=True)
class GenerationReport:
  """Offline generated-token metrics, baselines and gate outcomes."""

  split: str
  windows: int
  seed: int
  sampler_calls: int
  grid: dict[str, object]
  process: dict[str, object]
  process_identity: str
  sample_sha256: str
  metrics: dict[str, float]
  baselines: dict[str, float]
  gates: dict[str, bool]
  diagnostics_path: Path

  def as_dict(self) -> dict[str, object]:
    """Return a JSON-compatible representation of the generation report."""
    return {
      "split": self.split,
      "windows": self.windows,
      "seed": self.seed,
      "sampler_calls": self.sampler_calls,
      "grid": dict(self.grid),
      "process": dict(self.process),
      "process_identity": self.process_identity,
      "sample_sha256": self.sample_sha256,
      "metrics": dict(self.metrics),
      "baselines": dict(self.baselines),
      "gates": dict(self.gates),
      "diagnostics_path": str(self.diagnostics_path),
    }


def _cpu_generator(seed: int) -> torch.Generator:
  """Create a CPU generator that owns one noise stream regardless of device."""
  return torch.Generator(device="cpu").manual_seed(int(seed))


def _as_window_tokens(record: Any, *, contract: DiffusionContract) -> np.ndarray:
  """Read and validate one record's token array without changing its values."""
  tokens = np.asarray(record.tokens, dtype=np.float32)
  expected = (contract.window_steps, contract.token_dimension)
  if tokens.shape != expected or not np.isfinite(tokens).all():
    raise ValueError(f"generation records must contain finite {expected} tokens")
  return tokens


def _baseline_tokens(
  clean: Tensor, *, current_index: int, state_dimension: int
) -> dict[str, Tensor]:
  """Build the three conditioning-only baselines from one clean batch.

  ``copy_current_state`` repeats the known current state over the future state
  rows, ``copy_current_latent`` repeats the last known history latent over every
  unknown latent row, and ``zero_latent`` writes zero latent.  Every baseline
  reads conditioning entries only; none reads a held-out future value.  The
  current latent is unknown under the D1 conditioning mask, which is why
  ``copy_current_latent`` uses the last history latent instead.
  """
  if current_index < 1:
    raise ValueError("baseline tokens require at least one history row")
  if not 0 < state_dimension < clean.shape[-1]:
    raise ValueError("state_dimension must lie inside the token width")
  current_state = clean[:, current_index, :state_dimension]
  copy_state = clean.clone()
  copy_state[:, current_index + 1 :, :state_dimension] = current_state[:, None, :]
  history_latent = clean[:, current_index - 1, state_dimension:]
  copy_latent = clean.clone()
  copy_latent[:, current_index:, state_dimension:] = history_latent[:, None, :]
  zero_latent = clean.clone()
  zero_latent[:, current_index:, state_dimension:] = 0.0
  return {
    "copy_current_state": copy_state,
    "copy_current_latent": copy_latent,
    "zero_latent": zero_latent,
  }


def _add_errors(
  accumulator: _MetricAccumulator,
  name: str,
  prediction: np.ndarray,
  target: np.ndarray,
  unknown: np.ndarray,
  *,
  state_dimension: int,
  scope: str | None = None,
) -> None:
  """Accumulate unknown-entry errors, optionally restricted to one slice.

  ``scope`` restricts a baseline to the slice it predicts, so a state baseline is
  not scored on latent entries and vice versa.
  """
  squared = np.square(prediction - target, dtype=np.float64)
  selection = np.array(unknown, dtype=bool, copy=True)
  if scope == "state":
    selection[..., state_dimension:] = False
  elif scope == "latent":
    selection[..., :state_dimension] = False
  elif scope is not None:
    raise ValueError(f"unknown baseline scope {scope!r}")
  accumulator.add(
    f"{name}_unknown_state_mse",
    squared[..., :state_dimension][selection[..., :state_dimension]],
  )
  accumulator.add(
    f"{name}_unknown_latent_mse",
    squared[..., state_dimension:][selection[..., state_dimension:]],
  )
  accumulator.add(f"{name}_unknown_token_mse", squared[selection])


def _baseline_scope(name: str) -> str | None:
  if name == "copy_current_state":
    return "state"
  if name in {"copy_current_latent", "zero_latent"}:
    return "latent"
  return None


def _prior_reference(
  sampling_calls: int,
  clean: Tensor,
  mask: Tensor,
  *,
  generator: torch.Generator,
) -> Tensor:
  """Reproduce the sampler's single Gaussian initialization as a baseline.

  ``sample_trajectory`` draws one Gaussian tensor from its generator before the
  first network call and restores the fixed entries.  A second generator with
  the same seed therefore yields exactly the initialization the model saw, which
  makes the prior-only baseline a like-for-like comparison.  A batch with no
  unknown entry draws no noise at all, so this baseline must not advance either.
  """
  if sampling_calls == 0:
    return clean.clone()
  noise = draw_gaussian(
    clean.shape, generator=generator, device=clean.device, dtype=clean.dtype
  )
  return apply_clean_mask(noise, clean, mask)


def evaluate_generation(
  model: Any,
  *,
  process: LinearNoiseProcess,
  records: Sequence[WindowRecord],
  contract: DiffusionContract,
  seed: int,
  device: str | torch.device,
  grid: FlowGrid | None = None,
  max_windows: int | None = None,
  diagnostics_path: str | Path | None = None,
  split: str = "",
) -> GenerationReport:
  """Score generated unknown state, latent and token entries plus baselines.

  The sampler and the prior-only baseline use separate CPU generators seeded
  with the same value, so the prior baseline sees the same initial Gaussian draw
  as the model.  ``sample_sha256`` fingerprints the generated unknown entries,
  which makes seed determinism checkable without retaining every array.
  """
  if max_windows is not None and max_windows < 0:
    raise ValueError("max_windows must be non-negative")
  if not isinstance(process, LinearNoiseProcess):
    raise TypeError("process must be a LinearNoiseProcess")
  selected = tuple(records if max_windows is None else records[:max_windows])
  count = len(selected)
  destination = (
    Path(diagnostics_path) if diagnostics_path is not None else Path("<not-written>")
  )
  # A checkpoint with max_level < DEFAULT_CALLS still has a reachable level 0,
  # so cap the default call count at the process level count.
  resolved_grid = (
    grid
    if grid is not None
    else FlowGrid.uniform(
      calls=min(DEFAULT_CALLS, process.max_level), max_level=process.max_level
    )
  )
  state_dimension = contract.projected_state_dimension
  if count == 0:
    report = GenerationReport(
      split=split,
      windows=0,
      seed=int(seed),
      sampler_calls=0,
      grid=resolved_grid.as_dict(),
      process=process.as_dict(),
      process_identity=process.identity_hash(),
      sample_sha256=hashlib.sha256(b"").hexdigest(),
      metrics={},
      baselines={},
      gates={},
      diagnostics_path=destination,
    )
    _write_generation_diagnostics(destination, report)
    return report

  torch_device = torch.device(device)
  if torch_device.type == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA device requested for generation but CUDA is unavailable")
  if hasattr(model, "to"):
    model.to(torch_device)
  mask = clean_mask(
    current_index=contract.current_index,
    state_dimension=state_dimension,
    token_dimension=contract.token_dimension,
    device=torch_device,
  )
  mask_unknown = (~mask).detach().cpu().numpy().astype(bool)
  sampler_generator = _cpu_generator(seed)
  prior_generator = _cpu_generator(seed)
  accumulator = _MetricAccumulator()
  sample_digest = hashlib.sha256()
  sampler_calls = 0
  model_was_training = bool(getattr(model, "training", False))
  if hasattr(model, "eval"):
    model.eval()
  try:
    with torch.no_grad():
      for start in range(0, count, _BATCH_WINDOWS):
        stop = min(count, start + _BATCH_WINDOWS)
        batch = np.stack(
          [
            _as_window_tokens(record, contract=contract)
            for record in selected[start:stop]
          ]
        )
        clean = torch.from_numpy(batch).to(device=torch_device, dtype=torch.float32)
        generated, diagnostics = sample_trajectory(
          model,
          conditions=FlowConditions(
            clean_tokens=clean,
            clean_mask=mask,
            state_dimension=state_dimension,
          ),
          config=FlowSamplerConfig(grid=resolved_grid, process=process),
          generator=sampler_generator,
        )
        if diagnostics.denoiser_calls != resolved_grid.calls:
          raise ValueError(
            "the flow sampler did not evaluate the configured grid: "
            f"{diagnostics.denoiser_calls} calls for {resolved_grid.calls} updates"
          )
        if tuple(diagnostics.source_levels) != resolved_grid.source_levels:
          raise ValueError("the flow sampler evaluated unexpected source levels")
        if diagnostics.final_level != 0:
          raise ValueError("the flow sampler did not finish at clean level 0")
        sampler_calls += diagnostics.denoiser_calls
        prior = _prior_reference(
          diagnostics.denoiser_calls, clean, mask, generator=prior_generator
        )
        target = clean.detach().cpu().numpy().astype(np.float64)
        generated_array = generated.detach().cpu().numpy().astype(np.float64)
        unknown = np.broadcast_to(mask_unknown, target.shape).copy()
        predictions: dict[str, tuple[np.ndarray, str | None]] = {
          "model": (generated_array, None),
          "prior_only": (prior.detach().cpu().numpy().astype(np.float64), None),
        }
        for name, value in _baseline_tokens(
          clean,
          current_index=contract.current_index,
          state_dimension=state_dimension,
        ).items():
          predictions[name] = (
            value.detach().cpu().numpy().astype(np.float64),
            _baseline_scope(name),
          )
        for name, (prediction, scope) in predictions.items():
          _add_errors(
            accumulator,
            name,
            prediction,
            target,
            unknown,
            state_dimension=state_dimension,
            scope=scope,
          )
        sample_digest.update(
          np.ascontiguousarray(generated_array[unknown], dtype=np.float64).tobytes()
        )
  finally:
    if hasattr(model, "train"):
      model.train(model_was_training)

  metrics = accumulator.values()
  baseline_keys = {
    "copy_current_state": "copy_current_state_unknown_state_mse",
    "copy_current_latent": "copy_current_latent_unknown_latent_mse",
    "zero_latent": "zero_latent_unknown_latent_mse",
    "prior_only_state": "prior_only_unknown_state_mse",
    "prior_only_latent": "prior_only_unknown_latent_mse",
  }
  baselines = {
    name: metrics[key] for name, key in baseline_keys.items() if key in metrics
  }
  model_state = metrics.get("model_unknown_state_mse", float("inf"))
  model_latent = metrics.get("model_unknown_latent_mse", float("inf"))
  gates = {
    "beats_prior_only": model_state < baselines.get("prior_only_state", float("inf"))
    and model_latent < baselines.get("prior_only_latent", float("inf")),
    "beats_copy_current_state": model_state
    < baselines.get("copy_current_state", float("inf")),
    "beats_copy_current_latent": model_latent
    < baselines.get("copy_current_latent", float("inf")),
    "beats_zero_latent": model_latent < baselines.get("zero_latent", float("inf")),
  }
  report = GenerationReport(
    split=split,
    windows=count,
    seed=int(seed),
    sampler_calls=sampler_calls,
    grid=resolved_grid.as_dict(),
    process=process.as_dict(),
    process_identity=process.identity_hash(),
    sample_sha256=sample_digest.hexdigest(),
    metrics=metrics,
    baselines=baselines,
    gates=gates,
    diagnostics_path=destination,
  )
  _write_generation_diagnostics(destination, report)
  return report


def _write_generation_diagnostics(path: Path, report: GenerationReport) -> None:
  """Write a generation report only when the caller supplied an output path."""
  if str(path) == "<not-written>":
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n")


__all__ = ["GenerationReport", "evaluate_generation"]
