"""Offline generation diagnostics and bounded dataset audits for D2.

This module deliberately stays outside the simulator/runtime stack.  It consumes
materialized D1 windows, the frozen projection and the frozen DDIM sampler so
that an offline report can distinguish model-native metrics from recovered
physical-space diagnostics and from the trivial baselines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .contract import DiffusionContract
from .noising import apply_clean_mask, clean_mask
from .projection import ProjectionBundle
from .sampler import SamplerConditions, SamplerConfig, sample_trajectory
from .schedule import DiffusionSchedule, InferenceGrid
from .window_dataset import (
  DatasetSource,
  LoadedDataset,
  WindowCacheError,
  WindowRecord,
  _cache_metadata,
  _projection_identity,
  _source_dataset_hash,
  _validate_cache,
  iter_window_provenance,
  load_window_records,
)

_STATE_WIDTH = 199
_TOKEN_WIDTH = 231
_HORIZON_BUCKETS = (
  ("9_16", 9, 17),
  ("17_24", 17, 25),
  ("25_32", 25, 33),
  ("33_40", 33, 41),
)


@dataclass(frozen=True, slots=True)
class GenerationReport:
  """Offline conditioned-generation metrics and baseline gate outcomes."""

  split: str
  windows: int
  seed: int
  metrics: dict[str, float]
  baselines: dict[str, float]
  beats_trivial_copy: bool
  beats_zero_latent: bool
  diagnostics_path: Path

  def as_dict(self) -> dict[str, object]:
    """Return a JSON-compatible representation of the generation report."""
    return {
      "split": self.split,
      "windows": self.windows,
      "seed": self.seed,
      "metrics": dict(self.metrics),
      "baselines": dict(self.baselines),
      "beats_trivial_copy": self.beats_trivial_copy,
      "beats_zero_latent": self.beats_zero_latent,
      "diagnostics_path": str(self.diagnostics_path),
    }


class _MetricAccumulator:
  """Accumulate sums and element counts without retaining all predictions."""

  def __init__(self) -> None:
    self._sums: dict[str, float] = {}
    self._counts: dict[str, int] = {}

  def add(self, name: str, values: np.ndarray | Tensor | float) -> None:
    """Add finite scalar or array values under one metric name."""
    if isinstance(values, Tensor):
      array = values.detach().cpu().numpy()
    else:
      array = np.asarray(values)
    if array.ndim == 0:
      array = array.reshape(1)
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
      raise ValueError(f"generation metric {name!r} is non-finite")
    self._sums[name] = self._sums.get(name, 0.0) + float(array.sum())
    self._counts[name] = self._counts.get(name, 0) + int(array.size)

  def values(self) -> dict[str, float]:
    """Return means for all accumulated metrics."""
    return {
      name: self._sums[name] / self._counts[name]
      for name in sorted(self._sums)
      if self._counts[name] > 0
    }


class _Moments:
  """Streaming vector moments used by the bounded dataset audit."""

  def __init__(self, width: int) -> None:
    self.count = 0
    self.mean = np.zeros(width, dtype=np.float64)
    self.m2 = np.zeros(width, dtype=np.float64)
    self.minimum = np.full(width, np.inf, dtype=np.float64)
    self.maximum = np.full(width, -np.inf, dtype=np.float64)

  def update(self, values: np.ndarray) -> None:
    """Update population moments from a finite ``[N, width]`` chunk."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != self.mean.size:
      raise ValueError("audit moment chunks must have shape [N, width]")
    if not np.isfinite(values).all():
      raise ValueError("audit moment chunk contains non-finite values")
    if not len(values):
      return
    count = int(values.shape[0])
    batch_mean = values.mean(axis=0, dtype=np.float64)
    centered = values - batch_mean
    batch_m2 = np.sum(centered * centered, axis=0, dtype=np.float64)
    if self.count == 0:
      self.count = count
      self.mean = batch_mean
      self.m2 = batch_m2
    else:
      total = self.count + count
      delta = batch_mean - self.mean
      self.m2 += batch_m2 + delta * delta * (self.count * count / total)
      self.mean += delta * (count / total)
      self.count = total
    self.minimum = np.minimum(self.minimum, values.min(axis=0))
    self.maximum = np.maximum(self.maximum, values.max(axis=0))

  def summary(self, start: int, stop: int) -> dict[str, float | int]:
    """Summarize one contiguous feature slice."""
    if self.count <= 0:
      return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    values = slice(start, stop)
    std = np.sqrt(np.maximum(self.m2[values] / self.count, 0.0))
    return {
      "count": int(self.count * (stop - start)),
      "mean": float(self.mean[values].mean()),
      "std": float(std.mean()),
      "min": float(self.minimum[values].min()),
      "max": float(self.maximum[values].max()),
    }


def _cpu_generator(seed: int) -> torch.Generator:
  """Create a CPU generator for stable initial-noise streams on every device."""
  return torch.Generator(device="cpu").manual_seed(int(seed))


def _as_record_tokens(record: Any) -> np.ndarray:
  """Read and validate one record's token array without changing its values."""
  tokens = np.asarray(record.tokens, dtype=np.float32)
  if tokens.shape != (41, _TOKEN_WIDTH) or not np.isfinite(tokens).all():
    raise ValueError("generation records must contain finite [41, 231] tokens")
  return tokens


def _add_error_metrics(
  accumulator: _MetricAccumulator,
  prefix: str,
  prediction: np.ndarray,
  target: np.ndarray,
  mask: np.ndarray,
  *,
  physical_prediction: tuple[np.ndarray, np.ndarray] | None = None,
  physical_target: tuple[np.ndarray, np.ndarray] | None = None,
  physical_prefix: str | None = None,
) -> None:
  """Accumulate normalized and optionally physical state/latent MSE values."""
  squared = np.square(prediction - target, dtype=np.float64)
  state_mask = mask[..., :_STATE_WIDTH]
  latent_mask = mask[..., _STATE_WIDTH:]
  accumulator.add(
    f"{prefix}_unknown_state_mse", squared[..., :_STATE_WIDTH][state_mask]
  )
  accumulator.add(
    f"{prefix}_unknown_latent_mse", squared[..., _STATE_WIDTH:][latent_mask]
  )
  accumulator.add(f"{prefix}_unknown_token_mse", squared[mask])
  if physical_prediction is None or physical_target is None or physical_prefix is None:
    return
  predicted_state, predicted_latent = physical_prediction
  target_state, target_latent = physical_target
  state_error = np.square(predicted_state - target_state, dtype=np.float64)
  latent_error = np.square(predicted_latent - target_latent, dtype=np.float64)
  physical_state_mask = np.broadcast_to(
    np.any(state_mask, axis=-1, keepdims=True), state_error.shape
  )
  physical_latent_mask = latent_mask
  selected_state_error = state_error[physical_state_mask]
  selected_latent_error = latent_error[physical_latent_mask]
  accumulator.add(f"{physical_prefix}_unknown_state_mse", selected_state_error)
  accumulator.add(f"{physical_prefix}_unknown_latent_mse", selected_latent_error)
  accumulator.add(
    f"{physical_prefix}_unknown_token_mse",
    np.concatenate((selected_state_error, selected_latent_error), axis=0),
  )


def _baseline_tokens(clean: Tensor) -> dict[str, Tensor]:
  """Construct the four specified baselines from the same conditioned windows."""
  current = clean[:, 8]
  copy_state = clean.clone()
  copy_state[:, 9:, :_STATE_WIDTH] = current[:, None, :_STATE_WIDTH]

  copy_latent = clean.clone()
  # The current latent (token 8) is unknown under D0.  Use the last known
  # historical latent instead of leaking the held-out target into this baseline.
  history_latent = clean[:, 7, _STATE_WIDTH:]
  copy_latent[:, 8:, _STATE_WIDTH:] = history_latent[:, None, :]

  zero_latent = clean.clone()
  zero_latent[:, 8:, _STATE_WIDTH:] = 0.0
  return {
    "copy_current_state": copy_state,
    "copy_current_latent": copy_latent,
    "zero_latent": zero_latent,
  }


def _physical_tokens(
  projection: ProjectionBundle, tokens: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """Recover unnormalized physical state and latent arrays from token arrays."""
  state = projection.inverse_state(tokens[..., :_STATE_WIDTH])
  latent = projection.denormalize_latent(tokens[..., _STATE_WIDTH:])
  return state, latent


def _add_phase_metrics(
  accumulator: _MetricAccumulator,
  name: str,
  prediction: np.ndarray,
  target: np.ndarray,
  unknown: np.ndarray,
  phase_mask: np.ndarray,
  projection: ProjectionBundle,
) -> None:
  """Add per-phase metrics while retaining the exact unknown-entry mask."""
  selected = unknown & phase_mask[:, None, None]
  physical_prediction = _physical_tokens(projection, prediction)
  physical_target = _physical_tokens(projection, target)
  _add_error_metrics(
    accumulator,
    f"{name}_phase",
    prediction,
    target,
    selected,
    physical_prediction=physical_prediction,
    physical_target=physical_target,
    physical_prefix=f"{name}_physical_phase",
  )


def _add_horizon_metrics(
  accumulator: _MetricAccumulator,
  name: str,
  prediction: np.ndarray,
  target: np.ndarray,
  unknown: np.ndarray,
  projection: ProjectionBundle,
) -> None:
  """Add future-horizon bucket metrics for one prediction or baseline."""
  physical_prediction = _physical_tokens(projection, prediction)
  physical_target = _physical_tokens(projection, target)
  for bucket, start, stop in _HORIZON_BUCKETS:
    selected = np.zeros_like(unknown)
    selected[:, start:stop] = unknown[:, start:stop]
    _add_error_metrics(
      accumulator,
      f"{name}_horizon_{bucket}",
      prediction,
      target,
      selected,
      physical_prediction=physical_prediction,
      physical_target=physical_target,
      physical_prefix=f"{name}_physical_horizon_{bucket}",
    )


def evaluate_generation(
  model: Any,
  *,
  schedule: DiffusionSchedule,
  grid: InferenceGrid,
  records: Sequence[WindowRecord],
  projection: ProjectionBundle,
  contract: DiffusionContract,
  seed: int,
  device: str | torch.device,
  max_windows: int | None = None,
  diagnostics_path: str | Path | None = None,
  split: str | None = None,
) -> GenerationReport:
  """Score conditioned DDIM generation against the specified baselines.

  The model and prior streams use separate CPU generators initialized with the
  same seed.  Consequently the prior-only baseline receives the same initial
  Gaussian draw that the model would see, while batching and device selection
  do not change the generated noise stream.  Only D0 unknown entries are
  included in any score.
  """
  if max_windows is not None and max_windows < 0:
    raise ValueError("max_windows must be non-negative")
  projection_contract = getattr(projection, "contract", None)
  if projection_contract is not None and hasattr(projection_contract, "identity_hash"):
    if projection_contract.identity_hash() != contract.identity_hash():
      raise ValueError("projection contract identity differs from evaluation contract")
  selected = tuple(records if max_windows is None else records[:max_windows])
  target_count = len(selected)
  destination = (
    Path(diagnostics_path) if diagnostics_path is not None else Path("<not-written>")
  )
  if target_count == 0:
    report = GenerationReport(
      split=split or "",
      windows=0,
      seed=int(seed),
      metrics={
        "model_unknown_state_mse": 0.0,
        "model_unknown_latent_mse": 0.0,
        "model_unknown_token_mse": 0.0,
        "gate_beats_prior_only": 0.0,
      },
      baselines={},
      beats_trivial_copy=False,
      beats_zero_latent=False,
      diagnostics_path=destination,
    )
    _write_generation_diagnostics(destination, report)
    return report

  selected_tokens = np.stack([_as_record_tokens(record) for record in selected])
  split_values = {str(getattr(record, "split", "")) for record in selected}
  split = split or (next(iter(split_values)) if len(split_values) == 1 else "mixed")
  torch_device = torch.device(device)
  if torch_device.type == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA device requested for generation but CUDA is unavailable")
  # Move the model here rather than relying on every caller: a CPU model
  # evaluated with ``device="cuda"`` fails on the first Linear with a device
  # mismatch.  Unlike a wrong metric this is loud, but it is easy to omit from
  # a new call path.  ``.to`` is a no-op when the model is already placed.
  if hasattr(model, "to"):
    model.to(torch_device)
  mask_cpu = clean_mask(
    current_index=contract.current_index,
    state_dimension=contract.projected_state_dimension,
    token_dimension=contract.token_dimension,
    device="cpu",
  )
  mask = mask_cpu.to(torch_device)
  sampler_generator = _cpu_generator(seed)
  prior_generator = _cpu_generator(seed)
  accumulator = _MetricAccumulator()
  sampler_config = SamplerConfig(grid=grid)
  model_was_training = bool(getattr(model, "training", False))
  if hasattr(model, "eval"):
    model.eval()

  try:
    with torch.no_grad():
      for start in range(0, target_count, 32):
        stop = min(target_count, start + 32)
        clean = torch.from_numpy(selected_tokens[start:stop]).to(
          device=torch_device, dtype=torch.float32
        )
        generated, _ = sample_trajectory(
          model,
          schedule=schedule,
          conditions=SamplerConditions(clean, mask),
          config=sampler_config,
          generator=sampler_generator,
        )
        prior_noise = torch.randn(
          clean.shape,
          generator=prior_generator,
          device="cpu",
          dtype=clean.dtype,
        ).to(torch_device)
        prior = apply_clean_mask(prior_noise, clean, mask)
        baselines = _baseline_tokens(clean)
        target = clean.detach().cpu().numpy().astype(np.float64)
        generated_np = generated.detach().cpu().numpy().astype(np.float64)
        prior_np = prior.detach().cpu().numpy().astype(np.float64)
        unknown = (~mask).detach().cpu().numpy().astype(bool)
        unknown = np.broadcast_to(unknown, target.shape).copy()
        predictions: dict[str, tuple[np.ndarray, str | None]] = {
          "model": (generated_np, None),
          "prior_only": (prior_np, None),
        }
        for name, value in baselines.items():
          scope = "state" if name == "copy_current_state" else "latent"
          predictions[name] = (value.detach().cpu().numpy().astype(np.float64), scope)

        for name, (prediction, scope) in predictions.items():
          physical_prediction = _physical_tokens(projection, prediction)
          physical_target = _physical_tokens(projection, target)
          selected_unknown = unknown.copy()
          if scope == "state":
            selected_unknown[..., _STATE_WIDTH:] = False
          elif scope == "latent":
            selected_unknown[..., :_STATE_WIDTH] = False
          _add_error_metrics(
            accumulator,
            f"{name}_normalized",
            prediction,
            target,
            selected_unknown,
            physical_prediction=physical_prediction,
            physical_target=physical_target,
            physical_prefix=f"{name}_physical",
          )
          _add_horizon_metrics(
            accumulator,
            f"{name}_normalized",
            prediction,
            target,
            selected_unknown,
            projection,
          )
          phase_values = [
            str(getattr(record, "phase", "") or "unknown")
            for record in selected[start:stop]
          ]
          for phase in sorted(set(phase_values)):
            phase_mask = np.asarray(phase_values) == phase
            _add_phase_metrics(
              accumulator,
              f"{name}_normalized",
              prediction,
              target,
              selected_unknown,
              phase_mask,
              projection,
            )

  finally:
    if hasattr(model, "train"):
      model.train(model_was_training)

  metrics = accumulator.values()
  aliases = {
    "model_unknown_state_mse": "model_normalized_unknown_state_mse",
    "model_unknown_latent_mse": "model_normalized_unknown_latent_mse",
    "model_unknown_token_mse": "model_normalized_unknown_token_mse",
  }
  for alias, source in aliases.items():
    if source in metrics:
      metrics[alias] = metrics[source]

  baseline_keys = {
    "copy_current_state": "copy_current_state_normalized_unknown_state_mse",
    "copy_current_latent": "copy_current_latent_normalized_unknown_latent_mse",
    "zero_latent": "zero_latent_normalized_unknown_latent_mse",
    "prior_only_state": "prior_only_normalized_unknown_state_mse",
    "prior_only_latent": "prior_only_normalized_unknown_latent_mse",
  }
  baselines = {
    name: metrics[key] for name, key in baseline_keys.items() if key in metrics
  }
  model_state = metrics.get("model_normalized_unknown_state_mse", float("inf"))
  model_latent = metrics.get("model_normalized_unknown_latent_mse", float("inf"))
  copy_state = baselines.get("copy_current_state", float("inf"))
  copy_latent = baselines.get("copy_current_latent", float("inf"))
  zero_latent = baselines.get("zero_latent", float("inf"))
  prior_state = baselines.get("prior_only_state", float("inf"))
  prior_latent = baselines.get("prior_only_latent", float("inf"))
  metrics.update(
    {
      "margin_model_vs_copy_current_state": copy_state - model_state,
      "margin_model_vs_copy_current_latent": copy_latent - model_latent,
      "margin_model_vs_zero_latent": zero_latent - model_latent,
      "margin_model_vs_prior_only_state": prior_state - model_state,
      "margin_model_vs_prior_only_latent": prior_latent - model_latent,
    }
  )
  beats_trivial_copy = model_state < copy_state and model_latent < copy_latent
  beats_zero_latent = model_latent < zero_latent
  beats_prior = model_state < prior_state and model_latent < prior_latent
  metrics["gate_beats_prior_only"] = float(beats_prior)
  metrics["gate_beats_copy_current_state"] = float(model_state < copy_state)
  metrics["gate_beats_copy_current_latent"] = float(model_latent < copy_latent)
  metrics["gate_beats_zero_latent"] = float(beats_zero_latent)
  report = GenerationReport(
    split=split,
    windows=target_count,
    seed=int(seed),
    metrics=metrics,
    baselines=baselines,
    beats_trivial_copy=beats_trivial_copy,
    beats_zero_latent=beats_zero_latent,
    diagnostics_path=destination,
  )
  _write_generation_diagnostics(destination, report)
  return report


def _write_generation_diagnostics(path: Path, report: GenerationReport) -> None:
  """Write a report only when the caller supplied an output path."""
  if str(path) == "<not-written>":
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n")


def _audit_source(
  source: DatasetSource | LoadedDataset,
) -> tuple[LoadedDataset, DatasetSource, Path]:
  if isinstance(source, DatasetSource):
    return source.load(), source, source.directory
  if isinstance(source, LoadedDataset):
    directory = Path(source.store.root).parent
    # The source path is only a location holder for the identity helper; the
    # already-loaded contract remains authoritative for validation.
    source_obj = DatasetSource(directory, Path("<loaded-contract>"))
    return source, source_obj, directory
  raise TypeError("source must be DatasetSource or LoadedDataset")


def _validate_audit_cache(
  cache: Path,
  metadata_path: Path,
  loaded: LoadedDataset,
  source: DatasetSource,
) -> dict[str, object]:
  """Validate cache bytes and counts derived from D1's streaming selection."""
  metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
  if not isinstance(metadata, Mapping):
    raise ValueError("cache metadata must be an object")
  split = metadata.get("split")
  if not isinstance(split, str):
    raise ValueError("cache split metadata is invalid")
  raw_motions = metadata.get("motion_ids")
  if raw_motions is None:
    motion_ids = None
  elif isinstance(raw_motions, list):
    motion_ids = tuple(sorted({str(value) for value in raw_motions}))
  else:
    raise ValueError("cache motion_ids metadata is invalid")
  raw_limit = metadata.get("limit")
  if raw_limit is not None and (
    isinstance(raw_limit, bool) or not isinstance(raw_limit, (int, float, str))
  ):
    raise ValueError("cache limit metadata is invalid")
  limit = None if raw_limit is None else int(raw_limit)
  raw_shape = metadata.get("shape")
  if not isinstance(raw_shape, list) or len(raw_shape) != 3:
    raise ValueError("cache shape metadata is invalid")
  shape = tuple(int(value) for value in raw_shape)
  raw_count = metadata.get("window_count")
  if isinstance(raw_count, bool) or not isinstance(raw_count, (int, float, str)):
    raise ValueError("cache window_count metadata is invalid")
  metadata_count = int(raw_count)
  derived_count = sum(
    1
    for _ in iter_window_provenance(
      loaded,
      split,
      motion_ids=motion_ids,
      limit=limit,
    )
  )
  derived_shape = (
    derived_count,
    loaded.contract.window_steps,
    loaded.contract.token_dimension,
  )
  if metadata_count != derived_count:
    raise WindowCacheError(
      "token cache window_count disagrees with D1 selection: "
      f"metadata={metadata_count}, derived={derived_count}"
    )
  if shape != derived_shape:
    raise WindowCacheError(
      "token cache shape disagrees with D1 selection: "
      f"metadata={shape}, derived={derived_shape}"
    )
  dtype = np.dtype(str(metadata.get("dtype", "")))
  expected = _cache_metadata(
    source_hash=_source_dataset_hash(source, loaded),
    split=split,
    contract_hash=loaded.contract.identity_hash(),
    projection_hash=_projection_identity(loaded.projection),
    window_count=derived_count,
    dtype=dtype,
    motion_ids=motion_ids,
    limit=limit,
    shape=derived_shape,
    created_at=None,
    content_sha256=metadata.get("content_sha256")
    if isinstance(metadata.get("content_sha256"), str)
    else None,
  )
  _validate_cache(cache, metadata_path, expected)
  return {
    "valid": True,
    "window_count": metadata.get("window_count"),
    "dtype": metadata.get("dtype"),
    "shape": metadata.get("shape"),
  }


def audit_dataset(source: DatasetSource | LoadedDataset) -> dict[str, object]:
  """Audit pilot/bulk D1 identity, coverage, token slices and phase provenance."""
  loaded, source_obj, directory = _audit_source(source)
  coverage = loaded.index.coverage()
  moments = _Moments(_TOKEN_WIDTH)
  motion_phase: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}
  motion_phase_total: dict[str, dict[str, dict[str, float | int]]] = {}
  noise_norms: list[float] = []
  finite_windows = 0
  split_counts: dict[str, int] = {}
  for split in ("train", "validation", "test"):
    records = load_window_records(loaded, split)
    split_counts[split] = len(records)
    for record in records:
      tokens = _as_record_tokens(record)
      moments.update(tokens.reshape(-1, _TOKEN_WIDTH))
      finite_windows += int(np.isfinite(tokens).all())
      motion = str(record.motion_id)
      phase = str(record.phase or "unknown")
      split_entry = (
        motion_phase.setdefault(split, {})
        .setdefault(motion, {})
        .setdefault(phase, {"windows": 0, "ou_noise_norm_sum": 0.0})
      )
      split_entry["windows"] = int(split_entry["windows"]) + 1
      split_entry["ou_noise_norm_sum"] = float(
        split_entry["ou_noise_norm_sum"]
      ) + float(record.ou_noise_norm)
      total_entry = motion_phase_total.setdefault(motion, {}).setdefault(
        phase, {"windows": 0, "ou_noise_norm_sum": 0.0}
      )
      total_entry["windows"] = int(total_entry["windows"]) + 1
      total_entry["ou_noise_norm_sum"] = float(
        total_entry["ou_noise_norm_sum"]
      ) + float(record.ou_noise_norm)
      noise_norms.append(float(record.ou_noise_norm))

  for motions in motion_phase.values():
    for phases in motions.values():
      for entry in phases.values():
        count = int(entry["windows"])
        total = float(entry.pop("ou_noise_norm_sum"))
        entry["mean_ou_noise_norm"] = total / count if count else 0.0
  for phases in motion_phase_total.values():
    for entry in phases.values():
      count = int(entry["windows"])
      total = float(entry.pop("ou_noise_norm_sum"))
      entry["mean_ou_noise_norm"] = total / count if count else 0.0

  cache_sizes: list[dict[str, object]] = []
  cache_valid = True
  offline = directory / "offline"
  for cache in sorted(offline.glob("tokens-*.npy")):
    metadata_path = cache.with_suffix(".json")
    item: dict[str, object] = {
      "path": str(cache),
      "bytes": cache.stat().st_size,
      "metadata": str(metadata_path),
    }
    try:
      item.update(_validate_audit_cache(cache, metadata_path, loaded, source_obj))
    except (KeyError, OSError, TypeError, ValueError, WindowCacheError) as exc:
      item["valid"] = False
      item["metadata_error"] = str(exc)
      cache_valid = False
    cache_sizes.append(item)

  noise_array = np.asarray(noise_norms, dtype=np.float64)
  if noise_array.size:
    noise_summary: dict[str, object] = {
      "windows": int(noise_array.size),
      "finite": bool(np.isfinite(noise_array).all()),
      "nonzero": int(np.count_nonzero(noise_array > 0.0)),
      "min": float(noise_array.min()),
      "max": float(noise_array.max()),
      "mean": float(noise_array.mean()),
    }
  else:
    noise_summary = {
      "windows": 0,
      "finite": True,
      "nonzero": 0,
      "min": 0.0,
      "max": 0.0,
      "mean": 0.0,
    }
  payload: dict[str, object] = {
    "ok": (
      loaded.store.contract.identity_hash() == loaded.contract.identity_hash()
      and loaded.projection.contract.identity_hash() == loaded.contract.identity_hash()
      and split_counts == coverage
      and finite_windows == sum(split_counts.values())
      and cache_valid
    ),
    "dataset": str(directory),
    "contract_identity": loaded.contract.identity_hash(),
    "store_contract_identity": loaded.store.contract.identity_hash(),
    "projection_contract_identity": loaded.projection.contract.identity_hash(),
    "assignments_hash": loaded.assignments.sha256(),
    "rows": loaded.store.row_count,
    "shards": loaded.store.shard_count,
    "coverage": coverage,
    "motion_phase": motion_phase,
    "motion_phase_total": motion_phase_total,
    "cache_sizes": cache_sizes,
    "token_statistics": {
      "windows": sum(split_counts.values()),
      "tokens": moments.count,
      "projected_rows": moments.summary(0, 64),
      "identity_rows": moments.summary(64, 199),
      "latent": moments.summary(199, 231),
    },
    "noise_level_sanity": {
      "forward_diffusion_levels": "trainer samples independent integers in [0, 1000]",
      "dataset_levels_persisted": False,
      "ou_noise_norm": noise_summary,
    },
  }
  return payload


__all__ = ["GenerationReport", "audit_dataset", "evaluate_generation"]
