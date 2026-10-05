"""Measure how much of the future window is predictable from its clean prefix.

Motivation: the D2 offline gates pass while the D3 closed loop falls, and the
generation metrics sit at or near the mean predictor.  Before spending more
GPU-hours it matters whether that is an optimization failure or the ceiling of
what the conditioning contains.

This measures the ceiling directly from the stored dataset, with no model:
for each validation window it finds windows from *other trials* whose clean
prefix is nearly identical, and reports the residual spread of their futures.
Because the evaluation convention is that ``1.0`` equals the unconditional
future variance (a mean predictor scores 1.0), the same number is directly
comparable with ``model_normalized_unknown_*_mse`` from the generation reports.

Same-trial neighbours are excluded on purpose: the dataset slides a 41-step
window over each trajectory, so adjacent windows share 40 of 41 timesteps and
would make the future look trivially predictable.

Read-only with respect to every run and dataset directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SPLITS = ("train", "validation", "test")

# Set by load_dataset so the alignment check can materialise the same windows.
CACHE_ALIGNMENT_SOURCE = None


def load_dataset(
  dataset_dir: Path, contract_path: Path, cache_dir: Path
) -> tuple[dict[str, np.ndarray], dict[str, list]]:
  """Load cached windows plus their provenance, verifying row alignment."""
  from mjlab.tasks.tracking.diffusion.window_dataset import (  # noqa: PLC0415
    DatasetSource,
    load_window_records,
  )

  source = DatasetSource.resolve(str(dataset_dir), contract_path=contract_path)
  loaded = source.load()
  global CACHE_ALIGNMENT_SOURCE  # noqa: PLW0603
  CACHE_ALIGNMENT_SOURCE = loaded
  tokens: dict[str, np.ndarray] = {}
  records: dict[str, list] = {}
  for split in SPLITS:
    cache_path = cache_dir / f"tokens-{split}-all.npy"
    if not cache_path.exists():
      raise SystemExit(f"missing token cache {cache_path}")
    array = np.load(cache_path)
    split_records = load_window_records(loaded, split)
    if array.shape[0] != len(split_records):
      raise SystemExit(
        f"{split}: cache has {array.shape[0]} rows but {len(split_records)} records"
      )
    tokens[split] = array.astype(np.float32, copy=False)
    records[split] = split_records
    print(
      f"  {split}: {array.shape} windows, aligned with {len(split_records)} records"
    )
  return tokens, records


def split_conditioning(
  array: np.ndarray, *, current_index: int, state_dimension: int
) -> tuple[np.ndarray, np.ndarray]:
  """Return (clean prefix, unknown future) for a token array."""
  prefix = np.concatenate(
    (
      array[:, :current_index, :].reshape(array.shape[0], -1),
      array[:, current_index, :state_dimension],
    ),
    axis=1,
  )
  future = array[:, current_index + 1 :, :]
  return prefix, future


def verify_cache_alignment(
  loaded: object, tokens: np.ndarray, split: str, *, count: int = 12
) -> str:
  """Verify that cache row i is the dataset's own window i.

  Everything below keys neighbours by row index, so a permutation between the
  cache and the window records would invalidate the result. This compares rows
  against the freshly materialised windows instead of assuming an order.

  Note the dataset does *not* consist of sliding windows: the windows of one
  (motion, pair family, phase) group are independent rollouts of the same trial
  and share no timesteps (checked by best-overlap-over-shift, min disagreement
  0.155, i.e. no shift relation exists at any lag). Excluding same-family
  windows is therefore conservative rather than load-bearing, and both the
  excluded and unexcluded numbers are reported.
  """
  view = loaded.dataset(split)  # type: ignore[attr-defined]
  total = int(tokens.shape[0])
  step = max(1, total // max(1, count))
  worst = 0.0
  checked = 0
  for index in range(0, total, step):
    if checked >= count:
      break
    sample = view[index]
    difference = float(np.max(np.abs(np.asarray(sample.tokens) - tokens[index])))
    worst = max(worst, difference)
    checked += 1
  return (
    f"{checked} rows compared against their own window, max disagreement {worst:.3e}"
  )


def neighbourhood_report(
  prefix: np.ndarray,
  future: np.ndarray,
  trial: np.ndarray,
  phase: np.ndarray,
  query_index: np.ndarray,
  *,
  neighbours: int,
  blocks: dict[str, slice],
  exclude_same_trial: bool = True,
) -> dict[str, object]:
  """Residual future spread among near-identical prefixes from other trials."""
  # Coerce once: integer-array indexing needs real arrays, not lists.
  prefix = np.asarray(prefix, dtype=np.float64)
  future = np.asarray(future, dtype=np.float64)
  trial = np.asarray(trial)
  phase = np.asarray(phase)
  query_index = np.asarray(query_index)
  mean = prefix.mean(axis=0, keepdims=True)
  scale = prefix.std(axis=0, keepdims=True)
  scale[scale < 1e-8] = 1.0
  normalised = (prefix - mean) / scale

  report: dict[str, object] = {
    "queries": int(len(query_index)),
    "exclude_same_trial": bool(exclude_same_trial),
  }
  for name, block in blocks.items():
    # ``future`` is (windows, steps, channels): the block slices CHANNELS, so the
    # last axis must be addressed explicitly.  ``future[:, block]`` would slice
    # steps instead, silently returning the whole token for a wide block and an
    # empty array for a narrow one.
    values = future[..., block]
    total = float(values.var(axis=0).mean())
    report[f"total_variance_{name}"] = total

  per_block_residual: dict[str, list[np.ndarray]] = {name: [] for name in blocks}
  per_block_distance: list[np.ndarray] = []
  cross_phase_residual: dict[str, list[np.ndarray]] = {name: [] for name in blocks}
  block_slices = list(blocks.values())
  skipped = 0
  picked_same_trial = 0
  picked_total = 0

  chunk = 64
  # Squared norms let the distance be computed as a Gram matrix; broadcasting
  # the difference would materialise a (chunk, n_windows, prefix_dim) array.
  all_norm = np.einsum("ij,ij->i", normalised, normalised)
  for start in range(0, len(query_index), chunk):
    selection = query_index[start : start + chunk]
    query = normalised[selection]
    squared = (
      all_norm[None, :]
      + np.einsum("ij,ij->i", query, query)[:, None]
      - 2.0 * (query @ normalised.T)
    )
    np.maximum(squared, 0.0, out=squared)
    distances = np.sqrt(squared)
    if exclude_same_trial:
      # Never compare a window with a window of its own trial: the dataset slides
      # a 41-step window along each trajectory, so same-trial neighbours share up
      # to 40 of 41 timesteps and would make the future look trivially predictable.
      same_trial = trial[selection, None] == trial[None, :]
      distances[same_trial] = np.inf
    order = np.argsort(distances, axis=1)[:, :neighbours]
    rows = np.arange(len(selection))[:, None]
    chosen_distance = distances[rows, order]
    for position, index in enumerate(selection):
      picked = order[position]
      usable = np.isfinite(distances[position, picked])
      if int(usable.sum()) < 2:
        skipped += 1
        continue
      picked = picked[usable]
      picked_same_trial += int((trial[picked] == trial[index]).sum())
      picked_total += int(len(picked))
      # Unbiased variance needs the Bessel factor; without it the residual is
      # inflated by (1 - 1/k) and a mean predictor would not score exactly 1.
      correction = len(picked) / (len(picked) - 1)
      for name, slot in zip(blocks, block_slices, strict=True):
        block_values = future[..., slot][picked]
        residual = block_values - block_values.mean(axis=0, keepdims=True)
        per_block_residual[name].append(correction * (residual**2).mean(axis=0))
        same = phase[picked] == phase[index]
        if int(same.sum()) >= 2:
          same_values = future[..., slot][picked[same]]
          same_residual = same_values - same_values.mean(axis=0, keepdims=True)
          factor = int(same.sum()) / (int(same.sum()) - 1)
          cross_phase_residual[name].append(factor * (same_residual**2).mean(axis=0))
      per_block_distance.append(chosen_distance[position][usable])

  report["queries_skipped_for_few_neighbours"] = skipped
  # Directly checkable invariant: with the exclusion on this must be exactly 0.0,
  # which is what proves same-trial (sliding-window) neighbours were removed.
  report["picked_same_trial_fraction"] = (
    float(picked_same_trial / picked_total) if picked_total else float("nan")
  )
  if not per_block_distance:
    report["neighbour_distance_median"] = float("nan")
    report["neighbour_distance_p90"] = float("nan")
    for name in blocks:
      report[f"residual_variance_{name}"] = float("nan")
      report[f"ratio_{name}"] = float("nan")
      report[f"ratio_tight_{name}"] = float("nan")
    return report
  all_distances = np.concatenate(per_block_distance)
  report["neighbour_distance_median"] = float(np.median(all_distances))
  report["neighbour_distance_p90"] = float(np.percentile(all_distances, 90))
  for name in blocks:
    residual = np.stack(per_block_residual[name])
    distances = np.stack(per_block_distance)
    furthest = distances[:, -1]
    tight = furthest <= np.percentile(furthest, 50)
    report[f"residual_variance_{name}"] = float(residual.mean())
    report[f"ratio_{name}"] = float(residual.mean() / report[f"total_variance_{name}"])
    report[f"ratio_tight_{name}"] = (
      float(residual[tight].mean() / report[f"total_variance_{name}"])
      if tight.any()
      else float("nan")
    )
    if cross_phase_residual[name]:
      same_phase = np.stack(cross_phase_residual[name])
      report[f"ratio_same_phase_{name}"] = float(
        same_phase.mean() / report[f"total_variance_{name}"]
      )
  return report


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dataset-dir", type=Path, required=True)
  parser.add_argument("--contract", type=Path, required=True)
  parser.add_argument("--cache-dir", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--neighbours", type=int, default=16)
  parser.add_argument("--max-queries", type=int, default=1500)
  args = parser.parse_args()

  tokens, records = load_dataset(args.dataset_dir, args.contract, args.cache_dir)
  arrays = [tokens[split] for split in SPLITS]
  stacked = np.concatenate(arrays, axis=0)
  phase = np.concatenate([np.array([str(r.phase) for r in records[s]]) for s in SPLITS])
  # Episode identity: one (motion, pair family, phase) group is a set of
  # *independent* rollouts of the same trial, so this key is a conservative
  # exclusion unit rather than a way to remove overlapping samples.
  trial = np.concatenate(
    [
      np.array([f"{r.motion_id}|{r.pair_id}|{r.phase}" for r in records[s]])
      for s in SPLITS
    ]
  )
  split_of = np.concatenate([np.full(len(records[s]), s, dtype=object) for s in SPLITS])

  from mjlab.tasks.tracking.diffusion.contract import (  # noqa: PLC0415
    DiffusionContract,
  )

  contract = DiffusionContract.from_yaml(args.contract)
  current_index = int(contract.current_index)
  state_dimension = int(contract.projected_state_dimension)
  prefix, future = split_conditioning(
    stacked, current_index=current_index, state_dimension=state_dimension
  )
  blocks = {
    "state": slice(0, state_dimension),
    "latent": slice(state_dimension, stacked.shape[2]),
  }

  alignment = "not checked"
  for split in SPLITS:
    if CACHE_ALIGNMENT_SOURCE is not None:
      alignment = verify_cache_alignment(
        loaded=CACHE_ALIGNMENT_SOURCE, tokens=tokens[split], split=split, count=8
      )
      break
  print("cache alignment check:", alignment)

  validation_index = np.flatnonzero(split_of == "validation")
  if len(validation_index) > args.max_queries:
    step = len(validation_index) // args.max_queries
    validation_index = validation_index[::step]

  report = {
    "windows_total": int(stacked.shape[0]),
    "windows_validation_queried": int(len(validation_index)),
    "current_index": current_index,
    "state_dimension": state_dimension,
    "neighbours": args.neighbours,
    "alignment_check": alignment,
  }
  for label, mask in (
    ("all_phases", None),
    ("clean_only", phase == "clean"),
  ):
    selected = validation_index if mask is None else np.flatnonzero(mask)
    if mask is not None:
      selected = np.array([i for i in validation_index if mask[i]])
      if len(selected) == 0:
        continue
    print(f"--- analysis over {label}: {len(selected)} queries")
    for excluded, suffix in ((True, "cross_trial"), (False, "any_trial")):
      report[f"{label}_{suffix}"] = neighbourhood_report(
        prefix,
        future,
        trial,
        phase,
        selected,
        neighbours=args.neighbours,
        blocks=blocks,
        exclude_same_trial=excluded,
      )
      summary = report[f"{label}_{suffix}"]
      print(
        f"  [{label} / {suffix}] skip={summary['queries_skipped_for_few_neighbours']}"
      )
      for name in blocks:
        print(
          "    %-7s residual/total ratio=%.4f (tight %.4f) same-phase %.4f  "
          "median neighbour distance=%.3f"
          % (
            name,
            summary[f"ratio_{name}"],
            summary[f"ratio_tight_{name}"],
            summary.get(f"ratio_same_phase_{name}", float("nan")),
            summary["neighbour_distance_median"],
          )
        )

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"report written to {args.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
