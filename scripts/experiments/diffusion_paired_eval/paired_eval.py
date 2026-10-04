"""Paired offline evaluation of two diffusion runs on identical windows.

`diffusion evaluate-offline` reports aggregates over whatever validation windows a
checkpoint's own dataset contains.  Two arms built from different collections do
not share that window set exactly, so their aggregates are not directly
comparable.  This tool closes that gap: it loads each arm, intersects the window
identities (motion, group, phase, pair, start tick), and scores every arm on the
*same* records in a deterministic order, once overall and once per
motion x phase group so a spread can be reported instead of a single number.

Read-only: it writes one JSON report and never touches a run directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from mjlab.tasks.tracking.diffusion.checkpoint import load_checkpoint
from mjlab.tasks.tracking.diffusion.contract import DiffusionContract
from mjlab.tasks.tracking.diffusion.evaluation import evaluate_generation
from mjlab.tasks.tracking.diffusion.model import (
  DenoiserSettings,
  StateLatentTransformer,
)
from mjlab.tasks.tracking.diffusion.schedule import (
  DiffusionSchedule,
  build_inference_grid,
)
from mjlab.tasks.tracking.diffusion.trainer import ExponentialMovingAverage
from mjlab.tasks.tracking.diffusion.window_dataset import (
  DatasetSource,
  _source_dataset_hash,
  load_window_records,
)

SCORED = (
  ("model_normalized_unknown_latent_mse", "normalized latent"),
  ("model_normalized_unknown_state_mse", "normalized state"),
  ("model_normalized_unknown_token_mse", "normalized token"),
)
BASELINES = (
  "copy_current_latent_normalized_phase_unknown_latent_mse",
  "copy_current_state_normalized_phase_unknown_state_mse",
  "copy_current_normalized_phase_unknown_token_mse",
  "prior_only_latent_normalized_phase_unknown_latent_mse",
  "prior_only_state_normalized_phase_unknown_state_mse",
  "zero_latent_normalized_phase_unknown_latent_mse",
)


def _window_identity(record: Any) -> tuple[str, str, str, str, int]:
  return (
    str(record.motion_id),
    str(record.group_key),
    str(record.phase),
    str(record.pair_id),
    int(record.start_tick),
  )


def load_arm(
  name: str,
  run_dir: Path,
  contract_path: Path,
  checkpoint_name: str,
  split: str,
  device,
) -> dict[str, Any]:
  """Load one arm exactly as `evaluate-offline` does, including its checks."""
  contract = DiffusionContract.from_yaml(contract_path)
  schedule = DiffusionSchedule.from_contract(contract)
  settings = DenoiserSettings.from_contract(contract, training_k=schedule.training_k)
  model = StateLatentTransformer(settings)
  ema = ExponentialMovingAverage(model)
  checkpoint_path = run_dir / checkpoint_name
  state = load_checkpoint(
    checkpoint_path,
    model=model,
    ema=ema,
    map_location=device,
    expected_contract_identity=contract.identity_hash(),
  )
  dataset_path = state.dataset_identity.get("directory")
  if not isinstance(dataset_path, str) or not dataset_path:
    raise ValueError(f"{name}: checkpoint records no dataset directory")
  source = DatasetSource.resolve(dataset_path, contract_path=contract_path)
  loaded = source.load()
  expected = state.dataset_identity
  actual_projection = {
    "matrix": loaded.projection.matrix_sha256,
    "pseudoinverse": loaded.projection.pseudoinverse_sha256,
    "statistics": loaded.projection.statistics_sha256,
  }
  checks = {
    "contract_hash": (expected.get("contract_hash"), contract.identity_hash()),
    "assignments_hash": (expected.get("assignments_hash"), loaded.assignments.sha256()),
    "split_coverage": (expected.get("split_coverage"), loaded.index.coverage()),
    "projection_hashes": (expected.get("projection_hashes"), actual_projection),
    "source_dataset_hash": (
      expected.get("source_dataset_hash"),
      _source_dataset_hash(source, loaded),
    ),
  }
  mismatched = [key for key, (want, got) in checks.items() if want != got]
  if mismatched:
    raise ValueError(f"{name}: checkpoint dataset identity mismatch in {mismatched}")
  if state.ema_state is not None:
    ema.copy_to(model)
  model.to(device)
  model.eval()
  records = load_window_records(loaded, split)
  return {
    "name": name,
    "contract": contract,
    "schedule": schedule,
    "model": model,
    "loaded": loaded,
    "state": state,
    "records": records,
    "checkpoint": str(checkpoint_path),
    "dataset": str(source.directory),
    "selection_split": state.selection_split,
    "best_validation_loss": state.best_validation_loss,
    "global_step": state.global_step,
  }


def score(
  arm: dict[str, Any], records: list[Any], seed: int, device, split: str
) -> dict[str, Any]:
  report = evaluate_generation(
    arm["model"],
    schedule=arm["schedule"],
    grid=build_inference_grid(arm["schedule"], contract=arm["contract"]),
    records=records,
    projection=arm["loaded"].projection,
    contract=arm["contract"],
    seed=seed,
    device=device,
    split=split,
  )
  return report.as_dict()


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--arm",
    action="append",
    required=True,
    metavar="NAME=RUN_DIR:CONTRACT[:CHECKPOINT]",
    help="repeat once per arm; the first arm is the reference",
  )
  parser.add_argument("--split", default="validation")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--checkpoint", default="checkpoint-best.pt")
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--max-windows", type=int, default=None)
  args = parser.parse_args()

  device = torch.device(args.device)
  specs = []
  for raw in args.arm:
    name, rest = raw.split("=", 1)
    parts = rest.split(":")
    run_dir = Path(parts[0])
    contract_path = Path(parts[1])
    checkpoint = parts[2] if len(parts) > 2 else args.checkpoint
    specs.append((name, run_dir, contract_path, checkpoint))

  arms = [
    load_arm(name, run_dir, contract, ckpt, args.split, device)
    for name, run_dir, contract, ckpt in specs
  ]

  # Identical window set for every arm: intersect on stable provenance.
  maps = [{_window_identity(r): r for r in arm["records"]} for arm in arms]
  shared = sorted(set.intersection(*(set(m) for m in maps)))
  if args.max_windows is not None:
    shared = shared[: args.max_windows]
  if not shared:
    raise SystemExit("no window identities are shared by the arms")

  per_arm_records = [[m[key] for key in shared] for m in maps]
  groups: dict[tuple[str, str], list[int]] = {}
  for index, key in enumerate(shared):
    groups.setdefault((key[0], key[2]), []).append(index)

  overall: dict[str, Any] = {}
  by_group: dict[str, Any] = {}
  for arm, records in zip(arms, per_arm_records, strict=True):
    overall[arm["name"]] = score(arm, records, args.seed, device, args.split)
    for (motion, phase), indexes in sorted(groups.items()):
      subset = [records[i] for i in indexes]
      by_group.setdefault(f"{motion}|{phase}", {})[arm["name"]] = score(
        arm, subset, args.seed, device, args.split
      )

  reference = arms[0]["name"]
  summary: dict[str, Any] = {
    "split": args.split,
    "seed": args.seed,
    "checkpoint": args.checkpoint,
    "reference_arm": reference,
    "window_identities": {
      "per_arm": {arm["name"]: len(arm["records"]) for arm in arms},
      "shared": len(shared),
      "groups": {f"{m}|{p}": len(v) for (m, p), v in sorted(groups.items())},
    },
    "arms": {
      arm["name"]: {
        "run_dir": str(Path(arm["checkpoint"]).parent),
        "checkpoint": arm["checkpoint"],
        "dataset": arm["dataset"],
        "global_step": arm["global_step"],
        "selection_split": arm["selection_split"],
        "best_validation_loss": arm["best_validation_loss"],
        "windows": overall[arm["name"]]["windows"],
        "metrics": {
          label: overall[arm["name"]]["metrics"].get(key) for key, label in SCORED
        },
        "baselines": {
          key: overall[arm["name"]]["metrics"].get(key) for key in BASELINES
        },
      }
      for arm in arms
    },
    "paired": {},
    "per_group": {},
  }

  for name in [arm["name"] for arm in arms[1:]]:
    entry: dict[str, Any] = {}
    for key, label in SCORED:
      base = overall[reference]["metrics"].get(key)
      cand = overall[name]["metrics"].get(key)
      entry[label] = {
        "reference": base,
        "candidate": cand,
        "delta": None if base is None or cand is None else cand - base,
        "ratio": None if not base or cand is None else cand / base,
      }
    deltas = []
    for _group, per_arm in by_group.items():
      base = per_arm[reference]["metrics"].get(SCORED[0][0])
      cand = per_arm[name]["metrics"].get(SCORED[0][0])
      if base is None or cand is None:
        continue
      deltas.append(cand - base)
    entry["latent_delta_by_group"] = deltas
    entry["latent_delta_groups_better"] = sum(1 for d in deltas if d < 0)
    entry["latent_delta_groups_total"] = len(deltas)
    summary["paired"][f"{reference} -> {name}"] = entry

  for group, per_arm in sorted(by_group.items()):
    summary["per_group"][group] = {
      name: {label: per_arm[name]["metrics"].get(key) for key, label in SCORED}
      for name in per_arm
    }

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

  print(f"shared windows: {len(shared)}")
  for arm in arms:
    metrics = summary["arms"][arm["name"]]
    print(
      f"  {arm['name']:22s} latent={metrics['metrics']['normalized latent']:.6f} "
      f"state={metrics['metrics']['normalized state']:.6f} "
      f"token={metrics['metrics']['normalized token']:.6f}"
    )
  for pair, entry in summary["paired"].items():
    latent = entry["normalized latent"]
    print(
      f"  {pair}: latent delta {latent['delta']:+.6f} "
      f"(ratio {latent['ratio']:.4f}); better in "
      f"{entry['latent_delta_groups_better']}/{entry['latent_delta_groups_total']} groups"
    )
  print(f"report: {args.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
