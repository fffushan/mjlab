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
import re
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


def motion_key(motion_id: str) -> str:
  """Canonical motion identity, independent of the collection's naming scheme.

  The same physical clip appears as `tennis_000` in the published collection and
  as `inc000` in the revision-2 collections (the `incXXX` suffix is defined to be
  the tennis motion id), and one clip carries a `_ss` suffix, so the *first*
  digit group is the stable identity rather than a trailing one.
  """
  match = re.search(r"(\d+)", str(motion_id))
  return match.group(1) if match else str(motion_id)


def trial_key(record: Any) -> str:
  """Canonical trial id: `<motion>:<seed>:<start_frame>` with the motion renamed.

  `pair_id`/`group_key` already encode the protocol coordinates, so replacing the
  collection-local motion name keeps them comparable across collections.
  """
  raw = str(getattr(record, "pair_id", "") or getattr(record, "group_key", ""))
  parts = raw.split(":")
  if len(parts) == 3:
    return ":".join([motion_key(parts[0]), parts[1], parts[2]])
  return raw


def _window_identity(record: Any) -> tuple[str, str, str, int]:
  """Semantic identity of the underlying motion sample.

  A window is only identified by its protocol coordinates: the motion, the
  perturbation phase, the trial (seed and start frame), and the tick.  Without
  the trial the key is many-to-one (85 ticks per trial and many trials per
  motion/phase), and the raw motion name and pair id are dataset-local strings.
  """
  return (
    motion_key(record.motion_id),
    str(record.phase),
    trial_key(record),
    int(record.start_tick),
  )


def _unique_identities(records: list[Any], name: str) -> tuple[dict, list]:
  """Map identity -> record, excluding identities that are not one-to-one."""
  buckets: dict[tuple[str, str, str, int], list[Any]] = {}
  for record in records:
    buckets.setdefault(_window_identity(record), []).append(record)
  unique = {key: rows[0] for key, rows in buckets.items() if len(rows) == 1}
  ambiguous = sorted(key for key, rows in buckets.items() if len(rows) > 1)
  if ambiguous:
    print(
      f"  note: {name} has {len(ambiguous)} ambiguous identities (excluded), "
      f"e.g. {ambiguous[0]}"
    )
  return unique, ambiguous


def estimated_reference_frame(record: Any) -> int:
  """Estimate the motion frame a window covers: pair-family start + tick.

  `pair_id` encodes `motion:seed:start_frame` (see `build_bulk_trials`), so the
  rollout's start frame is its last integer group; a window at `start_tick` then
  covers approximately `start_frame + start_tick` of the reference clip.  This is
  an estimate used only to bucket windows by region, never an identity.
  """
  numbers = re.findall(r"(\d+)", str(record.pair_id))
  start_frame = int(numbers[-1]) if numbers else -1
  return start_frame + int(record.start_tick)


def in_frame_range(record: Any, frame_range: str) -> bool:
  """True when the estimated reference frame falls inside `LO:HI` (inclusive)."""
  low, high = (int(part) for part in frame_range.split(":"))
  frame = estimated_reference_frame(record)
  return low <= frame <= high


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
  parser.add_argument(
    "--dry-run-identities",
    action="store_true",
    help="check the shared window set and exit before scoring anything",
  )
  parser.add_argument(
    "--diagnose-identities",
    action="store_true",
    help="print per-arm window identity inventories and exit",
  )
  parser.add_argument("--max-windows", type=int, default=None)
  parser.add_argument(
    "--only-motion",
    default=None,
    help="keep only this canonical motion group (e.g. 001)",
  )
  parser.add_argument(
    "--frame-range",
    default=None,
    metavar="LO:HI",
    help="keep only windows whose estimated reference frame is in LO..HI",
  )
  parser.add_argument(
    "--exclude-frame-range",
    default=None,
    metavar="LO:HI",
    help="drop windows whose estimated reference frame is in LO..HI",
  )
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

  # Identical window set for every arm: intersect on semantic identity.
  maps = []
  for arm in arms:
    unique, ambiguous = _unique_identities(arm["records"], arm["name"])
    maps.append(unique)
    arm["ambiguous_identities"] = len(ambiguous)
  shared = sorted(set.intersection(*(set(m) for m in maps)))

  if args.diagnose_identities:
    for arm in arms:
      rows = arm["records"]
      pairs: dict[tuple[str, str], list[int]] = {}
      for record in rows:
        pairs.setdefault((str(record.motion_id), str(record.phase)), []).append(
          int(record.start_tick)
        )
      print(f"--- {arm['name']}: {len(rows)} windows, datasets={arm['dataset']}")
      print(f"    distinct motion_ids: {sorted({str(r.motion_id) for r in rows})}")
      print(f"    distinct phases: {sorted({str(r.phase) for r in rows})}")
      for key, ticks in sorted(pairs.items()):
        print(
          "    %-24s n=%-5d ticks min=%d max=%d distinct=%d first=%s"
          % (
            ":".join(key),
            len(ticks),
            min(ticks),
            max(ticks),
            len(set(ticks)),
            sorted(set(ticks))[:6],
          )
        )
    print("--- candidate key overlaps (first arm vs last arm)")
    first, last = arms[0]["records"], arms[-1]["records"]

    def keyset(rows, fields):
      out = set()
      for record in rows:
        values = {
          "motion": str(record.motion_id),
          "canon": motion_key(record.motion_id),
          "phase": str(record.phase),
          "tick": int(record.start_tick),
          "pair": str(record.pair_id),
          "trial": trial_key(record),
          "group": str(record.group_key),
        }
        out.add(tuple(values[f] for f in fields))
      return out

    for fields in (
      ("motion",),
      ("canon",),
      ("phase",),
      ("tick",),
      ("canon", "phase"),
      ("canon", "tick"),
      ("phase", "tick"),
      ("canon", "phase", "tick"),
      ("canon", "phase", "trial"),
      ("canon", "phase", "trial", "tick"),
    ):
      a, b = keyset(first, fields), keyset(last, fields)
      print(
        "    %-28s first=%-6d last=%-6d shared=%d"
        % ("+".join(fields), len(a), len(b), len(a & b))
      )
    for arm in arms:
      sample = sorted({str(r.pair_id) for r in arm["records"]})[:3]
      group = sorted({str(r.group_key) for r in arm["records"]})[:3]
      print(f"    {arm['name']} pair_id sample={sample} group_key sample={group}")
    return 0

  # A window is only the same physical sample if its perturbation matches too.
  def perturbation_matches(key) -> bool:
    reference = float(maps[0][key].ou_noise_norm)
    for other in maps[1:]:
      value = float(other[key].ou_noise_norm)
      if abs(value - reference) > 1e-9 * max(1.0, abs(reference)):
        return False
    return True

  matched = [key for key in shared if perturbation_matches(key)]
  all_count, matched_count = len(shared), len(matched)
  shared = matched
  filter_counts: dict[str, int] = {}
  if args.only_motion is not None:
    before = len(shared)
    shared = [key for key in shared if key[0] == args.only_motion]
    filter_counts["only_motion_removed"] = before - len(shared)
  if args.frame_range is not None:
    before = len(shared)
    shared = [key for key in shared if in_frame_range(maps[0][key], args.frame_range)]
    filter_counts["frame_range_removed"] = before - len(shared)
  if args.exclude_frame_range is not None:
    before = len(shared)
    shared = [
      key
      for key in shared
      if not in_frame_range(maps[0][key], args.exclude_frame_range)
    ]
    filter_counts["exclude_frame_range_removed"] = before - len(shared)
  if args.max_windows is not None:
    # Sample across every motion rather than truncating the sorted prefix.
    per_bucket: dict[str, list] = {}
    for key in shared:
      per_bucket.setdefault(key[0], []).append(key)
    quota = max(1, args.max_windows // max(1, len(per_bucket)))
    sampled: list = []
    for keys in per_bucket.values():
      sampled.extend(keys[:quota])
    shared = sorted(sampled)
  if not shared:
    raise SystemExit("no window identities are shared by the arms")

  per_arm_records = [[m[key] for key in shared] for m in maps]
  groups: dict[tuple[str, str], list[int]] = {}
  for index, key in enumerate(shared):
    groups.setdefault((key[0], key[1]), []).append(index)

  if args.dry_run_identities:
    payload = {
      "dry_run": True,
      "split": args.split,
      "arms": {
        arm["name"]: {
          "checkpoint": arm["checkpoint"],
          "dataset": arm["dataset"],
          "records": len(arm["records"]),
          "unique_identities": len(m),
          "ambiguous_identities": arm["ambiguous_identities"],
        }
        for arm, m in zip(arms, maps, strict=True)
      },
      "shared_identity_only": all_count,
      "shared_with_matching_perturbation": matched_count,
      "excluded_perturbation_mismatch": all_count - matched_count,
      "filters": {
        "only_motion": args.only_motion,
        "frame_range": args.frame_range,
        "exclude_frame_range": args.exclude_frame_range,
      },
      "filter_counts": filter_counts,
      "groups": {f"{mn}|{p}": len(v) for (mn, p), v in sorted(groups.items())},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["groups"], indent=2, sort_keys=True))
    print(
      f"  shared identities: {all_count}; matching perturbation: {matched_count}; "
      f"excluded: {all_count - matched_count}"
    )
    print(f"report: {args.out}")
    return 0

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
      "identity_fields": [
        "motion_id_canonical(first digits)",
        "phase",
        "trial_canonical(motion:seed:start_frame)",
        "start_tick",
      ],
      "motion_ids": {
        arm["name"]: {
          str(key): str(value)
          for key, value in sorted(
            {r.motion_id: motion_key(r.motion_id) for r in arm["records"]}.items()
          )
        }
        for arm in arms
      },
      "per_arm": {arm["name"]: len(arm["records"]) for arm in arms},
      "per_arm_unique": {
        arm["name"]: len(m) for arm, m in zip(arms, maps, strict=True)
      },
      "per_arm_ambiguous": {arm["name"]: arm["ambiguous_identities"] for arm in arms},
      "shared_identity_only": all_count,
      "shared_with_matching_perturbation": matched_count,
      "excluded_perturbation_mismatch": all_count - matched_count,
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
        "all_metrics": overall[arm["name"]]["metrics"],
        "baselines": {
          key: overall[arm["name"]]["metrics"].get(key) for key in BASELINES
        },
      }
      for arm in arms
    },
    "paired": {},
    "per_group": {},
    "per_group_all_metrics": {
      group: {name: per_arm[name].get("metrics", {}) for name in per_arm}
      for group, per_arm in by_group.items()
    },
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
