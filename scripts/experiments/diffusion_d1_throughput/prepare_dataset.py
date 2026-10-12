#!/usr/bin/env python3
"""Scalable D1 dataset preparation entry point.

This tool turns validated per-worker D1 shard stores into a trainable dataset
directory.  It is deliberately *not* a collection tool and *not* a training
tool: it never constructs a simulator, never starts a collection runtime and
never starts a training run.

Phases, in order:

1. ``merge`` (optional, only with ``--input``): gate every worker store with its
   own collection report, ``inspect`` and ``replay``, then merge the stores
   byte-preservingly.  A partial worker, a stopped worker, a row-accounting
   mismatch and a duplicate identity are refused before anything is written.
2. ``dataset``: build the grouped split assignments and the window index once,
   persist the index beside the dataset and reload it to prove it describes the
   exact store and assignments.
3. ``statistics``: fit train-only projection statistics from the persisted
   index, so the index is not rebuilt for the fit.
4. ``cache``: prepare the token caches with bounded worker counts.

Every phase reports its wall time.  The merged store, the window index and the
JSON reports are staged and published atomically, so a reader never sees those
files partially written; the projection bundle is written by
``ProjectionBundle.save``.  Training is out of scope: the tool prints the
training command that consumes the prepared dataset with
``--require-prepared-cache``.

The scientific contract is unchanged.  This tool only orchestrates the existing
library functions and passes their bounded worker counts through.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

DEFAULT_CONTRACT_PATH = Path(
  "docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml"
)


class PreparationError(RuntimeError):
  """A preparation gate failed, so no output was published."""


def _load_contract(path: Path) -> Any:
  from mjlab.tasks.tracking.diffusion.contract import DiffusionContract

  return DiffusionContract.from_yaml(path)


def _json_write(path: Path, payload: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f".{path.name}.tmp")
  temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  temporary.replace(path)


def _require_absent(path: Path, *, what: str) -> None:
  """Refuse to overwrite an existing output, so every run stages new files."""
  if path.exists():
    raise PreparationError(f"refusing to overwrite existing {what}: {path}")


def validate_worker_report(
  report_path: Path,
  *,
  expected_contract_sha256: str,
  stored_rows: int,
) -> dict[str, Any]:
  """Apply the strict partial, stop and row-accounting gates to one worker.

  A worker is admissible only when its report says the schedule completed, it
  names the active contract and the rows it claims to have stored equal the
  rows the store actually holds.
  """
  try:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
    raise PreparationError(
      f"unreadable collection report {report_path}: {exc}"
    ) from exc
  if not isinstance(payload, dict):
    raise PreparationError(f"collection report {report_path} is not a JSON object")
  if bool(payload.get("partial", False)):
    raise PreparationError(
      f"worker report {report_path} is partial "
      f"(stop_reason={payload.get('stop_reason')!r}); do not merge a partial store"
    )
  if payload.get("stop_reason") is not None:
    raise PreparationError(
      f"worker report {report_path} stopped with reason "
      f"{payload['stop_reason']!r}; do not merge a stopped store"
    )
  recorded_contract = payload.get("contract_sha256")
  if recorded_contract != expected_contract_sha256:
    raise PreparationError(
      f"worker report {report_path} contract {recorded_contract!r} does not match "
      f"the active contract {expected_contract_sha256!r}"
    )
  results = payload.get("results")
  if not isinstance(results, list):
    raise PreparationError(f"worker report {report_path} has no results list")
  reported_rows = sum(int(entry.get("rows", 0)) for entry in results)
  if reported_rows != stored_rows:
    raise PreparationError(
      f"worker report {report_path} accounts for {reported_rows} rows but the "
      f"store holds {stored_rows}"
    )
  return {
    "report": str(report_path),
    "reported_rows": reported_rows,
    "contract_sha256": recorded_contract,
  }


def validate_worker_store(
  store_root: Path,
  *,
  contract: Any,
  jobs: int,
  report_path: Path | None = None,
  require_report: bool = True,
) -> dict[str, Any]:
  """Run per-input inspect and replay, plus the report gates when present."""
  from mjlab.tasks.tracking.diffusion.offline_validation import (
    inspect_store,
    replay_store,
  )

  inspection = inspect_store(store_root, contract=contract, jobs=jobs)
  if not inspection["unique_identities"]:
    raise PreparationError(f"store {store_root} has duplicate row identities")
  replay = replay_store(store_root, contract=contract, jobs=jobs)
  if not replay["ok"]:
    raise PreparationError(
      f"store {store_root} failed offline replay with {len(replay['errors'])} errors: "
      f"{replay['errors'][:3]}"
    )
  entry: dict[str, Any] = {
    "store": str(store_root),
    "shards": inspection["shards"],
    "rows": inspection["rows"],
    "controllers": inspection["controllers"],
    "unique_identities": inspection["unique_identities"],
    "replay_ok": replay["ok"],
  }
  if report_path is not None and report_path.is_file():
    entry["report"] = validate_worker_report(
      report_path,
      expected_contract_sha256=contract.contract_sha256,
      stored_rows=int(inspection["rows"]),
    )
  elif require_report:
    raise PreparationError(
      f"no collection report for {store_root}: expected {report_path}"
    )
  return entry


def _report_path_for_store(store_root: Path) -> Path:
  """The runbook convention: a store at ``<run>/store`` reports at ``<run>``."""
  if store_root.name == "store":
    return store_root.parent / "report.json"
  return store_root / "report.json"


def main(argv: Sequence[str] | None = None) -> int:  # noqa: PLR0915 - one entry point
  parser = argparse.ArgumentParser(
    prog="prepare_dataset",
    description=(
      "validate and merge worker stores, build the dataset, fit train-only "
      "statistics and prepare token caches.  Never starts training."
    ),
  )
  parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
  parser.add_argument("--dataset-dir", type=Path, required=True)
  parser.add_argument(
    "--input",
    type=Path,
    action="append",
    default=None,
    help="per-worker store to validate and merge; repeat once per worker",
  )
  parser.add_argument(
    "--input-report",
    type=Path,
    action="append",
    default=None,
    help="collection report for the matching --input; default is derived",
  )
  parser.add_argument(
    "--jobs",
    type=int,
    default=8,
    help="bounded workers for validation/index/statistics (default: 8)",
  )
  parser.add_argument(
    "--batch-windows",
    type=int,
    default=512,
    help="windows per statistics accumulator update (default: 512)",
  )
  parser.add_argument("--splits", default="train,validation,test")
  parser.add_argument(
    "--motion",
    action="append",
    default=None,
    help="limit the token cache to these motion ids; repeatable",
  )
  parser.add_argument("--cache-jobs", type=int, default=1)
  parser.add_argument("--max-cache-gib", type=float, default=None)
  parser.add_argument("--lock-timeout", type=float, default=1800.0)
  parser.add_argument("--report", type=Path)
  args = parser.parse_args(argv)

  if args.jobs < 1:
    print("prepare: --jobs must be one or more", file=sys.stderr)
    return 2
  if args.cache_jobs < 1:
    print("prepare: --cache-jobs must be one or more", file=sys.stderr)
    return 2
  if args.batch_windows is not None and args.batch_windows < 1:
    print("prepare: --batch-windows must be one or more", file=sys.stderr)
    return 2

  from mjlab.tasks.tracking.diffusion.dataset import (
    build_window_index,
    grouped_split,
    load_window_index,
    save_window_index,
    store_manifest_sha256,
    window_index_path,
  )
  from mjlab.tasks.tracking.diffusion.storage import (
    AppendOnlyShardStore,
    merge_shard_stores,
  )

  try:
    contract = _load_contract(args.contract)
  except (OSError, RuntimeError, ValueError, TypeError) as exc:
    print(f"prepare: contract: {exc}", file=sys.stderr)
    return 2

  dataset = Path(args.dataset_dir)
  offline = dataset / "offline"
  report: dict[str, Any] = {
    "dataset": str(dataset),
    "contract_sha256": contract.contract_sha256,
    "jobs": args.jobs,
    "cache_jobs": args.cache_jobs,
    "phases": [],
    "training_started": False,
  }
  phases: list[dict[str, Any]] = report["phases"]

  def run_phase(name: str, action: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    print(f"prepare[{name}]: starting", flush=True)
    try:
      detail = action()
    except PreparationError:
      raise
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
      raise PreparationError(f"{name} phase failed: {exc}") from exc
    seconds = round(time.perf_counter() - started, 3)
    entry = {"phase": name, "seconds": seconds, **detail}
    phases.append(entry)
    print(f"prepare[{name}]: done in {seconds:.1f}s", flush=True)
    return entry

  try:
    store_root = dataset / "store"
    if args.input:
      reports = list(args.input_report or [])
      entries: list[dict[str, Any]] = []
      inputs: list[Path] = []
      for position, raw_input in enumerate(args.input):
        source = Path(raw_input)
        if not source.is_dir():
          raise PreparationError(f"input store is missing: {source}")
        report_path = (
          Path(reports[position])
          if position < len(reports)
          else _report_path_for_store(source)
        )
        entries.append(
          validate_worker_store(
            source,
            contract=contract,
            jobs=args.jobs,
            report_path=report_path,
            require_report=True,
          )
        )
        inputs.append(source)

      def stage_merge() -> dict[str, Any]:
        _require_absent(store_root, what="dataset store")
        merged = merge_shard_stores(inputs, store_root, contract=contract)
        audit = validate_worker_store(
          store_root, contract=contract, jobs=args.jobs, require_report=False
        )
        return {"merge": merged.as_payload(), "merged_audit": audit}

      run_phase("merge", stage_merge)
      report["workers"] = entries
    else:
      if not store_root.is_dir():
        raise PreparationError(
          f"no --input given and the dataset store is missing: {store_root}"
        )
      run_phase(
        "audit",
        lambda: {
          "store": str(store_root),
          "audit": validate_worker_store(
            store_root, contract=contract, jobs=args.jobs, require_report=False
          ),
        },
      )

    store = AppendOnlyShardStore(store_root, contract=contract)

    def stage_dataset() -> dict[str, Any]:
      offline.mkdir(parents=True, exist_ok=True)
      assignments = grouped_split(store.iter_rows(), contract=contract)
      index, method = build_window_index(
        store, assignments, jobs=args.jobs, contract=contract
      )
      assignment_path = offline / "assignments.json"
      _require_absent(assignment_path, what="split assignments")
      _json_write(assignment_path, assignments.as_dict())
      _require_absent(offline / "dataset.json", what="dataset manifest")
      _json_write(
        offline / "dataset.json",
        {
          "contract_sha256": contract.contract_sha256,
          "groups": assignments.coverage(),
          "windows": index.coverage(),
          "total_windows": len(index),
        },
      )
      index_path = window_index_path(dataset)
      _require_absent(index_path, what="window index")
      record = save_window_index(
        index_path,
        index,
        store_manifest_sha256=store_manifest_sha256(store),
        assignments_sha256=assignments.sha256(),
      )
      # Reload once: the index is source-bound and must describe this exact
      # store and assignment set before any later phase trusts it.
      reloaded = load_window_index(
        index_path, store=store, assignments=assignments, contract=contract
      )
      if reloaded.refs != index.refs:
        raise PreparationError("persisted window index does not reload identically")
      return {
        "method": method,
        "index_source": str(index_path),
        "refs": len(index),
        "coverage": index.coverage(),
        **record,
      }

    dataset_entry = run_phase("dataset", stage_dataset)
    index_path = Path(dataset_entry["index_source"])

    def stage_statistics() -> dict[str, Any]:
      from mjlab.tasks.tracking.diffusion.dataset import fit_training_statistics

      assignments = _load_assignments(offline / "assignments.json")
      index = load_window_index(
        index_path, store=store, assignments=assignments, contract=contract
      )
      statistics_kwargs = (
        {} if args.batch_windows is None else {"batch_windows": args.batch_windows}
      )
      bundle = fit_training_statistics(
        index, contract=contract, jobs=args.jobs, **statistics_kwargs
      )
      bundle_path = offline / "projection.npz"
      _require_absent(bundle_path, what="projection bundle")
      bundle.save(bundle_path)
      payload = {
        "bundle": str(bundle_path),
        "batch_windows": args.batch_windows,
        "train_windows_used": len(index.refs_for("train")),
        "matrix_sha256": bundle.matrix_sha256,
        "statistics_sha256": bundle.statistics_sha256,
      }
      _require_absent(offline / "stats.json", what="statistics report")
      _json_write(offline / "stats.json", payload)
      return payload

    run_phase("statistics", stage_statistics)

    def stage_cache() -> dict[str, Any]:
      from mjlab.tasks.tracking.diffusion.token_cache_lifecycle import (
        parse_splits,
        prepare_token_cache,
      )
      from mjlab.tasks.tracking.diffusion.window_dataset import (
        _DEFAULT_MAX_CACHE_GIB,
        DatasetSource,
        _source_dataset_hash,
      )

      source = DatasetSource.resolve(dataset, contract_path=args.contract)
      loaded = source.load()
      source_hash = _source_dataset_hash(source, loaded)
      ceiling = (
        _DEFAULT_MAX_CACHE_GIB if args.max_cache_gib is None else args.max_cache_gib
      )
      splits = parse_splits(args.splits)
      prepared: dict[str, Any] = {}
      for split in splits:
        result = prepare_token_cache(
          source,
          loaded,
          split,
          motion_ids=args.motion,
          max_cache_gib=ceiling,
          jobs=args.cache_jobs,
          source_hash=source_hash,
          lock_timeout=args.lock_timeout,
        )
        prepared[split] = {
          "status": result.cache_status,
          "path": str(result.path),
          "windows": result.count,
        }
        print(f"prepare[cache]: {split} {result.cache_status}", flush=True)
      return {"splits": prepared, "source_hash": source_hash}

    run_phase("cache", stage_cache)
  except (PreparationError, OSError, RuntimeError, ValueError, TypeError) as exc:
    report["error"] = str(exc)
    report["complete"] = False
    print(f"prepare: {exc}", file=sys.stderr)
    if args.report is not None:
      _json_write(args.report, report)
    return 1

  report["complete"] = True
  report["training_command"] = _training_command(args, dataset)
  if args.report is not None:
    _json_write(args.report, report)
  print(json.dumps(report, indent=2, sort_keys=True))
  print(
    "prepare: dataset is ready.  Training is out of scope for this tool; run:\n"
    f"  {report['training_command']}",
    file=sys.stderr,
  )
  return 0


def _load_assignments(path: Path) -> Any:
  from mjlab.tasks.tracking.diffusion.dataset import SplitAssignments

  return SplitAssignments.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _training_command(args: argparse.Namespace, dataset: Path) -> str:
  """Print the training command; the tool never runs it."""
  cache_flag = " --require-prepared-cache"
  return (
    "uv run --no-sync python -m mjlab.scripts.diffusion train "
    f"--dataset-dir {dataset} --config <TRAIN_CONFIG> --contract {args.contract} "
    "--out <RUN_DIR> --device cuda:0" + cache_flag
  )


if __name__ == "__main__":
  raise SystemExit(main())
