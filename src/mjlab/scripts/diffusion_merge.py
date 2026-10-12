"""Merge per-worker D1 shard stores into one store.

Parallel collection writes one store per worker (the manifest is a single-writer
file).  The default merge is byte-preserving: each validated source shard is
hash-checked and copied, so the merged store keeps the source payloads and never
decodes or recompresses a row.  ``--legacy-row-merge`` keeps the historical
row-level reference path, which streams every row through the store's own
validated ``append``.

Both paths refuse an input store that was written under another contract or
artifact identity, an input manifest that repeats a shard file or names an
unsafe file, a forged row count, malformed row metadata, a duplicate row identity
and an input store that changed while it was read.  The byte-preserving path
refuses an existing output, stages the merged store in a sibling directory and
publishes it with one atomic rename; the source stores are opened read-only and
never linked, so they stay byte-identical.

Row accounting is reported in the same JSON payload key names the historical
command used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
    prog="diffusion_merge", description="merge per-worker D1 shard stores"
  )
  parser.add_argument(
    "--contract",
    type=Path,
    default=Path("docs/plans/beyondmimic_diffusion_d0_contract.yaml"),
  )
  parser.add_argument("--input", type=Path, action="append", required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
    "--max-rows-per-shard",
    type=int,
    default=None,
    help=(
      "output shard bound; the default is the largest source shard for the "
      "byte-preserving merge and 4096 for the legacy row merge"
    ),
  )
  parser.add_argument(
    "--batch",
    type=int,
    default=4096,
    help="rows appended per batch; only used by --legacy-row-merge",
  )
  parser.add_argument(
    "--legacy-row-merge",
    action="store_true",
    help="rewrite every row through the store writer instead of copying shards",
  )
  args = parser.parse_args(argv)

  from mjlab.scripts.diffusion import _load_contract
  from mjlab.tasks.tracking.diffusion.storage import (
    StorageError,
    merge_shard_stores,
    merge_shard_stores_legacy,
  )

  contract = _load_contract(args.contract)
  try:
    if args.legacy_row_merge:
      report = merge_shard_stores_legacy(
        args.input,
        args.output,
        contract=contract,
        max_rows_per_shard=(
          4096 if args.max_rows_per_shard is None else args.max_rows_per_shard
        ),
        batch_rows=args.batch,
      )
    else:
      report = merge_shard_stores(
        args.input,
        args.output,
        contract=contract,
        max_rows_per_shard=args.max_rows_per_shard,
      )
  except StorageError as exc:
    raise SystemExit(str(exc)) from None
  payload = report.as_payload()
  print(json.dumps(payload, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
