"""Merge per-worker D1 shard stores into one store.

Parallel collection writes one store per worker (the manifest is a single-writer
file).  This joins them by streaming every row through the store's own validated
``append``, so the merged store satisfies the same integrity checks the single
process would have produced and duplicate identities fail closed.
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
  parser.add_argument("--max-rows-per-shard", type=int, default=4096)
  parser.add_argument("--batch", type=int, default=4096)
  args = parser.parse_args(argv)

  from mjlab.scripts.diffusion import _load_contract
  from mjlab.tasks.tracking.diffusion.storage import AppendOnlyShardStore

  contract = _load_contract(args.contract)
  output = AppendOnlyShardStore(
    args.output, contract=contract, max_rows_per_shard=args.max_rows_per_shard
  )
  before = output.row_count

  per_input: dict[str, int] = {}
  for path in args.input:
    source = AppendOnlyShardStore(path, contract=contract)
    rows = 0
    batch: list[object] = []
    for row in source.iter_rows():
      batch.append(row)
      if len(batch) >= args.batch:
        output.append(batch)  # type: ignore[arg-type]
        rows += len(batch)
        batch = []
    if batch:
      output.append(batch)  # type: ignore[arg-type]
      rows += len(batch)
    per_input[str(path)] = rows

  payload = {
    "inputs": per_input,
    "input_rows": sum(per_input.values()),
    "output": str(args.output),
    "rows_before": before,
    "rows_after": output.row_count,
    "shards": output.shard_count,
  }
  if before != 0 or payload["rows_after"] != payload["input_rows"]:
    raise SystemExit(f"merge row accounting failed: {json.dumps(payload)}")
  print(json.dumps(payload, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
