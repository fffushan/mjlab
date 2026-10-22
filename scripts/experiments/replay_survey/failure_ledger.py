"""Accumulate survey passes into one per-segment failure ledger.

The ledger is the single place that answers "how many times did this segment fail
when it was replayed as a reference".  A pass is identified by an id; re-merging a
pass replaces its contribution instead of adding it twice, so a crashed or
repeated merge cannot inflate a count.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

LEDGER = Path("/home/fushan/data/mjlab-latent-free-144/failure_ledger.json")
SCHEMA = 1


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--pass-dir", type=Path, required=True, help="dir with lane*/batch*.json")
  ap.add_argument("--pass-id", required=True)
  ap.add_argument("--ledger", type=Path, default=LEDGER)
  ap.add_argument("--batch", type=int, default=128)
  args = ap.parse_args()

  ledger = {"schema": SCHEMA, "passes": {}, "segments": {}}
  if args.ledger.is_file():
    ledger = json.loads(args.ledger.read_text())
    if ledger.get("schema") != SCHEMA:
      raise RuntimeError(f"ledger schema {ledger.get('schema')} is not {SCHEMA}")

  files = sorted(args.pass_dir.glob("lane*/batch*.json"))
  if not files:
    raise RuntimeError(f"no batch results under {args.pass_dir}")
  runs = 0
  this_pass: dict[str, dict] = {}
  for path in files:
    payload = json.loads(path.read_text())
    for row in payload["rows"]:
      seg = str(row["segment"])
      runs += 1
      entry = this_pass.setdefault(seg, {
          "runs": 0,
          "failures": 0,
          "terms": Counter(),
          "steps": [],
          "group": row["group"],
          "motion": row["motion"],
          "windows": row["windows"],
      })
      if entry["runs"]:
        raise RuntimeError(f"segment {seg} appears twice in pass {args.pass_id}")
      entry["runs"] += 1
      if row["outcome"] != "complete":
        entry["failures"] += 1
        entry["terms"][row["term"] or row["outcome"]] += 1
      entry["steps"].append(int(row["steps"]))

  # Replace this pass's contribution, then recompute the per-segment totals.
  previous = ledger["passes"].get(args.pass_id)
  if previous is not None:
    print(f"[INFO] replacing existing pass {args.pass_id} ({len(previous['segments'])} rows)")
  totals = ledger["segments"]
  for seg in (previous["segments"] if previous else {}):
    totals.pop(seg, None)

  for seg, entry in this_pass.items():
    totals[seg] = {
        "runs": entry["runs"],
        "failures": entry["failures"],
        "terms": dict(entry["terms"]),
        "steps": entry["steps"],
        "group": entry["group"],
        "motion": entry["motion"],
        "windows": entry["windows"],
    }

  ledger["passes"][args.pass_id] = {
      "created_utc": datetime.now(timezone.utc).isoformat(),
      "rows": runs,
      "segments": sorted(this_pass),
      "batch": args.batch,
  }

  # Summary by group, motion, and termination term.
  by_group: dict[str, dict[str, int]] = defaultdict(
      lambda: {"segments": 0, "failed": 0, "windows": 0, "failed_windows": 0}
  )
  by_motion: dict[str, dict[str, int]] = defaultdict(lambda: {"segments": 0, "failed": 0})
  by_term: Counter = Counter()
  for e in totals.values():
    g = by_group[e["group"]]
    g["segments"] += 1
    g["windows"] += e["windows"]
    if e["failures"]:
      g["failed"] += 1
      g["failed_windows"] += e["windows"]
    m = by_motion[e["motion"].replace("_zhanghongyu_agibot_x2_tracking.npz", "")]
    m["segments"] += 1
    m["failed"] += 1 if e["failures"] else 0
    for term, count in e["terms"].items():
      by_term[term] += count

  failed = sum(1 for e in totals.values() if e["failures"])
  ledger["summary"] = {
      "segments": len(totals),
      "segments_with_failure": failed,
      "segment_failure_rate": round(failed / len(totals), 4) if totals else None,
      "windows": sum(e["windows"] for e in totals.values()),
      "windows_with_failure": sum(e["windows"] for e in totals.values() if e["failures"]),
      "terms": dict(by_term),
      "by_group": dict(by_group),
      "motions_with_failure": sum(1 for v in by_motion.values() if v["failed"]),
      "motions": len(by_motion),
      "passes": sorted(ledger["passes"]),
  }
  args.ledger.parent.mkdir(parents=True, exist_ok=True)
  args.ledger.write_text(json.dumps(ledger))
  print(json.dumps(ledger["summary"], indent=1))


if __name__ == "__main__":
  main()
