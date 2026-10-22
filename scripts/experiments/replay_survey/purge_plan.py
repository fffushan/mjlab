"""Turn the failure ledger into an exclusion list, without touching any data.

This tool only reads. It reports what a given purge rule would remove, so the
rule can be chosen before anything is deleted or rebuilt.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path("/home/fushan/data/mjlab-latent-free-144/data")
LEDGER = Path("/home/fushan/data/mjlab-latent-free-144/failure_ledger.json")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--ledger", type=Path, default=LEDGER)
  ap.add_argument("--assignments", type=Path, default=DATA / "assignments.json")
  ap.add_argument("--min-runs", type=int, default=1)
  ap.add_argument("--min-failures", type=int, default=1)
  ap.add_argument("--out", type=Path, default=None)
  args = ap.parse_args()

  ledger = json.loads(args.ledger.read_text())

  assignments: dict[str, str] = {}
  if args.assignments.is_file():
    payload = json.loads(args.assignments.read_text())
    # assignments.json maps each motion name to its split name under "motions".
    table_map = payload.get("motions", {})
    if isinstance(table_map, dict):
      for motion, value in table_map.items():
        split = value if isinstance(value, str) else value.get("split")
        if split is not None:
          assignments[str(motion)] = str(split)

  totals = ledger["segments"]
  excluded = []
  rule_hist: Counter = Counter()
  group_purged: dict[str, int] = defaultdict(int)
  group_total: dict[str, int] = defaultdict(int)
  per_motion_purged: dict[str, int] = defaultdict(int)
  per_motion_total: dict[str, int] = defaultdict(int)
  windows_total = 0
  windows_purged = 0
  excluded_windows_by_split: dict[str, int] = defaultdict(int)

  for seg, entry in totals.items():
    windows_total += entry["windows"]
    group_total[entry["group"]] += 1
    per_motion_total[entry["motion"]] += 1
    hit = entry["runs"] >= args.min_runs and entry["failures"] >= args.min_failures
    rule_hist[f"{entry['failures']}/{entry['runs']}"] += 1
    if hit:
      excluded.append(int(seg))
      windows_purged += entry["windows"]
      group_purged[entry["group"]] += 1
      per_motion_purged[entry["motion"]] += 1
      excluded_windows_by_split[assignments.get(entry["motion"], "unassigned")] += entry["windows"]

  # A motion that loses every segment would vanish from its split.
  emptied = sorted(
      motion
      for motion, total in per_motion_total.items()
      if per_motion_purged.get(motion, 0) == total
  )
  heaviest = sorted(
      per_motion_purged.items(), key=lambda item: -item[1] / max(per_motion_total[item[0]], 1)
  )[:8]

  report = {
      "rule": {"min_runs": args.min_runs, "min_failures": args.min_failures},
      "segments_measured": len(totals),
      "segments_excluded": len(excluded),
      "segments_retained": len(totals) - len(excluded),
      "windows_total": windows_total,
      "windows_excluded": windows_purged,
      "windows_retained": windows_total - windows_purged,
      "windows_excluded_by_split": dict(excluded_windows_by_split),
      "failure_histogram": dict(sorted(rule_hist.items())),
      "by_group": {
          g: {"total": group_total[g], "excluded": group_purged.get(g, 0)}
          for g in sorted(group_total)
      },
      "motions_emptied": emptied,
      "heaviest_motions": [
          {"motion": m, "excluded": n, "total": per_motion_total[m]}
          for m, n in heaviest
      ],
  }
  print(json.dumps(report, indent=1))
  if args.out is not None:
    args.out.write_text(json.dumps({"rule": report["rule"], "segments": excluded}))
    print(f"[INFO] wrote {args.out} ({len(excluded)} segments)")


if __name__ == "__main__":
  main()
