"""Summarize D3 parallel fall sweeps over several checkpoints.

Each wave writes one ``member-<motion>.json`` per motion.  This tool reads every
label directory under the evidence root, or any ``--extra-label name=path``,
and prints one comparison table.  It reports what is present, so an incomplete
wave cannot be mistaken for a finished one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_MOTIONS = tuple(f"vgk{index:03d}" for index in range(8))


def _read_member(
  root: Path, motion: str, pattern: str
) -> dict[str, Any] | None:
  """Return the newest member record for one motion, or None when absent."""
  candidates = sorted(
    root.glob(pattern.format(motion=motion)),
    key=lambda path: path.stat().st_mtime,
  )
  for path in reversed(candidates):
    try:
      payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
      continue
    if "fall_count" in payload and "rows" in payload:
      payload["_path"] = str(path)
      return payload
  return None


def _labels(args: argparse.Namespace, root: Path) -> dict[str, Path]:
  """Collect label directories: discovered under the root plus explicit ones."""
  labels: dict[str, Path] = {}
  if root.is_dir():
    for child in sorted(root.iterdir()):
      if child.is_dir() and any(child.glob("*/*/member-*.json")):
        labels[child.name] = child
  for item in args.extra_label or ():
    if "=" not in item:
      raise SystemExit(f"--extra-label needs name=path, got {item!r}")
    name, path = item.split("=", 1)
    labels[name] = Path(path)
  return labels


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--repo-root", default=".")
  parser.add_argument(
    "--root",
    default="logs/diagnostics/diffusion-d3-models",
    help="evidence root whose subdirectories are checkpoint labels",
  )
  parser.add_argument(
    "--extra-label",
    action="append",
    default=None,
    help="add one evidence directory as name=path (repeatable), for older runs",
  )
  parser.add_argument(
    "--motions",
    default=",".join(DEFAULT_MOTIONS),
    help="comma-separated motion ids to report",
  )
  parser.add_argument(
    "--pattern",
    default="{motion}/*/member-{motion}.json",
    help=(
      "glob under one label for a motion record; the default matches "
      "<label>/<motion>/<run-stamp>/member-<motion>.json"
    ),
  )
  parser.add_argument("--json", type=Path, default=None, help="also write JSON")
  args = parser.parse_args(argv)
  repo = Path(args.repo_root).resolve()
  root = (repo / args.root).resolve() if not Path(args.root).is_absolute() else Path(args.root)
  motions = tuple(item.strip() for item in args.motions.split(",") if item.strip())
  labels = _labels(args, root)
  if not labels:
    raise SystemExit(f"no labelled evidence under {root}")

  print(f"{'checkpoint':26s}" + "".join(f"{motion:>9s}" for motion in motions))
  collected: dict[str, Any] = {}
  budget_sets: list[dict[str, int]] = []
  for label in sorted(labels):
    folder = labels[label]
    records = {motion: _read_member(folder, motion, args.pattern) for motion in motions}
    collected[label] = {
      "root": str(folder),
      "motions": {
        motion: (
          None
          if record is None
          else {
            "rows": record["rows"],
            "falls": record["fall_count"],
            "rate": record["fall_rate"],
            "steps_budget": record["steps_budget"],
            "stop_counts": record["stop_counts"],
            "fall_first_tick": record["fall_first_tick"],
            "wall_seconds": record["wall_seconds"],
            "path": record["_path"],
          }
        )
        for motion, record in records.items()
      },
    }
    cells = ""
    rows = falls = 0
    stops: dict[str, int] = {}
    for motion in motions:
      record = records[motion]
      if record is None:
        cells += f"{'--':>9s}"
        continue
      cells += f"{record['fall_count']:>9d}"
      rows += int(record["rows"])
      falls += int(record["fall_count"])
      for key, value in record["stop_counts"].items():
        stops[key] = stops.get(key, 0) + int(value)
    total = f"{falls}/{rows}" if rows else "0/0"
    rate = f"{falls / rows:.1%}" if rows else "n/a"
    missing = sum(1 for motion in motions if records[motion] is None)
    flag = "" if missing == 0 else f"  [{len(motions) - missing}/{len(motions)} motions]"
    print(f"{label:26s}{cells}{total:>12s}{rate:>8s}{flag}")
    budget_by_motion = {
      motion: records[motion]["steps_budget"]
      for motion in motions
      if records[motion] is not None
    }
    budget_sets.append(budget_by_motion)
  reference = budget_sets[0] if budget_sets else {}
  if reference:
    print(
      "per-motion tick budget: "
      + " ".join(f"{motion}={reference[motion]}" for motion in motions if motion in reference)
    )
    mismatch = [entry for entry in budget_sets if entry != reference]
    if mismatch:
      print(f"WARNING: {len(mismatch)} label(s) use a different budget than the first")
  if args.json is not None:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(collected, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.json}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
