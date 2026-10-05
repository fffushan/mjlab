"""Summarise a diffusion training wave and a paired per-motion comparison.

Two jobs, both needed the moment a wave finishes:

A. Training curves.  For each arm, report the best validation loss and where in the
   budget it occurred, the final validation loss, and whether the tail of the curve
   is still descending.  That last flag is the budget question: the three-clip runs
   bottomed at ~40-45% of their budget and then overfit for the remainder, while the
   twelve-motion arm was still descending when it stopped.

B. Per-motion sign consistency.  An aggregate delta can hide inversions: in both
   earlier experiments motion 001 moved the opposite way to every other motion, so
   an arm reported as a win also made one motion worse on every channel.  This
   reports, per arm and channel, how many motions improved, how many worsened, the
   median and the extremes, so a wave is never judged on its mean alone.

Both parts are read-only and operate on files already on disk (no GPU, no dataset).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from typing import Any


def curve_summary(
  validation_by_epoch: dict[int, float], *, tail_fraction: float = 0.05
) -> dict[str, Any]:
  """Best/final validation loss and whether the tail is still descending."""
  if not validation_by_epoch:
    raise ValueError("no validation values")
  epochs = sorted(validation_by_epoch)
  values = [validation_by_epoch[epoch] for epoch in epochs]
  best_index = min(range(len(values)), key=lambda i: values[i])
  n = len(values)
  tail = max(1, round(n * tail_fraction))
  last = values[-tail:]
  previous = values[-2 * tail : -tail] if n >= 2 * tail else values[: max(1, n - tail)]
  still_descending = bool(previous) and statistics.median(last) < statistics.median(
    previous
  )
  return {
    "epochs_evaluated": n,
    "best_validation_loss": values[best_index],
    "best_epoch": epochs[best_index],
    "best_budget_fraction": (best_index + 1) / n,
    "final_validation_loss": values[-1],
    "final_over_best": values[-1] / values[best_index] if values[best_index] else None,
    "tail_still_descending": still_descending,
    "tail_last_median": statistics.median(last),
    "tail_previous_median": statistics.median(previous) if previous else None,
  }


def read_validation_curve(metrics_path: pathlib.Path) -> dict[int, float]:
  """Validation loss per epoch from a trainer ``metrics.jsonl``."""
  curve: dict[int, float] = {}
  with metrics_path.open() as handle:
    for line in handle:
      line = line.strip()
      if not line:
        continue
      row = json.loads(line)
      if row.get("kind") != "epoch":
        continue
      value = (row.get("eval_ema") or {}).get("loss")
      if isinstance(value, (int, float)):
        curve[int(row["epoch"])] = float(value)
  return curve


def count_nonfinite_rows(metrics_path: pathlib.Path) -> int:
  """Rows carrying any non-finite numeric metric, for the integrity statement."""
  bad = 0
  with metrics_path.open() as handle:
    for line in handle:
      line = line.strip()
      if not line:
        continue
      row = json.loads(line)
      values: list[float] = []
      for value in row.values():
        if isinstance(value, float):
          values.append(value)
        elif isinstance(value, dict):
          values.extend(inner for inner in value.values() if isinstance(inner, float))
      if any(v != v or v in (float("inf"), float("-inf")) for v in values):
        bad += 1
  return bad


def per_motion_deltas(
  per_group: dict[str, Any],
  *,
  reference: str,
  arm: str,
  channel: str,
  weights: dict[str, int] | None = None,
) -> dict[str, float]:
  """``arm - reference`` per motion, phases combined by window count when known."""
  if arm == reference:
    raise ValueError("arm and reference must differ")
  parts: dict[str, list[tuple[float, float]]] = {}
  for group, arms in per_group.items():
    if "|" not in group:
      raise ValueError(f"unexpected group label {group!r}")
    motion, _phase = group.split("|", 1)
    if arm not in arms or reference not in arms:
      raise ValueError(f"group {group!r} lacks arm {arm!r} or reference {reference!r}")
    delta = float(arms[arm][channel]) - float(arms[reference][channel])
    weight = float((weights or {}).get(group, 1.0))
    parts.setdefault(motion, []).append((delta, weight))
  return {
    motion: sum(d * w for d, w in pairs) / sum(w for _, w in pairs)
    for motion, pairs in parts.items()
  }


def sign_consistency(
  deltas: dict[str, float], *, tolerance: float = 1e-12
) -> dict[str, Any]:
  """How many motions improved or worsened, and by how much."""
  if not deltas:
    raise ValueError("no deltas")
  better = {m: d for m, d in deltas.items() if d < -tolerance}
  worse = {m: d for m, d in deltas.items() if d > tolerance}
  ordered = sorted(deltas.items(), key=lambda kv: kv[1])
  return {
    "motions": len(deltas),
    "better": len(better),
    "worse": len(worse),
    "unchanged": len(deltas) - len(better) - len(worse),
    "median_delta": statistics.median([d for _, d in deltas.items()]),
    "best_motion": ordered[0][0],
    "best_delta": ordered[0][1],
    "worst_motion": ordered[-1][0],
    "worst_delta": ordered[-1][1],
  }


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--run",
    action="append",
    default=[],
    metavar="NAME=RUN_DIR",
    help="repeat per arm; needs train-report.json and metrics.jsonl in RUN_DIR",
  )
  parser.add_argument("--paired", type=pathlib.Path, default=None)
  parser.add_argument(
    "--reference", default=None, help="reference arm in the paired file"
  )
  parser.add_argument(
    "--channels", default="normalized latent,normalized state,normalized token"
  )
  parser.add_argument("--out", type=pathlib.Path, required=True)
  args = parser.parse_args()

  payload: dict[str, Any] = {"arms": {}, "paired": None}

  if args.run:
    for raw in args.run:
      name, _, run_dir = raw.partition("=")
      directory = pathlib.Path(run_dir)
      report_path = directory / "train-report.json"
      metrics_path = directory / "metrics.jsonl"
      entry: dict[str, Any] = {"run_dir": str(directory)}
      if report_path.exists():
        report = json.loads(report_path.read_text())
        for key in (
          "resolved_updates",
          "epochs_completed",
          "best_validation_loss",
          "final_train_loss",
          "config_hash",
        ):
          if key in report:
            entry[key] = report[key]
      if metrics_path.exists():
        entry.update(curve_summary(read_validation_curve(metrics_path)))
        entry["nonfinite_rows"] = count_nonfinite_rows(metrics_path)
      entry["periodic_checkpoints"] = len(
        sorted(directory.glob("checkpoint-iter-*.pt"))
      )
      payload["arms"][name] = entry
    header = f"{'arm':26s} {'updates':>8} {'ep':>5} {'best':>9} {'@ep':>5} {'final':>9} {'final/best':>10} {'desc':>5} {'ckpt':>5} {'nan':>4}"
    print(header)
    print("-" * len(header))
    for name, entry in payload["arms"].items():
      print(
        "%-26s %8s %5s %9.6f %5s %9.6f %10s %5s %5s %4s"
        % (
          name,
          entry.get("resolved_updates", "-"),
          entry.get("epochs_completed", "-"),
          entry.get("best_validation_loss", float("nan")),
          entry.get("best_epoch", "-"),
          entry.get("final_validation_loss", float("nan")),
          ("%.3f" % entry["final_over_best"]) if entry.get("final_over_best") else "-",
          "yes" if entry.get("tail_still_descending") else "no",
          entry.get("periodic_checkpoints", "-"),
          entry.get("nonfinite_rows", "-"),
        )
      )

  if args.paired is not None:
    paired = json.loads(args.paired.read_text())
    reference = args.reference or paired.get("reference_arm")
    if not reference:
      raise SystemExit("--reference is required when the file names no reference arm")
    groups = paired.get("per_group") or {}
    weights = (paired.get("window_identities") or {}).get("groups") or None
    payload["paired"] = {"reference": reference, "channels": {}, "motions": {}}
    for channel in [c.strip() for c in args.channels.split(",") if c.strip()]:
      payload["paired"]["channels"][channel] = {}
      for arm in sorted({a for arms in groups.values() for a in arms}):
        if arm == reference:
          continue
        deltas = per_motion_deltas(
          groups, reference=reference, arm=arm, channel=channel, weights=weights
        )
        payload["paired"]["channels"][channel][arm] = {
          **sign_consistency(deltas),
          "per_motion": deltas,
        }
        payload["paired"]["motions"].setdefault(arm, {})[channel] = deltas
      print(f"\n=== {channel} (arm - {reference}), per motion")
      for arm, entry in sorted(payload["paired"]["channels"][channel].items()):
        print(
          "  %-26s better %2d/%d  worse %2d  median %+.5f  best %s %+.5f  worst %s %+.5f"
          % (
            arm,
            entry["better"],
            entry["motions"],
            entry["worse"],
            entry["median_delta"],
            entry["best_motion"],
            entry["best_delta"],
            entry["worst_motion"],
            entry["worst_delta"],
          )
        )

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  print(f"\nreport: {args.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
