"""Summarise the eight E0 evaluation arms into one batch report.

Reads the per-arm artifacts that :mod:`evaluate_arm` wrote and produces:

* one row per arm: mode, target, episodes, qualified flights, longest sequence,
  losses, falls, holds, gate result,
* the cross-arm check that every arm started from the same recorded release states,
* the per-target view of the plan's 26/32 gate across the three seeds.

The summary never recomputes a rate from training aggregates: every number comes
from the per-episode records of this batch.

Exit status: 0 when every arm finished and the release states match. 20 when an arm
failed. 21 when the release states disagree, because then the arms are not a
like-for-like comparison even if each arm scored fine.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))

ARM_ORDER = (
  "t020-s7",
  "t020-s17",
  "t020-s27",
  "t040-s7",
  "t040-s17",
  "t040-s27",
  "zero-t020",
  "zero-t040",
)
"""The eight arms in report order: three seeds per target, then the two controls."""

EXIT_ARM_FAILED = 20
EXIT_STATE_MISMATCH = 21


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--batch-dir",
    type=Path,
    required=True,
    help="Directory holding one sub-directory per arm.",
  )
  return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any] | None:
  if not path.is_file():
    return None
  return json.loads(path.read_text())


def _initial_states(path: Path) -> dict[str, Any] | None:
  import numpy as np

  if not path.is_file():
    return None
  with np.load(path) as data:
    return {
      "initial_ball_pos_w": data["initial_ball_pos_w"],
      "initial_ball_vel_w": data["initial_ball_vel_w"],
      "initial_plate_pos_w": data["initial_plate_pos_w"],
      "initial_plate_axes_w": data["initial_plate_axes_w"],
      "env_origins": data["env_origins"],
    }


def _arm_row(name: str, batch_dir: Path) -> dict[str, Any]:
  """Collect one arm's headline numbers, tolerating a missing or failed arm."""
  arm_dir = batch_dir / "arms" / name
  metadata = _load_json(arm_dir / "arm_metadata.json")
  summary = _load_json(arm_dir / "arm_summary.json")
  row: dict[str, Any] = {
    "arm": name,
    "directory": str(arm_dir),
    "metadata_present": metadata is not None,
    "status": (metadata or {}).get("status", "missing"),
    "mode": (metadata or {}).get("mode"),
    "target_m": (metadata or {}).get("target_m"),
    "elapsed_s": (metadata or {}).get("elapsed_s"),
    "error": (metadata or {}).get("error"),
  }
  if summary is None:
    row["summary_present"] = False
    return row
  row.update(
    {
      "summary_present": True,
      "episodes": summary["episodes"],
      "timed_out": summary["timed_out"],
      "terminated": summary["terminated"],
      "losses": summary["losses"],
      "falls": summary["falls"],
      "episodes_with_hold": summary["episodes_with_hold"],
      "qualified_flights_total": summary["qualified_flights_total"],
      "episodes_with_qualified_flight": summary["episodes_with_qualified_flight"],
      "longest_qualified_sequence_max": summary["longest_qualified_sequence"]["max"],
      "longest_qualified_sequence_median": summary["longest_qualified_sequence"][
        "median"
      ],
      "episodes_reaching_min_sequence": summary["episodes_reaching_min_sequence"],
      "episodes_passing_gate": summary["episodes_passing_gate"],
      "gate_episode_count_matches_plan": summary["gate_episode_count_matches_plan"],
      "gate_passed": summary["gate_passed"],
      "rejected_by_criterion": summary["rejected_by_criterion"],
      "termination_terms": summary["termination_terms"],
    }
  )
  return row


def summarize(batch_dir: Path) -> dict[str, Any]:
  """Build the batch report and return it."""
  import numpy as np
  from scoring import compare_matched_initial_states

  arms = [_arm_row(name, batch_dir) for name in ARM_ORDER]
  states = {
    name: state
    for name in ARM_ORDER
    if (state := _initial_states(batch_dir / "arms" / name / "trace_raw.npz"))
    is not None
  }
  matched = compare_matched_initial_states(states) if states else {"matched": False}

  per_target: dict[str, Any] = {}
  for target in ("t020", "t040"):
    names = [name for name in ARM_ORDER if name.startswith(target)]
    rows = [row for row in arms if row["arm"] in names]
    evaluated = [row for row in rows if row.get("summary_present")]
    per_target[target] = {
      "arms": names,
      "evaluated": len(evaluated),
      "gate_passed_every_evaluated_arm": bool(evaluated)
      and all(row["gate_passed"] for row in evaluated),
      "episodes_passing_gate_per_arm": {
        row["arm"]: row.get("episodes_passing_gate") for row in evaluated
      },
      "all_arms_evaluated": len(evaluated) == len(names),
    }

  controls = [row for row in arms if str(row["arm"]).startswith("zero-")]
  report = {
    "batch_dir": str(batch_dir),
    "arms": arms,
    "release_states": matched,
    "per_target": per_target,
    "zero_action_gate_passed": {
      row["arm"]: row.get("gate_passed")
      for row in controls
      if row.get("summary_present")
    },
    "all_arms_finished": all(
      row["status"] == "ok" and row.get("summary_present") for row in arms
    ),
    "max_abs_initial_ball_pos_deviation_m": float(
      matched.get("max_abs_deviation", {}).get("initial_ball_pos_w", float("nan"))
    )
    if states
    else None,
    "not_reported": [
      "hardware behaviour: no robot was used",
      "policy reliability: 32 reset samples are not 32 independent policies",
      "sim-to-real agreement: not measured by this batch",
    ],
    "np_version": np.__version__,
  }
  return report


def _print_table(report: dict[str, Any]) -> None:
  header = (
    f"{'arm':10s} {'mode':11s} {'tgt':>5s} {'eps':>4s} {'qf':>5s} {'maxseq':>6s} "
    f"{'medseq':>6s} {'>=10':>5s} {'hold':>4s} {'loss':>4s} {'fall':>4s} "
    f"{'gate':>5s} {'s':>6s}"
  )
  print(header)
  print("-" * len(header))
  for row in report["arms"]:
    if not row.get("summary_present"):
      print(
        f"{row['arm']:10s} {(row['mode'] or '?'):11s} {'':>5s} {'':>4s} {'':>5s} "
        f"{'':>6s} {'':>6s} {'':>5s} {'':>4s} {'':>4s} {'':>4s} "
        f"{'FAIL':>5s} {str(row['elapsed_s'] or '-'):>6s}"
      )
      continue
    print(
      f"{row['arm']:10s} {str(row['mode']):11s} {row['target_m']:5.2f} "
      f"{row['episodes']:4d} {row['qualified_flights_total']:5d} "
      f"{row['longest_qualified_sequence_max']:6.0f} "
      f"{row['longest_qualified_sequence_median']:6.1f} "
      f"{row['episodes_reaching_min_sequence']:5d} "
      f"{row['episodes_with_hold']:4d} {row['losses']:4d} {row['falls']:4d} "
      f"{('yes' if row['gate_passed'] else 'no'):>5s} "
      f"{(row['elapsed_s'] or 0.0):6.1f}"
    )
  states = report["release_states"]
  print(
    "\nrelease states matched across arms: "
    f"{states.get('matched')} ({json.dumps(states.get('max_abs_deviation', {}))})"
  )
  print(f"all arms finished: {report['all_arms_finished']}")


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  report = summarize(args.batch_dir)
  path = args.batch_dir / "batch_summary.json"
  path.write_text(json.dumps(report, indent=2, default=str) + "\n")
  _print_table(report)
  print(f"\nwrote {path}")
  if not report["all_arms_finished"]:
    return EXIT_ARM_FAILED
  if not report["release_states"].get("matched", False):
    return EXIT_STATE_MISMATCH
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
