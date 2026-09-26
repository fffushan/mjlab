"""Distillation CLI.

M1 ships validation only: ``distill validate-teachers`` loads the selected
teacher cohort on CPU and checks native inference against the original ONNX
exports. No training command is provided yet.
"""

import json
import sys
from pathlib import Path

import tyro

import mjlab
from mjlab.tasks.tracking.distillation.config import (
  DistillationError,
  MissingValidationDependencyError,
  load_manifest,
  resolve_cohort,
)
from mjlab.tasks.tracking.distillation.parity import (
  DEFAULT_ATOL,
  DEFAULT_RTOL,
  DEFAULT_SAMPLES,
  DEFAULT_SEED,
  validate_teachers,
)

_COMMANDS = ("validate-teachers",)


def _print_help(stream) -> None:
  print("usage: distill <COMMAND> [OPTIONS]", file=stream)
  print(file=stream)
  print("Commands:", file=stream)
  print(
    "  validate-teachers  Validate a teacher manifest and check native/ONNX parity.",
    file=stream,
  )
  print(file=stream)
  print(
    "Run 'distill validate-teachers --help' for command-specific options.",
    file=stream,
  )


def _validate_teachers(
  manifest: Path = Path("configs/distillation/x2_tennis.yaml"),
  repo_root: Path = Path("."),
  samples: int = DEFAULT_SAMPLES,
  seed: int = DEFAULT_SEED,
  atol: float = DEFAULT_ATOL,
  rtol: float = DEFAULT_RTOL,
  report: Path | None = None,
) -> int:
  """Validate a teacher cohort and compare native inference with original ONNX exports.

  Relative manifest paths resolve against ``repo_root``. The JSON report is
  printed to stdout and, when ``report`` is given, written to that path. The
  command exits nonzero when any check fails.
  """
  try:
    cohort = resolve_cohort(load_manifest(manifest, repo_root))
    result = validate_teachers(cohort, samples=samples, seed=seed, atol=atol, rtol=rtol)
  except (DistillationError, MissingValidationDependencyError) as exc:
    print(f"[FAIL] {exc}", file=sys.stderr)
    return 1

  result["command"] = " ".join(sys.argv)
  payload = json.dumps(result, indent=2)
  print(payload)
  if report is not None:
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(payload + "\n")

  actor = result["actor"]
  control = result["control"]
  passed_teachers = sum(1 for teacher in result["teachers"] if teacher["passed"])
  print(
    f"cohort {result['manifest']['name']}: {passed_teachers}/{len(result['teachers'])} "
    f"teachers passed the parity gate (obs {actor['obs_dim']}, "
    f"actions {actor['action_dim']}, {control['control_hz']:.3f} Hz)",
    file=sys.stderr,
  )
  for teacher in result["teachers"]:
    parity = teacher["parity"]
    association = teacher["checkpoint_onnx_association"]
    print(
      f"  {teacher['id']}: parity max|delta|={parity['max_abs_error']:.3g} "
      f"(atol={parity['atol']:g}, rtol={parity['rtol']:g}, n={parity['samples']}), "
      f"association max|delta|={association['max_abs_diff']:.3g}, "
      f"reference exact="
      f"{all(item['passed'] for item in teacher['embedded_reference'])} -> "
      f"{'PASS' if teacher['passed'] else 'FAIL'}",
      file=sys.stderr,
    )
  for note in result["unverified"]:
    print(f"  unverified: {note}", file=sys.stderr)
  print(
    f"[{'OK' if result['passed'] else 'FAIL'}] {passed_teachers}/"
    f"{len(result['teachers'])} teachers passed",
    file=sys.stderr,
  )
  return 0 if result["passed"] else 1


def main() -> None:
  if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    _print_help(sys.stdout)
    return
  command = sys.argv[1]
  if command not in _COMMANDS:
    print(f"distill: unknown command {command!r}", file=sys.stderr)
    _print_help(sys.stderr)
    sys.exit(2)
  raise SystemExit(
    tyro.cli(
      _validate_teachers,
      args=sys.argv[2:],
      prog=f"{Path(sys.argv[0]).name} {command}",
      config=mjlab.TYRO_FLAGS,
    )
  )


if __name__ == "__main__":
  main()
