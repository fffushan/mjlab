#!/usr/bin/env python3
"""Generate a matched D1 collection benchmark matrix; never run it.

The matrix varies exactly two engineering dimensions:

* VAE policy device: CPU (``cpu``) and GPU (``cuda:0``);
* environment count: 64, 128 and 256.

Every arm shares the same contract, artifact root, collection config, runtime
factory, trial budget, motion seeds and streaming flags, so the arms are
matched.  The tool writes a JSON plan and a shell script with one command per
arm.  It does not build a runtime, does not touch a GPU and does not run a
simulator.  Every arm writes to a new output directory under the plan's
``--stamp`` and refuses an existing output.

The inputs are read-only references.  The tool never copies or mutates them.

Run the generated commands only with separate authorization.  A measured
speedup needs the same machine, a warm page cache and a matched baseline; this
tool only produces the commands.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

DEVICES = ("cpu", "cuda:0")
ENV_COUNTS = (64, 128, 256)


def _slug(device: str) -> str:
  return device.replace(":", "").replace("/", "")


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
  """Build the matched arm matrix as plain data, without importing mjlab."""
  output_root = Path(args.output_root)
  if args.bulk_seeds < 1 or not args.teachers.strip():
    raise ValueError("bulk seeds and explicit teacher selection must be nonempty")
  arms: list[dict[str, Any]] = []
  for device in DEVICES:
    for envs in ENV_COUNTS:
      arm = f"vae-{_slug(device)}-envs-{envs}"
      arm_output = output_root / args.stamp / arm
      if arm_output.exists():
        raise ValueError(f"refusing to overwrite existing output {arm_output}")
      argv = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "mjlab.scripts.diffusion",
        "collect",
        "--contract",
        str(args.contract),
        "--artifact-root",
        str(args.artifact_root),
        "--config",
        str(args.config),
        "--execute",
        "--runtime-factory",
        args.runtime_factory,
        "--output",
        str(arm_output / "store"),
        "--report",
        str(arm_output / "report.json"),
        "--progress",
        str(arm_output / "progress.jsonl"),
        "--max-trials",
        str(args.max_trials),
        "--max-envs",
        str(envs),
        "--max-gpus",
        "1",
        "--max-wall-minutes",
        str(args.max_wall_minutes),
        "--monitor-inactivity-seconds",
        str(args.monitor_inactivity_seconds),
        "--max-output-mib",
        str(args.max_output_mib),
        "--policy-device",
        device,
        "--stream-rows",
        "--compact-results",
        "--max-rows-per-append",
        str(args.max_rows_per_append),
      ]
      if args.tensor_dispatch:
        argv.append("--tensor-dispatch")
      arms.append(
        {
          "arm": arm,
          "vae_policy_device": device,
          "max_envs": envs,
          "output": str(arm_output),
          "argv": argv,
          "env": {
            "MJLAB_D1_POLICY_DEVICE": device,
            "MJLAB_D1_SCHEDULE": args.schedule,
            "MJLAB_D1_BULK_SEEDS": str(args.bulk_seeds),
            "MJLAB_D1_COHORT_MANIFEST": str(args.manifest),
            "MJLAB_D1_ROLLOUT_TEACHERS": args.teachers,
            "MJLAB_D1_WORKERS": "1",
            "MJLAB_D1_WORKER": "0",
            "MJLAB_D1_WORKER_OFFSET": "0",
            "MJLAB_D1_RUN_ID": f"d1-benchmark-{args.stamp}-{arm}",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
          },
        }
      )
  return {
    "schema": "mjlab-x2-diffusion-d1-benchmark-matrix-v1",
    "stamp": args.stamp,
    "varied_dimensions": {
      "vae_policy_device": list(DEVICES),
      "max_envs": list(ENV_COUNTS),
    },
    "shared": {
      "contract": str(args.contract),
      "artifact_root": str(args.artifact_root),
      "config": str(args.config),
      "runtime_factory": args.runtime_factory,
      "max_trials": args.max_trials,
      "max_wall_minutes": args.max_wall_minutes,
      "monitor_inactivity_seconds": args.monitor_inactivity_seconds,
      "max_output_mib": args.max_output_mib,
      "max_rows_per_append": args.max_rows_per_append,
      "tensor_dispatch": args.tensor_dispatch,
      "schedule": args.schedule,
      "bulk_seeds": args.bulk_seeds,
      "manifest": str(args.manifest),
      "teachers": args.teachers,
    },
    "read_only_inputs": [str(args.contract), str(args.artifact_root), str(args.config)],
    "new_outputs": [arm["output"] for arm in arms],
    "arms": arms,
    "note": (
      "generated commands only; this tool did not run a benchmark, a simulator "
      "or a GPU job"
    ),
  }


def render_shell(plan: dict[str, Any]) -> str:
  """Render one shell command per arm, with its environment prefix."""
  lines = [
    "#!/usr/bin/env bash",
    "# Generated by benchmark_matrix.py.  Review and authorize before running.",
    "set -euo pipefail",
    "",
  ]
  for arm in plan["arms"]:
    env_prefix = " ".join(
      f"{key}={shlex.quote(value)}" for key, value in sorted(arm["env"].items())
    )
    command = " ".join(shlex.quote(part) for part in arm["argv"])
    lines.append(f"# {arm['arm']}")
    lines.append(f"mkdir -p {shlex.quote(str(arm['output']))}")
    lines.extend(
      [
        f"if {env_prefix} {command}; then",
        "  arm_status=0",
        "else",
        "  arm_status=$?",
        "fi",
        f"printf '%s\\n' \"$arm_status\" > {shlex.quote(str(Path(arm['output']) / 'exit.status'))}",
        f"printf 'arm %s exited %s\\n' {shlex.quote(arm['arm'])} \"$arm_status\" >&2",
      ]
    )
    lines.append("")
  return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
    prog="benchmark_matrix",
    description="generate a matched D1 collection benchmark matrix (never run it)",
  )
  parser.add_argument(
    "--contract",
    type=Path,
    default=Path("docs/plans/beyondmimic_diffusion_d1_no_mixtures_vgk8_contract.yaml"),
  )
  parser.add_argument("--artifact-root", type=Path, default=Path("."))
  parser.add_argument(
    "--config",
    type=Path,
    default=Path("configs/diffusion/x2_50hz_collection_vgk8.yaml"),
  )
  parser.add_argument(
    "--runtime-factory",
    default="mjlab.tasks.tracking.diffusion.runtime_x2:build_x2_runtime",
  )
  parser.add_argument(
    "--schedule", default="bulk", help="MJLAB_D1_SCHEDULE value for every arm"
  )
  parser.add_argument(
    "--output-root",
    type=Path,
    default=Path("logs/diffusion/d1-vgk8-throughput-benchmark"),
  )
  parser.add_argument(
    "--stamp", required=True, help="one new timestamp shared by every arm"
  )
  parser.add_argument("--bulk-seeds", type=int, default=1)
  parser.add_argument(
    "--manifest",
    type=Path,
    default=Path("configs/distillation/x2_tennis_vendor_gains_dr_waist_kd4_v1.yaml"),
  )
  parser.add_argument(
    "--teachers",
    default=",".join(f"vgk{i:03d}" for i in range(8)),
    help="explicit reference cohort selection; align with the selected contract",
  )
  parser.add_argument("--max-trials", type=int, default=48)
  parser.add_argument("--max-wall-minutes", type=int, default=12)
  parser.add_argument("--monitor-inactivity-seconds", type=int, default=180)
  parser.add_argument("--max-output-mib", type=int, default=1024)
  parser.add_argument("--max-rows-per-append", type=int, default=512)
  parser.add_argument(
    "--tensor-dispatch",
    action="store_true",
    help="add the device tensor dispatch to every arm (matched)",
  )
  parser.add_argument(
    "--plan-dir",
    type=Path,
    required=True,
    help="new directory that receives plan.json and commands.sh",
  )
  args = parser.parse_args(argv)

  plan_dir = Path(args.plan_dir)
  if plan_dir.exists():
    print(
      f"benchmark_matrix: plan directory already exists: {plan_dir}", file=sys.stderr
    )
    return 2
  try:
    plan = build_plan(args)
  except ValueError as exc:
    print(f"benchmark_matrix: {exc}", file=sys.stderr)
    return 2
  plan_dir.mkdir(parents=True)
  (plan_dir / "plan.json").write_text(
    json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  commands = plan_dir / "commands.sh"
  commands.write_text(render_shell(plan), encoding="utf-8")
  commands.chmod(0o755)
  print(json.dumps(plan, indent=2, sort_keys=True))
  print(
    f"benchmark_matrix: wrote {len(plan['arms'])} commands to {commands}. "
    "No benchmark was run.",
    file=sys.stderr,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
