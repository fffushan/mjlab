#!/usr/bin/env python3
"""Preregistered serial headless pilot driver for the D3 matched-pair study.

The driver runs exactly the six preregistered ``tennis_000`` conditions
(``seeds 0/1/2`` x ``start fractions 0.1/0.5``) through the reviewed single-pair
evaluator ``evaluate.py``.  The plan is fixed before any physics runs:

* eight actual VAE bootstrap transitions per branch (baseline and diffusion);
* at most ``250`` post-bootstrap evaluation ticks per branch, shortened
  deterministically to ``min(250, clip_frames - 2 - start_frame - 8)`` so
  neither branch attempts a final-reference-frame action or clip wrap;
* one local GPU environment at a time, headless, 0.02 s simulated step;
* one serial batch with a whole-batch wall cap of ``1200`` s and a whole-batch
  output cap of ``128`` MiB (``16`` MiB per condition).

The driver never retries, retunes, reseeds or expands the matrix.  A physical
fall, reference boundary, bounded budget stop, pair mismatch or physical
bootstrap failure is an attempted outcome that is retained and does not stop
later preregistered conditions.  A software, integrity, output or
untrusted-runtime failure stops the batch and preserves the partial artifacts.
A condition is killed only when the whole-batch wall cap would otherwise be
exceeded; the driver records that as ``wall_timeout``.

Without ``--execute`` this is a simulator-free preflight: it verifies the frozen
hash gates, loads the cohort manifest, prints the preregistered plan and
constructs no environment.  This tool never builds a simulator, policy, viewer,
controller or training job itself.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mjlab.tasks.tracking.diffusion.replay_integration import (
  ReplayError,
  sha256_file,
  verify_file_digest,
)
from mjlab.tasks.tracking.distillation.config import (
  DistillationError,
  load_manifest,
  resolve_cohort,
)

_HERE = Path(__file__).resolve().parent
_EVALUATOR = _HERE / "evaluate.py"

MEMBER = "tennis_000"
SEEDS = (0, 1, 2)
START_FRACTIONS = (0.1, 0.5)
BOOTSTRAP_PAIRS = 8
MAX_CONTINUATION_STEPS = 250
MAX_WALL_SECONDS = 1200.0
MAX_OUTPUT_BYTES = 128 * 1024 * 1024
CONDITION_OUTPUT_BYTES = 16 * 1024 * 1024
MIN_OUTPUT_CAP_BYTES = 4096
STDERR_CAP_BYTES = 65536
_EXPECTED_CLIP_FPS = 50.0
_CLIP_FPS_ABS_TOL = 1.0e-5

_HORIZON_FORMULA = (
  "min(250, clip_frames - 2 - start_frame - 8); start_frame = "
  "floor(start_fraction * (clip_frames - 1))"
)

# Stop reasons that mean the software or its recorded evidence is untrusted, as
# opposed to a bounded physical outcome.  Any of these stops the batch.
_SOFTWARE_STOP_REASONS = frozenset(
  {
    "runtime_error",
    "inference_error",
    "history_record_error",
    "nonfinite",
    "reset",
    "teleport",
    "missing_terminal_evidence",
    "segment_change",
    "tick_discontinuity",
    "period_discontinuity",
    "reference_leakage",
    "handover_state_changed",
    "handover_clock_changed",
  }
)

# Bounded stops that are expected physical outcomes of the frozen budget.  A
# branch that fell on its very first tick has ``closed_loop=False`` (no accepted
# step) but is still a physical outcome, so a fall is never treated as an
# untrusted runtime.
_BOUNDED_STOP_REASONS = frozenset(
  {
    "step_budget",
    "time_budget",
    "reference_boundary",
    "terminated",
    "timeout",
  }
)


class PilotError(ValueError):
  """A pilot request, frozen artifact or recorded batch is invalid."""


@dataclass(frozen=True, slots=True)
class ConditionSpec:
  """One preregistered matched-pair condition."""

  index: int
  seed: int
  start_fraction: float
  start_frame: int
  continuation_steps: int


@dataclass(frozen=True, slots=True)
class ConditionExecution:
  """One evaluator subprocess outcome, measured by the driver."""

  returncode: int | None
  timed_out: bool
  elapsed_s: float
  stderr_bytes: int
  stderr_truncated: bool


def safe_continuation_steps(clip_frames: int, start_frame: int) -> int:
  """Return the baseline-boundary-safe available horizon.

  The formula is declared before physics and is never adjusted from outcomes.
  The last continuation post-step frame is ``clip_frames - 2``, one below the
  runtime's final-reference-frame boundary.
  """
  return min(MAX_CONTINUATION_STEPS, clip_frames - 2 - start_frame - BOOTSTRAP_PAIRS)


def build_conditions(clip_frames: int) -> tuple[ConditionSpec, ...]:
  """Build the frozen seed x start-fraction matrix for one clip length."""
  if isinstance(clip_frames, bool) or not isinstance(clip_frames, int):
    raise PilotError("clip_frames must be an integer")
  if clip_frames < BOOTSTRAP_PAIRS + 4:
    raise PilotError(f"clip_frames {clip_frames} is too short for the pilot")
  conditions: list[ConditionSpec] = []
  for seed in SEEDS:
    for fraction in START_FRACTIONS:
      start_frame = math.floor(fraction * (clip_frames - 1))
      steps = safe_continuation_steps(clip_frames, start_frame)
      if steps < 1:
        raise PilotError(
          f"condition seed={seed} fraction={fraction} has no safe horizon"
        )
      conditions.append(
        ConditionSpec(
          index=len(conditions),
          seed=seed,
          start_fraction=fraction,
          start_frame=start_frame,
          continuation_steps=steps,
        )
      )
  return tuple(conditions)


def condition_dirname(spec: ConditionSpec) -> str:
  return f"seed{spec.seed}-start{spec.start_frame:04d}-frac{spec.start_fraction:g}"


def read_condition_summary(condition_dir: Path) -> dict[str, Any] | None:
  """Read the persisted evaluator summary, or ``None`` when unavailable."""
  try:
    payload = json.loads((condition_dir / "summary.json").read_text("utf-8"))
  except (OSError, json.JSONDecodeError, ValueError):
    return None
  return payload if isinstance(payload, dict) else None


def measure_output_bytes(root: Path) -> int:
  """Sum the actual bytes of every file written under ``root``."""
  total = 0
  if not root.exists():
    return 0
  for path in root.rglob("*"):
    if not path.is_file():
      continue
    try:
      total += path.stat().st_size
    except OSError:
      continue
  return total


def _branch_stop_reason(summary: Mapping[str, Any], name: str) -> str | None:
  branches = summary.get("branches")
  if not isinstance(branches, Mapping):
    return None
  branch = branches.get(name)
  if not isinstance(branch, Mapping):
    return None
  reason = branch.get("stop_reason")
  return reason if isinstance(reason, str) else None


def _evidence(
  summary: Mapping[str, Any],
) -> tuple[bool | None, bool | None, object]:
  """Return integrity/budget/output evidence from full, compact or marker JSON."""
  provenance: Any = summary.get("provenance")
  if not isinstance(provenance, Mapping):
    after_run = summary.get("after_run")
    provenance = after_run if isinstance(after_run, Mapping) else {}
  integrity = provenance.get("integrity")
  unchanged = integrity.get("unchanged") if isinstance(integrity, Mapping) else None
  execution = provenance.get("execution")
  within_wall = (
    execution.get("within_wall_budget") if isinstance(execution, Mapping) else None
  )
  output = summary.get("output")
  error = output.get("error") if isinstance(output, Mapping) else None
  return (
    unchanged if isinstance(unchanged, bool) else None,
    within_wall if isinstance(within_wall, bool) else None,
    error,
  )


def classify_condition(
  *, timed_out: bool, summary: Mapping[str, Any] | None
) -> tuple[str, str]:
  """Return ``(classification, batch_action)`` for one recorded condition.

  ``batch_action`` is ``"stop"`` only for software, integrity, output,
  untrusted-runtime or budget failures; bounded physical outcomes continue.
  """
  if timed_out:
    return "wall_timeout", "stop"
  if not isinstance(summary, Mapping):
    return "unreadable_summary", "stop"
  unchanged, within_wall, output_error = _evidence(summary)
  if output_error:
    return "output_error", "stop"
  if unchanged is not True:
    return "integrity_unverified_or_changed", "stop"
  if within_wall is False:
    return "per_condition_wall_budget_exceeded", "stop"
  reasons = {
    reason
    for reason in (
      _branch_stop_reason(summary, "baseline"),
      _branch_stop_reason(summary, "diffusion"),
    )
    if reason
  }
  software = sorted(reasons & _SOFTWARE_STOP_REASONS)
  if software:
    return f"software_failure:{software[0]}", "stop"
  if "physical_fall" in reasons:
    return "physical_fall", "continue"
  trusted = summary.get("closed_loop") is True
  if "pair_mismatch" in reasons:
    if not trusted:
      return "untrusted_runtime", "stop"
    return "pair_mismatch", "continue"
  if reasons and reasons <= _BOUNDED_STOP_REASONS:
    if not trusted:
      return "untrusted_runtime", "stop"
    if summary.get("paired") is True:
      return "paired_complete", "continue"
    return "bounded_stop", "continue"
  if not trusted:
    return "untrusted_runtime", "stop"
  if summary.get("paired") is True:
    return "paired_complete", "continue"
  if summary.get("status") == "partial":
    return "unpaired_partial", "continue"
  return "unpaired", "continue"


def run_evaluator_subprocess(
  command: Sequence[str],
  *,
  cwd: Path,
  timeout_s: float,
  stderr_path: Path,
  env: Mapping[str, str] | None = None,
) -> ConditionExecution:
  """Run one evaluator command, discarding stdout and bounding stderr."""
  started = time.monotonic()
  process = subprocess.Popen(
    list(command),
    cwd=str(cwd),
    env=None if env is None else dict(env),
    stdout=subprocess.DEVNULL,
    stderr=subprocess.PIPE,
  )
  timed_out = False
  try:
    _, raw_stderr = process.communicate(timeout=timeout_s)
    returncode: int | None = process.returncode
  except subprocess.TimeoutExpired:
    process.kill()
    _, raw_stderr = process.communicate()
    returncode = None
    timed_out = True
  elapsed = time.monotonic() - started
  stderr = bytes(raw_stderr or b"")
  stderr_path.parent.mkdir(parents=True, exist_ok=True)
  stderr_path.write_bytes(stderr[:STDERR_CAP_BYTES])
  return ConditionExecution(
    returncode=returncode,
    timed_out=timed_out,
    elapsed_s=elapsed,
    stderr_bytes=len(stderr),
    stderr_truncated=len(stderr) > STDERR_CAP_BYTES,
  )


def run_batch(
  conditions: Sequence[ConditionSpec],
  *,
  output_root: Path,
  run_condition: Callable[[ConditionSpec, Path, float, int], ConditionExecution],
  max_wall_seconds: float = MAX_WALL_SECONDS,
  max_output_bytes: int = MAX_OUTPUT_BYTES,
  condition_output_bytes: int = CONDITION_OUTPUT_BYTES,
  clock: Callable[[], float] = time.monotonic,
  measure_bytes: Callable[[Path], int] = measure_output_bytes,
  progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
  """Run the preregistered conditions serially under both batch caps."""
  emit = progress if progress is not None else (lambda _event: None)
  batch_start = clock()
  records: list[dict[str, Any]] = []
  stop_reason: str | None = None
  for spec in conditions:
    wall_elapsed = clock() - batch_start
    remaining_wall = max_wall_seconds - wall_elapsed
    used_bytes = measure_bytes(output_root)
    remaining_output = max_output_bytes - used_bytes
    if remaining_wall <= 0.0:
      stop_reason = "wall_budget_exhausted"
      break
    if remaining_output < MIN_OUTPUT_CAP_BYTES:
      stop_reason = "output_budget_exhausted"
      break
    condition_dir = output_root / condition_dirname(spec)
    if condition_dir.exists():
      raise PilotError(f"refusing to reuse condition directory {condition_dir}")
    output_cap = min(condition_output_bytes, remaining_output)
    emit(
      {
        "event": "condition_start",
        "index": spec.index,
        "seed": spec.seed,
        "start_fraction": spec.start_fraction,
        "start_frame": spec.start_frame,
        "continuation_steps": spec.continuation_steps,
        "batch_wall_elapsed_s": wall_elapsed,
        "remaining_wall_s": remaining_wall,
        "output_bytes": used_bytes,
        "remaining_output_bytes": remaining_output,
      }
    )
    execution = run_condition(spec, condition_dir, remaining_wall, output_cap)
    summary = read_condition_summary(condition_dir)
    classification, action = classify_condition(
      timed_out=execution.timed_out, summary=summary
    )
    record = {
      "index": spec.index,
      "seed": spec.seed,
      "start_fraction": spec.start_fraction,
      "start_frame": spec.start_frame,
      "continuation_steps": spec.continuation_steps,
      "condition_dir": str(condition_dir),
      "returncode": execution.returncode,
      "timed_out": execution.timed_out,
      "elapsed_s": execution.elapsed_s,
      "stderr_bytes": execution.stderr_bytes,
      "stderr_truncated": execution.stderr_truncated,
      "output_bytes": measure_bytes(output_root),
      "classification": classification,
      "batch_action": action,
      "status": None if summary is None else summary.get("status"),
      "paired": None if summary is None else summary.get("paired"),
      "baseline_stop_reason": (
        None if summary is None else _branch_stop_reason(summary, "baseline")
      ),
      "diffusion_stop_reason": (
        None if summary is None else _branch_stop_reason(summary, "diffusion")
      ),
    }
    records.append(record)
    emit({"event": "condition_end", **record})
    if action == "stop":
      stop_reason = classification
      break
  result = {
    "schema": "mjlab-d3-recorded-pilot-batch-v1",
    "conditions_planned": len(conditions),
    "conditions_attempted": len(records),
    "stop_reason": stop_reason,
    "wall_seconds": clock() - batch_start,
    "max_wall_seconds": max_wall_seconds,
    "output_bytes": measure_bytes(output_root),
    "max_output_bytes": max_output_bytes,
    "records": records,
  }
  emit(
    {
      "event": "batch_end",
      "conditions_planned": result["conditions_planned"],
      "conditions_attempted": result["conditions_attempted"],
      "stop_reason": stop_reason,
      "wall_seconds": result["wall_seconds"],
      "output_bytes": result["output_bytes"],
    }
  )
  return result


@dataclass(frozen=True, slots=True)
class PilotPlan:
  """Preregistered matrix plus the exact frozen assets it will run against."""

  repo_root: Path
  contract: Path
  checkpoint: Path
  projection: Path
  vae_checkpoint: Path
  manifest: Path
  assets: Mapping[str, Mapping[str, str]]
  clip_frames: int
  clip_fps: float
  conditions: tuple[ConditionSpec, ...]
  device: str
  max_wall_seconds: float
  max_output_bytes: int

  def condition_command(
    self, spec: ConditionSpec, condition_dir: Path, output_cap_bytes: int
  ) -> list[str]:
    return [
      sys.executable,
      str(_EVALUATOR),
      "--repo-root",
      str(self.repo_root),
      "--contract",
      str(self.contract),
      "--contract-sha256",
      self.assets["contract"]["sha256"],
      "--checkpoint",
      str(self.checkpoint),
      "--checkpoint-sha256",
      self.assets["diffusion_checkpoint"]["sha256"],
      "--projection",
      str(self.projection),
      "--projection-sha256",
      self.assets["projection"]["sha256"],
      "--vae-checkpoint",
      str(self.vae_checkpoint),
      "--vae-checkpoint-sha256",
      self.assets["vae_checkpoint"]["sha256"],
      "--manifest",
      str(self.manifest),
      "--manifest-sha256",
      self.assets["vae_manifest"]["sha256"],
      "--member",
      MEMBER,
      "--seed",
      str(spec.seed),
      "--start-frame",
      str(spec.start_frame),
      "--continuation-steps",
      str(spec.continuation_steps),
      "--max-wall-seconds",
      str(self.max_wall_seconds),
      "--max-output-bytes",
      str(output_cap_bytes),
      "--output-dir",
      str(condition_dir),
      "--device",
      self.device,
      "--execute",
    ]

  def as_manifest(self) -> dict[str, Any]:
    return {
      "schema": "mjlab-d3-recorded-pilot-manifest-v1",
      "member": MEMBER,
      "clip_frames": self.clip_frames,
      "clip_fps": self.clip_fps,
      "seeds": list(SEEDS),
      "start_fractions": list(START_FRACTIONS),
      "bootstrap_pairs": BOOTSTRAP_PAIRS,
      "max_continuation_steps": MAX_CONTINUATION_STEPS,
      "horizon_formula": _HORIZON_FORMULA,
      "conditions": [
        {
          "index": spec.index,
          "seed": spec.seed,
          "start_fraction": spec.start_fraction,
          "start_frame": spec.start_frame,
          "continuation_steps": spec.continuation_steps,
        }
        for spec in self.conditions
      ],
      "device": self.device,
      "max_wall_seconds": self.max_wall_seconds,
      "max_output_bytes": self.max_output_bytes,
      "condition_output_bytes": CONDITION_OUTPUT_BYTES,
      "frozen_assets": {
        role: {"path": str(entry["path"]), "sha256": entry["sha256"]}
        for role, entry in self.assets.items()
      },
      "source": {
        "driver": {
          "path": str(_HERE / "run_pilot.py"),
          "sha256": sha256_file(_HERE / "run_pilot.py"),
        },
        "evaluator": {"path": str(_EVALUATOR), "sha256": sha256_file(_EVALUATOR)},
      },
      "frozen_preserved": {
        "token_channels": 167,
        "ddim_updates": 20,
        "start_level": 1000,
        "eta": 0.0,
        "guidance_strength": 0.0,
        "precision": "fp32",
        "physical_fall_degrees": 70.0,
      },
      "retention": (
        "attempted/partial/failed outcomes retained; no retry, retuning, extra "
        "seed/member or follow-up expansion"
      ),
    }


def _resolve_asset(role: str, path: Path, expected_sha256: str) -> dict[str, str]:
  try:
    verify_file_digest(path, expected_sha256, role=role)
  except (ReplayError, OSError, ValueError) as exc:
    raise PilotError(f"frozen {role} gate failed: {exc}") from exc
  return {"path": str(path), "sha256": expected_sha256}


def build_plan(args: argparse.Namespace) -> PilotPlan:
  """Verify the frozen gates and declare the preregistered matrix."""
  repo_root = args.repo_root.resolve()
  if not repo_root.is_dir():
    raise PilotError(f"repo-root is not a directory: {repo_root}")
  contract = args.contract.resolve()
  checkpoint = args.checkpoint.resolve()
  projection = args.projection.resolve()
  vae_checkpoint = args.vae_checkpoint.resolve()
  manifest_path = args.manifest.resolve()
  assets = {
    "contract": _resolve_asset("contract", contract, args.contract_sha256),
    "diffusion_checkpoint": _resolve_asset(
      "diffusion_checkpoint", checkpoint, args.checkpoint_sha256
    ),
    "projection": _resolve_asset("projection", projection, args.projection_sha256),
    "vae_checkpoint": _resolve_asset(
      "vae_checkpoint", vae_checkpoint, args.vae_checkpoint_sha256
    ),
    "vae_manifest": _resolve_asset("vae_manifest", manifest_path, args.manifest_sha256),
  }
  try:
    manifest = load_manifest(manifest_path, repo_root=repo_root)
    cohort = resolve_cohort(manifest)
    teacher = cohort.teacher(MEMBER)
  except (DistillationError, OSError, ValueError) as exc:
    raise PilotError(f"cohort preflight failed: {exc}") from exc
  clip_frames = int(teacher.reference.frames)
  clip_fps = float(teacher.reference.fps)
  if not math.isclose(
    clip_fps, _EXPECTED_CLIP_FPS, abs_tol=_CLIP_FPS_ABS_TOL, rel_tol=0.0
  ):
    raise PilotError(f"member {MEMBER} must be 50 Hz, got {clip_fps}")
  conditions = build_conditions(clip_frames)
  if not _EVALUATOR.is_file():
    raise PilotError(f"evaluator is not a readable file: {_EVALUATOR}")
  return PilotPlan(
    repo_root=repo_root,
    contract=contract,
    checkpoint=checkpoint,
    projection=projection,
    vae_checkpoint=vae_checkpoint,
    manifest=manifest_path,
    assets=assets,
    clip_frames=clip_frames,
    clip_fps=clip_fps,
    conditions=conditions,
    device=args.device,
    max_wall_seconds=float(args.max_wall_seconds),
    max_output_bytes=int(args.max_output_bytes),
  )


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description=(
      "Run the preregistered serial D3 tennis_000 pilot (6 matched pairs) "
      "through evaluate.py, or print its simulator-free plan."
    )
  )
  parser.add_argument("--contract", type=Path, required=True)
  parser.add_argument("--contract-sha256", required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--checkpoint-sha256", required=True)
  parser.add_argument("--projection", type=Path, required=True)
  parser.add_argument("--projection-sha256", required=True)
  parser.add_argument("--vae-checkpoint", type=Path, required=True)
  parser.add_argument("--vae-checkpoint-sha256", required=True)
  parser.add_argument("--manifest", type=Path, required=True)
  parser.add_argument("--manifest-sha256", required=True)
  parser.add_argument("--repo-root", type=Path, required=True)
  parser.add_argument("--output-root", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument(
    "--max-wall-seconds",
    type=float,
    default=MAX_WALL_SECONDS,
  )
  parser.add_argument(
    "--max-output-bytes",
    type=int,
    default=MAX_OUTPUT_BYTES,
  )
  parser.add_argument(
    "--execute",
    action="store_true",
    help="launch the six evaluator subprocesses; otherwise print the plan",
  )
  return parser


def _validate_bounds(args: argparse.Namespace) -> None:
  if (
    isinstance(args.max_wall_seconds, bool)
    or not math.isfinite(float(args.max_wall_seconds))
    or not 0.0 < float(args.max_wall_seconds) <= MAX_WALL_SECONDS
  ):
    raise PilotError(f"max-wall-seconds must be finite and in (0, {MAX_WALL_SECONDS}]")
  if (
    isinstance(args.max_output_bytes, bool)
    or not isinstance(args.max_output_bytes, int)
    or not MIN_OUTPUT_CAP_BYTES <= args.max_output_bytes <= MAX_OUTPUT_BYTES
  ):
    raise PilotError(
      f"max-output-bytes must be an integer in [{MIN_OUTPUT_CAP_BYTES}, "
      f"{MAX_OUTPUT_BYTES}]"
    )


def _evaluator_runner(
  plan: PilotPlan,
) -> Callable[[ConditionSpec, Path, float, int], ConditionExecution]:
  env = os.environ.copy()
  env.setdefault("PYTHONPATH", str(plan.repo_root / "src"))

  def run(
    spec: ConditionSpec, condition_dir: Path, timeout_s: float, output_cap: int
  ) -> ConditionExecution:
    stderr_path = condition_dir.parent / f"{condition_dir.name}.stderr.txt"
    return run_evaluator_subprocess(
      plan.condition_command(spec, condition_dir, output_cap),
      cwd=plan.repo_root,
      timeout_s=timeout_s,
      stderr_path=stderr_path,
      env=env,
    )

  return run


def main(argv: Sequence[str] | None = None) -> int:
  parser = _parser()
  args = parser.parse_args(argv)
  try:
    _validate_bounds(args)
    plan = build_plan(args)
  except PilotError as exc:
    print(f"D3 pilot preflight failed: {exc}", file=sys.stderr)
    return 1
  if not args.execute:
    print(json.dumps(plan.as_manifest(), indent=1, sort_keys=True))
    return 0
  output_root = args.output_root.resolve()
  if output_root.exists():
    print(
      f"D3 pilot output-root must be new and absent: {output_root}",
      file=sys.stderr,
    )
    return 1
  output_root.mkdir(parents=True)
  manifest_path = output_root / "pilot_manifest.json"
  manifest_path.write_text(
    json.dumps(plan.as_manifest(), indent=1, sort_keys=True) + "\n", encoding="utf-8"
  )
  progress_path = output_root / "progress.jsonl"
  with progress_path.open("a", encoding="utf-8") as handle:

    def emit(event: Mapping[str, Any]) -> None:
      handle.write(json.dumps(event, sort_keys=True) + "\n")
      handle.flush()
      print(json.dumps({"progress": dict(event)}, sort_keys=True), flush=True)

    result = run_batch(
      plan.conditions,
      output_root=output_root,
      run_condition=_evaluator_runner(plan),
      max_wall_seconds=plan.max_wall_seconds,
      max_output_bytes=plan.max_output_bytes,
      progress=emit,
    )
  (output_root / "batch_result.json").write_text(
    json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(
    json.dumps(
      {
        "schema": result["schema"],
        "conditions_planned": result["conditions_planned"],
        "conditions_attempted": result["conditions_attempted"],
        "stop_reason": result["stop_reason"],
        "wall_seconds": result["wall_seconds"],
        "output_bytes": result["output_bytes"],
        "output_root": str(output_root),
      },
      sort_keys=True,
    )
  )
  complete = result["conditions_attempted"] == result["conditions_planned"]
  return 0 if complete and result["stop_reason"] is None else 1


if __name__ == "__main__":
  raise SystemExit(main())
