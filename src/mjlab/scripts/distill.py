"""Bounded single-teacher distillation commands.

The CLI keeps M1 teacher validation lightweight and adds the opt-in M3 train and
bounded evaluation surfaces.  Train/evaluate construct the trusted native Luna
adapter; defaults are deliberately small and never start an unbounded run.
Student evaluation is model-only: the saved schema and model settings are
inferred from the checkpoint instead of being repeated on the command line.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import torch
import tyro

import mjlab
from mjlab.tasks.tracking.distillation.adapter import make_distillation_adapter
from mjlab.tasks.tracking.distillation.checkpoint import (
  CheckpointValidationError,
  InferenceModel,
  LifecycleState,
  load_inference_checkpoint,
)
from mjlab.tasks.tracking.distillation.collector import (
  DAggerCollector,
  EvaluationMode,
  RolloutLatent,
  evaluate_distillation,
)
from mjlab.tasks.tracking.distillation.config import (
  DistillationError,
  MissingValidationDependencyError,
  load_manifest,
  resolve_cohort,
)
from mjlab.tasks.tracking.distillation.model import ConditionalVAE
from mjlab.tasks.tracking.distillation.parity import (
  DEFAULT_ATOL,
  DEFAULT_RTOL,
  DEFAULT_SAMPLES,
  DEFAULT_SEED,
  validate_teachers,
)
from mjlab.tasks.tracking.distillation.playback import (
  DistillationPlayEnvironment,
  DistillationPlayPolicy,
  discover_distillation_checkpoints,
)
from mjlab.tasks.tracking.distillation.runner import (
  DistillationRunner,
  RunnerConfig,
)
from mjlab.tasks.tracking.distillation.storage import LabeledReplayBuffer
from mjlab.tasks.tracking.distillation.trainer import (
  TrainingConfig,
  VaeDistillationTrainer,
)
from mjlab.tasks.tracking.distillation.vae_config import DEFAULT_MODEL_SETTINGS

_COMMANDS = ("validate-teachers", "train", "evaluate", "play")
SamplingMode = Literal["start", "uniform"]
ReportBoundaries = Literal["summary", "full"]
PlayViewer = Literal["viser", "native", "auto"]
PROVENANCE_VERSION = 1
"""Version of the ``resolved_config`` record written into checkpoints."""

_RESUME_INVARIANT_KEYS = (
  "teacher_id",
  "teacher_hashes",
  "motion",
  "task",
  "task_id",
  "control_contract",
  "runtime",
  "trainer",
  "replay",
  "model",
)
"""Provenance entries a resume must reproduce; only the iteration budget grows."""


def _print_help(stream) -> None:
  print("usage: distill <COMMAND> [OPTIONS]", file=stream)
  print(file=stream)
  print("Commands:", file=stream)
  print(
    "  validate-teachers  Validate a teacher manifest and check native/ONNX parity.",
    file=stream,
  )
  print("  train              Run a bounded single-teacher M3 lifecycle.", file=stream)
  print("  evaluate           Run bounded teacher/student evaluation.", file=stream)
  print(
    "  play               Play a checkpointed student in a Viser/native viewer.",
    file=stream,
  )
  print(file=stream)
  print("Notes:", file=stream)
  print(
    "  --seed is applied to environment startup before randomized construction and",
    file=stream,
  )
  print(
    "  the resolved seed plus the full resolved configuration are reported.",
    file=stream,
  )
  print(
    "  'evaluate --mode student' infers the saved schema/model settings from",
    file=stream,
  )
  print("  --checkpoint and takes no trainer-only flags.", file=stream)
  print(
    "  'play' reuses the evaluate environment contract (no play-mode overrides)",
    file=stream,
  )
  print(
    "  and decodes the mean latent; it constructs no trainer, replay, or runner.",
    file=stream,
  )
  print(file=stream)
  print("Run 'distill <COMMAND> --help' for command-specific options.", file=stream)


def _default_train_output_dir(task_id: str | None) -> Path:
  """Choose a unique, human-readable default directory for a training run."""
  if task_id:
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", task_id).strip("-.")
  else:
    name = ""
  if not name:
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  return Path("logs") / "distillation" / name


def _write_report(payload: dict, report: Path | None) -> str:
  text = json.dumps(payload, indent=2, sort_keys=True)
  print(text)
  if report is not None:
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text + "\n")
  return text


def _resolve(manifest: Path, repo_root: Path):
  return resolve_cohort(load_manifest(manifest, repo_root))


def _control_metadata(cohort, teacher_id: str) -> dict:
  selected = cohort.teacher(teacher_id)
  return {
    "control_period_s": cohort.control.control_period_s,
    "control_hz": cohort.control.control_hz,
    "sim_timestep": cohort.control.sim_timestep,
    "decimation": cohort.control.decimation,
    "action_joint_names": list(cohort.actions.joint_names),
    "action_dim": cohort.actions.dim,
    "action_scales": list(cohort.actions.joint_scales),
    "action_offset": cohort.actions.offset,
    "teacher_id": teacher_id,
    "motion": str(selected.entry.motion),
    "task": cohort.manifest.base_task,
  }


def _seed_audit(adapter) -> dict[str, Any]:
  """Resolved seed/provenance recorded by the environment factory, as data.

  ``--seed`` is forwarded into adapter construction so the private environment
  config is seeded *before* startup randomization.  This reads back what the
  factory actually applied instead of assuming the request was honored.
  """
  provenance = getattr(getattr(adapter, "audit", None), "seed_provenance", None)
  if provenance is None:
    return {
      "available": False,
      "requested_seed": None,
      "effective_seed": None,
      "applied_before_construction": False,
    }
  return {"available": True, **asdict(provenance)}


def _require_requested_seed_applied(seed: int, audit: Mapping[str, Any]) -> None:
  """Refuse to report a resolved seed that environment startup did not use."""
  if not audit["available"]:
    raise DistillationError(
      "adapter construction did not report seed provenance, so the resolved seed "
      "cannot be audited"
    )
  if not audit["applied_before_construction"] or audit["effective_seed"] != seed:
    raise DistillationError(
      f"requested seed {seed} was not applied before environment construction "
      f"(requested={audit['requested_seed']}, resolved={audit['effective_seed']}, "
      f"applied_before_construction={audit['applied_before_construction']})"
    )


def _schedule_config(
  runner: DistillationRunner, evaluation_sampling_mode: str
) -> dict[str, Any]:
  """The resolved bounded lifecycle schedule owned by the runner config.

  ``evaluation_sampling_mode`` is the mode the runner's periodic evaluation
  samples reference segments with; the caller reads it from the live motion
  command so the record is the resolved setting rather than an assumption.
  """
  config = runner.config
  return {
    "max_iterations": config.max_iterations,
    "bootstrap_steps": config.bootstrap_steps,
    "collection_steps": config.collection_steps,
    "updates_per_iteration": config.updates_per_iteration,
    "teacher_probability": config.teacher_probability,
    "evaluate_every": config.evaluate_every,
    "evaluation_steps": config.evaluation_steps,
    "evaluation_mode": config.evaluation_mode,
    "evaluation_sampling_mode": evaluation_sampling_mode,
    "rollout_latent": config.rollout_latent,
  }


def _resolved_config(
  *,
  command: str,
  manifest: Path,
  repo_root: Path,
  cohort,
  teacher_id: str,
  task_id: str | None,
  device: str,
  num_envs: int,
  seed: int,
  seed_audit: Mapping[str, Any],
  evaluation_sampling_mode: str,
  runner: DistillationRunner,
  resume: Path | None,
  checkpoint_every: int,
) -> dict[str, Any]:
  """Complete resolved runner/trainer/runtime configuration for one run.

  Everything a resume must reproduce is recorded here, plus the requested save
  cadence (``checkpoint_every``) which is audit-only: it is deliberately kept
  out of :data:`_RESUME_INVARIANT_KEYS` so changing ``--checkpoint-every`` never
  refuses a resume.  The one compared budget is ``schedule.max_iterations``,
  documented as a total lifetime budget that the caller may explicitly extend
  on resume; every other compared entry is checked by
  :func:`_check_resume_compatibility`.
  """
  teacher = cohort.teacher(teacher_id)
  trainer = runner.trainer
  return {
    "provenance_version": PROVENANCE_VERSION,
    "command": command,
    "manifest": str(manifest),
    "repo_root": str(repo_root),
    "teacher_id": teacher_id,
    "teacher_hashes": dict(teacher.hashes),
    "motion": str(teacher.entry.motion),
    "task": cohort.manifest.base_task,
    "task_id": task_id,
    "control_contract": _control_metadata(cohort, teacher_id),
    "runtime": {
      "device": device,
      "num_envs": num_envs,
      "seed": seed,
      "resolved_seed": seed_audit["effective_seed"],
      "seed_provenance": dict(seed_audit),
    },
    "schedule": _schedule_config(runner, evaluation_sampling_mode),
    "checkpoint_every": checkpoint_every,
    "trainer": {
      "learning_rate": trainer.config.learning_rate,
      "beta": trainer.config.beta,
      "accumulation_steps": trainer.config.accumulation_steps,
      "minibatch_size": trainer.config.minibatch_size,
      "latent_mode": trainer.config.latent_mode,
    },
    "replay": {"capacity": runner.replay.capacity},
    "model": {
      "settings": trainer.model.settings.to_metadata(),
      "schema": trainer.model.schema_metadata,
    },
    "resumed_from": None if resume is None else str(resume),
  }


def _check_resume_compatibility(
  state: LifecycleState, requested: Mapping[str, Any]
) -> dict[str, Any]:
  """Refuse a resume that would silently change stored settings/schedules.

  Every stored semantic entry must be reproduced; only the total lifetime
  iteration budget may grow.  A checkpoint that predates this provenance record
  is reported as unverified rather than being failed against fields it never
  stored, and any entry the checkpoint *did* record is still compared.
  """
  stored = state.resolved_config
  verified = stored.get("provenance_version") == PROVENANCE_VERSION
  audit: dict[str, Any] = {
    "stored_provenance_version": stored.get("provenance_version"),
    "provenance_verified": verified,
    "checked": [],
    "mismatches": [],
    "budget": None,
    "unverified": [],
  }
  if not stored:
    audit["unverified"].append("resolved_config")
  for key in _RESUME_INVARIANT_KEYS:
    if key not in stored:
      audit["unverified"].append(key)
      continue
    audit["checked"].append(key)
    if stored[key] != requested[key]:
      audit["mismatches"].append(key)
  stored_schedule = state.schedule or stored.get("schedule") or {}
  if not stored_schedule:
    audit["unverified"].append("schedule")
  requested_schedule = requested["schedule"]
  for key in sorted(set(stored_schedule) - {"max_iterations"}):
    audit["checked"].append(f"schedule.{key}")
    if stored_schedule[key] != requested_schedule.get(key):
      audit["mismatches"].append(f"schedule.{key}")
  stored_budget = stored_schedule.get("max_iterations")
  requested_budget = requested_schedule["max_iterations"]
  if isinstance(stored_budget, int) and not isinstance(stored_budget, bool):
    audit["budget"] = {
      "stored_max_iterations": stored_budget,
      "requested_max_iterations": requested_budget,
      "extended": requested_budget > stored_budget,
    }
    if requested_budget < stored_budget:
      audit["mismatches"].append("schedule.max_iterations")
  else:
    audit["unverified"].append("schedule.max_iterations")
  if not verified:
    audit["unverified"].append("provenance_version")
  if audit["mismatches"]:
    raise DistillationError(
      "resume refused: the checkpoint records different "
      + ", ".join(sorted(set(audit["mismatches"])))
      + "; repeat the stored settings or start a new run"
    )
  return audit


def _checkpoint_model_report(inference: InferenceModel | None) -> dict[str, Any] | None:
  """Model-only identity inferred from a student checkpoint, for the report."""
  if inference is None:
    return None
  return {
    "schema": inference.schema.compatibility_metadata(),
    "settings": inference.settings.to_metadata(),
    "counters": dict(inference.counters),
    "schedule": dict(inference.schedule),
  }


def _build_runner(
  *,
  cohort,
  teacher_id: str,
  device: str,
  num_envs: int,
  task_id: str | None,
  replay_capacity: int,
  minibatch_size: int,
  accumulation_steps: int,
  learning_rate: float,
  beta: float,
  seed: int,
  max_iterations: int,
  bootstrap_steps: int,
  collection_steps: int,
  updates_per_iteration: int,
  teacher_probability: float,
  evaluate_every: int,
  evaluation_steps: int,
  rollout_latent: RolloutLatent,
):
  adapter = make_distillation_adapter(
    cohort,
    teacher_id,
    task_id=task_id,
    num_envs=num_envs,
    device=device,
    seed=seed,
  )
  # Model initialization only: environment startup randomization is seeded by
  # the adapter factory above, before the private env config is constructed.
  torch.manual_seed(seed)
  model = ConditionalVAE(adapter.schema, DEFAULT_MODEL_SETTINGS).to(device)
  replay = LabeledReplayBuffer(
    replay_capacity, adapter.schema, device=torch.device(device), dtype=torch.float32
  )
  trainer = VaeDistillationTrainer(
    model,
    replay,
    TrainingConfig(
      learning_rate=learning_rate,
      beta=beta,
      accumulation_steps=accumulation_steps,
      minibatch_size=minibatch_size,
    ),
    seed=seed,
    teacher=adapter.teacher,
  )
  collector = DAggerCollector(adapter, adapter.teacher, model, replay)
  runner = DistillationRunner(
    collector,
    trainer,
    RunnerConfig(
      max_iterations=max_iterations,
      bootstrap_steps=bootstrap_steps,
      collection_steps=collection_steps,
      updates_per_iteration=updates_per_iteration,
      teacher_probability=teacher_probability,
      evaluate_every=evaluate_every,
      evaluation_steps=evaluation_steps,
      evaluation_mode="student",
      rollout_latent=rollout_latent,
      seed=seed,
    ),
  )
  sampling_mode = adapter.env.command_manager.get_term("motion").cfg.sampling_mode
  return runner, cohort.teacher(teacher_id), sampling_mode


def _observed_range(values: list[int]) -> list[int] | None:
  """Inclusive ``[min, max]`` of the observed ids, or ``None`` if none."""
  return None if not values else [min(values), max(values)]


def _boundary_summary(collection) -> dict[str, Any]:
  """Aggregate one iteration's boundaries instead of their per-environment arrays.

  A raw boundary record carries four 4096-entry arrays (``before_segment``,
  ``after_segment``, ``before_generation``, ``after_generation``) plus
  ``env_indices``, so a full run's report is dominated by a payload an auditor
  never reads element by element.  This summary keeps the auditable signal:
  how many records and environment mentions each ``reason`` produced, and the
  before/after segment and generation ranges observed at the mentioned
  environments.  It is computed in one pass over the iteration's records and
  retains no array.
  """
  reasons: dict[str, dict[str, int]] = {}
  before_segment: list[int] = []
  after_segment: list[int] = []
  before_generation: list[int] = []
  after_generation: list[int] = []
  env_mentions = 0
  for record in collection.boundaries:
    mentions = len(record.env_indices)
    env_mentions += mentions
    counts = reasons.setdefault(record.reason, {"records": 0, "env_mentions": 0})
    counts["records"] += 1
    counts["env_mentions"] += mentions
    for index in record.env_indices:
      before_segment.append(record.before_segment[index])
      after_segment.append(record.after_segment[index])
      before_generation.append(record.before_generation[index])
      after_generation.append(record.after_generation[index])
  return {
    "mode": "summary",
    "ticks": collection.ticks,
    "records": len(collection.boundaries),
    "env_mentions": env_mentions,
    "reasons": reasons,
    "before_segment_range": _observed_range(before_segment),
    "after_segment_range": _observed_range(after_segment),
    "before_generation_range": _observed_range(before_generation),
    "after_generation_range": _observed_range(after_generation),
  }


def _boundaries_report(collection, mode: ReportBoundaries):
  """Boundary evidence for one iteration: compact summary, or raw detail."""
  if mode == "full":
    return [asdict(item) for item in collection.boundaries]
  return _boundary_summary(collection)


def _iteration_report(iteration, report_boundaries: ReportBoundaries) -> dict:
  collection = iteration.collection
  return {
    "iteration": iteration.iteration,
    "resumed_reset": iteration.resumed_reset,
    "collection": None
    if collection is None
    else {
      "ticks": collection.ticks,
      "samples": collection.samples,
      "teacher_steps": collection.teacher_steps,
      "student_steps": collection.student_steps,
      "disagreement_mean": collection.disagreement_mean,
      "diagnostics": list(collection.diagnostics),
      "boundaries": _boundaries_report(collection, report_boundaries),
      "fresh_data_samples": (
        None
        if collection.fresh_data is None
        else collection.fresh_data.batch.batch_size
      ),
    },
    "updates": [asdict(update) for update in iteration.updates],
    "evaluation": None
    if iteration.evaluation is None
    else asdict(iteration.evaluation),
  }


def _train(
  manifest: Path = Path("configs/distillation/x2_tennis.yaml"),
  repo_root: Path = Path("."),
  teacher_id: str = "tennis_000",
  task_id: str | None = None,
  device: str = "cpu",
  num_envs: int = 1,
  max_iterations: int = 1,
  bootstrap_steps: int = 0,
  collection_steps: int = 32,
  updates_per_iteration: int = 1,
  teacher_probability: float = 0.0,
  evaluate_every: int = 0,
  evaluation_steps: int = 0,
  minibatch_size: int = 256,
  accumulation_steps: int = 15,
  replay_capacity: int = 16_384,
  learning_rate: float = 5e-4,
  beta: float = 0.01,
  seed: int = 0,
  rollout_latent: RolloutLatent = "mean",
  checkpoint_every: int = 500,
  progress_every: int = 10,
  report_boundaries: ReportBoundaries = "summary",
  output_dir: Path | None = None,
  resume: Path | None = None,
) -> int:
  """Run a bounded native single-teacher collect/update lifecycle.

  ``max_iterations`` is the total lifetime iteration budget, including when
  ``resume`` is supplied.  Resume restores replay, normalizers, optimizer, and
  RNG state; the simulator is restarted and bootstrap is not repeated when the
  checkpoint contains replay.  ``seed`` is forwarded into environment
  construction before startup randomization, and the resolved seed, the seed
  provenance, and the complete resolved configuration are reported.

  ``checkpoint_every`` is a periodic save cadence counted in *completed*
  iterations.  ``0`` disables periodic checkpoints and keeps only the final
  ``checkpoint-final.pt``.  When positive, a checkpoint named from the total
  lifetime iteration counter is written atomically every ``checkpoint_every``
  completed iterations, every checkpoint is a complete resume input, and the
  report lists every checkpoint written by this invocation.

  The default cadence is 500.  ``progress_every`` defaults to 10 and writes
  flushed progress lines to stderr.  When ``output_dir`` is omitted, the run is
  stored under ``logs/distillation/<task-id>/`` when ``task_id`` is supplied,
  otherwise under ``logs/distillation/<UTC timestamp>/``.

  ``progress_every`` writes a flushed ``[progress]`` line to **stderr** every
  ``N`` completed iterations, reporting the lifetime iteration, collected
  samples, last update loss, teacher/student disagreement, elapsed time,
  seconds per iteration for this invocation, and a projected ETA.  ``0``
  disables progress output.  Progress deliberately goes to stderr so that
  stdout stays exactly the machine-readable JSON report that callers and tests
  parse; a long run is otherwise observable only through its periodic
  checkpoints.

  ``report_boundaries`` selects the boundary evidence in each iteration's
  ``collection.boundaries``.  ``summary`` (the default) reports per-``reason``
  record/environment-mention counts and the observed before/after segment and
  generation ranges, and never retains a boundary's per-environment arrays;
  ``full`` writes those arrays as before, which is roughly 2 MB per iteration
  at 4096 environments, so it is a debugging option, not the default.
  """
  runner = None
  try:
    if report_boundaries not in ("summary", "full"):
      raise ValueError(
        f"report_boundaries must be 'summary' or 'full' (got {report_boundaries!r})"
      )
    if checkpoint_every < 0:
      raise ValueError(
        f"--checkpoint-every must be a non-negative integer (got "
        f"{checkpoint_every}); use 0 to disable periodic checkpoints and keep "
        "only the final checkpoint"
      )
    if progress_every < 0:
      raise ValueError(
        f"--progress-every must be a non-negative integer (got {progress_every}); "
        "use 0 to disable progress output"
      )
    cohort = _resolve(manifest, repo_root)
    runner, selected, evaluation_sampling_mode = _build_runner(
      cohort=cohort,
      teacher_id=teacher_id,
      device=device,
      num_envs=num_envs,
      task_id=task_id,
      replay_capacity=replay_capacity,
      minibatch_size=minibatch_size,
      accumulation_steps=accumulation_steps,
      learning_rate=learning_rate,
      beta=beta,
      seed=seed,
      max_iterations=max_iterations,
      bootstrap_steps=bootstrap_steps,
      collection_steps=collection_steps,
      updates_per_iteration=updates_per_iteration,
      teacher_probability=teacher_probability,
      evaluate_every=evaluate_every,
      evaluation_steps=evaluation_steps,
      rollout_latent=rollout_latent,
    )
    seed_audit = _seed_audit(runner.collector.adapter)
    _require_requested_seed_applied(seed, seed_audit)
    resolved_config = _resolved_config(
      command="train",
      manifest=manifest,
      repo_root=repo_root,
      cohort=cohort,
      teacher_id=teacher_id,
      task_id=task_id,
      device=device,
      num_envs=num_envs,
      seed=seed,
      seed_audit=seed_audit,
      evaluation_sampling_mode=evaluation_sampling_mode,
      runner=runner,
      resume=resume,
      checkpoint_every=checkpoint_every,
    )
    resume_audit = None
    if resume is not None:
      state = runner.resume(
        str(resume),
        teacher_hashes=selected.hashes,
        control_contract=resolved_config["control_contract"],
        map_location=device,
      )
      resume_audit = _check_resume_compatibility(state, resolved_config)
    if output_dir is None:
      output_dir = _default_train_output_dir(task_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    def write_checkpoint(path: Path) -> None:
      runner.save(
        str(path),
        teacher_hashes=selected.hashes,
        control_contract=resolved_config["control_contract"],
        resolved_config=resolved_config,
        schedule=resolved_config["schedule"],
      )

    # The existing bounded runner is driven in chunks and its one save call is
    # reused, so a periodic checkpoint carries the same provenance/schedule
    # metadata as the final one and is an ordinary resume input.  Filenames use
    # ``runner.iteration``, the total lifetime counter, so a resumed run
    # continues the same sequence instead of colliding with earlier files.
    # Each iteration is converted to its report form as soon as it returns, so
    # a summary report never holds more than one iteration's boundary arrays.
    iteration_reports: list[dict] = []
    checkpoints: list[Path] = []
    progress_started = time.monotonic()
    iterations_this_run = 0
    if progress_every > 0:
      print(
        f"[progress] starting at iteration {runner.iteration}/{max_iterations} "
        f"device={device} num_envs={num_envs} progress_every={progress_every}",
        file=sys.stderr,
        flush=True,
      )
    while runner.iteration < max_iterations:
      remaining = max_iterations - runner.iteration
      chunk = remaining if checkpoint_every <= 0 else min(checkpoint_every, remaining)
      for _ in range(chunk):
        iteration_reports.append(
          _iteration_report(runner.run_iteration(), report_boundaries)
        )
        iterations_this_run += 1
        if progress_every > 0 and (
          runner.iteration % progress_every == 0 or runner.iteration >= max_iterations
        ):
          latest = iteration_reports[-1]
          collection = latest.get("collection") or {}
          updates = latest.get("updates") or []
          raw_loss = updates[-1].get("total_loss") if updates else None
          raw_disagreement = collection.get("disagreement_mean")
          loss_text = "n/a" if raw_loss is None else f"{raw_loss:.4f}"
          disagreement_text = (
            "n/a" if raw_disagreement is None else f"{raw_disagreement:.4f}"
          )
          elapsed = time.monotonic() - progress_started
          per_iteration = elapsed / max(iterations_this_run, 1)
          eta_hours = per_iteration * (max_iterations - runner.iteration) / 3600.0
          print(
            f"[progress] iter {runner.iteration}/{max_iterations} "
            f"samples={collection.get('samples')} loss={loss_text} "
            f"disagreement={disagreement_text} elapsed={elapsed:.1f}s "
            f"({per_iteration:.2f} s/iter this run) eta={eta_hours:.2f}h",
            file=sys.stderr,
            flush=True,
          )
      if checkpoint_every > 0:
        periodic = output_dir / f"checkpoint-iter-{runner.iteration:06d}.pt"
        write_checkpoint(periodic)
        checkpoints.append(periodic)
    checkpoint = output_dir / "checkpoint-final.pt"
    write_checkpoint(checkpoint)
    checkpoints.append(checkpoint)
    payload = {
      "command": " ".join(sys.argv),
      "status": "implementation_smoke_only",
      "checkpoint": str(checkpoint),
      "checkpoints": [str(path) for path in checkpoints],
      "iteration": runner.iteration,
      "iterations": iteration_reports,
      "events": runner.events,
      "runtime": {
        "device": device,
        "num_envs": num_envs,
        "requested_seed": seed,
        "resolved_seed": seed_audit["effective_seed"],
        "seed_provenance": seed_audit,
      },
      "schedule": resolved_config["schedule"],
      "resolved_config": resolved_config,
      "resume": resume_audit,
      "quality": {
        "implementation": "bounded native lifecycle constructed",
        "smoke": "not a policy-quality or convergence claim",
        "policy_quality": "not assessed by this command",
        "hardware_readiness": "not assessed",
      },
    }
    _write_report(payload, output_dir / "train-report.json")
    return 0
  except (
    DistillationError,
    CheckpointValidationError,
    ValueError,
    RuntimeError,
  ) as exc:
    print(f"[FAIL] {exc}", file=sys.stderr)
    return 1
  finally:
    if runner is not None:
      close = getattr(runner.collector.adapter, "close", None)
      if callable(close):
        close()


def _evaluate(
  manifest: Path = Path("configs/distillation/x2_tennis.yaml"),
  repo_root: Path = Path("."),
  teacher_id: str = "tennis_000",
  task_id: str | None = None,
  device: str = "cpu",
  num_envs: int = 1,
  mode: EvaluationMode = "student",
  checkpoint: Path | None = None,
  steps: int = 512,
  seed: int = 0,
  sampling_mode: SamplingMode = "start",
  rollout_latent: RolloutLatent = "mean",
  report: Path | None = None,
) -> int:
  """Evaluate one teacher or a checkpointed student without mutation/training.

  Student evaluation is model-only: ``--checkpoint`` supplies the saved schema
  and model settings, so no trainer-only flag (learning rate, beta,
  accumulation, minibatch, replay capacity) is accepted here.  ``--seed`` is
  forwarded into environment construction before startup randomization, and the
  resolved seed plus its provenance is reported.  The reported
  ``tracking_root_relative_pose_error`` removes only root translation while
  keeping world axes, so a root yaw difference between reference and robot
  contributes to it; it is not heading-invariant articulation error.  Heading
  is reported separately as ``tracking_heading_error`` (wrapped root-relative
  yaw delta).
  """
  adapter = None
  try:
    if sampling_mode not in ("start", "uniform"):
      raise ValueError("sampling_mode must be 'start' or 'uniform'")
    if mode not in ("teacher", "student"):
      raise ValueError("mode must be 'teacher' or 'student'")
    if mode == "student" and checkpoint is None:
      raise ValueError("student evaluation requires --checkpoint")
    cohort = _resolve(manifest, repo_root)
    adapter = make_distillation_adapter(
      cohort,
      teacher_id,
      task_id=task_id,
      num_envs=num_envs,
      device=device,
      seed=seed,
    )
    seed_audit = _seed_audit(adapter)
    _require_requested_seed_applied(seed, seed_audit)
    # This is an evaluation-only sampling override on the private command copy;
    # validate_live_contract has already checked the saved semantic contract.
    motion = adapter.env.command_manager.get_term("motion")
    motion.cfg.sampling_mode = sampling_mode
    student = None
    inference = None
    if mode == "student":
      assert checkpoint is not None
      # Model-only load: the saved optimizer, replay, and collector RNG are not
      # required, and no training tensor is moved to ``device``.  The saved
      # schema/model settings are inferred instead of being re-declared here.
      inference = load_inference_checkpoint(
        checkpoint,
        device=device,
        expected_schema=adapter.schema,
        expected_teacher_hashes=cohort.teacher(teacher_id).hashes,
        expected_control_contract=_control_metadata(cohort, teacher_id),
      )
      student = inference.model
    result = evaluate_distillation(
      adapter,
      adapter.teacher,
      student,
      mode=mode,
      steps=steps,
      rollout_latent=rollout_latent,
      seed=seed,
      control_period_s=cohort.control.control_period_s,
    )
    payload = {
      "command": " ".join(sys.argv),
      "status": "evaluation",
      "mode": mode,
      "sampling_mode": sampling_mode,
      "checkpoint": None if checkpoint is None else str(checkpoint),
      "teacher_id": teacher_id,
      "motion": str(cohort.teacher(teacher_id).entry.motion),
      "control_hz": cohort.control.control_hz,
      "runtime": {
        "device": device,
        "num_envs": num_envs,
        "requested_seed": seed,
        "resolved_seed": seed_audit["effective_seed"],
        "seed_provenance": seed_audit,
      },
      "checkpoint_model": _checkpoint_model_report(inference),
      "checkpoint_resolved_config": (
        None if inference is None else dict(inference.resolved_config)
      ),
      "result": asdict(result),
      "quality": {
        "implementation": "bounded native evaluation",
        "smoke": "metrics are bounded-run evidence only",
        "policy_quality": "requires parent baseline-relative interpretation",
        "hardware_readiness": "not assessed",
      },
    }
    _write_report(payload, report)
    return 0
  except (
    DistillationError,
    CheckpointValidationError,
    ValueError,
    RuntimeError,
  ) as exc:
    print(f"[FAIL] {exc}", file=sys.stderr)
    return 1
  finally:
    if adapter is not None:
      adapter.close()


def _resolve_play_viewer(viewer: str) -> str:
  """Resolve ``auto`` to native when a display is present, otherwise Viser."""
  if viewer != "auto":
    return viewer
  has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
  return "native" if has_display else "viser"


def _playback_checkpoint_manager(
  checkpoint: Path,
  cohort,
  teacher_id: str,
  adapter,
  device: str,
):
  """Build a Viser checkpoint manager that discovers and validates swaps.

  Discovery uses the distillation naming scheme, and every load (including a
  hot swap) goes through :func:`load_inference_checkpoint` with the live
  schema, teacher hashes, and control contract, so an incompatible artifact is
  rejected instead of silently replacing the running policy.  The live schema
  is the one the initialized environment packs with, so a swap whose saved
  schema differs is refused rather than mis-packed.
  """
  from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago

  directory = checkpoint.parent
  expected = {
    "expected_schema": adapter.schema,
    "expected_teacher_hashes": cohort.teacher(teacher_id).hashes,
    "expected_control_contract": _control_metadata(cohort, teacher_id),
  }

  def fetch_available() -> list[tuple[str, str]]:
    discovered = discover_distillation_checkpoints(directory)
    if checkpoint not in discovered:
      discovered = [checkpoint, *discovered]
    now = time.time()
    return [
      (path.name, format_time_ago(int(now - path.stat().st_mtime)))
      for path in discovered
    ]

  def load(name: str):
    inference = load_inference_checkpoint(directory / name, device=device, **expected)
    return DistillationPlayPolicy(inference.model)

  return CheckpointManager(
    current_name=checkpoint.name,
    fetch_available=fetch_available,
    load_checkpoint=load,
  )


def _play(
  manifest: Path = Path("configs/distillation/x2_tennis.yaml"),
  repo_root: Path = Path("."),
  teacher_id: str = "tennis_000",
  task_id: str | None = None,
  device: str = "cpu",
  num_envs: int = 1,
  checkpoint: Path | None = None,
  seed: int = 0,
  sampling_mode: SamplingMode = "start",
  viewer: PlayViewer = "viser",
  frame_rate: float = 60.0,
) -> int:
  """Play a checkpointed student interactively in the shared viewers.

  The environment is the same audited native contract ``distill evaluate``
  builds: the registered task is loaded *without* its ``play=True`` overrides,
  so actor corruption, resets, and episode length stay those the teacher was
  trained under, while ``--sampling-mode`` gives an explicit start/uniform
  reference-frame override on the private command copy.  The saved student is
  reconstructed model-only (no optimizer, replay, trainer, or PPO runner) and
  decoded with deterministic mean-latent inference.  ``--viewer viser``
  (default) serves the browser viewer; ``native`` uses MuJoCo's passive viewer.

  The checkpoint is loaded *before* the simulator is built, so a missing or
  incompatible artifact never constructs an environment, and the saved schema
  is what the live packing is built from.  One audited seeded reset runs after
  the sampling override and before the first viewer action (the environment
  constructor does not reset, and unlike the PPO path there is no vector-env
  wrapper doing it).
  """
  adapter = None
  try:
    if sampling_mode not in ("start", "uniform"):
      raise ValueError("sampling_mode must be 'start' or 'uniform'")
    if viewer not in ("viser", "native", "auto"):
      raise ValueError("viewer must be 'viser', 'native', or 'auto'")
    if checkpoint is None:
      raise ValueError("playback requires --checkpoint")
    if num_envs <= 0:
      raise ValueError("num_envs must be a positive integer")
    if not math.isfinite(frame_rate) or frame_rate <= 0.0:
      raise ValueError("frame_rate must be finite and positive")
    cohort = _resolve(manifest, repo_root)
    teacher = cohort.teacher(teacher_id)
    # Early model-only load: cheap, validates teacher identity and the control
    # contract, and yields the SAVED schema the live packing must reproduce
    # (the saved settings imply the trained architecture, not a default one).
    inference = load_inference_checkpoint(
      checkpoint,
      device=device,
      expected_teacher_hashes=teacher.hashes,
      expected_control_contract=_control_metadata(cohort, teacher_id),
    )
    # The adapter re-checks the live joint order, teacher sensors/action, and
    # timing against the saved contract; passing the saved schema is what makes
    # an anchor/gravity_anchor checkpoint pack correctly instead of being
    # rejected for disagreeing with a default gravity schema.
    adapter = make_distillation_adapter(
      cohort,
      teacher_id,
      task_id=task_id,
      num_envs=num_envs,
      device=device,
      schema=inference.schema,
      seed=seed,
    )
    seed_audit = _seed_audit(adapter)
    _require_requested_seed_applied(seed, seed_audit)
    # Playback-only sampling override on the private command copy; the live
    # contract was already validated against the saved teacher.
    motion = adapter.env.command_manager.get_term("motion")
    motion.cfg.sampling_mode = sampling_mode
    # One audited seeded reset after the sampling override and before the first
    # viewer action, so the first policy action reads reset state.
    adapter.reset(seed=seed)
    env = DistillationPlayEnvironment(adapter)
    policy = DistillationPlayPolicy(inference.model)
    checkpoint_manager = _playback_checkpoint_manager(
      checkpoint, cohort, teacher_id, adapter, device
    )
    resolved_viewer = _resolve_play_viewer(viewer)
    print(
      f"[INFO]: Playing {checkpoint.name} as {teacher_id} on {resolved_viewer} "
      f"(device={device}, num_envs={num_envs}, sampling_mode={sampling_mode}, "
      f"mean latent)",
      file=sys.stderr,
      flush=True,
    )
    if resolved_viewer == "native":
      from mjlab.viewer import NativeMujocoViewer

      NativeMujocoViewer(env, policy, frame_rate=frame_rate).run()
    else:
      from mjlab.viewer import ViserPlayViewer

      ViserPlayViewer(
        env,
        policy,
        frame_rate=frame_rate,
        checkpoint_manager=checkpoint_manager,
      ).run()
    return 0
  except (
    DistillationError,
    CheckpointValidationError,
    ValueError,
    RuntimeError,
  ) as exc:
    print(f"[FAIL] {exc}", file=sys.stderr)
    return 1
  finally:
    if adapter is not None:
      adapter.close()


def _validate_teachers(
  manifest: Path = Path("configs/distillation/x2_tennis.yaml"),
  repo_root: Path = Path("."),
  samples: int = DEFAULT_SAMPLES,
  seed: int = DEFAULT_SEED,
  atol: float = DEFAULT_ATOL,
  rtol: float = DEFAULT_RTOL,
  report: Path | None = None,
) -> int:
  """Validate a teacher cohort and compare native inference with ONNX exports."""
  try:
    cohort = _resolve(manifest, repo_root)
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
  handlers = {
    "validate-teachers": _validate_teachers,
    "train": _train,
    "evaluate": _evaluate,
    "play": _play,
  }
  raise SystemExit(
    tyro.cli(
      handlers[command],
      args=sys.argv[2:],
      prog=f"{Path(sys.argv[0]).name} {command}",
      config=mjlab.TYRO_FLAGS,
    )
  )


if __name__ == "__main__":
  main()
