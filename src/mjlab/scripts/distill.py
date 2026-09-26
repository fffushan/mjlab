"""Bounded single-teacher distillation commands.

The CLI keeps M1 teacher validation lightweight and adds the opt-in M3 train and
bounded evaluation surfaces.  Train/evaluate construct the trusted native Luna
adapter; defaults are deliberately small and never start an unbounded run.
Student evaluation is model-only: the saved schema and model settings are
inferred from the checkpoint instead of being repeated on the command line.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import asdict
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

_COMMANDS = ("validate-teachers", "train", "evaluate")
SamplingMode = Literal["start", "uniform"]
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
  print(file=stream)
  print("Run 'distill <COMMAND> --help' for command-specific options.", file=stream)


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
) -> dict[str, Any]:
  """Complete resolved runner/trainer/runtime configuration for one run.

  Everything a resume must reproduce is recorded here.  The single exception is
  ``schedule.max_iterations``, documented as a total lifetime budget that the
  caller may explicitly extend on resume; every other entry is compared by
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


def _iteration_report(iteration) -> dict:
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
      "boundaries": [asdict(item) for item in collection.boundaries],
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
  output_dir: Path = Path("distillation-runs/latest"),
  resume: Path | None = None,
) -> int:
  """Run a bounded native single-teacher collect/update lifecycle.

  ``max_iterations`` is the total lifetime iteration budget, including when
  ``resume`` is supplied.  Resume restores replay, normalizers, optimizer, and
  RNG state; the simulator is restarted and bootstrap is not repeated when the
  checkpoint contains replay.  ``seed`` is forwarded into environment
  construction before startup randomization, and the resolved seed, the seed
  provenance, and the complete resolved configuration are reported.
  """
  runner = None
  try:
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
    iterations = runner.run()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoint-final.pt"
    runner.save(
      str(checkpoint),
      teacher_hashes=selected.hashes,
      control_contract=resolved_config["control_contract"],
      resolved_config=resolved_config,
      schedule=resolved_config["schedule"],
    )
    payload = {
      "command": " ".join(sys.argv),
      "status": "implementation_smoke_only",
      "checkpoint": str(checkpoint),
      "iteration": runner.iteration,
      "iterations": [_iteration_report(item) for item in iterations],
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
