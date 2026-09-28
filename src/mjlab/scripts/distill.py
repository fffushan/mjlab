"""Bounded single- and multi-teacher distillation commands.

The CLI keeps M1 teacher validation lightweight and adds the opt-in M3
single-teacher and M4 shared-student cohort (``--teacher-ids``) train and
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
import signal
import sys
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Literal, Sequence

import torch
import tyro

import mjlab
from mjlab.tasks.tracking.distillation.adapter import (
  InitializationKind,
  make_distillation_adapter,
  make_multi_teacher_distillation_adapter,
)
from mjlab.tasks.tracking.distillation.balanced_storage import BalancedReplayBuffer
from mjlab.tasks.tracking.distillation.checkpoint import (
  CHECKPOINT_VERSION,
  COHORT_CHECKPOINT_VERSION,
  STANDING_COHORT_CHECKPOINT_VERSION,
  CheckpointValidationError,
  CohortLifecycleState,
  CohortMemberInference,
  InferenceModel,
  LifecycleState,
  load_cohort_member_inference,
  load_inference_checkpoint,
)
from mjlab.tasks.tracking.distillation.cohort_contract import (
  CohortIdentity,
  cohort_identity_from_adapter,
  cohort_identity_from_parts,
  require_member_matches,
)
from mjlab.tasks.tracking.distillation.cohort_setup import (
  CohortSetup,
  worker_env_seed,
)
from mjlab.tasks.tracking.distillation.collector import (
  DAggerCollector,
  EvaluationMode,
  EvaluationResult,
  MotionEvaluationStats,
  RolloutLatent,
  evaluate_distillation,
)
from mjlab.tasks.tracking.distillation.config import (
  CohortContract,
  DistillationError,
  MissingValidationDependencyError,
  load_manifest,
  resolve_cohort,
)
from mjlab.tasks.tracking.distillation.export import (
  ExportValidationError,
  export_bundle,
  export_cohort_bundle,
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
from mjlab.tasks.tracking.distillation.reset_policy import (
  ResetPolicy,
  ResetPolicyKind,
  make_reset_policy,
)
from mjlab.tasks.tracking.distillation.runner import (
  DistillationRunner,
  RunnerConfig,
)
from mjlab.tasks.tracking.distillation.sharded_collection import ShardedCollection
from mjlab.tasks.tracking.distillation.storage import LabeledReplayBuffer
from mjlab.tasks.tracking.distillation.teachers import build_cohort_teacher_bank
from mjlab.tasks.tracking.distillation.trainer import (
  TrainingConfig,
  VaeDistillationTrainer,
)
from mjlab.tasks.tracking.distillation.vae_config import DEFAULT_MODEL_SETTINGS
from mjlab.tasks.tracking.distillation.worker import WorkerPool, worker_specs

_COMMANDS = (
  "validate-teachers",
  "train",
  "evaluate",
  "evaluate-cohort",
  "play",
  "export",
)
SamplingMode = Literal["start", "uniform"]
ResetProfile = Literal["standing-start", "standing-window"]
ResetPerturbations = Literal["configured", "clean"]
"""Reset-perturbation handling for a standing evaluation profile.

``configured`` keeps the task's own pose/velocity/joint-position randomization,
which is what training sees.  ``clean`` zeroes those reset ranges so the
standing entry is measured without startup randomization.  Neither choice
disables sensor noise, observation delay, actuator randomization, or domain
randomization events: those are properties of the task, not of the reset.
"""
_STANDING_EVALUATION_WINDOW_FRAMES = 25
"""Early-window frames for the standing evaluation profiles.

Matches :data:`MotionCommandCfg.standing_start_window_frames`, so a standing
profile asks about the same transition window the training mixture targets.
"""
CohortEvaluationMode = Literal["teacher", "student", "both"]
ReportBoundaries = Literal["summary", "full"]
PlayViewer = Literal["viser", "native", "auto"]
PROVENANCE_VERSION = 1
"""Version of the ``resolved_config`` record written into checkpoints."""

_DEFAULT_TEACHER_ID = "tennis_000"
"""Single-teacher default used when neither selection flag is supplied."""

_COHORT_PHASE_POLICY = "uniform"
"""Phase policy of an M4 training build; pinned evaluation overrides it."""

_WORKER_SEED_SCHEME_VERSION = 1
"""Version of the per-worker environment and generator seeding derivation.

Recorded in the resolved configuration so a future change to the derivation is
refused as a resume mismatch instead of silently collecting a different data
mix under the same stored settings.
"""

_MAX_WORKERS = 8
"""Upper bound on the ``--worker-devices`` entries one training run accepts."""

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

_COHORT_RESUME_INVARIANT_KEYS = (
  "manifest",
  "teacher_ids",
  "cohort_digest",
  "mapping_digest",
  "phase_policy",
  "task",
  "task_id",
  "runtime",
  "trainer",
  "replay",
  "model",
)
"""Cohort entries a version-2 resume must reproduce; only the budget grows.

The member identities, artifact digests, clip extents, slot/phase policy,
replay partition policy, and seed/device resources live in the checkpoint's
stored cohort record and are compared strictly by ``resume_cohort`` before this
list is consulted, so this list only pins the remaining runner/trainer/model
settings that are not part of the cohort contract.
"""


def _print_help(stream) -> None:
  print("usage: distill <COMMAND> [OPTIONS]", file=stream)
  print(file=stream)
  print("Commands:", file=stream)
  print(
    "  validate-teachers  Validate a teacher manifest and check native/ONNX parity.",
    file=stream,
  )
  print(
    "  train              Run a bounded M3 single-teacher or M4 shared cohort run.",
    file=stream,
  )
  print("  evaluate           Run bounded teacher/student evaluation.", file=stream)
  print(
    "  evaluate-cohort    Pin every selected motion and report per motion.",
    file=stream,
  )
  print(
    "  play               Play a checkpointed student in a Viser/native viewer.",
    file=stream,
  )
  print(
    "  export              Export an audited checkpoint into a v2 VAE bundle.",
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
  print(
    "  'train --teacher-ids A B' trains one shared student over a multi-motion",
    file=stream,
  )
  print(
    "  cohort; 'train' also accepts the mutually exclusive singular --teacher-id.",
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


def _checkpoint_version(path: Path) -> int | None:
  """On-disk envelope version of a checkpoint, without loading a model.

  The version tag is the stable part of the saved envelope.  A payload whose
  version is neither known value is reported as ``None`` so the corresponding
  loader raises its own malformed-artifact error instead of this probe
  guessing which format the file claims to be.
  """
  try:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
  except OSError as exc:
    raise DistillationError(f"cannot read checkpoint {path}: {exc}") from exc
  if not isinstance(payload, Mapping):
    return None
  version = payload.get("version")
  if isinstance(version, bool) or not isinstance(version, int):
    return None
  return (
    version
    if version
    in (
      CHECKPOINT_VERSION,
      COHORT_CHECKPOINT_VERSION,
      STANDING_COHORT_CHECKPOINT_VERSION,
    )
    else None
  )


def _is_cohort_checkpoint(path: Path) -> bool:
  """True when the checkpoint is a version-2 multi-teacher cohort artifact."""
  return _checkpoint_version(Path(path)) in (
    COHORT_CHECKPOINT_VERSION,
    STANDING_COHORT_CHECKPOINT_VERSION,
  )


def _load_member_inference(
  checkpoint: Path,
  cohort: CohortContract,
  teacher_id: str,
  *,
  device: str,
  expected_schema=None,
  reset_profile: str | None = None,
):
  """Load one checked member of a saved M4 cohort for inference only.

  The member must exist in the stored cohort, and its artifact digests, clip
  extent, saved reference digest, and common action/control/observation
  contract must match the live manifest.  The stored cohort identity is kept as
  provenance, so a pinned report never loses the identity of the shared
  student it evaluated.
  """
  return load_cohort_member_inference(
    checkpoint,
    cohort,
    teacher_id,
    device=device,
    expected_schema=expected_schema,
    reset_profile=reset_profile,
  )


def _load_student_inference(
  checkpoint: Path,
  cohort: CohortContract,
  teacher_id: str,
  *,
  device: str,
  expected_schema=None,
):
  """Dispatch to the version-1 or version-2 model-only loader by envelope.

  Legacy callers of ``load_inference_checkpoint`` keep its exact behavior and
  relaxation; a version-2 artifact is loaded through the checked cohort member
  selection instead, so a pinned report always names the member it evaluated.
  """
  if _is_cohort_checkpoint(checkpoint):
    return _load_member_inference(
      checkpoint,
      cohort,
      teacher_id,
      device=device,
      expected_schema=expected_schema,
    )
  return load_inference_checkpoint(
    checkpoint,
    device=device,
    expected_schema=expected_schema,
    expected_teacher_hashes=cohort.teacher(teacher_id).hashes,
    expected_control_contract=_control_metadata(cohort, teacher_id),
  )


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


def _reset_policy(
  *,
  reset_policy: ResetPolicyKind,
  standing_start_fraction: float,
  standing_start_window_frames: int,
  standing_start_frame_zero_fraction: float,
) -> ResetPolicy:
  """Validate standing options before resolving artifacts or building a scene."""
  return make_reset_policy(
    kind=reset_policy,
    standing_start_fraction=standing_start_fraction,
    standing_start_window_frames=standing_start_window_frames,
    standing_start_frame_zero_fraction=standing_start_frame_zero_fraction,
  )


def _reset_options_requested(policy: ResetPolicy) -> bool:
  """Whether CLI values opt into new reset semantics."""
  return policy != ResetPolicy()


def _evaluation_reset_policy(profile: ResetProfile, window_frames: int) -> ResetPolicy:
  """Build the singleton standing profile through the training reset contract."""
  return _reset_policy(
    reset_policy="standing-mixture",
    standing_start_fraction=1.0,
    standing_start_window_frames=window_frames,
    standing_start_frame_zero_fraction=1.0 if profile == "standing-start" else 0.0,
  )


def _apply_reset_perturbations(live: Any, mode: ResetPerturbations) -> None:
  """Apply the reset-perturbation choice to one private live command config.

  ``clean`` zeroes the reset pose, velocity, and joint-position ranges.  It
  changes only the private evaluation copy of the command; the saved task and
  the teacher's own configuration are untouched, and the standing branch it
  measures is otherwise identical.  A private zeroed range is written as a
  zero-width range rather than by deleting the key, so the config keeps the
  same keys the audit compares.
  """
  if mode == "configured":
    return
  cfg = live.cfg
  cfg.pose_range = {axis: (0.0, 0.0) for axis in cfg.pose_range}
  if not cfg.pose_range:
    cfg.pose_range = {axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch")}
  cfg.velocity_range = {axis: (0.0, 0.0) for axis in cfg.velocity_range}
  cfg.joint_position_range = (0.0, 0.0)


def _reset_provenance_report(live: Any, mode: ResetPerturbations) -> dict[str, Any]:
  """Report the resolved reset perturbations of one live standing command.

  Read from the live command after the choice was applied, so the report states
  what the rollout actually used.  ``clean`` is reported only when every reset
  range is exactly zero; a partially zeroed range would otherwise read as a
  clean start when it is not.
  """
  cfg = live.cfg
  pose_range = {
    axis: [float(bounds[0]), float(bounds[1])]
    for axis, bounds in cfg.pose_range.items()
  }
  velocity_range = {
    axis: [float(bounds[0]), float(bounds[1])]
    for axis, bounds in cfg.velocity_range.items()
  }
  joint_range = [
    float(cfg.joint_position_range[0]),
    float(cfg.joint_position_range[1]),
  ]
  all_ranges = [*pose_range.values(), *velocity_range.values(), joint_range]
  return {
    "mode": mode,
    "clean": all(bounds[0] == 0.0 and bounds[1] == 0.0 for bounds in all_ranges),
    "pose_range": pose_range,
    "velocity_range": velocity_range,
    "joint_position_range": joint_range,
    "scope": (
      "reset initialization only; sensor noise, observation delay, actuator "
      "randomization and domain randomization events are unchanged"
    ),
  }


_REFERENCE_RESET_POLICY = ResetPolicy()


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


def _resolve_worker_devices(
  worker_devices: tuple[str, ...] | None, *, num_envs: int
) -> tuple[str, ...]:
  """Validate and canonicalize the opt-in worker device list.

  ``None`` keeps the single-process path: one environment holding all of
  ``num_envs`` on the trainer device.  An explicit list names one device per
  sharded collection worker and splits ``num_envs`` evenly across them, so the
  collected row count per iteration is unchanged and only the parallelism is.
  Every refusal below happens before a simulator exists.

  Entries are canonicalized through ``torch.device``, which needs no CUDA
  runtime: an unparseable device string is refused here rather than surfacing
  from a worker later, and an omitted CUDA index is normalized so ``cuda`` and
  ``cuda:0`` are recognized as the same device.  Entries must otherwise be
  exactly what ``torch.device`` accepts (no case or zero-padded aliases), which
  is the same requirement the trainer device already carries.  Repeated CPU
  entries are deliberately allowed: a CPU shard contends for no physical device,
  and repeating ``'cpu'`` is how an N-worker merge path is exercised without
  GPUs.
  """
  if worker_devices is None:
    return ()
  devices = tuple(worker_devices)
  if not devices:
    raise ValueError("--worker-devices needs at least one device")
  canonical: list[str] = []
  for device in devices:
    if not isinstance(device, str) or not device.strip():
      raise ValueError(
        f"--worker-devices entries must be non-empty device strings; got {device!r}"
      )
    try:
      resolved = torch.device(device)
    except (RuntimeError, ValueError) as exc:
      raise ValueError(
        f"--worker-devices entry {device!r} is not a device: {exc}"
      ) from exc
    if resolved.type == "cuda" and resolved.index is None:
      resolved = torch.device("cuda", 0)
    canonical.append(str(resolved))
  repeated = sorted(
    {
      device
      for device in canonical
      if canonical.count(device) > 1 and torch.device(device).type != "cpu"
    }
  )
  if repeated:
    raise ValueError(
      f"--worker-devices repeats a device: {repeated}; one worker per device"
    )
  if len(canonical) > _MAX_WORKERS:
    raise ValueError(
      f"--worker-devices accepts at most {_MAX_WORKERS} workers; got {len(canonical)}"
    )
  if num_envs % len(canonical) != 0:
    raise ValueError(
      f"--num-envs {num_envs} does not split evenly over the {len(canonical)} "
      "worker devices in --worker-devices; every worker must own the same "
      "number of environments"
    )
  return tuple(canonical)


def _workers_identity(worker_devices: tuple[str, ...], num_envs: int) -> dict[str, Any]:
  """The worker identity one resume must reproduce, as a single object.

  A single-process run records ``mode: 'single'`` with no devices, so the record
  states that no worker exists instead of leaving it to be inferred from an
  empty list.  Keeping the whole identity in one object is what makes the
  legacy default safe: a checkpoint recorded before workers existed is defaulted
  as a whole, so a partially populated worker record can never be read as the
  single-process case.
  """
  count = max(len(worker_devices), 1)
  return {
    "mode": "single" if not worker_devices else "sharded",
    "devices": list(worker_devices),
    "count": count,
    "envs_per_worker": num_envs // count,
    "seed_scheme": _WORKER_SEED_SCHEME_VERSION,
  }


def _collection_transport(worker_devices: tuple[str, ...]) -> str:
  """Audit-only name of the path collected rows take to the trainer.

  Recorded in the resolved configuration but deliberately kept out of the
  resume invariants: the transport does not change which rows are sampled, so
  replacing it with a faster one must not refuse an existing run's resume.  It
  sits with ``checkpoint_every`` on the audit-only side of that distinction.
  """
  return "in-process" if not worker_devices else "pinned-host-v1"


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


def _resume_compatible_value(key: str, stored: Any, requested: Any) -> Any:
  """Stored value with the documented default for late-added sub-keys.

  A checkpoint recorded before a sub-key existed inside a stored entry stored
  the older entry verbatim; comparing that verbatim against the newer requested
  entry would treat the mere absence of the new option as a settings change.
  The documented default is what the old run actually used, so the missing
  sub-key is filled in on the stored side before the comparison.  Currently:
  ``replay.phase_bins`` (cohort entries only) defaults to 0 in checkpoints
  that predate the option; single-teacher replay entries never carry it, so
  they are compared verbatim.  Likewise ``runtime.workers`` defaults to the
  single-process identity in cohort checkpoints recorded before multi-worker
  collection existed, because a single-process run is exactly what those
  checkpoints used.  That default is applied to the whole worker object, never
  per field, so a partially populated worker record cannot be mistaken for the
  single-process case; a checkpoint that records workers is compared verbatim.
  """
  if (
    key == "replay"
    and isinstance(stored, Mapping)
    and isinstance(requested, Mapping)
    and "phase_bins" in requested
  ):
    stored_equivalent = dict(stored)
    stored_equivalent.setdefault("phase_bins", 0)
    return stored_equivalent
  if (
    key == "runtime"
    and isinstance(stored, Mapping)
    and isinstance(requested, Mapping)
    and "workers" in requested
  ):
    # Only a cohort runtime carries the worker identity, so the requested side
    # decides whether this entry is one of them; a single-teacher runtime entry
    # never carries it and is compared verbatim.  The default is the whole
    # single-process object, never a per-field fill.
    stored_equivalent = dict(stored)
    stored_num_envs = stored.get("num_envs")
    if isinstance(stored_num_envs, int) and not isinstance(stored_num_envs, bool):
      stored_equivalent.setdefault("workers", _workers_identity((), stored_num_envs))
    return stored_equivalent
  return stored


def _check_resume_compatibility(
  state: LifecycleState | CohortLifecycleState,
  requested: Mapping[str, Any],
  *,
  invariant_keys: tuple[str, ...] = _RESUME_INVARIANT_KEYS,
) -> dict[str, Any]:
  """Refuse a resume that would silently change stored settings/schedules.

  Every stored semantic entry listed in ``invariant_keys`` must be reproduced;
  only the total lifetime iteration budget may grow.  A checkpoint that
  predates this provenance record is reported as unverified rather than being
  failed against fields it never stored, and any entry the checkpoint *did*
  record is still compared.
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
  for key in invariant_keys:
    if key not in stored:
      audit["unverified"].append(key)
      continue
    audit["checked"].append(key)
    if _resume_compatible_value(key, stored[key], requested[key]) != requested[key]:
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


def _member_inference_report(inference) -> dict[str, Any] | None:
  """Model-only identity of one checked member of a saved M4 cohort.

  A pinned report keeps the whole stored cohort identity next to the member it
  evaluated, so the shared student's trained cohort is never reduced to the one
  pinned motion.  ``relocated_artifact_roles`` names artifacts accepted only by
  content digest, so a moved inference asset is visible instead of implicit.
  """
  if inference is None:
    return None
  report = _checkpoint_model_report(inference)
  assert report is not None
  return {
    **report,
    "checkpoint_version": COHORT_CHECKPOINT_VERSION,
    "cohort_digest": inference.cohort.digest(),
    "trained_teacher_ids": list(inference.cohort.teacher_ids),
    "mapping_digest": inference.cohort.mapping_digest,
    "requested_teacher_id": inference.requested_teacher_id,
    "motion_id": inference.motion_id,
    "teacher_code": inference.teacher_code,
    "artifact_hashes": dict(inference.artifact_hashes),
    "relocated_artifact_roles": list(inference.relocated_artifact_roles),
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


def _build_cohort_runner(
  *,
  cohort: CohortContract,
  setup: CohortSetup,
  teacher_ids: tuple[str, ...],
  device: str,
  num_envs: int,
  task_id: str | None,
  replay_capacity: int,
  phase_bins: int,
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
  reset_policy: ResetPolicy,
):
  """Build the M4 shared-student cohort run: one adapter, bank, and replay.

  The environment and the student come from ``setup``, the same recipe a sharded
  collection worker calls, so the single-process path and every worker build one
  environment by construction rather than by convention.

  One environment carries every selected clip (its rows keep their own
  reference), one frozen ``TeacherBank`` labels the mixed rows by their per-row
  codes, and one per-motion balanced replay partitions the capacity by the
  manifest weights.  The cohort identity is built from the live adapter and
  replay, so it records what was actually constructed rather than a summary the
  caller supplied.

  Cheap budgets and the selection are validated *before* any environment is
  constructed, and everything that runs after the adapter exists is wrapped so a
  later failure closes the already-built environment exactly once instead of
  leaking it (the caller has no runner to close in that case).
  """
  _validate_cohort_run(cohort, teacher_ids, replay_capacity)
  # Pure-data configuration validates before a simulator exists.
  training_config = TrainingConfig(
    learning_rate=learning_rate,
    beta=beta,
    accumulation_steps=accumulation_steps,
    minibatch_size=minibatch_size,
  )
  runner_config = RunnerConfig(
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
  )
  # The adapter and student come from the shared cohort recipe, so a worker
  # builds the environment this path builds; worker index 0 is the
  # single-process derivation and leaves the requested seed unchanged.
  adapter = setup.build_adapter(device=device, num_envs=num_envs)
  try:
    # Model initialization only: environment startup randomization is seeded
    # inside the adapter factory above, before the private env config is
    # constructed.
    model = setup.build_student(schema=adapter.schema, device=device)
    weights = {
      clip.motion_id: float(cohort.teacher(clip.teacher_id).entry.sampling_weight)
      for clip in adapter.library.clips
    }
    replay = BalancedReplayBuffer(
      replay_capacity,
      adapter.schema,
      weights,
      teacher_codes=dict(adapter.motion_teacher_codes),
      frame_counts={clip.motion_id: int(clip.frames) for clip in adapter.library.clips},
      device=torch.device(device),
      dtype=torch.float32,
      phase_bins=phase_bins,
    )
    identity = cohort_identity_from_adapter(adapter, replay=replay)
    trainer = VaeDistillationTrainer(
      model,
      replay,
      training_config,
      seed=seed,
      teacher=adapter.bank,
    )
    collector = DAggerCollector(adapter, adapter.bank, model, replay)
    runner = DistillationRunner(collector, trainer, runner_config)
  except BaseException:
    # The caller never received a runner, so this is the only chance to release
    # the environment the adapter already built.
    adapter.close()
    raise
  sampling_mode = adapter.env.command_manager.get_term("motion").cfg.sampling_mode
  return runner, adapter, identity, sampling_mode, _seed_audit(adapter)


def _build_sharded_cohort_runner(
  *,
  cohort: CohortContract,
  setup: CohortSetup,
  teacher_ids: tuple[str, ...],
  device: str,
  num_envs: int,
  worker_devices: tuple[str, ...],
  replay_capacity: int,
  phase_bins: int,
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
  """Build the sharded M4 cohort run: N worker environments, one parent replay.

  The parent owns no environment in this layout, so the schema, the reference
  library, the audited slot allocation and the motion-to-teacher codes come from
  the workers that did build one.  Everything the run does with them — the
  replay, the trainer, the cohort identity, the checkpoints — is the same code
  the single-process path uses, on the same values, so a sharded run and a
  single-process run of one recipe record the same cohort identity.

  Returns the runner, the collection source, the identity, the sampling mode and
  the seed audit, exactly like the single-process builder.
  """
  _validate_cohort_run(cohort, teacher_ids, replay_capacity)
  training_config = TrainingConfig(
    learning_rate=learning_rate,
    beta=beta,
    accumulation_steps=accumulation_steps,
    minibatch_size=minibatch_size,
  )
  runner_config = RunnerConfig(
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
  )
  specs = worker_specs(setup, worker_devices, num_envs=num_envs)
  pool = WorkerPool(specs)
  source = None
  try:
    pool.start()
    descriptions = pool.describe()
    primary = _require_one_shard_contract(descriptions, setup)
    torch.manual_seed(seed)
    model = ConditionalVAE(primary.schema, DEFAULT_MODEL_SETTINGS).to(device)
    weights = {
      clip.motion_id: float(cohort.teacher(clip.teacher_id).entry.sampling_weight)
      for clip in primary.library.clips
    }
    replay = BalancedReplayBuffer(
      replay_capacity,
      primary.schema,
      weights,
      teacher_codes=dict(primary.motion_teacher_codes or {}),
      frame_counts={clip.motion_id: int(clip.frames) for clip in primary.library.clips},
      device=torch.device(device),
      dtype=torch.float32,
      phase_bins=phase_bins,
    )
    identity = cohort_identity_from_parts(
      cohort=cohort,
      library=primary.library,
      audit=primary.audit,
      replay=replay,
      device=device,
    )
    trainer = VaeDistillationTrainer(
      model,
      replay,
      training_config,
      seed=seed,
      teacher=build_cohort_teacher_bank(cohort, device=device),
    )
    source = ShardedCollection(
      pool, replay=replay, model=model, descriptions=descriptions
    )
    runner = DistillationRunner(None, trainer, runner_config, sharded=source)
  except BaseException:
    # The caller never received a source, so releasing the pool is ours to do.
    if source is not None:
      source.close()
    else:
      pool.close()
    raise
  return (
    runner,
    source,
    identity,
    primary.sampling_mode,
    _seed_audit(SimpleNamespace(audit=primary.audit)),
  )


def _cohort_resolved_config(
  *,
  manifest: Path,
  repo_root: Path,
  sampling_mode: str,
  reset_policy_dict: Mapping[str, Any],
  base_task: str,
  identity: CohortIdentity,
  task_id: str | None,
  device: str,
  num_envs: int,
  worker_devices: tuple[str, ...] = (),
  seed: int,
  seed_audit: Mapping[str, Any],
  runner: DistillationRunner,
  resume: Path | None,
  checkpoint_every: int,
) -> dict[str, Any]:
  """Complete resolved M4 configuration recorded in every cohort checkpoint.

  The member identities and their artifact digests are not repeated here: they
  are the checkpoint's stored cohort record, which ``resume_cohort`` compares
  strictly.  This record pins what the cohort record does not carry (trainer
  settings, model settings, replay partition shape, and the resolved schedule),
  using the same ``checkpoint_every``-is-audit-only rule as the single-teacher
  path so a resumed run may change its save cadence.
  """
  trainer = runner.trainer
  replay = runner.replay
  assert isinstance(replay, BalancedReplayBuffer)
  quota = replay.quota
  return {
    "provenance_version": PROVENANCE_VERSION,
    "command": "train",
    "manifest": str(manifest),
    "repo_root": str(repo_root),
    "teacher_ids": list(identity.teacher_ids),
    "cohort_digest": identity.digest(),
    "mapping_digest": identity.mapping_digest,
    "phase_policy": identity.slots.phase_policy,
    "reset_policy": dict(reset_policy_dict),
    "task": base_task,
    "task_id": task_id,
    "runtime": {
      "device": device,
      "num_envs": num_envs,
      "seed": seed,
      "resolved_seed": seed_audit["effective_seed"],
      "seed_provenance": dict(seed_audit),
      "workers": _workers_identity(worker_devices, num_envs),
    },
    "execution": {"transport": _collection_transport(worker_devices)},
    "schedule": _schedule_config(runner, sampling_mode),
    "checkpoint_every": checkpoint_every,
    "trainer": {
      "learning_rate": trainer.config.learning_rate,
      "beta": trainer.config.beta,
      "accumulation_steps": trainer.config.accumulation_steps,
      "minibatch_size": trainer.config.minibatch_size,
      "latent_mode": trainer.config.latent_mode,
    },
    "replay": {
      "kind": "balanced-motion-replay",
      "capacity": replay.capacity,
      "weights": list(replay.weights),
      "quotas": list(quota.quotas),
      "motion_ids": list(replay.motion_ids),
      "phase_bins": replay.phase_bins,
    },
    "model": {
      "settings": trainer.model.settings.to_metadata(),
      "schema": trainer.model.schema_metadata,
    },
    "resumed_from": None if resume is None else str(resume),
  }


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


def _boundary_record(boundary) -> dict[str, Any]:
  """Plain-data boundary record, with the shard tag only when it has one.

  A single-process run has one environment batch, so tagging each record with a
  null worker would change an unchanged report while adding no information.
  A merged record keeps the tag, because its environment indices belong to one
  shard and cannot be read as global indices.
  """
  record = asdict(boundary)
  if record.get("worker_index") is None:
    record.pop("worker_index", None)
  return record


def _boundaries_report(collection, mode: ReportBoundaries):
  """Boundary evidence for one iteration: compact summary, or raw detail."""
  if mode == "full":
    return [_boundary_record(item) for item in collection.boundaries]
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
      "motion_stats": [stats.as_dict() for stats in collection.motion_stats],
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


def _release(owner) -> Callable[[], None]:
  """Return a callable that releases the run's owner, at most once.

  The single-process layout owns one adapter; the sharded layout owns a pool of
  worker processes.  Both are released by ``close``, so callers that must clean
  up on failure or on a signal do not branch on which layout is running.

  Releasing twice must be impossible rather than merely unexpected: a signal
  handler releases the owner before raising, and the surrounding ``finally``
  releases it again, so a second ``close`` would run against a live environment.
  An exception raised there would also replace the ``SystemExit`` that carries
  the operator's exit status, turning a clean stop into an opaque failure.  The
  returned action is therefore built once and shared by both sites.
  """
  close = getattr(owner, "close", None)
  if not callable(close):
    return lambda: None
  released = False

  def release() -> None:
    nonlocal released
    if released:
      return
    released = True
    close()

  return release


@contextmanager
def _teardown_on_signal(teardown):
  """Run ``teardown`` when an operator or scheduler stops the run.

  A sharded run holds worker processes on collection devices.  Without this,
  Ctrl-C or a scheduler's ``SIGTERM`` would leave them to their own watchdog
  rather than closing them before the process exits, which is the difference
  between releasing a GPU promptly and holding it while a collection finishes.
  Handlers are restored on exit, so the surrounding process keeps its own
  disposition.
  """
  previous: dict[int, Any] = {}

  def handler(signum, _frame):
    teardown()
    raise SystemExit(128 + signum)

  for signum in (signal.SIGINT, signal.SIGTERM):
    try:
      previous[signum] = signal.getsignal(signum)
      signal.signal(signum, handler)
    except ValueError:
      # Not the main thread: the caller's own teardown still runs.
      pass
  try:
    yield
  finally:
    for signum, original in previous.items():
      signal.signal(signum, original)


def _require_one_shard_contract(descriptions: Sequence[Any], setup: CohortSetup) -> Any:
  """Require every shard to describe the same cohort contract as shard 0.

  The parent takes the replay's clip weights and frame counts, the
  motion-to-teacher routing, and the whole cohort identity from worker 0, so a
  shard that disagrees about the library, the audited slots, the phase policy,
  the routing or the reset contract would have the model trained on one
  contract while that shard collected under another.  Comparing the observation
  schema alone left all of that unverified.

  The environment seed is deliberately not compared for equality: each shard's
  environment is built on its own stratum by construction, so its recorded seed
  differs by design.  What is checked instead is that each shard's environment
  actually carries the seed its index derives, which is the invariant the cohort
  identity's single seed record and a sharded resume both rest on.

  Returns:
    The description every shard agreed with.

  Raises:
    DistillationError: if any shard disagrees about any of the above.
  """
  primary = descriptions[0]
  fields = (
    ("observation schema", lambda item: item.schema),
    ("reference library clips", lambda item: item.library.clips),
    ("library body selection", lambda item: item.library.body_selection),
    ("library source body count", lambda item: item.library.source_body_count),
    ("audited slot allocation", lambda item: item.audit.slots),
    ("audited body mapping digest", lambda item: item.audit.mapping_digest),
    ("audited phase policy", lambda item: item.audit.phase_policy),
    ("audited common contract", lambda item: item.audit.asset),
    ("reset policy", lambda item: item.audit.reset_policy),
    ("motion teacher codes", lambda item: item.motion_teacher_codes),
    ("sampling mode", lambda item: item.sampling_mode),
    ("reset policy enablement", lambda item: item.reset_policy_enabled),
  )
  for index, description in enumerate(descriptions):
    for name, read in fields:
      if read(description) != read(primary):
        raise DistillationError(
          f"worker {index} reported a different {name} than worker 0; the "
          "shards do not describe one cohort contract"
        )
    expected_seed = worker_env_seed(setup.base_seed, index)
    recorded = getattr(description.audit.seed_provenance, "effective_seed", None)
    if recorded is not None and recorded != expected_seed:
      raise DistillationError(
        f"worker {index} built its environment on seed {recorded} instead of "
        f"the seed {expected_seed} its shard index derives; the shards are not "
        "the strata this run's identity and resume describe"
      )
  return primary


def _validate_cohort_run(
  cohort: CohortContract, teacher_ids: tuple[str, ...], replay_capacity: int
) -> None:
  """Selection and budget checks shared by both collection layouts.

  Cheap to run and complete before any environment exists, so a typo in a
  teacher id or an impossible replay capacity never costs a simulator build
  under either layout.
  """
  known = {teacher.id for teacher in cohort.teachers}
  unknown = [teacher_id for teacher_id in teacher_ids if teacher_id not in known]
  if unknown:
    raise DistillationError(f"cohort has no teacher(s) {unknown}")
  if not teacher_ids:
    raise DistillationError("a cohort run needs at least one teacher id")
  if len(set(teacher_ids)) != len(teacher_ids):
    raise DistillationError(f"duplicate teacher ids {list(teacher_ids)}")
  if replay_capacity < len(teacher_ids):
    raise DistillationError(
      f"--replay-capacity {replay_capacity} cannot give each of the "
      f"{len(teacher_ids)} selected motions a slot"
    )


def _fail(exc: BaseException) -> int:
  """Report one CLI-level refusal on stderr and return the failure exit code."""
  print(f"[FAIL] {exc}", file=sys.stderr)
  return 1


def _require_train_settings(
  report_boundaries: str, checkpoint_every: int, progress_every: int
) -> None:
  """Validate `train` reporting/save settings before any environment exists."""
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


def _drive_lifecycle(
  runner: DistillationRunner,
  *,
  max_iterations: int,
  checkpoint_every: int,
  progress_every: int,
  output_dir: Path,
  report_boundaries: ReportBoundaries,
  write_checkpoint,
  device: str,
  num_envs: int,
) -> tuple[list[dict], list[Path]]:
  """Drive one bounded runner and return its per-iteration reports and checkpoints.

  The existing bounded runner is driven in chunks and its one save call is
  reused, so a periodic checkpoint carries the same provenance/schedule
  metadata as the final one and is an ordinary resume input.  Filenames use
  ``runner.iteration``, the total lifetime counter, so a resumed run continues
  the same sequence instead of colliding with earlier files.  Each iteration is
  converted to its report form as soon as it returns, so a summary report never
  holds more than one iteration's boundary arrays.  This driver is shared by the
  single-teacher and shared-student cohort runs; only the save call differs.
  """
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
      iteration_report = _iteration_report(runner.run_iteration(), report_boundaries)
      if isinstance(runner.replay, BalancedReplayBuffer):
        iteration_report["replay"] = runner.replay.report().as_dict()
      iteration_reports.append(iteration_report)
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
  return iteration_reports, checkpoints


def _train_single(
  manifest: Path = Path("configs/distillation/x2_tennis.yaml"),
  repo_root: Path = Path("."),
  teacher_id: str = _DEFAULT_TEACHER_ID,
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
  reset_policy: ResetPolicy = _REFERENCE_RESET_POLICY,
  output_dir: Path | None = None,
  resume: Path | None = None,
) -> int:
  """Run the bounded native single-teacher (M3) collect/update lifecycle."""
  runner = None
  try:
    _require_train_settings(report_boundaries, checkpoint_every, progress_every)
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

    iteration_reports, checkpoints = _drive_lifecycle(
      runner,
      max_iterations=max_iterations,
      checkpoint_every=checkpoint_every,
      progress_every=progress_every,
      output_dir=output_dir,
      report_boundaries=report_boundaries,
      write_checkpoint=write_checkpoint,
      device=device,
      num_envs=num_envs,
    )
    checkpoint = checkpoints[-1]
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


def _train_cohort(
  *,
  manifest: Path,
  repo_root: Path,
  teacher_ids: tuple[str, ...],
  task_id: str | None,
  device: str,
  num_envs: int,
  max_iterations: int,
  bootstrap_steps: int,
  collection_steps: int,
  updates_per_iteration: int,
  teacher_probability: float,
  evaluate_every: int,
  evaluation_steps: int,
  minibatch_size: int,
  accumulation_steps: int,
  replay_capacity: int,
  phase_bins: int,
  worker_devices: tuple[str, ...] = (),
  learning_rate: float,
  beta: float,
  seed: int,
  rollout_latent: RolloutLatent,
  checkpoint_every: int,
  progress_every: int,
  report_boundaries: ReportBoundaries,
  output_dir: Path | None,
  resume: Path | None,
  reset_policy: ResetPolicy,
) -> int:
  """Run the bounded shared-student M4 cohort collect/update lifecycle.

  One environment carries every selected clip, one frozen bank labels rows by
  their per-row codes, and one per-motion balanced replay holds the capacity.
  The checkpoint is a version-2 cohort artifact, so it is saved and resumed
  through the cohort lifecycle; a version-1 artifact is refused before any
  environment is constructed instead of being reinterpreted.
  """
  runner = None
  primary = None
  # Bound before the run is built so the final cleanup can always call it: an
  # error between acquiring the owner and reaching the drive loop must not turn
  # into a cleanup failure that hides the original one.
  release: Callable[[], None] = _release(None)
  try:
    _require_train_settings(report_boundaries, checkpoint_every, progress_every)
    if resume is not None and not _is_cohort_checkpoint(resume):
      raise DistillationError(
        "--teacher-ids requires a version-2 M4 cohort checkpoint to resume; "
        f"{resume} is not one. Use --teacher-id for a version-1 resume, or "
        "start a new cohort run"
      )
    cohort = _resolve(manifest, repo_root)
    setup = CohortSetup(
      manifest=manifest,
      repo_root=repo_root,
      teacher_ids=tuple(teacher_ids),
      task_id=task_id,
      phase_policy=_COHORT_PHASE_POLICY,
      reset_policy=reset_policy,
      base_seed=seed,
    )
    if worker_devices:
      primary_build = _build_sharded_cohort_runner(
        cohort=cohort,
        setup=setup,
        teacher_ids=teacher_ids,
        device=device,
        num_envs=num_envs,
        worker_devices=worker_devices,
        replay_capacity=replay_capacity,
        phase_bins=phase_bins,
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
      # The workers built their environments from exactly this policy, and
      # their live audit is what the cohort identity records.
      reset_policy_dict = reset_policy.as_dict()
    else:
      primary_build = _build_cohort_runner(
        cohort=cohort,
        setup=setup,
        teacher_ids=teacher_ids,
        device=device,
        num_envs=num_envs,
        task_id=task_id,
        replay_capacity=replay_capacity,
        phase_bins=phase_bins,
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
        reset_policy=reset_policy,
      )
      reset_policy_dict = getattr(
        primary_build[1].env.command_manager.get_term("motion"),
        "reset_policy",
        ResetPolicy(),
      ).as_dict()
    runner, primary, identity, evaluation_sampling_mode, seed_audit = primary_build
    release = _release(primary)
    _require_requested_seed_applied(seed, seed_audit)
    assert isinstance(runner.replay, BalancedReplayBuffer)
    resolved_config = _cohort_resolved_config(
      manifest=manifest,
      repo_root=repo_root,
      sampling_mode=evaluation_sampling_mode,
      reset_policy_dict=reset_policy_dict,
      base_task=cohort.manifest.base_task,
      identity=identity,
      task_id=task_id,
      device=device,
      num_envs=num_envs,
      worker_devices=worker_devices,
      seed=seed,
      seed_audit=seed_audit,
      runner=runner,
      resume=resume,
      checkpoint_every=checkpoint_every,
    )
    resume_audit = None
    if resume is not None:
      # ``resume_cohort`` reproduces the whole stored cohort record strictly
      # (members, digests, clip extents, slot/phase policy, replay partitions,
      # and seed/device resources) before the runner settings below are checked.
      state = runner.resume_cohort(str(resume), identity, map_location=device)
      resume_audit = _check_resume_compatibility(
        state, resolved_config, invariant_keys=_COHORT_RESUME_INVARIANT_KEYS
      )
    if output_dir is None:
      output_dir = _default_train_output_dir(task_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    def write_checkpoint(path: Path) -> None:
      runner.save_cohort(
        str(path),
        identity,
        resolved_config=resolved_config,
        schedule=resolved_config["schedule"],
      )

    # One release action for the whole run: the teardown-on-signal path and the
    # final cleanup both call it, and it must close the owner exactly once.  It
    # is bound here, next to the owner it releases, so the final cleanup never
    # has to reconstruct it from a partially built run.
    release = _release(primary)
    with _teardown_on_signal(release):
      iteration_reports, checkpoints = _drive_lifecycle(
        runner,
        max_iterations=max_iterations,
        checkpoint_every=checkpoint_every,
        progress_every=progress_every,
        output_dir=output_dir,
        report_boundaries=report_boundaries,
        write_checkpoint=write_checkpoint,
        device=device,
        num_envs=num_envs,
      )
    checkpoint = checkpoints[-1]
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
        "workers": dict(resolved_config["runtime"]["workers"]),
        "transport": resolved_config["execution"]["transport"],
      },
      "schedule": resolved_config["schedule"],
      "resolved_config": resolved_config,
      "resume": resume_audit,
      "replay": runner.replay.report().as_dict(),
      "cohort": {
        "teacher_ids": list(identity.teacher_ids),
        "cohort_digest": identity.digest(),
        "mapping_digest": identity.mapping_digest,
        "phase_policy": identity.slots.phase_policy,
        "slot_counts": list(identity.slots.counts),
        "slot_weights": list(identity.slots.weights),
        "num_envs": identity.slots.num_envs,
        "replay_motion_ids": list(identity.replay.motion_ids),
        "replay_quotas": list(identity.replay.quotas),
        "clip_frames": [member.frames for member in identity.members],
        "evaluation_sampling_mode": evaluation_sampling_mode,
      },
      "quality": {
        "implementation": "bounded native shared-student cohort lifecycle constructed",
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
    if primary is not None:
      release()


def _train(
  manifest: Path = Path("configs/distillation/x2_tennis.yaml"),
  repo_root: Path = Path("."),
  teacher_id: str | None = None,
  teacher_ids: tuple[str, ...] = (),
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
  phase_bins: int | None = None,
  worker_devices: tuple[str, ...] | None = None,
  learning_rate: float = 5e-4,
  beta: float = 0.01,
  seed: int = 0,
  rollout_latent: RolloutLatent = "mean",
  checkpoint_every: int = 500,
  progress_every: int = 10,
  report_boundaries: ReportBoundaries = "summary",
  reset_policy: ResetPolicyKind = "reference",
  standing_start_fraction: float = 0.25,
  standing_start_window_frames: int = 25,
  standing_start_frame_zero_fraction: float = 0.5,
  output_dir: Path | None = None,
  resume: Path | None = None,
) -> int:
  """Run a bounded native single-teacher or shared-student cohort lifecycle.

  Selection is explicit and mutually exclusive: ``--teacher-id`` (singular)
  keeps the M3 single-teacher path, ``--teacher-ids`` (plural) trains one shared
  student over a multi-motion cohort in one environment, and supplying both is
  refused before any environment is constructed.  This repository's Tyro
  configuration uses Python literal syntax for collections, so the plural form
  is ``--teacher-ids ('tennis_000','tennis_001')``; a single element keeps a
  trailing comma (``--teacher-ids ('tennis_000',)``).  With neither flag the
  single-teacher path uses the historical ``tennis_000`` default, so an existing
  invocation behaves exactly as before.  A cohort run always samples phases
  uniformly and records that policy as an explicit private override, and it
  saves and resumes version-2 cohort checkpoints instead of version-1 ones.

  ``phase_bins`` applies only to the cohort path (``--teacher-ids``): unset
  (the default) keeps the historical density-proportional within-motion draw,
  equivalent to an explicit 0; a positive value splits each clip's
  reference-phase axis into that many cells and equalizes each draw's
  within-motion exposure over the non-empty cells.  It is recorded in the
  resolved configuration and must be reproduced on resume.  Passing it with
  the single-teacher path — including an explicit ``--phase-bins 0`` — is
  refused, and negative values are refused by the replay constructor.

  ``worker_devices`` is the opt-in multi-device collection configuration and
  applies only to the cohort path.  Unset (the default) keeps the historical
  single-process path: one environment holding all of ``num_envs`` on
  ``device``.  An explicit list, e.g. ``--worker-devices "('cuda:1','cuda:2')"``,
  names one device per sharded collection worker and splits ``num_envs`` evenly
  across them, while the trainer, replay and checkpoint writer stay on
  ``device``.  The list is recorded in the resolved configuration and is
  therefore a resume invariant.  A list that repeats a device, exceeds eight
  entries, or does not divide ``num_envs`` evenly is refused before any
  environment is constructed, as is any use on the single-teacher path.

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
  try:
    resolved_reset_policy = _reset_policy(
      reset_policy=reset_policy,
      standing_start_fraction=standing_start_fraction,
      standing_start_window_frames=standing_start_window_frames,
      standing_start_frame_zero_fraction=standing_start_frame_zero_fraction,
    )
  except ValueError as exc:
    return _fail(exc)
  if teacher_ids and teacher_id is not None:
    return _fail(
      ValueError(
        "--teacher-id and --teacher-ids are mutually exclusive: pass exactly one "
        f"selection (got teacher_id={teacher_id!r} and "
        f"teacher_ids={list(teacher_ids)!r})"
      )
    )
  if not teacher_ids and _reset_options_requested(resolved_reset_policy):
    return _fail(
      ValueError(
        "standing reset options require the cohort path; use singleton cohort "
        "syntax --teacher-ids \"('tennis_000',)\" (and omit --teacher-id)"
      )
    )
  resolved_phase_bins = 0 if phase_bins is None else phase_bins
  if not teacher_ids and phase_bins is not None:
    return _fail(
      ValueError(
        "--phase-bins applies only to the cohort path; use singleton cohort "
        "syntax --teacher-ids \"('tennis_000',)\" (and omit --teacher-id)"
      )
    )
  if not teacher_ids and worker_devices is not None:
    return _fail(
      ValueError(
        "--worker-devices applies only to the cohort path; use singleton "
        "cohort syntax --teacher-ids \"('tennis_000',)\" (and omit --teacher-id)"
      )
    )
  try:
    resolved_worker_devices = _resolve_worker_devices(worker_devices, num_envs=num_envs)
  except ValueError as exc:
    return _fail(exc)
  if teacher_ids:
    return _train_cohort(
      manifest=manifest,
      repo_root=repo_root,
      teacher_ids=tuple(teacher_ids),
      task_id=task_id,
      device=device,
      num_envs=num_envs,
      worker_devices=resolved_worker_devices,
      max_iterations=max_iterations,
      bootstrap_steps=bootstrap_steps,
      collection_steps=collection_steps,
      updates_per_iteration=updates_per_iteration,
      teacher_probability=teacher_probability,
      evaluate_every=evaluate_every,
      evaluation_steps=evaluation_steps,
      minibatch_size=minibatch_size,
      accumulation_steps=accumulation_steps,
      replay_capacity=replay_capacity,
      phase_bins=resolved_phase_bins,
      learning_rate=learning_rate,
      beta=beta,
      seed=seed,
      rollout_latent=rollout_latent,
      checkpoint_every=checkpoint_every,
      progress_every=progress_every,
      report_boundaries=report_boundaries,
      output_dir=output_dir,
      resume=resume,
      reset_policy=resolved_reset_policy,
    )
  return _train_single(
    manifest=manifest,
    repo_root=repo_root,
    teacher_id=teacher_id or _DEFAULT_TEACHER_ID,
    task_id=task_id,
    device=device,
    num_envs=num_envs,
    max_iterations=max_iterations,
    bootstrap_steps=bootstrap_steps,
    collection_steps=collection_steps,
    updates_per_iteration=updates_per_iteration,
    teacher_probability=teacher_probability,
    evaluate_every=evaluate_every,
    evaluation_steps=evaluation_steps,
    minibatch_size=minibatch_size,
    accumulation_steps=accumulation_steps,
    replay_capacity=replay_capacity,
    learning_rate=learning_rate,
    beta=beta,
    seed=seed,
    rollout_latent=rollout_latent,
    checkpoint_every=checkpoint_every,
    progress_every=progress_every,
    report_boundaries=report_boundaries,
    output_dir=output_dir,
    resume=resume,
  )


def _student_identity_report(inference) -> dict[str, Any] | None:
  """Model-only student identity for the report, version-dispatched.

  A version-2 cohort artifact reports the member it pinned and the whole stored
  cohort identity it belongs to; a version-1 artifact keeps its original
  model-only report.
  """
  if inference is None:
    return None
  if isinstance(inference, CohortMemberInference):
    return _member_inference_report(inference)
  return _checkpoint_model_report(inference)


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
  reset_profile: ResetProfile | None = None,
  reset_perturbations: ResetPerturbations = "configured",
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

  A version-2 M4 cohort checkpoint is supported through checked member
  selection: ``--teacher-id`` must name a member of the saved cohort, that
  member's artifact digests and common contract are validated against the live
  manifest, and the report keeps the full stored cohort identity next to the
  pinned member.  The environment is still one pinned single-motion simulator,
  so this command evaluates a member; use ``evaluate-cohort`` to pin every
  selected motion in one bounded run.
  """
  adapter = None
  try:
    if sampling_mode not in ("start", "uniform"):
      raise ValueError("sampling_mode must be 'start' or 'uniform'")
    if mode not in ("teacher", "student"):
      raise ValueError("mode must be 'teacher' or 'student'")
    if reset_profile is not None:
      if mode != "student":
        raise ValueError(
          "standing reset profiles are available only for --mode student"
        )
      if sampling_mode != "start":
        raise ValueError(
          "--reset-profile conflicts with explicit --sampling-mode; omit the phase "
          "option when selecting a standing profile"
        )
    elif reset_perturbations != "configured":
      raise ValueError(
        "--reset-perturbations applies only to a standing --reset-profile; "
        "omit it or select a profile"
      )
    if mode == "student" and checkpoint is None:
      raise ValueError("student evaluation requires --checkpoint")
    if (
      reset_profile is not None
      and checkpoint is not None
      and _checkpoint_version(checkpoint) != STANDING_COHORT_CHECKPOINT_VERSION
    ):
      raise DistillationError(
        "--reset-profile requires a version-3 standing cohort checkpoint; "
        "legacy version-1 inference does not carry checked cohort provenance"
      )
    cohort = _resolve(manifest, repo_root)
    member = None
    if mode == "student":
      assert checkpoint is not None
      if _is_cohort_checkpoint(checkpoint):
        # A v2 member is loaded before the simulator because its saved schema is
        # what the live packing must be built from.
        member = _load_member_inference(
          checkpoint, cohort, teacher_id, device=device, reset_profile=reset_profile
        )
    adapter = (
      make_multi_teacher_distillation_adapter(
        cohort,
        (teacher_id,),
        phase_policy="start",
        task_id=task_id,
        num_envs=num_envs,
        device=device,
        schema=None if member is None else member.schema,
        seed=seed,
        reset_policy=_evaluation_reset_policy(
          reset_profile, _STANDING_EVALUATION_WINDOW_FRAMES
        )
        if reset_profile is not None
        else None,
      )
      if reset_profile is not None
      else make_distillation_adapter(
        cohort,
        teacher_id,
        task_id=task_id,
        num_envs=num_envs,
        device=device,
        schema=None if member is None else member.schema,
        seed=seed,
      )
    )
    seed_audit = _seed_audit(adapter)
    _require_requested_seed_applied(seed, seed_audit)
    # This is an evaluation-only sampling override on the private command copy;
    # validate_live_contract has already checked the saved semantic contract.
    motion = adapter.env.command_manager.get_term("motion")
    motion.cfg.sampling_mode = sampling_mode
    if reset_profile is not None:
      # Applied before the first reset, so every reset in this rollout uses the
      # selected perturbation mode rather than only the later ones.
      _apply_reset_perturbations(motion, reset_perturbations)
    student = None
    inference = None
    if mode == "student":
      assert checkpoint is not None
      # Model-only load: the saved optimizer, replay, and collector RNG are not
      # required, and no training tensor is moved to ``device``.  The saved
      # schema/model settings are inferred instead of being re-declared here.
      inference = (
        _load_student_inference(
          checkpoint,
          cohort,
          teacher_id,
          device=device,
          expected_schema=adapter.schema,
        )
        if member is None
        else member
      )
      student = inference.model
    teacher_source = getattr(adapter, "teacher", getattr(adapter, "bank", None))
    if teacher_source is None:
      raise DistillationError("evaluation adapter exposes no teacher bank")
    result = evaluate_distillation(
      adapter,
      teacher_source,
      student,
      mode=mode,
      steps=steps,
      rollout_latent=rollout_latent,
      seed=seed,
      control_period_s=cohort.control.control_period_s,
      standing_trials=reset_profile is not None,
      trial_window_steps=_STANDING_EVALUATION_WINDOW_FRAMES,
    )
    if member is not None:
      # The pinned adapter has local ID zero. Expose the saved cohort identity
      # consistently, including raw segment records, without changing the rollout.
      result = replace(
        result,
        segments=tuple(
          replace(segment, motion_id=member.motion_id, teacher_code=member.teacher_code)
          for segment in result.segments
        ),
        settings={**result.settings, "motion_ids": [member.motion_id]},
      )
    payload = {
      "command": " ".join(sys.argv),
      "status": "evaluation",
      "mode": mode,
      "sampling_mode": sampling_mode,
      "reset_profile": reset_profile,
      "reset_window_frames": (
        None if reset_profile is None else _STANDING_EVALUATION_WINDOW_FRAMES
      ),
      "reset_perturbations": (
        None
        if reset_profile is None
        else _reset_provenance_report(motion, reset_perturbations)
      ),
      "checkpoint": None if checkpoint is None else str(checkpoint),
      "checkpoint_version": (
        None if checkpoint is None else _checkpoint_version(checkpoint)
      ),
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
      "checkpoint_model": _student_identity_report(inference),
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


def _aggregate_motion_metrics(
  per_motion: Mapping[int, Mapping[str, float]],
  weights: Mapping[int, float],
) -> dict[str, Any]:
  """Equal-motion macro and clip-duration-weighted aggregate of one metric set.

  ``macro`` gives every motion the same weight and ``clip_duration_weighted``
  weights each motion by its reference duration in seconds (frames/FPS), so a
  long clip cannot be read as a pooled episode mean and a short clip cannot be
  read as representative.  Each aggregate reports the per-motion values it
  consumed, how many motions contributed to each metric, and any metric present
  in only some motions (reported, never silently dropped).  These are metric
  aggregates over per-motion means, not a quality verdict.
  """
  motion_ids = sorted(per_motion)
  names: set[str] = set()
  for values in per_motion.values():
    names.update(values)
  macro: dict[str, float] = {}
  weighted: dict[str, float] = {}
  contributed: dict[str, int] = {}
  partial: dict[str, list[str]] = {}
  for name in sorted(names):
    macro_values = [
      float(per_motion[motion][name])
      for motion in motion_ids
      if name in per_motion[motion]
    ]
    if len(macro_values) != len(motion_ids):
      partial[name] = [
        str(motion) for motion in motion_ids if name not in per_motion[motion]
      ]
    if macro_values:
      macro[name] = sum(macro_values) / len(macro_values)
      contributed[name] = len(macro_values)
    pairs = [
      (float(weights[motion]), float(per_motion[motion][name]))
      for motion in motion_ids
      if name in per_motion[motion] and weights[motion] > 0.0
    ]
    total_weight = sum(weight for weight, _ in pairs)
    if total_weight > 0.0:
      weighted[name] = sum(weight * value for weight, value in pairs) / total_weight
  return {
    "motion_ids": motion_ids,
    "per_motion": {str(motion): dict(per_motion[motion]) for motion in motion_ids},
    "macro": macro,
    "clip_duration_weighted": weighted,
    "contributed_motions": contributed,
    "metrics_missing_from_some_motion": {
      name: missing for name, missing in sorted(partial.items())
    },
  }


def _combine_outcomes(
  per_label: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
  """Raw outcome counts per motion plus the total, so no failure is hidden."""
  totals: dict[str, int] = {}
  for label in sorted(per_label):
    for outcome, count in per_label[label].items():
      totals[outcome] = totals.get(outcome, 0) + int(count)
  return {
    "per_motion": {label: dict(per_label[label]) for label in sorted(per_label)},
    "total": totals,
  }


def _trial_provenance_summary(result: EvaluationResult) -> dict[str, Any] | None:
  """Bounded per-motion provenance for the trials one evaluation produced.

  A cohort report otherwise keeps only aggregates, so a reader could not see
  which initialization kind and which initial reference frame the trials
  actually used.  This reports counts only - never per-segment arrays - and
  returns ``None`` for a reference profile, whose reports stay unchanged.
  """
  if result.trial_window_steps is None:
    return None
  per_motion: dict[str, dict[str, Any]] = {}
  for segment in result.segments:
    if segment.motion_id is None or not segment.is_trial:
      continue
    entry = per_motion.setdefault(
      str(segment.motion_id),
      {
        "trials": 0,
        "initialization_kind_counts": {},
        "initial_reference_frame_counts": {},
        "start_reason_counts": {},
        "continuation_segments": 0,
      },
    )
    entry["trials"] += 1
    kind = (
      "unreported"
      if segment.initialization_kind is None
      else InitializationKind(segment.initialization_kind).name.lower()
    )
    entry["initialization_kind_counts"][kind] = (
      entry["initialization_kind_counts"].get(kind, 0) + 1
    )
    frame = (
      "unreported"
      if segment.segment_initial_reference_frame is None
      else str(segment.segment_initial_reference_frame)
    )
    entry["initial_reference_frame_counts"][frame] = (
      entry["initial_reference_frame_counts"].get(frame, 0) + 1
    )
    reason = segment.start_reason or "unavailable"
    entry["start_reason_counts"][reason] = (
      entry["start_reason_counts"].get(reason, 0) + 1
    )
  for segment in result.segments:
    if segment.motion_id is None or segment.is_trial:
      continue
    entry = per_motion.get(str(segment.motion_id))
    if entry is not None:
      entry["continuation_segments"] += 1
  return {
    "trial_window_steps": result.trial_window_steps,
    "per_motion": per_motion,
    "note": (
      "counts only; a continuation segment is a reference-wrap or timer teleport "
      "inside a trial and never a new trial"
    ),
  }


def _cohort_mode_report(result: EvaluationResult) -> dict[str, Any]:
  """One bounded per-mode result: total metrics plus per-motion aggregates.

  ``per_motion`` keeps every motion's own segment count, outcome counts,
  completion/failure denominators, and censored segment count, so an aggregate
  can never hide one motion's failures.  ``completion_rate`` is defined over
  completed-or-failed segments only; ``censored_segments`` counts the segments
  whose outcome was a timeout, teleport, timer resample, reset, or step cap, so
  a high completion rate cannot be read as all initiated segments completing.
  """
  stats: tuple[MotionEvaluationStats, ...] = result.per_motion
  return {
    "mode": result.mode,
    "steps": result.steps,
    "rollout_latent": result.rollout_latent,
    "settings": dict(result.settings),
    "metrics": dict(result.metrics),
    "segments": len(result.segments),
    "per_motion": [item.as_dict() for item in stats],
    "trial_provenance": _trial_provenance_summary(result),
    "per_motion_censoring": {
      str(item.motion_id): {
        "segments": item.segments,
        "completion_known_segments": item.completion_known_segments,
        "censored_segments": item.segments - item.completion_known_segments,
        "outcomes": dict(item.outcomes),
      }
      for item in stats
    },
  }


def _rekey_mode_report(
  report: dict[str, Any], *, motion_id: int, teacher_code: int
) -> dict[str, Any]:
  """Rewrite one pinned singleton's local ids to the cohort's authoritative ids.

  A pinned single-motion environment always reports its own local clip 0, so
  presenting that value as the cohort motion id would misattribute every member
  after the first (a ``tennis_001`` pin reports local 0 while its cohort motion
  id is 1).  This rewrites the per-motion entry and its censoring entry - and
  therefore every aggregate that reads them - so no nested level disagrees with
  the cohort identity.  A pinned evaluation must report exactly one motion;
  anything else is refused instead of being folded into one cohort identity.
  """
  per_motion = report["per_motion"]
  if len(per_motion) != 1:
    raise DistillationError(
      "a pinned single-motion evaluation must report exactly one motion, got "
      f"{len(per_motion)}"
    )
  entry = per_motion[0]
  local_key = str(entry["motion_id"])
  censoring = report["per_motion_censoring"].get(local_key)
  if censoring is None:
    raise DistillationError(
      "a pinned evaluation's censoring map does not match its local motion id"
    )
  entry["motion_id"] = motion_id
  entry["teacher_code"] = teacher_code
  report["settings"] = {**report.get("settings", {}), "motion_ids": [motion_id]}
  # The trial provenance map is built from the pinned environment's own local
  # ids, so it needs the same rewrite: leaving it keyed by a local id would
  # attribute this member's trials to whichever member holds that local id.
  provenance = report.get("trial_provenance")
  if isinstance(provenance, dict):
    per_motion = provenance.get("per_motion")
    if isinstance(per_motion, dict) and local_key in per_motion:
      report["trial_provenance"] = {
        **provenance,
        "per_motion": {str(motion_id): per_motion[local_key]},
      }
  report["per_motion_censoring"] = {
    str(motion_id): {
      **censoring,
      "motion_id": motion_id,
      "teacher_code": teacher_code,
    }
  }
  return report


def _evaluate_cohort(
  manifest: Path = Path("configs/distillation/x2_tennis.yaml"),
  repo_root: Path = Path("."),
  teacher_ids: tuple[str, ...] = (),
  task_id: str | None = None,
  device: str = "cpu",
  num_envs: int = 1,
  mode: CohortEvaluationMode = "both",
  checkpoint: Path | None = None,
  steps: int = 512,
  seed: int = 0,
  sampling_mode: SamplingMode = "start",
  reset_profile: ResetProfile | None = None,
  reset_perturbations: ResetPerturbations = "configured",
  rollout_latent: RolloutLatent = "mean",
  report: Path | None = None,
) -> int:
  """Evaluate every selected motion separately with one shared student.

  Each selected motion is pinned into its own single-motion environment and
  **each mode gets a fresh identically seeded adapter**: the baseline and the
  student never share a live environment, because a reset is not proof that
  startup randomization, event timers, and adapter state were restored.  At most
  one simulator exists at a time, and one student model is loaded once and
  reused across every mode and motion.  All evaluation freezes normalizers and
  constructs no optimizer or replay.

  ``--teacher-ids`` defaults to every manifest teacher and uses this
  repository's Python literal collection syntax
  (``--teacher-ids ('tennis_000','tennis_001')``).  A request is normalized to
  **manifest order**, repeated ids are refused, and unknown ids are refused
  before any environment is constructed.  With ``--mode student``/``both`` one
  version-2 M4 cohort checkpoint supplies the single student identity: each
  requested motion must be a stored member whose artifact digests, clip extent,
  and common contract match the live manifest.

  Every reported level is attributed to the motion's **cohort identity** rather
  than to the pinned environment's local clip 0: a saved member's ``motion_id``
  and ``teacher_code`` come from the stored cohort record (already checked
  against the live manifest by ``require_member_matches``), and a teacher-only
  run states the explicit manifest position instead and records which source it
  used (``motion_id_source``).  The per-motion entry, its censoring entry, the
  aggregate per-motion maps, and the outcome counts all use that identity, so a
  reversed or subset request never presents a local singleton 0 as another
  member's cohort motion.

  The report holds one pinned block per motion plus two aggregate summaries per
  mode: ``macro`` (equal motion weight) and ``clip_duration_weighted`` (weights
  proportional to clip frames/FPS).  Both keep the per-motion values, raw
  counts, outcome counts, and censoring denominators, because a good aggregate
  must never hide one failing motion.  An evaluation that raises is recorded as
  an execution error separate from a policy failure, and a missing report makes
  ``complete`` false and blocks any overall quality pass.  This command does not
  assess policy quality.
  """
  try:
    if sampling_mode not in ("start", "uniform"):
      raise ValueError("sampling_mode must be 'start' or 'uniform'")
    if mode not in ("teacher", "student", "both"):
      raise ValueError("mode must be 'teacher', 'student', or 'both'")
    if reset_profile is not None:
      if mode != "student":
        raise ValueError(
          "standing reset profiles are available only for --mode student; "
          "teacher/both transition evaluation is not scheduled"
        )
      if sampling_mode != "start":
        raise ValueError(
          "--reset-profile conflicts with explicit --sampling-mode; omit the "
          "phase option when selecting a standing profile"
        )
    elif reset_perturbations != "configured":
      raise ValueError(
        "--reset-perturbations applies only to a standing --reset-profile; "
        "omit it or select a profile"
      )
    if mode in ("student", "both") and checkpoint is None:
      raise ValueError("student cohort evaluation requires --checkpoint")
    if num_envs <= 0:
      raise ValueError("num_envs must be a positive integer")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
      raise ValueError("steps must be a positive integer")
    cohort = _resolve(manifest, repo_root)
    manifest_order = tuple(teacher.id for teacher in cohort.teachers)
    requested = tuple(teacher_ids)
    if requested:
      duplicates = sorted({item for item in requested if requested.count(item) > 1})
      if duplicates:
        raise ValueError(f"--teacher-ids repeats {duplicates}; pass each motion once")
      unknown = [item for item in requested if item not in manifest_order]
      if unknown:
        raise DistillationError(f"cohort has no teacher(s) {unknown}")
      wanted = set(requested)
      # Normalized to manifest order, so a reversed request cannot renumber the
      # cohort identity or the aggregate keys.
      selected = tuple(item for item in manifest_order if item in wanted)
    else:
      selected = manifest_order
    if not selected:
      raise ValueError("the manifest selects no teacher to evaluate")
    member = None
    identities: dict[str, tuple[int, int, str]] = {}
    if checkpoint is not None:
      if mode == "teacher":
        raise ValueError(
          "--mode teacher ignores a checkpoint; omit --checkpoint or evaluate a student"
        )
      if not _is_cohort_checkpoint(checkpoint):
        raise DistillationError(
          "cohort evaluation of a student requires a version-2 M4 cohort "
          f"checkpoint; {checkpoint} is not one. Use 'distill evaluate' for a "
          "version-1 single-teacher checkpoint"
        )
      if (
        reset_profile is not None
        and _checkpoint_version(checkpoint) != STANDING_COHORT_CHECKPOINT_VERSION
      ):
        raise DistillationError(
          "--reset-profile requires a version-3 standing cohort checkpoint"
        )
      member = _load_member_inference(
        checkpoint, cohort, selected[0], device=device, reset_profile=reset_profile
      )
      if member.relocated_artifact_roles:
        print(
          "[INFO] accepted relocated inference artifacts by content digest: "
          f"{list(member.relocated_artifact_roles)}",
          file=sys.stderr,
          flush=True,
        )
      # The same checked selection the single-member paths use, applied per
      # motion, and the source of the authoritative cohort motion id/code.
      for teacher_id in selected:
        stored = require_member_matches(member.cohort, cohort, teacher_id)
        identities[teacher_id] = (
          int(stored.motion_id),
          int(stored.teacher_code),
          "saved_cohort_member",
        )
    else:
      # Teacher-only: there is no saved cohort to provide a library clip
      # position, so the explicit manifest position is the only authoritative
      # identity available, and it is recorded as such.
      identities = {
        teacher_id: (
          manifest_order.index(teacher_id),
          manifest_order.index(teacher_id),
          "manifest_position",
        )
        for teacher_id in selected
      }
    schema = None if member is None else member.schema
    student = None if member is None else member.model
    if student is not None:
      student.eval()

    modes = ("teacher", "student") if mode == "both" else (mode,)
    motions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for teacher_id in selected:
      teacher = cohort.teacher(teacher_id)
      motion_id, teacher_code, id_source = identities[teacher_id]
      entry: dict[str, Any] = {
        "teacher_id": teacher_id,
        "motion_id": motion_id,
        "teacher_code": teacher_code,
        "motion_id_source": id_source,
        "motion": str(teacher.entry.motion),
        "frames": teacher.reference.frames,
        "fps": teacher.reference.fps,
        "clip_seconds": (
          teacher.reference.frames / teacher.reference.fps
          if teacher.reference.fps
          else None
        ),
        "reports": {},
      }
      for current in modes:
        adapter = None
        try:
          # A fresh identically seeded adapter per mode: the next mode (and the
          # next motion) never inherits this rollout's simulator state.
          adapter = (
            make_multi_teacher_distillation_adapter(
              cohort,
              (teacher_id,),
              phase_policy="start",
              task_id=task_id,
              num_envs=num_envs,
              device=device,
              schema=schema,
              seed=seed,
              reset_policy=_evaluation_reset_policy(
                reset_profile, _STANDING_EVALUATION_WINDOW_FRAMES
              ),
            )
            if reset_profile is not None
            else make_distillation_adapter(
              cohort,
              teacher_id,
              task_id=task_id,
              num_envs=num_envs,
              device=device,
              schema=schema,
              seed=seed,
            )
          )
          seed_audit = _seed_audit(adapter)
          _require_requested_seed_applied(seed, seed_audit)
          resolved_runtime = {
            "device": device,
            "num_envs": num_envs,
            "requested_seed": seed,
            "resolved_seed": seed_audit["effective_seed"],
            "seed_provenance": seed_audit,
          }
          if "runtime" in entry and entry["runtime"] != resolved_runtime:
            raise DistillationError(
              f"{teacher_id} resolved a different runtime for {current}: "
              f"{entry['runtime']} != {resolved_runtime}"
            )
          entry["runtime"] = resolved_runtime
          live = adapter.env.command_manager.get_term("motion")
          live.cfg.sampling_mode = sampling_mode
          if reset_profile is not None:
            # One pinned environment per mode; the choice is applied before the
            # first reset and reported from this pinned command copy afterwards,
            # because each pinned adapter is its own private config copy.
            _apply_reset_perturbations(live, reset_perturbations)
            entry["reset_perturbations"] = _reset_provenance_report(
              live, reset_perturbations
            )
          teacher_source = getattr(adapter, "teacher", getattr(adapter, "bank", None))
          if teacher_source is None:
            raise DistillationError("evaluation adapter exposes no teacher bank")
          result = evaluate_distillation(
            adapter,
            teacher_source,
            None if current == "teacher" else student,
            mode=current,
            steps=steps,
            rollout_latent=rollout_latent,
            seed=seed,
            control_period_s=cohort.control.control_period_s,
            standing_trials=reset_profile is not None,
            trial_window_steps=_STANDING_EVALUATION_WINDOW_FRAMES,
          )
          entry["reports"][current] = _rekey_mode_report(
            _cohort_mode_report(result),
            motion_id=motion_id,
            teacher_code=teacher_code,
          )
        except (
          DistillationError,
          CheckpointValidationError,
          ValueError,
          RuntimeError,
        ) as exc:
          errors.append(
            {
              "teacher_id": teacher_id,
              "mode": current,
              "error_type": type(exc).__name__,
              "error": str(exc),
            }
          )
          print(f"[FAIL] {teacher_id} {current}: {exc}", file=sys.stderr)
        finally:
          # One active simulator at a time: this mode's environment is closed
          # before the next mode or motion builds its own.
          if adapter is not None:
            adapter.close()
      motions.append(entry)

    expected_reports = modes
    missing = [
      {"teacher_id": entry["teacher_id"], "mode": name}
      for entry in motions
      for name in expected_reports
      if name not in entry["reports"]
    ]
    # Aggregate keys are the authoritative cohort motion ids, matching the
    # per-motion entries, so a reversed request cannot renumber an aggregate.
    weights = {
      identities[entry["teacher_id"]][0]: entry["clip_seconds"] for entry in motions
    }
    aggregates: dict[str, Any] = {}
    for name in expected_reports:
      per_motion_metrics: dict[int, dict[str, float]] = {}
      per_motion_outcomes: dict[str, dict[str, int]] = {}
      for entry in motions:
        mode_report = entry["reports"].get(name)
        if mode_report is None:
          continue
        item = mode_report["per_motion"][0]
        per_motion_metrics[int(item["motion_id"])] = dict(item["metrics"])
        per_motion_outcomes[str(item["motion_id"])] = dict(item["outcomes"])
      if not per_motion_metrics:
        continue
      aggregate = _aggregate_motion_metrics(per_motion_metrics, weights)
      aggregate["outcomes"] = _combine_outcomes(per_motion_outcomes)
      aggregates[name] = aggregate

    complete = not errors and not missing
    payload = {
      "command": " ".join(sys.argv),
      "status": "cohort_evaluation",
      "mode": mode,
      "sampling_mode": sampling_mode,
      "reset_profile": reset_profile,
      "reset_window_frames": (
        None if reset_profile is None else _STANDING_EVALUATION_WINDOW_FRAMES
      ),
      "checkpoint": None if checkpoint is None else str(checkpoint),
      "checkpoint_version": (
        None if checkpoint is None else _checkpoint_version(checkpoint)
      ),
      "requested_teacher_ids": list(requested),
      "teacher_ids": list(selected),
      "selection_normalized_to_manifest_order": list(requested) != list(selected),
      "control_hz": cohort.control.control_hz,
      "checkpoint_model": _student_identity_report(member),
      "motions": motions,
      "aggregate": {
        "aggregation_weights": {
          "macro": "equal per motion",
          "clip_duration_weighted": {
            "formula": "frames/fps",
            "values": {
              entry["teacher_id"]: weights[entry["motion_id"]] for entry in motions
            },
          },
        },
        "motion_ids": {entry["teacher_id"]: entry["motion_id"] for entry in motions},
        "per_mode": aggregates,
      },
      "evaluation_errors": errors,
      "missing_reports": missing,
      "complete": complete,
      "quality": {
        "implementation": "bounded pinned per-motion cohort evaluation",
        "smoke": "metrics are bounded-run evidence only",
        "policy_quality": "requires parent baseline-relative interpretation",
        "aggregate_is_not_a_quality_pass": True,
        "overall_pass": False,
        "hardware_readiness": "not assessed",
      },
    }
    _write_report(payload, report)
    return 0 if complete else 1
  except (
    DistillationError,
    CheckpointValidationError,
    ValueError,
    RuntimeError,
  ) as exc:
    print(f"[FAIL] {exc}", file=sys.stderr)
    return 1


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
  hot swap) is validated against the live schema, teacher hashes, control
  contract, and (for a version-2 artifact) the saved cohort membership, so an
  incompatible artifact is rejected instead of silently replacing the running
  policy.  The live schema is the one the initialized environment packs with,
  so a swap whose saved schema differs is refused rather than mis-packed.
  """
  from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago

  directory = checkpoint.parent
  # The live schema the running environment packs with, so a swapped-in policy
  # cannot mis-pack a checkpoint trained for a different conditioning layout.
  expected_schema = adapter.schema

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
    # Version-dispatched model-only load: a version-1 artifact keeps its
    # existing teacher-hash/control-contract check, while a version-2 cohort
    # artifact must name a member of the stored cohort and match the live one.
    inference = _load_student_inference(
      directory / name,
      cohort,
      teacher_id,
      device=device,
      expected_schema=expected_schema,
    )
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

  A version-2 M4 cohort checkpoint is supported through checked member
  selection: ``--teacher-id`` must name a member of the saved cohort, its
  artifacts and common contract are validated against the live manifest, and
  the pinned member is played in a single-motion environment of that member's
  clip.  Hot swaps use the same version-dispatched validation, so a version-1
  and a version-2 artifact in one directory can both be inspected.

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
    # Early model-only load: cheap, validates teacher/cohort identity and the
    # control contract, and yields the SAVED schema the live packing must
    # reproduce (the saved settings imply the trained architecture, not a
    # default one).
    inference = _load_student_inference(checkpoint, cohort, teacher_id, device=device)
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


def _export(
  checkpoint: Path,
  manifest: Path = Path("configs/distillation/x2_tennis.yaml"),
  repo_root: Path = Path("."),
  teacher_id: str = "tennis_000",
  output_dir: Path = Path("rl_model"),
  asset_audit: Path | None = None,
  asset_audits: Path | None = None,
) -> int:
  """Export an audited gravity VAE checkpoint into a v2 or v3 bundle.

  A version-1 single-teacher checkpoint exports through ``--asset-audit`` as
  before.  A version-2/3 M4 cohort checkpoint exports through ``--asset-audits``:
  a JSON index mapping every manifest member's teacher id to that member's
  audit file, each produced by ``make_export_audit`` against a pinned
  single-teacher environment of that member.  A cohort checkpoint with only the
  singular audit is refused rather than silently reduced to one teacher, and a
  single-teacher checkpoint ignores ``--asset-audits``.
  """
  if asset_audit is None and asset_audits is None:
    print(
      "[FAIL] --asset-audit (single teacher) or --asset-audits (cohort index) is "
      "required; tensor schema is not physical frame evidence",
      file=sys.stderr,
    )
    return 1
  try:
    if _is_cohort_checkpoint(checkpoint):
      if asset_audits is None:
        raise ExportValidationError(
          "this checkpoint is a version-2/3 M4 cohort checkpoint; pass the cohort "
          "audit index with --asset-audits (one audit per manifest member), not "
          "the singular --asset-audit"
        )
      if asset_audit is not None:
        raise ExportValidationError(
          "--asset-audit names one teacher's audit but this checkpoint is a "
          "cohort artifact; pass --asset-audits instead"
        )
      result = export_cohort_bundle(
        checkpoint,
        manifest,
        output_dir,
        repo_root=repo_root,
        asset_audits=asset_audits,
      )
    else:
      if asset_audit is None:
        raise ExportValidationError(
          "this checkpoint is a version-1 single-teacher artifact; pass its audit "
          "with --asset-audit"
        )
      if asset_audits is not None:
        raise ExportValidationError(
          "--asset-audits is the cohort audit index; this checkpoint is a "
          "version-1 single-teacher artifact that needs --asset-audit"
        )
      result = export_bundle(
        checkpoint,
        manifest,
        teacher_id,
        output_dir,
        repo_root=repo_root,
        asset_audit=asset_audit,
      )
  except (ExportValidationError, DistillationError, CheckpointValidationError) as exc:
    print(f"[FAIL] {exc}", file=sys.stderr)
    return 1
  print(json.dumps(result.report, indent=2, sort_keys=True))
  print(f"[OK] wrote audited VAE bundle {result.descriptor.parent}", file=sys.stderr)
  return 0


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
    "evaluate-cohort": _evaluate_cohort,
    "play": _play,
    "export": _export,
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
