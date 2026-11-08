"""The construction recipe one cohort collection environment reproduces.

Sharded collection builds its environments inside worker processes, so the
recipe that turns a cohort, a reset configuration, and a seed into a live
adapter and student model has to live in exactly one place.  The CLI's
single-process path and every worker call these functions, so a worker cannot
quietly simulate a different environment from the one the run planned while
every identity check still passes.

Seeding here is *derived*, never drawn.  A worker's environment seed and the
seed of its per-iteration collection generator are pure functions of the run's
base seed, the worker index, and the iteration.  That is what lets a sharded
resume reproduce collection exactly without adding per-worker generator state
to the checkpoint: iteration ``k`` always draws the same randomness for worker
``w``, whether or not the run was interrupted before it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mjlab.tasks.tracking.distillation.adapter import (
  MultiMotionDistillationAdapter,
  make_multi_teacher_distillation_adapter,
)
from mjlab.tasks.tracking.distillation.config import (
  CohortContract,
  load_manifest,
  resolve_cohort,
)
from mjlab.tasks.tracking.distillation.coverage_policy import CoveragePolicy
from mjlab.tasks.tracking.distillation.model import ConditionalVAE
from mjlab.tasks.tracking.distillation.multi_motion import PhasePolicy
from mjlab.tasks.tracking.distillation.reset_policy import ResetPolicy
from mjlab.tasks.tracking.distillation.vae_config import (
  DEFAULT_MODEL_SETTINGS,
  VaeSchema,
  make_schema,
)

ENV_SEED_STRIDE = 1_000_003
"""Distance between two workers' environment seeds."""

GENERATOR_SEED_STRIDE = 7_919

EVALUATION_SEED_STRIDE = 1_000_003
"""Offset that separates an evaluation's reset stream from the collection's."""
"""Distance between two workers' collection generator seeds."""


def worker_env_seed(base_seed: int, worker_index: int) -> int:
  """Return the environment seed of one worker shard.

  Workers must never share an environment seed: identical seeds would simulate
  the same instances N times and quietly reduce the collected coverage to one
  shard's worth of states while every counter still looked right.
  """
  return base_seed + worker_index * ENV_SEED_STRIDE


def worker_generator_seed(base_seed: int, iteration: int, worker_index: int) -> int:
  """Return the collection-generator seed of one worker for one iteration.

  The runner already derives its collection seed from the iteration
  (``config.seed + iteration``); adding the worker stride keeps the same
  per-iteration derivation while making two workers' draws differ.  Because the
  value depends only on the iteration, a resumed run reproduces it exactly.
  """
  return base_seed + iteration + worker_index * GENERATOR_SEED_STRIDE


def worker_evaluation_seed(base_seed: int, iteration: int, worker_index: int) -> int:
  """Return the seed one worker's evaluation derives its standing resets from.

  Evaluation draws its own standing/reference decisions, so it must not replay
  the decisions the collection of the same iteration already made: a run whose
  evaluation reused the collection stream would score the policy on the very
  resets it had just been trained against.  The offset keeps both streams
  reproducible from (base seed, iteration, worker) and disjoint for any run this
  project configures -- up to the CLI's ``_MAX_WORKERS`` shards and iterations
  below
  ``EVALUATION_SEED_STRIDE``, which is 100x the planned budget.  The bounds are
  not implicit: ``test_collection_and_evaluation_seeds_stay_disjoint`` scans the
  reachable domain, because a container can otherwise always be built that makes
  two additive formulas meet.
  """
  return (
    base_seed
    + iteration
    + worker_index * GENERATOR_SEED_STRIDE
    + EVALUATION_SEED_STRIDE
  )


@dataclass(frozen=True, slots=True)
class CohortSetup:
  """Plain-data description of one cohort collection environment.

  Everything is data, so a worker receives the recipe itself instead of
  reconstructing it from its own command line, and the parent can resolve the
  same cohort for its own identity and replay without building an environment.
  ``motion_ids`` pairs each of ``teacher_ids`` with its cohort-wide clip id, and
  is empty for a build that owns the whole selection in manifest order; it is
  carried by value so a shard cannot renumber the clips it happens to hold.
  """

  manifest: Path
  repo_root: Path
  teacher_ids: tuple[str, ...]
  task_id: str | None
  phase_policy: PhasePolicy
  reset_policy: ResetPolicy
  base_seed: int
  schema_mode: str = "gravity"
  numerical_controls: bool = False
  coverage_policy: CoveragePolicy | None = None
  coverage_parity_evidence: str | None = None
  motion_ids: tuple[tuple[str, int], ...] = ()

  def __post_init__(self) -> None:
    if not self.motion_ids:
      return
    ids = tuple(teacher_id for teacher_id, _ in self.motion_ids)
    if ids != tuple(self.teacher_ids):
      raise ValueError(
        f"setup clip ids {ids} do not align with its teacher ids "
        f"{tuple(self.teacher_ids)}: one global clip id per selected teacher is "
        "required, in the same order"
      )
    numbers = tuple(motion_id for _, motion_id in self.motion_ids)
    if len(set(numbers)) != len(numbers) or any(
      isinstance(motion_id, bool) or not isinstance(motion_id, int) or motion_id < 0
      for motion_id in numbers
    ):
      raise ValueError(f"setup clip ids {numbers} must be non-negative and unique")

  def clip_motion_ids(self) -> dict[str, int] | None:
    """Cohort-wide clip id per selected teacher, or ``None`` when positional."""
    return dict(self.motion_ids) if self.motion_ids else None

  def resolve(self) -> CohortContract:
    """Load and validate the cohort manifest this setup names."""
    return resolve_cohort(load_manifest(self.manifest, self.repo_root))

  def schema_for_cohort(self, cohort: CohortContract) -> VaeSchema:
    """Resolve the student schema a cohort implies, from the task configuration.

    One reader, used by both the environment builder and the coverage gates, so a
    gate cannot disagree with the environment it authorizes.  The reference preview
    follows the cohort's declared ``lookahead_s`` and the joint order follows its
    action specification.
    """
    actions = getattr(cohort, "actions", None)
    joint_order = getattr(actions, "joint_names", None)
    # The student observes the same reference preview the teacher used.  The cohort fixes
    # one lookahead_s for every member, and every cohort that declares zero keeps the
    # frozen version-1 layout, so existing runs and checkpoints are unaffected.
    preview = float(cohort.lookahead_s) > 0.0
    return (
      make_schema(self.schema_mode, preview=preview)
      if joint_order is None
      else make_schema(self.schema_mode, joint_order=joint_order, preview=preview)
    )

  def build_adapter(
    self, *, device: str, num_envs: int, worker_index: int = 0
  ) -> MultiMotionDistillationAdapter:
    """Build one collection environment for the given worker shard.

    ``worker_index`` 0 with a base seed reproduces the single-process path's
    environment exactly, because the derivation adds zero for the first worker.
    """
    cohort = self.resolve()
    schema = self.schema_for_cohort(cohort)
    if self.coverage_policy is not None:
      # The policy describes the student it tracks, so the two must agree.  Compare
      # the declared scope with the resolved schema instead of naming one layout:
      # that catches a mismatch for any schema, including a preview cohort.
      if self.schema_mode != "gravity":
        raise ValueError(
          "difficult-state coverage requires the gravity conditioning layout"
        )
      declared = dict(self.coverage_policy.scope)
      live = {
        "reference_dim": schema.reference_dim,
        "conditioning_dim": schema.conditioning_dim,
        "latent_dim": schema.latent_dim,
        "action_dim": schema.action_dim,
      }
      if any(int(declared.get(name, -1)) != value for name, value in live.items()):
        raise ValueError(
          "difficult-state coverage policy declares a different schema than the "
          f"resolved cohort: policy {declared}, cohort {live}"
        )
    return make_multi_teacher_distillation_adapter(
      cohort,
      self.teacher_ids,
      phase_policy=self.phase_policy,
      task_id=self.task_id,
      num_envs=num_envs,
      device=device,
      seed=worker_env_seed(self.base_seed, worker_index),
      reset_policy=self.reset_policy,
      schema=schema,
      coverage_policy=self.coverage_policy,
      coverage_parity_evidence=self.coverage_parity_evidence,
      motion_ids=self.clip_motion_ids(),
    )

  def build_student(
    self, *, schema: VaeSchema, device: str, worker_index: int = 0
  ) -> ConditionalVAE:
    """Build the student model a shard starts from.

    A worker's initial weights never survive its first collection call: the
    parent broadcasts the trained state before every call, so what matters is
    that the initialization is deterministic and identical in shape to the
    parent's model, not that it is the parent's initialization.
    """
    torch.manual_seed(worker_env_seed(self.base_seed, worker_index))
    return ConditionalVAE(
      schema,
      DEFAULT_MODEL_SETTINGS,
      numerical_controls=self.numerical_controls,
    ).to(device)


COVERAGE_POLICY_CHOICES = ("off", "difficult-state-coverage")


def resolve_coverage_policy(
  choice: str, *, schema: VaeSchema | None = None
) -> CoveragePolicy | None:
  """Resolve a CLI choice to a coverage policy, or return None for off.

  ``schema`` is the student schema the resolved task configuration implies.  The
  policy records it in its scope, so the policy identity says which layout it was
  built for.  Omitting it keeps the version-1 default identity.
  """
  if choice not in COVERAGE_POLICY_CHOICES:
    raise ValueError(
      f"unknown coverage policy {choice!r}; expected one of {COVERAGE_POLICY_CHOICES}"
    )
  if choice == "off":
    return None
  if schema is None:
    return CoveragePolicy.difficult_state_coverage()
  return CoveragePolicy.difficult_state_coverage(
    reference_dim=schema.reference_dim,
    conditioning_dim=schema.conditioning_dim,
    latent_dim=schema.latent_dim,
    action_dim=schema.action_dim,
  )


COVERAGE_SCHEMA_FIELDS = (
  "reference_dim",
  "conditioning_dim",
  "latent_dim",
  "action_dim",
)
"""The schema dimensions a coverage policy declares and parity evidence records."""


def coverage_schema_payload(schema: VaeSchema) -> dict[str, int]:
  """The four dimensions, in the form the parity evidence and the policy use."""
  return {name: int(getattr(schema, name)) for name in COVERAGE_SCHEMA_FIELDS}


def require_coverage_execution(
  policy: CoveragePolicy | None,
  parity_evidence: Path | str | None = None,
  *,
  manifest_sha256: str | None = None,
  expected_teacher_ids: Sequence[str] | None = None,
  expected_schema: Mapping[str, int] | None = None,
) -> dict[str, Any]:
  """Validate the bounded MVP execution gate and return evidence metadata.

  The MVP does not execute teacher branches.  It does require a measured
  restore-parity report before it accepts captured states for training.

  ``expected_schema`` is the schema the resolved task configuration implies.  The
  evidence must describe that same layout, so a report measured for one reference
  width cannot authorize a run at another.  When it is omitted the version-1 layout
  is required, which keeps every earlier caller on its existing contract.
  """
  if policy is None:
    return {}
  if parity_evidence is None or not str(parity_evidence).strip():
    raise ValueError(
      "difficult-state coverage is enabled and requires "
      "--coverage-parity-evidence pointing to a passing restore-parity report "
      "for this cohort; produce it with `distill coverage-parity --manifest "
      "<manifest> --teacher-ids (<ids>)`, or pass --coverage-policy off to "
      "train without coverage"
    )
  path = Path(parity_evidence)
  if not path.is_file():
    raise ValueError(f"coverage parity evidence does not exist: {path}")
  try:
    payload = json.loads(path.read_text())
  except (OSError, json.JSONDecodeError) as exc:
    raise ValueError(f"coverage parity evidence is not valid JSON: {path}") from exc
  if not isinstance(payload, dict):
    raise ValueError("coverage parity evidence must contain a JSON object")
  if payload.get("kind") != "x2-vae-coverage-restore-parity-v1":
    raise ValueError("coverage parity evidence has an unsupported kind")
  if payload.get("status") != "pass":
    raise ValueError("coverage parity evidence must have status 'pass'")
  if payload.get("policy_input_exact") is not True:
    raise ValueError("coverage parity evidence must prove exact policy-input parity")
  schema = payload.get("schema")
  if expected_schema is None:
    # The version-1 layout, for callers that do not resolve a cohort schema.
    expected_schema = {
      "reference_dim": 68,
      "conditioning_dim": 99,
      "latent_dim": 32,
      "action_dim": 31,
    }
  else:
    expected_schema = {name: int(value) for name, value in expected_schema.items()}
  if schema != expected_schema:
    raise ValueError(
      "coverage parity evidence describes schema "
      f"{dict(schema) if isinstance(schema, Mapping) else schema}, but the resolved "
      f"cohort describes {expected_schema}"
    )
  errors = payload.get("policy_input_max_abs_error")
  required_errors = {
    "teacher_actor_view",
    "student_reference",
    "student_conditioning",
    "previous_action",
  }
  if not isinstance(errors, Mapping) or set(errors) != required_errors:
    raise ValueError("coverage parity evidence has incomplete policy-input errors")
  for name in required_errors:
    value = errors[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
      raise ValueError(f"coverage parity error {name!r} is not numeric")
    if not math.isfinite(float(value)) or float(value) != 0.0:
      raise ValueError(f"coverage parity error {name!r} is not exactly zero")
  teacher_ids = payload.get("teacher_ids")
  if (
    not isinstance(teacher_ids, list)
    or not teacher_ids
    or any(not isinstance(value, str) or not value for value in teacher_ids)
    or len(set(teacher_ids)) != len(teacher_ids)
  ):
    raise ValueError("coverage parity evidence has invalid teacher IDs")
  if expected_teacher_ids is not None and tuple(teacher_ids) != tuple(
    expected_teacher_ids
  ):
    raise ValueError("coverage parity evidence belongs to a different teacher cohort")
  if manifest_sha256 is not None and payload.get("manifest_sha256") != manifest_sha256:
    raise ValueError("coverage parity evidence belongs to a different cohort manifest")
  digest = hashlib.sha256(path.read_bytes()).hexdigest()
  return {
    "kind": payload["kind"],
    "status": payload["status"],
    "path": str(path),
    "sha256": digest,
    "manifest_sha256": payload.get("manifest_sha256"),
    "teacher_ids": list(teacher_ids),
    "schema": dict(expected_schema),
  }
