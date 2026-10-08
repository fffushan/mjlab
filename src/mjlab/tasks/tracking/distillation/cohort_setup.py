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

  def resolve(self) -> CohortContract:
    """Load and validate the cohort manifest this setup names."""
    return resolve_cohort(load_manifest(self.manifest, self.repo_root))

  def build_adapter(
    self, *, device: str, num_envs: int, worker_index: int = 0
  ) -> MultiMotionDistillationAdapter:
    """Build one collection environment for the given worker shard.

    ``worker_index`` 0 with a base seed reproduces the single-process path's
    environment exactly, because the derivation adds zero for the first worker.
    """
    cohort = self.resolve()
    actions = getattr(cohort, "actions", None)
    joint_order = getattr(actions, "joint_names", None)
    schema = (
      make_schema(self.schema_mode)
      if joint_order is None
      else make_schema(self.schema_mode, joint_order=joint_order)
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


def resolve_coverage_policy(choice: str) -> CoveragePolicy | None:
  """Resolve a CLI choice to a coverage policy, or return None for off."""
  if choice not in COVERAGE_POLICY_CHOICES:
    raise ValueError(
      f"unknown coverage policy {choice!r}; expected one of {COVERAGE_POLICY_CHOICES}"
    )
  if choice == "off":
    return None
  return CoveragePolicy.difficult_state_coverage()


def require_coverage_execution(
  policy: CoveragePolicy | None,
  parity_evidence: Path | str | None = None,
  *,
  manifest_sha256: str | None = None,
) -> dict[str, Any]:
  """Validate the bounded MVP execution gate and return evidence metadata.

  The MVP does not execute teacher branches.  It does require a measured
  restore-parity report before it accepts captured states for training.
  """
  if policy is None:
    return {}
  if parity_evidence is None or not str(parity_evidence).strip():
    raise ValueError(
      "difficult-state coverage requires --coverage-parity-evidence pointing to "
      "a passing restore-parity report"
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
  if manifest_sha256 is not None and payload.get("manifest_sha256") != manifest_sha256:
    raise ValueError("coverage parity evidence belongs to a different cohort manifest")
  digest = hashlib.sha256(path.read_bytes()).hexdigest()
  return {
    "kind": payload["kind"],
    "status": payload["status"],
    "path": str(path),
    "sha256": digest,
    "manifest_sha256": payload.get("manifest_sha256"),
  }
