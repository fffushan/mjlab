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

from dataclasses import dataclass
from pathlib import Path

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
from mjlab.tasks.tracking.distillation.model import ConditionalVAE
from mjlab.tasks.tracking.distillation.multi_motion import PhasePolicy
from mjlab.tasks.tracking.distillation.reset_policy import ResetPolicy
from mjlab.tasks.tracking.distillation.vae_config import (
  DEFAULT_MODEL_SETTINGS,
  VaeSchema,
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
  reproducible from (base seed, iteration, worker) while making them disjoint
  for every iteration a run can reach.
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
    return make_multi_teacher_distillation_adapter(
      self.resolve(),
      self.teacher_ids,
      phase_policy=self.phase_policy,
      task_id=self.task_id,
      num_envs=num_envs,
      device=device,
      seed=worker_env_seed(self.base_seed, worker_index),
      reset_policy=self.reset_policy,
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
    return ConditionalVAE(schema, DEFAULT_MODEL_SETTINGS).to(device)
