"""Bounded bootstrap/collect/update/evaluate lifecycle orchestration.

The runner is deliberately a Python API, not a CLI.  It owns schedule
counters and lifecycle boundaries while the adapter/collector owns simulator
state and the trainer owns pure optimization.  Resume always requests a fresh
adapter reset; simulator bitwise continuation is never claimed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from mjlab.tasks.tracking.distillation.checkpoint import (
  LifecycleState,
  load_checkpoint,
  save_checkpoint,
)
from mjlab.tasks.tracking.distillation.collector import (
  CollectionConfig,
  CollectionResult,
  DAggerCollector,
  EvaluationMode,
  EvaluationResult,
  RolloutLatent,
  evaluate_distillation,
)
from mjlab.tasks.tracking.distillation.storage import LabeledReplayBuffer
from mjlab.tasks.tracking.distillation.trainer import (
  TrainingUpdate,
  VaeDistillationTrainer,
)


class RunnerValidationError(ValueError):
  """An invalid bounded lifecycle setting or incompatible callback was used."""


@dataclass(frozen=True, slots=True)
class RunnerConfig:
  """Small, explicit lifecycle budgets; no setting launches an unbounded run."""

  max_iterations: int = 1
  bootstrap_steps: int = 0
  collection_steps: int = 32
  updates_per_iteration: int = 1
  teacher_probability: float = 0.0
  evaluate_every: int = 0
  evaluation_steps: int = 0
  evaluation_mode: EvaluationMode = "student"
  rollout_latent: RolloutLatent = "mean"
  seed: int = 0

  def __post_init__(self) -> None:
    for name in (
      "max_iterations",
      "bootstrap_steps",
      "collection_steps",
      "updates_per_iteration",
      "evaluate_every",
      "evaluation_steps",
    ):
      value = getattr(self, name)
      if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RunnerValidationError(f"{name} must be a non-negative integer")
    if self.max_iterations <= 0:
      raise RunnerValidationError("max_iterations must be positive")
    if not 0.0 <= self.teacher_probability <= 1.0:
      raise RunnerValidationError("teacher_probability must be in [0, 1]")
    if self.evaluate_every and self.evaluation_steps <= 0:
      raise RunnerValidationError("evaluation_steps must be positive when evaluating")
    if self.evaluation_mode not in ("teacher", "student"):
      raise RunnerValidationError("evaluation_mode must be teacher or student")
    if self.rollout_latent not in ("mean", "sampled"):
      raise RunnerValidationError("rollout_latent must be mean or sampled")
    if not isinstance(self.seed, int) or isinstance(self.seed, bool):
      raise RunnerValidationError("seed must be an integer")


@dataclass(frozen=True, slots=True)
class LifecycleIteration:
  """Machine-readable result for one bounded collect/update cycle."""

  iteration: int
  collection: CollectionResult | None
  updates: tuple[TrainingUpdate, ...]
  evaluation: EvaluationResult | None
  resumed_reset: bool = False


@dataclass(slots=True)
class DistillationRunner:
  """Coordinate one teacher, one collector, one trainer, and bounded cycles."""

  collector: DAggerCollector
  trainer: VaeDistillationTrainer
  config: RunnerConfig = field(default_factory=RunnerConfig)
  evaluation_teacher: Any | None = None
  evaluation_student: Any | None = None
  iteration: int = 0
  events: list[dict[str, Any]] = field(default_factory=list)
  _resume_reset_pending: bool = False
  _reset_required: bool = False
  _segment_namespace: int = 0

  def __post_init__(self) -> None:
    if self.collector.replay is not self.trainer.replay:
      raise RunnerValidationError("collector and trainer must share one replay owner")
    if self.config.evaluation_mode == "student" and self.config.evaluate_every:
      if self.evaluation_student is None:
        self.evaluation_student = self.trainer.model
    if self.config.evaluation_mode == "teacher" and self.evaluation_teacher is None:
      self.evaluation_teacher = self.collector.teacher

  @property
  def replay(self) -> LabeledReplayBuffer:
    return self.trainer.replay

  def _collect(
    self, steps: int, probability: float, *, reset: bool
  ) -> CollectionResult:
    self.trainer.begin_collection()
    return self.collector.collect(
      CollectionConfig(
        steps=steps,
        teacher_probability=probability,
        rollout_latent=self.config.rollout_latent,
        seed=self.config.seed + self.iteration,
        collector_iteration=self.iteration,
        rng_mode="persistent",
      ),
      reset=reset,
    )

  def _rebase_collection(self, result: CollectionResult) -> CollectionResult:
    if self._segment_namespace == 0 or result.fresh_data is None:
      return result
    changed = self.replay.rebase_segment_ids(self.iteration, self._segment_namespace)
    if changed == 0:
      raise RunnerValidationError(
        "resume segment namespace could not find the fresh collection records"
      )
    fresh_batch = result.fresh_data.batch
    rebased_batch = replace(
      fresh_batch,
      episode_id=fresh_batch.episode_id + self._segment_namespace,
    )
    return replace(
      result,
      fresh_data=replace(result.fresh_data, batch=rebased_batch),
    )

  def run_iteration(self) -> LifecycleIteration:
    """Run at most one bootstrap or collection/update/evaluation cycle."""
    if self.iteration >= self.config.max_iterations:
      raise RunnerValidationError("configured lifecycle budget is exhausted")
    resumed_reset = self._resume_reset_pending
    bootstrap = self.replay.is_empty and self.config.bootstrap_steps > 0
    # Evaluation advances the shared adapter outside the collector and
    # invalidates its snapshot with ``requires_reset=True``.  The runner must
    # therefore carry that requirement into the next collection call instead
    # of only resetting on resume or iteration zero.
    reset_required = (
      self._resume_reset_pending or self._reset_required or self.iteration == 0
    )
    if bootstrap:
      collection = self._collect(self.config.bootstrap_steps, 1.0, reset=True)
    else:
      collection = self._collect(
        self.config.collection_steps,
        self.config.teacher_probability,
        reset=reset_required,
      )
    collection = self._rebase_collection(collection)
    self._resume_reset_pending = False
    self._reset_required = False
    updates: list[TrainingUpdate] = []
    fresh = collection.fresh_data
    if fresh is not None:
      self.trainer.begin_training(fresh)
    if len(self.replay) > 0:
      for _ in range(self.config.updates_per_iteration):
        updates.append(self.trainer.train_update())
    self.trainer.freeze_normalizers()
    evaluation = None
    if (
      self.config.evaluate_every
      and (self.iteration + 1) % self.config.evaluate_every == 0
      and self.config.evaluation_steps > 0
    ):
      teacher = self.evaluation_teacher or self.collector.teacher
      student = self.evaluation_student or self.trainer.model
      try:
        evaluation = evaluate_distillation(
          self.collector.adapter,
          teacher,
          student,
          mode=self.config.evaluation_mode,
          steps=self.config.evaluation_steps,
          rollout_latent=self.config.rollout_latent,
          seed=self.config.seed + self.iteration,
        )
      finally:
        # Evaluation resets/steps the shared adapter independently of the
        # collector.  Never retain the pre-evaluation snapshot for collection,
        # and record that the next collection must resynchronize with a reset.
        self.collector.invalidate_snapshot()
        self._reset_required = True
    result = LifecycleIteration(
      self.iteration,
      collection,
      tuple(updates),
      evaluation,
      resumed_reset,
    )
    self.events.append(
      {
        "event": "iteration",
        "iteration": self.iteration,
        "samples": collection.samples,
        "optimizer_steps": len(updates),
        "resumed_simulator_reset": result.resumed_reset,
      }
    )
    self.iteration += 1
    return result

  def run(self, *, iterations: int | None = None) -> tuple[LifecycleIteration, ...]:
    """Run a bounded number of cycles and return their immutable results."""
    count = (
      self.config.max_iterations - self.iteration if iterations is None else iterations
    )
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
      raise RunnerValidationError("iterations must be a non-negative integer")
    if self.iteration + count > self.config.max_iterations:
      raise RunnerValidationError("requested iterations exceed configured budget")
    return tuple(self.run_iteration() for _ in range(count))

  def save(
    self,
    path: str,
    *,
    teacher_hashes: Mapping[str, str] | None = None,
    control_contract: Mapping[str, Any] | None = None,
    resolved_config: Mapping[str, Any] | None = None,
    schedule: Mapping[str, Any] | None = None,
  ) -> None:
    """Persist runner/trainer/replay state without serializing the simulator."""
    self.trainer.assert_healthy()
    save_checkpoint(
      path,
      self.trainer,
      self.replay,
      counters={
        "iteration": self.iteration,
        "segment_namespace": self._segment_namespace,
      },
      schedule=schedule or {"max_iterations": self.config.max_iterations},
      resolved_config=resolved_config or {},
      teacher_hashes=teacher_hashes or {},
      control_contract=control_contract or {},
      collector=self.collector,
    )
    self.events.append({"event": "checkpoint_saved", "path": path})

  def resume(
    self,
    path: str,
    *,
    teacher_hashes: Mapping[str, str] | None = None,
    control_contract: Mapping[str, Any] | None = None,
    map_location: str | torch.device = "cpu",
  ) -> LifecycleState:
    """Restore CPU next-update state and force a simulator/history restart."""
    state = load_checkpoint(
      path,
      self.trainer,
      self.replay,
      expected_teacher_hashes=teacher_hashes,
      expected_control_contract=control_contract,
      collector=self.collector,
      map_location=map_location,
    )
    self.iteration = state.counters.get("iteration", self.iteration)
    # Always allocate a fresh namespace above the *valid* FIFO records.  The
    # saved namespace may itself be stale after repeated resumes, and unused
    # ring slots must never influence this boundary.
    replay_state = self.replay.state_dict()
    storage = replay_state["storage"]
    if storage is not None and self.replay.size:
      size = self.replay.size
      next_index = int(replay_state["next"])
      start = (next_index - size) % self.replay.capacity
      physical = (torch.arange(size) + start) % self.replay.capacity
      self._segment_namespace = int(storage["episode_id"][physical].max().item()) + 1
    else:
      self._segment_namespace = 0
    self._resume_reset_pending = True
    self.events.append(
      {
        "event": "checkpoint_resumed",
        "iteration": self.iteration,
        "simulator_restart": True,
        "segment_identity": "new adapter reset generation; no bitwise continuation claimed",
      }
    )
    return state


__all__ = [
  "DistillationRunner",
  "LifecycleIteration",
  "RunnerConfig",
  "RunnerValidationError",
]
