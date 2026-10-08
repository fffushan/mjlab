"""Pure collection, supervision and retention budgets for VAE distillation.

Derived options are opt-in. Legacy explicit totals remain valid, but cannot be
combined with the corresponding derived option. Draws count replay selections,
not distinct rows, and the ratio includes all replay sources.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Sequence


def _integer(name: str, value: int, *, minimum: int = 1) -> int:
  if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
    raise ValueError(f"{name} must be an integer >= {minimum}")
  return value


def _positive_ratio(name: str, value: float) -> Fraction:
  if (
    isinstance(value, bool)
    or not isinstance(value, (int, float))
    or not math.isfinite(value)
    or value <= 0
  ):
    raise ValueError(f"{name} must be finite and positive")
  # Decimal CLI values should not add an update because of binary roundoff.
  return Fraction(str(value))


@dataclass(frozen=True, slots=True)
class SampleBudget:
  """Resolved totals plus the user inputs that produced them.

  The runner still performs a fixed integer number of complete optimizer
  updates. Upward rounding meets the requested draw ratio; it does not change
  the effective batch, loss reduction, sampling distribution or learning rate.
  """

  motion_count: int
  num_envs: int
  collection_steps: int
  minibatch_size: int
  accumulation_steps: int
  updates_per_iteration: int
  replay_capacity: int
  envs_per_motion: int | None
  draw_to_collection_ratio: float | None
  replay_retention_iterations: float | None

  @property
  def fresh_rows(self) -> int:
    return self.num_envs * self.collection_steps

  @property
  def effective_batch_size(self) -> int:
    return self.minibatch_size * self.accumulation_steps

  @property
  def training_draws(self) -> int:
    return self.updates_per_iteration * self.effective_batch_size

  def as_dict(
    self,
    *,
    teacher_ids: Sequence[str],
    slot_counts: Sequence[int],
    weights: Sequence[float],
    replay_quotas: Sequence[int],
  ) -> dict[str, Any]:
    """Report live integer allocations and expected (not measured) draws.

    Sharded callers supply merged live slots, not a hypothetical global
    allocation. Integer allocation can change per-motion ratios for small or
    unevenly weighted cohorts, so the report includes each realized ratio.
    """
    if not all(
      len(values) == self.motion_count
      for values in (teacher_ids, slot_counts, weights, replay_quotas)
    ):
      raise ValueError("sample budget allocations must align with selected motions")
    if sum(slot_counts) != self.num_envs or sum(replay_quotas) != self.replay_capacity:
      raise ValueError("sample budget allocations do not match resolved totals")
    total_weight = math.fsum(weights)
    motions = []
    for name, slots, weight, quota in zip(
      teacher_ids, slot_counts, weights, replay_quotas, strict=True
    ):
      rows = slots * self.collection_steps
      expected_draws = self.training_draws * weight / total_weight
      motions.append(
        {
          "teacher_id": name,
          "num_envs": slots,
          "fresh_rows_per_iteration": rows,
          "expected_training_draws_per_iteration": expected_draws,
          "expected_draw_to_collection_ratio": (
            expected_draws / rows if rows else None
          ),
          "replay_capacity": quota,
          "replay_retention_iterations": quota / rows if rows else None,
        }
      )
    return {
      "version": 1,
      "requested": {
        "envs_per_motion": self.envs_per_motion,
        "draw_to_collection_ratio": self.draw_to_collection_ratio,
        "replay_retention_iterations": self.replay_retention_iterations,
      },
      "resolved": {
        "motion_count": self.motion_count,
        "num_envs": self.num_envs,
        "fresh_rows_per_iteration": self.fresh_rows,
        "effective_batch_size": self.effective_batch_size,
        "updates_per_iteration": self.updates_per_iteration,
        "training_draws_per_iteration": self.training_draws,
        "draw_to_collection_ratio": (
          self.training_draws / self.fresh_rows if self.fresh_rows else None
        ),
        "replay_capacity": self.replay_capacity,
        "replay_retention_iterations": (
          self.replay_capacity / self.fresh_rows if self.fresh_rows else None
        ),
      },
      "collection_budget_scope": (
        "nominal main collection: num_envs * collection_steps; "
        "excludes bootstrap, evaluation and extra coverage rows"
      ),
      "draw_budget_scope": "total replay draws, including any coverage rows",
      "coverage_claim": "expected draw budget, not unique-row or state coverage",
      "motions": motions,
    }


def resolve_sample_budget(
  *,
  motion_count: int,
  collection_steps: int,
  minibatch_size: int,
  accumulation_steps: int,
  num_envs: int | None = None,
  updates_per_iteration: int | None = None,
  replay_capacity: int | None = None,
  envs_per_motion: int | None = None,
  draw_to_collection_ratio: float | None = None,
  replay_retention_iterations: float | None = None,
) -> SampleBudget:
  """Derive totals independently, or keep the historical defaults.

  Equal motion weights give ``envs_per_motion`` rows to each motion when the
  worker layout permits. Unequal weights distribute the total by the existing
  manifest policy; the value is then the mean, not a minimum per motion.
  """
  _integer("motion_count", motion_count)
  _integer("collection_steps", collection_steps, minimum=0)
  _integer("minibatch_size", minibatch_size)
  _integer("accumulation_steps", accumulation_steps)
  for explicit_name, explicit, derived_name, derived in (
    ("num_envs", num_envs, "envs_per_motion", envs_per_motion),
    (
      "updates_per_iteration",
      updates_per_iteration,
      "draw_to_collection_ratio",
      draw_to_collection_ratio,
    ),
    (
      "replay_capacity",
      replay_capacity,
      "replay_retention_iterations",
      replay_retention_iterations,
    ),
  ):
    if explicit is not None and derived is not None:
      raise ValueError(
        f"--{explicit_name.replace('_', '-')} and "
        f"--{derived_name.replace('_', '-')} are mutually exclusive"
      )
  if envs_per_motion is not None:
    num_envs = motion_count * _integer("envs_per_motion", envs_per_motion)
  num_envs = _integer("num_envs", 1 if num_envs is None else num_envs)
  fresh_rows = num_envs * collection_steps
  effective_batch = minibatch_size * accumulation_steps
  if draw_to_collection_ratio is not None or replay_retention_iterations is not None:
    if not fresh_rows:
      raise ValueError("derived draw/retention budgets require collection_steps > 0")
  if draw_to_collection_ratio is not None:
    ratio = _positive_ratio("draw_to_collection_ratio", draw_to_collection_ratio)
    updates_per_iteration = math.ceil(ratio * fresh_rows / effective_batch)
  updates_per_iteration = _integer(
    "updates_per_iteration",
    1 if updates_per_iteration is None else updates_per_iteration,
    minimum=0,
  )
  if replay_retention_iterations is not None:
    horizon = _positive_ratio(
      "replay_retention_iterations", replay_retention_iterations
    )
    replay_capacity = math.ceil(horizon * fresh_rows)
  replay_capacity = _integer(
    "replay_capacity", 16_384 if replay_capacity is None else replay_capacity
  )
  return SampleBudget(
    motion_count=motion_count,
    num_envs=num_envs,
    collection_steps=collection_steps,
    minibatch_size=minibatch_size,
    accumulation_steps=accumulation_steps,
    updates_per_iteration=updates_per_iteration,
    replay_capacity=replay_capacity,
    envs_per_motion=envs_per_motion,
    draw_to_collection_ratio=draw_to_collection_ratio,
    replay_retention_iterations=replay_retention_iterations,
  )
