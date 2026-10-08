"""Offline fixed-state fidelity evaluation for difficult-state coverage.

This module consumes already-recorded policy tuples and frozen teacher labels. It
never constructs an environment, updates normalizers, or chooses settings from
the hardware fixture. Family-level splitting is explicit so related main/branch
rows cannot straddle train and holdout reports.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import torch

from .coverage_metadata import CoverageBatchMetadata
from .model import ConditionalVAE
from .storage import LabeledReplayBatch

CoverageSplit = Literal["train", "validation", "test"]


class CoverageEvaluationError(ValueError):
  """A fixed-state benchmark is malformed or unsafe to evaluate."""


@dataclass(frozen=True, slots=True)
class CoverageSplitAssignment:
  """Family-disjoint assignment of benchmark rows."""

  split_by_family: dict[str, CoverageSplit]
  row_indices: dict[CoverageSplit, tuple[int, ...]]
  seed: int
  row_family_ids: tuple[str, ...] = ()
  row_ids: tuple[str, ...] = ()

  def __post_init__(self) -> None:
    if set(self.row_indices) != {"train", "validation", "test"}:
      raise CoverageEvaluationError("split assignment must contain all splits")
    if any(
      split not in ("train", "validation", "test")
      for split in self.split_by_family.values()
    ):
      raise CoverageEvaluationError("split assignment contains an unknown split")
    if self.row_family_ids:
      seen: set[int] = set()
      for split, indices in self.row_indices.items():
        for index in indices:
          if index < 0 or index >= len(self.row_family_ids):
            raise CoverageEvaluationError("split contains an out-of-range row")
          if index in seen:
            raise CoverageEvaluationError("a benchmark row appears in two splits")
          seen.add(index)
          family = self.row_family_ids[index]
          if self.split_by_family.get(family) != split:
            raise CoverageEvaluationError("row/family split assignment is inconsistent")
    if self.row_ids and len(self.row_ids) != len(self.row_family_ids):
      raise CoverageEvaluationError("row IDs and family IDs are misaligned")
    if self.row_ids and len(set(self.row_ids)) != len(self.row_ids):
      raise CoverageEvaluationError("row IDs must be unique")


@dataclass(frozen=True, slots=True)
class CoverageMetric:
  """Aggregate and distribution metrics for one evaluated stratum."""

  rows: int
  families: int
  rms: float | None
  p95: float | None
  by_motion: dict[str, dict[str, float]]
  by_source: dict[str, dict[str, float]]
  by_severity: dict[str, dict[str, float]]

  def as_dict(self) -> dict[str, Any]:
    return {
      "rows": self.rows,
      "families": self.families,
      "rms": self.rms,
      "p95": self.p95,
      "by_motion": self.by_motion,
      "by_source": self.by_source,
      "by_severity": self.by_severity,
    }


@dataclass(frozen=True, slots=True)
class CoverageEvaluationReport:
  """A fixed-state report with explicit provenance and no quality gate."""

  split: CoverageSplit
  metrics: CoverageMetric
  target_position: CoverageMetric | None
  model_schema: dict[str, Any]
  normalizers_frozen: bool
  hardware_fixture_included: bool = False
  provenance: str = "teacher_fidelity_on_fixed_states_only_cpu"

  def as_dict(self) -> dict[str, Any]:
    return {
      "split": self.split,
      "metrics": self.metrics.as_dict(),
      "target_position": None
      if self.target_position is None
      else self.target_position.as_dict(),
      "model_schema": self.model_schema,
      "normalizers_frozen": self.normalizers_frozen,
      "hardware_fixture_included": self.hardware_fixture_included,
      "provenance": self.provenance,
      "quality_claim": "teacher_fidelity_on_fixed_states_only",
    }


def make_family_split(
  metadata: CoverageBatchMetadata,
  *,
  seed: int = 0,
  fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> CoverageSplitAssignment:
  """Assign complete source families to train/validation/test deterministically."""
  if not isinstance(metadata, CoverageBatchMetadata) or not len(metadata):
    raise CoverageEvaluationError("family split needs non-empty metadata")
  if len(fractions) != 3 or any(
    not math.isfinite(float(x)) or x < 0 for x in fractions
  ):
    raise CoverageEvaluationError(
      "split fractions must be three finite nonnegative values"
    )
  if not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
    raise CoverageEvaluationError("split fractions must sum to one")
  families = sorted({row.family_id for row in metadata})
  # A per-family hash gives every family a stable bucket.  Unlike shuffling and
  # slicing, adding a new unrelated family cannot move an existing family to a
  # different split.  The requested fractions are therefore expected fractions,
  # not an exact quota for small family sets.
  cumulative = tuple(
    sum(float(fraction) for fraction in fractions[:index]) for index in range(4)
  )
  assignments: dict[str, CoverageSplit] = {}
  for family in families:
    digest = hashlib.sha256(f"coverage-family-v1\0{seed}\0{family}".encode()).digest()
    bucket = int.from_bytes(digest, "big") / float(1 << 256)
    split_index = next(index for index in range(3) if bucket < cumulative[index + 1])
    assignments[family] = ("train", "validation", "test")[split_index]
  rows: dict[CoverageSplit, list[int]] = {"train": [], "validation": [], "test": []}
  for index, row in enumerate(metadata):
    rows[assignments[row.family_id]].append(index)
  return CoverageSplitAssignment(
    assignments,
    {key: tuple(value) for key, value in rows.items()},
    seed,
    tuple(row.family_id for row in metadata),
    tuple(row.row_id for row in metadata),
  )


def _validate_assignment(
  assignment: CoverageSplitAssignment,
  metadata: CoverageBatchMetadata,
) -> None:
  """Reject stale positional assignments before indexing the fixed-state batch."""
  families = tuple(row.family_id for row in metadata)
  row_ids = tuple(row.row_id for row in metadata)
  if assignment.row_family_ids and assignment.row_family_ids != families:
    raise CoverageEvaluationError("split assignment metadata rows are misaligned")
  if assignment.row_ids and assignment.row_ids != row_ids:
    raise CoverageEvaluationError("split assignment row IDs are misaligned")
  if set(assignment.split_by_family) != set(families):
    raise CoverageEvaluationError("split assignment families are stale")
  expected: dict[CoverageSplit, set[int]] = {
    "train": set(),
    "validation": set(),
    "test": set(),
  }
  for index, family in enumerate(families):
    split = assignment.split_by_family.get(family)
    if split not in expected:
      raise CoverageEvaluationError("split assignment contains an unknown family split")
    expected[split].add(index)
  actual = {split: set(indices) for split, indices in assignment.row_indices.items()}
  if set(actual) != set(expected) or actual != expected:
    raise CoverageEvaluationError("split assignment indices are misaligned")


def _summary(values: torch.Tensor, metadata: Sequence[Any]) -> CoverageMetric:
  if values.ndim != 1 or values.shape[0] != len(metadata):
    raise CoverageEvaluationError("metric values and metadata are misaligned")
  if not values.numel():
    return CoverageMetric(0, 0, None, None, {}, {}, {})
  values = values.detach().to(device="cpu", dtype=torch.float64)
  if not bool(torch.isfinite(values).all()):
    raise CoverageEvaluationError("metric values contain non-finite values")

  def grouped(field: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for value, row in zip(values.tolist(), metadata, strict=True):
      groups[str(getattr(row, field))].append(float(value))
    result: dict[str, dict[str, float]] = {}
    for key, items in sorted(groups.items()):
      tensor = torch.tensor(items, dtype=torch.float64)
      result[key] = {
        "rows": float(len(items)),
        "rms": float(torch.sqrt(torch.mean(tensor.square()))),
        "p95": float(torch.quantile(tensor, 0.95)),
      }
    return result

  families = {row.family_id for row in metadata}
  return CoverageMetric(
    len(metadata),
    len(families),
    float(torch.sqrt(torch.mean(values.square()))),
    float(torch.quantile(values, 0.95)),
    grouped("motion_id"),
    grouped("source_kind"),
    grouped("severity_band"),
  )


def evaluate_fixed_states(
  model: ConditionalVAE,
  batch: LabeledReplayBatch,
  metadata: CoverageBatchMetadata,
  *,
  split: CoverageSplit = "test",
  assignment: CoverageSplitAssignment | None = None,
  target_position: Callable[[torch.Tensor], torch.Tensor] | None = None,
  hardware_fixture: bool = False,
) -> CoverageEvaluationReport:
  """Evaluate frozen labels on one family-disjoint fixed-state split.

  ``model`` is run in inference mode with frozen normalizers. The function does
  not call a teacher or update model state; callers must supply labels produced
  by the frozen teacher and may keep the hardware fixture out by passing
  ``hardware_fixture=False`` (the default).
  """
  if hardware_fixture:
    raise CoverageEvaluationError(
      "the hardware fixture is held out and cannot be included"
    )
  if len(metadata) != batch.batch_size:
    raise CoverageEvaluationError("metadata row count does not match benchmark batch")
  if any(
    tensor is not None and tensor.device.type != "cpu"
    for tensor in (
      batch.reference,
      batch.conditioning,
      batch.teacher_action,
      batch.motion_id,
      batch.teacher_id,
      batch.reference_frame,
      batch.episode_id,
      batch.collector_iteration,
      batch.initialization_kind,
      batch.segment_initial_reference_frame,
      batch.segment_age,
    )
  ):
    raise CoverageEvaluationError("fixed-state fidelity evaluation is CPU-only")
  if any(t.device.type != "cpu" for t in model.parameters()) or any(
    t.device.type != "cpu" for t in model.buffers()
  ):
    raise CoverageEvaluationError("fixed-state fidelity evaluation is CPU-only")
  if assignment is None:
    assignment = make_family_split(metadata)
  _validate_assignment(assignment, metadata)
  indices = assignment.row_indices.get(split)
  if indices is None:
    raise CoverageEvaluationError(f"unknown benchmark split {split!r}")
  if not model.reference_normalizer.frozen or not model.conditioning_normalizer.frozen:
    raise CoverageEvaluationError("evaluation requires frozen student normalizers")
  index = torch.tensor(indices, dtype=torch.long, device=batch.device)
  selected = batch if len(indices) == batch.batch_size else _take(batch, index)
  rows = tuple(metadata.rows[i] for i in indices)
  normalizer_states = [
    {name: value.detach().clone() for name, value in normalizer.state_dict().items()}
    for normalizer in (
      model.reference_normalizer,
      model.conditioning_normalizer,
    )
  ]
  rng_state = torch.random.get_rng_state()
  training_modes = {module: module.training for module in model.modules()}
  model.eval()
  try:
    with torch.no_grad():
      predicted = model.mean_inference(selected.reference, selected.conditioning)
    error = (predicted - selected.teacher_action).pow(2).mean(dim=1).sqrt()
    target_metric = None
    if target_position is not None:
      predicted_target = target_position(predicted)
      teacher_target = target_position(selected.teacher_action)
      if predicted_target.shape != teacher_target.shape or predicted_target.shape[
        0
      ] != len(rows):
        raise CoverageEvaluationError(
          "target-position transform returned incompatible shape"
        )
      target_metric = _summary(
        (predicted_target - teacher_target).pow(2).mean(dim=1).sqrt(), rows
      )
    return CoverageEvaluationReport(
      split,
      _summary(error, rows),
      target_metric,
      model.schema_metadata,
      True,
      False,
    )
  finally:
    for module, training in training_modes.items():
      module.training = training
    for normalizer, state in zip(
      (model.reference_normalizer, model.conditioning_normalizer),
      normalizer_states,
      strict=True,
    ):
      normalizer.load_state_dict(state)
    torch.random.set_rng_state(rng_state)


def _take(batch: LabeledReplayBatch, index: torch.Tensor) -> LabeledReplayBatch:
  observations = batch.observations
  from dataclasses import replace

  return replace(
    batch,
    observations=replace(
      observations,
      reference=observations.reference[index],
      conditioning=observations.conditioning[index],
    ),
    teacher_action=batch.teacher_action[index],
    motion_id=batch.motion_id[index],
    teacher_id=batch.teacher_id[index],
    reference_frame=batch.reference_frame[index],
    episode_id=batch.episode_id[index],
    collector_iteration=batch.collector_iteration[index],
    initialization_kind=None
    if batch.initialization_kind is None
    else batch.initialization_kind[index],
    segment_initial_reference_frame=None
    if batch.segment_initial_reference_frame is None
    else batch.segment_initial_reference_frame[index],
    segment_age=None if batch.segment_age is None else batch.segment_age[index],
  )


__all__ = [
  "CoverageEvaluationError",
  "CoverageEvaluationReport",
  "CoverageMetric",
  "CoverageSplitAssignment",
  "evaluate_fixed_states",
  "make_family_split",
]
