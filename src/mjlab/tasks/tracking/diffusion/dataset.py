"""Lazy 41-step windows, grouped splits and train-only preprocessing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Sequence, cast

import numpy as np

from .contract import DEFAULT_CONTRACT, DiffusionContract
from .projection import ProjectionBundle, ProjectionError, fit_feature_stats
from .state import WorldState, world_to_hybrid
from .storage import (
  AppendOnlyShardStore,
  StorageError,
  TrajectoryRow,
  validate_monotonic_timestamps,
)


class DatasetError(ValueError):
  """A row sequence cannot form a valid diffusion dataset."""


SplitName = str
_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class SplitAssignments:
  """Deterministic whole-group assignment made before windows are indexed."""

  group_to_split: Mapping[str, SplitName]
  seed: int = 42

  def __post_init__(self) -> None:
    if any(
      not key or value not in _SPLITS for key, value in self.group_to_split.items()
    ):
      raise DatasetError(
        "split groups must map non-empty keys to train/validation/test"
      )

  def split_for(self, group_key: str) -> SplitName:
    try:
      return self.group_to_split[group_key]
    except KeyError as exc:
      raise DatasetError(f"group {group_key!r} has no split assignment") from exc

  def coverage(self) -> dict[str, int]:
    return {
      name: sum(value == name for value in self.group_to_split.values())
      for name in _SPLITS
    }

  def as_dict(self) -> dict[str, object]:
    return {"seed": self.seed, "groups": dict(sorted(self.group_to_split.items()))}

  @classmethod
  def from_dict(cls, payload: Mapping[str, object]) -> "SplitAssignments":
    if set(payload) != {"seed", "groups"} or not isinstance(payload["groups"], Mapping):
      raise DatasetError("malformed split assignment")
    return cls(
      {str(k): str(v) for k, v in payload["groups"].items()},
      int(cast(int, payload["seed"])),
    )

  def sha256(self) -> str:
    return hashlib.sha256(
      json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


_GroupNode = tuple[str, ...]


def _provenance_phase(row: TrajectoryRow) -> str:
  """Return the stable reference phase, not the clean/OU rollout label."""
  for name in ("reference_phase", "initial_start_frame", "phase"):
    value = row.provenance.get(name, "")
    if value and not (name == "phase" and value in {"clean", "ou"}):
      return value
  return ""


def _provenance_group(row: TrajectoryRow) -> str:
  provenance = row.provenance
  pair_id = provenance.get("pair_id", "")
  initial_state_id = provenance.get("initial_state_id", "")
  if not pair_id and not initial_state_id:
    return ""
  if pair_id:
    identity = ("provenance", row.motion_id, pair_id, initial_state_id)
  else:
    identity = (
      "provenance",
      row.motion_id,
      "",
      initial_state_id,
      _provenance_phase(row),
    )
  return "provenance:" + json.dumps(identity, separators=(",", ":"))


def _row_group(row: TrajectoryRow) -> str:
  if row.group_key:
    return row.group_key
  provenance_group = _provenance_group(row)
  if provenance_group:
    return provenance_group
  return f"episode:{row.run_id}:{row.env_id}:{row.episode_id}"


def _row_group_nodes(row: TrajectoryRow) -> tuple[_GroupNode, ...]:
  """Return group identity plus provenance aliases for one rollout row.

  A pair identity links clean and OU variants.  An initial-state/phase identity
  links exact duplicate initializations even when separate collection pairs
  produced them.  ``grouped_split`` unions these aliases before assigning a
  split, so either provenance relation cannot leak across the boundary.
  """
  group_key = _row_group(row)
  base = ("group", group_key)
  # Collector-generated keys are stable labels, not explicit split overrides.
  if row.group_key and row.provenance.get("group_key_generated") != "true":
    return (base,)
  nodes: list[_GroupNode] = [base]
  provenance = row.provenance
  pair_id = provenance.get("pair_id", "")
  if pair_id:
    nodes.append(("pair", row.motion_id, pair_id))
  initial_state_id = provenance.get("initial_state_id", "")
  if initial_state_id:
    nodes.append(("initial", row.motion_id, initial_state_id, _provenance_phase(row)))
  return tuple(nodes)


def grouped_split(
  rows: Iterable[TrajectoryRow], *, contract: DiffusionContract = DEFAULT_CONTRACT
) -> SplitAssignments:
  """Assign families (including paired clean/OU variants) before windowing."""
  parent: dict[_GroupNode, _GroupNode] = {}
  group_nodes: dict[str, _GroupNode] = {}

  def find(node: _GroupNode) -> _GroupNode:
    parent.setdefault(node, node)
    if parent[node] != node:
      parent[node] = find(parent[node])
    return parent[node]

  def union(left: _GroupNode, right: _GroupNode) -> None:
    left_root, right_root = find(left), find(right)
    if left_root == right_root:
      return
    # Stable root selection makes the component identities independent of row
    # iteration order before the seeded split permutation is applied.
    if right_root < left_root:
      left_root, right_root = right_root, left_root
    parent[right_root] = left_root

  for row in rows:
    nodes = _row_group_nodes(row)
    base = nodes[0]
    for node in nodes:
      find(node)
    for node in nodes[1:]:
      union(base, node)
    group_nodes[nodes[0][1]] = base

  roots = sorted({find(node) for node in group_nodes.values()})
  rng = np.random.Generator(np.random.PCG64(contract.split_seed))
  order = [roots[i] for i in rng.permutation(len(roots))]
  n = len(order)
  # Rounding is deterministic and deliberately reports empty partitions rather
  # than duplicating/leaking a group to make a split appear populated.
  train_n = int(np.floor(n * contract.split_fractions[0]))
  valid_n = int(np.floor(n * contract.split_fractions[1]))
  component_assignment: dict[_GroupNode, SplitName] = {}
  for index, root in enumerate(order):
    component_assignment[root] = (
      "train"
      if index < train_n
      else "validation"
      if index < train_n + valid_n
      else "test"
    )
  assignment = {
    key: component_assignment[find(node)] for key, node in group_nodes.items()
  }
  return SplitAssignments(assignment, contract.split_seed)


@dataclass(frozen=True, slots=True)
class WindowRef:
  """Constant-size index entry; raw overlapping windows are never stored."""

  run_id: str
  env_id: int
  episode_id: str
  segment_id: str
  start_tick: int
  split: SplitName
  group_key: str
  length: int = 41


@dataclass(frozen=True, slots=True)
class WindowSample:
  """Lazy materialization returned by :class:`WindowDataset`."""

  tokens: np.ndarray
  projected_state: np.ndarray
  normalized_latent: np.ndarray
  clean_actions: np.ndarray
  executed_actions: np.ndarray
  rows: tuple[TrajectoryRow, ...]
  current_index: int


class WindowIndex:
  """Index eligible windows while retaining only identities, not token arrays."""

  def __init__(
    self,
    store: AppendOnlyShardStore,
    assignments: SplitAssignments,
    refs: Sequence[WindowRef],
    *,
    contract: DiffusionContract = DEFAULT_CONTRACT,
  ) -> None:
    self.store, self.assignments, self.refs, self.contract = (
      store,
      assignments,
      tuple(refs),
      contract,
    )
    # The index is a snapshot of the store at build time.  Windows are resolved
    # from one row pass instead of re-scanning (and re-decompressing) every shard
    # per window, which made offline statistics fit quadratic in the window count.
    self._segment_cache: (
      dict[tuple[str, int, str, str], tuple[TrajectoryRow, ...]] | None
    ) = None

  @classmethod
  def build(
    cls,
    store: AppendOnlyShardStore,
    assignments: SplitAssignments,
    *,
    contract: DiffusionContract = DEFAULT_CONTRACT,
  ) -> "WindowIndex":
    rows = tuple(store.iter_rows())
    by_segment: dict[tuple[str, int, str, str], list[TrajectoryRow]] = {}
    for row in rows:
      by_segment.setdefault(
        (row.run_id, row.env_id, row.episode_id, row.segment_id), []
      ).append(row)
    refs: list[WindowRef] = []
    width = contract.window_steps
    for segment_rows in by_segment.values():
      segment_rows.sort(key=lambda row: row.tick)
      for start in range(0, max(0, len(segment_rows) - width + 1)):
        chunk = segment_rows[start : start + width]
        first = chunk[0]
        if not cls._eligible(chunk, contract):
          continue
        group = _row_group(first)
        split = assignments.split_for(group)
        refs.append(
          WindowRef(
            first.run_id,
            first.env_id,
            first.episode_id,
            first.segment_id,
            first.tick,
            split,
            group,
            width,
          )
        )
    refs.sort(
      key=lambda ref: (
        ref.split,
        ref.group_key,
        ref.run_id,
        ref.env_id,
        ref.episode_id,
        ref.segment_id,
        ref.start_tick,
      )
    )
    return cls(store, assignments, refs, contract=contract)

  @staticmethod
  def _eligible(rows: Sequence[TrajectoryRow], contract: DiffusionContract) -> bool:
    if len(rows) != contract.window_steps:
      return False
    first = rows[0]
    if any(
      row.controller != "vae"
      or row.run_id != first.run_id
      or row.env_id != first.env_id
      or row.episode_id != first.episode_id
      or row.segment_id != first.segment_id
      or row.motion_id != first.motion_id
      or row.latent is None
      or not row.segment_qualified
      or row.reset
      or row.teleport
      or row.reference_boundary
      or row.terminal_evidence is None
      or not row.terminal_evidence.available
      or not np.isfinite(row.timestamp)
      for row in rows
    ):
      return False
    if any(rows[i + 1].tick != rows[i].tick + 1 for i in range(len(rows) - 1)):
      return False
    try:
      validate_monotonic_timestamps(rows)
    except StorageError:
      return False
    return all(abs(row.fps - first.fps) <= 1e-12 for row in rows)

  def refs_for(self, split: SplitName | None = None) -> tuple[WindowRef, ...]:
    if split is None:
      return self.refs
    if split not in _SPLITS:
      raise DatasetError(f"unknown split {split!r}")
    return tuple(ref for ref in self.refs if ref.split == split)

  def coverage(self) -> dict[str, int]:
    return {name: len(self.refs_for(name)) for name in _SPLITS}

  def _segment_rows(self) -> dict[tuple[str, int, str, str], tuple[TrajectoryRow, ...]]:
    if self._segment_cache is None:
      grouped: dict[tuple[str, int, str, str], list[TrajectoryRow]] = {}
      for row in self.store.iter_rows():
        grouped.setdefault(
          (row.run_id, row.env_id, row.episode_id, row.segment_id), []
        ).append(row)
      self._segment_cache = {
        key: tuple(sorted(value, key=lambda row: row.tick))
        for key, value in grouped.items()
      }
    return self._segment_cache

  def _rows(self, ref: WindowRef) -> tuple[TrajectoryRow, ...]:
    segment = self._segment_rows().get(
      (ref.run_id, ref.env_id, ref.episode_id, ref.segment_id), ()
    )
    rows = tuple(
      row for row in segment if ref.start_tick <= row.tick < ref.start_tick + ref.length
    )
    if len(rows) != ref.length or not self._eligible(rows, self.contract):
      raise DatasetError("indexed window no longer satisfies storage integrity checks")
    return rows

  def __len__(self) -> int:
    return len(self.refs)


class WindowDataset:
  """A split view whose windows and projected tokens are materialized lazily."""

  def __init__(
    self, index: WindowIndex, split: SplitName, statistics: ProjectionBundle
  ) -> None:
    if split not in _SPLITS:
      raise DatasetError(f"unknown split {split!r}")
    self.index, self.split, self.statistics = index, split, statistics
    self.refs = index.refs_for(split)

  def __len__(self) -> int:
    return len(self.refs)

  def _sample(self, ref: WindowRef) -> WindowSample:
    rows = self.index._rows(ref)
    states = WorldState.stack(row.state for row in rows)
    hybrid, _ = world_to_hybrid(states, center_index=self.index.contract.current_index)
    latent = np.stack([row.latent for row in rows])
    projected = self.statistics.project_state(hybrid)
    normalized_latent = self.statistics.normalize_latent(latent)
    tokens = np.concatenate((projected, normalized_latent), axis=-1)
    return WindowSample(
      tokens,
      projected,
      normalized_latent,
      np.stack([row.clean_action for row in rows]),
      np.stack([row.executed_action for row in rows]),
      rows,
      self.index.contract.current_index,
    )

  def __getitem__(self, index: int) -> WindowSample:
    return self._sample(self.refs[index])

  def __iter__(self) -> Iterator[WindowSample]:
    for ref in self.refs:
      yield self._sample(ref)


def fit_training_statistics(
  index: WindowIndex, *, contract: DiffusionContract = DEFAULT_CONTRACT
) -> ProjectionBundle:
  """Fit moments from train windows only; validation/test are never inspected."""
  train_refs = index.refs_for("train")
  if not train_refs:
    raise ProjectionError("training split has no eligible windows")

  def chunks(kind: str) -> Iterator[np.ndarray]:
    for ref in train_refs:
      rows = index._rows(ref)
      hybrid, _ = world_to_hybrid(
        WorldState.stack(row.state for row in rows), center_index=contract.current_index
      )
      if kind == "state":
        yield hybrid
      else:
        yield np.stack([row.latent for row in rows])

  state_stats = fit_feature_stats(chunks("state"), std_floor=contract.std_floor)
  latent_stats = fit_feature_stats(chunks("latent"), std_floor=contract.std_floor)
  return ProjectionBundle.create(state_stats, latent_stats, contract=contract)


def build_dataset(
  store: AppendOnlyShardStore,
  *,
  contract: DiffusionContract = DEFAULT_CONTRACT,
  assignments: SplitAssignments | None = None,
) -> tuple[WindowIndex, SplitAssignments]:
  """Convenience API preserving the required split-before-window ordering."""
  if assignments is None:
    assignments = grouped_split(store.iter_rows(), contract=contract)
  index = WindowIndex.build(store, assignments, contract=contract)
  return index, assignments


__all__ = [
  "DatasetError",
  "SplitAssignments",
  "WindowDataset",
  "WindowIndex",
  "WindowRef",
  "WindowSample",
  "build_dataset",
  "fit_training_statistics",
  "grouped_split",
]
