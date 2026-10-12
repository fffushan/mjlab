"""Lazy 41-step windows, grouped splits and train-only preprocessing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, cast

import numpy as np

from .contract import DEFAULT_CONTRACT, DiffusionContract
from .projection import (
  FeatureMoments,
  FeatureMomentsAccumulator,
  ProjectionBundle,
  ProjectionError,
  combine_feature_moments,
)
from .state import WorldState, world_to_hybrid, world_to_hybrid_windows
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
SegmentKey = tuple[str, int, str, str]


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


def _segment_key_of(row: TrajectoryRow) -> SegmentKey:
  """Return the storage identity that groups rows into one segment."""
  return row.run_id, row.env_id, row.episode_id, row.segment_id


def _row_can_be_indexed(row: TrajectoryRow) -> bool:
  """Return the single-row conditions of one indexable window row.

  The chunk check :meth:`WindowIndex._eligible` and the per-segment scan
  :func:`segment_window_scan` both call this helper, so the two cannot drift.
  """
  return (
    row.controller == "vae"
    and row.latent is not None
    and row.segment_qualified
    and not row.reset
    and not row.teleport
    and not row.reference_boundary
    and row.terminal_evidence is not None
    and row.terminal_evidence.available
    and bool(np.isfinite(row.timestamp))
  )


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


@dataclass(frozen=True, slots=True)
class SegmentWindowScan:
  """One validated segment: rows by tick plus every eligible window start.

  ``eligible_starts[p]`` means that ``rows[p : p + width]`` satisfies every
  condition :meth:`WindowIndex._eligible` applies to a window chunk.  One scan
  validates a whole segment, so the index build, the offline statistics fit and
  every later window lookup do not repeat a 41-row check per window.
  """

  rows: tuple[TrajectoryRow, ...]
  ticks: np.ndarray
  eligible_starts: np.ndarray
  width: int

  def position(self, start_tick: int, length: int) -> int | None:
    """Return the row position of the window at ``start_tick``, or ``None``.

    ``None`` means that this snapshot holds no complete eligible window at that
    tick, which is exactly what a per-window eligibility re-check reports.
    Ticks are unique inside a segment because storage refuses a duplicate row
    identity, so the first row at ``start_tick`` fixes the window position.
    """
    count = len(self.rows)
    if length != self.width or count < self.width:
      return None
    position = int(np.searchsorted(self.ticks, start_tick, side="left"))
    if position + self.width > count or int(self.ticks[position]) != start_tick:
      return None
    if not self.eligible_starts[position]:
      return None
    return position


def segment_window_scan(
  rows: Sequence[TrajectoryRow], contract: DiffusionContract
) -> SegmentWindowScan:
  """Validate the rows of one segment once and mark its eligible windows.

  Rows are ordered by tick, because the index build resolves a window by tick
  range and not by storage position.  Every condition that depends on one row,
  or on two neighbouring rows, becomes one prefix sum over the segment, and the
  frame-rate tolerance uses the window maximum and minimum.  The result is the
  per-window answer of :meth:`WindowIndex._eligible` for the whole segment.
  """
  width = contract.window_steps
  ordered = tuple(sorted(rows, key=lambda row: row.tick))
  count = len(ordered)
  ticks = np.empty(count, dtype=np.int64)
  timestamps = np.empty(count, dtype=np.float64)
  fps = np.empty(count, dtype=np.float64)
  row_ok = np.empty(count, dtype=bool)
  motion = np.empty(count, dtype=np.int64)
  if not ordered:
    return SegmentWindowScan(ordered, ticks, np.zeros(0, dtype=bool), width)
  first_key = _segment_key_of(ordered[0])
  motion_codes: dict[str, int] = {}
  for position, row in enumerate(ordered):
    if _segment_key_of(row) != first_key:
      raise DatasetError("a segment scan received rows from more than one segment")
    ticks[position] = row.tick
    timestamps[position] = row.timestamp
    fps[position] = row.fps
    row_ok[position] = _row_can_be_indexed(row)
    code = motion_codes.get(row.motion_id)
    if code is None:
      code = len(motion_codes)
      motion_codes[row.motion_id] = code
    motion[position] = code

  eligible = np.zeros(count, dtype=bool)
  if count < width:
    return SegmentWindowScan(ordered, ticks, eligible, width)
  starts = np.arange(count - width + 1)
  blocked_rows = np.zeros(count, dtype=np.int64)
  blocked_rows[~row_ok] = 1
  blocked_pairs = np.zeros(count, dtype=np.int64)
  if count > 1:
    blocked_pairs[:-1] = ~(
      (ticks[1:] == ticks[:-1] + 1)
      & (timestamps[1:] > timestamps[:-1])
      & (motion[1:] == motion[:-1])
    )
  row_prefix = np.concatenate(([0], np.cumsum(blocked_rows)))
  pair_prefix = np.concatenate(([0], np.cumsum(blocked_pairs)))
  eligible = (row_prefix[starts + width] - row_prefix[starts] == 0) & (
    pair_prefix[starts + width - 1] - pair_prefix[starts] == 0
  )
  # ``_eligible`` compares every row with the frame rate of the first row of its
  # window.  Equal frame rates satisfy that tolerance for every window; a
  # segment with mixed frame rates keeps the exact comparison.
  if float(fps.max()) != float(fps.min()):
    spans = np.lib.stride_tricks.sliding_window_view(fps, width)
    heads = fps[starts]
    eligible &= (spans.max(axis=1) - heads <= 1e-12) & (
      heads - spans.min(axis=1) <= 1e-12
    )
  return SegmentWindowScan(ordered, ticks, eligible, width)


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
    # Each segment is then validated once: ``_scans`` holds its derived window
    # eligibility, so reading overlapping windows does not repeat a 41-row
    # eligibility scan per window.
    self._segment_cache: dict[SegmentKey, tuple[TrajectoryRow, ...]] | None = None
    self._scans: dict[SegmentKey, SegmentWindowScan] = {}

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
    for segment_rows in by_segment.values():
      refs.extend(_segment_refs(segment_rows, assignments, contract))
    refs.sort(key=_ref_sort_key)
    return cls(store, assignments, refs, contract=contract)

  @staticmethod
  def _eligible(rows: Sequence[TrajectoryRow], contract: DiffusionContract) -> bool:
    if len(rows) != contract.window_steps:
      return False
    first = rows[0]
    if any(
      row.run_id != first.run_id
      or row.env_id != first.env_id
      or row.episode_id != first.episode_id
      or row.segment_id != first.segment_id
      or row.motion_id != first.motion_id
      or not _row_can_be_indexed(row)
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

  def refs_by_segment(
    self, split: SplitName | None = None
  ) -> dict[SegmentKey, tuple[WindowRef, ...]]:
    """Group indexed references by the segment that owns them.

    The index is the immutable window snapshot, so a caller that processes one
    segment at a time - the offline statistics fit, or later token
    materialization - looks its references up here instead of re-deriving
    window boundaries from rows.  The order of :attr:`refs` is preserved inside
    each segment, so the grouped starts are ascending.
    """
    grouped: dict[SegmentKey, list[WindowRef]] = {}
    for ref in self.refs_for(split):
      grouped.setdefault(
        (ref.run_id, ref.env_id, ref.episode_id, ref.segment_id), []
      ).append(ref)
    return {key: tuple(value) for key, value in grouped.items()}

  def coverage(self) -> dict[str, int]:
    return {name: len(self.refs_for(name)) for name in _SPLITS}

  def _segment_rows(self) -> dict[SegmentKey, tuple[TrajectoryRow, ...]]:
    if self._segment_cache is None:
      grouped: dict[SegmentKey, list[TrajectoryRow]] = {}
      for row in self.store.iter_rows():
        grouped.setdefault(_segment_key_of(row), []).append(row)
      self._segment_cache = {
        key: tuple(sorted(value, key=lambda row: row.tick))
        for key, value in grouped.items()
      }
    return self._segment_cache

  def _segment_scan(self, key: SegmentKey) -> SegmentWindowScan:
    """Return the segment's validated window scan, built on first use."""
    scan = self._scans.get(key)
    if scan is None:
      scan = segment_window_scan(self._segment_rows().get(key, ()), self.contract)
      self._scans[key] = scan
    return scan

  def _rows(self, ref: WindowRef) -> tuple[TrajectoryRow, ...]:
    scan = self._segment_scan((ref.run_id, ref.env_id, ref.episode_id, ref.segment_id))
    position = scan.position(ref.start_tick, ref.length)
    if position is None:
      raise DatasetError("indexed window no longer satisfies storage integrity checks")
    return scan.rows[position : position + scan.width]

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


_STATISTICS_BATCH_WINDOWS = 16


@dataclass(frozen=True, slots=True)
class RangeStatistics:
  """Partial train-only statistics of one owned shard range.

  ``windows`` lists the start ticks of every train window of the segments this
  range owns, so the caller can compare the ranges against the indexed window
  snapshot instead of trusting that the two derivations agree.
  """

  windows: tuple[tuple[SegmentKey, np.ndarray], ...]
  state: FeatureMoments
  latent: FeatureMoments


_StatisticsPayload = tuple[
  str, DiffusionContract, DiffusionContract, SplitAssignments, int, int, int
]


def _latent_matrix(rows: Sequence[TrajectoryRow], width: int) -> np.ndarray:
  """Stack the latents of one segment.

  A row without a latent can never be part of an eligible window, so its
  placeholder row is never read by a window that this module materializes.
  """
  matrix = np.zeros((len(rows), width), dtype=np.float64)
  for position, row in enumerate(rows):
    if row.latent is not None:
      matrix[position] = row.latent
  return matrix


def _fit_range_statistics(
  store: AppendOnlyShardStore,
  *,
  contract: DiffusionContract,
  assignments: SplitAssignments,
  shard_start: int,
  shard_stop: int,
  batch: int,
) -> RangeStatistics:
  """Fit the train windows of the segments whose first row is in one range.

  Only the train split is resolved: a validation or test window is never
  stacked, transformed or accumulated, so its values cannot reach the fit.
  Windows are transformed straight out of one batched state per segment, and
  ``batch`` windows are mapped and accumulated as one array, which keeps both
  memory and the number of accumulator merges bounded.  The caller owns the
  store, so the in-process (single-range) fit reuses the caller's open store
  instead of hashing every shard again.
  """
  width = contract.window_steps
  state_moments = FeatureMomentsAccumulator(contract.state_dimension)
  latent_moments = FeatureMomentsAccumulator(contract.latent_dimension)
  windows: list[tuple[SegmentKey, np.ndarray]] = []
  for key, rows in iter_owned_segment_rows(
    store, shard_start=shard_start, shard_stop=shard_stop
  ):
    scan = segment_window_scan(rows, contract)
    positions: list[int] = []
    starts: list[int] = []
    for position in np.flatnonzero(scan.eligible_starts):
      first = scan.rows[position]
      if assignments.split_for(_row_group(first)) != "train":
        continue
      positions.append(int(position))
      starts.append(int(first.tick))
    if not positions:
      continue
    windows.append((key, np.asarray(starts, dtype=np.int64)))
    state = WorldState.stack(row.state for row in scan.rows)
    latent = _latent_matrix(scan.rows, contract.latent_dimension)
    offsets = np.arange(width, dtype=np.int64)[None, :]
    for begin in range(0, len(positions), batch):
      chunk = np.asarray(positions[begin : begin + batch], dtype=np.int64)
      # One vectorized transform covers the whole block of windows, so the fit
      # no longer runs one ``world_to_hybrid`` call per window in Python.
      hybrid = world_to_hybrid_windows(
        state, chunk, center_index=contract.current_index, length=width
      )
      latent_block = latent[chunk[:, None] + offsets]
      state_moments.update(hybrid.reshape(-1, contract.state_dimension))
      latent_moments.update(latent_block.reshape(-1, contract.latent_dimension))
  return RangeStatistics(
    tuple(windows), state_moments.partial(), latent_moments.partial()
  )


def _statistics_range_worker(payload: _StatisticsPayload) -> RangeStatistics:
  """Fit one range in its own process from a trusted, read-only store view.

  The worker opens the store with ``validate_shards=False``: opening a store
  normally hashes every shard, so a partitioned fit would pay one whole-store
  hash pass per worker before reading its own range.  The option only defers
  that pass; every shard the worker actually reads is still hash-checked when it
  is loaded, so a corrupt consumed shard is refused exactly as before.
  """
  (
    store_root,
    store_contract,
    contract,
    assignments,
    shard_start,
    shard_stop,
    batch,
  ) = payload
  store = AppendOnlyShardStore(
    store_root, contract=store_contract, validate_shards=False
  )
  return _fit_range_statistics(
    store,
    contract=contract,
    assignments=assignments,
    shard_start=shard_start,
    shard_stop=shard_stop,
    batch=batch,
  )


def _fit_statistics_ranges(
  payloads: Sequence[_StatisticsPayload],
) -> tuple[RangeStatistics, ...]:
  """Fit each payload in its own process, one process per owned range."""
  import multiprocessing

  try:
    context = multiprocessing.get_context("fork")
  except ValueError as exc:  # pragma: no cover - the supported hosts provide fork
    raise DatasetError(
      "parallel statistics fitting needs the fork start method; use jobs=1"
    ) from exc
  with context.Pool(processes=len(payloads)) as pool:
    return tuple(pool.map(_statistics_range_worker, payloads))


def _combine_statistics(
  results: Sequence[RangeStatistics],
  expected: Mapping[SegmentKey, tuple[WindowRef, ...]],
  contract: DiffusionContract,
) -> ProjectionBundle:
  """Refuse a range partition that disagrees with the index, then reduce."""
  reported: dict[SegmentKey, np.ndarray] = {}
  state_partials: list[FeatureMoments] = []
  latent_partials: list[FeatureMoments] = []
  for result in results:
    for key, starts in result.windows:
      if key in reported:
        raise DatasetError(f"two statistics ranges reported segment {key!r}")
      reported[key] = starts
    state_partials.append(result.state)
    latent_partials.append(result.latent)
  missing = sorted(set(expected) - set(reported))
  unexpected = sorted(set(reported) - set(expected))
  if missing or unexpected:
    raise DatasetError(
      "statistics ranges disagree with the window index: "
      f"missing {missing[:3]}, unexpected {unexpected[:3]}"
    )
  for key, refs in expected.items():
    wanted = np.fromiter(
      (ref.start_tick for ref in refs), dtype=np.int64, count=len(refs)
    )
    if not np.array_equal(reported[key], wanted):
      raise DatasetError(
        f"statistics windows of segment {key!r} differ from the window index"
      )
  state_stats = combine_feature_moments(state_partials, std_floor=contract.std_floor)
  latent_stats = combine_feature_moments(latent_partials, std_floor=contract.std_floor)
  return ProjectionBundle.create(state_stats, latent_stats, contract=contract)


def fit_training_statistics(
  index: WindowIndex,
  *,
  contract: DiffusionContract = DEFAULT_CONTRACT,
  jobs: int = 1,
  batch_windows: int = _STATISTICS_BATCH_WINDOWS,
) -> ProjectionBundle:
  """Fit moments from train windows only; validation/test are never inspected.

  The state transform, the latent stack and both moment accumulators are fed in
  one traversal of one validated segment snapshot, instead of transforming the
  same state once for the state moments and again for the latent moments.

  ``jobs`` splits the store into that many disjoint contiguous shard ranges.
  Each range owns the segments whose first stored row it contains, reads no
  other segment, and reduces its own partial moments; the caller combines the
  partials in range order.  A different partition therefore gives statistics
  that are equal to the single-range fit only to floating-point tolerance, and
  not bit for bit.  ``batch_windows`` bounds how many windows are transformed
  into one accumulator update; the default keeps one batch well below a segment.

  A single range reuses the caller's open store, so ``jobs=1`` never re-hashes
  the store.  A partitioned fit opens each worker's store with manifest-only
  validation, then hash-checks every shard that worker consumes.

  The fitted windows are the indexed snapshot.  The result is refused when the
  derivation and the index disagree about any segment, so a stale index fails
  loudly instead of silently changing the dataset definition.
  """
  if jobs < 1:
    raise DatasetError("jobs must be one or more")
  if batch_windows < 1:
    raise DatasetError("batch_windows must be one or more")
  expected = index.refs_by_segment("train")
  if not expected:
    raise ProjectionError("training split has no eligible windows")
  ranges = shard_partitions(index.store, jobs)
  if not ranges:
    raise ProjectionError("training split has no eligible windows")
  if len(ranges) == 1:
    start, stop = ranges[0]
    results = (
      _fit_range_statistics(
        index.store,
        contract=contract,
        assignments=index.assignments,
        shard_start=start,
        shard_stop=stop,
        batch=batch_windows,
      ),
    )
  else:
    payloads = tuple(
      (
        str(index.store.root),
        index.store.contract,
        contract,
        index.assignments,
        start,
        stop,
        batch_windows,
      )
      for start, stop in ranges
    )
    results = _fit_statistics_ranges(payloads)
  return _combine_statistics(results, expected, contract)


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


_INDEX_SCHEMA = "mjlab-x2-diffusion-window-index-v1"
_INDEX_FILENAME = "window_index.npz"


def window_index_path(directory: str | Path) -> Path:
  """Return the conventional persisted-index path inside a dataset directory."""
  return Path(directory) / "offline" / _INDEX_FILENAME


def store_manifest_sha256(store: AppendOnlyShardStore) -> str:
  """Digest the store manifest, so a stale index cannot be trusted."""
  return hashlib.sha256((Path(store.root) / "manifest.json").read_bytes()).hexdigest()


def _split_names() -> tuple[str, ...]:
  """Split names in a deterministic order.

  ``_SPLITS`` must not be relied on for order: a set would serialise one order
  and compare against another, which would look like a corrupt index.
  """
  return tuple(sorted(_SPLITS))


def _intern(values: Sequence[str]) -> tuple[np.ndarray, dict[str, int]]:
  """Intern repeated identity strings; the per-ref arrays then stay numeric."""
  unique = sorted(set(values))
  return np.asarray(unique, dtype="<U160"), {value: i for i, value in enumerate(unique)}


def save_window_index(
  path: str | Path,
  index: WindowIndex,
  *,
  store_manifest_sha256: str,
  assignments_sha256: str,
) -> dict[str, object]:
  """Persist a window index beside its dataset, atomically.

  The index is derived metadata: references only, no tokens.  Interning the
  repeated identity strings keeps the artifact near 20 MB for 609k windows,
  against the 12-25 minutes that rebuilding it costs.
  """
  refs = index.refs
  run_ids, run_of = _intern([ref.run_id for ref in refs])
  episode_ids, episode_of = _intern([ref.episode_id for ref in refs])
  segment_ids, segment_of = _intern([ref.segment_id for ref in refs])
  group_keys, group_of = _intern([ref.group_key for ref in refs])
  names = _split_names()
  split_of = {name: position for position, name in enumerate(names)}
  coverage = index.coverage()
  arrays: dict[str, np.ndarray] = {
    "schema": np.asarray(_INDEX_SCHEMA),
    "contract_hash": np.asarray(index.contract.identity_hash()),
    "store_manifest_sha256": np.asarray(store_manifest_sha256),
    "assignments_sha256": np.asarray(assignments_sha256),
    "ref_count": np.asarray(len(refs), dtype=np.int64),
    "coverage": np.asarray([coverage[name] for name in names], dtype=np.int64),
    "split_names": np.asarray(names, dtype="<U16"),
    "run_ids": run_ids,
    "episode_ids": episode_ids,
    "segment_ids": segment_ids,
    "group_keys": group_keys,
    "ref_run": np.asarray([run_of[ref.run_id] for ref in refs], dtype=np.int32),
    "ref_episode": np.asarray(
      [episode_of[ref.episode_id] for ref in refs], dtype=np.int32
    ),
    "ref_segment": np.asarray(
      [segment_of[ref.segment_id] for ref in refs], dtype=np.int32
    ),
    "ref_group": np.asarray([group_of[ref.group_key] for ref in refs], dtype=np.int32),
    "ref_env": np.asarray([ref.env_id for ref in refs], dtype=np.int64),
    "ref_start": np.asarray([ref.start_tick for ref in refs], dtype=np.int64),
    "ref_length": np.asarray([ref.length for ref in refs], dtype=np.int32),
    "ref_split": np.asarray([split_of[ref.split] for ref in refs], dtype=np.int8),
  }
  target = Path(path)
  target.parent.mkdir(parents=True, exist_ok=True)
  # Write beside the target, then rename: a reader never sees a partial index.
  temporary = target.with_name(target.name + ".partial")
  with temporary.open("wb") as handle:
    np.savez(handle, **cast(Any, arrays))
  digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
  temporary.replace(target)
  return {
    "path": str(target),
    "sha256": digest,
    "refs": len(refs),
    "coverage": coverage,
    "bytes": target.stat().st_size,
  }


def load_window_index(
  path: str | Path,
  *,
  store: AppendOnlyShardStore,
  assignments: SplitAssignments,
  contract: DiffusionContract = DEFAULT_CONTRACT,
) -> WindowIndex:
  """Load a persisted index, refusing one that cannot describe this store."""
  target = Path(path)
  with np.load(target, allow_pickle=False) as payload:
    schema = str(payload["schema"])
    if schema != _INDEX_SCHEMA:
      raise DatasetError(f"unsupported window index schema {schema!r}")
    if contract.schema_version != store.contract.schema_version:
      raise DatasetError("window index contract identity schema differs")
    if str(payload["store_manifest_sha256"]) != store_manifest_sha256(store):
      raise DatasetError("window index does not describe this store")
    if str(payload["assignments_sha256"]) != assignments.sha256():
      raise DatasetError("window index does not describe these split assignments")
    names = [str(value) for value in payload["split_names"]]
    if tuple(names) != _split_names():
      raise DatasetError("window index split names differ")
    run_ids = payload["run_ids"].tolist()
    episode_ids = payload["episode_ids"].tolist()
    segment_ids = payload["segment_ids"].tolist()
    group_keys = payload["group_keys"].tolist()
    count = int(payload["ref_count"])
    # Hoist every per-ref array out of the npz ONCE.  Subscripting the NpzFile
    # inside the comprehension re-opens the zip member and builds a fresh memmap
    # on each of the millions of reads, which cost about 0.5 ms per access and
    # made loading an index far more expensive than rebuilding it.
    ref_run = payload["ref_run"].tolist()
    ref_env = payload["ref_env"].tolist()
    ref_episode = payload["ref_episode"].tolist()
    ref_segment = payload["ref_segment"].tolist()
    ref_start = payload["ref_start"].tolist()
    ref_split = payload["ref_split"].tolist()
    ref_group = payload["ref_group"].tolist()
    ref_length = payload["ref_length"].tolist()
    for name, values in (
      ("ref_run", ref_run),
      ("ref_env", ref_env),
      ("ref_episode", ref_episode),
      ("ref_segment", ref_segment),
      ("ref_start", ref_start),
      ("ref_split", ref_split),
      ("ref_group", ref_group),
      ("ref_length", ref_length),
    ):
      if len(values) != count:
        raise DatasetError(
          f"window index {name} holds {len(values)} entries for {count} refs"
        )
    refs = tuple(
      WindowRef(
        run_ids[ref_run[position]],
        ref_env[position],
        episode_ids[ref_episode[position]],
        segment_ids[ref_segment[position]],
        ref_start[position],
        cast(SplitName, names[ref_split[position]]),
        group_keys[ref_group[position]],
        ref_length[position],
      )
      for position in range(count)
    )
    recorded = [int(value) for value in payload["coverage"]]
  index = WindowIndex(store, assignments, refs, contract=contract)
  actual = index.coverage()
  expected = {name: recorded[position] for position, name in enumerate(_split_names())}
  if actual != expected:
    raise DatasetError(
      f"window index coverage disagrees with itself: recorded {expected}, got {actual}"
    )
  return index


def _segment_refs(
  segment_rows: Sequence[TrajectoryRow],
  assignments: SplitAssignments,
  contract: DiffusionContract,
) -> list[WindowRef]:
  """Emit every eligible window of one segment.

  Shared by the sequential and the parallel builders, so the two cannot drift.
  The segment is validated once by :func:`segment_window_scan` instead of once
  per window, and the emitted references are in ascending start-tick order.
  """
  scan = segment_window_scan(segment_rows, contract)
  refs: list[WindowRef] = []
  for position in np.flatnonzero(scan.eligible_starts):
    first = scan.rows[position]
    group = _row_group(first)
    refs.append(
      WindowRef(
        first.run_id,
        first.env_id,
        first.episode_id,
        first.segment_id,
        first.tick,
        assignments.split_for(group),
        group,
        scan.width,
      )
    )
  return refs


def _ref_sort_key(ref: WindowRef) -> tuple[object, ...]:
  """Total order over references, shared by both builders.

  This sort is what makes the index independent of the order in which segments
  are visited, and therefore what makes a parallel build comparable to the
  sequential one.
  """
  return (
    ref.split,
    ref.group_key,
    ref.run_id,
    ref.env_id,
    ref.episode_id,
    ref.segment_id,
    ref.start_tick,
  )


def shard_partitions(store: AppendOnlyShardStore, jobs: int) -> list[tuple[int, int]]:
  """Contiguous shard ranges, balanced by the manifest's row counts.

  Shard boundaries do not align with segment boundaries, so a range is only a
  hint about who reads what.  Ownership of a segment is decided by the row that
  starts it, never by which range contains most of it.
  """
  counts = [int(item["rows"]) for item in store.shard_items()]
  if not counts:
    return []
  target = max(1, -(-sum(counts) // jobs))
  parts: list[tuple[int, int]] = []
  start = 0
  weight = 0
  for index, count in enumerate(counts):
    weight += count
    if weight >= target and len(parts) < jobs - 1:
      parts.append((start, index + 1))
      start = index + 1
      weight = 0
  if start < len(counts):
    parts.append((start, len(counts)))
  return parts


def _segment_key(segment: SegmentKey) -> str:
  """Render a segment key in the manifest's identity format."""
  return "|".join(str(part) for part in segment)


def _identity_key(value: object) -> str | None:
  """Read a shard manifest's first/last identity as a segment key."""
  if not isinstance(value, (list, tuple)) or len(value) < 4:
    return None
  return "|".join(str(part) for part in tuple(value)[:4])


def _close_segment(
  key: SegmentKey,
  rows: list[TrajectoryRow],
  owner_shard: int,
  shard_stop: int,
  yielded: set[SegmentKey],
  skip: bool,
) -> tuple[tuple[SegmentKey, tuple[TrajectoryRow, ...]], ...]:
  """Register one finished segment of a range, unless another range owns it.

  Ownership is decided by the stored run, never by where a range boundary
  falls, so the ranges of one partition yield each segment exactly once.
  """
  if skip or owner_shard >= shard_stop:
    return ()
  if key in yielded:
    raise DatasetError(f"segment {key!r} is stored in more than one run")
  yielded.add(key)
  return ((key, tuple(rows)),)


def _refuse_split_segment_runs(items: Sequence[Mapping[str, object]]) -> None:
  """Refuse a segment whose stored rows are not one contiguous shard run.

  The manifest records only shard-boundary identities. This guard can detect
  repeated boundary runs, but not arbitrary A/B/A segment interleaving entirely
  inside shards. Supported collectors store each trial contiguously. Callers
  with hand-built interleaved stores must use the sequential index path.
  A boundary key seen again after it ended is refused, because a range reader
  could otherwise omit one run's windows.
  """
  closed: set[str] = set()
  total = len(items)
  for position, item in enumerate(items):
    first = _identity_key(item["first"])
    last = _identity_key(item["last"])
    for boundary in (first, last):
      if boundary is not None and boundary in closed:
        raise DatasetError(f"segment {boundary!r} is stored in more than one run")
    next_first = (
      _identity_key(items[position + 1]["first"]) if position + 1 < total else None
    )
    for boundary in (first, last):
      if boundary is not None and boundary != next_first:
        closed.add(boundary)


def iter_owned_segment_rows(
  store: AppendOnlyShardStore,
  *,
  shard_start: int,
  shard_stop: int,
) -> Iterator[tuple[SegmentKey, tuple[TrajectoryRow, ...]]]:
  """Yield each complete segment whose first stored row lies in a shard range.

  Contiguous non-overlapping ranges therefore hand every segment of the store
  to exactly one reader: the range that holds the first row of the segment
  reads it, and a range that only holds its tail does not.  A reader holds one
  segment plus the rows of the shards it has read, so its memory stays bounded
  by its own range.

  A segment can be longer than a shard, so a range keeps reading the shards its
  last segment continues into; the manifest's per-shard first identity decides
  that without reading row data.  The direction is not symmetric: the manifest
  of the earlier shards says whether the first segment of this range started
  before it, and a segment stored in two separate runs is refused instead of
  being yielded as a partial segment.
  """
  items = store.shard_items()
  total = len(items)
  if shard_start < 0 or not shard_start < shard_stop <= total:
    raise DatasetError(
      f"shard range [{shard_start}, {shard_stop}) is outside the {total} shard store"
    )
  # A range that read only its own span could yield a partial segment when a
  # segment is stored in more than one run, so refuse that shape up front.
  _refuse_split_segment_runs(items)
  # Segment identities that a shard before this range holds.  A range boundary
  # can cut a segment: the first segment then belongs to the previous range,
  # which reads it to its end.
  earlier_keys: set[str] = set()
  for item in items[:shard_start]:
    for name in ("first", "last"):
      boundary = _identity_key(item[name])
      if boundary is not None:
        earlier_keys.add(boundary)
  current: SegmentKey | None = None
  pending: list[TrajectoryRow] = []
  owner_shard = shard_start
  skip_current = False
  first_group = True
  yielded: set[SegmentKey] = set()
  index = shard_start
  while index < total:
    if index >= shard_stop and (
      current is None or _identity_key(items[index]["first"]) != _segment_key(current)
    ):
      break
    for row in store.iter_rows_range(index, index + 1):
      key = _segment_key_of(row)
      if key == current:
        pending.append(row)
        continue
      if current is not None:
        yield from _close_segment(
          current, pending, owner_shard, shard_stop, yielded, skip_current
        )
        first_group = False
        if owner_shard >= shard_stop:
          # That segment started in a read-ahead shard, so it is not this
          # range's to read: stop here.
          return
      current, pending, owner_shard = key, [row], index
      if owner_shard >= shard_stop:
        # This group starts in a read-ahead shard, so another range owns it.
        return
      skip_current = _segment_key(key) in earlier_keys
      if skip_current and not first_group:
        raise DatasetError(f"segment {key!r} is stored in more than one run")
    index += 1
  if current is not None:
    yield from _close_segment(
      current, pending, owner_shard, shard_stop, yielded, skip_current
    )


def _build_refs_worker(
  payload: tuple[str, DiffusionContract, dict[str, str], int, int],
) -> tuple[WindowRef, ...]:
  """Emit refs for the segments that START inside this worker's shard range.

  Ownership comes from :func:`iter_owned_segment_rows`, so the workers of one
  partition emit each segment exactly once and the union of the workers is the
  store.  Each segment is handed to :func:`_segment_refs`, which is also what
  the sequential builder uses, so the two builders cannot drift.  A segment
  whose stored ticks are not one contiguous run is refused, because a partial
  segment would silently drop the windows the sequential build finds.
  """
  store_root, contract, group_to_split, start, stop = payload
  # A partitioned reader consumes only its own shards, and each consumed shard
  # is hash-checked on read, so construction does not hash the whole store.
  store = AppendOnlyShardStore(store_root, contract=contract, validate_shards=False)
  assignments = SplitAssignments(group_to_split, contract.split_seed)
  refs: list[WindowRef] = []
  for _, rows in iter_owned_segment_rows(store, shard_start=start, shard_stop=stop):
    # A stored segment is one contiguous run of trial-relative ticks.  A range
    # that saw a gapped run would derive fewer windows than the sequential
    # build derives, so refuse it instead of dropping windows silently.
    ticks = [row.tick for row in rows]
    if not ticks or ticks != list(range(ticks[0], ticks[0] + len(ticks))):
      raise DatasetError(
        "parallel index build saw an incomplete segment; refusing to drop windows"
      )
    refs.extend(_segment_refs(rows, assignments, contract))
  return tuple(refs)


def build_window_index(
  store: AppendOnlyShardStore,
  assignments: SplitAssignments,
  *,
  jobs: int = 1,
  contract: DiffusionContract = DEFAULT_CONTRACT,
) -> tuple[WindowIndex, str]:
  """Build the index, partitioned across processes when asked.

  A segment is emitted by exactly the worker whose range contains its first row.
  Partitioning is by shard range, so nothing is assumed about where segment
  boundaries fall.
  """
  if jobs <= 1:
    return WindowIndex.build(store, assignments, contract=contract), "sequential"
  parts = shard_partitions(store, jobs)
  if len(parts) <= 1:
    return (
      WindowIndex.build(store, assignments, contract=contract),
      "sequential-single-part",
    )
  group_to_split = dict(assignments.group_to_split)
  payloads = [
    (str(store.root), contract, group_to_split, start, stop) for start, stop in parts
  ]
  import multiprocessing

  context = multiprocessing.get_context("fork")
  with context.Pool(processes=len(payloads)) as pool:
    collected = pool.map(_build_refs_worker, payloads)
  refs = [ref for part in collected for ref in part]
  refs.sort(key=_ref_sort_key)
  return WindowIndex(
    store, assignments, refs, contract=contract
  ), f"parallel-{len(parts)}"


__all__ = [
  "DatasetError",
  "RangeStatistics",
  "SegmentKey",
  "SegmentWindowScan",
  "SplitAssignments",
  "WindowDataset",
  "WindowIndex",
  "WindowRef",
  "WindowSample",
  "build_dataset",
  "fit_training_statistics",
  "grouped_split",
  "iter_owned_segment_rows",
  "load_window_index",
  "save_window_index",
  "segment_window_scan",
  "store_manifest_sha256",
  "window_index_path",
  "build_window_index",
  "shard_partitions",
]
