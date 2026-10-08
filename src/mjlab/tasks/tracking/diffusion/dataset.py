"""Lazy 41-step windows, grouped splits and train-only preprocessing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
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
    np.savez(handle, **arrays)
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
  """
  width = contract.window_steps
  ordered = sorted(segment_rows, key=lambda row: row.tick)
  refs: list[WindowRef] = []
  for start in range(0, max(0, len(ordered) - width + 1)):
    chunk = ordered[start : start + width]
    if not WindowIndex._eligible(chunk, contract):
      continue
    first = chunk[0]
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
        width,
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


def _segment_key(segment: tuple[str, int, str, str]) -> str:
  return "|".join(str(part) for part in segment)


def _identity_key(value: object) -> str | None:
  """Read a shard manifest's first/last identity as a segment key."""
  if not isinstance(value, (list, tuple)) or len(value) < 4:
    return None
  return "|".join(str(part) for part in value[:4])


def _build_refs_worker(
  payload: tuple[str, DiffusionContract, dict[str, str], int, int],
) -> tuple[WindowRef, ...]:
  """Emit refs for the segments that START inside this worker's shard range.

  Rows are trial-relative, so a segment's first row has tick 0.  Ownership is
  decided by that row, never by which range holds most of a segment, so the
  union of the workers is exactly the store.

  A segment can be longer than one shard, so the worker keeps reading further
  shards while the shard it has just read ends inside one of its own segments.
  The manifest's per-shard last identity says which segment a shard ends in, so
  this needs no data read to decide.  If a segment still does not form a
  contiguous tick run, the worker raises instead of emitting a truncated set.
  """
  store_root, contract, group_to_split, start, stop = payload
  store = AppendOnlyShardStore(store_root, contract=contract)
  assignments = SplitAssignments(group_to_split, contract.split_seed)
  items = store.shard_items()
  total = len(items)
  rows_of: dict[tuple[str, int, str, str], list[TrajectoryRow]] = {}
  starts_here: set[tuple[str, int, str, str]] = set()

  def absorb(shard_index: int, owned: bool) -> None:
    for row in store.iter_rows_range(shard_index, shard_index + 1):
      segment = (row.run_id, row.env_id, row.episode_id, row.segment_id)
      rows_of.setdefault(segment, []).append(row)
      if owned and row.tick == 0:
        starts_here.add(segment)

  read_end = stop
  for shard_index in range(start, read_end):
    absorb(shard_index, shard_index < stop)
  while read_end < total:
    boundary = _identity_key(items[read_end - 1].get("last"))
    if boundary is None or boundary not in {_segment_key(s) for s in starts_here}:
      break
    absorb(read_end, False)
    read_end += 1

  refs: list[WindowRef] = []
  for segment in sorted(starts_here):
    rows = sorted(rows_of[segment], key=lambda row: row.tick)
    if [row.tick for row in rows] != list(range(len(rows))):
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
  "SplitAssignments",
  "WindowDataset",
  "WindowIndex",
  "WindowRef",
  "WindowSample",
  "build_dataset",
  "fit_training_statistics",
  "grouped_split",
  "load_window_index",
  "save_window_index",
  "store_manifest_sha256",
  "window_index_path",
  "build_window_index",
  "shard_partitions",
]
