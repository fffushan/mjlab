"""Bounded shard-local token-cache materialization.

The cache positions belong to the window index, not to storage order.  This
module therefore assigns each indexed segment to the shard that contains its
first row, then lets one worker own all positions for that segment.  Workers
read only their assigned shard interval and the shards needed to finish their
window rows.
"""

from __future__ import annotations

import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from .contract import DiffusionContract
from .dataset import (
  DatasetError,
  SplitAssignments,
  WindowDataset,
  WindowIndex,
  WindowRef,
  shard_partitions,
)
from .projection import ProjectionBundle
from .storage import AppendOnlyShardStore, TrajectoryRow

if TYPE_CHECKING:
  from .window_dataset import DatasetSource


SegmentKey = tuple[str, int, str, str]
WorkerResult = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _SegmentPlan:
  """Output positions and bounded row extent for one owned segment."""

  key: SegmentKey
  positions: tuple[int, ...]
  refs: tuple[WindowRef, ...]
  owner_shard: int
  required_tick: int


@dataclass(frozen=True, slots=True)
class _WorkerPayload:
  """Pickle-safe input for one spawn worker."""

  directory: str
  contract_path: str
  split: str
  destination: str
  dtype: str
  read_start: int
  read_stop: int
  plans: tuple[_SegmentPlan, ...]


class _BoundedWindowIndex(WindowIndex):
  """Resolve windows from rows supplied by one worker, never from the store."""

  def __init__(
    self,
    store: AppendOnlyShardStore,
    refs: Sequence[WindowRef],
    rows_by_segment: dict[SegmentKey, tuple[TrajectoryRow, ...]],
    contract: DiffusionContract,
  ) -> None:
    # Split assignments are not consulted by WindowDataset._sample.  The
    # worker already received references from one validated split.
    super().__init__(
      store,
      SplitAssignments({}, contract.split_seed),
      refs,
      contract=contract,
    )
    self._bounded_rows = rows_by_segment

  def _rows(self, ref: WindowRef) -> tuple[TrajectoryRow, ...]:
    key = _ref_key(ref)
    segment = self._bounded_rows.get(key, ())
    rows = tuple(
      row for row in segment if ref.start_tick <= row.tick < ref.start_tick + ref.length
    )
    if len(rows) != ref.length or not WindowIndex._eligible(rows, self.contract):
      raise DatasetError(
        "bounded token worker could not resolve an eligible indexed window"
      )
    return rows


def _ref_key(ref: WindowRef) -> SegmentKey:
  return ref.run_id, ref.env_id, ref.episode_id, ref.segment_id


def _row_key(row: TrajectoryRow) -> SegmentKey:
  return row.run_id, row.env_id, row.episode_id, row.segment_id


def _segment_owners(
  store: AppendOnlyShardStore, segment_keys: set[SegmentKey]
) -> dict[SegmentKey, int]:
  """Find the first shard for each referenced segment in one bounded pass."""
  owners: dict[SegmentKey, int] = {}
  for shard_index, _ in enumerate(store.shard_items()):
    for row in store.iter_rows_range(shard_index, shard_index + 1):
      key = _row_key(row)
      if key in segment_keys and key not in owners:
        owners[key] = shard_index
    if len(owners) == len(segment_keys):
      break
  if owners.keys() != segment_keys:
    missing = sorted(segment_keys - owners.keys())
    raise DatasetError(f"indexed segments are missing from storage: {missing[:3]}")
  return owners


def _partition_index(owner: int, partitions: Sequence[tuple[int, int]]) -> int:
  for index, (start, stop) in enumerate(partitions):
    if start <= owner < stop:
      return index
  raise DatasetError(f"segment owner shard {owner} is outside its partitions")


def plan_token_cache_workers(
  *,
  store: AppendOnlyShardStore,
  refs: Sequence[WindowRef],
  directory: str | Path,
  contract_path: str | Path,
  split: str,
  destination: str | Path,
  dtype: Any,
  jobs: int,
) -> tuple[_WorkerPayload, ...]:
  """Plan disjoint segment ownership without retaining source rows."""
  if jobs < 1:
    raise DatasetError("jobs must be one or more")
  if not refs:
    return ()

  positions_by_segment: dict[SegmentKey, list[int]] = {}
  for position, ref in enumerate(refs):
    positions_by_segment.setdefault(_ref_key(ref), []).append(position)
  owners = _segment_owners(store, set(positions_by_segment))
  partitions = shard_partitions(store, jobs)
  if not partitions:
    raise DatasetError("cannot plan token workers for an empty shard store")

  plans_by_partition: dict[int, list[_SegmentPlan]] = {}
  for key, positions in positions_by_segment.items():
    refs_for_segment = tuple(refs[position] for position in positions)
    required_tick = max(ref.start_tick + ref.length - 1 for ref in refs_for_segment)
    owner = owners[key]
    partition = _partition_index(owner, partitions)
    plans_by_partition.setdefault(partition, []).append(
      _SegmentPlan(
        key,
        tuple(positions),
        refs_for_segment,
        owner,
        required_tick,
      )
    )

  payloads: list[_WorkerPayload] = []
  for partition in sorted(plans_by_partition):
    plans = tuple(
      sorted(
        plans_by_partition[partition], key=lambda plan: (plan.owner_shard, plan.key)
      )
    )
    payloads.append(
      _WorkerPayload(
        str(directory),
        str(contract_path),
        split,
        str(destination),
        np.dtype(dtype).str,
        min(plan.owner_shard for plan in plans),
        max(plan.owner_shard for plan in plans) + 1,
        plans,
      )
    )
  return tuple(payloads)


def _required_rows(
  store: AppendOnlyShardStore, payload: _WorkerPayload
) -> dict[SegmentKey, tuple[TrajectoryRow, ...]]:
  """Read owned rows through the last tick required by this worker."""
  required = {plan.key: plan.required_tick for plan in payload.plans}
  rows: dict[SegmentKey, list[TrajectoryRow]] = {key: [] for key in required}
  shard_count = len(store.shard_items())
  if payload.read_start < 0 or payload.read_stop > shard_count:
    raise DatasetError("token worker shard bounds are invalid")

  shard_index = payload.read_start
  while shard_index < shard_count:
    for row in store.iter_rows_range(shard_index, shard_index + 1):
      key = _row_key(row)
      if key in required and row.tick <= required[key]:
        rows[key].append(row)
    if all(
      values and max(row.tick for row in values) >= required[key]
      for key, values in rows.items()
    ):
      break
    shard_index += 1

  incomplete = [
    key
    for key, values in rows.items()
    if not values or max(row.tick for row in values) < required[key]
  ]
  if incomplete:
    raise DatasetError(
      f"token worker could not read required segment rows: {incomplete[:3]}"
    )
  return {
    key: tuple(sorted(values, key=lambda row: row.tick)) for key, values in rows.items()
  }


def _fill_token_cache_range(payload: _WorkerPayload) -> WorkerResult:
  """Materialize the exact output positions assigned to one worker."""
  if not payload.plans:
    return ()
  contract = DiffusionContract.from_yaml(payload.contract_path)
  directory = Path(payload.directory)
  projection = ProjectionBundle.load(
    directory / "offline" / "projection.npz", contract=contract
  )
  store_root = directory / "store"
  if not (store_root / "manifest.json").is_file():
    raise DatasetError(f"dataset store manifest is missing: {store_root}")
  # A token worker reads only its own segments; each shard it consumes is still
  # hash-checked on read, so opening does not hash the whole store.
  store = AppendOnlyShardStore(store_root, contract=contract, validate_shards=False)
  rows_by_segment = _required_rows(store, payload)
  refs = tuple(ref for plan in payload.plans for ref in plan.refs)
  index = _BoundedWindowIndex(store, refs, rows_by_segment, contract)
  view = WindowDataset(index, payload.split, projection)
  cache_dtype = np.dtype(payload.dtype)
  positions = tuple(position for plan in payload.plans for position in plan.positions)
  if len(positions) != len(set(positions)):
    raise DatasetError("one token worker received duplicate output positions")

  try:
    output = np.load(payload.destination, mmap_mode="r+")
  except (OSError, ValueError) as exc:
    raise DatasetError("token worker could not open the parent cache array") from exc
  try:
    if output.ndim != 3 or np.dtype(output.dtype).str != cache_dtype.str:
      raise DatasetError("parent cache array shape or dtype differs from the worker")
    if positions and max(positions) >= output.shape[0]:
      raise DatasetError("token worker output position is outside the cache array")
    written: list[int] = []
    from .window_dataset import _record_from_sample

    for plan in payload.plans:
      for position, ref in zip(plan.positions, plan.refs, strict=True):
        record = _record_from_sample(view._sample(ref), ref, payload.split)
        output[position] = record.tokens.astype(cache_dtype, copy=False)
        written.append(position)
    output.flush()
    return tuple(written)
  finally:
    del output


def validate_token_cache_coverage(
  expected_count: int,
  payloads: Sequence[_WorkerPayload],
  results: Sequence[WorkerResult],
) -> None:
  """Reject missing or duplicate positions using worker-reported identities."""
  if expected_count < 0 or len(payloads) != len(results):
    raise DatasetError("parallel token cache returned an invalid worker result set")
  expected = set(range(expected_count))
  covered: set[int] = set()
  for payload, result in zip(payloads, results, strict=True):
    assigned = tuple(position for plan in payload.plans for position in plan.positions)
    if len(result) != len(set(result)):
      raise DatasetError("parallel token cache reported duplicate output positions")
    if set(result) != set(assigned):
      raise DatasetError(
        "parallel token cache worker coverage differs from its assigned positions"
      )
    overlap = covered.intersection(result)
    if overlap:
      raise DatasetError(
        f"parallel token cache wrote duplicate positions: {sorted(overlap)[:3]}"
      )
    covered.update(result)
  if covered != expected:
    missing = sorted(expected - covered)
    extra = sorted(covered - expected)
    raise DatasetError(
      f"parallel token cache coverage mismatch; missing={missing[:3]}, extra={extra[:3]}"
    )


def _run_workers(
  payloads: Sequence[_WorkerPayload], worker: Any = None
) -> tuple[Any, ...]:
  """Run only the pool owned by this builder and always join it."""
  if worker is None:
    worker = _fill_token_cache_range
  context = multiprocessing.get_context("spawn")
  pool = context.Pool(processes=len(payloads))
  try:
    results = pool.map(worker, payloads)
  except BaseException:
    pool.terminate()
    pool.join()
    raise
  else:
    pool.close()
    pool.join()
  return tuple(results)


def fill_token_cache_parallel(
  temporary_array: str | Path,
  source: "DatasetSource",
  split: str,
  count: int,
  dtype: Any,
  jobs: int,
) -> None:
  """Fill a parent-created cache using bounded, segment-owned workers."""
  if jobs < 1:
    raise DatasetError("jobs must be one or more")
  if count < 0:
    raise DatasetError("token cache count must be non-negative")
  if count == 0:
    return

  loaded = source.load()
  refs = loaded.refs(split)
  if len(refs) != count:
    raise DatasetError(
      f"parallel token cache count {count} differs from indexed refs {len(refs)}"
    )
  payloads = plan_token_cache_workers(
    store=loaded.store,
    refs=refs,
    directory=source.directory,
    contract_path=source.contract_path,
    split=split,
    destination=temporary_array,
    dtype=dtype,
    jobs=jobs,
  )
  if not payloads:
    raise DatasetError("parallel token cache has no worker payloads")
  results = _run_workers(payloads)
  validate_token_cache_coverage(count, payloads, results)


def _fill_provenance_range(payload: _WorkerPayload) -> tuple[tuple[int, Any], ...]:
  """Return provenance for the exact output positions owned by one worker."""
  if not payload.plans:
    return ()
  from .window_dataset import window_provenance_from_rows

  contract = DiffusionContract.from_yaml(payload.contract_path)
  directory = Path(payload.directory)
  store_root = directory / "store"
  if not (store_root / "manifest.json").is_file():
    raise DatasetError(f"dataset store manifest is missing: {store_root}")
  # A provenance worker reads only its own segments; each shard it consumes is
  # still hash-checked on read, so opening does not hash the whole store.
  store = AppendOnlyShardStore(store_root, contract=contract, validate_shards=False)
  rows_by_segment = _required_rows(store, payload)
  refs = tuple(ref for plan in payload.plans for ref in plan.refs)
  index = _BoundedWindowIndex(store, refs, rows_by_segment, contract)
  records: list[tuple[int, Any]] = []
  for plan in payload.plans:
    for position, ref in zip(plan.positions, plan.refs, strict=True):
      records.append(
        (
          position,
          window_provenance_from_rows(payload.split, ref, index._rows(ref)),
        )
      )
  return tuple(records)


def build_provenance_parallel(
  loaded: Any,
  *,
  split: str,
  jobs: int,
  directory: str | Path,
  contract_path: str | Path,
  cache_path: str | Path,
  dtype: Any,
) -> tuple[Any, ...]:
  """Build window provenance with bounded, segment-owned workers.

  The returned tuple is ordered by output position, so it equals the tuple the
  sequential preparation builds for the same split.
  """
  if jobs < 1:
    raise DatasetError("jobs must be one or more")
  refs = loaded.refs(split)
  if not refs:
    return ()
  payloads = plan_token_cache_workers(
    store=loaded.store,
    refs=refs,
    directory=directory,
    contract_path=contract_path,
    split=split,
    destination=cache_path,
    dtype=dtype,
    jobs=jobs,
  )
  if not payloads:
    raise DatasetError("parallel provenance has no worker payloads")
  results: tuple[tuple[tuple[int, Any], ...], ...] = _run_workers(
    payloads, _fill_provenance_range
  )
  validate_token_cache_coverage(
    len(refs),
    payloads,
    tuple(tuple(sorted(position for position, _ in result)) for result in results),
  )
  ordered: dict[int, Any] = {}
  for result in results:
    for position, record in result:
      ordered[position] = record
  return tuple(ordered[position] for position in range(len(refs)))


__all__ = [
  "build_provenance_parallel",
  "fill_token_cache_parallel",
  "plan_token_cache_workers",
  "validate_token_cache_coverage",
]
