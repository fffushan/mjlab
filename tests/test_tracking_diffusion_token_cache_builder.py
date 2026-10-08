"""Focused gates for bounded token-cache materialization."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pytest
from tracking_diffusion_fixtures import make_row

from mjlab.tasks.tracking.diffusion import (
  AppendOnlyShardStore,
  DatasetSource,
  DiffusionContract,
  SplitAssignments,
  WindowDataset,
  WindowIndex,
  build_token_cache,
  fit_training_statistics,
)
from mjlab.tasks.tracking.diffusion import token_cache_builder as builder
from mjlab.tasks.tracking.diffusion.dataset import (
  save_window_index,
  store_manifest_sha256,
)
from mjlab.tasks.tracking.diffusion.storage import TrajectoryRow

CONTRACT_PATH = Path("docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml")


def _synthetic_source(
  tmp_path: Path,
  assignments_payload: dict[str, str],
  lengths: dict[str, int],
  *,
  shard_rows: int = 7,
) -> DatasetSource:
  """Create persisted D1 metadata around deliberately cross-shard segments."""
  contract = DiffusionContract.from_yaml(CONTRACT_PATH)
  dataset = tmp_path / "dataset"
  store = AppendOnlyShardStore(
    dataset / "store", contract=contract, max_rows_per_shard=shard_rows
  )
  rows: list[TrajectoryRow] = []
  for group, length in lengths.items():
    for tick in range(length):
      rows.append(
        make_row(
          tick,
          group=group,
          run=f"run-{group}",
          episode=f"episode-{group}",
        )
      )
  store.append(rows)
  assignments = SplitAssignments(assignments_payload, contract.split_seed)
  index = WindowIndex.build(store, assignments, contract=contract)
  projection = fit_training_statistics(index, contract=contract)
  offline = dataset / "offline"
  offline.mkdir(parents=True)
  (offline / "assignments.json").write_text(
    json.dumps(assignments.as_dict()), encoding="utf-8"
  )
  (offline / "dataset.json").write_text(
    json.dumps({"windows": index.coverage()}), encoding="utf-8"
  )
  projection.save(offline / "projection.npz")
  save_window_index(
    offline / "window_index.npz",
    index,
    store_manifest_sha256=store_manifest_sha256(store),
    assignments_sha256=assignments.sha256(),
  )
  return DatasetSource.resolve(dataset, contract_path=CONTRACT_PATH)


def test_synthetic_cross_shard_parallel_cache_has_exact_sequential_bytes(
  tmp_path,
) -> None:
  source = _synthetic_source(
    tmp_path,
    {
      "group-a": "train",
      "group-b": "train",
      "group-c": "validation",
      "group-d": "test",
    },
    {"group-a": 59, "group-b": 67, "group-c": 73, "group-d": 79},
    shard_rows=7,
  )
  sequential = tmp_path / "sequential.npy"
  parallel = tmp_path / "parallel.npy"
  build_token_cache(source, "train", sequential, jobs=1)
  build_token_cache(source, "train", parallel, jobs=8)

  left = np.load(sequential, mmap_mode="r")
  right = np.load(parallel, mmap_mode="r")
  assert left.dtype == right.dtype == np.dtype(np.float32)
  assert np.array_equal(np.asarray(left), np.asarray(right))
  assert sequential.read_bytes() == parallel.read_bytes()
  assert (
    hashlib.sha256(sequential.read_bytes()).hexdigest()
    == hashlib.sha256(parallel.read_bytes()).hexdigest()
  )
  left_metadata = json.loads(sequential.with_suffix(".json").read_text())
  right_metadata = json.loads(parallel.with_suffix(".json").read_text())
  assert left_metadata["content_sha256"] == right_metadata["content_sha256"]


def test_parallel_dtype_cast_matches_sequential_float16(tmp_path) -> None:
  source = _synthetic_source(
    tmp_path,
    {"group-a": "train"},
    {"group-a": 47},
    shard_rows=5,
  )
  sequential = tmp_path / "sequential-f16.npy"
  parallel = tmp_path / "parallel-f16.npy"
  build_token_cache(source, "train", sequential, dtype=np.float16, jobs=1)
  build_token_cache(source, "train", parallel, dtype=np.float16, jobs=3)
  assert np.array_equal(
    np.load(sequential, mmap_mode="r"), np.load(parallel, mmap_mode="r")
  )
  assert sequential.read_bytes() == parallel.read_bytes()


def test_zero_count_returns_without_creating_a_pool(tmp_path, monkeypatch) -> None:
  source = _synthetic_source(
    tmp_path,
    {"group-a": "train"},
    {"group-a": 47},
  )

  def fail_if_called(_payloads):
    raise AssertionError("zero-count cache must not create worker tasks")

  monkeypatch.setattr(builder, "_run_workers", fail_if_called)
  builder.fill_token_cache_parallel(
    tmp_path / "not-created.npy", source, "validation", 0, np.float32, 16
  )


def test_planner_assigns_each_segment_once_and_reads_boundary_overlap(tmp_path) -> None:
  source = _synthetic_source(
    tmp_path,
    {"group-a": "train", "group-b": "train", "group-c": "train"},
    {"group-a": 59, "group-b": 67, "group-c": 73},
    shard_rows=5,
  )
  loaded = source.load()
  refs = loaded.refs("train")
  payloads = builder.plan_token_cache_workers(
    store=loaded.store,
    refs=refs,
    directory=source.directory,
    contract_path=source.contract_path,
    split="train",
    destination=tmp_path / "tokens.npy",
    dtype=np.float32,
    jobs=4,
  )
  assert 1 <= len(payloads) <= 4
  plans = [plan for payload in payloads for plan in payload.plans]
  assert len(plans) == len({plan.key for plan in plans}) == 3
  positions = [position for plan in plans for position in plan.positions]
  assert sorted(positions) == list(range(len(refs)))
  for payload in payloads:
    assert payload.read_start < payload.read_stop
    assert all(
      payload.read_start <= plan.owner_shard < payload.read_stop
      for plan in payload.plans
    )

  payload = max(
    payloads, key=lambda item: max(plan.required_tick for plan in item.plans)
  )
  calls: list[tuple[int, int]] = []
  original = loaded.store.iter_rows_range

  def tracking_iter_rows_range(start: int, stop: int):
    calls.append((start, stop))
    yield from original(start, stop)

  loaded.store.iter_rows_range = tracking_iter_rows_range  # type: ignore[method-assign]
  builder._required_rows(loaded.store, payload)
  assert calls
  assert calls[0][0] == payload.read_start
  assert all(start >= payload.read_start for start, _ in calls)
  assert all(stop == start + 1 for start, stop in calls)
  # A worker may read beyond its owned range only to finish a required window.
  assert max(stop for _, stop in calls) >= payload.read_stop


def test_bounded_window_index_never_falls_back_to_full_store_scan(
  tmp_path, monkeypatch
) -> None:
  source = _synthetic_source(
    tmp_path,
    {"group-a": "train"},
    {"group-a": 59},
    shard_rows=5,
  )
  loaded = source.load()
  refs = loaded.refs("train")
  payload = builder.plan_token_cache_workers(
    store=loaded.store,
    refs=refs,
    directory=source.directory,
    contract_path=source.contract_path,
    split="train",
    destination=tmp_path / "tokens.npy",
    dtype=np.float32,
    jobs=2,
  )[0]
  rows = builder._required_rows(loaded.store, payload)
  assert all(
    row.tick <= plan.required_tick for plan in payload.plans for row in rows[plan.key]
  )

  def fail_if_full_scan(_index):
    raise AssertionError("bounded workers must not call WindowIndex._segment_rows")

  monkeypatch.setattr(WindowIndex, "_segment_rows", fail_if_full_scan)
  index = builder._BoundedWindowIndex(
    loaded.store,
    refs,
    rows,
    loaded.contract,
  )
  sample = WindowDataset(index, "train", loaded.projection)._sample(refs[0])
  assert sample.tokens.shape == (
    loaded.contract.window_steps,
    loaded.contract.token_dimension,
  )


def test_coverage_validation_rejects_missing_and_duplicate_positions(tmp_path) -> None:
  source = _synthetic_source(
    tmp_path,
    {"group-a": "train", "group-b": "train"},
    {"group-a": 59, "group-b": 67},
    shard_rows=5,
  )
  loaded = source.load()
  refs = loaded.refs("train")
  payloads = builder.plan_token_cache_workers(
    store=loaded.store,
    refs=refs,
    directory=source.directory,
    contract_path=source.contract_path,
    split="train",
    destination=tmp_path / "tokens.npy",
    dtype=np.float32,
    jobs=3,
  )
  valid = [
    tuple(position for plan in payload.plans for position in plan.positions)
    for payload in payloads
  ]
  missing = list(valid)
  missing[0] = missing[0][:-1]
  with pytest.raises(ValueError, match="coverage"):
    builder.validate_token_cache_coverage(len(refs), payloads, missing)

  duplicate = list(valid)
  duplicate[0] = duplicate[0] + (duplicate[0][0],)
  with pytest.raises(ValueError, match="duplicate"):
    builder.validate_token_cache_coverage(len(refs), payloads, duplicate)


def test_worker_failure_joins_owned_children_and_leaves_no_publishable_array(
  tmp_path,
) -> None:
  source = _synthetic_source(
    tmp_path,
    {"group-a": "train", "group-b": "train"},
    {"group-a": 59, "group-b": 67},
    shard_rows=5,
  )
  loaded = source.load()
  count = len(loaded.refs("train"))
  missing_destination = tmp_path / "missing.npy"
  with pytest.raises(ValueError, match="parallel token cache"):
    from mjlab.tasks.tracking.diffusion.window_dataset import (
      _fill_token_cache_parallel,
    )

    _fill_token_cache_parallel(
      missing_destination, source, "train", count, np.float32, jobs=3
    )
  assert not missing_destination.exists()
  assert not [child for child in multiprocessing.active_children() if child.is_alive()]
