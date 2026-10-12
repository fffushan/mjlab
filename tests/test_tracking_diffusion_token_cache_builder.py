"""Focused gates for bounded token-cache materialization."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
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
  window_dataset,
)
from mjlab.tasks.tracking.diffusion import (
  token_cache_builder as builder,
)
from mjlab.tasks.tracking.diffusion.dataset import (
  DatasetError,
  save_window_index,
  store_manifest_sha256,
)
from mjlab.tasks.tracking.diffusion.storage import TrajectoryRow
from mjlab.tasks.tracking.diffusion.token_cache_builder import (
  build_provenance_parallel,
)
from mjlab.tasks.tracking.diffusion.token_cache_lifecycle import (
  prepare_token_cache,
  provenance_cache_path,
)
from mjlab.tasks.tracking.diffusion.window_dataset import (
  WindowCacheError,
  iter_index_provenance,
)

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


def test_token_worker_does_not_validate_the_whole_store(tmp_path, monkeypatch) -> None:
  """A token worker reads its own shards and hash-checks only those.

  Opening the worker store used to hash every shard of the dataset, once per
  worker.  The manifest-only open removes that cost; each shard the worker
  reads is still hash-checked, so correctness is unchanged.
  """
  from mjlab.tasks.tracking.diffusion import storage as storage_module

  source = _synthetic_source(
    tmp_path,
    {"group-a": "train", "group-b": "train"},
    {"group-a": 59, "group-b": 67},
    shard_rows=5,
  )
  loaded = source.load()
  refs = loaded.refs("train")
  cache = tmp_path / "tokens.npy"
  np.save(
    cache,
    np.zeros(
      (len(refs), loaded.contract.window_steps, loaded.contract.token_dimension),
      dtype=np.float32,
    ),
  )
  payloads = builder.plan_token_cache_workers(
    store=loaded.store,
    refs=refs,
    directory=source.directory,
    contract_path=source.contract_path,
    split="train",
    destination=cache,
    dtype=np.float32,
    jobs=2,
  )

  validations: list[int] = []
  original_validate = AppendOnlyShardStore._validate_manifest

  def counting_validate(self):
    validations.append(1)
    return original_validate(self)

  hashes: list[str] = []
  original_hash = storage_module.sha256_file

  def counting_hash(path):
    hashes.append(str(path))
    return original_hash(path)

  monkeypatch.setattr(AppendOnlyShardStore, "_validate_manifest", counting_validate)
  monkeypatch.setattr(storage_module, "sha256_file", counting_hash)

  written: set[int] = set()
  for payload in payloads:
    written.update(builder._fill_token_cache_range(payload))

  assert validations == []
  assert hashes
  assert written == set(range(len(refs)))


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


def test_parallel_provenance_matches_sequential_for_every_split(tmp_path) -> None:
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
  loaded = source.load()
  checked = 0
  for split in ("test", "train", "validation"):
    if not loaded.refs(split):
      continue
    sequential = tuple(
      iter_index_provenance(loaded, split, motion_ids=None, limit=None)
    )
    parallel = build_provenance_parallel(
      loaded,
      split=split,
      jobs=3,
      directory=source.directory,
      contract_path=source.contract_path,
      cache_path=source.directory / "offline" / "unused.npy",
      dtype=np.float32,
    )
    assert parallel == sequential
    assert len(parallel) == len(loaded.refs(split))
    checked += 1
  assert checked == 3


def test_parallel_prepared_provenance_bytes_match_sequential(tmp_path) -> None:
  source = _synthetic_source(
    tmp_path, {"group-a": "train"}, {"group-a": 59}, shard_rows=7
  )
  loaded = source.load()
  first = prepare_token_cache(source, loaded, "train", jobs=1)
  sidecar = provenance_cache_path(first.path)
  sequential_bytes = sidecar.read_bytes()
  sidecar.unlink()
  second = prepare_token_cache(source, loaded, "train", jobs=4)
  assert second.provenance_status == "built"
  assert second.provenance == first.provenance
  assert provenance_cache_path(second.path).read_bytes() == sequential_bytes


def test_parallel_provenance_rejects_a_motion_filter(tmp_path) -> None:
  source = _synthetic_source(
    tmp_path, {"group-a": "train"}, {"group-a": 59}, shard_rows=7
  )
  loaded = source.load()
  with pytest.raises(DatasetError, match="motion filter"):
    prepare_token_cache(source, loaded, "train", motion_ids=["001"], jobs=4)


def test_reused_cache_skips_the_digest_when_the_file_is_unchanged(
  tmp_path, monkeypatch
) -> None:
  """A second start must not re-read the whole cache."""
  source = _synthetic_source(
    tmp_path, {"group-a": "train"}, {"group-a": 59}, shard_rows=7
  )
  loaded = source.load()
  prepare_token_cache(source, loaded, "train", jobs=1)

  digests: list[Path] = []
  original = window_dataset._sha256_file

  def traced(path):
    digests.append(Path(path))
    return original(path)

  monkeypatch.setattr(window_dataset, "_sha256_file", traced)
  again = prepare_token_cache(source, loaded, "train", jobs=1)
  assert again.cache_status == "reused"
  assert digests == []


def test_changed_cache_file_is_digest_verified(tmp_path, monkeypatch) -> None:
  """A later modification invalidates the recorded identity and is verified."""
  source = _synthetic_source(
    tmp_path, {"group-a": "train"}, {"group-a": 59}, shard_rows=7
  )
  loaded = source.load()
  prepared = prepare_token_cache(source, loaded, "train", jobs=1)
  os.utime(prepared.path, ns=(0, 0))

  digests: list[Path] = []
  original = window_dataset._sha256_file

  def traced(path):
    digests.append(Path(path))
    return original(path)

  monkeypatch.setattr(window_dataset, "_sha256_file", traced)
  again = prepare_token_cache(source, loaded, "train", jobs=1)
  assert again.cache_status == "reused"
  assert digests == [prepared.path]


def test_fast_path_accepts_in_place_edit_that_restores_mtime(tmp_path) -> None:
  """Record the limit of the identity check, and that verify_hash closes it."""
  source = _synthetic_source(
    tmp_path, {"group-a": "train"}, {"group-a": 59}, shard_rows=7
  )
  loaded = source.load()
  prepared = prepare_token_cache(source, loaded, "train", jobs=1)
  path = prepared.path
  metadata_path = path.with_suffix(".json")
  before = os.stat(path)

  raw = bytearray(path.read_bytes())
  raw[-1] ^= 0xFF
  path.write_bytes(bytes(raw))
  os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))

  metadata = json.loads(metadata_path.read_text())
  window_dataset._validate_cache(path, metadata_path, metadata)
  with pytest.raises(WindowCacheError):
    window_dataset._validate_cache(path, metadata_path, metadata, verify_hash=True)
