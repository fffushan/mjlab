"""D2 dataset loading and cache regression gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from mjlab.tasks.tracking.diffusion import (
  DEFAULT_CONTRACT,
  DatasetError,
  DatasetSource,
  DiffusionContract,
  LoadedDataset,
  TokenWindowDataset,
  WindowCacheError,
  WindowRecord,
  build_token_cache,
  dataset_token_cache_path,
  iter_window_provenance,
  load_window_records,
  token_cache_identity_key,
)

PILOT = Path("logs/diffusion/d1-pilot-20260930b")
CONTRACT_YAML = Path("docs/plans/beyondmimic_diffusion_d0_contract.yaml")


def _dataset_owned_copy(tmp_path: Path, source: DatasetSource) -> DatasetSource:
  """A dataset directory that shares the store but owns its offline directory."""
  dataset = tmp_path / "owned-dataset"
  (dataset / "offline").mkdir(parents=True, exist_ok=True)
  (dataset / "store").symlink_to((source.directory / "store").resolve())
  for name in ("assignments.json", "dataset.json", "projection.npz"):
    origin = source.directory / "offline" / name
    (dataset / "offline" / name).write_bytes(origin.read_bytes())
  return DatasetSource(dataset, source.contract_path)


def test_token_cache_identity_key_is_stable_and_identity_sensitive() -> None:
  base = {
    "source_hash": "a" * 64,
    "split": "train",
    "contract_hash": "c" * 64,
    "projection_hash": "e" * 64,
  }
  key = token_cache_identity_key(**base)
  assert key == token_cache_identity_key(**base)
  assert len(key) == 16
  assert all(character in "0123456789abcdef" for character in key)
  for field, value in (
    ("source_hash", "b" * 64),
    ("split", "test"),
    ("contract_hash", "d" * 64),
    ("projection_hash", "f" * 64),
  ):
    changed = dict(base)
    changed[field] = value
    assert token_cache_identity_key(**changed) != key, field
  assert token_cache_identity_key(**base, dtype=np.float64) != key
  assert token_cache_identity_key(**base, motion_ids=["tennis_000"]) != key
  assert token_cache_identity_key(**base, limit=10) != key
  # A motion filter is a set, so its order must not change the key.
  assert token_cache_identity_key(**base, motion_ids=["b", "a"]) == (
    token_cache_identity_key(**base, motion_ids=["a", "b"])
  )
  with pytest.raises(DatasetError, match="unknown split"):
    token_cache_identity_key(**{**base, "split": "unknown"})


def test_dataset_token_cache_path_is_owned_by_the_dataset_and_keyed() -> None:
  source = _source()
  loaded = source.load()
  path = dataset_token_cache_path(source, loaded, "train")
  # The cache belongs to the dataset offline directory, not to a training run.
  assert path.parent == source.directory / "offline"
  assert path.suffix == ".npy"
  # The audit in evaluation.py globs offline/tokens-*.npy, so the name must match.
  assert path.match("tokens-*.npy")
  prefix, key = path.stem.rsplit("-", 1)
  assert prefix == "tokens-train"
  assert len(key) == 16
  # The key alone names the identity, and the resolver is deterministic.
  assert path == source.token_cache_path("train", key)
  assert dataset_token_cache_path(source, loaded, "train") == path
  # A different split or motion filter must not collide with this path.
  assert dataset_token_cache_path(source, loaded, "test") != path
  filtered = dataset_token_cache_path(
    source, loaded, "train", motion_ids=["tennis_000"]
  )
  assert filtered != path
  assert filtered.parent == path.parent


def test_parallel_token_cache_matches_the_sequential_cache(tmp_path) -> None:
  owned = _dataset_owned_copy(tmp_path, _source())
  sequential = tmp_path / "sequential.npy"
  parallel = tmp_path / "parallel.npy"
  build_token_cache(owned, "train", sequential, jobs=1)
  build_token_cache(owned, "train", parallel, jobs=4)
  left = np.load(sequential, mmap_mode="r")
  right = np.load(parallel, mmap_mode="r")
  assert left.shape == right.shape
  assert left.dtype == right.dtype
  assert np.array_equal(np.asarray(left), np.asarray(right))
  # Identical bytes means the two caches are interchangeable, digest included.
  assert sequential.stat().st_size == parallel.stat().st_size
  assert (
    json.loads(sequential.with_suffix(".json").read_text())["content_sha256"]
    == json.loads(parallel.with_suffix(".json").read_text())["content_sha256"]
  )


def test_parallel_token_cache_refuses_a_motion_filter(tmp_path) -> None:
  owned = _dataset_owned_copy(tmp_path, _source())
  with pytest.raises(DatasetError, match="motion filter"):
    build_token_cache(
      owned, "train", tmp_path / "filtered.npy", motion_ids=["tennis_000"], jobs=2
    )
  with pytest.raises(DatasetError, match="jobs"):
    build_token_cache(owned, "train", tmp_path / "zero.npy", jobs=0)


def test_dataset_scoped_cache_builds_once_and_is_reused(tmp_path) -> None:
  owned = _dataset_owned_copy(tmp_path, _source())
  loaded = owned.load()
  path = dataset_token_cache_path(owned, loaded, "train", motion_ids=["tennis_000"])
  assert not path.exists()
  first_path, first_count = build_token_cache(
    owned, "train", path, motion_ids=["tennis_000"]
  )
  assert first_path == path
  assert path.with_suffix(".json").is_file()
  before = path.stat()
  second_path, second_count = build_token_cache(
    owned, "train", path, motion_ids=["tennis_000"]
  )
  after = second_path.stat()
  assert second_path == path
  assert second_count == first_count
  # Reuse must not rewrite the file.
  assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)
  # Every cache in the dataset offline directory matches the audit glob.
  assert [cache.name for cache in sorted(path.parent.glob("tokens-*.npy"))] == [
    path.name
  ]


def _source() -> DatasetSource:
  return DatasetSource.resolve(PILOT, contract_path=CONTRACT_YAML)


def test_yaml_and_default_contract_identity_hashes_are_pinned() -> None:
  yaml_contract = DiffusionContract.from_yaml(CONTRACT_YAML)
  assert yaml_contract.identity_hash() == (
    "fc0bc53d0efb1a58f700f78716e5cf36dca24d527c2a7d9b46bffe49cc9a23de"
  )
  assert DEFAULT_CONTRACT.identity_hash() == (
    "eae0d2fff601f85742306310cfac1f7332dfdeddba4fc59009595a9ad632fc76"
  )
  assert yaml_contract.identity_hash() != DEFAULT_CONTRACT.identity_hash()


def test_pilot_reopens_with_yaml_contract_and_empty_validation() -> None:
  loaded = _source().load()
  assert loaded.store.row_count == 13_500
  assert loaded.store.contract.identity_hash() == loaded.contract.identity_hash()
  assert loaded.index.coverage() == {"train": 2640, "validation": 0, "test": 1020}
  assert len(loaded.dataset("validation")) == 0
  assert loaded.refs("validation") == ()


def test_token_cache_round_trip_filter_and_stale_metadata_rejection(tmp_path) -> None:
  source = _source()
  path = tmp_path / "tokens-train.npy"
  cache_path, count = build_token_cache(
    source,
    "train",
    path,
    motion_ids=["tennis_000"],
  )
  records = load_window_records(source, "train", motion_ids=["tennis_000"])
  assert cache_path == path
  assert count == len(records) == 1320
  dataset = TokenWindowDataset(
    path,
    records,
    source=source,
    split="train",
    motion_ids=["tennis_000"],
  )
  tokens, weight, index = dataset[0]
  assert tokens.shape == (41, 231)
  assert tokens.dtype == torch.float32
  assert np.isfinite(tokens.numpy()).all()
  assert float(weight) == 1.0
  metadata_path = path.with_suffix(".json")
  metadata = json.loads(metadata_path.read_text())
  assert metadata["content_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
  metadata["projection_hash"] = "stale"
  metadata_path.write_text(json.dumps(metadata))
  with pytest.raises(WindowCacheError, match="metadata|stale"):
    TokenWindowDataset(
      path,
      records,
      source=source,
      split="train",
      motion_ids=["tennis_000"],
    )


def test_token_cache_digest_rejects_array_tampering(tmp_path) -> None:
  source = _source()
  path = tmp_path / "tokens-tampered.npy"
  build_token_cache(source, "train", path, motion_ids=["tennis_000"])
  records = load_window_records(source, "train", motion_ids=["tennis_000"])
  array = np.load(path, mmap_mode="r+")
  array[0, 0, 0] += 100.0
  array.flush()
  del array
  with pytest.raises(WindowCacheError, match="content_sha256|rebuild"):
    TokenWindowDataset(
      path,
      records,
      source=source,
      split="train",
      motion_ids=["tennis_000"],
    )


def test_token_cache_digest_rejects_metadata_tampering(tmp_path) -> None:
  source = _source()
  path = tmp_path / "tokens-metadata-tampered.npy"
  build_token_cache(source, "train", path, motion_ids=["tennis_000"])
  records = load_window_records(source, "train", motion_ids=["tennis_000"])
  metadata_path = path.with_suffix(".json")
  metadata = json.loads(metadata_path.read_text())
  metadata["content_sha256"] = "0" * 64
  metadata_path.write_text(json.dumps(metadata))
  with pytest.raises(WindowCacheError, match="content_sha256|rebuild"):
    TokenWindowDataset(
      path,
      records,
      source=source,
      split="train",
      motion_ids=["tennis_000"],
    )


def test_token_cache_v1_requires_actionable_rebuild(tmp_path) -> None:
  source = _source()
  path = tmp_path / "tokens-v1.npy"
  build_token_cache(source, "train", path, motion_ids=["tennis_000"])
  metadata_path = path.with_suffix(".json")
  metadata = json.loads(metadata_path.read_text())
  metadata["format"] = "mjlab-x2-diffusion-token-cache-v1"
  metadata_path.write_text(json.dumps(metadata))
  with pytest.raises(WindowCacheError, match="v1|rebuild|format"):
    TokenWindowDataset(path, source=source, split="train", motion_ids=["tennis_000"])


def _synthetic_record(index: int, phase: str) -> WindowRecord:
  tokens = np.full((41, 167), index + 0.25, dtype=np.float32)
  return WindowRecord(
    tokens=tokens,
    split="train",
    group_key=f"group-{index}",
    motion_id=f"motion-{index}",
    phase=phase,
    ou_noise_norm=float(index),
    pair_id=f"pair-{index}",
    start_tick=index,
  )


def test_window_provenance_stream_is_token_free_and_weight_equivalent(
  tmp_path, monkeypatch
) -> None:
  """Use a structural bound because peak-memory checks are environment-dependent."""
  from mjlab.tasks.tracking.diffusion import window_dataset as module

  records = [_synthetic_record(0, "clean"), _synthetic_record(1, "ou")]
  loaded = object()
  monkeypatch.setattr(module, "_source_and_loaded", lambda _source: (None, loaded))
  monkeypatch.setattr(
    module,
    "_iter_window_records",
    lambda _loaded, _split, *, motion_ids, limit: iter(records),
  )
  provenance = list(iter_window_provenance(cast(LoadedDataset, loaded), "train"))
  assert len(provenance) == len(records)
  assert all(not hasattr(item, "tokens") for item in provenance)
  assert np.array_equal(
    module.sampling_weights(records, clean_weight=1.0, perturbed_weight=2.0),
    module.sampling_weights(provenance, clean_weight=1.0, perturbed_weight=2.0),
  )

  path = tmp_path / "tokens.npy"
  np.save(path, np.stack([record.tokens for record in records]))
  metadata = module._cache_metadata(
    source_hash="a" * 64,
    split="train",
    contract_hash="b" * 64,
    projection_hash="c" * 64,
    window_count=2,
    dtype=np.dtype(np.float32),
    motion_ids=None,
    limit=None,
    shape=(2, 41, 167),
    created_at="now",
    content_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
  )
  path.with_suffix(".json").write_text(json.dumps(metadata))
  record_dataset = TokenWindowDataset(
    path, records, clean_weight=1.0, perturbed_weight=2.0
  )
  provenance_dataset = TokenWindowDataset(
    path, provenance, clean_weight=1.0, perturbed_weight=2.0
  )
  assert len(record_dataset) == len(provenance_dataset) == 2
  assert np.array_equal(record_dataset.weights, provenance_dataset.weights)


def test_empty_validation_cache_is_a_valid_zero_length_memmap(tmp_path) -> None:
  path, count = build_token_cache(_source(), "validation", tmp_path / "tokens.npy")
  assert count == 0
  dataset = TokenWindowDataset(path, source=_source(), split="validation")
  assert len(dataset) == 0


def test_pilot_token_slice_statistics_are_nonuniform_and_finite() -> None:
  records = load_window_records(_source(), "train", limit=50)
  tokens = np.stack([record.tokens for record in records])
  assert np.isfinite(tokens).all()
  projected_std = float(tokens[..., :64].std())
  identity_std = float(tokens[..., 64:199].std())
  latent_std = float(tokens[..., 199:].std())
  # Historical D0 diagnostics remain byte/semantics compatible.
  assert 12.0 < projected_std < 22.0
  assert 0.3 < identity_std < 1.2
  assert 0.5 < latent_std < 1.5


def test_sampling_filter_is_a_subset_of_unfiltered_records() -> None:
  source = _source()
  all_records = load_window_records(source, "train")
  filtered = load_window_records(source, "train", motion_ids=["tennis_000"])
  assert len(filtered) < len(all_records)
  all_keys = {
    (record.group_key, record.motion_id, record.start_tick) for record in all_records
  }
  assert all(
    (record.group_key, record.motion_id, record.start_tick) in all_keys
    for record in filtered
  )
