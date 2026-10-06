"""Lifecycle tests for exclusive cache publication and token-free provenance."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import pytest

from mjlab.tasks.tracking.diffusion.token_cache_lifecycle import (
  CacheProvenanceError,
  parse_splits,
  prepare_token_cache,
  provenance_cache_path,
  validate_token_cache_read_only,
)
from mjlab.tasks.tracking.diffusion.window_dataset import (
  CacheLock,
  DatasetSource,
  TokenCacheBusyError,
  iter_index_provenance,
  iter_window_provenance,
)

PILOT = Path("logs/diffusion/d1-pilot-20260930b")
CONTRACT = Path("docs/plans/beyondmimic_diffusion_d0_contract.yaml")


def _owned_source(tmp_path: Path) -> DatasetSource:
  source = DatasetSource.resolve(PILOT, contract_path=CONTRACT)
  dataset = tmp_path / "dataset"
  (dataset / "offline").mkdir(parents=True)
  (dataset / "store").symlink_to((source.directory / "store").resolve())
  for name in ("assignments.json", "dataset.json", "projection.npz"):
    (dataset / "offline" / name).write_bytes(
      (source.directory / "offline" / name).read_bytes()
    )
  return DatasetSource(dataset, source.contract_path)


def _hold_lock(path: str, ready: object, release: object) -> None:
  # ``multiprocessing`` events are intentionally typed as object here because
  # their concrete synchronized type differs between fork and spawn contexts.
  with CacheLock(path, timeout=2.0):
    ready.set()  # type: ignore[attr-defined]
    release.wait(2.0)  # type: ignore[attr-defined]


def test_cache_lock_excludes_a_second_process(tmp_path: Path) -> None:
  context = multiprocessing.get_context("spawn")
  ready = context.Event()
  release = context.Event()
  path = tmp_path / "tokens.npy"
  process = context.Process(target=_hold_lock, args=(str(path), ready, release))
  process.start()
  try:
    assert ready.wait(5.0)
    with pytest.raises(TokenCacheBusyError, match="busy"):
      with CacheLock(path, timeout=0.1):
        pass
  finally:
    release.set()
    process.join(5.0)
    assert process.exitcode == 0


def test_provenance_matches_historical_materialization_without_sampling(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  source = _owned_source(tmp_path)
  loaded = source.load()
  historical = list(iter_window_provenance(loaded, "train", limit=8))

  def fail_sample(_self: object, _ref: object) -> object:
    raise AssertionError("provenance path called WindowDataset._sample")

  from mjlab.tasks.tracking.diffusion import dataset as dataset_module

  monkeypatch.setattr(dataset_module.WindowDataset, "_sample", fail_sample)
  direct = list(iter_index_provenance(loaded, "train", limit=8))
  assert direct == historical


def test_prepare_reuses_v2_cache_and_sidecar_without_rewriting(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  source = _owned_source(tmp_path)
  loaded = source.load()
  first = prepare_token_cache(
    source,
    loaded,
    "train",
    motion_ids=["tennis_000"],
    max_cache_gib=0.1,
  )
  assert first.cache_status == "built"
  assert first.provenance_status == "built"
  sidecar = provenance_cache_path(first.path)
  assert sidecar.is_file()
  before = {
    first.path: first.path.stat().st_mtime_ns,
    first.path.with_suffix(".json"): first.path.with_suffix(".json").stat().st_mtime_ns,
    sidecar: sidecar.stat().st_mtime_ns,
  }
  from mjlab.tasks.tracking.diffusion import dataset as dataset_module

  monkeypatch.setattr(
    dataset_module.WindowDataset,
    "_sample",
    lambda *_args: (_ for _ in ()).throw(
      AssertionError("cache reuse recomputed token samples")
    ),
  )
  second = prepare_token_cache(
    source,
    loaded,
    "train",
    motion_ids=["tennis_000"],
    max_cache_gib=0.1,
  )
  assert second.cache_status == "reused"
  assert second.provenance_status == "reused"
  assert second.provenance == first.provenance
  assert {path: path.stat().st_mtime_ns for path in before} == before
  read_only = validate_token_cache_read_only(
    source, loaded, "train", motion_ids=["tennis_000"]
  )
  assert read_only.cache_status == "ready"
  assert read_only.provenance == first.provenance


def test_failed_provenance_publication_can_retry_without_rebuilding_tokens(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  source = _owned_source(tmp_path)
  loaded = source.load()
  from mjlab.tasks.tracking.diffusion import token_cache_lifecycle as lifecycle

  original = lifecycle._write_provenance

  def fail_once(*_args: object, **_kwargs: object) -> None:
    raise CacheProvenanceError("controlled failure")

  monkeypatch.setattr(lifecycle, "_write_provenance", fail_once)
  with pytest.raises(CacheProvenanceError, match="controlled"):
    prepare_token_cache(source, loaded, "validation", max_cache_gib=0.1)
  monkeypatch.setattr(lifecycle, "_write_provenance", original)
  prepared = prepare_token_cache(source, loaded, "validation", max_cache_gib=0.1)
  assert prepared.cache_status == "reused"
  assert prepared.provenance_status == "built"
  assert not prepared.path.with_suffix(".lock").exists()


def test_provenance_tamper_is_refused(tmp_path: Path) -> None:
  source = _owned_source(tmp_path)
  loaded = source.load()
  prepared = prepare_token_cache(
    source,
    loaded,
    "validation",
    max_cache_gib=0.1,
  )
  sidecar = provenance_cache_path(prepared.path)
  payload = json.loads(sidecar.read_text(encoding="utf-8"))
  payload["records"] = [
    {
      "group_key": "tampered",
      "motion_id": "tampered",
      "phase": "",
      "pair_id": "",
      "start_tick": 0,
      "ou_noise_norm": 0.0,
    }
  ]
  sidecar.write_text(json.dumps(payload), encoding="utf-8")
  with pytest.raises(CacheProvenanceError, match="count|digest"):
    validate_token_cache_read_only(source, loaded, "validation")


def test_split_parser_rejects_empty_duplicate_and_unknown_names() -> None:
  assert parse_splits("train, validation") == ("train", "validation")
  for value in ("", "train,train", "missing"):
    with pytest.raises(ValueError):
      parse_splits(value)
