"""Bounded shard-level inspect and replay for raw D1 shard stores."""

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from tracking_diffusion_fixtures import make_row

from mjlab.tasks.tracking.diffusion import (
  AppendOnlyShardStore,
  StorageError,
  offline_validation,
)
from mjlab.tasks.tracking.diffusion.offline_validation import (
  inspect_store,
  replay_store,
  validate_store,
)
from mjlab.tasks.tracking.diffusion.storage import shard_partitions


def _rows(run: str, count: int) -> list:
  return [make_row(index, run=run, episode="episode") for index in range(count)]


def _store(root: Path, rows: list, *, shard_rows: int = 3) -> AppendOnlyShardStore:
  store = AppendOnlyShardStore(root, max_rows_per_shard=shard_rows)
  store.append(rows)
  return store


def _rewrite_shard(path: Path, mutate) -> None:
  """Rewrite one shard's arrays and keep the manifest hash truthful."""
  with np.load(path, allow_pickle=False) as data:
    payload = {name: data[name] for name in data.files}
  mutate(payload)
  np.savez_compressed(path, **payload)
  manifest_path = path.parent / "manifest.json"
  manifest = json.loads(manifest_path.read_text())
  for item in manifest["shards"]:
    if item["file"] == path.name:
      item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
  manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2))


def test_inspect_and_replay_keep_their_payload_shape(tmp_path) -> None:
  store = _store(tmp_path / "store", _rows("run", 7), shard_rows=3)
  inspect = inspect_store(store.root)
  assert set(inspect) == {
    "contract_sha256",
    "store",
    "shards",
    "rows",
    "controllers",
    "unique_identities",
  }
  assert inspect["store"] == str(store.root)
  assert inspect["shards"] == store.shard_count
  assert inspect["rows"] == 7
  assert inspect["controllers"] == {"vae": 7, "ppo": 0}
  assert inspect["unique_identities"] is True

  replay = replay_store(store.root)
  assert set(replay) == {"ok", "rows", "errors", "note"}
  assert replay["ok"] is True
  assert replay["rows"] == 7
  assert replay["errors"] == []
  assert replay["note"] == "offline shard replay diagnostics; no simulator was launched"


def test_serial_and_parallel_reductions_are_identical(tmp_path) -> None:
  store = _store(tmp_path / "store", _rows("run", 11), shard_rows=3)
  serial_inspect = inspect_store(store.root, jobs=1)
  serial_replay = replay_store(store.root, jobs=1)
  for jobs in (2, 3, 4):
    assert inspect_store(store.root, jobs=jobs) == serial_inspect
    assert replay_store(store.root, jobs=jobs) == serial_replay
  single_shard = _store(tmp_path / "single", _rows("single", 2), shard_rows=8)
  assert inspect_store(single_shard.root, jobs=4) == inspect_store(
    single_shard.root, jobs=1
  )


def test_replay_reports_row_errors_in_order_for_serial_and_parallel(tmp_path) -> None:
  rows = _rows("run", 6)
  # Row 1 breaks the executed = clean + noise arithmetic.
  rows[1] = dataclasses.replace(rows[1], executed_action=rows[1].clean_action)
  store = _store(tmp_path / "store", rows, shard_rows=2)

  def corrupt(payload):
    # Row 2 becomes a PPO row that still carries a latent; row 3 stays a VAE row
    # but loses its latent flag.
    fields = json.loads(str(payload["metadata"][0]))
    fields["controller"] = "ppo"
    payload["metadata"][0] = json.dumps(fields, sort_keys=True)
    payload["latent_present"][0] = 1
    payload["latent_present"][1] = 0

  _rewrite_shard(store.root / "shard-000001.npz", corrupt)
  expected = [
    f"{rows[1].identity}: executed action arithmetic mismatch",
    f"{rows[2].identity}: PPO row has a latent",
    f"{rows[3].identity}: VAE row has no latent",
  ]
  serial = replay_store(store.root, jobs=1)
  parallel = replay_store(store.root, jobs=3)
  assert serial == parallel
  assert serial["ok"] is False
  assert serial["rows"] == 6
  assert serial["errors"] == expected
  # The row reader still fails closed on the same shard.
  with pytest.raises(StorageError):
    tuple(store.iter_rows())


def test_inspect_fails_closed_on_controller_latent_inconsistency(tmp_path) -> None:
  """``inspect`` must refuse controller/latent disagreement, not only replay it.

  The historical inspect built row objects, so a VAE row without a latent or a
  PPO row with one raised before any payload was printed.  Replay keeps
  reporting the same rows as errors, and the reduction separates that subset.
  """
  rows = _rows("run", 6)
  store = _store(tmp_path / "store", rows, shard_rows=2)

  def corrupt(payload):
    fields = json.loads(str(payload["metadata"][0]))
    fields["controller"] = "ppo"
    payload["metadata"][0] = json.dumps(fields, sort_keys=True)
    payload["latent_present"][0] = 1
    payload["latent_present"][1] = 0

  _rewrite_shard(store.root / "shard-000001.npz", corrupt)
  expected = [
    f"{rows[2].identity}: PPO row has a latent",
    f"{rows[3].identity}: VAE row has no latent",
  ]
  for jobs in (1, 2, 3):
    with pytest.raises(StorageError, match="controller/latent"):
      inspect_store(store.root, jobs=jobs)

  serial = validate_store(store.root, jobs=1)
  parallel = validate_store(store.root, jobs=3)
  assert serial.errors == tuple(expected)
  assert serial.semantic_errors == tuple(expected)
  assert parallel.semantic_errors == serial.semantic_errors
  replay = replay_store(store.root, jobs=3)
  assert replay["ok"] is False
  assert replay["errors"] == expected
  # The row reader still fails closed on the same shard.
  with pytest.raises(StorageError):
    tuple(store.iter_rows())


def test_each_worker_reads_only_its_assigned_shards(tmp_path, monkeypatch) -> None:
  store = _store(tmp_path / "store", _rows("run", 12), shard_rows=3)
  log = tmp_path / "reads.log"
  real = offline_validation.load_shard_arrays

  def recording(root, contract, entry):
    with log.open("a") as handle:
      handle.write(f"{os.getpid()} {entry.file}\n")
    return real(root, contract, entry)

  monkeypatch.setattr(offline_validation, "load_shard_arrays", recording)
  validation = validate_store(store.root, jobs=3)

  entries = store.shard_entries()
  assigned = [
    [entry.file for entry in entries[start:stop]]
    for start, stop in shard_partitions(entries, 3)
  ]
  reads: dict[int, list[str]] = {}
  for line in log.read_text().splitlines():
    pid, name = line.split()
    reads.setdefault(int(pid), []).append(name)
  # Each worker read exactly the shards of its partition, once, and nothing else.
  assert sorted(reads.values()) == sorted(assigned)
  assert sorted(name for names in reads.values() for name in names) == [
    entry.file for entry in entries
  ]
  assert validation.rows == 12


def test_validation_reports_cross_shard_duplicate_identities(tmp_path) -> None:
  store = _store(tmp_path / "store", _rows("run", 4), shard_rows=2)
  shard = store.root / "shard-000000.npz"
  copy = store.root / "shard-000002.npz"
  copy.write_bytes(shard.read_bytes())
  manifest_path = store.root / "manifest.json"
  manifest = json.loads(manifest_path.read_text())
  manifest["shards"].append({**manifest["shards"][0], "file": "shard-000002.npz"})
  manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2))

  serial = validate_store(store.root, jobs=1)
  parallel = validate_store(store.root, jobs=2)
  assert serial.unique_identities is False
  assert parallel.unique_identities is False
  assert serial.duplicate_identity == parallel.duplicate_identity
  assert serial.duplicate_identity == make_row(0, run="run", episode="episode").identity
  assert inspect_store(store.root)["unique_identities"] is False
  assert inspect_store(store.root)["rows"] == 6


def test_validation_rejects_hash_and_nonfinite_shards_serial_and_parallel(
  tmp_path,
) -> None:
  store = _store(tmp_path / "store", _rows("run", 6), shard_rows=3)
  shard = store.root / "shard-000001.npz"
  shard.write_bytes(shard.read_bytes() + b"corruption")
  for jobs in (1, 2):
    with pytest.raises(StorageError, match="corrupt shard hash"):
      inspect_store(store.root, jobs=jobs)

  nonfinite = _store(tmp_path / "nonfinite", _rows("run", 6), shard_rows=3)
  _rewrite_shard(
    nonfinite.root / "shard-000001.npz",
    lambda payload: payload.update(
      root_position=np.full_like(payload["root_position"], np.nan)
    ),
  )
  for jobs in (1, 2):
    with pytest.raises(StorageError, match="non-finite values"):
      replay_store(nonfinite.root, jobs=jobs)


def test_validation_rejects_invalid_jobs(tmp_path) -> None:
  store = _store(tmp_path / "store", _rows("run", 4), shard_rows=2)
  for jobs in (0, -1):
    with pytest.raises(StorageError, match="jobs must be one or more"):
      inspect_store(store.root, jobs=jobs)


def test_validation_reports_shard_boundaries_without_building_rows(
  tmp_path, monkeypatch
) -> None:
  rows = _rows("run", 6)
  store = _store(tmp_path / "store", rows, shard_rows=3)

  def forbidden(*args, **kwargs):
    raise AssertionError("validation built whole row objects")

  with monkeypatch.context() as patch:
    patch.setattr(AppendOnlyShardStore, "_read_shard", forbidden)
    patch.setattr(AppendOnlyShardStore, "iter_rows", forbidden)
    validation = validate_store(store.root, jobs=2)

  assert [
    (shard.rows, shard.vae_rows, shard.ppo_rows) for shard in validation.shards
  ] == [(3, 3, 0), (3, 3, 0)]
  for shard, chunk in zip(validation.shards, (rows[:3], rows[3:]), strict=True):
    assert shard.first_identity == chunk[0].identity
    assert shard.last_identity == chunk[-1].identity
    assert shard.first_timestamp == chunk[0].timestamp
    assert shard.last_timestamp == chunk[-1].timestamp
    assert shard.errors == ()
