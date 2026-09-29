import dataclasses
import hashlib
import json

import numpy as np
import pytest
from tracking_diffusion_fixtures import make_row

from mjlab.tasks.tracking.diffusion import (
  AppendOnlyShardStore,
  StorageError,
  TerminalEvidence,
  validate_monotonic_timestamps,
)


def test_append_only_bounded_shards_replay_rows(tmp_path) -> None:
  store = AppendOnlyShardStore(tmp_path, max_rows_per_shard=2)
  rows = [make_row(i) for i in range(5)]
  assert len(store.append(rows)) == 3
  assert store.row_count == 5
  assert [row.tick for row in store.iter_rows()] == list(range(5))
  with pytest.raises(StorageError):
    store.append([make_row(0)])


def test_corrupted_shard_is_rejected(tmp_path) -> None:
  store = AppendOnlyShardStore(tmp_path)
  store.append([make_row(0)])
  shard = tmp_path / "shard-000000.npz"
  shard.write_bytes(shard.read_bytes() + b"corruption")
  with pytest.raises(StorageError):
    tuple(store.iter_rows())


def test_required_physical_timestamps_are_monotonic_without_spacing_assumption() -> (
  None
):
  rows = [make_row(0, timestamp=10.0), make_row(1, timestamp=10.031)]
  validate_monotonic_timestamps(rows)
  with pytest.raises(StorageError):
    validate_monotonic_timestamps(
      [make_row(0, timestamp=2.0), make_row(1, timestamp=2.0)]
    )


def test_shard_zero_quaternion_corruption_fails_closed(tmp_path) -> None:
  store = AppendOnlyShardStore(tmp_path)
  store.append([make_row(0), make_row(1)])
  shard = tmp_path / "shard-000000.npz"
  with np.load(shard, allow_pickle=False) as data:
    payload = {name: data[name] for name in data.files}
  payload["root_quaternion_wxyz"][1] = 0.0
  np.savez_compressed(shard, **payload)
  manifest_path = tmp_path / "manifest.json"
  manifest = json.loads(manifest_path.read_text())
  manifest["shards"][0]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
  manifest_path.write_text(json.dumps(manifest))
  with pytest.raises(StorageError):
    tuple(store.iter_rows())


def test_hash_consistent_malformed_shard_is_storage_error(tmp_path) -> None:
  store = AppendOnlyShardStore(tmp_path)
  store.append([make_row(0)])
  shard = tmp_path / "shard-000000.npz"
  with np.load(shard, allow_pickle=False) as data:
    payload = {name: data[name] for name in data.files if name != "clean_action"}
  replacement = tmp_path / "replacement.npz"
  np.savez_compressed(replacement, **payload)
  shard.write_bytes(replacement.read_bytes())
  manifest_path = tmp_path / "manifest.json"
  manifest = json.loads(manifest_path.read_text())
  manifest["shards"][0]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
  manifest_path.write_text(json.dumps(manifest))
  with pytest.raises(StorageError):
    tuple(store.iter_rows())


def test_hash_consistent_string_terminal_evidence_flags_are_rejected(tmp_path) -> None:
  store = AppendOnlyShardStore(tmp_path)
  store.append([make_row(0)])
  shard = tmp_path / "shard-000000.npz"
  with np.load(shard, allow_pickle=False) as data:
    payload = {name: data[name] for name in data.files}
  fields = json.loads(str(payload["metadata"][0]))
  for name in (
    "available",
    "terminated",
    "truncated",
    "physical_fall",
    "tracking_rejection",
  ):
    fields["terminal_evidence"][name] = "false"
  payload["metadata"] = np.asarray([json.dumps(fields, sort_keys=True)])
  np.savez_compressed(shard, **payload)
  manifest_path = tmp_path / "manifest.json"
  manifest = json.loads(manifest_path.read_text())
  manifest["shards"][0]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
  manifest_path.write_text(json.dumps(manifest))
  with pytest.raises(StorageError, match="terminal evidence field 'available'"):
    tuple(store.iter_rows())


def test_corrupt_npz_container_is_storage_error(tmp_path) -> None:
  store = AppendOnlyShardStore(tmp_path)
  store.append([make_row(0)])
  shard = tmp_path / "shard-000000.npz"
  shard.write_bytes(b"not an npz container")
  manifest_path = tmp_path / "manifest.json"
  manifest = json.loads(manifest_path.read_text())
  manifest["shards"][0]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
  manifest_path.write_text(json.dumps(manifest))
  with pytest.raises(StorageError, match="invalid shard"):
    tuple(store.iter_rows())


def test_terminal_evidence_accepts_numpy_boolean_scalars() -> None:
  payload = TerminalEvidence(True, False, False).as_dict()
  payload["available"] = np.bool_(True)
  payload["terminated"] = np.bool_(False)
  payload["truncated"] = np.bool_(False)
  payload["physical_fall"] = np.bool_(False)
  payload["tracking_rejection"] = np.bool_(False)
  assert TerminalEvidence.from_dict(payload).available


def test_executed_action_matches_independent_clean_plus_noise() -> None:
  row = make_row(3)
  expected = np.add(row.clean_action, row.ou_noise)
  np.testing.assert_array_equal(row.executed_action, expected)


def test_ppo_row_has_no_latent() -> None:
  row = make_row(0)
  with pytest.raises(StorageError):
    dataclasses.replace(row, controller="ppo")


def test_append_does_not_rescan_prior_shards(tmp_path) -> None:
  """``append`` must be incremental, not O(whole store) per call.

  The earlier implementation called ``identities()`` on every append, which
  re-read, re-hashed and decompressed every existing shard: collection cost grew
  linearly per trial, i.e. quadratically over a run.
  """
  from tracking_diffusion_fixtures import make_row

  store = AppendOnlyShardStore(tmp_path, max_rows_per_shard=8)
  store.append([make_row(i, group="a", episode="a") for i in range(8)])

  reads = 0
  original = store._read_shard

  def counting_read_shard(item):
    nonlocal reads
    reads += 1
    return original(item)

  store._read_shard = counting_read_shard  # type: ignore[method-assign]
  for block in range(5):
    base = 8 * (block + 1)
    store.append([make_row(i, group="a", episode="a") for i in range(base, base + 8)])
  # The identity set was seeded by the first append, so later appends read
  # nothing from disk.
  assert reads == 0
  assert store.row_count == 48


def test_shard_read_materialises_each_array_once(tmp_path, monkeypatch) -> None:
  """Reading a shard must not re-decompress arrays per row.

  ``NpzFile.__getitem__`` re-decompresses the whole array on every access.  The
  row loop used to index ``data[name]`` directly, which cost ~4.5 MB per row
  (18.6 GB for a 4096-row shard) and exhausted memory in the offline tools.
  """
  import numpy as np
  from tracking_diffusion_fixtures import make_row

  store = AppendOnlyShardStore(tmp_path, max_rows_per_shard=64)
  store.append([make_row(i, group="a", episode="a") for i in range(64)])

  real_load = np.load
  calls = {"n": 0}

  class _CountingNpz:
    def __init__(self, inner) -> None:
      self._inner = inner

    def __enter__(self):
      self._inner.__enter__()
      return self

    def __exit__(self, *exc):
      return self._inner.__exit__(*exc)

    def __getitem__(self, key):
      calls["n"] += 1
      return self._inner[key]

  monkeypatch.setattr(np, "load", lambda *a, **k: _CountingNpz(real_load(*a, **k)))
  rows = tuple(store.iter_rows())
  assert len(rows) == 64
  # 13 arrays, once each; the per-row loop reuses the materialised references.
  assert calls["n"] <= 32
