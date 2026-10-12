"""Byte-preserving merge of validated per-worker D1 shard stores."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from tracking_diffusion_fixtures import make_row

from mjlab.scripts import diffusion, diffusion_merge
from mjlab.tasks.tracking.diffusion import AppendOnlyShardStore, StorageError
from mjlab.tasks.tracking.diffusion import storage as storage_module
from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT
from mjlab.tasks.tracking.diffusion.storage import (
  manifest_sha256,
  merge_shard_stores,
  merge_shard_stores_legacy,
)


def _rows(run: str, start: int, stop: int, *, env: int = 0) -> list:
  return [
    make_row(index, run=run, episode="episode", env=env) for index in range(start, stop)
  ]


def _store(root: Path, rows: list, *, shard_rows: int = 2) -> AppendOnlyShardStore:
  store = AppendOnlyShardStore(root, max_rows_per_shard=shard_rows)
  store.append(rows)
  return store


def _shard_bytes(root: Path) -> list[bytes]:
  return [path.read_bytes() for path in sorted(root.glob("shard-*.npz"))]


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


def _rewrite_manifest(root: Path, mutate) -> None:
  path = root / "manifest.json"
  manifest = json.loads(path.read_text())
  mutate(manifest)
  path.write_text(json.dumps(manifest, sort_keys=True, indent=2))


def _snapshot(root: Path) -> dict[str, tuple[int, int, str]]:
  snapshot = {}
  for path in sorted(root.rglob("*")):
    if path.is_file():
      stat = path.stat()
      snapshot[str(path.relative_to(root))] = (
        stat.st_size,
        stat.st_mtime_ns,
        hashlib.sha256(path.read_bytes()).hexdigest(),
      )
  return snapshot


def _assert_rows_equal(left, right) -> None:
  assert left.identity == right.identity
  assert left.timestamp == right.timestamp
  assert (left.motion_id, left.reference_frame) == (
    right.motion_id,
    right.reference_frame,
  )
  assert left.controller == right.controller
  assert (left.group_key, left.fps) == (right.group_key, right.fps)
  assert (
    left.segment_qualified,
    left.reset,
    left.teleport,
    left.reference_boundary,
  ) == (
    right.segment_qualified,
    right.reset,
    right.teleport,
    right.reference_boundary,
  )
  assert dict(left.provenance) == dict(right.provenance)
  assert left.terminal_evidence == right.terminal_evidence
  for name in (
    "previous_executed_action",
    "clean_action",
    "ou_noise",
    "executed_action",
  ):
    np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
  if left.latent is None or right.latent is None:
    assert left.latent is None and right.latent is None
  else:
    np.testing.assert_array_equal(left.latent, right.latent)
  for name in (
    "root_position",
    "root_quaternion_wxyz",
    "root_linear_velocity",
    "root_angular_velocity",
    "body_positions",
    "body_linear_velocities",
  ):
    np.testing.assert_array_equal(getattr(left.state, name), getattr(right.state, name))


def test_byte_preserving_merge_never_decodes_or_reserialises_a_row(
  tmp_path, monkeypatch
) -> None:
  left = _store(tmp_path / "left", _rows("left", 0, 5))
  right = _store(tmp_path / "right", _rows("right", 0, 3), shard_rows=4)
  source_bytes = _shard_bytes(left.root) + _shard_bytes(right.root)

  def forbidden(*args, **kwargs):
    raise AssertionError("the byte-preserving merge used the row-level path")

  with monkeypatch.context() as patch:
    patch.setattr(AppendOnlyShardStore, "_read_shard", forbidden)
    patch.setattr(AppendOnlyShardStore, "_write_shard", forbidden)
    report = merge_shard_stores([left.root, right.root], tmp_path / "merged")

  assert report.mode == "shard-copy"
  assert report.input_rows == 8
  assert report.rows_after == 8
  merged = AppendOnlyShardStore(tmp_path / "merged")
  assert merged.row_count == 8
  # Every copied shard is the source shard byte for byte, under a new name.
  assert _shard_bytes(merged.root) == source_bytes
  assert [row.identity for row in merged.iter_rows()] == [
    row.identity for row in (*left.iter_rows(), *right.iter_rows())
  ]


def test_byte_preserving_merge_matches_the_row_level_reference(tmp_path) -> None:
  left = _store(tmp_path / "left", _rows("left", 0, 5))
  right = _store(tmp_path / "right", _rows("right", 0, 3))

  fast = merge_shard_stores([left.root, right.root], tmp_path / "fast")
  legacy = merge_shard_stores_legacy([left.root, right.root], tmp_path / "legacy")

  assert (fast.input_rows, fast.rows_after) == (legacy.input_rows, legacy.rows_after)
  assert fast.inputs == legacy.inputs
  fast_store = AppendOnlyShardStore(tmp_path / "fast")
  legacy_store = AppendOnlyShardStore(tmp_path / "legacy")
  fast_rows = list(fast_store.iter_rows())
  legacy_rows = list(legacy_store.iter_rows())
  assert len(fast_rows) == len(legacy_rows) == 8
  for fast_row, legacy_row in zip(fast_rows, legacy_rows, strict=True):
    _assert_rows_equal(fast_row, legacy_row)


def test_merge_refuses_an_existing_output(tmp_path) -> None:
  source = _store(tmp_path / "source", _rows("source", 0, 2))
  store = tmp_path / "existing"
  store.mkdir()
  (store / "manifest.json").write_text("{}")
  with pytest.raises(StorageError, match="refuses an existing output"):
    merge_shard_stores([source.root], store)
  empty = tmp_path / "empty"
  empty.mkdir()
  with pytest.raises(StorageError, match="refuses an existing output"):
    merge_shard_stores([source.root], empty)
  file_path = tmp_path / "file"
  file_path.write_text("occupied")
  with pytest.raises(StorageError, match="refuses an existing output"):
    merge_shard_stores([source.root], file_path)


def test_merge_refuses_a_repeated_input_store(tmp_path) -> None:
  source = _store(tmp_path / "source", _rows("source", 0, 2))
  with pytest.raises(StorageError, match="repeat the same store"):
    merge_shard_stores([source.root, source.root], tmp_path / "merged")
  assert not (tmp_path / "merged").exists()


def test_merge_refuses_duplicate_identities_across_inputs(tmp_path) -> None:
  left = _store(tmp_path / "left", _rows("shared", 0, 2))
  right = _store(tmp_path / "right", _rows("shared", 0, 2))
  with pytest.raises(StorageError, match="duplicate row identity"):
    merge_shard_stores([left.root, right.root], tmp_path / "merged")
  assert not (tmp_path / "merged").exists()
  assert list(tmp_path.glob(".*merge-*")) == []


def test_merge_refuses_a_duplicate_identity_inside_one_store(tmp_path) -> None:
  source = _store(tmp_path / "source", _rows("source", 0, 2))
  shard = source.root / "shard-000000.npz"
  copy = source.root / "shard-000001.npz"
  copy.write_bytes(shard.read_bytes())
  _rewrite_manifest(
    source.root,
    lambda manifest: manifest["shards"].append(
      {**manifest["shards"][0], "file": "shard-000001.npz"}
    ),
  )
  with pytest.raises(StorageError, match="duplicate row identity"):
    merge_shard_stores([source.root], tmp_path / "merged")


def test_merge_refuses_a_forged_row_count(tmp_path) -> None:
  source = _store(tmp_path / "source", _rows("source", 0, 2))
  shard = source.root / "shard-000000.npz"
  _rewrite_shard(
    shard, lambda payload: payload.update(metadata=payload["metadata"][:1])
  )
  with pytest.raises(StorageError, match="invalid header"):
    merge_shard_stores([source.root], tmp_path / "merged")
  assert not (tmp_path / "merged").exists()

  truncated = _store(tmp_path / "truncated", _rows("truncated", 0, 2))
  shard = truncated.root / "shard-000000.npz"
  _rewrite_shard(
    shard, lambda payload: payload.update(clean_action=payload["clean_action"][:1])
  )
  with pytest.raises(StorageError, match="invalid header"):
    merge_shard_stores([truncated.root], tmp_path / "truncated-merged")


def test_merge_refuses_forged_boundary_identities(tmp_path) -> None:
  source = _store(tmp_path / "source", _rows("source", 0, 2))
  _rewrite_manifest(
    source.root,
    lambda manifest: manifest["shards"][0].update(last=["forged", 0, "e", "s", 9]),
  )
  with pytest.raises(StorageError, match="boundary identities differ"):
    merge_shard_stores([source.root], tmp_path / "merged")


def test_merge_refuses_malformed_row_metadata(tmp_path) -> None:
  source = _store(tmp_path / "source", _rows("source", 0, 2))
  shard = source.root / "shard-000000.npz"

  def drop_fields(payload):
    payload["metadata"][0] = '{"run_id": "source"}'

  _rewrite_shard(shard, drop_fields)
  with pytest.raises(StorageError, match="malformed row metadata"):
    merge_shard_stores([source.root], tmp_path / "merged")


def test_merge_refuses_unsafe_or_repeated_shard_file_names(tmp_path) -> None:
  source = _store(tmp_path / "source", _rows("source", 0, 2))
  _rewrite_manifest(
    source.root, lambda manifest: manifest["shards"][0].update(file="../escape.npz")
  )
  with pytest.raises(StorageError, match="malformed shard manifest entry"):
    merge_shard_stores([source.root], tmp_path / "unsafe")

  repeated = _store(tmp_path / "repeated", _rows("repeated", 0, 2))
  _rewrite_manifest(
    repeated.root,
    lambda manifest: manifest["shards"].append(dict(manifest["shards"][0])),
  )
  with pytest.raises(StorageError, match="lists a shard file more than once"):
    merge_shard_stores([repeated.root], tmp_path / "repeated-merged")


def test_merge_refuses_another_contract_or_artifact_identity(tmp_path) -> None:
  source = _store(tmp_path / "source", _rows("source", 0, 2))
  _rewrite_manifest(
    source.root,
    lambda manifest: manifest["contract"].update(action_dimension=30),
  )
  with pytest.raises(StorageError, match="written under another contract"):
    merge_shard_stores([source.root], tmp_path / "other-contract")

  artifacts = _store(tmp_path / "artifacts", _rows("artifacts", 0, 2))
  _rewrite_manifest(
    artifacts.root,
    lambda manifest: manifest["artifact_identities"][0].update(sha256="0" * 64),
  )
  with pytest.raises(StorageError, match="different pinned artifacts"):
    merge_shard_stores([artifacts.root], tmp_path / "other-artifacts")


def test_merge_accepts_a_historical_digest_inclusive_contract_hash(tmp_path) -> None:
  """Older stores record the contract file digest inside ``contract_hash``.

  ``DiffusionContract.identity_hash`` no longer counts that descriptive digest,
  so a merge must compare the stored contract fields instead of that field.
  """
  source = _store(tmp_path / "source", _rows("source", 0, 2))
  _rewrite_manifest(
    source.root, lambda manifest: manifest.update(contract_hash="0" * 64)
  )
  report = merge_shard_stores([source.root], tmp_path / "merged")
  assert report.rows_after == 2
  recorded = json.loads((tmp_path / "merged" / "manifest.json").read_text())
  assert recorded["contract_hash"] == DEFAULT_CONTRACT.identity_hash()


def test_merge_refuses_a_source_shard_that_changed(tmp_path) -> None:
  source = _store(tmp_path / "source", _rows("source", 0, 2))
  before = _snapshot(source.root)
  shard = source.root / "shard-000000.npz"
  shard.write_bytes(shard.read_bytes() + b"corruption")
  with pytest.raises(StorageError, match="does not match its manifest hash"):
    merge_shard_stores([source.root], tmp_path / "merged")
  assert not (tmp_path / "merged").exists()
  assert _snapshot(source.root) != before  # the test itself mutated the source


def test_merge_leaves_the_source_stores_byte_identical(tmp_path) -> None:
  left = _store(tmp_path / "left", _rows("left", 0, 5))
  right = _store(tmp_path / "right", _rows("right", 0, 3))
  before = (_snapshot(left.root), _snapshot(right.root), manifest_sha256(left.root))

  merge_shard_stores([left.root, right.root], tmp_path / "merged")

  assert (_snapshot(left.root), _snapshot(right.root)) == before[:2]
  assert manifest_sha256(left.root) == before[2]


def test_merge_refuses_unreadable_latent_flags(tmp_path) -> None:
  source = _store(tmp_path / "source", _rows("source", 0, 2))
  _rewrite_shard(
    source.root / "shard-000000.npz",
    lambda payload: payload.update(latent_present=np.full(2, 3, dtype=np.uint8)),
  )
  with pytest.raises(StorageError, match="latent_present"):
    merge_shard_stores([source.root], tmp_path / "merged")
  assert not (tmp_path / "merged").exists()


def test_merge_publishes_atomically_and_leaves_no_staging(tmp_path) -> None:
  source = _store(tmp_path / "source", _rows("source", 0, 2))
  merge_shard_stores([source.root], tmp_path / "merged")
  assert (tmp_path / "merged" / "manifest.json").is_file()
  assert list(tmp_path.glob(".*merge-*")) == []

  # The second input fails while it is copied, after the first input's shards
  # were already staged: the staging directory must not survive the failure.
  good = _store(tmp_path / "good", _rows("good", 0, 2))
  broken = _store(tmp_path / "broken", _rows("broken", 0, 2))
  shard = broken.root / "shard-000000.npz"
  shard.write_bytes(shard.read_bytes() + b"corruption")
  with pytest.raises(StorageError, match="does not match its manifest hash"):
    merge_shard_stores([good.root, broken.root], tmp_path / "never-published")
  assert not (tmp_path / "never-published").exists()
  assert list(tmp_path.glob(".*never-published-merge-*")) == []


def test_merged_manifest_records_inputs_and_survives_an_append(tmp_path) -> None:
  left = _store(tmp_path / "left", _rows("left", 0, 2))
  right = _store(tmp_path / "right", _rows("right", 0, 2))
  merge_shard_stores([left.root, right.root], tmp_path / "merged")
  manifest_path = tmp_path / "merged" / "manifest.json"
  recorded = json.loads(manifest_path.read_text())["merged_from"]
  assert [item["root"] for item in recorded] == [str(left.root), str(right.root)]
  assert [item["rows"] for item in recorded] == [2, 2]

  store = AppendOnlyShardStore(tmp_path / "merged")
  store.append(_rows("later", 0, 2, env=7))
  assert store.row_count == 6
  assert json.loads(manifest_path.read_text())["merged_from"] == recorded
  assert [row.identity for row in store.iter_rows()][-2:] == [
    row.identity for row in _rows("later", 0, 2, env=7)
  ]


def test_merge_shard_bound_is_reported_and_refuses_an_impossible_bound(
  tmp_path,
) -> None:
  source = _store(tmp_path / "source", _rows("source", 0, 5), shard_rows=4)
  report = merge_shard_stores([source.root], tmp_path / "auto")
  assert report.max_rows_per_shard == 4
  assert (
    json.loads((tmp_path / "auto" / "manifest.json").read_text())["max_rows_per_shard"]
    == 4
  )

  with pytest.raises(StorageError, match="cannot honour max_rows_per_shard=2"):
    merge_shard_stores([source.root], tmp_path / "bounded", max_rows_per_shard=2)
  assert not (tmp_path / "bounded").exists()

  # A store declares a write bound (8) while its shards hold fewer rows (2): the
  # merged store keeps the declared collection convention.
  sparse = _store(tmp_path / "sparse", _rows("sparse", 0, 2), shard_rows=8)
  assert (
    merge_shard_stores([sparse.root], tmp_path / "declared").max_rows_per_shard == 8
  )


def test_merge_refuses_an_identity_digest_collision(tmp_path, monkeypatch) -> None:
  """A digest collision must refuse the merge, never accept a duplicate."""
  left = _store(tmp_path / "left", _rows("left", 0, 2))
  right = _store(tmp_path / "right", _rows("right", 0, 2))
  monkeypatch.setattr(storage_module, "identity_digest", lambda identity: b"\0" * 16)
  with pytest.raises(StorageError, match="identity digest collision"):
    merge_shard_stores([left.root, right.root], tmp_path / "merged")
  assert not (tmp_path / "merged").exists()


def test_merge_command_prints_the_historical_payload(tmp_path, monkeypatch, capsys):
  monkeypatch.setattr(diffusion, "_load_contract", lambda path: DEFAULT_CONTRACT)
  left = _store(tmp_path / "left", _rows("left", 0, 3))
  right = _store(tmp_path / "right", _rows("right", 0, 3))
  argv = [
    "--contract",
    "unused.yaml",
    "--input",
    str(left.root),
    "--input",
    str(right.root),
  ]

  assert diffusion_merge.main([*argv, "--output", str(tmp_path / "merged")]) == 0
  payload = json.loads(capsys.readouterr().out)
  assert payload["inputs"] == {str(left.root): 3, str(right.root): 3}
  assert payload["input_rows"] == 6
  assert payload["output"] == str(tmp_path / "merged")
  assert payload["rows_before"] == 0
  assert payload["rows_after"] == 6
  assert payload["shards"] == 4
  assert payload["mode"] == "shard-copy"
  assert payload["output_manifest_sha256"] == manifest_sha256(tmp_path / "merged")

  assert (
    diffusion_merge.main(
      [*argv, "--output", str(tmp_path / "legacy"), "--legacy-row-merge"]
    )
    == 0
  )
  legacy_payload = json.loads(capsys.readouterr().out)
  assert legacy_payload["mode"] == "row-merge"
  for key in ("inputs", "input_rows", "output", "rows_before", "rows_after", "shards"):
    assert key in legacy_payload


def test_merge_command_reports_a_refusal_as_one_line(tmp_path, monkeypatch, capsys):
  monkeypatch.setattr(diffusion, "_load_contract", lambda path: DEFAULT_CONTRACT)
  source = _store(tmp_path / "source", _rows("source", 0, 2))
  existing = tmp_path / "existing"
  existing.mkdir()
  with pytest.raises(SystemExit) as exit_info:
    diffusion_merge.main(
      [
        "--contract",
        "unused.yaml",
        "--input",
        str(source.root),
        "--output",
        str(existing),
      ]
    )
  assert "refuses an existing output" in str(exit_info.value)
  assert capsys.readouterr().out == ""
