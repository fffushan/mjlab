"""Gates for the columnar provenance index beside a token-cache sidecar."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mjlab.tasks.tracking.diffusion.provenance_index import (
  SCHEMA,
  ProvenanceIndex,
  ProvenanceIndexError,
  build_provenance_index,
  load_provenance_index,
  provenance_index_binary_path,
  provenance_index_header_path,
)
from mjlab.tasks.tracking.diffusion.token_cache_lifecycle import (
  _canonical_provenance,
  _provenance_digest,
  _provenance_identity,
  _validate_provenance,
  provenance_cache_path,
)
from mjlab.tasks.tracking.diffusion.window_dataset import (
  TokenWindowDataset,
  ValidatedTokenCache,
  WindowCacheError,
  WindowProvenance,
  _cache_metadata,
  sampling_weights,
)

PHASES = ("", "clean", "ou")


def _records(count: int = 32) -> list[WindowProvenance]:
  """Synthetic provenance whose ``ou_noise_norm`` is exact in float32."""
  return [
    WindowProvenance(
      split="train",
      group_key=f"group-{index % 2}",
      motion_id=f"tennis_{index % 3:03d}",
      phase=PHASES[index % 3],
      ou_noise_norm=index * 0.125,
      pair_id="" if index % 3 != 2 else f"pair-{index}",
      start_tick=index * 7,
    )
    for index in range(count)
  ]


def _fixture(
  tmp_path: Path, *, count: int = 32
) -> tuple[Path, ValidatedTokenCache, list[WindowProvenance], dict[str, object]]:
  """Write one cache, its metadata and its sidecar; return the load inputs."""
  records = _records(count)
  offline = tmp_path / "offline"
  offline.mkdir(parents=True, exist_ok=True)
  cache_path = offline / "tokens-train-0123456789abcdef.npy"
  memmap = np.lib.format.open_memmap(
    cache_path, mode="w+", dtype=np.float32, shape=(count, 41, 231)
  )
  memmap.flush()
  del memmap
  metadata = _cache_metadata(
    source_hash="a" * 64,
    split="train",
    contract_hash="b" * 64,
    projection_hash="c" * 64,
    window_count=count,
    dtype=np.dtype(np.float32),
    motion_ids=None,
    limit=None,
    shape=(count, 41, 231),
    created_at=None,
    content_sha256=hashlib.sha256(cache_path.read_bytes()).hexdigest(),
  )
  cache_path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
  expected = _provenance_identity(
    source_hash=str(metadata["source_dataset_hash"]),
    split="train",
    contract_hash=str(metadata["contract_hash"]),
    projection_hash=str(metadata["projection_hash"]),
    window_count=count,
    motion_ids=None,
    limit=None,
    cache_content_sha256=str(metadata["content_sha256"]),
  )
  sidecar = provenance_cache_path(cache_path)
  sidecar.write_text(
    json.dumps(
      {
        **expected,
        "provenance_sha256": _provenance_digest(records),
        "records": _canonical_provenance(records),
      },
      sort_keys=True,
      indent=2,
    )
    + "\n",
    encoding="utf-8",
  )
  cache = ValidatedTokenCache(cache_path, metadata, np.load(cache_path, mmap_mode="r"))
  return sidecar, cache, records, expected


def test_build_and_load_round_trip(tmp_path: Path) -> None:
  sidecar, cache, records, expected = _fixture(tmp_path)
  build = build_provenance_index(sidecar, cache.metadata, "train", sidecar.parent)
  assert build.status == "built"
  assert build.count == len(records)
  assert build.binary_path == provenance_index_binary_path(sidecar)
  assert build.header_path == provenance_index_header_path(sidecar)
  assert build.binary_path.name.endswith(".provenance.bin")
  assert build.header_path.name.endswith(".provenance.bin.json")

  header = json.loads(build.header_path.read_text(encoding="utf-8"))
  assert set(header) == {
    "schema",
    "split",
    "count",
    "source_json",
    "arrays",
    "columns",
    "dictionary",
    "cache_identity",
    "provenance_sha256",
  }
  assert header["schema"] == SCHEMA
  assert header["split"] == "train"
  assert header["count"] == len(records)
  assert header["source_json"] == {
    "sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
    "size": sidecar.stat().st_size,
  }
  assert (
    header["arrays"]["digest"]
    == hashlib.sha256(build.binary_path.read_bytes()).hexdigest()
  )
  assert header["cache_identity"] == expected
  assert header["provenance_sha256"] == _provenance_digest(records)
  assert header["dictionary"]["phase"] == list(PHASES)

  # Offsets and dtypes are explicit, cumulative in the declared column order.
  offset = 0
  for name, dtype in (
    ("group_key", np.int32),
    ("motion_id", np.int32),
    ("phase", np.int8),
    ("ou_noise_norm", np.float32),
    ("pair_id", np.int32),
    ("start_tick", np.int64),
  ):
    layout = header["columns"][name]
    assert layout["dtype"] == np.dtype(dtype).str
    assert layout["offset"] == offset
    assert layout["shape"] == [len(records)]
    offset += np.dtype(dtype).itemsize * len(records)
  assert header["arrays"]["bytes"] == offset == build.binary_bytes

  index = load_provenance_index(sidecar, cache.metadata)
  assert isinstance(index, ProvenanceIndex)
  assert index.count == len(records) == len(index)
  assert index.group_key.dtype == np.int32
  assert index.motion_id.dtype == np.int32
  assert index.phase.dtype == np.int8
  assert index.ou_noise_norm.dtype == np.float32
  assert index.pair_id.dtype == np.int32
  assert index.start_tick.dtype == np.int64
  assert [index[position] for position in range(len(records))] == records
  assert list(index) == records
  assert index[-1] == records[-1]
  with pytest.raises(IndexError):
    index[len(records)]


def test_sampling_weights_match_the_json_records(tmp_path: Path) -> None:
  sidecar, cache, records, _ = _fixture(tmp_path)
  build_provenance_index(sidecar, cache.metadata, "train", sidecar.parent)
  index = load_provenance_index(sidecar, cache.metadata)
  for clean_weight, perturbed_weight in ((1.0, 1.0), (1.0, 3.0), (0.5, 0.0)):
    expected = sampling_weights(
      records, clean_weight=clean_weight, perturbed_weight=perturbed_weight
    )
    actual = sampling_weights(
      index, clean_weight=clean_weight, perturbed_weight=perturbed_weight
    )
    assert actual.dtype == np.float64
    assert np.array_equal(actual, expected)


def test_token_window_dataset_accepts_the_columnar_index(tmp_path: Path) -> None:
  sidecar, cache, records, _ = _fixture(tmp_path)
  build_provenance_index(sidecar, cache.metadata, "train", sidecar.parent)
  index = load_provenance_index(sidecar, cache.metadata)
  dataset = TokenWindowDataset(
    cache.path, index, clean_weight=1.0, perturbed_weight=2.0
  )
  reference = TokenWindowDataset(
    cache.path, records, clean_weight=1.0, perturbed_weight=2.0
  )
  assert len(dataset) == len(records)
  assert np.array_equal(dataset.weights, reference.weights)
  _, weight, position = dataset[3]
  assert position == 3
  assert float(weight) == float(reference.weights[3])
  with pytest.raises(WindowCacheError, match="split"):
    TokenWindowDataset(cache.path, index, split="validation")


def test_flipped_byte_falls_back_with_the_named_reason(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  sidecar, cache, records, expected = _fixture(tmp_path)
  build_provenance_index(sidecar, cache.metadata, "train", sidecar.parent)
  binary = provenance_index_binary_path(sidecar)
  payload = bytearray(binary.read_bytes())
  payload[0] ^= 0xFF
  binary.write_bytes(bytes(payload))

  with pytest.raises(ProvenanceIndexError) as excinfo:
    load_provenance_index(sidecar, cache.metadata)
  assert excinfo.value.reason == "tampered"

  result = _validate_provenance(sidecar, cache=cache, expected=expected)
  assert not isinstance(result, ProvenanceIndex)
  assert list(result) == records
  lines = capsys.readouterr().err.splitlines()
  assert len(lines) == 1
  assert "tampered" in lines[0]


def test_stale_source_digest_falls_back_with_the_named_reason(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  sidecar, cache, records, expected = _fixture(tmp_path)
  build_provenance_index(sidecar, cache.metadata, "train", sidecar.parent)
  # Same records and digest, different bytes: the index no longer describes the
  # sidecar that is on disk, while the JSON path itself still validates.
  payload = json.loads(sidecar.read_text(encoding="utf-8"))
  sidecar.write_text(json.dumps(payload, sort_keys=True, indent=4), encoding="utf-8")

  with pytest.raises(ProvenanceIndexError) as excinfo:
    load_provenance_index(sidecar, cache.metadata)
  assert excinfo.value.reason == "stale"

  result = _validate_provenance(sidecar, cache=cache, expected=expected)
  assert not isinstance(result, ProvenanceIndex)
  assert list(result) == records
  lines = capsys.readouterr().err.splitlines()
  assert len(lines) == 1
  assert "stale" in lines[0]


def test_absent_index_uses_the_json_path_silently(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  sidecar, cache, records, expected = _fixture(tmp_path)
  assert not provenance_index_binary_path(sidecar).exists()
  result = _validate_provenance(sidecar, cache=cache, expected=expected)
  assert not isinstance(result, ProvenanceIndex)
  assert list(result) == records
  assert capsys.readouterr().err == ""


def test_environment_switch_forces_the_json_path(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  sidecar, cache, records, expected = _fixture(tmp_path)
  build_provenance_index(sidecar, cache.metadata, "train", sidecar.parent)
  # A byte flip would be reported if the index were consulted at all.
  binary = provenance_index_binary_path(sidecar)
  payload = bytearray(binary.read_bytes())
  payload[0] ^= 0xFF
  binary.write_bytes(bytes(payload))
  monkeypatch.setenv("MJLAB_PROVENANCE_INDEX", "off")
  result = _validate_provenance(sidecar, cache=cache, expected=expected)
  assert not isinstance(result, ProvenanceIndex)
  assert list(result) == records
  assert capsys.readouterr().err == ""


def test_valid_index_never_parses_the_json_sidecar(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  sidecar, cache, records, expected = _fixture(tmp_path)
  build_provenance_index(sidecar, cache.metadata, "train", sidecar.parent)

  def refuse(*_args: object, **_kwargs: object) -> object:
    raise AssertionError("a valid provenance index parsed JSON")

  monkeypatch.setattr(json, "loads", refuse)
  result = _validate_provenance(sidecar, cache=cache, expected=expected)
  assert isinstance(result, ProvenanceIndex)
  assert result.count == len(records)
  assert result[1] == records[1]


def test_second_build_reuses_and_a_tampered_index_is_rebuilt(tmp_path: Path) -> None:
  sidecar, cache, _, _ = _fixture(tmp_path)
  first = build_provenance_index(sidecar, cache.metadata, "train", sidecar.parent)
  binary = provenance_index_binary_path(sidecar)
  header = provenance_index_header_path(sidecar)
  before = (binary.stat().st_mtime_ns, header.stat().st_mtime_ns)

  second = build_provenance_index(sidecar, cache.metadata, "train", sidecar.parent)
  assert second.status == "reused"
  assert second.binary_path == first.binary_path
  assert (binary.stat().st_mtime_ns, header.stat().st_mtime_ns) == before

  payload = bytearray(binary.read_bytes())
  payload[0] ^= 0xFF
  binary.write_bytes(bytes(payload))
  repaired = build_provenance_index(sidecar, cache.metadata, "train", sidecar.parent)
  assert repaired.status == "built"
  assert load_provenance_index(sidecar, cache.metadata).count == first.count


def test_empty_split_round_trips_without_a_memmap(tmp_path: Path) -> None:
  """A validation split may hold no windows; its index must still load."""
  sidecar, cache, records, expected = _fixture(tmp_path, count=0)
  build = build_provenance_index(sidecar, cache.metadata, "train", sidecar.parent)
  assert build.status == "built"
  assert build.binary_bytes == 0
  index = load_provenance_index(sidecar, cache.metadata)
  assert index.count == 0 == len(index)
  assert index.group_key.shape == index.start_tick.shape == (0,)
  assert list(index) == records == []
  result = _validate_provenance(sidecar, cache=cache, expected=expected)
  assert isinstance(result, ProvenanceIndex)
  assert len(result) == 0
