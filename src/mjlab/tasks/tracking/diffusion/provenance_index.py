"""Columnar provenance index for one token-cache split.

The JSON sidecar beside a token cache is the authoritative provenance record,
but parsing it and rebuilding its objects costs minutes for a multi-million
window split.  This module writes a fixed-width binary companion beside that
sidecar and reads it back without a JSON parse.

Identity chain.  Build time pays the strong check once: it hashes the sidecar,
recomputes ``_provenance_digest(records)`` with the same code that wrote the
sidecar, and refuses to write when the two disagree.  Load time does not
re-derive the record digest.  It verifies that the binary matches
``arrays.digest``, that the sidecar on disk matches ``source_json``, that
``split``, ``count`` and ``cache_identity`` match the cache metadata, and that
``provenance_sha256`` was carried through from the sidecar unchanged.

That is a deliberate weakening of the existing gate.  The sidecar bytes stay
pinned in the header, so a swapped record is still detected, and the digest
remains reproducible from the sidecar plus
``python -m mjlab.scripts.diffusion build-provenance-index``.  A missing or
unusable index never fails a consumer: it falls back to the sidecar.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence, cast

import numpy as np

from .token_cache_lifecycle import _provenance_digest, _provenance_identity
from .window_dataset import (
  DatasetError,
  WindowCacheError,
  WindowProvenance,
  _sha256_file,
)

SCHEMA = "mjlab-x2-provenance-index-v1"

# Column order is part of the artefact: offsets are cumulative in this order.
_COLUMNS: tuple[tuple[str, str], ...] = (
  ("group_key", "<i4"),
  ("motion_id", "<i4"),
  ("phase", "|i1"),
  ("ou_noise_norm", "<f4"),
  ("pair_id", "<i4"),
  ("start_tick", "<i8"),
)
_DICTIONARY_TABLES = ("group_key", "motion_id", "phase", "pair_id")
# The header is a few kilobytes, and it must stay readable even when a caller
# replaces the module-level ``json.loads``; one decoder instance is reused.
_HEADER_DECODER = json.JSONDecoder()


class ProvenanceIndexError(WindowCacheError):
  """A provenance index is absent, stale, tampered or malformed.

  ``reason`` names the condition so a consumer can stay quiet for a missing
  optimisation and report a changed one.
  """

  def __init__(self, reason: str, detail: str) -> None:
    super().__init__(f"{reason}: {detail}")
    self.reason = reason


@dataclass(frozen=True, slots=True)
class ProvenanceIndexBuild:
  """Result of one build; ``status`` is ``built`` or ``reused``."""

  binary_path: Path
  header_path: Path
  status: str
  count: int
  binary_bytes: int


@dataclass(frozen=True, slots=True)
class ProvenanceIndex:
  """Columnar provenance for one token-cache split.

  Invariant: every column holds ``count`` rows, and ``dictionary[name][code]``
  is the string the sidecar recorded for that row.  Records are built on
  demand, so opening a split never materialises one object per window.
  """

  split: str
  dictionary: Mapping[str, tuple[str, ...]]
  group_key: np.ndarray
  motion_id: np.ndarray
  phase: np.ndarray
  ou_noise_norm: np.ndarray
  pair_id: np.ndarray
  start_tick: np.ndarray
  header: Mapping[str, object]

  @property
  def count(self) -> int:
    return int(self.start_tick.shape[0])

  def __len__(self) -> int:
    return self.count

  def __getitem__(self, index: int) -> WindowProvenance:
    position = index + self.count if index < 0 else index
    if position < 0 or position >= self.count:
      raise IndexError(index)
    tables = self.dictionary
    return WindowProvenance(
      split=self.split,
      group_key=tables["group_key"][int(self.group_key[position])],
      motion_id=tables["motion_id"][int(self.motion_id[position])],
      phase=tables["phase"][int(self.phase[position])],
      ou_noise_norm=float(self.ou_noise_norm[position]),
      pair_id=tables["pair_id"][int(self.pair_id[position])],
      start_tick=int(self.start_tick[position]),
    )

  def __iter__(self) -> Iterator[WindowProvenance]:
    for position in range(self.count):
      yield self[position]


def _sidecar_stem(sidecar: Path) -> str:
  suffix = ".json"
  name = sidecar.name
  return name[: -len(suffix)] if name.endswith(suffix) else name


def provenance_index_binary_path(sidecar_path: str | Path) -> Path:
  """Return ``<cache>.provenance.bin`` for one ``<cache>.provenance.json``."""
  sidecar = Path(sidecar_path)
  return sidecar.with_name(f"{_sidecar_stem(sidecar)}.bin")


def provenance_index_header_path(sidecar_path: str | Path) -> Path:
  """Return the header path that names one index binary."""
  sidecar = Path(sidecar_path)
  return sidecar.with_name(f"{_sidecar_stem(sidecar)}.bin.json")


def _int_field(metadata: Mapping[str, object], key: str) -> int:
  value = metadata.get(key)
  if isinstance(value, bool) or not isinstance(value, (int, float, str)):
    raise ProvenanceIndexError("malformed", f"cache metadata field {key!r} is invalid")
  try:
    return int(value)
  except (TypeError, ValueError) as exc:
    raise ProvenanceIndexError(
      "malformed", f"cache metadata field {key!r} is invalid"
    ) from exc


def _text_field(metadata: Mapping[str, object], key: str) -> str:
  value = metadata.get(key)
  if not isinstance(value, str) or not value:
    raise ProvenanceIndexError("malformed", f"cache metadata field {key!r} is invalid")
  return value


def _cache_identity(cache_metadata: Mapping[str, object]) -> dict[str, object]:
  """Return the sidecar identity dict that this cache metadata implies.

  This is the same dict the JSON path compares, rebuilt from fields the cache
  metadata already carries, so the index needs no dataset object to load.
  """
  raw_motion_ids = cache_metadata.get("motion_ids")
  motion_ids: tuple[str, ...] | None = None
  if raw_motion_ids is not None:
    if not isinstance(raw_motion_ids, list) or any(
      not isinstance(value, str) for value in raw_motion_ids
    ):
      raise ProvenanceIndexError("malformed", "cache metadata motion_ids is invalid")
    motion_ids = tuple(cast("list[str]", raw_motion_ids))
  raw_limit = cache_metadata.get("limit")
  return _provenance_identity(
    source_hash=_text_field(cache_metadata, "source_dataset_hash"),
    split=_text_field(cache_metadata, "split"),
    contract_hash=_text_field(cache_metadata, "contract_hash"),
    projection_hash=_text_field(cache_metadata, "projection_hash"),
    window_count=_int_field(cache_metadata, "window_count"),
    motion_ids=motion_ids,
    limit=None if raw_limit is None else _int_field(cache_metadata, "limit"),
    cache_content_sha256=_text_field(cache_metadata, "content_sha256"),
  )


def _read_sidecar_payload(sidecar: Path) -> Mapping[str, object]:
  """Parse the sidecar.  Only a build pays this cost."""
  try:
    text = sidecar.read_text(encoding="utf-8")
  except OSError as exc:
    raise ProvenanceIndexError(
      "stale", f"source sidecar is unreadable: {sidecar}"
    ) from exc
  try:
    payload = json.loads(text)
  except (TypeError, ValueError) as exc:
    raise ProvenanceIndexError(
      "malformed", f"source sidecar is not JSON: {sidecar}"
    ) from exc
  if not isinstance(payload, dict):
    raise ProvenanceIndexError(
      "malformed", f"source sidecar is not an object: {sidecar}"
    )
  return cast("Mapping[str, object]", payload)


def _records_from_payload(
  payload: Mapping[str, object], split: str, count: int
) -> list[WindowProvenance]:
  raw_records = payload.get("records")
  if not isinstance(raw_records, list) or len(raw_records) != count:
    raise ProvenanceIndexError(
      "stale", f"sidecar records do not match the cache window count {count}"
    )
  records: list[WindowProvenance] = []
  for raw in raw_records:
    if not isinstance(raw, dict):
      raise ProvenanceIndexError("malformed", "sidecar record is not an object")
    record = cast("Mapping[str, object]", raw)
    fields = (record.get("group_key"), record.get("motion_id"), record.get("phase"))
    pair_id = record.get("pair_id")
    if any(not isinstance(value, str) for value in (*fields, pair_id)):
      raise ProvenanceIndexError("malformed", "sidecar record fields are malformed")
    try:
      records.append(
        WindowProvenance(
          split=split,
          group_key=cast(str, fields[0]),
          motion_id=cast(str, fields[1]),
          phase=cast(str, fields[2]),
          ou_noise_norm=float(cast(float, record.get("ou_noise_norm"))),
          pair_id=cast(str, pair_id),
          start_tick=int(cast(int, record.get("start_tick"))),
        )
      )
    except (TypeError, ValueError, DatasetError) as exc:
      raise ProvenanceIndexError("malformed", "sidecar record is invalid") from exc
  return records


def _build_columns(
  records: Sequence[WindowProvenance], count: int
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]]]:
  """Densify records into one dictionary code per row and table per field."""
  tables = {
    name: tuple(sorted({cast(str, getattr(record, name)) for record in records}))
    for name in _DICTIONARY_TABLES
  }
  codes = {
    name: {value: position for position, value in enumerate(table)}
    for name, table in tables.items()
  }
  if len(tables["phase"]) > int(np.iinfo(np.int8).max):
    raise ProvenanceIndexError("malformed", "phase table does not fit int8 codes")
  columns = {
    "group_key": np.fromiter(
      (codes["group_key"][record.group_key] for record in records),
      dtype=np.int32,
      count=count,
    ),
    "motion_id": np.fromiter(
      (codes["motion_id"][record.motion_id] for record in records),
      dtype=np.int32,
      count=count,
    ),
    "phase": np.fromiter(
      (codes["phase"][record.phase] for record in records),
      dtype=np.int8,
      count=count,
    ),
    "ou_noise_norm": np.fromiter(
      (record.ou_noise_norm for record in records),
      dtype=np.float32,
      count=count,
    ),
    "pair_id": np.fromiter(
      (codes["pair_id"][record.pair_id] for record in records),
      dtype=np.int32,
      count=count,
    ),
    "start_tick": np.fromiter(
      (record.start_tick for record in records),
      dtype=np.int64,
      count=count,
    ),
  }
  return columns, tables


def _write_arrays(
  binary_path: Path, columns: Mapping[str, np.ndarray], count: int
) -> tuple[dict[str, dict[str, object]], str, int]:
  """Publish the binary atomically and return its layout, digest and size."""
  layout: dict[str, dict[str, object]] = {}
  digest = hashlib.sha256()
  written = 0
  temporary: Path | None = None
  try:
    with tempfile.NamedTemporaryFile(
      dir=binary_path.parent,
      prefix=f".{binary_path.name}.",
      suffix=".tmp",
      delete=False,
    ) as handle:
      temporary = Path(handle.name)
      for name, dtype in _COLUMNS:
        block = np.ascontiguousarray(columns[name], dtype=np.dtype(dtype)).tobytes()
        handle.write(block)
        digest.update(block)
        layout[name] = {
          "dtype": np.dtype(dtype).str,
          "offset": written,
          "shape": [int(count)],
        }
        written += len(block)
    os.replace(temporary, binary_path)
    temporary = None
  finally:
    if temporary is not None:
      temporary.unlink(missing_ok=True)
  return layout, digest.hexdigest(), written


def _write_header(header_path: Path, header: Mapping[str, object]) -> None:
  temporary: Path | None = None
  try:
    with tempfile.NamedTemporaryFile(
      dir=header_path.parent,
      prefix=f".{header_path.name}.",
      suffix=".tmp",
      mode="w",
      encoding="utf-8",
      delete=False,
    ) as handle:
      temporary = Path(handle.name)
      json.dump(header, handle, sort_keys=True, indent=2)
      handle.write("\n")
    os.replace(temporary, header_path)
    temporary = None
  finally:
    if temporary is not None:
      temporary.unlink(missing_ok=True)


def build_provenance_index(
  sidecar_path: str | Path,
  cache_metadata: Mapping[str, object],
  split: str,
  out_dir: str | Path,
) -> ProvenanceIndexBuild:
  """Write the columnar index for one sidecar, or reuse a valid one.

  The write is refused unless the sidecar's records reproduce the digest the
  sidecar records, so an index is only ever published from provenance that
  already passed the strong check.
  """
  sidecar = Path(sidecar_path)
  if not sidecar.is_file():
    raise ProvenanceIndexError("absent", f"provenance sidecar is missing: {sidecar}")
  cache_split = _text_field(cache_metadata, "split")
  if split != cache_split:
    raise ProvenanceIndexError(
      "malformed", f"split {split!r} does not match cache split {cache_split!r}"
    )
  count = _int_field(cache_metadata, "window_count")
  target_dir = Path(out_dir)
  target_dir.mkdir(parents=True, exist_ok=True)
  binary_path = target_dir / provenance_index_binary_path(sidecar).name
  header_path = target_dir / provenance_index_header_path(sidecar).name
  if header_path.is_file() and binary_path.is_file():
    try:
      load_provenance_index(sidecar, cache_metadata)
    except ProvenanceIndexError:
      pass
    else:
      return ProvenanceIndexBuild(
        binary_path, header_path, "reused", count, int(binary_path.stat().st_size)
      )

  source_json = {
    "sha256": _sha256_file(sidecar),
    "size": int(sidecar.stat().st_size),
  }
  payload = _read_sidecar_payload(sidecar)
  identity = _cache_identity(cache_metadata)
  if {key: payload.get(key) for key in identity} != identity:
    raise ProvenanceIndexError(
      "stale", "sidecar identity does not match the token cache metadata"
    )
  records = _records_from_payload(payload, split, count)
  provenance_sha256 = _provenance_digest(records)
  recorded = payload.get("provenance_sha256")
  if recorded != provenance_sha256:
    raise ProvenanceIndexError(
      "tampered", "sidecar records do not match the digest the sidecar records"
    )
  columns, tables = _build_columns(records, count)
  layout, digest, written = _write_arrays(binary_path, columns, count)
  header = {
    "schema": SCHEMA,
    "split": split,
    "count": count,
    "source_json": source_json,
    "arrays": {"digest": digest, "bytes": written},
    "columns": layout,
    "dictionary": {name: list(table) for name, table in tables.items()},
    "cache_identity": identity,
    "provenance_sha256": provenance_sha256,
  }
  # The header is the commit point: a binary without its header is ignored.
  _write_header(header_path, header)
  return ProvenanceIndexBuild(binary_path, header_path, "built", count, written)


def _read_header(header_path: Path) -> Mapping[str, object]:
  try:
    text = header_path.read_text(encoding="utf-8")
  except FileNotFoundError as exc:
    raise ProvenanceIndexError(
      "absent", f"provenance index header is missing: {header_path}"
    ) from exc
  except OSError as exc:
    raise ProvenanceIndexError(
      "malformed", f"provenance index header is unreadable: {header_path}"
    ) from exc
  try:
    payload = _HEADER_DECODER.decode(text)
  except (TypeError, ValueError) as exc:
    raise ProvenanceIndexError(
      "malformed", f"provenance index header is not JSON: {header_path}"
    ) from exc
  if not isinstance(payload, dict):
    raise ProvenanceIndexError(
      "malformed", f"provenance index header is not an object: {header_path}"
    )
  return cast("Mapping[str, object]", payload)


def _header_mapping(header: Mapping[str, object], key: str) -> Mapping[str, object]:
  value = header.get(key)
  if not isinstance(value, dict):
    raise ProvenanceIndexError(
      "malformed", f"provenance index header field {key!r} is not an object"
    )
  return cast("Mapping[str, object]", value)


def _header_digest(header: Mapping[str, object], key: str) -> str:
  value = header.get(key)
  if not isinstance(value, str) or len(value) != 64:
    raise ProvenanceIndexError(
      "malformed", f"provenance index header field {key!r} is not a sha256 digest"
    )
  return value


def _read_dictionary(
  header: Mapping[str, object], count: int
) -> dict[str, tuple[str, ...]]:
  raw = _header_mapping(header, "dictionary")
  dictionary: dict[str, tuple[str, ...]] = {}
  for name in _DICTIONARY_TABLES:
    table = raw.get(name)
    if not isinstance(table, list) or any(
      not isinstance(value, str) for value in table
    ):
      raise ProvenanceIndexError(
        "malformed", f"provenance index dictionary table {name!r} is invalid"
      )
    # An empty split legitimately has no strings; a populated one has no excuse.
    if not table and count:
      raise ProvenanceIndexError(
        "malformed", f"provenance index dictionary table {name!r} is empty"
      )
    dictionary[name] = tuple(cast("list[str]", table))
  if set(raw) != set(_DICTIONARY_TABLES):
    raise ProvenanceIndexError("malformed", "provenance index dictionary is incomplete")
  return dictionary


def _load_columns(
  header: Mapping[str, object], buffer: np.ndarray, count: int
) -> dict[str, np.ndarray]:
  layout = _header_mapping(header, "columns")
  columns: dict[str, np.ndarray] = {}
  for name, dtype in _COLUMNS:
    entry = layout.get(name)
    if not isinstance(entry, dict):
      raise ProvenanceIndexError(
        "malformed", f"provenance index column {name!r} is missing"
      )
    spec = cast("Mapping[str, object]", entry)
    recorded_dtype = spec.get("dtype")
    offset = spec.get("offset")
    shape = spec.get("shape")
    expected = np.dtype(dtype)
    if recorded_dtype != expected.str or shape != [count]:
      raise ProvenanceIndexError(
        "malformed", f"provenance index column {name!r} has an unexpected layout"
      )
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
      raise ProvenanceIndexError(
        "malformed", f"provenance index column {name!r} has an invalid offset"
      )
    if offset + expected.itemsize * count > buffer.nbytes:
      raise ProvenanceIndexError(
        "tampered", f"provenance index column {name!r} exceeds the binary"
      )
    columns[name] = np.ndarray(
      shape=(count,), dtype=expected, buffer=buffer, offset=offset
    )
  return columns


def _check_codes(
  name: str, column: np.ndarray, dictionary: Mapping[str, tuple[str, ...]]
) -> None:
  table_size = len(dictionary[name])
  if column.size and (int(column.min()) < 0 or int(column.max()) >= table_size):
    raise ProvenanceIndexError(
      "tampered", f"provenance index column {name!r} is out of dictionary range"
    )


def load_provenance_index(
  sidecar_path: str | Path,
  cache_metadata: Mapping[str, object],
) -> ProvenanceIndex:
  """Open and validate the index that belongs to one sidecar.

  Raises :class:`ProvenanceIndexError` with a named ``reason`` instead of
  returning unchecked rows, so the caller can choose between a silent fallback
  (``absent``) and a reported one (``stale``, ``tampered``, ``malformed``).
  """
  sidecar = Path(sidecar_path)
  binary_path = provenance_index_binary_path(sidecar)
  header = _read_header(provenance_index_header_path(sidecar))
  if header.get("schema") != SCHEMA:
    raise ProvenanceIndexError(
      "malformed", f"unsupported provenance index schema {header.get('schema')!r}"
    )
  if not binary_path.is_file():
    raise ProvenanceIndexError(
      "tampered", f"provenance index binary is missing: {binary_path}"
    )
  split = _text_field(cache_metadata, "split")
  if header.get("split") != split:
    raise ProvenanceIndexError(
      "stale", f"index split {header.get('split')!r} is not cache split {split!r}"
    )
  count = _int_field(cache_metadata, "window_count")
  if header.get("count") != count:
    raise ProvenanceIndexError(
      "stale", f"index count {header.get('count')!r} is not cache count {count}"
    )
  if header.get("cache_identity") != _cache_identity(cache_metadata):
    raise ProvenanceIndexError(
      "stale", "index cache identity does not match the cache metadata"
    )

  source_json = _header_mapping(header, "source_json")
  size = source_json.get("size")
  if isinstance(size, bool) or not isinstance(size, int):
    raise ProvenanceIndexError("malformed", "index source_json size is invalid")
  if not sidecar.is_file() or sidecar.stat().st_size != size:
    raise ProvenanceIndexError(
      "stale", f"source sidecar is missing or resized: {sidecar}"
    )
  if _sha256_file(sidecar) != _header_digest(source_json, "sha256"):
    raise ProvenanceIndexError(
      "stale", f"source sidecar digest changed since the index was built: {sidecar}"
    )

  arrays = _header_mapping(header, "arrays")
  recorded_bytes = arrays.get("bytes")
  if isinstance(recorded_bytes, bool) or not isinstance(recorded_bytes, int):
    raise ProvenanceIndexError("malformed", "index arrays bytes is invalid")
  if binary_path.stat().st_size != recorded_bytes:
    raise ProvenanceIndexError(
      "tampered", f"provenance index binary size changed: {binary_path}"
    )
  if _sha256_file(binary_path) != _header_digest(arrays, "digest"):
    raise ProvenanceIndexError(
      "tampered", f"provenance index binary digest changed: {binary_path}"
    )

  # The record digest is not re-derived here; the sidecar that records it is
  # pinned above.  A cache that copies the value into its metadata is compared.
  provenance_sha256 = _header_digest(header, "provenance_sha256")
  recorded_digest = cache_metadata.get("provenance_sha256")
  if recorded_digest is not None and recorded_digest != provenance_sha256:
    raise ProvenanceIndexError(
      "stale", "index provenance digest does not match the cache metadata"
    )

  dictionary = _read_dictionary(header, count)
  # An empty split writes an empty binary, which cannot be memory-mapped.
  buffer = (
    np.memmap(binary_path, dtype=np.uint8, mode="r")
    if count
    else np.empty(0, dtype=np.uint8)
  )
  columns = _load_columns(header, buffer, count)
  for name in _DICTIONARY_TABLES:
    _check_codes(name, columns[name], dictionary)
  return ProvenanceIndex(
    split=split,
    dictionary=dictionary,
    group_key=columns["group_key"],
    motion_id=columns["motion_id"],
    phase=columns["phase"],
    ou_noise_norm=columns["ou_noise_norm"],
    pair_id=columns["pair_id"],
    start_tick=columns["start_tick"],
    header=header,
  )


__all__ = [
  "ProvenanceIndex",
  "ProvenanceIndexBuild",
  "ProvenanceIndexError",
  "SCHEMA",
  "build_provenance_index",
  "load_provenance_index",
  "provenance_index_binary_path",
  "provenance_index_header_path",
]
