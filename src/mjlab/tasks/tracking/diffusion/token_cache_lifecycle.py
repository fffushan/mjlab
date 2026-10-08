"""Safe preparation and read-only reuse of dataset-owned token caches.

The token array remains the v2 cache format.  This module adds one aligned JSON
sidecar for dataset-derived provenance.  The sidecar does not contain weights or
training settings, so changing an arm configuration cannot change its identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .window_dataset import (
  _CACHE_VERSION,
  CacheLock,
  DatasetError,
  DatasetSource,
  LoadedDataset,
  TokenCacheBusyError,
  ValidatedTokenCache,
  WindowCacheError,
  WindowProvenance,
  _build_token_cache_unlocked,
  _cache_metadata,
  _cache_source_and_loaded,
  _identity_metadata,
  _normalise_motion_ids,
  _projection_identity,
  _read_cache_metadata,
  _source_dataset_hash,
  _validate_cache,
  _validate_split,
  dataset_token_cache_path,
  iter_index_provenance,
)

_PROVENANCE_VERSION = "mjlab-x2-diffusion-token-cache-provenance-v1"


class CacheProvenanceError(WindowCacheError):
  """A provenance sidecar is missing, stale or malformed."""


@dataclass(frozen=True, slots=True)
class PreparedTokenCache:
  """A validated token cache and its aligned, token-free provenance."""

  cache: ValidatedTokenCache
  provenance: tuple[WindowProvenance, ...]
  cache_status: str
  provenance_status: str

  @property
  def path(self) -> Path:
    return self.cache.path

  @property
  def count(self) -> int:
    return len(self.provenance)


def provenance_cache_path(path: str | Path) -> Path:
  """Return the sidecar path associated with one token-cache ``.npy`` file."""
  cache = Path(path)
  return cache.with_name(f"{cache.stem}.provenance.json")


def _canonical_provenance(value: Sequence[WindowProvenance]) -> list[dict[str, object]]:
  return [
    {
      "group_key": item.group_key,
      "motion_id": item.motion_id,
      "phase": item.phase,
      "pair_id": item.pair_id,
      "start_tick": item.start_tick,
      "ou_noise_norm": item.ou_noise_norm,
    }
    for item in value
  ]


def _provenance_digest(records: Sequence[WindowProvenance]) -> str:
  encoded = json.dumps(
    _canonical_provenance(records), sort_keys=True, separators=(",", ":")
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def _provenance_identity(
  *,
  source_hash: str,
  split: str,
  contract_hash: str,
  projection_hash: str,
  window_count: int,
  motion_ids: tuple[str, ...] | None,
  limit: int | None,
  cache_content_sha256: str,
) -> dict[str, object]:
  return {
    "format": _PROVENANCE_VERSION,
    "cache_format": _CACHE_VERSION,
    "source_dataset_hash": source_hash,
    "split": split,
    "contract_hash": contract_hash,
    "projection_hash": projection_hash,
    "window_count": window_count,
    "motion_ids": None if motion_ids is None else list(motion_ids),
    "limit": limit,
    "cache_content_sha256": cache_content_sha256,
  }


def _expected_cache(
  source: DatasetSource | LoadedDataset,
  split: str,
  path: str | Path,
  *,
  motion_ids: Sequence[str] | None,
  dtype: Any,
  loaded: LoadedDataset | None,
  source_hash: str | None,
  count: int | None = None,
) -> tuple[DatasetSource, LoadedDataset, dict[str, object], tuple[str, ...] | None]:
  _validate_split(split)
  source_obj, loaded_obj = _cache_source_and_loaded(source, loaded)
  selected = _normalise_motion_ids(motion_ids)
  count = (
    sum(
      1
      for _ in iter_index_provenance(loaded_obj, split, motion_ids=selected, limit=None)
    )
    if count is None
    else int(count)
  )
  shape = (
    count,
    loaded_obj.contract.window_steps,
    loaded_obj.contract.token_dimension,
  )
  expected = _cache_metadata(
    source_hash=(
      _source_dataset_hash(source_obj, loaded_obj)
      if source_hash is None
      else source_hash
    ),
    split=split,
    contract_hash=loaded_obj.contract.identity_hash(),
    projection_hash=_projection_identity(loaded_obj.projection),
    window_count=count,
    dtype=np.dtype(dtype),
    motion_ids=selected,
    limit=None,
    shape=shape,
    created_at=None,
    content_sha256=None,
  )
  return source_obj, loaded_obj, expected, selected


def _read_validated_cache(
  source: DatasetSource | LoadedDataset,
  split: str,
  path: str | Path,
  *,
  motion_ids: Sequence[str] | None,
  dtype: Any,
  loaded: LoadedDataset | None,
  source_hash: str | None,
  count: int | None = None,
) -> ValidatedTokenCache:
  cache_path = Path(path)
  metadata_path = cache_path.with_suffix(".json")
  source_obj, loaded_obj, expected, _ = _expected_cache(
    source,
    split,
    path,
    motion_ids=motion_ids,
    dtype=dtype,
    loaded=loaded,
    source_hash=source_hash,
    count=count,
  )
  metadata, cache = _validate_cache(cache_path, metadata_path, expected)
  identity = {
    "directory": str(source_obj.directory),
    "assignments_hash": loaded_obj.assignments.sha256(),
    "split_coverage": loaded_obj.index.coverage(),
    "split": split,
    "motion_ids": metadata.get("motion_ids"),
    "limit": metadata.get("limit"),
    "source_dataset_hash": metadata["source_dataset_hash"],
    "contract_hash": metadata["contract_hash"],
    "projection_hash": metadata["projection_hash"],
    "projection_hashes": {
      "matrix": loaded_obj.projection.matrix_sha256,
      "pseudoinverse": loaded_obj.projection.pseudoinverse_sha256,
      "statistics": loaded_obj.projection.statistics_sha256,
    },
  }
  return ValidatedTokenCache(cache_path, metadata, cache, identity)


def _read_validated_published_cache(
  source: DatasetSource | LoadedDataset,
  split: str,
  path: str | Path,
  *,
  motion_ids: Sequence[str] | None,
  dtype: Any,
  loaded: LoadedDataset | None,
  source_hash: str | None,
  count: int,
) -> ValidatedTokenCache:
  """Read a cache just published by ``build_token_cache`` without rehashing it."""
  cache_path = Path(path)
  metadata = _read_cache_metadata(cache_path.with_suffix(".json"))
  array = np.load(cache_path, mmap_mode="r", allow_pickle=False)
  if not isinstance(array, np.memmap):
    raise WindowCacheError("token cache must be a NumPy .npy memmap")
  # The builder already validated or hashed this exact pair while holding its
  # lock.  Validate the source identity and shape here, but do not scan bytes a
  # second time during the same startup.
  _, loaded_obj, expected, _ = _expected_cache(
    source,
    split,
    path,
    motion_ids=motion_ids,
    dtype=dtype,
    loaded=loaded,
    source_hash=source_hash,
    count=count,
  )
  if _identity_metadata(metadata) != _identity_metadata(expected):
    raise WindowCacheError("published token cache metadata does not match request")
  expected_shape = tuple(int(value) for value in expected["shape"])  # type: ignore[arg-type]
  if array.shape != expected_shape or np.dtype(array.dtype).str != str(
    expected["dtype"]
  ):
    raise WindowCacheError("published token cache shape or dtype mismatch")
  source_obj, _ = _cache_source_and_loaded(source, loaded)
  identity = {
    "directory": str(source_obj.directory),
    "assignments_hash": loaded_obj.assignments.sha256(),
    "split_coverage": loaded_obj.index.coverage(),
    "split": split,
    "motion_ids": metadata.get("motion_ids"),
    "limit": metadata.get("limit"),
    "source_dataset_hash": metadata["source_dataset_hash"],
    "contract_hash": metadata["contract_hash"],
    "projection_hash": metadata["projection_hash"],
    "projection_hashes": {
      "matrix": loaded_obj.projection.matrix_sha256,
      "pseudoinverse": loaded_obj.projection.pseudoinverse_sha256,
      "statistics": loaded_obj.projection.statistics_sha256,
    },
  }
  return ValidatedTokenCache(cache_path, metadata, array, identity)


def _metadata_int(metadata: Mapping[str, object], key: str) -> int:
  value = metadata.get(key)
  if isinstance(value, bool) or not isinstance(value, (int, float, str)):
    raise CacheProvenanceError(f"provenance metadata field {key!r} is invalid")
  try:
    return int(value)
  except (TypeError, ValueError) as exc:
    raise CacheProvenanceError(f"provenance metadata field {key!r} is invalid") from exc


def _validate_provenance(
  path: Path,
  *,
  cache: ValidatedTokenCache,
  expected: Mapping[str, object],
) -> tuple[WindowProvenance, ...]:
  if not path.is_file():
    raise CacheProvenanceError(f"provenance sidecar is missing: {path}")
  try:
    payload = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, TypeError, json.JSONDecodeError) as exc:
    raise CacheProvenanceError(f"invalid provenance sidecar: {path}") from exc
  if not isinstance(payload, dict):
    raise CacheProvenanceError("provenance sidecar must be an object")
  identity = {key: payload.get(key) for key in expected}
  if identity != dict(expected):
    raise CacheProvenanceError("provenance sidecar identity does not match cache")
  raw_records = payload.get("records")
  if not isinstance(raw_records, list) or len(raw_records) != _metadata_int(
    expected, "window_count"
  ):
    raise CacheProvenanceError("provenance sidecar count does not match cache")
  records: list[WindowProvenance] = []
  try:
    for raw in raw_records:
      if not isinstance(raw, dict):
        raise CacheProvenanceError("provenance record must be an object")
      if set(raw) != {
        "group_key",
        "motion_id",
        "phase",
        "pair_id",
        "start_tick",
        "ou_noise_norm",
      } or any(
        not isinstance(raw[name], str)
        for name in ("group_key", "motion_id", "phase", "pair_id")
      ):
        raise CacheProvenanceError("provenance record fields are malformed")
      records.append(
        WindowProvenance(
          split=str(expected["split"]),
          group_key=raw["group_key"],
          motion_id=raw["motion_id"],
          phase=raw["phase"],
          ou_noise_norm=raw["ou_noise_norm"],
          pair_id=raw["pair_id"],
          start_tick=raw["start_tick"],
        )
      )
  except (DatasetError, TypeError, ValueError) as exc:
    if isinstance(exc, CacheProvenanceError):
      raise
    raise CacheProvenanceError("provenance record is invalid") from exc
  if payload.get("provenance_sha256") != _provenance_digest(records):
    raise CacheProvenanceError("provenance sidecar digest does not match records")
  if expected["cache_content_sha256"] != cache.metadata.get("content_sha256"):
    raise CacheProvenanceError("provenance cache digest does not match token cache")
  return tuple(records)


def _provenance_expected(
  cache: ValidatedTokenCache,
  loaded: LoadedDataset,
  split: str,
  motion_ids: Sequence[str] | None,
  limit: int | None = None,
) -> dict[str, object]:
  selected = _normalise_motion_ids(motion_ids)
  metadata = cache.metadata
  return _provenance_identity(
    source_hash=str(metadata["source_dataset_hash"]),
    split=split,
    contract_hash=loaded.contract.identity_hash(),
    projection_hash=_projection_identity(loaded.projection),
    window_count=_metadata_int(metadata, "window_count"),
    motion_ids=selected,
    limit=limit,
    cache_content_sha256=str(metadata["content_sha256"]),
  )


def validate_provenance_cache(
  cache: ValidatedTokenCache,
  loaded: LoadedDataset,
  split: str,
  *,
  motion_ids: Sequence[str] | None = None,
) -> tuple[WindowProvenance, ...]:
  """Validate an aligned provenance sidecar without touching token rows."""
  expected = _provenance_expected(cache, loaded, split, motion_ids)
  return _validate_provenance(
    provenance_cache_path(cache.path), cache=cache, expected=expected
  )


def _write_provenance(
  path: Path,
  records: Sequence[WindowProvenance],
  *,
  expected: Mapping[str, object],
) -> None:
  payload = {
    **expected,
    "provenance_sha256": _provenance_digest(records),
    "records": _canonical_provenance(records),
  }
  temporary: Path | None = None
  try:
    with tempfile.NamedTemporaryFile(
      dir=path.parent,
      prefix=f".{path.stem}.",
      suffix=".json",
      mode="w",
      encoding="utf-8",
      delete=False,
    ) as handle:
      temporary = Path(handle.name)
      json.dump(payload, handle, sort_keys=True, indent=2)
      handle.write("\n")
    os.replace(temporary, path)
    temporary = None
  except OSError as exc:
    raise CacheProvenanceError(f"could not publish provenance sidecar {path}") from exc
  finally:
    if temporary is not None:
      temporary.unlink(missing_ok=True)


def prepare_token_cache(
  source: DatasetSource,
  loaded: LoadedDataset,
  split: str,
  *,
  motion_ids: Sequence[str] | None = None,
  dtype: Any = np.float32,
  max_cache_gib: float = 8.0,
  jobs: int = 1,
  source_hash: str | None = None,
  lock_timeout: float = 30.0,
) -> PreparedTokenCache:
  identity_hash = (
    _source_dataset_hash(source, loaded) if source_hash is None else source_hash
  )
  cache_path = dataset_token_cache_path(
    source,
    loaded,
    split,
    dtype=dtype,
    motion_ids=motion_ids,
    source_hash=identity_hash,
  )
  had_pair = cache_path.is_file() and cache_path.with_suffix(".json").is_file()
  validated: list[ValidatedTokenCache] = []
  with CacheLock(cache_path, timeout=lock_timeout):
    cache_path, cache_count = _build_token_cache_unlocked(
      source,
      split,
      cache_path,
      motion_ids=motion_ids,
      dtype=dtype,
      max_cache_gib=max_cache_gib,
      jobs=jobs,
      loaded_dataset=loaded,
      source_hash=identity_hash,
      validated_output=validated,
    )
    if len(validated) != 1:
      raise CacheProvenanceError("cache builder did not return one validated cache")
    cache = validated[0]
    identity = {
      "directory": str(source.directory),
      "assignments_hash": loaded.assignments.sha256(),
      "split_coverage": loaded.index.coverage(),
      "split": split,
      "motion_ids": cache.metadata.get("motion_ids"),
      "limit": cache.metadata.get("limit"),
      "source_dataset_hash": cache.metadata["source_dataset_hash"],
      "contract_hash": cache.metadata["contract_hash"],
      "projection_hash": cache.metadata["projection_hash"],
      "projection_hashes": {
        "matrix": loaded.projection.matrix_sha256,
        "pseudoinverse": loaded.projection.pseudoinverse_sha256,
        "statistics": loaded.projection.statistics_sha256,
      },
    }
    cache = ValidatedTokenCache(cache.path, cache.metadata, cache.cache, identity)
    cache_count = _metadata_int(cache.metadata, "window_count")
    sidecar = provenance_cache_path(cache_path)
    if sidecar.exists():
      records = validate_provenance_cache(cache, loaded, split, motion_ids=motion_ids)
      provenance_status = "reused"
    else:
      if jobs > 1:
        from .token_cache_builder import build_provenance_parallel

        records = build_provenance_parallel(
          loaded,
          split=split,
          jobs=jobs,
          directory=source.directory,
          contract_path=source.contract_path,
          cache_path=cache.path,
          dtype=dtype,
        )
      else:
        records = tuple(
          iter_index_provenance(loaded, split, motion_ids=motion_ids, limit=None)
        )
      if len(records) != cache_count:
        raise CacheProvenanceError("provenance count does not match token cache")
      _write_provenance(
        sidecar,
        records,
        expected=_provenance_expected(cache, loaded, split, motion_ids),
      )
      provenance_status = "built"
  return PreparedTokenCache(
    cache=cache,
    provenance=records,
    cache_status="reused" if had_pair else "built",
    provenance_status=provenance_status,
  )


def validate_token_cache_read_only(
  source: DatasetSource,
  loaded: LoadedDataset,
  split: str,
  *,
  motion_ids: Sequence[str] | None = None,
  dtype: Any = np.float32,
  source_hash: str | None = None,
  lock_timeout: float = 0.0,
) -> PreparedTokenCache:
  identity_hash = (
    _source_dataset_hash(source, loaded) if source_hash is None else source_hash
  )
  cache_path = dataset_token_cache_path(
    source,
    loaded,
    split,
    dtype=dtype,
    motion_ids=motion_ids,
    source_hash=identity_hash,
  )
  lock_path = cache_path.with_suffix(".lock")
  if lock_path.exists():
    raise TokenCacheBusyError(f"token cache writer is busy: {lock_path}")
  cache = _read_validated_cache(
    source,
    split,
    cache_path,
    motion_ids=motion_ids,
    dtype=dtype,
    loaded=loaded,
    source_hash=identity_hash,
  )
  records = validate_provenance_cache(cache, loaded, split, motion_ids=motion_ids)
  return PreparedTokenCache(cache, records, "ready", "ready")


def parse_splits(value: str) -> tuple[str, ...]:
  """Validate requested split names before opening or building cache data."""
  parts = [part.strip() for part in str(value).split(",")]
  if not parts or any(not part for part in parts):
    raise DatasetError("splits must be non-empty comma-separated names")
  if len(set(parts)) != len(parts):
    raise DatasetError("splits must not contain duplicates")
  for part in parts:
    _validate_split(part)
  return tuple(parts)


__all__ = [
  "CacheProvenanceError",
  "PreparedTokenCache",
  "parse_splits",
  "prepare_token_cache",
  "provenance_cache_path",
  "validate_provenance_cache",
  "validate_token_cache_read_only",
]
