"""Lazy dataset loading and token-cache utilities for diffusion training.

The D1 window index deliberately remains the source of truth for eligibility.  This
module only materializes its already-qualified windows into a bounded, float32
memory-mapped cache for the trainer; filtering therefore cannot create new windows
or change split membership.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, cast

import numpy as np
import torch
from torch import Tensor

from .contract import DiffusionContract
from .dataset import (
  DatasetError,
  SplitAssignments,
  WindowDataset,
  WindowIndex,
  WindowRef,
)
from .projection import ProjectionBundle, ProjectionError
from .storage import AppendOnlyShardStore

_SPLITS = ("train", "validation", "test")
_CACHE_VERSION = "mjlab-x2-diffusion-token-cache-v2"
_CACHE_NON_IDENTITY_KEYS = frozenset(("created_at", "content_sha256"))
_DEFAULT_MAX_CACHE_GIB = 8.0


class WindowCacheError(DatasetError):
  """A token cache or its source metadata is invalid."""


class TokenCacheBusyError(WindowCacheError):
  """Another process owns the cache writer lock."""


def cache_lock_path(path: str | Path) -> Path:
  """Return the per-cache writer lock path.

  The lock uses exclusive file creation.  This is safe when the cache directory
  provides atomic ``O_EXCL`` creation, which includes local filesystems and NAS
  mounts with normal close-to-open semantics.  A stale lock is never deleted by
  this module.  An operator must remove it after checking the owning process.
  """
  return Path(path).with_suffix(".lock")


class CacheLock:
  """Bounded cross-process exclusion for one cache identity."""

  def __init__(self, path: str | Path, *, timeout: float = 30.0) -> None:
    if not np.isfinite(timeout) or timeout < 0.0:
      raise DatasetError("cache lock timeout must be finite and non-negative")
    self.path = cache_lock_path(path)
    self.timeout = float(timeout)
    self._contents: str | None = None

  def __enter__(self) -> "CacheLock":
    token = uuid.uuid4().hex
    payload = {
      "pid": os.getpid(),
      "host": socket.gethostname(),
      "token": token,
      "created_at": datetime.now(timezone.utc).isoformat(),
    }
    contents = json.dumps(payload, sort_keys=True) + "\n"
    deadline = time.monotonic() + self.timeout
    while True:
      try:
        descriptor = os.open(
          self.path,
          os.O_CREAT | os.O_EXCL | os.O_WRONLY,
          0o644,
        )
        try:
          os.write(descriptor, contents.encode("utf-8"))
        finally:
          os.close(descriptor)
        self._contents = contents
        return self
      except FileExistsError:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
          raise TokenCacheBusyError(
            f"token cache writer is busy: {self.path}; waited {self.timeout:.1f}s"
          ) from None
        time.sleep(min(0.1, remaining))
      except OSError as exc:
        raise WindowCacheError(
          f"could not create token cache lock {self.path}"
        ) from exc

  def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
    contents = self._contents
    self._contents = None
    if contents is None:
      return
    try:
      if self.path.read_text(encoding="utf-8") == contents:
        self.path.unlink()
    except OSError:
      # Never remove a lock whose ownership cannot be established.  The caller
      # already has the original build exception, if any.
      return


def recover_cache_lock(path: str | Path) -> bool:
  """Remove one explicitly selected lock only when its local owner is dead.

  This function does not inspect lock age.  It accepts only a lock created by
  :class:`CacheLock`, requires the recorded host to be this host, and refuses
  removal while the recorded process exists or cannot be checked.  A caller
  must invoke this function explicitly after crash inspection.
  """
  lock_path = cache_lock_path(path)
  if not lock_path.exists():
    return False
  try:
    contents = lock_path.read_text(encoding="utf-8")
    payload = json.loads(contents)
  except (OSError, TypeError, json.JSONDecodeError) as exc:
    raise WindowCacheError(
      f"cannot recover malformed token cache lock {lock_path}; inspect and remove it manually"
    ) from exc
  if not isinstance(payload, dict):
    raise WindowCacheError(
      f"cannot recover malformed token cache lock {lock_path}; inspect and remove it manually"
    )
  pid = payload.get("pid")
  host = payload.get("host")
  token = payload.get("token")
  if (
    isinstance(pid, bool)
    or not isinstance(pid, int)
    or pid <= 0
    or not isinstance(host, str)
    or not host
    or not isinstance(token, str)
    or not token
  ):
    raise WindowCacheError(
      f"cannot recover malformed token cache lock {lock_path}; inspect and remove it manually"
    )
  if host != socket.gethostname():
    raise TokenCacheBusyError(
      f"token cache lock belongs to host {host!r}; remove it on that host after inspection"
    )
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    pass
  except PermissionError as exc:
    raise TokenCacheBusyError(
      f"token cache lock owner pid {pid} cannot be checked; remove it after inspection"
    ) from exc
  except OSError as exc:
    raise TokenCacheBusyError(
      f"token cache lock owner pid {pid} cannot be checked; remove it after inspection"
    ) from exc
  else:
    raise TokenCacheBusyError(
      f"token cache lock owner pid {pid} is still alive; no lock was removed"
    )
  try:
    if lock_path.read_text(encoding="utf-8") != contents:
      raise TokenCacheBusyError(
        "token cache lock changed during recovery; retry after inspection"
      )
    lock_path.unlink()
  except TokenCacheBusyError:
    raise
  except OSError as exc:
    raise WindowCacheError(f"could not remove token cache lock {lock_path}") from exc
  return True


@dataclass(frozen=True, slots=True)
class DatasetSource:
  """Paths needed to reopen one D1 dataset with its frozen contract."""

  directory: Path
  contract_path: Path

  def __post_init__(self) -> None:
    directory = Path(self.directory)
    contract_path = Path(self.contract_path)
    if not directory:
      raise DatasetError("dataset directory must be non-empty")
    if not contract_path:
      raise DatasetError("contract path must be non-empty")
    object.__setattr__(self, "directory", directory)
    object.__setattr__(self, "contract_path", contract_path)

  @classmethod
  def resolve(
    cls, directory: str | Path, *, contract_path: str | Path
  ) -> "DatasetSource":
    """Resolve a dataset directory and the immutable D0 YAML path."""
    return cls(Path(directory), Path(contract_path))

  def load(self) -> "LoadedDataset":
    """Load contract, projection, assignments, store and lazy window index.

    The YAML-derived contract is intentionally threaded through every D1 object.
    Opening a pilot store with :data:`DEFAULT_CONTRACT` would produce a different
    identity hash and is rejected by :class:`AppendOnlyShardStore`.
    """
    contract = DiffusionContract.from_yaml(self.contract_path)
    store_path = self.directory / "store"
    offline = self.directory / "offline"
    if not store_path.is_dir():
      raise DatasetError(f"dataset store is missing: {store_path}")
    if not offline.is_dir():
      raise DatasetError(f"dataset offline directory is missing: {offline}")

    try:
      assignments_payload = json.loads(
        (offline / "assignments.json").read_text(encoding="utf-8")
      )
      assignments = SplitAssignments.from_dict(assignments_payload)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
      raise DatasetError("invalid offline split assignments") from exc

    try:
      projection = ProjectionBundle.load(offline / "projection.npz", contract=contract)
    except (OSError, ProjectionError, ValueError) as exc:
      raise DatasetError(f"invalid offline projection bundle: {exc}") from exc

    store = AppendOnlyShardStore(store_path, contract=contract)
    if store.contract.identity_hash() != contract.identity_hash():
      raise DatasetError("store contract identity differs from loaded YAML contract")
    if projection.contract.identity_hash() != contract.identity_hash():
      raise DatasetError(
        "projection contract identity differs from loaded YAML contract"
      )

    index, index_source = self._resolve_index(store, assignments, contract)
    self._validate_dataset_manifest(index.coverage(), offline / "dataset.json")
    return LoadedDataset(
      contract, projection, assignments, store, index, index_source=index_source
    )

  def _resolve_index(
    self,
    store: AppendOnlyShardStore,
    assignments: SplitAssignments,
    contract: DiffusionContract,
  ) -> tuple[WindowIndex, str]:
    """Prefer the persisted index; rebuild only when it cannot be trusted.

    The index is derived metadata and belongs to the dataset.  A stale index
    means the store or the assignments changed, so rebuilding is the correct
    recovery and is exactly what ran before this artifact existed; a malformed
    or inconsistent index is refused rather than accepted.
    """
    import sys

    from .dataset import load_window_index, window_index_path

    persisted = window_index_path(self.directory)
    if persisted.is_file():
      try:
        loaded = load_window_index(
          persisted, store=store, assignments=assignments, contract=contract
        )
        return loaded, "persisted"
      except (DatasetError, OSError, ValueError, KeyError) as exc:
        print(
          f"window index {persisted} is unusable ({exc}); rebuilding it",
          file=sys.stderr,
        )
    return WindowIndex.build(store, assignments, contract=contract), "rebuilt"

  @staticmethod
  def _validate_dataset_manifest(coverage: Mapping[str, int], path: Path) -> None:
    """Check persisted coverage when D1 recorded it, including empty splits."""
    if not path.is_file():
      raise DatasetError(f"offline dataset manifest is missing: {path}")
    try:
      payload = json.loads(path.read_text(encoding="utf-8"))
      recorded = payload["windows"]
      expected = {name: int(recorded[name]) for name in _SPLITS}
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
      raise DatasetError("invalid offline dataset manifest") from exc
    actual = {name: int(coverage.get(name, 0)) for name in _SPLITS}
    if actual != expected:
      raise DatasetError(
        f"window coverage differs from dataset.json: expected {expected}, got {actual}"
      )

  def token_cache_path(self, split: str, key: str | None = None) -> Path:
    """Return the dataset-scoped ``.npy`` path for a split cache.

    A key names one cache identity, so a different dataset, contract or
    projection can never resolve to the same file.
    """
    _validate_split(split)
    name = f"tokens-{split}.npy" if key is None else f"tokens-{split}-{key}.npy"
    return self.directory / "offline" / name


@dataclass(frozen=True, slots=True)
class LoadedDataset:
  """All D1 artifacts opened under one YAML-derived contract identity."""

  contract: DiffusionContract
  projection: ProjectionBundle
  assignments: SplitAssignments
  store: AppendOnlyShardStore
  index: WindowIndex
  index_source: str = "rebuilt"

  def dataset(self, split: str) -> WindowDataset:
    """Return a lazy projected window view, including an empty split."""
    _validate_split(split)
    return WindowDataset(self.index, split, self.projection)

  def refs(self, split: str) -> tuple[WindowRef, ...]:
    """Return immutable eligible references for ``split``."""
    _validate_split(split)
    return self.index.refs_for(split)


@dataclass(frozen=True, slots=True)
class WindowProvenance:
  """Stable provenance for one window without retaining its token array."""

  split: str
  group_key: str
  motion_id: str
  phase: str
  ou_noise_norm: float
  pair_id: str
  start_tick: int

  def __post_init__(self) -> None:
    _validate_provenance_fields(
      self.split, self.phase, self.ou_noise_norm, self.start_tick
    )
    object.__setattr__(self, "ou_noise_norm", float(self.ou_noise_norm))


@dataclass(frozen=True, slots=True)
class ValidatedTokenCache:
  """One cache pair validated once for reuse by a consumer."""

  path: Path
  metadata: Mapping[str, object]
  cache: np.memmap
  dataset_identity: Mapping[str, object] | None = None

  def __post_init__(self) -> None:
    path = Path(self.path)
    if not isinstance(self.cache, np.memmap):
      raise WindowCacheError("validated token cache must be a NumPy memmap")
    object.__setattr__(self, "path", path)
    object.__setattr__(self, "metadata", dict(self.metadata))
    if self.dataset_identity is not None:
      object.__setattr__(self, "dataset_identity", dict(self.dataset_identity))


@dataclass(frozen=True, slots=True)
class WindowRecord:
  """One materialized token window plus stable D1 provenance."""

  tokens: np.ndarray
  split: str
  group_key: str
  motion_id: str
  phase: str
  ou_noise_norm: float
  pair_id: str
  start_tick: int

  def __post_init__(self) -> None:
    tokens = np.asarray(self.tokens, dtype=np.float32)
    if (
      tokens.ndim != 2
      or tokens.shape[0] != 41
      or tokens.shape[1] not in {167, 231}
      or not np.isfinite(tokens).all()
    ):
      raise DatasetError(
        "window tokens must be finite with shape (41, 167) for active data "
        "or (41, 231) for explicit legacy data"
      )
    _validate_provenance_fields(
      self.split, self.phase, self.ou_noise_norm, self.start_tick
    )
    object.__setattr__(self, "tokens", np.ascontiguousarray(tokens))
    object.__setattr__(self, "ou_noise_norm", float(self.ou_noise_norm))


def _validate_provenance_fields(
  split: str, phase: str, ou_noise_norm: float, start_tick: int
) -> None:
  """Validate fields shared by materialized and bounded window metadata."""
  if split not in _SPLITS:
    raise DatasetError(f"unknown window split {split!r}")
  if phase not in {"", "clean", "ou"}:
    raise DatasetError(f"unknown window phase {phase!r}")
  if not np.isfinite(ou_noise_norm) or ou_noise_norm < 0.0:
    raise DatasetError("ou_noise_norm must be finite and non-negative")
  if start_tick < 0:
    raise DatasetError("window start_tick must be non-negative")


def _record_provenance(record: WindowRecord) -> WindowProvenance:
  """Drop the token payload while retaining the stable window identity."""
  return WindowProvenance(
    split=record.split,
    group_key=record.group_key,
    motion_id=record.motion_id,
    phase=record.phase,
    ou_noise_norm=record.ou_noise_norm,
    pair_id=record.pair_id,
    start_tick=record.start_tick,
  )


def _validate_split(split: str) -> None:
  if split not in _SPLITS:
    raise DatasetError(f"unknown split {split!r}; expected one of {_SPLITS}")


def _source_and_loaded(
  source: DatasetSource | LoadedDataset,
) -> tuple[DatasetSource | None, LoadedDataset]:
  if isinstance(source, DatasetSource):
    return source, source.load()
  if isinstance(source, LoadedDataset):
    return None, source
  raise TypeError("source must be DatasetSource or LoadedDataset")


def _normalise_motion_ids(motion_ids: Sequence[str] | None) -> tuple[str, ...] | None:
  if motion_ids is None:
    return None
  values = tuple(sorted({str(value) for value in motion_ids}))
  if any(not value for value in values):
    raise DatasetError("motion_ids must contain non-empty strings")
  return values


def _record_from_sample(sample: Any, ref: WindowRef, split: str) -> WindowRecord:
  tokens = np.asarray(sample.tokens, dtype=np.float32)
  rows = sample.rows
  if (
    tokens.ndim != 2
    or tokens.shape[0] != 41
    or tokens.shape[1] not in {167, 231}
    or not np.isfinite(tokens).all()
  ):
    raise DatasetError("window materialization produced an invalid token shape")
  if not rows:
    raise DatasetError("window materialization produced no source rows")
  first = rows[0]
  phase = first.provenance.get("phase", "")
  if phase not in {"", "clean", "ou"}:
    phase = ""
  pair_id = first.provenance.get("pair_id", "")
  noise = np.stack([np.asarray(row.ou_noise, dtype=np.float64) for row in rows])
  if not np.isfinite(noise).all():
    raise DatasetError("window provenance contains non-finite OU noise")
  # A per-row mean L2 norm is stable across the fixed 41-step window and keeps
  # the provenance diagnostic in the same units as one action vector.
  noise_norm = float(np.linalg.norm(noise, axis=1).mean())
  return WindowRecord(
    tokens=tokens,
    split=split,
    group_key=ref.group_key,
    motion_id=first.motion_id,
    phase=phase,
    ou_noise_norm=noise_norm,
    pair_id=pair_id,
    start_tick=ref.start_tick,
  )


def _iter_window_records(
  loaded: LoadedDataset,
  split: str,
  *,
  motion_ids: tuple[str, ...] | None,
  limit: int | None,
) -> Iterator[WindowRecord]:
  """Yield records one at a time so cache construction stays bounded."""
  view = loaded.dataset(split)
  selected = 0
  for index, ref in enumerate(loaded.refs(split)):
    sample = view[index]
    record = _record_from_sample(sample, ref, split)
    if motion_ids is not None and record.motion_id not in motion_ids:
      continue
    yield record
    selected += 1
    if limit is not None and selected >= limit:
      break


def _iter_index_provenance(
  loaded: LoadedDataset,
  split: str,
  *,
  motion_ids: tuple[str, ...] | None,
  limit: int | None,
) -> Iterator[WindowProvenance]:
  """Yield provenance directly from indexed raw rows without projecting tokens."""
  selected = 0
  for ref in loaded.refs(split):
    rows = loaded.index._rows(ref)
    if not rows:
      raise DatasetError("indexed window contains no source rows")
    first = rows[0]
    if motion_ids is not None and first.motion_id not in motion_ids:
      continue
    phase = first.provenance.get("phase", "")
    if phase not in {"", "clean", "ou"}:
      phase = ""
    noise = np.stack([np.asarray(row.ou_noise, dtype=np.float64) for row in rows])
    if not np.isfinite(noise).all():
      raise DatasetError("window provenance contains non-finite OU noise")
    yield WindowProvenance(
      split=split,
      group_key=ref.group_key,
      motion_id=first.motion_id,
      phase=phase,
      ou_noise_norm=float(np.linalg.norm(noise, axis=1).mean()),
      pair_id=first.provenance.get("pair_id", ""),
      start_tick=ref.start_tick,
    )
    selected += 1
    if limit is not None and selected >= limit:
      break


def iter_index_provenance(
  source: DatasetSource | LoadedDataset,
  split: str,
  *,
  motion_ids: Sequence[str] | None = None,
  limit: int | None = None,
) -> Iterator[WindowProvenance]:
  """Stream provenance from raw rows without calling ``WindowDataset._sample``."""
  _validate_split(split)
  if limit is not None and limit < 0:
    raise DatasetError("limit must be non-negative")
  selected_motions = _normalise_motion_ids(motion_ids)
  _, loaded = _source_and_loaded(source)
  yield from _iter_index_provenance(
    loaded,
    split,
    motion_ids=selected_motions,
    limit=limit,
  )


def iter_window_provenance(
  source: DatasetSource | LoadedDataset,
  split: str,
  *,
  motion_ids: Sequence[str] | None = None,
  limit: int | None = None,
) -> Iterator[WindowProvenance]:
  """Stream stable window metadata without retaining token arrays."""
  _validate_split(split)
  if limit is not None and limit < 0:
    raise DatasetError("limit must be non-negative")
  selected_motions = _normalise_motion_ids(motion_ids)
  _, loaded = _source_and_loaded(source)
  for record in _iter_window_records(
    loaded,
    split,
    motion_ids=selected_motions,
    limit=limit,
  ):
    yield _record_provenance(record)


def load_window_records(
  source: DatasetSource | LoadedDataset,
  split: str,
  *,
  motion_ids: Sequence[str] | None = None,
  limit: int | None = None,
) -> list[WindowRecord]:
  """Materialize eligible records, applying filters only after indexing."""
  _validate_split(split)
  if limit is not None and limit < 0:
    raise DatasetError("limit must be non-negative")
  selected_motions = _normalise_motion_ids(motion_ids)
  _, loaded = _source_and_loaded(source)
  return list(
    _iter_window_records(
      loaded,
      split,
      motion_ids=selected_motions,
      limit=limit,
    )
  )


def _sha256_bytes(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


def _projection_identity(projection: ProjectionBundle) -> str:
  payload = json.dumps(
    projection.as_dict(), sort_keys=True, separators=(",", ":")
  ).encode("utf-8")
  return _sha256_bytes(payload)


def _source_dataset_hash(source: DatasetSource, loaded: LoadedDataset) -> str:
  """Hash source manifests and split metadata, including shard digests."""
  files = (
    source.directory / "store" / "manifest.json",
    source.directory / "offline" / "assignments.json",
    source.directory / "offline" / "dataset.json",
  )
  entries: dict[str, str] = {}
  for path in files:
    if not path.is_file():
      raise DatasetError(f"dataset identity input is missing: {path}")
    entries[str(path.relative_to(source.directory))] = _sha256_bytes(path.read_bytes())
  payload = {
    "contract_hash": loaded.contract.identity_hash(),
    "files": entries,
    "row_count": loaded.store.row_count,
    "shard_count": loaded.store.shard_count,
    "assignments_hash": loaded.assignments.sha256(),
    "coverage": loaded.index.coverage(),
  }
  return _sha256_bytes(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
  )


def token_cache_identity_key(
  *,
  source_hash: str,
  split: str,
  contract_hash: str,
  projection_hash: str,
  dtype: Any = np.float32,
  motion_ids: Sequence[str] | None = None,
  limit: int | None = None,
) -> str:
  """Return a short key that names one token-cache identity.

  The key becomes part of the file name.  Two identities therefore cannot
  share a path, and a stale cache cannot be mistaken for the wanted one.
  """
  _validate_split(split)
  selected = _normalise_motion_ids(motion_ids)
  payload = {
    "format": _CACHE_VERSION,
    "source_dataset_hash": source_hash,
    "split": split,
    "contract_hash": contract_hash,
    "projection_hash": projection_hash,
    "dtype": np.dtype(dtype).str,
    "motion_ids": None if selected is None else list(selected),
    "limit": limit,
  }
  encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
  return _sha256_bytes(encoded)[:16]


def dataset_token_cache_path(
  source: DatasetSource,
  loaded: LoadedDataset,
  split: str,
  *,
  dtype: Any = np.float32,
  motion_ids: Sequence[str] | None = None,
  limit: int | None = None,
  source_hash: str | None = None,
) -> Path:
  """Resolve the dataset-scoped token-cache path for this source identity.

  The cache belongs to the dataset, not to one training run.  It is a function
  of the stored rows, the contract and the projection, so every run over the
  same dataset shares it.
  """
  key = token_cache_identity_key(
    source_hash=(
      _source_dataset_hash(source, loaded) if source_hash is None else source_hash
    ),
    split=split,
    contract_hash=loaded.contract.identity_hash(),
    projection_hash=_projection_identity(loaded.projection),
    dtype=dtype,
    motion_ids=motion_ids,
    limit=limit,
  )
  return source.token_cache_path(split, key)


def _source_for_loaded(loaded: LoadedDataset) -> DatasetSource:
  """Resolve a source path for a loaded dataset without reopening it."""
  return DatasetSource(
    Path(loaded.store.root).parent,
    Path(
      "docs/plans/beyondmimic_diffusion_d0_contract.yaml"
      if loaded.contract.projection_rows
      else "docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml"
    ),
  )


def _cache_source_and_loaded(
  source: DatasetSource | LoadedDataset,
  loaded_dataset: LoadedDataset | None,
) -> tuple[DatasetSource, LoadedDataset]:
  if loaded_dataset is not None:
    source_obj = (
      source
      if isinstance(source, DatasetSource)
      else _source_for_loaded(loaded_dataset)
    )
    return source_obj, loaded_dataset
  source_obj, loaded = _source_and_loaded(source)
  if source_obj is None:
    source_obj = _source_for_loaded(loaded)
  return source_obj, loaded


def _cache_metadata(
  *,
  source_hash: str,
  split: str,
  contract_hash: str,
  projection_hash: str,
  window_count: int,
  dtype: np.dtype[Any],
  motion_ids: tuple[str, ...] | None,
  limit: int | None,
  shape: tuple[int, int, int],
  created_at: str | None,
  content_sha256: str | None,
) -> dict[str, object]:
  metadata: dict[str, object] = {
    "format": _CACHE_VERSION,
    "source_dataset_hash": source_hash,
    "split": split,
    "contract_hash": contract_hash,
    "projection_hash": projection_hash,
    "window_count": int(window_count),
    "dtype": np.dtype(dtype).str,
    "shape": list(shape),
    "motion_ids": None if motion_ids is None else list(motion_ids),
    "limit": limit,
    "content_sha256": content_sha256,
  }
  if created_at is not None:
    metadata["created_at"] = created_at
  return metadata


def _read_cache_metadata(path: Path) -> dict[str, object]:
  try:
    payload = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError, TypeError) as exc:
    raise WindowCacheError(f"invalid token cache metadata: {path}") from exc
  if not isinstance(payload, dict):
    raise WindowCacheError(f"token cache metadata must be an object: {path}")
  return payload


def _identity_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
  """Return cache metadata fields that identify the requested source."""
  return {
    key: value for key, value in metadata.items() if key not in _CACHE_NON_IDENTITY_KEYS
  }


def _sha256_file(path: Path) -> str:
  """Hash a file in bounded chunks, including the NumPy header bytes."""
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _validate_cache(
  path: Path,
  metadata_path: Path,
  expected: Mapping[str, object],
) -> tuple[dict[str, object], np.memmap]:
  """Validate identity, shape and bytes; digest verification is O(file size)."""
  if not path.is_file() or not metadata_path.is_file():
    raise WindowCacheError(
      f"token cache requires both {path.name} and {metadata_path.name}"
    )
  metadata = _read_cache_metadata(metadata_path)
  if metadata.get("format") != _CACHE_VERSION:
    raise WindowCacheError(
      "unsupported token cache format; rebuild the cache with the current D2 code"
    )
  if _identity_metadata(metadata) != _identity_metadata(expected):
    raise WindowCacheError("token cache metadata does not match the requested dataset")
  try:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
  except (OSError, ValueError) as exc:
    raise WindowCacheError(f"invalid token cache array: {path}") from exc
  expected_shape = tuple(int(value) for value in expected["shape"])  # type: ignore[arg-type]
  if array.shape != expected_shape:
    raise WindowCacheError(
      f"token cache shape mismatch: expected {expected_shape}, got {array.shape}"
    )
  if np.dtype(array.dtype).str != str(expected["dtype"]):
    raise WindowCacheError(
      f"token cache dtype mismatch: expected {expected['dtype']}, got {array.dtype.str}"
    )
  if not isinstance(array, np.memmap):
    raise WindowCacheError("token cache must be a NumPy .npy memmap")
  content_sha256 = metadata.get("content_sha256")
  if not isinstance(content_sha256, str) or len(content_sha256) != 64:
    raise WindowCacheError(
      "token cache is missing a valid content_sha256; rebuild the cache"
    )
  actual_sha256 = _sha256_file(path)
  if actual_sha256 != content_sha256:
    raise WindowCacheError(
      "token cache content_sha256 does not match the .npy bytes; rebuild the cache"
    )
  return metadata, array


def _fill_token_cache_range(payload: Any) -> tuple[int, ...]:
  """Materialise one bounded, segment-owned cache task."""
  from .token_cache_builder import _fill_token_cache_range as fill_range

  return fill_range(payload)


def _fill_token_cache_parallel(
  temporary_array: Path,
  source: DatasetSource,
  split: str,
  count: int,
  dtype: Any,
  jobs: int,
) -> None:
  """Fill a parent-created cache with bounded shard-local workers."""
  from .token_cache_builder import fill_token_cache_parallel

  try:
    fill_token_cache_parallel(temporary_array, source, split, count, dtype, jobs)
  except WindowCacheError:
    raise
  except (DatasetError, OSError, RuntimeError, TypeError, ValueError) as exc:
    raise WindowCacheError("parallel token cache materialization failed") from exc


def _build_token_cache_unlocked(
  source: DatasetSource | LoadedDataset,
  split: str,
  path: str | Path,
  *,
  motion_ids: Sequence[str] | None = None,
  dtype: Any = np.float32,
  max_cache_gib: float = _DEFAULT_MAX_CACHE_GIB,
  jobs: int = 1,
  loaded_dataset: LoadedDataset | None = None,
  source_hash: str | None = None,
  validated_output: list[ValidatedTokenCache] | None = None,
) -> tuple[Path, int]:
  """Build or validate a bounded float32 ``.npy`` token cache.

  Existing caches are never silently replaced: a source, projection, split,
  filter, dtype or count mismatch raises :class:`WindowCacheError`.
  """
  _validate_split(split)
  selected_motions = _normalise_motion_ids(motion_ids)
  if not np.isfinite(max_cache_gib) or max_cache_gib <= 0.0:
    raise DatasetError("max_cache_gib must be finite and positive")
  try:
    cache_dtype = np.dtype(dtype)
  except TypeError as exc:
    raise DatasetError("dtype must be a NumPy dtype") from exc
  if cache_dtype.kind != "f" or cache_dtype.itemsize not in {2, 4, 8}:
    raise DatasetError("token cache dtype must be a floating NumPy dtype")
  # Validate before any temporary file exists: the write block below converts
  # ValueError into a generic write error, which would hide these reasons.
  if jobs < 1:
    raise DatasetError("jobs must be one or more")
  if jobs > 1 and selected_motions is not None:
    raise DatasetError(
      "a parallel token cache does not support a motion filter; use jobs=1"
    )

  source_obj, loaded = _cache_source_and_loaded(source, loaded_dataset)
  # projected size is available before the token walk.  With a motion
  # filter the count needs the records themselves.
  if selected_motions is None:
    selected_count = len(loaded.refs(split))
  else:
    selected_count = sum(
      1
      for _ in _iter_index_provenance(
        loaded,
        split,
        motion_ids=selected_motions,
        limit=None,
      )
    )
  shape = (
    selected_count,
    loaded.contract.window_steps,
    loaded.contract.token_dimension,
  )
  projected_bytes = math.prod(shape) * cache_dtype.itemsize
  projected_gib = projected_bytes / (1024**3)
  print(
    f"token cache projected size: {projected_bytes} bytes ({projected_gib:.6f} GiB), "
    f"{selected_count} windows"
  )
  if projected_gib > max_cache_gib:
    raise WindowCacheError(
      f"token cache for split {split!r} would require {projected_gib:.6f} GiB, "
      f"above max_cache_gib={max_cache_gib:.6f}; raise the ceiling with "
      f"--max-cache-gib (train) or the max_cache_gib argument"
    )

  source_hash = (
    _source_dataset_hash(source_obj, loaded) if source_hash is None else source_hash
  )
  expected = _cache_metadata(
    source_hash=source_hash,
    split=split,
    contract_hash=loaded.contract.identity_hash(),
    projection_hash=_projection_identity(loaded.projection),
    window_count=selected_count,
    dtype=cache_dtype,
    motion_ids=selected_motions,
    limit=None,
    shape=shape,
    created_at=None,
    content_sha256=None,
  )
  destination = Path(path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  metadata_path = destination.with_suffix(".json")
  if destination.exists() or metadata_path.exists():
    metadata, cache = _validate_cache(destination, metadata_path, expected)
    if validated_output is not None:
      validated_output.append(ValidatedTokenCache(destination, metadata, cache))
    return destination, selected_count

  temporary_array: Path | None = None
  temporary_metadata: Path | None = None
  published_array = False
  try:
    with tempfile.NamedTemporaryFile(
      dir=destination.parent,
      prefix=f".{destination.stem}.",
      suffix=".npy",
      delete=False,
    ) as handle:
      temporary_array = Path(handle.name)
    memmap = np.lib.format.open_memmap(
      temporary_array, mode="w+", dtype=cache_dtype, shape=shape
    )
    if jobs > 1:
      _fill_token_cache_parallel(
        temporary_array, source_obj, split, selected_count, cache_dtype, jobs
      )
    else:
      for index, record in enumerate(
        _iter_window_records(
          loaded,
          split,
          motion_ids=selected_motions,
          limit=None,
        )
      ):
        memmap[index] = record.tokens.astype(cache_dtype, copy=False)
    memmap.flush()
    del memmap
    assert temporary_array is not None
    content_sha256 = _sha256_file(temporary_array)
    os.replace(temporary_array, destination)
    temporary_array = None
    published_array = True

    metadata = dict(expected)
    metadata["content_sha256"] = content_sha256
    metadata["created_at"] = datetime.now(timezone.utc).isoformat()
    with tempfile.NamedTemporaryFile(
      dir=metadata_path.parent,
      prefix=f".{metadata_path.stem}.",
      suffix=".json",
      mode="w",
      encoding="utf-8",
      delete=False,
    ) as handle:
      temporary_metadata = Path(handle.name)
      json.dump(metadata, handle, sort_keys=True, indent=2)
      handle.write("\n")
    os.replace(temporary_metadata, metadata_path)
    temporary_metadata = None
    if validated_output is not None:
      cache = np.load(destination, mmap_mode="r", allow_pickle=False)
      if not isinstance(cache, np.memmap):
        raise WindowCacheError("published token cache must be a NumPy .npy memmap")
      validated_output.append(ValidatedTokenCache(destination, metadata, cache))
  except (OSError, ValueError) as exc:
    if published_array and not metadata_path.exists():
      # This process owns the newly published array and has not published its
      # metadata pair.  Roll back only that owned partial publication so a
      # normal failed build can be retried safely.
      destination.unlink(missing_ok=True)
    raise WindowCacheError(f"could not write token cache {destination}") from exc
  finally:
    if temporary_array is not None:
      temporary_array.unlink(missing_ok=True)
    if temporary_metadata is not None:
      temporary_metadata.unlink(missing_ok=True)
  return destination, selected_count


def build_token_cache(
  source: DatasetSource | LoadedDataset,
  split: str,
  path: str | Path,
  *,
  motion_ids: Sequence[str] | None = None,
  dtype: Any = np.float32,
  max_cache_gib: float = _DEFAULT_MAX_CACHE_GIB,
  jobs: int = 1,
  loaded_dataset: LoadedDataset | None = None,
  source_hash: str | None = None,
  lock_timeout: float = 30.0,
) -> tuple[Path, int]:
  """Build or validate one cache under exclusive cross-process ownership."""
  destination = Path(path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  with CacheLock(destination, timeout=lock_timeout):
    return _build_token_cache_unlocked(
      source,
      split,
      destination,
      motion_ids=motion_ids,
      dtype=dtype,
      max_cache_gib=max_cache_gib,
      jobs=jobs,
      loaded_dataset=loaded_dataset,
      source_hash=source_hash,
    )


class TokenWindowDataset(torch.utils.data.Dataset):
  """Dataset backed by a validated read-only token-cache memmap.

  ``records`` supplies provenance and phase weights.  Supplying ``source``
  additionally re-computes the source/projection identity and rejects a stale
  cache before any sample is returned.
  """

  def __init__(
    self,
    path: str | Path,
    records: Sequence[WindowRecord | WindowProvenance] | None = None,
    *,
    clean_weight: float = 1.0,
    perturbed_weight: float = 1.0,
    source: DatasetSource | LoadedDataset | None = None,
    split: str | None = None,
    motion_ids: Sequence[str] | None = None,
    limit: int | None = None,
    expected_metadata: Mapping[str, object] | None = None,
    validated_cache: ValidatedTokenCache | None = None,
  ) -> None:
    self.path = Path(path)
    metadata_path = self.path.with_suffix(".json")
    metadata = _read_cache_metadata(metadata_path)
    if metadata.get("format") != _CACHE_VERSION:
      raise WindowCacheError(
        "unsupported token cache format; rebuild the cache with the current D2 code"
      )
    if split is None:
      split = str(metadata.get("split", ""))
    _validate_split(split)
    values = None if records is None else tuple(records)
    if values is not None and any(record.split != split for record in values):
      raise WindowCacheError("record split does not match token cache split")
    try:
      raw_count = metadata.get("window_count", -1)
      count = int(raw_count) if isinstance(raw_count, (int, str, float)) else -1
    except (TypeError, ValueError):
      count = -1
    if count < 0 or (values is not None and len(values) != count):
      raise WindowCacheError("token cache window count does not match records")
    selected_motions = _normalise_motion_ids(motion_ids)
    if selected_motions is not None:
      raw_motions = metadata.get("motion_ids")
      if raw_motions is None:
        recorded_motions: tuple[str, ...] = ()
      elif isinstance(raw_motions, list):
        recorded_motions = tuple(str(value) for value in raw_motions)
      else:
        raise WindowCacheError("token cache motion_ids metadata is invalid")
      if recorded_motions != selected_motions:
        raise WindowCacheError("token cache motion filter does not match request")
    if limit != metadata.get("limit"):
      raise WindowCacheError("token cache limit does not match request")

    if expected_metadata is not None:
      if _identity_metadata(metadata) != _identity_metadata(expected_metadata):
        raise WindowCacheError("token cache metadata does not match expected metadata")
    if source is not None and validated_cache is None:
      source_obj, loaded = _source_and_loaded(source)
      if source_obj is None:
        source_obj = DatasetSource(
          Path(loaded.store.root).parent,
          Path(
            "docs/plans/beyondmimic_diffusion_d0_contract.yaml"
            if loaded.contract.projection_rows
            else "docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml"
          ),
        )
      raw_shape = metadata.get("shape")
      if not isinstance(raw_shape, list) or len(raw_shape) != 3:
        raise WindowCacheError("token cache metadata has an invalid shape")
      shape_values = cast("list[int | float | str]", raw_shape)
      if any(not isinstance(value, (int, float, str)) for value in shape_values):
        raise WindowCacheError("token cache metadata has an invalid shape")
      shape = tuple(int(value) for value in shape_values)
      raw_content_sha256 = metadata.get("content_sha256")
      request_content_sha256 = (
        raw_content_sha256 if isinstance(raw_content_sha256, str) else None
      )
      expected_request = _cache_metadata(
        source_hash=_source_dataset_hash(source_obj, loaded),
        split=split,
        contract_hash=loaded.contract.identity_hash(),
        projection_hash=_projection_identity(loaded.projection),
        window_count=count,
        dtype=np.dtype(str(metadata.get("dtype", ""))),
        motion_ids=selected_motions,
        limit=limit,
        shape=shape,  # type: ignore[arg-type]
        created_at=None,
        content_sha256=request_content_sha256,
      )
      if _identity_metadata(metadata) != _identity_metadata(expected_request):
        raise WindowCacheError("token cache is stale for the requested source")
      dataset_identity = {
        "directory": str(source_obj.directory),
        "assignments_hash": loaded.assignments.sha256(),
        "split_coverage": loaded.index.coverage(),
        "split": split,
        "motion_ids": None if selected_motions is None else list(selected_motions),
        "limit": limit,
        "source_dataset_hash": metadata["source_dataset_hash"],
        "contract_hash": metadata["contract_hash"],
        "projection_hash": metadata["projection_hash"],
        "projection_hashes": {
          "matrix": loaded.projection.matrix_sha256,
          "pseudoinverse": loaded.projection.pseudoinverse_sha256,
          "statistics": loaded.projection.statistics_sha256,
        },
      }
    elif validated_cache is not None and validated_cache.dataset_identity is not None:
      dataset_identity = dict(validated_cache.dataset_identity)
    else:
      dataset_identity = {
        "directory": str(self.path.parent),
        "source_dataset_hash": metadata["source_dataset_hash"],
        "contract_hash": metadata["contract_hash"],
        "projection_hash": metadata["projection_hash"],
        "split": split,
        "motion_ids": metadata.get("motion_ids"),
        "limit": metadata.get("limit"),
      }

    if validated_cache is not None:
      if validated_cache.path != self.path:
        raise WindowCacheError("validated token cache path does not match dataset path")
      if dict(validated_cache.metadata) != metadata:
        raise WindowCacheError("token cache changed after validation")
    _, cache = (
      (validated_cache.metadata, validated_cache.cache)
      if validated_cache is not None
      else _validate_cache(self.path, metadata_path, metadata)
    )
    if values is None:
      values = tuple()
    self.records = values
    self.dataset_identity = dataset_identity
    self.metadata = metadata
    self.cache = cache
    self.weights = (
      sampling_weights(
        self.records,
        clean_weight=clean_weight,
        perturbed_weight=perturbed_weight,
      )
      if self.records
      else np.ones(count, dtype=np.float64)
    )
    if len(self.weights) != count:
      raise WindowCacheError("token cache weights do not match window count")

  def __len__(self) -> int:
    return int(self.cache.shape[0])

  def __getitem__(self, index: int) -> tuple[Tensor, Tensor, int]:
    if index < 0:
      index += len(self)
    if index < 0 or index >= len(self):
      raise IndexError(index)
    tokens = np.array(self.cache[index], dtype=np.float32, copy=True)
    if not np.isfinite(tokens).all():
      raise WindowCacheError(f"token cache contains non-finite values at row {index}")
    return (
      torch.from_numpy(tokens),
      torch.tensor(float(self.weights[index]), dtype=torch.float32),
      int(index),
    )


def sampling_weights(
  records: Sequence[WindowRecord | WindowProvenance],
  *,
  clean_weight: float,
  perturbed_weight: float,
) -> np.ndarray:
  """Return phase weights without changing which records are eligible."""
  clean_weight = float(clean_weight)
  perturbed_weight = float(perturbed_weight)
  if (
    not np.isfinite(clean_weight)
    or not np.isfinite(perturbed_weight)
    or clean_weight < 0.0
    or perturbed_weight < 0.0
  ):
    raise DatasetError("sampling weights must be finite and non-negative")
  if records and clean_weight == 0.0 and perturbed_weight == 0.0:
    raise DatasetError("at least one sampling weight must be positive")
  result = np.empty(len(records), dtype=np.float64)
  for index, record in enumerate(records):
    if record.phase == "ou":
      result[index] = perturbed_weight
    elif record.phase in {"", "clean"}:
      result[index] = clean_weight
    else:  # pragma: no cover - both provenance types validate this
      raise DatasetError(f"unknown record phase {record.phase!r}")
  if len(result) and not np.any(result > 0.0):
    raise DatasetError("sampling weights select no windows")
  return result


__all__ = [
  "CacheLock",
  "DatasetSource",
  "LoadedDataset",
  "TokenCacheBusyError",
  "TokenWindowDataset",
  "ValidatedTokenCache",
  "WindowCacheError",
  "WindowProvenance",
  "WindowRecord",
  "build_token_cache",
  "cache_lock_path",
  "recover_cache_lock",
  "dataset_token_cache_path",
  "iter_index_provenance",
  "iter_window_provenance",
  "load_window_records",
  "sampling_weights",
  "token_cache_identity_key",
]
