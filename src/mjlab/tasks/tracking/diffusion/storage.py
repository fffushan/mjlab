"""Append-only bounded storage for raw state/latent collector rows.

The store validates everything it reads.  Two offline paths read the same bytes
with less work:

* a metadata-only read decodes the ``metadata`` column and nothing else, and
* a byte-preserving merge copies shards and never re-encodes a row.

Both share the array, metadata and manifest checks below, so a fast path accepts
exactly the rows the row reader accepts.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, TypeGuard

import numpy as np

from .contract import DEFAULT_CONTRACT, DiffusionContract
from .state import WorldState

STORE_FORMAT = "mjlab-x2-diffusion-raw-v1"
MANIFEST_NAME = "manifest.json"

# Files are hashed and copied in bounded blocks, so no file is held in memory.
_BLOCK_BYTES = 8 << 20
# An .npy header is far shorter than this, so probing a header never
# decompresses the row data of that array.
_HEADER_PROBE_BYTES = 4096
# Row identities are compared through 128-bit digests.  A collision can only
# make a bounded check refuse a merge; it can never accept a real duplicate.
_IDENTITY_DIGEST_BYTES = 16
_ROW_METADATA_FIELDS = frozenset(
  {
    "run_id",
    "env_id",
    "episode_id",
    "segment_id",
    "tick",
    "timestamp",
    "motion_id",
    "reference_frame",
    "controller",
    "terminal_evidence",
    "segment_qualified",
    "reset",
    "teleport",
    "reference_boundary",
    "fps",
    "group_key",
    "provenance",
  }
)
_STRING_METADATA_FIELDS = (
  "run_id",
  "episode_id",
  "segment_id",
  "motion_id",
  "controller",
  "group_key",
)
_FLAG_METADATA_FIELDS = (
  "segment_qualified",
  "reset",
  "teleport",
  "reference_boundary",
)
_STATE_FIELDS = (
  "root_position",
  "root_quaternion_wxyz",
  "root_linear_velocity",
  "root_angular_velocity",
  "body_positions",
  "body_linear_velocities",
)


class StorageError(ValueError):
  """A shard or row failed integrity validation."""


@dataclass(frozen=True, slots=True)
class TerminalEvidence:
  """Post-step evidence captured before reset/teleport can overwrite it."""

  available: bool
  terminated: bool
  truncated: bool
  reason: str | None = None
  physical_fall: bool = False
  tracking_rejection: bool = False

  def as_dict(self) -> dict[str, Any]:
    return {
      "available": self.available,
      "terminated": self.terminated,
      "truncated": self.truncated,
      "reason": self.reason,
      "physical_fall": self.physical_fall,
      "tracking_rejection": self.tracking_rejection,
    }

  @classmethod
  def from_dict(cls, payload: Mapping[str, Any]) -> "TerminalEvidence":
    required = {
      "available",
      "terminated",
      "truncated",
      "reason",
      "physical_fall",
      "tracking_rejection",
    }
    if set(payload) != required:
      raise StorageError("malformed terminal evidence fields")
    boolean_fields = (
      "available",
      "terminated",
      "truncated",
      "physical_fall",
      "tracking_rejection",
    )
    for name in boolean_fields:
      value = payload[name]
      if not isinstance(value, (bool, np.bool_)):
        raise StorageError(
          f"terminal evidence field {name!r} must be bool, got {type(value).__name__}"
        )
    reason = payload["reason"]
    if reason is not None and not isinstance(reason, str):
      raise StorageError(
        "terminal evidence field 'reason' must be string or null, "
        f"got {type(reason).__name__}"
      )
    return cls(
      bool(payload["available"]),
      bool(payload["terminated"]),
      bool(payload["truncated"]),
      reason,
      bool(payload["physical_fall"]),
      bool(payload["tracking_rejection"]),
    )


def validate_row_header(
  *,
  run_id: str,
  env_id: int,
  episode_id: str,
  segment_id: str,
  tick: int,
  motion_id: str,
  controller: str,
  timestamp: float,
  fps: float,
  provenance: Mapping[str, str],
) -> None:
  """Validate the scalar header of one row.

  :class:`TrajectoryRow` and the metadata-only readers both call this, so a row
  that a bounded offline path accepts is a row the row reader accepts.
  """
  if not run_id or not episode_id or not segment_id or not motion_id:
    raise StorageError("row identities must be non-empty")
  if controller not in {"vae", "ppo"}:
    raise StorageError("controller must be vae or ppo")
  if isinstance(env_id, bool) or not isinstance(env_id, (int, np.integer)):
    raise StorageError("env_id must be a non-negative integer")
  if env_id < 0:
    raise StorageError("env_id must be a non-negative integer")
  if isinstance(tick, bool) or tick < 0:
    raise StorageError("tick is invalid")
  if not isinstance(timestamp, (int, float, np.integer, np.floating)):
    raise StorageError("timestamp is required physical data")
  if not np.isfinite(float(timestamp)):
    raise StorageError("timestamp must be finite physical data")
  if not np.isfinite(fps) or fps <= 0:
    raise StorageError("fps must be finite and positive")
  if not all(isinstance(k, str) and isinstance(v, str) for k, v in provenance.items()):
    raise StorageError("provenance must be a string mapping")


@dataclass(frozen=True, slots=True)
class RowMetadata:
  """One stored row's validated metadata, without its state or action arrays."""

  run_id: str
  env_id: int
  episode_id: str
  segment_id: str
  tick: int
  timestamp: float
  motion_id: str
  reference_frame: int
  controller: str
  terminal_evidence: TerminalEvidence | None
  segment_qualified: bool
  reset: bool
  teleport: bool
  reference_boundary: bool
  fps: float
  group_key: str
  provenance: Mapping[str, str]

  @property
  def identity(self) -> tuple[str, int, str, str, int]:
    return self.run_id, self.env_id, self.episode_id, self.segment_id, self.tick


def parse_row_metadata(text: str) -> RowMetadata:
  """Decode and validate one stored row metadata string.

  This is the metadata contract of a stored row.  The row reader and the
  metadata-only readers both use it, so both reject the same malformed rows.
  """
  try:
    fields = json.loads(text)
  except json.JSONDecodeError as exc:
    raise StorageError("row metadata is not valid JSON") from exc
  if not isinstance(fields, dict) or set(fields) != _ROW_METADATA_FIELDS:
    raise StorageError("malformed row metadata")
  if not isinstance(fields["provenance"], dict):
    raise StorageError("row provenance must be a mapping")
  if not all(isinstance(fields[name], str) for name in _STRING_METADATA_FIELDS):
    raise StorageError("row identity metadata has invalid types")
  if (
    isinstance(fields["env_id"], bool)
    or not isinstance(fields["env_id"], int)
    or isinstance(fields["tick"], bool)
    or not isinstance(fields["tick"], int)
    or isinstance(fields["reference_frame"], bool)
    or not isinstance(fields["reference_frame"], int)
    or isinstance(fields["timestamp"], bool)
    or not isinstance(fields["timestamp"], (int, float))
    or isinstance(fields["fps"], bool)
    or not isinstance(fields["fps"], (int, float))
  ):
    raise StorageError("row numeric metadata has invalid types")
  if not all(isinstance(fields[name], bool) for name in _FLAG_METADATA_FIELDS):
    raise StorageError("row flag metadata has invalid types")
  evidence = fields["terminal_evidence"]
  if evidence is not None and not isinstance(evidence, dict):
    raise StorageError("terminal evidence must be a mapping or null")
  validate_row_header(
    run_id=fields["run_id"],
    env_id=fields["env_id"],
    episode_id=fields["episode_id"],
    segment_id=fields["segment_id"],
    tick=fields["tick"],
    motion_id=fields["motion_id"],
    controller=fields["controller"],
    timestamp=fields["timestamp"],
    fps=fields["fps"],
    provenance=fields["provenance"],
  )
  return RowMetadata(
    fields["run_id"],
    fields["env_id"],
    fields["episode_id"],
    fields["segment_id"],
    fields["tick"],
    float(fields["timestamp"]),
    fields["motion_id"],
    fields["reference_frame"],
    fields["controller"],
    None if evidence is None else TerminalEvidence.from_dict(evidence),
    fields["segment_qualified"],
    fields["reset"],
    fields["teleport"],
    fields["reference_boundary"],
    float(fields["fps"]),
    fields["group_key"],
    fields["provenance"],
  )


@dataclass(frozen=True, slots=True)
class TrajectoryRow:
  """One pre-action row, with ownership and qualification metadata.

  VAE rows must carry the actual encoder latent.  PPO recovery rows are
  representable for replay/audit but explicitly carry ``latent=None`` and can
  never be accepted by the window index.
  """

  run_id: str
  env_id: int
  episode_id: str
  segment_id: str
  tick: int
  timestamp: float
  motion_id: str
  reference_frame: int
  controller: str
  state: WorldState
  previous_executed_action: np.ndarray
  clean_action: np.ndarray
  ou_noise: np.ndarray
  executed_action: np.ndarray
  latent: np.ndarray | None
  terminal_evidence: TerminalEvidence | None
  segment_qualified: bool
  reset: bool = False
  teleport: bool = False
  reference_boundary: bool = False
  fps: float = 50.0
  group_key: str = ""
  provenance: Mapping[str, str] = field(default_factory=dict)

  def __post_init__(self) -> None:
    c = DEFAULT_CONTRACT
    validate_row_header(
      run_id=self.run_id,
      env_id=self.env_id,
      episode_id=self.episode_id,
      segment_id=self.segment_id,
      tick=self.tick,
      motion_id=self.motion_id,
      controller=self.controller,
      timestamp=self.timestamp,
      fps=self.fps,
      provenance=self.provenance,
    )
    object.__setattr__(self, "timestamp", float(self.timestamp))
    for name, value, width in (
      ("previous_executed_action", self.previous_executed_action, c.action_dimension),
      ("clean_action", self.clean_action, c.action_dimension),
      ("ou_noise", self.ou_noise, c.action_dimension),
      ("executed_action", self.executed_action, c.action_dimension),
    ):
      array = np.asarray(value, dtype=np.float64)
      if array.shape != (width,) or not np.isfinite(array).all():
        raise StorageError(f"{name} must be finite with shape [{width}]")
      object.__setattr__(self, name, array)
    if self.latent is not None:
      latent = np.asarray(self.latent, dtype=np.float64)
      if latent.shape != (c.latent_dimension,) or not np.isfinite(latent).all():
        raise StorageError("latent must be finite with shape [32]")
      object.__setattr__(self, "latent", latent)
    if self.controller == "vae" and self.latent is None:
      raise StorageError("VAE-owned rows require an actual executed latent")
    if self.controller == "ppo" and self.latent is not None:
      raise StorageError("PPO recovery rows must not invent a VAE latent")

  @property
  def identity(self) -> tuple[str, int, str, str, int]:
    return self.run_id, self.env_id, self.episode_id, self.segment_id, self.tick


def validate_monotonic_timestamps(rows: Iterable[TrajectoryRow]) -> None:
  """Require finite, strictly increasing physical timestamps in row order.

  The values are supplied by the runtime; this helper deliberately does not
  require nominal control-period spacing.
  """
  values = tuple(rows)
  if any(not isinstance(row, TrajectoryRow) for row in values):
    raise StorageError("timestamp validation accepts TrajectoryRow values")
  for previous, current in zip(values, values[1:], strict=False):
    if not current.timestamp > previous.timestamp:
      raise StorageError("timestamps must be strictly increasing physical data")


VaeRow = TrajectoryRow


def sha256_file(path: str | Path) -> str:
  """SHA-256 of a file, streamed in bounded blocks."""
  digest = hashlib.sha256()
  with Path(path).open("rb") as handle:
    while chunk := handle.read(_BLOCK_BYTES):
      digest.update(chunk)
  return digest.hexdigest()


def contract_artifact_identities(
  contract: DiffusionContract,
) -> tuple[tuple[str, str, str], ...]:
  """The pinned artifact identities a manifest records for a contract."""
  return (
    ("vae", contract.vae_checkpoint, contract.vae_sha256),
    ("recovery", contract.recovery_onnx, contract.recovery_sha256),
    ("x2_xml", contract.state_xml, contract.state_xml_sha256),
    (
      "physical_fall_source",
      contract.physical_fall_source,
      contract.physical_fall_source_sha256,
    ),
    (
      "vae_tracking_rejection_source",
      contract.vae_tracking_rejection_source,
      contract.vae_tracking_rejection_source_sha256,
    ),
  )


def read_store_manifest(
  root: str | Path, *, contract: DiffusionContract = DEFAULT_CONTRACT
) -> dict[str, Any]:
  """Read and schema-check a store manifest without hashing the shards."""
  try:
    payload = json.loads((Path(root) / MANIFEST_NAME).read_text())
  except (OSError, json.JSONDecodeError) as exc:
    raise StorageError("manifest is unreadable") from exc
  if not isinstance(payload, dict) or payload.get("format") != STORE_FORMAT:
    raise StorageError("manifest schema mismatch")
  manifest_contract = payload.get("contract")
  manifest_schema = (
    manifest_contract.get("schema_version")
    if isinstance(manifest_contract, dict)
    else None
  )
  if manifest_schema != contract.schema_version or not isinstance(
    payload.get("shards"), list
  ):
    raise StorageError("manifest schema mismatch")
  return payload


def write_store_manifest(
  root: str | Path,
  *,
  contract: DiffusionContract,
  max_rows_per_shard: int,
  shards: Sequence[Mapping[str, Any]],
  merged_from: Sequence[Mapping[str, Any]] | None = None,
) -> None:
  """Write a store manifest atomically.

  This is the only writer of the manifest schema.  A byte-preserving merge calls
  it for shards it copied rather than wrote; ``merged_from`` records the input
  stores of such a merge and is omitted for an ordinary store.
  """
  payload: dict[str, Any] = {
    "format": STORE_FORMAT,
    "contract_hash": contract.identity_hash(),
    "contract": contract.as_dict(),
    "artifact_identities": [
      {"role": role, "path": path, "sha256": digest}
      for role, path, digest in contract_artifact_identities(contract)
    ],
    "max_rows_per_shard": max_rows_per_shard,
    "shards": [dict(item) for item in shards],
  }
  if merged_from:
    payload["merged_from"] = [dict(item) for item in merged_from]
  root = Path(root)
  temp = root / ".manifest.tmp"
  temp.write_text(json.dumps(payload, sort_keys=True, indent=2))
  os.replace(temp, root / MANIFEST_NAME)


def manifest_sha256(root: str | Path) -> str:
  """SHA-256 of a store's manifest file."""
  return sha256_file(Path(root) / MANIFEST_NAME)


def _is_safe_shard_name(name: str) -> bool:
  """A manifest may only name a plain file inside the store root."""
  if not name or name in {".", ".."} or "\x00" in name:
    return False
  if "/" in name or "\\" in name:
    return False
  return name == Path(name).name


def _is_sha256(value: Any) -> TypeGuard[str]:
  return (
    isinstance(value, str)
    and len(value) == 64
    and all(character in "0123456789abcdef" for character in value)
  )


def _manifest_identity(value: Any) -> tuple[str, int, str, str, int]:
  if not isinstance(value, (list, tuple)) or len(value) != 5:
    raise StorageError("malformed shard manifest identity")
  run_id, env_id, episode_id, segment_id, tick = value
  if not all(isinstance(part, str) for part in (run_id, episode_id, segment_id)):
    raise StorageError("malformed shard manifest identity")
  if (
    isinstance(env_id, bool)
    or not isinstance(env_id, int)
    or isinstance(tick, bool)
    or not isinstance(tick, int)
  ):
    raise StorageError("malformed shard manifest identity")
  return run_id, env_id, episode_id, segment_id, tick


@dataclass(frozen=True, slots=True)
class ShardEntry:
  """One validated shard manifest entry.

  ``first`` and ``last`` are the boundary identities the manifest declares for
  the shard; every store this module writes records them.
  """

  file: str
  rows: int
  sha256: str
  first: tuple[str, int, str, str, int]
  last: tuple[str, int, str, str, int]

  @classmethod
  def from_manifest(cls, item: Mapping[str, Any]) -> "ShardEntry":
    if not isinstance(item, Mapping):
      raise StorageError("malformed shard manifest entry")
    filename = item.get("file")
    rows = item.get("rows")
    digest = item.get("sha256")
    if (
      not isinstance(filename, str)
      or not _is_safe_shard_name(filename)
      or isinstance(rows, bool)
      or not isinstance(rows, int)
      or rows <= 0
      or not _is_sha256(digest)
    ):
      raise StorageError("malformed shard manifest entry")
    return cls(
      filename,
      rows,
      digest,
      _manifest_identity(item.get("first")),
      _manifest_identity(item.get("last")),
    )

  def as_manifest_entry(self) -> dict[str, Any]:
    return {
      "file": self.file,
      "rows": self.rows,
      "sha256": self.sha256,
      "first": list(self.first),
      "last": list(self.last),
    }


def parse_shard_entries(payload: Mapping[str, Any]) -> tuple[ShardEntry, ...]:
  """Validate a manifest's shard list and reject a repeated shard file."""
  raw = payload.get("shards")
  if not isinstance(raw, list):
    raise StorageError("manifest schema mismatch")
  entries = tuple(ShardEntry.from_manifest(item) for item in raw)
  names = [entry.file for entry in entries]
  if len(set(names)) != len(names):
    raise StorageError("manifest lists a shard file more than once")
  return entries


def shard_path(root: str | Path, entry: ShardEntry) -> Path:
  """Resolve a validated entry to its path inside the store root."""
  if not _is_safe_shard_name(entry.file):
    raise StorageError("unsafe shard file name")
  return Path(root) / entry.file


def shard_array_shapes(
  contract: DiffusionContract, rows: int
) -> dict[str, tuple[int, ...]]:
  """Row-aligned shapes of every array a stored shard carries."""
  return {
    "root_position": (rows, 3),
    "root_quaternion_wxyz": (rows, 4),
    "root_linear_velocity": (rows, 3),
    "root_angular_velocity": (rows, 3),
    "body_positions": (rows, contract.body_count, 3),
    "body_linear_velocities": (rows, contract.body_count, 3),
    "previous_executed_action": (rows, contract.action_dimension),
    "clean_action": (rows, contract.action_dimension),
    "ou_noise": (rows, contract.action_dimension),
    "executed_action": (rows, contract.action_dimension),
    "latent": (rows, contract.latent_dimension),
    "latent_present": (rows,),
    "metadata": (rows,),
  }


def _npy_header(head: bytes, name: str) -> tuple[tuple[int, ...], np.dtype, bool]:
  stream = io.BytesIO(head)
  try:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
      shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
      shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
    else:  # pragma: no cover - numpy writes version 1 or 2
      raise StorageError(f"shard array {name!r} uses an unsupported npy version")
  except StorageError:
    raise
  except Exception as exc:
    raise StorageError(f"shard array {name!r} has an unreadable header") from exc
  return tuple(int(value) for value in shape), np.dtype(dtype), bool(fortran_order)


def _header_dtype_is_valid(name: str, dtype: np.dtype) -> bool:
  if name == "latent_present":
    return dtype == np.dtype(np.uint8)
  if name == "metadata":
    return dtype.kind in {"U", "S"}
  return dtype == np.dtype(np.float64)


def verify_shard_layout(
  root: str | Path, contract: DiffusionContract, entry: ShardEntry
) -> None:
  """Check every array header of a shard without decompressing row data.

  The manifest alone does not prove a row count: the shard's own array headers
  must agree with it.  A header probe is a few hundred bytes per array.
  """
  path = shard_path(root, entry)
  expected = shard_array_shapes(contract, entry.rows)
  # ``np.savez`` stores every array as an ``.npy`` member of the zip container.
  members = {f"{name}.npy": name for name in expected}
  try:
    with zipfile.ZipFile(path) as archive:
      names = set(archive.namelist())
      missing = sorted(members[member] for member in set(members) - names)
      if missing:
        raise StorageError(f"shard {entry.file} is missing arrays {missing}")
      unexpected = sorted(names - set(members))
      if unexpected:
        raise StorageError(f"shard {entry.file} has unexpected arrays {unexpected}")
      for member, name in members.items():
        with archive.open(member) as handle:
          head = handle.read(_HEADER_PROBE_BYTES)
        actual_shape, dtype, fortran_order = _npy_header(head, name)
        if actual_shape != expected[name] or not _header_dtype_is_valid(name, dtype):
          raise StorageError(f"shard array {name!r} has an invalid header")
        if fortran_order:
          raise StorageError(f"shard array {name!r} must be C-contiguous")
  except StorageError:
    raise
  except Exception as exc:
    raise StorageError(f"invalid shard {path}") from exc


def load_shard_arrays(
  root: str | Path, contract: DiffusionContract, entry: ShardEntry
) -> dict[str, Any]:
  """Hash-check a shard and materialise each of its arrays exactly once.

  ``NpzFile.__getitem__`` re-decompresses the whole array on every access, so
  every array is fetched once here and reused by the caller.
  """
  path = shard_path(root, entry)
  try:
    if sha256_file(path) != entry.sha256:
      raise StorageError(f"corrupt shard hash: {entry.file}")
    with np.load(path, allow_pickle=False) as data:
      arrays: dict[str, Any] = {}
      for name, shape in shard_array_shapes(contract, entry.rows).items():
        array = data[name]
        arrays[name] = array
        if array.shape != shape:
          raise StorageError(f"shard array {name!r} has invalid shape")
        if name == "latent_present":
          if array.dtype != np.dtype(np.uint8) or not np.isin(array, (0, 1)).all():
            raise StorageError("latent_present must be uint8 flags")
        elif name == "metadata":
          if array.dtype.kind not in {"U", "S"}:
            raise StorageError("metadata must be a string array")
        elif array.dtype != np.dtype(np.float64):
          raise StorageError(f"shard array {name!r} must be float64")
        elif not np.isfinite(array).all():
          raise StorageError(f"shard array {name!r} contains non-finite values")
      return arrays
  except StorageError:
    raise
  except Exception as exc:
    # ``np.load`` may defer malformed ZIP/container decoding until an array
    # is accessed; keep every ordinary decode failure at the storage API.
    raise StorageError(f"invalid shard {path}") from exc


def iter_row_metadata(
  root: str | Path, contract: DiffusionContract, entry: ShardEntry
) -> Iterator[RowMetadata]:
  """Stream validated per-row metadata for one shard.

  Only the metadata column and the one-byte latent-presence column are
  decompressed; the state and action arrays are never touched.  The manifest row
  count is checked against the metadata length, so a forged count fails here.
  """
  path = shard_path(root, entry)
  try:
    with np.load(path, allow_pickle=False) as data:
      metadata = data["metadata"]
      flags = data["latent_present"]
      if metadata.shape != (entry.rows,):
        raise StorageError(f"shard {entry.file} metadata rows differ from the manifest")
      if metadata.dtype.kind not in {"U", "S"}:
        raise StorageError("metadata must be a string array")
      if (
        flags.shape != (entry.rows,)
        or flags.dtype != np.dtype(np.uint8)
        or not np.isin(flags, (0, 1)).all()
      ):
        raise StorageError("latent_present must be uint8 flags")
      for text in metadata:
        if not isinstance(text, str):
          raise StorageError("metadata entry must be a string")
        yield parse_row_metadata(text)
  except StorageError:
    raise
  except Exception as exc:
    raise StorageError(f"invalid shard {path}") from exc


@dataclass(frozen=True, slots=True)
class ShardIdentities:
  """Metadata-only reduction for one shard: counts, boundaries and digests."""

  rows: int
  first: tuple[str, int, str, str, int]
  last: tuple[str, int, str, str, int]
  first_timestamp: float
  last_timestamp: float
  digests: np.ndarray


def read_shard_identities(
  root: str | Path, contract: DiffusionContract, entry: ShardEntry
) -> ShardIdentities:
  """Read one shard's identities without decoding state or action arrays."""
  verify_shard_layout(root, contract, entry)
  first: tuple[str, int, str, str, int] | None = None
  last: tuple[str, int, str, str, int] | None = None
  first_timestamp = last_timestamp = 0.0
  rows = 0
  digests: list[bytes] = []
  for row in iter_row_metadata(root, contract, entry):
    identity = row.identity
    if first is None:
      first, first_timestamp = identity, row.timestamp
    last, last_timestamp = identity, row.timestamp
    rows += 1
    digests.append(identity_digest(identity))
  if first is None or last is None or rows != entry.rows:
    raise StorageError(f"shard {entry.file} metadata rows differ from the manifest")
  if first != entry.first or last != entry.last:
    raise StorageError(
      f"shard {entry.file} boundary identities differ from the manifest"
    )
  return ShardIdentities(
    rows, first, last, first_timestamp, last_timestamp, digest_matrix(digests)
  )


def identity_digest(identity: Sequence[Any]) -> bytes:
  """128-bit digest of one row identity.

  Identity comparison in a bounded reader goes through this digest, so a worker
  never transports a whole identity set.  A digest collision can only make a
  caller refuse work, never accept a duplicate row.
  """
  payload = [part if isinstance(part, str) else int(part) for part in identity]
  encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
  return hashlib.blake2b(encoded, digest_size=_IDENTITY_DIGEST_BYTES).digest()


def digest_matrix(digests: Sequence[bytes]) -> np.ndarray:
  """Stack fixed-width identity digests into an ``(n, 2)`` uint64 matrix."""
  for digest in digests:
    if len(digest) != _IDENTITY_DIGEST_BYTES:
      raise StorageError("identity digests must be 16 bytes")
  if not digests:
    return np.empty((0, 2), dtype=np.uint64)
  raw = np.frombuffer(b"".join(digests), dtype=">u8").reshape(-1, 2)
  return raw.astype(np.uint64)


@dataclass(frozen=True, slots=True)
class ShardStoreSource:
  """One store's root, contract and validated shard entries.

  The manifest is read and its entries are validated once; shard contents are
  verified per shard as they are read, so opening a source does not hash the
  whole store.
  """

  root: Path
  contract: DiffusionContract
  entries: tuple[ShardEntry, ...]

  @classmethod
  def open(
    cls, root: str | Path, *, contract: DiffusionContract = DEFAULT_CONTRACT
  ) -> "ShardStoreSource":
    payload = read_store_manifest(root, contract=contract)
    return cls(Path(root), contract, parse_shard_entries(payload))

  @property
  def rows(self) -> int:
    """Rows the manifest declares; every shard read verifies the real count."""
    return sum(entry.rows for entry in self.entries)


def shard_partitions(entries: Sequence[ShardEntry], jobs: int) -> list[tuple[int, int]]:
  """Contiguous shard index ranges, balanced by the manifest's row counts.

  ``dataset.shard_partitions`` builds the same ranges for the parallel index
  builder; this variant takes validated entries so that a bounded offline
  worker needs no index module.
  """
  if jobs < 1:
    raise StorageError("jobs must be one or more")
  counts = [entry.rows for entry in entries]
  if not counts:
    return []
  target = max(1, -(-sum(counts) // jobs))
  parts: list[tuple[int, int]] = []
  start = 0
  weight = 0
  for index, count in enumerate(counts):
    weight += count
    if weight >= target and len(parts) < jobs - 1:
      parts.append((start, index + 1))
      start = index + 1
      weight = 0
  if start < len(counts):
    parts.append((start, len(counts)))
  return parts


def duplicate_row_identity(
  sources: Sequence[ShardStoreSource],
) -> tuple[str, int, str, str, int] | None:
  """Find the first repeated row identity across the given stores.

  This is the exact diagnostic counterpart to the digest comparison.  It reads
  metadata only and holds identities, not rows, so it runs after a digest set
  has already reported a repeat.
  """
  seen: set[tuple[str, int, str, str, int]] = set()
  for source in sources:
    for entry in source.entries:
      for row in iter_row_metadata(source.root, source.contract, entry):
        identity = row.identity
        if identity in seen:
          return identity
        seen.add(identity)
  return None


def _copy_verified_shard(source: Path, destination: Path, expected_sha256: str) -> None:
  """Copy one shard byte for byte and prove it matches the source manifest.

  Sources are copied, never hard-linked: a hard link would let a later write to
  a source store change the merged store.  The hash is computed while copying,
  so a shard that changed after collection is refused.
  """
  digest = hashlib.sha256()
  with source.open("rb") as reader, destination.open("wb") as writer:
    while chunk := reader.read(_BLOCK_BYTES):
      digest.update(chunk)
      writer.write(chunk)
  if digest.hexdigest() != expected_sha256:
    raise StorageError(f"shard {source.name} does not match its manifest hash")


def _manifest_shard_bound(payload: Mapping[str, Any], root: Path) -> int:
  """The shard write bound an input store declares; 0 when it records none."""
  value = payload.get("max_rows_per_shard")
  if value is None:
    return 0
  if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
    raise StorageError(f"input store {root} records an invalid shard bound")
  return value


def _artifact_identity_key(items: Any) -> tuple[tuple[str, str, str], ...]:
  if not isinstance(items, list):
    raise StorageError("malformed artifact identities")
  key: list[tuple[str, str, str]] = []
  for item in items:
    if not isinstance(item, Mapping):
      raise StorageError("malformed artifact identities")
    role, path, digest = item.get("role"), item.get("path"), item.get("sha256")
    if not (
      isinstance(role, str) and isinstance(path, str) and isinstance(digest, str)
    ):
      raise StorageError("malformed artifact identities")
    key.append((role, path, digest))
  return tuple(sorted(key))


def _require_manifest_identity(
  payload: Mapping[str, Any], contract: DiffusionContract, root: Path
) -> None:
  """Refuse a store that was written under another semantic contract.

  The stored contract fields and the pinned artifact identities are compared
  against the active contract.  ``contract_hash`` itself is not compared: it is a
  descriptive digest, no reader uses it, and stores written before the contract
  identity stopped counting the contract file digest record a digest-inclusive
  value that the current ``identity_hash`` cannot reproduce.
  """
  stored = payload.get("contract")
  if not isinstance(stored, Mapping):
    raise StorageError(f"input store {root} records no contract fields")
  fields = {key: value for key, value in stored.items() if key != "contract_sha256"}
  active = contract.as_dict()
  active.pop("contract_sha256", None)
  if fields != active:
    raise StorageError(f"input store {root} was written under another contract")
  expected = _artifact_identity_key(
    [
      {"role": role, "path": path, "sha256": digest}
      for role, path, digest in contract_artifact_identities(contract)
    ]
  )
  if _artifact_identity_key(payload.get("artifact_identities")) != expected:
    raise StorageError(f"input store {root} records different pinned artifacts")


@dataclass(frozen=True, slots=True)
class MergeInputReport:
  """Rows and manifest identity of one merged input store."""

  root: str
  rows: int
  shards: int
  manifest_sha256: str


@dataclass(frozen=True, slots=True)
class MergeReport:
  """Result of a store merge, in the shape the merge command reports."""

  mode: str
  inputs: tuple[MergeInputReport, ...]
  input_rows: int
  output: str
  rows_before: int
  rows_after: int
  shards: int
  max_rows_per_shard: int
  manifest_sha256: str

  def as_payload(self) -> dict[str, Any]:
    return {
      "inputs": {report.root: report.rows for report in self.inputs},
      "input_rows": self.input_rows,
      "output": self.output,
      "rows_before": self.rows_before,
      "rows_after": self.rows_after,
      "shards": self.shards,
      "mode": self.mode,
      "max_rows_per_shard": self.max_rows_per_shard,
      "output_manifest_sha256": self.manifest_sha256,
    }


def _accounting_failure(payload: Mapping[str, Any]) -> StorageError:
  return StorageError(f"merge row accounting failed: {json.dumps(payload)}")


def merge_shard_stores(
  inputs: Sequence[str | Path],
  output: str | Path,
  *,
  contract: DiffusionContract = DEFAULT_CONTRACT,
  max_rows_per_shard: int | None = None,
) -> MergeReport:
  """Copy validated worker stores into one new store without re-encoding rows.

  Every source shard is hash-checked and then copied byte for byte, so the
  merged store holds the source shard payloads unchanged.  Row identities come
  from the metadata column only: the state and action arrays are never decoded
  and never recompressed.  The merged manifest lists the copied shards under new
  sequential names with the verified row counts, the verified first and last
  identities and the source hashes.

  The output is staged in a sibling directory and published with one atomic
  rename, and an existing output path is refused.  An input manifest that names
  a repeated file, an unsafe file name, an unexpected shard layout, a forged row
  count, a duplicate identity or another contract identity is refused before
  anything is published.  Sources are opened read-only and never linked, so a
  merge cannot change them.
  """
  roots = tuple(Path(path) for path in inputs)
  if not roots:
    raise StorageError("merge requires at least one input store")
  destination = Path(output)
  if destination.exists():
    raise StorageError(f"merge refuses an existing output: {destination}")
  resolved = [str(root.resolve()) for root in roots]
  if len(set(resolved)) != len(resolved):
    raise StorageError("merge inputs repeat the same store")

  sources: list[ShardStoreSource] = []
  declared_bounds: list[int] = []
  for root in roots:
    payload = read_store_manifest(root, contract=contract)
    _require_manifest_identity(payload, contract, root)
    declared_bounds.append(_manifest_shard_bound(payload, root))
    sources.append(ShardStoreSource(root, contract, parse_shard_entries(payload)))

  manifest_hashes = [manifest_sha256(source.root) for source in sources]
  per_source: list[list[tuple[ShardEntry, ShardIdentities]]] = []
  digest_parts: list[np.ndarray] = []
  for source in sources:
    reduced = [
      (entry, read_shard_identities(source.root, source.contract, entry))
      for entry in source.entries
    ]
    per_source.append(reduced)
    digest_parts.extend(identities.digests for _, identities in reduced)

  all_digests = (
    np.concatenate(digest_parts) if digest_parts else np.empty((0, 2), np.uint64)
  )
  if len(all_digests) and len(np.unique(all_digests, axis=0)) != len(all_digests):
    duplicate = duplicate_row_identity(sources)
    if duplicate is None:
      raise StorageError("identity digest collision: refusing to merge")
    raise StorageError(f"duplicate row identity {duplicate}")

  input_reports = tuple(
    MergeInputReport(
      str(source.root),
      sum(identities.rows for _, identities in reduced),
      len(reduced),
      manifest_hash,
    )
    for source, reduced, manifest_hash in zip(
      sources, per_source, manifest_hashes, strict=True
    )
  )
  input_rows = sum(report.rows for report in input_reports)

  largest_shard = max(
    (entry.rows for source in sources for entry in source.entries), default=0
  )
  if max_rows_per_shard is None:
    # Keep the collection convention of the inputs: a store commonly declares a
    # bound (for example 4096) while its shards hold fewer rows, so neither the
    # observed shard size nor the declared bound alone describes the output.
    bound = max(largest_shard, max(declared_bounds), 1)
  else:
    if max_rows_per_shard <= 0:
      raise StorageError("max_rows_per_shard must be positive")
    if largest_shard > max_rows_per_shard:
      raise StorageError(
        f"a source shard holds {largest_shard} rows: the byte-preserving merge "
        f"cannot honour max_rows_per_shard={max_rows_per_shard}"
      )
    bound = max_rows_per_shard

  destination.parent.mkdir(parents=True, exist_ok=True)
  staging = Path(
    tempfile.mkdtemp(prefix=f".{destination.name}-merge-", dir=destination.parent)
  )
  published = False
  try:
    staged: list[ShardEntry] = []
    for source, reduced in zip(sources, per_source, strict=True):
      for entry, identities in reduced:
        name = f"shard-{len(staged):06d}.npz"
        _copy_verified_shard(
          shard_path(source.root, entry), staging / name, entry.sha256
        )
        staged.append(
          ShardEntry(
            name, identities.rows, entry.sha256, identities.first, identities.last
          )
        )
    write_store_manifest(
      staging,
      contract=contract,
      max_rows_per_shard=bound,
      shards=[entry.as_manifest_entry() for entry in staged],
      merged_from=[
        {
          "root": report.root,
          "manifest_sha256": report.manifest_sha256,
          "rows": report.rows,
          "shards": report.shards,
        }
        for report in input_reports
      ],
    )
    for source, manifest_hash in zip(sources, manifest_hashes, strict=True):
      if manifest_sha256(source.root) != manifest_hash:
        raise StorageError(f"source store changed during the merge: {source.root}")
    # Validate the staged store through the normal reader, which re-hashes every
    # copied shard, before the rename publishes it.
    validated = AppendOnlyShardStore(staging, contract=contract)
    if validated.row_count != input_rows or validated.shard_count != len(staged):
      raise _accounting_failure(
        {
          "inputs": {report.root: report.rows for report in input_reports},
          "input_rows": input_rows,
          "output": str(destination),
          "rows_before": 0,
          "rows_after": validated.row_count,
          "shards": validated.shard_count,
          "mode": "shard-copy",
        }
      )
    try:
      os.replace(staging, destination)
    except OSError as exc:
      raise StorageError(f"could not publish the merged store {destination}") from exc
    published = True
  finally:
    if not published:
      shutil.rmtree(staging, ignore_errors=True)

  return MergeReport(
    mode="shard-copy",
    inputs=input_reports,
    input_rows=input_rows,
    output=str(destination),
    rows_before=0,
    rows_after=input_rows,
    shards=len(staged),
    max_rows_per_shard=bound,
    manifest_sha256=manifest_sha256(destination),
  )


def merge_shard_stores_legacy(
  inputs: Sequence[str | Path],
  output: str | Path,
  *,
  contract: DiffusionContract = DEFAULT_CONTRACT,
  max_rows_per_shard: int = 4096,
  batch_rows: int = 4096,
) -> MergeReport:
  """Merge stores by rewriting every row through the store writer.

  This is the historical row-level reference path.  It decodes and recompresses
  every row, and it is kept selectable so the byte-preserving merge can be
  compared against it.  It appends to an existing store, and refuses the result
  if the row accounting disagrees.
  """
  if batch_rows <= 0:
    raise StorageError("batch_rows must be positive")
  destination = Path(output)
  destination.mkdir(parents=True, exist_ok=True)
  output_store = AppendOnlyShardStore(
    destination, contract=contract, max_rows_per_shard=max_rows_per_shard
  )
  before = output_store.row_count
  reports: list[MergeInputReport] = []
  for path in inputs:
    root = Path(path)
    payload = read_store_manifest(root, contract=contract)
    _require_manifest_identity(payload, contract, root)
    source = AppendOnlyShardStore(root, contract=contract)
    rows = 0
    batch: list[TrajectoryRow] = []
    for row in source.iter_rows():
      batch.append(row)
      if len(batch) >= batch_rows:
        output_store.append(batch)
        rows += len(batch)
        batch = []
    if batch:
      output_store.append(batch)
      rows += len(batch)
    reports.append(
      MergeInputReport(str(root), rows, source.shard_count, manifest_sha256(root))
    )

  input_reports = tuple(reports)
  input_rows = sum(report.rows for report in input_reports)
  report = MergeReport(
    mode="row-merge",
    inputs=input_reports,
    input_rows=input_rows,
    output=str(destination),
    rows_before=before,
    rows_after=output_store.row_count,
    shards=output_store.shard_count,
    max_rows_per_shard=max_rows_per_shard,
    manifest_sha256=manifest_sha256(destination),
  )
  payload = report.as_payload()
  if before != 0 or report.rows_after != input_rows:
    raise _accounting_failure(payload)
  return report


class AppendOnlyShardStore:
  """Durable, hash-checked row shards with bounded per-shard capacity.

  ``validate_shards`` selects how much integrity work construction performs.
  ``True`` (the default) also hashes every shard file of an existing store, so a
  shard that is corrupt on disk is refused before it can be read.  ``False``
  validates only the manifest metadata (shard names, row counts, boundary
  identities and recorded hashes) and defers each shard's byte check to the read
  that consumes it: every read still runs the full hash, shape, dtype, finite
  and row checks.  ``False`` therefore suits a partitioned reader that opens the
  whole store but reads only its own shards; it does not weaken a read, and a
  shard that no caller reads stays unverified.
  """

  FORMAT = STORE_FORMAT

  def __init__(
    self,
    root: str | Path,
    *,
    contract: DiffusionContract = DEFAULT_CONTRACT,
    max_rows_per_shard: int = 4096,
    max_total_rows: int | None = None,
    validate_shards: bool = True,
  ) -> None:
    if max_rows_per_shard <= 0 or (max_total_rows is not None and max_total_rows <= 0):
      raise StorageError("shard bounds must be positive")
    self.root = Path(root)
    self.contract = contract
    self.max_rows_per_shard = max_rows_per_shard
    self.max_total_rows = max_total_rows
    self.root.mkdir(parents=True, exist_ok=True)
    manifest = self.root / MANIFEST_NAME
    if manifest.exists():
      if validate_shards:
        self._validate_manifest()
      else:
        # Metadata-only check: the shard list, names, counts, boundaries and
        # recorded hashes are validated, while the shard bytes are verified by
        # each read instead of by one whole-store scan per store object.
        parse_shard_entries(self._manifest())
    else:
      self._write_manifest([])
    # The set of written identities is cached because the previous append re-read,
    # re-hashed and decompressed every existing shard, which made collection
    # quadratic in the trial count (and re-read the store over the network
    # filesystem once per trial).  The manifest itself is still read from disk on
    # every access so that an on-disk rewrite is always observed.
    self._identity_cache: set[tuple[str, int, str, str, int]] | None = None

  def _manifest(self) -> dict[str, Any]:
    return read_store_manifest(self.root, contract=self.contract)

  def _write_manifest(
    self,
    shards: list[dict[str, Any]],
    *,
    merged_from: list[dict[str, Any]] | None = None,
  ) -> None:
    # A store keeps the input provenance a byte-preserving merge recorded, so a
    # later append does not silently drop it.
    if merged_from is None and (self.root / MANIFEST_NAME).is_file():
      merged_from = list(self._manifest().get("merged_from", ()))
    write_store_manifest(
      self.root,
      contract=self.contract,
      max_rows_per_shard=self.max_rows_per_shard,
      shards=shards,
      merged_from=merged_from,
    )

  def _validate_manifest(self) -> None:
    for entry in parse_shard_entries(self._manifest()):
      path = shard_path(self.root, entry)
      if not path.is_file() or sha256_file(path) != entry.sha256:
        raise StorageError(f"corrupt or missing shard {entry.file}")

  @property
  def row_count(self) -> int:
    return sum(entry.rows for entry in self.shard_entries())

  @property
  def shard_count(self) -> int:
    return len(self._manifest()["shards"])

  def shard_items(self) -> tuple[Mapping[str, Any], ...]:
    """Return the manifest's shard entries, in store order."""
    return tuple(self._manifest()["shards"])

  def shard_entries(self) -> tuple[ShardEntry, ...]:
    """Return the manifest's validated shard entries, in store order."""
    return parse_shard_entries(self._manifest())

  def iter_rows_range(self, start: int, stop: int) -> Iterator[TrajectoryRow]:
    """Yield rows from shard indices ``[start, stop)`` only.

    A partitioned reader must open only its own shards, otherwise every worker
    pays the full store read and the partition buys nothing.
    """
    items = self._manifest()["shards"]
    if start < 0 or stop > len(items) or start > stop:
      raise StorageError("shard range is out of bounds")
    for item in items[start:stop]:
      yield from self._read_shard(item)

  def _known_identities(self) -> set[tuple[str, int, str, str, int]]:
    """Identities already written, seeded once from the existing shards.

    The first call reads every shard (the store has no identity index on disk);
    every later call is incremental, so ``append`` is O(rows) rather than
    O(rows + whole store).
    """
    if self._identity_cache is None:
      known: set[tuple[str, int, str, str, int]] = set()
      for item in self._manifest()["shards"]:
        for row in self._read_shard(item):
          known.add(row.identity)
      self._identity_cache = known
    return self._identity_cache

  def append(self, rows: Iterable[TrajectoryRow]) -> tuple[str, ...]:
    values = tuple(rows)
    if not values:
      return ()
    seen: set[tuple[str, int, str, str, int]] = set()
    for row in values:
      if not isinstance(row, TrajectoryRow):
        raise StorageError("append accepts TrajectoryRow values")
      if row.identity in seen:
        raise StorageError(f"duplicate row identity {row.identity}")
      seen.add(row.identity)
    existing = self._known_identities()
    duplicate = seen & existing
    if duplicate:
      raise StorageError(f"duplicate row identity {next(iter(duplicate))}")
    if (
      self.max_total_rows is not None
      and self.row_count + len(values) > self.max_total_rows
    ):
      raise StorageError("append would exceed bounded store capacity")
    manifest = self._manifest()
    files: list[str] = []
    for start in range(0, len(values), self.max_rows_per_shard):
      chunk = values[start : start + self.max_rows_per_shard]
      index = len(manifest["shards"])
      filename = f"shard-{index:06d}.npz"
      path = self.root / filename
      self._write_shard(path, chunk)
      digest = sha256_file(path)
      manifest["shards"].append(
        {
          "file": filename,
          "rows": len(chunk),
          "sha256": digest,
          "first": list(chunk[0].identity),
          "last": list(chunk[-1].identity),
        }
      )
      files.append(filename)
    self._write_manifest(manifest["shards"])
    if self._identity_cache is not None:
      self._identity_cache |= seen
    return tuple(files)

  def _write_shard(self, path: Path, rows: tuple[TrajectoryRow, ...]) -> None:
    states = WorldState.stack(row.state for row in rows)
    latent = np.zeros((len(rows), self.contract.latent_dimension), dtype=np.float64)
    mask = np.zeros(len(rows), dtype=np.uint8)
    for i, row in enumerate(rows):
      if row.latent is not None:
        latent[i] = row.latent
        mask[i] = 1
    metadata = np.asarray(
      [
        json.dumps(
          {
            "run_id": row.run_id,
            "env_id": row.env_id,
            "episode_id": row.episode_id,
            "segment_id": row.segment_id,
            "tick": row.tick,
            "timestamp": row.timestamp,
            "motion_id": row.motion_id,
            "reference_frame": row.reference_frame,
            "controller": row.controller,
            "terminal_evidence": None
            if row.terminal_evidence is None
            else row.terminal_evidence.as_dict(),
            "segment_qualified": row.segment_qualified,
            "reset": row.reset,
            "teleport": row.teleport,
            "reference_boundary": row.reference_boundary,
            "fps": row.fps,
            "group_key": row.group_key,
            "provenance": dict(row.provenance),
          },
          sort_keys=True,
        )
        for row in rows
      ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
      dir=path.parent, suffix=".npz", delete=False
    ) as handle:
      temp = Path(handle.name)
    try:
      np.savez_compressed(
        temp,
        root_position=states.root_position,
        root_quaternion_wxyz=states.root_quaternion_wxyz,
        root_linear_velocity=states.root_linear_velocity,
        root_angular_velocity=states.root_angular_velocity,
        body_positions=states.body_positions,
        body_linear_velocities=states.body_linear_velocities,
        previous_executed_action=np.stack(
          [row.previous_executed_action for row in rows]
        ),
        clean_action=np.stack([row.clean_action for row in rows]),
        ou_noise=np.stack([row.ou_noise for row in rows]),
        executed_action=np.stack([row.executed_action for row in rows]),
        latent=latent,
        latent_present=mask,
        metadata=metadata,
      )
      # ``np.savez`` appends ``.npz`` to names without that suffix.
      actual = temp if temp.exists() else Path(str(temp) + ".npz")
      os.replace(actual, path)
    finally:
      if temp.exists():
        temp.unlink()

  def _read_shard(self, item: Mapping[str, Any]) -> tuple[TrajectoryRow, ...]:
    try:
      entry = ShardEntry.from_manifest(item)
      arrays = load_shard_arrays(self.root, self.contract, entry)
      states = WorldState(
        arrays["root_position"],
        arrays["root_quaternion_wxyz"],
        arrays["root_linear_velocity"],
        arrays["root_angular_velocity"],
        arrays["body_positions"],
        arrays["body_linear_velocities"],
      )
      result = []
      for i, text in enumerate(arrays["metadata"]):
        if not isinstance(text, str):
          raise StorageError("metadata entry must be a string")
        row = parse_row_metadata(text)
        result.append(
          TrajectoryRow(
            run_id=row.run_id,
            env_id=row.env_id,
            episode_id=row.episode_id,
            segment_id=row.segment_id,
            tick=row.tick,
            timestamp=row.timestamp,
            motion_id=row.motion_id,
            reference_frame=row.reference_frame,
            controller=row.controller,
            state=WorldState(*(getattr(states, name)[i] for name in _STATE_FIELDS)),
            previous_executed_action=arrays["previous_executed_action"][i],
            clean_action=arrays["clean_action"][i],
            ou_noise=arrays["ou_noise"][i],
            executed_action=arrays["executed_action"][i],
            latent=arrays["latent"][i] if bool(arrays["latent_present"][i]) else None,
            terminal_evidence=row.terminal_evidence,
            segment_qualified=row.segment_qualified,
            reset=row.reset,
            teleport=row.teleport,
            reference_boundary=row.reference_boundary,
            fps=row.fps,
            group_key=row.group_key,
            provenance=row.provenance,
          )
        )
      return tuple(result)
    except StorageError:
      raise
    except Exception as exc:
      # ``np.load`` may defer malformed ZIP/container decoding until an array is
      # accessed, and world-state construction validates the physical arrays;
      # keep every ordinary decode failure at the storage API.
      raise StorageError(
        f"invalid shard {self.root / str(item.get('file', ''))}"
      ) from exc

  def iter_rows(self) -> Iterator[TrajectoryRow]:
    for item in self._manifest()["shards"]:
      yield from self._read_shard(item)

  def identities(self) -> Iterator[tuple[str, int, str, str, int]]:
    for row in self.iter_rows():
      yield row.identity

  def rows_for(
    self, identities: Iterable[tuple[str, int, str, str, int]]
  ) -> tuple[TrajectoryRow, ...]:
    wanted = set(identities)
    return tuple(row for row in self.iter_rows() if row.identity in wanted)


ShardStore = AppendOnlyShardStore
ShardWriter = AppendOnlyShardStore

__all__ = [
  "AppendOnlyShardStore",
  "MANIFEST_NAME",
  "MergeInputReport",
  "MergeReport",
  "RowMetadata",
  "STORE_FORMAT",
  "ShardEntry",
  "ShardIdentities",
  "ShardStore",
  "ShardStoreSource",
  "ShardWriter",
  "StorageError",
  "TerminalEvidence",
  "TrajectoryRow",
  "VaeRow",
  "contract_artifact_identities",
  "digest_matrix",
  "duplicate_row_identity",
  "identity_digest",
  "iter_row_metadata",
  "load_shard_arrays",
  "manifest_sha256",
  "merge_shard_stores",
  "merge_shard_stores_legacy",
  "parse_row_metadata",
  "parse_shard_entries",
  "read_shard_identities",
  "read_store_manifest",
  "sha256_file",
  "shard_array_shapes",
  "shard_partitions",
  "shard_path",
  "validate_monotonic_timestamps",
  "validate_row_header",
  "verify_shard_layout",
  "write_store_manifest",
]
