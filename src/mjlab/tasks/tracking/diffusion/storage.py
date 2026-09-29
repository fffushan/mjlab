"""Append-only bounded storage for raw state/latent collector rows."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

from .contract import DEFAULT_CONTRACT, DiffusionContract
from .state import WorldState


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
    if (
      not self.run_id
      or not self.episode_id
      or not self.segment_id
      or not self.motion_id
    ):
      raise StorageError("row identities must be non-empty")
    if self.controller not in {"vae", "ppo"}:
      raise StorageError("controller must be vae or ppo")
    if isinstance(self.env_id, bool) or not isinstance(self.env_id, (int, np.integer)):
      raise StorageError("env_id must be a non-negative integer")
    if self.env_id < 0:
      raise StorageError("env_id must be a non-negative integer")
    if isinstance(self.tick, bool) or self.tick < 0:
      raise StorageError("tick is invalid")
    if not isinstance(self.timestamp, (int, float, np.integer, np.floating)):
      raise StorageError("timestamp is required physical data")
    timestamp = float(self.timestamp)
    if not np.isfinite(timestamp):
      raise StorageError("timestamp must be finite physical data")
    object.__setattr__(self, "timestamp", timestamp)
    if not np.isfinite(self.fps) or self.fps <= 0:
      raise StorageError("fps must be finite and positive")
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
    if not all(
      isinstance(k, str) and isinstance(v, str) for k, v in self.provenance.items()
    ):
      raise StorageError("provenance must be a string mapping")

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


class AppendOnlyShardStore:
  """Durable, hash-checked row shards with bounded per-shard capacity."""

  FORMAT = "mjlab-x2-diffusion-raw-v1"

  def __init__(
    self,
    root: str | Path,
    *,
    contract: DiffusionContract = DEFAULT_CONTRACT,
    max_rows_per_shard: int = 4096,
    max_total_rows: int | None = None,
  ) -> None:
    if max_rows_per_shard <= 0 or (max_total_rows is not None and max_total_rows <= 0):
      raise StorageError("shard bounds must be positive")
    self.root = Path(root)
    self.contract = contract
    self.max_rows_per_shard = max_rows_per_shard
    self.max_total_rows = max_total_rows
    self.root.mkdir(parents=True, exist_ok=True)
    manifest = self.root / "manifest.json"
    if manifest.exists():
      self._validate_manifest()
    else:
      self._write_manifest([])
    # The set of written identities is cached because the previous append re-read,
    # re-hashed and decompressed every existing shard, which made collection
    # quadratic in the trial count (and re-read the store over the network
    # filesystem once per trial).  The manifest itself is still read from disk on
    # every access so that an on-disk rewrite is always observed.
    self._identity_cache: set[tuple[str, int, str, str, int]] | None = None

  def _manifest(self) -> dict[str, Any]:
    try:
      payload = json.loads((self.root / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
      raise StorageError("manifest is unreadable") from exc
    if (
      not isinstance(payload, dict)
      or payload.get("format") != self.FORMAT
      or payload.get("contract_hash") != self.contract.identity_hash()
    ):
      raise StorageError("manifest schema or contract hash mismatch")
    return payload

  def _write_manifest(self, shards: list[dict[str, Any]]) -> None:
    payload = {
      "format": self.FORMAT,
      "contract_hash": self.contract.identity_hash(),
      "contract": self.contract.as_dict(),
      "artifact_identities": [
        {
          "role": role,
          "path": path,
          "sha256": digest,
        }
        for role, path, digest in (
          ("vae", self.contract.vae_checkpoint, self.contract.vae_sha256),
          ("recovery", self.contract.recovery_onnx, self.contract.recovery_sha256),
          ("x2_xml", self.contract.state_xml, self.contract.state_xml_sha256),
          (
            "physical_fall_source",
            self.contract.physical_fall_source,
            self.contract.physical_fall_source_sha256,
          ),
          (
            "vae_tracking_rejection_source",
            self.contract.vae_tracking_rejection_source,
            self.contract.vae_tracking_rejection_source_sha256,
          ),
        )
      ],
      "max_rows_per_shard": self.max_rows_per_shard,
      "shards": shards,
    }
    temp = self.root / ".manifest.tmp"
    temp.write_text(json.dumps(payload, sort_keys=True, indent=2))
    os.replace(temp, self.root / "manifest.json")

  def _validate_manifest(self) -> None:
    payload = self._manifest()
    for shard in payload["shards"]:
      path = self.root / str(shard["file"])
      if (
        not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != shard["sha256"]
      ):
        raise StorageError(f"corrupt or missing shard {path.name}")
      if int(shard["rows"]) <= 0:
        raise StorageError("manifest contains an empty shard")

  @property
  def row_count(self) -> int:
    return sum(int(item["rows"]) for item in self._manifest()["shards"])

  @property
  def shard_count(self) -> int:
    return len(self._manifest()["shards"])

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
      digest = hashlib.sha256(path.read_bytes()).hexdigest()
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
      filename = item["file"]
      expected_rows = item["rows"]
      expected_hash = item["sha256"]
      if (
        not isinstance(filename, str)
        or not isinstance(expected_rows, int)
        or isinstance(expected_rows, bool)
        or expected_rows <= 0
        or not isinstance(expected_hash, str)
      ):
        raise StorageError("malformed shard manifest entry")
      path = self.root / filename
      if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise StorageError(f"corrupt shard hash: {path.name}")
      with np.load(path, allow_pickle=False) as data:
        n = expected_rows
        expected = {
          "root_position": (n, 3),
          "root_quaternion_wxyz": (n, 4),
          "root_linear_velocity": (n, 3),
          "root_angular_velocity": (n, 3),
          "body_positions": (n, self.contract.body_count, 3),
          "body_linear_velocities": (n, self.contract.body_count, 3),
          "previous_executed_action": (n, self.contract.action_dimension),
          "clean_action": (n, self.contract.action_dimension),
          "ou_noise": (n, self.contract.action_dimension),
          "executed_action": (n, self.contract.action_dimension),
          "latent": (n, self.contract.latent_dimension),
          "latent_present": (n,),
          "metadata": (n,),
        }
        # ``NpzFile.__getitem__`` re-decompresses the whole array on every
        # access, so each array is materialised exactly once here and reused for
        # every row.  Indexing ``data[name]`` inside the row loop re-decompressed
        # the full array per row: ~4.5 MB times the shard's row count (18.6 GB
        # for a 4096-row shard), which made the offline tools exhaust memory.
        arrays: dict[str, Any] = {}
        for name, shape in expected.items():
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
        states = WorldState(
          arrays["root_position"],
          arrays["root_quaternion_wxyz"],
          arrays["root_linear_velocity"],
          arrays["root_angular_velocity"],
          arrays["body_positions"],
          arrays["body_linear_velocities"],
        )
        result = []
        required_fields = {
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
        for i, text in enumerate(arrays["metadata"]):
          if not isinstance(text, str):
            raise StorageError("metadata entry must be a string")
          fields = json.loads(text)
          if not isinstance(fields, dict) or set(fields) != required_fields:
            raise StorageError("malformed row metadata")
          if not isinstance(fields["provenance"], dict):
            raise StorageError("row provenance must be a mapping")
          if not all(
            isinstance(fields[name], str)
            for name in (
              "run_id",
              "episode_id",
              "segment_id",
              "motion_id",
              "controller",
              "group_key",
            )
          ):
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
          if not all(
            isinstance(fields[name], bool)
            for name in ("segment_qualified", "reset", "teleport", "reference_boundary")
          ):
            raise StorageError("row flag metadata has invalid types")
          evidence = fields["terminal_evidence"]
          if evidence is not None and not isinstance(evidence, dict):
            raise StorageError("terminal evidence must be a mapping or null")
          result.append(
            TrajectoryRow(
              fields["run_id"],
              fields["env_id"],
              fields["episode_id"],
              fields["segment_id"],
              fields["tick"],
              fields["timestamp"],
              fields["motion_id"],
              fields["reference_frame"],
              fields["controller"],
              WorldState(
                *(
                  getattr(states, name)[i]
                  for name in (
                    "root_position",
                    "root_quaternion_wxyz",
                    "root_linear_velocity",
                    "root_angular_velocity",
                    "body_positions",
                    "body_linear_velocities",
                  )
                )
              ),
              arrays["previous_executed_action"][i],
              arrays["clean_action"][i],
              arrays["ou_noise"][i],
              arrays["executed_action"][i],
              arrays["latent"][i] if bool(arrays["latent_present"][i]) else None,
              None if evidence is None else TerminalEvidence.from_dict(evidence),
              fields["segment_qualified"],
              fields["reset"],
              fields["teleport"],
              fields["reference_boundary"],
              fields["fps"],
              fields["group_key"],
              fields["provenance"],
            )
          )
        return tuple(result)
    except StorageError:
      raise
    except Exception as exc:
      # ``np.load`` may defer malformed ZIP/container decoding until an array
      # is accessed; keep every ordinary decode failure at the storage API.
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
  "ShardStore",
  "ShardWriter",
  "StorageError",
  "TerminalEvidence",
  "TrajectoryRow",
  "VaeRow",
  "validate_monotonic_timestamps",
]
