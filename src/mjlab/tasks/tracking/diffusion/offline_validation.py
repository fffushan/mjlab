"""Bounded shard-level inspect and replay for raw D1 diffusion stores.

These are the reads behind the offline ``inspect`` and ``replay`` commands.  Each
worker decodes only the shards it is assigned and returns a compact reduction:
row and controller counts, error strings, boundary identities and one 128-bit
identity digest per row.  A worker never returns row objects, so the parent's
memory is proportional to the digest matrix and not to the state/action payload.

Reductions are grouped per shard and concatenated in shard order, so a serial
run (``jobs=1``) and a parallel run report identical rows, errors and identity
verdicts.

A controller/latent inconsistency (a VAE row without a latent, or a PPO row
with one) is reported separately from the arithmetic errors, so ``inspect`` can
refuse such a store while ``replay`` keeps reporting every row error.
"""

from __future__ import annotations

import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contract import DEFAULT_CONTRACT, DiffusionContract
from .storage import (
  ShardEntry,
  ShardStoreSource,
  StorageError,
  digest_matrix,
  duplicate_row_identity,
  identity_digest,
  load_shard_arrays,
  parse_row_metadata,
  shard_partitions,
)

# Action arithmetic tolerance of the historical replay command.
ACTION_ATOL = 1.0e-5
ACTION_RTOL = 1.0e-5
# Wording of the historical replay payload.
REPLAY_NOTE = "offline shard replay diagnostics; no simulator was launched"

_Identity = tuple[str, int, str, str, int]


@dataclass(frozen=True, slots=True)
class ShardValidation:
  """Compact reduction for one shard.

  ``digests`` holds one 128-bit identity digest per row, which lets the parent
  detect a duplicate identity across shards without transporting identities or
  rows.  The boundary identities and timestamps are the shard's physical
  boundaries.  ``errors`` holds every row error in row order; the subset in
  ``semantic_errors`` is the controller/latent disagreement.
  """

  file: str
  rows: int
  vae_rows: int
  ppo_rows: int
  errors: tuple[str, ...]
  semantic_errors: tuple[str, ...]
  first_identity: _Identity
  last_identity: _Identity
  first_timestamp: float
  last_timestamp: float
  digests: np.ndarray


@dataclass(frozen=True, slots=True)
class StoreValidation:
  """Store-level reduction: per-shard reductions plus whole-store totals."""

  root: str
  rows: int
  vae_rows: int
  ppo_rows: int
  unique_identities: bool
  duplicate_identity: _Identity | None
  errors: tuple[str, ...]
  semantic_errors: tuple[str, ...]
  shards: tuple[ShardValidation, ...]


@dataclass(frozen=True, slots=True)
class _ShardTask:
  root: str
  contract: DiffusionContract
  entries: tuple[ShardEntry, ...]


def validate_shard(
  root: str | Path, contract: DiffusionContract, entry: ShardEntry
) -> ShardValidation:
  """Validate one shard from its decoded arrays, without building row objects.

  The array checks (hash, shape, dtype, finite values, latent flags) and the
  per-row metadata contract are enforced by the shared storage readers.  This
  function adds the row-level evidence the replay payload reports: action
  arithmetic and controller/latent agreement.
  """
  arrays = load_shard_arrays(root, contract, entry)
  arithmetic = np.isclose(
    arrays["clean_action"] + arrays["ou_noise"],
    arrays["executed_action"],
    atol=ACTION_ATOL,
    rtol=ACTION_RTOL,
  ).all(axis=1)
  latent_present = arrays["latent_present"]
  vae_rows = 0
  ppo_rows = 0
  errors: list[str] = []
  semantic_errors: list[str] = []
  digests: list[bytes] = []
  first: _Identity | None = None
  last: _Identity | None = None
  first_timestamp = last_timestamp = 0.0
  for index, text in enumerate(arrays["metadata"]):
    if not isinstance(text, str):
      raise StorageError("metadata entry must be a string")
    row = parse_row_metadata(text)
    identity = row.identity
    digests.append(identity_digest(identity))
    if first is None:
      first, first_timestamp = identity, row.timestamp
    last, last_timestamp = identity, row.timestamp
    if not bool(arithmetic[index]):
      errors.append(f"{identity}: executed action arithmetic mismatch")
    if row.controller == "vae":
      vae_rows += 1
      if not bool(latent_present[index]):
        message = f"{identity}: VAE row has no latent"
        errors.append(message)
        semantic_errors.append(message)
    else:
      ppo_rows += 1
      if bool(latent_present[index]):
        message = f"{identity}: PPO row has a latent"
        errors.append(message)
        semantic_errors.append(message)
  if first is None or last is None:
    raise StorageError(f"shard {entry.file} has no rows to validate")
  return ShardValidation(
    entry.file,
    entry.rows,
    vae_rows,
    ppo_rows,
    tuple(errors),
    tuple(semantic_errors),
    first,
    last,
    first_timestamp,
    last_timestamp,
    digest_matrix(digests),
  )


def _validate_partition(task: _ShardTask) -> tuple[ShardValidation, ...]:
  return tuple(
    validate_shard(task.root, task.contract, entry) for entry in task.entries
  )


def validate_store(
  store_root: str | Path,
  *,
  contract: DiffusionContract = DEFAULT_CONTRACT,
  jobs: int = 1,
) -> StoreValidation:
  """Validate every shard of a store, partitioned over at most ``jobs`` workers.

  A worker receives its shard entries and reads only those shards.  ``jobs=1``
  runs the same reductions in this process, so serial and parallel results are
  produced by the same code.
  """
  source = ShardStoreSource.open(store_root, contract=contract)
  tasks = tuple(
    _ShardTask(str(source.root), contract, source.entries[start:stop])
    for start, stop in shard_partitions(source.entries, jobs)
  )
  if jobs == 1 or len(tasks) <= 1:
    reductions = tuple(_validate_partition(task) for task in tasks)
  else:
    context = multiprocessing.get_context("fork")
    with context.Pool(processes=len(tasks)) as pool:
      reductions = tuple(pool.map(_validate_partition, tasks))

  shards = tuple(shard for part in reductions for shard in part)
  expected = [entry.file for entry in source.entries]
  if [shard.file for shard in shards] != expected:
    raise StorageError("validation did not cover the store's shards in order")
  digest_parts = [shard.digests for shard in shards]
  digests = (
    np.concatenate(digest_parts) if digest_parts else np.empty((0, 2), dtype=np.uint64)
  )
  unique = len(np.unique(digests, axis=0)) == len(digests)
  duplicate: _Identity | None = None
  if not unique:
    duplicate = duplicate_row_identity((source,))
    if duplicate is None:
      raise StorageError("identity digest collision: refusing an identity verdict")
  return StoreValidation(
    root=str(source.root),
    rows=sum(shard.rows for shard in shards),
    vae_rows=sum(shard.vae_rows for shard in shards),
    ppo_rows=sum(shard.ppo_rows for shard in shards),
    unique_identities=unique,
    duplicate_identity=duplicate,
    errors=tuple(error for shard in shards for error in shard.errors),
    semantic_errors=tuple(error for shard in shards for error in shard.semantic_errors),
    shards=shards,
  )


def inspect_store(
  store_root: str | Path,
  *,
  contract: DiffusionContract = DEFAULT_CONTRACT,
  jobs: int = 1,
) -> dict[str, Any]:
  """Report store size, controller mix and identity uniqueness.

  A controller/latent inconsistency fails closed here, exactly as the historical
  row-building inspect refused an inconsistent store before it printed anything.
  Action-arithmetic disagreement stays a replay finding, and the payload keys of
  an inconsistent-free store are unchanged.
  """
  validation = validate_store(store_root, contract=contract, jobs=jobs)
  if validation.semantic_errors:
    raise StorageError(
      f"store {store_root} has {len(validation.semantic_errors)} controller/latent "
      f"inconsistencies: {list(validation.semantic_errors[:3])}"
    )
  return {
    "contract_sha256": contract.contract_sha256,
    "store": str(store_root),
    "shards": len(validation.shards),
    "rows": validation.rows,
    "controllers": {"vae": validation.vae_rows, "ppo": validation.ppo_rows},
    "unique_identities": validation.unique_identities,
  }


def replay_store(
  store_root: str | Path,
  *,
  contract: DiffusionContract = DEFAULT_CONTRACT,
  jobs: int = 1,
) -> dict[str, Any]:
  """Replay the stored row evidence offline: action arithmetic and latents."""
  validation = validate_store(store_root, contract=contract, jobs=jobs)
  return {
    "ok": not validation.errors,
    "rows": validation.rows,
    "errors": list(validation.errors),
    "note": REPLAY_NOTE,
  }


__all__ = [
  "ACTION_ATOL",
  "ACTION_RTOL",
  "REPLAY_NOTE",
  "ShardValidation",
  "StoreValidation",
  "inspect_store",
  "replay_store",
  "validate_shard",
  "validate_store",
]
