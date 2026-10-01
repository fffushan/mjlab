"""Explicit conversion of legacy projected raw stores to no-mixture datasets.

This module is intentionally not used by normal loaders.  It copies raw physical
state/latent rows and the existing split assignment, then fits fresh train-only
normalization statistics under the versioned no-mixture contract.  The legacy
store and its projection remain byte-intact.
"""

from __future__ import annotations

import hashlib
import json
from itertools import islice
from pathlib import Path
from typing import Any

from .contract import DiffusionContract
from .dataset import WindowIndex, fit_training_statistics
from .storage import AppendOnlyShardStore
from .window_dataset import DatasetSource


def rebuild_no_mixture_dataset(
  source_directory: str | Path,
  destination_directory: str | Path,
  *,
  source_contract: str | Path,
  contract_path: str | Path,
  shard_rows: int = 4096,
) -> dict[str, Any]:
  """Rebuild a distinct no-mixture dataset without simulator recollection.

  ``source_contract`` must be the historical D0 YAML and ``contract_path`` the
  new no-mixture YAML.  The operation refuses an existing destination and
  preserves the source rows, split groups, VAE identities and row provenance.
  """
  source = DatasetSource.resolve(source_directory, contract_path=source_contract)
  loaded = source.load()
  active = DiffusionContract.from_yaml(contract_path)
  if loaded.contract.schema_version != "mjlab-x2-state-latent-d0-v1":
    raise ValueError("source must use the historical projected D0 contract")
  if active.projection_rows != 0 or active.token_dimension != 167:
    raise ValueError("destination must be the active no-mixture contract")
  destination = Path(destination_directory)
  if destination.exists() and any(destination.iterdir()):
    raise FileExistsError(f"destination is not empty: {destination}")
  destination.mkdir(parents=True, exist_ok=True)
  store = AppendOnlyShardStore(
    destination / "store", contract=active, max_rows_per_shard=shard_rows
  )
  source_rows = loaded.store.iter_rows()
  row_count = 0
  while chunk := tuple(islice(source_rows, shard_rows)):
    store.append(chunk)
    row_count += len(chunk)
  if row_count != loaded.store.row_count:
    raise ValueError("migration row count differs from the source store")
  assignments = loaded.assignments
  offline = destination / "offline"
  offline.mkdir(parents=True, exist_ok=True)
  (offline / "assignments.json").write_text(
    json.dumps(assignments.as_dict(), indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  index = WindowIndex.build(store, assignments, contract=active)
  bundle = fit_training_statistics(index, contract=active)
  bundle.save(offline / "projection.npz")
  coverage = index.coverage()
  if coverage != loaded.index.coverage():
    raise ValueError("migration changed eligible windows or split coverage")
  manifest = {
    "schema_version": active.schema_version,
    "contract_hash": active.identity_hash(),
    "representation": "normalized_physical_state_plus_normalized_vae_latent",
    "source": {
      "directory": str(Path(source_directory)),
      "contract_schema": loaded.contract.schema_version,
      "contract_hash": loaded.contract.identity_hash(),
      "store_manifest_sha256": hashlib.sha256(
        (Path(source_directory) / "store" / "manifest.json").read_bytes()
      ).hexdigest(),
    },
    "preserved": {
      "row_count": row_count,
      "assignments_hash": assignments.sha256(),
      "vae_sha256": active.vae_sha256,
      "body_names": list(active.body_names),
    },
    "windows": coverage,
    "projection_rows": 0,
    "token_dimension": active.token_dimension,
  }
  (offline / "dataset.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  return manifest


__all__ = ["rebuild_no_mixture_dataset"]
