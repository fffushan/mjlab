from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from tracking_diffusion_fixtures import make_row

from mjlab.tasks.tracking.diffusion import (
  AppendOnlyShardStore,
  DatasetError,
  DiffusionContract,
  FeatureStats,
  ProjectionBundle,
  ProjectionError,
  SplitAssignments,
  WindowIndex,
  rebuild_no_mixture_dataset,
)
from mjlab.tasks.tracking.diffusion.window_dataset import DatasetSource


def test_explicit_migration_copies_rows_splits_and_builds_167_tokens(
  tmp_path: Path,
) -> None:
  old_contract = DiffusionContract.from_yaml(
    "docs/plans/beyondmimic_diffusion_d0_contract.yaml"
  )
  source = tmp_path / "legacy"
  store = AppendOnlyShardStore(
    source / "store", contract=old_contract, max_rows_per_shard=64
  )
  rows = [make_row(i, group="g", episode="e") for i in range(50)]
  store.append(rows)
  assignments = SplitAssignments({"g": "train"})
  offline = source / "offline"
  offline.mkdir()
  (offline / "assignments.json").write_text(
    json.dumps(assignments.as_dict()), encoding="utf-8"
  )
  old_bundle = ProjectionBundle.create(
    FeatureStats(1, np.zeros(135), np.ones(135)),
    FeatureStats(1, np.zeros(32), np.ones(32)),
    contract=old_contract,
  )
  old_bundle.save(offline / "projection.npz")
  old_index = WindowIndex.build(store, assignments, contract=old_contract)
  (offline / "dataset.json").write_text(
    json.dumps({"windows": old_index.coverage()}), encoding="utf-8"
  )

  with pytest.raises(DatasetError, match="legacy projected D0.*migration"):
    DatasetSource.resolve(
      source,
      contract_path="docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml",
    ).load()

  destination = tmp_path / "active"
  manifest = rebuild_no_mixture_dataset(
    source,
    destination,
    source_contract="docs/plans/beyondmimic_diffusion_d0_contract.yaml",
    contract_path="docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml",
    shard_rows=64,
  )
  assert manifest["preserved"]["row_count"] == 50
  assert manifest["preserved"]["assignments_hash"] == assignments.sha256()
  loaded = DatasetSource.resolve(
    destination,
    contract_path="docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml",
  ).load()
  assert loaded.store.row_count == store.row_count
  assert loaded.assignments.as_dict() == assignments.as_dict()
  assert loaded.index.coverage() == manifest["windows"]
  assert loaded.dataset("train")[0].tokens.shape == (41, 167)


def test_legacy_projected_bundle_is_rejected_by_active_contract(tmp_path: Path) -> None:
  old = DiffusionContract.from_yaml("docs/plans/beyondmimic_diffusion_d0_contract.yaml")
  bundle = ProjectionBundle.create(
    FeatureStats(1, np.zeros(135), np.ones(135)),
    FeatureStats(1, np.zeros(32), np.ones(32)),
    contract=old,
  )
  path = tmp_path / "legacy-projection.npz"
  bundle.save(path)
  with pytest.raises(ProjectionError, match="legacy projected D0|migration"):
    ProjectionBundle.load(path)
