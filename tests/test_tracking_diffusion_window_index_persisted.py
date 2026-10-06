"""The window index is a dataset artifact, not a per-task computation.

These tests pin the two properties that make persisting it safe: the round trip
is exact, and an index that no longer describes its store is refused rather than
silently reused.
"""

from __future__ import annotations

import json

import pytest
from tracking_diffusion_fixtures import make_row

from mjlab.tasks.tracking.diffusion import (
  AppendOnlyShardStore,
  DiffusionContract,
  SplitAssignments,
  WindowIndex,
  fit_training_statistics,
)
from mjlab.tasks.tracking.diffusion.dataset import (
  DatasetError,
  load_window_index,
  save_window_index,
  store_manifest_sha256,
  window_index_path,
)

CONTRACT_PATH = "docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml"


def build_store(root):
  store = AppendOnlyShardStore(root / "store", max_rows_per_shard=17)
  rows = [make_row(i, group="family-a", episode="a") for i in range(41)]
  rows += [make_row(i, group="family-b", episode="b") for i in range(41)]
  store.append(rows)
  return store


def test_persisted_index_round_trips_exactly(tmp_path) -> None:
  store = build_store(tmp_path)
  assignments = SplitAssignments({"family-a": "train", "family-b": "test"})
  index = WindowIndex.build(store, assignments)
  path = window_index_path(tmp_path)

  record = save_window_index(
    path,
    index,
    store_manifest_sha256=store_manifest_sha256(store),
    assignments_sha256=assignments.sha256(),
  )

  assert path.is_file()
  assert record["refs"] == len(index) == 2
  assert record["coverage"] == index.coverage()

  loaded = load_window_index(path, store=store, assignments=assignments)

  # Dataclass equality compares every field of every reference.
  assert loaded.refs == index.refs
  assert loaded.coverage() == index.coverage()


def test_persisted_index_is_refused_when_the_store_changed(tmp_path) -> None:
  store = build_store(tmp_path)
  assignments = SplitAssignments({"family-a": "train", "family-b": "test"})
  index = WindowIndex.build(store, assignments)
  path = window_index_path(tmp_path)
  save_window_index(
    path,
    index,
    store_manifest_sha256=store_manifest_sha256(store),
    assignments_sha256=assignments.sha256(),
  )

  store.append([make_row(41, group="family-c", episode="c")])

  with pytest.raises(DatasetError, match="does not describe this store"):
    load_window_index(path, store=store, assignments=assignments)


def test_persisted_index_is_refused_for_different_assignments(tmp_path) -> None:
  store = build_store(tmp_path)
  assignments = SplitAssignments({"family-a": "train", "family-b": "test"})
  index = WindowIndex.build(store, assignments)
  path = window_index_path(tmp_path)
  save_window_index(
    path,
    index,
    store_manifest_sha256=store_manifest_sha256(store),
    assignments_sha256=assignments.sha256(),
  )

  other = SplitAssignments({"family-a": "test", "family-b": "test"})
  assert other.sha256() != assignments.sha256()

  with pytest.raises(DatasetError, match="split assignments"):
    load_window_index(path, store=store, assignments=other)


def test_persisted_index_is_refused_for_another_contract(tmp_path) -> None:
  store = build_store(tmp_path)
  assignments = SplitAssignments({"family-a": "train", "family-b": "test"})
  index = WindowIndex.build(store, assignments)
  path = window_index_path(tmp_path)
  save_window_index(
    path,
    index,
    store_manifest_sha256=store_manifest_sha256(store),
    assignments_sha256=assignments.sha256(),
  )

  other_contract = DiffusionContract.from_yaml(
    "docs/plans/beyondmimic_diffusion_d0_contract.yaml"
  )
  assert other_contract.identity_hash() != index.contract.identity_hash()

  with pytest.raises(DatasetError, match="contract identity"):
    load_window_index(
      path, store=store, assignments=assignments, contract=other_contract
    )


def test_parallel_build_is_identical_to_sequential(tmp_path) -> None:
  """Partitioned work must reproduce the sequential index exactly."""
  from mjlab.tasks.tracking.diffusion.dataset import (
    build_window_index,
    segment_shard_plan,
  )

  store = AppendOnlyShardStore(tmp_path / "store", max_rows_per_shard=17)
  rows = []
  for episode in "abcdef":
    rows += [make_row(i, group=f"fam-{episode}", episode=episode) for i in range(41)]
  store.append(rows)
  assignments = SplitAssignments(
    {f"fam-{e}": ("train", "test")[index % 2] for index, e in enumerate("abcdef")}
  )

  plan = segment_shard_plan(store)
  assert plan is not None
  assert len(plan) == 6
  # Neighbouring segments share a boundary shard, so their extents overlap.
  assert any(
    plan[position + 1][1] < plan[position][2] for position in range(len(plan) - 1)
  )

  sequential, how_sequential = build_window_index(store, assignments, jobs=1)
  assert how_sequential == "sequential"
  assert len(sequential) == 6

  for jobs in (2, 3, 6):
    parallel, how = build_window_index(store, assignments, jobs=jobs)
    assert how.startswith("parallel")
    assert parallel.refs == sequential.refs
    assert parallel.coverage() == sequential.coverage()


def test_dataset_source_loads_the_index_instead_of_rebuilding(
  tmp_path, monkeypatch
) -> None:
  """The payoff: a dataset opens without the 12-25 minute rebuild."""
  from mjlab.tasks.tracking.diffusion import dataset as dataset_module
  from mjlab.tasks.tracking.diffusion.window_dataset import DatasetSource

  contract = DiffusionContract.from_yaml(CONTRACT_PATH)
  store = AppendOnlyShardStore(
    tmp_path / "store", contract=contract, max_rows_per_shard=17
  )
  rows = [make_row(i, group="family-a", episode="a") for i in range(41)]
  rows += [make_row(i, group="family-b", episode="b") for i in range(41)]
  store.append(rows)

  assignments = SplitAssignments({"family-a": "train", "family-b": "test"})
  index = WindowIndex.build(store, assignments, contract=contract)
  offline = tmp_path / "offline"
  offline.mkdir(parents=True, exist_ok=True)
  (offline / "assignments.json").write_text(
    json.dumps(assignments.as_dict()), encoding="utf-8"
  )
  (offline / "dataset.json").write_text(
    json.dumps({"windows": index.coverage()}), encoding="utf-8"
  )
  fit_training_statistics(index, contract=contract).save(offline / "projection.npz")
  save_window_index(
    window_index_path(tmp_path),
    index,
    store_manifest_sha256=store_manifest_sha256(store),
    assignments_sha256=assignments.sha256(),
  )

  def refuse(*args, **kwargs):
    raise AssertionError("a valid persisted index must not be rebuilt")

  monkeypatch.setattr(dataset_module.WindowIndex, "build", staticmethod(refuse))

  loaded = DatasetSource(tmp_path, CONTRACT_PATH).load()

  assert loaded.index_source == "persisted"
  assert loaded.index.refs == index.refs
