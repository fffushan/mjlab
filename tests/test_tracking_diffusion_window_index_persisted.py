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
  build_window_index,
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


def _geometry_store(tmp_path, episodes, rows_per_episode, shard_rows):
  store = AppendOnlyShardStore(tmp_path / "store", max_rows_per_shard=shard_rows)
  rows = []
  for episode in episodes:
    rows += [
      make_row(index, group=f"fam-{episode}", episode=episode)
      for index in range(rows_per_episode)
    ]
  store.append(rows)
  assignments = SplitAssignments(
    {
      f"fam-{episode}": ("train", "test")[position % 2]
      for position, episode in enumerate(episodes)
    }
  )
  return store, assignments


def test_parallel_build_keeps_segments_inside_a_shard(tmp_path) -> None:
  """The regression this test exists for: interior segments must survive.

  With 41-row segments in 300-row shards, most segments begin and end inside one
  shard.  A partition deriving its work from shard first/last identities sees
  only the few segments touching a boundary and silently drops the rest, which
  is exactly what happened on the real 607-shard dataset: 30,228 references
  instead of 609,376.  The first version of this test could not catch it, because
  17-row shards forced every segment to straddle a boundary.
  """
  episodes = "abcdefghijkl"
  store, assignments = _geometry_store(tmp_path, episodes, 41, 300)
  assert store.shard_count >= 2

  sequential, how = build_window_index(store, assignments, jobs=1)
  assert how == "sequential"
  assert len(sequential) == len(episodes)

  for jobs in (2, 3, 12):
    parallel, how_parallel = build_window_index(store, assignments, jobs=jobs)
    assert how_parallel.startswith("parallel")
    assert len(parallel.refs) == len(episodes)
    assert parallel.refs == sequential.refs
    assert parallel.coverage() == sequential.coverage()


def test_parallel_build_is_identical_when_every_segment_straddles_shards(
  tmp_path,
) -> None:
  episodes = "abcdef"
  store, assignments = _geometry_store(tmp_path, episodes, 41, 17)

  sequential, how_sequential = build_window_index(store, assignments, jobs=1)
  assert how_sequential == "sequential"
  assert len(sequential) == len(episodes)

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
