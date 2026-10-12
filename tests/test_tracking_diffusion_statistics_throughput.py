"""Focused tests for the one-pass, segment-batched statistics fit.

The fit must compute exactly the moments the previous per-window traversal
computed: the same 41-row windows, the same center frame, the same window
overlap multiplicity, the same train-only contamination rule and the same scalar
standard-deviation floor.  These tests pin that contract, and pin the two
properties the throughput change exists for: one state transform per window, and
one validated segment snapshot per segment.

A range partition changes the order in which float64 partial moments are
combined, so a partition gives statistics equal to the single-range fit to
floating-point tolerance and not bit for bit.  The tolerance is asserted here
instead of a hash comparison.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from tracking_diffusion_fixtures import make_row, make_state

from mjlab.tasks.tracking.diffusion import (
  AppendOnlyShardStore,
  DatasetError,
  DiffusionContract,
  ProjectionBundle,
  ProjectionError,
  SplitAssignments,
  WindowIndex,
  WindowRef,
  WorldState,
  fit_feature_stats,
  fit_training_statistics,
)
from mjlab.tasks.tracking.diffusion import dataset as dataset_module
from mjlab.tasks.tracking.diffusion import storage as storage_module
from mjlab.tasks.tracking.diffusion.dataset import (
  build_window_index,
  iter_owned_segment_rows,
  segment_window_scan,
  shard_partitions,
)
from mjlab.tasks.tracking.diffusion.projection import (
  FeatureMoments,
  FeatureMomentsAccumulator,
  combine_feature_moments,
)
from mjlab.tasks.tracking.diffusion.state import world_to_hybrid
from mjlab.tasks.tracking.diffusion.storage import StorageError

CONTRACT = DiffusionContract()
_MOMENT_RTOL = 1.0e-12
_MOMENT_ATOL = 1.0e-15


def _store_with_segments(
  root: Any,
  *,
  segments: int = 3,
  rows_per_segment: int = 70,
  shard_rows: int = 23,
  split: str = "train",
) -> AppendOnlyShardStore:
  """One independent segment per episode, deliberately spreading shard cuts."""
  store = AppendOnlyShardStore(root, max_rows_per_shard=shard_rows)
  for index in range(segments):
    store.append(
      [
        make_row(
          tick,
          group=f"fam-{index}",
          episode=f"episode-{index}",
          run=f"run-{index}",
          env=index,
        )
        for tick in range(rows_per_segment)
      ]
    )
  return store


def _train_assignments(segments: int) -> SplitAssignments:
  return SplitAssignments({f"fam-{index}": "train" for index in range(segments)})


def _scalar_reference(
  index: WindowIndex, contract: DiffusionContract
) -> ProjectionBundle:
  """Resolve and transform one window at a time, as the previous fit did.

  The rows are resolved by tick range from one store pass, which is the
  resolution the previous implementation used, so this reference is independent
  of the batched segment traversal.
  """
  rows_by_segment = index._segment_rows()
  state_chunks: list[np.ndarray] = []
  latent_chunks: list[np.ndarray] = []
  for ref in index.refs_for("train"):
    segment = rows_by_segment[(ref.run_id, ref.env_id, ref.episode_id, ref.segment_id)]
    rows = tuple(
      row for row in segment if ref.start_tick <= row.tick < ref.start_tick + ref.length
    )
    assert len(rows) == contract.window_steps
    hybrid, _ = world_to_hybrid(
      WorldState.stack(row.state for row in rows),
      center_index=contract.current_index,
    )
    state_chunks.append(hybrid)
    latent_chunks.append(np.stack([row.latent for row in rows]))
  return ProjectionBundle.create(
    fit_feature_stats(state_chunks, std_floor=contract.std_floor),
    fit_feature_stats(latent_chunks, std_floor=contract.std_floor),
    contract=contract,
  )


def _assert_same_statistics(
  actual: ProjectionBundle, expected: ProjectionBundle
) -> None:
  assert actual.state_stats.count == expected.state_stats.count
  assert actual.latent_stats.count == expected.latent_stats.count
  for name in ("state_stats", "latent_stats"):
    left, right = getattr(actual, name), getattr(expected, name)
    np.testing.assert_allclose(
      left.mean, right.mean, rtol=_MOMENT_RTOL, atol=_MOMENT_ATOL
    )
    np.testing.assert_allclose(
      left.std, right.std, rtol=_MOMENT_RTOL, atol=_MOMENT_ATOL
    )


def test_batched_fit_matches_the_per_window_reference(tmp_path) -> None:
  """Nontrivial yaw, velocity and overlapping windows, one reference fit."""
  store = AppendOnlyShardStore(tmp_path / "store", max_rows_per_shard=29)
  rows = []
  for tick in range(120):
    yaw = 0.4 * tick
    state = WorldState(
      np.array([0.3 * tick, -0.2 + 0.01 * tick, 0.5]),
      np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]),
      np.array([0.2 * tick, -0.1, 0.03 * tick]),
      np.array([0.05 * tick, 0.1, -0.2]),
      np.arange(60, dtype=np.float64).reshape(20, 3) * 0.01,
      np.ones((20, 3), dtype=np.float64) * (0.2 + 0.001 * tick),
    )
    rows.append(
      replace(
        make_row(tick, state=state),
        latent=np.linspace(-1.0, 1.0, 32) * (1.0 + tick),
      )
    )
  store.append(rows)
  index = WindowIndex.build(store, SplitAssignments({"group": "train"}))
  assert len(index.refs_for("train")) == 120 - 41 + 1

  bundle = fit_training_statistics(index)
  reference = _scalar_reference(index, CONTRACT)

  _assert_same_statistics(bundle, reference)
  assert bundle.state_stats.count == 41 * len(index.refs_for("train"))


def test_statistics_moments_use_the_center_frame(tmp_path) -> None:
  store = _store_with_segments(tmp_path / "store", shard_rows=31)
  index = WindowIndex.build(store, _train_assignments(3))

  bundle = fit_training_statistics(index)
  shifted = _scalar_reference(
    index,
    replace(
      CONTRACT,
      past_steps=CONTRACT.past_steps + 1,
      current_index=CONTRACT.current_index + 1,
      future_steps=CONTRACT.future_steps - 1,
    ),
  )

  assert not np.allclose(bundle.state_stats.mean, shifted.state_stats.mean, atol=1e-9)


def test_every_train_window_is_transformed_once_in_batches(
  tmp_path, monkeypatch
) -> None:
  """The fit maps windows in blocks, and each window is mapped exactly once."""
  store = _store_with_segments(tmp_path / "store", segments=2, rows_per_segment=60)
  index = WindowIndex.build(store, _train_assignments(2))
  train_windows = len(index.refs_for("train"))
  assert train_windows == 2 * (60 - 41 + 1)

  sizes: list[int] = []
  original = dataset_module.world_to_hybrid_windows

  def counting(states: Any, starts: Any, **kwargs: Any) -> Any:
    sizes.append(len(starts))
    return original(states, starts, **kwargs)

  monkeypatch.setattr(dataset_module, "world_to_hybrid_windows", counting)
  fit_training_statistics(index)
  # Every window is mapped once, and the traversal uses blocks, not one call
  # per window in Python.
  assert sum(sizes) == train_windows
  assert 0 < len(sizes) < train_windows
  assert max(sizes) <= dataset_module._STATISTICS_BATCH_WINDOWS


def test_fit_transforms_only_train_windows(tmp_path, monkeypatch) -> None:
  """Validation and test values must not reach the fit or its transforms."""
  store = AppendOnlyShardStore(tmp_path / "store", max_rows_per_shard=23)
  train = [
    make_row(tick, group="fam", episode="train", run="train-run") for tick in range(60)
  ]
  store.append(train)
  store.append(
    [
      replace(
        make_row(tick, group="hold", episode="validation", run="validation-run"),
        state=make_state(tick, angular_velocity=1.0e6),
        latent=np.full(32, 1.0e6),
      )
      for tick in range(60)
    ]
  )
  store.append(
    [
      replace(
        make_row(tick, group="heldout", episode="test", run="test-run"),
        state=make_state(tick, angular_velocity=-2.0e6),
        latent=np.full(32, -2.0e6),
      )
      for tick in range(60)
    ]
  )
  assignments = SplitAssignments(
    {"fam": "train", "hold": "validation", "heldout": "test"}
  )
  index = WindowIndex.build(store, assignments)
  train_windows = len(index.refs_for("train"))
  assert train_windows == 60 - 41 + 1

  transformed: list[int] = []
  original = dataset_module.world_to_hybrid_windows

  def counting(states: Any, starts: Any, **kwargs: Any) -> Any:
    transformed.append(len(starts))
    return original(states, starts, **kwargs)

  monkeypatch.setattr(dataset_module, "world_to_hybrid_windows", counting)
  bundle = fit_training_statistics(index)

  assert sum(transformed) == train_windows
  assert bundle.state_stats.count == 41 * train_windows

  train_only = AppendOnlyShardStore(tmp_path / "train-only", max_rows_per_shard=23)
  train_only.append(train)
  train_only_index = WindowIndex.build(train_only, SplitAssignments({"fam": "train"}))
  assert (
    bundle.statistics_sha256
    == fit_training_statistics(train_only_index).statistics_sha256
  )


def test_jobs_and_batch_partitions_keep_window_multiplicity(tmp_path) -> None:
  """Overlapping windows keep their weight; a partition is a tolerance change."""
  store = _store_with_segments(tmp_path / "store", segments=3, rows_per_segment=70)
  index = WindowIndex.build(store, _train_assignments(3))
  windows = len(index.refs_for("train"))
  assert windows == 3 * (70 - 41 + 1)
  assert store.shard_count >= 6

  reference = fit_training_statistics(index)
  assert reference.state_stats.count == 41 * windows
  assert reference.latent_stats.count == 41 * windows

  for batch_windows in (1, 4, 64):
    _assert_same_statistics(
      fit_training_statistics(index, batch_windows=batch_windows), reference
    )
  for jobs in (2, 3, 7):
    _assert_same_statistics(fit_training_statistics(index, jobs=jobs), reference)

  assert fit_training_statistics(index).statistics_sha256 == reference.statistics_sha256


def test_std_floor_is_applied_to_constant_features(tmp_path) -> None:
  store = AppendOnlyShardStore(tmp_path / "store", max_rows_per_shard=23)
  store.append([make_row(tick, state=make_state(0)) for tick in range(45)])
  index = WindowIndex.build(store, SplitAssignments({"group": "train"}))

  bundle = fit_training_statistics(index)

  np.testing.assert_allclose(bundle.state_stats.std, CONTRACT.std_floor)


def test_segment_scan_reproduces_window_eligibility() -> None:
  """The one-pass scan must answer exactly what the chunk check answers."""
  width = CONTRACT.window_steps
  base = [make_row(tick) for tick in range(45)]
  variants: dict[str, list[Any]] = {"clean": base}
  for name, changes in {
    "reset": {"reset": True},
    "teleport": {"teleport": True},
    "reference_boundary": {"reference_boundary": True},
    "unqualified": {"segment_qualified": False},
    "no_terminal_evidence": {"terminal_evidence": None},
  }.items():
    rows = list(base)
    rows[20] = replace(rows[20], **changes)
    variants[name] = rows
  rows = list(base)
  evidence = rows[20].terminal_evidence
  assert evidence is not None
  rows[20] = replace(rows[20], terminal_evidence=replace(evidence, available=False))
  variants["terminal_evidence_unavailable"] = rows
  rows = list(base)
  rows[20] = replace(rows[20], controller="ppo", latent=None)
  variants["ppo_row"] = rows
  rows = list(base)
  for position in range(25, 45):
    rows[position] = replace(rows[position], tick=rows[position].tick + 1)
  variants["tick_gap"] = rows
  rows = list(base)
  rows[30] = replace(rows[30], timestamp=rows[29].timestamp)
  variants["repeated_timestamp"] = rows
  rows = list(base)
  for position in range(25, 45):
    rows[position] = replace(rows[position], fps=51.0)
  variants["frame_rate_change"] = rows
  rows = list(base)
  for position in (26, 31):
    rows[position] = replace(rows[position], fps=50.0 + 1.0e-13)
  variants["frame_rate_within_tolerance"] = rows
  rows = list(base)
  for position in range(30, 45):
    rows[position] = replace(rows[position], motion_id="other-motion")
  variants["motion_change"] = rows

  for name, rows in variants.items():
    ordered = tuple(sorted(rows, key=lambda row: row.tick))
    scan = segment_window_scan(rows, CONTRACT)
    assert scan.width == width
    for position in range(len(ordered) - width + 1):
      window = ordered[position : position + width]
      expected = WindowIndex._eligible(window, CONTRACT)
      assert bool(scan.eligible_starts[position]) == expected, (name, position)


def test_window_lookup_validates_a_segment_once(tmp_path, monkeypatch) -> None:
  store = _store_with_segments(tmp_path / "store", segments=2, rows_per_segment=60)
  index = WindowIndex.build(store, _train_assignments(2))

  scans: list[int] = []
  original = dataset_module.segment_window_scan

  def recording(rows: Any, contract: DiffusionContract) -> Any:
    scans.append(len(rows))
    return original(rows, contract)

  monkeypatch.setattr(dataset_module, "segment_window_scan", recording)
  reads = 0
  read_shard = AppendOnlyShardStore.iter_rows

  def counting_iter_rows(self: Any) -> Any:
    nonlocal reads
    reads += 1
    yield from read_shard(self)

  monkeypatch.setattr(AppendOnlyShardStore, "iter_rows", counting_iter_rows)

  for ref in index.refs:
    assert len(index._rows(ref)) == 41
  for ref in index.refs:
    assert len(index._rows(ref)) == 41

  assert scans == [60, 60]
  assert reads == 1


def test_refs_by_segment_groups_the_indexed_snapshot(tmp_path) -> None:
  store = _store_with_segments(tmp_path / "store", segments=2, rows_per_segment=60)
  assignments = SplitAssignments({"fam-0": "train", "fam-1": "test"})
  index = WindowIndex.build(store, assignments)

  grouped = index.refs_by_segment("train")
  assert sum(len(refs) for refs in grouped.values()) == len(index.refs_for("train"))
  for key, refs in grouped.items():
    assert key == (
      refs[0].run_id,
      refs[0].env_id,
      refs[0].episode_id,
      refs[0].segment_id,
    )
    assert [ref.start_tick for ref in refs] == sorted(ref.start_tick for ref in refs)


def test_shard_ranges_own_every_segment_exactly_once(tmp_path) -> None:
  """A segment longer than a shard is still read by exactly one range."""
  store = AppendOnlyShardStore(tmp_path / "store", max_rows_per_shard=17)
  episodes = ("a", "b", "c", "d")
  for position, episode in enumerate(episodes):
    store.append(
      [
        make_row(
          tick,
          group=f"fam-{episode}",
          episode=episode,
          run=f"run-{episode}",
          env=position,
        )
        for tick in range(41)
      ]
    )
  assert store.shard_count >= 9
  expected = [
    (f"run-{episode}", position, episode, "segment")
    for position, episode in enumerate(episodes)
  ]

  whole = list(
    iter_owned_segment_rows(store, shard_start=0, shard_stop=store.shard_count)
  )
  assert [key for key, _ in whole] == expected
  assert all(len(rows) == 41 for _, rows in whole)

  ranges = shard_partitions(store, 3)
  assert len(ranges) == 3
  seen: list[Any] = []
  for start, stop in ranges:
    for key, rows in iter_owned_segment_rows(store, shard_start=start, shard_stop=stop):
      assert len(rows) == 41
      assert [row.tick for row in rows] == list(range(41))
      seen.append(key)
  assert seen == expected


def test_fit_refuses_an_index_that_disagrees_with_the_store(tmp_path) -> None:
  """A stale index must fail loudly instead of silently refitting."""
  store = _store_with_segments(tmp_path / "store", segments=2, rows_per_segment=41)
  assignments = _train_assignments(2)
  index = WindowIndex.build(store, assignments)
  assert len(index.refs) == 2

  partial = WindowIndex(store, assignments, index.refs[:1])
  with pytest.raises(DatasetError, match="unexpected"):
    fit_training_statistics(partial)

  ghost = WindowRef("ghost-run", 0, "ghost", "segment", 0, "train", "fam-9")
  invented = WindowIndex(store, assignments, (*index.refs, ghost))
  with pytest.raises(DatasetError, match="missing"):
    fit_training_statistics(invented)

  shared = WindowRef("run-0", 0, "episode-0", "segment", 0, "train", "fam-0")
  misplaced = WindowIndex(
    store, assignments, (replace(shared, start_tick=1), index.refs[1])
  )
  with pytest.raises(DatasetError, match="differ from the window index"):
    fit_training_statistics(misplaced)


def test_fit_refuses_a_corrupt_shard_and_invalid_controls(tmp_path) -> None:
  store = _store_with_segments(tmp_path / "store", segments=1, rows_per_segment=41)
  index = WindowIndex.build(store, _train_assignments(1))

  with pytest.raises(DatasetError, match="jobs"):
    fit_training_statistics(index, jobs=0)
  with pytest.raises(DatasetError, match="batch_windows"):
    fit_training_statistics(index, batch_windows=0)

  shard = sorted((tmp_path / "store").glob("shard-*.npz"))[0]
  corrupted = bytearray(shard.read_bytes())
  corrupted[len(corrupted) // 2] ^= 0xFF
  shard.write_bytes(bytes(corrupted))
  with pytest.raises(StorageError):
    fit_training_statistics(index)
  with pytest.raises(StorageError):
    fit_training_statistics(index, jobs=2)


def test_a_split_segment_is_refused_by_every_reader(tmp_path) -> None:
  """A segment stored in two runs must never be read as a partial segment."""
  store = AppendOnlyShardStore(tmp_path / "store", max_rows_per_shard=17)
  first = [make_row(tick, group="fam", episode="a", run="run-a") for tick in range(20)]
  other = [make_row(tick, group="fam", episode="b", run="run-b") for tick in range(20)]
  rest = [
    make_row(tick, group="fam", episode="a", run="run-a") for tick in range(20, 41)
  ]
  store.append(first + other + rest)
  assignments = SplitAssignments({"fam": "train"})
  # The sequential index build groups the two runs into one 41-row segment.
  index = WindowIndex.build(store, assignments)
  assert len(index.refs) == 1

  # One reader sees both runs, and a shard range sees the later run after the
  # manifest already named that segment, so every reader refuses the shape.
  with pytest.raises(DatasetError, match="more than one run"):
    fit_training_statistics(index)
  with pytest.raises(DatasetError, match="more than one run"):
    build_window_index(store, assignments, jobs=2)


def test_fit_does_not_hash_the_whole_store_per_worker(tmp_path, monkeypatch) -> None:
  """A fit must not validate every shard before reading its own range.

  Store construction hashes every shard.  The fit used to build one store per
  worker, so a ``jobs=J`` fit hashed the whole store ``J`` times (and once even
  at ``jobs=1``).  The single range reuses the caller's store and each parallel
  worker opens a manifest-only store; a consumed shard is still hash-checked on
  read, so whole-store validation must be zero in both paths.
  """
  store = AppendOnlyShardStore(tmp_path / "store", max_rows_per_shard=10)
  for index in range(4):
    store.append(
      [
        make_row(
          tick,
          group=f"fam-{index}",
          episode=f"e{index}",
          run=f"r{index}",
          env=index,
        )
        for tick in range(41)
      ]
    )
  assignments = SplitAssignments({f"fam-{index}": "train" for index in range(4)})
  index = WindowIndex.build(store, assignments)
  assert store.shard_count == 20

  validations: list[int] = []
  original_validate = AppendOnlyShardStore._validate_manifest

  def counting_validate(self: Any) -> Any:
    validations.append(1)
    return original_validate(self)

  hashes: list[str] = []
  original_hash = storage_module.sha256_file

  def counting_hash(path: Any) -> Any:
    hashes.append(str(path))
    return original_hash(path)

  monkeypatch.setattr(AppendOnlyShardStore, "_validate_manifest", counting_validate)
  monkeypatch.setattr(storage_module, "sha256_file", counting_hash)

  fit_training_statistics(index, jobs=1)
  assert validations == []
  # Read-time hashing still covers every consumed shard exactly once.
  assert len(hashes) == store.shard_count
  hashes.clear()

  payload = (
    str(store.root),
    store.contract,
    CONTRACT,
    assignments,
    0,
    store.shard_count,
    16,
  )
  dataset_module._statistics_range_worker(payload)
  assert validations == []
  assert len(hashes) == store.shard_count


def test_high_window_count_fit_matches_the_scalar_reference(tmp_path) -> None:
  """Production-sized window counts stay within the explicit tolerance."""
  store = AppendOnlyShardStore(tmp_path / "store", max_rows_per_shard=97)
  rows = []
  for tick in range(600):
    yaw = 0.01 * tick
    rows.append(
      replace(
        make_row(tick, group="big", episode="big", run="big-run"),
        state=WorldState(
          np.array([0.01 * tick, np.sin(0.03 * tick), 0.5 + 0.001 * tick]),
          np.array([np.cos(yaw / 2), 0.0, np.sin(yaw / 2), 0.0]),
          np.array([0.02 * tick, 0.1 * np.cos(0.05 * tick), -0.01 * tick]),
          np.array([0.3 * np.sin(0.02 * tick), 0.1, 0.2 * np.cos(0.02 * tick)]),
          np.arange(60, dtype=np.float64).reshape(20, 3) * 0.01,
          np.ones((20, 3)) * (0.2 + 0.001 * tick),
        ),
        latent=np.linspace(-1.0, 1.0, 32) * (1.0 + tick),
      )
    )
  store.append(rows)
  index = WindowIndex.build(store, SplitAssignments({"big": "train"}))
  assert len(index.refs_for("train")) == 600 - 41 + 1

  bundle = fit_training_statistics(index)
  reference = _scalar_reference(index, CONTRACT)

  _assert_same_statistics(bundle, reference)
  assert bundle.state_stats.count == 41 * len(index.refs_for("train"))


def test_segment_starting_above_tick_zero_is_owned_once(tmp_path) -> None:
  """Ownership follows the first stored row, not the tick-zero sentinel.

  The previous rule treated ``row.tick == 0`` as the start of a segment, so a
  segment whose first tick was not zero could be skipped.  The current rule
  hands the segment to the range that holds its first stored row.
  """
  store = AppendOnlyShardStore(tmp_path / "store", max_rows_per_shard=17)
  store.append(
    [make_row(tick, group="fam", episode="a", run="run-a") for tick in range(10, 70)]
  )
  assignments = SplitAssignments({"fam": "train"})
  index = WindowIndex.build(store, assignments)
  assert [ref.start_tick for ref in index.refs] == list(range(10, 30))

  whole = list(
    iter_owned_segment_rows(store, shard_start=0, shard_stop=store.shard_count)
  )
  assert [key for key, _ in whole] == [("run-a", 0, "a", "segment")]
  assert len(whole[0][1]) == 60

  for jobs in (2, 3, 5):
    parallel, how = build_window_index(store, assignments, jobs=jobs)
    assert how.startswith("parallel")
    assert parallel.refs == index.refs

  _assert_same_statistics(
    fit_training_statistics(index), _scalar_reference(index, CONTRACT)
  )


def test_fit_refuses_an_interleaved_segment_run(tmp_path) -> None:
  """A segment stored as two runs around another segment is refused.

  The earlier range stopped at the interleaved shard and the later range
  treated the tail as a foreign segment, so the two prefixes could silently
  drop the segment's windows.  The manifest scan now refuses any segment whose
  stored rows are not one contiguous run.
  """
  store = AppendOnlyShardStore(tmp_path / "store", max_rows_per_shard=10)
  store.append(
    [make_row(tick, group="fam", episode="a", run="run-a") for tick in range(20)]
  )
  store.append(
    [make_row(tick, group="fam", episode="b", run="run-b") for tick in range(20)]
  )
  store.append(
    [make_row(tick, group="fam", episode="a", run="run-a") for tick in range(20, 41)]
  )
  assignments = SplitAssignments({"fam": "train"})
  # The sequential build groups the two runs into one 41-row segment.
  index = WindowIndex.build(store, assignments)
  assert len(index.refs) == 1

  with pytest.raises(DatasetError, match="more than one run"):
    fit_training_statistics(index)
  with pytest.raises(DatasetError, match="more than one run"):
    build_window_index(store, assignments, jobs=2)


def test_read_ahead_hands_a_new_segment_to_its_owner(tmp_path) -> None:
  """A range that reads ahead to finish a long segment must not steal the next."""
  store = AppendOnlyShardStore(tmp_path / "store", max_rows_per_shard=10)
  for index, (episode, run) in enumerate((("a", "run-a"), ("b", "run-b"))):
    store.append(
      [
        make_row(tick, group=f"fam-{episode}", episode=episode, run=run, env=index)
        for tick in range(41)
      ]
    )
  assert store.shard_count == 10
  expected = [("run-a", 0, "a", "segment"), ("run-b", 1, "b", "segment")]

  seen: list[Any] = []
  for start, stop in ((0, 3), (3, 5), (5, 10)):
    for key, rows in iter_owned_segment_rows(store, shard_start=start, shard_stop=stop):
      assert [row.tick for row in rows] == list(range(41))
      seen.append(key)
  assert seen == expected

  first_range = list(iter_owned_segment_rows(store, shard_start=0, shard_stop=3))
  assert [key for key, _ in first_range] == [expected[0]]
  later = list(iter_owned_segment_rows(store, shard_start=5, shard_stop=10))
  assert [key for key, _ in later] == [expected[1]]


def test_moment_accumulators_refuse_nonfinite_and_mismatched_input() -> None:
  with pytest.raises(ProjectionError, match="non-finite"):
    FeatureMomentsAccumulator(3).update(np.full((2, 3), np.inf))
  with pytest.raises(ProjectionError, match="non-finite"):
    FeatureMoments(1, np.full(3, np.nan), np.zeros(3))
  with pytest.raises(ProjectionError, match="width"):
    FeatureMomentsAccumulator(3).absorb(FeatureMoments(1, np.zeros(4), np.zeros(4)))
  with pytest.raises(ProjectionError, match="empty"):
    combine_feature_moments([FeatureMoments(0, np.zeros(3), np.zeros(3))])


def test_combined_partials_are_reduced_in_iteration_order() -> None:
  rng = np.random.default_rng(7)
  values = rng.normal(size=(64, 5)) * 3.0 + 1.0
  whole = fit_feature_stats([values], std_floor=1e-6)

  accumulator = FeatureMomentsAccumulator(5)
  for chunk in (values[:12], values[12:40], values[40:]):
    accumulator.update(chunk)
  np.testing.assert_allclose(accumulator.finish(1e-6).mean, whole.mean, rtol=1e-14)

  partials = [_partials(chunk) for chunk in (values[:12], values[12:40], values[40:])]
  combined = combine_feature_moments(partials, std_floor=1e-6)
  np.testing.assert_allclose(combined.mean, whole.mean, rtol=1e-14)
  np.testing.assert_allclose(combined.std, whole.std, rtol=1e-14)
  again = combine_feature_moments(partials, std_floor=1e-6)
  assert again.mean.tolist() == combined.mean.tolist()
  assert again.std.tolist() == combined.std.tolist()


def _partials(values: np.ndarray) -> FeatureMoments:
  centered = values - values.mean(axis=0)
  return FeatureMoments(
    len(values), values.mean(axis=0), np.sum(centered * centered, axis=0)
  )
