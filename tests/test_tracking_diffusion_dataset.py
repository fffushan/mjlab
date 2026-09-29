from dataclasses import replace

import numpy as np
from tracking_diffusion_fixtures import make_row, make_state

from mjlab.tasks.tracking.diffusion import (
  AppendOnlyShardStore,
  SplitAssignments,
  WindowDataset,
  WindowIndex,
  fit_training_statistics,
  grouped_split,
)
from mjlab.tasks.tracking.diffusion.dataset import _row_group


def test_grouping_is_deterministic_and_windows_are_lazy(tmp_path) -> None:
  store = AppendOnlyShardStore(tmp_path, max_rows_per_shard=17)
  rows = [make_row(i, group="family-a", episode="a") for i in range(41)]
  rows += [make_row(i, group="family-b", episode="b") for i in range(41)]
  store.append(rows)
  first = grouped_split(store.iter_rows())
  second = grouped_split(store.iter_rows())
  assert first.as_dict() == second.as_dict()
  assert first.split_for("family-a") == first.split_for("family-a")
  index = WindowIndex.build(
    store, SplitAssignments({"family-a": "train", "family-b": "test"})
  )
  assert index.coverage() == {"train": 1, "validation": 0, "test": 1}
  bundle = fit_training_statistics(index)
  sample = WindowDataset(index, "train", bundle)[0]
  assert sample.tokens.shape == (41, 231)
  assert sample.current_index == 8


def test_windows_never_cross_run_or_environment_boundaries(tmp_path) -> None:
  store = AppendOnlyShardStore(tmp_path)
  rows = [make_row(i, run="run-a", env=0) for i in range(21)]
  rows += [
    make_row(i + 21, run="run-b", env=1, timestamp=0.02 * (i + 21)) for i in range(20)
  ]
  store.append(rows)
  index = WindowIndex.build(store, SplitAssignments({"group": "train"}))
  assert len(index) == 0


def test_irregular_physical_timestamps_remain_eligible(tmp_path) -> None:
  store = AppendOnlyShardStore(tmp_path)
  rows = [
    make_row(i, timestamp=i * 0.02 + (0.001 if i > 10 else 0.0)) for i in range(41)
  ]
  store.append(rows)
  index = WindowIndex.build(store, SplitAssignments({"group": "train"}))
  assert len(index) == 1


def test_paired_and_duplicate_collector_groups_share_one_split() -> None:
  def collector_provenance(pair_id: str, phase: str) -> dict[str, str]:
    return {
      "pair_id": pair_id,
      "initial_state_id": "init-7",
      "reference_phase": "2",
      "phase": phase,
      "group_key_generated": "true",
    }

  family = [
    make_row(
      0,
      group="motion:3",
      run="clean-run",
      env=0,
      episode="clean-episode",
      provenance=collector_provenance("pair-a", "clean"),
    ),
    make_row(
      1,
      group="motion:3",
      run="ou-run",
      env=1,
      episode="ou-episode",
      provenance=collector_provenance("pair-a", "ou"),
    ),
    make_row(
      2,
      group="motion:7",
      run="duplicate-run-a",
      env=2,
      episode="duplicate-episode-a",
      provenance=collector_provenance("pair-b", "clean"),
    ),
    make_row(
      3,
      group="motion:11",
      run="duplicate-run-b",
      env=3,
      episode="duplicate-episode-b",
      provenance=collector_provenance("pair-c", "clean"),
    ),
  ]
  unrelated = make_row(
    4,
    group="motion:19",
    run="unrelated-run",
    env=4,
    episode="unrelated-episode",
    provenance={
      "pair_id": "pair-unrelated",
      "initial_state_id": "init-other",
      "reference_phase": "2",
      "phase": "clean",
      "group_key_generated": "true",
    },
  )
  rows = family + [unrelated]
  assert len({(row.run_id, row.env_id) for row in rows}) == len(rows)

  assignments = grouped_split(rows)
  family_splits = {assignments.split_for(_row_group(row)) for row in family}
  assert len(family_splits) == 1
  assert len(assignments.group_to_split) >= 2


def test_explicit_group_key_does_not_acquire_provenance_aliases() -> None:
  rows = [
    make_row(
      0,
      group="explicit-a",
      provenance={
        "pair_id": "pair-a",
        "initial_state_id": "init-7",
        "reference_phase": "2",
      },
    ),
    make_row(
      1,
      group="explicit-b",
      run="run-b",
      env=1,
      episode="episode-b",
      provenance={
        "pair_id": "pair-b",
        "initial_state_id": "init-7",
        "reference_phase": "2",
      },
    ),
  ]
  assignments = grouped_split(rows)
  assert assignments.split_for("explicit-a") != assignments.split_for("explicit-b")


def test_statistics_ignore_far_validation_and_test_values(tmp_path) -> None:
  store = AppendOnlyShardStore(tmp_path)
  train = [make_row(i, group="train", episode="train") for i in range(41)]
  validation = [
    replace(
      make_row(i, group="validation", episode="validation", run="run-v"),
      state=make_state(i, angular_velocity=1.0e6),
      latent=np.full(32, 1.0e6),
    )
    for i in range(41)
  ]
  test = [
    replace(
      make_row(i, group="test", episode="test", run="run-t"),
      state=make_state(i, angular_velocity=-2.0e6),
      latent=np.full(32, -2.0e6),
    )
    for i in range(41)
  ]
  store.append(train + validation + test)
  index = WindowIndex.build(
    store,
    SplitAssignments({"train": "train", "validation": "validation", "test": "test"}),
  )
  bundle = fit_training_statistics(index)

  train_only_store = AppendOnlyShardStore(tmp_path / "train-only")
  train_only_store.append(train)
  train_only_index = WindowIndex.build(
    train_only_store, SplitAssignments({"train": "train"})
  )
  train_only_bundle = fit_training_statistics(train_only_index)
  assert bundle.statistics_sha256 == train_only_bundle.statistics_sha256
  np.testing.assert_allclose(
    bundle.state_stats.mean, train_only_bundle.state_stats.mean
  )
  np.testing.assert_allclose(bundle.state_stats.std, train_only_bundle.state_stats.std)
  np.testing.assert_allclose(
    bundle.latent_stats.mean, train_only_bundle.latent_stats.mean
  )
  np.testing.assert_allclose(
    bundle.latent_stats.std, train_only_bundle.latent_stats.std
  )
  train_latent = train[20].latent
  assert train_latent is not None
  np.testing.assert_allclose(bundle.latent_stats.mean, train_latent, atol=1.0)
  assert np.max(np.abs(bundle.latent_stats.mean)) < 2.0


def test_windows_reject_missing_terminal_evidence_and_boundaries(tmp_path) -> None:
  store = AppendOnlyShardStore(tmp_path)
  rows = [make_row(i) for i in range(41)]
  rows[20] = type(rows[20])(
    **{
      **{field: getattr(rows[20], field) for field in rows[20].__dataclass_fields__},
      "terminal_evidence": None,
    }
  )
  store.append(rows)
  index = WindowIndex.build(store, SplitAssignments({"group": "train"}))
  assert len(index) == 0


def test_train_statistics_do_not_require_validation_or_test_data(tmp_path) -> None:
  store = AppendOnlyShardStore(tmp_path)
  store.append([make_row(i, group="train") for i in range(41)])
  index = WindowIndex.build(store, SplitAssignments({"train": "train"}))
  bundle = fit_training_statistics(index)
  np.testing.assert_allclose(bundle.state_stats.mean[0], 0.0, atol=10.0)


def test_window_rows_are_resolved_with_one_store_scan(tmp_path) -> None:
  """Reading many windows must not re-scan the store once per window.

  ``fit_training_statistics`` resolves every train window; the earlier
  implementation called ``store.iter_rows()`` per window, which is quadratic in
  the window count and re-decompressed every shard.
  """
  store = AppendOnlyShardStore(tmp_path, max_rows_per_shard=31)
  rows = [make_row(i, group="family-a", episode="a") for i in range(60)]
  store.append(rows)
  assignments = SplitAssignments({"family-a": "train"})
  index = WindowIndex.build(store, assignments)
  assert len(index) > 1

  scans = 0
  original = store.iter_rows

  def counting_iter_rows():
    nonlocal scans
    scans += 1
    yield from original()

  store.iter_rows = counting_iter_rows  # type: ignore[method-assign]
  bundle = fit_training_statistics(index)
  assert bundle.matrix_sha256
  assert scans == 1
