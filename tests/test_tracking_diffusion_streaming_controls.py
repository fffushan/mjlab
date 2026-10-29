"""CPU fake-environment tests for batched collection streaming controls."""

from __future__ import annotations

import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from tracking_diffusion_fixtures import make_state

from mjlab.tasks.tracking.diffusion import (
  ActionContract,
  AppendOnlyShardStore,
  CollectionResult,
  CollectorError,
  DiffusionCollector,
  FrozenVaePolicy,
  PostStepBundleBatch,
  QualificationConfig,
  StepEvidence,
  StepEvidenceBatch,
  TrialSpec,
  VaeObservationBatch,
  WorldState,
)
from mjlab.tasks.tracking.diffusion.adapter import VaeObservationTensorBatch
from mjlab.tasks.tracking.diffusion.collection_controls import (
  BoundedRowPublisher,
  CollectionControlError,
  CollectionControls,
  CollectionLimits,
  CollectionProgress,
  CollectionSummary,
  limits_from_request,
)


class _Schema:
  reference_dim = 68
  conditioning_dim = 99


class _Model(torch.nn.Module):
  """Deterministic CPU VAE stand-in for the frozen gravity schema."""

  schema = _Schema()

  def encode(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    latent = reference[:, :1].repeat(1, 32)
    return latent, torch.zeros_like(latent)

  def decode(self, latent: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
    return latent[:, :1].repeat(1, 31) + conditioning[:, :1] * 0.0


@dataclass(frozen=True)
class _Snapshot:
  """Clean initial-state payload observed through a weak reference.

  ``generation`` stands in for the simulator initial state: every reset picks a
  new value, so a restored snapshot is observable in the rows.
  """

  generation: int


@dataclass(frozen=True)
class _DefaultRow:
  """Standing padding row served by the fake environment."""

  reference: np.ndarray
  state: WorldState


class _FakeBatchEnv:
  """Batch-protocol stand-in that records actions, snapshots and resets.

  ``clip_frames`` is the clip length of every fake motion, so a rollout ends on
  the post-step row that reaches the last frame, exactly like the real runtime.
  """

  def __init__(self, num_envs: int, *, clip_frames: int = 3) -> None:
    self.num_envs = num_envs
    self.clip_frames = clip_frames
    self.motion_ids: list[str] = [""] * num_envs
    self.step_counts = np.zeros(num_envs, dtype=np.int64)
    self.generation = np.zeros(num_envs, dtype=np.int64)
    self.active = np.ones(num_envs, dtype=bool)
    self.terminate_at: dict[int, int] = {}
    self.step_actions: list[np.ndarray] = []
    self.events: list[str] = []
    self.active_masks: list[np.ndarray] = []
    self.vae_observation_ids: list[np.ndarray] = []
    self.captured = 0
    self.restored = 0
    self.snapshot_refs: list[weakref.ReferenceType[_Snapshot]] = []
    self.snapshot_alive_at_restore: list[bool] = []

  def set_active_env_ids(self, env_ids: Any) -> None:
    self.active = np.zeros(self.num_envs, dtype=bool)
    self.active[np.asarray(env_ids, dtype=np.int64)] = True
    self.active_masks.append(self.active.copy())

  def reset_batch(self, specs: Sequence[Any], *, env_ids: Any) -> None:
    for slot, spec in zip(np.asarray(env_ids).tolist(), specs, strict=True):
      self.motion_ids[slot] = spec.motion_id
      self.step_counts[slot] = 0
      self.generation[slot] += 1

  def _reference_phase(self, slots: Sequence[int]) -> np.ndarray:
    return np.asarray([0.25 * int(self.generation[slot]) for slot in slots])

  def _vae_batch(self, slots: Sequence[int]) -> VaeObservationBatch:
    return VaeObservationBatch(
      WorldState.stack(make_state(self.step_counts[slot]) for slot in slots),
      np.stack([np.full(68, 0.25 + 0.5 * slot) for slot in slots]),
      np.stack([np.full(99, 0.5) for slot in slots]),
      tuple(self.motion_ids[slot] for slot in slots),
      np.asarray([int(self.step_counts[slot]) for slot in slots]),
      self._reference_phase(slots),
      anchor_z_error=np.zeros(len(slots)),
      gravity_z_error=np.zeros(len(slots)),
      end_effector_z_error=np.zeros(len(slots)),
    )

  def observe_vae_batch(self, env_ids: Any) -> VaeObservationBatch:
    ids = np.asarray(env_ids, dtype=np.int64)
    self.vae_observation_ids.append(ids.copy())
    return self._vae_batch(ids.tolist())

  def default_row(self, motion_id: str) -> _DefaultRow:
    del motion_id
    return _DefaultRow(
      reference=np.full(68, 0.5),
      state=WorldState(
        np.array([0.0, 0.0, 0.66]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        np.zeros(3),
        np.zeros((20, 3)),
        np.zeros((20, 3)),
      ),
    )

  def _evidence(self, slot: int) -> StepEvidence:
    boundary = self.step_counts[slot] >= self.clip_frames
    limit = self.terminate_at.get(slot)
    terminated = limit is not None and self.step_counts[slot] >= limit
    return StepEvidence(
      True,
      terminated=terminated,
      truncated=terminated,
      reference_boundary=boundary,
      clip_ended=boundary,
    )

  def post_step_evidence_batch(self, env_ids: Any) -> PostStepBundleBatch:
    slots = np.asarray(env_ids, dtype=np.int64).tolist()
    return PostStepBundleBatch(
      StepEvidenceBatch.from_values([self._evidence(slot) for slot in slots]),
      np.asarray([float(self.step_counts[slot]) for slot in slots]),
      self._vae_batch(slots),
    )

  def step_batch(self, action: np.ndarray) -> None:
    values = np.asarray(action, dtype=np.float64)
    assert values.shape == (self.num_envs, 31)
    assert np.all(values[~self.active] == 0.0), "idle rows must not be stepped"
    self.events.append("step")
    self.step_actions.append(values.copy())
    self.step_counts[self.active] += 1

  def capture_initial_state_batch(self, env_ids: Any) -> tuple[_Snapshot, ...]:
    slots = np.asarray(env_ids, dtype=np.int64).tolist()
    snapshots = tuple(_Snapshot(int(self.generation[slot])) for slot in slots)
    self.captured += len(snapshots)
    self.snapshot_refs.extend(weakref.ref(snapshot) for snapshot in snapshots)
    return snapshots

  def restore_initial_state_batch(
    self, env_ids: Any, snapshots: Sequence[_Snapshot]
  ) -> None:
    slots = np.asarray(env_ids, dtype=np.int64).tolist()
    for slot, snapshot in zip(slots, tuple(snapshots), strict=True):
      self.snapshot_alive_at_restore.append(
        any(reference() is snapshot for reference in self.snapshot_refs)
      )
      self.generation[slot] = snapshot.generation
      self.restored += 1

  def reset_after_trial_batch(self, env_ids: Any) -> None:
    ids = np.asarray(env_ids, dtype=np.int64).tolist()
    self.events.append(f"reset{[int(value) for value in ids]}")


class _TensorBatchEnv(_FakeBatchEnv):
  """Batch-protocol stand-in that also serves the device tensor API.

  The device observation is the base :class:`VaeObservationTensorBatch`, so the
  collector exercises ``FrozenVaePolicy.batch_action_tensor`` directly and the
  row-evidence half of the same capture.
  """

  def __init__(self, num_envs: int, *, clip_frames: int = 3) -> None:
    super().__init__(num_envs, clip_frames=clip_frames)
    self.tensor_observation_ids: list[np.ndarray] = []
    self.tensor_steps: list[np.ndarray] = []
    self.tensor_devices: list[torch.device] = []

  def observe_vae_tensor_batch(self, env_ids: Any) -> VaeObservationTensorBatch:
    ids = np.asarray(env_ids, dtype=np.int64)
    self.tensor_observation_ids.append(ids.copy())
    batch = self._vae_batch(ids.tolist())
    return VaeObservationTensorBatch(
      torch.as_tensor(batch.reference, dtype=torch.float32),
      torch.as_tensor(batch.conditioning, dtype=torch.float32),
      batch.state,
      batch.motion_id,
      batch.reference_frame,
      batch.reference_phase,
    )

  def step_tensor_batch(self, action: torch.Tensor) -> None:
    assert isinstance(action, torch.Tensor)
    assert action.shape == (self.num_envs, 31)
    values = action.detach().to("cpu").numpy().copy()
    assert np.all(values[~self.active] == 0.0), "idle rows must not be stepped"
    self.events.append("tensor_step")
    self.tensor_steps.append(values)
    self.tensor_devices.append(action.device)
    self.step_counts[self.active] += 1


class _RecordingStore:
  """Store wrapper that records append sizes over a real shard store."""

  def __init__(self, root: Path, *, max_rows_per_shard: int = 64) -> None:
    self.inner = AppendOnlyShardStore(root, max_rows_per_shard=max_rows_per_shard)
    self.calls: list[int] = []
    self.rows: list[Any] = []

  @property
  def root(self) -> Path:
    return self.inner.root

  def append(self, rows: Any) -> tuple[str, ...]:
    values = tuple(rows)
    self.calls.append(len(values))
    self.rows.extend(values)
    return self.inner.append(values)

  def iter_rows(self) -> Any:
    return self.inner.iter_rows()


class _ProgressLog:
  """Collect every progress record the collector emits."""

  def __init__(self) -> None:
    self.records: list[CollectionProgress] = []

  def __call__(self, record: CollectionProgress) -> None:
    self.records.append(record)


class _SteppingClock:
  """Monotonic clock that advances one tick per read."""

  def __init__(self, tick: float) -> None:
    self.tick = tick
    self.now = 0.0

  def __call__(self) -> float:
    value = self.now
    self.now += self.tick
    return value


def _trial(
  motion: str,
  *,
  phase: str = "clean",
  seed: int = 7,
  pair: str = "",
  initial_state: str = "",
  run: str = "run",
) -> TrialSpec:
  return TrialSpec(
    run,
    motion,
    seed,
    3,
    phase,
    pair_id=pair,
    initial_state_id=initial_state,
  )


def _pair_specs(motion: str) -> tuple[TrialSpec, TrialSpec]:
  return (
    _trial(motion, pair="pair", initial_state="state"),
    _trial(motion, phase="ou", pair="pair", initial_state="state"),
  )


def _policy() -> FrozenVaePolicy:
  contract = ActionContract(np.ones(31), np.zeros(31), np.ones(31), np.ones(31))
  return FrozenVaePolicy(_Model(), contract)


def _collector(
  env: Any,
  store: Any,
  *,
  padding_steps: int = 0,
) -> DiffusionCollector:
  return DiffusionCollector(
    env,
    _policy(),
    qualification=QualificationConfig(padding_steps=padding_steps),
    store=store,
  )


def _collect(
  root: Path,
  specs: Sequence[TrialSpec],
  *,
  controls: CollectionControls | None = None,
  slots: int = 1,
  control_steps: int = 4,
  tensor: bool = False,
  terminate_at: dict[int, int] | None = None,
) -> tuple[Any, _RecordingStore, Any]:
  env = (_TensorBatchEnv if tensor else _FakeBatchEnv)(
    slots, clip_frames=control_steps
  )
  env.terminate_at.update(terminate_at or {})
  store = _RecordingStore(root)
  collector = _collector(env, store)
  return env, store, collector.collect_batched(specs, slots=slots, controls=controls)


_STATE_FIELDS = (
  "root_position",
  "root_quaternion_wxyz",
  "root_linear_velocity",
  "root_angular_velocity",
  "body_positions",
  "body_linear_velocities",
)
_ROW_ARRAY_FIELDS = (
  "previous_executed_action",
  "clean_action",
  "ou_noise",
  "executed_action",
  "latent",
)


def _assert_row_equal(left: Any, right: Any) -> None:
  assert left.identity == right.identity
  assert left.controller == right.controller
  assert (left.timestamp, left.motion_id, left.reference_frame) == (
    right.timestamp,
    right.motion_id,
    right.reference_frame,
  )
  assert (left.fps, left.group_key, left.segment_qualified) == (
    right.fps,
    right.group_key,
    right.segment_qualified,
  )
  assert (left.reset, left.teleport, left.reference_boundary) == (
    right.reset,
    right.teleport,
    right.reference_boundary,
  )
  assert left.provenance == right.provenance
  assert left.terminal_evidence is not None and right.terminal_evidence is not None
  assert left.terminal_evidence.as_dict() == right.terminal_evidence.as_dict()
  for field in _STATE_FIELDS:
    np.testing.assert_array_equal(
      getattr(left.state, field), getattr(right.state, field)
    )
  for field in _ROW_ARRAY_FIELDS:
    left_value = getattr(left, field)
    right_value = getattr(right, field)
    if left_value is None or right_value is None:
      assert left_value is None and right_value is None
      continue
    np.testing.assert_array_equal(left_value, right_value)


def _assert_rows_equal(left_rows: Sequence[Any], right_rows: Sequence[Any]) -> None:
  assert len(left_rows) == len(right_rows)
  for left, right in zip(left_rows, right_rows, strict=True):
    _assert_row_equal(left, right)


def test_streaming_coalesces_finished_trials_into_one_bounded_shard(tmp_path) -> None:
  specs = (_trial("a"), _trial("b"), _trial("c"))
  legacy_env, legacy_store, legacy = _collect(
    tmp_path / "legacy", specs, slots=1, control_steps=3
  )
  assert legacy_store.calls == [9]
  assert legacy.stop_reason is None and not legacy.partial

  log = _ProgressLog()
  env, store, batch = _collect(
    tmp_path / "streamed",
    specs,
    slots=1,
    control_steps=3,
    controls=CollectionControls(stream_rows=True, progress=log),
  )
  # Three finished trials coalesce into one full shard at the final flush.
  assert store.calls == [9]
  assert batch.store_files == ("shard-000000.npz",)
  assert len(batch.store_files) < len(specs)
  assert [len(result.store_files) for result in batch.results] == [0, 0, 1]
  # The final flush is the only publication, so only the final record counts
  # stored rows.
  assert [(record.event, record.stored_rows) for record in log.records] == [
    ("trial", 0),
    ("trial", 0),
    ("trial", 0),
    ("final", 9),
  ]
  assert len(store.rows) == 9
  assert batch.progress is not None and batch.progress.stored_rows == 9
  for left, right in zip(legacy.results, batch.results, strict=True):
    _assert_rows_equal(left.rows, right.rows)
    _assert_rows_equal(left.accepted_rows, right.accepted_rows)
    _assert_rows_equal(left.verification_rows, right.verification_rows)
    assert left.qualification == right.qualification


def test_publication_files_scale_with_row_chunks_not_trials(tmp_path) -> None:
  specs = tuple(_trial(name) for name in "abcdefgh")
  env, store, batch = _collect(
    tmp_path,
    specs,
    slots=1,
    control_steps=3,
    controls=CollectionControls(stream_rows=True, max_rows_per_append=6),
  )
  # Eight three-row trials make twenty-four rows: four six-row shards.
  assert store.calls == [6, 6, 6, 6]
  assert len(batch.store_files) == 4
  assert len(batch.store_files) < len(specs)
  assert len(store.rows) == 24
  assert max(store.calls) <= 6


def test_max_rows_per_append_is_an_upper_bound_across_trials(tmp_path) -> None:
  specs = tuple(_trial(name) for name in "abcdef")
  env, store, batch = _collect(
    tmp_path,
    specs,
    slots=1,
    control_steps=3,
    controls=CollectionControls(stream_rows=True, max_rows_per_append=4),
  )
  assert store.calls == [4, 4, 4, 4, 2]
  assert max(store.calls) <= 4
  assert len(batch.store_files) == 5
  assert len(store.rows) == 18
  assert batch.stop_reason is None


def test_streamed_rows_match_the_legacy_single_append_exactly(tmp_path) -> None:
  specs = _pair_specs("motion")
  legacy_env, legacy_store, legacy = _collect(
    tmp_path / "legacy", specs, slots=1, control_steps=5
  )
  env, store, batch = _collect(
    tmp_path / "streamed",
    specs,
    slots=1,
    control_steps=5,
    controls=CollectionControls(stream_rows=True),
  )
  assert legacy.stop_reason is None and batch.stop_reason is None
  assert len(legacy.results) == len(batch.results) == 2
  for left, right in zip(legacy.results, batch.results, strict=True):
    assert left.qualification == right.qualification
    assert left.stopped_reason == right.stopped_reason
    _assert_rows_equal(left.rows, right.rows)
    _assert_rows_equal(left.accepted_rows, right.accepted_rows)
    _assert_rows_equal(left.verification_rows, right.verification_rows)
  _assert_rows_equal(legacy_store.rows, store.rows)
  assert legacy_store.calls == [10]
  assert store.calls == [10]
  # The persisted shards carry the same rows as the legacy single append.
  _assert_rows_equal(tuple(legacy_store.iter_rows()), tuple(store.iter_rows()))
  assert sum(row.segment_qualified for row in store.iter_rows()) == sum(
    len(result.accepted_rows) for result in batch.results
  )
  # The seeded OU stream runs on every step of an OU rollout, and the clean
  # partner stays noise-free.
  ou_rows = batch.results[1].rows
  assert all(np.any(row.ou_noise != 0.0) for row in ou_rows)
  assert all(np.all(row.ou_noise == 0.0) for row in batch.results[0].rows)


def test_compact_results_keep_row_accounting_after_rows_are_released(tmp_path) -> None:
  specs = _pair_specs("motion")
  full_env, full_store, full = _collect(
    tmp_path / "full",
    specs,
    slots=1,
    control_steps=5,
    controls=CollectionControls(stream_rows=True),
  )
  compact_env, compact_store, compact = _collect(
    tmp_path / "compact",
    specs,
    slots=1,
    control_steps=5,
    controls=CollectionControls(stream_rows=True, compact_results=True),
  )
  assert all(isinstance(item, CollectionSummary) for item in compact.results)
  assert len(compact.results) == len(full.results) == 2
  for left, right in zip(full.results, compact.results, strict=True):
    assert right.qualification == left.qualification
    assert right.stopped_reason == left.stopped_reason
    assert right.store_files == left.store_files
    assert right.row_count == len(left.rows)
    assert right.accepted_row_count == len(left.accepted_rows)
    assert right.verification_row_count == len(left.verification_rows)
    assert right.row_count == (right.accepted_row_count + right.verification_row_count)
  qualified = next(item for item in compact.results if item.qualification.qualified)
  assert qualified.row_count == 5
  assert qualified.accepted_row_count == 5
  _assert_rows_equal(full_store.rows, compact_store.rows)


def test_wall_clock_limit_stops_before_more_physics_work(tmp_path) -> None:
  env, store, batch = _collect(
    tmp_path,
    (_trial("a"),),
    slots=1,
    control_steps=6,
    controls=CollectionControls(
      stream_rows=True,
      limits=CollectionLimits(wall_seconds=25.0),
      clock=_SteppingClock(10.0),
    ),
  )
  assert len(env.step_actions) == 2
  assert batch.stop_reason == "wall_clock_limit"
  assert batch.partial
  assert batch.unstarted_trials == 0
  assert store.calls == [2]
  assert len(store.rows) == 2
  assert len(batch.results) == 1
  result = batch.results[0]
  assert result.stopped_reason == "wall_clock_limit"
  assert not result.qualification.qualified
  assert result.qualification.label == "incomplete"
  assert batch.progress is not None
  assert batch.progress.stop_reason == "wall_clock_limit"
  assert batch.progress.trials_completed == 1
  assert batch.progress.transitions == 2
  assert batch.progress.stored_rows == 2
  assert batch.progress.final


def test_inactivity_limit_stops_a_silent_run(tmp_path) -> None:
  env, store, batch = _collect(
    tmp_path,
    (_trial("a"),),
    slots=1,
    control_steps=6,
    controls=CollectionControls(
      stream_rows=True,
      limits=CollectionLimits(inactivity_seconds=25.0),
      clock=_SteppingClock(10.0),
    ),
  )
  assert len(env.step_actions) == 2
  assert batch.stop_reason == "inactivity_limit"
  assert batch.results[0].stopped_reason == "inactivity_limit"
  assert len(store.rows) == 2


def test_inactivity_deadline_restarts_after_each_completed_trial(tmp_path) -> None:
  specs = (_trial("a"), _trial("b"), _trial("c"))
  env, store, batch = _collect(
    tmp_path,
    specs,
    slots=1,
    control_steps=1,
      controls=CollectionControls(
      stream_rows=True,
      limits=CollectionLimits(inactivity_seconds=25.0),
      clock=_SteppingClock(20.0),
    ),
  )
  assert batch.stop_reason is None
  assert not batch.partial
  assert len(batch.results) == 3
  assert all(result.qualification.qualified for result in batch.results)
  # The three one-row trials coalesce into one final flush.
  assert store.calls == [3]


def test_output_size_limit_stops_after_a_bounded_publication(tmp_path) -> None:
  env = _FakeBatchEnv(1)
  store = _RecordingStore(tmp_path)
  baseline = sum(
    path.stat().st_size for path in store.root.glob("**/*") if path.is_file()
  )
  collector = _collector(env, store)
  batch = collector.collect_batched(
    (_trial("a"), _trial("b")),
    slots=1,
    controls=CollectionControls(
      stream_rows=True,
      max_rows_per_append=3,
      limits=CollectionLimits(max_output_bytes=baseline + 64),
    ),
  )
  assert batch.stop_reason == "output_size_limit"
  assert batch.partial
  assert store.calls == [3]
  assert len(env.step_actions) == 3
  assert len(batch.results) == 2
  first = batch.results[0]
  assert isinstance(first, CollectionResult)
  assert len(first.rows) == 3
  assert first.qualification.qualified
  cut = batch.results[1]
  assert cut.stopped_reason == "output_size_limit"
  assert isinstance(cut, CollectionResult) and cut.rows == ()
  assert cut.qualification.steps == 0
  assert batch.unstarted_trials == 0
  assert batch.progress is not None
  assert batch.progress.stop_reason == "output_size_limit"
  assert batch.progress.stored_rows == 3
  assert batch.progress.stored_bytes is not None
  assert batch.progress.stored_bytes > baseline


def test_output_limit_stops_before_any_physics_work(tmp_path) -> None:
  env = _FakeBatchEnv(1, clip_frames=3)
  store = _RecordingStore(tmp_path)
  collector = _collector(env, store)
  batch = collector.collect_batched(
    (_trial("a"), _trial("b")),
    slots=1,
    controls=CollectionControls(
      stream_rows=True, limits=CollectionLimits(max_output_bytes=1)
    ),
  )
  assert env.step_actions == []
  assert store.calls == []
  assert batch.stop_reason == "output_size_limit"
  assert batch.partial
  assert [result.stopped_reason for result in batch.results] == ["output_size_limit"]
  assert batch.unstarted_trials == 1
  assert batch.progress is not None
  assert batch.progress.transitions == 0


def test_final_flush_output_overage_keeps_the_stop_verdict(tmp_path) -> None:
  env = _FakeBatchEnv(1, clip_frames=1)
  store = _RecordingStore(tmp_path)
  baseline = sum(
    path.stat().st_size for path in store.root.glob("**/*") if path.is_file()
  )
  collector = _collector(env, store)
  batch = collector.collect_batched(
    (_trial("a"), _trial("b"), _trial("c")),
    slots=1,
    controls=CollectionControls(
      stream_rows=True,
      limits=CollectionLimits(max_output_bytes=baseline + 64),
    ),
  )
  # The schedule completed, but the final flush exceeded the declared budget.
  assert all(result.qualification.qualified for result in batch.results)
  assert store.calls == [3]
  assert batch.stop_reason == "output_size_limit"
  assert batch.partial and batch.output_limit_exceeded
  assert batch.unstarted_trials == 0
  assert batch.stored_bytes is not None and batch.stored_bytes > baseline


def test_progress_shares_the_cached_stored_byte_measurement(tmp_path) -> None:
  measurements: list[int] = []

  def measure() -> int:
    value = 4096 * (len(measurements) + 1)
    measurements.append(value)
    return value

  log = _ProgressLog()
  env, store, batch = _collect(
    tmp_path,
    (_trial("a"), _trial("b"), _trial("c")),
    slots=1,
    control_steps=2,
    controls=CollectionControls(stream_rows=True, progress=log, output_bytes=measure),
  )
  assert batch.stop_reason is None
  # One measurement for the first event and one after the single final flush:
  # the per-trial events reuse the watcher's cached count instead of walking
  # the store again.
  assert len(measurements) == 2
  assert batch.progress is not None
  assert batch.progress.stored_bytes == measurements[-1]


def test_bounded_row_publisher_coalesces_and_flushes() -> None:
  class _ListStore:
    def __init__(self) -> None:
      self.calls: list[int] = []
      self.rows: list[str] = []

    def append(self, rows: Any) -> tuple[str, ...]:
      values = tuple(rows)
      self.calls.append(len(values))
      self.rows.extend(values)
      return (f"shard-{len(self.calls) - 1}.npz",)

  store = _ListStore()
  publisher = BoundedRowPublisher(store, max_rows_per_append=4)
  assert publisher.add(cast(Any, ("a", "b", "c"))) == ()
  assert publisher.pending_rows == 3
  assert publisher.stored_rows == 0
  assert publisher.add(cast(Any, ("d", "e"))) == ("shard-0.npz",)
  assert publisher.pending_rows == 1
  assert store.calls == [4]
  assert publisher.flush() == ("shard-1.npz",)
  assert publisher.pending_rows == 0
  assert store.calls == [4, 1]
  assert publisher.stored_rows == 5
  assert publisher.files == ["shard-0.npz", "shard-1.npz"]
  assert publisher.flush() == ()


def test_progress_reports_actual_published_counts(tmp_path) -> None:
  log = _ProgressLog()
  env, store, batch = _collect(
    tmp_path,
    _pair_specs("motion"),
    slots=1,
    control_steps=4,
    controls=CollectionControls(
      stream_rows=True, compact_results=True, progress=log, max_rows_per_append=2
    ),
  )
  assert [record.event for record in log.records] == ["trial", "trial", "final"]
  assert [record.final for record in log.records] == [False, False, True]
  assert [record.trials_completed for record in log.records] == [1, 2, 2]
  assert [record.stored_rows for record in log.records] == [4, 8, 8]
  assert log.records[-1] is batch.progress
  assert batch.progress is not None
  assert batch.progress.transitions == 8
  assert batch.progress.accepted_rows == 8
  assert batch.progress.store_files == batch.store_files
  assert batch.progress.stored_bytes is not None
  assert batch.progress.stored_bytes > 0


def test_progress_callback_failure_propagates_and_keeps_written_rows(
  tmp_path,
) -> None:
  class _ProgressFailure(RuntimeError):
    """A failing progress consumer must not be swallowed."""

  class _FailingLog(_ProgressLog):
    def __call__(self, record: CollectionProgress) -> None:
      super().__call__(record)
      if record.trials_completed == 2:
        raise _ProgressFailure("progress consumer failed")

  log = _FailingLog()
  env = _FakeBatchEnv(1, clip_frames=2)
  store = _RecordingStore(tmp_path)
  collector = _collector(env, store)
  with pytest.raises(_ProgressFailure, match="progress consumer failed"):
    collector.collect_batched(
      (_trial("a"), _trial("b"), _trial("c")),
      slots=1,
      controls=CollectionControls(
        stream_rows=True, progress=log, max_rows_per_append=2
      ),
    )
  assert store.calls == [2, 2]
  assert len(store.rows) == 4
  assert [record.trials_completed for record in log.records] == [1, 2]


@pytest.mark.parametrize(
  ("fault", "message"),
  ((OSError("store is full"), "store is full"), (None, "wrote no shard")),
)
def test_storage_faults_fail_closed(tmp_path, fault, message) -> None:
  class _FaultyStore(_RecordingStore):
    def append(self, rows: Any) -> tuple[str, ...]:
      if fault is not None:
        raise fault
      del rows
      return ()

  store = _FaultyStore(tmp_path)
  collector = _collector(_FakeBatchEnv(1, clip_frames=2), store)
  with pytest.raises((OSError, CollectionControlError), match=message):
    collector.collect_batched(
      (_trial("a"),),
      slots=1,
      controls=CollectionControls(stream_rows=True),
    )
  assert len(store.rows) == 0


def test_collection_controls_reject_incoherent_requests(tmp_path) -> None:
  with pytest.raises(CollectionControlError, match="compact_results requires"):
    CollectionControls(compact_results=True)
  with pytest.raises(CollectionControlError, match="max_rows_per_append requires"):
    CollectionControls(max_rows_per_append=4)
  with pytest.raises(CollectionControlError, match="max_rows_per_append must be"):
    CollectionControls(stream_rows=True, max_rows_per_append=0)
  with pytest.raises(CollectionControlError, match="max_output_bytes requires"):
    CollectionControls(limits=CollectionLimits(max_output_bytes=1024))
  with pytest.raises(CollectionControlError, match="wall_seconds must be positive"):
    CollectionLimits(wall_seconds=0.0)
  collector = DiffusionCollector(
    _FakeBatchEnv(1),
    _policy(),
    qualification=QualificationConfig(padding_steps=0),
  )
  with pytest.raises(CollectorError, match="needs a collector store"):
    collector.collect_batched(
      (_trial("a"),),
      slots=1,
      controls=CollectionControls(stream_rows=True),
    )


def test_output_limit_requires_a_measurable_store(tmp_path) -> None:
  class _OpaqueStore:
    def append(self, rows: Any) -> tuple[str, ...]:
      del rows
      return ()

  collector = _collector(_FakeBatchEnv(1, clip_frames=2), _OpaqueStore())
  with pytest.raises(CollectorError, match="needs a measurable store output"):
    collector.collect_batched(
      (_trial("a"),),
      slots=1,
      controls=CollectionControls(
        stream_rows=True, limits=CollectionLimits(max_output_bytes=1)
      ),
    )


def test_tensor_dispatch_requires_the_full_tensor_api(tmp_path) -> None:
  # The base FrozenVaePolicy now exposes batch_action_tensor, so this test
  # disables the device API explicitly to exercise the gate.
  class _NoTensorVae(FrozenVaePolicy):
    batch_action_tensor = None

  contract = ActionContract(np.ones(31), np.zeros(31), np.ones(31), np.ones(31))
  vae = _NoTensorVae(_Model(), contract)
  store = cast(Any, _RecordingStore(tmp_path))
  collector = DiffusionCollector(
    _FakeBatchEnv(1),
    vae,
    qualification=QualificationConfig(padding_steps=0),
    store=store,
  )
  with pytest.raises(CollectorError, match="tensor dispatch is unavailable") as error:
    collector.collect_batched(
      (_trial("a"),),
      slots=1,
      controls=CollectionControls(stream_rows=True, tensor_dispatch=True),
    )
  message = str(error.value)
  assert "env.observe_vae_tensor_batch" in message
  assert "env.step_tensor_batch" in message
  assert "vae.batch_action_tensor" in message


def test_tensor_dispatch_refuses_missing_row_evidence_before_step(tmp_path) -> None:
  class _InferenceOnlyEnv(_TensorBatchEnv):
    def observe_vae_tensor_batch(self, env_ids):
      batch = super().observe_vae_tensor_batch(env_ids)
      return VaeObservationTensorBatch(batch.reference, batch.conditioning)

  env = _InferenceOnlyEnv(1)
  store = _RecordingStore(tmp_path)
  collector = DiffusionCollector(
    env,
    _policy(),
    qualification=QualificationConfig(padding_steps=0),
    store=cast(Any, store),
  )
  with pytest.raises(CollectorError, match="lacks collection row evidence"):
    collector.collect_batched(
      (_trial("a"),),
      slots=1,
      controls=CollectionControls(stream_rows=True, tensor_dispatch=True),
    )
  assert env.tensor_steps == []
  assert store.rows == []


def test_tensor_dispatch_runs_the_base_frozen_vae_tensor_method(tmp_path) -> None:
  env = _TensorBatchEnv(1)
  store = _RecordingStore(tmp_path)
  vae = _policy()
  collector = DiffusionCollector(
    env,
    vae,
    qualification=QualificationConfig(padding_steps=0),
    store=cast(Any, store),
  )
  # The dispatch must exercise the base policy method, not a test subclass.
  assert type(vae) is FrozenVaePolicy
  assert type(vae).batch_action_tensor is FrozenVaePolicy.batch_action_tensor
  batch = collector.collect_batched(
    (_trial("a"),),
    slots=1,
    controls=CollectionControls(stream_rows=True, tensor_dispatch=True),
  )
  assert batch.stop_reason is None
  assert len(env.tensor_steps) == 3
  assert len(store.rows) == 3
  # The tensor path never asked for the full NumPy VAE observation.
  assert env.vae_observation_ids == []


def test_tensor_dispatch_matches_the_numpy_rows_and_noise(tmp_path) -> None:
  specs = _pair_specs("motion")
  numpy_env, numpy_store, numpy_batch = _collect(
    tmp_path / "numpy", specs, slots=1, control_steps=5
  )
  tensor_env, tensor_store, tensor_batch = _collect(
    tmp_path / "tensor",
    specs,
    slots=1,
    control_steps=5,
    tensor=True,
    controls=CollectionControls(stream_rows=True, tensor_dispatch=True),
  )
  assert numpy_batch.stop_reason is None and tensor_batch.stop_reason is None
  assert len(numpy_env.step_actions) == len(tensor_env.tensor_steps) == 10
  for expected, actual in zip(
    numpy_env.step_actions, tensor_env.tensor_steps, strict=True
  ):
    np.testing.assert_array_equal(expected, actual)
  assert all(device.type == "cpu" for device in tensor_env.tensor_devices)
  for left, right in zip(numpy_batch.results, tensor_batch.results, strict=True):
    assert left.qualification == right.qualification
    _assert_rows_equal(left.rows, right.rows)
    _assert_rows_equal(left.accepted_rows, right.accepted_rows)
    _assert_rows_equal(left.verification_rows, right.verification_rows)
  _assert_rows_equal(numpy_store.rows, tensor_store.rows)
  controllers = {
    row.controller for result in tensor_batch.results for row in result.rows
  }
  assert controllers == {"vae"}
  # Both dispatch modes selected the same rows for the VAE observation, and the
  # tensor path never requested the full NumPy VAE observation.
  assert [ids.tolist() for ids in tensor_env.tensor_observation_ids] == [
    ids.tolist() for ids in numpy_env.vae_observation_ids
  ]
  assert tensor_env.vae_observation_ids == []


def test_tensor_dispatch_keeps_row_identity_for_noncontiguous_slots(tmp_path) -> None:
  specs = tuple(_trial(name) for name in ("a", "b", "c", "d"))
  numpy_env, numpy_store, numpy_batch = _collect(
    tmp_path / "numpy",
    specs,
    slots=4,
    control_steps=3,
    terminate_at={1: 1, 3: 1},
  )
  tensor_env, tensor_store, tensor_batch = _collect(
    tmp_path / "tensor",
    specs,
    slots=4,
    control_steps=3,
    terminate_at={1: 1, 3: 1},
    tensor=True,
    controls=CollectionControls(stream_rows=True, tensor_dispatch=True),
  )
  assert [ids.tolist() for ids in tensor_env.tensor_observation_ids] == [
    [0, 1, 2, 3],
    [0, 2],
    [0, 2],
  ]
  assert [ids.tolist() for ids in numpy_env.vae_observation_ids] == [
    [0, 1, 2, 3],
    [0, 2],
    [0, 2],
  ]
  assert tensor_env.vae_observation_ids == []
  assert len(tensor_env.tensor_steps) == len(numpy_env.step_actions) == 3
  idle = tensor_env.tensor_steps[1][[1, 3]]
  assert np.all(idle == 0.0)
  assert np.any(tensor_env.tensor_steps[1][[0, 2]] != 0.0)
  for expected, actual in zip(
    numpy_env.step_actions, tensor_env.tensor_steps, strict=True
  ):
    np.testing.assert_array_equal(expected, actual)
  for left, right in zip(numpy_batch.results, tensor_batch.results, strict=True):
    assert left.qualification == right.qualification
    _assert_rows_equal(left.rows, right.rows)
  assert tensor_batch.stop_reason is None
  _assert_rows_equal(numpy_store.rows, tensor_store.rows)
  terminated = [
    result for result in tensor_batch.results if not result.qualification.qualified
  ]
  assert len(terminated) == 2
  assert all(
    result.qualification.reason == "early_termination" for result in terminated
  )


def test_transition_budget_reports_partial_completion(tmp_path) -> None:
  env = _FakeBatchEnv(1, clip_frames=3)
  store = _RecordingStore(tmp_path)
  collector = _collector(env, store)
  batch = collector.collect_batched(
    (_trial("a"), _trial("b")), slots=1, remaining_transitions=2
  )
  assert len(env.step_actions) == 2
  assert batch.stop_reason == "control_transition_budget"
  assert batch.partial
  assert batch.unstarted_trials == 1
  assert len(batch.results) == 1
  cut = batch.results[0]
  assert cut.stopped_reason == "control_transition_budget"
  assert isinstance(cut, CollectionResult)
  assert len(cut.rows) == 2
  assert store.calls == [2]
  assert batch.progress is not None
  assert batch.progress.transitions == 2


def test_pair_snapshots_are_released_after_the_last_scheduled_ou_trial(
  tmp_path,
) -> None:
  specs = (
    _trial("motion", pair="pair", initial_state="state"),
    _trial("motion", phase="ou", pair="pair", initial_state="state"),
    _trial("motion", phase="ou", pair="pair", initial_state="state", seed=8),
  )
  env, store, batch = _collect(tmp_path, specs, slots=1, control_steps=2)
  assert env.captured == 1
  assert env.restored == 2
  # Both OU trials could still use the snapshot while they ran.
  assert env.snapshot_alive_at_restore == [True, True]
  # The last scheduled OU use released it.
  assert all(reference() is None for reference in env.snapshot_refs)
  assert [len(result.rows) for result in batch.results] == [2, 2, 2]
  assert batch.stop_reason is None
  # Every OU trial started from the captured clean initial state, not from a
  # fresh reset: the generation counter is only restored by the snapshot.
  clean_first = batch.results[0].rows[0]
  assert clean_first.provenance["reference_phase"] == "0.25"
  for result in batch.results[1:]:
    first = result.rows[0]
    assert (
      first.provenance["reference_phase"] == clean_first.provenance["reference_phase"]
    )
    assert first.state.root_position.tolist() == (
      clean_first.state.root_position.tolist()
    )
  assert batch.results[2].rows[0].provenance["initial_seed"] == "8"


def test_streaming_preserves_strict_idle_reset(tmp_path) -> None:
  specs = (_trial("a"), _trial("b"))
  env, store, batch = _collect(
    tmp_path,
    specs,
    slots=2,
    control_steps=3,
    terminate_at={0: 1},
    controls=CollectionControls(stream_rows=True),
  )
  assert env.events == ["step", "reset[0]", "step", "step", "reset[1]"]
  assert store.calls == [4]
  assert np.all(env.step_actions[-1][0] == 0.0)
  assert np.any(env.step_actions[-1][1] != 0.0)
  assert [len(result.rows) for result in batch.results] == [1, 3]
  assert batch.results[0].qualification.reason == "early_termination"
  assert batch.results[1].qualification.qualified
  assert batch.stop_reason is None


def test_limits_from_request_converts_minutes_and_mebibytes() -> None:
  class _Request:
    max_wall_minutes = 2
    monitor_inactivity_seconds = 30
    max_output_mib = 3

  assert limits_from_request(_Request()) == CollectionLimits(
    wall_seconds=120.0,
    inactivity_seconds=30.0,
    max_output_bytes=3 * 1024 * 1024,
  )
