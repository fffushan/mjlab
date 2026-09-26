"""CPU lifecycle/checkpoint tests for bounded M3 resume semantics."""

from __future__ import annotations

import pytest
import torch

from mjlab.tasks.tracking.distillation.adapter import (
  DistillationSnapshot,
  DistillationStep,
)
from mjlab.tasks.tracking.distillation.checkpoint import (
  CheckpointValidationError,
  load_checkpoint,
  load_inference_checkpoint,
  save_checkpoint,
)
from mjlab.tasks.tracking.distillation.collector import DAggerCollector
from mjlab.tasks.tracking.distillation.model import ConditionalVAE
from mjlab.tasks.tracking.distillation.observations import (
  ObservationSnapshot,
  PackedObservationBatch,
  pack_observations,
)
from mjlab.tasks.tracking.distillation.runner import DistillationRunner, RunnerConfig
from mjlab.tasks.tracking.distillation.storage import (
  LabeledReplayBatch,
  LabeledReplayBuffer,
  ReplayValidationError,
)
from mjlab.tasks.tracking.distillation.trainer import (
  FreshTrainingData,
  TrainerPoisonedError,
  VaeDistillationTrainer,
)
from mjlab.tasks.tracking.distillation.training_config import TrainingConfig
from mjlab.tasks.tracking.distillation.vae_config import (
  DEFAULT_SCHEMA,
  ModelSettings,
  make_schema,
)


def make_batch(size: int = 8, seed: int = 2) -> LabeledReplayBatch:
  generator = torch.Generator().manual_seed(seed)
  reference = torch.randn(size, 68, generator=generator)
  conditioning = torch.randn(size, 99, generator=generator)
  action = torch.randn(size, 31, generator=generator)
  ids = torch.arange(size, dtype=torch.int64)
  return LabeledReplayBatch(
    PackedObservationBatch(reference, conditioning, DEFAULT_SCHEMA),
    action,
    ids,
    ids + 10,
    ids + 20,
    ids + 30,
    ids + 40,
  )


def _runner_snapshot(step: int) -> DistillationSnapshot:
  values = torch.full((2, 31), float(step))
  features = ObservationSnapshot(
    reference_q=values,
    reference_dq=values + 1,
    anchor_orientation_error=torch.zeros(2, 6),
    projected_gravity=torch.zeros(2, 3),
    gyro=torch.zeros(2, 3),
    relative_joint_q=values + 2,
    joint_dq=values + 3,
    previous_action=values - 1,
  )
  return DistillationSnapshot(
    teacher_observation=torch.zeros(2, 164),
    features=features,
    packed=pack_observations(features),
    teacher_id="fake",
    teacher_code=0,
    motion_id=torch.zeros(2, dtype=torch.int64),
    reference_frame=torch.full((2,), step, dtype=torch.int64),
    segment_id=torch.zeros(2, dtype=torch.int64),
    generation_id=torch.zeros(2, dtype=torch.int64),
  )


class _RunnerTeacher:
  action_dim = 31

  def label(self, observations: torch.Tensor) -> torch.Tensor:
    return torch.ones(observations.shape[0], 31)


class _RunnerAdapter:
  auto_reset = True

  def __init__(self) -> None:
    self.current = _runner_snapshot(0)
    self.step_calls = 0
    self.reset_calls = 0

  def reset(self, seed: int | None = None) -> DistillationSnapshot:
    del seed
    self.reset_calls += 1
    self.current = _runner_snapshot(0)
    return self.current

  def step(self, action: torch.Tensor) -> DistillationStep:
    assert torch.isfinite(action).all()
    self.step_calls += 1
    self.current = _runner_snapshot(self.step_calls)
    return DistillationStep(
      self.current,
      torch.zeros(2),
      torch.zeros(2, dtype=torch.bool),
      torch.zeros(2, dtype=torch.bool),
      {},
    )


def make_trainer(model: ConditionalVAE | None = None) -> VaeDistillationTrainer:
  model = model or ConditionalVAE(DEFAULT_SCHEMA, ModelSettings(hidden_dims=(8, 8)))
  replay = LabeledReplayBuffer(16, DEFAULT_SCHEMA)
  replay.insert(make_batch())
  return VaeDistillationTrainer(
    model,
    replay,
    TrainingConfig(accumulation_steps=1, minibatch_size=4),
    seed=11,
  )


def test_replay_state_roundtrip_preserves_wrapped_and_empty_storage() -> None:
  replay = LabeledReplayBuffer(3, DEFAULT_SCHEMA)
  replay.insert(make_batch(2))
  replay.insert(make_batch(2, seed=4))
  state = replay.state_dict()
  restored = LabeledReplayBuffer(3, DEFAULT_SCHEMA)
  restored.load_state_dict(state)
  first = replay.sample(
    3, replacement=False, generator=torch.Generator().manual_seed(8)
  )
  second = restored.sample(
    3, replacement=False, generator=torch.Generator().manual_seed(8)
  )
  torch.testing.assert_close(first.reference, second.reference)
  torch.testing.assert_close(first.episode_id, second.episode_id)
  assert sorted(
    restored.sample(
      3, replacement=False, generator=torch.Generator().manual_seed(8)
    ).episode_id.tolist()
  ) == sorted(first.episode_id.tolist())
  assert restored.rebase_segment_ids(40, 100) == 1
  rebased = restored.state_dict()["storage"]["episode_id"]
  assert int(rebased.max().item()) >= 130

  empty = LabeledReplayBuffer(3, DEFAULT_SCHEMA)
  empty_state = empty.state_dict()
  empty_restored = LabeledReplayBuffer(3, DEFAULT_SCHEMA)
  empty_restored.load_state_dict(empty_state)
  assert empty_restored.is_empty
  bad = dict(empty_state)
  bad["size"] = 1
  with pytest.raises(ReplayValidationError, match="no storage"):
    empty_restored.load_state_dict(bad)
  assert empty_restored.is_empty


def test_checkpoint_roundtrip_has_deterministic_next_update(tmp_path) -> None:
  torch.manual_seed(7)
  trainer = make_trainer()
  trainer.begin_training(FreshTrainingData(make_batch(), "initial"))
  trainer.train_update()
  path = tmp_path / "state.pt"
  save_checkpoint(
    path,
    trainer,
    trainer.replay,
    counters={"iteration": 3},
    schedule={"next": 4},
    resolved_config={"device": "cpu"},
    teacher_hashes={"tennis_000": "abc"},
    control_contract={"period": 0.02},
  )

  restored = make_trainer()
  state = load_checkpoint(
    path,
    restored,
    restored.replay,
    expected_teacher_hashes={"tennis_000": "abc"},
    expected_control_contract={"period": 0.02},
  )
  assert state.counters == {"iteration": 3}
  next_a = trainer.train_update()
  next_b = restored.train_update()
  assert next_a.total_loss == pytest.approx(next_b.total_loss, abs=1e-7)
  assert next_a.kl == pytest.approx(next_b.kl, abs=1e-7)
  for left, right in zip(
    trainer.model.parameters(), restored.model.parameters(), strict=True
  ):
    torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-7)


def test_checkpoint_contract_rejection_precedes_mutation(tmp_path) -> None:
  trainer = make_trainer()
  path = tmp_path / "state.pt"
  save_checkpoint(path, trainer, trainer.replay, teacher_hashes={"t": "one"})
  before = {
    key: value.detach().clone()
    for key, value in trainer.model.state_dict().items()
    if isinstance(value, torch.Tensor)
  }
  with pytest.raises(CheckpointValidationError, match="hashes"):
    load_checkpoint(path, trainer, trainer.replay, expected_teacher_hashes={"t": "two"})
  after = {
    key: value
    for key, value in trainer.model.state_dict().items()
    if isinstance(value, torch.Tensor)
  }
  assert all(torch.equal(before[key], after[key]) for key in before)


def test_invalid_two_stream_rng_state_is_rejected_before_mutation(tmp_path) -> None:
  trainer = make_trainer()
  path = tmp_path / "rng.pt"
  save_checkpoint(path, trainer, trainer.replay)
  payload = torch.load(path, weights_only=True)
  payload["rng"]["trainer"]["latent"] = torch.zeros(1, dtype=torch.uint8)
  torch.save(payload, path)
  before = trainer.generator_states()
  with pytest.raises(CheckpointValidationError, match="latent RNG state"):
    load_checkpoint(path, trainer, trainer.replay)
  after = trainer.generator_states()
  torch.testing.assert_close(before["replay"], after["replay"])
  torch.testing.assert_close(before["latent"], after["latent"])


def test_checkpoint_restore_normalizes_stored_rng_dtypes_before_install(
  tmp_path, monkeypatch
) -> None:
  """Only the normalized CPU uint8 state may be installed by a restore.

  ``torch.random.set_rng_state`` (and ``Generator.set_state``) accept nothing
  but a CPU byte vector, so a state read by ``torch.load(map_location=...)``
  onto another device fails.  The CPU-reproducible half of that defect is a
  state stored under a wider dtype: the previous code validated a CPU copy and
  then installed the raw payload tensor instead.
  """
  trainer = make_trainer()
  collector = DAggerCollector(
    _RunnerAdapter(), _RunnerTeacher(), trainer.model, trainer.replay
  )
  path = tmp_path / "rng-dtypes.pt"
  save_checkpoint(path, trainer, trainer.replay, collector=collector)
  payload = torch.load(path, weights_only=True)
  expected = {
    "global_cpu": payload["rng"]["global_cpu"].clone(),
    "replay": payload["rng"]["trainer"]["replay"].clone(),
    "latent": payload["rng"]["trainer"]["latent"].clone(),
    "collector": payload["rng"]["collector"].clone(),
  }
  payload["rng"]["global_cpu"] = expected["global_cpu"].view(torch.int32)
  payload["rng"]["trainer"] = {
    "replay": expected["replay"].view(torch.int16),
    "latent": expected["latent"].view(torch.bool),
  }
  payload["rng"]["collector"] = expected["collector"].view(torch.int32)
  for state in (
    payload["rng"]["global_cpu"],
    payload["rng"]["trainer"]["replay"],
    payload["rng"]["trainer"]["latent"],
    payload["rng"]["collector"],
  ):
    assert state.dtype is not torch.uint8
  torch.save(payload, path)

  installed: list[torch.Tensor] = []
  original_set_rng_state = torch.random.set_rng_state

  def spy(state: torch.Tensor) -> None:
    installed.append(state)
    original_set_rng_state(state)

  monkeypatch.setattr(torch.random, "set_rng_state", spy)

  restored = make_trainer()
  restored_collector = DAggerCollector(
    _RunnerAdapter(), _RunnerTeacher(), restored.model, restored.replay
  )
  load_checkpoint(path, restored, restored.replay, collector=restored_collector)

  assert len(installed) == 1
  normalized = installed[0]
  assert normalized.dtype is torch.uint8
  assert normalized.device.type == "cpu"
  torch.testing.assert_close(normalized, expected["global_cpu"])
  torch.testing.assert_close(torch.random.get_rng_state(), expected["global_cpu"])
  torch.testing.assert_close(restored.replay_generator.get_state(), expected["replay"])
  torch.testing.assert_close(restored.latent_generator.get_state(), expected["latent"])
  torch.testing.assert_close(
    restored_collector.generator_state(), expected["collector"]
  )
  torch.testing.assert_close(
    torch.randn(8, generator=trainer.replay_generator),
    torch.randn(8, generator=restored.replay_generator),
  )


@pytest.mark.parametrize(
  "stored",
  [
    None,
    [0, 1, 2, 3],
    torch.zeros(4, dtype=torch.uint8),
    torch.zeros(4, dtype=torch.float32),
    torch.zeros(2, 2, dtype=torch.uint8),
  ],
  ids=["none", "list", "too-short", "wrong-bytes", "two-dimensional"],
)
def test_checkpoint_restore_rejects_malformed_rng_states(tmp_path, stored) -> None:
  """Normalization must not accept a payload that is not a real state."""
  trainer = make_trainer()
  path = tmp_path / "malformed-rng.pt"
  save_checkpoint(path, trainer, trainer.replay)
  payload = torch.load(path, weights_only=True)
  payload["rng"]["global_cpu"] = stored
  torch.save(payload, path)
  before = trainer.generator_states()
  with pytest.raises(CheckpointValidationError, match="global RNG state"):
    load_checkpoint(path, trainer, trainer.replay)
  after = trainer.generator_states()
  torch.testing.assert_close(before["replay"], after["replay"])
  torch.testing.assert_close(before["latent"], after["latent"])


def _batch_on_device(
  batch: LabeledReplayBatch, device: torch.device
) -> LabeledReplayBatch:
  return LabeledReplayBatch(
    PackedObservationBatch(
      batch.reference.to(device), batch.conditioning.to(device), batch.schema
    ),
    batch.teacher_action.to(device),
    batch.motion_id.to(device),
    batch.teacher_id.to(device),
    batch.reference_frame.to(device),
    batch.episode_id.to(device),
    batch.collector_iteration.to(device),
  )


def make_device_trainer(device: torch.device) -> VaeDistillationTrainer:
  """Build the trainer/replay on ``device`` so replay metadata matches it."""
  model = ConditionalVAE(DEFAULT_SCHEMA, ModelSettings(hidden_dims=(8, 8))).to(device)
  replay = LabeledReplayBuffer(16, DEFAULT_SCHEMA, device=device, dtype=torch.float32)
  replay.insert(_batch_on_device(make_batch(), device))
  return VaeDistillationTrainer(
    model,
    replay,
    TrainingConfig(accumulation_steps=1, minibatch_size=4),
    seed=11,
  )


@pytest.mark.skipif(
  not torch.cuda.is_available(),
  reason="CUDA-mapped resume requires a GPU device",
)
def test_cuda_mapped_checkpoint_restore_normalizes_rng_states(tmp_path) -> None:
  """A GPU checkpoint read onto CUDA must still restore CPU RNG state."""
  saved_global_rng = torch.random.get_rng_state()
  device = torch.device("cuda", torch.cuda.current_device())
  trainer = make_device_trainer(device)
  path = tmp_path / "cuda-rng.pt"
  save_checkpoint(path, trainer, trainer.replay)
  expected_global = torch.load(path, weights_only=True)["rng"]["global_cpu"].clone()
  expected_replay = trainer.generator_states()["replay"]
  raw = torch.load(path, map_location=device, weights_only=True)
  assert raw["rng"]["global_cpu"].device.type == "cuda"
  assert raw["rng"]["trainer"]["replay"].device.type == "cuda"

  restored = make_device_trainer(device)
  load_checkpoint(path, restored, restored.replay, map_location=device)
  installed = torch.random.get_rng_state()
  assert installed.dtype is torch.uint8
  assert installed.device.type == "cpu"
  torch.testing.assert_close(installed, expected_global)
  torch.testing.assert_close(restored.replay_generator.get_state(), expected_replay)
  torch.random.set_rng_state(saved_global_rng)


def test_checkpoint_is_atomic_on_invalid_replay_state(tmp_path) -> None:
  trainer = make_trainer()
  path = tmp_path / "state.pt"
  save_checkpoint(path, trainer, trainer.replay)
  payload = torch.load(path, weights_only=True)
  payload["replay"]["size"] = 99
  torch.save(payload, path)
  before = trainer.replay.state_dict()
  with pytest.raises(CheckpointValidationError, match="replay size"):
    load_checkpoint(path, trainer, trainer.replay)
  assert trainer.replay.size == before["size"]


def test_bounded_runner_bootstrap_collect_update_cycle() -> None:
  adapter = _RunnerAdapter()
  replay = LabeledReplayBuffer(16, DEFAULT_SCHEMA)
  student = ConditionalVAE(DEFAULT_SCHEMA, ModelSettings(hidden_dims=(8, 8)))
  trainer = VaeDistillationTrainer(
    student,
    replay,
    TrainingConfig(accumulation_steps=1, minibatch_size=2),
    seed=5,
  )
  collector = DAggerCollector(adapter, _RunnerTeacher(), student, replay)
  runner = DistillationRunner(
    collector,
    trainer,
    RunnerConfig(
      max_iterations=2,
      bootstrap_steps=2,
      collection_steps=1,
      updates_per_iteration=1,
      teacher_probability=0.0,
    ),
  )
  results = runner.run()
  assert len(results) == 2
  assert results[0].collection is not None
  assert results[0].collection.samples == 4
  assert len(results[0].updates) == 1
  assert adapter.reset_calls == 1
  assert runner.iteration == 2
  assert len(runner.events) == 2


def test_resume_restarts_fake_simulator_and_rebases_segments(tmp_path) -> None:
  adapter = _RunnerAdapter()
  replay = LabeledReplayBuffer(16, DEFAULT_SCHEMA)
  student = ConditionalVAE(DEFAULT_SCHEMA, ModelSettings(hidden_dims=(8, 8)))
  trainer = VaeDistillationTrainer(
    student,
    replay,
    TrainingConfig(accumulation_steps=1, minibatch_size=2),
    seed=5,
  )
  collector = DAggerCollector(adapter, _RunnerTeacher(), student, replay)
  runner = DistillationRunner(
    collector,
    trainer,
    RunnerConfig(max_iterations=3, bootstrap_steps=1, collection_steps=1),
  )
  runner.run(iterations=1)
  old_state = replay.state_dict()
  old_storage = old_state["storage"]
  assert old_storage is not None
  old_size = int(old_state["size"])
  old_start = (int(old_state["next"]) - old_size) % replay.capacity
  old_indices = (torch.arange(old_size) + old_start) % replay.capacity
  old_max = int(old_storage["episode_id"][old_indices].max().item())
  path = tmp_path / "resume.pt"
  runner.save(path)

  new_adapter = _RunnerAdapter()
  new_replay = LabeledReplayBuffer(16, DEFAULT_SCHEMA)
  new_student = ConditionalVAE(DEFAULT_SCHEMA, ModelSettings(hidden_dims=(8, 8)))
  new_trainer = VaeDistillationTrainer(
    new_student,
    new_replay,
    TrainingConfig(accumulation_steps=1, minibatch_size=2),
    seed=5,
  )
  new_collector = DAggerCollector(
    new_adapter, _RunnerTeacher(), new_student, new_replay
  )
  resumed = DistillationRunner(
    new_collector,
    new_trainer,
    RunnerConfig(max_iterations=3, collection_steps=1),
  )
  resumed.resume(path)
  result = resumed.run(iterations=1)[0]
  assert result.resumed_reset
  assert new_adapter.reset_calls == 1
  new_state = new_replay.state_dict()
  new_storage = new_state["storage"]
  assert new_storage is not None
  new_size = int(new_state["size"])
  new_start = (int(new_state["next"]) - new_size) % new_replay.capacity
  new_indices = (torch.arange(new_size) + new_start) % new_replay.capacity
  new_max = int(new_storage["episode_id"][new_indices].max().item())
  assert new_max > old_max


def test_runner_config_is_bounded_and_cpu_safe() -> None:
  assert RunnerConfig(max_iterations=1, collection_steps=0).max_iterations == 1
  with pytest.raises(ValueError):
    RunnerConfig(max_iterations=1, teacher_probability=2.0)
  with pytest.raises(ValueError):
    RunnerConfig(max_iterations=1, evaluate_every=1, evaluation_steps=0)


def _active_max_episode_id(replay: LabeledReplayBuffer) -> int:
  """Return the largest episode ID among the *valid* FIFO records only."""
  state = replay.state_dict()
  storage = state["storage"]
  assert storage is not None
  size = int(state["size"])
  start = (int(state["next"]) - size) % replay.capacity
  physical = (torch.arange(size) + start) % replay.capacity
  return int(storage["episode_id"][physical].max().item())


def test_periodic_evaluation_with_steps_keeps_collection_collectable() -> None:
  """evaluate_every=1 with evaluation_steps>0 must not break iteration 1.

  Evaluation advances the shared adapter and invalidates the collector
  snapshot with ``requires_reset=True``.  The pre-fix runner passed
  reset=False on the next iteration and raised
  ``CollectionNumericalError: ... reset=True is required``.
  """
  adapter = _RunnerAdapter()
  replay = LabeledReplayBuffer(16, DEFAULT_SCHEMA)
  student = ConditionalVAE(DEFAULT_SCHEMA, ModelSettings(hidden_dims=(8, 8)))
  trainer = VaeDistillationTrainer(
    student,
    replay,
    TrainingConfig(accumulation_steps=1, minibatch_size=2),
    seed=5,
  )
  collector = DAggerCollector(adapter, _RunnerTeacher(), student, replay)
  runner = DistillationRunner(
    collector,
    trainer,
    RunnerConfig(
      max_iterations=2,
      collection_steps=1,
      updates_per_iteration=1,
      teacher_probability=0.0,
      evaluate_every=1,
      evaluation_steps=1,
    ),
  )
  results = runner.run()
  assert results[0].evaluation is not None
  assert results[1].evaluation is not None
  assert results[1].collection is not None
  assert results[1].collection.ticks == 1
  assert results[1].collection.samples == 2
  assert len(results[1].updates) == 1
  assert not results[1].resumed_reset


def test_repeated_resume_namespaces_advance_above_valid_replay_records(
  tmp_path,
) -> None:
  """Partial-FIFO namespaces stay above valid records across two resumes."""

  def build(bootstrap_steps: int) -> DistillationRunner:
    adapter = _RunnerAdapter()
    replay = LabeledReplayBuffer(8, DEFAULT_SCHEMA)
    student = ConditionalVAE(DEFAULT_SCHEMA, ModelSettings(hidden_dims=(8, 8)))
    trainer = VaeDistillationTrainer(
      student,
      replay,
      TrainingConfig(accumulation_steps=1, minibatch_size=2),
      seed=5,
    )
    collector = DAggerCollector(adapter, _RunnerTeacher(), student, replay)
    return DistillationRunner(
      collector,
      trainer,
      RunnerConfig(
        max_iterations=6,
        bootstrap_steps=bootstrap_steps,
        collection_steps=1,
      ),
    )

  first = build(bootstrap_steps=1)
  first.run(iterations=1)
  assert first.replay.size == 2
  assert _active_max_episode_id(first.replay) == 0
  cycle_one = tmp_path / "cycle-one.pt"
  first.save(str(cycle_one))

  # Inactive ring slots are not valid records.  A namespace computed from the
  # raw ring instead of the active window would jump to this poisoned value.
  payload = torch.load(cycle_one, weights_only=True)
  payload["replay"]["storage"]["episode_id"][first.replay.size :] = 10**6
  torch.save(payload, cycle_one)

  second = build(bootstrap_steps=0)
  second.resume(str(cycle_one))
  assert second.iteration == 1
  second.run(iterations=1)
  assert second._segment_namespace == 1
  max_after_second = _active_max_episode_id(second.replay)
  assert max_after_second == 1
  assert max_after_second < 10**6
  cycle_two = tmp_path / "cycle-two.pt"
  second.save(str(cycle_two))

  third = build(bootstrap_steps=0)
  third.resume(str(cycle_two))
  assert third.iteration == 2
  third.run(iterations=1)
  assert third._segment_namespace == 2
  max_after_third = _active_max_episode_id(third.replay)
  assert max_after_third == 2
  assert max_after_third > max_after_second


def test_inference_loader_restores_non_default_model_without_training_state(
  tmp_path,
) -> None:
  torch.manual_seed(3)
  model = ConditionalVAE(DEFAULT_SCHEMA, ModelSettings(hidden_dims=(12, 6)))
  replay = LabeledReplayBuffer(24, DEFAULT_SCHEMA)
  replay.insert(make_batch())
  trainer = VaeDistillationTrainer(
    model,
    replay,
    TrainingConfig(
      accumulation_steps=3,
      minibatch_size=4,
      learning_rate=1e-3,
      beta=0.3,
    ),
    seed=9,
  )
  trainer.begin_training(FreshTrainingData(make_batch(), "initial"))
  trainer.train_update()
  collector = DAggerCollector(_RunnerAdapter(), _RunnerTeacher(), model, replay)
  train_shaped = tmp_path / "train-shaped.pt"
  save_checkpoint(
    train_shaped,
    trainer,
    replay,
    counters={"iteration": 5, "segment_namespace": 2},
    schedule={"max_iterations": 8},
    resolved_config={"trainer": "non-default"},
    teacher_hashes={"tennis_000": "hash-a"},
    control_contract={"control_period_s": 0.02},
    collector=collector,
  )

  # A training checkpoint legitimately carries optimizer/replay/RNG/trainer
  # settings; deleting them must not make model-only inference impossible.
  payload = torch.load(train_shaped, weights_only=True)
  for training_only in ("optimizer", "trainer_config", "replay", "rng"):
    del payload[training_only]
  model_only = tmp_path / "model-only.pt"
  torch.save(payload, model_only)

  inference = load_inference_checkpoint(
    model_only,
    device="cpu",
    expected_schema=DEFAULT_SCHEMA,
    expected_teacher_hashes={"tennis_000": "hash-a"},
    expected_control_contract={"control_period_s": 0.02},
  )
  assert inference.settings == ModelSettings(hidden_dims=(12, 6))
  assert (
    inference.schema.compatibility_metadata() == DEFAULT_SCHEMA.compatibility_metadata()
  )
  assert inference.counters == {"iteration": 5, "segment_namespace": 2}
  assert inference.schedule == {"max_iterations": 8}
  assert inference.resolved_config == {"trainer": "non-default"}
  assert inference.teacher_hashes == {"tennis_000": "hash-a"}
  assert not hasattr(inference, "optimizer")
  assert not inference.model.training
  for name, value in trainer.model.state_dict().items():
    restored = inference.model.state_dict()[name]
    if isinstance(value, torch.Tensor):
      torch.testing.assert_close(value, restored)
    else:
      assert value == restored
  torch.testing.assert_close(
    trainer.model.reference_normalizer.state_dict()["m2"],
    inference.model.reference_normalizer.state_dict()["m2"],
  )
  batch = make_batch(size=5, seed=11)
  expected = trainer.model.mean_inference(batch.reference, batch.conditioning)
  actual = inference.model.mean_inference(batch.reference, batch.conditioning)
  torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-6)


def test_inference_loader_rejects_mismatched_contracts(tmp_path) -> None:
  trainer = make_trainer()
  path = tmp_path / "train-shaped.pt"
  save_checkpoint(
    path,
    trainer,
    trainer.replay,
    teacher_hashes={"tennis_000": "one"},
    control_contract={"period": 0.02},
  )
  with pytest.raises(CheckpointValidationError, match="hashes"):
    load_inference_checkpoint(path, expected_teacher_hashes={"tennis_000": "two"})
  with pytest.raises(CheckpointValidationError, match="control contract"):
    load_inference_checkpoint(path, expected_control_contract={"period": 0.01})
  with pytest.raises(CheckpointValidationError, match="schema"):
    load_inference_checkpoint(path, expected_schema=make_schema("anchor"))
  inference = load_inference_checkpoint(
    path,
    expected_teacher_hashes={"tennis_000": "one"},
    expected_control_contract={"period": 0.02},
  )
  assert (
    inference.schema.compatibility_metadata() == DEFAULT_SCHEMA.compatibility_metadata()
  )


def test_inference_loader_rejects_non_finite_weights(tmp_path) -> None:
  trainer = make_trainer()
  path = tmp_path / "train-shaped.pt"
  save_checkpoint(path, trainer, trainer.replay)
  payload = torch.load(path, weights_only=True)
  payload["model"]["action_head.weight"].fill_(float("nan"))
  torch.save(payload, path)
  with pytest.raises(CheckpointValidationError, match="non-finite"):
    load_inference_checkpoint(path)


def test_poisoned_trainer_save_is_refused_and_cleared_only_by_valid_restore(
  tmp_path, monkeypatch
) -> None:
  trainer = make_trainer()
  trainer.begin_training(FreshTrainingData(make_batch(), "initial"))
  trainer.train_update()
  good = tmp_path / "good.pt"
  save_checkpoint(good, trainer, trainer.replay, teacher_hashes={"t": "one"})

  original_step = trainer.optimizer.step

  def poison_step(*args: object, **kwargs: object) -> None:
    original_step(*args, **kwargs)
    with torch.no_grad():
      next(trainer.model.parameters()).fill_(float("nan"))

  monkeypatch.setattr(trainer.optimizer, "step", poison_step)
  with pytest.raises(TrainerPoisonedError):
    trainer.train_update()
  assert trainer.poisoned

  rejected = tmp_path / "rejected.pt"
  with pytest.raises(TrainerPoisonedError, match="poisoned"):
    save_checkpoint(rejected, trainer, trainer.replay)
  assert not rejected.exists()

  adapter = _RunnerAdapter()
  collector = DAggerCollector(adapter, _RunnerTeacher(), trainer.model, trainer.replay)
  runner = DistillationRunner(collector, trainer)
  runner_rejected = tmp_path / "runner-rejected.pt"
  with pytest.raises(TrainerPoisonedError, match="poisoned"):
    runner.save(str(runner_rejected))
  assert not runner_rejected.exists()

  # A rejected load must neither mutate state nor clear poison.
  with pytest.raises(CheckpointValidationError, match="hashes"):
    load_checkpoint(good, trainer, trainer.replay, expected_teacher_hashes={"t": "two"})
  assert trainer.poisoned

  payload = torch.load(good, weights_only=True)
  payload["optimizer"]["state"][0]["exp_avg"].fill_(float("nan"))
  corrupt = tmp_path / "corrupt-optimizer.pt"
  torch.save(payload, corrupt)
  with pytest.raises(CheckpointValidationError):
    load_checkpoint(corrupt, trainer, trainer.replay)
  assert trainer.poisoned

  load_checkpoint(good, trainer, trainer.replay)
  assert not trainer.poisoned
  assert trainer.is_healthy
  trainer.assert_healthy()
  assert torch.isfinite(next(trainer.model.parameters())).all().item()
