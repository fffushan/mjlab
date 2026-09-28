"""Version-2 cohort identity, checkpoint, and multi-teacher runner tests.

Pure CPU and simulator-free: the cohort identity is built from the real saved
X2 manifest and a real reference library (both load without an environment) plus
the audited tracked-body mapping, and the multi-teacher runner is driven by a
two-row mixed-slot adapter and a real ``TeacherBank`` of small synthetic actors.
The audited live-contract object is a stand-in that carries exactly the evidence
:func:`cohort_identity_from_adapter` consumes, so the identity guards are
exercised without compiling a robot.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from tracking_distillation_fixtures import make_actor

from mjlab.tasks.tracking.distillation.adapter import (
  DistillationSnapshot,
  DistillationStep,
)
from mjlab.tasks.tracking.distillation.balanced_storage import BalancedReplayBuffer
from mjlab.tasks.tracking.distillation.checkpoint import (
  CheckpointValidationError,
  load_checkpoint,
  load_cohort_checkpoint,
  load_cohort_member_inference,
  load_inference_checkpoint,
  save_checkpoint,
  save_cohort_checkpoint,
)
from mjlab.tasks.tracking.distillation.cohort_contract import (
  CohortContractError,
  CohortIdentity,
  ResetProvenance,
  build_cohort_identity,
  cohort_identity_from_adapter,
  require_same_cohort,
)
from mjlab.tasks.tracking.distillation.collector import (
  CollectionResult,
  DAggerCollector,
)
from mjlab.tasks.tracking.distillation.config import (
  ActorArchitecture,
  CohortContract,
  load_manifest,
  resolve_cohort,
)
from mjlab.tasks.tracking.distillation.environment import RuntimeSeedProvenance
from mjlab.tasks.tracking.distillation.model import ConditionalVAE
from mjlab.tasks.tracking.distillation.motion_library import BodySelection
from mjlab.tasks.tracking.distillation.multi_motion import (
  MotionSlotAllocation,
  MultiMotionPlan,
  plan_multi_motion,
)
from mjlab.tasks.tracking.distillation.observations import (
  ObservationSnapshot,
  pack_observations,
)
from mjlab.tasks.tracking.distillation.reset_policy import make_reset_policy
from mjlab.tasks.tracking.distillation.runner import DistillationRunner, RunnerConfig
from mjlab.tasks.tracking.distillation.storage import (
  LabeledReplayBatch,
  LabeledReplayBuffer,
)
from mjlab.tasks.tracking.distillation.teachers import (
  TeacherBank,
  build_frozen_teacher,
)
from mjlab.tasks.tracking.distillation.trainer import (
  FreshTrainingData,
  VaeDistillationTrainer,
)
from mjlab.tasks.tracking.distillation.training_config import TrainingConfig
from mjlab.tasks.tracking.distillation.vae_config import ModelSettings, make_schema

TEACHER_IDS = ("tennis_000", "tennis_001")
OBS_WIDTH = 164
ACTION_DIM = 31
SEED = 7
SEMANTIC_OVERRIDES = (
  "private multi-motion config: commands.motion.sampling_mode: 'adaptive' -> 'uniform'",
)


# Fixtures: the real saved cohort, loaded without a simulator.


@pytest.fixture(scope="module")
def real_cohort() -> CohortContract:
  return resolve_cohort(
    load_manifest("configs/distillation/x2_tennis.yaml", repo_root=Path("."))
  )


@pytest.fixture(scope="module")
def schema(real_cohort: CohortContract):
  """The live VAE schema: the saved X2 joint order, not a synthetic default."""
  return make_schema(joint_order=tuple(real_cohort.actions.joint_names))


@pytest.fixture(scope="module")
def plan(real_cohort: CohortContract) -> MultiMotionPlan:
  return plan_multi_motion(
    real_cohort,
    TEACHER_IDS,
    2,
    phase_policy="uniform",
    slot_generator=torch.Generator().manual_seed(SEED),
  )


@pytest.fixture(scope="module")
def body_selection(real_cohort: CohortContract, plan: MultiMotionPlan) -> BodySelection:
  """The audited tracked-body mapping of the two reference clips.

  Resolving this against the compiled robot needs a simulator; the identity
  contract only requires the audited mapping itself, so the tests supply the
  declared tracked bodies with the clips' own source-body count.
  """
  return BodySelection(
    indices=tuple(range(len(real_cohort.body_names))),
    source_body_count=plan.library.source_body_count,
    names=tuple(real_cohort.body_names),
  )


# Builders.


def make_replay(
  schema: Any,
  plan: MultiMotionPlan,
  *,
  capacity: int = 8,
  weights: dict[int, float] | None = None,
  frame_counts: dict[int, int] | None = None,
) -> BalancedReplayBuffer:
  clips = plan.library.clips
  return BalancedReplayBuffer(
    capacity,
    schema,
    weights if weights is not None else {clip.motion_id: 1.0 for clip in clips},
    teacher_codes={clip.motion_id: clip.teacher_code for clip in clips},
    frame_counts=(
      frame_counts
      if frame_counts is not None
      else {clip.motion_id: clip.frames for clip in clips}
    ),
  )


def make_identity(
  real_cohort: CohortContract,
  plan: MultiMotionPlan,
  replay: Any,
  body_selection: BodySelection | None,
  *,
  phase_policy: str = "uniform",
  mapping_digest: str | None = None,
  device: str = "cpu",
  requested_seed: int = SEED,
  effective_seed: int | None = SEED,
) -> CohortIdentity:
  return build_cohort_identity(
    real_cohort,
    plan.library,
    plan.slots,
    phase_policy=phase_policy,  # type: ignore[arg-type]
    replay=replay,
    device=device,
    requested_seed=requested_seed,
    effective_seed=effective_seed,
    semantic_overrides=SEMANTIC_OVERRIDES,
    body_selection=body_selection,
    mapping_digest=mapping_digest,
  )


def make_batch(
  schema: Any,
  plan: MultiMotionPlan,
  *,
  size: int = 2,
  seed: int = 2,
  episode_ids: tuple[int, ...] | None = None,
) -> LabeledReplayBatch:
  generator = torch.Generator().manual_seed(seed)
  motion = torch.arange(size, dtype=torch.int64) % len(plan.library.clips)
  return LabeledReplayBatch(
    pack_observations(
      ObservationSnapshot(
        reference_q=torch.randn(size, ACTION_DIM, generator=generator),
        reference_dq=torch.randn(size, ACTION_DIM, generator=generator),
        anchor_orientation_error=torch.normal(0, 1, (size, 3, 3), generator=generator),
        projected_gravity=torch.randn(size, 3, generator=generator),
        gyro=torch.randn(size, 3, generator=generator),
        relative_joint_q=torch.randn(size, ACTION_DIM, generator=generator),
        joint_dq=torch.randn(size, ACTION_DIM, generator=generator),
        previous_action=torch.randn(size, ACTION_DIM, generator=generator),
      ),
      schema,
    ),
    torch.randn(size, schema.action_dim, generator=generator),
    motion,
    torch.tensor(
      [plan.library.clip(int(item)).teacher_code for item in motion.tolist()]
    ),
    torch.zeros(size, dtype=torch.int64),
    torch.tensor(
      episode_ids if episode_ids is not None else (0,) * size, dtype=torch.int64
    ),
    torch.zeros(size, dtype=torch.int64),
  )


def make_cohort_trainer(
  schema: Any,
  plan: MultiMotionPlan,
  body_selection: BodySelection,
  real_cohort: CohortContract,
  *,
  capacity: int = 8,
  updates: int = 1,
  model_seed: int = 3,
  run_seed: int = 11,
) -> tuple[VaeDistillationTrainer, CohortIdentity]:
  torch.manual_seed(model_seed)
  model = ConditionalVAE(schema, ModelSettings(hidden_dims=(8, 8)))
  replay = make_replay(schema, plan, capacity=capacity)
  trainer = VaeDistillationTrainer(
    model,
    replay,
    TrainingConfig(accumulation_steps=1, minibatch_size=2),
    seed=run_seed,
  )
  batch = make_batch(schema, plan)
  replay.insert(batch)
  if updates:
    trainer.begin_training(FreshTrainingData(batch, "initial"))
    for _ in range(updates):
      trainer.train_update()
    trainer.freeze_normalizers()
  identity = make_identity(real_cohort, plan, replay, body_selection)
  return trainer, identity


def make_reset_provenance(plan: MultiMotionPlan) -> ResetProvenance:
  return ResetProvenance(
    reset_policy=make_reset_policy(
      kind="standing-mixture",
      standing_start_fraction=0.25,
      standing_start_window_frames=25,
      standing_start_frame_zero_fraction=0.5,
    ),
    effective_windows=tuple(min(25, clip.frames) for clip in plan.library.clips),
    boundary_semantics={
      "full_reset": "sample standing/reference once per affected row",
      "timer_resample": "reference-state teleport; no standing mixture",
      "reference_wrap": "reference-state teleport; no standing mixture",
      "reset_to_frame": "explicit reference-state teleport; no standing mixture",
    },
    standing_pose={
      "joint_position": [0.0] * ACTION_DIM,
      "root_height": 0.9,
      "yaw_alignment": "upright yaw from selected reference anchor quaternion",
      "joint_order": [f"joint_{index}" for index in range(ACTION_DIM)],
    },
    perturbations={
      "pose_range": [0.0] * 12,
      "velocity_range": [0.0] * 12,
      "joint_position_range": [-0.05, 0.05],
    },
    provenance={
      "command_type": "FakeMultiMotionCommand",
      "entity_name": "robot",
      "standing_pose_source": "robot.data.default_joint_pos/default_root_state",
      "yaw_source": "reference anchor quaternion",
    },
  )


def make_v3_trainer(
  schema: Any,
  plan: MultiMotionPlan,
  body_selection: BodySelection,
  real_cohort: CohortContract,
) -> tuple[VaeDistillationTrainer, CohortIdentity, FakeMultiAdapter, DAggerCollector]:
  adapter = FakeMultiAdapter(real_cohort, plan, body_selection, schema)
  replay = make_replay(schema, plan)
  batch = make_batch(schema, plan)
  batch = replace(
    batch,
    initialization_kind=torch.tensor([1, 2], dtype=torch.int64),
    segment_initial_reference_frame=torch.zeros(2, dtype=torch.int64),
    segment_age=torch.zeros(2, dtype=torch.int64),
  )
  replay.insert(batch)
  torch.manual_seed(3)
  model = ConditionalVAE(schema, ModelSettings(hidden_dims=(8, 8)))
  trainer = VaeDistillationTrainer(
    model,
    replay,
    TrainingConfig(accumulation_steps=1, minibatch_size=2),
    seed=11,
  )
  collector = DAggerCollector(adapter, make_bank(), model, replay)
  identity = make_identity(real_cohort, plan, replay, body_selection)
  return trainer, identity, adapter, collector


def make_architecture(obs_dim: int = OBS_WIDTH) -> ActorArchitecture:
  return ActorArchitecture(
    class_name="MLPModel",
    hidden_dims=(8, 4),
    activation="elu",
    obs_normalization=True,
    obs_groups=("actor",),
    distribution_class_name="GaussianDistribution",
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "std_type": "scalar",
      "init_std": 1.0,
    },
    obs_dim=obs_dim,
    action_dim=ACTION_DIM,
  )


def make_bank(seeds: tuple[int, ...] = (0, 5)) -> TeacherBank:
  return TeacherBank(
    [
      build_frozen_teacher(
        teacher_id,
        make_actor(OBS_WIDTH, ACTION_DIM, seed=seed).state_dict(),
        make_architecture(),
      )
      for teacher_id, seed in zip(TEACHER_IDS, seeds, strict=True)
    ],
    device="cpu",
  )


class FakeResetCommand:
  def __init__(self) -> None:
    self._generator = torch.Generator(device="cpu").manual_seed(41)

  def reset_rng_state(self) -> torch.Tensor:
    return self._generator.get_state().clone()

  def set_reset_rng_state(self, state: torch.Tensor) -> None:
    self._generator.set_state(state.clone())


class FakeMultiAdapter:
  """Two-row mixed-slot adapter with owned row state and no simulator.

  It exposes exactly what the collector consumes (owned per-row motion ids and
  teacher codes, aligned frames/segments, one post-step snapshot) plus the audit
  evidence :func:`cohort_identity_from_adapter` reads.
  """

  auto_reset = True

  def __init__(
    self,
    real_cohort: CohortContract,
    plan: MultiMotionPlan,
    body_selection: BodySelection,
    schema: Any,
    *,
    rows: int = 2,
    seed: int = SEED,
  ) -> None:
    self.cohort = real_cohort
    self.slots = plan.slots
    self.library = plan.library.with_body_selection(body_selection)
    self.audit = SimpleNamespace(
      slots=plan.slots,
      mapping_digest=plan.library.mapping_digest(),
      phase_policy="uniform",
      semantic_overrides=SEMANTIC_OVERRIDES,
      seed_provenance=RuntimeSeedProvenance(
        requested_seed=seed,
        effective_seed=seed,
        applied_before_construction=True,
      ),
    )
    self.reset_command = FakeResetCommand()
    self.env = SimpleNamespace(
      device=torch.device("cpu"),
      num_envs=rows,
      command_manager=SimpleNamespace(
        get_term=lambda name: self.reset_command if name == "motion" else None
      ),
    )
    self.schema = schema
    self.rows = rows
    self.reset_calls = 0
    self.step_calls = 0
    self._frame = 0
    self._segment = 0

  def _snapshot(self) -> DistillationSnapshot:
    motion = torch.arange(self.rows, dtype=torch.int64) % len(self.library.clips)
    codes = torch.tensor(
      [self.library.clip(int(item)).teacher_code for item in motion.tolist()]
    )
    values = torch.full((self.rows, ACTION_DIM), float(self._frame))
    features = ObservationSnapshot(
      reference_q=values,
      reference_dq=values + 1,
      anchor_orientation_error=torch.zeros(self.rows, 6),
      projected_gravity=torch.zeros(self.rows, 3),
      gyro=torch.zeros(self.rows, 3),
      relative_joint_q=values + 2,
      joint_dq=values + 3,
      previous_action=values - 1,
    )
    return DistillationSnapshot(
      teacher_observation=torch.zeros(self.rows, OBS_WIDTH),
      features=features,
      packed=pack_observations(features, self.schema),
      teacher_id="multiple",
      teacher_code=-1,
      motion_id=motion,
      reference_frame=torch.full((self.rows,), self._frame, dtype=torch.int64),
      segment_id=torch.full((self.rows,), self._segment, dtype=torch.int64),
      generation_id=torch.zeros(self.rows, dtype=torch.int64),
      teacher_codes=codes,
    )

  def reset(self, seed: int | None = None) -> DistillationSnapshot:
    del seed
    self.reset_calls += 1
    self._frame = 0
    self._segment += 1
    return self._snapshot()

  def step(self, action: torch.Tensor) -> DistillationStep:
    assert torch.isfinite(action).all()
    self.step_calls += 1
    self._frame += 1
    return DistillationStep(
      self._snapshot(),
      torch.zeros(self.rows),
      torch.zeros(self.rows, dtype=torch.bool),
      torch.zeros(self.rows, dtype=torch.bool),
      {},
    )

  def close(self) -> None:
    pass


def make_runner(
  real_cohort: CohortContract,
  plan: MultiMotionPlan,
  body_selection: BodySelection,
  schema: Any,
  *,
  capacity: int = 8,
  max_iterations: int = 4,
  model_seed: int = 3,
  run_seed: int = 5,
) -> tuple[DistillationRunner, FakeMultiAdapter]:
  adapter = FakeMultiAdapter(real_cohort, plan, body_selection, schema)
  torch.manual_seed(model_seed)
  model = ConditionalVAE(schema, ModelSettings(hidden_dims=(8, 8)))
  replay = make_replay(schema, plan, capacity=capacity)
  trainer = VaeDistillationTrainer(
    model,
    replay,
    TrainingConfig(accumulation_steps=1, minibatch_size=2),
    seed=run_seed,
  )
  collector = DAggerCollector(adapter, make_bank(), model, replay)
  runner = DistillationRunner(
    collector,
    trainer,
    RunnerConfig(
      max_iterations=max_iterations,
      bootstrap_steps=2,
      collection_steps=1,
      updates_per_iteration=1,
      seed=run_seed,
    ),
  )
  return runner, adapter


def assert_same_state(left: Any, right: Any, path: str = "state") -> None:
  """Compare nested checkpoint state, comparing tensors by value.

  A plain ``==`` on a nested snapshot would compare tensors elementwise and
  raise instead of reporting the difference.
  """
  if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
    assert isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor), path
    assert left.dtype == right.dtype and left.shape == right.shape, path
    assert torch.equal(left, right), path
    return
  if isinstance(left, dict) or isinstance(right, dict):
    assert isinstance(left, dict) and isinstance(right, dict), path
    assert set(left) == set(right), path
    for key in left:
      assert_same_state(left[key], right[key], f"{path}.{key}")
    return
  if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
    assert type(left) is type(right), path
    assert len(left) == len(right), path
    for index, (item, other) in enumerate(zip(left, right, strict=True)):
      assert_same_state(item, other, f"{path}[{index}]")
    return
  assert left == right, path


def assert_state_unchanged(
  trainer: VaeDistillationTrainer,
  replay: Any,
  collector: DAggerCollector | None,
  before: dict[str, Any],
) -> None:
  """Every owned state must be identical to the captured snapshot."""
  assert_same_state(
    {
      key: value.detach().clone() if isinstance(value, torch.Tensor) else value
      for key, value in trainer.model.state_dict().items()
    },
    before["model"],
    "model",
  )
  assert_same_state(trainer.optimizer.state_dict(), before["optimizer"], "optimizer")
  assert_same_state(replay.state_dict(), before["replay"], "replay")
  assert (trainer.optimizer_steps, trainer.samples_seen) == before["counters"]
  torch.testing.assert_close(
    trainer.generator_states()["replay"], before["rng"]["replay"]
  )
  torch.testing.assert_close(
    trainer.generator_states()["latent"], before["rng"]["latent"]
  )
  torch.testing.assert_close(torch.random.get_rng_state(), before["global"])
  if collector is not None:
    torch.testing.assert_close(collector.generator_state(), before["collector"])


def capture_state(
  trainer: VaeDistillationTrainer, replay: Any, collector: DAggerCollector | None
) -> dict[str, Any]:
  return {
    "model": {
      key: value.detach().clone() if isinstance(value, torch.Tensor) else value
      for key, value in trainer.model.state_dict().items()
    },
    "optimizer": {
      "param_groups": [
        dict(group) for group in trainer.optimizer.state_dict()["param_groups"]
      ],
      "state": {
        index: {
          name: value.detach().clone() if isinstance(value, torch.Tensor) else value
          for name, value in state.items()
        }
        for index, state in trainer.optimizer.state_dict()["state"].items()
      },
    },
    "replay": replay.state_dict(),
    "counters": (trainer.optimizer_steps, trainer.samples_seen),
    "rng": {name: state.clone() for name, state in trainer.generator_states().items()},
    "global": torch.random.get_rng_state().clone(),
    "collector": None if collector is None else collector.generator_state(),
  }


def sampled_reference(replay: Any, seed: int = 4) -> torch.Tensor:
  """One reproducible draw, used to prove a failed load changed no behavior."""
  return replay.sample(4, generator=torch.Generator().manual_seed(seed)).reference


def partition_sizes(replay: Any) -> list[int]:
  """Retained rows per motion partition, read through the public state seam."""
  return [int(part["size"]) for part in replay.state_dict()["partitions"]]


# Cohort identity.


def test_cohort_identity_records_ordered_members_and_round_trips(
  real_cohort: CohortContract,
  plan: MultiMotionPlan,
  schema: Any,
  body_selection: BodySelection,
) -> None:
  replay = make_replay(schema, plan)
  identity = make_identity(real_cohort, plan, replay, body_selection)

  assert identity.teacher_ids == TEACHER_IDS
  assert identity.teacher_codes == (0, 1)
  assert [member.frames for member in identity.members] == [
    clip.frames for clip in plan.library.clips
  ]
  assert [member.motion_source_hash for member in identity.members] == [
    real_cohort.teacher(teacher_id).hashes["motion"] for teacher_id in TEACHER_IDS
  ]
  for member in identity.members:
    assert set(member.hashes) == {
      "checkpoint",
      "motion",
      "env_config",
      "agent_config",
      "onnx",
    }
    assert member.tracked_body_names == tuple(real_cohort.body_names)
    assert member.tracked_body_indices == body_selection.indices
  assert identity.common.joint_names == tuple(real_cohort.actions.joint_names)
  assert identity.slots.counts == (1, 1)
  assert identity.slots.row_motion_ids == plan.slots.row_motion_ids
  assert identity.slots.phase_policy == "uniform"
  assert identity.replay.kind == "balanced-motion-replay"
  assert identity.replay.quotas == (4, 4)
  assert identity.replay.frame_counts == tuple(
    clip.frames for clip in plan.library.clips
  )
  assert identity.resources.device == "cpu"
  assert identity.resources.requested_seed == SEED
  assert identity.mapping_digest == plan.library.mapping_digest()

  # The record is plain data that round-trips exactly, and its digest is stable.
  restored = CohortIdentity.from_dict(identity.as_dict())
  assert restored == identity
  assert restored.as_dict() == identity.as_dict()
  assert restored.digest() == identity.digest()
  assert identity.member("tennis_001").motion_id == 1
  require_same_cohort(identity, restored)


def test_cohort_identity_refuses_incomplete_or_mismatched_evidence(
  real_cohort: CohortContract,
  plan: MultiMotionPlan,
  schema: Any,
  body_selection: BodySelection,
) -> None:
  replay = make_replay(schema, plan)
  with pytest.raises(CohortContractError, match="ordered mapping digest"):
    make_identity(real_cohort, plan, replay, body_selection, mapping_digest="deadbeef")
  with pytest.raises(CohortContractError, match="body mapping"):
    build_cohort_identity(
      real_cohort,
      plan.library,
      plan.slots,
      phase_policy="uniform",
      replay=replay,
      device="cpu",
      requested_seed=SEED,
      effective_seed=SEED,
    )
  with pytest.raises(CohortContractError, match="body selection covers"):
    build_cohort_identity(
      real_cohort,
      plan.library,
      plan.slots,
      phase_policy="uniform",
      replay=replay,
      device="cpu",
      requested_seed=SEED,
      effective_seed=SEED,
      body_selection=BodySelection(indices=(0,), source_body_count=1, names=None),
    )
  with pytest.raises(CohortContractError, match="requires the per-motion balanced"):
    make_identity(
      real_cohort,
      plan,
      LabeledReplayBuffer(8, schema),
      body_selection,
    )
  with pytest.raises(CohortContractError, match="frame limits"):
    make_identity(
      real_cohort,
      plan,
      make_replay(schema, plan, frame_counts={0: 10, 1: 340}),
      body_selection,
    )
  with pytest.raises(CohortContractError, match="slot weight for 'tennis_000'"):
    build_cohort_identity(
      real_cohort,
      plan.library,
      MotionSlotAllocation(
        weights=(2.0, 1.0),
        counts=(1, 1),
        teacher_ids=TEACHER_IDS,
        row_motion_ids=(0, 1),
      ),
      phase_policy="uniform",
      replay=replay,
      device="cpu",
      requested_seed=SEED,
      effective_seed=SEED,
      body_selection=body_selection,
    )
  with pytest.raises(CohortContractError, match="slot allocation teachers"):
    build_cohort_identity(
      real_cohort,
      plan.library,
      MotionSlotAllocation(
        weights=(1.0, 1.0),
        counts=(1, 1),
        teacher_ids=("tennis_001", "tennis_000"),
        row_motion_ids=(0, 1),
      ),
      phase_policy="uniform",
      replay=replay,
      device="cpu",
      requested_seed=SEED,
      effective_seed=SEED,
      body_selection=body_selection,
    )
  with pytest.raises(CohortContractError, match="phase policy"):
    make_identity(real_cohort, plan, replay, body_selection, phase_policy="adaptive")
  with pytest.raises(CohortContractError, match="replay weight"):
    make_identity(
      real_cohort,
      plan,
      make_replay(schema, plan, weights={0: 2.0, 1: 1.0}),
      body_selection,
    )


def test_cohort_identity_from_adapter_requires_audited_evidence(
  real_cohort: CohortContract,
  plan: MultiMotionPlan,
  schema: Any,
  body_selection: BodySelection,
) -> None:
  replay = make_replay(schema, plan)
  adapter = FakeMultiAdapter(real_cohort, plan, body_selection, schema)
  identity = cohort_identity_from_adapter(adapter, replay=replay)
  assert identity == make_identity(real_cohort, plan, replay, body_selection)

  adapter.audit.mapping_digest = "deadbeef"
  with pytest.raises(CohortContractError, match="mapping digest"):
    cohort_identity_from_adapter(adapter, replay=replay)
  adapter.audit.mapping_digest = plan.library.mapping_digest()

  adapter.audit.seed_provenance = None
  with pytest.raises(CohortContractError, match="applied before construction"):
    cohort_identity_from_adapter(adapter, replay=replay)
  adapter.audit.seed_provenance = RuntimeSeedProvenance(
    requested_seed=SEED, effective_seed=SEED, applied_before_construction=False
  )
  with pytest.raises(CohortContractError, match="applied before construction"):
    cohort_identity_from_adapter(adapter, replay=replay)
  adapter.audit.seed_provenance = RuntimeSeedProvenance(
    requested_seed=None, effective_seed=None, applied_before_construction=True
  )
  with pytest.raises(CohortContractError, match="no requested seed"):
    cohort_identity_from_adapter(adapter, replay=replay)

  with pytest.raises(CohortContractError, match="mixed-slot environment"):
    cohort_identity_from_adapter(SimpleNamespace(cohort=real_cohort), replay=replay)


def reorder(identity: CohortIdentity) -> CohortIdentity:
  """The same members in the other order, with consistent slots and routing."""
  first, second = identity.members
  return replace(
    identity,
    members=(
      replace(second, motion_id=0),
      replace(first, motion_id=1),
    ),
    slots=replace(identity.slots, teacher_ids=("tennis_001", "tennis_000")),
    replay=replace(
      identity.replay,
      motion_ids=(0, 1),
      teacher_codes=(1, 0),
      frame_counts=(second.frames, first.frames),
    ),
  )


def drop_member(identity: CohortIdentity) -> CohortIdentity:
  return replace(
    identity,
    members=identity.members[:1],
    slots=replace(
      identity.slots,
      teacher_ids=identity.slots.teacher_ids[:1],
      weights=identity.slots.weights[:1],
      counts=(2,),
      row_motion_ids=(0, 0),
    ),
    replay=replace(
      identity.replay,
      motion_ids=(0,),
      weights=(1.0,),
      quotas=(8,),
      teacher_codes=(0,),
      frame_counts=(identity.members[0].frames,),
    ),
  )


@pytest.mark.parametrize(
  ("mutate", "message"),
  [
    pytest.param(
      lambda identity: replace(
        identity,
        members=(
          replace(
            identity.members[0],
            hashes={
              "checkpoint": "deadbeef",
              **{
                role: value
                for role, value in identity.members[0].hashes.items()
                if role != "checkpoint"
              },
            },
          ),
          identity.members[1],
        ),
      ),
      "artifact 'checkpoint' digest",
      id="member-hash",
    ),
    pytest.param(
      lambda identity: replace(
        identity,
        members=(
          replace(
            identity.members[0],
            artifacts={**identity.members[0].artifacts, "onnx": "/tmp/moved.onnx"},
          ),
          identity.members[1],
        ),
      ),
      "artifact 'onnx' path",
      id="member-path",
    ),
    pytest.param(
      lambda identity: replace(identity, mapping_digest="0" * 64),
      "ordered reference mapping digest",
      id="mapping-digest",
    ),
    pytest.param(
      lambda identity: replace(identity, manifest_sha256="0" * 64),
      "manifest digest",
      id="manifest",
    ),
    pytest.param(
      lambda identity: replace(
        identity,
        common=replace(
          identity.common, control_period_s=identity.common.control_period_s + 0.01
        ),
      ),
      "common contract control_period_s",
      id="control-contract",
    ),
    pytest.param(
      lambda identity: replace(
        identity,
        common=replace(
          identity.common, joint_names=tuple(reversed(identity.common.joint_names))
        ),
      ),
      "common contract joint_names",
      id="joint-order",
    ),
    pytest.param(
      lambda identity: replace(
        identity, slots=replace(identity.slots, phase_policy="start")
      ),
      "phase policy",
      id="phase-policy",
    ),
    pytest.param(
      lambda identity: replace(
        identity,
        slots=replace(
          identity.slots,
          row_motion_ids=tuple(reversed(identity.slots.row_motion_ids)),
        ),
      ),
      "slot row motion ids",
      id="slot-rows",
    ),
    pytest.param(
      lambda identity: replace(
        identity, replay=replace(identity.replay, weights=(2.0, 1.0))
      ),
      r"replay weight\[0\]",
      id="replay-weights",
    ),
    pytest.param(
      lambda identity: replace(
        identity, replay=replace(identity.replay, capacity=16, quotas=(8, 8))
      ),
      "replay capacity",
      id="replay-capacity",
    ),
    pytest.param(
      lambda identity: replace(
        identity, resources=replace(identity.resources, device="cuda:0")
      ),
      "resource setting device",
      id="device",
    ),
    pytest.param(
      lambda identity: replace(
        identity,
        resources=replace(identity.resources, requested_seed=SEED + 1),
      ),
      "resource setting requested_seed",
      id="seed",
    ),
  ],
)
def test_require_same_cohort_reports_each_changed_field(
  real_cohort: CohortContract,
  plan: MultiMotionPlan,
  schema: Any,
  body_selection: BodySelection,
  mutate: Any,
  message: str,
) -> None:
  identity = make_identity(real_cohort, plan, make_replay(schema, plan), body_selection)
  with pytest.raises(CohortContractError, match=message):
    require_same_cohort(identity, mutate(identity))


def test_require_same_cohort_reports_reordered_and_missing_members(
  real_cohort: CohortContract,
  plan: MultiMotionPlan,
  schema: Any,
  body_selection: BodySelection,
) -> None:
  identity = make_identity(real_cohort, plan, make_replay(schema, plan), body_selection)
  with pytest.raises(CohortContractError, match="reordered"):
    require_same_cohort(identity, reorder(identity))
  with pytest.raises(
    CohortContractError, match="not in the live cohort \\['tennis_001'\\]"
  ):
    require_same_cohort(identity, drop_member(identity))


# Version-2 checkpoints.


def test_cohort_checkpoint_round_trips_and_keeps_a_deterministic_next_update(
  real_cohort: CohortContract,
  plan: MultiMotionPlan,
  schema: Any,
  body_selection: BodySelection,
  tmp_path: Path,
) -> None:
  trainer, identity = make_cohort_trainer(
    schema, plan, body_selection, real_cohort, updates=2
  )
  path = tmp_path / "cohort.pt"
  save_cohort_checkpoint(
    path,
    trainer,
    trainer.replay,
    cohort=identity,
    counters={"iteration": 2, "segment_namespace": 1},
    schedule={"max_iterations": 6},
    resolved_config={"command": "train", "teacher_ids": list(TEACHER_IDS)},
  )
  payload = torch.load(path, weights_only=True)
  assert payload["version"] == 2
  assert payload["kind"] == "mjlab-m4-cohort-distillation"
  assert payload["cohort_digest"] == identity.digest()
  assert payload["cohort"]["members"][0]["teacher_id"] == "tennis_000"
  assert payload["replay_policy"]["kind"] == "balanced-motion-replay"
  assert "teacher_hashes" not in payload

  restored, _ = make_cohort_trainer(
    schema, plan, body_selection, real_cohort, updates=0
  )
  state = load_cohort_checkpoint(
    path, restored, restored.replay, expected_cohort=identity
  )
  assert state.counters == {"iteration": 2, "segment_namespace": 1}
  assert state.schedule == {"max_iterations": 6}
  assert state.cohort == identity
  assert state.replay_policy == identity.replay
  assert state.resume_restarts_simulator

  # Sampling is deterministic from the restored buffer plus generator state, so
  # the next optimizer update reproduces the uninterrupted run's update.
  next_a = trainer.train_update()
  next_b = restored.train_update()
  assert next_a.total_loss == pytest.approx(next_b.total_loss, abs=1e-7)
  assert next_a.kl == pytest.approx(next_b.kl, abs=1e-7)
  for left, right in zip(
    trainer.model.parameters(), restored.model.parameters(), strict=True
  ):
    torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-7)


def test_enabled_standing_cohort_checkpoint_is_v3_and_round_trips_reset_state(
  real_cohort: CohortContract,
  plan: MultiMotionPlan,
  schema: Any,
  body_selection: BodySelection,
  tmp_path: Path,
) -> None:
  trainer, identity, adapter, collector = make_v3_trainer(
    schema, plan, body_selection, real_cohort
  )
  provenance = make_reset_provenance(plan)
  path = tmp_path / "standing-v3.pt"
  before_reset = adapter.reset_command.reset_rng_state()
  save_cohort_checkpoint(
    path,
    trainer,
    trainer.replay,
    cohort=identity,
    reset_provenance=provenance,
    collector=collector,
  )
  payload = torch.load(path, weights_only=True)
  assert payload["version"] == 3
  assert payload["reset_provenance"] == provenance.as_dict()
  selected = load_cohort_member_inference(
    path,
    real_cohort,
    "tennis_000",
    reset_profile="standing-window",
  )
  assert selected.reset_provenance == provenance
  assert selected.evaluation_reset_profile == "standing-window"

  restored, restored_identity, restored_adapter, restored_collector = make_v3_trainer(
    schema, plan, body_selection, real_cohort
  )
  restored_adapter.reset_command.set_reset_rng_state(torch.Generator().get_state())
  state = load_cohort_checkpoint(
    path,
    restored,
    restored.replay,
    expected_cohort=restored_identity,
    expected_reset_provenance=provenance,
    collector=restored_collector,
  )
  assert state.reset_provenance == provenance
  assert torch.equal(restored_adapter.reset_command.reset_rng_state(), before_reset)


def test_v3_standing_policy_mismatch_and_v2_migration_refuse_without_mutation(
  real_cohort: CohortContract,
  plan: MultiMotionPlan,
  schema: Any,
  body_selection: BodySelection,
  tmp_path: Path,
) -> None:
  trainer, identity, _adapter, collector = make_v3_trainer(
    schema, plan, body_selection, real_cohort
  )
  provenance = make_reset_provenance(plan)
  path = tmp_path / "standing-v3.pt"
  save_cohort_checkpoint(
    path,
    trainer,
    trainer.replay,
    cohort=identity,
    reset_provenance=provenance,
    collector=collector,
  )
  before = capture_state(trainer, trainer.replay, collector)
  changed = replace(
    provenance,
    reset_policy=make_reset_policy(
      kind="standing-mixture", standing_start_fraction=0.5
    ),
  )
  with pytest.raises(CheckpointValidationError, match="policy/version/provenance"):
    load_cohort_checkpoint(
      path,
      trainer,
      trainer.replay,
      expected_cohort=identity,
      expected_reset_provenance=changed,
      collector=collector,
    )
  assert_state_unchanged(trainer, trainer.replay, collector, before)

  with pytest.raises(CheckpointValidationError, match="expected reset provenance"):
    load_cohort_checkpoint(path, trainer, trainer.replay, expected_cohort=identity)


def test_version_one_and_version_two_loaders_refuse_each_other(
  real_cohort: CohortContract,
  plan: MultiMotionPlan,
  schema: Any,
  body_selection: BodySelection,
  tmp_path: Path,
) -> None:
  trainer, identity = make_cohort_trainer(
    schema, plan, body_selection, real_cohort, updates=1
  )
  cohort_path = tmp_path / "cohort.pt"
  save_cohort_checkpoint(cohort_path, trainer, trainer.replay, cohort=identity)

  # A version-1 file needs the FIFO layout, so build the single-teacher pair.
  torch.manual_seed(3)
  fifo_model = ConditionalVAE(schema, ModelSettings(hidden_dims=(8, 8)))
  fifo_replay = LabeledReplayBuffer(8, schema)
  fifo_replay.insert(make_batch(schema, plan))
  fifo_trainer = VaeDistillationTrainer(
    fifo_model, fifo_replay, TrainingConfig(accumulation_steps=1, minibatch_size=2)
  )
  single_path = tmp_path / "single.pt"
  save_checkpoint(single_path, fifo_trainer, fifo_replay)

  with pytest.raises(CheckpointValidationError, match="version-1.*load_checkpoint"):
    load_cohort_checkpoint(
      single_path, trainer, trainer.replay, expected_cohort=identity
    )
  with pytest.raises(CheckpointValidationError, match="version-2.*load_cohort"):
    load_checkpoint(cohort_path, fifo_trainer, fifo_replay)
  with pytest.raises(
    CheckpointValidationError, match="version-2.*load_cohort_member_inference"
  ):
    load_inference_checkpoint(cohort_path)
  with pytest.raises(
    CheckpointValidationError, match="version-1.*load_inference_checkpoint"
  ):
    load_cohort_member_inference(single_path, real_cohort, "tennis_000")

  # A version-1 checkpoint never stores the per-motion balanced replay layout.
  with pytest.raises(CheckpointValidationError, match="per-motion balanced"):
    save_checkpoint(
      tmp_path / "bad.pt", trainer, trainer.replay, teacher_hashes={"t": "one"}
    )


@pytest.mark.parametrize(
  ("mutate", "message"),
  [
    pytest.param(reorder, "reordered", id="reordered-members"),
    pytest.param(
      drop_member, r"not in the live cohort \['tennis_001'\]", id="missing-member"
    ),
    pytest.param(
      lambda identity: replace(
        identity,
        common=replace(
          identity.common, joint_names=tuple(reversed(identity.common.joint_names))
        ),
      ),
      "common contract joint_names",
      id="changed-schema-order",
    ),
    pytest.param(
      lambda identity: replace(
        identity,
        common=replace(
          identity.common, control_period_s=identity.common.control_period_s + 0.01
        ),
      ),
      "common contract control_period_s",
      id="changed-control",
    ),
    pytest.param(
      lambda identity: replace(
        identity, slots=replace(identity.slots, weights=(2.0, 1.0))
      ),
      "slot weight",
      id="changed-slot-weights",
    ),
    pytest.param(
      lambda identity: replace(
        identity, replay=replace(identity.replay, quotas=(6, 2))
      ),
      "replay quotas",
      id="changed-quotas",
    ),
    pytest.param(
      lambda identity: replace(
        identity, resources=replace(identity.resources, device="cuda:0")
      ),
      "resource setting device",
      id="changed-device",
    ),
  ],
)
def test_strict_cohort_resume_rejects_an_incompatible_live_cohort(
  real_cohort: CohortContract,
  plan: MultiMotionPlan,
  schema: Any,
  body_selection: BodySelection,
  mutate: Any,
  message: str,
  tmp_path: Path,
) -> None:
  trainer, identity = make_cohort_trainer(
    schema, plan, body_selection, real_cohort, updates=1
  )
  path = tmp_path / "cohort.pt"
  save_cohort_checkpoint(path, trainer, trainer.replay, cohort=identity)
  before = capture_state(trainer, trainer.replay, None)
  live = mutate(identity)
  with pytest.raises(CheckpointValidationError, match=message):
    load_cohort_checkpoint(path, trainer, trainer.replay, expected_cohort=live)
  assert_state_unchanged(trainer, trainer.replay, None, before)


def test_strict_cohort_resume_rejects_a_tampered_cohort_record(
  real_cohort: CohortContract,
  plan: MultiMotionPlan,
  schema: Any,
  body_selection: BodySelection,
  tmp_path: Path,
) -> None:
  trainer, identity = make_cohort_trainer(
    schema, plan, body_selection, real_cohort, updates=1
  )
  path = tmp_path / "cohort.pt"
  save_cohort_checkpoint(path, trainer, trainer.replay, cohort=identity)
  before = capture_state(trainer, trainer.replay, None)

  # A stored member digest that no longer matches the live manifest.
  payload = torch.load(path, weights_only=True)
  tampered = {**payload["cohort"]["members"][0]["hashes"], "motion": "deadbeef"}
  payload["cohort"]["members"][0]["hashes"] = tampered
  payload["cohort_digest"] = CohortIdentity.from_dict(payload["cohort"]).digest()
  torch.save(payload, path)
  with pytest.raises(CheckpointValidationError, match="artifact 'motion' digest"):
    load_cohort_checkpoint(path, trainer, trainer.replay, expected_cohort=identity)

  # A cohort record whose own digest does not match it.
  payload = torch.load(path, weights_only=True)
  payload["cohort"]["slots"]["phase_policy"] = "start"
  torch.save(payload, path)
  with pytest.raises(CheckpointValidationError, match="cohort digest does not match"):
    load_cohort_checkpoint(path, trainer, trainer.replay, expected_cohort=identity)

  # A replay policy that disagrees with the live buffer's partition layout.
  payload = torch.load(path, weights_only=True)
  payload["cohort"]["slots"]["phase_policy"] = "uniform"
  payload["cohort_digest"] = CohortIdentity.from_dict(payload["cohort"]).digest()
  payload["replay_policy"]["capacity"] = 16
  torch.save(payload, path)
  with pytest.raises(
    CheckpointValidationError, match="replay policy|replay partitions"
  ):
    load_cohort_checkpoint(path, trainer, trainer.replay, expected_cohort=identity)
  assert_state_unchanged(trainer, trainer.replay, None, before)
  # The model-only member selection verifies the same record integrity, so a
  # tampered replay policy cannot be used to pin a member either.
  with pytest.raises(
    CheckpointValidationError, match="replay policy|replay partitions"
  ):
    load_cohort_member_inference(path, real_cohort, "tennis_000")


def test_cohort_save_requires_the_balanced_replay_and_a_matching_identity(
  real_cohort: CohortContract,
  plan: MultiMotionPlan,
  schema: Any,
  body_selection: BodySelection,
  tmp_path: Path,
) -> None:
  trainer, identity = make_cohort_trainer(
    schema, plan, body_selection, real_cohort, updates=1
  )
  fifo_replay = LabeledReplayBuffer(8, schema)
  fifo_replay.insert(make_batch(schema, plan))
  fifo_model = ConditionalVAE(schema, ModelSettings(hidden_dims=(8, 8)))
  fifo_trainer = VaeDistillationTrainer(fifo_model, fifo_replay)
  with pytest.raises(CheckpointValidationError, match="per-motion balanced"):
    save_cohort_checkpoint(
      tmp_path / "fifo.pt", fifo_trainer, fifo_replay, cohort=identity
    )
  with pytest.raises(CheckpointValidationError, match="live replay buffer"):
    save_cohort_checkpoint(
      tmp_path / "stale.pt",
      trainer,
      trainer.replay,
      cohort=replace(
        identity,
        replay=replace(identity.replay, capacity=16, quotas=(8, 8)),
      ),
    )
  assert not (tmp_path / "fifo.pt").exists()
  assert not (tmp_path / "stale.pt").exists()


def test_cohort_resume_rejects_a_fifo_replay_buffer(
  real_cohort: CohortContract,
  plan: MultiMotionPlan,
  schema: Any,
  body_selection: BodySelection,
  tmp_path: Path,
) -> None:
  trainer, identity = make_cohort_trainer(
    schema, plan, body_selection, real_cohort, updates=1
  )
  path = tmp_path / "cohort.pt"
  save_cohort_checkpoint(path, trainer, trainer.replay, cohort=identity)
  fifo_replay = LabeledReplayBuffer(8, schema)
  fifo_replay.insert(make_batch(schema, plan))
  torch.manual_seed(3)
  fifo_trainer = VaeDistillationTrainer(
    ConditionalVAE(schema, ModelSettings(hidden_dims=(8, 8))),
    fifo_replay,
    TrainingConfig(accumulation_steps=1, minibatch_size=2),
  )
  with pytest.raises(CheckpointValidationError, match="per-motion balanced"):
    load_cohort_checkpoint(path, fifo_trainer, fifo_replay, expected_cohort=identity)


def test_cohort_resume_rejects_a_corrupted_last_replay_partition(
  real_cohort: CohortContract,
  plan: MultiMotionPlan,
  schema: Any,
  body_selection: BodySelection,
  tmp_path: Path,
) -> None:
  trainer, identity = make_cohort_trainer(
    schema, plan, body_selection, real_cohort, updates=1
  )
  path = tmp_path / "cohort.pt"
  save_cohort_checkpoint(path, trainer, trainer.replay, cohort=identity)
  before = capture_state(trainer, trainer.replay, None)

  payload = torch.load(path, weights_only=True)
  partition = payload["replay"]["partitions"][1]
  assert partition["size"] == 1 and partition["next"] == 1
  # The reproducible draw happens before the state snapshot, because a draw is
  # itself a state change (it advances the partition's drawn counter).
  before_sample = sampled_reference(trainer.replay)
  before = capture_state(trainer, trainer.replay, None)
  partition["storage"]["reference"][0, 0] = float("nan")
  torch.save(payload, path)
  with pytest.raises(CheckpointValidationError, match="'reference' is invalid"):
    load_cohort_checkpoint(path, trainer, trainer.replay, expected_cohort=identity)
  assert_state_unchanged(trainer, trainer.replay, None, before)
  torch.testing.assert_close(sampled_reference(trainer.replay), before_sample)


def test_cohort_resume_rejects_a_corrupted_generator_state_without_mutation(
  real_cohort: CohortContract,
  plan: MultiMotionPlan,
  schema: Any,
  body_selection: BodySelection,
  tmp_path: Path,
) -> None:
  trainer, identity = make_cohort_trainer(
    schema, plan, body_selection, real_cohort, updates=1
  )
  path = tmp_path / "cohort.pt"
  save_cohort_checkpoint(path, trainer, trainer.replay, cohort=identity)
  before = capture_state(trainer, trainer.replay, None)

  payload = torch.load(path, weights_only=True)
  payload["rng"]["trainer"]["latent"] = torch.zeros(1, dtype=torch.uint8)
  torch.save(payload, path)
  with pytest.raises(CheckpointValidationError, match="latent RNG state"):
    load_cohort_checkpoint(path, trainer, trainer.replay, expected_cohort=identity)
  assert_state_unchanged(trainer, trainer.replay, None, before)


def test_cohort_restore_rolls_back_all_state_after_a_late_failure(
  real_cohort: CohortContract,
  plan: MultiMotionPlan,
  schema: Any,
  body_selection: BodySelection,
  tmp_path: Path,
) -> None:
  """A failure after the model was installed must still roll everything back."""
  trainer, identity = make_cohort_trainer(
    schema, plan, body_selection, real_cohort, updates=1
  )
  path = tmp_path / "cohort.pt"
  save_cohort_checkpoint(path, trainer, trainer.replay, cohort=identity)

  payload = torch.load(path, weights_only=True)
  # Adam rejects a parameter group whose size disagrees with the live optimizer
  # while every earlier restore step (model, then optimizer groups) already ran.
  payload["optimizer"]["param_groups"][0]["params"] = [10**6]
  before_sample = sampled_reference(trainer.replay)
  before = capture_state(trainer, trainer.replay, None)
  torch.save(payload, path)
  with pytest.raises(
    CheckpointValidationError, match="restore failed before completion"
  ) as failure:
    load_cohort_checkpoint(path, trainer, trainer.replay, expected_cohort=identity)
  # The refusal came from the optimizer install, i.e. after the model weights
  # were already replaced, so this exercises the late-failure rollback path.
  assert "parameter group" in str(failure.value)
  assert_state_unchanged(trainer, trainer.replay, None, before)
  # Sampling and the optimizer are still exactly reproducible from the rolled
  # back state, so a refused checkpoint cannot quietly perturb training.
  torch.testing.assert_close(sampled_reference(trainer.replay), before_sample)
  assert not trainer.poisoned
  update = trainer.train_update()
  assert math.isfinite(update.total_loss) and math.isfinite(update.gradient_norm)


# Checked member selection.


def test_cohort_member_inference_selects_every_member_with_identical_parameters(
  real_cohort: CohortContract,
  plan: MultiMotionPlan,
  schema: Any,
  body_selection: BodySelection,
  tmp_path: Path,
) -> None:
  trainer, identity = make_cohort_trainer(
    schema, plan, body_selection, real_cohort, updates=2
  )
  path = tmp_path / "cohort.pt"
  save_cohort_checkpoint(
    path,
    trainer,
    trainer.replay,
    cohort=identity,
    counters={"iteration": 4},
    resolved_config={"command": "train"},
  )
  first = load_cohort_member_inference(path, real_cohort, "tennis_000")
  second = load_cohort_member_inference(path, real_cohort, "tennis_001")

  assert first.teacher_ids == TEACHER_IDS
  assert first.cohort == identity
  assert first.motion_id == 0 and first.teacher_code == 0
  assert second.motion_id == 1 and second.teacher_code == 1
  assert first.relocated_artifact_roles == ()
  assert first.artifact_hashes == identity.member("tennis_000").hashes
  assert first.counters == {"iteration": 4}
  assert not first.model.training
  # One shared student: pinning either member restores the same parameters.
  for left, right in zip(
    first.model.parameters(), second.model.parameters(), strict=True
  ):
    torch.testing.assert_close(left, right)
  # And they are the trained weights, not a fresh initialization.
  for left, right in zip(
    first.model.parameters(), trainer.model.parameters(), strict=True
  ):
    torch.testing.assert_close(left, right)

  batch = make_batch(schema, plan, seed=12)
  torch.testing.assert_close(
    first.model.mean_inference(batch.reference, batch.conditioning),
    trainer.model.mean_inference(batch.reference, batch.conditioning),
    atol=1e-6,
    rtol=1e-6,
  )


def test_cohort_member_inference_refuses_unknown_members_and_changed_artifacts(
  real_cohort: CohortContract,
  plan: MultiMotionPlan,
  schema: Any,
  body_selection: BodySelection,
  tmp_path: Path,
) -> None:
  trainer, identity = make_cohort_trainer(
    schema, plan, body_selection, real_cohort, updates=1
  )
  path = tmp_path / "cohort.pt"
  save_cohort_checkpoint(path, trainer, trainer.replay, cohort=identity)

  with pytest.raises(CheckpointValidationError, match="has no member 'tennis_999'"):
    load_cohort_member_inference(path, real_cohort, "tennis_999")

  # A changed artifact fails on its digest even when the record's own digest is
  # recomputed, so a tampered record cannot smuggle unverified content.
  payload = torch.load(path, weights_only=True)
  payload["cohort"]["members"][1]["hashes"]["onnx"] = "deadbeef"
  payload["cohort_digest"] = CohortIdentity.from_dict(payload["cohort"]).digest()
  torch.save(payload, path)
  with pytest.raises(CheckpointValidationError, match="artifact 'onnx' digest"):
    load_cohort_member_inference(path, real_cohort, "tennis_001")

  # The requested member must exist and the requested schema must match the
  # saved one before any model is rebuilt.
  payload = torch.load(path, weights_only=True)
  payload["cohort"]["members"][1]["hashes"]["onnx"] = identity.member(
    "tennis_001"
  ).hashes["onnx"]
  payload["cohort_digest"] = CohortIdentity.from_dict(payload["cohort"]).digest()
  torch.save(payload, path)
  with pytest.raises(CheckpointValidationError, match="expected schema"):
    load_cohort_member_inference(
      path, real_cohort, "tennis_000", expected_schema=make_schema("anchor")
    )


def test_cohort_member_inference_accepts_relocation_but_resume_stays_strict(
  real_cohort: CohortContract,
  plan: MultiMotionPlan,
  schema: Any,
  body_selection: BodySelection,
  tmp_path: Path,
) -> None:
  """A byte-identical relocated artifact is inference-only evidence."""
  trainer, identity = make_cohort_trainer(
    schema, plan, body_selection, real_cohort, updates=1
  )
  moved_motion = "/elsewhere/tennis_000_tracking.npz"
  relocated_member = replace(
    identity.members[0],
    artifacts={**identity.members[0].artifacts, "motion": moved_motion},
  )
  relocated = replace(identity, members=(relocated_member, identity.members[1]))
  path = tmp_path / "relocated.pt"
  save_cohort_checkpoint(path, trainer, trainer.replay, cohort=relocated)

  selected = load_cohort_member_inference(path, real_cohort, "tennis_000")
  assert selected.relocated_artifact_roles == ("motion",)
  # The stored provenance is retained verbatim, never rewritten.
  assert selected.cohort.member("tennis_000").artifacts["motion"] == moved_motion
  assert selected.artifact_hashes == identity.member("tennis_000").hashes
  for left, right in zip(
    selected.model.parameters(), trainer.model.parameters(), strict=True
  ):
    torch.testing.assert_close(left, right)

  # Strict resume compares the whole record, so the relocation is refused there.
  with pytest.raises(CheckpointValidationError, match="artifact 'motion' path"):
    load_cohort_checkpoint(path, trainer, trainer.replay, expected_cohort=identity)
  state = load_cohort_checkpoint(
    path, trainer, trainer.replay, expected_cohort=relocated
  )
  assert state.cohort == relocated


# Multi-teacher runner lifecycle.


def test_multi_teacher_runner_saves_and_strictly_resumes(
  real_cohort: CohortContract,
  plan: MultiMotionPlan,
  schema: Any,
  body_selection: BodySelection,
  tmp_path: Path,
) -> None:
  runner, adapter = make_runner(real_cohort, plan, body_selection, schema)
  results = runner.run(iterations=2)
  assert len(results) == 2
  # The first cycle bootstraps twice (two ticks of two rows), the second
  # collects one tick.
  bootstrap_collection = results[0].collection
  collect_collection = results[1].collection
  assert bootstrap_collection is not None
  assert collect_collection is not None
  assert bootstrap_collection.samples == 4
  assert collect_collection.samples == 2
  assert [len(item.updates) for item in results] == [1, 1]
  identity = cohort_identity_from_adapter(adapter, replay=runner.replay)
  assert identity.replay.motion_ids == (0, 1)
  # Every motion partition retains its own rows: one per motion per tick.
  assert partition_sizes(runner.replay) == [3, 3]

  path = tmp_path / "cohort.pt"
  runner.save_cohort(
    str(path),
    identity,
    resolved_config={"command": "train", "teacher_ids": list(TEACHER_IDS)},
    schedule={"max_iterations": 4},
  )
  assert runner.events[-1]["event"] == "cohort_checkpoint_saved"

  resumed, resumed_adapter = make_runner(real_cohort, plan, body_selection, schema)
  live = cohort_identity_from_adapter(resumed_adapter, replay=resumed.replay)
  state = resumed.resume_cohort(str(path), live)
  assert state.cohort.teacher_ids == TEACHER_IDS
  assert resumed.iteration == 2
  assert resumed.events[-1]["event"] == "cohort_resumed"
  assert resumed.events[-1]["retained_records"] == len(runner.replay)
  assert resumed.events[-1]["replay_partitions"] == 2

  result = resumed.run(iterations=1)[0]
  assert result.resumed_reset
  # Exactly one adapter reset for the resumed cycle: the simulator restarts and
  # is never resumed bitwise.
  assert resumed_adapter.reset_calls == 1
  assert resumed_adapter.step_calls == 1
  assert result.collection is not None and result.collection.samples == 2
  assert len(result.updates) == 1
  # The resumed cycle appended to both retained partitions, and the restarted
  # simulator's fresh records sit above every retained segment ID.
  assert partition_sizes(resumed.replay) == [4, 4]
  assert resumed.replay.max_valid_segment_id() == 3


def test_multi_teacher_resume_namespaces_above_every_retained_partition(
  real_cohort: CohortContract,
  plan: MultiMotionPlan,
  schema: Any,
  body_selection: BodySelection,
  tmp_path: Path,
) -> None:
  runner, adapter = make_runner(real_cohort, plan, body_selection, schema)
  runner.run(iterations=2)
  identity = cohort_identity_from_adapter(adapter, replay=runner.replay)
  path = tmp_path / "cohort.pt"
  runner.save_cohort(str(path), identity)

  # A high segment ID retained in *one later partition* is what the namespace
  # must clear: reading a single partition, or only the first one, would reuse
  # IDs the restarted simulator regenerates.
  payload = torch.load(path, weights_only=True)
  partition = payload["replay"]["partitions"][1]
  assert partition["size"] >= 1
  partition["storage"]["episode_id"][0] = 10**6
  torch.save(payload, path)

  resumed, resumed_adapter = make_runner(real_cohort, plan, body_selection, schema)
  live = cohort_identity_from_adapter(resumed_adapter, replay=resumed.replay)
  resumed.resume_cohort(str(path), live)
  assert resumed._segment_namespace == 10**6 + 1

  resumed.run(iterations=1)
  highest = resumed.replay.max_valid_segment_id()
  assert highest is not None
  assert highest >= 10**6 + 1
  assert highest < 10**6 + 10
  assert resumed_adapter.reset_calls == 1


def test_multi_teacher_runner_evaluates_both_motions_per_iteration(
  real_cohort: CohortContract,
  plan: MultiMotionPlan,
  schema: Any,
  body_selection: BodySelection,
) -> None:
  adapter = FakeMultiAdapter(real_cohort, plan, body_selection, schema)
  torch.manual_seed(3)
  model = ConditionalVAE(schema, ModelSettings(hidden_dims=(8, 8)))
  replay = make_replay(schema, plan)
  trainer = VaeDistillationTrainer(
    model, replay, TrainingConfig(accumulation_steps=1, minibatch_size=2), seed=5
  )
  collector = DAggerCollector(adapter, make_bank(), model, replay)
  runner = DistillationRunner(
    collector,
    trainer,
    RunnerConfig(
      max_iterations=2,
      bootstrap_steps=2,
      collection_steps=1,
      updates_per_iteration=1,
      evaluate_every=1,
      evaluation_steps=1,
      evaluation_mode="teacher",
      seed=5,
    ),
  )
  results = runner.run()
  for result in results:
    assert result.evaluation is not None
    assert [item.motion_id for item in result.evaluation.per_motion] == [0, 1]
    assert [item.teacher_code for item in result.evaluation.per_motion] == [0, 1]
  # Evaluation advanced the shared adapter, so the next collection resets.
  assert adapter.reset_calls >= 2


class FakeShardedSource:
  """Parent-side view of a sharded run, with no workers and no simulator.

  A sharded parent owns no environment: it reads the standing contract from a
  worker description and every shard derives its own reset RNG for the
  collection call it is about to run.  This stands in for that source so the
  checkpoint contract can be exercised without devices.
  """

  def __init__(
    self, provenance: ResetProvenance, *, policy_enabled: bool = True
  ) -> None:
    self.reset_provenance = provenance if policy_enabled else None
    self.invalidated = 0
    self.closed = False
    self.collections: list[tuple[int, int, bool]] = []

  def collect(self, config: Any, *, reset: bool) -> CollectionResult:
    """Record the call without workers, so a lifecycle can be driven on CPU."""
    self.collections.append((config.collector_iteration, config.seed, reset))
    return CollectionResult(
      ticks=1,
      samples=2,
      teacher_steps=1,
      student_steps=1,
      boundaries=(),
      disagreement_mean=0.0,
    )

  def invalidate_snapshot(self, *, requires_reset: bool = True) -> None:
    self.invalidated += 1

  def close(self) -> None:
    self.closed = True


def make_sharded_runner(
  real_cohort: CohortContract,
  plan: MultiMotionPlan,
  body_selection: BodySelection,
  schema: Any,
  provenance: ResetProvenance,
) -> tuple[DistillationRunner, Any, FakeShardedSource]:
  trainer, identity, _adapter, _collector = make_v3_trainer(
    schema, plan, body_selection, real_cohort
  )
  source = FakeShardedSource(provenance)
  runner = DistillationRunner(
    None,
    trainer,
    RunnerConfig(max_iterations=2, seed=5),
    sharded=source,
  )
  return runner, identity, source


def test_sharded_standing_cohort_checkpoint_derives_its_reset_state(
  real_cohort: CohortContract,
  plan: MultiMotionPlan,
  schema: Any,
  body_selection: BodySelection,
  tmp_path: Path,
) -> None:
  """A sharded standing run saves and resumes with no owned reset stream.

  Each shard derives its standing resets from the collection it is about to
  run, so the checkpoint records the standing contract and no reset RNG state,
  and a resume installs none.  Before this contract existed the save failed
  outright -- "enabled v3 cohort save requires a multi-motion command reset RNG
  state" -- which made a standing multi-GPU run impossible to checkpoint at all,
  not merely untested.
  """
  provenance = make_reset_provenance(plan)
  runner, identity, _source = make_sharded_runner(
    real_cohort, plan, body_selection, schema, provenance
  )
  # A resumed run must continue the stored iteration counter, because every
  # derived seed depends on it: saving at zero would leave the counters plumbing
  # untested.
  runner.iteration = 3
  path = tmp_path / "sharded-standing-v3.pt"
  runner.save_cohort(str(path), identity)

  payload = torch.load(path, weights_only=True)
  assert payload["counters"]["iteration"] == 3
  assert payload["version"] == 3
  assert payload["reset_provenance"] == provenance.as_dict()
  # The standing contract is recorded; the RNG that draws it is derived per
  # call, so there is no stream to store and nothing for a resume to install.
  assert set(payload["rng"]) == {"global_cpu", "trainer"}

  restored, restored_identity, _restored_source = make_sharded_runner(
    real_cohort, plan, body_selection, schema, provenance
  )
  state = restored.resume_cohort(str(path), restored_identity)
  assert state.reset_provenance == provenance
  assert restored.iteration == 3


def test_sharded_declaration_is_checked_against_the_stored_rng_set(
  real_cohort: CohortContract,
  plan: MultiMotionPlan,
  schema: Any,
  body_selection: BodySelection,
  tmp_path: Path,
) -> None:
  """The derived declaration is verified against the payload, not trusted.

  ``_validate_state_payload`` compares the RNG key set exactly, so claiming a
  stored reset stream for a payload that has none is refused rather than
  resumed into a run whose resets would silently restart from an arbitrary
  state.
  """
  provenance = make_reset_provenance(plan)
  runner, identity, _source = make_sharded_runner(
    real_cohort, plan, body_selection, schema, provenance
  )
  path = tmp_path / "sharded-standing-v3.pt"
  runner.save_cohort(str(path), identity)

  restored, restored_identity, _restored_source = make_sharded_runner(
    real_cohort, plan, body_selection, schema, provenance
  )
  with pytest.raises(CheckpointValidationError, match="RNG streams"):
    load_cohort_checkpoint(
      path,
      restored.trainer,
      restored.replay,
      expected_cohort=restored_identity,
      expected_reset_provenance=provenance,
      collector=None,
    )


def test_cohort_save_refuses_an_undeclared_derived_reset_rng(
  real_cohort: CohortContract,
  plan: MultiMotionPlan,
  schema: Any,
  body_selection: BodySelection,
  tmp_path: Path,
) -> None:
  """A standing save without a stream must say where its resets come from.

  The historical refusal is preserved for a caller that has neither a live
  collector nor a declaration, and a declaration without a standing contract is
  refused as well: a derived RNG only has meaning when something is drawing it.
  """
  provenance = make_reset_provenance(plan)
  trainer, identity, _adapter, _collector = make_v3_trainer(
    schema, plan, body_selection, real_cohort
  )
  with pytest.raises(CheckpointValidationError, match="reset RNG state"):
    save_cohort_checkpoint(
      tmp_path / "undeclared.pt",
      trainer,
      trainer.replay,
      cohort=identity,
      reset_provenance=provenance,
      collector=None,
    )
  with pytest.raises(CheckpointValidationError, match="standing run"):
    save_cohort_checkpoint(
      tmp_path / "unprovenanced.pt",
      trainer,
      trainer.replay,
      cohort=identity,
      collector=None,
      reset_rng_derived=True,
    )


def test_a_resumed_sharded_run_hands_the_stored_iteration_to_its_source(
  real_cohort: CohortContract,
  plan: MultiMotionPlan,
  schema: Any,
  body_selection: BodySelection,
  tmp_path: Path,
) -> None:
  """The restored counter must reach the collection call, not just the record.

  Every derived seed -- standing resets, the rollout generator and the
  process-global RNG -- is a function of (base seed, iteration, worker), so a
  resume that restored the counter but collected at zero would reproduce the
  wrong randomness while every checkpoint field still looked right.  This is the
  lifecycle the CPU suites otherwise leave untested: a real saved counter, a
  resume, and the call the runner then makes.
  """
  provenance = make_reset_provenance(plan)
  runner, identity, _source = make_sharded_runner(
    real_cohort, plan, body_selection, schema, provenance
  )
  runner.iteration = 1
  path = tmp_path / "sharded-resumed-lifecycle.pt"
  runner.save_cohort(str(path), identity)

  restored, restored_identity, source = make_sharded_runner(
    real_cohort, plan, body_selection, schema, provenance
  )
  restored.resume_cohort(str(path), restored_identity)
  assert restored.iteration == 1

  restored.run_iteration()

  assert source.collections, "the resumed run never collected"
  iteration, seed, reset = source.collections[0]
  assert iteration == 1
  # The runner derives its per-iteration seed from the restored counter, and a
  # resumed run always resets the simulator before its first collection.
  assert seed == restored.config.seed + 1
  assert reset is True


def test_cohort_save_refuses_a_derived_declaration_with_a_live_collector(
  real_cohort: CohortContract,
  plan: MultiMotionPlan,
  schema: Any,
  body_selection: BodySelection,
  tmp_path: Path,
) -> None:
  """A declaration is checked against the live components it contradicts.

  The exact RNG key-set check already refuses a payload whose streams disagree
  with the declaration.  A live collector owns a persistent reset stream, so
  claiming derivation *while holding one* would store a checkpoint whose resets
  nothing reproduces -- refused at the save, before any file exists.
  """
  provenance = make_reset_provenance(plan)
  trainer, identity, _adapter, collector = make_v3_trainer(
    schema, plan, body_selection, real_cohort
  )
  path = tmp_path / "contradictory.pt"
  with pytest.raises(CheckpointValidationError, match="live collector owns"):
    save_cohort_checkpoint(
      path,
      trainer,
      trainer.replay,
      cohort=identity,
      reset_provenance=provenance,
      collector=collector,
      reset_rng_derived=True,
    )
  assert not path.exists()
