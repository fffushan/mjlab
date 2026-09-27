"""Mixed-cohort routing, attribution, and frozen-bank tests.

The fake layer drives the real :class:`MultiMotionCommand` routing contract with
an owned row state, so a stored label can be checked against the teacher that
owns the row's clip; the real-environment layer checks that both clips of the
saved cohort are audited and that the adapter reads one cached observation.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from tracking_distillation_fixtures import make_actor, write_reference_clip

from mjlab.tasks.tracking.distillation.adapter import (
  DistillationSnapshot,
  DistillationStep,
  MultiMotionDistillationAdapter,
  make_multi_teacher_distillation_adapter,
  validate_live_contract,
  validate_multi_motion_live_contract,
)
from mjlab.tasks.tracking.distillation.collector import (
  CollectionConfig,
  CollectionNumericalError,
  DAggerCollector,
  SegmentBoundary,
  TeacherRoutingError,
  evaluate_distillation,
)
from mjlab.tasks.tracking.distillation.config import (
  ActorArchitecture,
  load_manifest,
  resolve_cohort,
)
from mjlab.tasks.tracking.distillation.environment import (
  ReferenceBoundaryEvents,
  build_multi_motion_environment,
)
from mjlab.tasks.tracking.distillation.model import ConditionalVAE
from mjlab.tasks.tracking.distillation.motion_library import (
  MotionClipSpec,
  MotionLibrary,
)
from mjlab.tasks.tracking.distillation.multi_motion import (
  MotionSlotAllocation,
  MultiMotionCommand,
)
from mjlab.tasks.tracking.distillation.observations import (
  ObservationSnapshot,
  pack_observations,
)
from mjlab.tasks.tracking.distillation.storage import LabeledReplayBuffer
from mjlab.tasks.tracking.distillation.teachers import (
  TeacherBank,
  build_cohort_teacher_bank,
  build_frozen_teacher,
)
from mjlab.tasks.tracking.distillation.vae_config import DEFAULT_SCHEMA, ModelSettings

OBS_WIDTH = 164
ACTION_DIM = 31
TEACHER_IDS = ("tiny_000", "tiny_001")
_TERM_NAMES_WIDTHS = (
  ("command", 62),
  ("motion_anchor_ori_b", 6),
  ("base_ang_vel", 3),
  ("joint_pos", 31),
  ("joint_vel", 31),
  ("actions", 31),
)


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


def make_library(root: Path) -> MotionLibrary:
  specs = [
    MotionClipSpec(
      teacher_id=teacher_id,
      motion_file=write_reference_clip(
        root / f"{teacher_id}.npz",
        frames=frames,
        joint_dim=ACTION_DIM,
        bodies=2,
        joint_base=100.0 * index,
        body_base=100.0 * index,
      ),
      teacher_code=index,
    )
    for index, (teacher_id, frames) in enumerate(zip(TEACHER_IDS, (5, 3), strict=True))
  ]
  return MotionLibrary.from_clips(specs)


def make_cohort() -> Any:
  terms = tuple(
    SimpleNamespace(name=name, width=width) for name, width in _TERM_NAMES_WIDTHS
  )
  return SimpleNamespace(
    observations=SimpleNamespace(
      terms=terms, names=tuple(name for name, _ in _TERM_NAMES_WIDTHS)
    ),
    actions=SimpleNamespace(
      dim=ACTION_DIM,
      joint_names=tuple(f"joint_{index:02d}" for index in range(ACTION_DIM)),
    ),
  )


class FakeMultiMotionCommand(MultiMotionCommand):
  """Multi-motion command with directly owned row state and metric tensors."""

  pending_events: ReferenceBoundaryEvents | None = None
  _body_pos_w: torch.Tensor
  _body_quat_w: torch.Tensor
  _robot_body_pos_w: torch.Tensor
  _robot_body_quat_w: torch.Tensor
  _anchor_pos_w: torch.Tensor
  _robot_anchor_pos_w: torch.Tensor

  @property
  def body_pos_w(self) -> torch.Tensor:
    return self._body_pos_w

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self._body_quat_w

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self._robot_body_pos_w

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self._robot_body_quat_w

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return self._anchor_pos_w

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self._robot_anchor_pos_w

  def consume_boundary_events(
    self, pre_generation: torch.Tensor | None = None
  ) -> ReferenceBoundaryEvents:
    """Return the injected events once, or an empty event set."""
    del pre_generation
    events = self.pending_events
    self.pending_events = None
    if events is not None:
      return events
    return ReferenceBoundaryEvents.empty(
      self.motion_ids.numel(), self.motion_ids.device
    )


def make_fake_command(
  library: MotionLibrary,
  row_motion_ids: tuple[int, ...],
  *,
  teacher_codes: tuple[int, ...] | None = None,
  frames: tuple[int, ...] | None = None,
) -> FakeMultiMotionCommand:
  num_envs = len(row_motion_ids)
  command: Any = object.__new__(FakeMultiMotionCommand)
  command.cfg = SimpleNamespace(entity_name="robot")
  command.library = library
  command.motion_ids = torch.tensor(row_motion_ids, dtype=torch.long)
  codes = (
    teacher_codes
    if teacher_codes is not None
    else tuple(library.clips[motion].teacher_code for motion in row_motion_ids)
  )
  command._teacher_codes = torch.tensor(codes, dtype=torch.long)
  command.time_steps = torch.tensor(
    frames if frames is not None else (0,) * num_envs, dtype=torch.long
  )
  command.segment_ids = torch.zeros(num_envs, dtype=torch.long)
  command.generation_ids = torch.zeros(num_envs, dtype=torch.long)
  command.pending_events = None
  command._body_pos_w = torch.zeros(num_envs, 2, 3)
  command._body_quat_w = torch.zeros(num_envs, 2, 4)
  command._body_quat_w[:, :, 0] = 1.0
  command._robot_body_pos_w = torch.zeros(num_envs, 2, 3)
  command._robot_body_quat_w = command._body_quat_w.clone()
  command._anchor_pos_w = torch.zeros(num_envs, 3)
  command._robot_anchor_pos_w = torch.zeros(num_envs, 3)
  return command


class FakeRouterEnv:
  """Minimal environment whose rows can be re-pinned mid-step, like a reset."""

  def __init__(self, command: FakeMultiMotionCommand) -> None:
    self.num_envs = command.motion_ids.numel()
    self.device = torch.device("cpu")
    self.command = command
    self.command_manager = SimpleNamespace(get_term=lambda name: command)
    self.scene = {
      "robot": SimpleNamespace(
        data=SimpleNamespace(projected_gravity_b=torch.zeros(self.num_envs, 3))
      )
    }
    self.observation_calls = 0
    self.actions: list[torch.Tensor] = []
    self.step_calls = 0
    self.initial_frames = command.time_steps.detach().clone()
    self._observations = {"actor": self._observations_for(command.time_steps)}
    self.on_step: Any = None
    self.next_events: ReferenceBoundaryEvents | None = None

  def _observations_for(self, steps: torch.Tensor) -> torch.Tensor:
    values = torch.zeros(self.num_envs, OBS_WIDTH)
    values[:, :ACTION_DIM] = steps.to(torch.float32)[:, None]
    values[:, ACTION_DIM : 2 * ACTION_DIM] = steps.to(torch.float32)[:, None] + 100.0
    return values

  def observation_for_frame(self, frame: int) -> torch.Tensor:
    """The single-row actor observation this fake produces at ``frame``."""
    values = torch.zeros(1, OBS_WIDTH)
    values[:, :ACTION_DIM] = float(frame)
    values[:, ACTION_DIM : 2 * ACTION_DIM] = float(frame) + 100.0
    return values

  def get_observations(self) -> dict[str, torch.Tensor]:
    self.observation_calls += 1
    return self._observations

  def reset(self, seed: int | None = None) -> None:
    del seed
    self.command.time_steps = self.initial_frames.clone()
    self._observations = {"actor": self._observations_for(self.command.time_steps)}

  def step(self, action: torch.Tensor) -> tuple[Any, ...]:
    self.actions.append(action.detach().clone())
    self.step_calls += 1
    self.command.time_steps = self.command.time_steps + 1
    self._observations = {"actor": self._observations_for(self.command.time_steps)}
    terminated = torch.zeros(self.num_envs, dtype=torch.bool)
    time_outs = torch.zeros(self.num_envs, dtype=torch.bool)
    if self.on_step is not None:
      terminated = self.on_step(self.command, self.step_calls)
    self.command.pending_events = self.next_events
    return (
      self._observations,
      torch.zeros(self.num_envs),
      terminated,
      time_outs,
      {},
    )

  def close(self) -> None:
    pass


def make_multi_adapter(
  command: FakeMultiMotionCommand, bank: TeacherBank
) -> tuple[MultiMotionDistillationAdapter, FakeRouterEnv]:
  env = FakeRouterEnv(command)
  adapter = MultiMotionDistillationAdapter(
    env, make_cohort(), bank, audit=cast(Any, object())
  )
  return adapter, env


# Snapshot identity and routing metadata.


def test_snapshot_owns_per_row_codes_aligned_with_the_other_row_tensors(
  tmp_path: Path,
) -> None:
  library = make_library(tmp_path)
  command = make_fake_command(library, (1, 0, 1, 0), frames=(2, 1, 0, 3))
  adapter, env = make_multi_adapter(command, make_bank())
  snapshot = adapter.snapshot()

  assert snapshot.motion_id.tolist() == [1, 0, 1, 0]
  assert snapshot.teacher_codes is not None
  assert snapshot.teacher_codes.tolist() == [1, 0, 1, 0]
  assert snapshot.teacher_code == -1
  assert snapshot.reference_frame.tolist() == [2, 1, 0, 3]
  assert snapshot.segment_id.shape == (4,)
  assert snapshot.generation_id.shape == (4,)
  # One cached actor observation: features and teacher inputs share it.
  assert env.observation_calls == 1
  torch.testing.assert_close(
    snapshot.features.reference_q, snapshot.teacher_observation[:, :ACTION_DIM]
  )

  # The snapshot owns its tensors: mutating the live cache and the command's
  # row state afterwards cannot change what the collector will label and store.
  env._observations["actor"][:] = -1000.0
  command.motion_ids[:] = 0
  command._teacher_codes[:] = 0
  command.time_steps[:] = 99
  command.segment_ids[:] = 99
  command.generation_ids[:] = 99
  assert snapshot.motion_id.tolist() == [1, 0, 1, 0]
  assert snapshot.teacher_codes is not None
  assert snapshot.teacher_codes.tolist() == [1, 0, 1, 0]
  assert snapshot.reference_frame.tolist() == [2, 1, 0, 3]
  assert snapshot.segment_id.tolist() == [0, 0, 0, 0]
  assert snapshot.generation_id.tolist() == [0, 0, 0, 0]
  assert snapshot.teacher_observation[0, 0].item() == 2.0
  assert snapshot.features.reference_q[0, 0].item() == 2.0


def test_snapshot_rejects_a_mapping_that_disagrees_with_the_library(
  tmp_path: Path,
) -> None:
  library = make_library(tmp_path)
  command = make_fake_command(library, (0, 1), teacher_codes=(1, 0))
  adapter, _ = make_multi_adapter(command, make_bank())
  with pytest.raises(ValueError, match="motion-to-teacher mapping"):
    adapter.snapshot()


def test_snapshot_rejects_malformed_row_metadata() -> None:
  values = torch.zeros(2, 31)
  features = ObservationSnapshot(
    reference_q=values,
    reference_dq=values,
    anchor_orientation_error=torch.zeros(2, 6),
    projected_gravity=torch.zeros(2, 3),
    gyro=torch.zeros(2, 3),
    relative_joint_q=values,
    joint_dq=values,
    previous_action=values,
  )

  def build(**kwargs: Any) -> DistillationSnapshot:
    return DistillationSnapshot(
      teacher_observation=torch.zeros(2, OBS_WIDTH),
      features=features,
      packed=pack_observations(features),
      teacher_id="fake",
      teacher_code=kwargs.pop("teacher_code", 0),
      motion_id=torch.zeros(2, dtype=torch.int64),
      reference_frame=torch.zeros(2, dtype=torch.int64),
      segment_id=torch.zeros(2, dtype=torch.int64),
      generation_id=torch.zeros(2, dtype=torch.int64),
      **kwargs,
    )

  with pytest.raises(ValueError, match="teacher_codes must have shape"):
    build(teacher_codes=torch.zeros(3, dtype=torch.int64))
  with pytest.raises(ValueError, match="integer tensor"):
    build(teacher_codes=torch.zeros(2, dtype=torch.float32))
  with pytest.raises(ValueError, match="non-negative"):
    build(teacher_codes=torch.tensor([0, -1]), teacher_code=-1)
  with pytest.raises(ValueError, match="disagree with the scalar teacher_code"):
    build(teacher_codes=torch.tensor([0, 1]), teacher_code=0)
  with pytest.raises(ValueError, match="teacher_codes must be a torch.Tensor"):
    build(teacher_codes=[0, 1])
  # A uniformly coded batch with a matching scalar identity stays valid.
  assert build(teacher_codes=torch.tensor([0, 0])).teacher_codes is not None


# Mixed collection routing.


def _expected_labels(bank: TeacherBank, snapshot: DistillationSnapshot) -> torch.Tensor:
  """Label each row with its own teacher, independently of the collector."""
  assert snapshot.teacher_codes is not None
  expected = torch.zeros(snapshot.teacher_observation.shape[0], bank.action_dim)
  for code in torch.unique(snapshot.teacher_codes).tolist():
    rows = snapshot.teacher_codes == code
    expected[rows] = bank.teacher(code).label(snapshot.teacher_observation[rows])
  return expected


def _collector(
  adapter: Any, bank: TeacherBank, student: ConditionalVAE | None = None
) -> tuple[DAggerCollector, LabeledReplayBuffer]:
  replay = LabeledReplayBuffer(64, DEFAULT_SCHEMA)
  return (
    DAggerCollector(adapter, bank, student or _student(), replay),
    replay,
  )


def _student() -> ConditionalVAE:
  return ConditionalVAE(DEFAULT_SCHEMA, ModelSettings(hidden_dims=(8, 8)))


def test_mixed_collection_stores_pre_step_labels_and_row_identity(
  tmp_path: Path,
) -> None:
  library = make_library(tmp_path)
  initial_frames = torch.tensor([2, 0, 1, 2])
  command = make_fake_command(
    library, (1, 0, 1, 0), frames=tuple(initial_frames.tolist())
  )
  bank = make_bank()
  adapter, env = make_multi_adapter(command, bank)
  collector, replay = _collector(adapter, bank)
  pre_step = adapter.snapshot()

  result = collector.collect(
    CollectionConfig(steps=2, teacher_probability=1.0), reset=True
  )

  stored = replay.sample(replay.size, replacement=False)
  assert stored.batch_size == 8
  # Each stored row is labeled by the teacher its own captured code names,
  # applied to the observation that produced its captured frame.  A swap of
  # rows, teachers, or steps would break these equalities.
  for row in range(stored.batch_size):
    code = int(stored.teacher_id[row].item())
    frame = int(stored.reference_frame[row].item())
    expected = bank.teacher(code).label(env.observation_for_frame(frame))
    torch.testing.assert_close(stored.teacher_action[row : row + 1], expected)
    assert int(stored.motion_id[row].item()) == code
  assert sorted(set(stored.reference_frame.tolist())) == [0, 1, 2, 3]
  assert stored.episode_id.tolist() == [0] * 8
  assert pre_step.reference_frame.tolist() == initial_frames.tolist()
  assert pre_step.motion_id.tolist() == [1, 0, 1, 0]
  assert adapter.snapshot().reference_frame.tolist() == [4, 2, 3, 4]
  assert result.samples == 8
  assert result.teacher_steps == 8
  assert result.student_steps == 0
  assert len(result.motion_stats) == 2
  assert {item.motion_id for item in result.motion_stats} == {0, 1}
  assert sum(item.samples for item in result.motion_stats) == 8
  assert all(
    item.teacher_steps + item.student_steps == item.samples
    for item in result.motion_stats
  )
  assert {item.motion_id: item.teacher_code for item in result.motion_stats} == {
    0: 0,
    1: 1,
  }
  assert {item.motion_id: item.teacher_id for item in result.motion_stats} == {
    0: "tiny_000",
    1: "tiny_001",
  }
  assert env.step_calls == 2
  assert result.disagreement_mean > 0.0
  assert all(item.disagreement_mean is not None for item in result.motion_stats)


def test_boundary_attribution_vectors_must_align_with_the_mention_list() -> None:
  with pytest.raises(ValueError, match="before_motion_ids must align"):
    SegmentBoundary(
      env_indices=(0, 1),
      reason="generation",
      before_segment=(0, 0),
      after_segment=(1, 0),
      before_generation=(0, 0),
      after_generation=(1, 0),
      before_motion_ids=(0,),
      before_teacher_codes=(0, 0),
    )
  # A boundary without attribution stays valid for producers that report none.
  boundary = SegmentBoundary((0,), "generation", (0,), (1,), (0,), (1,))
  assert boundary.before_motion_ids == ()


def test_subset_reset_attributes_boundaries_to_the_motion_that_started_it(
  tmp_path: Path,
) -> None:
  library = make_library(tmp_path)
  command = make_fake_command(library, (0, 1, 0, 1), frames=(1, 0, 2, 1))
  bank = make_bank()
  adapter, env = make_multi_adapter(command, bank)

  def reset_rows(command_: FakeMultiMotionCommand, call: int) -> torch.Tensor:
    del call
    # Rows 0 and 2 auto-reset into the other clip, exactly like a per-row reset
    # whose freshly pinned clip differs from the segment that just ended.
    ids = torch.tensor([0, 2])
    command_.motion_ids[ids] = 1 - command_.motion_ids[ids]
    command_._teacher_codes[ids] = command_.motion_ids[ids]
    command_.segment_ids[ids] += 1
    command_.generation_ids[ids] += 1
    command_.time_steps[ids] = 0
    return torch.tensor([True, False, True, False])

  env.on_step = reset_rows
  collector, replay = _collector(adapter, bank)
  result = collector.collect(CollectionConfig(steps=1), reset=True)

  final = [
    boundary for boundary in result.boundaries if boundary.reason != "explicit_reset"
  ]
  assert len(final) == 1
  assert final[0].env_indices == (0, 2)
  assert final[0].before_motion_ids == (0, 0)
  assert final[0].before_teacher_codes == (0, 0)
  # A terminated row is reported once as an event reason and once as the flag;
  # the composed reason names each token exactly once.
  assert final[0].reason == "terminated+generation"
  stats = {item.motion_id: item for item in result.motion_stats}
  assert stats[0].samples == 2
  assert stats[1].samples == 2
  assert stats[0].boundaries == 2
  assert stats[1].boundaries == 0
  # The two untouched rows keep their clip in the next snapshot.
  assert adapter.snapshot().motion_id.tolist() == [1, 1, 1, 1]
  assert replay.size == 4


def test_boundary_reason_merges_event_tokens_without_loss_or_repetition(
  tmp_path: Path,
) -> None:
  library = make_library(tmp_path)
  command = make_fake_command(library, (0, 1), frames=(2, 1))
  bank = make_bank()
  adapter, env = make_multi_adapter(command, bank)
  # Row 0 both timers out and completes its reference inside the step, then
  # terminates; row 1 only moves forward.  The event reason is itself a joined
  # group, so the composed reason must keep every token and repeat none.
  env.next_events = ReferenceBoundaryEvents(
    available=torch.tensor([True, False]),
    completed=torch.tensor([True, False]),
    interrupted=torch.tensor([True, False]),
    pre_generation=torch.zeros(2, dtype=torch.long),
    post_generation=torch.zeros(2, dtype=torch.long),
    completed_generation=torch.zeros(2, dtype=torch.long),
    interrupted_generation=torch.tensor([0, -1]),
    reasons=("timer_resampled+reference_completed", "unavailable"),
  )

  def terminate_row_zero(command_: FakeMultiMotionCommand, call: int) -> torch.Tensor:
    del call, command_
    return torch.tensor([True, False])

  env.on_step = terminate_row_zero
  collector, _ = _collector(adapter, bank)
  result = collector.collect(CollectionConfig(steps=1), reset=True)

  boundary = next(item for item in result.boundaries if item.reason != "explicit_reset")
  assert boundary.env_indices == (0,)
  assert boundary.reason == "timer_resampled+reference_completed+terminated"
  stats = {item.motion_id: item for item in result.motion_stats}
  assert stats[0].boundaries == 1
  assert stats[1].boundaries == 0


def test_unequal_clip_wrap_keeps_the_label_on_the_starting_motion(
  tmp_path: Path,
) -> None:
  library = make_library(tmp_path)
  # Row 0 is on the 5-frame clip and wraps immediately; row 1 is on the
  # 3-frame clip and has two frames left.
  command = make_fake_command(library, (0, 1), frames=(4, 1))
  bank = make_bank()
  adapter, env = make_multi_adapter(command, bank)

  def wrap_short_clip(command_: FakeMultiMotionCommand, call: int) -> torch.Tensor:
    del call
    ids = torch.tensor([0])
    command_.time_steps[ids] = 0
    command_.segment_ids[ids] += 1
    command_.generation_ids[ids] += 1
    return torch.zeros(2, dtype=torch.bool)

  env.on_step = wrap_short_clip
  collector, _ = _collector(adapter, bank)
  result = collector.collect(CollectionConfig(steps=1), reset=True)

  boundary = next(item for item in result.boundaries if item.reason != "explicit_reset")
  assert boundary.reason == "generation"
  assert boundary.env_indices == (0,)
  assert boundary.before_motion_ids == (0,)
  assert boundary.before_teacher_codes == (0,)
  stats = {item.motion_id: item for item in result.motion_stats}
  assert stats[0].boundaries == 1
  assert stats[1].boundaries == 0
  assert stats[0].reference_frame_max == 4
  assert stats[1].reference_frame_max == 1
  assert stats[1].reference_frames_observed == 1


def test_mixed_batch_needs_a_bank_and_a_valid_code(tmp_path: Path) -> None:
  library = make_library(tmp_path)
  command = make_fake_command(library, (0, 1), frames=(1, 1))
  bank = make_bank()
  adapter, env = make_multi_adapter(command, bank)

  single = DAggerCollector(
    adapter, bank.teacher(0), _student(), LabeledReplayBuffer(8, DEFAULT_SCHEMA)
  )
  with pytest.raises(TeacherRoutingError, match="different teacher codes"):
    single.collect(CollectionConfig(steps=1), reset=True)
  assert env.step_calls == 0
  with pytest.raises(CollectionNumericalError, match="reset=True"):
    single.collect(CollectionConfig(steps=1), reset=False)

  forged = make_fake_command(library, (0, 1), teacher_codes=(0, 9))
  forged_adapter, forged_env = make_multi_adapter(forged, bank)
  forged_collector, forged_replay = _collector(forged_adapter, bank)
  with pytest.raises(ValueError, match="motion-to-teacher mapping"):
    forged_collector.collect(CollectionConfig(steps=1), reset=True)
  assert forged_env.step_calls == 0
  assert forged_replay.size == 0

  # A code outside the bank is rejected by the bank itself, before any step.
  unknown = ScriptedRowAdapter(ticks=(((0, 1), (5, 1)),))
  unknown_collector = DAggerCollector(
    unknown, bank, _student(), LabeledReplayBuffer(8, DEFAULT_SCHEMA)
  )
  with pytest.raises(ValueError, match="out of range"):
    unknown_collector.collect(CollectionConfig(steps=1), reset=True)
  assert unknown.index == 0
  with pytest.raises(CollectionNumericalError, match="reset=True"):
    unknown_collector.collect(CollectionConfig(steps=1), reset=False)


def test_a_snapshot_without_codes_cannot_be_routed_by_a_bank(
  tmp_path: Path,
) -> None:
  library = make_library(tmp_path)
  command = make_fake_command(library, (0, 1))
  bank = make_bank()
  adapter, env = make_multi_adapter(command, bank)

  class LegacyAdapter:
    """A pre-routing adapter that only exposes the scalar teacher identity."""

    auto_reset = True

    def __init__(self, inner: MultiMotionDistillationAdapter) -> None:
      self.inner = inner
      self.current = inner.snapshot()

    def reset(self, seed: int | None = None) -> DistillationSnapshot:
      self.current = self.inner.reset(seed)
      return replace(
        self.current,
        teacher_id="legacy",
        teacher_code=0,
        teacher_codes=None,
      )

    def step(self, action: torch.Tensor) -> DistillationStep:
      step = self.inner.step(action)
      self.current = replace(
        step.snapshot, teacher_id="legacy", teacher_code=0, teacher_codes=None
      )
      return replace(step, snapshot=self.current)

  collector, replay = _collector(LegacyAdapter(adapter), bank)
  with pytest.raises(TeacherRoutingError, match="per-row snapshot.teacher_codes"):
    collector.collect(CollectionConfig(steps=1), reset=True)
  assert env.step_calls == 0
  assert replay.size == 0


def test_pre_failure_labels_stay_and_teacher_parameters_stay_frozen(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  library = make_library(tmp_path)
  command = make_fake_command(library, (0, 1), frames=(0, 0))
  bank = make_bank()
  adapter, env = make_multi_adapter(command, bank)
  student = _student()
  before = {key: value.clone() for key, value in bank.state_dict().items()}
  original = student.mean_inference
  calls = 0

  def fail_on_second(
    reference: torch.Tensor, conditioning: torch.Tensor
  ) -> torch.Tensor:
    nonlocal calls
    calls += 1
    action = original(reference, conditioning)
    if calls == 2:
      action = torch.full_like(action, float("nan"))
    return action

  monkeypatch.setattr(student, "mean_inference", fail_on_second)
  collector, replay = _collector(adapter, bank, student)
  with pytest.raises(CollectionNumericalError, match="student action"):
    collector.collect(CollectionConfig(steps=3))
  # Tick one is stored, and tick two keeps its valid pre-failure teacher label
  # without any simulator step.
  assert replay.size == 4
  assert env.step_calls == 1
  stored = replay.sample(replay.size, replacement=False)
  assert sorted(stored.motion_id.tolist()) == [0, 0, 1, 1]
  for code in (0, 1):
    rows = stored.teacher_id == code
    finite = stored.teacher_action[rows]
    assert torch.isfinite(finite).all()
  for key, value in bank.state_dict().items():
    assert torch.equal(value, before[key]), key
  with pytest.raises(CollectionNumericalError, match="reset=True"):
    collector.collect(CollectionConfig(steps=1), reset=False)
  monkeypatch.undo()
  collector.collect(CollectionConfig(steps=1), reset=True)


@dataclass
class ScriptedRowAdapter:
  """Adapter that yields snapshots with explicitly scripted row identity.

  The multi-motion adapter validates its own mapping before a snapshot leaves
  the boundary, so a mapping that changes mid-collection can only be produced by
  a custom adapter like this one; the collector must still reject it instead of
  storing two teachers' labels under one motion id.
  """

  ticks: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
  auto_reset: bool = True
  index: int = 0

  def __post_init__(self) -> None:
    self.batch = len(self.ticks[0][0])

  def _snapshot(self, index: int) -> DistillationSnapshot:
    motion_ids, teacher_codes = self.ticks[index]
    values = torch.full((self.batch, ACTION_DIM), float(index))
    features = ObservationSnapshot(
      reference_q=values,
      reference_dq=values + 1,
      anchor_orientation_error=torch.zeros(self.batch, 6),
      projected_gravity=torch.zeros(self.batch, 3),
      gyro=torch.zeros(self.batch, 3),
      relative_joint_q=values + 2,
      joint_dq=values + 3,
      previous_action=values - 1,
    )
    return DistillationSnapshot(
      teacher_observation=torch.cat(
        (values, values, torch.zeros(self.batch, OBS_WIDTH - 2 * ACTION_DIM)), 1
      ),
      features=features,
      packed=pack_observations(features),
      teacher_id="scripted",
      teacher_code=-1,
      motion_id=torch.tensor(motion_ids, dtype=torch.int64),
      reference_frame=torch.full((self.batch,), index, dtype=torch.int64),
      segment_id=torch.full((self.batch,), index, dtype=torch.int64),
      generation_id=torch.full((self.batch,), index, dtype=torch.int64),
      teacher_codes=torch.tensor(teacher_codes, dtype=torch.int64),
    )

  def reset(self, seed: int | None = None) -> DistillationSnapshot:
    del seed
    self.index = 0
    return self._snapshot(0)

  def step(self, action: torch.Tensor) -> DistillationStep:
    self.index += 1
    snapshot = self._snapshot(min(self.index, len(self.ticks) - 1))
    zeros = torch.zeros(self.batch)
    return DistillationStep(
      snapshot,
      torch.zeros(self.batch),
      zeros.to(torch.bool),
      zeros.to(torch.bool),
      {},
    )


def test_teacher_parameters_and_mapping_are_stable_across_a_collection(
  tmp_path: Path,
) -> None:
  library = make_library(tmp_path)
  command = make_fake_command(library, (0, 1, 0, 1), frames=(0, 2, 4, 1))
  bank = make_bank()
  before = {key: value.clone() for key, value in bank.state_dict().items()}
  adapter, _ = make_multi_adapter(command, bank)
  collector, _ = _collector(adapter, bank)
  result = collector.collect(CollectionConfig(steps=2, teacher_probability=0.0))
  # A student-only rollout still records the captured routing metadata, so the
  # labels can be attributed to the right motion even when no teacher acts.
  assert result.teacher_steps == 0
  assert result.student_steps == 8
  assert all(item.teacher_steps == 0 for item in result.motion_stats)
  assert {item.motion_id: item.samples for item in result.motion_stats} == {0: 4, 1: 4}
  for key, value in bank.state_dict().items():
    assert torch.equal(value, before[key]), key

  scripted = ScriptedRowAdapter(
    ticks=(
      ((0, 1), (0, 1)),
      ((0, 1), (1, 1)),
    )
  )
  replay = LabeledReplayBuffer(16, DEFAULT_SCHEMA)
  scripted_collector = DAggerCollector(scripted, bank, _student(), replay)
  with pytest.raises(
    TeacherRoutingError, match="motion 0 was labeled by teacher codes 0 and 1"
  ):
    scripted_collector.collect(CollectionConfig(steps=2, teacher_probability=1.0))
  assert scripted.index == 1


# Evaluation attribution.


def test_evaluation_attributes_segments_to_the_motion_that_started_them(
  tmp_path: Path,
) -> None:
  library = make_library(tmp_path)
  command = make_fake_command(library, (0, 1), frames=(1, 0))
  bank = make_bank()
  adapter, env = make_multi_adapter(command, bank)

  def reset_row_zero(command_: FakeMultiMotionCommand, call: int) -> torch.Tensor:
    if call > 1:
      return torch.zeros(2, dtype=torch.bool)
    command_.motion_ids[0] = 1
    command_._teacher_codes[0] = 1
    command_.time_steps[0] = 0
    command_.segment_ids[0] += 1
    command_.generation_ids[0] += 1
    return torch.tensor([True, False])

  env.on_step = reset_row_zero
  result = evaluate_distillation(
    adapter, bank, _student(), mode="student", steps=2, seed=3
  )

  assert result.settings["motion_ids"] == [0, 1]
  assert result.settings["motion_attribution"] == "pre_step_snapshot_motion_id"
  started = [item for item in result.segments if item.motion_id == 0]
  assert len(started) == 1
  assert started[0].teacher_code == 0
  assert started[0].env_index == 0
  assert started[0].outcome == "failure"
  assert started[0].steps == 1
  # The failure belongs to the motion the segment started in, and the segment
  # the row was reset into is counted separately under its own motion.
  assert [item.motion_id for item in result.per_motion] == [0, 1]
  first = result.per_motion[0]
  assert first.teacher_code == 0
  assert first.outcomes == {"failure": 1}
  assert first.segments == 1
  assert first.steps == 1
  assert first.completion_known_segments == 1
  assert first.completion_rate == pytest.approx(0.0)
  assert first.failure_rate == pytest.approx(1.0)
  assert first.as_dict()["motion_id"] == 0
  second = result.per_motion[1]
  assert second.teacher_code == 1
  assert second.outcomes == {"step_cap": 2}
  assert second.completion_rate is None
  assert second.failure_rate is None
  # Per-motion metric means are unweighted over that motion's own segments.
  assert second.metrics["tracking_pose_error"] == pytest.approx(0.0)
  assert "tracking_pose_error" in first.metrics
  assert first.metrics["teacher_student_disagreement"] > 0.0


def test_evaluation_routes_teacher_mode_actions_per_row(tmp_path: Path) -> None:
  library = make_library(tmp_path)
  command = make_fake_command(library, (1, 0, 1, 0), frames=(0, 0, 0, 0))
  bank = make_bank()
  adapter, env = make_multi_adapter(command, bank)
  snapshot = adapter.snapshot()
  expected = _expected_labels(bank, snapshot)

  result = evaluate_distillation(adapter, bank, None, mode="teacher", steps=1, seed=1)

  assert result.settings["num_envs"] == 4
  # Teacher-mode rollouts execute the routed teacher command for every row.
  assert env.actions[0].shape == (4, ACTION_DIM)
  torch.testing.assert_close(env.actions[0], expected)
  assert all(
    item.metrics["teacher_student_disagreement"] == 0.0 for item in result.segments
  )


# Real cohort and real environment evidence.


@pytest.fixture(scope="module")
def real_cohort() -> Any:
  return resolve_cohort(
    load_manifest("configs/distillation/x2_tennis.yaml", repo_root=Path("."))
  )


@pytest.fixture(scope="module")
def real_multi_env(real_cohort: Any) -> Any:
  output = io.StringIO()
  with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
    # A reversed request still resolves through the manifest order, so numeric
    # motion ids never depend on the caller's argument order.
    env = build_multi_motion_environment(
      real_cohort,
      ("tennis_001", "tennis_000"),
      phase_policy="uniform",
      num_envs=4,
      device="cpu",
      seed=7,
    )
  try:
    yield env
  finally:
    env.close()


@pytest.fixture(scope="module")
def real_multi_adapter(real_cohort: Any) -> Any:
  output = io.StringIO()
  with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
    adapter = make_multi_teacher_distillation_adapter(
      real_cohort,
      ("tennis_000", "tennis_001"),
      phase_policy="uniform",
      num_envs=4,
      device="cpu",
      seed=7,
    )
  try:
    yield adapter
  finally:
    adapter.close()


def test_real_multi_motion_audit_covers_both_clips(
  real_cohort: Any, real_multi_env: Any
) -> None:
  audit = validate_multi_motion_live_contract(
    real_multi_env, real_cohort, ("tennis_001", "tennis_000")
  )
  assert audit.teacher_ids == ("tennis_000", "tennis_001")
  assert audit.phase_policy == "uniform"
  assert audit.joint_names == real_cohort.actions.joint_names
  assert audit.control_period_s == real_cohort.control.control_period_s
  assert audit.mapping_digest
  assert audit.slots.counts == (2, 2)
  assert [clip.teacher_id for clip in audit.asset.clips] == [
    "tennis_000",
    "tennis_001",
  ]
  assert [clip.frames for clip in audit.asset.clips] == [453, 340]
  assert [clip.fps for clip in audit.asset.clips] == pytest.approx([50.0, 50.0])
  assert [clip.teacher_code for clip in audit.asset.clips] == [0, 1]
  for clip in audit.asset.clips:
    assert clip.body_reference_compared == (
      "body_pos_w",
      "body_quat_w",
      "body_lin_vel_w",
      "body_ang_vel_w",
    )
    assert len(clip.tracked_body_indices) == len(real_cohort.body_names)
    for name, maximum in clip.body_reference_max_abs_error:
      assert maximum <= clip.body_reference_atol, (clip.teacher_id, name, maximum)
  assert len(audit.asset.tracked_body_indices) == len(real_cohort.body_names)
  assert any(
    item.expected_type == "gyro" and item.frame_verified
    for item in audit.asset.sensor_evidence
  )
  assert audit.asset.unresolved == ()
  assert "sampling_mode" in " ".join(audit.semantic_overrides)


def test_real_single_teacher_validation_refuses_a_mixed_environment(
  real_cohort: Any, real_multi_env: Any
) -> None:
  with pytest.raises(ValueError, match="mixed-slot"):
    validate_live_contract(real_multi_env, real_cohort, "tennis_000")


def test_real_adapter_routes_rows_and_reads_one_cached_observation(
  real_multi_adapter: Any, real_cohort: Any
) -> None:
  adapter = real_multi_adapter
  env = adapter.env
  calls = 0
  original = env.get_observations

  def counted() -> dict[str, torch.Tensor]:
    nonlocal calls
    calls += 1
    return original()

  env.get_observations = counted
  try:
    snapshot = adapter.reset(seed=7)
    command = env.command_manager.get_term("motion")
    assert isinstance(command, MultiMotionCommand)
    assert snapshot.teacher_codes is not None
    assert snapshot.motion_id.tolist() == command.motion_ids.tolist()
    assert snapshot.teacher_codes.tolist() == command.teacher_codes.tolist()
    assert snapshot.teacher_code == -1
    assert snapshot.teacher_id == "multiple"
    assert adapter.motion_teacher_codes == {0: 0, 1: 1}
    assert adapter.teacher_ids == ("tennis_000", "tennis_001")
    assert snapshot.packed.reference.shape[1] == DEFAULT_SCHEMA.reference_dim
    assert snapshot.packed.conditioning.shape[1] == DEFAULT_SCHEMA.conditioning_dim
    assert calls == 1
    # Every row reports a clip-local frame inside its own clip.
    lengths = torch.tensor(
      [command.library.clip(int(motion)).frames for motion in command.motion_ids]
    )
    assert bool((snapshot.reference_frame < lengths).all())
    assert snapshot.segment_id.shape == (env.num_envs,)
    assert snapshot.boundary_events is not None
    labels = adapter.bank.label(snapshot.teacher_codes, snapshot.teacher_observation)
    assert labels.shape == (env.num_envs, adapter.bank.action_dim)
    assert torch.isfinite(labels).all()
    for code in (0, 1):
      rows = snapshot.teacher_codes == code
      expected = adapter.bank.teacher(code).label(snapshot.teacher_observation[rows])
      torch.testing.assert_close(labels[rows], expected)
    assert calls == 1

    # Re-pinning one row selects the other clip and resets exactly that row; the
    # routing identity follows the clip, never the argument order.
    other = 1 - int(command.motion_ids[0].item())
    before = command.motion_ids.clone()
    command.select_motion_ids(torch.tensor([0]), torch.tensor([other]))
    pinned = adapter.snapshot()
    assert int(pinned.motion_id[0].item()) == other
    assert int(pinned.teacher_codes[0].item()) == other
    assert bool((command.motion_ids[1:] == before[1:]).all())
    assert int(pinned.reference_frame[0].item()) < command.library.clip(other).frames
  finally:
    env.get_observations = original


def test_real_teacher_bank_codes_match_the_library_clip_codes(
  real_cohort: Any,
) -> None:
  bank = build_cohort_teacher_bank(real_cohort, device="cpu")
  assert bank.teacher_ids == real_cohort.teacher_ids
  for code, teacher_id in enumerate(real_cohort.teacher_ids):
    assert bank.code(teacher_id) == code
    assert bank.teacher_id(code) == teacher_id
  slots = MotionSlotAllocation(
    weights=(1.0, 1.0),
    counts=(2, 2),
    teacher_ids=real_cohort.teacher_ids,
    row_motion_ids=(0, 0, 1, 1),
  )
  assert slots.as_dict()["counts"] == [2, 2]
