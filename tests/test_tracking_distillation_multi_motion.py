"""CPU tests for the opt-in multi-motion command, slots, and build plan.

The command tests drive a fake environment that mirrors the real reset and
compute contract (reset resamples the selected rows, forwards the simulator, and
advances commands with ``dt=0``); no MuJoCo model is compiled here, so live
environment evidence stays with the parent-monitored smoke run.
"""

from __future__ import annotations

import contextlib
import io
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import torch
from tracking_distillation_fixtures import build_tiny_cohort, write_reference_clip

from mjlab.tasks.tracking.distillation.config import load_manifest, resolve_cohort
from mjlab.tasks.tracking.distillation.environment import (
  build_multi_motion_environment,
)
from mjlab.tasks.tracking.distillation.motion_library import (
  MotionClipSpec,
  MotionLibrary,
  MotionLibraryError,
)
from mjlab.tasks.tracking.distillation.multi_motion import (
  MotionSlotAllocation,
  MultiMotionCommand,
  MultiMotionCommandCfg,
  MultiMotionError,
  PhasePolicy,
  make_multi_motion_cfg,
  plan_multi_motion,
  stratified_slot_allocation,
)
from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg
from mjlab.utils.lab_api.string import resolve_matching_names

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

JOINT_DIM = 3
SOURCE_BODIES = 4
ROBOT_BODIES = ("pelvis", "torso_link", "left_knee_link", "right_knee_link")
TRACKED_BODIES = ("pelvis", "torso_link")
TRACKED_INDICES = (0, 1)
SENTINEL_BASE = 1000.0
STANDING_START_TASK = (
  "Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR-Reduced-"
  "Perturbations-Standing-Start"
)


# Fake environment: only the surface the motion command consumes.


class _FakeRobotData:
  def __init__(self, num_envs: int, bodies: int, joints: int) -> None:
    self.joint_pos = torch.zeros(num_envs, joints)
    self.joint_vel = torch.zeros(num_envs, joints)
    self.body_link_pos_w = torch.zeros(num_envs, bodies, 3)
    self.body_link_quat_w = torch.zeros(num_envs, bodies, 4)
    self.body_link_quat_w[..., 0] = 1.0
    self.body_link_lin_vel_w = torch.zeros(num_envs, bodies, 3)
    self.body_link_ang_vel_w = torch.zeros(num_envs, bodies, 3)
    self.soft_joint_pos_limits = torch.zeros(num_envs, joints, 2)
    self.soft_joint_pos_limits[..., 0] = -1.0e6
    self.soft_joint_pos_limits[..., 1] = 1.0e6
    self.default_root_state = torch.zeros(num_envs, 13)
    self.default_root_state[:, 2] = 0.7
    self.default_joint_pos = torch.zeros(num_envs, joints)


class _FakeRobot:
  def __init__(self, num_envs: int, bodies: tuple[str, ...] = ROBOT_BODIES) -> None:
    self.body_names = tuple(bodies)
    self.data = _FakeRobotData(num_envs, len(self.body_names), JOINT_DIM)
    self.root_writes: list[torch.Tensor | None] = []

  def find_bodies(
    self, name_keys: tuple[str, ...], preserve_order: bool = False
  ) -> tuple[list[int], list[str]]:
    return resolve_matching_names(name_keys, self.body_names, preserve_order)

  def write_joint_state_to_sim(
    self,
    position: torch.Tensor,
    velocity: torch.Tensor,
    joint_ids: torch.Tensor | None = None,
    env_ids: torch.Tensor | None = None,
  ) -> None:
    del joint_ids
    self.data.joint_pos[env_ids] = position
    self.data.joint_vel[env_ids] = velocity

  def write_root_state_to_sim(
    self, root_state: torch.Tensor, env_ids: torch.Tensor | None = None
  ) -> None:
    self.root_writes.append(env_ids)
    self.data.body_link_pos_w[env_ids, 0] = root_state[:, 0:3]
    self.data.body_link_quat_w[env_ids, 0] = root_state[:, 3:7]

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    del env_ids


class _FakeSim:
  def __init__(self) -> None:
    self.forward_calls = 0

  def forward(self) -> None:
    self.forward_calls += 1


class _FakeScene:
  def __init__(self, robot: _FakeRobot, num_envs: int) -> None:
    self.robot = robot
    self.env_origins = torch.zeros(num_envs, 3)

  def __getitem__(self, name: str) -> _FakeRobot:
    assert name == "robot"
    return self.robot


class _FakeEnv:
  """Minimal environment that mirrors the real reset/compute contract."""

  def __init__(self, num_envs: int, bodies: tuple[str, ...] = ROBOT_BODIES) -> None:
    self.num_envs = num_envs
    self.device = "cpu"
    self.step_dt = 0.02
    self.robot = _FakeRobot(num_envs, bodies)
    self.scene = _FakeScene(self.robot, num_envs)
    self.sim = _FakeSim()
    self.command: MultiMotionCommand | None = None
    self.reset_calls: list[torch.Tensor] = []

  def reset(
    self, *, seed: int | None = None, env_ids: torch.Tensor | None = None
  ) -> None:
    del seed
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, dtype=torch.int64)
    assert self.command is not None
    self.reset_calls.append(env_ids.clone())
    self.command.reset(env_ids)
    self.sim.forward()
    self.command.compute(dt=0.0, env_ids=env_ids)


# Harness.


def make_library(
  root: Path,
  *,
  clip_frames: tuple[int, ...] = (2, 5),
  clip_bodies: int = SOURCE_BODIES,
  joint_dim: int = JOINT_DIM,
) -> MotionLibrary:
  specs = [
    MotionClipSpec(
      teacher_id=f"tiny_{index:03d}",
      motion_file=write_reference_clip(
        root / f"clip_{index}.npz",
        frames=count,
        joint_dim=joint_dim,
        bodies=clip_bodies,
        joint_base=SENTINEL_BASE * index,
        body_base=SENTINEL_BASE * index,
      ),
    )
    for index, count in enumerate(clip_frames)
  ]
  return MotionLibrary.from_clips(specs)


def make_slots(
  row_motion_ids: tuple[int, ...],
  *,
  weights: tuple[float, ...] | None = None,
  teacher_ids: tuple[str, ...] = ("tiny_000", "tiny_001"),
) -> MotionSlotAllocation:
  counts = Counter(row_motion_ids)
  return MotionSlotAllocation(
    weights=weights if weights is not None else (1.0,) * len(teacher_ids),
    counts=tuple(counts[index] for index in range(len(teacher_ids))),
    teacher_ids=teacher_ids,
    row_motion_ids=row_motion_ids,
  )


def make_command(
  library: MotionLibrary,
  slots: MotionSlotAllocation,
  *,
  num_envs: int = 4,
  phase_policy: PhasePolicy = "start",
  lookahead_s: float = 0.0,
  standing_start_prob: float = 0.0,
  robot_bodies: tuple[str, ...] = ROBOT_BODIES,
  body_names: tuple[str, ...] = TRACKED_BODIES,
  anchor_body_name: str | None = None,
) -> tuple[MultiMotionCommand, _FakeEnv]:
  cfg = MultiMotionCommandCfg(
    motion_file="",
    anchor_body_name=anchor_body_name or body_names[-1],
    body_names=body_names,
    entity_name="robot",
    resampling_time_range=(1.0e9, 1.0e9),
    sampling_mode=phase_policy,
    standing_start_prob=standing_start_prob,
    lookahead_s=lookahead_s,
    # Deterministic reset writes: no pose/velocity/joint perturbation, so the
    # written reference state can be compared exactly.
    joint_position_range=(0.0, 0.0),
    library=library,
    slots=slots,
  )
  env = _FakeEnv(num_envs, robot_bodies)
  command = MultiMotionCommand(cfg, cast("ManagerBasedRlEnv", env))
  env.command = command
  return command, env


def stabilize(command: MultiMotionCommand) -> None:
  """Keep the timer from expiring so stepping only exercises frames/wrap."""
  command.time_left[:] = 1.0e9


def source_joint_pos(
  library: MotionLibrary, motion_id: int, frame: int
) -> torch.Tensor:
  with np.load(library.clips[motion_id].motion_file, allow_pickle=False) as data:
    return torch.from_numpy(np.asarray(data["joint_pos"]))[frame]


def source_body_pos(
  library: MotionLibrary, motion_id: int, frame: int, tracked_index: int
) -> torch.Tensor:
  with np.load(library.clips[motion_id].motion_file, allow_pickle=False) as data:
    values = torch.from_numpy(np.asarray(data["body_pos_w"]))
  return values[frame, TRACKED_INDICES[tracked_index]]


@pytest.fixture
def harness(tmp_path: Path) -> tuple[MultiMotionCommand, _FakeEnv]:
  return make_command(make_library(tmp_path), make_slots((0, 1, 0, 1)))


# Reference queries.


def test_rows_read_their_own_clip(harness: tuple[MultiMotionCommand, _FakeEnv]) -> None:
  command, env = harness
  library = command.library
  rows = ((0, 0), (1, 4), (0, 1), (1, 0))
  command.time_steps = torch.tensor([frame for _, frame in rows], dtype=torch.long)

  assert command.motion_ids.tolist() == [0, 1, 0, 1]
  assert command.teacher_codes.tolist() == [0, 1, 0, 1]
  assert command.row_lengths.tolist() == [2, 5, 2, 5]
  assert command.lookahead_command.shape == (4, 0)
  assert bool((command.row_bin_counts > 0).all())

  joints = command.joint_pos
  for row, (motion_id, frame) in enumerate(rows):
    assert torch.equal(joints[row], source_joint_pos(library, motion_id, frame))
    assert torch.equal(
      command.body_pos_w[row, 0], source_body_pos(library, motion_id, frame, 0)
    )
    assert torch.equal(
      command.anchor_pos_w[row], source_body_pos(library, motion_id, frame, 1)
    )
    assert torch.equal(
      command.body_quat_w[row, 0],
      torch.from_numpy(
        np.asarray(
          np.load(library.clips[motion_id].motion_file, allow_pickle=False)[
            "body_quat_w"
          ]
        )
      )[frame, TRACKED_INDICES[0]],
    )
  # The two clips' sentinel bases differ, so a scalar clip length would show up
  # as the wrong clip's values here.
  assert joints[0, 0] < SENTINEL_BASE and joints[1, 0] > SENTINEL_BASE
  command.update_relative_body_poses()
  assert env.sim.forward_calls == 0
  assert command.body_pos_relative_w.shape == (4, len(TRACKED_BODIES), 3)


def test_unequal_clip_wrap_uses_each_row_length(
  harness: tuple[MultiMotionCommand, _FakeEnv],
) -> None:
  command, _ = harness
  stabilize(command)
  command.time_steps = torch.tensor([1, 3, 0, 2], dtype=torch.long)
  before = command.generation_ids.clone()
  command._update_command(None)

  assert command.time_steps.tolist() == [0, 4, 1, 3]
  assert command.generation_ids.tolist() == [1, 0, 0, 0]
  assert command.segment_ids.tolist() == [1, 0, 0, 0]
  events = command.consume_boundary_events(before)
  assert events.completed.tolist() == [True, False, False, False]
  assert events.interrupted.tolist() == [False, False, False, False]
  assert events.completed_generation.tolist() == [0, -1, -1, -1]
  assert events.reasons == (
    "reference_completed",
    "unavailable",
    "unavailable",
    "unavailable",
  )


def test_wrap_with_termination_takes_failure_precedence(
  harness: tuple[MultiMotionCommand, _FakeEnv],
) -> None:
  command, _ = harness
  stabilize(command)
  command.time_steps = torch.tensor([1, 0, 0, 0], dtype=torch.long)
  before = command.generation_ids.clone()
  command._update_command(None)
  events = command.consume_boundary_events(before)
  assert events.completed.tolist() == [True, False, False, False]

  resolved = events.with_step_outcome(torch.tensor([True, False, False, False]))
  assert resolved.completed.tolist() == [False, False, False, False]
  assert resolved.interrupted.tolist() == [True, False, False, False]
  assert resolved.completed_generation.tolist() == [-1, -1, -1, -1]
  assert resolved.interrupted_generation.tolist() == [0, -1, -1, -1]
  assert resolved.reasons[0] == "reference_completed+terminated"
  assert resolved.reasons[1:] == ("unavailable",) * 3


def test_consecutive_looking_frames_still_advance_generation(
  tmp_path: Path,
) -> None:
  command, _ = make_command(
    make_library(tmp_path, clip_frames=(1, 3)),
    make_slots((0, 0, 1)),
    num_envs=3,
  )
  stabilize(command)
  command.time_steps = torch.zeros(3, dtype=torch.long)
  before = command.generation_ids.clone()
  command._update_command(None)
  # Rows on the single-frame clip wrap every step while their frame stays 0.
  assert command.time_steps.tolist() == [0, 0, 1]
  assert command.generation_ids.tolist() == [1, 1, 0]
  events = command.consume_boundary_events(before)
  assert events.completed.tolist() == [True, True, False]

  command._update_command(None)
  assert command.time_steps.tolist() == [0, 0, 2]
  assert command.generation_ids.tolist() == [2, 2, 0]
  events = command.consume_boundary_events(torch.tensor([1, 1, 0], dtype=torch.long))
  assert events.completed_generation.tolist() == [1, 1, -1]


def test_timer_resample_then_wrap_tags_each_resample_once(
  harness: tuple[MultiMotionCommand, _FakeEnv], monkeypatch: pytest.MonkeyPatch
) -> None:
  command, _ = harness
  stabilize(command)
  command.consume_boundary_events()

  def sample_last_frame(env_ids: torch.Tensor) -> None:
    command.time_steps[env_ids] = command.row_lengths[env_ids] - 1

  # A uniform resample can land on the row's last frame; force that outcome so
  # the same compute also wraps the frame it just sampled.
  monkeypatch.setattr(command.cfg, "sampling_mode", "uniform")
  monkeypatch.setattr(command, "_uniform_sampling", sample_last_frame)

  before = command.generation_ids.clone()
  command.time_left[:] = -1.0
  command.compute(dt=0.01)

  assert command.generation_ids.tolist() == (before + 2).tolist()
  assert command.segment_ids.tolist() == (before + 2).tolist()
  events = command.consume_boundary_events(before)
  assert events.available.tolist() == [True] * 4
  assert events.interrupted.tolist() == [True] * 4
  assert events.completed.tolist() == [True] * 4
  assert events.interrupted_generation.tolist() == before.tolist()
  assert events.completed_generation.tolist() == (before + 1).tolist()
  assert events.post_generation.tolist() == (before + 2).tolist()
  assert events.reasons == ("timer_resampled+reference_completed",) * 4


def test_zero_step_reset_compute_is_row_scoped(
  harness: tuple[MultiMotionCommand, _FakeEnv],
) -> None:
  command, _ = harness
  stabilize(command)
  command.time_steps = torch.tensor([0, 3, 1, 0], dtype=torch.long)
  rows = torch.tensor([1])
  command.compute(dt=0.0, env_ids=rows)
  assert command.time_steps.tolist() == [0, 4, 1, 0]
  assert command.generation_ids.tolist() == [0, 0, 0, 0]

  command.compute(dt=0.0, env_ids=rows)
  # Clip 1 has 5 frames, so frame 4 wraps against the row's own length.
  assert command.time_steps.tolist() == [0, 0, 1, 0]
  assert command.generation_ids.tolist() == [0, 1, 0, 0]


def test_start_policy_initializes_rows_at_their_clip_start(
  harness: tuple[MultiMotionCommand, _FakeEnv],
) -> None:
  command, env = harness
  env.reset()
  assert command.time_steps.tolist() == [1, 1, 1, 1]
  for row, motion_id in enumerate(command.motion_ids.tolist()):
    # The reset wrote clip-local frame 0 of that row's own clip.
    assert torch.equal(
      env.robot.data.joint_pos[row], source_joint_pos(command.library, motion_id, 0)
    )
    assert torch.equal(
      command.joint_pos[row], source_joint_pos(command.library, motion_id, 1)
    )
  assert env.reset_calls[-1].tolist() == [0, 1, 2, 3]


def test_uniform_phase_stays_inside_each_rows_clip(
  harness: tuple[MultiMotionCommand, _FakeEnv],
) -> None:
  command, _ = harness
  assert command.cfg.sampling_mode == "start"
  torch.manual_seed(11)
  command.cfg.sampling_mode = "uniform"
  seen: dict[int, set[int]] = {0: set(), 1: set()}
  for _ in range(200):
    command._uniform_sampling(torch.arange(command.num_envs))
    assert bool((command.time_steps < command.row_lengths).all())
    for row in range(command.num_envs):
      seen[int(command.motion_ids[row].item())].add(int(command.time_steps[row]))
  assert seen[0] == {0, 1}
  assert seen[1] == {0, 1, 2, 3, 4}


def test_lookahead_clamps_inside_the_row_clip(tmp_path: Path) -> None:
  command, _ = make_command(
    make_library(tmp_path),
    make_slots((0, 1, 0, 1)),
    lookahead_s=0.02,
  )
  assert command.row_lookahead_steps.tolist() == [1, 1, 1, 1]
  frames = (1, 4, 0, 3)
  command.time_steps = torch.tensor(frames, dtype=torch.long)
  lookahead = command.lookahead_command
  assert lookahead.shape == (4, 2 * JOINT_DIM)
  for row, (motion_id, frame) in enumerate(
    zip(command.motion_ids.tolist(), frames, strict=True)
  ):
    clamped = min(frame + 1, int(command.row_lengths[row].item()) - 1)
    with np.load(
      command.library.clips[motion_id].motion_file, allow_pickle=False
    ) as data:
      joint_pos = torch.from_numpy(np.asarray(data["joint_pos"]))[clamped]
      joint_vel = torch.from_numpy(np.asarray(data["joint_vel"]))[clamped]
    assert torch.equal(lookahead[row], torch.cat([joint_pos, joint_vel]))
  # Clip 0's row at its last frame previews its own last frame, never clip 1.
  assert lookahead[0, 0] < SENTINEL_BASE


# Selection seam.


def test_subset_selection_resets_only_selected_rows(
  harness: tuple[MultiMotionCommand, _FakeEnv],
) -> None:
  command, env = harness
  stabilize(command)
  env.robot.data.joint_pos[:] = 7.0
  command.time_steps = torch.tensor([1, 2, 0, 3], dtype=torch.long)
  before_frames = command.time_steps.clone()
  before_generations = command.generation_ids.clone()

  command.select_motion_ids(torch.tensor([0]), torch.tensor([1]))

  assert command.motion_ids.tolist() == [1, 1, 0, 1]
  assert command.row_lengths.tolist() == [5, 5, 2, 5]
  assert command.teacher_codes.tolist() == [1, 1, 0, 1]
  assert command.generation_ids.tolist() == [
    before_generations[0].item() + 1,
    before_generations[1].item(),
    before_generations[2].item(),
    before_generations[3].item(),
  ]
  # Only the selected row moved: frames, reference state, and clip of the others
  # are untouched.
  assert command.time_steps[1:].tolist() == before_frames[1:].tolist()
  assert env.reset_calls[-1].tolist() == [0]
  assert env.robot.root_writes[-1] is not None
  assert env.robot.root_writes[-1].tolist() == [0]
  assert torch.equal(
    env.robot.data.joint_pos[0], source_joint_pos(command.library, 1, 0)
  )
  assert torch.equal(env.robot.data.joint_pos[1:], torch.full((3, JOINT_DIM), 7.0))
  assert 0 <= int(command.time_steps[0]) < 5


def test_multiple_rows_can_select_the_same_motion(
  harness: tuple[MultiMotionCommand, _FakeEnv],
) -> None:
  command, env = harness
  before = command.generation_ids.clone()
  command.select_motion_ids(torch.tensor([0, 2]), torch.tensor([1, 1]))
  assert command.motion_ids[[0, 2]].tolist() == [1, 1]
  assert command.generation_ids[[0, 2]].tolist() == (before[[0, 2]] + 1).tolist()
  assert torch.equal(command.generation_ids[[1, 3]], before[[1, 3]])
  assert env.reset_calls[-1].tolist() == [0, 2]


def test_selection_validation(harness: tuple[MultiMotionCommand, _FakeEnv]) -> None:
  command, _ = harness
  with pytest.raises(MultiMotionError, match="at least one row"):
    command.select_motion_ids(
      torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    )
  with pytest.raises(MultiMotionError, match=r"must be in \[0, 2\)"):
    command.select_motion_ids(torch.tensor([0]), torch.tensor([2]))
  with pytest.raises(MultiMotionError, match="must not repeat rows"):
    command.select_motion_ids(torch.tensor([0, 0]), torch.tensor([0, 1]))
  with pytest.raises(MultiMotionError, match="rows but 2 motion ids"):
    command.select_motion_ids(torch.tensor([0]), torch.tensor([0, 1]))
  with pytest.raises(MultiMotionError, match="integer tensor"):
    command.select_motion_ids(torch.tensor([0.0]), torch.tensor([0]))
  with pytest.raises(MultiMotionError, match="must be one-dimensional"):
    command.select_motion_ids(torch.zeros(1, 1, dtype=torch.long), torch.tensor([0]))
  with pytest.raises(MultiMotionError, match=r"must be in \[0, 4\)"):
    command.select_motion_ids(torch.tensor([9]), torch.tensor([0]))
  with pytest.raises(MultiMotionError, match="non-negative"):
    command.select_motion_ids(torch.tensor([-1]), torch.tensor([0]))


def test_reset_to_frame_is_clip_local(
  harness: tuple[MultiMotionCommand, _FakeEnv],
) -> None:
  command, _ = harness
  before = command.generation_ids.clone()
  command.reset_to_frame(torch.tensor([1]), 4)
  assert command.time_steps[1].item() == 4
  assert command.generation_ids.tolist() == [
    before[0].item(),
    before[1].item() + 1,
    before[2].item(),
    before[3].item(),
  ]
  events = command.consume_boundary_events()
  assert events.interrupted.tolist() == [False, True, False, False]
  assert events.reasons[1].startswith("reset_to_frame")

  with pytest.raises(MultiMotionError, match="outside the selected rows"):
    command.reset_to_frame(torch.tensor([0]), 4)
  with pytest.raises(MultiMotionError, match="outside the selected rows"):
    command.reset_to_frame(torch.tensor([1, 0]), 2)
  with pytest.raises(MultiMotionError, match="frame must be an integer"):
    command.reset_to_frame(torch.tensor([1]), cast(int, 2.0))
  with pytest.raises(MultiMotionError, match="at least one row"):
    command.reset_to_frame(torch.empty(0, dtype=torch.long), 0)


# Refusals and configuration contract.


def test_unimplemented_policies_and_standing_start_are_refused(
  tmp_path: Path,
) -> None:
  library = make_library(tmp_path)
  slots = make_slots((0, 1, 0, 1))
  for policy in ("adaptive", "weighted"):
    with pytest.raises(MultiMotionError, match="multi-motion phase sampling supports"):
      make_command(library, slots, phase_policy=cast(PhasePolicy, policy))
  with pytest.raises(MultiMotionError, match="standing-start initialization"):
    make_command(library, slots, standing_start_prob=0.3)
  with pytest.raises(MultiMotionError, match="compiled robot has 3"):
    make_command(library, slots, robot_bodies=ROBOT_BODIES[:3])
  with pytest.raises(MultiMotionError, match="slot allocation teachers"):
    make_command(
      library,
      make_slots((0, 1, 0, 1), teacher_ids=("tiny_000", "tiny_009")),
    )

  command, _ = make_command(library, slots)
  with pytest.raises(MultiMotionError, match="no single 'motion' loader"):
    _ = command.motion
  with pytest.raises(MultiMotionError, match="adaptive phase sampling is refused"):
    command._adaptive_sampling(torch.arange(4))
  with pytest.raises(MultiMotionError, match="weighted phase sampling is refused"):
    command._weighted_sampling(torch.arange(4))
  assert command.slot_allocation is slots
  assert command.library.body_selection is not None
  assert command.library.body_selection.indices == TRACKED_INDICES


def test_command_needs_a_library_and_slots(tmp_path: Path) -> None:
  env = cast("ManagerBasedRlEnv", _FakeEnv(2))
  without_library = MultiMotionCommandCfg(
    motion_file="",
    anchor_body_name="torso_link",
    body_names=TRACKED_BODIES,
    entity_name="robot",
    resampling_time_range=(1.0e9, 1.0e9),
    sampling_mode="start",
    library=cast(MotionLibrary, "nope"),
    slots=make_slots((0, 1)),
  )
  with pytest.raises(MultiMotionError, match="needs a MotionLibrary"):
    MultiMotionCommand(without_library, env)

  without_slots = MultiMotionCommandCfg(
    motion_file="",
    anchor_body_name="torso_link",
    body_names=TRACKED_BODIES,
    entity_name="robot",
    resampling_time_range=(1.0e9, 1.0e9),
    sampling_mode="start",
    library=make_library(tmp_path),
    slots=cast(MotionSlotAllocation, None),
  )
  with pytest.raises(MultiMotionError, match="needs a MotionSlotAllocation"):
    MultiMotionCommand(without_slots, env)


def test_body_queries_need_a_resolved_selection(
  harness: tuple[MultiMotionCommand, _FakeEnv],
) -> None:
  command, _ = harness
  # The command resolves the tracked subset against the compiled robot.
  unresolved = cast(MultiMotionCommandCfg, command.cfg).library
  assert unresolved.body_selection is None
  assert command.library.body_selection is not None
  with pytest.raises(MotionLibraryError, match="with_body_selection"):
    unresolved.body_pos_w(torch.tensor([0]), torch.tensor([0]))


# Slot allocation.


def test_stratified_allocation_counts_and_permutation() -> None:
  equal = stratified_slot_allocation([1.0, 1.0], 5, teacher_ids=("a", "b"))
  assert equal.counts == (3, 2)
  assert sorted(equal.row_motion_ids) == [0, 0, 0, 1, 1]
  assert equal.fractions == (0.6, 0.4)
  assert equal.num_envs == 5
  assert equal.as_dict()["counts"] == [3, 2]
  assert stratified_slot_allocation([2.0, 2.0], 4, teacher_ids=("a", "b")).counts == (
    2,
    2,
  )
  assert stratified_slot_allocation([3.0, 1.0], 4, teacher_ids=("a", "b")).counts == (
    3,
    1,
  )
  assert stratified_slot_allocation(
    [1.0, 1.0, 1.0], 6, teacher_ids=("a", "b", "c")
  ).counts == (2, 2, 2)
  assert stratified_slot_allocation(
    [1.0, 1.0, 1.0], 4, teacher_ids=("a", "b", "c")
  ).counts == (2, 1, 1)
  # A tiny weight still needs a row, taken from the largest allocation, and the
  # repair never leaves another motion unrepresented.
  assert stratified_slot_allocation([100.0, 1.0], 3, teacher_ids=("a", "b")).counts == (
    2,
    1,
  )
  assert stratified_slot_allocation(
    [0.5, 0.5, 2.0], 3, teacher_ids=("a", "b", "c")
  ).counts == (1, 1, 1)
  assert stratified_slot_allocation(
    [2.9, 0.05, 0.05], 3, teacher_ids=("a", "b", "c")
  ).counts == (1, 1, 1)
  assert stratified_slot_allocation(
    [5.0, 5.0, 0.1], 4, teacher_ids=("a", "b", "c")
  ).counts == (2, 1, 1)

  seeded = stratified_slot_allocation(
    [1.0, 1.0], 32, teacher_ids=("a", "b"), generator=torch.Generator().manual_seed(7)
  )
  again = stratified_slot_allocation(
    [1.0, 1.0], 32, teacher_ids=("a", "b"), generator=torch.Generator().manual_seed(7)
  )
  assert seeded.row_motion_ids == again.row_motion_ids
  assert Counter(seeded.row_motion_ids) == Counter({0: 16, 1: 16})


def test_stratified_allocation_is_representative_for_many_budgets() -> None:
  cases = (
    ((1.0, 1.0), 2),
    ((1.0, 1.0), 7),
    ((0.5, 0.5, 2.0), 3),
    ((0.5, 0.5, 2.0), 9),
    ((2.9, 0.05, 0.05), 3),
    ((5.0, 5.0, 0.1), 4),
    ((1.0, 2.0, 3.0), 12),
    ((7.0, 1.0, 1.0), 3),
    ((1.0, 1.0, 1.0, 1.0), 4),
    ((1.0, 1.0, 1.0, 1.0), 5),
    ((0.1, 10.0, 0.1), 5),
  )
  for weights, num_envs in cases:
    teacher_ids = tuple(f"m{index}" for index in range(len(weights)))
    allocation = stratified_slot_allocation(
      list(weights), num_envs, teacher_ids=teacher_ids
    )
    assert sum(allocation.counts) == num_envs
    assert all(count >= 1 for count in allocation.counts)
    assert sorted(allocation.row_motion_ids) == sorted(
      motion_id
      for motion_id, count in enumerate(allocation.counts)
      for _ in range(count)
    )


def test_stratified_allocation_rejects_impossible_requests() -> None:
  with pytest.raises(MultiMotionError, match="cannot represent"):
    stratified_slot_allocation([1.0, 1.0], 1, teacher_ids=("a", "b"))
  for bad in (0.0, -1.0, float("nan")):
    with pytest.raises(MultiMotionError, match="finite and positive"):
      stratified_slot_allocation([1.0, bad], 4, teacher_ids=("a", "b"))
  with pytest.raises(MultiMotionError, match="teacher ids must be unique"):
    stratified_slot_allocation([1.0, 1.0], 4, teacher_ids=("a", "a"))
  with pytest.raises(MultiMotionError, match="weights but 1 teacher ids"):
    stratified_slot_allocation([1.0, 1.0], 4, teacher_ids=("a",))
  with pytest.raises(MultiMotionError, match="at least one motion"):
    stratified_slot_allocation([], 4, teacher_ids=())
  with pytest.raises(MultiMotionError, match="num_envs must be an integer"):
    stratified_slot_allocation([1.0], True, teacher_ids=("a",))


def test_slot_allocation_is_validated() -> None:
  with pytest.raises(MultiMotionError, match="at least one motion"):
    MotionSlotAllocation(weights=(), counts=(), teacher_ids=(), row_motion_ids=())
  with pytest.raises(MultiMotionError, match="must align"):
    MotionSlotAllocation(
      weights=(1.0, 1.0), counts=(2,), teacher_ids=("a", "b"), row_motion_ids=(0, 1)
    )
  with pytest.raises(MultiMotionError, match="at least one row"):
    MotionSlotAllocation(
      weights=(1.0, 1.0), counts=(0, 2), teacher_ids=("a", "b"), row_motion_ids=(1, 1)
    )
  with pytest.raises(MultiMotionError, match="do not cover"):
    MotionSlotAllocation(
      weights=(1.0, 1.0), counts=(2, 2), teacher_ids=("a", "b"), row_motion_ids=(0, 1)
    )
  with pytest.raises(MultiMotionError, match="must be in"):
    MotionSlotAllocation(
      weights=(1.0, 1.0), counts=(1, 1), teacher_ids=("a", "b"), row_motion_ids=(0, 2)
    )
  with pytest.raises(MultiMotionError, match="finite and positive"):
    MotionSlotAllocation(
      weights=(1.0, 0.0), counts=(1, 1), teacher_ids=("a", "b"), row_motion_ids=(0, 1)
    )


# Planning and the opt-in factory (no simulator construction).


@pytest.fixture(scope="module")
def tiny_cohort(tmp_path_factory: pytest.TempPathFactory):
  root = tmp_path_factory.mktemp("multi-tiny")
  with (
    contextlib.redirect_stdout(io.StringIO()),
    contextlib.redirect_stderr(io.StringIO()),
  ):
    return resolve_cohort(load_manifest(build_tiny_cohort(root).manifest, root))


def test_plan_selects_in_manifest_order_with_stable_codes(tiny_cohort) -> None:
  plan = plan_multi_motion(tiny_cohort, ("tiny_001", "tiny_000"), 4)
  assert plan.teacher_ids == ("tiny_000", "tiny_001")
  assert [clip.teacher_id for clip in plan.library.clips] == ["tiny_000", "tiny_001"]
  assert [clip.motion_id for clip in plan.library.clips] == [0, 1]
  assert [clip.teacher_code for clip in plan.library.clips] == [0, 1]
  assert plan.slots.teacher_ids == ("tiny_000", "tiny_001")
  assert plan.slots.counts == (2, 2)
  assert plan.phase_policy == "uniform"
  assert plan.library.body_selection is None
  assert (
    plan.library.mapping_digest()
    == plan_multi_motion(
      tiny_cohort, ("tiny_000", "tiny_001"), 4
    ).library.mapping_digest()
  )

  subset = plan_multi_motion(tiny_cohort, ("tiny_001",), 2)
  assert subset.teacher_ids == ("tiny_001",)
  assert subset.library.clips[0].teacher_code == 1
  assert subset.library.mapping_digest() != plan.library.mapping_digest()


def test_plan_rejects_invalid_selections(tiny_cohort) -> None:
  with pytest.raises(MultiMotionError, match="at least one teacher id"):
    plan_multi_motion(tiny_cohort, (), 2)
  with pytest.raises(MultiMotionError, match="duplicate teacher ids"):
    plan_multi_motion(tiny_cohort, ("tiny_000", "tiny_000"), 2)
  with pytest.raises(MultiMotionError, match="cohort has no teacher"):
    plan_multi_motion(tiny_cohort, ("tiny_009",), 2)
  with pytest.raises(MultiMotionError, match="phase sampling supports"):
    plan_multi_motion(
      tiny_cohort, ("tiny_000",), 2, phase_policy=cast(PhasePolicy, "adaptive")
    )
  with pytest.raises(MultiMotionError, match="cannot represent"):
    plan_multi_motion(tiny_cohort, ("tiny_000", "tiny_001"), 1)


def test_manifest_reordering_changes_the_mapping_digest(tmp_path: Path) -> None:
  root = tmp_path / "cohort"
  with (
    contextlib.redirect_stdout(io.StringIO()),
    contextlib.redirect_stderr(io.StringIO()),
  ):
    tiny = build_tiny_cohort(root)
  entries = []
  for teacher in tiny.teachers:
    entries.append(
      "\n".join(
        (
          f"  - id: {teacher.id}",
          f"    checkpoint: {Path(teacher.checkpoint).relative_to(root)}",
          f"    motion: {Path(teacher.motion).relative_to(root)}",
          f"    env_config: {Path(teacher.env_config).relative_to(root)}",
          f"    agent_config: {Path(teacher.agent_config).relative_to(root)}",
          f"    onnx: {Path(teacher.onnx).relative_to(root)}",
          "    sampling_weight: 1.0",
        )
      )
    )
  header = (
    "version: 1\nname: reorder\nrobot: agibot_x2\n"
    "base_task: Mjlab-Tracking-Flat-AgiBot-X2\nteachers:\n"
  )
  original_path = root / "configs" / "original.yaml"
  reordered_path = root / "configs" / "reordered.yaml"
  original_path.write_text(header + "\n".join(entries) + "\n")
  reordered_path.write_text(header + "\n".join(reversed(entries)) + "\n")

  with (
    contextlib.redirect_stdout(io.StringIO()),
    contextlib.redirect_stderr(io.StringIO()),
  ):
    original = resolve_cohort(load_manifest(original_path, root))
    reordered = resolve_cohort(load_manifest(reordered_path, root))
  first = plan_multi_motion(original, ("tiny_000", "tiny_001"), 2)
  second = plan_multi_motion(reordered, ("tiny_000", "tiny_001"), 2)

  # Numeric ids are the manifest position, so a reordered manifest maps the same
  # request onto different teachers: the digest changes, so strict resume rejects
  # it instead of silently reinterpreting what id 0 means.
  assert first.teacher_ids == ("tiny_000", "tiny_001")
  assert second.teacher_ids == ("tiny_001", "tiny_000")
  assert first.library.mapping_digest() != second.library.mapping_digest()
  assert first.library.clips[0].source_hash == second.library.clips[1].source_hash
  assert second.library.clips[0].source_hash == first.library.clips[1].source_hash


def test_make_multi_motion_cfg_replaces_only_the_motion_command(tiny_cohort) -> None:
  plan = plan_multi_motion(
    tiny_cohort, ("tiny_000", "tiny_001"), 4, phase_policy="start"
  )
  base = MotionCommandCfg(
    motion_file="data/tennis/single_000.npz",
    anchor_body_name="torso_link",
    body_names=TRACKED_BODIES,
    entity_name="robot",
    resampling_time_range=(1.0e9, 1.0e9),
    joint_position_range=(-0.02, 0.02),
    sampling_mode="adaptive",
  )
  cfg = make_multi_motion_cfg(base, plan)
  assert cfg.motion_file == ""
  assert cfg.sampling_mode == "start"
  assert cfg.joint_position_range == (-0.02, 0.02)
  assert cfg.body_names == TRACKED_BODIES
  assert cfg.anchor_body_name == "torso_link"
  assert cfg.resampling_time_range == (1.0e9, 1.0e9)
  assert cfg.library is plan.library
  assert cfg.slots is plan.slots
  with pytest.raises(MultiMotionError, match="already multi-motion"):
    make_multi_motion_cfg(cfg, plan)


def test_factory_validates_before_constructing_an_environment(tiny_cohort) -> None:
  with pytest.raises(MultiMotionError, match="cannot represent"):
    build_multi_motion_environment(tiny_cohort, ("tiny_000", "tiny_001"), num_envs=1)
  with pytest.raises(MultiMotionError, match="cohort has no teacher"):
    build_multi_motion_environment(tiny_cohort, ("tiny_009",), num_envs=2)
  with pytest.raises(MultiMotionError, match="duplicate teacher ids"):
    build_multi_motion_environment(tiny_cohort, ("tiny_000", "tiny_000"), num_envs=2)
  with pytest.raises(ValueError, match="phase_policy must be"):
    build_multi_motion_environment(
      tiny_cohort, ("tiny_000",), num_envs=2, phase_policy="adaptive"
    )
  with pytest.raises(ValueError, match="num_envs must be positive"):
    build_multi_motion_environment(tiny_cohort, ("tiny_000",), num_envs=0)
  with pytest.raises(ValueError, match="seed"):
    build_multi_motion_environment(tiny_cohort, ("tiny_000",), num_envs=2, seed=True)
  # The saved standing-start experiment is refused rather than silently disabled.
  with pytest.raises(ValueError, match="standing_start_prob"):
    build_multi_motion_environment(
      tiny_cohort,
      ("tiny_000", "tiny_001"),
      num_envs=2,
      task_id=STANDING_START_TASK,
    )
