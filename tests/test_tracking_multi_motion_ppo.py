"""Critical tests for the file-configured multi-motion PPO command.

The fake environment mirrors the real reset and compute contract: a reset
resamples the selected rows, the environment forwards the simulator, and the
command update that follows a reset runs with ``dt=0``.  No MuJoCo model is
compiled here, so live-environment evidence stays with the monitored smoke run.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pytest
import torch
from tracking_distillation_fixtures import write_reference_clip

from mjlab.tasks.tracking.distillation.multi_motion import (
  MultiMotionCommandCfg,
  MultiMotionError,
)
from mjlab.tasks.tracking.mdp.multi_motion_ppo import (
  MultiMotionPpoCommand,
  MultiMotionPpoCommandCfg,
  MultiMotionPpoError,
  clip_complete,
  repository_root,
  resolve_motion_files,
)
from mjlab.utils.lab_api.string import resolve_matching_names

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

JOINT_DIM = 31
SOURCE_BODIES = 4
ROBOT_BODIES = ("pelvis", "torso_link", "left_knee_link", "right_knee_link")
TRACKED_BODIES = ("pelvis", "torso_link")
ANCHOR_BODY = "torso_link"
SENTINEL_BASE = 1000.0
PREVIEW_STEPS = 10  # 0.2 s of preview at the 50 Hz control rate
ENV_RATE = 50.0


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
  def __init__(self, num_envs: int) -> None:
    self.body_names = ROBOT_BODIES
    self.data = _FakeRobotData(num_envs, len(ROBOT_BODIES), JOINT_DIM)
    self.root_writes: list[torch.Tensor] = []

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
    assert env_ids is not None
    self.root_writes.append(env_ids.clone())
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


class _FakeCommandManager:
  def __init__(self, command: MultiMotionPpoCommand) -> None:
    self._command = command

  def get_term(self, name: str) -> MultiMotionPpoCommand:
    assert name == "motion"
    return self._command


class _FakeEnv:
  """Minimal environment that mirrors the real reset/compute contract."""

  def __init__(self, num_envs: int) -> None:
    self.num_envs = num_envs
    self.device = "cpu"
    self.step_dt = 1.0 / ENV_RATE
    self.robot = _FakeRobot(num_envs)
    self.scene = _FakeScene(self.robot, num_envs)
    self.sim = _FakeSim()
    self.command: MultiMotionPpoCommand | None = None
    self.command_manager: _FakeCommandManager | None = None

  def attach(self, command: MultiMotionPpoCommand) -> None:
    self.command = command
    self.command_manager = _FakeCommandManager(command)

  def reset(
    self, *, seed: int | None = None, env_ids: torch.Tensor | None = None
  ) -> None:
    del seed
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, dtype=torch.int64)
    assert self.command is not None
    self.command.reset(env_ids)
    self.sim.forward()
    self.command.compute(dt=0.0, env_ids=env_ids)


# Harness.


def write_clips(
  root: Path,
  frame_counts: tuple[int, ...],
  *,
  fps: float = ENV_RATE,
  joint_dim: int = JOINT_DIM,
) -> list[Path]:
  return [
    write_reference_clip(
      root / f"clip_{index}.npz",
      frames=count,
      joint_dim=joint_dim,
      bodies=SOURCE_BODIES,
      fps=fps,
      joint_base=SENTINEL_BASE * index,
      body_base=SENTINEL_BASE * index,
    )
    for index, count in enumerate(frame_counts)
  ]


def write_manifest(
  root: Path, entries: Sequence[str], name: str = "motions.yaml"
) -> Path:
  path = root / name
  path.write_text(
    "motions:\n" + "".join(f"  - {entry}\n" for entry in entries), encoding="utf-8"
  )
  return path


def make_cfg(
  root: Path,
  *,
  files: list[Path],
  num_envs: int = 4,
  sampling_mode: Literal["uniform", "start"] = "uniform",
  lookahead_s: float = 0.2,
  motion_file: str = "",
) -> tuple[MultiMotionPpoCommandCfg, _FakeEnv]:
  manifest = (
    "" if motion_file else str(write_manifest(root, [str(clip) for clip in files]))
  )
  cfg = MultiMotionPpoCommandCfg(
    motion_manifest=manifest,
    motion_file=motion_file,
    anchor_body_name=ANCHOR_BODY,
    body_names=TRACKED_BODIES,
    entity_name="robot",
    resampling_time_range=(1.0e9, 1.0e9),
    sampling_mode=sampling_mode,
    lookahead_s=lookahead_s,
    # Deterministic reset teleports: the written reference state is exact.
    joint_position_range=(0.0, 0.0),
  )
  return cfg, _FakeEnv(num_envs)


def make_command(
  root: Path,
  *,
  frame_counts: tuple[int, ...],
  num_envs: int = 4,
  sampling_mode: Literal["uniform", "start"] = "uniform",
) -> tuple[MultiMotionPpoCommand, _FakeEnv]:
  cfg, env = make_cfg(
    root,
    files=write_clips(root, frame_counts),
    num_envs=num_envs,
    sampling_mode=sampling_mode,
  )
  command = cfg.build(cast("ManagerBasedRlEnv", env))
  env.attach(command)
  return command, env


def endpoint_mask(env: _FakeEnv) -> list[bool]:
  """Per-row clip-endpoint mask, as the registered termination term sees it."""
  assert env.command_manager is not None
  return clip_complete(cast("ManagerBasedRlEnv", env)).tolist()


def source_clip(path: Path, name: str, frame: int) -> torch.Tensor:
  with np.load(path, allow_pickle=False) as data:
    values = torch.from_numpy(np.asarray(data[name]))
  return values[frame]


# Tests.


def test_unequal_clip_isolation_and_exact_preview_clamp(tmp_path: Path) -> None:
  frame_counts = (12, 20)
  clips = write_clips(tmp_path, frame_counts)
  cfg, env = make_cfg(tmp_path, files=clips)
  command = cfg.build(cast("ManagerBasedRlEnv", env))
  env.attach(command)

  # The registered config stays plain data; the loaded library and slots live
  # only in the runtime config that build() resolved once.
  assert isinstance(command.cfg, MultiMotionCommandCfg)
  assert not isinstance(command.cfg, MultiMotionPpoCommandCfg)
  assert not hasattr(cfg, "library")

  # Equal pinned rows within rounding, and every selected clip represented.
  counts = command.slot_allocation.counts
  assert sum(counts) == env.num_envs
  assert max(counts) - min(counts) <= 1
  assert min(counts) >= 1

  # Rows read their own clip: each clip's frame zero carries its own sentinel.
  command.time_steps.zero_()
  joints = command.joint_pos
  for row in range(env.num_envs):
    motion_id = int(command.motion_ids[row])
    assert torch.equal(joints[row], source_clip(clips[motion_id], "joint_pos", 0))
    assert float(joints[row, 0]) == SENTINEL_BASE * motion_id

  # Preview is one 62-wide future frame at exactly lookahead_s * env rate.
  preview = command.lookahead_command
  assert preview.shape == (env.num_envs, 2 * JOINT_DIM)
  for row in range(env.num_envs):
    motion_id = int(command.motion_ids[row])
    assert torch.equal(
      preview[row, :JOINT_DIM],
      source_clip(clips[motion_id], "joint_pos", PREVIEW_STEPS),
    )
    assert torch.equal(
      preview[row, JOINT_DIM:],
      source_clip(clips[motion_id], "joint_vel", PREVIEW_STEPS),
    )

  # The clamp is per clip and per row: an endpoint row previews its own last
  # frame, never the following clip's frame zero in flat storage.
  command.time_steps = command.row_lengths - 1
  clamped = command.lookahead_command
  for row in range(env.num_envs):
    motion_id = int(command.motion_ids[row])
    last = int(command.row_lengths[row]) - 1
    assert torch.equal(
      clamped[row, :JOINT_DIM], source_clip(clips[motion_id], "joint_pos", last)
    )
    assert torch.equal(
      clamped[row, JOINT_DIM:], source_clip(clips[motion_id], "joint_vel", last)
    )
    assert float(clamped[row, 0]) < SENTINEL_BASE * (motion_id + 1)


def test_endpoint_holds_without_wrap_or_timer_teleport(tmp_path: Path) -> None:
  frame_counts = (3, 4)
  command, env = make_command(
    tmp_path, frame_counts=frame_counts, sampling_mode="start"
  )
  env.reset()
  command.time_steps = command.row_lengths - 2
  env.robot.root_writes.clear()
  forward_calls = env.sim.forward_calls
  # Terminations are computed before command frames advance, so the endpoint
  # term describes the action of the current step.
  assert endpoint_mask(env) == [False] * env.num_envs

  # A timer expiry refills the timer without resampling a row: the reference
  # advances one frame instead of teleporting to the sampled start frame.
  command.time_left[:] = -1.0
  command.compute(dt=env.step_dt)
  assert bool((command.time_left > 0.0).all())
  assert command.time_steps.tolist() == (command.row_lengths - 1).tolist()
  # The advanced row is a reference advance only: no teleport and no forward.
  assert env.robot.root_writes == []
  assert env.sim.forward_calls == forward_calls
  assert endpoint_mask(env) == [True] * env.num_envs
  endpoint_joints = command.joint_pos.clone()

  # A following step holds the final frame instead of wrapping to frame zero.
  command.compute(dt=env.step_dt)
  assert command.time_steps.tolist() == (command.row_lengths - 1).tolist()
  assert torch.equal(command.joint_pos, endpoint_joints)
  assert env.robot.root_writes == []
  assert endpoint_mask(env) == [True] * env.num_envs


def test_subset_reset_keeps_frame_zero_and_other_rows(tmp_path: Path) -> None:
  command, env = make_command(tmp_path, frame_counts=(3, 4), sampling_mode="start")
  env.reset()
  initial = command.row_lengths - 2
  command.time_steps = initial.clone()
  env.robot.root_writes.clear()

  reset_rows = torch.tensor([1, 3], dtype=torch.int64)
  env.reset(env_ids=reset_rows)

  # Only the reset rows were teleported, and the frame their reset sampled
  # survives the reset's own dt=0 command update instead of being skipped.
  assert len(env.robot.root_writes) == 1
  assert env.robot.root_writes[0].tolist() == reset_rows.tolist()
  expected = initial.clone()
  expected[reset_rows] = 0
  assert command.time_steps.tolist() == expected.tolist()

  # The next step advances the reset rows from the sampled frame.
  command.compute(dt=env.step_dt)
  reset_list = reset_rows.tolist()
  for row in range(env.num_envs):
    expected[row] = 1 if row in reset_list else initial[row] + 1
  assert command.time_steps.tolist() == expected.tolist()

  # An explicit frame teleport is the other entry point: the caller takes the
  # next action at that frame before any command update, so the update that
  # follows advances one frame instead of repeating the teleported frame.
  command.reset_to_frame(reset_rows, 1)
  assert command.time_steps[reset_rows].tolist() == [1, 1]
  command.compute(dt=env.step_dt)
  assert command.time_steps[reset_rows].tolist() == [2, 2]


def test_reference_input_validation_and_serializable_config(tmp_path: Path) -> None:
  clips = write_clips(tmp_path, (12, 20))
  manifest = write_manifest(tmp_path, [str(clip) for clip in clips])
  base: dict = {
    "anchor_body_name": ANCHOR_BODY,
    "body_names": TRACKED_BODIES,
    "entity_name": "robot",
    "resampling_time_range": (1.0e9, 1.0e9),
    "sampling_mode": "uniform",
    "lookahead_s": 0.2,
    "joint_position_range": (0.0, 0.0),
  }
  cfg = MultiMotionPpoCommandCfg(motion_manifest=str(manifest), **base)

  # The configuration is plain data, and the resolved order is the file order.
  assert asdict(cfg)["motion_manifest"] == str(manifest)
  assert resolve_motion_files(cfg) == tuple(clip.resolve() for clip in clips)

  # A singleton clip is the playback alternative; the two sources are exclusive.
  singleton = MultiMotionPpoCommandCfg(motion_file=str(clips[0]), **base)
  assert resolve_motion_files(singleton) == (clips[0].resolve(),)
  with pytest.raises(MultiMotionPpoError, match="one reference source"):
    resolve_motion_files(
      MultiMotionPpoCommandCfg(
        motion_manifest=str(manifest), motion_file=str(clips[0]), **base
      )
    )
  with pytest.raises(MultiMotionPpoError, match="needs a nonempty motion_manifest"):
    resolve_motion_files(MultiMotionPpoCommandCfg(**base))

  # Relative entries resolve from the isolated repository root, not the cwd.
  expected = repository_root() / "data" / "tennis" / "missing_clip.npz"
  relative_manifest = write_manifest(
    tmp_path, ["data/tennis/missing_clip.npz"], name="relative.yaml"
  )
  with pytest.raises(MultiMotionPpoError) as relative_error:
    resolve_motion_files(
      MultiMotionPpoCommandCfg(motion_manifest=str(relative_manifest), **base)
    )
  assert str(expected) in str(relative_error.value)

  # Malformed, repeated, and unknown manifest content is refused.
  empty = write_manifest(tmp_path, [], name="empty.yaml")
  with pytest.raises(MultiMotionPpoError, match="nonempty 'motions' list"):
    resolve_motion_files(MultiMotionPpoCommandCfg(motion_manifest=str(empty), **base))
  (tmp_path / "typo.yaml").write_text("motion:\n  - clip.npz\n", encoding="utf-8")
  with pytest.raises(MultiMotionPpoError, match="unknown keys"):
    resolve_motion_files(
      MultiMotionPpoCommandCfg(motion_manifest=str(tmp_path / "typo.yaml"), **base)
    )
  repeated = write_manifest(
    tmp_path, [str(clips[0]), str(clips[0])], name="repeated.yaml"
  )
  with pytest.raises(MultiMotionPpoError, match="repeats a file"):
    resolve_motion_files(
      MultiMotionPpoCommandCfg(motion_manifest=str(repeated), **base)
    )

  # A wrong control rate is refused before any command state exists.
  slow = write_reference_clip(
    tmp_path / "slow" / "slow.npz",
    frames=12,
    joint_dim=JOINT_DIM,
    bodies=SOURCE_BODIES,
    fps=30.0,
  )
  with pytest.raises(MultiMotionPpoError, match="control rate"):
    MultiMotionPpoCommandCfg(motion_file=str(slow), **base).build(
      cast("ManagerBasedRlEnv", _FakeEnv(4))
    )

  # Fewer joints than the compiled robot has, and fewer rows than clips, are
  # both refused rather than silently reinterpreted.
  narrow = write_reference_clip(
    tmp_path / "narrow.npz",
    frames=12,
    joint_dim=JOINT_DIM - 1,
    bodies=SOURCE_BODIES,
    fps=ENV_RATE,
  )
  with pytest.raises(MultiMotionPpoError, match="joints but the compiled robot"):
    MultiMotionPpoCommandCfg(motion_file=str(narrow), **base).build(
      cast("ManagerBasedRlEnv", _FakeEnv(4))
    )
  with pytest.raises(MultiMotionError, match="cannot represent"):
    cfg.build(cast("ManagerBasedRlEnv", _FakeEnv(1)))

  # A singleton clip builds, with every row pinned to that clip.
  singleton_cfg, singleton_env = make_cfg(
    tmp_path, files=clips[:1], motion_file=str(clips[0])
  )
  singleton_command = singleton_cfg.build(cast("ManagerBasedRlEnv", singleton_env))
  assert singleton_command.library.num_motions == 1
  assert singleton_command.motion_ids.tolist() == [0] * singleton_env.num_envs
