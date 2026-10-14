"""CPU-only tests for the recorded-state player.

The player maps a recorded row onto the standalone X2 MJCF and computes the
kinematics.  These tests build that model and a synthetic recording whose body
positions are computed by MuJoCo itself, so a reconstruction mismatch means the
mapping broke.  No checkpoint, VAE, dataset or GPU is needed.
"""

from __future__ import annotations

import importlib.util
import shutil
import tempfile
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

SCRIPT = (
  Path(__file__).resolve().parents[1]
  / "scripts"
  / "experiments"
  / "diffusion_d3_closed_loop"
  / "play_recording.py"
)


def _module() -> Any:
  spec = importlib.util.spec_from_file_location("play_recording_under_test", SCRIPT)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


play = _module()


@pytest.fixture(scope="module")
def model() -> Any:
  work_dir = Path(tempfile.mkdtemp(prefix="x2-play-test-"))
  try:
    yield play._build_model(play._DEFAULT_MODEL, work_dir)
  finally:
    shutil.rmtree(work_dir, ignore_errors=True)


def _joint_names(model: Any) -> list[str]:
  names = []
  for joint_id in range(model.njnt):
    if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE:
      names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
  return names


def _body_names(model: Any) -> list[str]:
  return [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    for body_id in range(1, model.nbody)
  ]


def _write_recording(
  path: Path,
  model: Any,
  *,
  rows: int = 2,
  ticks: int = 5,
  fall_row: int | None = 1,
  fall_tick: int = 3,
  joint_names: list[str] | None = None,
  body_names: list[str] | None = None,
  with_bodies: bool = True,
  joint_width_override: int | None = None,
) -> Path:
  """Build a recording whose body positions come from MuJoCo kinematics."""
  names = joint_names if joint_names is not None else _joint_names(model)
  bodies = body_names if body_names is not None else _body_names(model)
  joint_addresses = np.array(
    [
      int(model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
      ])
      for name in names
    ],
    dtype=np.int64,
  )
  body_indexes = np.array(
    [
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
      for name in bodies
    ],
    dtype=np.int64,
  )
  data = mujoco.MjData(model)
  root_position = np.zeros((ticks, rows, 3), dtype=np.float32)
  root_quaternion = np.zeros((ticks, rows, 4), dtype=np.float32)
  joint_position = np.zeros((ticks, rows, len(names)), dtype=np.float32)
  body_positions = np.zeros((ticks, rows, len(bodies) * 3), dtype=np.float32)
  for row in range(rows):
    for tick in range(ticks):
      data.qpos[:] = model.qpos0
      # A pose that changes with the tick, so a stale reconstruction fails.
      data.qpos[:3] = [0.5 * row, -0.25 * row, 0.6 + 0.01 * tick]
      data.qpos[joint_addresses] = 0.05 * tick * np.linspace(-1.0, 1.0, len(names))
      mujoco.mj_kinematics(model, data)
      root_position[tick, row] = data.qpos[:3]
      root_quaternion[tick, row] = data.qpos[3:7]
      joint_position[tick, row] = data.qpos[joint_addresses]
      body_positions[tick, row] = data.xpos[body_indexes].reshape(-1)
  fall_first_tick = np.full(rows, -1, dtype=np.int64)
  if fall_row is not None and fall_row < rows:
    fall_first_tick[fall_row] = fall_tick
  stop_code = np.where(fall_first_tick >= 0, 1, 0).astype(np.int64)
  payload: dict[str, Any] = {
    "member": np.array("synthetic"),
    "rows": np.int64(rows),
    "recorded_rows": np.int64(rows),
    "tick": np.arange(ticks, dtype=np.int64),
    "phase": np.zeros(ticks, dtype=np.int64),
    "active": np.ones((ticks, rows), dtype=bool),
    "joint_names": np.array(names),
    "body_names": np.array(bodies),
    "fall_first_tick": fall_first_tick,
    "stop_labels": np.array(["step_budget", "physical_fall"]),
    "stop_code": stop_code,
    "period_seconds": np.float64(0.02),
    "joint_position": joint_position,
    "root_position": root_position,
    "root_quaternion_wxyz": root_quaternion,
    "tilt_degrees": np.zeros((ticks, rows), dtype=np.float32),
  }
  if joint_width_override is not None:
    payload["joint_position"] = np.zeros(
      (ticks, rows, joint_width_override), dtype=np.float32
    )
  if with_bodies:
    payload["body_positions"] = body_positions
  path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(path, **payload)
  return path


def test_reconstruction_matches_the_model(tmp_path: Path, model: Any) -> None:
  path = _write_recording(tmp_path / "rec.npz", model)
  recording = play.Recording(path)
  assert play._check(recording, model, recording.chosen_row(None), 1.0e-5) == 0


def test_recording_without_body_channels_cannot_be_checked(
  tmp_path: Path, model: Any
) -> None:
  path = _write_recording(tmp_path / "rec.npz", model, with_bodies=False)
  recording = play.Recording(path)
  with pytest.raises(play.PlayRecordingError, match="no body positions"):
    play._check(recording, model, 0, 1.0e-5)


def test_declared_joint_count_must_match_the_arrays(
  tmp_path: Path, model: Any
) -> None:
  path = _write_recording(tmp_path / "rec.npz", model, joint_width_override=30)
  with pytest.raises(play.PlayRecordingError, match="declares 31 joints"):
    play.Recording(path)


def test_missing_joint_name_is_refused(tmp_path: Path, model: Any) -> None:
  names = _joint_names(model)
  names[0] = "not_a_joint"
  path = _write_recording(tmp_path / "rec.npz", model, joint_names=names)
  recording = play.Recording(path)
  with pytest.raises(play.PlayRecordingError, match="no joint named"):
    play.RecordingPlayback(recording, model, 0, recenter=True)


def test_missing_body_name_is_refused(tmp_path: Path, model: Any) -> None:
  bodies = _body_names(model)
  bodies[0] = "not_a_body"
  path = _write_recording(tmp_path / "rec.npz", model, body_names=bodies)
  recording = play.Recording(path)
  with pytest.raises(play.PlayRecordingError, match="no body named"):
    play.RecordingPlayback(recording, model, 0, recenter=True)


def test_default_row_is_the_first_that_falls(tmp_path: Path, model: Any) -> None:
  path = _write_recording(tmp_path / "rec.npz", model, rows=3, fall_row=2)
  recording = play.Recording(path)
  assert recording.chosen_row(None) == 2
  assert recording.chosen_row(0) == 0
  with pytest.raises(play.PlayRecordingError, match="outside the 3 recorded rows"):
    recording.chosen_row(3)


def test_a_row_without_a_fall_plays_to_the_end(tmp_path: Path, model: Any) -> None:
  path = _write_recording(tmp_path / "rec.npz", model, ticks=20, fall_row=1, fall_tick=8)
  recording = play.Recording(path)
  assert recording.play_end(1, 25) == 19
  assert recording.play_end(0, 25) == 19


def test_a_fallen_row_stops_shortly_after_the_fall(
  tmp_path: Path, model: Any
) -> None:
  path = _write_recording(tmp_path / "rec.npz", model, ticks=200, fall_row=0, fall_tick=50)
  recording = play.Recording(path)
  assert recording.play_end(0, 25) == 75
  assert recording.play_end(0, 0) == 50


def test_frame_ticks_include_the_fall(tmp_path: Path, model: Any) -> None:
  path = _write_recording(tmp_path / "rec.npz", model, ticks=200, fall_row=0, fall_tick=50)
  recording = play.Recording(path)
  ticks = play._frame_ticks(recording, 0, 75)
  assert 50 in ticks and 0 in ticks and 75 in ticks
  assert ticks == sorted(ticks)
  assert len(set(ticks)) == len(ticks)


def test_row_table_lists_falling_rows_first(tmp_path: Path, model: Any) -> None:
  path = _write_recording(tmp_path / "rec.npz", model, rows=4, fall_row=3, fall_tick=6)
  recording = play.Recording(path)
  table = recording.row_table(limit=4)
  assert [row for row, _ in table] == [3, 0, 1, 2]
  assert "falls at tick 6" in table[0][1]
  assert any("no fall" in label for _, label in table)


def test_row_table_orders_several_falls_by_their_tick(
  tmp_path: Path, model: Any
) -> None:
  path = _write_recording(tmp_path / "rec.npz", model, rows=3, fall_row=None)
  with np.load(path, allow_pickle=False) as payload:
    stored = {name: payload[name] for name in payload.files}
  stored["fall_first_tick"] = np.array([30, 12, -1], dtype=np.int64)
  later = tmp_path / "ordered.npz"
  np.savez(later, **stored)
  recording = play.Recording(later)
  assert [row for row, _ in recording.row_table(limit=3)] == [1, 0, 2]  # noqa: E501


def test_playback_recenters_the_start_and_keeps_the_shape(
  tmp_path: Path, model: Any
) -> None:
  path = _write_recording(tmp_path / "rec.npz", model, rows=1, ticks=4)
  recording = play.Recording(path)
  playback = play.RecordingPlayback(recording, model, 0, recenter=True)
  assert np.allclose(playback.data.qpos[:3], [0.0, 0.0, 0.6], atol=1.0e-6)
  playback.seek(2)
  assert playback.data.qpos[2] == pytest.approx(0.62, abs=1.0e-6)
  playback.seek(99)
  assert playback.data.qpos[2] == pytest.approx(0.63, abs=1.0e-6)
  assert playback.lowest_body_z() > -1.0


def test_playback_keeps_the_world_frame_when_asked(tmp_path: Path, model: Any) -> None:
  path = _write_recording(tmp_path / "rec.npz", model, rows=2, ticks=3)
  recording = play.Recording(path)
  playback = play.RecordingPlayback(recording, model, 1, recenter=False)
  assert playback.data.qpos[0] == pytest.approx(0.5, abs=1.0e-6)
  assert playback.data.qpos[1] == pytest.approx(-0.25, abs=1.0e-6)


def test_named_rows_select_first_last_and_survivor(
  tmp_path: Path, model: Any
) -> None:
  path = _write_recording(tmp_path / "rec.npz", model, rows=3, fall_row=None)
  with np.load(path, allow_pickle=False) as payload:
    stored = {name: payload[name] for name in payload.files}
  stored["fall_first_tick"] = np.array([4, -1, 6], dtype=np.int64)
  named = tmp_path / "named.npz"
  np.savez(named, **stored)
  recording = play.Recording(named)
  assert recording.chosen_row("first-fall") == 0
  assert recording.chosen_row("last-fall") == 2
  assert recording.chosen_row("survivor") == 1
  assert recording.chosen_row("FIRST-FALL") == 0
  with pytest.raises(play.PlayRecordingError, match="not a row index"):
    recording.chosen_row("latest")


def test_named_fall_rows_need_a_fall(tmp_path: Path, model: Any) -> None:
  path = _write_recording(tmp_path / "rec.npz", model, rows=2, fall_row=None)
  recording = play.Recording(path)
  with pytest.raises(play.PlayRecordingError, match="no row in this recording falls"):
    recording.chosen_row("first-fall")
  with pytest.raises(play.PlayRecordingError, match="no row in this recording falls"):
    recording.chosen_row("last-fall")
  assert recording.chosen_row("survivor") == 0


def test_caption_reports_the_fall(tmp_path: Path, model: Any) -> None:
  path = _write_recording(tmp_path / "rec.npz", model, ticks=20, fall_row=0, fall_tick=8)
  recording = play.Recording(path)
  playback = play.RecordingPlayback(recording, model, 0, recenter=True)
  before = play._caption(recording, playback, 5)
  after = play._caption(recording, playback, 12)
  assert "synthetic row 0" in before
  assert "t=0.10s" in before
  assert "fall@8" in before and "past" not in before
  assert "fall@8 past" in after


def test_gif_writer_produces_an_animation(tmp_path: Path, model: Any) -> None:
  try:
    import mujoco

    mujoco.Renderer(model, height=48, width=64)
  except Exception as exc:  # pragma: no cover - no headless GL in this environment
    pytest.skip(f"offscreen rendering unavailable: {exc}")
  from PIL import Image

  path = _write_recording(tmp_path / "rec.npz", model, rows=1, ticks=5)
  recording = play.Recording(path)
  playback = play.RecordingPlayback(recording, model, 0, recenter=True)
  out = tmp_path / "out.gif"
  assert (
    play._save_gif(
      recording,
      playback,
      out,
      stride=1,
      height=48,
      width=64,
      margin_ticks=0,
    )
    == 0
  )
  with Image.open(out) as image:
    assert image.n_frames == 5
    assert image.info.get("duration") == 20
  with pytest.raises(play.PlayRecordingError, match="refusing to overwrite"):
    play._save_gif(
      recording,
      playback,
      out,
      stride=1,
      height=48,
      width=64,
      margin_ticks=0,
    )
