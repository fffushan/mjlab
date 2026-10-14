#!/usr/bin/env python3
"""Play a recorded robot state file in a browser.

``evaluate_parallel.py --record-states DIR`` writes one ``<member>.npz`` per
motion.  This tool opens one of those files and plays one row of it, so the
recorded motion, including a fall, can be seen.

The player is independent of the training artifacts.  It loads the standalone
X2 MJCF, writes the recorded root pose and joint positions into the model, and
computes the kinematics.  No physics runs and no policy is loaded, so the
picture shows the recorded motion exactly, at any speed.

Usage:
  play_recording.py RECORDING.npz [--row N] [--rate 0.25] [--port 8080]

  play_recording.py RECORDING.npz --list-rows      # candidate rows, then exit
  play_recording.py RECORDING.npz --check          # rebuild and compare, no GUI
  play_recording.py RECORDING.npz --selftest 15    # render headlessly for N s

The default row is the row that falls first.  Use ``--row`` to choose another.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MODEL = (
  _REPO_ROOT / "src/mjlab/asset_zoo/robots/agibot_x2/xmls/x2_ultra.xml"
)
_FLOOR_RGBA = "0.55 0.55 0.55 1"
_VIEWER_STEP_SECONDS = 0.01


CameraAzimuth = 135.0
CameraElevation = 18.0


class PlayRecordingError(RuntimeError):
  """A recording or model problem that stops playback."""


def _dependency(name: str) -> Any:
  try:
    return __import__(name)
  except ImportError as exc:  # pragma: no cover - environment problem
    raise PlayRecordingError(f"{name} is required for playback: {exc}") from exc


def _build_model(model_path: Path, work_dir: Path) -> Any:
  """Compile the robot XML together with a floor plane.

  MuJoCo resolves the mesh directory of an included file against that file's
  own directory, so the robot XML is copied next to a link to its assets and a
  small wrapper adds the floor.  The repository is left unchanged.
  """
  mujoco = _dependency("mujoco")
  if not model_path.is_file():
    raise PlayRecordingError(f"robot model not found: {model_path}")
  assets = model_path.parent / "assets"
  if not assets.is_dir():
    raise PlayRecordingError(f"robot mesh directory not found: {assets}")
  shutil.copyfile(model_path, work_dir / model_path.name)
  link = work_dir / "assets"
  if not link.exists():
    os.symlink(assets, link)
  wrapper = work_dir / "with_floor.xml"
  wrapper.write_text(
    f'<mujoco model="x2_floor">\n'
    f'  <include file="{model_path.name}"/>\n'
    f'  <worldbody>\n'
    f'    <geom name="floor" type="plane" size="0 0 0.05" rgba="{_FLOOR_RGBA}"'
    f' condim="3" friction="1 0.005 0.0001"/>\n'
    f'  </worldbody>\n'
    f"</mujoco>\n"
  )
  try:
    return mujoco.MjModel.from_xml_path(str(wrapper))
  except Exception as exc:
    raise PlayRecordingError(f"cannot compile the robot model: {exc}") from exc


class Recording:
  """One recorded state file, read once into memory."""

  def __init__(self, path: Path) -> None:
    if not path.is_file():
      raise PlayRecordingError(f"recording not found: {path}")
    with np.load(path, allow_pickle=False) as payload:
      self.path = path
      self.member = str(payload["member"])
      self.rows = int(payload["rows"])
      self.recorded_rows = int(payload["recorded_rows"])
      self.ticks = np.array(payload["tick"])
      self.phase = np.array(payload["phase"])
      self.active = np.array(payload["active"])
      self.joint_names = [str(name) for name in payload["joint_names"]]
      self.body_names = [str(name) for name in payload["body_names"]]
      self.fall_first_tick = np.array(payload["fall_first_tick"])
      self.stop_labels = [str(label) for label in payload["stop_labels"]]
      self.stop_code = np.array(payload["stop_code"])
      self.period_seconds = float(payload.get("period_seconds", 0.02))
      self.joint_position = np.array(payload["joint_position"])
      self.root_position = np.array(payload["root_position"])
      self.root_quaternion = np.array(payload["root_quaternion_wxyz"])
      self.tilt_degrees = np.array(payload["tilt_degrees"]).reshape(-1, self.rows)
      self.body_positions = (
        np.array(payload["body_positions"])
        if "body_positions" in payload.files
        else None
      )
    width = self.joint_position.shape[2]
    if width != len(self.joint_names):
      raise PlayRecordingError(
        f"recording declares {len(self.joint_names)} joints but holds {width}"
      )
    self.width = width
    self.count = self.ticks.shape[0]

  def stop_label(self, row: int) -> str | None:
    code = int(self.stop_code[row])
    if code < 0 or code >= len(self.stop_labels):
      return None
    return self.stop_labels[code]

  def chosen_row(self, requested: int | str | None) -> int:
    """Resolve ``--row``: an index, or one of the named rows."""
    fell = np.flatnonzero(self.fall_first_tick >= 0)
    if isinstance(requested, str):
      token = requested.strip().lower()
      if token == "first-fall":
        if not fell.size:
          raise PlayRecordingError("no row in this recording falls")
        return int(fell[np.argmin(self.fall_first_tick[fell])])
      if token == "last-fall":
        if not fell.size:
          raise PlayRecordingError("no row in this recording falls")
        return int(fell[np.argmax(self.fall_first_tick[fell])])
      if token == "survivor":
        alive = [r for r in range(self.rows) if int(self.fall_first_tick[r]) < 0]
        if not alive:
          raise PlayRecordingError("every row in this recording falls")
        return alive[0]
      raise PlayRecordingError(
        f"--row {requested!r} is not a row index or one of "
        f"first-fall, last-fall, survivor"
      )
    if requested is not None:
      if not 0 <= requested < self.rows:
        raise PlayRecordingError(
          f"--row {requested} is outside the {self.rows} recorded rows"
        )
      return requested
    if fell.size:
      return int(fell[np.argmin(self.fall_first_tick[fell])])
    return 0

  def play_end(self, row: int, margin_ticks: int) -> int:
    """Last tick worth playing: a fallen row stops moving after its fall."""
    fall = int(self.fall_first_tick[row])
    if fall < 0:
      return self.count - 1
    return int(min(self.count - 1, fall + margin_ticks))

  def row_table(self, limit: int) -> list[tuple[int, str]]:
    """Rows that fell first (earliest fall first), then survivors.

    The order is the option order, so the most interesting rows come first in a
    dropdown.  Each group keeps the recorded row order.
    """
    fell = [int(r) for r in np.flatnonzero(self.fall_first_tick >= 0)]
    fell.sort(key=lambda r: (int(self.fall_first_tick[r]), r))
    fell = fell[:limit]
    survivors = [r for r in range(self.rows) if int(self.fall_first_tick[r]) < 0]
    rows = fell + survivors[: max(0, limit - len(fell))]
    return [(r, self.summary(r)) for r in rows]

  def summary(self, row: int) -> str:
    fall = int(self.fall_first_tick[row])
    if fall < 0:
      return f"row {row:3d} · no fall · {self.stop_label(row) or 'unknown'}"
    return f"row {row:3d} · falls at tick {fall}"


class RecordingPlayback:
  """Maps recorded rows onto MuJoCo positions without running physics."""

  def __init__(self, recording: Recording, model: Any, row: int, recenter: bool):
    mujoco = _dependency("mujoco")
    self._mujoco = mujoco
    self.recording = recording
    self.model = model
    self.row = row
    self.data = mujoco.MjData(model)
    self.joint_qpos_address = self._joint_addresses()
    self.body_index = self._body_indexes()
    self.origin = np.zeros(3)
    if recenter:
      self.origin = np.array(recording.root_position[0, row], dtype=np.float64)
      self.origin[2] = 0.0
    self._apply(0)

  def _joint_addresses(self) -> np.ndarray:
    mujoco = self._mujoco
    addresses = []
    for name in self.recording.joint_names:
      joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
      if joint_id < 0:
        raise PlayRecordingError(
          f"robot model has no joint named {name!r}; the recording and the model "
          f"must come from the same robot"
        )
      if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
        raise PlayRecordingError(f"joint {name!r} is not a hinge joint")
      addresses.append(int(self.model.jnt_qposadr[joint_id]))
    return np.array(addresses, dtype=np.int64)

  def _body_indexes(self) -> np.ndarray:
    mujoco = self._mujoco
    indexes = []
    for name in self.recording.body_names:
      body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
      if body_id < 0:
        raise PlayRecordingError(f"robot model has no body named {name!r}")
      indexes.append(body_id)
    return np.array(indexes, dtype=np.int64)

  def _apply(self, tick: int) -> None:
    recording = self.recording
    qpos = self.data.qpos
    qpos[0:3] = recording.root_position[tick, self.row] - self.origin
    qpos[3:7] = recording.root_quaternion[tick, self.row]
    qpos[self.joint_qpos_address] = recording.joint_position[tick, self.row]
    self._mujoco.mj_kinematics(self.model, self.data)

  def seek(self, tick: int) -> int:
    tick = int(np.clip(tick, 0, self.recording.count - 1))
    self._apply(tick)
    return tick

  def lowest_body_z(self) -> float:
    return float(self.data.xpos[self.body_index, 2].min())

  def rebuilt_bodies(self) -> np.ndarray:
    return np.array(self.data.xpos[self.body_index], dtype=np.float64)


def _check(recording: Recording, model: Any, row: int, tolerance: float) -> int:
  """Rebuild sampled ticks and compare them with the recorded body positions."""
  if recording.body_positions is None:
    raise PlayRecordingError(
      "this recording holds no body positions; record again with --record-bodies"
    )
  playback = RecordingPlayback(recording, model, row, recenter=False)
  fall = int(recording.fall_first_tick[row])
  sample = {0, recording.count // 4, recording.count // 2, recording.count - 1}
  if 0 <= fall < recording.count:
    sample.update({fall, min(recording.count - 1, fall + 10)})
  print(f"row {row}: {recording.summary(row)}   tolerance {tolerance:.0e} m")
  worst = 0.0
  for tick in sorted(sample):
    playback.seek(tick)
    rebuilt = playback.rebuilt_bodies()
    truth = recording.body_positions[tick, row].reshape(rebuilt.shape)
    difference = float(np.max(np.abs(rebuilt - truth)))
    worst = max(worst, difference)
    print(f"  tick {int(recording.ticks[tick]):4d}  max_body_difference_m {difference:.3e}")
  verdict = "PASS" if worst <= tolerance else "FAIL"
  print(f"worst {worst:.3e} m: {verdict}")
  return 0 if verdict == "PASS" else 1


def _frame_ticks(recording: Recording, row: int, end: int) -> list[int]:
  """Ticks worth a still: standing, before the fall, the fall, then the ground."""
  fall = int(recording.fall_first_tick[row])
  if fall < 0:
    return sorted({0, end // 2, end})
  return sorted({0, max(0, fall - 25), fall, min(end, fall + 10), end})


def _caption(recording: Recording, playback: RecordingPlayback, tick: int) -> str:
  row = playback.row
  fall = int(recording.fall_first_tick[row])
  parts = [
    f"{recording.member} row {row}",
    f"t={tick * recording.period_seconds:.2f}s",
    f"tilt={float(recording.tilt_degrees[tick, row]):.0f}deg",
  ]
  if fall >= 0:
    parts.append(f"fall@{fall}" + ("" if tick < fall else " past"))
  return " | ".join(parts)


def _track_camera(mujoco: Any, playback: RecordingPlayback) -> Any:
  """Follow the robot at one tick, so a GIF shows the pose and not the travel."""
  points = playback.rebuilt_bodies()
  center = points.mean(axis=0)
  extent = points.max(axis=0) - points.min(axis=0)
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  camera.lookat[:] = [float(center[0]), float(center[1]), float(center[2] + 0.1)]
  camera.distance = max(2.4, 1.6 * float(extent.max()))
  camera.azimuth = CameraAzimuth
  camera.elevation = CameraElevation
  return camera


def _save_gif(
  recording: Recording,
  playback: RecordingPlayback,
  out_path: Path,
  *,
  stride: int,
  height: int,
  width: int,
  margin_ticks: int,
  track: bool = False,
) -> int:
  """Write the played row as an animated GIF, with a caption on each frame."""
  mujoco = _dependency("mujoco")
  from PIL import Image, ImageDraw

  if out_path.exists():
    raise PlayRecordingError(f"refusing to overwrite an existing GIF: {out_path}")
  end = recording.play_end(playback.row, margin_ticks)
  ticks = list(range(0, end + 1, stride))
  renderer = mujoco.Renderer(playback.model, height=height, width=width)
  fixed_camera = _still_camera(mujoco, playback)
  frames: list[Any] = []
  for tick in ticks:
    playback.seek(tick)
    camera = _track_camera(mujoco, playback) if track else fixed_camera
    renderer.update_scene(playback.data, camera=camera)
    image = Image.fromarray(renderer.render())
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, width, 16], fill=(0, 0, 0))
    draw.text((3, 3), _caption(recording, playback, tick), fill=(235, 235, 235))
    frames.append(image)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  delay_ms = max(20, int(round(1000.0 * recording.period_seconds * stride)))
  frames[0].save(
    out_path,
    save_all=True,
    append_images=frames[1:],
    duration=delay_ms,
    loop=0,
    optimize=True,
  )
  print(
    f"  {out_path}  {len(frames)} frames at {delay_ms} ms "
    f"({out_path.stat().st_size / 1e6:.1f} MB)"
  )
  return 0


def _still_camera(mujoco: Any, playback: RecordingPlayback) -> Any:
  """Frame the whole played trajectory: the robot walks about 4 m while it falls."""
  lower, upper = _trajectory_box(playback)
  extent = upper - lower
  fov = np.deg2rad(float(playback.model.vis.global_.fovy))
  half = max(float(extent[0]), float(extent[1]), float(extent[2])) * 0.5
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  camera.lookat[:] = [
    float(lower[0] + extent[0] * 0.5),
    float(lower[1] + extent[1] * 0.5),
    float(lower[2] + max(0.35, extent[2] * 0.4)),
  ]
  camera.distance = max(2.0, 1.15 * half / np.tan(fov * 0.5))
  camera.azimuth = CameraAzimuth
  camera.elevation = CameraElevation
  return camera


def _save_frames(
  recording: Recording,
  playback: RecordingPlayback,
  out_dir: Path,
  ticks: list[int],
  height: int,
  width: int,
) -> int:
  """Write one PNG per tick, framed on the played trajectory.

  MuJoCo elevates the free camera by a positive angle, so the camera looks down
  at the floor rather than up at its underside.
  """
  mujoco = _dependency("mujoco")
  import imageio.v2 as imageio

  out_dir.mkdir(parents=True, exist_ok=True)
  renderer = mujoco.Renderer(playback.model, height=height, width=width)
  camera = _still_camera(mujoco, playback)
  for tick in ticks:
    playback.seek(tick)
    renderer.update_scene(playback.data, camera=camera)
    image = renderer.render()
    path = out_dir / f"tick_{int(recording.ticks[tick]):04d}.png"
    if path.exists():
      raise PlayRecordingError(f"refusing to overwrite an existing frame: {path}")
    imageio.imwrite(path, image)
    print(
      f"  {path}  tick {int(recording.ticks[tick])} "
      f"({int(recording.ticks[tick]) * recording.period_seconds:.2f} s) · "
      f"tilt {float(recording.tilt_degrees[tick, playback.row]):.1f} deg · "
      f"lowest body {playback.lowest_body_z():.3f} m"
    )
  return 0


def _trajectory_box(playback: RecordingPlayback) -> tuple[np.ndarray, np.ndarray]:
  """Bounds of the played body positions, in the displayed frame."""
  recording = playback.recording
  end = recording.play_end(playback.row, 25)
  samples = np.linspace(0, end, num=min(32, end + 1), dtype=int)
  points = recording.body_positions[samples, playback.row] if (
    recording.body_positions is not None
  ) else None
  if points is None:
    collected = []
    for tick in samples:
      playback.seek(int(tick))
      collected.append(playback.rebuilt_bodies())
    points = np.concatenate(collected)
  points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
  if playback.origin.any():
    points = points - playback.origin
  return points.min(axis=0), points.max(axis=0)


def _camera_frame(playback: RecordingPlayback) -> tuple[np.ndarray, float]:
  """Centre for the interactive camera, with a radius that covers the motion."""
  lower, upper = _trajectory_box(playback)
  return (lower + upper) * 0.5, float(
    max(1.0, np.linalg.norm(upper - lower) * 0.6)
  )


def _open_viewer(
  recording: Recording,
  playback: RecordingPlayback,
  *,
  host: str,
  port: int,
  rate: float,
  loop: bool,
  margin_ticks: int,
  selftest_seconds: float | None,
) -> int:
  import viser

  from mjlab.viewer.viser.scene import MjlabViserScene

  server = viser.ViserServer(host=host, port=port, label="mjlab-recording")
  rendered = 0
  try:
    scene = MjlabViserScene(server, playback.model, num_envs=1)
    scene.debug_visualization_enabled = False
    scene.camera_tracking_enabled = False
    end = recording.play_end(playback.row, margin_ticks)
    look_at, radius = _camera_frame(playback)
    lock = threading.RLock()
    state: dict[str, Any] = {
      "tick": 0,
      "playing": False,
      "speed": rate,
      "loop": loop,
      "row": playback.row,
      "next": 0.0,
      "syncing": False,
    }

    def _center(client: Any) -> None:
      fov = np.deg2rad(45.0)
      direction = np.array([-1.0, -1.0, 0.45])
      direction /= np.linalg.norm(direction)
      distance = max(1.5, 1.2 * radius / np.sin(fov * 0.5))
      client.camera.up_direction = (0.0, 0.0, 1.0)
      client.camera.fov = fov
      client.camera.position = look_at + direction * distance
      client.camera.look_at = look_at

    def _status_html() -> str:
      tick = int(state["tick"])
      fall = int(recording.fall_first_tick[int(state["row"])])
      tilt = float(recording.tilt_degrees[tick, int(state["row"])])
      time_seconds = tick * recording.period_seconds
      phase = int(recording.phase[tick])
      mode = "playing" if state["playing"] else "paused"
      fall_text = "no fall" if fall < 0 else f"fall at tick {fall}"
      past = "" if fall < 0 or tick < fall else " · past the fall"
      return (
        f"<b>{recording.member}</b> · row {int(state['row'])} · {fall_text}<br>"
        f"tick {tick} / {recording.count - 1} · {time_seconds:.2f} s · "
        f"{'bootstrap' if phase == 0 else 'continuation'}<br>"
        f"tilt {tilt:.1f} deg · lowest body {playback.lowest_body_z():.3f} m · "
        f"{mode} at {state['speed']:.2f}x{past}"
      )

    def _render() -> None:
      nonlocal rendered
      tick = playback.seek(int(state["tick"]))
      scene.update_from_mjdata(playback.data)
      status.content = _status_html()
      rendered += 1
      return tick

    def _sync() -> None:
      state["syncing"] = True
      try:
        tick_slider.value = int(state["tick"])
        play.label = "Pause" if state["playing"] else "Play"
      finally:
        state["syncing"] = False
      _render()

    status = server.gui.add_html(_status_html())
    with server.gui.add_folder("Playback"):
      play = server.gui.add_button("Play")
      step = server.gui.add_button("Step")
      back = server.gui.add_button("Back")
      reset = server.gui.add_button("Reset")
      tick_slider = server.gui.add_slider(
        "Tick", min=0, max=end, step=1, initial_value=0,
        hint="Seek pauses playback.",
      )
      speed = server.gui.add_slider(
        "Speed", min=0.05, max=1.0, step=0.05, initial_value=rate,
        hint="1.00x is real time (50 control steps per second).",
      )
      loop_box = server.gui.add_checkbox("Loop", initial_value=loop)
      center = server.gui.add_button("Center view")
      options = recording.row_table(limit=40)
      if all(row != int(state["row"]) for row, _ in options):
        options.append((int(state["row"]), recording.summary(int(state["row"]))))
      labels = [label for _, label in options]
      codes = [row for row, _ in options]
      row_select = server.gui.add_dropdown(
        "Row",
        options=labels,
        initial_value=recording.summary(int(state["row"])),
        hint=f"the first {len(labels)} of {recording.rows} rows; "
        f"rows that fall are listed first",
      )

    def _switch_row(code: int) -> None:
      state["row"] = code
      playback.row = code
      nonlocal end
      end = recording.play_end(code, margin_ticks)
      tick_slider.max = max(1, end)
      state["tick"] = 0
      state["playing"] = False
      _sync()

    @play.on_click
    def _(_: Any) -> None:
      with lock:
        state["playing"] = not state["playing"]
        state["next"] = time.perf_counter() + recording.period_seconds / state["speed"]
        _sync()

    @step.on_click
    def _(_: Any) -> None:
      with lock:
        state["playing"] = False
        state["tick"] = min(end, int(state["tick"]) + 1)
        _sync()

    @back.on_click
    def _(_: Any) -> None:
      with lock:
        state["playing"] = False
        state["tick"] = max(0, int(state["tick"]) - 1)
        _sync()

    @reset.on_click
    def _(_: Any) -> None:
      with lock:
        state["playing"] = False
        state["tick"] = 0
        _sync()

    @center.on_click
    def _(_: Any) -> None:
      for client in server.get_clients().values():
        _center(client)

    @speed.on_update
    def _(_: Any) -> None:
      with lock:
        state["speed"] = float(speed.value)
        state["next"] = time.perf_counter() + recording.period_seconds / state["speed"]
        _render()

    @loop_box.on_update
    def _(_: Any) -> None:
      with lock:
        state["loop"] = bool(loop_box.value)

    @tick_slider.on_update
    def _(_: Any) -> None:
      with lock:
        if state["syncing"]:
          return
        state["playing"] = False
        state["tick"] = int(tick_slider.value)
        _sync()

    @row_select.on_update
    def _(_: Any) -> None:
      with lock:
        if state["syncing"]:
          return
        if str(row_select.value) in labels:
          _switch_row(codes[labels.index(str(row_select.value))])

    def advance(now: float) -> None:
      if not state["playing"] or now < state["next"]:
        return
      with lock:
        tick = int(state["tick"]) + 1
        if tick > end:
          if not state["loop"]:
            state["playing"] = False
            play.label = "Play"
            _render()
            return
          tick = 0
        state["tick"] = tick
      with lock:
        state["next"] = now + recording.period_seconds / state["speed"]
        _sync()

    server.on_client_connect(_center)
    with lock:
      _sync()
    for client in server.get_clients().values():
      _center(client)
    print(
      f"[INFO] {recording.path.name}: {recording.member}, row {state['row']}, "
      f"ticks 0-{end} of {recording.count - 1}, {recording.period_seconds:.3f} s per tick"
    )
    print(f"[INFO] playback: http://{host}:{port} (Ctrl+C to close)")

    if selftest_seconds is not None:
      deadline = time.perf_counter() + selftest_seconds
      state["playing"] = True
      state["next"] = time.perf_counter()
      while time.perf_counter() < deadline:
        advance(time.perf_counter())
        time.sleep(_VIEWER_STEP_SECONDS)
      print(
        f"[SELFTEST] rendered {rendered} frames, reached tick {int(state['tick'])} "
        f"of {end}, no exception"
      )
      return 0 if rendered > 1 else 1

    while True:
      advance(time.perf_counter())
      time.sleep(_VIEWER_STEP_SECONDS)
  except KeyboardInterrupt:
    return 0
  finally:
    server.stop()


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
    description="Play one recorded robot state file in a browser."
  )
  parser.add_argument("recording", type=Path)
  parser.add_argument("--row", default=None,
                      help="row index, or first-fall (default), last-fall, survivor")
  parser.add_argument("--model", type=Path, default=_DEFAULT_MODEL,
                      help="robot MJCF used for the picture")
  parser.add_argument("--rate", type=float, default=0.25,
                      help="initial playback speed, 1.0 is real time")
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=8080)
  parser.add_argument("--loop", action="store_true", help="start with loop enabled")
  parser.add_argument("--margin-ticks", type=int, default=25,
                      help="ticks played after a fall, 25 is half a second")
  parser.add_argument("--no-recenter", action="store_true",
                      help="keep the recorded world position instead of moving "
                           "the row's start to the origin")
  parser.add_argument("--tolerance", type=float, default=1.0e-4)
  parser.add_argument("--list-rows", action="store_true",
                      help="list candidate rows and exit")
  parser.add_argument("--check", action="store_true",
                      help="rebuild sampled ticks, compare with the recording, exit")
  parser.add_argument("--frames", type=Path, default=None, metavar="DIR",
                      help="write PNG stills for this row instead of opening a browser")
  parser.add_argument("--gif", type=Path, default=None, metavar="PATH",
                      help="write an animated GIF of this row instead of a browser")
  parser.add_argument("--gif-stride", type=int, default=1,
                      help="render every K-th tick into the GIF (default 1)")
  parser.add_argument("--track", action="store_true",
                      help="follow the robot in each GIF frame instead of holding "
                           "one camera on the whole trajectory")
  parser.add_argument("--frame-height", type=int, default=480)
  parser.add_argument("--frame-width", type=int, default=640)
  parser.add_argument("--selftest", type=float, default=None, metavar="SECONDS",
                      help="render headlessly for this many seconds, then exit")
  args = parser.parse_args(argv)

  try:
    recording = Recording(args.recording)
    row = recording.chosen_row(args.row)
    if args.list_rows:
      print(f"{recording.member}: {recording.rows} rows, {recording.count} ticks")
      for _code, label in recording.row_table(limit=recording.rows):
        print(f"  {label}")
      return 0
    work_dir = Path(tempfile.mkdtemp(prefix="x2-play-"))
    try:
      model = _build_model(args.model, work_dir)
      if args.check:
        return _check(recording, model, row, args.tolerance)
      if args.frames is not None:
        playback = RecordingPlayback(
          recording, model, row, recenter=not args.no_recenter
        )
        end = recording.play_end(row, args.margin_ticks)
        print(f"row {row}: {recording.summary(row)}")
        return _save_frames(
          recording,
          playback,
          args.frames,
          _frame_ticks(recording, row, end),
          args.frame_height,
          args.frame_width,
        )
      if args.gif is not None:
        if args.gif_stride < 1:
          raise PlayRecordingError("--gif-stride must be positive")
        playback = RecordingPlayback(
          recording, model, row, recenter=not args.no_recenter
        )
        print(f"row {row}: {recording.summary(row)}")
        return _save_gif(
          recording,
          playback,
          args.gif,
          stride=args.gif_stride,
          height=args.frame_height,
          width=args.frame_width,
          margin_ticks=args.margin_ticks,
          track=args.track,
        )
      playback = RecordingPlayback(
        recording, model, row, recenter=not args.no_recenter
      )
      return _open_viewer(
        recording,
        playback,
        host=args.host,
        port=args.port,
        rate=args.rate,
        loop=args.loop,
        margin_ticks=args.margin_ticks,
        selftest_seconds=args.selftest,
      )
    finally:
      shutil.rmtree(work_dir, ignore_errors=True)
  except PlayRecordingError as exc:
    print(f"play_recording: {exc}", file=sys.stderr)
    return 2


if __name__ == "__main__":
  raise SystemExit(main())
