"""Play one tracking reference kinematically: no robot dynamics, no policy.

The reference file supplies the root pose (``body_pos_w[:, 0]`` and
``body_quat_w[:, 0]``) and the 31 joint angles (``joint_pos``).  This tool writes
that pose into the X2 model every frame, calls ``mj_forward`` and renders.  It
runs no physics, no policy and no episode reset, so playback never stops because
the reference would make the simulated robot fall.

That makes it the tool for inspecting a generated reference, not for judging
whether a policy can track it.  Playback repeats until the window closes; the
camera tracks the pelvis, so the reference stays framed while it moves.  A generated joint-pose rollout carries the
recorded clip's root channels, so the root path and root heading shown here
belong to the clip while the joint angles belong to the model.

Usage::

    uv run python scripts/play_reference.py \\
        --motion-file logs/.../generated.npz [--speed 1.0] [--passes N]

    # verify the root/joint assembly against the recorded body poses and exit
    uv run python scripts/play_reference.py --motion-file clip.npz --check-only
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.agibot_x2.x2_constants import X2_XML

_ROOT_WIDTH = 7
"""``x``, ``y``, ``z``, then the ``wxyz`` root quaternion."""

_TOLERANCE = 5.0e-3
"""Metres. Above this the stored bodies disagree with the stored joints."""

_BROKEN_TOLERANCE = 0.1
"""Metres. Above this the joint order or the root layout is certainly wrong."""


def _load(path: Path) -> dict[str, np.ndarray | float]:
  try:
    with np.load(path, allow_pickle=False) as data:
      missing = [
        key
        for key in ("joint_pos", "body_pos_w", "body_quat_w", "fps")
        if key not in data.files
      ]
      if missing:
        raise SystemExit(f"{path} is not a tracking motion file; missing {missing}")
      return {
        "joint_pos": np.asarray(data["joint_pos"], dtype=np.float64),
        "body_pos_w": np.asarray(data["body_pos_w"], dtype=np.float64),
        "body_quat_w": np.asarray(data["body_quat_w"], dtype=np.float64),
        "fps": float(np.asarray(data["fps"]).ravel()[0]),
      }
  except (OSError, ValueError) as exc:
    raise SystemExit(f"could not read motion file {path}: {exc}") from exc


def _model() -> mujoco.MjModel:
  """Compile the X2 model with a ground plane for visual reference."""
  spec = mujoco.MjSpec.from_file(str(X2_XML))
  plane = spec.worldbody.add_geom()
  plane.name = "floor"
  plane.type = mujoco.mjtGeom.mjGEOM_PLANE
  plane.size = np.array([3.0, 3.0, 0.1])
  plane.rgba = np.array([0.30, 0.33, 0.38, 1.0])
  return spec.compile()


def _follow_root(viewer: object, model: mujoco.MjModel) -> None:
  """Track the pelvis so the reference stays framed while it moves."""
  body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
  if body < 0:
    body = 1
  cam = viewer.cam
  cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
  cam.trackbodyid = int(body)
  cam.distance = 2.4
  cam.azimuth = 135.0
  cam.elevation = -15.0


def _pose(
  joint_pos: np.ndarray, body_pos_w: np.ndarray, body_quat_w: np.ndarray, frame: int
) -> np.ndarray:
  """Return the ``qpos`` of one reference frame."""
  return np.concatenate((body_pos_w[frame, 0], body_quat_w[frame, 0], joint_pos[frame]))


def check(
  model: mujoco.MjModel, data: mujoco.MjData, arrays: dict, frames: np.ndarray
) -> tuple[float, float]:
  """Return the worst and the best forward-kinematics error over ``frames``.

  The best frame decides whether the joint order is usable at all: a file whose
  body channels are the recorded clip's while its joints are a model's matches
  at its clip-prefix rows and disagrees later.  A scrambled joint order matches
  at no frame.
  """
  worst = 0.0
  best = float("inf")
  for frame in frames:
    data.qpos[:] = _pose(
      arrays["joint_pos"], arrays["body_pos_w"], arrays["body_quat_w"], int(frame)
    )
    mujoco.mj_forward(model, data)
    error = float(np.abs(data.xpos[1:] - arrays["body_pos_w"][frame]).max())
    worst = max(worst, error)
    best = min(best, error)
  return worst, best


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--motion-file", type=Path, required=True)
  parser.add_argument("--start-frame", type=int, default=0)
  parser.add_argument("--end-frame", type=int, default=None)
  parser.add_argument(
    "--speed", type=float, default=1.0, help="playback rate; 0.5 is half speed"
  )
  parser.add_argument(
    "--passes",
    type=int,
    default=0,
    help="number of passes over the range; 0 repeats until the window closes",
  )
  parser.add_argument(
    "--once", action="store_true", help="play the range once instead of repeating"
  )
  parser.add_argument(
    "--free-camera",
    action="store_true",
    help="keep the camera where you put it instead of tracking the pelvis",
  )
  parser.add_argument(
    "--check-only",
    action="store_true",
    help="verify the pose assembly and print the worst error, then exit",
  )
  parser.add_argument(
    "--allow-mismatch",
    action="store_true",
    help="play even when forward kinematics disagrees with the stored bodies",
  )
  parser.add_argument(
    "--close-at-end",
    action="store_true",
    help="close the window when the range ends instead of holding the last pose",
  )
  args = parser.parse_args()

  arrays = _load(args.motion_file)
  joint_pos = arrays["joint_pos"]
  body_pos_w = arrays["body_pos_w"]
  body_quat_w = arrays["body_quat_w"]
  frames_total = int(joint_pos.shape[0])
  start = max(0, min(int(args.start_frame), frames_total - 1))
  end = (
    frames_total if args.end_frame is None else min(int(args.end_frame), frames_total)
  )
  if end <= start:
    raise SystemExit(f"empty frame range [{start}, {end}) of {frames_total}")
  if args.speed <= 0.0:
    raise SystemExit("--speed must be positive")

  model = _model()
  data = mujoco.MjData(model)
  expected = _ROOT_WIDTH + joint_pos.shape[1]
  if model.nq != expected:
    raise SystemExit(f"model nq {model.nq} does not match {expected} pose values")

  # The layout check covers the whole file, not just the requested range, so it
  # does not depend on where playback starts.
  probe = np.unique(
    np.linspace(0, frames_total - 1, min(frames_total, 9), dtype=np.int64)
  )
  worst, best = check(model, data, arrays, probe)
  print(
    f"{args.motion_file.name}: {frames_total} frames, {joint_pos.shape[1]} joints, "
    f"fps {arrays['fps']:g}",
    flush=True,
  )
  print(
    f"pose assembly error: {worst * 1000.0:.2f} mm worst, {best * 1000.0:.2f} mm best",
    flush=True,
  )
  layout_ok = best <= _TOLERANCE
  if not layout_ok:
    print(
      "no sampled frame reproduces its stored body poses, so the joint order or "
      "the root layout is wrong",
      file=sys.stderr,
    )
    if args.check_only:
      return 2
    if not args.allow_mismatch:
      print("refusing to play; pass --allow-mismatch to play anyway", file=sys.stderr)
      return 2
  elif worst > _TOLERANCE:
    print(
      "note: the stored body poses do not match the stored joints. A generated "
      "joint-pose file keeps the recorded clip's body channels, so the root path "
      "and heading are the clip's while the joints are the model's. The viewer "
      "shows forward kinematics of the joints under the stored root.",
      file=sys.stderr,
    )
  if args.check_only:
    return 0

  period = 1.0 / (float(arrays["fps"]) * args.speed)
  import mujoco.viewer as mjv

  try:
    viewer = mjv.launch_passive(model, data)
  except Exception as exc:  # no display, no GLFW, and so on
    print(f"cannot open the viewer ({type(exc).__name__}: {exc})", file=sys.stderr)
    print("use --check-only on a headless host", file=sys.stderr)
    return 1

  if not args.free_camera:
    _follow_root(viewer, model)
  passes = 1 if args.once else max(0, int(args.passes))
  repeats = "until the window closes" if passes == 0 else f"{passes} pass(es)"
  print(
    f"playing frames {start}..{end - 1} at {args.speed:g}x, {repeats}; "
    "close the window to stop",
    flush=True,
  )
  done = 0
  try:
    while viewer.is_running():
      for frame in range(start, end):
        if not viewer.is_running():
          break
        deadline = time.monotonic() + period
        data.qpos[:] = _pose(joint_pos, body_pos_w, body_quat_w, frame)
        mujoco.mj_forward(model, data)
        viewer.sync()
        remaining = deadline - time.monotonic()
        if remaining > 0.0:
          time.sleep(remaining)
      done += 1
      if passes and done >= passes:
        break
    if not args.close_at_end:
      # Hold the last pose until the window is closed.
      while viewer.is_running():
        viewer.sync()
        time.sleep(0.05)
  finally:
    viewer.close()
  # GLFW teardown segfaults after the window closes on some drivers, so leave
  # before interpreter shutdown. The viewer holds no state worth flushing.
  sys.stdout.flush()
  sys.stderr.flush()
  os._exit(0)


if __name__ == "__main__":
  raise SystemExit(main())
