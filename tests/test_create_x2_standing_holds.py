"""Offline, synthetic tests for the X2 standing-hold reference generator."""

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mjlab.asset_zoo.robots.agibot_x2.x2_constants import X2_XML

FPS = 50.0


@pytest.fixture(scope="module")
def holds_module() -> Any:
  script = Path(__file__).parents[1] / "scripts" / "create_x2_standing_holds.py"
  spec = importlib.util.spec_from_file_location("create_x2_standing_holds", script)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
  return mujoco.MjModel.from_xml_path(str(X2_XML))


def write_source(
  module: Any, model: mujoco.MjModel, path: Path, *, last_hip_offset: float = 0.0
) -> dict[str, np.ndarray]:
  """Write a moving 4-frame source clip whose last frame is a chosen pose.

  The motion is in the arms, which barely move the whole-body CoM, so the last
  frame stays statically supportable. ``last_hip_offset`` shifts the leg joints
  of that last frame and is used to build a deliberately unsupportable pose.
  """
  data = mujoco.MjData(model)
  joints = []
  positions = []
  quaternions = []
  for index in range(4):
    q = module.default_stand_pose(model).joints.copy()
    q[15:22] += 0.02 * index
    if index == 3:
      q[:6] += last_hip_offset
    data.qpos[:] = np.concatenate(
      [[0.0, 0.0, 0.66], Rotation.from_euler("z", 0.2).as_quat(scalar_first=True), q]
    )
    mujoco.mj_forward(model, data)
    joints.append(data.qpos[7:].copy())
    positions.append(data.xpos[1:].copy())
    quaternions.append(data.xquat[1:].copy())
  joint_pos = np.asarray(joints, dtype=np.float32)
  body_pos = np.asarray(positions, dtype=np.float32)
  clip: dict[str, Any] = {
    "joint_pos": joint_pos,
    "joint_vel": np.gradient(joint_pos, 1.0 / FPS, axis=0).astype(np.float32),
    "body_pos_w": body_pos,
    "body_quat_w": np.asarray(quaternions, dtype=np.float32),
    "body_lin_vel_w": np.gradient(body_pos, 1.0 / FPS, axis=0).astype(np.float32),
    "body_ang_vel_w": np.zeros_like(body_pos),
    "fps": np.array([FPS], dtype=np.float32),
  }
  np.savez(path, **clip)
  return clip


def build(
  module: Any, model: mujoco.MjModel, tmp_path: Path, **kwargs: Any
) -> dict[str, Any]:
  source = write_source(module, model, tmp_path / "source.npz", **kwargs)
  poses, record = module.build_poses(model, source, len(source["joint_pos"]) - 1)
  return {"poses": poses, "record": record, "source": source}


def test_clips_are_static_holds_with_one_exact_step(
  holds_module: Any, model: mujoco.MjModel, tmp_path: Path
) -> None:
  bundle = build(holds_module, model, tmp_path)
  poses = bundle["poses"]
  source = bundle["source"]
  clips = {
    name: holds_module.build_clip(model, poses, segments)
    for name, segments in holds_module.CLIP_SPECS
  }
  assert set(clips) == {name for name, _ in holds_module.CLIP_SPECS}
  for clip in clips.values():
    assert clip["joint_pos"].shape == (500, 31)
    assert clip["body_pos_w"].shape == (500, 32, 3)
    assert float(clip["fps"][0]) == FPS
    for key in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
      np.testing.assert_array_equal(clip[key], 0.0)
    steps = np.flatnonzero(np.abs(np.diff(clip["joint_pos"], axis=0)).max(axis=1) > 0)
    assert steps.size <= 1

  default = clips["standing_joint_default_10s"]
  endpoint = clips["standing_005_end_10s"]
  np.testing.assert_allclose(
    default["joint_pos"][0], poses["joint_default"].joints, rtol=1e-6
  )
  np.testing.assert_array_equal(endpoint["joint_pos"][0], source["joint_pos"][-1])
  assert np.array_equal(
    endpoint["joint_pos"], np.broadcast_to(endpoint["joint_pos"][0], (500, 31))
  )
  assert np.array_equal(
    clips["standing_joint_default_then_005_end_10s"]["joint_pos"],
    np.concatenate([default["joint_pos"][:250], endpoint["joint_pos"][250:]]),
  )
  assert np.array_equal(
    clips["standing_005_end_then_joint_default_10s"]["joint_pos"],
    np.concatenate([endpoint["joint_pos"][:250], default["joint_pos"][250:]]),
  )
  for name in (
    "standing_joint_default_then_005_end_10s",
    "standing_005_end_then_joint_default_10s",
  ):
    assert (
      np.flatnonzero(np.abs(np.diff(clips[name]["joint_pos"], axis=0)).max(axis=1) > 0)[
        0
      ]
      == 249
    )


def test_alignment_keeps_height_and_tilt_and_clears_xy_yaw(
  holds_module: Any, model: mujoco.MjModel, tmp_path: Path
) -> None:
  poses = build(holds_module, model, tmp_path)["poses"]
  default = poses["joint_default"]
  endpoint = poses["endpoint_005"]
  raw = holds_module.source_frame_pose(
    holds_module.load_motion(tmp_path / "source.npz", model), -1
  )
  np.testing.assert_array_equal(endpoint.joints, raw.joints)
  np.testing.assert_allclose(endpoint.root_pos[:2], default.root_pos[:2])
  np.testing.assert_allclose(endpoint.root_pos[2], raw.root_pos[2])
  assert abs(holds_module.yaw_of(endpoint.root_quat)) < 1e-9
  expected = Rotation.from_quat(raw.root_quat, scalar_first=True).as_euler("xyz")
  actual = Rotation.from_quat(endpoint.root_quat, scalar_first=True).as_euler("xyz")
  np.testing.assert_allclose(actual[:2], expected[:2], atol=1e-9)
  assert abs(np.linalg.norm(endpoint.root_quat) - 1.0) < 1e-12


def test_authored_frames_match_model_forward_kinematics(
  holds_module: Any, model: mujoco.MjModel, tmp_path: Path
) -> None:
  poses = build(holds_module, model, tmp_path)["poses"]
  data = mujoco.MjData(model)
  for name, pose in poses.items():
    positions, quaternions = holds_module.fk_body_arrays(model, data, pose)
    data.qpos[:] = pose.qpos()
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(positions, data.xpos[1:], atol=1e-9)
    np.testing.assert_allclose(np.abs((quaternions * data.xquat[1:]).sum(-1)), 1.0)
    assert name in holds_module.POSE_NAMES


def test_main_writes_clips_manifest_and_refuses_overwrite(
  holds_module: Any, model: mujoco.MjModel, tmp_path: Path, monkeypatch: Any
) -> None:
  source = tmp_path / "source.npz"
  write_source(holds_module, model, source)
  output = tmp_path / "out"
  argv = [
    "create_x2_standing_holds.py",
    "--source",
    str(source),
    "--output-dir",
    str(output),
  ]
  monkeypatch.setattr(sys, "argv", argv)
  holds_module.main()
  members = json.loads((output / "manifest.json").read_text())["clips"]
  assert [member["clip"] for member in members] == [
    f"{name}.npz" for name, _ in holds_module.CLIP_SPECS
  ]
  for member in members:
    assert member["frames"] == 500
    assert (output / member["clip"]).is_file()
  assert (output / "README.md").is_file()
  with pytest.raises(SystemExit):
    holds_module.main()


def test_infeasible_pose_is_refused_unless_allowed(
  holds_module: Any, model: mujoco.MjModel, tmp_path: Path, monkeypatch: Any
) -> None:
  source = tmp_path / "leaning.npz"
  write_source(holds_module, model, source, last_hip_offset=0.3)
  output = tmp_path / "out"
  monkeypatch.setattr(
    sys,
    "argv",
    ["prog", "--source", str(source), "--output-dir", str(output)],
  )
  with pytest.raises(SystemExit):
    holds_module.main()
  assert not output.exists()
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "prog",
      "--source",
      str(source),
      "--output-dir",
      str(output),
      "--allow-infeasible-poses",
    ],
  )
  holds_module.main()
  assert (output / "manifest.json").is_file()
  report = json.loads((output / "manifest.json").read_text())["static_stability"]
  assert report["endpoint_005"]["planar_support_margin"] < 0.0
  assert math.isfinite(report["joint_default"]["planar_support_margin"])
