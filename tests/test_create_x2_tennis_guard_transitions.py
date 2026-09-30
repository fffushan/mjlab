"""Offline, synthetic tests for tennis guard reference generation."""

import importlib.util
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mjlab.asset_zoo.robots.agibot_x2.x2_constants import X2_XML

FPS = 50.0


@pytest.fixture(scope="module")
def guard_module() -> Any:
  script = (
    Path(__file__).parents[1] / "scripts" / "create_x2_tennis_guard_transitions.py"
  )
  spec = importlib.util.spec_from_file_location(
    "create_x2_tennis_guard_transitions", script
  )
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
  return mujoco.MjModel.from_xml_path(str(X2_XML))


def motion(model: mujoco.MjModel) -> dict[str, np.ndarray]:
  """Construct 4 complete, slightly moving reference frames by X2 FK."""
  data = mujoco.MjData(model)
  positions = []
  quaternions = []
  joints = []
  for i in range(4):
    data.qpos[:3] = [2.0 + i * 0.01, -1.0, 0.66 + i * 0.001]
    data.qpos[3:7] = Rotation.from_euler("z", 0.2 + i * 0.01).as_quat(scalar_first=True)
    data.qpos[7:] = 0.01 * i
    mujoco.mj_forward(model, data)
    positions.append(data.xpos[1:].copy())
    quaternions.append(data.xquat[1:].copy())
    joints.append(data.qpos[7:].copy())
  pos = np.asarray(positions, dtype=np.float32)
  quat = np.asarray(quaternions, dtype=np.float32)
  q = np.asarray(joints, dtype=np.float32)
  return {
    "joint_pos": q,
    "joint_vel": np.gradient(q, 1.0 / FPS, axis=0),
    "body_pos_w": pos,
    "body_quat_w": quat,
    "body_lin_vel_w": np.gradient(pos, 1.0 / FPS, axis=0),
    "body_ang_vel_w": np.zeros_like(pos),
    "fps": np.array([FPS], dtype=np.float32),
  }


def guard_pose(model: mujoco.MjModel) -> np.ndarray:
  pose = np.zeros(model.nq)
  pose[:3] = [-3.0, 6.0, 0.641]
  pose[3:7] = Rotation.from_euler("xyz", [0.08, -0.06, -1.0]).as_quat(scalar_first=True)
  pose[7:] = 0.24
  return pose


def yaw(quat: np.ndarray) -> float:
  matrix = Rotation.from_quat(quat, scalar_first=True).as_matrix()
  return math.atan2(matrix[1, 0], matrix[0, 0])


def test_interpolation_keeps_source_and_has_exact_endpoints(
  model: mujoco.MjModel, guard_module: Any
) -> None:
  original = motion(model)
  guard = guard_pose(model)
  steps = 25
  generated = guard_module.create_motion(original, guard, model, steps)
  n = original["joint_pos"].shape[0]
  assert generated["joint_pos"].shape == (n + 2 * steps, model.nq - 7)
  assert generated["body_pos_w"].shape == (n + 2 * steps, model.nbody - 1, 3)
  for key in ("joint_pos", "body_pos_w", "body_quat_w"):
    np.testing.assert_array_equal(generated[key][steps : steps + n], original[key])
  for key in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
    np.testing.assert_array_equal(
      generated[key][steps + 1 : steps + n - 1], original[key][1:-1]
    )
  for i, endpoint in ((0, 0), (-1, -1)):
    np.testing.assert_allclose(generated["joint_pos"][i], guard[7:], atol=1e-6)
    np.testing.assert_allclose(
      generated["body_pos_w"][i, 0, :2],
      original["body_pos_w"][endpoint, 0, :2],
      atol=1e-6,
    )
    np.testing.assert_allclose(generated["body_pos_w"][i, 0, 2], guard[2], atol=1e-6)
    assert (
      abs(
        yaw(generated["body_quat_w"][i, 0]) - yaw(original["body_quat_w"][endpoint, 0])
      )
      < 1e-5
    )
  np.testing.assert_allclose(
    np.linalg.norm(generated["body_quat_w"], axis=-1), 1.0, atol=1e-6
  )


def test_added_body_poses_fk_and_join_velocities(
  model: mujoco.MjModel, guard_module: Any
) -> None:
  original = motion(model)
  out = guard_module.create_motion(original, guard_pose(model), model, 25)
  data = mujoco.MjData(model)
  end_of_original = 25 + len(original["joint_pos"]) - 1
  for i in (0, 1, 24, end_of_original + 1, len(out["joint_pos"]) - 1):
    data.qpos[:3] = out["body_pos_w"][i, 0]
    data.qpos[3:7] = out["body_quat_w"][i, 0]
    data.qpos[7:] = out["joint_pos"][i]
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.xpos[1:], out["body_pos_w"][i], atol=1e-5)
  for i in (0, 25, end_of_original, len(out["joint_pos"]) - 1):
    prev = max(0, i - 1)
    nxt = min(len(out["joint_pos"]) - 1, i + 1)
    estimate = (out["joint_pos"][nxt] - out["joint_pos"][prev]) * (FPS / (nxt - prev))
    np.testing.assert_allclose(out["joint_vel"][i], estimate, atol=2e-6)


def test_aligned_guard_and_linear_samples_preserve_tilt(
  model: mujoco.MjModel, guard_module: Any
) -> None:
  guard = guard_pose(model)
  target = guard.copy()
  target[:3] = [5.0, 7.0, 0.58]
  target[3:7] = Rotation.from_euler("xyz", [-0.02, 0.05, 0.8]).as_quat(
    scalar_first=True
  )
  target[7:] = -0.2
  aligned = guard_module.aligned_guard(guard, target)
  np.testing.assert_allclose(aligned[:2], target[:2])
  np.testing.assert_allclose(aligned[2], guard[2])
  assert abs(yaw(aligned[3:7]) - yaw(target[3:7])) < 1e-6
  assert abs(np.linalg.norm(aligned[3:7]) - 1.0) < 1e-6
  samples = guard_module.interpolate_pose(aligned, target, np.array([0.0, 0.5, 1.0]))
  np.testing.assert_allclose(samples[0], aligned, atol=1e-7)
  np.testing.assert_allclose(samples[-1], target, atol=1e-7)
  np.testing.assert_allclose(samples[1, :3], (aligned[:3] + target[:3]) / 2)
  np.testing.assert_allclose(samples[1, 7:], (aligned[7:] + target[7:]) / 2)
