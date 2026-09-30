"""Add a 0.5 s X2 guard-to-motion prefix and motion-to-guard suffix.

The sources and frozen teacher assets are NEVER changed. The 040 frame-0 joint
pose and root height/orientation are the guard seed. Each guard is yaw-aligned
to its clip endpoint and translated to the endpoint's pelvis XY; the guard's
own pelvis height is retained. Both root position and joints are interpolated
linearly; root orientation uses shortest-path spherical interpolation. Body
poses for new frames come from the X2 MuJoCo model, not array interpolation.

Run from mjlab: uv run python scripts/create_x2_tennis_guard_transitions.py
Outputs data/tennis/guard040_linear05/ (new directory; refuses to overwrite).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from mjlab.asset_zoo.robots.agibot_x2.x2_constants import X2_XML

FIELDS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
  "fps",
)
FPS = 50.0


def load_motion(path: Path, model: mujoco.MjModel) -> dict[str, np.ndarray]:
  with np.load(path, allow_pickle=False) as file:
    if set(file.files) != set(FIELDS):
      raise ValueError(f"{path}: expected only {FIELDS}, found {file.files}")
    data = {key: file[key].copy() for key in FIELDS}
  frames = data["joint_pos"].shape[0]
  expected = {
    "joint_pos": (frames, model.nq - 7),
    "joint_vel": (frames, model.nq - 7),
    "body_pos_w": (frames, model.nbody - 1, 3),
    "body_quat_w": (frames, model.nbody - 1, 4),
    "body_lin_vel_w": (frames, model.nbody - 1, 3),
    "body_ang_vel_w": (frames, model.nbody - 1, 3),
    "fps": (1,),
  }
  for key, shape in expected.items():
    if data[key].shape != shape or data[key].dtype != np.float32:
      raise ValueError(
        f"{path}: {key}: expected float32 {shape}, got {data[key].dtype} {data[key].shape}"
      )
    if not np.isfinite(data[key]).all():
      raise ValueError(f"{path}: nonfinite {key}")
  if frames < 3 or float(data["fps"][0]) != FPS:
    raise ValueError(f"{path}: expected at least 3 frames at {FPS} Hz")
  norms = np.linalg.norm(data["body_quat_w"], axis=-1)
  if not np.allclose(norms, 1.0, atol=2e-4):
    raise ValueError(f"{path}: nonunit body quaternion")
  return data


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as file:
    for block in iter(lambda: file.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def root_state(data: dict[str, np.ndarray], frame: int) -> np.ndarray:
  return np.concatenate(
    [
      data["body_pos_w"][frame, 0],
      data["body_quat_w"][frame, 0],
      data["joint_pos"][frame],
    ]
  ).astype(np.float64)


def aligned_guard(guard: np.ndarray, endpoint: np.ndarray) -> np.ndarray:
  """Retain guard height; match the endpoint's pelvis XY and world yaw."""
  guard_rot = Rotation.from_quat(guard[3:7], scalar_first=True)
  endpoint_rot = Rotation.from_quat(endpoint[3:7], scalar_first=True)

  def yaw(rotation: Rotation) -> float:
    matrix = rotation.as_matrix()
    return math.atan2(matrix[1, 0], matrix[0, 0])

  rot = Rotation.from_euler("z", yaw(endpoint_rot) - yaw(guard_rot)) * guard_rot
  result = guard.copy()
  result[:2] = endpoint[:2]
  result[3:7] = rot.as_quat(scalar_first=True)
  return result


def interpolate_pose(a: np.ndarray, b: np.ndarray, phases: np.ndarray) -> np.ndarray:
  """Straight-line pelvis position and joint angles; quaternion SLERP."""
  blended = (1.0 - phases[:, None]) * a[None, :] + phases[:, None] * b[None, :]
  r0 = Rotation.from_quat(a[3:7], scalar_first=True)
  r1 = Rotation.from_quat(b[3:7], scalar_first=True)
  blended[:, 3:7] = Slerp([0.0, 1.0], Rotation.concatenate([r0, r1]))(phases).as_quat(
    scalar_first=True
  )
  return blended


def body_angular_velocity(quats: np.ndarray, fps: float) -> np.ndarray:
  """World-frame finite differences, using shortest rotation across signed quats."""
  frames, bodies, _ = quats.shape
  out = np.empty((frames, bodies, 3), dtype=np.float32)

  def relative(prev: np.ndarray, nxt: np.ndarray, delta_frames: int) -> np.ndarray:
    p = Rotation.from_quat(prev.reshape(-1, 4), scalar_first=True)
    n = Rotation.from_quat(nxt.reshape(-1, 4), scalar_first=True)
    return ((n * p.inv()).as_rotvec() * (fps / delta_frames)).reshape(-1, bodies, 3)

  out[0] = relative(quats[:1], quats[1:2], 1)[0]
  out[1:-1] = relative(quats[:-2], quats[2:], 2)
  out[-1] = relative(quats[-2:-1], quats[-1:], 1)[0]
  return out


def create_motion(
  source: dict[str, np.ndarray], guard: np.ndarray, model: mujoco.MjModel, steps: int
) -> dict[str, np.ndarray]:
  """Keep original poses bit-exact; replace velocities only at the two joins."""
  frames = source["joint_pos"].shape[0]
  count = frames + 2 * steps
  result = {
    key: np.empty((count, *value.shape[1:]), dtype=np.float32)
    for key, value in source.items()
    if key != "fps"
  }
  result["fps"] = source["fps"].copy()
  for key in result.keys() - {"fps"}:
    result[key][steps : steps + frames] = source[key]

  start, end = root_state(source, 0), root_state(source, -1)
  before = interpolate_pose(
    aligned_guard(guard, start), start, np.arange(steps) / steps
  )
  after = interpolate_pose(
    end, aligned_guard(guard, end), np.arange(1, steps + 1) / steps
  )
  model_data = mujoco.MjData(model)
  for frame, pose in list(enumerate(before)) + list(
    enumerate(after, start=steps + frames)
  ):
    model_data.qpos[:] = pose
    mujoco.mj_forward(model, model_data)
    result["joint_pos"][frame] = pose[7:]
    result["body_pos_w"][frame] = model_data.xpos[1:]
    result["body_quat_w"][frame] = model_data.xquat[1:]

  # Check actual, unmodified source seams agree with the very same X2 FK model.
  for index in (0, frames - 1):
    model_data.qpos[:] = root_state(source, index)
    mujoco.mj_forward(model, model_data)
    if not np.allclose(model_data.xpos[1:], source["body_pos_w"][index], atol=2e-4):
      raise ValueError(f"source frame {index}: body positions do not match X2 FK")
    dots = np.abs(np.sum(model_data.xquat[1:] * source["body_quat_w"][index], axis=-1))
    if np.min(dots) < 1.0 - 2e-4:
      raise ValueError(f"source frame {index}: body orientations do not match X2 FK")

  # At the joins, source velocity refers to its OLD temporal neighbours. Use
  # finite differences of the stitched positions/quaternions there and in the
  # new segments. All original interior arrays remain bit-identical.
  dt = 1.0 / FPS
  for pos_key, vel_key in (
    ("joint_pos", "joint_vel"),
    ("body_pos_w", "body_lin_vel_w"),
  ):
    derived = np.gradient(result[pos_key].astype(np.float64), dt, axis=0)
    result[vel_key][: steps + 1] = derived[: steps + 1]
    result[vel_key][steps + frames - 1 :] = derived[steps + frames - 1 :]
  angular = body_angular_velocity(result["body_quat_w"], FPS)
  result["body_ang_vel_w"][: steps + 1] = angular[: steps + 1]
  result["body_ang_vel_w"][steps + frames - 1 :] = angular[steps + frames - 1 :]
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input-dir", type=Path, default=Path("data/tennis"))
  parser.add_argument(
    "--output-dir", type=Path, default=Path("data/tennis/guard040_linear05")
  )
  parser.add_argument(
    "--duration", type=float, default=0.5, help="seconds per transition"
  )
  args = parser.parse_args()
  model = mujoco.MjModel.from_xml_path(str(X2_XML))
  steps_float = args.duration * FPS
  if not math.isfinite(steps_float) or steps_float < 1 or not steps_float.is_integer():
    parser.error("duration must be a positive integer number of 50 Hz steps")
  steps = int(steps_float)
  if args.output_dir.exists():
    parser.error(f"output already exists; refusing overwrite: {args.output_dir}")
  guard_paths = list(args.input_dir.glob("single_040_*_tracking.npz"))
  if len(guard_paths) != 1:
    parser.error("expected exactly one single_040_*_tracking.npz guard reference")
  guard_path = guard_paths[0]
  inputs = []
  missing = []
  for number in range(21):
    matches = list(args.input_dir.glob(f"single_{number:03d}_*_tracking.npz"))
    if len(matches) > 1:
      parser.error(f"multiple source clips found for tennis {number:03d}")
    if matches:
      inputs.append(matches[0])
    else:
      missing.append(f"{number:03d}")
  if missing != ["012"] or len(inputs) != 20:
    parser.error(f"expected clips 000–020 except missing 012, got missing {missing}")
  guard = root_state(load_motion(guard_path, model), 0)
  args.output_dir.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(
    prefix=".guard040-stage-", dir=args.output_dir.parent
  ) as stage_name:
    stage = Path(stage_name)
    members = []
    for source_path in inputs:
      source = load_motion(source_path, model)
      result = create_motion(source, guard, model, steps)
      destination = stage / source_path.name
      np.savez(
        destination,
        joint_pos=result["joint_pos"],
        joint_vel=result["joint_vel"],
        body_pos_w=result["body_pos_w"],
        body_quat_w=result["body_quat_w"],
        body_lin_vel_w=result["body_lin_vel_w"],
        body_ang_vel_w=result["body_ang_vel_w"],
        fps=result["fps"],
      )
      members.append(
        {
          "clip": source_path.name,
          "source_sha256": sha256(source_path),
          "output_sha256": sha256(destination),
          "source_frames": int(source["joint_pos"].shape[0]),
          "output_frames": int(result["joint_pos"].shape[0]),
          "original_frames_in_output": [steps, steps + len(source["joint_pos"]) - 1],
        }
      )
      print(
        f"{source_path.name}: {members[-1]['source_frames']} -> {members[-1]['output_frames']}"
      )
    manifest = {
      "recipe": "x2_guard040_frame0_linear_joint_and_root_position_slerp_root_orientation",
      "guard_source": guard_path.name,
      "guard_sha256": sha256(guard_path),
      "robot_xml": str(X2_XML),
      "robot_xml_sha256": sha256(X2_XML),
      "fps": FPS,
      "transition_seconds_each_end": args.duration,
      "added_frames_each_end": steps,
      "guard_alignment": "yaw to source endpoint, pelvis XY from endpoint, pelvis Z from guard frame0",
      "source_data_missing_clip_ids": missing,
      "velocity_method": "finite differences at inserted frames and both original seam frames; untouched source interior",
      "limitations": "straight interpolation can slide/penetrate feet, has nonzero endpoint velocity and has no guard dwell; no PPO/hold/sim validation",
      "members": members,
    }
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.output_dir.exists():
      raise FileExistsError(args.output_dir)
    stage.rename(args.output_dir)
    print(f"Wrote {len(members)} clips, missing {missing}, to {args.output_dir}")


if __name__ == "__main__":
  main()
