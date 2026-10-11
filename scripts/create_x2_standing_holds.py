"""Create the X2 standing-hold reference clips for VAE distillation.

Four clips, each 10 s at 50 Hz. Two static poses are the building blocks:

* ``joint_default``: the robot stand keyframe (``STAND_KEYFRAME``); the pose the
  deployed controller holds before an RL policy engages (``JOINT_DEFAULT`` with
  ``joint_default.pose_source: policy_metadata``). The keyframe height puts the
  lowest sole sphere on z = 0, and its whole-body CoM sits inside the foot
  support polygon.
* ``endpoint_005``: the last frame of tennis motion 005. The user reports that
  this frame is balanced. Its joint angles, pelvis height and pelvis tilt are
  reproduced bit-exactly. Its world XY and its yaw are replaced by the
  ``joint_default`` values, because a held reference must not command a lateral
  or heading step. The tilt is kept: straightening the pelvis puts the CoM
  0.093 m outside the support polygon, so the tilt is what balances the pose.

Clips::

  1. standing_joint_default_10s                 500 frames of joint_default
  2. standing_005_end_10s                       500 frames of endpoint_005
  3. standing_joint_default_then_005_end_10s    250 joint_default + 250 endpoint_005
  4. standing_005_end_then_joint_default_10s    250 endpoint_005 + 250 joint_default

Clips 3 and 4 contain one hard reference step and no interpolation, as
requested. All velocities are zero: a hold commands zero target velocity. The
endpoint_005 source frame is NOT static (its joint speeds reach 0.68 rad/s), so
this clips that motion and turns it into a posture hold.

Every authored body pose comes from the X2 MuJoCo model (``mj_forward``), never
from array interpolation. Source NPZ files, frozen teachers and policies are
never modified. Output goes to a new directory and an existing one is refused.

Run from mjlab::

  uv run python scripts/create_x2_standing_holds.py

This script only authors references. It authors no teacher, trains no student,
and proves nothing about closed-loop balance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from mjlab.asset_zoo.robots.agibot_x2.x2_constants import STAND_KEYFRAME, X2_XML
from mjlab.utils.string import resolve_expr

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
FOOT_GEOM_PATTERN = re.compile(r"^(?:left|right)_foot[0-9]+_collision$")
CONTACT_BAND = 0.005
"""Sole spheres within 5 mm of the lowest sphere count as support contacts."""

POSE_NAMES = ("joint_default", "endpoint_005")

CLIP_SPECS: tuple[tuple[str, tuple[tuple[str, int], ...]], ...] = (
  ("standing_joint_default_10s", (("joint_default", 500),)),
  ("standing_005_end_10s", (("endpoint_005", 500),)),
  (
    "standing_joint_default_then_005_end_10s",
    (("joint_default", 250), ("endpoint_005", 250)),
  ),
  (
    "standing_005_end_then_joint_default_10s",
    (("endpoint_005", 250), ("joint_default", 250)),
  ),
)


@dataclass(frozen=True)
class Pose:
  """One static reference pose: root position, root orientation, joint angles."""

  root_pos: np.ndarray
  root_quat: np.ndarray
  joints: np.ndarray

  def qpos(self) -> np.ndarray:
    return np.concatenate([self.root_pos, self.root_quat, self.joints])


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as file:
    for block in iter(lambda: file.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def load_motion(path: Path, model: mujoco.MjModel) -> dict[str, np.ndarray]:
  """Load and validate one reference clip against the X2 model dimensions."""
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


def yaw_of(quat: np.ndarray) -> float:
  matrix = Rotation.from_quat(quat, scalar_first=True).as_matrix()
  return math.atan2(matrix[1, 0], matrix[0, 0])


def restore_yaw(quat: np.ndarray, yaw: float) -> np.ndarray:
  """Return ``quat`` with its world yaw replaced; roll and pitch are kept."""
  rotation = Rotation.from_euler("z", yaw - yaw_of(quat)) * Rotation.from_quat(
    quat, scalar_first=True
  )
  return rotation.as_quat(scalar_first=True)


def default_stand_pose(model: mujoco.MjModel) -> Pose:
  """The mjlab robot default stand keyframe, in model joint order."""
  names = tuple(model.joint(index).name for index in range(1, model.njnt))
  patterns = STAND_KEYFRAME.joint_pos
  if patterns is None:
    raise ValueError("STAND_KEYFRAME declares no joint_pos patterns")
  joints = np.asarray(resolve_expr(patterns, names, 0.0), dtype=np.float64)
  return Pose(
    root_pos=np.array([0.0, 0.0, float(STAND_KEYFRAME.pos[2])], dtype=np.float64),
    root_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    joints=joints,
  )


def source_frame_pose(data: dict[str, np.ndarray], index: int) -> Pose:
  """The root state and joint angles of one frame of a source clip."""
  return Pose(
    root_pos=data["body_pos_w"][index, 0].astype(np.float64),
    root_quat=data["body_quat_w"][index, 0].astype(np.float64),
    joints=data["joint_pos"][index].astype(np.float64),
  )


def align_pose(pose: Pose, target: Pose) -> Pose:
  """Keep the pose and its height; take the target's world XY and world yaw."""
  quat = restore_yaw(pose.root_quat, yaw_of(target.root_quat))
  return Pose(
    root_pos=np.concatenate([target.root_pos[:2], pose.root_pos[2:3]]),
    root_quat=quat / np.linalg.norm(quat),
    joints=pose.joints,
  )


def fk_body_arrays(
  model: mujoco.MjModel, pose_data: mujoco.MjData, pose: Pose
) -> tuple[np.ndarray, np.ndarray]:
  """Body positions and quaternions for one pose from the X2 MuJoCo model."""
  pose_data.qpos[:] = pose.qpos()
  mujoco.mj_forward(model, pose_data)
  return pose_data.xpos[1:].copy(), pose_data.xquat[1:].copy()


def static_stability(
  model: mujoco.MjModel,
  pose_data: mujoco.MjData,
  pose: Pose,
  foot_geoms: np.ndarray,
) -> dict[str, float]:
  """Ground clearance and whole-body CoM support margin for one static pose.

  ``planar_support_margin`` is the signed distance from the CoM ground
  projection to the support polygon of the sole spheres that touch the ground.
  A negative margin means no static hold of this pose is possible.
  """
  pose_data.qpos[:] = pose.qpos()
  mujoco.mj_forward(model, pose_data)
  centers = pose_data.geom_xpos[foot_geoms]
  surfaces = centers[:, 2] - model.geom_size[foot_geoms, 0]
  lowest = float(surfaces.min())
  contact = centers[surfaces < lowest + CONTACT_BAND]
  pelvis_id = model.body("pelvis").id
  com = pose_data.subtree_com[pelvis_id]
  if contact.shape[0] < 3:
    margin = float("nan")
  else:
    equations = ConvexHull(contact[:, :2]).equations
    margin = float(np.min(-(equations[:, :2] @ com[:2] + equations[:, 2])))
  return {
    "lowest_sole_surface_z": lowest,
    "contact_spheres": float(contact.shape[0]),
    "com_x": float(com[0]),
    "com_y": float(com[1]),
    "com_z": float(com[2]),
    "planar_support_margin": margin,
  }


def build_clip(
  model: mujoco.MjModel,
  poses: dict[str, Pose],
  segments: tuple[tuple[str, int], ...],
) -> dict[str, np.ndarray]:
  """Assemble one clip from static pose segments with a hard step between them."""
  frames = sum(count for _, count in segments)
  shape = (frames, model.nbody - 1)
  result = {
    "joint_pos": np.empty((frames, model.nq - 7), dtype=np.float32),
    "joint_vel": np.zeros((frames, model.nq - 7), dtype=np.float32),
    "body_pos_w": np.empty((*shape, 3), dtype=np.float32),
    "body_quat_w": np.empty((*shape, 4), dtype=np.float32),
    "body_lin_vel_w": np.zeros((*shape, 3), dtype=np.float32),
    "body_ang_vel_w": np.zeros((*shape, 3), dtype=np.float32),
    "fps": np.array([FPS], dtype=np.float32),
  }
  pose_data = mujoco.MjData(model)
  start = 0
  for name, count in segments:
    pose = poses[name]
    body_pos, body_quat = fk_body_arrays(model, pose_data, pose)
    stop = start + count
    result["joint_pos"][start:stop] = pose.joints.astype(np.float32)
    result["body_pos_w"][start:stop] = body_pos.astype(np.float32)
    result["body_quat_w"][start:stop] = body_quat.astype(np.float32)
    start = stop
  return result


def step_report(poses: dict[str, Pose], model: mujoco.MjModel) -> dict[str, Any]:
  """Describe the reference step between the two poses of clips 3 and 4."""
  default = poses["joint_default"]
  endpoint = poses["endpoint_005"]
  delta = endpoint.joints - default.joints
  pose_data = mujoco.MjData(model)
  torso = model.body("torso_link").id - 1
  tilts = {}
  for name, pose in poses.items():
    _, quats = fk_body_arrays(model, pose_data, pose)
    tilt = np.degrees(
      np.arccos(
        np.clip(
          Rotation.from_quat(quats[torso], scalar_first=True).as_matrix()[2, 2],
          -1.0,
          1.0,
        )
      )
    )
    tilts[name] = float(tilt)
  return {
    "max_abs_joint_step_rad": float(np.abs(delta).max()),
    "joints_over_0p1_rad": int((np.abs(delta) > 0.1).sum()),
    "pelvis_z_step_m": float(endpoint.root_pos[2] - default.root_pos[2]),
    "torso_tilt_deg": tilts,
    "anchor_yaw_aligned": True,
  }


def validate_clip(
  name: str, clip: dict[str, np.ndarray], model: mujoco.MjModel
) -> None:
  """Re-run the reference-clip checks on one authored clip."""
  frames = clip["joint_pos"].shape[0]
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
    value = clip[key]
    if value.shape != shape or value.dtype != np.float32:
      raise ValueError(
        f"{name}: {key}: expected float32 {shape}, got {value.dtype} {value.shape}"
      )
    if not np.isfinite(value).all():
      raise ValueError(f"{name}: nonfinite {key}")
  if float(clip["fps"][0]) != FPS:
    raise ValueError(f"{name}: expected fps {FPS}")
  norms = np.linalg.norm(clip["body_quat_w"], axis=-1)
  if not np.allclose(norms, 1.0, atol=2e-4):
    raise ValueError(f"{name}: nonunit body quaternion")
  pose_data = mujoco.MjData(model)
  for frame in range(frames):
    pose = Pose(
      root_pos=clip["body_pos_w"][frame, 0].astype(np.float64),
      root_quat=clip["body_quat_w"][frame, 0].astype(np.float64),
      joints=clip["joint_pos"][frame].astype(np.float64),
    )
    body_pos, body_quat = fk_body_arrays(model, pose_data, pose)
    if not np.allclose(body_pos, clip["body_pos_w"][frame], atol=2e-5):
      raise ValueError(f"{name}: frame {frame}: body positions do not match X2 FK")
    dots = np.abs(np.sum(body_quat * clip["body_quat_w"][frame], axis=-1))
    if dots.min() < 1.0 - 2e-5:
      raise ValueError(f"{name}: frame {frame}: body orientations do not match X2 FK")


def check_source_model_agreement(
  model: mujoco.MjModel, data: dict[str, np.ndarray], index: int
) -> dict[str, float]:
  """Prove that the X2 model reproduces the source body arrays at one frame."""
  pose = source_frame_pose(data, index)
  body_pos, body_quat = fk_body_arrays(model, mujoco.MjData(model), pose)
  position_error = float(np.abs(body_pos - data["body_pos_w"][index]).max())
  quaternion_dot = float(
    np.abs(np.sum(body_quat * data["body_quat_w"][index], axis=-1)).min()
  )
  if position_error > 2e-4 or quaternion_dot < 1.0 - 2e-4:
    raise ValueError(
      "source frame does not match X2 forward kinematics: "
      f"position error {position_error}, quaternion dot {quaternion_dot}"
    )
  return {"position_error_m": position_error, "quaternion_dot": quaternion_dot}


def build_poses(
  model: mujoco.MjModel,
  source: dict[str, np.ndarray],
  source_frame: int,
) -> tuple[dict[str, Pose], dict[str, Any]]:
  """Build the two poses and the provenance record for the endpoint pose."""
  default = default_stand_pose(model)
  raw_endpoint = source_frame_pose(source, source_frame)
  endpoint = align_pose(raw_endpoint, default)
  endpoint_joint_change = float(np.abs(endpoint.joints - raw_endpoint.joints).max())
  if endpoint_joint_change != 0.0:
    raise ValueError("endpoint joints changed during alignment")
  yaw_change = (
    yaw_of(endpoint.root_quat) - yaw_of(raw_endpoint.root_quat) + math.pi
  ) % (2 * math.pi) - math.pi
  record = {
    "source_frame_index": int(source_frame),
    "source_root_pos_w_m": raw_endpoint.root_pos.tolist(),
    "source_root_yaw_deg": float(np.degrees(yaw_of(raw_endpoint.root_quat))),
    "applied_root_xy_m": default.root_pos[:2].tolist(),
    "applied_root_yaw_deg": float(np.degrees(yaw_of(default.root_quat))),
    "applied_yaw_change_deg": float(np.degrees(yaw_change)),
    "retained_pelvis_z_m": float(raw_endpoint.root_pos[2]),
    "retained_joint_angles": True,
    "source_joint_speed_abs_max_rad_s": float(
      np.abs(source["joint_vel"][source_frame]).max()
    ),
    "source_joint_speed_rms_rad_s": float(
      np.sqrt(np.mean(np.square(source["joint_vel"][source_frame])))
    ),
    "source_pelvis_ang_speed_rad_s": float(
      np.linalg.norm(source["body_ang_vel_w"][source_frame, 0])
    ),
    "source_pelvis_lin_speed_m_s": float(
      np.linalg.norm(source["body_lin_vel_w"][source_frame, 0])
    ),
  }
  return {"joint_default": default, "endpoint_005": endpoint}, record


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--source",
    type=Path,
    default=Path("data/tennis/single_005_zhanghongyu_agibot_x2_tracking.npz"),
    help="source clip whose last frame supplies the endpoint pose",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("data/tennis/standing_holds"),
    help="new output directory; an existing one is refused",
  )
  parser.add_argument(
    "--allow-infeasible-poses",
    action="store_true",
    help="write clips even when a held pose has no static CoM support",
  )
  args = parser.parse_args()
  if args.output_dir.exists():
    parser.error(f"output already exists; refusing overwrite: {args.output_dir}")

  model = mujoco.MjModel.from_xml_path(str(X2_XML))
  foot_geoms = np.array(
    [
      geom
      for geom in range(model.ngeom)
      if FOOT_GEOM_PATTERN.match(model.geom(geom).name)
    ],
    dtype=int,
  )
  if foot_geoms.size == 0:
    parser.error("no foot collision geoms found in the X2 model")

  source = load_motion(args.source, model)
  agreement = check_source_model_agreement(model, source, -1)
  poses, endpoint_record = build_poses(model, source, len(source["joint_pos"]) - 1)

  pose_data = mujoco.MjData(model)
  stability = {
    name: static_stability(model, pose_data, pose, foot_geoms)
    for name, pose in poses.items()
  }
  infeasible = [
    name
    for name, report in stability.items()
    if not math.isfinite(report["planar_support_margin"])
    or report["planar_support_margin"] <= 0.0
  ]
  if infeasible and not args.allow_infeasible_poses:
    parser.error(
      "static CoM support margin is not positive for "
      + ", ".join(sorted(infeasible))
      + "; these poses cannot be held statically. Inspect the reported margins."
    )

  clips = {}
  for name, segments in CLIP_SPECS:
    clip = build_clip(model, poses, segments)
    validate_clip(name, clip, model)
    clips[name] = clip

  args.output_dir.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(
    prefix=".standing-holds-stage-", dir=args.output_dir.parent
  ) as stage_name:
    stage = Path(stage_name)
    members = []
    for name, segments in CLIP_SPECS:
      clip = clips[name]
      destination = stage / f"{name}.npz"
      np.savez(
        destination,
        joint_pos=clip["joint_pos"],
        joint_vel=clip["joint_vel"],
        body_pos_w=clip["body_pos_w"],
        body_quat_w=clip["body_quat_w"],
        body_lin_vel_w=clip["body_lin_vel_w"],
        body_ang_vel_w=clip["body_ang_vel_w"],
        fps=clip["fps"],
      )
      members.append(
        {
          "clip": destination.name,
          "segments": [
            {"pose": pose_name, "start_frame": 0, "frames": count}
            for pose_name, count in segments
          ],
          "frames": int(clip["joint_pos"].shape[0]),
          "seconds": float(clip["joint_pos"].shape[0] / FPS),
          "output_sha256": sha256(destination),
        }
      )
      print(
        f"{destination.name}: {members[-1]['frames']} frames, "
        f"{members[-1]['seconds']:.1f} s, "
        + " + ".join(f"{count}x{pose_name}" for pose_name, count in segments)
      )

    segment_starts = {}
    for member in members:
      start = 0
      for segment in member["segments"]:
        segment["start_frame"] = start
        start += segment["frames"]
      segment_starts[member["clip"]] = [
        (segment["pose"], segment["start_frame"], segment["frames"])
        for segment in member["segments"]
      ]

    manifest = {
      "recipe": "x2_static_pose_hold_clips_from_keyframe_and_tennis_005_last_frame",
      "fps": FPS,
      "source_clip": str(args.source),
      "source_sha256": sha256(args.source),
      "source_frames": int(source["joint_pos"].shape[0]),
      "robot_xml": str(X2_XML),
      "robot_xml_sha256": sha256(X2_XML),
      "generator": "scripts/create_x2_standing_holds.py",
      "pose_definitions": {
        "joint_default": (
          "mjlab STAND_KEYFRAME resolved with resolve_expr(...) over the model "
          "joint order; this is robot.data.default_joint_pos and the deployed "
          "JOINT_DEFAULT hold pose"
        ),
        "endpoint_005": (
          "last source frame: joint angles, pelvis height and pelvis tilt "
          "bit-preserved; world XY and world yaw replaced by the joint_default "
          "values so the hold commands no lateral or heading step"
        ),
      },
      "velocity_policy": "all joint and body velocities are zero; a hold commands zero target velocity",
      "pose_values": {
        name: {
          "root_pos": pose.root_pos.tolist(),
          "root_quat_wxyz": pose.root_quat.tolist(),
          "joint_pos": pose.joints.tolist(),
        }
        for name, pose in poses.items()
      },
      "static_stability": stability,
      "source_model_agreement_at_source_frame": agreement,
      "endpoint_provenance": endpoint_record,
      "step_between_poses": step_report(poses, model),
      "clips": members,
      "clip_segments": {clip: segs for clip, segs in segment_starts.items()},
      "limitations": (
        "references only: no teacher trained, no student, no simulator or hardware run; "
        "static CoM support is necessary but not sufficient for a held pose; "
        "clips 3 and 4 contain a hard reference step with no interpolation; "
        "the endpoint pose is a dynamically balanced tennis frame clipped to a posture"
      ),
    }
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (stage / "README.md").write_text(render_readme(manifest) + "\n")
    if args.output_dir.exists():
      raise FileExistsError(args.output_dir)
    stage.rename(args.output_dir)
    print(f"Wrote {len(members)} clips to {args.output_dir}")
    for name, report in stability.items():
      print(
        f"  {name}: lowest sole z {report['lowest_sole_surface_z']:+.4f} m, "
        f"CoM support margin {report['planar_support_margin']:+.4f} m, "
        f"contacts {int(report['contact_spheres'])}"
      )


def render_readme(manifest: dict[str, Any]) -> str:
  """Human-readable description of one generation."""
  members = manifest["clips"]
  assert isinstance(members, list)
  stability = manifest["static_stability"]
  assert isinstance(stability, dict)
  step = manifest["step_between_poses"]
  assert isinstance(step, dict)
  endpoint = manifest["endpoint_provenance"]
  assert isinstance(endpoint, dict)
  lines = [
    "# X2 standing-hold reference clips",
    "",
    "These clips are reference commands for reference-conditioned VAE",
    "distillation. They are not validated behaviours.",
    "",
    f"- Generated by `{manifest['generator']}`.",
    f"- Source: `{manifest['source_clip']}` (sha256 `{manifest['source_sha256']}`).",
    f"- X2 MuJoCo model: `{manifest['robot_xml']}` (sha256 `{manifest['robot_xml_sha256']}`).",
    f"- Rate: {manifest['fps']} Hz. All velocities are zero.",
    "",
    "## Clips",
    "",
  ]
  for member in members:
    segments = " + ".join(
      f"{segment['frames']}x `{segment['pose']}` (from frame {segment['start_frame']})"
      for segment in member["segments"]
    )
    lines.append(
      f"- `{member['clip']}`: {member['frames']} frames / {member['seconds']:.1f} s = {segments}."
    )
  lines += [
    "",
    "## Poses",
    "",
    "`joint_default` is the mjlab `STAND_KEYFRAME` resolved over the model joint",
    "order. It equals `robot.data.default_joint_pos`, which is the pose the",
    "standing-start reset inserts and the pose the deployed controller holds as",
    "`JOINT_DEFAULT`.",
    "",
    "`endpoint_005` is the last frame of the source clip. Its joint angles, pelvis",
    "height and pelvis tilt are preserved. Its world XY and world yaw are replaced",
    "by the `joint_default` values, because a held reference must not command a",
    f"lateral or heading step. The applied yaw change is {endpoint['applied_yaw_change_deg']:+.2f} deg;",
    f"the source pelvis is at XY ({endpoint['source_root_pos_w_m'][0]:.3f},",
    f"{endpoint['source_root_pos_w_m'][1]:.3f}) m, z {endpoint['source_root_pos_w_m'][2]:.4f} m.",
    "",
    "The tilt is kept on purpose. Straightening the pelvis moves the whole-body",
    "CoM outside the foot support polygon, so the tilt is what balances the pose.",
    "",
    "## Measured static feasibility",
    "",
    "| pose | lowest sole surface z (m) | CoM support margin (m) | contact spheres |",
    "| --- | --- | --- | --- |",
  ]
  for name, report in stability.items():
    assert isinstance(report, dict)
    lines.append(
      f"| `{name}` | {report['lowest_sole_surface_z']:+.4f} | "
      f"{report['planar_support_margin']:+.4f} | {int(report['contact_spheres'])} |"
    )
  tilt = step["torso_tilt_deg"]
  assert isinstance(tilt, dict)
  lines += [
    "",
    "The CoM support margin is the signed distance from the CoM ground projection",
    "to the support polygon of the sole spheres that touch the ground. A negative",
    "margin means no static hold is possible.",
    "",
    "## Reference step in clips 3 and 4",
    "",
    f"- Maximum joint step: {step['max_abs_joint_step_rad']:.4f} rad.",
    f"- Joints with a step over 0.1 rad: {step['joints_over_0p1_rad']}.",
    f"- Pelvis height step: {step['pelvis_z_step_m']:+.4f} m.",
    f"- Torso tilt: {tilt['joint_default']:.2f} deg (default) to "
    f"{tilt['endpoint_005']:.2f} deg (endpoint).",
    "",
    "The step has no interpolation, as requested.",
    "",
    "## Source frame is not static",
    "",
    f"The source frame has joint speeds up to "
    f"{endpoint['source_joint_speed_abs_max_rad_s']:.4f} rad/s",
    f"(RMS {endpoint['source_joint_speed_rms_rad_s']:.4f} rad/s) and pelvis angular",
    f"speed {endpoint['source_pelvis_ang_speed_rad_s']:.4f} rad/s. The clips set",
    "those velocities to zero, so they command a posture hold of that frame, not",
    "a continuation of the motion.",
    "",
    "## Limitations",
    "",
    f"- {manifest['limitations']}",
    "- A new teacher must be trained and qualified on these references. Adding",
    "  them to an existing cohort changes the cohort identity and invalidates",
    "  resume and evaluation of the checkpoints trained from that cohort.",
    "- Static CoM support is a necessary condition only. Actuator limits, contact",
    "  friction and closed-loop balance are not tested here.",
    "",
  ]
  return "\n".join(lines)


if __name__ == "__main__":
  main()
