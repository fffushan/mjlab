"""Write one open-loop joint-pose rollout as a generated/control motion pair.

The tool loads one joint-pose flow checkpoint as a window sampler, rolls one
reference clip into a long generated series, and writes three files:

* ``generated.npz`` -- generated joint positions and velocities, with the body
  channels of the recorded clip for the same absolute frames;
* ``control.npz`` -- the same span of the recorded clip, unchanged;
* ``rollout.json`` -- the checkpoint schema, sampler settings and per-chunk
  diagnostics.

Both motion files hold the same seven keys and the same number of rows, so the
frozen teacher can run on each and be compared row by row.  This is open-loop
generation: the model never sees the robot, and only joint-space fidelity is
under test.

Usage::

    uv run python -m mjlab.scripts.joint_pose_rollout \\
        --checkpoint-file model_0400000.pt \\
        --motion-file data/tennis/single_000_zhanghongyu_agibot_x2_tracking.npz \\
        --output-dir logs/diagnostics/joint-pose-rollout/single_000 \\
        --start-frame 0 --target-frames 453 --steps 20 --seed 0

The projection statistics default to ``<checkpoint data_dir>/projection.npz``.
Pass ``--projection`` when that recorded dataset directory is absent.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

import mjlab
from mjlab.tasks.tracking.joint_pose.evaluate import (
  ChannelNormalization,
  JointPoseEvaluationError,
)
from mjlab.tasks.tracking.joint_pose.rollout import (
  ClipArrays,
  RolloutError,
  RolloutResult,
  joint_pose_flow,
  rollout,
)

MOTION_KEYS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
  "fps",
)
"""The seven keys a tracking motion file and both output files carry."""

OUTPUT_FILES = ("generated.npz", "control.npz", "rollout.json")

_BODY_KEYS = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
"""Channels the substitution copies from the recorded clip."""

_RECORD_KEYS = (
  "checkpoint",
  "global_step",
  "parameters",
  "schema",
  "projection",
  "clip",
  "seed",
)
"""Record fields the caller supplies; the rest are written from the rollout."""

SUBSTITUTION_STATEMENT = (
  "body arrays are the recorded clip rows for the same absolute frames; "
  "joint arrays are generated"
)
OPEN_LOOP_STATEMENT = "open-loop generation: the model never sees the robot"


def _read_motion(path: Path) -> dict[str, np.ndarray]:
  """Read the seven arrays of one tracking motion file."""
  try:
    with np.load(path, allow_pickle=False) as data:
      missing = [key for key in MOTION_KEYS if key not in data.files]
      if missing:
        raise SystemExit(
          f"{path} is not a tracking motion file; missing {', '.join(missing)}"
        )
      return {key: np.asarray(data[key]) for key in MOTION_KEYS}
  except (OSError, ValueError) as exc:
    raise SystemExit(f"could not read motion file {path}: {exc}") from exc


def _clip_arrays(arrays: dict[str, np.ndarray]) -> ClipArrays:
  """Build the rollout clip from one motion-file dict."""
  try:
    return ClipArrays.from_arrays(arrays)
  except (RolloutError, IndexError, TypeError, ValueError) as exc:
    raise SystemExit(f"motion file is invalid: {exc}") from exc


def _projection_path(override: str | None, scheme: dict[str, object]) -> Path:
  """Return the projection file, refusing a missing default with a hint."""
  if override is not None:
    path = Path(override).expanduser()
  else:
    path = Path(str(scheme["data_dir"])).expanduser() / "projection.npz"
  if not path.is_file():
    raise SystemExit(
      f"projection statistics not found: {path}; pass --projection <path>"
    )
  return path


def _normalization(path: Path) -> ChannelNormalization:
  """Read the per-channel mean and standard deviation from one npz file."""
  try:
    with np.load(path, allow_pickle=False) as stats:
      mean = np.asarray(stats["mean"])
      std = np.asarray(stats["std"])
  except (OSError, KeyError, ValueError) as exc:
    raise SystemExit(f"could not read projection {path}: {exc}") from exc
  try:
    return ChannelNormalization(mean=mean, std=std, source=str(path))
  except JointPoseEvaluationError as exc:
    raise SystemExit(f"projection {path} is invalid: {exc}") from exc


def _clip_rows(
  clip: dict[str, np.ndarray], key: str, start: int, rows: int
) -> np.ndarray:
  """Return one clip array as float32 rows ``[start, start + rows)``."""
  values = np.asarray(clip[key])
  available = 0 if values.ndim == 0 else int(values.shape[0])
  if available < start + rows:
    raise ValueError(
      f"clip {key} holds {available} rows, need {start + rows} from start_frame {start}"
    )
  return np.ascontiguousarray(values[start : start + rows], dtype=np.float32)


def _clip_fps(clip: dict[str, np.ndarray]) -> np.ndarray:
  """Return the clip rate as the ``[1]`` float32 array a motion file expects."""
  values = np.asarray(clip["fps"], dtype=np.float64).ravel()
  if values.size < 1 or not np.isfinite(values[0]) or values[0] <= 0.0:
    raise ValueError("clip fps must hold one finite positive value")
  return np.array([values[0]], dtype=np.float32)


def write_pair(
  *,
  out_dir: Path,
  clip: dict[str, np.ndarray],
  result: RolloutResult,
  start_frame: int,
  target_frames: int,
  record: dict[str, Any],
) -> dict[str, Any]:
  """Write ``generated.npz``, ``control.npz`` and ``rollout.json``.

  ``clip`` holds the seven tracking arrays the control arm is copied from.
  ``result`` supplies the generated joint columns; its body channels are
  replaced by the clip rows for the same absolute frames, because the model
  does not generate root translation or world orientation.  ``record`` carries
  the provenance fields the caller read from the checkpoint and sampler; the
  fields that follow from the arguments and the result are overwritten here.

  Both files hold the same rows.  A rollout that stopped early on
  ``drift_limit`` writes the rows it produced and records the shorter count.
  """
  missing = [key for key in MOTION_KEYS if key not in clip]
  if missing:
    raise ValueError(f"clip is missing {', '.join(missing)}")
  missing = [key for key in _RECORD_KEYS if key not in record]
  if missing:
    raise ValueError(f"rollout record is missing {', '.join(missing)}")
  first = int(start_frame)
  wanted = int(target_frames)
  rows = int(np.asarray(result.joint_pos).shape[0])
  if first < 0 or wanted < 1:
    raise ValueError(f"start_frame {first} and target_frames {wanted} are invalid")
  if rows > wanted:
    raise ValueError(f"rollout produced {rows} rows, more than target_frames {wanted}")
  if rows < wanted and result.stop_reason != "drift_limit":
    raise ValueError(
      f"rollout stopped at {rows} rows without reaching target_frames {wanted}"
    )
  joint_width = int(np.asarray(result.joint_pos).shape[1])
  clip_joints = np.asarray(clip["joint_pos"])
  if clip_joints.ndim != 2 or clip_joints.shape[1] != joint_width:
    raise ValueError(
      f"clip joint_pos shape {clip_joints.shape} does not match the generated "
      f"joint width {joint_width}"
    )
  schema = record["schema"]
  if not isinstance(schema, dict):
    raise ValueError("rollout record schema must be a mapping")
  try:
    window = int(schema["window_steps"])
    past = int(schema["past_steps"])
  except (KeyError, TypeError, ValueError) as exc:
    raise ValueError(f"rollout record schema is unusable: {exc}") from exc
  # Derived here, never trusted from the caller: the stride is the geometry.
  stride = window - past - 1
  if stride <= 0:
    raise ValueError(f"window_steps {window} leaves no stride after past_steps {past}")

  generated = {
    "joint_pos": np.ascontiguousarray(result.joint_pos[:rows], dtype=np.float32),
    "joint_vel": np.ascontiguousarray(result.joint_vel[:rows], dtype=np.float32),
    **{key: _clip_rows(clip, key, first, rows) for key in _BODY_KEYS},
    "fps": _clip_fps(clip),
  }
  control = {
    key: _clip_rows(clip, key, first, rows) for key in MOTION_KEYS if key != "fps"
  }
  control["fps"] = _clip_fps(clip)

  destination = Path(out_dir)
  destination.mkdir(parents=True, exist_ok=True)
  # Explicit keyword arguments keep the exported key set visible here.
  np.savez(
    destination / "generated.npz",
    joint_pos=generated["joint_pos"],
    joint_vel=generated["joint_vel"],
    body_pos_w=generated["body_pos_w"],
    body_quat_w=generated["body_quat_w"],
    body_lin_vel_w=generated["body_lin_vel_w"],
    body_ang_vel_w=generated["body_ang_vel_w"],
    fps=generated["fps"],
  )
  np.savez(
    destination / "control.npz",
    joint_pos=control["joint_pos"],
    joint_vel=control["joint_vel"],
    body_pos_w=control["body_pos_w"],
    body_quat_w=control["body_quat_w"],
    body_lin_vel_w=control["body_lin_vel_w"],
    body_ang_vel_w=control["body_ang_vel_w"],
    fps=control["fps"],
  )

  payload = dict(record)
  payload.update(
    start_frame=first,
    target_frames=wanted,
    rows=rows,
    stride=stride,
    chunk_count=len(result.chunks),
    steps=result.steps,
    stop_reason=result.stop_reason,
    chunks=[asdict(chunk) for chunk in result.chunks],
    substitution=SUBSTITUTION_STATEMENT,
    open_loop=OPEN_LOOP_STATEMENT,
  )
  text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
  (destination / "rollout.json").write_text(text, encoding="utf-8")
  return payload


def main(
  checkpoint_file: str,
  motion_file: str,
  output_dir: str,
  start_frame: int = 0,
  target_frames: int | None = None,
  steps: int = 20,
  seed: int = 0,
  device: str = "cpu",
  parameters: str = "auto",
  projection: str | None = None,
  drift_limit: float | None = None,
) -> None:
  """Roll one clip with one checkpoint and write the generated/control pair."""
  clip_path = Path(motion_file).expanduser()
  arrays = _read_motion(clip_path)
  clip = _clip_arrays(arrays)
  first = int(start_frame)
  if first < 0 or first >= clip.rows:
    raise SystemExit(
      f"start_frame {first} must lie in 0..{clip.rows - 1} for {clip_path}"
    )
  available = clip.rows - first
  frames = available if target_frames is None else int(target_frames)
  if frames > available:
    raise SystemExit(
      f"target_frames {frames} exceeds the {available} clip rows left after "
      f"start_frame {first}; pass a smaller --target-frames"
    )

  try:
    sampler = joint_pose_flow(checkpoint_file, parameters=parameters, device=device)
  except RolloutError as exc:
    raise SystemExit(f"checkpoint {checkpoint_file}: {exc}") from exc
  scheme = sampler.scheme

  projection_path = _projection_path(projection, scheme)
  normalization = _normalization(projection_path)

  generator = torch.Generator(device=device).manual_seed(int(seed))
  try:
    result = rollout(
      clip=clip,
      sampler=sampler,
      normalization=normalization,
      start_frame=first,
      target_frames=frames,
      steps=int(steps),
      generator=generator,
      drift_limit=drift_limit,
    )
  except (RolloutError, JointPoseEvaluationError) as exc:
    raise SystemExit(f"rollout failed: {exc}") from exc

  record: dict[str, Any] = {
    "checkpoint": str(Path(checkpoint_file).expanduser()),
    "global_step": sampler.global_step,
    "parameters": sampler.parameters,
    "schema": dict(scheme),
    "projection": str(projection_path),
    "clip": str(clip_path),
    "seed": int(seed),
  }
  destination = Path(output_dir).expanduser()
  record = write_pair(
    out_dir=destination,
    clip=arrays,
    result=result,
    start_frame=first,
    target_frames=frames,
    record=record,
  )
  print(
    f"rollout {record['rows']}/{record['target_frames']} rows, "
    f"{len(record['chunks'])} chunks, stride {record['stride']}, "
    f"stop {record['stop_reason']}, step {record['global_step']}",
    flush=True,
  )
  for name in OUTPUT_FILES:
    print(f"wrote {destination / name}", flush=True)


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
