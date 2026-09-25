"""Immutable tennis-endpoint state pool for X2 recovery fine-tuning.

Loads tennis motion-capture NPZ files and exposes validated, read-only NumPy
arrays of the last *N* frames of each trajectory — the candidate recovery
states for the 80% recovery environment group.

Conventions (verified against the X2 MuJoCo model):

* ``joint_pos``/``joint_vel`` follow the exact X2 MuJoCo hinge-joint order
  excluding the free joint (31 joints: hip→ankle, waist yaw/pitch/roll,
  shoulders→wrists, head yaw/pitch).
* ``body_pos_w``/``body_quat_w`` rows exclude the ``world`` body and begin at
  ``pelvis`` (index 0 = root link); 32 bodies total.
* Quaternions are **wxyz** unit quaternions.
* ``root_lin_vel_w`` is the root-link *body-frame-origin* linear velocity in
  world coordinates. This matches MuJoCo free-joint ``qvel[0:3]`` (the free
  joint stores world-frame linear velocity).
* ``root_ang_vel_w`` is the **world-frame** angular velocity of the root link.
  This does **not** match MuJoCo free-joint ``qvel[3:6]`` directly: the free
  joint stores **body-local** angular velocity. To write ``root_ang_vel_w`` to
  qvel, rotate it into the root body frame first (``R_root.T @ world_omega``)
  or use ``Entity.write_root_velocity`` / ``write_root_state_to_sim``, which
  perform that conversion internally (see ``entity/data.py``).

Importing this module does **not** read any dataset. Data is loaded lazily at
construction time via :meth:`EndpointPool.from_directory`.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

_SPLIT = Literal["train", "validation", "all"]

# Expected array shapes.
_NUM_JOINTS = 31
_NUM_BODIES = 32
_REQUIRED_KEYS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
  "fps",
)


def _sha256_file(path: Path) -> str:
  """Return the hex SHA-256 digest of a file."""
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def _load_one_file(
  path: Path,
) -> tuple[
  np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int
]:
  """Load and validate a single NPZ file, returning its arrays.

  Returns ``(joint_pos, joint_vel, body_pos_w, body_quat_w,
  body_lin_vel_w, body_ang_vel_w, fps, num_frames)`` as float64 NumPy arrays
  in their original per-file ordering.

  Raises ValueError on any structural, finiteness, or quaternion problem.
  """
  with np.load(path, allow_pickle=False) as data:
    keys = set(data.keys())
    missing = set(_REQUIRED_KEYS) - keys
    if missing:
      raise ValueError(
        f"{path}: missing required NPZ keys: {sorted(missing)}. "
        f"Found keys: {sorted(keys)}"
      )

    fps_array = np.asarray(data["fps"])
    if fps_array.size != 1:
      raise ValueError(
        f"{path}: fps must be a scalar (size-1 array), got shape {fps_array.shape}"
      )
    if not np.isfinite(fps_array).all():
      raise ValueError(f"{path}: fps contains non-finite values")
    fps = float(fps_array.item())
    if fps <= 0.0:
      raise ValueError(f"{path}: fps must be positive, got {fps}")

    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    joint_vel = np.asarray(data["joint_vel"], dtype=np.float64)
    body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float64)
    body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float64)
    body_lin_vel_w = np.asarray(data["body_lin_vel_w"], dtype=np.float64)
    body_ang_vel_w = np.asarray(data["body_ang_vel_w"], dtype=np.float64)

  # Scalar joint_pos (0-d) would produce IndexError on shape[0]; catch early.
  if joint_pos.ndim < 2:
    raise ValueError(
      f"{path}: joint_pos must be 2-D (frames, joints), got {joint_pos.ndim}-D "
      f"with shape {joint_pos.shape}"
    )

  num_frames = joint_pos.shape[0]
  if num_frames == 0:
    raise ValueError(f"{path}: joint_pos has 0 frames (empty file)")

  _validate_shapes(
    path,
    joint_pos,
    joint_vel,
    body_pos_w,
    body_quat_w,
    body_lin_vel_w,
    body_ang_vel_w,
    num_frames,
  )
  _validate_finite(
    path, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w
  )
  _validate_quaternions(path, body_quat_w)

  return (
    joint_pos,
    joint_vel,
    body_pos_w,
    body_quat_w,
    body_lin_vel_w,
    body_ang_vel_w,
    fps,
    num_frames,
  )


def _validate_shapes(
  path: Path,
  joint_pos: np.ndarray,
  joint_vel: np.ndarray,
  body_pos_w: np.ndarray,
  body_quat_w: np.ndarray,
  body_lin_vel_w: np.ndarray,
  body_ang_vel_w: np.ndarray,
  num_frames: int,
) -> None:
  """Verify per-file array shapes are consistent."""
  if joint_pos.shape != (num_frames, _NUM_JOINTS):
    raise ValueError(
      f"{path}: joint_pos shape {joint_pos.shape} != ({num_frames}, {_NUM_JOINTS})"
    )
  if joint_vel.shape != (num_frames, _NUM_JOINTS):
    raise ValueError(
      f"{path}: joint_vel shape {joint_vel.shape} != ({num_frames}, {_NUM_JOINTS})"
    )
  if body_pos_w.shape != (num_frames, _NUM_BODIES, 3):
    raise ValueError(
      f"{path}: body_pos_w shape {body_pos_w.shape} != ({num_frames}, {_NUM_BODIES}, 3)"
    )
  if body_quat_w.shape != (num_frames, _NUM_BODIES, 4):
    raise ValueError(
      f"{path}: body_quat_w shape {body_quat_w.shape} != ({num_frames}, {_NUM_BODIES}, 4)"
    )
  if body_lin_vel_w.shape != (num_frames, _NUM_BODIES, 3):
    raise ValueError(
      f"{path}: body_lin_vel_w shape {body_lin_vel_w.shape} != ({num_frames}, {_NUM_BODIES}, 3)"
    )
  if body_ang_vel_w.shape != (num_frames, _NUM_BODIES, 3):
    raise ValueError(
      f"{path}: body_ang_vel_w shape {body_ang_vel_w.shape} != ({num_frames}, {_NUM_BODIES}, 3)"
    )


def _validate_finite(
  path: Path,
  joint_pos: np.ndarray,
  joint_vel: np.ndarray,
  body_pos_w: np.ndarray,
  body_quat_w: np.ndarray,
  body_lin_vel_w: np.ndarray,
  body_ang_vel_w: np.ndarray,
) -> None:
  """Verify all values are finite in float64 (catches float32 overflow)."""
  for name, arr in [
    ("joint_pos", joint_pos),
    ("joint_vel", joint_vel),
    ("body_pos_w", body_pos_w),
    ("body_quat_w", body_quat_w),
    ("body_lin_vel_w", body_lin_vel_w),
    ("body_ang_vel_w", body_ang_vel_w),
  ]:
    if not np.isfinite(arr).all():
      bad = np.argwhere(~np.isfinite(arr))
      raise ValueError(
        f"{path}: {name} contains {len(bad)} non-finite values; "
        f"first at index {tuple(bad[0])}"
      )


def _validate_quaternions(path: Path, body_quat_w: np.ndarray) -> None:
  """Verify all quaternions are unit quaternions (wxyz)."""
  norms = np.linalg.norm(body_quat_w, axis=-1)  # (num_frames, num_bodies)
  bad_mask = ~np.isclose(norms, 1.0, atol=1e-5)
  if bad_mask.any():
    bad_idx = np.argwhere(bad_mask)[0]
    f_idx, b_idx = int(bad_idx[0]), int(bad_idx[1])
    raise ValueError(
      f"{path}: degenerate quaternion at frame {f_idx}, body {b_idx}: "
      f"norm={norms[f_idx, b_idx]:.6f}, value={body_quat_w[f_idx, b_idx]}"
    )


@dataclass(frozen=True)
class EndpointPool:
  """Immutable pool of validated tennis-end recovery states.

  Construct via :meth:`from_directory`. All arrays are read-only float32
  NumPy arrays in the stable trajectory-then-chronological row order.

  Attributes:
      joint_pos: ``[N, 31]`` joint positions (rad), X2 model order.
      joint_vel: ``[N, 31]`` joint velocities (rad/s), same order.
      root_pos_w: ``[N, 3]`` pelvis/root world position (m).
      root_quat_w: ``[N, 4]`` root unit quaternion (wxyz).
      root_lin_vel_w: ``[N, 3]`` root-link-origin world linear velocity (m/s).
          Matches MuJoCo free-joint ``qvel[0:3]``.
      root_ang_vel_w: ``[N, 3]`` root **world-frame** angular velocity
          (rad/s). To write to MuJoCo qvel, convert to body-local via
          ``R_root.T @ world_omega`` or use ``write_root_velocity``.
      trajectory_ids: ``[N]`` int64 stable trajectory IDs.
      frame_indices: ``[N]`` int64 original frame indices within each trajectory.
      joint_names: tuple of 31 verified model-order joint names.
      body_names: tuple of 32 verified model-order body names (index 0 = pelvis).
      manifest: JSON-serializable dict with full source inventory, hashes,
          split, seed, FPS, and ordering fingerprint.
  """

  joint_pos: np.ndarray
  joint_vel: np.ndarray
  root_pos_w: np.ndarray
  root_quat_w: np.ndarray
  root_lin_vel_w: np.ndarray
  root_ang_vel_w: np.ndarray
  trajectory_ids: np.ndarray
  frame_indices: np.ndarray
  joint_names: tuple[str, ...]
  body_names: tuple[str, ...]
  manifest: dict[str, Any] = field(default_factory=dict)

  def __len__(self) -> int:
    return self.joint_pos.shape[0]

  def __post_init__(self) -> None:
    n = self.joint_pos.shape[0]
    for attr_name in (
      "joint_pos",
      "joint_vel",
      "root_pos_w",
      "root_quat_w",
      "root_lin_vel_w",
      "root_ang_vel_w",
      "trajectory_ids",
      "frame_indices",
    ):
      arr = object.__getattribute__(self, attr_name)
      if arr.shape[0] != n:
        raise ValueError(f"{attr_name} has {arr.shape[0]} rows, expected {n}")
      arr.setflags(write=False)

  @classmethod
  def from_directory(
    cls,
    directory: str | Path,
    *,
    last_n_frames: int = 10,
    split: str = "train",
    validation_fraction: float = 0.2,
    seed: int = 42,
  ) -> EndpointPool:
    """Load and validate tennis endpoint states from a directory of NPZ files.

    All source files are loaded and validated once (full inventory with
    hashes), then the selected split's endpoint rows are materialized. This
    ensures common-FPS validation and complete source identity in every
    split's manifest, including held-out files.

    Args:
        directory: Path to a directory containing ``*.npz`` motion files.
        last_n_frames: Number of trailing frames to extract from each
            trajectory. Must be a positive integer. Trajectories shorter than
            this raise ``ValueError`` rather than silently taking fewer rows.
        split: ``"train"``, ``"validation"``, or ``"all"``.
        validation_fraction: Fraction of trajectories reserved for
            validation (0–1, exclusive of 0 and 1).
        seed: Seed for the deterministic trajectory-level split.

    Returns:
        A validated, immutable :class:`EndpointPool`.

    Raises:
        ValueError: On empty directory, missing/extra keys, wrong shapes,
            non-finite values, degenerate quaternions, incompatible FPS,
            trajectories shorter than ``last_n_frames``, invalid split,
            or an empty selected split (e.g. one-file train).
    """
    if not isinstance(last_n_frames, int) or isinstance(last_n_frames, bool):
      raise ValueError(
        f"last_n_frames must be an int, got {type(last_n_frames).__name__}"
      )
    if last_n_frames <= 0:
      raise ValueError(f"last_n_frames must be positive, got {last_n_frames}")
    if not (0.0 < validation_fraction < 1.0):
      raise ValueError(
        f"validation_fraction must be in (0, 1), got {validation_fraction}"
      )
    if split not in ("train", "validation", "all"):
      raise ValueError(f"split must be 'train', 'validation', or 'all', got {split!r}")

    dir_path = Path(directory)
    if not dir_path.is_dir():
      raise ValueError(f"Directory does not exist: {dir_path}")

    files = sorted(dir_path.glob("*.npz"))
    if not files:
      raise ValueError(f"No .npz files found in {dir_path}")

    # Resolve model ordering fingerprint for the manifest (exact model order,
    # not caller-provided names).
    joint_names, body_names = _resolve_model_names()

    # Deterministic trajectory-level train/validation split.
    train_ids, val_ids = _trajectory_split(
      num_trajectories=len(files),
      validation_fraction=validation_fraction,
      seed=seed,
    )
    if split == "train":
      selected_indices = train_ids
    elif split == "validation":
      selected_indices = val_ids
    else:
      selected_indices = list(range(len(files)))

    if not selected_indices:
      raise ValueError(
        f"Selected split '{split}' is empty (0 trajectories). "
        f"Use split='all' or reduce validation_fraction."
      )

    # Load and validate ALL source files once (full inventory + common FPS).
    file_data: list[dict[str, Any]] = []
    fps_values: set[float] = set()
    for file_path in files:
      (
        jp,
        jv,
        bpw,
        bqw,
        blw,
        baw,
        fps,
        num_frames,
      ) = _load_one_file(file_path)
      fps_values.add(fps)
      file_data.append(
        {
          "joint_pos": jp,
          "joint_vel": jv,
          "body_pos_w": bpw,
          "body_quat_w": bqw,
          "body_lin_vel_w": blw,
          "body_ang_vel_w": baw,
          "fps": fps,
          "num_frames": num_frames,
          "filename": file_path.name,
          "sha256": _sha256_file(file_path),
        }
      )

    if len(fps_values) > 1:
      raise ValueError(f"Incompatible FPS across files: {sorted(fps_values)}")
    fps_common = fps_values.pop() if fps_values else 0.0

    # Build full source inventory (all files, not just selected).
    all_sources: list[dict[str, Any]] = []
    for file_idx, fd in enumerate(file_data):
      nf = fd["num_frames"]
      start = max(0, nf - last_n_frames)
      all_sources.append(
        {
          "trajectory_id": file_idx,
          "filename": fd["filename"],
          "sha256": fd["sha256"],
          "num_frames": nf,
          "frame_start": start,
          "frame_end": nf - 1,
          "fps": fd["fps"],
        }
      )

    # Materialize selected endpoint rows.
    all_joint_pos: list[np.ndarray] = []
    all_joint_vel: list[np.ndarray] = []
    all_root_pos: list[np.ndarray] = []
    all_root_quat: list[np.ndarray] = []
    all_root_lin_vel: list[np.ndarray] = []
    all_root_ang_vel: list[np.ndarray] = []
    all_traj_ids: list[np.ndarray] = []
    all_frame_indices: list[np.ndarray] = []

    for file_idx in selected_indices:
      fd = file_data[file_idx]
      num_frames = fd["num_frames"]
      if num_frames < last_n_frames:
        raise ValueError(
          f"{fd['filename']}: has {num_frames} frames, need at least "
          f"{last_n_frames} (last_n_frames). Refusing to silently "
          f"take a smaller window."
        )
      start = num_frames - last_n_frames
      frame_sl = slice(start, num_frames)  # chronological within window

      # Convert to float32; overflow (finite float64 -> inf float32) is
      # caught by the post-conversion finiteness check below.
      with np.errstate(over="ignore"):
        all_joint_pos.append(fd["joint_pos"][frame_sl].astype(np.float32))
        all_joint_vel.append(fd["joint_vel"][frame_sl].astype(np.float32))
        # Body index 0 = pelvis/root.
      all_root_pos.append(fd["body_pos_w"][frame_sl, 0].astype(np.float32))
      all_root_quat.append(fd["body_quat_w"][frame_sl, 0].astype(np.float32))
      all_root_lin_vel.append(fd["body_lin_vel_w"][frame_sl, 0].astype(np.float32))
      all_root_ang_vel.append(fd["body_ang_vel_w"][frame_sl, 0].astype(np.float32))

      all_traj_ids.append(np.full(last_n_frames, file_idx, dtype=np.int64))
      all_frame_indices.append(np.arange(start, num_frames, dtype=np.int64))

    # Validate quaternion norms on concatenated arrays (post-float32).
    root_quat_all = np.concatenate(all_root_quat, axis=0)
    norms = np.linalg.norm(root_quat_all, axis=-1)
    if not np.allclose(norms, 1.0, atol=1e-5):
      bad = np.argwhere(~np.isclose(norms, 1.0, atol=1e-5))[0]
      raise ValueError(
        f"Degenerate root quaternion at row {bad[0]}: norm={norms[bad[0]]:.6f}"
      )

    # Validate finiteness on concatenated float32 arrays (catches overflow).
    for name, arr in [
      ("joint_pos", np.concatenate(all_joint_pos, axis=0)),
      ("joint_vel", np.concatenate(all_joint_vel, axis=0)),
      ("root_pos_w", np.concatenate(all_root_pos, axis=0)),
      ("root_lin_vel_w", np.concatenate(all_root_lin_vel, axis=0)),
      ("root_ang_vel_w", np.concatenate(all_root_ang_vel, axis=0)),
    ]:
      if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values after float32 conversion")

    manifest = {
      "directory": str(dir_path),
      "num_files_total": len(files),
      "num_files_selected": len(selected_indices),
      "split": split,
      "validation_fraction": validation_fraction,
      "seed": seed,
      "last_n_frames": last_n_frames,
      "num_states": sum(len(t) for t in all_traj_ids),
      "fps": fps_common,
      "train_trajectory_ids": sorted(train_ids),
      "validation_trajectory_ids": sorted(val_ids),
      "selected_trajectory_ids": sorted(selected_indices),
      "joint_names": list(joint_names),
      "body_names": list(body_names),
      "joint_order_fingerprint": hashlib.sha256(
        "|".join(joint_names).encode()
      ).hexdigest(),
      "body_order_fingerprint": hashlib.sha256(
        "|".join(body_names).encode()
      ).hexdigest(),
      "sources": all_sources,
    }

    return cls(
      joint_pos=np.concatenate(all_joint_pos, axis=0),
      joint_vel=np.concatenate(all_joint_vel, axis=0),
      root_pos_w=np.concatenate(all_root_pos, axis=0),
      root_quat_w=np.concatenate(all_root_quat, axis=0),
      root_lin_vel_w=np.concatenate(all_root_lin_vel, axis=0),
      root_ang_vel_w=np.concatenate(all_root_ang_vel, axis=0),
      trajectory_ids=np.concatenate(all_traj_ids, axis=0),
      frame_indices=np.concatenate(all_frame_indices, axis=0),
      joint_names=joint_names,
      body_names=body_names,
      manifest=manifest,
    )

  def sample_indices(self, count: int, rng: np.random.Generator) -> np.ndarray:
    """Sample ``count`` row indices uniformly with replacement.

    Since every trajectory contributes exactly ``last_n_frames`` rows,
    uniform row sampling is exactly equivalent to uniform trajectory
    followed by uniform frame.

    Args:
        count: Number of indices to sample.
        rng: A :class:`numpy.random.Generator`.

    Returns:
        ``int64`` array of shape ``[count]``.
    """
    if not isinstance(count, int) or isinstance(count, bool):
      raise ValueError(f"count must be an int, got {type(count).__name__}")
    if count < 0:
      raise ValueError(f"count must be non-negative, got {count}")
    if len(self) == 0:
      raise ValueError("Cannot sample from an empty pool")
    return rng.integers(0, len(self), size=count, dtype=np.int64)


def _trajectory_split(
  num_trajectories: int,
  validation_fraction: float,
  seed: int,
) -> tuple[list[int], list[int]]:
  """Deterministic trajectory-level train/validation split.

  Uses a seeded RNG to shuffle trajectory indices, then assigns the first
  ``ceil(fraction * N)`` to validation and the rest to training. With 144
  trajectories and fraction 0.2, this gives 29 validation and 115 training.

  No trajectory's frames appear on both sides (whole-trajectory holdout).
  """
  rng = np.random.default_rng(seed)
  perm = rng.permutation(num_trajectories)
  num_val = math.ceil(validation_fraction * num_trajectories)
  val_ids = sorted(int(x) for x in perm[:num_val])
  train_ids = sorted(int(x) for x in perm[num_val:])
  return train_ids, val_ids


def _resolve_model_names() -> tuple[tuple[str, ...], tuple[str, ...]]:
  """Resolve X2 model joint/body names (excluding free joint/world).

  Imported lazily so module import does not require the X2 asset. The
  returned names are the exact model order — the ordering contract, not a
  caller-provided fingerprint.
  """
  from mjlab.asset_zoo.robots.agibot_x2.x2_constants import get_spec

  spec = get_spec()
  model = spec.compile()
  joint_names = tuple(model.joint(i).name for i in range(1, model.njnt))
  body_names = tuple(model.body(i).name for i in range(1, model.nbody))
  return joint_names, body_names
