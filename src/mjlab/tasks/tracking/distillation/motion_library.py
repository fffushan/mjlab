"""Immutable clip-local reference library for multi-motion distillation.

Scope: load each selected reference motion once into owned float32 tensors,
gather ``(motion_id, local_frame)`` pairs without ever crossing a clip
boundary, and record the ordered clip mapping that cohort identity must
persist.  No simulator, command, teacher, or model is built here.

Body references follow the source clip's body axis, which is the compiled
robot body order (``Entity.body_names``, excluding the world body).  The
tracked subset is resolved by the caller against the compiled robot and the
teacher's export, then recorded on the library as a :class:`BodySelection`;
physical body names are never inferred from tensor widths.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mjlab.tasks.tracking.distillation.config import sha256_file

_JOINT_ARRAYS = ("joint_pos", "joint_vel")
_BODY_ARRAYS = (
  ("body_pos_w", 3),
  ("body_quat_w", 4),
  ("body_lin_vel_w", 3),
  ("body_ang_vel_w", 3),
)
_INTEGER_DTYPES = (torch.int64, torch.int32, torch.int16, torch.int8)
_FPS_RTOL = 1e-4
"""Relative tolerance accepted between a clip's fps and a teacher's saved fps."""


class MotionLibraryError(ValueError):
  """A clip definition or a reference query violates the library contract."""


@dataclass(frozen=True, slots=True)
class MotionClipSpec:
  """One clip declaration, loaded in the order it appears here.

  ``motion_id`` is the clip's position in that order: it is the numeric clip
  identity stored per environment row and persisted as cohort identity.
  ``teacher_code`` is the frozen teacher's label code and defaults to the same
  position.  ``expected_frames``/``expected_fps`` carry the teacher's declared
  reference extent and are validated when given, so a clip that does not match
  its teacher never loads silently.
  """

  teacher_id: str
  motion_file: Path | str
  teacher_code: int | None = None
  expected_frames: int | None = None
  expected_fps: float | None = None


@dataclass(frozen=True, slots=True)
class MotionClip:
  """One loaded clip: identity, source digest, extent, and library placement."""

  motion_id: int
  teacher_id: str
  teacher_code: int
  motion_file: Path
  source_hash: str
  frames: int
  fps: float
  storage_offset: int
  source_body_count: int
  body_indices: tuple[int, ...] | None = None
  """Source body indexes of the tracked subset, once it is resolved."""


@dataclass(frozen=True, slots=True)
class BodySelection:
  """Tracked-body mapping shared by every clip of one library."""

  indices: tuple[int, ...]
  source_body_count: int
  names: tuple[str, ...] | None = None

  def __post_init__(self) -> None:
    if not self.indices:
      raise MotionLibraryError("body selection needs at least one body index")
    if len(set(self.indices)) != len(self.indices):
      raise MotionLibraryError(f"body selection has duplicate indexes {self.indices}")
    for index in self.indices:
      if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise MotionLibraryError(f"invalid body index {index!r} in body selection")
      if index >= self.source_body_count:
        raise MotionLibraryError(
          f"body index {index} is outside the {self.source_body_count} source bodies"
        )
    if self.names is not None and len(self.names) != len(self.indices):
      raise MotionLibraryError(
        f"body selection names {len(self.names)} do not align with "
        f"{len(self.indices)} indexes"
      )


class MotionLibrary:
  """Owned reference clips with validated, clip-local vectorized gathers.

  Every gather validates ``(motion_ids, local_frames)`` before offsetting into
  flat storage, so a frame index can never read another clip, and no query
  converts between clip rates or stores into the source arrays.  Returned
  tensors are fresh copies: mutating them cannot change the library.
  """

  __slots__ = (
    "_body_ang_vel_w",
    "_body_lin_vel_w",
    "_body_pos_w",
    "_body_quat_w",
    "_clips",
    "_device",
    "_dtype",
    "_frame_counts",
    "_joint_pos",
    "_joint_vel",
    "_offsets",
    "_selection",
    "_selection_index",
    "_teacher_codes",
  )

  def __init__(
    self,
    *,
    clips: tuple[MotionClip, ...],
    device: torch.device,
    dtype: torch.dtype,
    offsets: torch.Tensor,
    frame_counts: torch.Tensor,
    teacher_codes: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    body_lin_vel_w: torch.Tensor,
    body_ang_vel_w: torch.Tensor,
    selection: BodySelection | None = None,
    selection_index: torch.Tensor | None = None,
  ) -> None:
    """Build from validated parts; use :meth:`from_clips` instead."""
    self._clips = clips
    # Device aliases (cpu:0, cuda) resolve when tensors are allocated. Compare
    # future indices against the actual allocation, not the requested spelling.
    self._device = joint_pos.device
    self._dtype = dtype
    self._offsets = offsets
    self._frame_counts = frame_counts
    self._teacher_codes = teacher_codes
    self._joint_pos = joint_pos
    self._joint_vel = joint_vel
    self._body_pos_w = body_pos_w
    self._body_quat_w = body_quat_w
    self._body_lin_vel_w = body_lin_vel_w
    self._body_ang_vel_w = body_ang_vel_w
    self._selection = selection
    self._selection_index = selection_index

  @classmethod
  def from_clips(
    cls,
    specs: Sequence[MotionClipSpec],
    *,
    device: str | torch.device = "cpu",
  ) -> MotionLibrary:
    """Load one clip per spec, in the given order, and validate compatibility.

    Raises:
      MotionLibraryError: if a clip is missing, malformed, non-finite, empty,
        incompatible with the other clips, or disagrees with the extent or fps
        its spec declares.
    """
    if not specs:
      raise MotionLibraryError("a motion library needs at least one clip")
    target = torch.device(device)
    arrays: dict[str, list[torch.Tensor]] = {
      name: [] for name in (*_JOINT_ARRAYS, *(name for name, _ in _BODY_ARRAYS))
    }
    clips: list[MotionClip] = []
    teacher_ids: set[str] = set()
    teacher_codes: set[int] = set()
    joint_dim: int | None = None
    body_count: int | None = None
    offset = 0
    for position, spec in enumerate(specs):
      if not isinstance(spec.teacher_id, str) or not spec.teacher_id:
        raise MotionLibraryError(f"clip {position} needs a non-empty teacher id")
      if spec.teacher_id in teacher_ids:
        raise MotionLibraryError(f"duplicate teacher id {spec.teacher_id!r}")
      teacher_ids.add(spec.teacher_id)
      teacher_code = position if spec.teacher_code is None else spec.teacher_code
      if (
        isinstance(teacher_code, bool)
        or not isinstance(teacher_code, int)
        or teacher_code < 0
      ):
        raise MotionLibraryError(
          f"clip {position} needs a non-negative integer teacher code"
        )
      if teacher_code in teacher_codes:
        raise MotionLibraryError(f"duplicate teacher code {teacher_code}")
      teacher_codes.add(teacher_code)

      loaded, frames, fps = _load_clip(spec, position)
      joint_pos = loaded["joint_pos"]
      if joint_pos.ndim != 2:
        raise MotionLibraryError(
          f"clip {position} joint_pos must be [frames, joints], got "
          f"{tuple(joint_pos.shape)}"
        )
      if loaded["joint_vel"].shape != joint_pos.shape:
        raise MotionLibraryError(
          f"clip {position} joint_vel shape {tuple(loaded['joint_vel'].shape)} "
          f"disagrees with joint_pos {tuple(joint_pos.shape)}"
        )
      clip_joint_dim = int(joint_pos.shape[1])
      if clip_joint_dim < 1:
        raise MotionLibraryError(f"clip {position} has no joints")
      body_pos_w = loaded["body_pos_w"]
      if body_pos_w.ndim != 3:
        raise MotionLibraryError(
          f"clip {position} body_pos_w must be [frames, bodies, 3], got "
          f"{tuple(body_pos_w.shape)}"
        )
      clip_body_count = int(body_pos_w.shape[1])
      if clip_body_count < 1:
        raise MotionLibraryError(f"clip {position} has no bodies")
      for name, width in _BODY_ARRAYS:
        expected = (frames, clip_body_count, width)
        if tuple(loaded[name].shape) != expected:
          raise MotionLibraryError(
            f"clip {position} {name} shape {tuple(loaded[name].shape)} is not "
            f"{expected}"
          )
      if joint_dim is None:
        joint_dim, body_count = clip_joint_dim, clip_body_count
      elif clip_joint_dim != joint_dim or clip_body_count != body_count:
        raise MotionLibraryError(
          f"clip {position} carries {clip_joint_dim} joints and "
          f"{clip_body_count} bodies; earlier clips carry {joint_dim} joints and "
          f"{body_count} bodies"
        )

      for name, values in loaded.items():
        arrays[name].append(values)
      clips.append(
        MotionClip(
          motion_id=position,
          teacher_id=spec.teacher_id,
          teacher_code=teacher_code,
          motion_file=Path(spec.motion_file),
          source_hash=sha256_file(Path(spec.motion_file)),
          frames=frames,
          fps=fps,
          storage_offset=offset,
          source_body_count=clip_body_count,
        )
      )
      offset += frames

    flat = {
      name: torch.cat(values, dim=0).to(target) for name, values in arrays.items()
    }
    return cls(
      clips=tuple(clips),
      device=target,
      dtype=flat["joint_pos"].dtype,
      offsets=torch.tensor(
        [clip.storage_offset for clip in clips], dtype=torch.long, device=target
      ),
      frame_counts=torch.tensor(
        [clip.frames for clip in clips], dtype=torch.long, device=target
      ),
      teacher_codes=torch.tensor(
        [clip.teacher_code for clip in clips], dtype=torch.long, device=target
      ),
      joint_pos=flat["joint_pos"],
      joint_vel=flat["joint_vel"],
      body_pos_w=flat["body_pos_w"],
      body_quat_w=flat["body_quat_w"],
      body_lin_vel_w=flat["body_lin_vel_w"],
      body_ang_vel_w=flat["body_ang_vel_w"],
    )

  # Identity.

  @property
  def clips(self) -> tuple[MotionClip, ...]:
    """Ordered clip entries; ``motion_id`` is the entry's position."""
    return self._clips

  @property
  def num_motions(self) -> int:
    return len(self._clips)

  @property
  def total_frames(self) -> int:
    """Frames across all clips, i.e. the extent of the flat storage."""
    return int(self._joint_pos.shape[0])

  @property
  def joint_dim(self) -> int:
    return int(self._joint_pos.shape[1])

  @property
  def source_body_count(self) -> int:
    """Bodies per clip on the source body axis, before any selection."""
    return int(self._body_pos_w.shape[1])

  @property
  def device(self) -> torch.device:
    return self._device

  @property
  def dtype(self) -> torch.dtype:
    return self._dtype

  @property
  def body_selection(self) -> BodySelection | None:
    """Resolved tracked-body selection, or ``None`` before it is resolved."""
    return self._selection

  def clip(self, motion_id: int) -> MotionClip:
    """Return the clip entry for ``motion_id``."""
    if isinstance(motion_id, bool) or not isinstance(motion_id, int):
      raise MotionLibraryError(f"motion id must be an integer, got {motion_id!r}")
    if not 0 <= motion_id < self.num_motions:
      raise MotionLibraryError(
        f"motion id {motion_id} is outside the {self.num_motions} loaded clips"
      )
    return self._clips[motion_id]

  def ordered_mapping(self) -> tuple[MotionClip, ...]:
    """Return the ordered clip mapping that cohort identity must persist."""
    return self._clips

  def mapping_digest(self) -> str:
    """Digest of the ordered clip identity that strict resume must reproduce.

    Covers motion id, teacher id/code, source digest, frame count, and fps of
    every clip in order.  Local file paths and the robot-derived body selection
    are excluded: relocated assets are accepted by content digest, and the body
    mapping is audited against the compiled robot instead.
    """
    payload = json.dumps(
      [
        {
          "motion_id": clip.motion_id,
          "teacher_id": clip.teacher_id,
          "teacher_code": clip.teacher_code,
          "source_hash": clip.source_hash,
          "frames": clip.frames,
          "fps": clip.fps,
        }
        for clip in self._clips
      ],
      sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

  # Body mapping.

  def with_body_selection(self, selection: BodySelection) -> MotionLibrary:
    """Return a library whose body queries use ``selection``.

    The owned reference tensors are shared, not copied, and stay immutable.
    """
    if not isinstance(selection, BodySelection):
      raise MotionLibraryError("with_body_selection requires a BodySelection")
    if selection.source_body_count != self.source_body_count:
      raise MotionLibraryError(
        f"body selection was resolved against {selection.source_body_count} source "
        f"bodies but the clips carry {self.source_body_count}"
      )
    derived = MotionLibrary(
      clips=tuple(
        MotionClip(
          motion_id=clip.motion_id,
          teacher_id=clip.teacher_id,
          teacher_code=clip.teacher_code,
          motion_file=clip.motion_file,
          source_hash=clip.source_hash,
          frames=clip.frames,
          fps=clip.fps,
          storage_offset=clip.storage_offset,
          source_body_count=clip.source_body_count,
          body_indices=selection.indices,
        )
        for clip in self._clips
      ),
      device=self._device,
      dtype=self._dtype,
      offsets=self._offsets,
      frame_counts=self._frame_counts,
      teacher_codes=self._teacher_codes,
      joint_pos=self._joint_pos,
      joint_vel=self._joint_vel,
      body_pos_w=self._body_pos_w,
      body_quat_w=self._body_quat_w,
      body_lin_vel_w=self._body_lin_vel_w,
      body_ang_vel_w=self._body_ang_vel_w,
      selection=selection,
      selection_index=torch.tensor(
        selection.indices, dtype=torch.long, device=self._device
      ),
    )
    return derived

  def compare_body_references(
    self, motion_id: int, expected: Mapping[str, np.ndarray]
  ) -> dict[str, float]:
    """Max absolute error between a clip's tracked bodies and an export.

    ``expected`` holds one array per body reference name (the teacher export's
    embedded tensors, already restricted to the tracked bodies in tracked
    order).  The caller applies the cohort tolerance; a non-zero error means
    the resolved body mapping disagrees with the teacher that trained on it.
    """
    clip = self.clip(motion_id)
    if self._selection_index is None:
      raise MotionLibraryError(
        "body references need a resolved body selection; call with_body_selection"
      )
    errors: dict[str, float] = {}
    for name, _ in _BODY_ARRAYS:
      if name not in expected:
        raise MotionLibraryError(f"expected body references have no {name!r} array")
      reference = self._flat_body(name)[
        clip.storage_offset : clip.storage_offset + clip.frames
      ]
      selected = (
        reference[:, self._selection_index].detach().to(torch.float64).cpu().numpy()
      )
      exported = np.asarray(expected[name], dtype=np.float64)
      if exported.shape != selected.shape:
        raise MotionLibraryError(
          f"clip {motion_id} {name} tracked shape {selected.shape} disagrees with "
          f"the export shape {exported.shape}"
        )
      if selected.size and not np.isfinite(exported).all():
        raise MotionLibraryError(f"export {name!r} array is not finite")
      errors[name] = float(np.abs(selected - exported).max()) if selected.size else 0.0
    return errors

  # Per-row metadata.

  def frame_counts_for(self, motion_ids: torch.Tensor) -> torch.Tensor:
    """Per-row clip length in frames, as a fresh ``int64`` tensor."""
    return self._frame_counts[self._index_tensor(motion_ids, "motion_ids")]

  def teacher_codes_for(self, motion_ids: torch.Tensor) -> torch.Tensor:
    """Per-row teacher label code, as a fresh ``int64`` tensor."""
    return self._teacher_codes[self._index_tensor(motion_ids, "motion_ids")]

  def clamp_local_frames(
    self, motion_ids: torch.Tensor, local_frames: torch.Tensor
  ) -> torch.Tensor:
    """Clamp frames to the last frame of each row's own clip.

    Endpoint lookahead uses this so a preview never reads the next clip.
    """
    ids = self._index_tensor(motion_ids, "motion_ids")
    frames = self._index_tensor(local_frames, "local_frames")
    self._require_same_shape(ids, frames)
    if frames.numel() and bool((frames < 0).any()):
      raise MotionLibraryError("local_frames must be non-negative")
    return torch.minimum(frames, self._frame_counts[ids] - 1)

  # Reference gathers.

  def joint_pos(
    self, motion_ids: torch.Tensor, local_frames: torch.Tensor
  ) -> torch.Tensor:
    """Reference joint positions ``[B, joints]`` at ``(motion_id, local_frame)``."""
    return self._joint_pos[self.flat_index(motion_ids, local_frames)]

  def joint_vel(
    self, motion_ids: torch.Tensor, local_frames: torch.Tensor
  ) -> torch.Tensor:
    """Reference joint velocities ``[B, joints]`` at ``(motion_id, local_frame)``."""
    return self._joint_vel[self.flat_index(motion_ids, local_frames)]

  def body_pos_w(
    self, motion_ids: torch.Tensor, local_frames: torch.Tensor
  ) -> torch.Tensor:
    """Tracked body positions ``[B, bodies, 3]`` in world coordinates."""
    return self._select(self._body_pos_w, motion_ids, local_frames)

  def body_quat_w(
    self, motion_ids: torch.Tensor, local_frames: torch.Tensor
  ) -> torch.Tensor:
    """Tracked body orientations ``[B, bodies, 4]`` (MuJoCo ``wxyz``)."""
    return self._select(self._body_quat_w, motion_ids, local_frames)

  def body_lin_vel_w(
    self, motion_ids: torch.Tensor, local_frames: torch.Tensor
  ) -> torch.Tensor:
    """Tracked body linear velocities ``[B, bodies, 3]`` in world coordinates."""
    return self._select(self._body_lin_vel_w, motion_ids, local_frames)

  def body_ang_vel_w(
    self, motion_ids: torch.Tensor, local_frames: torch.Tensor
  ) -> torch.Tensor:
    """Tracked body angular velocities ``[B, bodies, 3]`` in world coordinates."""
    return self._select(self._body_ang_vel_w, motion_ids, local_frames)

  def flat_index(
    self, motion_ids: torch.Tensor, local_frames: torch.Tensor
  ) -> torch.Tensor:
    """Validated flat storage index for each ``(motion_id, local_frame)`` row.

    Ids and frames are checked against the clip table *before* any offset is
    added, so a negative frame or an over-long frame raises instead of reading
    a neighbouring clip.
    """
    ids = self._index_tensor(motion_ids, "motion_ids")
    frames = self._index_tensor(local_frames, "local_frames")
    self._require_same_shape(ids, frames)
    if ids.numel():
      if bool((ids < 0).any()) or bool((ids >= self.num_motions).any()):
        raise MotionLibraryError(
          f"motion_ids must be in [0, {self.num_motions}); ids "
          f"[{int(ids.min())}, {int(ids.max())}] were requested"
        )
      counts = self._frame_counts[ids]
      if bool((frames < 0).any()):
        raise MotionLibraryError("local_frames must be non-negative")
      if bool((frames >= counts).any()):
        offending = int(frames[frames >= counts][0])
        raise MotionLibraryError(
          f"local frame {offending} is outside its clip; clip lengths are "
          "per-row and frames are never padded or carried into the next clip"
        )
    return self._offsets[ids] + frames

  def __repr__(self) -> str:
    clips = ", ".join(
      f"{clip.motion_id}:{clip.teacher_id}({clip.frames}f@{clip.fps:g}Hz)"
      for clip in self._clips
    )
    return (
      f"MotionLibrary([{clips}], bodies={self.source_body_count}, "
      f"selection={None if self._selection is None else self._selection.indices}, "
      f"device={self._device})"
    )

  # Internals.

  def _select(
    self, flat: torch.Tensor, motion_ids: torch.Tensor, local_frames: torch.Tensor
  ) -> torch.Tensor:
    if self._selection_index is None:
      raise MotionLibraryError(
        "body references need a resolved body selection; call with_body_selection"
      )
    gathered = flat[self.flat_index(motion_ids, local_frames)]
    return gathered[:, self._selection_index]

  def _flat_body(self, name: str) -> torch.Tensor:
    if name == "body_pos_w":
      return self._body_pos_w
    if name == "body_quat_w":
      return self._body_quat_w
    if name == "body_lin_vel_w":
      return self._body_lin_vel_w
    assert name == "body_ang_vel_w"
    return self._body_ang_vel_w

  def _require_same_shape(self, ids: torch.Tensor, frames: torch.Tensor) -> None:
    if ids.shape != frames.shape:
      raise MotionLibraryError(
        f"motion_ids shape {tuple(ids.shape)} disagrees with local_frames shape "
        f"{tuple(frames.shape)}"
      )

  def _index_tensor(self, value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
      raise MotionLibraryError(f"{name} must be a torch.Tensor")
    if value.dtype not in _INTEGER_DTYPES:
      raise MotionLibraryError(f"{name} must be an integer tensor, got {value.dtype}")
    if value.ndim != 1:
      raise MotionLibraryError(f"{name} must be one-dimensional")
    if value.device != self._device:
      raise MotionLibraryError(
        f"{name} is on {value.device} but the library is on {self._device}"
      )
    return value.to(torch.int64)


def _load_clip(
  spec: MotionClipSpec, position: int
) -> tuple[dict[str, torch.Tensor], int, float]:
  """Load one clip's owned arrays, frame count, and rate."""
  path = Path(spec.motion_file)
  if not path.is_file():
    raise MotionLibraryError(f"clip {position} motion file not found: {path}")
  names = (*_JOINT_ARRAYS, *(name for name, _ in _BODY_ARRAYS))
  loaded: dict[str, torch.Tensor] = {}
  try:
    with np.load(path, allow_pickle=False) as data:
      available = set(data.files)
      missing = [name for name in names if name not in available]
      if missing:
        raise MotionLibraryError(
          f"clip {position} ({path}) is missing arrays {missing}"
        )
      if "fps" not in available:
        raise MotionLibraryError(f"clip {position} ({path}) has no fps array")
      fps = _load_fps(data["fps"], spec, position, path)
      for name in names:
        loaded[name] = _load_array(data[name], name, position, path)
  except MotionLibraryError:
    raise
  except (OSError, ValueError) as exc:
    raise MotionLibraryError(
      f"clip {position} ({path}) could not be loaded: {exc}"
    ) from exc

  frames = int(loaded["joint_pos"].shape[0])
  if frames < 1:
    raise MotionLibraryError(f"clip {position} ({path}) has no frames")
  for name, values in loaded.items():
    if int(values.shape[0]) != frames:
      raise MotionLibraryError(
        f"clip {position} ({path}) {name} carries {int(values.shape[0])} frames "
        f"but joint_pos carries {frames}"
      )
  if spec.expected_frames is not None and frames != spec.expected_frames:
    raise MotionLibraryError(
      f"clip {position} ({path}) has {frames} frames but the teacher declares "
      f"{spec.expected_frames}"
    )
  return loaded, frames, fps


def _load_fps(
  array: np.ndarray, spec: MotionClipSpec, position: int, path: Path
) -> float:
  values = np.asarray(array)
  if values.size != 1 or not np.issubdtype(values.dtype, np.number):
    raise MotionLibraryError(f"clip {position} ({path}) has an invalid fps array")
  fps = float(values.reshape(-1)[0])
  if not math.isfinite(fps) or fps <= 0.0:
    raise MotionLibraryError(f"clip {position} ({path}) has a non-positive fps")
  if spec.expected_fps is not None and not math.isclose(
    fps, float(spec.expected_fps), rel_tol=_FPS_RTOL
  ):
    raise MotionLibraryError(
      f"clip {position} ({path}) is {fps:g} Hz but the teacher declares "
      f"{float(spec.expected_fps):g} Hz"
    )
  return fps


def _load_array(
  array: np.ndarray, name: str, position: int, path: Path
) -> torch.Tensor:
  values = np.asarray(array)
  if not (
    np.issubdtype(values.dtype, np.floating) or np.issubdtype(values.dtype, np.integer)
  ):
    raise MotionLibraryError(f"clip {position} ({path}) {name} is not a numeric array")
  owned = np.ascontiguousarray(values, dtype=np.float32)
  if not np.isfinite(owned).all():
    raise MotionLibraryError(
      f"clip {position} ({path}) {name} contains non-finite values"
    )
  return torch.from_numpy(owned)


__all__ = [
  "BodySelection",
  "MotionClip",
  "MotionClipSpec",
  "MotionLibrary",
  "MotionLibraryError",
]
