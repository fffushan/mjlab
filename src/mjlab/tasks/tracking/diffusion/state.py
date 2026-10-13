"""World-frame X2 state construction used by D1 and later runtime code.

Two named packed layouts share the same window-center and per-timestep-root
frames.  :data:`HYBRID_STATE` is the frozen 135-D contract state, which keeps
the root position and root linear velocity of every step.
:data:`LOCAL_KINEMATIC_STATE` is the 129-D state that removes those two
translation fields, so a generated plan can move the body relative to the root
but cannot move the root in the world.  The frozen
:class:`~.contract.DiffusionContract` still declares the 135-D hybrid width; the
129-D layout is selected by its opt-in local-kinematic diffusion contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .contract import DEFAULT_CONTRACT


class StateValidationError(ValueError):
  """A world state is malformed or contains non-finite values."""


def _array(
  name: str,
  value: np.ndarray | Iterable[float],
  shape: tuple[int, ...],
  *,
  batched: bool = False,
) -> np.ndarray:
  result = np.asarray(value, dtype=np.float64)
  valid = result.shape == shape or (
    batched and result.ndim == len(shape) + 1 and result.shape[1:] == shape
  )
  if not valid:
    raise StateValidationError(f"{name} must have shape {shape}, got {result.shape}")
  if not np.isfinite(result).all():
    raise StateValidationError(f"{name} contains non-finite values")
  return result


_STATE_FIELDS = (
  "root_position",
  "root_quaternion_wxyz",
  "root_linear_velocity",
  "root_angular_velocity",
  "body_positions",
  "body_linear_velocities",
)


@dataclass(frozen=True, slots=True)
class StateRepresentation:
  """Named channel layout of one packed diffusion state vector.

  ``slices`` maps a field name to its half-open channel range.  The fields are
  contiguous and cover ``dimension`` exactly, so a packed vector is the
  concatenation of the named fields in mapping order.
  """

  name: str
  dimension: int
  slices: Mapping[str, slice]

  def __post_init__(self) -> None:
    if not self.name or self.dimension <= 0 or not self.slices:
      raise StateValidationError("state representation is incomplete")
    start = 0
    for field, item in self.slices.items():
      if item.start != start or item.stop <= item.start or item.step is not None:
        raise StateValidationError(
          f"state representation field {field!r} is not a contiguous slice"
        )
      start = item.stop
    if start != self.dimension:
      raise StateValidationError("state representation slices must cover the dimension")

  def field(self, name: str) -> slice:
    """Return the channel range of ``name``, or fail closed."""
    try:
      return self.slices[name]
    except KeyError as exc:
      raise StateValidationError(
        f"representation {self.name!r} has no {name!r} field"
      ) from exc


# The two root translation fields.  They are the only fields that the
# local-kinematic layout removes from the hybrid layout.
_TRANSLATION_FIELDS = ("root_position", "root_linear_velocity")

HYBRID_STATE = StateRepresentation(
  name="mjlab-x2-diffusion-state-hybrid-135-v1",
  dimension=DEFAULT_CONTRACT.state_dimension,
  slices={
    "root_position": slice(0, 3),
    "root_rot6d": slice(3, 9),
    "root_linear_velocity": slice(9, 12),
    "root_angular_velocity": slice(12, 15),
    "body_positions": slice(15, 75),
    "body_linear_velocities": slice(75, 135),
  },
)


def _without_fields(
  representation: StateRepresentation, removed: tuple[str, ...]
) -> dict[str, slice]:
  """Rebuild contiguous slices after removing fields, keeping field order."""
  slices: dict[str, slice] = {}
  start = 0
  for name, item in representation.slices.items():
    if name in removed:
      continue
    width = item.stop - item.start
    slices[name] = slice(start, start + width)
    start += width
  return slices


LOCAL_KINEMATIC_STATE = StateRepresentation(
  name="mjlab-x2-diffusion-state-local-kinematic-129-v1",
  dimension=HYBRID_STATE.dimension
  - sum(
    HYBRID_STATE.field(name).stop - HYBRID_STATE.field(name).start
    for name in _TRANSLATION_FIELDS
  ),
  slices=_without_fields(HYBRID_STATE, _TRANSLATION_FIELDS),
)

_LOCAL_KINEMATIC_FIELDS = tuple(LOCAL_KINEMATIC_STATE.slices)


def _quat_matrix(quaternion: np.ndarray) -> np.ndarray:
  q = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)
  w, x, y, z = np.moveaxis(q, -1, 0)
  return np.stack(
    (
      1 - 2 * (y * y + z * z),
      2 * (x * y - z * w),
      2 * (x * z + y * w),
      2 * (x * y + z * w),
      1 - 2 * (x * x + z * z),
      2 * (y * z - x * w),
      2 * (x * z - y * w),
      2 * (y * z + x * w),
      1 - 2 * (x * x + y * y),
    ),
    axis=-1,
  ).reshape(*q.shape[:-1], 3, 3)


def _yaw_matrix(yaw: np.ndarray) -> np.ndarray:
  c, s = np.cos(yaw), np.sin(yaw)
  return np.stack(
    (
      c,
      -s,
      np.zeros_like(c),
      s,
      c,
      np.zeros_like(c),
      np.zeros_like(c),
      np.zeros_like(c),
      np.ones_like(c),
    ),
    axis=-1,
  ).reshape(*yaw.shape, 3, 3)


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
  """Convert proper rotation matrices to wxyz quaternions, vectorized."""
  # The state uses rotations generated by the matching forward transform, so
  # this stable diagonal-branch implementation is sufficient and avoids scipy.
  out = np.empty((*matrix.shape[:-2], 4), dtype=np.float64)
  trace = np.trace(matrix, axis1=-2, axis2=-1)
  positive = trace > 0.0
  s = np.sqrt(np.maximum(trace[positive] + 1.0, 1e-15)) * 2
  out[positive, 0] = 0.25 * s
  out[positive, 1] = (matrix[positive, 2, 1] - matrix[positive, 1, 2]) / s
  out[positive, 2] = (matrix[positive, 0, 2] - matrix[positive, 2, 0]) / s
  out[positive, 3] = (matrix[positive, 1, 0] - matrix[positive, 0, 1]) / s
  for index, (a, b, c) in enumerate(((0, 1, 2), (1, 0, 2), (2, 0, 1)), start=1):
    mask = ~positive & (
      np.diagonal(matrix, axis1=-2, axis2=-1)[..., index - 1]
      > np.diagonal(matrix, axis1=-2, axis2=-1)[..., (index) % 3]
    )
    if index == 3:
      mask = ~positive & ~(
        (
          np.diagonal(matrix, axis1=-2, axis2=-1)[..., 0]
          > np.diagonal(matrix, axis1=-2, axis2=-1)[..., 1]
        )
        | False
      )
    # A generic eigen-free fallback for rare near-180-degree rotations.
    if np.any(mask):
      d = (
        np.sqrt(
          np.maximum(
            1 + matrix[mask, a, a] - matrix[mask, b, b] - matrix[mask, c, c], 1e-15
          )
        )
        * 2
      )
      out[mask, index] = 0.25 * d
      out[mask, 0] = (matrix[mask, c, b] - matrix[mask, b, c]) / d
      out[mask, b] = (matrix[mask, b, a] + matrix[mask, a, b]) / d
      out[mask, c] = (matrix[mask, c, a] + matrix[mask, a, c]) / d
  # The branches above overlap only in numerical tie cases; normalize and use
  # the positive scalar convention for deterministic serialization.
  out /= np.linalg.norm(out, axis=-1, keepdims=True)
  out *= np.where(out[..., :1] < 0, -1.0, 1.0)
  return out


def _rot6d(matrix: np.ndarray) -> np.ndarray:
  return np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), axis=-1)


def _rot6d_matrix(value: np.ndarray) -> np.ndarray:
  first, second = value[..., :3], value[..., 3:6]
  # The six values came from an orthonormal matrix.  Gram-Schmidt also makes
  # inverse transforms robust to a model prediction slightly off-manifold.
  first = first / np.linalg.norm(first, axis=-1, keepdims=True)
  second = second - np.sum(first * second, axis=-1, keepdims=True) * first
  second = second / np.linalg.norm(second, axis=-1, keepdims=True)
  third = np.cross(first, second)
  return np.stack((first, second, third), axis=-1)


@dataclass(frozen=True, slots=True)
class WorldState:
  """One simulator-ground-truth state in world coordinates."""

  root_position: np.ndarray
  root_quaternion_wxyz: np.ndarray
  root_linear_velocity: np.ndarray
  root_angular_velocity: np.ndarray
  body_positions: np.ndarray
  body_linear_velocities: np.ndarray

  def __post_init__(self) -> None:
    body_count = DEFAULT_CONTRACT.body_count
    object.__setattr__(
      self,
      "root_position",
      _array("root_position", self.root_position, (3,), batched=True),
    )
    object.__setattr__(
      self,
      "root_quaternion_wxyz",
      _array("root_quaternion_wxyz", self.root_quaternion_wxyz, (4,), batched=True),
    )
    object.__setattr__(
      self,
      "root_linear_velocity",
      _array("root_linear_velocity", self.root_linear_velocity, (3,), batched=True),
    )
    object.__setattr__(
      self,
      "root_angular_velocity",
      _array("root_angular_velocity", self.root_angular_velocity, (3,), batched=True),
    )
    object.__setattr__(
      self,
      "body_positions",
      _array("body_positions", self.body_positions, (body_count, 3), batched=True),
    )
    object.__setattr__(
      self,
      "body_linear_velocities",
      _array(
        "body_linear_velocities",
        self.body_linear_velocities,
        (body_count, 3),
        batched=True,
      ),
    )
    quaternion_norms = np.linalg.norm(
      self.root_quaternion_wxyz,
      axis=-1,
    )
    if np.any(quaternion_norms < 1e-12):
      raise StateValidationError("root quaternion has zero norm")

  @classmethod
  def stack(cls, states: Iterable["WorldState"]) -> "WorldState":
    values = tuple(states)
    if not values:
      raise StateValidationError("cannot stack an empty state sequence")
    return cls(
      *(
        np.stack([getattr(state, field) for state in values]) for field in _STATE_FIELDS
      )
    )

  @classmethod
  def _from_fields(cls, fields: Iterable[np.ndarray]) -> "WorldState":
    """Build a state from arrays that a caller already validated."""
    instance = object.__new__(cls)
    for name, value in zip(_STATE_FIELDS, fields, strict=True):
      object.__setattr__(instance, name, value)
    return instance

  def window(self, start: int, stop: int) -> "WorldState":
    """Return rows ``[start, stop)`` of a batched state as a slice view.

    :meth:`stack` validates every row of a batch, which is what makes the
    batched fields trustworthy.  An offline pass over overlapping windows of one
    segment resolves the same rows again and again, so this method returns a
    view of already validated rows instead of repeating those per-row checks.
    """
    if self.root_position.ndim != 2:
      raise StateValidationError("window requires a batched state")
    length = self.root_position.shape[0]
    if not 0 <= start < stop <= length:
      raise StateValidationError("window bounds are outside the state batch")
    return self._from_fields(getattr(self, name)[start:stop] for name in _STATE_FIELDS)

  @property
  def count(self) -> int:
    return self.root_position.shape[0] if self.root_position.ndim == 2 else 1

  def select(self, indices: np.ndarray | Iterable[int]) -> "WorldState":
    """Return a copied row subset while preserving the batch axis."""
    ids = np.asarray(tuple(indices), dtype=np.int64).reshape(-1)
    if ids.size == 0 or np.any(ids < 0) or np.any(ids >= self.count):
      raise StateValidationError("state selection indices are outside the batch")
    if self.root_position.ndim != 2:
      if ids.size != 1 or ids[0] != 0:
        raise StateValidationError("a single state only supports selecting row zero")
      return self
    return type(self)(
      *(np.asarray(getattr(self, field))[ids].copy() for field in _STATE_FIELDS)
    )


@dataclass(frozen=True, slots=True)
class WorldFrameContext:
  """World values retained so a hybrid state can be inverted for physical costs."""

  center_position: np.ndarray
  center_yaw: float
  center_linear_velocity: np.ndarray
  root_positions: np.ndarray
  root_linear_velocities: np.ndarray
  root_yaws: np.ndarray


def _as_sequence(states: WorldState) -> tuple[np.ndarray, ...]:
  if states.root_position.ndim != 2:
    return tuple(np.expand_dims(getattr(states, field), 0) for field in _STATE_FIELDS)
  return tuple(getattr(states, field) for field in _STATE_FIELDS)


def world_to_hybrid(
  states: WorldState, *, center_index: int
) -> tuple[np.ndarray, WorldFrameContext]:
  """Pack a trajectory into the contract's 135-D pelvis-rooted hybrid state."""
  root_p, root_q, root_v, root_w, body_p, body_v = _as_sequence(states)
  n = root_p.shape[0]
  if center_index < 0 or center_index >= n:
    raise StateValidationError("center_index is outside the trajectory")
  root_r = _quat_matrix(root_q)
  yaw = np.arctan2(root_r[:, 1, 0], root_r[:, 0, 0])
  center_yaw = float(yaw[center_index])
  center_r = _yaw_matrix(np.asarray(center_yaw))
  center_rt = center_r.T
  root_rel_p = (root_p - root_p[center_index]) @ center_r
  root_rel_v = (root_v - root_v[center_index]) @ center_r
  root_rel_r = np.einsum("ij,njk->nik", center_rt, root_r)
  root_rel_w = root_w @ center_r
  body_rt = np.swapaxes(_yaw_matrix(yaw), -1, -2)
  body_rel_p = np.einsum("nij,nkj->nki", body_rt, body_p - root_p[:, None, :])
  body_rel_v = np.einsum("nij,nkj->nki", body_rt, body_v - root_v[:, None, :])
  hybrid = np.concatenate(
    (
      root_rel_p,
      _rot6d(root_rel_r),
      root_rel_v,
      root_rel_w,
      body_rel_p.reshape(n, -1),
      body_rel_v.reshape(n, -1),
    ),
    axis=-1,
  )
  context = WorldFrameContext(
    root_p[center_index].copy(),
    center_yaw,
    root_v[center_index].copy(),
    root_p.copy(),
    root_v.copy(),
    yaw.copy(),
  )
  return hybrid, context


def world_to_hybrid_windows(
  states: WorldState,
  starts: np.ndarray,
  *,
  center_index: int,
  length: int,
) -> np.ndarray:
  """Pack many equal-length windows of one batched state in a single pass.

  Offline statistics used to call :func:`world_to_hybrid` once per window from a
  Python loop.  This function applies the same per-window arithmetic to a whole
  block of ``len(starts)`` windows at once, so the caller issues one vectorized
  transform per block instead of one scalar call per window.  ``center_index`` is
  relative to each window, exactly as :func:`world_to_hybrid` treats it, and the
  result is ``(len(starts), length, state_dimension)`` in float64.  Each row is
  gathered straight from the validated batch, so the operation is deterministic
  and its temporary arrays are bounded by ``len(starts) * length``.
  """
  if states.root_position.ndim != 2:
    raise StateValidationError("world_to_hybrid_windows requires a batched state")
  if length <= 0 or center_index < 0 or center_index >= length:
    raise StateValidationError("window length and center index are invalid")
  rows = np.asarray(starts, dtype=np.int64)
  if rows.ndim != 1:
    raise StateValidationError("window starts must be one-dimensional")
  count = rows.size
  if count == 0:
    return np.empty((0, length, DEFAULT_CONTRACT.state_dimension), dtype=np.float64)
  root_p, root_q, root_v, root_w, body_p, body_v = _as_sequence(states)
  total = root_p.shape[0]
  if np.any(rows < 0) or np.any(rows + length > total):
    raise StateValidationError("window starts are outside the state batch")
  # Row index per (window, step).  Every gather below reuses this one array.
  offsets = rows[:, None] + np.arange(length, dtype=np.int64)[None, :]
  root_r = _quat_matrix(root_q)
  yaw = np.arctan2(root_r[:, 1, 0], root_r[:, 0, 0])
  body_rt = np.swapaxes(_yaw_matrix(yaw), -1, -2)
  p = root_p[offsets]
  v = root_v[offsets]
  w = root_w[offsets]
  center_r = _yaw_matrix(yaw[rows + center_index])
  center_rt = np.swapaxes(center_r, -1, -2)
  # The centre-relative terms use ``matmul`` and the rotations use ``einsum``, so
  # each window keeps the operation order of :func:`world_to_hybrid`.
  root_rel_p = np.matmul(p - p[:, center_index : center_index + 1], center_r)
  root_rel_v = np.matmul(v - v[:, center_index : center_index + 1], center_r)
  root_rel_r = np.einsum("wij,wljk->wlik", center_rt, root_r[offsets])
  root_rel_w = np.matmul(w, center_r)
  body_rel_p = np.einsum(
    "wlij,wlkj->wlki", body_rt[offsets], body_p[offsets] - p[:, :, None, :]
  )
  body_rel_v = np.einsum(
    "wlij,wlkj->wlki", body_rt[offsets], body_v[offsets] - v[:, :, None, :]
  )
  return np.concatenate(
    (
      root_rel_p,
      _rot6d(root_rel_r),
      root_rel_v,
      root_rel_w,
      body_rel_p.reshape(count, length, -1),
      body_rel_v.reshape(count, length, -1),
    ),
    axis=-1,
  )


def _state_vector(name: str, value: np.ndarray, width: int) -> np.ndarray:
  result = np.asarray(value, dtype=np.float64)
  if result.ndim == 0 or result.shape[-1] != width:
    raise StateValidationError(f"{name} must have shape [..., {width}]")
  return result


def hybrid_to_local_kinematic(hybrid: np.ndarray) -> np.ndarray:
  """Drop the root translation fields from a 135-D hybrid state.

  The input keeps the frozen :func:`world_to_hybrid` frame semantics: the root
  rot6d and angular velocity are relative to the window-center yaw frame, and
  the body positions and linear velocities are relative to the same-timestep
  root yaw frame.  The result uses the :data:`LOCAL_KINEMATIC_STATE` layout and
  keeps every retained channel unchanged.  Only the window-center-relative root
  position and root linear velocity are removed.
  """
  value = _state_vector("hybrid state", hybrid, HYBRID_STATE.dimension)
  return np.concatenate(
    [value[..., HYBRID_STATE.field(name)] for name in _LOCAL_KINEMATIC_FIELDS],
    axis=-1,
  )


def local_kinematic_to_hybrid(
  local: np.ndarray,
  *,
  root_position: np.ndarray,
  root_linear_velocity: np.ndarray,
) -> np.ndarray:
  """Insert the root translation fields into a 129-D local-kinematic state.

  ``root_position`` and ``root_linear_velocity`` are the values that the hybrid
  layout stores in its first and third fields: world root values measured
  relative to the window center and rotated into the window-center yaw frame.
  A local-kinematic state alone does not determine them.
  """
  value = _state_vector("local-kinematic state", local, LOCAL_KINEMATIC_STATE.dimension)
  inserted = {
    "root_position": _state_vector("root_position", root_position, 3),
    "root_linear_velocity": _state_vector(
      "root_linear_velocity", root_linear_velocity, 3
    ),
  }
  return np.concatenate(
    [
      inserted[name]
      if name in inserted
      else value[..., LOCAL_KINEMATIC_STATE.field(name)]
      for name in HYBRID_STATE.slices
    ],
    axis=-1,
  )


def world_to_local_kinematic(
  states: WorldState, *, center_index: int
) -> tuple[np.ndarray, WorldFrameContext]:
  """Pack a trajectory into the 129-D local-kinematic state.

  This is :func:`world_to_hybrid` followed by
  :func:`hybrid_to_local_kinematic`, so every retained channel keeps the frame
  of the 135-D hybrid state.  The returned context still holds the world root
  positions and linear velocities that :func:`local_kinematic_to_world` needs.
  """
  hybrid, context = world_to_hybrid(states, center_index=center_index)
  return hybrid_to_local_kinematic(hybrid), context


def local_kinematic_to_world(
  local: np.ndarray, context: WorldFrameContext
) -> WorldState:
  """Invert :func:`world_to_local_kinematic` with retained world frame context.

  A local-kinematic state does not encode the root world position or the root
  world linear velocity.  The inverse reads them from ``context.root_positions``
  and ``context.root_linear_velocities`` and rotates the center-relative
  difference into the window-center yaw frame before
  :func:`local_kinematic_to_hybrid`.  Generation code that feeds a predicted
  129-D state here must supply the intended root trajectory in the context.
  """
  value = _state_vector("local-kinematic state", local, LOCAL_KINEMATIC_STATE.dimension)
  if value.ndim != 2:
    raise StateValidationError("local-kinematic state must have shape [T, 129]")
  n = value.shape[0]
  if context.root_positions.shape != (n, 3) or context.root_linear_velocities.shape != (
    n,
    3,
  ):
    raise StateValidationError("world context does not match local-kinematic length")
  center_r = _yaw_matrix(np.asarray(context.center_yaw))
  root_position = (context.root_positions - context.center_position) @ center_r
  root_linear_velocity = (
    context.root_linear_velocities - context.center_linear_velocity
  ) @ center_r
  hybrid = local_kinematic_to_hybrid(
    value,
    root_position=root_position,
    root_linear_velocity=root_linear_velocity,
  )
  return hybrid_to_world(hybrid, context)


def hybrid_to_world(hybrid: np.ndarray, context: WorldFrameContext) -> WorldState:
  """Invert :func:`world_to_hybrid` using retained world planning context."""
  value = np.asarray(hybrid, dtype=np.float64)
  if value.ndim != 2 or value.shape[1] != DEFAULT_CONTRACT.state_dimension:
    raise StateValidationError("hybrid must have shape [T, 135]")
  n = value.shape[0]
  if context.root_positions.shape != (n, 3) or context.root_linear_velocities.shape != (
    n,
    3,
  ):
    raise StateValidationError("world context does not match hybrid length")
  center_r = _yaw_matrix(np.asarray(context.center_yaw))
  root_r = np.einsum("ij,njk->nik", center_r, _rot6d_matrix(value[:, 3:9]))
  root_p = context.center_position + value[:, :3] @ center_r.T
  root_v = context.center_linear_velocity + value[:, 9:12] @ center_r.T
  root_w = value[:, 12:15] @ center_r.T
  body = value[:, 15:75].reshape(n, DEFAULT_CONTRACT.body_count, 3)
  body_v = value[:, 75:].reshape(n, DEFAULT_CONTRACT.body_count, 3)
  body_r = _yaw_matrix(context.root_yaws)
  body_p = root_p[:, None, :] + np.einsum("nij,nkj->nki", body_r, body)
  body_world_v = root_v[:, None, :] + np.einsum("nij,nkj->nki", body_r, body_v)
  return WorldState(
    root_p, _matrix_to_quaternion(root_r), root_v, root_w, body_p, body_world_v
  )


__all__ = [
  "HYBRID_STATE",
  "LOCAL_KINEMATIC_STATE",
  "StateRepresentation",
  "StateValidationError",
  "WorldFrameContext",
  "WorldState",
  "hybrid_to_local_kinematic",
  "hybrid_to_world",
  "local_kinematic_to_hybrid",
  "local_kinematic_to_world",
  "world_to_hybrid",
  "world_to_hybrid_windows",
  "world_to_local_kinematic",
]
