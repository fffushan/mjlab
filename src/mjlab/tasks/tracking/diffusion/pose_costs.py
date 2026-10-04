"""D4a differentiable physical pose cost for sparse soft pose goals.

This module owns one seam: turning absolute-tick :class:`TimedPoseGoal` targets
plus explicit :class:`PoseCostWeights` into a torch-only, differentiable
``G(x0)`` that input-VJP guidance (``guidance.py``) differentiates through the
frozen denoiser.  Goals are *soft*: nothing here writes a conditioning mask, so
a partial pose can never be promoted to hard inpainting.

The physical recovery inverts :func:`~.state.world_to_hybrid` for the first 135
normalized channels of a token window:

* denormalize with the frozen per-feature state mean/std of the projection
  bundle (never a re-fitted statistic);
* restore the world root position with the measured current window-center
  position and **yaw** taken from the current measurement, not from a recorded
  trajectory;
* build a proper rotation from the predicted rot6d and express it in the world
  frame through the same center rotation;
* restore every body world position with the **predicted timestep's** root yaw
  re-derived from the predicted root rotation.

The stored future root yaws of :class:`~.state.WorldFrameContext` are
deliberately unused: :func:`~.state.hybrid_to_world` is the inverse of
``world_to_hybrid`` only when the recorded trajectory (including its future
yaws) is known, which is exactly what a generated sample does not have.

The objective contains position (squared metres) and orientation (squared
chordal rotation-matrix distance, dimensionless) terms only.  Velocity,
contact, torque and smoothing objectives are deliberately absent in this slice.

Every numerical geometry guard is documented at
:data:`_GEOMETRY_EPSILON` and is separate from the frozen training
normalization ``std_floor`` in the projection bundle.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from torch import Tensor

from .contract import DEFAULT_CONTRACT, DiffusionContract
from .projection import ProjectionBundle
from .state import WorldState

_ACTIVE_SCHEMA = "mjlab-x2-state-latent-d1-no-mixtures-v1"

# Minimum scheduling spacing between two soft pose goals, in control ticks.
# The frozen control period is 0.02 s, so this is 0.2 s of motion.
_MINIMUM_GOAL_SPACING = 10

# Numerical geometry guard for a *predicted* rotation.  A rot6d whose first or
# second column has collapsed below this norm cannot define a rotation matrix,
# so the goal is rejected instead of returning a fabricated orientation.  This
# is intentionally separate from the frozen training normalization ``std_floor``
# in :class:`~.projection.FeatureStats`: that floor makes immutable dataset
# statistics invertible, this guard rejects a degenerate generated prediction.
_GEOMETRY_EPSILON = 1.0e-8

_STATE_ARRAY_FIELDS = (
  "root_position",
  "root_quaternion_wxyz",
  "root_linear_velocity",
  "root_angular_velocity",
  "body_positions",
  "body_linear_velocities",
)


class PoseCostError(ValueError):
  """A pose goal, weight set, token window or predicted pose is invalid."""


def _require_active_contract(contract: DiffusionContract) -> None:
  """Reject legacy projected (231-channel) contracts at the pose-cost seam."""
  if contract.projection_rows != 0 or contract.schema_version != _ACTIVE_SCHEMA:
    raise PoseCostError(
      "the pose cost requires the active no-mixture 167-channel contract; "
      f"got schema {contract.schema_version!r} with {contract.projection_rows} "
      "added projection rows"
    )


def _require_matching_projection(
  projection: ProjectionBundle, contract: DiffusionContract
) -> None:
  projection_contract = getattr(projection, "contract", None)
  if projection_contract is None or not hasattr(projection_contract, "identity_hash"):
    raise PoseCostError("projection bundle does not carry its diffusion contract")
  if projection_contract.identity_hash() != contract.identity_hash():
    raise PoseCostError(
      "projection bundle contract identity differs from the pose-cost contract"
    )


def _require_tick(value: object, name: str) -> int:
  """Return ``value`` as a non-negative integer control tick."""
  if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
    raise PoseCostError(f"{name} must be a non-negative integer control tick")
  tick = int(value)
  if tick < 0:
    raise PoseCostError(f"{name} must be non-negative, got {tick}")
  return tick


def _require_weight(value: object, name: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
    raise PoseCostError(f"{name} weight must be a finite non-negative number")
  weight = float(value)
  if not np.isfinite(weight) or weight < 0.0:
    raise PoseCostError(f"{name} weight must be finite and non-negative, got {weight}")
  return weight


@dataclass(frozen=True, slots=True)
class TimedPoseGoal:
  """One sparse soft pose goal: absolute control tick, world pose, origin.

  The goal is a complete unbatched six-array :class:`WorldState` so it is a
  measured world pose with the full 20-body order; the cost reads only its root
  position, root orientation and body positions.  The goal owns copies of all
  six arrays, so callers may reuse their buffers afterwards.  It is an explicit
  task input, never a hard conditioning row: use ``inpainting.py`` for hard
  keyframes.
  """

  tick: int
  state: WorldState
  provenance: str

  def __post_init__(self) -> None:
    object.__setattr__(self, "tick", _require_tick(self.tick, "pose goal tick"))
    if not isinstance(self.state, WorldState):
      raise PoseCostError(
        "pose goal state must be a complete WorldState with the six physical arrays"
      )
    if self.state.root_position.ndim != 1:
      raise PoseCostError("pose goal state must be one unbatched measurement")
    if not isinstance(self.provenance, str) or not self.provenance.strip():
      raise PoseCostError(
        "pose goal provenance must be a non-empty source label (recording, "
        "selection or fixture it came from)"
      )
    object.__setattr__(
      self,
      "state",
      WorldState(
        **{
          name: np.array(getattr(self.state, name), dtype=np.float64, copy=True)
          for name in _STATE_ARRAY_FIELDS
        }
      ),
    )


@dataclass(frozen=True, slots=True)
class ActivePoseGoal:
  """One pose goal that lands inside the plan's ``1..future_steps`` window."""

  tick: int
  offset: int
  row: int


@dataclass(frozen=True, slots=True)
class PoseCostWeights:
  """Explicit non-negative weights of the pose-cost terms.

  ``root_position`` and ``body_positions`` scale squared-metre position costs;
  ``root_orientation`` scales the dimensionless squared chordal rotation
  distance ``||R_pred - R_goal||_F^2``.  ``body_coefficients`` optionally gives
  one non-negative coefficient per contract body: a mapping names exactly the
  weighted bodies and leaves every unlisted body at coefficient ``0.0`` (the
  partial-pose case), while a sequence must supply all 20 in contract order.
  ``None`` weights every body at ``1.0``.
  """

  root_position: float
  root_orientation: float
  body_positions: float
  body_coefficients: Sequence[float] | Mapping[str, float] | None = None

  def __post_init__(self) -> None:
    for name in ("root_position", "root_orientation", "body_positions"):
      object.__setattr__(self, name, _require_weight(getattr(self, name), name))
    if (
      self.root_position == 0.0
      and self.root_orientation == 0.0
      and self.body_positions == 0.0
    ):
      raise PoseCostError(
        "at least one pose-cost weight must be positive; an all-zero weight set "
        "would report guidance without any objective"
      )
    object.__setattr__(
      self, "body_coefficients", _coefficient_tuple(self.body_coefficients)
    )

  @property
  def coefficients(self) -> tuple[float, ...]:
    """Return the 20 per-body coefficients, defaulting to all ones."""
    if self.body_coefficients is None:
      return (1.0,) * DEFAULT_CONTRACT.body_count
    # ``__post_init__`` replaced whatever input was accepted with a validated
    # tuple, so the field is only ever ``None`` or a 20-tuple of floats here.
    return cast("tuple[float, ...]", self.body_coefficients)


def _coefficient_tuple(
  value: Sequence[float] | Mapping[str, float] | None,
) -> tuple[float, ...] | None:
  """Validate per-body coefficients and return them in contract body order."""
  if value is None:
    return None
  count = DEFAULT_CONTRACT.body_count
  if isinstance(value, Mapping):
    unknown = sorted(set(value) - set(DEFAULT_CONTRACT.body_names))
    if unknown:
      raise PoseCostError(f"body coefficients name unknown bodies {unknown}")
    coefficients = [0.0] * count
    for name, coefficient in value.items():
      coefficients[DEFAULT_CONTRACT.body_names.index(name)] = _require_weight(
        coefficient, f"body coefficient for {name}"
      )
    return tuple(coefficients)
  if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
    raise PoseCostError(
      "body coefficients must be a mapping of body names, a 20-length sequence or None"
    )
  if len(value) != count:
    raise PoseCostError(
      f"body coefficients must contain {count} values in contract order, "
      f"got {len(value)}"
    )
  return tuple(
    _require_weight(coefficient, f"body coefficient {index}")
    for index, coefficient in enumerate(value)
  )


def active_pose_goals(
  plan_tick: int,
  goals: Iterable[TimedPoseGoal],
  *,
  contract: DiffusionContract = DEFAULT_CONTRACT,
) -> tuple[ActivePoseGoal, ...]:
  """Return the bounded active soft goals of ``goals``, sorted by tick.

  Active means ``1 <= tick - plan_tick <= future_steps`` and the goal is mapped
  to row ``current_index + offset``.  Reached, expired and beyond-horizon goals
  are inactive and are reported by omission; they are not shifted forward or
  replaced by measured history.
  """
  _require_active_contract(contract)
  tick = _require_tick(plan_tick, "plan tick")
  active: list[ActivePoseGoal] = []
  for goal in _validated_goals(goals):
    offset = goal.tick - tick
    if 1 <= offset <= contract.future_steps:
      active.append(
        ActivePoseGoal(
          tick=goal.tick,
          offset=offset,
          row=contract.current_index + offset,
        )
      )
  return tuple(active)


def _validated_goals(goals: Iterable[TimedPoseGoal]) -> tuple[TimedPoseGoal, ...]:
  """Return the goals sorted by tick after duplicate/spacing validation.

  Duplicate ticks and pairs closer than :data:`_MINIMUM_GOAL_SPACING` ticks are
  input errors, not silently deduplicated or merged: the scheduled soft-goal
  list is a task-level contract, exactly like the hard-keyframe list.
  """
  if goals is None or isinstance(goals, (str, bytes, Mapping)):
    raise PoseCostError(
      "goals must be an iterable of TimedPoseGoal, not a scalar or mapping"
    )
  try:
    items = tuple(goals)
  except TypeError as exc:
    raise PoseCostError("goals must be an iterable of TimedPoseGoal") from exc
  for item in items:
    if not isinstance(item, TimedPoseGoal):
      raise PoseCostError("goals must contain only TimedPoseGoal values")
  ticks = [item.tick for item in items]
  if len(set(ticks)) != len(ticks):
    raise PoseCostError("duplicate pose goal ticks are not accepted")
  ordered = tuple(sorted(items, key=lambda item: item.tick))
  for previous, following in zip(ordered, ordered[1:], strict=False):
    if following.tick - previous.tick < _MINIMUM_GOAL_SPACING:
      raise PoseCostError(
        f"pose goals {previous.tick} and {following.tick} are closer than the "
        f"{_MINIMUM_GOAL_SPACING}-tick minimum spacing"
      )
  return ordered


def _yaw_matrix(yaw: Tensor) -> Tensor:
  """Return the +z rotation matrix of ``yaw``, matching ``state._yaw_matrix``."""
  zeros = torch.zeros_like(yaw)
  ones = torch.ones_like(yaw)
  cosine, sine = torch.cos(yaw), torch.sin(yaw)
  return torch.stack(
    (cosine, -sine, zeros, sine, cosine, zeros, zeros, zeros, ones), dim=-1
  ).reshape(*yaw.shape, 3, 3)


def _quaternion_matrix(quaternion: Tensor) -> Tensor:
  """Convert wxyz quaternions to rotation matrices, matching ``state.py``."""
  norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
  if bool(torch.any(norm < _GEOMETRY_EPSILON)):
    raise PoseCostError(
      "goal root quaternion is degenerate (norm below the geometric guard)"
    )
  w, x, y, z = (quaternion / norm).unbind(-1)
  return torch.stack(
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
    dim=-1,
  ).reshape(*quaternion.shape[:-1], 3, 3)


def _rot6d_matrix(value: Tensor) -> Tensor:
  """Rebuild a proper rotation from six predicted values, matching ``state.py``.

  Gram-Schmidt is applied to the two predicted columns so a slightly
  off-manifold prediction still yields a rotation matrix.  A collapsed column
  is a degenerate prediction and is rejected through :data:`_GEOMETRY_EPSILON`.
  """
  first, second = value[..., :3], value[..., 3:6]
  first_norm = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
  if bool(torch.any(first_norm < _GEOMETRY_EPSILON)):
    raise PoseCostError(
      "predicted root rot6d is degenerate: its first column is below the "
      "geometric guard"
    )
  first = first / first_norm
  second = second - (first * second).sum(dim=-1, keepdim=True) * first
  second_norm = torch.linalg.vector_norm(second, dim=-1, keepdim=True)
  if bool(torch.any(second_norm < _GEOMETRY_EPSILON)):
    raise PoseCostError(
      "predicted root rot6d is degenerate: its second column is parallel to "
      "the first within the geometric guard"
    )
  second = second / second_norm
  third = torch.cross(first, second, dim=-1)
  return torch.stack((first, second, third), dim=-1)


@dataclass(frozen=True, slots=True)
class WorldPoses:
  """Predicted world pose tensors recovered from one normalized state slice."""

  root_position: Tensor
  root_rotation: Tensor
  body_positions: Tensor


def _validated_normalized_state(
  normalized_state: Tensor, contract: DiffusionContract
) -> Tensor:
  if (
    not isinstance(normalized_state, Tensor) or not normalized_state.is_floating_point()
  ):
    raise PoseCostError("normalized state must be a floating-point torch tensor")
  if (
    normalized_state.ndim < 1 or normalized_state.shape[-1] != contract.state_dimension
  ):
    raise PoseCostError(
      f"normalized state last dimension must be {contract.state_dimension}, got "
      f"{tuple(normalized_state.shape)}"
    )
  if not bool(torch.isfinite(normalized_state).all()):
    raise PoseCostError("normalized state contains non-finite values")
  return normalized_state


def _world_poses(
  normalized_state: Tensor,
  *,
  mean: Tensor,
  std: Tensor,
  center_position: Tensor,
  center_yaw: Tensor,
) -> WorldPoses:
  """Recover world root/body poses from a normalized state slice.

  ``normalized_state`` has shape ``[..., 135]``; all context tensors are
  constants copied from the measured window center.  The result keeps the same
  leading dimensions.
  """
  hybrid = normalized_state * std + mean
  center_rotation = _yaw_matrix(center_yaw)
  root_position = center_position + hybrid[..., 0:3] @ center_rotation.transpose(-1, -2)
  root_relative_rotation = _rot6d_matrix(hybrid[..., 3:9])
  root_rotation = center_rotation @ root_relative_rotation
  # The body offsets of every row are expressed in that row's own root yaw, so
  # the yaw must come from the predicted rotation of the same row.  Using the
  # measured center yaw here would silently rotate the whole pose.
  yaw = torch.atan2(root_rotation[..., 1, 0], root_rotation[..., 0, 0])
  body_local = hybrid[..., 15:75].reshape(*hybrid.shape[:-1], 20, 3)
  body_positions = root_position.unsqueeze(-2) + torch.einsum(
    "...ij,...kj->...ki", _yaw_matrix(yaw), body_local
  )
  return WorldPoses(
    root_position=root_position,
    root_rotation=root_rotation,
    body_positions=body_positions,
  )


def _context_tensors(
  projection: ProjectionBundle, current_state: WorldState, dtype: torch.dtype
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
  """Return frozen state mean/std and the measured center position/yaw."""
  if not isinstance(current_state, WorldState):
    raise PoseCostError("current state must be a complete WorldState")
  if current_state.root_position.ndim != 1:
    raise PoseCostError("current state must be one unbatched measurement")
  mean = torch.as_tensor(projection.state_stats.mean, dtype=dtype)
  std = torch.as_tensor(projection.state_stats.std, dtype=dtype)
  center_position = torch.as_tensor(current_state.root_position, dtype=dtype)
  center_rotation = _quaternion_matrix(
    torch.as_tensor(current_state.root_quaternion_wxyz, dtype=dtype)
  )
  center_yaw = torch.atan2(center_rotation[1, 0], center_rotation[0, 0])
  return mean, std, center_position, center_yaw


def world_poses_from_normalized_state(
  normalized_state: Tensor,
  *,
  projection: ProjectionBundle,
  current_state: WorldState,
  contract: DiffusionContract = DEFAULT_CONTRACT,
) -> WorldPoses:
  """Recover world poses from normalized state channels relative to ``current_state``.

  This is the differentiable physical-recovery seam: it performs no NumPy
  conversion and no ``detach``, so a cost built on it differentiates back to the
  normalized state, through the denoiser, and on to the noisy trajectory.
  """
  _require_active_contract(contract)
  _require_matching_projection(projection, contract)
  normalized = _validated_normalized_state(normalized_state, contract)
  mean, std, center_position, center_yaw = _context_tensors(
    projection, current_state, normalized.dtype
  )
  return _world_poses(
    normalized,
    mean=mean,
    std=std,
    center_position=center_position,
    center_yaw=center_yaw,
  )


def _stack_goal_poses(
  active: list[TimedPoseGoal], *, body_count: int
) -> tuple[Tensor, Tensor, Tensor]:
  """Stack the goal pose tensors, keeping an empty goal list well-formed."""
  if not active:
    empty = torch.zeros(0, dtype=torch.float64)
    return (
      empty.reshape(0, 3),
      empty.reshape(0, 3, 3),
      empty.reshape(0, body_count, 3),
    )
  return (
    torch.stack(
      [
        torch.as_tensor(goal.state.root_position, dtype=torch.float64)
        for goal in active
      ],
      dim=0,
    ),
    torch.stack(
      [
        _quaternion_matrix(
          torch.as_tensor(goal.state.root_quaternion_wxyz, dtype=torch.float64)
        )
        for goal in active
      ],
      dim=0,
    ),
    torch.stack(
      [
        torch.as_tensor(goal.state.body_positions, dtype=torch.float64).reshape(
          body_count, 3
        )
        for goal in active
      ],
      dim=0,
    ),
  )


class PoseCost:
  """Differentiable ``G(x0)`` over sparse active soft pose goals.

  ``G = period_seconds * sum(active goal costs)`` per trajectory: the position
  terms are squared metres, the orientation term is the dimensionless squared
  chordal distance, and ``body_coefficients`` scale the per-body terms.  The
  callable consumes a clean token window ``[B, 41, 167]`` and returns ``[B]``;
  only the first 135 channels of the goal rows are read.
  """

  def __init__(
    self,
    *,
    projection: ProjectionBundle,
    current_state: WorldState,
    plan_tick: int,
    goals: Iterable[TimedPoseGoal],
    weights: PoseCostWeights,
    contract: DiffusionContract = DEFAULT_CONTRACT,
  ) -> None:
    _require_active_contract(contract)
    _require_matching_projection(projection, contract)
    if not isinstance(weights, PoseCostWeights):
      raise PoseCostError("weights must be a PoseCostWeights instance")
    self.contract = contract
    self.projection = projection
    self.weights = weights
    self.plan_tick = _require_tick(plan_tick, "plan tick")
    self.goals = _validated_goals(goals)
    self.active_goals = active_pose_goals(self.plan_tick, self.goals, contract=contract)
    active_ticks = {goal.tick for goal in self.active_goals}
    self.inactive_goal_ticks = tuple(
      goal.tick for goal in self.goals if goal.tick not in active_ticks
    )
    mean, std, center_position, center_yaw = _context_tensors(
      projection, current_state, torch.float64
    )
    self._mean = mean
    self._std = std
    self._center_position = center_position
    self._center_yaw = center_yaw
    self._by_tick = {goal.tick: goal for goal in self.goals}
    active = [self._by_tick[goal.tick] for goal in self.active_goals]
    (
      self._goal_root_positions,
      self._goal_root_rotations,
      self._goal_body_positions,
    ) = _stack_goal_poses(active, body_count=contract.body_count)
    self._coefficients = torch.as_tensor(self.weights.coefficients, dtype=torch.float64)

  @property
  def rows(self) -> tuple[int, ...]:
    """Return the window rows of the active goals, in tick order."""
    return tuple(goal.row for goal in self.active_goals)

  def __call__(self, clean_tokens: Tensor) -> Tensor:
    """Return the per-trajectory pose cost ``G`` of a clean token window."""
    tokens = _validated_tokens(clean_tokens, self.contract)
    batch = tokens.shape[0]
    if not self.active_goals:
      return torch.zeros(batch, dtype=tokens.dtype, device=tokens.device)
    normalized = tokens[:, self.rows, : self.contract.state_dimension]
    poses = _world_poses(
      normalized,
      mean=self._mean.to(device=tokens.device, dtype=tokens.dtype),
      std=self._std.to(device=tokens.device, dtype=tokens.dtype),
      center_position=self._center_position.to(
        device=tokens.device, dtype=tokens.dtype
      ),
      center_yaw=self._center_yaw.to(device=tokens.device, dtype=tokens.dtype),
    )
    goal_positions = self._goal_root_positions.to(tokens.device, tokens.dtype)
    goal_rotations = self._goal_root_rotations.to(tokens.device, tokens.dtype)
    goal_bodies = self._goal_body_positions.to(tokens.device, tokens.dtype)
    coefficients = self._coefficients.to(tokens.device, tokens.dtype)
    position_cost = ((poses.root_position - goal_positions) ** 2).sum(dim=-1)
    orientation_cost = ((poses.root_rotation - goal_rotations) ** 2).sum(dim=(-1, -2))
    body_cost = (
      ((poses.body_positions - goal_bodies) ** 2).sum(dim=-1) * coefficients
    ).sum(dim=-1)
    pose_cost = (
      self.weights.root_position * position_cost
      + self.weights.root_orientation * orientation_cost
      + self.weights.body_positions * body_cost
    )
    total = self.contract.period_seconds * pose_cost.sum(dim=-1)
    if not bool(torch.isfinite(total).all()):
      raise PoseCostError(
        "pose cost is non-finite; the predicted or goal world pose is degenerate"
      )
    return total


def _validated_tokens(clean_tokens: Tensor, contract: DiffusionContract) -> Tensor:
  if not isinstance(clean_tokens, Tensor) or not clean_tokens.is_floating_point():
    raise PoseCostError("clean tokens must be a floating-point torch tensor")
  expected = (contract.window_steps, contract.token_dimension)
  if clean_tokens.ndim != 3 or tuple(clean_tokens.shape[1:]) != expected:
    raise PoseCostError(
      f"clean tokens must have shape [B, {expected[0]}, {expected[1]}], got "
      f"{tuple(clean_tokens.shape)}"
    )
  if not bool(torch.isfinite(clean_tokens).all()):
    raise PoseCostError("clean tokens contain non-finite values")
  return clean_tokens


__all__ = [
  "ActivePoseGoal",
  "PoseCost",
  "PoseCostError",
  "PoseCostWeights",
  "TimedPoseGoal",
  "WorldPoses",
  "active_pose_goals",
  "world_poses_from_normalized_state",
]
