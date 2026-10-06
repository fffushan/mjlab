"""Owned D5 latent planning over the frozen reference-free diffusion seam.

The planner is deliberately separate from the generic scheduler and from the
VAE decoder.  It snapshots eight executed state/latent pairs, the measured
current state, task inputs, source identity and a request-local Torch random
state.  A synchronous planner callable reconstructs only those snapshots,
reuses the D3 conditioning window and the D4 hooks, and returns raw 32-D latent
rows.  It never publishes decoder actions and never calls an encoder or
decoder.

The wire representation contains tuples and scalar values only.  It is safe to
put in :class:`~.scheduler.PlannerRequest.context`; no live model, simulator,
scoring ledger or unbounded context registry is needed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, cast

import numpy as np
import torch
from torch import Tensor

from .conditioned_continuation import (
  SparseTaskSpec,
  TaskConditioning,
  TaskSource,
)
from .contract import DEFAULT_CONTRACT, DiffusionContract
from .handover import (
  ContinuationPlan,
  ExecutedPair,
  HandoverHistory,
  build_reference_free_plan,
)
from .inpainting import TimedStateKeyframe, inpaint_keyframes
from .pose_costs import PoseCostWeights, TimedPoseGoal
from .projection import ProjectionBundle
from .sampler import SamplerConditions, SamplerConfig, sample_trajectory
from .schedule import DiffusionSchedule, InferenceGrid, build_inference_grid
from .scheduler import FrozenContext, Plan, PlannerRequest
from .state import WorldState

LATENT_PLANNING_SCHEMA = "mjlab-d5-latent-planning-v1"
RAW_LATENT_REPRESENTATION = "latent/raw/32"
ROOT_FRAME = "world"

_STATE_FIELDS = (
  "root_position",
  "root_quaternion_wxyz",
  "root_linear_velocity",
  "root_angular_velocity",
  "body_positions",
  "body_linear_velocities",
)


class LatentPlanningError(ValueError):
  """A latent planning packet, identity or result is invalid."""


def _finite_float(value: object, name: str) -> float:
  if isinstance(value, (bool, np.bool_)):
    raise LatentPlanningError(f"{name} must be a finite scalar number")
  try:
    scalar = float(cast(Any, value))
  except (TypeError, ValueError, OverflowError) as exc:
    raise LatentPlanningError(f"{name} must be a finite scalar number") from exc
  if not math.isfinite(scalar):
    raise LatentPlanningError(f"{name} must be finite")
  return scalar


def _nonnegative_integer(value: object, name: str) -> int:
  if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
    raise LatentPlanningError(f"{name} must be a non-negative integer")
  result = int(value)
  if result < 0:
    raise LatentPlanningError(f"{name} must be non-negative")
  return result


def _identity(value: object, name: str) -> str:
  if not isinstance(value, str) or not value.strip():
    raise LatentPlanningError(f"{name} must be a non-empty identity string")
  return value


def _tuple_vector(value: object, name: str, width: int) -> tuple[float, ...]:
  array = np.asarray(value, dtype=np.float64)
  if array.shape != (width,) or not np.isfinite(array).all():
    raise LatentPlanningError(f"{name} must be finite with shape [{width}]")
  return tuple(float(item) for item in array)


def _tuple_body_array(value: object, name: str) -> tuple[tuple[float, ...], ...]:
  array = np.asarray(value, dtype=np.float64)
  if array.shape != (DEFAULT_CONTRACT.body_count, 3) or not np.isfinite(array).all():
    raise LatentPlanningError(
      f"{name} must be finite with shape [{DEFAULT_CONTRACT.body_count}, 3]"
    )
  return tuple(tuple(float(item) for item in row) for row in array)


def _world_state_copy(state: WorldState) -> WorldState:
  if not isinstance(state, WorldState):
    raise LatentPlanningError("state must be a complete WorldState")
  if state.root_position.ndim != 1:
    raise LatentPlanningError("state must be one unbatched measurement")
  return WorldState(
    **{
      name: np.array(getattr(state, name), dtype=np.float64, copy=True)
      for name in _STATE_FIELDS
    }
  )


def _mapping(value: object, name: str) -> Mapping[str, object]:
  if not isinstance(value, Mapping):
    raise LatentPlanningError(f"{name} must be a mapping")
  return cast(Mapping[str, object], value)


@dataclass(frozen=True, slots=True)
class StateSnapshot:
  """Tuple-backed immutable copy of one unbatched world state."""

  root_position: tuple[float, ...]
  root_quaternion_wxyz: tuple[float, ...]
  root_linear_velocity: tuple[float, ...]
  root_angular_velocity: tuple[float, ...]
  body_positions: tuple[tuple[float, ...], ...]
  body_linear_velocities: tuple[tuple[float, ...], ...]

  def __post_init__(self) -> None:
    object.__setattr__(
      self, "root_position", _tuple_vector(self.root_position, "root_position", 3)
    )
    object.__setattr__(
      self,
      "root_quaternion_wxyz",
      _tuple_vector(self.root_quaternion_wxyz, "root_quaternion_wxyz", 4),
    )
    object.__setattr__(
      self,
      "root_linear_velocity",
      _tuple_vector(self.root_linear_velocity, "root_linear_velocity", 3),
    )
    object.__setattr__(
      self,
      "root_angular_velocity",
      _tuple_vector(self.root_angular_velocity, "root_angular_velocity", 3),
    )
    object.__setattr__(
      self, "body_positions", _tuple_body_array(self.body_positions, "body_positions")
    )
    object.__setattr__(
      self,
      "body_linear_velocities",
      _tuple_body_array(self.body_linear_velocities, "body_linear_velocities"),
    )
    quaternion_norm = np.linalg.norm(self.root_quaternion_wxyz)
    if quaternion_norm < 1.0e-12:
      raise LatentPlanningError("root quaternion has zero norm")

  @classmethod
  def from_world_state(cls, state: WorldState) -> "StateSnapshot":
    state = _world_state_copy(state)
    return cls(
      tuple(state.root_position),
      tuple(state.root_quaternion_wxyz),
      tuple(state.root_linear_velocity),
      tuple(state.root_angular_velocity),
      tuple(tuple(row) for row in state.body_positions),
      tuple(tuple(row) for row in state.body_linear_velocities),
    )

  @classmethod
  def from_wire(cls, payload: Mapping[str, object]) -> "StateSnapshot":
    fields = _mapping(payload, "state snapshot")
    if set(fields) != set(_STATE_FIELDS):
      raise LatentPlanningError(
        "state snapshot must contain exactly the six state fields"
      )
    return cls(
      tuple(cast(Sequence[float], fields["root_position"])),
      tuple(cast(Sequence[float], fields["root_quaternion_wxyz"])),
      tuple(cast(Sequence[float], fields["root_linear_velocity"])),
      tuple(cast(Sequence[float], fields["root_angular_velocity"])),
      tuple(
        tuple(row) for row in cast(Sequence[Sequence[float]], fields["body_positions"])
      ),
      tuple(
        tuple(row)
        for row in cast(Sequence[Sequence[float]], fields["body_linear_velocities"])
      ),
    )

  def as_wire(self) -> dict[str, object]:
    return {
      "root_position": list(self.root_position),
      "root_quaternion_wxyz": list(self.root_quaternion_wxyz),
      "root_linear_velocity": list(self.root_linear_velocity),
      "root_angular_velocity": list(self.root_angular_velocity),
      "body_positions": [list(row) for row in self.body_positions],
      "body_linear_velocities": [list(row) for row in self.body_linear_velocities],
    }

  def to_world_state(self) -> WorldState:
    return WorldState(
      np.asarray(self.root_position, dtype=np.float64).copy(),
      np.asarray(self.root_quaternion_wxyz, dtype=np.float64).copy(),
      np.asarray(self.root_linear_velocity, dtype=np.float64).copy(),
      np.asarray(self.root_angular_velocity, dtype=np.float64).copy(),
      np.asarray(self.body_positions, dtype=np.float64).copy(),
      np.asarray(self.body_linear_velocities, dtype=np.float64).copy(),
    )


@dataclass(frozen=True, slots=True)
class PlanningPair:
  """One immutable measured pre-action state and executed raw latent pair."""

  state: StateSnapshot
  latent: tuple[float, ...]

  def __post_init__(self) -> None:
    if not isinstance(self.state, StateSnapshot):
      raise LatentPlanningError("planning pair state must be a StateSnapshot")
    latent = np.asarray(self.latent, dtype=np.float64)
    if (
      latent.shape != (DEFAULT_CONTRACT.latent_dimension,)
      or not np.isfinite(latent).all()
    ):
      raise LatentPlanningError("executed latent must be finite with shape [32]")
    object.__setattr__(self, "latent", tuple(float(item) for item in latent))

  @classmethod
  def from_pair(cls, value: object) -> "PlanningPair":
    if isinstance(value, PlanningPair):
      return cls(value.state, value.latent)
    if isinstance(value, ExecutedPair):
      return cls(StateSnapshot.from_world_state(value.state), tuple(value.latent))
    if isinstance(value, (tuple, list)) and len(value) == 2:
      state, latent = value
      return cls(
        StateSnapshot.from_world_state(cast(WorldState, state)),
        tuple(cast(Sequence[float], latent)),
      )
    raise LatentPlanningError(
      "history must contain ExecutedPair values or (WorldState, latent) pairs"
    )

  def as_wire(self) -> dict[str, object]:
    return {"state": self.state.as_wire(), "latent": list(self.latent)}

  @classmethod
  def from_wire(cls, payload: Mapping[str, object]) -> "PlanningPair":
    fields = _mapping(payload, "planning pair")
    if set(fields) != {"state", "latent"}:
      raise LatentPlanningError("planning pair must contain state and latent")
    return cls(
      StateSnapshot.from_wire(_mapping(fields["state"], "planning pair state")),
      tuple(cast(Sequence[float], fields["latent"])),
    )

  def to_executed_pair(self) -> ExecutedPair:
    return ExecutedPair(
      self.state.to_world_state(), np.asarray(self.latent, dtype=np.float64)
    )


@dataclass(frozen=True, slots=True)
class TaskWeightsSnapshot:
  """Immutable D4 pose-cost weights used by a planning request."""

  root_position: float
  root_orientation: float
  body_positions: float
  body_coefficients: tuple[float, ...] | None = None

  def __post_init__(self) -> None:
    values = (self.root_position, self.root_orientation, self.body_positions)
    normalized = tuple(_finite_float(value, "pose-cost weight") for value in values)
    if any(value < 0.0 for value in normalized):
      raise LatentPlanningError("pose-cost weights must be non-negative")
    if not any(value > 0.0 for value in normalized):
      raise LatentPlanningError("at least one pose-cost weight must be positive")
    object.__setattr__(self, "root_position", normalized[0])
    object.__setattr__(self, "root_orientation", normalized[1])
    object.__setattr__(self, "body_positions", normalized[2])
    if self.body_coefficients is not None:
      coefficients = np.asarray(self.body_coefficients, dtype=np.float64)
      if (
        coefficients.shape != (DEFAULT_CONTRACT.body_count,)
        or not np.isfinite(coefficients).all()
      ):
        raise LatentPlanningError(
          "body coefficients must have shape [20] and be finite"
        )
      if np.any(coefficients < 0.0):
        raise LatentPlanningError("body coefficients must be non-negative")
      object.__setattr__(
        self, "body_coefficients", tuple(float(item) for item in coefficients)
      )

  @classmethod
  def from_weights(cls, weights: PoseCostWeights) -> "TaskWeightsSnapshot":
    if not isinstance(weights, PoseCostWeights):
      raise LatentPlanningError("weights must be a PoseCostWeights instance")
    return cls(
      weights.root_position,
      weights.root_orientation,
      weights.body_positions,
      tuple(weights.coefficients),
    )

  def as_wire(self) -> dict[str, object]:
    return {
      "root_position": self.root_position,
      "root_orientation": self.root_orientation,
      "body_positions": self.body_positions,
      "body_coefficients": self.body_coefficients,
    }

  @classmethod
  def from_wire(cls, payload: Mapping[str, object]) -> "TaskWeightsSnapshot":
    fields = _mapping(payload, "task weights")
    return cls(
      cast(float, fields.get("root_position", 0.0)),
      cast(float, fields.get("root_orientation", 0.0)),
      cast(float, fields.get("body_positions", 0.0)),
      None
      if fields.get("body_coefficients") is None
      else tuple(cast(Sequence[float], fields["body_coefficients"])),
    )

  def to_weights(self) -> PoseCostWeights:
    return PoseCostWeights(
      self.root_position,
      self.root_orientation,
      self.body_positions,
      self.body_coefficients,
    )


@dataclass(frozen=True, slots=True)
class TaskTargetSnapshot:
  """Immutable absolute-tick D4 target snapshot."""

  tick: int
  state: StateSnapshot
  provenance: str

  def __post_init__(self) -> None:
    object.__setattr__(self, "tick", _nonnegative_integer(self.tick, "target tick"))
    if not isinstance(self.state, StateSnapshot):
      raise LatentPlanningError("task target state must be a StateSnapshot")
    object.__setattr__(
      self, "provenance", _identity(self.provenance, "target provenance")
    )

  def as_wire(self) -> dict[str, object]:
    return {
      "tick": self.tick,
      "state": self.state.as_wire(),
      "provenance": self.provenance,
    }

  @classmethod
  def from_wire(cls, payload: Mapping[str, object]) -> "TaskTargetSnapshot":
    fields = _mapping(payload, "task target")
    return cls(
      cast(int, fields["tick"]),
      StateSnapshot.from_wire(_mapping(fields["state"], "task target state")),
      cast(str, fields["provenance"]),
    )

  @classmethod
  def from_target(
    cls, target: TimedStateKeyframe | TimedPoseGoal
  ) -> "TaskTargetSnapshot":
    if not isinstance(target, (TimedStateKeyframe, TimedPoseGoal)):
      raise LatentPlanningError("task target must be a timed keyframe or pose goal")
    return cls(
      target.tick, StateSnapshot.from_world_state(target.state), target.provenance
    )

  def to_keyframe(self) -> TimedStateKeyframe:
    return TimedStateKeyframe(self.tick, self.state.to_world_state(), self.provenance)

  def to_goal(self) -> TimedPoseGoal:
    return TimedPoseGoal(self.tick, self.state.to_world_state(), self.provenance)


@dataclass(frozen=True, slots=True)
class TaskSnapshot:
  """Serializable D4 settings and absolute targets, without a scoring ledger."""

  mode: str = "unguided"
  keyframes: tuple[TaskTargetSnapshot, ...] = ()
  goals: tuple[TaskTargetSnapshot, ...] = ()
  weights: TaskWeightsSnapshot | None = None
  strength: float = 0.0
  revision: str = "none"
  config_sha256: str = (
    "0000000000000000000000000000000000000000000000000000000000000000"
  )
  schema_version: str = "mjlab-d4-sparse-task-v1"

  def __post_init__(self) -> None:
    if self.mode not in {"unguided", "keyframe", "poseguided", "combined"}:
      raise LatentPlanningError(f"unsupported task mode {self.mode!r}")
    object.__setattr__(self, "strength", _finite_float(self.strength, "task strength"))
    if self.strength < 0.0:
      raise LatentPlanningError("task strength must be non-negative")
    object.__setattr__(self, "revision", _identity(self.revision, "task revision"))
    if not isinstance(self.config_sha256, str) or len(self.config_sha256) != 64:
      raise LatentPlanningError("task config sha256 must contain 64 characters")
    if self.schema_version != "mjlab-d4-sparse-task-v1":
      raise LatentPlanningError(f"unsupported task schema {self.schema_version!r}")
    keyframes = tuple(self.keyframes)
    goals = tuple(self.goals)
    if not all(isinstance(item, TaskTargetSnapshot) for item in keyframes + goals):
      raise LatentPlanningError("task targets must be TaskTargetSnapshot values")
    if len({item.tick for item in keyframes}) != len(keyframes):
      raise LatentPlanningError("duplicate hard-keyframe ticks are not accepted")
    if len({item.tick for item in goals}) != len(goals):
      raise LatentPlanningError("duplicate pose-goal ticks are not accepted")
    object.__setattr__(
      self, "keyframes", tuple(sorted(keyframes, key=lambda item: item.tick))
    )
    object.__setattr__(self, "goals", tuple(sorted(goals, key=lambda item: item.tick)))
    if self.mode == "unguided" and (keyframes or goals or self.weights is not None):
      raise LatentPlanningError(
        "unguided task snapshots cannot carry targets or weights"
      )
    if self.mode == "keyframe" and (not keyframes or goals or self.weights is not None):
      raise LatentPlanningError("keyframe snapshots require hard keyframes only")
    if self.mode == "poseguided" and (
      keyframes or not goals or self.weights is None or self.strength <= 0.0
    ):
      raise LatentPlanningError(
        "poseguided snapshots require goals, weights and strength"
      )
    if self.mode == "combined" and (
      not keyframes or not goals or self.weights is None or self.strength <= 0.0
    ):
      raise LatentPlanningError(
        "combined snapshots require hard keyframes, goals, weights and strength"
      )
    if self.mode in {"unguided", "keyframe"} and self.strength != 0.0:
      raise LatentPlanningError(
        "unguided and keyframe snapshots do not accept guidance"
      )

  @classmethod
  def from_inputs(
    cls,
    task: object | None,
    *,
    revision: str | None = None,
  ) -> "TaskSnapshot":
    if task is None:
      return cls(revision=revision or "none")
    if isinstance(task, TaskConditioning):
      raise LatentPlanningError(
        "a live TaskConditioning ledger cannot cross the planning request boundary"
      )
    if isinstance(task, TaskSnapshot):
      if revision is None or revision == task.revision:
        return cls(
          mode=task.mode,
          keyframes=task.keyframes,
          goals=task.goals,
          weights=task.weights,
          strength=task.strength,
          revision=task.revision,
          config_sha256=task.config_sha256,
          schema_version=task.schema_version,
        )
      return cls(
        mode=task.mode,
        keyframes=task.keyframes,
        goals=task.goals,
        weights=task.weights,
        strength=task.strength,
        revision=revision,
        config_sha256=task.config_sha256,
        schema_version=task.schema_version,
      )
    if isinstance(task, SparseTaskSpec):
      resolved_revision = revision or task.config_sha256
      return cls(
        mode=task.mode,
        keyframes=tuple(
          TaskTargetSnapshot.from_target(item) for item in task.keyframes
        ),
        goals=tuple(TaskTargetSnapshot.from_target(item) for item in task.goals),
        weights=None
        if task.weights is None
        else TaskWeightsSnapshot.from_weights(task.weights),
        strength=task.strength,
        revision=resolved_revision,
        config_sha256=task.config_sha256,
        schema_version=task.schema_version,
      )
    fields = _mapping(task, "task inputs")
    mode = fields.get("mode", "unguided")
    keyframes = tuple(
      _target_from_input(item, hard=True)
      for item in _sequence(fields.get("keyframes", ()))
    )
    goals = tuple(
      _target_from_input(item, hard=False)
      for item in _sequence(fields.get("goals", ()))
    )
    weights_value = fields.get("weights")
    weights = None
    if weights_value is not None:
      if isinstance(weights_value, TaskWeightsSnapshot):
        weights = weights_value
      elif isinstance(weights_value, PoseCostWeights):
        weights = TaskWeightsSnapshot.from_weights(weights_value)
      else:
        weights = TaskWeightsSnapshot.from_wire(_mapping(weights_value, "task weights"))
    config_sha = fields.get("config_sha256", "0" * 64)
    return cls(
      mode=cast(str, mode),
      keyframes=keyframes,
      goals=goals,
      weights=weights,
      strength=cast(float, fields.get("strength", 0.0)),
      revision=revision or cast(str | None, fields.get("revision")) or str(config_sha),
      config_sha256=cast(str, config_sha),
      schema_version=cast(str, fields.get("schema_version", "mjlab-d4-sparse-task-v1")),
    )

  @classmethod
  def from_wire(cls, payload: Mapping[str, object]) -> "TaskSnapshot":
    fields = _mapping(payload, "task snapshot")
    keyframes = tuple(
      TaskTargetSnapshot.from_wire(_mapping(item, "hard-keyframe snapshot"))
      for item in _sequence(fields.get("keyframes", ()))
    )
    goals = tuple(
      TaskTargetSnapshot.from_wire(_mapping(item, "pose-goal snapshot"))
      for item in _sequence(fields.get("goals", ()))
    )
    weights_value = fields.get("weights")
    weights = (
      None
      if weights_value is None
      else TaskWeightsSnapshot.from_wire(_mapping(weights_value, "task weights"))
    )
    return cls(
      mode=cast(str, fields.get("mode", "unguided")),
      keyframes=keyframes,
      goals=goals,
      weights=weights,
      strength=cast(float, fields.get("strength", 0.0)),
      revision=cast(str, fields.get("revision", "none")),
      config_sha256=cast(str, fields.get("config_sha256", "0" * 64)),
      schema_version=cast(str, fields.get("schema_version", "mjlab-d4-sparse-task-v1")),
    )

  @property
  def active(self) -> bool:
    return bool(self.keyframes or self.goals)

  @property
  def keyframe_ticks(self) -> tuple[int, ...]:
    return tuple(item.tick for item in self.keyframes)

  @property
  def goal_ticks(self) -> tuple[int, ...]:
    return tuple(item.tick for item in self.goals)

  def as_wire(self) -> dict[str, object]:
    return {
      "mode": self.mode,
      "keyframes": [item.as_wire() for item in self.keyframes],
      "goals": [item.as_wire() for item in self.goals],
      "weights": None if self.weights is None else self.weights.as_wire(),
      "strength": self.strength,
      "revision": self.revision,
      "config_sha256": self.config_sha256,
      "schema_version": self.schema_version,
    }

  def to_spec(self) -> SparseTaskSpec:
    source = TaskSource(
      kind="synthetic_fixture",
      qualification="unverified",
      verification="loader_structural_only",
      provenance="owned D5 planning snapshot; scoring stays in the owner thread",
    )
    try:
      return SparseTaskSpec(
        mode=self.mode,
        keyframes=tuple(item.to_keyframe() for item in self.keyframes),
        goals=tuple(item.to_goal() for item in self.goals),
        weights=None if self.weights is None else self.weights.to_weights(),
        strength=self.strength,
        source=source,
        config_sha256=self.config_sha256,
        schema_version=self.schema_version,
      )
    except (TypeError, ValueError) as exc:
      raise LatentPlanningError(
        f"task snapshot is not a valid D4 specification: {exc}"
      ) from exc


def _sequence(value: object) -> tuple[object, ...]:
  if value is None:
    return ()
  if isinstance(value, (str, bytes, Mapping)):
    raise LatentPlanningError("task target list must be a sequence")
  try:
    return tuple(cast(Iterable[object], value))
  except TypeError as exc:
    raise LatentPlanningError("task target list must be a sequence") from exc


def _target_from_input(value: object, *, hard: bool) -> TaskTargetSnapshot:
  if isinstance(value, TaskTargetSnapshot):
    return value
  if hard and isinstance(value, TimedStateKeyframe):
    return TaskTargetSnapshot.from_target(value)
  if not hard and isinstance(value, TimedPoseGoal):
    return TaskTargetSnapshot.from_target(value)
  fields = _mapping(value, "task target")
  state_value = fields.get("state")
  if isinstance(state_value, StateSnapshot):
    state = state_value
  elif isinstance(state_value, WorldState):
    state = StateSnapshot.from_world_state(state_value)
  else:
    state = StateSnapshot.from_wire(_mapping(state_value, "task target state"))
  return TaskTargetSnapshot(
    cast(int, fields["tick"]), state, cast(str, fields.get("provenance", "snapshot"))
  )


def _projection_identity(
  projection: ProjectionBundle, contract: DiffusionContract
) -> str:
  projection_contract = getattr(projection, "contract", None)
  if projection_contract is None or not hasattr(projection_contract, "identity_hash"):
    raise LatentPlanningError("projection bundle does not carry a contract identity")
  if projection_contract.identity_hash() != contract.identity_hash():
    raise LatentPlanningError(
      "projection contract identity differs from active contract"
    )
  fields = {
    "contract": contract.identity_hash(),
    "statistics": str(projection.statistics_sha256),
    "matrix": str(projection.matrix_sha256),
    "pseudoinverse": str(projection.pseudoinverse_sha256),
  }
  return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def _capture_generator(generator: torch.Generator) -> tuple[tuple[int, ...], str]:
  if not isinstance(generator, torch.Generator):
    raise LatentPlanningError("rng must be a torch.Generator")
  state = generator.get_state().detach().to(device="cpu", dtype=torch.uint8)
  return tuple(int(value) for value in state.tolist()), str(generator.device)


def _restore_generator(state: Sequence[int], device: str) -> torch.Generator:
  if not state:
    raise LatentPlanningError("request-local RNG state is empty")
  try:
    generator = torch.Generator(device=device)
    tensor = torch.as_tensor(tuple(int(value) for value in state), dtype=torch.uint8)
    generator.set_state(tensor)
  except (RuntimeError, TypeError, ValueError) as exc:
    raise LatentPlanningError("request-local RNG state is invalid") from exc
  return generator


def _snapshot_plan(
  plan: ContinuationPlan,
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[bool, ...], ...]]:
  if plan.clean_tokens.shape != (
    DEFAULT_CONTRACT.window_steps,
    DEFAULT_CONTRACT.token_dimension,
  ):
    raise LatentPlanningError(
      "conditioning tokens do not match the active 41x167 window"
    )
  if (
    plan.clean_mask.shape != plan.clean_tokens.shape
    or plan.clean_mask.dtype != torch.bool
  ):
    raise LatentPlanningError(
      "conditioning mask does not match the active 41x167 window"
    )
  tokens = plan.clean_tokens.detach().to(device="cpu", dtype=torch.float32).numpy()
  mask = plan.clean_mask.detach().to(device="cpu", dtype=torch.bool).numpy()
  if not np.isfinite(tokens).all():
    raise LatentPlanningError("conditioning tokens are non-finite")
  return (
    tuple(tuple(float(item) for item in row) for row in tokens),
    tuple(tuple(bool(item) for item in row) for row in mask),
  )


@dataclass(frozen=True, slots=True)
class LatentPlanningPacket:
  """Owned, serializable input to one synchronous latent-planning call."""

  pairs: tuple[PlanningPair, ...]
  current_state: StateSnapshot
  source_timestamp: float
  source_tick: int
  source_frame: int
  period_seconds: float
  root_frame: str
  contract_identity: str
  vae_identity: str
  projection_identity: str
  model_identity: str
  task: TaskSnapshot | None
  clean_tokens: tuple[tuple[float, ...], ...]
  clean_mask: tuple[tuple[bool, ...], ...]
  rng_state: tuple[int, ...]
  rng_device: str

  def __post_init__(self) -> None:
    pairs = tuple(self.pairs)
    if len(pairs) != DEFAULT_CONTRACT.past_steps:
      raise LatentPlanningError(
        f"planning packet requires exactly {DEFAULT_CONTRACT.past_steps} executed pairs"
      )
    if not all(isinstance(pair, PlanningPair) for pair in pairs):
      raise LatentPlanningError("planning packet pairs must be PlanningPair values")
    object.__setattr__(self, "pairs", pairs)
    if not isinstance(self.current_state, StateSnapshot):
      raise LatentPlanningError("planning packet current state must be a StateSnapshot")
    timestamp = _finite_float(self.source_timestamp, "source_timestamp")
    if timestamp < 0.0:
      raise LatentPlanningError("source_timestamp must be non-negative")
    object.__setattr__(self, "source_timestamp", timestamp)
    object.__setattr__(
      self, "source_tick", _nonnegative_integer(self.source_tick, "source_tick")
    )
    object.__setattr__(
      self, "source_frame", _nonnegative_integer(self.source_frame, "source_frame")
    )
    period = _finite_float(self.period_seconds, "period_seconds")
    if period <= 0.0:
      raise LatentPlanningError("period_seconds must be positive")
    object.__setattr__(self, "period_seconds", period)
    if self.root_frame != ROOT_FRAME:
      raise LatentPlanningError(
        f"root_frame must be {ROOT_FRAME!r}, got {self.root_frame!r}"
      )
    for name in (
      "contract_identity",
      "vae_identity",
      "projection_identity",
      "model_identity",
      "rng_device",
    ):
      object.__setattr__(self, name, _identity(getattr(self, name), name))
    if self.task is not None and not isinstance(self.task, TaskSnapshot):
      raise LatentPlanningError("planning packet task must be a TaskSnapshot or None")
    tokens = np.asarray(self.clean_tokens, dtype=np.float64)
    mask = np.asarray(self.clean_mask, dtype=bool)
    expected = (DEFAULT_CONTRACT.window_steps, DEFAULT_CONTRACT.token_dimension)
    if tokens.shape != expected or mask.shape != expected:
      raise LatentPlanningError(
        "planning packet conditioning must have shape [41, 167]"
      )
    if not np.isfinite(tokens).all():
      raise LatentPlanningError("planning packet conditioning is non-finite")
    object.__setattr__(
      self,
      "clean_tokens",
      tuple(tuple(float(item) for item in row) for row in tokens),
    )
    object.__setattr__(
      self,
      "clean_mask",
      tuple(tuple(bool(item) for item in row) for row in mask),
    )
    rng = tuple(
      _nonnegative_integer(value, "rng state byte") for value in self.rng_state
    )
    if not rng or any(value > 255 for value in rng):
      raise LatentPlanningError("rng_state must contain non-empty uint8 values")
    object.__setattr__(self, "rng_state", rng)

  @property
  def history(self) -> tuple[PlanningPair, ...]:
    return self.pairs

  @property
  def executed_pairs(self) -> tuple[PlanningPair, ...]:
    return self.pairs

  @property
  def ticks(self) -> tuple[int, ...]:
    return tuple(
      self.source_tick + row for row in range(DEFAULT_CONTRACT.future_steps + 1)
    )

  @property
  def input_snapshot(self) -> np.ndarray:
    snapshot = np.asarray(self.clean_tokens, dtype=np.float64).copy()
    snapshot.setflags(write=False)
    return snapshot

  def current_world_state(self) -> WorldState:
    return self.current_state.to_world_state()

  def as_wire(self) -> dict[str, object]:
    return {
      "schema": LATENT_PLANNING_SCHEMA,
      "pairs": [pair.as_wire() for pair in self.pairs],
      "current_state": self.current_state.as_wire(),
      "source_timestamp": self.source_timestamp,
      "source_tick": self.source_tick,
      "source_frame": self.source_frame,
      "period_seconds": self.period_seconds,
      "root_frame": self.root_frame,
      "contract_identity": self.contract_identity,
      "vae_identity": self.vae_identity,
      "projection_identity": self.projection_identity,
      "model_identity": self.model_identity,
      "task": None if self.task is None else self.task.as_wire(),
      "clean_tokens": [list(row) for row in self.clean_tokens],
      "clean_mask": [list(row) for row in self.clean_mask],
      "rng_state": list(self.rng_state),
      "rng_device": self.rng_device,
    }

  def as_context(self) -> dict[str, object]:
    """Return the plain mapping intended for ``PlannerRequest.context``."""
    return self.as_wire()

  def to_wire(self) -> dict[str, object]:
    """Alias for the request-context wire mapping."""
    return self.as_wire()

  @classmethod
  def from_wire(cls, payload: Mapping[str, object]) -> "LatentPlanningPacket":
    fields = _mapping(payload, "latent planning packet")
    if fields.get("schema") != LATENT_PLANNING_SCHEMA:
      raise LatentPlanningError("latent planning packet schema is incompatible")
    pairs = tuple(
      PlanningPair.from_wire(_mapping(item, "planning packet pair"))
      for item in _sequence(fields.get("pairs"))
    )
    task_value = fields.get("task")
    task = (
      None
      if task_value is None
      else TaskSnapshot.from_wire(_mapping(task_value, "planning packet task"))
    )
    return cls(
      pairs=pairs,
      current_state=StateSnapshot.from_wire(
        _mapping(fields["current_state"], "planning packet current state")
      ),
      source_timestamp=cast(float, fields["source_timestamp"]),
      source_tick=cast(int, fields["source_tick"]),
      source_frame=cast(int, fields["source_frame"]),
      period_seconds=cast(float, fields["period_seconds"]),
      root_frame=cast(str, fields["root_frame"]),
      contract_identity=cast(str, fields["contract_identity"]),
      vae_identity=cast(str, fields["vae_identity"]),
      projection_identity=cast(str, fields["projection_identity"]),
      model_identity=cast(str, fields["model_identity"]),
      task=task,
      clean_tokens=tuple(
        tuple(row) for row in cast(Sequence[Sequence[float]], fields["clean_tokens"])
      ),
      clean_mask=tuple(
        tuple(row) for row in cast(Sequence[Sequence[bool]], fields["clean_mask"])
      ),
      rng_state=tuple(cast(Sequence[int], fields["rng_state"])),
      rng_device=cast(str, fields["rng_device"]),
    )

  @classmethod
  def from_request(cls, request: PlannerRequest) -> "LatentPlanningPacket":
    if not isinstance(request, PlannerRequest):
      raise LatentPlanningError(
        "planner input must be a PlannerRequest or LatentPlanningPacket"
      )
    if request.context is None:
      raise LatentPlanningError("planner request has no latent planning context")
    packet = cls.from_wire(_mapping(request.context, "planner request context"))
    if (
      packet.source_tick != request.source_tick
      or packet.source_timestamp != request.source_timestamp
    ):
      raise LatentPlanningError(
        "request identity differs from its latent planning context"
      )
    return packet


def build_planning_packet(
  history: HandoverHistory | Iterable[object],
  current_state: WorldState,
  *,
  projection: ProjectionBundle,
  source_timestamp: float,
  source_tick: int,
  source_frame: int,
  period_seconds: float = DEFAULT_CONTRACT.period_seconds,
  root_frame: str = ROOT_FRAME,
  task_inputs: object | None = None,
  task: object | None = None,
  task_revision: str | None = None,
  rng: torch.Generator | None = None,
  generator: torch.Generator | None = None,
  contract: DiffusionContract = DEFAULT_CONTRACT,
  vae_identity: str = "unspecified-vae",
  model_identity: str = "unspecified-model",
  projection_identity: str | None = None,
) -> LatentPlanningPacket:
  """Snapshot measured history and immutable planning inputs for one request."""
  if task_inputs is not None and task is not None:
    raise LatentPlanningError("pass only one of task_inputs or task")
  task_value = task_inputs if task_inputs is not None else task
  if rng is not None and generator is not None:
    raise LatentPlanningError("pass only one of rng or generator")
  random = rng if rng is not None else generator
  if random is None:
    raise LatentPlanningError("an explicit request-local torch.Generator is required")
  if (
    contract.schema_version != DEFAULT_CONTRACT.schema_version
    or contract.projection_rows != 0
  ):
    raise LatentPlanningError("latent planning requires the active no-mixture contract")
  if contract.identity_hash() != DEFAULT_CONTRACT.identity_hash():
    raise LatentPlanningError(
      "latent planning contract identity is not the active identity"
    )
  if (
    abs(_finite_float(period_seconds, "period_seconds") - contract.period_seconds)
    > 1.0e-12
  ):
    raise LatentPlanningError("period_seconds differs from the active contract")
  if root_frame != ROOT_FRAME:
    raise LatentPlanningError(f"root_frame must be {ROOT_FRAME!r}")
  source_timestamp = _finite_float(source_timestamp, "source_timestamp")
  if source_timestamp < 0.0:
    raise LatentPlanningError("source_timestamp must be non-negative")
  source_tick = _nonnegative_integer(source_tick, "source_tick")
  source_frame = _nonnegative_integer(source_frame, "source_frame")
  current_snapshot = StateSnapshot.from_world_state(current_state)
  if isinstance(history, HandoverHistory):
    values = tuple(history.pairs)
  else:
    values = tuple(history)
  pairs = tuple(PlanningPair.from_pair(value) for value in values)
  if len(pairs) != contract.past_steps:
    raise LatentPlanningError(
      f"planning requires exactly {contract.past_steps} measured pre-action pairs"
    )
  snapshot_task = TaskSnapshot.from_inputs(task_value, revision=task_revision)
  identity = _projection_identity(projection, contract)
  supplied_projection_identity = (
    identity
    if projection_identity is None
    else _identity(projection_identity, "projection_identity")
  )
  if supplied_projection_identity != identity:
    raise LatentPlanningError(
      "projection identity differs from the supplied statistics"
    )
  history_copy = HandoverHistory(contract=contract)
  for pair in pairs:
    history_copy.push(
      pair.state.to_world_state(), np.asarray(pair.latent, dtype=np.float64)
    )
  base_plan = build_reference_free_plan(
    history_copy,
    current_snapshot.to_world_state(),
    projection=projection,
    tick=source_tick,
    contract=contract,
    device="cpu",
  )
  if snapshot_task.mode in {"keyframe", "combined"}:
    spec = snapshot_task.to_spec()
    base_plan = inpaint_keyframes(
      base_plan,
      current_snapshot.to_world_state(),
      spec.keyframes,
      projection=projection,
      contract=contract,
    )
  tokens, mask = _snapshot_plan(base_plan)
  rng_state, rng_device = _capture_generator(random)
  return LatentPlanningPacket(
    pairs=pairs,
    current_state=current_snapshot,
    source_timestamp=source_timestamp,
    source_tick=source_tick,
    source_frame=source_frame,
    period_seconds=contract.period_seconds,
    root_frame=root_frame,
    contract_identity=contract.identity_hash(),
    vae_identity=_identity(vae_identity, "vae_identity"),
    projection_identity=identity,
    model_identity=_identity(model_identity, "model_identity"),
    task=snapshot_task if task_value is not None or task_revision is not None else None,
    clean_tokens=tokens,
    clean_mask=mask,
    rng_state=rng_state,
    rng_device=rng_device,
  )


# Explicit aliases make the public seam discoverable without introducing a
# second representation.
make_planning_packet = build_planning_packet
LatentPlannerRequest = LatentPlanningPacket


@dataclass(frozen=True, slots=True)
class LatentPlan:
  """Raw denormalized latent rows for source tick through source tick + 32."""

  plan_sequence_id: int
  source_timestamp: float
  source_tick: int
  source_frame: int
  period_seconds: float
  raw_latents: np.ndarray
  contract_identity: str
  vae_identity: str
  projection_identity: str
  model_identity: str
  task_revision: str
  denoiser_calls: int
  task_mode: str

  def __post_init__(self) -> None:
    sequence = _nonnegative_integer(self.plan_sequence_id, "plan_sequence_id")
    if sequence == 0:
      raise LatentPlanningError("plan_sequence_id must be positive")
    object.__setattr__(self, "plan_sequence_id", sequence)
    timestamp = _finite_float(self.source_timestamp, "source_timestamp")
    if timestamp < 0.0:
      raise LatentPlanningError("source_timestamp must be non-negative")
    object.__setattr__(self, "source_timestamp", timestamp)
    object.__setattr__(
      self, "source_tick", _nonnegative_integer(self.source_tick, "source_tick")
    )
    object.__setattr__(
      self, "source_frame", _nonnegative_integer(self.source_frame, "source_frame")
    )
    period = _finite_float(self.period_seconds, "period_seconds")
    if period <= 0.0:
      raise LatentPlanningError("period_seconds must be positive")
    object.__setattr__(self, "period_seconds", period)
    latents = np.asarray(self.raw_latents, dtype=np.float64)
    expected = (DEFAULT_CONTRACT.future_steps + 1, DEFAULT_CONTRACT.latent_dimension)
    if latents.shape != expected or not np.isfinite(latents).all():
      raise LatentPlanningError("raw_latents must be finite with shape [33, 32]")
    latents = np.array(latents, dtype=np.float64, copy=True)
    latents.setflags(write=False)
    object.__setattr__(self, "raw_latents", latents)
    for name in (
      "contract_identity",
      "vae_identity",
      "projection_identity",
      "model_identity",
      "task_revision",
    ):
      object.__setattr__(self, name, _identity(getattr(self, name), name))
    calls = _nonnegative_integer(self.denoiser_calls, "denoiser_calls")
    object.__setattr__(self, "denoiser_calls", calls)
    if self.task_mode not in {"unguided", "keyframe", "poseguided", "combined"}:
      raise LatentPlanningError("latent plan task_mode is invalid")

  @property
  def valid_from_tick(self) -> int:
    return self.source_tick

  @property
  def valid_until_tick(self) -> int:
    return self.source_tick + DEFAULT_CONTRACT.future_steps

  @property
  def ticks(self) -> tuple[int, ...]:
    return tuple(range(self.valid_from_tick, self.valid_until_tick + 1))

  @property
  def latent_dimension(self) -> int:
    return DEFAULT_CONTRACT.latent_dimension

  @property
  def representation(self) -> str:
    return RAW_LATENT_REPRESENTATION

  @property
  def metadata(self) -> FrozenContext:
    return FrozenContext(
      {
        "schema": LATENT_PLANNING_SCHEMA,
        "contract": self.contract_identity,
        "representation": RAW_LATENT_REPRESENTATION,
        "kind": "latent",
        "semantic": "raw",
        "latent_dimension": DEFAULT_CONTRACT.latent_dimension,
        "latent_space": "raw",
        "source_tick": self.source_tick,
        "valid_from_tick": self.valid_from_tick,
        "valid_until_tick": self.valid_until_tick,
        "source_timestamp": self.source_timestamp,
        "source_frame": self.source_frame,
        "root_frame": ROOT_FRAME,
        "period_seconds": self.period_seconds,
        "vae_identity": self.vae_identity,
        "projection_identity": self.projection_identity,
        "model_identity": self.model_identity,
        "task_revision": self.task_revision,
        "task_mode": self.task_mode,
      }
    )

  @property
  def schema_metadata(self) -> FrozenContext:
    """Explicit metadata that prevents raw latents being read as commands."""
    return self.metadata

  @property
  def latents(self) -> np.ndarray:
    return self.raw_latents

  def row_for_tick(self, tick: int) -> np.ndarray:
    tick = _nonnegative_integer(tick, "execution tick")
    if tick < self.valid_from_tick or tick > self.valid_until_tick:
      raise LatentPlanningError("execution tick is outside the latent plan horizon")
    return np.array(self.raw_latents[tick - self.valid_from_tick], copy=True)

  def as_scheduler_plan(self) -> Plan:
    """Transport latents through the generic scheduler with explicit semantics."""
    return Plan(
      plan_sequence_id=self.plan_sequence_id,
      source_timestamp=self.source_timestamp,
      valid_from_tick=self.valid_from_tick,
      actions=self.raw_latents,
      schema_metadata=dict(self.metadata.items()),
    )

  def to_scheduler_plan(self) -> Plan:
    """Alias for the generic scheduler transport form."""
    return self.as_scheduler_plan()

  def as_base_plan(self) -> Plan:
    """Return the scheduler transport form without changing its semantics."""
    return self.as_scheduler_plan()

  def as_array(self) -> np.ndarray:
    return np.array(self.raw_latents, copy=True)


class LatentPlanner:
  """Synchronous CPU-callable planner owning denoiser and frozen transforms."""

  def __init__(
    self,
    *,
    denoiser: Callable[[Tensor, Tensor], Tensor],
    schedule: DiffusionSchedule,
    grid: InferenceGrid,
    projection: ProjectionBundle,
    contract: DiffusionContract = DEFAULT_CONTRACT,
    device: str | torch.device = "cpu",
    sampler_config: SamplerConfig | None = None,
    vae_identity: str | None = None,
    model_identity: str | None = None,
    expected_source_frame: int | None = None,
    expected_task_revision: str | None = None,
    expected_root_frame: str = ROOT_FRAME,
  ) -> None:
    if not callable(denoiser):
      raise LatentPlanningError("denoiser must be callable")
    if (
      contract.schema_version != DEFAULT_CONTRACT.schema_version
      or contract.projection_rows != 0
    ):
      raise LatentPlanningError(
        "latent planning requires the active no-mixture contract"
      )
    if contract.identity_hash() != DEFAULT_CONTRACT.identity_hash():
      raise LatentPlanningError(
        "latent planning contract identity is not the active identity"
      )
    if not isinstance(schedule, DiffusionSchedule):
      raise LatentPlanningError("schedule must be a DiffusionSchedule")
    if not isinstance(grid, InferenceGrid) or grid != build_inference_grid(schedule):
      raise LatentPlanningError(
        "latent planning requires the frozen 20-jump inference grid"
      )
    _projection_identity(projection, contract)
    try:
      resolved_device = torch.device(device)
      torch.empty(0, device=resolved_device)
    except (RuntimeError, ValueError) as exc:
      raise LatentPlanningError(f"invalid planner device {device!r}") from exc
    config = sampler_config or SamplerConfig(grid)
    if (
      config.grid != grid
      or config.eta != 0.0
      or config.guidance_strength != 0.0
      or config.start_level != 1000
      or config.initial_noise_scale != 1.0
    ):
      raise LatentPlanningError(
        "latent planning is pinned to the default 20-jump eta=0 sampler"
      )
    self.denoiser = denoiser
    self.schedule = schedule
    self.grid = grid
    self.projection = projection
    self.contract = contract
    self.device = resolved_device
    self.sampler_config = config
    self.vae_identity = (
      None if vae_identity is None else _identity(vae_identity, "vae_identity")
    )
    self.model_identity = (
      None if model_identity is None else _identity(model_identity, "model_identity")
    )
    self.expected_source_frame = (
      None
      if expected_source_frame is None
      else _nonnegative_integer(expected_source_frame, "expected_source_frame")
    )
    self.expected_task_revision = (
      None
      if expected_task_revision is None
      else _identity(expected_task_revision, "expected_task_revision")
    )
    if expected_root_frame != ROOT_FRAME:
      raise LatentPlanningError(f"expected_root_frame must be {ROOT_FRAME!r}")
    self.expected_root_frame = expected_root_frame
    self._projection_identity = _projection_identity(projection, contract)

  def __call__(
    self,
    request_or_packet: PlannerRequest | LatentPlanningPacket,
    *,
    plan_sequence_id: int | None = None,
  ) -> LatentPlan:
    return self.plan(request_or_packet, plan_sequence_id=plan_sequence_id)

  def plan_request(self, request: PlannerRequest) -> LatentPlan:
    return self.plan(request)

  def plan(
    self,
    request_or_packet: PlannerRequest | LatentPlanningPacket,
    *,
    plan_sequence_id: int | None = None,
  ) -> LatentPlan:
    if isinstance(request_or_packet, PlannerRequest):
      packet = LatentPlanningPacket.from_request(request_or_packet)
      sequence = request_or_packet.request_sequence_id
    elif isinstance(request_or_packet, LatentPlanningPacket):
      packet = request_or_packet
      sequence = 1 if plan_sequence_id is None else plan_sequence_id
    else:
      raise LatentPlanningError(
        "planner input must be a PlannerRequest or LatentPlanningPacket"
      )
    if plan_sequence_id is not None:
      sequence = plan_sequence_id
    self._validate_packet(packet)
    plan = ContinuationPlan(
      tick=packet.source_tick,
      clean_tokens=torch.as_tensor(
        packet.clean_tokens, dtype=torch.float32, device=self.device
      ).clone(),
      clean_mask=torch.as_tensor(
        packet.clean_mask, dtype=torch.bool, device=self.device
      ).clone(),
    )
    current_state = packet.current_world_state()
    task_spec = None if packet.task is None else packet.task.to_spec()
    sampler: Callable[..., tuple[Tensor, Any]] | None = None
    conditioning: TaskConditioning | None = None
    if task_spec is not None and task_spec.goals:
      conditioning = TaskConditioning(
        task_spec,
        projection=self.projection,
        contract=self.contract,
      )
      sampler = conditioning.sampler_factory(plan, current_state)
    conditions = SamplerConditions(plan.clean_tokens, plan.clean_mask)
    request_generator = _restore_generator(packet.rng_state, packet.rng_device)
    with torch.no_grad():
      if sampler is None:
        sample, diagnostics = sample_trajectory(
          self.denoiser,
          schedule=self.schedule,
          conditions=conditions,
          config=self.sampler_config,
          generator=request_generator,
        )
      else:
        sample, diagnostics = sampler(
          self.denoiser,
          schedule=self.schedule,
          conditions=conditions,
          config=self.sampler_config,
          generator=request_generator,
        )
    if not isinstance(sample, Tensor) or tuple(sample.shape) != (
      self.contract.window_steps,
      self.contract.token_dimension,
    ):
      raise LatentPlanningError("sampler returned an invalid 41x167 token window")
    normalized = (
      sample[self.contract.current_index :, self.contract.projected_state_dimension :]
      .detach()
      .to(device="cpu", dtype=torch.float32)
      .numpy()
      .astype(np.float64)
    )
    raw_latents = self.projection.denormalize_latent(normalized)
    if raw_latents.shape != (33, 32) or not np.isfinite(raw_latents).all():
      raise LatentPlanningError("sampler returned non-finite raw 32-D latents")
    denoiser_calls = int(getattr(diagnostics, "denoiser_calls", 0))
    task_mode = "unguided" if task_spec is None else task_spec.mode
    revision = "none" if packet.task is None else packet.task.revision
    return LatentPlan(
      plan_sequence_id=sequence,
      source_timestamp=packet.source_timestamp,
      source_tick=packet.source_tick,
      source_frame=packet.source_frame,
      period_seconds=packet.period_seconds,
      raw_latents=raw_latents,
      contract_identity=packet.contract_identity,
      vae_identity=packet.vae_identity,
      projection_identity=packet.projection_identity,
      model_identity=packet.model_identity,
      task_revision=revision,
      denoiser_calls=denoiser_calls,
      task_mode=task_mode,
    )

  def process(self, request: PlannerRequest) -> Plan:
    """Process an executor request and return the explicit base transport."""
    return self.plan(request).as_scheduler_plan()

  def process_request(self, request: PlannerRequest) -> Plan:
    """Alias for worker factories that use a named request processor."""
    return self.process(request)

  def _validate_packet(self, packet: LatentPlanningPacket) -> None:
    if packet.contract_identity != self.contract.identity_hash():
      raise LatentPlanningError(
        "planning packet contract identity differs from planner"
      )
    if packet.projection_identity != self._projection_identity:
      raise LatentPlanningError(
        "planning packet projection identity differs from planner"
      )
    if self.vae_identity is not None and packet.vae_identity != self.vae_identity:
      raise LatentPlanningError("planning packet VAE identity differs from planner")
    if self.model_identity is not None and packet.model_identity != self.model_identity:
      raise LatentPlanningError("planning packet model identity differs from planner")
    if (
      self.expected_source_frame is not None
      and packet.source_frame != self.expected_source_frame
    ):
      raise LatentPlanningError("planning packet source frame differs from planner")
    if packet.root_frame != self.expected_root_frame:
      raise LatentPlanningError("planning packet root frame differs from planner")
    if abs(packet.period_seconds - self.contract.period_seconds) > 1.0e-12:
      raise LatentPlanningError("planning packet period differs from active contract")
    if self.expected_task_revision is not None:
      revision = "none" if packet.task is None else packet.task.revision
      if revision != self.expected_task_revision:
        raise LatentPlanningError("planning packet task revision differs from planner")
    if packet.source_tick < 0 or not math.isfinite(packet.source_timestamp):
      raise LatentPlanningError("planning packet source identity is invalid")
    expected_history = HandoverHistory(contract=self.contract)
    for pair in packet.pairs:
      expected_history.push(
        pair.state.to_world_state(), np.asarray(pair.latent, dtype=np.float64)
      )
    expected = build_reference_free_plan(
      expected_history,
      packet.current_world_state(),
      projection=self.projection,
      tick=packet.source_tick,
      contract=self.contract,
      device="cpu",
    )
    if packet.task is not None and packet.task.mode in {"keyframe", "combined"}:
      spec = packet.task.to_spec()
      expected = inpaint_keyframes(
        expected,
        packet.current_world_state(),
        spec.keyframes,
        projection=self.projection,
        contract=self.contract,
      )
    expected_tokens, expected_mask = _snapshot_plan(expected)
    if expected_tokens != packet.clean_tokens or expected_mask != packet.clean_mask:
      raise LatentPlanningError("planning packet conditioning snapshot is inconsistent")


SynchronousLatentPlanner = LatentPlanner
build_latent_planning_packet = build_planning_packet


__all__ = [
  "LATENT_PLANNING_SCHEMA",
  "RAW_LATENT_REPRESENTATION",
  "ROOT_FRAME",
  "LatentPlan",
  "LatentPlanner",
  "LatentPlannerRequest",
  "LatentPlanningError",
  "LatentPlanningPacket",
  "PlanningPair",
  "SynchronousLatentPlanner",
  "StateSnapshot",
  "TaskSnapshot",
  "TaskTargetSnapshot",
  "TaskWeightsSnapshot",
  "build_latent_planning_packet",
  "build_planning_packet",
  "make_planning_packet",
]
