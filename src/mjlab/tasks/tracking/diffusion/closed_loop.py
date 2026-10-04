"""Simulator-independent matched closed-loop evaluation for D3.

The runner owns evaluation policy and evidence bookkeeping, but it does not
construct a simulator.  A concrete :class:`ClosedLoopRuntime` is supplied by
its caller and is created once per branch.  The baseline branch uses the VAE
reference path for its whole bounded trial.  The diffusion branch uses that
same VAE path for exactly eight measured bootstrap pairs, then hands ownership
to :class:`ReferenceFreeContinuation` without resetting or advancing the
runtime.

The public entry point is :func:`run_matched_pair`.  It deliberately returns a
serializable report rather than a success boolean: a fall, boundary, missing
evidence, or partial runtime failure is an attempted result and must remain
visible to a later evaluator.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
from torch import Tensor

from .closed_loop_types import ClosedLoopRuntime, D3Observation, D3Transition
from .handover import ContinuationStep, ReferenceFreeContinuation
from .state import WorldState, world_to_hybrid

_STATE_FIELDS = (
  "root_position",
  "root_quaternion_wxyz",
  "root_linear_velocity",
  "root_angular_velocity",
  "body_positions",
  "body_linear_velocities",
)
_STATE_WIDTH = 135
_LATENT_WIDTH = 32
_ACTION_WIDTH = 31
_REFERENCE_WIDTH = 68
_CONDITIONING_WIDTH = 99
_BOOTSTRAP_PAIRS = 8
_DEFAULT_PERIOD = 0.02
_DEFAULT_FALL_DEGREES = 70.0
_DEFAULT_OUTPUT_CAP = 128 * 1024 * 1024
_MIN_OUTPUT_CAP_BYTES = 4096
MIN_OUTPUT_CAP_BYTES = _MIN_OUTPUT_CAP_BYTES
# Raw matched-array tolerance for the D3 initialization/bootstrap gate and the
# handover state check.  This is the user-selected teacher-tracking error scale
# (1e-3), not a universal length unit and not an unguided reference-tracking
# loss or failure criterion.  Original 1e-6 runs remain historical.
_COMPARISON_TOLERANCE = 1.0e-3
_TIME_TOLERANCE = 1.0e-8


class ClosedLoopEvaluationError(ValueError):
  """A runner input or output contract is invalid."""


class VaeModel(Protocol):
  """The only VAE surface used by the runner."""

  def encode(self, reference: Tensor) -> tuple[Tensor, Tensor]: ...

  def decode(self, latent: Tensor, conditioning: Tensor) -> Tensor: ...


RuntimeFactory = Callable[[], Any]
ContinuationFactory = Callable[[], Any]
LatentNormalizer = Callable[[np.ndarray], np.ndarray]


def _is_integer(value: object) -> bool:
  return isinstance(value, (int, np.integer)) and not isinstance(
    value, (bool, np.bool_)
  )


def _finite_array(name: str, value: object, shape: tuple[int, ...]) -> np.ndarray:
  array = np.asarray(value, dtype=np.float64)
  if array.shape != shape:
    raise ClosedLoopEvaluationError(
      f"{name} must have shape {shape}, got {array.shape}"
    )
  if not np.isfinite(array).all():
    raise ClosedLoopEvaluationError(f"{name} contains non-finite values")
  return np.array(array, dtype=np.float64, copy=True)


def _copy_state(state: WorldState) -> WorldState:
  return WorldState(
    **{
      field: np.array(getattr(state, field), dtype=np.float64, copy=True)
      for field in _STATE_FIELDS
    }
  )


def _state_vector(state: WorldState) -> np.ndarray:
  return np.concatenate(
    [
      np.asarray(getattr(state, field), dtype=np.float64).reshape(-1)
      for field in _STATE_FIELDS
    ]
  )


def _state_max_abs_diff(first: WorldState, second: WorldState) -> float:
  left, right = _state_vector(first), _state_vector(second)
  if left.shape != right.shape:
    return float("inf")
  return float(np.max(np.abs(left - right)))


def _state_dict(state: WorldState) -> dict[str, Any]:
  return {field: np.asarray(getattr(state, field)).tolist() for field in _STATE_FIELDS}


def _json_value(value: Any) -> Any:
  """Convert report values without allowing NumPy scalars into JSON."""
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, (np.integer, np.floating)):
    return value.item()
  if isinstance(value, Mapping):
    return {str(key): _json_value(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_value(item) for item in value]
  if isinstance(value, Path):
    return str(value)
  return value


@dataclass(frozen=True, slots=True)
class EvaluationBudget:
  """Explicit bounds for one matched pair.

  ``max_steps`` and ``max_sim_time_s`` apply to the post-bootstrap scoring
  phase.  The eight bootstrap transitions are always required and are counted
  separately in every branch report.  ``output_cap_bytes`` bounds the optional
  JSON/JSONL artifact, not the in-memory report returned by the runner.
  """

  max_steps: int = 25
  max_sim_time_s: float | None = None
  bootstrap_pairs: int = _BOOTSTRAP_PAIRS
  period_seconds: float = _DEFAULT_PERIOD
  fall_degrees: float = _DEFAULT_FALL_DEGREES
  comparison_tolerance: float = _COMPARISON_TOLERANCE
  output_cap_bytes: int = _DEFAULT_OUTPUT_CAP

  def __post_init__(self) -> None:
    for name, value in (
      ("max_steps", self.max_steps),
      ("bootstrap_pairs", self.bootstrap_pairs),
      ("output_cap_bytes", self.output_cap_bytes),
    ):
      if not _is_integer(value):
        raise ClosedLoopEvaluationError(f"{name} must be an integer, not boolean")
    if self.max_steps < 0:
      raise ClosedLoopEvaluationError("max_steps must be nonnegative")
    if self.bootstrap_pairs != _BOOTSTRAP_PAIRS:
      raise ClosedLoopEvaluationError(
        f"D3 requires exactly {_BOOTSTRAP_PAIRS} bootstrap pairs"
      )
    if self.output_cap_bytes < _MIN_OUTPUT_CAP_BYTES:
      raise ClosedLoopEvaluationError(
        f"output_cap_bytes must be at least {_MIN_OUTPUT_CAP_BYTES}"
      )
    for name, value in (
      ("period_seconds", self.period_seconds),
      ("fall_degrees", self.fall_degrees),
      ("comparison_tolerance", self.comparison_tolerance),
    ):
      if isinstance(value, (bool, np.bool_)) or not np.isfinite(float(value)):
        raise ClosedLoopEvaluationError(f"{name} must be finite")
    if self.period_seconds <= 0:
      raise ClosedLoopEvaluationError("period_seconds must be positive")
    if not math.isclose(self.period_seconds, _DEFAULT_PERIOD, abs_tol=_TIME_TOLERANCE):
      raise ClosedLoopEvaluationError(
        f"D3 control period must be exactly {_DEFAULT_PERIOD} seconds"
      )
    if self.fall_degrees <= 0 or self.comparison_tolerance <= 0:
      raise ClosedLoopEvaluationError(
        "fall_degrees and comparison_tolerance must be positive"
      )
    if self.max_sim_time_s is not None:
      if isinstance(self.max_sim_time_s, (bool, np.bool_)) or not np.isfinite(
        float(self.max_sim_time_s)
      ):
        raise ClosedLoopEvaluationError("max_sim_time_s must be finite or None")
      if self.max_sim_time_s <= 0:
        raise ClosedLoopEvaluationError("max_sim_time_s must be positive or None")


# Explicit aliases make the runner interface easy to discover without adding
# another configuration shape that could drift from the frozen budget.
ClosedLoopBudget = EvaluationBudget
RunnerConfig = EvaluationBudget
RunBudget = EvaluationBudget


@dataclass(frozen=True, slots=True)
class BranchReport:
  """Serializable evidence from one attempted baseline or diffusion branch."""

  name: str
  seed: int
  start_frame: int
  attempted_ticks: int
  completed_ticks: int
  bootstrap_attempted: int
  bootstrap_completed: int
  evaluation_attempted: int
  evaluation_completed: int
  stop_reason: str
  outcome: str
  survival_observed: bool
  simulation_duration_s: float
  initial_signature: tuple[float, ...] | None
  initial_reference: int | None
  tilt_degrees: tuple[float, ...]
  peak_tilt_degrees: float | None
  pelvis_heights_m: tuple[float, ...]
  minimum_pelvis_height_m: float | None
  ticks: tuple[dict[str, Any], ...]
  counters: Mapping[str, int]
  errors: tuple[str, ...]
  provenance: Mapping[str, Any]
  contact_metrics: None = None
  torque_saturation: None = None
  reference_deviation: None = None

  def as_dict(self) -> dict[str, Any]:
    return _json_value(
      {
        "name": self.name,
        "seed": self.seed,
        "start_frame": self.start_frame,
        "attempted_ticks": self.attempted_ticks,
        "completed_ticks": self.completed_ticks,
        "bootstrap_attempted": self.bootstrap_attempted,
        "bootstrap_completed": self.bootstrap_completed,
        "evaluation_attempted": self.evaluation_attempted,
        "evaluation_completed": self.evaluation_completed,
        "stop_reason": self.stop_reason,
        "outcome": self.outcome,
        "survival_observed": self.survival_observed,
        "simulation_duration_s": self.simulation_duration_s,
        "initial_signature": self.initial_signature,
        "initial_reference": self.initial_reference,
        "tilt_degrees": self.tilt_degrees,
        "peak_tilt_degrees": self.peak_tilt_degrees,
        "pelvis_heights_m": self.pelvis_heights_m,
        "minimum_pelvis_height_m": self.minimum_pelvis_height_m,
        "ticks": self.ticks,
        "counters": dict(self.counters),
        "errors": self.errors,
        "provenance": dict(self.provenance),
        "contact_metrics": None,
        "torque_saturation": None,
        "reference_deviation": None,
      }
    )


@dataclass(frozen=True, slots=True)
class EvaluationReport:
  """Matched report returned by :func:`run_matched_pair`."""

  status: str
  paired: bool
  closed_loop: bool
  synthetic: bool
  seed: int
  start_frame: int
  budget: EvaluationBudget
  baseline: BranchReport | None
  diffusion: BranchReport | None
  pairing: Mapping[str, Any]
  provenance: Mapping[str, Any]
  errors: tuple[str, ...]
  output: dict[str, Any]

  @property
  def valid(self) -> bool:
    """Whether the matched attempt is complete without software failure."""
    return self.status == "complete" and self.paired

  def as_dict(self) -> dict[str, Any]:
    payload = {
      "schema": "mjlab-d3-closed-loop-evaluation-v1",
      "status": self.status,
      "paired": self.paired,
      "valid": self.valid,
      "closed_loop": self.closed_loop,
      "synthetic": self.synthetic,
      "seed": self.seed,
      "start_frame": self.start_frame,
      "budget": {
        "max_steps": self.budget.max_steps,
        "max_sim_time_s": self.budget.max_sim_time_s,
        "bootstrap_pairs": self.budget.bootstrap_pairs,
        "period_seconds": self.budget.period_seconds,
        "fall_degrees": self.budget.fall_degrees,
        "comparison_tolerance": self.budget.comparison_tolerance,
        "output_cap_bytes": self.budget.output_cap_bytes,
      },
      "branches": {
        "baseline": None if self.baseline is None else self.baseline.as_dict(),
        "diffusion": None if self.diffusion is None else self.diffusion.as_dict(),
      },
      "pairing": dict(self.pairing),
      "provenance": dict(self.provenance),
      "errors": list(self.errors),
      "output": dict(self.output),
      "claims": {
        "closed_loop": self.closed_loop,
        "synthetic": self.synthetic,
        "contact_metrics": "unavailable-unless-runtime-captures-known-semantics",
        "torque_saturation": "unavailable-unless-runtime-captures-known-semantics",
        "pelvis_height": "descriptive-only",
        "reference_deviation": "descriptive-only-and-not-an-unguided-failure-criterion",
        "wall_time_realtime": "not-qualified-by-this-runner",
      },
    }
    return _json_value(payload)

  def __getitem__(self, key: str) -> Any:
    """Provide mapping-style access for CLI/report integrations."""
    return self.as_dict()[key]


@dataclass(frozen=True, slots=True)
class _BootstrapRow:
  state: WorldState
  conditioning: np.ndarray
  latent: np.ndarray
  action: np.ndarray
  tick: int
  sim_time_s: float
  segment_id: str
  reference: np.ndarray | None
  reference_frame: int | None


class _BranchBuilder:
  """Mutable accumulator kept private until converted to ``BranchReport``."""

  def __init__(self, name: str, seed: int, start_frame: int) -> None:
    self.name = name
    self.seed = seed
    self.start_frame = start_frame
    self.attempted_ticks = 0
    self.completed_ticks = 0
    self.bootstrap_attempted = 0
    self.bootstrap_completed = 0
    self.evaluation_attempted = 0
    self.evaluation_completed = 0
    self.stop_reason: str | None = None
    self.errors: list[str] = []
    self.ticks: list[dict[str, Any]] = []
    self.bootstrap_rows: list[_BootstrapRow] = []
    self.bootstrap_end_state: WorldState | None = None
    self.bootstrap_end_conditioning: np.ndarray | None = None
    self.bootstrap_end_tick: int | None = None
    self.bootstrap_end_sim_time_s: float | None = None
    self.bootstrap_end_segment: str | None = None
    self.initial_signature: tuple[float, ...] | None = None
    self.initial_signature_digest: str | None = None
    self.handover_record: dict[str, Any] = {
      "entered": False,
      "advanced_simulation": None,
      "reset_physical_state": None,
    }
    self.accepted_step_count = 0
    self.untrusted_terminal_evidence = False
    self.initial_reference: int | None = None
    self.initial_time_s = 0.0
    self.last_time_s = 0.0
    self.last_action: np.ndarray | None = None
    self.tilts: list[float] = []
    self.pelvis_heights: list[float] = []
    self.counters = {"encoder": 0, "decoder": 0, "denoiser": 0}
    self.runtime_provenance: Mapping[str, Any] = {}

  def stop(self, reason: str, detail: str | None = None) -> None:
    if self.stop_reason is None:
      self.stop_reason = reason
    if detail:
      self.errors.append(detail)

  def add_error(self, error: object) -> None:
    text = str(error)
    self.errors.append(text)
    if self.stop_reason is None:
      self.stop_reason = "runtime_error"

  def observe_state(self, state: WorldState) -> float:
    tilt = _tilt_degrees(state)
    height = float(state.root_position[2])
    self.tilts.append(tilt)
    self.pelvis_heights.append(height)
    return tilt

  def finish(self, *, last_time_s: float | None = None) -> BranchReport:
    reason = self.stop_reason or "runtime_error"
    if last_time_s is not None and np.isfinite(last_time_s):
      self.last_time_s = float(last_time_s)
    duration = max(0.0, self.last_time_s - self.initial_time_s)
    bounded_stop = reason in {"step_budget", "time_budget", "reference_boundary"}
    complete_bootstrap = self.bootstrap_completed == _BOOTSTRAP_PAIRS
    survival = bool(bounded_stop and complete_bootstrap)
    if survival and reason == "reference_boundary":
      outcome = "stopped_at_reference_boundary"
    elif survival:
      outcome = "bounded_survival_observed"
    elif reason == "physical_fall":
      outcome = "physical_fall"
    elif complete_bootstrap:
      outcome = "partial_failure"
    else:
      outcome = "bootstrap_failure"
    peak = max(self.tilts) if self.tilts else None
    minimum_height = min(self.pelvis_heights) if self.pelvis_heights else None
    return BranchReport(
      name=self.name,
      seed=self.seed,
      start_frame=self.start_frame,
      attempted_ticks=self.attempted_ticks,
      completed_ticks=self.completed_ticks,
      bootstrap_attempted=self.bootstrap_attempted,
      bootstrap_completed=self.bootstrap_completed,
      evaluation_attempted=self.evaluation_attempted,
      evaluation_completed=self.evaluation_completed,
      stop_reason=reason,
      outcome=outcome,
      survival_observed=survival,
      simulation_duration_s=duration,
      initial_signature=self.initial_signature,
      initial_reference=self.initial_reference,
      tilt_degrees=tuple(self.tilts),
      peak_tilt_degrees=None if peak is None else float(peak),
      pelvis_heights_m=tuple(self.pelvis_heights),
      minimum_pelvis_height_m=(
        None if minimum_height is None else float(minimum_height)
      ),
      ticks=tuple(self.ticks),
      counters=dict(self.counters),
      errors=tuple(self.errors),
      provenance=dict(self.runtime_provenance),
    )


def _tilt_degrees(state: WorldState) -> float:
  """Return root-up tilt from the world quaternion, independent of yaw."""
  q = np.asarray(state.root_quaternion_wxyz, dtype=np.float64)
  norm = float(np.linalg.norm(q))
  if not np.isfinite(norm) or norm < 1.0e-12:
    return float("inf")
  w, x, y, z = q / norm
  up_z = 1.0 - 2.0 * (x * x + y * y)
  return float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0))))


def _validate_observation(observation: Any) -> D3Observation:
  """Validate a runtime-owned observation, including fake-runtime inputs."""
  try:
    if not _is_integer(observation.tick):
      raise ClosedLoopEvaluationError("observation tick must be integral and not bool")
    if int(observation.tick) < 0:
      raise ClosedLoopEvaluationError("observation tick must be nonnegative")
    sim_time = float(observation.sim_time_s)
    if not np.isfinite(sim_time) or sim_time < 0:
      raise ClosedLoopEvaluationError("observation simulation time is not finite")
    if not isinstance(observation.segment_id, str) or not observation.segment_id:
      raise ClosedLoopEvaluationError("observation segment_id is invalid")
    state = _copy_state(observation.state)
    conditioning = _finite_array(
      "observation conditioning", observation.conditioning, (_CONDITIONING_WIDTH,)
    )
    reference_value = observation.reference
    reference = (
      None
      if reference_value is None
      else _finite_array("observation reference", reference_value, (_REFERENCE_WIDTH,))
    )
    reference_frame = observation.reference_frame
    if reference_frame is not None and not _is_integer(reference_frame):
      raise ClosedLoopEvaluationError("reference_frame must be integral or None")
    if reference_frame is not None and int(reference_frame) < 0:
      raise ClosedLoopEvaluationError("reference_frame must be nonnegative")
  except ClosedLoopEvaluationError:
    raise
  except Exception as exc:
    raise ClosedLoopEvaluationError(f"malformed runtime observation: {exc}") from exc
  return D3Observation(
    state=state,
    conditioning=conditioning,
    tick=int(observation.tick),
    sim_time_s=sim_time,
    segment_id=observation.segment_id,
    reference=reference,
    reference_frame=None if reference_frame is None else int(reference_frame),
  )


def _validate_runtime(runtime: Any, budget: EvaluationBudget) -> None:
  try:
    period = float(runtime.period_seconds)
  except Exception as exc:
    raise ClosedLoopEvaluationError("runtime does not expose period_seconds") from exc
  if not np.isfinite(period) or not math.isclose(
    period, budget.period_seconds, abs_tol=_TIME_TOLERANCE
  ):
    raise ClosedLoopEvaluationError(
      f"runtime control period must be {budget.period_seconds}, got {period}"
    )
  clip_frames = runtime.clip_frames
  if not _is_integer(clip_frames) or int(clip_frames) <= 0:
    raise ClosedLoopEvaluationError("runtime clip_frames must be a positive integer")
  for field in ("joint_order", "body_order"):
    values = getattr(runtime, field, None)
    if not isinstance(values, (tuple, list)) or not values:
      raise ClosedLoopEvaluationError(f"runtime {field} must be a nonempty sequence")
    if not all(isinstance(value, str) and value for value in values):
      raise ClosedLoopEvaluationError(f"runtime {field} contains an invalid name")
    if len(set(values)) != len(values):
      raise ClosedLoopEvaluationError(f"runtime {field} contains duplicate names")
  if len(runtime.body_order) != 20:
    raise ClosedLoopEvaluationError(
      f"runtime body_order must contain 20 names, got {len(runtime.body_order)}"
    )
  provenance = getattr(runtime, "provenance", {})
  if not isinstance(provenance, Mapping):
    raise ClosedLoopEvaluationError("runtime provenance must be a mapping")


def _initial_signature(runtime: Any) -> tuple[float, ...]:
  try:
    value = np.asarray(runtime.initial_signature(), dtype=np.float64)
  except Exception as exc:
    raise ClosedLoopEvaluationError(f"runtime initial signature failed: {exc}") from exc
  if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
    raise ClosedLoopEvaluationError(
      "runtime initial signature must be a finite nonempty vector"
    )
  return tuple(float(item) for item in value)


def _validate_phase_observation(
  observation: D3Observation,
  *,
  reference_free: bool,
  clip_frames: int,
) -> str | None:
  try:
    _validate_observation(observation)
  except ClosedLoopEvaluationError:
    return "nonfinite"
  if reference_free:
    if observation.reference is not None or observation.reference_frame is not None:
      return "reference_leakage"
    return None
  if observation.reference is None:
    return "missing_reference"
  if (
    observation.reference_frame is not None
    and observation.reference_frame >= clip_frames
  ):
    return "reference_boundary"
  return None


def _validate_action(value: Any) -> np.ndarray:
  if isinstance(value, Tensor):
    array = value.detach().to(device="cpu", dtype=torch.float32).numpy()
  else:
    array = np.asarray(value)
  array = np.asarray(array, dtype=np.float64).reshape(-1)
  if array.shape != (_ACTION_WIDTH,) or not np.isfinite(array).all():
    raise ClosedLoopEvaluationError(
      f"decoded action must be finite with shape [{_ACTION_WIDTH}]"
    )
  return np.array(array, copy=True)


def _validate_latent(value: Any, name: str = "latent") -> np.ndarray:
  if isinstance(value, Tensor):
    array = value.detach().to(device="cpu", dtype=torch.float32).numpy()
  else:
    array = np.asarray(value)
  array = np.asarray(array, dtype=np.float64)
  if array.shape == (1, _LATENT_WIDTH):
    array = array[0]
  if array.shape != (_LATENT_WIDTH,) or not np.isfinite(array).all():
    raise ClosedLoopEvaluationError(
      f"{name} must be finite with shape [{_LATENT_WIDTH}]"
    )
  return np.array(array, copy=True)


def _model_device(model: Any, fallback: torch.device) -> torch.device:
  try:
    parameter = next(model.parameters())
  except (AttributeError, StopIteration, TypeError):
    return fallback
  return parameter.device


def _vae_action(
  model: VaeModel,
  observation: D3Observation,
  counters: dict[str, int],
  *,
  device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
  if observation.reference is None:
    raise ClosedLoopEvaluationError("VAE action requires a live reference")
  reference = torch.from_numpy(
    np.ascontiguousarray(observation.reference, dtype=np.float32)
  ).to(device=device)[None]
  conditioning = torch.from_numpy(
    np.ascontiguousarray(observation.conditioning, dtype=np.float32)
  ).to(device=device)[None]
  counters["encoder"] += 1
  with torch.no_grad():
    encoded = model.encode(reference)
  if not isinstance(encoded, tuple) or len(encoded) != 2:
    raise ClosedLoopEvaluationError("VAE encode must return (mu, logvar)")
  mu, logvar = encoded
  if not isinstance(mu, Tensor) or not isinstance(logvar, Tensor):
    raise ClosedLoopEvaluationError("VAE encode outputs must be tensors")
  if mu.shape != (1, _LATENT_WIDTH) or logvar.shape != (1, _LATENT_WIDTH):
    raise ClosedLoopEvaluationError("VAE encode outputs must have shape [1, 32]")
  if not torch.isfinite(mu).all().item() or not torch.isfinite(logvar).all().item():
    raise ClosedLoopEvaluationError("VAE encode outputs contain non-finite values")
  counters["decoder"] += 1
  with torch.no_grad():
    action = model.decode(mu.detach().clone(), conditioning)
  return _validate_latent(mu), _validate_action(action)


def _normalised_latent(
  latent: np.ndarray,
  *,
  normalizer: LatentNormalizer | None = None,
) -> np.ndarray | None:
  if normalizer is None:
    return None
  try:
    value = np.asarray(normalizer(np.array(latent, copy=True)), dtype=np.float64)
  except Exception:
    return None
  if value.shape != (_LATENT_WIDTH,) or not np.isfinite(value).all():
    return None
  return value


def _metric(value: np.ndarray) -> dict[str, float]:
  array = np.asarray(value, dtype=np.float64).reshape(-1)
  return {
    "max_abs": float(np.max(np.abs(array))),
    "l2": float(np.linalg.norm(array)),
    "rmse": float(np.sqrt(np.mean(np.square(array)))),
  }


def _prediction_metrics(
  predicted_next_hybrid_state: np.ndarray | None,
  pre_state: WorldState,
  post_state: WorldState,
) -> dict[str, Any] | None:
  if predicted_next_hybrid_state is None:
    return None
  predicted = _finite_array(
    "predicted next hybrid state", predicted_next_hybrid_state, (_STATE_WIDTH,)
  )
  actual, _ = world_to_hybrid(WorldState.stack([pre_state, post_state]), center_index=0)
  difference = predicted - actual[1]
  blocks = {
    "root_position": difference[0:3],
    "root_rot6d": difference[3:9],
    "root_linear_velocity": difference[9:12],
    "root_angular_velocity": difference[12:15],
    "body_position": difference[15:75],
    "body_velocity": difference[75:135],
  }
  return {
    "available": True,
    "center": "pre_action_state",
    "actual_next_hybrid_state": actual[1].tolist(),
    "predicted_next_hybrid_state": predicted.tolist(),
    "blocks": {name: _metric(value) for name, value in blocks.items()},
  }


def _base_tick_record(
  builder: _BranchBuilder,
  observation: D3Observation,
  *,
  phase: str,
  latent: np.ndarray | None,
  action: np.ndarray | None,
  normalizer: LatentNormalizer | None,
) -> dict[str, Any]:
  normalized = (
    None if latent is None else _normalised_latent(latent, normalizer=normalizer)
  )
  action_derivative = None
  if action is not None and builder.last_action is not None:
    action_derivative = float(
      np.linalg.norm(action - builder.last_action) / _DEFAULT_PERIOD
    )
  record: dict[str, Any] = {
    "phase": phase,
    "tick": int(observation.tick),
    "sim_time_s": float(observation.sim_time_s),
    "segment_id": observation.segment_id,
    "state": _state_dict(observation.state),
    "conditioning": observation.conditioning.tolist(),
    "reference": None
    if observation.reference is None
    else observation.reference.tolist(),
    "reference_frame": observation.reference_frame,
    "latent": None if latent is None else latent.tolist(),
    "latent_raw_l2": None if latent is None else float(np.linalg.norm(latent)),
    "latent_raw_max_abs": None if latent is None else float(np.max(np.abs(latent))),
    "latent_normalized_l2": (
      None if normalized is None else float(np.linalg.norm(normalized))
    ),
    "latent_normalized_max_abs": (
      None if normalized is None else float(np.max(np.abs(normalized)))
    ),
    "action": None if action is None else action.tolist(),
    "action_l2": None if action is None else float(np.linalg.norm(action)),
    "action_max_abs": None if action is None else float(np.max(np.abs(action))),
    "action_derivative_l2_per_s": action_derivative,
    "pre_action_tilt_degrees": float(_tilt_degrees(observation.state)),
    "pre_action_pelvis_height_m": float(observation.state.root_position[2]),
    "completed": False,
    "stop_reason": None,
    "post": None,
    "prediction": None,
    "denoiser_calls": 0,
    "final_index": None,
  }
  if action is not None:
    builder.last_action = np.array(action, copy=True)
  return record


def _post_step_reason(
  transition: D3Transition,
  pre: D3Observation,
  *,
  reference_free: bool,
  clip_frames: int,
  budget: EvaluationBudget,
) -> tuple[str | None, D3Observation | None]:
  terminated = bool(getattr(transition, "terminated", False))
  reset = bool(getattr(transition, "reset", False))
  teleport = bool(getattr(transition, "teleport", False))
  terminal_available = bool(getattr(transition, "terminal_available", False))
  if reset or teleport:
    # A reset/teleport observation is not the physical state reached by the
    # action.  Preserve explicit termination precedence, but never pass this
    # transplanted state to fall or prediction metrics.
    if not terminal_available:
      return "missing_terminal_evidence", None
    if terminated:
      return "terminated", None
    return ("reset" if reset else "teleport"), None
  try:
    post = _validate_observation(transition.observation)
  except ClosedLoopEvaluationError:
    return "nonfinite", None
  post_tilt = _tilt_degrees(post.state)
  # A fall on the final frame is more informative than the simultaneous clip
  # boundary marker, so physical fall intentionally has first precedence when
  # the post-step state is still trustworthy.
  if post_tilt > budget.fall_degrees:
    return "physical_fall", post
  if not terminal_available:
    return "missing_terminal_evidence", post
  if terminated:
    return "terminated", post
  if bool(getattr(transition, "truncated", False)):
    return "timeout", post
  if post.segment_id != pre.segment_id:
    return "segment_change", post
  if post.tick != pre.tick + 1:
    return "tick_discontinuity", post
  if not math.isclose(
    post.sim_time_s - pre.sim_time_s,
    budget.period_seconds,
    abs_tol=_TIME_TOLERANCE,
  ):
    return "period_discontinuity", post
  if bool(getattr(transition, "reference_boundary", False)):
    return "reference_boundary", post
  if (
    not reference_free
    and post.reference_frame is not None
    and post.reference_frame >= clip_frames
  ):
    return "reference_boundary", post
  if reference_free and (
    post.reference is not None or post.reference_frame is not None
  ):
    return "reference_leakage", post
  return None, post


def _finish_tick(
  builder: _BranchBuilder,
  record: dict[str, Any],
  *,
  post: D3Observation | None,
  reason: str | None,
  prediction: dict[str, Any] | None = None,
) -> None:
  if post is not None:
    record["post"] = {
      "tick": int(post.tick),
      "sim_time_s": float(post.sim_time_s),
      "segment_id": post.segment_id,
      "state": _state_dict(post.state),
      "reference": None if post.reference is None else post.reference.tolist(),
      "reference_frame": post.reference_frame,
      "tilt_degrees": float(_tilt_degrees(post.state)),
      "pelvis_height_m": float(post.state.root_position[2]),
    }
    builder.observe_state(post.state)
  if reason == "terminated" and post is None:
    builder.untrusted_terminal_evidence = True
  record["prediction"] = prediction
  record["completed"] = reason is None
  record["stop_reason"] = reason
  builder.ticks.append(record)
  if reason is None:
    builder.completed_ticks += 1
    builder.accepted_step_count += 1
  else:
    builder.stop(reason)


def _run_bootstrap(
  runtime: ClosedLoopRuntime,
  builder: _BranchBuilder,
  observation: D3Observation,
  *,
  budget: EvaluationBudget,
  reference_free: bool,
  model: VaeModel,
  model_device: torch.device,
  latent_normalizer: LatentNormalizer | None,
) -> D3Observation | None:
  current = observation
  for _ in range(_BOOTSTRAP_PAIRS):
    phase_reason = _validate_phase_observation(
      current,
      reference_free=reference_free,
      clip_frames=int(runtime.clip_frames),
    )
    if phase_reason is not None:
      builder.stop(phase_reason)
      break
    pre = _copy_state(current.state)
    builder.bootstrap_attempted += 1
    builder.attempted_ticks += 1
    latent: np.ndarray | None = None
    action: np.ndarray | None = None
    try:
      latent, action = _vae_action(
        model, current, builder.counters, device=model_device
      )
      record = _base_tick_record(
        builder,
        current,
        phase="bootstrap",
        latent=latent,
        action=action,
        normalizer=latent_normalizer,
      )
    except Exception as exc:
      record = _base_tick_record(
        builder,
        current,
        phase="bootstrap",
        latent=latent,
        action=action,
        normalizer=latent_normalizer,
      )
      record["error"] = str(exc)
      _finish_tick(builder, record, post=None, reason="inference_error")
      break
    try:
      transition = runtime.step(np.array(action, copy=True))
      reason, post = _post_step_reason(
        transition,
        current,
        reference_free=reference_free,
        clip_frames=int(runtime.clip_frames),
        budget=budget,
      )
    except Exception as exc:
      reason, post = "runtime_error", None
      record["error"] = str(exc)
    prediction = None
    _finish_tick(builder, record, post=post, reason=reason, prediction=prediction)
    if reason is not None or post is None:
      break
    assert latent is not None and action is not None
    builder.bootstrap_completed += 1
    builder.bootstrap_rows.append(
      _BootstrapRow(
        state=pre,
        conditioning=np.array(current.conditioning, copy=True),
        latent=np.array(latent, copy=True),
        action=np.array(action, copy=True),
        tick=current.tick,
        sim_time_s=current.sim_time_s,
        segment_id=current.segment_id,
        reference=None
        if current.reference is None
        else np.array(current.reference, copy=True),
        reference_frame=current.reference_frame,
      )
    )
    current = post
    builder.last_time_s = current.sim_time_s
  return current if builder.bootstrap_completed == _BOOTSTRAP_PAIRS else None


def _score_baseline(
  runtime: ClosedLoopRuntime,
  builder: _BranchBuilder,
  current: D3Observation,
  *,
  model: VaeModel,
  model_device: torch.device,
  budget: EvaluationBudget,
  latent_normalizer: LatentNormalizer | None,
) -> None:
  steps = 0
  while True:
    if steps >= budget.max_steps:
      builder.stop("step_budget")
      break
    if budget.max_sim_time_s is not None:
      origin = builder.bootstrap_end_sim_time_s
      if origin is None:
        origin = builder.initial_time_s
      if current.sim_time_s - origin >= budget.max_sim_time_s - _TIME_TOLERANCE:
        builder.stop("time_budget")
        break
    phase_reason = _validate_phase_observation(
      current,
      reference_free=False,
      clip_frames=int(runtime.clip_frames),
    )
    if phase_reason is not None:
      builder.stop(phase_reason)
      break
    builder.evaluation_attempted += 1
    builder.attempted_ticks += 1
    latent: np.ndarray | None = None
    action: np.ndarray | None = None
    try:
      latent, action = _vae_action(
        model, current, builder.counters, device=model_device
      )
      record = _base_tick_record(
        builder,
        current,
        phase="baseline",
        latent=latent,
        action=action,
        normalizer=latent_normalizer,
      )
    except Exception as exc:
      record = _base_tick_record(
        builder,
        current,
        phase="baseline",
        latent=latent,
        action=action,
        normalizer=latent_normalizer,
      )
      record["error"] = str(exc)
      _finish_tick(builder, record, post=None, reason="inference_error")
      break
    try:
      transition = runtime.step(np.array(action, copy=True))
      reason, post = _post_step_reason(
        transition,
        current,
        reference_free=False,
        clip_frames=int(runtime.clip_frames),
        budget=budget,
      )
    except Exception as exc:
      reason, post = "runtime_error", None
      record["error"] = str(exc)
    _finish_tick(builder, record, post=post, reason=reason)
    if reason is not None or post is None:
      break
    builder.evaluation_completed += 1
    steps += 1
    current = post
    builder.last_time_s = current.sim_time_s


def _continuation_latent_normalizer(
  continuation: ReferenceFreeContinuation,
) -> LatentNormalizer | None:
  projection = getattr(continuation, "projection", None)
  normalize = getattr(projection, "normalize_latent", None)
  if not callable(normalize):
    return None

  def apply(value: np.ndarray) -> np.ndarray:
    return np.asarray(normalize(value), dtype=np.float64)

  return apply


def _trusted_post_observation(
  pre: D3Observation,
  transition: D3Transition | None,
  post: D3Observation | None,
  *,
  reason: str | None,
  budget: EvaluationBudget,
) -> D3Observation | None:
  """Return a post-step observation only when it is a trusted measurement.

  D3 stop precedence is unchanged: this runs after ``_post_step_reason`` has
  selected the stop reason, and it only decides whether the post observation may
  enter a task lane's realized-error ledger.  A validated, finite,
  reset/teleport-free post that continues the same segment at the next tick and
  the exact control period with terminal evidence available and no reference
  leak is trustworthy; anything else (including a fall without terminal
  evidence) is refused rather than scored.
  """
  if post is None or transition is None or reason not in {None, "physical_fall"}:
    return None
  if bool(getattr(transition, "reset", False)) or bool(
    getattr(transition, "teleport", False)
  ):
    return None
  if not bool(getattr(transition, "terminal_available", False)):
    return None
  if post.segment_id != pre.segment_id:
    return None
  if post.tick != pre.tick + 1:
    return None
  if not math.isclose(
    post.sim_time_s - pre.sim_time_s,
    budget.period_seconds,
    abs_tol=_TIME_TOLERANCE,
  ):
    return None
  if post.reference is not None or post.reference_frame is not None:
    return None
  return post


def _observe_measured_post(
  builder: _BranchBuilder,
  continuation: ReferenceFreeContinuation,
  observation: D3Observation,
) -> bool:
  """Report a trusted post; False requires immediate caller termination.

  Only an installed observer adds a task counter.  A reporting failure retains
  the executed post but must not permit another action or history insertion.
  """
  hook = getattr(continuation, "observe_measured_post", None)
  if not callable(hook):
    return True
  try:
    observed = hook(int(observation.tick), observation.state)
  except Exception as exc:
    builder.stop("task_observation_error", str(exc))
    return False
  if observed is True:
    builder.counters["task_measured_posts"] = (
      builder.counters.get("task_measured_posts", 0) + 1
    )
  return True


def _mark_prediction_source(
  prediction: dict[str, Any] | None, task_record: Mapping[str, Any] | None
) -> None:
  """Label a task-run next-row metric as hard-keyframe or learned forecast.

  A hard keyframe fixes the next row, so the sampler's next value is that
  constraint rather than a denoiser forecast.  The metric values are unchanged
  (the sample is the same), but the label keeps learned-prediction quality from
  being overstated.  Non-task runs keep the original D3 metric payload.
  """
  if prediction is None or task_record is None:
    return
  prediction["source"] = (
    "hard_keyframe_row"
    if bool(task_record.get("next_row_hard", False))
    else "denoiser_forecast"
  )


def _score_diffusion(
  runtime: ClosedLoopRuntime,
  builder: _BranchBuilder,
  current: D3Observation,
  continuation: ReferenceFreeContinuation,
  *,
  budget: EvaluationBudget,
) -> None:
  steps = 0
  normalizer = _continuation_latent_normalizer(continuation)
  while True:
    if steps >= budget.max_steps:
      builder.stop("step_budget")
      break
    if budget.max_sim_time_s is not None:
      origin = builder.bootstrap_end_sim_time_s
      if origin is None:
        origin = builder.initial_time_s
      if current.sim_time_s - origin >= budget.max_sim_time_s - _TIME_TOLERANCE:
        builder.stop("time_budget")
        break
    phase_reason = _validate_phase_observation(
      current,
      reference_free=True,
      clip_frames=int(runtime.clip_frames),
    )
    if phase_reason is not None:
      builder.stop(phase_reason)
      break
    pre = _copy_state(current.state)
    builder.evaluation_attempted += 1
    builder.attempted_ticks += 1
    step: ContinuationStep | None = None
    latent: np.ndarray | None = None
    action: np.ndarray | None = None
    transition: D3Transition | None = None
    try:
      step = continuation.step(
        pre,
        np.array(current.conditioning, copy=True),
        tick=current.tick,
      )
      if int(step.tick) != current.tick:
        raise ClosedLoopEvaluationError("continuation returned a mismatched tick")
      latent = _validate_latent(step.executed_latent, "continuation latent")
      action = _validate_action(step.action)
      calls = int(step.denoiser_calls)
      if calls < 0:
        raise ClosedLoopEvaluationError(
          "continuation denoiser_calls must be nonnegative"
        )
      builder.counters["decoder"] += 1
      builder.counters["denoiser"] += calls
      record = _base_tick_record(
        builder,
        current,
        phase="diffusion",
        latent=latent,
        action=action,
        normalizer=normalizer,
      )
      record["denoiser_calls"] = calls
      record["final_index"] = int(step.final_index)
      task_record = getattr(step, "task", None)
      if task_record is not None:
        record["task"] = dict(task_record)
    except Exception as exc:
      record = _base_tick_record(
        builder,
        current,
        phase="diffusion",
        latent=latent,
        action=action,
        normalizer=normalizer,
      )
      record["error"] = str(exc)
      _finish_tick(builder, record, post=None, reason="inference_error")
      break
    try:
      transition = runtime.step(np.array(action, copy=True))
      reason, post = _post_step_reason(
        transition,
        current,
        reference_free=True,
        clip_frames=int(runtime.clip_frames),
        budget=budget,
      )
      prediction = (
        None
        if post is None or step is None
        else _prediction_metrics(step.predicted_next_hybrid_state, pre, post.state)
      )
      _mark_prediction_source(prediction, task_record)
    except Exception as exc:
      reason, post, prediction = "runtime_error", None, None
      record["error"] = str(exc)
    _finish_tick(builder, record, post=post, reason=reason, prediction=prediction)
    trusted_post = _trusted_post_observation(
      current, transition, post, reason=reason, budget=budget
    )
    if trusted_post is not None and not _observe_measured_post(
      builder, continuation, trusted_post
    ):
      # The physical transition is retained; a reporting failure cannot allow
      # another action or turn this state/latent into accepted history.
      record["task_observation_error"] = builder.errors[-1]
      if reason is None:
        record["stop_reason"] = "task_observation_error"
      break
    if reason is not None or post is None:
      break
    assert latent is not None
    try:
      # The pre-action state is intentionally retained until step evidence has
      # been accepted.  A failed boundary never enters diffusion history.
      continuation.record_executed(pre, latent)
    except Exception as exc:
      builder.stop("history_record_error", str(exc))
      break
    builder.evaluation_completed += 1
    steps += 1
    current = post
    builder.last_time_s = current.sim_time_s


def _compare_bootstrap(
  baseline: _BranchBuilder,
  diffusion: _BranchBuilder,
  *,
  tolerance: float,
) -> dict[str, Any]:
  result: dict[str, Any] = {
    "passed": False,
    "tolerance": float(tolerance),
    "initial_signature_max_abs_diff": None,
    "state_max_abs_diff": None,
    "conditioning_max_abs_diff": None,
    "action_max_abs_diff": None,
    "latent_max_abs_diff": None,
    "reference_max_abs_diff": None,
    "bootstrap_end_state_max_abs_diff": None,
    "bootstrap_end_conditioning_max_abs_diff": None,
    "bootstrap_end_ticks_match": False,
    "bootstrap_end_segments_match": False,
    "bootstrap_end_time_max_abs_diff": None,
    "ticks_match": False,
    "segments_match": False,
    "reason": None,
  }
  if baseline.initial_signature is None or diffusion.initial_signature is None:
    result["reason"] = "missing_initial_signature"
    return result
  if len(baseline.initial_signature) != len(diffusion.initial_signature):
    result["reason"] = "initial_signature_width_mismatch"
    return result
  signature_diff = float(
    np.max(
      np.abs(
        np.asarray(baseline.initial_signature) - np.asarray(diffusion.initial_signature)
      )
    )
  )
  result["initial_signature_max_abs_diff"] = signature_diff
  if (
    len(baseline.bootstrap_rows) != _BOOTSTRAP_PAIRS
    or len(diffusion.bootstrap_rows) != _BOOTSTRAP_PAIRS
  ):
    result["reason"] = "bootstrap_incomplete"
    return result
  if (
    baseline.bootstrap_end_state is None
    or diffusion.bootstrap_end_state is None
    or baseline.bootstrap_end_conditioning is None
    or diffusion.bootstrap_end_conditioning is None
    or baseline.bootstrap_end_tick is None
    or diffusion.bootstrap_end_tick is None
    or baseline.bootstrap_end_sim_time_s is None
    or diffusion.bootstrap_end_sim_time_s is None
    or baseline.bootstrap_end_segment is None
    or diffusion.bootstrap_end_segment is None
  ):
    result["reason"] = "missing_bootstrap_end_state"
    return result
  end_state_diff = _state_max_abs_diff(
    baseline.bootstrap_end_state, diffusion.bootstrap_end_state
  )
  end_condition_diff = float(
    np.max(
      np.abs(baseline.bootstrap_end_conditioning - diffusion.bootstrap_end_conditioning)
    )
  )
  end_time_diff = abs(
    baseline.bootstrap_end_sim_time_s - diffusion.bootstrap_end_sim_time_s
  )
  end_ticks_match = baseline.bootstrap_end_tick == diffusion.bootstrap_end_tick
  end_segments_match = baseline.bootstrap_end_segment == diffusion.bootstrap_end_segment
  result.update(
    {
      "bootstrap_end_state_max_abs_diff": end_state_diff,
      "bootstrap_end_conditioning_max_abs_diff": end_condition_diff,
      "bootstrap_end_ticks_match": end_ticks_match,
      "bootstrap_end_segments_match": end_segments_match,
      "bootstrap_end_time_max_abs_diff": end_time_diff,
    }
  )
  state_diff = condition_diff = action_diff = latent_diff = reference_diff = 0.0
  ticks_match = segments_match = True
  for left, right in zip(
    baseline.bootstrap_rows, diffusion.bootstrap_rows, strict=True
  ):
    state_diff = max(state_diff, _state_max_abs_diff(left.state, right.state))
    condition_diff = max(
      condition_diff, float(np.max(np.abs(left.conditioning - right.conditioning)))
    )
    action_diff = max(action_diff, float(np.max(np.abs(left.action - right.action))))
    latent_diff = max(latent_diff, float(np.max(np.abs(left.latent - right.latent))))
    if left.reference is None or right.reference is None:
      if left.reference is not right.reference:
        reference_diff = float("inf")
    else:
      reference_diff = max(
        reference_diff, float(np.max(np.abs(left.reference - right.reference)))
      )
    ticks_match &= left.tick == right.tick
    segments_match &= left.segment_id == right.segment_id
  result.update(
    {
      "state_max_abs_diff": state_diff,
      "conditioning_max_abs_diff": condition_diff,
      "action_max_abs_diff": action_diff,
      "latent_max_abs_diff": latent_diff,
      "reference_max_abs_diff": reference_diff,
      "ticks_match": ticks_match,
      "segments_match": segments_match,
    }
  )
  passed = bool(
    signature_diff <= tolerance
    and state_diff <= tolerance
    and condition_diff <= tolerance
    and action_diff <= tolerance
    and latent_diff <= tolerance
    and reference_diff <= tolerance
    and end_state_diff <= tolerance
    and end_condition_diff <= tolerance
    and end_time_diff <= tolerance
    and end_ticks_match
    and end_segments_match
    and ticks_match
    and segments_match
  )
  result["passed"] = passed
  if not passed:
    result["reason"] = "bootstrap_or_initialization_mismatch"
  return result


def _write_atomic(path: Path, data: bytes) -> None:
  temporary = path.with_name(f".{path.name}.tmp")
  temporary.write_bytes(data)
  temporary.replace(path)


def _marker_digest(value: Any) -> str:
  encoded = json.dumps(
    _json_value(value), sort_keys=True, separators=(",", ":")
  ).encode()
  return hashlib.sha256(encoded).hexdigest()


def _compact_task_marker(value: Mapping[str, Any]) -> dict[str, Any]:
  """Reduce a bounded task summary to the bounded marker fields.

  The minimum-cap marker must still honor the task identity, the executed
  counts and the realized/unreached target flags.  Per-step norms are digested
  into one payload hash instead of being repeated entry by entry.
  """
  compact: dict[str, Any] = {"payload_sha256": _marker_digest(value)}
  for field in ("schema", "mode", "config_sha256", "guidance_strength"):
    item = value.get(field)
    if isinstance(item, (str, int, float, bool, type(None))):
      compact[field] = item[:128] if isinstance(item, str) else item
  source = value.get("source")
  if isinstance(source, Mapping):
    compact["source_kind"] = source.get("kind")
    compact["source_qualification"] = source.get("qualification")
    compact["source_verification"] = source.get("verification")
    compact["source_sha256"] = source.get("sha256")
  scheduled = value.get("scheduled")
  if isinstance(scheduled, Mapping):
    compact["target_count"] = scheduled.get("target_count")
    compact["keyframe_count"] = len(scheduled.get("keyframe_ticks") or ())
    compact["goal_count"] = len(scheduled.get("goal_ticks") or ())
  executed = value.get("executed")
  if isinstance(executed, Mapping):
    compact["executed"] = {
      field: executed.get(field)
      for field in (
        "plan_steps",
        "guided_steps",
        "keyframe_steps",
        "denoiser_calls",
        "vjp_calls",
        "cost_calls",
      )
    }
  outcome = value.get("outcome")
  if isinstance(outcome, Mapping):
    compact["outcome"] = {
      field: outcome.get(field)
      for field in (
        "scored_targets",
        "unreached_targets",
        "missing_targets",
        "expired_targets",
      )
    }
  if "source_fingerprint" in value:
    compact["source_fingerprint"] = _json_value(value["source_fingerprint"])
  compact["has_error"] = bool(value.get("outcome", {}).get("stop_reasons"))
  compact["error_flags"] = {
    "unreached_targets": bool(compact.get("outcome", {}).get("unreached_targets")),
    "missing_targets": bool(compact.get("outcome", {}).get("missing_targets")),
    "expired_targets": bool(compact.get("outcome", {}).get("expired_targets")),
  }
  return compact


def _after_run_marker(report: EvaluationReport) -> dict[str, Any]:
  """Keep final status and digests, not unbounded model/file metadata."""
  after: dict[str, Any] = {}
  for key in (
    "integrity",
    "execution",
    "wall_status",
    "wall_time",
    "wall_seconds",
    "after_run_integrity",
  ):
    if key not in report.provenance:
      continue
    value = _json_value(report.provenance[key])
    if isinstance(value, Mapping):
      compact: dict[str, Any] = {"payload_sha256": _marker_digest(value)}
      for field in (
        "unchanged",
        "within_wall_budget",
        "partial_outcomes_retained",
        "wall_seconds",
        "max_wall_seconds",
        "wall_status",
        "status",
      ):
        item = value.get(field)
        if field in value and isinstance(item, (str, int, float, bool, type(None))):
          compact[field] = item[:128] if isinstance(item, str) else item
      if "error" in value:
        compact["has_error"] = bool(value["error"])
      for field in ("before", "after"):
        if field in value:
          compact[f"{field}_sha256"] = _marker_digest(value[field])
      after[key] = compact
    elif isinstance(value, (str, int, float, bool, type(None))):
      after[key] = value[:128] if isinstance(value, str) else value
    else:
      after[key] = {"payload_sha256": _marker_digest(value)}
  task = report.provenance.get("task")
  if isinstance(task, Mapping):
    after["task"] = _compact_task_marker(cast("Mapping[str, Any]", task))
  return after


def _tick_lines(payload: Mapping[str, Any]) -> list[bytes]:
  lines: list[bytes] = []
  branches = payload.get("branches", {})
  if not isinstance(branches, Mapping):
    return lines
  for branch_name in ("baseline", "diffusion"):
    branch = branches.get(branch_name)
    if not isinstance(branch, Mapping):
      continue
    ticks = branch.get("ticks", [])
    if not isinstance(ticks, list):
      continue
    for tick in ticks:
      lines.append(
        (
          json.dumps({"branch": branch_name, "tick": tick}, sort_keys=True) + "\n"
        ).encode()
      )
  return lines


def _serialize_report_payload(
  report: EvaluationReport,
  output: Mapping[str, Any],
  *,
  omit_ticks: bool,
) -> tuple[bytes, list[bytes]]:
  payload = report.as_dict()
  payload["output"] = dict(output)
  if omit_ticks:
    branches = payload.get("branches", {})
    if isinstance(branches, dict):
      for branch in branches.values():
        if isinstance(branch, dict):
          branch["ticks"] = []
          branch["ticks_omitted"] = True
  summary = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
  return summary, [] if omit_ticks else _tick_lines(payload)


def _output_metadata(
  output_dir: Path,
  report: EvaluationReport,
  *,
  representation: str,
  truncated: bool,
  omitted_rows: int,
  actual_bytes: int,
  error: str | None = None,
) -> dict[str, Any]:
  return {
    "directory": str(output_dir),
    "summary": str(output_dir / "summary.json"),
    "ticks": str(output_dir / "ticks.jsonl"),
    "representation": representation,
    "truncated": truncated,
    "omitted_row_count": int(omitted_rows),
    "actual_bytes": int(actual_bytes),
    "cap_bytes": report.budget.output_cap_bytes,
    "error": error,
  }


def _persist_evaluation_report_once(
  report: EvaluationReport, output_dir: Path
) -> dict[str, Any]:
  """Serialize one report through the bounded full/compact/marker path."""
  if output_dir.exists() and not output_dir.is_dir():
    raise ClosedLoopEvaluationError(f"output path is not a directory: {output_dir}")
  output_dir.mkdir(parents=True, exist_ok=True)
  cap = report.budget.output_cap_bytes
  payload = report.as_dict()
  branches = payload.get("branches", {})
  omitted_rows = 0
  if isinstance(branches, Mapping):
    for branch in branches.values():
      if isinstance(branch, Mapping) and isinstance(branch.get("ticks"), list):
        omitted_rows += len(branch["ticks"])

  full_output = _output_metadata(
    output_dir,
    report,
    representation="full",
    truncated=False,
    omitted_rows=0,
    actual_bytes=0,
  )
  full_summary, full_ticks = _serialize_report_payload(
    report, full_output, omit_ticks=False
  )
  full_bytes = len(full_summary) + sum(len(line) for line in full_ticks)
  full_output["actual_bytes"] = full_bytes
  full_summary, full_ticks = _serialize_report_payload(
    report, full_output, omit_ticks=False
  )
  full_bytes = len(full_summary) + sum(len(line) for line in full_ticks)
  if full_bytes <= cap:
    _write_atomic(output_dir / "summary.json", full_summary)
    _write_atomic(output_dir / "ticks.jsonl", b"".join(full_ticks))
    full_output["actual_bytes"] = full_bytes
    return full_output

  compact_output = _output_metadata(
    output_dir,
    report,
    representation="compact",
    truncated=True,
    omitted_rows=omitted_rows,
    actual_bytes=0,
  )
  compact_summary, _ = _serialize_report_payload(
    report, compact_output, omit_ticks=True
  )
  compact_bytes = len(compact_summary)
  compact_output["actual_bytes"] = compact_bytes
  compact_summary, _ = _serialize_report_payload(
    report, compact_output, omit_ticks=True
  )
  compact_bytes = len(compact_summary)
  if compact_bytes <= cap:
    _write_atomic(output_dir / "summary.json", compact_summary)
    _write_atomic(output_dir / "ticks.jsonl", b"")
    compact_output["actual_bytes"] = compact_bytes
    return compact_output

  marker_output = _output_metadata(
    output_dir,
    report,
    representation="marker",
    truncated=True,
    omitted_rows=omitted_rows,
    actual_bytes=0,
  )
  marker: dict[str, Any] = {
    "schema": "mjlab-d3-closed-loop-evaluation-v1",
    "status": report.status,
    "paired": report.paired,
    "valid": report.valid,
    "closed_loop": report.closed_loop,
    "synthetic": report.synthetic,
    "truncation": {
      "truncated": True,
      "omitted_row_count": omitted_rows,
    },
    "output": marker_output,
  }
  after_run = _after_run_marker(report)
  if after_run:
    marker["after_run"] = after_run
  marker_bytes = (json.dumps(marker, separators=(",", ":")) + "\n").encode()
  for _ in range(4):
    marker_output["actual_bytes"] = len(marker_bytes)
    marker["output"] = marker_output
    marker_bytes = (json.dumps(marker, separators=(",", ":")) + "\n").encode()
  if len(marker_bytes) > cap:
    raise ClosedLoopEvaluationError(
      "minimum output cap is too small for the bounded marker"
    )
  _write_atomic(output_dir / "summary.json", marker_bytes)
  _write_atomic(output_dir / "ticks.jsonl", b"")
  marker_output["actual_bytes"] = len(marker_bytes)
  return marker_output


def persist_evaluation_report(
  report: EvaluationReport, output_dir: str | Path
) -> EvaluationReport:
  """Persist initial or final report state through one bounded atomic path.

  The destination may already contain this run's summary/tick files so the
  integrator can call this helper again after adding integrity and wall-time
  provenance.  The in-memory report retains its bounded tick records; only the
  persisted representation is compacted or reduced to a marker.
  """
  try:
    metadata = _persist_evaluation_report_once(report, Path(output_dir))
  except Exception as exc:
    metadata = {
      "directory": str(output_dir),
      "summary": str(Path(output_dir) / "summary.json"),
      "ticks": str(Path(output_dir) / "ticks.jsonl"),
      "representation": "marker",
      "truncated": True,
      "omitted_row_count": sum(
        len(branch.ticks) for branch in (report.baseline, report.diffusion) if branch
      ),
      "actual_bytes": 0,
      "cap_bytes": report.budget.output_cap_bytes,
      "error": str(exc),
    }
    object.__setattr__(report, "status", "partial")
    object.__setattr__(
      report, "errors", (*report.errors, f"report persistence failed: {exc}")
    )
  report.output.clear()
  report.output.update(metadata)
  return report


def _runtime_provenance(runtime: Any) -> dict[str, Any]:
  value = getattr(runtime, "provenance", {})
  if not isinstance(value, Mapping):
    return {}
  try:
    return copy.deepcopy(dict(value))
  except Exception as exc:
    raise ClosedLoopEvaluationError(
      f"runtime provenance is not safely copyable: {exc}"
    ) from exc


def _final_branch_provenance(runtime: Any, builder: _BranchBuilder) -> dict[str, Any]:
  """Snapshot runtime metadata before close, adding evaluator-owned evidence."""
  provenance = _runtime_provenance(runtime)
  if builder.initial_signature_digest is not None:
    provenance.setdefault(
      "initial_state_signature_digest", builder.initial_signature_digest
    )
  handover = copy.deepcopy(builder.handover_record)
  provenance.setdefault("handover", handover)
  provenance["evaluator_handover"] = copy.deepcopy(handover)
  provenance["accepted_step_count"] = int(builder.accepted_step_count)
  provenance["untrusted_terminal_evidence"] = bool(builder.untrusted_terminal_evidence)
  return provenance


def _branch_has_software_failure(branch: BranchReport | None) -> bool:
  if branch is None:
    return True
  return (
    bool(branch.errors)
    or bool(branch.provenance.get("untrusted_terminal_evidence", False))
    or branch.stop_reason
    in {
      "runtime_error",
      "inference_error",
      "history_record_error",
      "nonfinite",
      "reset",
      "teleport",
      "missing_terminal_evidence",
      "segment_change",
      "tick_discontinuity",
      "period_discontinuity",
      "reference_leakage",
      "handover_state_changed",
      "handover_clock_changed",
    }
  )


def _branch_is_physical(branch: BranchReport | None) -> bool:
  if branch is None:
    return False
  provenance = branch.provenance
  try:
    accepted_steps = int(provenance.get("accepted_step_count", 0))
  except (TypeError, ValueError):
    accepted_steps = 0
  return bool(
    provenance.get("physical_runtime") == "mjlab"
    and provenance.get("synthetic") is not True
    and accepted_steps > 0
  )


def _run_branch(
  *,
  name: str,
  runtime_factory: RuntimeFactory,
  model: VaeModel,
  continuation_factory: ContinuationFactory | None,
  seed: int,
  start_frame: int,
  budget: EvaluationBudget,
  model_device: torch.device,
  latent_normalizer: LatentNormalizer | None = None,
  baseline: _BranchBuilder | None = None,
) -> tuple[BranchReport, _BranchBuilder]:
  builder = _BranchBuilder(name, seed, start_frame)
  runtime: ClosedLoopRuntime | None = None
  current: D3Observation | None = None
  try:
    runtime = runtime_factory()
    _validate_runtime(runtime, budget)
    if runtime is None:
      raise ClosedLoopEvaluationError("runtime factory returned None")
    if start_frame < 0 or start_frame >= int(runtime.clip_frames):
      raise ClosedLoopEvaluationError(
        f"start_frame {start_frame} is outside clip [0, {runtime.clip_frames})"
      )
    builder.runtime_provenance = _runtime_provenance(runtime)
    current = _validate_observation(runtime.reset(seed=seed, start_frame=start_frame))
    builder.initial_time_s = current.sim_time_s
    builder.last_time_s = current.sim_time_s
    builder.initial_reference = current.reference_frame
    builder.initial_signature = _initial_signature(runtime)
    builder.initial_signature_digest = hashlib.sha256(
      np.asarray(builder.initial_signature, dtype=np.float64).tobytes()
    ).hexdigest()
    builder.observe_state(current.state)
    current = _run_bootstrap(
      runtime,
      builder,
      current,
      budget=budget,
      reference_free=False,
      model=model,
      model_device=model_device,
      latent_normalizer=latent_normalizer,
    )
    if current is None:
      pass
    else:
      builder.bootstrap_end_state = _copy_state(current.state)
      builder.bootstrap_end_conditioning = np.array(current.conditioning, copy=True)
      builder.bootstrap_end_tick = current.tick
      builder.bootstrap_end_sim_time_s = current.sim_time_s
      builder.bootstrap_end_segment = current.segment_id
      if name == "baseline":
        _score_baseline(
          runtime,
          builder,
          current,
          model=model,
          model_device=model_device,
          budget=budget,
          latent_normalizer=latent_normalizer,
        )
      elif baseline is None:
        builder.stop("pair_mismatch", "baseline bootstrap is unavailable")
      else:
        pairing = _compare_bootstrap(
          baseline,
          builder,
          tolerance=budget.comparison_tolerance,
        )
        if not pairing["passed"]:
          builder.stop("pair_mismatch", str(pairing))
        elif continuation_factory is None:
          builder.stop("runtime_error", "continuation_factory is required")
        else:
          continuation = continuation_factory()
          if not all(
            callable(getattr(continuation, attribute, None))
            for attribute in ("record_executed", "step")
          ):
            raise ClosedLoopEvaluationError(
              "continuation_factory must return a reference-free continuation"
            )
          for row in builder.bootstrap_rows:
            continuation.record_executed(row.state, row.latent)
          handover_state = _copy_state(current.state)
          handover_tick, handover_time, handover_segment = (
            current.tick,
            current.sim_time_s,
            current.segment_id,
          )
          handed = _validate_observation(runtime.enter_reference_free())
          builder.handover_record = {
            "entered": True,
            "advanced_simulation": False,
            "reset_physical_state": False,
            "reference_free_observation": True,
          }
          if (
            _state_max_abs_diff(handover_state, handed.state)
            > budget.comparison_tolerance
          ):
            builder.stop("handover_state_changed")
          elif (
            handed.tick != handover_tick
            or not math.isclose(
              handed.sim_time_s, handover_time, abs_tol=_TIME_TOLERANCE
            )
            or handed.segment_id != handover_segment
          ):
            builder.stop("handover_clock_changed")
          elif handed.reference is not None or handed.reference_frame is not None:
            builder.stop("reference_leakage")
          else:
            current = handed
            _score_diffusion(runtime, builder, current, continuation, budget=budget)
  except Exception as exc:
    builder.add_error(exc)
  finally:
    if runtime is not None:
      try:
        builder.runtime_provenance = _final_branch_provenance(runtime, builder)
      except Exception as exc:
        builder.add_error(f"runtime provenance snapshot failed: {exc}")
      try:
        runtime.close()
      except Exception as exc:
        builder.add_error(f"runtime close failed: {exc}")
  return builder.finish(last_time_s=builder.last_time_s), builder


def _check_output_destination(output_dir: str | Path | None) -> Path | None:
  if output_dir is None:
    return None
  path = Path(output_dir)
  if path.exists():
    if not path.is_dir():
      raise ClosedLoopEvaluationError(f"output path is not a directory: {path}")
    if any(path.iterdir()):
      raise ClosedLoopEvaluationError(
        f"refusing to overwrite existing closed-loop output directory: {path}"
      )
  return path


def run_matched_pair(
  runtime_factory: RuntimeFactory,
  *,
  model: VaeModel,
  continuation_factory: ContinuationFactory,
  seed: int,
  start_frame: int,
  budget: EvaluationBudget | None = None,
  device: str | torch.device = "cpu",
  latent_normalizer: LatentNormalizer | None = None,
  output_dir: str | Path | None = None,
  provenance: Mapping[str, Any] | None = None,
) -> EvaluationReport:
  """Run one baseline then one matched reference-free diffusion branch.

  Parameters
  ----------
  runtime_factory:
    Zero-argument factory producing a fresh runtime for each branch.  The
    runner calls ``reset(seed=seed, start_frame=start_frame)`` on each result
    and always calls ``close`` exactly once when construction succeeded.
  model:
    Frozen gravity-mode VAE implementing ``encode(reference)`` and
    ``decode(raw_mu, conditioning)``.  The runner never samples a latent and
    never normalizes conditioning outside ``model.decode``.
  continuation_factory:
    Zero-argument factory producing a fresh configured
    :class:`ReferenceFreeContinuation` for the diffusion branch.  The runner
    records the eight accepted bootstrap pairs before calling its ``step``.
  output_dir:
    Optional new directory receiving atomic ``summary.json`` and bounded
    ``ticks.jsonl`` files.  Existing non-empty directories are refused.
  latent_normalizer:
    Optional frozen raw-to-normalized latent transform used only for reported
    norm diagnostics; no normalization is applied before VAE decoding.

  The returned report is evidence, not a policy-quality verdict.  In
  particular, ``survival_observed`` does not certify a collapsed low-tilt
  pose, and pelvis height/contact/torque values are descriptive or unavailable
  unless the runtime captures them with known semantics.
  """
  if budget is None:
    budget = EvaluationBudget()
  if not callable(runtime_factory):
    raise ClosedLoopEvaluationError("runtime_factory must be callable")
  if not hasattr(model, "encode") or not hasattr(model, "decode"):
    raise ClosedLoopEvaluationError("model must expose encode and decode")
  if not callable(continuation_factory):
    raise ClosedLoopEvaluationError("continuation_factory must be callable")
  if latent_normalizer is not None and not callable(latent_normalizer):
    raise ClosedLoopEvaluationError("latent_normalizer must be callable or None")
  if not _is_integer(seed) or int(seed) < 0:
    raise ClosedLoopEvaluationError("seed must be a nonnegative integer")
  if not _is_integer(start_frame) or int(start_frame) < 0:
    raise ClosedLoopEvaluationError("start_frame must be a nonnegative integer")
  destination = _check_output_destination(output_dir)
  try:
    resolved_device = torch.device(device)
    torch.empty(0, device=resolved_device)
  except (RuntimeError, ValueError) as exc:
    raise ClosedLoopEvaluationError(f"invalid model device {device!r}") from exc
  actual_device = _model_device(model, resolved_device)
  if actual_device.type != resolved_device.type and hasattr(model, "parameters"):
    # A model with parameters determines its own input device; a parameterless
    # fake model continues to use the explicit device above.
    resolved_device = actual_device

  baseline_report, baseline_builder = _run_branch(
    name="baseline",
    runtime_factory=runtime_factory,
    model=model,
    continuation_factory=None,
    seed=int(seed),
    start_frame=int(start_frame),
    budget=budget,
    model_device=resolved_device,
    latent_normalizer=latent_normalizer,
  )
  diffusion_report, _diffusion_builder = _run_branch(
    name="diffusion",
    runtime_factory=runtime_factory,
    model=model,
    continuation_factory=continuation_factory,
    seed=int(seed),
    start_frame=int(start_frame),
    budget=budget,
    model_device=resolved_device,
    baseline=baseline_builder,
  )
  pairing = _compare_bootstrap(
    baseline_builder,
    _diffusion_builder,
    tolerance=budget.comparison_tolerance,
  )
  paired = bool(pairing["passed"])
  errors = tuple([*baseline_report.errors, *diffusion_report.errors])
  software_failure = _branch_has_software_failure(
    baseline_report
  ) or _branch_has_software_failure(diffusion_report)
  status = "complete" if paired and not software_failure else "partial"
  supplied_provenance = dict(provenance or {})
  synthetic = bool(
    supplied_provenance.get("synthetic", False)
    or baseline_report.provenance.get("synthetic", False)
    or diffusion_report.provenance.get("synthetic", False)
  )
  physical_evidence = _branch_is_physical(baseline_report) and _branch_is_physical(
    diffusion_report
  )
  closed_loop = bool(physical_evidence and not synthetic)
  report = EvaluationReport(
    status=status,
    paired=paired,
    closed_loop=closed_loop,
    synthetic=synthetic,
    seed=int(seed),
    start_frame=int(start_frame),
    budget=budget,
    baseline=baseline_report,
    diffusion=diffusion_report,
    pairing=pairing,
    provenance={
      **supplied_provenance,
      "runner": "pure-evaluation-runner",
      "physical_evidence": physical_evidence,
      "runtime_branches": [
        dict(baseline_report.provenance),
        dict(diffusion_report.provenance),
      ],
      "model_inputs": {
        "vae_reference_width": _REFERENCE_WIDTH,
        "vae_conditioning_width": _CONDITIONING_WIDTH,
        "latent_width": _LATENT_WIDTH,
        "action_width": _ACTION_WIDTH,
        "conditioning_normalized_by": "model.decode",
        "physical_evidence": physical_evidence,
      },
    },
    errors=errors,
    output={},
  )
  if destination is not None:
    persist_evaluation_report(report, destination)
  return report


__all__ = [
  "BranchReport",
  "ClosedLoopBudget",
  "ClosedLoopEvaluationError",
  "ContinuationFactory",
  "EvaluationBudget",
  "EvaluationReport",
  "LatentNormalizer",
  "MIN_OUTPUT_CAP_BYTES",
  "RunBudget",
  "RunnerConfig",
  "RuntimeFactory",
  "VaeModel",
  "persist_evaluation_report",
  "run_matched_pair",
]
