"""Pure perturbation, scheduling, quaternion and trigger helpers for P1."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from mjlab.tasks.tracking.distillation.coverage_policy import (
  CoveragePolicy,
  CoverageTier,
  ExcitationKind,
  RootDirection,
  choose_excitation,
  choose_joint_chain,
  choose_root_direction,
  choose_tier,
  draw_uniform,
)


class CoveragePerturbationError(ValueError):
  """Invalid pure perturbation input."""


ZERO_VECTOR: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _vec(values: Sequence[float]) -> tuple[float, float, float]:
  if len(values) != 3 or not all(math.isfinite(float(value)) for value in values):
    raise CoveragePerturbationError("a finite 3-vector is required")
  return tuple(float(value) for value in values)  # type: ignore[return-value]


def _draw(rng: Any, magnitude: float) -> float:
  return draw_uniform(rng, magnitude) if magnitude else 0.0


def _env(policy: CoveragePolicy, tier: CoverageTier):
  try:
    return dict(policy.root_reset_envelopes)[tier]
  except KeyError as exc:
    raise CoveragePerturbationError(f"missing envelope for {tier}") from exc


def _mag(env: Any, field: str, axis: str) -> float:
  return dict(getattr(env, field))[axis]


@dataclass(frozen=True, slots=True)
class ResetPerturbation:
  tier: CoverageTier
  excitation: ExcitationKind
  root_direction: RootDirection
  joint_chain: str | None
  position_delta_m: tuple[float, float, float]
  rotation_delta_rad: tuple[float, float, float]
  linear_velocity_delta_m_s: tuple[float, float, float]
  angular_velocity_delta_rad_s: tuple[float, float, float]
  joint_position_delta_rad: tuple[tuple[str, float], ...]
  joint_velocity_delta_rad_s: tuple[tuple[str, float], ...]

  @property
  def is_noop(self) -> bool:
    values = (
      self.position_delta_m
      + self.rotation_delta_rad
      + self.linear_velocity_delta_m_s
      + self.angular_velocity_delta_rad_s
    )
    return not any(abs(value) for value in values) and not any(
      abs(value)
      for _, value in self.joint_position_delta_rad + self.joint_velocity_delta_rad_s
    )


@dataclass(frozen=True, slots=True)
class RuntimePerturbation:
  tier: CoverageTier
  excitation: ExcitationKind
  root_direction: RootDirection
  joint_chain: str | None
  linear_velocity_delta_m_s: tuple[float, float, float]
  angular_velocity_delta_rad_s: tuple[float, float, float]
  joint_velocity_delta_rad_s: tuple[tuple[str, float], ...]

  @property
  def is_noop(self) -> bool:
    return not any(
      abs(value)
      for value in self.linear_velocity_delta_m_s
      + self.angular_velocity_delta_rad_s
      + tuple(value for _, value in self.joint_velocity_delta_rad_s)
    )


def _root_draw(
  policy: CoveragePolicy,
  rng: Any,
  tier: CoverageTier,
  direction: RootDirection,
  *,
  runtime: bool,
  enhance: bool = True,
) -> tuple[
  tuple[float, float, float],
  tuple[float, float, float],
  tuple[float, float, float],
  tuple[float, float, float],
]:
  base = _env(policy, "baseline")
  selected = _env(policy, tier)
  if runtime:
    position, rotation = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    linear, angular = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    if direction == "sagittal":
      linear[0] = _draw(rng, _mag(selected, "linear_velocity_m_s", "x"))
      angular[1] = _draw(rng, _mag(selected, "angular_velocity_rad_s", "pitch"))
    elif direction == "lateral":
      linear[1] = _draw(rng, _mag(selected, "linear_velocity_m_s", "y"))
      angular[0] = _draw(rng, _mag(selected, "angular_velocity_rad_s", "roll"))
    else:
      angular[2] = _draw(rng, _mag(selected, "angular_velocity_rad_s", "yaw"))
  else:
    # Build the complete envelope first, then make exactly one draw per field.
    # A direction replaces only its selected reset axes; every other axis stays
    # at the existing baseline envelope.  This is important for RNG parity.
    position_m = {axis: _mag(base, "position_m", axis) for axis in ("x", "y", "z")}
    rotation = {
      axis: _mag(base, "rotation_rad", axis) for axis in ("roll", "pitch", "yaw")
    }
    linear = {axis: _mag(base, "linear_velocity_m_s", axis) for axis in ("x", "y", "z")}
    angular = {
      axis: _mag(base, "angular_velocity_rad_s", axis)
      for axis in ("roll", "pitch", "yaw")
    }
    if enhance:
      if direction == "sagittal":
        rotation["pitch"] = _mag(selected, "rotation_rad", "pitch")
        angular["pitch"] = _mag(selected, "angular_velocity_rad_s", "pitch")
      elif direction == "lateral":
        rotation["roll"] = _mag(selected, "rotation_rad", "roll")
        angular["roll"] = _mag(selected, "angular_velocity_rad_s", "roll")
      elif direction == "yaw":
        rotation["yaw"] = _mag(selected, "rotation_rad", "yaw")
        angular["yaw"] = _mag(selected, "angular_velocity_rad_s", "yaw")
      else:
        axis = (
          int(rng.integers(0, 2)) if hasattr(rng, "integers") else int(rng.randrange(2))
        )
        axis_name = ("x", "y")[axis]
        position_m[axis_name] = _mag(selected, "position_m", axis_name)
        linear[axis_name] = _mag(selected, "linear_velocity_m_s", axis_name)
    position = [_draw(rng, position_m[axis]) for axis in ("x", "y", "z")]
    rotation_values = [_draw(rng, rotation[axis]) for axis in ("roll", "pitch", "yaw")]
    linear_values = [_draw(rng, linear[axis]) for axis in ("x", "y", "z")]
    angular_values = [_draw(rng, angular[axis]) for axis in ("roll", "pitch", "yaw")]
    return (
      _vec(position),
      _vec(rotation_values),
      _vec(linear_values),
      _vec(angular_values),
    )
  return _vec(position), _vec(rotation), _vec(linear), _vec(angular)


def _joints(
  policy: CoveragePolicy,
  rng: Any,
  tier: CoverageTier,
  chain: str | None,
  *,
  runtime: bool,
) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, float], ...]]:
  if chain is None:
    return (), ()
  names = dict(policy.joint_chains)[chain]
  positions = dict(dict(policy.joint_position_envelopes_rad).get(tier, ()))
  velocities = dict(dict(policy.joint_velocity_delta_envelopes_rad_s).get(tier, ()))
  position_deltas: list[tuple[str, float]] = []
  velocity_deltas: list[tuple[str, float]] = []
  for name in names:
    if runtime:
      position_magnitude = 0.0
      velocity_magnitude = velocities.get("selected_chain", 0.0)
      if tier == "strong" and "knee" in name:
        velocity_magnitude = velocities.get("knee", velocity_magnitude)
      position_delta = 0.0
      velocity_delta = _draw(rng, velocity_magnitude) if velocity_magnitude else 0.0
    else:
      role = next((key for key in positions if key != "all" and key in name), "all")
      if role == "all" and ("waist_pitch" in name or "waist_roll" in name):
        role = "waist_pitch_roll"
      position_magnitude = positions.get(role, positions.get("all", 0.05))
      velocity_magnitude = velocities.get("selected_chain", 0.0)
      if tier == "strong" and "knee" in name:
        velocity_magnitude = velocities.get("knee", velocity_magnitude)
      position_delta = _draw(rng, position_magnitude) if position_magnitude else 0.0
      velocity_delta = _draw(rng, velocity_magnitude) if velocity_magnitude else 0.0
    position_deltas.append((name, position_delta))
    velocity_deltas.append((name, velocity_delta))
  return tuple(position_deltas), tuple(velocity_deltas)


def sample_reset_perturbation(policy: CoveragePolicy, rng: Any) -> ResetPerturbation:
  """Draw a reset; disabled policy consumes no RNG and is a no-op."""
  if not policy.enabled:
    return ResetPerturbation(
      "baseline",
      "root_only",
      "sagittal",
      None,
      ZERO_VECTOR,
      ZERO_VECTOR,
      ZERO_VECTOR,
      ZERO_VECTOR,
      (),
      (),
    )
  tier = choose_tier(policy, rng)
  if tier == "baseline":
    return ResetPerturbation(
      "baseline",
      "root_only",
      "sagittal",
      None,
      ZERO_VECTOR,
      ZERO_VECTOR,
      ZERO_VECTOR,
      ZERO_VECTOR,
      (),
      (),
    )
  excitation = choose_excitation(policy, rng)
  direction = choose_root_direction(policy, rng)
  chain = None if excitation == "root_only" else choose_joint_chain(policy, rng)
  position, rotation, linear, angular = _root_draw(
    policy,
    rng,
    tier,
    direction,
    runtime=False,
    enhance=excitation != "joint_chain_only",
  )
  joint_position, joint_velocity = _joints(policy, rng, tier, chain, runtime=False)
  return ResetPerturbation(
    tier,
    excitation,
    direction,
    chain,
    position,
    rotation,
    linear,
    angular,
    joint_position,
    joint_velocity,
  )


def sample_runtime_perturbation(
  policy: CoveragePolicy, rng: Any
) -> RuntimePerturbation:
  """Draw an additive world-frame velocity event; never a position teleport."""
  if not policy.enabled:
    return RuntimePerturbation(
      "baseline", "root_only", "sagittal", None, ZERO_VECTOR, ZERO_VECTOR, ()
    )
  tier = choose_tier(policy, rng)
  excitation = choose_excitation(policy, rng)
  direction = choose_root_direction(policy, rng, runtime=True)
  chain = None if excitation == "root_only" else choose_joint_chain(policy, rng)
  _, _, linear, angular = _root_draw(
    policy, rng, tier, direction, runtime=True, enhance=True
  )
  if excitation == "joint_chain_only":
    linear, angular = ZERO_VECTOR, ZERO_VECTOR
  _, joint_velocity = _joints(policy, rng, tier, chain, runtime=True)
  return RuntimePerturbation(
    tier,
    excitation,
    direction,
    chain,
    linear,
    angular,
    joint_velocity,
  )


@dataclass(frozen=True, slots=True)
class PhaseEvent:
  frame: int
  phase_bin: int
  phase_bins: int


def schedule_phase_event(
  policy: CoveragePolicy,
  rng: Any,
  *,
  current_reference_frame: int,
  initial_reference_frame: int,
  clip_frames: int,
  segment_age_steps: int,
) -> PhaseEvent | None:
  """Choose one future absolute reference frame in a real clip phase bin."""
  values = (
    current_reference_frame,
    initial_reference_frame,
    clip_frames,
    segment_age_steps,
  )
  if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
    raise CoveragePerturbationError("phase scheduling arguments must be integers")
  if (
    clip_frames <= 0
    or initial_reference_frame < 0
    or current_reference_frame < initial_reference_frame
    or current_reference_frame >= clip_frames
    or segment_age_steps < 0
    or current_reference_frame - initial_reference_frame != segment_age_steps
  ):
    raise CoveragePerturbationError("invalid current/initial frame or clip length")
  if not policy.enabled:
    return None
  if float(rng.random()) >= policy.runtime.per_segment_opportunity_probability:
    return None
  earliest = max(
    current_reference_frame + 1,
    initial_reference_frame + policy.runtime.min_segment_age_steps,
  )
  latest = clip_frames - 1 - policy.runtime.min_steps_remaining
  if earliest > latest:
    return None
  eligible = list(range(earliest, latest + 1))
  phase_bins = policy.runtime.target_phase_bins
  by_bin: dict[int, list[int]] = {phase_bin: [] for phase_bin in range(phase_bins)}
  for frame in eligible:
    by_bin[min(frame * phase_bins // clip_frames, phase_bins - 1)].append(frame)

  def choose(upper: int) -> int:
    return (
      int(rng.integers(0, upper))
      if hasattr(rng, "integers")
      else int(rng.randrange(upper))
    )

  nonempty = [phase_bin for phase_bin, frames in by_bin.items() if frames]
  phase_bin = nonempty[choose(len(nonempty))]
  frames = by_bin[phase_bin]
  return PhaseEvent(frames[choose(len(frames))], phase_bin, phase_bins)


def quaternion_multiply(left_wxyz: Any, right_wxyz: Any) -> np.ndarray:
  """Multiply finite ``wxyz`` quaternions without normalization."""
  a, b = np.asarray(left_wxyz, dtype=float), np.asarray(right_wxyz, dtype=float)
  if (
    a.shape != (4,)
    or b.shape != (4,)
    or not np.isfinite(a).all()
    or not np.isfinite(b).all()
  ):
    raise CoveragePerturbationError("quaternions must be finite shape-(4,) arrays")
  aw, ax, ay, az = a
  bw, bx, by, bz = b
  return np.array(
    (
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    )
  )


def world_xyz_delta_quaternion(delta_xyz_rad: Sequence[float]) -> np.ndarray:
  """Make an XYZ world-frame delta quaternion in ``wxyz`` order."""
  x, y, z = _vec(delta_xyz_rad)
  qx = np.array((math.cos(x / 2), math.sin(x / 2), 0.0, 0.0))
  qy = np.array((math.cos(y / 2), 0.0, math.sin(y / 2), 0.0))
  qz = np.array((math.cos(z / 2), 0.0, 0.0, math.sin(z / 2)))
  return quaternion_multiply(quaternion_multiply(qz, qy), qx)


def left_multiply_world_xyz_delta(
  quaternion_wxyz: Sequence[float], delta_xyz_rad: Sequence[float]
) -> np.ndarray:
  return quaternion_multiply(world_xyz_delta_quaternion(delta_xyz_rad), quaternion_wxyz)


def trigger_severity(metrics: Mapping[str, float], policy: CoveragePolicy) -> float:
  """Compute max reference-relative normalized severity (no outcome inputs)."""
  thresholds = dict(policy.triggers.thresholds)
  unknown = set(metrics) - set(thresholds)
  if unknown:
    raise CoveragePerturbationError(f"unknown trigger metrics: {sorted(unknown)}")
  values = []
  for name, threshold in thresholds.items():
    value = float(metrics.get(name, 0.0))
    if not math.isfinite(value) or value < 0:
      raise CoveragePerturbationError(
        f"trigger metric {name} must be finite and nonnegative"
      )
    values.append(value / threshold)
  return max(values, default=0.0)


def difficulty_band(score: float) -> Literal["nominal", "moderate", "strong"]:
  if not math.isfinite(score) or score < 0:
    raise CoveragePerturbationError("difficulty score must be finite and nonnegative")
  return "nominal" if score < 1 else "moderate" if score < 2 else "strong"


@dataclass(slots=True)
class TriggerState:
  armed: bool = True
  low_dwell: int = 0
  cooldown: int = 0
  pending: bool = False
  threshold_candidates: int = 0


def trigger_candidate(
  metrics: Mapping[str, float],
  policy: CoveragePolicy,
  state: TriggerState,
  *,
  scheduled: bool = False,
  terminal: bool = False,
) -> tuple[bool, float, str]:
  """Apply bounded crossing/cooldown/rearm logic."""
  if not policy.enabled:
    return False, 0.0, "disabled"
  score = trigger_severity(metrics, policy)
  high, low = score >= 1.0, score < policy.triggers.rearm_low_fraction
  if low:
    state.low_dwell += 1
    if state.low_dwell >= policy.triggers.low_dwell_steps:
      state.armed = True
      state.pending = False
  else:
    state.low_dwell = 0
  if state.cooldown:
    state.cooldown -= 1
  if scheduled:
    return True, score, "scheduled"
  if terminal:
    return True, score, "physical_termination"
  if state.threshold_candidates >= policy.triggers.max_threshold_candidates_per_segment:
    state.pending = False
    return False, score, "candidate_cap"
  # Only a genuinely armed high crossing may create a deferred candidate.
  # Remaining continuously high after an emission does not rearm the trigger.
  if state.cooldown == 0 and state.pending and high:
    state.armed, state.pending = False, False
    state.threshold_candidates += 1
    state.cooldown = policy.triggers.cooldown_steps
    return True, score, "threshold_crossing"
  if not high or not state.armed:
    return False, score, "cooldown_or_disarmed"
  state.armed = False
  if state.cooldown:
    state.pending = True
    return False, score, "cooldown"
  state.pending = False
  state.threshold_candidates += 1
  state.cooldown = policy.triggers.cooldown_steps
  return True, score, "threshold_crossing"


def _randbelow(rng: Any, upper: int) -> int:
  if upper <= 0:
    raise CoveragePerturbationError("random upper bound must be positive")
  if hasattr(rng, "integers"):
    return int(rng.integers(0, upper))
  if hasattr(rng, "randrange"):
    return int(rng.randrange(upper))
  raise CoveragePerturbationError("reservoir RNG must provide integers or randrange")


def _reservoir_sample(values: Sequence[Any], count: int, rng: Any) -> list[Any]:
  if count < 0 or count > len(values):
    raise CoveragePerturbationError("invalid within-stratum reservoir count")
  selected = list(values[:count])
  for index in range(count, len(values)):
    replacement = _randbelow(rng, index + 1)
    if replacement < count:
      selected[replacement] = values[index]
  return selected


def bounded_stratified_reservoir(
  candidates: Sequence[Any], capacity: int, rng: Any, *, key: Any = lambda item: item
) -> list[Any]:
  """Select a bounded reservoir while preserving nonempty strata when possible.

  If capacity covers all strata, every stratum receives one row and remaining
  slots are assigned by deterministic largest-remainder allocation. Each
  stratum then uses an ordinary reservoir sample, rather than permanently
  retaining its first row. If there are more strata than slots, a reproducible
  random permutation chooses which strata receive the unavoidable slots.
  """
  if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
    raise CoveragePerturbationError("capacity must be a nonnegative integer")
  if not callable(key):
    raise CoveragePerturbationError("key must be callable")
  if capacity == 0 or not candidates:
    return []
  groups: dict[Any, list[Any]] = {}
  for candidate in candidates:
    try:
      group_key = key(candidate)
      hash(group_key)
    except (TypeError, ValueError) as exc:
      raise CoveragePerturbationError("stratum key must be hashable") from exc
    groups.setdefault(group_key, []).append(candidate)
  ordered_keys = sorted(groups, key=lambda item: (type(item).__name__, repr(item)))
  target = min(capacity, len(candidates))
  counts: dict[Any, int] = {group_key: 0 for group_key in ordered_keys}
  if len(ordered_keys) > target:
    shuffled = list(ordered_keys)
    for index in range(len(shuffled) - 1, 0, -1):
      swap = _randbelow(rng, index + 1)
      shuffled[index], shuffled[swap] = shuffled[swap], shuffled[index]
    for group_key in shuffled[:target]:
      counts[group_key] = 1
  else:
    for group_key in ordered_keys:
      counts[group_key] = 1
    remaining = target - len(ordered_keys)
    capacities = {group_key: len(groups[group_key]) - 1 for group_key in ordered_keys}
    total_capacity = sum(capacities.values())
    if remaining and total_capacity:
      exact = {
        group_key: remaining * capacities[group_key] / total_capacity
        for group_key in ordered_keys
      }
      for group_key in ordered_keys:
        extra = min(capacities[group_key], int(math.floor(exact[group_key])))
        counts[group_key] += extra
        remaining -= extra
      ranked = sorted(
        ordered_keys,
        key=lambda group_key: (
          -(exact[group_key] - math.floor(exact[group_key])),
          type(group_key).__name__,
          repr(group_key),
        ),
      )
      for group_key in ranked:
        if remaining <= 0:
          break
        if counts[group_key] < len(groups[group_key]):
          counts[group_key] += 1
          remaining -= 1
  selected: list[Any] = []
  for group_key in ordered_keys:
    selected.extend(_reservoir_sample(groups[group_key], counts[group_key], rng))
  return selected


sample_phase_event = schedule_phase_event
apply_world_orientation_delta = left_multiply_world_xyz_delta
normalized_difficulty_score = trigger_severity
select_bounded_candidates = bounded_stratified_reservoir
