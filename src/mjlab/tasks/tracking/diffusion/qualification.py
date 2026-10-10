"""Fail-closed qualification rules for frozen-policy D1 trials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, cast

import numpy as np

from .contract import DEFAULT_CONTRACT
from .storage import TerminalEvidence


class QualificationError(ValueError):
  """Qualification evidence is malformed."""


@dataclass(frozen=True, slots=True)
class StepEvidence:
  """Post-step evidence captured before an environment reset can occur."""

  terminal_available: bool
  terminated: bool = False
  truncated: bool = False
  reason: str | None = None
  finite: bool = True
  physical_fall: bool = False
  tracking_rejection: bool = False
  reset: bool = False
  teleport: bool = False
  reference_boundary: bool = False
  clip_ended: bool = False

  def __post_init__(self) -> None:
    if self.reason is not None and not isinstance(self.reason, str):
      raise QualificationError("evidence reason must be a string or None")

  @property
  def terminal(self) -> TerminalEvidence:
    return TerminalEvidence(
      available=self.terminal_available,
      terminated=self.terminated,
      truncated=self.truncated,
      reason=self.reason,
      physical_fall=self.physical_fall,
      tracking_rejection=self.tracking_rejection,
    )


def _state_tilt_degrees(observation: Any) -> float:
  state = getattr(observation, "state", None)
  quaternion = getattr(state, "root_quaternion_wxyz", None)
  if quaternion is None:
    return float("inf")
  value = np.asarray(quaternion, dtype=np.float64).reshape(-1)
  if value.shape != (4,) or not np.isfinite(value).all():
    return float("inf")
  norm = np.linalg.norm(value)
  if norm == 0.0:
    return float("inf")
  value = value / norm
  projected_gravity_z = 1.0 - 2.0 * (value[1] ** 2 + value[2] ** 2)
  return float(np.degrees(np.arccos(np.clip(projected_gravity_z, -1.0, 1.0))))


def _bool(payload: Mapping[str, Any], name: str, default: bool = False) -> bool:
  value = payload.get(name, default)
  if not isinstance(value, (bool, np.bool_)):
    raise QualificationError(f"evidence field {name!r} must be boolean")
  return bool(value)


def parse_step_evidence(value: Any) -> StepEvidence:
  """Decode adapter evidence without inferring survival from a missing value."""
  if isinstance(value, StepEvidence):
    return value
  if value is None:
    return StepEvidence(False, finite=False, reason="missing_terminal_evidence")
  if isinstance(value, TerminalEvidence):
    return StepEvidence(
      value.available,
      value.terminated,
      value.truncated,
      value.reason,
      physical_fall=value.physical_fall,
      tracking_rejection=value.tracking_rejection,
    )
  if isinstance(value, Mapping):
    nested = value.get("terminal_evidence")
    if nested is not None:
      terminal = (
        nested
        if isinstance(nested, TerminalEvidence)
        else TerminalEvidence(
          _bool(nested, "available"),
          _bool(nested, "terminated"),
          _bool(nested, "truncated"),
          nested.get("reason"),
          _bool(nested, "physical_fall"),
          _bool(nested, "tracking_rejection"),
        )
      )
      return StepEvidence(
        terminal.available,
        terminal.terminated,
        terminal.truncated,
        terminal.reason,
        _bool(value, "finite", True),
        terminal.physical_fall,
        terminal.tracking_rejection,
        _bool(value, "reset"),
        _bool(value, "teleport"),
        _bool(value, "reference_boundary"),
        _bool(value, "clip_ended"),
      )
    if "available" not in value:
      return StepEvidence(False, finite=False, reason="missing_terminal_evidence")
    reason = value.get("reason")
    if reason is not None and not isinstance(reason, str):
      raise QualificationError("evidence reason must be a string or None")
    return StepEvidence(
      _bool(value, "available"),
      _bool(value, "terminated"),
      _bool(value, "truncated"),
      reason,
      _bool(value, "finite", True),
      _bool(value, "physical_fall"),
      _bool(value, "tracking_rejection"),
      _bool(value, "reset"),
      _bool(value, "teleport"),
      _bool(value, "reference_boundary"),
      _bool(value, "clip_ended"),
    )
  raise QualificationError(f"unsupported post-step evidence type {type(value)!r}")


def _batch_bool(name: str, value: Any, batch: int) -> np.ndarray:
  result = np.asarray(value, dtype=bool).reshape(-1)
  if result.shape != (batch,):
    raise QualificationError(f"{name} must have shape [{batch}]")
  return result.copy()

  if result.shape != (batch,):
    raise QualificationError(f"{name} must have shape [{batch}]")
  return result.copy()


@dataclass(frozen=True, slots=True)
class StepEvidenceBatch:
  """Post-step evidence stored as aligned ``[N]`` arrays."""

  terminal_available: np.ndarray
  terminated: np.ndarray
  truncated: np.ndarray
  reason: tuple[str | None, ...]
  finite: np.ndarray
  physical_fall: np.ndarray
  tracking_rejection: np.ndarray
  reset: np.ndarray
  teleport: np.ndarray
  reference_boundary: np.ndarray
  clip_ended: np.ndarray

  def __post_init__(self) -> None:
    batch = np.asarray(self.terminal_available).reshape(-1).size
    if batch == 0:
      raise QualificationError("evidence batches must be non-empty")
    for name in (
      "terminal_available",
      "terminated",
      "truncated",
      "finite",
      "physical_fall",
      "tracking_rejection",
      "reset",
      "teleport",
      "reference_boundary",
      "clip_ended",
    ):
      object.__setattr__(self, name, _batch_bool(name, getattr(self, name), batch))
    if len(self.reason) != batch:
      raise QualificationError("evidence reasons must match the batch size")
    if any(value is not None and not isinstance(value, str) for value in self.reason):
      raise QualificationError("evidence reasons must be strings or None")

  @property
  def batch_size(self) -> int:
    return self.terminal_available.size

  @classmethod
  def from_values(cls, values: Any) -> "StepEvidenceBatch":
    if isinstance(values, cls):
      return values
    if not isinstance(values, (tuple, list)):
      raise QualificationError("a batch needs StepEvidence values")
    parsed = tuple(parse_step_evidence(value) for value in values)
    if not parsed:
      raise QualificationError("evidence batches must be non-empty")
    return cls(
      np.asarray([value.terminal_available for value in parsed]),
      np.asarray([value.terminated for value in parsed]),
      np.asarray([value.truncated for value in parsed]),
      tuple(value.reason for value in parsed),
      np.asarray([value.finite for value in parsed]),
      np.asarray([value.physical_fall for value in parsed]),
      np.asarray([value.tracking_rejection for value in parsed]),
      np.asarray([value.reset for value in parsed]),
      np.asarray([value.teleport for value in parsed]),
      np.asarray([value.reference_boundary for value in parsed]),
      np.asarray([value.clip_ended for value in parsed]),
    )

  def select(self, indices: Any) -> "StepEvidenceBatch":
    ids = np.asarray(indices, dtype=np.int64).reshape(-1)
    if ids.size == 0 or np.any(ids < 0) or np.any(ids >= self.batch_size):
      raise QualificationError("evidence selection is outside the batch")
    return type(self)(
      self.terminal_available[ids],
      self.terminated[ids],
      self.truncated[ids],
      tuple(self.reason[int(index)] for index in ids),
      self.finite[ids],
      self.physical_fall[ids],
      self.tracking_rejection[ids],
      self.reset[ids],
      self.teleport[ids],
      self.reference_boundary[ids],
      self.clip_ended[ids],
    )


def parse_step_evidence_batch(value: Any) -> StepEvidenceBatch:
  """Decode a vectorized evidence object without inferring survival."""
  return StepEvidenceBatch.from_values(value)


@dataclass(frozen=True, slots=True)
class QualificationConfig:
  """Pinned D0 trial limits and physical/reference guard thresholds."""

  control_steps: int = 250
  retained_steps: int = 125
  period_seconds: float = DEFAULT_CONTRACT.period_seconds
  physical_fall_degrees: float = 70.0
  anchor_z_error_m: float = 0.25
  gravity_z_error: float = 0.8
  end_effector_z_error_m: float = 0.25

  def __post_init__(self) -> None:
    if self.control_steps <= 0 or self.retained_steps <= 0:
      raise QualificationError("qualification step limits must be positive")
    if self.retained_steps > self.control_steps:
      raise QualificationError("retained interval cannot exceed trial duration")


@dataclass(frozen=True, slots=True)
class QualificationResult:
  """Final verdict and audit counters for one attempted trial."""

  qualified: bool
  label: str
  reason: str | None
  steps: int
  vae_steps: int
  retained_vae_steps: int
  handoff_step: int | None


class QualificationTracker:
  """Evaluate every post-step outcome, with final-step failure precedence."""

  def __init__(self, config: QualificationConfig | None = None) -> None:
    self.config = config or QualificationConfig()
    self.steps = 0
    self.vae_steps = 0
    self.retained_vae_steps = 0
    self.handoff_step: int | None = None
    self._failure: str | None = None
    self._interrupted = False
    self._saw_handoff = False

  @property
  def failure(self) -> str | None:
    return self._failure

  def observe(
    self, controller: str, evidence: Any, observation: Any | None = None
  ) -> StepEvidence:
    """Record one post-step event; physical guards apply in both phases."""
    if controller not in {"vae", "ppo"}:
      raise QualificationError("controller must be vae or ppo")
    if self.steps >= self.config.control_steps:
      raise QualificationError("cannot observe after the trial budget")
    parsed = parse_step_evidence(evidence)
    if observation is None:
      parsed = StepEvidence(
        parsed.terminal_available,
        parsed.terminated,
        parsed.truncated,
        parsed.reason,
        parsed.finite,
        parsed.physical_fall or controller == "ppo",
        parsed.tracking_rejection or controller == "vae",
        parsed.reset,
        parsed.teleport,
        parsed.reference_boundary,
        parsed.clip_ended,
      )
    else:
      state_tilt = _state_tilt_degrees(observation)
      tilt_value = getattr(observation, "physical_tilt_degrees", None)
      tilt = state_tilt
      if tilt_value is not None:
        try:
          tilt = max(float(tilt_value), state_tilt)
        except (TypeError, ValueError):
          tilt = float("inf")
      if tilt > self.config.physical_fall_degrees:
        parsed = StepEvidence(
          parsed.terminal_available,
          parsed.terminated,
          parsed.truncated,
          parsed.reason,
          parsed.finite,
          True,
          parsed.tracking_rejection,
          parsed.reset,
          parsed.teleport,
          parsed.reference_boundary,
          parsed.clip_ended,
        )
      if controller == "vae":
        metric_names = ("anchor_z_error", "gravity_z_error", "end_effector_z_error")
        metrics = [getattr(observation, name, None) for name in metric_names]
        if any(value is None for value in metrics):
          parsed = StepEvidence(
            parsed.terminal_available,
            parsed.terminated,
            parsed.truncated,
            parsed.reason,
            parsed.finite,
            parsed.physical_fall,
            True,
            parsed.reset,
            parsed.teleport,
            parsed.reference_boundary,
            parsed.clip_ended,
          )
        elif (
          abs(float(cast(float, metrics[0]))) > self.config.anchor_z_error_m
          or abs(float(cast(float, metrics[1]))) > self.config.gravity_z_error
          or abs(float(cast(float, metrics[2]))) > self.config.end_effector_z_error_m
        ):
          parsed = StepEvidence(
            parsed.terminal_available,
            parsed.terminated,
            parsed.truncated,
            parsed.reason,
            parsed.finite,
            parsed.physical_fall,
            True,
            parsed.reset,
            parsed.teleport,
            parsed.reference_boundary,
            parsed.clip_ended,
          )
    self.steps += 1
    if controller == "vae":
      self.vae_steps += 1
      if self.vae_steps <= self.config.retained_steps:
        self.retained_vae_steps += 1
    if not parsed.terminal_available:
      self._fail("missing_terminal_evidence")
    elif not parsed.finite:
      self._fail("nonfinite_post_step")
    elif parsed.reset:
      self._fail("unexpected_reset")
    elif parsed.teleport:
      self._fail("reference_teleport")
    elif parsed.physical_fall:
      self._fail("physical_fall")
    elif controller == "vae" and parsed.tracking_rejection:
      self._fail("vae_tracking_rejection")
    if parsed.reference_boundary or parsed.clip_ended:
      if controller != "vae":
        self._fail("invalid_ppo_reference_boundary")
      elif not self._saw_handoff:
        self._saw_handoff = True
        self.handoff_step = self.steps
    if parsed.terminated or parsed.truncated:
      if not (controller == "vae" and (parsed.reference_boundary or parsed.clip_ended)):
        if self._failure is None:
          self._interrupted = True
    return parsed

  def finish(self) -> QualificationResult:
    """Close the trial; a failure on the last transition always wins."""
    if self._failure is not None:
      return QualificationResult(
        False,
        "failed",
        self._failure,
        self.steps,
        self.vae_steps,
        self.retained_vae_steps,
        self.handoff_step,
      )
    if self._interrupted or self.steps < self.config.control_steps:
      reason = "early_termination" if self._interrupted else "incomplete_timeout"
      return QualificationResult(
        False,
        "incomplete",
        reason,
        self.steps,
        self.vae_steps,
        self.retained_vae_steps,
        self.handoff_step,
      )
    label = "vae_then_standing" if self._saw_handoff else "vae_only"
    return QualificationResult(
      True,
      label,
      None,
      self.steps,
      self.vae_steps,
      self.retained_vae_steps,
      self.handoff_step,
    )

  def _fail(self, reason: str) -> None:
    if self._failure is None:
      self._failure = reason


class BatchedQualificationTracker:
  """Vectorized qualification state for independent trial slots."""

  _REASONS = (
    "",
    "missing_terminal_evidence",
    "nonfinite_post_step",
    "unexpected_reset",
    "reference_teleport",
    "physical_fall",
    "vae_tracking_rejection",
    "invalid_ppo_reference_boundary",
  )
  _REASON_CODES = {name: index for index, name in enumerate(_REASONS) if name}

  def __init__(
    self, batch_size: int, config: QualificationConfig | None = None
  ) -> None:
    if isinstance(batch_size, bool) or batch_size <= 0:
      raise QualificationError("batched qualification needs a positive batch size")
    self.config = config or QualificationConfig()
    self.batch_size = int(batch_size)
    self.steps = np.zeros(self.batch_size, dtype=np.int64)
    self.vae_steps = np.zeros(self.batch_size, dtype=np.int64)
    self.retained_vae_steps = np.zeros(self.batch_size, dtype=np.int64)
    self.handoff_step = np.full(self.batch_size, -1, dtype=np.int64)
    self.failure_codes = np.zeros(self.batch_size, dtype=np.int8)
    self.interrupted = np.zeros(self.batch_size, dtype=bool)
    self.saw_handoff = np.zeros(self.batch_size, dtype=bool)

  @property
  def failure(self) -> np.ndarray:
    return self.failure_codes.copy()

  @property
  def done(self) -> np.ndarray:
    return (
      (self.failure_codes != 0)
      | self.interrupted
      | (self.steps >= self.config.control_steps)
    )

  def reset(self, env_ids: Any) -> None:
    ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
    if ids.size == 0 or np.any(ids < 0) or np.any(ids >= self.batch_size):
      raise QualificationError("qualification reset indices are outside the batch")
    self.steps[ids] = 0
    self.vae_steps[ids] = 0
    self.retained_vae_steps[ids] = 0
    self.handoff_step[ids] = -1
    self.failure_codes[ids] = 0
    self.interrupted[ids] = False
    self.saw_handoff[ids] = False

  def _set_failure(self, ids: np.ndarray, reason: str) -> None:
    code = self._REASON_CODES[reason]
    unset = ids[self.failure_codes[ids] == 0]
    self.failure_codes[unset] = code

  @staticmethod
  def _tilt_degrees(observation: Any) -> np.ndarray:
    state = getattr(observation, "state", None)
    quaternion = getattr(state, "root_quaternion_wxyz", None)
    if quaternion is None:
      return np.full(observation.batch_size, np.inf, dtype=np.float64)
    value = np.asarray(quaternion, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 4:
      return np.full(observation.batch_size, np.inf, dtype=np.float64)
    norms = np.linalg.norm(value, axis=1)
    safe = np.divide(
      value, norms[:, None], out=np.zeros_like(value), where=norms[:, None] != 0
    )
    projected_gravity_z = 1.0 - 2.0 * (safe[:, 1] ** 2 + safe[:, 2] ** 2)
    return np.degrees(np.arccos(np.clip(projected_gravity_z, -1.0, 1.0)))

  def observe(
    self,
    env_ids: Any,
    controller_ppo: Any,
    evidence: Any,
    observation: Any | None = None,
  ) -> StepEvidenceBatch:
    """Record one post-step event for the supplied independent rows."""
    ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
    if ids.size == 0 or np.any(ids < 0) or np.any(ids >= self.batch_size):
      raise QualificationError(
        "qualification observation indices are outside the batch"
      )
    ppo = _batch_bool("controller_ppo", controller_ppo, ids.size)
    parsed = parse_step_evidence_batch(evidence)
    if parsed.batch_size != ids.size:
      raise QualificationError("evidence and qualification rows must align")
    terminal_available = parsed.terminal_available.copy()
    terminated = parsed.terminated.copy()
    truncated = parsed.truncated.copy()
    finite = parsed.finite.copy()
    physical_fall = parsed.physical_fall.copy()
    tracking_rejection = parsed.tracking_rejection.copy()
    reset = parsed.reset.copy()
    teleport = parsed.teleport.copy()
    reference_boundary = parsed.reference_boundary.copy()
    clip_ended = parsed.clip_ended.copy()
    reasons = list(parsed.reason)

    self.steps[ids] += 1
    self.vae_steps[ids] += ~ppo
    self.retained_vae_steps[ids] += (~ppo) & (
      self.vae_steps[ids] <= self.config.retained_steps
    )
    if observation is not None:
      tilt = self._tilt_degrees(observation)
      explicit_tilt = getattr(observation, "physical_tilt_degrees", None)
      if explicit_tilt is not None:
        tilt = np.maximum(tilt, np.asarray(explicit_tilt, dtype=np.float64))
      physical_fall |= tilt > self.config.physical_fall_degrees
      for name, limit in (
        ("anchor_z_error", self.config.anchor_z_error_m),
        ("gravity_z_error", self.config.gravity_z_error),
        ("end_effector_z_error", self.config.end_effector_z_error_m),
      ):
        values = getattr(observation, name, None)
        if values is None:
          tracking_rejection |= ~ppo
        else:
          tracking_rejection |= (~ppo) & (
            np.abs(np.asarray(values, dtype=np.float64)) > limit
          )
    else:
      physical_fall |= ppo
      tracking_rejection |= ~ppo

    selected = ids
    for mask, reason in (
      (~terminal_available, "missing_terminal_evidence"),
      (~finite, "nonfinite_post_step"),
      (reset, "unexpected_reset"),
      (teleport, "reference_teleport"),
      (physical_fall, "physical_fall"),
      ((~ppo) & tracking_rejection, "vae_tracking_rejection"),
      (ppo & (reference_boundary | clip_ended), "invalid_ppo_reference_boundary"),
    ):
      self._set_failure(selected[mask], reason)
    valid_boundary = (
      (~ppo) & (reference_boundary | clip_ended) & ~(terminated | truncated)
    )
    boundary_ids = selected[valid_boundary & ~self.saw_handoff[selected]]
    self.saw_handoff[boundary_ids] = True
    self.handoff_step[boundary_ids] = self.steps[boundary_ids]
    interrupted = (terminated | truncated) & ~valid_boundary
    # Apply the Boolean mask BEFORE selecting global integer row identities.
    # Bitwise-AND of an ID and a Boolean maps even IDs to 0 and odd IDs to 1.
    interrupted_ids = selected[interrupted & (self.failure_codes[selected] == 0)]
    self.interrupted[interrupted_ids] = True
    return StepEvidenceBatch(
      terminal_available,
      terminated,
      truncated,
      tuple(reasons),
      finite,
      physical_fall,
      tracking_rejection,
      reset,
      teleport,
      reference_boundary,
      clip_ended,
    )

  def finish(self, env_id: int) -> QualificationResult:
    if env_id < 0 or env_id >= self.batch_size:
      raise QualificationError("qualification result index is outside the batch")
    code = int(self.failure_codes[env_id])
    if code:
      return QualificationResult(
        False,
        "failed",
        self._REASONS[code],
        int(self.steps[env_id]),
        int(self.vae_steps[env_id]),
        int(self.retained_vae_steps[env_id]),
        None if self.handoff_step[env_id] < 0 else int(self.handoff_step[env_id]),
      )
    if self.interrupted[env_id] or self.steps[env_id] < self.config.control_steps:
      return QualificationResult(
        False,
        "incomplete",
        "early_termination" if self.interrupted[env_id] else "incomplete_timeout",
        int(self.steps[env_id]),
        int(self.vae_steps[env_id]),
        int(self.retained_vae_steps[env_id]),
        None if self.handoff_step[env_id] < 0 else int(self.handoff_step[env_id]),
      )
    return QualificationResult(
      True,
      "vae_then_standing" if self.saw_handoff[env_id] else "vae_only",
      None,
      int(self.steps[env_id]),
      int(self.vae_steps[env_id]),
      int(self.retained_vae_steps[env_id]),
      None if self.handoff_step[env_id] < 0 else int(self.handoff_step[env_id]),
    )


__all__ = [
  "BatchedQualificationTracker",
  "QualificationConfig",
  "QualificationError",
  "QualificationResult",
  "QualificationTracker",
  "StepEvidence",
  "StepEvidenceBatch",
  "parse_step_evidence",
  "parse_step_evidence_batch",
]
