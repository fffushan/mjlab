"""Fail-closed qualification rules for frozen-policy D1 trials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

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


MAX_ROLLOUT_STEPS = 512
"""Hard bound on one rollout, in control steps.

A rollout ends when its reference reaches the last clip frame, so this bound
only closes a run whose environment never reports that end.  The tracking task's
own 500-step time-out normally fires first.
"""


@dataclass(frozen=True, slots=True)
class QualificationConfig:
  """Pinned D1 padding width and endpoint stability limits."""

  padding_steps: int = DEFAULT_CONTRACT.padding_steps
  period_seconds: float = DEFAULT_CONTRACT.period_seconds
  physical_fall_degrees: float = 70.0
  anchor_z_error_m: float = 0.25
  gravity_z_error: float = 0.8
  end_effector_z_error_m: float = 0.25

  def __post_init__(self) -> None:
    if self.padding_steps < 0:
      raise QualificationError("padding_steps must be non-negative")
    if self.period_seconds <= 0:
      raise QualificationError("the control period must be positive")


@dataclass(frozen=True, slots=True)
class EndpointMetrics:
  """The four stability measurements of one row."""

  tilt_degrees: float
  anchor_z_error: float
  gravity_z_error: float
  end_effector_z_error: float

  def __post_init__(self) -> None:
    if not np.isfinite(
      (
        self.tilt_degrees,
        self.anchor_z_error,
        self.gravity_z_error,
        self.end_effector_z_error,
      )
    ).all():
      raise QualificationError("endpoint metrics must be finite")


def row_metrics(observation: Any | None) -> EndpointMetrics | None:
  """Read one row's stability metrics, or ``None`` when the row has no metrics.

  A missing observation or a missing metric is reported as ``None`` so the
  endpoint test can refuse the rollout instead of guessing that the robot stood
  still.
  """
  if observation is None:
    return None
  tilt = _state_tilt_degrees(observation)
  explicit = getattr(observation, "physical_tilt_degrees", None)
  if explicit is not None:
    try:
      tilt = max(tilt, float(explicit))
    except (TypeError, ValueError):
      return None
  anchor = getattr(observation, "anchor_z_error", None)
  gravity = getattr(observation, "gravity_z_error", None)
  end_effector = getattr(observation, "end_effector_z_error", None)
  if anchor is None or gravity is None or end_effector is None:
    return None
  values = (float(tilt), float(anchor), float(gravity), float(end_effector))
  if not np.isfinite(values).all():
    return None
  return EndpointMetrics(*values)


def endpoint_rejection(
  metrics: EndpointMetrics, config: QualificationConfig
) -> str | None:
  """Return the first stability check the endpoint row fails, if any."""
  if metrics.tilt_degrees > config.physical_fall_degrees:
    return "endpoint_tilt"
  if abs(metrics.anchor_z_error) > config.anchor_z_error_m:
    return "endpoint_anchor_z_error"
  if abs(metrics.gravity_z_error) > config.gravity_z_error:
    return "endpoint_gravity_z_error"
  if abs(metrics.end_effector_z_error) > config.end_effector_z_error_m:
    return "endpoint_end_effector_z_error"
  return None


@dataclass(frozen=True, slots=True)
class QualificationResult:
  """Endpoint verdict and per-rollout summary for one attempted rollout."""

  qualified: bool
  label: str
  reason: str | None
  steps: int
  min_tilt_degrees: float
  max_anchor_z_error_m: float
  max_end_effector_z_error_m: float
  fall_frames: int


class QualificationTracker:
  """Integrity checks on every row, stability checks on the last row only."""

  def __init__(self, config: QualificationConfig | None = None) -> None:
    self.config = config or QualificationConfig()
    self.steps = 0
    self.min_tilt_degrees = float("inf")
    self.max_anchor_z_error_m = 0.0
    self.max_end_effector_z_error_m = 0.0
    self.fall_frames = 0
    self._failure: str | None = None
    self._interrupted = False
    self._endpoint = False
    self._last: EndpointMetrics | None = None

  @property
  def failure(self) -> str | None:
    return self._failure

  @property
  def done(self) -> bool:
    """True when no further row may extend this rollout."""
    return (
      self._failure is not None
      or self._interrupted
      or self._endpoint
      or self.steps >= MAX_ROLLOUT_STEPS
    )

  def observe(self, evidence: Any, observation: Any | None = None) -> StepEvidence:
    """Record one post-step row; only integrity failures end a rollout here."""
    if self.steps >= MAX_ROLLOUT_STEPS:
      raise QualificationError("cannot observe past the rollout step bound")
    parsed = parse_step_evidence(evidence)
    self.steps += 1
    metrics = row_metrics(observation)
    self._last = metrics
    if metrics is not None:
      self.min_tilt_degrees = min(self.min_tilt_degrees, metrics.tilt_degrees)
      self.max_anchor_z_error_m = max(
        self.max_anchor_z_error_m, abs(metrics.anchor_z_error)
      )
      self.max_end_effector_z_error_m = max(
        self.max_end_effector_z_error_m, abs(metrics.end_effector_z_error)
      )
      if metrics.tilt_degrees > self.config.physical_fall_degrees:
        self.fall_frames += 1
    if not parsed.terminal_available:
      self._fail("missing_terminal_evidence")
    elif not parsed.finite:
      self._fail("nonfinite_post_step")
    elif parsed.reset:
      self._fail("unexpected_reset")
    elif parsed.teleport:
      self._fail("reference_teleport")
    if parsed.reference_boundary or parsed.clip_ended:
      self._endpoint = True
    elif parsed.terminated or parsed.truncated:
      if self._failure is None:
        self._interrupted = True
    return parsed

  def finish(self) -> QualificationResult:
    """Close the rollout and apply the endpoint test to its last row."""
    if self._failure is not None:
      return self._result(False, "failed", self._failure)
    if self._interrupted:
      return self._result(False, "incomplete", "early_termination")
    if not self._endpoint:
      reason = (
        "rollout_step_cap"
        if self.steps >= MAX_ROLLOUT_STEPS
        else "missing_endpoint_evidence"
      )
      return self._result(False, "incomplete", reason)
    if self._last is None:
      return self._result(False, "incomplete", "missing_stability_metrics")
    rejection = endpoint_rejection(self._last, self.config)
    if rejection is not None:
      return self._result(False, "failed", rejection)
    return self._result(True, "endpoint_pass", None)

  def _result(
    self, qualified: bool, label: str, reason: str | None
  ) -> QualificationResult:
    return QualificationResult(
      qualified,
      label,
      reason,
      self.steps,
      self.min_tilt_degrees,
      self.max_anchor_z_error_m,
      self.max_end_effector_z_error_m,
      self.fall_frames,
    )

  def _fail(self, reason: str) -> None:
    if self._failure is None:
      self._failure = reason


class BatchedQualificationTracker:
  """Vectorized endpoint qualification state for independent rollout slots."""

  _REASONS = (
    "",
    "missing_terminal_evidence",
    "nonfinite_post_step",
    "unexpected_reset",
    "reference_teleport",
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
    self.failure_codes = np.zeros(self.batch_size, dtype=np.int8)
    self.interrupted = np.zeros(self.batch_size, dtype=bool)
    self.endpoint = np.zeros(self.batch_size, dtype=bool)
    self.min_tilt_degrees = np.full(self.batch_size, np.inf, dtype=np.float64)
    self.max_anchor_z_error_m = np.zeros(self.batch_size, dtype=np.float64)
    self.max_end_effector_z_error_m = np.zeros(self.batch_size, dtype=np.float64)
    self.fall_frames = np.zeros(self.batch_size, dtype=np.int64)
    self.last_metrics = np.full((self.batch_size, 4), np.nan, dtype=np.float64)

  @property
  def failure(self) -> np.ndarray:
    return self.failure_codes.copy()

  @property
  def done(self) -> np.ndarray:
    return (
      (self.failure_codes != 0)
      | self.interrupted
      | self.endpoint
      | (self.steps >= MAX_ROLLOUT_STEPS)
    )

  def reset(self, env_ids: Any) -> None:
    ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
    if ids.size == 0 or np.any(ids < 0) or np.any(ids >= self.batch_size):
      raise QualificationError("qualification reset indices are outside the batch")
    self.steps[ids] = 0
    self.failure_codes[ids] = 0
    self.interrupted[ids] = False
    self.endpoint[ids] = False
    self.min_tilt_degrees[ids] = np.inf
    self.max_anchor_z_error_m[ids] = 0.0
    self.max_end_effector_z_error_m[ids] = 0.0
    self.fall_frames[ids] = 0
    self.last_metrics[ids] = np.nan

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

  def _batch_metrics(self, observation: Any, rows: int) -> np.ndarray:
    """Return one ``[rows, 4]`` metric block; non-finite entries mark a gap."""
    tilt = self._tilt_degrees(observation)
    explicit = getattr(observation, "physical_tilt_degrees", None)
    if explicit is not None:
      tilt = np.maximum(tilt, np.asarray(explicit, dtype=np.float64))
    columns = [tilt]
    for name in ("anchor_z_error", "gravity_z_error", "end_effector_z_error"):
      values = getattr(observation, name, None)
      if values is None:
        return np.full((rows, 4), np.nan, dtype=np.float64)
      columns.append(np.asarray(values, dtype=np.float64).reshape(-1))
    metrics = np.stack(columns, axis=1)
    if metrics.shape != (rows, 4):
      return np.full((rows, 4), np.nan, dtype=np.float64)
    metrics[~np.isfinite(metrics)] = np.nan
    return metrics

  def observe(
    self,
    env_ids: Any,
    evidence: Any,
    observation: Any | None = None,
  ) -> StepEvidenceBatch:
    """Record one post-step row for every supplied independent slot."""
    ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
    if ids.size == 0 or np.any(ids < 0) or np.any(ids >= self.batch_size):
      raise QualificationError(
        "qualification observation indices are outside the batch"
      )
    parsed = parse_step_evidence_batch(evidence)
    if parsed.batch_size != ids.size:
      raise QualificationError("evidence and qualification rows must align")
    terminal_available = parsed.terminal_available.copy()
    finite = parsed.finite.copy()
    reset = parsed.reset.copy()
    teleport = parsed.teleport.copy()
    reference_boundary = parsed.reference_boundary.copy()
    clip_ended = parsed.clip_ended.copy()
    terminated = parsed.terminated.copy()
    truncated = parsed.truncated.copy()
    reasons = list(parsed.reason)

    self.steps[ids] += 1
    metrics = (
      np.full((ids.size, 4), np.nan, dtype=np.float64)
      if observation is None
      else self._batch_metrics(observation, ids.size)
    )
    self.last_metrics[ids] = metrics
    valid = np.isfinite(metrics[:, 0])
    if valid.any():
      rows = ids[valid]
      block = metrics[valid]
      self.min_tilt_degrees[rows] = np.minimum(self.min_tilt_degrees[rows], block[:, 0])
      self.max_anchor_z_error_m[rows] = np.maximum(
        self.max_anchor_z_error_m[rows], np.abs(block[:, 1])
      )
      self.max_end_effector_z_error_m[rows] = np.maximum(
        self.max_end_effector_z_error_m[rows], np.abs(block[:, 3])
      )
      self.fall_frames[rows] += (
        block[:, 0] > self.config.physical_fall_degrees
      ).astype(np.int64)

    for mask, reason in (
      (~terminal_available, "missing_terminal_evidence"),
      (~finite, "nonfinite_post_step"),
      (reset, "unexpected_reset"),
      (teleport, "reference_teleport"),
    ):
      if mask.any():
        self._set_failure(ids[mask], reason)
    boundary = reference_boundary | clip_ended
    self.endpoint[ids[boundary]] = True
    interrupted = (terminated | truncated) & ~boundary
    interrupted_ids = ids[interrupted & (self.failure_codes[ids] == 0)]
    self.interrupted[interrupted_ids] = True
    return StepEvidenceBatch(
      terminal_available,
      terminated,
      truncated,
      tuple(reasons),
      finite,
      parsed.physical_fall,
      parsed.tracking_rejection,
      reset,
      teleport,
      reference_boundary,
      clip_ended,
    )

  def finish(self, env_id: int) -> QualificationResult:
    if env_id < 0 or env_id >= self.batch_size:
      raise QualificationError("qualification result index is outside the batch")

    def result(qualified: bool, label: str, reason: str | None) -> QualificationResult:
      return QualificationResult(
        qualified,
        label,
        reason,
        int(self.steps[env_id]),
        float(self.min_tilt_degrees[env_id]),
        float(self.max_anchor_z_error_m[env_id]),
        float(self.max_end_effector_z_error_m[env_id]),
        int(self.fall_frames[env_id]),
      )

    code = int(self.failure_codes[env_id])
    if code:
      return result(False, "failed", self._REASONS[code])
    if self.interrupted[env_id]:
      return result(False, "incomplete", "early_termination")
    if not self.endpoint[env_id]:
      return result(
        False,
        "incomplete",
        "rollout_step_cap"
        if self.steps[env_id] >= MAX_ROLLOUT_STEPS
        else "missing_endpoint_evidence",
      )
    block = self.last_metrics[env_id]
    if not np.isfinite(block).all():
      return result(False, "incomplete", "missing_stability_metrics")
    rejection = endpoint_rejection(
      EndpointMetrics(*(float(value) for value in block)), self.config
    )
    if rejection is not None:
      return result(False, "failed", rejection)
    return result(True, "endpoint_pass", None)


__all__ = [
  "BatchedQualificationTracker",
  "EndpointMetrics",
  "MAX_ROLLOUT_STEPS",
  "QualificationConfig",
  "QualificationError",
  "QualificationResult",
  "QualificationTracker",
  "StepEvidence",
  "StepEvidenceBatch",
  "endpoint_rejection",
  "parse_step_evidence",
  "parse_step_evidence_batch",
  "row_metrics",
]
