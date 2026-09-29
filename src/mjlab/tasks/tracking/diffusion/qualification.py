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


__all__ = [
  "QualificationConfig",
  "QualificationError",
  "QualificationResult",
  "QualificationTracker",
  "StepEvidence",
  "parse_step_evidence",
]
