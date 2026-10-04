"""Owned measurements and protocol for bounded D3 mjlab evaluation.

The evaluation runner is simulator-independent. Only its concrete runtime may
construct mjlab environments; reference is present during VAE control only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np

from .state import WorldState

_STATE_FIELDS = (
  "root_position",
  "root_quaternion_wxyz",
  "root_linear_velocity",
  "root_angular_velocity",
  "body_positions",
  "body_linear_velocities",
)


@dataclass(frozen=True, slots=True)
class D3Observation:
  """An owned, single-tick measurement; no reference in diffusion mode."""

  state: WorldState
  conditioning: np.ndarray
  tick: int
  sim_time_s: float
  segment_id: str
  reference: np.ndarray | None = None
  reference_frame: int | None = None

  def __post_init__(self) -> None:
    if isinstance(self.tick, (bool, np.bool_)) or not isinstance(
      self.tick, (int, np.integer)
    ):
      raise ValueError("tick must be an integer, not boolean")
    if self.tick < 0 or not np.isfinite(self.sim_time_s) or self.sim_time_s < 0:
      raise ValueError("tick and simulation time must be finite and nonnegative")
    if not isinstance(self.segment_id, str) or not self.segment_id:
      raise ValueError("segment_id must be a nonempty string")
    if self.state.root_position.ndim != 1:
      raise ValueError("state must be one unbatched WorldState")
    object.__setattr__(
      self,
      "state",
      WorldState(
        **{
          name: np.array(getattr(self.state, name), dtype=np.float64, copy=True)
          for name in _STATE_FIELDS
        }
      ),
    )
    for name, width in (("conditioning", 99), ("reference", 68)):
      value = getattr(self, name)
      if value is None and name == "reference":
        continue
      array = np.array(value, dtype=np.float64, copy=True)
      if array.shape != (width,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape [{width}]")
      object.__setattr__(self, name, array)
    if self.reference_frame is not None and (
      isinstance(self.reference_frame, (bool, np.bool_))
      or not isinstance(self.reference_frame, (int, np.integer))
      or self.reference_frame < 0
    ):
      raise ValueError("reference_frame must be a nonnegative integer or None")


@dataclass(frozen=True, slots=True)
class D3Transition:
  """Post-step physical evidence, captured before any possible reset."""

  observation: D3Observation
  terminal_available: bool = True
  terminated: bool = False
  truncated: bool = False
  reset: bool = False
  teleport: bool = False
  reference_boundary: bool = False
  reason: str | None = None


class ClosedLoopRuntime(Protocol):
  """One-env simulator boundary; baseline and diffusion use fresh resets."""

  period_seconds: float
  clip_frames: int
  joint_order: tuple[str, ...]
  body_order: tuple[str, ...]
  provenance: Mapping[str, Any]

  def reset(self, *, seed: int, start_frame: int) -> D3Observation: ...

  def initial_signature(self) -> np.ndarray:
    """Owned root pose, joint position and joint velocity matching evidence."""
    ...

  def enter_reference_free(self) -> D3Observation:
    """Disable reference control/resampling without advancing physical time."""
    ...

  def step(self, action: np.ndarray) -> D3Transition: ...

  def close(self) -> None: ...
