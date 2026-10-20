"""Owned measurements and protocol for bounded D3 mjlab evaluation.

The evaluation runner is simulator-independent. Only its concrete runtime may
construct mjlab environments; reference is present during VAE control only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np

from mjlab.tasks.tracking.distillation.vae_config import (
  REFERENCE_DIM,
  REFERENCE_DIM_WITH_PREVIEW,
)

from .state import WorldState

REFERENCE_WIDTHS = (REFERENCE_DIM, REFERENCE_DIM_WITH_PREVIEW)
"""The supported gravity reference widths: version 1 and version 2 preview."""

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
    conditioning = np.array(self.conditioning, dtype=np.float64, copy=True)
    if conditioning.shape != (99,) or not np.isfinite(conditioning).all():
      raise ValueError("conditioning must be finite with shape [99]")
    object.__setattr__(self, "conditioning", conditioning)
    if self.reference is not None:
      reference = np.array(self.reference, dtype=np.float64, copy=True)
      if reference.shape not in {(width,) for width in REFERENCE_WIDTHS}:
        raise ValueError(
          f"reference must have shape [68] or [130], got {reference.shape}"
        )
      if not np.isfinite(reference).all():
        raise ValueError("reference must contain finite values")
      object.__setattr__(self, "reference", reference)
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


def reference_width_from_model(model: Any) -> int:
  """Return the loaded VAE reference width, defaulting to the version-1 layout.

  ``LoadedVae`` carries ``reference_dim``; the loaded model carries its saved
  ``schema``.  Only the two supported gravity layouts are accepted here, so a
  runtime measurement cannot widen the encoder input by accident.
  """
  schema = getattr(model, "schema", None)
  width = getattr(schema, "reference_dim", None)
  if width is None:
    width = getattr(model, "reference_dim", None)
  if width is None:
    return REFERENCE_DIM
  if isinstance(width, (bool, np.bool_)) or not isinstance(width, (int, np.integer)):
    raise ValueError(f"loaded VAE reference width must be an integer, got {width!r}")
  if int(width) not in REFERENCE_WIDTHS:
    raise ValueError(
      "loaded VAE reference width must be 68 (version 1) or 130 "
      f"(version 2 preview), got {int(width)}"
    )
  return int(width)


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
