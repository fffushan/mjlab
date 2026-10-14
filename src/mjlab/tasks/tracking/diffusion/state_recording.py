"""Record per-tick robot state from a parallel D3 diffusion rollout.

A recorded file is render-ready: the root pose and the 31 joint positions are
enough to rebuild the whole body pose with forward kinematics, which is what a
playback tool needs.  Root velocities, joint velocities, the applied action and
the measured tilt are kept beside them for analysis.  Every array has the shape
``[ticks, rows, width]``, so row ``r`` is one episode and tick ``t`` is one
control step.

Channel layout per row per tick:

===========================  =====
``root_position``            3
``root_quaternion_wxyz``     4
``root_linear_velocity``     3
``root_angular_velocity``    3
``joint_position``           31
``joint_velocity``           31
``action``                   31
``tilt_degrees``             1
===========================  =====

That is 107 values, 428 bytes in float32.  ``--record-bodies`` adds
``body_positions`` and ``body_linear_velocities`` (120 more values), which is
useful for a contact or end-effector check and as an independent check that the
recorded pose reconstructs the body.

The state recorded for a tick is the measured pre-action robot state: the state
the simulator held when the policy produced the action stored in the same slot.

The joint channels are the simulator's own joint position and velocity, not the
actor observation.  This task family feeds the policy a *relative* joint
position with added encoder bias and noise, delivered with up to one control
step of lag, so the observed values cannot rebuild the pose.  A recording that
is meant to be rendered must read the simulator.

This module writes no simulator object.  It owns an array buffer and an atomic
file write, so a CPU test can exercise it without MuJoCo.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from mjlab.tasks.tracking.diffusion.state import WorldState

STATE_FIELDS: tuple[str, ...] = (
  "root_position",
  "root_quaternion_wxyz",
  "root_linear_velocity",
  "root_angular_velocity",
)
JOINT_FIELDS: tuple[str, ...] = ("joint_position", "joint_velocity")
BODY_FIELDS: tuple[str, ...] = ("body_positions", "body_linear_velocities")
CHANNEL_NAMES: tuple[str, ...] = (
  "root_position",
  "root_quaternion_wxyz",
  "root_linear_velocity",
  "root_angular_velocity",
  "joint_position",
  "joint_velocity",
  "action",
  "tilt_degrees",
)
CHANNEL_WIDTHS: tuple[int, ...] = (3, 4, 3, 3, 31, 31, 31, 1)
CHANNEL_WIDTH = sum(CHANNEL_WIDTHS)
JOINT_WIDTH = 31
ACTION_WIDTH = 31
PHASE_VAE = 0
PHASE_CONTINUATION = 1


class StateRecordingError(ValueError):
  """A recording request, buffer or file is invalid."""


def _array(value: Any, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
  result = np.asarray(value, dtype=np.float64)
  if result.shape != shape:
    raise StateRecordingError(f"{name} must have shape {shape}, got {result.shape}")
  if not np.isfinite(result).all():
    raise StateRecordingError(f"{name} contains non-finite values")
  return result


class StateRecorder:
  """Append the pre-action robot state of every tick to one array file.

  The buffer is preallocated from the run geometry, so recording costs one copy
  per tick and no allocation.  ``write`` trims it to the ticks that happened and
  writes one ``.npz`` file atomically.  The target file must not exist: a
  recording is evidence, and a silent overwrite would destroy it.
  """

  def __init__(
    self,
    path: str | Path,
    *,
    rows: int,
    max_ticks: int,
    stride: int = 1,
    recorded_rows: int | None = None,
    body_channels: bool = False,
    period_seconds: float = 0.02,
    joint_names: Sequence[str] = (),
    body_names: Sequence[str] = (),
  ) -> None:
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
      raise StateRecordingError("rows must be a positive integer")
    if isinstance(max_ticks, bool) or not isinstance(max_ticks, int) or max_ticks < 1:
      raise StateRecordingError("max_ticks must be a positive integer")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
      raise StateRecordingError("stride must be a positive integer")
    if recorded_rows is None:
      recorded_rows = rows
    if (
      isinstance(recorded_rows, bool)
      or not isinstance(recorded_rows, int)
      or not 1 <= recorded_rows <= rows
    ):
      raise StateRecordingError(f"recorded_rows must be in [1, {rows}]")
    if not np.isfinite(float(period_seconds)) or float(period_seconds) <= 0.0:
      raise StateRecordingError("period_seconds must be finite and positive")
    self.path = Path(path)
    if self.path.exists():
      raise StateRecordingError(
        f"refusing to overwrite an existing recording: {self.path}"
      )
    self.rows = rows
    self.recorded_rows = recorded_rows
    self.max_ticks = max_ticks
    self.stride = stride
    self.body_channels = bool(body_channels)
    self.period_seconds = float(period_seconds)
    self.joint_names = tuple(str(name) for name in joint_names)
    self.body_names = tuple(str(name) for name in body_names)
    self._slots = (max_ticks + stride - 1) // stride
    shape = (self._slots, recorded_rows)
    self._channels: dict[str, np.ndarray] = {
      name: np.full(shape + (width,), np.nan, dtype=np.float32)
      for name, width in zip(CHANNEL_NAMES, CHANNEL_WIDTHS, strict=True)
    }
    if self.body_channels:
      self._channels["body_positions"] = np.full(
        shape + (3 * len(self.body_names),), np.nan, dtype=np.float32
      )
      self._channels["body_linear_velocities"] = np.full(
        shape + (3 * len(self.body_names),), np.nan, dtype=np.float32
      )
    self._ticks = np.full(self._slots, -1, dtype=np.int64)
    self._phase = np.zeros(self._slots, dtype=np.uint8)
    self._active = np.zeros(shape, dtype=bool)
    self._count = 0

  @property
  def recorded_ticks(self) -> int:
    return self._count

  def append(
    self,
    *,
    tick: int,
    state: WorldState,
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    action: np.ndarray,
    active: np.ndarray,
    phase: int,
    bodies: WorldState | None = None,
  ) -> None:
    """Record one tick.  Ticks that the stride skips are ignored."""
    if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
      raise StateRecordingError("tick must be a non-negative integer")
    if phase not in (PHASE_VAE, PHASE_CONTINUATION):
      raise StateRecordingError(f"phase must be {PHASE_VAE} or {PHASE_CONTINUATION}")
    if tick % self.stride != 0:
      return
    if self._count >= self._slots:
      raise StateRecordingError(
        f"recording buffer is full at {self._slots} slots; widen max_ticks"
      )
    if state.count != self.rows:
      raise StateRecordingError(
        f"state batch must hold {self.rows} rows, got {state.count}"
      )
    joint_position_values = _array(
      joint_position, name="joint_position", shape=(self.rows, JOINT_WIDTH)
    )
    joint_velocity_values = _array(
      joint_velocity, name="joint_velocity", shape=(self.rows, JOINT_WIDTH)
    )
    action_values = _array(action, name="action", shape=(self.rows, ACTION_WIDTH))
    active_values = np.asarray(active, dtype=bool).reshape(-1)
    if active_values.shape != (self.rows,):
      raise StateRecordingError(
        f"active mask must hold {self.rows} flags, got {active_values.shape}"
      )
    if self.body_channels:
      if bodies is None:
        raise StateRecordingError("body channels requested but no body state supplied")
      if bodies.count != self.rows:
        raise StateRecordingError(
          f"body state must hold {self.rows} rows, got {bodies.count}"
        )
    elif bodies is not None:
      raise StateRecordingError("body state supplied while body channels are off")

    slot = self._count
    kept = slice(0, self.recorded_rows)
    values = {
      "root_position": state.root_position,
      "root_quaternion_wxyz": state.root_quaternion_wxyz,
      "root_linear_velocity": state.root_linear_velocity,
      "root_angular_velocity": state.root_angular_velocity,
      "joint_position": joint_position_values,
      "joint_velocity": joint_velocity_values,
      "action": action_values,
      "tilt_degrees": self._tilt(state)[:, None],
    }
    if self.body_channels and bodies is not None:
      values["body_positions"] = np.asarray(bodies.body_positions).reshape(self.rows, -1)
      values["body_linear_velocities"] = np.asarray(
        bodies.body_linear_velocities
      ).reshape(self.rows, -1)
    for name, value in values.items():
      array = np.asarray(value, dtype=np.float64)
      width = self._channels[name].shape[-1]
      if array.shape != (self.rows, width):
        raise StateRecordingError(
          f"{name} must have shape ({self.rows}, {width}), got {array.shape}"
        )
      if not np.isfinite(array).all():
        raise StateRecordingError(f"{name} contains non-finite values")
      self._channels[name][slot] = array[kept].astype(np.float32)
    self._ticks[slot] = int(tick)
    self._phase[slot] = int(phase)
    self._active[slot] = active_values[kept]
    self._count += 1

  @staticmethod
  def _tilt(state: WorldState) -> np.ndarray:
    quaternion = np.asarray(state.root_quaternion_wxyz, dtype=np.float64)
    norms = np.linalg.norm(quaternion, axis=1)
    result = np.full(norms.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(norms) & (norms >= 1.0e-12)
    if bool(valid.any()):
      normalized = quaternion[valid] / norms[valid][:, None]
      up_z = 1.0 - 2.0 * (
        normalized[:, 1] * normalized[:, 1] + normalized[:, 2] * normalized[:, 2]
      )
      result[valid] = np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0)))
    return result

  def write(self, metadata: Mapping[str, Any] | None = None) -> Path:
    """Write the recorded ticks to the target path, or raise if nothing is there."""
    if self._count == 0:
      raise StateRecordingError("no tick was recorded; nothing to write")
    payload: dict[str, Any] = {
      name: array[: self._count] for name, array in self._channels.items()
    }
    payload["tick"] = self._ticks[: self._count]
    payload["phase"] = self._phase[: self._count]
    payload["active"] = self._active[: self._count]
    payload["channel_names"] = np.array(CHANNEL_NAMES)
    payload["channel_widths"] = np.array(CHANNEL_WIDTHS, dtype=np.int64)
    payload["rows"] = np.int64(self.rows)
    payload["recorded_rows"] = np.int64(self.recorded_rows)
    payload["stride"] = np.int64(self.stride)
    payload["period_seconds"] = np.float64(self.period_seconds)
    payload["joint_names"] = np.array(self.joint_names)
    payload["body_names"] = np.array(self.body_names)
    payload["channel_width"] = np.int64(CHANNEL_WIDTH)
    payload["body_channels"] = np.int64(1 if self.body_channels else 0)
    for key, value in (metadata or {}).items():
      if key in payload:
        raise StateRecordingError(f"metadata key {key!r} would overwrite an array")
      if isinstance(value, (list, tuple)):
        payload[key] = np.array([str(item) for item in value])
      elif isinstance(value, (np.ndarray, np.generic)):
        payload[key] = value
      elif isinstance(value, bool):
        payload[key] = np.int64(1 if value else 0)
      elif isinstance(value, int):
        payload[key] = np.int64(value)
      elif isinstance(value, float):
        payload[key] = np.float64(value)
      elif isinstance(value, str):
        payload[key] = np.array(value)
      else:
        raise StateRecordingError(f"metadata value for {key!r} is unsupported")
    self.path.parent.mkdir(parents=True, exist_ok=True)
    temporary = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
      np.savez(handle, **payload)
    temporary.replace(self.path)
    return self.path


__all__ = [
  "BODY_FIELDS",
  "CHANNEL_NAMES",
  "CHANNEL_WIDTH",
  "CHANNEL_WIDTHS",
  "JOINT_FIELDS",
  "PHASE_CONTINUATION",
  "PHASE_VAE",
  "STATE_FIELDS",
  "StateRecorder",
  "StateRecordingError",
]
