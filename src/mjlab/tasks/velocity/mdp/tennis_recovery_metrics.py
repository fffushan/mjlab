"""Pure recovery/retention trace metrics for X2 tennis-end evaluation.

This module computes reusable per-trajectory recovery/retention statistics from
*state/command/termination traces*. It has no dependency on the endpoint-pool
loader, the runtime factory or the resume runner: integration wires real
simulator/runner traces into :class:`RecoveryTrace` / :class:`RetentionTrace`
and calls :func:`compute_recovery_metrics` / :func:`aggregate_recovery_metrics`.

Recovery vs retention
---------------------
Recovery traces (zero command) are evaluated with the sustained-settling test:
the robot must reach and hold a calm state. A terminal failure (fall,
orientation termination, invalid data) **invalidates** any prior calm window —
the episode cannot count as successful recovery even if it was briefly calm
before falling. The earliest calm-window timing is preserved as a separate
diagnostic (``earliest_calm_window_s``), not as success.

Retention traces (original nonzero command distribution) are evaluated with
body-frame command-tracking error, **not** zero-command settling. Intended
walking is never reclassified as a recovery failure. Missing retention command
data is explicitly unavailable (``None``), never silently treated as a zero
command.

Settling and sample gaps
------------------------
Settling requires a *sustained* window where all criteria hold continuously for
``sustained_s`` seconds. Consecutive samples within the window must not be
separated by a gap exceeding ``max_sample_gap_s`` (default: ``sustained_s``).
Two calm samples ten seconds apart do **not** establish a sustained 0.5-second
calm interval — the trace is censored (insufficient evidence). The real
simulator caller knows ``step_dt`` and may pass an explicit ``max_sample_gap_s``.

No favorable aggregate is produced from failed/censored episodes:
``settle_rate`` counts only actually-settled recovery trajectories, and
time-to-settle aggregates are over settled episodes only. Unavailable aggregate
values are represented as ``None``, never ``NaN``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = [
  "SettlingThresholds",
  "TerminationCategory",
  "RecoveryTrace",
  "RetentionTrace",
  "RecoveryMetrics",
  "RecoveryGroupStats",
  "torso_tilt_from_quat",
  "signed_backward_displacement",
  "total_planar_travel",
  "peak_tilt",
  "first_window_index",
  "settle_time",
  "compute_recovery_metrics",
  "aggregate_recovery_metrics",
]


# --------------------------------------------------------------------------- #
# Termination categories.
# --------------------------------------------------------------------------- #

TerminationCategory = Literal[
  "fall",  # true fall (e.g. illegal contact / height below minimum)
  "orientation",  # bad-orientation termination (exceeded tilt limit)
  "timeout",  # episode reached the horizon without a fall
  "invalid",  # non-finite / malformed trace data
  "not_settled",  # ran to horizon, did not fall, but never settled
]

_VALID_TERMINATIONS: frozenset[str] = frozenset(
  {"fall", "orientation", "timeout", "invalid", "not_settled"}
)

# Terminal failures: the episode ended badly. These invalidate any prior calm
# window — the episode cannot count as successful recovery.
_TERMINAL_FAILURE_CATEGORIES: frozenset[str] = frozenset(
  {"fall", "orientation", "invalid"}
)

_VALID_GROUPS: frozenset[str] = frozenset({"recovery", "retention"})


# --------------------------------------------------------------------------- #
# Configuration.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SettlingThresholds:
  """Thresholds for the sustained-settling test.

  Defaults match the plan: planar speed < 0.1 m/s, ``|yaw rate|`` < 0.2 rad/s,
  torso tilt < 10 deg, sustained for 0.5 s.

  ``max_sample_gap_s`` bounds the allowable inter-sample gap within a sustained
  window. When ``None`` it defaults to ``sustained_s``: two calm samples ten
  seconds apart cannot establish a 0.5-second sustained calm interval. The real
  simulator caller knows ``step_dt`` and may pass an explicit tighter bound
  (e.g. ``2 * step_dt``).
  """

  speed_xy: float = 0.1  # m/s
  yaw_rate: float = 0.2  # rad/s (absolute)
  torso_tilt_deg: float = 10.0  # degrees
  sustained_s: float = 0.5  # seconds the criteria must hold continuously
  max_sample_gap_s: float | None = None

  def __post_init__(self) -> None:
    for name, val in [
      ("speed_xy", self.speed_xy),
      ("yaw_rate", self.yaw_rate),
      ("torso_tilt_deg", self.torso_tilt_deg),
      ("sustained_s", self.sustained_s),
    ]:
      if (
        not isinstance(val, (int, float, np.floating))
        or not np.isfinite(val)
        or val <= 0
      ):
        raise ValueError(f"{name} must be a positive finite number, got {val!r}")
    if self.max_sample_gap_s is not None:
      if (
        not isinstance(self.max_sample_gap_s, (int, float, np.floating))
        or not np.isfinite(self.max_sample_gap_s)
        or self.max_sample_gap_s <= 0
      ):
        raise ValueError(
          f"max_sample_gap_s must be positive finite or None, got {self.max_sample_gap_s!r}"
        )


# --------------------------------------------------------------------------- #
# Trace inputs.
# --------------------------------------------------------------------------- #


@dataclass
class RecoveryTrace:
  """A single recovery (zero-command) trajectory trace.

  All arrays are 2-D ``[T, D]`` except 1-D scalars; ``T`` is the number of
  post-reset frames. ``timestamps`` are seconds since the trajectory's
  reset/time origin and must be strictly increasing.
  """

  timestamps: np.ndarray  # [T], seconds since reset origin
  root_pos_w: np.ndarray  # [T,3], m
  root_quat_w: np.ndarray  # [T,4], unit wxyz
  root_lin_vel_w: np.ndarray  # [T,3], m/s
  root_ang_vel_w: np.ndarray  # [T,3], rad/s
  torso_tilt_deg: np.ndarray  # [T], degrees from upright
  entry_yaw: float  # rad, the reset/entry yaw
  trajectory_id: int
  group: str  # "recovery" | "retention"
  termination: TerminationCategory
  # Requested vs applied torque diagnostics, only when explicitly available.
  applied_torque: np.ndarray | None = None
  requested_torque: np.ndarray | None = None


@dataclass
class RetentionTrace:
  """A single retention (original-command) trajectory trace.

  Same state fields as :class:`RecoveryTrace`; the ``group`` is always
  ``"retention"``. Retention retains the original command distribution, so the
  trace carries the commanded body-frame velocity ``vel_command_b`` ``[T,3]``
  (lin_x, lin_y, ang_z) for command-tracking diagnostics. When ``vel_command_b``
  is ``None`` the tracking diagnostics are explicitly unavailable (``None``),
  never silently treated as a zero command.
  """

  timestamps: np.ndarray
  root_pos_w: np.ndarray
  root_quat_w: np.ndarray
  root_lin_vel_w: np.ndarray
  root_ang_vel_w: np.ndarray
  torso_tilt_deg: np.ndarray
  entry_yaw: float
  trajectory_id: int
  termination: TerminationCategory
  vel_command_b: np.ndarray | None = None  # [T,3], commanded body-frame vel
  applied_torque: np.ndarray | None = None
  requested_torque: np.ndarray | None = None
  group: str = "retention"


# --------------------------------------------------------------------------- #
# Per-trajectory metrics.
# --------------------------------------------------------------------------- #


@dataclass
class RecoveryMetrics:
  """Per-trajectory recovery/retention statistics."""

  trajectory_id: int
  group: str
  termination: TerminationCategory
  settled: bool
  time_to_settle: float | None  # seconds; None when not settled / censored
  earliest_calm_window_s: float | None  # raw diagnostic; can be non-None for falls
  censored: bool  # True when settling could not be established
  not_settled: bool  # True when horizon was sufficient but no sustained window
  peak_tilt_deg: float
  signed_backward_displacement_m: float  # +ve == backward along entry yaw
  total_planar_travel_m: float
  # First-``early_window_s`` statistics (only frames within the early window).
  early_peak_tilt_deg: float
  early_signed_backward_displacement_m: float
  early_total_planar_travel_m: float
  early_window_s: float  # actual span of timestamps in the early window
  # Full-episode statistics.
  duration_s: float
  num_frames: int
  # Torque diagnostics: None when not explicitly available/labeled.
  torque_requested_vs_applied_ratio: float | None
  # Retention command-tracking diagnostics: None for recovery or when
  # vel_command_b is unavailable.
  mean_command_lin_error_mps: float | None
  mean_command_yaw_error_radps: float | None
  peak_command_lin_error_mps: float | None
  # Coverage: did this trajectory contribute a settled outcome?
  contributes_settled: bool


@dataclass
class RecoveryGroupStats:
  """Aggregate statistics for a group of trajectories.

  Unavailable aggregate values are ``None``, never ``NaN``, so strict JSON
  serialization is safe. ``num_command_tracked`` is the count of retention
  episodes with command-tracking data.
  """

  group: str
  num_trajectories: int
  num_settled: int
  num_falls: int
  num_orientation_terminations: int
  num_timeouts: int
  num_invalid: int
  num_not_settled: int
  num_censored: int
  settle_rate: float  # settled / num_trajectories
  # Aggregates over *settled* recovery episodes only.
  median_time_to_settle_s: float | None
  mean_time_to_settle_s: float | None
  # Means over non-invalid trajectories; None when no valid data.
  mean_peak_tilt_deg: float | None
  mean_signed_backward_displacement_m: float | None
  mean_total_planar_travel_m: float | None
  coverage: float  # fraction of trajectories with usable (non-invalid) data
  # Retention command-tracking aggregates; None for recovery groups.
  mean_command_lin_error_mps: float | None
  mean_command_yaw_error_radps: float | None
  num_command_tracked: int


# --------------------------------------------------------------------------- #
# Geometry helpers (pure NumPy).
# --------------------------------------------------------------------------- #


def torso_tilt_from_quat(quat_wxyz: np.ndarray) -> np.ndarray:
  """Torso tilt from upright in degrees for ``wxyz`` quaternions ``[N,4]``.

  Uses the same projected-gravity convention as ``bad_orientation``: the tilt
  angle is ``acos(-g_b_z)`` where ``g_b`` is gravity expressed in the body
  frame. For an upright torso ``g_b = (0,0,-1)`` so ``-g_b_z = 1`` and the
  tilt is 0 deg.
  """
  q = np.asarray(quat_wxyz, dtype=np.float64)
  if q.ndim == 1:
    q = q[None, :]
  x, y = q[:, 1], q[:, 2]
  # Gravity in body frame: R^T @ (0,0,-1); the body-frame z-component is
  # -1 + 2*(x^2 + y^2). Tilt = acos(-g_b_z) = acos(1 - 2*(x^2 + y^2)).
  cos_tilt = np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)
  return np.degrees(np.arccos(cos_tilt))


def _world_lin_vel_to_body_xy(
  lin_vel_w: np.ndarray, quat_wxyz: np.ndarray
) -> np.ndarray:
  """Body-frame xy linear velocity from world velocity + wxyz quaternion.

  Returns ``[N,2]`` (forward, lateral) in the body frame. Used for retention
  command-tracking: the commanded velocity is in body frame, so the achieved
  velocity must be transformed to body frame for a matched comparison.
  """
  q = np.asarray(quat_wxyz, dtype=np.float64)
  w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
  vx, vy, vz = lin_vel_w[:, 0], lin_vel_w[:, 1], lin_vel_w[:, 2]
  # R^T (world to body) rows 0 and 1.
  vb_x = (
    (1 - 2 * (y * y + z * z)) * vx + 2 * (x * y + w * z) * vy + 2 * (x * z - w * y) * vz
  )
  vb_y = (
    2 * (x * y - w * z) * vx + (1 - 2 * (x * x + z * z)) * vy + 2 * (y * z + w * x) * vz
  )
  return np.stack([vb_x, vb_y], axis=-1)


def _world_ang_vel_yaw_rate_body(
  ang_vel_w: np.ndarray, quat_wxyz: np.ndarray
) -> np.ndarray:
  """Body-frame z-component of angular velocity (yaw rate) from world + quat.

  Returns ``[N]``. For a pure-yaw orientation this equals the world z-component,
  but for a tilted torso the body z-axis differs from world z.
  """
  q = np.asarray(quat_wxyz, dtype=np.float64)
  w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
  wx, wy, wz = ang_vel_w[:, 0], ang_vel_w[:, 1], ang_vel_w[:, 2]
  # R^T row 2.
  return (
    2 * (x * z + w * y) * wx + 2 * (y * z - w * x) * wy + (1 - 2 * (x * x + y * y)) * wz
  )


def _project_planar_onto_entry_yaw(
  displacement_xy: np.ndarray, entry_yaw: float
) -> np.ndarray:
  """Forward component of a 2-D displacement along the entry-yaw axis."""
  fx = np.cos(entry_yaw)
  fy = np.sin(entry_yaw)
  return displacement_xy[:, 0] * fx + displacement_xy[:, 1] * fy


def signed_backward_displacement(root_pos_w: np.ndarray, entry_yaw: float) -> float:
  """Signed backward displacement (m) relative to the entry yaw.

  Positive means the root moved *backward* (against the entry-yaw forward
  axis) over the whole trace. The displacement is measured from the first
  frame's planar position to the last frame's planar position, projected onto
  the entry-yaw axis. The sign is heading-invariant: rotating the entry yaw
  rotates the reference axis consistently.
  """
  pos = np.asarray(root_pos_w, dtype=np.float64)
  delta = pos[-1, :2] - pos[0, :2]
  forward = _project_planar_onto_entry_yaw(delta[None, :], entry_yaw)[0]
  return float(-forward)


def total_planar_travel(root_pos_w: np.ndarray) -> float:
  """Total planar arc-length travel (m): sum of consecutive XY step lengths."""
  pos = np.asarray(root_pos_w, dtype=np.float64)
  if pos.shape[0] < 2:
    return 0.0
  diffs = np.diff(pos[:, :2], axis=0)
  return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))


def peak_tilt(torso_tilt_deg: np.ndarray) -> float:
  """Peak torso tilt (degrees) over the trace."""
  return float(np.max(np.asarray(torso_tilt_deg, dtype=np.float64)))


def first_window_index(timestamps: np.ndarray, window_s: float = 3.0) -> np.ndarray:
  """Boolean index of frames within the first ``window_s`` seconds.

  The window starts at the reset/time origin (``timestamps[0]``). Returns an
  empty mask when there are no frames.
  """
  ts = np.asarray(timestamps, dtype=np.float64)
  if ts.size == 0:
    return np.zeros(0, dtype=bool)
  return ts <= ts[0] + window_s


# --------------------------------------------------------------------------- #
# Settling.
# --------------------------------------------------------------------------- #


def settle_time(
  timestamps: np.ndarray,
  root_lin_vel_w: np.ndarray,
  root_ang_vel_w: np.ndarray,
  torso_tilt_deg: np.ndarray,
  thresholds: SettlingThresholds | None = None,
) -> tuple[float | None, bool, bool]:
  """Time to settle (s) with explicit censoring and sample-gap breaking.

  Returns ``(time_to_settle, censored, not_settled)``:

  * ``time_to_settle`` — elapsed time since the reset origin at the start of the
    first sustained window where all three criteria hold continuously for
    ``sustained_s`` seconds, or ``None`` when no such window is found.
  * ``censored`` — ``True`` when the horizon is shorter than the sustained
    window, or when the trace is too sparse (inter-sample gaps exceed
    ``max_sample_gap_s``) to verify continuous settling.
  * ``not_settled`` — ``True`` when the horizon and sample density were
    sufficient but no sustained window was found (a real failure, not censoring).

  Settling requires a *sustained* window, not a single tick. Consecutive
  samples within the window must not be separated by a gap exceeding
  ``max_sample_gap_s`` (default ``sustained_s``): two calm samples ten seconds
  apart do not establish a sustained 0.5-second calm interval.
  """
  if thresholds is None:
    thresholds = SettlingThresholds()
  ts = np.asarray(timestamps, dtype=np.float64)
  lin = np.asarray(root_lin_vel_w, dtype=np.float64)
  ang = np.asarray(root_ang_vel_w, dtype=np.float64)
  tilt = np.asarray(torso_tilt_deg, dtype=np.float64)

  n = ts.shape[0]
  if n == 0:
    return None, True, False

  duration = float(ts[-1] - ts[0])
  if duration < thresholds.sustained_s:
    # Insufficient horizon to confirm a sustained window.
    return None, True, False

  # Effective max sample gap: explicit or default to sustained_s (conservative).
  effective_max_gap = thresholds.max_sample_gap_s
  if effective_max_gap is None:
    effective_max_gap = thresholds.sustained_s

  # Sparseness check: if any inter-sample gap exceeds the max, the trace is
  # too sparse to verify continuous settling (unless a dense sub-run exists).
  gaps = np.diff(ts)
  is_sparse = bool(np.any(gaps > effective_max_gap)) if gaps.size > 0 else False

  speed_xy = np.hypot(lin[:, 0], lin[:, 1])
  yaw_rate = np.abs(ang[:, 2]) if ang.shape[1] >= 3 else np.zeros(n)
  ok = (
    (speed_xy < thresholds.speed_xy)
    & (yaw_rate < thresholds.yaw_rate)
    & (tilt < thresholds.torso_tilt_deg)
  )

  # Find the first contiguous run of `ok` where (a) all inter-sample gaps within
  # the run are <= effective_max_gap and (b) the run spans >= sustained_s.
  origin = ts[0]
  i = 0
  while i < n:
    if not ok[i]:
      i += 1
      continue
    j = i + 1
    while j < n and ok[j] and (ts[j] - ts[j - 1]) <= effective_max_gap:
      j += 1
    # Run is [i, j-1]; check its spanned duration.
    run_duration = ts[j - 1] - ts[i]
    if run_duration >= thresholds.sustained_s:
      return float(ts[i] - origin), False, False
    i = j

  # No valid sustained window found.
  if is_sparse:
    # Too sparse to verify continuous settling — report uncertainty.
    return None, True, False
  return None, False, True


# --------------------------------------------------------------------------- #
# Validation.
# --------------------------------------------------------------------------- #


def _validate_trace_arrays(
  timestamps: np.ndarray,
  root_pos_w: np.ndarray,
  root_quat_w: np.ndarray,
  root_lin_vel_w: np.ndarray,
  root_ang_vel_w: np.ndarray,
  torso_tilt_deg: np.ndarray,
) -> None:
  """Validate shapes, finiteness and strictly-increasing timestamps.

  Raises ``ValueError`` on empty/non-finite input, shape mismatch, duplicate or
  backwards timestamps, or non-unit quaternions.
  """
  ts = np.asarray(timestamps)
  pos = np.asarray(root_pos_w)
  quat = np.asarray(root_quat_w)
  lin = np.asarray(root_lin_vel_w)
  ang = np.asarray(root_ang_vel_w)
  tilt = np.asarray(torso_tilt_deg)

  if ts.ndim != 1 or ts.size == 0:
    raise ValueError("timestamps must be a non-empty 1-D array")
  n = ts.shape[0]
  _require_2d(pos, "root_pos_w", n, 3)
  _require_2d(quat, "root_quat_w", n, 4)
  _require_2d(lin, "root_lin_vel_w", n, 3)
  _require_2d(ang, "root_ang_vel_w", n, 3)
  if tilt.ndim != 1 or tilt.shape[0] != n:
    raise ValueError(
      f"torso_tilt_deg must be 1-D with length {n}, got shape {tilt.shape}"
    )

  for name, arr in [
    ("timestamps", ts),
    ("root_pos_w", pos),
    ("root_quat_w", quat),
    ("root_lin_vel_w", lin),
    ("root_ang_vel_w", ang),
    ("torso_tilt_deg", tilt),
  ]:
    if not np.all(np.isfinite(arr)):
      raise ValueError(f"{name} contains non-finite values")

  # Strictly increasing: no duplicates or backwards timestamps.
  if n >= 2 and np.any(np.diff(ts) <= 0.0):
    raise ValueError(
      "timestamps must be strictly increasing (no duplicates or backwards steps)"
    )

  qn = np.linalg.norm(quat, axis=1)
  if np.any(np.abs(qn - 1.0) > 1e-3):
    raise ValueError("root_quat_w rows must be unit quaternions")


def _require_2d(arr: np.ndarray, name: str, n: int, d: int) -> None:
  if arr.ndim != 2 or arr.shape[0] != n or arr.shape[1] != d:
    raise ValueError(f"{name} must have shape [{n},{d}], got {arr.shape}")


def _validate_scalar_fields(
  entry_yaw: float,
  group: str,
  termination: str,
  early_window_s: float,
) -> None:
  """Validate finite entry_yaw, legal group/termination labels, early window."""
  if not isinstance(entry_yaw, (int, float, np.floating)) or not np.isfinite(entry_yaw):
    raise ValueError(f"entry_yaw must be finite, got {entry_yaw!r}")
  if group not in _VALID_GROUPS:
    raise ValueError(f"group must be one of {sorted(_VALID_GROUPS)}, got {group!r}")
  if termination not in _VALID_TERMINATIONS:
    raise ValueError(
      f"termination must be one of {sorted(_VALID_TERMINATIONS)}, got {termination!r}"
    )
  if (
    not isinstance(early_window_s, (int, float, np.floating))
    or not np.isfinite(early_window_s)
    or early_window_s < 0
  ):
    raise ValueError(
      f"early_window_s must be nonnegative finite, got {early_window_s!r}"
    )


def _validate_vel_command_b(vel_command_b: np.ndarray | None, n: int) -> None:
  """Validate commanded body-frame velocity when provided."""
  if vel_command_b is None:
    return
  cmd = np.asarray(vel_command_b)
  if cmd.ndim != 2 or cmd.shape[0] != n or cmd.shape[1] != 3:
    raise ValueError(f"vel_command_b must have shape [{n},3], got {cmd.shape}")
  if not np.all(np.isfinite(cmd)):
    raise ValueError("vel_command_b contains non-finite values")


# --------------------------------------------------------------------------- #
# Per-trajectory computation.
# --------------------------------------------------------------------------- #


def compute_recovery_metrics(
  trace: RecoveryTrace | RetentionTrace,
  thresholds: SettlingThresholds | None = None,
  early_window_s: float = 3.0,
) -> RecoveryMetrics:
  """Compute per-trajectory recovery/retention metrics.

  For **recovery** traces (zero command): validates arrays, computes settling
  with explicit censoring and sample-gap breaking, signed backward displacement
  relative to the entry yaw, total planar travel, peak tilt, first-window
  statistics and torque diagnostics. A terminal failure (fall/orientation/
  invalid) invalidates any prior calm window — ``settled`` is ``False`` even if
  a calm window was observed before the failure. The earliest calm-window
  timing is preserved as ``earliest_calm_window_s`` (diagnostic, not success).

  For **retention** traces (original nonzero command): keeps the original
  terminal category (no settling recategorization), computes body-frame
  command-tracking error (linear + yaw) in matched frames, and reports
  displacement/travel/tilt. Intended walking is never reclassified as a
  recovery failure. Missing ``vel_command_b`` makes tracking diagnostics
  explicitly unavailable (``None``), never a silent zero command.
  """
  _validate_trace_arrays(
    trace.timestamps,
    trace.root_pos_w,
    trace.root_quat_w,
    trace.root_lin_vel_w,
    trace.root_ang_vel_w,
    trace.torso_tilt_deg,
  )
  _validate_scalar_fields(
    trace.entry_yaw, trace.group, trace.termination, early_window_s
  )
  if thresholds is None:
    thresholds = SettlingThresholds()

  n = trace.timestamps.shape[0]
  duration = float(trace.timestamps[-1] - trace.timestamps[0])

  bwd = signed_backward_displacement(trace.root_pos_w, trace.entry_yaw)
  travel = total_planar_travel(trace.root_pos_w)
  pk = peak_tilt(trace.torso_tilt_deg)

  # Early-window statistics: actual span of included timestamps.
  early_mask = first_window_index(trace.timestamps, early_window_s)
  if np.any(early_mask):
    early_pos = trace.root_pos_w[early_mask]
    early_tilt = trace.torso_tilt_deg[early_mask]
    early_bwd = signed_backward_displacement(early_pos, trace.entry_yaw)
    early_travel = total_planar_travel(early_pos)
    early_peak = peak_tilt(early_tilt)
    early_used = float(trace.timestamps[early_mask][-1] - trace.timestamps[0])
  else:
    early_bwd = bwd
    early_travel = travel
    early_peak = pk
    early_used = 0.0

  torque_ratio = _torque_ratio(trace.applied_torque, trace.requested_torque)

  is_retention = trace.group == "retention"

  if is_retention:
    # Retention: no settling recategorization; command-tracking diagnostics.
    cmd_lin_err, cmd_yaw_err, cmd_peak_lin = _command_tracking_error(trace, n)
    return RecoveryMetrics(
      trajectory_id=trace.trajectory_id,
      group="retention",
      termination=trace.termination,
      settled=False,
      time_to_settle=None,
      earliest_calm_window_s=None,
      censored=False,
      not_settled=False,
      peak_tilt_deg=pk,
      signed_backward_displacement_m=bwd,
      total_planar_travel_m=travel,
      early_peak_tilt_deg=early_peak,
      early_signed_backward_displacement_m=early_bwd,
      early_total_planar_travel_m=early_travel,
      early_window_s=early_used,
      duration_s=duration,
      num_frames=n,
      torque_requested_vs_applied_ratio=torque_ratio,
      mean_command_lin_error_mps=cmd_lin_err,
      mean_command_yaw_error_radps=cmd_yaw_err,
      peak_command_lin_error_mps=cmd_peak_lin,
      contributes_settled=False,
    )

  # Recovery: settling with terminal-outcome masking.
  tts, censored, _ns = settle_time(
    trace.timestamps,
    trace.root_lin_vel_w,
    trace.root_ang_vel_w,
    trace.torso_tilt_deg,
    thresholds,
  )
  earliest_calm = tts  # raw diagnostic, preserved even for falls

  terminal_failed = trace.termination in _TERMINAL_FAILURE_CATEGORIES
  # Settled only when a sustained calm window exists, not censored, and the
  # episode did not end in a terminal failure (fall/orientation/invalid).
  settled = (
    tts is not None
    and not censored
    and not terminal_failed
    and trace.termination == "timeout"
  )

  # Recategorize: timeout that never settled (and wasn't censored) → not_settled.
  category: str = trace.termination
  not_settled = False
  if trace.termination == "timeout" and not settled and not censored:
    category = "not_settled"
    not_settled = True

  return RecoveryMetrics(
    trajectory_id=trace.trajectory_id,
    group="recovery",
    termination=category,
    settled=settled,
    time_to_settle=tts if settled else None,
    earliest_calm_window_s=earliest_calm,
    censored=censored,
    not_settled=not_settled,
    peak_tilt_deg=pk,
    signed_backward_displacement_m=bwd,
    total_planar_travel_m=travel,
    early_peak_tilt_deg=early_peak,
    early_signed_backward_displacement_m=early_bwd,
    early_total_planar_travel_m=early_travel,
    early_window_s=early_used,
    duration_s=duration,
    num_frames=n,
    torque_requested_vs_applied_ratio=torque_ratio,
    mean_command_lin_error_mps=None,
    mean_command_yaw_error_radps=None,
    peak_command_lin_error_mps=None,
    contributes_settled=settled,
  )


def _command_tracking_error(
  trace: RecoveryTrace | RetentionTrace, n: int
) -> tuple[float | None, float | None, float | None]:
  """Body-frame command-tracking error for retention traces.

  Returns ``(mean_lin_error_mps, mean_yaw_error_radps, peak_lin_error_mps)``.
  All ``None`` when ``vel_command_b`` is unavailable (explicitly, never a
  silent zero command). Compares commanded body-frame velocity against the
  achieved body-frame velocity (world velocity transformed to body frame via
  the root quaternion) in matched frames.
  """
  vel_cmd = getattr(trace, "vel_command_b", None)
  if vel_cmd is None:
    return None, None, None
  _validate_vel_command_b(vel_cmd, n)
  cmd = np.asarray(vel_cmd, dtype=np.float64)

  achieved_lin_b = _world_lin_vel_to_body_xy(trace.root_lin_vel_w, trace.root_quat_w)
  achieved_yaw_b = _world_ang_vel_yaw_rate_body(trace.root_ang_vel_w, trace.root_quat_w)

  lin_err = np.hypot(
    cmd[:, 0] - achieved_lin_b[:, 0],
    cmd[:, 1] - achieved_lin_b[:, 1],
  )
  yaw_err = np.abs(cmd[:, 2] - achieved_yaw_b)

  return (
    float(np.mean(lin_err)),
    float(np.mean(yaw_err)),
    float(np.max(lin_err)),
  )


def _torque_ratio(
  applied: np.ndarray | None,
  requested: np.ndarray | None,
) -> float | None:
  """Mean |applied|/|requested| torque ratio, or None when unavailable.

  Only computed when both arrays are explicitly provided and labeled. Guards
  against zero-division and non-finite entries.
  """
  if applied is None or requested is None:
    return None
  ap = np.asarray(applied, dtype=np.float64)
  rq = np.asarray(requested, dtype=np.float64)
  if ap.shape != rq.shape or ap.size == 0:
    return None
  if not (np.all(np.isfinite(ap)) and np.all(np.isfinite(rq))):
    return None
  denom = np.abs(rq)
  numer = np.abs(ap)
  mask = denom > 1e-9
  if not np.any(mask):
    return None
  return float(np.mean(numer[mask] / denom[mask]))


# --------------------------------------------------------------------------- #
# Aggregation.
# --------------------------------------------------------------------------- #


def aggregate_recovery_metrics(
  metrics: list[RecoveryMetrics],
) -> list[RecoveryGroupStats]:
  """Aggregate per-trajectory metrics into per-group statistics.

  Splits by ``group`` field. No favorable aggregate is produced from
  failed/censored episodes: ``settle_rate`` counts only actually-settled
  recovery trajectories, and time-to-settle aggregates are over settled
  episodes only (``None`` when none settled). Unavailable means are ``None``,
  never ``NaN``. Retention groups report command-tracking aggregates.
  """
  groups: dict[str, list[RecoveryMetrics]] = {}
  for m in metrics:
    groups.setdefault(m.group, []).append(m)

  stats: list[RecoveryGroupStats] = []
  for group, ms in sorted(groups.items()):
    total = len(ms)
    num_settled = sum(1 for m in ms if m.settled)
    num_falls = sum(1 for m in ms if m.termination == "fall")
    num_orient = sum(1 for m in ms if m.termination == "orientation")
    num_timeout = sum(1 for m in ms if m.termination == "timeout")
    num_invalid = sum(1 for m in ms if m.termination == "invalid")
    num_not_settled = sum(1 for m in ms if m.termination == "not_settled")
    num_censored = sum(1 for m in ms if m.censored)

    settled_tts = [
      m.time_to_settle for m in ms if m.settled and m.time_to_settle is not None
    ]
    if settled_tts:
      med = float(np.median(settled_tts))
      mean = float(np.mean(settled_tts))
    else:
      med = None
      mean = None

    # Means over non-invalid trajectories; None when no valid data.
    valid = [m for m in ms if m.termination != "invalid"]
    if valid:
      mean_peak = float(np.mean([m.peak_tilt_deg for m in valid]))
      mean_bwd = float(np.mean([m.signed_backward_displacement_m for m in valid]))
      mean_travel = float(np.mean([m.total_planar_travel_m for m in valid]))
      coverage = len(valid) / total if total else 0.0
    else:
      mean_peak = None
      mean_bwd = None
      mean_travel = None
      coverage = 0.0

    # Retention command-tracking aggregates.
    tracked = [m for m in ms if m.mean_command_lin_error_mps is not None]
    if tracked:
      lin_errs = np.array([m.mean_command_lin_error_mps for m in tracked])
      yaw_errs = np.array([m.mean_command_yaw_error_radps for m in tracked])
      mean_cmd_lin = float(np.mean(lin_errs))
      mean_cmd_yaw = float(np.mean(yaw_errs))
    else:
      mean_cmd_lin = None
      mean_cmd_yaw = None

    stats.append(
      RecoveryGroupStats(
        group=group,
        num_trajectories=total,
        num_settled=num_settled,
        num_falls=num_falls,
        num_orientation_terminations=num_orient,
        num_timeouts=num_timeout,
        num_invalid=num_invalid,
        num_not_settled=num_not_settled,
        num_censored=num_censored,
        settle_rate=(num_settled / total) if total else 0.0,
        median_time_to_settle_s=med,
        mean_time_to_settle_s=mean,
        mean_peak_tilt_deg=mean_peak,
        mean_signed_backward_displacement_m=mean_bwd,
        mean_total_planar_travel_m=mean_travel,
        coverage=coverage,
        mean_command_lin_error_mps=mean_cmd_lin,
        mean_command_yaw_error_radps=mean_cmd_yaw,
        num_command_tracked=len(tracked),
      )
    )
  return stats
