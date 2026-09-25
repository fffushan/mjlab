"""Tests for X2 tennis-end recovery/retention trace metrics and bounded
evaluation schedule primitives.

Uses synthetic traces (pure NumPy, no simulator/runner) to prove:

* displacement sign under arbitrary heading,
* sustained (not single-tick) settling with sample-gap breaking,
* reset/time-origin handling,
* insufficient horizon / censoring,
* early-window statistics,
* empty / non-finite / non-strictly-increasing input rejection,
* coverage / split semantics,
* deterministic bounded schedules,
* no favorable aggregate from failed/censored/terminal-failure episodes,
* requested-vs-applied torque diagnostics only when explicitly available,
* retention command-tracking diagnostics (body-frame, matched frames).

Regression tests for the three parent-review blockers:
* initially calm then later fall is not counted settled,
* perfect nonzero-command retention tracking is not reclassified as a recovery
  failure,
* two calm samples ten seconds apart are not claimed continuously settled.

No dependency on the pool loader, runtime factory or resume runner.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mjlab.tasks.velocity.mdp.tennis_recovery_metrics import (
  RecoveryGroupStats,
  RecoveryMetrics,
  RecoveryTrace,
  RetentionTrace,
  SettlingThresholds,
  TerminationCategory,
  aggregate_recovery_metrics,
  compute_recovery_metrics,
  first_window_index,
  peak_tilt,
  settle_time,
  signed_backward_displacement,
  torso_tilt_from_quat,
  total_planar_travel,
)
from mjlab.tasks.velocity.scripts.tennis_recovery_eval import (
  EvaluationCliArgs,
  EvaluationSchedule,
  build_evaluation_schedule,
  format_group_summary,
)

FPS = 50.0
DT = 1.0 / FPS


# --------------------------------------------------------------------------- #
# Synthetic trace helpers.
# --------------------------------------------------------------------------- #


def _timestamps(n: int, origin: float = 0.0) -> np.ndarray:
  return origin + np.arange(n, dtype=np.float64) * DT


def _unit_quat(n: int) -> np.ndarray:
  """n upright unit wxyz quaternions (1,0,0,0)."""
  q = np.zeros((n, 4), dtype=np.float64)
  q[:, 0] = 1.0
  return q


def _quat_from_yaw(yaw: float, n: int) -> np.ndarray:
  """n unit wxyz quaternions with only yaw component."""
  q = np.zeros((n, 4), dtype=np.float64)
  q[:, 0] = math.cos(yaw / 2.0)
  q[:, 3] = math.sin(yaw / 2.0)
  return q


def _make_trace(
  n: int = 100,
  *,
  origin: float = 0.0,
  entry_yaw: float = 0.0,
  root_pos_w=None,
  root_lin_vel_w=None,
  root_ang_vel_w=None,
  torso_tilt_deg=None,
  root_quat_w=None,
  trajectory_id: int = 0,
  group: str = "recovery",
  termination: TerminationCategory = "timeout",
  applied_torque=None,
  requested_torque=None,
) -> RecoveryTrace:
  ts = _timestamps(n, origin)
  if root_pos_w is None:
    root_pos_w = np.zeros((n, 3), dtype=np.float64)
  if root_lin_vel_w is None:
    root_lin_vel_w = np.zeros((n, 3), dtype=np.float64)
  if root_ang_vel_w is None:
    root_ang_vel_w = np.zeros((n, 3), dtype=np.float64)
  if torso_tilt_deg is None:
    torso_tilt_deg = np.zeros(n, dtype=np.float64)
  if root_quat_w is None:
    root_quat_w = _unit_quat(n)
  return RecoveryTrace(
    timestamps=ts,
    root_pos_w=np.asarray(root_pos_w, dtype=np.float64),
    root_quat_w=np.asarray(root_quat_w, dtype=np.float64),
    root_lin_vel_w=np.asarray(root_lin_vel_w, dtype=np.float64),
    root_ang_vel_w=np.asarray(root_ang_vel_w, dtype=np.float64),
    torso_tilt_deg=np.asarray(torso_tilt_deg, dtype=np.float64),
    entry_yaw=entry_yaw,
    trajectory_id=trajectory_id,
    group=group,
    termination=termination,
    applied_torque=applied_torque,
    requested_torque=requested_torque,
  )


def _calm_trace(n: int = 100, **kw) -> RecoveryTrace:
  """A trace that is fully calm (zero velocity, zero tilt)."""
  return _make_trace(n, **kw)


def _make_metrics(**overrides) -> RecoveryMetrics:
  """Build a RecoveryMetrics with defaults, applying overrides."""
  defaults: dict = dict(
    trajectory_id=0,
    group="recovery",
    termination="timeout",
    settled=True,
    time_to_settle=1.0,
    earliest_calm_window_s=1.0,
    censored=False,
    not_settled=False,
    peak_tilt_deg=5.0,
    signed_backward_displacement_m=-0.5,
    total_planar_travel_m=1.0,
    early_peak_tilt_deg=5.0,
    early_signed_backward_displacement_m=-0.5,
    early_total_planar_travel_m=1.0,
    early_window_s=3.0,
    duration_s=20.0,
    num_frames=1000,
    torque_requested_vs_applied_ratio=None,
    mean_command_lin_error_mps=None,
    mean_command_yaw_error_radps=None,
    peak_command_lin_error_mps=None,
    contributes_settled=True,
  )
  defaults.update(overrides)
  return RecoveryMetrics(**defaults)


# --------------------------------------------------------------------------- #
# Geometry helpers.
# --------------------------------------------------------------------------- #


def test_torso_tilt_from_quat_upright_is_zero():
  tilt = torso_tilt_from_quat(_unit_quat(5))
  assert np.allclose(tilt, 0.0)


def test_torso_tilt_from_quat_90deg_pitch():
  q = np.array([[math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]])
  tilt = torso_tilt_from_quat(q)
  assert tilt[0] == pytest.approx(90.0, abs=1e-6)


def test_torso_tilt_from_quat_yaw_only_is_zero():
  q = _quat_from_yaw(1.7, 3)
  tilt = torso_tilt_from_quat(q)
  assert np.allclose(tilt, 0.0, atol=1e-9)


def test_total_planar_travel_zero():
  pos = np.zeros((5, 3))
  assert total_planar_travel(pos) == 0.0


def test_total_planar_travel_arc_length():
  pos = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float64)
  assert total_planar_travel(pos) == pytest.approx(2.0)


def test_peak_tilt():
  tilt = np.array([1.0, 5.0, 3.0])
  assert peak_tilt(tilt) == 5.0


# --------------------------------------------------------------------------- #
# Displacement sign under arbitrary heading.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("yaw", [0.0, 0.5, 1.0, math.pi / 2, math.pi, -1.3, 2.7])
def test_displacement_forward_is_negative_backward(yaw):
  fwd = np.array([math.cos(yaw), math.sin(yaw)])
  pos = np.zeros((2, 3))
  pos[1, :2] = fwd
  bwd = signed_backward_displacement(pos, yaw)
  assert bwd == pytest.approx(-1.0, abs=1e-9)


@pytest.mark.parametrize("yaw", [0.0, 0.7, math.pi / 3, math.pi, -2.1])
def test_displacement_backward_is_positive(yaw):
  back = -np.array([math.cos(yaw), math.sin(yaw)])
  pos = np.zeros((2, 3))
  pos[1, :2] = back
  bwd = signed_backward_displacement(pos, yaw)
  assert bwd == pytest.approx(1.0, abs=1e-9)


def test_displacement_lateral_is_zero_along_entry_yaw():
  yaw = 0.0
  pos = np.zeros((2, 3))
  pos[1, 1] = 2.0
  bwd = signed_backward_displacement(pos, yaw)
  assert bwd == pytest.approx(0.0, abs=1e-9)


def test_displacement_sign_heading_invariant_under_rotation():
  for yaw in np.linspace(-math.pi, math.pi, 13):
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    pos = np.zeros((2, 3))
    pos[1, :2] = fwd * 0.5
    assert signed_backward_displacement(pos, yaw) == pytest.approx(-0.5, abs=1e-9)


# --------------------------------------------------------------------------- #
# Sustained (not single-tick) settling with sample-gap breaking.
# --------------------------------------------------------------------------- #


def test_settle_requires_sustained_window_not_single_tick():
  n = 100
  ts = _timestamps(n)
  lin = np.full((n, 3), 0.5)
  ang = np.full((n, 3), 0.0)
  tilt = np.full(n, 0.0)
  lin[50] = 0.0  # single calm tick
  tts, censored, not_settled = settle_time(ts, lin, ang, tilt)
  assert tts is None
  assert not censored
  assert not_settled


def test_settle_sustained_window_settles():
  n = 100
  ts = _timestamps(n)
  lin = np.full((n, 3), 0.5)
  ang = np.full((n, 3), 0.0)
  tilt = np.full(n, 0.0)
  lin[:30] = 0.0  # 0.58s >= 0.5s
  tts, censored, not_settled = settle_time(ts, lin, ang, tilt)
  assert tts is not None
  assert tts == pytest.approx(0.0, abs=1e-9)
  assert not censored
  assert not not_settled


def test_settle_window_shorter_than_sustained_does_not_settle():
  n = 100
  ts = _timestamps(n)
  lin = np.full((n, 3), 0.5)
  ang = np.full((n, 3), 0.0)
  tilt = np.full(n, 0.0)
  lin[:20] = 0.0  # 0.38s < 0.5s
  tts, censored, not_settled = settle_time(ts, lin, ang, tilt)
  assert tts is None
  assert not censored
  assert not_settled


def test_settle_settles_at_first_sustained_window():
  n = 100
  ts = _timestamps(n)
  lin = np.full((n, 3), 0.0)
  ang = np.full((n, 3), 0.0)
  tilt = np.full(n, 0.0)
  lin[:50] = 0.5  # first 1s unsettled
  tts, _, _ = settle_time(ts, lin, ang, tilt)
  assert tts is not None
  assert tts == pytest.approx(1.0, abs=1e-9)


def test_settle_respects_yaw_rate_threshold():
  n = 100
  ts = _timestamps(n)
  lin = np.zeros((n, 3))
  ang = np.full((n, 3), 0.3)
  tilt = np.zeros(n)
  tts, censored, not_settled = settle_time(ts, lin, ang, tilt)
  assert tts is None
  assert not_settled


def test_settle_respects_tilt_threshold():
  n = 100
  ts = _timestamps(n)
  lin = np.zeros((n, 3))
  ang = np.zeros((n, 3))
  tilt = np.full(n, 15.0)
  tts, censored, not_settled = settle_time(ts, lin, ang, tilt)
  assert tts is None
  assert not_settled


# --------------------------------------------------------------------------- #
# Regression: blocker 3 — two calm samples ten seconds apart.
# --------------------------------------------------------------------------- #


def test_settle_two_samples_far_apart_is_censored():
  # Two calm samples 10s apart: cannot establish a sustained 0.5s interval.
  tts, censored, not_settled = settle_time(
    np.array([0.0, 10.0]),
    np.zeros((2, 3)),
    np.zeros((2, 3)),
    np.zeros(2),
  )
  assert tts is None
  assert censored
  assert not not_settled


def test_settle_dense_50hz_window_settles():
  # 100 frames at 50Hz (2s) calm: a valid continuous window.
  n = 100
  ts = _timestamps(n)
  tts, censored, not_settled = settle_time(
    ts, np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n)
  )
  assert tts is not None
  assert tts == pytest.approx(0.0, abs=1e-9)
  assert not censored
  assert not not_settled


def test_settle_gap_breaks_window_within_trace():
  # Dense calm, then a 1s gap, then dense calm again. The first dense block
  # (0.5s) should settle before the gap is reached.
  n = 50
  ts = _timestamps(n)  # 1.0s
  # Insert a large gap by replacing the middle timestamps.
  ts[25:] = ts[25:] + 5.0  # 5s gap between frame 24 and 25
  tts, censored, not_settled = settle_time(
    ts, np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n)
  )
  # First block [0,24] = 0.48s < 0.5s, so it must not settle there; the gap
  # breaks the second block. Since the gap > sustained_s, the whole trace is
  # censored (too sparse to verify).
  assert censored or tts is None


def test_settle_explicit_max_sample_gap_breaks_window():
  # Tight max_sample_gap: a 0.1s gap breaks the window even though the default
  # (sustained_s=0.5) would allow it.
  n = 100
  ts = _timestamps(n)
  ts[50:] = ts[50:] + 0.2  # 0.2s gap; with max_gap=0.1 this breaks
  thresholds = SettlingThresholds(max_sample_gap_s=0.1)
  tts, censored, not_settled = settle_time(
    ts, np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n), thresholds
  )
  # First block [0,49] = 0.98s, but gap at 50 is 0.2 > 0.1 so the window
  # cannot extend. First block alone is >= 0.5s so it settles at t=0.
  # The first contiguous run [0,49] has gap 0.02 each, span 0.98 >= 0.5.
  assert tts is not None
  assert tts == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Reset / time-origin handling.
# --------------------------------------------------------------------------- #


def test_settle_time_origin_invariant():
  n = 100
  lin = np.zeros((n, 3))
  ang = np.zeros((n, 3))
  tilt = np.zeros(n)
  lin[:50] = 0.5  # unsettled first 1s
  tts0, _, _ = settle_time(_timestamps(n, 0.0), lin, ang, tilt)
  tts5, _, _ = settle_time(_timestamps(n, 5.0), lin, ang, tilt)
  assert tts0 == pytest.approx(1.0, abs=1e-9)
  assert tts5 == pytest.approx(1.0, abs=1e-9)


def test_first_window_index_uses_origin():
  ts = _timestamps(200, 7.0)
  mask = first_window_index(ts, 3.0)
  assert mask.sum() == 151
  assert not mask[151]
  assert mask[0]


def test_compute_metrics_origin_invariant_settle_time():
  trace = _calm_trace(100, origin=10.0)
  m = compute_recovery_metrics(trace)
  assert m.settled
  assert m.time_to_settle == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Insufficient horizon / censoring.
# --------------------------------------------------------------------------- #


def test_insufficient_horizon_censored():
  trace = _calm_trace(10)
  m = compute_recovery_metrics(trace)
  assert m.censored
  assert not m.settled
  assert m.time_to_settle is None


def test_single_frame_trace_censored():
  trace = _calm_trace(1)
  m = compute_recovery_metrics(trace)
  assert m.censored
  assert not m.settled


def test_sufficient_horizon_no_settle_is_not_settled():
  n = 100
  lin = np.full((n, 3), 0.5)
  trace = _make_trace(n, root_lin_vel_w=lin, termination="timeout")
  m = compute_recovery_metrics(trace)
  assert not m.censored
  assert not m.settled
  assert m.not_settled
  assert m.termination == "not_settled"


def test_censored_episode_keeps_caller_termination():
  trace = _calm_trace(10, termination="timeout")
  m = compute_recovery_metrics(trace)
  assert m.censored
  assert m.termination == "timeout"


# --------------------------------------------------------------------------- #
# Regression: blocker 1 — initially calm then later fall.
# --------------------------------------------------------------------------- #


def test_calm_then_fall_not_settled():
  # 101 samples at 0.02s (2.02s), zero velocities, last frame tilt=90 (fall).
  n = 101
  dt = 0.02
  ts = np.arange(n, dtype=np.float64) * dt
  lin = np.zeros((n, 3))
  ang = np.zeros((n, 3))
  tilt = np.zeros(n)
  tilt[-1] = 90.0
  trace = RecoveryTrace(
    timestamps=ts,
    root_pos_w=np.zeros((n, 3)),
    root_quat_w=_unit_quat(n),
    root_lin_vel_w=lin,
    root_ang_vel_w=ang,
    torso_tilt_deg=tilt,
    entry_yaw=0.0,
    trajectory_id=0,
    group="recovery",
    termination="fall",
  )
  m = compute_recovery_metrics(trace)
  assert not m.settled
  assert not m.contributes_settled
  assert m.termination == "fall"
  assert m.time_to_settle is None
  # Earliest calm window is preserved as a diagnostic, not success.
  assert m.earliest_calm_window_s is not None
  assert m.earliest_calm_window_s == pytest.approx(0.0, abs=1e-9)


def test_calm_then_fall_aggregate_no_favorable():
  n = 101
  dt = 0.02
  ts = np.arange(n, dtype=np.float64) * dt
  trace = RecoveryTrace(
    timestamps=ts,
    root_pos_w=np.zeros((n, 3)),
    root_quat_w=_unit_quat(n),
    root_lin_vel_w=np.zeros((n, 3)),
    root_ang_vel_w=np.zeros((n, 3)),
    torso_tilt_deg=np.concatenate([np.zeros(n - 1), [90.0]]),
    entry_yaw=0.0,
    trajectory_id=0,
    group="recovery",
    termination="fall",
  )
  m = compute_recovery_metrics(trace)
  stats = aggregate_recovery_metrics([m])
  s = stats[0]
  assert s.num_settled == 0
  assert s.num_falls == 1
  assert s.settle_rate == 0.0
  assert s.median_time_to_settle_s is None
  assert s.mean_time_to_settle_s is None


def test_calm_then_orientation_termination_not_settled():
  n = 100
  tilt = np.zeros(n)
  tilt[-5:] = 30.0  # bad orientation at the end
  trace = _make_trace(n, torso_tilt_deg=tilt, termination="orientation")
  m = compute_recovery_metrics(trace)
  assert not m.settled
  assert m.termination == "orientation"
  assert m.earliest_calm_window_s is not None


# --------------------------------------------------------------------------- #
# Regression: blocker 2 — retention command tracking.
# --------------------------------------------------------------------------- #


def test_perfect_retention_tracking_not_reclassified():
  # Perfect 1m/s forward walk, commanded 1m/s, reaches timeout.
  n = 100
  ts = _timestamps(n)
  pos = np.zeros((n, 3))
  pos[:, 0] = np.arange(n) * DT
  lin = np.zeros((n, 3))
  lin[:, 0] = 1.0
  cmd = np.zeros((n, 3))
  cmd[:, 0] = 1.0
  trace = RetentionTrace(
    timestamps=ts,
    root_pos_w=pos,
    root_quat_w=_unit_quat(n),
    root_lin_vel_w=lin,
    root_ang_vel_w=np.zeros((n, 3)),
    torso_tilt_deg=np.zeros(n),
    entry_yaw=0.0,
    trajectory_id=0,
    termination="timeout",
    vel_command_b=cmd,
  )
  m = compute_recovery_metrics(trace)
  assert m.termination == "timeout"
  assert not m.settled
  assert not m.not_settled
  assert m.mean_command_lin_error_mps == pytest.approx(0.0, abs=1e-9)
  assert m.mean_command_yaw_error_radps == pytest.approx(0.0, abs=1e-9)
  assert m.peak_command_lin_error_mps == pytest.approx(0.0, abs=1e-9)


def test_retention_tracking_with_nonzero_heading():
  # Walking at 1m/s along a 90deg yaw. World velocity is +y; body-frame forward
  # (rotated by the yaw) should be 1m/s. Commanded forward is 1m/s.
  n = 50
  yaw = math.pi / 2
  ts = _timestamps(n)
  pos = np.zeros((n, 3))
  pos[:, 1] = np.arange(n) * DT  # moving in +y (world)
  lin = np.zeros((n, 3))
  lin[:, 1] = 1.0  # world velocity +y
  cmd = np.zeros((n, 3))
  cmd[:, 0] = 1.0  # commanded body-frame forward
  trace = RetentionTrace(
    timestamps=ts,
    root_pos_w=pos,
    root_quat_w=_quat_from_yaw(yaw, n),
    root_lin_vel_w=lin,
    root_ang_vel_w=np.zeros((n, 3)),
    torso_tilt_deg=np.zeros(n),
    entry_yaw=yaw,
    trajectory_id=0,
    termination="timeout",
    vel_command_b=cmd,
  )
  m = compute_recovery_metrics(trace)
  # Body-frame forward velocity = R^T(world_v) projected onto body x = 1m/s.
  assert m.mean_command_lin_error_mps == pytest.approx(0.0, abs=1e-9)


def test_retention_tracking_with_yaw_rate_command():
  # Commanded yaw rate 0.5 rad/s; achieved world ang_vel_z = 0.5.
  n = 50
  ts = _timestamps(n)
  cmd = np.zeros((n, 3))
  cmd[:, 2] = 0.5
  ang = np.zeros((n, 3))
  ang[:, 2] = 0.5
  trace = RetentionTrace(
    timestamps=ts,
    root_pos_w=np.zeros((n, 3)),
    root_quat_w=_unit_quat(n),
    root_lin_vel_w=np.zeros((n, 3)),
    root_ang_vel_w=ang,
    torso_tilt_deg=np.zeros(n),
    entry_yaw=0.0,
    trajectory_id=0,
    termination="timeout",
    vel_command_b=cmd,
  )
  m = compute_recovery_metrics(trace)
  assert m.mean_command_yaw_error_radps == pytest.approx(0.0, abs=1e-9)


def test_retention_missing_command_data_is_unavailable():
  # vel_command_b=None: tracking diagnostics explicitly None, not zero.
  n = 50
  trace = RetentionTrace(
    timestamps=_timestamps(n),
    root_pos_w=np.zeros((n, 3)),
    root_quat_w=_unit_quat(n),
    root_lin_vel_w=np.zeros((n, 3)),
    root_ang_vel_w=np.zeros((n, 3)),
    torso_tilt_deg=np.zeros(n),
    entry_yaw=0.0,
    trajectory_id=0,
    termination="timeout",
  )
  m = compute_recovery_metrics(trace)
  assert m.mean_command_lin_error_mps is None
  assert m.mean_command_yaw_error_radps is None
  assert m.peak_command_lin_error_mps is None
  assert m.termination == "timeout"


def test_retention_changing_commands():
  # Command switches from 1m/s to 0 at the midpoint; achieved tracks it.
  n = 100
  ts = _timestamps(n)
  lin = np.zeros((n, 3))
  lin[:50, 0] = 1.0
  cmd = np.zeros((n, 3))
  cmd[:50, 0] = 1.0
  trace = RetentionTrace(
    timestamps=ts,
    root_pos_w=np.zeros((n, 3)),
    root_quat_w=_unit_quat(n),
    root_lin_vel_w=lin,
    root_ang_vel_w=np.zeros((n, 3)),
    torso_tilt_deg=np.zeros(n),
    entry_yaw=0.0,
    trajectory_id=0,
    termination="timeout",
    vel_command_b=cmd,
  )
  m = compute_recovery_metrics(trace)
  assert m.mean_command_lin_error_mps == pytest.approx(0.0, abs=1e-9)


def test_retention_aggregate_reports_command_tracking():
  n = 50
  cmd = np.zeros((n, 3))
  cmd[:, 0] = 1.0
  lin = np.zeros((n, 3))
  lin[:, 0] = 1.0
  trace = RetentionTrace(
    timestamps=_timestamps(n),
    root_pos_w=np.zeros((n, 3)),
    root_quat_w=_unit_quat(n),
    root_lin_vel_w=lin,
    root_ang_vel_w=np.zeros((n, 3)),
    torso_tilt_deg=np.zeros(n),
    entry_yaw=0.0,
    trajectory_id=0,
    termination="timeout",
    vel_command_b=cmd,
  )
  m = compute_recovery_metrics(trace)
  stats = aggregate_recovery_metrics([m])
  s = stats[0]
  assert s.group == "retention"
  assert s.mean_command_lin_error_mps == pytest.approx(0.0, abs=1e-9)
  assert s.num_command_tracked == 1


# --------------------------------------------------------------------------- #
# Early-window statistics.
# --------------------------------------------------------------------------- #


def test_early_window_stats_use_first_3s_only():
  n = 200  # 4s
  tilt = np.zeros(n)
  tilt[151:] = 30.0  # after the 3s boundary
  trace = _make_trace(n, torso_tilt_deg=tilt)
  m = compute_recovery_metrics(trace)
  assert m.early_peak_tilt_deg == pytest.approx(0.0)
  assert m.peak_tilt_deg == pytest.approx(30.0)
  assert m.early_window_s == pytest.approx(3.0, abs=1e-9)


def test_early_window_displacement_uses_first_3s():
  n = 200
  pos = np.zeros((n, 3))
  pos[:151, 0] = np.linspace(0, 1, 151)
  pos[151:, 0] = 1.0 - np.linspace(0, 6, 49)
  trace = _make_trace(n, root_pos_w=pos, entry_yaw=0.0)
  m = compute_recovery_metrics(trace)
  assert m.early_signed_backward_displacement_m == pytest.approx(-1.0, abs=1e-9)
  assert m.signed_backward_displacement_m == pytest.approx(5.0, abs=1e-9)


def test_early_window_shorter_than_3s_uses_actual_span():
  # 1s trace: early window actual span is the trace span.
  trace = _calm_trace(50)  # 1s
  m = compute_recovery_metrics(trace)
  assert m.early_window_s == pytest.approx(1.0 - DT, abs=1e-6)


# --------------------------------------------------------------------------- #
# Empty / non-finite / non-strictly-increasing input rejection.
# --------------------------------------------------------------------------- #


def test_empty_trace_rejected():
  trace = _make_trace(0)
  with pytest.raises(ValueError, match="non-empty"):
    compute_recovery_metrics(trace)


def test_nonfinite_velocity_rejected():
  n = 10
  lin = np.zeros((n, 3))
  lin[3, 0] = float("nan")
  trace = _make_trace(n, root_lin_vel_w=lin)
  with pytest.raises(ValueError, match="non-finite"):
    compute_recovery_metrics(trace)


def test_nonfinite_position_rejected():
  n = 10
  pos = np.zeros((n, 3))
  pos[2, 1] = float("inf")
  trace = _make_trace(n, root_pos_w=pos)
  with pytest.raises(ValueError, match="non-finite"):
    compute_recovery_metrics(trace)


def test_non_unit_quaternion_rejected():
  n = 10
  q = _unit_quat(n)
  q[5] = np.array([2.0, 0.0, 0.0, 0.0])
  trace = _make_trace(n, root_quat_w=q)
  with pytest.raises(ValueError, match="unit quaternions"):
    compute_recovery_metrics(trace)


def test_duplicate_timestamps_rejected():
  n = 10
  ts = _timestamps(n)
  ts[5] = ts[4]  # duplicate
  trace = _make_trace(n)
  trace.timestamps = ts
  with pytest.raises(ValueError, match="strictly increasing"):
    compute_recovery_metrics(trace)


def test_backwards_timestamps_rejected():
  n = 10
  ts = _timestamps(n)
  ts[5] = ts[2]  # goes backward
  trace = _make_trace(n)
  trace.timestamps = ts
  with pytest.raises(ValueError, match="strictly increasing"):
    compute_recovery_metrics(trace)


def test_shape_mismatch_rejected():
  n = 10
  lin = np.zeros((n + 1, 3))
  trace = _make_trace(n, root_lin_vel_w=lin)
  with pytest.raises(ValueError, match="root_lin_vel_w"):
    compute_recovery_metrics(trace)


def test_bad_group_label_rejected():
  trace = _make_trace(50, group="bogus")
  with pytest.raises(ValueError, match="group"):
    compute_recovery_metrics(trace)


def test_bad_termination_label_rejected():
  trace = _make_trace(50, termination="bogus")  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="termination"):
    compute_recovery_metrics(trace)


def test_nonfinite_entry_yaw_rejected():
  trace = _make_trace(50, entry_yaw=float("nan"))
  with pytest.raises(ValueError, match="entry_yaw"):
    compute_recovery_metrics(trace)


def test_negative_early_window_rejected():
  trace = _make_trace(50)
  with pytest.raises(ValueError, match="early_window_s"):
    compute_recovery_metrics(trace, early_window_s=-1.0)


# --------------------------------------------------------------------------- #
# Torque diagnostics only when explicitly available.
# --------------------------------------------------------------------------- #


def test_torque_ratio_none_when_unavailable():
  trace = _calm_trace(50)
  m = compute_recovery_metrics(trace)
  assert m.torque_requested_vs_applied_ratio is None


def test_torque_ratio_computed_when_available():
  n = 50
  applied = np.full((n, 4), 2.0)
  requested = np.full((n, 4), 1.0)
  trace = _make_trace(n, applied_torque=applied, requested_torque=requested)
  m = compute_recovery_metrics(trace)
  assert m.torque_requested_vs_applied_ratio == pytest.approx(2.0)


def test_torque_ratio_handles_zero_requested():
  n = 50
  applied = np.full((n, 4), 2.0)
  requested = np.zeros((n, 4))
  requested[0] = 1.0
  trace = _make_trace(n, applied_torque=applied, requested_torque=requested)
  m = compute_recovery_metrics(trace)
  assert m.torque_requested_vs_applied_ratio == pytest.approx(2.0)


def test_torque_ratio_none_on_shape_mismatch():
  n = 50
  applied = np.full((n, 4), 2.0)
  requested = np.full((n, 3), 1.0)
  trace = _make_trace(n, applied_torque=applied, requested_torque=requested)
  m = compute_recovery_metrics(trace)
  assert m.torque_requested_vs_applied_ratio is None


# --------------------------------------------------------------------------- #
# SettlingThresholds validation.
# --------------------------------------------------------------------------- #


def test_settling_thresholds_reject_nonpositive():
  with pytest.raises(ValueError, match="speed_xy"):
    SettlingThresholds(speed_xy=0.0)


def test_settling_thresholds_reject_nonfinite():
  with pytest.raises(ValueError, match="sustained_s"):
    SettlingThresholds(sustained_s=float("nan"))


def test_settling_thresholds_reject_bad_max_gap():
  with pytest.raises(ValueError, match="max_sample_gap_s"):
    SettlingThresholds(max_sample_gap_s=-1.0)


def test_settling_thresholds_accept_explicit_max_gap():
  t = SettlingThresholds(max_sample_gap_s=0.04)
  assert t.max_sample_gap_s == 0.04


# --------------------------------------------------------------------------- #
# Coverage / split semantics and no favorable aggregate.
# --------------------------------------------------------------------------- #


def test_aggregate_splits_by_group():
  metrics = [
    _make_metrics(trajectory_id=0, group="recovery"),
    _make_metrics(
      trajectory_id=1,
      group="retention",
      settled=False,
      time_to_settle=None,
      contributes_settled=False,
      not_settled=False,
      mean_command_lin_error_mps=0.1,
      mean_command_yaw_error_radps=0.05,
    ),
  ]
  stats = aggregate_recovery_metrics(metrics)
  assert len(stats) == 2
  groups = {s.group for s in stats}
  assert groups == {"recovery", "retention"}


def test_no_favorable_aggregate_from_failed_episodes():
  metrics = [
    _make_metrics(
      trajectory_id=i,
      group="recovery",
      termination="fall",
      settled=False,
      time_to_settle=None,
      earliest_calm_window_s=0.0,
      contributes_settled=False,
      not_settled=False,
      peak_tilt_deg=90.0,
    )
    for i in range(5)
  ]
  stats = aggregate_recovery_metrics(metrics)
  assert len(stats) == 1
  s = stats[0]
  assert s.settle_rate == 0.0
  assert s.num_settled == 0
  assert s.num_falls == 5
  assert s.median_time_to_settle_s is None
  assert s.mean_time_to_settle_s is None


def test_no_favorable_aggregate_from_censored_episodes():
  metrics = [
    _make_metrics(
      trajectory_id=i,
      group="recovery",
      termination="timeout",
      settled=False,
      time_to_settle=None,
      earliest_calm_window_s=None,
      censored=True,
      contributes_settled=False,
    )
    for i in range(3)
  ]
  stats = aggregate_recovery_metrics(metrics)
  s = stats[0]
  assert s.settle_rate == 0.0
  assert s.num_censored == 3
  assert s.median_time_to_settle_s is None


def test_coverage_excludes_invalid():
  metrics = [
    _make_metrics(trajectory_id=0, group="recovery"),
    _make_metrics(
      trajectory_id=1,
      group="recovery",
      termination="invalid",
      settled=False,
      time_to_settle=None,
      contributes_settled=False,
    ),
  ]
  stats = aggregate_recovery_metrics(metrics)
  s = stats[0]
  assert s.num_trajectories == 2
  assert s.num_invalid == 1
  assert s.coverage == pytest.approx(0.5)


def test_settle_rate_counts_only_settled():
  metrics = [
    _make_metrics(
      trajectory_id=i,
      group="recovery",
      termination="timeout" if i % 2 == 0 else "fall",
      settled=(i % 2 == 0),
      time_to_settle=1.0 if i % 2 == 0 else None,
      contributes_settled=(i % 2 == 0),
    )
    for i in range(4)
  ]
  stats = aggregate_recovery_metrics(metrics)
  s = stats[0]
  assert s.num_settled == 2
  assert s.settle_rate == pytest.approx(0.5)
  assert s.median_time_to_settle_s == pytest.approx(1.0)


def test_aggregate_unavailable_means_are_none_not_nan():
  # All invalid: means must be None, not NaN (strict-JSON-safe).
  metrics = [
    _make_metrics(
      trajectory_id=i,
      group="recovery",
      termination="invalid",
      settled=False,
      time_to_settle=None,
      contributes_settled=False,
    )
    for i in range(3)
  ]
  stats = aggregate_recovery_metrics(metrics)
  s = stats[0]
  assert s.mean_peak_tilt_deg is None
  assert s.mean_signed_backward_displacement_m is None
  assert s.mean_total_planar_travel_m is None


# --------------------------------------------------------------------------- #
# Retention trace.
# --------------------------------------------------------------------------- #


def test_retention_trace_group_always_retention():
  trace = RetentionTrace(
    timestamps=_timestamps(50),
    root_pos_w=np.zeros((50, 3)),
    root_quat_w=_unit_quat(50),
    root_lin_vel_w=np.zeros((50, 3)),
    root_ang_vel_w=np.zeros((50, 3)),
    torso_tilt_deg=np.zeros(50),
    entry_yaw=0.0,
    trajectory_id=0,
    termination="timeout",
  )
  assert trace.group == "retention"
  m = compute_recovery_metrics(trace)
  assert m.group == "retention"


# --------------------------------------------------------------------------- #
# Deterministic bounded schedules.
# --------------------------------------------------------------------------- #


def test_schedule_deterministic_same_seed():
  s = EvaluationSchedule(num_episodes=10, episode_length=100, duration_s=2.0, fps=50.0)
  idx1 = s.sample_indices(100)
  idx2 = s.sample_indices(100)
  np.testing.assert_array_equal(idx1, idx2)
  np.testing.assert_array_equal(s.episode_seeds(), s.episode_seeds())
  np.testing.assert_array_equal(s.group_assignment(), s.group_assignment())


def test_schedule_different_seed_different_indices():
  s1 = EvaluationSchedule(
    num_episodes=10, episode_length=100, duration_s=2.0, fps=50.0, seed=1
  )
  s2 = EvaluationSchedule(
    num_episodes=10, episode_length=100, duration_s=2.0, fps=50.0, seed=2
  )
  assert not np.array_equal(s1.sample_indices(100), s2.sample_indices(100))


def test_schedule_identical_for_source_and_candidate():
  s = EvaluationSchedule(num_episodes=8, episode_length=100, duration_s=2.0, fps=50.0)
  src_idx = s.sample_indices(50)
  cand_idx = s.sample_indices(50)
  np.testing.assert_array_equal(src_idx, cand_idx)
  np.testing.assert_array_equal(s.episode_seeds(), s.episode_seeds())


def test_schedule_group_assignment_80_20():
  s = EvaluationSchedule(
    num_episodes=10,
    episode_length=100,
    duration_s=2.0,
    fps=50.0,
    recovery_fraction=0.8,
  )
  groups = s.group_assignment()
  assert (groups == "recovery").sum() == 8
  assert (groups == "retention").sum() == 2


def test_schedule_recovery_commands_zero():
  s = EvaluationSchedule(num_episodes=10, episode_length=100, duration_s=2.0, fps=50.0)
  cmds = s.recovery_commands()
  assert cmds.shape == (8, 3)
  assert np.all(cmds == 0.0)


def test_schedule_retention_commands_use_distribution():
  s = EvaluationSchedule(num_episodes=10, episode_length=100, duration_s=2.0, fps=50.0)

  def dist(rng, n):
    return rng.uniform(-1, 1, size=(n, 3))

  cmds = s.retention_commands(dist)
  assert cmds.shape == (2, 3)
  assert np.all(np.isfinite(cmds))


def test_schedule_retention_commands_reject_bad_shape():
  s = EvaluationSchedule(num_episodes=10, episode_length=100, duration_s=2.0, fps=50.0)

  def bad_dist(rng, n):
    return rng.uniform(-1, 1, size=(n, 2))  # wrong last dim

  with pytest.raises(ValueError, match="distribution_fn"):
    s.retention_commands(bad_dist)


def test_schedule_retention_commands_deterministic():
  s = EvaluationSchedule(num_episodes=10, episode_length=100, duration_s=2.0, fps=50.0)

  def dist(rng, n):
    return rng.uniform(-1, 1, size=(n, 3))

  c1 = s.retention_commands(dist)
  c2 = s.retention_commands(dist)
  np.testing.assert_array_equal(c1, c2)


def test_schedule_episode_timestamps():
  s = EvaluationSchedule(num_episodes=5, episode_length=100, duration_s=2.0, fps=50.0)
  ts = s.episode_timestamps()
  assert ts.shape == (100,)
  assert ts[0] == 0.0
  assert ts[-1] == pytest.approx(2.0 - DT, abs=1e-9)


def test_schedule_bounds_reject_nonpositive_episodes():
  with pytest.raises(ValueError, match="num_episodes"):
    EvaluationSchedule(num_episodes=0, episode_length=100, duration_s=2.0, fps=50.0)


def test_schedule_bounds_reject_nonpositive_fps():
  with pytest.raises(ValueError, match="fps"):
    EvaluationSchedule(num_episodes=5, episode_length=100, duration_s=2.0, fps=0.0)


def test_schedule_bounds_reject_episode_length_exceeds_duration():
  with pytest.raises(ValueError, match="exceeds"):
    EvaluationSchedule(num_episodes=5, episode_length=200, duration_s=2.0, fps=50.0)


def test_schedule_bounds_reject_bad_split():
  with pytest.raises(ValueError, match="split"):
    EvaluationSchedule(
      num_episodes=5, episode_length=100, duration_s=2.0, fps=50.0, split="bogus"
    )


def test_schedule_bounds_reject_bad_recovery_fraction():
  with pytest.raises(ValueError, match="recovery_fraction"):
    EvaluationSchedule(
      num_episodes=5,
      episode_length=100,
      duration_s=2.0,
      fps=50.0,
      recovery_fraction=1.5,
    )


def test_schedule_sample_indices_pool_too_small_uses_replacement():
  s = EvaluationSchedule(num_episodes=10, episode_length=100, duration_s=2.0, fps=50.0)
  idx = s.sample_indices(3)
  assert idx.shape == (10,)
  assert np.all(idx < 3)


def test_schedule_sample_indices_rejects_nonpositive_pool():
  s = EvaluationSchedule(num_episodes=10, episode_length=100, duration_s=2.0, fps=50.0)
  with pytest.raises(ValueError, match="pool_size"):
    s.sample_indices(0)


# --------------------------------------------------------------------------- #
# CLI args / build_evaluation_schedule.
# --------------------------------------------------------------------------- #


def test_build_evaluation_schedule_from_cli_args():
  args = EvaluationCliArgs(num_episodes=12, episode_length=500, duration_s=10.0)
  s = build_evaluation_schedule(args)
  assert isinstance(s, EvaluationSchedule)
  assert s.num_episodes == 12
  assert s.episode_length == 500
  assert s.duration_s == 10.0
  assert s.split == "validation"
  assert s.recovery_fraction == 0.8


def test_cli_args_defaults_match_plan():
  args = EvaluationCliArgs()
  assert args.duration_s == 20.0
  assert args.fps == 50.0
  assert args.episode_length == 1000
  assert args.early_window_s == 3.0
  assert args.settling_speed_xy == 0.1
  assert args.settling_yaw_rate == 0.2
  assert args.settling_torso_tilt_deg == 10.0
  assert args.settling_sustained_s == 0.5


def test_format_group_summary_handles_no_settled():
  s = RecoveryGroupStats(
    group="recovery",
    num_trajectories=3,
    num_settled=0,
    num_falls=2,
    num_orientation_terminations=0,
    num_timeouts=0,
    num_invalid=1,
    num_not_settled=0,
    num_censored=0,
    settle_rate=0.0,
    median_time_to_settle_s=None,
    mean_time_to_settle_s=None,
    mean_peak_tilt_deg=90.0,
    mean_signed_backward_displacement_m=1.0,
    mean_total_planar_travel_m=5.0,
    coverage=2 / 3,
    mean_command_lin_error_mps=None,
    mean_command_yaw_error_radps=None,
    num_command_tracked=0,
  )
  out = format_group_summary([s])
  assert "recovery" in out
  assert "n/a" in out


def test_format_group_summary_shows_retention_tracking():
  s = RecoveryGroupStats(
    group="retention",
    num_trajectories=2,
    num_settled=0,
    num_falls=0,
    num_orientation_terminations=0,
    num_timeouts=2,
    num_invalid=0,
    num_not_settled=0,
    num_censored=0,
    settle_rate=0.0,
    median_time_to_settle_s=None,
    mean_time_to_settle_s=None,
    mean_peak_tilt_deg=5.0,
    mean_signed_backward_displacement_m=0.0,
    mean_total_planar_travel_m=2.0,
    coverage=1.0,
    mean_command_lin_error_mps=0.1,
    mean_command_yaw_error_radps=0.05,
    num_command_tracked=2,
  )
  out = format_group_summary([s])
  assert "cmd_lin_err=0.100m/s" in out
  assert "cmd_yaw_err=0.050rad/s" in out


# --------------------------------------------------------------------------- #
# End-to-end synthetic: compute + aggregate.
# --------------------------------------------------------------------------- #


def test_compute_then_aggregate_end_to_end():
  # 3 recovery traces: 1 settles, 1 falls, 1 not-settled.
  settled = _calm_trace(100, trajectory_id=0)
  settled.root_lin_vel_w[:50] = 0.5  # unsettled first 1s

  fall = _calm_trace(100, trajectory_id=1, termination="fall")
  fall.torso_tilt_deg[:] = 90.0

  not_settled = _calm_trace(100, trajectory_id=2)
  not_settled.root_lin_vel_w[:] = 0.5

  metrics = [
    compute_recovery_metrics(settled),
    compute_recovery_metrics(fall),
    compute_recovery_metrics(not_settled),
  ]
  assert metrics[0].settled
  assert metrics[1].termination == "fall"
  assert metrics[2].termination == "not_settled"

  stats = aggregate_recovery_metrics(metrics)
  assert len(stats) == 1
  s = stats[0]
  assert s.num_settled == 1
  assert s.settle_rate == pytest.approx(1 / 3)
  assert s.num_falls == 1
  assert s.num_not_settled == 1
  assert s.median_time_to_settle_s == pytest.approx(1.0)
