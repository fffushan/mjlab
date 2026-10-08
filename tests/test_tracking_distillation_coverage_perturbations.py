import numpy as np
import pytest

from mjlab.tasks.tracking.distillation.coverage_perturbations import (
  CoveragePerturbationError,
  TriggerState,
  _joints,
  _root_draw,
  apply_world_orientation_delta,
  difficulty_band,
  sample_reset_perturbation,
  sample_runtime_perturbation,
  schedule_phase_event,
  select_bounded_candidates,
  trigger_candidate,
  trigger_severity,
)
from mjlab.tasks.tracking.distillation.coverage_policy import CoveragePolicy


def test_reset_and_runtime_directions_are_bounded_and_world_frame() -> None:
  policy = CoveragePolicy.difficult_state_coverage()
  rng = np.random.default_rng(7)
  reset = sample_reset_perturbation(policy, rng)
  root = (
    reset.position_delta_m
    + reset.rotation_delta_rad
    + reset.linear_velocity_delta_m_s
    + reset.angular_velocity_delta_rad_s
  )
  assert all(np.isfinite(root))
  runtime = sample_runtime_perturbation(policy, rng)
  assert runtime.linear_velocity_delta_m_s[2] == 0.0
  assert (
    runtime.angular_velocity_delta_rad_s[2] == 0.0 or runtime.root_direction == "yaw"
  )
  assert all(np.isfinite(runtime.linear_velocity_delta_m_s))


def test_off_perturbations_are_noops_and_runtime_event_is_bounded() -> None:
  policy = CoveragePolicy.off()
  rng = np.random.default_rng(1)
  assert sample_reset_perturbation(policy, rng).is_noop
  assert sample_runtime_perturbation(policy, rng).is_noop
  assert (
    schedule_phase_event(
      policy,
      rng,
      current_reference_frame=0,
      initial_reference_frame=0,
      clip_frames=50,
      segment_age_steps=0,
    )
    is None
  )
  policy = CoveragePolicy.difficult_state_coverage()
  event = schedule_phase_event(
    policy,
    np.random.default_rng(2),
    current_reference_frame=0,
    initial_reference_frame=0,
    clip_frames=50,
    segment_age_steps=0,
  )
  assert event is not None
  assert 10 <= event.frame <= 44
  with pytest.raises(CoveragePerturbationError):
    schedule_phase_event(
      policy,
      np.random.default_rng(2),
      current_reference_frame=4,
      initial_reference_frame=5,
      clip_frames=50,
      segment_age_steps=0,
    )


def test_selected_reset_axes_do_not_expand_inactive_axes() -> None:
  policy = CoveragePolicy.difficult_state_coverage()
  rng = np.random.default_rng(11)
  envelopes = dict(policy.root_reset_envelopes)
  for _ in range(200):
    draw = sample_reset_perturbation(policy, rng)
    base = envelopes["baseline"]
    if draw.root_direction == "sagittal" or draw.excitation == "joint_chain_only":
      assert abs(draw.position_delta_m[1]) <= dict(base.position_m)["y"]
      assert (
        abs(draw.linear_velocity_delta_m_s[1]) <= dict(base.linear_velocity_m_s)["y"]
      )
    if draw.root_direction == "lateral" or draw.excitation == "joint_chain_only":
      assert abs(draw.position_delta_m[0]) <= dict(base.position_m)["x"]
      assert (
        abs(draw.linear_velocity_delta_m_s[0]) <= dict(base.linear_velocity_m_s)["x"]
      )
    assert abs(draw.position_delta_m[2]) <= dict(base.position_m)["z"]


def test_world_quaternion_composition_and_rejection() -> None:
  mixed = (0.2, -0.3, 0.4)
  qx = np.array((np.cos(mixed[0] / 2), np.sin(mixed[0] / 2), 0.0, 0.0))
  qy = np.array((np.cos(mixed[1] / 2), 0.0, np.sin(mixed[1] / 2), 0.0))
  qz = np.array((np.cos(mixed[2] / 2), 0.0, 0.0, np.sin(mixed[2] / 2)))
  from mjlab.tasks.tracking.distillation.coverage_perturbations import (
    quaternion_multiply,
  )

  assert np.allclose(
    apply_world_orientation_delta((1.0, 0.0, 0.0, 0.0), mixed),
    quaternion_multiply(quaternion_multiply(qz, qy), qx),
  )
  result = apply_world_orientation_delta((1.0, 0.0, 0.0, 0.0), (np.pi / 2, 0.0, 0.0))
  assert np.allclose(result, (np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0))
  with pytest.raises(CoveragePerturbationError):
    apply_world_orientation_delta((float("nan"), 0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def test_trigger_score_bands_and_cooldown() -> None:
  policy = CoveragePolicy.difficult_state_coverage()
  threshold = dict(policy.triggers.thresholds)["root_angular_velocity_error_rad_s"]
  assert trigger_severity(
    {"root_angular_velocity_error_rad_s": threshold}, policy
  ) == pytest.approx(1.0)
  assert difficulty_band(0.9) == "nominal"
  assert difficulty_band(1.0) == "moderate"
  assert difficulty_band(2.0) == "strong"
  state = TriggerState()
  emitted, _, reason = trigger_candidate(
    {"root_angular_velocity_error_rad_s": 2.0}, policy, state
  )
  assert emitted and reason == "threshold_crossing"
  emitted, _, _ = trigger_candidate(
    {"root_angular_velocity_error_rad_s": 2.0}, policy, state
  )
  assert not emitted
  with pytest.raises(CoveragePerturbationError):
    trigger_severity({"teacher_student_disagreement": 1.0}, policy)


class _MaxRng:
  def __init__(self) -> None:
    self.uniform_calls = 0

  def uniform(self, low: float, high: float) -> float:
    del low
    self.uniform_calls += 1
    return high

  def integers(self, low: int, high: int) -> int:
    del low, high
    return 0


class _AlternatingRng(_MaxRng):
  def uniform(self, low: float, high: float) -> float:
    value = high if self.uniform_calls % 2 == 0 else low
    self.uniform_calls += 1
    return value


def test_reset_envelopes_are_single_draw_and_directional() -> None:
  policy = CoveragePolicy.difficult_state_coverage()
  rng = _MaxRng()
  position, rotation, linear, angular = _root_draw(
    policy, rng, "strong", "horizontal_translation", runtime=False
  )
  assert rng.uniform_calls == 12
  assert position == pytest.approx((0.075, 0.025, 0.005))
  assert rotation == pytest.approx((0.05, 0.05, 0.10))
  assert linear == pytest.approx((1.0, 0.25, 0.10))
  assert angular == pytest.approx((0.26, 0.26, 0.39))
  rng = _MaxRng()
  position, rotation, linear, angular = _root_draw(
    policy, rng, "strong", "sagittal", runtime=False
  )
  assert rng.uniform_calls == 12
  assert position == pytest.approx((0.025, 0.025, 0.005))
  assert rotation == pytest.approx((0.05, 0.30, 0.10))
  assert linear == pytest.approx((0.25, 0.25, 0.10))
  assert angular == pytest.approx((0.26, 1.50, 0.39))


def test_reset_joint_velocity_deltas_use_selected_chain_envelopes() -> None:
  policy = CoveragePolicy.difficult_state_coverage()
  max_rng = _MaxRng()
  position, velocity = _joints(policy, max_rng, "strong", "waist", runtime=False)
  assert max_rng.uniform_calls == 6
  assert dict(position) == pytest.approx(
    {"waist_yaw_joint": 0.10, "waist_pitch_joint": 0.15, "waist_roll_joint": 0.15}
  )
  assert dict(velocity) == pytest.approx(
    {"waist_yaw_joint": 1.0, "waist_pitch_joint": 1.0, "waist_roll_joint": 1.0}
  )
  alternating = _AlternatingRng()
  position, velocity = _joints(policy, alternating, "strong", "waist", runtime=False)
  assert alternating.uniform_calls == 6
  assert all(value > 0 for _, value in position)
  assert all(value < 0 for _, value in velocity)
  runtime_rng = _MaxRng()
  runtime_position, runtime_velocity = _joints(
    policy, runtime_rng, "strong", "waist", runtime=True
  )
  assert runtime_rng.uniform_calls == 3
  assert all(value == 0.0 for _, value in runtime_position)
  assert all(value > 0 for _, value in runtime_velocity)


def test_trigger_cap_and_disabled_path_are_side_effect_free() -> None:
  policy = CoveragePolicy.difficult_state_coverage()
  state = TriggerState(armed=False, cooldown=1, pending=True, threshold_candidates=2)
  emitted, _, reason = trigger_candidate(
    {"root_angular_velocity_error_rad_s": 2.0}, policy, state
  )
  assert not emitted and reason == "candidate_cap"
  assert not state.pending
  disabled = CoveragePolicy.off()
  off_state = TriggerState(
    armed=False, cooldown=1, pending=True, threshold_candidates=1
  )
  assert trigger_candidate(
    {"root_angular_velocity_error_rad_s": 2.0}, disabled, off_state
  ) == (False, 0.0, "disabled")
  assert (
    off_state.armed,
    off_state.cooldown,
    off_state.pending,
    off_state.threshold_candidates,
  ) == (False, 1, True, 1)


def test_disarmed_high_trigger_does_not_repeatedly_emit() -> None:
  policy = CoveragePolicy.difficult_state_coverage()
  state = TriggerState(armed=False)
  assert not trigger_candidate(
    {"root_angular_velocity_error_rad_s": 2.0}, policy, state
  )[0]
  assert not trigger_candidate(
    {"root_angular_velocity_error_rad_s": 2.0}, policy, state
  )[0]
  assert state.threshold_candidates == 0


def test_continuous_high_requires_rearm_before_second_candidate() -> None:
  policy = CoveragePolicy.difficult_state_coverage()
  state = TriggerState()
  high = {"root_angular_velocity_error_rad_s": 2.0}
  emissions = [trigger_candidate(high, policy, state)[0] for _ in range(60)]
  assert sum(emissions) == 1
  assert not state.pending
  assert state.threshold_candidates == 1


def test_rearmed_crossing_during_cooldown_is_deferred_once() -> None:
  policy = CoveragePolicy.difficult_state_coverage()
  state = TriggerState()
  high = {"root_angular_velocity_error_rad_s": 2.0}
  assert trigger_candidate(high, policy, state)[0]
  for _ in range(5):
    assert not trigger_candidate({}, policy, state)[0]
  assert state.armed
  assert not trigger_candidate(high, policy, state)[0]
  assert state.pending and not state.armed
  # Tick 25 emits the second candidate; continuous high cannot add a third.
  emissions = [trigger_candidate(high, policy, state)[0] for _ in range(50)]
  assert sum(emissions) == 1
  assert state.threshold_candidates == 2
  assert not state.pending


def test_stratified_reservoir_preserves_strata_and_is_reproducible() -> None:
  candidates = [("a", index) for index in range(4)] + [("b", 0)]
  first = select_bounded_candidates(
    candidates, 3, np.random.default_rng(4), key=lambda item: item[0]
  )
  second = select_bounded_candidates(
    candidates, 3, np.random.default_rng(4), key=lambda item: item[0]
  )
  assert first == second
  assert {item[0] for item in first} == {"a", "b"}
  many = [(name, 0) for name in ("a", "b", "c")]
  selected = select_bounded_candidates(
    many, 2, np.random.default_rng(2), key=lambda item: item[0]
  )
  assert len({item[0] for item in selected}) == 2
  with pytest.raises(CoveragePerturbationError):
    select_bounded_candidates(candidates, -1, np.random.default_rng(0))
  with pytest.raises(CoveragePerturbationError):
    select_bounded_candidates(candidates, 2, np.random.default_rng(0), key=lambda _: [])
