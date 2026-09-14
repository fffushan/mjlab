"""Tests for the sim-to-real randomization wired into the tracking tasks.

These lock in the wiring rather than the numbers: a randomization axis that
silently stops being applied (or silently loses its actuator/geom selection) is
the failure mode worth catching, since it costs a training run to notice
otherwise.
"""

import math

import pytest

from mjlab.asset_zoo.robots.agibot_x2.x2_constants import X2_ARTICULATION
from mjlab.tasks.tracking.config.agibot_x2.env_cfgs import (
  agibot_x2_flat_tracking_env_cfg,
)
from mjlab.tasks.tracking.tracking_env_cfg import DR_AXES, selected_dr_axes

PRE_EXISTING_EVENTS = ["push_robot", "base_com", "encoder_bias", "foot_friction"]
FOOT_GEOMS = r"^(left|right)_foot[0-9]+_collision$"
WRIST_ACTUATOR_ID = 4

# Axis -> the event it wires (``obs_delay`` configures observation terms instead).
AXIS_EVENTS = {
  "inertia": "randomize_inertia",
  "armature": "randomize_armature",
  "effort_limits": "randomize_effort_limits",
  "joint_friction": "randomize_joint_friction",
  "joint_damping": "randomize_joint_damping",
  "foot_size": "randomize_foot_size",
  "pd_gains": "randomize_pd_gains",
  "obs_delay": None,
}


@pytest.fixture(autouse=True)
def clear_axis_selection(monkeypatch: pytest.MonkeyPatch) -> None:
  """Every test starts from the default axis selection."""
  monkeypatch.delenv("MJLAB_DR_AXES", raising=False)


def x2_cfg(has_state_estimation: bool = False):
  return agibot_x2_flat_tracking_env_cfg(has_state_estimation=has_state_estimation)


def test_all_axes_enabled_by_default() -> None:
  events = list(x2_cfg().events)

  for event in PRE_EXISTING_EVENTS:
    assert event in events
  for axis, event in AXIS_EVENTS.items():
    if event is not None:
      assert event in events, f"missing {event} for axis {axis}"
  assert "randomize_effort_limits_wrist" in events


def test_pre_existing_events_keep_their_order() -> None:
  events = list(x2_cfg().events)

  assert sorted(PRE_EXISTING_EVENTS, key=events.index) == PRE_EXISTING_EVENTS


@pytest.mark.parametrize("axis", DR_AXES)
def test_each_axis_can_be_selected_alone(
  axis: str, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv("MJLAB_DR_AXES", axis)

  assert selected_dr_axes() == {axis}
  cfg = x2_cfg()
  events = list(cfg.events)
  # The pre-existing randomization is never switched off by the axis knob.
  for event in PRE_EXISTING_EVENTS:
    assert event in events
  # ... and no other axis leaks in.
  for other_axis, event in AXIS_EVENTS.items():
    if event is None:
      continue
    assert (event in events) == (other_axis == axis), f"{event} for axis {other_axis}"
  if axis == "obs_delay":
    assert cfg.observations["actor"].terms["joint_vel"].delay_max_lag == 1
  else:
    assert cfg.observations["actor"].terms["joint_vel"].delay_max_lag == 0


def test_wrist_derating_follows_the_effort_axis(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("MJLAB_DR_AXES", "armature")
  assert "randomize_effort_limits_wrist" not in x2_cfg().events

  monkeypatch.setenv("MJLAB_DR_AXES", "effort_limits")
  assert "randomize_effort_limits_wrist" in x2_cfg().events


def test_no_axes_matches_the_pre_axis_behaviour(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("MJLAB_DR_AXES", "none")

  assert list(x2_cfg().events) == PRE_EXISTING_EVENTS


def test_unknown_axis_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("MJLAB_DR_AXES", "armature,typo")

  with pytest.raises(ValueError, match="unknown axes"):
    selected_dr_axes()


def test_inertia_precedes_the_torso_com_term() -> None:
  # Both write body_ipos and each samples from the compiled defaults, so the
  # torso keeps its own wider payload offset only if it comes second.
  events = list(x2_cfg().events)

  assert events.index("randomize_inertia") < events.index("base_com")


def test_wrist_effort_term_follows_the_whole_robot_term() -> None:
  cfg = x2_cfg()
  events = list(cfg.events)

  assert events.index("randomize_effort_limits_wrist") > events.index(
    "randomize_effort_limits"
  )
  wrist = cfg.events["randomize_effort_limits_wrist"].params
  assert wrist["asset_cfg"].actuator_ids == [WRIST_ACTUATOR_ID]
  assert wrist["effort_limit_range"] == (0.45, 1.0)
  # The index must still be the wrist group.
  group = X2_ARTICULATION.actuators[WRIST_ACTUATOR_ID]
  assert group.target_names_expr == (".*_wrist_pitch_joint", ".*_wrist_roll_joint")


def test_ranges_match_the_documented_ones() -> None:
  events = x2_cfg().events

  alpha = events["randomize_inertia"].params["alpha_range"]
  assert math.exp(2 * alpha[0]) == pytest.approx(0.90)
  assert math.exp(2 * alpha[1]) == pytest.approx(1.10)
  assert events["randomize_inertia"].params["t_range"] == (-0.02, 0.02)
  assert events["randomize_armature"].params["ranges"] == (0.7, 1.3)
  assert events["randomize_effort_limits"].params["effort_limit_range"] == (0.75, 1.0)
  assert events["randomize_joint_friction"].params["ranges"] == (0.5, 1.5)
  assert events["randomize_joint_damping"].params["ranges"] == (0.0, 0.3)
  assert events["randomize_joint_damping"].params["operation"] == "abs"
  assert events["randomize_foot_size"].params["ranges"] == (0.97, 1.03)
  assert events["randomize_foot_size"].params["asset_cfg"].geom_names == FOOT_GEOMS
  assert events["randomize_pd_gains"].params["kp_range"] == (0.7, 1.3)
  assert events["randomize_pd_gains"].params["kd_range"] == (0.7, 1.3)


def test_command_delay_on_every_x2_actuator_group() -> None:
  for group in X2_ARTICULATION.actuators:
    assert group.delay_min_lag == 0
    assert group.delay_max_lag == 2
    assert group.delay_hold_prob == 0.9


def test_observation_delay_covers_sensor_terms_only() -> None:
  terms = x2_cfg().observations["actor"].terms

  for name in ("joint_pos", "joint_vel", "base_ang_vel"):
    assert (terms[name].delay_min_lag, terms[name].delay_max_lag) == (0, 1)
  for name in ("command", "motion_lookahead", "motion_anchor_ori_b", "actions"):
    assert (terms[name].delay_min_lag, terms[name].delay_max_lag) == (0, 0)


def test_no_state_estimation_drops_exactly_the_estimator_terms() -> None:
  nse = x2_cfg(has_state_estimation=False).observations
  se = x2_cfg(has_state_estimation=True).observations

  assert set(se["actor"].terms) - set(nse["actor"].terms) == {
    "motion_anchor_pos_b",
    "base_lin_vel",
  }
  # The critic keeps the privileged state in both variants.
  assert list(nse["critic"].terms) == list(se["critic"].terms)
