"""Configuration contracts for X2 correlated-DR tracking ablations."""

from copy import deepcopy

import pytest

from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.config.agibot_x2.env_cfgs import (
  agibot_x2_flat_tracking_correlated_dr_env_cfg,
  agibot_x2_flat_tracking_env_cfg,
)
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import VELOCITY_RANGE

CORRELATED_TASK = "Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR"
REDUCED_TASK = f"{CORRELATED_TASK}-Reduced-Perturbations"


@pytest.fixture(autouse=True)
def clear_axis_selection(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.delenv("MJLAB_DR_AXES", raising=False)


def motion_cfg(cfg) -> MotionCommandCfg:
  motion = cfg.commands["motion"]
  assert isinstance(motion, MotionCommandCfg)
  return motion


def test_registered_tasks_have_isolated_experiment_directories() -> None:
  assert CORRELATED_TASK in list_tasks()
  assert REDUCED_TASK in list_tasks()
  assert (
    load_rl_cfg(CORRELATED_TASK).experiment_name == "agibot_x2_tracking_correlated_dr"
  )
  assert (
    load_rl_cfg(REDUCED_TASK).experiment_name
    == "agibot_x2_tracking_correlated_dr_reduced_perturbations"
  )


def test_correlated_cfg_only_changes_the_requested_nse_dr_knobs() -> None:
  baseline = agibot_x2_flat_tracking_env_cfg(has_state_estimation=False)
  correlated = agibot_x2_flat_tracking_correlated_dr_env_cfg()

  assert list(correlated.observations["actor"].terms) == list(
    baseline.observations["actor"].terms
  )
  assert correlated.observations["critic"] == baseline.observations["critic"]
  assert correlated.actions == baseline.actions
  assert correlated.rewards == baseline.rewards
  assert correlated.terminations == baseline.terminations
  assert correlated.commands["motion"] == baseline.commands["motion"]

  pd_params = correlated.events["randomize_pd_gains"].params
  assert pd_params["shared_gain_scale"] is True
  assert pd_params["kp_range"] == pd_params["kd_range"] == (0.7, 1.3)

  actor = correlated.observations["actor"].terms
  assert actor["joint_pos"].delay_group == "encoder_packet"
  assert actor["joint_vel"].delay_group == "encoder_packet"
  for name in ("joint_pos", "joint_vel", "base_ang_vel"):
    assert actor[name].delay_hold_prob == 0.9
    assert actor[name].delay_update_period == 1
    assert (actor[name].delay_min_lag, actor[name].delay_max_lag) == (0, 1)
  assert actor["base_ang_vel"].delay_group is None

  baseline_pd = baseline.events["randomize_pd_gains"].params
  assert "shared_gain_scale" not in baseline_pd
  assert baseline.observations["actor"].terms["joint_pos"].delay_group is None


def test_reduced_variant_changes_only_reset_and_push_perturbations() -> None:
  correlated = agibot_x2_flat_tracking_correlated_dr_env_cfg()
  reduced = agibot_x2_flat_tracking_correlated_dr_env_cfg(reduced_perturbations=True)
  correlated_motion = motion_cfg(correlated)
  reduced_motion = motion_cfg(reduced)

  assert reduced.rewards == correlated.rewards
  assert reduced.terminations == correlated.terminations
  assert reduced.actions == correlated.actions
  assert reduced.observations == correlated.observations
  assert set(reduced.events) == set(correlated.events)
  for name in set(correlated.events) - {"push_robot"}:
    assert reduced.events[name] == correlated.events[name]

  for axis, bounds in correlated_motion.pose_range.items():
    assert reduced_motion.pose_range[axis] == tuple(bound * 0.5 for bound in bounds)
  for axis, bounds in correlated_motion.velocity_range.items():
    assert reduced_motion.velocity_range[axis] == tuple(bound * 0.5 for bound in bounds)
  assert reduced_motion.joint_position_range == tuple(
    bound * 0.5 for bound in correlated_motion.joint_position_range
  )
  assert reduced.events["push_robot"].interval_range_s == (4.0, 8.0)
  for axis, bounds in correlated.events["push_robot"].params["velocity_range"].items():
    assert reduced.events["push_robot"].params["velocity_range"][axis] == tuple(
      bound * 0.5 for bound in bounds
    )


def test_variant_build_order_does_not_mutate_baseline_or_global_velocity_range() -> (
  None
):
  velocity_range = deepcopy(VELOCITY_RANGE)
  baseline_before = agibot_x2_flat_tracking_env_cfg(has_state_estimation=False)
  _ = agibot_x2_flat_tracking_correlated_dr_env_cfg(reduced_perturbations=True)
  _ = agibot_x2_flat_tracking_correlated_dr_env_cfg()
  baseline_after = agibot_x2_flat_tracking_env_cfg(has_state_estimation=False)

  assert VELOCITY_RANGE == velocity_range
  assert baseline_after.commands["motion"] == baseline_before.commands["motion"]
  assert baseline_after.events == baseline_before.events
  assert baseline_after.events["push_robot"].params["velocity_range"] == VELOCITY_RANGE


@pytest.mark.parametrize("axes", ["none", "pd_gains", "obs_delay"])
def test_axis_selection_never_recreates_disabled_correlated_terms(
  axes: str, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv("MJLAB_DR_AXES", axes)
  cfg = agibot_x2_flat_tracking_correlated_dr_env_cfg()
  actor = cfg.observations["actor"].terms

  if axes == "pd_gains":
    assert cfg.events["randomize_pd_gains"].params["shared_gain_scale"] is True
  else:
    assert "randomize_pd_gains" not in cfg.events

  if axes == "obs_delay":
    assert actor["joint_pos"].delay_group == "encoder_packet"
    assert actor["joint_vel"].delay_group == "encoder_packet"
  else:
    for name in ("joint_pos", "joint_vel", "base_ang_vel"):
      assert (actor[name].delay_min_lag, actor[name].delay_max_lag) == (0, 0)
      assert actor[name].delay_group is None
      assert actor[name].delay_hold_prob == 0.0
      assert actor[name].delay_update_period == 0


def test_play_preserves_standard_overrides_and_reduced_joint_reset_range() -> None:
  for task_id, expected_joint_range in (
    (CORRELATED_TASK, (-0.1, 0.1)),
    (REDUCED_TASK, (-0.05, 0.05)),
  ):
    cfg = load_env_cfg(task_id, play=True)
    motion = motion_cfg(cfg)

    assert cfg.observations["actor"].enable_corruption is False
    assert "push_robot" not in cfg.events
    assert motion.sampling_mode == "start"
    assert motion.pose_range == {}
    assert motion.velocity_range == {}
    assert motion.joint_position_range == expected_joint_range
