"""Behavioral tests for shared observation delay schedules."""

from typing import cast
from unittest.mock import Mock

import pytest
import torch
from conftest import get_test_device

from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationManager,
  ObservationTermCfg,
)


def _mock_env(device, num_envs: int = 4):
  env = Mock()
  env.num_envs = num_envs
  env.device = device
  env.step_dt = 0.02
  return env


def _counting_term(offset: float, device):
  counter = {"value": 0}

  def observation(env):
    counter["value"] += 1
    return torch.full(
      (env.num_envs, 1), counter["value"] * 10.0 + offset, device=device
    )

  return observation, counter


def _delayed_term(func, **kwargs) -> ObservationTermCfg:
  return ObservationTermCfg(func=func, params={}, **kwargs)


def test_delay_group_synchronizes_nonadjacent_stochastic_term_histories():
  device = get_test_device()
  env = _mock_env(device)
  first, _ = _counting_term(1.0, device)
  middle, _ = _counting_term(50.0, device)
  last, _ = _counting_term(3.0, device)
  cfg = {
    "actor": ObservationGroupCfg(
      concatenate_terms=False,
      terms={
        "first": _delayed_term(
          first,
          delay_min_lag=0,
          delay_max_lag=3,
          delay_group="proprioception",
          history_length=3,
          flatten_history_dim=False,
        ),
        "middle": _delayed_term(
          middle,
          delay_min_lag=1,
          delay_max_lag=1,
          delay_group="middle",
        ),
        "last": _delayed_term(
          last,
          delay_min_lag=0,
          delay_max_lag=3,
          delay_group="proprioception",
          history_length=3,
          flatten_history_dim=False,
        ),
      },
    )
  }
  manager = ObservationManager(cfg, env)

  for _ in range(6):
    observations = cast(
      dict[str, torch.Tensor], manager.compute(update_history=True)["actor"]
    )
    first_history = observations["first"]
    last_history = observations["last"]
    assert torch.allclose(
      first_history - last_history, -2.0 * torch.ones_like(first_history)
    )
    delay_buffers = manager._group_obs_term_delay_buffer["actor"]
    assert torch.equal(
      delay_buffers["first"].current_lags, delay_buffers["last"].current_lags
    )


def test_same_delay_group_name_is_independent_across_observation_groups():
  device = get_test_device()
  env = _mock_env(device)
  actor_term, _ = _counting_term(1.0, device)
  critic_term, _ = _counting_term(2.0, device)
  cfg = {
    "actor": ObservationGroupCfg(
      terms={
        "obs": _delayed_term(
          actor_term,
          delay_min_lag=1,
          delay_max_lag=1,
          delay_group="shared-name",
        )
      }
    ),
    "critic": ObservationGroupCfg(
      terms={
        "obs": _delayed_term(
          critic_term,
          delay_min_lag=2,
          delay_max_lag=2,
          delay_group="shared-name",
        )
      }
    ),
  }
  manager = ObservationManager(cfg, env)
  manager.compute(update_history=True)

  assert torch.equal(
    manager._group_obs_delay_group_buffer["actor"]["shared-name"].current_lags,
    torch.ones(env.num_envs, dtype=torch.long, device=device),
  )
  assert torch.equal(
    manager._group_obs_delay_group_buffer["critic"]["shared-name"].current_lags,
    torch.full((env.num_envs,), 2, dtype=torch.long, device=device),
  )


def test_delay_group_partial_reset_backfills_only_reset_environments():
  device = get_test_device()
  env = _mock_env(device)
  first, _ = _counting_term(1.0, device)
  second, _ = _counting_term(2.0, device)
  cfg = {
    "actor": ObservationGroupCfg(
      terms={
        "first": _delayed_term(
          first,
          delay_min_lag=2,
          delay_max_lag=2,
          delay_group="shared",
          history_length=3,
        ),
        "second": _delayed_term(
          second,
          delay_min_lag=2,
          delay_max_lag=2,
          delay_group="shared",
          history_length=3,
        ),
      }
    )
  }
  manager = ObservationManager(cfg, env)
  for _ in range(4):
    manager.compute(update_history=True)

  reset_ids = torch.tensor([0, 2], device=device)
  untouched_ids = torch.tensor([1, 3], device=device)
  delay_buffer = manager._group_obs_term_delay_buffer["actor"]["first"]
  schedule_buffer = manager._group_obs_delay_group_buffer["actor"]["shared"]
  history_buffer = manager._group_obs_term_history_buffer["actor"]["first"]
  delayed_before = delay_buffer.peek()[untouched_ids].clone()
  lags_before = schedule_buffer.current_lags[untouched_ids].clone()
  history_before = history_buffer.buffer[untouched_ids].clone()

  manager.reset(env_ids=reset_ids)
  manager.compute(update_history=True, env_ids=reset_ids)

  assert torch.equal(delay_buffer.peek()[untouched_ids], delayed_before)
  assert torch.equal(schedule_buffer.current_lags[untouched_ids], lags_before)
  assert torch.equal(history_buffer.buffer[untouched_ids], history_before)
  assert torch.equal(
    schedule_buffer.current_lags[reset_ids], torch.zeros_like(reset_ids)
  )
  second_buffer = manager._group_obs_term_delay_buffer["actor"]["second"]
  assert torch.equal(delay_buffer.current_lags, second_buffer.current_lags)


def test_delay_group_compute_cache_does_not_advance_shared_schedule():
  device = get_test_device()
  env = _mock_env(device)
  first, first_counter = _counting_term(1.0, device)
  last, last_counter = _counting_term(2.0, device)
  cfg = {
    "actor": ObservationGroupCfg(
      terms={
        "first": _delayed_term(
          first, delay_min_lag=0, delay_max_lag=3, delay_group="shared"
        ),
        "last": _delayed_term(
          last, delay_min_lag=0, delay_max_lag=3, delay_group="shared"
        ),
      }
    )
  }
  manager = ObservationManager(cfg, env)
  first_result = manager.compute(update_history=True)
  schedule = manager._group_obs_delay_group_buffer["actor"]["shared"]
  lags_before = schedule.current_lags.clone()
  counter_before = (first_counter["value"], last_counter["value"])

  assert manager.compute() is first_result
  assert manager.compute() is first_result
  assert torch.equal(schedule.current_lags, lags_before)
  assert (first_counter["value"], last_counter["value"]) == counter_before


@pytest.mark.parametrize(
  ("terms", "match"),
  [
    (
      {
        "first": ObservationTermCfg(
          func=lambda env: torch.zeros((env.num_envs, 1), device=env.device),
          delay_min_lag=0,
          delay_max_lag=1,
          delay_group="shared",
        ),
        "second": ObservationTermCfg(
          func=lambda env: torch.zeros((env.num_envs, 1), device=env.device),
          delay_min_lag=1,
          delay_max_lag=1,
          delay_group="shared",
        ),
      },
      "inconsistent delay settings",
    ),
    (
      {
        "obs": ObservationTermCfg(
          func=lambda env: torch.zeros((env.num_envs, 1), device=env.device),
          delay_min_lag=-1,
          delay_max_lag=0,
          delay_group="shared",
        )
      },
      "delay_min_lag < 0",
    ),
  ],
)
def test_delay_group_rejects_invalid_settings(terms, match):
  device = get_test_device()
  with pytest.raises(ValueError, match=match):
    ObservationManager({"actor": ObservationGroupCfg(terms=terms)}, _mock_env(device))


def test_default_delay_terms_keep_independent_schedules():
  device = get_test_device()
  env = _mock_env(device)
  first, _ = _counting_term(1.0, device)
  second, _ = _counting_term(2.0, device)
  cfg = {
    "actor": ObservationGroupCfg(
      terms={
        "first": _delayed_term(first, delay_min_lag=1, delay_max_lag=1),
        "second": _delayed_term(second, delay_min_lag=2, delay_max_lag=2),
      }
    )
  }
  manager = ObservationManager(cfg, env)
  manager.compute(update_history=True)

  assert manager._group_obs_delay_group_buffer["actor"] == {}
  delay_buffers = manager._group_obs_term_delay_buffer["actor"]
  assert delay_buffers["first"] is not delay_buffers["second"]
  assert torch.equal(
    delay_buffers["first"].current_lags,
    torch.ones(env.num_envs, dtype=torch.long, device=device),
  )
  assert torch.equal(
    delay_buffers["second"].current_lags,
    torch.full((env.num_envs,), 2, dtype=torch.long, device=device),
  )
