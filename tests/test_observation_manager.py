"""Tests for ObservationManager behavior."""

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
from mjlab.utils.noise.noise_cfg import UniformNoiseCfg


@pytest.fixture
def mock_env():
  env = Mock()
  env.num_envs = 4
  env.device = get_test_device()
  env.step_dt = 0.02
  return env


def _dummy_term(mock_env):
  def func(_env, **_kwargs):
    return torch.zeros(mock_env.num_envs, 3, device=mock_env.device)

  return ObservationTermCfg(func=func)


def test_empty_terms_dict_skipped(mock_env):
  """A group declared with no terms is skipped rather than raising."""
  cfg = {"actor": ObservationGroupCfg(terms={})}
  mgr = ObservationManager(cfg, mock_env)

  assert "actor" not in mgr.active_terms
  assert "actor" not in mgr.group_obs_dim


def test_all_terms_none_skipped(mock_env):
  """A group whose every term is None is skipped rather than raising."""
  cfg = {
    "actor": ObservationGroupCfg(
      terms={"a": None, "b": None},  # type: ignore[dict-item]
    ),
  }
  mgr = ObservationManager(cfg, mock_env)

  assert "actor" not in mgr.active_terms
  assert "actor" not in mgr.group_obs_dim


def test_empty_group_skipped_alongside_active_group(mock_env):
  """Active groups coexist with empty ones; only the empty group is dropped."""
  cfg = {
    "actor": ObservationGroupCfg(terms={"a": _dummy_term(mock_env)}),
    "critic": ObservationGroupCfg(terms={}),
  }
  mgr = ObservationManager(cfg, mock_env)

  assert "actor" in mgr.active_terms
  assert "critic" not in mgr.active_terms


def _stateful_term(state: torch.Tensor):
  def func(_env, **_kwargs):
    return state.clone()

  return func


def _stateful_env_state(mock_env) -> torch.Tensor:
  return torch.arange(
    1, mock_env.num_envs + 1, dtype=torch.float32, device=mock_env.device
  ).reshape(-1, 1)


def _actor_tensor(manager: ObservationManager) -> torch.Tensor:
  value = manager.compute()["actor"]
  assert isinstance(value, torch.Tensor)
  return value


def test_refresh_preserves_unedited_env_rows(mock_env):
  """A scoped refresh must not resample the unedited envs' cached rows."""
  state = _stateful_env_state(mock_env)
  cfg = {
    "actor": ObservationGroupCfg(
      terms={
        "raw": ObservationTermCfg(
          func=_stateful_term(state),
          noise=UniformNoiseCfg(n_min=-1.0, n_max=1.0),
        ),
        "history": ObservationTermCfg(func=_stateful_term(state), history_length=2),
      },
      enable_corruption=True,
    )
  }
  manager = ObservationManager(cfg, mock_env)
  manager.compute(update_history=True)
  manager.compute(update_history=True)
  torch.manual_seed(17)
  before = _actor_tensor(manager).clone()
  history = manager._group_obs_term_history_buffer["actor"]["history"]

  state[2] = 99.0
  merged = manager.refresh(torch.tensor([2], device=mock_env.device))
  after = merged["actor"]
  assert isinstance(after, torch.Tensor)

  for index in (0, 1, 3):
    assert torch.equal(before[index], after[index])
  # The edited row is freshly recomputed: a noisy raw value plus backfilled
  # (un-noised) history slots.
  assert abs(float(after[2, 0]) - 99.0) <= 1.0
  assert after[2, 1:].tolist() == [99.0, 99.0]
  # Only the edited env's history was backfilled; nobody gained a tick.
  assert history._num_pushes.tolist() == [2, 2, 1, 2]
  # The cache is the merged buffer, so the next compute() returns it unchanged.
  assert _actor_tensor(manager) is after


def test_refresh_splices_non_concatenated_groups(mock_env):
  state = _stateful_env_state(mock_env)
  cfg = {
    "actor": ObservationGroupCfg(
      terms={"value": ObservationTermCfg(func=_stateful_term(state))},
      concatenate_terms=False,
    )
  }
  manager = ObservationManager(cfg, mock_env)
  manager.compute(update_history=True)
  before_group = manager.compute()["actor"]
  before = cast("dict[str, torch.Tensor]", before_group)["value"].clone()

  state[1] = 42.0
  merged = manager.refresh(torch.tensor([1], device=mock_env.device))
  after_group = merged["actor"]
  after = cast("dict[str, torch.Tensor]", after_group)["value"]

  assert torch.equal(before[0], after[0])
  assert torch.equal(before[2], after[2])
  assert float(after[1]) == 42.0


def test_refresh_without_a_cache_returns_recomputed_rows(mock_env):
  state = _stateful_env_state(mock_env)
  cfg = {
    "actor": ObservationGroupCfg(
      terms={"value": ObservationTermCfg(func=_stateful_term(state))}
    )
  }
  manager = ObservationManager(cfg, mock_env)
  assert manager._obs_buffer is None

  merged = manager.refresh(torch.tensor([1], device=mock_env.device))

  recomputed = merged["actor"]
  assert isinstance(recomputed, torch.Tensor)
  assert recomputed.shape == (mock_env.num_envs, 1)
  assert _actor_tensor(manager) is recomputed


def test_refresh_rejects_malformed_env_ids(mock_env):
  cfg = {
    "actor": ObservationGroupCfg(terms={"value": _dummy_term(mock_env)}),
  }
  manager = ObservationManager(cfg, mock_env)
  manager.compute()
  with pytest.raises(ValueError, match="rank-1 integer"):
    manager.refresh(torch.zeros((1, 1), dtype=torch.long))
  with pytest.raises(ValueError, match="rank-1 integer"):
    manager.refresh(torch.tensor([0.5]))


def test_cached_observations_is_an_owned_snapshot(mock_env):
  """A baseline snapshot must not alias live buffer storage."""
  state = _stateful_env_state(mock_env)
  cfg = {
    "actor": ObservationGroupCfg(
      terms={
        "history": ObservationTermCfg(func=_stateful_term(state), history_length=2)
      }
    )
  }
  manager = ObservationManager(cfg, mock_env)
  assert manager.cached_observations() is None

  manager.compute(update_history=True)
  baseline = manager.cached_observations()
  assert baseline is not None
  snapshot = baseline["actor"]
  assert isinstance(snapshot, torch.Tensor)
  assert snapshot[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]

  # Mutating the live state and recomputing cannot change the snapshot.
  state[:] = 77.0
  manager.compute(update_history=True)
  assert snapshot[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_refresh_baseline_restores_rows_from_before_a_partial_reset(mock_env):
  """A partial reset plus edit must not leak the reset's whole-batch resample."""
  state = _stateful_env_state(mock_env)
  cfg = {
    "actor": ObservationGroupCfg(
      terms={
        "raw": ObservationTermCfg(
          func=_stateful_term(state),
          noise=UniformNoiseCfg(n_min=-1.0, n_max=1.0),
        ),
        "history": ObservationTermCfg(func=_stateful_term(state), history_length=2),
      },
      enable_corruption=True,
    )
  }
  manager = ObservationManager(cfg, mock_env)
  manager.compute(update_history=True)
  baseline = manager.cached_observations()
  before = _actor_tensor(manager).clone()

  # Reproduce the viewer's transaction: partial reset, then the command edit.
  edited = torch.tensor([2], device=mock_env.device)
  manager.reset(edited)
  state[2] = 1.0
  manager.compute(update_history=True, env_ids=edited)
  state[2] = 9.0

  merged = manager.refresh(edited, baseline=baseline)
  after = merged["actor"]
  assert isinstance(after, torch.Tensor)

  for index in (0, 1, 3):
    assert torch.equal(before[index], after[index])
  assert after[2, 1:].tolist() == [9.0, 9.0]
