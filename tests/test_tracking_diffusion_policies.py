"""CPU tests for exact OU and action arithmetic."""

from __future__ import annotations

import numpy as np
import pytest

from mjlab.tasks.tracking.diffusion import OUProcess, PolicyError, make_policy_action


def test_ou_is_deterministic_and_resets_per_trial() -> None:
  first = OUProcess(seed=7)
  values = np.stack([first.sample(), first.sample()])
  second = OUProcess(seed=7)
  np.testing.assert_array_equal(values, np.stack([second.sample(), second.sample()]))
  first.reset(7)
  np.testing.assert_array_equal(first.sample(), values[0])


def test_policy_action_preserves_pre_and_post_action_alignment() -> None:
  action = make_policy_action(np.ones(31), np.full(31, 0.25))
  np.testing.assert_allclose(action.executed, 1.25)
  with pytest.raises(PolicyError):
    from mjlab.tasks.tracking.diffusion.policies import PolicyAction

    PolicyAction(np.ones(31), np.zeros(31), np.full(31, 2.0))
