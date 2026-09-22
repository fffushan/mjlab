"""Runtime tests for correlated PD-gain domain randomization."""

from unittest.mock import Mock

import pytest
import torch

from mjlab import actuator
from mjlab.envs.mdp import dr
from mjlab.managers.scene_entity_config import SceneEntityCfg


@pytest.fixture
def device():
  return "cpu"


def _make_env(device: str, num_envs: int = 2):
  """Create a CPU mock environment with supported PD actuator layouts."""
  env = Mock()
  env.num_envs = num_envs
  env.device = device

  position = Mock(spec=actuator.BuiltinPositionActuator)
  position.global_ctrl_ids = torch.tensor([0, 1], device=device)

  paired = Mock(spec=actuator.BuiltinPdActuator)
  paired.global_ctrl_ids = torch.tensor([2, 3, 4, 5], device=device)
  paired.num_targets = 2

  ideal = Mock(spec=actuator.IdealPdActuator)
  ideal.global_ctrl_ids = torch.tensor([6, 7], device=device)
  ideal.stiffness = torch.full((num_envs, 2), 100.0, device=device)
  ideal.damping = torch.full((num_envs, 2), 10.0, device=device)
  ideal.default_stiffness = ideal.stiffness.clone()
  ideal.default_damping = ideal.damping.clone()
  ideal.set_gains = actuator.IdealPdActuator.set_gains.__get__(ideal)

  entity = Mock()
  entity.actuators = [position, paired, ideal]
  env.scene = {"robot": entity}

  gainprm = torch.zeros((num_envs, 8, 10), device=device)
  biasprm = torch.zeros((num_envs, 8, 10), device=device)
  gainprm[:, [0, 1, 2, 3, 6, 7], 0] = 100.0
  biasprm[:, [0, 1, 2, 3, 6, 7], 1] = -100.0
  biasprm[:, [0, 1], 2] = -10.0
  gainprm[:, [4, 5], 0] = 10.0
  biasprm[:, [4, 5], 2] = -10.0

  env.sim = Mock()
  env.sim.model = Mock()
  env.sim.model.actuator_gainprm = gainprm.clone()
  env.sim.model.actuator_biasprm = biasprm.clone()
  defaults = {
    "actuator_gainprm": gainprm[0].clone(),
    "actuator_biasprm": biasprm[0].clone(),
  }
  env.sim.get_default_field = lambda field: defaults[field]
  return env, ideal


def _shared_kwargs():
  return {
    "kp_range": (0.5, 2.0),
    "kd_range": (0.5, 2.0),
    "asset_cfg": SceneEntityCfg("robot"),
    "shared_gain_scale": True,
  }


def test_shared_gain_scale_matches_kp_and_kd_per_target(device):
  env, ideal = _make_env(device)
  torch.manual_seed(7)

  dr.pd_gains(env, env_ids=None, **_shared_kwargs())

  gains = env.sim.model.actuator_gainprm
  bias = env.sim.model.actuator_biasprm
  position_kp = gains[:, [0, 1], 0] / 100.0
  position_kd = -bias[:, [0, 1], 2] / 10.0
  paired_kp = gains[:, [2, 3], 0] / 100.0
  paired_kd = gains[:, [4, 5], 0] / 10.0
  ideal_kp = ideal.stiffness / 100.0
  ideal_kd = ideal.damping / 10.0

  torch.testing.assert_close(position_kp, position_kd)
  torch.testing.assert_close(paired_kp, paired_kd)
  torch.testing.assert_close(ideal_kp, ideal_kd)
  assert not torch.allclose(paired_kp[0], paired_kp[1])
  assert not torch.allclose(paired_kp[:, 0], paired_kp[:, 1])
  assert torch.allclose(bias[:, [2, 3], 2], torch.zeros_like(bias[:, [2, 3], 2]))


def test_shared_gain_scale_uses_defaults_without_accumulation(device):
  env, ideal = _make_env(device)
  kwargs = {
    "kp_range": (1.5, 1.5),
    "kd_range": (1.5, 1.5),
    "asset_cfg": SceneEntityCfg("robot"),
    "shared_gain_scale": True,
  }

  for _ in range(3):
    dr.pd_gains(env, env_ids=None, **kwargs)

  gains = env.sim.model.actuator_gainprm
  assert torch.allclose(gains[:, [0, 1, 2, 3], 0], torch.full((2, 4), 150.0))
  assert torch.allclose(gains[:, [4, 5], 0], torch.full((2, 2), 15.0))
  assert torch.allclose(ideal.stiffness, torch.full((2, 2), 150.0))
  assert torch.allclose(ideal.damping, torch.full((2, 2), 15.0))


def test_default_pd_gain_scale_samples_kp_and_kd_independently(device):
  env, _ = _make_env(device)
  torch.manual_seed(7)

  dr.pd_gains(
    env,
    env_ids=None,
    kp_range=(0.5, 2.0),
    kd_range=(0.5, 2.0),
    asset_cfg=SceneEntityCfg("robot"),
  )

  gains = env.sim.model.actuator_gainprm
  assert not torch.allclose(gains[:, [2, 3], 0] / 100.0, gains[:, [4, 5], 0] / 10.0)


@pytest.mark.parametrize(
  ("operation", "kp_range", "kd_range", "message"),
  [
    ("abs", (1.0, 1.0), (1.0, 1.0), "requires operation='scale'"),
    ("scale", (0.8, 1.2), (0.9, 1.2), "requires identical kp_range"),
  ],
)
def test_shared_gain_scale_rejects_incompatible_configuration(
  device, operation, kp_range, kd_range, message
):
  env, _ = _make_env(device)

  with pytest.raises(ValueError, match=message):
    dr.pd_gains(
      env,
      env_ids=None,
      kp_range=kp_range,
      kd_range=kd_range,
      asset_cfg=SceneEntityCfg("robot"),
      operation=operation,
      shared_gain_scale=True,
    )
