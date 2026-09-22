"""Tests for the additive X2 tracking observation ablations."""

import math
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import projected_gravity
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationManager,
  ObservationTermCfg,
)
from mjlab.sensor import BuiltinSensor
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.config.agibot_x2.env_cfgs import (
  X2TrackingObservationAblation,
  agibot_x2_flat_tracking_correlated_dr_env_cfg,
  agibot_x2_flat_tracking_observation_ablation_env_cfg,
)
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.noise import ConstantNoiseCfg

PREFIX = "Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR-"
REDUCED = PREFIX + "Reduced-Perturbations"
VARIANTS: dict[str, tuple[X2TrackingObservationAblation, str, list[str], int]] = {
  PREFIX + "Reduced-Perturbations-Projected-Gravity": (
    "projected_gravity",
    "agibot_x2_tracking_correlated_dr_reduced_perturbations_projected_gravity",
    [
      "command",
      "motion_lookahead",
      "projected_gravity",
      "base_ang_vel",
      "joint_pos",
      "joint_vel",
      "actions",
    ],
    161,
  ),
  PREFIX + "Reduced-Perturbations-Projected-Gravity-And-Anchor": (
    "projected_gravity_anchor",
    "agibot_x2_tracking_correlated_dr_reduced_perturbations_projected_gravity_anchor",
    [
      "command",
      "motion_lookahead",
      "projected_gravity",
      "motion_anchor_ori_b",
      "base_ang_vel",
      "joint_pos",
      "joint_vel",
      "actions",
    ],
    167,
  ),
  PREFIX + "Reduced-Perturbations-Vendor-Velocity-Scaling": (
    "vendor_velocity_scaling",
    "agibot_x2_tracking_correlated_dr_reduced_perturbations_vendor_velocity_scaling",
    [
      "command",
      "motion_lookahead",
      "motion_anchor_ori_b",
      "base_ang_vel",
      "joint_pos",
      "joint_vel",
      "actions",
    ],
    164,
  ),
}


@pytest.fixture(autouse=True)
def clear_axis_selection(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.delenv("MJLAB_DR_AXES", raising=False)


def motion_cfg(cfg) -> MotionCommandCfg:
  motion = cfg.commands["motion"]
  assert isinstance(motion, MotionCommandCfg)
  return motion


def _zero_lookahead_env() -> SimpleNamespace:
  zeros_3 = torch.zeros((1, 3))
  identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
  data = SimpleNamespace(
    default_joint_pos=torch.zeros((1, 31)),
    default_joint_vel=torch.zeros((1, 31)),
    joint_pos=torch.zeros((1, 31)),
    joint_pos_biased=torch.zeros((1, 31)),
    joint_vel=torch.zeros((1, 31)),
    projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
  )
  command = SimpleNamespace(
    lookahead_command=torch.empty((1, 0)),
    robot_anchor_pos_w=zeros_3,
    robot_anchor_quat_w=identity,
    anchor_pos_w=zeros_3,
    anchor_quat_w=identity,
  )
  sensor = BuiltinSensor.from_existing("robot/imu_ang_vel")
  sensor._data_view = zeros_3
  command_manager = SimpleNamespace(
    get_command=lambda name: torch.zeros((1, 62)),
    get_term=lambda name: command,
  )
  return SimpleNamespace(
    num_envs=1,
    scene={
      "robot": SimpleNamespace(data=data),
      "robot/imu_ang_vel": sensor,
    },
    command_manager=command_manager,
    action_manager=SimpleNamespace(action=torch.zeros((1, 31))),
  )


def test_variants_are_registered_with_isolated_ppo_directories() -> None:
  for task_id, (_, experiment_name, _, _) in VARIANTS.items():
    assert task_id in list_tasks()
    assert load_rl_cfg(task_id).experiment_name == experiment_name


def test_actor_contracts_have_expected_order_and_dimensions() -> None:
  for task_id, (ablation, _, expected_terms, expected_dim) in VARIANTS.items():
    cfg = load_env_cfg(task_id)
    actor = cfg.observations["actor"].terms
    assert list(actor) == expected_terms
    assert motion_cfg(cfg).lookahead_s == 0.0

    # Execute the configured terms on an actual 31-joint, zero-lookahead input.
    # Corruption and scaling do not change the concatenated tensor dimension.
    env = _zero_lookahead_env()
    observation = torch.cat(
      [term.func(env, **term.params) for term in actor.values()], dim=-1
    )
    assert observation.shape == (1, expected_dim)

    built = agibot_x2_flat_tracking_observation_ablation_env_cfg(ablation)
    assert list(built.observations["actor"].terms) == expected_terms


def test_variants_keep_critic_and_training_contract_unchanged() -> None:
  parent = agibot_x2_flat_tracking_correlated_dr_env_cfg(reduced_perturbations=True)
  for task_id in VARIANTS:
    cfg = load_env_cfg(task_id)
    assert cfg.observations["critic"] == parent.observations["critic"]
    assert cfg.actions == parent.actions
    assert cfg.commands == parent.commands
    assert cfg.rewards == parent.rewards
    assert cfg.terminations == parent.terminations
    assert cfg.episode_length_s == parent.episode_length_s
    assert cfg.events == parent.events

  assert load_env_cfg(REDUCED).observations == parent.observations


def test_projected_gravity_is_root_body_frame_and_yaw_invariant() -> None:
  class RootData:
    def __init__(self, quat: torch.Tensor):
      self.root_link_quat_w = quat
      self.gravity_vec_w = torch.tensor([[0.0, 0.0, -1.0]])

    @property
    def projected_gravity_b(self) -> torch.Tensor:
      return quat_apply_inverse(self.root_link_quat_w, self.gravity_vec_w)

  upright = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
  rolled = torch.tensor(
    [[math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]], dtype=torch.float32
  )
  yawed = torch.tensor(
    [[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]], dtype=torch.float32
  )
  data = RootData(upright)
  env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})
  manager_env = cast(ManagerBasedRlEnv, env)
  torch.testing.assert_close(
    projected_gravity(manager_env), torch.tensor([[0.0, 0.0, -1.0]])
  )
  data.root_link_quat_w = rolled
  torch.testing.assert_close(
    projected_gravity(manager_env),
    torch.tensor([[0.0, -1.0, 0.0]]),
    atol=1e-6,
    rtol=0,
  )
  data.root_link_quat_w = yawed
  torch.testing.assert_close(
    projected_gravity(manager_env),
    torch.tensor([[0.0, 0.0, -1.0]]),
    atol=1e-6,
    rtol=0,
  )

  parent = agibot_x2_flat_tracking_correlated_dr_env_cfg(reduced_perturbations=True)
  for task_id, (ablation, _, _, _) in VARIANTS.items():
    if ablation.startswith("projected_gravity"):
      term = load_env_cfg(task_id).observations["actor"].terms["projected_gravity"]
      anchor = parent.observations["actor"].terms["motion_anchor_ori_b"]
      assert term.noise == anchor.noise
      assert term.noise is not anchor.noise
      assert term.params == {}
      assert term.delay_group is None
      assert term.history_length == 0
      assert term.scale == 1.0


def test_factory_calls_are_isolated_and_play_overrides_are_preserved() -> None:
  first = agibot_x2_flat_tracking_observation_ablation_env_cfg("projected_gravity")
  second = agibot_x2_flat_tracking_observation_ablation_env_cfg(
    "vendor_velocity_scaling"
  )
  first.observations["actor"].terms["projected_gravity"].noise.n_min = -9.0  # type: ignore[union-attr]
  assert "projected_gravity" not in second.observations["actor"].terms

  for task_id in VARIANTS:
    cfg = load_env_cfg(task_id, play=True)
    assert cfg.observations["actor"].enable_corruption is False
    assert "push_robot" not in cfg.events
    motion = motion_cfg(cfg)
    assert motion.sampling_mode == "start"
    assert motion.joint_position_range == (-0.05, 0.05)


@pytest.mark.parametrize("axes", ["none", "pd_gains", "obs_delay"])
def test_variants_respect_selected_dr_axes(
  axes: str, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv("MJLAB_DR_AXES", axes)
  cfg = agibot_x2_flat_tracking_observation_ablation_env_cfg("projected_gravity")
  actor = cfg.observations["actor"].terms
  if axes == "pd_gains":
    assert "randomize_pd_gains" in cfg.events
  else:
    assert "randomize_pd_gains" not in cfg.events
  if axes == "obs_delay":
    assert actor["joint_pos"].delay_group == "encoder_packet"
  else:
    assert actor["joint_pos"].delay_max_lag == 0
    assert actor["joint_vel"].delay_group is None


def test_vendor_scaling_only_changes_measured_velocity_terms() -> None:
  cfg = load_env_cfg(next(task for task in VARIANTS if "Vendor" in task))
  actor = cfg.observations["actor"].terms
  assert actor["base_ang_vel"].scale == 0.25
  assert actor["joint_vel"].scale == 0.05
  assert actor["joint_pos"].scale is None
  assert actor["command"].scale is None
  assert actor["motion_lookahead"].scale is None
  assert actor["actions"].scale is None


def test_observation_manager_applies_noise_before_scale() -> None:
  env = SimpleNamespace(num_envs=1, device="cpu")
  manager = ObservationManager(
    {
      "actor": ObservationGroupCfg(
        terms={
          "value": ObservationTermCfg(
            func=lambda env: torch.full((1, 1), 2.0),
            noise=ConstantNoiseCfg(bias=1.0),
            scale=2.0,
          )
        },
        enable_corruption=True,
      )
    },
    env,
  )
  torch.testing.assert_close(manager.compute()["actor"], torch.tensor([[6.0]]))
