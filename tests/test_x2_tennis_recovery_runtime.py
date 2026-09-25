"""Tests for the X2 tennis-end recovery runtime component.

These tests use a narrow injected synthetic pool (matching the
:class:`EndpointPoolProtocol`) and the ``floating_base_articulated`` test
fixture. They do not require the real endpoint-pool loader or the task
registry.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
import torch
from conftest import get_test_device, load_fixture_xml, make_scene_and_sim

from mjlab.managers.event_manager import EventManager
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.tennis_recovery import (
  TennisRecoveryResetEvent,
  TennisRecoveryVelocityCommand,
  TennisRecoveryVelocityCommandCfg,
  _assign_recovery_groups,
)
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.command_manager import CommandManager
  from mjlab.tasks.velocity.mdp.tennis_recovery import EndpointPoolProtocol

# ---------------------------------------------------------------------------
# Synthetic pool fixture.
# ---------------------------------------------------------------------------


def _make_synthetic_pool(
  n_rows: int = 6,
  n_joints: int = 2,
  joint_names: tuple[str, ...] = ("joint1", "joint2"),
  seed: int = 0,
) -> SimpleNamespace:
  """Create a synthetic pool matching the EndpointPoolProtocol."""
  rng = np.random.default_rng(seed)
  return SimpleNamespace(
    joint_pos=rng.normal(0, 0.3, (n_rows, n_joints)).astype(np.float32),
    joint_vel=rng.normal(0, 1.0, (n_rows, n_joints)).astype(np.float32),
    root_pos_w=rng.normal([0.0, 0.0, 0.8], [0.1, 0.1, 0.05], (n_rows, 3)).astype(
      np.float32
    ),
    root_quat_w=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n_rows, 1)),
    root_lin_vel_w=rng.normal(0, 0.5, (n_rows, 3)).astype(np.float32),
    root_ang_vel_w=rng.normal(0, 0.5, (n_rows, 3)).astype(np.float32),
    trajectory_ids=np.tile(
      np.arange(n_rows // 2, dtype=np.int64)[:, None], (1, 2)
    ).flatten()[:n_rows],
    frame_indices=np.arange(n_rows, dtype=np.int64),
    joint_names=joint_names,
    body_names=("base", "link1", "link2"),
    manifest={"split": "train", "seed": 42},
  )


def _make_nonidentity_pool(
  n_rows: int = 4,
  joint_names: tuple[str, ...] = ("joint1", "joint2"),
) -> SimpleNamespace:
  """Pool with non-identity quaternions and nonzero velocities for rotation tests."""
  rng = np.random.default_rng(123)
  # Yaw 45 degrees, pitch 10 degrees, roll 5 degrees.
  yaw = math.radians(45)
  pitch = math.radians(10)
  roll = math.radians(5)
  cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
  cr, sr = math.cos(roll / 2), math.sin(roll / 2)
  cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
  qw = cy * cr * cp + sy * sr * sp
  qx = cy * sr * cp - sy * cr * sp
  qy = cy * cr * sp + sy * sr * cp
  qz = sy * cr * cp - cy * sr * sp
  quat = np.array([qw, qx, qy, qz], dtype=np.float32)
  quat = quat / np.linalg.norm(quat)

  return SimpleNamespace(
    joint_pos=rng.normal(0, 0.2, (n_rows, 2)).astype(np.float32),
    joint_vel=rng.normal(0, 0.5, (n_rows, 2)).astype(np.float32),
    root_pos_w=np.tile(np.array([0.1, -0.2, 0.75], dtype=np.float32), (n_rows, 1)),
    root_quat_w=np.tile(quat, (n_rows, 1)),
    root_lin_vel_w=np.tile(np.array([0.3, -0.4, 0.1], dtype=np.float32), (n_rows, 1)),
    root_ang_vel_w=np.tile(np.array([0.1, 0.2, 0.5], dtype=np.float32), (n_rows, 1)),
    trajectory_ids=np.array([0, 0, 1, 1], dtype=np.int64),
    frame_indices=np.arange(n_rows, dtype=np.int64),
    joint_names=joint_names,
    body_names=("base", "link1", "link2"),
    manifest={"split": "train"},
  )


# ---------------------------------------------------------------------------
# Helpers to build a minimal env with event + command managers.
# ---------------------------------------------------------------------------


def _build_env(
  device: str,
  num_envs: int = 4,
  recovery_fraction: float = 0.5,
  force_mode: str = "none",
  pool: Any | None = None,
  yaw_aug_range: tuple[float, float] | None = None,
  seed: int = 42,
  resampling_time_range: tuple[float, float] = (1e9, 1e9),
) -> tuple[ManagerBasedRlEnv, TennisRecoveryResetEvent, TennisRecoveryVelocityCommand]:
  """Build a minimal env with event and command managers wired up."""
  scene, sim = make_scene_and_sim(
    device, load_fixture_xml("floating_base_articulated"), sensors=(), num_envs=num_envs
  )

  # Build the recovery reset event term.
  event_cfg_dict: dict[str, Any] = {
    "tennis_recovery_reset": type(
      "Cfg",
      (),
      {
        "mode": "reset",
        "func": TennisRecoveryResetEvent,
        "params": {
          "asset_cfg": SceneEntityCfg("robot"),
          "pool_directory": None,
          "last_n_frames": 10,
          "split": "train",
          "validation_fraction": 0.2,
          "seed": seed,
          "recovery_fraction": recovery_fraction,
          "force_mode": force_mode,
          "yaw_aug_range": yaw_aug_range,
        },
        "interval_range_s": None,
        "is_global_time": False,
        "min_step_count_between_reset": 0,
      },
    )(),
  }

  env = cast(
    "ManagerBasedRlEnv",
    SimpleNamespace(
      scene=scene,
      sim=sim,
      num_envs=num_envs,
      device=device,
      step_dt=0.02,
      common_step_counter=0,
      extras={"log": {}},
      cfg=SimpleNamespace(auto_reset=True, decimation=4),
      episode_length_buf=torch.zeros(num_envs, dtype=torch.long, device=device),
      reset_buf=torch.zeros(num_envs, dtype=torch.bool, device=device),
      reset_terminated=torch.zeros(num_envs, dtype=torch.bool, device=device),
      reset_time_outs=torch.zeros(num_envs, dtype=torch.bool, device=device),
      reward_buf=torch.zeros(num_envs, device=device),
      obs_buf=None,
      _manual_reset_pending=torch.zeros(num_envs, dtype=torch.bool, device=device),
      _sim_step_counter=0,
      single_action_space=None,
      observation_space=None,
      action_space=None,
      seed_value=None,
      recorder_manager=SimpleNamespace(
        record_pre_reset=lambda ids: None,
        record_post_reset=lambda ids: None,
      ),
    ),
  )

  # Resolve SceneEntityCfg in params.
  for v in event_cfg_dict["tennis_recovery_reset"].params.values():
    if isinstance(v, SceneEntityCfg):
      v.resolve(scene)

  event_term_cfg = event_cfg_dict["tennis_recovery_reset"]
  event_term = TennisRecoveryResetEvent(cfg=event_term_cfg, env=env)
  event_term_cfg.func = event_term

  env.event_manager = cast(
    EventManager,
    SimpleNamespace(
      get_term_cfg=lambda name: event_term_cfg,
      available_modes=["reset", "startup"],
      reset=lambda env_ids=None: {},
      apply=lambda mode, env_ids=None, dt=None, global_env_step_count=None: None,
    ),
  )

  # Build the recovery velocity command.
  cmd_cfg = TennisRecoveryVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=resampling_time_range,
    rel_standing_envs=0.1,
    rel_heading_envs=0.3,
    rel_forward_envs=0.0,
    heading_command=True,
    heading_control_stiffness=0.5,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 1.0),
      lin_vel_y=(-1.0, 1.0),
      ang_vel_z=(-0.5, 0.5),
      heading=(-math.pi, math.pi),
    ),
    recovery_reset_event_name="tennis_recovery_reset",
  )
  cmd_term = cmd_cfg.build(env)
  env.command_manager = cast(
    "CommandManager",
    SimpleNamespace(
      get_term=lambda name: cmd_term,
      get_command=lambda name: cmd_term.command,
      reset=lambda env_ids: cmd_term.reset(env_ids),
      compute=lambda dt, env_ids=None: cmd_term.compute(dt, env_ids),
      active_terms=["twist"],
    ),
  )

  # Inject pool if provided.
  if pool is not None:
    event_term.pool = cast("EndpointPoolProtocol", pool)

  return env, event_term, cmd_term


def _full_reset(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  event_term: TennisRecoveryResetEvent,
  cmd_term: TennisRecoveryVelocityCommand,
) -> None:
  """Emulate the env._reset_idx + reset() lifecycle for the recovery components."""
  # _reset_idx: events fire first.
  event_term(env, env_ids)
  # Command reset (metrics + _resample).
  cmd_term.reset(env_ids)
  # write_data + forward.
  env.scene.write_data_to_sim()
  env.sim.forward()
  # command compute(dt=0).
  cmd_term.compute(dt=0.0, env_ids=env_ids)
  env.sim.sense()


# ---------------------------------------------------------------------------
# Tests: group assignment.
# ---------------------------------------------------------------------------


def test_group_assignment_deterministic(device: str) -> None:
  mask1 = _assign_recovery_groups(10, 0.8, 42, "none", device)
  mask2 = _assign_recovery_groups(10, 0.8, 42, "none", device)
  assert torch.equal(mask1, mask2)
  assert mask1.sum().item() == 8
  assert (~mask1).sum().item() == 2


def test_group_assignment_force_modes(device: str) -> None:
  assert _assign_recovery_groups(5, 0.8, 42, "recovery", device).all()
  assert not _assign_recovery_groups(5, 0.8, 42, "retention", device).any()


def test_group_assignment_rejects_invalid_fraction(device: str) -> None:
  with pytest.raises(ValueError, match="finite value in \\[0, 1\\]"):
    _assign_recovery_groups(10, 1.5, 42, "none", device)
  with pytest.raises(ValueError, match="finite value in \\[0, 1\\]"):
    _assign_recovery_groups(10, -0.1, 42, "none", device)
  with pytest.raises(ValueError, match="finite value in \\[0, 1\\]"):
    _assign_recovery_groups(10, float("nan"), 42, "none", device)


def test_group_assignment_tiny_envs(device: str) -> None:
  # 2 envs at 0.8 → round(1.6) = 2 recovery, 0 retention.
  mask = _assign_recovery_groups(2, 0.8, 42, "none", device)
  assert mask.sum().item() == 2


# ---------------------------------------------------------------------------
# Tests: recovery reset writes full state with nonzero velocity.
# ---------------------------------------------------------------------------


@pytest.fixture
def device() -> str:
  return get_test_device()


def test_recovery_reset_writes_full_state(device: str) -> None:
  pool = _make_synthetic_pool(n_rows=6, n_joints=2)
  env, event_term, cmd_term = _build_env(
    device, num_envs=4, recovery_fraction=0.5, force_mode="recovery", pool=pool
  )
  env_ids = torch.arange(4, device=device)

  _full_reset(env, env_ids, event_term, cmd_term)

  robot = env.scene["robot"]
  # Joint pos should be from the pool (not default zeros).
  assert not torch.allclose(
    robot.data.joint_pos, robot.data.default_joint_pos, atol=1e-5
  )
  # Root velocity should be nonzero (from pool).
  assert robot.data.root_link_lin_vel_w.abs().sum() > 0.0
  # XY recentered to env origin.
  origins = env.scene.env_origins[env_ids]
  assert torch.allclose(robot.data.root_link_pos_w[:, :2], origins[:, :2], atol=1e-5)
  # Height preserved from pool (not default).
  default_z = robot.data.default_root_state[:, 2]
  assert not torch.allclose(robot.data.root_link_pos_w[:, 2], default_z, atol=1e-5)


def test_recovery_reset_nonidentity_quaternion_velocity(device: str) -> None:
  """Non-identity quaternion: world omega correctly converted to local qvel."""
  pool = _make_nonidentity_pool(n_rows=4)
  env, event_term, cmd_term = _build_env(
    device, num_envs=2, force_mode="recovery", pool=pool
  )
  env_ids = torch.arange(2, device=device)

  _full_reset(env, env_ids, event_term, cmd_term)

  robot = env.scene["robot"]
  # The root quaternion should be non-identity (from pool).
  quat = robot.data.root_link_quat_w
  assert not torch.allclose(quat, torch.tensor([1.0, 0, 0, 0], device=device))

  # The world-frame angular velocity written should match the pool's value
  # after forward() refreshes derived kinematics. The entity's
  # write_root_link_velocity_to_sim converts world omega → local qvel using
  # the inverse root quaternion, so the stored qvel angular part is in body
  # frame. After forward, root_link_ang_vel_w should recover the world omega.
  expected_ang_vel = torch.from_numpy(np.array([0.1, 0.2, 0.5], dtype=np.float32)).to(
    device
  )
  assert torch.allclose(
    robot.data.root_link_ang_vel_w[env_ids],
    expected_ang_vel.expand(2, -1),
    atol=1e-4,
  )

  # Linear velocity should also survive.
  expected_lin_vel = torch.from_numpy(np.array([0.3, -0.4, 0.1], dtype=np.float32)).to(
    device
  )
  assert torch.allclose(
    robot.data.root_link_lin_vel_w[env_ids],
    expected_lin_vel.expand(2, -1),
    atol=1e-4,
  )


# ---------------------------------------------------------------------------
# Tests: recovery command is zero.
# ---------------------------------------------------------------------------


def test_recovery_command_zero_after_reset(device: str) -> None:
  pool = _make_synthetic_pool()
  env, event_term, cmd_term = _build_env(
    device, num_envs=4, force_mode="recovery", pool=pool
  )
  env_ids = torch.arange(4, device=device)

  _full_reset(env, env_ids, event_term, cmd_term)

  assert torch.all(cmd_term.vel_command_b == 0.0)
  assert torch.all(cmd_term.vel_command_w == 0.0)


def test_recovery_command_zero_through_resamples(device: str) -> None:
  pool = _make_synthetic_pool()
  env, event_term, cmd_term = _build_env(
    device,
    num_envs=4,
    force_mode="recovery",
    pool=pool,
    resampling_time_range=(0.001, 0.001),  # Expires on first compute.
  )
  env_ids = torch.arange(4, device=device)

  _full_reset(env, env_ids, event_term, cmd_term)

  # Step: timer expires and resamples.
  cmd_term.compute(dt=1.0)
  assert torch.all(cmd_term.vel_command_b == 0.0)
  assert torch.all(cmd_term.vel_command_w == 0.0)

  # Another resample.
  cmd_term.compute(dt=1.0)
  assert torch.all(cmd_term.vel_command_b == 0.0)


def test_resample_does_not_write_physical_qvel(device: str) -> None:
  """Command timer expiry resamples but does not write sim velocity."""
  pool = _make_synthetic_pool()
  env, event_term, cmd_term = _build_env(
    device,
    num_envs=2,
    force_mode="recovery",
    pool=pool,
    resampling_time_range=(0.001, 0.001),
  )
  env_ids = torch.arange(2, device=device)

  _full_reset(env, env_ids, event_term, cmd_term)

  robot = env.scene["robot"]
  # Record the velocity after reset.
  lin_vel_before = robot.data.root_link_lin_vel_w.clone()
  ang_vel_before = robot.data.root_link_ang_vel_w.clone()

  # Step: timer expires and resamples (command only, no sim write).
  cmd_term.compute(dt=1.0)

  # Sim velocity unchanged.
  assert torch.allclose(robot.data.root_link_lin_vel_w, lin_vel_before, atol=1e-6)
  assert torch.allclose(robot.data.root_link_ang_vel_w, ang_vel_before, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: partial reset isolation.
# ---------------------------------------------------------------------------


def test_partial_reset_isolation(device: str) -> None:
  pool = _make_synthetic_pool(n_rows=20)
  env, event_term, cmd_term = _build_env(
    device, num_envs=4, force_mode="recovery", pool=pool
  )
  all_ids = torch.arange(4, device=device)

  # First reset: all envs.
  _full_reset(env, all_ids, event_term, cmd_term)
  robot = env.scene["robot"]
  pos_all = robot.data.root_link_pos_w.clone()
  vel_all = robot.data.root_link_lin_vel_w.clone()

  # Reset only env 0. Seed RNG so env 0 samples a different row than it did
  # in the first reset (which consumed RNG in cmd_term.reset/compute).
  torch.manual_seed(999)
  partial_ids = torch.tensor([0], device=device)
  _full_reset(env, partial_ids, event_term, cmd_term)

  # Env 0 changed; envs 1-3 unchanged.
  assert not torch.allclose(robot.data.root_link_pos_w[0], pos_all[0], atol=1e-5)
  assert torch.allclose(robot.data.root_link_pos_w[1:4], pos_all[1:4], atol=1e-6)
  assert torch.allclose(robot.data.root_link_lin_vel_w[1:4], vel_all[1:4], atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: retention envs keep original behavior.
# ---------------------------------------------------------------------------


def test_retention_envs_get_nonzero_commands(device: str) -> None:
  pool = _make_synthetic_pool()
  env, event_term, cmd_term = _build_env(
    device,
    num_envs=4,
    force_mode="retention",
    pool=pool,
    resampling_time_range=(1e9, 1e9),
  )
  env_ids = torch.arange(4, device=device)

  # Reset retention envs (no recovery reset, original command distribution).
  cmd_term.reset(env_ids)
  cmd_term.compute(dt=0.0, env_ids=env_ids)

  # At least some retention envs should have nonzero commands.
  assert cmd_term.vel_command_b.abs().sum() > 0.0


def test_retention_envs_not_touched_by_recovery_event(device: str) -> None:
  pool = _make_synthetic_pool()
  env, event_term, cmd_term = _build_env(
    device, num_envs=4, force_mode="retention", pool=pool
  )
  env_ids = torch.arange(4, device=device)

  # Write default state first.
  robot = env.scene["robot"]
  default_root = robot.data.default_root_state[env_ids].clone()
  default_root[:, :2] += env.scene.env_origins[env_ids, :2]
  robot.write_root_state_to_sim(default_root, env_ids=env_ids)
  env.sim.forward()

  pos_before = robot.data.root_link_pos_w.clone()

  # Call the recovery event (should do nothing for retention envs).
  event_term(env, env_ids)

  # State unchanged.
  assert torch.allclose(robot.data.root_link_pos_w, pos_before, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: forced retention with no loader (no pool import).
# ---------------------------------------------------------------------------


def test_forced_retention_no_pool_loading(device: str) -> None:
  """Forced retention: __call__ returns before _ensure_pool, no loader needed."""
  env, event_term, cmd_term = _build_env(
    device, num_envs=4, force_mode="retention", pool=None
  )
  env_ids = torch.arange(4, device=device)

  # This should NOT raise even though pool is None and pool_directory is None.
  event_term(env, env_ids)

  # Pool was never loaded.
  assert event_term._cached is None
  assert event_term.pool is None


# ---------------------------------------------------------------------------
# Tests: provenance tracking.
# ---------------------------------------------------------------------------


def test_provenance_tracking(device: str) -> None:
  pool = _make_synthetic_pool(n_rows=6)
  env, event_term, cmd_term = _build_env(
    device, num_envs=2, force_mode="recovery", pool=pool
  )
  env_ids = torch.arange(2, device=device)

  _full_reset(env, env_ids, event_term, cmd_term)

  # Provenance should be set for recovery envs.
  assert (event_term.last_row_indices[env_ids] >= 0).all()
  assert (event_term.last_trajectory_ids[env_ids] >= 0).all()
  assert (event_term.last_frame_indices[env_ids] >= 0).all()

  # Non-reset envs should still have -1.
  if event_term._num_envs > 2:
    assert (event_term.last_row_indices[2:] == -1).all()


# ---------------------------------------------------------------------------
# Tests: seed reproducibility.
# ---------------------------------------------------------------------------


def test_seed_reproducible_sampling(device: str) -> None:
  pool = _make_synthetic_pool(n_rows=10, seed=0)
  env1, event1, cmd1 = _build_env(
    device, num_envs=2, force_mode="recovery", pool=pool, seed=42
  )
  env2, event2, cmd2 = _build_env(
    device, num_envs=2, force_mode="recovery", pool=pool, seed=42
  )
  env_ids = torch.arange(2, device=device)

  # Control global RNG state so the row sampling (torch.randint) is identical.
  # In the real env, the event fires at a deterministic point in the env's
  # RNG sequence given the same seed.
  torch.manual_seed(0)
  event1(env1, env_ids)
  rows1 = event1.last_row_indices[env_ids].clone()

  torch.manual_seed(0)
  event2(env2, env_ids)
  rows2 = event2.last_row_indices[env_ids].clone()

  assert torch.equal(rows1, rows2)


# ---------------------------------------------------------------------------
# Tests: yaw augmentation.
# ---------------------------------------------------------------------------


def test_yaw_augmentation_rotates_velocities(device: str) -> None:
  pool = _make_nonidentity_pool(n_rows=4)
  env, event_term, cmd_term = _build_env(
    device,
    num_envs=2,
    force_mode="recovery",
    pool=pool,
    yaw_aug_range=(math.pi / 2, math.pi / 2),  # Fixed 90° yaw.
  )
  env_ids = torch.arange(2, device=device)

  _full_reset(env, env_ids, event_term, cmd_term)

  robot = env.scene["robot"]
  # Original lin vel was [0.3, -0.4, 0.1]. After 90° yaw rotation:
  # R_z(90°) * [0.3, -0.4, 0.1] = [0.4, 0.3, 0.1].
  expected = torch.tensor([0.4, 0.3, 0.1], dtype=torch.float32, device=device)
  assert torch.allclose(
    robot.data.root_link_lin_vel_w[env_ids],
    expected.expand(2, -1),
    atol=1e-4,
  )


# ---------------------------------------------------------------------------
# Tests: factory / base invariance.
# ---------------------------------------------------------------------------


def test_factory_preserves_base_config():
  from mjlab.tasks.velocity.config.agibot_x2.env_cfgs import (
    agibot_x2_flat_velocity_env_cfg,
  )
  from mjlab.tasks.velocity.config.agibot_x2.tennis_recovery_env_cfg import (
    agibot_x2_tennis_recovery_env_cfg,
  )

  base = agibot_x2_flat_velocity_env_cfg()
  rec = agibot_x2_tennis_recovery_env_cfg()

  # Episode length unchanged.
  assert base.episode_length_s == rec.episode_length_s
  # Rewards unchanged.
  assert set(base.rewards.keys()) == set(rec.rewards.keys())
  for key in base.rewards:
    assert base.rewards[key].weight == rec.rewards[key].weight
  # Terminations unchanged.
  assert set(base.terminations.keys()) == set(rec.terminations.keys())
  # Actions unchanged.
  from mjlab.envs.mdp.actions import JointPositionActionCfg

  base_joint = cast(JointPositionActionCfg, base.actions["joint_pos"])
  rec_joint = cast(JointPositionActionCfg, rec.actions["joint_pos"])
  assert base_joint.scale == rec_joint.scale
  # Observations unchanged.
  assert set(base.observations["actor"].terms.keys()) == set(
    rec.observations["actor"].terms.keys()
  )
  assert set(base.observations["critic"].terms.keys()) == set(
    rec.observations["critic"].terms.keys()
  )
  # Events: recovery adds exactly one event.
  assert "tennis_recovery_reset" in rec.events
  assert "tennis_recovery_reset" not in base.events
  assert set(base.events.keys()) == set(rec.events.keys()) - {"tennis_recovery_reset"}
  # Command is the recovery specialization.
  assert isinstance(rec.commands["twist"], TennisRecoveryVelocityCommandCfg)


def test_factory_play_mode():
  from mjlab.tasks.velocity.config.agibot_x2.tennis_recovery_env_cfg import (
    agibot_x2_tennis_recovery_env_cfg,
  )

  rec = agibot_x2_tennis_recovery_env_cfg(play=True)
  # Play mode: infinite episode, no curriculum, no push.
  assert rec.episode_length_s == int(1e9)
  assert "push_robot" not in rec.events
  assert len(rec.curriculum) == 0


# ---------------------------------------------------------------------------
# Tests: real event-manager integration.
# ---------------------------------------------------------------------------


def test_real_event_manager_integration(device: str) -> None:
  """Verify the event manager correctly instantiates and fires the class term."""
  from mjlab.managers.event_manager import EventTermCfg

  scene, sim = make_scene_and_sim(
    device, load_fixture_xml("floating_base_articulated"), sensors=(), num_envs=2
  )
  robot = scene["robot"]

  env = cast(
    "ManagerBasedRlEnv",
    SimpleNamespace(
      scene=scene,
      sim=sim,
      num_envs=2,
      device=device,
      step_dt=0.02,
      common_step_counter=0,
      extras={"log": {}},
      cfg=SimpleNamespace(auto_reset=True, decimation=4),
      episode_length_buf=torch.zeros(2, dtype=torch.long, device=device),
      reset_buf=torch.zeros(2, dtype=torch.bool, device=device),
      reset_terminated=torch.zeros(2, dtype=torch.bool, device=device),
      reset_time_outs=torch.zeros(2, dtype=torch.bool, device=device),
      reward_buf=torch.zeros(2, device=device),
      obs_buf=None,
      _manual_reset_pending=torch.zeros(2, dtype=torch.bool, device=device),
      _sim_step_counter=0,
      single_action_space=None,
      observation_space=None,
      action_space=None,
      seed_value=None,
      recorder_manager=SimpleNamespace(
        record_pre_reset=lambda ids: None,
        record_post_reset=lambda ids: None,
      ),
    ),
  )

  pool = _make_synthetic_pool(n_rows=4)

  event_cfg = EventTermCfg(
    mode="reset",
    func=TennisRecoveryResetEvent,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "pool_directory": None,
      "last_n_frames": 10,
      "split": "train",
      "validation_fraction": 0.2,
      "seed": 42,
      "recovery_fraction": 1.0,
      "force_mode": "none",
      "yaw_aug_range": None,
    },
  )

  event_manager = EventManager({"tennis_recovery_reset": event_cfg}, env)

  # The event manager should have instantiated the class term.
  term_cfg = event_manager.get_term_cfg("tennis_recovery_reset")
  assert isinstance(term_cfg.func, TennisRecoveryResetEvent)

  # Inject pool.
  term_cfg.func.pool = cast("EndpointPoolProtocol", pool)

  # Fire the reset event through the real event manager.
  env_ids = torch.arange(2, device=device)
  event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=0)

  # Refresh derived kinematics (the env's reset() does this after events).
  scene.write_data_to_sim()
  sim.forward()

  # State should be written.
  assert not torch.allclose(
    robot.data.joint_pos, robot.data.default_joint_pos, atol=1e-5
  )
  assert robot.data.root_link_lin_vel_w.abs().sum() > 0.0
