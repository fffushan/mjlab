"""Focused tests for ``scripts/record_tracking_policy_commands.py``.

The rollout tests use a fake vector env that mimics the real environment's step
ordering (command frame and observations advance *after* the step) so that
mis-pairing an action with a post-step observation fails the test. No GPU or
native simulator is required here.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg

SCRIPT_PATH = (
  Path(__file__).parents[1] / "scripts" / "record_tracking_policy_commands.py"
)
NUM_JOINTS = 3
JOINT_NAMES = ("left_hip_pitch_joint", "left_knee_joint", "right_knee_joint")
OBS_DIM = 5
STEP_DT = 0.02
ENCODER_BIAS = (0.01, 0.02, 0.03)
KP = (100.0, 150.0, 120.0)
KD = (2.0, 3.0, 2.5)
EFFORT_LIMIT = (60.0, 50.0, 40.0)
ACTION_SCALE = (0.35, 0.35, 0.35)
ACTION_OFFSET = (0.5, 0.4, 0.3)


def _load_diagnostic_module() -> Any:
  spec = importlib.util.spec_from_file_location(
    "record_tracking_policy_commands", SCRIPT_PATH
  )
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  # Register before executing so dataclasses can resolve string annotations.
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


@pytest.fixture(scope="module")
def diag() -> Any:
  return _load_diagnostic_module()


##
# Fake environment.
##


class FakeData:
  def __init__(self, num_envs: int = 1, num_joints: int = NUM_JOINTS) -> None:
    self.indexing = SimpleNamespace(ctrl_ids=torch.arange(num_joints))
    self.data = SimpleNamespace(ctrl=torch.zeros(num_envs, num_joints))
    self.joint_pos = torch.zeros(num_envs, num_joints)
    self.joint_vel = torch.zeros(num_envs, num_joints)
    self.encoder_bias = torch.tensor([ENCODER_BIAS], dtype=torch.float32)
    self.joint_pos_target = torch.zeros(num_envs, num_joints)
    self.qfrc_actuator = torch.zeros(num_envs, num_joints)
    self.actuator_force = torch.zeros(num_envs, num_joints)
    self.root_link_pose_w = torch.zeros(num_envs, 7)
    self.root_link_vel_w = torch.zeros(num_envs, 6)

  @property
  def joint_pos_biased(self) -> torch.Tensor:
    return self.joint_pos + self.encoder_bias


class FakeActionTerm:
  def __init__(self) -> None:
    self.target_ids = torch.arange(NUM_JOINTS)
    self.target_names = list(JOINT_NAMES)
    self.scale = torch.tensor([ACTION_SCALE])
    self.offset = torch.tensor([ACTION_OFFSET])
    self.raw_action = torch.zeros(1, NUM_JOINTS)
    self._processed_actions = torch.zeros(1, NUM_JOINTS)


class FakeCommand:
  def __init__(self, frames: int) -> None:
    self.time_steps = torch.zeros(1, dtype=torch.long)
    self.motion = SimpleNamespace(fps=1.0 / STEP_DT, time_step_total=frames)
    self.wrap_count = 0
    self._joint_pos = torch.zeros(frames, NUM_JOINTS)
    self._joint_vel = torch.zeros(frames, NUM_JOINTS)
    for frame in range(frames):
      self._joint_pos[frame] = float(frame) * 0.1
      self._joint_vel[frame] = float(frame) * 0.01

  @property
  def joint_pos(self) -> torch.Tensor:
    return self._joint_pos[self.time_steps]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self._joint_vel[self.time_steps]

  def advance(self, robot_data: FakeData) -> None:
    """Advance the reference frame the way ``MotionCommand._update_command`` does.

    Native behavior: the frame increments by one and, on reaching
    ``time_step_total``, ``MotionCommand`` resamples: it teleports the robot to a
    new frame (for ``sampling_mode='start'``, back to frame 0) and calls
    ``robot.reset()``, which clears the position targets.
    """
    self.time_steps += 1
    if int(self.time_steps[0].item()) >= self.motion.time_step_total:
      self.wrap_count += 1
      self.time_steps = torch.zeros(1, dtype=torch.long)
      robot_data.joint_pos_target.zero_()
      robot_data.data.ctrl.zero_()


class FakeTerminationManager:
  def __init__(self, terms: tuple[str, ...], terminate_at: int | None) -> None:
    self.active_terms = list(terms)
    self.terminate_at = terminate_at
    self.terminated = torch.zeros(1, dtype=torch.bool)
    self.time_outs = torch.zeros(1, dtype=torch.bool)
    self._flags = {name: torch.zeros(1, dtype=torch.bool) for name in terms}

  def get_term(self, name: str) -> torch.Tensor:
    return self._flags[name]

  def update(self, step_index: int) -> None:
    fired = self.terminate_at is not None and step_index == self.terminate_at
    for name, flag in self._flags.items():
      flag.fill_(fired and name == "ee_body_pos")
    self.terminated.fill_(fired and self._flags["ee_body_pos"][0].item())

  def clear(self) -> None:
    self.terminated.zero_()
    self.time_outs.zero_()
    for flag in self._flags.values():
      flag.zero_()


class FakeRawEnv:
  def __init__(self, frames: int, terms: tuple[str, ...], terminate_at: int | None):
    self.num_envs = 1
    self.device = "cpu"
    self.step_dt = STEP_DT
    self.episode_length_buf = torch.zeros(1, dtype=torch.long)
    self.sim = SimpleNamespace(
      data=SimpleNamespace(time=torch.zeros(1, dtype=torch.float64)),
      forward=lambda: None,
    )
    self.termination_manager = FakeTerminationManager(terms, terminate_at)
    self.command = FakeCommand(frames)
    self.action_term = FakeActionTerm()
    self.robot = SimpleNamespace(data=FakeData())
    self.command_manager = SimpleNamespace(get_term=lambda name: self.command)
    self.action_manager = SimpleNamespace(get_term=lambda name: self.action_term)
    self.scene = {"robot": self.robot}


class FakeVecEnv:
  """Mimics ``RslRlVecEnvWrapper`` step ordering for a single environment."""

  def __init__(
    self,
    *,
    frames: int = 50,
    terms: tuple[str, ...] = ("ee_body_pos", "time_out"),
    terminate_at: int | None = None,
    clip_actions: float | None = None,
  ) -> None:
    self.unwrapped = FakeRawEnv(frames, terms, terminate_at)
    self.clip_actions = clip_actions
    self.step_index = 0
    self.step_count = 0
    self.reset_count = 0
    self._obs = torch.full((1, OBS_DIM), 1.0)

  def get_observations(self) -> dict[str, torch.Tensor]:
    return {"actor": self._obs}

  def step(self, actions: torch.Tensor) -> tuple[Any, Any, Any, Any]:
    raw = self.unwrapped
    if bool(raw.termination_manager.terminated[0]):
      raise AssertionError("step() called after a termination without a reset")
    self.step_count += 1
    action_term = raw.action_term
    action_term.raw_action = actions.detach().clone()
    action_term._processed_actions = (
      actions.detach() * action_term.scale + action_term.offset
    )
    target = action_term._processed_actions - raw.robot.data.encoder_bias
    raw.robot.data.joint_pos_target = target.clone()
    raw.robot.data.data.ctrl = target.clone()
    previous_q = raw.robot.data.joint_pos.clone()
    raw.robot.data.joint_pos = previous_q + 0.1 * target
    raw.robot.data.joint_vel = (raw.robot.data.joint_pos - previous_q) / STEP_DT
    # Mimic a MuJoCo <position> actuator: kp*(ctrl - q) - kd*dq, effort-limited.
    kp = torch.tensor(KP)
    kd = torch.tensor(KD)
    limit = torch.tensor(EFFORT_LIMIT)
    raw.robot.data.qfrc_actuator = torch.clamp(
      kp * (target - raw.robot.data.joint_pos) - kd * raw.robot.data.joint_vel,
      -limit,
      limit,
    )
    raw.robot.data.actuator_force = raw.robot.data.qfrc_actuator.clone()
    raw.robot.data.root_link_pose_w = torch.full((1, 7), float(self.step_index + 1))
    raw.robot.data.root_link_vel_w = torch.full((1, 6), float(self.step_index + 1))
    raw.sim.data.time = raw.sim.data.time + STEP_DT
    raw.command.advance(raw.robot.data)
    raw.episode_length_buf += 1

    raw.termination_manager.update(self.step_index)
    terminated = raw.termination_manager.terminated.clone()
    truncated = raw.termination_manager.time_outs.clone()
    self.step_index += 1
    # The environment refreshes observations after the step: obs_{t+1}.
    self._obs = torch.full((1, OBS_DIM), float(self.step_index + 1))
    dones = (terminated | truncated).long()
    return {"actor": self._obs}, torch.zeros(1), dones, {}


def policy_from_obs(obs: Any) -> torch.Tensor:
  """Deterministic stub actor: the first action columns mirror the observation."""
  return obs["actor"][:, :NUM_JOINTS]


def make_joint_control(diag: Any) -> Any:
  # One record per joint, because each joint has its own gains and effort limit.
  records = [
    diag.PositionActuatorRecord(
      target_ids=(i,),
      ctrl_ids=(i,),
      kp=KP[i],
      kd=KD[i],
      effort_limit=EFFORT_LIMIT[i],
      delay_max_lag=0,
    )
    for i in range(NUM_JOINTS)
  ]
  return diag.build_joint_control(JOINT_NAMES, records)


def run_fake_rollout(diag: Any, **kwargs: Any) -> tuple[Any, Any]:
  steps = int(kwargs.pop("steps", 4))
  env = FakeVecEnv(**kwargs)
  joint_control = make_joint_control(diag)
  result = diag.record_policy_commands(
    wrapped=env,
    policy=policy_from_obs,
    command=env.unwrapped.command,
    action_term=env.unwrapped.action_term,
    joint_control=joint_control,
    steps=steps,
    motion_fps=1.0 / STEP_DT,
    start_frame=0,
  )
  return result, joint_control


##
# Layout helpers.
##


def test_build_joint_control_maps_records_and_flags_uncovered_joints(diag: Any) -> None:
  control = make_joint_control(diag)

  assert control.num_joints == NUM_JOINTS
  assert control.is_complete
  assert control.uncovered_joints == []
  assert control.kp.tolist() == list(KP)
  assert control.kd.tolist() == list(KD)
  assert control.effort_limit.tolist() == list(EFFORT_LIMIT)
  assert control.ctrl_index.tolist() == [0, 1, 2]


def test_build_joint_control_leaves_uncovered_joints_visible(diag: Any) -> None:
  records = [
    diag.PositionActuatorRecord(
      target_ids=(1,),
      ctrl_ids=(0,),
      kp=10.0,
      kd=1.0,
      effort_limit=None,
      delay_max_lag=3,
    )
  ]
  control = diag.build_joint_control(("a", "b", "c"), records)

  assert not control.is_complete
  assert control.uncovered_joints == ["a", "c"]
  assert control.ctrl_index.tolist() == [-1, 0, -1]
  assert bool(torch.isnan(control.kp[0])) and bool(torch.isinf(control.effort_limit[0]))
  assert control.kp[1] == 10.0
  assert control.delay_max_lag[1] == 3


def test_build_joint_control_rejects_mismatched_target_and_ctrl_counts(
  diag: Any,
) -> None:
  records = [
    diag.PositionActuatorRecord(
      target_ids=(0, 1),
      ctrl_ids=(0,),
      kp=1.0,
      kd=1.0,
      effort_limit=None,
      delay_max_lag=0,
    )
  ]
  with pytest.raises(ValueError, match="one ctrl index per target joint"):
    diag.build_joint_control(("a", "b"), records)


def test_broadcast_per_joint_accepts_scalar_vector_and_per_env(diag: Any) -> None:
  assert diag.broadcast_per_joint(0.5, 3).tolist() == [0.5, 0.5, 0.5]
  assert diag.broadcast_per_joint(torch.arange(3.0), 3).tolist() == [0.0, 1.0, 2.0]
  per_env = torch.tensor([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
  assert diag.broadcast_per_joint(per_env, 3).tolist() == pytest.approx([0.1, 0.2, 0.3])
  with pytest.raises(ValueError, match="differs between envs"):
    diag.broadcast_per_joint(torch.tensor([[0.1], [0.2]]), 3)
  with pytest.raises(ValueError, match="Expected a scalar or 3 values"):
    diag.broadcast_per_joint(torch.arange(4.0), 3)


##
# Action pipeline stages.
##


def test_reconstruct_helpers_distinguish_target_stages(diag: Any) -> None:
  raw = torch.tensor([[0.2, -0.4, 0.6]])
  scale = torch.tensor(ACTION_SCALE)
  offset = torch.tensor(ACTION_OFFSET)

  processed = diag.reconstruct_processed_target(raw, scale, offset)
  assert not torch.allclose(processed, raw)
  assert torch.allclose(processed, raw * scale + offset)

  bias = torch.tensor([ENCODER_BIAS])
  sim_target = diag.reconstruct_sim_target(processed, bias)
  assert not torch.allclose(sim_target, processed)
  assert torch.allclose(sim_target, processed - bias)

  torque = diag.reconstruct_pd_torque(
    sim_target,
    joint_pos=torch.zeros(1, NUM_JOINTS),
    joint_vel=torch.ones(1, NUM_JOINTS),
    kp=torch.tensor(KP),
    kd=torch.tensor(KD),
  )
  assert torch.allclose(torque, torch.tensor(KP) * sim_target - torch.tensor(KD))

  limit = torch.tensor([1.0, 1.0, 1.0])
  clamped = diag.clamp_effort(torque, limit)
  assert clamped.abs().max() <= 1.0
  assert torch.allclose(clamped, torch.clamp(torque, -1.0, 1.0))


##
# Rollout alignment, termination handling and bounds.
##


def test_record_policy_commands_pairs_inputs_before_the_step(diag: Any) -> None:
  result, joint_control = run_fake_rollout(diag, frames=50, steps=4)
  arrays = result.arrays

  assert result.steps_run == 4
  assert result.end_reason == "step_bound"
  assert arrays["actor_observation"].shape == (4, OBS_DIM)
  assert arrays["raw_action"].shape == (4, NUM_JOINTS)

  # obs_t (the value the action was computed from) is t + 1; the post-step
  # observation would be t + 2, so this catches post-step pairing.
  for step in range(4):
    assert np.allclose(arrays["actor_observation"][step], float(step + 1))
    assert np.allclose(arrays["raw_action"][step], float(step + 1))
    assert arrays["ref_frame"][step] == step
    assert arrays["policy_step"][step] == step
    assert arrays["sim_time_pre"][step] == pytest.approx(step * STEP_DT)
    assert arrays["sim_time_post"][step] == pytest.approx((step + 1) * STEP_DT)
    assert np.allclose(arrays["ref_joint_pos"][step], step * 0.1)
    assert np.allclose(arrays["ref_joint_vel"][step], step * 0.01)

  # Pre/post continuity: the post-state of one step is the pre-state of the next.
  for step in range(1, 4):
    assert np.allclose(
      arrays["joint_pos_pre"][step], arrays["joint_pos_post"][step - 1]
    )
    assert np.allclose(
      arrays["joint_vel_pre"][step], arrays["joint_vel_post"][step - 1]
    )
    assert arrays["episode_length_pre"][step] == step

  assert np.allclose(
    arrays["joint_pos_biased_pre"], arrays["joint_pos_pre"] + arrays["encoder_bias"]
  )


def test_record_policy_commands_logs_consistent_target_stages(diag: Any) -> None:
  result, joint_control = run_fake_rollout(diag, frames=50, steps=3)
  arrays = result.arrays

  scale = np.array(ACTION_SCALE)
  offset = np.array(ACTION_OFFSET)
  bias = np.array(ENCODER_BIAS)
  kp = np.array(KP)
  kd = np.array(KD)
  limit = np.array(EFFORT_LIMIT)

  processed = arrays["raw_action"] * scale + offset
  assert np.allclose(arrays["processed_target"], processed)
  assert np.allclose(arrays["sim_target"], processed - bias)
  # Encoder bias is nonzero in the fake env, so the stages must differ.
  assert not np.allclose(arrays["processed_target"], arrays["sim_target"])
  assert np.allclose(arrays["ctrl"], arrays["sim_target"])
  assert np.allclose(arrays["clipped_action"], arrays["raw_action"])

  requested = (
    kp * (arrays["sim_target"] - arrays["joint_pos_pre"]) - kd * arrays["joint_vel_pre"]
  )
  assert np.allclose(arrays["pd_torque_requested_pre"], requested)
  assert np.allclose(arrays["pd_torque_applied_pre"], np.clip(requested, -limit, limit))
  assert np.allclose(arrays["actuator_force_post"], arrays["qfrc_actuator_post"])
  assert joint_control.ctrl_index.tolist() == [0, 1, 2]
  # The recorded gains reproduce the fake actuator's torque law, which is the
  # same consistency check the diagnostic records for the native simulator.
  assert result.consistency["torque_law_post_max_abs_error"] == pytest.approx(
    0.0, abs=1e-3
  )
  assert result.consistency[
    "processed_target_from_clipped_action_max_abs_error"
  ] == pytest.approx(0.0, abs=1e-6)


def test_record_policy_commands_stops_at_termination_without_reset(diag: Any) -> None:
  env = FakeVecEnv(frames=50, terminate_at=2)
  joint_control = make_joint_control(diag)
  result = diag.record_policy_commands(
    wrapped=env,
    policy=policy_from_obs,
    command=env.unwrapped.command,
    action_term=env.unwrapped.action_term,
    joint_control=joint_control,
    steps=10,
    motion_fps=1.0 / STEP_DT,
  )
  arrays = result.arrays

  assert result.steps_run == 3
  assert result.end_reason == "terminated"
  assert "ee_body_pos" in result.end_detail
  assert result.termination_term_names == ["ee_body_pos", "time_out"]
  assert env.step_count == 3
  assert env.reset_count == 0
  assert arrays["terminated"].tolist() == [False, False, True]
  assert arrays["done"].tolist() == [False, False, True]
  assert arrays["truncated"].tolist() == [False, False, False]
  assert arrays["termination_flags"][2].tolist() == [True, False]
  # The terminal row keeps the terminal state; no post-reset observation leaks in.
  assert np.allclose(arrays["actor_observation"][-1], 3.0)
  assert result.achieved_ref_frame == 2
  assert result.achieved_reference_s == pytest.approx(2 * STEP_DT)


def test_record_policy_commands_respects_the_step_bound(diag: Any) -> None:
  env = FakeVecEnv(frames=50)
  result = diag.record_policy_commands(
    wrapped=env,
    policy=policy_from_obs,
    command=env.unwrapped.command,
    action_term=env.unwrapped.action_term,
    joint_control=make_joint_control(diag),
    steps=5,
    motion_fps=1.0 / STEP_DT,
  )

  assert result.steps_run == 5
  assert env.step_count == 5
  assert result.end_reason == "step_bound"
  assert result.achieved_ref_frame == 4
  assert result.arrays["sim_time_post"][-1] == pytest.approx(5 * STEP_DT)


def test_record_policy_commands_rejects_reference_desync(diag: Any) -> None:
  env = FakeVecEnv(frames=50)
  env.unwrapped.command.time_steps = torch.tensor([7])
  with pytest.raises(RuntimeError, match="Reference frame desynchronized"):
    diag.record_policy_commands(
      wrapped=env,
      policy=policy_from_obs,
      command=env.unwrapped.command,
      action_term=env.unwrapped.action_term,
      joint_control=make_joint_control(diag),
      steps=4,
      motion_fps=1.0 / STEP_DT,
    )


def test_check_motion_window_boundaries(diag: Any) -> None:
  # The final step's post-step frame must stay inside the motion, so equality is
  # already a wrap, not just an overflow.
  with pytest.raises(ValueError, match="start_frame \\+ steps < motion_frames"):
    diag.check_motion_window(2285, 1, 2286)
  with pytest.raises(ValueError, match="at most 0 steps"):
    diag.check_motion_window(2285, 1, 2286)
  with pytest.raises(ValueError, match="start_frame \\+ steps < motion_frames"):
    diag.check_motion_window(0, 2286, 2286)
  with pytest.raises(ValueError, match="start_frame \\+ steps < motion_frames"):
    diag.check_motion_window(2286, 1, 2286)
  with pytest.raises(ValueError, match="steps must be >= 1"):
    diag.check_motion_window(0, 0, 2286)

  # Safe neighbours: the last frame needs to remain reachable as a pre-step frame.
  diag.check_motion_window(2284, 1, 2286)
  diag.check_motion_window(0, 2285, 2286)
  diag.check_motion_window(0, 750, 2286)


def test_record_policy_commands_rejects_motion_wraparound(diag: Any) -> None:
  # frames=3 with 3 steps: the third step's post-step frame reaches the end of the
  # motion, which natively resamples and clears the targets.
  env = FakeVecEnv(frames=3)
  with pytest.raises(RuntimeError, match="Reference frame resampled after step 2"):
    diag.record_policy_commands(
      wrapped=env,
      policy=policy_from_obs,
      command=env.unwrapped.command,
      action_term=env.unwrapped.action_term,
      joint_control=make_joint_control(diag),
      steps=3,
      motion_fps=1.0 / STEP_DT,
    )
  assert env.unwrapped.command.wrap_count == 1
  assert env.step_count == 3


def test_record_policy_commands_reports_termination_when_wrapping(diag: Any) -> None:
  # A coincident physical termination must still be reported, not swallowed by the
  # wraparound invariant.
  env = FakeVecEnv(frames=3, terminate_at=2)
  with pytest.raises(RuntimeError) as excinfo:
    diag.record_policy_commands(
      wrapped=env,
      policy=policy_from_obs,
      command=env.unwrapped.command,
      action_term=env.unwrapped.action_term,
      joint_control=make_joint_control(diag),
      steps=3,
      motion_fps=1.0 / STEP_DT,
    )
  message = str(excinfo.value)
  assert "Reference frame resampled after step 2" in message
  assert "ee_body_pos" in message


def test_record_policy_commands_allows_the_last_safe_frame(diag: Any) -> None:
  # frames=4 with 3 steps ends on pre-frame 2 / post-frame 3, inside the motion.
  env = FakeVecEnv(frames=4)
  result = diag.record_policy_commands(
    wrapped=env,
    policy=policy_from_obs,
    command=env.unwrapped.command,
    action_term=env.unwrapped.action_term,
    joint_control=make_joint_control(diag),
    steps=3,
    motion_fps=1.0 / STEP_DT,
  )
  assert result.steps_run == 3
  assert result.end_reason == "step_bound"
  assert env.unwrapped.command.wrap_count == 0
  assert result.arrays["ref_frame"].tolist() == [0, 1, 2]


def test_record_policy_commands_clips_actions_and_reports_them(diag: Any) -> None:
  env = FakeVecEnv(frames=50, clip_actions=0.5)
  result = diag.record_policy_commands(
    wrapped=env,
    policy=policy_from_obs,
    command=env.unwrapped.command,
    action_term=env.unwrapped.action_term,
    joint_control=make_joint_control(diag),
    steps=3,
    motion_fps=1.0 / STEP_DT,
  )
  arrays = result.arrays

  assert np.allclose(arrays["raw_action"], [[1.0] * 3, [2.0] * 3, [3.0] * 3])
  assert np.allclose(arrays["clipped_action"], [[0.5] * 3, [0.5] * 3, [0.5] * 3])
  # The clipped action, and not the raw one, is what reached the environment.
  scale = np.array(ACTION_SCALE)
  assert np.allclose(
    arrays["processed_target"], arrays["clipped_action"] * scale + ACTION_OFFSET
  )


##
# Output artifacts.
##


def test_check_fresh_outputs_refuses_to_overwrite(diag: Any, tmp_path: Path) -> None:
  paths = diag.output_paths(tmp_path, with_onnx=True)
  assert set(paths) == {
    "policy_commands.npz",
    "policy_commands.csv",
    "actor_observations.csv",
    "metadata.json",
    "onnx_parity.json",
  }
  assert set(diag.output_paths(tmp_path, with_onnx=False)) == set(paths) - {
    "onnx_parity.json"
  }

  diag.check_fresh_outputs(paths)
  paths["policy_commands.csv"].write_text("step\n")
  with pytest.raises(FileExistsError, match="Refusing to overwrite"):
    diag.check_fresh_outputs(paths)


def test_write_outputs_schema_and_overwrite_protection(
  diag: Any, tmp_path: Path
) -> None:
  result, joint_control = run_fake_rollout(diag, frames=50, steps=3)
  out_dir = tmp_path / "recording"
  paths = diag.output_paths(out_dir, with_onnx=False)
  metadata = {"task": "fake-task", "overrides": []}
  written = diag.write_outputs(
    paths,
    result=result,
    joint_control=joint_control,
    metadata=metadata,
    scalars={"step_dt": STEP_DT},
  )

  assert all(path.exists() for path in written.values())
  with np.load(written["policy_commands.npz"]) as data:
    assert data["joint_names"].tolist() == list(JOINT_NAMES)
    assert data["actor_observation"].shape == (3, OBS_DIM)
    assert data["raw_action"].shape == (3, NUM_JOINTS)
    assert data["termination_flags"].shape == (3, 2)
    assert data["root_link_pose_w_pre"].shape == (3, 7)
    assert data["root_link_vel_w_post"].shape == (3, 6)
    assert data["kp"].tolist() == list(KP)
    assert data["episode_index"].tolist() == [0]
    assert float(data["step_dt"][0]) == pytest.approx(STEP_DT)
    for key in ("ref_joint_pos", "processed_target", "sim_target", "ctrl"):
      assert data[key].shape == (3, NUM_JOINTS)

  with written["policy_commands.csv"].open() as handle:
    rows = handle.read().splitlines()
  assert len(rows) == 1 + 3 * NUM_JOINTS
  assert rows[0] == ",".join(diag.CSV_COLUMNS)
  assert len(rows[1].split(",")) == len(diag.CSV_COLUMNS)

  with written["actor_observations.csv"].open() as handle:
    obs_rows = handle.read().splitlines()
  assert obs_rows[0] == ",".join(
    ["step", "sim_time_pre", "ref_frame"] + [f"obs_{i:03d}" for i in range(OBS_DIM)]
  )
  assert len(obs_rows) == 4

  with written["metadata.json"].open() as handle:
    assert json.load(handle)["task"] == "fake-task"

  with pytest.raises(FileExistsError, match="Refusing to overwrite"):
    diag.write_outputs(
      paths,
      result=result,
      joint_control=joint_control,
      metadata=metadata,
      scalars={},
    )


def test_summarize_reports_target_deviation_and_torque_use(diag: Any) -> None:
  result, joint_control = run_fake_rollout(diag, frames=50, steps=3)
  summary = diag.summarize(result, joint_control)

  knee = summary["left_knee_joint"]
  expected = np.degrees(
    result.arrays["sim_target"][:, 1] - result.arrays["ref_joint_pos"][:, 1]
  )
  assert knee["target_minus_ref_deg_mean"] == pytest.approx(expected.mean())
  assert knee["torque_utilization_max"] == pytest.approx(
    np.abs(result.arrays["qfrc_actuator_post"][:, 1]).max() / EFFORT_LIMIT[1]
  )


##
# Motion window preflight (before environment construction).
##


def _write_motion(path: Path, frames: int) -> None:
  """Write a minimal tracking-format motion npz."""
  np.savez(
    path,
    joint_pos=np.zeros((frames, NUM_JOINTS), dtype=np.float32),
    joint_vel=np.zeros((frames, NUM_JOINTS), dtype=np.float32),
    body_pos_w=np.zeros((frames, 1, 3), dtype=np.float32),
    body_quat_w=np.tile(
      np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (frames, 1, 1)
    ),
    body_lin_vel_w=np.zeros((frames, 1, 3), dtype=np.float32),
    body_ang_vel_w=np.zeros((frames, 1, 3), dtype=np.float32),
    fps=np.array([1.0 / STEP_DT], dtype=np.float32),
  )


def test_main_rejects_wrapping_window_before_env_construction(
  diag: Any, tmp_path: Path
) -> None:
  """The motion-end preflight must fail before an environment is built.

  With a 3-frame motion and ``start_frame=2``, one step would land the post-step
  reference frame on the end of the motion. ``main`` must raise the preflight
  error; if it ever ran later, this test would have to build a simulator first.
  """
  motion_path = tmp_path / "motion.npz"
  _write_motion(motion_path, frames=3)
  checkpoint_path = tmp_path / "model_0.pt"
  checkpoint_path.write_bytes(b"not a real checkpoint")

  with pytest.raises(ValueError, match="start_frame \\+ steps < motion_frames"):
    diag.main(
      checkpoint_file=str(checkpoint_path),
      output_dir=str(tmp_path / "out"),
      motion_file=str(motion_path),
      start_frame=2,
      max_steps=1,
    )
  assert not (tmp_path / "out").exists()


##
# Saved-config comparison and overrides.
##


def test_compare_saved_action_scale_reports_unverifiable_cases(diag: Any) -> None:
  names = JOINT_NAMES
  effective = torch.tensor(ACTION_SCALE)

  matched = diag.compare_saved_action_scale(
    {"left_hip_pitch_joint": ACTION_SCALE[0], ".*_knee_joint": ACTION_SCALE[1]},
    names,
    effective,
  )
  assert matched == pytest.approx(0.0)

  mismatched = diag.compare_saved_action_scale(
    {"left_hip_pitch_joint": 9.9, ".*_knee_joint": ACTION_SCALE[1]}, names, effective
  )
  assert mismatched == pytest.approx(abs(9.9 - ACTION_SCALE[0]))

  assert diag.compare_saved_action_scale(None, names, effective) is None
  assert (
    diag.compare_saved_action_scale({"left_hip_pitch_joint": 0.35}, names, effective)
    is None
  )
  assert diag.compare_saved_action_scale({"nope_joint": 0.35}, names, effective) is None


def test_apply_nominal_overrides_records_and_preserves_terminations(diag: Any) -> None:
  env_cfg = load_env_cfg(diag.DEFAULT_TASK, play=False)
  terminations_before = list(env_cfg.terminations)
  events_before = list(env_cfg.events)
  assert "push_robot" in events_before
  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)

  overrides = diag.apply_nominal_overrides(
    env_cfg,
    motion_file="motion.npz",
    saved_lookahead_s=0.0,
    horizon_s=16.0,
    seed=42,
  )
  fields = {override.field for override in overrides}

  assert env_cfg.auto_reset is False
  assert env_cfg.episode_length_s == 16.0
  assert motion_cmd.motion_file == "motion.npz"
  assert motion_cmd.sampling_mode == "start"
  assert motion_cmd.pose_range == {}
  assert motion_cmd.velocity_range == {}
  assert motion_cmd.joint_position_range == (0.0, 0.0)
  assert env_cfg.observations["actor"].enable_corruption is False
  assert env_cfg.events == {}
  assert env_cfg.scene.num_envs == 1
  assert env_cfg.seed == 42
  assert set(env_cfg.terminations) == set(terminations_before)
  for field in (
    "commands.motion.motion_file",
    "commands.motion.lookahead_s",
    "commands.motion.sampling_mode",
    "commands.motion.pose_range",
    "commands.motion.velocity_range",
    "commands.motion.joint_position_range",
    "observations.actor.enable_corruption",
    "events",
    "episode_length_s",
    "auto_reset",
    "scene.num_envs",
    "seed",
  ):
    assert field in fields
  assert any(field.startswith("commands.motion") for field in fields)
  events_override = [o for o in overrides if o.field == "events"]
  assert events_override, "the events override must be recorded"
  # The recorded "before" text is truncated, but it names a removed term.
  assert "push_robot" in events_override[0].before
  assert bool(env_cfg.terminations["time_out"].time_out)


##
# ONNX parity helper.
##


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_run_onnx_parity_matches_a_linear_export(diag: Any, tmp_path: Path) -> None:
  ort = pytest.importorskip("onnxruntime")
  del ort
  from tensordict import TensorDict

  class Linear(torch.nn.Module):
    def __init__(self) -> None:
      super().__init__()
      self.linear = torch.nn.Linear(OBS_DIM, NUM_JOINTS)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
      return self.linear(obs)

  module = Linear()
  module.eval()
  onnx_file = tmp_path / "policy.onnx"
  torch.onnx.export(
    module,
    (torch.zeros(1, OBS_DIM),),
    str(onnx_file),
    input_names=["obs"],
    output_names=["actions"],
    opset_version=18,
    dynamo=False,
  )

  steps = 6
  observations = (
    np.random.default_rng(0).normal(size=(steps, OBS_DIM)).astype(np.float32)
  )

  def policy(obs: TensorDict) -> torch.Tensor:
    return module(obs["actor"])

  result = SimpleNamespace(
    arrays={
      "actor_observation": observations,
      "raw_action": np.stack(
        [
          module(torch.from_numpy(observations[i : i + 1])).detach().numpy()[0]
          for i in range(steps)
        ]
      ),
    }
  )
  parity = diag.run_onnx_parity(
    onnx_file=onnx_file,
    result=result,
    policy=policy,
    device="cpu",
    max_samples=4,
  )

  assert parity["samples_tested"] == 4
  assert parity["sample_indices"] == [0, 2, 3, 5]
  assert parity["action_dim"] == NUM_JOINTS
  assert parity["inputs"][0]["name"] == "obs"
  assert parity["outputs"][0]["name"] == "actions"
  assert parity["max_abs_error"] <= parity["tolerance"]
  assert parity["within_tolerance"] is True
  assert parity["torch_recompute_max_abs_error"] == pytest.approx(0.0, abs=1e-6)


def test_run_onnx_parity_rejects_multi_input_models(diag: Any, tmp_path: Path) -> None:
  pytest.importorskip("onnxruntime")
  import onnx

  graph = onnx.helper.make_graph(
    nodes=[onnx.helper.make_node("Identity", ["time_step"], ["actions"])],
    name="two_inputs",
    inputs=[
      onnx.helper.make_tensor_value_info("obs", onnx.TensorProto.FLOAT, [1, OBS_DIM]),
      onnx.helper.make_tensor_value_info("time_step", onnx.TensorProto.FLOAT, [1, 1]),
    ],
    outputs=[
      onnx.helper.make_tensor_value_info("actions", onnx.TensorProto.FLOAT, [1, 1])
    ],
  )
  model = onnx.helper.make_model(
    graph,
    opset_imports=[onnx.helper.make_opsetid("", 17)],
  )
  onnx_file = tmp_path / "wrapped.onnx"
  onnx.save(model, onnx_file)

  result = SimpleNamespace(
    arrays={
      "actor_observation": np.zeros((1, OBS_DIM), dtype=np.float32),
      "raw_action": np.zeros((1, NUM_JOINTS), dtype=np.float32),
    }
  )
  with pytest.raises(RuntimeError, match="single ONNX input"):
    diag.run_onnx_parity(
      onnx_file=onnx_file,
      result=result,
      policy=policy_from_obs,
      device="cpu",
      max_samples=1,
    )


##
# Saved-versus-effective provenance.
##


def _g1_joint_names() -> tuple[str, ...]:
  """Joint names of the mode-15 robot in the canonical entity order."""
  import mujoco

  from mjlab.asset_zoo.robots import get_g1_29dof_mode_15_robot_cfg

  spec = get_g1_29dof_mode_15_robot_cfg().spec_fn()
  return tuple(
    joint.name.split("/")[-1]
    for joint in spec.joints
    if joint.type != mujoco.mjtJoint.mjJNT_FREE
  )


def _provenance_inputs(diag: Any) -> tuple[Any, Any, dict]:
  """Real task config, matching joint control and the resolved action scale."""
  from mjlab.utils.lab_api.string import resolve_matching_names_values

  cfg = load_env_cfg(diag.DEFAULT_TASK, play=False)
  joint_names = _g1_joint_names()
  joint_control = diag.build_joint_control(
    joint_names,
    [
      diag.PositionActuatorRecord(
        target_ids=(i,),
        ctrl_ids=(i,),
        kp=1.0,
        kd=1.0,
        effort_limit=1.0,
        delay_max_lag=0,
      )
      for i in range(len(joint_names))
    ],
  )
  action_cfg = cfg.actions["joint_pos"]
  assert isinstance(action_cfg, JointPositionActionCfg)
  scale_cfg = action_cfg.scale
  assert isinstance(scale_cfg, dict)
  index_list, _, value_list = resolve_matching_names_values(
    scale_cfg, list(joint_names)
  )
  action_scale = torch.ones(len(joint_names))
  action_scale[index_list] = torch.tensor(value_list)
  return cfg, joint_control, {"action_scale": action_scale}


def _saved_from_effective(cfg: Any) -> dict:
  """A saved-config dict whose entries mirror the loaded config.

  Mirrors the structure of ``params/env.yaml``, including
  ``actions.joint_pos.clip: null``: a key that is present with a null value is a
  real saved value, unlike a key that is absent.
  """
  from dataclasses import asdict

  actuators = list(cfg.scene.entities["robot"].articulation.actuators)
  action_cfg = cfg.actions["joint_pos"]
  assert isinstance(action_cfg, JointPositionActionCfg)
  return {
    "decimation": cfg.decimation,
    "sim": {"mujoco": {"timestep": cfg.sim.mujoco.timestep}},
    "commands": {"motion": {"lookahead_s": cfg.commands["motion"].lookahead_s}},
    "scene": {
      "entities": {
        "robot": {"articulation": {"actuators": [asdict(a) for a in actuators]}}
      }
    },
    "actions": {
      "joint_pos": {
        "scale": action_cfg.scale,
        "clip": None,
        "use_default_offset": action_cfg.use_default_offset,
      }
    },
  }


def _statuses(provenance: dict) -> dict[str, str]:
  checks = provenance["fidelity_checks"]
  statuses = {
    key: check["status"]
    for key, check in checks.items()
    if isinstance(check, dict) and "status" in check
  }
  settings = checks["actuator_settings"]["settings"]
  statuses.update({f"actuator.{key}": c["status"] for key, c in settings.items()})
  return statuses


def test_resolve_actuator_setting_per_joint_maps_patterns_and_rejects_ambiguity(
  diag: Any,
) -> None:
  names = ("left_knee_joint", "right_knee_joint", "waist_yaw_joint")
  entries = [
    {"target_names_expr": (".*_knee_joint",), "stiffness": 99.0},
    {"target_names_expr": ("waist_yaw_joint",), "stiffness": 40.0},
  ]
  values, detail = diag.resolve_actuator_setting_per_joint(entries, names, "stiffness")
  assert values == [99.0, 99.0, 40.0]
  assert "one value per joint" in detail

  # Uncovered joint.
  values, detail = diag.resolve_actuator_setting_per_joint(
    entries[:1], names, "stiffness"
  )
  assert values is None and "uncovered joints: ['waist_yaw_joint']" in detail

  # A joint matched by two entries is ambiguous, never last-wins.
  ambiguous = [*entries, {"target_names_expr": (".*_knee_joint",), "stiffness": 1.0}]
  values, detail = diag.resolve_actuator_setting_per_joint(
    ambiguous, names, "stiffness"
  )
  assert values is None and "matched more than once" in detail

  # Missing, non-numeric and unsupported settings.
  values, detail = diag.resolve_actuator_setting_per_joint(
    [{"target_names_expr": (".*",)}], names, "stiffness"
  )
  assert values is None and "does not specify 'stiffness'" in detail
  values, detail = diag.resolve_actuator_setting_per_joint(
    [{**entries[0], "stiffness": None}, entries[1]], names, "stiffness"
  )
  assert values is None and "unspecified (null)" in detail
  values, detail = diag.resolve_actuator_setting_per_joint(
    [{**entries[0], "stiffness": "fast"}, entries[1]], names, "stiffness"
  )
  assert values is None and "non-numeric" in detail
  values, detail = diag.resolve_actuator_setting_per_joint(None, names, "stiffness")
  assert values is None and "no actuator entries" in detail
  assert diag.resolve_actuator_setting_per_joint([], names, "stiffness")[0] is None

  # A missing effective side makes every setting unverifiable, never a match.
  checks = diag.compare_saved_actuator_settings(
    entries, names, None, "no effective actuator list"
  )
  assert set(checks) == set(diag.ACTUATOR_SETTING_KEYS)
  assert all(check.status == "unverifiable" for check in checks.values())
  assert all("no effective actuator list" in check.detail for check in checks.values())

  # effort_limit: None means unbounded and is compared as inf.
  unlimited = [
    {"target_names_expr": (".*",), "effort_limit": None},
  ]
  values, _ = diag.resolve_actuator_setting_per_joint(unlimited, names, "effort_limit")
  assert values == [float("inf")] * 3


@pytest.mark.parametrize("patterns", [(".*", "knee"), ("missing",), ("[",)])
def test_compare_actuator_settings_reports_unresolvable_patterns(
  diag: Any, patterns: tuple[str, ...]
) -> None:
  valid = {
    "target_names_expr": ("knee",),
    "stiffness": 1.0,
    "damping": 1.0,
    "effort_limit": 139.0,
    "delay_min_lag": 0,
    "delay_max_lag": 0,
  }
  invalid = {**valid, "target_names_expr": patterns}
  for saved, effective in (([invalid], [valid]), ([valid], [invalid])):
    checks = diag.compare_saved_actuator_settings(saved, ("knee",), effective, "")
    assert all(check.status == "unverifiable" for check in checks.values())
    assert all(
      "unresolved target patterns" in check.detail for check in checks.values()
    )


@pytest.mark.parametrize(
  ("saved_limit", "effective_limit", "status", "difference"),
  [
    (None, None, "match", 0.0),
    (139.0, None, "mismatch", float("inf")),
    (None, 139.0, "mismatch", float("inf")),
    (139.0, 139.0, "match", 0.0),
    (139.0, 138.0, "mismatch", 1.0),
  ],
)
def test_compare_actuator_settings_handles_unbounded_limits(
  diag: Any,
  saved_limit: float | None,
  effective_limit: float | None,
  status: str,
  difference: float,
) -> None:
  entry = {
    "target_names_expr": ("knee",),
    "stiffness": 1.0,
    "damping": 1.0,
    "effort_limit": saved_limit,
    "delay_min_lag": 0,
    "delay_max_lag": 0,
  }
  checks = diag.compare_saved_actuator_settings(
    [entry], ("knee",), [{**entry, "effort_limit": effective_limit}], ""
  )
  check = checks["effort_limit"]
  assert check.status == status
  assert check.max_abs_diff == difference
  assert check.differing_joints == (() if status == "match" else ("knee",))
  assert all(c.status == "match" for key, c in checks.items() if key != "effort_limit")


@pytest.mark.parametrize(
  ("key", "value"),
  [("stiffness", float("nan")), ("damping", float("inf"))],
)
def test_compare_actuator_settings_rejects_nonfinite_parameters(
  diag: Any, key: str, value: float
) -> None:
  entry = {
    "target_names_expr": ("knee",),
    "stiffness": 1.0,
    "damping": 1.0,
    "effort_limit": 139.0,
    "delay_min_lag": 0,
    "delay_max_lag": 0,
  }
  check = diag.compare_saved_actuator_settings(
    [{**entry, key: value}], ("knee",), [entry], ""
  )[key]
  assert check.status == "unverifiable"
  assert "non-finite" in check.detail


def test_collect_saved_provenance_compares_actuator_settings(
  diag: Any, tmp_path: Path
) -> None:
  cfg, joint_control, extra = _provenance_inputs(diag)
  saved = _saved_from_effective(cfg)
  checkpoint = tmp_path / "model_0.pt"
  checkpoint.write_bytes(b"x")

  provenance = diag.collect_saved_provenance(
    checkpoint, saved, cfg, joint_control, extra["action_scale"]
  )
  statuses = _statuses(provenance)
  assert statuses["actuator.stiffness"] == "match"
  assert statuses["actuator.damping"] == "match"
  assert statuses["actuator.effort_limit"] == "match"
  assert statuses["actuator.delay_min_lag"] == "match"
  assert statuses["actuator.delay_max_lag"] == "match"
  assert statuses["action_scale"] == "match"
  assert statuses["decimation"] == "match"
  assert statuses["physics_dt"] == "match"
  assert statuses["lookahead_s"] == "match"
  # A key present with a null value is compared, not treated as absent.
  assert statuses["action_clip"] == "match"
  clip = provenance["fidelity_checks"]["action_clip"]
  assert clip["saved"] is None and clip["effective"] is None
  assert "null values" in clip["detail"]
  assert statuses["use_default_offset"] == "match"
  settings = provenance["fidelity_checks"]["actuator_settings"]
  assert settings["unverifiable_settings"] == []
  assert settings["mismatched_settings"] == []
  stiffness = settings["settings"]["stiffness"]
  assert stiffness["max_abs_diff"] == 0.0
  assert len(stiffness["saved"]) == joint_control.num_joints
  assert len(stiffness["effective"]) == joint_control.num_joints

  # A single changed saved gain is reported as a mismatch naming the joints.
  differing = _saved_from_effective(cfg)
  entries = differing["scene"]["entities"]["robot"]["articulation"]["actuators"]
  knee_group = next(
    group for group in entries if ".*_knee_joint" in group["target_names_expr"]
  )
  original = knee_group["stiffness"]
  knee_group["stiffness"] = original + 1.0
  provenance = diag.collect_saved_provenance(
    checkpoint, differing, cfg, joint_control, extra["action_scale"]
  )
  settings = provenance["fidelity_checks"]["actuator_settings"]
  assert settings["settings"]["stiffness"]["status"] == "mismatch"
  # The knees share an actuator group with hip pitch/roll in the mode-15 robot.
  assert set(settings["settings"]["stiffness"]["differing_joints"]) == {
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_knee_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_knee_joint",
  }
  assert settings["settings"]["stiffness"]["max_abs_diff"] == pytest.approx(1.0)
  assert settings["mismatched_settings"] == ["stiffness"]
  assert settings["settings"]["damping"]["status"] == "match"


def test_collect_saved_provenance_without_saved_settings_is_unverifiable(
  diag: Any, tmp_path: Path
) -> None:
  cfg, joint_control, extra = _provenance_inputs(diag)
  checkpoint = tmp_path / "model_0.pt"
  checkpoint.write_bytes(b"x")

  provenance = diag.collect_saved_provenance(
    checkpoint, {}, cfg, joint_control, extra["action_scale"]
  )
  statuses = _statuses(provenance)
  assert set(statuses.values()) == {"unverifiable"}
  text = json.dumps(provenance)
  assert '"match"' not in text
  actuator_settings = provenance["fidelity_checks"]["actuator_settings"]["settings"]
  assert set(actuator_settings) == {
    "stiffness",
    "damping",
    "effort_limit",
    "delay_min_lag",
    "delay_max_lag",
  }
  for check in actuator_settings.values():
    assert check["saved"] is None
    assert check["max_abs_diff"] is None
  assert provenance["fidelity_checks"]["actuator_settings"][
    "unverifiable_settings"
  ] == list(actuator_settings)
  assert "no saved actuator entries" in actuator_settings["stiffness"]["detail"]


def test_collect_saved_provenance_marks_partial_and_unsupported_unverifiable(
  diag: Any, tmp_path: Path
) -> None:
  cfg, joint_control, extra = _provenance_inputs(diag)
  checkpoint = tmp_path / "model_0.pt"
  checkpoint.write_bytes(b"x")

  # Partial coverage: keep only the knee group, so other joints are uncovered.
  partial = _saved_from_effective(cfg)
  entries = partial["scene"]["entities"]["robot"]["articulation"]["actuators"]
  partial["scene"]["entities"]["robot"]["articulation"]["actuators"] = [
    group for group in entries if ".*_knee_joint" in group["target_names_expr"]
  ]
  provenance = diag.collect_saved_provenance(
    checkpoint, partial, cfg, joint_control, extra["action_scale"]
  )
  for check in provenance["fidelity_checks"]["actuator_settings"]["settings"].values():
    assert check["status"] == "unverifiable"
    assert check["max_abs_diff"] is None
    assert "uncovered joints" in check["detail"]

  # Unsupported shape: a mapping instead of a list of actuator entries.
  unsupported = _saved_from_effective(cfg)
  unsupported["scene"]["entities"]["robot"]["articulation"]["actuators"] = {
    "stiffness": 1.0
  }
  provenance = diag.collect_saved_provenance(
    checkpoint, unsupported, cfg, joint_control, extra["action_scale"]
  )
  text = json.dumps(provenance)
  for check in provenance["fidelity_checks"]["actuator_settings"]["settings"].values():
    assert check["status"] == "unverifiable"
    assert "unsupported type: dict" in check["detail"]
  # Nothing internal leaks into the report (it must stay plain JSON data).
  assert "object object" not in text and "unverifiable" in text


def test_collect_saved_provenance_does_not_claim_unperformed_comparisons(
  diag: Any, tmp_path: Path
) -> None:
  cfg, joint_control, extra = _provenance_inputs(diag)
  checkpoint = tmp_path / "model_0.pt"
  checkpoint.write_bytes(b"x")
  provenance = diag.collect_saved_provenance(
    checkpoint, _saved_from_effective(cfg), cfg, joint_control, extra["action_scale"]
  )

  notes = provenance["notes"]
  assert "unverifiable" in notes
  assert "no agent setting is compared" in notes
  # The saved env.yaml stores clip as null, which is compared as a real value
  # rather than being reported as a match that was never performed.
  assert provenance["fidelity_checks"]["action_clip"]["status"] == "match"
  assert provenance["saved_values_not_compared"] == [
    "init_weight_s",
    "sampling_mode",
    "seed",
  ]
  delay = provenance["fidelity_checks"]["effective_command_delay"]
  assert "effective (loaded) actuator config only" in delay["detail"]
  assert (
    "saved" not in delay or "comparison is under actuator_settings" in delay["detail"]
  )
