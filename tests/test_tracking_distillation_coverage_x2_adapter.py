"""CPU fake-bridge tests for the explicit X2 branch-state adapter."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from mjlab.actuator.builtin_actuator import BuiltinPositionActuator
from mjlab.envs.mdp.actions import JointPositionAction as RealJointPositionAction
from mjlab.sensor.contact_sensor import ContactSensor
from mjlab.sim.sim_data import TorchArray
from mjlab.tasks.tracking.distillation.branch_state import (
  DYNAMIC_FIELDS,
  MODEL_FIELDS,
  BranchState,
  BranchStateError,
  BranchStateMismatch,
)
from mjlab.tasks.tracking.distillation.coverage_environment import (
  CoverageEnvironmentComponent,
  CoverageRuntimeError,
  TerminalCapture,
)
from mjlab.tasks.tracking.distillation.coverage_policy import CoveragePolicy
from mjlab.tasks.tracking.distillation.multi_motion import (
  MultiMotionCommand as RealMultiMotionCommand,
)
from mjlab.tasks.tracking.distillation.x2_state_adapter import (
  X2FixedTopologyStateAdapter,
)
from mjlab.utils.buffers import CircularBuffer, DelayBuffer


def make_multi_motion_command(num_envs: int) -> RealMultiMotionCommand:
  command = object.__new__(RealMultiMotionCommand)
  command.motion_ids = torch.arange(num_envs, dtype=torch.long)
  command.time_steps = torch.arange(num_envs, dtype=torch.long) + 2
  command._row_lengths = torch.full((num_envs,), 100, dtype=torch.long)
  command._teacher_codes = torch.arange(num_envs, dtype=torch.long) + 10
  command._row_lookahead_steps = torch.ones(num_envs, dtype=torch.long)
  command._row_bin_counts = torch.full((num_envs,), 20, dtype=torch.long)
  command.generation_ids = torch.arange(num_envs, dtype=torch.long) + 3
  command.segment_ids = torch.arange(num_envs, dtype=torch.long) + 5
  command.time_left = torch.full((num_envs,), 4.0)
  command.command_counter = torch.arange(num_envs, dtype=torch.long)
  command.body_pos_relative_w = torch.randn(num_envs, 2, 3)
  command.body_quat_relative_w = torch.zeros(num_envs, 2, 4)
  command.body_quat_relative_w[..., 0] = 1.0
  command.metrics = {
    "error_anchor_pos": torch.arange(num_envs, dtype=torch.float32),
    "standing_start": torch.zeros(num_envs),
  }
  command._pending_forward = False
  command._pending_reset_frame_ids = torch.tensor([0], dtype=torch.long)
  command._reset_resample_ids = torch.tensor([2], dtype=torch.long)
  command._timer_resample_ids = torch.tensor([], dtype=torch.long)
  command._wrap_resample_ids = torch.tensor([3], dtype=torch.long)
  command._event_available = torch.zeros(num_envs, dtype=torch.bool)
  command._event_completed = torch.zeros(num_envs, dtype=torch.bool)
  command._event_interrupted = torch.zeros(num_envs, dtype=torch.bool)
  command._event_completed_generation = torch.full((num_envs,), -1, dtype=torch.long)
  command._event_interrupted_generation = torch.full((num_envs,), -1, dtype=torch.long)
  command._event_reasons = ["unavailable"] * num_envs
  command._reset_generator = torch.Generator(device="cpu")
  command._reset_generator.manual_seed(11)
  return command


def make_joint_position_action(robot, num_envs: int) -> RealJointPositionAction:
  term = object.__new__(RealJointPositionAction)
  term._entity = robot
  term._raw_actions = torch.randn(num_envs, 2)
  term._processed_actions = torch.randn(num_envs, 2)
  term._target_ids = torch.tensor([0, 1], dtype=torch.long)
  term._target_names = ["left", "right"]
  term._scale = 0.5
  term._offset = torch.zeros(num_envs, 2)
  return term


def make_contact_sensor(num_envs: int) -> ContactSensor:
  sensor: Any = object.__new__(ContactSensor)
  sensor._air_time_state = SimpleNamespace(
    current_air_time=torch.ones(num_envs, 1),
    last_air_time=torch.ones(num_envs, 1) * 2,
    current_contact_time=torch.ones(num_envs, 1) * 3,
    last_contact_time=torch.ones(num_envs, 1) * 4,
  )
  sensor._history_state = {"force": torch.randn(num_envs, 1, 2, 3)}
  return cast(ContactSensor, sensor)


def _torch_array(value: torch.Tensor) -> TorchArray:
  result = object.__new__(TorchArray)
  object.__setattr__(result, "_tensor", value)
  return result


class FakeActuator:
  def __new__(cls, delay, index: int):
    actuator = object.__new__(BuiltinPositionActuator)
    actuator._delay_buffer = delay
    actuator._target_names = [f"joint_{index}"]
    actuator._ctrl_ids = torch.tensor([index], dtype=torch.long)
    return actuator


class FakeSim:
  def __init__(self, num_envs: int):
    self.expanded_fields = {"body_mass", "body_inertia"}
    self.data = SimpleNamespace()
    self.model = SimpleNamespace()
    self.recompute_calls = []
    self.forward_calls = 0
    for name in DYNAMIC_FIELDS:
      if name == "qpos":
        value = torch.zeros(num_envs, 7)
        value[:, 6] = 1.0
      elif name == "qvel":
        value = torch.zeros(num_envs, 6)
      elif name == "mocap_pos":
        value = torch.zeros(num_envs, 1, 3)
      elif name == "mocap_quat":
        value = torch.zeros(num_envs, 1, 4)
        value[..., 0] = 1.0
      else:
        value = torch.randn(num_envs, 2)
      setattr(self.data, name, value)
    self.data.sensordata = torch.randn(num_envs, 2)
    self.data.qpos = _torch_array(self.data.qpos)
    self.data.qacc_warmstart = _torch_array(self.data.qacc_warmstart)
    for name in MODEL_FIELDS:
      value = (
        torch.randn(num_envs, 2) if name in self.expanded_fields else torch.randn(2)
      )
      setattr(self.model, name, _torch_array(value) if name == "body_mass" else value)

  def recompute_constants(self, level):
    self.recompute_calls.append(level)

  def forward(self):
    self.forward_calls += 1
    self.data.qacc_warmstart._tensor.fill_(123.0)


class FakeRobotData:
  def __init__(self, num_envs: int):
    self.indexing = SimpleNamespace(free_joint_q_adr=torch.arange(7, dtype=torch.long))
    self.projected_gravity_b = torch.zeros(num_envs, 3)
    self.projected_gravity_b[:, 2] = 1.0
    fields = (
      "encoder_bias",
      "default_root_state",
      "default_joint_pos",
      "default_joint_vel",
      "default_joint_pos_limits",
      "joint_pos_limits",
      "soft_joint_pos_limits",
      "gravity_vec_w",
      "forward_vec_b",
      "joint_pos_target",
      "joint_vel_target",
      "joint_effort_target",
      "tendon_len_target",
      "tendon_vel_target",
      "tendon_effort_target",
      "site_effort_target",
    )
    for name in fields:
      width = 13 if name == "default_root_state" else 2
      setattr(self, name, torch.randn(num_envs, width))


class FakeRobot:
  def __init__(self, num_envs: int):
    self.data = FakeRobotData(num_envs)
    delay = DelayBuffer(
      min_lag=0,
      max_lag=2,
      batch_size=num_envs,
      generator=torch.Generator(device="cpu").manual_seed(31),
    )
    delay.append(torch.arange(num_envs, dtype=torch.float32).reshape(num_envs, 1))
    self.actuators = [FakeActuator(delay, 0), FakeActuator(delay, 1)]


class FakeObservationManager:
  def __init__(self, num_envs: int, obs_buf):
    self._obs_buffer = obs_buf
    self._group_obs_term_names = {
      "actor": [
        "command",
        "motion_anchor_pos_b",
        "motion_anchor_ori_b",
        "base_lin_vel",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "actions",
      ]
    }
    self._group_obs_term_dim = {
      "actor": [(62,), (3,), (6,), (3,), (3,), (31,), (31,), (31,)]
    }
    self._group_obs_concatenate = {"actor": True}
    self._group_obs_class_instances = {"actor": {}}
    term_delay = DelayBuffer(
      min_lag=0,
      max_lag=1,
      batch_size=num_envs,
      generator=torch.Generator(device="cpu").manual_seed(32),
    )
    term_delay.append(torch.zeros(num_envs, 1))
    group_delay = DelayBuffer(
      min_lag=0,
      max_lag=1,
      batch_size=num_envs,
      generator=torch.Generator(device="cpu").manual_seed(33),
    )
    group_delay.append(torch.zeros(num_envs, 1))
    history = CircularBuffer(max_len=2, batch_size=num_envs, device="cpu")
    history.append(torch.zeros(num_envs, 1))
    self._group_obs_term_delay_buffer = {"actor": {"joint_pos": term_delay}}
    self._group_obs_delay_group_buffer = {"actor": {"encoder_packet": group_delay}}
    self._group_obs_term_history_buffer = {"actor": {"joint_vel": history}}

  def cached_observations(self):
    return {name: value.clone() for name, value in self._obs_buffer.items()}


class FakeEventManager:
  def __init__(self, num_envs: int):
    self._mode_term_cfgs = {}
    self._interval_term_time_left = [torch.arange(num_envs, dtype=torch.float32)]
    self._reset_term_last_triggered_step_id = [
      torch.arange(num_envs, dtype=torch.int32)
    ]
    self._reset_term_last_triggered_once = [torch.zeros(num_envs, dtype=torch.bool)]


class FakeCommandManager:
  def __init__(self, command):
    self.command = command

  def get_term(self, name):
    assert name == "motion"
    return self.command


class FakeScene:
  def __init__(self, num_envs: int):
    self.env_origins = torch.arange(num_envs * 3, dtype=torch.float32).reshape(
      num_envs, 3
    )
    self.entities = {"robot": FakeRobot(num_envs)}
    self.sensors = {"self_collision": make_contact_sensor(num_envs)}

  def collect_variant_info(self):
    return []


class FakeEnv:
  def __init__(self, num_envs: int):
    self.num_envs = num_envs
    self.sim = FakeSim(num_envs)
    self.scene = FakeScene(num_envs)
    actor = torch.arange(num_envs * 170, dtype=torch.float32).reshape(num_envs, 170)
    self.obs_buf = {"actor": actor, "critic": torch.zeros(num_envs, 170)}
    self.observation_manager = FakeObservationManager(num_envs, self.obs_buf)
    self.action_manager = SimpleNamespace(
      _action=torch.randn(num_envs, 2),
      _prev_action=torch.randn(num_envs, 2),
      _prev_prev_action=torch.randn(num_envs, 2),
    )
    robot = self.scene.entities["robot"]
    self.action_manager._terms = {
      "joint_pos": make_joint_position_action(robot, num_envs)
    }
    self.command_manager = FakeCommandManager(make_multi_motion_command(num_envs))
    self.event_manager = FakeEventManager(num_envs)
    self._sim_step_counter = 17
    self.common_step_counter = 9
    self.episode_length_buf = torch.arange(num_envs, dtype=torch.long)
    self._manual_reset_pending = torch.zeros(num_envs, dtype=torch.bool)
    self._command_dt = torch.ones(num_envs)


def _clone_mapping(value: Any) -> Any:
  if isinstance(value, dict):
    return {key: _clone_mapping(item) for key, item in value.items()}
  if isinstance(value, torch.Tensor):
    return value.clone()
  if isinstance(value, tuple):
    return tuple(_clone_mapping(item) for item in value)
  return value


def test_x2_adapter_maps_different_batch_rows_and_restores_all_fake_bridge_state():
  source = FakeEnv(5)
  destination = FakeEnv(3)
  source_row, destination_row = 4, 1
  source.sim.data.qpos._tensor[source_row, :3] = torch.tensor([1.0, 2.0, 3.0])
  source.sim.data.mocap_pos[source_row] = torch.tensor([[4.0, 5.0, 6.0]])
  source.command_manager.command._pending_reset_frame_ids = torch.tensor(
    [0, source_row], dtype=torch.long
  )
  expected_qpos = source.sim.data.qpos._tensor[source_row].clone()
  expected_mocap = source.sim.data.mocap_pos[source_row].clone()
  delta = (
    destination.scene.env_origins[destination_row]
    - source.scene.env_origins[source_row]
  )
  expected_qpos[:3] += delta
  expected_mocap += delta
  untouched_qpos = {
    row: destination.sim.data.qpos._tensor[row].clone() for row in (0, 2)
  }
  destination_delay = destination.scene.entities["robot"].actuators[0]._delay_buffer
  assert destination_delay is not None
  before_delay_history = destination_delay._buffer.buffer.clone()
  before_delay_pointer = destination_delay._buffer._pointer
  destination_history = destination.observation_manager._group_obs_term_history_buffer[
    "actor"
  ]["joint_vel"]
  before_observation_history = destination_history.buffer.clone()
  before_observation_pointer = destination_history._pointer
  state = BranchState.capture(source, env_index=source_row)
  state.restore(destination, env_index=destination_row)
  assert destination.sim.data.qpos._tensor[destination_row].equal(expected_qpos)
  assert destination.sim.data.mocap_pos[destination_row].equal(expected_mocap)
  for row, value in untouched_qpos.items():
    assert destination.sim.data.qpos._tensor[row].equal(value)
  assert destination.sim.forward_calls == 1
  assert destination.sim.recompute_calls
  assert destination.sim.data.qacc_warmstart._tensor[destination_row].equal(
    state.fields["qacc_warmstart"]
  )
  assert (
    destination.scene.entities["robot"].actuators[0]._delay_buffer
    is destination.scene.entities["robot"].actuators[1]._delay_buffer
  )
  assert destination.command_manager.command.reset_rng_state().equal(
    state.manager_state["command"]["reset_rng_state"]
  )
  actuator_generator_state = next(
    iter(state.buffer_state["actuator_aliases"].values())
  )["generator_state"]
  destination_delay = destination.scene.entities["robot"].actuators[0]._delay_buffer
  assert destination_delay is not None
  assert destination_delay.generator is not None
  assert destination_delay.generator.get_state().equal(actuator_generator_state)
  assert destination_delay._buffer.buffer[0].equal(before_delay_history[0])
  assert destination_delay._buffer.buffer[2].equal(before_delay_history[2])
  assert destination_delay._buffer._pointer == before_delay_pointer
  assert destination_history.buffer[0].equal(before_observation_history[0])
  assert destination_history.buffer[2].equal(before_observation_history[2])
  assert destination_history._pointer == before_observation_pointer
  assert destination.obs_buf["actor"][destination_row].equal(
    state.fields["teacher_actor_view"]
  )


def test_x2_adapter_validates_before_mutating_on_field_corruption():
  source = FakeEnv(5)
  destination = FakeEnv(3)
  state = BranchState.capture(source, env_index=4)
  before = destination.sim.data.qpos._tensor[1].clone()
  state.fields["qpos"] = torch.zeros(2)
  with pytest.raises(BranchStateMismatch, match="qpos"):
    X2FixedTopologyStateAdapter(destination, source_row=1).restore(
      state.fields, state.manager_state, state.buffer_state, destination_row=1
    )
  assert destination.sim.data.qpos._tensor[1].equal(before)


def test_x2_adapter_rejects_alias_and_history_corruption_before_copy():
  source = FakeEnv(5)
  destination = FakeEnv(3)
  state = BranchState.capture(source, env_index=4)
  before = destination.sim.data.qpos._tensor[1].clone()

  alias_buffers = _clone_mapping(state.buffer_state)
  alias_key = next(iter(alias_buffers["actuator_alias_members"]))
  alias_buffers["actuator_alias_members"][alias_key] = (0,)
  with pytest.raises(BranchStateMismatch, match="alias members"):
    X2FixedTopologyStateAdapter(destination, source_row=1).restore(
      state.fields, state.manager_state, alias_buffers, destination_row=1
    )
  assert destination.sim.data.qpos._tensor[1].equal(before)

  history_buffers = _clone_mapping(state.buffer_state)
  history = history_buffers["observation"]["history:actor:joint_vel"]
  history["chronological"] = torch.zeros(1, 99, 1)
  with pytest.raises(BranchStateMismatch, match="chronology"):
    X2FixedTopologyStateAdapter(destination, source_row=1).restore(
      state.fields, state.manager_state, history_buffers, destination_row=1
    )
  assert destination.sim.data.qpos._tensor[1].equal(before)


def test_x2_adapter_rejects_unknown_active_sensor_before_capture():
  env = FakeEnv(2)
  sensors: Any = env.scene.sensors
  sensors["unknown"] = object()
  with pytest.raises(BranchStateError, match="unsupported active sensor"):
    X2FixedTopologyStateAdapter(env, source_row=0).capture()


def test_x2_adapter_rejects_stateful_event_and_noise_components():
  env = FakeEnv(2)
  env.event_manager._mode_term_cfgs = {
    "step": [SimpleNamespace(func=lambda *_args: None)]
  }
  with pytest.raises(BranchStateError, match="stateful or unknown event"):
    X2FixedTopologyStateAdapter(env, source_row=0)
  env = FakeEnv(2)
  env.observation_manager._group_obs_class_instances["actor"]["noise"] = object()
  with pytest.raises(BranchStateError, match="stateful observation noise"):
    X2FixedTopologyStateAdapter(env, source_row=0).capture()


def test_terminal_reconstruction_has_explicit_synthetic_only_escape_hatch():
  source = FakeEnv(5)
  destination = FakeEnv(3)
  state = BranchState.capture(source, env_index=4)
  capture = TerminalCapture(state, torch.zeros(1), torch.zeros(1), "done")
  component = CoverageEnvironmentComponent(
    source, CoveragePolicy.difficult_state_coverage()
  )
  with pytest.raises(CoverageRuntimeError, match="fail-closed"):
    component.reconstruct_terminal(capture, destination, env_index=1)

  def boundary_handler(action):
    del action

  destination_any: Any = destination
  destination_any.coverage_reconstruct_next_boundary = boundary_handler
  restored = component.reconstruct_terminal(
    capture, destination, env_index=1, synthetic=True
  )
  assert restored.capture_instant == "raw_reached_pre_reset"


# ---------------------------------------------------------------------------
# The three environment handlers the coverage capture calls.  They live on
# MultiMotionDistillationAdapter and read the cached instant: no recompute, no new
# noise draw.  These tests drive the handlers directly with a stub owner, so they
# do not need a full cohort or teacher bank.
# ---------------------------------------------------------------------------


def test_coverage_handlers_read_the_cached_view_and_the_applied_action() -> None:
  from types import SimpleNamespace

  from mjlab.tasks.tracking.distillation.adapter import MultiMotionDistillationAdapter

  actor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
  cache_reads = {"count": 0}

  def cached_observations():
    cache_reads["count"] += 1
    return {"actor": actor, "critic": actor * 0}

  owner = SimpleNamespace(
    env=SimpleNamespace(
      observation_manager=SimpleNamespace(cached_observations=cached_observations)
    ),
    _coverage_last_action=torch.arange(9, dtype=torch.float32).reshape(3, 3),
  )
  view = MultiMotionDistillationAdapter.coverage_cached_policy_view(owner, 1)
  assert torch.equal(view, actor[1])
  # A cached read, not a recompute, and the returned row is owned by the caller.
  assert cache_reads["count"] == 1
  view[0] = -1.0
  assert torch.equal(actor[1], torch.tensor([4.0, 5.0, 6.0, 7.0]))

  action = MultiMotionDistillationAdapter.coverage_executed_action(owner, 2)
  assert torch.equal(action, torch.tensor([6.0, 7.0, 8.0]))
  action[0] = -1.0
  assert torch.equal(owner._coverage_last_action[2], torch.tensor([6.0, 7.0, 8.0]))


def test_coverage_handlers_fail_closed_without_the_needed_state() -> None:
  from types import SimpleNamespace

  import pytest

  from mjlab.tasks.tracking.distillation.adapter import MultiMotionDistillationAdapter
  from mjlab.tasks.tracking.distillation.coverage_environment import (
    CoverageRuntimeError,
  )

  # No cached_observations seam on the environment.
  bare = SimpleNamespace(env=SimpleNamespace(observation_manager=SimpleNamespace()))
  with pytest.raises(CoverageRuntimeError, match="cached_observations"):
    MultiMotionDistillationAdapter.coverage_cached_policy_view(bare, 0)

  # No action applied yet.
  owner = SimpleNamespace(
    env=SimpleNamespace(
      observation_manager=SimpleNamespace(
        cached_observations=lambda: {"actor": torch.zeros(1, 4)}
      )
    ),
  )
  with pytest.raises(CoverageRuntimeError, match="no action has been applied"):
    MultiMotionDistillationAdapter.coverage_executed_action(owner, 0)

  # A row outside the cached batch.
  with pytest.raises(CoverageRuntimeError, match="outside"):
    MultiMotionDistillationAdapter.coverage_cached_policy_view(owner, 5)
