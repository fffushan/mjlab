"""CPU fake-runtime tests for the private D3 simulator boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from mjlab.tasks.tracking.diffusion import closed_loop_runtime_x2 as runtime_module
from mjlab.tasks.tracking.diffusion.closed_loop_runtime_x2 import (
  ClosedLoopMotionCommand,
  RuntimeX2Error,
  X2ClosedLoopRuntime,
  build_closed_loop_runtime,
)
from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT


@dataclass
class _Term:
  func: Any
  params: dict[str, Any]
  time_out: bool = False


class _Observations:
  terms = tuple(
    SimpleNamespace(name=name, width=width)
    for name, width in zip(
      (
        "command",
        "motion_anchor_ori_b",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "actions",
      ),
      (62, 6, 3, 31, 31, 31),
      strict=True,
    )
  )
  total_dim = 164
  names = tuple(term.name for term in terms)
  widths = tuple(term.width for term in terms)


class _FakeCommand:
  def __init__(self) -> None:
    self._reference_free = False
    self._time_steps = torch.tensor([0], dtype=torch.long)
    self.generation_ids = torch.tensor([0], dtype=torch.long)
    self.motion = SimpleNamespace(time_step_total=32)
    self.cfg = SimpleNamespace(body_names=DEFAULT_CONTRACT.body_names)
    self.timer_resample_calls = 0
    self.consume_calls = 0

  @property
  def time_steps(self) -> torch.Tensor:
    if self._reference_free:
      raise AssertionError("reference frame read after handover")
    return self._time_steps

  def freeze_reference(self) -> None:
    self._reference_free = True

  def restore_reference(self) -> None:
    self._reference_free = False

  def reset_to_frame(self, _env_ids: torch.Tensor, frame: int) -> None:
    self._reference_free = False
    self._time_steps[:] = frame
    self.generation_ids += 1

  def consume_boundary_events(self, _pre_generation: Any = None) -> Any:
    self.consume_calls += 1
    return SimpleNamespace(
      available=torch.tensor([False]),
      completed=torch.tensor([False]),
      interrupted=torch.tensor([False]),
      reasons=("unavailable",),
    )

  def compute(self) -> None:
    if self._reference_free:
      self.timer_resample_calls += 1


class _FakeData:
  def __init__(self) -> None:
    self.root_link_pos_w = torch.tensor([[0.0, 0.0, 0.8]])
    self.root_link_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    self.root_link_lin_vel_w = torch.zeros((1, 3))
    self.root_link_ang_vel_w = torch.zeros((1, 3))
    self.body_link_pos_w = torch.zeros((1, 20, 3))
    self.body_link_lin_vel_w = torch.zeros((1, 20, 3))
    self.joint_pos = torch.arange(31, dtype=torch.float32)[None] * 0.01
    self.joint_vel = torch.zeros((1, 31))
    self.projected_gravity_b = torch.tensor([[0.0, 0.0, 1.0]])


class _FakeRobot:
  joint_names = tuple(f"joint_{index:02d}" for index in range(31))
  body_names = DEFAULT_CONTRACT.body_names

  def __init__(self) -> None:
    self.data = _FakeData()


class _FakeObservationManager:
  def __init__(self, env: _FakeEnv) -> None:
    self.env = env
    self.calls: list[tuple[bool, torch.Tensor | None]] = []
    self._obs_buffer: dict[str, torch.Tensor] | None = None

  def compute(
    self, *, update_history: bool = False, env_ids: torch.Tensor | None = None
  ) -> dict[str, torch.Tensor]:
    self.calls.append((update_history, env_ids))
    assert self._obs_buffer is not None
    return self._obs_buffer


class _FakeCommandManager:
  def __init__(self, command: _FakeCommand) -> None:
    self.command = command

  def get_term(self, name: str) -> _FakeCommand:
    assert name == "motion"
    return self.command


class _FakeScene:
  def __init__(self, robot: _FakeRobot) -> None:
    self.robot = robot

  def __getitem__(self, name: str) -> _FakeRobot:
    assert name == "robot"
    return self.robot

  def write_data_to_sim(self) -> None:
    pass


class _FakeEnv:
  num_envs = 1
  device = "cpu"
  step_dt = 0.02

  def __init__(self) -> None:
    self.robot = _FakeRobot()
    self.command = _FakeCommand()
    self.scene = _FakeScene(self.robot)
    self.command_manager = _FakeCommandManager(self.command)
    self.termination_manager = SimpleNamespace(
      active_terms=["time_out", "anchor_pos", "anchor_ori", "ee_body_pos"],
      _term_cfgs=[
        _Term(lambda *_args, **_kwargs: torch.zeros(1, dtype=torch.bool), {}, True),
        _Term(lambda *_args, **_kwargs: torch.zeros(1, dtype=torch.bool), {"x": 1}),
        _Term(lambda *_args, **_kwargs: torch.zeros(1, dtype=torch.bool), {"x": 2}),
        _Term(lambda *_args, **_kwargs: torch.zeros(1, dtype=torch.bool), {"x": 3}),
      ],
    )
    self.cfg = SimpleNamespace(auto_reset=True)
    self.sim = SimpleNamespace(forward=lambda: None, sense=lambda: None)
    self.observation_manager = _FakeObservationManager(self)
    self.obs_buf = {"actor": self._actor_cache()}
    self.observation_manager._obs_buffer = self.obs_buf
    self.seen_actions: list[torch.Tensor] = []
    self.terminate_next = False
    self.reset_calls = 0

  def _actor_cache(self) -> torch.Tensor:
    actor = torch.zeros((1, 164), dtype=torch.float32)
    actor[:, :62] = torch.arange(62, dtype=torch.float32)
    actor[:, 62:68] = 0.1
    actor[:, 68:71] = torch.tensor([[0.1, 0.2, 0.3]])
    actor[:, 71:102] = torch.arange(31, dtype=torch.float32) * 0.01
    actor[:, 102:133] = 0.2
    actor[:, 133:164] = 0.0
    return actor

  def _refresh_cache(self) -> None:
    actor = self._actor_cache()
    actor[:, 133:164] = self.seen_actions[-1].cpu()
    self.obs_buf = {"actor": actor}
    self.observation_manager._obs_buffer = self.obs_buf

  def reset(self, *, seed: int) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    del seed
    self.reset_calls += 1
    self.robot.data.root_link_pos_w[:] = torch.tensor([[0.0, 0.0, 0.8]])
    self.robot.data.joint_vel.zero_()
    self.obs_buf = {"actor": self._actor_cache()}
    self.observation_manager._obs_buffer = self.obs_buf
    return self.obs_buf, {}

  def step(self, action: torch.Tensor) -> tuple[Any, ...]:
    assert action.shape == (1, 31)
    assert action.dtype is torch.float32
    self.seen_actions.append(action.detach().clone())
    if not self.command._reference_free and not self.terminate_next:
      self.command._time_steps += 1
    self.robot.data.root_link_pos_w[:, 0] += 0.05
    self._refresh_cache()
    terminated = torch.tensor([self.terminate_next])
    self.terminate_next = False
    return self.obs_buf, torch.zeros(1), terminated, torch.tensor([False]), {}

  def close(self) -> None:
    pass


class _Cohort:
  observations: Any = _Observations()
  actions = SimpleNamespace(
    joint_names=_FakeRobot.joint_names,
    dim=31,
  )
  control = SimpleNamespace(control_period_s=0.02)
  actor = SimpleNamespace(obs_dim=164)
  body_names = DEFAULT_CONTRACT.body_names
  manifest = SimpleNamespace(base_task=None)


def _runtime() -> tuple[X2ClosedLoopRuntime, _FakeEnv]:
  env = _FakeEnv()
  runtime = X2ClosedLoopRuntime(env, _Cohort(), "tennis_000")
  return runtime, env


def test_zero_width_motion_lookahead_preserves_reference_and_conditioning() -> None:
  original, _ = _runtime()
  expected = original.reset(seed=0, start_frame=2)
  cohort = _Cohort()
  terms = list(cohort.observations.terms)
  terms.insert(1, SimpleNamespace(name="motion_lookahead", width=0))
  cohort.observations = SimpleNamespace(
    terms=tuple(terms),
    names=tuple(term.name for term in terms),
    widths=tuple(term.width for term in terms),
    total_dim=164,
  )
  runtime_module._validate_cohort_static(cohort)
  runtime = X2ClosedLoopRuntime(_FakeEnv(), cohort, "tennis_000")
  observed = runtime.reset(seed=0, start_frame=2)
  np.testing.assert_array_equal(observed.reference, expected.reference)
  np.testing.assert_array_equal(observed.conditioning, expected.conditioning)
  assert runtime._slices["motion_lookahead"] == slice(62, 62)
  assert runtime.provenance["actor_observation_layout"]["ignored_zero_width_terms"] == [
    "motion_lookahead"
  ]


@pytest.mark.parametrize("name,width", [("motion_lookahead", 1), ("unknown", 0)])
def test_only_known_zero_width_lookahead_is_compatible(name: str, width: int) -> None:
  cohort = _Cohort()
  terms = list(cohort.observations.terms)
  terms.insert(1, SimpleNamespace(name=name, width=width))
  cohort.observations = SimpleNamespace(
    terms=tuple(terms),
    names=tuple(term.name for term in terms),
    widths=tuple(term.width for term in terms),
    total_dim=164 + width,
  )
  with pytest.raises(RuntimeX2Error, match="unsupported schema"):
    runtime_module._validate_cohort_static(cohort)
  with pytest.raises(RuntimeX2Error, match="unsupported schema"):
    X2ClosedLoopRuntime(_FakeEnv(), cohort, "tennis_000")


def test_frozen_cohort_static_preflight_accepts_zero_width_lookahead() -> None:
  # Artifact/model metadata only: do not build or step an environment.
  repo_root = Path(__file__).parents[1]
  _, cohort = runtime_module._validate_factory_inputs(
    repo_root / "configs/distillation/x2_tennis_mixed.yaml",
    repo_root,
    "tennis_000",
    0,
  )
  assert cohort.observations.total_dim == 164
  assert (
    dict(zip(cohort.observations.names, cohort.observations.widths, strict=True))[
      "motion_lookahead"
    ]
    == 0
  )


def test_factory_marks_only_verified_path_as_mjlab(
  tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
  manifest = tmp_path / "manifest.yaml"
  manifest.write_text("synthetic test manifest")
  env = _FakeEnv()
  monkeypatch.setattr(
    runtime_module,
    "_validate_factory_inputs",
    lambda *_args, **_kwargs: (manifest, _Cohort()),
  )
  monkeypatch.setattr(
    runtime_module,
    "_build_private_environment",
    lambda *_args, **_kwargs: env,
  )
  monkeypatch.setattr(
    runtime_module,
    "validate_live_contract",
    lambda *_args, **_kwargs: object(),
  )

  runtime = build_closed_loop_runtime(
    manifest_path=manifest,
    repo_root=tmp_path,
    member="tennis_000",
    device="cpu",
    seed=0,
  )
  try:
    assert runtime.provenance["physical_runtime"] == "mjlab"
    assert runtime.provenance["synthetic"] is False
    assert runtime.provenance["measurement_source"] == (
      "real_mjlab_simulator_robot_data"
    )
  finally:
    runtime.close()

  runtime, env = _runtime()
  observation = runtime.reset(seed=4, start_frame=2)

  assert observation.reference is not None
  assert observation.reference.shape == (68,)
  assert observation.conditioning.shape == (99,)
  np.testing.assert_allclose(observation.conditioning[:3], [0.0, 0.0, 1.0])
  np.testing.assert_allclose(
    runtime.initial_signature()[7:38], env.robot.data.joint_pos[0]
  )
  assert runtime.provenance["physical_runtime"] == "synthetic"
  assert runtime.provenance["synthetic"] is True
  assert runtime.provenance["measurement_source"] == (
    "injected_test_environment_robot_data"
  )
  assert runtime.provenance["initial_state_signature"]["fields"] == [
    "root_position",
    "root_quaternion_wxyz",
    "joint_position",
    "joint_velocity",
  ]

  saved = observation.conditioning.copy()
  env.obs_buf["actor"][:] = 999.0
  env.robot.data.root_link_pos_w[:] = 999.0
  np.testing.assert_array_equal(observation.conditioning, saved)
  assert observation.state.root_position[0] != 999.0


def test_reference_free_handover_does_not_step_or_read_reference() -> None:
  runtime, env = _runtime()
  runtime.reset(seed=0, start_frame=0)
  before = runtime.initial_signature()
  cache_calls = len(env.observation_manager.calls)

  observation = runtime.enter_reference_free()

  np.testing.assert_array_equal(runtime.initial_signature(), before)
  assert observation.reference is None
  assert observation.reference_frame is None
  assert observation.tick == 0
  assert env.command._reference_free
  assert len(env.observation_manager.calls) == cache_calls
  frozen_frame = env.command._time_steps.clone()
  env.command._time_steps[:] = env.command.motion.time_step_total - 1
  runtime.step(np.zeros(31))
  assert torch.equal(
    env.command._time_steps,
    torch.full_like(frozen_frame, env.command.motion.time_step_total - 1),
  )
  with pytest.raises(AssertionError, match="reference frame"):
    _ = env.command.time_steps


def test_step_is_direct_boundary_with_exact_period_and_terminal_capture() -> None:
  runtime, env = _runtime()
  runtime.reset(seed=1, start_frame=0)
  runtime.enter_reference_free()
  transition = runtime.step(np.full(31, 0.25))

  assert env.seen_actions[0].shape == (1, 31)
  assert transition.observation.tick == 1
  assert transition.observation.sim_time_s == 0.02
  assert transition.terminal_available
  assert not transition.terminated
  assert transition.observation.reference is None

  boundary_runtime, _boundary_env = _runtime()
  boundary_runtime.reset(seed=1, start_frame=30)
  boundary = boundary_runtime.step(np.zeros(31))
  assert boundary.reference_boundary
  assert boundary.reason == "reference_boundary"

  env.terminate_next = True
  terminal = runtime.step(np.zeros(31))
  assert terminal.terminated
  assert terminal.terminal_available
  assert not terminal.reset
  np.testing.assert_allclose(terminal.observation.state.root_position[0], 0.1)


def test_reset_restores_vae_mode_guards_action_cache_and_ticks() -> None:
  runtime, env = _runtime()
  runtime.reset(seed=2, start_frame=0)
  runtime.enter_reference_free()
  runtime.step(np.ones(31))
  assert env.cfg.auto_reset is False

  observation = runtime.reset(seed=3, start_frame=4)

  assert observation.reference is not None
  assert observation.reference_frame == 4
  assert observation.tick == 0
  assert observation.sim_time_s == 0.0
  np.testing.assert_allclose(observation.conditioning[-31:], 0.0)
  assert not env.command._reference_free
  assert env.reset_calls == 2
  assert [term.params for term in env.termination_manager._term_cfgs[1:]] == [
    {"x": 1},
    {"x": 2},
    {"x": 3},
  ]


def test_last_frame_refuses_action_without_physical_step() -> None:
  runtime, env = _runtime()
  runtime.reset(seed=5, start_frame=31)
  before_position = env.robot.data.root_link_pos_w.clone()

  with pytest.raises(RuntimeX2Error, match="final reference frame"):
    runtime.step(np.zeros(31))

  assert not env.seen_actions
  assert torch.equal(env.robot.data.root_link_pos_w, before_position)
  assert runtime.provenance["last_frame_refusal"]["physical_step_executed"] is False


def test_terminal_vae_step_preserves_post_action_state_without_wrap() -> None:
  runtime, env = _runtime()
  runtime.reset(seed=6, start_frame=30)
  env.terminate_next = True

  transition = runtime.step(np.zeros(31))

  assert transition.terminated
  assert not transition.reference_boundary
  assert not transition.teleport
  assert transition.observation.reference_frame == 30
  np.testing.assert_allclose(transition.observation.state.root_position[0], 0.05)


def test_terminal_command_suppresses_timer_resample_and_frame_update() -> None:
  command = object.__new__(ClosedLoopMotionCommand)
  command._reference_free = False
  object.__setattr__(
    command,
    "_env",
    SimpleNamespace(
      termination_manager=SimpleNamespace(
        terminated=torch.tensor([True]),
        time_outs=torch.tensor([False]),
      ),
      device="cpu",
    ),
  )
  command._resample(torch.tensor([0]))
  command._update_command(None)
  assert command._terminal_step()

  command = object.__new__(ClosedLoopMotionCommand)
  command._reference_free = True
  command.time_left = torch.tensor([-1.0])
  before_timer = command.time_left.clone()
  command._resample(torch.tensor([0]))
  command._resample_command(torch.tensor([0]))
  command._update_metrics()
  command._update_command(None)
  command.compute(0.02)
  assert torch.equal(command.time_left, before_timer)
  assert command.reference_free


def test_unsupported_schema_is_rejected_before_runtime_use() -> None:
  cohort = SimpleNamespace(
    observations=SimpleNamespace(
      terms=(SimpleNamespace(name="wrong", width=164),),
      names=("wrong",),
      widths=(164,),
      total_dim=164,
    ),
    actions=_Cohort.actions,
    control=_Cohort.control,
    actor=_Cohort.actor,
    body_names=_Cohort.body_names,
  )
  with pytest.raises(RuntimeX2Error, match="unsupported schema"):
    X2ClosedLoopRuntime(_FakeEnv(), cohort, "tennis_000")
