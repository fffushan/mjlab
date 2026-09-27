"""Tests for command manager."""

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_balance_env_cfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand


@pytest.fixture(scope="module")
def device():
  return get_test_device()


class CounterCommand(CommandTerm):
  """Stateful command term: a per-env counter ticked by _update_command."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.ticks = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.ticks.unsqueeze(-1).float()

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.ticks[env_ids] = 0

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      self.ticks += 1
    else:
      self.ticks[env_ids] += 1


@dataclass(kw_only=True)
class CounterCommandCfg(CommandTermCfg):
  resampling_time_range: tuple[float, float] = (1e9, 1e9)

  def build(self, env) -> CounterCommand:
    return CounterCommand(self, env)


@pytest.fixture
def counter_env(device):
  cfg = cartpole_balance_env_cfg()
  cfg.scene.num_envs = 4
  cfg.commands = {"counter": CounterCommandCfg()}
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  yield env
  env.close()


def test_partial_reset_does_not_advance_other_envs(counter_env):
  env = counter_env
  term = env.command_manager.get_term("counter")
  assert isinstance(term, CounterCommand)

  # Full reset: every env is resampled (counter zeroed) then ticked once.
  env.reset()
  assert term.ticks.tolist() == [1, 1, 1, 1]

  action = torch.zeros((env.num_envs, 1), device=env.device)
  env.step(action)
  assert term.ticks.tolist() == [2, 2, 2, 2]

  # Partial reset: only env 1 is resampled and ticked. Before the fix for
  # issue #1138 the other envs advanced too (to 3).
  env.reset(env_ids=torch.tensor([1], dtype=torch.int64, device=env.device))
  assert term.ticks.tolist() == [2, 1, 2, 2]

  # The next step advances everyone by exactly one.
  env.step(action)
  assert term.ticks.tolist() == [3, 2, 3, 3]


def test_old_style_update_command_raises(counter_env):
  """Terms with the old zero-arg _update_command fail fast at construction."""

  class OldStyleCommand(CounterCommand):
    def _update_command(self) -> None:  # type: ignore[override]
      self.ticks += 1

  @dataclass(kw_only=True)
  class OldStyleCommandCfg(CounterCommandCfg):
    def build(self, env) -> "OldStyleCommand":
      return OldStyleCommand(self, env)

  with pytest.raises(TypeError, match="env_ids"):
    OldStyleCommandCfg().build(counter_env)


def _make_motion_command_stub(time_steps, total, sampling_mode="uniform"):
  """A stub with just enough state to drive MotionCommand._update_command."""
  cmd = Mock()
  cmd.time_steps = torch.tensor(time_steps, dtype=torch.long)
  cmd.motion = Mock()
  cmd.motion.time_step_total = total
  cmd.cfg = Mock()
  cmd.cfg.sampling_mode = sampling_mode
  cmd.cfg.adaptive_alpha = 0.5
  cmd.bin_failed_count = torch.tensor([1.0, 1.0])
  cmd._current_bin_failed = torch.tensor([4.0, 4.0])
  cmd._pending_forward = False
  cmd._resample_command = Mock(
    side_effect=lambda ids: setattr(cmd, "_pending_forward", True)
  )
  return cmd


def test_motion_command_update_scopes_time_advance():
  cmd = _make_motion_command_stub([2, 5, 7], total=100)
  MotionCommand._update_command(cmd, env_ids=torch.tensor([1]))
  assert cmd.time_steps.tolist() == [2, 6, 7]
  cmd._resample_command.assert_not_called()
  cmd.update_relative_body_poses.assert_called_once()

  MotionCommand._update_command(cmd, env_ids=None)
  assert cmd.time_steps.tolist() == [3, 7, 8]


def test_motion_command_update_resamples_on_wraparound():
  cmd = _make_motion_command_stub([2, 9], total=10)
  MotionCommand._update_command(cmd, env_ids=torch.tensor([1]))
  # Env 1 wrapped past the end of the motion and must be resampled; env 0
  # is untouched.
  (wrap_ids,), _ = cmd._resample_command.call_args
  assert wrap_ids.tolist() == [1]
  assert cmd.time_steps[0].item() == 2
  cmd._env.sim.forward.assert_called_once()


def test_motion_command_ema_folds_only_on_step_update():
  cmd = _make_motion_command_stub([2, 5], total=100, sampling_mode="adaptive")
  MotionCommand._update_command(cmd, env_ids=torch.tensor([0]))
  # Reset-scoped update: EMA untouched, pending failure counts preserved.
  assert cmd.bin_failed_count.tolist() == [1.0, 1.0]
  assert cmd._current_bin_failed.tolist() == [4.0, 4.0]

  MotionCommand._update_command(cmd, env_ids=None)
  # Per-step update: EMA folds the counts and clears them.
  assert cmd.bin_failed_count.tolist() == [2.5, 2.5]
  assert cmd._current_bin_failed.tolist() == [0.0, 0.0]


def test_motion_command_gui_reset_forwards_before_pose_update():
  """apply_gui_reset must refresh kinematics between the state write and
  update_relative_body_poses (viewer forwards only after it returns)."""
  calls = []
  cmd = Mock()
  cmd._scrubber_handles = (Mock(value=5),)
  cmd.reset_to_frame = lambda ids, frame: calls.append("reset_to_frame")
  cmd._env.sim.forward = lambda: calls.append("forward")
  cmd.update_relative_body_poses = lambda: calls.append("update_poses")

  assert MotionCommand.apply_gui_reset(cmd, torch.tensor([0])) is True
  assert calls == ["reset_to_frame", "forward", "update_poses"]


def test_motion_command_timer_resample_triggers_forward():
  """A timer-expiry resample (flag set before _update_command) forwards."""
  cmd = _make_motion_command_stub([2, 5], total=100)
  cmd._pending_forward = True  # As set by a compute-path _resample_command.
  MotionCommand._update_command(cmd, env_ids=None)
  cmd._env.sim.forward.assert_called_once()
  assert cmd._pending_forward is False

  cmd._env.sim.forward.reset_mock()
  MotionCommand._update_command(cmd, env_ids=None)
  cmd._env.sim.forward.assert_not_called()


def _motion_gui_harness(request_action, on_change):
  """Build a mocked Viser server and extract the motion scrubber callbacks."""
  server = MagicMock()
  cmd = MagicMock()
  cmd.motion.time_step_total = 10  # max frame index is then 9
  MotionCommand.create_gui(
    cmd,
    "motion",
    server,
    lambda: 1,
    on_change=on_change,
    request_action=request_action,
  )
  scrubber = server.gui.add_slider.return_value
  start_button = server.gui.add_button.return_value
  all_envs = server.gui.add_checkbox.return_value
  return (
    cmd,
    scrubber,
    scrubber.on_update.call_args[0][0],
    all_envs,
    start_button.on_click.call_args[0][0],
  )


def test_motion_command_slider_queues_when_a_viewer_hook_is_present():
  """The Viser callback runs off-thread, so it must only enqueue the edit."""
  queued: list[tuple[str, Any]] = []
  changed: list[None] = []
  cmd, scrubber, on_frame, _, _ = _motion_gui_harness(
    lambda name, payload: queued.append((name, payload)),
    lambda: changed.append(None),
  )
  scrubber.value = 7

  on_frame(None)

  assert queued == [("CUSTOM", {"type": "motion_scrub", "env_idx": 1, "frame": 7})]
  cmd.time_steps.__setitem__.assert_not_called()
  assert changed == []


def test_motion_command_start_here_queues_the_click_frame():
  queued: list[tuple[str, Any]] = []
  cmd, scrubber, _, all_envs, on_start = _motion_gui_harness(
    lambda name, payload: queued.append((name, payload)), None
  )
  scrubber.value = 4
  all_envs.value = False

  on_start(None)

  # The environment is captured with the frame, so a later selection change
  # cannot retarget the queued reset.
  assert queued == [
    ("CUSTOM", {"type": "gui_reset", "all_envs": False, "frame": 4, "env_idx": 1})
  ]
  cmd.reset_to_frame.assert_not_called()


def test_motion_command_slider_writes_directly_without_a_viewer_hook():
  """A bare GUI host keeps the original synchronous write behavior."""
  changed: list[None] = []
  cmd, scrubber, on_frame, _, _ = _motion_gui_harness(
    None, lambda: changed.append(None)
  )
  scrubber.value = 5

  on_frame(None)

  cmd.time_steps.__setitem__.assert_called_once_with(1, 5)
  assert changed == [None]


def test_motion_command_applies_a_queued_scrub_to_one_env():
  cmd = Mock()
  cmd._scrubber_handles = (Mock(value=0),)
  cmd.time_steps = MagicMock()

  assert MotionCommand.apply_gui_scrub(cmd, 2, 8) is True

  cmd.time_steps.__setitem__.assert_called_once_with(2, 8)


def test_motion_command_gui_reset_prefers_the_captured_frame():
  """The click-time frame wins over the live (mutable) scrubber handle."""
  calls: list[Any] = []
  cmd = Mock()
  cmd._scrubber_handles = (Mock(value=5),)
  cmd.reset_to_frame = lambda ids, frame: calls.append(frame)
  cmd._env.sim.forward = lambda: calls.append("forward")
  cmd.update_relative_body_poses = lambda: calls.append("poses")

  assert MotionCommand.apply_gui_reset(cmd, torch.tensor([0]), 3) is True
  assert calls == [3, "forward", "poses"]

  # Without a captured frame the live handle remains the fallback.
  calls.clear()
  assert MotionCommand.apply_gui_reset(cmd, torch.tensor([0])) is True
  assert calls == [5, "forward", "poses"]


def test_command_manager_gui_scrub_defaults_to_false(counter_env):
  """A term without a scrubber reports no application instead of raising."""
  assert counter_env.command_manager.apply_gui_scrub(0, 3) is False
