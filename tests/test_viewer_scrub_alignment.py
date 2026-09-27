"""Regression tests for observation alignment after motion scrubbing.

The Viser viewer computes and caches observations inside ``env.reset`` and then
rewrites robot/reference state for a ``Start Here`` scrub (or a paused frame
slider move).  These tests drive the real ``ViserPlayViewer`` methods against a
real ``ObservationManager`` (raw noisy, history, delayed, and grouped-delay
terms) with mocked physics, so they fail if the first policy action after
scrubbing would read the pre-scrub observation, if the scrub changes another
environment's cached row or advances its history/delay timeline, or if the GUI
callback recomputes observations off the viewer's main loop.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch
from test_command_manager import _motion_gui_harness

from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationManager,
  ObservationTermCfg,
)
from mjlab.utils.noise.noise_cfg import UniformNoiseCfg
from mjlab.viewer.base import EnvProtocol
from mjlab.viewer.viser.viewer import ViserPlayViewer


def _state_term(env: _ScrubEnv, **_kwargs) -> torch.Tensor:
  return env.state.clone()


class _FakeSim:
  def __init__(self, env: _ScrubEnv) -> None:
    self._env = env

  def forward(self) -> None:
    self._env.forward_calls += 1

  def sense(self) -> None:
    self._env.sense_calls += 1


class _FakeCommandManager:
  """Applies a deterministic scrub of the selected envs (frame -> scrub_value)

  The frame is recorded rather than applied, so a test can prove the click-time
  frame reached the command while the scrubbed state stays deterministic.
  """

  def __init__(self, env: _ScrubEnv) -> None:
    self._env = env
    self.scrub_value = 9.0
    self.scrubbed: list[tuple[int, int]] = []
    self.reset_frames: list[int | None] = []
    self.reset_env_ids: list[list[int]] = []

  def apply_gui_reset(self, env_ids: torch.Tensor, frame: int | None = None) -> bool:
    self._env.scrub_calls += 1
    self.reset_frames.append(frame)
    self.reset_env_ids.append([int(index) for index in env_ids.tolist()])
    self._env.state[env_ids] = self.scrub_value
    self._env.sim.forward()
    return True

  def apply_gui_scrub(self, env_idx: int, frame: int) -> bool:
    # A frame move rewrites the command timeline; the observation term above
    # reads it, so the frame becomes the env's observable state.
    self.scrubbed.append((int(env_idx), int(frame)))
    self._env.state[int(env_idx)] = float(frame)
    return True

  def on_viewer_pause(self, paused: bool) -> None:
    del paused


class _NoScrubCommandManager(_FakeCommandManager):
  """A command term without a scrubber: the GUI reset must not refresh."""

  def apply_gui_reset(self, env_ids: torch.Tensor, frame: int | None = None) -> bool:
    del env_ids, frame
    self._env.scrub_calls += 1
    return False


class _ScrubEnv:
  """Unwrapped-env stand-in with a real ObservationManager cache."""

  def __init__(self, num_envs: int = 2) -> None:
    self.num_envs = num_envs
    self.device = "cpu"
    self.step_dt = 0.02
    self.cfg = SimpleNamespace(viewer=SimpleNamespace())
    self.state = torch.arange(1, num_envs + 1, dtype=torch.float32).reshape(-1, 1) * 10
    self.observation_manager = ObservationManager(
      {
        "actor": ObservationGroupCfg(
          terms={
            # Three history slots: a backfill writes one frame to every slot,
            # while an accidental append would leave a stale trailing frame.
            "state": ObservationTermCfg(func=_state_term, history_length=3),
            # A delayed term also exercises the delay-buffer backfill path.
            "delayed": ObservationTermCfg(func=_state_term, delay_max_lag=1),
          }
        )
      },
      self,
    )
    self.command_manager = _FakeCommandManager(self)
    self.scene = SimpleNamespace(write_data_to_sim=lambda: None)
    self.sim = _FakeSim(self)
    self.obs_buf: dict[str, Any] | None = {}
    self.forward_calls = 0
    self.sense_calls = 0
    self.scrub_calls = 0

  def reset(self, env_ids=None) -> None:
    # Mirrors ManagerBasedRlEnv.reset: the observation manager is reset and its
    # post-reset frame is computed/backfilled/published, all before the GUI
    # scrub rewrites the robot/reference state.
    self.observation_manager.reset(env_ids)
    self.state[env_ids] = 1.0
    self.obs_buf = self.observation_manager.compute(
      update_history=True, env_ids=env_ids
    )

  @property
  def unwrapped(self) -> _ScrubEnv:
    return self


def _make_viewer(
  env: _ScrubEnv, selected: int = 1, scene: Any = None
) -> ViserPlayViewer:
  # An external MagicMock server keeps the constructor from binding a port.
  viewer = ViserPlayViewer(
    cast(EnvProtocol, env), MagicMock(), viser_server=MagicMock()
  )
  viewer._scene = scene if scene is not None else MagicMock(env_idx=selected)
  viewer._pending_update_reasons = set()
  viewer._pause_button = MagicMock()
  viewer._status_html = MagicMock()
  return viewer


def _actor(env: _ScrubEnv) -> torch.Tensor:
  obs = env.observation_manager.compute()["actor"]
  assert isinstance(obs, torch.Tensor)
  return obs


def _history(env: _ScrubEnv):
  return env.observation_manager._group_obs_term_history_buffer["actor"]["state"]


def _delay(env: _ScrubEnv):
  return env.observation_manager._group_obs_term_delay_buffer["actor"]["delayed"]


def test_start_here_refreshes_scrubbed_observation() -> None:
  env = _ScrubEnv(num_envs=2)
  viewer = _make_viewer(env, selected=1)
  env.observation_manager.compute(update_history=True)  # one normal playback step

  viewer._handle_gui_reset(all_envs=False)

  obs = _actor(env)
  # The scrubbed env sees the POST-scrub state in every history slot.
  assert obs[1].tolist() == [9.0, 9.0, 9.0, 9.0]
  # The other env's cached observation is unchanged.
  assert obs[0].tolist() == [10.0, 10.0, 10.0, 10.0]
  assert env.scrub_calls == 1
  # The scrub applies state, forward, and sense before the refresh.
  assert env.sense_calls == 1
  assert env.forward_calls >= 2
  # The environment's published buffer agrees with the manager cache.
  assert env.obs_buf is env.observation_manager._obs_buffer
  published = env.obs_buf
  assert published is not None
  assert torch.equal(published["actor"], obs)


def test_start_here_uses_the_frame_and_env_captured_by_the_callback() -> None:
  env = _ScrubEnv(num_envs=2)
  viewer = _make_viewer(env, selected=1)
  env.observation_manager.compute(update_history=True)

  # The payload frame and environment are authoritative, not the live handle or
  # the current viewer selection.
  viewer._handle_gui_reset(all_envs=False, frame=3, env_idx=0)

  assert env.command_manager.reset_frames == [3]
  assert env.command_manager.reset_env_ids == [[0]]
  assert env.scrub_calls == 1
  obs = _actor(env)
  assert obs[0].tolist() == [9.0, 9.0, 9.0, 9.0]
  # The unselected env keeps its pre-reset rows.
  assert obs[1].tolist() == [20.0, 20.0, 20.0, 20.0]


def test_start_here_all_envs_ignores_the_click_time_environment() -> None:
  env = _ScrubEnv(num_envs=3)
  viewer = _make_viewer(env, selected=1)
  env.observation_manager.compute(update_history=True)

  viewer._handle_gui_reset(all_envs=True, frame=3, env_idx=2)

  obs = _actor(env)
  for index in range(3):
    assert obs[index].tolist() == [9.0, 9.0, 9.0, 9.0]


def test_start_here_all_envs_backfills_every_env() -> None:
  env = _ScrubEnv(num_envs=2)
  viewer = _make_viewer(env, selected=1)
  env.observation_manager.compute(update_history=True)

  viewer._handle_gui_reset(all_envs=True)

  obs = _actor(env)
  assert obs[0].tolist() == [9.0, 9.0, 9.0, 9.0]
  assert obs[1].tolist() == [9.0, 9.0, 9.0, 9.0]


def _noisy_scrub_env(num_envs: int = 2, *, concatenate: bool = True) -> _ScrubEnv:
  """Env whose observation has a raw noisy term plus a history term."""
  env = _ScrubEnv(num_envs=num_envs)
  env.observation_manager = ObservationManager(
    {
      "actor": ObservationGroupCfg(
        terms={
          "raw": ObservationTermCfg(
            func=_state_term, noise=UniformNoiseCfg(n_min=-1.0, n_max=1.0)
          ),
          "history": ObservationTermCfg(func=_state_term, history_length=3),
        },
        concatenate_terms=concatenate,
        enable_corruption=True,
      )
    },
    env,
  )
  return env


def _term_values(env: _ScrubEnv, term: str) -> torch.Tensor:
  group = env.observation_manager.compute()["actor"]
  terms = cast("dict[str, torch.Tensor]", group)
  return terms[term]


def test_start_here_preserves_other_envs_raw_noisy_rows() -> None:
  """``env.reset`` must not leak its whole-batch noise resample into env 0."""
  env = _noisy_scrub_env(num_envs=2)
  viewer = _make_viewer(env, selected=1)
  torch.manual_seed(41)
  env.observation_manager.compute(update_history=True)
  before = _actor(env).clone()

  viewer._handle_gui_reset(all_envs=False, frame=3)

  after = _actor(env)
  assert torch.equal(before[0], after[0]), (
    f"Unedited env changed: {before[0]} -> {after[0]}"
  )
  assert after[1, 1:].tolist() == [9.0, 9.0, 9.0]


def test_repeated_start_here_keeps_unedited_rows_and_timelines() -> None:
  env = _noisy_scrub_env(num_envs=2)
  viewer = _make_viewer(env, selected=1)
  torch.manual_seed(5)
  env.observation_manager.compute(update_history=True)
  before = _actor(env).clone()
  history = env.observation_manager._group_obs_term_history_buffer["actor"]["history"]
  pushes_before = history._num_pushes.clone()

  viewer._handle_gui_reset(all_envs=False)
  viewer._handle_gui_reset(all_envs=False)

  after = _actor(env)
  assert torch.equal(before[0], after[0])
  assert history._num_pushes[0] == pushes_before[0]


def test_start_here_preserves_nonselected_noncatenated_rows() -> None:
  """A non-concatenated group cache uses the same baseline handling."""
  env = _noisy_scrub_env(num_envs=2, concatenate=False)
  viewer = _make_viewer(env, selected=1)
  torch.manual_seed(12)
  env.observation_manager.compute(update_history=True)
  before_raw = _term_values(env, "raw").clone()
  before_history = _term_values(env, "history").clone()

  viewer._handle_gui_reset(all_envs=False)

  assert torch.equal(before_raw[0], _term_values(env, "raw")[0])
  assert torch.equal(before_history[0], _term_values(env, "history")[0])
  assert _term_values(env, "history")[1].tolist() == [9.0, 9.0, 9.0]


def test_queued_start_here_targets_the_env_captured_at_callback_time() -> None:
  """The real command callback's click-time env survives a selection change."""
  env = _ScrubEnv(num_envs=3)
  scene = MagicMock(env_idx=1)
  viewer = _make_viewer(env, selected=1, scene=scene)
  env.observation_manager.compute(update_history=True)

  _, slider, _, all_envs, on_start = _motion_gui_harness(viewer.request_action, None)
  slider.value = 4
  all_envs.value = False
  on_start(None)

  # The user switches environment before the main loop drains the action.
  scene.env_idx = 0
  viewer._process_actions()

  assert env.state.flatten().tolist() == [10.0, 9.0, 30.0]
  assert env.command_manager.reset_env_ids == [[1]]
  assert env.command_manager.reset_frames == [4]


def test_scrub_preserves_other_envs_raw_noisy_rows() -> None:
  """A raw noisy term must not resample the unedited env's cached rows."""
  env = _ScrubEnv(num_envs=2)
  env.observation_manager = ObservationManager(
    {
      "actor": ObservationGroupCfg(
        terms={
          "raw": ObservationTermCfg(
            func=_state_term, noise=UniformNoiseCfg(n_min=-1.0, n_max=1.0)
          ),
          "history": ObservationTermCfg(func=_state_term, history_length=3),
        },
        enable_corruption=True,
      )
    },
    env,
  )
  viewer = _make_viewer(env, selected=1)
  env.observation_manager.compute(update_history=True)
  torch.manual_seed(41)
  before = _actor(env).clone()

  env.state[1] = 5.0
  viewer._on_command_gui_change()
  viewer._process_actions()
  after = _actor(env)

  assert torch.equal(before[0], after[0]), f"Unedited env changed: {before[0]}"
  # The edited env sees the new state in every history slot.
  assert after[1, 1:].tolist() == [5.0, 5.0, 5.0]


def test_scrub_refresh_adds_no_extra_history_ticks_or_lag_updates() -> None:
  env = _ScrubEnv(num_envs=2)
  viewer = _make_viewer(env, selected=1)
  env.observation_manager.compute(update_history=True)
  history = _history(env)
  delay = _delay(env)

  viewer._handle_gui_reset(all_envs=False)
  # Reading the cache again (repeated renders) must not advance anything.
  for _ in range(3):
    env.observation_manager.compute()
  # One history frame per env; the selected env's zeroed lag counter was not
  # advanced past the reset, and the other env's schedule is untouched.
  assert history._num_pushes.tolist() == [1, 1]
  assert delay._step_count.tolist() == [1, 0]

  # A second scrub still never advances the lag schedule for either env.
  viewer._handle_gui_reset(all_envs=False)
  assert history._num_pushes.tolist() == [1, 1]
  assert delay._step_count.tolist() == [1, 0]

  # Exactly one normal playback step advances both envs once.
  env.state[:] = 11.0
  env.observation_manager.compute(update_history=True)
  assert history._num_pushes.tolist() == [2, 2]


def test_grouped_delays_backfill_only_the_edited_env() -> None:
  env = _ScrubEnv(num_envs=2)
  env.observation_manager = ObservationManager(
    {
      "actor": ObservationGroupCfg(
        terms={
          "a": ObservationTermCfg(
            func=_state_term, delay_max_lag=1, delay_group="packet"
          ),
          "b": ObservationTermCfg(
            func=_state_term, delay_max_lag=1, delay_group="packet"
          ),
        }
      )
    },
    env,
  )
  env.command_manager.scrub_value = 7.0
  viewer = _make_viewer(env, selected=1)
  env.observation_manager.compute(update_history=True)

  viewer._handle_gui_reset(all_envs=False)

  obs = _actor(env)
  assert obs[1].tolist() == [7.0, 7.0]
  assert obs[0].tolist() == [10.0, 10.0]
  schedule = env.observation_manager._group_obs_delay_group_buffer["actor"]["packet"]
  assert schedule._step_count.tolist() == [1, 0]


def test_queued_slider_move_applies_on_the_main_thread_and_uses_the_frame() -> None:
  env = _ScrubEnv(num_envs=2)
  scene = MagicMock(env_idx=1)
  viewer = _make_viewer(env, selected=1, scene=scene)
  env.observation_manager.compute(update_history=True)
  history = _history(env)

  # The payload is what the command GUI callback enqueues for the main loop,
  # delivered through the viewer's real thread-safe request hook.
  viewer.request_action("CUSTOM", {"type": "motion_scrub", "env_idx": 1, "frame": 5})
  assert env.command_manager.scrubbed == []

  viewer._process_actions()

  assert env.command_manager.scrubbed == [(1, 5)]
  obs = _actor(env)
  assert obs[1].tolist() == [5.0, 5.0, 5.0, 5.0]
  assert obs[0].tolist() == [10.0, 10.0, 10.0, 10.0]
  assert history._num_pushes.tolist() == [1, 1]
  scene.request_update.assert_called_once()


def test_queued_scrub_targets_the_env_captured_at_callback_time() -> None:
  env = _ScrubEnv(num_envs=3)
  scene = MagicMock(env_idx=2)
  viewer = _make_viewer(env, selected=2, scene=scene)
  env.observation_manager.compute(update_history=True)

  viewer.request_action("CUSTOM", {"type": "motion_scrub", "env_idx": 2, "frame": 4})
  # The user switches environment before the main loop drains the action.
  scene.env_idx = 0
  viewer._process_actions()

  # The edit follows the env captured at callback time, not the new selection.
  assert env.command_manager.scrubbed == [(2, 4)]
  obs = _actor(env)
  assert obs[2].tolist() == [4.0, 4.0, 4.0, 4.0]
  assert obs[0].tolist() == [10.0, 10.0, 10.0, 10.0]


def test_queued_scrub_is_applied_before_the_next_physics_step(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Resuming after a slider move consumes the edit before any step."""
  env = _ScrubEnv(num_envs=2)
  viewer = _make_viewer(env, selected=1)
  env.observation_manager.compute(update_history=True)
  monkeypatch.setattr(viewer, "sync_env_to_viewer", lambda: None)
  observed: list[tuple[str, list[tuple[int, int]]]] = []
  real_step_physics = viewer._step_physics

  def recording_step_physics(dt: float) -> None:
    observed.append(("physics", list(env.command_manager.scrubbed)))
    real_step_physics(dt)

  monkeypatch.setattr(viewer, "_step_physics", recording_step_physics)
  viewer.request_action("CUSTOM", {"type": "motion_scrub", "env_idx": 1, "frame": 6})
  viewer.resume()

  viewer.tick()

  # Action processing (which applies the queued edit and refreshes the
  # observation) runs before the physics phase of the same tick.
  assert observed == [("physics", [(1, 6)])]
  assert _actor(env)[1].tolist() == [6.0, 6.0, 6.0, 6.0]


def test_gui_callback_queues_without_touching_observations_off_thread() -> None:
  env = _ScrubEnv(num_envs=2)
  viewer = _make_viewer(env, selected=1)
  env.observation_manager.compute(update_history=True)
  main_thread = threading.get_ident()
  threads: list[int] = []
  compute = env.observation_manager.compute

  def recording_compute(*args, **kwargs):
    threads.append(threading.get_ident())
    return compute(*args, **kwargs)

  cast(Any, env.observation_manager).compute = recording_compute
  env.state[1] = 5.0

  # Viser runs GUI callbacks on a worker thread.
  worker = threading.Thread(target=viewer._on_command_gui_change, name="gui-callback")
  worker.start()
  worker.join(timeout=5)
  assert not worker.is_alive()
  assert threads == [], "GUI callback recomputed observations off the main loop"

  # The main loop applies the queued edit and recomputes once, on its thread.
  viewer._process_actions()
  assert threads and all(thread == main_thread for thread in threads)
  assert _actor(env)[1].tolist() == [5.0, 5.0, 5.0, 5.0]


def test_gui_reset_without_scrubber_leaves_post_reset_observation() -> None:
  env = _ScrubEnv(num_envs=2)
  env.command_manager = _NoScrubCommandManager(env)
  viewer = _make_viewer(env, selected=1)
  env.observation_manager.compute(update_history=True)

  viewer._handle_gui_reset(all_envs=False)

  obs = _actor(env)
  # No command state changed, so the post-reset observation stands and no
  # forward/sense/refresh runs.
  assert obs[1].tolist() == [1.0, 1.0, 1.0, 1.0]
  assert env.sense_calls == 0
