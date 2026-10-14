"""CPU-only fake-runtime tests for the opt-in D3 visual diagnostic.

These tests never construct MuJoCo, a Viser server, a window, or a socket.
They exercise the playback controller and the thin viewer glue with injected
fakes, and make no physical or matched-acceptance claim.
"""

from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from mjlab.tasks.tracking.diffusion import visual_playback as vp
from mjlab.tasks.tracking.diffusion.closed_loop import _vae_action
from mjlab.tasks.tracking.diffusion.closed_loop_types import (
  D3Observation,
  D3Transition,
)
from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT
from mjlab.tasks.tracking.diffusion.state import WorldState
from mjlab.tasks.tracking.diffusion.visual_playback import (
  D3DiagnosticEnv,
  D3DiagnosticPolicy,
  D3DiagnosticViewer,
  VisualDiagnosticPlayback,
)
from mjlab.viewer import ViewerConfig
from mjlab.viewer.base import ViewerAction

_JOINTS = tuple(f"joint_{index:02d}" for index in range(31))
_BODIES = tuple(f"body_{index:02d}" for index in range(20))

_VIEW_PATH = (
  Path(__file__).parents[1] / "scripts/experiments/diffusion_d3_closed_loop/view.py"
)
_VIEW_SPEC = importlib.util.spec_from_file_location("d3_visual_view_cli", _VIEW_PATH)
assert _VIEW_SPEC is not None and _VIEW_SPEC.loader is not None
view = importlib.util.module_from_spec(_VIEW_SPEC)
sys.modules[_VIEW_SPEC.name] = view
_VIEW_SPEC.loader.exec_module(view)


class _FakeVae:
  """Minimal VAE surface with deterministic, input-dependent outputs."""

  def __init__(self) -> None:
    self.encode_calls = 0
    self.decode_calls = 0

  def encode(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    self.encode_calls += 1
    mu = torch.full((1, 32), float(reference.sum().item()), dtype=torch.float32)
    return mu, torch.zeros((1, 32), dtype=torch.float32)

  def decode(self, latent: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
    self.decode_calls += 1
    del conditioning
    return torch.full((1, 31), float(latent.sum().item()), dtype=torch.float32)


class _FakeContinuation:
  """Records executed pairs and returns a fixed reference-free step."""

  def __init__(self) -> None:
    self.records: list[tuple[WorldState, np.ndarray]] = []
    self.steps = 0

  def record_executed(self, state: WorldState, latent: Any) -> None:
    self.records.append((state, np.array(latent, dtype=np.float64)))

  def step(self, state: WorldState, conditioning: Any, *, tick: int) -> SimpleNamespace:
    assert tick >= 0
    if not self.records:
      raise RuntimeError("history is not ready")
    self.steps += 1
    return SimpleNamespace(
      action=np.zeros(31, dtype=np.float64),
      executed_latent=np.full(32, 0.25, dtype=np.float64),
      denoiser_calls=20,
      final_index=0,
      predicted_next_hybrid_state=None,
    )


class _FakeRuntime:
  """Simulator-free ``ClosedLoopRuntime`` for the visual diagnostic."""

  period_seconds = 0.02
  joint_order = _JOINTS
  body_order = _BODIES
  provenance: Mapping[str, Any] = {"member": "tennis_000"}

  def __init__(
    self,
    *,
    clip_frames: int = 64,
    tilt_after: int | None = None,
    terminate_after: int | None = None,
    reset_raises: bool = False,
  ) -> None:
    self.clip_frames = clip_frames
    self.tilt_after = tilt_after
    self.terminate_after = terminate_after
    self.reset_raises = reset_raises
    self.closed = False
    self.step_calls = 0
    self._tick = 0
    self._start_frame = 0
    self._phase = "vae"
    self._tilt = 0.0

  def _state(self) -> WorldState:
    half = self._tilt / 2.0
    quaternion = np.array([np.cos(half), np.sin(half), 0.0, 0.0], dtype=np.float64)
    return WorldState(
      root_position=np.array([0.0, 0.0, 0.8], dtype=np.float64),
      root_quaternion_wxyz=quaternion,
      root_linear_velocity=np.zeros(3),
      root_angular_velocity=np.zeros(3),
      body_positions=np.zeros((20, 3)),
      body_linear_velocities=np.zeros((20, 3)),
    )

  def _observation(self) -> D3Observation:
    reference = None
    reference_frame = None
    if self._phase == "vae":
      reference = np.full(68, 0.1 * (self._tick + 1), dtype=np.float64)
      reference_frame = self._start_frame + self._tick
    return D3Observation(
      state=self._state(),
      conditioning=np.full(99, 0.5, dtype=np.float64),
      tick=self._tick,
      sim_time_s=self._tick * self.period_seconds,
      segment_id=f"segment:{self._start_frame}",
      reference=reference,
      reference_frame=reference_frame,
    )

  def reset(self, *, seed: int, start_frame: int) -> D3Observation:
    self._start_frame = int(start_frame)
    self._tick = 0
    self._tilt = 0.0
    self._phase = "vae"
    if self.reset_raises:
      raise RuntimeError("reset refused")
    return self._observation()

  def initial_signature(self) -> np.ndarray:
    return np.zeros(3 + 4 + 31 + 31, dtype=np.float64)

  def enter_reference_free(self) -> D3Observation:
    self._phase = "reference_free"
    return self._observation()

  def step(self, action: np.ndarray) -> D3Transition:
    values = np.asarray(action, dtype=np.float64)
    if values.shape != (31,) or not np.isfinite(values).all():
      raise ValueError("action must be finite shape [31]")
    self.step_calls += 1
    self._tick += 1
    if self.tilt_after is not None and self._tick >= self.tilt_after:
      self._tilt = np.radians(80.0)
    terminated = self.terminate_after is not None and self._tick >= self.terminate_after
    boundary = (
      self._phase == "vae" and self._start_frame + self._tick >= self.clip_frames - 1
    )
    return D3Transition(
      observation=self._observation(),
      terminal_available=True,
      terminated=terminated,
      truncated=False,
      reset=False,
      teleport=False,
      reference_boundary=boundary,
      reason=None,
    )

  def close(self) -> None:
    self.closed = True


def _playback(
  *,
  mode: str,
  max_steps: int = 25,
  start_frame: int = 2,
  max_wall_seconds: float | None = None,
  session_start: float | None = None,
  runtime: _FakeRuntime | None = None,
  model: _FakeVae | None = None,
  continuations: list[_FakeContinuation] | None = None,
) -> tuple[VisualDiagnosticPlayback, _FakeRuntime, _FakeVae, list[_FakeContinuation]]:
  runtime = runtime or _FakeRuntime()
  model = model or _FakeVae()
  created: list[_FakeContinuation] = continuations if continuations is not None else []

  def factory() -> _FakeContinuation:
    continuation = _FakeContinuation()
    created.append(continuation)
    return continuation

  playback = VisualDiagnosticPlayback(
    runtime=runtime,
    mode=mode,
    model=model,
    continuation_factory=factory if mode == "diffusion" else None,
    seed=0,
    start_frame=start_frame,
    max_steps=max_steps,
    max_wall_seconds=max_wall_seconds,
    session_start=session_start,
  )
  playback.reset()
  return playback, runtime, model, created


def _tick(playback: VisualDiagnosticPlayback) -> None:
  playback.advance(playback.act(playback.observe()))


# Playback core.


def test_reset_starts_in_bootstrap_paused_and_unstopped() -> None:
  playback, runtime, model, _ = _playback(mode="diffusion")
  status = playback.status()
  assert status["phase"] == "bootstrap"
  assert status["stopped"] is False
  assert status["tick"] == 0
  assert status["bootstrap_completed"] == 0
  assert status["diagnostic_only"] is True
  assert status["matched_acceptance"] is False
  assert runtime.step_calls == 0
  assert model.encode_calls == 0


def test_repeated_render_is_cached_and_does_not_advance() -> None:
  playback, runtime, model, _ = _playback(mode="diffusion")
  first = playback.observe()
  second = playback.observe()
  assert first is second
  np.testing.assert_array_equal(first.conditioning, second.conditioning)
  assert runtime.step_calls == 0
  assert model.encode_calls == 0


def test_baseline_runs_to_step_budget_and_keeps_final_pose() -> None:
  playback, runtime, model, _ = _playback(mode="baseline", max_steps=3)
  for _ in range(3):
    _tick(playback)
  status = playback.status()
  assert status["branch"] == "baseline"
  assert status["phase"] == "tracking"
  assert status["stopped"] is True
  assert status["stop_reason"] == "step_budget"
  assert status["evaluation_steps"] == 3
  assert runtime.step_calls == 3
  assert model.encode_calls == 3
  frozen = playback.observe()
  playback.advance(playback.act(playback.observe()))
  assert playback.observe() is frozen
  assert runtime.step_calls == 3


def test_diffusion_bootstrap_handover_then_reference_free() -> None:
  playback, runtime, model, continuations = _playback(mode="diffusion", max_steps=2)
  for _ in range(8):
    _tick(playback)
  status = playback.status()
  assert status["phase"] == "reference_free"
  assert status["bootstrap_completed"] == 8
  assert status["stopped"] is False
  assert model.encode_calls == 8
  assert playback.observe().reference is None

  assert len(continuations) == 1
  assert len(continuations[0].records) == 8

  _tick(playback)
  after_step = playback.status()
  assert after_step["counters"]["encoder"] == 8  # no encoder in reference-free mode
  assert after_step["counters"]["decoder"] == 9
  assert after_step["counters"]["denoiser"] == 20
  assert runtime.step_calls == 9
  assert continuations[0].steps == 1

  _tick(playback)
  status = playback.status()
  assert status["stopped"] is True
  assert status["stop_reason"] == "step_budget"
  assert status["evaluation_steps"] == 2
  assert status["counters"]["denoiser"] == 40


def test_fall_stops_and_keeps_final_pose() -> None:
  runtime = _FakeRuntime(tilt_after=3)
  playback, _, _, _ = _playback(mode="diffusion", runtime=runtime)
  for _ in range(3):
    _tick(playback)
  status = playback.status()
  assert status["stopped"] is True
  assert status["stop_reason"] == "physical_fall"
  assert status["tilt_degrees"] > 70.0
  assert runtime.step_calls == 3


def test_reference_boundary_stops_before_bootstrap_completes() -> None:
  runtime = _FakeRuntime(clip_frames=4)
  playback, _, _, continuations = _playback(
    mode="diffusion", runtime=runtime, start_frame=0
  )
  for _ in range(4):
    _tick(playback)
  status = playback.status()
  assert status["stopped"] is True
  assert status["stop_reason"] == "reference_boundary"
  assert status["bootstrap_completed"] == 2
  assert continuations == []


def test_terminal_flag_stops_and_keeps_final_pose() -> None:
  runtime = _FakeRuntime(terminate_after=2)
  playback, _, _, _ = _playback(mode="diffusion", runtime=runtime)
  _tick(playback)
  assert playback.status()["stopped"] is False
  _tick(playback)
  status = playback.status()
  assert status["stopped"] is True
  assert status["stop_reason"] == "terminated"
  assert status["tick"] == 2
  frozen = playback.observe()
  assert frozen.tick == 2


def test_reset_clears_history_and_never_mixes_generations() -> None:
  playback, _, model, continuations = _playback(mode="diffusion", max_steps=5)
  for _ in range(9):
    _tick(playback)
  assert playback.status()["counters"]["encoder"] == 8
  assert model.encode_calls == 8
  first_generation = continuations[0]

  playback.reset()
  status = playback.status()
  assert status["phase"] == "bootstrap"
  assert status["bootstrap_completed"] == 0
  assert status["evaluation_steps"] == 0
  assert status["stopped"] is False
  assert status["counters"] == {"encoder": 0, "decoder": 0, "denoiser": 0}
  assert playback.observe().reference is not None

  for _ in range(8):
    _tick(playback)
  assert len(continuations) == 2
  assert continuations[1] is not first_generation
  assert len(continuations[1].records) == 8


def test_close_is_idempotent_and_closes_runtime() -> None:
  playback, runtime, _, _ = _playback(mode="baseline")
  playback.close()
  playback.close()
  assert runtime.closed is True


class _FakeClock:
  """Injectable monotonic clock for deterministic session-budget tests."""

  def __init__(self) -> None:
    self.now = 0.0

  def perf_counter(self) -> float:
    return self.now


def test_session_wall_budget_is_monotonic_across_idle_and_reset(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  clock = _FakeClock()
  monkeypatch.setattr(vp, "time", clock)
  playback, runtime, _, _ = _playback(
    mode="baseline", max_steps=25, max_wall_seconds=0.1
  )
  assert playback.session_deadline == pytest.approx(0.1)
  assert playback.session_expired is False

  # Idle clock advances far past the deadline with no step or render.
  clock.now = 3600.0
  assert playback.session_expired is True
  action = playback.act(playback.observe())
  assert playback.stopped is True
  assert playback.stop_reason == "wall_budget"
  assert runtime.step_calls == 0
  playback.advance(action)
  assert runtime.step_calls == 0

  # A user reset must not extend the fixed session budget.
  playback.reset()
  clock.now = 7200.0
  assert playback.session_expired is True
  playback.act(playback.observe())
  assert playback.stopped is True
  assert playback.stop_reason == "wall_budget"
  assert runtime.step_calls == 0


def test_slow_inference_is_bounded_by_the_session_deadline(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  clock = _FakeClock()
  monkeypatch.setattr(vp, "time", clock)
  model = _FakeVae()
  original_encode = model.encode

  def slow_encode(reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Inference itself consumed the whole session budget.
    clock.now = 3600.0
    return original_encode(reference)

  monkeypatch.setattr(model, "encode", slow_encode)
  playback, runtime, _, _ = _playback(
    mode="baseline", model=model, max_steps=25, max_wall_seconds=0.1
  )

  # An in-flight inference may finish, but its action must not be applied
  # after the fixed session deadline.
  first = playback.act(playback.observe())
  assert playback.stopped is False
  playback.advance(first)
  assert runtime.step_calls == 0
  assert playback.stopped is True
  assert playback.stop_reason == "wall_budget"

  playback.act(playback.observe())
  playback.advance(playback.act(playback.observe()))
  assert runtime.step_calls == 0


def test_viewer_run_stops_at_session_deadline_while_paused(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  clock = _FakeClock()
  monkeypatch.setattr(vp, "time", clock)
  playback, _, _, _ = _playback(mode="baseline", max_steps=25, max_wall_seconds=0.1)
  viewer, _ = _viewer(playback)
  assert viewer._is_paused is True
  assert viewer.is_running() is True
  clock.now = 3600.0
  assert viewer.is_running() is False
  assert playback.stopped is True
  assert playback.stop_reason == "wall_budget"


@pytest.mark.parametrize("tilt_after", [None, 1])
def test_missing_terminal_evidence_stops_before_history(
  monkeypatch: pytest.MonkeyPatch, tilt_after: int | None
) -> None:
  runtime = _FakeRuntime(tilt_after=tilt_after)
  playback, _, _, continuations = _playback(mode="diffusion", runtime=runtime)
  original = runtime.step

  def untrusted(action: np.ndarray) -> D3Transition:
    return replace(original(action), terminal_available=False)

  monkeypatch.setattr(runtime, "step", untrusted)
  _tick(playback)
  status = playback.status()
  assert status["stopped"] is True
  assert status["stop_reason"] == "missing_terminal_evidence"
  assert status["bootstrap_completed"] == 0
  assert playback.observe().tick == 0  # cached trusted pose preserved
  assert runtime.step_calls == 1
  assert continuations == []


def test_reset_flag_stops_without_adopting_transplanted_pose(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  runtime = _FakeRuntime()
  playback, _, _, _ = _playback(mode="baseline", runtime=runtime, max_steps=5)
  original = runtime.step

  def resetting(action: np.ndarray) -> D3Transition:
    return replace(original(action), reset=True)

  monkeypatch.setattr(runtime, "step", resetting)
  _tick(playback)
  status = playback.status()
  assert status["stopped"] is True
  assert status["stop_reason"] == "reset"
  assert status["tick"] == 0
  assert status["evaluation_steps"] == 0


def test_vae_action_matches_frozen_evaluator_helper() -> None:
  runtime = _FakeRuntime()
  observation = runtime.reset(seed=0, start_frame=2)
  reference_model = _FakeVae()
  counters = {"encoder": 0, "decoder": 0, "denoiser": 0}
  expected_latent, expected_action = _vae_action(
    reference_model, observation, counters, device=torch.device("cpu")
  )

  playback, _, playback_model, _ = _playback(
    mode="baseline", runtime=runtime, model=_FakeVae()
  )
  actual_latent, actual_action = playback._vae_action(playback.observe())

  np.testing.assert_allclose(actual_latent, expected_latent, rtol=0, atol=1e-6)
  np.testing.assert_allclose(actual_action, expected_action, rtol=0, atol=1e-6)
  assert counters["encoder"] == 1
  assert playback_model.encode_calls == 1


# Facades and viewer glue.


def test_env_and_policy_facades_delegate_to_playback() -> None:
  playback, runtime, _, _ = _playback(mode="baseline", max_steps=1)
  env = D3DiagnosticEnv(playback, render_env=MagicMock())
  policy = D3DiagnosticPolicy(playback)
  assert env.num_envs == 1
  observation = env.get_observations()
  action = policy(observation)
  env.step(action)
  assert runtime.step_calls == 1
  assert playback.status()["evaluation_steps"] == 1


def _viewer(playback: VisualDiagnosticPlayback) -> tuple[D3DiagnosticViewer, MagicMock]:
  env = MagicMock()
  command = MagicMock()
  env.unwrapped.command_manager.get_term.return_value = command
  viewer = D3DiagnosticViewer(
    env,
    MagicMock(),
    playback,
    viser_server=MagicMock(),
  )
  return viewer, command


def test_viewer_defaults_paused_without_constructing_a_server() -> None:
  playback, _, _, _ = _playback(mode="diffusion")
  viewer, _ = _viewer(playback)
  assert viewer._is_paused is True


def test_viewer_rejects_scrub_and_gui_reset_payloads(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  playback, _, _, _ = _playback(mode="diffusion")
  viewer, _ = _viewer(playback)
  scrub = MagicMock()
  gui_reset = MagicMock()
  monkeypatch.setattr(viewer, "_handle_motion_scrub", scrub)
  monkeypatch.setattr(viewer, "_handle_gui_reset", gui_reset)
  assert (
    viewer._handle_custom_action(
      ViewerAction.CUSTOM, {"type": "motion_scrub", "env_idx": 0, "frame": 5}
    )
    is True
  )
  assert (
    viewer._handle_custom_action(ViewerAction.CUSTOM, {"type": "gui_reset"}) is True
  )
  scrub.assert_not_called()
  gui_reset.assert_not_called()


def test_viewer_disables_reference_scrubber_handles() -> None:
  playback, _, _, _ = _playback(mode="diffusion")
  viewer, command = _viewer(playback)
  viewer._disable_reference_controls()
  command._set_scrubber_disabled.assert_called_once_with(True)


def test_viewer_status_html_marks_diagnostic_only() -> None:
  playback, _, _, _ = _playback(mode="diffusion")
  viewer, _ = _viewer(playback)
  viewer._status_html = cast(Any, SimpleNamespace(content=""))
  viewer._update_status_display()
  content = viewer._status_html.content
  assert "DIAGNOSTIC ONLY" in content
  assert "diffusion" in content
  assert "bootstrap" in content


def _render_viewer(playback: VisualDiagnosticPlayback) -> D3DiagnosticViewer:
  """Real facade/fake render env so stepping goes through the playback."""
  render = SimpleNamespace(
    cfg=SimpleNamespace(viewer=ViewerConfig()),
    step_dt=0.02,
    command_manager=MagicMock(),
  )
  env = D3DiagnosticEnv(playback, render_env=render)
  viewer = D3DiagnosticViewer(
    env, D3DiagnosticPolicy(playback), playback, viser_server=MagicMock()
  )
  viewer._scene = cast(Any, SimpleNamespace(needs_update=False, env_idx=0))
  viewer._pending_update_reasons = set()
  viewer._pause_button = cast(Any, SimpleNamespace(label="Pause", icon=None))
  viewer._status_html = cast(Any, SimpleNamespace(content=""))
  return viewer


def test_stop_on_accepted_step_pauses_and_submits_final_pose() -> None:
  playback, _, _, _ = _playback(mode="baseline", max_steps=1)
  viewer = _render_viewer(playback)
  assert viewer._is_paused is True
  viewer.resume()

  viewer._step_physics(0.04)

  assert playback.stopped is True
  assert playback.stop_reason == "step_budget"
  assert viewer._is_paused is True
  assert viewer._pause_button.label == "Play"
  has_pending = bool(viewer._pending_update_reasons) or viewer._scene.needs_update
  assert viewer._should_submit_scene_update(2, viewer._is_paused, has_pending) is True


def test_single_step_stop_pauses_and_submits_final_pose() -> None:
  playback, _, _, _ = _playback(mode="baseline", max_steps=1)
  viewer = _render_viewer(playback)
  assert viewer._is_paused is True

  viewer._single_step()

  assert playback.stopped is True
  assert playback.stop_reason == "step_budget"
  assert viewer._is_paused is True
  assert viewer._pause_button.label == "Play"
  has_pending = bool(viewer._pending_update_reasons) or viewer._scene.needs_update
  assert viewer._should_submit_scene_update(2, viewer._is_paused, has_pending) is True


# CLI gates.


def _fake_prepared(tmp_path: Path) -> SimpleNamespace:
  args = Namespace(
    branch="diffusion",
    host="127.0.0.1",
    port=8080,
    frame_rate=60.0,
    continuation_steps=5,
    max_wall_seconds=120.0,
    start_frame=3,
    member="tennis_000",
    seed=0,
    vae_checkpoint=tmp_path / "vae.pt",
    vae_checkpoint_sha256="a" * 64,
    contract=tmp_path / "contract.yaml",
    output_dir=tmp_path / "run",
  )
  return SimpleNamespace(
    args=args,
    contract=DEFAULT_CONTRACT,
    clip_frames=100,
    clip_fps=50.0,
    manifest_sha256="b" * 64,
    denoiser=SimpleNamespace(global_step=100, ema_selected=True),
    device=torch.device("cpu"),
    repo_root=tmp_path,
    integrity_before={"files": {}},
  )


def _artifact_args(tmp_path: Path, *, execute: bool) -> list[str]:
  args = [
    "--contract",
    "contract.yaml",
    "--contract-sha256",
    "a" * 64,
    "--checkpoint",
    "checkpoint.pt",
    "--checkpoint-sha256",
    "a" * 64,
    "--projection",
    "projection.npz",
    "--projection-sha256",
    "a" * 64,
    "--vae-checkpoint",
    "vae.pt",
    "--vae-checkpoint-sha256",
    "a" * 64,
    "--manifest",
    "manifest.yaml",
    "--manifest-sha256",
    "a" * 64,
    "--repo-root",
    str(tmp_path),
    "--member",
    "tennis_000",
    "--seed",
    "0",
    "--start-frame",
    "3",
    "--output-dir",
    str(tmp_path / "run"),
    "--branch",
    "diffusion",
  ]
  if execute:
    args.append("--execute")
  return args


def test_cli_help_exits_without_construction(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(
    view,
    "_run_viewer",
    lambda prepared: (_ for _ in ()).throw(AssertionError("viewer launched")),
  )
  with pytest.raises(SystemExit) as excinfo:
    view.main(["--help"])
  assert excinfo.value.code == 0


def test_cli_without_execute_blocks_dispatch(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  prepared = _fake_prepared(tmp_path)
  monkeypatch.setattr(view.evaluate, "_prepare", lambda args: prepared)
  monkeypatch.setattr(
    view,
    "_run_viewer",
    lambda prepared: (_ for _ in ()).throw(AssertionError("viewer launched")),
  )
  code = view.main(_artifact_args(tmp_path, execute=False))
  assert code == 0
  out = capsys.readouterr().out
  assert '"diagnostic_only": true' in out
  assert '"matched_acceptance": false' in out
  assert "not-run" in out


def test_cli_with_execute_dispatches_viewer(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  prepared = _fake_prepared(tmp_path)
  monkeypatch.setattr(view.evaluate, "_prepare", lambda args: prepared)
  calls: list[object] = []
  monkeypatch.setattr(view, "_run_viewer", lambda value: calls.append(value) or 0)
  assert view.main(_artifact_args(tmp_path, execute=True)) == 0
  assert calls == [prepared]


def test_viewer_options_reject_bad_bind_and_rate() -> None:
  with pytest.raises(view.evaluate.ClosedLoopCliError, match="port"):
    view._validate_viewer_options(Namespace(host="127.0.0.1", port=0, frame_rate=60))
  with pytest.raises(view.evaluate.ClosedLoopCliError, match="frame-rate"):
    view._validate_viewer_options(Namespace(host="127.0.0.1", port=8080, frame_rate=0))
  with pytest.raises(view.evaluate.ClosedLoopCliError, match="host"):
    view._validate_viewer_options(Namespace(host="", port=8080, frame_rate=60))


# Runtime ownership on early setup errors.


def _runtime_prepared(tmp_path: Path) -> SimpleNamespace:
  prepared = _fake_prepared(tmp_path)
  prepared.denoiser.model = MagicMock()
  prepared.vae = SimpleNamespace(model=MagicMock())
  return prepared


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, runtime: _FakeRuntime) -> None:
  monkeypatch.setattr(
    view.evaluate, "_runtime_factory", lambda prepared: lambda: runtime
  )
  monkeypatch.setattr(
    view.evaluate,
    "_continuation_factory",
    lambda prepared: lambda: _FakeContinuation(),
  )


def test_run_viewer_closes_runtime_when_playback_construction_fails(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  prepared = _runtime_prepared(tmp_path)
  runtime = _FakeRuntime()
  _patch_runtime(monkeypatch, runtime)
  monkeypatch.setattr(
    vp, "VisualDiagnosticPlayback", MagicMock(side_effect=RuntimeError("boom"))
  )

  assert view._run_viewer(prepared) == 1
  assert runtime.closed is True


def test_run_viewer_closes_runtime_when_reset_fails(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  prepared = _runtime_prepared(tmp_path)
  runtime = _FakeRuntime(reset_raises=True)
  _patch_runtime(monkeypatch, runtime)

  assert view._run_viewer(prepared) == 1
  assert runtime.closed is True
