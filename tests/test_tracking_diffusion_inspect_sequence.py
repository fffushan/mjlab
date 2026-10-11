"""CPU-only tests for the one-shot D3 future-pose inspection CLI.

The tests use a fake runtime, fake frozen models, a fake continuation and a fake
``sequence_poses`` helper.  They never construct MuJoCo, start a Viser server or
claim physical evidence.  The IK solver is owned by the sibling pose/IK lane, so
this suite exercises the CLI's use of its API, not the solver itself.
"""

from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from mjlab.tasks.tracking.diffusion.closed_loop_types import (
  D3Observation,
  D3Transition,
)
from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT
from mjlab.tasks.tracking.diffusion.state import WorldState

_CLI_PATH = (
  Path(__file__).parents[1]
  / "scripts/experiments/diffusion_d3_closed_loop/inspect_sequence.py"
)
_CLI_SPEC = spec_from_file_location("d3_future_pose_inspection", _CLI_PATH)
assert _CLI_SPEC is not None and _CLI_SPEC.loader is not None
inspect_sequence = module_from_spec(_CLI_SPEC)
sys.modules[_CLI_SPEC.name] = inspect_sequence
_CLI_SPEC.loader.exec_module(inspect_sequence)
evaluate = inspect_sequence.evaluate
_ORIGINAL_OPEN_VIEWER = inspect_sequence._open_viewer

_BOOTSTRAP_TRANSITIONS = 8
_BODIES = 20
_JOINTS = 31
_NQ = 7 + _JOINTS
_PERIOD = 0.02
_SAMPLE_SHAPE = (41, 167)
_FUTURE_STEPS = 32


def _state(offset: float = 0.0) -> WorldState:
  return WorldState(
    root_position=np.array([0.0, 0.0, 0.80 + offset]),
    root_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    root_linear_velocity=np.zeros(3),
    root_angular_velocity=np.zeros(3),
    body_positions=np.zeros((_BODIES, 3)),
    body_linear_velocities=np.zeros((_BODIES, 3)),
  )


def _observation(tick: int, *, reference: bool = True) -> D3Observation:
  return D3Observation(
    state=_state(offset=0.001 * tick),
    conditioning=np.zeros(99),
    tick=tick,
    sim_time_s=_PERIOD * tick,
    segment_id="vgk005:seed=0:frame=45",
    reference=np.zeros(68) if reference else None,
    reference_frame=45 + tick if reference else None,
  )


class _FakeRuntime:
  """One-environment runtime stand-in: counted transitions, no physics."""

  period_seconds = _PERIOD
  clip_frames = 400
  joint_order = tuple(f"joint_{index}" for index in range(_JOINTS))
  body_order = tuple(DEFAULT_CONTRACT.body_names)
  provenance = {"member": "vgk005"}

  def __init__(self, *, fail_at: int | None = None) -> None:
    self.fail_at = fail_at
    self.step_calls = 0
    self.handovers = 0
    self.closes = 0
    self.tick = 0

  def reset(self, *, seed: int, start_frame: int) -> D3Observation:
    assert (seed, start_frame) == (0, 45)
    self.tick = 0
    return _observation(0)

  def step(self, action: np.ndarray) -> D3Transition:
    self.step_calls += 1
    assert action.shape == (31,), "the executed action must be the decoded VAE action"
    if self.fail_at is not None and self.step_calls >= self.fail_at:
      raise RuntimeError("simulator step failed")
    self.tick += 1
    return D3Transition(observation=_observation(self.tick))

  def enter_reference_free(self) -> D3Observation:
    self.handovers += 1
    return _observation(self.tick, reference=False)

  def close(self) -> None:
    self.closes += 1


class _FakeVae:
  """Deterministic stand-in for the frozen VAE decoder."""

  def __init__(self) -> None:
    self.encodes = 0

  def to(self, *_args: Any, **_kwargs: Any) -> "_FakeVae":
    return self

  def eval(self) -> "_FakeVae":
    return self

  def encode(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    self.encodes += 1
    batch = reference.shape[0]
    return torch.full((batch, 32), 0.25), torch.zeros(batch, 32)

  def decode(self, latent: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
    assert latent.shape == (1, 32)
    assert conditioning.shape == (1, 99)
    return torch.zeros(latent.shape[0], 31)


class _FakeDenoiser:
  def to(self, *_args: Any, **_kwargs: Any) -> "_FakeDenoiser":
    return self

  def eval(self) -> "_FakeDenoiser":
    return self


class _FakeContinuation:
  """Records history and the plan call; ``step`` must never be reached."""

  def __init__(self) -> None:
    self.recorded: list[tuple[Any, np.ndarray]] = []
    self.plan_calls: list[tuple[Any, int]] = []
    self.plans: list[Any] = []

  def record_executed(self, state: Any, latent: np.ndarray) -> None:
    self.recorded.append((state, np.asarray(latent, dtype=np.float64)))

  def plan(self, current_state: Any, *, tick: int) -> SimpleNamespace:
    self.plan_calls.append((current_state, int(tick)))
    plan = SimpleNamespace(
      clean_tokens=torch.zeros(1, *_SAMPLE_SHAPE),
      clean_mask=torch.zeros(1, *_SAMPLE_SHAPE, dtype=torch.bool),
    )
    self.plans.append(plan)
    return plan

  def step(self, *_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("the diffusion action must never be executed")


class _FakeConditioning:
  """Active D4 task conditioning stand-in."""

  def __init__(self) -> None:
    self.active = True
    self.sampler_calls: list[tuple[Any, Any]] = []
    self.closed = 0

  def sampler_factory(self, plan: Any, current_state: Any) -> Any:
    self.sampler_calls.append((plan, current_state))

    def sampler(_denoiser: Any, **_kwargs: Any) -> tuple[torch.Tensor, SimpleNamespace]:
      return torch.ones(1, *_SAMPLE_SHAPE) * 0.5, SimpleNamespace(denoiser_calls=7)

    return sampler

  def close(self, *, stop_reason: str | None = None) -> None:
    del stop_reason
    self.closed += 1


class _Stack:
  """The patched CLI collaborators one test run needs."""

  def __init__(
    self,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_at: int | None = None,
  ) -> None:
    self.runtime = _FakeRuntime(fail_at=fail_at)
    self.vae = _FakeVae()
    self.continuation = _FakeContinuation()
    self.sample_calls: list[Any] = []
    self.recover_calls: list[tuple[Any, Any, Any, Any]] = []
    self.fit_calls: list[tuple[Any, dict[str, Any]]] = []
    self.saved_models: list[Path] = []
    self.prepared = _prepared_for(
      tmp_path, self.vae, _argv(tmp_path, "--execute", "--no-view")
    )
    monkeypatch.setattr(
      evaluate, "_runtime_factory", lambda prepared: lambda: self.runtime
    )
    monkeypatch.setattr(
      evaluate,
      "_continuation_factory",
      lambda prepared, conditioning=None: lambda: self.continuation,
    )
    monkeypatch.setattr(
      inspect_sequence,
      "_isolated_model_context",
      lambda runtime: (
        SimpleNamespace(nq=_NQ),
        np.zeros(7, dtype=np.int64),
        np.arange(_JOINTS, dtype=np.int64),
        np.zeros(_NQ),
      ),
    )
    monkeypatch.setattr(inspect_sequence, "_sequence_poses", lambda: _helper(self))
    monkeypatch.setattr(
      inspect_sequence, "_sampler_config", lambda prepared: "frozen-sampler-config"
    )
    monkeypatch.setattr(inspect_sequence, "_save_model", self._save_model)
    monkeypatch.setattr(inspect_sequence, "_open_viewer", self._open_viewer)
    monkeypatch.setattr(inspect_sequence, "_task_conditioning", lambda prepared: None)
    self.sampler_module = sys.modules["mjlab.tasks.tracking.diffusion.sampler"]
    monkeypatch.setattr(
      self.sampler_module, "sample_trajectory", self.sample_trajectory
    )

  def _save_model(self, model: Any, path: Path) -> None:
    path.write_bytes(b"mjb")
    self.saved_models.append(path)

  def _open_viewer(
    self, display: Any, *, model_path: Path, host: str, port: int
  ) -> None:
    raise AssertionError("a Viser server must not be opened in these tests")

  def sample_trajectory(
    self, denoiser: Any, **kwargs: Any
  ) -> tuple[torch.Tensor, SimpleNamespace]:
    self.sample_calls.append({"denoiser": denoiser, **kwargs})
    return torch.zeros(1, *_SAMPLE_SHAPE), SimpleNamespace(denoiser_calls=20)


def _helper(stack: _Stack) -> SimpleNamespace:
  def recover_future_poses(
    sample: Any, *, projection: Any, current_state: Any, contract: Any
  ) -> dict[str, np.ndarray]:
    stack.recover_calls.append(
      (np.asarray(sample), projection, current_state, contract)
    )
    return {
      "root_positions": np.zeros((_FUTURE_STEPS, 3)),
      "root_rotations": np.tile(np.eye(3), (_FUTURE_STEPS, 1, 1)),
      "body_positions": np.zeros((_FUTURE_STEPS, _BODIES, 3)),
    }

  def fit_pose_sequence(model: Any, **kwargs: Any) -> dict[str, np.ndarray]:
    stack.fit_calls.append((model, kwargs))
    nq = model.nq
    return {
      "qpos": np.zeros((_FUTURE_STEPS, nq)),
      "body_errors": np.full((_FUTURE_STEPS, _BODIES), 0.25),
      "converged": np.ones(_FUTURE_STEPS, dtype=bool),
      "iterations": np.full(_FUTURE_STEPS, 3, dtype=np.int64),
    }

  return SimpleNamespace(
    recover_future_poses=recover_future_poses, fit_pose_sequence=fit_pose_sequence
  )


def _argv(tmp_path: Path, *extra: str) -> list[str]:
  return [
    "--checkpoint",
    "checkpoint.pt",
    "--member",
    "vgk005",
    "--seed",
    "0",
    "--start-frame",
    "45",
    "--output-dir",
    str(tmp_path / "run"),
    "--device",
    "cpu",
    *extra,
  ]


def _prepared_for(tmp_path: Path, vae: _FakeVae, argv: list[str]) -> SimpleNamespace:
  args = inspect_sequence.build_parser().parse_args(argv)
  return SimpleNamespace(
    args=args,
    repo_root=tmp_path,
    contract=DEFAULT_CONTRACT,
    projection=object(),
    denoiser=SimpleNamespace(
      model=_FakeDenoiser(), schedule=object(), grid=object(), global_step=104_000
    ),
    vae=SimpleNamespace(model=vae),
    clip_frames=400,
    clip_fps=50.0,
    device=torch.device("cpu"),
    task=None,
  )


def test_parser_allows_replay_without_a_checkpoint() -> None:
  parser = inspect_sequence.build_parser()
  args = parser.parse_args(["--replay", "saved-run", "--no-view"])
  assert args.replay == Path("saved-run")
  assert args.no_view is True
  assert args.checkpoint is None


def test_no_execute_preflights_without_a_runtime_sample_or_server(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
) -> None:
  stack = _Stack(tmp_path, monkeypatch)
  argv = _argv(tmp_path)
  monkeypatch.setattr(
    evaluate, "_prepare", lambda args: _prepared_for(tmp_path, stack.vae, argv)
  )
  monkeypatch.setattr(
    inspect_sequence,
    "generate",
    lambda prepared: pytest.fail("generation dispatched without --execute"),
  )
  assert inspect_sequence.main(argv) == 0
  payload = json.loads(capsys.readouterr().out)
  assert payload["execution"].startswith("not-run")
  assert payload["budget"]["diffusion_action_executed"] is False
  assert payload["budget"]["bootstrap_transitions"] == _BOOTSTRAP_TRANSITIONS
  assert payload["viewer"]["enabled"] is True
  assert stack.runtime.step_calls == 0
  assert stack.runtime.closes == 0


def test_generate_runs_eight_bootstrap_transitions_and_samples_once(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  stack = _Stack(tmp_path, monkeypatch)
  status = inspect_sequence.generate(stack.prepared)

  # Exactly eight real VAE bootstrap transitions, one handover, one close.
  assert stack.runtime.step_calls == _BOOTSTRAP_TRANSITIONS
  assert stack.runtime.handovers == 1
  assert stack.runtime.closes == 1
  assert stack.vae.encodes == _BOOTSTRAP_TRANSITIONS
  assert len(stack.continuation.recorded) == _BOOTSTRAP_TRANSITIONS
  assert all(latent.shape == (32,) for _, latent in stack.continuation.recorded)
  # One complete sample, planned at the handed observation's tick.
  assert len(stack.sample_calls) == 1
  assert stack.continuation.plan_calls[0][1] == _BOOTSTRAP_TRANSITIONS
  assert stack.sample_calls[0]["config"] == "frozen-sampler-config"
  assert stack.sample_calls[0]["conditions"].clean_tokens.shape == (1, *_SAMPLE_SHAPE)
  # One pose recovery and exactly one isolated IK call.
  assert len(stack.recover_calls) == 1
  sample, _, current_state, contract = stack.recover_calls[0]
  assert sample.shape == _SAMPLE_SHAPE
  assert not np.any(sample)
  assert current_state is stack.continuation.plan_calls[0][0]
  assert contract is DEFAULT_CONTRACT
  assert len(stack.fit_calls) == 1
  model, kwargs = stack.fit_calls[0]
  assert model.nq == _NQ
  assert kwargs["max_iterations"] == inspect_sequence._MAX_IK_ITERATIONS
  assert kwargs["body_names"] == DEFAULT_CONTRACT.body_names
  assert kwargs["initial_qpos"].shape == (_NQ,)
  assert kwargs["root_positions"].shape == (_FUTURE_STEPS, 3)
  assert kwargs["root_rotations"].shape == (_FUTURE_STEPS, 3, 3)
  assert kwargs["body_positions"].shape == (_FUTURE_STEPS, _BODIES, 3)
  # Bounded status and saved artifacts.
  assert status["diffusion_action_executed"] is False
  assert status["physics_after_handover"] is False
  assert status["bootstrap_transitions"] == _BOOTSTRAP_TRANSITIONS
  assert status["plan_tick"] == _BOOTSTRAP_TRANSITIONS
  assert status["future_ticks"] == list(range(9, 41))
  assert status["ik"]["solver_calls"] == 1
  assert status["ik"]["converged"] == _FUTURE_STEPS
  assert status["sampler"]["denoiser_calls"] == 20
  assert stack.saved_models == [Path(status["artifacts"]["model"])]
  assert Path(status["artifacts"]["poses"]).is_file()
  assert (tmp_path / "run" / inspect_sequence._STATUS_NAME).is_file()


def test_execute_generates_and_opens_playback_in_one_command(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  stack = _Stack(tmp_path, monkeypatch)
  opened: list[Any] = []

  def prepare(args: Any) -> Any:
    stack.prepared.args = args
    return stack.prepared

  def open_viewer(display: Any, *, model_path: Path, host: str, port: int) -> None:
    # Generation and simulator cleanup finish before playback starts.
    assert stack.runtime.step_calls == 8
    assert stack.runtime.closes == 1
    assert model_path.is_file()
    assert display.qpos.shape == (32, _NQ)
    opened.append(display)

  monkeypatch.setattr(evaluate, "_prepare", prepare)
  monkeypatch.setattr(inspect_sequence, "_open_viewer", open_viewer)
  assert inspect_sequence.main(_argv(tmp_path, "--execute")) == 0
  assert stack.prepared.args.replay is None
  assert len(stack.sample_calls) == 1
  assert len(stack.fit_calls) == 1
  assert len(opened) == 1


def test_task_conditioning_installs_the_task_sampler(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  stack = _Stack(tmp_path, monkeypatch)
  conditioning = _FakeConditioning()
  monkeypatch.setattr(
    inspect_sequence, "_task_conditioning", lambda prepared: conditioning
  )
  monkeypatch.setattr(
    stack.sampler_module,
    "sample_trajectory",
    lambda *args, **kwargs: pytest.fail("the core sampler must not be used"),
  )
  status = inspect_sequence.generate(stack.prepared)

  assert stack.sample_calls == []
  assert len(conditioning.sampler_calls) == 1
  plan, current_state = conditioning.sampler_calls[0]
  assert plan is stack.continuation.plans[0]
  assert current_state is stack.continuation.plan_calls[0][0]
  assert conditioning.closed == 1
  # The task sampler's sample is the one recovered and fitted.
  assert np.allclose(stack.recover_calls[0][0], 0.5)
  # The task diagnostics, not the core sampler count, are reported.
  assert status["sampler"]["task_conditioning_active"] is True
  assert status["sampler"]["denoiser_calls"] == 7


def test_task_conditioning_is_absent_without_a_task_config(tmp_path: Path) -> None:
  prepared = _prepared_for(tmp_path, _FakeVae(), _argv(tmp_path, "--execute"))
  assert prepared.task is None
  assert inspect_sequence._task_conditioning(prepared) is None


def test_runtime_is_closed_when_a_bootstrap_transition_fails(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  stack = _Stack(tmp_path, monkeypatch, fail_at=3)
  monkeypatch.setattr(evaluate, "_prepare", lambda args: stack.prepared)
  # An unexpected simulator failure keeps its traceback but still runs cleanup.
  with pytest.raises(RuntimeError, match="simulator step failed"):
    inspect_sequence.main(_argv(tmp_path, "--execute", "--no-view"))
  assert stack.runtime.closes == 1
  assert stack.runtime.step_calls == 3
  assert stack.runtime.handovers == 0
  assert stack.saved_models == []


def test_sampling_failure_closes_the_runtime_and_leaves_no_artifacts(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  stack = _Stack(tmp_path, monkeypatch)

  def _fail(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("sampler failed")

  monkeypatch.setattr(stack.sampler_module, "sample_trajectory", _fail)
  with pytest.raises(RuntimeError, match="sampler failed"):
    inspect_sequence.generate(stack.prepared)
  assert stack.runtime.closes == 1
  assert stack.saved_models == []
  assert not (tmp_path / "run" / inspect_sequence._NPZ_NAME).exists()


def test_replay_reopens_saved_poses_without_neural_artifacts_or_physics(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
) -> None:
  stack = _Stack(tmp_path, monkeypatch)
  inspect_sequence.generate(stack.prepared)
  directory = tmp_path / "run"
  assert stack.runtime.closes == 1

  def _explode(args: Any) -> Any:
    raise AssertionError("replay must not load frozen artifacts")

  opened: list[tuple[Any, Path, str, int]] = []

  def _record_open(display: Any, *, model_path: Path, host: str, port: int) -> None:
    opened.append((display, model_path, host, port))

  monkeypatch.setattr(evaluate, "_prepare", _explode)
  monkeypatch.setattr(inspect_sequence, "_open_viewer", _record_open)

  code = inspect_sequence.main(["--replay", str(directory), "--execute", "--no-view"])
  assert code == 0
  summary = json.loads(capsys.readouterr().out)
  assert summary["plan_tick"] == _BOOTSTRAP_TRANSITIONS
  assert summary["future_steps"] == _FUTURE_STEPS
  assert summary["future_ticks"] == list(range(9, 41))
  assert summary["ik"]["converged"] == _FUTURE_STEPS
  assert opened == []
  # Generation ran exactly once, so replay stepped no physics and fitted no IK.
  assert stack.runtime.step_calls == _BOOTSTRAP_TRANSITIONS
  assert len(stack.fit_calls) == 1

  code = inspect_sequence.main(["--replay", str(directory)])
  assert code == 0
  preflight = json.loads(capsys.readouterr().out)
  assert preflight["replay"]["neural_artifacts_loaded"] is False
  assert preflight["replay"]["physics_steps"] == 0
  assert preflight["execution"].startswith("not-run")
  assert opened == []

  code = inspect_sequence.main(["--replay", str(directory), "--execute"])
  assert code == 0
  assert len(opened) == 1
  assert opened[0][1] == directory / inspect_sequence._MODEL_NAME
  assert opened[0][3] == inspect_sequence._DEFAULT_PORT


def test_replay_fails_closed_on_a_missing_directory(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  code = inspect_sequence.main(["--replay", str(tmp_path / "absent")])
  assert code == 1
  assert "replay directory is not a directory" in capsys.readouterr().err


def test_replay_fails_closed_on_a_truncated_bundle(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  directory = tmp_path / "run"
  directory.mkdir()
  np.savez(directory / inspect_sequence._NPZ_NAME, qpos=np.zeros((3, _NQ)))
  (directory / inspect_sequence._MODEL_NAME).write_bytes(b"mjb")
  code = inspect_sequence.main(["--replay", str(directory), "--execute", "--no-view"])
  assert code == 1
  assert "saved pose bundle is missing fields" in capsys.readouterr().err


def test_replay_reports_one_dimensional_qpos(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  stack = _Stack(tmp_path, monkeypatch)
  inspect_sequence.generate(stack.prepared)
  path = tmp_path / "run" / inspect_sequence._NPZ_NAME
  with np.load(path) as stored:
    arrays = {key: stored[key] for key in stored.files}
  arrays["qpos"] = np.zeros(_NQ)
  np.savez(path, **arrays)
  assert (
    inspect_sequence.main(["--replay", str(path.parent), "--execute", "--no-view"]) == 1
  )
  assert "saved qpos must have shape" in capsys.readouterr().err


def test_static_scene_controls_render_without_sampling(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  import mujoco

  import mjlab.viewer.viser.scene as scene_module

  stack = _Stack(tmp_path, monkeypatch)
  inspect_sequence.generate(stack.prepared)
  display = inspect_sequence._load_display(
    inspect_sequence._replay_paths(tmp_path / "run")
  )
  controls: dict[str, Any] = {}
  clock = {"now": 0.0}
  monkeypatch.setattr(inspect_sequence.time, "perf_counter", lambda: clock["now"])
  monkeypatch.setattr(mujoco, "mj_kinematics", lambda *args: None)
  display.qpos[:, 0] = np.arange(32)
  display.body_positions[:, :, 0] = np.arange(32)[:, None]
  render_data = SimpleNamespace(qpos=np.zeros(_NQ))

  class Control:
    def __init__(self, label: str, **kwargs: Any) -> None:
      self.label = label
      self._value = kwargs.get("initial_value", False)
      self.callback: Any = None

    @property
    def value(self) -> Any:
      return self._value

    @value.setter
    def value(self, value: Any) -> None:
      if value == self._value:
        return
      self._value = value
      if self.callback is not None:
        # Match Viser: server-side assignments invoke callbacks synchronously.
        self.callback(None)

    def on_update(self, callback: Any) -> None:
      self.callback = callback

    on_click = on_update

  def control(label: str, **kwargs: Any) -> Any:
    handle = Control(label, **kwargs)
    controls[label] = handle
    return handle

  class Scene:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
      self._ghost_handles = {0: SimpleNamespace(opacity=0.35)}
      self.camera_tracking_enabled = True
      self.ghosts: list[Any] = []
      self.points: list[Any] = []
      self.renders: list[int] = []

    def clear(self) -> None:
      self.ghosts.clear()
      self.points.clear()

    def add_ghost_mesh(self, qpos: Any, *args: Any, **kwargs: Any) -> None:
      self.ghosts.append(qpos)

    def add_sphere(self, point: Any, *args: Any) -> None:
      self.points.append(point)

    def update_from_mjdata(self, data: Any) -> None:
      self.renders.append(len(self.ghosts))

  scene = Scene()
  monkeypatch.setattr(scene_module, "MjlabViserScene", lambda *a, **kw: scene)
  look_at = np.array([-5.0, -2.0, 0.7])
  monkeypatch.setattr(inspect_sequence, "_camera_frame", lambda *args: (look_at, 0.9))
  client = SimpleNamespace(camera=SimpleNamespace())
  connections: list[Any] = []
  server = SimpleNamespace(
    gui=SimpleNamespace(
      add_html=lambda value: SimpleNamespace(content=value),
      add_folder=lambda label: nullcontext(),
      add_checkbox=control,
      add_slider=control,
      add_button=control,
    ),
    on_client_connect=connections.append,
    get_clients=lambda: {0: client},
  )
  tick = inspect_sequence._build_static_scene(
    display, SimpleNamespace(stat=SimpleNamespace(extent=1e6)), render_data, server
  )
  assert not scene.camera_tracking_enabled
  assert np.allclose(client.camera.look_at, look_at)
  assert np.linalg.norm(client.camera.position - look_at) < 4.0
  assert client.camera.up_direction == (0.0, 0.0, 1.0)
  new_client = SimpleNamespace(camera=SimpleNamespace())
  connections[0](new_client)
  assert np.allclose(new_client.camera.look_at, look_at)
  client.camera.look_at = np.zeros(3)
  controls["Center view"].callback(None)
  assert np.allclose(client.camera.look_at, look_at)
  assert scene.renders == [0]  # Single fitted robot, initially paused at pose 1.
  assert controls["Future index"].value == 1
  assert controls["All ghosts"].value is False
  tick(10.0)
  assert controls["Future index"].value == 1
  controls["All ghosts"].value = True
  assert scene.renders[-1] == 32
  controls["Ghost opacity"].value = 0.7
  assert scene._ghost_handles[0].opacity == 0.7
  controls["Future index"].value = 32
  assert scene.renders[-1] == 0
  assert render_data.qpos[0] == 31
  controls["Raw body points"].value = True
  assert len(scene.points) == 20
  assert scene.points[0][0] == 31

  # Step and reset always pause. Step wraps the final pose back to pose 1.
  controls["Step"].callback(None)
  assert controls["Future index"].value == 1
  assert controls["Play"].label == "Play"
  controls["Step"].callback(None)
  assert controls["Future index"].value == 2
  controls["Reset"].callback(None)
  assert controls["Future index"].value == 1

  # Default 0.25x: advance after 0.08 s, never on render-only/pause.
  controls["Play"].callback(None)
  assert controls["Play"].label == "Pause"
  tick(0.079)
  assert controls["Future index"].value == 1
  tick(0.08)
  assert controls["Future index"].value == 2
  assert controls["Play"].label == "Pause"  # Slider sync must not pause playback.
  controls["Play"].callback(None)
  tick(100.0)
  assert controls["Future index"].value == 2

  controls["Reset"].callback(None)
  controls["Play"].callback(None)
  for expected_index in range(2, 33):
    clock["now"] += 0.1
    tick(clock["now"])
    assert controls["Future index"].value == expected_index
    assert render_data.qpos[0] == expected_index - 1
  clock["now"] += 0.1
  tick(clock["now"])
  assert controls["Future index"].value == 1
  clock["now"] += 100.0  # A delayed render advances one pose, not a catch-up burst.
  tick(clock["now"])
  assert controls["Future index"].value == 2
  controls["Future index"].value = 20
  assert controls["Play"].label == "Play"  # Manual seek pauses.
  tick(clock["now"] + 100.0)
  assert controls["Future index"].value == 20
  controls["Reset"].callback(None)
  controls["Playback speed"].value = 0.5
  controls["Play"].callback(None)
  tick(clock["now"] + 0.03)
  assert controls["Future index"].value == 1
  tick(clock["now"] + 0.041)
  assert controls["Future index"].value == 2
  controls["Reset"].callback(None)
  assert controls["Play"].label == "Play"
  assert controls["Future index"].value == 1
  assert stack.runtime.step_calls == 8
  assert len(stack.sample_calls) == 1
  assert len(stack.fit_calls) == 1


def test_camera_frames_fitted_bodies_far_from_world_origin() -> None:
  import mujoco

  model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <geom type="plane" size="100 100 0.1"/>
      <body name="robot/pelvis"><freejoint/>
        <geom type="sphere" size="0.1"/>
        <body name="robot/head" pos="0 0 1"><geom type="sphere" size="0.1"/></body>
      </body>
    </worldbody></mujoco>
  """)
  model.stat.extent = 10000.0
  center = np.array([10.0, -7.0, 0.5, 1.0, 0.0, 0.0, 0.0])
  qpos = np.tile(center, (32, 1))
  qpos[:, 0] += np.linspace(0.0, 1.0, 32)
  display = inspect_sequence.FuturePoseDisplay(
    qpos=qpos,
    center_qpos=center,
    root_positions=qpos[:, :3],
    body_positions=np.full((32, 20, 3), 1e6),  # Raw targets must not drive zoom.
    body_errors=np.zeros((32, 20)),
    converged=np.ones(32, dtype=bool),
    iterations=np.zeros(32, dtype=int),
    plan_tick=8,
    member="test",
    seed=0,
    start_frame=0,
  )
  before = qpos.copy()
  look_at, radius = inspect_sequence._camera_frame(display, model)
  assert np.allclose(look_at, [10.5, -7.0, 1.0])
  assert 0.8 < radius < 1.2
  assert np.array_equal(display.qpos, before)


@pytest.mark.parametrize("fail_setup", [True, False])
def test_pose_viewer_stops_server_on_setup_failure_or_interrupt(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_setup: bool
) -> None:
  import mujoco
  import viser

  stack = _Stack(tmp_path, monkeypatch)
  inspect_sequence.generate(stack.prepared)
  display = inspect_sequence._load_display(
    inspect_sequence._replay_paths(tmp_path / "run")
  )
  stops: list[bool] = []
  server = SimpleNamespace(stop=lambda: stops.append(True))
  monkeypatch.setattr(viser, "ViserServer", lambda **kwargs: server)
  monkeypatch.setattr(
    mujoco,
    "MjModel",
    SimpleNamespace(from_binary_path=lambda path: SimpleNamespace(nq=_NQ)),
  )
  monkeypatch.setattr(
    mujoco, "MjData", lambda model: SimpleNamespace(qpos=np.zeros(_NQ))
  )
  monkeypatch.setattr(mujoco, "mj_forward", lambda model, data: None)

  def build(*args: Any) -> Any:
    if fail_setup:
      raise RuntimeError("scene setup failed")

    def tick(now: float) -> None:
      raise KeyboardInterrupt

    return tick

  monkeypatch.setattr(inspect_sequence, "_build_static_scene", build)
  if fail_setup:
    with pytest.raises(RuntimeError, match="scene setup failed"):
      _ORIGINAL_OPEN_VIEWER(
        display, model_path=tmp_path / "model.mjb", host="127.0.0.1", port=8080
      )
  else:
    _ORIGINAL_OPEN_VIEWER(
      display, model_path=tmp_path / "model.mjb", host="127.0.0.1", port=8080
    )
  assert stops == [True]
