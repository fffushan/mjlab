"""CPU-only startup tests for the preview-aware (version 2) VAE reference.

The preview layout appends one 62-wide ``motion_lookahead`` sample to the
frozen 68-wide gravity reference.  These tests pin the loaded-schema width
agreement, the single and batched startup reference order, and the
reference-free handover. They use fake environments,
fake commands and fake VAE surfaces only; no MuJoCo physics is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from tracking_diffusion_fixtures import make_state

from mjlab.tasks.tracking.diffusion import replay_integration as replay
from mjlab.tasks.tracking.diffusion.closed_loop import (
  EvaluationBudget,
  run_matched_pair,
)
from mjlab.tasks.tracking.diffusion.closed_loop_runtime_x2 import (
  X2ClosedLoopRuntime,
)
from mjlab.tasks.tracking.diffusion.closed_loop_types import (
  D3Observation,
  D3Transition,
)
from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT
from mjlab.tasks.tracking.diffusion.handover import ContinuationStep
from mjlab.tasks.tracking.diffusion.parallel_closed_loop_x2 import (
  X2ParallelClosedLoopRuntime,
)
from mjlab.tasks.tracking.distillation.vae_config import (
  PREVIEW_DIM,
  REFERENCE_DIM,
  REFERENCE_DIM_WITH_PREVIEW,
  REFERENCE_PREVIEW_FIELD,
)

_JOINTS = tuple(f"joint_{index:02d}" for index in range(31))
_BODIES = DEFAULT_CONTRACT.body_names

# Distinguishable actor-cache channels so a reordered reference cannot pass.
_COMMAND_VALUES = np.arange(2 * len(_JOINTS), dtype=np.float64)
_ANCHOR_VALUES = np.arange(6, dtype=np.float64) + 3000.0


# ---------------------------------------------------------------------------
# Manifest loader gate
# ---------------------------------------------------------------------------


def _loader_cohort(lookahead_s: float) -> Any:
  return SimpleNamespace(lookahead_s=lookahead_s)


def _loader_inference(reference_dim: int, *, conditioning_dim: int = 99) -> Any:
  class _Model(torch.nn.Module):
    def __init__(self) -> None:
      super().__init__()
      self.schema = SimpleNamespace(
        mode="gravity",
        reference_dim=reference_dim,
        conditioning_dim=conditioning_dim,
        action_dim=31,
      )
      self.settings = SimpleNamespace(latent_dim=32)

    def decode(self, latent: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
      return torch.zeros((latent.shape[0], 31), dtype=latent.dtype)

  return SimpleNamespace(
    model=_Model(),
    relocated_artifact_roles=(),
    cohort=SimpleNamespace(common=SimpleNamespace(joint_names=_JOINTS)),
  )


def _load_vae(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  *,
  lookahead_s: float,
  reference_dim: int,
  conditioning_dim: int = 99,
) -> Any:
  import mjlab.tasks.tracking.distillation.checkpoint as checkpoint_module
  import mjlab.tasks.tracking.distillation.config as config_module

  inference = _loader_inference(reference_dim, conditioning_dim=conditioning_dim)
  monkeypatch.setattr(
    config_module, "load_manifest", lambda *_args, **_kwargs: object()
  )
  monkeypatch.setattr(
    config_module,
    "resolve_cohort",
    lambda _manifest: _loader_cohort(lookahead_s),
  )
  monkeypatch.setattr(
    checkpoint_module,
    "load_cohort_member_inference",
    lambda *_args, **_kwargs: inference,
  )
  manifest = tmp_path / "manifest.yaml"
  manifest.write_text("preview startup test manifest\n")
  return replay.load_vae_decoder(
    tmp_path / "vae.pt",
    manifest_path=manifest,
    repo_root=tmp_path,
    member="single_000",
    contract=DEFAULT_CONTRACT,
  )


@pytest.mark.parametrize(
  ("lookahead_s", "reference_dim"),
  [(0.0, REFERENCE_DIM), (0.2, REFERENCE_DIM_WITH_PREVIEW)],
)
def test_loader_accepts_the_cohort_declared_layout(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  lookahead_s: float,
  reference_dim: int,
) -> None:
  loaded = _load_vae(
    monkeypatch,
    tmp_path,
    lookahead_s=lookahead_s,
    reference_dim=reference_dim,
  )
  assert loaded.reference_dim == reference_dim
  assert loaded.conditioning_dim == 99
  assert loaded.latent_dim == 32
  assert loaded.action_dim == 31


@pytest.mark.parametrize(
  ("lookahead_s", "reference_dim"),
  [(0.2, REFERENCE_DIM), (0.0, REFERENCE_DIM_WITH_PREVIEW)],
)
def test_loader_refuses_a_layout_that_disagrees_with_the_cohort(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  lookahead_s: float,
  reference_dim: int,
) -> None:
  with pytest.raises(replay.ReplayError, match="reference"):
    _load_vae(
      monkeypatch,
      tmp_path,
      lookahead_s=lookahead_s,
      reference_dim=reference_dim,
    )


# ---------------------------------------------------------------------------
# Single-row startup reference
# ---------------------------------------------------------------------------


@dataclass
class _Term:
  func: Any
  params: dict[str, Any]
  time_out: bool = False


class _PreviewObservations:
  terms = (
    SimpleNamespace(name="command", width=2 * len(_JOINTS)),
    SimpleNamespace(name=REFERENCE_PREVIEW_FIELD, width=PREVIEW_DIM),
    SimpleNamespace(name="motion_anchor_ori_b", width=6),
    SimpleNamespace(name="base_ang_vel", width=3),
    SimpleNamespace(name="joint_pos", width=31),
    SimpleNamespace(name="joint_vel", width=31),
    SimpleNamespace(name="actions", width=31),
  )
  names = tuple(term.name for term in terms)
  widths = tuple(term.width for term in terms)
  total_dim = sum(widths)


class _PreviewCohort:
  observations: Any = _PreviewObservations()
  actions = SimpleNamespace(joint_names=_JOINTS, dim=31)
  control = SimpleNamespace(control_period_s=0.02)
  actor = SimpleNamespace(obs_dim=_PreviewObservations.total_dim)
  body_names = _BODIES
  manifest = SimpleNamespace(base_task=None)
  lookahead_s = 0.2


class _PreviewCommand:
  def __init__(self) -> None:
    self._reference_free = False
    self._time_steps = torch.tensor([0], dtype=torch.long)
    self.generation_ids = torch.tensor([0], dtype=torch.long)
    self.motion = SimpleNamespace(time_step_total=32)
    self.cfg = SimpleNamespace(body_names=_BODIES)

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
    return SimpleNamespace(
      available=torch.tensor([False]),
      completed=torch.tensor([False]),
      interrupted=torch.tensor([False]),
      reasons=("unavailable",),
    )


class _PreviewData:
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


class _PreviewRobot:
  joint_names = _JOINTS
  body_names = _BODIES

  def __init__(self) -> None:
    self.data = _PreviewData()


class _PreviewObservationManager:
  def __init__(self, env: _PreviewEnv) -> None:
    self.env = env
    self._obs_buffer: dict[str, torch.Tensor] | None = None

  def compute(
    self, *, update_history: bool = False, env_ids: torch.Tensor | None = None
  ) -> dict[str, torch.Tensor]:
    del env_ids
    if update_history:
      self.env.refresh_cache()
    assert self._obs_buffer is not None
    return self._obs_buffer


class _PreviewScene:
  def __init__(self, robot: _PreviewRobot) -> None:
    self.robot = robot

  def __getitem__(self, name: str) -> _PreviewRobot:
    assert name == "robot"
    return self.robot

  def write_data_to_sim(self) -> None:
    pass


class _PreviewCommandManager:
  def __init__(self, command: _PreviewCommand) -> None:
    self.command = command

  def get_term(self, name: str) -> _PreviewCommand:
    assert name == "motion"
    return self.command


class _PreviewEnv:
  num_envs = 1
  device = "cpu"
  step_dt = 0.02

  def __init__(self, rows: int = 1) -> None:
    self.num_envs = rows
    self.robot = _PreviewRobot()
    for name, value in vars(self.robot.data).items():
      setattr(self.robot.data, name, value.repeat_interleave(rows, dim=0))
    self.command = _PreviewCommand()
    self.scene = _PreviewScene(self.robot)
    self.command_manager = _PreviewCommandManager(self.command)
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
    self.observation_manager = _PreviewObservationManager(self)
    self.obs_buf = {"actor": self._actor_cache()}
    self.observation_manager._obs_buffer = self.obs_buf
    self.reset_calls = 0

  def _frame(self) -> int:
    return int(self.command._time_steps[0])

  def _actor_cache(self) -> torch.Tensor:
    frame_offset = 1000.0 * self._frame() if not self.command._reference_free else 0.0
    actor = torch.zeros(
      (self.num_envs, _PreviewObservations.total_dim), dtype=torch.float32
    )
    actor[:, 0:62] = torch.as_tensor(_COMMAND_VALUES, dtype=torch.float32)
    actor[:, 62:124] = (
      torch.arange(PREVIEW_DIM, dtype=torch.float32) + 2000.0 + frame_offset
    )
    actor[:, 124:130] = torch.as_tensor(_ANCHOR_VALUES, dtype=torch.float32)
    actor[:, 130:133] = torch.tensor([[0.1, 0.2, 0.3]])
    actor[:, 133:164] = torch.arange(31, dtype=torch.float32) * 0.01
    actor[:, 164:195] = 0.2
    actor[:, 195:226] = 0.0
    return actor

  def refresh_cache(self) -> None:
    self.obs_buf = {"actor": self._actor_cache()}
    self.observation_manager._obs_buffer = self.obs_buf

  def reset(self, *, seed: int) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    del seed
    self.reset_calls += 1
    self.refresh_cache()
    return self.obs_buf, {}

  def step(self, action: torch.Tensor) -> tuple[Any, ...]:
    assert action.shape == (1, 31)
    if not self.command._reference_free:
      self.command._time_steps += 1
    self.refresh_cache()
    return (
      self.obs_buf,
      torch.zeros(1),
      torch.tensor([False]),
      torch.tensor([False]),
      {},
    )

  def close(self) -> None:
    pass


def _preview_runtime(
  runtime_cls: type[X2ClosedLoopRuntime] = X2ClosedLoopRuntime,
) -> tuple[X2ClosedLoopRuntime, _PreviewEnv]:
  env = _PreviewEnv()
  runtime = runtime_cls(env, _PreviewCohort(), "single_000")
  return runtime, env


def test_single_runtime_builds_the_preview_reference_in_schema_order() -> None:
  runtime, env = _preview_runtime()
  observation = runtime.reset(seed=0, start_frame=5)

  actor = env.obs_buf["actor"][0].numpy()
  expected = np.concatenate(
    (
      actor[0:62],
      actor[124:130],
      actor[62:124],
    )
  )
  assert observation.reference is not None
  assert observation.reference.shape == (REFERENCE_DIM_WITH_PREVIEW,)
  np.testing.assert_array_equal(observation.reference, expected)
  np.testing.assert_array_equal(observation.reference[:62], _COMMAND_VALUES)
  np.testing.assert_array_equal(observation.reference[62:68], _ANCHOR_VALUES)
  np.testing.assert_array_equal(
    observation.reference[68:], np.arange(PREVIEW_DIM) + 2000.0 + 5000.0
  )
  transition = runtime.step(np.zeros(31))
  assert transition.observation.reference is not None
  assert transition.observation.reference.shape == (REFERENCE_DIM_WITH_PREVIEW,)
  assert runtime.provenance["reference_layout"]["preview"] is True
  assert runtime.provenance["reference_layout"]["width"] == REFERENCE_DIM_WITH_PREVIEW


class _GuardedActor:
  """Fake observation source that refuses reference slices after handover."""

  _REFERENCE_TERMS = ("command", REFERENCE_PREVIEW_FIELD, "motion_anchor_ori_b")

  def __init__(self, array: np.ndarray, runtime: Any) -> None:
    self._array = array
    self._runtime = runtime

  def __getitem__(self, key: Any) -> np.ndarray:
    span = key[-1] if isinstance(key, tuple) else key
    if self._runtime._phase != "vae":
      for name in self._REFERENCE_TERMS:
        if span == self._runtime._slices[name]:
          raise AssertionError(f"reference term {name} read after handover")
    return self._array[key]


class _GuardedRuntime(X2ClosedLoopRuntime):
  def _cached_actor(self) -> Any:
    return _GuardedActor(super()._cached_actor(), self)


def test_reference_terms_are_not_read_after_handover() -> None:
  runtime, env = _preview_runtime(_GuardedRuntime)
  runtime.reset(seed=0, start_frame=3)
  handed = runtime.enter_reference_free()
  assert handed.reference is None
  assert handed.reference_frame is None
  transition = runtime.step(np.zeros(31))
  assert transition.observation.reference is None
  runtime.step(np.zeros(31))
  assert runtime._phase == "reference_free"
  env.command._time_steps[:] = env.command.motion.time_step_total - 1
  runtime.step(np.zeros(31))
  assert int(env.command._time_steps[0]) == env.command.motion.time_step_total - 1
  with pytest.raises(AssertionError, match="reference frame"):
    _ = env.command.time_steps


# ---------------------------------------------------------------------------
# Batched reference and observation
# ---------------------------------------------------------------------------


class _GuardedParallelRuntime(X2ParallelClosedLoopRuntime):
  def _cached_actor(self) -> Any:
    return _GuardedActor(super()._cached_actor(), self)


def test_parallel_runtime_constructs_preview_and_hands_over_without_reference() -> None:
  env = _PreviewEnv(rows=2)
  runtime = _GuardedParallelRuntime(env, _PreviewCohort(), "single_000")
  observation = runtime.reset(seed=0, start_frame=5)
  actor = env.obs_buf["actor"].numpy()
  expected = np.concatenate(
    (actor[:, :62], actor[:, 124:130], actor[:, 62:124]), axis=1
  )
  assert observation.reference is not None
  assert observation.reference.shape == (2, 130)
  np.testing.assert_array_equal(observation.reference, expected)
  handed = runtime.enter_reference_free()
  assert handed.reference is None
  assert handed.reference_frame is None
  assert handed.conditioning.shape == (2, 99)
  with pytest.raises(AssertionError, match="reference frame"):
    _ = env.command.time_steps


# ---------------------------------------------------------------------------
# Matched-pair runner
# ---------------------------------------------------------------------------


class _RunnerVae:
  def __init__(self, reference_dim: int) -> None:
    self.schema = SimpleNamespace(reference_dim=reference_dim)
    self.encode_calls = 0
    self.encode_widths: list[int] = []

  def encode(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    self.encode_calls += 1
    self.encode_widths.append(int(reference.shape[-1]))
    return torch.zeros((1, 32)), torch.zeros((1, 32))

  def decode(self, latent: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
    del latent, conditioning
    return torch.zeros((1, 31))


class _RunnerRuntime:
  period_seconds = 0.02
  clip_frames = 100
  joint_order = _JOINTS
  body_order = _BODIES
  provenance: dict[str, Any] = {"synthetic": True}

  def __init__(self, reference_width: int) -> None:
    self.reference_width = reference_width
    self.entered_reference_free = False
    self.closed = False
    self._index = 0
    self._tick = 0

  def _observation(self) -> D3Observation:
    return D3Observation(
      state=make_state(self._index),
      conditioning=np.full(99, 0.25),
      tick=self._tick,
      sim_time_s=self._tick * self.period_seconds,
      segment_id="segment",
      reference=(
        None if self.entered_reference_free else np.full(self.reference_width, 0.5)
      ),
      reference_frame=None if self.entered_reference_free else self._index,
    )

  def reset(self, *, seed: int, start_frame: int) -> D3Observation:
    del seed
    self._index = start_frame
    self._tick = 0
    self.entered_reference_free = False
    return self._observation()

  def initial_signature(self) -> np.ndarray:
    return np.arange(37, dtype=np.float64)

  def enter_reference_free(self) -> D3Observation:
    self.entered_reference_free = True
    return self._observation()

  def step(self, action: np.ndarray) -> D3Transition:
    del action
    self._index += 1
    self._tick += 1
    return D3Transition(self._observation())

  def close(self) -> None:
    self.closed = True


class _RunnerContinuation:
  def __init__(self) -> None:
    self.contract = DEFAULT_CONTRACT
    self.records: list[tuple[Any, np.ndarray]] = []

  def record_executed(self, state: Any, latent: np.ndarray) -> None:
    self.records.append((state, np.array(latent, copy=True)))

  def step(
    self, state: Any, conditioning: np.ndarray, *, tick: int
  ) -> ContinuationStep:
    del state, conditioning
    return ContinuationStep(
      tick=tick,
      action=np.zeros(31),
      executed_latent=np.full(32, 0.1),
      denoiser_calls=1,
      final_index=0,
      predicted_next_hybrid_state=None,
    )


def test_matched_pair_uses_the_loaded_schema_reference_width() -> None:
  model = _RunnerVae(REFERENCE_DIM_WITH_PREVIEW)
  report = run_matched_pair(
    lambda: _RunnerRuntime(REFERENCE_DIM_WITH_PREVIEW),
    model=model,
    continuation_factory=_RunnerContinuation,
    seed=0,
    start_frame=0,
    budget=EvaluationBudget(max_steps=1),
  )
  assert report.status == "complete"
  assert report.provenance["model_inputs"]["vae_reference_width"] == (
    REFERENCE_DIM_WITH_PREVIEW
  )
  assert report.provenance["model_inputs"]["vae_reference_preview"] is True
  # Baseline: eight bootstrap pairs plus one scored VAE step.  Diffusion:
  # eight bootstrap pairs only; the reference-free continuation never calls the
  # encoder.
  assert model.encode_calls == 17
  assert set(model.encode_widths) == {REFERENCE_DIM_WITH_PREVIEW}
