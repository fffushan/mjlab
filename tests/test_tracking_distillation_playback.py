"""The ``distill play`` surface, its playback facade, and checkpoint swaps.

These tests use the generated tiny cohort plus real train-shaped checkpoints.
Only the simulator/adapter boundary and the GUI viewers are mocked, so the
model-only load, deterministic mean-latent policy, checkpoint discovery, and
adapter cleanup on early errors/viewer exit are the real objects.
"""

from __future__ import annotations

import inspect
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch
from tracking_distillation_fixtures import DEFAULT_TEACHER_IDS, build_tiny_cohort

from mjlab.scripts import distill
from mjlab.scripts.distill import (
  _control_metadata,
  _playback_checkpoint_manager,
  main,
)
from mjlab.tasks.tracking.distillation.adapter import (
  DistillationEnvironmentAdapter,
  DistillationSnapshot,
  DistillationStep,
)
from mjlab.tasks.tracking.distillation.checkpoint import (
  CheckpointValidationError,
  save_checkpoint,
)
from mjlab.tasks.tracking.distillation.config import (
  DistillationError,
  load_manifest,
  resolve_cohort,
)
from mjlab.tasks.tracking.distillation.environment import RuntimeSeedProvenance
from mjlab.tasks.tracking.distillation.model import ConditionalVAE
from mjlab.tasks.tracking.distillation.observations import (
  ObservationSnapshot,
  PackedObservationBatch,
  pack_observations,
)
from mjlab.tasks.tracking.distillation.playback import (
  DistillationPlayEnvironment,
  DistillationPlayPolicy,
  discover_distillation_checkpoints,
)
from mjlab.tasks.tracking.distillation.storage import (
  LabeledReplayBatch,
  LabeledReplayBuffer,
)
from mjlab.tasks.tracking.distillation.trainer import VaeDistillationTrainer
from mjlab.tasks.tracking.distillation.training_config import TrainingConfig
from mjlab.tasks.tracking.distillation.vae_config import (
  CONDITIONING_DIMS,
  DEFAULT_SCHEMA,
  ModelSettings,
  VaeSchema,
  make_schema,
)
from mjlab.viewer.base import EnvProtocol, ViewerAction
from mjlab.viewer.viser.viewer import CheckpointManager, ViserPlayViewer

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

TEACHER_ID = DEFAULT_TEACHER_IDS[0]


@pytest.fixture(autouse=True)
def _require_validation_dependencies():
  pytest.importorskip("onnx")
  pytest.importorskip("onnxruntime")


def invoke(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> str | int | None:
  monkeypatch.setattr(sys, "argv", argv)
  with pytest.raises(SystemExit) as exit_info:
    main()
  return exit_info.value.code


def _resolved_cohort(tmp_path: Path):
  return resolve_cohort(load_manifest(build_tiny_cohort(tmp_path).manifest, tmp_path))


def _batch(
  size: int = 8, seed: int = 2, schema: VaeSchema = DEFAULT_SCHEMA
) -> LabeledReplayBatch:
  generator = torch.Generator().manual_seed(seed)
  ids = torch.arange(size, dtype=torch.int64)
  return LabeledReplayBatch(
    PackedObservationBatch(
      torch.randn(size, 68, generator=generator),
      torch.randn(size, CONDITIONING_DIMS[schema.mode.value], generator=generator),
      schema,
    ),
    torch.randn(size, 31, generator=generator),
    ids,
    ids,
    ids,
    ids,
    ids,
  )


def _snapshot(
  step: int, batch: int = 1, schema: VaeSchema = DEFAULT_SCHEMA
) -> DistillationSnapshot:
  values = torch.full((batch, 31), float(step))
  features = ObservationSnapshot(
    reference_q=values,
    reference_dq=values + 1,
    anchor_orientation_error=torch.zeros(batch, 6),
    projected_gravity=torch.zeros(batch, 3),
    gyro=torch.zeros(batch, 3),
    relative_joint_q=values + 2,
    joint_dq=values + 3,
    previous_action=values - 1,
  )
  return DistillationSnapshot(
    teacher_observation=torch.zeros(batch, 164),
    features=features,
    packed=pack_observations(features, schema),
    teacher_id=TEACHER_ID,
    teacher_code=0,
    motion_id=torch.zeros(batch, dtype=torch.int64),
    reference_frame=torch.full((batch,), step, dtype=torch.int64),
    segment_id=torch.zeros(batch, dtype=torch.int64),
    generation_id=torch.zeros(batch, dtype=torch.int64),
  )


class _FakeAdapter:
  """Stand-in for the trusted native adapter; constructs no simulated scene."""

  def __init__(self, seed: int | None, num_envs: int = 1) -> None:
    self.schema = DEFAULT_SCHEMA
    self.num_envs = num_envs
    self.closed = False
    self.reset_calls = 0
    self.step_calls = 0
    self.reset_observed_sampling_modes: list[str] = []
    self.last_step: DistillationStep | None = None
    self.motion = SimpleNamespace(cfg=SimpleNamespace(sampling_mode="adaptive"))
    self.env = SimpleNamespace(
      num_envs=num_envs,
      device="cpu",
      cfg=SimpleNamespace(viewer=SimpleNamespace()),
      command_manager=SimpleNamespace(get_term=lambda name: self.motion),
    )
    self.audit = SimpleNamespace(
      seed_provenance=RuntimeSeedProvenance(
        requested_seed=seed,
        effective_seed=seed,
        applied_before_construction=seed is not None,
      )
    )
    self._snapshot = _snapshot(0, num_envs)

  def snapshot(self) -> DistillationSnapshot:
    return self._snapshot

  def reset(self, seed: int | None = None) -> DistillationSnapshot:
    del seed
    self.reset_calls += 1
    self.reset_observed_sampling_modes.append(self.motion.cfg.sampling_mode)
    # A distinguishable reset snapshot (frame -1) so a test can prove the first
    # policy action followed the reset rather than a pre-reset snapshot.
    self._snapshot = _snapshot(-1, self.num_envs, self.schema)
    return self._snapshot

  def step(self, action: torch.Tensor) -> DistillationStep:
    self.step_calls += 1
    self.last_step = DistillationStep(
      self._snapshot,
      torch.zeros(self.num_envs),
      torch.zeros(self.num_envs, dtype=torch.bool),
      torch.zeros(self.num_envs, dtype=torch.bool),
      {},
    )
    return self.last_step

  def close(self) -> None:
    self.closed = True


def _install_fake_adapter(
  monkeypatch: pytest.MonkeyPatch, calls: list[dict], adapters: list[_FakeAdapter]
) -> None:
  """Mock only the simulator boundary; honor the checkpoint-supplied schema."""

  def factory(cohort, teacher_id: str = TEACHER_ID, **kwargs):
    calls.append({"teacher_id": teacher_id, **kwargs})
    adapter = _FakeAdapter(kwargs.get("seed"), kwargs.get("num_envs", 1))
    schema = kwargs.get("schema")
    if schema is not None:
      adapter.schema = schema
      adapter._snapshot = replace(
        adapter._snapshot,
        packed=pack_observations(adapter._snapshot.features, schema),
      )
    adapters.append(adapter)
    return adapter

  monkeypatch.setattr("mjlab.scripts.distill.make_distillation_adapter", factory)


class _FakePlayViewer:
  """Runs one obs -> policy -> step cycle and records the action."""

  instances: list[_FakePlayViewer] = []

  def __init__(self, env, policy, frame_rate=60.0, checkpoint_manager=None) -> None:
    self.env = env
    self.policy = policy
    self.checkpoint_manager = checkpoint_manager
    self.action: torch.Tensor | None = None
    self.snapshot: DistillationSnapshot | None = None
    _FakePlayViewer.instances.append(self)

  def run(self) -> None:
    snapshot = self.env.get_observations()
    self.snapshot = snapshot
    self.action = self.policy(snapshot)
    self.env.step(self.action)


class _RaisingViewer:
  def __init__(self, *args, **kwargs) -> None:
    del args, kwargs

  def run(self) -> None:
    raise RuntimeError("viewer failed to start")


def _install_fake_viewers(monkeypatch: pytest.MonkeyPatch, viewer: Any = None) -> None:
  _FakePlayViewer.instances = []
  target = _FakePlayViewer if viewer is None else viewer
  monkeypatch.setattr("mjlab.viewer.ViserPlayViewer", target)
  monkeypatch.setattr("mjlab.viewer.NativeMujocoViewer", target)


def _write_train_shaped_checkpoint(
  path: Path,
  resolved,
  *,
  hidden_dims: tuple[int, ...] = (12, 6),
  contract: dict | None = None,
  schema: VaeSchema = DEFAULT_SCHEMA,
) -> ConditionalVAE:
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.manual_seed(5)
  model = ConditionalVAE(schema, ModelSettings(hidden_dims=hidden_dims))
  replay = LabeledReplayBuffer(24, schema)
  replay.insert(_batch(schema=schema))
  trainer = VaeDistillationTrainer(
    model,
    replay,
    TrainingConfig(
      accumulation_steps=3, minibatch_size=4, learning_rate=1e-3, beta=0.3
    ),
    seed=9,
  )
  save_checkpoint(
    path,
    trainer,
    replay,
    counters={"iteration": 5},
    schedule={"max_iterations": 5},
    resolved_config={"command": "train"},
    teacher_hashes=resolved.teacher(TEACHER_ID).hashes,
    control_contract=contract or _control_metadata(resolved, TEACHER_ID),
  )
  return model


def _play_argv(tmp_path: Path, checkpoint: Path, *extra: str) -> list[str]:
  return [
    "distill",
    "play",
    "--manifest",
    str(tmp_path / "configs" / "tiny_teachers.yaml"),
    "--repo-root",
    str(tmp_path),
    "--teacher-id",
    TEACHER_ID,
    "--checkpoint",
    str(checkpoint),
    *extra,
  ]


def _explode(*args, **kwargs):
  raise AssertionError("play must not construct training or evaluation machinery")


def test_play_help_lists_checkpoint_viewer_and_shows_environment_defaults(
  monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  code = invoke(monkeypatch, ["distill", "play", "--help"])
  captured = capsys.readouterr()
  assert code == 0
  for flag in ("--checkpoint", "--viewer", "--sampling-mode", "--num-envs"):
    assert flag in captured.out
  assert "viser" in captured.out
  assert "--minibatch-size" not in captured.out
  # The default is browser Viser; native is an explicit choice.
  assert inspect.signature(distill._play).parameters["viewer"].default == "viser"


def test_play_requires_checkpoint_before_building_the_environment(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  calls: list[dict] = []
  adapters: list[_FakeAdapter] = []
  _install_fake_adapter(monkeypatch, calls, adapters)

  code = invoke(
    monkeypatch,
    [
      "distill",
      "play",
      "--manifest",
      str(tmp_path / "does-not-exist.yaml"),
      "--repo-root",
      str(tmp_path),
    ],
  )

  captured = capsys.readouterr()
  assert code == 1
  assert "requires --checkpoint" in captured.err
  # The early rejection happens before the manifest or adapter is touched.
  assert calls == []
  assert adapters == []


def test_play_reuses_evaluate_environment_with_mean_latent_policy(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  """Play builds the audited env, loads model-only, and decodes the mean latent.

  The train/evaluation machinery is poisoned so any accidental construction
  fails the run.
  """
  resolved = _resolved_cohort(tmp_path)
  checkpoint = tmp_path / "run" / "checkpoint-final.pt"
  model = _write_train_shaped_checkpoint(checkpoint, resolved)
  calls: list[dict] = []
  adapters: list[_FakeAdapter] = []
  _install_fake_adapter(monkeypatch, calls, adapters)
  for name in ("_build_runner", "evaluate_distillation", "DAggerCollector"):
    monkeypatch.setattr(f"mjlab.scripts.distill.{name}", _explode)
  _install_fake_viewers(monkeypatch)

  code = invoke(
    monkeypatch,
    _play_argv(
      tmp_path,
      checkpoint,
      "--num-envs",
      "1",
      "--viewer",
      "viser",
      "--sampling-mode",
      "uniform",
      "--seed",
      "11",
    ),
  )

  captured = capsys.readouterr()
  assert code == 0, captured.err
  assert captured.out == ""  # play writes no JSON report; only the viewer runs
  assert calls[0]["seed"] == 11
  assert calls[0]["teacher_id"] == TEACHER_ID
  assert calls[0]["num_envs"] == 1
  # The packing schema comes from the checkpoint, not a default assumption.
  assert (
    calls[0]["schema"].compatibility_metadata()
    == DEFAULT_SCHEMA.compatibility_metadata()
  )

  viewer = _FakePlayViewer.instances[-1]
  assert isinstance(viewer.env, DistillationPlayEnvironment)
  assert isinstance(viewer.policy, DistillationPlayPolicy)
  assert viewer.checkpoint_manager is not None
  # The audited env object backs the facade for the shared debug GUI.
  assert viewer.env.unwrapped is adapters[0].env
  # The playback-only sampling override lands on the private command copy, and
  # it is applied before the single audited reset.
  assert adapters[0].motion.cfg.sampling_mode == "uniform"
  assert adapters[0].reset_calls == 1
  assert adapters[0].reset_observed_sampling_modes == ["uniform"]
  # The first policy action read the post-reset snapshot, not a pre-reset one.
  assert viewer.snapshot is not None
  assert viewer.snapshot.reference_frame.tolist() == [-1]

  snapshot = adapters[0].snapshot()
  expected = model.mean_inference(
    snapshot.packed.reference, snapshot.packed.conditioning
  )
  torch.testing.assert_close(viewer.action, expected)

  # The checkpoint's saved settings drive the loaded model, not the default.
  assert viewer.policy.model.settings == ModelSettings(hidden_dims=(12, 6))
  assert not viewer.policy.model.training

  # Playback never mutates the normalizers.
  normalizer_before = {
    name: value.clone()
    for name, value in viewer.policy.model.reference_normalizer.state_dict().items()
  }
  viewer.policy(snapshot)
  for name, value in viewer.policy.model.reference_normalizer.state_dict().items():
    torch.testing.assert_close(value, normalizer_before[name])


def test_play_defaults_to_one_env_and_start_sampling(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  resolved = _resolved_cohort(tmp_path)
  checkpoint = tmp_path / "run" / "checkpoint-final.pt"
  _write_train_shaped_checkpoint(checkpoint, resolved)
  calls: list[dict] = []
  adapters: list[_FakeAdapter] = []
  _install_fake_adapter(monkeypatch, calls, adapters)
  _install_fake_viewers(monkeypatch)

  code = invoke(monkeypatch, _play_argv(tmp_path, checkpoint, "--seed", "3"))

  captured = capsys.readouterr()
  assert code == 0, captured.err
  assert calls[0]["num_envs"] == 1
  assert adapters[0].motion.cfg.sampling_mode == "start"
  assert adapters[0].reset_observed_sampling_modes == ["start"]


def test_play_resets_once_after_the_sampling_override_before_the_viewer(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """The constructor does not reset, so play must do it exactly once.

  The reset is audited (seeded) and happens after the sampling override and
  before the viewer runs, so the first policy action reads reset state.
  """
  resolved = _resolved_cohort(tmp_path)
  checkpoint = tmp_path / "run" / "checkpoint-final.pt"
  _write_train_shaped_checkpoint(checkpoint, resolved)
  calls: list[dict] = []
  adapters: list[_FakeAdapter] = []
  _install_fake_adapter(monkeypatch, calls, adapters)
  observed: list[tuple[int, list[str]]] = []

  class _Viewer(_FakePlayViewer):
    def run(self) -> None:
      adapter = self.env.adapter
      observed.append(
        (adapter.reset_calls, list(adapter.reset_observed_sampling_modes))
      )
      super().run()

  _install_fake_viewers(monkeypatch, _Viewer)

  code = invoke(
    monkeypatch, _play_argv(tmp_path, checkpoint, "--sampling-mode", "uniform")
  )

  assert code == 0
  assert observed == [(1, ["uniform"])]
  assert _FakePlayViewer.instances[-1].env.adapter.reset_calls == 1


@pytest.mark.parametrize("mode", ["gravity", "anchor", "gravity_anchor"])
def test_play_builds_packing_from_the_saved_schema(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  mode: str,
) -> None:
  """A saved non-default schema is honored instead of a default gravity one."""
  resolved = _resolved_cohort(tmp_path)
  schema = make_schema(mode)
  checkpoint = tmp_path / "run" / "checkpoint-final.pt"
  _write_train_shaped_checkpoint(checkpoint, resolved, schema=schema)
  calls: list[dict] = []
  adapters: list[_FakeAdapter] = []
  _install_fake_adapter(monkeypatch, calls, adapters)
  _install_fake_viewers(monkeypatch)

  code = invoke(monkeypatch, _play_argv(tmp_path, checkpoint))

  captured = capsys.readouterr()
  assert code == 0, captured.err
  assert calls[0]["schema"].mode.value == mode
  viewer = _FakePlayViewer.instances[-1]
  assert viewer.env.adapter.schema.mode.value == mode
  assert viewer.snapshot is not None
  assert viewer.snapshot.packed.conditioning.shape[1] == CONDITIONING_DIMS[mode]
  assert viewer.action is not None and viewer.action.shape == (1, 31)


def test_playback_adapter_rejects_incompatible_joint_order(tmp_path: Path) -> None:
  """The live action joints must match the schema the adapter packs with."""
  resolved = _resolved_cohort(tmp_path)
  env = SimpleNamespace(num_envs=1, device="cpu")
  with pytest.raises(DistillationError, match="joint_order"):
    DistillationEnvironmentAdapter(
      env,
      resolved,
      cast(Any, SimpleNamespace()),
      TEACHER_ID,
      schema=DEFAULT_SCHEMA,
      audit=cast(Any, SimpleNamespace()),
    )


def test_play_native_viewer_is_an_explicit_choice(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  resolved = _resolved_cohort(tmp_path)
  checkpoint = tmp_path / "run" / "checkpoint-final.pt"
  _write_train_shaped_checkpoint(checkpoint, resolved)
  calls: list[dict] = []
  adapters: list[_FakeAdapter] = []
  _install_fake_adapter(monkeypatch, calls, adapters)
  chosen: list[str] = []

  class _RecordingNative(_FakePlayViewer):
    def __init__(self, *args, **kwargs) -> None:
      chosen.append("native")
      super().__init__(*args, **kwargs)

  class _RecordingViser(_FakePlayViewer):
    def __init__(self, *args, **kwargs) -> None:
      chosen.append("viser")
      super().__init__(*args, **kwargs)

  monkeypatch.setattr("mjlab.viewer.NativeMujocoViewer", _RecordingNative)
  monkeypatch.setattr("mjlab.viewer.ViserPlayViewer", _RecordingViser)

  code = invoke(monkeypatch, _play_argv(tmp_path, checkpoint, "--viewer", "native"))
  captured = capsys.readouterr()
  assert code == 0, captured.err
  assert chosen == ["native"]
  assert adapters[0].closed


def test_play_rejects_incompatible_checkpoint_before_building_the_environment(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  """The model-only load runs first, so a bad checkpoint builds no simulator."""
  resolved = _resolved_cohort(tmp_path)
  checkpoint = tmp_path / "run" / "checkpoint-final.pt"
  _write_train_shaped_checkpoint(
    checkpoint,
    resolved,
    contract={**_control_metadata(resolved, TEACHER_ID), "control_hz": 30.0},
  )
  calls: list[dict] = []
  adapters: list[_FakeAdapter] = []
  _install_fake_adapter(monkeypatch, calls, adapters)
  _install_fake_viewers(monkeypatch)

  code = invoke(monkeypatch, _play_argv(tmp_path, checkpoint))

  captured = capsys.readouterr()
  assert code == 1
  assert "[FAIL]" in captured.err
  assert "control contract" in captured.err
  assert calls == []
  assert adapters == []
  assert _FakePlayViewer.instances == []


def test_play_rejects_missing_checkpoint_before_building_the_environment(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  calls: list[dict] = []
  adapters: list[_FakeAdapter] = []
  _install_fake_adapter(monkeypatch, calls, adapters)
  _install_fake_viewers(monkeypatch)

  code = invoke(monkeypatch, _play_argv(tmp_path, tmp_path / "run" / "absent.pt"))

  captured = capsys.readouterr()
  assert code == 1
  assert "[FAIL]" in captured.err
  assert calls == [] and adapters == []


def test_play_validates_num_envs_and_frame_rate_before_any_work(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  calls: list[dict] = []
  adapters: list[_FakeAdapter] = []
  _install_fake_adapter(monkeypatch, calls, adapters)

  for extra, expected in (
    (["--num-envs", "0"], "num_envs"),
    (["--num-envs", "-2"], "num_envs"),
    (["--frame-rate", "0"], "frame_rate"),
    (["--frame-rate", "nan"], "frame_rate"),
  ):
    code = invoke(
      monkeypatch,
      _play_argv(tmp_path, tmp_path / "run" / "absent.pt", *extra),
    )
    captured = capsys.readouterr()
    assert code == 1, (extra, captured.err)
    assert expected in captured.err
  assert calls == [] and adapters == []


def test_play_closes_the_adapter_when_the_viewer_fails(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  resolved = _resolved_cohort(tmp_path)
  checkpoint = tmp_path / "run" / "checkpoint-final.pt"
  _write_train_shaped_checkpoint(checkpoint, resolved)
  calls: list[dict] = []
  adapters: list[_FakeAdapter] = []
  _install_fake_adapter(monkeypatch, calls, adapters)
  _install_fake_viewers(monkeypatch, _RaisingViewer)

  code = invoke(monkeypatch, _play_argv(tmp_path, checkpoint, "--viewer", "native"))

  captured = capsys.readouterr()
  assert code == 1
  assert "[FAIL]" in captured.err
  assert adapters[0].closed


def test_play_accepts_relocated_identical_artifacts(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  """A checkpoint from one checkout plays after the artifacts move wholesale.

  The manifest resolves the same byte-identical motion at a new path, so only
  ``control_contract['motion']`` changes; matching teacher hashes prove the
  content is identical and model-only playback is allowed.
  """
  root_a = tmp_path / "a"
  root_a.mkdir()
  build_tiny_cohort(root_a)
  resolved_a = resolve_cohort(
    load_manifest(root_a / "configs" / "tiny_teachers.yaml", root_a)
  )
  checkpoint = tmp_path / "run" / "checkpoint-final.pt"
  _write_train_shaped_checkpoint(checkpoint, resolved_a)

  root_b = tmp_path / "b"
  shutil.copytree(root_a, root_b)
  resolved_b = resolve_cohort(
    load_manifest(root_b / "configs" / "tiny_teachers.yaml", root_b)
  )
  # The content — and therefore every digest — is identical; only the path moved.
  assert resolved_b.teacher(TEACHER_ID).hashes == resolved_a.teacher(TEACHER_ID).hashes
  assert (
    resolved_b.teacher(TEACHER_ID).entry.motion
    != resolved_a.teacher(TEACHER_ID).entry.motion
  )

  calls: list[dict] = []
  adapters: list[_FakeAdapter] = []
  _install_fake_adapter(monkeypatch, calls, adapters)
  _install_fake_viewers(monkeypatch)

  code = invoke(
    monkeypatch,
    [
      "distill",
      "play",
      "--manifest",
      str(root_b / "configs" / "tiny_teachers.yaml"),
      "--repo-root",
      str(root_b),
      "--teacher-id",
      TEACHER_ID,
      "--checkpoint",
      str(checkpoint),
    ],
  )

  captured = capsys.readouterr()
  assert code == 0, captured.err
  assert len(_FakePlayViewer.instances) == 1


def test_play_environment_forwards_snapshot_step_reset_and_close() -> None:
  adapter = _FakeAdapter(seed=0, num_envs=2)
  env = DistillationPlayEnvironment(cast(Any, adapter))

  assert env.num_envs == 2
  assert env.device == "cpu"
  assert env.cfg is adapter.env.cfg
  assert env.unwrapped is adapter.env

  snapshot = env.get_observations()
  assert snapshot is adapter.snapshot()

  # ``step``/``reset`` stay on the audited adapter, whose step consumes the
  # segment boundary events instead of letting them accumulate.
  step = env.step(torch.zeros(2, 31))
  assert adapter.step_calls == 1
  assert step is adapter.last_step
  env.reset()
  assert adapter.reset_calls == 1

  env.close()
  assert adapter.closed


def test_discover_distillation_checkpoints_sorts_iterations_then_final(
  tmp_path: Path,
) -> None:
  for name in (
    "checkpoint-final.pt",
    "checkpoint-iter-000010.pt",
    "checkpoint-iter-000002.pt",
    "checkpoint-iter-bad.pt",
    "model_4000.pt",
    "notes.txt",
  ):
    (tmp_path / name).write_text("x")
  (tmp_path / "checkpoint-iter-000003.pt").mkdir()

  names = [path.name for path in discover_distillation_checkpoints(tmp_path)]

  assert names == [
    "checkpoint-iter-000002.pt",
    "checkpoint-iter-000010.pt",
    "checkpoint-final.pt",
  ]
  assert discover_distillation_checkpoints(tmp_path / "missing") == []


def test_playback_checkpoint_manager_discovers_and_validates_swaps(
  tmp_path: Path,
) -> None:
  resolved = _resolved_cohort(tmp_path)
  directory = tmp_path / "run"
  good = directory / "checkpoint-final.pt"
  _write_train_shaped_checkpoint(good, resolved)
  incompatible = directory / "checkpoint-iter-000003.pt"
  _write_train_shaped_checkpoint(
    incompatible,
    resolved,
    contract={**_control_metadata(resolved, TEACHER_ID), "control_hz": 30.0},
  )
  adapter = SimpleNamespace(schema=DEFAULT_SCHEMA)

  manager = _playback_checkpoint_manager(good, resolved, TEACHER_ID, adapter, "cpu")

  assert manager.current_name == "checkpoint-final.pt"
  assert [name for name, _ in manager.fetch_available()] == [
    "checkpoint-iter-000003.pt",
    "checkpoint-final.pt",
  ]
  policy = manager.load_checkpoint("checkpoint-final.pt")
  assert isinstance(policy, DistillationPlayPolicy)
  # A swap goes through the same model-only validation; an incompatible
  # artifact is refused instead of being installed.
  with pytest.raises(CheckpointValidationError):
    manager.load_checkpoint("checkpoint-iter-000003.pt")


def test_viewer_rejected_swap_keeps_the_running_policy() -> None:
  class _Env:
    num_envs = 1
    device = "cpu"
    step_dt = 0.02
    cfg = SimpleNamespace(viewer=SimpleNamespace())

    def reset(self) -> None:
      pass

  env = _Env()
  env.unwrapped = env  # type: ignore[attr-defined]
  viewer = ViserPlayViewer(
    cast(EnvProtocol, env), MagicMock(), viser_server=MagicMock()
  )
  viewer._ckpt_user_event = Event()
  viewer._ckpt_dropdown = MagicMock(
    options=["good.pt  (1s ago)", "bad.pt  (1s ago)"],
    value="bad.pt  (1s ago)",
  )
  sentinel = MagicMock(name="running-policy")
  viewer.policy = sentinel

  def _rejecting(name: str):
    raise CheckpointValidationError(f"swap refused: {name}")

  rejecting = CheckpointManager(
    current_name="good.pt",
    fetch_available=lambda: [("good.pt", "1s ago"), ("bad.pt", "1s ago")],
    load_checkpoint=_rejecting,
  )
  viewer._ckpt_mgr = rejecting

  # The rejected load must not raise, install a policy, or leave the dropdown
  # pointing at the artifact that failed.
  assert viewer._handle_custom_action(ViewerAction.FETCH_CHECKPOINT, "selected")
  assert viewer.policy is sentinel
  assert viewer._ckpt_dropdown.value == "good.pt  (1s ago)"

  # A valid swap installs the returned policy and resets the environment.
  replacement = MagicMock(name="replacement-policy")

  def _accepting(name: str):
    assert name == "bad.pt"
    return replacement

  accepting = CheckpointManager(
    current_name="good.pt",
    fetch_available=lambda: [("good.pt", "1s ago"), ("bad.pt", "1s ago")],
    load_checkpoint=_accepting,
  )
  viewer._ckpt_mgr = accepting
  viewer._ckpt_dropdown.value = "bad.pt  (1s ago)"
  reset_mock = MagicMock()
  viewer.reset_environment = reset_mock  # type: ignore[method-assign]

  assert viewer._handle_custom_action(ViewerAction.FETCH_CHECKPOINT, "selected")
  assert viewer.policy is replacement
  assert accepting.current_name == "bad.pt"
  reset_mock.assert_called_once()
