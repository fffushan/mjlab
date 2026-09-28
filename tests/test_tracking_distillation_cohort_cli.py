"""M4 cohort CLI integration: plural selection, member pins, cohort reports.

The simulator boundary is mocked (no scene is ever constructed), while the
model, balanced replay, trainer, runner, cohort identity, checkpoints, and
reports are the real objects.  The cohort is the real saved X2 manifest with its
two clips of *unequal* length (453 and 340 frames), so the clip-duration
weighting of the aggregate report is exercised with a real difference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mjlab.scripts import distill
from mjlab.scripts.distill import (
  _apply_reset_perturbations,
  _control_metadata,
  _reset_provenance_report,
  _trial_provenance_summary,
  main,
)
from mjlab.tasks.tracking.distillation.adapter import (
  DistillationSnapshot,
  DistillationStep,
  InitializationKind,
)
from mjlab.tasks.tracking.distillation.checkpoint import (
  CheckpointValidationError,
  save_checkpoint,
)
from mjlab.tasks.tracking.distillation.collector import (
  EvaluationResult,
  EvaluationSegment,
)
from mjlab.tasks.tracking.distillation.config import (
  CohortContract,
  DistillationError,
  load_manifest,
  resolve_cohort,
)
from mjlab.tasks.tracking.distillation.environment import RuntimeSeedProvenance
from mjlab.tasks.tracking.distillation.motion_library import (
  BodySelection,
  MotionClipSpec,
  MotionLibrary,
)
from mjlab.tasks.tracking.distillation.multi_motion import stratified_slot_allocation
from mjlab.tasks.tracking.distillation.observations import (
  ObservationSnapshot,
  pack_observations,
)
from mjlab.tasks.tracking.distillation.playback import DistillationPlayPolicy
from mjlab.tasks.tracking.distillation.storage import LabeledReplayBuffer
from mjlab.tasks.tracking.distillation.teachers import build_cohort_teacher_bank
from mjlab.tasks.tracking.distillation.trainer import VaeDistillationTrainer
from mjlab.tasks.tracking.distillation.training_config import TrainingConfig
from mjlab.tasks.tracking.distillation.vae_config import (
  DEFAULT_SCHEMA,
  DecoderMode,
  make_schema,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

MANIFEST = Path("configs/distillation/x2_tennis.yaml")
REPO_ROOT = Path(".")
TEACHER_IDS = ("tennis_000", "tennis_001")
TEACHER_ID = TEACHER_IDS[0]
OTHER_TEACHER_ID = TEACHER_IDS[1]

_TRAIN_ARGV = [
  "--num-envs",
  "2",
  "--bootstrap-steps",
  "1",
  "--collection-steps",
  "1",
  "--minibatch-size",
  "4",
  "--accumulation-steps",
  "3",
  "--replay-capacity",
  "24",
  "--seed",
  "7",
]


@pytest.fixture(autouse=True)
def _require_validation_dependencies():
  pytest.importorskip("onnx")


@pytest.fixture(autouse=True)
def _forbid_real_simulator(monkeypatch: pytest.MonkeyPatch) -> None:
  """Fail loudly if any test in this module constructs a real environment.

  Every command exercised here patches its adapter factory, so a real
  ``ManagerBasedRlEnv`` would mean a missing fake (an unapproved live
  simulation), not a legitimate test setup.
  """

  def _refuse(*args, **kwargs):
    raise AssertionError(
      "this test module must not construct a simulator; patch the adapter factory"
    )

  monkeypatch.setattr("mjlab.envs.ManagerBasedRlEnv", _refuse)


@pytest.fixture(scope="module")
def cohort() -> CohortContract:
  """The real saved two-teacher cohort, resolved without a simulator."""
  return resolve_cohort(load_manifest(MANIFEST, repo_root=REPO_ROOT))


@pytest.fixture(scope="module")
def schema(cohort: CohortContract):
  """The live VAE schema: the saved X2 joint order, not a synthetic default."""
  return make_schema(DecoderMode.GRAVITY, tuple(cohort.actions.joint_names))


@pytest.fixture(scope="module")
def motion_frames(cohort: CohortContract) -> dict[str, int]:
  return {teacher.id: int(teacher.reference.frames) for teacher in cohort.teachers}


@pytest.fixture(scope="module")
def motion_fps(cohort: CohortContract) -> float:
  return float(cohort.teachers[0].reference.fps)


def invoke(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> str | int | None:
  monkeypatch.setattr(sys, "argv", argv)
  with pytest.raises(SystemExit) as exit_info:
    main()
  return exit_info.value.code


def _teacher_ids_arg(teacher_ids: tuple[str, ...]) -> str:
  """The exact plural-selection spelling this repo's tyro config accepts.

  ``mjlab.TYRO_FLAGS`` sets ``UsePythonSyntaxForLiteralCollections``, so a
  collection flag takes a Python literal (``--teacher-ids ('a','b')``) instead
  of a space-separated list.  The literal always carries a trailing comma, so a
  single-element selection is a tuple and not a bare string.  These tests use
  the documented spelling rather than an invented one.
  """
  if not teacher_ids:
    return "()"
  return "(" + ",".join(repr(item) for item in teacher_ids) + ",)"


class _MultiAdapter:
  """Stand-in for the trusted multi-motion adapter; builds no scene.

  It exposes the same surface the CLI consumes from the real adapter: the
  manifest contract, an immutable reference library with a resolved body
  selection, the ordered motion-to-teacher-code mapping, the frozen bank, a live
  audit carrying the slot allocation/phase policy/seed provenance, one private
  command copy, and owned snapshots.
  """

  auto_reset = True

  def __init__(
    self,
    cohort: CohortContract,
    schema,
    teacher_ids,
    *,
    phase_policy: str = "uniform",
    num_envs: int = 2,
    seed: int | None = None,
    device: str = "cpu",
  ) -> None:
    selected = tuple(t for t in cohort.teachers if t.id in set(teacher_ids))
    self.cohort = cohort
    self.schema = schema
    library = MotionLibrary.from_clips(
      [
        MotionClipSpec(
          teacher_id=teacher.id,
          motion_file=teacher.entry.motion,
          teacher_code=index,
          expected_frames=teacher.reference.frames,
          expected_fps=teacher.reference.fps,
        )
        for index, teacher in enumerate(selected)
      ],
      device=device,
    )
    self.library = library.with_body_selection(
      BodySelection(
        indices=tuple(range(len(cohort.body_names))),
        source_body_count=int(library.source_body_count),
        names=cohort.body_names,
      )
    )
    self.bank = build_cohort_teacher_bank(cohort, device=device)
    slots = stratified_slot_allocation(
      [teacher.entry.sampling_weight for teacher in selected],
      num_envs,
      teacher_ids=[teacher.id for teacher in selected],
    )
    self.audit = SimpleNamespace(
      slots=slots,
      mapping_digest=self.library.mapping_digest(),
      phase_policy=phase_policy,
      semantic_overrides=(),
      seed_provenance=RuntimeSeedProvenance(
        requested_seed=seed,
        effective_seed=seed,
        applied_before_construction=seed is not None,
      ),
    )
    self.motion = SimpleNamespace(
      cfg=SimpleNamespace(
        sampling_mode=phase_policy,
        pose_range={
          "x": (-0.025, 0.025),
          "y": (-0.025, 0.025),
          "z": (-0.005, 0.005),
          "roll": (-0.05, 0.05),
          "pitch": (-0.05, 0.05),
          "yaw": (-0.1, 0.1),
        },
        velocity_range={
          "x": (-0.25, 0.25),
          "y": (-0.25, 0.25),
          "z": (-0.1, 0.1),
        },
        joint_position_range=(-0.05, 0.05),
      )
    )
    self.env = SimpleNamespace(
      num_envs=num_envs,
      device=device,
      command_manager=SimpleNamespace(get_term=lambda name: self.motion),
    )
    self.num_envs = num_envs
    self.reset_calls = 0
    self.step_calls = 0
    self.close_calls = 0
    self.closed = False
    self.motion_ids = torch.tensor(
      [int(item) for item in slots.row_motion_ids], dtype=torch.int64
    )
    self._current = self._snapshot(0)

  @property
  def motion_teacher_codes(self) -> dict[int, int]:
    return {clip.motion_id: clip.teacher_code for clip in self.library.clips}

  def _snapshot(self, step: int) -> DistillationSnapshot:
    ids = self.motion_ids
    batch = int(ids.shape[0])
    codes = self.library.teacher_codes_for(ids)
    lengths = self.library.frame_counts_for(ids)
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
      teacher_observation=torch.full((batch, int(self.bank.obs_dim)), 0.25),
      features=features,
      packed=pack_observations(features, self.schema),
      teacher_id="multiple",
      teacher_code=-1,
      motion_id=ids.clone(),
      reference_frame=torch.remainder(torch.full((batch,), step), lengths),
      segment_id=torch.zeros(batch, dtype=torch.int64),
      generation_id=torch.zeros(batch, dtype=torch.int64),
      teacher_codes=codes,
    )

  def snapshot(self) -> DistillationSnapshot:
    return self._current

  def reset(self, seed: int | None = None) -> DistillationSnapshot:
    del seed
    self.reset_calls += 1
    self._current = self._snapshot(0)
    return self._current

  def step(self, action: torch.Tensor) -> DistillationStep:
    self.step_calls += 1
    self._current = self._snapshot(self.step_calls)
    batch = int(self._current.packed.batch_size)
    return DistillationStep(
      self._current,
      torch.zeros(batch),
      torch.zeros(batch, dtype=torch.bool),
      torch.zeros(batch, dtype=torch.bool),
      {},
    )

  def close(self) -> None:
    self.close_calls += 1
    self.closed = True


def _install_fake_multi_adapter(
  monkeypatch: pytest.MonkeyPatch,
  calls: list[dict],
  schema,
  adapters: list[_MultiAdapter] | None = None,
) -> None:
  def factory(cohort, teacher_ids, **kwargs):
    calls.append({"teacher_ids": tuple(teacher_ids), **kwargs})
    adapter = _MultiAdapter(
      cohort,
      schema,
      teacher_ids,
      phase_policy=kwargs.get("phase_policy", "uniform"),
      num_envs=kwargs.get("num_envs", 2),
      seed=kwargs.get("seed"),
      device=kwargs.get("device", "cpu"),
    )
    if adapters is not None:
      adapters.append(adapter)
    return adapter

  # The cohort training path now builds its environment through the shared
  # cohort recipe (``cohort_setup``), so the fake must stand in for the factory
  # there as well as for the CLI's own remaining call sites (pinned evaluation
  # and playback).
  monkeypatch.setattr(
    "mjlab.tasks.tracking.distillation.cohort_setup.make_multi_teacher_distillation_adapter",
    factory,
  )
  monkeypatch.setattr(
    "mjlab.scripts.distill.make_multi_teacher_distillation_adapter", factory
  )


class _Labeler(torch.nn.Module):
  """Minimal label source standing in for a frozen teacher."""

  action_dim = 31

  def label(self, observations: torch.Tensor) -> torch.Tensor:
    return torch.ones(observations.shape[0], self.action_dim)


def _single_snapshot(step: int, batch: int, schema) -> DistillationSnapshot:
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


class _SingleAdapter:
  """Stand-in for a pinned single-motion adapter; builds no scene.

  ``persistent`` models simulator state that survives a reset (startup
  randomization, event timers, accumulated rollout state), so a test can prove
  that a second mode did not inherit the first mode's environment: each mode
  must get a freshly constructed adapter.
  """

  auto_reset = True

  def __init__(self, seed: int | None, num_envs: int, schema) -> None:
    self.schema = DEFAULT_SCHEMA if schema is None else schema
    self.teacher = _Labeler()
    self.num_envs = num_envs
    self.reset_calls = 0
    self.step_calls = 0
    self.close_calls = 0
    self.closed = False
    self.persistent = 0
    self.persistent_at_construction = 0
    self.observed_prior_persistent: list[int] = []
    self.index = -1
    self.motion = SimpleNamespace(cfg=SimpleNamespace(sampling_mode="weighted"))
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
    self._current = _single_snapshot(0, num_envs, self.schema)

  def snapshot(self) -> DistillationSnapshot:
    return self._current

  def reset(self, seed: int | None = None) -> DistillationSnapshot:
    del seed
    self.reset_calls += 1
    self._current = _single_snapshot(0, self.num_envs, self.schema)
    return self._current

  def step(self, action: torch.Tensor) -> DistillationStep:
    self.step_calls += 1
    self.persistent += 1
    self._current = _single_snapshot(self.step_calls, self.num_envs, self.schema)
    batch = self.num_envs
    return DistillationStep(
      self._current,
      torch.zeros(batch),
      torch.zeros(batch, dtype=torch.bool),
      torch.zeros(batch, dtype=torch.bool),
      {},
    )

  def close(self) -> None:
    self.close_calls += 1
    self.closed = True
    _SINGLE_ADAPTER_EVENTS.append(("close", self.index))


_SINGLE_ADAPTER_EVENTS: list[tuple[str, int]] = []
"""Ordered construct/close log for the pinned fake adapters."""


def _install_fake_single_adapter(
  monkeypatch: pytest.MonkeyPatch,
  calls: list[dict],
  schema,
  adapters: list[_SingleAdapter] | None = None,
) -> None:
  _SINGLE_ADAPTER_EVENTS.clear()
  counter = [0]

  def factory(cohort, teacher_id: str = TEACHER_ID, **kwargs):
    calls.append({"teacher_id": teacher_id, **kwargs})
    prior = [] if adapters is None else [item.persistent for item in adapters]
    adapter = _SingleAdapter(
      kwargs.get("seed"),
      kwargs.get("num_envs", 1),
      kwargs.get("schema") or schema,
    )
    adapter.index = counter[0]
    counter[0] += 1
    adapter.observed_prior_persistent = prior
    adapter.persistent_at_construction = adapter.persistent
    if adapters is not None:
      adapters.append(adapter)
    _SINGLE_ADAPTER_EVENTS.append(("construct", adapter.index))
    return adapter

  monkeypatch.setattr("mjlab.scripts.distill.make_distillation_adapter", factory)


class _FakePlayViewer:
  """Runs one obs -> policy -> step cycle and records what it saw."""

  instances: list[_FakePlayViewer] = []

  def __init__(self, env, policy, frame_rate=60.0, checkpoint_manager=None) -> None:
    self.env = env
    self.policy = policy
    self.checkpoint_manager = checkpoint_manager
    self.action: torch.Tensor | None = None
    _FakePlayViewer.instances.append(self)

  def run(self) -> None:
    snapshot = self.env.get_observations()
    self.action = self.policy(snapshot)
    self.env.step(self.action)


def _install_fake_viewers(monkeypatch: pytest.MonkeyPatch) -> None:
  _FakePlayViewer.instances = []
  monkeypatch.setattr("mjlab.viewer.ViserPlayViewer", _FakePlayViewer)
  monkeypatch.setattr("mjlab.viewer.NativeMujocoViewer", _FakePlayViewer)


def _train_cohort(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  extra: list[str],
  *,
  teacher_ids: tuple[str, ...] = TEACHER_IDS,
) -> tuple[str | int | None, str, str]:
  code = invoke(
    monkeypatch,
    [
      "distill",
      "train",
      "--manifest",
      str(MANIFEST),
      "--repo-root",
      str(REPO_ROOT),
      "--teacher-ids",
      _teacher_ids_arg(teacher_ids),
      *_TRAIN_ARGV,
      *extra,
      # Every invocation writes under the test's tmp directory: a test must
      # never add an artifact to the repository's protected logs/ tree.
      *([] if "--output-dir" in extra else ["--output-dir", str(tmp_path / "run")]),
    ],
  )
  captured = capsys.readouterr()
  return code, captured.out, captured.err


def _write_version_one_checkpoint(path: Path, cohort, schema, teacher_id: str) -> None:
  model = distill.ConditionalVAE(schema)
  replay = LabeledReplayBuffer(8, schema)
  trainer = VaeDistillationTrainer(
    model, replay, TrainingConfig(accumulation_steps=1, minibatch_size=1)
  )
  path.parent.mkdir(parents=True, exist_ok=True)
  save_checkpoint(
    path,
    trainer,
    replay,
    counters={"iteration": 3, "segment_namespace": 0},
    schedule={"max_iterations": 3},
    resolved_config={"provenance_version": distill.PROVENANCE_VERSION},
    teacher_hashes=cohort.teacher(teacher_id).hashes,
    control_contract=_control_metadata(cohort, teacher_id),
  )


# --- pure aggregation contracts -------------------------------------------------


def test_cohort_aggregate_reports_macro_and_clip_duration_weights() -> None:
  """The two aggregates differ and neither drops a partially covered metric."""
  weights = {0: 6.0 / 50.0, 1: 4.0 / 50.0}
  per_motion = {
    0: {
      "tracking_global_body_pose_error": 1.0,
      "completion_rate": 0.5,
      "timeout_rate": 0.0,
    },
    1: {
      "tracking_global_body_pose_error": 3.0,
      "completion_rate": 1.0,
      "reset_rate": 0.0,
    },
  }

  aggregate = distill._aggregate_motion_metrics(per_motion, weights)

  assert aggregate["motion_ids"] == [0, 1]
  assert aggregate["per_motion"]["0"]["completion_rate"] == 0.5
  # Equal motion weight: (1 + 3) / 2.
  assert aggregate["macro"]["tracking_global_body_pose_error"] == 2.0
  # Clip-duration weight: (6/50 * 1 + 4/50 * 3) / (10/50) = 1.8, not 2.0.
  assert aggregate["clip_duration_weighted"]["tracking_global_body_pose_error"] == (
    pytest.approx(1.8)
  )
  assert aggregate["contributed_motions"]["completion_rate"] == 2
  assert aggregate["metrics_missing_from_some_motion"] == {
    "reset_rate": ["0"],
    "timeout_rate": ["1"],
  }


def test_cohort_aggregate_keeps_a_failing_motion_in_the_macro() -> None:
  """One bad motion cannot be averaged away from the reported metrics."""
  per_motion = {
    0: {"completion_rate": 1.0, "failure_rate": 0.0},
    1: {"completion_rate": 0.0, "failure_rate": 1.0},
  }
  aggregate = distill._aggregate_motion_metrics(per_motion, {0: 1.0, 1: 1.0})

  assert aggregate["macro"]["failure_rate"] == 0.5
  assert aggregate["per_motion"]["1"]["failure_rate"] == 1.0
  assert aggregate["clip_duration_weighted"]["failure_rate"] == 0.5


def test_cohort_mode_report_names_censoring_and_outcomes_explicitly() -> None:
  """Completion is over completed-or-failed segments; censoring stays visible."""
  segments = (
    EvaluationSegment(
      env_index=0,
      segment_id=0,
      generation_id=0,
      steps=10,
      completed=True,
      failed=False,
      capped=False,
      metrics={"tracking_global_body_pose_error": 1.0},
      outcome="reference_complete",
      motion_id=0,
      teacher_code=0,
    ),
    EvaluationSegment(
      env_index=0,
      segment_id=0,
      generation_id=1,
      steps=5,
      completed=False,
      failed=False,
      capped=True,
      metrics={},
      outcome="step_cap",
      motion_id=0,
      teacher_code=0,
    ),
  )
  result = EvaluationResult(
    mode="student",
    rollout_latent="mean",
    steps=15,
    segments=segments,
    metrics={"completion_rate": 1.0},
  )

  report = distill._cohort_mode_report(result)

  assert report["segments"] == 2
  censoring = report["per_motion_censoring"]["0"]
  assert censoring["segments"] == 2
  assert censoring["completion_known_segments"] == 1
  assert censoring["censored_segments"] == 1
  assert censoring["outcomes"] == {"reference_complete": 1, "step_cap": 1}
  assert report["per_motion"][0]["completion_rate"] == 1.0


# --- selector and refusal contracts --------------------------------------------


def test_cohort_help_exposes_the_new_surfaces(
  monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  code = invoke(monkeypatch, ["distill", "train", "--help"])
  captured = capsys.readouterr()
  assert code == 0
  assert "--teacher-ids" in captured.out
  assert "--teacher-id" in captured.out

  code = invoke(monkeypatch, ["distill", "evaluate-cohort", "--help"])
  captured = capsys.readouterr()
  assert code == 0
  for flag in (
    "--teacher-ids",
    "--mode",
    "--sampling-mode",
    "--checkpoint",
    "--reset-profile",
  ):
    assert flag in captured.out
  # Cohort evaluation is model-only: no trainer setting is advertised.
  for flag in ("--minibatch-size", "--accumulation-steps", "--replay-capacity"):
    assert flag not in captured.out

  monkeypatch.setattr(sys, "argv", ["distill", "--help"])
  main()
  captured = capsys.readouterr()
  # The top-level surface must keep naming every command, and must not have
  # become a passthrough that prints nothing.
  assert captured.out.startswith("usage: distill <COMMAND>")
  for command in (
    "validate-teachers",
    "train",
    "evaluate",
    "evaluate-cohort",
    "play",
    "export",
  ):
    assert command in captured.out


def test_legacy_train_rejects_standing_options_with_singleton_pointer(
  monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  code = invoke(
    monkeypatch,
    ["distill", "train", "--reset-policy", "standing-mixture"],
  )
  captured = capsys.readouterr()
  assert code == 1
  assert "--teacher-ids \"('tennis_000',)\"" in captured.err


def test_train_rejects_both_singular_and_plural_selection_before_construction(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  calls: list[dict] = []
  _install_fake_multi_adapter(monkeypatch, calls, schema=None)

  code = invoke(
    monkeypatch,
    [
      "distill",
      "train",
      "--manifest",
      str(MANIFEST),
      "--repo-root",
      str(REPO_ROOT),
      "--teacher-id",
      OTHER_TEACHER_ID,
      "--teacher-ids",
      _teacher_ids_arg(TEACHER_IDS),
      *_TRAIN_ARGV,
    ],
  )

  captured = capsys.readouterr()
  assert code == 1
  assert captured.out == ""
  assert "mutually exclusive" in captured.err
  assert calls == []


def test_train_reports_an_unknown_plural_selection(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  calls: list[dict] = []
  _install_fake_multi_adapter(monkeypatch, calls, schema=None)

  code, out, err = _train_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    ["--max-iterations", "1"],
    teacher_ids=(TEACHER_ID, "not-a-teacher"),
  )

  assert code == 1
  assert out == ""
  assert "not-a-teacher" in err
  # A bad selection is refused before any environment is constructed.
  assert calls == []


def test_train_rejects_a_repeated_plural_selection_before_construction(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  calls: list[dict] = []
  _install_fake_multi_adapter(monkeypatch, calls, schema=None)

  code, out, err = _train_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    ["--max-iterations", "1"],
    teacher_ids=(TEACHER_ID, TEACHER_ID),
  )

  assert code == 1
  assert out == ""
  assert "duplicate" in err.lower()
  assert calls == []


def test_evaluate_cohort_rejects_a_missing_student_checkpoint(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  calls: list[dict] = []
  _install_fake_single_adapter(monkeypatch, calls, schema=None)

  code = invoke(
    monkeypatch,
    [
      "distill",
      "evaluate-cohort",
      "--manifest",
      str(MANIFEST),
      "--repo-root",
      str(REPO_ROOT),
      "--mode",
      "student",
    ],
  )

  captured = capsys.readouterr()
  assert code == 1
  assert "requires --checkpoint" in captured.err
  assert calls == []


def test_evaluate_cohort_rejects_a_non_cohort_checkpoint(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  """A version-1 artifact is refused instead of being pinned as a cohort."""
  resolved = resolve_cohort(load_manifest(MANIFEST, repo_root=REPO_ROOT))
  live_schema = make_schema(DecoderMode.GRAVITY, tuple(resolved.actions.joint_names))
  version_one = tmp_path / "version-one.pt"
  _write_version_one_checkpoint(version_one, resolved, live_schema, TEACHER_ID)
  assert distill._checkpoint_version(version_one) == 1
  calls: list[dict] = []
  _install_fake_single_adapter(monkeypatch, calls, schema=None)

  code = invoke(
    monkeypatch,
    [
      "distill",
      "evaluate-cohort",
      "--manifest",
      str(MANIFEST),
      "--repo-root",
      str(REPO_ROOT),
      "--checkpoint",
      str(version_one),
      "--mode",
      "student",
    ],
  )

  captured = capsys.readouterr()
  assert code == 1
  assert "version-2 M4 cohort checkpoint" in captured.err
  assert calls == []


# --- train / resume / evaluate / cohort evaluation end to end -------------------


def test_cohort_train_saves_a_version_two_checkpoint_and_member_reports(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  schema,
  motion_frames: dict[str, int],
) -> None:
  """One shared student trains over both clips and each member is pinnable."""
  calls: list[dict] = []
  adapters: list[_MultiAdapter] = []
  _install_fake_multi_adapter(monkeypatch, calls, schema, adapters)
  output_dir = tmp_path / "run"

  code, out, err = _train_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    ["--max-iterations", "2", "--output-dir", str(output_dir)],
  )

  assert code == 0, err
  report = json.loads(out)
  assert calls[0]["teacher_ids"] == TEACHER_IDS
  assert calls[0]["phase_policy"] == "uniform"
  assert calls[0]["seed"] == 7
  assert calls[0]["task_id"] is None
  assert report["iteration"] == 2
  assert report["cohort"]["teacher_ids"] == list(TEACHER_IDS)
  assert report["cohort"]["slot_counts"] == [1, 1]
  assert report["cohort"]["slot_weights"] == [1.0, 1.0]
  assert report["cohort"]["replay_motion_ids"] == [0, 1]
  assert report["cohort"]["clip_frames"] == [
    motion_frames[TEACHER_IDS[0]],
    motion_frames[TEACHER_IDS[1]],
  ]
  assert report["cohort"]["phase_policy"] == "uniform"
  assert report["resolved_config"]["phase_policy"] == "uniform"
  assert report["resolved_config"]["teacher_ids"] == list(TEACHER_IDS)
  assert report["resolved_config"]["cohort_digest"] == report["cohort"]["cohort_digest"]
  # The balanced per-motion replay is the buffer this run actually trained with.
  assert report["resolved_config"]["replay"]["kind"] == "balanced-motion-replay"
  assert report["resolved_config"]["replay"]["motion_ids"] == [0, 1]
  assert sum(report["resolved_config"]["replay"]["quotas"]) == 24
  telemetry = report["replay"]
  assert telemetry == report["iterations"][-1]["replay"]
  assert telemetry["capacity"] == 24
  assert telemetry["ready"] is True
  assert telemetry["inserted"] >= telemetry["retained"] > 0
  assert telemetry["drawn"] > 0
  assert telemetry["occupancy"] == telemetry["retained"] / telemetry["capacity"]
  assert [item["motion_id"] for item in telemetry["motions"]] == [0, 1]
  assert all(
    item["inserted"] > 0 and item["drawn"] > 0 for item in telemetry["motions"]
  )
  assert report["schedule"]["evaluation_sampling_mode"] == "uniform"
  # Per-motion collection attribution is reported alongside the totals.
  motion_stats = report["iterations"][-1]["collection"]["motion_stats"]
  assert sorted(item["motion_id"] for item in motion_stats) == [0, 1]
  assert all(item["samples"] > 0 for item in motion_stats)

  checkpoint = output_dir / "checkpoint-final.pt"
  assert distill._is_cohort_checkpoint(checkpoint)
  assert distill._checkpoint_version(checkpoint) == 2

  # Each member is separately evaluable from the same shared checkpoint, and
  # the report keeps the whole trained cohort identity.
  _install_fake_single_adapter(monkeypatch, [], schema)
  member_ids = []
  for teacher_id in TEACHER_IDS:
    code = invoke(
      monkeypatch,
      [
        "distill",
        "evaluate",
        "--manifest",
        str(MANIFEST),
        "--repo-root",
        str(REPO_ROOT),
        "--teacher-id",
        teacher_id,
        "--checkpoint",
        str(checkpoint),
        "--mode",
        "student",
        "--steps",
        "2",
        "--num-envs",
        "1",
        "--seed",
        "7",
      ],
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    evaluation = json.loads(captured.out)
    identity = evaluation["checkpoint_model"]
    assert identity["checkpoint_version"] == 2
    assert identity["trained_teacher_ids"] == list(TEACHER_IDS)
    assert identity["requested_teacher_id"] == teacher_id
    assert identity["cohort_digest"] == report["cohort"]["cohort_digest"]
    assert evaluation["checkpoint_version"] == 2
    assert evaluation["result"]["settings"]["motion_ids"] == [identity["motion_id"]]
    for segment in evaluation["result"]["segments"]:
      assert segment["motion_id"] == identity["motion_id"]
      assert segment["teacher_code"] == identity["teacher_code"]
    member_ids.append(identity["motion_id"])

  assert member_ids == sorted(member_ids)
  assert len(set(member_ids)) == len(TEACHER_IDS)

  # A member the saved cohort does not contain is refused, not silently
  # re-labeled as another motion.
  code = invoke(
    monkeypatch,
    [
      "distill",
      "evaluate",
      "--manifest",
      str(MANIFEST),
      "--repo-root",
      str(REPO_ROOT),
      "--teacher-id",
      "not-a-member",
      "--checkpoint",
      str(checkpoint),
      "--mode",
      "student",
      "--steps",
      "1",
    ],
  )
  captured = capsys.readouterr()
  assert code == 1
  assert captured.out == ""
  assert "not-a-member" in captured.err


@pytest.mark.parametrize("steps", [0, -1])
def test_cohort_evaluate_refuses_invalid_step_budget_before_resolution(
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  steps: int,
) -> None:
  def forbidden(*args, **kwargs):
    pytest.fail("invalid step budget must be rejected before resolving artifacts")

  monkeypatch.setattr(distill, "_resolve", forbidden)
  assert distill._evaluate_cohort(mode="teacher", steps=steps) == 1
  assert "positive integer" in capsys.readouterr().err


def test_cohort_evaluate_cohort_reports_every_motion_with_weights(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  schema,
  motion_frames: dict[str, int],
  motion_fps: float,
) -> None:
  _install_fake_multi_adapter(monkeypatch, [], schema)
  code, _out, err = _train_cohort(
    tmp_path, monkeypatch, capsys, ["--max-iterations", "1"]
  )
  assert code == 0, err
  checkpoint = tmp_path / "run" / "checkpoint-final.pt"
  calls: list[dict] = []
  adapters: list[_SingleAdapter] = []
  _install_fake_single_adapter(monkeypatch, calls, schema, adapters)
  report_path = tmp_path / "cohort.json"

  code = invoke(
    monkeypatch,
    [
      "distill",
      "evaluate-cohort",
      "--manifest",
      str(MANIFEST),
      "--repo-root",
      str(REPO_ROOT),
      "--teacher-ids",
      _teacher_ids_arg(TEACHER_IDS),
      "--checkpoint",
      str(checkpoint),
      "--num-envs",
      "1",
      "--steps",
      "2",
      "--seed",
      "7",
      "--sampling-mode",
      "start",
      "--report",
      str(report_path),
    ],
  )

  captured = capsys.readouterr()
  assert code == 0, captured.err
  report = json.loads(captured.out)
  assert report["status"] == "cohort_evaluation"
  assert report["mode"] == "both"
  assert report["sampling_mode"] == "start"
  assert report["teacher_ids"] == list(TEACHER_IDS)
  assert report["complete"] is True
  assert report["evaluation_errors"] == []
  assert report["missing_reports"] == []
  assert report["checkpoint_model"]["trained_teacher_ids"] == list(TEACHER_IDS)
  # One pinned environment per motion, each with separate teacher/student
  # reports and its own clip identity.
  assert [entry["teacher_id"] for entry in report["motions"]] == list(TEACHER_IDS)
  for entry in report["motions"]:
    assert set(entry["reports"]) == {"teacher", "student"}
    assert entry["reports"]["student"]["mode"] == "student"
    assert entry["reports"]["teacher"]["mode"] == "teacher"
    assert entry["reports"]["student"]["per_motion_censoring"]
    assert entry["runtime"]["requested_seed"] == 7
    assert entry["frames"] == motion_frames[entry["teacher_id"]]
  # The pinned sampling override reached the live command of every motion, and
  # the packing was built from the saved cohort schema, not a default one.
  assert [call["teacher_id"] for call in calls] == [
    TEACHER_ID,
    TEACHER_ID,
    OTHER_TEACHER_ID,
    OTHER_TEACHER_ID,
  ]
  assert all(adapter.motion.cfg.sampling_mode == "start" for adapter in adapters)
  assert all(adapter.close_calls == 1 for adapter in adapters)
  assert all(call["schema"] is not None for call in calls)
  weights = report["aggregate"]["aggregation_weights"]["clip_duration_weighted"][
    "values"
  ]
  assert weights[TEACHER_IDS[0]] == pytest.approx(
    motion_frames[TEACHER_IDS[0]] / motion_fps
  )
  assert weights[TEACHER_IDS[1]] == pytest.approx(
    motion_frames[TEACHER_IDS[1]] / motion_fps
  )
  # The two aggregates are genuinely different summaries here (unequal clips).
  assert weights[TEACHER_IDS[0]] > weights[TEACHER_IDS[1]]
  for name in ("teacher", "student"):
    aggregate = report["aggregate"]["per_mode"][name]
    assert aggregate["macro"]
    assert aggregate["clip_duration_weighted"]
    assert aggregate["per_motion"]
    assert aggregate["outcomes"]["total"]
  assert report["quality"]["aggregate_is_not_a_quality_pass"] is True
  assert report["quality"]["overall_pass"] is False
  assert json.loads(report_path.read_text()) == report


def test_cohort_evaluate_cohort_records_an_execution_error_per_motion(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  """A failing motion is reported as an execution error, not a silent pass."""
  calls: list[str] = []

  def factory(cohort, teacher_id: str = TEACHER_ID, **kwargs):
    calls.append(teacher_id)
    if teacher_id == OTHER_TEACHER_ID:
      raise RuntimeError("simulator refused to start")
    schema = kwargs.get("schema")
    return _SingleAdapter(kwargs.get("seed"), kwargs.get("num_envs", 1), schema)

  monkeypatch.setattr("mjlab.scripts.distill.make_distillation_adapter", factory)

  code = invoke(
    monkeypatch,
    [
      "distill",
      "evaluate-cohort",
      "--manifest",
      str(MANIFEST),
      "--repo-root",
      str(REPO_ROOT),
      "--mode",
      "teacher",
      "--steps",
      "2",
      "--seed",
      "7",
    ],
  )

  captured = capsys.readouterr()
  assert code == 1
  assert calls == list(TEACHER_IDS)
  report = json.loads(captured.out)
  assert report["complete"] is False
  assert report["evaluation_errors"] == [
    {
      "teacher_id": OTHER_TEACHER_ID,
      "mode": "teacher",
      "error_type": "RuntimeError",
      "error": "simulator refused to start",
    }
  ]
  assert report["missing_reports"] == [
    {"teacher_id": OTHER_TEACHER_ID, "mode": "teacher"}
  ]
  # The healthy motion's own report is still present and separately attributable.
  healthy = report["motions"][0]
  assert healthy["teacher_id"] == TEACHER_ID
  assert "teacher" in healthy["reports"]
  assert set(report["aggregate"]["per_mode"]) == {"teacher"}


def test_cohort_evaluate_cohort_teacher_mode_needs_no_checkpoint(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  schema,
) -> None:
  calls: list[dict] = []
  _install_fake_single_adapter(monkeypatch, calls, schema)

  code = invoke(
    monkeypatch,
    [
      "distill",
      "evaluate-cohort",
      "--manifest",
      str(MANIFEST),
      "--repo-root",
      str(REPO_ROOT),
      "--mode",
      "teacher",
      "--steps",
      "2",
      "--seed",
      "7",
    ],
  )

  captured = capsys.readouterr()
  assert code == 0, captured.err
  report = json.loads(captured.out)
  assert report["mode"] == "teacher"
  assert report["checkpoint"] is None
  assert report["checkpoint_model"] is None
  assert report["complete"] is True
  assert [entry["teacher_id"] for entry in report["motions"]] == list(TEACHER_IDS)
  assert all(set(entry["reports"]) == {"teacher"} for entry in report["motions"])
  assert sorted(report["aggregate"]["per_mode"]) == ["teacher"]

  # A checkpoint supplied in teacher-only mode is refused, not silently ignored.
  code = invoke(
    monkeypatch,
    [
      "distill",
      "evaluate-cohort",
      "--manifest",
      str(MANIFEST),
      "--repo-root",
      str(REPO_ROOT),
      "--mode",
      "teacher",
      "--checkpoint",
      "state.pt",
      "--steps",
      "1",
    ],
  )
  captured = capsys.readouterr()
  assert code == 1
  assert "ignores a checkpoint" in captured.err


def test_cohort_resume_requires_the_whole_stored_cohort(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  schema,
) -> None:
  """Resume reproduces the stored cohort and only the budget may grow."""
  calls: list[dict] = []
  _install_fake_multi_adapter(monkeypatch, calls, schema)

  first_dir = tmp_path / "first"
  code, out, err = _train_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    ["--max-iterations", "1", "--output-dir", str(first_dir)],
  )
  assert code == 0, err
  first = first_dir / "checkpoint-final.pt"

  second_dir = tmp_path / "second"
  code, out, err = _train_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    ["--max-iterations", "2", "--resume", str(first), "--output-dir", str(second_dir)],
  )
  assert code == 0, err
  report = json.loads(out)
  assert report["iteration"] == 2
  assert report["resume"]["provenance_verified"] is True
  assert report["resume"]["mismatches"] == []
  assert report["resume"]["budget"]["extended"] is True
  second = second_dir / "checkpoint-final.pt"

  # A different collection schedule is refused instead of silently changing it.
  code, out, err = _train_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    [
      "--max-iterations",
      "3",
      "--collection-steps",
      "2",
      "--resume",
      str(second),
      "--output-dir",
      str(tmp_path / "refused"),
    ],
  )
  assert code == 1, (out, err)
  assert out == ""
  assert "schedule.collection_steps" in err

  # A subset cohort is a different stored cohort and cannot resume the run.
  code, out, err = _train_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    [
      "--max-iterations",
      "3",
      "--resume",
      str(second),
      "--output-dir",
      str(tmp_path / "subset"),
    ],
    teacher_ids=(TEACHER_ID,),
  )
  assert code == 1, (out, err)
  assert out == ""
  # The strict cohort/replay-policy comparison refuses the smaller cohort.
  assert "does not match" in err
  assert not (tmp_path / "subset" / "checkpoint-final.pt").exists()

  # The version-1 loader never reinterprets a version-2 artifact.  The single
  # adapter is mocked too: this check must not build a real environment.
  _install_fake_single_adapter(monkeypatch, [], schema)
  code = invoke(
    monkeypatch,
    [
      "distill",
      "train",
      "--manifest",
      str(MANIFEST),
      "--repo-root",
      str(REPO_ROOT),
      "--teacher-id",
      TEACHER_ID,
      *_TRAIN_ARGV,
      "--max-iterations",
      "2",
      "--resume",
      str(second),
      "--output-dir",
      str(tmp_path / "legacy"),
    ],
  )
  captured = capsys.readouterr()
  assert code == 1
  assert captured.out == ""
  assert "load_cohort_checkpoint" in captured.err


def test_cohort_train_resume_refuses_a_single_teacher_checkpoint(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  cohort: CohortContract,
  schema,
) -> None:
  """The plural path never reinterprets a version-1 resume input."""
  version_one = tmp_path / "version-one.pt"
  _write_version_one_checkpoint(version_one, cohort, schema, TEACHER_ID)
  calls: list[dict] = []
  _install_fake_multi_adapter(monkeypatch, calls, schema)

  code, out, err = _train_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    ["--max-iterations", "2", "--resume", str(version_one)],
  )

  assert code == 1
  assert out == ""
  assert "version-2 M4 cohort checkpoint" in err
  assert calls == []


def test_cohort_evaluate_cohort_attributes_every_level_to_the_cohort_motion(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  cohort: CohortContract,
  schema,
) -> None:
  """A reversed or subset request never reports a local singleton 0 as cohort 0.

  Every pinned single-motion environment reports its own local clip 0, so the
  second member must still be attributed to its saved cohort motion id and
  teacher code at the motion block, the per-motion entry, the censoring entry,
  the aggregate per-motion map, and the aggregate outcome counts.
  """
  _install_fake_multi_adapter(monkeypatch, [], schema)
  code, _out, err = _train_cohort(
    tmp_path, monkeypatch, capsys, ["--max-iterations", "1"]
  )
  assert code == 0, err
  checkpoint = tmp_path / "run" / "checkpoint-final.pt"
  calls: list[dict] = []
  _install_fake_single_adapter(monkeypatch, calls, schema)

  # The request is deliberately reversed relative to the manifest order.
  code = invoke(
    monkeypatch,
    [
      "distill",
      "evaluate-cohort",
      "--manifest",
      str(MANIFEST),
      "--repo-root",
      str(REPO_ROOT),
      "--teacher-ids",
      _teacher_ids_arg((OTHER_TEACHER_ID, TEACHER_ID)),
      "--checkpoint",
      str(checkpoint),
      "--steps",
      "2",
      "--seed",
      "7",
    ],
  )

  captured = capsys.readouterr()
  assert code == 0, captured.err
  report = json.loads(captured.out)
  # Normalized to manifest order, and the normalization is visible.
  assert report["requested_teacher_ids"] == [OTHER_TEACHER_ID, TEACHER_ID]
  assert report["teacher_ids"] == list(TEACHER_IDS)
  assert report["selection_normalized_to_manifest_order"] is True
  assert [entry["teacher_id"] for entry in report["motions"]] == list(TEACHER_IDS)
  # The live environments were pinned in manifest order as well.
  assert [call["teacher_id"] for call in calls] == [
    TEACHER_ID,
    TEACHER_ID,
    OTHER_TEACHER_ID,
    OTHER_TEACHER_ID,
  ]

  first, second = report["motions"]
  assert (first["motion_id"], first["teacher_code"]) == (0, 0)
  assert (second["motion_id"], second["teacher_code"]) == (1, 1)
  assert second["motion_id_source"] == "saved_cohort_member"
  assert first["motion"] == str(cohort.teacher(TEACHER_ID).entry.motion)
  for entry, expected_id in ((first, 0), (second, 1)):
    for name in ("teacher", "student"):
      mode_report = entry["reports"][name]
      assert mode_report["settings"]["motion_ids"] == [expected_id]
      item = mode_report["per_motion"][0]
      assert item["motion_id"] == expected_id
      assert item["teacher_code"] == expected_id
      censoring = mode_report["per_motion_censoring"][str(expected_id)]
      assert censoring["motion_id"] == expected_id
      assert censoring["teacher_code"] == expected_id
      aggregate = report["aggregate"]["per_mode"][name]
      assert sorted(aggregate["per_motion"]) == ["0", "1"]
      assert sorted(aggregate["outcomes"]["per_motion"]) == ["0", "1"]
      assert aggregate["motion_ids"] == [0, 1]
  assert report["aggregate"]["motion_ids"] == {
    TEACHER_ID: 0,
    OTHER_TEACHER_ID: 1,
  }

  # A subset selection keeps the saved member's cohort id (1), not local 0.
  calls.clear()
  code = invoke(
    monkeypatch,
    [
      "distill",
      "evaluate-cohort",
      "--manifest",
      str(MANIFEST),
      "--repo-root",
      str(REPO_ROOT),
      "--teacher-ids",
      _teacher_ids_arg((OTHER_TEACHER_ID,)),
      "--checkpoint",
      str(checkpoint),
      "--steps",
      "2",
      "--seed",
      "7",
    ],
  )
  captured = capsys.readouterr()
  assert code == 0, captured.err
  subset = json.loads(captured.out)
  assert subset["teacher_ids"] == [OTHER_TEACHER_ID]
  assert subset["selection_normalized_to_manifest_order"] is False
  assert calls and {call["teacher_id"] for call in calls} == {OTHER_TEACHER_ID}
  only = subset["motions"][0]
  assert (only["motion_id"], only["teacher_code"]) == (1, 1)
  assert only["reports"]["student"]["per_motion"][0]["motion_id"] == 1
  assert "1" in only["reports"]["student"]["per_motion_censoring"]
  assert list(subset["aggregate"]["per_mode"]["student"]["per_motion"]) == ["1"]
  assert subset["aggregate"]["per_mode"]["student"]["motion_ids"] == [1]

  # Repeated ids are refused before any environment is constructed.
  calls.clear()
  code = invoke(
    monkeypatch,
    [
      "distill",
      "evaluate-cohort",
      "--manifest",
      str(MANIFEST),
      "--repo-root",
      str(REPO_ROOT),
      "--teacher-ids",
      _teacher_ids_arg((TEACHER_ID, TEACHER_ID)),
      "--checkpoint",
      str(checkpoint),
      "--steps",
      "1",
    ],
  )
  captured = capsys.readouterr()
  assert code == 1
  assert "repeats" in captured.err
  assert calls == []


def test_cohort_evaluate_cohort_teacher_baselines_use_the_manifest_mapping(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  schema,
) -> None:
  """A teacher-only run states the explicit manifest position, never None."""
  calls: list[dict] = []
  _install_fake_single_adapter(monkeypatch, calls, schema)

  code = invoke(
    monkeypatch,
    [
      "distill",
      "evaluate-cohort",
      "--manifest",
      str(MANIFEST),
      "--repo-root",
      str(REPO_ROOT),
      "--teacher-ids",
      _teacher_ids_arg((OTHER_TEACHER_ID,)),
      "--mode",
      "teacher",
      "--steps",
      "2",
      "--seed",
      "7",
    ],
  )

  captured = capsys.readouterr()
  assert code == 0, captured.err
  report = json.loads(captured.out)
  assert report["checkpoint_model"] is None
  entry = report["motions"][0]
  assert entry["teacher_id"] == OTHER_TEACHER_ID
  assert entry["motion_id_source"] == "manifest_position"
  assert (entry["motion_id"], entry["teacher_code"]) == (1, 1)
  item = entry["reports"]["teacher"]["per_motion"][0]
  assert (item["motion_id"], item["teacher_code"]) == (1, 1)
  assert report["aggregate"]["motion_ids"] == {OTHER_TEACHER_ID: 1}
  assert list(report["aggregate"]["per_mode"]["teacher"]["per_motion"]) == ["1"]
  assert report["aggregate"]["per_mode"]["teacher"]["outcomes"]["per_motion"] == {
    "1": item["outcomes"]
  }


def test_cohort_evaluate_cohort_builds_a_fresh_seeded_adapter_per_mode(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  schema,
) -> None:
  """The student never inherits the teacher rollout's environment state.

  The fake adapter accumulates persistent state in ``step`` (state that a
  ``reset`` does not restore).  Each mode must therefore get its own freshly
  constructed, identically seeded adapter, and each must be closed exactly once
  before the next is built.  No simulator is constructed here.
  """
  _install_fake_multi_adapter(monkeypatch, [], schema)
  code, _out, err = _train_cohort(
    tmp_path, monkeypatch, capsys, ["--max-iterations", "1"]
  )
  assert code == 0, err
  checkpoint = tmp_path / "run" / "checkpoint-final.pt"
  calls: list[dict] = []
  adapters: list[_SingleAdapter] = []
  _install_fake_single_adapter(monkeypatch, calls, schema, adapters)

  code = invoke(
    monkeypatch,
    [
      "distill",
      "evaluate-cohort",
      "--manifest",
      str(MANIFEST),
      "--repo-root",
      str(REPO_ROOT),
      "--teacher-ids",
      _teacher_ids_arg((TEACHER_ID,)),
      "--checkpoint",
      str(checkpoint),
      "--steps",
      "3",
      "--seed",
      "7",
    ],
  )

  captured = capsys.readouterr()
  assert code == 0, captured.err
  # Two modes, one motion: two distinct adapters, both identically seeded.
  assert len(adapters) == 2
  teacher_adapter, student_adapter = adapters
  assert teacher_adapter is not student_adapter
  assert [call["seed"] for call in calls] == [7, 7]
  assert {call["teacher_id"] for call in calls} == {TEACHER_ID}
  # The teacher rollout mutated persistent state, and the student's adapter was
  # constructed clean while that state existed.
  assert teacher_adapter.persistent == 3
  assert student_adapter.observed_prior_persistent == [3]
  assert student_adapter.persistent_at_construction == 0
  # Both were closed, and only one was ever active.
  assert teacher_adapter.close_calls == 1
  assert student_adapter.close_calls == 1
  assert teacher_adapter.closed and student_adapter.closed
  assert _SINGLE_ADAPTER_EVENTS == [
    ("construct", 0),
    ("close", 0),
    ("construct", 1),
    ("close", 1),
  ]
  # The sampling override reached both fresh one-motion commands.
  assert all(adapter.motion.cfg.sampling_mode == "start" for adapter in adapters)


def test_cohort_train_closes_the_adapter_when_a_later_build_step_fails(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  schema,
) -> None:
  """A failure after the environment exists closes it exactly once.

  The caller receives no runner in that case, so the builder is the only owner
  that can release the environment; nothing may be written.
  """
  calls: list[dict] = []
  adapters: list[_MultiAdapter] = []
  _install_fake_multi_adapter(monkeypatch, calls, schema, adapters)

  def _boom(*args, **kwargs):
    raise DistillationError("cohort identity refused after allocation")

  monkeypatch.setattr("mjlab.scripts.distill.cohort_identity_from_adapter", _boom)
  output_dir = tmp_path / "failed-run"

  code, out, err = _train_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    ["--max-iterations", "1", "--output-dir", str(output_dir)],
  )

  assert code == 1, (out, err)
  assert out == ""
  assert "cohort identity refused after allocation" in err
  assert len(adapters) == 1
  assert adapters[0].close_calls == 1
  assert adapters[0].closed
  assert not (output_dir / "checkpoint-final.pt").exists()
  assert not (output_dir / "train-report.json").exists()


def test_cohort_train_closes_the_adapter_when_the_trainer_fails(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  schema,
) -> None:
  """A trainer-construction failure after allocation also closes the adapter."""
  calls: list[dict] = []
  adapters: list[_MultiAdapter] = []
  _install_fake_multi_adapter(monkeypatch, calls, schema, adapters)

  def _boom(*args, **kwargs):
    raise DistillationError("trainer refused after allocation")

  monkeypatch.setattr("mjlab.scripts.distill.VaeDistillationTrainer", _boom)
  output_dir = tmp_path / "failed-trainer"

  code, out, err = _train_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    ["--max-iterations", "1", "--output-dir", str(output_dir)],
  )

  assert code == 1, (out, err)
  assert out == ""
  assert "trainer refused after allocation" in err
  assert len(adapters) == 1
  assert adapters[0].close_calls == 1
  assert adapters[0].closed
  assert not (output_dir / "checkpoint-final.pt").exists()
  assert not (output_dir / "train-report.json").exists()


def test_cohort_train_rejects_impossible_budgets_before_construction(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  """An impossible replay capacity is refused without building an environment."""
  calls: list[dict] = []
  _install_fake_multi_adapter(monkeypatch, calls, schema=None)

  code, out, err = _train_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    ["--max-iterations", "1", "--replay-capacity", "1"],
  )

  assert code == 1
  assert out == ""
  assert "--replay-capacity" in err
  assert calls == []


def test_cohort_export_is_refused_explicitly(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  schema,
) -> None:
  """A cohort checkpoint with the singular audit is refused, not reduced."""
  _install_fake_multi_adapter(monkeypatch, [], schema)
  code, _out, err = _train_cohort(
    tmp_path, monkeypatch, capsys, ["--max-iterations", "1"]
  )
  assert code == 0, err
  checkpoint = tmp_path / "run" / "checkpoint-final.pt"
  audit = tmp_path / "asset-audit.json"
  audit.write_text("{}")

  code = invoke(
    monkeypatch,
    [
      "distill",
      "export",
      "--checkpoint",
      str(checkpoint),
      "--manifest",
      str(MANIFEST),
      "--repo-root",
      str(REPO_ROOT),
      "--asset-audit",
      str(audit),
      "--output-dir",
      str(tmp_path / "bundle"),
    ],
  )

  captured = capsys.readouterr()
  assert code == 1
  # The cohort seam needs one audit per manifest member; the singular audit
  # must never silently reduce the cohort to one teacher.
  assert (
    "cohort audit index with --asset-audits" in captured.err
    or "pass the cohort audit index with --asset-audits" in captured.err
  )
  assert not (tmp_path / "bundle").exists()


# --- playback of a pinned member ------------------------------------------------


def test_cohort_play_pins_a_member_and_refuses_a_non_member(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  schema,
) -> None:
  _install_fake_multi_adapter(monkeypatch, [], schema)
  code, _out, err = _train_cohort(
    tmp_path, monkeypatch, capsys, ["--max-iterations", "1"]
  )
  assert code == 0, err
  checkpoint = tmp_path / "run" / "checkpoint-final.pt"
  calls: list[dict] = []
  _install_fake_single_adapter(monkeypatch, calls, schema)
  _install_fake_viewers(monkeypatch)

  code = invoke(
    monkeypatch,
    [
      "distill",
      "play",
      "--manifest",
      str(MANIFEST),
      "--repo-root",
      str(REPO_ROOT),
      "--teacher-id",
      OTHER_TEACHER_ID,
      "--checkpoint",
      str(checkpoint),
      "--sampling-mode",
      "start",
      "--seed",
      "7",
    ],
  )

  captured = capsys.readouterr()
  assert code == 0, captured.err
  assert len(_FakePlayViewer.instances) == 1
  played = _FakePlayViewer.instances[0]
  assert isinstance(played.policy, DistillationPlayPolicy)
  assert played.action is not None and torch.isfinite(played.action).all()
  # The pinned member's saved schema drove the live packing.
  assert calls[0]["teacher_id"] == OTHER_TEACHER_ID
  assert calls[0]["schema"] is not None

  # A member that the stored cohort does not contain never builds an environment.
  calls.clear()
  _FakePlayViewer.instances = []
  code = invoke(
    monkeypatch,
    [
      "distill",
      "play",
      "--manifest",
      str(MANIFEST),
      "--repo-root",
      str(REPO_ROOT),
      "--teacher-id",
      "not-a-member",
      "--checkpoint",
      str(checkpoint),
      "--viewer",
      "native",
    ],
  )
  captured = capsys.readouterr()
  assert code == 1
  assert "not-a-member" in captured.err
  assert calls == []
  assert _FakePlayViewer.instances == []


def test_playback_hot_swap_dispatches_by_checkpoint_version(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  cohort: CohortContract,
  schema,
) -> None:
  """A directory holding a version-1 and a version-2 artifact both load."""
  directory = tmp_path / "run"
  version_one = directory / "checkpoint-iter-000003.pt"
  _write_version_one_checkpoint(version_one, cohort, schema, TEACHER_ID)
  _install_fake_multi_adapter(monkeypatch, [], schema)
  code, _out, err = _train_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    ["--max-iterations", "1", "--output-dir", str(directory)],
  )
  assert code == 0, err
  version_two = directory / "checkpoint-final.pt"
  assert distill._checkpoint_version(version_one) == 1
  assert distill._checkpoint_version(version_two) == 2

  adapter = _SingleAdapter(7, 1, schema)
  manager = distill._playback_checkpoint_manager(
    version_two, cohort, TEACHER_ID, adapter, "cpu"
  )

  assert [name for name, _ in manager.fetch_available()] == [
    "checkpoint-iter-000001.pt",
    "checkpoint-iter-000003.pt",
    "checkpoint-final.pt",
  ]
  assert isinstance(
    manager.load_checkpoint("checkpoint-final.pt"), DistillationPlayPolicy
  )
  assert isinstance(
    manager.load_checkpoint("checkpoint-iter-000003.pt"), DistillationPlayPolicy
  )

  # The same directory is refused for a teacher the saved cohort lacks.
  other = distill._playback_checkpoint_manager(
    version_two, cohort, "not-a-member", adapter, "cpu"
  )
  with pytest.raises(CheckpointValidationError):
    other.load_checkpoint("checkpoint-final.pt")
  # A version-1 artifact still holds its own teacher identity: it loads for its
  # own teacher and is refused for another one's hashes.
  foreign = distill._playback_checkpoint_manager(
    version_one, cohort, OTHER_TEACHER_ID, adapter, "cpu"
  )
  with pytest.raises(CheckpointValidationError):
    foreign.load_checkpoint("checkpoint-iter-000003.pt")


# Standing-profile reset provenance and trial accounting.
#
# These pin the two contract gaps an independent review found in the first
# implementation: a standing profile must state the perturbations it actually
# used, and its report must expose the trials' real initialization provenance
# instead of only aggregate metrics.


def _live_command() -> SimpleNamespace:
  """A live command stand-in carrying the real reset ranges."""
  return SimpleNamespace(
    cfg=SimpleNamespace(
      sampling_mode="start",
      pose_range={"x": (-0.025, 0.025), "yaw": (-0.1, 0.1)},
      velocity_range={"x": (-0.25, 0.25)},
      joint_position_range=(-0.05, 0.05),
    )
  )


def test_configured_reset_perturbations_are_reported_verbatim() -> None:
  live = _live_command()

  report = _reset_provenance_report(live, "configured")

  assert report["mode"] == "configured"
  assert report["clean"] is False
  assert report["pose_range"]["x"] == [-0.025, 0.025]
  assert report["joint_position_range"] == [-0.05, 0.05]
  # Sensor noise and domain randomization are explicitly out of scope.
  assert "reset initialization only" in report["scope"]


def test_clean_reset_perturbations_zero_every_range_and_report_clean() -> None:
  live = _live_command()

  _apply_reset_perturbations(live, "clean")
  report = _reset_provenance_report(live, "clean")

  assert report["clean"] is True
  assert report["pose_range"]["x"] == [0.0, 0.0]
  assert report["velocity_range"]["x"] == [0.0, 0.0]
  assert report["joint_position_range"] == [0.0, 0.0]
  assert live.cfg.sampling_mode == "start"
  # The keys survive zeroing, so the live contract keeps its shape.
  assert set(report["pose_range"]) == {"x", "yaw"}


def test_partially_zeroed_perturbations_are_not_reported_as_clean() -> None:
  live = _live_command()
  live.cfg.velocity_range = {"x": (-0.25, 0.25)}
  live.cfg.pose_range = {"x": (0.0, 0.0), "yaw": (0.0, 0.0)}
  live.cfg.joint_position_range = (0.0, 0.0)

  report = _reset_provenance_report(live, "clean")

  # One non-zero range is enough to refuse the clean claim.
  assert report["clean"] is False


def test_configured_perturbations_leave_the_command_unchanged() -> None:
  live = _live_command()
  before = dict(live.cfg.pose_range)

  _apply_reset_perturbations(live, "configured")

  assert live.cfg.pose_range == before
  assert live.cfg.joint_position_range == (-0.05, 0.05)


def _trial_segment(
  *,
  kind: int | None,
  frame: int | None,
  reason: str | None,
  steps: int,
  is_trial: bool,
  failed: bool,
) -> EvaluationSegment:
  return EvaluationSegment(
    env_index=0,
    segment_id=0,
    generation_id=0,
    steps=steps,
    completed=False,
    failed=failed,
    capped=False,
    metrics={},
    outcome="failure" if failed else "step_cap",
    motion_id=3,
    initialization_kind=kind,
    segment_initial_reference_frame=frame,
    is_trial=is_trial,
    start_reason=reason,
  )


def test_trial_provenance_reports_counts_for_trials_only() -> None:
  result = EvaluationResult(
    mode="teacher",
    rollout_latent="mean",
    steps=8,
    segments=(
      _trial_segment(
        kind=int(InitializationKind.STANDING),
        frame=0,
        reason="reset",
        steps=8,
        is_trial=True,
        failed=False,
      ),
      _trial_segment(
        kind=int(InitializationKind.STANDING),
        frame=3,
        reason="reset",
        steps=2,
        is_trial=True,
        failed=True,
      ),
      _trial_segment(
        kind=int(InitializationKind.REFERENCE),
        frame=0,
        reason="reference_completed",
        steps=5,
        is_trial=False,
        failed=False,
      ),
    ),
    metrics={},
    settings={},
    trial_window_steps=25,
  )

  summary = _trial_provenance_summary(result)

  assert summary is not None
  assert summary["trial_window_steps"] == 25
  entry = summary["per_motion"]["3"]
  assert entry["trials"] == 2
  assert entry["continuation_segments"] == 1
  assert entry["initialization_kind_counts"] == {"standing": 2}
  assert entry["initial_reference_frame_counts"] == {"0": 1, "3": 1}
  assert entry["start_reason_counts"] == {"reset": 2}
  # The raw per-segment arrays are not copied into the report.
  assert "segments" not in summary


def test_trial_provenance_is_absent_for_a_reference_profile() -> None:
  result = EvaluationResult(
    mode="teacher",
    rollout_latent="mean",
    steps=4,
    segments=(
      _trial_segment(
        kind=None,
        frame=None,
        reason=None,
        steps=4,
        is_trial=False,
        failed=False,
      ),
    ),
    metrics={},
    settings={},
  )

  assert _trial_provenance_summary(result) is None


def test_standing_perturbation_choice_requires_a_profile(
  monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  code = invoke(
    monkeypatch,
    ["distill", "evaluate", "--reset-perturbations", "clean"],
  )

  assert code == 1
  assert "--reset-perturbations applies only to a standing" in capsys.readouterr().err


def test_cohort_standing_perturbation_choice_requires_a_profile(
  monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  code = invoke(
    monkeypatch,
    [
      "distill",
      "evaluate-cohort",
      "--mode",
      "teacher",
      "--reset-perturbations",
      "clean",
    ],
  )

  assert code == 1
  assert "--reset-perturbations applies only to a standing" in capsys.readouterr().err


def test_rekey_rewrites_trial_provenance_to_the_cohort_motion_id() -> None:
  # A pinned environment always reports its own local clip 0.  The trial
  # provenance map is built from those local ids, so a member holding local id 0
  # would otherwise publish its trials under every other member's name.
  report = {
    "per_motion": [{"motion_id": 0, "teacher_code": 0, "segments": 2}],
    "per_motion_censoring": {"0": {"segments": 2}},
    "trial_provenance": {
      "trial_window_steps": 25,
      "per_motion": {
        "0": {
          "trials": 2,
          "initialization_kind_counts": {"standing": 2},
          "initial_reference_frame_counts": {"0": 2},
          "start_reason_counts": {"reset": 2},
          "continuation_segments": 1,
        }
      },
    },
  }

  rekeyed = distill._rekey_mode_report(report, motion_id=1, teacher_code=1)

  provenance = rekeyed["trial_provenance"]["per_motion"]
  assert list(provenance) == ["1"]
  assert provenance["1"]["trials"] == 2
  assert provenance["1"]["continuation_segments"] == 1
  assert rekeyed["per_motion"][0]["motion_id"] == 1


def test_rekey_leaves_a_reference_report_without_provenance_alone() -> None:
  report = {
    "per_motion": [{"motion_id": 0, "teacher_code": 0, "segments": 1}],
    "per_motion_censoring": {"0": {"segments": 1}},
    "trial_provenance": None,
  }

  rekeyed = distill._rekey_mode_report(report, motion_id=4, teacher_code=4)

  assert rekeyed["trial_provenance"] is None
  assert rekeyed["per_motion"][0]["motion_id"] == 4


def test_cohort_train_accepts_phase_bins_and_records_it(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  schema,
  motion_frames: dict[str, int],
) -> None:
  """--phase-bins reaches the live replay and the resolved configuration."""
  calls: list[dict] = []
  adapters: list[_MultiAdapter] = []
  _install_fake_multi_adapter(monkeypatch, calls, schema, adapters)
  output_dir = tmp_path / "run"

  code, out, err = _train_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    ["--max-iterations", "2", "--phase-bins", "10", "--output-dir", str(output_dir)],
  )

  assert code == 0, err
  report = json.loads(out)
  assert report["resolved_config"]["replay"]["phase_bins"] == 10
  # The per-phase telemetry is present in every iteration's replay report.
  replay_stats = report["replay"]["motions"]
  assert all("retained_by_phase_bin" in motion for motion in replay_stats)
  # The default run records the historical policy instead.
  code, out, err = _train_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    ["--max-iterations", "1", "--output-dir", str(tmp_path / "default")],
  )
  assert code == 0, err
  default_report = json.loads(out)
  assert default_report["resolved_config"]["replay"]["phase_bins"] == 0


def test_cohort_resume_default_phase_bins_zero_is_compatible(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  schema,
) -> None:
  """A checkpoint recorded before --phase-bins resumes with the default 0.

  The stored resolved configuration carries no ``phase_bins`` entry; the
  comparison must treat that as the documented default (density-proportional
  within-motion draws) instead of refusing the resume.
  """
  calls: list[dict] = []
  adapters: list[_MultiAdapter] = []
  _install_fake_multi_adapter(monkeypatch, calls, schema, adapters)
  first_dir = tmp_path / "first"
  code, out, err = _train_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    ["--max-iterations", "1", "--output-dir", str(first_dir)],
  )
  assert code == 0, err
  first = first_dir / "checkpoint-final.pt"

  # Rewrite the stored resolved config exactly as a pre-option checkpoint
  # would look: the replay entry simply predates the key.
  payload = torch.load(first, map_location="cpu", weights_only=False)
  stored = payload["resolved_config"]
  assert "phase_bins" in stored["replay"]
  del stored["replay"]["phase_bins"]
  torch.save(payload, first)

  second_dir = tmp_path / "second"
  code, out, err = _train_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    ["--max-iterations", "2", "--resume", str(first), "--output-dir", str(second_dir)],
  )
  assert code == 0, err
  report = json.loads(out)
  assert report["iteration"] == 2
  assert report["resume"]["mismatches"] == []
  assert report["resolved_config"]["replay"]["phase_bins"] == 0


def test_cohort_resume_refuses_a_changed_phase_bins(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  schema,
) -> None:
  """A different phase_bins value is a settings change, not a resume."""
  calls: list[dict] = []
  _install_fake_multi_adapter(monkeypatch, calls, schema, adapters=[])
  first_dir = tmp_path / "first"
  code, out, err = _train_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    [
      "--max-iterations",
      "1",
      "--phase-bins",
      "10",
      "--output-dir",
      str(first_dir),
    ],
  )
  assert code == 0, err
  first = first_dir / "checkpoint-final.pt"

  code, out, err = _train_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    [
      "--max-iterations",
      "2",
      "--phase-bins",
      "4",
      "--resume",
      str(first),
      "--output-dir",
      str(tmp_path / "refused"),
    ],
  )
  assert code == 1, (out, err)
  assert out == ""
  assert "replay" in err
  assert not (tmp_path / "refused" / "checkpoint-final.pt").exists()


def test_single_teacher_train_refuses_phase_bins(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  schema,
) -> None:
  """--phase-bins exists only on the cohort path; the M3 path refuses it.

  An explicit ``--phase-bins 0`` is still an explicit use of an option the
  single-teacher replay cannot honor, so it is refused the same way rather
  than being silently accepted as "the default".
  """
  _install_fake_single_adapter(monkeypatch, [], schema)
  for value in ("10", "0"):
    code = invoke(
      monkeypatch,
      [
        "distill",
        "train",
        "--manifest",
        str(MANIFEST),
        "--repo-root",
        str(REPO_ROOT),
        "--teacher-id",
        TEACHER_ID,
        *_TRAIN_ARGV,
        "--phase-bins",
        value,
        "--max-iterations",
        "1",
        "--output-dir",
        str(tmp_path / "refused"),
      ],
    )
    captured = capsys.readouterr()
    assert code == 1, (value, captured.err)
    assert captured.out == ""
    assert "cohort path" in captured.err
  assert not (tmp_path / "refused" / "checkpoint-final.pt").exists()


def test_cohort_train_records_single_device_workers(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  schema,
) -> None:
  """The default path records one implicit worker holding every environment.

  The multi-worker runtime keys are recorded even when the option is unset,
  because a resume compares that entry verbatim; the recorded values are the
  single-process equivalent, so a checkpoint written by this path stays
  resumable by this path, and the run report repeats them for visibility.
  """
  calls: list[dict] = []
  _install_fake_multi_adapter(monkeypatch, calls, schema, adapters=[])
  code, out, err = _train_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    ["--max-iterations", "1"],
  )
  assert code == 0, err
  report = json.loads(out)
  runtime = report["resolved_config"]["runtime"]
  assert runtime["num_envs"] == 2
  assert runtime["workers"] == {
    "mode": "single",
    "devices": [],
    "count": 1,
    "envs_per_worker": 2,
    "seed_scheme": 1,
  }
  # The runtime entry gains exactly the documented worker identity and nothing
  # else, so the default path's resolved configuration is otherwise unchanged.
  assert set(runtime) == {
    "device",
    "num_envs",
    "seed",
    "resolved_seed",
    "seed_provenance",
    "workers",
  }
  # The run report mirrors the identity a resume compares, transport included.
  assert report["runtime"]["workers"] == runtime["workers"]
  assert report["runtime"]["transport"] == "in-process"
  assert report["resolved_config"]["execution"]["transport"] == "in-process"


def test_cohort_worker_devices_are_refused_before_an_environment_exists(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  schema,
) -> None:
  """A malformed worker list is refused before any simulator is constructed.

  An empty list, a repeated device, a device string torch cannot parse, a
  repeat that differs only by an omitted CUDA index (``cuda`` against
  ``cuda:0``), a list that does not divide ``--num-envs`` evenly (two
  environments over three workers), and a list above the eight worker bound are
  all configuration errors rather than implicit fallbacks to a single device,
  and every one of them is decided without touching a simulator: the fake
  adapter records no call.  An uppercase or zero-padded alias is refused as an
  unparseable device, matching how the trainer device is validated.
  """
  calls: list[dict] = []
  adapters: list[_MultiAdapter] = []
  _install_fake_multi_adapter(monkeypatch, calls, schema, adapters)
  for value, expected in (
    ("()", "at least one device"),
    ("('cuda:1','cuda:1')", "repeats a device"),
    ("('cuda','cuda:0')", "repeats a device"),
    ("('not-a-device',)", "is not a device"),
    ("('CUDA:01',)", "is not a device"),
    ("('cuda:1','cuda:2','cuda:3')", "does not split evenly"),
    (
      "('cuda:1','cuda:2','cuda:3','cuda:4','cuda:5','cuda:6','cuda:7','cuda:8',"
      "'cuda:9')",
      "at most 8",
    ),
  ):
    code, out, err = _train_cohort(
      tmp_path,
      monkeypatch,
      capsys,
      [
        "--worker-devices",
        value,
        "--max-iterations",
        "1",
        "--output-dir",
        str(tmp_path / "refused"),
      ],
    )
    assert code == 1, (value, out, err)
    assert out == ""
    assert expected in err, (value, err)
  assert calls == []
  assert adapters == []
  assert not (tmp_path / "refused" / "checkpoint-final.pt").exists()


def test_cohort_multi_worker_is_refused_until_collection_is_sharded(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  schema,
) -> None:
  """A valid multi-worker list is refused instead of silently using one device.

  The list is well formed (two devices, two environments), so it passes
  validation; the collection pool is not wired into the runner yet, and
  accepting the flag would collect from one environment while the resolved
  configuration claimed two workers.  The refusal is therefore explicit and
  happens before the simulator exists.
  """
  calls: list[dict] = []
  _install_fake_multi_adapter(monkeypatch, calls, schema, adapters=[])
  code, out, err = _train_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    [
      "--worker-devices",
      "('cuda:1','cuda:2')",
      "--max-iterations",
      "1",
      "--output-dir",
      str(tmp_path / "refused"),
    ],
  )
  assert code == 1, (out, err)
  assert out == ""
  assert "not wired" in err
  assert calls == []
  assert not (tmp_path / "refused" / "checkpoint-final.pt").exists()


def test_single_teacher_train_refuses_worker_devices(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  schema,
) -> None:
  """--worker-devices exists only on the cohort path; the M3 path refuses it.

  The refusal is decided from the selection alone, so a well formed list is
  refused with the cohort-path diagnostic rather than being validated as if
  the single-teacher path could shard it, and the observable adapter call list
  proves no environment was constructed before the refusal.
  """
  calls: list[dict] = []
  _install_fake_single_adapter(monkeypatch, calls, schema)
  code = invoke(
    monkeypatch,
    [
      "distill",
      "train",
      "--manifest",
      str(MANIFEST),
      "--repo-root",
      str(REPO_ROOT),
      "--teacher-id",
      TEACHER_ID,
      *_TRAIN_ARGV,
      "--worker-devices",
      "('cuda:1','cuda:2')",
      "--max-iterations",
      "1",
      "--output-dir",
      str(tmp_path / "refused"),
    ],
  )
  captured = capsys.readouterr()
  assert code == 1, captured.err
  assert captured.out == ""
  assert "cohort path" in captured.err
  assert calls == []
  assert not (tmp_path / "refused" / "checkpoint-final.pt").exists()


def test_cohort_resume_legacy_runtime_without_worker_keys_is_compatible(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  schema,
) -> None:
  """A checkpoint recorded before multi-worker collection resumes unchanged.

  The stored runtime entry carries no worker keys; the comparison must treat
  that as the documented single-process equivalent (no worker devices, every
  environment in the one environment, seed scheme 1) instead of refusing a
  resume of a run that never had workers.
  """
  calls: list[dict] = []
  _install_fake_multi_adapter(monkeypatch, calls, schema, adapters=[])
  first_dir = tmp_path / "first"
  code, out, err = _train_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    ["--max-iterations", "1", "--output-dir", str(first_dir)],
  )
  assert code == 0, err
  first = first_dir / "checkpoint-final.pt"

  payload = torch.load(first, map_location="cpu", weights_only=False)
  stored = payload["resolved_config"]["runtime"]
  assert stored["workers"]["mode"] == "single"
  del stored["workers"]
  torch.save(payload, first)

  second_dir = tmp_path / "second"
  code, out, err = _train_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    ["--max-iterations", "2", "--resume", str(first), "--output-dir", str(second_dir)],
  )
  assert code == 0, err
  report = json.loads(out)
  assert report["iteration"] == 2
  assert report["resume"]["mismatches"] == []
  assert report["resolved_config"]["runtime"]["workers"]["mode"] == "single"


def test_cohort_resume_refuses_a_changed_worker_devices(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  schema,
) -> None:
  """A checkpoint trained with sharded workers is not resumable unsharded.

  Worker devices, per-worker environment counts and the seed scheme are all
  recorded in the compared runtime entry, so a single-process invocation
  cannot silently continue a run whose rows came from two environments.
  """
  calls: list[dict] = []
  _install_fake_multi_adapter(monkeypatch, calls, schema, adapters=[])
  first_dir = tmp_path / "first"
  code, out, err = _train_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    ["--max-iterations", "1", "--output-dir", str(first_dir)],
  )
  assert code == 0, err
  first = first_dir / "checkpoint-final.pt"

  payload = torch.load(first, map_location="cpu", weights_only=False)
  stored = payload["resolved_config"]["runtime"]
  stored["workers"] = {
    "mode": "sharded",
    "devices": ["cuda:1", "cuda:2"],
    "count": 2,
    "envs_per_worker": 1,
    "seed_scheme": 1,
  }
  torch.save(payload, first)

  code, out, err = _train_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    [
      "--max-iterations",
      "2",
      "--resume",
      str(first),
      "--output-dir",
      str(tmp_path / "refused"),
    ],
  )
  assert code == 1, (out, err)
  assert out == ""
  assert "runtime" in err
  assert not (tmp_path / "refused" / "checkpoint-final.pt").exists()


def test_cohort_transport_is_recorded_but_not_a_resume_invariant(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  schema,
) -> None:
  """Changing only the transport name must not refuse a resume.

  The transport does not change which rows are collected, so it is recorded for
  audit next to the worker identity but deliberately excluded from the compared
  entries -- the distinction ``checkpoint_every`` already has.  A stored
  transport the current build does not use is therefore resumed, and the
  resumed run reports its own transport rather than the stored audit value.
  """
  calls: list[dict] = []
  _install_fake_multi_adapter(monkeypatch, calls, schema, adapters=[])
  first_dir = tmp_path / "first"
  code, out, err = _train_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    ["--max-iterations", "1", "--output-dir", str(first_dir)],
  )
  assert code == 0, err
  first = first_dir / "checkpoint-final.pt"

  payload = torch.load(first, map_location="cpu", weights_only=False)
  assert payload["resolved_config"]["execution"]["transport"] == "in-process"
  payload["resolved_config"]["execution"]["transport"] = "pinned-host-v9"
  torch.save(payload, first)

  code, out, err = _train_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    [
      "--max-iterations",
      "2",
      "--resume",
      str(first),
      "--output-dir",
      str(tmp_path / "second"),
    ],
  )
  assert code == 0, err
  report = json.loads(out)
  assert report["iteration"] == 2
  assert report["resume"]["mismatches"] == []
  assert report["runtime"]["transport"] == "in-process"


def test_single_explicit_worker_device_is_refused(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  schema,
) -> None:
  """An explicit one-device worker list is refused, not silently ignored.

  A singleton list is not "multi-worker" by length, but it still asks for a
  sharded collection on a named device while the environment is built on the
  trainer device.  Accepting it records a worker identity no process honors, so
  every explicit list is refused until the collection pool exists.
  """
  calls: list[dict] = []
  adapters: list[_MultiAdapter] = []
  _install_fake_multi_adapter(monkeypatch, calls, schema, adapters)
  code, out, err = _train_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    [
      "--worker-devices",
      "('cuda:1',)",
      "--max-iterations",
      "1",
      "--output-dir",
      str(tmp_path / "refused"),
    ],
  )
  assert code == 1, (out, err)
  assert out == ""
  assert "not wired" in err
  assert calls == []
  assert adapters == []
  assert not (tmp_path / "refused" / "checkpoint-final.pt").exists()
