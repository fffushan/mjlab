"""The ``distill`` CLI: dispatch, machine-readable report, and exit codes."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tracking_distillation_fixtures import (
  DEFAULT_TEACHER_IDS,
  build_tiny_cohort,
  write_manifest,
)

from mjlab.scripts.distill import PROVENANCE_VERSION, _control_metadata, main
from mjlab.tasks.tracking.distillation.adapter import (
  DistillationSnapshot,
  DistillationStep,
)
from mjlab.tasks.tracking.distillation.checkpoint import save_checkpoint
from mjlab.tasks.tracking.distillation.collector import DAggerCollector
from mjlab.tasks.tracking.distillation.config import load_manifest, resolve_cohort
from mjlab.tasks.tracking.distillation.environment import (
  RuntimeSeedProvenance,
  SegmentMotionCommandCfg,
  build_distillation_environment,
)
from mjlab.tasks.tracking.distillation.model import ConditionalVAE
from mjlab.tasks.tracking.distillation.observations import (
  ObservationSnapshot,
  PackedObservationBatch,
  pack_observations,
)
from mjlab.tasks.tracking.distillation.runner import DistillationRunner
from mjlab.tasks.tracking.distillation.storage import (
  LabeledReplayBatch,
  LabeledReplayBuffer,
)
from mjlab.tasks.tracking.distillation.trainer import (
  FreshTrainingData,
  TrainerPoisonedError,
  VaeDistillationTrainer,
)
from mjlab.tasks.tracking.distillation.training_config import TrainingConfig
from mjlab.tasks.tracking.distillation.vae_config import (
  DEFAULT_MODEL_SETTINGS,
  DEFAULT_SCHEMA,
  ModelSettings,
)
from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg

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
  """Resolve the generated tiny cohort (real artifacts, no simulator)."""
  return resolve_cohort(load_manifest(build_tiny_cohort(tmp_path).manifest, tmp_path))


def _snapshot(step: int, batch: int = 2) -> DistillationSnapshot:
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
    packed=pack_observations(features),
    teacher_id=TEACHER_ID,
    teacher_code=0,
    motion_id=torch.zeros(batch, dtype=torch.int64),
    reference_frame=torch.full((batch,), step, dtype=torch.int64),
    segment_id=torch.zeros(batch, dtype=torch.int64),
    generation_id=torch.zeros(batch, dtype=torch.int64),
  )


class _Teacher(torch.nn.Module):
  """Minimal label source standing in for the frozen M1 teacher."""

  action_dim = 31

  def label(self, observations: torch.Tensor) -> torch.Tensor:
    return torch.ones(observations.shape[0], self.action_dim)


class _FakeAdapter:
  """Stand-in for the trusted native adapter; constructs no simulated scene."""

  auto_reset = True

  def __init__(self, seed: int | None) -> None:
    self.schema = DEFAULT_SCHEMA
    self.teacher = _Teacher()
    self.reset_calls = 0
    self.step_calls = 0
    self.closed = False
    self.current = _snapshot(0)
    self.audit = SimpleNamespace(
      seed_provenance=RuntimeSeedProvenance(
        requested_seed=seed,
        effective_seed=seed,
        applied_before_construction=seed is not None,
      )
    )
    self.motion = SimpleNamespace(cfg=SimpleNamespace(sampling_mode="weighted"))
    self.env = SimpleNamespace(
      command_manager=SimpleNamespace(get_term=lambda name: self.motion)
    )

  def reset(self, seed: int | None = None) -> DistillationSnapshot:
    del seed
    self.reset_calls += 1
    self.current = _snapshot(0)
    return self.current

  def step(self, action: torch.Tensor) -> DistillationStep:
    self.step_calls += 1
    self.current = _snapshot(self.step_calls)
    batch = self.current.packed.batch_size
    return DistillationStep(
      self.current,
      torch.zeros(batch),
      torch.zeros(batch, dtype=torch.bool),
      torch.zeros(batch, dtype=torch.bool),
      {},
    )

  def close(self) -> None:
    self.closed = True


def _install_fake_adapter(
  monkeypatch: pytest.MonkeyPatch,
  calls: list[dict],
  adapters: list | None = None,
) -> None:
  """Mock only the simulator/adapter boundary, recording every construction."""

  def factory(cohort, teacher_id: str = TEACHER_ID, **kwargs):
    calls.append({"teacher_id": teacher_id, **kwargs})
    adapter = _FakeAdapter(kwargs.get("seed"))
    if adapters is not None:
      adapters.append(adapter)
    return adapter

  monkeypatch.setattr("mjlab.scripts.distill.make_distillation_adapter", factory)


def _batch(size: int = 8, seed: int = 2) -> LabeledReplayBatch:
  generator = torch.Generator().manual_seed(seed)
  ids = torch.arange(size, dtype=torch.int64)
  return LabeledReplayBatch(
    PackedObservationBatch(
      torch.randn(size, 68, generator=generator),
      torch.randn(size, 99, generator=generator),
      DEFAULT_SCHEMA,
    ),
    torch.randn(size, 31, generator=generator),
    ids,
    ids + 10,
    ids + 20,
    ids + 30,
    ids + 40,
  )


def test_cli_prints_report_and_exits_zero(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  cohort = build_tiny_cohort(tmp_path)
  report_path = tmp_path / "report.json"

  code = invoke(
    monkeypatch,
    [
      "distill",
      "validate-teachers",
      "--manifest",
      str(cohort.manifest),
      "--repo-root",
      str(tmp_path),
      "--samples",
      "4",
      "--report",
      str(report_path),
    ],
  )

  captured = capsys.readouterr()
  report = json.loads(captured.out)
  assert code == 0
  assert report["passed"] is True
  assert report["command"].startswith("distill validate-teachers")
  assert report["teachers"][0]["parity"]["samples"] == 4
  assert json.loads(report_path.read_text()) == report
  assert "[OK]" in captured.err


def test_cli_resolves_manifest_against_repo_root_from_another_cwd(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  build_tiny_cohort(tmp_path)
  elsewhere = tmp_path / "elsewhere"
  elsewhere.mkdir()
  monkeypatch.chdir(elsewhere)

  code = invoke(
    monkeypatch,
    [
      "distill",
      "validate-teachers",
      "--manifest",
      "configs/tiny_teachers.yaml",
      "--repo-root",
      str(tmp_path),
      "--samples",
      "4",
    ],
  )

  captured = capsys.readouterr()
  assert code == 0
  assert json.loads(captured.out)["passed"] is True


def test_cli_exits_nonzero_when_the_parity_gate_fails(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  cohort = build_tiny_cohort(tmp_path)
  swapped = [
    replace(cohort.teachers[0], onnx=cohort.teachers[1].onnx),
    cohort.teachers[1],
  ]
  manifest = write_manifest(
    tmp_path / "configs" / "swapped.yaml", swapped, root=tmp_path
  )

  code = invoke(
    monkeypatch,
    [
      "distill",
      "validate-teachers",
      "--manifest",
      str(manifest),
      "--repo-root",
      str(tmp_path),
      "--samples",
      "4",
    ],
  )

  captured = capsys.readouterr()
  assert code == 1
  assert json.loads(captured.out)["passed"] is False
  assert "[FAIL]" in captured.err


def test_cli_reports_missing_artifacts_actionably(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  cohort = build_tiny_cohort(tmp_path)
  manifest = tmp_path / "configs" / "broken.yaml"
  manifest.write_text(
    cohort.manifest.read_text().replace(
      "runs/tiny_001/model_5.pt", "runs/tiny_001/model_absent.pt"
    )
  )

  code = invoke(
    monkeypatch,
    [
      "distill",
      "validate-teachers",
      "--manifest",
      str(manifest),
      "--repo-root",
      str(tmp_path),
    ],
  )

  captured = capsys.readouterr()
  assert code == 1
  assert captured.out == ""
  assert "[FAIL]" in captured.err
  assert "model_absent.pt" in captured.err


def test_cli_rejects_unknown_commands(
  monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  monkeypatch.setattr(sys, "argv", ["distill", "unknown-command"])

  with pytest.raises(SystemExit) as exit_info:
    main()

  captured = capsys.readouterr()
  assert exit_info.value.code == 2
  assert "unknown command" in captured.err
  assert "validate-teachers" in captured.err


def test_cli_exposes_bounded_m3_commands(
  monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  for command, expected in (
    ("train", "--max-iterations"),
    ("train", "--checkpoint-every"),
    ("evaluate", "--sampling-mode"),
  ):
    code = invoke(monkeypatch, ["distill", command, "--help"])
    captured = capsys.readouterr()
    assert code == 0
    assert expected in captured.out


def test_cli_prints_help_without_a_command(
  monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  monkeypatch.setattr(sys, "argv", ["distill", "--help"])

  main()

  captured = capsys.readouterr()
  assert "usage: distill <COMMAND> [OPTIONS]" in captured.out


def test_train_cli_applies_seed_before_construction_and_saves_full_provenance(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  """The train path forwards ``--seed`` and stores the resolved provenance.

  The simulator/adapter boundary is mocked; the model, replay, trainer, runner,
  checkpoint, and report are the real objects.
  """
  resolved = _resolved_cohort(tmp_path)
  calls: list[dict] = []
  _install_fake_adapter(monkeypatch, calls)
  output_dir = tmp_path / "run"

  code = invoke(
    monkeypatch,
    [
      "distill",
      "train",
      "--manifest",
      str(tmp_path / "configs" / "tiny_teachers.yaml"),
      "--repo-root",
      str(tmp_path),
      "--teacher-id",
      TEACHER_ID,
      "--num-envs",
      "2",
      "--max-iterations",
      "2",
      "--bootstrap-steps",
      "1",
      "--collection-steps",
      "1",
      "--minibatch-size",
      "4",
      "--accumulation-steps",
      "3",
      "--learning-rate",
      "0.001",
      "--beta",
      "0.3",
      "--replay-capacity",
      "24",
      "--seed",
      "7",
      "--output-dir",
      str(output_dir),
    ],
  )

  captured = capsys.readouterr()
  assert code == 0, captured.err
  report = json.loads(captured.out)
  # The requested seed reaches the factory that seeds the private env config
  # before startup randomization, rather than a post-construction seed call.
  assert calls[0]["seed"] == 7
  assert calls[0]["teacher_id"] == TEACHER_ID
  assert calls[0]["device"] == "cpu"
  assert calls[0]["num_envs"] == 2
  assert calls[0]["task_id"] is None
  assert report["runtime"] == {
    "device": "cpu",
    "num_envs": 2,
    "requested_seed": 7,
    "resolved_seed": 7,
    "seed_provenance": {
      "available": True,
      "requested_seed": 7,
      "effective_seed": 7,
      "applied_before_construction": True,
    },
  }
  assert report["iteration"] == 2

  payload = torch.load(output_dir / "checkpoint-final.pt", weights_only=True)
  provenance = payload["resolved_config"]
  assert set(provenance) == {
    "provenance_version",
    "command",
    "manifest",
    "repo_root",
    "teacher_id",
    "teacher_hashes",
    "motion",
    "task",
    "task_id",
    "control_contract",
    "runtime",
    "schedule",
    "checkpoint_every",
    "trainer",
    "replay",
    "model",
    "resumed_from",
  }
  assert provenance["provenance_version"] == PROVENANCE_VERSION
  assert provenance["command"] == "train"
  assert provenance["teacher_id"] == TEACHER_ID
  assert provenance["teacher_hashes"] == resolved.teacher(TEACHER_ID).hashes
  assert provenance["control_contract"] == _control_metadata(resolved, TEACHER_ID)
  assert provenance["runtime"]["seed"] == 7
  assert provenance["runtime"]["resolved_seed"] == 7
  assert provenance["schedule"] == {
    "max_iterations": 2,
    "bootstrap_steps": 1,
    "collection_steps": 1,
    "updates_per_iteration": 1,
    "teacher_probability": 0.0,
    "evaluate_every": 0,
    "evaluation_steps": 0,
    "evaluation_mode": "student",
    "evaluation_sampling_mode": "weighted",
    "rollout_latent": "mean",
  }
  assert provenance["trainer"] == {
    "learning_rate": 0.001,
    "beta": 0.3,
    "accumulation_steps": 3,
    "minibatch_size": 4,
    "latent_mode": "sampled",
  }
  assert provenance["replay"] == {"capacity": 24}
  assert provenance["model"] == {
    "settings": DEFAULT_MODEL_SETTINGS.to_metadata(),
    "schema": DEFAULT_SCHEMA.compatibility_metadata(),
  }
  assert provenance["checkpoint_every"] == 0
  assert provenance["resumed_from"] is None
  assert payload["schedule"] == provenance["schedule"]
  assert report["schedule"] == provenance["schedule"]
  assert report["resolved_config"] == provenance
  assert report["resume"] is None
  assert report["checkpoints"] == [str(output_dir / "checkpoint-final.pt")]
  assert (output_dir / "train-report.json").exists()

  # The checkpoint this command just wrote is evaluable without repeating any
  # trainer-only setting: student evaluation loads the model only.
  code = invoke(
    monkeypatch,
    [
      "distill",
      "evaluate",
      "--manifest",
      str(tmp_path / "configs" / "tiny_teachers.yaml"),
      "--repo-root",
      str(tmp_path),
      "--teacher-id",
      TEACHER_ID,
      "--num-envs",
      "2",
      "--mode",
      "student",
      "--checkpoint",
      str(output_dir / "checkpoint-final.pt"),
      "--steps",
      "2",
      "--seed",
      "7",
    ],
  )
  captured = capsys.readouterr()
  assert code == 0, captured.err
  evaluation = json.loads(captured.out)
  assert calls[1]["seed"] == 7
  trained_settings = DEFAULT_MODEL_SETTINGS.to_metadata()
  assert evaluation["checkpoint_model"]["settings"] == trained_settings
  assert evaluation["checkpoint_resolved_config"] == provenance
  assert evaluation["result"]["steps"] == 2


def test_train_cli_resume_extends_budget_and_refuses_changed_settings(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  """Only the documented total-budget extension may differ on resume."""
  _resolved_cohort(tmp_path)
  calls: list[dict] = []
  _install_fake_adapter(monkeypatch, calls)

  def train(argv: list[str]) -> tuple[str | int | None, str, str]:
    code = invoke(
      monkeypatch,
      [
        "distill",
        "train",
        "--manifest",
        str(tmp_path / "configs" / "tiny_teachers.yaml"),
        "--repo-root",
        str(tmp_path),
        "--teacher-id",
        TEACHER_ID,
        "--num-envs",
        "2",
        "--bootstrap-steps",
        "1",
        "--collection-steps",
        "1",
        "--replay-capacity",
        "24",
        "--seed",
        "7",
        *argv,
      ],
    )
    captured = capsys.readouterr()
    return code, captured.out, captured.err

  first_dir = tmp_path / "first"
  code, out, err = train(["--max-iterations", "1", "--output-dir", str(first_dir)])
  assert code == 0, err
  first = first_dir / "checkpoint-final.pt"

  # The explicitly requested total-budget extension is allowed and audited.
  second_dir = tmp_path / "second"
  code, out, err = train(
    [
      "--max-iterations",
      "3",
      "--resume",
      str(first),
      "--output-dir",
      str(second_dir),
    ]
  )
  assert code == 0, err
  report = json.loads(out)
  assert report["iteration"] == 3
  assert report["resume"]["provenance_verified"] is True
  assert report["resume"]["mismatches"] == []
  assert report["resume"]["budget"] == {
    "stored_max_iterations": 1,
    "requested_max_iterations": 3,
    "extended": True,
  }
  assert report["resolved_config"]["resumed_from"] == str(first)
  second = second_dir / "checkpoint-final.pt"

  # A changed schedule is refused instead of silently altering stored semantics.
  for argv in (
    ["--max-iterations", "2", "--resume", str(second)],
    ["--max-iterations", "3", "--collection-steps", "2", "--resume", str(second)],
  ):
    code, out, err = train([*argv, "--output-dir", str(tmp_path / "refused")])
    assert code == 1, (argv, out, err)
    assert out == ""
    assert "resume refused" in err
    assert "schedule.max_iterations" in err or "schedule.collection_steps" in err
  assert not (tmp_path / "refused" / "checkpoint-final.pt").exists()

  # An unverifiable legacy record is reported, not silently compared.
  payload = torch.load(first, weights_only=True)
  payload["resolved_config"] = {"device": "cpu"}
  legacy = tmp_path / "legacy.pt"
  torch.save(payload, legacy)
  code, out, err = train(
    [
      "--max-iterations",
      "3",
      "--resume",
      str(legacy),
      "--output-dir",
      str(tmp_path / "legacy-run"),
    ]
  )
  assert code == 0, err
  legacy_report = json.loads(out)
  assert legacy_report["resume"]["provenance_verified"] is False
  assert {"teacher_id", "runtime", "model", "provenance_version"} <= set(
    legacy_report["resume"]["unverified"]
  )
  assert legacy_report["resume"]["mismatches"] == []

  # Entries a legacy record *did* store are still compared, not ignored.
  payload["resolved_config"] = {
    "device": "cpu",
    "trainer": {
      "learning_rate": 5e-4,
      "beta": 0.01,
      "accumulation_steps": 15,
      "minibatch_size": 4,
      "latent_mode": "sampled",
    },
  }
  changed = tmp_path / "legacy-changed.pt"
  torch.save(payload, changed)
  code, out, err = train(
    [
      "--max-iterations",
      "3",
      "--resume",
      str(changed),
      "--output-dir",
      str(tmp_path / "legacy-changed-run"),
    ]
  )
  assert code == 1, (out, err)
  assert "resume refused" in err
  assert "trainer" in err


def _train_with_tiny_cohort(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  extra: list[str],
) -> tuple[str | int | None, str, str]:
  """Invoke ``distill train`` on the tiny cohort with fixed stored semantics.

  Only the settings under test (budget, save cadence, output directory, resume)
  are passed through ``extra``, so a resumed run repeats the stored semantics.
  """
  code = invoke(
    monkeypatch,
    [
      "distill",
      "train",
      "--manifest",
      str(tmp_path / "configs" / "tiny_teachers.yaml"),
      "--repo-root",
      str(tmp_path),
      "--teacher-id",
      TEACHER_ID,
      "--num-envs",
      "2",
      "--bootstrap-steps",
      "1",
      "--collection-steps",
      "1",
      "--replay-capacity",
      "24",
      "--seed",
      "7",
      *extra,
    ],
  )
  captured = capsys.readouterr()
  return code, captured.out, captured.err


def test_train_cli_writes_periodic_checkpoints_and_resumes_from_intermediate(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  """Periodic checkpoints are lifetime-indexed, complete, and resumable.

  The cadence is recorded in ``resolved_config`` but is deliberately not a
  resume invariant, so a resumed run may change ``--checkpoint-every`` and
  continues from the checkpoint's lifetime iteration instead of restarting at
  iteration zero.  The simulator/adapter boundary is mocked; the model, replay,
  trainer, runner, checkpoint, and report are the real objects.
  """
  _resolved_cohort(tmp_path)
  calls: list[dict] = []
  _install_fake_adapter(monkeypatch, calls)

  first_dir = tmp_path / "periodic"
  code, out, err = _train_with_tiny_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    [
      "--max-iterations",
      "6",
      "--checkpoint-every",
      "2",
      "--output-dir",
      str(first_dir),
    ],
  )

  assert code == 0, err
  report = json.loads(out)
  assert report["iteration"] == 6
  assert report["resolved_config"]["checkpoint_every"] == 2
  periodic = [first_dir / f"checkpoint-iter-{step:06d}.pt" for step in (2, 4, 6)]
  final = first_dir / "checkpoint-final.pt"
  # Every checkpoint of this invocation is listed, with the final one included.
  assert report["checkpoints"] == [str(path) for path in [*periodic, final]]
  expected_provenance = torch.load(final, weights_only=True)["resolved_config"]
  for path, step in zip(periodic, (2, 4, 6), strict=True):
    payload = torch.load(path, weights_only=True)
    assert payload["counters"]["iteration"] == step
    # A periodic checkpoint is the same save with the same metadata: the
    # provenance record and schedule match the final checkpoint exactly.
    assert payload["resolved_config"] == expected_provenance
    assert payload["schedule"] == report["schedule"]

  # Resume from the iteration-2 checkpoint while *changing* the cadence.  The
  # filename sequence continues the lifetime counter (5, 6) instead of
  # restarting and colliding with the first run's files.
  second_dir = tmp_path / "resumed"
  code, out, err = _train_with_tiny_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    [
      "--max-iterations",
      "6",
      "--checkpoint-every",
      "3",
      "--resume",
      str(periodic[0]),
      "--output-dir",
      str(second_dir),
    ],
  )

  assert code == 0, err
  resumed = json.loads(out)
  assert resumed["iteration"] == 6
  assert resumed["checkpoints"] == [
    str(second_dir / "checkpoint-iter-000005.pt"),
    str(second_dir / "checkpoint-iter-000006.pt"),
    str(second_dir / "checkpoint-final.pt"),
  ]
  # No cadence-relative restarts: nothing is written at 3 (2 + 1 chunk).
  assert not (second_dir / "checkpoint-iter-000003.pt").exists()
  assert resumed["resume"]["provenance_verified"] is True
  assert resumed["resume"]["mismatches"] == []
  assert resumed["resume"]["budget"] == {
    "stored_max_iterations": 6,
    "requested_max_iterations": 6,
    "extended": False,
  }
  # The run continued from the checkpoint (2) and stopped at the budget (6).
  assert [item["iteration"] for item in resumed["iterations"]] == [2, 3, 4, 5]
  assert resumed["resolved_config"]["checkpoint_every"] == 3


def test_train_cli_default_checkpoint_every_writes_only_the_final_checkpoint(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  """``--checkpoint-every 0`` is the default and keeps the original one save."""
  _resolved_cohort(tmp_path)
  calls: list[dict] = []
  _install_fake_adapter(monkeypatch, calls)
  output_dir = tmp_path / "single"

  code, out, err = _train_with_tiny_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    ["--max-iterations", "4", "--output-dir", str(output_dir)],
  )

  assert code == 0, err
  report = json.loads(out)
  assert report["iteration"] == 4
  assert report["resolved_config"]["checkpoint_every"] == 0
  assert report["checkpoints"] == [str(output_dir / "checkpoint-final.pt")]
  assert list(output_dir.glob("checkpoint-iter-*.pt")) == []


def test_train_cli_rejects_negative_checkpoint_every_before_construction(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  """A negative cadence is refused actionably, before any adapter is built."""
  _resolved_cohort(tmp_path)
  calls: list[dict] = []
  _install_fake_adapter(monkeypatch, calls)
  output_dir = tmp_path / "invalid"

  code, out, err = _train_with_tiny_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    [
      "--max-iterations",
      "1",
      "--checkpoint-every=-1",
      "--output-dir",
      str(output_dir),
    ],
  )

  assert code == 1
  assert out == ""
  assert "--checkpoint-every must be a non-negative integer" in err
  assert "0 to disable periodic checkpoints" in err
  assert calls == []
  assert not output_dir.exists()


def test_train_cli_surfaces_periodic_save_failures(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  """A refused periodic save aborts the run instead of being swallowed.

  ``DistillationRunner.save`` is where the trainer poison gate and the atomic
  write live, so injecting its refusal is how a checkpoint-level save failure
  reaches the CLI.  The failure must propagate as a non-zero exit with no final
  checkpoint and no report, not be logged and skipped.
  """
  _resolved_cohort(tmp_path)
  calls: list[dict] = []
  _install_fake_adapter(monkeypatch, calls)
  output_dir = tmp_path / "failure"
  real_save = DistillationRunner.save

  def refusing_save(self, path, **kwargs):
    if "checkpoint-iter-" in str(path):
      raise TrainerPoisonedError("trainer is poisoned and requires a validated restore")
    return real_save(self, path, **kwargs)

  monkeypatch.setattr(DistillationRunner, "save", refusing_save)

  code, out, err = _train_with_tiny_cohort(
    tmp_path,
    monkeypatch,
    capsys,
    [
      "--max-iterations",
      "4",
      "--checkpoint-every",
      "2",
      "--output-dir",
      str(output_dir),
    ],
  )

  assert code == 1
  assert out == ""
  assert "[FAIL]" in err
  assert "poisoned" in err
  assert not (output_dir / "checkpoint-final.pt").exists()
  assert list(output_dir.glob("checkpoint-iter-*.pt")) == []
  assert not (output_dir / "train-report.json").exists()


def test_evaluate_cli_loads_train_produced_checkpoint_without_training_flags(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  """A real train-shaped checkpoint evaluates with no unrelated train flags."""
  resolved = _resolved_cohort(tmp_path)
  torch.manual_seed(5)
  model = ConditionalVAE(DEFAULT_SCHEMA, ModelSettings(hidden_dims=(12, 6)))
  replay = LabeledReplayBuffer(24, DEFAULT_SCHEMA)
  replay.insert(_batch())
  trainer = VaeDistillationTrainer(
    model,
    replay,
    TrainingConfig(
      accumulation_steps=3, minibatch_size=4, learning_rate=1e-3, beta=0.3
    ),
    seed=9,
  )
  trainer.begin_training(FreshTrainingData(_batch(), "initial"))
  trainer.train_update()
  collector = DAggerCollector(_FakeAdapter(9), _Teacher(), model, replay)
  resolved_config = {
    "provenance_version": PROVENANCE_VERSION,
    "command": "train",
    "teacher_id": TEACHER_ID,
    "runtime": {
      "device": "cpu",
      "num_envs": 2,
      "seed": 9,
      "resolved_seed": 9,
      "seed_provenance": {
        "available": True,
        "requested_seed": 9,
        "effective_seed": 9,
        "applied_before_construction": True,
      },
    },
    "trainer": {
      "learning_rate": 1e-3,
      "beta": 0.3,
      "accumulation_steps": 3,
      "minibatch_size": 4,
      "latent_mode": "sampled",
    },
    "replay": {"capacity": 24},
    "model": {
      "settings": model.settings.to_metadata(),
      "schema": model.schema_metadata,
    },
  }
  checkpoint = tmp_path / "train-shaped.pt"
  save_checkpoint(
    checkpoint,
    trainer,
    replay,
    counters={"iteration": 5, "segment_namespace": 0},
    schedule={"max_iterations": 5},
    resolved_config=resolved_config,
    teacher_hashes=resolved.teacher(TEACHER_ID).hashes,
    control_contract=_control_metadata(resolved, TEACHER_ID),
    collector=collector,
  )
  payload = torch.load(checkpoint, weights_only=True)
  assert payload["trainer_config"]["minibatch_size"] == 4
  assert payload["optimizer"]["state"]
  assert payload["replay"]["storage"] is not None
  assert set(payload["rng"]) == {"global_cpu", "trainer", "collector"}

  calls: list[dict] = []
  adapters: list = []
  _install_fake_adapter(monkeypatch, calls, adapters)
  report_path = tmp_path / "evaluate.json"
  code = invoke(
    monkeypatch,
    [
      "distill",
      "evaluate",
      "--manifest",
      str(tmp_path / "configs" / "tiny_teachers.yaml"),
      "--repo-root",
      str(tmp_path),
      "--teacher-id",
      TEACHER_ID,
      "--num-envs",
      "2",
      "--mode",
      "student",
      "--checkpoint",
      str(checkpoint),
      "--steps",
      "2",
      "--seed",
      "11",
      "--sampling-mode",
      "uniform",
      "--report",
      str(report_path),
    ],
  )

  captured = capsys.readouterr()
  assert code == 0, captured.err
  report = json.loads(captured.out)
  assert calls[0]["seed"] == 11
  assert report["runtime"]["resolved_seed"] == 11
  assert report["runtime"]["seed_provenance"]["applied_before_construction"] is True
  assert report["mode"] == "student"
  assert report["sampling_mode"] == "uniform"
  # The evaluation-only sampling override is applied to the live command.
  live = adapters[0].env.command_manager.get_term("motion")
  assert live.cfg.sampling_mode == "uniform"
  # Settings come from the checkpoint, not from DEFAULT_MODEL_SETTINGS.
  assert report["checkpoint_model"]["settings"] == {
    "latent_dim": 32,
    "hidden_dims": [12, 6],
    "activation": "ELU",
    "beta": 0.01,
  }
  assert report["checkpoint_model"]["settings"] != DEFAULT_MODEL_SETTINGS.to_metadata()
  assert report["checkpoint_model"]["counters"] == {
    "iteration": 5,
    "segment_namespace": 0,
  }
  assert report["checkpoint_model"]["schedule"] == {"max_iterations": 5}
  assert report["checkpoint_resolved_config"] == resolved_config
  assert report["result"]["mode"] == "student"
  assert report["result"]["steps"] == 2
  assert json.loads(report_path.read_text()) == report


def test_evaluate_cli_teacher_mode_needs_no_checkpoint(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  """The teacher baseline path keeps working without a student checkpoint."""
  _resolved_cohort(tmp_path)
  calls: list[dict] = []
  _install_fake_adapter(monkeypatch, calls)

  def evaluate(extra: list[str]) -> tuple[str | int | None, str, str]:
    code = invoke(
      monkeypatch,
      [
        "distill",
        "evaluate",
        "--manifest",
        str(tmp_path / "configs" / "tiny_teachers.yaml"),
        "--repo-root",
        str(tmp_path),
        "--teacher-id",
        TEACHER_ID,
        *extra,
      ],
    )
    captured = capsys.readouterr()
    return code, captured.out, captured.err

  code, out, err = evaluate(
    ["--mode", "teacher", "--steps", "1", "--num-envs", "2", "--seed", "3"]
  )
  assert code == 0, err
  report = json.loads(out)
  assert report["mode"] == "teacher"
  assert calls[0]["seed"] == 3
  assert report["runtime"]["resolved_seed"] == 3
  assert report["checkpoint_model"] is None
  assert report["checkpoint_resolved_config"] is None
  assert report["result"]["mode"] == "teacher"

  code, out, err = evaluate(["--mode", "student", "--steps", "1"])
  assert code == 1
  assert out == ""
  assert "requires --checkpoint" in err
  assert len(calls) == 1


def test_package_exports_model_only_inference_api() -> None:
  """The CLI-facing package exports the model-only inference loader."""
  import mjlab.tasks.tracking.distillation as distillation
  from mjlab.tasks.tracking.distillation import checkpoint

  assert distillation.InferenceModel is checkpoint.InferenceModel
  assert distillation.load_inference_checkpoint is checkpoint.load_inference_checkpoint


def test_evaluate_cli_takes_no_trainer_only_flags(
  monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  """Model-only evaluation must not advertise unused training settings."""
  code = invoke(monkeypatch, ["distill", "evaluate", "--help"])
  captured = capsys.readouterr()
  assert code == 0
  for flag in (
    "--minibatch-size",
    "--accumulation-steps",
    "--learning-rate",
    "--beta",
    "--replay-capacity",
  ):
    assert flag not in captured.out
  assert "--checkpoint" in captured.out
  assert "root yaw" in captured.out

  code = invoke(
    monkeypatch,
    ["distill", "evaluate", "--checkpoint", "state.pt", "--minibatch-size", "4"],
  )
  assert code != 0


def test_environment_factory_applies_forwarded_seed_before_construction(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """The forwarded seed lands on the private cfg before startup events.

  This mirrors the CLI contract with a synthetic registered config and a stub
  environment constructor, so no simulated scene or asset is loaded.
  """
  resolved = _resolved_cohort(tmp_path)
  constructions: list[tuple[int | None, str, object]] = []

  class _StubEnv:
    def __init__(self, cfg, device: str = "cpu", render_mode=None) -> None:
      constructions.append((cfg.seed, str(device), render_mode))
      self.cfg = cfg

  def _fake_env_cfg(task_id: str):
    del task_id
    return SimpleNamespace(
      seed=None,
      observations={"actor": SimpleNamespace(terms={})},
      scene=SimpleNamespace(num_envs=1),
      commands={
        "motion": MotionCommandCfg(
          resampling_time_range=(0.0, 0.0),
          motion_file="placeholder.npz",
          anchor_body_name="torso_link",
          body_names=("torso_link",),
          entity_name="robot",
        )
      },
    )

  monkeypatch.setattr("mjlab.tasks.registry.load_env_cfg", _fake_env_cfg)
  monkeypatch.setattr("mjlab.envs.ManagerBasedRlEnv", _StubEnv)

  first = build_distillation_environment(
    resolved, TEACHER_ID, num_envs=2, device="cpu", seed=123
  )
  second = build_distillation_environment(
    resolved, TEACHER_ID, num_envs=2, device="cpu", seed=123
  )

  assert constructions == [(123, "cpu", None), (123, "cpu", None)]
  provenance = first.cfg.__dict__["_distillation_seed_provenance"]
  assert provenance == RuntimeSeedProvenance(
    requested_seed=123, effective_seed=123, applied_before_construction=True
  )
  assert second.cfg.__dict__["_distillation_seed_provenance"] == provenance
  motion_cfg = first.cfg.commands["motion"]
  assert isinstance(motion_cfg, SegmentMotionCommandCfg)
  assert motion_cfg.motion_file == str(resolved.teacher(TEACHER_ID).entry.motion)
