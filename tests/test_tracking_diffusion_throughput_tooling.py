"""Focused tests for the D1 throughput integration seams.

These tests cover the integration-owned surface only: the CLI wiring for
``inspect``/``replay``/``build``/``stats``, the reusable preparation entry point
and the matched benchmark command generator.  They use synthetic CPU stores and
do not construct a simulator, a GPU runtime or a training run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from tracking_diffusion_fixtures import make_row

from mjlab.scripts import diffusion
from mjlab.tasks.tracking.diffusion import (
  AppendOnlyShardStore,
  BatchedCollectionResult,
  QualificationResult,
  TrialSpec,
)
from mjlab.tasks.tracking.diffusion.collection_controls import (
  CollectionControls,
  CollectionProgress,
  CollectionSummary,
)
from mjlab.tasks.tracking.diffusion.contract import DiffusionContract

_CONTRACT_PATH = Path("docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml")

_TOOLING_DIR = Path(diffusion.__file__).resolve().parents[3] / (
  "scripts/experiments/diffusion_d1_throughput"
)


def _load_module(name: str, filename: str) -> ModuleType:
  spec = importlib.util.spec_from_file_location(name, _TOOLING_DIR / filename)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _segments(root: Path, *, count: int, rows: int = 60, shard_rows: int = 23):
  """One independent segment (episode) per family, spread across shards."""
  store = AppendOnlyShardStore(root, max_rows_per_shard=shard_rows)
  for index in range(count):
    store.append(
      [
        make_row(
          tick,
          group=f"fam-{index}",
          episode=f"episode-{index}",
          run=f"run-{index}",
          env=index,
        )
        for tick in range(rows)
      ]
    )
  return store


def _worker_run(root: Path, *, start: int, stop: int, rows: int = 60) -> None:
  """Write one worker store and its completed collection report."""
  store_dir = root / "store"
  store = AppendOnlyShardStore(store_dir, max_rows_per_shard=23)
  for index in range(start, stop):
    store.append(
      [
        make_row(
          tick,
          group=f"fam-{index}",
          episode=f"episode-{index}",
          run=f"run-{index}",
          env=index,
        )
        for tick in range(rows)
      ]
    )
  report = {
    "contract_sha256": DiffusionContract.from_yaml(_CONTRACT_PATH).contract_sha256,
    "partial": False,
    "stop_reason": None,
    "results": [{"rows": rows} for _ in range(start, stop)],
  }
  (root / "report.json").write_text(json.dumps(report), encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI wiring
# --------------------------------------------------------------------------- #


def test_inspect_and_replay_accept_jobs(tmp_path: Path, capsys) -> None:
  store_root = tmp_path / "store"
  _segments(store_root, count=2, rows=50)
  code = diffusion.main(["inspect", "--store", str(store_root), "--jobs", "2"])
  assert code == 0
  payload = json.loads(capsys.readouterr().out)
  assert payload["rows"] == 100
  assert payload["unique_identities"] is True
  assert payload["controllers"] == {"vae": 100, "ppo": 0}

  code = diffusion.main(["replay", "--store", str(store_root), "--jobs", "2"])
  assert code == 0
  replay = json.loads(capsys.readouterr().out)
  assert replay["ok"] is True
  assert replay["errors"] == []


def test_inspect_rejects_a_non_positive_worker_count(tmp_path: Path, capsys) -> None:
  code = diffusion.main(["inspect", "--store", str(tmp_path), "--jobs", "0"])
  assert code == 2
  assert "jobs must be one or more" in capsys.readouterr().err


def test_stats_reuses_the_persisted_index(tmp_path: Path, capsys) -> None:
  prepare = _load_module("d1_prepare_cli", "prepare_dataset.py")
  dataset = tmp_path / "dataset"
  worker = tmp_path / "worker"
  _worker_run(worker, start=0, stop=8)
  assert (
    prepare.main(
      [
        "--contract",
        str(Path("docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml")),
        "--dataset-dir",
        str(dataset),
        "--input",
        str(worker / "store"),
        "--jobs",
        "1",
      ]
    )
    == 0
  )
  capsys.readouterr()
  index_path = dataset / "offline" / "window_index.npz"

  code = diffusion.main(
    [
      "stats",
      "--store",
      str(dataset / "store"),
      "--assignments",
      str(dataset / "offline" / "assignments.json"),
      "--bundle",
      str(tmp_path / "projection.npz"),
      "--report",
      str(tmp_path / "stats.json"),
      "--jobs",
      "1",
    ]
  )
  assert code == 0
  payload = json.loads(capsys.readouterr().out)
  assert payload["index_source"] == str(index_path)
  assert payload["train_windows_used"] > 0


def test_stats_refuses_a_foreign_persisted_index(tmp_path: Path, capsys) -> None:
  """The CLI wiring for the index-reuse refusal, not just the library guard."""
  prepare = _load_module("d1_prepare_foreign_index", "prepare_dataset.py")

  def build(name: str, *, families: int) -> Path:
    dataset = tmp_path / name
    worker = tmp_path / f"{name}-worker"
    _worker_run(worker, start=0, stop=families)
    assert (
      prepare.main(
        [
          "--dataset-dir",
          str(dataset),
          "--input",
          str(worker / "store"),
          "--jobs",
          "1",
        ]
      )
      == 0
    )
    capsys.readouterr()
    return dataset

  first = build("first", families=6)
  second = build("second", families=8)
  summary_path = tmp_path / "summary.json"

  code = diffusion.main(
    [
      "stats",
      "--store",
      str(second / "store"),
      "--assignments",
      str(second / "offline" / "assignments.json"),
      "--bundle",
      str(tmp_path / "projection.npz"),
      "--report",
      str(summary_path),
      "--jobs",
      "1",
      "--index",
      str(first / "offline" / "window_index.npz"),
    ]
  )
  assert code == 1
  captured = capsys.readouterr()
  assert "refusing the persisted window index" in captured.err
  assert "does not describe this store" in captured.err
  assert not (tmp_path / "projection.npz").exists()
  assert not summary_path.exists()


def test_build_jobs_matches_the_sequential_index(tmp_path: Path, capsys) -> None:
  store_root = tmp_path / "store"
  _segments(store_root, count=4, rows=55)
  assignments_a = tmp_path / "a.json"
  assignments_b = tmp_path / "b.json"
  report = tmp_path / "report.json"
  assert (
    diffusion.main(
      [
        "build",
        "--store",
        str(store_root),
        "--assignments",
        str(assignments_a),
        "--report",
        str(report),
        "--jobs",
        "1",
      ]
    )
    == 0
  )
  sequential = json.loads(capsys.readouterr().out)
  assert sequential["method"] == "sequential"
  assert (
    diffusion.main(
      [
        "build",
        "--store",
        str(store_root),
        "--assignments",
        str(assignments_b),
        "--report",
        str(report),
        "--jobs",
        "2",
      ]
    )
    == 0
  )
  parallel = json.loads(capsys.readouterr().out)
  assert parallel["total_windows"] == sequential["total_windows"]
  assert json.loads(assignments_a.read_text()) == json.loads(assignments_b.read_text())


def test_streaming_control_validation() -> None:
  base = {
    "max_rows_per_append": None,
    "compact_results": False,
    "stream_rows": False,
  }
  args = argparse.Namespace(**base)
  assert diffusion._validate_streaming_controls(args) is None
  args = argparse.Namespace(
    max_rows_per_append=4, compact_results=False, stream_rows=False
  )
  result = diffusion._validate_streaming_controls(args)
  assert result is not None and "requires --stream-rows" in result
  args = argparse.Namespace(
    max_rows_per_append=0, compact_results=False, stream_rows=True
  )
  result = diffusion._validate_streaming_controls(args)
  assert result is not None and "one or more" in result
  args = argparse.Namespace(
    max_rows_per_append=None, compact_results=True, stream_rows=False
  )
  result = diffusion._validate_streaming_controls(args)
  assert result is not None and "requires --stream-rows" in result


def test_collect_parser_accepts_the_explicit_speed_options(tmp_path: Path) -> None:
  parser = diffusion.build_parser()
  args = parser.parse_args(
    [
      "collect",
      "--execute",
      "--runtime-factory",
      "pkg:factory",
      "--output",
      str(tmp_path / "store"),
      "--report",
      str(tmp_path / "report.json"),
      "--stream-rows",
      "--compact-results",
      "--max-rows-per-append",
      "128",
      "--tensor-dispatch",
      "--policy-device",
      "cuda:0",
      "--progress",
      str(tmp_path / "progress.jsonl"),
    ]
  )
  assert args.stream_rows is True
  assert args.compact_results is True
  assert args.max_rows_per_append == 128
  assert args.tensor_dispatch is True
  assert args.policy_device == "cuda:0"


def test_progress_callback_appends_json_lines(tmp_path: Path) -> None:
  from dataclasses import dataclass

  @dataclass(frozen=True)
  class _Event:
    event: str
    trials_completed: int

  target = tmp_path / "progress.jsonl"
  callback = diffusion._progress_callback(target)
  assert callback is not None
  callback(_Event("trial", 1))
  callback(_Event("final", 2))
  lines = [json.loads(line) for line in target.read_text().splitlines()]
  assert lines == [
    {"event": "trial", "trials_completed": 1},
    {"event": "final", "trials_completed": 2},
  ]


def test_batched_outcome_payload_reads_compact_counts() -> None:
  class _Spec:
    run_id = "run"
    motion_id = "motion"
    phase = "clean"

  class _Qualification:
    qualified = True
    label = "vae_only"
    reason = ""

  class _Summary:
    spec = _Spec()
    qualification = _Qualification()
    row_count = 7
    accepted_row_count = 5
    verification_row_count = 2
    stopped_reason = None
    store_files = ("shard-000000.npz",)

  payload = diffusion._batched_outcome_payload(_Summary())
  assert payload["rows"] == 7
  assert payload["accepted_rows"] == 5
  assert payload["verification_rows"] == 2


# --------------------------------------------------------------------------- #
# Batched collection CLI wiring
# --------------------------------------------------------------------------- #


def _summary(spec: Any, *, rows: int = 4, steps: int = 4) -> CollectionSummary:
  return CollectionSummary(
    spec,
    QualificationResult(
      qualified=True,
      label="vae_only",
      reason=None,
      steps=steps,
      vae_steps=steps,
      retained_vae_steps=2,
      handoff_step=None,
    ),
    row_count=rows,
    accepted_row_count=rows - 1,
    verification_row_count=1,
    store_files=("shard-000000.npz",),
  )


class _FakeBatchedCollector:
  """Record one ``collect_batched`` call and return a prepared result.

  The concrete collector's limits, publication and progress behaviour are
  covered by the streaming tests.  This stand-in isolates the CLI wiring: the
  controls the CLI builds from the command line and the report it writes for a
  clean run and for a run that stopped early.
  """

  def __init__(self, *, stop_reason: str | None = None) -> None:
    self.env: Any = None
    self.vae: Any = None
    self.stop_reason = stop_reason
    self.calls: list[dict[str, Any]] = []

  def collect_batched(
    self, specs: Any, *, slots: int | None = None, controls: Any = None
  ) -> BatchedCollectionResult:
    controls = controls or CollectionControls()
    self.calls.append({"specs": tuple(specs), "slots": slots, "controls": controls})
    completed = tuple(specs) if self.stop_reason is None else tuple(specs[:-1])
    summaries = tuple(_summary(spec) for spec in completed)
    transitions = sum(summary.qualification.steps for summary in summaries)
    final: CollectionProgress | None = None
    if controls.progress is not None:
      for position, summary in enumerate(summaries, start=1):
        controls.progress(
          CollectionProgress(
            event="trial",
            final=False,
            trials_completed=position,
            transitions=position * 4,
            stored_rows=position * 3,
            accepted_rows=position * 2,
            store_files=summary.store_files,
          )
        )
      final = CollectionProgress(
        event="final",
        final=True,
        trials_completed=len(summaries),
        transitions=transitions,
        stored_rows=len(summaries) * 3,
        accepted_rows=len(summaries) * 2,
        store_files=tuple(name for item in summaries for name in item.store_files),
        stop_reason=self.stop_reason,
      )
      controls.progress(final)
    return BatchedCollectionResult(
      summaries,
      store_files=("shard-000000.npz",),
      stop_reason=self.stop_reason,
      progress=final,
      unstarted_trials=0 if self.stop_reason is None else len(tuple(specs)) - 1,
    )


def _collect_trials() -> tuple[TrialSpec, TrialSpec]:
  clean = TrialSpec("run", "motion", 7, 3, pair_id="pair", initial_state_id="state")
  ou = TrialSpec("run", "motion", 7, 3, "ou", pair_id="pair", initial_state_id="state")
  return clean, ou


def _collect_argv(root: Path, *extra: str) -> list[str]:
  return [
    "collect",
    "--execute",
    "--runtime-factory",
    "unused:factory",
    "--output",
    str(root / "store"),
    "--report",
    str(root / "report.json"),
    "--max-envs",
    "4",
    "--max-trials",
    "8",
    *extra,
  ]


def _install_fake_collect(
  monkeypatch: pytest.MonkeyPatch,
  collector: Any,
  seen_settings: list[str],
  *,
  batched: bool = True,
) -> None:
  """Replace preflight, runtime validation and the factory loader.

  The pinned frozen artifacts are absent in this worktree and a stand-in
  collector cannot carry their identity, so the CLI's report wiring is what
  this test exercises.  The factory records the policy-device setting it sees.
  """

  def factory(request: Any) -> Any:
    setting = os.environ.get("MJLAB_D1_POLICY_DEVICE", "cpu")
    seen_settings.append(setting)
    return diffusion.RuntimeBundle(
      collector,
      _collect_trials(),
      True,
      {
        "environment": "agibot_x2",
        "max_envs": request.max_envs,
        "max_gpus": request.max_gpus,
        "batched": batched,
        "policy_device": "cpu",
        "policy_device_setting": setting,
      },
    )

  monkeypatch.setattr(
    diffusion,
    "run_preflight",
    lambda contract, artifact_root=None: diffusion.PreflightReport(
      True, "sha256", {}, ()
    ),
  )
  monkeypatch.setattr(
    diffusion, "_validate_runtime_bundle", lambda bundle, **kwargs: None
  )
  monkeypatch.setattr(diffusion, "_load_factory", lambda spec: factory)


def test_collect_batched_clean_run_reports_truthful_counts(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
  collector = _FakeBatchedCollector()
  settings: list[str] = []
  _install_fake_collect(monkeypatch, collector, settings)

  code = diffusion.main(_collect_argv(tmp_path))
  assert code == 0
  payload = json.loads(capsys.readouterr().out)
  assert payload["partial"] is False
  assert payload["stop_reason"] is None
  assert payload["unstarted_trials"] == 0
  assert payload["trials_attempted"] == 2
  # Without a progress callback the transitions come from the trial verdicts.
  assert payload["control_transitions"] == 8
  assert payload["progress"] is None
  assert [entry["rows"] for entry in payload["results"]] == [4, 4]
  assert [entry["accepted_rows"] for entry in payload["results"]] == [3, 3]
  assert payload["policy_device"] == "cpu"
  assert payload["policy_device_setting"] == "cpu"
  assert json.loads((tmp_path / "report.json").read_text())["partial"] is False

  assert len(collector.calls) == 1
  call = collector.calls[0]
  assert call["slots"] == 4
  controls = call["controls"]
  # Wall-clock and inactivity limits always bind a batched run.
  assert controls.limits.wall_seconds == 60.0
  assert controls.limits.inactivity_seconds == 120.0
  # Default streaming also enforces the stored-output limit.
  assert controls.limits.max_output_bytes == 512 * 1024 * 1024
  assert controls.stream_rows is True
  assert controls.compact_results is True
  # A factory without tensor methods keeps the compatible NumPy path.
  assert controls.tensor_dispatch is False


def test_collect_defaults_use_supported_tensor_methods_and_legacy_opt_out(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
  collector = _FakeBatchedCollector()
  collector.env = type(
    "TensorEnv",
    (),
    {
      "observe_vae_tensor_batch": lambda self, ids: None,
      "step_tensor_batch": lambda self, action: None,
    },
  )()
  collector.vae = type(
    "TensorVae", (), {"batch_action_tensor": lambda self, observation: None}
  )()
  settings: list[str] = []
  _install_fake_collect(monkeypatch, collector, settings)
  assert diffusion.main(_collect_argv(tmp_path)) == 0
  capsys.readouterr()
  controls = collector.calls[-1]["controls"]
  assert controls.stream_rows and controls.compact_results and controls.tensor_dispatch
  assert (
    diffusion.main(
      _collect_argv(
        tmp_path, "--no-stream-rows", "--no-compact-results", "--no-tensor-dispatch"
      )
    )
    == 0
  )
  capsys.readouterr()
  controls = collector.calls[-1]["controls"]
  assert (
    not controls.stream_rows
    and not controls.compact_results
    and not controls.tensor_dispatch
  )
  assert controls.limits.max_output_bytes is None


def test_collect_batched_stopped_run_returns_nonzero_and_warns(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
  collector = _FakeBatchedCollector(stop_reason="wall_clock_limit")
  settings: list[str] = []
  _install_fake_collect(monkeypatch, collector, settings)

  code = diffusion.main(_collect_argv(tmp_path))
  assert code == 1
  captured = capsys.readouterr()
  payload = json.loads(captured.out)
  assert payload["partial"] is True
  assert payload["stop_reason"] == "wall_clock_limit"
  assert payload["unstarted_trials"] == 1
  assert payload["trials_attempted"] == 1
  assert "partial" in captured.err
  assert "wall_clock_limit" in captured.err
  assert "do not merge" in captured.err
  assert json.loads((tmp_path / "report.json").read_text())["partial"] is True


def test_collect_batched_streaming_controls_write_progress(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
  collector = _FakeBatchedCollector()
  settings: list[str] = []
  _install_fake_collect(monkeypatch, collector, settings)
  progress_path = tmp_path / "progress.jsonl"

  code = diffusion.main(
    _collect_argv(
      tmp_path,
      "--stream-rows",
      "--compact-results",
      "--max-rows-per-append",
      "2",
      "--tensor-dispatch",
      "--progress",
      str(progress_path),
    )
  )
  assert code == 0
  payload = json.loads(capsys.readouterr().out)
  assert payload["streaming"] == {
    "stream_rows": True,
    "compact_results": True,
    "max_rows_per_append": 2,
    "tensor_dispatch": True,
  }
  # The final progress record is the report's progress, and it agrees with the
  # transition total the CLI reports.
  assert payload["progress"]["event"] == "final"
  assert payload["progress"]["stored_rows"] == 6
  assert payload["progress"]["stop_reason"] is None
  assert payload["control_transitions"] == payload["progress"]["transitions"] == 8

  controls = collector.calls[0]["controls"]
  assert controls.max_rows_per_append == 2
  assert controls.tensor_dispatch is True
  assert controls.limits.max_output_bytes == 512 * 1024 * 1024

  lines = [json.loads(line) for line in progress_path.read_text().splitlines()]
  assert [line["event"] for line in lines] == ["trial", "trial", "final"]
  assert lines[-1]["transitions"] == 8


def test_collect_refuses_batched_only_options_on_a_scalar_request(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
  collector = _FakeBatchedCollector()
  settings: list[str] = []
  _install_fake_collect(monkeypatch, collector, settings)
  cases = (
    ["--stream-rows"],
    ["--stream-rows", "--compact-results"],
    ["--stream-rows", "--max-rows-per-append", "8"],
    ["--tensor-dispatch"],
    ["--progress", str(tmp_path / "progress.jsonl")],
  )
  for extra in cases:
    root = tmp_path / extra[0].lstrip("-")
    code = diffusion.main(
      [
        "collect",
        "--execute",
        "--runtime-factory",
        "unused:factory",
        "--output",
        str(root / "store"),
        "--report",
        str(root / "report.json"),
        "--max-envs",
        "1",
        *extra,
      ]
    )
    assert code == 2
    assert "honoured only by the batched runtime" in capsys.readouterr().err
    # Refused before any runtime work: no report, no store, no factory call.
    assert not (root / "report.json").exists()
    assert not (root / "store").exists()
  assert collector.calls == []
  assert settings == []


def test_collect_refuses_a_batched_only_option_when_the_factory_is_scalar(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
  collector = _FakeBatchedCollector()
  settings: list[str] = []
  _install_fake_collect(monkeypatch, collector, settings, batched=False)

  code = diffusion.main(_collect_argv(tmp_path, "--stream-rows"))
  assert code == 2
  assert "honoured only by the batched runtime" in capsys.readouterr().err
  assert collector.calls == []


def test_collect_scopes_the_policy_device_override(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
  monkeypatch.delenv("MJLAB_D1_POLICY_DEVICE", raising=False)
  collector = _FakeBatchedCollector()
  settings: list[str] = []
  _install_fake_collect(monkeypatch, collector, settings)

  assert (
    diffusion.main(_collect_argv(tmp_path / "first", "--policy-device", "cpu")) == 0
  )
  payload = json.loads(capsys.readouterr().out)
  assert settings == ["cpu"]
  assert payload["policy_device"] == "cpu"
  assert payload["policy_device_setting"] == "cpu"
  # The override must not outlive the factory call it was set for.
  assert "MJLAB_D1_POLICY_DEVICE" not in os.environ

  # A setting that is already exported is used and restored unchanged.
  monkeypatch.setenv("MJLAB_D1_POLICY_DEVICE", "simulation")
  assert diffusion.main(_collect_argv(tmp_path / "second")) == 0
  payload = json.loads(capsys.readouterr().out)
  assert settings == ["cpu", "simulation"]
  assert payload["policy_device"] == "cpu"
  assert payload["policy_device_setting"] == "simulation"
  assert os.environ["MJLAB_D1_POLICY_DEVICE"] == "simulation"


# --------------------------------------------------------------------------- #
# Preparation entry point
# --------------------------------------------------------------------------- #


def test_prepare_refuses_a_partial_worker_report(tmp_path: Path, capsys) -> None:
  prepare = _load_module("d1_prepare_partial", "prepare_dataset.py")
  worker = tmp_path / "worker"
  _worker_run(worker, start=0, stop=4)
  report_path = worker / "report.json"
  report = json.loads(report_path.read_text())
  report["partial"] = True
  report["stop_reason"] = "wall_clock_limit"
  report_path.write_text(json.dumps(report))

  code = prepare.main(
    [
      "--dataset-dir",
      str(tmp_path / "dataset"),
      "--input",
      str(worker / "store"),
      "--jobs",
      "1",
    ]
  )
  assert code == 1
  assert "partial" in capsys.readouterr().err
  assert not (tmp_path / "dataset" / "store").exists()


def test_prepare_requires_a_worker_report(tmp_path: Path, capsys) -> None:
  prepare = _load_module("d1_prepare_missing", "prepare_dataset.py")
  worker = tmp_path / "worker"
  _worker_run(worker, start=0, stop=4)
  (worker / "report.json").unlink()
  code = prepare.main(
    [
      "--dataset-dir",
      str(tmp_path / "dataset"),
      "--input",
      str(worker / "store"),
      "--jobs",
      "1",
    ]
  )
  assert code == 1
  assert "no collection report" in capsys.readouterr().err


def test_prepare_refuses_a_row_accounting_mismatch(tmp_path: Path, capsys) -> None:
  prepare = _load_module("d1_prepare_accounting", "prepare_dataset.py")
  worker = tmp_path / "worker"
  _worker_run(worker, start=0, stop=4)
  report_path = worker / "report.json"
  report = json.loads(report_path.read_text())
  report["results"][0]["rows"] = report["results"][0]["rows"] + 1
  report_path.write_text(json.dumps(report))
  code = prepare.main(
    [
      "--dataset-dir",
      str(tmp_path / "dataset"),
      "--input",
      str(worker / "store"),
      "--jobs",
      "1",
    ]
  )
  assert code == 1
  assert "accounts for" in capsys.readouterr().err


@pytest.fixture(scope="module")
def prepared_dataset(tmp_path_factory) -> Path:
  prepare = _load_module("d1_prepare_fixture", "prepare_dataset.py")
  root = tmp_path_factory.mktemp("prepared")
  dataset = root / "dataset"
  worker_a = root / "worker-a"
  worker_b = root / "worker-b"
  _worker_run(worker_a, start=0, stop=5)
  _worker_run(worker_b, start=5, stop=10)
  code = prepare.main(
    [
      "--dataset-dir",
      str(dataset),
      "--input",
      str(worker_a / "store"),
      "--input",
      str(worker_b / "store"),
      "--jobs",
      "1",
      "--cache-jobs",
      "1",
      "--report",
      str(root / "prepare.json"),
    ]
  )
  assert code == 0
  return dataset


def test_prepare_builds_a_trainable_dataset_without_training(
  prepared_dataset: Path, capsys
) -> None:
  capsys.readouterr()
  offline = prepared_dataset / "offline"
  assert (offline / "assignments.json").is_file()
  assert (offline / "dataset.json").is_file()
  assert (offline / "window_index.npz").is_file()
  assert (offline / "projection.npz").is_file()
  assert list(offline.glob("tokens-train-*.npy"))
  report = json.loads((prepared_dataset.parent / "prepare.json").read_text())
  assert report["complete"] is True
  assert report["training_started"] is False
  phases = [entry["phase"] for entry in report["phases"]]
  assert phases == ["merge", "dataset", "statistics", "cache"]
  assert "--require-prepared-cache" in report["training_command"]


def test_prepare_parallel_jobs_preserve_sources_and_output(
  tmp_path: Path, capsys
) -> None:
  """Three inputs and ``--jobs 2`` must not touch a source or lose a row."""
  prepare = _load_module("d1_prepare_parallel", "prepare_dataset.py")
  workers = []
  for index in range(3):
    worker = tmp_path / f"worker-{index}"
    _worker_run(worker, start=4 * index, stop=4 * index + 4)
    workers.append(worker)
  before = {
    path: hashlib.sha256(path.read_bytes()).hexdigest()
    for worker in workers
    for path in sorted(worker.rglob("*"))
    if path.is_file()
  }
  dataset = tmp_path / "dataset"
  report_path = tmp_path / "prepare.json"
  argv = [
    "--dataset-dir",
    str(dataset),
    "--jobs",
    "2",
    "--cache-jobs",
    "2",
    "--batch-windows",
    "2",
    "--report",
    str(report_path),
  ]
  for worker in workers:
    argv += ["--input", str(worker / "store")]

  code = prepare.main(argv)
  assert code == 0, capsys.readouterr().err
  report = json.loads(report_path.read_text())
  assert report["complete"] is True
  assert len(report["workers"]) == 3
  merge_phase = report["phases"][0]
  assert merge_phase["phase"] == "merge"
  assert merge_phase["merge"]["inputs"] == {
    str(worker / "store"): 4 * 60 for worker in workers
  }
  assert merge_phase["merge"]["rows_before"] == 0
  assert merge_phase["merge"]["input_rows"] == 12 * 60
  assert merge_phase["merge"]["rows_after"] == 12 * 60
  assert merge_phase["merged_audit"]["rows"] == 12 * 60

  after = {
    path: hashlib.sha256(path.read_bytes()).hexdigest()
    for worker in workers
    for path in sorted(worker.rglob("*"))
    if path.is_file()
  }
  assert after == before

  offline = dataset / "offline"
  for name in (
    "assignments.json",
    "dataset.json",
    "window_index.npz",
    "projection.npz",
  ):
    assert (offline / name).is_file()
  assert list(offline.glob("tokens-train-*.npy"))
  rows = list(AppendOnlyShardStore(dataset / "store").iter_rows())
  assert len(rows) == 12 * 60
  identities = [row.identity for row in rows]
  assert len(set(identities)) == len(identities)


def test_prepare_refuses_to_overwrite_an_existing_store(
  prepared_dataset: Path, tmp_path: Path, capsys
) -> None:
  prepare = _load_module("d1_prepare_overwrite", "prepare_dataset.py")
  worker = tmp_path / "worker"
  _worker_run(worker, start=0, stop=3)
  code = prepare.main(
    [
      "--dataset-dir",
      str(prepared_dataset),
      "--input",
      str(worker / "store"),
      "--jobs",
      "1",
    ]
  )
  assert code == 1
  assert "refusing to overwrite" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Benchmark matrix generator
# --------------------------------------------------------------------------- #


def test_preparation_writes_report_for_store_validation_failure(
  tmp_path: Path, monkeypatch, capsys
) -> None:
  prepare = _load_module("d1_prepare_validation_failure", "prepare_dataset.py")
  worker = tmp_path / "worker"
  _worker_run(worker, start=0, stop=1)

  def refuse(*args, **kwargs):
    raise ValueError("malformed controller/latent row")

  monkeypatch.setattr(prepare, "validate_worker_store", refuse)
  report = tmp_path / "failure.json"
  assert (
    prepare.main(
      [
        "--dataset-dir",
        str(tmp_path / "dataset"),
        "--input",
        str(worker / "store"),
        "--report",
        str(report),
      ]
    )
    == 1
  )
  assert json.loads(report.read_text())["complete"] is False
  assert "malformed controller/latent row" in capsys.readouterr().err


def test_generated_benchmark_shell_records_failure_and_continues(
  tmp_path: Path,
) -> None:
  matrix = _load_module("d1_benchmark_shell", "benchmark_matrix.py")
  arms = []
  for label, status in (("stopped", 1), ("finished", 0)):
    output = tmp_path / label
    output.mkdir()
    arms.append(
      {
        "arm": label,
        "output": str(output),
        "env": {},
        "argv": [sys.executable, "-c", f"raise SystemExit({status})"],
      }
    )
  script = tmp_path / "commands.sh"
  script.write_text(matrix.render_shell({"arms": arms}))
  result = subprocess.run(
    ["bash", str(script)], capture_output=True, text=True, timeout=10
  )
  assert result.returncode == 0
  assert (tmp_path / "stopped/exit.status").read_text().strip() == "1"
  assert (tmp_path / "finished/exit.status").read_text().strip() == "0"
  assert "arm stopped exited 1" in result.stderr


def test_benchmark_matrix_is_matched_and_inert(tmp_path: Path, capsys) -> None:
  matrix = _load_module("d1_benchmark_matrix", "benchmark_matrix.py")
  plan_dir = tmp_path / "plan"
  code = matrix.main(
    [
      "--plan-dir",
      str(plan_dir),
      "--stamp",
      "20261012T000000Z",
      "--output-root",
      str(tmp_path / "runs"),
    ]
  )
  assert code == 0
  capsys.readouterr()
  plan = json.loads((plan_dir / "plan.json").read_text())
  assert plan["varied_dimensions"]["vae_policy_device"] == ["cpu", "cuda:0"]
  assert plan["varied_dimensions"]["max_envs"] == [64, 128, 256]
  assert len(plan["arms"]) == 6
  shared = plan["shared"]
  for arm in plan["arms"]:
    argv = arm["argv"]
    assert argv[argv.index("--policy-device") + 1] == arm["vae_policy_device"]
    assert argv[argv.index("--max-envs") + 1] == str(arm["max_envs"])
    assert argv[argv.index("--max-trials") + 1] == str(shared["max_trials"])
    assert "--stream-rows" in argv and "--compact-results" in argv
    assert arm["env"]["MJLAB_D1_BULK_SEEDS"] == "1"
    assert arm["env"]["MJLAB_D1_WORKERS"] == "1"
    assert arm["env"]["MJLAB_D1_WORKER_OFFSET"] == "0"
    assert arm["env"]["MJLAB_D1_ROLLOUT_TEACHERS"].split(",") == [
      f"vgk{i:03d}" for i in range(8)
    ]
    assert arm["output"] not in plan["read_only_inputs"]
  assert plan["note"].startswith("generated commands only")
  assert (plan_dir / "commands.sh").is_file()


def test_benchmark_matrix_refuses_existing_outputs(tmp_path: Path, capsys) -> None:
  matrix = _load_module("d1_benchmark_matrix_refuse", "benchmark_matrix.py")
  plan_dir = tmp_path / "plan"
  plan_dir.mkdir()
  code = matrix.main(["--plan-dir", str(plan_dir), "--stamp", "s"])
  assert code == 2
  assert "already exists" in capsys.readouterr().err

  fresh = tmp_path / "fresh"
  (tmp_path / "runs" / "s" / "vae-cpu-envs-64").mkdir(parents=True)
  code = matrix.main(
    [
      "--plan-dir",
      str(fresh),
      "--stamp",
      "s",
      "--output-root",
      str(tmp_path / "runs"),
    ]
  )
  assert code == 2
  assert "refusing to overwrite" in capsys.readouterr().err
