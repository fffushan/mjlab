"""Parent regressions for source receipts, ordering and complete coverage.

D3-layout fixtures are synthetic declarations for parser testing, not physics.
"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_tracking_diffusion_d4_integration import (
  _keyframe,
  _patch_prepare,
  _prepare_args,
  _spec,
  _write_d3_recording,
  evaluate,
  prepare_task_config,
  task_config,
)

from mjlab.tasks.tracking.diffusion.conditioned_continuation import TaskSource


@pytest.mark.parametrize(
  "fault", ["valid", "integrity", "vae", "body_order", "coverage", "encoder"]
)
def test_recording_rejects_inconsistent_qualification(
  tmp_path: Path, fault: str
) -> None:
  root = _write_d3_recording(tmp_path / "recording")
  path = root / "summary.json"
  summary = json.loads(path.read_text())
  if fault == "valid":
    summary["valid"] = False
  elif fault == "integrity":
    summary["provenance"]["integrity"]["unchanged"] = False
  elif fault == "vae":
    summary["provenance"]["vae"]["sha256"] = "0" * 64
  elif fault == "body_order":
    summary["branches"]["baseline"]["provenance"]["body_order"].reverse()
  elif fault == "coverage":
    summary["branches"]["baseline"]["completed_ticks"] -= 1
  elif fault == "encoder":
    summary["branches"]["baseline"]["counters"]["encoder"] -= 1
  path.write_text(json.dumps(summary))
  with pytest.raises(task_config.TaskConfigError):
    task_config.read_d3_recording(root / "ticks.jsonl", summary_path=path)


@pytest.mark.parametrize("fault", ["duplicate", "missing_prefix", "bad_last_post"])
def test_recording_rejects_bad_coverage_or_terminal_clock(
  tmp_path: Path, fault: str
) -> None:
  root = _write_d3_recording(tmp_path / "recording")
  path = root / "ticks.jsonl"
  rows = [json.loads(line) for line in path.read_text().splitlines()]
  if fault == "duplicate":
    rows.append(rows[0])
  elif fault == "missing_prefix":
    rows = rows[2:]
  elif fault == "bad_last_post":
    baseline = [row for row in rows if row["branch"] == "baseline"]
    baseline[-1]["tick"]["post"]["tick"] += 1
  path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
  with pytest.raises(task_config.TaskConfigError):
    task_config.read_d3_recording(path, summary_path=root / "summary.json")


def test_npz_receipt_joins_preflight_integrity(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  shard = tmp_path / "diagnostic-source.npz"
  shard.write_bytes(b"receipt mapping fixture, not an actual NPZ recording")
  digest = hashlib.sha256(shard.read_bytes()).hexdigest()
  source = TaskSource(
    kind="baseline_recording",
    qualification="declared",
    verification="loader_recheck",
    provenance="diagnostic receipt mapping",
    path=str(shard),
    sha256=digest,
    verification_detail={
      "loader_recheck": {"shard_path": str(shard), "shard_sha256": digest}
    },
  )
  spec = replace(_spec(mode="keyframe", keyframes=(_keyframe(18),)), source=source)
  _patch_prepare(monkeypatch)
  monkeypatch.setattr(
    evaluate,
    "_task_loader",
    lambda: SimpleNamespace(
      ScenarioExpectation=task_config.ScenarioExpectation,
      load_task_config=lambda *args, **kwargs: spec,
    ),
  )
  task = tmp_path / "task.json"
  task.write_text("{}")
  args = _prepare_args(tmp_path, task_config=task, execute=False)
  prepared = evaluate._prepare(args)
  assert prepared.file_expectations["task_source_shard"] == (shard, digest)


def test_historical_d3_body_order_uses_audited_contract(tmp_path: Path) -> None:
  root = _write_d3_recording(tmp_path / "recording")
  path = root / "summary.json"
  summary = json.loads(path.read_text())
  provenance = summary["branches"]["baseline"]["provenance"]
  provenance.pop("body_order")
  provenance.update(live_contract_audited=True, factory_trusted=True)
  path.write_text(json.dumps(summary))
  assert (
    task_config.read_d3_recording(root / "ticks.jsonl", summary_path=path).qualification
    == "verified"
  )
  provenance["live_contract_audited"] = False
  path.write_text(json.dumps(summary))
  with pytest.raises(task_config.TaskConfigError):
    task_config.read_d3_recording(root / "ticks.jsonl", summary_path=path)


def test_fresh_nested_source_receipt_joins_cli_integrity(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  root = _write_d3_recording(tmp_path / "recording")
  task = tmp_path / "task.json"
  assert (
    prepare_task_config.main(
      [
        "--recording-dir",
        str(root),
        "--member",
        "tennis_000",
        "--seed",
        "0",
        "--start-frame",
        "45",
        "--mode",
        "keyframe",
        "--keyframe-ticks",
        "18",
        "--output",
        str(task),
      ]
    )
    == 0
  )
  _patch_prepare(monkeypatch)
  args = _prepare_args(tmp_path, output_dir=tmp_path / "not_created")
  args.task_config = task
  args.start_frame = 45
  prepared = evaluate._prepare(args)
  assert "task_config" in prepared.file_expectations
  assert "task_source_ticks" in prepared.file_expectations
  assert "task_source_summary" in prepared.file_expectations
  assert prepared.task is not None
  assert prepared.task.source.verification_detail is not None
  receipt = prepared.task.source.verification_detail["loader_recheck"]
  assert prepared.file_expectations["task_source_ticks"][1] == receipt["ticks_sha256"]
  assert (
    prepared.file_expectations["task_source_summary"][1] == receipt["summary_sha256"]
  )
