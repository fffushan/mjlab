"""The ``distill`` CLI: dispatch, machine-readable report, and exit codes."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from tracking_distillation_fixtures import build_tiny_cohort, write_manifest

from mjlab.scripts.distill import main

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(autouse=True)
def _require_validation_dependencies():
  pytest.importorskip("onnx")
  pytest.importorskip("onnxruntime")


def invoke(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> str | int | None:
  monkeypatch.setattr(sys, "argv", argv)
  with pytest.raises(SystemExit) as exit_info:
    main()
  return exit_info.value.code


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
  monkeypatch.setattr(sys, "argv", ["distill", "train"])

  with pytest.raises(SystemExit) as exit_info:
    main()

  captured = capsys.readouterr()
  assert exit_info.value.code == 2
  assert "unknown command" in captured.err
  assert "validate-teachers" in captured.err


def test_cli_prints_help_without_a_command(
  monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  monkeypatch.setattr(sys, "argv", ["distill", "--help"])

  main()

  captured = capsys.readouterr()
  assert "usage: distill <COMMAND> [OPTIONS]" in captured.out
