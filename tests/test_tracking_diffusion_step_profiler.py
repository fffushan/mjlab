"""Focused tests for the bounded step profiler.

The profiler exists to attribute the cost of one training step when the main
thread is saturated and the GPU is mostly idle. It must be inert unless the
environment asks for it, so these tests pin the default-off contract and the
artifact-writing path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mjlab.tasks.tracking.diffusion.profiling import StepProfiler


def _profiler(tmp_path: Path, kind: str | None, **kwargs: int) -> StepProfiler:
  return StepProfiler(
    kind,
    steps=kwargs.get("steps", 2),
    warmup=kwargs.get("warmup", 1),
    out_dir=tmp_path / "profile",
  )


def test_default_off_is_inert(tmp_path: Path) -> None:
  profiler = _profiler(tmp_path, None)
  for step in range(10):
    profiler.on_step_start(step)
    profiler.on_step_end()
  profiler.finish()
  assert profiler.records == 0
  assert not (tmp_path / "profile").exists()


def test_from_environment_defaults_off(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.delenv("MJLAB_DIFFUSION_PROFILE", raising=False)
  profiler = StepProfiler.from_environment(tmp_path)
  assert profiler.kind is None
  assert profiler.out_dir == tmp_path / "profile"


def test_from_environment_reads_settings(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv("MJLAB_DIFFUSION_PROFILE", "cprofile")
  monkeypatch.setenv("MJLAB_DIFFUSION_PROFILE_STEPS", "7")
  monkeypatch.setenv("MJLAB_DIFFUSION_PROFILE_WARMUP", "3")
  monkeypatch.setenv("MJLAB_DIFFUSION_PROFILE_OUT", str(tmp_path / "elsewhere"))
  profiler = StepProfiler.from_environment(tmp_path)
  assert profiler.kind == "cprofile"
  assert (profiler.steps, profiler.warmup) == (7, 3)
  assert profiler.out_dir == tmp_path / "elsewhere"


def test_invalid_kind_is_refused(tmp_path: Path) -> None:
  with pytest.raises(ValueError, match="MJLAB_DIFFUSION_PROFILE"):
    _profiler(tmp_path, "flamegraph")


def test_warmup_is_skipped_and_artifacts_are_written(tmp_path: Path) -> None:
  profiler = _profiler(tmp_path, "cprofile", steps=2, warmup=1)
  for step in range(5):
    profiler.on_step_start(step)
    sum(range(50))  # some measurable work
    profiler.on_step_end()
  profiler.finish()

  summary = json.loads((profiler.out_dir / "profile_summary.json").read_text())
  assert summary["kind"] == "cprofile"
  assert summary["recorded_steps"] == 2
  assert summary["warmup_steps_skipped"] == 1
  assert summary["step_seconds_mean"] > 0
  assert (
    summary["step_seconds_min"]
    <= summary["step_seconds_mean"]
    <= summary["step_seconds_max"]
  )
  # A clock that starts once sums a cumulative ramp, whose maximum is about twice
  # the mean. Per-step timings must be far tighter than that.
  assert summary["step_seconds_max"] < 3 * summary["step_seconds_mean"]
  assert "cprofile.txt" in summary["artifacts"]
  assert (profiler.out_dir / "cprofile.txt").read_text().strip()
  assert profiler.records == 2


def test_recording_stops_at_the_budget(tmp_path: Path) -> None:
  profiler = _profiler(tmp_path, "cprofile", steps=1, warmup=0)
  for step in range(6):
    profiler.on_step_start(step)
    profiler.on_step_end()
  assert profiler.records == 1
  assert profiler.active is False
  profiler.finish()
  assert (
    json.loads((profiler.out_dir / "profile_summary.json").read_text())[
      "recorded_steps"
    ]
    == 1
  )


def test_torch_kind_writes_summary(tmp_path: Path) -> None:
  torch = pytest.importorskip("torch")
  profiler = _profiler(tmp_path, "torch", steps=2, warmup=0)
  for step in range(3):
    profiler.on_step_start(step)
    assert (torch.ones(64) * 2).shape == (64,)
    profiler.on_step_end()
  profiler.finish()
  summary = json.loads((profiler.out_dir / "profile_summary.json").read_text())
  assert summary["kind"] == "torch"
  assert summary["recorded_steps"] == 2
  assert (
    "torch_key_averages.txt" in summary["artifacts"]
    or (profiler.out_dir / "torch_export_error.txt").exists()
  )
