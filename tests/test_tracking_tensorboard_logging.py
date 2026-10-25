"""Tests for the optional TensorBoard mirror used by the flow/diffusion trainers.

The sink reads its configuration from the environment rather than the training
config, because the config hash is part of the resume identity.  These tests
cover that contract: flattening, step selection, the disable switch, the
directory override, and a non-finite guard for diverged runs.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from mjlab.tasks.tracking import tensorboard_logging as tb


def _tags(directory: Path) -> dict[str, list[tuple[int, float]]]:
  """Read one event directory into ``{tag: [(step, value), ...]}``."""
  accumulator = EventAccumulator(str(directory))
  accumulator.Reload()
  return {
    tag: [(event.step, event.value) for event in accumulator.Scalars(tag)]
    for tag in accumulator.Tags()["scalars"]
  }


def test_flatten_row_expands_nested_metric_blocks() -> None:
  pairs = dict(
    tb.flatten_row(
      "",
      {
        "kind": "epoch",
        "epoch": 3,
        "loss": 0.5,
        "eval_model": {"loss": 0.25, "state_mse": 0.1},
        "eval_ema": {"loss": 0.2},
        "eval_split": "validation",
        "note": None,
        "nan_metric": math.nan,
      },
    )
  )
  assert pairs["loss"] == pytest.approx(0.5)
  assert pairs["eval_model/loss"] == pytest.approx(0.25)
  assert pairs["eval_model/state_mse"] == pytest.approx(0.1)
  assert pairs["eval_ema/loss"] == pytest.approx(0.2)
  # Strings, None and non-finite values must not reach TensorBoard.
  assert "eval_split" not in pairs
  assert "note" not in pairs
  assert "nan_metric" not in pairs
  assert pairs["epoch"] == pytest.approx(3.0)


def test_writer_mirrors_rows_at_their_own_step(tmp_path, monkeypatch) -> None:
  monkeypatch.delenv(tb.ENVIRONMENT_FLAG, raising=False)
  monkeypatch.delenv(tb.ENVIRONMENT_DIR, raising=False)
  run_dir = tmp_path / "run"
  writer = tb.open_for(run_dir)
  assert writer is not None
  writer.write_row({"kind": "update", "global_step": 10, "loss": 1.0})
  writer.write_row({"kind": "epoch", "global_step": 20, "eval_model": {"loss": 0.5}})
  directory = writer.directory
  writer.close()

  assert directory == run_dir / "tensorboard"
  tags = _tags(directory)
  assert tags["loss"] == [(10, 1.0)]
  assert tags["eval_model/loss"] == [(20, 0.5)]


def test_flag_zero_disables_the_writer(tmp_path, monkeypatch) -> None:
  monkeypatch.setenv(tb.ENVIRONMENT_FLAG, "0")
  monkeypatch.delenv(tb.ENVIRONMENT_DIR, raising=False)
  assert tb.open_for(tmp_path / "run") is None
  assert not (tmp_path / "run" / "tensorboard").exists()


def test_directory_override_is_honoured(tmp_path, monkeypatch) -> None:
  target = tmp_path / "somewhere" / "events"
  monkeypatch.delenv(tb.ENVIRONMENT_FLAG, raising=False)
  monkeypatch.setenv(tb.ENVIRONMENT_DIR, str(target))
  writer = tb.open_for(tmp_path / "run")
  assert writer is not None
  writer.write_row({"global_step": 1, "gradient_norm": 2.5})
  writer.close()
  assert target.is_dir()
  assert _tags(target)["gradient_norm"] == [(1, 2.5)]


def test_row_step_falls_back_when_no_step_field_is_present(
  tmp_path, monkeypatch
) -> None:
  monkeypatch.delenv(tb.ENVIRONMENT_FLAG, raising=False)
  monkeypatch.delenv(tb.ENVIRONMENT_DIR, raising=False)
  writer = tb.open_for(tmp_path / "run")
  assert writer is not None
  assert writer.write_row({"loss": 1.0}, fallback_step=77) == 1
  directory = writer.directory
  writer.close()
  assert _tags(directory)["loss"] == [(77, 1.0)]
