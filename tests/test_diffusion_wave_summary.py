"""Focused tests for the wave-summary helpers.

The tail-descending flag decides whether the next wave extends the budget, and the
sign-consistency counts decide whether an arm may be called a win, so both are
pinned here.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

SCRIPT = (
  pathlib.Path(__file__).resolve().parents[1]
  / "scripts"
  / "experiments"
  / "diffusion_wave_summary"
  / "report.py"
)


def _module():
  spec = importlib.util.spec_from_file_location("wave_summary_under_test", SCRIPT)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_curve_summary_detects_a_u_shaped_curve() -> None:
  module = _module()
  # Descends to a minimum at epoch 40 of 100, then overfits.
  curve = {e: 0.05 + abs(e - 40) * 0.0004 for e in range(1, 101)}
  summary = module.curve_summary(curve)
  assert summary["best_epoch"] == 40
  assert summary["best_budget_fraction"] == pytest.approx(0.40)
  assert summary["final_over_best"] > 1.1
  assert summary["tail_still_descending"] is False


def test_curve_summary_detects_a_still_descending_tail() -> None:
  module = _module()
  curve = {e: 1.0 / e for e in range(1, 101)}
  summary = module.curve_summary(curve)
  assert summary["best_epoch"] == 100
  assert summary["tail_still_descending"] is True
  assert summary["final_over_best"] == pytest.approx(1.0)


def test_curve_summary_rejects_an_empty_curve() -> None:
  module = _module()
  with pytest.raises(ValueError, match="no validation values"):
    module.curve_summary({})


def _groups() -> dict:
  return {
    "000|clean": {"ref": {"normalized state": 1.0}, "arm": {"normalized state": 0.5}},
    "000|ou": {"ref": {"normalized state": 1.0}, "arm": {"normalized state": 0.5}},
    "001|clean": {"ref": {"normalized state": 1.0}, "arm": {"normalized state": 1.4}},
    "001|ou": {"ref": {"normalized state": 1.0}, "arm": {"normalized state": 1.4}},
  }


def test_per_motion_deltas_combine_phases() -> None:
  module = _module()
  deltas = module.per_motion_deltas(
    _groups(), reference="ref", arm="arm", channel="normalized state"
  )
  assert deltas["000"] == pytest.approx(-0.5)
  assert deltas["001"] == pytest.approx(0.4)


def test_per_motion_deltas_weight_by_window_count() -> None:
  module = _module()
  weights = {"000|clean": 3, "000|ou": 1, "001|clean": 1, "001|ou": 1}
  deltas = module.per_motion_deltas(
    _groups(), reference="ref", arm="arm", channel="normalized state", weights=weights
  )
  # 000: (-0.5*3 + -0.5*1)/4 = -0.5; 001 unchanged by equal weights.
  assert deltas["000"] == pytest.approx(-0.5)
  assert deltas["001"] == pytest.approx(0.4)


def test_per_motion_deltas_rejects_a_same_arm_comparison() -> None:
  module = _module()
  with pytest.raises(ValueError, match="must differ"):
    module.per_motion_deltas(
      _groups(), reference="ref", arm="ref", channel="normalized state"
    )


def test_per_motion_deltas_reports_a_missing_arm() -> None:
  module = _module()
  with pytest.raises(ValueError, match="lacks arm"):
    module.per_motion_deltas(
      _groups(), reference="ref", arm="absent", channel="normalized state"
    )


def test_sign_consistency_counts_both_directions() -> None:
  module = _module()
  result = module.sign_consistency({"000": -0.5, "001": 0.4, "002": -0.1})
  assert result["motions"] == 3
  assert result["better"] == 2
  assert result["worse"] == 1
  assert result["best_motion"] == "000"
  assert result["worst_motion"] == "001"
  assert result["median_delta"] == pytest.approx(-0.1)


def test_sign_consistency_treats_tiny_deltas_as_unchanged() -> None:
  module = _module()
  result = module.sign_consistency({"a": 1e-15, "b": -1e-15}, tolerance=1e-12)
  assert result["better"] == 0
  assert result["worse"] == 0
  assert result["unchanged"] == 2
