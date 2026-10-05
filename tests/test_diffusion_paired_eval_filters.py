"""Focused tests for the paired-evaluator window-subset filters.

The frame math is load-bearing for the motion-001 jump analysis: it decides which
windows count as "near the jump", so a silent off-by-one or a wrong integer group
from `pair_id` would produce a plausible but wrong conclusion.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

SCRIPT = (
  pathlib.Path(__file__).resolve().parents[1]
  / "scripts"
  / "experiments"
  / "diffusion_paired_eval"
  / "paired_eval.py"
)


def _module():
  spec = importlib.util.spec_from_file_location("paired_eval_under_test", SCRIPT)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class _Record:
  def __init__(self, pair_id: str, start_tick: int) -> None:
    self.pair_id = pair_id
    self.start_tick = start_tick


def test_estimated_reference_frame_uses_last_integer_group() -> None:
  module = _module()
  # `motion:seed:start_frame` -> start frame is 226, not the seed or motion id.
  assert module.estimated_reference_frame(_Record("tennis_000:0:226", 10)) == 236
  assert module.estimated_reference_frame(_Record("tennis_001:7:33", 0)) == 33
  # A suffixed motion name must still resolve to its start frame.
  assert module.estimated_reference_frame(_Record("tennis_002_ss:3:271", 5)) == 276


def test_estimated_reference_frame_marks_missing_numbers() -> None:
  module = _module()
  assert module.estimated_reference_frame(_Record("no-numbers", 4)) == 3


def test_in_frame_range_is_inclusive() -> None:
  module = _module()
  record = _Record("tennis_001:0:169", 0)  # frame 169
  assert module.in_frame_range(record, "153:183")
  assert module.in_frame_range(record, "169:169")
  assert not module.in_frame_range(record, "170:200")
  assert not module.in_frame_range(record, "0:168")


def test_in_frame_range_rejects_malformed_spec() -> None:
  module = _module()
  with pytest.raises(ValueError):
    module.in_frame_range(_Record("tennis_001:0:169", 0), "153")
