"""Tests for the deploy-export metadata of the X2 tracking family."""

import math
from dataclasses import dataclass

import pytest
import torch

from mjlab.tasks.tracking.rl.export_metadata import (
  MOTION_LOOKAHEAD_STEPS_KEY,
  DeployExportError,
  command_lookahead_steps,
  motion_export_metadata,
  resolve_lookahead_steps,
)


@dataclass
class _Cfg:
  anchor_body_name: str = "torso_link"
  body_names: list[str] | None = None
  lookahead_s: float = 0.0

  def __post_init__(self):
    if self.body_names is None:
      self.body_names = ["pelvis", "torso_link"]


class _SingleCommand:
  """Stands in for the single-motion command, which stores one resolved offset."""

  def __init__(self, steps: int):
    self.lookahead_steps = steps


class _MultiCommand:
  """Stands in for the multi-motion command, which resolves one offset per row."""

  def __init__(self, steps: list[int]):
    self.row_lookahead_steps = torch.tensor(steps, dtype=torch.long)


def test_preview_offset_matches_the_training_formula():
  """The export must read the same reference frame the policy trained on."""
  # The command computes max(1, ceil(lookahead_s * fps)) for a positive preview.
  assert resolve_lookahead_steps(0.2, 50.0) == 10
  assert resolve_lookahead_steps(0.1, 50.0) == 5
  assert resolve_lookahead_steps(0.02, 50.0) == 1
  # A sub-frame preview still reads one frame, never zero.
  assert resolve_lookahead_steps(0.001, 50.0) == 1
  # A rounding case that would be 3 with floor and 4 with ceil.
  assert resolve_lookahead_steps(0.05, 50.0) == math.ceil(0.05 * 50.0) == 3


def test_disabled_preview_reports_zero_offset():
  assert resolve_lookahead_steps(0.0, 50.0) == 0
  assert resolve_lookahead_steps(-1.0, 50.0) == 0


def test_non_finite_preview_is_refused():
  for value in (float("nan"), float("inf"), float("-inf")):
    with pytest.raises(DeployExportError):
      resolve_lookahead_steps(value, 50.0)
  with pytest.raises(DeployExportError):
    resolve_lookahead_steps(0.2, 0.0)


def test_single_motion_command_reports_its_resolved_offset():
  assert command_lookahead_steps(_SingleCommand(0)) == 0
  assert command_lookahead_steps(_SingleCommand(10)) == 10


def test_multi_motion_command_reports_the_shared_row_offset():
  assert command_lookahead_steps(_MultiCommand([10, 10, 10])) == 10
  assert command_lookahead_steps(_MultiCommand([0, 0])) == 0


def test_rows_that_disagree_fail_closed():
  """One export declares one offset, so mixed row rates are refused, not averaged."""
  with pytest.raises(DeployExportError, match="disagree"):
    command_lookahead_steps(_MultiCommand([10, 10, 11]))


def test_a_command_without_a_preview_offset_is_refused():
  class _Unknown:
    pass

  with pytest.raises(DeployExportError):
    command_lookahead_steps(_Unknown())


def test_metadata_states_the_layout_including_a_disabled_preview():
  metadata = motion_export_metadata(_Cfg(lookahead_s=0.2), 10)
  assert metadata == {
    "anchor_body_name": "torso_link",
    "body_names": ["pelvis", "torso_link"],
    MOTION_LOOKAHEAD_STEPS_KEY: 10,
  }
  # A disabled preview is still written, so an export states its own layout instead
  # of relying on a reader default.
  disabled = motion_export_metadata(_Cfg(lookahead_s=0.0), 0)
  assert disabled[MOTION_LOOKAHEAD_STEPS_KEY] == 0


def test_metadata_copies_the_body_names_instead_of_aliasing_them():
  cfg = _Cfg()
  metadata = motion_export_metadata(cfg, 0)
  metadata["body_names"].append("extra")  # type: ignore[union-attr]
  assert cfg.body_names == ["pelvis", "torso_link"]


def test_metadata_refuses_a_missing_anchor_or_body_list():
  with pytest.raises(DeployExportError, match="anchor"):
    motion_export_metadata(_Cfg(anchor_body_name=""), 0)
  with pytest.raises(DeployExportError, match="body names"):
    motion_export_metadata(_Cfg(body_names=[]), 0)
  with pytest.raises(DeployExportError, match="negative"):
    motion_export_metadata(_Cfg(), -1)
