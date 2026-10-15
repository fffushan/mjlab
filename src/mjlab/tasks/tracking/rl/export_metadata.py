"""Deploy-export metadata that the X2 controller needs but the term list cannot carry.

The X2 controller compiles a fixed observation plan from the export metadata and derives
every term width itself. One fact is not derivable: whether the reference preview is
enabled and how far ahead it reads. :data:`MOTION_LOOKAHEAD_STEPS_KEY` carries that fact.

``motion_lookahead`` appears in ``observation_names`` for both a disabled and an enabled
preview, so the term list alone cannot distinguish a zero-width term from a live one. The
controller rejects a layout it cannot derive rather than guessing, so the export must
state the offset explicitly.
"""

from __future__ import annotations

import math
from typing import Any, Protocol, Sequence

MOTION_LOOKAHEAD_STEPS_KEY = "motion_lookahead_steps"
"""Export key naming the preview offset in frames.

An absent key means a disabled preview, which keeps every export written before this key
existed valid. A present key must be a non-negative integer.
"""

MOTION_REFERENCE_FPS = 50.0
"""The only reference rate the X2 deploy contract accepts."""


class _MotionMetadataSource(Protocol):
  """The part of a motion command cfg that the deploy metadata needs."""

  anchor_body_name: str
  body_names: Sequence[str]
  lookahead_s: float


class DeployExportError(ValueError):
  """The export cannot describe the trained policy unambiguously."""


def resolve_lookahead_steps(lookahead_s: float, fps: float) -> int:
  """Preview offset in frames, or ``0`` when the preview is disabled.

  This mirrors the training command exactly, so the export and the trained policy read
  the same reference frame. The command computes
  ``max(1, ceil(lookahead_s * fps))`` for a positive preview.
  """
  if not math.isfinite(lookahead_s):
    raise DeployExportError("lookahead_s must be finite")
  if not math.isfinite(fps) or fps <= 0.0:
    raise DeployExportError("reference fps must be finite and positive")
  if lookahead_s <= 0.0:
    return 0
  return max(1, math.ceil(lookahead_s * fps))


def command_lookahead_steps(command: Any) -> int:
  """One preview offset for the whole export, or ``0`` when the preview is disabled.

  One export metadata value must describe every rollout row, and the controller reads one
  offset. This fails closed when the rows disagree, which happens when the reference
  clips do not share one rate.
  """
  single = getattr(command, "lookahead_steps", None)
  if isinstance(single, int):
    return single

  rows = getattr(command, "row_lookahead_steps", None)
  if rows is None:
    raise DeployExportError(
      "motion command exposes neither lookahead_steps nor row_lookahead_steps"
    )
  unique = sorted({int(value) for value in rows.reshape(-1).tolist()})
  if len(unique) != 1:
    raise DeployExportError(
      "rollout rows disagree on the preview offset "
      f"{unique}; one export cannot declare them all"
    )
  return unique[0]


def motion_export_metadata(
  motion_cfg: _MotionMetadataSource, lookahead_steps: int
) -> dict[str, object]:
  """Metadata keys describing the reference contract of one tracking export.

  ``anchor_body_name`` and ``body_names`` select the reference frame and the reference
  rows. ``MOTION_LOOKAHEAD_STEPS_KEY`` is always written, including ``0``, so an export
  states its own layout instead of relying on a reader default.
  """
  if lookahead_steps < 0:
    raise DeployExportError("lookahead steps must not be negative")
  if not motion_cfg.anchor_body_name:
    raise DeployExportError("the motion command declares no anchor body")
  if not motion_cfg.body_names:
    raise DeployExportError("the motion command declares no reference body names")
  return {
    "anchor_body_name": motion_cfg.anchor_body_name,
    "body_names": list(motion_cfg.body_names),
    MOTION_LOOKAHEAD_STEPS_KEY: int(lookahead_steps),
  }
