"""Optional TensorBoard mirror for the flow and diffusion metric rows.

The flow and diffusion trainers write one JSON object per update to
``metrics.jsonl``.  The PPO runner has a TensorBoard writer, but these trainers
did not, so every training curve had to be read out of the JSONL.  This module
mirrors the same rows into event files as they are written, so the JSONL stays
the source of truth and the event file becomes a view of it.

The writer is not part of the training config.  The config hash is part of the
resume identity, so adding a field there would reject every existing
checkpoint.  Configuration therefore comes from the environment:

``MJLAB_TENSORBOARD``
  ``0`` disables the writer.  Any other non-empty value enables it.  Unset
  enables it, because a run that silently logs nothing is worse than a run
  that writes a few hundred kilobytes nobody opened.
``MJLAB_TENSORBOARD_DIR``
  Event directory.  Defaults to ``<output_dir>/tensorboard``.

If the ``tensorboard`` package is missing the writer disables itself and warns
once, so training never depends on an optional import.
"""

from __future__ import annotations

import math
import os
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ENVIRONMENT_FLAG = "MJLAB_TENSORBOARD"
ENVIRONMENT_DIR = "MJLAB_TENSORBOARD_DIR"

_WARNED = False


def _scalar(value: Any) -> float | None:
  """Return a finite float for a numeric value, else ``None``."""
  if isinstance(value, bool):
    return float(value)
  if isinstance(value, (int, float)):
    number = float(value)
    return number if math.isfinite(number) else None
  return None


def flatten_row(prefix: str, value: Any) -> list[tuple[str, float]]:
  """Flatten a metric row into ``(tag, value)`` pairs.

  Nested objects become ``eval_model/loss`` style tags, so the evaluation
  block written at an epoch boundary lands beside the per-update curves.
  Strings, ``None`` and non-finite numbers are dropped: a diverged run can
  record ``inf`` and TensorBoard cannot plot it.
  """
  if isinstance(value, Mapping):
    pairs: list[tuple[str, float]] = []
    for key, item in value.items():
      tag = f"{prefix}/{key}" if prefix else str(key)
      pairs.extend(flatten_row(tag, item))
    return pairs
  number = _scalar(value)
  return [(prefix, number)] if number is not None and prefix else []


class RunTensorboard:
  """Mirror metric rows into TensorBoard event files."""

  def __init__(self, writer: Any, directory: Path) -> None:
    self._writer = writer
    self.directory = directory
    self.scalars_written = 0

  @classmethod
  def from_environment(cls, output_dir: Path) -> "RunTensorboard | None":
    """Build a writer for ``output_dir``, or ``None`` when it is not wanted."""
    global _WARNED
    flag = os.environ.get(ENVIRONMENT_FLAG)
    if flag is not None and flag.strip() == "0":
      return None
    configured = os.environ.get(ENVIRONMENT_DIR)
    directory = Path(configured) if configured else output_dir / "tensorboard"
    try:
      from torch.utils.tensorboard import SummaryWriter
    except ImportError:
      if not _WARNED:
        _WARNED = True
        warnings.warn(
          "tensorboard is not installed; training continues without event files",
          RuntimeWarning,
          stacklevel=2,
        )
      return None
    try:
      directory.mkdir(parents=True, exist_ok=True)
      writer = SummaryWriter(log_dir=str(directory), flush_secs=10)
    except OSError as exc:
      if not _WARNED:
        _WARNED = True
        warnings.warn(
          f"could not open a TensorBoard writer at {directory}: {exc};"
          " training continues without event files",
          RuntimeWarning,
          stacklevel=2,
        )
      return None
    return cls(writer, directory)

  def row_step(self, row: Mapping[str, Any], fallback: int) -> int:
    """Return the step a row belongs to."""
    for key in ("global_step", "update", "step"):
      number = _scalar(row.get(key))
      if number is not None:
        return int(number)
    return fallback

  def write_row(self, row: Mapping[str, Any], *, fallback_step: int = 0) -> int:
    """Mirror one metric row and return the number of scalars written."""
    step = self.row_step(row, fallback_step)
    written = 0
    for tag, value in flatten_row("", row):
      self._writer.add_scalar(tag, value, step)
      written += 1
    self.scalars_written += written
    return written

  def close(self) -> None:
    """Flush and close the underlying writer."""
    self._writer.flush()
    self._writer.close()

  def __enter__(self) -> "RunTensorboard":
    return self

  def __exit__(self, *exc_info: object) -> None:
    self.close()


def open_for(output_dir: Path) -> "RunTensorboard | None":
  """Short alias used by the trainers."""
  return RunTensorboard.from_environment(output_dir)
