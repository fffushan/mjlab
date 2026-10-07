"""Environment-gated performance switches for the diffusion trainer.

Every switch defaults to the current behaviour, so an unset environment leaves
the training path unchanged and the arms are never affected. Each switch exists
to measure one hypothesis about the CPU cost of one step. None of them changes
the training objective, and two of them are value-preserving by construction.

Switches
--------
``MJLAB_DIFFUSION_FINITE_COMBINED=1``
    Replace the per-parameter finiteness sweeps with one combined check per
    update. Coverage is preserved; only the number of dispatches falls.
``MJLAB_DIFFUSION_METRICS_SINGLE_READ=1``
    Read all metric scalars in one transfer instead of one per metric. The
    values are identical, because only the read is batched.
``MJLAB_DIFFUSION_LOADER_WORKERS=N``
    Build batches in N worker processes instead of the main thread.
``MJLAB_DIFFUSION_PERF_LABEL=<name>``
    Label for logs and artifacts. Derived from the switches when absent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_TRUTHY = frozenset(("1", "true", "yes", "on"))


def _flag(name: str, default: bool = False) -> bool:
  raw = os.environ.get(name)
  if raw is None:
    return default
  return raw.strip().lower() in _TRUTHY


def _count(name: str, default: int = 0) -> int:
  raw = os.environ.get(name)
  if raw is None or not raw.strip():
    return default
  value = int(raw)
  if value < 0:
    raise ValueError(f"{name} must not be negative; got {value}")
  return value


@dataclass(frozen=True)
class PerfSwitches:
  """Resolved switches for one run."""

  finite_combined: bool = False
  metrics_single_read: bool = False
  loader_workers: int = 0
  label: str = "default"

  @property
  def active(self) -> bool:
    return self.finite_combined or self.metrics_single_read or self.loader_workers > 0

  @classmethod
  def from_environment(cls) -> "PerfSwitches":
    finite_combined = _flag("MJLAB_DIFFUSION_FINITE_COMBINED")
    metrics_single_read = _flag("MJLAB_DIFFUSION_METRICS_SINGLE_READ")
    loader_workers = _count("MJLAB_DIFFUSION_LOADER_WORKERS")
    label = os.environ.get("MJLAB_DIFFUSION_PERF_LABEL", "").strip()
    if not label:
      parts: list[str] = []
      if finite_combined:
        parts.append("finite")
      if metrics_single_read:
        parts.append("metrics1")
      if loader_workers:
        parts.append(f"workers{loader_workers}")
      label = "+".join(parts) if parts else "default"
    return cls(
      finite_combined=finite_combined,
      metrics_single_read=metrics_single_read,
      loader_workers=loader_workers,
      label=label,
    )
