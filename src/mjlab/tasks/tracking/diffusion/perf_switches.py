"""Environment-gated performance switches for the diffusion trainer.

The l1, l4 and l6 switches default to on. Set their environment variable to
``0`` to disable one switch for a controlled comparison. Other switches remain
off unless explicitly enabled. None of these switches changes the training
objective. The l4 switch is value-preserving by construction. The l1 switch
preserves coverage but changes the timing of its finiteness reads. The l6
switch can change floating-point operation order.

Switches
--------
``MJLAB_DIFFUSION_FINITE_COMBINED``
    Set to ``0`` to restore per-parameter finiteness reads. The default combines
    the checks into one read per update.
``MJLAB_DIFFUSION_METRICS_SINGLE_READ=1``
    Read all metric scalars in one transfer instead of one per metric. The
    values are identical, because only the read is batched.
``MJLAB_DIFFUSION_LOADER_WORKERS=N``
    Build batches in N worker processes instead of the main thread.
``MJLAB_DIFFUSION_EMA_FOREACH``
    Set to ``0`` to restore per-tensor EMA updates. The default uses batched
    foreach updates.
``MJLAB_DIFFUSION_COMPILED_FORWARD``
    Set to ``0`` to restore the normal model call. The default uses a compiled
    callable over the model.
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

  finite_combined: bool = True
  metrics_single_read: bool = False
  loader_workers: int = 0
  ema_foreach: bool = True
  compiled_forward: bool = True
  label: str = "finite+ema4+compiled"

  @property
  def active(self) -> bool:
    return (
      self.finite_combined
      or self.metrics_single_read
      or self.loader_workers > 0
      or self.ema_foreach
      or self.compiled_forward
    )

  @classmethod
  def from_environment(cls) -> "PerfSwitches":
    finite_combined = _flag("MJLAB_DIFFUSION_FINITE_COMBINED", default=True)
    metrics_single_read = _flag("MJLAB_DIFFUSION_METRICS_SINGLE_READ")
    loader_workers = _count("MJLAB_DIFFUSION_LOADER_WORKERS")
    ema_foreach = _flag("MJLAB_DIFFUSION_EMA_FOREACH", default=True)
    compiled_forward = _flag("MJLAB_DIFFUSION_COMPILED_FORWARD", default=True)
    label = os.environ.get("MJLAB_DIFFUSION_PERF_LABEL", "").strip()
    if not label:
      parts: list[str] = []
      if finite_combined:
        parts.append("finite")
      if metrics_single_read:
        parts.append("metrics1")
      if loader_workers:
        parts.append(f"workers{loader_workers}")
      if ema_foreach:
        parts.append("ema4")
      if compiled_forward:
        parts.append("compiled")
      label = "+".join(parts) if parts else "default"
    return cls(
      finite_combined=finite_combined,
      metrics_single_read=metrics_single_read,
      loader_workers=loader_workers,
      ema_foreach=ema_foreach,
      compiled_forward=compiled_forward,
      label=label,
    )
