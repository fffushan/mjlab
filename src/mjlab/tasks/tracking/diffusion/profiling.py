"""Bounded, default-off step profiling for the diffusion trainer.

Purpose: identify which statement dominates a training step when the main
thread is saturated and the GPU is mostly idle, without an external sampler.
An external sampler (``py-spy``) cannot attach in the managed container, so the
profiler must run inside the process.

The profiler is gated by environment variables and is inert unless
``MJLAB_DIFFUSION_PROFILE`` is set, so the default training path is unchanged.

Environment
-----------
``MJLAB_DIFFUSION_PROFILE``   ``cprofile``, ``torch``, or ``both``. Unset = off.
``MJLAB_DIFFUSION_PROFILE_STEPS``   Number of steps to record (default 200).
``MJLAB_DIFFUSION_PROFILE_WARMUP``  Steps to skip first (default 20).
``MJLAB_DIFFUSION_PROFILE_OUT``     Output directory (default: output_dir/profile).
"""

from __future__ import annotations

import cProfile
import json
import os
import pstats
import time
from pathlib import Path
from typing import Any

_VALID_KINDS = ("cprofile", "torch", "both")


class StepProfiler:
  """Record a bounded steady-state window of training steps.

  ``on_step_start`` and ``on_step_end`` bracket one step. Warm-up steps are
  counted and skipped, so start-up cost and first-step autotuning stay out of
  the recorded window.
  """

  def __init__(
    self,
    kind: str | None,
    *,
    steps: int,
    warmup: int,
    out_dir: Path,
  ) -> None:
    if kind is not None and kind not in _VALID_KINDS:
      raise ValueError(
        f"MJLAB_DIFFUSION_PROFILE must be one of {_VALID_KINDS}; got {kind!r}"
      )
    self.kind = kind
    self.steps = max(0, steps)
    self.warmup = max(0, warmup)
    self.out_dir = out_dir
    self.active = False
    self.records = 0
    self.skipped = 0
    self._closed = False
    self._last_start = 0.0
    self._step_seconds: list[float] = []
    self._cprofile: cProfile.Profile | None = None
    self._torch_profiler: Any | None = None
    self._torch_ctx: Any | None = None

  @classmethod
  def from_environment(cls, output_dir: Path) -> "StepProfiler":
    """Build a profiler from the environment. Absent settings mean off."""
    kind = os.environ.get("MJLAB_DIFFUSION_PROFILE") or None
    steps = int(os.environ.get("MJLAB_DIFFUSION_PROFILE_STEPS", "200"))
    warmup = int(os.environ.get("MJLAB_DIFFUSION_PROFILE_WARMUP", "20"))
    out = os.environ.get("MJLAB_DIFFUSION_PROFILE_OUT")
    out_dir = Path(out) if out else output_dir / "profile"
    return cls(kind, steps=steps, warmup=warmup, out_dir=out_dir)

  def on_step_start(self, step: int) -> None:
    """Begin recording when the warm-up is over; ignore later steps."""
    if self.kind is None or self.records >= self.steps:
      return
    if self.skipped < self.warmup:
      self.skipped += 1
      return
    # Open the recorders once per window, but restart the clock on EVERY
    # recorded step: a clock set only once measures a cumulative delta and
    # inflates the reported step time.
    if not self.active:
      self.active = True
      self.out_dir.mkdir(parents=True, exist_ok=True)
      if self.kind in ("cprofile", "both"):
        self._cprofile = cProfile.Profile()
        self._cprofile.enable()
      if self.kind in ("torch", "both"):
        import torch

        if torch.cuda.is_available():
          self._torch_profiler = torch.profiler.profile(
            activities=[
              torch.profiler.ProfilerActivity.CPU,
              torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            with_stack=True,
          )
          self._torch_ctx = self._torch_profiler.__enter__()
    self._last_start = time.perf_counter()

  def on_step_end(self) -> None:
    """Close out one recorded step and stop when the budget is reached."""
    if not self.active:
      return
    self._step_seconds.append(time.perf_counter() - self._last_start)
    self.records += 1
    if self.records >= self.steps:
      self._stop()

  def _stop(self) -> None:
    """Close the recorders exactly once, so a later call cannot raise."""
    if self._closed:
      return
    self._closed = True
    if self.kind in ("cprofile", "both") and self._cprofile is not None:
      self._cprofile.disable()
    if self.kind in ("torch", "both") and self._torch_ctx is not None:
      self._torch_ctx.__exit__(None, None, None)
    self.active = False

  def finish(self) -> None:
    """Write profile artifacts. Safe to call when profiling never started."""
    if self.kind is None or self.records == 0:
      return
    self._stop()
    self.out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    if self.kind in ("cprofile", "both") and self._cprofile is not None:
      raw = self.out_dir / "cprofile.prof"
      self._cprofile.dump_stats(str(raw))
      text = self.out_dir / "cprofile.txt"
      with text.open("w", encoding="utf-8") as handle:
        stats = pstats.Stats(self._cprofile, stream=handle)
        stats.sort_stats("tottime").print_stats(45)
      written += [raw.name, text.name]
    if self.kind in ("torch", "both") and self._torch_profiler is not None:
      table_path = self.out_dir / "torch_key_averages.txt"
      try:
        table_path.write_text(
          self._torch_profiler.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=40
          ),
          encoding="utf-8",
        )
        trace = self.out_dir / "torch_trace.json"
        self._torch_profiler.export_chrome_trace(str(trace))
        written += [table_path.name, trace.name]
      except Exception as error:  # pragma: no cover - platform dependent
        (self.out_dir / "torch_export_error.txt").write_text(
          f"{type(error).__name__}: {error}\n", encoding="utf-8"
        )
        written.append("torch_export_error.txt")
    total = sum(self._step_seconds)
    summary = {
      "kind": self.kind,
      "recorded_steps": self.records,
      "warmup_steps_skipped": self.skipped,
      "step_seconds_total": total,
      "step_seconds_mean": total / self.records,
      "step_seconds_min": min(self._step_seconds),
      "step_seconds_max": max(self._step_seconds),
      "artifacts": written,
    }
    (self.out_dir / "profile_summary.json").write_text(
      json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
      f"profile: kind={self.kind} steps={self.records} "
      f"mean_step={summary['step_seconds_mean'] * 1000:.1f} ms "
      f"out={self.out_dir}",
      flush=True,
    )
