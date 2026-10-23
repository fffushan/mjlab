"""Optional sidecar diagnostics for one flow-matching run.

The recorder is off by default.  It costs nothing when it is off: the trainer
holds ``None`` and calls no method.

When the environment variable ``MJLAB_FLOW_DIAGNOSTICS_DIR`` is set, the trainer
writes two kinds of evidence into that directory:

* ``updates.jsonl`` - one row per optimizer update.  It holds the per-block
  gradient norms, the parameter norm, and a decomposition of the loss by noise
  level, by window row and by state channel group.  These values are too bulky
  for ``metrics.jsonl``, which already grows by about 0.5 KB per update.
* ``captures/capture-<step>.npz`` - a ring buffer of the most recent
  micro-batches, written only when the loss crosses a trigger.  A capture stores
  the window indices, the two noise level tensors, both masks, the per-window
  loss and the noise generator state.  The state makes the draw reproducible, so
  the offline analyzer rebuilds the exact noisy input instead of storing it.

The recorder captures inputs rather than activations on purpose.  Attention
logits, LayerNorm input variance and AdamW moments need hooks that would slow
every step; the analyzer applies them offline to a captured batch instead.
"""

from __future__ import annotations

import json
import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

ENV_DIRECTORY = "MJLAB_FLOW_DIAGNOSTICS_DIR"
ENV_RING_SIZE = "MJLAB_FLOW_DIAGNOSTICS_RING"
ENV_TRIGGER_FACTOR = "MJLAB_FLOW_DIAGNOSTICS_TRIGGER"
ENV_BASELINE_WINDOW = "MJLAB_FLOW_DIAGNOSTICS_BASELINE"
ENV_LEVEL_BINS = "MJLAB_FLOW_DIAGNOSTICS_LEVEL_BINS"
ENV_MAX_CHECKPOINTS = "MJLAB_FLOW_DIAGNOSTICS_MAX_CHECKPOINTS"

_DEFAULT_RING_SIZE = 16
_DEFAULT_TRIGGER_FACTOR = 8.0
_DEFAULT_BASELINE_WINDOW = 100
_DEFAULT_LEVEL_BINS = 10
_PARAMETER_NORM_EVERY = 50
_DEFAULT_MAX_CHECKPOINTS = 5


class FlowDiagnosticsError(RuntimeError):
  """Raised when the sidecar recorder cannot honour its own settings."""


def _env_int(name: str, default: int, *, minimum: int) -> int:
  raw = os.environ.get(name)
  if raw is None or raw == "":
    return default
  try:
    value = int(raw)
  except ValueError as exc:
    raise FlowDiagnosticsError(f"{name} must be an integer, got {raw!r}") from exc
  if value < minimum:
    raise FlowDiagnosticsError(f"{name} must be at least {minimum}, got {value}")
  return value


def _env_float(name: str, default: float, *, minimum: float) -> float:
  raw = os.environ.get(name)
  if raw is None or raw == "":
    return default
  try:
    value = float(raw)
  except ValueError as exc:
    raise FlowDiagnosticsError(f"{name} must be a number, got {raw!r}") from exc
  if not math.isfinite(value) or value < minimum:
    raise FlowDiagnosticsError(f"{name} must be at least {minimum}, got {value}")
  return value


@dataclass(frozen=True)
class ChannelGroup:
  """One named contiguous channel range of the token dimension."""

  name: str
  start: int
  stop: int


def channel_groups(
  slices: Mapping[str, slice], *, state_dimension: int, token_dimension: int
) -> tuple[ChannelGroup, ...]:
  """Build channel groups from a packed-state layout plus the latent tail.

  ``slices`` must describe the state channels in order.  A layout that does not
  match ``state_dimension`` falls back to the two coarse groups, so a contract
  change cannot silently mislabel the loss.
  """
  groups: list[ChannelGroup] = []
  covered = 0
  for name, item in slices.items():
    start = 0 if item.start is None else int(item.start)
    stop = state_dimension if item.stop is None else int(item.stop)
    if start != covered or stop <= start or stop > state_dimension:
      groups = []
      break
    groups.append(ChannelGroup(str(name), start, stop))
    covered = stop
  if not groups or covered != state_dimension:
    groups = [ChannelGroup("state", 0, state_dimension)]
  if token_dimension > state_dimension:
    groups.append(ChannelGroup("latent", state_dimension, token_dimension))
  return tuple(groups)


def gradient_norms_by_block(model: torch.nn.Module) -> dict[str, float]:
  """Return the gradient L2 norm of each parameter block.

  Groups follow the module tree one level deep, and encoder layers are split
  individually, so a diverging run shows which block moved first.
  """
  totals: dict[str, float] = {}
  for name, parameter in model.named_parameters():
    gradient = parameter.grad
    if gradient is None:
      continue
    parts = name.split(".")
    if parts[0] == "encoder" and len(parts) > 2:
      block = ".".join(parts[:3])
    elif len(parts) > 1:
      block = parts[0]
    else:
      block = "root"
    value = float(gradient.detach().float().pow(2).sum())
    totals[block] = totals.get(block, 0.0) + value
  return {name: math.sqrt(value) for name, value in sorted(totals.items())}


def parameter_norm(model: torch.nn.Module) -> float:
  total = 0.0
  for parameter in model.parameters():
    total += float(parameter.detach().float().pow(2).sum())
  return math.sqrt(total)


def _bin_indices(levels: Tensor, *, level_max: int, bins: int) -> Tensor:
  """Map integer levels in ``[0, level_max]`` to ``bins + 1`` buckets.

  Bucket 0 is the exactly clean row.  The remaining buckets split the noised
  levels evenly.  A clean row is one draw out of ``level_max + 1``, so it needs
  its own bucket to stay visible.
  """
  if level_max <= 0:
    raise FlowDiagnosticsError("level_max must be positive")
  noised = levels > 0
  scaled = (levels - 1).clamp_min(0) * bins // level_max
  return torch.where(noised, scaled + 1, torch.zeros_like(levels))


@dataclass
class _BatchCapture:
  """One micro-batch, small enough to keep in a ring buffer."""

  update_index: int
  window_indices: np.ndarray
  state_levels: np.ndarray
  latent_levels: np.ndarray
  history_dropout: np.ndarray
  state_mask: np.ndarray
  window_loss: np.ndarray
  generator_state: np.ndarray


class FlowDiagnostics:
  """Sidecar recorder for one training run."""

  def __init__(
    self,
    directory: Path,
    *,
    state_dimension: int,
    token_dimension: int,
    window_steps: int,
    level_max: int,
    groups: Sequence[ChannelGroup],
    ring_size: int = _DEFAULT_RING_SIZE,
    trigger_factor: float = _DEFAULT_TRIGGER_FACTOR,
    baseline_window: int = _DEFAULT_BASELINE_WINDOW,
    level_bins: int = _DEFAULT_LEVEL_BINS,
    max_checkpoints: int = _DEFAULT_MAX_CHECKPOINTS,
  ) -> None:
    if state_dimension <= 0 or token_dimension <= state_dimension:
      raise FlowDiagnosticsError(
        "token_dimension must exceed a positive state_dimension"
      )
    if window_steps <= 0:
      raise FlowDiagnosticsError("window_steps must be positive")
    if ring_size < 1:
      raise FlowDiagnosticsError("ring_size must be at least 1")
    if trigger_factor <= 0.0:
      raise FlowDiagnosticsError("trigger_factor must be positive")
    if baseline_window < 1:
      raise FlowDiagnosticsError("baseline_window must be at least 1")
    if level_bins < 1:
      raise FlowDiagnosticsError("level_bins must be at least 1")
    self.directory = Path(directory)
    self.directory.mkdir(parents=True, exist_ok=True)
    (self.directory / "captures").mkdir(exist_ok=True)
    self.state_dimension = int(state_dimension)
    self.token_dimension = int(token_dimension)
    self.window_steps = int(window_steps)
    self.level_max = int(level_max)
    self.groups = tuple(groups)
    self.ring_size = int(ring_size)
    self.trigger_factor = float(trigger_factor)
    self.baseline_window = int(baseline_window)
    self.level_bins = int(level_bins)
    if max_checkpoints < 0:
      raise FlowDiagnosticsError("max_checkpoints must not be negative")
    self.max_checkpoints = int(max_checkpoints)
    self._checkpoints_written = 0
    self._pending_trigger: dict[str, Any] | None = None
    self.updates_path = self.directory / "updates.jsonl"
    self._ring: deque[_BatchCapture] = deque(maxlen=self.ring_size)
    self._pending: _BatchCapture | None = None
    self._losses: deque[float] = deque(maxlen=self.baseline_window)
    self._group_sums: dict[str, float] = {}
    self._bin_sums: dict[str, float] = {}
    self._bin_counts: dict[str, float] = {}
    self._row_sums: np.ndarray | None = None
    self._microbatches = 0
    self._triggered = 0

  @classmethod
  def from_environment(
    cls,
    *,
    state_dimension: int,
    token_dimension: int,
    window_steps: int,
    level_max: int,
    groups: Sequence[ChannelGroup],
  ) -> FlowDiagnostics | None:
    """Return a recorder when ``MJLAB_FLOW_DIAGNOSTICS_DIR`` is set."""
    raw = os.environ.get(ENV_DIRECTORY)
    if raw is None or raw == "":
      return None
    return cls(
      Path(raw),
      state_dimension=state_dimension,
      token_dimension=token_dimension,
      window_steps=window_steps,
      level_max=level_max,
      groups=groups,
      ring_size=_env_int(ENV_RING_SIZE, _DEFAULT_RING_SIZE, minimum=1),
      trigger_factor=_env_float(
        ENV_TRIGGER_FACTOR, _DEFAULT_TRIGGER_FACTOR, minimum=0.0
      ),
      baseline_window=_env_int(
        ENV_BASELINE_WINDOW, _DEFAULT_BASELINE_WINDOW, minimum=1
      ),
      level_bins=_env_int(ENV_LEVEL_BINS, _DEFAULT_LEVEL_BINS, minimum=1),
      max_checkpoints=_env_int(
        ENV_MAX_CHECKPOINTS, _DEFAULT_MAX_CHECKPOINTS, minimum=0
      ),
    )

  def begin_batch(self, *, update_index: int, window_indices: Tensor | None) -> None:
    """Record what the next forward pass is about to consume."""
    self._pending = _BatchCapture(
      update_index=int(update_index),
      window_indices=(
        np.zeros((0,), dtype=np.int64)
        if window_indices is None
        else window_indices.detach().to("cpu", torch.int64).numpy()
      ),
      state_levels=np.zeros((0,), dtype=np.int16),
      latent_levels=np.zeros((0,), dtype=np.int16),
      history_dropout=np.zeros((0,), dtype=bool),
      state_mask=np.zeros((0,), dtype=bool),
      window_loss=np.zeros((0,), dtype=np.float32),
      generator_state=np.zeros((0,), dtype=np.uint8),
    )

  def note_noise_state(self, generator: Any) -> None:
    """Record the generator state at the exact point the noise draw starts.

    The level draws consume this generator first, so a state taken before the
    forward pass would replay a different Gaussian draw.  Taking it here is what
    makes the offline rebuild bit-exact without storing the noise tensor.
    """
    if self._pending is None:
      return
    state = generator.get_state()
    self._pending.generator_state = (
      state.detach().to("cpu").numpy().copy()
      if isinstance(state, Tensor)
      else np.frombuffer(state, dtype=np.uint8).copy()
    )

  def observe_batch(
    self,
    *,
    prediction: Tensor,
    target: Tensor,
    state_levels: Tensor,
    latent_levels: Tensor,
    history_dropout: Tensor | None,
    state_mask: Tensor | None,
  ) -> None:
    """Accumulate the loss decomposition and close the pending capture."""
    with torch.no_grad():
      squared = (prediction.detach().float() - target.detach().float()).square()
      state_squared = squared[..., : self.state_dimension]
      latent_squared = squared[..., self.state_dimension :]
      self._microbatches += 1
      for group in self.groups:
        piece = squared[..., group.start : group.stop]
        self._group_sums[group.name] = self._group_sums.get(group.name, 0.0) + float(
          piece.mean()
        )
      state_bins = _bin_indices(
        state_levels.detach().to("cpu"),
        level_max=self.level_max,
        bins=self.level_bins,
      )
      latent_bins = _bin_indices(
        latent_levels.detach().to("cpu"),
        level_max=self.level_max,
        bins=self.level_bins,
      )
      state_cpu = state_squared.detach().to("cpu").double()
      latent_cpu = latent_squared.detach().to("cpu").double()
      for index in range(self.level_bins + 1):
        state_mask_bin = state_bins == index
        latent_mask_bin = latent_bins == index
        state_values = state_cpu[state_mask_bin]
        latent_values = latent_cpu[latent_mask_bin]
        for prefix, values in (("state", state_values), ("latent", latent_values)):
          count = float(values.numel())
          if count == 0.0:
            # An empty bucket stays out of the row instead of reporting NaN.
            continue
          key = f"{prefix}_level_{index}"
          self._bin_sums[key] = (
            self._bin_sums.get(key, 0.0) + float(values.mean()) * count
          )
          self._bin_counts[key] = self._bin_counts.get(key, 0.0) + count
      rows = squared.mean(dim=2).sum(dim=0).detach().to("cpu").double().numpy()
      self._row_sums = rows if self._row_sums is None else self._row_sums + rows
      if self._pending is not None:
        capture = self._pending
        self._pending = None
        capture.state_levels = _as_int16(state_levels)
        capture.latent_levels = _as_int16(latent_levels)
        capture.history_dropout = _as_bool(history_dropout)
        capture.state_mask = _as_bool(state_mask)
        capture.window_loss = (
          squared.mean(dim=(1, 2)).detach().to("cpu", torch.float32).numpy()
        )
        self._ring.append(capture)

  def would_trigger(self, *, update_index: int, loss: float) -> dict[str, Any] | None:
    """Decide whether this update triggers, before the optimizer step.

    The decision has to happen before the step for two reasons: the triggering
    checkpoint must hold the weights that produced the batch, or the replay
    cannot be checked against the recorded loss, and the checkpoint budget must
    be honoured before any file is written.
    """
    self._pending_trigger = self._check_trigger(update_index, loss, append=False)
    return self._pending_trigger

  def checkpoint_allowed(self) -> bool:
    """Return whether one more trigger checkpoint fits the budget."""
    return self._checkpoints_written < self.max_checkpoints

  def checkpoint_written(self) -> None:
    self._checkpoints_written += 1

  def end_update(
    self,
    *,
    update_index: int,
    learning_rate: float,
    loss: float,
    gradient_norm: float,
    model: torch.nn.Module,
    blocks: Mapping[str, float] | None = None,
  ) -> dict[str, Any] | None:
    """Write one update row and report a trigger when the loss jumps."""
    row: dict[str, Any] = {
      "kind": "diagnostics",
      "update_index": int(update_index),
      "learning_rate": float(learning_rate),
      "loss": float(loss),
      "gradient_norm": float(gradient_norm),
      "microbatches": int(self._microbatches),
      "microbatch_loss": float(loss),
    }
    block_norms = gradient_norms_by_block(model) if blocks is None else blocks
    for name, value in block_norms.items():
      row[f"gradient_norm_block/{name}"] = float(value)
    if not self._group_sums:
      raise FlowDiagnosticsError("no micro-batch was observed before this update")
    for name, total in sorted(self._group_sums.items()):
      row[f"loss_channel_group/{name}"] = total / float(self._microbatches)
    for key, total in sorted(self._bin_sums.items()):
      count = self._bin_counts.get(key, 0.0)
      row[f"loss_by_level/{key}"] = total / count if count else float("nan")
    if self._row_sums is not None:
      per_row = self._row_sums / float(self._microbatches)
      for step, value in enumerate(per_row.tolist()):
        row[f"loss_by_row/{step}"] = float(value)
    if update_index % _PARAMETER_NORM_EVERY == 0:
      row["parameter_norm"] = parameter_norm(model)
    self._group_sums = {}
    self._bin_sums = {}
    self._bin_counts = {}
    self._row_sums = None
    self._microbatches = 0
    with self.updates_path.open("a", encoding="utf-8") as handle:
      handle.write(json.dumps(row, sort_keys=True) + "\n")
    self._losses.append(float(loss))
    trigger = self._pending_trigger
    self._pending_trigger = None
    if trigger is not None:
      self._triggered += 1
    return trigger

  def _check_trigger(
    self, update_index: int, loss: float, *, append: bool = True
  ) -> dict[str, Any] | None:
    """Decide a trigger without consuming the baseline sample.

    ``would_trigger`` asks before the optimizer step and leaves the sample for
    ``end_update`` to append, so one update contributes exactly one baseline
    entry no matter how many times the decision is inspected.
    """
    baseline = (
      float(np.median(np.asarray(self._losses, dtype=float))) if self._losses else 0.0
    )
    previous = len(self._losses) + (0 if append else 1)
    if append:
      self._losses.append(float(loss))
    if previous < self.baseline_window or baseline <= 0.0:
      return None
    if not math.isfinite(loss) or loss <= self.trigger_factor * baseline:
      return None
    return {
      "kind": "trigger",
      "update_index": int(update_index),
      "loss": float(loss),
      "baseline": baseline,
      "trigger_factor": self.trigger_factor,
      "triggered": self._triggered + 1,
    }

  def dump(self, *, update_index: int, reason: str) -> Path:
    """Write the ring buffer to one compressed npz file."""
    if not self._ring:
      raise FlowDiagnosticsError("the capture ring is empty")
    path = self.directory / "captures" / f"capture-{int(update_index):07d}.npz"
    payload: dict[str, np.ndarray] = {
      "update_index": np.asarray(
        [item.update_index for item in self._ring], dtype=np.int64
      ),
      "state_dimension": np.asarray([self.state_dimension], dtype=np.int64),
      "token_dimension": np.asarray([self.token_dimension], dtype=np.int64),
    }
    # Row-indexed fields concatenate.  The generator state and the row count
    # stay per micro-batch, because the replay must redraw each batch from its
    # own state or it cannot reproduce the same noise.
    for field in (
      "window_indices",
      "state_levels",
      "latent_levels",
      "history_dropout",
      "state_mask",
      "window_loss",
    ):
      payload[field] = np.concatenate(
        [getattr(item, field) for item in self._ring], axis=0
      )
    payload["generator_state"] = np.stack([item.generator_state for item in self._ring])
    payload["microbatch_rows"] = np.asarray(
      [len(item.window_indices) for item in self._ring], dtype=np.int64
    )
    payload["microbatch_update_index"] = np.asarray(
      [item.update_index for item in self._ring], dtype=np.int64
    )
    np.savez_compressed(path, **payload)
    (self.directory / "captures" / f"capture-{int(update_index):07d}.json").write_text(
      json.dumps(
        {
          "reason": reason,
          "update_index": int(update_index),
          "microbatches": len(self._ring),
          "trigger_factor": self.trigger_factor,
          "baseline_window": self.baseline_window,
        },
        sort_keys=True,
      )
      + "\n",
      encoding="utf-8",
    )
    return path


def _as_int16(values: Tensor | None) -> np.ndarray:
  if values is None:
    return np.zeros((0,), dtype=np.int16)
  return values.detach().to("cpu", torch.int16).numpy()


def _as_bool(values: Tensor | None) -> np.ndarray:
  if values is None:
    return np.zeros((0,), dtype=bool)
  return values.detach().to("cpu", torch.bool).numpy()


__all__ = [
  "ChannelGroup",
  "ENV_BASELINE_WINDOW",
  "ENV_DIRECTORY",
  "ENV_LEVEL_BINS",
  "ENV_RING_SIZE",
  "ENV_TRIGGER_FACTOR",
  "FlowDiagnostics",
  "FlowDiagnosticsError",
  "channel_groups",
  "gradient_norms_by_block",
  "parameter_norm",
]
