"""Circular buffer for storing a history of batched tensor data.

Understanding Dimensions
========================

Internal storage shape: (max_len, batch_size, ...)
                         ↑         ↑
                         time      environments
                         axis      axis

External view (buffer property): (batch_size, max_len, ...)
                                  ↑           ↑
                                  environments time
                                  axis        axis

Why different?
  - Internal: time-first for clean API (self._buffer[pointer] = data)
  - External: batch-first for typical use (iterate over environments)

Backfill Behavior
=================

When you first append to a new or reset batch row, that first value is copied
to ALL history slots for that row:

  buffer = CircularBuffer(max_len=3, batch_size=2, device="cpu")
  buffer.append(torch.tensor([[5.0], [10.0]]))

  # buffer.buffer contains (shape: 2, 3, 1):
  # Batch 0: [5.0, 5.0, 5.0]   <- all slots filled with first value
  # Batch 1: [10.0, 10.0, 10.0] <- all slots filled with first value

Why backfill?
  You always have valid data. If training expects 3 frames of history, you
  don't want garbage/zeros for the first 2 timesteps.

Per-Batch Reset
===============

Reset affects specific batch rows. The circular pointer advances globally, but
reset rows get "first-append" treatment on their next write:

  buffer = CircularBuffer(max_len=3, batch_size=3, device="cpu")

  buffer.append(torch.tensor([[1.0], [10.0], [100.0]]))   # t0
  buffer.append(torch.tensor([[2.0], [20.0], [200.0]]))   # t1
  buffer.append(torch.tensor([[3.0], [30.0], [300.0]]))   # t2

  # buffer.buffer:
  # Batch 0: [1.0, 2.0, 3.0]
  # Batch 1: [10.0, 20.0, 30.0]
  # Batch 2: [100.0, 200.0, 300.0]

  buffer.reset(batch_ids=[1])  # Reset only batch 1

  # buffer.buffer after reset:
  # Batch 0: [1.0, 2.0, 3.0]       <- unchanged
  # Batch 1: [0.0, 0.0, 0.0]       <- zeroed
  # Batch 2: [100.0, 200.0, 300.0] <- unchanged

  # current_length: [3, 0, 3]  <- batch 1 has 0 valid frames

  buffer.append(torch.tensor([[4.0], [99.0], [400.0]]))  # t3

  # buffer.buffer after append:
  # Batch 0: [2.0, 3.0, 4.0]        <- oldest overwritten (normal)
  # Batch 1: [99.0, 99.0, 99.0]     <- BACKFILLED with 99.0
  # Batch 2: [200.0, 300.0, 400.0]  <- oldest overwritten (normal)

Key insight:
  Reset only affects specific batch rows. The pointer keeps advancing for
  everyone, but reset rows get backfilled on their next append.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


class CircularBuffer:
  """Fixed-length circular buffer for batched tensor history.

  Stores history with shape (max_len, batch_size, ...) internally.
  The `buffer` property returns chronologically ordered data (oldest to newest)
  with shape (batch_size, max_len, ...).

  Storage and Retrieval
  ---------------------
  Internal storage (circular):
    ┌─────┬─────┬─────┐
    │  2  │  3  │  1  │  <- pointer at index 1 (newest = 3)
    └─────┴─────┴─────┘
     idx:0  idx:1  idx:2

  buffer property returns (chronological, oldest→newest):
    ┌─────┬─────┬─────┐
    │  1  │  2  │  3  │
    └─────┴─────┴─────┘

  LIFO Retrieval via __getitem__
  -------------------------------
  Given buffer with [1, 2, 3] (oldest to newest):
    buffer[lag=0] -> 3  (most recent)
    buffer[lag=1] -> 2  (one step back)
    buffer[lag=2] -> 1  (oldest)

  Per-Batch Reset
  ---------------
  After reset(batch_ids=[1]):
    Batch 0: [1, 2, 3]  (unchanged)
    Batch 1: [0, 0, 0]  (zeroed, current_length=0)
    Batch 2: [1, 2, 3]  (unchanged)

  Next append backfills reset rows with first value.

  Args:
    max_len: Maximum number of historical frames to retain.
    batch_size: Size of the batch dimension.
    device: Torch device for storage.
  """

  def __init__(self, max_len: int, batch_size: int, device: str) -> None:
    if max_len < 1:
      raise ValueError(f"Buffer size must be >= 1, got {max_len}")

    self._max_len = max_len
    self._batch_size = batch_size
    self._device = device
    self._pointer: int = -1
    self._buffer: torch.Tensor | None = None
    self._all_indices = torch.arange(batch_size, device=device)
    self._num_pushes = torch.zeros(batch_size, dtype=torch.long, device=device)
    self._max_len_tensor = torch.full(
      (batch_size,), max_len, dtype=torch.long, device=device
    )

  @property
  def batch_size(self) -> int:
    return self._batch_size

  @property
  def device(self) -> str:
    return self._device

  @property
  def max_length(self) -> int:
    return self._max_len

  @property
  def current_length(self) -> torch.Tensor:
    """Per-batch count of valid frames. Shape: (batch_size,)."""
    return torch.minimum(self._num_pushes, self._max_len_tensor)

  @property
  def is_initialized(self) -> bool:
    """Check if the buffer has been initialized with at least one append."""
    return self._buffer is not None

  @property
  def buffer(self) -> torch.Tensor:
    """History in chronological order (oldest to newest).

    Returns:
      Tensor of shape (batch_size, max_len, ...) where index 0 is oldest
      and index -1 is newest.
    """
    if self._buffer is None:
      raise RuntimeError("Buffer not initialized. Call append() first.")

    start = (self._pointer + 1) % self._max_len
    idx = (torch.arange(self._max_len, device=self._device) + start) % self._max_len
    buf = self._buffer.index_select(0, idx)  # (max_len, batch, ...)
    return buf.transpose(0, 1)  # (batch, max_len, ...)

  def reset(self, batch_ids: Sequence[int] | torch.Tensor | None = None) -> None:
    """Zero out values and counters for specified batch rows.

    Args:
      batch_ids: Batch indices to reset, or None to reset all.
    """
    ids: Sequence[int] | torch.Tensor | slice = (
      slice(None) if batch_ids is None else batch_ids
    )
    self._num_pushes[ids] = 0
    if self._buffer is not None:
      self._buffer[:, ids] = 0.0

  def backfill(self, data: torch.Tensor, batch_ids: torch.Tensor) -> None:
    """Fill the given rows' entire history with one frame, without advancing time.

    Unlike append, the global pointer does not move and other rows are
    untouched. Used after a partial reset: the reset rows get their first
    post-reset frame in every slot (the same backfill their next append would
    apply) while the remaining rows keep their history intact.

    Args:
      data: Tensor of shape (batch_size, ...); only rows at batch_ids are read.
      batch_ids: Batch indices to backfill.
    """
    if data.shape[0] != self._batch_size:
      raise ValueError(f"Expected batch size {self._batch_size}, got {data.shape[0]}")
    if self._buffer is None:
      raise RuntimeError("Buffer not initialized. Call append() first.")

    data = data.to(self._device)
    self._buffer[:, batch_ids] = data[batch_ids].unsqueeze(0)
    self._num_pushes[batch_ids] = 1

  def capture_state(
    self, batch_ids: torch.Tensor | Sequence[int] | None = None
  ) -> dict[str, object]:
    """Capture selected rows in chronological order, never the whole batch."""
    if batch_ids is None:
      ids = torch.arange(self._batch_size, device=self._device)
    else:
      ids = torch.as_tensor(batch_ids, dtype=torch.long, device=self._device)
    if (
      ids.ndim != 1
      or ids.numel() == 0
      or bool((ids < 0).any())
      or bool((ids >= self._batch_size).any())
      or ids.unique().numel() != ids.numel()
    ):
      raise ValueError("capture batch_ids must be unique in-range row IDs")
    chronological = (
      None
      if self._buffer is None
      else self.buffer.index_select(0, ids).detach().clone()
    )
    if chronological is not None:
      if chronological.device != torch.device(self._device):
        raise ValueError("captured history device mismatch")
    return {
      "max_len": self._max_len,
      "source_batch_size": self._batch_size,
      "source_row_ids": ids.detach().clone(),
      "dtype": None if chronological is None else str(chronological.dtype),
      "device": self._device,
      "pointer": self._pointer,
      "current_length": self._num_pushes.index_select(0, ids).detach().clone(),
      "chronological": chronological,
    }

  def restore_state(
    self, state: dict[str, object], batch_ids: torch.Tensor | None = None
  ) -> None:
    """Restore selected rows at the destination pointer after full validation."""
    if state.get("max_len") != self._max_len:
      raise ValueError("circular-buffer max_len mismatch")
    source_size = state.get("source_batch_size", state.get("batch_size"))
    if not isinstance(source_size, int):
      raise ValueError("circular-buffer source batch size is missing")
    rows = state.get("source_row_ids")
    if not isinstance(rows, torch.Tensor) or rows.ndim != 1 or rows.numel() == 0:
      raise ValueError("circular-buffer source row IDs are malformed")
    if (
      rows.dtype != torch.long
      or bool((rows < 0).any())
      or bool((rows >= source_size).any())
    ):
      raise ValueError("circular-buffer source row IDs are invalid")
    if rows.unique().numel() != rows.numel():
      raise ValueError("circular-buffer source row IDs are duplicated")
    if batch_ids is None:
      if rows.numel() != self._batch_size:
        raise ValueError(
          "destination row mapping is required for different batch sizes"
        )
      ids = torch.arange(self._batch_size, device=self._device)
    else:
      ids = torch.as_tensor(batch_ids, dtype=torch.long, device=self._device)
    if (
      ids.ndim != 1
      or ids.numel() != rows.numel()
      or bool((ids < 0).any())
      or bool((ids >= self._batch_size).any())
      or ids.unique().numel() != ids.numel()
    ):
      raise ValueError("destination row mapping is malformed")
    chronological = state.get("chronological")
    lengths = state.get("current_length")
    if chronological is None:
      if not isinstance(lengths, torch.Tensor) or lengths.shape != (rows.numel(),):
        raise ValueError("circular-buffer current lengths are malformed")
      if bool((lengths < 0).any()) or bool((lengths > self._max_len).any()):
        raise ValueError("circular-buffer current lengths are out of range")
      self.reset(ids)
      return
    if (
      not isinstance(chronological, torch.Tensor)
      or chronological.ndim < 2
      or chronological.shape[:2] != (rows.numel(), self._max_len)
    ):
      raise ValueError("circular-buffer chronological history has wrong shape")
    if (
      str(chronological.dtype) != state.get("dtype")
      or state.get("device") != self._device
    ):
      raise ValueError("circular-buffer dtype/device mismatch")
    if (
      not isinstance(lengths, torch.Tensor)
      or lengths.shape != (rows.numel(),)
      or lengths.dtype != torch.long
    ):
      raise ValueError("circular-buffer current lengths are malformed")
    if bool((lengths < 0).any()):
      raise ValueError("circular-buffer current lengths are out of range")
    if self._buffer is not None and self._buffer.shape[2:] != chronological.shape[2:]:
      raise ValueError("circular-buffer trailing shape mismatch")
    destination = self._pointer
    if self._buffer is None:
      self._buffer = torch.zeros(
        (self._max_len, self._batch_size, *chronological.shape[2:]),
        dtype=chronological.dtype,
        device=self._device,
      )
      destination = self._max_len - 1
      self._pointer = destination
    order = (
      torch.arange(self._max_len, device=self._device) + destination + 1
    ) % self._max_len
    self._buffer[order[:, None], ids] = chronological.to(self._device).transpose(0, 1)
    self._num_pushes[ids] = lengths

  def append(self, data: torch.Tensor) -> None:
    """Append a new frame for all batch elements.

    Args:
      data: Tensor of shape (batch_size, ...).
    """
    if data.shape[0] != self._batch_size:
      raise ValueError(f"Expected batch size {self._batch_size}, got {data.shape[0]}")

    data = data.to(self._device)

    if self._buffer is None:
      self._pointer = -1
      self._buffer = torch.empty(
        (self._max_len, *data.shape), dtype=data.dtype, device=self._device
      )

    self._pointer = (self._pointer + 1) % self._max_len
    self._buffer[self._pointer] = data

    # Backfill entire history with first frame for newly initialized batches.
    # The condition and data are reshaped to broadcast against the buffer's
    # (max_len, batch_size, ...) shape for data of any rank.
    is_first_push = self._num_pushes == 0
    condition = is_first_push.view(1, self._batch_size, *([1] * (data.ndim - 1)))
    torch.where(condition, data.unsqueeze(0), self._buffer, out=self._buffer)

    self._num_pushes += 1

  def __getitem__(self, key: torch.Tensor | int) -> torch.Tensor:
    """Retrieve lagged frames per batch (LIFO).

    Args:
      key: Per-batch lags (Tensor) or shared lag (int). Shape (batch_size,) or scalar.
    """
    if self._buffer is None:
      raise RuntimeError("Buffer not initialized. Call append() first.")

    if isinstance(key, int):
      key = torch.full((self._batch_size,), key, dtype=torch.long, device=self._device)
    else:
      if key.ndim == 0:
        key = key.expand(self._batch_size)
      key = key.to(device=self._device, dtype=torch.long)

    if key.numel() != self._batch_size:
      raise ValueError(f"Expected {self._batch_size} lags, got {key.numel()}")

    # Clamp to the oldest retained frame: without the max_len bound, a lag
    # beyond the buffer length would wrap around to a newer frame once
    # num_pushes exceeds max_len.
    pushes = self._num_pushes.clamp_min(1)
    max_lag = torch.minimum(pushes, self._max_len_tensor) - 1
    valid = torch.minimum(key, max_lag).clamp_min(0)

    idx = torch.remainder(self._pointer - valid, self._max_len)
    return self._buffer[idx, self._all_indices]
