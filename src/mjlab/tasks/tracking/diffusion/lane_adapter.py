"""D5 controller-lane token adapter for a reduced diffusion state layout.

The X2 D5 lane is compiled for one frozen token layout: the 135-channel hybrid
state followed by the 32-channel raw latent, so 167 token channels.  Its loader
validates the declared state/token widths and the denoiser graph input shape, so
a model trained on a *reduced* state representation cannot be loaded by it.

Every reduced representation in this package is a channel selection of
:data:`~.state.HYBRID_STATE`:
:func:`~.state.hybrid_to_local_kinematic` keeps every retained channel unchanged
and removes only the root translation fields.  A lane-token adapter therefore
exists and is exact for every channel the model reads:

1. gather the model's own channels out of the lane token;
2. run the unchanged model;
3. write the model output back into the lane token.

Lane channels the model does not own pass through unchanged, so they stay
finite and the measured prefix stays consistent.  Nothing here changes the
model, its weights, its normalization or the lane: the adapter is a graph
boundary, and the model's own channels are not touched by any arithmetic.

The lane layout is a compile-time constant of the controller
(``diffusion_types.h``: ``kStateDim == 135`` and ``kTokenDim == 167``), not a
mjlab contract.  Reproduce it here from the frozen
:data:`~.state.HYBRID_STATE` representation.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .state import HYBRID_STATE, StateRepresentation

LANE_STATE_REPRESENTATION = HYBRID_STATE
LANE_STATE_DIMENSION = HYBRID_STATE.dimension
"""The controller lane's physical state width (135)."""

LANE_LATENT_DIMENSION = 32
"""The controller lane's raw latent width (32)."""

LANE_TOKEN_DIMENSION = LANE_STATE_DIMENSION + LANE_LATENT_DIMENSION
"""The controller lane's token width (167)."""

LANE_ADAPTER_FILL_MEAN = 0.0
LANE_ADAPTER_FILL_STD = 1.0
"""Normalization filler for lane state channels the model does not own.

The filler is never read by the model.  It only has to be finite, and the
standard deviation must clear the loader's ``1e-6`` floor.
"""


class LaneAdapterError(ValueError):
  """A reduced state layout cannot be served by the frozen controller lane."""


# ``torch.jit.is_tracing`` is the public way to detect an ONNX/JIT trace, but its stub is
# marked private, so strict type checkers reject the direct attribute access.  Resolve it
# once instead, and treat a missing symbol as "never tracing".
_IS_TRACING = getattr(torch.jit, "is_tracing", None)  # noqa: B009 - see the note above


def _tracing() -> bool:
  """Return True while exporting or tracing, so shape guards stay out of the graph."""
  return bool(_IS_TRACING()) if _IS_TRACING is not None else False


def _field_spans(
  state_slices: Mapping[str, tuple[int, int] | slice],
) -> dict[str, tuple[int, int]]:
  spans: dict[str, tuple[int, int]] = {}
  for name, span in state_slices.items():
    if isinstance(span, slice):
      if span.step is not None or span.start is None or span.stop is None:
        raise LaneAdapterError(
          f"state slice {name!r} must be a contiguous [start, stop) range"
        )
      start, stop = int(span.start), int(span.stop)
    elif isinstance(span, (tuple, list)) and len(span) == 2:
      start, stop = int(span[0]), int(span[1])
    else:
      raise LaneAdapterError(f"state slice {name!r} must be a [start, stop] pair")
    if stop <= start:
      raise LaneAdapterError(f"state slice {name!r} is empty")
    spans[str(name)] = (start, stop)
  if not spans:
    raise LaneAdapterError("a state layout must declare at least one field")
  return spans


def _lane_spans(
  state_slices: Mapping[str, tuple[int, int] | slice],
  *,
  lane: StateRepresentation = LANE_STATE_REPRESENTATION,
) -> dict[str, tuple[int, int]]:
  """Bind every model field to a lane field of the same width, or fail closed."""
  spans = _field_spans(state_slices)
  lane_spans: dict[str, tuple[int, int]] = {}
  for name, (start, stop) in spans.items():
    try:
      lane_field = lane.slices[name]
    except KeyError as exc:
      raise LaneAdapterError(
        f"the controller lane has no {name!r} channel group, so the model's "
        f"{name!r} block cannot be served by it"
      ) from exc
    if (lane_field.stop - lane_field.start) != (stop - start):
      raise LaneAdapterError(
        f"state field {name!r} has width {stop - start} but the lane's "
        f"{name!r} width is {lane_field.stop - lane_field.start}"
      )
    lane_spans[name] = (lane_field.start, lane_field.stop)
  return lane_spans


def lane_dropped_fields(
  state_slices: Mapping[str, tuple[int, int] | slice],
  *,
  lane: StateRepresentation = LANE_STATE_REPRESENTATION,
) -> tuple[str, ...]:
  """Return the lane state fields the model does not own, in lane order."""
  _lane_spans(state_slices, lane=lane)
  owned = set(_field_spans(state_slices))
  return tuple(name for name in lane.slices if name not in owned)


def lane_state_dimension(
  state_slices: Mapping[str, tuple[int, int] | slice],
) -> int:
  """Return the model's physical state width implied by its field slices."""
  return sum(stop - start for start, stop in _field_spans(state_slices).values())


def lane_gather_index(
  state_slices: Mapping[str, tuple[int, int] | slice],
  *,
  latent_dimension: int = LANE_LATENT_DIMENSION,
  lane: StateRepresentation = LANE_STATE_REPRESENTATION,
) -> np.ndarray:
  """Return the lane token channel of each model token channel, in model order.

  The model token is its state block followed by its raw latent block.  The
  returned index is a pure channel selection: it never repeats a lane channel
  and it never invents one.
  """
  if latent_dimension <= 0:
    raise LaneAdapterError("latent_dimension must be positive")
  spans = _field_spans(state_slices)
  lane_spans = _lane_spans(spans, lane=lane)
  index: list[int] = []
  for name, (start, stop) in spans.items():
    lane_start, _ = lane_spans[name]
    index.extend(range(lane_start, lane_start + (stop - start)))
  index.extend(range(lane.dimension, lane.dimension + latent_dimension))
  result = np.asarray(index, dtype=np.int64)
  if result.size != lane_state_dimension(spans) + latent_dimension:
    raise LaneAdapterError("lane gather index does not cover the model token")
  if len(set(int(value) for value in result)) != int(result.size):
    raise LaneAdapterError("lane gather index repeats a channel")
  return result


def lane_model_token_dimension(
  state_slices: Mapping[str, tuple[int, int] | slice],
  *,
  latent_dimension: int = LANE_LATENT_DIMENSION,
) -> int:
  """Return the model token width for a reduced state layout."""
  return lane_state_dimension(state_slices) + latent_dimension


def lane_state_statistics(
  state_mean: np.ndarray,
  state_std: np.ndarray,
  state_slices: Mapping[str, tuple[int, int] | slice],
  *,
  lane: StateRepresentation = LANE_STATE_REPRESENTATION,
  fill_mean: float = LANE_ADAPTER_FILL_MEAN,
  fill_std: float = LANE_ADAPTER_FILL_STD,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
  """Place reduced-layout state statistics into the lane state layout.

  Every channel the model owns keeps its own mean and standard deviation
  exactly.  The lane channels the model does not own receive the declared
  filler, because nothing reads them.
  """
  spans = _field_spans(state_slices)
  lane_spans = _lane_spans(spans, lane=lane)
  dimension = lane_state_dimension(spans)
  mean = np.asarray(state_mean, dtype=np.float64).reshape(-1)
  std = np.asarray(state_std, dtype=np.float64).reshape(-1)
  if mean.shape != (dimension,) or std.shape != (dimension,):
    raise LaneAdapterError(
      f"state statistics must have shape ({dimension},), got {mean.shape} and {std.shape}"
    )
  if not np.isfinite(mean).all() or not np.isfinite(std).all():
    raise LaneAdapterError("state statistics contain non-finite values")
  lane_mean = np.full(lane.dimension, float(fill_mean), dtype=np.float64)
  lane_std = np.full(lane.dimension, float(fill_std), dtype=np.float64)
  for name, (start, stop) in spans.items():
    lane_start, lane_stop = lane_spans[name]
    lane_mean[lane_start:lane_stop] = mean[start:stop]
    lane_std[lane_start:lane_stop] = std[start:stop]
  return lane_mean, lane_std, lane_dropped_fields(spans, lane=lane)


def describe_lane_adapter(
  state_slices: Mapping[str, tuple[int, int] | slice],
  *,
  latent_dimension: int = LANE_LATENT_DIMENSION,
  lane: StateRepresentation = LANE_STATE_REPRESENTATION,
) -> dict[str, Any]:
  """Return a bounded, JSON-ready description of the lane boundary."""
  gather = lane_gather_index(state_slices, latent_dimension=latent_dimension, lane=lane)
  dropped = lane_dropped_fields(state_slices, lane=lane)
  model_state = lane_state_dimension(state_slices)
  return {
    "lane_state_dimension": int(lane.dimension),
    "lane_latent_dimension": int(latent_dimension),
    "lane_token_dimension": int(lane.dimension + latent_dimension),
    "model_state_dimension": int(model_state),
    "model_token_dimension": int(model_state + latent_dimension),
    "required": bool(model_state != lane.dimension),
    "dropped_lane_state_fields": list(dropped),
    "dropped_lane_state_channels": [
      int(channel)
      for name in dropped
      for channel in range(lane.slices[name].start, lane.slices[name].stop)
    ],
    "gather_index": [int(value) for value in gather],
    "fill_mean": float(LANE_ADAPTER_FILL_MEAN),
    "fill_std": float(LANE_ADAPTER_FILL_STD),
    "lane_adapter": "gather_scatter_v1",
  }


class LaneTokenAdapter(nn.Module):
  """Serve a reduced-layout denoiser through the frozen lane token width.

  ``core`` is the unchanged model boundary: it takes the model's own token width
  and returns the same width.  This module only moves channels across the lane
  boundary.
  """

  def __init__(
    self,
    core: nn.Module,
    state_slices: Mapping[str, tuple[int, int] | slice],
    *,
    latent_dimension: int = LANE_LATENT_DIMENSION,
    lane: StateRepresentation = LANE_STATE_REPRESENTATION,
  ) -> None:
    super().__init__()
    gather = lane_gather_index(
      state_slices, latent_dimension=latent_dimension, lane=lane
    )
    lane_dimension = lane.dimension + latent_dimension
    pass_through = np.ones(lane_dimension, dtype=np.float32)
    pass_through[gather] = 0.0
    self.core: nn.Module = core
    self.gather_index: torch.Tensor
    self.scatter_index: torch.Tensor
    self.pass_through_mask: torch.Tensor
    self.lane_token_dimension = int(lane_dimension)
    self.model_token_dimension = int(gather.size)
    if self.model_token_dimension == self.lane_token_dimension:
      raise LaneAdapterError(
        "the model token already fills the lane token; no adapter is required"
      )
    self.register_buffer("gather_index", torch.from_numpy(gather))
    self.register_buffer(
      "scatter_index", torch.from_numpy(gather).reshape(1, 1, -1).expand(1, 1, -1)
    )
    self.register_buffer("pass_through_mask", torch.from_numpy(pass_through))

  def forward(self, noisy_tokens: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
    # The channel width is static.  Skip these Python guards while tracing so the export
    # does not constant-fold a shape check into the graph; the export gate separately
    # validates the exported graph's width and proves channel equality on real inputs.
    if not _tracing():
      width = int(noisy_tokens.shape[-1])
      if width != self.lane_token_dimension:
        raise LaneAdapterError(
          f"lane adapter expects {self.lane_token_dimension} token channels, got {width}"
        )
    model_tokens = torch.index_select(noisy_tokens, -1, self.gather_index)
    model_output = self.core(model_tokens, step_ids)
    if not _tracing() and int(model_output.shape[-1]) != self.model_token_dimension:
      raise LaneAdapterError(
        f"lane adapter core returned {int(model_output.shape[-1])} token channels, "
        f"expected {self.model_token_dimension}"
      )
    lane_output = noisy_tokens * self.pass_through_mask
    index = self.scatter_index.expand(noisy_tokens.shape[0], noisy_tokens.shape[-2], -1)
    return lane_output.scatter(-1, index, model_output)


__all__ = [
  "LANE_ADAPTER_FILL_MEAN",
  "LANE_ADAPTER_FILL_STD",
  "LANE_LATENT_DIMENSION",
  "LANE_STATE_DIMENSION",
  "LANE_STATE_REPRESENTATION",
  "LANE_TOKEN_DIMENSION",
  "LaneAdapterError",
  "LaneTokenAdapter",
  "describe_lane_adapter",
  "lane_dropped_fields",
  "lane_gather_index",
  "lane_model_token_dimension",
  "lane_state_dimension",
  "lane_state_statistics",
]
