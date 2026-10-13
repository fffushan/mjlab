"""X0 Euler sampling primitives for the X2 flow-matching family.

Sampling descends a configurable, strictly decreasing integer level grid from
``process.max_level`` to 0.  Every step evaluates the network at the current
source level and applies one Euler step of the x0-parameterized flow ODE::

  x_r = (r / s) * x_s + (1 - r / s) * x0_hat

Known slices keep level 0 and are restored from the clean tokens after
initialization and after every update, using the existing token-mask semantics.
The network is never evaluated at a source level of 0.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..diffusion.noising import apply_clean_mask
from .process import (
  DEFAULT_MAX_LEVEL,
  DEFAULT_PROCESS,
  FlowMatchingError,
  LinearNoiseProcess,
  _integer_levels,
  draw_gaussian,
)

DEFAULT_CALLS = 20
_LATENT_DIMENSION = 32
_Denoiser = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True, slots=True)
class FlowGrid:
  """Strictly descending integer source levels that end at clean level 0."""

  levels: tuple[int, ...]

  def __post_init__(self) -> None:
    levels = tuple(self.levels)
    if len(levels) < 2:
      raise FlowMatchingError("a flow grid needs at least one update")
    if any(isinstance(level, bool) or not isinstance(level, int) for level in levels):
      raise FlowMatchingError("flow grid levels must be integers")
    if levels[-1] != 0:
      raise FlowMatchingError("a flow grid must end at clean level 0")
    if any(level < 0 for level in levels):
      raise FlowMatchingError("flow grid levels must be non-negative")
    if len(set(levels)) != len(levels):
      raise FlowMatchingError("flow grid levels must not repeat")
    if any(levels[index] <= levels[index + 1] for index in range(len(levels) - 1)):
      raise FlowMatchingError("flow grid levels must be strictly descending")
    object.__setattr__(self, "levels", levels)

  @property
  def source_levels(self) -> tuple[int, ...]:
    """Levels used as network inputs, highest first."""
    return self.levels[:-1]

  @property
  def destination_levels(self) -> tuple[int, ...]:
    """Levels reached after each update, lowest level last."""
    return self.levels[1:]

  @property
  def calls(self) -> int:
    """Number of network calls and Euler updates."""
    return len(self.levels) - 1

  def as_dict(self) -> dict[str, object]:
    """Return the grid in a JSON-compatible representation."""
    return {"levels": list(self.levels), "calls": self.calls}

  @classmethod
  def uniform(
    cls, *, calls: int = DEFAULT_CALLS, max_level: int = DEFAULT_MAX_LEVEL
  ) -> "FlowGrid":
    """Build a grid with evenly spaced levels from ``max_level`` down to 0."""
    if isinstance(calls, bool) or not isinstance(calls, int) or calls < 1:
      raise FlowMatchingError("calls must be a positive integer")
    if isinstance(max_level, bool) or not isinstance(max_level, int):
      raise FlowMatchingError("max_level must be an integer")
    if max_level < calls:
      raise FlowMatchingError("max_level must be at least as large as calls")
    levels = tuple(
      round(max_level * (calls - index) / calls) for index in range(calls + 1)
    )
    return cls(levels)


@dataclass(frozen=True, slots=True)
class FlowConditions:
  """Clean conditioning tokens and the fixed-entry token mask."""

  clean_tokens: Tensor
  clean_mask: Tensor
  state_dimension: int | None = None


@dataclass(frozen=True, slots=True)
class FlowSamplerConfig:
  """The level grid and the linear process used by one sampling run."""

  grid: FlowGrid = FlowGrid.uniform()
  process: LinearNoiseProcess = DEFAULT_PROCESS

  def __post_init__(self) -> None:
    if self.grid.levels[0] != self.process.max_level:
      raise FlowMatchingError(
        "the sampling grid must start at the process max_level "
        f"{self.process.max_level}, got {self.grid.levels[0]}"
      )


@dataclass(frozen=True, slots=True)
class FlowSamplerDiagnostics:
  """Observable sampling diagnostics returned together with the sample."""

  denoiser_calls: int
  source_levels: tuple[int, ...]
  destination_levels: tuple[int, ...]
  final_level: int


def _validate_float_tensor(value: Any, name: str) -> Tensor:
  if not isinstance(value, Tensor):
    raise FlowMatchingError(f"{name} must be a torch tensor")
  if not value.is_floating_point():
    raise FlowMatchingError(f"{name} must be floating point")
  if not bool(torch.isfinite(value).all()):
    raise FlowMatchingError(f"{name} contains non-finite values")
  return value


def _validate_prediction(sample: Tensor, x0_hat: Any) -> Tensor:
  prediction = _validate_float_tensor(x0_hat, "x0_hat")
  if prediction.dtype != sample.dtype:
    raise FlowMatchingError(
      f"x0_hat must use the sample dtype {sample.dtype}, got {prediction.dtype}"
    )
  if prediction.shape != sample.shape:
    raise FlowMatchingError(
      "x0_hat must match the token input shape "
      f"{tuple(sample.shape)}, got {tuple(prediction.shape)}"
    )
  return prediction


def _broadcast_levels(
  value: Any, *, name: str, process: LinearNoiseProcess, expected: tuple[int, ...]
) -> Tensor:
  levels = _integer_levels(value, name=name, max_level=process.max_level)
  try:
    broadcast = tuple(torch.broadcast_shapes(tuple(levels.shape), tuple(expected)))
  except RuntimeError as exc:
    raise FlowMatchingError(
      f"{name} with shape {tuple(levels.shape)} does not broadcast to {expected}"
    ) from exc
  if broadcast != tuple(expected):
    raise FlowMatchingError(
      f"{name} with shape {tuple(levels.shape)} does not broadcast to {expected}"
    )
  return levels


def euler_x0_update(
  sample: Tensor,
  x0_hat: Tensor,
  *,
  source_levels: Any,
  destination_levels: Any,
  process: LinearNoiseProcess = DEFAULT_PROCESS,
) -> Tensor:
  """Apply one Euler step of the x0-parameterized linear flow ODE.

  ``source_levels`` and ``destination_levels`` may be integers or integer
  tensors broadcastable to ``sample.shape[:-1]``, which gives one source and
  destination level per row and state/latent slice.  ``destination_levels`` of
  zero returns ``x0_hat`` on those slices.  Ascending or zero-source updates are
  rejected because the network must never be evaluated at clean level 0.
  """
  sample = _validate_float_tensor(sample, "sample")
  x0_hat = _validate_prediction(sample, x0_hat)
  expected = tuple(sample.shape[:-1])
  source = _broadcast_levels(
    source_levels, name="source_levels", process=process, expected=expected
  )
  destination = _broadcast_levels(
    destination_levels, name="destination_levels", process=process, expected=expected
  )
  if bool((source <= 0).any()):
    raise FlowMatchingError("the network must never be called at source level 0")
  if bool((destination >= source).any()):
    raise FlowMatchingError(
      "flow updates must descend: destination level must be below source level"
    )
  scale = float(process.max_level)
  # Integer levels are typically CPU literals or CPU tensors, so the ratio must
  # follow the sample device and dtype before it multiplies the sample.
  ratio = (
    (destination.to(dtype=sample.dtype) / scale)
    / (source.to(dtype=sample.dtype) / scale)
  ).to(device=sample.device, dtype=sample.dtype)
  return ratio.unsqueeze(-1) * sample + (1.0 - ratio).unsqueeze(-1) * x0_hat


def _condition_values(conditions: Any) -> tuple[Any, Any, int | None]:
  if isinstance(conditions, Mapping):
    try:
      return (
        conditions["clean_tokens"],
        conditions["clean_mask"],
        conditions.get("state_dimension"),
      )
    except KeyError as exc:
      raise FlowMatchingError(
        "conditions must contain clean_tokens and clean_mask"
      ) from exc
  try:
    return (
      conditions.clean_tokens,
      conditions.clean_mask,
      conditions.state_dimension,
    )
  except AttributeError as exc:
    raise FlowMatchingError(
      "conditions must be FlowConditions-like or a mapping"
    ) from exc


def _as_batched(clean_tokens: Any, clean_mask: Any) -> tuple[Tensor, Tensor, bool]:
  clean = _validate_float_tensor(clean_tokens, "clean_tokens")
  if clean.ndim == 2:
    clean_batched = clean.unsqueeze(0)
    squeezed = True
  elif clean.ndim == 3:
    clean_batched = clean
    squeezed = False
  else:
    raise FlowMatchingError("clean_tokens must have shape [T, D] or [B, T, D]")
  if clean_batched.shape[0] < 1:
    raise FlowMatchingError("clean_tokens must contain at least one row")

  if not isinstance(clean_mask, Tensor):
    raise FlowMatchingError("clean_mask must be a torch tensor")
  if clean_mask.dtype != torch.bool:
    raise FlowMatchingError("clean_mask must be a bool tensor")
  if clean_mask.ndim == 2:
    mask_batched = clean_mask.unsqueeze(0)
  elif clean_mask.ndim == 3:
    mask_batched = clean_mask
  else:
    raise FlowMatchingError("clean_mask must have shape [T, D] or [B, T, D]")
  if mask_batched.shape[-2:] != clean_batched.shape[-2:]:
    raise FlowMatchingError("clean_mask does not match the clean token shape")
  if mask_batched.shape[0] not in {1, clean_batched.shape[0]}:
    raise FlowMatchingError("clean_mask batch dimension does not match clean_tokens")
  if mask_batched.shape[0] == 1 and clean_batched.shape[0] != 1:
    mask_batched = mask_batched.expand(clean_batched.shape[0], -1, -1)
  return (
    clean_batched,
    mask_batched.to(device=clean_batched.device),
    squeezed,
  )


def _state_dimension(explicit: int | None, mask: Tensor, token_dimension: int) -> int:
  if explicit is not None:
    if isinstance(explicit, bool) or not isinstance(explicit, int):
      raise FlowMatchingError("state_dimension must be an integer")
    if not 0 < explicit < token_dimension:
      raise FlowMatchingError("state_dimension must lie inside the token width")
    return explicit
  first = mask[0]
  partial = (first.any(dim=-1) & ~first.all(dim=-1)).nonzero(as_tuple=False)
  if partial.numel():
    inferred = int(first[int(partial[0].item())].sum().item())
    if 0 < inferred < token_dimension:
      return inferred
  # A mask that conditions only the latent slice, or a synthetic mask with no
  # partial row, carries no state/latent boundary.  The frozen token width minus
  # the latent width is the boundary used by the diffusion sampler, and callers
  # that condition other slices must pass state_dimension explicitly.
  fallback = token_dimension - _LATENT_DIMENSION
  if fallback <= 0:
    raise FlowMatchingError("state_dimension is required for this clean mask")
  return fallback


def _step_ids(*, source: int, fixed_state: Tensor, fixed_latent: Tensor) -> Tensor:
  """Build ``[B, T, 2]`` state/latent level ids for one network call."""
  source_ids = torch.full(
    fixed_state.shape, int(source), dtype=torch.long, device=fixed_state.device
  )
  zeros = torch.zeros_like(source_ids)
  state_ids = torch.where(fixed_state, zeros, source_ids)
  latent_ids = torch.where(fixed_latent, zeros, source_ids)
  return torch.stack((state_ids, latent_ids), dim=-1)


def _diagnostics(
  seen: tuple[int, ...], config: FlowSamplerConfig
) -> FlowSamplerDiagnostics:
  return FlowSamplerDiagnostics(
    denoiser_calls=len(seen),
    source_levels=seen,
    destination_levels=config.grid.destination_levels,
    final_level=config.grid.levels[-1],
  )


def sample_trajectory(
  denoiser: _Denoiser,
  *,
  conditions: FlowConditions | Mapping[str, Tensor],
  config: FlowSamplerConfig,
  generator: torch.Generator,
) -> tuple[Tensor, FlowSamplerDiagnostics]:
  """Descend ``config.grid`` and return the sampled tokens.

  The initial sample is Gaussian at level ``process.max_level`` with the known
  slices restored.  ``denoiser`` receives that sample and the ``[B, T, 2]``
  integer step ids: level 0 on known state/latent slices and the current source
  level everywhere else.  Known slices are restored bit-identically after
  initialization and after every update.  If the mask marks every slice known,
  the denoiser is not called at all, because every slice already has source
  level 0.
  """
  clean_tokens, clean_mask, explicit_dimension = _condition_values(conditions)
  clean_batched, mask_batched, squeezed = _as_batched(clean_tokens, clean_mask)
  state_dimension = _state_dimension(
    explicit_dimension, mask_batched, clean_batched.shape[-1]
  )
  if not bool((~mask_batched).any()):
    sample = clean_batched.clone()
    return (sample[0] if squeezed else sample), _diagnostics((), config)

  noise = draw_gaussian(
    clean_batched.shape,
    generator=generator,
    device=clean_batched.device,
    dtype=clean_batched.dtype,
  )
  sample = apply_clean_mask(noise, clean_batched, mask_batched)
  fixed_state = mask_batched[..., :state_dimension].all(dim=-1)
  fixed_latent = mask_batched[..., state_dimension:].all(dim=-1)

  seen: list[int] = []
  for source, destination in zip(
    config.grid.source_levels, config.grid.destination_levels, strict=True
  ):
    step_ids = _step_ids(
      source=source, fixed_state=fixed_state, fixed_latent=fixed_latent
    )
    x0_hat = _validate_prediction(sample, denoiser(sample, step_ids))
    seen.append(int(source))
    stepped = euler_x0_update(
      sample,
      x0_hat,
      source_levels=source,
      destination_levels=destination,
      process=config.process,
    )
    sample = apply_clean_mask(stepped, clean_batched, mask_batched)

  diagnostics = _diagnostics(tuple(seen), config)
  return (sample[0] if squeezed else sample), diagnostics


__all__ = [
  "DEFAULT_CALLS",
  "FlowConditions",
  "FlowGrid",
  "FlowSamplerConfig",
  "FlowSamplerDiagnostics",
  "euler_x0_update",
  "sample_trajectory",
]
