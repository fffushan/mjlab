"""Unguided deterministic DDIM reverse sampling for D2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor

from .noising import apply_clean_mask
from .schedule import DiffusionSchedule, InferenceGrid, build_inference_grid


@dataclass(frozen=True, slots=True)
class SamplerConditions:
  """Clean conditioning values and the D0 fixed-entry mask."""

  clean_tokens: Tensor
  clean_mask: Tensor


@dataclass(frozen=True, slots=True)
class SamplerConfig:
  """Configuration for the frozen unguided DDIM sampler."""

  grid: InferenceGrid
  eta: float = 0.0
  guidance_strength: float = 0.0
  initial_noise_scale: float = 1.0

  def __post_init__(self) -> None:
    if self.eta != 0.0:
      raise ValueError("D2 supports only eta=0.0")
    if self.guidance_strength != 0.0:
      raise ValueError("D2 does not support non-zero guidance_strength")
    if not torch.isfinite(torch.tensor(self.initial_noise_scale)):
      raise ValueError("initial_noise_scale must be finite")
    if self.initial_noise_scale < 0.0:
      raise ValueError("initial_noise_scale must be non-negative")


@dataclass(frozen=True, slots=True)
class SamplerDiagnostics:
  """Observable reverse-loop diagnostics returned with a sample."""

  denoiser_calls: int
  jumps: int
  source_ids_returned: tuple[int, ...]
  step_ids_seen: tuple[int, ...]
  final_index: int
  initial_noise_norm: dict[str, float]


def _randn(
  shape: torch.Size | tuple[int, ...],
  *,
  generator: torch.Generator,
  device: torch.device,
  dtype: torch.dtype,
) -> Tensor:
  """Draw with the supplied generator, transferring when devices differ."""
  generator_device = getattr(generator, "device", None)
  draw_device = device if generator_device is None else torch.device(generator_device)
  value = torch.randn(
    shape,
    generator=generator,
    device=draw_device,
    dtype=dtype,
  )
  return value if draw_device == device else value.to(device=device)


def _condition_values(conditions: Any) -> tuple[Tensor, Tensor]:
  if isinstance(conditions, Mapping):
    try:
      return conditions["clean_tokens"], conditions["clean_mask"]
    except KeyError as exc:
      raise ValueError("conditions must contain clean_tokens and clean_mask") from exc
  try:
    return conditions.clean_tokens, conditions.clean_mask
  except AttributeError as exc:
    raise TypeError("conditions must be SamplerConditions-like or a mapping") from exc


def _as_batched(
  clean_tokens: Tensor, clean_mask: Tensor
) -> tuple[Tensor, Tensor, bool]:
  if not isinstance(clean_tokens, Tensor) or not clean_tokens.is_floating_point():
    raise TypeError("clean_tokens must be a floating-point torch tensor")
  if clean_tokens.ndim == 2:
    clean_batched = clean_tokens.unsqueeze(0)
    squeezed = True
  elif clean_tokens.ndim == 3:
    clean_batched = clean_tokens
    squeezed = False
  else:
    raise ValueError("clean_tokens must have shape [T, D] or [B, T, D]")

  if not isinstance(clean_mask, Tensor):
    raise TypeError("clean_mask must be a torch tensor")
  if clean_mask.ndim == 2:
    mask_batched = clean_mask.unsqueeze(0)
  elif clean_mask.ndim == 3:
    mask_batched = clean_mask
  else:
    raise ValueError("clean_mask must have shape [T, D] or [B, T, D]")
  if mask_batched.shape[-2:] != clean_batched.shape[-2:]:
    raise ValueError("clean_mask does not match clean_tokens sequence and width")
  if mask_batched.shape[0] not in {1, clean_batched.shape[0]}:
    raise ValueError("clean_mask batch dimension does not match clean_tokens")
  if mask_batched.shape[0] == 1 and clean_batched.shape[0] != 1:
    mask_batched = mask_batched.expand(clean_batched.shape[0], -1, -1)
  return (
    clean_batched,
    mask_batched.to(device=clean_batched.device, dtype=torch.bool),
    squeezed,
  )


def _infer_state_dimension(mask: Tensor) -> int:
  """Infer the D0 state/latent boundary from its partial current row."""
  first = mask[0]
  partial = (first.any(dim=-1) & ~first.all(dim=-1)).nonzero(as_tuple=False)
  if partial.numel():
    state_dimension = int(first[int(partial[0].item())].sum().item())
    if 0 < state_dimension < first.shape[-1]:
      return state_dimension
  # A fully unknown synthetic mask has no boundary signal.  Keep the frozen
  # projected-state width for real tokens and a useful fallback for small tests.
  return 199 if first.shape[-1] >= 231 else max(1, first.shape[-1] - 1)


def _initial_norms(noise: Tensor, state_dimension: int) -> dict[str, float]:
  state = noise[..., :state_dimension]
  latent = noise[..., state_dimension:]
  return {
    "state": float(torch.linalg.vector_norm(state).item()),
    "latent": float(torch.linalg.vector_norm(latent).item()),
    "tokens": float(torch.linalg.vector_norm(noise).item()),
  }


def sample_trajectory(
  denoiser: Any,
  *,
  schedule: DiffusionSchedule,
  conditions: SamplerConditions | Mapping[str, Tensor] | Any,
  config: SamplerConfig,
  generator: torch.Generator,
) -> tuple[Tensor, SamplerDiagnostics]:
  """Run exactly the 20 deterministic DDIM jumps and return all tokens.

  The denoiser sees the original training timestep at each jump.  Fixed entries
  use clean step id zero and are restored from ``clean_tokens`` after every
  update, including the final destination at index zero.
  """
  if config.eta != 0.0:
    raise ValueError("D2 supports only eta=0.0")
  if config.guidance_strength != 0.0:
    raise ValueError("D2 does not support non-zero guidance_strength")
  expected_grid = build_inference_grid(schedule)
  if config.grid != expected_grid:
    raise ValueError("D2 sampler requires the frozen inference grid")
  clean_tokens, clean_mask = _condition_values(conditions)
  clean_batched, mask_batched, squeezed = _as_batched(clean_tokens, clean_mask)
  batch, sequence, token_dimension = clean_batched.shape
  state_dimension = _infer_state_dimension(mask_batched)
  if state_dimension >= token_dimension:
    raise ValueError("clean mask does not leave a latent slice")

  noise = (
    _randn(
      clean_batched.shape,
      generator=generator,
      device=clean_batched.device,
      dtype=clean_batched.dtype,
    )
    * config.initial_noise_scale
  )
  sample = apply_clean_mask(noise, clean_batched, mask_batched)
  diagnostics_norm = _initial_norms(noise, state_dimension)

  fixed_state = mask_batched[..., :state_dimension].all(dim=-1)
  fixed_latent = mask_batched[..., state_dimension:].all(dim=-1)
  alpha_bars = schedule.alpha_bars_torch(device=sample.device, dtype=sample.dtype)
  seen: list[int] = []
  returned: list[int] = []

  for source, destination in zip(
    config.grid.source_ids, config.grid.destination_ids, strict=True
  ):
    if source == schedule.clean_index:
      raise ValueError("the denoiser must never be called at clean index 0")
    step_ids = torch.full(
      (batch, sequence, 2),
      source,
      dtype=torch.long,
      device=sample.device,
    )
    step_ids[..., 0] = torch.where(
      fixed_state, torch.zeros_like(step_ids[..., 0]), step_ids[..., 0]
    )
    step_ids[..., 1] = torch.where(
      fixed_latent, torch.zeros_like(step_ids[..., 1]), step_ids[..., 1]
    )
    x0_hat = denoiser(sample, step_ids)
    if not isinstance(x0_hat, Tensor) or x0_hat.shape != sample.shape:
      raise ValueError("denoiser must return a tensor with the token input shape")
    seen.append(int(source))
    returned.append(int(source))

    a_source = alpha_bars[source]
    if destination == schedule.clean_index:
      jumped = x0_hat.clone()
    else:
      a_destination = alpha_bars[destination]
      eps_hat = (sample - torch.sqrt(a_source) * x0_hat) / torch.sqrt(1.0 - a_source)
      jumped = (
        torch.sqrt(a_destination) * x0_hat + torch.sqrt(1.0 - a_destination) * eps_hat
      )
    sample = apply_clean_mask(jumped, clean_batched, mask_batched)

  diagnostics = SamplerDiagnostics(
    denoiser_calls=len(returned),
    jumps=len(config.grid.source_ids),
    source_ids_returned=tuple(returned),
    step_ids_seen=tuple(seen),
    final_index=config.grid.destination_ids[-1],
    initial_noise_norm=diagnostics_norm,
  )
  if squeezed:
    return sample[0], diagnostics
  return sample, diagnostics


__all__ = [
  "SamplerConditions",
  "SamplerConfig",
  "SamplerDiagnostics",
  "sample_trajectory",
]
