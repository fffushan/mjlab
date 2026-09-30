"""Independent per-token state/latent forward noising utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

_SEQUENCE_LENGTH = 41


@dataclass(frozen=True, slots=True)
class NoisedPair:
  """Noised state and latent slices plus their independent Gaussian draws."""

  state: Tensor
  latent: Tensor
  noise_state: Tensor
  noise_latent: Tensor
  tokens: Tensor


def _generator_device(generator: torch.Generator) -> torch.device | None:
  """Return a generator's device when available."""
  return getattr(generator, "device", None)


def sample_levels(
  shape: tuple[int, ...] | torch.Size,
  *,
  training_k: int,
  generator: torch.Generator,
) -> Tensor:
  """Sample inclusive integer diffusion levels independently per token."""
  if training_k < 0:
    raise ValueError("training_k must be non-negative")
  normalized_shape = tuple(int(value) for value in shape)
  if any(value < 0 for value in normalized_shape):
    raise ValueError("level shape dimensions must be non-negative")
  return torch.randint(
    0,
    training_k + 1,
    normalized_shape,
    generator=generator,
    device=_generator_device(generator),
    dtype=torch.long,
  )


def _alpha_coefficients(
  levels: Tensor,
  alpha_bars: Any,
  *,
  dtype: torch.dtype,
  device: torch.device,
) -> tuple[Tensor, Tensor]:
  """Gather square-root signal and noise coefficients at ``levels``."""
  alpha = torch.as_tensor(alpha_bars, dtype=dtype, device=device)
  if alpha.ndim != 1 or alpha.numel() == 0:
    raise ValueError("alpha_bars must be a non-empty one-dimensional array")
  levels = levels.to(device=device, dtype=torch.long)
  if torch.any(levels < 0) or torch.any(levels >= alpha.numel()):
    raise ValueError("noise levels are outside alpha_bars")
  values = alpha[levels]
  # Do not clamp 1 - values: the exact zero at clean index 0 is required.
  return torch.sqrt(values), torch.sqrt(1.0 - values)


def _randn(
  shape: torch.Size | tuple[int, ...],
  *,
  generator: torch.Generator,
  device: torch.device,
  dtype: torch.dtype,
) -> Tensor:
  """Draw with the supplied generator, transferring when devices differ."""
  generator_device = _generator_device(generator)
  draw_device = device if generator_device is None else torch.device(generator_device)
  value = torch.randn(
    shape,
    generator=generator,
    device=draw_device,
    dtype=dtype,
  )
  return value if draw_device == device else value.to(device=device)


def _validate_slice(value: Tensor, levels: Tensor, name: str) -> None:
  if value.ndim < 1:
    raise ValueError(f"{name} must have at least one dimension")
  if levels.shape != value.shape[:-1]:
    raise ValueError(
      f"{name} levels must have shape {value.shape[:-1]}, got {levels.shape}"
    )
  if not value.is_floating_point():
    raise TypeError(f"{name} must be a floating-point tensor")


def add_independent_noise(
  state: Tensor,
  latent: Tensor,
  k_state: Tensor,
  k_latent: Tensor,
  alpha_bars: Any,
  *,
  generator: torch.Generator,
) -> NoisedPair:
  """Add independent Gaussian noise to state and latent token slices.

  ``k_state`` and ``k_latent`` contain one inclusive training level for every
  leading token position.  The two calls to :func:`torch.randn` are separate on
  purpose: state and latent noise must not alias or consume one shared draw.
  """
  _validate_slice(state, k_state, "state")
  _validate_slice(latent, k_latent, "latent")
  if state.shape[:-1] != latent.shape[:-1]:
    raise ValueError("state and latent leading dimensions must match")
  if state.device != latent.device:
    raise ValueError("state and latent must be on the same device")
  if state.dtype != latent.dtype:
    raise ValueError("state and latent must have the same dtype")

  signal_state, scale_state = _alpha_coefficients(
    k_state,
    alpha_bars,
    dtype=state.dtype,
    device=state.device,
  )
  signal_latent, scale_latent = _alpha_coefficients(
    k_latent,
    alpha_bars,
    dtype=latent.dtype,
    device=latent.device,
  )
  noise_state = _randn(
    state.shape,
    generator=generator,
    device=state.device,
    dtype=state.dtype,
  )
  noise_latent = _randn(
    latent.shape,
    generator=generator,
    device=latent.device,
    dtype=latent.dtype,
  )
  noised_state = (
    signal_state.unsqueeze(-1) * state + scale_state.unsqueeze(-1) * noise_state
  )
  noised_latent = (
    signal_latent.unsqueeze(-1) * latent + scale_latent.unsqueeze(-1) * noise_latent
  )
  tokens = torch.cat((noised_state, noised_latent), dim=-1)
  return NoisedPair(noised_state, noised_latent, noise_state, noise_latent, tokens)


def x0_target(clean_tokens: Tensor) -> Tensor:
  """Return an independent clean ``x0`` target tensor."""
  if not isinstance(clean_tokens, Tensor) or not clean_tokens.is_floating_point():
    raise TypeError("clean_tokens must be a floating-point torch tensor")
  return clean_tokens.clone()


def clean_mask(
  *,
  current_index: int,
  state_dimension: int,
  token_dimension: int,
  device: torch.device | str,
) -> Tensor:
  """Build the D0 conditioning mask for a 41-step token trajectory."""
  if current_index < 0 or current_index >= _SEQUENCE_LENGTH:
    raise ValueError("current_index must be within the 41-step window")
  if state_dimension <= 0 or state_dimension > token_dimension:
    raise ValueError("state_dimension must be within token_dimension")
  mask = torch.zeros(
    (_SEQUENCE_LENGTH, token_dimension), dtype=torch.bool, device=device
  )
  if current_index:
    mask[:current_index] = True
  mask[current_index, :state_dimension] = True
  return mask


def apply_clean_mask(sample: Tensor, clean: Tensor, mask: Tensor) -> Tensor:
  """Replace fixed entries in ``sample`` with bit-identical clean values."""
  if sample.ndim < 2 or clean.ndim < 2:
    raise ValueError("sample and clean must include sequence and token dimensions")
  if sample.shape != clean.shape and sample.shape[-2:] != clean.shape[-2:]:
    raise ValueError("sample and clean shapes are incompatible")
  if mask.shape[-2:] != sample.shape[-2:]:
    raise ValueError("clean mask shape does not match sequence and token dimensions")
  clean_value = clean.to(device=sample.device, dtype=sample.dtype)
  mask_value = mask.to(device=sample.device, dtype=torch.bool)
  return torch.where(mask_value, clean_value, sample)


__all__ = [
  "NoisedPair",
  "add_independent_noise",
  "apply_clean_mask",
  "clean_mask",
  "sample_levels",
  "x0_target",
]
