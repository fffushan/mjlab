"""Exact linear Gaussian noising process for the X2 x0 flow-matching family.

The frozen contract for this family is ``s = k / max_level`` for an integer
level ``k`` in ``[0, max_level]``, ``x_s = (1 - s) * x0 + s * e`` with
independent Gaussian ``e``, and a network that predicts clean ``x0``.  Level 0 is
exactly clean, level ``max_level`` is exactly Gaussian noise, and state and
latent slices receive independent per-row levels and independent noise.  This
module defines no cosine schedule and no velocity objective.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

DEFAULT_MAX_LEVEL = 1000
FAMILY = "x0_linear_flow"
_INTEGER_DTYPES = (
  torch.int8,
  torch.int16,
  torch.int32,
  torch.int64,
  torch.uint8,
)


class FlowMatchingError(ValueError):
  """A flow-matching definition, input tensor or update is invalid."""


@dataclass(frozen=True, slots=True)
class LinearNoiseProcess:
  """The linear Gaussian interpolation ``x_s = (1 - s) * x0 + s * e``."""

  max_level: int = DEFAULT_MAX_LEVEL

  def __post_init__(self) -> None:
    if isinstance(self.max_level, bool) or not isinstance(self.max_level, int):
      raise FlowMatchingError("max_level must be an integer")
    if self.max_level < 1:
      raise FlowMatchingError("max_level must be positive")

  @property
  def embedding_count(self) -> int:
    """Number of integer level embeddings, including the clean level."""
    return self.max_level + 1

  def noise_fraction_at(self, level: int) -> float:
    """Return the exact noise fraction ``level / max_level``."""
    if isinstance(level, bool) or not isinstance(level, int):
      raise FlowMatchingError("noise level must be an integer")
    if level < 0 or level > self.max_level:
      raise FlowMatchingError(f"noise level {level} is outside [0, {self.max_level}]")
    return level / float(self.max_level)

  def noise_fractions_torch(
    self,
    levels: Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None = None,
  ) -> Tensor:
    """Convert integer levels to noise fractions in ``dtype``."""
    integer = _integer_levels(levels, name="levels", max_level=self.max_level)
    return integer.to(device=device, dtype=dtype) / float(self.max_level)

  def as_dict(self) -> dict[str, object]:
    """Return the process identity recorded in flow artifacts."""
    return {
      "family": FAMILY,
      "prediction_type": "x0",
      "path": "linear_gaussian",
      "loss": "x0_mse",
      "time_encoding": "integer_noise_fraction",
      "max_level": self.max_level,
      "embedding_count": self.embedding_count,
    }

  def identity_hash(self) -> str:
    """Return the stable hash of :meth:`as_dict`."""
    payload = json.dumps(self.as_dict(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


DEFAULT_PROCESS = LinearNoiseProcess()


@dataclass(frozen=True, slots=True)
class LevelPair:
  """Independent integer state and latent levels for one batch of rows."""

  state: Tensor
  latent: Tensor


@dataclass(frozen=True, slots=True)
class LinearNoisedPair:
  """Noised slices, their Gaussian draws and the integer levels used."""

  state: Tensor
  latent: Tensor
  noise_state: Tensor
  noise_latent: Tensor
  tokens: Tensor
  state_levels: Tensor
  latent_levels: Tensor


def _integer_levels(value: Any, *, name: str, max_level: int) -> Tensor:
  """Return ``value`` as an in-range ``torch.long`` level tensor."""
  if isinstance(value, bool):
    raise FlowMatchingError(f"{name} must be an integer level or integer tensor")
  if isinstance(value, int):
    levels = torch.tensor(value, dtype=torch.long)
  elif isinstance(value, Tensor):
    if value.dtype not in _INTEGER_DTYPES:
      raise FlowMatchingError(f"{name} must use an integer dtype, got {value.dtype}")
    levels = value.to(dtype=torch.long)
  else:
    raise FlowMatchingError(f"{name} must be an integer level or integer tensor")
  if levels.numel() == 0:
    raise FlowMatchingError(f"{name} must not be empty")
  minimum = int(levels.min().item())
  maximum = int(levels.max().item())
  if minimum < 0 or maximum > max_level:
    raise FlowMatchingError(
      f"{name} must lie in [0, {max_level}], got [{minimum}, {maximum}]"
    )
  return levels


def draw_gaussian(
  shape: tuple[int, ...] | torch.Size,
  *,
  generator: torch.Generator,
  device: torch.device | str,
  dtype: torch.dtype,
) -> Tensor:
  """Draw standard Gaussian noise reproducibly from ``generator``.

  The generator owns the draw sequence.  When the generator device differs from
  ``device`` the draw is transferred afterwards, so the sequence depends only on
  the generator state and the requested shape.
  """
  target = torch.device(device)
  generator_device = getattr(generator, "device", None)
  draw_device = target if generator_device is None else torch.device(generator_device)
  value = torch.randn(shape, generator=generator, device=draw_device, dtype=dtype)
  return value if draw_device == target else value.to(device=target)


def sample_independent_levels(
  shape: tuple[int, ...] | torch.Size,
  *,
  process: LinearNoiseProcess = DEFAULT_PROCESS,
  generator: torch.Generator,
) -> LevelPair:
  """Draw independent per-row state and latent integer levels.

  Two separate ``torch.randint`` draws are made on purpose, state first and then
  latent, so the two slices never share one draw.  Both draws are inclusive of
  level 0 and level ``process.max_level``.
  """
  normalized = tuple(int(value) for value in shape)
  if not normalized or any(value <= 0 for value in normalized):
    raise FlowMatchingError("level shape dimensions must be positive")
  device = getattr(generator, "device", None)
  state = torch.randint(
    0,
    process.embedding_count,
    normalized,
    generator=generator,
    device=device,
    dtype=torch.long,
  )
  latent = torch.randint(
    0,
    process.embedding_count,
    normalized,
    generator=generator,
    device=device,
    dtype=torch.long,
  )
  return LevelPair(state, latent)


def _validate_slice(value: Any, name: str) -> Tensor:
  if not isinstance(value, Tensor):
    raise FlowMatchingError(f"{name} must be a torch tensor")
  if value.ndim < 1:
    raise FlowMatchingError(f"{name} must have at least one dimension")
  if not value.is_floating_point():
    raise FlowMatchingError(f"{name} must be a floating-point tensor")
  if not bool(torch.isfinite(value).all()):
    raise FlowMatchingError(f"{name} contains non-finite values")
  return value


def _levels_for(
  value: Tensor, levels: Any, *, name: str, process: LinearNoiseProcess
) -> Tensor:
  expected = tuple(value.shape[:-1])
  integer = _integer_levels(levels, name=name, max_level=process.max_level)
  if tuple(integer.shape) != expected:
    raise FlowMatchingError(
      f"{name} must have shape {expected} for one row per token position, "
      f"got {tuple(integer.shape)}"
    )
  return integer


def add_linear_noise(
  state: Tensor,
  latent: Tensor,
  *,
  state_levels: Any,
  latent_levels: Any,
  process: LinearNoiseProcess = DEFAULT_PROCESS,
  generator: torch.Generator,
) -> LinearNoisedPair:
  """Form ``x_s = (1 - s) * x0 + s * e`` for state and latent slices.

  ``state_levels`` and ``latent_levels`` hold one inclusive integer level for
  every leading token position.  The state and latent noise draws are separate
  on purpose so the two slices never alias.  Level 0 returns the clean slice
  exactly and level ``process.max_level`` returns the Gaussian draw exactly.
  """
  _validate_slice(state, "state")
  _validate_slice(latent, "latent")
  if state.shape[:-1] != latent.shape[:-1]:
    raise FlowMatchingError("state and latent leading dimensions must match")
  if state.dtype != latent.dtype:
    raise FlowMatchingError("state and latent must have the same dtype")
  if state.device != latent.device:
    raise FlowMatchingError("state and latent must be on the same device")
  k_state = _levels_for(state, state_levels, name="state_levels", process=process)
  k_latent = _levels_for(latent, latent_levels, name="latent_levels", process=process)

  noise_state = draw_gaussian(
    state.shape, generator=generator, device=state.device, dtype=state.dtype
  )
  noise_latent = draw_gaussian(
    latent.shape, generator=generator, device=latent.device, dtype=latent.dtype
  )
  fraction_state = process.noise_fractions_torch(
    k_state, dtype=state.dtype, device=state.device
  )
  fraction_latent = process.noise_fractions_torch(
    k_latent, dtype=latent.dtype, device=latent.device
  )
  noised_state = (1.0 - fraction_state).unsqueeze(-1) * state + (
    fraction_state.unsqueeze(-1) * noise_state
  )
  noised_latent = (1.0 - fraction_latent).unsqueeze(-1) * latent + (
    fraction_latent.unsqueeze(-1) * noise_latent
  )
  return LinearNoisedPair(
    state=noised_state,
    latent=noised_latent,
    noise_state=noise_state,
    noise_latent=noise_latent,
    tokens=torch.cat((noised_state, noised_latent), dim=-1),
    state_levels=k_state,
    latent_levels=k_latent,
  )


__all__ = [
  "DEFAULT_MAX_LEVEL",
  "DEFAULT_PROCESS",
  "FAMILY",
  "FlowMatchingError",
  "LevelPair",
  "LinearNoiseProcess",
  "LinearNoisedPair",
  "add_linear_noise",
  "draw_gaussian",
  "sample_independent_levels",
]
