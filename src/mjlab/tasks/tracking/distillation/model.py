"""Conditional Gaussian VAE and explicit student normalization for M2.

This module is a pure tensor core.  Statistics are updated only through the
explicit :meth:`StudentNormalizer.update` method; model forward, mean
inference, and sampled inference never mutate them.  The default gravity
schema therefore reaches the decoder through the sampled (or mean) latent and
through its selected conditioning tensor only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from mjlab.tasks.tracking.distillation.vae_config import (
  ACTION_DIM,
  DEFAULT_MODEL_SETTINGS,
  DEFAULT_SCHEMA,
  LATENT_DIM,
  ModelSettings,
  VaeSchema,
)


class ModelValidationError(ValueError):
  """Raised when model, normalization, or loss inputs violate the contract."""


def _require_batch(name: str, value: torch.Tensor, width: int) -> None:
  if not isinstance(value, torch.Tensor):
    raise ModelValidationError(f"{name} must be a torch.Tensor")
  if value.ndim != 2 or value.shape[1] != width:
    raise ModelValidationError(f"{name} must have shape [B, {width}]")
  if value.shape[0] <= 0:
    raise ModelValidationError(f"{name} batch must be non-empty")
  if not value.is_floating_point():
    raise ModelValidationError(f"{name} must use a floating-point dtype")
  if not torch.isfinite(value).all().item():
    raise ModelValidationError(f"{name} contains non-finite values")


def _require_same_batch(name: str, value: torch.Tensor, batch: int) -> None:
  if value.shape[0] != batch:
    raise ModelValidationError(f"{name} batch size does not match")


def _require_finite_result(name: str, value: torch.Tensor) -> None:
  if not torch.isfinite(value).all().item():
    raise ModelValidationError(f"{name} produced non-finite values")


class StudentNormalizer(nn.Module):
  """A buffer-backed running normalizer with explicit, atomic updates.

  ``m2`` stores the sum of squared deviations and ``count`` is the number of
  observations.  The variance convention is population variance, ``m2 /``
  ``count``; normalization uses ``sqrt(variance + eps)``.  Updates reject the
  complete batch before changing any moment, and frozen instances reject
  updates rather than silently accepting stale training statistics.
  """

  def __init__(
    self,
    dimension: int,
    *,
    eps: float = 1e-5,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
  ) -> None:
    super().__init__()
    if not isinstance(dimension, int) or dimension <= 0:
      raise ValueError("normalizer dimension must be a positive integer")
    if (
      not isinstance(eps, (float, int))
      or not torch.isfinite(torch.tensor(float(eps)))
      or float(eps) <= 0
    ):
      raise ValueError("normalizer eps must be finite and positive")
    if not dtype.is_floating_point:
      raise ValueError("normalizer dtype must be floating point")
    self.dimension = dimension
    self.register_buffer("_eps", torch.tensor(float(eps), device=device, dtype=dtype))
    self.register_buffer("count", torch.zeros((), device=device, dtype=dtype))
    self.register_buffer("mean", torch.zeros(dimension, device=device, dtype=dtype))
    self.register_buffer("m2", torch.zeros(dimension, device=device, dtype=dtype))
    self.register_buffer(
      "_frozen", torch.tensor(False, device=device, dtype=torch.bool)
    )

  def _buffer(self, name: str) -> torch.Tensor:
    value = self._buffers.get(name)
    if not isinstance(value, torch.Tensor):
      raise RuntimeError(f"missing normalizer buffer {name!r}")
    return value

  @property
  def eps(self) -> float:
    """Numerical epsilon restored with this normalizer's state."""
    return float(self._buffer("_eps").item())

  @property
  def frozen(self) -> bool:
    return bool(self._buffer("_frozen").item())

  @property
  def variance(self) -> torch.Tensor:
    count = self._buffer("count")
    return self._buffer("m2") / count.clamp_min(1.0)

  def freeze(self) -> None:
    """Prevent future explicit moment updates, for evaluation or export."""
    self._buffer("_frozen").fill_(True)

  def unfreeze(self) -> None:
    """Allow explicit updates again."""
    self._buffer("_frozen").fill_(False)

  def update(self, samples: torch.Tensor) -> None:
    """Merge a non-empty raw sample batch using parallel Welford moments."""
    _require_batch("normalizer samples", samples, self.dimension)
    count = self._buffer("count")
    mean = self._buffer("mean")
    m2 = self._buffer("m2")
    if samples.device != mean.device or samples.dtype != mean.dtype:
      raise ModelValidationError(
        "normalizer samples must match the normalizer device and dtype"
      )
    if self.frozen:
      raise RuntimeError("normalizer is frozen")

    # Compute moments from detached values so statistics never retain an
    # autograd graph from the caller's raw batch.
    with torch.no_grad():
      batch_count = samples.new_tensor(float(samples.shape[0]))
      batch_mean = samples.mean(dim=0)
      batch_m2 = ((samples - batch_mean) ** 2).sum(dim=0)
      delta = batch_mean - mean
      total = count + batch_count
      new_mean = mean + delta * batch_count / total
      new_m2 = m2 + batch_m2 + delta.square() * count * batch_count / total
      for name, value in (
        ("normalizer count", total),
        ("normalizer mean", new_mean),
        ("normalizer m2", new_m2),
      ):
        _require_finite_result(name, value)
      count.copy_(total)
      mean.copy_(new_mean)
      m2.copy_(new_m2)

  def normalize(self, samples: torch.Tensor) -> torch.Tensor:
    """Normalize raw samples without updating moments or clipping values."""
    _require_batch("normalizer samples", samples, self.dimension)
    mean = self._buffer("mean")
    if samples.device != mean.device or samples.dtype != mean.dtype:
      raise ModelValidationError(
        "normalizer samples must match the normalizer device and dtype"
      )
    variance = self.variance
    # A cold-start normalizer is an identity transform (zero mean, unit
    # scale).  Explicit updates switch to empirical population variance.
    scale = torch.where(
      self._buffer("count") > 0,
      torch.sqrt(variance + self._buffer("_eps")),
      torch.ones_like(variance),
    )
    result = (samples - mean) / scale
    _require_finite_result("normalizer", result)
    return result

  def forward(self, samples: torch.Tensor) -> torch.Tensor:
    return self.normalize(samples)


# Descriptive aliases for callers using either terminology.
ControlledNormalizer = StudentNormalizer
RunningNormalizer = StudentNormalizer


class _Mlp(nn.Module):
  def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]) -> None:
    super().__init__()
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
      layers.extend((nn.Linear(previous, width), nn.ELU()))
      previous = width
    self.network = nn.Sequential(*layers)
    self.output_dim = previous

  def forward(self, value: torch.Tensor) -> torch.Tensor:
    return self.network(value)


@dataclass(frozen=True, slots=True)
class VaeOutput:
  """Outputs from one explicit VAE forward pass."""

  action: torch.Tensor
  mu: torch.Tensor
  logvar: torch.Tensor
  latent: torch.Tensor


@dataclass(frozen=True, slots=True)
class VaeLoss:
  """Summed-per-sample reconstruction and KL terms and their total."""

  total: torch.Tensor
  reconstruction: torch.Tensor
  kl: torch.Tensor


def _activation_settings(settings: ModelSettings) -> tuple[int, ...]:
  # ModelSettings currently constrains the activation to ELU.  Keeping this
  # check here makes a future serialized settings mismatch fail explicitly.
  if settings.activation != "ELU":
    raise ModelValidationError("only ELU model settings are supported")
  return settings.hidden_dims


class ConditionalVAE(nn.Module):
  """Conditional Gaussian VAE for one immutable :class:`VaeSchema`.

  ``decode`` has exactly two data inputs: a latent tensor and the selected
  conditioning tensor.  In particular it cannot receive reference q/dq,
  motion IDs, or teacher IDs as a hidden bypass.
  """

  def __init__(
    self,
    schema: VaeSchema = DEFAULT_SCHEMA,
    settings: ModelSettings = DEFAULT_MODEL_SETTINGS,
    *,
    normalizer_eps: float = 1e-5,
  ) -> None:
    super().__init__()
    if not isinstance(schema, VaeSchema):
      raise ModelValidationError("schema must be a VaeSchema")
    if not isinstance(settings, ModelSettings):
      raise ModelValidationError("settings must be ModelSettings")
    self.schema = schema
    self.settings = settings
    hidden_dims = _activation_settings(settings)
    self.reference_normalizer = StudentNormalizer(
      schema.reference_dim, eps=normalizer_eps
    )
    self.conditioning_normalizer = StudentNormalizer(
      schema.conditioning_dim, eps=normalizer_eps
    )
    self.encoder = _Mlp(schema.reference_dim, hidden_dims)
    self.mu_head = nn.Linear(self.encoder.output_dim, settings.latent_dim)
    self.logvar_head = nn.Linear(self.encoder.output_dim, settings.latent_dim)
    self.decoder = _Mlp(schema.decoder_input_dim, hidden_dims)
    self.action_head = nn.Linear(self.decoder.output_dim, schema.action_dim)

  @property
  def schema_metadata(self) -> dict[str, Any]:
    return self.schema.compatibility_metadata()

  def assert_schema_compatible(self, metadata: Mapping[str, Any]) -> None:
    """Reject a same-width checkpoint whose serialized schema differs."""
    self.schema.assert_compatible(dict(metadata))

  def get_extra_state(self) -> dict[str, Any]:
    """Include schema identity in ``state_dict`` roundtrips."""
    return {
      "schema": self.schema_metadata,
      "settings": self.settings.to_metadata(),
    }

  def set_extra_state(self, state: Any) -> None:
    if not isinstance(state, Mapping):
      raise ModelValidationError("VAE extra state must be a mapping")
    metadata = state.get("schema")
    if metadata != self.schema_metadata:
      raise ModelValidationError("serialized VAE schema does not match model schema")
    settings = state.get("settings")
    if settings != self.settings.to_metadata():
      raise ModelValidationError("serialized VAE settings do not match model settings")

  def _check_model_inputs(
    self, reference: torch.Tensor, conditioning: torch.Tensor
  ) -> None:
    _require_batch("reference", reference, self.schema.reference_dim)
    _require_batch("conditioning", conditioning, self.schema.conditioning_dim)
    _require_same_batch("conditioning", conditioning, reference.shape[0])
    if reference.device != conditioning.device or reference.dtype != conditioning.dtype:
      raise ModelValidationError(
        "reference and conditioning must share device and dtype"
      )

  def encode(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Gaussian mean and log variance from normalized reference data."""
    _require_batch("reference", reference, self.schema.reference_dim)
    normalized = self.reference_normalizer(reference)
    encoded = self.encoder(normalized)
    mu, logvar = self.mu_head(encoded), self.logvar_head(encoded)
    _require_finite_result("encoder mean", mu)
    _require_finite_result("encoder log variance", logvar)
    return mu, logvar

  def sample_latent(
    self,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    noise: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Sample ``mu + exp(0.5 * logvar) * noise`` with injectable noise."""
    _require_batch("mu", mu, LATENT_DIM)
    _require_batch("logvar", logvar, LATENT_DIM)
    _require_same_batch("logvar", logvar, mu.shape[0])
    if mu.device != logvar.device or mu.dtype != logvar.dtype:
      raise ModelValidationError("mu and logvar must share device and dtype")
    if noise is None:
      noise = torch.randn_like(mu)
    else:
      _require_batch("noise", noise, LATENT_DIM)
      _require_same_batch("noise", noise, mu.shape[0])
      if noise.device != mu.device or noise.dtype != mu.dtype:
        raise ModelValidationError("noise must match mu device and dtype")
    _require_finite_result("sampling noise", noise)
    latent = mu + torch.exp(0.5 * logvar) * noise
    _require_finite_result("sampled latent", latent)
    return latent

  def decode(self, latent: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
    """Decode only a latent and the schema-selected conditioning tensor."""
    _require_batch("latent", latent, self.settings.latent_dim)
    _require_batch("conditioning", conditioning, self.schema.conditioning_dim)
    _require_same_batch("conditioning", conditioning, latent.shape[0])
    if latent.device != conditioning.device or latent.dtype != conditioning.dtype:
      raise ModelValidationError("latent and conditioning must share device and dtype")
    normalized = self.conditioning_normalizer(conditioning)
    decoded = self.decoder(torch.cat((latent, normalized), dim=1))
    action = self.action_head(decoded)
    _require_finite_result("decoder action", action)
    return action

  def mean_inference(
    self, reference: torch.Tensor, conditioning: torch.Tensor
  ) -> torch.Tensor:
    """Decode the deterministic Gaussian mean, without sampling noise."""
    self._check_model_inputs(reference, conditioning)
    mu, _ = self.encode(reference)
    return self.decode(mu, conditioning)

  def sampled_inference(
    self,
    reference: torch.Tensor,
    conditioning: torch.Tensor,
    *,
    noise: torch.Tensor | None = None,
  ) -> VaeOutput:
    """Decode an explicit reparameterized sample, optionally with supplied noise."""
    self._check_model_inputs(reference, conditioning)
    mu, logvar = self.encode(reference)
    latent = self.sample_latent(mu, logvar, noise)
    return VaeOutput(self.decode(latent, conditioning), mu, logvar, latent)

  def forward(
    self,
    reference: torch.Tensor,
    conditioning: torch.Tensor,
    *,
    sample: bool = False,
    noise: torch.Tensor | None = None,
  ) -> VaeOutput:
    """Run explicit mean inference by default, or explicit sampled inference."""
    self._check_model_inputs(reference, conditioning)
    mu, logvar = self.encode(reference)
    latent = self.sample_latent(mu, logvar, noise) if sample else mu
    return VaeOutput(self.decode(latent, conditioning), mu, logvar, latent)


# Names used by different layers of the architecture document.
DistillationVAE = ConditionalVAE
VAE = ConditionalVAE


def vae_loss(
  predicted_action: torch.Tensor,
  teacher_action: torch.Tensor,
  mu: torch.Tensor,
  logvar: torch.Tensor,
  *,
  beta: float = DEFAULT_MODEL_SETTINGS.beta,
) -> VaeLoss:
  """Compute the exact summed-per-sample reconstruction plus Gaussian KL loss."""
  if (
    not isinstance(beta, (float, int))
    or not torch.isfinite(torch.tensor(float(beta)))
    or float(beta) < 0
  ):
    raise ModelValidationError("beta must be finite and non-negative")
  _require_batch("predicted_action", predicted_action, ACTION_DIM)
  _require_batch("teacher_action", teacher_action, ACTION_DIM)
  _require_batch("mu", mu, LATENT_DIM)
  _require_batch("logvar", logvar, LATENT_DIM)
  if predicted_action.shape != teacher_action.shape:
    raise ModelValidationError("predicted_action and teacher_action shapes must match")
  if mu.shape != logvar.shape:
    raise ModelValidationError("mu and logvar shapes must match")
  batch = predicted_action.shape[0]
  for name, value in (
    ("teacher_action", teacher_action),
    ("mu", mu),
    ("logvar", logvar),
  ):
    _require_same_batch(name, value, batch)
    if value.device != predicted_action.device or value.dtype != predicted_action.dtype:
      raise ModelValidationError("loss tensors must share device and dtype")
  reconstruction = (
    (predicted_action - teacher_action.detach()).square().sum(dim=1).mean()
  )
  kl_per_sample = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1)
  kl = kl_per_sample.mean()
  total = reconstruction + float(beta) * kl
  for name, value in (("reconstruction", reconstruction), ("kl", kl), ("total", total)):
    _require_finite_result(name, value)
  return VaeLoss(total=total, reconstruction=reconstruction, kl=kl)


reconstruction_kl_loss = vae_loss
compute_vae_loss = vae_loss

__all__ = [
  "ConditionalVAE",
  "ControlledNormalizer",
  "DistillationVAE",
  "ModelValidationError",
  "RunningNormalizer",
  "StudentNormalizer",
  "VAE",
  "VaeLoss",
  "VaeOutput",
  "compute_vae_loss",
  "reconstruction_kl_loss",
  "vae_loss",
]
