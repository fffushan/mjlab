"""Frozen bidirectional state-latent Transformer denoiser."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

_FROZEN_PARAMETER_COUNT = 20_197_607


@dataclass(frozen=True, slots=True)
class DenoiserSettings:
  """Transformer dimensions and embedding settings from the D0 contract."""

  token_dimension: int = 231
  sequence_length: int = 41
  embedding_count: int = 1001
  width: int = 512
  layers: int = 6
  attention_heads: int = 8
  ffn_width: int = 2048
  dropout: float = 0.0
  activation: str = "gelu"
  norm_first: bool = True

  def __post_init__(self) -> None:
    if self.token_dimension <= 0 or self.sequence_length <= 0:
      raise ValueError("token and sequence dimensions must be positive")
    if self.embedding_count <= 0 or self.width <= 0 or self.layers <= 0:
      raise ValueError("embedding count, width and layers must be positive")
    if self.attention_heads <= 0 or self.width % self.attention_heads != 0:
      raise ValueError("width must be divisible by attention_heads")
    if self.ffn_width <= 0:
      raise ValueError("ffn_width must be positive")
    if self.dropout < 0.0 or self.dropout >= 1.0:
      raise ValueError("dropout must be in [0, 1)")
    if self.activation not in {"gelu", "relu"}:
      raise ValueError("activation must be gelu or relu")
    if not isinstance(self.norm_first, bool):
      raise TypeError("norm_first must be bool")

  @classmethod
  def from_contract(
    cls, contract: Any, *, training_k: int | None = None
  ) -> "DenoiserSettings":
    """Construct frozen model settings from a D0/D1 contract object."""
    k = 1000 if training_k is None else int(training_k)
    if k < 0:
      raise ValueError("training_k must be non-negative")
    token_dimension = int(getattr(contract, "token_dimension", 231))
    sequence_length = int(getattr(contract, "window_steps", 41))
    if token_dimension != 231 or sequence_length != 41:
      raise ValueError("contract dimensions disagree with the frozen D0 model")
    return cls(
      token_dimension=token_dimension,
      sequence_length=sequence_length,
      embedding_count=k + 1,
      width=512,
      layers=6,
      attention_heads=8,
      ffn_width=2048,
      dropout=0.0,
      activation="gelu",
      norm_first=True,
    )


class StateLatentTransformer(nn.Module):
  """Bidirectional Transformer that predicts clean state-latent tokens."""

  def __init__(self, settings: DenoiserSettings) -> None:
    super().__init__()
    self.settings = settings
    self.input_projection = nn.Linear(settings.token_dimension, settings.width)
    self.position_embedding = nn.Parameter(
      torch.zeros(settings.sequence_length, settings.width)
    )
    self.state_step_embedding = nn.Embedding(settings.embedding_count, settings.width)
    self.latent_step_embedding = nn.Embedding(settings.embedding_count, settings.width)
    layer = nn.TransformerEncoderLayer(
      d_model=settings.width,
      nhead=settings.attention_heads,
      dim_feedforward=settings.ffn_width,
      dropout=settings.dropout,
      activation=settings.activation,
      batch_first=True,
      norm_first=settings.norm_first,
    )
    # No mask is passed in forward: this is intentionally bidirectional.  A
    # final encoder norm is omitted because it is not part of the frozen count.
    self.encoder = nn.TransformerEncoder(
      layer,
      num_layers=settings.layers,
      norm=None,
      enable_nested_tensor=False,
    )
    self.output_projection = nn.Linear(settings.width, settings.token_dimension)

    if (
      self.parameter_count() != _FROZEN_PARAMETER_COUNT
      and settings == DenoiserSettings()
    ):
      raise RuntimeError(
        "frozen denoiser parameter count changed: "
        f"expected {_FROZEN_PARAMETER_COUNT}, got {self.parameter_count()}"
      )

  def forward(self, noisy_tokens: Tensor, step_ids: Tensor) -> Tensor:
    """Predict clean tokens from noisy tokens and two per-token step ids."""
    if noisy_tokens.ndim != 3:
      raise ValueError(
        "noisy_tokens must have shape [B, sequence_length, token_dimension]"
      )
    batch, sequence, width = noisy_tokens.shape
    if (
      sequence != self.settings.sequence_length
      or width != self.settings.token_dimension
    ):
      raise ValueError(
        "noisy_tokens shape does not match settings: "
        f"expected [B, {self.settings.sequence_length}, "
        f"{self.settings.token_dimension}], got {tuple(noisy_tokens.shape)}"
      )
    if not noisy_tokens.is_floating_point():
      raise TypeError("noisy_tokens must be floating point")
    if step_ids.shape != (batch, sequence, 2):
      raise ValueError(
        f"step_ids must have shape {(batch, sequence, 2)}, got {tuple(step_ids.shape)}"
      )
    if step_ids.dtype != torch.long:
      raise TypeError("step_ids must have torch.long dtype")
    if step_ids.device != noisy_tokens.device:
      raise ValueError("step_ids and noisy_tokens must be on the same device")
    if torch.any(step_ids < 0) or torch.any(step_ids >= self.settings.embedding_count):
      minimum = int(step_ids.min().item()) if step_ids.numel() else 0
      maximum = int(step_ids.max().item()) if step_ids.numel() else 0
      raise ValueError(
        "step_ids must be in "
        f"[0, {self.settings.embedding_count - 1}], got range [{minimum}, {maximum}]"
      )

    projected = self.input_projection(noisy_tokens)
    projected = projected + self.position_embedding.unsqueeze(0)
    projected = projected + self.state_step_embedding(step_ids[..., 0])
    projected = projected + self.latent_step_embedding(step_ids[..., 1])
    encoded = self.encoder(projected)
    return self.output_projection(encoded)

  def parameter_count(self) -> int:
    """Return the number of trainable and non-trainable model parameters."""
    return sum(parameter.numel() for parameter in self.parameters())


__all__ = ["DenoiserSettings", "StateLatentTransformer"]
