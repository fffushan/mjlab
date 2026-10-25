"""Frozen bidirectional state-latent Transformer denoiser."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

_FROZEN_PARAMETER_COUNT = 20_132_007


@dataclass(frozen=True, slots=True)
class ContractDimensions:
  """State, latent, token and window widths declared by a diffusion contract.

  The frozen full no-mixture contract declares ``(135, 32, 167, 41)``.  A
  reduced local-kinematic contract removes the 3-D root position and 3-D root
  linear velocity slices and declares ``(129, 32, 161, 41)``.  Nothing here
  assumes either width: the numbers come from the contract, so the model and
  loader support a reduced contract without a literal 135 or 167.
  """

  state_dimension: int
  latent_dimension: int
  token_dimension: int
  window_steps: int

  @property
  def label(self) -> str:
    """Short ``state-D/token-channel`` description used in failure messages."""
    return f"{self.state_dimension}-D state/{self.token_dimension}-channel token"


def contract_dimensions(contract: Any) -> ContractDimensions:
  """Read and validate the model layout a contract declares.

  The no-mixture token layout is exactly the physical-state width followed by
  the latent width.  This reads the declared widths instead of a literal
  ``167``, so the full 135-D contract and the reduced 129-D local-kinematic
  contract both resolve.  A contract whose token width is not its state width
  plus its latent width is rejected, which is the invariant the frozen model
  has always relied on.
  """
  fields = ("state_dimension", "latent_dimension", "token_dimension", "window_steps")
  try:
    values = {name: int(getattr(contract, name)) for name in fields}
  except (AttributeError, TypeError, ValueError) as exc:
    raise ValueError(
      f"contract does not declare the model state/latent/token widths: {exc}"
    ) from exc
  if min(values.values()) <= 0:
    raise ValueError("contract state, latent, token and window widths must be positive")
  dimensions = ContractDimensions(
    state_dimension=values["state_dimension"],
    latent_dimension=values["latent_dimension"],
    token_dimension=values["token_dimension"],
    window_steps=values["window_steps"],
  )
  if dimensions.token_dimension != (
    dimensions.state_dimension + dimensions.latent_dimension
  ):
    raise ValueError(
      "contract token width must equal its state width plus its latent width: "
      f"{dimensions.token_dimension} != {dimensions.state_dimension} + "
      f"{dimensions.latent_dimension}"
    )
  return dimensions


@dataclass(frozen=True, slots=True)
class DenoiserSettings:
  """Transformer dimensions for the active no-mixture token contract."""

  token_dimension: int = 167
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
    """Construct model settings from the widths a contract declares.

    The frozen full no-mixture contract yields the 167/41 settings.  A reduced
    local-kinematic contract yields the 161/41 settings.  The frozen D0
    parameter-count identity in :class:`StateLatentTransformer` still guards the
    default settings, so the full path keeps its frozen shape and weight count.
    """
    k = 1000 if training_k is None else int(training_k)
    if k < 0:
      raise ValueError("training_k must be non-negative")
    dimensions = contract_dimensions(contract)
    return cls(
      token_dimension=dimensions.token_dimension,
      sequence_length=dimensions.window_steps,
      embedding_count=k + 1,
      width=512,
      layers=6,
      attention_heads=8,
      ffn_width=2048,
      dropout=0.0,
      activation="gelu",
      norm_first=True,
    )


_MIXED_PRECISION_DTYPES = {"bf16-pure": torch.bfloat16}


def denoiser_dtype(mixed_precision: str) -> torch.dtype | None:
  """Return the parameter dtype a run stores the model in.

  ``None`` means fp32 parameters, which is what both ``fp32`` and plain ``bf16``
  mode use: the latter casts only inside the forward, so its master weights and
  optimizer state stay fp32.  ``bf16-pure`` stores everything in bf16.
  """
  return _MIXED_PRECISION_DTYPES.get(mixed_precision)


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
    return self.output_projection(self._encode(noisy_tokens, step_ids))

  def _encode(self, noisy_tokens: Tensor, step_ids: Tensor) -> Tensor:
    """Validate the inputs and return the encoder residual stream.

    The residual stream is returned unnormalised, because this class omits the
    final encoder norm.  A subclass that adds one applies it to this value
    before the output projection, not inside this method.
    """
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
    return self.encoder(projected)

  def parameter_count(self) -> int:
    """Return the number of trainable and non-trainable model parameters."""
    return sum(parameter.numel() for parameter in self.parameters())


__all__ = [
  "ContractDimensions",
  "DenoiserSettings",
  "StateLatentTransformer",
  "contract_dimensions",
]
