"""Denoiser construction for the latent-free joint-pose flow model.

The Transformer is the frozen diffusion denoiser, imported rather than copied.
This module only maps a :class:`JointPoseDatasetCfg` onto its settings, so the
runner and the evaluation tool cannot build different shapes.

The 68-channel joint-pose window is one state slice: there is no latent slice.
Both level-embedding slots of the denoiser therefore carry the same integer
noise level, which keeps the two-slot input contract and never hides the level
from the model.
"""

from __future__ import annotations

from ..diffusion.model import DenoiserSettings, StateLatentTransformer
from .config import JointPoseDatasetCfg


def denoiser_settings(
  dataset: JointPoseDatasetCfg,
  *,
  max_level: int,
  width: int | None = None,
  layers: int | None = None,
  heads: int | None = None,
) -> DenoiserSettings:
  """Return the denoiser settings for one prepared joint-pose dataset.

  ``width``, ``layers`` and ``heads`` override the frozen defaults so a test can
  run a small model; ``ffn_width``, ``dropout``, activation and norm stay at the
  diffusion defaults.
  """
  frozen = DenoiserSettings()
  return DenoiserSettings(
    token_dimension=dataset.channels,
    sequence_length=dataset.window_steps,
    embedding_count=max_level + 1,
    width=frozen.width if width is None else width,
    layers=frozen.layers if layers is None else layers,
    attention_heads=frozen.attention_heads if heads is None else heads,
    ffn_width=frozen.ffn_width,
    dropout=frozen.dropout,
    activation=frozen.activation,
    norm_first=frozen.norm_first,
  )


def build_denoiser(
  dataset: JointPoseDatasetCfg,
  *,
  max_level: int,
  width: int | None = None,
  layers: int | None = None,
  heads: int | None = None,
) -> StateLatentTransformer:
  """Build the x0-predicting denoiser for one prepared joint-pose dataset."""
  return StateLatentTransformer(
    denoiser_settings(
      dataset, max_level=max_level, width=width, layers=layers, heads=heads
    )
  )


__all__ = ["build_denoiser", "denoiser_settings"]
