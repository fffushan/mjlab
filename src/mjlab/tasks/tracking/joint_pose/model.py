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

import torch

from ..diffusion.model import DenoiserSettings, StateLatentTransformer
from ..diffusion.model_final_norm import build_final_norm_denoiser
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
  final_norm: bool = False,
  init_seed: int | None = None,
) -> StateLatentTransformer:
  """Build the x0-predicting denoiser for one prepared joint-pose dataset.

  ``final_norm`` selects the ablation variant that normalises the residual
  stream before the output projection; ``False`` builds the shared Transformer
  unchanged.  ``init_seed`` makes the initial draw reproducible, so a base arm
  and a final-norm arm can start from identical shared weights.
  """
  settings = denoiser_settings(
    dataset, max_level=max_level, width=width, layers=layers, heads=heads
  )
  if final_norm:
    return build_final_norm_denoiser(settings, init_seed=init_seed)
  if init_seed is None:
    return StateLatentTransformer(settings)
  with torch.random.fork_rng(devices=[]):
    torch.manual_seed(int(init_seed) % (2**63))
    return StateLatentTransformer(settings)


__all__ = ["build_denoiser", "denoiser_settings"]
