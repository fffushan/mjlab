"""Ablation variant: the shared state-latent Transformer with a final LayerNorm.

``StateLatentTransformer`` ends with ``output_projection(_encode(...))``.  Its
encoder is built with ``norm=None``, because the frozen D0 parameter count omits
a final encoder norm, so the output projection reads the residual stream
directly.  Internal pre-normalisation bounds the input of each block but not the
residual stream itself.  This module adds the missing norm as a separate variant
so that the change can be measured against the unchanged model.

Use this class only for an explicit ablation.  The frozen model, its parameter
count identity and its checkpoint loading path stay untouched, and the two
classes are not interchangeable without a state-dict conversion: the variant
holds two extra parameter tensors, ``final_norm.weight`` and ``final_norm.bias``
(``2 * width`` values).  :func:`load_base_state_dict` performs that conversion
and refuses any state dict that is not a one-to-one match apart from those two
tensors.

Both classes draw their initial weights from the process RNG in the same order,
and ``nn.LayerNorm`` initialises to constants without consuming random numbers.
A variant built directly after a base model under the same seed therefore shares
every base weight with it; :func:`build_denoiser` in ``joint_pose/model.py``
exposes ``init_seed`` to make that pairing explicit.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from .model import DenoiserSettings, StateLatentTransformer

FINAL_NORM_PARAMETER_NAMES = ("final_norm.weight", "final_norm.bias")
"""Parameters the variant adds to the shared Transformer."""


class StateLatentTransformerFinalNorm(StateLatentTransformer):
  """The shared Transformer with a LayerNorm before the output projection.

  Every other part is inherited unchanged: the input projection, the three
  embeddings, the six pre-normalised blocks and the output projection.  Only the
  final normalisation is new, so a difference measured against the base class is
  attributable to it alone.
  """

  def __init__(self, settings: DenoiserSettings) -> None:
    super().__init__(settings)
    # Default LayerNorm initialisation is weight 1, bias 0, which consumes no
    # random numbers: the base weights above stay identical to a base model
    # built under the same process RNG state.
    self.final_norm = nn.LayerNorm(settings.width)

  def forward(self, noisy_tokens: Tensor, step_ids: Tensor) -> Tensor:
    """Predict clean tokens, normalising the residual stream first."""
    encoded = self._encode(noisy_tokens, step_ids)
    return self.output_projection(self.final_norm(encoded))

  def extra_parameter_count(self) -> int:
    """Return the number of parameters this variant adds to the base model."""
    return sum(parameter.numel() for parameter in self.final_norm.parameters())


def load_base_state_dict(
  model: StateLatentTransformerFinalNorm, state_dict: Mapping[str, Tensor]
) -> None:
  """Load a base-model or variant state dict into the variant.

  A base checkpoint lacks the two ``final_norm`` tensors.  Those are allowed to
  be missing and keep their initialisation.  Every other key must match exactly:
  an unexpected key, or a missing key outside ``FINAL_NORM_PARAMETER_NAMES``, is
  a real mismatch and raises instead of being ignored.
  """
  if not isinstance(model, StateLatentTransformerFinalNorm):
    raise TypeError("model must be a StateLatentTransformerFinalNorm")
  missing, unexpected = model.load_state_dict(dict(state_dict), strict=False)
  allowed = set(FINAL_NORM_PARAMETER_NAMES)
  unexpected_missing = sorted(set(missing) - allowed)
  if unexpected_missing or unexpected:
    raise RuntimeError(
      "checkpoint does not match the final-norm variant: "
      f"missing {unexpected_missing}, unexpected {sorted(unexpected)}"
    )


def build_final_norm_denoiser(
  settings: DenoiserSettings, *, init_seed: int | None = None
) -> StateLatentTransformerFinalNorm:
  """Build the variant, optionally from a reproducible initial draw.

  ``init_seed`` reseeds the CPU RNG around construction only, so a base model and
  a variant built with the same seed start from identical shared weights.  The
  surrounding RNG state is restored afterwards.
  """
  if init_seed is None:
    return StateLatentTransformerFinalNorm(settings)
  with torch.random.fork_rng(devices=[]):
    torch.manual_seed(int(init_seed) % (2**63))
    return StateLatentTransformerFinalNorm(settings)


__all__ = [
  "FINAL_NORM_PARAMETER_NAMES",
  "StateLatentTransformerFinalNorm",
  "build_final_norm_denoiser",
  "load_base_state_dict",
]
