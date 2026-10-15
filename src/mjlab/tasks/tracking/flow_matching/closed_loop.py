"""Flow-family sampler adapter for the existing closed-loop continuation seam.

The continuation seams (``diffusion.handover.ReferenceFreeContinuation`` and
``diffusion.parallel_closed_loop_x2.BatchedReferenceFreeContinuation``) sample
with the frozen diffusion ``sample_trajectory`` by default and read
``denoiser_calls`` plus a terminal ``final_index`` from its diagnostics.  This
module supplies the same seam for the x0 linear-flow family without touching
the diffusion sampler:

* :class:`FlowSamplingAdapter` closes over the flow model and its
  :class:`~mjlab.tasks.tracking.flow_matching.sampler.FlowSamplerConfig`, builds
  the existing :class:`~mjlab.tasks.tracking.flow_matching.sampler.FlowConditions`
  from the seam's clean tokens and mask, and calls the existing
  :func:`~mjlab.tasks.tracking.flow_matching.sampler.sample_trajectory`;
* :class:`FlowContinuationDiagnostics` reports the flow sampler's terminal
  ``final_level`` under the seam's ``final_index`` name.  The flow family
  defines no diffusion ladder, so no diffusion grid index exists for it.

The flow family builds no diffusion schedule or inference grid, so a
continuation that receives this adapter takes neither.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from ..diffusion.sampler import SamplerConditions
from .sampler import FlowConditions, FlowSamplerConfig, sample_trajectory

# The flow network consumes (token sample, [B, T, 2] integer step ids).
FlowDenoiser = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True, slots=True)
class FlowContinuationDiagnostics:
  """Flow sampling diagnostics under the continuation seam's field names.

  ``denoiser_calls`` is the number of flow-network evaluations.  ``final_index``
  is the sampler's terminal level, which any accepted :class:`FlowGrid` ends at
  clean level 0; the seam reads it as the ladder's terminal index because a flow
  run has no diffusion grid index to report.
  """

  denoiser_calls: int
  final_index: int


class FlowSamplingAdapter:
  """Sampler callable that runs the x0 flow family inside the closed-loop seam.

  Pass one instance as the continuation's injected ``sampler``.  The adapter
  owns the frozen flow model and its sampler config, so the seam needs no
  diffusion schedule, inference grid or ``SamplerConfig``; the seam supplies
  the built conditions and its shared generator.  One adapter serves both the
  single-row and the batched continuation, because the flow sampler accepts
  ``[T, D]`` and ``[B, T, D]`` clean tokens.

  ``state_dimension`` is the state/latent token boundary used by the flow
  sampler's mask handling.  The continuation's clean mask carries a partial
  current row, so the sampler infers it; pass
  ``contract.projected_state_dimension`` explicitly to avoid inference.
  """

  def __init__(
    self,
    *,
    denoiser: FlowDenoiser,
    config: FlowSamplerConfig,
    state_dimension: int | None = None,
  ) -> None:
    if not callable(denoiser):
      raise ValueError("the flow denoiser must be callable")
    if not isinstance(config, FlowSamplerConfig):
      raise ValueError("config must be a FlowSamplerConfig")
    self.denoiser = denoiser
    self.config = config
    self.state_dimension = state_dimension

  def __call__(
    self, conditions: SamplerConditions, generator: torch.Generator
  ) -> tuple[Tensor, FlowContinuationDiagnostics]:
    """Sample one continuation conditioning batch with the flow sampler."""
    sample, diagnostics = sample_trajectory(
      self.denoiser,
      conditions=FlowConditions(
        clean_tokens=conditions.clean_tokens,
        clean_mask=conditions.clean_mask,
        state_dimension=self.state_dimension,
      ),
      config=self.config,
      generator=generator,
    )
    return sample, FlowContinuationDiagnostics(
      denoiser_calls=int(diagnostics.denoiser_calls),
      final_index=int(diagnostics.final_level),
    )


__all__ = [
  "FlowContinuationDiagnostics",
  "FlowDenoiser",
  "FlowSamplingAdapter",
]
