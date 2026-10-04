"""D4a physical pose-cost input-VJP guidance on the frozen DDIM reverse grid.

This module owns one seam: computing the pose-cost gradient with respect to the
*current noisy* trajectory (an input VJP through the frozen denoiser) and
applying the reviewed engineering update at every reverse level.  It reuses the
frozen DDIM arithmetic (:func:`~.sampler.ddim_jump`), conditioning mask
(:func:`~.noising.apply_clean_mask`) and the sampler's own batching, step-id,
state-boundary, initial-norm and Gaussian-RNG helpers, so the guided and
unguided paths share one conditioning convention instead of re-deriving it.

Per reverse level ``k``:

1. the current ``x_k`` is detached into a grad-enabled leaf and the frozen
   denoiser is called exactly once;
2. clean conditions are re-applied to the predicted ``x0``;
3. the physical cost ``G`` is evaluated on that clean ``x0`` and
   ``g = dG/dx_k`` is taken with :func:`torch.autograd.grad`, so no parameter
   ``.grad`` is written and the graph is released at the next level;
4. fixed entries are zeroed *before* the per-trajectory L2 clip to
   ``GuidanceConfig.gradient_clip_norm`` (1.0), and gradients already inside the
   radius are left untouched rather than normalized;
5. ``x0_guided = x0 - strength * (1 - alpha_bar[k]) * g_clipped``;
6. the jump is recomputed with the frozen DDIM update and every fixed entry is
   restored bit-for-bit before the next level starts from a detached tensor.

Nothing here runs a decoder backward, an optimizer step, a normalizer update or
an extra denoiser pass, and no latent/action clipping is applied.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .noising import apply_clean_mask
from .pose_costs import PoseCost
from .sampler import (
  SamplerConfig,
  _as_batched,
  _condition_values,
  _infer_state_dimension,
  _initial_norms,
  _randn,
  _step_ids,
  ddim_jump,
  sample_trajectory,
)
from .schedule import DiffusionSchedule, build_inference_grid


class GuidanceError(ValueError):
  """A guidance configuration, gradient or guided update is invalid."""


@dataclass(frozen=True, slots=True)
class GuidanceConfig:
  """Configuration for physical pose-cost input-VJP guidance.

  ``strength`` is the guidance weight ``lambda``; it must be finite and
  non-negative, and ``0.0`` selects the original unguided sampler exactly.
  ``gradient_clip_norm`` is pinned to the declared ``1.0``: the per-trajectory
  L2 clip of the physical gradient is part of the reviewed update, not a free
  tuning knob.
  """

  strength: float = 0.0
  gradient_clip_norm: float = 1.0

  def __post_init__(self) -> None:
    if isinstance(self.strength, bool) or not isinstance(self.strength, (int, float)):
      raise GuidanceError("guidance strength must be a finite non-negative number")
    strength = float(self.strength)
    if not math.isfinite(strength) or strength < 0.0:
      raise GuidanceError(
        f"guidance strength must be finite and non-negative, got {strength}"
      )
    object.__setattr__(self, "strength", strength)
    if self.gradient_clip_norm != 1.0:
      raise GuidanceError(
        "the guidance gradient clip norm is pinned to 1.0, got "
        f"{self.gradient_clip_norm}"
      )


@dataclass(frozen=True, slots=True)
class PoseGuidanceGradient:
  """One reverse-level input VJP: clean prediction, ``dG/dx_k`` and ``G``."""

  x0_clean: Tensor
  gradient: Tensor
  cost: Tensor


@dataclass(frozen=True, slots=True)
class GuidanceStepSummary:
  """Bounded scalars recorded at one guided reverse source level.

  ``cost`` is the maximum per-trajectory pose cost at that level and every norm
  is the maximum over the batch, so no per-entry or per-goal tensor is retained.
  """

  source_id: int
  cost: float
  gradient_norm: float
  clipped_gradient_norm: float
  guidance_norm: float


@dataclass(frozen=True, slots=True)
class GuidedSamplerDiagnostics:
  """Observable reverse-loop diagnostics of one guided (or delegated) sample.

  ``denoiser_calls``, ``jumps``, ``source_ids_returned``, ``step_ids_seen``,
  ``final_index`` and ``initial_noise_norm`` have the same meaning as in
  :class:`~.sampler.SamplerDiagnostics`.  ``active_goal_ids`` are the soft goals
  inside the window, ``inactive_goal_ids`` are reached/expired/beyond-horizon
  goals that did no work, and ``fixed_goal_ids`` are active goals whose
  physical-state row is hard-fixed, so guidance cannot move them and their
  predicted cost must not be reported as an improvement.
  """

  denoiser_calls: int
  vjp_calls: int
  cost_calls: int
  jumps: int
  source_ids_returned: tuple[int, ...]
  step_ids_seen: tuple[int, ...]
  final_index: int
  initial_noise_norm: dict[str, float]
  guided: bool
  guidance_strength: float
  gradient_clip_norm: float
  active_goal_ids: tuple[int, ...]
  inactive_goal_ids: tuple[int, ...]
  fixed_goal_ids: tuple[int, ...]
  fixed_state_rows: tuple[int, ...]
  precision: str
  device: str
  steps: tuple[GuidanceStepSummary, ...]


def _validate_inputs(
  denoiser: Any,
  schedule: DiffusionSchedule,
  config: SamplerConfig,
  guidance: GuidanceConfig,
  cost: PoseCost,
  generator: torch.Generator,
) -> None:
  if torch.is_inference_mode_enabled():
    raise GuidanceError(
      "pose-cost guidance needs an input-VJP graph, which torch.inference_mode() "
      "disables; call the guided sampler under torch.no_grad() instead"
    )
  if not callable(denoiser):
    raise GuidanceError("denoiser must be a callable (sample, step_ids) -> x0_hat")
  if not isinstance(guidance, GuidanceConfig):
    raise GuidanceError("guidance must be a GuidanceConfig")
  if not isinstance(cost, PoseCost):
    raise GuidanceError("cost must be a PoseCost")
  if config.eta != 0.0:
    raise GuidanceError("D2 supports only eta=0.0")
  if config.guidance_strength != 0.0:
    raise GuidanceError("the guided sampler reuses D2 rejection of legacy guidance")
  if config.grid != build_inference_grid(schedule):
    raise GuidanceError("guided sampling requires the frozen inference grid")
  if not isinstance(generator, torch.Generator):
    raise GuidanceError("generator must be a torch.Generator")


def _fixed_state_rows(mask_batched: Tensor, state_dimension: int) -> tuple[int, ...]:
  """Return window rows whose physical-state slice is fixed in every trajectory."""
  fixed = mask_batched[..., :state_dimension].all(dim=-1)
  return tuple(row for row in range(fixed.shape[-1]) if bool(fixed[:, row].all()))


def _fixed_goal_ids(cost: PoseCost, fixed_rows: tuple[int, ...]) -> tuple[int, ...]:
  """Return the active goal ticks whose state row is hard-fixed."""
  fixed = set(fixed_rows)
  return tuple(goal.tick for goal in cost.active_goals if goal.row in fixed)


def pose_guidance_gradient(
  denoiser: Callable[[Tensor, Tensor], Tensor],
  *,
  x_k: Tensor,
  step_ids: Tensor,
  clean_tokens: Tensor,
  clean_mask: Tensor,
  cost: PoseCost,
) -> PoseGuidanceGradient:
  """Differentiate ``G(clean x0)`` with respect to the noisy input ``x_k``.

  This is the independently testable VJP seam: a finite-difference check in
  float64 with a state/latent-coupled denoiser validates exactly this function.
  ``x_k`` is detached into a grad-enabled leaf before the single denoiser call,
  and the gradient is taken with :func:`torch.autograd.grad` so no parameter
  ``.grad`` is populated.
  """
  if not isinstance(x_k, Tensor) or not x_k.is_floating_point() or x_k.ndim != 3:
    raise GuidanceError("x_k must be a floating-point torch tensor [B, T, D]")
  if (
    not isinstance(step_ids, Tensor)
    or step_ids.dtype != torch.long
    or step_ids.shape != (x_k.shape[0], x_k.shape[1], 2)
  ):
    raise GuidanceError("step_ids must be a long tensor [B, T, 2]")
  if (
    not isinstance(clean_tokens, Tensor)
    or not clean_tokens.is_floating_point()
    or clean_tokens.shape[-2:] != x_k.shape[-2:]
  ):
    raise GuidanceError("clean_tokens must cover the x_k [T, D] dimensions")
  if (
    not isinstance(clean_mask, Tensor)
    or clean_mask.dtype != torch.bool
    or clean_mask.shape[-2:] != x_k.shape[-2:]
  ):
    raise GuidanceError("clean_mask must be a bool tensor over [T, D]")

  leaf = x_k.detach().clone().requires_grad_(True)
  with torch.enable_grad():
    x0_hat = denoiser(leaf, step_ids)
    if not isinstance(x0_hat, Tensor) or x0_hat.shape != leaf.shape:
      raise GuidanceError("denoiser must return a tensor with the token input shape")
    x0_clean = apply_clean_mask(x0_hat, clean_tokens, clean_mask)
    cost_value = cost(x0_clean)
    if not isinstance(cost_value, Tensor) or tuple(cost_value.shape) != (x_k.shape[0],):
      raise GuidanceError("pose cost must return one value per trajectory")
    if not bool(torch.isfinite(cost_value).all()):
      raise GuidanceError("pose cost is non-finite at this reverse level")
    gradient = torch.autograd.grad(cost_value.sum(), leaf, allow_unused=True)[0]
  if gradient is None:
    # No path from the cost to the noisy input means zero guidance for this
    # level (for example when every active goal row is hard-fixed).
    gradient = torch.zeros_like(leaf)
  if not bool(torch.isfinite(gradient).all()):
    raise GuidanceError("pose-cost gradient is non-finite at this reverse level")
  return PoseGuidanceGradient(x0_clean=x0_clean, gradient=gradient, cost=cost_value)


def clip_guidance_gradient(
  gradient: Tensor, *, fixed_mask: Tensor, clip_norm: float
) -> Tensor:
  """Zero fixed entries, then clip each trajectory's L2 norm to ``clip_norm``.

  The order is part of the reviewed update: the per-trajectory clip must see
  only the entries guidance may actually move, otherwise a large fixed-entry
  gradient would shrink an otherwise small free gradient.  A gradient already
  inside the radius is returned unchanged and never normalized to unit length.
  """
  if not isinstance(gradient, Tensor) or not gradient.is_floating_point():
    raise GuidanceError("gradient must be a floating-point torch tensor")
  if gradient.ndim != 3:
    raise GuidanceError("gradient must have shape [B, T, D]")
  if not isinstance(fixed_mask, Tensor) or fixed_mask.dtype != torch.bool:
    raise GuidanceError("fixed_mask must be a bool torch tensor")
  if fixed_mask.shape[-2:] != gradient.shape[-2:]:
    raise GuidanceError("fixed_mask must cover the gradient's [T, D] dimensions")
  if not math.isfinite(clip_norm) or clip_norm <= 0.0:
    raise GuidanceError("clip_norm must be finite and positive")
  masked = torch.where(
    fixed_mask.to(device=gradient.device),
    torch.zeros_like(gradient),
    gradient,
  )
  norms = torch.linalg.vector_norm(masked.reshape(masked.shape[0], -1), dim=-1)
  scale = torch.minimum(torch.ones_like(norms), clip_norm / norms.clamp_min(clip_norm))
  return masked * scale.reshape((-1,) + (1,) * (masked.ndim - 1))


def _delegated_diagnostics(
  sample: Tensor,
  diagnostics: Any,
  cost: PoseCost,
  guidance: GuidanceConfig,
  fixed_rows: tuple[int, ...],
) -> GuidedSamplerDiagnostics:
  """Describe the zero-work delegation to the original unguided sampler."""
  return GuidedSamplerDiagnostics(
    denoiser_calls=diagnostics.denoiser_calls,
    vjp_calls=0,
    cost_calls=0,
    jumps=diagnostics.jumps,
    source_ids_returned=diagnostics.source_ids_returned,
    step_ids_seen=diagnostics.step_ids_seen,
    final_index=diagnostics.final_index,
    initial_noise_norm=diagnostics.initial_noise_norm,
    guided=False,
    guidance_strength=guidance.strength,
    gradient_clip_norm=guidance.gradient_clip_norm,
    active_goal_ids=tuple(goal.tick for goal in cost.active_goals),
    inactive_goal_ids=cost.inactive_goal_ticks,
    fixed_goal_ids=_fixed_goal_ids(cost, fixed_rows),
    fixed_state_rows=fixed_rows,
    precision=str(sample.dtype),
    device=str(sample.device),
    steps=(),
  )


def sample_trajectory_guided(
  denoiser: Any,
  *,
  schedule: DiffusionSchedule,
  conditions: Mapping[str, Tensor] | Any,
  config: SamplerConfig,
  generator: torch.Generator,
  guidance: GuidanceConfig,
  cost: PoseCost,
) -> tuple[Tensor, GuidedSamplerDiagnostics]:
  """Run the frozen DDIM grid with optional physical pose-cost guidance.

  With ``strength == 0`` or no active soft goal the original
  :func:`~.sampler.sample_trajectory` runs unchanged: the same RNG draws, the
  same 20 calls and bit-identical output, with no cost evaluation and no VJP.
  Otherwise the reviewed per-level update above is applied and the returned
  diagnostics expose the actual forward/VJP/cost counts and bounded per-level
  cost and norm scalars.
  """
  _validate_inputs(denoiser, schedule, config, guidance, cost, generator)
  clean_tokens, clean_mask = _condition_values(conditions)
  clean_batched, mask_batched, squeezed = _as_batched(clean_tokens, clean_mask)
  state_dimension = _infer_state_dimension(mask_batched)
  if state_dimension >= clean_batched.shape[-1]:
    raise GuidanceError("clean mask does not leave a latent slice")
  fixed_rows = _fixed_state_rows(mask_batched, state_dimension)

  if guidance.strength == 0.0 or not cost.active_goals:
    sample, diagnostics = sample_trajectory(
      denoiser,
      schedule=schedule,
      conditions=conditions,
      config=config,
      generator=generator,
    )
    return sample, _delegated_diagnostics(
      sample, diagnostics, cost, guidance, fixed_rows
    )

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
  initial_norm = _initial_norms(noise, state_dimension)
  fixed_state = mask_batched[..., :state_dimension].all(dim=-1)
  fixed_latent = mask_batched[..., state_dimension:].all(dim=-1)
  alpha_bars = schedule.alpha_bars_torch(device=sample.device, dtype=sample.dtype)

  start_index = config.grid.source_ids.index(config.start_level)
  active_sources = config.grid.source_ids[start_index:]
  active_destinations = config.grid.destination_ids[start_index:]
  summaries: list[GuidanceStepSummary] = []
  seen: list[int] = []
  returned: list[int] = []
  vjp_calls = 0
  cost_calls = 0

  with torch.no_grad():
    for source, destination in zip(active_sources, active_destinations, strict=True):
      if source == schedule.clean_index:
        raise GuidanceError("the denoiser must never be called at clean index 0")
      step_ids = _step_ids(
        source=int(source),
        fixed_state=fixed_state,
        fixed_latent=fixed_latent,
      )
      backward = pose_guidance_gradient(
        denoiser,
        x_k=sample,
        step_ids=step_ids,
        clean_tokens=clean_batched,
        clean_mask=mask_batched,
        cost=cost,
      )
      vjp_calls += 1
      cost_calls += 1
      gradient = backward.gradient
      clipped = clip_guidance_gradient(
        gradient,
        fixed_mask=mask_batched,
        clip_norm=guidance.gradient_clip_norm,
      )
      applied = (guidance.strength * (1.0 - alpha_bars[int(source)])) * clipped
      x0_guided = backward.x0_clean - applied
      jumped = ddim_jump(
        sample,
        x0_guided,
        schedule=schedule,
        source=int(source),
        destination=int(destination),
      )
      sample = apply_clean_mask(jumped, clean_batched, mask_batched)
      seen.append(int(source))
      returned.append(int(source))
      summaries.append(
        GuidanceStepSummary(
          source_id=int(source),
          cost=float(backward.cost.detach().max()),
          gradient_norm=_max_row_norm(gradient),
          clipped_gradient_norm=_max_row_norm(clipped),
          guidance_norm=_max_row_norm(applied),
        )
      )

  diagnostics = GuidedSamplerDiagnostics(
    denoiser_calls=len(returned),
    vjp_calls=vjp_calls,
    cost_calls=cost_calls,
    jumps=len(active_sources),
    source_ids_returned=tuple(returned),
    step_ids_seen=tuple(seen),
    final_index=config.grid.destination_ids[-1],
    initial_noise_norm=initial_norm,
    guided=True,
    guidance_strength=guidance.strength,
    gradient_clip_norm=guidance.gradient_clip_norm,
    active_goal_ids=tuple(goal.tick for goal in cost.active_goals),
    inactive_goal_ids=cost.inactive_goal_ticks,
    fixed_goal_ids=_fixed_goal_ids(cost, fixed_rows),
    fixed_state_rows=fixed_rows,
    precision=str(sample.dtype),
    device=str(sample.device),
    steps=tuple(summaries),
  )
  if squeezed:
    return sample[0], diagnostics
  return sample, diagnostics


def _max_row_norm(value: Tensor) -> float:
  """Return the maximum per-trajectory L2 norm of ``value``."""
  norms = torch.linalg.vector_norm(value.reshape(value.shape[0], -1), dim=-1)
  return float(norms.max())


__all__ = [
  "GuidanceConfig",
  "GuidanceError",
  "GuidanceStepSummary",
  "GuidedSamplerDiagnostics",
  "PoseGuidanceGradient",
  "clip_guidance_gradient",
  "pose_guidance_gradient",
  "sample_trajectory_guided",
]
