"""Focused D4a guidance tests: input VJP, clipping and frozen-loop invariants.

The denoisers here are small deterministic analytic stand-ins (one nonlinear
state/latent-coupled oracle, one parameterized module).  No learned checkpoint,
simulator, controller or hardware operation is involved, and no claim is made
about model quality.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor

from mjlab.tasks.tracking.diffusion.guidance import (
  GuidanceConfig,
  GuidanceError,
  clip_guidance_gradient,
  pose_guidance_gradient,
  sample_trajectory_guided,
)
from mjlab.tasks.tracking.diffusion.noising import apply_clean_mask, clean_mask
from mjlab.tasks.tracking.diffusion.pose_costs import (
  PoseCost,
  PoseCostWeights,
  TimedPoseGoal,
)
from mjlab.tasks.tracking.diffusion.projection import (
  FeatureStats,
  ProjectionBundle,
  fit_feature_stats,
)
from mjlab.tasks.tracking.diffusion.sampler import (
  SamplerConditions,
  SamplerConfig,
  sample_trajectory,
)
from mjlab.tasks.tracking.diffusion.schedule import DiffusionSchedule, InferenceGrid
from mjlab.tasks.tracking.diffusion.state import WorldState


def _stats(width: int, *, seed: int) -> FeatureStats:
  rng = np.random.default_rng(seed)
  return fit_feature_stats([rng.normal(size=(64, width)) + 3.0], std_floor=1e-6)


def _projection() -> ProjectionBundle:
  return ProjectionBundle.create(_stats(135, seed=1), _stats(32, seed=2))


def _state(
  position: tuple[float, float, float] = (0.0, 0.0, 0.0), yaw: float = 0.0
) -> WorldState:
  root = np.asarray(position, dtype=np.float64)
  cosine, sine = np.cos(yaw), np.sin(yaw)
  rotation = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
  local = np.arange(60, dtype=np.float64).reshape(20, 3) * 0.01
  return WorldState(
    root,
    np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]),
    np.array([0.2, -0.1, 0.03]),
    np.array([0.0, 0.1, -0.2]),
    root + local @ rotation.T,
    np.full((20, 3), 0.2),
  )


def _goal(tick: int, state: WorldState) -> TimedPoseGoal:
  return TimedPoseGoal(tick=tick, state=state, provenance="fixture")


def _cost(
  *,
  plan_tick: int = 0,
  goals: list[TimedPoseGoal] | None = None,
  weights: PoseCostWeights | None = None,
) -> PoseCost:
  return PoseCost(
    projection=_projection(),
    current_state=_state((0.5, -0.25, 0.75), yaw=0.4),
    plan_tick=plan_tick,
    goals=[_goal(5, _state((1.5, 0.5, 0.6), yaw=-0.3))] if goals is None else goals,
    weights=weights or PoseCostWeights(1.0, 1.0, 1.0),
  )


def _conditions(
  *, batch: int | None = None, fixed_row: int | None = None
) -> SamplerConditions:
  clean = (
    torch.randn(41, 167, dtype=torch.float64)
    if batch is None
    else torch.randn(batch, 41, 167, dtype=torch.float64)
  )
  mask = clean_mask(
    current_index=8, state_dimension=135, token_dimension=167, device=clean.device
  )
  if fixed_row is not None:
    mask[fixed_row, :135] = True
  return SamplerConditions(clean, mask)


class _CoupledOracle:
  """Nonlinear, cross-row state/latent-coupled stand-in for the frozen denoiser.

  The returned ``x0`` state slice depends on the whole window's latent content
  (and the returned latent slice on the whole window's state content), so a cost
  that reads only state channels at one future row still has a non-zero gradient
  with respect to latent entries of the noisy input, including the current row.
  That is what distinguishes an input VJP ``dG/dx_k`` from ``dG/dx0`` alone and
  lets guidance move the current latent and hence the decoded action.
  """

  def __init__(self) -> None:
    self.calls = 0
    self.inputs: list[Tensor] = []

  def __call__(self, sample: Tensor, step_ids: Tensor) -> Tensor:
    del step_ids
    self.calls += 1
    self.inputs.append(sample.detach().clone())
    state, latent = sample[..., :135], sample[..., 135:]
    latent_summary = torch.tanh(latent.mean(dim=(-2, -1), keepdim=True))
    state_summary = torch.tanh(state.mean(dim=(-2, -1), keepdim=True))
    state_mix = 0.5 * state + 0.25 * latent_summary
    latent_mix = 0.5 * latent + 0.25 * state_summary
    return torch.cat((state_mix, latent_mix), dim=-1)


class _ParameterizedDenoiser(torch.nn.Module):
  """Frozen-parameter stand-in that still exposes a real ``.grad`` slot."""

  def __init__(self) -> None:
    super().__init__()
    self.weight = torch.nn.Parameter(torch.full((167,), 0.25, dtype=torch.float64))
    self.calls = 0

  def forward(self, sample: Tensor, step_ids: Tensor) -> Tensor:
    del step_ids
    self.calls += 1
    return sample * self.weight * 0.5 + torch.tanh(sample) * 0.1


def test_input_vjp_matches_float64_finite_differences() -> None:
  cost = _cost()
  torch.manual_seed(3)
  x_k = torch.randn(1, 41, 167, dtype=torch.float64)
  step_ids = torch.full((1, 41, 2), 500, dtype=torch.long)
  clean_tokens = torch.zeros(41, 167, dtype=torch.float64)
  mask = clean_mask(
    current_index=8, state_dimension=135, token_dimension=167, device="cpu"
  )
  denoiser = _CoupledOracle()
  backward = pose_guidance_gradient(
    denoiser,
    x_k=x_k,
    step_ids=step_ids,
    clean_tokens=clean_tokens,
    clean_mask=mask,
    cost=cost,
  )
  assert denoiser.calls == 1
  assert float(backward.cost[0].detach()) > 0.0
  assert backward.gradient.shape == x_k.shape

  def value(candidate: Tensor) -> float:
    x0 = apply_clean_mask(denoiser(candidate, step_ids), clean_tokens, mask)
    return float(cost(x0).sum())

  # Entry (13, 140) is a latent channel: the cost never reads latent channels of
  # x0, so a non-zero numerical derivative there can only come from the VJP
  # through the denoiser.  Entry (13, 10) sits in the unused root-velocity slice
  # and must be zero in both the analytic and the numerical derivative.
  epsilon = 1e-6
  nonzero = ((13, 5), (13, 0), (13, 140), (8, 150), (8, 160))
  for row, channel in nonzero + ((13, 10), (13, 132), (8, 10)):
    plus, minus = x_k.clone(), x_k.clone()
    plus[0, row, channel] += epsilon
    minus[0, row, channel] -= epsilon
    numeric = (value(plus) - value(minus)) / (2.0 * epsilon)
    analytic = float(backward.gradient[0, row, channel])
    assert numeric == pytest.approx(analytic, rel=1e-6, abs=1e-7)
    if (row, channel) in nonzero:
      assert abs(analytic) > 1e-9
    else:
      assert analytic == 0.0


def test_clip_guidance_gradient_zeroes_fixed_entries_before_clipping() -> None:
  mask = clean_mask(
    current_index=8, state_dimension=135, token_dimension=167, device="cpu"
  )
  gradient = torch.zeros(1, 41, 167, dtype=torch.float64)
  gradient[0, 5, :] = 1000.0
  gradient[0, 13, 0:3] = torch.tensor([0.3, -0.4, 0.0], dtype=torch.float64)
  clipped = clip_guidance_gradient(gradient, fixed_mask=mask, clip_norm=1.0)
  assert torch.equal(clipped[0, :8], torch.zeros(8, 167, dtype=torch.float64))
  assert float(clipped[0, 13, 0:3].norm()) == pytest.approx(0.5, rel=1e-12)
  # Clipping before zeroing would have divided the free entries by the fixed-entry norm.
  total_norm = float(torch.linalg.vector_norm(gradient[0].reshape(-1)))
  naive = gradient / max(1.0, total_norm)
  naive = torch.where(mask, torch.zeros_like(gradient), naive)
  assert float(naive[0, 13, 0:3].norm()) == pytest.approx(0.5 / total_norm, rel=1e-12)
  assert float(naive[0, 13, 0:3].norm()) < 1e-3
  assert float(gradient[0, 5, :].abs().max()) == 1000.0


def test_clip_guidance_gradient_never_normalizes_small_gradients() -> None:
  mask = clean_mask(
    current_index=8, state_dimension=135, token_dimension=167, device="cpu"
  )
  gradient = torch.zeros(2, 41, 167, dtype=torch.float64)
  gradient[0, 13, 0:3] = torch.tensor([0.15, -0.2, 0.0], dtype=torch.float64)
  gradient[1, 13, 0:3] = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64)
  clipped = clip_guidance_gradient(gradient, fixed_mask=mask, clip_norm=1.0)
  assert torch.equal(clipped[0, 13, 0:3], gradient[0, 13, 0:3])
  assert float(torch.linalg.vector_norm(clipped[1].reshape(-1))) == pytest.approx(
    1.0, rel=1e-12
  )
  assert torch.equal(
    clipped[1, 13, 0:3], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
  )


def test_strength_zero_delegates_to_the_unguided_sampler_bitwise() -> None:
  schedule = DiffusionSchedule.build()
  grid = InferenceGrid.uniform(schedule)
  conditions = _conditions()
  cost = _cost()
  unguided, unguided_diagnostics = sample_trajectory(
    _CoupledOracle(),
    schedule=schedule,
    conditions=conditions,
    config=SamplerConfig(grid),
    generator=torch.Generator().manual_seed(5),
  )
  guided, diagnostics = sample_trajectory_guided(
    _CoupledOracle(),
    schedule=schedule,
    conditions=conditions,
    config=SamplerConfig(grid),
    generator=torch.Generator().manual_seed(5),
    guidance=GuidanceConfig(),
    cost=cost,
  )
  assert torch.equal(unguided, guided)
  assert diagnostics.guided is False
  assert diagnostics.vjp_calls == 0
  assert diagnostics.cost_calls == 0
  assert diagnostics.steps == ()
  assert diagnostics.denoiser_calls == unguided_diagnostics.denoiser_calls == 20
  assert diagnostics.jumps == 20
  assert diagnostics.source_ids_returned == unguided_diagnostics.source_ids_returned
  assert diagnostics.step_ids_seen == unguided_diagnostics.step_ids_seen
  assert diagnostics.final_index == unguided_diagnostics.final_index == 0
  assert diagnostics.initial_noise_norm == unguided_diagnostics.initial_noise_norm
  assert diagnostics.active_goal_ids == (5,)
  assert diagnostics.inactive_goal_ids == ()
  assert diagnostics.guidance_strength == 0.0
  assert diagnostics.gradient_clip_norm == 1.0


def test_guided_sampler_runs_exactly_twenty_forwards_and_vjps() -> None:
  schedule = DiffusionSchedule.build()
  grid = InferenceGrid.uniform(schedule)
  denoiser = _CoupledOracle()
  _, diagnostics = sample_trajectory_guided(
    denoiser,
    schedule=schedule,
    conditions=_conditions(),
    config=SamplerConfig(grid),
    generator=torch.Generator().manual_seed(9),
    guidance=GuidanceConfig(0.3),
    cost=_cost(),
  )
  assert denoiser.calls == 20
  assert diagnostics.denoiser_calls == 20
  assert diagnostics.vjp_calls == 20
  assert diagnostics.cost_calls == 20
  assert diagnostics.jumps == 20
  assert len(diagnostics.steps) == 20
  assert tuple(summary.source_id for summary in diagnostics.steps) == tuple(
    range(1000, 0, -50)
  )
  assert diagnostics.guided is True
  assert diagnostics.precision == "torch.float64"
  assert diagnostics.device == "cpu"


def test_guided_step_diagnostics_follow_the_reviewed_scaling() -> None:
  schedule = DiffusionSchedule.build()
  grid = InferenceGrid.uniform(schedule)
  far_goal = _goal(5, _state((100.0, 0.0, 0.0)))
  _, diagnostics = sample_trajectory_guided(
    _CoupledOracle(),
    schedule=schedule,
    conditions=_conditions(),
    config=SamplerConfig(grid),
    generator=torch.Generator().manual_seed(21),
    guidance=GuidanceConfig(0.5),
    cost=_cost(goals=[far_goal]),
  )
  alpha_bars = schedule.alpha_bars
  for summary in diagnostics.steps:
    alpha = float(alpha_bars[summary.source_id])
    assert summary.clipped_gradient_norm == pytest.approx(
      min(summary.gradient_norm, 1.0), rel=1e-12, abs=1e-15
    )
    assert summary.guidance_norm == pytest.approx(
      0.5 * (1.0 - alpha) * summary.clipped_gradient_norm, rel=1e-12
    )
  assert all(summary.gradient_norm > 1.0 for summary in diagnostics.steps)
  assert all(
    summary.clipped_gradient_norm == pytest.approx(1.0, rel=1e-12)
    for summary in diagnostics.steps
  )


def test_guided_sampler_changes_current_latent_and_decoded_action() -> None:
  schedule = DiffusionSchedule.build()
  grid = InferenceGrid.uniform(schedule)
  conditions = _conditions()
  cost = _cost()
  unguided, _ = sample_trajectory(
    _CoupledOracle(),
    schedule=schedule,
    conditions=conditions,
    config=SamplerConfig(grid),
    generator=torch.Generator().manual_seed(11),
  )
  guided, diagnostics = sample_trajectory_guided(
    _CoupledOracle(),
    schedule=schedule,
    conditions=conditions,
    config=SamplerConfig(grid),
    generator=torch.Generator().manual_seed(11),
    guidance=GuidanceConfig(0.5),
    cost=cost,
  )
  unguided_latent, guided_latent = unguided[8, 135:], guided[8, 135:]
  assert not torch.equal(unguided_latent, guided_latent)
  assert float((guided_latent - unguided_latent).abs().max()) > 1e-6
  decoder = torch.arange(32 * 31, dtype=torch.float64).reshape(32, 31) / (32 * 31)
  unguided_action, guided_action = unguided_latent @ decoder, guided_latent @ decoder
  assert not torch.equal(unguided_action, guided_action)
  assert float((guided_action - unguided_action).abs().max()) > 1e-6
  assert not torch.equal(unguided[13, :135], guided[13, :135])
  assert torch.equal(
    guided[conditions.clean_mask], conditions.clean_tokens[conditions.clean_mask]
  )
  assert diagnostics.fixed_state_rows == (0, 1, 2, 3, 4, 5, 6, 7, 8)
  assert diagnostics.fixed_goal_ids == ()
  assert diagnostics.active_goal_ids == (5,)


def test_guided_sampler_reports_fixed_goals_without_claiming_improvement() -> None:
  schedule = DiffusionSchedule.build()
  grid = InferenceGrid.uniform(schedule)
  conditions = _conditions(fixed_row=13)
  cost = _cost(goals=[_goal(5, _state((1.5, 0.5, 0.6), yaw=-0.3))])
  unguided, _ = sample_trajectory(
    _CoupledOracle(),
    schedule=schedule,
    conditions=conditions,
    config=SamplerConfig(grid),
    generator=torch.Generator().manual_seed(4),
  )
  guided, diagnostics = sample_trajectory_guided(
    _CoupledOracle(),
    schedule=schedule,
    conditions=conditions,
    config=SamplerConfig(grid),
    generator=torch.Generator().manual_seed(4),
    guidance=GuidanceConfig(0.5),
    cost=cost,
  )
  assert torch.equal(unguided, guided)
  assert diagnostics.guided is True
  assert diagnostics.fixed_goal_ids == (5,)
  assert diagnostics.fixed_state_rows == (0, 1, 2, 3, 4, 5, 6, 7, 8, 13)
  assert all(summary.gradient_norm == 0.0 for summary in diagnostics.steps)
  assert all(summary.guidance_norm == 0.0 for summary in diagnostics.steps)


def test_guided_sampler_skips_work_when_no_goal_is_active() -> None:
  schedule = DiffusionSchedule.build()
  grid = InferenceGrid.uniform(schedule)
  conditions = _conditions()
  cost = _cost(
    plan_tick=10,
    goals=[_goal(0, _state((1.0, 0.0, 0.0))), _goal(10, _state((2.0, 0.0, 0.0)))],
  )
  unguided, _ = sample_trajectory(
    _CoupledOracle(),
    schedule=schedule,
    conditions=conditions,
    config=SamplerConfig(grid),
    generator=torch.Generator().manual_seed(6),
  )
  guided, diagnostics = sample_trajectory_guided(
    _CoupledOracle(),
    schedule=schedule,
    conditions=conditions,
    config=SamplerConfig(grid),
    generator=torch.Generator().manual_seed(6),
    guidance=GuidanceConfig(0.75),
    cost=cost,
  )
  assert torch.equal(unguided, guided)
  assert diagnostics.guided is False
  assert diagnostics.active_goal_ids == ()
  assert diagnostics.inactive_goal_ids == (0, 10)
  assert diagnostics.vjp_calls == 0
  assert diagnostics.cost_calls == 0
  assert diagnostics.steps == ()
  assert diagnostics.guidance_strength == 0.75


def test_guided_sampler_leaves_parameters_rng_and_fixed_entries_untouched() -> None:
  schedule = DiffusionSchedule.build()
  grid = InferenceGrid.uniform(schedule)
  conditions = _conditions()
  module = _ParameterizedDenoiser()
  before = module.weight.detach().clone()
  unguided_generator = torch.Generator().manual_seed(7)
  unguided, _ = sample_trajectory(
    module,
    schedule=schedule,
    conditions=conditions,
    config=SamplerConfig(grid),
    generator=unguided_generator,
  )
  guided_generator = torch.Generator().manual_seed(7)
  guided, diagnostics = sample_trajectory_guided(
    module,
    schedule=schedule,
    conditions=conditions,
    config=SamplerConfig(grid),
    generator=guided_generator,
    guidance=GuidanceConfig(0.25),
    cost=_cost(),
  )
  assert module.calls == 40
  assert module.weight.grad is None
  assert torch.equal(module.weight.detach(), before)
  assert torch.equal(unguided_generator.get_state(), guided_generator.get_state())
  assert not torch.equal(unguided, guided)
  assert diagnostics.denoiser_calls == 20
  assert torch.equal(
    guided[conditions.clean_mask], conditions.clean_tokens[conditions.clean_mask]
  )


def test_guided_sampler_is_deterministic_and_keeps_batch_trajectories_separate() -> (
  None
):
  schedule = DiffusionSchedule.build()
  grid = InferenceGrid.uniform(schedule)
  conditions = _conditions(batch=2)
  cost = _cost()
  first, diagnostics = sample_trajectory_guided(
    _CoupledOracle(),
    schedule=schedule,
    conditions=conditions,
    config=SamplerConfig(grid),
    generator=torch.Generator().manual_seed(13),
    guidance=GuidanceConfig(0.4),
    cost=cost,
  )
  second, _ = sample_trajectory_guided(
    _CoupledOracle(),
    schedule=schedule,
    conditions=conditions,
    config=SamplerConfig(grid),
    generator=torch.Generator().manual_seed(13),
    guidance=GuidanceConfig(0.4),
    cost=cost,
  )
  assert first.shape == (2, 41, 167)
  assert torch.equal(first, second)
  assert not torch.equal(first[0], first[1])
  assert diagnostics.denoiser_calls == 20
  assert diagnostics.vjp_calls == 20
  for index in range(2):
    assert torch.equal(
      first[index][conditions.clean_mask],
      conditions.clean_tokens[index][conditions.clean_mask],
    )


def test_guided_sampler_rejects_inference_mode_and_bad_configs() -> None:
  schedule = DiffusionSchedule.build()
  grid = InferenceGrid.uniform(schedule)
  conditions = _conditions()
  cost = _cost()
  with pytest.raises(GuidanceError, match="inference_mode"):
    with torch.inference_mode():
      sample_trajectory_guided(
        _CoupledOracle(),
        schedule=schedule,
        conditions=conditions,
        config=SamplerConfig(grid),
        generator=torch.Generator().manual_seed(1),
        guidance=GuidanceConfig(0.5),
        cost=cost,
      )
  with pytest.raises(GuidanceError, match="strength"):
    GuidanceConfig(-1.0)
  with pytest.raises(GuidanceError, match="strength"):
    GuidanceConfig(float("nan"))
  with pytest.raises(GuidanceError, match="strength"):
    GuidanceConfig(True)
  with pytest.raises(GuidanceError, match="clip"):
    GuidanceConfig(0.5, 0.5)
  assert GuidanceConfig(0.5).gradient_clip_norm == 1.0

  local_grid = InferenceGrid(
    tuple(range(1000, 0, -50)), tuple(range(949, 48, -50)) + (0,)
  )
  with pytest.raises(GuidanceError, match="frozen inference grid"):
    sample_trajectory_guided(
      _CoupledOracle(),
      schedule=schedule,
      conditions=conditions,
      config=SamplerConfig(local_grid),
      generator=torch.Generator().manual_seed(1),
      guidance=GuidanceConfig(0.5),
      cost=cost,
    )
  with pytest.raises(GuidanceError, match="GuidanceConfig"):
    sample_trajectory_guided(
      _CoupledOracle(),
      schedule=schedule,
      conditions=conditions,
      config=SamplerConfig(grid),
      generator=torch.Generator().manual_seed(1),
      guidance=0.5,  # type: ignore[arg-type]
      cost=cost,
    )
  with pytest.raises(GuidanceError, match="PoseCost"):
    sample_trajectory_guided(
      _CoupledOracle(),
      schedule=schedule,
      conditions=conditions,
      config=SamplerConfig(grid),
      generator=torch.Generator().manual_seed(1),
      guidance=GuidanceConfig(0.5),
      cost=_CoupledOracle(),  # type: ignore[arg-type]
    )


def test_pose_guidance_gradient_rejects_bad_inputs() -> None:
  cost = _cost()
  x_k = torch.zeros(1, 41, 167, dtype=torch.float64)
  step_ids = torch.zeros(1, 41, 2, dtype=torch.long)
  clean_tokens = torch.zeros(41, 167, dtype=torch.float64)
  mask = clean_mask(
    current_index=8, state_dimension=135, token_dimension=167, device="cpu"
  )
  with pytest.raises(GuidanceError, match="x_k"):
    pose_guidance_gradient(
      _CoupledOracle(),
      x_k=x_k[0],
      step_ids=step_ids,
      clean_tokens=clean_tokens,
      clean_mask=mask,
      cost=cost,
    )
  with pytest.raises(GuidanceError, match="step_ids"):
    pose_guidance_gradient(
      _CoupledOracle(),
      x_k=x_k,
      step_ids=step_ids.to(torch.float64),
      clean_tokens=clean_tokens,
      clean_mask=mask,
      cost=cost,
    )
  with pytest.raises(GuidanceError, match="clean_mask"):
    pose_guidance_gradient(
      _CoupledOracle(),
      x_k=x_k,
      step_ids=step_ids,
      clean_tokens=clean_tokens,
      clean_mask=mask.to(torch.float64),
      cost=cost,
    )
  with pytest.raises(GuidanceError, match="clean_tokens"):
    pose_guidance_gradient(
      _CoupledOracle(),
      x_k=x_k,
      step_ids=step_ids,
      clean_tokens=torch.zeros(1, 41, 166, dtype=torch.float64),
      clean_mask=mask,
      cost=cost,
    )
  with pytest.raises(GuidanceError, match="token input shape"):
    pose_guidance_gradient(
      _CoupledOracleWrongShape(),
      x_k=x_k,
      step_ids=step_ids,
      clean_tokens=clean_tokens,
      clean_mask=mask,
      cost=cost,
    )


class _CoupledOracleWrongShape:
  def __call__(self, sample: Tensor, step_ids: Tensor) -> Tensor:
    del step_ids
    return sample[..., :100]


def test_guided_sampler_delegates_for_zero_strength_parity() -> None:
  """A zero-strength run must not depend on the pose-cost configuration at all."""
  schedule = DiffusionSchedule.build()
  grid = InferenceGrid.uniform(schedule)
  conditions = _conditions()
  far = _cost(goals=[_goal(5, _state((1000.0, 40.0, -5.0), yaw=2.0))])
  near = _cost(goals=[_goal(5, _state((0.5, -0.25, 0.75), yaw=0.4))])
  far_sample, far_diagnostics = sample_trajectory_guided(
    _CoupledOracle(),
    schedule=schedule,
    conditions=conditions,
    config=SamplerConfig(grid),
    generator=torch.Generator().manual_seed(17),
    guidance=GuidanceConfig(0.0),
    cost=far,
  )
  near_sample, _ = sample_trajectory_guided(
    _CoupledOracle(),
    schedule=schedule,
    conditions=conditions,
    config=SamplerConfig(grid),
    generator=torch.Generator().manual_seed(17),
    guidance=GuidanceConfig(0.0),
    cost=near,
  )
  assert torch.equal(far_sample, near_sample)
  assert far_diagnostics.vjp_calls == 0
