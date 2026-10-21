"""Integration tests for the injected flow sampler in the closed-loop seam.

These tests drive the real linear-flow sampler through the existing single-row
and batched continuations with small deterministic stand-in networks.  They
check sampler dispatch, the clean-conditioning contract and decoded shapes, plus
one real-architecture step.  No trained checkpoint, simulator, physics or
closed-loop quality claim is involved.  The last tests confirm that the default
diffusion path and its arguments are unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tracking_diffusion_fixtures import make_state

from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT
from mjlab.tasks.tracking.diffusion.handover import (
  HandoverError,
  HandoverHistory,
  ReferenceFreeContinuation,
  build_reference_free_plan,
)
from mjlab.tasks.tracking.diffusion.model import (
  DenoiserSettings,
  StateLatentTransformer,
)
from mjlab.tasks.tracking.diffusion.parallel_closed_loop_x2 import (
  BatchedReferenceFreeContinuation,
  ParallelRuntimeX2Error,
)
from mjlab.tasks.tracking.diffusion.projection import (
  ProjectionBundle,
  fit_feature_stats,
)
from mjlab.tasks.tracking.diffusion.sampler import SamplerConditions, SamplerConfig
from mjlab.tasks.tracking.diffusion.schedule import DiffusionSchedule, InferenceGrid
from mjlab.tasks.tracking.diffusion.state import WorldState
from mjlab.tasks.tracking.flow_matching.closed_loop import (
  FlowContinuationDiagnostics,
  FlowSamplingAdapter,
)
from mjlab.tasks.tracking.flow_matching.process import LinearNoiseProcess
from mjlab.tasks.tracking.flow_matching.sampler import FlowGrid, FlowSamplerConfig

_FLOW_LEVEL = 100
_FLOW_CALLS = 4


class _FlowNetwork:
  """Small x0 stand-in for the frozen flow network, driven by the real sampler."""

  def __init__(self) -> None:
    self.calls = 0
    self.step_ids: list[torch.Tensor] = []

  def __call__(self, sample: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
    if sample.ndim != 3 or step_ids.shape != (*sample.shape[:2], 2):
      raise AssertionError("the flow network expects [B, T, D] and [B, T, 2] ids")
    if step_ids.dtype != torch.long:
      raise AssertionError("flow step ids must be integer source levels")
    self.calls += 1
    self.step_ids.append(step_ids.detach().clone())
    return torch.tanh(sample * 0.5)


class _DiffusionNetwork:
  """Stand-in for the frozen DDIM network; the sampler owns the step semantics."""

  def __init__(self) -> None:
    self.calls = 0

  def __call__(self, sample: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
    del step_ids
    self.calls += 1
    return 0.1 * sample


class _Decoder:
  """Row-independent gravity-mode decoder stand-in returning ``[rows, 31]``."""

  def __call__(self, latent: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
    if latent.ndim != 2 or latent.shape[1] != DEFAULT_CONTRACT.latent_dimension:
      raise AssertionError("decoder latent width is wrong")
    if conditioning.shape != (latent.shape[0], 99):
      raise AssertionError("decoder conditioning width is wrong")
    return latent[:, :31] * 0.5


class _RecordingSampler:
  """Injected sampler stand-in: it records its inputs and reports fixed counts."""

  def __init__(self, *, denoiser_calls: int = 5, final_index: int = 0) -> None:
    self.denoiser_calls = denoiser_calls
    self.final_index = final_index
    self.calls = 0
    self.conditions: list[SamplerConditions] = []
    self.generators: list[torch.Generator] = []

  def __call__(
    self, conditions: SamplerConditions, generator: torch.Generator
  ) -> tuple[torch.Tensor, FlowContinuationDiagnostics]:
    self.calls += 1
    self.conditions.append(conditions)
    self.generators.append(generator)
    return torch.full_like(conditions.clean_tokens, 0.25), FlowContinuationDiagnostics(
      denoiser_calls=self.denoiser_calls, final_index=self.final_index
    )


def _stats(width: int, *, seed: int):
  rng = np.random.default_rng(seed)
  return fit_feature_stats([rng.normal(size=(64, width)) + 3.0], std_floor=1e-6)


def _projection() -> ProjectionBundle:
  return ProjectionBundle.create(
    _stats(DEFAULT_CONTRACT.state_dimension, seed=1),
    _stats(DEFAULT_CONTRACT.latent_dimension, seed=2),
  )


def _history() -> HandoverHistory:
  history = HandoverHistory()
  for index in range(DEFAULT_CONTRACT.past_steps):
    history.push(make_state(index), np.full(32, 0.05 * index))
  return history


def _flow_adapter(
  denoiser,
  *,
  max_level: int = _FLOW_LEVEL,
  calls: int = _FLOW_CALLS,
  state_dimension: int | None = None,
) -> FlowSamplingAdapter:
  return FlowSamplingAdapter(
    denoiser=denoiser,
    config=FlowSamplerConfig(
      grid=FlowGrid.uniform(calls=calls, max_level=max_level),
      process=LinearNoiseProcess(max_level=max_level),
    ),
    state_dimension=state_dimension,
  )


def _bootstrap(continuation) -> None:
  for pair in _history().pairs:
    continuation.record_executed(pair.state, pair.latent)


def _batched_rows(rows: int, index: int) -> WorldState:
  return WorldState.stack([make_state(index)] * rows)


def _flow_continuation(**overrides: object) -> ReferenceFreeContinuation:
  """Build the single-row seam with the given overrides, for failure checks."""
  arguments: dict[str, object] = {
    "denoiser": _DiffusionNetwork(),
    "decode": _Decoder(),
    "projection": _projection(),
    "generator": torch.Generator().manual_seed(0),
  }
  arguments.update(overrides)
  return ReferenceFreeContinuation(**arguments)  # type: ignore[arg-type]


def test_single_row_step_calls_the_injected_sampler_with_the_frozen_plan() -> None:
  sampler = _RecordingSampler(denoiser_calls=7, final_index=3)
  continuation = ReferenceFreeContinuation(
    denoiser=_FlowNetwork(),
    decode=_Decoder(),
    projection=_projection(),
    generator=torch.Generator().manual_seed(0),
    sampler=sampler,
  )
  _bootstrap(continuation)

  step = continuation.step(make_state(8), np.linspace(-0.5, 0.5, 99), tick=8)

  assert sampler.calls == 1
  assert sampler.generators[0].initial_seed() == 0
  assert step.denoiser_calls == 7
  assert step.final_index == 3
  assert step.action.shape == (31,) and np.isfinite(step.action).all()
  assert step.executed_latent.shape == (32,) and np.isfinite(step.executed_latent).all()
  # The seam hands the injected sampler the frozen plan conditioning unchanged.
  plan = continuation.plan(make_state(8), tick=9)
  assert torch.equal(sampler.conditions[0].clean_tokens, plan.clean_tokens)
  assert torch.equal(sampler.conditions[0].clean_mask, plan.clean_mask)


def test_batched_step_calls_the_injected_sampler_once_for_the_row_batch() -> None:
  rows = 3
  sampler = _RecordingSampler(denoiser_calls=4, final_index=0)
  continuation = BatchedReferenceFreeContinuation(
    denoiser=_FlowNetwork(),
    decode=_Decoder(),
    projection=_projection(),
    device="cpu",
    generator=torch.Generator().manual_seed(0),
    sampler=sampler,
    rows=rows,
  )
  for index in range(DEFAULT_CONTRACT.past_steps):
    continuation.record_executed(
      np.arange(rows),
      _batched_rows(rows, index),
      np.tile(np.full(32, 0.05 * index), (rows, 1)),
    )

  step = continuation.step(
    _batched_rows(rows, 8),
    np.tile(np.linspace(-0.5, 0.5, 99), (rows, 1)),
    tick=8,
    rows=[0, 1, 2],
  )

  assert sampler.calls == 1
  assert sampler.generators[0].initial_seed() == 0
  assert step.denoiser_calls == 4 and step.final_index == 0
  assert step.actions.shape == (rows, 31) and np.isfinite(step.actions).all()
  assert step.executed_latents.shape == (rows, 32)
  assert np.isfinite(step.executed_latents).all()
  assert sampler.conditions[0].clean_tokens.shape == (rows, 41, 167)
  assert sampler.conditions[0].clean_mask.shape == (41, 167)


def test_flow_adapter_keeps_the_clean_mask_semantics_of_the_seam() -> None:
  network = _FlowNetwork()
  adapter = _flow_adapter(network)
  plan = build_reference_free_plan(
    _history(), make_state(8), projection=_projection(), tick=0
  )

  sample, diagnostics = adapter(
    SamplerConditions(plan.clean_tokens, plan.clean_mask),
    torch.Generator().manual_seed(0),
  )

  assert network.calls == _FLOW_CALLS
  assert diagnostics.denoiser_calls == _FLOW_CALLS
  assert diagnostics.final_index == 0
  assert sample.shape == plan.clean_tokens.shape
  assert torch.isfinite(sample).all()
  # Known past/current entries come back bit-identical; unknown entries are
  # generated, because the flow sampler restores the mask after every update.
  known = plan.clean_mask
  assert torch.equal(sample[known], plan.clean_tokens[known])
  assert not torch.equal(sample[~known], torch.zeros_like(sample[~known]))
  # The mask reaches the network as level 0 on known state/latent slices and as
  # the source level on every unknown row.
  ids = network.step_ids[0][0]
  assert ids.shape == (41, 2)
  assert torch.equal(ids[:8], torch.zeros_like(ids[:8]))
  assert torch.equal(ids[8], torch.tensor([0, _FLOW_LEVEL]))
  assert torch.equal(ids[9:], torch.full_like(ids[9:], _FLOW_LEVEL))


def test_batched_flow_step_decodes_finite_actions_for_every_row() -> None:
  rows = 2
  network = _FlowNetwork()
  continuation = BatchedReferenceFreeContinuation(
    denoiser=network,
    decode=_Decoder(),
    projection=_projection(),
    device="cpu",
    generator=torch.Generator().manual_seed(0),
    sampler=_flow_adapter(network),
    rows=rows,
  )
  for index in range(DEFAULT_CONTRACT.past_steps):
    continuation.record_executed(
      np.arange(rows),
      _batched_rows(rows, index),
      np.tile(np.full(32, 0.05 * index), (rows, 1)),
    )

  step = continuation.step(
    _batched_rows(rows, 8),
    np.tile(np.linspace(-0.5, 0.5, 99), (rows, 1)),
    tick=8,
    rows=[0, 1],
  )

  # One batched flow sampling call covers every row, so the call count is the
  # grid size, not the grid size times the row count.
  assert network.calls == _FLOW_CALLS
  assert step.denoiser_calls == _FLOW_CALLS and step.final_index == 0
  assert step.actions.shape == (rows, 31) and np.isfinite(step.actions).all()
  assert step.executed_latents.shape == (rows, 32)
  assert np.isfinite(step.executed_latents).all()
  assert not np.allclose(step.executed_latents[0], step.executed_latents[1])


def test_single_row_flow_step_with_the_real_flow_architecture_decodes_actions() -> None:
  process = LinearNoiseProcess(max_level=_FLOW_LEVEL)
  model = StateLatentTransformer(
    DenoiserSettings.from_contract(DEFAULT_CONTRACT, training_k=process.max_level)
  ).eval()
  continuation = ReferenceFreeContinuation(
    denoiser=model,
    decode=_Decoder(),
    projection=_projection(),
    generator=torch.Generator().manual_seed(0),
    sampler=_flow_adapter(
      model,
      calls=2,
      state_dimension=DEFAULT_CONTRACT.projected_state_dimension,
    ),
  )
  _bootstrap(continuation)

  with torch.no_grad():
    step = continuation.step(make_state(8), np.linspace(-0.5, 0.5, 99), tick=8)

  assert step.denoiser_calls == 2 and step.final_index == 0
  assert step.action.shape == (31,) and np.isfinite(step.action).all()
  assert step.executed_latent.shape == (32,) and np.isfinite(step.executed_latent).all()
  predicted = step.predicted_next_hybrid_state
  assert predicted is not None and predicted.shape == (
    DEFAULT_CONTRACT.state_dimension,
  )
  assert np.isfinite(predicted).all()


def test_default_diffusion_path_is_unchanged_without_an_injected_sampler() -> None:
  schedule = DiffusionSchedule.build()
  grid = InferenceGrid.uniform(schedule)
  continuation = ReferenceFreeContinuation(
    denoiser=_DiffusionNetwork(),
    decode=_Decoder(),
    schedule=schedule,
    grid=grid,
    projection=_projection(),
    generator=torch.Generator().manual_seed(0),
  )
  _bootstrap(continuation)
  step = continuation.step(make_state(8), np.linspace(-0.5, 0.5, 99), tick=8)
  assert step.denoiser_calls == 20 and step.final_index == 0
  assert step.action.shape == (31,) and np.isfinite(step.action).all()

  rows = 2
  batched = BatchedReferenceFreeContinuation(
    denoiser=_DiffusionNetwork(),
    decode=_Decoder(),
    schedule=schedule,
    projection=_projection(),
    contract=DEFAULT_CONTRACT,
    device="cpu",
    generator=torch.Generator().manual_seed(0),
    sampler_config=SamplerConfig(grid, start_level=1000),
    rows=rows,
  )
  for index in range(DEFAULT_CONTRACT.past_steps):
    batched.record_executed(
      np.arange(rows),
      _batched_rows(rows, index),
      np.tile(np.full(32, 0.05 * index), (rows, 1)),
    )
  measured = batched.step(
    _batched_rows(rows, 8), np.zeros((rows, 99)), tick=8, rows=[0, 1]
  )
  assert measured.denoiser_calls == 20 and measured.final_index == 0
  assert measured.actions.shape == (rows, 31) and np.isfinite(measured.actions).all()


def test_injected_sampler_rejects_the_diffusion_arguments_and_accepts_a_task_factory() -> (
  None
):
  adapter = _flow_adapter(_FlowNetwork())
  # A task factory may replace the tick's sampler; its result then takes the
  # family convention (conditions, generator).  Only the diffusion arguments are
  # refused with an injected family sampler.
  _flow_continuation(sampler=adapter, sampler_factory=lambda plan, state: adapter)
  with pytest.raises(HandoverError, match="owns its own solver and level grid"):
    _flow_continuation(sampler=adapter, schedule=DiffusionSchedule.build())
  with pytest.raises(HandoverError, match="owns its own solver and level grid"):
    _flow_continuation(
      sampler=adapter, grid=InferenceGrid.uniform(DiffusionSchedule.build())
    )
  with pytest.raises(HandoverError, match="owns its own solver and level grid"):
    _flow_continuation(
      sampler=adapter,
      sampler_config=SamplerConfig(InferenceGrid.uniform(DiffusionSchedule.build())),
    )
  with pytest.raises(
    ParallelRuntimeX2Error, match="owns its own solver and level grid"
  ):
    BatchedReferenceFreeContinuation(
      denoiser=_DiffusionNetwork(),
      decode=_Decoder(),
      projection=_projection(),
      generator=torch.Generator().manual_seed(0),
      sampler=adapter,
      sampler_config=SamplerConfig(InferenceGrid.uniform(DiffusionSchedule.build())),
      rows=1,
    )


def test_continuations_fail_closed_without_a_sampler_or_diffusion_arguments() -> None:
  with pytest.raises(HandoverError, match="requires both schedule and grid"):
    _flow_continuation()
  with pytest.raises(HandoverError, match="sampler must be callable"):
    _flow_continuation(sampler="not a sampler")
  with pytest.raises(ParallelRuntimeX2Error, match="requires both schedule"):
    BatchedReferenceFreeContinuation(
      denoiser=_DiffusionNetwork(),
      decode=_Decoder(),
      projection=_projection(),
      device="cpu",
      generator=torch.Generator().manual_seed(0),
      rows=1,
    )
