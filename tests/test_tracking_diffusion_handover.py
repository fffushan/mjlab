"""Focused D3 handover-seam tests: history, masks, and reference-free invariants.

These tests exercise the offline tensor seam only.  No simulator, controller,
checkpoint or hardware operation is involved; the denoiser and decoder are
small deterministic stubs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tracking_diffusion_fixtures import make_state

from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT
from mjlab.tasks.tracking.diffusion.handover import (
  ExecutedPair,
  HandoverError,
  HandoverHistory,
  PlanAgeError,
  ReferenceFreeContinuation,
  build_reference_free_plan,
)
from mjlab.tasks.tracking.diffusion.noising import apply_clean_mask
from mjlab.tasks.tracking.diffusion.projection import (
  FeatureStats,
  ProjectionBundle,
  fit_feature_stats,
)
from mjlab.tasks.tracking.diffusion.schedule import DiffusionSchedule, InferenceGrid
from mjlab.tasks.tracking.diffusion.state import WorldState, world_to_hybrid


def _stats(width: int, *, seed: int) -> FeatureStats:
  rng = np.random.default_rng(seed)
  return fit_feature_stats([rng.normal(size=(64, width)) + 3.0], std_floor=1e-6)


def _projection() -> ProjectionBundle:
  return ProjectionBundle.create(_stats(135, seed=1), _stats(32, seed=2))


def _history(*, length: int = 8) -> HandoverHistory:
  history = HandoverHistory()
  for index in range(length):
    history.push(make_state(index), np.full(32, 0.05 * index))
  return history


class _StubDenoiser:
  """Records calls and returns a scaled copy of its input."""

  def __init__(self) -> None:
    self.calls = 0
    self.inputs: list[torch.Tensor] = []
    self.step_ids: list[torch.Tensor] = []

  def __call__(self, sample: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
    self.calls += 1
    self.inputs.append(sample.detach().clone())
    self.step_ids.append(step_ids.detach().clone())
    return 0.1 * sample


class _RecordingDecoder:
  """Gravity-mode stand-in: its inputs contain no reference field at all."""

  def __init__(self) -> None:
    self.latents: list[np.ndarray] = []
    self.conditionings: list[np.ndarray] = []

  def __call__(self, latent: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
    assert latent.shape == (1, 32) and conditioning.shape == (1, 99)
    assert latent.device == conditioning.device
    assert latent.dtype == conditioning.dtype == torch.float32
    self.latents.append(latent[0].detach().cpu().numpy().astype(np.float64))
    self.conditionings.append(conditioning[0].detach().cpu().numpy().astype(np.float64))
    return latent[:, :31] * 0.5


def test_executed_pair_rejects_bad_latents_and_batched_states() -> None:
  with pytest.raises(HandoverError):
    ExecutedPair(make_state(0), np.zeros(31))
  with pytest.raises(HandoverError):
    ExecutedPair(make_state(0), np.array([np.inf] + [0.0] * 31))
  batched = WorldState(
    np.zeros((2, 3)),
    np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2, 1)),
    np.zeros((2, 3)),
    np.zeros((2, 3)),
    np.zeros((2, 20, 3)),
    np.zeros((2, 20, 3)),
  )
  with pytest.raises(HandoverError):
    ExecutedPair(batched, np.zeros(32))


def test_history_owns_snapshots_against_caller_buffer_mutation() -> None:
  """Reusing caller sensor buffers after push() must not rewrite recorded rows."""
  fields = (
    "root_position",
    "root_quaternion_wxyz",
    "root_linear_velocity",
    "root_angular_velocity",
    "body_positions",
    "body_linear_velocities",
  )
  state = make_state(0)
  latent = np.linspace(-1.0, 1.0, 32)
  expected_state = {name: getattr(state, name).copy() for name in fields}
  expected_latent = latent.copy()
  history = HandoverHistory()
  history.push(state, latent)
  recorded = history.pairs[0]

  # Mutate every caller-owned buffer in place, as a controller reusing its
  # sensor buffers between ticks would.
  for name in fields:
    getattr(state, name)[...] = 9.5
  latent[...] = -123.0

  for name in fields:
    stored = getattr(recorded.state, name)
    assert not np.shares_memory(stored, getattr(state, name))
    np.testing.assert_array_equal(stored, expected_state[name])
  assert not np.shares_memory(recorded.latent, latent)
  np.testing.assert_array_equal(recorded.latent, expected_latent)


def test_history_requires_eight_measured_pairs() -> None:
  history = _history(length=7)
  assert not history.ready
  with pytest.raises(HandoverError, match="measured state/executed-latent pairs"):
    build_reference_free_plan(history, make_state(7), projection=_projection(), tick=0)
  history.push(make_state(7), np.zeros(32))
  assert history.ready
  assert len(history.pairs) == DEFAULT_CONTRACT.past_steps


def test_plan_alignment_and_mask_semantics() -> None:
  history = _history()
  plan = build_reference_free_plan(
    history, make_state(8), projection=_projection(), tick=0
  )
  assert plan.clean_tokens.shape == (41, 167)
  assert plan.clean_mask.shape == (41, 167)
  assert plan.clean_mask.dtype == torch.bool
  mask = plan.clean_mask
  assert mask[:8].all()
  assert bool(mask[8, :135].all()) and not bool(mask[8, 135:].any())
  assert not mask[9:].any()
  # Unknown entries are zero placeholders in clean_tokens; the mask alone marks
  # them unknown, so the sampler never reads their values.
  assert torch.count_nonzero(plan.clean_tokens[9:]) == 0
  assert torch.count_nonzero(plan.clean_tokens[8, 135:]) == 0
  assert torch.count_nonzero(plan.clean_tokens[0, 135:]) != 0


def test_plan_rebases_history_into_current_frame() -> None:
  """History must be expressed in the new current window-center frame."""
  history = _history()
  current = make_state(8)
  plan = build_reference_free_plan(history, current, projection=_projection(), tick=0)
  states = list(pair.state for pair in history.pairs) + [current]
  hybrid, _ = world_to_hybrid(
    WorldState_stack(states), center_index=DEFAULT_CONTRACT.current_index
  )
  expected_state = _projection().project_state(hybrid)
  assert torch.allclose(
    plan.clean_tokens[:9, :135],
    torch.from_numpy(expected_state.astype(np.float32)),
    atol=1e-6,
  )


def WorldState_stack(states):  # helper shared by two tests
  from mjlab.tasks.tracking.diffusion.state import WorldState

  return WorldState.stack(list(states))


def test_sampler_conditions_round_trip_through_clean_mask() -> None:
  plan = build_reference_free_plan(
    _history(), make_state(8), projection=_projection(), tick=0
  )
  noise = torch.randn_like(plan.clean_tokens)
  sample = apply_clean_mask(noise, plan.clean_tokens, plan.clean_mask)
  assert torch.equal(sample[plan.clean_mask], plan.clean_tokens[plan.clean_mask])
  # Unknown entries keep the Gaussian draw; nothing else can leak in.
  assert torch.equal(sample[~plan.clean_mask], noise[~plan.clean_mask])


def test_continuation_rejects_handover_before_history() -> None:
  continuation = ReferenceFreeContinuation(
    denoiser=_StubDenoiser(),
    decode=_RecordingDecoder(),
    schedule=DiffusionSchedule.build(),
    grid=InferenceGrid.uniform(DiffusionSchedule.build()),
    projection=_projection(),
    generator=torch.Generator().manual_seed(0),
  )
  with pytest.raises(HandoverError, match="measured state/executed-latent pairs"):
    continuation.step(make_state(8), np.zeros(99), tick=0)


def test_continuation_step_mask_and_decode_semantics() -> None:
  denoiser = _StubDenoiser()
  decoder = _RecordingDecoder()
  schedule = DiffusionSchedule.build()
  grid = InferenceGrid.uniform(schedule)
  continuation = ReferenceFreeContinuation(
    denoiser=denoiser,
    decode=decoder,
    schedule=schedule,
    grid=grid,
    projection=_projection(),
    generator=torch.Generator().manual_seed(0),
  )
  for index in range(8):
    continuation.record_executed(make_state(index), np.full(32, 0.1 * index))
  conditioning = np.linspace(-0.5, 0.5, 99)
  step = continuation.step(make_state(8), conditioning, tick=8)
  assert step.denoiser_calls == 20
  assert step.final_index == 0
  assert step.action.shape == (31,)
  assert len(decoder.latents) == 1 and len(decoder.conditionings) == 1
  assert decoder.conditionings[0].shape == (99,)
  np.testing.assert_allclose(decoder.conditionings[0], conditioning, atol=0.0)
  # Past clean tokens were hard conditioning: their values are restored after
  # every jump, so the first denoiser input's fixed rows equal the measured
  # window bit-for-bit.
  first_input = denoiser.inputs[0][0]
  reference_plan = continuation.plan(make_state(8), tick=1)
  fixed = reference_plan.clean_mask
  assert torch.allclose(first_input[fixed], reference_plan.clean_tokens[fixed])
  assert step.executed_latent.shape == (32,)


def test_continuation_plan_age_rejects_stale_or_repeated_ticks() -> None:
  continuation = ReferenceFreeContinuation(
    denoiser=_StubDenoiser(),
    decode=_RecordingDecoder(),
    schedule=DiffusionSchedule.build(),
    grid=InferenceGrid.uniform(DiffusionSchedule.build()),
    projection=_projection(),
    generator=torch.Generator().manual_seed(0),
  )
  for index in range(8):
    continuation.record_executed(make_state(index), np.zeros(32))
  continuation.step(make_state(8), np.zeros(99), tick=3)
  with pytest.raises(PlanAgeError):
    continuation.step(make_state(9), np.zeros(99), tick=3)
  with pytest.raises(PlanAgeError):
    continuation.step(make_state(9), np.zeros(99), tick=2)
  # After reset, history is gone and the seam fails closed on handover again.
  continuation.reset()
  with pytest.raises(HandoverError, match="measured state/executed-latent pairs"):
    continuation.step(make_state(9), np.zeros(99), tick=4)


def test_reference_free_step_has_no_reference_or_encoder_path() -> None:
  """The step signature and decoder inputs admit no reference/teacher input."""
  import inspect

  parameters = inspect.signature(ReferenceFreeContinuation.step).parameters
  assert set(parameters) == {"self", "current_state", "conditioning", "tick"}
  decoder = _RecordingDecoder()
  continuation = ReferenceFreeContinuation(
    denoiser=_StubDenoiser(),
    decode=decoder,
    schedule=DiffusionSchedule.build(),
    grid=InferenceGrid.uniform(DiffusionSchedule.build()),
    projection=_projection(),
    generator=torch.Generator().manual_seed(0),
  )
  for index in range(8):
    continuation.record_executed(make_state(index), np.full(32, -0.2))
  step = continuation.step(make_state(8), np.linspace(-1.0, 1.0, 99), tick=0)
  assert step.action.shape == (31,)
  # The decoder saw only (32 latent | 99 conditioning): no 68-D reference row
  # was constructed or forwarded anywhere in the step.
  assert decoder.latents[0].shape == (32,)
  assert decoder.conditionings[0].shape == (99,)


def test_legacy_projected_contract_is_rejected() -> None:
  from dataclasses import replace as dataclass_replace

  legacy = dataclass_replace(
    DEFAULT_CONTRACT,
    schema_version="mjlab-x2-state-latent-d0-v1",
    projected_state_dimension=199,
    projection_rows=64,
    token_dimension=231,
  )
  projection = ProjectionBundle.create(
    _stats(135, seed=3), _stats(32, seed=4), contract=legacy
  )
  with pytest.raises(HandoverError, match="231-channel"):
    build_reference_free_plan(
      _history(), make_state(8), projection=projection, tick=0, contract=legacy
    )
  with pytest.raises(HandoverError, match="231-channel"):
    ReferenceFreeContinuation(
      denoiser=_StubDenoiser(),
      decode=_RecordingDecoder(),
      schedule=DiffusionSchedule.build(),
      grid=InferenceGrid.uniform(DiffusionSchedule.build()),
      projection=projection,
      contract=legacy,
      generator=torch.Generator().manual_seed(0),
    )


def test_projection_identity_mismatch_fails_closed() -> None:
  from dataclasses import replace as dataclass_replace

  other = dataclass_replace(DEFAULT_CONTRACT, std_floor=1e-5)
  projection = ProjectionBundle.create(
    _stats(135, seed=5), _stats(32, seed=6), contract=other
  )
  with pytest.raises(HandoverError, match="identity"):
    build_reference_free_plan(_history(), make_state(8), projection=projection, tick=0)
