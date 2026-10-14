"""CPU-only tests for the parallel D3 boundary and its batched continuation.

No test here builds MuJoCo physics.  The batched continuation is compared
against the frozen single-row continuation on identical inputs, which is the
numerical claim the parallel driver relies on.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from tracking_diffusion_fixtures import make_state

from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT
from mjlab.tasks.tracking.diffusion.handover import ReferenceFreeContinuation
from mjlab.tasks.tracking.diffusion.parallel_closed_loop_x2 import (
  BatchedReferenceFreeContinuation,
  ParallelObservation,
  ParallelRuntimeX2Error,
  _any_flag,
  _unbatched,
  tilt_degrees_batch,
)
from mjlab.tasks.tracking.diffusion.projection import (
  ProjectionBundle,
  fit_feature_stats,
)
from mjlab.tasks.tracking.diffusion.sampler import SamplerConfig
from mjlab.tasks.tracking.diffusion.schedule import DiffusionSchedule, InferenceGrid
from mjlab.tasks.tracking.diffusion.state import WorldState
from mjlab.tasks.tracking.diffusion.visual_playback import tilt_degrees


class _ElementwiseDenoiser:
  """Deterministic row-independent denoiser: batching cannot change a row."""

  def __call__(self, sample: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
    if sample.ndim != 3 or step_ids.shape[:2] != sample.shape[:2]:
      raise AssertionError("denoiser input shapes are wrong")
    scale = step_ids[..., :1].to(dtype=sample.dtype) / 1000.0
    return torch.tanh(sample * 0.5) * (1.0 - scale)


class _RowWiseDecoder:
  """Deterministic row-independent decoder returning ``[rows, 31]``."""

  def __call__(self, latent: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
    if latent.ndim != 2 or latent.shape[1] != 32:
      raise AssertionError("decoder latent width is wrong")
    if conditioning.shape != (latent.shape[0], 99):
      raise AssertionError("decoder conditioning width is wrong")
    return torch.tanh(latent[:, :31] * 0.25) + 0.01 * conditioning[:, :31]


def _projection() -> ProjectionBundle:
  rng = np.random.default_rng(7)
  state_stats = fit_feature_stats(
    [rng.normal(size=(64, DEFAULT_CONTRACT.state_dimension))], std_floor=1e-6
  )
  latent_stats = fit_feature_stats(
    [rng.normal(size=(64, DEFAULT_CONTRACT.latent_dimension))], std_floor=1e-6
  )
  return ProjectionBundle.create(state_stats, latent_stats)


def _sampler_config(grid: InferenceGrid) -> SamplerConfig:
  return SamplerConfig(
    grid,
    eta=0.0,
    guidance_strength=0.0,
    initial_noise_scale=1.0,
    start_level=1000,
  )


def _pair(rows: int, seed: int = 5) -> tuple[ReferenceFreeContinuation, Any]:
  schedule = DiffusionSchedule.build()
  grid = InferenceGrid.uniform(schedule)
  projection = _projection()
  single = ReferenceFreeContinuation(
    denoiser=_ElementwiseDenoiser(),
    decode=_RowWiseDecoder(),
    schedule=schedule,
    grid=grid,
    projection=projection,
    generator=torch.Generator(device="cpu").manual_seed(seed),
  )
  batched = BatchedReferenceFreeContinuation(
    denoiser=_ElementwiseDenoiser(),
    decode=_RowWiseDecoder(),
    schedule=schedule,
    projection=projection,
    contract=DEFAULT_CONTRACT,
    device="cpu",
    generator=torch.Generator(device="cpu").manual_seed(seed),
    sampler_config=_sampler_config(grid),
    rows=rows,
  )
  return single, batched


def _bootstrap(rows: int, seed: int = 5) -> tuple[ReferenceFreeContinuation, Any]:
  single, batched = _pair(rows, seed)
  for index in range(DEFAULT_CONTRACT.past_steps):
    state = make_state(index)
    latent = np.full(32, 0.05 * index, dtype=np.float64)
    single.record_executed(state, latent)
    batched.record_executed(
      np.arange(rows), WorldState.stack([state] * rows), np.tile(latent, (rows, 1))
    )
  return single, batched


def _batched_state(rows: int, index: int = 8) -> WorldState:
  return WorldState.stack([make_state(index)] * rows)


def test_batched_tilt_matches_the_single_row_diagnostic() -> None:
  states = [make_state(index) for index in range(6)]
  batched = tilt_degrees_batch(WorldState.stack(states))
  expected = np.array([tilt_degrees(state) for state in states])
  assert np.allclose(batched, expected, atol=1e-12)


def test_batched_tilt_reports_known_inclinations() -> None:
  """A pure roll inclination must read back as its own angle."""
  rows = []
  for degrees in (0.0, 30.0, 90.0, 150.0):
    half = np.radians(degrees) / 2.0
    quaternion = np.array([np.cos(half), np.sin(half), 0.0, 0.0])
    rows.append(
      WorldState(
        np.array([0.0, 0.0, 1.0]),
        quaternion,
        np.zeros(3),
        np.zeros(3),
        np.zeros((20, 3)),
        np.zeros((20, 3)),
      )
    )
  batched = tilt_degrees_batch(WorldState.stack(rows))
  assert np.allclose(batched, np.array([0.0, 30.0, 90.0, 150.0]), atol=1e-9)
  assert np.allclose(
    batched, np.array([tilt_degrees(row) for row in rows]), atol=1e-12
  )


def test_unbatched_row_equals_the_original_measurement() -> None:
  states = [make_state(index) for index in range(3)]
  batched = WorldState.stack(states)
  row = _unbatched(batched, 1)
  assert row.root_position.ndim == 1
  assert np.allclose(row.root_position, states[1].root_position)
  assert np.allclose(row.body_positions, states[1].body_positions)
  assert np.allclose(row.body_linear_velocities, states[1].body_linear_velocities)
  with pytest.raises(ParallelRuntimeX2Error, match="outside the batch"):
    _unbatched(batched, 3)


def test_batched_continuation_matches_the_single_row_step() -> None:
  single, batched = _bootstrap(rows=1)
  conditioning = np.linspace(-0.5, 0.5, 99)
  expected = single.step(make_state(8), conditioning, tick=8)
  measured = batched.step(
    _batched_state(rows=1), conditioning[None, :], tick=8, rows=[0]
  )
  assert np.allclose(measured.actions[0], expected.action, atol=1e-9)
  assert np.allclose(measured.executed_latents[0], expected.executed_latent, atol=1e-9)
  assert measured.denoiser_calls == expected.denoiser_calls
  assert measured.final_index == expected.final_index


def test_batched_rows_with_equal_inputs_still_differ_by_their_own_noise() -> None:
  _, batched = _bootstrap(rows=4)
  conditioning = np.tile(np.linspace(-0.5, 0.5, 99), (4, 1))
  measured = batched.step(_batched_state(rows=4), conditioning, tick=8, rows=[0, 1, 2, 3])
  assert measured.executed_latents.shape == (4, 32)
  assert measured.actions.shape == (4, 31)
  assert not np.allclose(
    measured.executed_latents[0], measured.executed_latents[1]
  ), "rows must not share one noise draw"


def test_batched_continuation_is_reproducible_for_one_seed() -> None:
  conditioning = np.tile(np.linspace(-0.5, 0.5, 99), (3, 1))
  first_single, first = _bootstrap(rows=3, seed=11)
  second_single, second = _bootstrap(rows=3, seed=11)
  one = first.step(_batched_state(rows=3), conditioning, tick=8, rows=[0, 1, 2])
  two = second.step(_batched_state(rows=3), conditioning, tick=8, rows=[0, 1, 2])
  assert np.array_equal(one.actions, two.actions)
  assert np.array_equal(one.executed_latents, two.executed_latents)
  assert first_single is not None and second_single is not None


def test_batched_continuation_fails_closed_on_incomplete_history() -> None:
  _, batched = _pair(rows=2)
  batched.record_executed(np.array([0, 1]), _batched_state(rows=2), np.zeros((2, 32)))
  conditioning = np.zeros((2, 99))
  with pytest.raises(ParallelRuntimeX2Error, match="measured state/latent pairs"):
    batched.step(_batched_state(rows=2), conditioning, tick=0, rows=[0, 1])


def test_batched_continuation_rejects_a_stale_tick_and_bad_rows() -> None:
  _, batched = _bootstrap(rows=2)
  conditioning = np.zeros((2, 99))
  batched.step(_batched_state(rows=2), conditioning, tick=8, rows=[0, 1])
  with pytest.raises(ParallelRuntimeX2Error, match="not newer"):
    batched.step(_batched_state(rows=2), conditioning, tick=8, rows=[0, 1])
  with pytest.raises(ParallelRuntimeX2Error, match="outside the batch"):
    batched.step(_batched_state(rows=2), conditioning, tick=9, rows=[7])
  with pytest.raises(ParallelRuntimeX2Error, match="must be unique"):
    batched.step(_batched_state(rows=2), conditioning, tick=9, rows=[0, 0])
  with pytest.raises(ParallelRuntimeX2Error, match="conditioning must have shape"):
    batched.step(_batched_state(rows=2), np.zeros((3, 99)), tick=9, rows=[0])


def test_batched_continuation_history_is_per_row() -> None:
  """Rows keep independent histories, and an unready row fails closed."""
  _, batched = _pair(rows=3)
  assert not batched.ready([0, 1, 2]).any()
  for index in range(DEFAULT_CONTRACT.past_steps):
    state = make_state(index)
    latent = np.full(32, 0.05 * index)
    batched.record_executed(
      np.array([0, 2]), WorldState.stack([state, state]), np.tile(latent, (2, 1))
    )
  assert batched.ready([0, 1, 2]).tolist() == [True, False, True]
  assert batched.history_sizes() == [DEFAULT_CONTRACT.past_steps, 0, DEFAULT_CONTRACT.past_steps]
  with pytest.raises(ParallelRuntimeX2Error, match="fewer than"):
    batched.step(_batched_state(rows=3), np.zeros((3, 99)), tick=0, rows=[0, 1])


def test_parallel_observation_requires_owned_shapes() -> None:
  state = _batched_state(rows=2)
  valid = ParallelObservation(
    state=state,
    conditioning=np.zeros((2, 99)),
    tick=0,
    sim_time_s=0.0,
    segment_id="member:seed=0:frame=45",
    reference=np.zeros((2, 68)),
    reference_frame=np.zeros(2, dtype=np.int64),
  )
  assert valid.rows == 2
  assert valid.tilt_degrees.shape == (2,)
  assert valid.reference is not None
  with pytest.raises(ParallelRuntimeX2Error, match="conditioning must be finite"):
    ParallelObservation(
      state=state,
      conditioning=np.zeros((1, 99)),
      tick=0,
      sim_time_s=0.0,
      segment_id="segment",
    )
  with pytest.raises(ParallelRuntimeX2Error, match="reference must be finite"):
    ParallelObservation(
      state=state,
      conditioning=np.zeros((2, 99)),
      tick=0,
      sim_time_s=0.0,
      segment_id="segment",
      reference=np.zeros((2, 67)),
    )


def test_any_flag_reads_torch_and_numpy_flags() -> None:
  assert _any_flag(torch.tensor([False, True]), name="flag")
  assert not _any_flag(np.zeros(3, dtype=bool), name="flag")
  assert _any_flag(torch.tensor(True), name="flag")
  with pytest.raises(ParallelRuntimeX2Error, match="is empty"):
    _any_flag(np.zeros(0, dtype=bool), name="flag")


def test_generator_prefix_parity_is_device_dependent() -> None:
  """The parity claim in the driver holds where the draw keeps its prefix.

  A one-row batch must reproduce a single-row draw.  Row zero of a wider draw
  keeps that prefix on CUDA, which is the device the driver defaults to; on CPU
  it does not, so a CPU parity run would disagree for that reason alone.
  """
  single = torch.randn((1, 3, 7), generator=torch.Generator().manual_seed(0))
  wider = torch.randn((4, 3, 7), generator=torch.Generator().manual_seed(0))
  cpu_prefix = bool(torch.equal(single[0], wider[0]))
  assert not cpu_prefix
  if torch.cuda.is_available():
    single_cuda = torch.randn(
      (1, 3, 7), generator=torch.Generator(device="cuda:0").manual_seed(0), device="cuda:0"
    )
    wider_cuda = torch.randn(
      (4, 3, 7), generator=torch.Generator(device="cuda:0").manual_seed(0), device="cuda:0"
    )
    assert torch.equal(single_cuda[0], wider_cuda[0])
