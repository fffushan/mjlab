"""Focused D4a hard-keyframe inpainting tests: absolute ticks, rebasing, masks.

These are CPU-only tensor tests of the offline D4a seam.  No simulator, learned
model, decoder or hardware operation is involved; the denoiser is a small
deterministic stub.  A keyframe is rebased with the frozen ``world_to_hybrid``
and normalized with the same frozen statistics as the D3 plan, so these tests
prove framing and mask mechanics, not tracking quality.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch
from tracking_diffusion_fixtures import make_state

from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT
from mjlab.tasks.tracking.diffusion.handover import (
  ContinuationPlan,
  HandoverHistory,
  build_reference_free_plan,
)
from mjlab.tasks.tracking.diffusion.inpainting import (
  InpaintingError,
  TimedStateKeyframe,
  active_keyframe_targets,
  inpaint_keyframes,
)
from mjlab.tasks.tracking.diffusion.noising import clean_mask as d3_clean_mask
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
from mjlab.tasks.tracking.diffusion.state import (
  StateValidationError,
  WorldState,
  hybrid_to_world,
  world_to_hybrid,
)

_STATE_ARRAY_FIELDS = (
  "root_position",
  "root_quaternion_wxyz",
  "root_linear_velocity",
  "root_angular_velocity",
  "body_positions",
  "body_linear_velocities",
)


def _stats(width: int, *, seed: int) -> FeatureStats:
  rng = np.random.default_rng(seed)
  return fit_feature_stats([rng.normal(size=(64, width)) + 3.0], std_floor=1e-6)


def _projection(contract=DEFAULT_CONTRACT) -> ProjectionBundle:
  return ProjectionBundle.create(
    _stats(135, seed=1), _stats(32, seed=2), contract=contract
  )


def _history(*, length: int = 8) -> HandoverHistory:
  history = HandoverHistory()
  for index in range(length):
    history.push(make_state(index), np.full(32, 0.05 * index))
  return history


def _plan(
  *,
  tick: int,
  current: WorldState | None = None,
  projection: ProjectionBundle | None = None,
) -> ContinuationPlan:
  return build_reference_free_plan(
    _history(),
    make_state(8) if current is None else current,
    projection=_projection() if projection is None else projection,
    tick=tick,
  )


def _keyframe(
  tick: int,
  *,
  position=(1.0, 0.5, 0.25),
  yaw=0.3,
  velocity=(0.3, -0.1, 0.05),
  provenance: str = "fixture:asymmetric-target",
) -> TimedStateKeyframe:
  return TimedStateKeyframe(
    tick=tick,
    state=WorldState(
      np.asarray(position, dtype=np.float64),
      np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]),
      np.asarray(velocity, dtype=np.float64),
      np.array([0.04, -0.11, 0.23]),
      np.linspace(0.1, 1.0, 60, dtype=np.float64).reshape(20, 3),
      np.full((20, 3), -0.05),
    ),
    provenance=provenance,
  )


def _shifted_current() -> WorldState:
  """A second current state with a different position, yaw and velocity."""
  yaw = -0.65
  return WorldState(
    np.array([-3.0, 4.0, 0.2]),
    np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]),
    np.array([-0.4, 0.3, -0.05]),
    np.array([0.2, 0.3, -0.1]),
    np.arange(60, dtype=np.float64).reshape(20, 3) * 0.02,
    np.full((20, 3), 0.15),
  )


def _batched_state() -> WorldState:
  return WorldState(
    np.zeros((2, 3)),
    np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2, 1)),
    np.zeros((2, 3)),
    np.zeros((2, 3)),
    np.zeros((2, 20, 3)),
    np.zeros((2, 20, 3)),
  )


def test_keyframe_owns_arrays_and_rejects_invalid_ticks_and_provenance() -> None:
  state = make_state(4)
  numpy_tick: Any = np.int64(7)
  keyframe = TimedStateKeyframe(tick=numpy_tick, state=state, provenance="fixture")
  assert keyframe.tick == 7 and isinstance(keyframe.tick, int)
  expected = {name: getattr(state, name).copy() for name in _STATE_ARRAY_FIELDS}
  for name in _STATE_ARRAY_FIELDS:
    getattr(state, name)[...] = 9.5
  for name in _STATE_ARRAY_FIELDS:
    stored = getattr(keyframe.state, name)
    assert not np.shares_memory(stored, getattr(state, name))
    np.testing.assert_array_equal(stored, expected[name])

  bad_ticks: tuple[Any, ...] = (True, 3.0, "3", None, np.float64(2.0))
  for bad_tick in bad_ticks:
    with pytest.raises(InpaintingError, match="integer control tick"):
      TimedStateKeyframe(tick=bad_tick, state=make_state(0), provenance="fixture")
  with pytest.raises(InpaintingError, match="non-negative"):
    TimedStateKeyframe(tick=-1, state=make_state(0), provenance="fixture")
  with pytest.raises(InpaintingError, match="provenance"):
    TimedStateKeyframe(tick=1, state=make_state(0), provenance="  ")
  not_a_state: Any = ("not a state",)
  with pytest.raises(InpaintingError, match="WorldState"):
    TimedStateKeyframe(tick=1, state=not_a_state, provenance="fixture")
  with pytest.raises(InpaintingError, match="unbatched"):
    TimedStateKeyframe(tick=1, state=_batched_state(), provenance="fixture")
  # Non-finite targets are rejected by the frozen WorldState boundary itself.
  with pytest.raises(StateValidationError):
    _keyframe(10, position=(np.nan, 0.0, 0.0))


def test_keyframe_serialization_round_trip_and_identity_guards() -> None:
  keyframe = _keyframe(110, position=(1.0, 0.2, 0.3), yaw=0.4)
  payload = keyframe.as_dict()
  restored = TimedStateKeyframe.from_dict(payload)
  assert restored.tick == keyframe.tick
  assert restored.provenance == keyframe.provenance
  for name in _STATE_ARRAY_FIELDS:
    np.testing.assert_array_equal(
      getattr(restored.state, name), getattr(keyframe.state, name)
    )
    assert not np.shares_memory(
      getattr(restored.state, name), getattr(keyframe.state, name)
    )

  with pytest.raises(InpaintingError, match="unknown fields"):
    TimedStateKeyframe.from_dict({**payload, "note": "extra"})
  with pytest.raises(InpaintingError, match="missing fields"):
    TimedStateKeyframe.from_dict(
      {key: value for key, value in payload.items() if key != "provenance"}
    )
  other = replace(DEFAULT_CONTRACT, std_floor=1e-5)
  with pytest.raises(InpaintingError, match="identity"):
    TimedStateKeyframe.from_dict(payload, contract=other)
  reordered = {
    **payload,
    "body_names": list(reversed(DEFAULT_CONTRACT.body_names)),
  }
  with pytest.raises(InpaintingError, match="body order"):
    TimedStateKeyframe.from_dict(reordered)
  malformed = {
    **payload,
    "state": {**payload["state"], "root_position": [np.nan, 0.0, 0.0]},
  }
  with pytest.raises(InpaintingError, match="malformed keyframe state"):
    TimedStateKeyframe.from_dict(malformed)
  legacy = replace(
    DEFAULT_CONTRACT,
    schema_version="mjlab-x2-state-latent-d0-v1",
    projected_state_dimension=199,
    projection_rows=64,
    token_dimension=231,
  )
  with pytest.raises(InpaintingError, match="231-channel"):
    TimedStateKeyframe.from_dict(payload, contract=legacy)
  with pytest.raises(InpaintingError, match="231-channel"):
    keyframe.as_dict(contract=legacy)


def test_inpaint_rebases_target_into_current_measured_frame() -> None:
  projection = _projection()
  current = make_state(8)
  keyframe = _keyframe(112, position=(2.5, 1.25, -0.4), yaw=0.9)
  plan = _plan(tick=100, current=current, projection=projection)
  result = inpaint_keyframes(plan, current, [keyframe], projection=projection)
  row = DEFAULT_CONTRACT.current_index + 12

  hybrid, _context = world_to_hybrid(
    WorldState.stack([current, keyframe.state]), center_index=0
  )
  expected = projection.project_state(hybrid[1:2])[0].astype(np.float32)
  assert torch.equal(result.clean_tokens[row, :135], torch.from_numpy(expected))
  assert bool(result.clean_mask[row, :135].all())
  # Only the 135 state entries are fixed; the 32 latent entries stay unknown.
  assert not bool(result.clean_mask[row, 135:].any())
  assert torch.equal(result.clean_tokens[row, 135:], plan.clean_tokens[row, 135:])
  # Past/current rows and the unknown placeholders are preserved bit-for-bit.
  assert torch.equal(result.clean_tokens[:9], plan.clean_tokens[:9])
  assert torch.equal(result.clean_mask[:9], plan.clean_mask[:9])
  others = [index for index in range(9, DEFAULT_CONTRACT.window_steps) if index != row]
  assert torch.equal(result.clean_tokens[others], plan.clean_tokens[others])
  assert torch.equal(result.clean_mask[others], plan.clean_mask[others])
  # The target is rebased against the current root frame, not centered on
  # itself: the current row is at the origin and the target row is not.
  assert np.allclose(hybrid[0][:3], 0.0)
  assert np.linalg.norm(hybrid[1][:3]) > 0.5
  # The returned plan owns independent tensors.
  assert result.clean_tokens.data_ptr() != plan.clean_tokens.data_ptr()
  assert result.clean_mask.data_ptr() != plan.clean_mask.data_ptr()
  assert torch.equal(plan.clean_mask[9:], torch.zeros_like(plan.clean_mask[9:]))


def test_absolute_world_goal_survives_current_frame_change() -> None:
  projection = _projection()
  keyframe = _keyframe(
    112, position=(3.0, -1.5, 0.6), yaw=1.2, velocity=(0.2, 0.4, 0.0)
  )
  rows: list[np.ndarray] = []
  for current in (make_state(8), _shifted_current()):
    plan = _plan(tick=100, current=current, projection=projection)
    result = inpaint_keyframes(plan, current, [keyframe], projection=projection)
    row = DEFAULT_CONTRACT.current_index + 12
    row_values = result.clean_tokens[row, :135].numpy().astype(np.float64)[None, :]
    rows.append(row_values[0].copy())
    hybrid, context = world_to_hybrid(
      WorldState.stack([current, keyframe.state]), center_index=0
    )
    np.testing.assert_allclose(
      projection.inverse_state(row_values), hybrid[1:2], atol=1e-5
    )
    recovered = hybrid_to_world(hybrid, context)
    np.testing.assert_allclose(
      recovered.root_position[1], keyframe.state.root_position, atol=1e-4
    )
    np.testing.assert_allclose(
      recovered.root_linear_velocity[1], keyframe.state.root_linear_velocity, atol=1e-4
    )
    np.testing.assert_allclose(
      recovered.body_positions[1], keyframe.state.body_positions, atol=1e-4
    )
    sign = float(
      np.dot(recovered.root_quaternion_wxyz[1], keyframe.state.root_quaternion_wxyz)
    )
    assert abs(abs(sign) - 1.0) < 1e-6
  # The same absolute world goal needs different normalized rows in the two
  # current frames: it is not a fixed target-frame representation.
  assert not np.allclose(rows[0], rows[1])


def test_empty_keyframes_is_an_independent_bitwise_copy() -> None:
  projection = _projection()
  current = make_state(8)
  plan = _plan(tick=100, current=current, projection=projection)
  result = inpaint_keyframes(plan, current, [], projection=projection)
  assert torch.equal(result.clean_tokens, plan.clean_tokens)
  assert torch.equal(result.clean_mask, plan.clean_mask)
  assert result.clean_tokens.data_ptr() != plan.clean_tokens.data_ptr()
  assert result.clean_mask.data_ptr() != plan.clean_mask.data_ptr()
  assert result.tick == plan.tick
  assert active_keyframe_targets(plan, []) == ()


def test_active_targets_bounds_rolling_window_and_expiry() -> None:
  projection = _projection()
  current = make_state(8)
  plan = _plan(tick=100, current=current, projection=projection)
  expired = _keyframe(90)
  first = _keyframe(101, position=(1.2, 0.0, 0.0))
  last = _keyframe(132, position=(1.4, 0.0, 0.0))
  beyond = _keyframe(150, position=(1.6, 0.0, 0.0))
  keyframes = [expired, first, last, beyond]

  active = active_keyframe_targets(plan, keyframes)
  assert [(item.tick, item.offset, item.row) for item in active] == [
    (101, 1, 9),
    (132, 32, 40),
  ]
  result = inpaint_keyframes(plan, current, keyframes, projection=projection)
  assert bool(result.clean_mask[9, :135].all())
  assert bool(result.clean_mask[40, :135].all())
  # Expired and beyond-horizon targets stay untouched; they are not replaced
  # by, and do not replace, the measured current/past rows.
  assert not bool(result.clean_mask[18].any())
  assert not bool(result.clean_mask[26].any())
  assert torch.equal(result.clean_mask[:9], plan.clean_mask[:9])

  # Rolling replanning keeps the absolute tick and only moves the row.
  later_tick = 132
  later_current = make_state(9)
  later_plan = _plan(tick=later_tick, current=later_current, projection=projection)
  assert [
    (item.tick, item.offset, item.row)
    for item in active_keyframe_targets(later_plan, [last, beyond])
  ] == [(150, 18, 26)]
  later_result = inpaint_keyframes(
    later_plan, later_current, [last, beyond], projection=projection
  )
  assert not bool(later_result.clean_mask[40, :135].any())
  assert bool(later_result.clean_mask[26, :135].all())


def test_duplicate_and_dense_targets_are_rejected() -> None:
  projection = _projection()
  current = make_state(8)
  plan = _plan(tick=100, current=current, projection=projection)
  duplicate = [_keyframe(110), _keyframe(110, position=(2.0, 0.0, 0.0))]
  with pytest.raises(InpaintingError, match="duplicate"):
    inpaint_keyframes(plan, current, duplicate, projection=projection)
  with pytest.raises(InpaintingError, match="duplicate"):
    active_keyframe_targets(plan, duplicate)
  dense = [_keyframe(101), _keyframe(109, position=(2.0, 0.0, 0.0))]
  with pytest.raises(InpaintingError, match="minimum spacing"):
    inpaint_keyframes(plan, current, dense, projection=projection)
  with pytest.raises(InpaintingError, match="minimum spacing"):
    active_keyframe_targets(plan, dense)


def test_conflicting_fixed_entries_are_rejected_and_repeat_is_noop() -> None:
  projection = _projection()
  current = make_state(8)
  plan = _plan(tick=100, current=current, projection=projection)
  keyframe = _keyframe(110, position=(1.5, 0.25, 0.0), yaw=0.3)
  first = inpaint_keyframes(plan, current, [keyframe], projection=projection)
  repeat = inpaint_keyframes(first, current, [keyframe], projection=projection)
  assert torch.equal(repeat.clean_tokens, first.clean_tokens)
  assert torch.equal(repeat.clean_mask, first.clean_mask)

  row = DEFAULT_CONTRACT.current_index + 10
  conflicting = _keyframe(110, position=(1.6, 0.25, 0.0), yaw=0.3)
  with pytest.raises(InpaintingError, match="conflict"):
    inpaint_keyframes(first, current, [conflicting], projection=projection)
  # The conflicting request never overwrote the existing fixed values.
  reference = inpaint_keyframes(plan, current, [keyframe], projection=projection)
  assert torch.equal(first.clean_tokens[row, :135], reference.clean_tokens[row, :135])
  assert torch.equal(first.clean_mask, reference.clean_mask)


class _RecordingDenoiser:
  """Deterministic stub recording every reverse call's step IDs and input."""

  def __init__(self) -> None:
    self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

  def __call__(self, sample: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
    self.calls.append((step_ids.detach().clone(), sample.detach().clone()))
    return 0.1 * sample


def test_sampler_keeps_future_state_clean_and_latent_noisy_every_call() -> None:
  projection = _projection()
  current = make_state(8)
  plan = _plan(tick=100, current=current, projection=projection)
  keyframe = _keyframe(112, position=(2.0, 0.5, 0.1), yaw=0.7)
  result = inpaint_keyframes(plan, current, [keyframe], projection=projection)
  row = DEFAULT_CONTRACT.current_index + 12

  schedule = DiffusionSchedule.build()
  grid = InferenceGrid.uniform(schedule)
  denoiser = _RecordingDenoiser()
  sample, diagnostics = sample_trajectory(
    denoiser,
    schedule=schedule,
    conditions=SamplerConditions(result.clean_tokens, result.clean_mask),
    config=SamplerConfig(grid),
    generator=torch.Generator().manual_seed(0),
  )

  assert diagnostics.denoiser_calls == 20
  assert diagnostics.final_index == 0
  assert len(denoiser.calls) == 20
  assert [int(ids[0, row, 1]) for ids, _ in denoiser.calls] == list(grid.source_ids)
  for ids, call_sample in denoiser.calls:
    # The keyframe row is clean state (id 0) with an unknown noisy latent.
    assert ids.shape == (1, DEFAULT_CONTRACT.window_steps, 2)
    assert call_sample.shape == (1, DEFAULT_CONTRACT.window_steps, 167)
    assert int(ids[0, row, 0]) == 0
    assert int(ids[0, row, 1]) > 0
    assert torch.equal(call_sample[0, row, :135], result.clean_tokens[row, :135])
    # The keyframe leaves the latent unknown: the row keeps its noisy draw
    # instead of the zero placeholder, and the sampler still generates it.
    assert float(torch.linalg.vector_norm(call_sample[0, row, 135:])) > 0.0
    assert not torch.equal(call_sample[0, row, 135:], result.clean_tokens[row, 135:])
    # Measured past rows and the current state slice stay fixed at id 0.
    assert torch.equal(ids[0, :8, 0], torch.zeros(8, dtype=torch.long))
    assert torch.equal(ids[0, :8, 1], torch.zeros(8, dtype=torch.long))
    assert int(ids[0, 8, 0]) == 0
    assert int(ids[0, 8, 1]) == int(ids[0, row, 1])
  # Bit-identical fixed values survive all 20 jumps; the measured past rows are
  # fully fixed, the current row only in its state slice (the current latent is
  # still generated, exactly as in D3), and the keyframe row in its state slice.
  assert torch.equal(sample[row, :135], result.clean_tokens[row, :135])
  assert torch.equal(sample[:8], result.clean_tokens[:8])
  assert torch.equal(sample[8, :135], result.clean_tokens[8, :135])
  assert not torch.equal(sample[8, 135:], result.clean_tokens[8, 135:])
  # The un-inpainted D3 plan leaves exactly this row unknown (no keyframe).
  assert not bool(plan.clean_mask[row].any())
  assert torch.equal(
    plan.clean_mask[9:],
    d3_clean_mask(
      current_index=DEFAULT_CONTRACT.current_index,
      state_dimension=DEFAULT_CONTRACT.projected_state_dimension,
      token_dimension=DEFAULT_CONTRACT.token_dimension,
      device=plan.clean_mask.device,
    )[9:],
  )


def test_inpaint_rejects_legacy_contract_and_mismatched_projection() -> None:
  current = make_state(8)
  keyframe = _keyframe(110)
  plan = _plan(tick=100, current=current)
  legacy = replace(
    DEFAULT_CONTRACT,
    schema_version="mjlab-x2-state-latent-d0-v1",
    projected_state_dimension=199,
    projection_rows=64,
    token_dimension=231,
  )
  legacy_projection = ProjectionBundle.create(
    _stats(135, seed=3), _stats(32, seed=4), contract=legacy
  )
  with pytest.raises(InpaintingError, match="231-channel"):
    inpaint_keyframes(
      plan, current, [keyframe], projection=legacy_projection, contract=legacy
    )
  other = replace(DEFAULT_CONTRACT, std_floor=1e-5)
  other_projection = ProjectionBundle.create(
    _stats(135, seed=5), _stats(32, seed=6), contract=other
  )
  with pytest.raises(InpaintingError, match="identity"):
    inpaint_keyframes(plan, current, [keyframe], projection=other_projection)


def test_inpaint_rejects_malformed_plans_states_and_keyframes() -> None:
  projection = _projection()
  current = make_state(8)
  keyframe = _keyframe(110)
  plan = _plan(tick=100, current=current, projection=projection)

  not_a_plan: Any = object()
  with pytest.raises(InpaintingError, match="ContinuationPlan"):
    inpaint_keyframes(not_a_plan, current, [keyframe], projection=projection)
  with pytest.raises(InpaintingError, match="shape"):
    inpaint_keyframes(
      ContinuationPlan(
        tick=100,
        clean_tokens=torch.zeros(40, 167),
        clean_mask=torch.zeros(40, 167, dtype=torch.bool),
      ),
      current,
      [keyframe],
      projection=projection,
    )
  with pytest.raises(InpaintingError, match="past history row"):
    inpaint_keyframes(
      ContinuationPlan(
        tick=100,
        clean_tokens=torch.zeros(41, 167),
        clean_mask=torch.zeros(41, 167, dtype=torch.bool),
      ),
      current,
      [keyframe],
      projection=projection,
    )
  broken_current = plan.clean_mask.clone()
  broken_current[DEFAULT_CONTRACT.current_index, 135:] = True
  with pytest.raises(InpaintingError, match="current physical-state slice"):
    inpaint_keyframes(
      ContinuationPlan(plan.tick, plan.clean_tokens, broken_current),
      current,
      [keyframe],
      projection=projection,
    )
  with pytest.raises(InpaintingError, match="unbatched"):
    inpaint_keyframes(plan, _batched_state(), [keyframe], projection=projection)
  bad_items: Any = [object()]
  with pytest.raises(InpaintingError, match="TimedStateKeyframe"):
    inpaint_keyframes(plan, current, bad_items, projection=projection)
  mapping_items: Any = {"110": keyframe}
  with pytest.raises(InpaintingError, match="TimedStateKeyframe"):
    inpaint_keyframes(plan, current, mapping_items, projection=projection)
  string_items: Any = "keyframes"
  with pytest.raises(InpaintingError, match="TimedStateKeyframe"):
    active_keyframe_targets(plan, string_items)
