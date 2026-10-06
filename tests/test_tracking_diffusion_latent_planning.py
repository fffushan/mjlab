"""Focused CPU tests for the owned D5 latent-planning component."""

from __future__ import annotations

import pickle
from dataclasses import replace
from typing import Any, Callable, cast

import numpy as np
import pytest
import torch
from tracking_diffusion_fixtures import make_state

from mjlab.tasks.tracking.diffusion.conditioned_continuation import (
  SparseTaskSpec,
  TaskSource,
)
from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT
from mjlab.tasks.tracking.diffusion.inpainting import TimedStateKeyframe
from mjlab.tasks.tracking.diffusion.latent_planning import (
  LATENT_PLANNING_SCHEMA,
  RAW_LATENT_REPRESENTATION,
  LatentPlanner,
  LatentPlanningError,
  LatentPlanningPacket,
  PlanningPair,
  StateSnapshot,
  TaskSnapshot,
  build_planning_packet,
)
from mjlab.tasks.tracking.diffusion.pose_costs import PoseCostWeights, TimedPoseGoal
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
from mjlab.tasks.tracking.diffusion.scheduler import PlannerRequest
from mjlab.tasks.tracking.diffusion.state import WorldState


def _stats(width: int, *, seed: int) -> FeatureStats:
  rng = np.random.default_rng(seed)
  values = rng.normal(size=(64, width)) + 3.0
  return fit_feature_stats([values], std_floor=1e-6)


def _projection() -> ProjectionBundle:
  return ProjectionBundle.create(_stats(135, seed=1), _stats(32, seed=2))


def _history_values() -> list[tuple[WorldState, np.ndarray]]:
  return [(make_state(index), np.full(32, index * 0.05)) for index in range(8)]


def _packet(
  *,
  task: object | None = None,
  source_tick: int = 100,
  source_frame: int = 17,
  generator: torch.Generator | None = None,
) -> LatentPlanningPacket:
  random = generator or torch.Generator(device="cpu").manual_seed(9)
  return build_planning_packet(
    _history_values(),
    make_state(8),
    projection=_projection(),
    source_timestamp=2.0,
    source_tick=source_tick,
    source_frame=source_frame,
    task_inputs=task,
    rng=random,
    vae_identity="vae-test",
    model_identity="model-test",
  )


class _ScaledDenoiser:
  def __init__(self, scale: float = 0.2) -> None:
    self.scale = scale
    self.calls = 0
    self.inputs: list[torch.Tensor] = []

  def __call__(self, sample: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
    del step_ids
    self.calls += 1
    self.inputs.append(sample.detach().clone())
    return self.scale * sample


class _CoupledDenoiser(_ScaledDenoiser):
  def __call__(self, sample: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
    del step_ids
    self.calls += 1
    self.inputs.append(sample.detach().clone())
    state, latent = sample[..., :135], sample[..., 135:]
    state_summary = torch.tanh(state.mean(dim=(-2, -1), keepdim=True))
    latent_summary = torch.tanh(latent.mean(dim=(-2, -1), keepdim=True))
    return torch.cat(
      (0.3 * state + 0.1 * latent_summary, 0.3 * latent + 0.1 * state_summary),
      dim=-1,
    )


def _planner(
  denoiser: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
  *,
  projection: ProjectionBundle | None = None,
  **kwargs: Any,
) -> LatentPlanner:
  schedule = DiffusionSchedule.build()
  return LatentPlanner(
    denoiser=denoiser,
    schedule=schedule,
    grid=InferenceGrid.uniform(schedule),
    projection=projection or _projection(),
    vae_identity=kwargs.pop("vae_identity", "vae-test"),
    model_identity=kwargs.pop("model_identity", "model-test"),
    **kwargs,
  )


def _generator_from_packet(packet: LatentPlanningPacket) -> torch.Generator:
  generator = torch.Generator(device=packet.rng_device)
  generator.set_state(torch.as_tensor(packet.rng_state, dtype=torch.uint8))
  return generator


def _spec(
  mode: str, *, keyframes: tuple[TimedStateKeyframe, ...] = (), goals=(), strength=0.0
):
  source = TaskSource(
    kind="synthetic_fixture",
    qualification="unverified",
    provenance="latent-planning CPU fixture",
    verification="loader_structural_only",
  )
  return SparseTaskSpec(
    mode=mode,
    keyframes=keyframes,
    goals=goals,
    weights=None if not goals else PoseCostWeights(1.0, 1.0, 1.0),
    strength=strength,
    source=source,
    config_sha256="a" * 64,
  )


def test_packet_owns_eight_pairs_current_state_masks_targets_and_rng() -> None:
  values = _history_values()
  current = make_state(8)
  target_state = make_state(12)
  keyframe = TimedStateKeyframe(112, target_state, "fixture-keyframe")
  task = _spec("keyframe", keyframes=(keyframe,))
  generator = torch.Generator(device="cpu").manual_seed(31)
  before = generator.get_state().clone()
  packet = build_planning_packet(
    values,
    current,
    projection=_projection(),
    source_timestamp=2.0,
    source_tick=100,
    source_frame=17,
    task_inputs=task,
    rng=generator,
    vae_identity="vae-test",
    model_identity="model-test",
  )
  assert len(packet.pairs) == 8
  assert packet.ticks == tuple(range(100, 133))
  assert packet.task is not None
  assert packet.task.keyframe_ticks == (112,)
  assert packet.clean_mask[0] == (True,) * 167
  assert packet.clean_mask[8][:135] == (True,) * 135
  assert packet.clean_mask[8][135:] == (False,) * 32
  assert packet.clean_mask[20][:135] == (True,) * 135
  assert packet.clean_mask[20][135:] == (False,) * 32
  assert not any(any(packet.clean_mask[row]) for row in range(9, 20))
  assert torch.equal(generator.get_state(), before)
  assert packet.input_snapshot.flags.writeable is False

  values[0][0].root_position[0] = 999.0
  values[0][1][0] = 999.0
  current.root_position[0] = 999.0
  target_state.root_position[0] = 999.0
  assert packet.pairs[0].state.root_position[0] != 999.0
  assert packet.pairs[0].latent[0] != 999.0
  assert packet.current_state.root_position[0] != 999.0
  assert packet.task is not None
  assert packet.task.keyframes[0].state.root_position[0] != 999.0


def test_packet_wire_round_trip_has_no_live_objects_or_mutable_arrays() -> None:
  packet = _packet(task={"mode": "unguided", "revision": "task-r1"})
  wire = packet.as_wire()
  restored = LatentPlanningPacket.from_wire(wire)
  assert restored == packet
  encoded = pickle.dumps(wire)
  assert pickle.loads(encoded) == wire
  request = PlannerRequest(
    packet.input_snapshot,
    packet.source_timestamp,
    packet.source_tick,
    4,
    0,
    context=wire,
  )
  assert isinstance(request.context, dict | object)
  round_trip = LatentPlanningPacket.from_request(request)
  assert round_trip == packet
  with pytest.raises((TypeError, ValueError)):
    packet.input_snapshot[0, 0] = 1.0


def test_compact_context_round_trip_avoids_python_scalar_expansion() -> None:
  packet = _packet(task={"mode": "unguided", "revision": "task-r1"})
  readable = pickle.dumps(packet.as_wire(), protocol=pickle.HIGHEST_PROTOCOL)
  compact = packet.as_context()
  assert compact["encoding"] == "pickle-numpy-v1"
  assert isinstance(compact["payload"], bytes)
  assert len(pickle.dumps(compact, protocol=pickle.HIGHEST_PROTOCOL)) < len(readable)
  assert LatentPlanningPacket.from_wire(compact) == packet


def test_fixed_noise_output_matches_existing_synchronous_sampler_and_denormalizes() -> (
  None
):
  projection = _projection()
  packet = _packet()
  denoiser = _ScaledDenoiser()
  planner = _planner(denoiser, projection=projection)
  result = planner(packet)
  schedule = DiffusionSchedule.build()
  conditions = SamplerConditions(
    torch.as_tensor(packet.clean_tokens, dtype=torch.float32),
    torch.as_tensor(packet.clean_mask, dtype=torch.bool),
  )
  normalized, diagnostics = sample_trajectory(
    _ScaledDenoiser(),
    schedule=schedule,
    conditions=conditions,
    config=SamplerConfig(InferenceGrid.uniform(schedule)),
    generator=_generator_from_packet(packet),
  )
  expected = projection.denormalize_latent(
    normalized[8:, 135:].detach().to(dtype=torch.float32).numpy().astype(np.float64)
  )
  np.testing.assert_array_equal(result.raw_latents, expected)
  assert result.denoiser_calls == diagnostics.denoiser_calls == 20
  assert result.raw_latents.shape == (33, 32)
  assert result.representation == RAW_LATENT_REPRESENTATION
  assert result.metadata["latent_dimension"] == 32
  assert result.metadata["semantic"] == "raw"
  assert denoiser.calls == 20


def test_plan_rows_align_to_absolute_ticks_and_scheduler_metadata_is_latent_only() -> (
  None
):
  packet = _packet(source_tick=230)
  result = _planner(_ScaledDenoiser())(packet, plan_sequence_id=7)
  assert result.valid_from_tick == 230
  assert result.valid_until_tick == 262
  assert result.ticks == tuple(range(230, 263))
  np.testing.assert_array_equal(result.row_for_tick(230), result.raw_latents[0])
  np.testing.assert_array_equal(result.row_for_tick(262), result.raw_latents[-1])
  with pytest.raises(LatentPlanningError, match="outside"):
    result.row_for_tick(263)
  base = result.as_scheduler_plan()
  assert base.actions.shape == (33, 32)
  assert base.schema_metadata["representation"] == RAW_LATENT_REPRESENTATION
  assert base.schema_metadata["semantic"] == "raw"
  assert base.schema_metadata["latent_dimension"] == 32
  assert base.schema_metadata["latent_space"] == "raw"


def test_planner_does_not_call_decoder_encoder_or_mutate_caller_model_or_rng() -> None:
  packet = _packet()
  model = torch.nn.Linear(167, 167, bias=False)
  with torch.no_grad():
    model.weight.fill_(0.001)
  state_before = {key: value.clone() for key, value in model.state_dict().items()}
  rng_before = torch.Generator(device="cpu").manual_seed(9).get_state()

  def denoiser(sample: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
    del step_ids
    return model(sample)

  result = _planner(denoiser)(packet)
  assert result.raw_latents.shape == (33, 32)
  assert all(parameter.grad is None for parameter in model.parameters())
  for key, value in model.state_dict().items():
    assert torch.equal(value, state_before[key])
  caller_rng = torch.Generator(device="cpu").manual_seed(9)
  assert torch.equal(caller_rng.get_state(), rng_before)
  assert not hasattr(result, "action")


def test_incompatible_contract_projection_model_task_and_source_identities_reject() -> (
  None
):
  packet = _packet()
  planner = _planner(_ScaledDenoiser())
  with pytest.raises(LatentPlanningError, match="VAE identity"):
    _planner(_ScaledDenoiser(), vae_identity="other-vae")(packet)
  with pytest.raises(LatentPlanningError, match="model identity"):
    _planner(_ScaledDenoiser(), model_identity="other-model")(packet)
  with pytest.raises(LatentPlanningError, match="source frame"):
    _planner(_ScaledDenoiser(), expected_source_frame=99)(packet)
  with pytest.raises(LatentPlanningError, match="contract identity"):
    planner(replace(packet, contract_identity="bad-contract"))
  with pytest.raises(LatentPlanningError, match="projection identity"):
    planner(replace(packet, projection_identity="bad-projection"))
  task_packet = _packet(task={"mode": "unguided", "revision": "rev-a"})
  with pytest.raises(LatentPlanningError, match="task revision"):
    _planner(_ScaledDenoiser(), expected_task_revision="rev-b")(task_packet)


@pytest.mark.parametrize(
  ("kwargs", "message"),
  [
    ({"period_seconds": 0.01}, "period_seconds"),
    ({"source_tick": -1}, "source_tick"),
    ({"source_frame": -1}, "source_frame"),
    ({"root_frame": "body"}, "root_frame"),
  ],
)
def test_source_tick_time_period_and_root_frame_are_validated(
  kwargs: dict[str, object], message: str
) -> None:
  values = _history_values()
  with pytest.raises(LatentPlanningError, match=message):
    call_kwargs: dict[str, Any] = {
      "projection": _projection(),
      "source_timestamp": 2.0,
      "source_tick": 100,
      "source_frame": 17,
      "rng": torch.Generator(device="cpu").manual_seed(9),
    }
    call_kwargs.update(kwargs)
    build_planning_packet(values, make_state(8), **call_kwargs)


def test_hard_soft_and_combined_d4_paths_use_absolute_targets() -> None:
  keyframe = TimedStateKeyframe(112, make_state(12), "keyframe")
  goal = TimedPoseGoal(120, make_state(20), "goal")
  hard = _packet(task=_spec("keyframe", keyframes=(keyframe,)))
  hard_result = _planner(_ScaledDenoiser())(hard)
  assert hard_result.task_mode == "keyframe"
  assert hard.clean_mask[20][:135] == (True,) * 135

  soft = _packet(task=_spec("poseguided", goals=(goal,), strength=0.5))
  soft_result = _planner(_CoupledDenoiser())(soft)
  assert soft_result.task_mode == "poseguided"
  assert soft_result.denoiser_calls == 20

  combined = _packet(
    task=_spec("combined", keyframes=(keyframe,), goals=(goal,), strength=0.5)
  )
  combined_result = _planner(_CoupledDenoiser())(combined)
  assert combined_result.task_mode == "combined"
  assert combined_result.ticks == tuple(range(100, 133))


def test_task_snapshot_rejects_live_scoring_ledger() -> None:
  from mjlab.tasks.tracking.diffusion.conditioned_continuation import TaskConditioning

  with pytest.raises(LatentPlanningError, match="ledger"):
    TaskSnapshot.from_inputs(object.__new__(TaskConditioning))


def test_planner_rejects_non_default_sampler_configuration() -> None:
  schedule = DiffusionSchedule.build()
  with pytest.raises(LatentPlanningError, match="20-jump"):
    _planner(
      _ScaledDenoiser(),
      sampler_config=SamplerConfig(
        InferenceGrid.uniform(schedule), initial_noise_scale=0.5
      ),
    )


def test_planning_pair_and_state_snapshot_round_trip() -> None:
  pair = PlanningPair.from_pair(_history_values()[0])
  assert pair.to_executed_pair().state.root_position.shape == (3,)
  state = StateSnapshot.from_world_state(make_state(3))
  assert state == StateSnapshot.from_wire(state.as_wire())
  assert LATENT_PLANNING_SCHEMA in cast(str, _packet().as_wire()["schema"])


def test_base_plan_transport_keeps_raw_latents_distinct_from_robot_commands() -> None:
  packet = _packet()
  result = _planner(_ScaledDenoiser())(packet)
  base = result.as_scheduler_plan()
  assert base.actions.shape[1] == DEFAULT_CONTRACT.latent_dimension
  assert base.actions.shape[1] != DEFAULT_CONTRACT.action_dimension
  assert "command" not in cast(str, base.schema_metadata["representation"])
