import numpy as np
import pytest
from tracking_diffusion_fixtures import make_state

from mjlab.tasks.tracking.diffusion import (
  HYBRID_STATE,
  LOCAL_KINEMATIC_STATE,
  StateRepresentation,
  WorldFrameContext,
  WorldState,
  hybrid_to_local_kinematic,
  hybrid_to_world,
  local_kinematic_to_hybrid,
  local_kinematic_to_world,
  world_to_hybrid,
  world_to_local_kinematic,
)

_LOCAL_SLICES = {
  "root_rot6d": (0, 6),
  "root_angular_velocity": (6, 9),
  "body_positions": (9, 69),
  "body_linear_velocities": (69, 129),
}


def _hybrid_trajectory(
  length: int = 5, center_index: int = 2
) -> tuple[WorldState, np.ndarray, WorldFrameContext]:
  states = WorldState.stack(make_state(i) for i in range(length))
  hybrid, context = world_to_hybrid(states, center_index=center_index)
  return states, hybrid, context


def test_state_representations_name_the_frozen_channel_layouts() -> None:
  assert HYBRID_STATE.dimension == 135
  assert {
    name: (item.start, item.stop) for name, item in HYBRID_STATE.slices.items()
  } == {
    "root_position": (0, 3),
    "root_rot6d": (3, 9),
    "root_linear_velocity": (9, 12),
    "root_angular_velocity": (12, 15),
    "body_positions": (15, 75),
    "body_linear_velocities": (75, 135),
  }
  assert LOCAL_KINEMATIC_STATE.dimension == 129
  assert {
    name: (item.start, item.stop) for name, item in LOCAL_KINEMATIC_STATE.slices.items()
  } == _LOCAL_SLICES
  assert HYBRID_STATE.name != LOCAL_KINEMATIC_STATE.name


def test_state_representation_rejects_non_contiguous_slices() -> None:
  with pytest.raises(ValueError):
    StateRepresentation("gapped", 6, {"left": slice(0, 3), "right": slice(4, 7)})
  with pytest.raises(ValueError):
    StateRepresentation("short", 6, {"only": slice(0, 3)})


def test_local_kinematic_projection_drops_only_root_translation() -> None:
  _states, hybrid, _context = _hybrid_trajectory()
  local = hybrid_to_local_kinematic(hybrid)
  assert local.shape == (5, 129)
  for name, (start, stop) in _LOCAL_SLICES.items():
    hybrid_item = HYBRID_STATE.field(name)
    np.testing.assert_array_equal(local[:, start:stop], hybrid[:, hybrid_item])
  rebuilt = local_kinematic_to_hybrid(
    local,
    root_position=hybrid[:, HYBRID_STATE.field("root_position")],
    root_linear_velocity=hybrid[:, HYBRID_STATE.field("root_linear_velocity")],
  )
  np.testing.assert_array_equal(rebuilt, hybrid)


def test_local_kinematic_projection_rejects_wrong_width() -> None:
  with pytest.raises(ValueError):
    hybrid_to_local_kinematic(np.zeros((5, 134)))
  with pytest.raises(ValueError):
    local_kinematic_to_hybrid(
      np.zeros((5, 135)),
      root_position=np.zeros((5, 3)),
      root_linear_velocity=np.zeros((5, 3)),
    )
  with pytest.raises(ValueError):
    local_kinematic_to_hybrid(
      np.zeros((5, 129)),
      root_position=np.zeros((5, 2)),
      root_linear_velocity=np.zeros((5, 3)),
    )


def test_local_kinematic_frame_roundtrip_preserves_world_state() -> None:
  states, hybrid, hybrid_context = _hybrid_trajectory()
  local, context = world_to_local_kinematic(states, center_index=2)
  assert local.shape == (5, 129)
  np.testing.assert_array_equal(local, hybrid_to_local_kinematic(hybrid))
  np.testing.assert_array_equal(context.root_positions, states.root_position)
  np.testing.assert_array_equal(
    context.root_linear_velocities, states.root_linear_velocity
  )
  recovered = local_kinematic_to_world(local, context)
  reference = hybrid_to_world(hybrid, hybrid_context)
  for field in (
    "root_position",
    "root_quaternion_wxyz",
    "root_linear_velocity",
    "root_angular_velocity",
    "body_positions",
    "body_linear_velocities",
  ):
    np.testing.assert_allclose(
      getattr(recovered, field), getattr(states, field), atol=1e-10
    )
    np.testing.assert_allclose(getattr(recovered, field), getattr(reference, field))
  rebuilt, _context = world_to_local_kinematic(recovered, center_index=2)
  np.testing.assert_allclose(rebuilt, local, atol=1e-10)


def test_local_kinematic_inverse_requires_matching_root_context() -> None:
  states, _hybrid, context = _hybrid_trajectory()
  local, _context = world_to_local_kinematic(states, center_index=2)
  with pytest.raises(ValueError):
    local_kinematic_to_world(local[:-1], context)
  with pytest.raises(ValueError):
    local_kinematic_to_world(local.reshape(-1), context)


def test_frame_roundtrip_with_yaw_and_velocity_offset() -> None:
  states = WorldState.stack(make_state(i) for i in range(5))
  hybrid, context = world_to_hybrid(states, center_index=2)
  assert hybrid.shape == (5, 135)
  np.testing.assert_allclose(hybrid[2, 9:12], 0.0, atol=1e-12)
  recovered = hybrid_to_world(hybrid, context)
  np.testing.assert_allclose(recovered.root_position, states.root_position, atol=1e-10)
  np.testing.assert_allclose(
    recovered.root_linear_velocity, states.root_linear_velocity, atol=1e-10
  )
  rebuilt, _ = world_to_hybrid(recovered, center_index=2)
  np.testing.assert_allclose(rebuilt, hybrid, atol=1e-10)


def test_state_rejects_nonfinite_quaternion() -> None:
  state = make_state()
  bad = np.array(state.root_quaternion_wxyz)
  bad[0] = np.nan
  with pytest.raises(ValueError):
    WorldState(
      state.root_position,
      bad,
      state.root_linear_velocity,
      state.root_angular_velocity,
      state.body_positions,
      state.body_linear_velocities,
    )


def test_batched_zero_quaternion_is_rejected() -> None:
  state = make_state()
  with pytest.raises(ValueError):
    WorldState(
      np.stack([state.root_position, state.root_position]),
      np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
      np.stack([state.root_linear_velocity, state.root_linear_velocity]),
      np.stack([state.root_angular_velocity, state.root_angular_velocity]),
      np.stack([state.body_positions, state.body_positions]),
      np.stack([state.body_linear_velocities, state.body_linear_velocities]),
    )
