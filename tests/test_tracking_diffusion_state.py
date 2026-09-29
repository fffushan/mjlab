import numpy as np
import pytest
from tracking_diffusion_fixtures import make_state

from mjlab.tasks.tracking.diffusion import WorldState, hybrid_to_world, world_to_hybrid


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
