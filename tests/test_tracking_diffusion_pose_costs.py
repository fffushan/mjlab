"""Focused D4a pose-cost tests: physical recovery, weights and fail-closed inputs.

These are CPU analytic checks against synthetic fixtures.  No simulator,
learned model, checkpoint, controller or hardware operation is involved.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT, DiffusionContract
from mjlab.tasks.tracking.diffusion.pose_costs import (
  PoseCost,
  PoseCostError,
  PoseCostWeights,
  TimedPoseGoal,
  active_pose_goals,
  world_poses_from_normalized_state,
)
from mjlab.tasks.tracking.diffusion.projection import (
  FeatureStats,
  ProjectionBundle,
  fit_feature_stats,
)
from mjlab.tasks.tracking.diffusion.state import WorldState, world_to_hybrid

_LEGACY_CONTRACT = DiffusionContract(
  schema_version="mjlab-x2-state-latent-d0-v1",
  projection_rows=96,
  projected_state_dimension=231,
  token_dimension=263,
)


def _stats(width: int, *, seed: int) -> FeatureStats:
  rng = np.random.default_rng(seed)
  return fit_feature_stats([rng.normal(size=(64, width)) + 3.0], std_floor=1e-6)


def _projection() -> ProjectionBundle:
  return ProjectionBundle.create(_stats(135, seed=1), _stats(32, seed=2))


def _quaternion(yaw: float, pitch: float = 0.0) -> np.ndarray:
  """Return the wxyz quaternion of a yaw-then-pitch root rotation."""
  half_yaw, half_pitch = yaw / 2.0, pitch / 2.0
  yaw_quaternion = np.array(
    [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)], dtype=np.float64
  )
  pitch_quaternion = np.array(
    [np.cos(half_pitch), 0.0, np.sin(half_pitch), 0.0], dtype=np.float64
  )
  w1, x1, y1, z1 = yaw_quaternion
  w2, x2, y2, z2 = pitch_quaternion
  return np.array(
    [
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ],
    dtype=np.float64,
  )


def _matrix(quaternion: np.ndarray) -> np.ndarray:
  """Return the rotation matrix of a wxyz quaternion."""
  w, x, y, z = quaternion / np.linalg.norm(quaternion)
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
  )


def _yaw_matrix(yaw: float) -> np.ndarray:
  cosine, sine = np.cos(yaw), np.sin(yaw)
  return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def _local_bodies() -> np.ndarray:
  return np.arange(60, dtype=np.float64).reshape(20, 3) * 0.01


def _state(
  position: tuple[float, float, float] = (0.0, 0.0, 0.0),
  yaw: float = 0.0,
  pitch: float = 0.0,
) -> WorldState:
  root = np.asarray(position, dtype=np.float64)
  bodies = root + _local_bodies() @ _yaw_matrix(yaw).T
  return WorldState(
    root,
    _quaternion(yaw, pitch),
    np.array([0.2, -0.1, 0.03]),
    np.array([0.0, 0.1, -0.2]),
    bodies,
    np.full((20, 3), 0.2),
  )


def _translated(state: WorldState, delta: tuple[float, float, float]) -> WorldState:
  shift = np.asarray(delta, dtype=np.float64)
  return replace(
    state,
    root_position=state.root_position + shift,
    body_positions=state.body_positions + shift,
  )


def _frame_shifted(
  state: WorldState, yaw_shift: float, translation: tuple[float, float, float]
) -> WorldState:
  """Return the same motion expressed after a global yaw rotation and translation."""
  rotation = _yaw_matrix(yaw_shift)
  shift = np.asarray(translation, dtype=np.float64)
  return WorldState(
    rotation @ state.root_position + shift,
    _quaternion_product(_quaternion(yaw_shift), state.root_quaternion_wxyz),
    rotation @ state.root_linear_velocity,
    rotation @ state.root_angular_velocity,
    state.body_positions @ rotation.T + shift,
    state.body_linear_velocities @ rotation.T,
  )


def _quaternion_product(first: np.ndarray, second: np.ndarray) -> np.ndarray:
  """Return the wxyz product of two quaternions."""
  w1, x1, y1, z1 = first
  w2, x2, y2, z2 = second
  return np.array(
    [
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ],
    dtype=np.float64,
  )


def _normalized_row(projection: ProjectionBundle, hybrid: np.ndarray) -> torch.Tensor:
  normalized = projection.project_state(np.asarray(hybrid, dtype=np.float64)[None])[0]
  return torch.as_tensor(normalized, dtype=torch.float64)


def _goal_tokens(
  projection: ProjectionBundle,
  current: WorldState,
  target: WorldState,
  *,
  row: int = 13,
) -> torch.Tensor:
  hybrid, _ = world_to_hybrid(WorldState.stack([current, target]), center_index=0)
  tokens = torch.zeros(1, 41, 167, dtype=torch.float64)
  tokens[0, row, :135] = _normalized_row(projection, hybrid[1])
  return tokens


def _goal(tick: int, state: WorldState, provenance: str = "fixture") -> TimedPoseGoal:
  return TimedPoseGoal(tick=tick, state=state, provenance=provenance)


def _cost(
  projection: ProjectionBundle,
  current: WorldState,
  goals: list[TimedPoseGoal],
  *,
  plan_tick: int = 100,
  weights: PoseCostWeights | None = None,
  contract: DiffusionContract = DEFAULT_CONTRACT,
) -> PoseCost:
  return PoseCost(
    projection=projection,
    current_state=current,
    plan_tick=plan_tick,
    goals=goals,
    weights=weights or PoseCostWeights(1.0, 1.0, 1.0),
    contract=contract,
  )


def test_world_poses_invert_world_to_hybrid_at_nonzero_center_frame() -> None:
  projection = _projection()
  current = _state((1.0, -2.0, 0.5), yaw=0.37, pitch=0.12)
  target = _state((2.5, 0.75, 0.4), yaw=-0.8, pitch=-0.2)
  tokens = _goal_tokens(projection, current, target)
  poses = world_poses_from_normalized_state(
    tokens[0, 13, :135],
    projection=projection,
    current_state=current,
  )
  assert np.allclose(poses.root_position.numpy(), target.root_position, atol=1e-12)
  assert np.allclose(poses.body_positions.numpy(), target.body_positions, atol=1e-12)
  assert np.allclose(
    poses.root_rotation.numpy(), _matrix(target.root_quaternion_wxyz), atol=1e-12
  )


def test_body_positions_use_the_predicted_row_yaw_not_the_center_yaw() -> None:
  projection = _projection()
  current = _state((0.3, 0.4, 0.9), yaw=0.25)
  predicted_relative_yaw = 0.9
  predicted_row_yaw = 0.25 + predicted_relative_yaw
  root_relative = np.array([0.4, -0.2, 0.1])
  local = _local_bodies()
  hybrid = np.zeros(135)
  hybrid[0:3] = root_relative
  hybrid[3:9] = _matrix(_quaternion(predicted_relative_yaw))[:, :2].T.reshape(-1)
  hybrid[15:75] = local.reshape(-1)
  poses = world_poses_from_normalized_state(
    _normalized_row(projection, hybrid),
    projection=projection,
    current_state=current,
  )
  center_rotation = _yaw_matrix(0.25)
  expected_root = current.root_position + root_relative @ center_rotation.T
  expected_bodies = expected_root + local @ _yaw_matrix(predicted_row_yaw).T
  assert np.allclose(poses.root_position.numpy(), expected_root, atol=1e-12)
  assert np.allclose(poses.body_positions.numpy(), expected_bodies, atol=1e-12)
  wrong_bodies = expected_root + local @ center_rotation.T
  assert not np.allclose(expected_bodies, wrong_bodies, atol=1e-6)


def test_pose_cost_is_zero_when_the_goal_row_holds_the_target_pose() -> None:
  projection = _projection()
  current = _state((1.0, -2.0, 0.5), yaw=0.37, pitch=0.12)
  target = _state((2.5, 0.75, 0.4), yaw=-0.8, pitch=-0.2)
  cost = _cost(projection, current, [_goal(105, target)])
  tokens = _goal_tokens(projection, current, target)
  tokens[0, 20, :] = 3.0
  tokens[0, 13, 135:] = -2.0
  value = cost(tokens)
  assert value.shape == (1,)
  assert abs(float(value[0])) < 1e-18


def test_pose_cost_scales_with_period_weights_and_squared_terms() -> None:
  projection = _projection()
  current = _state((1.0, -2.0, 0.5), yaw=0.37, pitch=0.12)
  target = _state((2.5, 0.75, 0.4), yaw=-0.8, pitch=-0.2)
  tokens = _goal_tokens(projection, current, target)

  position_only = PoseCostWeights(1.0, 0.0, 0.0)
  shifted = _translated(target, (0.0, 0.0, 1.0))
  value = _cost(projection, current, [_goal(105, shifted)], weights=position_only)
  assert float(value(tokens)[0]) == pytest.approx(0.02, rel=0.0, abs=1e-12)
  halved = _cost(
    projection,
    current,
    [_goal(105, shifted)],
    weights=PoseCostWeights(0.5, 0.0, 0.0),
  )
  assert float(halved(tokens)[0]) == pytest.approx(0.01, rel=0.0, abs=1e-12)

  two_goals = _cost(
    projection,
    current,
    [_goal(105, shifted), _goal(115, shifted)],
    weights=position_only,
  )
  tokens[0, 23, :135] = tokens[0, 13, :135]
  assert float(two_goals(tokens)[0]) == pytest.approx(0.04, rel=0.0, abs=1e-12)

  body_only = PoseCostWeights(0.0, 0.0, 1.0)
  bodies = target.body_positions.copy()
  bodies[7] += np.array([0.0, 0.0, 0.5])
  body_goal = replace(target, body_positions=bodies)
  body_value = _cost(projection, current, [_goal(105, body_goal)], weights=body_only)
  assert float(body_value(tokens)[0]) == pytest.approx(0.02 * 0.25, rel=0.0, abs=1e-12)


def test_orientation_cost_is_squared_chordal_and_sign_invariant() -> None:
  projection = _projection()
  current = _state((0.1, 0.2, 0.3), yaw=0.25)
  target = _state((0.4, -0.2, 0.35), yaw=0.25)
  tokens = _goal_tokens(projection, current, target)
  angle = 0.3
  rotated = np.zeros(135)
  rotated[0:3] = 0.0
  rotated[3:9] = _matrix(_quaternion(angle))[:, :2].T.reshape(-1)
  tokens[0, 13, :135] = _normalized_row(projection, rotated)
  orientation_only = PoseCostWeights(0.0, 1.0, 0.0)
  cost = _cost(projection, current, [_goal(105, target)], weights=orientation_only)
  expected = 0.02 * 4.0 * (1.0 - np.cos(angle))
  assert float(cost(tokens)[0]) == pytest.approx(expected, rel=1e-9, abs=1e-12)

  flipped = replace(target, root_quaternion_wxyz=-target.root_quaternion_wxyz)
  flipped_cost = _cost(
    projection, current, [_goal(105, flipped)], weights=orientation_only
  )
  assert float(flipped_cost(tokens)[0]) == pytest.approx(
    float(cost(tokens)[0]), rel=0.0, abs=1e-15
  )


def test_yaw_translation_frame_change_leaves_the_cost_unchanged() -> None:
  projection = _projection()
  current = _state((1.0, -2.0, 0.5), yaw=0.37, pitch=0.12)
  target = _state((2.5, 0.75, 0.4), yaw=-0.8, pitch=-0.2)
  tokens = _goal_tokens(projection, current, target)
  base = _cost(projection, current, [_goal(105, target)])

  shift, translation = 0.6, (3.0, -1.5, 0.25)
  moved_current = _frame_shifted(current, shift, translation)
  moved_target = _frame_shifted(target, shift, translation)
  moved = _cost(projection, moved_current, [_goal(105, moved_target)])
  assert float(moved(tokens)[0]) == pytest.approx(
    float(base(tokens)[0]), rel=0.0, abs=1e-15
  )
  poses = world_poses_from_normalized_state(
    tokens[0, 13, :135], projection=projection, current_state=moved_current
  )
  assert np.allclose(
    poses.body_positions.numpy(), moved_target.body_positions, atol=1e-10
  )


def test_active_goals_map_ticks_to_rows_and_skip_out_of_window() -> None:
  current = _state()
  goals = [
    _goal(80, _state((0.1, 0.0, 0.0))),
    _goal(100, _state((0.2, 0.0, 0.0))),
    _goal(110, _state((0.3, 0.0, 0.0))),
    _goal(132, _state((0.4, 0.0, 0.0))),
    _goal(142, _state((0.5, 0.0, 0.0))),
  ]
  active = active_pose_goals(100, goals)
  assert tuple((goal.tick, goal.offset, goal.row) for goal in active) == (
    (110, 10, 18),
    (132, 32, 40),
  )
  cost = _cost(_projection(), current, goals)
  assert cost.rows == (18, 40)
  assert cost.inactive_goal_ticks == (80, 100, 142)
  assert tuple(goal.tick for goal in cost.active_goals) == (110, 132)


def test_batch_elements_are_independent() -> None:
  projection = _projection()
  current = _state((1.0, -2.0, 0.5), yaw=0.37)
  target = _state((2.5, 0.75, 0.4), yaw=-0.8)
  cost = _cost(projection, current, [_goal(105, target)])
  first = _goal_tokens(projection, current, target)
  second = first.clone()
  second[0, 13, 0:3] += torch.tensor([0.5, -0.25, 0.75], dtype=torch.float64)
  batched = cost(torch.cat([first, second], dim=0))
  assert float(batched[0]) == pytest.approx(float(cost(first)[0]), rel=0.0, abs=1e-15)
  assert float(batched[1]) == pytest.approx(float(cost(second)[0]), rel=0.0, abs=1e-15)
  assert float(batched[1]) > float(batched[0])


def test_partial_body_coefficients_weight_only_listed_bodies() -> None:
  projection = _projection()
  current = _state((1.0, -2.0, 0.5), yaw=0.37)
  target = _state((2.5, 0.75, 0.4), yaw=-0.8)
  tokens = _goal_tokens(projection, current, target)
  bodies = target.body_positions.copy()
  bodies[2] += np.array([0.0, 0.5, 0.0])
  goal = _goal(105, replace(target, body_positions=bodies))
  body_only = PoseCostWeights(0.0, 0.0, 1.0)
  weighted = _cost(
    projection,
    current,
    [goal],
    weights=replace(body_only, body_coefficients={"left_knee_link": 2.0}),
  )
  assert float(weighted(tokens)[0]) == pytest.approx(0.02 * 2.0 * 0.25, abs=1e-12)
  other = _cost(
    projection,
    current,
    [goal],
    weights=replace(body_only, body_coefficients={"pelvis": 1.0}),
  )
  assert float(other(tokens)[0]) == pytest.approx(0.0, abs=1e-15)


def test_pose_cost_gradient_reaches_only_goal_position_and_orientation_channels() -> (
  None
):
  projection = _projection()
  current = _state((1.0, -2.0, 0.5), yaw=0.37)
  target = _state((2.5, 0.75, 0.4), yaw=-0.8)
  cost = _cost(projection, current, [_goal(105, target)])
  tokens = torch.randn(1, 41, 167, dtype=torch.float64, requires_grad=True)
  cost(tokens).sum().backward()
  gradient = tokens.grad
  assert gradient is not None
  assert float(gradient[0, 13, 0:9].abs().sum()) > 0.0
  assert float(gradient[0, 13, 15:75].abs().sum()) > 0.0
  assert float(gradient[0, 13, 9:15].abs().sum()) == 0.0
  assert float(gradient[0, 13, 75:135].abs().sum()) == 0.0
  assert float(gradient[0, 13, 135:].abs().sum()) == 0.0
  assert float(gradient[0, 14].abs().sum()) == 0.0


def test_timed_pose_goal_copies_its_arrays() -> None:
  target = _state((2.5, 0.75, 0.4), yaw=-0.8)
  goal = _goal(105, target)
  target.root_position[:] += 5.0
  target.body_positions[:] += 5.0
  assert np.allclose(goal.state.root_position, _state((2.5, 0.75, 0.4)).root_position)
  assert np.allclose(
    goal.state.body_positions,
    _state((2.5, 0.75, 0.4), yaw=-0.8).body_positions,
  )


@pytest.mark.parametrize("tick", [-1, True, 1.5, "5"])
def test_invalid_goal_ticks_fail_closed(tick: object) -> None:
  with pytest.raises(PoseCostError, match="tick"):
    TimedPoseGoal(tick=tick, state=_state(), provenance="fixture")  # type: ignore[arg-type]


def test_goal_requires_full_state_measurement_and_provenance() -> None:
  with pytest.raises(PoseCostError, match="WorldState"):
    TimedPoseGoal(tick=5, state=_state().root_position, provenance="fixture")  # type: ignore[arg-type]
  with pytest.raises(PoseCostError, match="provenance"):
    TimedPoseGoal(tick=5, state=_state(), provenance=" ")
  batched = WorldState.stack([_state(), _state()])
  with pytest.raises(PoseCostError, match="unbatched"):
    TimedPoseGoal(tick=5, state=batched, provenance="fixture")
  with pytest.raises(ValueError, match="non-finite"):
    replace(_state(), root_position=np.full(3, np.nan))


def test_duplicate_and_too_close_goals_fail_closed() -> None:
  goals = [_goal(105, _state()), _goal(105, _state((0.1, 0.0, 0.0)))]
  with pytest.raises(PoseCostError, match="duplicate"):
    active_pose_goals(100, goals)
  with pytest.raises(PoseCostError, match="minimum spacing"):
    active_pose_goals(100, [_goal(105, _state()), _goal(110, _state())])
  with pytest.raises(PoseCostError, match="iterable"):
    active_pose_goals(100, 5)  # type: ignore[arg-type]
  with pytest.raises(PoseCostError, match="TimedPoseGoal"):
    active_pose_goals(100, [_state()])  # type: ignore[list-item]


@pytest.mark.parametrize(
  "weights",
  [
    {"root_position": -1.0},
    {"root_position": float("nan")},
    {"root_position": float("inf")},
    {"root_orientation": True},
  ],
)
def test_invalid_weights_fail_closed(weights: dict[str, float | bool]) -> None:
  values: dict[str, float | bool] = {
    "root_position": 1.0,
    "root_orientation": 0.0,
    "body_positions": 0.0,
  }
  values.update(weights)
  with pytest.raises(PoseCostError, match="weight"):
    PoseCostWeights(**values)  # type: ignore[arg-type]


def test_all_zero_weights_and_bad_body_coefficients_fail_closed() -> None:
  with pytest.raises(PoseCostError, match="positive"):
    PoseCostWeights(0.0, 0.0, 0.0)
  with pytest.raises(PoseCostError, match="unknown bodies"):
    PoseCostWeights(0.0, 0.0, 1.0, body_coefficients={"not_a_body": 1.0})
  with pytest.raises(PoseCostError, match="20 values"):
    PoseCostWeights(0.0, 0.0, 1.0, body_coefficients=(1.0, 2.0))
  with pytest.raises(PoseCostError, match="weight"):
    PoseCostWeights(0.0, 0.0, 1.0, body_coefficients={"pelvis": -0.5})
  assert PoseCostWeights(0.0, 0.0, 1.0).coefficients == (1.0,) * 20
  assert (
    PoseCostWeights(0.0, 0.0, 1.0, body_coefficients={"pelvis": 2.0}).coefficients[0]
    == 2.0
  )


def test_cost_rejects_bad_token_windows() -> None:
  projection = _projection()
  current = _state()
  target = _state((0.5, 0.0, 0.0))
  cost = _cost(projection, current, [_goal(105, target)])
  with pytest.raises(PoseCostError, match="shape"):
    cost(torch.zeros(41, 167, dtype=torch.float64))
  with pytest.raises(PoseCostError, match="shape"):
    cost(torch.zeros(1, 41, 166, dtype=torch.float64))
  with pytest.raises(PoseCostError, match="floating-point"):
    cost(torch.zeros(1, 41, 167, dtype=torch.long))
  non_finite = torch.zeros(1, 41, 167, dtype=torch.float64)
  non_finite[0, 3, 3] = float("nan")
  with pytest.raises(PoseCostError, match="non-finite"):
    cost(non_finite)


def test_degenerate_predicted_rot6d_is_rejected() -> None:
  projection = _projection()
  current = _state((1.0, -2.0, 0.5), yaw=0.37)
  target = _state((2.5, 0.75, 0.4), yaw=-0.8)
  cost = _cost(projection, current, [_goal(105, target)])
  tokens = _goal_tokens(projection, current, target)
  tokens[0, 13, 3:9] = _normalized_row(projection, np.zeros(135))[3:9]
  with pytest.raises(PoseCostError, match="degenerate"):
    cost(tokens)


def test_cost_rejects_legacy_contracts_and_mismatched_projections() -> None:
  projection = _projection()
  current = _state()
  target = _state((0.5, 0.0, 0.0))
  with pytest.raises(PoseCostError, match="active no-mixture"):
    _cost(projection, current, [_goal(105, target)], contract=_LEGACY_CONTRACT)
  with pytest.raises(PoseCostError, match="identity differs"):
    _cost(
      projection,
      current,
      [_goal(105, target)],
      contract=replace(DEFAULT_CONTRACT, control_hz=100),
    )
  legacy_bundle = ProjectionBundle.create(
    _stats(135, seed=9), _stats(32, seed=10), contract=_LEGACY_CONTRACT
  )
  with pytest.raises(PoseCostError, match="identity differs"):
    PoseCost(
      projection=legacy_bundle,
      current_state=current,
      plan_tick=100,
      goals=[_goal(105, target)],
      weights=PoseCostWeights(1.0, 0.0, 0.0),
    )
  with pytest.raises(PoseCostError, match="PoseCostWeights"):
    PoseCost(
      projection=projection,
      current_state=current,
      plan_tick=100,
      goals=[_goal(105, target)],
      weights=(1.0, 0.0, 0.0),  # type: ignore[arg-type]
    )


def test_world_poses_reject_bad_normalized_states() -> None:
  projection = _projection()
  current = _state()
  with pytest.raises(PoseCostError, match="last dimension"):
    world_poses_from_normalized_state(
      torch.zeros(3, 134, dtype=torch.float64),
      projection=projection,
      current_state=current,
    )
  with pytest.raises(PoseCostError, match="floating-point"):
    world_poses_from_normalized_state(
      torch.zeros(135, dtype=torch.long),
      projection=projection,
      current_state=current,
    )
  with pytest.raises(PoseCostError, match="non-finite"):
    world_poses_from_normalized_state(
      torch.full((135,), float("nan"), dtype=torch.float64),
      projection=projection,
      current_state=current,
    )
