"""CPU-only tests for the D3 display-only future-pose recovery and IK helper.

These tests use a synthetic projection fixture and the tracked X2 model, but
they never construct a simulator, environment, viewer, server, learned model,
controller or hardware session.  They make no policy, physics or deployment
claim; the fitted joints are validated as a display aid only.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT
from mjlab.tasks.tracking.diffusion.projection import (
  ProjectionBundle,
  fit_feature_stats,
)
from mjlab.tasks.tracking.diffusion.state import WorldState, world_to_hybrid

_SEQUENCE_PATH = (
  Path(__file__).parents[1]
  / "scripts/experiments/diffusion_d3_closed_loop/sequence_poses.py"
)
_SPEC = importlib.util.spec_from_file_location("d3_sequence_poses", _SEQUENCE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
sequence_poses = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = sequence_poses
_SPEC.loader.exec_module(sequence_poses)

recover_future_poses = sequence_poses.recover_future_poses
fit_pose_sequence = sequence_poses.fit_pose_sequence
SequencePoseError = sequence_poses.SequencePoseError

_XML = (
  Path(__file__).parents[1] / "src/mjlab/asset_zoo/robots/agibot_x2/xmls/x2_ultra.xml"
)
_FUTURE = DEFAULT_CONTRACT.future_steps
_CURRENT = DEFAULT_CONTRACT.current_index
_BODIES = DEFAULT_CONTRACT.body_names


def _yaw_matrix(yaw: float) -> np.ndarray:
  cosine, sine = np.cos(yaw), np.sin(yaw)
  return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def _quaternion(yaw: float) -> np.ndarray:
  return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def _local_bodies() -> np.ndarray:
  return np.arange(60, dtype=np.float64).reshape(20, 3) * 0.01


def _state(row: int) -> WorldState:
  """Return one synthetic yaw-only world state of a smooth trajectory."""
  yaw = 0.05 * row
  position = np.array([0.01 * row, -0.02 * row, 0.7 + 0.001 * row])
  bodies = position + _local_bodies() @ _yaw_matrix(yaw).T
  return WorldState(
    position,
    _quaternion(yaw),
    np.zeros(3),
    np.zeros(3),
    bodies,
    np.zeros((20, 3)),
  )


def _projection() -> ProjectionBundle:
  rng = np.random.default_rng(7)
  return ProjectionBundle.create(
    fit_feature_stats([rng.normal(size=(64, 135)) + 3.0], std_floor=1e-6),
    fit_feature_stats([rng.normal(size=(64, 32))], std_floor=1e-6),
  )


def _window_sample(projection: ProjectionBundle) -> tuple[np.ndarray, WorldState]:
  """Return a [41, 167] sample whose future rows hold the synthetic states."""
  states = [_state(row) for row in range(DEFAULT_CONTRACT.window_steps)]
  hybrid, _ = world_to_hybrid(WorldState.stack(states), center_index=_CURRENT)
  normalized = projection.project_state(hybrid)
  sample = np.zeros((DEFAULT_CONTRACT.window_steps, DEFAULT_CONTRACT.token_dimension))
  sample[:, :135] = normalized
  sample[:, 135:] = 0.25
  return sample, states[_CURRENT]


def _normalized_row(projection: ProjectionBundle, hybrid: np.ndarray) -> np.ndarray:
  return projection.project_state(np.asarray(hybrid, dtype=np.float64)[None])[0]


@pytest.fixture(scope="module")
def x2_model() -> Iterator[tuple[mujoco.MjModel, np.ndarray, np.ndarray]]:
  """Compiled X2 model, ordered body ids and ordered joint qpos addresses."""
  model = mujoco.MjModel.from_xml_path(str(_XML))
  body_ids = np.array(
    [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in _BODIES],
    dtype=np.int64,
  )
  assert (body_ids >= 0).all()
  joint_adr = np.array(
    sorted(
      int(model.jnt_qposadr[joint])
      for joint in range(model.njnt)
      if mujoco.mjtJoint(model.jnt_type[joint])
      in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
    ),
    dtype=np.int64,
  )
  yield model, body_ids, joint_adr


def _fk_bodies(
  model: mujoco.MjModel,
  body_ids: np.ndarray,
  root_position: np.ndarray,
  yaw: float,
  joints: np.ndarray,
) -> np.ndarray:
  """Return the predicted world body positions of one yaw-only root pose."""
  data = mujoco.MjData(model)
  qpos = model.qpos0.copy()
  qpos[:3] = root_position
  qpos[3:7] = _quaternion(yaw)
  qpos[7:] = joints
  data.qpos[:] = qpos
  mujoco.mj_kinematics(model, data)
  return data.xpos[body_ids].copy()


def test_recover_future_poses_returns_the_predicted_future_rows() -> None:
  projection = _projection()
  sample, current = _window_sample(projection)
  poses = recover_future_poses(
    sample, projection=projection, current_state=current, contract=DEFAULT_CONTRACT
  )
  assert set(poses) == {"root_positions", "root_rotations", "body_positions"}
  assert poses["root_positions"].shape == (_FUTURE, 3)
  assert poses["root_rotations"].shape == (_FUTURE, 3, 3)
  assert poses["body_positions"].shape == (_FUTURE, 20, 3)
  for value in poses.values():
    assert value.dtype == np.float64
    assert np.isfinite(value).all()
  future = [_state(row) for row in range(_CURRENT + 1, _CURRENT + 1 + _FUTURE)]
  assert np.allclose(
    poses["root_positions"],
    np.stack([state.root_position for state in future]),
    atol=1e-9,
  )
  # Every synthetic root rotation is yaw-only, so the recovered body positions
  # must reproduce the recorded ones exactly.
  assert np.allclose(
    poses["body_positions"],
    np.stack([state.body_positions for state in future]),
    atol=1e-9,
  )
  for row, state in enumerate(future):
    quaternion = state.root_quaternion_wxyz
    yaw = 2.0 * np.arctan2(quaternion[3], quaternion[0])
    assert np.allclose(poses["root_rotations"][row], _yaw_matrix(yaw), atol=1e-9)


def test_recover_future_poses_ignores_history_and_current_rows() -> None:
  projection = _projection()
  sample, current = _window_sample(projection)
  baseline = recover_future_poses(
    sample, projection=projection, current_state=current, contract=DEFAULT_CONTRACT
  )
  edited = sample.copy()
  edited[: _CURRENT + 1, :] = 5.0
  changed = recover_future_poses(
    edited, projection=projection, current_state=current, contract=DEFAULT_CONTRACT
  )
  for key in baseline:
    assert np.array_equal(baseline[key], changed[key])
  edited[_CURRENT + 1, :135] = -3.0
  shifted = recover_future_poses(
    edited, projection=projection, current_state=current, contract=DEFAULT_CONTRACT
  )
  assert not np.allclose(
    baseline["root_positions"][0], shifted["root_positions"][0], atol=1e-6
  )


def test_recover_future_poses_restores_body_offsets_with_the_predicted_yaw() -> None:
  projection = _projection()
  current = _state(3)
  relative_yaw = 0.9
  row_yaw = 0.05 * 3 + relative_yaw
  local = _local_bodies()
  hybrid = np.zeros(135)
  hybrid[0:3] = [0.4, -0.2, 0.1]
  hybrid[3:9] = _yaw_matrix(relative_yaw)[:, :2].T.reshape(-1)
  hybrid[15:75] = local.reshape(-1)
  normalized = _normalized_row(projection, hybrid)
  sample = np.zeros((DEFAULT_CONTRACT.window_steps, DEFAULT_CONTRACT.token_dimension))
  sample[_CURRENT + 1, :135] = normalized
  poses = recover_future_poses(
    sample, projection=projection, current_state=current, contract=DEFAULT_CONTRACT
  )
  center_rotation = _yaw_matrix(0.05 * 3)
  expected_root = current.root_position + np.array([0.4, -0.2, 0.1]) @ center_rotation.T
  expected_bodies = expected_root + local @ _yaw_matrix(row_yaw).T
  assert np.allclose(poses["root_positions"][0], expected_root, atol=1e-9)
  assert np.allclose(poses["body_positions"][0], expected_bodies, atol=1e-9)
  wrong_bodies = expected_root + local @ center_rotation.T
  assert not np.allclose(poses["body_positions"][0], wrong_bodies, atol=1e-6)


def test_recover_future_poses_detaches_torch_input_without_mutating_it() -> None:
  projection = _projection()
  sample, current = _window_sample(projection)
  reference = recover_future_poses(
    sample, projection=projection, current_state=current, contract=DEFAULT_CONTRACT
  )
  before = sample.copy()
  tensor = torch.tensor(sample.astype(np.float32), requires_grad=True)
  poses = recover_future_poses(
    tensor, projection=projection, current_state=current, contract=DEFAULT_CONTRACT
  )
  for key in reference:
    assert np.allclose(reference[key], poses[key], atol=1e-5)
  assert np.array_equal(sample, before)
  poses["body_positions"][:] = -1.0
  assert np.array_equal(sample, before)


def test_recover_future_poses_rejects_bad_windows() -> None:
  projection = _projection()
  sample, current = _window_sample(projection)
  with pytest.raises(SequencePoseError):
    recover_future_poses(
      sample[:-1],
      projection=projection,
      current_state=current,
      contract=DEFAULT_CONTRACT,
    )
  broken = sample.copy()
  broken[12, 0] = np.nan
  with pytest.raises(SequencePoseError):
    recover_future_poses(
      broken, projection=projection, current_state=current, contract=DEFAULT_CONTRACT
    )
  with pytest.raises(SequencePoseError):
    recover_future_poses(
      "not a sample",
      projection=projection,
      current_state=current,
      contract=DEFAULT_CONTRACT,
    )


def test_fit_pose_sequence_recovers_fk_reachable_rows(
  x2_model: tuple[mujoco.MjModel, np.ndarray, np.ndarray],
) -> None:
  model, body_ids, joint_adr = x2_model
  lower, upper = _limits(model, joint_adr)
  joint_seed = model.qpos0[joint_adr].copy()
  rows = _FUTURE
  roots = np.array([[0.03 * row, -0.01 * row, 0.7] for row in range(rows)])
  yaws = 0.02 * np.arange(rows)
  targets = np.empty((rows, len(_BODIES), 3))
  for row in range(rows):
    track = np.clip(
      joint_seed + 0.2 * np.sin(0.2 * row) * np.cos(np.arange(joint_adr.size)),
      lower,
      upper,
    )
    targets[row] = _fk_bodies(model, body_ids, roots[row], yaws[row], track)
  initial_qpos = model.qpos0.copy()
  before_qpos = initial_qpos.copy()
  before_roots = roots.copy()
  before_targets = targets.copy()
  before_model = (model.qpos0.copy(), model.jnt_range.copy())
  fitted = fit_pose_sequence(
    model,
    initial_qpos=initial_qpos,
    free_joint_qpos_adr=np.arange(7, dtype=np.int64),
    joint_qpos_adr=joint_adr,
    body_names=_BODIES,
    root_positions=roots,
    root_rotations=np.stack([_yaw_matrix(yaw) for yaw in yaws]),
    body_positions=targets,
  )
  assert set(fitted) == {"qpos", "body_errors", "converged", "iterations"}
  assert fitted["qpos"].shape == (rows, model.nq)
  assert fitted["body_errors"].shape == (rows, len(_BODIES))
  assert fitted["converged"].shape == (rows,)
  assert fitted["iterations"].shape == (rows,)
  assert fitted["qpos"].dtype == np.float64
  assert fitted["converged"].dtype == np.bool_
  assert fitted["iterations"].dtype == np.int64
  assert fitted["converged"].all()
  assert fitted["body_errors"].max() < 1e-3
  assert np.isfinite(fitted["qpos"]).all()
  assert np.isfinite(fitted["body_errors"]).all()
  # The predicted root pose is fixed and never solved.
  assert np.allclose(fitted["qpos"][:, :3], roots, atol=1e-12)
  for row in range(rows):
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, fitted["qpos"][row, 3:7])
    assert np.allclose(rotation.reshape(3, 3), _yaw_matrix(yaws[row]), atol=1e-12)
  # Inputs and the model are never written.
  assert np.array_equal(initial_qpos, before_qpos)
  assert np.array_equal(roots, before_roots)
  assert np.array_equal(targets, before_targets)
  assert np.array_equal(model.qpos0, before_model[0])
  assert np.array_equal(model.jnt_range, before_model[1])
  fitted["qpos"][:] = -5.0
  assert np.array_equal(initial_qpos, before_qpos)


def test_fit_pose_sequence_respects_joint_limits(
  x2_model: tuple[mujoco.MjModel, np.ndarray, np.ndarray],
) -> None:
  model, body_ids, joint_adr = x2_model
  lower, upper = _limits(model, joint_adr)
  rng = np.random.default_rng(5)
  rows = 8
  roots = np.zeros((rows, 3))
  roots[:, 2] = 0.7
  yaws = np.zeros(rows)
  targets = np.empty((rows, len(_BODIES), 3))
  for row in range(rows):
    track = np.clip(
      model.qpos0[joint_adr] + 0.8 * rng.normal(size=joint_adr.size), lower, upper
    )
    targets[row] = _fk_bodies(model, body_ids, roots[row], 0.0, track)
  fitted = fit_pose_sequence(
    model,
    initial_qpos=model.qpos0.copy(),
    free_joint_qpos_adr=np.arange(7, dtype=np.int64),
    joint_qpos_adr=joint_adr,
    body_names=_BODIES,
    root_positions=roots,
    root_rotations=np.stack([_yaw_matrix(yaw) for yaw in yaws]),
    body_positions=targets,
  )
  solved = fitted["qpos"][:, joint_adr]
  assert np.all(solved >= lower[None, :] - 1e-12)
  assert np.all(solved <= upper[None, :] + 1e-12)
  assert fitted["iterations"].max() <= 30
  assert np.isfinite(fitted["body_errors"]).all()


def test_fit_pose_sequence_reports_unreachable_targets(
  x2_model: tuple[mujoco.MjModel, np.ndarray, np.ndarray],
) -> None:
  model, body_ids, joint_adr = x2_model
  roots = np.array([[0.0, 0.0, 0.7]])
  targets = _fk_bodies(model, body_ids, roots[0], 0.0, model.qpos0[joint_adr].copy())[
    None
  ]
  shifted = targets + np.array([0.0, 0.0, 0.6])
  fitted = fit_pose_sequence(
    model,
    initial_qpos=model.qpos0.copy(),
    free_joint_qpos_adr=np.arange(7, dtype=np.int64),
    joint_qpos_adr=joint_adr,
    body_names=_BODIES,
    root_positions=roots,
    root_rotations=_yaw_matrix(0.0)[None],
    body_positions=shifted,
    max_iterations=5,
  )
  assert not fitted["converged"].any()
  assert fitted["iterations"][0] == 5
  assert fitted["body_errors"].min() > 0.05
  assert np.isfinite(fitted["body_errors"]).all()
  lower, upper = _limits(model, joint_adr)
  assert np.all(fitted["qpos"][:, joint_adr] >= lower[None, :] - 1e-12)
  assert np.all(fitted["qpos"][:, joint_adr] <= upper[None, :] + 1e-12)
  assert np.allclose(fitted["qpos"][:, :3], roots, atol=1e-12)


def test_fit_pose_sequence_keeps_an_unobservable_joint_at_its_seed(
  x2_model: tuple[mujoco.MjModel, np.ndarray, np.ndarray],
) -> None:
  model, body_ids, joint_adr = x2_model
  # ``left_ankle_roll_joint`` spins a link about its own origin, and the
  # contract observes only that link's origin, so no predicted body position
  # can constrain the joint.
  roll = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_ankle_roll_joint")
  roll_adr = int(model.jnt_qposadr[roll])
  index = int(np.flatnonzero(joint_adr == roll_adr)[0])
  measured = model.qpos0.copy()
  measured[joint_adr[index]] = 0.2
  target_joints = measured[joint_adr].copy()
  target_joints[index] = -0.05
  root = np.array([0.0, 0.0, 0.7])
  targets = _fk_bodies(model, body_ids, root, 0.0, target_joints)[None]
  assert np.allclose(
    targets[0], _fk_bodies(model, body_ids, root, 0.0, measured[joint_adr]), atol=1e-9
  )
  fitted = fit_pose_sequence(
    model,
    initial_qpos=measured,
    free_joint_qpos_adr=np.arange(7, dtype=np.int64),
    joint_qpos_adr=joint_adr,
    body_names=_BODIES,
    root_positions=root[None],
    root_rotations=_yaw_matrix(0.0)[None],
    body_positions=targets,
  )
  assert fitted["converged"].all()
  assert fitted["body_errors"].max() < 1e-9
  assert fitted["qpos"][0, joint_adr[index]] == pytest.approx(0.2, abs=1e-12)


def test_recover_then_fit_round_trips_a_recorded_pose_sequence(
  x2_model: tuple[mujoco.MjModel, np.ndarray, np.ndarray],
) -> None:
  """A physically consistent window must recover and fit as its own FK pose."""
  model, body_ids, joint_adr = x2_model
  lower, upper = _limits(model, joint_adr)
  data = mujoco.MjData(model)
  states = []
  qpos_rows = []
  for row in range(DEFAULT_CONTRACT.window_steps):
    joints = np.clip(
      model.qpos0[joint_adr]
      + 0.15 * np.sin(0.1 * row) * np.cos(np.arange(joint_adr.size)),
      lower,
      upper,
    )
    qpos = model.qpos0.copy()
    qpos[:3] = [0.03 * row, -0.01 * row, 0.7]
    qpos[3:7] = _quaternion(0.02 * row)
    qpos[7:] = joints
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    states.append(
      WorldState(
        qpos[:3].copy(),
        qpos[3:7].copy(),
        np.zeros(3),
        np.zeros(3),
        data.xpos[body_ids].copy(),
        np.zeros((20, 3)),
      )
    )
    qpos_rows.append(qpos)
  projection = _projection()
  hybrid, _ = world_to_hybrid(WorldState.stack(states), center_index=_CURRENT)
  sample = np.zeros((DEFAULT_CONTRACT.window_steps, DEFAULT_CONTRACT.token_dimension))
  sample[:, :135] = projection.project_state(hybrid)
  poses = recover_future_poses(
    sample,
    projection=projection,
    current_state=states[_CURRENT],
    contract=DEFAULT_CONTRACT,
  )
  assert np.allclose(
    poses["root_positions"],
    np.stack([state.root_position for state in states[_CURRENT + 1 :]]),
    atol=1e-9,
  )
  fitted = fit_pose_sequence(
    model,
    initial_qpos=qpos_rows[_CURRENT],
    free_joint_qpos_adr=np.arange(7, dtype=np.int64),
    joint_qpos_adr=joint_adr,
    body_names=_BODIES,
    root_positions=poses["root_positions"],
    root_rotations=poses["root_rotations"],
    body_positions=poses["body_positions"],
  )
  assert fitted["converged"].all()
  assert fitted["body_errors"].max() < 1e-3
  assert np.isfinite(fitted["qpos"]).all()


def test_fit_pose_sequence_rejects_invalid_requests(
  x2_model: tuple[mujoco.MjModel, np.ndarray, np.ndarray],
) -> None:
  model, body_ids, joint_adr = x2_model
  root = np.array([[0.0, 0.0, 0.7]])
  targets = _fk_bodies(model, body_ids, root[0], 0.0, model.qpos0[joint_adr].copy())[
    None
  ]
  request = dict(
    initial_qpos=model.qpos0.copy(),
    free_joint_qpos_adr=np.arange(7, dtype=np.int64),
    joint_qpos_adr=joint_adr,
    body_names=_BODIES,
    root_positions=root,
    root_rotations=_yaw_matrix(0.0)[None],
    body_positions=targets,
  )
  with pytest.raises(SequencePoseError):
    fit_pose_sequence(model, **{**request, "free_joint_qpos_adr": np.arange(6)})
  with pytest.raises(SequencePoseError):
    fit_pose_sequence(model, **{**request, "body_names": _BODIES[:-1] + ("missing",)})
  with pytest.raises(SequencePoseError):
    fit_pose_sequence(model, **{**request, "body_names": _BODIES[:-1] + ("",)})
  with pytest.raises(SequencePoseError):
    fit_pose_sequence(
      model, **{**request, "joint_qpos_adr": np.append(joint_adr, joint_adr[0])}
    )
  with pytest.raises(SequencePoseError):
    fit_pose_sequence(model, **{**request, "max_iterations": 0})
  with pytest.raises(SequencePoseError):
    fit_pose_sequence(model, **{**request, "root_rotations": np.full((1, 3, 3), 0.5)})
  with pytest.raises(SequencePoseError):
    fit_pose_sequence(model, **{**request, "body_positions": targets[:, :19]})


def _limits(
  model: mujoco.MjModel, joint_adr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """Return the independent hard joint limits of the robot joint addresses."""
  by_qpos = {
    int(model.jnt_qposadr[joint]): joint
    for joint in range(model.njnt)
    if mujoco.mjtJoint(model.jnt_type[joint])
    in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
  }
  lower = np.empty(joint_adr.size)
  upper = np.empty(joint_adr.size)
  for index, address in enumerate(joint_adr):
    joint = by_qpos[int(address)]
    lower[index], upper[index] = model.jnt_range[joint]
  return lower, upper


def test_fit_clips_joint_seed_before_zero_iteration_return(
  x2_model: tuple[mujoco.MjModel, np.ndarray, np.ndarray],
) -> None:
  model, body_ids, joint_adr = x2_model
  lower, upper = _limits(model, joint_adr)
  initial = model.qpos0.copy()
  initial[:3] = [0.0, 0.0, 0.8]
  initial[joint_adr[0]] = upper[0] + 0.5
  clipped = np.clip(initial[joint_adr], lower, upper)
  targets = _fk_bodies(model, body_ids, initial[:3], 0.0, clipped)
  fitted = fit_pose_sequence(
    model,
    initial_qpos=initial,
    free_joint_qpos_adr=np.arange(7),
    joint_qpos_adr=joint_adr,
    body_names=_BODIES,
    root_positions=initial[None, :3],
    root_rotations=np.eye(3)[None],
    body_positions=targets[None],
  )
  assert fitted["iterations"].tolist() == [0]
  assert fitted["converged"].tolist() == [True]
  assert np.allclose(fitted["qpos"][0, joint_adr], clipped)


def test_body_lookup_handles_scene_attachment_prefixes() -> None:
  parent = mujoco.MjSpec()
  robot = mujoco.MjSpec()
  robot.worldbody.add_body(name="pelvis")
  robot.worldbody.add_body(name="torso_link")
  parent.attach(robot, prefix="robot/", frame=parent.worldbody.add_frame())
  model = parent.compile()
  assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis") == -1
  ids = sequence_poses._body_ids(model, ("pelvis", "torso_link"))
  assert [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(i)) for i in ids] == [
    "robot/pelvis",
    "robot/torso_link",
  ]


def test_body_lookup_rejects_ambiguous_entity_names() -> None:
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="first/pelvis")
  spec.worldbody.add_body(name="second/pelvis")
  with pytest.raises(SequencePoseError, match="ambiguous"):
    sequence_poses._body_ids(spec.compile(), ("pelvis",))
