#!/usr/bin/env python3
"""Display-only future-pose recovery and IK fitting for the D3 sequence viewer.

This is a local experiment helper, not a new core diffusion seam.  It owns the
two pure functions that the D3 sequence viewer needs:

* :func:`recover_future_poses` converts the ``future_steps`` predicted rows of
  one completed sampler window into world root/body poses.  It delegates to the
  reviewed :func:`~mjlab.tasks.tracking.diffusion.pose_costs.world_poses_from_normalized_state`
  conversion, so the viewer and the pose-cost guidance seam can never disagree
  about what a generated state means.
* :func:`fit_pose_sequence` fits display-only joint angles to those predicted
  body positions with an isolated CPU MuJoCo model.  The predicted root pose is
  fixed, joint limits are enforced, and the fit is seeded from the measured
  joints and then from the previous fitted row.

The fitted joints are a rendering aid.  They are never model predictions, they
never feed physics, policy or control, and a failed fit reports its residual
error instead of a fabricated pose.  No simulator, learned model, checkpoint,
controller, viewer server or hardware operation is involved here.
"""

from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np
import torch

from mjlab.tasks.tracking.diffusion.contract import DiffusionContract
from mjlab.tasks.tracking.diffusion.pose_costs import world_poses_from_normalized_state
from mjlab.tasks.tracking.diffusion.projection import ProjectionBundle
from mjlab.tasks.tracking.diffusion.state import WorldState

# Display fit acceptance: a body inside one millimetre of its predicted world
# position is converged for a transparent ghost mesh.
_FIT_TOLERANCE_METERS = 1.0e-3

# Damped least-squares damping (metres per radian).  The damping also acts as
# the weak seed regularization: it penalizes the step away from the seed, so a
# joint that moves no observed body keeps its seed value instead of drifting.
_DAMPING = 5.0e-3

# Bound on one damped least-squares joint step, in radians.  It keeps the
# linearization valid when a row is already far from its seed.
_MAX_JOINT_STEP_RADIANS = 0.5

# A recovered rotation must be a proper rotation within this tolerance before
# it is converted to a fixed-root quaternion.
_ROTATION_TOLERANCE = 1.0e-6


class SequencePoseError(ValueError):
  """A recovered future pose or an IK request is invalid."""


def recover_future_poses(
  sample: np.ndarray | torch.Tensor,
  *,
  projection: ProjectionBundle,
  current_state: WorldState,
  contract: DiffusionContract,
) -> dict[str, np.ndarray]:
  """Recover the predicted future world poses of one completed window.

  ``sample`` is the final sampler result with shape
  ``[window_steps, token_dimension]`` (``[41, 167]`` for the frozen contract) as
  a CPU NumPy array or a torch tensor.  The returned arrays hold one row per
  predicted future step: ``root_positions`` ``[32, 3]``, ``root_rotations``
  ``[32, 3, 3]`` and ``body_positions`` ``[32, 20, 3]``, all owned float64
  NumPy copies.  History and current rows are ignored, and the body offsets are
  restored with each row's predicted yaw, never with a recorded future yaw.
  """
  measured = _measured_sample(sample, contract)
  future = torch.as_tensor(
    measured[
      contract.current_index + 1 : contract.current_index + 1 + contract.future_steps,
      : contract.state_dimension,
    ]
  )
  poses = world_poses_from_normalized_state(
    future,
    projection=projection,
    current_state=current_state,
    contract=contract,
  )
  return {
    "root_positions": _owned_array(poses.root_position),
    "root_rotations": _owned_array(poses.root_rotation),
    "body_positions": _owned_array(poses.body_positions),
  }


def fit_pose_sequence(
  model: mujoco.MjModel,
  *,
  initial_qpos: np.ndarray,
  free_joint_qpos_adr: np.ndarray,
  joint_qpos_adr: np.ndarray,
  body_names: Sequence[str],
  root_positions: np.ndarray,
  root_rotations: np.ndarray,
  body_positions: np.ndarray,
  max_iterations: int = 30,
) -> dict[str, np.ndarray]:
  """Fit display-only joint angles to a sequence of world body poses.

  ``model`` is an isolated CPU MuJoCo model and ``initial_qpos`` is the
  measured window-center ``qpos``.  ``free_joint_qpos_adr`` holds the seven
  measured root ``qpos`` addresses in free-joint order (position xyz, then
  quaternion wxyz) and ``joint_qpos_adr`` holds the scalar robot-joint ``qpos``
  addresses in model/robot order.  ``body_names`` is the ordered contract body
  list, and the three pose arrays hold one row per predicted future step.

  Every row fixes its predicted root position and rotation, then solves the
  robot joints with damped least-squares MuJoCo body Jacobians, joint limits
  and a bounded iteration count.  The first row is seeded from the measured
  joints and each later row from the previous fit, so a joint that moves no
  observed body stays near its seed.

  The result owns ``qpos`` ``[32, nq]``, ``body_errors`` ``[32, n_bodies]`` in
  metres, ``converged`` ``[32]`` and ``iterations`` ``[32]``.  No input array
  and no model field is written.
  """
  if not isinstance(model, mujoco.MjModel):
    raise SequencePoseError("model must be a mujoco.MjModel")
  if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
    raise SequencePoseError("max_iterations must be an integer")
  if max_iterations < 1:
    raise SequencePoseError("max_iterations must be at least one")
  names = tuple(body_names)
  if (
    not names
    or any(not isinstance(name, str) or not name for name in names)
    or len(set(names)) != len(names)
  ):
    raise SequencePoseError("body_names must be non-empty and unique")
  initial = _initial_qpos(initial_qpos, model.nq)
  free_adr = _addresses(free_joint_qpos_adr, 7, model.nq, "free_joint_qpos_adr")
  joint_adr = _addresses(joint_qpos_adr, None, model.nq, "joint_qpos_adr")
  if np.intersect1d(free_adr, joint_adr).size:
    raise SequencePoseError("free and joint qpos addresses overlap")
  body_ids = _body_ids(model, names)
  joints, dofs = _joint_columns(model, joint_adr)
  lower, upper = _joint_limits(model, joints)
  roots, rotations, targets = _predicted_rows(
    names, root_positions, root_rotations, body_positions
  )

  count = roots.shape[0]
  data = mujoco.MjData(model)
  jacobian = np.empty((3 * len(names), joint_adr.size), dtype=np.float64)
  translation = np.zeros((3, model.nv), dtype=np.float64)
  rotation_jacobian = np.zeros((3, model.nv), dtype=np.float64)
  normal = np.empty((joint_adr.size, joint_adr.size), dtype=np.float64)
  qpos = np.array(initial, dtype=np.float64, copy=True)
  seed = np.array(initial[joint_adr], dtype=np.float64, copy=True)
  solved = np.empty((count, model.nq), dtype=np.float64)
  errors = np.empty((count, len(names)), dtype=np.float64)
  converged = np.zeros(count, dtype=bool)
  iterations = np.zeros(count, dtype=np.int64)
  for row in range(count):
    qpos[:] = initial
    qpos[free_adr[:3]] = roots[row]
    qpos[free_adr[3:]] = _matrix_to_quaternion(rotations[row])
    qpos[joint_adr] = np.clip(seed, lower, upper)
    updates = 0
    for _ in range(max_iterations):
      _kinematics(model, data, qpos)
      residual = targets[row] - data.xpos[body_ids]
      if float(np.linalg.norm(residual, axis=1).max()) <= _FIT_TOLERANCE_METERS:
        break
      _fill_body_jacobian(
        model,
        data,
        body_ids,
        dofs,
        jacobian,
        translation,
        rotation_jacobian,
      )
      normal[:] = jacobian.T @ jacobian
      normal[np.diag_indices(joint_adr.size)] += _DAMPING**2
      step = np.linalg.solve(normal, jacobian.T @ residual.reshape(-1))
      np.clip(step, -_MAX_JOINT_STEP_RADIANS, _MAX_JOINT_STEP_RADIANS, out=step)
      qpos[joint_adr] = np.clip(qpos[joint_adr] + step, lower, upper)
      updates += 1
    _kinematics(model, data, qpos)
    solved[row] = qpos
    errors[row] = np.linalg.norm(targets[row] - data.xpos[body_ids], axis=1)
    iterations[row] = updates
    converged[row] = bool(errors[row].max() <= _FIT_TOLERANCE_METERS)
    seed = qpos[joint_adr].copy()
  return {
    "qpos": solved,
    "body_errors": errors,
    "converged": converged,
    "iterations": iterations,
  }


def _measured_sample(
  sample: np.ndarray | torch.Tensor, contract: DiffusionContract
) -> np.ndarray:
  """Return the float64 sample window; the caller never writes to it."""
  if isinstance(sample, torch.Tensor):
    measured = sample.detach().to(device="cpu", dtype=torch.float64).numpy()
  elif isinstance(sample, np.ndarray):
    measured = np.asarray(sample, dtype=np.float64)
  else:
    raise SequencePoseError("sample must be a NumPy array or a torch tensor")
  expected = (contract.window_steps, contract.token_dimension)
  if measured.ndim != 2 or measured.shape != expected:
    raise SequencePoseError(f"sample must have shape {expected}, got {measured.shape}")
  if not bool(np.isfinite(measured).all()):
    raise SequencePoseError("sample contains non-finite values")
  return measured


def _owned_array(value: torch.Tensor) -> np.ndarray:
  """Detach one pose tensor into an owned float64 NumPy array."""
  return np.array(value.detach().cpu().numpy(), dtype=np.float64, copy=True)


def _initial_qpos(initial_qpos: np.ndarray, nq: int) -> np.ndarray:
  initial = np.asarray(initial_qpos, dtype=np.float64)
  if initial.ndim != 1 or initial.shape[0] != nq:
    raise SequencePoseError(f"initial_qpos must have shape [{nq}]")
  if not bool(np.isfinite(initial).all()):
    raise SequencePoseError("initial_qpos contains non-finite values")
  return initial


def _addresses(value: np.ndarray, width: int | None, nq: int, name: str) -> np.ndarray:
  addresses = np.asarray(value, dtype=np.int64)
  if addresses.ndim != 1 or addresses.size == 0:
    raise SequencePoseError(f"{name} must be a non-empty one-dimensional array")
  if width is not None and addresses.size != width:
    raise SequencePoseError(f"{name} must hold exactly {width} addresses")
  if np.unique(addresses).size != addresses.size:
    raise SequencePoseError(f"{name} contains duplicate qpos addresses")
  if int(addresses.min()) < 0 or int(addresses.max()) >= nq:
    raise SequencePoseError(f"{name} addresses fall outside qpos [0, {nq})")
  return addresses


def _body_ids(model: mujoco.MjModel, names: tuple[str, ...]) -> np.ndarray:
  body_ids = np.empty(len(names), dtype=np.int64)
  for index, name in enumerate(names):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
      # Scene attachment adds an entity prefix, e.g. robot/pelvis.
      matches = [
        candidate
        for candidate in range(1, model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, candidate) or "").rsplit(
          "/", 1
        )[-1]
        == name
      ]
      if len(matches) != 1:
        raise SequencePoseError(f"model body name {name!r} is missing or ambiguous")
      body_id = matches[0]
    body_ids[index] = body_id
  return body_ids


def _joint_columns(
  model: mujoco.MjModel, joint_adr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """Map robot-joint qpos addresses to their model joints and velocity DOFs."""
  by_qpos: dict[int, int] = {}
  for joint in range(model.njnt):
    kind = mujoco.mjtJoint(model.jnt_type[joint])
    if kind in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
      by_qpos[int(model.jnt_qposadr[joint])] = joint
  joints = np.empty(joint_adr.size, dtype=np.int64)
  for index, address in enumerate(joint_adr):
    joint = by_qpos.get(int(address))
    if joint is None:
      raise SequencePoseError(
        f"qpos address {address} is not a one-dof robot joint in the model"
      )
    joints[index] = joint
  dofs = np.array([int(model.jnt_dofadr[joint]) for joint in joints], dtype=np.int64)
  return joints, dofs


def _joint_limits(
  model: mujoco.MjModel, joints: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """Return the hard joint limits; an unlimited joint gets an infinite range."""
  lower = np.empty(joints.size, dtype=np.float64)
  upper = np.empty(joints.size, dtype=np.float64)
  for index, joint in enumerate(joints):
    if model.jnt_limited[joint]:
      lower[index], upper[index] = model.jnt_range[joint]
    else:
      lower[index], upper[index] = -np.inf, np.inf
  return lower, upper


def _predicted_rows(
  names: tuple[str, ...],
  root_positions: np.ndarray,
  root_rotations: np.ndarray,
  body_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  roots = np.asarray(root_positions, dtype=np.float64)
  rotations = np.asarray(root_rotations, dtype=np.float64)
  targets = np.asarray(body_positions, dtype=np.float64)
  if roots.ndim != 2 or roots.shape[1] != 3:
    raise SequencePoseError("root_positions must have shape [rows, 3]")
  count = roots.shape[0]
  if count == 0:
    raise SequencePoseError("root_positions must hold at least one row")
  if rotations.shape != (count, 3, 3):
    raise SequencePoseError("root_rotations must have shape [rows, 3, 3]")
  if targets.shape != (count, len(names), 3):
    raise SequencePoseError(
      f"body_positions must have shape [{count}, {len(names)}, 3]"
    )
  for name, value in (
    ("root_positions", roots),
    ("root_rotations", rotations),
    ("body_positions", targets),
  ):
    if not bool(np.isfinite(value).all()):
      raise SequencePoseError(f"{name} contains non-finite values")
  if not bool(
    np.allclose(
      rotations @ np.swapaxes(rotations, -1, -2),
      np.eye(3),
      atol=_ROTATION_TOLERANCE,
    )
  ):
    raise SequencePoseError("root_rotations must be orthonormal matrices")
  if not bool(np.allclose(np.linalg.det(rotations), 1.0, atol=_ROTATION_TOLERANCE)):
    raise SequencePoseError("root_rotations must be proper rotations")
  return roots, rotations, targets


def _kinematics(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray) -> None:
  """Refresh frame kinematics only.

  Contacts, constraints and actuators are never computed, so the display fit
  cannot be influenced by physics state.  ``mj_comPos`` is required because
  ``mj_jacBody`` reads the body DOF Jacobians it builds.
  """
  data.qpos[:] = qpos
  mujoco.mj_kinematics(model, data)
  mujoco.mj_comPos(model, data)


def _fill_body_jacobian(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  body_ids: np.ndarray,
  dofs: np.ndarray,
  jacobian: np.ndarray,
  translation: np.ndarray,
  rotation: np.ndarray,
) -> None:
  """Stack the per-body world-position Jacobians over the robot-joint DOFs."""
  for index, body_id in enumerate(body_ids):
    mujoco.mj_jacBody(model, data, translation, rotation, int(body_id))
    jacobian[3 * index : 3 * index + 3] = translation[:, dofs]


def _matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
  """Return the wxyz quaternion of one proper rotation matrix."""
  quaternion = np.empty(4, dtype=np.float64)
  mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(rotation).reshape(9))
  return quaternion


__all__ = ["SequencePoseError", "fit_pose_sequence", "recover_future_poses"]
