"""Single-teacher adapter for aligned live observations.

The adapter deliberately exposes a small boundary:

* ``reset(seed=None) -> DistillationSnapshot``
* ``step(action) -> DistillationStep``
* ``snapshot() -> DistillationSnapshot``

``snapshot`` reads the observation manager cache, never ``compute_group``.  It
owns copies of all tensors before the simulator can mutate its buffers.  The
collector can label ``snapshot.teacher_observation`` and pack
``snapshot.features`` without recomputing noise, delay, or history.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from mjlab.tasks.tracking.distillation.config import CohortContract, DistillationError
from mjlab.tasks.tracking.distillation.environment import (
  ReferenceBoundaryEvents,
  RuntimeSeedProvenance,
  SegmentMotionCommand,
  build_distillation_environment,
)
from mjlab.tasks.tracking.distillation.observations import (
  ObservationSnapshot,
  PackedObservationBatch,
  pack_observations,
)
from mjlab.tasks.tracking.distillation.teachers import (
  FrozenTeacher,
  load_frozen_teacher,
)
from mjlab.tasks.tracking.distillation.vae_config import (
  VaeSchema,
  make_schema,
)
from mjlab.utils.lab_api.math import euler_xyz_from_quat


@dataclass(frozen=True, slots=True)
class SensorFrameEvidence:
  """Compiled MuJoCo identity and local frame of one declared sensor."""

  name: str
  sensor_type: int
  object_type: int
  object_name: str
  site_name: str
  body_name: str
  local_quat_wxyz: tuple[float, float, float, float]
  expected_type: str
  frame_verified: bool


@dataclass(frozen=True, slots=True)
class PhysicalTrackingMetrics:
  """Owned per-environment errors captured at one aligned simulator instant."""

  aligned_time: str
  global_body_pose_error: torch.Tensor
  root_relative_pose_error: torch.Tensor
  root_relative_yaw_error: torch.Tensor
  heading_error: torch.Tensor
  anchor_position_error: torch.Tensor


@dataclass(frozen=True, slots=True)
class AssetFrameAudit:
  """CPU/live asset evidence, including unresolved NPZ body correspondence."""

  robot_bodies: tuple[str, ...]
  tracked_bodies: tuple[str, ...]
  tracked_body_indices: tuple[int, ...]
  anchor_body: str
  root_frame: str
  sensor_evidence: tuple[SensorFrameEvidence, ...]
  root_body_name: str
  anchor_body_id: int
  body_reference_compared: tuple[str, ...]
  body_reference_max_abs_error: tuple[tuple[str, float], ...]
  body_reference_atol: float
  body_reference_rtol: float
  body_reference_convention: str
  unresolved: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LiveContractAudit:
  """Saved/live checks made before a collector may use an environment."""

  teacher_id: str
  task_id: str
  joint_names: tuple[str, ...]
  control_period_s: float
  actor_terms: tuple[str, ...]
  asset: AssetFrameAudit
  additional_gravity_policy: str
  semantic_overrides: tuple[str, ...] = ()
  seed_provenance: RuntimeSeedProvenance | None = None


@dataclass(frozen=True, slots=True)
class DistillationSnapshot:
  """Owned, aligned tensors captured at one post-observation-manager instant."""

  teacher_observation: torch.Tensor
  features: ObservationSnapshot
  packed: PackedObservationBatch
  teacher_id: str
  teacher_code: int
  motion_id: torch.Tensor
  reference_frame: torch.Tensor
  segment_id: torch.Tensor
  generation_id: torch.Tensor
  metrics: PhysicalTrackingMetrics | None = None
  boundary_events: ReferenceBoundaryEvents | None = None

  def __post_init__(self) -> None:
    batch = self.teacher_observation.shape[0]
    for name in ("motion_id", "reference_frame", "segment_id", "generation_id"):
      value = getattr(self, name)
      if value.shape != (batch,):
        raise ValueError(f"{name} must have shape [{batch}], got {tuple(value.shape)}")
      if value.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be an integer tensor")


@dataclass(frozen=True, slots=True)
class DistillationStep:
  """Result of one unchanged simulator action step."""

  snapshot: DistillationSnapshot
  reward: torch.Tensor
  terminated: torch.Tensor
  time_outs: torch.Tensor
  extras: dict[str, Any]
  events: ReferenceBoundaryEvents | None = None


_TERM_FEATURES = {
  "command",
  "motion_anchor_ori_b",
  "base_ang_vel",
  "joint_pos",
  "joint_vel",
  "actions",
}


def _tensor_equal(left: Any, right: Any) -> bool:
  if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
      return False
    return bool(torch.equal(left.detach().cpu(), right.detach().cpu()))
  return left == right


def _live_noise_signature(noise: Any) -> tuple[str, dict[str, Any]] | None:
  if noise is None:
    return None
  operation = getattr(noise, "operation", None)
  if not isinstance(operation, str):
    return None
  params: dict[str, Any] = {}
  for name in ("n_min", "n_max", "mean", "std", "bias"):
    if hasattr(noise, name):
      params[name] = getattr(noise, name)
  return operation, params


def _plain(value: Any) -> Any:
  if isinstance(value, torch.Tensor):
    return tuple(value.detach().cpu().reshape(-1).tolist())
  if isinstance(value, Mapping):
    return {str(key): _plain(item) for key, item in value.items()}
  if isinstance(value, (tuple, list)):
    return tuple(_plain(item) for item in value)
  return value


def _callable_name(value: Any) -> str:
  module = getattr(value, "__module__", None)
  qualname = getattr(value, "__qualname__", None)
  if isinstance(module, str) and isinstance(qualname, str):
    return f"{module}.{qualname}"
  return f"{type(value).__module__}.{type(value).__qualname__}"


def _scale_signature(value: Any) -> tuple[float, ...] | None:
  if value is None:
    return None
  tensor = torch.as_tensor(value, dtype=torch.float64).detach().cpu()
  return tuple(float(item) for item in tensor.reshape(-1).tolist())


def _validate_observation_contract(
  env: Any, cohort: CohortContract, saved_env_config: Mapping[str, Any]
) -> None:
  manager = env.observation_manager
  names = tuple(manager.active_terms.get("actor", ()))
  expected = cohort.observations
  if names != expected.names:
    raise DistillationError(
      f"live actor observation order {names} disagrees with saved {expected.names}"
    )
  actor_cfg = saved_env_config.get("observations", {}).get("actor", {})
  if not isinstance(actor_cfg, Mapping):
    raise DistillationError("saved actor observation group is not a mapping")
  saved_terms = actor_cfg.get("terms", {})
  if not isinstance(saved_terms, Mapping):
    raise DistillationError("saved actor observation terms are not a mapping")
  live_group_cfg = manager.cfg["actor"]
  if live_group_cfg.enable_corruption != expected.enable_corruption:
    raise DistillationError(
      "live actor corruption setting disagrees with saved teacher"
    )
  if not manager.group_obs_concatenate.get("actor", False):
    raise DistillationError("live actor observations must be one concatenated tensor")
  dims = manager.group_obs_term_dim["actor"]
  for name, dim, saved in zip(names, dims, expected.terms, strict=True):
    width = math.prod(dim)
    if width != saved.width:
      raise DistillationError(
        f"live actor term {name!r} width {width} disagrees with saved {saved.width}"
      )
    raw_saved = saved_terms.get(name)
    if not isinstance(raw_saved, Mapping):
      raise DistillationError(f"saved actor term {name!r} is missing")
    live = manager.get_term_cfg("actor", name)
    saved_func = raw_saved.get("func")
    if not isinstance(saved_func, str) or _callable_name(live.func) != saved_func:
      raise DistillationError(f"live actor term {name!r} function disagrees with saved")
    saved_params = raw_saved.get("params", {})
    if _plain(live.params) != _plain(saved_params):
      raise DistillationError(
        f"live actor term {name!r} parameters disagree with saved"
      )
    if _scale_signature(live.scale) != _scale_signature(saved.scale):
      raise DistillationError(f"live actor term {name!r} scale disagrees with saved")
    checks = {
      "delay_min_lag": live.delay_min_lag,
      "delay_max_lag": live.delay_max_lag,
      "delay_per_env": live.delay_per_env,
      "delay_hold_prob": live.delay_hold_prob,
      "delay_update_period": live.delay_update_period,
      "delay_per_env_phase": live.delay_per_env_phase,
      "delay_group": live.delay_group,
      "history_length": live.history_length,
      "flatten_history_dim": live.flatten_history_dim,
      "clip": live.clip,
    }
    expected_checks = {
      key: raw_saved.get(key, getattr(saved, key, None)) for key in checks
    }
    for field, actual in checks.items():
      if _plain(actual) != _plain(expected_checks[field]):
        raise DistillationError(
          f"live actor term {name!r} {field}={actual!r} disagrees with "
          f"saved {expected_checks[field]!r}"
        )
    noise = _live_noise_signature(live.noise)
    saved_noise = (
      None if saved.noise is None else (saved.noise.operation, saved.noise.params)
    )
    if noise is not None and saved_noise is not None:
      if noise[0] != saved_noise[0] or any(
        not _tensor_equal(noise[1].get(key), saved_noise[1].get(key))
        for key in set(noise[1]) | set(saved_noise[1])
      ):
        raise DistillationError(f"live actor term {name!r} noise disagrees with saved")
    elif noise != saved_noise:
      raise DistillationError(
        f"live actor term {name!r} noise presence disagrees with saved"
      )


def _compiled_sensor_evidence(
  env: Any, sensor_name: str, *, expected_type: str, root_body_id: int
) -> SensorFrameEvidence:
  import mujoco

  model = env.sim.mj_model
  try:
    sensor_id = int(model.sensor(sensor_name).id)
  except (KeyError, ValueError) as exc:
    raise DistillationError(f"compiled model has no sensor {sensor_name!r}") from exc
  object_type = int(model.sensor_objtype[sensor_id])
  object_id = int(model.sensor_objid[sensor_id])
  expected_sensor_type = int(mujoco.mjtSensor.mjSENS_GYRO)
  expected_object_type = int(mujoco.mjtObj.mjOBJ_SITE)
  if (
    expected_type == "gyro"
    and int(model.sensor_type[sensor_id]) != expected_sensor_type
  ):
    raise DistillationError(f"sensor {sensor_name!r} is not a compiled gyro sensor")
  if object_type != expected_object_type:
    raise DistillationError(f"sensor {sensor_name!r} is not attached to a site")
  site_name = model.site(object_id).name
  body_id = int(model.site_bodyid[object_id])
  body_name = model.body(body_id).name
  local_quat_values = tuple(float(item) for item in model.site_quat[object_id])
  if len(local_quat_values) != 4:
    raise DistillationError(
      f"sensor {sensor_name!r} has an invalid local site rotation"
    )
  local_quat = (
    local_quat_values[0],
    local_quat_values[1],
    local_quat_values[2],
    local_quat_values[3],
  )
  if not np.isfinite(local_quat).all() or not math.isclose(
    float(np.linalg.norm(local_quat)), 1.0, abs_tol=1e-6
  ):
    raise DistillationError(
      f"sensor {sensor_name!r} has an invalid local site rotation"
    )
  if expected_type == "gyro" and body_id != root_body_id:
    raise DistillationError(
      f"gyro sensor {sensor_name!r} is attached to body {body_name!r}, "
      "not the declared root body"
    )
  if expected_type == "gyro" and not np.allclose(
    local_quat, (1.0, 0.0, 0.0, 0.0), atol=1e-6, rtol=0.0
  ):
    raise DistillationError(
      f"gyro sensor {sensor_name!r} has a non-identity local site rotation"
    )
  return SensorFrameEvidence(
    name=sensor_name,
    sensor_type=int(model.sensor_type[sensor_id]),
    object_type=object_type,
    object_name=site_name,
    site_name=site_name,
    body_name=body_name,
    local_quat_wxyz=local_quat,
    expected_type=expected_type,
    frame_verified=True,
  )


def _compare_body_reference_arrays(
  selected: Any, body_indices: tuple[int, ...]
) -> tuple[tuple[str, ...], tuple[tuple[str, float], ...]]:
  atol = 1e-5
  rtol = 1e-5
  reference_arrays = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
  try:
    npz = np.load(selected.entry.motion, allow_pickle=False)
  except (OSError, ValueError) as exc:
    raise DistillationError("could not load selected reference motion arrays") from exc
  compared: list[str] = []
  errors: list[tuple[str, float]] = []
  try:
    for name in reference_arrays:
      if name not in npz:
        raise DistillationError(f"reference NPZ has no {name!r} body array")
      embedded = selected.onnx.reference_tensor(name)
      if embedded is None:
        raise DistillationError(f"teacher ONNX has no embedded {name!r} body array")
      indexed = np.asarray(npz[name])[:, list(body_indices)]
      exported = np.asarray(embedded[1])
      if indexed.shape != exported.shape:
        raise DistillationError(
          f"indexed NPZ {name!r} shape {indexed.shape} disagrees with ONNX "
          f"shape {exported.shape}"
        )
      if not np.isfinite(indexed).all() or not np.isfinite(exported).all():
        raise DistillationError(f"non-finite values in body reference array {name!r}")
      maximum = float(np.max(np.abs(indexed - exported)))
      if not np.allclose(indexed, exported, atol=atol, rtol=rtol):
        raise DistillationError(
          f"indexed NPZ and ONNX body reference {name!r} disagree; "
          f"max_abs_error={maximum:g}, atol={atol:g}, rtol={rtol:g}"
        )
      compared.append(name)
      errors.append((name, maximum))
  finally:
    npz.close()
  return tuple(compared), tuple(errors)


def audit_live_asset(
  env: Any, cohort: CohortContract, teacher_id: str
) -> AssetFrameAudit:
  """Audit compiled frames, numeric body references, and real joint bindings."""
  command = env.command_manager.get_term("motion")
  if not isinstance(command, SegmentMotionCommand):
    raise DistillationError(
      "distillation requires SegmentMotionCommand; use the opt-in factory instead "
      "of changing the shared tracking command"
    )
  robot = env.scene[command.cfg.entity_name]
  model = env.sim.mj_model
  robot_bodies = tuple(robot.body_names)
  missing = [name for name in cohort.body_names if name not in robot_bodies]
  if missing:
    raise DistillationError(f"live robot is missing tracked bodies {missing}")
  tracked_indices = tuple(robot_bodies.index(name) for name in cohort.body_names)
  if command.cfg.body_names != cohort.body_names:
    raise DistillationError(
      f"live motion body order {command.cfg.body_names} disagrees with saved "
      f"{cohort.body_names}"
    )
  if command.cfg.anchor_body_name != cohort.anchor_body_name:
    raise DistillationError(
      f"live anchor {command.cfg.anchor_body_name!r} disagrees with saved "
      f"{cohort.anchor_body_name!r}"
    )
  root_body_id = int(robot.indexing.root_body_id)
  root_body_name = model.body(root_body_id).name
  anchor_body_id = int(
    robot.indexing.body_ids[robot_bodies.index(cohort.anchor_body_name)]
  )
  selected = cohort.teacher(teacher_id)
  sensor_evidence = tuple(
    _compiled_sensor_evidence(
      env,
      sensor.sensor_name,
      expected_type="gyro" if sensor.term == "base_ang_vel" else "declared",
      root_body_id=root_body_id,
    )
    for sensor in selected.sensors
  )
  compared, errors = _compare_body_reference_arrays(selected, tracked_indices)
  sensor_names = tuple(sensor.sensor_name for sensor in selected.sensors)
  for sensor_name in sensor_names:
    try:
      env.scene[sensor_name]
    except (KeyError, ValueError) as exc:
      raise DistillationError(
        f"live environment has no saved sensor {sensor_name!r}"
      ) from exc
  return AssetFrameAudit(
    robot_bodies=robot_bodies,
    tracked_bodies=cohort.body_names,
    tracked_body_indices=tracked_indices,
    anchor_body=cohort.anchor_body_name,
    root_frame=root_body_name,
    sensor_evidence=sensor_evidence,
    root_body_name=root_body_name,
    anchor_body_id=anchor_body_id,
    body_reference_compared=compared,
    body_reference_max_abs_error=errors,
    body_reference_atol=1e-5,
    body_reference_rtol=1e-5,
    body_reference_convention=(
      "direct NPZ body arrays indexed by actual robot body order; quaternions "
      "compared directly in MuJoCo wxyz convention"
    ),
    unresolved=(),
  )


def _metadata_float_list(metadata: Mapping[str, str], key: str) -> tuple[float, ...]:
  value = metadata.get(key)
  if value is None:
    raise DistillationError(f"teacher ONNX metadata has no {key!r}")
  try:
    values = tuple(float(item) for item in value.split(",") if item != "")
  except ValueError as exc:
    raise DistillationError(f"teacher ONNX metadata {key!r} is not numeric") from exc
  if not values or not all(math.isfinite(item) for item in values):
    raise DistillationError(f"teacher ONNX metadata {key!r} is invalid")
  return values


def _validate_live_control_parameters(
  env: Any, cohort: CohortContract, teacher_id: str, action_term: Any
) -> None:
  selected = cohort.teacher(teacher_id)
  robot = env.scene[action_term.cfg.entity_name]
  joint_names = selected.actions.joint_names
  saved_default = _metadata_float_list(selected.onnx.metadata, "default_joint_pos")
  if len(saved_default) != len(joint_names):
    raise DistillationError("saved default joint position metadata has wrong width")
  actual_default = (
    robot.data.default_joint_pos[0, action_term.target_ids].detach().cpu()
  )
  if not torch.allclose(
    actual_default,
    torch.tensor(saved_default, dtype=actual_default.dtype),
    atol=1e-3,
    rtol=1e-5,
  ):
    raise DistillationError(
      "compiled robot default joint offsets disagree with teacher"
    )
  saved_stiffness = _metadata_float_list(selected.onnx.metadata, "joint_stiffness")
  saved_damping = _metadata_float_list(selected.onnx.metadata, "joint_damping")
  if len(saved_stiffness) != len(joint_names) or len(saved_damping) != len(joint_names):
    raise DistillationError("saved joint gain metadata has wrong width")
  model = env.sim.mj_model
  actual_stiffness: list[float] = []
  actual_damping: list[float] = []
  for joint_name in joint_names:
    try:
      actuator_id = int(model.actuator(f"robot/{joint_name}").id)
    except (KeyError, ValueError) as exc:
      raise DistillationError(
        f"compiled model has no actuator for joint {joint_name!r}"
      ) from exc
    actual_stiffness.append(float(model.actuator_gainprm[actuator_id, 0]))
    actual_damping.append(float(-model.actuator_biasprm[actuator_id, 2]))
  if not np.allclose(actual_stiffness, saved_stiffness, atol=1e-3, rtol=1e-5):
    raise DistillationError("compiled actuator stiffness disagrees with teacher export")
  if not np.allclose(actual_damping, saved_damping, atol=1e-3, rtol=1e-5):
    raise DistillationError("compiled actuator damping disagrees with teacher export")


def validate_live_contract(
  env: Any,
  cohort: CohortContract,
  teacher_id: str = "tennis_000",
  *,
  task_id: str | None = None,
) -> LiveContractAudit:
  """Reject semantic saved/live mismatches before collecting any labels."""
  teacher = cohort.teacher(teacher_id)
  selected_task = task_id or cohort.manifest.base_task
  if selected_task is None:
    raise DistillationError("a registered task_id is required for live validation")
  if (
    cohort.manifest.base_task is not None and selected_task != cohort.manifest.base_task
  ):
    raise DistillationError(
      f"selected task {selected_task!r} disagrees with saved base task "
      f"{cohort.manifest.base_task!r}"
    )
  if abs(float(env.cfg.sim.mujoco.timestep) - teacher.control.sim_timestep) > 1e-9:
    raise DistillationError("live MuJoCo timestep disagrees with saved teacher")
  if int(env.cfg.decimation) != teacher.control.decimation:
    raise DistillationError("live decimation disagrees with saved teacher")
  _validate_observation_contract(env, cohort, teacher.env_config)
  command = env.command_manager.get_term("motion")
  if not isinstance(command, SegmentMotionCommand):
    raise DistillationError("live motion command is not segment-aware")
  if abs(float(command.motion.fps) - teacher.reference.fps) > 1e-5:
    raise DistillationError("live motion FPS disagrees with saved teacher")
  if str(command.cfg.motion_file) != str(teacher.entry.motion):
    raise DistillationError("live motion file differs from selected teacher artifact")

  action_term = env.action_manager.get_term(teacher.actions.term)
  target_names = tuple(action_term.target_names)
  if target_names != teacher.actions.joint_names:
    raise DistillationError(
      f"live action joints {target_names} disagree with saved teacher order "
      f"{teacher.actions.joint_names}"
    )
  if action_term.action_dim != teacher.actions.dim:
    raise DistillationError("live action dimension disagrees with saved teacher")
  if getattr(action_term.cfg, "clip", None) is not None:
    raise DistillationError(
      "live action clipping is unsupported by saved teacher contract"
    )
  if not _action_value_matches(action_term.scale, teacher.actions.joint_scales):
    raise DistillationError("live action scale disagrees with saved teacher")
  use_default_offset = bool(getattr(action_term.cfg, "use_default_offset", False))
  if use_default_offset != teacher.actions.uses_default_offset:
    raise DistillationError(
      "live action default-offset setting disagrees with saved teacher"
    )
  if not use_default_offset and not _action_value_matches(
    action_term.offset, (teacher.actions.offset,)
  ):
    raise DistillationError("live action offset disagrees with saved teacher")
  asset = audit_live_asset(env, cohort, teacher_id)
  _validate_live_control_parameters(env, cohort, teacher_id, action_term)
  return LiveContractAudit(
    teacher_id=teacher_id,
    task_id=selected_task,
    joint_names=teacher.actions.joint_names,
    control_period_s=teacher.control.control_period_s,
    actor_terms=cohort.observations.names,
    asset=asset,
    additional_gravity_policy=(
      "projected gravity is captured once from robot root_link orientation and "
      "gravity_vec_w; no additional noise, delay, or history is applied"
    ),
    semantic_overrides=tuple(getattr(env.cfg, "_distillation_semantic_overrides", ())),
    seed_provenance=getattr(env.cfg, "_distillation_seed_provenance", None),
  )


def _action_value_matches(value: Any, expected: tuple[float, ...]) -> bool:
  actual = torch.as_tensor(value, dtype=torch.float64).detach().cpu()
  if actual.ndim == 2:
    actual = actual[0]
  if actual.ndim == 0:
    return len(set(expected)) == 1 and bool(
      torch.isclose(actual, torch.tensor(expected[0]))
    )
  if len(expected) == 1:
    return bool(torch.allclose(actual, torch.tensor(expected, dtype=torch.float64)))
  return actual.shape == (len(expected),) and bool(
    torch.allclose(
      actual, torch.tensor(expected, dtype=torch.float64), atol=1e-6, rtol=0.0
    )
  )


def _capture_physical_metrics(command: SegmentMotionCommand) -> PhysicalTrackingMetrics:
  reference_pos = command.body_pos_w.detach()
  actual_pos = command.robot_body_pos_w.detach()
  root_relative_pose = (reference_pos - reference_pos[:, :1]) - (
    actual_pos - actual_pos[:, :1]
  )
  reference_root_quat = command.body_quat_w[:, 0].detach()
  actual_root_quat = command.robot_body_quat_w[:, 0].detach()
  _, _, reference_yaw = euler_xyz_from_quat(reference_root_quat)
  _, _, actual_yaw = euler_xyz_from_quat(actual_root_quat)
  yaw_delta = torch.atan2(
    torch.sin(reference_yaw - actual_yaw), torch.cos(reference_yaw - actual_yaw)
  ).abs()
  return PhysicalTrackingMetrics(
    aligned_time="post_command_update_post_sim_sense_observation_cache",
    global_body_pose_error=torch.linalg.vector_norm(reference_pos - actual_pos, dim=-1)
    .mean(dim=-1)
    .clone(),
    root_relative_pose_error=torch.linalg.vector_norm(root_relative_pose, dim=-1)
    .mean(dim=-1)
    .clone(),
    root_relative_yaw_error=yaw_delta.clone(),
    heading_error=yaw_delta.clone(),
    anchor_position_error=torch.linalg.vector_norm(
      command.anchor_pos_w.detach() - command.robot_anchor_pos_w.detach(), dim=-1
    ).clone(),
  )


def _term_slices(cohort: CohortContract) -> dict[str, slice]:
  offset = 0
  result: dict[str, slice] = {}
  for term in cohort.observations.terms:
    result[term.name] = slice(offset, offset + term.width)
    offset += term.width
  return result


class DistillationEnvironmentAdapter:
  """One selected frozen teacher and one live vector environment."""

  def __init__(
    self,
    env: Any,
    cohort: CohortContract,
    teacher: FrozenTeacher,
    teacher_id: str = "tennis_000",
    *,
    schema: VaeSchema | None = None,
    audit: LiveContractAudit | None = None,
  ) -> None:
    self.env = env
    self.cohort = cohort
    self.teacher_id = teacher_id
    self.teacher_code = 0
    self.schema = schema or make_schema(joint_order=cohort.actions.joint_names)
    if self.schema.joint_order != cohort.actions.joint_names:
      raise DistillationError(
        "live schema joint_order must exactly match cohort.actions.joint_names"
      )
    self.audit = audit or validate_live_contract(env, cohort, teacher_id)
    self.teacher = teacher
    self._slices = _term_slices(cohort)
    self._motion_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  @classmethod
  def from_artifacts(
    cls,
    env: Any,
    cohort: CohortContract,
    teacher_id: str = "tennis_000",
    *,
    schema: VaeSchema | None = None,
    audit: LiveContractAudit | None = None,
  ) -> DistillationEnvironmentAdapter:
    selected = cohort.teacher(teacher_id)
    teacher = load_frozen_teacher(
      teacher_id, selected.entry.checkpoint, selected.actor, device=env.device
    )
    return cls(env, cohort, teacher, teacher_id, schema=schema, audit=audit)

  def snapshot(self) -> DistillationSnapshot:
    """Capture the manager's cached actor observation without recomputation."""
    obs = self.env.get_observations()
    if not isinstance(obs, dict) or not isinstance(obs.get("actor"), torch.Tensor):
      raise DistillationError("live actor observations must be a cached flat tensor")
    teacher_observation = obs["actor"].detach().clone()
    if teacher_observation.ndim != 2:
      raise DistillationError("live actor observation must have shape [B, 164]")
    values = {
      name: teacher_observation[:, span].clone() for name, span in self._slices.items()
    }
    required = _TERM_FEATURES - set(values)
    if required:
      raise DistillationError(
        f"saved teacher observation lacks terms {sorted(required)}"
      )
    command = self.env.command_manager.get_term("motion")
    if not isinstance(command, SegmentMotionCommand):
      raise DistillationError("live motion command is not segment-aware")
    robot = self.env.scene[command.cfg.entity_name]
    try:
      gravity = robot.data.projected_gravity_b.detach().clone()
    except AttributeError as exc:
      raise DistillationError("robot asset has no root projected-gravity data") from exc
    features = ObservationSnapshot(
      reference_q=values["command"][:, :31],
      reference_dq=values["command"][:, 31:62],
      anchor_orientation_error=values["motion_anchor_ori_b"],
      projected_gravity=gravity,
      gyro=values["base_ang_vel"],
      relative_joint_q=values["joint_pos"],
      joint_dq=values["joint_vel"],
      previous_action=values["actions"],
    )
    packed = pack_observations(features, self.schema)
    frame = command.time_steps.detach().clone()
    segment = command.segment_ids.detach().clone()
    generation = command.generation_ids.detach().clone()
    metrics = _capture_physical_metrics(command)
    return DistillationSnapshot(
      teacher_observation=teacher_observation,
      features=features,
      packed=packed,
      teacher_id=self.teacher_id,
      teacher_code=self.teacher_code,
      motion_id=self._motion_id.detach().clone(),
      reference_frame=frame,
      segment_id=segment,
      generation_id=generation,
      metrics=metrics,
    )

  def reset(self, seed: int | None = None) -> DistillationSnapshot:
    """Reset the simulator and return the owned post-reset snapshot."""
    self.env.reset(seed=seed)
    command = self.env.command_manager.get_term("motion")
    events = (
      command.consume_boundary_events()
      if isinstance(command, SegmentMotionCommand)
      else None
    )
    return replace(self.snapshot(), boundary_events=events)

  def step(self, action: torch.Tensor) -> DistillationStep:
    """Execute one normalized action and return the post-step snapshot."""
    if not isinstance(action, torch.Tensor) or action.shape != (
      self.env.num_envs,
      self.cohort.actions.dim,
    ):
      raise ValueError(
        f"action must have shape [{self.env.num_envs}, {self.cohort.actions.dim}]"
      )
    if not torch.isfinite(action).all().item():
      raise ValueError("refusing to step a non-finite action")
    before = self.snapshot()
    _, reward, terminated, time_outs, extras = self.env.step(action)
    command = self.env.command_manager.get_term("motion")
    events = (
      command.consume_boundary_events(before.generation_id)
      if isinstance(command, SegmentMotionCommand)
      else None
    )
    if events is not None:
      events = events.with_step_outcome(terminated)
    return DistillationStep(
      snapshot=self.snapshot(),
      reward=reward.detach().clone(),
      terminated=terminated.detach().clone(),
      time_outs=time_outs.detach().clone(),
      extras=dict(extras),
      events=events,
    )

  def close(self) -> None:
    self.env.close()


def make_distillation_adapter(
  cohort: CohortContract,
  teacher_id: str = "tennis_000",
  *,
  task_id: str | None = None,
  num_envs: int | None = None,
  device: str = "cpu",
  render_mode: str | None = None,
  schema: VaeSchema | None = None,
  seed: int | None = None,
) -> DistillationEnvironmentAdapter:
  """Resolve one teacher, build the trusted opt-in env, validate, and adapt it."""
  env = build_distillation_environment(
    cohort,
    teacher_id,
    task_id=task_id,
    num_envs=num_envs,
    device=device,
    render_mode=render_mode,
    seed=seed,
  )
  try:
    audit = validate_live_contract(env, cohort, teacher_id, task_id=task_id)
    return DistillationEnvironmentAdapter.from_artifacts(
      env, cohort, teacher_id, schema=schema, audit=audit
    )
  except Exception:
    env.close()
    raise


__all__ = [
  "AssetFrameAudit",
  "DistillationEnvironmentAdapter",
  "DistillationSnapshot",
  "DistillationStep",
  "LiveContractAudit",
  "audit_live_asset",
  "make_distillation_adapter",
  "validate_live_contract",
]
