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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from mjlab.tasks.tracking.distillation.config import (
  CohortContract,
  DistillationError,
  ResolvedTeacher,
)
from mjlab.tasks.tracking.distillation.environment import (
  ReferenceBoundaryEvents,
  RuntimeSeedProvenance,
  SegmentMotionCommand,
  build_distillation_environment,
  build_multi_motion_environment,
)
from mjlab.tasks.tracking.distillation.motion_library import MotionClip, MotionLibrary
from mjlab.tasks.tracking.distillation.multi_motion import (
  MotionSlotAllocation,
  MultiMotionCommand,
  PhasePolicy,
)
from mjlab.tasks.tracking.distillation.observations import (
  ObservationSnapshot,
  PackedObservationBatch,
  pack_observations,
)
from mjlab.tasks.tracking.distillation.teachers import (
  FrozenTeacher,
  TeacherBank,
  build_cohort_teacher_bank,
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
class ClipReferenceEvidence:
  """Saved/live reference evidence for one clip of a multi-motion cohort."""

  motion_id: int
  teacher_id: str
  teacher_code: int
  motion_file: str
  frames: int
  fps: float
  tracked_body_indices: tuple[int, ...]
  body_reference_compared: tuple[str, ...]
  body_reference_max_abs_error: tuple[tuple[str, float], ...]
  body_reference_atol: float
  body_reference_rtol: float
  body_reference_convention: str


@dataclass(frozen=True, slots=True)
class MultiMotionAssetAudit:
  """Compiled asset evidence shared by every row, plus per-clip references.

  One selected motion never stands in for the cohort: ``clips`` holds the
  correspondence evidence of every selected clip, and the shared fields are the
  compiled robot facts the mixed rows have in common.
  """

  robot_bodies: tuple[str, ...]
  tracked_bodies: tuple[str, ...]
  tracked_body_indices: tuple[int, ...]
  anchor_body: str
  root_frame: str
  root_body_name: str
  anchor_body_id: int
  clips: tuple[ClipReferenceEvidence, ...]
  sensor_evidence: tuple[SensorFrameEvidence, ...]
  unresolved: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MultiMotionLiveContractAudit:
  """Saved/live checks for one mixed-cohort environment, clip by clip."""

  task_id: str
  teacher_ids: tuple[str, ...]
  joint_names: tuple[str, ...]
  control_period_s: float
  actor_terms: tuple[str, ...]
  asset: MultiMotionAssetAudit
  slots: MotionSlotAllocation
  mapping_digest: str
  phase_policy: PhasePolicy
  additional_gravity_policy: str
  semantic_overrides: tuple[str, ...] = ()
  seed_provenance: RuntimeSeedProvenance | None = None


@dataclass(frozen=True, slots=True)
class DistillationSnapshot:
  """Owned, aligned tensors captured at one post-observation-manager instant.

  ``teacher_codes`` is the owned per-row routing vector aligned with
  ``motion_id``/``reference_frame``/``segment_id``/``generation_id``.  The
  legacy scalar ``teacher_code`` stays for single-teacher callers: a mixed
  batch reports ``teacher_code == -1`` and must be routed row-wise through a
  ``TeacherBank`` with the vector, never through the scalar.
  """

  teacher_observation: torch.Tensor
  features: ObservationSnapshot
  packed: PackedObservationBatch
  teacher_id: str
  teacher_code: int
  motion_id: torch.Tensor
  reference_frame: torch.Tensor
  segment_id: torch.Tensor
  generation_id: torch.Tensor
  teacher_codes: torch.Tensor | None = None
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
    if isinstance(self.teacher_code, bool) or not isinstance(self.teacher_code, int):
      raise ValueError(
        "teacher_code must be an integer; -1 marks a batch with no single teacher"
      )
    codes = self.teacher_codes
    if codes is None:
      return
    if not isinstance(codes, torch.Tensor):
      raise ValueError("teacher_codes must be a torch.Tensor or None")
    if codes.shape != (batch,):
      raise ValueError(
        f"teacher_codes must have shape [{batch}], got {tuple(codes.shape)}"
      )
    if codes.dtype not in (torch.int32, torch.int64, torch.uint8, torch.int16):
      raise ValueError(f"teacher_codes must be an integer tensor, got {codes.dtype}")
    if codes.numel() and bool((codes < 0).any()):
      raise ValueError("teacher_codes must be non-negative")
    if self.teacher_code >= 0 and not bool((codes == self.teacher_code).all()):
      raise ValueError(
        f"teacher_codes disagree with the scalar teacher_code {self.teacher_code}: a "
        "mixed batch reports teacher_code=-1 and is routed by a TeacherBank"
      )


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

_BODY_REFERENCE_ARRAYS = (
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)
_BODY_REFERENCE_ATOL = 1e-5
_BODY_REFERENCE_RTOL = 1e-5
_MIXED_TEACHER_ID = "multiple"
"""Scalar ``teacher_id`` of a mixed batch, which has no single teacher."""
_MIXED_TEACHER_CODE = -1
"""Scalar ``teacher_code`` of a mixed batch; real teacher codes are non-negative."""


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
  atol = _BODY_REFERENCE_ATOL
  rtol = _BODY_REFERENCE_RTOL
  reference_arrays = _BODY_REFERENCE_ARRAYS
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
  if isinstance(command, MultiMotionCommand):
    raise DistillationError(
      "a mixed-slot environment cannot be audited for one selected motion; use "
      "validate_multi_motion_live_contract instead"
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


def _require_multi_motion_command(env: Any) -> MultiMotionCommand:
  command = env.command_manager.get_term("motion")
  if not isinstance(command, MultiMotionCommand):
    raise DistillationError(
      "multi-motion distillation requires MultiMotionCommand; build the "
      "environment with build_multi_motion_environment instead of changing a "
      "registered task"
    )
  return command


def _selected_teachers(
  cohort: CohortContract, teacher_ids: Sequence[str]
) -> tuple[ResolvedTeacher, ...]:
  """Selected teachers in manifest order, rejecting unknown or repeated ids.

  Manifest order is what the multi-motion library uses for numeric motion ids,
  so selection order can never reinterpret an id that a checkpoint recorded.
  """
  requested = tuple(teacher_ids)
  if not requested:
    raise DistillationError("multi-motion validation needs at least one teacher id")
  if len(set(requested)) != len(requested):
    raise DistillationError(f"duplicate teacher ids {requested}")
  known = {teacher.id for teacher in cohort.teachers}
  unknown = [teacher_id for teacher_id in requested if teacher_id not in known]
  if unknown:
    raise DistillationError(f"cohort has no teacher(s) {unknown}")
  wanted = set(requested)
  return tuple(teacher for teacher in cohort.teachers if teacher.id in wanted)


def _require_clip_teacher_mapping(
  cohort: CohortContract, library: MotionLibrary
) -> None:
  """Reject a clip table whose code/teacher pairing contradicts the cohort.

  A clip's numeric teacher code decides which frozen teacher labels its rows, so
  a code that the cohort assigns to a different teacher would silently train on
  the wrong labels.
  """
  positions = tuple(clip.motion_id for clip in library.clips)
  if positions != tuple(range(library.num_motions)):
    raise DistillationError(
      f"library motion ids {positions} must be their position in the clip order"
    )
  for clip in library.clips:
    if not 0 <= clip.teacher_code < len(cohort.teachers):
      raise DistillationError(
        f"clip {clip.teacher_id!r} carries teacher code {clip.teacher_code}, which "
        f"is outside the {len(cohort.teachers)} cohort teachers"
      )
    assigned = cohort.teachers[clip.teacher_code].id
    if assigned != clip.teacher_id:
      raise DistillationError(
        f"clip {clip.teacher_id!r} uses teacher code {clip.teacher_code}, but the "
        f"cohort assigns that code to {assigned!r}: the motion-to-teacher mapping "
        "would label rows with the wrong frozen teacher"
      )


def _compare_clip_body_references(
  command: MultiMotionCommand, clip: MotionClip, teacher: ResolvedTeacher
) -> tuple[tuple[str, ...], tuple[tuple[str, float], ...]]:
  """Compare every frame of one library clip with its teacher export.

  Both directions are compared with the milestone tolerance: the library's own
  owned, tracked reference tensors using element-wise ``allclose``, and the saved
  NPZ file itself (see :func:`_compare_body_reference_arrays`), so a library that
  loaded the right bytes but indexed the wrong bodies is caught as well.
  """
  library = command.library
  frames = torch.arange(clip.frames, dtype=torch.long, device=library.device)
  ids = torch.full(
    (clip.frames,), clip.motion_id, dtype=torch.long, device=library.device
  )
  compared: list[str] = []
  errors: list[tuple[str, float]] = []
  for name in _BODY_REFERENCE_ARRAYS:
    embedded = teacher.onnx.reference_tensor(name)
    if embedded is None:
      raise DistillationError(f"teacher ONNX has no embedded {name!r} body array")
    expected = np.asarray(embedded[1], dtype=np.float64)
    actual = (
      getattr(library, name)(ids, frames).detach().to(torch.float64).cpu().numpy()
    )
    if actual.shape != expected.shape:
      raise DistillationError(
        f"clip {clip.teacher_id!r} {name!r} shape {actual.shape} disagrees with the "
        f"export shape {expected.shape}"
      )
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
      raise DistillationError(
        f"non-finite values in body reference array {name!r} of {clip.teacher_id!r}"
      )
    maximum = float(np.max(np.abs(actual - expected))) if expected.size else 0.0
    if not np.allclose(
      actual, expected, atol=_BODY_REFERENCE_ATOL, rtol=_BODY_REFERENCE_RTOL
    ):
      raise DistillationError(
        f"clip {clip.teacher_id!r} {name!r} disagrees with its teacher export; "
        f"max_abs_error={maximum:g}, atol={_BODY_REFERENCE_ATOL:g}, "
        f"rtol={_BODY_REFERENCE_RTOL:g}"
      )
    compared.append(name)
    errors.append((name, maximum))
  return tuple(compared), tuple(errors)


def audit_multi_motion_live_asset(
  env: Any, cohort: CohortContract, teacher_ids: Sequence[str]
) -> MultiMotionAssetAudit:
  """Audit the compiled robot and every selected clip's reference evidence."""
  selected = _selected_teachers(cohort, teacher_ids)
  command = _require_multi_motion_command(env)
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
  expected_ids = tuple(teacher.id for teacher in selected)
  clip_ids = tuple(clip.teacher_id for clip in command.library.clips)
  if clip_ids != expected_ids:
    raise DistillationError(
      f"live library clips {clip_ids} disagree with the selected teachers "
      f"{expected_ids}"
    )
  _require_clip_teacher_mapping(cohort, command.library)
  if command.library.source_body_count != len(robot_bodies):
    raise DistillationError(
      f"reference clips carry {command.library.source_body_count} source bodies but "
      f"the compiled robot has {len(robot_bodies)}: the clip body axis must be the "
      "compiled robot body order"
    )
  selection = command.library.body_selection
  if selection is None or tuple(selection.indices) != tracked_indices:
    raise DistillationError(
      "the library's resolved tracked-body indices disagree with the compiled "
      "robot's body order for the saved tracked bodies"
    )
  if selection.names is not None and tuple(selection.names) != cohort.body_names:
    raise DistillationError(
      f"the library's tracked-body names {tuple(selection.names)} disagree with the "
      f"saved {cohort.body_names}"
    )
  root_body_id = int(robot.indexing.root_body_id)
  root_body_name = model.body(root_body_id).name
  anchor_body_id = int(
    robot.indexing.body_ids[robot_bodies.index(cohort.anchor_body_name)]
  )
  sensor_evidence: dict[str, SensorFrameEvidence] = {}
  for teacher in selected:
    for sensor in teacher.sensors:
      evidence = _compiled_sensor_evidence(
        env,
        sensor.sensor_name,
        expected_type="gyro" if sensor.term == "base_ang_vel" else "declared",
        root_body_id=root_body_id,
      )
      previous = sensor_evidence.get(sensor.sensor_name)
      if previous is not None and previous != evidence:
        raise DistillationError(
          f"teachers disagree about compiled sensor {sensor.sensor_name!r}"
        )
      sensor_evidence[sensor.sensor_name] = evidence
      try:
        env.scene[sensor.sensor_name]
      except (KeyError, ValueError) as exc:
        raise DistillationError(
          f"live environment has no saved sensor {sensor.sensor_name!r}"
        ) from exc
  clips: list[ClipReferenceEvidence] = []
  for clip in command.library.clips:
    teacher = cohort.teacher(clip.teacher_id)
    if str(clip.motion_file) != str(teacher.entry.motion):
      raise DistillationError(
        f"clip {clip.teacher_id!r} loads {clip.motion_file} but its teacher artifact "
        f"is {teacher.entry.motion}"
      )
    if clip.frames != teacher.reference.frames:
      raise DistillationError(
        f"clip {clip.teacher_id!r} has {clip.frames} frames but the teacher declares "
        f"{teacher.reference.frames}"
      )
    if not math.isclose(clip.fps, teacher.reference.fps, rel_tol=1e-5, abs_tol=1e-5):
      raise DistillationError(
        f"clip {clip.teacher_id!r} is {clip.fps:g} Hz but the teacher declares "
        f"{teacher.reference.fps:g} Hz"
      )
    compared, errors = _compare_clip_body_references(command, clip, teacher)
    _compare_body_reference_arrays(teacher, tracked_indices)
    clips.append(
      ClipReferenceEvidence(
        motion_id=clip.motion_id,
        teacher_id=clip.teacher_id,
        teacher_code=clip.teacher_code,
        motion_file=str(clip.motion_file),
        frames=clip.frames,
        fps=clip.fps,
        tracked_body_indices=(
          tracked_indices if clip.body_indices is None else tuple(clip.body_indices)
        ),
        body_reference_compared=compared,
        body_reference_max_abs_error=errors,
        body_reference_atol=_BODY_REFERENCE_ATOL,
        body_reference_rtol=_BODY_REFERENCE_RTOL,
        body_reference_convention=(
          "library-owned tracked reference tensors for every clip frame compared "
          "with the teacher export's embedded arrays in tracked-body order"
        ),
      )
    )
  return MultiMotionAssetAudit(
    robot_bodies=robot_bodies,
    tracked_bodies=cohort.body_names,
    tracked_body_indices=tracked_indices,
    anchor_body=cohort.anchor_body_name,
    root_frame=root_body_name,
    root_body_name=root_body_name,
    anchor_body_id=anchor_body_id,
    clips=tuple(clips),
    sensor_evidence=tuple(sensor_evidence.values()),
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


def _validate_live_action_term(teacher: ResolvedTeacher, action_term: Any) -> None:
  """Reject a live action term that disagrees with a saved teacher contract."""
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


def validate_multi_motion_live_contract(
  env: Any,
  cohort: CohortContract,
  teacher_ids: Sequence[str],
  *,
  task_id: str | None = None,
) -> MultiMotionLiveContractAudit:
  """Reject saved/live mismatches before a mixed-slot collector labels rows.

  Every selected teacher's saved contract is checked against the same live
  environment, and every selected clip's reference is compared with its own
  teacher export, so one selected motion never stands in for the cohort.  The
  returned evidence keeps the realized slot allocation and the library's
  ordered mapping digest that strict resume must reproduce.
  """
  selected = _selected_teachers(cohort, teacher_ids)
  first = selected[0]
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
  for teacher in selected:
    if abs(float(env.cfg.sim.mujoco.timestep) - teacher.control.sim_timestep) > 1e-9:
      raise DistillationError(
        f"live MuJoCo timestep disagrees with saved teacher {teacher.id!r}"
      )
    if int(env.cfg.decimation) != teacher.control.decimation:
      raise DistillationError(
        f"live decimation disagrees with saved teacher {teacher.id!r}"
      )
    _validate_observation_contract(env, cohort, teacher.env_config)
  command = _require_multi_motion_command(env)
  if str(command.cfg.motion_file) != "":
    raise DistillationError(
      "a multi-motion command must not carry a single motion file; each row's "
      "reference comes from its own clip"
    )
  if command.cfg.sampling_mode not in ("uniform", "start"):
    raise DistillationError(
      f"multi-motion phase sampling {command.cfg.sampling_mode!r} is not supported"
    )
  clip_ids = tuple(clip.teacher_id for clip in command.library.clips)
  if command.slot_allocation.teacher_ids != clip_ids:
    raise DistillationError(
      f"slot allocation teachers {command.slot_allocation.teacher_ids} disagree with "
      f"the library clip order {clip_ids}"
    )
  for teacher in selected:
    action_term = env.action_manager.get_term(teacher.actions.term)
    _validate_live_action_term(teacher, action_term)
    _validate_live_control_parameters(env, cohort, teacher.id, action_term)
  asset = audit_multi_motion_live_asset(env, cohort, teacher_ids)
  return MultiMotionLiveContractAudit(
    task_id=selected_task,
    teacher_ids=tuple(teacher.id for teacher in selected),
    joint_names=first.actions.joint_names,
    control_period_s=first.control.control_period_s,
    actor_terms=cohort.observations.names,
    asset=asset,
    slots=command.slot_allocation,
    mapping_digest=command.library.mapping_digest(),
    phase_policy=command.cfg.sampling_mode,
    additional_gravity_policy=(
      "projected gravity is captured once from robot root_link orientation and "
      "gravity_vec_w; no additional noise, delay, or history is applied"
    ),
    semantic_overrides=tuple(getattr(env.cfg, "_distillation_semantic_overrides", ())),
    seed_provenance=getattr(env.cfg, "_distillation_seed_provenance", None),
  )


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
  if isinstance(command, MultiMotionCommand):
    raise DistillationError(
      "a mixed-slot environment has no single motion to validate against one "
      "teacher; use validate_multi_motion_live_contract instead"
    )
  if abs(float(command.motion.fps) - teacher.reference.fps) > 1e-5:
    raise DistillationError("live motion FPS disagrees with saved teacher")
  if str(command.cfg.motion_file) != str(teacher.entry.motion):
    raise DistillationError("live motion file differs from selected teacher artifact")

  action_term = env.action_manager.get_term(teacher.actions.term)
  _validate_live_action_term(teacher, action_term)
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


def _require_normalized_action(env: Any, action_dim: int, action: torch.Tensor) -> None:
  if not isinstance(action, torch.Tensor) or action.shape != (env.num_envs, action_dim):
    raise ValueError(f"action must have shape [{env.num_envs}, {action_dim}]")
  if not torch.isfinite(action).all().item():
    raise ValueError("refusing to step a non-finite action")


def _capture_snapshot(
  env: Any,
  *,
  cohort: CohortContract,
  slices: Mapping[str, slice],
  schema: VaeSchema,
  command: SegmentMotionCommand,
  teacher_id: str,
  teacher_code: int,
  motion_id: torch.Tensor,
  teacher_codes: torch.Tensor | None,
) -> DistillationSnapshot:
  """Capture one owned snapshot from the observation manager cache.

  Features and teacher inputs are two views of the same cached actor tensor, so
  no observation term, noise draw, delay step, or history window is recomputed.
  The caller supplies the per-row identity and routing tensors, which both
  adapters read from their own command.
  """
  obs = env.get_observations()
  if not isinstance(obs, dict) or not isinstance(obs.get("actor"), torch.Tensor):
    raise DistillationError("live actor observations must be a cached flat tensor")
  teacher_observation = obs["actor"].detach().clone()
  if teacher_observation.ndim != 2:
    raise DistillationError("live actor observation must have shape [B, 164]")
  values = {name: teacher_observation[:, span].clone() for name, span in slices.items()}
  required = _TERM_FEATURES - set(values)
  if required:
    raise DistillationError(f"saved teacher observation lacks terms {sorted(required)}")
  robot = env.scene[command.cfg.entity_name]
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
  packed = pack_observations(features, schema)
  return DistillationSnapshot(
    teacher_observation=teacher_observation,
    features=features,
    packed=packed,
    teacher_id=teacher_id,
    teacher_code=teacher_code,
    motion_id=motion_id.detach().clone(),
    reference_frame=command.time_steps.detach().clone(),
    segment_id=command.segment_ids.detach().clone(),
    generation_id=command.generation_ids.detach().clone(),
    teacher_codes=None if teacher_codes is None else teacher_codes.detach().clone(),
    metrics=_capture_physical_metrics(command),
  )


def _consume_boundary_events(
  env: Any, pre_generation: torch.Tensor | None = None
) -> ReferenceBoundaryEvents | None:
  command = env.command_manager.get_term("motion")
  if not isinstance(command, SegmentMotionCommand):
    return None
  return command.consume_boundary_events(pre_generation)


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
    command = self.env.command_manager.get_term("motion")
    if not isinstance(command, SegmentMotionCommand):
      raise DistillationError("live motion command is not segment-aware")
    motion_id = self._motion_id.detach().clone()
    return _capture_snapshot(
      self.env,
      cohort=self.cohort,
      slices=self._slices,
      schema=self.schema,
      command=command,
      teacher_id=self.teacher_id,
      teacher_code=self.teacher_code,
      motion_id=motion_id,
      teacher_codes=torch.full_like(motion_id, self.teacher_code),
    )

  def reset(self, seed: int | None = None) -> DistillationSnapshot:
    """Reset the simulator and return the owned post-reset snapshot."""
    self.env.reset(seed=seed)
    return replace(self.snapshot(), boundary_events=_consume_boundary_events(self.env))

  def step(self, action: torch.Tensor) -> DistillationStep:
    """Execute one normalized action and return the post-step snapshot."""
    _require_normalized_action(self.env, self.cohort.actions.dim, action)
    before = self.snapshot()
    _, reward, terminated, time_outs, extras = self.env.step(action)
    events = _consume_boundary_events(self.env, before.generation_id)
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


class MultiMotionDistillationAdapter:
  """One mixed-slot environment whose rows keep their own reference clip.

  There is deliberately no single teacher identity: per-row routing codes come
  from the command's own clip mapping, and every code means the same teacher in
  the frozen bank, so no scalar identity is ever used to label a mixed batch.
  The observation cache, delay, noise, and action semantics are unchanged from
  the single-teacher adapter, because both capture the same cached actor tensor.
  """

  def __init__(
    self,
    env: Any,
    cohort: CohortContract,
    bank: TeacherBank,
    *,
    schema: VaeSchema | None = None,
    audit: MultiMotionLiveContractAudit | None = None,
  ) -> None:
    self.env = env
    self.cohort = cohort
    self.bank = bank
    self.schema = schema or make_schema(joint_order=cohort.actions.joint_names)
    if self.schema.joint_order != cohort.actions.joint_names:
      raise DistillationError(
        "live schema joint_order must exactly match cohort.actions.joint_names"
      )
    command = _require_multi_motion_command(env)
    self.teacher_ids = tuple(clip.teacher_id for clip in command.library.clips)
    for clip in command.library.clips:
      code = bank.code(clip.teacher_id)
      if code != clip.teacher_code:
        raise DistillationError(
          f"teacher bank assigns code {code} to {clip.teacher_id!r} but the "
          f"reference library routes it with code {clip.teacher_code}; build the "
          "bank over the whole cohort in manifest order"
        )
    self.audit = audit or validate_multi_motion_live_contract(
      env, cohort, self.teacher_ids
    )
    self._slices = _term_slices(cohort)

  @property
  def library(self) -> MotionLibrary:
    """Reference library of the adapted environment, resolved per row."""
    return _require_multi_motion_command(self.env).library

  @property
  def motion_teacher_codes(self) -> dict[int, int]:
    """Ordered motion-id to teacher-code mapping of this adapted cohort."""
    return {clip.motion_id: clip.teacher_code for clip in self.library.clips}

  def snapshot(self) -> DistillationSnapshot:
    """Capture one owned snapshot with validated per-row routing metadata."""
    command = _require_multi_motion_command(self.env)
    motion_id = command.motion_ids.detach().clone().to(dtype=torch.long)
    if motion_id.shape != (self.env.num_envs,):
      raise DistillationError(
        f"live motion ids must have shape [{self.env.num_envs}], got "
        f"{tuple(motion_id.shape)}"
      )
    teacher_codes = command.teacher_codes.detach().clone().to(dtype=torch.long)
    if teacher_codes.shape != motion_id.shape:
      raise DistillationError(
        "per-row teacher codes must be aligned with the per-row motion ids"
      )
    expected = command.library.teacher_codes_for(motion_id)
    if not torch.equal(teacher_codes, expected):
      raise DistillationError(
        "per-row teacher codes disagree with the reference library's "
        "motion-to-teacher mapping; rows would be labeled by the wrong teacher"
      )
    return _capture_snapshot(
      self.env,
      cohort=self.cohort,
      slices=self._slices,
      schema=self.schema,
      command=command,
      teacher_id=_MIXED_TEACHER_ID,
      teacher_code=_MIXED_TEACHER_CODE,
      motion_id=motion_id,
      teacher_codes=teacher_codes,
    )

  def reset(self, seed: int | None = None) -> DistillationSnapshot:
    """Reset the simulator and return the owned post-reset snapshot."""
    self.env.reset(seed=seed)
    return replace(self.snapshot(), boundary_events=_consume_boundary_events(self.env))

  def step(self, action: torch.Tensor) -> DistillationStep:
    """Execute one normalized action and return the post-step snapshot."""
    _require_normalized_action(self.env, self.cohort.actions.dim, action)
    before = self.snapshot()
    _, reward, terminated, time_outs, extras = self.env.step(action)
    events = _consume_boundary_events(self.env, before.generation_id)
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


def make_multi_teacher_distillation_adapter(
  cohort: CohortContract,
  teacher_ids: Sequence[str],
  *,
  phase_policy: PhasePolicy = "uniform",
  task_id: str | None = None,
  num_envs: int | None = None,
  device: str = "cpu",
  render_mode: str | None = None,
  schema: VaeSchema | None = None,
  seed: int | None = None,
) -> MultiMotionDistillationAdapter:
  """Build one mixed-slot env, audit every selected clip, and adapt it.

  The frozen bank is built over the whole cohort in manifest order, which is the
  same order the reference library uses for numeric teacher codes, so routing a
  code can never select a different teacher than the one that trained on the
  row's clip.
  """
  env = build_multi_motion_environment(
    cohort,
    teacher_ids,
    phase_policy=phase_policy,
    task_id=task_id,
    num_envs=num_envs,
    device=device,
    render_mode=render_mode,
    seed=seed,
  )
  try:
    bank = build_cohort_teacher_bank(cohort, device=device)
    audit = validate_multi_motion_live_contract(
      env, cohort, teacher_ids, task_id=task_id
    )
    return MultiMotionDistillationAdapter(env, cohort, bank, schema=schema, audit=audit)
  except Exception:
    env.close()
    raise


__all__ = [
  "AssetFrameAudit",
  "ClipReferenceEvidence",
  "DistillationEnvironmentAdapter",
  "DistillationSnapshot",
  "DistillationStep",
  "LiveContractAudit",
  "MultiMotionAssetAudit",
  "MultiMotionDistillationAdapter",
  "MultiMotionLiveContractAudit",
  "audit_live_asset",
  "audit_multi_motion_live_asset",
  "make_distillation_adapter",
  "make_multi_teacher_distillation_adapter",
  "validate_live_contract",
  "validate_multi_motion_live_contract",
]
