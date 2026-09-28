"""Versioned cohort identity for multi-teacher (M4) distillation.

Scope: turn the validated teacher manifest, the resolved reference library, the
pinned row-slot policy, and the per-motion replay policy into one ordered,
plain-data identity record; compare two such records strictly for resume; and
check one requested cohort member for inference.  No simulator, model, or
checkpoint file is opened here, so this contract can be built, compared, and
tested on CPU tensors alone.

The record is deliberately *data*: every field is a string, number, boolean, or
list of those, so a checkpoint stores it without any executable content and a
reader can rebuild it with :meth:`CohortIdentity.from_dict`.  Strict resume
requires the whole record to be reproduced; inference of one member is checked
against the member's content digests, which is what allows a byte-identical
relocated artifact to be accepted without weakening resume.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, get_args

from mjlab.tasks.tracking.distillation.balanced_storage import BalancedReplayBuffer
from mjlab.tasks.tracking.distillation.config import CohortContract
from mjlab.tasks.tracking.distillation.motion_library import (
  BodySelection,
  MotionLibrary,
)
from mjlab.tasks.tracking.distillation.multi_motion import (
  MotionSlotAllocation,
  PhasePolicy,
)
from mjlab.tasks.tracking.distillation.reset_policy import (
  ResetPolicy,
  ResetPolicyError,
  effective_windows,
)
from mjlab.tasks.tracking.distillation.storage import ReplayBufferProtocol

COHORT_CONTRACT_VERSION = 1
"""Schema version of the persisted cohort identity."""

RESET_PROVENANCE_VERSION = 3
"""Checkpoint contract version for enabled standing-start provenance."""

COHORT_ARTIFACT_ROLES = ("checkpoint", "motion", "env_config", "agent_config", "onnx")
"""Manifest artifact roles every cohort member must record a digest for."""

BALANCED_REPLAY_KIND = "balanced-motion-replay"
"""Replay policy kind of the per-motion partitioned buffer M4 cohorts require."""

FIFO_REPLAY_KIND = "fifo-single-partition"
"""Replay policy kind of the M3 single-partition FIFO buffer."""

PHASE_POLICIES: tuple[PhasePolicy, ...] = get_args(PhasePolicy)
"""Phase-sampling policies a cohort identity may record."""


class CohortContractError(ValueError):
  """A cohort identity is malformed, incomplete, or does not match."""


# Plain-data readers.  Every stored field is rebuilt and type-checked, so a
# truncated or hand-edited record fails here instead of silently degrading the
# comparison it feeds.


def _require_keys(payload: object, required: Sequence[str], where: str) -> Mapping:
  if not isinstance(payload, Mapping):
    raise CohortContractError(f"{where} must be a mapping")
  keys = [key for key in payload if isinstance(key, str)]
  if len(keys) != len(payload):
    raise CohortContractError(f"{where} must use string keys")
  missing = [key for key in required if key not in payload]
  extra = [key for key in keys if key not in required]
  if missing or extra:
    raise CohortContractError(
      f"{where} has missing or unknown fields: missing {sorted(missing)}, "
      f"unknown {sorted(extra)}"
    )
  return payload


def _require_str(value: object, where: str) -> str:
  if not isinstance(value, str) or not value:
    raise CohortContractError(f"{where} must be a non-empty string")
  return value


def _require_optional_str(value: object, where: str) -> str | None:
  if value is None:
    return None
  return _require_str(value, where)


def _require_int(value: object, where: str, *, minimum: int | None = None) -> int:
  if isinstance(value, bool) or not isinstance(value, int):
    raise CohortContractError(f"{where} must be an integer")
  if minimum is not None and value < minimum:
    raise CohortContractError(f"{where} must be at least {minimum}")
  return value


def _require_optional_int(value: object, where: str) -> int | None:
  if value is None:
    return None
  return _require_int(value, where)


def _require_bool(value: object, where: str) -> bool:
  if not isinstance(value, bool):
    raise CohortContractError(f"{where} must be a boolean")
  return value


def _require_str_tuple(value: object, where: str) -> tuple[str, ...]:
  if not isinstance(value, (list, tuple)):
    raise CohortContractError(f"{where} must be a list of strings")
  return tuple(
    _require_str(item, f"{where}[{index}]") for index, item in enumerate(value)
  )


def _require_int_tuple(
  value: object, where: str, *, minimum: int | None = None
) -> tuple[int, ...]:
  if not isinstance(value, (list, tuple)):
    raise CohortContractError(f"{where} must be a list of integers")
  return tuple(
    _require_int(item, f"{where}[{index}]", minimum=minimum)
    for index, item in enumerate(value)
  )


def _require_float(value: object, where: str, *, positive: bool = False) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise CohortContractError(f"{where} must be a number")
  number = float(value)
  if not math.isfinite(number):
    raise CohortContractError(f"{where} must be finite")
  if positive and number <= 0.0:
    raise CohortContractError(f"{where} must be positive")
  return number


def _require_float_tuple(value: object, where: str) -> tuple[float, ...]:
  if not isinstance(value, (list, tuple)):
    raise CohortContractError(f"{where} must be a list of numbers")
  return tuple(
    _require_float(item, f"{where}[{index}]") for index, item in enumerate(value)
  )


def _plain_reset_record(value: object, where: str = "reset record") -> dict[str, Any]:
  """Copy nested JSON-like metadata while rejecting executable values."""
  if not isinstance(value, Mapping):
    raise CohortContractError(f"{where} must be a mapping")
  result: dict[str, Any] = {}
  for key, item in value.items():
    if not isinstance(key, str):
      raise CohortContractError(f"{where} keys must be strings")
    if isinstance(item, Mapping):
      result[key] = _plain_reset_record(item, f"{where}.{key}")
    elif isinstance(item, (list, tuple)):
      result[key] = [
        _plain_reset_record(part, f"{where}.{key}")
        if isinstance(part, Mapping)
        else part
        for part in item
      ]
    elif item is None or isinstance(item, (str, int, float, bool)):
      if isinstance(item, float) and not math.isfinite(item):
        raise CohortContractError(f"{where}.{key} must be finite")
      result[key] = item
    else:
      raise CohortContractError(f"{where}.{key} contains unsupported {type(item)!r}")
  return result


@dataclass(frozen=True, slots=True)
class ResetProvenance:
  """Canonical resolved standing-reset contract persisted by v3 checkpoints."""

  reset_policy: ResetPolicy
  effective_windows: tuple[int, ...]
  boundary_semantics: dict[str, str]
  standing_pose: dict[str, Any]
  perturbations: dict[str, Any]
  provenance: dict[str, str]
  replay_layout: str = "provenance-v1"

  def __post_init__(self) -> None:
    if not isinstance(self.reset_policy, ResetPolicy) or not self.reset_policy.enabled:
      raise CohortContractError(
        "v3 reset provenance requires an enabled standing-mixture ResetPolicy"
      )
    if any(
      isinstance(value, bool) or not isinstance(value, int) or value <= 0
      for value in self.effective_windows
    ):
      raise CohortContractError("effective reset windows must be positive integers")
    if self.replay_layout != "provenance-v1":
      raise CohortContractError(
        "standing reset checkpoints require the provenance-v1 replay layout"
      )
    _plain_reset_record(self.standing_pose, "standing_pose")
    _plain_reset_record(self.perturbations, "perturbations")
    _plain_reset_record(self.as_dict(), "reset_provenance")

  @property
  def version(self) -> int:
    return RESET_PROVENANCE_VERSION

  def as_dict(self) -> dict[str, Any]:
    return {
      "version": RESET_PROVENANCE_VERSION,
      "reset_policy": self.reset_policy.as_dict(),
      "effective_windows": list(self.effective_windows),
      "boundary_semantics": dict(self.boundary_semantics),
      "standing_pose": dict(self.standing_pose),
      "perturbations": dict(self.perturbations),
      "provenance": dict(self.provenance),
      "replay_layout": self.replay_layout,
    }

  def digest(self) -> str:
    return hashlib.sha256(canonical_json(self.as_dict()).encode("utf-8")).hexdigest()

  @classmethod
  def from_dict(
    cls, payload: object, where: str = "reset_provenance"
  ) -> "ResetProvenance":
    raw = _require_keys(
      payload,
      (
        "version",
        "reset_policy",
        "effective_windows",
        "boundary_semantics",
        "standing_pose",
        "perturbations",
        "provenance",
        "replay_layout",
      ),
      where,
    )
    version = _require_int(raw["version"], f"{where}.version", minimum=1)
    if version != RESET_PROVENANCE_VERSION:
      raise CohortContractError(
        f"{where} version {version} is not supported; expected {RESET_PROVENANCE_VERSION}"
      )
    try:
      policy = ResetPolicy.from_dict(raw["reset_policy"])
    except ResetPolicyError as exc:
      raise CohortContractError(f"{where}.reset_policy is invalid: {exc}") from exc
    return cls(
      policy,
      _require_int_tuple(
        raw["effective_windows"], f"{where}.effective_windows", minimum=1
      ),
      _require_str_mapping(raw["boundary_semantics"], f"{where}.boundary_semantics"),
      _plain_reset_record(raw["standing_pose"], f"{where}.standing_pose"),
      _plain_reset_record(raw["perturbations"], f"{where}.perturbations"),
      _require_str_mapping(raw["provenance"], f"{where}.provenance"),
      _require_str(raw["replay_layout"], f"{where}.replay_layout"),
    )


def reset_provenance_from_adapter(adapter: Any) -> ResetProvenance:
  """Resolve the command's standing contract into canonical plain data."""
  command = adapter.env.command_manager.get_term("motion")
  policy = getattr(command, "reset_policy", None)
  if not isinstance(policy, ResetPolicy) or not policy.enabled:
    raise CohortContractError(
      "the adapted command does not have enabled standing resets"
    )
  lengths = tuple(int(clip.frames) for clip in command.library.clips)
  windows = tuple(int(value) for value in effective_windows(policy, lengths).tolist())
  cfg = command.cfg
  robot = command.robot
  default_joints = robot.data.default_joint_pos[0].detach().cpu().flatten().tolist()
  default_height = float(robot.data.default_root_state[0, 2].detach().cpu().item())

  def _ranges(name: str) -> list[float]:
    value = getattr(cfg, name)
    if isinstance(value, Mapping):
      keys = ("x", "y", "z", "roll", "pitch", "yaw")
      return [float(item) for key in keys for item in value.get(key, (0.0, 0.0))]
    return [float(item) for item in value]

  return ResetProvenance(
    reset_policy=policy,
    effective_windows=windows,
    boundary_semantics={
      "full_reset": "sample standing/reference once per affected row",
      "timer_resample": "reference-state teleport; no standing mixture",
      "reference_wrap": "reference-state teleport; no standing mixture",
      "reset_to_frame": "explicit reference-state teleport; no standing mixture",
    },
    standing_pose={
      "joint_position": [float(item) for item in default_joints],
      "root_height": default_height,
      "yaw_alignment": "upright yaw from selected reference anchor quaternion",
      "joint_order": list(
        getattr(cfg, "joint_names", getattr(robot.data, "joint_names", ()))
      ),
    },
    perturbations={
      "pose_range": _ranges("pose_range"),
      "velocity_range": _ranges("velocity_range"),
      "joint_position_range": [float(item) for item in cfg.joint_position_range],
    },
    provenance={
      "command_type": type(command).__qualname__,
      "entity_name": str(getattr(cfg, "entity_name", "")),
      "standing_pose_source": "robot.data.default_joint_pos/default_root_state",
      "yaw_source": "reference anchor quaternion",
    },
  )


def _require_str_mapping(value: object, where: str) -> dict[str, str]:
  if not isinstance(value, Mapping):
    raise CohortContractError(f"{where} must be a mapping")
  return {
    _require_str(key, f"{where} key"): _require_str(item, f"{where}.{key}")
    for key, item in value.items()
  }


def _require_close(name: str, stored: float, expected: float) -> None:
  """Require two configuration floats to agree, with a names-only message."""
  if not math.isclose(stored, expected, rel_tol=1e-12, abs_tol=1e-12):
    raise CohortContractError(f"{name} differs: {stored!r} != {expected!r}")


def canonical_json(payload: Any) -> str:
  """Stable JSON used for every cohort digest."""
  return json.dumps(payload, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class CohortMember:
  """One ordered cohort member: its teacher, clip, and resolved body mapping."""

  motion_id: int
  teacher_id: str
  teacher_code: int
  artifacts: dict[str, str]
  hashes: dict[str, str]
  motion_path: str
  motion_source_hash: str
  frames: int
  fps: float
  tracked_body_names: tuple[str, ...]
  tracked_body_indices: tuple[int, ...]
  source_body_count: int

  def as_dict(self) -> dict[str, Any]:
    return {
      "motion_id": self.motion_id,
      "teacher_id": self.teacher_id,
      "teacher_code": self.teacher_code,
      "artifacts": dict(self.artifacts),
      "hashes": dict(self.hashes),
      "motion_path": self.motion_path,
      "motion_source_hash": self.motion_source_hash,
      "frames": self.frames,
      "fps": self.fps,
      "tracked_body_names": list(self.tracked_body_names),
      "tracked_body_indices": list(self.tracked_body_indices),
      "source_body_count": self.source_body_count,
    }

  @classmethod
  def from_dict(cls, payload: object, where: str) -> CohortMember:
    fields = (
      "motion_id",
      "teacher_id",
      "teacher_code",
      "artifacts",
      "hashes",
      "motion_path",
      "motion_source_hash",
      "frames",
      "fps",
      "tracked_body_names",
      "tracked_body_indices",
      "source_body_count",
    )
    raw = _require_keys(payload, fields, where)
    artifacts = _require_str_mapping(raw["artifacts"], f"{where}.artifacts")
    if set(artifacts) != set(COHORT_ARTIFACT_ROLES):
      raise CohortContractError(
        f"{where}.artifacts must record exactly {list(COHORT_ARTIFACT_ROLES)}"
      )
    hashes = _require_str_mapping(raw["hashes"], f"{where}.hashes")
    if set(hashes) != set(COHORT_ARTIFACT_ROLES):
      raise CohortContractError(
        f"{where}.hashes must record exactly {list(COHORT_ARTIFACT_ROLES)}"
      )
    indices = _require_int_tuple(
      raw["tracked_body_indices"], f"{where}.tracked_body_indices", minimum=0
    )
    if len(set(indices)) != len(indices):
      raise CohortContractError(f"{where}.tracked_body_indices must be unique")
    source_body_count = _require_int(
      raw["source_body_count"], f"{where}.source_body_count", minimum=1
    )
    if any(index >= source_body_count for index in indices):
      raise CohortContractError(
        f"{where}.tracked_body_indices must be inside {source_body_count} source bodies"
      )
    return cls(
      motion_id=_require_int(raw["motion_id"], f"{where}.motion_id", minimum=0),
      teacher_id=_require_str(raw["teacher_id"], f"{where}.teacher_id"),
      teacher_code=_require_int(
        raw["teacher_code"], f"{where}.teacher_code", minimum=0
      ),
      artifacts=artifacts,
      hashes=hashes,
      motion_path=_require_str(raw["motion_path"], f"{where}.motion_path"),
      motion_source_hash=_require_str(
        raw["motion_source_hash"], f"{where}.motion_source_hash"
      ),
      frames=_require_int(raw["frames"], f"{where}.frames", minimum=1),
      fps=_require_float(raw["fps"], f"{where}.fps", positive=True),
      tracked_body_names=_require_str_tuple(
        raw["tracked_body_names"], f"{where}.tracked_body_names"
      ),
      tracked_body_indices=indices,
      source_body_count=source_body_count,
    )


@dataclass(frozen=True, slots=True)
class CohortCommon:
  """Common manifest/action/control/observation contract of every member."""

  name: str
  robot: str
  task: str | None
  joint_names: tuple[str, ...]
  action_dim: int
  action_scales: tuple[float, ...]
  action_offset: float
  uses_default_offset: bool
  control_period_s: float
  control_hz: float
  sim_timestep: float
  decimation: int
  observation_names: tuple[str, ...]
  observation_widths: tuple[int, ...]
  anchor_body_name: str
  body_names: tuple[str, ...]
  reference_fps: float
  lookahead_s: float
  semantic_overrides: tuple[str, ...]

  @classmethod
  def from_contract(
    cls, cohort: CohortContract, semantic_overrides: Sequence[str] = ()
  ) -> CohortCommon:
    """Build the common record from a resolved cohort contract."""
    if not isinstance(cohort, CohortContract):
      raise CohortContractError("a common cohort record needs a CohortContract")
    overrides = tuple(
      _require_str(item, "semantic override") for item in semantic_overrides
    )
    return cls(
      name=cohort.manifest.name,
      robot=cohort.manifest.robot,
      task=cohort.manifest.base_task,
      joint_names=tuple(cohort.actions.joint_names),
      action_dim=int(cohort.actions.dim),
      action_scales=tuple(float(value) for value in cohort.actions.joint_scales),
      action_offset=float(cohort.actions.offset),
      uses_default_offset=bool(cohort.actions.uses_default_offset),
      control_period_s=float(cohort.control.control_period_s),
      control_hz=float(cohort.control.control_hz),
      sim_timestep=float(cohort.control.sim_timestep),
      decimation=int(cohort.control.decimation),
      observation_names=tuple(cohort.observations.names),
      observation_widths=tuple(int(width) for width in cohort.observations.widths),
      anchor_body_name=cohort.anchor_body_name,
      body_names=tuple(cohort.body_names),
      reference_fps=float(cohort.fps),
      lookahead_s=float(cohort.lookahead_s),
      semantic_overrides=overrides,
    )

  def as_dict(self) -> dict[str, Any]:
    return {
      "name": self.name,
      "robot": self.robot,
      "task": self.task,
      "joint_names": list(self.joint_names),
      "action_dim": self.action_dim,
      "action_scales": list(self.action_scales),
      "action_offset": self.action_offset,
      "uses_default_offset": self.uses_default_offset,
      "control_period_s": self.control_period_s,
      "control_hz": self.control_hz,
      "sim_timestep": self.sim_timestep,
      "decimation": self.decimation,
      "observation_names": list(self.observation_names),
      "observation_widths": list(self.observation_widths),
      "anchor_body_name": self.anchor_body_name,
      "body_names": list(self.body_names),
      "reference_fps": self.reference_fps,
      "lookahead_s": self.lookahead_s,
      "semantic_overrides": list(self.semantic_overrides),
    }

  @classmethod
  def from_dict(cls, payload: object, where: str = "cohort.common") -> CohortCommon:
    fields = (
      "name",
      "robot",
      "task",
      "joint_names",
      "action_dim",
      "action_scales",
      "action_offset",
      "uses_default_offset",
      "control_period_s",
      "control_hz",
      "sim_timestep",
      "decimation",
      "observation_names",
      "observation_widths",
      "anchor_body_name",
      "body_names",
      "reference_fps",
      "lookahead_s",
      "semantic_overrides",
    )
    raw = _require_keys(payload, fields, where)
    joint_names = _require_str_tuple(raw["joint_names"], f"{where}.joint_names")
    action_scales = _require_float_tuple(raw["action_scales"], f"{where}.action_scales")
    if len(action_scales) != len(joint_names):
      raise CohortContractError(
        f"{where}.action_scales must align with {len(joint_names)} joints"
      )
    observation_names = _require_str_tuple(
      raw["observation_names"], f"{where}.observation_names"
    )
    observation_widths = _require_int_tuple(
      raw["observation_widths"], f"{where}.observation_widths", minimum=0
    )
    if len(observation_widths) != len(observation_names):
      raise CohortContractError(
        f"{where}.observation_widths must align with {len(observation_names)} terms"
      )
    action_dim = _require_int(raw["action_dim"], f"{where}.action_dim", minimum=1)
    if action_dim != len(joint_names):
      raise CohortContractError(
        f"{where}.action_dim must equal the {len(joint_names)} saved joint names"
      )
    return cls(
      name=_require_str(raw["name"], f"{where}.name"),
      robot=_require_str(raw["robot"], f"{where}.robot"),
      task=_require_optional_str(raw["task"], f"{where}.task"),
      joint_names=joint_names,
      action_dim=action_dim,
      action_scales=action_scales,
      action_offset=_require_float(raw["action_offset"], f"{where}.action_offset"),
      uses_default_offset=_require_bool(
        raw["uses_default_offset"], f"{where}.uses_default_offset"
      ),
      control_period_s=_require_float(
        raw["control_period_s"], f"{where}.control_period_s", positive=True
      ),
      control_hz=_require_float(
        raw["control_hz"], f"{where}.control_hz", positive=True
      ),
      sim_timestep=_require_float(
        raw["sim_timestep"], f"{where}.sim_timestep", positive=True
      ),
      decimation=_require_int(raw["decimation"], f"{where}.decimation", minimum=1),
      observation_names=observation_names,
      observation_widths=observation_widths,
      anchor_body_name=_require_str(
        raw["anchor_body_name"], f"{where}.anchor_body_name"
      ),
      body_names=_require_str_tuple(raw["body_names"], f"{where}.body_names"),
      reference_fps=_require_float(
        raw["reference_fps"], f"{where}.reference_fps", positive=True
      ),
      lookahead_s=_require_float(raw["lookahead_s"], f"{where}.lookahead_s"),
      semantic_overrides=_require_str_tuple(
        raw["semantic_overrides"], f"{where}.semantic_overrides"
      ),
    )


@dataclass(frozen=True, slots=True)
class SlotPolicy:
  """Collection slot policy: requested weights, realized counts, phase sampling."""

  teacher_ids: tuple[str, ...]
  weights: tuple[float, ...]
  counts: tuple[int, ...]
  row_motion_ids: tuple[int, ...]
  phase_policy: str

  def __post_init__(self) -> None:
    if not self.teacher_ids:
      raise CohortContractError("a slot policy needs at least one motion")
    size = len(self.teacher_ids)
    if len(self.weights) != size or len(self.counts) != size:
      raise CohortContractError("slot weights and counts must align with motions")
    if any(count < 1 for count in self.counts):
      raise CohortContractError("every selected motion needs at least one row")
    if sum(self.counts) != len(self.row_motion_ids):
      raise CohortContractError("slot counts must cover every environment row")
    if any(not 0 <= motion < size for motion in self.row_motion_ids):
      raise CohortContractError("row motion ids must be inside the selected motions")
    if self.phase_policy not in PHASE_POLICIES:
      raise CohortContractError(
        f"phase policy must be one of {list(PHASE_POLICIES)}, got {self.phase_policy!r}"
      )

  @property
  def num_envs(self) -> int:
    return len(self.row_motion_ids)

  @classmethod
  def from_allocation(
    cls, slots: MotionSlotAllocation, phase_policy: PhasePolicy
  ) -> SlotPolicy:
    """Build the record from the command's realized slot allocation."""
    if not isinstance(slots, MotionSlotAllocation):
      raise CohortContractError("a slot policy needs a MotionSlotAllocation")
    return cls(
      teacher_ids=tuple(slots.teacher_ids),
      weights=tuple(float(value) for value in slots.weights),
      counts=tuple(int(count) for count in slots.counts),
      row_motion_ids=tuple(int(motion) for motion in slots.row_motion_ids),
      phase_policy=str(phase_policy),
    )

  def as_dict(self) -> dict[str, Any]:
    return {
      "teacher_ids": list(self.teacher_ids),
      "weights": list(self.weights),
      "counts": list(self.counts),
      "row_motion_ids": list(self.row_motion_ids),
      "num_envs": self.num_envs,
      "phase_policy": self.phase_policy,
    }

  @classmethod
  def from_dict(cls, payload: object, where: str = "cohort.slots") -> SlotPolicy:
    raw = _require_keys(
      payload,
      (
        "teacher_ids",
        "weights",
        "counts",
        "row_motion_ids",
        "num_envs",
        "phase_policy",
      ),
      where,
    )
    row_motion_ids = _require_int_tuple(
      raw["row_motion_ids"], f"{where}.row_motion_ids", minimum=0
    )
    num_envs = _require_int(raw["num_envs"], f"{where}.num_envs", minimum=1)
    if num_envs != len(row_motion_ids):
      raise CohortContractError(
        f"{where}.num_envs disagrees with the {len(row_motion_ids)} row motion ids"
      )
    return cls(
      teacher_ids=_require_str_tuple(raw["teacher_ids"], f"{where}.teacher_ids"),
      weights=_require_float_tuple(raw["weights"], f"{where}.weights"),
      counts=_require_int_tuple(raw["counts"], f"{where}.counts", minimum=1),
      row_motion_ids=row_motion_ids,
      phase_policy=_require_str(raw["phase_policy"], f"{where}.phase_policy"),
    )


@dataclass(frozen=True, slots=True)
class ReplayPolicy:
  """Replay partition policy that strict resume must reproduce."""

  kind: str
  capacity: int
  motion_ids: tuple[int, ...]
  weights: tuple[float, ...]
  quotas: tuple[int, ...]
  teacher_codes: tuple[int, ...]
  frame_counts: tuple[int, ...]

  def __post_init__(self) -> None:
    size = len(self.motion_ids)
    for name, values in (
      ("weights", self.weights),
      ("quotas", self.quotas),
      ("teacher_codes", self.teacher_codes),
      ("frame_counts", self.frame_counts),
    ):
      if len(values) != size:
        raise CohortContractError(f"replay policy {name} must align with motions")
    if size and sum(self.quotas) != self.capacity:
      raise CohortContractError(
        f"replay quotas {self.quotas} do not fill capacity {self.capacity}"
      )
    if any(quota < 1 for quota in self.quotas):
      raise CohortContractError("every selected motion needs a replay slot")

  @classmethod
  def from_buffer(cls, replay: ReplayBufferProtocol) -> ReplayPolicy:
    """Record the policy of the live replay buffer.

    M4 cohorts require the per-motion balanced buffer, because the identity
    carries each motion's quota and routing.  The single-partition FIFO buffer is
    recorded as its own kind so a version-2 checkpoint can never be resumed
    against a different replay layout.
    """
    if isinstance(replay, BalancedReplayBuffer):
      quota = replay.quota
      return cls(
        kind=BALANCED_REPLAY_KIND,
        capacity=replay.capacity,
        motion_ids=tuple(int(motion) for motion in replay.motion_ids),
        weights=tuple(float(value) for value in replay.weights),
        quotas=tuple(int(value) for value in quota.quotas),
        teacher_codes=tuple(int(code) for code in replay.teacher_codes),
        frame_counts=tuple(int(frames) for frames in replay.frame_counts),
      )
    if isinstance(replay, ReplayBufferProtocol):
      return cls(
        kind=FIFO_REPLAY_KIND,
        capacity=replay.capacity,
        motion_ids=(),
        weights=(),
        quotas=(),
        teacher_codes=(),
        frame_counts=(),
      )
    raise CohortContractError(
      f"unsupported replay implementation {type(replay).__name__}: a cohort "
      "identity needs a labeled replay buffer"
    )

  def as_dict(self) -> dict[str, Any]:
    return {
      "kind": self.kind,
      "capacity": self.capacity,
      "motion_ids": list(self.motion_ids),
      "weights": list(self.weights),
      "quotas": list(self.quotas),
      "teacher_codes": list(self.teacher_codes),
      "frame_counts": list(self.frame_counts),
    }

  @classmethod
  def from_dict(cls, payload: object, where: str = "cohort.replay") -> ReplayPolicy:
    raw = _require_keys(
      payload,
      (
        "kind",
        "capacity",
        "motion_ids",
        "weights",
        "quotas",
        "teacher_codes",
        "frame_counts",
      ),
      where,
    )
    kind = _require_str(raw["kind"], f"{where}.kind")
    return cls(
      kind=kind,
      capacity=_require_int(raw["capacity"], f"{where}.capacity", minimum=1),
      motion_ids=_require_int_tuple(
        raw["motion_ids"], f"{where}.motion_ids", minimum=0
      ),
      weights=_require_float_tuple(raw["weights"], f"{where}.weights"),
      quotas=_require_int_tuple(raw["quotas"], f"{where}.quotas", minimum=1),
      teacher_codes=_require_int_tuple(
        raw["teacher_codes"], f"{where}.teacher_codes", minimum=0
      ),
      frame_counts=_require_int_tuple(
        raw["frame_counts"], f"{where}.frame_counts", minimum=1
      ),
    )


@dataclass(frozen=True, slots=True)
class CohortResources:
  """Runtime resource and seed settings a strict resume must reproduce."""

  device: str
  requested_seed: int
  effective_seed: int | None

  def as_dict(self) -> dict[str, Any]:
    return {
      "device": self.device,
      "requested_seed": self.requested_seed,
      "effective_seed": self.effective_seed,
    }

  @classmethod
  def from_dict(
    cls, payload: object, where: str = "cohort.resources"
  ) -> CohortResources:
    raw = _require_keys(payload, ("device", "requested_seed", "effective_seed"), where)
    return cls(
      device=_require_str(raw["device"], f"{where}.device"),
      requested_seed=_require_int(
        raw["requested_seed"], f"{where}.requested_seed", minimum=0
      ),
      effective_seed=_require_optional_int(
        raw["effective_seed"], f"{where}.effective_seed"
      ),
    )


@dataclass(frozen=True, slots=True)
class CohortIdentity:
  """Ordered cohort identity of one multi-teacher distillation run."""

  version: int
  manifest_path: str
  manifest_sha256: str
  mapping_digest: str
  members: tuple[CohortMember, ...]
  common: CohortCommon
  slots: SlotPolicy
  replay: ReplayPolicy
  resources: CohortResources

  def __post_init__(self) -> None:
    if self.version != COHORT_CONTRACT_VERSION:
      raise CohortContractError(
        f"cohort contract version {self.version} is not supported"
      )
    if not self.members:
      raise CohortContractError("a cohort identity needs at least one member")
    if tuple(member.motion_id for member in self.members) != tuple(
      range(len(self.members))
    ):
      raise CohortContractError("member motion ids must be the ordered positions")
    if len(set(self.teacher_ids)) != len(self.members):
      raise CohortContractError("cohort member teacher ids must be unique")
    if self.slots.teacher_ids != self.teacher_ids:
      raise CohortContractError(
        "slot policy teachers must match the ordered cohort members"
      )
    if self.replay.kind != BALANCED_REPLAY_KIND:
      raise CohortContractError(
        "an M4 cohort identity requires the per-motion balanced replay policy, got "
        f"{self.replay.kind!r}"
      )
    if self.replay.motion_ids != tuple(member.motion_id for member in self.members):
      raise CohortContractError(
        "replay policy motions must match the ordered cohort members"
      )

  @property
  def teacher_ids(self) -> tuple[str, ...]:
    return tuple(member.teacher_id for member in self.members)

  @property
  def teacher_codes(self) -> tuple[int, ...]:
    return tuple(member.teacher_code for member in self.members)

  def member(self, teacher_id: str) -> CohortMember:
    """Return the member with ``teacher_id``."""
    for member in self.members:
      if member.teacher_id == teacher_id:
        return member
    raise CohortContractError(
      f"the cohort has no member {teacher_id!r}; members are {list(self.teacher_ids)}"
    )

  def member_by_motion(self, motion_id: int) -> CohortMember:
    if not 0 <= motion_id < len(self.members):
      raise CohortContractError(
        f"motion id {motion_id} is outside the {len(self.members)} cohort members"
      )
    return self.members[motion_id]

  def as_dict(self) -> dict[str, Any]:
    return {
      "version": self.version,
      "manifest_path": self.manifest_path,
      "manifest_sha256": self.manifest_sha256,
      "mapping_digest": self.mapping_digest,
      "members": [member.as_dict() for member in self.members],
      "common": self.common.as_dict(),
      "slots": self.slots.as_dict(),
      "replay": self.replay.as_dict(),
      "resources": self.resources.as_dict(),
    }

  def digest(self) -> str:
    """Content digest of the whole identity record."""
    return hashlib.sha256(canonical_json(self.as_dict()).encode("utf-8")).hexdigest()

  @classmethod
  def from_dict(cls, payload: object, where: str = "cohort") -> CohortIdentity:
    raw = _require_keys(
      payload,
      (
        "version",
        "manifest_path",
        "manifest_sha256",
        "mapping_digest",
        "members",
        "common",
        "slots",
        "replay",
        "resources",
      ),
      where,
    )
    version = _require_int(raw["version"], f"{where}.version", minimum=1)
    if version != COHORT_CONTRACT_VERSION:
      raise CohortContractError(f"{where} contract version {version} is not supported")
    members = raw["members"]
    if not isinstance(members, (list, tuple)):
      raise CohortContractError(f"{where}.members must be a list")
    return cls(
      version=version,
      manifest_path=_require_str(raw["manifest_path"], f"{where}.manifest_path"),
      manifest_sha256=_require_str(raw["manifest_sha256"], f"{where}.manifest_sha256"),
      mapping_digest=_require_str(raw["mapping_digest"], f"{where}.mapping_digest"),
      members=tuple(
        CohortMember.from_dict(item, f"{where}.members[{index}]")
        for index, item in enumerate(members)
      ),
      common=CohortCommon.from_dict(raw["common"], f"{where}.common"),
      slots=SlotPolicy.from_dict(raw["slots"], f"{where}.slots"),
      replay=ReplayPolicy.from_dict(raw["replay"], f"{where}.replay"),
      resources=CohortResources.from_dict(raw["resources"], f"{where}.resources"),
    )


def build_cohort_identity(
  cohort: CohortContract,
  library: MotionLibrary,
  slots: MotionSlotAllocation,
  *,
  phase_policy: PhasePolicy,
  replay: ReplayBufferProtocol,
  device: str,
  requested_seed: int,
  effective_seed: int | None,
  semantic_overrides: Sequence[str] = (),
  body_selection: BodySelection | None = None,
  mapping_digest: str | None = None,
) -> CohortIdentity:
  """Build the ordered cohort identity of one mixed-slot distillation run.

  Every member is cross-checked against the resolved manifest (artifact digests,
  clip extent, source digest, per-clip teacher code), the pinned slot policy, and
  the replay partition policy, so the record cannot claim a cohort the selected
  environment or the live replay does not implement.

  Args:
    cohort: resolved manifest contract of every member.
    library: reference clips in the order the command was built with.
    slots: realized per-row slot allocation of the command.
    phase_policy: phase-sampling policy the environment was built with.
    replay: the live per-motion balanced replay buffer.
    device: resolved runtime device string.
    requested_seed: seed the environment factory applied before construction.
    effective_seed: seed the environment actually reports, when available.
    semantic_overrides: private environment overrides recorded by the factory.
    body_selection: audited tracked-body mapping; defaults to the library's own
      resolved selection, which only exists after a command resolved it against
      the compiled robot.
    mapping_digest: digest the live audit reported; defaults to the library's.

  Raises:
    CohortContractError: if any cross-check fails or evidence is missing.
  """
  if not isinstance(cohort, CohortContract):
    raise CohortContractError("a cohort identity needs a resolved CohortContract")
  if not isinstance(library, MotionLibrary):
    raise CohortContractError("a cohort identity needs a MotionLibrary")
  if not isinstance(slots, MotionSlotAllocation):
    raise CohortContractError("a cohort identity needs a MotionSlotAllocation")
  if phase_policy not in PHASE_POLICIES:
    raise CohortContractError(
      f"phase policy must be one of {list(PHASE_POLICIES)}, got {phase_policy!r}"
    )
  selection = body_selection if body_selection is not None else library.body_selection
  if selection is None:
    raise CohortContractError(
      "the cohort identity needs the audited tracked-body mapping resolved against "
      "the compiled robot: build it from an adapted environment or pass the "
      "library's body selection"
    )
  if not isinstance(selection, BodySelection):
    raise CohortContractError("body_selection must be a BodySelection")
  if selection.names is not None and tuple(selection.names) != tuple(cohort.body_names):
    raise CohortContractError(
      f"tracked body names {tuple(selection.names)} disagree with the cohort's "
      f"declared body names {tuple(cohort.body_names)}"
    )
  if selection.source_body_count != library.source_body_count:
    raise CohortContractError(
      f"body selection covers {selection.source_body_count} source bodies but the "
      f"reference clips carry {library.source_body_count}"
    )

  clip_teacher_ids = tuple(clip.teacher_id for clip in library.clips)
  if clip_teacher_ids != tuple(slots.teacher_ids):
    raise CohortContractError(
      f"slot allocation teachers {tuple(slots.teacher_ids)} disagree with the "
      f"library clip order {clip_teacher_ids}"
    )
  manifest_teacher_ids = tuple(teacher.id for teacher in cohort.teachers)

  members: list[CohortMember] = []
  for clip in library.clips:
    teacher = cohort.teacher(clip.teacher_id)
    code = manifest_teacher_ids.index(clip.teacher_id)
    if code != clip.teacher_code:
      raise CohortContractError(
        f"clip {clip.teacher_id!r} is routed with code {clip.teacher_code} but the "
        f"manifest places it at position {code}"
      )
    hashes = {role: str(teacher.hashes.get(role, "")) for role in COHORT_ARTIFACT_ROLES}
    missing = [role for role, value in hashes.items() if not value]
    if missing:
      raise CohortContractError(
        f"teacher {clip.teacher_id!r} records no artifact digest for {missing}"
      )
    if clip.frames != teacher.reference.frames:
      raise CohortContractError(
        f"clip {clip.teacher_id!r} has {clip.frames} frames but its teacher declares "
        f"{teacher.reference.frames}"
      )
    if not math.isclose(clip.fps, teacher.reference.fps, rel_tol=1e-9):
      raise CohortContractError(
        f"clip {clip.teacher_id!r} has {clip.fps} fps but its teacher declares "
        f"{teacher.reference.fps}"
      )
    if clip.source_hash != hashes["motion"]:
      raise CohortContractError(
        f"clip {clip.teacher_id!r} was loaded from a different motion content than "
        "the teacher artifact digest records"
      )
    if clip.source_body_count != selection.source_body_count:
      raise CohortContractError(
        f"clip {clip.teacher_id!r} carries {clip.source_body_count} source bodies but "
        f"the body selection covers {selection.source_body_count}"
      )
    members.append(
      CohortMember(
        motion_id=int(clip.motion_id),
        teacher_id=clip.teacher_id,
        teacher_code=int(clip.teacher_code),
        artifacts={
          role: str(teacher.entry.paths()[role]) for role in COHORT_ARTIFACT_ROLES
        },
        hashes=hashes,
        motion_path=str(clip.motion_file),
        motion_source_hash=clip.source_hash,
        frames=int(clip.frames),
        fps=float(clip.fps),
        tracked_body_names=tuple(cohort.body_names),
        tracked_body_indices=tuple(int(index) for index in selection.indices),
        source_body_count=int(selection.source_body_count),
      )
    )

  for position, member in enumerate(members):
    weight = float(cohort.teacher(member.teacher_id).entry.sampling_weight)
    _require_close(
      f"slot weight for {member.teacher_id!r}", float(slots.weights[position]), weight
    )

  policy = ReplayPolicy.from_buffer(replay)
  if policy.kind != BALANCED_REPLAY_KIND:
    raise CohortContractError(
      "an M4 cohort identity requires the per-motion balanced replay buffer, got "
      f"{policy.kind!r}"
    )
  if policy.motion_ids != tuple(member.motion_id for member in members):
    raise CohortContractError(
      f"replay partitions {policy.motion_ids} disagree with the cohort motions "
      f"{tuple(member.motion_id for member in members)}"
    )
  if policy.teacher_codes != tuple(member.teacher_code for member in members):
    raise CohortContractError(
      f"replay routing {policy.teacher_codes} disagrees with the cohort teacher codes "
      f"{tuple(member.teacher_code for member in members)}"
    )
  if policy.frame_counts != tuple(member.frames for member in members):
    raise CohortContractError(
      f"replay frame limits {policy.frame_counts} disagree with the cohort clip "
      f"extents {tuple(member.frames for member in members)}"
    )
  for position, member in enumerate(members):
    _require_close(
      f"replay weight for {member.teacher_id!r}",
      float(policy.weights[position]),
      float(slots.weights[position]),
    )

  digest = library.mapping_digest()
  if mapping_digest is not None and mapping_digest != digest:
    raise CohortContractError(
      "the live audit's ordered mapping digest disagrees with the reference library"
    )
  return CohortIdentity(
    version=COHORT_CONTRACT_VERSION,
    manifest_path=str(cohort.manifest.path),
    manifest_sha256=str(cohort.manifest.sha256),
    mapping_digest=digest,
    members=tuple(members),
    common=CohortCommon.from_contract(cohort, semantic_overrides),
    slots=SlotPolicy.from_allocation(slots, phase_policy),
    replay=policy,
    resources=CohortResources(
      device=_require_str(device, "device"),
      requested_seed=_require_int(requested_seed, "requested_seed", minimum=0),
      effective_seed=effective_seed,
    ),
  )


def cohort_identity_from_adapter(
  adapter: Any, *, replay: ReplayBufferProtocol
) -> CohortIdentity:
  """Build the identity recorded by an adapted mixed-slot environment.

  The live audit is required evidence, not decoration: its slot allocation,
  ordered mapping digest, phase policy, and seed provenance are recorded, and a
  mismatch between the audit and the reference library is refused.  A cohort
  identity therefore never claims a seed or an allocation the factory did not
  actually apply.
  """
  for name in ("cohort", "library", "audit", "env"):
    if not hasattr(adapter, name):
      raise CohortContractError(
        f"a cohort identity needs an adapted mixed-slot environment exposing "
        f"{name!r}; got {type(adapter).__name__}"
      )
  return cohort_identity_from_parts(
    cohort=adapter.cohort,
    library=adapter.library,
    audit=adapter.audit,
    replay=replay,
    device=str(adapter.env.device),
  )


def cohort_identity_from_parts(
  *,
  cohort: Any,
  library: Any,
  audit: Any,
  replay: ReplayBufferProtocol,
  device: str,
) -> CohortIdentity:
  """Build the identity from one audited environment's parts.

  Split from the adapter form so a sharded run can record the identity its
  workers' environments actually produced: the parent owns no environment, but
  each worker builds its environment through the same recipe.  Worker 0's
  environment seed is the single-process derivation, so a sharded run records
  the same identity a non-sharded run of the same recipe would record — the
  per-worker devices and seeds live in the resolved configuration instead.

  The audit is still required evidence: its slot allocation, ordered mapping
  digest, phase policy and seed provenance are recorded, so a caller cannot
  claim an allocation the factory did not apply.
  """
  for name in ("slots", "mapping_digest", "phase_policy", "seed_provenance"):
    if not hasattr(audit, name):
      raise CohortContractError(
        f"the live audit exposes no {name!r}; rebuild the environment so the "
        "multi-motion contract is audited before a cohort identity is recorded"
      )
  provenance = audit.seed_provenance
  if provenance is None or not provenance.applied_before_construction:
    raise CohortContractError(
      "the cohort identity needs a seed the environment factory applied before "
      "construction; build the environment with seed=<int>"
    )
  if provenance.requested_seed is None:
    raise CohortContractError("the live audit reports no requested seed")
  return build_cohort_identity(
    cohort,
    library,
    audit.slots,
    phase_policy=audit.phase_policy,
    replay=replay,
    device=device,
    requested_seed=int(provenance.requested_seed),
    effective_seed=provenance.effective_seed,
    semantic_overrides=tuple(getattr(audit, "semantic_overrides", ())),
    mapping_digest=str(audit.mapping_digest),
  )


def _differ(name: str, stored: object, expected: object) -> CohortContractError:
  return CohortContractError(f"{name} differs: checkpoint {stored!r} != {expected!r}")


def require_same_cohort(stored: CohortIdentity, expected: CohortIdentity) -> None:
  """Require a checkpoint's cohort identity to be reproduced exactly.

  Strict resume reproduces the ordered members, every artifact path and digest,
  the clip extents and audited body mapping, the common contract, the slot and
  phase policy, the replay partition policy, and the resource/seed settings.
  Members that are missing, extra, or reordered, and every changed value, are
  reported by name so the caller can explain the refusal.
  """
  if stored.version != expected.version:
    raise _differ("cohort contract version", stored.version, expected.version)
  if stored.manifest_path != expected.manifest_path:
    raise _differ("manifest path", stored.manifest_path, expected.manifest_path)
  if stored.manifest_sha256 != expected.manifest_sha256:
    raise _differ("manifest digest", stored.manifest_sha256, expected.manifest_sha256)
  if stored.teacher_ids != expected.teacher_ids:
    not_in_live = [
      item for item in stored.teacher_ids if item not in expected.teacher_ids
    ]
    not_in_checkpoint = [
      item for item in expected.teacher_ids if item not in stored.teacher_ids
    ]
    if not_in_live or not_in_checkpoint:
      raise CohortContractError(
        "cohort members differ: the checkpoint was trained with "
        f"{list(stored.teacher_ids)} but the live cohort has "
        f"{list(expected.teacher_ids)} (not in the live cohort {not_in_live}, "
        f"not in the checkpoint {not_in_checkpoint})"
      )
    raise CohortContractError(
      "cohort members are reordered: the checkpoint was trained with "
      f"{list(stored.teacher_ids)} but the live cohort orders them "
      f"{list(expected.teacher_ids)}; strict resume keeps the stored order"
    )
  if stored.mapping_digest != expected.mapping_digest:
    raise _differ(
      "ordered reference mapping digest", stored.mapping_digest, expected.mapping_digest
    )
  for member, reference in zip(stored.members, expected.members, strict=True):
    _require_same_member(member, reference)
  _require_same_common(stored.common, expected.common)
  _require_same_slots(stored.slots, expected.slots)
  if stored.replay.as_dict() != expected.replay.as_dict():
    require_same_replay_policy(stored.replay, expected.replay)
  if stored.resources.as_dict() != expected.resources.as_dict():
    _require_same_resources(stored.resources, expected.resources)


def _require_same_member(member: CohortMember, reference: CohortMember) -> None:
  prefix = f"cohort member {member.teacher_id!r}"
  for name in ("motion_id", "teacher_code", "frames", "source_body_count"):
    if getattr(member, name) != getattr(reference, name):
      raise _differ(f"{prefix} {name}", getattr(member, name), getattr(reference, name))
  if not math.isclose(member.fps, reference.fps, rel_tol=1e-12):
    raise _differ(f"{prefix} fps", member.fps, reference.fps)
  if member.motion_source_hash != reference.motion_source_hash:
    raise _differ(
      f"{prefix} motion content digest",
      member.motion_source_hash,
      reference.motion_source_hash,
    )
  for role in COHORT_ARTIFACT_ROLES:
    if member.hashes.get(role) != reference.hashes.get(role):
      raise _differ(
        f"{prefix} artifact {role!r} digest",
        member.hashes.get(role),
        reference.hashes.get(role),
      )
  for role in COHORT_ARTIFACT_ROLES:
    if member.artifacts.get(role) != reference.artifacts.get(role):
      raise _differ(
        f"{prefix} artifact {role!r} path",
        member.artifacts.get(role),
        reference.artifacts.get(role),
      )
  if member.motion_path != reference.motion_path:
    raise _differ(f"{prefix} motion path", member.motion_path, reference.motion_path)
  if member.tracked_body_names != reference.tracked_body_names:
    raise _differ(
      f"{prefix} tracked body names",
      member.tracked_body_names,
      reference.tracked_body_names,
    )
  if member.tracked_body_indices != reference.tracked_body_indices:
    raise _differ(
      f"{prefix} tracked body indices",
      member.tracked_body_indices,
      reference.tracked_body_indices,
    )


def _require_same_common(common: CohortCommon, reference: CohortCommon) -> None:
  for name in common.as_dict():
    if getattr(common, name) != getattr(reference, name):
      raise _differ(
        f"common contract {name}", getattr(common, name), getattr(reference, name)
      )


def _require_same_slots(slots: SlotPolicy, reference: SlotPolicy) -> None:
  if slots.teacher_ids != reference.teacher_ids:
    raise _differ("slot policy motions", slots.teacher_ids, reference.teacher_ids)
  if slots.counts != reference.counts:
    raise _differ("slot row counts", slots.counts, reference.counts)
  if slots.row_motion_ids != reference.row_motion_ids:
    raise _differ("slot row motion ids", slots.row_motion_ids, reference.row_motion_ids)
  if slots.phase_policy != reference.phase_policy:
    raise _differ("phase policy", slots.phase_policy, reference.phase_policy)
  for position, (left, right) in enumerate(
    zip(slots.weights, reference.weights, strict=True)
  ):
    if not math.isclose(left, right, rel_tol=1e-12):
      raise _differ(f"slot weight[{position}]", left, right)


def require_same_replay_policy(policy: ReplayPolicy, reference: ReplayPolicy) -> None:
  """Require two replay partition policies to be identical, field by field."""
  if policy.kind != reference.kind:
    raise _differ("replay policy kind", policy.kind, reference.kind)
  if policy.capacity != reference.capacity:
    raise _differ("replay capacity", policy.capacity, reference.capacity)
  for name in ("motion_ids", "quotas", "teacher_codes", "frame_counts"):
    if getattr(policy, name) != getattr(reference, name):
      raise _differ(f"replay {name}", getattr(policy, name), getattr(reference, name))
  for position, (left, right) in enumerate(
    zip(policy.weights, reference.weights, strict=True)
  ):
    if not math.isclose(left, right, rel_tol=1e-12):
      raise _differ(f"replay weight[{position}]", left, right)


def _require_same_resources(
  resources: CohortResources, reference: CohortResources
) -> None:
  for name in resources.as_dict():
    if getattr(resources, name) != getattr(reference, name):
      raise _differ(
        f"resource setting {name}", getattr(resources, name), getattr(reference, name)
      )


def require_member_matches(
  identity: CohortIdentity, cohort: CohortContract, teacher_id: str
) -> CohortMember:
  """Check one requested member of a saved cohort against the live manifest.

  The requested member must exist in the stored cohort, and its artifact digests,
  clip extent, saved reference digest, and the common action/control/observation
  contract must match the live manifest.  Artifact *paths* are deliberately not
  compared: a relocated byte-identical artifact is accepted by its content digest
  and reported by the caller as a relocation, while a changed artifact fails on
  its digest.
  """
  member = identity.member(teacher_id)
  teacher = cohort.teacher(teacher_id)
  manifest_teacher_ids = tuple(item.id for item in cohort.teachers)
  if manifest_teacher_ids.index(teacher_id) != member.teacher_code:
    raise _differ(
      f"member {teacher_id!r} teacher code",
      member.teacher_code,
      manifest_teacher_ids.index(teacher_id),
    )
  for role in COHORT_ARTIFACT_ROLES:
    stored = member.hashes.get(role)
    live = str(teacher.hashes.get(role, ""))
    if not live:
      raise CohortContractError(
        f"the live cohort records no {role!r} digest for {teacher_id!r}"
      )
    if stored != live:
      raise _differ(f"member {teacher_id!r} artifact {role!r} digest", stored, live)
  if member.motion_source_hash != str(teacher.hashes.get("motion", "")):
    raise _differ(
      f"member {teacher_id!r} motion content digest",
      member.motion_source_hash,
      teacher.hashes.get("motion"),
    )
  if member.frames != teacher.reference.frames:
    raise _differ(
      f"member {teacher_id!r} reference frames",
      member.frames,
      teacher.reference.frames,
    )
  if not math.isclose(member.fps, float(teacher.reference.fps), rel_tol=1e-9):
    raise _differ(
      f"member {teacher_id!r} reference fps", member.fps, teacher.reference.fps
    )
  stored_common = identity.common.as_dict()
  live_common = CohortCommon.from_contract(cohort).as_dict()
  # ``semantic_overrides`` records private environment overrides applied by the
  # builder; they are not derivable from the manifest, so the env_config digest
  # checked above is what attests them here.
  stored_common.pop("semantic_overrides", None)
  live_common.pop("semantic_overrides", None)
  for name, value in live_common.items():
    if stored_common.get(name) != value:
      raise _differ(f"common contract {name}", stored_common.get(name), value)
  return member


__all__ = [
  "BALANCED_REPLAY_KIND",
  "COHORT_ARTIFACT_ROLES",
  "COHORT_CONTRACT_VERSION",
  "FIFO_REPLAY_KIND",
  "RESET_PROVENANCE_VERSION",
  "ResetProvenance",
  "CohortCommon",
  "CohortContractError",
  "CohortIdentity",
  "CohortMember",
  "CohortResources",
  "ReplayPolicy",
  "SlotPolicy",
  "build_cohort_identity",
  "canonical_json",
  "cohort_identity_from_adapter",
  "reset_provenance_from_adapter",
  "require_same_cohort",
  "require_same_replay_policy",
]
