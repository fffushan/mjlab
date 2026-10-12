"""Frozen-policy and simulator boundaries for the D1 collector.

The adapter layer is deliberately callback based: it can wrap the existing
mjlab environment without changing production reset/step behavior, while fake
implementations can exercise the lifecycle entirely on CPU.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, cast

import numpy as np
import torch

from .state import WorldState


class AdapterError(ValueError):
  """A policy or environment contract is invalid."""


_X2_JOINT_ORDER = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_pitch_joint",
  "waist_roll_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_yaw_joint",
  "left_wrist_pitch_joint",
  "left_wrist_roll_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_yaw_joint",
  "right_wrist_pitch_joint",
  "right_wrist_roll_joint",
  "head_yaw_joint",
  "head_pitch_joint",
)
_MIXED_COHORT_MANIFEST = "configs/distillation/x2_tennis_mixed.yaml"
_MIXED_COHORT_MANIFEST_SHA256 = (
  "587550f26345a278f14c350404be11843dca379d708c2a3c418f0aa8340f9b2d"
)
_PPO_OBSERVATION_LAYOUT = (
  ("base_ang_vel", 3),
  ("projected_gravity", 3),
  ("joint_pos", len(_X2_JOINT_ORDER)),
  ("joint_vel", len(_X2_JOINT_ORDER)),
  ("actions", len(_X2_JOINT_ORDER)),
  ("command", 3),
)
_PPO_ACTION_FIELD_INDEX = next(
  index for index, (name, _) in enumerate(_PPO_OBSERVATION_LAYOUT) if name == "actions"
)
_PPO_ACTION_SLICE = slice(
  sum(width for _, width in _PPO_OBSERVATION_LAYOUT[:_PPO_ACTION_FIELD_INDEX]),
  sum(width for _, width in _PPO_OBSERVATION_LAYOUT[: _PPO_ACTION_FIELD_INDEX + 1]),
)


def _vector(name: str, value: Sequence[float] | np.ndarray, width: int) -> np.ndarray:
  result = np.asarray(value, dtype=np.float64)
  if result.shape != (width,) or not np.isfinite(result).all():
    raise AdapterError(f"{name} must be finite with shape [{width}]")
  return result.copy()


def _matrix(name: str, value: Any, width: int) -> np.ndarray:
  result = np.asarray(value, dtype=np.float64)
  if result.ndim != 2 or result.shape[1] != width or not np.isfinite(result).all():
    raise AdapterError(f"{name} must be finite with shape [N, {width}]")
  if result.shape[0] == 0:
    raise AdapterError(f"{name} must contain at least one row")
  return result.copy()


def _batch_indices(value: Any, size: int) -> np.ndarray:
  indices = np.asarray(value, dtype=np.int64).reshape(-1)
  if indices.size == 0 or np.any(indices < 0) or np.any(indices >= size):
    raise AdapterError(f"batch indices must be non-empty and lie in [0, {size})")
  if np.unique(indices).size != indices.size:
    raise AdapterError("batch indices must not repeat rows")
  return indices


def _csv_values(name: str, value: Any, width: int) -> np.ndarray:
  """Decode an exporter list, whose ONNX representation is CSV text."""
  if isinstance(value, str):
    parts = [part.strip() for part in value.split(",")]
    if any(not part for part in parts):
      raise AdapterError(f"metadata {name} contains an empty CSV element")
    try:
      value = [float(part) for part in parts]
    except ValueError as exc:
      raise AdapterError(f"metadata {name} is not numeric CSV") from exc
  return _vector(f"metadata {name}", value, width)


def _metadata_names(name: str, value: Any, width: int) -> tuple[str, ...]:
  if isinstance(value, str):
    values = tuple(part.strip() for part in value.split(","))
  elif isinstance(value, (list, tuple)):
    values = tuple(value)
  else:
    raise AdapterError(f"metadata {name} must be a list or CSV string")
  if (
    len(values) != width
    or any(not isinstance(item, str) or not item for item in values)
    or len(set(values)) != width
  ):
    raise AdapterError(f"metadata {name} must contain {width} unique joint names")
  return values


def _metadata_field(metadata: Mapping[str, Any], *names: str) -> Any:
  for name in names:
    if name in metadata:
      return metadata[name]
  return None


def resolve_inference_device(value: str | torch.device) -> str:
  """Normalize and validate an explicit policy inference device.

  ``cpu`` keeps the frozen CPU policy path.  ``cuda[:N]`` is accepted only
  when CUDA is available and the requested index exists, so an unusable device
  fails before any checkpoint or environment is constructed.
  """
  try:
    device = torch.device(value)
  except (RuntimeError, TypeError, ValueError) as exc:
    raise AdapterError(f"invalid policy inference device {value!r}: {exc}") from exc
  if device.type == "cpu":
    return "cpu"
  if device.type != "cuda":
    raise AdapterError(
      f"policy inference device must be cpu or cuda[:N], got {value!r}"
    )
  if not torch.cuda.is_available():
    raise AdapterError(
      f"policy inference device {value!r} needs CUDA, which is unavailable"
    )
  if device.index is not None and device.index >= torch.cuda.device_count():
    raise AdapterError(
      f"policy inference device {value!r} exceeds {torch.cuda.device_count()} GPUs"
    )
  return "cuda" if device.index is None else f"cuda:{device.index}"


def _repo_root() -> Path:
  return Path(__file__).resolve().parents[5]


def _read_onnx_metadata(path: Path) -> dict[str, Any]:
  try:
    import onnx

    model = onnx.load(path)
  except Exception as exc:
    raise AdapterError(f"could not read ONNX metadata from {path}: {exc}") from exc
  return {entry.key: entry.value for entry in model.metadata_props}


@dataclass(frozen=True, slots=True)
class ActionContract:
  """Normalized action conversion plus separate actuator metadata.

  ``offset`` is the exported default joint position and ``scale`` is the
  exported action scale.  Stiffness and damping are retained as separate
  vectors; neither is a proxy for the other or for a collapsed gains vector.
  """

  scale: np.ndarray
  offset: np.ndarray
  stiffness: np.ndarray
  damping: np.ndarray
  joint_order: tuple[str, ...] = _X2_JOINT_ORDER

  def __post_init__(self) -> None:
    object.__setattr__(self, "scale", _vector("scale", self.scale, 31))
    object.__setattr__(self, "offset", _vector("offset", self.offset, 31))
    object.__setattr__(self, "stiffness", _vector("stiffness", self.stiffness, 31))
    object.__setattr__(self, "damping", _vector("damping", self.damping, 31))
    if np.any(self.scale == 0):
      raise AdapterError("action scale entries must be non-zero")
    if tuple(self.joint_order) != _X2_JOINT_ORDER:
      raise AdapterError("joint_order must match the frozen X2 action order")

  @classmethod
  def from_metadata(cls, metadata: Mapping[str, Any]) -> "ActionContract":
    """Build a contract from the real exporter metadata schema."""
    if not isinstance(metadata, Mapping):
      raise AdapterError("artifact action metadata is required")
    names_value = _metadata_field(metadata, "joint_order", "joint_names")
    scale_value = _metadata_field(
      metadata, "scale", "scales", "action_scales", "action_scale"
    )
    offset_value = _metadata_field(
      metadata, "offset", "action_offset", "default_joint_pos", "default_joint_position"
    )
    stiffness_value = _metadata_field(metadata, "stiffness", "joint_stiffness")
    damping_value = _metadata_field(metadata, "damping", "joint_damping")
    if (
      names_value is None
      or scale_value is None
      or offset_value is None
      or stiffness_value is None
      or damping_value is None
    ):
      raise AdapterError(
        "artifact metadata lacks complete action conversion/actuator fields"
      )
    names = _metadata_names("joint_names", names_value, 31)
    return cls(
      _csv_values("action_scale", scale_value, 31),
      _csv_values("default_joint_pos", offset_value, 31),
      _csv_values("joint_stiffness", stiffness_value, 31),
      _csv_values("joint_damping", damping_value, 31),
      names,
    )

  def verify_metadata(
    self, metadata: Any, *, require_actuator_metadata: bool = False
  ) -> None:
    """Fail closed unless artifact metadata matches this action contract.

    The ONNX exporter stores all vector fields as CSV strings.  VAE cohort
    metadata carries conversion fields but not actuator gains, while ONNX
    metadata must carry both ``joint_stiffness`` and ``joint_damping``.
    """
    if not isinstance(metadata, Mapping):
      raise AdapterError("artifact action metadata is required")
    nested = metadata.get("action_metadata")
    if isinstance(nested, Mapping):
      metadata = nested
    names_value = _metadata_field(metadata, "joint_order", "joint_names")
    scale_value = _metadata_field(
      metadata, "scale", "scales", "action_scales", "action_scale"
    )
    offset_value = _metadata_field(
      metadata, "offset", "action_offset", "default_joint_pos", "default_joint_position"
    )
    if names_value is None or scale_value is None or offset_value is None:
      raise AdapterError("artifact metadata lacks complete action conversion fields")
    names = _metadata_names("joint_names", names_value, 31)
    if names != self.joint_order:
      raise AdapterError("artifact joint order disagrees with action contract")
    scales = _csv_values("action scale", scale_value, 31)
    offsets = _csv_values("action offset", offset_value, 31)
    if not np.allclose(scales, self.scale, atol=1.0e-3, rtol=0.0):
      raise AdapterError("artifact action scale disagrees with action contract")
    if not np.allclose(offsets, self.offset, atol=1.0e-3, rtol=0.0):
      raise AdapterError(
        "artifact default joint position disagrees with action contract"
      )

    stiffness_value = _metadata_field(metadata, "stiffness", "joint_stiffness")
    damping_value = _metadata_field(metadata, "damping", "joint_damping")
    if (stiffness_value is None) != (damping_value is None):
      raise AdapterError(
        "artifact actuator metadata must include stiffness and damping"
      )
    if stiffness_value is None:
      if require_actuator_metadata:
        raise AdapterError("artifact metadata lacks separate stiffness and damping")
      return
    stiffness = _csv_values("joint stiffness", stiffness_value, 31)
    damping = _csv_values("joint damping", damping_value, 31)
    if not np.allclose(stiffness, self.stiffness, atol=1.0e-3, rtol=0.0):
      raise AdapterError("artifact joint stiffness disagrees with action contract")
    if not np.allclose(damping, self.damping, atol=1.0e-3, rtol=0.0):
      raise AdapterError("artifact joint damping disagrees with action contract")

  def to_physical(self, normalized: Sequence[float] | np.ndarray) -> np.ndarray:
    value = _vector("normalized action", normalized, 31)
    return self.offset + self.scale * value

  def from_physical(self, physical: Sequence[float] | np.ndarray) -> np.ndarray:
    value = _vector("physical action", physical, 31)
    return (value - self.offset) / self.scale


@dataclass(frozen=True, slots=True)
class PostStepBundle:
  """Post-action observation, evidence, and runtime physical timestamp."""

  evidence: Any
  timestamp: float
  observation: Any | None = None

  def __post_init__(self) -> None:
    if not np.isfinite(self.timestamp):
      raise AdapterError("post-step timestamp must be finite")


@dataclass(frozen=True, slots=True)
class VaeObservationBatch:
  """Aligned VAE observations for selected simulator rows."""

  state: WorldState
  reference: np.ndarray
  conditioning: np.ndarray
  motion_id: tuple[str, ...]
  reference_frame: np.ndarray
  reference_phase: np.ndarray
  physical_tilt_degrees: np.ndarray | None = None
  anchor_z_error: np.ndarray | None = None
  gravity_z_error: np.ndarray | None = None
  end_effector_z_error: np.ndarray | None = None

  def __post_init__(self) -> None:
    batch = self.state.count
    object.__setattr__(self, "reference", _matrix("reference", self.reference, 68))
    object.__setattr__(
      self, "conditioning", _matrix("conditioning", self.conditioning, 99)
    )
    if self.reference.shape[0] != batch or self.conditioning.shape[0] != batch:
      raise AdapterError("VAE batch state and inputs must have the same row count")
    if len(self.motion_id) != batch:
      raise AdapterError("VAE batch motion ids must match the row count")
    for name, value in (
      ("reference_frame", self.reference_frame),
      ("reference_phase", self.reference_phase),
      ("physical_tilt_degrees", self.physical_tilt_degrees),
      ("anchor_z_error", self.anchor_z_error),
      ("gravity_z_error", self.gravity_z_error),
      ("end_effector_z_error", self.end_effector_z_error),
    ):
      if value is None:
        continue
      array = np.asarray(value, dtype=np.float64).reshape(-1)
      if array.shape != (batch,) or not np.isfinite(array).all():
        raise AdapterError(f"{name} must be finite with shape [{batch}]")
      object.__setattr__(self, name, array.copy())
    frames = np.asarray(self.reference_frame, dtype=np.int64).reshape(-1)
    if frames.shape != (batch,) or np.any(frames < 0):
      raise AdapterError(f"reference_frame must be non-negative with shape [{batch}]")
    object.__setattr__(self, "reference_frame", frames.copy())

  @property
  def batch_size(self) -> int:
    return self.reference.shape[0]

  def select(self, indices: Any) -> "VaeObservationBatch":
    ids = _batch_indices(indices, self.batch_size)
    return type(self)(
      self.state.select(ids),
      self.reference[ids],
      self.conditioning[ids],
      tuple(self.motion_id[int(index)] for index in ids),
      self.reference_frame[ids],
      self.reference_phase[ids],
      None if self.physical_tilt_degrees is None else self.physical_tilt_degrees[ids],
      None if self.anchor_z_error is None else self.anchor_z_error[ids],
      None if self.gravity_z_error is None else self.gravity_z_error[ids],
      None if self.end_effector_z_error is None else self.end_effector_z_error[ids],
    )


@dataclass(frozen=True, slots=True)
class VaeObservationTensorBatch:
  """Device-native VAE inputs for one selected row batch.

  The tensors stay on the policy inference device, so the frozen VAE consumes
  them without the CPU observation round trip.  They carry exactly the packed
  ``reference``/``conditioning`` content of :class:`VaeObservationBatch`, and
  row order matches the caller's ``env_ids`` order.

  The optional row evidence (``state``, ``motion_id``, ``reference_frame``,
  ``reference_phase``) lets the collector record the required row fields from
  the same snapshot capture without copying the packed reference/conditioning
  tensors back to the host.  A batch built for inference alone leaves them
  empty.
  """

  reference: torch.Tensor
  conditioning: torch.Tensor
  state: WorldState | None = None
  motion_id: tuple[str, ...] = ()
  reference_frame: np.ndarray | None = None
  reference_phase: np.ndarray | None = None

  def __post_init__(self) -> None:
    for name, value, width in (
      ("reference", self.reference, 68),
      ("conditioning", self.conditioning, 99),
    ):
      if not isinstance(value, torch.Tensor):
        raise AdapterError(f"tensor {name} must be a torch.Tensor")
      if value.ndim != 2 or value.shape[1] != width or value.shape[0] == 0:
        raise AdapterError(f"tensor {name} must have shape [N, {width}] with N > 0")
      if value.dtype != torch.float32:
        raise AdapterError(f"tensor {name} must be float32")
    if self.reference.device != self.conditioning.device:
      raise AdapterError("tensor reference and conditioning must share a device")
    if self.reference.shape[0] != self.conditioning.shape[0]:
      raise AdapterError("tensor reference and conditioning rows must align")
    if (
      not torch.isfinite(self.reference).all()
      or not torch.isfinite(self.conditioning).all()
    ):
      raise AdapterError("tensor VAE observation must be finite")
    if self.state is None:
      return
    if self.reference_frame is None or self.reference_phase is None:
      raise AdapterError("tensor VAE row evidence requires frames and phases")
    batch = self.reference.shape[0]
    if self.state.count != batch or len(self.motion_id) != batch:
      raise AdapterError("tensor VAE row evidence must match the reference rows")
    frames = np.asarray(self.reference_frame, dtype=np.int64).reshape(-1)
    phases = np.asarray(self.reference_phase, dtype=np.float64).reshape(-1)
    if (
      frames.shape != (batch,)
      or np.any(frames < 0)
      or phases.shape != (batch,)
      or not np.isfinite(phases).all()
    ):
      raise AdapterError("tensor VAE row evidence frames must be finite and in range")
    object.__setattr__(self, "reference_frame", frames.copy())
    object.__setattr__(self, "reference_phase", phases.copy())

  @property
  def batch_size(self) -> int:
    return self.reference.shape[0]

  @property
  def device(self) -> torch.device:
    return self.reference.device


@dataclass(frozen=True, slots=True)
class PpoObservationBatch:
  """Aligned recovery-policy observations for selected simulator rows."""

  state: WorldState
  observation: np.ndarray
  motion_id: tuple[str, ...]
  reference_frame: np.ndarray
  reference_phase: np.ndarray
  anchor_z_error: np.ndarray | None = None
  gravity_z_error: np.ndarray | None = None
  end_effector_z_error: np.ndarray | None = None
  physical_tilt_degrees: np.ndarray | None = None

  def __post_init__(self) -> None:
    batch = self.state.count
    object.__setattr__(
      self, "observation", _matrix("PPO observation", self.observation, 102)
    )
    if self.observation.shape[0] != batch or len(self.motion_id) != batch:
      raise AdapterError("PPO batch fields must have the same row count")
    frames = np.asarray(self.reference_frame, dtype=np.int64).reshape(-1)
    if frames.shape != (batch,) or np.any(frames < 0):
      raise AdapterError(f"reference_frame must be non-negative with shape [{batch}]")
    object.__setattr__(self, "reference_frame", frames.copy())
    for name, value in (
      ("reference_phase", self.reference_phase),
      ("anchor_z_error", self.anchor_z_error),
      ("gravity_z_error", self.gravity_z_error),
      ("end_effector_z_error", self.end_effector_z_error),
      ("physical_tilt_degrees", self.physical_tilt_degrees),
    ):
      if value is None:
        continue
      array = np.asarray(value, dtype=np.float64).reshape(-1)
      if array.shape != (batch,) or not np.isfinite(array).all():
        raise AdapterError(f"{name} must be finite with shape [{batch}]")
      object.__setattr__(self, name, array.copy())

  @property
  def batch_size(self) -> int:
    return self.observation.shape[0]

  def select(self, indices: Any) -> "PpoObservationBatch":
    ids = _batch_indices(indices, self.batch_size)
    return type(self)(
      self.state.select(ids),
      self.observation[ids],
      tuple(self.motion_id[int(index)] for index in ids),
      self.reference_frame[ids],
      self.reference_phase[ids],
      None if self.anchor_z_error is None else self.anchor_z_error[ids],
      None if self.gravity_z_error is None else self.gravity_z_error[ids],
      None if self.end_effector_z_error is None else self.end_effector_z_error[ids],
      None if self.physical_tilt_degrees is None else self.physical_tilt_degrees[ids],
    )


@dataclass(frozen=True, slots=True)
class PostStepBundleBatch:
  """Evidence and physical timestamps aligned with a row batch."""

  evidence: Any
  timestamp: np.ndarray
  observation: VaeObservationBatch | PpoObservationBatch | None = None

  def __post_init__(self) -> None:
    values = np.asarray(self.timestamp, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
      raise AdapterError("post-step batch timestamps must be finite and non-empty")
    object.__setattr__(self, "timestamp", values.copy())
    if self.observation is not None and self.observation.batch_size != values.size:
      raise AdapterError("post-step observation and timestamp batches must align")

  @property
  def batch_size(self) -> int:
    return self.timestamp.size


@dataclass(frozen=True, slots=True)
class VaeObservation:
  """Actual pre-action VAE inputs and simulator state."""

  state: WorldState
  reference: np.ndarray
  conditioning: np.ndarray
  motion_id: str
  reference_frame: int
  reference_phase: float
  physical_tilt_degrees: float | None = None
  anchor_z_error: float | None = None
  gravity_z_error: float | None = None
  end_effector_z_error: float | None = None

  def __post_init__(self) -> None:
    object.__setattr__(self, "reference", _vector("reference", self.reference, 68))
    object.__setattr__(
      self, "conditioning", _vector("conditioning", self.conditioning, 99)
    )
    if not self.motion_id or self.reference_frame < 0:
      raise AdapterError("motion_id and reference_frame are invalid")
    present = [
      value
      for value in (
        self.reference_phase,
        self.physical_tilt_degrees,
        self.anchor_z_error,
        self.gravity_z_error,
        self.end_effector_z_error,
      )
      if value is not None
    ]
    if not np.isfinite(present).all():
      raise AdapterError("VAE physical/reference metrics must be finite when supplied")


@dataclass(frozen=True, slots=True)
class PpoObservation:
  """Recovery policy observation; it has no VAE latent field."""

  state: WorldState
  observation: np.ndarray
  motion_id: str
  reference_frame: int
  reference_phase: float
  anchor_z_error: float = 0.0
  gravity_z_error: float = 0.0
  end_effector_z_error: float = 0.0
  physical_tilt_degrees: float = 0.0

  def __post_init__(self) -> None:
    object.__setattr__(
      self, "observation", _vector("PPO observation", self.observation, 102)
    )
    metrics = (
      self.reference_phase,
      self.anchor_z_error,
      self.gravity_z_error,
      self.end_effector_z_error,
      self.physical_tilt_degrees,
    )
    if not np.isfinite(metrics).all():
      raise AdapterError("PPO physical/reference metrics must be finite")


class DiffusionEnvironment(Protocol):
  """Minimal no-reset handoff boundary consumed by :class:`DiffusionCollector`."""

  def reset(self, *, seed: int, motion_id: str, start_frame: int) -> None: ...
  def observe_vae(self) -> VaeObservation: ...
  def step(self, action: np.ndarray) -> Any: ...
  def observe_ppo(self) -> PpoObservation: ...
  def switch_to_ppo(self) -> None: ...
  def post_step_evidence(self, result: Any) -> PostStepBundle: ...
  def reset_after_trial(self) -> None: ...
  def capture_initial_state(self) -> Any: ...
  def restore_initial_state(self, snapshot: Any) -> None: ...


class DiffusionBatchEnvironment(Protocol):
  """Vectorized no-reset handoff boundary consumed by batched collection."""

  num_envs: int

  def reset_batch(self, specs: Sequence[Any]) -> None: ...
  def observe_vae_batch(self, env_ids: np.ndarray) -> VaeObservationBatch: ...
  def observe_ppo_batch(self, env_ids: np.ndarray) -> PpoObservationBatch: ...
  def step_batch(self, action: np.ndarray) -> Any: ...
  def post_step_evidence_batch(self, env_ids: np.ndarray) -> PostStepBundleBatch: ...
  def switch_to_ppo_batch(self, env_ids: np.ndarray) -> None: ...
  def capture_initial_state_batch(self, env_ids: np.ndarray) -> Any: ...
  def restore_initial_state_batch(
    self, env_ids: np.ndarray, snapshots: Any
  ) -> None: ...
  def reset_after_trial_batch(self, env_ids: np.ndarray) -> None: ...


class FrozenVaePolicy:
  """CPU-safe wrapper around the accepted ConditionalVAE inference path.

  ``action`` deliberately calls ``decode(mu, conditioning)``; sampled latent
  noise is never introduced during collection.
  """

  def __init__(
    self,
    model: Any,
    action_contract: ActionContract,
    artifact_metadata: Any | None = None,
  ) -> None:
    self.model = model
    self.action_contract = action_contract
    self.artifact_metadata = artifact_metadata
    self.artifact_path: Path | None = None
    self.artifact_sha256: str | None = None
    if artifact_metadata is not None:
      action_contract.verify_metadata(artifact_metadata)
    schema = getattr(model, "schema", None)

    if schema is None or schema.reference_dim != 68 or schema.conditioning_dim != 99:
      raise AdapterError("VAE model does not implement the frozen gravity schema")
    model.eval()
    parameters = getattr(model, "parameters", None)
    if parameters is not None:
      for parameter in parameters():
        parameter.requires_grad_(False)

  @classmethod
  def from_checkpoint(
    cls,
    checkpoint: str | Path,
    action_contract: ActionContract,
    *,
    teacher_id: str = "tennis_000",
    expected_teacher_hashes: dict[str, str] | None = None,
    artifact_metadata: Any | None = None,
    cohort_manifest: str | Path | None = None,
    cohort_manifest_sha256: str | None = None,
    contract: Any | None = None,
    device: str | torch.device = "cpu",
  ) -> "FrozenVaePolicy":
    """Load a pinned cohort member through its checked inference loader.

    The D0 VAE is a version-3 cohort checkpoint, not a version-1 single-teacher
    envelope.  The cohort manifest and selected teacher are therefore resolved
    before calling ``load_cohort_member_inference``; the selected teacher's real
    ONNX export supplies the action metadata shared by the frozen student.

    The cohort manifest, its expected digest and the pinned artifact identity
    default to the frozen three-clip revision.  A caller selecting another
    registered contract revision passes all three explicitly; the loaded
    checkpoint bytes are always compared against the selected contract.

    ``device`` is an explicit runtime input.  It defaults to ``cpu`` so the
    frozen CPU path is unchanged, and is validated before the checkpoint is
    read so an unusable inference device fails closed.
    """
    from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT
    from mjlab.tasks.tracking.distillation.checkpoint import (
      CheckpointValidationError,
      load_cohort_member_inference,
    )
    from mjlab.tasks.tracking.distillation.config import load_manifest, resolve_cohort

    selected_contract = contract if contract is not None else DEFAULT_CONTRACT
    selected_device = resolve_inference_device(device)
    repo_root = _repo_root()
    if cohort_manifest is None:
      manifest_path = repo_root / _MIXED_COHORT_MANIFEST
      expected_manifest_digest = _MIXED_COHORT_MANIFEST_SHA256
    else:
      manifest_path = Path(cohort_manifest)
      if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
      expected_manifest_digest = cohort_manifest_sha256
    if not manifest_path.is_file():
      raise AdapterError(f"frozen VAE cohort manifest is missing: {manifest_path}")
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if expected_manifest_digest is None:
      raise AdapterError(
        f"frozen VAE cohort manifest digest was not supplied: {manifest_path}"
      )
    if manifest_digest != expected_manifest_digest:
      raise AdapterError(f"frozen VAE cohort manifest hash mismatch: {manifest_path}")
    try:
      cohort = resolve_cohort(load_manifest(manifest_path, repo_root=repo_root))
      selected = cohort.teacher(teacher_id)
      inference = load_cohort_member_inference(
        checkpoint,
        cohort,
        teacher_id,
        device=selected_device,
      )
    except (CheckpointValidationError, ValueError, OSError) as exc:
      raise AdapterError(
        f"frozen VAE cohort member {teacher_id!r} could not be loaded: {exc}"
      ) from exc
    if inference.cohort.manifest_sha256 != manifest_digest:
      raise AdapterError("frozen VAE cohort manifest identity disagrees with config")
    if expected_teacher_hashes is not None:
      expected_hashes = {
        str(key): str(value) for key, value in expected_teacher_hashes.items()
      }
      if expected_hashes != inference.artifact_hashes:
        raise AdapterError("frozen VAE cohort teacher artifact hashes disagree")
    if inference.teacher_ids != cohort.teacher_ids:
      raise AdapterError("frozen VAE cohort teacher identity disagrees with config")
    if tuple(selected.actions.joint_names) != action_contract.joint_order:
      raise AdapterError("frozen VAE cohort joint order disagrees with action contract")
    if not np.allclose(
      np.asarray(selected.actions.joint_scales, dtype=np.float64),
      action_contract.scale,
      atol=1.0e-3,
      rtol=0.0,
    ):
      raise AdapterError(
        "frozen VAE cohort action scale disagrees with action contract"
      )
    if not selected.actions.uses_default_offset and not np.allclose(
      np.full(31, selected.actions.offset, dtype=np.float64),
      action_contract.offset,
      atol=1.0e-3,
      rtol=0.0,
    ):
      raise AdapterError(
        "frozen VAE cohort action offset disagrees with action contract"
      )
    metadata = _read_onnx_metadata(selected.onnx.path)
    if artifact_metadata is not None:
      action_contract.verify_metadata(artifact_metadata)
    action_contract.verify_metadata(metadata, require_actuator_metadata=True)
    policy = cls(inference.model, action_contract, metadata)
    policy.artifact_path = Path(checkpoint).resolve()
    policy.artifact_sha256 = hashlib.sha256(
      policy.artifact_path.read_bytes()
    ).hexdigest()
    if policy.artifact_sha256 != selected_contract.vae_sha256:
      raise AdapterError(
        "frozen VAE checkpoint hash does not match the selected contract artifact"
      )
    return policy

  def latent_and_action(
    self,
    reference: Sequence[float] | np.ndarray,
    conditioning: Sequence[float] | np.ndarray,
  ) -> tuple[np.ndarray, np.ndarray]:
    reference_tensor = torch.as_tensor(
      _vector("reference", reference, 68), dtype=torch.float32
    )[None]
    conditioning_tensor = torch.as_tensor(
      _vector("conditioning", conditioning, 99), dtype=torch.float32
    )[None]
    device = self._model_device()
    if device is not None:
      reference_tensor = reference_tensor.to(device)
      conditioning_tensor = conditioning_tensor.to(device)
    with torch.no_grad():
      mu, _ = self.model.encode(reference_tensor)
      action = self.model.decode(mu, conditioning_tensor)
    latent = mu[0].detach().cpu().numpy().astype(np.float64)
    clean = action[0].detach().cpu().numpy().astype(np.float64)
    if (
      latent.shape != (32,)
      or clean.shape != (31,)
      or not np.isfinite(latent).all()
      or not np.isfinite(clean).all()
    ):
      raise AdapterError("VAE produced an invalid latent or action")
    return latent, clean

  def action(self, observation: VaeObservation) -> tuple[np.ndarray, np.ndarray]:
    return self.latent_and_action(observation.reference, observation.conditioning)

  def _model_device(self) -> torch.device | None:
    """The frozen model's parameter device, or ``None`` for a parameterless model."""
    parameters = getattr(self.model, "parameters", None)
    if not callable(parameters):
      return None
    try:
      return next(cast(Any, parameters)()).device
    except StopIteration:
      return None

  def to_device(self, device: str | torch.device) -> "FrozenVaePolicy":
    """Move the frozen model for a batched rollout without changing scalar action."""
    mover = getattr(self.model, "to", None)
    if not callable(mover):
      raise AdapterError("VAE model cannot be moved to a batched inference device")
    mover(device)
    return self

  def batch_action(
    self, observation: VaeObservationBatch
  ) -> tuple[np.ndarray, np.ndarray]:
    """Run one VAE encode/decode call for all supplied rows."""
    reference = torch.as_tensor(observation.reference, dtype=torch.float32)
    conditioning = torch.as_tensor(observation.conditioning, dtype=torch.float32)
    device = self._model_device()
    if device is not None:
      reference = reference.to(device)
      conditioning = conditioning.to(device)
    with torch.no_grad():
      mu, _ = self.model.encode(reference)
      action = self.model.decode(mu, conditioning)
    latent = mu.detach().cpu().numpy().astype(np.float64)
    clean = action.detach().cpu().numpy().astype(np.float64)
    if latent.shape != (observation.batch_size, 32) or clean.shape != (
      observation.batch_size,
      31,
    ):
      raise AdapterError("batched VAE produced an invalid latent or action")
    if not np.isfinite(latent).all() or not np.isfinite(clean).all():
      raise AdapterError("batched VAE produced non-finite values")
    return latent, clean

  def batch_action_tensor(
    self, observation: VaeObservationTensorBatch
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode and decode one device-native observation batch.

    Returns the posterior mean (never a sampled latent) and the clean decoded
    action as float32 tensors on the model device.  The model stays frozen and
    no NumPy conversion is performed, so a device-native observation does not
    round-trip through the host.  The scalar and NumPy batch paths are
    unchanged.
    """
    if not isinstance(observation, VaeObservationTensorBatch):
      raise AdapterError("tensor VAE batch must be a VaeObservationTensorBatch")
    reference = observation.reference
    conditioning = observation.conditioning
    device = self._model_device()
    if device is not None:
      reference = reference.to(device)
      conditioning = conditioning.to(device)
    with torch.no_grad():
      latent, _ = self.model.encode(reference)
      action = self.model.decode(latent, conditioning)
    latent = latent.detach()
    action = action.detach()
    if latent.shape != (observation.batch_size, 32) or action.shape != (
      observation.batch_size,
      31,
    ):
      raise AdapterError("tensor VAE produced an invalid latent or action")
    if not torch.isfinite(latent).all() or not torch.isfinite(action).all():
      raise AdapterError("tensor VAE produced non-finite values")
    return latent, action


class RecoveryInference(Protocol):
  def __call__(self, observation: np.ndarray) -> np.ndarray: ...


def _default_ppo_observation_builder(
  observation: PpoObservation, previous_action: np.ndarray
) -> np.ndarray:
  """Patch the frozen actions field while preserving the zero command field."""
  model_observation = _vector("PPO observation", observation.observation, 102)
  model_observation[_PPO_ACTION_SLICE] = _vector("previous action", previous_action, 31)
  return model_observation


class FrozenPpoRecoveryPolicy:
  """Zero-command PPO recovery wrapper with an explicit previous-action state."""

  def __init__(
    self,
    inference: RecoveryInference,
    action_contract: ActionContract,
    observation_builder: Callable[[PpoObservation, np.ndarray], np.ndarray]
    | None = None,
    artifact_metadata: Any | None = None,
  ) -> None:
    self.inference = inference
    self.action_contract = action_contract
    self.artifact_metadata = artifact_metadata
    self.artifact_path: Path | None = None
    self.artifact_sha256: str | None = None
    if artifact_metadata is not None:
      action_contract.verify_metadata(artifact_metadata, require_actuator_metadata=True)
    self.observation_builder = observation_builder or _default_ppo_observation_builder
    self.previous_action = np.zeros(31, dtype=np.float64)

  @classmethod
  def from_onnx(
    cls,
    path: str | Path,
    action_contract: ActionContract,
    observation_builder: Callable[[PpoObservation, np.ndarray], np.ndarray]
    | None = None,
    artifact_metadata: Any | None = None,
  ) -> "FrozenPpoRecoveryPolicy":
    try:
      import onnxruntime as ort
    except ImportError as exc:
      raise AdapterError("ONNX recovery policy requires onnxruntime") from exc
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
      raise AdapterError("recovery ONNX must have one observation input and output")
    shape = inputs[0].shape
    if shape[-1] != 102 or outputs[0].shape[-1] != 31:
      raise AdapterError("recovery ONNX must be 102-D in and 31-D out")
    name, output = inputs[0].name, outputs[0].name
    metadata = dict(getattr(session.get_modelmeta(), "custom_metadata_map", {}))
    for key, value in tuple(metadata.items()):
      if isinstance(value, str):
        try:
          metadata[key] = json.loads(value)
        except json.JSONDecodeError:
          pass
    if not metadata:
      raise AdapterError("recovery ONNX has no verifiable action metadata")
    actual_metadata = dict(metadata)
    if artifact_metadata is not None:
      action_contract.verify_metadata(artifact_metadata, require_actuator_metadata=True)
    action_contract.verify_metadata(actual_metadata, require_actuator_metadata=True)
    static_batch = shape[0] if isinstance(shape[0], int) else None

    def inference(observation: np.ndarray) -> np.ndarray:
      values = np.asarray(observation, dtype=np.float32)
      if static_batch == 1 and values.shape[0] != 1:
        # Some frozen recovery graphs export a fixed [1, 102] input.  Preserve
        # batched collection by making one exact ONNX call per active recovery
        # row instead of passing an invalid larger batch to the graph.
        return np.concatenate(
          [np.asarray(session.run([output], {name: row[None]})[0]) for row in values],
          axis=0,
        )
      return np.asarray(session.run([output], {name: values})[0])

    policy = cls(
      inference,
      action_contract,
      observation_builder,
      actual_metadata,
    )
    policy.artifact_path = Path(path).resolve()
    policy.artifact_sha256 = hashlib.sha256(
      policy.artifact_path.read_bytes()
    ).hexdigest()
    return policy

  def reset(self) -> None:
    self.previous_action.fill(0.0)

  def action(self, observation: PpoObservation) -> np.ndarray:
    model_observation = _vector(
      "PPO observation",
      self.observation_builder(observation, self.previous_action),
      102,
    )

    raw = np.asarray(self.inference(model_observation[None]), dtype=np.float64).reshape(
      -1
    )
    raw = _vector("PPO action", raw, 31)
    # Previous action is in the policy's normalized decoder convention.  It is
    # updated only after inference, matching last-frame controller semantics.
    self.previous_action = raw.copy()
    return raw

  def batch_action(
    self, observations: PpoObservationBatch, previous_actions: np.ndarray
  ) -> np.ndarray:
    """Run one recovery-policy inference call for all supplied rows."""
    previous = _matrix("previous actions", previous_actions, 31)
    if previous.shape[0] != observations.batch_size:
      raise AdapterError("PPO previous-action rows must match observations")
    if self.observation_builder is _default_ppo_observation_builder:
      model_observation = observations.observation.copy()
      model_observation[:, _PPO_ACTION_SLICE] = previous
    else:
      model_observation = np.stack(
        [
          self.observation_builder(
            PpoObservation(
              observations.state.select(np.asarray([index])),
              observations.observation[index],
              observations.motion_id[index],
              int(observations.reference_frame[index]),
              float(observations.reference_phase[index]),
            ),
            previous[index],
          )
          for index in range(observations.batch_size)
        ]
      )
    raw = np.asarray(
      self.inference(model_observation.astype(np.float32)), dtype=np.float64
    )
    if raw.shape != (observations.batch_size, 31) or not np.isfinite(raw).all():
      raise AdapterError("batched PPO produced an invalid action")
    return raw.copy()

  def handoff_from_physical_batch(self, physical_action: np.ndarray) -> np.ndarray:
    """Convert batched physical endpoint actions to PPO policy units."""
    physical = _matrix("physical endpoint actions", physical_action, 31)
    return np.stack([self.action_contract.from_physical(row) for row in physical])

  def previous_from_physical(
    self, physical_action: Sequence[float] | np.ndarray
  ) -> None:
    """Convert an already-applied physical target into PPO policy units."""
    self.previous_action = self.action_contract.from_physical(physical_action)

  def handoff_from_physical(
    self, physical_action: Sequence[float] | np.ndarray
  ) -> None:
    """Name the endpoint handoff explicitly at the controller boundary."""
    self.previous_from_physical(physical_action)


@dataclass(frozen=True, slots=True)
class CallbackEnvironmentAdapter:
  """Concrete bridge for a real mjlab environment without changing its defaults."""

  reset_fn: Callable[..., None]
  vae_observe_fn: Callable[[], VaeObservation]
  step_fn: Callable[[np.ndarray], Any]
  ppo_observe_fn: Callable[[], PpoObservation]
  switch_fn: Callable[[], None]
  evidence_fn: Callable[[Any], Any]
  trial_reset_fn: Callable[[], None]
  capture_fn: Callable[[], Any] | None = None
  restore_fn: Callable[[Any], None] | None = None

  def reset(self, *, seed: int, motion_id: str, start_frame: int) -> None:
    self.reset_fn(seed=seed, motion_id=motion_id, start_frame=start_frame)

  def observe_vae(self) -> VaeObservation:
    return self.vae_observe_fn()

  def step(self, action: np.ndarray) -> Any:
    return self.step_fn(action)

  def observe_ppo(self) -> PpoObservation:
    return self.ppo_observe_fn()

  def switch_to_ppo(self) -> None:
    self.switch_fn()

  def post_step_evidence(self, result: Any) -> Any:
    return self.evidence_fn(result)

  def capture_initial_state(self) -> Any:
    if self.capture_fn is None:
      raise AdapterError("real adapter must provide initial-state snapshots")
    return self.capture_fn()

  def restore_initial_state(self, snapshot: Any) -> None:
    if self.restore_fn is None:
      raise AdapterError("real adapter must provide initial-state restoration")
    self.restore_fn(snapshot)

  def reset_after_trial(self) -> None:
    self.trial_reset_fn()


__all__ = [
  "ActionContract",
  "AdapterError",
  "CallbackEnvironmentAdapter",
  "DiffusionEnvironment",
  "FrozenPpoRecoveryPolicy",
  "FrozenVaePolicy",
  "PpoObservation",
  "PostStepBundle",
  "VaeObservation",
]
