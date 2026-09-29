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
from typing import Any, Callable, Mapping, Protocol, Sequence

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
  ) -> "FrozenVaePolicy":
    """Load the pinned M4 cohort member through its checked inference loader.

    The D0 VAE is a version-3 cohort checkpoint, not a version-1 single-teacher
    envelope.  The cohort manifest and selected teacher are therefore resolved
    before calling ``load_cohort_member_inference``; the selected teacher's real
    ONNX export supplies the action metadata shared by the frozen student.
    """
    from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT
    from mjlab.tasks.tracking.distillation.checkpoint import (
      CheckpointValidationError,
      load_cohort_member_inference,
    )
    from mjlab.tasks.tracking.distillation.config import load_manifest, resolve_cohort

    repo_root = _repo_root()
    manifest_path = repo_root / _MIXED_COHORT_MANIFEST
    if not manifest_path.is_file():
      raise AdapterError(f"frozen VAE cohort manifest is missing: {manifest_path}")
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if manifest_digest != _MIXED_COHORT_MANIFEST_SHA256:
      raise AdapterError(f"frozen VAE cohort manifest hash mismatch: {manifest_path}")
    try:
      cohort = resolve_cohort(load_manifest(manifest_path, repo_root=repo_root))
      selected = cohort.teacher(teacher_id)
      inference = load_cohort_member_inference(
        checkpoint,
        cohort,
        teacher_id,
        device="cpu",
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
    if policy.artifact_sha256 != DEFAULT_CONTRACT.vae_sha256:
      raise AdapterError("frozen VAE checkpoint hash does not match the D0 artifact")
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
    policy = cls(
      lambda observation: np.asarray(
        session.run([output], {name: observation.astype(np.float32)})[0]
      ),
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
