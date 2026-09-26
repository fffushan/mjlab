"""Typed manifest and compatibility validation for distillation teachers.

M1 scope: pair every frozen teacher checkpoint with its reference motion and
saved training configuration, then prove the pair is internally compatible
before any inference or distillation work. Nothing here builds a simulator, a
PPO algorithm, or a critic, and no callable named by a manifest or an agent
configuration is ever imported.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mjlab.utils.lab_api.string import resolve_matching_names_values
from mjlab.utils.os import load_saved_yaml

MANIFEST_VERSION = 1
"""Supported distillation manifest schema version."""

_ONNX_METADATA_TOLERANCE = 1e-3
"""Tolerance for ONNX metadata floats: the exporter formats them with ``%.3f``."""

_REFERENCE_FPS_RTOL = 1e-4
"""Relative tolerance when matching the control rate against the reference FPS."""

_ACTOR_STATE_DICT_PREFIXES = ("mlp.", "obs_normalizer.", "distribution.")
_SUPPORTED_ACTOR_CLASS = "MLPModel"
_SUPPORTED_DISTRIBUTION_CLASS = "GaussianDistribution"
_EXCLUDED_ENV_FIELDS = ("commands.motion.motion_file",)
"""Saved environment fields excluded from cross-teacher equality.

Every other saved environment field must agree between teachers, so a cohort
cannot silently mix observation, action, randomization, or timing contracts.
"""


class DistillationError(ValueError):
  """A distillation manifest or teacher contract is invalid."""


class UnsupportedTeacherError(DistillationError):
  """The artifact is well-formed but outside the M1-supported teacher scope."""


class MissingValidationDependencyError(RuntimeError):
  """A CPU validation dependency is not installed."""


def require_onnx() -> Any:
  """Import ``onnx`` for reading original teacher exports, or fail clearly."""
  try:
    import onnx
  except ImportError as exc:
    raise MissingValidationDependencyError(
      "Reading original teacher ONNX exports requires the 'onnx' package. "
      "Install the development dependency group with `uv sync --group dev`."
    ) from exc
  return onnx


def require_onnxruntime() -> Any:
  """Import ``onnxruntime`` for CPU parity checks, or fail clearly."""
  try:
    import onnxruntime
  except ImportError as exc:
    raise MissingValidationDependencyError(
      "Native-vs-ONNX teacher parity requires the 'onnxruntime' package. "
      "Install the development dependency group with `uv sync --group dev`."
    ) from exc
  return onnxruntime


# Manifest schema.


@dataclass(frozen=True, slots=True)
class TeacherEntry:
  """One manifest entry: a checkpoint plus the artifacts that identify it."""

  id: str
  checkpoint: Path
  motion: Path
  env_config: Path
  agent_config: Path
  onnx: Path
  sampling_weight: float

  def paths(self) -> dict[str, Path]:
    """Return the artifact paths keyed by role."""
    return {
      "checkpoint": self.checkpoint,
      "motion": self.motion,
      "env_config": self.env_config,
      "agent_config": self.agent_config,
      "onnx": self.onnx,
    }


@dataclass(frozen=True, slots=True)
class Manifest:
  """A resolved, validated distillation manifest."""

  path: Path
  sha256: str
  repo_root: Path
  version: int
  name: str
  robot: str
  base_task: str | None
  teachers: tuple[TeacherEntry, ...]

  def teacher(self, teacher_id: str) -> TeacherEntry:
    for entry in self.teachers:
      if entry.id == teacher_id:
        return entry
    raise DistillationError(f"Manifest has no teacher with id {teacher_id!r}")


def load_manifest(
  manifest_path: Path | str, repo_root: Path | str | None = None
) -> Manifest:
  """Load and validate a distillation manifest.

  Relative manifest and artifact paths resolve against ``repo_root``, which
  defaults to the current working directory. Absolute paths are used as given;
  no machine layout is assumed by the library.
  """
  import yaml

  root = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
  path = Path(manifest_path)
  if not path.is_absolute():
    path = root / path
  path = path.resolve()
  if not path.is_file():
    raise DistillationError(f"Manifest file not found: {path}")

  try:
    raw = yaml.safe_load(path.read_text())
  except yaml.YAMLError as exc:
    raise DistillationError(
      f"Manifest {path} must be plain YAML without python tags: {exc}"
    ) from exc
  if not isinstance(raw, Mapping):
    raise DistillationError(f"Manifest {path} must be a mapping")

  _reject_unknown_keys(
    raw, ("version", "name", "robot", "base_task", "teachers"), "manifest"
  )
  version = _require_int(raw.get("version"), "manifest.version")
  if version != MANIFEST_VERSION:
    raise DistillationError(
      f"Manifest {path} has version {version}; supported version is {MANIFEST_VERSION}"
    )
  name = _require_str(raw.get("name"), "manifest.name")
  robot = _require_str(raw.get("robot"), "manifest.robot")
  base_task = raw.get("base_task")
  if base_task is not None:
    base_task = _require_str(base_task, "manifest.base_task")

  raw_teachers = raw.get("teachers")
  if not isinstance(raw_teachers, list) or not raw_teachers:
    raise DistillationError("manifest.teachers must be a non-empty list")

  entries: list[TeacherEntry] = []
  seen: set[str] = set()
  missing: list[str] = []
  for index, raw_teacher in enumerate(raw_teachers):
    where = f"manifest.teachers[{index}]"
    if not isinstance(raw_teacher, Mapping):
      raise DistillationError(f"{where} must be a mapping")
    _reject_unknown_keys(
      raw_teacher,
      (
        "id",
        "checkpoint",
        "motion",
        "env_config",
        "agent_config",
        "onnx",
        "sampling_weight",
      ),
      where,
    )
    teacher_id = _require_str(raw_teacher.get("id"), f"{where}.id")
    if teacher_id in seen:
      raise DistillationError(f"{where}.id {teacher_id!r} is not unique")
    seen.add(teacher_id)
    weight = _require_float(
      raw_teacher.get("sampling_weight", 1.0), f"{where}.sampling_weight"
    )
    if not math.isfinite(weight) or weight <= 0.0:
      raise DistillationError(
        f"{where}.sampling_weight must be finite and positive, got {weight!r}"
      )
    resolved: dict[str, Path] = {}
    for field in ("checkpoint", "motion", "env_config", "agent_config", "onnx"):
      resolved[field] = _resolve_path(raw_teacher.get(field), root, f"{where}.{field}")
      if not resolved[field].is_file():
        missing.append(f"{where}.{field} -> {resolved[field]}")
    entries.append(TeacherEntry(id=teacher_id, sampling_weight=weight, **resolved))

  if missing:
    raise DistillationError("Missing teacher artifacts:\n  " + "\n  ".join(missing))

  return Manifest(
    path=path,
    sha256=sha256_file(path),
    repo_root=root,
    version=version,
    name=name,
    robot=robot,
    base_task=base_task,
    teachers=tuple(entries),
  )


def sha256_file(path: Path) -> str:
  """Return the SHA-256 digest of ``path``."""
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


# Resolved contract.


@dataclass(frozen=True, slots=True)
class ActorArchitecture:
  """Architecture of one frozen feedforward MLP actor."""

  class_name: str
  hidden_dims: tuple[int, ...]
  activation: str
  obs_normalization: bool
  obs_groups: tuple[str, ...]
  distribution_class_name: str | None
  distribution_cfg: dict[str, Any] | None
  obs_dim: int
  action_dim: int


@dataclass(frozen=True, slots=True)
class NoiseSpec:
  """Saved observation corruption parameters for one term, excluding runtime state."""

  operation: str
  params: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ObservationTerm:
  """One saved actor observation term."""

  name: str
  func: str
  width: int
  scale: tuple[float, ...] | None
  clip: tuple[float, float] | None
  history_length: int
  flatten_history_dim: bool
  noise: NoiseSpec | None
  delay_min_lag: int
  delay_max_lag: int
  delay_hold_prob: float
  delay_group: str | None
  params: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ObservationSchema:
  """Ordered actor observation schema with declared per-term widths."""

  group: str
  terms: tuple[ObservationTerm, ...]
  total_dim: int
  enable_corruption: bool

  @property
  def names(self) -> tuple[str, ...]:
    return tuple(term.name for term in self.terms)

  @property
  def widths(self) -> tuple[int, ...]:
    return tuple(term.width for term in self.terms)


@dataclass(frozen=True, slots=True)
class ActionSpec:
  """Saved joint-position action contract, resolved onto the exported joint order."""

  term: str
  joint_names: tuple[str, ...]
  joint_scales: tuple[float, ...]
  offset: float
  uses_default_offset: bool
  clip: None
  dim: int


@dataclass(frozen=True, slots=True)
class ControlSpec:
  """Control cadence implied by the saved environment configuration."""

  sim_timestep: float
  decimation: int
  control_period_s: float
  control_hz: float


@dataclass(frozen=True, slots=True)
class ReferenceSpec:
  """Reference motion contract."""

  motion_path: Path
  declared_motion_file: str
  frames: int
  fps: float
  joint_dim: int
  duration_s: float
  array_shapes: dict[str, tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class SensorDeclaration:
  """Declared sensor source of one actor observation term."""

  term: str
  sensor_name: str


@dataclass(frozen=True, slots=True)
class TensorInfo:
  """A static-shape ONNX graph input or output."""

  name: str
  shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class OnnxArtifact:
  """Read-only view of an original teacher ONNX export."""

  path: Path
  metadata: dict[str, str]
  inputs: tuple[TensorInfo, ...]
  outputs: tuple[TensorInfo, ...]
  initializers: dict[str, np.ndarray]
  constants: dict[str, np.ndarray]

  def tensor(self, name: str) -> np.ndarray | None:
    """Return a named graph tensor, from an initializer or a ``Constant`` node."""
    if name in self.initializers:
      return self.initializers[name]
    return self.constants.get(name)

  def reference_tensor(self, base_name: str) -> tuple[str, np.ndarray] | None:
    """Find the embedded reference tensor named ``base_name``.

    The exporter sometimes appends a numeric suffix to buffer-backed
    initializers (``joint_pos.1``), so a unique base-name match is accepted.
    """
    matches = [
      name
      for name in list(self.initializers) + list(self.constants)
      if name == base_name or name.split(".")[0] == base_name
    ]
    if len(matches) != 1:
      return None
    return matches[0], self.tensor(matches[0])  # type: ignore[return-value]

  def output(self, name: str) -> TensorInfo | None:
    for info in self.outputs:
      if info.name == name:
        return info
    return None

  def graph_input(self, name: str) -> TensorInfo | None:
    for info in self.inputs:
      if info.name == name:
        return info
    return None


@dataclass(frozen=True, slots=True)
class ResolvedTeacher:
  """One teacher with all artifacts resolved and validated."""

  entry: TeacherEntry
  hashes: dict[str, str]
  env_config: dict[str, Any]
  agent_config: dict[str, Any]
  actor: ActorArchitecture
  actor_state_dict: dict[str, torch.Tensor]
  observations: ObservationSchema
  actions: ActionSpec
  control: ControlSpec
  reference: ReferenceSpec
  onnx: OnnxArtifact
  anchor_body_name: str
  body_names: tuple[str, ...]
  lookahead_s: float
  sensors: tuple[SensorDeclaration, ...]
  unverified: tuple[str, ...]

  @property
  def id(self) -> str:
    return self.entry.id


@dataclass(frozen=True, slots=True)
class CohortContract:
  """Cross-teacher shared contract plus the per-teacher resolutions."""

  manifest: Manifest
  actor: ActorArchitecture
  observations: ObservationSchema
  actions: ActionSpec
  control: ControlSpec
  anchor_body_name: str
  body_names: tuple[str, ...]
  fps: float
  lookahead_s: float
  teachers: tuple[ResolvedTeacher, ...]
  excluded_env_fields: tuple[str, ...]
  unverified: tuple[str, ...]

  @property
  def teacher_ids(self) -> tuple[str, ...]:
    return tuple(teacher.id for teacher in self.teachers)

  def teacher(self, teacher_id: str) -> ResolvedTeacher:
    for teacher in self.teachers:
      if teacher.id == teacher_id:
        return teacher
    raise DistillationError(f"Cohort has no teacher with id {teacher_id!r}")


def load_actor_state_dict(path: Path) -> dict[str, torch.Tensor]:
  """Load the frozen-actor tensors of a training checkpoint.

  The current RSL-RL actor format is supported: ``actor_state_dict`` containing
  ``mlp.*``, ``obs_normalizer.*`` and ``distribution.*`` entries. Recurrent,
  CNN, and legacy ``model_state_dict`` checkpoints are rejected explicitly
  rather than silently reinterpreted.
  """
  try:
    # weights_only keeps a checkpoint from instantiating arbitrary Python
    # objects while loading.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
  except Exception as exc:
    raise DistillationError(
      f"Could not load checkpoint {path} with weights_only=True: {exc}"
    ) from exc
  if not isinstance(checkpoint, dict):
    raise DistillationError(f"Checkpoint {path} does not contain a dictionary")

  if "model_state_dict" in checkpoint:
    raise UnsupportedTeacherError(
      f"Checkpoint {path} uses the legacy rsl-rl ``model_state_dict`` format. "
      "M1 loads the current ``actor_state_dict`` format only; migrate it first "
      "with MjlabOnPolicyRunner.load instead of reinterpreting it here."
    )
  state = checkpoint.get("actor_state_dict")
  if not isinstance(state, dict) or not state:
    raise DistillationError(
      f"Checkpoint {path} has no non-empty 'actor_state_dict' entry "
      f"(found {sorted(checkpoint)})"
    )

  state_dict: dict[str, torch.Tensor] = {}
  for key, value in state.items():
    if not isinstance(key, str) or not isinstance(value, torch.Tensor):
      raise DistillationError(
        f"Checkpoint {path} actor_state_dict has a non-tensor entry {key!r}"
      )
    state_dict[key] = value

  # rsl-rl 4.x named the action-distribution parameters 'std'/'log_std'.
  if "std" in state_dict:
    state_dict["distribution.std_param"] = state_dict.pop("std")
  if "log_std" in state_dict:
    state_dict["distribution.log_std_param"] = state_dict.pop("log_std")

  unexpected = sorted(
    key for key in state_dict if not key.startswith(_ACTOR_STATE_DICT_PREFIXES)
  )
  if unexpected:
    raise UnsupportedTeacherError(
      f"Checkpoint {path} has unsupported actor entries {unexpected}; M1 supports "
      "feedforward MLPModel actors with an observation normalizer only."
    )
  return state_dict


def resolve_cohort(manifest: Manifest) -> CohortContract:
  """Resolve and cross-validate every teacher of a manifest.

  Raises:
    DistillationError: if any artifact, dimension, timing, or cross-teacher
      contract check fails.
  """
  teachers = tuple(_resolve_teacher(entry) for entry in manifest.teachers)
  first = teachers[0]

  for teacher in teachers[1:]:
    _require_same_contract(first, teacher)

  _require_equal_saved_configs(teachers)

  unverified = list(first.unverified)
  if not any("body reference" in note for note in unverified):
    unverified.append(
      "ONNX embedded body references are not compared with the NPZ: the NPZ has no "
      "body names, so the exported subset of tracked bodies cannot be identified."
    )
  unverified.append(
    "Physical sensor sites/frames are declared only; resolving them needs the robot "
    "asset, which M1 does not load."
  )

  return CohortContract(
    manifest=manifest,
    actor=first.actor,
    observations=first.observations,
    actions=first.actions,
    control=first.control,
    anchor_body_name=first.anchor_body_name,
    body_names=first.body_names,
    fps=first.reference.fps,
    lookahead_s=first.lookahead_s,
    teachers=teachers,
    excluded_env_fields=_EXCLUDED_ENV_FIELDS,
    unverified=tuple(dict.fromkeys(unverified)),
  )


def _resolve_teacher(entry: TeacherEntry) -> ResolvedTeacher:
  agent_config = _require_mapping(
    load_saved_yaml(entry.agent_config), f"{entry.id} agent config"
  )
  env_config = _require_mapping(
    load_saved_yaml(entry.env_config), f"{entry.id} env config"
  )
  actor_state_dict = load_actor_state_dict(entry.checkpoint)
  actor = _resolve_actor(entry, agent_config, actor_state_dict)
  control = _resolve_control(entry, env_config)
  onnx = load_onnx_artifact(entry.onnx, f"{entry.id} ONNX export")
  unverified: list[str] = []

  motion_cfg = _saved_motion_cfg(entry, env_config)
  lookahead_s = _require_float(
    motion_cfg.get("lookahead_s", 0.0), f"{entry.id} env.commands.motion.lookahead_s"
  )
  if not math.isfinite(lookahead_s) or lookahead_s < 0.0:
    raise DistillationError(f"{entry.id} has invalid lookahead_s {lookahead_s!r}")

  observations, sensors = _resolve_observations(
    entry, env_config, onnx, actor.action_dim, lookahead_s
  )
  actions = _resolve_actions(entry, env_config, onnx, actor.action_dim)
  _require_onnx_interface(entry, onnx, actor.obs_dim, actor.action_dim)
  reference = _resolve_reference(entry, env_config, actor.action_dim)
  _require_reference_cadence(entry, reference, control)

  anchor_body_name = _require_str(
    onnx.metadata.get("anchor_body_name"), f"{entry.id} ONNX anchor_body_name"
  )
  body_names = _parse_csv(
    _require_str(onnx.metadata.get("body_names"), f"{entry.id} ONNX body_names")
  )
  if len(body_names) != len(set(body_names)) or not body_names:
    raise DistillationError(f"{entry.id} ONNX body_names must be non-empty and unique")
  _require_declared_scene_matches_export(
    entry, env_config, anchor_body_name, body_names
  )

  if find_normalizer_divisor(onnx, actor.obs_dim) is None:
    unverified.append(
      "ONNX observation-normalizer scale could not be identified; only the normalizer "
      "mean was checked against the checkpoint."
    )

  return ResolvedTeacher(
    entry=entry,
    hashes={role: sha256_file(path) for role, path in entry.paths().items()},
    env_config=env_config,
    agent_config=agent_config,
    actor=actor,
    actor_state_dict=actor_state_dict,
    observations=observations,
    actions=actions,
    control=control,
    reference=reference,
    onnx=onnx,
    anchor_body_name=anchor_body_name,
    body_names=body_names,
    lookahead_s=lookahead_s,
    sensors=sensors,
    unverified=tuple(unverified),
  )


def _require_declared_scene_matches_export(
  entry: TeacherEntry,
  env_config: Mapping[str, Any],
  anchor_body_name: str,
  body_names: tuple[str, ...],
) -> None:
  """Require the saved command scene to match this teacher's own export metadata.

  Cross-teacher equality alone cannot catch an edit applied to every saved
  configuration, so the anchor and the tracked body list/order are checked
  against the export for each teacher independently.
  """
  motion_cfg = _saved_motion_cfg(entry, env_config)
  declared_anchor = _require_str(
    motion_cfg.get("anchor_body_name"),
    f"{entry.id} env.commands.motion.anchor_body_name",
  )
  if declared_anchor != anchor_body_name:
    raise DistillationError(
      f"{entry.id} saved anchor_body_name {declared_anchor!r} disagrees with the ONNX "
      f"export {anchor_body_name!r}"
    )
  declared_bodies = motion_cfg.get("body_names")
  if not isinstance(declared_bodies, (list, tuple)) or not declared_bodies:
    raise DistillationError(
      f"{entry.id} env.commands.motion.body_names must be a non-empty list"
    )
  declared_body_names = tuple(
    _require_str(body, f"{entry.id} env.commands.motion.body_names")
    for body in declared_bodies
  )
  if declared_body_names != body_names:
    raise DistillationError(
      f"{entry.id} saved tracked bodies {declared_body_names} disagree with the ONNX "
      f"export {body_names}"
    )


def find_normalizer_divisor(onnx: OnnxArtifact, obs_dim: int) -> np.ndarray | None:
  """Return the exported normalizer divisor, when it is unambiguous.

  ``EmpiricalNormalization`` normalizes with ``(x - mean) / (std + eps)``, and
  the graph folds ``std + eps`` into a single ``[1, obs_dim]`` constant. Any
  other ``[1, obs_dim]`` initializer names are exported parameters.
  """
  candidates = [
    array
    for name, array in onnx.initializers.items()
    if array.shape == (1, obs_dim) and not name.startswith("policy.")
  ]
  return candidates[0] if len(candidates) == 1 else None


def _resolve_actor(
  entry: TeacherEntry,
  agent_config: Mapping[str, Any],
  actor_state_dict: Mapping[str, torch.Tensor],
) -> ActorArchitecture:
  where = f"{entry.id} agent config"
  clip_actions = agent_config.get("clip_actions")
  if clip_actions is not None:
    raise UnsupportedTeacherError(
      f"{where}.clip_actions is {clip_actions!r}; M1 labels raw actor outputs, which "
      "the runner action clip would replace before execution"
    )
  actor_cfg = _require_mapping(agent_config.get("actor"), f"{where}.actor")
  class_name = _require_str(actor_cfg.get("class_name"), f"{where}.actor.class_name")
  if class_name != _SUPPORTED_ACTOR_CLASS:
    raise UnsupportedTeacherError(
      f"{where}.actor.class_name is {class_name!r}; M1 supports "
      f"{_SUPPORTED_ACTOR_CLASS!r} only"
    )
  if actor_cfg.get("rnn_type") is not None:
    raise UnsupportedTeacherError(
      f"{where}.actor.rnn_type is {actor_cfg['rnn_type']!r}; recurrent teachers are "
      "out of M1 scope"
    )
  if actor_cfg.get("cnn_cfg") is not None:
    raise UnsupportedTeacherError(
      f"{where}.actor.cnn_cfg is set; CNN teachers are out of M1 scope"
    )

  hidden_dims = _require_positive_int_sequence(
    actor_cfg.get("hidden_dims"), f"{where}.actor.hidden_dims"
  )
  activation = _require_str(actor_cfg.get("activation"), f"{where}.actor.activation")
  obs_normalization = _require_bool(
    actor_cfg.get("obs_normalization"), f"{where}.actor.obs_normalization"
  )
  distribution_cfg = _resolve_distribution_cfg(actor_cfg.get("distribution_cfg"), where)

  obs_groups = _require_mapping(agent_config.get("obs_groups"), f"{where}.obs_groups")
  actor_groups = obs_groups.get("actor")
  if not isinstance(actor_groups, (list, tuple)) or not actor_groups:
    raise DistillationError(f"{where}.obs_groups.actor must be a non-empty list")
  group_names = tuple(
    _require_str(group, f"{where}.obs_groups.actor") for group in actor_groups
  )
  if len(group_names) != 1:
    raise UnsupportedTeacherError(
      f"{where}.obs_groups.actor is {group_names}; M1 labels flat [B, D] observation "
      "vectors, so a single concatenated actor observation group is required"
    )

  mlp = _resolve_mlp_shapes(entry, actor_state_dict, hidden_dims)
  obs_dim = mlp["input_dim"]
  action_dim = mlp["output_dim"]

  if obs_normalization:
    mean = actor_state_dict.get("obs_normalizer._mean")
    if mean is None or tuple(mean.shape) != (1, obs_dim):
      raise DistillationError(
        f"{entry.id} checkpoint declares observation normalization but has no "
        f"obs_normalizer._mean of shape (1, {obs_dim})"
      )
  if distribution_cfg is not None:
    # GaussianDistribution names its learnable spread 'std_param' or
    # 'log_std_param' depending on the parameterization space.
    parameter = (
      "distribution.std_param"
      if distribution_cfg["std_type"] == "scalar"
      else "distribution.log_std_param"
    )
    if parameter not in actor_state_dict:
      raise DistillationError(
        f"{entry.id} agent config expects {parameter} in actor_state_dict "
        f"(found {sorted(k for k in actor_state_dict if k.startswith('distribution.'))})"
      )

  return ActorArchitecture(
    class_name=class_name,
    hidden_dims=hidden_dims,
    activation=activation,
    obs_normalization=obs_normalization,
    obs_groups=group_names,
    distribution_class_name=(
      None if distribution_cfg is None else str(distribution_cfg["class_name"])
    ),
    distribution_cfg=distribution_cfg,
    obs_dim=obs_dim,
    action_dim=action_dim,
  )


def _resolve_distribution_cfg(value: Any, where: str) -> dict[str, Any] | None:
  if value is None:
    return None
  cfg = _require_mapping(value, f"{where}.actor.distribution_cfg")
  class_name = _require_str(
    cfg.get("class_name"), f"{where}.actor.distribution_cfg.class_name"
  )
  if class_name != _SUPPORTED_DISTRIBUTION_CLASS:
    raise UnsupportedTeacherError(
      f"{where}.actor.distribution_cfg.class_name is {class_name!r}; M1 supports "
      f"{_SUPPORTED_DISTRIBUTION_CLASS!r} only"
    )
  std_type = _require_str(
    cfg.get("std_type", "scalar"), f"{where}.actor.distribution_cfg.std_type"
  )
  if std_type not in ("scalar", "log"):
    raise DistillationError(
      f"{where}.actor.distribution_cfg.std_type must be 'scalar' or 'log', "
      f"got {std_type!r}"
    )
  init_std = _require_float(
    cfg.get("init_std", 1.0), f"{where}.actor.distribution_cfg.init_std"
  )
  if not math.isfinite(init_std) or init_std <= 0.0:
    raise DistillationError(
      f"{where}.actor.distribution_cfg.init_std must be finite and positive"
    )
  return {"class_name": class_name, "std_type": std_type, "init_std": init_std}


def _resolve_mlp_shapes(
  entry: TeacherEntry,
  actor_state_dict: Mapping[str, torch.Tensor],
  hidden_dims: tuple[int, ...],
) -> dict[str, int]:
  """Check the checkpoint MLP weight chain against the declared hidden dimensions."""
  weights: dict[int, torch.Tensor] = {}
  for key, value in actor_state_dict.items():
    if key.startswith("mlp.") and key.endswith(".weight"):
      weights[int(key.split(".")[1])] = value
  expected_indices = list(range(0, 2 * len(hidden_dims) + 1, 2))
  if sorted(weights) != expected_indices:
    raise UnsupportedTeacherError(
      f"{entry.id} checkpoint mlp weights are at indices {sorted(weights)}, but "
      f"hidden_dims {hidden_dims} implies {expected_indices}"
    )

  input_dim = int(weights[0].shape[1])
  shape_chain = [input_dim, *(int(weights[i].shape[0]) for i in expected_indices)]
  if tuple(shape_chain[1:-1]) != hidden_dims:
    raise DistillationError(
      f"{entry.id} checkpoint MLP layers are {shape_chain}, which disagrees with the "
      f"saved hidden_dims {hidden_dims}"
    )
  for position, index in enumerate(expected_indices[1:], start=1):
    if int(weights[index].shape[1]) != shape_chain[position]:
      raise DistillationError(
        f"{entry.id} checkpoint MLP layer {index} expects {weights[index].shape[1]} "
        f"inputs but the previous layer outputs {shape_chain[position]}"
      )
  return {"input_dim": input_dim, "output_dim": shape_chain[-1]}


def _resolve_control(entry: TeacherEntry, env_config: Mapping[str, Any]) -> ControlSpec:
  decimation = _require_int(env_config.get("decimation"), f"{entry.id} env.decimation")
  if decimation <= 0:
    raise DistillationError(f"{entry.id} env.decimation must be positive")
  sim = _require_mapping(env_config.get("sim"), f"{entry.id} env.sim")
  mujoco = _require_mapping(sim.get("mujoco"), f"{entry.id} env.sim.mujoco")
  timestep = _require_float(
    mujoco.get("timestep"), f"{entry.id} env.sim.mujoco.timestep"
  )
  if not math.isfinite(timestep) or timestep <= 0.0:
    raise DistillationError(
      f"{entry.id} env.sim.mujoco.timestep must be finite and positive"
    )
  period = timestep * decimation
  return ControlSpec(
    sim_timestep=timestep,
    decimation=decimation,
    control_period_s=period,
    control_hz=1.0 / period,
  )


# Observation schema.
#
# Declared widths of the saved actor observation terms. Exact term-function
# names are required so an unrecognized observation cannot be silently
# accepted with a guessed width. Sensor widths are declared from the saved
# sensor name, not verified against the robot asset.
def _observation_width(
  where: str, func: str, params: Mapping[str, Any], joint_dim: int, lookahead_s: float
) -> int:
  if func == "mjlab.envs.mdp.observations.generated_commands":
    if params.get("command_name") != "motion":
      raise UnsupportedTeacherError(
        f"{where} uses command {params.get('command_name')!r}; M1 supports the "
        "single 'motion' tracking command only"
      )
    # MotionCommand.command concatenates reference joint positions and velocities.
    return 2 * joint_dim
  if func == "mjlab.tasks.tracking.mdp.observations.motion_lookahead":
    # Disabled lookahead returns a zero-width tensor.
    return 2 * joint_dim if lookahead_s > 0.0 else 0
  if func == "mjlab.tasks.tracking.mdp.observations.motion_anchor_ori_b":
    # First two columns of the 3x3 anchor orientation-error matrix.
    return 6
  if func == "mjlab.envs.mdp.observations.builtin_sensor":
    sensor_name = _require_str(params.get("sensor_name"), f"{where} sensor_name")
    if sensor_name.endswith(("_ang_vel", "_lin_vel")):
      return 3
    raise UnsupportedTeacherError(
      f"{where} reads builtin sensor {sensor_name!r}, whose width M1 does not declare"
    )
  if func in (
    "mjlab.envs.mdp.observations.joint_pos_rel",
    "mjlab.envs.mdp.observations.joint_vel_rel",
    "mjlab.envs.mdp.observations.last_action",
  ):
    return joint_dim
  raise UnsupportedTeacherError(
    f"{where} uses observation term function {func!r}, which M1 does not support"
  )


def _resolve_observations(
  entry: TeacherEntry,
  env_config: Mapping[str, Any],
  onnx: OnnxArtifact,
  joint_dim: int,
  lookahead_s: float,
) -> tuple[ObservationSchema, tuple[SensorDeclaration, ...]]:
  observed = onnx.graph_input("obs")
  if observed is None:
    raise DistillationError(f"{entry.id} ONNX export has no 'obs' input")
  obs_dim = observed.shape[-1]

  observations_cfg = _require_mapping(
    env_config.get("observations"), f"{entry.id} env.observations"
  )
  actor_cfg = _require_mapping(
    observations_cfg.get("actor"), f"{entry.id} env.observations.actor"
  )
  terms_cfg = _require_mapping(
    actor_cfg.get("terms"), f"{entry.id} actor observation terms"
  )
  if not terms_cfg:
    raise DistillationError(f"{entry.id} actor observation terms are empty")

  terms: list[ObservationTerm] = []
  sensors: list[SensorDeclaration] = []
  for name, raw_term in terms_cfg.items():
    where = f"{entry.id} actor observation {name!r}"
    term_cfg = _require_mapping(raw_term, where)
    func = _require_str(term_cfg.get("func"), f"{where}.func")
    params = _require_mapping(term_cfg.get("params", {}), f"{where}.params")
    history_length = _require_int(
      term_cfg.get("history_length", 0), f"{where}.history_length"
    )
    if history_length != 0:
      raise UnsupportedTeacherError(
        f"{where}.history_length is {history_length}; M1 supports history-free actor "
        "observation terms only"
      )
    width = _observation_width(where, func, params, joint_dim, lookahead_s)
    terms.append(
      ObservationTerm(
        name=name,
        func=func,
        width=width,
        scale=_resolve_scale(term_cfg.get("scale"), where),
        clip=_resolve_clip(term_cfg.get("clip"), where),
        history_length=history_length,
        flatten_history_dim=_require_bool(
          term_cfg.get("flatten_history_dim", True), f"{where}.flatten_history_dim"
        ),
        noise=_resolve_noise(term_cfg.get("noise"), where),
        delay_min_lag=_require_int(
          term_cfg.get("delay_min_lag", 0), f"{where}.delay_min_lag"
        ),
        delay_max_lag=_require_int(
          term_cfg.get("delay_max_lag", 0), f"{where}.delay_max_lag"
        ),
        delay_hold_prob=_require_float(
          term_cfg.get("delay_hold_prob", 0.0), f"{where}.delay_hold_prob"
        ),
        delay_group=term_cfg.get("delay_group"),
        params=dict(params),
      )
    )
    if func == "mjlab.envs.mdp.observations.builtin_sensor":
      sensors.append(
        SensorDeclaration(term=name, sensor_name=str(params["sensor_name"]))
      )

  schema = ObservationSchema(
    group="actor",
    terms=tuple(terms),
    total_dim=sum(term.width for term in terms),
    enable_corruption=_require_bool(
      actor_cfg.get("enable_corruption", False),
      f"{entry.id} env.observations.actor.enable_corruption",
    ),
  )
  if schema.total_dim != obs_dim:
    raise DistillationError(
      f"{entry.id} actor observation widths {schema.widths} sum to "
      f"{schema.total_dim}, but the ONNX 'obs' input has width {obs_dim}"
    )
  _require_onnx_observation_metadata(entry, schema, onnx)
  return schema, tuple(sensors)


def _require_onnx_observation_metadata(
  entry: TeacherEntry,
  schema: ObservationSchema,
  onnx: OnnxArtifact,
) -> None:
  """Check exported observation metadata against the saved observation terms."""
  names = _parse_csv(_require_metadata(entry, onnx, "observation_names"))
  if names != schema.names:
    raise DistillationError(
      f"{entry.id} ONNX observation_names {names} disagree with the saved actor "
      f"observation order {schema.names}"
    )
  scales = _parse_float_csv(_require_metadata(entry, onnx, "observation_terms_scale"))
  histories = _parse_float_csv(
    _require_metadata(entry, onnx, "observation_terms_history_length")
  )
  clips = _parse_clip_csv(_require_metadata(entry, onnx, "observation_terms_clip"))
  if not (len(scales) == len(histories) == len(clips) == len(schema.terms)):
    raise DistillationError(
      f"{entry.id} ONNX observation metadata has {len(scales)}/{len(histories)}/"
      f"{len(clips)} scale/history/clip entries for {len(schema.terms)} terms"
    )
  for term, scale, history, clip in zip(
    schema.terms, scales, histories, clips, strict=True
  ):
    declared_scale = 1.0 if term.scale is None else term.scale[0]
    if not math.isclose(scale, declared_scale, abs_tol=_ONNX_METADATA_TOLERANCE):
      raise DistillationError(
        f"{entry.id} observation {term.name!r} scale {declared_scale} disagrees with "
        f"the ONNX export {scale}"
      )
    if int(history) != term.history_length:
      raise DistillationError(
        f"{entry.id} observation {term.name!r} history {term.history_length} disagrees "
        f"with the ONNX export {history}"
      )
    declared_clip = (-math.inf, math.inf) if term.clip is None else term.clip
    if any(
      not math.isclose(a, b, rel_tol=0.0, abs_tol=_ONNX_METADATA_TOLERANCE)
      for a, b in zip(declared_clip, clip, strict=True)
    ):
      raise DistillationError(
        f"{entry.id} observation {term.name!r} clip {declared_clip} disagrees with "
        f"the ONNX export {clip}"
      )
  command_names = _parse_csv(_require_metadata(entry, onnx, "command_names"))
  for term in schema.terms:
    if term.func == "mjlab.envs.mdp.observations.generated_commands":
      command_name = str(term.params["command_name"])
      if command_name not in command_names:
        raise DistillationError(
          f"{entry.id} observation {term.name!r} reads command {command_name!r}, which "
          f"is not in the exported command_names {command_names}"
        )


def _resolve_actions(
  entry: TeacherEntry,
  env_config: Mapping[str, Any],
  onnx: OnnxArtifact,
  action_dim: int,
) -> ActionSpec:
  actions_cfg = _require_mapping(env_config.get("actions"), f"{entry.id} env.actions")
  if len(actions_cfg) != 1:
    raise UnsupportedTeacherError(
      f"{entry.id} declares action terms {sorted(actions_cfg)}; M1 supports a single "
      "joint-position action term"
    )
  term_name, raw_term = next(iter(actions_cfg.items()))
  if term_name != "joint_pos":
    raise UnsupportedTeacherError(
      f"{entry.id} action term is {term_name!r}; M1 supports 'joint_pos' only"
    )
  term_cfg = _require_mapping(raw_term, f"{entry.id} action term")
  if term_cfg.get("clip") is not None:
    raise UnsupportedTeacherError(
      f"{entry.id} action term sets clip {term_cfg['clip']!r}; M1 supports unclipped "
      "actions only"
    )
  if _require_bool(
    term_cfg.get("preserve_order", False), f"{entry.id} action preserve_order"
  ):
    raise UnsupportedTeacherError(
      f"{entry.id} action term preserves actuator order; M1 resolves the saved scale "
      "map onto the exported joint order"
    )

  joint_names = _parse_csv(_require_metadata(entry, onnx, "joint_names"))
  if len(joint_names) != action_dim or len(set(joint_names)) != action_dim:
    raise DistillationError(
      f"{entry.id} ONNX joint_names has {len(joint_names)} entries for an action "
      f"dimension of {action_dim}"
    )
  exported_scales = _parse_float_csv(_require_metadata(entry, onnx, "action_scale"))
  if len(exported_scales) != action_dim:
    raise DistillationError(
      f"{entry.id} ONNX action_scale has {len(exported_scales)} entries for an action "
      f"dimension of {action_dim}"
    )

  scale = term_cfg.get("scale", 1.0)
  if isinstance(scale, dict):
    try:
      indices, names, values = resolve_matching_names_values(
        dict(scale), list(joint_names)
      )
    except ValueError as exc:
      raise DistillationError(
        f"{entry.id} saved action scale map does not resolve onto the exported joint "
        f"order: {exc}"
      ) from exc
    if names != list(joint_names):
      raise DistillationError(
        f"{entry.id} saved action scale map does not cover the exported joint order"
      )
    resolved_scales = [0.0] * action_dim
    for index, value in zip(indices, values, strict=True):
      resolved_scales[index] = float(value)
  elif isinstance(scale, (int, float)) and not isinstance(scale, bool):
    resolved_scales = [float(scale)] * action_dim
  else:
    raise DistillationError(
      f"{entry.id} action scale {scale!r} is not a float or mapping"
    )

  for name, resolved, exported in zip(
    joint_names, resolved_scales, exported_scales, strict=True
  ):
    if not math.isclose(resolved, exported, abs_tol=_ONNX_METADATA_TOLERANCE):
      raise DistillationError(
        f"{entry.id} action scale for {name!r} is {resolved}, but the ONNX export has "
        f"{exported}; the saved action contract and the exported joint order disagree"
      )

  offset = term_cfg.get("offset", 0.0)
  if not isinstance(offset, (int, float)) or isinstance(offset, bool):
    raise UnsupportedTeacherError(
      f"{entry.id} action offset {offset!r} is not a scalar; M1 cannot resolve a "
      "per-actuator offset without the robot asset"
    )
  return ActionSpec(
    term=term_name,
    joint_names=joint_names,
    joint_scales=tuple(resolved_scales),
    offset=float(offset),
    uses_default_offset=_require_bool(
      term_cfg.get("use_default_offset", False), f"{entry.id} action use_default_offset"
    ),
    clip=None,
    dim=action_dim,
  )


def _resolve_reference(
  entry: TeacherEntry, env_config: Mapping[str, Any], joint_dim: int
) -> ReferenceSpec:
  with np.load(entry.motion, allow_pickle=False) as data:
    arrays = {name: data[name] for name in data.files}
  for name, array in arrays.items():
    if array.dtype == np.dtype("O"):
      raise DistillationError(f"{entry.id} motion array {name!r} has object dtype")
  for required in ("joint_pos", "joint_vel", "fps"):
    if required not in arrays:
      raise DistillationError(
        f"{entry.id} motion {entry.motion} has no {required!r} array "
        f"(found {sorted(arrays)})"
      )
  joint_pos = arrays["joint_pos"]
  joint_vel = arrays["joint_vel"]
  if joint_pos.shape != joint_vel.shape or joint_pos.ndim != 2:
    raise DistillationError(
      f"{entry.id} motion joint_pos/joint_vel must share a (frames, joints) shape, got "
      f"{joint_pos.shape} and {joint_vel.shape}"
    )
  frames, motion_joint_dim = (int(joint_pos.shape[0]), int(joint_pos.shape[1]))
  if motion_joint_dim != joint_dim:
    raise DistillationError(
      f"{entry.id} motion has {motion_joint_dim} joints but the actor has {joint_dim}"
    )
  fps_values = np.asarray(arrays["fps"], dtype=np.float64).reshape(-1)
  if (
    fps_values.size != 1
    or not math.isfinite(float(fps_values[0]))
    or fps_values[0] <= 0
  ):
    raise DistillationError(
      f"{entry.id} motion fps must be a single finite positive value, got {arrays['fps']!r}"
    )
  fps = float(fps_values[0])

  declared = _require_str(
    _saved_motion_cfg(entry, env_config).get("motion_file"),
    f"{entry.id} env.commands.motion.motion_file",
  )
  if Path(declared).name != entry.motion.name:
    raise DistillationError(
      f"{entry.id} saved env config references motion {Path(declared).name!r} but the "
      f"manifest selects {entry.motion.name!r}; the local override must match the "
      "saved reference"
    )
  return ReferenceSpec(
    motion_path=entry.motion,
    declared_motion_file=declared,
    frames=frames,
    fps=fps,
    joint_dim=motion_joint_dim,
    duration_s=frames / fps,
    array_shapes={
      name: tuple(int(d) for d in array.shape) for name, array in arrays.items()
    },
  )


def _require_onnx_interface(
  entry: TeacherEntry, onnx: OnnxArtifact, obs_dim: int, action_dim: int
) -> None:
  """Require the original tracking export interface: obs and time_step, batch 1."""
  expected = {"obs": (1, obs_dim), "time_step": (1, 1)}
  found = {info.name: info.shape for info in onnx.inputs}
  if found != expected:
    raise DistillationError(
      f"{entry.id} ONNX inputs are {found}; M1 expects exactly {expected}"
    )
  actions = onnx.output("actions")
  if actions is None or actions.shape != (1, action_dim):
    raise DistillationError(
      f"{entry.id} ONNX export needs an 'actions' output of shape (1, {action_dim}), "
      f"found {None if actions is None else actions.shape}"
    )


def _saved_motion_cfg(
  entry: TeacherEntry, env_config: Mapping[str, Any]
) -> Mapping[str, Any]:
  commands = _require_mapping(env_config.get("commands"), f"{entry.id} env.commands")
  return _require_mapping(commands.get("motion"), f"{entry.id} env.commands.motion")


def _require_reference_cadence(
  entry: TeacherEntry, reference: ReferenceSpec, control: ControlSpec
) -> None:
  if not math.isclose(
    control.control_hz, reference.fps, rel_tol=_REFERENCE_FPS_RTOL, abs_tol=0.0
  ):
    raise DistillationError(
      f"{entry.id} control rate is {control.control_hz} Hz but the reference is "
      f"{reference.fps} FPS; the tracker advances one reference frame per control step, "
      "so the cadences must agree"
    )


def load_onnx_artifact(path: Path, where: str) -> OnnxArtifact:
  """Read an original teacher ONNX export without running it."""
  onnx = require_onnx()
  model = onnx.load(str(path))
  graph = model.graph
  metadata = {prop.key: prop.value for prop in model.metadata_props}

  def tensor_infos(values: Any, what: str) -> tuple[TensorInfo, ...]:
    infos: list[TensorInfo] = []
    for value in values:
      shape: list[int] = []
      for dim in value.type.tensor_type.shape.dim:
        size = int(dim.dim_value)
        if size <= 0:
          raise DistillationError(
            f"{where} has a dynamic {what} shape for {value.name!r}; static shapes are "
            "required for parity checks"
          )
        shape.append(size)
      infos.append(TensorInfo(name=value.name, shape=tuple(shape)))
    return tuple(infos)

  initializers = {
    init.name: onnx.numpy_helper.to_array(init) for init in graph.initializer
  }
  constants: dict[str, np.ndarray] = {}
  for node in graph.node:
    if node.op_type != "Constant":
      continue
    for attribute in node.attribute:
      if attribute.name == "value":
        constants[node.output[0]] = onnx.numpy_helper.to_array(attribute.t)

  return OnnxArtifact(
    path=path,
    metadata=metadata,
    inputs=tensor_infos(graph.input, "input"),
    outputs=tensor_infos(graph.output, "output"),
    initializers=initializers,
    constants=constants,
  )


def _require_same_contract(first: ResolvedTeacher, other: ResolvedTeacher) -> None:
  """Reject a teacher that is incompatible with the cohort's first teacher."""
  checks: list[tuple[str, Any, Any]] = [
    ("actor.class_name", first.actor.class_name, other.actor.class_name),
    ("actor.hidden_dims", first.actor.hidden_dims, other.actor.hidden_dims),
    ("actor.activation", first.actor.activation, other.actor.activation),
    (
      "actor.obs_normalization",
      first.actor.obs_normalization,
      other.actor.obs_normalization,
    ),
    ("actor.obs_groups", first.actor.obs_groups, other.actor.obs_groups),
    ("actor.obs_dim", first.actor.obs_dim, other.actor.obs_dim),
    ("actor.action_dim", first.actor.action_dim, other.actor.action_dim),
    (
      "actor.distribution_class_name",
      first.actor.distribution_class_name,
      other.actor.distribution_class_name,
    ),
    (
      "actor.distribution_cfg",
      first.actor.distribution_cfg,
      other.actor.distribution_cfg,
    ),
    ("observation.names", first.observations.names, other.observations.names),
    ("observation.widths", first.observations.widths, other.observations.widths),
    (
      "observation.enable_corruption",
      first.observations.enable_corruption,
      other.observations.enable_corruption,
    ),
    ("action.term", first.actions.term, other.actions.term),
    ("action.joint_names", first.actions.joint_names, other.actions.joint_names),
    ("action.joint_scales", first.actions.joint_scales, other.actions.joint_scales),
    ("action.offset", first.actions.offset, other.actions.offset),
    (
      "action.uses_default_offset",
      first.actions.uses_default_offset,
      other.actions.uses_default_offset,
    ),
    ("control.sim_timestep", first.control.sim_timestep, other.control.sim_timestep),
    ("control.decimation", first.control.decimation, other.control.decimation),
    ("reference.fps", first.reference.fps, other.reference.fps),
    ("reference.joint_dim", first.reference.joint_dim, other.reference.joint_dim),
    ("anchor_body_name", first.anchor_body_name, other.anchor_body_name),
    ("body_names", first.body_names, other.body_names),
    ("lookahead_s", first.lookahead_s, other.lookahead_s),
    ("sensors", first.sensors, other.sensors),
  ]
  for name, left, right in checks:
    if left != right:
      raise DistillationError(
        f"Teachers {first.id!r} and {other.id!r} disagree on {name}: {left!r} != {right!r}"
      )


def _require_equal_saved_configs(teachers: tuple[ResolvedTeacher, ...]) -> None:
  """Require identical saved configurations apart from the reference motion file."""
  first = teachers[0]
  for teacher in teachers[1:]:
    left = _without_excluded_env_fields(first.env_config)
    right = _without_excluded_env_fields(teacher.env_config)
    difference = _first_difference(left, right)
    if difference is not None:
      raise DistillationError(
        f"Teachers {first.id!r} and {teacher.id!r} have different saved environment "
        f"configurations apart from {_EXCLUDED_ENV_FIELDS}: {difference}"
      )
    for section in ("obs_groups", "actor"):
      if first.agent_config.get(section) != teacher.agent_config.get(section):
        raise DistillationError(
          f"Teachers {first.id!r} and {teacher.id!r} disagree on agent config "
          f"{section!r}; other runner/optimizer fields are recorded as provenance only"
        )


def _without_excluded_env_fields(env_config: Mapping[str, Any]) -> dict[str, Any]:
  pruned = copy.deepcopy(dict(env_config))
  pruned["commands"]["motion"].pop("motion_file", None)
  return pruned


def _first_difference(left: Any, right: Any, path: str = "") -> str | None:
  if isinstance(left, Mapping) and isinstance(right, Mapping):
    keys = sorted(set(left) | set(right), key=str)
    for key in keys:
      child = f"{path}.{key}" if path else str(key)
      if key not in left or key not in right:
        return f"{child} is present in only one configuration"
      difference = _first_difference(left[key], right[key], child)
      if difference is not None:
        return difference
    return None
  if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
    if len(left) != len(right):
      return f"{path} has lengths {len(left)} and {len(right)}"
    for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
      difference = _first_difference(left_item, right_item, f"{path}[{index}]")
      if difference is not None:
        return difference
    return None
  if left != right:
    return f"{path} is {left!r} vs {right!r}"
  return None


def _resolve_scale(value: Any, where: str) -> tuple[float, ...] | None:
  if value is None:
    return None
  if isinstance(value, (int, float)) and not isinstance(value, bool):
    return (float(value),)
  if isinstance(value, (list, tuple)):
    scale = tuple(_require_float(item, f"{where}.scale") for item in value)
    if len(scale) != 1:
      raise UnsupportedTeacherError(
        f"{where}.scale has {len(scale)} entries; M1 compares a single exported "
        "per-term scale"
      )
    return scale
  raise DistillationError(f"{where}.scale {value!r} is not numeric")


def _resolve_clip(value: Any, where: str) -> tuple[float, float] | None:
  if value is None:
    return None
  if isinstance(value, (list, tuple)) and len(value) == 2:
    return (
      _require_float(value[0], f"{where}.clip"),
      _require_float(value[1], f"{where}.clip"),
    )
  raise DistillationError(f"{where}.clip {value!r} is not a (min, max) pair")


def _resolve_noise(value: Any, where: str) -> NoiseSpec | None:
  if value is None:
    return None
  cfg = _require_mapping(value, f"{where}.noise")
  operation = _require_str(cfg.get("operation"), f"{where}.noise.operation")
  # Runtime cache state is not part of the saved configuration contract.
  params = {
    key: item for key, item in cfg.items() if key not in ("operation", "_tensor_cache")
  }
  return NoiseSpec(operation=operation, params=params)


def _require_metadata(entry: TeacherEntry, onnx: OnnxArtifact, key: str) -> str:
  value = onnx.metadata.get(key)
  if value is None:
    raise DistillationError(f"{entry.id} ONNX export has no {key!r} metadata")
  return value


def _parse_csv(value: str) -> tuple[str, ...]:
  return tuple(part for part in value.split(","))


def _parse_float_csv(value: str) -> tuple[float, ...]:
  try:
    return tuple(float(part) for part in value.split(","))
  except ValueError as exc:
    raise DistillationError(
      f"ONNX metadata {value!r} is not a comma-separated float list"
    ) from exc


def _parse_clip_csv(value: str) -> tuple[tuple[float, float], ...]:
  clips: list[tuple[float, float]] = []
  for part in value.split(","):
    bounds = part.split(";")
    if len(bounds) != 2:
      raise DistillationError(f"ONNX metadata clip {part!r} is not a 'min;max' pair")
    try:
      clips.append((float(bounds[0]), float(bounds[1])))
    except ValueError as exc:
      raise DistillationError(
        f"ONNX metadata clip {part!r} is not a numeric 'min;max' pair"
      ) from exc
  return tuple(clips)


def _reject_unknown_keys(
  mapping: Mapping[str, Any], allowed: tuple[str, ...], where: str
) -> None:
  unknown = sorted(set(mapping) - set(allowed))
  if unknown:
    raise DistillationError(f"{where} has unsupported keys {unknown}")


def _resolve_path(value: Any, repo_root: Path, where: str) -> Path:
  text = _require_str(value, where).strip()
  if not text:
    raise DistillationError(f"{where} must not be empty")
  path = Path(text)
  return path if path.is_absolute() else (repo_root / path)


def _require_mapping(value: Any, where: str) -> dict[str, Any]:
  """Return a mapping loaded from YAML; saved artifacts always decode to dicts."""
  if not isinstance(value, dict):
    raise DistillationError(f"{where} must be a mapping, got {type(value).__name__}")
  return value


def _require_str(value: Any, where: str) -> str:
  if not isinstance(value, str):
    raise DistillationError(f"{where} must be a string, got {type(value).__name__}")
  if not value.strip():
    raise DistillationError(f"{where} must be a non-empty string")
  return value


def _require_float(value: Any, where: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise DistillationError(f"{where} must be a number, got {value!r}")
  return float(value)


def _require_int(value: Any, where: str) -> int:
  if isinstance(value, bool) or not isinstance(value, int):
    raise DistillationError(f"{where} must be an integer, got {value!r}")
  return int(value)


def _require_bool(value: Any, where: str) -> bool:
  if not isinstance(value, bool):
    raise DistillationError(f"{where} must be a boolean, got {value!r}")
  return value


def _require_positive_int_sequence(value: Any, where: str) -> tuple[int, ...]:
  if not isinstance(value, (list, tuple)) or not value:
    raise DistillationError(f"{where} must be a non-empty sequence of integers")
  dims = tuple(_require_int(item, where) for item in value)
  if any(dim <= 0 for dim in dims):
    raise DistillationError(f"{where} must contain positive integers, got {dims}")
  return dims
