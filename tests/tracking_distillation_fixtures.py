"""Generated teacher artifacts for distillation tests.

The builders here write a miniature but structurally faithful cohort: real
``params/env.yaml`` and ``params/agent.yaml`` dumps with python tags, a current
RSL-RL actor checkpoint, a motion NPZ, and an ONNX export that embeds the
reference tensors and export metadata exactly like the tracking runner does.
No private binary artifact is needed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict
from torch import nn

DEFAULT_TEACHER_IDS = ("tiny_000", "tiny_001")
DEFAULT_JOINT_NAMES = ("left_hip_joint", "right_hip_joint", "waist_joint")
DEFAULT_SCALE_MAP = {".*_hip_joint": 0.3, "waist_joint": 0.5}
DEFAULT_HIDDEN_DIMS = (8, 4)
DEFAULT_FRAMES = 6
DEFAULT_FPS = 50.0
DEFAULT_TERMS = (
  "command",
  "motion_lookahead",
  "motion_anchor_ori_b",
  "base_ang_vel",
  "joint_pos",
  "joint_vel",
  "actions",
)

_JOINT_MULTIPLE_TERMS = {
  "command": 2,
  "motion_lookahead": 2,
  "joint_pos": 1,
  "joint_vel": 1,
  "actions": 1,
}
_ABSOLUTE_TERMS = {"motion_anchor_ori_b": 6, "base_ang_vel": 3}

_TERM_BLOCKS = {
  "command": """      command:
        func: &id001 !!python/name:mjlab.envs.mdp.observations.generated_commands ''
        params:
          command_name: motion
        noise: null
        clip: null
        scale: null
        delay_min_lag: 0
        delay_max_lag: 0
        delay_per_env: true
        delay_hold_prob: 0.0
        delay_update_period: 0
        delay_per_env_phase: true
        delay_group: null
        history_length: 0
        flatten_history_dim: true
""",
  "motion_lookahead": """      motion_lookahead:
        func: !!python/name:mjlab.tasks.tracking.mdp.observations.motion_lookahead ''
        params:
          command_name: motion
        noise: null
        clip: null
        scale: null
        delay_min_lag: 0
        delay_max_lag: 0
        delay_per_env: true
        delay_hold_prob: 0.0
        delay_update_period: 0
        delay_per_env_phase: true
        delay_group: null
        history_length: 0
        flatten_history_dim: true
""",
  "motion_anchor_ori_b": """      motion_anchor_ori_b:
        func: !!python/name:mjlab.tasks.tracking.mdp.observations.motion_anchor_ori_b ''
        params:
          command_name: motion
        noise:
          operation: add
          _tensor_cache: {{}}
          n_min: -0.05
          n_max: 0.05
        clip: null
        scale: null
        delay_min_lag: 0
        delay_max_lag: 0
        delay_per_env: true
        delay_hold_prob: 0.0
        delay_update_period: 0
        delay_per_env_phase: true
        delay_group: null
        history_length: 0
        flatten_history_dim: true
""",
  "base_ang_vel": """      base_ang_vel:
        func: !!python/name:mjlab.envs.mdp.observations.builtin_sensor ''
        params:
          sensor_name: robot/imu_ang_vel
        noise:
          operation: add
          _tensor_cache: {{}}
          n_min: -0.2
          n_max: 0.2
        clip: null
        scale: null
        delay_min_lag: 0
        delay_max_lag: 1
        delay_per_env: true
        delay_hold_prob: 0.9
        delay_update_period: 1
        delay_per_env_phase: true
        delay_group: null
        history_length: 0
        flatten_history_dim: true
""",
  "joint_pos": """      joint_pos:
        func: !!python/name:mjlab.envs.mdp.observations.joint_pos_rel ''
        params:
          biased: true
        noise:
          operation: add
          _tensor_cache: {{}}
          n_min: {joint_position_noise_min}
          n_max: 0.01
        clip: null
        scale: null
        delay_min_lag: 0
        delay_max_lag: 1
        delay_per_env: true
        delay_hold_prob: 0.9
        delay_update_period: 1
        delay_per_env_phase: true
        delay_group: encoder_packet
        history_length: 0
        flatten_history_dim: true
""",
  "joint_vel": """      joint_vel:
        func: !!python/name:mjlab.envs.mdp.observations.joint_vel_rel ''
        params: {{}}
        noise:
          operation: add
          _tensor_cache: {{}}
          n_min: -0.5
          n_max: 0.5
        clip: null
        scale: null
        delay_min_lag: 0
        delay_max_lag: 1
        delay_per_env: true
        delay_hold_prob: 0.9
        delay_update_period: 1
        delay_per_env_phase: true
        delay_group: encoder_packet
        history_length: 0
        flatten_history_dim: true
""",
  "actions": """      actions:
        func: &id006 !!python/name:mjlab.envs.mdp.observations.last_action ''
        params: {{}}
        noise: null
        clip: null
        scale: null
        delay_min_lag: 0
        delay_max_lag: 0
        delay_per_env: true
        delay_hold_prob: 0.0
        delay_update_period: 0
        delay_per_env_phase: true
        delay_group: null
        history_length: 0
        flatten_history_dim: true
""",
}


@dataclass(frozen=True)
class TinyTeacher:
  """Paths and declared identities of one generated teacher."""

  id: str
  run_dir: Path
  checkpoint: Path
  env_config: Path
  agent_config: Path
  onnx: Path
  motion: Path
  declared_motion_file: str


@dataclass(frozen=True)
class TinyCohort:
  """A generated cohort plus the manifest that points at it."""

  root: Path
  manifest: Path
  teachers: tuple[TinyTeacher, ...]
  obs_dim: int
  action_dim: int
  joint_names: tuple[str, ...]


def term_width(term: str, joint_dim: int, lookahead_s: float = 0.0) -> int:
  """Declared width of a generated observation term."""
  if term == "motion_lookahead":
    return 2 * joint_dim if lookahead_s > 0.0 else 0
  if term in _ABSOLUTE_TERMS:
    return _ABSOLUTE_TERMS[term]
  return _JOINT_MULTIPLE_TERMS[term] * joint_dim


def observation_dim(
  terms: Sequence[str], joint_dim: int, lookahead_s: float = 0.0
) -> int:
  """Total width of a generated actor observation schema."""
  return sum(term_width(term, joint_dim, lookahead_s) for term in terms)


def make_actor(
  obs_dim: int,
  action_dim: int,
  *,
  hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
  obs_normalization: bool = True,
  seed: int = 0,
) -> MLPModel:
  """Create an MLPModel with deterministic weights and normalizer statistics."""
  torch.manual_seed(seed)
  obs = TensorDict({"actor": torch.zeros(1, obs_dim)})
  model = MLPModel(
    obs=obs,
    obs_groups={"actor": ["actor"]},
    obs_set="actor",
    output_dim=action_dim,
    hidden_dims=list(hidden_dims),
    activation="elu",
    obs_normalization=obs_normalization,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )
  if obs_normalization:
    # Learn deterministic, non-degenerate statistics the same way training does,
    # so the checkpoint carries _std == sqrt(_var) like a real actor.
    generator = torch.Generator().manual_seed(seed + 1)
    samples = torch.randn(512, obs_dim, generator=generator) * 0.6 + 0.25
    model.train()
    model.update_normalization(TensorDict({"actor": samples}))
  model.eval()
  return model


def write_checkpoint(path: Path, model: nn.Module, *, iteration: int = 5) -> None:
  """Write a current-format RSL-RL checkpoint containing only the actor."""
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(
    {
      "actor_state_dict": model.state_dict(),
      "critic_state_dict": {},
      "optimizer_state_dict": {},
      "iter": iteration,
      "infos": {"env_state": {"common_step_counter": 0}},
    },
    path,
  )


def resolved_action_scale(
  joint_names: Sequence[str], scale_map: Mapping[str, float] | None = None
) -> list[float]:
  """Resolve a saved action scale map onto a joint order, as the exporter does."""
  from mjlab.utils.lab_api.string import resolve_matching_names_values

  if scale_map is None:
    scale_map = DEFAULT_SCALE_MAP
  indices, _, values = resolve_matching_names_values(dict(scale_map), list(joint_names))
  scales = [0.0] * len(joint_names)
  for index, value in zip(indices, values, strict=True):
    scales[index] = float(value)
  return scales


def write_motion(
  path: Path,
  *,
  frames: int = DEFAULT_FRAMES,
  joint_dim: int = len(DEFAULT_JOINT_NAMES),
  fps: float = DEFAULT_FPS,
  seed: int = 0,
) -> None:
  """Write a motion NPZ with the arrays the tracking command consumes."""
  path.parent.mkdir(parents=True, exist_ok=True)
  generator = np.random.default_rng(seed)
  np.savez(
    path,
    joint_pos=generator.normal(size=(frames, joint_dim)).astype(np.float32),
    joint_vel=generator.normal(size=(frames, joint_dim)).astype(np.float32),
    body_pos_w=generator.normal(size=(frames, 2, 3)).astype(np.float32),
    body_quat_w=generator.normal(size=(frames, 2, 4)).astype(np.float32),
    body_lin_vel_w=generator.normal(size=(frames, 2, 3)).astype(np.float32),
    body_ang_vel_w=generator.normal(size=(frames, 2, 3)).astype(np.float32),
    fps=np.array([fps], dtype=np.float32),
  )


def write_saved_configs(
  env_config: Path,
  agent_config: Path,
  *,
  motion_file: str,
  hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
  terms: Sequence[str] = DEFAULT_TERMS,
  lookahead_s: float = 0.0,
  timestep: float = 0.005,
  decimation: int = 4,
  rnn_type: str | None = None,
  scale_map: Mapping[str, float] | None = None,
  joint_position_noise_min: float = -0.01,
) -> None:
  """Write ``params/env.yaml`` and ``params/agent.yaml`` like a training run does."""
  if scale_map is None:
    scale_map = DEFAULT_SCALE_MAP
  scale_lines = "\n".join(f"      {key}: {value}" for key, value in scale_map.items())
  term_blocks = "".join(
    _TERM_BLOCKS[name].format(joint_position_noise_min=joint_position_noise_min)
    for name in terms
  )
  env_config.parent.mkdir(parents=True, exist_ok=True)
  env_config.write_text(
    f"""decimation: {decimation}
observations:
  actor:
    terms:
{term_blocks}    concatenate_terms: true
    concatenate_dim: -1
    enable_corruption: true
    history_length: null
    nan_policy: disabled
actions:
  joint_pos:
    entity_name: robot
    clip: null
    transmission_type: !!python/object/apply:mjlab.actuator.actuator.TransmissionType
    - joint
    actuator_names: !!python/tuple
    - .*
    scale:
{scale_lines}
    offset: 0.0
    preserve_order: false
    use_default_offset: true
sim:
  mujoco:
    timestep: {timestep}
    integrator: implicitfast
episode_length_s: 10.0
commands:
  motion:
    resampling_time_range: !!python/tuple
    - 1000000000.0
    - 1000000000.0
    motion_file: {motion_file}
    anchor_body_name: torso_link
    body_names: !!python/tuple
    - pelvis
    - torso_link
    lookahead_s: {lookahead_s}
    sampling_mode: adaptive
"""
  )

  hidden_lines = "\n".join(f"  - {dim}" for dim in hidden_dims)
  rnn_type_yaml = "null" if rnn_type is None else rnn_type
  agent_config.parent.mkdir(parents=True, exist_ok=True)
  agent_config.write_text(
    f"""seed: 42
obs_groups:
  actor: !!python/tuple
  - actor
  critic: !!python/tuple
  - critic
clip_actions: null
class_name: OnPolicyRunner
actor:
  hidden_dims: !!python/tuple
{hidden_lines}
  activation: elu
  obs_normalization: true
  cnn_cfg: null
  distribution_cfg:
    class_name: GaussianDistribution
    init_std: 1.0
    std_type: scalar
  rnn_type: {rnn_type_yaml}
  class_name: MLPModel
critic:
  hidden_dims: !!python/tuple
  - 8
  - 4
  activation: elu
  obs_normalization: true
  cnn_cfg: null
  distribution_cfg: null
  rnn_type: null
  class_name: MLPModel
algorithm:
  learning_rate: 0.001
  num_learning_epochs: 5
"""
  )


class _OnnxExport(nn.Module):
  """Export wrapper bundling the policy with its reference tensors."""

  joint_pos: torch.Tensor
  joint_vel: torch.Tensor

  def __init__(
    self, model: nn.Module, joint_pos: torch.Tensor, joint_vel: torch.Tensor
  ) -> None:
    super().__init__()
    self.policy = model.as_onnx(verbose=False)  # type: ignore[attr-defined]
    self.register_buffer("joint_pos", joint_pos.clone())
    self.register_buffer("joint_vel", joint_vel.clone())

  def forward(
    self, obs: torch.Tensor, time_step: torch.Tensor
  ) -> tuple[torch.Tensor, ...]:
    steps = torch.clamp(time_step.long().squeeze(-1), max=self.joint_pos.shape[0] - 1)
    return (self.policy(obs), self.joint_pos[steps], self.joint_vel[steps])


def write_onnx_export(
  path: Path,
  model: nn.Module,
  motion: Path,
  *,
  obs_dim: int,
  joint_names: Sequence[str] = DEFAULT_JOINT_NAMES,
  terms: Sequence[str] = DEFAULT_TERMS,
  observation_names: Sequence[str] | None = None,
  action_scale: Sequence[float] | None = None,
) -> None:
  """Export the policy and attach the tracking runner's ONNX metadata."""
  import onnx

  with np.load(motion, allow_pickle=False) as data:
    joint_pos = torch.from_numpy(data["joint_pos"])
    joint_vel = torch.from_numpy(data["joint_vel"])
  if action_scale is None:
    action_scale = resolved_action_scale(joint_names)
  if observation_names is None:
    observation_names = terms

  wrapper = _OnnxExport(model, joint_pos, joint_vel).eval()
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.onnx.export(
    wrapper,
    (torch.zeros(1, obs_dim), torch.zeros(1, 1)),
    str(path),
    export_params=True,
    opset_version=18,
    input_names=["obs", "time_step"],
    output_names=["actions", "joint_pos", "joint_vel"],
    dynamic_axes={},
    dynamo=False,
  )
  exported = onnx.load(str(path))
  onnx.helper.set_model_props(
    exported,
    {
      "run_path": "local",
      "joint_names": ",".join(joint_names),
      "joint_stiffness": ",".join(f"{1.0:.3f}" for _ in joint_names),
      "joint_damping": ",".join(f"{1.0:.3f}" for _ in joint_names),
      "default_joint_pos": ",".join(f"{0.0:.3f}" for _ in joint_names),
      "command_names": "motion",
      "observation_names": ",".join(observation_names),
      "observation_terms_scale": ",".join(f"{1.0:.3f}" for _ in observation_names),
      "observation_terms_flatten_history_dim": ",".join(
        f"{1.0:.3f}" for _ in observation_names
      ),
      "observation_terms_history_length": ",".join(
        f"{0.0:.3f}" for _ in observation_names
      ),
      "observation_terms_clip": ",".join("-inf;inf" for _ in observation_names),
      "action_scale": ",".join(f"{value:.3f}" for value in action_scale),
      "anchor_body_name": "torso_link",
      "body_names": "pelvis,torso_link",
    },
  )
  onnx.save(exported, str(path))


def write_teacher(
  root: Path,
  teacher_id: str,
  *,
  seed: int = 0,
  terms: Sequence[str] = DEFAULT_TERMS,
  joint_names: Sequence[str] = DEFAULT_JOINT_NAMES,
  hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
  frames: int = DEFAULT_FRAMES,
  fps: float = DEFAULT_FPS,
  timestep: float = 0.005,
  decimation: int = 4,
  lookahead_s: float = 0.0,
  rnn_type: str | None = None,
  motion_name: str | None = None,
  declared_motion_file: str | None = None,
  observation_names: Sequence[str] | None = None,
  action_scale: Sequence[float] | None = None,
  joint_position_noise_min: float = -0.01,
  write_onnx: bool = True,
) -> TinyTeacher:
  """Write one complete generated teacher under ``root``."""
  joint_dim = len(joint_names)
  obs_dim = observation_dim(terms, joint_dim, lookahead_s)
  run_dir = root / "runs" / teacher_id
  name = motion_name if motion_name is not None else f"{teacher_id}_tracking.npz"
  motion = root / "data" / "tennis" / name
  declared = (
    declared_motion_file
    if declared_motion_file is not None
    else f"/home/fushan/mjlab/data/tennis/{name}"
  )

  write_motion(motion, frames=frames, joint_dim=joint_dim, fps=fps, seed=seed)
  env_config = run_dir / "params" / "env.yaml"
  agent_config = run_dir / "params" / "agent.yaml"
  write_saved_configs(
    env_config,
    agent_config,
    motion_file=declared,
    hidden_dims=hidden_dims,
    terms=terms,
    lookahead_s=lookahead_s,
    timestep=timestep,
    decimation=decimation,
    rnn_type=rnn_type,
    joint_position_noise_min=joint_position_noise_min,
  )
  model = make_actor(obs_dim, joint_dim, hidden_dims=hidden_dims, seed=seed)
  checkpoint = run_dir / "model_5.pt"
  write_checkpoint(checkpoint, model)
  onnx_path = run_dir / "export.onnx"
  if write_onnx:
    write_onnx_export(
      onnx_path,
      model,
      motion,
      obs_dim=obs_dim,
      joint_names=joint_names,
      terms=terms,
      observation_names=observation_names,
      action_scale=action_scale,
    )
  return TinyTeacher(
    id=teacher_id,
    run_dir=run_dir,
    checkpoint=checkpoint,
    env_config=env_config,
    agent_config=agent_config,
    onnx=onnx_path,
    motion=motion,
    declared_motion_file=declared,
  )


def write_manifest(
  path: Path,
  teachers: Sequence[TinyTeacher],
  *,
  root: Path,
  name: str = "tiny-cohort",
  version: int = 1,
  body: str | None = None,
) -> Path:
  """Write a manifest that references generated teachers with root-relative paths."""
  path.parent.mkdir(parents=True, exist_ok=True)
  if body is not None:
    path.write_text(body)
    return path
  lines = [
    f"version: {version}",
    f"name: {name}",
    "robot: agibot_x2",
    "base_task: Mjlab-Tracking-Flat-AgiBot-X2",
    "teachers:",
  ]
  for teacher in teachers:
    lines.append(f"  - id: {teacher.id}")
    for field, value in (
      ("checkpoint", teacher.checkpoint),
      ("motion", teacher.motion),
      ("env_config", teacher.env_config),
      ("agent_config", teacher.agent_config),
      ("onnx", teacher.onnx),
    ):
      lines.append(f"    {field}: {Path(value).relative_to(root)}")
    lines.append("    sampling_weight: 1.0")
  path.write_text("\n".join(lines) + "\n")
  return path


def build_tiny_cohort(
  root: Path,
  *,
  teacher_ids: Sequence[str] = DEFAULT_TEACHER_IDS,
  seeds: Sequence[int] | None = None,
  terms: Sequence[str] = DEFAULT_TERMS,
  joint_names: Sequence[str] = DEFAULT_JOINT_NAMES,
  frames: int = DEFAULT_FRAMES,
  lookahead_s: float = 0.0,
) -> TinyCohort:
  """Write a manifest plus one generated teacher per id."""
  if seeds is None:
    seeds = list(range(len(teacher_ids)))
  teachers = [
    write_teacher(
      root,
      teacher_id,
      seed=seed,
      terms=terms,
      joint_names=joint_names,
      frames=frames,
      lookahead_s=lookahead_s,
    )
    for teacher_id, seed in zip(teacher_ids, seeds, strict=True)
  ]
  return TinyCohort(
    root=root,
    manifest=write_manifest(
      root / "configs" / "tiny_teachers.yaml", teachers, root=root
    ),
    teachers=tuple(teachers),
    obs_dim=observation_dim(terms, len(joint_names), lookahead_s),
    action_dim=len(joint_names),
    joint_names=tuple(joint_names),
  )


def sha256_file(path: Path) -> str:
  """SHA-256 digest, for manifest hash assertions."""
  return hashlib.sha256(path.read_bytes()).hexdigest()
