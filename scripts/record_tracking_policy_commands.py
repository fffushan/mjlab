"""Record the motor commands a tracking checkpoint issues in the training simulator.

The motivation is a mismatch observed on hardware: the deployment controller
commands a knee position target (~47 deg) that is far below the reference
trajectory (~65-72 deg), without any observed torque saturation. Learned PD
targets need not equal reference angles, so the open question is whether the
mismatch is learned behavior or an observation/export/environment difference.
This script records, in the native mjlab simulator, the quantities needed to
compare a checkpoint's commands against the deployment bridge:

  - the reference joint positions/velocities of the motion frame being tracked
  - the actor observation that produced each action (the exact policy input)
  - the raw network action and the (possibly wrapper-clipped) action the
    environment actually consumed
  - the scaled/offset processed target and the encoder-bias-adjusted target
    written to the simulator
  - pre/post joint positions/velocities, joint-space actuator torque,
    actuator-space force, root pose/velocity, and pre/post simulation times
  - configured/effective gains, effort limits, and command-delay settings

Everything is recorded for all actuated joints (29 for the G1 29DOF mode-15
tracking task), not just the knees.

Units and stages (native G1 configuration)
------------------------------------------
The mode-15 G1 tracks joint positions with MuJoCo ``<position>`` actuators
(``BuiltinPositionActuator``). ``ctrl`` is therefore a *position target in
radians*, not a torque; the deployment bridge instead computes an external PD
torque, so ``ctrl`` is not comparable across the two runtimes. The comparable
quantities are the processed position target, the simulator position target and
the joint-space actuator torque:

  raw_action        policy output (unitless), shape (num_joints,)
  clipped_action    raw_action after the wrapper's clip_actions
  processed_target  clipped_action * action_scale + action_offset   [rad]
  sim_target        processed_target - encoder_bias                 [rad]
  ctrl              sim_target, the value written to ``data.ctrl``  [rad]
  qfrc_actuator     actuator contribution mapped into joint space   [N*m]
  actuator_force    scalar actuator output in actuation space       [N*m]

``pd_torque_requested_pre`` is a reconstruction, ``kp * (sim_target - q_pre) -
kd * dq_pre`` evaluated at the pre-step state, and is labeled as such.
``pd_torque_applied_pre`` additionally clamps to the configured effort limit.
The torque logged from the simulator (``qfrc_actuator_post``) is read after the
environment step, and the environment calls ``sim.forward()`` after the
decimation loop, so it is the actuator force *recomputed at the post-step
state*, not provably the force integrated during the last physics substep. A
substep-level trace would require instrumenting the environment step and is
deliberately out of scope here; see ``docs/tracking_policy_command_recording.md``.

Nominal diagnostic, not a training-distribution sample
------------------------------------------------------
The recorder is deterministic and noise-free by construction, and every
deviation from the saved training configuration is recorded in
``metadata.json``: observation corruption is disabled, all startup/interval
domain randomization is removed, the reference state is written exactly at the
start frame, the episode horizon is lengthened and auto-reset is off so the run
stops (and says why) at the first termination instead of silently stitching
resets. Physical termination terms stay active, and the saved dynamics,
timestep, decimation, gains, action scale/offset and command-delay settings are
preserved. Results must be labeled as a nominal evaluation; they are not a
sample from the training distribution.

The diagnostic requires a GPU for the native simulator. A bounded smoke run is
possible with ``--max-steps``.

Example:

  uv run --no-sync python scripts/record_tracking_policy_commands.py \\
    --checkpoint-file logs/rsl_rl/g1_29dof_mode_15_tracking/2026-09-11_20-36-01/model_15998.pt \\
    --output-dir /tmp/mjlab_policy_commands \\
    --onnx-file /home/agiuser/projects/deploy/robots/g1_29dof/config/policy/mimic/qianghuo_mode_15/exported/policy.onnx
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np
import torch
import tyro

import mjlab
from mjlab.actuator.builtin_actuator import BuiltinPositionActuator
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg

# ``_load_saved_env_yaml`` is private, but it is the repo's tag-safe loader for
# the saved ``params/env.yaml`` that training writes next to a checkpoint.
from mjlab.tasks.tracking.mdp.commands import (
  MotionCommand,
  _load_saved_env_yaml,
  load_saved_lookahead_s,
)
from mjlab.utils.lab_api.string import resolve_matching_names
from mjlab.utils.torch import configure_torch_backends

DEFAULT_TASK = "Mjlab-Tracking-Flat-Unitree-G1-29DOF-Mode-15-No-State-Estimation"
DEFAULT_MOTION_FILE = "data/qianghuo_smplx_unitree_g1_29dof_mode_15_tracking.npz"

NPZ_NAME = "policy_commands.npz"
CSV_NAME = "policy_commands.csv"
OBS_CSV_NAME = "actor_observations.csv"
METADATA_NAME = "metadata.json"
ONNX_PARITY_NAME = "onnx_parity.json"

# Extra episode time beyond the requested recording duration. The margin keeps
# the ``time_out`` term (a physical episode-limit termination) from firing
# before the bounded recording ends, so the run stops on the step bound or on a
# genuine failure termination.
HORIZON_MARGIN_S = 1.0

# Raw ONNX output should match the PyTorch actor exactly; this tolerance absorbs
# float32 export rounding only.
ONNX_TOLERANCE = 2e-4


##
# Joint/actuator control layout.
##


@dataclass(frozen=True)
class PositionActuatorRecord:
  """One ``BuiltinPositionActuator`` in entity-local joint/ctrl index space."""

  target_ids: tuple[int, ...]
  """Entity-local joint indices controlled by the actuator."""
  ctrl_ids: tuple[int, ...]
  """Entity-local control indices written by the actuator."""
  kp: float
  kd: float
  effort_limit: float | None
  delay_max_lag: int


@dataclass
class JointControl:
  """Per-joint gains, effort limits, delay and ctrl indices.

  All tensors are indexed by entity-local joint index, the same index space as
  ``robot.data.joint_pos`` and ``action_term.target_ids``.
  """

  joint_names: tuple[str, ...]
  kp: torch.Tensor
  kd: torch.Tensor
  effort_limit: torch.Tensor
  delay_max_lag: torch.Tensor
  ctrl_index: torch.Tensor

  @property
  def num_joints(self) -> int:
    return len(self.joint_names)

  @property
  def is_complete(self) -> bool:
    """True when every joint has a position actuator with a ctrl index."""
    return bool((self.ctrl_index >= 0).all().item())

  @property
  def uncovered_joints(self) -> list[str]:
    missing = (self.ctrl_index < 0).nonzero(as_tuple=False).flatten().tolist()
    return [self.joint_names[i] for i in missing]


def build_joint_control(
  joint_names: Sequence[str], records: Sequence[PositionActuatorRecord]
) -> JointControl:
  """Assemble per-joint control settings from position-actuator records.

  Joints without a position actuator keep ``NaN`` gains, an infinite effort
  limit and ``ctrl_index == -1``, so incomplete coverage is visible rather than
  silently zero-filled.
  """
  num_joints = len(joint_names)
  kp = torch.full((num_joints,), float("nan"), dtype=torch.float32)
  kd = torch.full((num_joints,), float("nan"), dtype=torch.float32)
  effort_limit = torch.full((num_joints,), float("inf"), dtype=torch.float32)
  delay_max_lag = torch.zeros(num_joints, dtype=torch.long)
  ctrl_index = torch.full((num_joints,), -1, dtype=torch.long)

  for record in records:
    if len(record.target_ids) != len(record.ctrl_ids):
      raise ValueError(
        "PositionActuatorRecord must pair one ctrl index per target joint; got "
        f"{len(record.target_ids)} targets and {len(record.ctrl_ids)} ctrl ids."
      )
    for joint_id, ctrl_id in zip(record.target_ids, record.ctrl_ids, strict=True):
      if not 0 <= joint_id < num_joints:
        raise ValueError(f"Joint index {joint_id} out of range [0, {num_joints}).")
      kp[joint_id] = record.kp
      kd[joint_id] = record.kd
      if record.effort_limit is not None:
        effort_limit[joint_id] = record.effort_limit
      delay_max_lag[joint_id] = record.delay_max_lag
      ctrl_index[joint_id] = ctrl_id

  return JointControl(
    joint_names=tuple(joint_names),
    kp=kp,
    kd=kd,
    effort_limit=effort_limit,
    delay_max_lag=delay_max_lag,
    ctrl_index=ctrl_index,
  )


def position_actuator_records(robot: Any) -> list[PositionActuatorRecord]:
  """Collect position-actuator records from a built entity."""
  records: list[PositionActuatorRecord] = []
  for actuator in robot.actuators:
    if not isinstance(actuator, BuiltinPositionActuator):
      continue
    cfg = actuator.cfg
    records.append(
      PositionActuatorRecord(
        target_ids=tuple(int(i) for i in actuator.target_ids.flatten().tolist()),
        ctrl_ids=tuple(int(i) for i in actuator.ctrl_ids.flatten().tolist()),
        kp=float(cfg.stiffness),
        kd=float(cfg.damping),
        effort_limit=None if cfg.effort_limit is None else float(cfg.effort_limit),
        delay_max_lag=int(cfg.delay_max_lag),
      )
    )
  return records


##
# Pure action-pipeline reconstruction (kept pure for unit tests).
##


def broadcast_per_joint(value: torch.Tensor | float, num_joints: int) -> torch.Tensor:
  """Resolve a scalar, (num_joints,) or (num_envs, num_joints) value to (J,).

  Action terms store a per-environment scale/offset tensor when the config is a
  dict; all environments share the same values, which is asserted here. The
  result is always a CPU tensor because callers use it for offline bookkeeping.
  """
  tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu")
  if tensor.ndim == 2:
    if tensor.shape[0] > 1 and not bool((tensor == tensor[:1]).all().item()):
      raise ValueError("Per-environment action scale/offset differs between envs.")
    tensor = tensor[0]
  tensor = tensor.reshape(-1)
  if tensor.numel() == 1:
    return tensor.repeat(num_joints)
  if tensor.numel() != num_joints:
    raise ValueError(f"Expected a scalar or {num_joints} values, got {tensor.numel()}.")
  return tensor.clone()


def reconstruct_processed_target(
  action: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor
) -> torch.Tensor:
  """Reproduce ``BaseAction.process_actions``: action * scale + offset.

  ``action`` is whatever reaches the action term, i.e. the wrapper-clipped
  action in this diagnostic. The optional term-level ``ActionTermCfg.clip``
  (applied after scale/offset) is not modeled here.
  """
  return action * scale + offset


def reconstruct_sim_target(
  processed_target: torch.Tensor, encoder_bias: torch.Tensor
) -> torch.Tensor:
  """Reproduce ``JointPositionAction.apply_actions``: target - encoder_bias."""
  return processed_target - encoder_bias


def reconstruct_pd_torque(
  sim_target: torch.Tensor,
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
  kp: torch.Tensor,
  kd: torch.Tensor,
) -> torch.Tensor:
  """Requested PD torque of a MuJoCo ``<position>`` actuator, before limits.

  A MuJoCo position actuator applies ``kp * (ctrl - q) - kd * qdot``, where the
  ctrl is the simulator position target, so no encoder bias enters here.
  """
  return kp * (sim_target - joint_pos) - kd * joint_vel


def clamp_effort(torque: torch.Tensor, effort_limit: torch.Tensor) -> torch.Tensor:
  """Clamp to the configured per-actuator effort limit (MuJoCo forcerange)."""
  return torch.clamp(torque, min=-effort_limit, max=effort_limit)


##
# Diagnostic configuration overrides.
##


@dataclass(frozen=True)
class Override:
  """One recorded deviation from the loaded configuration."""

  field: str
  before: str
  after: str
  reason: str


def _brief(value: Any, limit: int = 160) -> str:
  text = repr(value)
  return text if len(text) <= limit else text[: limit - 3] + "..."


def _set(
  records: list[Override], path: str, obj: Any, attr: str, value: Any, reason: str
) -> None:
  before = getattr(obj, attr)
  setattr(obj, attr, value)
  records.append(Override(path, _brief(before), _brief(value), reason))


def apply_nominal_overrides(
  env_cfg: ManagerBasedRlEnvCfg,
  *,
  motion_file: str,
  saved_lookahead_s: float | None,
  horizon_s: float,
  seed: int,
) -> list[Override]:
  """Make the loaded task config a deterministic, single-episode diagnostic.

  Only randomization, episode bookkeeping, the seed and the motion file are
  touched. The saved dynamics (timestep, decimation, actuator gains, effort
  limits, action scale/offset, command-delay settings) and the termination
  terms are preserved.
  """
  overrides: list[Override] = []

  motion_cmd = env_cfg.commands.get("motion")
  if not isinstance(motion_cmd, MotionCommandCfg):
    raise ValueError("Task is not a motion-tracking task (no MotionCommandCfg).")
  _set(
    overrides,
    "commands.motion.motion_file",
    motion_cmd,
    "motion_file",
    motion_file,
    "local motion artifact for the diagnostic",
  )
  if saved_lookahead_s is not None:
    _set(
      overrides,
      "commands.motion.lookahead_s",
      motion_cmd,
      "lookahead_s",
      saved_lookahead_s,
      "value saved next to the checkpoint (params/env.yaml)",
    )
  _set(
    overrides,
    "commands.motion.sampling_mode",
    motion_cmd,
    "sampling_mode",
    "start",
    "start-frame sampling is irrelevant: reset_to_frame writes the reference",
  )
  _set(
    overrides,
    "commands.motion.pose_range",
    motion_cmd,
    "pose_range",
    {},
    "reference-state randomization disabled for an exact start frame",
  )
  _set(
    overrides,
    "commands.motion.velocity_range",
    motion_cmd,
    "velocity_range",
    {},
    "reference-state randomization disabled for an exact start frame",
  )
  _set(
    overrides,
    "commands.motion.joint_position_range",
    motion_cmd,
    "joint_position_range",
    (0.0, 0.0),
    "reference-state randomization disabled for an exact start frame",
  )
  _set(
    overrides,
    "observations.actor.enable_corruption",
    env_cfg.observations["actor"],
    "enable_corruption",
    False,
    "nominal (noise-free) observations; NOT the training distribution",
  )
  _set(
    overrides,
    "events",
    env_cfg,
    "events",
    {},
    "startup/interval domain randomization removed: encoder_bias, base_com, "
    "foot_friction, push_robot",
  )
  _set(
    overrides,
    "episode_length_s",
    env_cfg,
    "episode_length_s",
    horizon_s,
    f"diagnostic horizon {horizon_s:.3f}s; only the episode limit is widened",
  )
  _set(
    overrides,
    "auto_reset",
    env_cfg,
    "auto_reset",
    False,
    "single continuous episode: stop on termination instead of stitching resets",
  )
  _set(
    overrides,
    "scene.num_envs",
    env_cfg.scene,
    "num_envs",
    1,
    "single-environment diagnostic",
  )
  _set(
    overrides,
    "seed",
    env_cfg,
    "seed",
    seed,
    "deterministic seed; all randomized events are disabled anyway",
  )
  return overrides


##
# Rollout.
##


class RolloutEnv(Protocol):
  """Minimal vector-env surface used by :func:`record_policy_commands`."""

  def get_observations(self) -> Any: ...

  def step(self, actions: torch.Tensor) -> Any: ...

  @property
  def unwrapped(self) -> Any: ...


@dataclass
class RolloutResult:
  """Recorded arrays plus rollout bookkeeping."""

  arrays: dict[str, np.ndarray]
  termination_term_names: list[str]
  steps_run: int
  end_reason: str
  end_detail: str
  achieved_ref_frame: int
  achieved_reference_s: float
  consistency: dict[str, float] = field(default_factory=dict)
  notes: list[str] = field(default_factory=list)


def _cpu(tensor: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
  """Detached CPU copy; safe to hold after the simulator advances."""
  return tensor.detach().to(device="cpu", dtype=dtype).clone()


def _numpy(tensor: torch.Tensor, dtype: torch.dtype = torch.float32) -> np.ndarray:
  return _cpu(tensor, dtype).numpy()


def record_policy_commands(
  *,
  wrapped: RolloutEnv,
  policy: Callable[[Any], torch.Tensor],
  command: Any,
  action_term: Any,
  joint_control: JointControl,
  steps: int,
  motion_fps: float,
  start_frame: int = 0,
) -> RolloutResult:
  """Run a bounded nominal rollout, snapshotting inputs before each step.

  ``env.step`` advances the motion command and refreshes observations, so the
  reference frame, the actor observation and the pre-step state are read
  *before* inference and before the step; the resulting action is never paired
  with a post-step observation.
  """
  if not joint_control.is_complete:
    raise RuntimeError(
      "No position actuator covers joints: "
      f"{joint_control.uncovered_joints}. The recorder assumes one MuJoCo "
      "<position> actuator per actuated joint."
    )
  raw_env = wrapped.unwrapped
  if raw_env.num_envs != 1:
    raise ValueError(
      f"The diagnostic records a single environment, got {raw_env.num_envs}."
    )
  robot = raw_env.scene["robot"]
  data = robot.data
  num_joints = joint_control.num_joints
  if data.joint_pos.shape[1] != num_joints:
    raise ValueError(
      f"Joint control lists {num_joints} joints but the entity has "
      f"{data.joint_pos.shape[1]}."
    )

  target_ids = _cpu(action_term.target_ids, torch.long).flatten()
  ctrl_index = joint_control.ctrl_index
  action_scale = broadcast_per_joint(action_term.scale, num_joints)
  action_offset = broadcast_per_joint(action_term.offset, num_joints)
  kp = joint_control.kp
  kd = joint_control.kd
  effort_limit = joint_control.effort_limit
  clip_actions = getattr(wrapped, "clip_actions", None)

  termination_names = list(raw_env.termination_manager.active_terms)
  rows: dict[str, list[np.ndarray]] = {}

  def add_scalar(name: str, value: Any, dtype: type = np.float32) -> None:
    rows.setdefault(name, []).append(np.array(value, dtype=dtype))

  def snapshot() -> dict[str, torch.Tensor]:
    """Copy the live (env 0) state; every value is a detached CPU copy."""
    entity_ctrl = data.data.ctrl[:, data.indexing.ctrl_ids]
    return {
      "sim_time": _cpu(raw_env.sim.data.time.flatten()[:1], torch.float64),
      "joint_pos": _cpu(data.joint_pos[0]),
      "joint_vel": _cpu(data.joint_vel[0]),
      "joint_pos_biased": _cpu(data.joint_pos_biased[0]),
      "encoder_bias": _cpu(data.encoder_bias[0]),
      "joint_target": _cpu(data.joint_pos_target[0]),
      "ctrl": _cpu(entity_ctrl[0][ctrl_index]),
      "qfrc_actuator": _cpu(data.qfrc_actuator[0]),
      "actuator_force": _cpu(data.actuator_force[0][ctrl_index]),
      "root_pose_w": _cpu(data.root_link_pose_w[0]),
      "root_vel_w": _cpu(data.root_link_vel_w[0]),
    }

  end_reason = "step_bound"
  end_detail = f"reached the requested {steps} policy steps"
  consistency = {
    "processed_target_from_clipped_action_max_abs_error": 0.0,
    "sim_target_from_processed_minus_bias_max_abs_error": 0.0,
    "torque_law_post_max_abs_error": 0.0,
  }

  for step in range(steps):
    frame = int(command.time_steps[0].item())
    expected_frame = start_frame + step
    if frame != expected_frame:
      raise RuntimeError(
        f"Reference frame desynchronized at step {step}: command is at frame "
        f"{frame}, expected {expected_frame}. Resets or motion wraparound would "
        "pair reference frames with the wrong observation; aborting."
      )

    # --- Snapshot the inputs of this step before inference. ---
    obs_input = wrapped.get_observations()
    actor_obs = obs_input["actor"]
    ref_joint_pos = _cpu(command.joint_pos[0])
    ref_joint_vel = _cpu(command.joint_vel[0])
    pre = snapshot()
    episode_length_pre = int(raw_env.episode_length_buf[0].item())

    with torch.no_grad():
      raw_action_dev = policy(obs_input)
    raw_action_dev = raw_action_dev.detach().to(dtype=torch.float32)
    if clip_actions is None:
      clipped_action_dev = raw_action_dev
    else:
      clipped_action_dev = torch.clamp(raw_action_dev, -clip_actions, clip_actions)
    raw_action = _cpu(raw_action_dev)[0]
    clipped_action = _cpu(clipped_action_dev)[0]

    # --- Step. env.step advances the command and produces obs_{t+1}. ---
    wrapped.step(clipped_action_dev)

    post = snapshot()
    episode_length_post = int(raw_env.episode_length_buf[0].item())
    terminated = bool(raw_env.termination_manager.terminated[0].item())
    truncated = bool(raw_env.termination_manager.time_outs[0].item())
    flags = np.array(
      [
        bool(raw_env.termination_manager.get_term(name)[0].item())
        for name in termination_names
      ],
      dtype=bool,
    )

    # The native command increments the reference frame by exactly one per step
    # and resamples (teleporting the robot and clearing its position targets) when
    # the frame reaches the end of the motion. Any other post-step frame means the
    # recorded row is not the tracked reference transition; physical termination
    # information for this step is carried into the error instead of being lost.
    post_frame = int(command.time_steps[0].item())
    if post_frame != expected_frame + 1:
      fired = [
        name for name, value in zip(termination_names, flags, strict=True) if value
      ]
      raise RuntimeError(
        f"Reference frame resampled after step {step}: expected frame "
        f"{expected_frame + 1}, command is at {post_frame}. Motion wraparound or a "
        "command resample invalidates the recorded row; aborting. Terminations "
        f"reported for this step: {', '.join(fired) if fired else 'none'}."
      )

    applied_action = _cpu(action_term.raw_action[0])
    applied_error = float((applied_action - clipped_action).abs().max().item())
    if applied_error > 1e-6:
      raise RuntimeError(
        f"Step {step}: the action term consumed a different action than the "
        f"recorder passed to the wrapper (max abs diff {applied_error:.2e})."
      )

    # ``_processed_actions`` is private state of the action term; it is the exact
    # value the environment applied this step. The public scale/offset
    # reconstruction below is cross-checked against it.
    processed_target = _cpu(action_term._processed_actions[0])
    processed_reconstructed = reconstruct_processed_target(
      clipped_action, action_scale, action_offset
    )
    processed_error = float(
      (processed_reconstructed - processed_target).abs().max().item()
    )
    # The position target is written by ``apply_action`` inside ``step``, so the
    # target this step used is only available after stepping.
    sim_target = post["joint_target"][target_ids]
    sim_target_reconstructed = reconstruct_sim_target(
      processed_target, pre["encoder_bias"][target_ids]
    )
    sim_target_error = float((sim_target_reconstructed - sim_target).abs().max().item())

    requested_torque = reconstruct_pd_torque(
      sim_target, pre["joint_pos"][target_ids], pre["joint_vel"][target_ids], kp, kd
    )
    applied_torque = clamp_effort(requested_torque, effort_limit)
    # Cross-check the logged gains and joint mapping against the simulator: for a
    # plain MuJoCo <position> actuator, the post-step actuator torque must match
    # kp*(target - q) - kd*dq evaluated at the post-step state.
    torque_law_error = float(
      (
        clamp_effort(
          reconstruct_pd_torque(
            sim_target,
            post["joint_pos"][target_ids],
            post["joint_vel"][target_ids],
            kp,
            kd,
          ),
          effort_limit,
        )
        - post["qfrc_actuator"][target_ids]
      )
      .abs()
      .max()
      .item()
    )
    consistency["processed_target_from_clipped_action_max_abs_error"] = max(
      consistency["processed_target_from_clipped_action_max_abs_error"],
      processed_error,
    )
    consistency["sim_target_from_processed_minus_bias_max_abs_error"] = max(
      consistency["sim_target_from_processed_minus_bias_max_abs_error"],
      sim_target_error,
    )
    consistency["torque_law_post_max_abs_error"] = max(
      consistency["torque_law_post_max_abs_error"], torque_law_error
    )

    add_scalar("policy_step", step, np.int64)
    add_scalar("ref_frame", frame, np.int64)
    add_scalar("ref_time_s", frame / motion_fps)
    add_scalar("sim_time_pre", float(pre["sim_time"][0].item()), np.float64)
    add_scalar("sim_time_post", float(post["sim_time"][0].item()), np.float64)
    add_scalar("episode_length_pre", episode_length_pre, np.int64)
    add_scalar("episode_length_post", episode_length_post, np.int64)
    rows.setdefault("actor_observation", []).append(_numpy(actor_obs[0]))
    rows.setdefault("raw_action", []).append(_numpy(raw_action))
    rows.setdefault("clipped_action", []).append(_numpy(clipped_action))
    rows.setdefault("processed_target", []).append(_numpy(processed_target))
    rows.setdefault("sim_target", []).append(_numpy(sim_target))
    rows.setdefault("ctrl", []).append(_numpy(post["ctrl"]))
    rows.setdefault("ref_joint_pos", []).append(_numpy(ref_joint_pos))
    rows.setdefault("ref_joint_vel", []).append(_numpy(ref_joint_vel))
    rows.setdefault("joint_pos_pre", []).append(_numpy(pre["joint_pos"]))
    rows.setdefault("joint_vel_pre", []).append(_numpy(pre["joint_vel"]))
    rows.setdefault("joint_pos_post", []).append(_numpy(post["joint_pos"]))
    rows.setdefault("joint_vel_post", []).append(_numpy(post["joint_vel"]))
    rows.setdefault("joint_pos_biased_pre", []).append(_numpy(pre["joint_pos_biased"]))
    rows.setdefault("joint_pos_biased_post", []).append(
      _numpy(post["joint_pos_biased"])
    )
    rows.setdefault("encoder_bias", []).append(_numpy(pre["encoder_bias"]))
    rows.setdefault("qfrc_actuator_post", []).append(_numpy(post["qfrc_actuator"]))
    rows.setdefault("actuator_force_post", []).append(_numpy(post["actuator_force"]))
    rows.setdefault("pd_torque_requested_pre", []).append(_numpy(requested_torque))
    rows.setdefault("pd_torque_applied_pre", []).append(_numpy(applied_torque))
    rows.setdefault("root_link_pose_w_pre", []).append(_numpy(pre["root_pose_w"]))
    rows.setdefault("root_link_vel_w_pre", []).append(_numpy(pre["root_vel_w"]))
    rows.setdefault("root_link_pose_w_post", []).append(_numpy(post["root_pose_w"]))
    rows.setdefault("root_link_vel_w_post", []).append(_numpy(post["root_vel_w"]))
    add_scalar("terminated", terminated, bool)
    add_scalar("truncated", truncated, bool)
    add_scalar("done", terminated or truncated, bool)
    rows.setdefault("termination_flags", []).append(flags)

    if processed_error > 1e-5:
      raise RuntimeError(
        f"Step {step}: processed target reconstructed from clipped*scale+offset "
        f"differs from the action term value by {processed_error:.2e}. The "
        "reconstruction models wrapper action clipping only; a term-level "
        "ActionTermCfg.clip (applied after scale/offset) is not modeled."
      )
    if sim_target_error > 1e-5:
      raise RuntimeError(
        f"Step {step}: processed target minus encoder bias differs from the "
        f"simulator position target by {sim_target_error:.2e}."
      )

    if terminated or truncated:
      fired = [
        name for name, value in zip(termination_names, flags, strict=True) if value
      ]
      end_reason = "terminated" if terminated else "truncated"
      end_detail = (
        f"stopped at policy step {step} (frame {frame}, "
        f"{frame / motion_fps:.3f}s of reference): {', '.join(fired)}"
      )
      break

  arrays = {name: np.stack(values) for name, values in rows.items()}
  steps_run = int(arrays["policy_step"].shape[0])
  return RolloutResult(
    arrays=arrays,
    termination_term_names=termination_names,
    steps_run=steps_run,
    end_reason=end_reason,
    end_detail=end_detail,
    achieved_ref_frame=int(arrays["ref_frame"][-1]),
    achieved_reference_s=float(arrays["ref_frame"][-1]) / motion_fps,
    consistency=consistency,
    notes=[],
  )


##
# Outputs.
##


def output_paths(out_dir: Path, *, with_onnx: bool) -> dict[str, Path]:
  """Resolve the artifact paths, including the optional ONNX parity report."""
  paths = {
    NPZ_NAME: out_dir / NPZ_NAME,
    CSV_NAME: out_dir / CSV_NAME,
    OBS_CSV_NAME: out_dir / OBS_CSV_NAME,
    METADATA_NAME: out_dir / METADATA_NAME,
  }
  if with_onnx:
    paths[ONNX_PARITY_NAME] = out_dir / ONNX_PARITY_NAME
  return paths


def check_fresh_outputs(paths: dict[str, Path]) -> None:
  """Refuse to overwrite an existing recording."""
  existing = [str(path) for path in paths.values() if path.exists()]
  if existing:
    raise FileExistsError(
      "Refusing to overwrite existing recording files: "
      + ", ".join(existing)
      + ". Choose a fresh --output-dir."
    )


CSV_COLUMNS = (
  "step",
  "sim_time_pre",
  "sim_time_post",
  "ref_frame",
  "ref_time_s",
  "joint",
  "ref_joint_pos",
  "ref_joint_vel",
  "joint_pos_pre",
  "joint_vel_pre",
  "joint_pos_biased_pre",
  "joint_pos_post",
  "joint_vel_post",
  "joint_pos_biased_post",
  "encoder_bias",
  "raw_action",
  "clipped_action",
  "processed_target",
  "sim_target",
  "ctrl",
  "qfrc_actuator_post",
  "actuator_force_post",
  "pd_torque_requested_pre",
  "pd_torque_applied_pre",
  "kp",
  "kd",
  "effort_limit",
  "terminated",
  "done",
)


def write_joint_csv(
  path: Path, result: RolloutResult, joint_control: JointControl
) -> None:
  """Write one row per (policy step, joint): readable and greppable."""
  arrays = result.arrays
  num_joints = joint_control.num_joints
  with path.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(CSV_COLUMNS)
    for step in range(result.steps_run):
      common = [
        step,
        f"{float(arrays['sim_time_pre'][step]):.6f}",
        f"{float(arrays['sim_time_post'][step]):.6f}",
        int(arrays["ref_frame"][step]),
        f"{float(arrays['ref_time_s'][step]):.6f}",
      ]
      for joint_id in range(num_joints):
        writer.writerow(
          common
          + [
            joint_control.joint_names[joint_id],
            f"{float(arrays['ref_joint_pos'][step, joint_id]):.9g}",
            f"{float(arrays['ref_joint_vel'][step, joint_id]):.9g}",
            f"{float(arrays['joint_pos_pre'][step, joint_id]):.9g}",
            f"{float(arrays['joint_vel_pre'][step, joint_id]):.9g}",
            f"{float(arrays['joint_pos_biased_pre'][step, joint_id]):.9g}",
            f"{float(arrays['joint_pos_post'][step, joint_id]):.9g}",
            f"{float(arrays['joint_vel_post'][step, joint_id]):.9g}",
            f"{float(arrays['joint_pos_biased_post'][step, joint_id]):.9g}",
            f"{float(arrays['encoder_bias'][step, joint_id]):.9g}",
            f"{float(arrays['raw_action'][step, joint_id]):.9g}",
            f"{float(arrays['clipped_action'][step, joint_id]):.9g}",
            f"{float(arrays['processed_target'][step, joint_id]):.9g}",
            f"{float(arrays['sim_target'][step, joint_id]):.9g}",
            f"{float(arrays['ctrl'][step, joint_id]):.9g}",
            f"{float(arrays['qfrc_actuator_post'][step, joint_id]):.9g}",
            f"{float(arrays['actuator_force_post'][step, joint_id]):.9g}",
            f"{float(arrays['pd_torque_requested_pre'][step, joint_id]):.9g}",
            f"{float(arrays['pd_torque_applied_pre'][step, joint_id]):.9g}",
            f"{float(joint_control.kp[joint_id]):.9g}",
            f"{float(joint_control.kd[joint_id]):.9g}",
            f"{float(joint_control.effort_limit[joint_id]):.9g}",
            int(bool(arrays["terminated"][step])),
            int(bool(arrays["done"][step])),
          ]
        )


def write_observation_csv(path: Path, result: RolloutResult) -> None:
  """Write the actor input observation per step, one column per element."""
  arrays = result.arrays
  observations = arrays["actor_observation"]
  with path.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(
      ["step", "sim_time_pre", "ref_frame"]
      + [f"obs_{i:03d}" for i in range(observations.shape[1])]
    )
    for step in range(result.steps_run):
      writer.writerow(
        [
          step,
          f"{float(arrays['sim_time_pre'][step]):.6f}",
          int(arrays["ref_frame"][step]),
        ]
        + [f"{float(value):.9g}" for value in observations[step]]
      )


def summarize(
  result: RolloutResult, joint_control: JointControl
) -> dict[str, dict[str, float]]:
  """Per-joint target-vs-reference deviation and torque utilization."""
  arrays = result.arrays
  delta_deg = np.degrees(arrays["sim_target"] - arrays["ref_joint_pos"])
  limit = joint_control.effort_limit.numpy()
  bounded = np.where(np.isfinite(limit), limit, np.nan)[None, :]
  utilization = np.abs(arrays["qfrc_actuator_post"]) / bounded
  summary: dict[str, dict[str, float]] = {}
  for joint_id, name in enumerate(joint_control.joint_names):
    summary[name] = {
      "target_minus_ref_deg_mean": float(delta_deg[:, joint_id].mean()),
      "target_minus_ref_deg_min": float(delta_deg[:, joint_id].min()),
      "target_minus_ref_deg_max": float(delta_deg[:, joint_id].max()),
      "torque_utilization_max": float(np.nanmax(utilization[:, joint_id])),
      "ref_q_min_deg": float(np.degrees(arrays["ref_joint_pos"][:, joint_id]).min()),
      "ref_q_max_deg": float(np.degrees(arrays["ref_joint_pos"][:, joint_id]).max()),
    }
  return summary


def write_outputs(
  paths: dict[str, Path],
  *,
  result: RolloutResult,
  joint_control: JointControl,
  metadata: dict[str, Any],
  scalars: dict[str, float],
) -> dict[str, Path]:
  """Write the NPZ, the two CSVs and the metadata JSON, returning real paths."""
  check_fresh_outputs(paths)
  paths[NPZ_NAME].parent.mkdir(parents=True, exist_ok=True)

  arrays = dict(result.arrays)
  arrays["joint_names"] = np.array(joint_control.joint_names)
  arrays["kp"] = joint_control.kp.numpy()
  arrays["kd"] = joint_control.kd.numpy()
  arrays["effort_limit"] = joint_control.effort_limit.numpy()
  arrays["delay_max_lag"] = joint_control.delay_max_lag.numpy()
  arrays["ctrl_index"] = joint_control.ctrl_index.numpy()
  arrays["termination_term_names"] = np.array(result.termination_term_names)
  arrays["episode_index"] = np.array([0], dtype=np.int64)
  arrays["num_resets_during_recording"] = np.array([0], dtype=np.int64)
  for key, value in scalars.items():
    arrays[key] = np.array([value], dtype=np.float32)
  arrays_any: dict[str, Any] = dict(arrays)
  np.savez_compressed(paths[NPZ_NAME], **arrays_any)

  write_joint_csv(paths[CSV_NAME], result, joint_control)
  write_observation_csv(paths[OBS_CSV_NAME], result)
  with paths[METADATA_NAME].open("w") as handle:
    json.dump(metadata, handle, indent=2, default=str)
    handle.write("\n")
  return paths


##
# ONNX parity (optional).
##


def _onnx_run(session: Any, input_name: str, observation: np.ndarray) -> np.ndarray:
  """Run one ONNX inference and return the (action_dim,) float32 output."""
  outputs = session.run(None, {input_name: observation})
  return np.asarray(outputs[0], dtype=np.float32).reshape(-1)


def _policy_single(
  policy: Callable[[Any], torch.Tensor],
  observation: np.ndarray,
  device: str,
  group: str,
) -> torch.Tensor:
  """Run the actor on one observation, exactly as the rollout did."""
  from tensordict import TensorDict

  with torch.no_grad():
    return policy(
      TensorDict({group: torch.from_numpy(observation).to(device)}, batch_size=[1])
    ).detach()


def run_onnx_parity(
  *,
  onnx_file: Path,
  result: RolloutResult,
  policy: Callable[[Any], torch.Tensor],
  device: str,
  max_samples: int,
) -> dict[str, Any]:
  """Compare the deploy ONNX export with the checkpoint actor on identical inputs.

  The recorded actor observations are the exact inputs the checkpoint policy saw,
  so both runtimes receive identical, unnormalized observations and each applies
  its own baked-in normalization exactly once. This is an export check only: it
  says nothing about whether native and deployed closed-loop behavior match.
  """
  try:
    import onnxruntime as ort
  except ModuleNotFoundError as exc:
    raise RuntimeError(
      "--onnx-file requires onnxruntime, a dev dependency (uv sync --group dev)."
    ) from exc

  observations = result.arrays["actor_observation"]
  recorded_actions = result.arrays["raw_action"]
  num_steps = observations.shape[0]
  sample_count = min(max(1, max_samples), num_steps)
  indices = np.unique(np.linspace(0, num_steps - 1, sample_count).round().astype(int))

  session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
  inputs = session.get_inputs()
  outputs = session.get_outputs()
  if len(inputs) != 1:
    raise RuntimeError(
      f"Expected a single ONNX input, got {[i.name for i in inputs]}. The "
      "auto-exported wrapped time_step ONNX is not a deploy raw export."
    )
  onnx_actions = np.stack(
    [_onnx_run(session, inputs[0].name, observations[i : i + 1]) for i in indices]
  )

  # Recompute the actor output one sample at a time, matching the batch shape the
  # rollout used, so any difference is a real reproducibility failure rather
  # than a batched-GEMM rounding difference.
  obs_group = str(getattr(policy, "obs_groups", ("actor",))[0])
  torch_actions = np.stack(
    [
      _numpy(
        _policy_single(policy, observations[i : i + 1], device, obs_group)
      ).reshape(-1)
      for i in indices
    ]
  )

  diff = np.abs(onnx_actions - recorded_actions[indices])
  determinism = np.abs(torch_actions - recorded_actions[indices])
  return {
    "onnx_file": str(onnx_file),
    "inputs": [
      {"name": i.name, "shape": list(i.shape), "type": i.type} for i in inputs
    ],
    "outputs": [
      {"name": o.name, "shape": list(o.shape), "type": o.type} for o in outputs
    ],
    "samples_tested": int(len(indices)),
    "sample_indices": [int(i) for i in indices],
    "action_dim": int(recorded_actions.shape[1]),
    "max_abs_error": float(diff.max()),
    "mean_abs_error": float(diff.mean()),
    "tolerance": ONNX_TOLERANCE,
    "within_tolerance": bool(diff.max() <= ONNX_TOLERANCE),
    "torch_recompute_max_abs_error": float(determinism.max()),
    "normalization_applied": "once per runtime (baked into actor and export)",
    "interpretation": (
      "Numerical export check on identical actor inputs only; it does not show "
      "that native-simulator and deployed closed-loop trajectories agree."
    ),
  }


##
# Provenance helpers.
##


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


# Sentinel distinguishing an absent saved key from a key whose saved value is
# null (for example ``actions.joint_pos.clip: null``, which is a real value).
_MISSING = object()
MISSING_DETAIL = "not present in the saved params/env.yaml"


def _saved_value(saved: dict, path: str) -> Any:
  """Look up a dotted path in the saved config; ``_MISSING`` when absent."""
  node: Any = saved
  for key in path.split("."):
    if not isinstance(node, dict) or key not in node:
      return _MISSING
    node = node[key]
  return node


def _saved_or_none(saved: dict, path: str) -> Any:
  value = _saved_value(saved, path)
  return None if value is _MISSING else value


# Saved-config paths recorded for provenance. A key is missing from the report
# (rather than silently treated as comparable) when it is absent from the saved
# params/env.yaml; see collect_saved_provenance for how each is verified.
_SAVED_SCALAR_PATHS = {
  "decimation": "decimation",
  "physics_dt": "sim.mujoco.timestep",
  "lookahead_s": "commands.motion.lookahead_s",
  "action_scale": "actions.joint_pos.scale",
  "action_clip": "actions.joint_pos.clip",
  "use_default_offset": "actions.joint_pos.use_default_offset",
  "sampling_mode": "commands.motion.sampling_mode",
  "init_weight_s": "commands.motion.init_weight_s",
  "seed": "seed",
}

# Actuator entries live in their own comparison; kept separate from the scalar
# provenance dump so the raw saved list is not duplicated there.
_SAVED_ACTUATOR_PATH = "scene.entities.robot.articulation.actuators"


def compare_saved_action_scale(
  saved_scale: Any, joint_names: Sequence[str], effective: torch.Tensor
) -> float | None:
  """Max |saved - effective| action scale over joints matched by regex.

  Returns ``None`` when the saved value is not a per-joint dict or when a saved
  pattern matches no joint, so an unverifiable comparison is visible instead of
  silently reported as equal.
  """
  if not isinstance(saved_scale, dict):
    return None
  effective_np = broadcast_per_joint(effective, len(joint_names)).numpy()
  worst = 0.0
  covered: set[int] = set()
  for pattern, value in saved_scale.items():
    regex = re.compile(f"^(?:{pattern})$")
    matched = [i for i, name in enumerate(joint_names) if regex.match(name)]
    if not matched:
      return None
    covered.update(matched)
    for index in matched:
      # The action term stores float32 scales; compare in that precision.
      worst = max(worst, abs(float(np.float32(value)) - float(effective_np[index])))
  if len(covered) != len(joint_names):
    return None
  return worst


# Per-joint actuator settings compared between the saved training config and the
# loaded task config. ``effort_limit`` may legitimately be unbounded; the other
# settings have no unbounded representation.
ACTUATOR_SETTING_KEYS = (
  "stiffness",
  "damping",
  "effort_limit",
  "delay_min_lag",
  "delay_max_lag",
)


@dataclass(frozen=True)
class SettingCheck:
  """Result of comparing one saved setting against the effective config."""

  status: str
  """One of ``match``, ``mismatch`` or ``unverifiable``."""
  detail: str
  saved: Any = None
  effective: Any = None
  max_abs_diff: float | None = None
  differing_joints: tuple[str, ...] = ()


def _unwrap_setting_value(entry: Any, key: str) -> tuple[float | None, str | None]:
  """Read one actuator setting from a plain-data entry.

  Returns ``(value, error)``. ``effort_limit: None`` means unbounded and maps to
  ``inf``; ``None`` for any other setting is unsupported.
  """
  if not isinstance(entry, dict):
    return None, "entry is not a mapping"
  if key not in entry:
    return None, f"entry does not specify '{key}'"
  value = entry[key]
  if value is None:
    if key == "effort_limit":
      return float("inf"), None
    return None, f"entry has an unspecified (null) '{key}'"
  try:
    numeric = float(value)
  except (TypeError, ValueError):
    return None, f"entry has a non-numeric '{key}': {value!r}"
  if not math.isfinite(numeric) and not (
    key == "effort_limit" and numeric == float("inf")
  ):
    return None, f"entry has an unsupported non-finite '{key}': {value!r}"
  return numeric, None


def _as_actuator_entries(value: Any) -> tuple[list[Any] | None, str]:
  """Normalize a saved actuator list to plain-data entries."""
  if value is None:
    return None, "no saved actuator entries in params/env.yaml"
  if not isinstance(value, (list, tuple)):
    return (
      None,
      f"saved actuator entries have an unsupported type: {type(value).__name__}",
    )
  if not value:
    return None, "the saved actuator entry list is empty"
  if not all(isinstance(entry, dict) for entry in value):
    return None, "saved actuator entries are not plain mappings"
  return list(value), ""


def _effective_actuator_entries(
  env_cfg: ManagerBasedRlEnvCfg,
) -> tuple[list[dict] | None, str]:
  """Read the loaded task config's robot actuator settings as plain data."""
  entity_cfg = env_cfg.scene.entities.get("robot")
  articulation = getattr(entity_cfg, "articulation", None)
  actuators = getattr(articulation, "actuators", None)
  if not isinstance(actuators, (list, tuple)) or not actuators:
    return None, "the loaded task config exposes no robot actuator list"
  entries: list[dict] = []
  for actuator_cfg in actuators:
    entries.append(
      {
        "target_names_expr": tuple(
          getattr(actuator_cfg, "target_names_expr", ()) or ()
        ),
        **{key: getattr(actuator_cfg, key, None) for key in ACTUATOR_SETTING_KEYS},
      }
    )
  return entries, ""


def resolve_actuator_setting_per_joint(
  entries: list[Any] | None, joint_names: Sequence[str], key: str
) -> tuple[list[float] | None, str]:
  """Resolve actuator entries to one value per joint.

  Returns ``(values, detail)``; ``values`` is ``None`` when the entries are
  missing, incomplete, ambiguous or unsupported, and ``detail`` then says why.
  A joint covered by more than one entry is ambiguous rather than last-wins.
  """
  if entries is None:
    return None, f"no actuator entries available for '{key}'"
  values: list[float] = [float("nan")] * len(joint_names)
  claims = [0] * len(joint_names)
  for index, entry in enumerate(entries):
    patterns = entry.get("target_names_expr") if isinstance(entry, dict) else None
    if isinstance(patterns, str):
      patterns = (patterns,)
    if not isinstance(patterns, (list, tuple)) or not patterns:
      return None, f"entry {index} has no usable 'target_names_expr'"
    if not all(isinstance(pattern, str) for pattern in patterns):
      return None, f"entry {index} has a non-string 'target_names_expr' entry"
    value, error = _unwrap_setting_value(entry, key)
    if error is not None:
      return None, f"entry {index} ({patterns[0]!r}): {error}"
    try:
      matched_ids, _ = resolve_matching_names(list(patterns), list(joint_names))
    except (ValueError, re.error) as exc:
      return None, f"entry {index} has unresolved target patterns {patterns}: {exc}"
    if not matched_ids:
      return None, f"entry {index} pattern(s) {patterns} match no joint"
    for joint_id in matched_ids:
      claims[joint_id] += 1
      values[joint_id] = float(value)  # type: ignore[arg-type]
  uncovered = [joint_names[i] for i, count in enumerate(claims) if count == 0]
  if uncovered:
    return None, f"uncovered joints: {uncovered}"
  ambiguous = [joint_names[i] for i, count in enumerate(claims) if count > 1]
  if ambiguous:
    return None, f"joints matched more than once: {ambiguous}"
  return values, "resolved one value per joint"


def compare_saved_actuator_settings(
  saved_actuators: Any,
  joint_names: Sequence[str],
  effective_actuators: list[Any] | None,
  effective_detail: str,
) -> dict[str, SettingCheck]:
  """Compare saved and effective per-joint kp/kd/effort-limit/delay settings.

  A setting is only reported as ``match`` after both sides were resolved to one
  value per joint and compared. Absent, incomplete, ambiguous or unsupported
  data on either side yields ``unverifiable``.
  """
  saved_entries, saved_detail = _as_actuator_entries(saved_actuators)
  checks: dict[str, SettingCheck] = {}
  for key in ACTUATOR_SETTING_KEYS:
    if saved_entries is None:
      checks[key] = SettingCheck("unverifiable", f"saved: {saved_detail}")
      continue
    if effective_actuators is None:
      checks[key] = SettingCheck("unverifiable", f"effective: {effective_detail}")
      continue
    saved_values, saved_reason = resolve_actuator_setting_per_joint(
      saved_entries, joint_names, key
    )
    if saved_values is None:
      checks[key] = SettingCheck("unverifiable", f"saved: {saved_reason}")
      continue
    effective_values, effective_reason = resolve_actuator_setting_per_joint(
      effective_actuators, joint_names, key
    )
    if effective_values is None:
      checks[key] = SettingCheck(
        "unverifiable", f"effective: {effective_reason}", saved=saved_values
      )
      continue
    # Equal unbounded limits are equal, not inf - inf (NaN). A finite limit
    # versus an unbounded one remains a mismatch with an infinite difference.
    diffs = [
      0.0 if saved == effective else abs(saved - effective)
      for saved, effective in zip(saved_values, effective_values, strict=True)
    ]
    differing = tuple(
      name
      for name, diff in zip(joint_names, diffs, strict=True)
      if not math.isclose(diff, 0.0, rel_tol=0.0, abs_tol=1e-9)
    )
    # Both sides are plain floats from YAML and dataclass configs, so this is a
    # float64 comparison with a guard band for repr round-tripping.
    checks[key] = SettingCheck(
      status="mismatch" if differing else "match",
      detail="compared per joint (float64, abs tolerance 1e-9)",
      saved=saved_values,
      effective=effective_values,
      max_abs_diff=max(diffs),
      differing_joints=differing,
    )
  return checks


def collect_saved_provenance(
  checkpoint_file: Path,
  saved: dict,
  env_cfg: ManagerBasedRlEnvCfg,
  joint_control: JointControl,
  action_scale: torch.Tensor,
) -> dict[str, Any]:
  """Compare saved training settings with the effective diagnostic config.

  Every ``fidelity_checks`` entry is computed from the saved ``params/env.yaml``
  against the loaded task config, so a check is only ``match`` when both sides
  were actually compared. Absent, incomplete, ambiguous or unsupported saved
  data is reported as ``unverifiable`` rather than assumed equal.
  """
  env_yaml = checkpoint_file.parent / "params" / "env.yaml"
  agent_yaml = checkpoint_file.parent / "params" / "agent.yaml"
  saved_values = {
    key: _saved_or_none(saved, path) for key, path in _SAVED_SCALAR_PATHS.items()
  }

  checks: dict[str, Any] = {}
  compared: list[str] = []

  def record(name: str, check: SettingCheck) -> None:
    compared.append(name)
    checks[name] = asdict(check)

  def compare_exact(name: str, saved_path: str, effective_value: Any) -> None:
    saved_value = _saved_value(saved, saved_path)
    if saved_value is _MISSING:
      record(
        name,
        SettingCheck("unverifiable", MISSING_DETAIL, effective=effective_value),
      )
    else:
      record(
        name,
        SettingCheck(
          status="match" if saved_value == effective_value else "mismatch",
          detail=(
            "compared (exact, null values)"
            if saved_value is None
            else "compared (exact)"
          ),
          saved=saved_value,
          effective=effective_value,
        ),
      )

  compare_exact("decimation", _SAVED_SCALAR_PATHS["decimation"], env_cfg.decimation)
  compare_exact(
    "physics_dt",
    _SAVED_SCALAR_PATHS["physics_dt"],
    env_cfg.sim.mujoco.timestep,
  )

  motion_cmd = env_cfg.commands.get("motion")
  lookahead_s = _saved_value(saved, _SAVED_SCALAR_PATHS["lookahead_s"])
  if not isinstance(motion_cmd, MotionCommandCfg):
    record(
      "lookahead_s",
      SettingCheck(
        "unverifiable",
        "the loaded task config exposes no motion command",
        saved=None if lookahead_s is _MISSING else lookahead_s,
      ),
    )
  elif lookahead_s is _MISSING:
    record(
      "lookahead_s",
      SettingCheck(
        "unverifiable",
        MISSING_DETAIL,
        effective=float(motion_cmd.lookahead_s),
      ),
    )
  else:
    effective_lookahead_s = float(motion_cmd.lookahead_s)
    record(
      "lookahead_s",
      SettingCheck(
        status=(
          "match"
          if math.isclose(
            float(lookahead_s), effective_lookahead_s, rel_tol=0.0, abs_tol=1e-12
          )
          else "mismatch"
        ),
        detail="compared (abs tolerance 1e-12)",
        saved=lookahead_s,
        effective=effective_lookahead_s,
      ),
    )

  action_cfg = env_cfg.actions.get("joint_pos")
  compare_exact(
    "action_clip", _SAVED_SCALAR_PATHS["action_clip"], getattr(action_cfg, "clip", None)
  )
  if isinstance(action_cfg, JointPositionActionCfg):
    compare_exact(
      "use_default_offset",
      _SAVED_SCALAR_PATHS["use_default_offset"],
      action_cfg.use_default_offset,
    )
  else:
    record(
      "use_default_offset",
      SettingCheck(
        "unverifiable",
        "the loaded task config exposes no joint position action",
        saved=saved_values["use_default_offset"],
      ),
    )

  scale_diff = compare_saved_action_scale(
    saved_values["action_scale"], joint_control.joint_names, action_scale
  )
  record(
    "action_scale",
    SettingCheck(
      status=(
        "unverifiable"
        if scale_diff is None
        else ("match" if scale_diff <= 1e-6 else "mismatch")
      ),
      detail=(
        "compared per joint (float32 action-term scale, abs tolerance 1e-6)"
        if scale_diff is not None
        else "saved action scale is missing, partial or of an unsupported shape"
      ),
      max_abs_diff=scale_diff,
    ),
  )

  effective_entries, effective_detail = _effective_actuator_entries(env_cfg)
  actuator_checks = compare_saved_actuator_settings(
    _saved_or_none(saved, _SAVED_ACTUATOR_PATH),
    joint_control.joint_names,
    effective_entries,
    effective_detail,
  )
  compared.append("actuator_settings")
  checks["actuator_settings"] = {
    "saved_source": "params/env.yaml scene.entities.robot.articulation.actuators",
    "effective_source": "loaded task config scene.entities.robot.articulation.actuators",
    "settings": {key: asdict(check) for key, check in actuator_checks.items()},
    "unverifiable_settings": [
      key for key, check in actuator_checks.items() if check.status == "unverifiable"
    ],
    "mismatched_settings": [
      key for key, check in actuator_checks.items() if check.status == "mismatch"
    ],
  }

  # Effective command delay: stated as effective-only, with the saved comparison
  # reported separately under actuator_settings.
  max_delay = int(joint_control.delay_max_lag.max().item())
  checks["effective_command_delay"] = {
    "max_delay_lag_steps": max_delay,
    "enabled": bool(max_delay > 0),
    "detail": (
      "describes the effective (loaded) actuator config only; the saved-versus-"
      "effective delay comparison is under actuator_settings"
    ),
  }

  return {
    "env_yaml": _file_record(env_yaml),
    "agent_yaml": _file_record(agent_yaml),
    "saved_values": saved_values,
    "fidelity_checks": checks,
    "saved_values_not_compared": sorted(set(saved_values) - set(compared)),
    "notes": (
      "Fidelity is asserted only where a fidelity_checks entry carries a status: "
      "the saved params/env.yaml and the loaded task config are each resolved to "
      "one value per joint and compared, and the status is 'unverifiable' when "
      "the saved data is absent, incomplete, ambiguous or of an unsupported "
      "shape. saved_values entries listed in saved_values_not_compared are "
      "recorded for context only. agent.yaml is recorded by path/hash/size and no "
      "agent setting is compared. Saved sampling_mode/init_weight_s only affect "
      "start-frame sampling and are superseded by reset_to_frame here."
    ),
  }


def _file_record(path: Path) -> dict[str, Any] | None:
  if not path.exists():
    return None
  return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def check_motion_window(start_frame: int, steps: int, motion_frames: int) -> None:
  """Reject a request whose final step would wrap the native motion command.

  The native command increments the reference frame by one after every step and
  resamples (teleporting the robot and clearing its position targets) as soon as
  the frame reaches ``time_step_total``. Step ``i`` therefore needs
  ``start_frame + i + 1 < motion_frames``, so the last step's post-step frame is
  the real boundary: ``start_frame + steps == motion_frames`` is already unsafe,
  not just an overflow.
  """
  if steps < 1:
    raise ValueError(f"steps must be >= 1, got {steps}.")
  if start_frame + steps >= motion_frames:
    max_safe = motion_frames - start_frame - 1
    raise ValueError(
      f"Requested {steps} steps from frame {start_frame} of a {motion_frames}-frame "
      "motion, but the native command increments the reference frame after every "
      "step and resamples (teleporting the robot and clearing its targets) once "
      "the frame reaches the end of the motion. Need "
      f"start_frame + steps < motion_frames, i.e. at most {max_safe} steps from "
      f"frame {start_frame}."
    )


##
# Entry point.
##


def main(
  checkpoint_file: str,
  output_dir: str,
  motion_file: str = DEFAULT_MOTION_FILE,
  task: str = DEFAULT_TASK,
  device: str | None = None,
  seed: int | None = None,
  duration_s: float = 15.0,
  max_steps: int | None = None,
  start_frame: int = 0,
  onnx_file: str | None = None,
  onnx_max_samples: int = 200,
  num_envs: int = 1,
) -> None:
  """Record the policy's simulator commands for a tracking checkpoint.

  Args:
    checkpoint_file: Trained ``model_*.pt`` checkpoint (with ``params/env.yaml``).
    output_dir: Fresh directory for the recording artifacts.
    motion_file: Tracking-format motion npz used to build the task.
    task: Task id used to build the environment.
    device: Device for the simulator; defaults to CUDA if available.
    seed: Environment seed; defaults to the seed saved next to the checkpoint.
    duration_s: Recording length in seconds (default 15s = 750 steps at 50Hz).
    max_steps: Optional hard bound on the policy steps (bounded smoke runs).
    start_frame: Reference frame the recording starts at (default frame 0).
    onnx_file: Optional deploy ONNX export to compare against the checkpoint.
    onnx_max_samples: Maximum number of recorded steps used for the ONNX parity.
    num_envs: Number of environments; the diagnostic supports exactly one.
  """
  if num_envs != 1:
    raise ValueError(
      f"The diagnostic records a single environment, got num_envs={num_envs}."
    )
  if start_frame < 0:
    raise ValueError(f"start_frame must be non-negative, got {start_frame}.")
  configure_torch_backends()
  device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  checkpoint_path = Path(checkpoint_file).expanduser().resolve()
  motion_path = Path(motion_file).expanduser().resolve()
  for path in (checkpoint_path, motion_path):
    if not path.exists():
      raise FileNotFoundError(f"Missing input file: {path}")

  out_dir = Path(output_dir).expanduser().resolve()
  paths = output_paths(out_dir, with_onnx=onnx_file is not None)
  check_fresh_outputs(paths)

  env_yaml = checkpoint_path.parent / "params" / "env.yaml"
  saved: dict = _load_saved_env_yaml(env_yaml) if env_yaml.exists() else {}
  env_cfg = load_env_cfg(task, play=False)
  agent_cfg = load_rl_cfg(task)
  saved_lookahead_s = load_saved_lookahead_s(checkpoint_path)
  if saved_lookahead_s is None:
    print("[WARN] No saved params/env.yaml; leaving lookahead_s at the task default.")

  step_dt = env_cfg.sim.mujoco.timestep * env_cfg.decimation
  requested_steps = max(1, int(round(duration_s / step_dt)))
  if max_steps is not None:
    requested_steps = min(requested_steps, max(1, int(max_steps)))
  horizon_s = requested_steps * step_dt + HORIZON_MARGIN_S

  saved_seed = _saved_or_none(saved, _SAVED_SCALAR_PATHS["seed"])
  if seed is None:
    seed = int(saved_seed) if isinstance(saved_seed, int) else 0

  overrides = apply_nominal_overrides(
    env_cfg,
    motion_file=str(motion_path),
    saved_lookahead_s=saved_lookahead_s,
    horizon_s=horizon_s,
    seed=seed,
  )

  with np.load(motion_path, allow_pickle=True) as motion:
    motion_frames = int(motion["joint_pos"].shape[0])
    motion_fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
  # Reject a wrapping request before the environment is constructed.
  check_motion_window(int(start_frame), requested_steps, motion_frames)
  if not math.isclose(motion_fps, 1.0 / step_dt, rel_tol=1e-3):
    print(
      f"[WARN] Motion fps {motion_fps:g} differs from the policy rate "
      f"{1.0 / step_dt:g} Hz; the trajectory advances one frame per policy step."
    )

  print(f"[INFO] Output directory: {out_dir}")
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
  runner.load(str(checkpoint_path), map_location=device)
  policy = runner.get_inference_policy(device=device)

  raw_env = wrapped.unwrapped
  command = raw_env.command_manager.get_term("motion")
  if not isinstance(command, MotionCommand):
    raise ValueError("Task does not expose a MotionCommand named 'motion'.")
  robot = raw_env.scene["robot"]
  action_term = raw_env.action_manager.get_term("joint_pos")
  if not isinstance(action_term, JointPositionAction):
    raise ValueError("Task does not expose a JointPositionAction named 'joint_pos'.")
  joint_control = build_joint_control(
    robot.joint_names, position_actuator_records(robot)
  )
  action_scale = broadcast_per_joint(action_term.scale, joint_control.num_joints)
  action_offset = broadcast_per_joint(action_term.offset, joint_control.num_joints)

  # Deterministic start: the wrapper already reset the env, so write the exact
  # reference frame and refresh derived state, sensors and observations in order.
  env_ids = torch.arange(raw_env.num_envs, device=device)
  notes: list[str] = []
  command.reset_to_frame(env_ids, int(start_frame))
  raw_env.sim.forward()
  command.update_relative_body_poses()
  raw_env.sim.sense()
  raw_env.observation_manager.compute(update_history=True)

  start_q_ref = _cpu(command.joint_pos[0])
  start_q_sim = _cpu(robot.data.joint_pos[0])
  start_mismatch = float((start_q_sim - start_q_ref).abs().max().item())
  if start_mismatch > 1e-5:
    clipped = [
      name
      for name, ref, sim in zip(
        robot.joint_names,
        start_q_ref.tolist(),
        start_q_sim.tolist(),
        strict=True,
      )
      if abs(ref - sim) > 1e-5
    ]
    notes.append(
      "Reference frame joint positions differ from the written sim state by up "
      f"to {start_mismatch:.3e} rad (command soft-limit clipping): {clipped}. "
      "The saved and loaded reference-shape conventions are assumed identical; "
      "a large mismatch would indicate a joint-order difference."
    )

  result = record_policy_commands(
    wrapped=wrapped,
    policy=policy,
    command=command,
    action_term=action_term,
    joint_control=joint_control,
    steps=requested_steps,
    motion_fps=motion_fps,
    start_frame=int(start_frame),
  )
  result.notes.extend(notes)

  summary = summarize(result, joint_control)
  observations = result.arrays["actor_observation"]
  obs_terms = raw_env.observation_manager.active_terms["actor"]
  obs_term_dims = [
    int(np.prod(dim)) for dim in raw_env.observation_manager.group_obs_term_dim["actor"]
  ]
  policy_obs_dim = int(getattr(policy, "obs_dim", observations.shape[1]))
  if policy_obs_dim != observations.shape[1]:
    raise RuntimeError(
      f"Actor expects {policy_obs_dim} observations but the actor group produces "
      f"{observations.shape[1]}."
    )

  metadata: dict[str, Any] = {
    "diagnostic": (
      "Native mjlab policy-command recording for a tracking checkpoint "
      "(nominal, noise-free, deterministic)"
    ),
    "not_a_training_sample": (
      "Observation corruption and all domain randomization are disabled and the "
      "reference state is written exactly; this is a nominal evaluation, not a "
      "sample from the training distribution."
    ),
    "task": task,
    "device": str(device),
    "seed": {"requested": seed, "effective": env_cfg.seed},
    "checkpoint": _file_record(checkpoint_path),
    "motion": {
      "path": str(motion_path),
      "sha256": sha256_file(motion_path),
      "bytes": motion_path.stat().st_size,
      "frames": motion_frames,
      "fps": motion_fps,
      "duration_s": motion_frames / motion_fps,
    },
    "saved_config": collect_saved_provenance(
      checkpoint_path, saved, env_cfg, joint_control, action_scale
    ),
    "overrides": [asdict(o) for o in overrides],
    "recording": {
      "start_frame": int(start_frame),
      "requested_steps": requested_steps,
      "duration_s": requested_steps * step_dt,
      "steps_run": result.steps_run,
      "end_reason": result.end_reason,
      "end_detail": result.end_detail,
      "achieved_ref_frame": result.achieved_ref_frame,
      "achieved_reference_s": result.achieved_reference_s,
      "step_dt": step_dt,
      "physics_dt": env_cfg.sim.mujoco.timestep,
      "decimation": env_cfg.decimation,
      "policy_hz": 1.0 / step_dt,
      "episode_length_s": env_cfg.episode_length_s,
      "auto_reset": env_cfg.auto_reset,
      "num_envs": raw_env.num_envs,
      "num_resets_during_recording": 0,
      "clip_actions": agent_cfg.clip_actions,
      "terminations_preserved": list(raw_env.termination_manager.active_terms),
      "notes": result.notes,
    },
    "actor": {
      "obs_dim": policy_obs_dim,
      "action_dim": joint_control.num_joints,
      "normalization": (
        "baked into the loaded actor (EmpiricalNormalization restored from the "
        "checkpoint by the standard runner load path)"
      ),
      "joint_names": list(joint_control.joint_names),
      "obs_terms": [
        {"name": name, "dim": dim}
        for name, dim in zip(obs_terms, obs_term_dims, strict=True)
      ],
      "action_target_names": list(action_term.target_names),
    },
    "gains": {
      "kp": joint_control.kp.tolist(),
      "kd": joint_control.kd.tolist(),
      "effort_limit": joint_control.effort_limit.tolist(),
      "delay_max_lag": joint_control.delay_max_lag.tolist(),
      "ctrl_index_entity_local": joint_control.ctrl_index.tolist(),
      "action_scale": action_scale.tolist(),
      "action_offset": action_offset.tolist(),
    },
    "consistency_checks": {
      "values": result.consistency,
      "note": (
        "Max abs errors over the whole recording. The processed/sim target "
        "errors compare the recorded action pipeline against the action term's "
        "own values. torque_law_post_max_abs_error compares "
        "clamp(kp*(sim_target - q_post) - kd*dq_post, +/-effort_limit) with the "
        "simulator's qfrc_actuator: a value well above float32 noise means the "
        "logged gains, the joint mapping, or the actuator model (extra bias or "
        "gravity-compensation terms) differs from the plain MuJoCo <position> "
        "law assumed here."
      ),
    },
    "units": {
      "raw_action": "unitless policy output",
      "clipped_action": "raw_action after the wrapper's clip_actions",
      "processed_target": ("radians (clipped_action * action_scale + action_offset)"),
      "sim_target": "radians (processed_target - encoder_bias)",
      "ctrl": "radians position target (MuJoCo <position> actuator)",
      "qfrc_actuator": "N*m joint-space actuator torque",
      "actuator_force": "N*m scalar actuator output (actuator space)",
      "pd_torque_requested_pre": (
        "N*m reconstruction kp*(sim_target - q_pre) - kd*dq_pre, before limits"
      ),
      "pd_torque_applied_pre": "N*m requested torque clamped to effort_limit",
    },
    "npz_layout": {
      "per_joint_arrays": "shape (T, num_joints); columns follow 'joint_names'",
      "actor_observation": "shape (T, obs_dim); see actor.obs_terms for the layout",
      "root_link_pose_w": "shape (T, 7): position (3) + quaternion (4)",
      "root_link_vel_w": "shape (T, 6): linear (3) + angular (3)",
      "qfrc_actuator_post": (
        "read after env.step, i.e. forward-recomputed at the post-step state; "
        "not provably the torque integrated during the last physics substep"
      ),
      "sim_time_pre/post": "MuJoCo simulation clock before/after the step",
      "episode_index": "always 0: one episode with auto_reset=False",
    },
    "summary": summary,
  }

  written = write_outputs(
    paths,
    result=result,
    joint_control=joint_control,
    metadata=metadata,
    scalars={
      "motion_fps": motion_fps,
      "step_dt": step_dt,
      "physics_dt": env_cfg.sim.mujoco.timestep,
      "decimation": float(env_cfg.decimation),
      "policy_hz": 1.0 / step_dt,
      "start_frame": float(start_frame),
    },
  )

  parity: dict[str, Any] | None = None
  if onnx_file is not None:
    parity = run_onnx_parity(
      onnx_file=Path(onnx_file).expanduser().resolve(),
      result=result,
      policy=policy,
      device=device,
      max_samples=onnx_max_samples,
    )
    with written[ONNX_PARITY_NAME].open("w") as handle:
      json.dump(parity, handle, indent=2, default=str)
      handle.write("\n")

  print("=" * 72)
  print(f"checkpoint      : {checkpoint_path}")
  print(f"steps recorded  : {result.steps_run} of {requested_steps} requested")
  print(f"reference frames: {start_frame}..{result.achieved_ref_frame}")
  print(f"reference time  : {result.achieved_reference_s:.3f}s")
  print(f"end reason      : {result.end_reason} ({result.end_detail})")
  print(f"policy rate     : {1.0 / step_dt:g} Hz (dt={step_dt:g}s)")
  print(
    "consistency     : "
    + ", ".join(f"{k} {v:.2e}" for k, v in result.consistency.items())
  )
  for name in (n for n in joint_control.joint_names if "knee" in n):
    stats = summary[name]
    print(
      f"  {name:20s}: target-ref mean "
      f"{stats['target_minus_ref_deg_mean']:+.1f} deg "
      f"[{stats['target_minus_ref_deg_min']:+.1f}, "
      f"{stats['target_minus_ref_deg_max']:+.1f}]"
      f"  ref {stats['ref_q_min_deg']:.1f}..{stats['ref_q_max_deg']:.1f} deg"
      f"  max |torque| use {stats['torque_utilization_max'] * 100:.1f}%"
    )
  ranked = sorted(
    joint_control.joint_names,
    key=lambda n: summary[n]["torque_utilization_max"],
    reverse=True,
  )[:3]
  print(
    "highest |torque|/effort_limit: "
    + ", ".join(
      f"{n} {summary[n]['torque_utilization_max'] * 100:.1f}%" for n in ranked
    )
  )
  if result.notes:
    print("notes:")
    for note in result.notes:
      print(f"  - {note}")
  for name, path in written.items():
    print(f"wrote {name}: {path}")
  if parity is not None:
    print(
      f"onnx parity     : max|diff| {parity['max_abs_error']:.3e} over "
      f"{parity['samples_tested']} samples "
      f"(tolerance {parity['tolerance']:g}, within={parity['within_tolerance']}, "
      f"torch recompute max|diff| {parity['torch_recompute_max_abs_error']:.3e})"
    )
  print("=" * 72)
  env.close()


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
