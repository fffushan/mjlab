"""Parallel (multi-row) X2 simulator boundary for bounded D3 diffusion runs.

This module mirrors :mod:`mjlab.tasks.tracking.diffusion.closed_loop_runtime_x2`
on many environment rows.  The single-row module cannot serve a parallel sweep:
:class:`X2ClosedLoopRuntime` rejects any environment whose ``num_envs`` is not
one, and its observation, world state and step helpers are written for one row.

The parallel boundary keeps the same semantics and the same numbers:

* the same private motion command class, so the reference is frozen and the
  reference guards are disabled at handover exactly as in the single-row path;
* the same 164-D cached actor observation, sliced by cohort term, so the 99-D
  gravity conditioning is built from the same channels;
* the same 68-D VAE reference during the bootstrap only;
* the same measured pre-action pairing of a state with the latent decoded on it.

Every row is pinned to one cohort member and one start frame, so the reference
frame clock is common to the whole batch.  Rows are independent only through
the environment's own per-row domain randomization and through the noise the
batched sampler draws per row; this module does not add any initial-state
perturbation.

:class:`BatchedReferenceFreeContinuation` calls the frozen plan builder once per
row and then one batched sampler call, so the per-row arithmetic is the frozen
arithmetic and the expensive denoiser call is shared.  It never encodes and
never substitutes a teacher or standing-policy latent.

This module makes no closed-loop quality claim.  It reports measurements only.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mjlab.tasks.tracking.diffusion.closed_loop_runtime_x2 import (
  ClosedLoopMotionCommandCfg,
  _never_terminate,
  _term_slices,
)
from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT
from mjlab.tasks.tracking.diffusion.handover import (
  HandoverHistory,
  build_reference_free_plan,
)
from mjlab.tasks.tracking.diffusion.sampler import (
  SamplerConditions,
  sample_trajectory,
)
from mjlab.tasks.tracking.diffusion.state import WorldState
from mjlab.tasks.tracking.distillation.adapter import validate_live_contract
from mjlab.tasks.tracking.distillation.config import load_manifest, resolve_cohort
from mjlab.tasks.tracking.distillation.environment import (
  RuntimeSeedProvenance,
  _apply_saved_observation_timing,
)

_REFERENCE_TERMINATIONS: tuple[str, ...] = ("anchor_pos", "anchor_ori", "ee_body_pos")
_SIGNATURE_FIELDS: tuple[str, ...] = (
  "root_position",
  "root_quaternion_wxyz",
  "joint_position",
  "joint_velocity",
)
_ACTION_WIDTH = 31
_LATENT_WIDTH = 32
_CONDITIONING_WIDTH = 99
_REFERENCE_WIDTH = 68
_PERIOD_SECONDS = 0.02


class ParallelRuntimeX2Error(RuntimeError):
  """The parallel X2 boundary cannot be built, driven or read safely."""


def tilt_degrees_batch(state: WorldState) -> np.ndarray:
  """Return the per-row root-up tilt in degrees, independent of yaw.

  This is the vectorized form of ``visual_playback.tilt_degrees`` and uses the
  same formula, so a fall decision agrees with the single-row diagnostic.
  """
  quaternion = np.asarray(state.root_quaternion_wxyz, dtype=np.float64)
  if quaternion.ndim == 1 and quaternion.shape == (4,):
    quaternion = quaternion[None]
  if quaternion.ndim != 2 or quaternion.shape[1] != 4:
    raise ParallelRuntimeX2Error("batched root quaternion must have shape [rows, 4]")
  norms = np.linalg.norm(quaternion, axis=1)
  result = np.full(norms.shape, np.inf, dtype=np.float64)
  valid = np.isfinite(norms) & (norms >= 1.0e-12)
  if bool(valid.any()):
    normalized = quaternion[valid] / norms[valid][:, None]
    x = normalized[:, 1]
    y = normalized[:, 2]
    up_z = 1.0 - 2.0 * (x * x + y * y)
    result[valid] = np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0)))
  return result


def _matrix(value: Any, *, name: str, width: int, rows: int) -> np.ndarray:
  """Convert one simulator tensor to an owned finite ``[rows, width]`` array."""
  if isinstance(value, torch.Tensor):
    value = value.detach().cpu().numpy()
  result = np.asarray(value, dtype=np.float64)
  if result.ndim == 1 and result.shape == (width,) and rows == 1:
    result = result[None]
  if result.shape != (rows, width):
    raise ParallelRuntimeX2Error(
      f"{name} must have shape [{rows}, {width}], got {result.shape}"
    )
  if not np.isfinite(result).all():
    raise ParallelRuntimeX2Error(f"{name} contains non-finite simulator data")
  return np.array(result, dtype=np.float64, copy=True)


def _body_matrix(
  value: Any, indexes: np.ndarray, *, name: str, rows: int
) -> np.ndarray:
  """Select contract bodies for every row, preserving the simulator row order."""
  if isinstance(value, torch.Tensor):
    value = value.detach().cpu().numpy()
  raw = np.asarray(value, dtype=np.float64)
  if raw.ndim != 3 or raw.shape[0] != rows:
    raise ParallelRuntimeX2Error(f"{name} must have shape [{rows}, bodies, 3]")
  result = np.array(raw[:, indexes, :], dtype=np.float64, copy=True)
  if result.shape != (rows, len(indexes), 3):
    raise ParallelRuntimeX2Error(
      f"{name} must select {len(indexes)} bodies per row, got {result.shape}"
    )
  if not np.isfinite(result).all():
    raise ParallelRuntimeX2Error(f"{name} contains non-finite simulator data")
  return result


def _flags(value: Any, *, name: str, rows: int) -> np.ndarray:
  """Read one boolean flag per row from a simulator result field."""
  if isinstance(value, torch.Tensor):
    value = value.detach().cpu().numpy()
  array = np.asarray(value)
  if array.shape == ():
    array = array.reshape(1)
  if array.size != rows:
    raise ParallelRuntimeX2Error(
      f"{name} must contain one flag per row ({rows}), got shape {array.shape}"
    )
  return array.reshape(rows).astype(bool)


def _unbatched(state: WorldState, index: int) -> WorldState:
  """Return one row of a batched state as a one-dimensional measurement.

  ``WorldState.select`` preserves the batch axis, but the frozen handover seam
  requires a genuinely unbatched state, so the row is rebuilt field by field.
  """
  if isinstance(index, bool) or not isinstance(index, int):
    raise ParallelRuntimeX2Error("state row index must be an integer")
  if not 0 <= index < state.count:
    raise ParallelRuntimeX2Error(f"state row {index} is outside the batch")
  return WorldState(
    root_position=np.array(state.root_position, dtype=np.float64, copy=True)[index],
    root_quaternion_wxyz=np.array(
      state.root_quaternion_wxyz, dtype=np.float64, copy=True
    )[index],
    root_linear_velocity=np.array(
      state.root_linear_velocity, dtype=np.float64, copy=True
    )[index],
    root_angular_velocity=np.array(
      state.root_angular_velocity, dtype=np.float64, copy=True
    )[index],
    body_positions=np.array(state.body_positions, dtype=np.float64, copy=True)[index],
    body_linear_velocities=np.array(
      state.body_linear_velocities, dtype=np.float64, copy=True
    )[index],
  )


def _any_flag(value: Any, *, name: str) -> bool:
  """Report whether any entry of one boundary flag is true."""
  if isinstance(value, torch.Tensor):
    value = value.detach().cpu().numpy()
  array = np.asarray(value)
  if array.size == 0:
    raise ParallelRuntimeX2Error(f"{name} is empty")
  return bool(array.astype(bool).any())


def _scalar_int(value: Any, *, name: str) -> int:
  """Read one shared integer from a simulator field."""
  if isinstance(value, torch.Tensor):
    value = value.detach().cpu().numpy()
  array = np.asarray(value)
  if array.size == 0:
    raise ParallelRuntimeX2Error(f"{name} is empty")
  numbers = array.reshape(-1)
  if not np.all(numbers == numbers[0]):
    raise ParallelRuntimeX2Error(
      f"{name} differs between rows; the batch must share one reference clock"
    )
  number = numbers[0]
  if not np.isfinite(number) or int(number) != number:
    raise ParallelRuntimeX2Error(f"{name} must be a finite integer")
  return int(number)


@dataclass(frozen=True, slots=True)
class ParallelObservation:
  """One owned measurement per environment row."""

  state: WorldState
  conditioning: np.ndarray
  tick: int
  sim_time_s: float
  segment_id: str
  reference: np.ndarray | None = None
  reference_frame: np.ndarray | None = None

  def __post_init__(self) -> None:
    rows = self.state.count
    for name, width in (
      ("conditioning", _CONDITIONING_WIDTH),
      ("reference", _REFERENCE_WIDTH),
    ):
      value = getattr(self, name)
      if value is None and name == "reference":
        continue
      array = np.array(value, dtype=np.float64, copy=True)
      if array.shape != (rows, width) or not np.isfinite(array).all():
        raise ParallelRuntimeX2Error(f"{name} must be finite with shape [{rows}, {width}]")
      object.__setattr__(self, name, array)
    if self.reference_frame is not None:
      frames = np.array(self.reference_frame, dtype=np.int64, copy=True).reshape(-1)
      if frames.shape != (rows,) or np.any(frames < 0):
        raise ParallelRuntimeX2Error("reference_frame must be non-negative per row")
      object.__setattr__(self, "reference_frame", frames)

  @property
  def rows(self) -> int:
    return self.state.count

  @property
  def tilt_degrees(self) -> np.ndarray:
    return tilt_degrees_batch(self.state)


@dataclass(frozen=True, slots=True)
class ParallelTransition:
  """Per-row post-step evidence, captured before any possible reset."""

  observation: ParallelObservation
  terminal_available: bool = True
  terminated: np.ndarray | None = None
  truncated: np.ndarray | None = None
  reset: np.ndarray | None = None
  teleport: np.ndarray | None = None
  reference_boundary: np.ndarray | None = None
  reason: str | None = None

  def __post_init__(self) -> None:
    rows = self.observation.rows
    for name in ("terminated", "truncated", "reset", "teleport", "reference_boundary"):
      value = getattr(self, name)
      if value is None:
        continue
      flags = _flags(value, name=name, rows=rows)
      object.__setattr__(self, name, flags)


class X2ParallelClosedLoopRuntime:
  """Multi-row X2 simulator boundary with the D3 single-row semantics."""

  period_seconds: float
  clip_frames: int
  joint_order: tuple[str, ...]
  body_order: tuple[str, ...]
  provenance: Mapping[str, Any]

  def __init__(
    self,
    env: Any,
    cohort: Any,
    member: str,
    *,
    audit: Any = None,
    manifest_path: Path | None = None,
    seed: int | None = None,
  ) -> None:
    self.env = env
    self.cohort = cohort
    self.member = member
    self.audit = audit
    self._closed = False
    self._has_reset = False
    self._phase = "vae"
    self._tick = 0
    self._sim_time = 0.0
    self._last_pre_action: np.ndarray | None = None
    self._initial_signature_value: np.ndarray | None = None
    self._segment_id = f"{member}:parallel:uninitialized"

    self.num_envs = int(getattr(env, "num_envs", 0))
    if self.num_envs < 1:
      raise ParallelRuntimeX2Error("a parallel runtime needs at least one row")
    try:
      self.period_seconds = float(env.step_dt)
    except (AttributeError, TypeError, ValueError) as exc:
      raise ParallelRuntimeX2Error(
        "environment does not expose a numeric step_dt"
      ) from exc
    if abs(self.period_seconds - _PERIOD_SECONDS) > 1e-9:
      raise ParallelRuntimeX2Error(
        "unsupported timing: the parallel D3 boundary requires a 0.02 s "
        f"simulator step, got {self.period_seconds!r}"
      )

    self._command = self._get_motion_command()
    self.joint_order, self.body_order = self._validate_live_order()
    try:
      self.clip_frames = int(self._command.motion.time_step_total)
    except (AttributeError, TypeError, ValueError) as exc:
      raise ParallelRuntimeX2Error(
        "motion command does not expose the clip frame count"
      ) from exc
    if self.clip_frames < 2:
      raise ParallelRuntimeX2Error(
        f"motion clip must contain at least two frames, got {self.clip_frames}"
      )
    self._slices = _term_slices(cohort)
    self._row_ids = torch.arange(self.num_envs, dtype=torch.int64, device=env.device)
    self._reference_terms = self._save_reference_terms()
    if hasattr(getattr(env, "cfg", None), "auto_reset"):
      env.cfg.auto_reset = False
    self._set_reference_terminations(True)

    self._provenance: dict[str, Any] = {
      "runtime": "mjlab_x2_parallel_closed_loop",
      "physical_runtime": "synthetic",
      "synthetic": True,
      "measurement_source": "injected_test_environment_robot_data",
      "measurement_status": (
        "synthetic/test measurements from an injected environment; not physical "
        "evidence"
      ),
      "member": member,
      "device": str(getattr(env, "device", "unknown")),
      "rows": self.num_envs,
      "control_period_s": self.period_seconds,
      "clip_frames": self.clip_frames,
      "row_semantics": (
        "every row is pinned to one member and one start frame; rows differ only "
        "through the environment's own per-row domain randomization and through "
        "the per-row sampler noise"
      ),
      "action_boundary": (
        f"direct env.step(action_tensor), shape [{self.num_envs}, 31]"
      ),
      "auto_reset": {"requested": False, "applied": True},
      "cached_observation_capture": {
        "source": "env.obs_buf / observation_manager cache",
        "update_history": False,
        "noise_or_delay_update": False,
        "projected_gravity": "robot.data.projected_gravity_b (root frame)",
        "conditioning_width": _CONDITIONING_WIDTH,
        "reference_width": _REFERENCE_WIDTH,
      },
      "reference_read_boundaries": {
        "reset": "allowed for deterministic frame initialization",
        "vae_bootstrap": "allowed for reference-tracking conditioning",
        "enter_reference_free": "no reference read and no physical-state advance",
        "continuation": "reference and reference_frame are absent; no command read",
      },
      "overrides": [
        "private motion command freezes timer resampling and clip wrap after handover",
        "private reference terminations are replaced with always-false functions",
        "private environment auto_reset=False",
      ],
      "initial_state_signature": {
        "fields": list(_SIGNATURE_FIELDS),
        "width": 3 + 4 + len(self.joint_order) + len(self.joint_order),
        "dtype": "float64",
        "owned_copy": True,
        "source": "robot.data root pose, joint position and joint velocity",
      },
    }
    self.provenance = self._provenance
    if manifest_path is not None:
      self._provenance["manifest"] = str(manifest_path)
      self._provenance["manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
      ).hexdigest()
    if seed is not None:
      self._provenance["construction_seed"] = int(seed)
    if audit is not None:
      self._provenance["live_contract_audited"] = True

  def _mark_factory_physical_runtime(self) -> None:
    """Mark only the verified factory path as real mjlab simulator evidence."""
    self._provenance.update(
      {
        "physical_runtime": "mjlab",
        "synthetic": False,
        "measurement_source": "real_mjlab_simulator_robot_data",
        "measurement_status": "real measurements from ManagerBasedRlEnv",
        "factory_trusted": True,
      }
    )

  # Construction helpers (mirrors of the single-row checks).

  def _get_motion_command(self) -> Any:
    try:
      command = self.env.command_manager.get_term("motion")
    except (AttributeError, KeyError, TypeError) as exc:
      raise ParallelRuntimeX2Error(
        "environment has no motion command term"
      ) from exc
    if not hasattr(command, "time_steps"):
      raise ParallelRuntimeX2Error(
        "motion command lacks an integer reference frame clock"
      )
    return command

  def _validate_live_order(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
    expected_joints = tuple(self.cohort.actions.joint_names)
    if len(expected_joints) != _ACTION_WIDTH:
      raise ParallelRuntimeX2Error(
        "unsupported joint schema: D3 gravity VAE requires 31 action joints"
      )
    robot = self.env.scene["robot"]
    actual_joints = tuple(getattr(robot, "joint_names", ()))
    if actual_joints != expected_joints:
      raise ParallelRuntimeX2Error(
        "unsupported joint order: live robot order differs from the cohort "
        f"({actual_joints!r} != {expected_joints!r})"
      )
    expected_bodies = tuple(DEFAULT_CONTRACT.body_names)
    declared_bodies = tuple(getattr(self.cohort, "body_names", ()))
    if declared_bodies != expected_bodies:
      raise ParallelRuntimeX2Error(
        "unsupported body order: cohort body order is not the frozen X2 order"
      )
    actual_bodies = tuple(getattr(robot, "body_names", ()))
    missing = [name for name in expected_bodies if name not in actual_bodies]
    if missing:
      raise ParallelRuntimeX2Error(
        f"unsupported body order: live robot is missing {missing}"
      )
    command_bodies = tuple(getattr(getattr(self._command, "cfg", None), "body_names", ()))
    if command_bodies and command_bodies != expected_bodies:
      raise ParallelRuntimeX2Error(
        "unsupported body order: motion command body order differs from the "
        "frozen diffusion state order"
      )
    observations = self.cohort.observations
    widths = tuple(observations.widths)
    if int(getattr(observations, "total_dim", sum(widths))) != 164:
      raise ParallelRuntimeX2Error(
        "unsupported schema: expected 164-D cached actor observations"
      )
    if int(getattr(self.cohort.actor, "obs_dim", 164)) != 164:
      raise ParallelRuntimeX2Error("unsupported schema: expected actor obs_dim 164")
    if abs(float(self.cohort.control.control_period_s) - _PERIOD_SECONDS) > 1e-9:
      raise ParallelRuntimeX2Error(
        "unsupported timing: cohort control period must be 0.02 s"
      )
    if int(self.cohort.actions.dim) != _ACTION_WIDTH:
      raise ParallelRuntimeX2Error("unsupported schema: action dimension must be 31")
    return expected_joints, expected_bodies

  def _save_reference_terms(self) -> dict[str, tuple[Any, Any, dict[str, Any]]]:
    manager = getattr(self.env, "termination_manager", None)
    if manager is None:
      raise ParallelRuntimeX2Error("environment has no termination manager")
    active = tuple(getattr(manager, "active_terms", ()))
    configs = getattr(manager, "_term_cfgs", ())
    saved: dict[str, tuple[Any, Any, dict[str, Any]]] = {}
    for name in _REFERENCE_TERMINATIONS:
      if name not in active:
        raise ParallelRuntimeX2Error(f"tracking task is missing reference guard {name!r}")
      index = active.index(name)
      try:
        term_cfg = configs[index]
        if bool(term_cfg.time_out):
          raise ParallelRuntimeX2Error(
            f"reference guard {name!r} cannot be a time-out term"
          )
        saved[name] = (term_cfg, term_cfg.func, dict(term_cfg.params))
      except (AttributeError, IndexError, TypeError) as exc:
        raise ParallelRuntimeX2Error(
          f"malformed termination config for {name!r}"
        ) from exc
    return saved

  def _set_reference_terminations(self, enabled: bool) -> None:
    manager = self.env.termination_manager
    active = tuple(manager.active_terms)
    for name, (term_cfg, func, params) in self._reference_terms.items():
      if name not in active:
        raise ParallelRuntimeX2Error(
          f"reference guard {name!r} disappeared from the manager"
        )
      if enabled:
        term_cfg.func = func
        term_cfg.params = dict(params)
      else:
        term_cfg.func = _never_terminate
        term_cfg.params = {}

  # Measurements.

  def _world_state(self) -> WorldState:
    data = self.env.scene["robot"].data
    rows = self.num_envs
    indexes = np.asarray(self._body_indexes, dtype=np.int64)
    return WorldState(
      root_position=_matrix(
        data.root_link_pos_w, name="root_link_pos_w", width=3, rows=rows
      ),
      root_quaternion_wxyz=_matrix(
        data.root_link_quat_w, name="root_link_quat_w", width=4, rows=rows
      ),
      root_linear_velocity=_matrix(
        data.root_link_lin_vel_w, name="root_link_lin_vel_w", width=3, rows=rows
      ),
      root_angular_velocity=_matrix(
        data.root_link_ang_vel_w, name="root_link_ang_vel_w", width=3, rows=rows
      ),
      body_positions=_body_matrix(
        data.body_link_pos_w, indexes, name="body_link_pos_w", rows=rows
      ),
      body_linear_velocities=_body_matrix(
        data.body_link_lin_vel_w, indexes, name="body_link_lin_vel_w", rows=rows
      ),
    )

  @property
  def _body_indexes(self) -> tuple[int, ...]:
    robot_names = tuple(self.env.scene["robot"].body_names)
    return tuple(robot_names.index(name) for name in self.body_order)

  def _signature_now(self) -> np.ndarray:
    data = self.env.scene["robot"].data
    rows = self.num_envs
    joints = len(self.joint_order)
    return np.concatenate(
      (
        _matrix(data.root_link_pos_w, name="root_link_pos_w", width=3, rows=rows),
        _matrix(
          data.root_link_quat_w, name="root_link_quat_w", width=4, rows=rows
        ),
        _matrix(data.joint_pos, name="joint_pos", width=joints, rows=rows),
        _matrix(data.joint_vel, name="joint_vel", width=joints, rows=rows),
      ),
      axis=1,
    )

  def _cached_actor(self) -> np.ndarray:
    """Read the existing post-reset/post-step cache without recomputation."""
    cached = getattr(self.env, "obs_buf", None)
    rows = self.num_envs
    if not isinstance(cached, Mapping) or "actor" not in cached:
      manager = getattr(self.env, "observation_manager", None)
      cached = getattr(manager, "_obs_buffer", None)
    if not isinstance(cached, Mapping) or "actor" not in cached:
      raise ParallelRuntimeX2Error("no cached actor observation is available")
    actor = cached.get("actor") if isinstance(cached, Mapping) else None
    if isinstance(actor, torch.Tensor):
      actor = actor.detach().cpu().numpy()
    array = np.asarray(actor, dtype=np.float64)
    if array.shape == (164,) and rows == 1:
      array = array[None]
    if array.shape != (rows, 164):
      raise ParallelRuntimeX2Error(
        f"cached actor observation must have shape [{rows}, 164], got {array.shape}"
      )
    if not np.isfinite(array).all():
      raise ParallelRuntimeX2Error(
        "cached actor observation contains non-finite values"
      )
    return np.array(array, dtype=np.float64, copy=True)

  def _frame(self) -> int:
    return _scalar_int(self._command.time_steps, name="reference frame")

  def _capture(self) -> ParallelObservation:
    actor = self._cached_actor()
    values = {name: actor[:, span] for name, span in self._slices.items()}
    data = self.env.scene["robot"].data
    gravity = _matrix(
      data.projected_gravity_b,
      name="projected_gravity_b",
      width=3,
      rows=self.num_envs,
    )
    conditioning = np.concatenate(
      (
        gravity,
        values["base_ang_vel"],
        values["joint_pos"],
        values["joint_vel"],
        values["actions"],
      ),
      axis=1,
    )
    if conditioning.shape != (self.num_envs, _CONDITIONING_WIDTH):
      raise ParallelRuntimeX2Error(
        f"gravity conditioning must have shape [{self.num_envs}, 99], "
        f"got {conditioning.shape}"
      )
    reference: np.ndarray | None = None
    reference_frame: np.ndarray | None = None
    if self._phase == "vae":
      reference = np.concatenate(
        (values["command"][:, :62], values["motion_anchor_ori_b"]), axis=1
      )
      if reference.shape != (self.num_envs, _REFERENCE_WIDTH):
        raise ParallelRuntimeX2Error(
          f"reference must have shape [{self.num_envs}, 68], got {reference.shape}"
        )
      reference_frame = np.full(self.num_envs, self._frame(), dtype=np.int64)
    return ParallelObservation(
      state=self._world_state(),
      conditioning=conditioning,
      tick=self._tick,
      sim_time_s=self._sim_time,
      segment_id=self._segment_id,
      reference=reference,
      reference_frame=reference_frame,
    )

  def _event_flags(self, pre_generation: Any) -> tuple[bool, bool, bool, str | None]:
    """Consume local command boundary evidence, if the command exposes it."""
    consume = getattr(self._command, "consume_boundary_events", None)
    if consume is None:
      return False, False, False, None
    event = consume(pre_generation)
    if event is None:
      return False, False, False, None
    available = _any_flag(event.available, name="boundary availability")
    completed = _any_flag(event.completed, name="completed boundary")
    interrupted = _any_flag(event.interrupted, name="interrupted boundary")
    reasons = getattr(event, "reasons", ())
    reason = reasons[0] if reasons and reasons[0] != "unavailable" else None
    return available, completed, interrupted, reason

  # Lifecycle.

  def reset(self, *, seed: int, start_frame: int) -> ParallelObservation:
    """Reset every row and place all of them on the same explicit start frame."""
    if isinstance(seed, bool) or not isinstance(seed, int):
      raise ParallelRuntimeX2Error("seed must be an integer")
    if isinstance(start_frame, bool) or not isinstance(start_frame, int):
      raise ParallelRuntimeX2Error("start_frame must be an integer")
    if start_frame < 0 or start_frame >= self.clip_frames:
      raise ParallelRuntimeX2Error(
        f"start frame {start_frame} is outside clip length {self.clip_frames}"
      )
    self._phase = "vae"
    restore = getattr(self._command, "restore_reference", None)
    if restore is not None:
      restore()
    self._set_reference_terminations(True)
    result = self.env.reset(seed=seed)
    if isinstance(result, tuple) and result and isinstance(result[0], Mapping):
      self.env.obs_buf = result[0]
    self._command.reset_to_frame(self._row_ids, int(start_frame))
    scene = getattr(self.env, "scene", None)
    if scene is not None and hasattr(scene, "write_data_to_sim"):
      scene.write_data_to_sim()
    sim = getattr(self.env, "sim", None)
    if sim is not None and hasattr(sim, "forward"):
      sim.forward()
    if sim is not None and hasattr(sim, "sense"):
      sim.sense()
    update_relative = getattr(self._command, "update_relative_body_poses", None)
    if update_relative is not None:
      update_relative()
    manager = getattr(self.env, "observation_manager", None)
    if manager is not None and hasattr(manager, "compute"):
      cached = manager.compute(update_history=True, env_ids=self._row_ids)
      if cached is not None:
        self.env.obs_buf = cached
    consume = getattr(self._command, "consume_boundary_events", None)
    if consume is not None:
      consume()
    self._tick = 0
    self._sim_time = 0.0
    self._last_pre_action = None
    self._segment_id = f"{self.member}:seed={seed}:frame={start_frame}"
    self._has_reset = True
    self._initial_signature_value = self._signature_now()
    self._provenance["initial_state_signature_digest"] = hashlib.sha256(
      self._initial_signature_value.tobytes()
    ).hexdigest()
    self._provenance["initial_state_signature_seed"] = int(seed)
    self._provenance["initial_state_signature_start_frame"] = int(start_frame)
    self._provenance["mode"] = "vae"
    return self._capture()

  @property
  def observation(self) -> ParallelObservation:
    if not self._has_reset:
      raise ParallelRuntimeX2Error("observation requested before reset")
    return self._capture()

  def initial_signature(self) -> np.ndarray:
    if self._initial_signature_value is None:
      raise ParallelRuntimeX2Error("initial_signature requested before reset")
    return self._initial_signature_value.copy()

  def enter_reference_free(self) -> ParallelObservation:
    """Freeze the reference and its guards for every row without stepping."""
    if not self._has_reset:
      raise ParallelRuntimeX2Error("enter_reference_free requires reset first")
    before = self._signature_now()
    freeze = getattr(self._command, "freeze_reference", None)
    if freeze is None:
      raise ParallelRuntimeX2Error(
        "private motion command cannot freeze reference updates"
      )
    freeze()
    self._set_reference_terminations(False)
    after = self._signature_now()
    if not np.array_equal(before, after):
      raise ParallelRuntimeX2Error(
        "enter_reference_free changed physical state; refusing a discontinuous "
        "handover"
      )
    self._phase = "reference_free"
    self._provenance["mode"] = "reference_free"
    self._provenance["handover"] = {
      "advanced_simulation": False,
      "reset_physical_state": False,
      "reference_command_frozen": True,
      "reference_guards_disabled": True,
      "rows": self.num_envs,
    }
    return self._capture()

  def step(self, actions: np.ndarray) -> ParallelTransition:
    """Apply one action per row and capture post-step evidence before any reset."""
    if not self._has_reset:
      raise ParallelRuntimeX2Error("step requires reset first")
    values = np.asarray(actions, dtype=np.float64)
    if values.shape != (self.num_envs, _ACTION_WIDTH):
      raise ParallelRuntimeX2Error(
        f"actions must have shape [{self.num_envs}, 31], got {values.shape}"
      )
    if not np.isfinite(values).all():
      raise ParallelRuntimeX2Error("refusing to step a non-finite action")
    self._last_pre_action = np.array(values, dtype=np.float64, copy=True)
    pre_frame: int | None = None
    pre_generation = None
    if self._phase == "vae":
      pre_frame = self._frame()
      if pre_frame >= self.clip_frames - 1:
        self._provenance["last_frame_refusal"] = {
          "reference_frame": pre_frame,
          "physical_step_executed": False,
          "reason": "final_reference_frame_has_no_successor",
        }
        raise ParallelRuntimeX2Error(
          "refusing VAE action at the final reference frame; "
          "no physical step was executed"
        )
      if hasattr(self._command, "generation_ids"):
        generation = self._command.generation_ids
        pre_generation = (
          generation.detach().clone()
          if isinstance(generation, torch.Tensor)
          else generation
        )
    tensor = torch.as_tensor(
      np.ascontiguousarray(values, dtype=np.float32),
      dtype=torch.float32,
      device=getattr(self.env, "device", "cpu"),
    )
    try:
      result = self.env.step(tensor)
    except Exception as exc:
      raise ParallelRuntimeX2Error(f"simulator action step failed: {exc}") from exc
    if not isinstance(result, tuple) or len(result) < 5:
      raise ParallelRuntimeX2Error(
        "simulator step must return observations, reward, done flags and extras"
      )
    if isinstance(result[0], Mapping):
      self.env.obs_buf = result[0]
    terminated = _flags(result[2], name="terminated", rows=self.num_envs)
    truncated = _flags(result[3], name="time_outs", rows=self.num_envs)
    extras = result[4] if isinstance(result[4], Mapping) else {}
    self._tick += 1
    self._sim_time = self._tick * self.period_seconds
    observation = self._capture()

    reset = _flags(
      extras.get("reset", np.zeros(self.num_envs, dtype=bool)),
      name="reset",
      rows=self.num_envs,
    )
    teleport = np.zeros(self.num_envs, dtype=bool)
    reference_boundary = np.zeros(self.num_envs, dtype=bool)
    reason: str | None = None
    if self._phase == "vae":
      post_frame = self._frame()
      available, completed, interrupted, reason = self._event_flags(pre_generation)
      # The private command gates its own updates on any terminal flag in the
      # batch, so the frame-advance test is shared, exactly as in the
      # single-row path where the batch holds one row.
      terminal_no_advance = bool((terminated | truncated).any()) and (
        pre_frame is not None and post_frame == pre_frame
      )
      frame_discontinuity = (
        pre_frame is not None
        and post_frame != pre_frame + 1
        and not terminal_no_advance
      )
      boundary = bool(available or post_frame >= self.clip_frames - 1 or frame_discontinuity)
      if boundary:
        reference_boundary[:] = True
      if interrupted or completed or frame_discontinuity:
        teleport[:] = True
      if boundary and reason is None:
        reason = "reference_boundary"
    if bool(reset.any()) and reason is None:
      reason = "reset"
    if bool((terminated | truncated).any()) and reason is None:
      reason = "terminated" if bool(terminated.any()) else "time_out"
    return ParallelTransition(
      observation=observation,
      terminal_available=True,
      terminated=terminated,
      truncated=truncated,
      reset=reset,
      teleport=teleport,
      reference_boundary=reference_boundary,
      reason=reason,
    )

  def close(self) -> None:
    if not self._closed:
      self._closed = True
      self.env.close()


@dataclass(frozen=True, slots=True)
class BatchedContinuationStep:
  """One reference-free tick for a row subset."""

  tick: int
  rows: np.ndarray
  actions: np.ndarray
  executed_latents: np.ndarray
  denoiser_calls: int
  final_index: int


class BatchedReferenceFreeContinuation:
  """Reference-free DDIM continuation for many independent rows.

  One plan is built per row with the frozen :func:`build_reference_free_plan`,
  the plans are stacked into one ``[rows, 41, channels]`` conditioning batch,
  and a single :func:`sample_trajectory` call runs the frozen 20 DDIM jumps for
  every row at once.  Each row keeps its own :class:`HandoverHistory`, so the
  measured pre-action pairing is never shared or reordered.

  The generator is shared.  Because ``sample_trajectory`` draws its noise in one
  ``torch.randn`` call of the whole batch shape, row ``r`` consumes the
  ``r``-th slice of that draw: every row sees different noise, and a one-row run
  of this class at the same seed sees the same values as row zero of a wider
  batch.  That is what makes the row-zero parity check in the driver meaningful.
  """

  def __init__(
    self,
    *,
    denoiser: Any,
    decode: Any,
    schedule: Any,
    projection: Any,
    contract: Any = DEFAULT_CONTRACT,
    device: str | torch.device = "cpu",
    generator: torch.Generator,
    sampler_config: Any,
    rows: int,
  ) -> None:
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
      raise ParallelRuntimeX2Error("continuation rows must be a positive integer")
    self.contract = contract
    self.projection = projection
    self.schedule = schedule
    self.device = torch.device(device)
    self.rows = rows
    self._denoiser = denoiser
    self._decode = decode
    self._generator = generator
    self._sampler_config = sampler_config
    self._histories = [HandoverHistory(contract=contract) for _ in range(rows)]
    self._last_tick: int | None = None
    self.denoiser_calls = 0

  @property
  def past_steps(self) -> int:
    return int(self.contract.past_steps)

  def ready(self, rows: Iterable[int]) -> np.ndarray:
    """Report which of the given rows hold a complete history."""
    return np.array(
      [self._histories[int(row)].ready for row in rows], dtype=bool
    )

  def record_executed(
    self, rows: Iterable[int], states: WorldState, latents: np.ndarray
  ) -> None:
    """Record one measured pre-action pair per row.

    ``states`` and ``latents`` are aligned with ``rows``: row entry ``i`` of
    both belongs to ``rows[i]``.  Pass ``WorldState.select(rows)`` for a row
    subset, or the full-width state when ``rows`` is every row in order.
    """
    ids = np.asarray(tuple(int(row) for row in rows), dtype=np.int64).reshape(-1)
    values = np.asarray(latents, dtype=np.float64)
    if values.shape != (ids.size, _LATENT_WIDTH):
      raise ParallelRuntimeX2Error(
        f"latents must have shape [{ids.size}, 32], got {values.shape}"
      )
    if states.count != ids.size:
      raise ParallelRuntimeX2Error(
        f"state batch must hold {ids.size} rows to match the selected rows, "
        f"got {states.count}"
      )
    for position, row in enumerate(ids):
      if not 0 <= row < self.rows:
        raise ParallelRuntimeX2Error(f"row {row} is outside the continuation batch")
      self._histories[row].push(_unbatched(states, int(position)), values[position])

  def history_sizes(self) -> list[int]:
    return [len(history.pairs) for history in self._histories]

  def step(
    self,
    states: WorldState,
    conditioning: np.ndarray,
    *,
    tick: int,
    rows: Iterable[int],
  ) -> BatchedContinuationStep:
    """Plan from each measured state, sample once for the batch, decode the rows."""
    ids = np.asarray(tuple(int(row) for row in rows), dtype=np.int64).reshape(-1)
    if ids.size == 0:
      raise ParallelRuntimeX2Error("continuation step needs at least one row")
    if np.any(ids < 0) or np.any(ids >= self.rows):
      raise ParallelRuntimeX2Error("continuation rows are outside the batch")
    if np.unique(ids).size != ids.size:
      raise ParallelRuntimeX2Error("continuation rows must be unique")
    conditioning_values = np.asarray(conditioning, dtype=np.float64)
    if conditioning_values.shape != (self.rows, _CONDITIONING_WIDTH):
      raise ParallelRuntimeX2Error(
        "conditioning must have shape "
        f"[{self.rows}, {_CONDITIONING_WIDTH}], got {conditioning_values.shape}"
      )
    if states.count != self.rows:
      raise ParallelRuntimeX2Error(
        f"state batch must hold {self.rows} rows, got {states.count}"
      )
    if self._last_tick is not None and int(tick) <= self._last_tick:
      raise ParallelRuntimeX2Error(
        f"control tick {tick} is not newer than the last executed tick "
        f"{self._last_tick}; reset the continuation after any discontinuity"
      )
    unready = [int(row) for row in ids if not self._histories[int(row)].ready]
    if unready:
      raise ParallelRuntimeX2Error(
        f"rows {unready[:8]} have fewer than {self.past_steps} measured "
        "state/latent pairs"
      )

    plans = [
      build_reference_free_plan(
        self._histories[int(row)],
        _unbatched(states, int(row)),
        projection=self.projection,
        tick=int(tick),
        contract=self.contract,
        device=self.device,
      )
      for row in ids
    ]
    tokens = torch.stack([plan.clean_tokens for plan in plans], dim=0)
    mask = plans[0].clean_mask
    sample, diagnostics = sample_trajectory(
      self._denoiser,
      schedule=self.schedule,
      conditions=SamplerConditions(tokens, mask),
      config=self._sampler_config,
      generator=self._generator,
    )
    latent_slice = slice(
      self.contract.projected_state_dimension, self.contract.token_dimension
    )
    normalized = (
      sample[:, self.contract.current_index, latent_slice]
      .detach()
      .to(device="cpu", dtype=torch.float32)
      .numpy()
      .astype(np.float64)
    )
    executed = np.stack(
      [self.projection.denormalize_latent(normalized[position]) for position in range(len(ids))],
      axis=0,
    )
    if executed.shape != (len(ids), _LATENT_WIDTH) or not np.isfinite(executed).all():
      raise ParallelRuntimeX2Error(
        f"executed latents must be finite with shape [{len(ids)}, 32]"
      )
    latent_tensor = torch.from_numpy(
      np.ascontiguousarray(executed, dtype=np.float32)
    ).to(device=self.device)
    conditioning_tensor = torch.from_numpy(
      np.ascontiguousarray(conditioning_values[ids], dtype=np.float32)
    ).to(device=self.device)
    with torch.no_grad():
      decoded = self._decode(latent_tensor, conditioning_tensor)
    if not isinstance(decoded, torch.Tensor):
      raise ParallelRuntimeX2Error("decoder must return a torch tensor")
    actions = (
      decoded.detach().to(device="cpu", dtype=torch.float32).numpy().reshape(len(ids), -1)
    )
    if actions.shape != (len(ids), _ACTION_WIDTH) or not np.isfinite(actions).all():
      raise ParallelRuntimeX2Error(
        f"decoded actions must be finite with shape [{len(ids)}, 31]"
      )
    self._last_tick = int(tick)
    self.denoiser_calls += int(diagnostics.denoiser_calls)
    return BatchedContinuationStep(
      tick=int(tick),
      rows=ids,
      actions=np.array(actions, dtype=np.float64, copy=True),
      executed_latents=np.array(executed, dtype=np.float64, copy=True),
      denoiser_calls=int(diagnostics.denoiser_calls),
      final_index=int(diagnostics.final_index),
    )


def _validate_factory_inputs(
  manifest_path: str | Path, repo_root: str | Path, member: str, seed: int
) -> tuple[Path, Any]:
  if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
    raise ParallelRuntimeX2Error("seed must be an integer in [0, 2**32 - 1]")
  if not isinstance(member, str) or not member:
    raise ParallelRuntimeX2Error("member must be a non-empty teacher id")
  root = Path(repo_root).resolve()
  manifest = Path(manifest_path)
  if not manifest.is_absolute():
    manifest = root / manifest
  manifest = manifest.resolve()
  try:
    cohort = resolve_cohort(load_manifest(manifest, repo_root=root))
    cohort.teacher(member)
  except Exception as exc:
    raise ParallelRuntimeX2Error(
      f"parallel closed-loop cohort validation failed: {exc}"
    ) from exc
  if DEFAULT_CONTRACT.projection_rows != 0:
    raise ParallelRuntimeX2Error(
      "unsupported diffusion schema: the parallel boundary requires the "
      "no-mixture contract"
    )
  try:
    from mjlab.tasks.tracking.diffusion.closed_loop_runtime_x2 import (
      _validate_cohort_static,
    )

    _validate_cohort_static(cohort)
  except Exception as exc:
    raise ParallelRuntimeX2Error(
      f"parallel closed-loop cohort schema rejected: {exc}"
    ) from exc
  return manifest, cohort


def _build_parallel_private_environment(
  cohort: Any,
  member: str,
  *,
  repo_root: Path,
  device: str,
  seed: int,
  num_envs: int,
) -> Any:
  """Build one private registry-derived config with ``num_envs`` rows.

  This mirrors the single-row private builder exactly.  The only difference is
  the row count: every row receives the same member clip, and the driver then
  places every row on the same start frame with ``reset_to_frame``.
  """
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg
  from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg

  if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
    raise ParallelRuntimeX2Error("num_envs must be a positive integer")
  selected_task = cohort.manifest.base_task
  if selected_task is None:
    raise ParallelRuntimeX2Error("cohort has no registered base_task")
  cfg = copy.deepcopy(load_env_cfg(selected_task))
  cfg.seed = seed
  teacher = cohort.teacher(member)
  overrides = _apply_saved_observation_timing(
    cfg, teacher.env_config, str(teacher.entry.env_config)
  )
  cfg.scene.num_envs = int(num_envs)
  cfg.auto_reset = False
  motion_cfg = cfg.commands.get("motion")
  if not isinstance(motion_cfg, MotionCommandCfg):
    raise ParallelRuntimeX2Error(
      f"registered task {selected_task!r} has no motion command"
    )
  private_motion = ClosedLoopMotionCommandCfg.from_config(motion_cfg)
  private_motion.motion_file = str(teacher.entry.motion)
  cfg.commands["motion"] = private_motion
  env = ManagerBasedRlEnv(cfg, device=device, render_mode=None)
  env.cfg.__dict__["_closed_loop_semantic_overrides"] = tuple(overrides)
  env.cfg.__dict__["_closed_loop_seed_provenance"] = RuntimeSeedProvenance(
    requested_seed=seed,
    effective_seed=env.cfg.seed,
    applied_before_construction=True,
  )
  return env


def build_parallel_closed_loop_runtime(
  *,
  manifest_path: str | Path,
  repo_root: str | Path,
  member: str,
  device: str,
  seed: int,
  num_envs: int,
) -> X2ParallelClosedLoopRuntime:
  """Build one verified private X2 runtime with ``num_envs`` rows."""
  manifest, cohort = _validate_factory_inputs(manifest_path, repo_root, member, seed)
  env: Any | None = None
  try:
    env = _build_parallel_private_environment(
      cohort,
      member,
      repo_root=Path(repo_root).resolve(),
      device=device,
      seed=seed,
      num_envs=num_envs,
    )
    audit = validate_live_contract(
      env, cohort, member, task_id=cohort.manifest.base_task
    )
    runtime = X2ParallelClosedLoopRuntime(
      env,
      cohort,
      member,
      audit=audit,
      manifest_path=manifest,
      seed=seed,
    )
    runtime._mark_factory_physical_runtime()
    return runtime
  except ParallelRuntimeX2Error:
    if env is not None:
      env.close()
    raise
  except Exception as exc:
    if env is not None:
      env.close()
    raise ParallelRuntimeX2Error(
      f"parallel closed-loop runtime construction failed: {exc}"
    ) from exc


def cohort_members(manifest_path: str | Path, repo_root: str | Path) -> tuple[str, ...]:
  """Return the ordered cohort member ids for a manifest."""
  root = Path(repo_root).resolve()
  manifest = Path(manifest_path)
  if not manifest.is_absolute():
    manifest = root / manifest
  cohort = resolve_cohort(load_manifest(manifest.resolve(), repo_root=root))
  return tuple(str(member) for member in cohort.teacher_ids)


__all__ = [
  "BatchedContinuationStep",
  "BatchedReferenceFreeContinuation",
  "ParallelObservation",
  "ParallelRuntimeX2Error",
  "ParallelTransition",
  "X2ParallelClosedLoopRuntime",
  "build_parallel_closed_loop_runtime",
  "cohort_members",
  "tilt_degrees_batch",
]
