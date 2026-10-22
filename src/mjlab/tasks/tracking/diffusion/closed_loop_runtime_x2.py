"""Private X2 simulator boundary for the D3 closed-loop evaluation.

This module is deliberately separate from the distillation adapter.  The
latter is a reference-labeling adapter and its ``step``/``snapshot`` pair is
not a safe continuation boundary: it reads the motion reference and permits
the normal command resampling path.  D3 uses the local command below instead.

Only the factory constructs a real mjlab environment.  The runtime itself is
small enough to exercise with fake environments in CPU tests; no test in this
module constructs MuJoCo physics.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mjlab.tasks.tracking.diffusion.closed_loop_types import (
  ClosedLoopRuntime,
  D3Observation,
  D3Transition,
)
from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT
from mjlab.tasks.tracking.diffusion.state import WorldState
from mjlab.tasks.tracking.distillation.adapter import validate_live_contract
from mjlab.tasks.tracking.distillation.config import load_manifest, resolve_cohort
from mjlab.tasks.tracking.distillation.environment import (
  RuntimeSeedProvenance,
  SegmentMotionCommand,
  SegmentMotionCommandCfg,
  _apply_saved_observation_timing,
)
from mjlab.tasks.tracking.distillation.vae_config import (
  JOINT_DIM,
  PREVIEW_DIM,
  REFERENCE_DIM,
  REFERENCE_DIM_WITH_PREVIEW,
  REFERENCE_PREVIEW_FIELD,
)


class RuntimeX2Error(RuntimeError):
  """The private X2 closed-loop runtime cannot be built or driven safely."""


_REFERENCE_TERMINATIONS: tuple[str, ...] = ("anchor_pos", "anchor_ori", "ee_body_pos")
_CONDITIONING_TERMS: tuple[str, ...] = (
  "base_ang_vel",
  "joint_pos",
  "joint_vel",
  "actions",
)
_EXPECTED_ACTOR_TERMS: tuple[str, ...] = (
  "command",
  "motion_anchor_ori_b",
  "base_ang_vel",
  "joint_pos",
  "joint_vel",
  "actions",
)
_EXPECTED_ACTOR_WIDTHS: tuple[int, ...] = (
  2 * JOINT_DIM,
  6,
  3,
  JOINT_DIM,
  JOINT_DIM,
  JOINT_DIM,
)
_ZERO_LOOKAHEAD_ACTOR_TERMS: tuple[str, ...] = (
  _EXPECTED_ACTOR_TERMS[0],
  REFERENCE_PREVIEW_FIELD,
  *_EXPECTED_ACTOR_TERMS[1:],
)
_ZERO_LOOKAHEAD_ACTOR_WIDTHS: tuple[int, ...] = (
  _EXPECTED_ACTOR_WIDTHS[0],
  0,
  *_EXPECTED_ACTOR_WIDTHS[1:],
)
_PREVIEW_ACTOR_TERMS: tuple[str, ...] = (
  _EXPECTED_ACTOR_TERMS[0],
  REFERENCE_PREVIEW_FIELD,
  *_EXPECTED_ACTOR_TERMS[1:],
)
_PREVIEW_ACTOR_WIDTHS: tuple[int, ...] = (
  _EXPECTED_ACTOR_WIDTHS[0],
  PREVIEW_DIM,
  *_EXPECTED_ACTOR_WIDTHS[1:],
)
_SIGNATURE_FIELDS: tuple[str, ...] = (
  "root_position",
  "root_quaternion_wxyz",
  "joint_position",
  "joint_velocity",
)


def _cohort_lookahead_s(cohort: Any) -> float:
  """Return the cohort's declared preview duration, defaulting to version 1."""
  try:
    value = float(getattr(cohort, "lookahead_s", 0.0))
  except (TypeError, ValueError) as exc:
    raise RuntimeX2Error(f"cohort lookahead_s is not a number: {exc}") from exc
  if not math.isfinite(value) or value < 0.0:
    raise RuntimeX2Error(
      f"cohort lookahead_s must be finite and non-negative, got {value!r}"
    )
  return value


def _has_reference_preview(cohort: Any) -> bool:
  return _cohort_lookahead_s(cohort) > 0.0


def _reference_width(preview: bool) -> int:
  return REFERENCE_DIM_WITH_PREVIEW if preview else REFERENCE_DIM


def _actor_width(preview: bool) -> int:
  return sum(_PREVIEW_ACTOR_WIDTHS if preview else _EXPECTED_ACTOR_WIDTHS)


def _validate_actor_layout(observations: Any, *, preview: bool) -> int:
  """Accept only the pinned channels plus the cohort-declared preview term."""
  names = tuple(observations.names)
  widths = tuple(observations.widths)
  if preview:
    allowed = ((_PREVIEW_ACTOR_TERMS, _PREVIEW_ACTOR_WIDTHS),)
  else:
    allowed = (
      (_EXPECTED_ACTOR_TERMS, _EXPECTED_ACTOR_WIDTHS),
      (_ZERO_LOOKAHEAD_ACTOR_TERMS, _ZERO_LOOKAHEAD_ACTOR_WIDTHS),
    )
  if (names, widths) not in allowed:
    expected = "the 62-wide declared preview" if preview else "no preview"
    raise RuntimeX2Error(
      "unsupported schema: expected the pinned actor channels with "
      f"{expected}, got {names!r}/{widths!r}"
    )
  expected_dim = sum(widths)
  if int(getattr(observations, "total_dim", expected_dim)) != expected_dim:
    raise RuntimeX2Error(
      "unsupported schema: actor term widths do not sum to the declared "
      f"observation width {expected_dim}"
    )
  return expected_dim


def _assemble_reference(
  values: Mapping[str, np.ndarray], *, preview: bool
) -> np.ndarray:
  """Build the gravity reference in saved schema order for one or many rows."""
  parts = [
    np.asarray(values["command"])[..., : 2 * JOINT_DIM],
    np.asarray(values["motion_anchor_ori_b"]),
  ]
  if preview:
    preview_value = values.get(REFERENCE_PREVIEW_FIELD)
    if preview_value is None:
      raise RuntimeX2Error(
        f"the preview layout needs the {REFERENCE_PREVIEW_FIELD!r} observation term"
      )
    parts.append(np.asarray(preview_value))
  reference = np.concatenate(parts, axis=-1)
  expected = _reference_width(preview)
  if reference.shape[-1] != expected:
    raise RuntimeX2Error(
      f"reference must have width {expected}, got {reference.shape[-1]}"
    )
  return reference


def _never_terminate(env: Any, **_: Any) -> torch.Tensor:
  """Disable one reference guard without changing the time-out term."""
  return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _array(value: Any, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
  """Convert one simulator value to an owned finite array."""
  if isinstance(value, torch.Tensor):
    value = value.detach().cpu().numpy()
  result = np.asarray(value, dtype=np.float64)
  if result.shape != shape:
    raise RuntimeX2Error(f"{name} must have shape {shape}, got {result.shape}")
  if not np.isfinite(result).all():
    raise RuntimeX2Error(f"{name} contains non-finite simulator data")
  return np.array(result, dtype=np.float64, copy=True)


def _row(value: Any, *, name: str, width: int) -> np.ndarray:
  """Read one row of a one-environment simulator tensor."""
  if isinstance(value, torch.Tensor):
    value = value.detach().cpu().numpy()
  result = np.asarray(value, dtype=np.float64)
  if result.shape == (1, width):
    result = result[0]
  if result.shape != (width,):
    raise RuntimeX2Error(f"{name} must have shape [1, {width}], got {result.shape}")
  if not np.isfinite(result).all():
    raise RuntimeX2Error(f"{name} contains non-finite simulator data")
  return np.array(result, dtype=np.float64, copy=True)


def _body_rows(value: Any, indexes: np.ndarray, *, name: str) -> np.ndarray:
  """Select contract bodies while preserving the simulator's row order."""
  if isinstance(value, torch.Tensor):
    if value.ndim != 3 or value.shape[0] != 1:
      raise RuntimeX2Error(f"{name} must have shape [1, bodies, 3]")
    selected = value[0].index_select(
      0, torch.as_tensor(indexes, dtype=torch.long, device=value.device)
    )
    return _array(selected, name=name, shape=(len(indexes), 3))
  raw = np.asarray(value, dtype=np.float64)
  if raw.ndim != 3 or raw.shape[0] != 1:
    raise RuntimeX2Error(f"{name} must have shape [1, bodies, 3]")
  return _array(raw[0, indexes], name=name, shape=(len(indexes), 3))


def _scalar_bool(value: Any, *, name: str) -> bool:
  if isinstance(value, torch.Tensor):
    value = value.detach().cpu().numpy()
  result = np.asarray(value)
  if result.size != 1:
    raise RuntimeX2Error(f"{name} must contain one environment flag")
  return bool(result.reshape(-1)[0])


def _scalar_int(value: Any, *, name: str) -> int:
  if isinstance(value, torch.Tensor):
    value = value.detach().cpu().numpy()
  result = np.asarray(value)
  if result.size != 1:
    raise RuntimeX2Error(f"{name} must contain one environment index")
  number = result.reshape(-1)[0]
  if not np.isfinite(number) or int(number) != number:
    raise RuntimeX2Error(f"{name} must be a finite integer")
  return int(number)


def _term_slices(cohort: Any) -> dict[str, slice]:
  """Build actor slices from the resolved cohort, not from a hard-coded offset."""
  offset = 0
  result: dict[str, slice] = {}
  for term in cohort.observations.terms:
    result[term.name] = slice(offset, offset + term.width)
    offset += term.width
  return result


def _copy_config_fields(cfg: Any) -> dict[str, Any]:
  return {field.name: getattr(cfg, field.name) for field in fields(cfg)}


class ClosedLoopMotionCommand(SegmentMotionCommand):
  """Segment command with an explicit reference-free frozen mode.

  The normal tracking command remains unchanged during the VAE bootstrap.  In
  reference-free mode every resampling and frame update is a no-op, including
  timer expiry and clip wrap.  Reference-derived metrics are also left at
  their last values; the surrounding runtime records this inactive computation
  in provenance rather than presenting it as continuation input.
  """

  def __init__(self, cfg: Any, env: Any):
    self._reference_free = False
    super().__init__(cfg, env)

  @property
  def reference_free(self) -> bool:
    return self._reference_free

  def freeze_reference(self) -> None:
    self._reference_free = True

  def restore_reference(self) -> None:
    self._reference_free = False

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    # A reset is an explicit ownership boundary and always returns to VAE mode.
    self._reference_free = False
    return super().reset(env_ids)

  def compute(
    self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None
  ) -> None:
    if self._reference_free:
      return
    super().compute(dt=dt, env_ids=env_ids)

  def _terminal_step(self) -> bool:
    """Report termination already computed before command-manager updates."""
    manager = getattr(self._env, "termination_manager", None)
    if manager is None:
      return False
    flags = tuple(getattr(manager, name, None) for name in ("terminated", "time_outs"))
    return any(
      value is not None
      and bool(torch.as_tensor(value, device=self.device, dtype=torch.bool).any())
      for value in flags
    )

  def _resample(self, env_ids: torch.Tensor) -> None:
    if not self._reference_free and not self._terminal_step():
      super()._resample(env_ids)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if not self._reference_free:
      super()._resample_command(env_ids)

  def _update_metrics(self) -> None:
    if not self._reference_free:
      super()._update_metrics()

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    if not self._reference_free and not self._terminal_step():
      super()._update_command(env_ids)


class ClosedLoopMotionCommandCfg(SegmentMotionCommandCfg):
  """Private config selecting :class:`ClosedLoopMotionCommand`."""

  def build(self, env: Any) -> ClosedLoopMotionCommand:
    return ClosedLoopMotionCommand(self, env)

  @classmethod
  def from_config(cls, cfg: Any) -> ClosedLoopMotionCommandCfg:
    return cls(**_copy_config_fields(cfg))


class X2ClosedLoopRuntime:
  """One-env X2 simulator boundary implementing ``ClosedLoopRuntime``."""

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
    self._lookahead_s = _cohort_lookahead_s(cohort)
    self._preview = self._lookahead_s > 0.0
    self._reference_width = _reference_width(self._preview)
    self._actor_width = _actor_width(self._preview)
    self._closed = False
    self._has_reset = False
    self._phase = "vae"
    self._tick = 0
    self._sim_time = 0.0
    self._last_pre_action: np.ndarray | None = None
    self._initial_signature_value: np.ndarray | None = None
    self._segment_id = f"{member}:uninitialized"

    if int(getattr(env, "num_envs", 0)) != 1:
      raise RuntimeX2Error("closed-loop runtime requires exactly one environment")
    try:
      self.period_seconds = float(env.step_dt)
    except (AttributeError, TypeError, ValueError) as exc:
      raise RuntimeX2Error("environment does not expose a numeric step_dt") from exc
    if not math.isclose(
      self.period_seconds, DEFAULT_CONTRACT.period_seconds, abs_tol=1e-9
    ):
      raise RuntimeX2Error(
        "unsupported timing: closed-loop D3 requires a 0.02 s simulator step, "
        f"got {self.period_seconds!r}"
      )

    self.joint_order, self.body_order = self._validate_live_order()
    self._command = self._get_motion_command()
    try:
      self.clip_frames = int(self._command.motion.time_step_total)
    except (AttributeError, TypeError, ValueError) as exc:
      raise RuntimeX2Error("motion command does not expose clip frame count") from exc
    if self.clip_frames < 2:
      raise RuntimeX2Error(
        f"motion clip must contain at least two frames, got {self.clip_frames}"
      )
    self._slices = _term_slices(cohort)
    self._row_ids = torch.arange(
      1, dtype=torch.int64, device=getattr(env, "device", "cpu")
    )
    self._reference_terms = self._save_reference_terms()
    if hasattr(getattr(env, "cfg", None), "auto_reset"):
      env.cfg.auto_reset = False

    self._set_reference_terminations(True)
    self._provenance: dict[str, Any] = {
      "runtime": "mjlab_x2_closed_loop",
      "physical_runtime": "synthetic",
      "synthetic": True,
      "measurement_source": "injected_test_environment_robot_data",
      "measurement_status": (
        "synthetic/test measurements from an injected environment; not physical "
        "evidence"
      ),
      "member": member,
      "device": str(getattr(env, "device", "unknown")),
      "control_period_s": self.period_seconds,
      "actor_observation_layout": {
        "names": list(cohort.observations.names),
        "widths": list(cohort.observations.widths),
        "dimension": self._actor_width,
        "ignored_zero_width_terms": [
          name
          for name, width in zip(
            cohort.observations.names, cohort.observations.widths, strict=True
          )
          if name == REFERENCE_PREVIEW_FIELD and width == 0
        ],
      },
      "action_boundary": "direct env.step(action_tensor), shape [1, 31]",
      "auto_reset": {"requested": False, "applied": True},
      "cached_observation_capture": {
        "source": "env.obs_buf / observation_manager cache",
        "update_history": False,
        "noise_or_delay_update": False,
        "projected_gravity": "robot.data.projected_gravity_b (root frame)",
        "conditioning_width": 99,
        "reference_width": self._reference_width,
      },
      "reference_layout": {
        "width": self._reference_width,
        "preview": self._preview,
        "preview_field": REFERENCE_PREVIEW_FIELD if self._preview else None,
        "preview_width": PREVIEW_DIM if self._preview else 0,
        "cohort_lookahead_s": self._lookahead_s,
        "order": [
          "reference_q",
          "reference_dq",
          "anchor_orientation_error",
          *([REFERENCE_PREVIEW_FIELD] if self._preview else []),
        ],
        "source": "cached actor observation of the same command frame",
      },
      "reference_read_boundaries": {
        "reset": "allowed for deterministic frame initialization",
        "vae_bootstrap": "allowed for reference-tracking conditioning",
        "enter_reference_free": "no reference read and no physical-state advance",
        "continuation": "reference and reference_frame are absent; no command read",
      },
      "overrides": [
        "private motion command freezes timer resampling and clip wrap after handover",
        "private reference terminations are replaced with always-false functions after handover",
        "private environment auto_reset=False",
      ],
      "inactive_reference_computation": (
        "The manager may retain cached reference terms and tracking rewards after "
        "handover; they are not read by continuation observations or actions."
      ),
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
    """Mark only the verified factory path as real mjlab runtime evidence."""
    self._provenance.update(
      {
        "physical_runtime": "mjlab",
        "synthetic": False,
        "measurement_source": "real_mjlab_simulator_robot_data",
        "measurement_status": "real measurements from ManagerBasedRlEnv",
        "factory_trusted": True,
      }
    )

  def _get_motion_command(self) -> Any:
    try:
      command = self.env.command_manager.get_term("motion")
    except (AttributeError, KeyError, TypeError) as exc:
      raise RuntimeX2Error("environment has no motion command term") from exc
    if not hasattr(command, "time_steps"):
      raise RuntimeX2Error("motion command lacks an integer reference frame clock")
    return command

  def _validate_live_order(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
    expected_joints = tuple(self.cohort.actions.joint_names)
    if len(expected_joints) != DEFAULT_CONTRACT.action_dimension:
      raise RuntimeX2Error(
        "unsupported joint schema: D3 gravity VAE requires 31 action joints"
      )
    robot = self.env.scene["robot"]
    actual_joints = tuple(getattr(robot, "joint_names", ()))
    if actual_joints != expected_joints:
      raise RuntimeX2Error(
        "unsupported joint order: live robot order differs from the cohort "
        f"({actual_joints!r} != {expected_joints!r})"
      )
    expected_bodies = tuple(DEFAULT_CONTRACT.body_names)
    declared_bodies = tuple(getattr(self.cohort, "body_names", ()))
    if declared_bodies != expected_bodies:
      raise RuntimeX2Error(
        "unsupported body order: cohort body order is not the frozen X2 order"
      )
    actual_bodies = tuple(getattr(robot, "body_names", ()))
    missing = [name for name in expected_bodies if name not in actual_bodies]
    if missing:
      raise RuntimeX2Error(f"unsupported body order: live robot is missing {missing}")
    command = self.env.command_manager.get_term("motion")
    command_bodies = tuple(getattr(getattr(command, "cfg", None), "body_names", ()))
    if command_bodies and command_bodies != expected_bodies:
      raise RuntimeX2Error(
        "unsupported body order: motion command body order differs from the "
        "frozen diffusion state order"
      )
    observations = self.cohort.observations
    expected_actor_dim = _validate_actor_layout(observations, preview=self._preview)
    actor_dim = int(getattr(self.cohort.actor, "obs_dim", expected_actor_dim))
    if actor_dim != expected_actor_dim:
      raise RuntimeX2Error(
        "unsupported schema: expected actor obs_dim "
        f"{expected_actor_dim}, got {actor_dim}"
      )
    control_period = float(self.cohort.control.control_period_s)
    if not math.isclose(control_period, DEFAULT_CONTRACT.period_seconds, abs_tol=1e-9):
      raise RuntimeX2Error(
        "unsupported timing: cohort control period must be 0.02 s, "
        f"got {control_period!r}"
      )
    if int(self.cohort.actions.dim) != DEFAULT_CONTRACT.action_dimension:
      raise RuntimeX2Error("unsupported schema: action dimension must be 31")
    return expected_joints, expected_bodies

  def _save_reference_terms(self) -> dict[str, tuple[Any, Any, dict[str, Any]]]:
    manager = getattr(self.env, "termination_manager", None)
    if manager is None:
      raise RuntimeX2Error("environment has no termination manager")
    active = tuple(getattr(manager, "active_terms", ()))
    configs = getattr(manager, "_term_cfgs", ())
    saved: dict[str, tuple[Any, Any, dict[str, Any]]] = {}
    for name in _REFERENCE_TERMINATIONS:
      if name not in active:
        raise RuntimeX2Error(f"tracking task is missing reference guard {name!r}")
      index = active.index(name)
      try:
        term_cfg = configs[index]
        if bool(term_cfg.time_out):
          raise RuntimeX2Error(f"reference guard {name!r} cannot be a time-out term")
        saved[name] = (term_cfg, term_cfg.func, dict(term_cfg.params))
      except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeX2Error(f"malformed termination config for {name!r}") from exc
    return saved

  def _set_reference_terminations(self, enabled: bool) -> None:
    manager = self.env.termination_manager
    active = tuple(manager.active_terms)
    for name, (term_cfg, func, params) in self._reference_terms.items():
      if name not in active:
        raise RuntimeX2Error(f"reference guard {name!r} disappeared from the manager")
      if enabled:
        term_cfg.func = func
        term_cfg.params = dict(params)
      else:
        term_cfg.func = _never_terminate
        term_cfg.params = {}

  def _world_state(self) -> WorldState:
    data = self.env.scene["robot"].data
    body_indexes = np.asarray(self._body_indexes, dtype=np.int64)
    return WorldState(
      root_position=_row(data.root_link_pos_w, name="root_link_pos_w", width=3),
      root_quaternion_wxyz=_row(
        data.root_link_quat_w, name="root_link_quat_w", width=4
      ),
      root_linear_velocity=_row(
        data.root_link_lin_vel_w, name="root_link_lin_vel_w", width=3
      ),
      root_angular_velocity=_row(
        data.root_link_ang_vel_w, name="root_link_ang_vel_w", width=3
      ),
      body_positions=_body_rows(
        data.body_link_pos_w, body_indexes, name="body_link_pos_w"
      ),
      body_linear_velocities=_body_rows(
        data.body_link_lin_vel_w, body_indexes, name="body_link_lin_vel_w"
      ),
    )

  @property
  def _body_indexes(self) -> tuple[int, ...]:
    robot_names = tuple(self.env.scene["robot"].body_names)
    return tuple(robot_names.index(name) for name in self.body_order)

  def _initial_signature_now(self) -> np.ndarray:
    data = self.env.scene["robot"].data
    values = np.concatenate(
      (
        _row(data.root_link_pos_w, name="root_link_pos_w", width=3),
        _row(data.root_link_quat_w, name="root_link_quat_w", width=4),
        _row(
          data.joint_pos,
          name="joint_pos",
          width=len(self.joint_order),
        ),
        _row(
          data.joint_vel,
          name="joint_vel",
          width=len(self.joint_order),
        ),
      )
    )
    return np.array(values, dtype=np.float64, copy=True)

  def _cached_actor(self) -> np.ndarray:
    """Read the existing post-reset/post-step cache without recomputation."""
    cached = getattr(self.env, "obs_buf", None)
    if not isinstance(cached, Mapping) or "actor" not in cached:
      manager = getattr(self.env, "observation_manager", None)
      cached = getattr(manager, "_obs_buffer", None)
    if not isinstance(cached, Mapping) or "actor" not in cached:
      manager = getattr(self.env, "observation_manager", None)
      if manager is None or not hasattr(manager, "compute"):
        raise RuntimeX2Error("no cached actor observation is available")
      # This fallback is only for a freshly constructed fake/live env with no
      # installed cache.  Normal reset/step paths never reach it.
      cached = manager.compute(update_history=False)
    actor = cached.get("actor") if isinstance(cached, Mapping) else None
    if isinstance(actor, torch.Tensor):
      actor = actor.detach().cpu().numpy()
    actor_array = np.asarray(actor, dtype=np.float64)
    if actor_array.shape == (self._actor_width,):
      actor_array = actor_array[None]
    if actor_array.shape != (1, self._actor_width):
      raise RuntimeX2Error(
        "cached actor observation must have shape "
        f"[1, {self._actor_width}], got {actor_array.shape}"
      )
    if not np.isfinite(actor_array).all():
      raise RuntimeX2Error("cached actor observation contains non-finite values")
    return np.array(actor_array[0], dtype=np.float64, copy=True)

  def _frame(self) -> int:
    return _scalar_int(self._command.time_steps, name="reference frame")

  def _capture_observation(self) -> D3Observation:
    actor = self._cached_actor()
    if self._phase == "vae":
      values = {name: actor[span].copy() for name, span in self._slices.items()}
    else:
      # Reference-free continuation reads only the actual conditioning terms;
      # the command, anchor and preview slices are not touched at all.
      values = {name: actor[self._slices[name]].copy() for name in _CONDITIONING_TERMS}
    robot_data = self.env.scene["robot"].data
    gravity = _row(robot_data.projected_gravity_b, name="projected_gravity_b", width=3)
    conditioning = np.concatenate(
      (
        gravity,
        values["base_ang_vel"],
        values["joint_pos"],
        values["joint_vel"],
        values["actions"],
      )
    )
    if conditioning.shape != (99,):
      raise RuntimeX2Error(
        f"gravity conditioning must have shape [99], got {conditioning.shape}"
      )
    reference: np.ndarray | None = None
    reference_frame: int | None = None
    if self._phase == "vae":
      reference = _assemble_reference(values, preview=self._preview)
      if reference.shape != (self._reference_width,):
        raise RuntimeX2Error(
          f"reference must have shape [{self._reference_width}], got {reference.shape}"
        )
      reference_frame = self._frame()
    return D3Observation(
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
    available = _scalar_bool(event.available, name="boundary availability")
    completed = _scalar_bool(event.completed, name="completed boundary")
    interrupted = _scalar_bool(event.interrupted, name="interrupted boundary")
    reasons = getattr(event, "reasons", ())
    reason = reasons[0] if reasons and reasons[0] != "unavailable" else None
    return available, completed, interrupted, reason

  def reset(self, *, seed: int, start_frame: int) -> D3Observation:
    """Reset all private state and deterministically place the reference frame."""
    if isinstance(seed, bool) or not isinstance(seed, int):
      raise RuntimeX2Error("seed must be an integer")
    if start_frame < 0 or start_frame >= self.clip_frames:
      raise RuntimeX2Error(
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
    self._initial_signature_value = self._initial_signature_now()
    self._provenance["initial_state_signature_digest"] = hashlib.sha256(
      self._initial_signature_value.tobytes()
    ).hexdigest()
    self._provenance["initial_state_signature_seed"] = int(seed)
    self._provenance["initial_state_signature_start_frame"] = int(start_frame)
    self._provenance["mode"] = "vae"
    return self._capture_observation()

  def initial_signature(self) -> np.ndarray:
    if self._initial_signature_value is None:
      raise RuntimeX2Error("initial_signature requested before reset")
    return self._initial_signature_value.copy()

  def enter_reference_free(self) -> D3Observation:
    """Freeze reference state and guards without stepping or resetting physics."""
    if not self._has_reset:
      raise RuntimeX2Error("enter_reference_free requires reset first")
    before = self._initial_signature_now()
    freeze = getattr(self._command, "freeze_reference", None)
    if freeze is None:
      raise RuntimeX2Error("private motion command cannot freeze reference updates")
    freeze()
    self._set_reference_terminations(False)
    after = self._initial_signature_now()
    if not np.array_equal(before, after):
      raise RuntimeX2Error(
        "enter_reference_free changed physical state; refusing a discontinuous handover"
      )
    self._phase = "reference_free"
    self._provenance["mode"] = "reference_free"
    self._provenance["handover"] = {
      "advanced_simulation": False,
      "reset_physical_state": False,
      "reference_command_frozen": True,
      "reference_guards_disabled": True,
    }
    return self._capture_observation()

  def step(self, action: np.ndarray) -> D3Transition:
    """Apply one action and capture post-step evidence before any reset."""
    if not self._has_reset:
      raise RuntimeX2Error("step requires reset first")
    values = np.asarray(action, dtype=np.float64)
    if values.shape != (DEFAULT_CONTRACT.action_dimension,):
      raise RuntimeX2Error(f"action must have shape [31], got {values.shape}")
    if not np.isfinite(values).all():
      raise RuntimeX2Error("refusing to step a non-finite action")
    self._last_pre_action = np.array(values, dtype=np.float64, copy=True)
    if self._phase == "vae":
      pre_frame = self._frame()
      if pre_frame >= self.clip_frames - 1:
        self._provenance["last_frame_refusal"] = {
          "reference_frame": pre_frame,
          "physical_step_executed": False,
          "reason": "final_reference_frame_has_no_successor",
        }
        raise RuntimeX2Error(
          "refusing VAE action at the final reference frame; "
          "no physical step was executed"
        )
    else:
      pre_frame = None
    pre_generation = None
    if self._phase == "vae" and hasattr(self._command, "generation_ids"):
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
    )[None]
    try:
      result = self.env.step(tensor)
    except Exception as exc:
      raise RuntimeX2Error(f"simulator action step failed: {exc}") from exc
    if not isinstance(result, tuple) or len(result) < 5:
      raise RuntimeX2Error(
        "simulator step must return observations, reward, done flags and extras"
      )
    if isinstance(result[0], Mapping):
      self.env.obs_buf = result[0]
    terminated = _scalar_bool(result[2], name="terminated")
    truncated = _scalar_bool(result[3], name="time_outs")
    extras = result[4] if isinstance(result[4], Mapping) else {}
    self._tick += 1
    self._sim_time = self._tick * self.period_seconds
    observation = self._capture_observation()
    teleport = False
    reference_boundary = False
    reason: str | None = None
    if self._phase == "vae":
      post_frame = self._frame()
      available, completed, interrupted, reason = self._event_flags(pre_generation)
      terminal_no_advance = bool(
        (terminated or truncated) and pre_frame is not None and post_frame == pre_frame
      )
      frame_discontinuity = bool(
        pre_frame is not None
        and post_frame != pre_frame + 1
        and not terminal_no_advance
      )
      reference_boundary = (
        available or post_frame >= self.clip_frames - 1 or frame_discontinuity
      )
      teleport = interrupted or completed or frame_discontinuity
      if reference_boundary and reason is None:
        reason = "reference_boundary"
    reset = bool(extras.get("reset", False))
    if reset and reason is None:
      reason = "reset"
    if (terminated or truncated) and reason is None:
      reason = "terminated" if terminated else "time_out"
    return D3Transition(
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


# Names used by callers/review tools; the implementation remains one class.
ClosedLoopRuntimeX2 = X2ClosedLoopRuntime


def _validate_cohort_static(cohort: Any) -> None:
  """Reject unsupported frozen D3 inputs before constructing an environment."""
  try:
    observations = cohort.observations
    preview = _has_reference_preview(cohort)
    expected_actor_dim = _validate_actor_layout(observations, preview=preview)
    actor_dim = int(getattr(cohort.actor, "obs_dim", expected_actor_dim))
    if int(observations.total_dim) != expected_actor_dim:
      raise RuntimeX2Error(
        "unsupported schema: expected a "
        f"{expected_actor_dim}-D actor observation, got {observations.total_dim}"
      )
    if actor_dim != expected_actor_dim:
      raise RuntimeX2Error(
        "unsupported schema: expected actor obs_dim "
        f"{expected_actor_dim}, got {actor_dim}"
      )
    if len(tuple(cohort.actions.joint_names)) != JOINT_DIM:
      raise RuntimeX2Error("unsupported joint schema: expected 31 action joints")
    if not math.isclose(
      float(cohort.control.control_period_s),
      DEFAULT_CONTRACT.period_seconds,
      abs_tol=1e-9,
    ):
      raise RuntimeX2Error(
        "unsupported timing: cohort control period must be exactly 0.02 s"
      )
    if tuple(cohort.body_names) != tuple(DEFAULT_CONTRACT.body_names):
      raise RuntimeX2Error(
        "unsupported body order: cohort body order differs from the frozen X2 order"
      )
  except RuntimeX2Error:
    raise
  except (AttributeError, TypeError, ValueError) as exc:
    raise RuntimeX2Error(f"unsupported cohort schema or timing: {exc}") from exc


def _validate_factory_inputs(
  manifest_path: str | Path,
  repo_root: str | Path,
  member: str,
  seed: int,
) -> tuple[Path, Any]:
  if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
    raise RuntimeX2Error("seed must be an integer in [0, 2**32 - 1]")
  if not isinstance(member, str) or not member:
    raise RuntimeX2Error("member must be a non-empty teacher id")
  root = Path(repo_root).resolve()
  manifest = Path(manifest_path)
  if not manifest.is_absolute():
    manifest = root / manifest
  manifest = manifest.resolve()
  try:
    cohort = resolve_cohort(load_manifest(manifest, repo_root=root))
    cohort.teacher(member)
  except Exception as exc:
    raise RuntimeX2Error(f"closed-loop cohort validation failed: {exc}") from exc
  if DEFAULT_CONTRACT.projection_rows != 0:
    raise RuntimeX2Error(
      "unsupported diffusion schema: D3 requires the no-mixture contract"
    )
  _validate_cohort_static(cohort)
  return manifest, cohort


def _build_private_environment(
  cohort: Any,
  member: str,
  *,
  repo_root: Path,
  device: str,
  seed: int,
) -> Any:
  """Build a private registry-derived config with the local command class."""
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg
  from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg

  selected_task = cohort.manifest.base_task
  if selected_task is None:
    raise RuntimeX2Error("cohort has no registered base_task")
  cfg = copy.deepcopy(load_env_cfg(selected_task))
  cfg.seed = seed
  teacher = cohort.teacher(member)
  overrides = list(
    _apply_saved_observation_timing(
      cfg, teacher.env_config, str(teacher.entry.env_config)
    )
  )
  cfg.scene.num_envs = 1
  cfg.auto_reset = False
  motion_cfg = cfg.commands.get("motion")
  if not isinstance(motion_cfg, MotionCommandCfg):
    raise RuntimeX2Error(f"registered task {selected_task!r} has no motion command")
  private_motion = ClosedLoopMotionCommandCfg.from_config(motion_cfg)
  private_motion.motion_file = str(teacher.entry.motion)
  # The saved teacher exports fix one preview duration for every cohort member,
  # and it already drives the student reference layout.  The base task often
  # declares no preview, so apply the cohort's value here as the private
  # builder does in the collection lane.
  lookahead_from_base = private_motion.lookahead_s
  lookahead_from_cohort = _cohort_lookahead_s(cohort)
  if lookahead_from_base != lookahead_from_cohort:
    private_motion.lookahead_s = lookahead_from_cohort
    overrides.append(
      "private preview override: commands.motion.lookahead_s: "
      f"{lookahead_from_base!r} -> {lookahead_from_cohort!r}"
    )
  cfg.commands["motion"] = private_motion
  env = ManagerBasedRlEnv(cfg, device=device, render_mode=None)
  env.cfg.__dict__["_closed_loop_semantic_overrides"] = tuple(overrides)
  env.cfg.__dict__["_closed_loop_seed_provenance"] = RuntimeSeedProvenance(
    requested_seed=seed,
    effective_seed=env.cfg.seed,
    applied_before_construction=True,
  )
  return env


def build_closed_loop_runtime(
  *,
  manifest_path: str | Path,
  repo_root: str | Path,
  member: str,
  device: str,
  seed: int,
  compare_reference_arrays: bool = True,
) -> ClosedLoopRuntime:
  """Build one verified private X2 runtime for a selected cohort member.

  ``compare_reference_arrays=False`` is for playback, where the teacher ONNX is
  never executed and the reference motion may be any clip.
  """
  manifest, cohort = _validate_factory_inputs(manifest_path, repo_root, member, seed)
  env: Any | None = None
  try:
    env = _build_private_environment(
      cohort,
      member,
      repo_root=Path(repo_root).resolve(),
      device=device,
      seed=seed,
    )
    audit = validate_live_contract(
      env,
      cohort,
      member,
      task_id=cohort.manifest.base_task,
      compare_reference_arrays=compare_reference_arrays,
    )
    runtime = X2ClosedLoopRuntime(
      env,
      cohort,
      member,
      audit=audit,
      manifest_path=manifest,
      seed=seed,
    )
    runtime._mark_factory_physical_runtime()
    return runtime
  except RuntimeX2Error:
    if env is not None:
      env.close()
    raise
  except Exception as exc:
    if env is not None:
      env.close()
    raise RuntimeX2Error(f"closed-loop runtime construction failed: {exc}") from exc


__all__ = [
  "ClosedLoopMotionCommand",
  "ClosedLoopMotionCommandCfg",
  "ClosedLoopRuntimeX2",
  "RuntimeX2Error",
  "X2ClosedLoopRuntime",
  "build_closed_loop_runtime",
]
