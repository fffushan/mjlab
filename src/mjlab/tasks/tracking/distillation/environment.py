"""Opt-in segment-aware motion command and trusted environment construction.

This module owns only the simulator boundary used by distillation.  It does not
alter the registered tracking tasks: :func:`build_distillation_environment`
loads a registered configuration, makes a private copy, and replaces only its
motion command with :class:`SegmentMotionCommand`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import torch

from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(frozen=True, slots=True)
class RuntimeSeedProvenance:
  """Seed applied to a private environment before construction."""

  requested_seed: int | None
  effective_seed: int | None
  applied_before_construction: bool


@dataclass(frozen=True, slots=True)
class ReferenceBoundaryEvents:
  """Per-environment motion events for one command-update window.

  ``completed_generation`` identifies the generation that actually reached its
  clip end. ``interrupted_generation`` identifies a generation replaced by a
  reset/timer resample. A timer resample followed by a same-update wrap thus
  interrupts the old generation and completes only the newly sampled one. The
  flags forwarded for one step therefore describe that step's PRE-step segment:
  termination takes precedence, and a generation is only ever reported as
  interrupted or completed when it was actually sampled.
  """

  available: torch.Tensor
  completed: torch.Tensor
  interrupted: torch.Tensor
  pre_generation: torch.Tensor
  post_generation: torch.Tensor
  completed_generation: torch.Tensor
  interrupted_generation: torch.Tensor
  reasons: tuple[str, ...]

  def __post_init__(self) -> None:
    batch = self.available.shape[0]
    for name in (
      "completed",
      "interrupted",
      "pre_generation",
      "post_generation",
      "completed_generation",
      "interrupted_generation",
    ):
      value = getattr(self, name)
      if value.shape != (batch,):
        raise ValueError(f"{name} must have shape [{batch}]")
    if len(self.reasons) != batch:
      raise ValueError(f"reasons must contain {batch} entries")

  @classmethod
  def empty(cls, batch: int, device: torch.device) -> ReferenceBoundaryEvents:
    return cls(
      available=torch.zeros(batch, dtype=torch.bool, device=device),
      completed=torch.zeros(batch, dtype=torch.bool, device=device),
      interrupted=torch.zeros(batch, dtype=torch.bool, device=device),
      pre_generation=torch.zeros(batch, dtype=torch.long, device=device),
      post_generation=torch.zeros(batch, dtype=torch.long, device=device),
      completed_generation=torch.full((batch,), -1, dtype=torch.long, device=device),
      interrupted_generation=torch.full((batch,), -1, dtype=torch.long, device=device),
      reasons=tuple("unavailable" for _ in range(batch)),
    )

  def with_pre_generation(self, generation: torch.Tensor) -> ReferenceBoundaryEvents:
    if generation.shape != self.pre_generation.shape:
      raise ValueError("pre-generation shape does not match boundary events")
    return type(self)(
      self.available,
      self.completed,
      self.interrupted,
      generation.detach().clone().to(dtype=torch.long),
      self.post_generation,
      self.completed_generation,
      self.interrupted_generation,
      self.reasons,
    )

  def with_step_outcome(self, terminated: torch.Tensor) -> ReferenceBoundaryEvents:
    terminated = terminated.to(dtype=torch.bool)
    if terminated.shape != self.completed.shape:
      raise ValueError("termination shape does not match boundary events")
    completed = self.completed & ~terminated
    interrupted = self.interrupted | terminated
    interrupted_generation = torch.where(
      terminated & ~self.interrupted,
      self.pre_generation,
      self.interrupted_generation,
    )
    reasons = tuple(
      (
        "terminated"
        if bool(flag) and reason == "unavailable"
        else (reason + "+terminated")
        if bool(flag)
        else reason
      )
      for reason, flag in zip(self.reasons, terminated.tolist(), strict=True)
    )
    return type(self)(
      self.available | terminated,
      completed,
      interrupted,
      self.pre_generation,
      self.post_generation,
      torch.where(
        completed,
        self.completed_generation,
        torch.full_like(self.completed_generation, -1),
      ),
      interrupted_generation,
      reasons,
    )


class SegmentMotionCommand(MotionCommand):
  """Motion command with explicit per-environment generation boundaries.

  ``generation_ids`` and ``segment_ids`` are incremented whenever the command
  resamples, including reset-path sampling, timer resampling, and wraparound
  teleports. A frame index is never used as a proxy for continuity. Each
  resample is classified exactly once, so the counters advance once per real
  boundary.
  """

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.generation_ids = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.segment_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._event_available = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._event_completed = torch.zeros_like(self._event_available)
    self._event_interrupted = torch.zeros_like(self._event_available)
    self._event_completed_generation = torch.full_like(self.generation_ids, -1)
    self._event_interrupted_generation = torch.full_like(self.generation_ids, -1)
    self._event_reasons = ["unavailable"] * self.num_envs
    self._reset_resample_ids = torch.empty(0, dtype=torch.long, device=self.device)
    self._timer_resample_ids = torch.empty(0, dtype=torch.long, device=self.device)
    self._wrap_resample_ids = torch.empty(0, dtype=torch.long, device=self.device)

  def _record_event(self, env_ids: torch.Tensor, reason: str) -> None:
    if env_ids.numel() == 0:
      return
    self._event_available[env_ids] = True
    if reason == "wrap":
      self._event_completed[env_ids] = True
      self._event_completed_generation[env_ids] = self.generation_ids[env_ids]
    else:
      self._event_interrupted[env_ids] = True
      self._event_interrupted_generation[env_ids] = self.generation_ids[env_ids]
    for index in env_ids.detach().cpu().tolist():
      previous = self._event_reasons[index]
      part = "reference_completed" if reason == "wrap" else reason
      self._event_reasons[index] = (
        part if previous == "unavailable" else f"{previous}+{part}"
      )

  def consume_boundary_events(
    self, pre_generation: torch.Tensor | None = None
  ) -> ReferenceBoundaryEvents:
    """Return and clear events since the previous consume call."""
    if pre_generation is None:
      pre_generation = self._event_interrupted_generation.clone()
      missing = pre_generation < 0
      pre_generation[missing] = self.generation_ids[missing]
    result = ReferenceBoundaryEvents(
      self._event_available.clone(),
      self._event_completed.clone(),
      self._event_interrupted.clone(),
      pre_generation.detach().clone().to(dtype=torch.long),
      self.generation_ids.detach().clone(),
      self._event_completed_generation.clone(),
      self._event_interrupted_generation.clone(),
      tuple(self._event_reasons),
    )
    self._event_available.zero_()
    self._event_completed.zero_()
    self._event_interrupted.zero_()
    self._event_completed_generation.fill_(-1)
    self._event_interrupted_generation.fill_(-1)
    self._event_reasons = ["unavailable"] * self.num_envs
    return result

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    if isinstance(env_ids, slice):
      ids = torch.arange(self.num_envs, device=self.device)
    elif env_ids is None:
      ids = torch.arange(self.num_envs, device=self.device)
    else:
      ids = env_ids
    self._reset_resample_ids = ids.detach().clone()
    try:
      return super().reset(env_ids)
    finally:
      self._reset_resample_ids = torch.empty(0, dtype=torch.long, device=self.device)

  def _resample(self, env_ids: torch.Tensor) -> None:
    """Tag only this timer-expiry resample, never a same-compute wrap.

    ``CommandTerm.compute`` runs the timer-expiry resample and then
    ``_update_command``, which can resample the same env again when the frame
    sampled here is the clip's last one.  A wrap-triggered resample must be
    classified once, as a wrap: leaving the timer tag set would record a second
    ``timer_resampled`` boundary and advance the generation twice for one
    resample.  Reset-path resamples are classified from
    ``_reset_resample_ids``, so they never take the timer tag.
    """
    if self._reset_resample_ids.numel() == 0:
      self._timer_resample_ids = env_ids.detach().clone()
    try:
      super()._resample(env_ids)
    finally:
      self._timer_resample_ids = torch.empty(0, dtype=torch.long, device=self.device)

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      candidate = torch.arange(self.num_envs, device=self.device)
    else:
      candidate = env_ids
    wraps = candidate[self.time_steps[candidate] + 1 >= self.motion.time_step_total]
    self._wrap_resample_ids = wraps
    try:
      super()._update_command(env_ids)
    finally:
      self._wrap_resample_ids = torch.empty(0, dtype=torch.long, device=self.device)

  def _mark_boundary(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    self.generation_ids[env_ids] += 1
    self.segment_ids[env_ids] += 1

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if hasattr(self, "generation_ids"):
      reset_ids = env_ids[torch.isin(env_ids, self._reset_resample_ids)]
      timer_ids = env_ids[torch.isin(env_ids, self._timer_resample_ids)]
      wrap_ids = env_ids[torch.isin(env_ids, self._wrap_resample_ids)]
      other_ids = env_ids[
        ~torch.isin(env_ids, torch.cat((reset_ids, timer_ids, wrap_ids)))
      ]
      for ids, reason in (
        (reset_ids, "reset"),
        (timer_ids, "timer_resampled"),
        (other_ids, "resampled"),
      ):
        if ids.numel():
          self._record_event(ids, reason)
          self._mark_boundary(ids)
      # Wrap is recorded after any timer/reset generation increment, so a
      # same-step timer/reset+wrap completes only the newly sampled generation.
      # The timer tag covers only the timer-expiry resample (see ``_resample``),
      # so a wrap in that same compute is never re-tagged as a second timer
      # resample: each resample is classified exactly once.
      if wrap_ids.numel():
        self._record_event(wrap_ids, "wrap")
        self._mark_boundary(wrap_ids)
    super()._resample_command(env_ids)

  def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
    self._record_event(env_ids, "reset_to_frame")
    self._mark_boundary(env_ids)
    super().reset_to_frame(env_ids, frame)


class SegmentMotionCommandCfg(MotionCommandCfg):
  """Private config type selecting :class:`SegmentMotionCommand`."""

  def build(self, env: ManagerBasedRlEnv) -> SegmentMotionCommand:
    return SegmentMotionCommand(self, env)

  @classmethod
  def from_config(cls, cfg: MotionCommandCfg) -> SegmentMotionCommandCfg:
    values = {field.name: getattr(cfg, field.name) for field in fields(cfg)}
    return cls(**values)


def make_segment_motion_cfg(cfg: MotionCommandCfg) -> SegmentMotionCommandCfg:
  """Copy a trusted motion config without executing saved YAML callables."""
  if isinstance(cfg, SegmentMotionCommandCfg):
    return cfg
  return SegmentMotionCommandCfg.from_config(cfg)


def _validate_saved_delay_value(field: str, value: Any) -> None:
  if field in {"delay_min_lag", "delay_max_lag", "delay_update_period"}:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
      raise ValueError(f"saved {field} must be a non-negative integer")
  elif field in {"delay_per_env", "delay_per_env_phase"}:
    if not isinstance(value, bool):
      raise ValueError(f"saved {field} must be boolean")
  elif field == "delay_hold_prob":
    if isinstance(value, bool) or not isinstance(value, (int, float)):
      raise ValueError("saved delay_hold_prob must be numeric")
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
      raise ValueError("saved delay_hold_prob must be finite and in [0, 1]")
  elif field == "delay_group" and value is not None and not isinstance(value, str):
    raise ValueError("saved delay_group must be a string or null")


def _apply_saved_observation_timing(
  cfg: Any, saved_env_config: Mapping[str, Any], source: str
) -> tuple[str, ...]:
  """Restore only primitive delay timing on the private trusted task copy."""
  saved_group = saved_env_config.get("observations", {}).get("actor", {})
  if not isinstance(saved_group, Mapping):
    raise ValueError("saved actor observation group is not a mapping")
  live_group = cfg.observations.get("actor")
  saved_terms = saved_group.get("terms", {})
  if live_group is None or not isinstance(saved_terms, Mapping):
    raise ValueError("saved actor observation terms are not a mapping")
  overrides: list[str] = []
  fields_to_restore = (
    "delay_min_lag",
    "delay_max_lag",
    "delay_per_env",
    "delay_hold_prob",
    "delay_update_period",
    "delay_per_env_phase",
    "delay_group",
  )
  for name, saved_term in saved_terms.items():
    if name not in live_group.terms or not isinstance(saved_term, Mapping):
      continue
    live_term = live_group.terms[name]
    for field in fields_to_restore:
      if field not in saved_term:
        continue
      value = saved_term[field]
      _validate_saved_delay_value(field, value)
      if field == "delay_max_lag" and value < saved_term.get("delay_min_lag", 0):
        raise ValueError("saved delay_max_lag must not be below delay_min_lag")
      old = getattr(live_term, field)
      if old != value:
        setattr(live_term, field, value)
        overrides.append(
          f"{source}: observations.actor.terms.{name}.{field}: {old!r} -> {value!r}"
        )
  return tuple(overrides)


def _validate_factory_seed(seed: int | None) -> int | None:
  if seed is None:
    return None
  if isinstance(seed, bool) or not isinstance(seed, int):
    raise ValueError("seed must be an integer or None")
  if not 0 <= seed <= 2**32 - 1:
    raise ValueError("seed must be in [0, 2**32 - 1]")
  return seed


def build_distillation_environment(
  cohort: Any,
  teacher_id: str = "tennis_000",
  *,
  task_id: str | None = None,
  num_envs: int | None = None,
  device: str = "cpu",
  render_mode: str | None = None,
  seed: int | None = None,
) -> ManagerBasedRlEnv:
  """Build one selected-teacher environment from a registered task factory.

  Only the registry task factory is executable.  Saved environment YAML is
  used by M1 for validation and never supplies constructors or callables.
  Resource overrides (``num_envs``, ``device``, and ``render_mode``) are
  intentionally explicit and are not semantic contract overrides.
  """
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  teacher = cohort.teacher(teacher_id)
  selected_task = task_id or cohort.manifest.base_task
  if selected_task is None:
    raise ValueError("cohort has no base_task; task_id is required")
  cfg = load_env_cfg(selected_task)
  validated_seed = _validate_factory_seed(seed)
  if validated_seed is not None:
    cfg.seed = validated_seed
  semantic_overrides = _apply_saved_observation_timing(
    cfg, teacher.env_config, str(teacher.entry.env_config)
  )
  if num_envs is not None:
    if num_envs <= 0:
      raise ValueError("num_envs must be positive")
    cfg.scene.num_envs = num_envs
  motion_cfg = cfg.commands.get("motion")
  if not isinstance(motion_cfg, MotionCommandCfg):
    raise ValueError(f"registered task {selected_task!r} has no motion command")
  motion_cfg = make_segment_motion_cfg(motion_cfg)
  motion_cfg.motion_file = str(teacher.entry.motion)
  cfg.commands["motion"] = motion_cfg
  env = ManagerBasedRlEnv(cfg, device=device, render_mode=render_mode)
  env.cfg.__dict__["_distillation_semantic_overrides"] = semantic_overrides
  env.cfg.__dict__["_distillation_seed_provenance"] = RuntimeSeedProvenance(
    requested_seed=validated_seed,
    effective_seed=env.cfg.seed,
    applied_before_construction=validated_seed is not None,
  )
  return env


__all__ = [
  "SegmentMotionCommand",
  "SegmentMotionCommandCfg",
  "build_distillation_environment",
  "make_segment_motion_cfg",
]
