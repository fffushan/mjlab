"""Opt-in segment-aware motion command and trusted environment construction.

This module owns only the simulator boundary used by distillation.  It does not
alter the registered tracking tasks: :func:`build_distillation_environment`
loads a registered configuration, makes a private copy, and replaces only its
motion command with :class:`SegmentMotionCommand`, and
:func:`build_multi_motion_environment` does the same for the opt-in
multi-motion command whose rows follow different reference clips.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from mjlab.tasks.tracking.distillation.coverage_policy import CoveragePolicy
from mjlab.tasks.tracking.distillation.reset_policy import ResetPolicy
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
    self._init_boundary_bookkeeping()

  def _init_boundary_bookkeeping(self) -> None:
    """Allocate the per-environment boundary event state.

    Kept as its own step so a motion command that does not build a single-clip
    ``MotionLoader`` can reuse the exact same bookkeeping.
    """
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
  motion_file: Path | str | None = None,
  disable_terminations: bool = False,
) -> ManagerBasedRlEnv:
  """Build one selected-teacher environment from a registered task factory.

  Only the registry task factory is executable.  Saved environment YAML is
  used by M1 for validation and never supplies constructors or callables.
  Resource overrides (``num_envs``, ``device``, and ``render_mode``) are
  intentionally explicit and are not semantic contract overrides.

  ``motion_file`` replaces the teacher's own clip for a reference-only playback
  of a motion the cohort never contained.  The teacher then supplies only the
  saved observation timing and the validation contract; its clip identity is not
  used.  The override is recorded as a semantic override so a report shows it.

  ``disable_terminations`` clears every termination term, so a fall, a large
  reference deviation, and the episode time limit no longer reset the scene and
  playback keeps running.  It is a playback aid: no reset means the episode
  counters and any reset-derived evidence describe one unbroken episode.
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
  semantic_overrides = list(
    _apply_saved_observation_timing(
      cfg, teacher.env_config, str(teacher.entry.env_config)
    )
  )
  if num_envs is not None:
    if num_envs <= 0:
      raise ValueError("num_envs must be positive")
    cfg.scene.num_envs = num_envs
  motion_cfg = cfg.commands.get("motion")
  if not isinstance(motion_cfg, MotionCommandCfg):
    raise ValueError(f"registered task {selected_task!r} has no motion command")
  motion_cfg = make_segment_motion_cfg(motion_cfg)
  if motion_file is None:
    motion_cfg.motion_file = str(teacher.entry.motion)
  else:
    override = Path(motion_file)
    if not override.is_file():
      raise ValueError(f"motion override is not a file: {override}")
    motion_cfg.motion_file = str(override)
    semantic_overrides.append(
      "private reference override: commands.motion.motion_file: "
      f"{teacher.entry.motion!s} -> {override!s}"
    )
  # The cohort's preview is not the base task's business.  Every member of a cohort
  # shares one ``lookahead_s`` and the saved teacher exports were produced with it, so a
  # pinned single-motion environment must produce that same sample.  Without this the
  # base task's (often zero) preview exposes a narrower ``motion_lookahead`` term than
  # the teacher artifact expects, and the pinned lane refuses a valid pair.  A cohort
  # that declares zero keeps the frozen version-1 layout, because the override is then
  # a no-op.  This mirrors ``build_multi_motion_environment`` exactly.
  lookahead_from_base = motion_cfg.lookahead_s
  if lookahead_from_base != cohort.lookahead_s:
    motion_cfg.lookahead_s = cohort.lookahead_s
    semantic_overrides.append(
      "private preview override: commands.motion.lookahead_s: "
      f"{lookahead_from_base!r} -> {cohort.lookahead_s!r}"
    )
  cfg.commands["motion"] = motion_cfg
  if disable_terminations:
    disabled = tuple(cfg.terminations)
    cfg.terminations = {}
    semantic_overrides.append(
      "private playback override: terminations disabled: " + ", ".join(disabled)
    )
  env = ManagerBasedRlEnv(cfg, device=device, render_mode=render_mode)
  env.cfg.__dict__["_distillation_semantic_overrides"] = semantic_overrides
  env.cfg.__dict__["_distillation_seed_provenance"] = RuntimeSeedProvenance(
    requested_seed=validated_seed,
    effective_seed=env.cfg.seed,
    applied_before_construction=validated_seed is not None,
  )
  return env


def build_multi_motion_environment(
  cohort: Any,
  teacher_ids: Sequence[str],
  *,
  phase_policy: str = "uniform",
  task_id: str | None = None,
  num_envs: int | None = None,
  device: str = "cpu",
  render_mode: str | None = None,
  seed: int | None = None,
  reset_policy: ResetPolicy | None = None,
  coverage_policy: CoveragePolicy | None = None,
  coverage_parity_evidence: str | None = None,
  motion_ids: Mapping[str, int] | None = None,
) -> ManagerBasedRlEnv:
  """Build one environment whose rows are pinned to different reference clips.

  This is the opt-in multi-motion sibling of
  :func:`build_distillation_environment`: it copies a registered task
  configuration privately, replaces only its motion command with
  :class:`MultiMotionCommand`, and pins each environment row to a selected
  clip.  Only the registry task factory is executable; saved environment YAML
  remains validation data.

  ``phase_policy`` is the private phase-sampling override recorded alongside
  the delay-timing overrides (``start`` for pinned evaluation, ``uniform`` for
  training).  ``seed`` seeds the environment and the row permutation of the
  stratified slot allocation.  ``motion_ids`` gives each selected clip its
  cohort-wide id; unset keeps them positional, which is a build that owns the
  whole selection.
  """
  # Imported here because the multi-motion module builds on this module's
  # SegmentMotionCommand; module-level import would be circular.
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg
  from mjlab.tasks.tracking.distillation.multi_motion import (
    MultiMotionCommandCfg,
    PhasePolicy,
    make_multi_motion_cfg,
    plan_multi_motion,
  )

  if reset_policy is None:
    reset_policy = ResetPolicy()
  if not isinstance(reset_policy, ResetPolicy):
    raise ValueError("reset_policy must be a validated ResetPolicy")
  if phase_policy not in ("uniform", "start"):
    raise ValueError(f"phase_policy must be 'uniform' or 'start', got {phase_policy!r}")
  selected_task = task_id or cohort.manifest.base_task
  if selected_task is None:
    raise ValueError("cohort has no base_task; task_id is required")
  cfg = load_env_cfg(selected_task)
  validated_seed = _validate_factory_seed(seed)
  if validated_seed is not None:
    cfg.seed = validated_seed
  if num_envs is not None:
    if num_envs <= 0:
      raise ValueError("num_envs must be positive")
    cfg.scene.num_envs = num_envs
  motion_cfg = cfg.commands.get("motion")
  if not isinstance(motion_cfg, MotionCommandCfg):
    raise ValueError(f"registered task {selected_task!r} has no motion command")
  if isinstance(motion_cfg, MultiMotionCommandCfg):
    raise ValueError("registered task already declares a multi-motion command")
  standing_override = motion_cfg.standing_start_prob
  if standing_override != 0.0:
    motion_cfg.standing_start_prob = 0.0
  # The cohort's preview is not the base task's business.  The saved teacher configs fix
  # one ``lookahead_s`` for every member and it already drives the student reference
  # layout, so the live command must produce that same sample.  Otherwise a base task
  # that declares no preview builds a narrower actor observation than the teacher export
  # expects, and the two halves of the lane would disagree about the same cohort.
  lookahead_from_base = motion_cfg.lookahead_s
  if lookahead_from_base != cohort.lookahead_s:
    motion_cfg.lookahead_s = cohort.lookahead_s

  slot_generator = (
    None if validated_seed is None else torch.Generator().manual_seed(validated_seed)
  )
  plan = plan_multi_motion(
    cohort,
    teacher_ids,
    int(cfg.scene.num_envs),
    phase_policy=cast(PhasePolicy, phase_policy),
    slot_generator=slot_generator,
    device=device,
    motion_ids=motion_ids,
  )
  first_teacher = cohort.teacher(plan.teacher_ids[0])
  semantic_overrides = list(
    _apply_saved_observation_timing(
      cfg, first_teacher.env_config, str(first_teacher.entry.env_config)
    )
  )
  if motion_cfg.sampling_mode != plan.phase_policy:
    semantic_overrides.append(
      "private multi-motion config: commands.motion.sampling_mode: "
      f"{motion_cfg.sampling_mode!r} -> {plan.phase_policy!r}"
    )
  if standing_override != 0.0:
    semantic_overrides.append(
      "private reset policy override: commands.motion.standing_start_prob: "
      f"{standing_override!r} -> 0.0"
    )
  if lookahead_from_base != cohort.lookahead_s:
    semantic_overrides.append(
      "private preview override: commands.motion.lookahead_s: "
      f"{lookahead_from_base!r} -> {cohort.lookahead_s!r}"
    )
  cfg.commands["motion"] = make_multi_motion_cfg(
    motion_cfg, plan, reset_policy=reset_policy
  )
  coverage_enabled = coverage_policy is not None and coverage_policy.enabled

  if coverage_enabled:
    # Opt-in only: a run with no policy keeps its exact recorder set.  The term
    # forwards the pre-reset ids while the reached state is still live.
    from mjlab.managers.recorder_manager import RecorderTermCfg
    from mjlab.tasks.tracking.distillation.coverage_collector import (
      CoveragePreResetRecorder,
    )

    cfg.recorders["coverage_pre_reset"] = RecorderTermCfg(func=CoveragePreResetRecorder)
  env = ManagerBasedRlEnv(cfg, device=device, render_mode=render_mode)
  if coverage_enabled:
    # The collector owns the environment, so it is built once the env exists.
    from mjlab.tasks.tracking.distillation.coverage_collector import (
      CoverageStateOwner,
      attach_coverage_collector,
      build_coverage_collector,
    )

    # The component must see the object that carries the cached-view handler and the
    # capture/restore fields, which is the fixed-topology state adapter, not the raw
    # environment.  BranchState.capture uses it directly for the same reason.
    coverage_owner = CoverageStateOwner(env)
    collector = build_coverage_collector(coverage_owner, coverage_policy)
    if collector is None:
      raise ValueError("an enabled coverage policy must produce a collector")
    if coverage_parity_evidence is None:
      raise ValueError(
        "an enabled coverage policy needs restore-parity evidence before construction"
      )
    collector.authorize_live(coverage_parity_evidence)
    attach_coverage_collector(env, collector)
  env.cfg.__dict__["_distillation_semantic_overrides"] = tuple(semantic_overrides)
  env.cfg.__dict__["_distillation_reset_policy"] = reset_policy
  env.cfg.__dict__["_distillation_seed_provenance"] = RuntimeSeedProvenance(
    requested_seed=validated_seed,
    effective_seed=env.cfg.seed,
    applied_before_construction=validated_seed is not None,
  )
  env.cfg.__dict__["_distillation_motion_slots"] = plan.slots
  return env


__all__ = [
  "SegmentMotionCommand",
  "SegmentMotionCommandCfg",
  "build_distillation_environment",
  "build_multi_motion_environment",
  "make_segment_motion_cfg",
]
