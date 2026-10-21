"""D4 opt-in sparse task conditioning on the frozen D3 continuation seam.

This module wires the two finished D4 components into the existing D3 path — it
does not re-implement either one:

* sparse **hard keyframes** (``inpainting.inpaint_keyframes``) rewrite the plan
  the seam already built from measured history and the current measurement;
* sparse **soft pose goals** (``pose_costs.PoseCost`` +
  ``guidance.sample_trajectory_guided``) replace the sampler for a tick when a
  positive guidance strength and at least one active goal exist.

A :class:`TaskConditioning` is the only object the wiring needs.  It owns the
validated :class:`SparseTaskSpec`, a measured-observation ledger for *realized*
target errors, and the bounded per-tick task record that
:class:`~.handover.ReferenceFreeContinuation` attaches to every
:class:`~.handover.ContinuationStep`.

Deliberately absent, with the boundaries owned elsewhere:

* no history mutation: keyframes and goals are task inputs and never replace the
  measured past/current conditioning rows, and no encoder/reference call is
  introduced after handover;
* no latent target and no encoder latent: only the 135 physical-state channels
  of a keyframe row are fixed, and its 32 latent entries stay unknown;
* no reset, standing-policy or fall rescue, no reward/action clipping and no
  model, sampler-default, projection or trainer change;
* no physical or model-quality claim.  A realized target error is measured
  against a trusted measured observation at the target tick, or is reported as
  unreached/missing/expired — never as a predicted value or a trivially zero
  hard-inpainting residual.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from torch import Tensor

from .contract import DEFAULT_CONTRACT, DiffusionContract
from .guidance import GuidanceConfig, sample_trajectory_guided
from .handover import (
  ContinuationPlan,
  FamilySampler,
  ReferenceFreeContinuation,
  SamplerCallable,
)
from .inpainting import (
  TimedStateKeyframe,
  active_keyframe_targets,
  inpaint_keyframes,
)
from .pose_costs import (
  ActivePoseGoal,
  PoseCost,
  PoseCostWeights,
  TimedPoseGoal,
  active_pose_goals,
)
from .projection import ProjectionBundle
from .sampler import SamplerConfig, sample_trajectory
from .schedule import DiffusionSchedule
from .state import WorldState, _quat_matrix

TASK_SCHEMA = "mjlab-d4-sparse-task-v1"

MODES = ("unguided", "keyframe", "poseguided", "combined")

# A task lane must not turn the report into an unbounded log: the per-tick task
# record, the run summary and the source verification detail are all bounded.
_MAX_TICK_RECORD_BYTES = 8192
_MAX_SUMMARY_BYTES = 16384
_MAX_SOURCE_DETAIL_BYTES = 4096

# Unit tags keep the two physical cost families separate: position terms are
# squared metres and the rotation term is the dimensionless chordal distance
# used by ``pose_costs.PoseCost``.
COST_UNIT_TAGS = {
  "root_position": "m^2",
  "body_positions": "m^2",
  "root_orientation": "dimensionless_chordal",
  "weighted_total": "seconds*(m^2 or dimensionless)",
}

DEFAULT_TICK_ORIGIN = "absolute_control_tick_including_bootstrap"
DEFAULT_FRAME = "world"


class TaskConditioningError(ValueError):
  """A task specification or a task-conditioned run state is invalid."""


@dataclass(frozen=True, slots=True)
class TaskSource:
  """Declared origin of a task specification.

  ``qualification`` is the *declared* claim.  ``verification`` records which
  checks were actually performed: ``preparation_tool`` means a preparation tool
  re-read the recording and verified its rows, ``loader_structural_only`` means
  only the task file itself was checked, and an unverified manual/synthetic
  source can never carry a verified physical qualification.
  """

  kind: str
  qualification: str
  provenance: str
  verification: str
  path: str | None = None
  format: str | None = None
  sha256: str | None = None
  summary_path: str | None = None
  summary_sha256: str | None = None
  member: str | None = None
  seed: int | None = None
  start_frame: int | None = None
  verification_detail: Mapping[str, Any] | None = None

  def __post_init__(self) -> None:
    if self.kind not in {"baseline_recording", "manual", "synthetic_fixture"}:
      raise TaskConditioningError(f"unsupported task source kind {self.kind!r}")
    if self.qualification not in {"verified", "declared", "unverified"}:
      raise TaskConditioningError(
        f"unsupported task source qualification {self.qualification!r}"
      )
    if not self.provenance.strip():
      raise TaskConditioningError("task source provenance must be non-empty")
    if self.kind != "baseline_recording" and self.qualification != "unverified":
      raise TaskConditioningError(
        "a manual or synthetic task source is diagnostic input and can only be "
        "declared unverified; it must never claim qualified physical provenance"
      )
    if self.kind == "baseline_recording" and self.qualification != "unverified":
      if not self.path or not self.sha256:
        raise TaskConditioningError(
          "a recording-bound task source requires its path and sha256"
        )
    if self.qualification == "verified" and not self.verification.startswith(
      ("preparation_tool", "loader_recheck")
    ):
      raise TaskConditioningError(
        "only a preparation tool or the strict loader can mark a source "
        "verified; a declared qualification must be 'declared', never 'verified'"
      )
    if self.verification_detail is not None:
      detail = dict(self.verification_detail)
      size = len(json.dumps(detail, sort_keys=True, default=str).encode())
      if size > _MAX_SOURCE_DETAIL_BYTES:
        raise TaskConditioningError(
          f"task source verification detail {size}B exceeds the "
          f"{_MAX_SOURCE_DETAIL_BYTES}B bound"
        )
      object.__setattr__(self, "verification_detail", detail)

  def as_dict(self) -> dict[str, Any]:
    return {
      "kind": self.kind,
      "qualification": self.qualification,
      "verification": self.verification,
      "verification_detail": None
      if self.verification_detail is None
      else dict(self.verification_detail),
      "provenance": self.provenance,
      "path": self.path,
      "format": self.format,
      "sha256": self.sha256,
      "summary_path": self.summary_path,
      "summary_sha256": self.summary_sha256,
      "member": self.member,
      "seed": self.seed,
      "start_frame": self.start_frame,
    }


@dataclass(frozen=True, slots=True)
class SparseTaskSpec:
  """One validated, versioned sparse-task input for the conditioned run."""

  mode: str
  keyframes: tuple[TimedStateKeyframe, ...]
  goals: tuple[TimedPoseGoal, ...]
  weights: PoseCostWeights | None
  strength: float
  source: TaskSource
  config_sha256: str
  schema_version: str = TASK_SCHEMA

  def __post_init__(self) -> None:
    if self.mode not in MODES:
      raise TaskConditioningError(f"unsupported task mode {self.mode!r}")
    if self.schema_version != TASK_SCHEMA:
      raise TaskConditioningError(f"unsupported task schema {self.schema_version!r}")
    if len(self.config_sha256) != 64:
      raise TaskConditioningError("task config sha256 must be a lowercase hex digest")
    strength = float(self.strength)
    if not math.isfinite(strength) or strength < 0.0:
      raise TaskConditioningError("guidance strength must be finite and non-negative")
    object.__setattr__(self, "strength", strength)
    if bool(self.keyframes) and bool(self.goals):
      if self.mode not in {"keyframe", "poseguided", "combined"}:
        raise TaskConditioningError(
          "sparse targets require mode keyframe, poseguided or combined"
        )
    if self.mode == "unguided":
      if self.keyframes or self.goals or self.weights is not None:
        raise TaskConditioningError(
          "the unguided mode must not carry keyframes, goals or weights"
        )
      if strength != 0.0:
        raise TaskConditioningError("the unguided mode must not carry guidance")
    if self.mode in {"poseguided", "combined"}:
      if not self.goals:
        raise TaskConditioningError(f"mode {self.mode!r} requires pose goals")
      if self.weights is None:
        raise TaskConditioningError(
          f"mode {self.mode!r} requires explicit pose cost weights"
        )
      if strength <= 0.0:
        raise TaskConditioningError(
          f"mode {self.mode!r} requires a positive guidance strength; a zero "
          "strength would be an unnamed unguided run"
        )
    else:
      if self.weights is not None or strength != 0.0:
        raise TaskConditioningError(f"mode {self.mode!r} does not accept guidance")
    if self.mode in {"keyframe", "combined"} and not self.keyframes:
      raise TaskConditioningError(f"mode {self.mode!r} requires hard keyframes")
    # Each mode owns its own target kind: a hard keyframe scheduled in a soft
    # pose mode (or the reverse) would silently change what the task means.
    if self.mode == "keyframe" and self.goals:
      raise TaskConditioningError(
        "mode 'keyframe' schedules hard keyframes only; move the soft pose "
        "goals to a poseguided or combined task"
      )
    if self.mode == "poseguided" and self.keyframes:
      raise TaskConditioningError(
        "mode 'poseguided' schedules soft pose goals only; move the hard "
        "keyframes to a keyframe or combined task"
      )
    # A hard keyframe and a soft goal on the same tick cannot both act: the hard
    # row makes the soft cost constant with zero gradient, so only the exact
    # collision is refused.  Nearby targets on different rows stay allowed.
    hard_ticks = {keyframe.tick for keyframe in self.keyframes}
    colliding = sorted(hard_ticks & {goal.tick for goal in self.goals})
    if colliding:
      raise TaskConditioningError(
        f"hard keyframes and soft goals collide on ticks {colliding}; the hard "
        "row would make the soft cost a constant with zero gradient"
      )

  @property
  def keyframe_ticks(self) -> tuple[int, ...]:
    return tuple(keyframe.tick for keyframe in self.keyframes)

  @property
  def goal_ticks(self) -> tuple[int, ...]:
    return tuple(goal.tick for goal in self.goals)


@dataclass(slots=True)
class _TickTaskState:
  """The bounded per-tick task record accumulated by the two hooks."""

  tick: int
  active_keyframe_ticks: list[int] = field(default_factory=list)
  active_keyframe_rows: list[int] = field(default_factory=list)
  active_goal_ids: list[int] = field(default_factory=list)
  inactive_goal_ids: list[int] = field(default_factory=list)
  fixed_goal_ids: list[int] = field(default_factory=list)
  guided: bool = False
  denoiser_calls: int = 0
  vjp_calls: int = 0
  cost_calls: int = 0
  cost_max: float = 0.0
  gradient_norm_max: float = 0.0
  clipped_gradient_norm_max: float = 0.0
  guidance_norm_max: float = 0.0
  steps: tuple[dict[str, float], ...] = ()
  # True when the row immediately after this tick's current row is hard-fixed by
  # a keyframe, so the sampler's next-row value is that constraint rather than a
  # learned forecast.
  next_row_hard: bool = False

  def as_dict(self, *, mode: str, strength: float) -> dict[str, Any]:
    return {
      "mode": mode,
      "strength": float(strength),
      "active_keyframes": [
        {"tick": tick, "row": row}
        for tick, row in zip(
          self.active_keyframe_ticks, self.active_keyframe_rows, strict=True
        )
      ],
      "active_goal_ids": list(self.active_goal_ids),
      "inactive_goal_ids": list(self.inactive_goal_ids),
      "fixed_goal_ids": list(self.fixed_goal_ids),
      "guided": self.guided,
      "next_row_hard": self.next_row_hard,
      "denoiser_calls": self.denoiser_calls,
      "vjp_calls": self.vjp_calls,
      "cost_calls": self.cost_calls,
      "cost_max": self.cost_max,
      "gradient_norm_max": self.gradient_norm_max,
      "clipped_gradient_norm_max": self.clipped_gradient_norm_max,
      "guidance_norm_max": self.guidance_norm_max,
      "steps": list(self.steps),
    }


@dataclass(frozen=True, slots=True)
class _TaskDiagnostics:
  """Sampler diagnostics plus the bounded task record of the same tick."""

  inner: Any
  task: Mapping[str, Any]

  def __getattr__(self, name: str) -> Any:
    return getattr(self.inner, name)


def _empty_summary() -> dict[str, float]:
  return {
    "denoiser_calls": 0,
    "vjp_calls": 0,
    "cost_calls": 0,
    "cost_max": 0.0,
    "gradient_norm_max": 0.0,
    "clipped_gradient_norm_max": 0.0,
    "guidance_norm_max": 0.0,
  }


def _json_bytes(value: Any) -> int:
  return len(json.dumps(value, sort_keys=True, default=str).encode())


class TaskConditioning:
  """Bounded task state for one conditioned reference-free continuation.

  The hooks are installed only when the task actually needs them, so an
  unguided spec leaves :class:`ReferenceFreeContinuation` on its D3 call path.
  ``plan_transform`` is called with every measured (plan, current state) pair,
  which is where realized target errors are measured and where hard keyframes
  are fixed; ``sampler_factory`` returns the guided (or delegated) sampler for
  that same tick; ``observe_measured_post`` receives trusted post-step
  measurements (including the final budget post) that never become a plan; and
  ``reset`` clears the whole task ledger so nothing leaks across a handover
  reset.
  """

  def __init__(
    self,
    spec: SparseTaskSpec,
    *,
    projection: ProjectionBundle,
    contract: DiffusionContract = DEFAULT_CONTRACT,
    delegate_sampler: FamilySampler | None = None,
  ) -> None:
    if not isinstance(spec, SparseTaskSpec):
      raise TaskConditioningError("task conditioning requires a SparseTaskSpec")
    if delegate_sampler is not None:
      if not callable(delegate_sampler):
        raise TaskConditioningError("delegate_sampler must be callable or None")
      # Soft pose goals need the DDIM guided sampler (PoseCost over a diffusion
      # schedule).  A family run supports hard keyframes only, so refuse the
      # combination here instead of silently sampling without the goals.
      if spec.goals or spec.strength > 0.0:
        raise TaskConditioningError(
          "soft pose goals need the guided DDIM sampler; a family sampler run "
          "supports hard keyframes only"
        )
    self.spec = spec
    self.projection = projection
    self.contract = contract
    self._delegate_sampler = delegate_sampler
    self._pending: dict[int, _TickTaskState] = {}
    self._plan_ticks: list[int] = []
    self._observed: dict[int, dict[str, Any]] = {}
    self._guided_steps = 0
    self._keyframe_steps = 0
    self._summary = _empty_summary()
    self._errors: list[str] = []

  # -- wiring ---------------------------------------------------------------

  @property
  def active(self) -> bool:
    """Whether any task hook is needed at all."""
    return bool(self.spec.keyframes) or bool(self.spec.goals)

  def reset(self) -> None:
    """Drop every task ledger entry before a fresh handover bootstrap.

    Called from :meth:`~.handover.ReferenceFreeContinuation.reset`.  A reset run
    must rebuild its eight measured pairs; no observation, realized target,
    pending tick record, counter or error from the previous episode survives,
    so a reused absolute tick is measured fresh instead of inheriting an old
    result.
    """
    self._pending.clear()
    self._plan_ticks.clear()
    self._observed.clear()
    self._guided_steps = 0
    self._keyframe_steps = 0
    self._summary = _empty_summary()
    self._errors.clear()

  def plan_transform(
    self, plan: ContinuationPlan, current_state: WorldState
  ) -> ContinuationPlan:
    """Measure realized targets at this tick, then fix active hard keyframes."""
    if not isinstance(plan, ContinuationPlan):
      raise TaskConditioningError("plan transform requires a ContinuationPlan")
    tick = int(plan.tick)
    self._plan_ticks.append(tick)
    self._observe(tick, current_state)
    state = _TickTaskState(tick=tick)
    if self.spec.keyframes:
      targets = active_keyframe_targets(
        plan, self.spec.keyframes, contract=self.contract
      )
      state.active_keyframe_ticks = [target.tick for target in targets]
      state.active_keyframe_rows = [target.row for target in targets]
      state.next_row_hard = any(target.offset == 1 for target in targets)
      if targets:
        self._keyframe_steps += 1
    if self.spec.goals:
      active = active_pose_goals(plan.tick, self.spec.goals, contract=self.contract)
      self._fill_goal_ids(state, active)
    self._pending[tick] = state
    if not self.spec.keyframes:
      return plan
    return inpaint_keyframes(
      plan,
      current_state,
      self.spec.keyframes,
      projection=self.projection,
      contract=self.contract,
    )

  def observe_measured_post(self, tick: int, state: WorldState) -> None:
    """Score targets at one trusted measured post-step tick.

    The evaluation runner calls this only for a post-step observation that is
    finite, reset/teleport free, contiguous in segment/tick/control period and
    reference free.  It never samples, decodes, records history or advances the
    planning count: it exists so the final budget post and a trustworthy fall
    post can be measured even when no further plan is built.
    """
    self._observe(int(tick), state)

  def sampler_factory(
    self, plan: ContinuationPlan, current_state: WorldState
  ) -> SamplerCallable:
    """Return the per-tick sampler: guided when it can act, D3 otherwise."""
    state = self._pending.get(int(plan.tick))
    if state is None:
      state = _TickTaskState(tick=int(plan.tick))
      self._pending[int(plan.tick)] = state
    if self.spec.goals:
      active = active_pose_goals(plan.tick, self.spec.goals, contract=self.contract)
      self._fill_goal_ids(state, active)
    if not self.spec.goals or self.spec.strength <= 0.0:
      return self._unguided_sampler(state)
    cost = PoseCost(
      projection=self.projection,
      current_state=current_state,
      plan_tick=plan.tick,
      goals=self.spec.goals,
      weights=self._weights,
      contract=self.contract,
    )
    guidance = GuidanceConfig(strength=self.spec.strength)
    record = state.as_dict(mode=self.spec.mode, strength=self.spec.strength)

    def sampler(
      denoiser: Any,
      *,
      schedule: DiffusionSchedule,
      conditions: Any,
      config: SamplerConfig,
      generator: Any,
    ) -> tuple[Tensor, Any]:
      sample, diagnostics = sample_trajectory_guided(
        denoiser,
        schedule=schedule,
        conditions=conditions,
        config=config,
        generator=generator,
        guidance=guidance,
        cost=cost,
      )
      self._absorb_diagnostics(state, diagnostics)
      self._guided_steps += 1 if state.guided else 0
      record.update(state.as_dict(mode=self.spec.mode, strength=self.spec.strength))
      self._validate_tick_record(record)
      return sample, _TaskDiagnostics(diagnostics, record)

    return sampler

  def _unguided_sampler(self, state: _TickTaskState) -> SamplerCallable:
    """Wrap the frozen unguided sampler only to expose the task record."""
    record = state.as_dict(mode=self.spec.mode, strength=self.spec.strength)
    delegate = self._delegate_sampler
    if delegate is not None:

      def family_sampler(conditions: Any, generator: Any) -> tuple[Tensor, Any]:
        sample, diagnostics = delegate(conditions, generator)
        state.denoiser_calls = int(diagnostics.denoiser_calls)
        self._summary["denoiser_calls"] += state.denoiser_calls
        record.update(state.as_dict(mode=self.spec.mode, strength=self.spec.strength))
        self._validate_tick_record(record)
        return sample, _TaskDiagnostics(diagnostics, record)

      return family_sampler

    def sampler(
      denoiser: Any,
      *,
      schedule: DiffusionSchedule,
      conditions: Any,
      config: SamplerConfig,
      generator: Any,
    ) -> tuple[Tensor, Any]:
      sample, diagnostics = sample_trajectory(
        denoiser,
        schedule=schedule,
        conditions=conditions,
        config=config,
        generator=generator,
      )
      state.denoiser_calls = int(diagnostics.denoiser_calls)
      self._summary["denoiser_calls"] += state.denoiser_calls
      record.update(state.as_dict(mode=self.spec.mode, strength=self.spec.strength))
      self._validate_tick_record(record)
      return sample, _TaskDiagnostics(diagnostics, record)

    return sampler

  @property
  def _weights(self) -> PoseCostWeights:
    weights = self.spec.weights
    if weights is None:  # pragma: no cover - guarded by SparseTaskSpec
      raise TaskConditioningError("pose goals require explicit cost weights")
    return weights

  def _fill_goal_ids(
    self, state: _TickTaskState, active: tuple[ActivePoseGoal, ...]
  ) -> None:
    active_ticks = {goal.tick for goal in active}
    state.active_goal_ids = [goal.tick for goal in active]
    state.inactive_goal_ids = [
      goal.tick for goal in self.spec.goals if goal.tick not in active_ticks
    ]

  def _absorb_diagnostics(self, state: _TickTaskState, diagnostics: Any) -> None:
    state.guided = bool(getattr(diagnostics, "guided", False))
    state.denoiser_calls = int(getattr(diagnostics, "denoiser_calls", 0))
    state.vjp_calls = int(getattr(diagnostics, "vjp_calls", 0))
    state.cost_calls = int(getattr(diagnostics, "cost_calls", 0))
    state.fixed_goal_ids = list(getattr(diagnostics, "fixed_goal_ids", ()))
    steps: tuple[Any, ...] = tuple(getattr(diagnostics, "steps", ()))
    state.steps = tuple(
      {
        "source_id": int(step.source_id),
        "cost": float(step.cost),
        "gradient_norm": float(step.gradient_norm),
        "clipped_gradient_norm": float(step.clipped_gradient_norm),
        "guidance_norm": float(step.guidance_norm),
      }
      for step in steps
    )
    if steps:
      state.cost_max = max(float(step.cost) for step in steps)
      state.gradient_norm_max = max(float(step.gradient_norm) for step in steps)
      state.clipped_gradient_norm_max = max(
        float(step.clipped_gradient_norm) for step in steps
      )
      state.guidance_norm_max = max(float(step.guidance_norm) for step in steps)
    self._summary["denoiser_calls"] += state.denoiser_calls
    self._summary["vjp_calls"] += state.vjp_calls
    self._summary["cost_calls"] += state.cost_calls
    for key in (
      "cost_max",
      "gradient_norm_max",
      "clipped_gradient_norm_max",
      "guidance_norm_max",
    ):
      self._summary[key] = max(self._summary[key], getattr(state, key))

  @staticmethod
  def _validate_tick_record(record: Mapping[str, Any]) -> None:
    size = _json_bytes(record)
    if size > _MAX_TICK_RECORD_BYTES:
      raise TaskConditioningError(
        f"per-tick task record {size}B exceeds the {_MAX_TICK_RECORD_BYTES}B bound"
      )

  # -- realized target errors ----------------------------------------------

  def _observe(self, tick: int, current_state: WorldState) -> None:
    """Score every scheduled target whose absolute tick is this measured tick.

    Hard keyframes and soft pose goals are scored by the same physical metric
    but stay separate targets with separate kinds and indexes.  Observations are
    deduplicated by absolute tick (the same tick can arrive once as a plan and
    once as the previous tick's trusted post); the first trustworthy
    measurement wins and no counter changes afterwards.
    """
    if not isinstance(current_state, WorldState):
      raise TaskConditioningError("measured state must be a complete WorldState")
    if current_state.root_position.ndim != 1:
      raise TaskConditioningError("measured state must be one unbatched measurement")
    tick = int(tick)
    if tick in self._observed:
      return
    entry: dict[str, Any] = {"tick": tick, "keyframe": None, "goal": None}
    for kind, targets in (
      ("keyframe", self.spec.keyframes),
      ("goal", self.spec.goals),
    ):
      for index, target in enumerate(targets):
        if target.tick != tick:
          continue
        entry[kind] = {
          "target_kind": kind,
          "target_index": index,
          "tick": tick,
          "observed_tick": tick,
          **realized_pose_error(current_state, target),
        }
    self._observed[tick] = entry

  def realized_errors(self) -> dict[str, Any]:
    """Return the realized/unreached/expired ledger for the run so far.

    Only a trusted measured observation at the target's own tick is scored, and
    hard keyframes and soft goals are reported separately through
    ``target_kind``.  Every other scheduled target is reported by name, never as
    zero error, never as the predicted value and never as the trivially zero
    hard-inpainting residual.
    """
    observed_ticks = sorted(self._observed)
    first = observed_ticks[0] if observed_ticks else None
    last = observed_ticks[-1] if observed_ticks else None
    scored: list[dict[str, Any]] = []
    unreached: list[int] = []
    missing: list[int] = []
    expired: list[int] = []
    for kind, targets in (
      ("keyframe", self.spec.keyframes),
      ("goal", self.spec.goals),
    ):
      for index, target in enumerate(targets):
        entry = self._observed.get(target.tick)
        measured = None if entry is None else entry[kind]
        if measured is not None and measured["target_index"] == index:
          scored.append(dict(measured))
        elif first is None or last is None:
          missing.append(target.tick)
        elif target.tick < first:
          expired.append(target.tick)
        elif target.tick > last:
          unreached.append(target.tick)
        else:
          missing.append(target.tick)
    return {
      "scored": scored,
      "unreached_ticks": sorted(set(unreached)),
      "missing_ticks": sorted(set(missing)),
      "expired_ticks": sorted(set(expired)),
      "observed_ticks": observed_ticks,
      "plan_ticks": sorted(set(self._plan_ticks)),
      "counts": {
        "scored_keyframes": sum(
          1 for item in scored if item["target_kind"] == "keyframe"
        ),
        "scored_goals": sum(1 for item in scored if item["target_kind"] == "goal"),
        "unreached_targets": len(set(unreached)),
        "missing_targets": len(set(missing)),
        "expired_targets": len(set(expired)),
        "measured_observations": len(observed_ticks),
        "plan_steps": len(self._plan_ticks),
      },
      "scoring_rule": (
        "realized physical error at the target tick from a trusted measured "
        "observation only; hard keyframes and soft goals are separate target "
        "kinds; unreached/missing/expired targets are not scored"
      ),
    }

  # -- reporting -----------------------------------------------------------

  def close(self, *, stop_reason: str | None = None) -> None:
    """Record why the run ended (task targets are never assumed reached)."""
    if stop_reason is not None:
      self._errors.append(str(stop_reason))

  def provenance(
    self, *, source_fingerprint: Mapping[str, Any] | None = None
  ) -> dict[str, Any]:
    """Return the bounded task metadata recorded in the report and marker."""
    realized = self.realized_errors()
    summary: dict[str, Any] = {
      "schema": self.spec.schema_version,
      "mode": self.spec.mode,
      "config_sha256": self.spec.config_sha256,
      "contract_identity": self.contract.identity_hash(),
      "projection_identity": None
      if getattr(self.projection, "contract", None) is None
      else self.projection.contract.identity_hash(),
      "source": self.spec.source.as_dict(),
      "scheduled": {
        "keyframe_ticks": list(self.spec.keyframe_ticks),
        "goal_ticks": list(self.spec.goal_ticks),
        "target_count": len(self.spec.keyframes) + len(self.spec.goals),
      },
      "guidance_strength": self.spec.strength,
      "weights": None
      if self.spec.weights is None
      else {
        "root_position": self.spec.weights.root_position,
        "root_orientation": self.spec.weights.root_orientation,
        "body_positions": self.spec.weights.body_positions,
        "body_coefficients": list(self.spec.weights.coefficients),
      },
      "unit_tags": dict(COST_UNIT_TAGS),
      "executed": {
        **self._summary,
        "guided_steps": self._guided_steps,
        "keyframe_steps": self._keyframe_steps,
        # Planning ticks and trusted measured observations are separate counts:
        # the final budget/fall post is measured without ever becoming a plan.
        "plan_steps": len(self._plan_ticks),
        "measured_observations": len(self._observed),
      },
      "realized": realized,
      "outcome": {
        "stop_reasons": list(self._errors),
        "scored_targets": len(realized["scored"]),
        "unreached_targets": len(realized["unreached_ticks"]),
        "missing_targets": len(realized["missing_ticks"]),
        "expired_targets": len(realized["expired_ticks"]),
      },
      "claims": {
        "physical_provenance": self.spec.source.qualification,
        "realized_error_source": "trusted measured observation at the target tick",
        "model_quality": "not claimed by task conditioning",
      },
    }
    if source_fingerprint is not None:
      summary["source_fingerprint"] = dict(source_fingerprint)
    size = _json_bytes(summary)
    if size > _MAX_SUMMARY_BYTES:
      raise TaskConditioningError(
        f"task summary {size}B exceeds the {_MAX_SUMMARY_BYTES}B bound"
      )
    return summary


def realized_pose_error(
  measured: WorldState, target: TimedStateKeyframe | TimedPoseGoal
) -> dict[str, float]:
  """Return the physical distance between a measured state and a task target.

  Root position and per-body terms are metres (RMS over the frozen body set),
  the rotation term is the dimensionless chordal matrix distance
  ``||R_measured - R_target||_F`` used by :class:`~.pose_costs.PoseCost`.  Both
  hard keyframes and soft pose goals are complete world states in the frozen
  20-body order and are measured by the same physical metric; no prediction, no
  target-row substitution and no trivially zero hard-inpainting residual is
  used.
  """
  if not isinstance(measured, WorldState):
    raise TaskConditioningError("realized error requires a measured WorldState")
  if not isinstance(target, (TimedStateKeyframe, TimedPoseGoal)):
    raise TaskConditioningError(
      "realized error requires a TimedStateKeyframe or TimedPoseGoal target"
    )
  position = measured.root_position - target.state.root_position
  bodies = np.asarray(measured.body_positions, dtype=np.float64).reshape(-1, 3)
  target_bodies = np.asarray(target.state.body_positions, dtype=np.float64).reshape(
    -1, 3
  )
  if bodies.shape != target_bodies.shape:
    raise TaskConditioningError("measured and target body positions must match")
  body_distances = np.linalg.norm(bodies - target_bodies, axis=-1)
  measured_rotation = _quat_matrix(
    np.asarray(measured.root_quaternion_wxyz, dtype=np.float64)[None, :]
  )[0]
  target_rotation = _quat_matrix(
    np.asarray(target.state.root_quaternion_wxyz, dtype=np.float64)[None, :]
  )[0]
  return {
    "root_position_m": float(np.linalg.norm(position)),
    "root_orientation_chordal": float(
      np.linalg.norm(measured_rotation - target_rotation)
    ),
    "body_positions_rms_m": float(np.sqrt(np.mean(body_distances**2))),
    "body_positions_max_m": float(np.max(body_distances)),
  }


def build_conditioned_continuation(
  *,
  conditioning: TaskConditioning | None,
  **continuation_kwargs: Any,
) -> ReferenceFreeContinuation:
  """Build the D3 continuation with the optional task hooks installed.

  With ``conditioning`` ``None`` or an inactive spec the returned object is the
  unchanged D3 continuation: no plan transform, no sampler replacement, no
  reset observer and no measured-post observer.
  """
  if conditioning is not None and not isinstance(conditioning, TaskConditioning):
    raise TaskConditioningError("conditioning must be a TaskConditioning or None")
  if conditioning is not None and conditioning.active:
    reserved = {"plan_transform", "sampler_factory", "on_reset", "measured_post"}
    if reserved & set(continuation_kwargs):
      raise TaskConditioningError(
        f"task hooks {sorted(reserved)} are installed by the conditioning "
        "object; pass only the D3 continuation arguments"
      )
    continuation_kwargs["plan_transform"] = conditioning.plan_transform
    continuation_kwargs["sampler_factory"] = conditioning.sampler_factory
    continuation_kwargs["on_reset"] = conditioning.reset
    continuation_kwargs["measured_post"] = conditioning.observe_measured_post
  return ReferenceFreeContinuation(**continuation_kwargs)


__all__ = [
  "COST_UNIT_TAGS",
  "DEFAULT_FRAME",
  "DEFAULT_TICK_ORIGIN",
  "MODES",
  "SparseTaskSpec",
  "TASK_SCHEMA",
  "TaskConditioning",
  "TaskConditioningError",
  "TaskSource",
  "build_conditioned_continuation",
  "realized_pose_error",
]
