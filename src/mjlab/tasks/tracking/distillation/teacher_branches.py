"""Bounded teacher branch lifecycle and actual main-row-step resource accounting."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
import torch

from .branch_state import BranchState, BranchStateError, isolated_branch_rng
from .coverage_metadata import CoverageRowMetadata

BranchOutcome = Literal[
  "recovered", "failed", "completed", "interrupted", "horizon_censored", "invalid"
]


@dataclass(frozen=True, slots=True)
class BranchBudget:
  horizon_control_steps: int = 100
  max_transitions_after_first_failure: int = 10
  floor_dwell_steps: int = 5
  floor_tilt_rad: float = 1.3089969389957472
  floor_height_m: float = 0.30
  branch_slots: int = 32
  max_admitted_seeds: int = 64
  max_starts_per_call: int = 64
  resident_slots: int = 32
  terminal_start_reservation: int = 13
  pre_step_start_reservation: int = 51
  max_main_row_steps_fraction: float = 0.10
  resident_bytes_cap: int = 1 << 30

  def __post_init__(self) -> None:
    if (
      min(
        self.horizon_control_steps,
        self.floor_dwell_steps,
        self.branch_slots,
        self.max_admitted_seeds,
        self.max_starts_per_call,
        self.resident_slots,
      )
      <= 0
    ):
      raise ValueError("branch budgets must be positive")
    if self.max_transitions_after_first_failure < 0:
      raise ValueError("post-failure budget must be nonnegative")
    if (
      self.terminal_start_reservation + self.pre_step_start_reservation
      > self.max_starts_per_call
    ):
      raise ValueError("branch start reservations exceed per-call cap")
    if not 0.0 < self.max_main_row_steps_fraction <= 1.0:
      raise ValueError("main row-step fraction must be in (0, 1]")
    if self.resident_bytes_cap <= 0:
      raise ValueError("resident byte cap must be positive")


@dataclass(frozen=True, slots=True)
class BranchSeed:
  snapshot: BranchState
  metadata: CoverageRowMetadata | None = None
  seed_id: str = ""
  source_family_id: str = ""
  source_kind: str = "student_difficult"
  source_failed: bool = False
  source_failure_reason: str | None = None
  source_timeout: bool = False
  source_reference_complete: bool = False


@dataclass(frozen=True, slots=True)
class BranchLabel:
  observation: Any
  action: Any
  metadata: CoverageRowMetadata | None
  valid: bool = True


@dataclass(frozen=True, slots=True)
class BranchResult:
  labels: tuple[BranchLabel, ...]
  outcome: BranchOutcome
  steps: int
  first_failure_step: int | None
  failure_reason: str | None
  stop_reason: str
  budget_censored: bool = False
  endpoint_observation: Any | None = None
  endpoint_reason: str | None = None


class TeacherBranch:
  def __init__(self, seed: BranchSeed, budget: BranchBudget, branch_id: str) -> None:
    self.seed, self.budget, self.branch_id = seed, budget, branch_id
    self.labels: list[BranchLabel] = []
    self.steps = 0
    self.first_failure_step: int | None = None
    self.failure_reason: str | None = None
    self.floor_steps = 0
    self._outcome: BranchOutcome | None = None
    source_failed = self.seed.source_failed or bool(
      self.seed.metadata is not None and self.seed.metadata.first_failure
    )
    if source_failed:
      self.first_failure_step = 0
      self.failure_reason = self.seed.source_failure_reason or "inherited_failure"
    self.stop_reason: str | None = None
    self.endpoint_observation: Any | None = None
    self.endpoint_reason: str | None = None

  @property
  def active(self) -> bool:
    return self._outcome is None

  def run(
    self,
    backend: Any,
    teacher: Callable[[Any], Any],
    *,
    seed: int = 0,
    metadata_factory: Callable[[int, Any], CoverageRowMetadata | None] | None = None,
  ) -> BranchResult:
    if not self.active:
      raise RuntimeError("branch has already run")
    with isolated_branch_rng(seed):
      self.seed.snapshot.restore(backend)
      self._run_loop(backend, teacher, metadata_factory)
    return self._result(self.failure_reason or self._outcome or "stopped")

  def _label(
    self, backend: Any, teacher: Callable[[Any], Any], metadata_factory: Any
  ) -> bool:
    view = _view(backend)
    if not _finite(view):
      self._outcome = "invalid"
      self.stop_reason = "nonfinite_policy_view"
      return False
    action = teacher(view)
    if not _finite(action):
      self._outcome = "invalid"
      self.stop_reason = "nonfinite_teacher_output"
      return False
    metadata = (
      metadata_factory(self.steps, view) if metadata_factory else self.seed.metadata
    )
    self.labels.append(
      BranchLabel(_owned(view), _owned(action), copy.deepcopy(metadata))
    )
    return True

  def _run_loop(
    self, backend: Any, teacher: Callable[[Any], Any], metadata_factory: Any
  ) -> None:
    while self.active:
      if self.steps >= self.budget.horizon_control_steps:
        view = _view(backend)
        if not _finite(view):
          self._outcome = "invalid"
          self.stop_reason = "nonfinite_endpoint"
          break
        if not self._label(backend, teacher, metadata_factory):
          break
        self.stop_reason = "horizon_censored"
        self._outcome = "horizon_censored"
        break
      if not self._label(backend, teacher, metadata_factory):
        break
      try:
        result = backend.step(self.labels[-1].action)
      except CoverageResourceError as exc:
        self._outcome = "horizon_censored"
        self.failure_reason = str(exc)
        self.stop_reason = "budget_censored"
        break
      self.steps += 1
      if not _finite(result):
        self._outcome = "invalid"
        self.stop_reason = "nonfinite_backend_result"
        break
      failed, reason = _failure(result)
      if failed and self.first_failure_step is None:
        self.first_failure_step, self.failure_reason = (
          self.steps,
          reason or "tracking_failure",
        )
      self.floor_steps = self.floor_steps + 1 if _floor(result, self.budget) else 0
      reference_complete = bool(
        _get(result, "reference_complete", self.seed.source_reference_complete)
      )
      timeout = bool(_get(result, "timeout", self.seed.source_timeout))
      next_reference_available = bool(_get(result, "next_reference_available", True))
      failure_cap = (
        self.first_failure_step is not None
        and self.steps - self.first_failure_step
        >= self.budget.max_transitions_after_first_failure
      )
      endpoint = (
        reference_complete
        or timeout
        or failure_cap
        or self.floor_steps >= self.budget.floor_dwell_steps
      )
      if endpoint:
        endpoint_view_finite = _finite(_view(backend))
        if not endpoint_view_finite:
          self._outcome = "invalid"
          self.stop_reason = "nonfinite_endpoint"
          break
        if next_reference_available:
          if not self._label(backend, teacher, metadata_factory):
            self.stop_reason = self.stop_reason or "nonfinite_endpoint"
            break
        else:
          self.endpoint_observation = _owned(_view(backend))
          self.endpoint_reason = "no_next_reference"
        if self.floor_steps >= self.budget.floor_dwell_steps:
          self.failure_reason = "floor_dwell"
          self.stop_reason = "floor_dwell"
        elif failure_cap:
          self.stop_reason = "failure_continuation_cap"
        elif reference_complete:
          self.stop_reason = "reference_complete"
        elif timeout:
          self.stop_reason = "timeout"
        else:
          self.stop_reason = "interrupted"
        self._outcome = (
          "completed"
          if reference_complete
          else "failed"
          if failed or failure_cap or self.floor_steps
          else "interrupted"
        )
        break
    if self._outcome is None:
      self.stop_reason = "interrupted"
      self._outcome = "interrupted"

  def _result(self, reason: str) -> BranchResult:
    return BranchResult(
      tuple(self.labels),
      self._outcome or "invalid",
      self.steps,
      self.first_failure_step,
      self.failure_reason,
      self.stop_reason or reason,
      self._outcome == "horizon_censored",
      self.endpoint_observation,
      self.endpoint_reason,
    )


class CoverageResourceError(RuntimeError):
  """The bounded actual row-step token budget is exhausted."""


class _BudgetBackend:
  def __init__(
    self, backend: Any, consume: Callable[[int], None], world_cost: int
  ) -> None:
    self._backend, self._consume, self._world_cost = backend, consume, world_cost
    resident = getattr(backend, "resident_bytes", None)
    if not isinstance(resident, int) or resident < 0:
      raise CoverageResourceError("backend must report resident_bytes")

  def __getattr__(self, name: str) -> Any:
    return getattr(self._backend, name)

  def step(self, action: Any) -> Any:
    self._consume(self._world_cost)
    return self._backend.step(action)


class TeacherBranchManager:
  def __init__(self, budget: BranchBudget | None = None) -> None:
    self.budget = budget or BranchBudget()
    self._seeds: list[BranchSeed] = []
    self._next_id = 0
    self.counters: dict[str, int] = {}

  def admit(self, seed: BranchSeed) -> bool:
    if len(self._seeds) >= self.budget.max_admitted_seeds:
      self.counters["seed_cap_dropped"] = self.counters.get("seed_cap_dropped", 0) + 1
      return False
    self._seeds.append(seed)
    return True

  def _select_call_seeds(self) -> list[BranchSeed]:
    """Apply simultaneous terminal/pre-step quotas with deterministic borrowing."""
    terminal = [seed for seed in self._seeds if _seed_is_terminal(seed)]
    pre_step = [seed for seed in self._seeds if not _seed_is_terminal(seed)]
    terminal_take = min(len(terminal), self.budget.terminal_start_reservation)
    pre_take = min(len(pre_step), self.budget.pre_step_start_reservation)
    selected = terminal[:terminal_take] + pre_step[:pre_take]
    remaining = self.budget.max_starts_per_call - len(selected)
    if remaining > 0:
      # Unused reservation capacity is borrowable, but no category receives
      # outcome-based priority.  Preserve queue order for the deterministic
      # remainder after each category's reserved round.
      selected_ids = {id(seed) for seed in selected}
      for seed in self._seeds:
        if id(seed) in selected_ids:
          continue
        selected.append(seed)
        selected_ids.add(id(seed))
        remaining -= 1
        if remaining == 0:
          break
    used_terminal = sum(_seed_is_terminal(seed) for seed in selected)
    used_pre = len(selected) - used_terminal
    self.counters["terminal_starts"] = used_terminal
    self.counters["pre_step_starts"] = used_pre
    reserved_round = terminal_take + pre_take
    self.counters["borrowed_start_reservations"] = max(
      0, len(selected) - reserved_round
    )
    dropped = len(self._seeds) - len(selected)
    if dropped:
      self.counters["start_cap_dropped"] = (
        self.counters.get("start_cap_dropped", 0) + dropped
      )
    return selected

  def run_available(
    self,
    backend_factory: Callable[[], Any],
    teacher: Callable[[Any], Any],
    *,
    rng_seed: int = 0,
    actual_main_row_steps: int | None = None,
    active_worlds: int | None = None,
  ) -> tuple[BranchResult, ...]:
    if (
      isinstance(actual_main_row_steps, bool)
      or actual_main_row_steps is None
      or actual_main_row_steps < 0
    ):
      raise ValueError(
        "actual_main_row_steps is required for branch resource accounting"
      )
    tokens = [int(actual_main_row_steps * self.budget.max_main_row_steps_fraction)]
    if tokens[0] <= 0 or not self._seeds:
      self._seeds.clear()
      self.counters["budget_censored"] = self.counters.get("budget_censored", 0) + 1
      return ()
    seeds = self._select_call_seeds()
    self._seeds.clear()
    results: list[BranchResult] = []
    slots: list[Any] = []

    def consume(cost: int) -> None:
      if cost < 0 or tokens[0] < cost:
        raise CoverageResourceError("branch physics budget exhausted")
      tokens[0] -= cost

    with isolated_branch_rng(rng_seed):
      try:
        resident_total = 0
        slot_count = min(
          self.budget.branch_slots, self.budget.resident_slots, len(seeds)
        )
        for _ in range(slot_count):
          slot = backend_factory()
          slots.append(slot)
          resident = _reported_int(slot, "resident_bytes")
          resident_total += resident
          if resident_total > self.budget.resident_bytes_cap:
            raise CoverageResourceError(
              "resident branch allocation exceeds configured resident byte cap"
            )
        for ordinal, seed in enumerate(seeds):
          slot = slots[ordinal % len(slots)]
          cost = _reported_int(slot, "resident_world_count")
          if active_worlds is not None:
            if isinstance(active_worlds, bool) or active_worlds < cost:
              raise CoverageResourceError(
                "active_worlds cannot underreport padded resident worlds"
              )
          branch = TeacherBranch(seed, self.budget, str(self._next_id))
          self._next_id += 1
          results.append(
            branch.run(
              _BudgetBackend(slot, consume, cost), teacher, seed=rng_seed + ordinal
            )
          )
        self.counters["resource_tokens_remaining"] = tokens[0]
        if len(seeds) > len(slots):
          self.counters["resident_slot_reuse"] = len(seeds) - len(slots)
      finally:
        for slot in slots:
          close = getattr(slot, "close", None)
          if callable(close):
            close()
    return tuple(results)


def _seed_is_terminal(seed: BranchSeed) -> bool:
  return seed.source_kind == "finite_post_failure" or bool(
    seed.metadata is not None and seed.metadata.source_kind == "finite_post_failure"
  )


def _reported_int(backend: Any, name: str) -> int:
  value = getattr(backend, name, None)
  if isinstance(value, bool) or not isinstance(value, int) or value < 0:
    raise CoverageResourceError(f"backend must report nonnegative integer {name}")
  if name == "resident_world_count" and value == 0:
    raise CoverageResourceError("backend must report resident_world_count >= 1")
  return value


def _owned(value: Any) -> Any:
  if isinstance(value, torch.Tensor):
    return value.detach().clone()
  if isinstance(value, np.ndarray):
    return value.copy()
  if isinstance(value, dict):
    return {key: _owned(item) for key, item in value.items()}
  if isinstance(value, list):
    return [_owned(item) for item in value]
  if isinstance(value, tuple):
    return tuple(_owned(item) for item in value)
  return copy.deepcopy(value)


def _finite(value: Any) -> bool:
  if isinstance(value, torch.Tensor):
    return bool(torch.isfinite(value).all())
  if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
    return bool(np.isfinite(value).all())
  if isinstance(value, dict):
    return all(_finite(item) for item in value.values())
  if isinstance(value, (list, tuple)):
    return all(_finite(item) for item in value)
  if isinstance(value, (float, np.floating)):
    return bool(np.isfinite(value))
  return True


def _get(value: Any, name: str, default: Any = None) -> Any:
  return (
    value.get(name, default)
    if isinstance(value, dict)
    else getattr(value, name, default)
  )


def _view(backend: Any) -> Any:
  for name in ("coverage_policy_view", "get_observations", "observation", "obs"):
    value = getattr(backend, name, None)
    if callable(value):
      return value()
    if value is not None:
      return value
  raise BranchStateError("branch backend has no policy-view handler")


def _failure(result: Any) -> tuple[bool, str | None]:
  return bool(
    _get(result, "tracking_failure", _get(result, "terminated", False))
  ), _get(result, "failure_reason", None)


def _floor(result: Any, budget: BranchBudget) -> bool:
  return (
    abs(float(_get(result, "root_tilt_rad", 0.0))) >= budget.floor_tilt_rad
    and float(_get(result, "root_height_m", float("inf"))) <= budget.floor_height_m
  )
