"""Small runtime collector seam used by integration-owned collection wiring.

This module owns no replay, optimizer, checkpoint, or CLI behavior.  It captures
terminal state before reset and exposes bounded branch execution while preserving
legacy collection when the policy is off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .coverage_environment import CoverageEnvironmentComponent, TerminalCapture
from mjlab.managers.recorder_manager import RecorderTerm
from .teacher_branches import (
  BranchBudget,
  BranchResult,
  BranchSeed,
  TeacherBranchManager,
)


@dataclass(frozen=True, slots=True)
class CoverageCollectionReport:
  terminal_captures: int = 0
  admitted_seeds: int = 0
  branch_results: tuple[BranchResult, ...] = ()
  counters: dict[str, int] = field(default_factory=dict)


class CoverageCollector:
  """Opt-in observer/branch component; disabled mode is an exact no-op."""

  def __init__(
    self,
    component: CoverageEnvironmentComponent,
    *,
    budget: BranchBudget | None = None,
  ) -> None:
    self.component, self.branches = component, TeacherBranchManager(budget)
    self._captures: list[TerminalCapture] = []

  @property
  def enabled(self) -> bool:
    return self.component.enabled

  def record_pre_reset(
    self, env_ids: torch.Tensor, *, termination: Any
  ) -> tuple[TerminalCapture, ...]:
    if not self.enabled:
      return ()
    captures = self.component.capture_raw_reached_pre_reset(
      env_ids, termination=termination
    )
    self._captures.extend(captures)
    return captures

  def admit(self, seed: BranchSeed) -> bool:
    if not self.enabled:
      return False
    return self.branches.admit(seed)

  def run_branches(
    self,
    backend_factory: Callable[[], Any],
    teacher: Callable[[Any], Any],
    *,
    rng_seed: int = 0,
    actual_main_row_steps: int | None = None,
    active_worlds: int | None = None,
  ) -> CoverageCollectionReport:
    if not self.enabled:
      return CoverageCollectionReport()
    results = self.branches.run_available(
      backend_factory,
      teacher,
      rng_seed=rng_seed,
      actual_main_row_steps=actual_main_row_steps,
      active_worlds=active_worlds,
    )
    return CoverageCollectionReport(
      len(self._captures), len(results), results, dict(self.branches.counters)
    )

  def clear_call_state(self) -> None:
    self._captures.clear()
    self.component.clear_terminal_captures()
    self.branches._seeds.clear()


COVERAGE_COLLECTOR_ATTRIBUTE = "coverage_collector"


class CoveragePreResetRecorder(RecorderTerm):
  """Forward the pre-reset environment ids to the coverage collector.

  The environment calls ``recorder_manager.record_pre_reset(reset_env_ids)`` before
  ``_reset_idx``, so this term sees the reached terminal state while it is still in
  the simulation.  The collector is attached to the environment by whoever enables
  coverage, so a run with no collector records nothing.
  """

  def __init__(self, cfg: Any, env: Any) -> None:
    super().__init__(cfg, env)
    self.records = 0

  @property
  def env(self) -> Any:
    """The environment, whichever attribute the term base exposes."""
    return getattr(self, "_env", None) or getattr(self, "env_", None)

  def record_pre_reset(self, env_ids: torch.Tensor) -> None:
    env = self.env
    collector = getattr(env, COVERAGE_COLLECTOR_ATTRIBUTE, None)
    if collector is None or not getattr(collector, "enabled", False):
      return
    collector.record_pre_reset(env_ids, termination=_termination_reason(env))
    self.records += 1


def _termination_reason(env: Any) -> dict[str, Any]:
  """Name the active termination terms at the pre-reset instant, when available."""
  manager = getattr(env, "termination_manager", None)
  names = getattr(manager, "active_terms", None)
  if names:
    return {"active_terms": tuple(str(name) for name in names)}
  return {}


def attach_coverage_collector(env: Any, collector: Any) -> None:
  """Attach the collector so the pre-reset recorder term can reach it."""
  setattr(env, COVERAGE_COLLECTOR_ATTRIBUTE, collector)


class CoverageStateOwner:
  """Create the fixed-topology state adapter on first use.

  The adapter validates that the cached-observation buffers alias, and that holds
  only once the environment has been reset.  Building it at environment-build time
  therefore fails; deferring it to the first capture succeeds.  Everything else is
  forwarded to the environment, so a preflight sees the environment's own managers.
  """

  def __init__(self, env: Any) -> None:
    self._env = env
    self._adapter: Any | None = None

  @property
  def adapter(self) -> Any:
    if self._adapter is None:
      from mjlab.tasks.tracking.distillation.x2_state_adapter import (
        X2FixedTopologyStateAdapter,
      )

      self._adapter = X2FixedTopologyStateAdapter(self._env, source_row=0)
    return self._adapter

  def coverage_cached_policy_view(self, index: int) -> Any:
    return self.adapter.coverage_cached_policy_view(index)

  def coverage_executed_action(self, index: int) -> Any:
    return self.adapter.coverage_executed_action(index)

  def __getattr__(self, name: str) -> Any:
    # sim, scene, managers and the capture/restore fields come from the environment.
    return getattr(self._env, name)


def build_coverage_collector(
  environment: Any,
  policy: Any | None,
  *,
  budget: BranchBudget | None = None,
) -> CoverageCollector | None:
  """Build the coverage collector for one environment, or return ``None``.

  Returns ``None`` when the run has no policy or the policy is disabled, so a
  disabled run constructs no component and keeps the existing collection path
  exactly as it was.  The environment owner calls this: in the sharded layout each
  worker owns its own environments, so each worker builds its own collector and
  reports its own branch rows in its reply.
  """
  if policy is None or not getattr(policy, "enabled", False):
    return None
  return CoverageCollector(CoverageEnvironmentComponent(environment, policy), budget=budget)
