"""Slice 3 (barrier half): drain and clear at the collection barrier, with counts.

The P0 contract fixes the policy as drain_and_clear_with_counts (user decision
2026-10-07), so this file also guards the contract against a silent change.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from mjlab.tasks.tracking.distillation.runner import (
  DistillationRunner,
  RunnerValidationError,
)

CONTRACT = Path("docs/plans/beyondmimic_vae_difficult_state_coverage_p0_contract.yaml")


def stand_in(*, coverage=None, collector_present: bool = True):
  collector = SimpleNamespace(coverage=coverage) if collector_present else None
  return SimpleNamespace(collector=collector)


def drain(target) -> dict:
  # The method needs no other runner state, so call it on a stand-in.
  return DistillationRunner.drain_branch_work_unit(target)


def test_no_collector_drains_nothing_and_adds_no_counters() -> None:
  assert drain(stand_in(collector_present=False)) == {}


def test_a_collector_without_a_coverage_component_drains_nothing() -> None:
  assert drain(stand_in(coverage=None)) == {}


def test_a_disabled_component_drains_nothing() -> None:
  assert drain(stand_in(coverage=SimpleNamespace(enabled=False))) == {}


def test_an_enabled_component_without_a_backend_refuses_to_checkpoint() -> None:
  with pytest.raises(RunnerValidationError, match="drain_for_barrier"):
    drain(stand_in(coverage=SimpleNamespace(enabled=True)))


def test_an_enabled_component_reports_drained_started_and_dropped() -> None:
  coverage = SimpleNamespace(
    enabled=True, drain_for_barrier=lambda: {"drained": 3, "started": 2, "dropped": 1}
  )
  counts = drain(stand_in(coverage=coverage))
  assert counts == {
    "coverage_barrier_drained": 3,
    "coverage_barrier_started": 2,
    "coverage_barrier_dropped": 1,
  }


def test_a_drain_that_returns_no_counts_is_rejected() -> None:
  coverage = SimpleNamespace(enabled=True, drain_for_barrier=lambda: [])
  with pytest.raises(RunnerValidationError, match="must return counts"):
    drain(stand_in(coverage=coverage))


def test_the_contract_fixes_the_barrier_policy() -> None:
  contract = yaml.safe_load(CONTRACT.read_text())
  # Drain and clear is the user decision.  Persisting the queue stays closed.
  assert contract["barrier_seed_queue_policy"] == "drain_and_clear_with_counts"
  assert contract["barrier_seed_queue_user_decision"] == "2026-10-07"
  assert contract["execution_authorized"] is False
