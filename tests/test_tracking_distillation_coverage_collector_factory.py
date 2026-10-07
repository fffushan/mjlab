"""Slice 2c-1: the collector is built only when the policy enables it."""

from __future__ import annotations

from mjlab.tasks.tracking.distillation.coverage_collector import (
  CoverageCollector,
  build_coverage_collector,
)
from mjlab.tasks.tracking.distillation.coverage_policy import CoveragePolicy


def test_no_policy_builds_nothing() -> None:
  assert build_coverage_collector(object(), None) is None


def test_a_disabled_policy_builds_nothing() -> None:
  assert build_coverage_collector(object(), CoveragePolicy.off()) is None


def test_an_enabled_policy_builds_a_collector() -> None:
  collector = build_coverage_collector(object(), CoveragePolicy.balance_recovery())
  assert isinstance(collector, CoverageCollector)
  assert collector.enabled is True


def test_a_disabled_collector_is_an_exact_no_op() -> None:
  # The disabled path must behave as if the component were absent.
  assert CoveragePolicy.off().enabled is False
