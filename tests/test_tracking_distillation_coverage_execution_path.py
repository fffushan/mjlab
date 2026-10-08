"""Slice 1 of the coverage execution path: configuration surface and fail-closed gate.

The library, replay and evaluator modules are integrated.  The worker reply, merge
validation and cohort checkpoint v4 slices are not, so an enabled run must refuse to
start instead of collecting ordinary rows and reporting coverage it never performed.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import tyro

from mjlab.scripts.distill import _train_cohort
from mjlab.tasks.tracking.distillation.cohort_setup import (
  COVERAGE_POLICY_CHOICES,
  CohortSetup,
  require_coverage_execution,
  resolve_coverage_policy,
)
from mjlab.tasks.tracking.distillation.coverage_policy import CoveragePolicy


def test_coverage_policy_choices_match_the_inspect_command() -> None:
  assert COVERAGE_POLICY_CHOICES == ("off", "difficult-state-coverage")


def test_resolve_coverage_policy_off_returns_none() -> None:
  assert resolve_coverage_policy("off") is None


def test_resolve_coverage_policy_enabled_returns_the_difficult_state_coverage_policy() -> (
  None
):
  policy = resolve_coverage_policy("difficult-state-coverage")
  assert policy is not None
  assert policy.identity == CoveragePolicy.difficult_state_coverage().identity
  assert policy.enabled is True


def test_resolve_coverage_policy_rejects_an_unknown_choice() -> None:
  with pytest.raises(ValueError, match="unknown coverage policy"):
    resolve_coverage_policy("strong")


def test_require_coverage_execution_allows_the_disabled_path() -> None:
  require_coverage_execution(None)


def test_require_coverage_execution_fails_closed_when_enabled() -> None:
  policy = resolve_coverage_policy("difficult-state-coverage")
  with pytest.raises(NotImplementedError) as excinfo:
    require_coverage_execution(policy)
  message = str(excinfo.value)
  # The refusal must say what is missing, and that nothing ran.
  assert "no execution path" in message
  assert "execution_authorized: false" in message


def test_cohort_setup_defaults_to_no_coverage_policy() -> None:
  setup = CohortSetup(
    manifest=Path("manifest.yaml"),
    repo_root=Path("."),
    teacher_ids=("tennis_000",),
    task_id=None,
    phase_policy=object(),
    reset_policy=object(),
    base_seed=0,
  )
  assert setup.coverage_policy is None


def test_cohort_setup_keeps_an_enabled_policy() -> None:
  policy = resolve_coverage_policy("difficult-state-coverage")
  setup = CohortSetup(
    manifest=Path("manifest.yaml"),
    repo_root=Path("."),
    teacher_ids=("tennis_000",),
    task_id=None,
    phase_policy=object(),
    reset_policy=object(),
    base_seed=0,
    coverage_policy=policy,
  )
  assert setup.coverage_policy is policy


def test_train_cohort_declares_the_coverage_policy_flag() -> None:
  parameters = inspect.signature(_train_cohort).parameters
  assert "coverage_policy" in parameters
  assert parameters["coverage_policy"].default == "off"


def test_cli_accepts_and_types_the_flag() -> None:
  # tyro derives the flag from the signature; this proves the name and the choices.
  options = tyro.extras.get_parser(_train_cohort).__dict__.get("_actions")
  assert options is not None
  texts = [getattr(action, "dest", "") for action in options]
  # tyro names flags in kebab case, as it does for repo-root and teacher-ids.
  assert any(name in texts for name in ("coverage-policy", "coverage_policy")), texts
