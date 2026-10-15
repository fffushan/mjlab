"""Slice 1 of the coverage execution path: configuration surface and fail-closed gate.

The library, replay and evaluator modules are integrated.  The worker reply, merge
validation and cohort checkpoint v4 slices are not, so an enabled run must refuse to
start instead of collecting ordinary rows and reporting coverage it never performed.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import tyro

from mjlab.scripts.distill import _resolve_coverage_choice, _train, _train_cohort
from mjlab.tasks.tracking.distillation.cohort_setup import (
  COVERAGE_POLICY_CHOICES,
  CohortSetup,
  coverage_schema_payload,
  require_coverage_execution,
  resolve_coverage_policy,
)
from mjlab.tasks.tracking.distillation.coverage_policy import CoveragePolicy
from mjlab.tasks.tracking.distillation.vae_config import make_schema


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


def test_require_coverage_execution_requires_parity_evidence() -> None:
  policy = resolve_coverage_policy("difficult-state-coverage")
  with pytest.raises(ValueError, match="coverage-parity-evidence") as excinfo:
    require_coverage_execution(policy)
  assert "restore-parity" in str(excinfo.value)


def test_require_coverage_execution_accepts_passing_parity_evidence(tmp_path) -> None:
  policy = resolve_coverage_policy("difficult-state-coverage")
  evidence = tmp_path / "parity.json"
  evidence.write_text(
    '{"kind":"x2-vae-coverage-restore-parity-v1",'
    '"status":"pass","manifest_sha256":"manifest",'
    '"policy_input_exact":true,'
    '"schema":{"reference_dim":68,"conditioning_dim":99,"latent_dim":32,"action_dim":31},'
    '"policy_input_max_abs_error":{"teacher_actor_view":0.0,'
    '"student_reference":0.0,"student_conditioning":0.0,"previous_action":0.0},'
    '"teacher_ids":["inc000"]}'
  )
  result = require_coverage_execution(policy, evidence, manifest_sha256="manifest")
  assert result["kind"] == "x2-vae-coverage-restore-parity-v1"
  assert result["status"] == "pass"
  assert len(result["sha256"]) == 64


def test_require_coverage_execution_rejects_incomplete_parity_evidence(
  tmp_path,
) -> None:
  policy = resolve_coverage_policy("difficult-state-coverage")
  evidence = tmp_path / "incomplete.json"
  evidence.write_text(
    '{"kind":"x2-vae-coverage-restore-parity-v1",'
    '"status":"pass","manifest_sha256":"manifest"}'
  )
  with pytest.raises(ValueError, match="exact policy-input parity"):
    require_coverage_execution(policy, evidence, manifest_sha256="manifest")

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


def test_public_train_leaves_the_coverage_choice_unset() -> None:
  # None means "unset": the cohort path enables coverage, the single-teacher path
  # cannot carry it.  An explicit value always wins.
  parameters = inspect.signature(_train).parameters
  assert parameters["coverage_policy"].default is None


def test_unset_coverage_is_enabled_on_the_cohort_path() -> None:
  assert _resolve_coverage_choice(None, ("tennis_000", "tennis_001")) == (
    "difficult-state-coverage"
  )


def test_unset_coverage_is_off_on_the_single_teacher_path() -> None:
  # A single-teacher run cannot carry coverage, so an omitted choice means off
  # rather than a refusal: `distill train` without --teacher-ids keeps working.
  assert _resolve_coverage_choice(None, ()) == "off"


def test_explicit_coverage_choice_always_wins() -> None:
  assert _resolve_coverage_choice("off", ("tennis_000",)) == "off"
  assert _resolve_coverage_choice("off", ()) == "off"
  assert _resolve_coverage_choice("difficult-state-coverage", ("tennis_000",)) == (
    "difficult-state-coverage"
  )


def test_enabled_coverage_without_evidence_names_the_remedy() -> None:
  policy = resolve_coverage_policy("difficult-state-coverage")
  with pytest.raises(ValueError) as excinfo:
    require_coverage_execution(policy, None, manifest_sha256="0" * 64)
  message = str(excinfo.value)
  assert "--coverage-parity-evidence" in message
  assert "distill coverage-parity" in message
  assert "--coverage-policy off" in message


def test_cli_accepts_and_types_the_flag() -> None:
  # tyro derives the flag from the signature; this proves the name and the choices.
  options = tyro.extras.get_parser(_train_cohort).__dict__.get("_actions")
  assert options is not None
  texts = [getattr(action, "dest", "") for action in options]
  # tyro names flags in kebab case, as it does for repo-root and teacher-ids.
  assert any(name in texts for name in ("coverage-policy", "coverage_policy")), texts


def _parity_evidence(
  schema: dict[str, int], teacher_ids: tuple[str, ...] = ("inc000",)
) -> str:
  ids = ",".join(f'"{value}"' for value in teacher_ids)
  return (
    '{"kind":"x2-vae-coverage-restore-parity-v1",'
    '"status":"pass","manifest_sha256":"manifest",'
    '"policy_input_exact":true,'
    f'"schema":{json.dumps(schema)},'
    '"policy_input_max_abs_error":{"teacher_actor_view":0.0,'
    '"student_reference":0.0,"student_conditioning":0.0,"previous_action":0.0},'
    f'"teacher_ids":[{ids}]}}'
  )


def _cohort_schema(preview: bool):
  return make_schema(preview=preview)


def test_resolve_coverage_policy_reads_the_schema_it_is_given() -> None:
  """The policy scope records the layout the task configuration resolved."""
  legacy = resolve_coverage_policy("difficult-state-coverage")
  assert dict(legacy.scope)["reference_dim"] == 68

  preview_schema = _cohort_schema(preview=True)
  preview = resolve_coverage_policy("difficult-state-coverage", schema=preview_schema)
  declared = dict(preview.scope)
  assert declared["reference_dim"] == preview_schema.reference_dim == 130
  assert declared["conditioning_dim"] == preview_schema.conditioning_dim
  assert declared["latent_dim"] == preview_schema.latent_dim
  assert declared["action_dim"] == preview_schema.action_dim
  # The identity names the layout, so the two policies are not interchangeable.
  assert preview.identity != legacy.identity


def test_coverage_gate_accepts_evidence_for_the_resolved_schema(tmp_path) -> None:
  """A preview cohort's own evidence authorizes it, which the version-1 pin denied."""
  schema = _cohort_schema(preview=True)
  policy = resolve_coverage_policy("difficult-state-coverage", schema=schema)
  evidence = tmp_path / "preview.json"
  evidence.write_text(_parity_evidence(coverage_schema_payload(schema)))
  result = require_coverage_execution(
    policy,
    evidence,
    manifest_sha256="manifest",
    expected_schema=coverage_schema_payload(schema),
  )
  assert result["status"] == "pass"


def test_coverage_gate_refuses_evidence_for_another_reference_width(tmp_path) -> None:
  """Width is the point: evidence for one layout must not authorize another."""
  preview_schema = _cohort_schema(preview=True)
  policy = resolve_coverage_policy("difficult-state-coverage", schema=preview_schema)
  legacy_evidence = tmp_path / "legacy.json"
  legacy_evidence.write_text(_parity_evidence(coverage_schema_payload(_cohort_schema(False))))
  with pytest.raises(ValueError, match="describes schema"):
    require_coverage_execution(
      policy,
      legacy_evidence,
      manifest_sha256="manifest",
      expected_schema=coverage_schema_payload(preview_schema),
    )
  # And the version-1 layout still refuses the preview evidence, so the gate is not
  # merely permissive in one direction.
  preview_evidence = tmp_path / "preview.json"
  preview_evidence.write_text(
    _parity_evidence(coverage_schema_payload(preview_schema))
  )
  with pytest.raises(ValueError, match="describes schema"):
    require_coverage_execution(policy, preview_evidence, manifest_sha256="manifest")


def test_coverage_gate_keeps_the_version_1_default_for_old_callers(tmp_path) -> None:
  """Omitting the resolved schema leaves the earlier contract exactly as it was."""
  policy = resolve_coverage_policy("difficult-state-coverage")
  evidence = tmp_path / "legacy.json"
  evidence.write_text(_parity_evidence(coverage_schema_payload(_cohort_schema(False))))
  assert (
    require_coverage_execution(policy, evidence, manifest_sha256="manifest")["status"]
    == "pass"
  )
