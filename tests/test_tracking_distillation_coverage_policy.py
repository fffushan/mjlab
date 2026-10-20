import json
import random

import pytest

from mjlab.tasks.tracking.distillation.coverage_policy import (
  CoveragePolicy,
  CoveragePolicyError,
)


def test_off_is_immutable_and_does_not_consume_rng() -> None:
  policy = CoveragePolicy.off()
  left, right = random.Random(10), random.Random(10)
  from mjlab.tasks.tracking.distillation.coverage_perturbations import (
    sample_reset_perturbation,
  )

  sample_reset_perturbation(policy, left)
  assert left.random() == right.random()
  with pytest.raises((AttributeError, TypeError)):
    policy.mode = "difficult-state-coverage"  # type: ignore[misc]


def test_confirmed_policy_round_trips_with_stable_identity() -> None:
  policy = CoveragePolicy.difficult_state_coverage()
  assert policy.enabled
  assert policy.identity == CoveragePolicy.from_dict(policy.as_dict()).identity
  assert dict(policy.scope)["reference_dim"] == 68
  assert dict(policy.scope)["conditioning_dim"] == 99
  assert dict(policy.scope)["latent_dim"] == 32
  assert dict(policy.scope)["action_dim"] == 31
  assert "cohort_manifest" not in dict(policy.scope)
  assert "manifest_sha256" not in dict(policy.scope)
  assert dict(policy.tier_probabilities) == {
    "baseline": 0.6,
    "moderate": 0.3,
    "strong": 0.1,
  }


def test_malformed_policy_and_nonfinite_values_are_rejected() -> None:
  with pytest.raises(CoveragePolicyError):
    CoveragePolicy.from_dict({"mode": "difficult-state-coverage"})
  payload = CoveragePolicy.difficult_state_coverage().as_dict()
  payload["tier_probabilities"]["strong"] = float("nan")
  with pytest.raises(CoveragePolicyError):
    CoveragePolicy.from_dict(json.loads(json.dumps(payload, allow_nan=True)))
  with pytest.raises(CoveragePolicyError):
    CoveragePolicy.from_dict({"mode": "unknown"})


def test_nested_direct_policy_validators_reject_malformed_settings() -> None:
  policy = CoveragePolicy.difficult_state_coverage()
  from dataclasses import replace

  with pytest.raises(CoveragePolicyError):
    replace(
      policy, tier_probabilities=(("baseline", 0.5), ("baseline", 0.5), ("strong", 0.0))
    )
  with pytest.raises(CoveragePolicyError):
    replace(
      policy,
      reset_excitation_probabilities=(
        ("root_only", 1.1),
        ("joint_chain_only", -0.1),
        ("coupled", 0.0),
      ),
    )
  with pytest.raises(CoveragePolicyError):
    replace(
      policy,
      root_direction_probabilities=(
        ("sagittal", 0.5),
        ("lateral", 0.5),
        ("unexpected", 0.1),
      ),
    )
  with pytest.raises(CoveragePolicyError):
    replace(
      policy,
      triggers=type(policy.triggers)(
        tuple(
          (name, value)
          for name, value in policy.triggers.thresholds
          if name != "root_angular_velocity_error_rad_s"
        )
      ),
    )
  with pytest.raises(CoveragePolicyError):
    replace(
      policy, runtime=type(policy.runtime)(per_segment_opportunity_probability=1.1)
    )


def test_preview_schema_policy_identity_round_trips_through_from_dict() -> None:
  """A preview cohort's identity survives from_dict and differs from version 1.

  The identity records the student layout the policy was built for.  A version-2
  cohort observes a wider reference preview, so its identity must differ from the
  version-1 one, and ``from_dict`` must accept its own payload.  The checkpoint
  gates rebuild the expected identity from the artifact's own schema, so a
  reconstruction that silently fell back to the version-1 defaults refused a valid
  checkpoint.
  """
  v1 = CoveragePolicy.difficult_state_coverage()
  v2 = CoveragePolicy.difficult_state_coverage(
    reference_dim=130, conditioning_dim=99, latent_dim=32, action_dim=31
  )
  assert v2.identity != v1.identity
  assert CoveragePolicy.from_dict(v1.as_dict()).identity == v1.identity
  assert CoveragePolicy.from_dict(v2.as_dict()).identity == v2.identity
  assert CoveragePolicy.from_dict(v2.as_dict()) == v2
