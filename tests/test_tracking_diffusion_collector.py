"""CPU fake-environment lifecycle tests for the D1 collector."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tracking_diffusion_fixtures import make_state

from mjlab.tasks.tracking.diffusion import (
  ActionContract,
  AppendOnlyShardStore,
  CollectorError,
  DiffusionCollector,
  FrozenPpoRecoveryPolicy,
  FrozenVaePolicy,
  PostStepBundle,
  PpoObservation,
  StepEvidence,
  TrialSpec,
  VaeObservation,
)


class _Schema:
  reference_dim = 68
  conditioning_dim = 99


class _Vae(torch.nn.Module):
  schema = _Schema()

  def encode(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros((reference.shape[0], 32)), torch.zeros((reference.shape[0], 32))

  def decode(self, latent: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
    return torch.full((latent.shape[0], 31), 0.1)


class _Environment:
  def __init__(self, *, final_fall: bool = False, handoff: bool = True) -> None:
    self.step_count = 0
    self.reset_calls = 0
    self.trial_reset_calls = 0
    self.switch_calls = 0
    self.controllers: list[str] = []
    self.final_fall = final_fall
    self.handoff = handoff

  def reset(self, *, seed: int, motion_id: str, start_frame: int) -> None:
    assert (seed, motion_id, start_frame) == (4, "motion", 3)
    self.step_count = 0
    self.reset_calls += 1

  def observe_vae(self) -> VaeObservation:
    self.controllers.append("vae")
    return VaeObservation(
      make_state(self.step_count),
      np.ones(68),
      np.ones(99),
      "motion",
      self.step_count,
      0.0,
    )

  def observe_ppo(self) -> PpoObservation:
    self.controllers.append("ppo")
    return PpoObservation(
      make_state(self.step_count), np.zeros(102), "motion", self.step_count, 0.0
    )

  def step(self, action: np.ndarray) -> int:
    assert action.shape == (31,)
    assert np.isfinite(action).all()
    self.step_count += 1
    return self.step_count

  def post_step_evidence(self, result: int) -> PostStepBundle:
    evidence = StepEvidence(
      True,
      terminated=self.handoff and result == 2,
      reference_boundary=self.handoff and result == 2,
      clip_ended=self.handoff and result == 2,
      physical_fall=self.final_fall and result == 250,
    )
    observation = VaeObservation(
      make_state(self.step_count),
      np.ones(68),
      np.ones(99),
      "motion",
      self.step_count,
      0.0,
      anchor_z_error=0.0,
      gravity_z_error=0.0,
      end_effector_z_error=0.0,
    )
    return PostStepBundle(evidence, float(result), observation)

  def switch_to_ppo(self) -> None:
    self.switch_calls += 1

  def capture_initial_state(self) -> str:
    return "initial"

  def restore_initial_state(self, snapshot: object) -> None:
    assert snapshot == "initial"

  def reset_after_trial(self) -> None:
    self.trial_reset_calls += 1


def _collector(env: _Environment) -> DiffusionCollector:
  contract = ActionContract(np.ones(31), np.zeros(31), np.ones(31), np.ones(31))
  return DiffusionCollector(
    env,
    FrozenVaePolicy(_Vae(), contract),
    FrozenPpoRecoveryPolicy(lambda observation: np.zeros((1, 31)), contract),
  )


def test_collector_aligns_pre_action_rows_and_preserves_last_frame_handoff() -> None:
  env = _Environment()
  collector = _collector(env)
  collector.collect(
    TrialSpec("run", "motion", 4, 3, pair_id="pair", initial_state_id="initial")
  )
  result = collector.collect(
    TrialSpec("run", "motion", 4, 3, "ou", pair_id="pair", initial_state_id="initial")
  )
  assert result.qualification.qualified
  assert result.qualification.label == "vae_then_standing"
  assert len(result.rows) == 250
  assert result.rows[0].controller == "vae"
  assert result.rows[0].previous_executed_action.tolist() == [0.0] * 31
  assert result.rows[0].executed_action.shape == (31,)
  assert result.rows[0].latent is not None
  assert result.rows[1].reference_boundary
  assert result.rows[2].controller == "ppo"
  assert result.rows[2].latent is None
  assert env.switch_calls == 2
  assert env.trial_reset_calls == 2


def test_post_step_tilt_is_derived_and_persisted_as_failure() -> None:
  class _PostTilt(_Environment):
    def post_step_evidence(self, result: int) -> PostStepBundle:
      pre = VaeObservation(
        make_state(self.step_count),
        np.ones(68),
        np.ones(99),
        "motion",
        self.step_count,
        0.0,
        physical_tilt_degrees=80.0 if result == 1 else 0.0,
        anchor_z_error=0.0,
        gravity_z_error=0.0,
        end_effector_z_error=0.0,
      )
      return PostStepBundle(StepEvidence(True), float(result), pre)

  result = _collector(_PostTilt()).collect(TrialSpec("run", "motion", 4, 3))
  assert result.qualification.reason == "physical_fall"
  assert result.rows[0].terminal_evidence is not None
  assert result.rows[0].terminal_evidence.physical_fall


def test_final_step_fall_is_failed_and_not_accepted() -> None:
  env = _Environment(final_fall=True)
  result = _collector(env).collect(TrialSpec("run", "motion", 4, 3))
  assert result.failed_attempt
  assert result.qualification.reason == "physical_fall"
  assert result.accepted_rows == ()
  assert env.trial_reset_calls == 1


def test_remaining_transition_budget_stops_before_extra_action() -> None:
  env = _Environment()
  result = _collector(env).collect(
    TrialSpec("run", "motion", 4, 3), remaining_transitions=2
  )
  assert env.step_count == 2
  assert result.stopped_reason == "control_transition_budget"
  assert not result.qualification.qualified
  assert result.rows[0].group_key == "motion:3"
  assert result.rows[0].provenance["group_key_generated"] == "true"


def test_missing_post_step_observation_rejects_vae_trial() -> None:
  class _MissingObservation(_Environment):
    def post_step_evidence(self, result: int) -> PostStepBundle:
      return PostStepBundle(StepEvidence(True), float(result))

  result = _collector(_MissingObservation()).collect(TrialSpec("run", "motion", 4, 3))
  assert result.qualification.reason == "vae_tracking_rejection"


def test_tilted_post_step_quaternion_rejects_without_metric() -> None:
  from dataclasses import replace

  class _QuaternionTilt(_Environment):
    def post_step_evidence(self, result: int) -> PostStepBundle:
      observation = VaeObservation(
        replace(
          make_state(self.step_count),
          root_quaternion_wxyz=np.array(
            [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]
          ),
        ),
        np.ones(68),
        np.ones(99),
        "motion",
        self.step_count,
        0.0,
        anchor_z_error=0.0,
        gravity_z_error=0.0,
        end_effector_z_error=0.0,
      )
      return PostStepBundle(StepEvidence(True), float(result), observation)

  result = _collector(_QuaternionTilt()).collect(TrialSpec("run", "motion", 4, 3))
  assert result.qualification.reason == "physical_fall"


def test_snapshot_setup_failure_still_resets_after_trial() -> None:
  class _CaptureFailure(_Environment):
    def capture_initial_state(self) -> str:
      raise RuntimeError("snapshot capture failed")

  env = _CaptureFailure()
  with pytest.raises(RuntimeError, match="snapshot capture failed"):
    _collector(env).collect(
      TrialSpec("run", "motion", 4, 3, pair_id="pair", initial_state_id="initial")
    )
  assert env.reset_calls == 1
  assert env.trial_reset_calls == 1


def test_pair_snapshots_do_not_alias_duplicate_initial_state_ids() -> None:
  class _DistinctSnapshots(_Environment):
    def __init__(self) -> None:
      super().__init__()
      self.capture_count = 0
      self.restored: list[str] = []

    def capture_initial_state(self) -> str:
      self.capture_count += 1
      return f"snapshot-{self.capture_count}"

    def restore_initial_state(self, snapshot: object) -> None:
      assert isinstance(snapshot, str)
      self.restored.append(snapshot)

  env = _DistinctSnapshots()
  collector = _collector(env)
  clean_a = TrialSpec("run", "motion", 4, 3, pair_id="a", initial_state_id="same")
  clean_b = TrialSpec("run", "motion", 4, 3, pair_id="b", initial_state_id="same")
  ou_a = TrialSpec("run", "motion", 4, 3, "ou", pair_id="a", initial_state_id="same")
  ou_b = TrialSpec("run", "motion", 4, 3, "ou", pair_id="b", initial_state_id="same")
  collector.collect(clean_a)
  collector.collect(clean_b)
  collector.collect(ou_a)
  collector.collect(ou_b)
  assert env.restored == ["snapshot-1", "snapshot-2"]


def test_collector_rejects_clean_after_ou_phase_starts() -> None:
  env = _Environment()
  collector = _collector(env)
  clean = TrialSpec("run", "motion", 4, 3, pair_id="a", initial_state_id="state")
  ou = TrialSpec("run", "motion", 4, 3, "ou", pair_id="a", initial_state_id="state")
  collector.collect(clean)
  collector.collect(ou)
  with pytest.raises(CollectorError, match="after the OU phase"):
    collector.collect(
      TrialSpec("run", "motion", 4, 3, pair_id="b", initial_state_id="state-b")
    )


def test_failed_clean_partner_blocks_ou_trial() -> None:
  collector = _collector(_Environment(final_fall=True))
  clean = TrialSpec("run", "motion", 4, 3, pair_id="pair", initial_state_id="state")
  collector.collect(clean)
  ou = TrialSpec("run", "motion", 4, 3, "ou", pair_id="pair", initial_state_id="state")
  with pytest.raises(CollectorError, match="clean partner did not qualify"):
    collector.collect(ou)


def test_qualified_clean_without_handoff_permits_ou_trial() -> None:
  """A ``vae_only`` clean partner is a valid OU pair.

  The perturbation axis only needs a clean rollout that survived its window; a
  clip end inside the budget is an endpoint-audit question, not a precondition.
  """
  collector = _collector(_Environment(handoff=False))
  clean = TrialSpec("run", "motion", 4, 3, pair_id="pair", initial_state_id="state")
  clean_result = collector.collect(clean)
  assert clean_result.qualification.qualified
  assert clean_result.qualification.label == "vae_only"
  assert clean_result.qualification.handoff_step is None
  ou = TrialSpec("run", "motion", 4, 3, "ou", pair_id="pair", initial_state_id="state")
  ou_result = collector.collect(ou)
  assert ou_result.qualification.qualified
  assert ou_result.qualification.label == "vae_only"
  assert ou_result.accepted_rows


def test_qualified_rows_are_persisted_with_segment_verdict(tmp_path) -> None:
  env = _Environment()
  contract = ActionContract(np.ones(31), np.zeros(31), np.ones(31), np.ones(31))
  store = AppendOnlyShardStore(tmp_path, max_rows_per_shard=300)
  result = DiffusionCollector(
    env,
    FrozenVaePolicy(_Vae(), contract),
    FrozenPpoRecoveryPolicy(lambda observation: np.zeros((1, 31)), contract),
    store=store,
  ).collect(TrialSpec("run", "motion", 4, 3))
  persisted = tuple(store.iter_rows())
  assert len(persisted) == len(result.rows)
  assert sum(row.segment_qualified for row in persisted) == len(result.accepted_rows)
