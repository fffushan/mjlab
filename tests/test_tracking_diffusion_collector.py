"""CPU fake-environment lifecycle tests for the D1 collector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from tracking_diffusion_fixtures import make_state

from mjlab.tasks.tracking.diffusion import (
  ActionContract,
  AppendOnlyShardStore,
  CollectorError,
  DiffusionCollector,
  FrozenVaePolicy,
  PostStepBundle,
  StepEvidence,
  TrialSpec,
  VaeObservation,
  WorldState,
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


@dataclass(frozen=True)
class _DefaultRow:
  """The standing padding row this fake environment serves."""

  reference: np.ndarray
  state: WorldState


def _default_row() -> _DefaultRow:
  return _DefaultRow(
    reference=np.arange(68, dtype=np.float64),
    state=WorldState(
      np.array([0.1, 0.2, 0.66]),
      np.array([1.0, 0.0, 0.0, 0.0]),
      np.zeros(3),
      np.zeros(3),
      np.arange(60, dtype=np.float64).reshape(20, 3) * 0.001,
      np.zeros((20, 3)),
    ),
  )


class _Environment:
  """Fake single-row environment with a ``clip_frames``-long clip."""

  def __init__(
    self, *, clip_frames: int = 3, endpoint_tilt: float = 0.0, mid_tilt: float = 0.0
  ) -> None:
    self.step_count = 0
    self.reset_calls = 0
    self.trial_reset_calls = 0
    self.clip_frames = clip_frames
    self.endpoint_tilt = endpoint_tilt
    self.mid_tilt = mid_tilt
    self.resets: list[tuple[int, str, bool]] = []
    self.default_row_calls = 0

  def reset(
    self,
    *,
    seed: int,
    motion_id: str,
    initial_pose: str = "reference",
    perturb: bool = True,
  ) -> None:
    assert motion_id == "motion"
    self.step_count = 0
    self.reset_calls += 1
    self.resets.append((seed, initial_pose, perturb))

  def default_row(self, motion_id: str) -> _DefaultRow:
    assert motion_id == "motion"
    self.default_row_calls += 1
    return _default_row()

  def observe_vae(self) -> VaeObservation:
    return VaeObservation(
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

  def step(self, action: np.ndarray) -> int:
    assert action.shape == (31,)
    assert np.isfinite(action).all()
    self.step_count += 1
    return self.step_count

  def post_step_evidence(self, result: int) -> PostStepBundle:
    boundary = result >= self.clip_frames
    observation = VaeObservation(
      make_state(self.step_count),
      np.ones(68),
      np.ones(99),
      "motion",
      self.step_count,
      0.0,
      physical_tilt_degrees=(self.endpoint_tilt if boundary else self.mid_tilt),
      anchor_z_error=0.0,
      gravity_z_error=0.0,
      end_effector_z_error=0.0,
    )
    return PostStepBundle(
      StepEvidence(True, reference_boundary=boundary, clip_ended=boundary),
      float(result),
      observation,
    )

  def capture_initial_state(self) -> str:
    return "initial"

  def restore_initial_state(self, snapshot: object) -> None:
    assert snapshot == "initial"

  def reset_after_trial(self) -> None:
    self.trial_reset_calls += 1


def _collector(env: _Environment | None = None, **kwargs) -> DiffusionCollector:
  contract = ActionContract(np.ones(31), np.zeros(31), np.ones(31), np.ones(31))
  return DiffusionCollector(
    env if env is not None else _Environment(),
    FrozenVaePolicy(_Vae(), contract),
    **kwargs,
  )


def _pair(seed: int = 4) -> tuple[TrialSpec, TrialSpec]:
  clean = TrialSpec("run", "motion", seed, 0, pair_id="pair", initial_state_id="s")
  ou = TrialSpec("run", "motion", seed, 0, "ou", pair_id="pair", initial_state_id="s")
  return clean, ou


def test_a_clean_rollout_runs_to_its_endpoint_and_qualifies() -> None:
  env = _Environment(clip_frames=3)
  clean, ou = _pair()
  collector = _collector(env)
  collector.collect(clean)
  result = collector.collect(ou)
  assert result.qualification.qualified
  assert result.qualification.label == "endpoint_pass"
  assert result.qualification.steps == 3
  # Eight padding rows precede the three clip rows.
  assert len(result.rows) == 8 + 3
  assert [row.tick for row in result.rows] == list(range(11))
  assert [row.reference_frame for row in result.rows] == [
    -8,
    -7,
    -6,
    -5,
    -4,
    -3,
    -2,
    -1,
    0,
    1,
    2,
  ]
  assert env.trial_reset_calls == 2
  assert env.resets == [(4, "reference", True), (4, "reference", True)]


def test_the_padding_rows_carry_the_standing_default_row() -> None:
  env = _Environment(clip_frames=2)
  result = _collector(env, qualification=None).collect(TrialSpec("run", "motion", 4, 0))
  padding = result.rows[:8]
  default = _default_row()
  for index, row in enumerate(padding):
    assert row.controller == "vae"
    assert row.reference_frame == index - 8
    assert row.timestamp == pytest.approx((index - 8) * 0.02)
    assert row.latent is not None and row.latent.shape == (32,)
    # The frozen encoder of the standing reference is zeros in this fake.
    np.testing.assert_array_equal(row.latent, np.zeros(32))
    np.testing.assert_array_equal(row.state.root_position, default.state.root_position)
    np.testing.assert_array_equal(
      row.state.root_quaternion_wxyz, np.array([1.0, 0.0, 0.0, 0.0])
    )
    np.testing.assert_array_equal(
      row.state.body_positions, default.state.body_positions
    )
    np.testing.assert_array_equal(row.state.root_linear_velocity, np.zeros(3))
    np.testing.assert_array_equal(row.state.body_linear_velocities, np.zeros((20, 3)))
    assert row.terminal_evidence is not None and row.terminal_evidence.available
    assert not (row.reset or row.teleport or row.reference_boundary)
  # The latent is encoded once per clip, not once per padding row.
  assert env.default_row_calls == 1


def test_a_mid_rollout_deviation_is_kept_and_the_endpoint_decides() -> None:
  env = _Environment(clip_frames=3, mid_tilt=80.0)
  result = _collector(env).collect(TrialSpec("run", "motion", 4, 0))
  assert result.qualification.qualified
  assert result.qualification.fall_frames == 2
  assert result.qualification.max_anchor_z_error_m == 0.0


def test_an_endpoint_fall_fails_and_no_row_is_accepted() -> None:
  env = _Environment(clip_frames=2, endpoint_tilt=80.0)
  result = _collector(env).collect(TrialSpec("run", "motion", 4, 0))
  assert result.failed_attempt
  assert result.qualification.reason == "endpoint_tilt"
  assert result.accepted_rows == ()
  assert env.trial_reset_calls == 1


def test_a_rollout_without_an_endpoint_is_not_qualified() -> None:
  class _NeverEnds(_Environment):
    def post_step_evidence(self, result: int) -> PostStepBundle:
      bundle = super().post_step_evidence(result)
      return PostStepBundle(
        StepEvidence(True, terminated=result >= 2, truncated=result >= 2),
        bundle.timestamp,
        bundle.observation,
      )

  result = _collector(_NeverEnds(clip_frames=9)).collect(
    TrialSpec("run", "motion", 4, 0)
  )
  assert result.qualification.label == "incomplete"
  assert result.qualification.reason == "early_termination"


def test_remaining_transition_budget_stops_before_extra_action() -> None:
  env = _Environment(clip_frames=5)
  result = _collector(env).collect(
    TrialSpec("run", "motion", 4, 0), remaining_transitions=2
  )
  assert env.step_count == 2
  assert result.stopped_reason == "control_transition_budget"
  assert not result.qualification.qualified


def test_rows_carry_the_anchor_group_and_the_pose_provenance() -> None:
  env = _Environment(clip_frames=2)
  spec = TrialSpec("run", "motion", 4, 0, initial_pose="standing")
  result = _collector(env).collect(spec)
  assert env.resets == [(4, "standing", True)]
  assert [row.group_key for row in result.rows] == [
    f"motion:4:{frame}" for frame in range(-8, 2)
  ]
  for row in result.rows:
    assert row.provenance["initial_pose"] == "standing"
    assert row.provenance["initial_state_mode"] == "perturbed"
    assert row.provenance["pair_id"] == ""
    assert row.provenance["group_key_generated"] == "true"


def test_an_ou_trial_without_a_perturbation_records_its_mode() -> None:
  env = _Environment(clip_frames=2)
  spec = TrialSpec("run", "motion", 4, 0, initial_state_mode="reference")
  result = _collector(env).collect(spec)
  assert env.resets == [(4, "reference", False)]
  assert result.rows[0].provenance["initial_state_mode"] == "reference"


def test_snapshot_setup_failure_still_resets_after_trial() -> None:
  class _CaptureFailure(_Environment):
    def capture_initial_state(self) -> str:
      raise RuntimeError("snapshot capture failed")

  env = _CaptureFailure()
  with pytest.raises(RuntimeError, match="snapshot capture failed"):
    _collector(env).collect(
      TrialSpec("run", "motion", 4, 0, pair_id="pair", initial_state_id="initial")
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
  collector.collect(
    TrialSpec("run", "motion", 4, 0, pair_id="a", initial_state_id="same")
  )
  collector.collect(
    TrialSpec("run", "motion", 4, 0, pair_id="b", initial_state_id="same")
  )
  collector.collect(
    TrialSpec("run", "motion", 4, 0, "ou", pair_id="a", initial_state_id="same")
  )
  collector.collect(
    TrialSpec("run", "motion", 4, 0, "ou", pair_id="b", initial_state_id="same")
  )
  assert env.restored == ["snapshot-1", "snapshot-2"]


def test_failed_clean_partner_blocks_ou_trial() -> None:
  collector = _collector(_Environment(endpoint_tilt=80.0))
  clean, ou = _pair()
  collector.collect(clean)
  with pytest.raises(CollectorError, match="clean partner did not qualify"):
    collector.collect(ou)


def test_qualified_clean_partner_permits_ou_trial() -> None:
  collector = _collector(_Environment(clip_frames=2))
  clean, ou = _pair()
  clean_result = collector.collect(clean)
  assert clean_result.qualification.qualified
  ou_result = collector.collect(ou)
  assert ou_result.qualification.qualified
  assert ou_result.accepted_rows
  # The OU rollout perturbs every clip step and its clean partner never does;
  # the padding rows are history and carry no noise.
  assert all(np.any(row.ou_noise != 0.0) for row in ou_result.rows[8:])
  assert all(np.all(row.ou_noise == 0.0) for row in ou_result.rows[:8])
  assert all(np.all(row.ou_noise == 0.0) for row in clean_result.rows)


def test_qualified_rows_are_persisted_with_the_segment_verdict(tmp_path) -> None:
  contract = ActionContract(np.ones(31), np.zeros(31), np.ones(31), np.ones(31))
  store = AppendOnlyShardStore(tmp_path, max_rows_per_shard=300)
  result = DiffusionCollector(
    _Environment(clip_frames=2),
    FrozenVaePolicy(_Vae(), contract),
    store=store,
  ).collect(TrialSpec("run", "motion", 4, 0))
  persisted = tuple(store.iter_rows())
  assert len(persisted) == len(result.rows) == 10
  assert sum(row.segment_qualified for row in persisted) == len(result.accepted_rows)
  assert all(row.segment_qualified for row in persisted)
