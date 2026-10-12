from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from tracking_diffusion_fixtures import make_state

from mjlab.tasks.tracking.diffusion import (
  ActionContract,
  CollectionResult,
  DiffusionCollector,
  FrozenPpoRecoveryPolicy,
  FrozenVaePolicy,
  PostStepBundle,
  PostStepBundleBatch,
  PpoObservation,
  PpoObservationBatch,
  QualificationConfig,
  StepEvidence,
  StepEvidenceBatch,
  TrialSlotScheduler,
  TrialSpec,
  VaeObservation,
  VaeObservationBatch,
  WorldState,
)


class _Schema:
  reference_dim = 68
  conditioning_dim = 99


class _Model(torch.nn.Module):
  schema = _Schema()

  def encode(self, reference: torch.Tensor):
    mu = reference[:, :1].repeat(1, 32)
    return mu, torch.zeros_like(mu)

  def decode(self, latent: torch.Tensor, conditioning: torch.Tensor):
    return latent[:, :1].repeat(1, 31) + conditioning[:, :1] * 0.0


class _ScalarEnv:
  def __init__(self) -> None:
    self.step_count = 0
    self.phase = "vae"

  def reset(self, *, seed: int, motion_id: str, start_frame: int) -> None:
    assert (seed, motion_id, start_frame) in {(7, "a", 3), (8, "b", 4)}
    self.step_count = 0
    self.phase = "vae"

  def observe_vae(self) -> VaeObservation:
    return VaeObservation(
      make_state(self.step_count),
      np.full(68, 0.25),
      np.full(99, 0.5),
      "a",
      self.step_count,
      0.0,
      anchor_z_error=0.0,
      gravity_z_error=0.0,
      end_effector_z_error=0.0,
    )

  def observe_ppo(self) -> PpoObservation:
    return PpoObservation(
      make_state(self.step_count), np.zeros(102), "a", self.step_count, 0.0
    )

  def step(self, action: np.ndarray) -> int:
    assert action.shape == (31,)
    self.step_count += 1
    return self.step_count

  def post_step_evidence(self, result: int) -> PostStepBundle:
    boundary = self.phase == "vae" and result == 2
    return PostStepBundle(
      StepEvidence(True, reference_boundary=boundary, clip_ended=boundary),
      float(result),
      self.observe_vae(),
    )

  def switch_to_ppo(self) -> None:
    self.phase = "ppo"

  def capture_initial_state(self) -> str:
    return "initial"

  def restore_initial_state(self, snapshot: object) -> None:
    assert snapshot == "initial"

  def reset_after_trial(self) -> None:
    pass


class _BatchEnv:
  num_envs = 1

  def __init__(self) -> None:
    self.rows = [_ScalarEnv() for _ in range(self.num_envs)]

  @staticmethod
  def _vae(values: list[VaeObservation]) -> VaeObservationBatch:
    return VaeObservationBatch(
      WorldState.stack(value.state for value in values),
      np.stack([value.reference for value in values]),
      np.stack([value.conditioning for value in values]),
      tuple(value.motion_id for value in values),
      np.asarray([value.reference_frame for value in values]),
      np.asarray([value.reference_phase for value in values]),
      anchor_z_error=np.zeros(len(values)),
      gravity_z_error=np.zeros(len(values)),
      end_effector_z_error=np.zeros(len(values)),
    )

  def reset_batch(self, specs, *, env_ids):
    for row, spec in zip(env_ids.tolist(), specs, strict=True):
      self.rows[row].reset(
        seed=spec.seed, motion_id=spec.motion_id, start_frame=spec.start_frame
      )

  def observe_vae_batch(self, env_ids):
    return self._vae([self.rows[row].observe_vae() for row in env_ids.tolist()])

  def observe_ppo_batch(self, env_ids):
    values = [self.rows[row].observe_ppo() for row in env_ids.tolist()]
    return PpoObservationBatch(
      WorldState.stack(value.state for value in values),
      np.stack([value.observation for value in values]),
      tuple(value.motion_id for value in values),
      np.asarray([value.reference_frame for value in values]),
      np.asarray([value.reference_phase for value in values]),
    )

  def step_batch(self, action):
    return [self.rows[row].step(action[row]) for row in range(self.num_envs)]

  def post_step_evidence_batch(self, env_ids):
    bundles = [
      self.rows[row].post_step_evidence(self.rows[row].step_count)
      for row in env_ids.tolist()
    ]
    return PostStepBundleBatch(
      StepEvidenceBatch.from_values([bundle.evidence for bundle in bundles]),
      np.asarray([bundle.timestamp for bundle in bundles]),
      self._vae([bundle.observation for bundle in bundles]),
    )

  def switch_to_ppo_batch(self, env_ids):
    for row in env_ids.tolist():
      self.rows[row].switch_to_ppo()

  def capture_initial_state_batch(self, env_ids):
    return tuple(self.rows[row].capture_initial_state() for row in env_ids.tolist())

  def restore_initial_state_batch(self, env_ids, snapshots):
    for row, snapshot in zip(env_ids.tolist(), snapshots, strict=True):
      self.rows[row].restore_initial_state(snapshot)

  def reset_after_trial_batch(self, env_ids):
    for row in env_ids.tolist():
      self.rows[row].reset_after_trial()


def _policies():
  contract = ActionContract(np.ones(31), np.zeros(31), np.ones(31), np.ones(31))
  return (
    FrozenVaePolicy(_Model(), contract),
    FrozenPpoRecoveryPolicy(
      lambda observation: np.zeros((observation.shape[0], 31)), contract
    ),
  )


def test_slot_scheduler_replaces_rows_and_admits_pair_locally() -> None:
  clean_a = TrialSpec("run", "a", 7, 3, pair_id="a", initial_state_id="ia")
  ou_a = TrialSpec("run", "a", 7, 3, "ou", pair_id="a", initial_state_id="ia")
  clean_b = TrialSpec("run", "b", 8, 4, pair_id="b", initial_state_id="ib")
  ou_b = TrialSpec("run", "b", 8, 4, "ou", pair_id="b", initial_state_id="ib")
  scheduler = TrialSlotScheduler((clean_a, ou_a, clean_b, ou_b), 2)
  started = scheduler.fill()
  assert {spec.pair_id for spec in started.values()} == {"a", "b"}
  clean_results = []
  for slot, spec in started.items():
    clean_results.append(
      (
        slot,
        type(
          "Result",
          (),
          {"spec": spec, "qualification": type("Q", (), {"qualified": True})()},
        )(),
      )
    )
  replacements = {}
  for slot, result in clean_results:
    replacements.update(scheduler.complete(slot, result))
  assert {spec.phase for spec in replacements.values()} == {"ou"}
  assert {spec.pair_id for spec in replacements.values()} == {"a", "b"}


def test_batched_rows_are_bit_identical_to_single_row_rows() -> None:
  config = QualificationConfig(control_steps=4, retained_steps=2)
  scalar_vae, scalar_ppo = _policies()
  scalar = DiffusionCollector(
    _ScalarEnv(), scalar_vae, scalar_ppo, qualification=config
  )
  clean = TrialSpec("run", "a", 7, 3, pair_id="pair", initial_state_id="state")
  ou = TrialSpec("run", "a", 7, 3, "ou", pair_id="pair", initial_state_id="state")
  scalar_clean = scalar.collect(clean)
  single = scalar.collect(ou)

  batch_vae, batch_ppo = _policies()
  batch = DiffusionCollector(_BatchEnv(), batch_vae, batch_ppo, qualification=config)
  result = batch.collect_batched((clean, ou), slots=1)
  batch_clean = next(item for item in result.results if item.spec.phase == "clean")
  for left, right in zip(scalar_clean.rows, batch_clean.rows, strict=True):
    np.testing.assert_array_equal(left.executed_action, right.executed_action)
    np.testing.assert_array_equal(right.ou_noise, np.zeros(31))
  batched = next(item for item in result.results if item.spec.phase == "ou")
  assert len(single.rows) == len(batched.rows)
  for left, right in zip(single.rows, batched.rows, strict=True):
    assert left.identity == right.identity
    assert left.controller == right.controller
    assert left.tick == right.tick
    assert left.timestamp == right.timestamp
    assert left.motion_id == right.motion_id
    for field in (
      "previous_executed_action",
      "clean_action",
      "ou_noise",
      "executed_action",
      "latent",
    ):
      left_value = getattr(left, field)
      right_value = getattr(right, field)
      if left_value is None:
        assert right_value is None
      else:
        np.testing.assert_array_equal(left_value, right_value)


def test_batched_store_receives_one_write_for_multiple_trials(tmp_path) -> None:
  vae, ppo = _policies()
  calls = []

  class _Store:
    def append(self, rows):
      values = tuple(rows)
      calls.append(values)
      return ("shard-000000.npz",)

  collector = DiffusionCollector(
    _BatchEnv(),
    vae,
    ppo,
    qualification=QualificationConfig(control_steps=2, retained_steps=1),
    store=_Store(),
  )
  specs = (
    TrialSpec("run", "a", 7, 3, pair_id="a", initial_state_id="ia", group_key="a"),
    TrialSpec(
      "run", "a", 7, 3, "ou", pair_id="a", initial_state_id="ia", group_key="a"
    ),
  )
  result = collector.collect_batched(specs, slots=1)
  assert len(calls) == 1
  assert result.store_files == ("shard-000000.npz",)
  assert all(isinstance(item, CollectionResult) for item in result.results)
  assert len(calls[0]) == sum(
    len(item.rows) for item in result.results if isinstance(item, CollectionResult)
  )
  assert all(row.group_key == "a" for row in calls[0])


def test_multi_slot_terminal_reset_and_idle_tail_do_not_block_other_rows() -> None:
  class _StrictBatchEnv(_BatchEnv):
    num_envs = 4

    def __init__(self):
      super().__init__()
      self.pending = np.zeros(4, dtype=bool)
      self.active_masks = []
      self.resets = []

    def set_active_env_ids(self, ids):
      self.active = np.zeros(4, dtype=bool)
      self.active[ids] = True
      self.active_masks.append(self.active.copy())

    def step_batch(self, action):
      assert not self.pending.any(), "terminal row was not reset"
      result = super().step_batch(action)
      if self.active[2] and self.rows[2].step_count == 1:
        self.pending[2] = True
      # Without the idle-row mask, row 3 would terminate and block this batch.
      if self.active[3] and self.rows[3].step_count >= 1:
        self.pending[3] = True
      return result

    def post_step_evidence_batch(self, ids):
      bundle = super().post_step_evidence_batch(ids)
      return replace(
        bundle, evidence=replace(bundle.evidence, terminated=self.pending[ids])
      )

    def reset_after_trial_batch(self, ids):
      self.resets.extend(ids.tolist())
      self.pending[ids] = False
      super().reset_after_trial_batch(ids)

  env = _StrictBatchEnv()
  vae, ppo = _policies()
  collector = DiffusionCollector(
    env, vae, ppo, qualification=QualificationConfig(control_steps=4, retained_steps=2)
  )
  specs = tuple(TrialSpec(f"run{i}", "a", 7, 3) for i in range(3))
  results = collector.collect_batched(specs, slots=4).results
  assert all(isinstance(result, CollectionResult) for result in results)
  assert [
    len(result.rows) for result in results if isinstance(result, CollectionResult)
  ] == [4, 4, 1]
  assert results[2].qualification.reason == "early_termination"
  assert results[0].qualification.qualified
  assert results[1].qualification.qualified
  assert not env.pending.any()
  assert sorted(env.resets) == [0, 1, 2]
  assert all(not mask[3] for mask in env.active_masks)


def test_batch_policy_methods_accept_selected_subsets() -> None:
  vae, ppo = _policies()
  state = WorldState.stack([make_state(0), make_state(1)])
  vae_observation = VaeObservationBatch(
    state,
    np.full((2, 68), 0.25),
    np.full((2, 99), 0.5),
    ("a", "a"),
    np.asarray([0, 1]),
    np.zeros(2),
  )
  latent, action = vae.batch_action(vae_observation.select(np.asarray([1])))
  assert latent.shape == (1, 32)
  assert action.shape == (1, 31)
  ppo_observation = PpoObservationBatch(
    state,
    np.zeros((2, 102)),
    ("a", "a"),
    np.asarray([0, 1]),
    np.zeros(2),
  )
  recovery = ppo.batch_action(
    ppo_observation.select(np.asarray([1])), np.zeros((1, 31))
  )
  assert recovery.shape == (1, 31)
