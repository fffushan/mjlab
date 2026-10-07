import numpy as np
import pytest
import torch

from mjlab.tasks.tracking.distillation.branch_state import (
  BranchState,
  BranchStateError,
  BranchStateMismatch,
  restore_field_parity,
)
from mjlab.tasks.tracking.distillation.coverage_environment import (
  CoverageEnvironmentComponent,
  CoverageRuntimeError,
)
from mjlab.tasks.tracking.distillation.coverage_policy import CoveragePolicy
from mjlab.tasks.tracking.distillation.teacher_branches import (
  BranchBudget,
  BranchSeed,
  TeacherBranch,
)
from mjlab.utils.buffers import CircularBuffer, DelayBuffer


class FakeState:
  def __init__(self, value: float = 1.0):
    self.value = value
    self.fields = {
      "qpos": torch.tensor([[value, 2.0]]),
      "qvel": torch.tensor([[3.0, 4.0]]),
    }
    self.manager = {"count": 2}
    self.buffers = {"history": torch.tensor([[1.0, 2.0]])}
    self.obs = torch.tensor([9.0])
    self.action = torch.tensor([0.5])
    self.steps = 0

  def coverage_capture_fields(self, index=None):
    return self.fields, self.manager, self.buffers

  def coverage_restore_fields(self, fields, manager, buffers, index=None):
    self.fields = fields
    self.manager = manager
    self.buffers = buffers

  def coverage_capture_rng(self):
    return {"seed": 7}

  def coverage_policy_view(self):
    return self.obs

  def coverage_cached_policy_view(self, index):
    return self.obs.clone()

  def coverage_executed_action(self, index):
    return self.action

  def coverage_reconstruct_next_boundary(self, action):
    self.steps += 1

  def step(self, action):
    self.steps += 1
    return {"tracking_failure": self.steps == 1, "failure_reason": "tilt"}


def test_branch_state_owns_fields_and_restores_without_source_mutation():
  source = FakeState()
  state = BranchState.capture(source, metadata={"phase": 2})
  source.fields["qpos"][0, 0] = 99
  destination = FakeState(0)
  state.restore(destination)
  assert destination.fields["qpos"].tolist() == [[1.0, 2.0]]
  assert state.metadata == {"phase": 2}


def test_branch_state_rejects_nonfinite_and_field_corruption():
  source = FakeState()
  source.fields["qpos"][0, 0] = float("nan")
  with pytest.raises(BranchStateError, match="non-finite"):
    BranchState.capture(source)
  good = BranchState.capture(FakeState())
  bad = BranchState.capture(FakeState())
  bad.fields["qpos"][0, 0] = 3
  with pytest.raises(BranchStateMismatch, match="qpos"):
    restore_field_parity(good, bad)


def test_circular_history_repacks_at_destination_pointer_and_delay_alias_state():
  buf = CircularBuffer(max_len=3, batch_size=2, device="cpu")
  for value in range(4):
    buf.append(torch.tensor([[value], [100 + value]], dtype=torch.float32))
  state = buf.capture_state()
  other = CircularBuffer(max_len=3, batch_size=2, device="cpu")
  other.append(torch.tensor([[50.0], [60.0]]))
  other.restore_state(state)
  assert other.buffer[:, :, 0].tolist() == [[1.0, 2.0, 3.0], [101.0, 102.0, 103.0]]
  delay = DelayBuffer(min_lag=0, max_lag=2, batch_size=2)
  delay.append(torch.ones(2, 1))
  delay.set_lags(torch.tensor([2, 1]))
  delay_state = delay.capture_state()
  delay.restore_state(delay_state)
  assert delay.current_lags.tolist() == [2, 1]


def test_delay_invalid_generator_state_is_transactional():
  generator = torch.Generator(device="cpu").manual_seed(7)
  delay = DelayBuffer(min_lag=0, max_lag=2, batch_size=2, generator=generator)
  delay.append(torch.tensor([[1.0], [2.0]]))
  delay.append(torch.tensor([[3.0], [4.0]]))
  delay.set_lags(torch.tensor([2, 1]))
  state = delay.capture_state(torch.tensor([1]))
  state["generator_state"] = torch.zeros(1, dtype=torch.uint8)
  before_history = delay._buffer.buffer.clone()
  before_pointer = delay._buffer._pointer
  before_lags = delay.current_lags.clone()
  before_steps = delay._step_count.clone()
  before_phases = delay._phase_offsets.clone()
  before_rng = generator.get_state().clone()
  with pytest.raises(ValueError, match="generator state is invalid"):
    delay.restore_state(state, torch.tensor([0]))
  assert delay._buffer.buffer.equal(before_history)
  assert delay._buffer._pointer == before_pointer
  assert delay.current_lags.equal(before_lags)
  assert delay._step_count.equal(before_steps)
  assert delay._phase_offsets.equal(before_phases)
  assert generator.get_state().equal(before_rng)


def test_terminal_reconstruction_is_fail_closed_until_explicit_parity():
  source = FakeState()
  component = CoverageEnvironmentComponent(source, CoveragePolicy.off())
  assert not component.enabled
  enabled = CoveragePolicy.balance_recovery()
  component = CoverageEnvironmentComponent(source, enabled)
  component.preflight()
  with pytest.raises(CoverageRuntimeError, match="fail-closed"):
    component.reconstruct_terminal(
      component.capture_raw_reached_pre_reset(torch.tensor([0]), termination="done")[0],
      source,
    )


def test_teacher_branch_keeps_valid_labels_after_soft_failure_and_is_bounded():
  backend = FakeState()
  seed = BranchSeed(BranchState.capture(backend))
  branch = TeacherBranch(
    seed,
    BranchBudget(horizon_control_steps=4, max_transitions_after_first_failure=1),
    "b0",
  )
  result = branch.run(backend, lambda view: torch.tensor([1.0]))
  assert result.outcome == "failed"
  assert result.steps == 2
  assert len(result.labels) == 3
  assert result.first_failure_step == 1


def test_branch_rng_does_not_change_main_rng():
  import random

  random.seed(4)
  np.random.seed(4)
  torch.manual_seed(4)
  expected = (random.random(), float(np.random.random()), float(torch.rand(())))
  random.seed(4)
  np.random.seed(4)
  torch.manual_seed(4)
  backend = FakeState()
  branch = TeacherBranch(
    BranchSeed(BranchState.capture(backend)), BranchBudget(horizon_control_steps=1), "b"
  )
  branch.run(backend, lambda view: torch.tensor([1.0]), seed=99)
  observed = (random.random(), float(np.random.random()), float(torch.rand(())))
  assert observed == expected
