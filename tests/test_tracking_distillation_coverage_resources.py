"""Parent CPU-only P3 regressions; no live environment or hardware imports."""

import random

import pytest
import torch

from mjlab.tasks.tracking.distillation.branch_state import BranchState
from mjlab.tasks.tracking.distillation.coverage_environment import (
  CoverageEnvironmentComponent,
  SnapshotCandidate,
  TerminalCapture,
)
from mjlab.tasks.tracking.distillation.coverage_policy import CoveragePolicy
from mjlab.tasks.tracking.distillation.teacher_branches import (
  BranchBudget,
  BranchSeed,
  CoverageResourceError,
  TeacherBranch,
  TeacherBranchManager,
)


class Backend:
  resident_world_count = 1
  resident_bytes = 16

  def __init__(self, *, close_rng=False):
    self.steps = 0
    self.closed = False
    self.close_rng = close_rng
    self.obs = torch.zeros(1)

  def coverage_restore_fields(self, fields, managers, buffers, index):
    self.steps = 0

  def coverage_policy_view(self):
    return self.obs

  def step(self, action):
    self.steps += 1
    return {"tracking_failure": False}

  def close(self):
    self.closed = True
    if self.close_rng:
      random.random()


def candidate(ordinal, kind="finite_post_failure", capture_fn=None, estimate=32):
  if capture_fn is None:

    def make_capture():
      return TerminalCapture(
        BranchState({"qpos": torch.zeros(1)}),
        torch.zeros(1),
        torch.zeros(1),
        None,
      )

    capture_fn = make_capture

  return SnapshotCandidate(
    candidate_id=f"{kind}:{ordinal}",
    source_kind=kind,
    motion_id=0,
    phase_bin=0,
    severity_band="moderate",
    family_id=f"family{ordinal}",
    estimated_bytes=estimate,
    capture=capture_fn,
  )


def test_candidate_reservations_are_simultaneous_not_last_call_capacity():
  component = CoverageEnvironmentComponent(object(), CoveragePolicy.balance_recovery())
  component.admit_snapshot_candidates([candidate(i) for i in range(60)])
  component.admit_snapshot_candidates(
    [candidate(i, "student_difficult") for i in range(205)]
  )
  records = list(component._terminal_records.values())
  terminal = sum(r.candidate.source_kind == "finite_post_failure" for r in records)
  prestep = sum(r.candidate.source_kind == "student_difficult" for r in records)
  assert (terminal, prestep) == (51, 205)


def test_old_captures_not_returned_as_new_fresh_rows():
  component = CoverageEnvironmentComponent(object(), CoveragePolicy.balance_recovery())
  first = component.admit_snapshot_candidates([candidate(0)])
  second = component.admit_snapshot_candidates([candidate(1)])
  assert len(first) == 1
  assert {r.candidate.candidate_id for r in second} == {candidate(1).candidate_id}


def test_combined_estimate_checked_before_expensive_callback():
  component = CoverageEnvironmentComponent(object(), CoveragePolicy.balance_recovery())
  component._terminal_bytes_cap = 32
  component.admit_snapshot_candidates([candidate(0)])
  calls = []

  def expensive():
    calls.append(1)
    return TerminalCapture(BranchState({}), torch.zeros(1), torch.zeros(1), None)

  component.admit_snapshot_candidates([candidate(1, capture_fn=expensive, estimate=32)])
  assert not calls


def test_actual_owned_bytes_overflow_is_rejected_after_copy_without_retention():
  component = CoverageEnvironmentComponent(object(), CoveragePolicy.balance_recovery())
  component._terminal_bytes_cap = 8

  def large_capture():
    return TerminalCapture(
      BranchState({"large": torch.zeros(4)}), torch.zeros(1), torch.zeros(1), None
    )

  assert (
    component.admit_snapshot_candidates(
      [candidate(0, capture_fn=large_capture, estimate=0)]
    )
    == ()
  )
  assert component._terminal_bytes == 0
  assert component.counters["byte_overflow"] == 1


def test_horizon_retains_finite_endpoint():
  backend = Backend()
  branch = TeacherBranch(
    BranchSeed(BranchState({})), BranchBudget(horizon_control_steps=2), "b"
  )
  result = branch.run(backend, lambda _: torch.zeros(1))
  assert result.steps == 2
  assert len(result.labels) == 3


def test_cleanup_rng_is_in_branch_isolation_scope():
  manager = TeacherBranchManager(BranchBudget(horizon_control_steps=1))
  manager.admit(BranchSeed(BranchState({})))
  random.seed(43)
  before = random.getstate()
  manager.run_available(
    lambda: Backend(close_rng=True),
    lambda _: torch.zeros(1),
    actual_main_row_steps=100,
  )
  assert random.getstate() == before


def test_aggregate_resident_bytes_exceeding_owner_cap_rejected():
  manager = TeacherBranchManager(
    BranchBudget(horizon_control_steps=1, resident_bytes_cap=100)
  )
  for _ in range(2):
    manager.admit(BranchSeed(BranchState({})))
  created = []

  def factory():
    b = Backend()
    b.resident_bytes = 60
    created.append(b)
    return b

  with pytest.raises(CoverageResourceError, match="resident|allocation"):
    manager.run_available(factory, lambda _: torch.zeros(1), actual_main_row_steps=100)
  assert all(b.closed for b in created)


def test_partial_factory_failure_closes_already_created_owner():
  manager = TeacherBranchManager(
    BranchBudget(horizon_control_steps=1, resident_bytes_cap=100)
  )
  for _ in range(2):
    manager.admit(BranchSeed(BranchState({})))
  created = []

  def factory():
    b = Backend()
    b.resident_bytes = 200 if created else 60
    created.append(b)
    return b

  with pytest.raises(CoverageResourceError):
    manager.run_available(factory, lambda _: torch.zeros(1), actual_main_row_steps=100)
  assert all(b.closed for b in created)


def test_zero_main_budget_discards_pending_seeds_with_reason():
  manager = TeacherBranchManager(BranchBudget(horizon_control_steps=1))
  manager.admit(BranchSeed(BranchState({})))
  assert (
    manager.run_available(
      lambda: Backend(), lambda _: torch.zeros(1), actual_main_row_steps=0
    )
    == ()
  )
  assert not manager._seeds
  assert manager.counters["budget_censored"] == 1


def test_start_reservations_borrow_unused_category_and_reuse_32_slots():
  manager = TeacherBranchManager(
    BranchBudget(horizon_control_steps=1, max_main_row_steps_fraction=1.0)
  )
  for index in range(64):
    assert manager.admit(
      BranchSeed(
        BranchState({}),
        source_kind="finite_post_failure" if index < 13 else "student_difficult",
      )
    )
  created = []

  def factory():
    backend = Backend()
    created.append(backend)
    return backend

  results = manager.run_available(
    factory, lambda _: torch.zeros(1), actual_main_row_steps=64
  )
  assert len(results) == 64
  assert len(created) == 32
  assert all(backend.closed for backend in created)
  assert manager.counters["resident_slot_reuse"] == 32
  assert manager.counters["borrowed_start_reservations"] == 0


def test_all_terminal_starts_borrow_the_pre_step_reservation():
  manager = TeacherBranchManager(
    BranchBudget(horizon_control_steps=1, max_main_row_steps_fraction=1.0)
  )
  for _ in range(64):
    manager.admit(BranchSeed(BranchState({}), source_kind="finite_post_failure"))
  results = manager.run_available(
    lambda: Backend(), lambda _: torch.zeros(1), actual_main_row_steps=64
  )
  assert len(results) == 64
  assert manager.counters["borrowed_start_reservations"] == 51


def test_padded_world_cost_is_counted_and_underreported_worlds_rejected():
  class PaddedBackend(Backend):
    resident_world_count = 4

  manager = TeacherBranchManager(
    BranchBudget(horizon_control_steps=2, max_main_row_steps_fraction=0.1)
  )
  manager.admit(BranchSeed(BranchState({})))
  result = manager.run_available(
    lambda: PaddedBackend(), lambda _: torch.zeros(1), actual_main_row_steps=10
  )[0]
  assert result.budget_censored
  assert result.steps == 0

  manager.admit(BranchSeed(BranchState({})))
  with pytest.raises(CoverageResourceError, match="underreport"):
    manager.run_available(
      lambda: PaddedBackend(),
      lambda _: torch.zeros(1),
      actual_main_row_steps=100,
      active_worlds=1,
    )


def test_missing_backend_resource_types_fail_closed():
  class MissingCostBackend(Backend):
    resident_bytes = "large"

  manager = TeacherBranchManager(BranchBudget(horizon_control_steps=1))
  manager.admit(BranchSeed(BranchState({})))
  with pytest.raises(CoverageResourceError, match="resident_bytes"):
    manager.run_available(
      lambda: MissingCostBackend(), lambda _: torch.zeros(1), actual_main_row_steps=100
    )


def test_inherited_failure_starts_at_branch_step_zero():
  manager = TeacherBranchManager(
    BranchBudget(horizon_control_steps=4, max_transitions_after_first_failure=1)
  )
  manager.admit(BranchSeed(BranchState({}), source_failed=True))
  result = manager.run_available(
    lambda: Backend(), lambda _: torch.zeros(1), actual_main_row_steps=100
  )[0]
  assert result.first_failure_step == 0
  assert result.steps == 1
  assert result.outcome == "failed"
  assert len(result.labels) == 2


def test_nonfinite_endpoint_is_invalid_without_fabricated_label():
  class NonfiniteEndpoint(Backend):
    def step(self, action):
      self.steps += 1
      self.obs.fill_(float("nan"))
      return {"reference_complete": True}

  branch = TeacherBranch(
    BranchSeed(BranchState({})), BranchBudget(horizon_control_steps=2), "b"
  )
  result = branch.run(NonfiniteEndpoint(), lambda _: torch.zeros(1))
  assert result.outcome == "invalid"
  assert result.stop_reason == "nonfinite_endpoint"
  assert len(result.labels) == 1


def test_teacher_and_backend_infrastructure_errors_propagate_and_close():
  manager = TeacherBranchManager(BranchBudget(horizon_control_steps=1))
  manager.admit(BranchSeed(BranchState({})))
  created = []

  def factory():
    backend = Backend()
    created.append(backend)
    return backend

  def bad_teacher(_):
    raise RuntimeError("teacher infrastructure")

  with pytest.raises(RuntimeError, match="teacher infrastructure"):
    manager.run_available(factory, bad_teacher, actual_main_row_steps=100)
  assert all(backend.closed for backend in created)

  class BadBackend(Backend):
    def step(self, action):
      raise RuntimeError("backend infrastructure")

  manager.admit(BranchSeed(BranchState({})))
  with pytest.raises(RuntimeError, match="backend infrastructure"):
    manager.run_available(
      lambda: BadBackend(), lambda _: torch.zeros(1), actual_main_row_steps=100
    )


def test_nonfinite_teacher_output_is_invalid_without_backend_step():
  backend = Backend()
  branch = TeacherBranch(
    BranchSeed(BranchState({})), BranchBudget(horizon_control_steps=2), "bad"
  )
  result = branch.run(backend, lambda _: torch.tensor([float("nan")]))
  assert result.outcome == "invalid"
  assert result.stop_reason == "nonfinite_teacher_output"
  assert result.steps == 0
  assert backend.steps == 0


def test_clip_endpoint_without_next_reference_keeps_raw_endpoint_unlabeled():
  class ClipEndBackend(Backend):
    def step(self, action):
      self.steps += 1
      return {"reference_complete": True, "next_reference_available": False}

  branch = TeacherBranch(
    BranchSeed(BranchState({})), BranchBudget(horizon_control_steps=2), "clip"
  )
  result = branch.run(ClipEndBackend(), lambda _: torch.zeros(1))
  assert result.outcome == "completed"
  assert result.stop_reason == "reference_complete"
  assert len(result.labels) == 1
  assert result.endpoint_reason == "no_next_reference"
  assert result.endpoint_observation is not None


def test_soft_failure_does_not_call_push_or_reset_and_keeps_labels():
  class SoftFailureBackend(Backend):
    pushes = 0
    resets = 0

    def push(self):
      self.pushes += 1

    def reset(self):
      self.resets += 1

    def step(self, action):
      self.steps += 1
      return {"tracking_failure": True, "failure_reason": "tilt"}

  backend = SoftFailureBackend()
  branch = TeacherBranch(
    BranchSeed(BranchState({})),
    BranchBudget(horizon_control_steps=4, max_transitions_after_first_failure=1),
    "soft",
  )
  result = branch.run(backend, lambda _: torch.zeros(1))
  assert result.outcome == "failed"
  assert len(result.labels) == 3
  assert backend.pushes == 0
  assert backend.resets == 0
