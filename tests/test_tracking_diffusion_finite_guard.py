"""Focused tests for the deferred finiteness guard.

The immediate form of the check reads a CUDA scalar per call, which the step
profile measured at about 170 device synchronisations per update. The guard
keeps the same detection inside the same update while paying a small number of
reads, and it must still name the offending tensor.
"""

from __future__ import annotations

import pytest
import torch

from mjlab.tasks.tracking.diffusion.trainer import (
  TrainerError,
  _finite_tensor,
  _FiniteGuard,
)


def test_finite_values_pass() -> None:
  guard = _FiniteGuard()
  guard.check(torch.ones(4), "a")
  guard.check(torch.zeros(2, 3), "b")
  guard.flush()  # must not raise


def test_checks_are_collected_then_cleared_in_one_flush() -> None:
  guard = _FiniteGuard()
  for index in range(5):
    guard.check(torch.ones(3), f"t{index}")
  assert len(guard._pending) == 5, "checks must be collected, not read one by one"
  guard.flush()
  assert guard._pending == [], "a flush must clear the collected checks"
  guard.flush()  # idempotent and a no-op when empty


def test_flush_names_the_offending_tensor() -> None:
  guard = _FiniteGuard()
  guard.check(torch.ones(3), "good")
  guard.check(torch.tensor([1.0, float("nan")]), "bad gradient")
  with pytest.raises(TrainerError, match="bad gradient"):
    guard.flush()


def test_flush_reports_an_unnamed_case_after_a_failure() -> None:
  guard = _FiniteGuard()
  guard.check(torch.tensor([float("inf")]), "with inf")
  with pytest.raises(TrainerError, match="with inf"):
    guard.flush()


def test_guard_matches_the_immediate_check() -> None:
  for value in (
    torch.ones(4),
    torch.full((4,), float("nan")),
    torch.full((2,), float("inf")),
  ):
    immediate_failed = False
    try:
      _finite_tensor(value, "value")
    except TrainerError:
      immediate_failed = True

    guard = _FiniteGuard()
    guard.check(value, "value")
    deferred_failed = False
    try:
      guard.flush()
    except TrainerError:
      deferred_failed = True

    assert immediate_failed == deferred_failed, f"behaviour differs for {value}"


def test_empty_values_are_skipped() -> None:
  guard = _FiniteGuard()
  guard.check(torch.empty(0), "empty")
  assert guard._pending == []
  guard.flush()


def test_cuda_flags_stay_unread_until_flush() -> None:
  if not torch.cuda.is_available():
    pytest.skip("no CUDA device")
  guard = _FiniteGuard()
  for index in range(4):
    guard.check(torch.ones(8, device="cuda"), f"c{index}")
  assert all(flag.is_cuda for _, flag in guard._pending), (
    "flags must stay on the device"
  )
  guard.flush()
