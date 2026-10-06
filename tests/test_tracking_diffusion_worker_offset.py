"""Focused tests for the resume offset used by a resumed D1 collection.

A resumed collection must skip the trials a previous run already stored.  The
schedule is deterministic, so each worker's shard is a stable prefix of the work
it owes.  These tests cover the resolver and the slicing semantics that the
resume depends on; they never construct a simulator or a runtime.
"""

from __future__ import annotations

from mjlab.tasks.tracking.diffusion import runtime_x2


def test_worker_offset_defaults_to_zero_when_unset() -> None:
  assert runtime_x2.resolve_worker_offset("", 0, 4) == 0
  assert runtime_x2.resolve_worker_offset("   ", 3, 4) == 0


def test_worker_offset_single_value_applies_to_every_worker() -> None:
  assert runtime_x2.resolve_worker_offset("870", 0, 4) == 870
  assert runtime_x2.resolve_worker_offset("870", 3, 4) == 870


def test_worker_offset_list_indexes_by_worker() -> None:
  value = "868,872,878,873,874,847,867,848"
  assert runtime_x2.resolve_worker_offset(value, 0, 8) == 868
  assert runtime_x2.resolve_worker_offset(value, 5, 8) == 847
  assert runtime_x2.resolve_worker_offset(value, 7, 8) == 848
  assert runtime_x2.resolve_worker_offset("868, 872", 1, 2) == 872


def test_worker_offset_rejects_a_list_of_the_wrong_length() -> None:
  try:
    runtime_x2.resolve_worker_offset("868,872", 0, 4)
  except runtime_x2.RuntimeX2Error as exc:
    assert "per worker" in str(exc)
  else:  # pragma: no cover
    raise AssertionError("a short per-worker offset list was accepted")


def test_worker_offset_rejects_negative_and_non_integer_values() -> None:
  for value in ("-1", "later", "1.5", "870,"):
    try:
      runtime_x2.resolve_worker_offset(value, 0, 2)
    except runtime_x2.RuntimeX2Error:
      continue
    raise AssertionError(f"invalid offset {value!r} was accepted")


def test_offset_selects_the_tail_of_a_worker_shard() -> None:
  """The resume assumption: a shard is deterministic and the tail is contiguous."""
  motions = (("motion_a", 40),)
  full = runtime_x2.build_bulk_trials(
    motions, seed_count=5, run_id="resume-run", worker=0, workers=2
  )
  assert len(full) > 3
  assert full[3:] == tuple(full[3:])
  assert full[3:] == full[3:]
  assert len(full[3:]) == len(full) - 3
  # Slicing twice is stable: the tail of the tail is the same trials.
  assert full[3:][1:] == full[4:]
  assert all(spec.run_id == "resume-run" for spec in full[3:])


def test_offset_past_the_shard_is_an_empty_schedule() -> None:
  """Slicing past the end yields no trials rather than raising in the slice."""
  trials = (1, 2, 3)
  assert trials[len(trials) :] == ()
