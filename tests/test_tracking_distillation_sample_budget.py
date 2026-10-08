"""CPU budget and public CLI dispatch regressions; no simulator or assets."""

from __future__ import annotations

import json
import math
import sys

import pytest
import torch

from mjlab.scripts import distill
from mjlab.tasks.tracking.distillation.sample_budget import resolve_sample_budget
from mjlab.tasks.tracking.distillation.training_config import TrainingConfig


def budget(**kwargs):
  options = dict(
    motion_count=1, collection_steps=32, minibatch_size=4096, accumulation_steps=8
  )
  options.update(kwargs)
  return resolve_sample_budget(**options)


@pytest.mark.parametrize("motions", [1, 8, 15, 30])
def test_budgets_scale_with_selected_motion_count(motions):
  resolved = budget(
    motion_count=motions,
    envs_per_motion=1024,
    draw_to_collection_ratio=0.5,
    replay_retention_iterations=2,
  )
  assert resolved.num_envs == motions * 1024
  assert resolved.fresh_rows == motions * 32768
  assert resolved.training_draws >= resolved.fresh_rows * 0.5
  assert resolved.training_draws - resolved.fresh_rows * 0.5 < 32768
  assert resolved.replay_capacity == motions * 65536
  assert resolved.effective_batch_size == 32768


def test_current_collection_can_keep_fixed_envs_and_derive_other_budgets():
  resolved = budget(
    motion_count=30,
    num_envs=28672,
    draw_to_collection_ratio=0.5,
    replay_retention_iterations=2,
  )
  assert resolved.fresh_rows == 917504
  assert resolved.updates_per_iteration == 14
  assert resolved.training_draws == 458752
  assert resolved.replay_capacity == 1835008


def test_legacy_totals_and_zero_collection_keep_original_behavior():
  defaults = budget()
  assert (
    defaults.num_envs,
    defaults.updates_per_iteration,
    defaults.replay_capacity,
  ) == (1, 1, 16384)
  explicit = budget(num_envs=28672, updates_per_iteration=4, replay_capacity=1966080)
  assert explicit.training_draws == 131072
  assert explicit.training_draws / explicit.fresh_rows == pytest.approx(1 / 7)
  offline = budget(collection_steps=0, updates_per_iteration=0)
  assert offline.fresh_rows == offline.training_draws == 0


def test_decimal_rounding_and_partial_update_round_up_without_changing_batch():
  exact = budget(
    num_envs=30,
    collection_steps=1,
    minibatch_size=1,
    accumulation_steps=1,
    draw_to_collection_ratio=0.1,
    replay_retention_iterations=0.1,
  )
  assert exact.updates_per_iteration == exact.replay_capacity == 3
  partial = budget(
    num_envs=3,
    collection_steps=1,
    minibatch_size=2,
    accumulation_steps=2,
    draw_to_collection_ratio=1.5,
    replay_retention_iterations=1.5,
  )
  assert partial.effective_batch_size == 4
  assert partial.updates_per_iteration == 2
  assert partial.training_draws == 8
  assert partial.replay_capacity == 5


@pytest.mark.parametrize(
  "options",
  [
    dict(num_envs=1, envs_per_motion=1),
    dict(updates_per_iteration=1, draw_to_collection_ratio=0.5),
    dict(replay_capacity=16384, replay_retention_iterations=2),
  ],
)
def test_explicit_and_derived_options_conflict_even_at_legacy_default(options):
  with pytest.raises(ValueError, match="mutually exclusive"):
    budget(**options)


@pytest.mark.parametrize(
  "field", ["draw_to_collection_ratio", "replay_retention_iterations"]
)
@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf, -math.inf, True])
def test_invalid_ratios_fail_closed(field, value):
  with pytest.raises(ValueError, match="finite and positive"):
    budget(**{field: value})


@pytest.mark.parametrize(
  "field",
  [
    "motion_count",
    "envs_per_motion",
    "minibatch_size",
    "accumulation_steps",
    "num_envs",
    "replay_capacity",
  ],
)
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_integer_budgets_fail_closed(field, value):
  with pytest.raises(ValueError, match="integer"):
    budget(**{field: value})


@pytest.mark.parametrize(
  "options",
  [
    dict(collection_steps=0, draw_to_collection_ratio=1),
    dict(collection_steps=0, replay_retention_iterations=2),
  ],
)
def test_derived_ratios_require_new_collection(options):
  with pytest.raises(ValueError, match="collection_steps > 0"):
    budget(**options)


def test_report_uses_live_weighted_allocations_and_does_not_consume_rng():
  rng = torch.get_rng_state().clone()
  resolved = budget(
    motion_count=2,
    envs_per_motion=8,
    collection_steps=2,
    minibatch_size=4,
    accumulation_steps=1,
    draw_to_collection_ratio=1,
    replay_retention_iterations=2,
  )
  report = resolved.as_dict(
    teacher_ids=("heavy", "light"),
    slot_counts=(12, 4),
    weights=(3, 1),
    replay_quotas=(48, 16),
  )
  assert torch.equal(rng, torch.get_rng_state())
  assert report["requested"]["envs_per_motion"] == 8
  assert [item["fresh_rows_per_iteration"] for item in report["motions"]] == [24, 8]
  assert [
    item["expected_training_draws_per_iteration"] for item in report["motions"]
  ] == [24, 8]
  assert all(
    item["expected_draw_to_collection_ratio"] == 1 for item in report["motions"]
  )
  assert all(item["replay_retention_iterations"] == 2 for item in report["motions"])
  assert "total" in report["draw_budget_scope"]
  assert "not unique-row" in report["coverage_claim"]
  json.dumps(report, allow_nan=False)


def test_report_exposes_rounding_and_zero_collection_without_fake_coverage():
  resolved = budget(
    motion_count=2,
    num_envs=3,
    collection_steps=1,
    minibatch_size=2,
    accumulation_steps=1,
    draw_to_collection_ratio=1,
    replay_retention_iterations=2,
  )
  report = resolved.as_dict(
    teacher_ids=("a", "b"),
    slot_counts=(2, 1),
    weights=(1, 1),
    replay_quotas=(3, 3),
  )
  assert report["resolved"]["draw_to_collection_ratio"] == pytest.approx(4 / 3)
  assert [item["expected_draw_to_collection_ratio"] for item in report["motions"]] == [
    1,
    2,
  ]
  assert [item["replay_retention_iterations"] for item in report["motions"]] == [1.5, 3]
  offline = budget(collection_steps=0).as_dict(
    teacher_ids=("a",), slot_counts=(1,), weights=(1,), replay_quotas=(16384,)
  )
  assert offline["resolved"]["draw_to_collection_ratio"] is None
  assert offline["motions"][0]["replay_retention_iterations"] is None
  json.dumps(offline, allow_nan=False)


def test_report_refuses_wrong_live_allocations():
  resolved = budget()
  with pytest.raises(ValueError, match="align"):
    resolved.as_dict(teacher_ids=(), slot_counts=(), weights=(), replay_quotas=())
  with pytest.raises(ValueError, match="totals"):
    resolved.as_dict(
      teacher_ids=("a",), slot_counts=(2,), weights=(1,), replay_quotas=(16384,)
    )


def invoke(monkeypatch, argv):
  monkeypatch.setattr(sys, "argv", ["distill", "train", *argv])
  with pytest.raises(SystemExit) as exc:
    distill.main()
  return exc.value.code


@pytest.mark.parametrize(
  "selection,motions,path",
  [
    ([], 1, "_train_single"),
    (["--teacher-id", "a"], 1, "_train_single"),
    (["--teacher-ids", "('a',)"], 1, "_train_cohort"),
    (["--teacher-ids", "('a','b','c')"], 3, "_train_cohort"),
  ],
)
def test_public_cli_derived_budget_dispatch(monkeypatch, selection, motions, path):
  calls = []
  monkeypatch.setattr(distill, path, lambda **kwargs: calls.append(kwargs) or 0)
  assert (
    invoke(
      monkeypatch,
      [
        *selection,
        "--envs-per-motion",
        "1024",
        "--draw-to-collection-ratio",
        "0.5",
        "--replay-retention-iterations",
        "2",
        "--minibatch-size",
        "4096",
        "--accumulation-steps",
        "8",
      ],
    )
    == 0
  )
  assert len(calls) == 1
  call = calls[0]
  assert call["num_envs"] == motions * 1024
  assert call["updates_per_iteration"] == math.ceil(motions / 2)
  assert call["replay_capacity"] == motions * 65536
  assert call["sample_budget"].motion_count == motions


@pytest.mark.parametrize(
  "argv",
  [
    ["--num-envs", "1", "--envs-per-motion", "1"],
    ["--updates-per-iteration", "1", "--draw-to-collection-ratio", "0.5"],
    ["--replay-capacity", "16384", "--replay-retention-iterations", "2"],
    ["--draw-to-collection-ratio", "nan"],
    ["--envs-per-motion", "0"],
    ["--collection-steps", "0", "--draw-to-collection-ratio", "1"],
  ],
)
def test_public_cli_rejects_invalid_budget_before_construction(
  monkeypatch, capsys, argv
):
  def forbidden(**kwargs):
    raise AssertionError("invalid budget reached lifecycle construction")

  monkeypatch.setattr(distill, "_train_single", forbidden)
  monkeypatch.setattr(distill, "_train_cohort", forbidden)
  assert invoke(monkeypatch, argv) == 1
  captured = capsys.readouterr()
  assert captured.out == ""
  assert "[FAIL]" in captured.err


def test_public_cli_uses_shared_throughput_defaults(monkeypatch):
  calls = []
  monkeypatch.setattr(
    distill, "_train_cohort", lambda **kwargs: calls.append(kwargs) or 0
  )
  assert (
    invoke(
      monkeypatch,
      [
        "--teacher-ids",
        "('a','b')",
        "--envs-per-motion",
        "1024",
        "--draw-to-collection-ratio",
        "1",
        "--replay-retention-iterations",
        "2",
      ],
    )
    == 0
  )
  call = calls[0]
  assert call["minibatch_size"] == TrainingConfig().minibatch_size == 16_384
  assert call["accumulation_steps"] == TrainingConfig().accumulation_steps == 2
  assert call["sample_budget"].effective_batch_size == 32_768
  assert call["updates_per_iteration"] == 2


def test_worker_divisibility_is_checked_after_deriving_envs(monkeypatch, capsys):
  calls = []
  monkeypatch.setattr(
    distill, "_train_cohort", lambda **kwargs: calls.append(kwargs) or 0
  )
  base = ["--teacher-ids", "('a','b','c')", "--worker-devices", "('cpu','cpu')"]
  assert invoke(monkeypatch, [*base, "--envs-per-motion", "2"]) == 0
  assert calls[0]["num_envs"] == 6
  assert calls[0]["worker_devices"] == ("cpu", "cpu")
  calls.clear()
  assert invoke(monkeypatch, [*base, "--envs-per-motion", "1"]) == 1
  assert calls == []
  assert "does not split evenly" in capsys.readouterr().err
