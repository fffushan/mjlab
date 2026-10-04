"""CPU-only D4 integration tests: opt-in task conditioning on the D3 seam.

These tests exercise the wiring only: strict task loading, the plan/sampler
hooks in the frozen handover seam, the CLI preflight gate and the preparation
tool.  The runtime, VAE and denoiser are small deterministic fakes; no
simulator, checkpoint, physics, GPU or hardware operation is involved and no
physical or model-quality claim is made.
"""

from __future__ import annotations

import json
import math
import sys
from argparse import Namespace
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, cast

import numpy as np
import pytest
import torch
from tracking_diffusion_fixtures import make_row, make_state

from mjlab.tasks.tracking.diffusion.closed_loop import (
  EvaluationBudget,
  _after_run_marker,
  run_matched_pair,
)
from mjlab.tasks.tracking.diffusion.closed_loop_types import D3Observation, D3Transition
from mjlab.tasks.tracking.diffusion.conditioned_continuation import (
  SparseTaskSpec,
  TaskConditioning,
  TaskConditioningError,
  TaskSource,
  build_conditioned_continuation,
  realized_pose_error,
)
from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT
from mjlab.tasks.tracking.diffusion.handover import (
  HandoverError,
  HandoverHistory,
  ReferenceFreeContinuation,
  build_reference_free_plan,
)
from mjlab.tasks.tracking.diffusion.inpainting import TimedStateKeyframe
from mjlab.tasks.tracking.diffusion.pose_costs import PoseCostWeights, TimedPoseGoal
from mjlab.tasks.tracking.diffusion.projection import (
  FeatureStats,
  ProjectionBundle,
  fit_feature_stats,
)
from mjlab.tasks.tracking.diffusion.schedule import DiffusionSchedule, InferenceGrid
from mjlab.tasks.tracking.diffusion.state import WorldState
from mjlab.tasks.tracking.diffusion.storage import AppendOnlyShardStore

_EXPERIMENT_DIR = (
  Path(__file__).parents[1] / "scripts/experiments/diffusion_d3_closed_loop"
)
_CLI_SPEC = spec_from_file_location(
  "d4_closed_loop_cli", _EXPERIMENT_DIR / "evaluate.py"
)
assert _CLI_SPEC is not None and _CLI_SPEC.loader is not None
evaluate = module_from_spec(_CLI_SPEC)
sys.modules[_CLI_SPEC.name] = evaluate
_CLI_SPEC.loader.exec_module(evaluate)

_LOADER_SPEC = spec_from_file_location(
  "mjlab_d4_sparse_task_config_test", _EXPERIMENT_DIR / "task_config.py"
)
assert _LOADER_SPEC is not None and _LOADER_SPEC.loader is not None
task_config = module_from_spec(_LOADER_SPEC)
sys.modules[_LOADER_SPEC.name] = task_config
_LOADER_SPEC.loader.exec_module(task_config)

_PREP_SPEC = spec_from_file_location(
  "mjlab_d4_prepare_task_config_test", _EXPERIMENT_DIR / "prepare_task_config.py"
)
assert _PREP_SPEC is not None and _PREP_SPEC.loader is not None
prepare_task_config = module_from_spec(_PREP_SPEC)
sys.modules[_PREP_SPEC.name] = prepare_task_config
_PREP_SPEC.loader.exec_module(prepare_task_config)

EXAMPLE_DIR = _EXPERIMENT_DIR / "task_examples"


# --------------------------------------------------------------------------
# focused CPU fakes
# --------------------------------------------------------------------------


def _stats(width: int, *, seed: int) -> FeatureStats:
  rng = np.random.default_rng(seed)
  return fit_feature_stats([rng.normal(size=(64, width)) + 3.0], std_floor=1e-6)


def _projection() -> ProjectionBundle:
  return ProjectionBundle.create(_stats(135, seed=11), _stats(32, seed=12))


class _RecordingDenoiser:
  """Deterministic stub denoiser that records every reverse-loop input."""

  def __init__(self, *, scale: float = 0.1, explode: bool = False) -> None:
    self.scale = scale
    self.explode = explode
    self.calls = 0
    self.inputs: list[torch.Tensor] = []

  def __call__(self, sample: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
    del step_ids
    self.calls += 1
    self.inputs.append(sample.detach().clone())
    if self.explode:
      return torch.full_like(sample, float("nan"))
    return self.scale * sample


class _CoupledDenoiser:
  """Deterministic stub whose latent channels couple to the state channels.

  The frozen denoiser is a transformer, so a physical pose cost at one row
  reaches the current latent through cross-row/cross-channel mixing.  A plain
  scaling stub would confine the effect to the goal row's state channels and
  would not exercise any latent/action delta at all.
  """

  def __init__(self) -> None:
    self.calls = 0
    self.inputs: list[torch.Tensor] = []

  def __call__(self, sample: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
    del step_ids
    self.calls += 1
    self.inputs.append(sample.detach().clone())
    state, latent = sample[..., :135], sample[..., 135:]
    latent_summary = torch.tanh(latent.mean(dim=(-2, -1), keepdim=True))
    state_summary = torch.tanh(state.mean(dim=(-2, -1), keepdim=True))
    return torch.cat(
      (0.5 * state + 0.25 * latent_summary, 0.5 * latent + 0.25 * state_summary),
      dim=-1,
    )


class _LatentDecoder:
  """Fake gravity-mode decoder: no reference field ever reaches it."""

  def __init__(self) -> None:
    self.calls = 0
    self.reference_free = True

  def __call__(self, latent: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
    self.calls += 1
    assert latent.shape == (1, 32) and conditioning.shape == (1, 99)
    return latent[:, :31] * 0.5


def _history(*, length: int = 8) -> HandoverHistory:
  history = HandoverHistory()
  for index in range(length):
    history.push(make_state(index), np.full(32, 0.05 * index))
  return history


def _continuation(
  *,
  denoiser: Any,
  conditioning: TaskConditioning | None = None,
  seed: int = 5,
  projection: ProjectionBundle | None = None,
) -> ReferenceFreeContinuation:
  schedule = DiffusionSchedule.build()
  grid = InferenceGrid.uniform(schedule)
  generator = torch.Generator(device="cpu")
  generator.manual_seed(seed)
  return build_conditioned_continuation(
    conditioning=conditioning,
    denoiser=denoiser,
    decode=_LatentDecoder(),
    schedule=schedule,
    grid=grid,
    projection=projection or _projection(),
    contract=DEFAULT_CONTRACT,
    device="cpu",
    generator=generator,
  )


def _source(*, qualification: str = "unverified") -> TaskSource:
  return TaskSource(
    kind="synthetic_fixture",
    qualification=qualification,
    verification="loader_structural_only",
    provenance="unit-test synthetic fixture; not a physical measurement",
  )


def _spec(
  *,
  mode: str,
  keyframes: tuple[TimedStateKeyframe, ...] = (),
  goals: tuple[TimedPoseGoal, ...] = (),
  strength: float = 0.0,
  weights: PoseCostWeights | None = None,
) -> SparseTaskSpec:
  return SparseTaskSpec(
    mode=mode,
    keyframes=keyframes,
    goals=goals,
    weights=weights,
    strength=strength,
    source=_source(),
    config_sha256="b" * 64,
  )


def _keyframe(tick: int, *, lift: float = 0.0, index: int = 12) -> TimedStateKeyframe:
  state = make_state(index)
  return TimedStateKeyframe(
    tick=tick,
    state=WorldState(
      state.root_position + np.array([0.1, -0.05, lift]),
      state.root_quaternion_wxyz,
      state.root_linear_velocity,
      state.root_angular_velocity,
      state.body_positions,
      state.body_linear_velocities,
    ),
    provenance="unit-test synthetic keyframe",
  )


def _goal(tick: int, *, lift: float = 0.0) -> TimedPoseGoal:
  state = make_state(20)
  return TimedPoseGoal(
    tick=tick,
    state=WorldState(
      state.root_position + np.array([0.2, 0.1, lift]),
      state.root_quaternion_wxyz,
      state.root_linear_velocity,
      state.root_angular_velocity,
      state.body_positions + np.array([0.03, 0.0, 0.02]),
      state.body_linear_velocities,
    ),
    provenance="unit-test synthetic pose goal",
  )


def _weights() -> PoseCostWeights:
  return PoseCostWeights(root_position=1.0, root_orientation=1.0, body_positions=1.0)


# --------------------------------------------------------------------------
# wiring: default parity, hard keyframes, guidance
# --------------------------------------------------------------------------


def test_unguided_task_spec_keeps_the_default_continuation_path() -> None:
  projection = _projection()
  default = _continuation(denoiser=_RecordingDenoiser(), projection=projection)
  for continuation in (default,):
    for row in _history().pairs:
      continuation.record_executed(row.state, row.latent)
  conditioning = TaskConditioning(
    _spec(mode="unguided"), projection=projection, contract=DEFAULT_CONTRACT
  )
  assert conditioning.active is False
  task = _continuation(
    denoiser=_RecordingDenoiser(), conditioning=conditioning, projection=projection
  )
  for row in _history().pairs:
    task.record_executed(row.state, row.latent)
  state = make_state(30)
  base_step = default.step(state, np.full(99, 0.25), tick=40)
  task_step = task.step(state, np.full(99, 0.25), tick=40)
  assert np.array_equal(base_step.action, task_step.action)
  assert np.array_equal(base_step.executed_latent, task_step.executed_latent)
  assert base_step.task is None and task_step.task is None
  assert base_step.denoiser_calls == task_step.denoiser_calls == 20


def test_keyframe_conditioning_fixes_the_state_slice_at_every_denoiser_call() -> None:
  projection = _projection()
  keyframe = _keyframe(48)
  conditioning = TaskConditioning(
    _spec(mode="keyframe", keyframes=(keyframe,)),
    projection=projection,
    contract=DEFAULT_CONTRACT,
  )
  denoiser = _RecordingDenoiser()
  continuation = _continuation(
    denoiser=denoiser, conditioning=conditioning, projection=projection
  )
  for row in _history().pairs:
    continuation.record_executed(row.state, row.latent)
  state = make_state(30)
  plan = continuation.plan(state, tick=40)
  step = continuation.step(state, np.full(99, 0.25), tick=40)
  assert step.task is not None
  assert step.task["active_keyframes"] == [{"tick": 48, "row": 16}]
  row = DEFAULT_CONTRACT.current_index + 8
  fixed = plan.clean_tokens[row, :135]
  assert bool(plan.clean_mask[row, :135].all())
  assert not bool(plan.clean_mask[row, 135:].any())
  assert step.denoiser_calls == denoiser.calls == 20
  for observed in denoiser.inputs:
    assert torch.equal(observed[0, row, :135], fixed)
  # The latent slice of the hard row stays unknown, so the reverse loop writes
  # it instead of pinning the conditioning placeholder.
  assert (
    max(float(observed[0, row, 135:].abs().max()) for observed in denoiser.inputs)
    > 1e-6
  )


def test_guided_task_changes_latent_and_action_with_fixed_noise() -> None:
  projection = _projection()
  goal = _goal(48)
  guided = TaskConditioning(
    _spec(mode="poseguided", goals=(goal,), strength=0.5, weights=_weights()),
    projection=projection,
    contract=DEFAULT_CONTRACT,
  )
  plain = _continuation(denoiser=_CoupledDenoiser(), projection=projection)
  task = _continuation(
    denoiser=_CoupledDenoiser(), conditioning=guided, projection=projection
  )
  for inner in (plain, task):
    for row in _history().pairs:
      inner.record_executed(row.state, row.latent)
  state = make_state(30)
  reference_step = plain.step(state, np.full(99, 0.25), tick=40)
  guided_step = task.step(state, np.full(99, 0.25), tick=40)
  assert guided_step.denoiser_calls == 20
  assert not np.array_equal(reference_step.executed_latent, guided_step.executed_latent)
  assert not np.array_equal(reference_step.action, guided_step.action)
  assert guided_step.task is not None
  assert guided_step.task["guided"] is True
  assert guided_step.task["active_goal_ids"] == [48]
  assert guided_step.task["vjp_calls"] == guided_step.task["cost_calls"] == 20
  assert len(guided_step.task["steps"]) == 20
  summary = guided.provenance()
  assert summary["executed"]["guided_steps"] == 1
  assert summary["unit_tags"]["root_orientation"] == "dimensionless_chordal"
  assert summary["unit_tags"]["root_position"] == "m^2"


def test_keyframe_only_task_does_no_cost_or_vjp_work_and_is_deterministic() -> None:
  projection = _projection()
  keyframe = _keyframe(48)
  steps = []
  for _ in range(2):
    conditioning = TaskConditioning(
      _spec(mode="keyframe", keyframes=(keyframe,)),
      projection=projection,
      contract=DEFAULT_CONTRACT,
    )
    continuation = _continuation(
      denoiser=_RecordingDenoiser(), conditioning=conditioning, projection=projection
    )
    for row in _history().pairs:
      continuation.record_executed(row.state, row.latent)
    state = make_state(30)
    steps.append(continuation.step(state, np.full(99, 0.25), tick=40))
  assert steps[0].task is not None
  assert steps[0].task["vjp_calls"] == 0
  assert steps[0].task["cost_calls"] == 0
  assert steps[0].task["guided"] is False
  assert np.array_equal(steps[0].executed_latent, steps[1].executed_latent)
  assert np.array_equal(steps[0].action, steps[1].action)


def test_realized_goal_error_is_measured_only_at_the_target_tick() -> None:
  projection = _projection()
  goal = _goal(41)
  conditioning = TaskConditioning(
    _spec(mode="poseguided", goals=(goal,), strength=0.5, weights=_weights()),
    projection=projection,
    contract=DEFAULT_CONTRACT,
  )
  continuation = _continuation(
    denoiser=_RecordingDenoiser(), conditioning=conditioning, projection=projection
  )
  for row in _history().pairs:
    continuation.record_executed(row.state, row.latent)
  state = make_state(30)
  continuation.step(state, np.full(99, 0.25), tick=40)
  unreached = conditioning.realized_errors()
  assert unreached["scored"] == [] and unreached["unreached_ticks"] == [41]
  measured = make_state(21)
  continuation.step(measured, np.full(99, 0.25), tick=41)
  realized = conditioning.realized_errors()
  assert len(realized["scored"]) == 1
  scored = realized["scored"][0]
  assert scored["tick"] == 41 and scored["observed_tick"] == 41
  expected = realized_pose_error(measured, goal)
  assert scored["root_position_m"] == pytest.approx(expected["root_position_m"])
  assert scored["root_orientation_chordal"] == pytest.approx(
    expected["root_orientation_chordal"]
  )
  assert 0.0 < scored["root_orientation_chordal"] < 2.0
  assert scored["root_position_m"] > 0.0
  assert scored["body_positions_rms_m"] > 0.0
  summary = conditioning.provenance()
  assert summary["outcome"]["scored_targets"] == 1
  assert "measured" in summary["realized"]["scoring_rule"]


def test_nonfinite_prediction_and_backward_inputs_fail_closed() -> None:
  projection = _projection()
  goal = _goal(48)
  conditioning = TaskConditioning(
    _spec(mode="poseguided", goals=(goal,), strength=0.5, weights=_weights()),
    projection=projection,
    contract=DEFAULT_CONTRACT,
  )
  continuation = _continuation(
    denoiser=_RecordingDenoiser(explode=True),
    conditioning=conditioning,
    projection=projection,
  )
  for row in _history().pairs:
    continuation.record_executed(row.state, row.latent)
  with pytest.raises(ValueError):
    continuation.step(make_state(30), np.full(99, 0.25), tick=40)
  with pytest.raises(TaskConditioningError):
    SparseTaskSpec(
      mode="poseguided",
      keyframes=(),
      goals=(goal,),
      weights=None,
      strength=0.5,
      source=_source(),
      config_sha256="b" * 64,
    )
  with pytest.raises(TaskConditioningError):
    SparseTaskSpec(
      mode="unguided",
      keyframes=(),
      goals=(),
      weights=None,
      strength=1.0,
      source=_source(),
      config_sha256="b" * 64,
    )


def test_default_plan_is_unchanged_when_no_task_is_installed() -> None:
  projection = _projection()
  continuation = _continuation(denoiser=_RecordingDenoiser(), projection=projection)
  for row in _history().pairs:
    continuation.record_executed(row.state, row.latent)
  state = make_state(30)
  plan = continuation.plan(state, tick=40)
  expected = build_reference_free_plan(
    continuation.history, state, projection=projection, tick=40
  )
  assert torch.equal(plan.clean_tokens, expected.clean_tokens)
  assert torch.equal(plan.clean_mask, expected.clean_mask)


# --------------------------------------------------------------------------
# strict task loader
# --------------------------------------------------------------------------


def _example() -> dict[str, Any]:
  return json.loads(
    (EXAMPLE_DIR / "synthetic_combined.example.json").read_text(encoding="utf-8")
  )


def test_loader_accepts_the_shipped_examples() -> None:
  combined = task_config.load_task_config(
    EXAMPLE_DIR / "synthetic_combined.example.json"
  )
  assert combined.mode == "combined"
  assert combined.keyframe_ticks == (40,) and combined.goal_ticks == (60,)
  assert combined.source.qualification == "unverified"
  keyframes = task_config.load_task_config(
    EXAMPLE_DIR / "synthetic_keyframe_only.example.json"
  )
  assert keyframes.mode == "keyframe" and keyframes.strength == 0.0


@pytest.mark.parametrize(
  "mutate",
  [
    lambda payload: payload.update({"unexpected": 1}),
    lambda payload: payload.update({"schema": "mjlab-x4-legacy-v0"}),
    lambda payload: payload.update({"frame": "hybrid"}),
    lambda payload: payload.update({"contract_identity": "c" * 64}),
    lambda payload: payload.update(
      {"body_order": list(reversed(payload["body_order"]))}
    ),
    lambda payload: payload["keyframes"][0].update({"tick": True}),
    lambda payload: payload["keyframes"][0].update({"tick": 40.5}),
    lambda payload: payload["keyframes"][0].update({"latent": [0.0] * 32}),
    lambda payload: payload["keyframes"][0]["state"].update(
      {"root_position": [0.0, 0.0]}
    ),
    lambda payload: payload["keyframes"][0]["state"].update(
      {"root_position": [float("nan"), 0.0, 0.8]}
    ),
    lambda payload: payload["keyframes"].append(dict(payload["keyframes"][0])),
    lambda payload: payload["goals"].append(
      {**payload["goals"][0], "tick": payload["goals"][0]["tick"] + 5}
    ),
    lambda payload: payload.update({"strength": 0.0}),
    lambda payload: payload.update({"weights": {"root_position": 1.0}}),
    lambda payload: payload.update(
      {
        "source": {
          **payload["source"],
          "qualification": "verified",
        }
      }
    ),
    lambda payload: payload.update(
      {"source": {**payload["source"], "kind": "manual", "qualification": "declared"}}
    ),
  ],
)
def test_loader_rejects_malformed_or_unsupported_task_inputs(
  tmp_path: Path, mutate: Any
) -> None:
  payload = _example()
  mutate(payload)
  if payload.get("source", {}).get("qualification") == "verified":
    payload["source"]["kind"] = "baseline_recording"
    payload["source"]["path"] = "recording.npz"
    payload["source"]["sha256"] = "d" * 64
  path = tmp_path / "task.json"
  path.write_text(json.dumps(payload), encoding="utf-8")
  with pytest.raises((task_config.TaskConfigError, ValueError)):
    task_config.load_task_config(path)


def test_loader_rejects_duplicate_json_keys_and_oversized_files(tmp_path: Path) -> None:
  raw = (EXAMPLE_DIR / "synthetic_combined.example.json").read_text(encoding="utf-8")
  duplicated = raw.replace(
    '"mode": "combined",', '"mode": "combined", "mode": "combined",'
  )
  path = tmp_path / "duplicate.json"
  path.write_text(duplicated, encoding="utf-8")
  with pytest.raises(task_config.TaskConfigError, match="duplicate JSON key"):
    task_config.load_task_config(path)
  big = tmp_path / "big.json"
  big.write_bytes(b"{" + b" " * (task_config.MAX_TASK_BYTES + 1))
  with pytest.raises(task_config.TaskConfigError, match="byte bound"):
    task_config.load_task_config(big)
  # A non-regular path and a malformed target list must both raise the strict
  # task error rather than leaking an OSError or a TypeError.
  with pytest.raises(task_config.TaskConfigError, match="regular file"):
    task_config.load_task_config(tmp_path)
  malformed = tmp_path / "malformed.json"
  payload = _example()
  payload["goals"][0] = {"tick": 60, "state": None, "provenance": "x"}
  malformed.write_text(json.dumps(payload), encoding="utf-8")
  with pytest.raises(task_config.TaskConfigError):
    task_config.load_task_config(malformed)


# --------------------------------------------------------------------------
# CLI gate: malformed task config never reaches a runtime
# --------------------------------------------------------------------------


def _prepare_args(tmp_path: Path, **updates: object) -> Namespace:
  values: dict[str, object] = {
    "contract": tmp_path / "contract.yaml",
    "contract_sha256": "a" * 64,
    "checkpoint": tmp_path / "checkpoint.pt",
    "checkpoint_sha256": "a" * 64,
    "projection": tmp_path / "projection.npz",
    "projection_sha256": "a" * 64,
    "vae_checkpoint": tmp_path / "vae.pt",
    "vae_checkpoint_sha256": "a" * 64,
    "manifest": tmp_path / "manifest.yaml",
    "manifest_sha256": "a" * 64,
    "repo_root": tmp_path,
    "member": "tennis_000",
    "seed": 0,
    "start_frame": 0,
    "continuation_steps": 2,
    "max_wall_seconds": 1200.0,
    "max_output_bytes": evaluate.MIN_OUTPUT_CAP_BYTES,
    "output_dir": tmp_path / "run",
    "device": "cpu",
    "task_config": None,
    "execute": True,
  }
  values.update(updates)
  return Namespace(**values)


def _patch_prepare(monkeypatch: pytest.MonkeyPatch) -> None:
  schedule = DiffusionSchedule.build()
  grid = InferenceGrid.uniform(schedule)
  contract = DEFAULT_CONTRACT
  denoiser = SimpleNamespace(
    schedule=schedule,
    grid=grid,
    global_step=1,
    ema_selected=True,
    ema_updates=1,
    selection_split="val",
    best_validation_loss=0.5,
    fp32_checked=True,
    config_seed=0,
    contract_identity="c" * 64,
    config_hash="d" * 64,
    projection_hashes={},
  )
  vae = SimpleNamespace(
    reference_dim=68,
    conditioning_dim=99,
    latent_dim=32,
    action_dim=31,
    model=object(),
    relocated_artifact_roles=(),
    joint_order=tuple(f"joint-{index}" for index in range(31)),
  )
  teacher = SimpleNamespace(reference=SimpleNamespace(frames=100, fps=50.0), hashes={})
  monkeypatch.setattr(evaluate, "verify_file_digest", lambda *a, **k: None)
  monkeypatch.setattr(evaluate, "load_active_contract", lambda path: contract)
  monkeypatch.setattr(evaluate, "_require_frozen_contract", lambda *a, **k: None)
  monkeypatch.setattr(
    evaluate, "load_manifest", lambda path, repo_root: SimpleNamespace(sha256="e" * 64)
  )
  monkeypatch.setattr(
    evaluate,
    "resolve_cohort",
    lambda manifest: SimpleNamespace(teacher=lambda m: teacher),
  )
  monkeypatch.setattr(evaluate, "load_projection_bundle", lambda *a, **k: _projection())
  monkeypatch.setattr(evaluate, "load_denoiser", lambda *a, **k: denoiser)
  monkeypatch.setattr(evaluate, "load_vae_decoder", lambda *a, **k: vae)
  monkeypatch.setattr(evaluate, "_source_fingerprints", lambda: {"modules": {}})
  monkeypatch.setattr(evaluate, "_integrity_snapshot", lambda prepared: {"files": {}})


def test_malformed_task_config_is_rejected_before_any_runtime_or_model_use(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _patch_prepare(monkeypatch)
  constructions: list[object] = []
  monkeypatch.setattr(
    evaluate, "_runtime_factory", lambda prepared: constructions.append(prepared)
  )
  monkeypatch.setattr(
    evaluate,
    "_execute",
    lambda prepared: (_ for _ in ()).throw(AssertionError("execution dispatched")),
  )
  bad = tmp_path / "task.json"
  bad.write_text("{not json", encoding="utf-8")
  args = _prepare_args(tmp_path, task_config=bad)
  with pytest.raises(evaluate.ClosedLoopCliError, match="task-config rejected"):
    evaluate._prepare(args)
  assert constructions == []

  payload = _example()
  payload["keyframes"][0]["tick"] = True
  bad.write_text(json.dumps(payload), encoding="utf-8")
  with pytest.raises(evaluate.ClosedLoopCliError, match="task-config rejected"):
    evaluate._prepare(args)
  assert constructions == []


def test_valid_task_config_reaches_preflight_with_bounded_metadata(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _patch_prepare(monkeypatch)
  args = _prepare_args(
    tmp_path,
    task_config=EXAMPLE_DIR / "synthetic_combined.example.json",
    execute=False,
  )
  prepared = evaluate._prepare(args)
  assert prepared.task is not None and prepared.task.mode == "combined"
  report = evaluate._preflight_report(prepared)
  assert report["task"]["provided"] is True
  assert report["task"]["keyframe_count"] == 1
  assert report["task"]["goal_count"] == 1
  assert report["task"]["config_sha256"] == prepared.task.config_sha256
  assert report["task"]["source"]["qualification"] == "unverified"
  assert "state" not in json.dumps(report["task"])


def test_load_time_gates_reject_unreadable_and_oversized_task_paths(
  tmp_path: Path,
) -> None:
  missing = _prepare_args(tmp_path, task_config=tmp_path / "absent.json")
  with pytest.raises(evaluate.ClosedLoopCliError, match="not a readable file"):
    evaluate._validate_request(missing)
  oversized = tmp_path / "large.json"
  oversized.write_bytes(b" " * ((1 << 20) + 1))
  with pytest.raises(evaluate.ClosedLoopCliError, match="at most 1 MiB"):
    evaluate._validate_request(_prepare_args(tmp_path, task_config=oversized))


# --------------------------------------------------------------------------
# end-to-end fake run: bootstrap, handover, bounded marker
# --------------------------------------------------------------------------


class _FakeVae:
  def __init__(self) -> None:
    self.encode_calls = 0
    self.decode_calls = 0

  def encode(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    self.encode_calls += 1
    assert reference.shape == (1, 68)
    mu = torch.zeros((1, 32), dtype=torch.float32, device=reference.device)
    return mu, torch.zeros_like(mu)

  def decode(self, latent: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
    self.decode_calls += 1
    assert latent.shape == (1, 32) and conditioning.shape == (1, 99)
    return latent[:, :31] * 0.5


class _FakeRuntime:
  period_seconds = 0.02
  clip_frames = 100
  joint_order = tuple(f"joint-{index}" for index in range(31))
  body_order = tuple(f"body-{index}" for index in range(20))
  provenance = {"kind": "fake-test", "synthetic": True, "physical_runtime": "synthetic"}

  def __init__(
    self, *, post_overrides: Mapping[int, Mapping[str, Any]] | None = None
  ) -> None:
    # ``post_overrides`` injects one post-step condition per resulting tick so a
    # test can build an untrusted or falling successor without touching the
    # frozen runner logic.  Keys: flags, tilt_degrees, tick_step, time_shift,
    # reference.
    self.post_overrides = dict(post_overrides or {})
    self.closed = False
    self.entered_reference_free = False
    self.actions: list[np.ndarray] = []
    self._index = 0
    self._tick = 0

  def _observation(self) -> D3Observation:
    return D3Observation(
      state=make_state(self._index),
      conditioning=np.full(99, 0.25),
      tick=self._tick,
      sim_time_s=self._tick * self.period_seconds,
      segment_id="segment",
      reference=None if self.entered_reference_free else np.full(68, 0.5),
      reference_frame=None if self.entered_reference_free else self._index,
    )

  def reset(self, *, seed: int, start_frame: int) -> D3Observation:
    del seed
    self._index = start_frame
    self._tick = 0
    self.entered_reference_free = False
    return self._observation()

  def initial_signature(self) -> np.ndarray:
    return np.arange(37, dtype=np.float64)

  def enter_reference_free(self) -> D3Observation:
    self.entered_reference_free = True
    return self._observation()

  def step(self, action: np.ndarray) -> D3Transition:
    self.actions.append(np.array(action, copy=True))
    self._index += 1
    self._tick += 1
    override = self.post_overrides.get(self._tick, {})
    if "tick_step" in override:
      self._tick += int(override["tick_step"]) - 1
    observation = self._observation()
    if "time_shift" in override:
      observation = replace(
        observation, sim_time_s=observation.sim_time_s + float(override["time_shift"])
      )
    if override.get("reference"):
      observation = replace(observation, reference=np.full(68, 0.5), reference_frame=0)
    if "tilt_degrees" in override:
      angle = math.radians(float(override["tilt_degrees"])) / 2.0
      observation = replace(
        observation,
        state=WorldState(
          observation.state.root_position,
          np.array([math.cos(angle), math.sin(angle), 0.0, 0.0]),
          observation.state.root_linear_velocity,
          observation.state.root_angular_velocity,
          observation.state.body_positions,
          observation.state.body_linear_velocities,
        ),
      )
    return D3Transition(observation, **dict(override.get("flags", {})))

  def close(self) -> None:
    self.closed = True


def test_fake_task_run_keeps_bootstrap_handover_and_a_bounded_marker(
  tmp_path: Path,
) -> None:
  projection = _projection()
  keyframe = _keyframe(20)
  task = _spec(mode="keyframe", keyframes=(keyframe,))
  conditioning = TaskConditioning(
    task, projection=projection, contract=DEFAULT_CONTRACT
  )
  model = _FakeVae()
  schedule = DiffusionSchedule.build()
  grid = InferenceGrid.uniform(schedule)
  decoders: list[_LatentDecoder] = []

  def factory() -> ReferenceFreeContinuation:
    decoder = _LatentDecoder()
    decoders.append(decoder)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    return build_conditioned_continuation(
      conditioning=conditioning,
      denoiser=_RecordingDenoiser(),
      decode=decoder,
      schedule=schedule,
      grid=grid,
      projection=projection,
      contract=DEFAULT_CONTRACT,
      device="cpu",
      generator=generator,
    )

  report = run_matched_pair(
    _FakeRuntime,
    model=model,
    continuation_factory=factory,
    seed=0,
    start_frame=0,
    budget=EvaluationBudget(
      max_steps=3, bootstrap_pairs=8, period_seconds=0.02, output_cap_bytes=4096
    ),
    device="cpu",
    latent_normalizer=projection.normalize_latent,
    output_dir=tmp_path / "run",
    provenance={"runner": "d4-integration-test"},
  )
  object.__setattr__(
    report, "provenance", {**report.provenance, "task": conditioning.provenance()}
  )
  evaluate.persist_evaluation_report(report, tmp_path / "run")
  diffusion = report.diffusion
  assert diffusion is not None
  assert diffusion.bootstrap_completed == 8
  assert (
    model.encode_calls == 8 + 8 + 3
  )  # baseline bootstrap+ticks, diffusion bootstrap
  diffusion_ticks = [tick for tick in diffusion.ticks if tick["phase"] == "diffusion"]
  assert diffusion_ticks, "the diffusion branch must reach the continuation phase"
  for tick in diffusion_ticks:
    assert tick["reference"] is None and tick["reference_frame"] is None
    assert tick["task"]["mode"] == "keyframe"
    assert tick["denoiser_calls"] == 20
    assert tick["prediction"]["center"] == "pre_action_state"
  assert conditioning.plan_transform is not None
  assert conditioning.provenance()["executed"]["keyframe_steps"] >= 1
  assert decoders and decoders[0].reference_free
  marker = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
  assert marker["output"]["representation"] == "marker"
  assert marker["after_run"]["task"]["mode"] == "keyframe"
  assert marker["after_run"]["task"]["keyframe_count"] == 1
  assert marker["after_run"]["task"]["config_sha256"] == task.config_sha256
  assert marker["after_run"]["task"]["executed"]["denoiser_calls"] == 20 * len(
    diffusion_ticks
  )
  assert marker["after_run"]["task"]["has_error"] in {True, False}


# --------------------------------------------------------------------------
# preparation tool against a real recording store
# --------------------------------------------------------------------------


def _rows_for_member(rows: list[Any], member: str) -> list[Any]:
  """Point every row at the requested motion id without changing its state."""
  from dataclasses import replace

  return [replace(row, motion_id=member) for row in rows]


def _store(tmp_path: Path, *, member: str = "tennis_000", seed: int = 0) -> Path:
  root = tmp_path / "store"
  store = AppendOnlyShardStore(root)
  rows = [
    make_row(
      index,
      provenance={
        "initial_seed": str(seed),
        "initial_start_frame": "0",
        "phase": "clean",
      },
    )
    for index in range(30)
  ]
  store.append(_rows_for_member(rows, member))
  return root


def test_preparation_tool_extracts_verified_targets_from_a_recording(
  tmp_path: Path,
) -> None:
  root = _store(tmp_path)
  shard = root / "shard-000000.npz"
  digest = prepare_task_config._sha256(shard)
  payload = prepare_task_config.prepare(
    source_shard=shard,
    source_sha256=digest,
    store_root=root,
    member="tennis_000",
    seed=0,
    start_frame=0,
    mode="combined",
    keyframe_ticks=(4, 14),
    goal_ticks=(24,),
    strength=0.5,
    weights={
      "root_position": 1.0,
      "root_orientation": 1.0,
      "body_positions": 1.0,
    },
    contract=DEFAULT_CONTRACT,
    diagnostic=False,
  )
  # A raw NPZ shard carries weaker recorded provenance than a D3 recording, so
  # the loader keeps it declared and the --execute gate refuses it.
  assert payload["source"]["qualification"] == "declared"
  assert payload["source"]["verification"].startswith("preparation_tool")
  assert payload["source"]["verification_detail"]["store_binding"]["shard_listed"]
  assert payload["source"]["verification_detail"]["scenario"]["motion_id"] == (
    "tennis_000"
  )
  out = tmp_path / "task.json"
  out.write_text(json.dumps(payload), encoding="utf-8")
  spec = task_config.load_task_config(out)
  assert spec.source.qualification == "declared"
  assert spec.keyframe_ticks == (4, 14) and spec.goal_ticks == (24,)
  with pytest.raises(task_config.TaskConfigError, match="verified recording source"):
    task_config.load_task_config(out, require_verified=True)


def test_preparation_tool_refuses_scenario_mismatch_and_marks_diagnostics(
  tmp_path: Path,
) -> None:
  root = _store(tmp_path, seed=7)
  shard = root / "shard-000000.npz"
  with pytest.raises(prepare_task_config.PreparationError, match="scenario"):
    prepare_task_config.prepare(
      source_shard=shard,
      source_sha256=prepare_task_config._sha256(shard),
      store_root=root,
      member="tennis_000",
      seed=0,
      start_frame=0,
      mode="keyframe",
      keyframe_ticks=(4, 14),
      goal_ticks=(),
      strength=0.0,
      weights=None,
      contract=DEFAULT_CONTRACT,
      diagnostic=False,
    )
  diagnostic = prepare_task_config.prepare(
    source_shard=shard,
    source_sha256=prepare_task_config._sha256(shard),
    store_root=None,
    member="tennis_000",
    seed=0,
    start_frame=0,
    mode="keyframe",
    keyframe_ticks=(4, 14),
    goal_ticks=(),
    strength=0.0,
    weights=None,
    contract=DEFAULT_CONTRACT,
    diagnostic=True,
  )
  assert diagnostic["source"]["qualification"] == "unverified"
  assert diagnostic["source"]["kind"] == "synthetic_fixture"
  assert diagnostic["source"]["verification_detail"]["scenario_mismatch"]


def test_main_validates_a_task_config_without_touching_a_simulator(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  code = task_config.main(
    ["--task-config", str(EXAMPLE_DIR / "synthetic_combined.example.json")]
  )
  assert code == 0
  assert '"ok": true' in capsys.readouterr().out
  bad = tmp_path / "bad.json"
  bad.write_text('{"schema": "legacy"}', encoding="utf-8")
  assert task_config.main(["--task-config", str(bad)]) == 1
  assert "rejected" in capsys.readouterr().out


# --------------------------------------------------------------------------
# R1: a run without a task config keeps the original D3 metadata
# --------------------------------------------------------------------------


def test_task_config_report_is_absent_without_a_config() -> None:
  assert evaluate._task_config_report(None) is None


def test_after_run_marker_omits_task_for_a_no_task_run() -> None:
  report: Any = SimpleNamespace(
    provenance={
      "integrity": {"unchanged": True},
      "execution": {"within_wall_budget": True},
    }
  )
  marker = _after_run_marker(cast(Any, report))
  assert "task" not in marker
  task_report: Any = SimpleNamespace(
    provenance={
      "execution": {"within_wall_budget": True},
      "task": {"mode": "keyframe", "config_sha256": "a" * 64, "executed": {}},
    }
  )
  assert "task" in _after_run_marker(cast(Any, task_report))


def test_provenance_and_preflight_include_task_only_when_configured(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _patch_prepare(monkeypatch)
  without = evaluate._prepare(_prepare_args(tmp_path, output_dir=tmp_path / "a"))
  assert without.task is None
  report = evaluate._preflight_report(without)
  assert "task" not in report
  provenance = evaluate._provenance(without, 0.0)
  assert "task" not in provenance

  # A manual/synthetic config is diagnostic-only: it loads without --execute.
  with_task = evaluate._prepare(
    _prepare_args(
      tmp_path,
      output_dir=tmp_path / "b",
      task_config=EXAMPLE_DIR / "synthetic_keyframe_only.example.json",
      execute=False,
    )
  )
  assert "task" in evaluate._preflight_report(with_task)
  assert "task" in evaluate._provenance(with_task, 0.0)


def test_unverified_task_source_is_refused_under_execute(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _patch_prepare(monkeypatch)
  args = _prepare_args(
    tmp_path,
    output_dir=tmp_path / "run",
    task_config=EXAMPLE_DIR / "synthetic_combined.example.json",
    execute=True,
  )
  with pytest.raises(evaluate.ClosedLoopCliError, match="verified recording source"):
    evaluate._prepare(args)


# --------------------------------------------------------------------------
# R2: pure modes and the exact hard/soft tick collision
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
  ("mode", "keyframes", "goals", "strength", "weights"),
  [
    ("keyframe", True, True, 0.0, None),
    ("poseguided", True, True, 0.5, True),
    ("combined", True, True, 0.5, True),
    ("keyframe", True, False, 0.5, True),
    ("poseguided", True, False, 0.5, True),
  ],
)
def test_cross_mode_lists_and_exact_tick_collisions_are_rejected(
  mode: str, keyframes: bool, goals: bool, strength: float, weights: bool | None
) -> None:
  keyframe_targets = (_keyframe(48),) if keyframes else ()
  goal_targets = (_goal(48),) if goals else ()
  with pytest.raises(TaskConditioningError):
    _spec(
      mode=mode,
      keyframes=keyframe_targets,
      goals=goal_targets,
      strength=strength,
      weights=_weights() if weights else None,
    )


def test_legitimate_combined_targets_on_different_ticks_are_accepted() -> None:
  spec = _spec(
    mode="combined",
    keyframes=(_keyframe(40),),
    goals=(_goal(60),),
    strength=0.5,
    weights=_weights(),
  )
  assert spec.keyframe_ticks == (40,) and spec.goal_ticks == (60,)
  # Nearby targets on different rows stay allowed: the frozen plan fixes only
  # the 10-tick spacing inside each list, not across the two lists.
  nearby = _spec(
    mode="combined",
    keyframes=(_keyframe(40),),
    goals=(_goal(41),),
    strength=0.5,
    weights=_weights(),
  )
  assert nearby.goal_ticks == (41,)


def test_loader_rejects_cross_mode_and_colliding_payloads(tmp_path: Path) -> None:
  payload = _example()
  payload["mode"] = "keyframe"
  payload["strength"] = 0.0
  payload["source"] = {
    "kind": "synthetic_fixture",
    "qualification": "unverified",
    "verification": "loader_structural_only",
    "provenance": "unit-test synthetic fixture; not a physical measurement",
  }
  path = tmp_path / "cross-mode.json"
  path.write_text(json.dumps(payload), encoding="utf-8")
  with pytest.raises(task_config.TaskConfigError, match="keyframe"):
    task_config.load_task_config(path)

  colliding = _example()
  colliding["goals"][0]["tick"] = colliding["keyframes"][0]["tick"]
  path.write_text(json.dumps(colliding), encoding="utf-8")
  with pytest.raises(task_config.TaskConfigError, match="collide"):
    task_config.load_task_config(path)


# --------------------------------------------------------------------------
# R3: reset clears the task ledger and demands a fresh bootstrap
# --------------------------------------------------------------------------


def test_reset_clears_the_task_ledger_and_requires_a_fresh_bootstrap() -> None:
  projection = _projection()
  goal = _goal(41)
  conditioning = TaskConditioning(
    _spec(mode="poseguided", goals=(goal,), strength=0.5, weights=_weights()),
    projection=projection,
    contract=DEFAULT_CONTRACT,
  )
  continuation = _continuation(
    denoiser=_RecordingDenoiser(), conditioning=conditioning, projection=projection
  )
  for row in _history().pairs:
    continuation.record_executed(row.state, row.latent)
  continuation.step(make_state(30), np.full(99, 0.25), tick=40)
  continuation.step(make_state(21), np.full(99, 0.25), tick=41)
  ledger = conditioning.realized_errors()
  assert len(ledger["scored"]) == 1 and ledger["plan_ticks"] == [40, 41]
  assert conditioning.provenance()["executed"]["plan_steps"] == 2

  continuation.reset()
  empty = conditioning.realized_errors()
  assert empty["scored"] == []
  assert empty["observed_ticks"] == [] and empty["plan_ticks"] == []
  executed = conditioning.provenance()["executed"]
  assert executed["plan_steps"] == 0 and executed["measured_observations"] == 0
  assert executed["denoiser_calls"] == 0 and executed["guided_steps"] == 0
  with pytest.raises(HandoverError):
    continuation.step(make_state(30), np.full(99, 0.25), tick=41)

  # A fresh eight-pair bootstrap is mandatory and a reused absolute tick is now
  # measured again rather than inheriting the pre-reset result.
  for row in _history().pairs:
    continuation.record_executed(row.state, row.latent)
  reused = continuation.step(make_state(21), np.full(99, 0.25), tick=41)
  assert reused.tick == 41
  after = conditioning.realized_errors()
  assert len(after["scored"]) == 1 and after["scored"][0]["observed_tick"] == 41
  assert conditioning.provenance()["executed"]["plan_steps"] == 1


def test_default_reset_and_measured_post_hooks_are_inert() -> None:
  projection = _projection()
  continuation = _continuation(denoiser=_RecordingDenoiser(), projection=projection)
  for row in _history().pairs:
    continuation.record_executed(row.state, row.latent)
  continuation.reset()
  assert continuation.history.pairs == ()
  # No hooks installed: both lifecycle seams stay inert and cannot raise.
  continuation.observe_measured_post(41, make_state(21))


# --------------------------------------------------------------------------
# R4: hard and soft targets are scored from trusted measured posts
# --------------------------------------------------------------------------


def _task_report(
  conditioning: TaskConditioning | None,
  projection: ProjectionBundle,
  *,
  runtime_factory: Any = _FakeRuntime,
  max_steps: int = 1,
) -> Any:
  def factory() -> ReferenceFreeContinuation:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    return _continuation(
      denoiser=_CoupledDenoiser(), conditioning=conditioning, projection=projection
    )

  return run_matched_pair(
    runtime_factory,
    model=_FakeVae(),
    continuation_factory=factory,
    seed=0,
    start_frame=0,
    budget=EvaluationBudget(max_steps=max_steps, output_cap_bytes=1 << 20),
    device="cpu",
    latent_normalizer=projection.normalize_latent,
    provenance={"runner": "d4-lifecycle-test"},
  )


def test_final_budget_post_scores_a_soft_goal() -> None:
  projection = _projection()
  goal = _goal(9)
  conditioning = TaskConditioning(
    _spec(mode="poseguided", goals=(goal,), strength=0.5, weights=_weights()),
    projection=projection,
    contract=DEFAULT_CONTRACT,
  )
  report = _task_report(conditioning, projection, max_steps=1)
  assert report.diffusion.ticks[-1]["post"]["tick"] == 9
  ledger = conditioning.realized_errors()
  assert ledger["unreached_ticks"] == []
  assert len(ledger["scored"]) == 1
  scored = ledger["scored"][0]
  assert scored["target_kind"] == "goal" and scored["tick"] == 9
  assert scored["observed_tick"] == 9
  assert scored["root_position_m"] > 0.0
  executed = conditioning.provenance()["executed"]
  # The final budget post is measured without becoming another plan step.
  assert executed["plan_steps"] == 1
  assert executed["measured_observations"] == 2


def test_keyframe_target_is_scored_from_the_measured_post() -> None:
  projection = _projection()
  keyframe = _keyframe(9)
  conditioning = TaskConditioning(
    _spec(mode="keyframe", keyframes=(keyframe,)),
    projection=projection,
    contract=DEFAULT_CONTRACT,
  )
  report = _task_report(conditioning, projection, max_steps=1)
  ledger = conditioning.realized_errors()
  assert len(ledger["scored"]) == 1
  scored = ledger["scored"][0]
  assert scored["target_kind"] == "keyframe" and scored["observed_tick"] == 9
  assert ledger["counts"]["scored_keyframes"] == 1
  tick = report.diffusion.ticks[-1]
  # The hard keyframe fixes the next row, so the row-9 metric is labelled as
  # that constraint rather than as a learned denoiser forecast.
  assert tick["task"]["next_row_hard"] is True
  assert tick["prediction"]["source"] == "hard_keyframe_row"


def test_prediction_source_is_labelled_on_task_runs_only(tmp_path: Path) -> None:
  projection = _projection()
  goal = _goal(20)
  conditioning = TaskConditioning(
    _spec(mode="poseguided", goals=(goal,), strength=0.5, weights=_weights()),
    projection=projection,
    contract=DEFAULT_CONTRACT,
  )
  report = _task_report(conditioning, projection, max_steps=1)
  assert report.diffusion.ticks[-1]["prediction"]["source"] == "denoiser_forecast"

  plain = _task_report(None, projection, max_steps=1)
  # A run without a task config keeps the original D3 prediction payload and
  # acquires no per-tick task metadata.
  prediction = plain.diffusion.ticks[-1]["prediction"]
  assert "source" not in prediction
  assert "task" not in plain.diffusion.ticks[-1]


def test_trusted_fall_post_is_scored_and_stop_precedence_is_unchanged() -> None:
  projection = _projection()
  goal = _goal(9)
  conditioning = TaskConditioning(
    _spec(mode="poseguided", goals=(goal,), strength=0.5, weights=_weights()),
    projection=projection,
    contract=DEFAULT_CONTRACT,
  )
  report = _task_report(
    conditioning,
    projection,
    runtime_factory=lambda: _FakeRuntime(post_overrides={9: {"tilt_degrees": 90.0}}),
    max_steps=3,
  )
  tick = report.diffusion.ticks[-1]
  assert tick["stop_reason"] == "physical_fall"
  assert report.diffusion.outcome == "physical_fall"
  ledger = conditioning.realized_errors()
  assert [item["tick"] for item in ledger["scored"]] == [9]
  assert ledger["unreached_ticks"] == []


@pytest.mark.parametrize(
  ("override", "stop_reason"),
  [
    ({"flags": {"terminal_available": False}}, "missing_terminal_evidence"),
    ({"flags": {"reset": True}}, "reset"),
    ({"flags": {"teleport": True}}, "teleport"),
    ({"flags": {"truncated": True}}, "timeout"),
    ({"tick_step": 2}, "tick_discontinuity"),
    ({"time_shift": 0.5}, "period_discontinuity"),
    ({"reference": True}, "reference_leakage"),
  ],
)
def test_untrusted_post_observations_never_enter_the_ledger(
  override: Mapping[str, Any], stop_reason: str
) -> None:
  projection = _projection()
  goal = _goal(9)
  conditioning = TaskConditioning(
    _spec(mode="poseguided", goals=(goal,), strength=0.5, weights=_weights()),
    projection=projection,
    contract=DEFAULT_CONTRACT,
  )
  report = _task_report(
    conditioning,
    projection,
    runtime_factory=lambda: _FakeRuntime(post_overrides={9: override}),
    max_steps=3,
  )
  assert report.diffusion.ticks[-1]["stop_reason"] == stop_reason
  ledger = conditioning.realized_errors()
  assert ledger["scored"] == []
  assert ledger["observed_ticks"] == [8]
  assert ledger["unreached_ticks"] == [9]
  assert "task_measured_posts" not in report.diffusion.counters


def test_non_task_run_adds_no_phantom_post_observer_counter() -> None:
  report = _task_report(None, _projection(), max_steps=3)
  assert "task_measured_posts" not in report.diffusion.counters
  assert report.diffusion.evaluation_attempted == 3


def test_post_observer_failure_stops_before_another_action(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  projection = _projection()
  conditioning = TaskConditioning(
    _spec(mode="keyframe", keyframes=(_keyframe(9),)),
    projection=projection,
    contract=DEFAULT_CONTRACT,
  )

  def fail(*_args: Any) -> None:
    raise ValueError("deterministic post observer failure")

  monkeypatch.setattr(conditioning, "observe_measured_post", fail)
  report = _task_report(conditioning, projection, max_steps=3)
  assert report.status == "partial"
  assert not report.valid
  assert report.diffusion.stop_reason == "task_observation_error"
  assert report.diffusion.evaluation_attempted == 1
  assert report.diffusion.evaluation_completed == 0
  ticks = [row for row in report.diffusion.ticks if row["phase"] == "diffusion"]
  assert len(ticks) == 1
  assert ticks[0]["post"]["tick"] == 9
  assert ticks[0]["stop_reason"] == "task_observation_error"
  assert ticks[0]["task_observation_error"] == "deterministic post observer failure"


# --------------------------------------------------------------------------
# R5/R6: recording re-verification, forged qualification, D3 extraction
# --------------------------------------------------------------------------


def _write_d3_recording(
  directory: Path,
  *,
  member: str = "tennis_000",
  seed: int = 0,
  start_frame: int = 45,
  ticks: int = 60,
  contract: Any = DEFAULT_CONTRACT,
  status: str = "complete",
  closed_loop: bool = True,
  synthetic: bool = False,
  denoiser_calls: int = 0,
) -> Path:
  """Write a small D3-layout recording (summary.json + ticks.jsonl)."""
  directory.mkdir(parents=True, exist_ok=True)
  lines = []
  for tick in range(ticks):
    state = make_state(tick)
    post = make_state(tick + 1)
    lines.append(
      json.dumps(
        {
          "branch": "baseline",
          "tick": {
            "phase": "bootstrap" if tick < 8 else "baseline",
            "tick": tick,
            "sim_time_s": tick * 0.02,
            "segment_id": "segment",
            "state": {
              name: np.asarray(getattr(state, name), dtype=np.float64).tolist()
              for name in (
                "root_position",
                "root_quaternion_wxyz",
                "root_linear_velocity",
                "root_angular_velocity",
                "body_positions",
                "body_linear_velocities",
              )
            },
            "completed": True,
            "stop_reason": None,
            "post": {
              "tick": tick + 1,
              "sim_time_s": (tick + 1) * 0.02,
              "segment_id": "segment",
              "state": {
                name: np.asarray(getattr(post, name), dtype=np.float64).tolist()
                for name in (
                  "root_position",
                  "root_quaternion_wxyz",
                  "root_linear_velocity",
                  "root_angular_velocity",
                  "body_positions",
                  "body_linear_velocities",
                )
              },
            },
          },
        },
        sort_keys=True,
      )
    )
    # A diffusion row must never become a verified baseline target.
    lines.append(json.dumps({"branch": "diffusion", "tick": {"tick": tick}}))
  (directory / "ticks.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
  summary = {
    "schema": "mjlab-d3-closed-loop-evaluation-v1",
    "status": status,
    "paired": True,
    "valid": True,
    "closed_loop": closed_loop,
    "synthetic": synthetic,
    "errors": [],
    "budget": {"bootstrap_pairs": 8, "period_seconds": contract.period_seconds},
    "provenance": {
      "integrity": {"unchanged": True},
      "vae": {"sha256": contract.vae_sha256},
      "checkpoint": {"contract_identity": contract.identity_hash()},
      "environment": {
        "member": member,
        "seed": seed,
        "start_frame": start_frame,
        "clip_fps": 50.0,
        "clip_frames": 453,
      },
    },
    "branches": {
      "baseline": {
        "completed_ticks": ticks,
        "attempted_ticks": ticks,
        "errors": [],
        "counters": {"encoder": ticks, "decoder": ticks, "denoiser": denoiser_calls},
        "provenance": {
          "body_order": list(contract.body_names),
          "physical_runtime": "mjlab",
          "synthetic": False,
          "untrusted_terminal_evidence": False,
        },
      }
    },
  }
  (directory / "summary.json").write_text(
    json.dumps(summary, sort_keys=True), encoding="utf-8"
  )
  return directory


def test_d3_recording_extraction_is_verified_and_loader_reaccepted(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  recording = _write_d3_recording(tmp_path / "recording")
  config_path = tmp_path / "task.json"
  code = prepare_task_config.main(
    [
      "--recording-dir",
      str(recording),
      "--member",
      "tennis_000",
      "--seed",
      "0",
      "--start-frame",
      "45",
      "--mode",
      "combined",
      "--keyframe-ticks",
      "18,38",
      "--goal-ticks",
      "28",
      "--strength",
      "0.1",
      "--root-position-weight",
      "1.0",
      "--root-orientation-weight",
      "1.0",
      "--body-positions-weight",
      "1.0",
      "--output",
      str(config_path),
    ]
  )
  assert code == 0, capsys.readouterr().out
  spec = task_config.load_task_config(config_path)
  assert spec.source.qualification == "verified"
  assert spec.source.verification.startswith("loader_recheck")
  assert spec.keyframe_ticks == (18, 38) and spec.goal_ticks == (28,)
  assert spec.source.member == "tennis_000"
  rec = spec.source.verification_detail["loader_recheck"]
  assert "post_pre_consistency" in rec["checks"]
  assert rec["scenario"]["seed"] == 0
  # The re-check is repeatable and survives a fresh read of the written bytes.
  assert (
    task_config.load_task_config(config_path, require_verified=True).config_sha256
    == spec.config_sha256
  )


def test_forged_verified_source_is_refused(tmp_path: Path) -> None:
  payload = json.loads(
    (EXAMPLE_DIR / "synthetic_combined.example.json").read_text(encoding="utf-8")
  )
  payload["source"] = {
    "kind": "baseline_recording",
    "qualification": "verified",
    "verification": "preparation_tool:forged_without_checks",
    "provenance": "forged",
    "path": str(tmp_path / "missing" / "ticks.jsonl"),
    "sha256": "f" * 64,
  }
  with pytest.raises(task_config.TaskConfigError):
    task_config.parse_task_config(
      payload, contract=DEFAULT_CONTRACT, config_sha256="a" * 64
    )


def test_scenario_and_modified_targets_are_refused(tmp_path: Path) -> None:
  recording = _write_d3_recording(tmp_path / "recording")
  config_path = tmp_path / "task.json"
  assert (
    prepare_task_config.main(
      [
        "--recording-dir",
        str(recording),
        "--member",
        "tennis_000",
        "--seed",
        "0",
        "--start-frame",
        "45",
        "--mode",
        "keyframe",
        "--keyframe-ticks",
        "18",
        "--output",
        str(config_path),
      ]
    )
    == 0
  )
  # A different run scenario must not accept the same recording.
  with pytest.raises(task_config.TaskConfigError, match="scenario does not match"):
    task_config.load_task_config(
      config_path,
      expected_scenario=task_config.ScenarioExpectation(
        member="tennis_001", seed=0, start_frame=45
      ),
    )
  # A modified target state, a modified source file and a truncated recording
  # are all refused by the re-check.
  payload = json.loads(config_path.read_text(encoding="utf-8"))
  payload["keyframes"][0]["state"]["root_position"] = [
    float(value) + 0.5 for value in payload["keyframes"][0]["state"]["root_position"]
  ]
  modified = tmp_path / "modified.json"
  modified.write_text(json.dumps(payload), encoding="utf-8")
  with pytest.raises(task_config.TaskConfigError, match="does not match the measured"):
    task_config.load_task_config(modified)

  ticks_path = recording / "ticks.jsonl"
  original = ticks_path.read_text(encoding="utf-8")
  ticks_path.write_text(original + '{"branch": "baseline", "tick": {"tick": 99}}\n')
  with pytest.raises(task_config.TaskConfigError):
    task_config.load_task_config(config_path)


def test_recording_rejects_unqualified_summaries(tmp_path: Path) -> None:
  for name, kwargs in (
    ("status", {"status": "partial"}),
    ("closed_loop", {"closed_loop": False}),
    ("synthetic", {"synthetic": True}),
    ("denoiser", {"denoiser_calls": 20}),
  ):
    recording = _write_d3_recording(tmp_path / name, **cast(Any, kwargs))
    with pytest.raises(task_config.TaskConfigError):
      task_config.read_d3_recording(
        recording / "ticks.jsonl",
        summary_path=recording / "summary.json",
        contract=DEFAULT_CONTRACT,
      )
