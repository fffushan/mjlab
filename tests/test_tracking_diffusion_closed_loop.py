"""CPU-only tests for the simulator-independent D3 evaluation runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import torch
from tracking_diffusion_fixtures import make_state

from mjlab.tasks.tracking.diffusion.closed_loop import (
  MIN_OUTPUT_CAP_BYTES,
  ClosedLoopEvaluationError,
  EvaluationBudget,
  EvaluationReport,
  persist_evaluation_report,
  run_matched_pair,
)
from mjlab.tasks.tracking.diffusion.closed_loop_types import D3Observation, D3Transition
from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT
from mjlab.tasks.tracking.diffusion.handover import (
  ContinuationStep,
  ReferenceFreeContinuation,
)
from mjlab.tasks.tracking.diffusion.projection import (
  ProjectionBundle,
  fit_feature_stats,
)
from mjlab.tasks.tracking.diffusion.schedule import DiffusionSchedule, InferenceGrid
from mjlab.tasks.tracking.diffusion.state import WorldState, world_to_hybrid


@dataclass
class _FakeVae:
  encode_calls: int = 0
  decode_calls: int = 0

  def encode(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    self.encode_calls += 1
    assert reference.shape == (1, 68)
    mu = torch.zeros((1, 32), dtype=torch.float32, device=reference.device)
    logvar = torch.zeros_like(mu)
    return mu, logvar

  def decode(self, latent: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
    self.decode_calls += 1
    assert latent.shape == (1, 32)
    assert conditioning.shape == (1, 99)
    return torch.zeros((1, 31), dtype=torch.float32, device=latent.device)


class _IdentityDenoiser:
  def __call__(self, sample: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
    return 0.1 * sample


class _ZeroDecoder:
  def __call__(self, latent: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
    return torch.zeros((1, 31), dtype=torch.float32, device=latent.device)


def _projection() -> ProjectionBundle:
  rng = np.random.default_rng(3)
  state_stats = fit_feature_stats(
    [rng.normal(size=(64, DEFAULT_CONTRACT.state_dimension))], std_floor=1e-6
  )
  latent_stats = fit_feature_stats(
    [rng.normal(size=(64, DEFAULT_CONTRACT.latent_dimension))], std_floor=1e-6
  )
  return ProjectionBundle.create(state_stats, latent_stats)


class _FakeRuntime:
  period_seconds = 0.02
  clip_frames = 100
  joint_order = tuple(f"joint-{index}" for index in range(31))
  body_order = tuple(f"body-{index}" for index in range(20))
  provenance: dict[str, Any] = {"kind": "fake-test", "synthetic": True}

  def __init__(
    self, *, fall_on_last: bool = False, condition_offset: float = 0.0
  ) -> None:
    self.fall_on_last = fall_on_last
    self.condition_offset = condition_offset
    self.closed = False
    self.entered_reference_free = False
    self.reset_calls: list[tuple[int, int]] = []
    self.actions: list[np.ndarray] = []
    self._index = 0
    self._tick = 0

  def _observation(self) -> D3Observation:
    state = make_state(self._index)
    if self.fall_on_last and self._index > 8:
      state = WorldState(
        state.root_position,
        np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]),
        state.root_linear_velocity,
        state.root_angular_velocity,
        state.body_positions,
        state.body_linear_velocities,
      )
    return D3Observation(
      state=state,
      conditioning=np.full(99, 0.25 + self.condition_offset),
      tick=self._tick,
      sim_time_s=self._tick * self.period_seconds,
      segment_id="segment",
      reference=None if self.entered_reference_free else np.full(68, 0.5),
      reference_frame=None if self.entered_reference_free else self._index,
    )

  def reset(self, *, seed: int, start_frame: int) -> D3Observation:
    self.reset_calls.append((seed, start_frame))
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
    return D3Transition(self._observation())

  def close(self) -> None:
    self.closed = True


class _FakeContinuation:
  def __init__(self) -> None:
    self.records: list[tuple[WorldState, np.ndarray]] = []
    self.step_calls = 0

  def record_executed(self, state: WorldState, latent: np.ndarray) -> None:
    self.records.append((state, np.array(latent, copy=True)))

  def step(
    self, state: WorldState, conditioning: np.ndarray, *, tick: int
  ) -> ContinuationStep:
    self.step_calls += 1
    next_index = int(round((state.root_position[0] - 1.0) / 0.01)) + 1
    next_state = make_state(next_index)
    predicted, _ = world_to_hybrid(
      WorldState.stack([state, next_state]), center_index=0
    )
    return ContinuationStep(
      tick=tick,
      action=np.zeros(31),
      executed_latent=np.full(32, 0.1),
      denoiser_calls=3,
      final_index=0,
      predicted_next_hybrid_state=predicted[1],
    )


class _FlagRuntime(_FakeRuntime):
  def __init__(self, flag: str) -> None:
    super().__init__()
    self.flag = flag

  def step(self, action: np.ndarray) -> D3Transition:
    transition = super().step(action)
    if self._index != 9:
      return transition
    if self.flag == "tick":
      observation = transition.observation
      observation = D3Observation(
        state=observation.state,
        conditioning=observation.conditioning,
        tick=observation.tick + 2,
        sim_time_s=observation.sim_time_s,
        segment_id=observation.segment_id,
        reference=observation.reference,
        reference_frame=observation.reference_frame,
      )
      return D3Transition(observation)
    return D3Transition(
      transition.observation,
      terminal_available=self.flag != "missing_terminal",
      terminated=self.flag in {"terminated", "terminated_teleport"},
      truncated=self.flag == "timeout",
      reset=self.flag == "reset",
      teleport=self.flag in {"teleport", "terminated_teleport"},
      reference_boundary=self.flag == "reference_boundary",
    )


def _run(
  *,
  budget: EvaluationBudget | None = None,
  runtime_factory=None,
  continuation_factory=None,
  model=None,
  **kwargs,
) -> EvaluationReport:
  vae = _FakeVae() if model is None else model

  def make_runtime() -> _FakeRuntime:
    return _FakeRuntime() if runtime_factory is None else runtime_factory()

  def make_continuation() -> _FakeContinuation:
    return (
      _FakeContinuation() if continuation_factory is None else continuation_factory()
    )

  return run_matched_pair(
    make_runtime,
    model=vae,
    continuation_factory=make_continuation,
    seed=7,
    start_frame=0,
    budget=budget or EvaluationBudget(max_steps=2),
    **kwargs,
  )


def test_handover_exposes_row9_from_same_sampler_invocation() -> None:
  projection = _projection()
  schedule = DiffusionSchedule.build()
  continuation = ReferenceFreeContinuation(
    denoiser=_IdentityDenoiser(),
    decode=_ZeroDecoder(),
    schedule=schedule,
    grid=InferenceGrid.uniform(schedule),
    projection=projection,
    generator=torch.Generator().manual_seed(11),
  )
  for index in range(8):
    continuation.record_executed(make_state(index), np.zeros(32))
  step = continuation.step(make_state(8), np.zeros(99), tick=8)
  assert step.predicted_next_hybrid_state is not None
  assert step.predicted_next_hybrid_state.shape == (135,)
  assert np.isfinite(step.predicted_next_hybrid_state).all()


def test_matched_runner_bootstraps_eight_pairs_then_never_encodes_diffusion() -> None:
  model = _FakeVae()
  report = _run(model=model)
  assert report.paired is True
  assert report.closed_loop is False
  assert report.synthetic is True
  assert report.baseline is not None and report.diffusion is not None
  assert report.baseline.bootstrap_completed == 8
  assert report.diffusion.bootstrap_completed == 8
  assert report.baseline.evaluation_completed == 2
  assert report.diffusion.evaluation_completed == 2
  assert model.encode_calls == 18  # 8 + 2 baseline, then 8 bootstrap only.
  assert model.decode_calls == 18
  assert report.diffusion.counters["encoder"] == 8
  assert report.diffusion.counters["denoiser"] == 6


def test_runner_rejects_bootstrap_mismatch_before_diffusion_handover() -> None:
  calls = 0

  def make_runtime() -> _FakeRuntime:
    nonlocal calls
    calls += 1
    runtime = _FakeRuntime()
    if calls == 2:
      original = runtime.initial_signature
      runtime.initial_signature = lambda: original() + 1.0
    return runtime

  report = _run(runtime_factory=make_runtime)
  assert report.paired is False
  assert (
    report.pairing["reason"] == "initial_signature_width_mismatch"
    or report.pairing["initial_signature_max_abs_diff"] == 1.0
  )
  assert report.diffusion is not None
  assert report.diffusion.stop_reason == "pair_mismatch"
  assert report.diffusion.evaluation_attempted == 0


def test_runner_keeps_pre_action_state_for_history_and_prediction_metrics() -> None:
  report = _run()
  assert report.diffusion is not None
  diffusion_ticks = [
    tick for tick in report.diffusion.ticks if tick["phase"] == "diffusion"
  ]
  assert diffusion_ticks
  first = diffusion_ticks[0]
  assert first["state"]["root_position"][0] == pytest.approx(1.08)
  assert first["prediction"]["available"] is True
  for metric in first["prediction"]["blocks"].values():
    assert metric["max_abs"] == pytest.approx(0.0, abs=1e-12)


def test_last_step_fall_precedes_boundary_and_is_not_success() -> None:
  def factory() -> _FakeRuntime:
    runtime = _FakeRuntime(fall_on_last=True)
    runtime.clip_frames = 9
    return runtime

  report = _run(runtime_factory=factory, budget=EvaluationBudget(max_steps=1))
  assert report.baseline is not None and report.diffusion is not None
  assert report.baseline.stop_reason == "physical_fall"
  assert report.diffusion.stop_reason == "physical_fall"
  assert report.baseline.survival_observed is False


def test_invalid_period_and_output_destination_are_fail_closed(tmp_path) -> None:
  class BadRuntime(_FakeRuntime):
    period_seconds = 0.01

  with pytest.raises(ClosedLoopEvaluationError, match="output"):
    existing = tmp_path / "run"
    existing.mkdir()
    (existing / "old.json").write_text("old")
    _run(output_dir=existing)

  report = _run(runtime_factory=BadRuntime)
  assert report.baseline is not None
  assert report.baseline.stop_reason == "runtime_error"
  assert report.diffusion is not None
  assert report.diffusion.stop_reason == "runtime_error"


def test_output_is_atomic_and_bounded(tmp_path) -> None:
  output = tmp_path / "new-run"
  report = _run(
    output_dir=output,
    budget=EvaluationBudget(max_steps=1, output_cap_bytes=MIN_OUTPUT_CAP_BYTES),
  )
  assert report.output["directory"] == str(output)
  assert (output / "summary.json").exists()
  assert (output / "ticks.jsonl").exists()
  assert (output / "summary.json").stat().st_size <= MIN_OUTPUT_CAP_BYTES
  assert (output / "summary.json").read_text().endswith("\n")


def test_budget_rejects_nonintegral_or_non_eight_bootstrap() -> None:
  with pytest.raises(ClosedLoopEvaluationError, match="integer"):
    EvaluationBudget(max_steps=True)
  with pytest.raises(ClosedLoopEvaluationError, match="exactly 8"):
    EvaluationBudget(bootstrap_pairs=7)
  with pytest.raises(ClosedLoopEvaluationError, match="period"):
    EvaluationBudget(period_seconds=0.01)


@pytest.mark.parametrize(
  ("flag", "reason"),
  (
    ("missing_terminal", "missing_terminal_evidence"),
    ("terminated", "terminated"),
    ("timeout", "timeout"),
    ("reset", "reset"),
    ("terminated_teleport", "terminated"),
    ("reference_boundary", "reference_boundary"),
    ("tick", "tick_discontinuity"),
  ),
)
def test_boundary_and_tick_evidence_stops_without_bad_history(
  flag: str, reason: str
) -> None:
  continuations: list[_FakeContinuation] = []

  def make_continuation() -> _FakeContinuation:
    continuation = _FakeContinuation()
    continuations.append(continuation)
    return continuation

  report = _run(
    runtime_factory=lambda: _FlagRuntime(flag),
    continuation_factory=make_continuation,
    budget=EvaluationBudget(max_steps=1),
  )
  assert report.baseline is not None and report.diffusion is not None
  assert report.baseline.stop_reason == reason
  assert report.diffusion.stop_reason == reason
  assert report.paired is True
  assert continuations[0].step_calls == 1
  assert len(continuations[0].records) == 8


def test_bootstrap_conditioning_mismatch_invalidates_pair() -> None:
  calls = 0

  def factory() -> _FakeRuntime:
    nonlocal calls
    calls += 1
    return _FakeRuntime(condition_offset=1.0 if calls == 2 else 0.0)

  report = _run(runtime_factory=factory)
  assert report.paired is False
  assert report.pairing["reason"] == "bootstrap_or_initialization_mismatch"
  assert report.diffusion is not None
  assert report.diffusion.stop_reason == "pair_mismatch"


def test_default_comparison_tolerance_is_the_user_selected_one_e_minus_three() -> None:
  assert EvaluationBudget(max_steps=2).comparison_tolerance == pytest.approx(1.0e-3)
  assert _run().pairing["tolerance"] == pytest.approx(1.0e-3)


@pytest.mark.parametrize("offset", [1.0e-6, 1.0e-4])
def test_subthreshold_bootstrap_micro_differences_pair_under_default(
  offset: float,
) -> None:
  calls = 0

  def factory() -> _FakeRuntime:
    nonlocal calls
    calls += 1
    return _FakeRuntime(condition_offset=offset if calls == 2 else 0.0)

  report = _run(runtime_factory=factory)
  assert report.pairing["conditioning_max_abs_diff"] == pytest.approx(offset)
  assert report.paired is True
  assert report.status == "complete"


def test_above_threshold_bootstrap_mismatch_is_rejected() -> None:
  calls = 0

  def factory() -> _FakeRuntime:
    nonlocal calls
    calls += 1
    return _FakeRuntime(condition_offset=1.0e-2 if calls == 2 else 0.0)

  report = _run(runtime_factory=factory)
  assert report.pairing["conditioning_max_abs_diff"] == pytest.approx(1.0e-2)
  assert report.paired is False
  assert report.pairing["reason"] == "bootstrap_or_initialization_mismatch"
  assert report.diffusion is not None
  assert report.diffusion.stop_reason == "pair_mismatch"


def test_bad_vae_width_is_a_partial_attempt_not_a_success() -> None:
  class BadVae(_FakeVae):
    def encode(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
      self.encode_calls += 1
      return torch.zeros((1, 31)), torch.zeros((1, 32))

  report = _run(model=BadVae())
  assert report.baseline is not None and report.diffusion is not None
  assert report.baseline.stop_reason == "inference_error"
  assert report.baseline.attempted_ticks == 1
  assert report.diffusion.stop_reason == "inference_error"


def test_height_is_descriptive_and_runtimes_are_closed() -> None:
  runtimes: list[_FakeRuntime] = []

  def factory() -> _FakeRuntime:
    runtime = _FakeRuntime()
    runtimes.append(runtime)
    return runtime

  report = _run(runtime_factory=factory, budget=EvaluationBudget(max_steps=0))
  assert report.baseline is not None
  assert report.baseline.minimum_pelvis_height_m == pytest.approx(0.5)
  assert report["claims"]["pelvis_height"] == "descriptive-only"
  assert all(runtime.closed for runtime in runtimes)


def test_post_bootstrap_time_budget_allows_five_ticks_at_point_one_seconds() -> None:
  report = _run(
    budget=EvaluationBudget(max_steps=10, max_sim_time_s=0.1),
  )
  assert report.baseline is not None and report.diffusion is not None
  assert report.baseline.evaluation_completed == 5
  assert report.diffusion.evaluation_completed == 5
  assert report.baseline.stop_reason == "time_budget"
  assert report.diffusion.stop_reason == "time_budget"


def test_factory_failure_and_synthetic_runtime_are_not_closed_loop() -> None:
  report = _run(runtime_factory=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
  assert report.closed_loop is False
  assert report.status == "partial"
  assert report.valid is False

  class PhysicalMarkerStub(_FakeRuntime):
    provenance = {"physical_runtime": "mjlab", "synthetic": False}

  stub_report = _run(
    runtime_factory=PhysicalMarkerStub,
    provenance={"synthetic": True, "test_stub": True},
  )
  assert stub_report.closed_loop is False
  assert stub_report.synthetic is True


def test_teleported_terminal_state_is_not_used_for_fall_or_metrics() -> None:
  report = _run(
    runtime_factory=lambda: _FlagRuntime("terminated_teleport"),
    budget=EvaluationBudget(max_steps=1),
  )
  assert report.baseline is not None and report.diffusion is not None
  assert report.baseline.stop_reason == "terminated"
  assert report.diffusion.stop_reason == "terminated"
  assert report.baseline.peak_tilt_degrees == pytest.approx(0.0)
  assert report.baseline.ticks[-1]["post"] is None
  assert report.status == "partial"


def test_final_provenance_is_deep_copied_before_runtime_close() -> None:
  runtimes: list[_FakeRuntime] = []

  def factory() -> _FakeRuntime:
    runtime = _FakeRuntime()
    runtime.provenance = {"nested": {"value": 1}, "synthetic": True}
    runtimes.append(runtime)
    return runtime

  report = _run(runtime_factory=factory)
  assert report.baseline is not None and report.diffusion is not None
  runtimes[0].provenance["nested"]["value"] = 99
  assert report.baseline.provenance["nested"]["value"] == 1
  assert "initial_state_signature_digest" in report.baseline.provenance
  assert report.diffusion.provenance["evaluator_handover"]["entered"] is True


def test_public_persistence_is_bounded_and_final_state_reusable(tmp_path) -> None:
  output = tmp_path / "persisted"
  report = _run(
    output_dir=output,
    budget=EvaluationBudget(max_steps=1, output_cap_bytes=MIN_OUTPUT_CAP_BYTES),
  )
  provenance = dict(report.provenance)
  provenance["integrity"] = {"unchanged": True}
  provenance["execution"] = {"wall_status": "within_budget"}
  object.__setattr__(report, "provenance", provenance)
  persisted = persist_evaluation_report(report, output)
  assert persisted is report
  assert report.output["representation"] in {"compact", "marker"}
  assert report.output["truncated"] is True
  assert report.output["omitted_row_count"] > 0
  assert report.output["actual_bytes"] <= MIN_OUTPUT_CAP_BYTES
  total = (output / "summary.json").stat().st_size + (
    output / "ticks.jsonl"
  ).stat().st_size
  assert total <= MIN_OUTPUT_CAP_BYTES
  marker_or_compact = (output / "summary.json").read_text()
  assert '"status"' in marker_or_compact


@pytest.mark.parametrize("unchanged", [True, False])
def test_minimum_cap_final_marker_keeps_large_integrity_status_and_digests(
  tmp_path, unchanged: bool
) -> None:
  output = tmp_path / "large-integrity"
  report = _run(
    output_dir=output,
    budget=EvaluationBudget(max_steps=1, output_cap_bytes=MIN_OUTPUT_CAP_BYTES),
  )
  # Real CLI snapshots retain transformer tensor-key names and extra state.
  # A marker must not copy this unbounded evidence into its 4 KiB budget.
  before = {
    "models": {
      "diffusion": {
        "state_dict_sha256": "a" * 64,
        "tensor_keys": [
          f"transformer.layers.{index}.attention.in_proj_weight" * 3
          for index in range(300)
        ],
        "non_tensor_extra_state": {"schema": {"token_dimension": 167}},
      }
    },
    "files": {"checkpoint": {"actual_sha256": "b" * 64}},
  }
  after = {**before, "comparison": "same" if unchanged else "changed"}
  assert len(json.dumps(before)) > 10 * MIN_OUTPUT_CAP_BYTES
  provenance = dict(report.provenance)
  provenance["integrity"] = {
    "before": before,
    "after": after,
    "unchanged": unchanged,
  }
  provenance["execution"] = {
    "wall_seconds": 1.25,
    "max_wall_seconds": 1200.0,
    "within_wall_budget": True,
    "partial_outcomes_retained": True,
  }
  object.__setattr__(report, "provenance", provenance)
  persist_evaluation_report(report, output)
  assert report.output["error"] is None
  assert report.output["representation"] == "marker"
  assert report.output["actual_bytes"] <= MIN_OUTPUT_CAP_BYTES
  total = (output / "summary.json").stat().st_size + (
    output / "ticks.jsonl"
  ).stat().st_size
  assert total <= MIN_OUTPUT_CAP_BYTES
  payload = json.loads((output / "summary.json").read_text())
  integrity = payload["after_run"]["integrity"]
  assert integrity["unchanged"] is unchanged
  assert len(integrity["before_sha256"]) == 64
  assert len(integrity["after_sha256"]) == 64
  assert integrity["before_sha256"] != integrity["after_sha256"]
  assert "tensor_keys" not in (output / "summary.json").read_text()
  assert payload["after_run"]["execution"]["within_wall_budget"] is True
  assert payload["synthetic"] is True
  assert payload["closed_loop"] is False
  assert payload["output"]["omitted_row_count"] > 0


def test_output_cap_below_public_minimum_is_rejected_before_runtime() -> None:
  calls = 0

  def factory() -> _FakeRuntime:
    nonlocal calls
    calls += 1
    return _FakeRuntime()

  with pytest.raises(ClosedLoopEvaluationError, match="at least"):
    run_matched_pair(
      factory,
      model=_FakeVae(),
      continuation_factory=_FakeContinuation,
      seed=0,
      start_frame=0,
      budget=EvaluationBudget(output_cap_bytes=MIN_OUTPUT_CAP_BYTES - 1),
    )
  assert calls == 0
