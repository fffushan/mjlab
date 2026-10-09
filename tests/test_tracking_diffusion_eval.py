"""Tests for offline generation metrics and baseline gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from mjlab.tasks.tracking.diffusion import evaluation as evaluation_module
from mjlab.tasks.tracking.diffusion.contract import DiffusionContract
from mjlab.tasks.tracking.diffusion.evaluation import (
  _baseline_tokens,
  audit_dataset,
  evaluate_generation,
  evaluate_pathwise_transition_diagnostic,
)
from mjlab.tasks.tracking.diffusion.noising import apply_clean_mask, clean_mask
from mjlab.tasks.tracking.diffusion.projection import (
  FeatureStats,
  ProjectionBundle,
)
from mjlab.tasks.tracking.diffusion.schedule import (
  DiffusionSchedule,
  build_inference_grid,
)
from mjlab.tasks.tracking.diffusion.window_dataset import (
  DatasetSource,
  WindowProvenance,
  WindowRecord,
  _cache_metadata,
)


class IdentityProjection:
  """Identity-shaped projection seam for fast normalized/physical tests."""

  def inverse_state(self, values: np.ndarray) -> np.ndarray:
    return values

  def denormalize_latent(self, values: np.ndarray) -> np.ndarray:
    return values


def _record(phase: str) -> WindowRecord:
  generator = np.random.default_rng(12 if phase == "clean" else 13)
  tokens = generator.normal(size=(41, 167)).astype(np.float32)
  return WindowRecord(
    tokens=tokens,
    split="test",
    group_key=f"group-{phase}",
    motion_id="tennis_000",
    phase=phase,
    ou_noise_norm=0.0 if phase == "clean" else 1.0,
    pair_id=f"pair-{phase}",
    start_tick=0,
  )


class EchoDenoiser(nn.Module):
  """Small deterministic denoiser that keeps the sampler path inexpensive."""

  def forward(self, noisy_tokens: Tensor, step_ids: Tensor) -> Tensor:
    del step_ids
    return noisy_tokens


def test_pathwise_transition_diagnostic_records_matched_noise_distributions(
  tmp_path: Path,
) -> None:
  contract = DiffusionContract()
  schedule = DiffusionSchedule.build()
  grid = build_inference_grid(schedule, contract=contract)
  report = evaluate_pathwise_transition_diagnostic(
    EchoDenoiser(),
    schedule=schedule,
    grid=grid,
    records=[_record("clean")],
    projection=IdentityProjection(),  # type: ignore[arg-type]
    contract=contract,
    seed=5,
    device="cpu",
    diagnostics_path=tmp_path / "pathwise.json",
  )
  assert report["kind"] == "pathwise_matched_noise_transition_diagnostic"
  assert report["windows"] == 1
  transitions = report["transitions"]
  assert isinstance(transitions, list)
  assert len(transitions) == 20
  first = transitions[0]
  assert isinstance(first, dict)
  first = cast(dict[str, Any], first)
  errors = first["errors"]
  assert isinstance(errors, dict)
  assert set(errors) == {
    "evolving_vs_q_source",
    "evolving_x0_error",
    "correctly_noised_x0_error",
    "destination_vs_q_destination",
  }
  assert "normalized_state" in errors["evolving_x0_error"]
  assert "normalized_projected_state_added" not in errors["evolving_x0_error"]
  assert "physical_recovered_latent" in errors["destination_vs_q_destination"]
  assert (tmp_path / "pathwise.json").is_file()


def test_pathwise_uses_one_epsilon_at_every_level_and_preserves_generation(
  tmp_path: Path,
) -> None:
  class RecordingDenoiser(EchoDenoiser):
    def __init__(self) -> None:
      super().__init__()
      self.inputs: list[Tensor] = []
      self.ids: list[Tensor] = []

    def forward(self, noisy_tokens: Tensor, step_ids: Tensor) -> Tensor:
      self.inputs.append(noisy_tokens.clone())
      self.ids.append(step_ids.clone())
      return super().forward(noisy_tokens, step_ids)

  contract = DiffusionContract()
  schedule = DiffusionSchedule.build()
  grid = build_inference_grid(schedule)
  record = _record("clean")
  model = RecordingDenoiser()
  projection = IdentityProjection()
  kwargs: dict[str, Any] = dict(
    schedule=schedule,
    grid=grid,
    records=[record],
    projection=projection,
    contract=contract,
    seed=5,
    device="cpu",
  )
  baseline = evaluate_generation(EchoDenoiser(), **kwargs)
  traced = evaluate_generation(
    model,
    **kwargs,
    pathwise_transition_diagnostics_path=tmp_path / "pathwise.json",
    reverse_diagnostics_path=tmp_path / "reverse.json",
  )
  assert traced.metrics == baseline.metrics
  assert traced.baselines == baseline.baselines
  reverse = json.loads((tmp_path / "reverse.json").read_text())
  assert [s["source_id"] for s in reverse["steps"]] == list(grid.source_ids)
  clean = torch.from_numpy(record.tokens).unsqueeze(0)
  mask = clean_mask(
    current_index=contract.current_index,
    state_dimension=contract.projected_state_dimension,
    token_dimension=contract.token_dimension,
    device="cpu",
  )
  epsilon = torch.randn(clean.shape, generator=torch.Generator().manual_seed(5))
  alpha_bars = schedule.alpha_bars_torch(dtype=clean.dtype)
  for index, source in enumerate(grid.source_ids):
    expected = apply_clean_mask(
      torch.sqrt(alpha_bars[source]) * clean
      + torch.sqrt(1 - alpha_bars[source]) * epsilon,
      clean,
      mask,
    )
    torch.testing.assert_close(model.inputs[2 * index + 1], expected)
    assert torch.equal(model.ids[2 * index], model.ids[2 * index + 1])
    assert torch.equal(model.inputs[2 * index][0][mask], clean[0][mask])
  # The subsequent production trace sees the same evolving inputs, exactly.
  for index in range(20):
    assert torch.equal(model.inputs[2 * index], model.inputs[40 + index])


def test_ddpm_rejects_lower_start_before_running() -> None:
  schedule = DiffusionSchedule.build()
  with pytest.raises(ValueError, match="only supported for DDIM"):
    evaluate_generation(
      EchoDenoiser(),
      schedule=schedule,
      grid=build_inference_grid(schedule),
      records=[],
      projection=IdentityProjection(),  # type: ignore[arg-type]
      contract=DiffusionContract(),
      seed=0,
      device="cpu",
      sampler_kind="ddpm",
      start_level=950,
    )


def test_generation_report_scores_unknown_entries_and_writes_diagnostics(
  tmp_path: Path,
) -> None:
  contract = DiffusionContract()
  schedule = DiffusionSchedule.build()
  grid = build_inference_grid(schedule, contract=contract)
  records = [_record("clean"), _record("ou")]
  report = evaluate_generation(
    EchoDenoiser(),
    schedule=schedule,
    grid=grid,
    records=records,
    projection=IdentityProjection(),  # type: ignore[arg-type]
    contract=contract,
    seed=7,
    device="cpu",
    diagnostics_path=tmp_path / "generation.json",
  )
  assert report.split == "test"
  assert report.windows == 2
  assert report.diagnostics_path.is_file()
  assert report.metrics["model_unknown_state_mse"] >= 0.0
  assert report.metrics["model_unknown_latent_mse"] >= 0.0
  assert "margin_model_vs_copy_current_latent" in report.metrics
  assert any("horizon_9_16" in key for key in report.metrics)
  assert any("phase" in key for key in report.metrics)


def test_generation_can_drop_both_history_state_and_latent_conditions() -> None:
  class RecordingDenoiser(EchoDenoiser):
    def __init__(self) -> None:
      super().__init__()
      self.inputs: list[Tensor] = []
      self.step_ids: list[Tensor] = []

    def forward(self, noisy_tokens: Tensor, step_ids: Tensor) -> Tensor:
      self.inputs.append(noisy_tokens.clone())
      self.step_ids.append(step_ids.clone())
      return super().forward(noisy_tokens, step_ids)

  contract = DiffusionContract()
  model = RecordingDenoiser()
  report = evaluate_generation(
    model,
    schedule=DiffusionSchedule.build(),
    grid=build_inference_grid(DiffusionSchedule.build(), contract=contract),
    records=[_record("clean"), _record("ou")],
    projection=IdentityProjection(),  # type: ignore[arg-type]
    contract=contract,
    seed=29,
    device="cpu",
    history_condition_dropout_probability=1.0,
  )
  assert report.history_condition_dropout_probability == 1.0
  assert report.history_condition_dropout_windows == 2
  first_ids = model.step_ids[0]
  assert torch.equal(first_ids[:, :8, 0], torch.full((2, 8), 1000))
  assert torch.equal(first_ids[:, :8, 1], torch.full((2, 8), 1000))
  assert torch.equal(first_ids[:, 8, 0], torch.zeros(2, dtype=torch.long))
  assert torch.equal(first_ids[:, 8, 1], torch.full((2,), 1000))


def test_generation_uses_raw_state_mask_for_real_projection_shape() -> None:
  contract = DiffusionContract()
  projection = ProjectionBundle.create(
    FeatureStats(
      1, np.zeros(contract.state_dimension), np.ones(contract.state_dimension)
    ),
    FeatureStats(
      1, np.zeros(contract.latent_dimension), np.ones(contract.latent_dimension)
    ),
    contract=contract,
  )
  report = evaluate_generation(
    EchoDenoiser(),
    schedule=DiffusionSchedule.build(),
    grid=build_inference_grid(DiffusionSchedule.build(), contract=contract),
    records=[_record("clean")],
    projection=projection,
    contract=contract,
    seed=3,
    device="cpu",
  )
  assert report.windows == 1
  assert report.metrics["model_physical_unknown_state_mse"] >= 0.0


def test_copy_current_latent_uses_last_known_history_not_target() -> None:
  clean = torch.zeros((1, 41, 167), dtype=torch.float32)
  clean[:, 7, 135:] = 3.0
  clean[:, 8, 135:] = 9.0
  baseline = _baseline_tokens(clean)["copy_current_latent"]
  assert torch.equal(baseline[:, 8:, 135:], torch.full((1, 33, 32), 3.0))
  assert not torch.equal(baseline[:, 8, 135:], clean[:, 8, 135:])


def test_generation_same_seed_is_repeatable(tmp_path: Path) -> None:
  contract = DiffusionContract()
  schedule = DiffusionSchedule.build()
  grid = build_inference_grid(schedule, contract=contract)
  records = [_record("clean")]
  first = evaluate_generation(
    EchoDenoiser(),
    schedule=schedule,
    grid=grid,
    records=records,
    projection=IdentityProjection(),  # type: ignore[arg-type]
    contract=contract,
    seed=19,
    device="cpu",
    diagnostics_path=tmp_path / "first.json",
  )
  second = evaluate_generation(
    EchoDenoiser(),
    schedule=schedule,
    grid=grid,
    records=records,
    projection=IdentityProjection(),  # type: ignore[arg-type]
    contract=contract,
    seed=19,
    device="cpu",
    diagnostics_path=tmp_path / "second.json",
  )
  assert first.metrics == second.metrics
  assert first.baselines == second.baselines


def test_audit_rejects_stale_cache_metadata(tmp_path: Path, monkeypatch) -> None:
  contract_identity = "c" * 64
  contract = SimpleNamespace(
    identity_hash=lambda: contract_identity,
    window_steps=41,
    token_dimension=167,
    projected_state_dimension=135,
    projection_rows=0,
  )
  loaded = SimpleNamespace(
    contract=contract,
    store=SimpleNamespace(
      root=tmp_path / "store",
      row_count=0,
      shard_count=0,
      contract=contract,
    ),
    projection=SimpleNamespace(
      contract=contract,
      matrix_sha256="m" * 64,
      pseudoinverse_sha256="p" * 64,
      statistics_sha256="s" * 64,
    ),
    index=SimpleNamespace(coverage=lambda: {"train": 0, "validation": 0, "test": 0}),
    assignments=SimpleNamespace(sha256=lambda: "a" * 64),
  )
  source = DatasetSource(tmp_path, Path("contract.yaml"))
  monkeypatch.setattr(
    evaluation_module,
    "_audit_source",
    lambda _: (loaded, source, tmp_path),
  )
  monkeypatch.setattr(
    evaluation_module,
    "load_window_records",
    lambda _loaded, _split: [],
  )
  monkeypatch.setattr(
    evaluation_module,
    "iter_window_provenance",
    lambda _loaded, _split, **_: iter(()),
  )
  monkeypatch.setattr(
    evaluation_module,
    "_source_dataset_hash",
    lambda _source, _loaded: "d" * 64,
  )
  monkeypatch.setattr(
    evaluation_module,
    "_projection_identity",
    lambda _projection: "e" * 64,
  )
  offline = tmp_path / "offline"
  offline.mkdir()
  cache = offline / "tokens-test.npy"
  np.save(cache, np.empty((0, 41, 167), dtype=np.float32))
  metadata = _cache_metadata(
    source_hash="d" * 64,
    split="test",
    contract_hash=contract_identity,
    projection_hash="e" * 64,
    window_count=0,
    dtype=np.dtype(np.float32),
    motion_ids=None,
    limit=None,
    shape=(0, 41, 167),
    created_at="now",
    content_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
  )
  cache.with_suffix(".json").write_text(json.dumps(metadata))
  assert audit_dataset(source)["ok"] is True
  metadata["contract_hash"] = "stale"
  cache.with_suffix(".json").write_text(json.dumps(metadata))
  payload = audit_dataset(source)
  assert payload["ok"] is False
  assert payload["cache_sizes"][0]["valid"] is False  # type: ignore[index]


def test_audit_rejects_self_consistent_truncated_cache_from_d1_count(
  tmp_path: Path, monkeypatch
) -> None:
  contract_identity = "c" * 64
  contract = SimpleNamespace(
    identity_hash=lambda: contract_identity,
    window_steps=41,
    token_dimension=167,
    projected_state_dimension=135,
    projection_rows=0,
  )
  loaded = SimpleNamespace(
    contract=contract,
    store=SimpleNamespace(
      root=tmp_path / "store",
      row_count=0,
      shard_count=0,
      contract=contract,
    ),
    projection=SimpleNamespace(
      contract=contract,
      matrix_sha256="m" * 64,
      pseudoinverse_sha256="p" * 64,
      statistics_sha256="s" * 64,
    ),
    index=SimpleNamespace(coverage=lambda: {"train": 0, "validation": 0, "test": 1}),
    assignments=SimpleNamespace(sha256=lambda: "a" * 64),
  )
  source = DatasetSource(tmp_path, Path("contract.yaml"))
  provenance = WindowProvenance(
    split="test",
    group_key="group",
    motion_id="tennis_000",
    phase="clean",
    ou_noise_norm=0.0,
    pair_id="pair",
    start_tick=0,
  )
  monkeypatch.setattr(
    evaluation_module,
    "_audit_source",
    lambda _: (loaded, source, tmp_path),
  )
  monkeypatch.setattr(
    evaluation_module,
    "load_window_records",
    lambda _loaded, _split: [],
  )
  monkeypatch.setattr(
    evaluation_module,
    "iter_window_provenance",
    lambda _loaded, split, **_: iter([provenance]) if split == "test" else iter(()),
  )
  monkeypatch.setattr(
    evaluation_module,
    "_source_dataset_hash",
    lambda _source, _loaded: "d" * 64,
  )
  monkeypatch.setattr(
    evaluation_module,
    "_projection_identity",
    lambda _projection: "e" * 64,
  )
  offline = tmp_path / "offline"
  offline.mkdir()
  cache = offline / "tokens-test.npy"
  np.save(cache, np.empty((0, 41, 167), dtype=np.float32))
  metadata = _cache_metadata(
    source_hash="d" * 64,
    split="test",
    contract_hash=contract_identity,
    projection_hash="e" * 64,
    window_count=0,
    dtype=np.dtype(np.float32),
    motion_ids=None,
    limit=None,
    shape=(0, 41, 167),
    created_at="now",
    content_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
  )
  cache.with_suffix(".json").write_text(json.dumps(metadata))

  payload = audit_dataset(source)
  assert payload["ok"] is False
  cache_payload = payload["cache_sizes"][0]  # type: ignore[index]
  assert cache_payload["valid"] is False
  assert "D1 selection" in cache_payload["metadata_error"]


def test_generation_zero_window_bound_writes_empty_report(tmp_path: Path) -> None:
  contract = DiffusionContract()
  schedule = DiffusionSchedule.build()
  grid = build_inference_grid(schedule, contract=contract)
  report = evaluate_generation(
    EchoDenoiser(),
    schedule=schedule,
    grid=grid,
    records=[_record("clean")],
    projection=IdentityProjection(),  # type: ignore[arg-type]
    contract=contract,
    seed=0,
    device="cpu",
    max_windows=0,
    diagnostics_path=tmp_path / "empty.json",
  )
  assert report.windows == 0
  assert not report.beats_trivial_copy
  assert report.diagnostics_path.is_file()
