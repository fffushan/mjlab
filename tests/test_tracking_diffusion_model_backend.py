"""Focused tests for the concrete D5 model-backend assembly."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from torch import nn
from tracking_diffusion_fixtures import make_state

from mjlab.tasks.tracking.diffusion import model_backend
from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT
from mjlab.tasks.tracking.diffusion.latent_planning import build_planning_packet
from mjlab.tasks.tracking.diffusion.model import StateLatentTransformer
from mjlab.tasks.tracking.diffusion.projection import (
  FeatureStats,
  StateLatentNormalizer,
)
from mjlab.tasks.tracking.diffusion.replay_integration import LoadedDenoiser, LoadedVae
from mjlab.tasks.tracking.diffusion.schedule import DiffusionSchedule, InferenceGrid
from mjlab.tasks.tracking.diffusion.scheduler import PlannerRequest


def _normalizer() -> StateLatentNormalizer:
  return StateLatentNormalizer.create(
    FeatureStats(1, np.zeros(135), np.ones(135)),
    FeatureStats(1, np.zeros(32), np.ones(32)),
  )


def _spec(tmp_path: Path) -> model_backend.ModelBackendSpec:
  for name in ("contract", "checkpoint", "normalizer", "vae", "manifest"):
    (tmp_path / f"{name}.bin").write_bytes(b"fixture")
  return model_backend.ModelBackendSpec(
    contract_path=tmp_path / "contract.bin",
    diffusion_checkpoint=tmp_path / "checkpoint.bin",
    normalization_bundle=tmp_path / "normalizer.bin",
    vae_checkpoint=tmp_path / "vae.bin",
    vae_manifest=tmp_path / "manifest.bin",
    repo_root=tmp_path,
    vae_member="member-0",
  )


class _Denoiser(nn.Module):
  def forward(self, sample: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
    del step_ids
    return 0.2 * sample


def test_backend_spec_is_spawnable_and_does_not_require_hashes(tmp_path: Path) -> None:
  spec = _spec(tmp_path)
  assert pickle.loads(pickle.dumps(spec)) == spec
  assert spec.resolved_model_identity.startswith("diffusion:")
  assert spec.resolved_vae_identity.endswith(":member-0")
  spec.validate_paths()


def test_backend_spec_rejects_missing_inputs(tmp_path: Path) -> None:
  spec = _spec(tmp_path)
  (tmp_path / "checkpoint.bin").unlink()
  with pytest.raises(model_backend.ModelBackendError, match="diffusion_checkpoint"):
    spec.validate_paths()


def test_planner_factory_loads_inside_factory_and_returns_raw_plan_processor(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  spec = _spec(tmp_path)
  normalizer = _normalizer()
  schedule = DiffusionSchedule.build()
  loaded = LoadedDenoiser(
    model=cast(StateLatentTransformer, _Denoiser()),
    schedule=schedule,
    grid=InferenceGrid.uniform(schedule),
    global_step=1,
    config_hash="a" * 64,
    contract_identity=DEFAULT_CONTRACT.identity_hash(),
    projection_hashes={},
    best_validation_loss=None,
    selection_split=None,
    ema_selected=False,
    ema_updates=None,
    fp32_checked=True,
    config_seed=None,
  )
  calls: list[dict[str, object]] = []

  monkeypatch.setattr(
    model_backend, "load_active_contract", lambda path: DEFAULT_CONTRACT
  )
  monkeypatch.setattr(
    model_backend,
    "load_normalizer_bundle",
    lambda path, *, contract: normalizer,
  )

  def fake_load_denoiser(path, **kwargs):
    calls.append({"path": path, **kwargs})
    return loaded

  monkeypatch.setattr(model_backend, "load_denoiser", fake_load_denoiser)
  factory = model_backend.ModelPlannerFactory(spec)
  assert pickle.loads(pickle.dumps(factory)) == factory
  processor = factory()
  assert callable(processor)
  assert calls[0]["checkpoint_sha256"] is None
  assert calls[0]["verify_projection_hashes"] is False
  assert calls[0]["verify_contract_identity"] is False

  packet = build_planning_packet(
    [(make_state(index), np.zeros(32)) for index in range(8)],
    make_state(8),
    normalizer=normalizer,
    source_timestamp=0.0,
    source_tick=0,
    source_frame=0,
    rng=torch.Generator(device="cpu").manual_seed(4),
    vae_identity=spec.resolved_vae_identity,
    model_identity=spec.resolved_model_identity,
  )
  request = PlannerRequest(
    packet.input_snapshot,
    packet.source_timestamp,
    packet.source_tick,
    1,
    0,
    context=packet.as_context(),
  )
  plan = processor(request)
  assert plan.actions.shape == (33, 32)
  assert plan.schema_metadata["representation"] == "latent/raw/32"


def test_owner_backend_loads_decoder_without_digest_gate(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  spec = _spec(tmp_path)
  normalizer = _normalizer()

  class _Decoder(nn.Module):
    def decode(self, latent: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
      return torch.zeros((latent.shape[0], 31), dtype=latent.dtype)

  loaded_vae = LoadedVae(
    model=_Decoder(),
    member_id=spec.vae_member,
    checkpoint_sha256=None,
    manifest_sha256="manifest-not-used",
    relocated_artifact_roles=(),
    joint_order=(),
    conditioning_dim=99,
    reference_dim=68,
    latent_dim=32,
    action_dim=31,
  )
  captured: dict[str, object] = {}
  monkeypatch.setattr(
    model_backend, "load_active_contract", lambda path: DEFAULT_CONTRACT
  )
  monkeypatch.setattr(
    model_backend,
    "load_normalizer_bundle",
    lambda path, *, contract: normalizer,
  )

  def fake_load_vae(path, **kwargs):
    captured.update(kwargs)
    return loaded_vae

  monkeypatch.setattr(model_backend, "load_vae_decoder", fake_load_vae)
  backend = model_backend.load_model_backend(spec)
  assert backend.contract is DEFAULT_CONTRACT
  assert backend.normalizer is normalizer
  assert backend.projection is normalizer
  assert getattr(backend.decoder, "__self__", None) is loaded_vae.model
  assert captured["vae_checkpoint_sha256"] is None
  assert captured["device"] == torch.device("cpu")


def test_statistics_only_normalizer_round_trip_has_no_matrix_fields(
  tmp_path: Path,
) -> None:
  normalizer = _normalizer()
  path = tmp_path / "normalizer.npz"
  normalizer.save(path)
  with np.load(path, allow_pickle=False) as data:
    assert set(data.files) == {
      "state_mean",
      "state_std",
      "latent_mean",
      "latent_std",
      "metadata",
    }
  loaded = StateLatentNormalizer.load(path)
  tokens = loaded.make_tokens(np.zeros((1, 135)), np.zeros((1, 32)))
  np.testing.assert_array_equal(tokens, np.zeros((1, 167)))
