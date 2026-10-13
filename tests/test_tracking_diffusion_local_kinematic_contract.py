"""Focused tests for the reduced local-kinematic contract in the model path.

The state/contract lane owns the concrete 129-D contract object.  These tests
use a duck-typed double that declares only the widths the model, checkpoint
loader, planner and preflight read, so the runtime side can be validated before
that lane lands.  The reduced contract removes the 3-D root position and 3-D
root linear velocity slices from the frozen 135-D state, giving 129 state
channels and 161 state+latent token channels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from tracking_diffusion_fixtures import make_state

from mjlab.scripts import diffusion
from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT
from mjlab.tasks.tracking.diffusion.latent_planning import (
  LatentPlanner,
  LatentPlanningError,
  LatentPlanningPacket,
  PlanningPair,
  StateSnapshot,
)
from mjlab.tasks.tracking.diffusion.model import (
  ContractDimensions,
  DenoiserSettings,
  StateLatentTransformer,
  contract_dimensions,
)
from mjlab.tasks.tracking.diffusion.model_backend import _require_contract_layout
from mjlab.tasks.tracking.diffusion.projection import (
  FeatureStats,
  StateLatentNormalizer,
)
from mjlab.tasks.tracking.diffusion.replay_integration import (
  ReplayError,
  load_denoiser,
)
from mjlab.tasks.tracking.diffusion.schedule import (
  DiffusionSchedule,
  build_inference_grid,
)

_REDUCED_STATE_DIMENSION = 129
_REDUCED_TOKEN_DIMENSION = 161
_REDUCED_IDENTITY = "b" * 64


class _ReducedContract:
  """Duck-typed 129-D local-kinematic contract for the runtime path."""

  def __init__(
    self,
    *,
    state_dimension: int = _REDUCED_STATE_DIMENSION,
    token_dimension: int = _REDUCED_TOKEN_DIMENSION,
  ) -> None:
    self.state_dimension = state_dimension
    self.latent_dimension = 32
    self.token_dimension = token_dimension
    self.window_steps = 41
    self.projected_state_dimension = state_dimension
    self.schema_version = DEFAULT_CONTRACT.schema_version
    self.projection_rows = 0
    self.body_count = 20
    self.past_steps = 8
    self.current_index = 8
    self.future_steps = 32
    self.period_seconds = 0.02
    self.action_dimension = 31
    self.recovery_onnx = "recovery.onnx"
    self.recovery_sha256 = "0" * 64
    self.vae_checkpoint = "vae.pt"
    self.vae_sha256 = DEFAULT_CONTRACT.vae_sha256

  def identity_hash(self) -> str:
    return _REDUCED_IDENTITY


def _normalizer(contract: Any) -> StateLatentNormalizer:
  return StateLatentNormalizer.create(
    FeatureStats(
      1, np.zeros(contract.state_dimension), np.ones(contract.state_dimension)
    ),
    FeatureStats(
      1, np.zeros(contract.latent_dimension), np.ones(contract.latent_dimension)
    ),
    contract=contract,
  )


def _packet(*, token_dimension: int, contract_identity: str) -> LatentPlanningPacket:
  pairs = tuple(
    PlanningPair(StateSnapshot.from_world_state(make_state(index)), tuple(np.zeros(32)))
    for index in range(8)
  )
  return LatentPlanningPacket(
    pairs=pairs,
    current_state=StateSnapshot.from_world_state(make_state(8)),
    source_timestamp=0.0,
    source_tick=0,
    source_frame=0,
    period_seconds=0.02,
    root_frame="world",
    contract_identity=contract_identity,
    vae_identity="vae",
    projection_identity="projection",
    model_identity="model",
    task=None,
    clean_tokens=tuple(tuple(0.0 for _ in range(token_dimension)) for _ in range(41)),
    clean_mask=tuple(tuple(False for _ in range(token_dimension)) for _ in range(41)),
    rng_state=(1, 2, 3, 4),
    rng_device="cpu",
  )


def _save(path: Path, model: StateLatentTransformer, contract: object) -> None:
  from mjlab.tasks.tracking.diffusion.checkpoint import save_checkpoint

  save_checkpoint(
    path,
    model=model,
    ema=None,
    optimizer=None,
    scaler=None,
    config={"seed": 3},
    contract=contract,
    schedule=DiffusionSchedule.build(),
    dataset_identity={},
    global_step=1,
    epoch=0,
    best_validation_loss=None,
    rng_state=None,
  )


def test_contract_dimensions_reads_full_and_reduced_layouts() -> None:
  full = contract_dimensions(DEFAULT_CONTRACT)
  assert full == ContractDimensions(135, 32, 167, 41)
  assert full.label == "135-D state/167-channel token"
  reduced = contract_dimensions(_ReducedContract())
  assert reduced == ContractDimensions(129, 32, 161, 41)
  assert reduced.label == "129-D state/161-channel token"


def test_contract_dimensions_rejects_inconsistent_token_width() -> None:
  with pytest.raises(ValueError, match="token width"):
    contract_dimensions(_ReducedContract(state_dimension=129, token_dimension=167))


def test_denoiser_settings_use_declared_widths_and_keep_the_frozen_default() -> None:
  assert DenoiserSettings.from_contract(DEFAULT_CONTRACT) == DenoiserSettings()
  reduced = DenoiserSettings.from_contract(_ReducedContract())
  assert reduced.token_dimension == 161
  assert reduced.sequence_length == 41
  assert reduced != DenoiserSettings()


def test_reduced_model_accepts_161_tokens_and_rejects_167() -> None:
  model = StateLatentTransformer(DenoiserSettings.from_contract(_ReducedContract()))
  ids = torch.zeros((1, 41, 2), dtype=torch.long)
  output = model(torch.zeros((1, 41, 161)), ids)
  assert output.shape == (1, 41, 161)
  with pytest.raises(ValueError, match="noisy_tokens shape"):
    model(torch.zeros((1, 41, 167)), ids)


def test_reduced_contract_loads_a_matching_checkpoint(tmp_path: Path) -> None:
  contract = _ReducedContract()
  model = StateLatentTransformer(DenoiserSettings.from_contract(contract))
  checkpoint = tmp_path / "reduced.pt"
  _save(checkpoint, model, contract)
  loaded = load_denoiser(
    checkpoint,
    contract=contract,  # type: ignore[arg-type]
    normalizer=_normalizer(contract),
    verify_projection_hashes=False,
    verify_contract_identity=True,
  )
  assert loaded.model.settings.token_dimension == 161
  assert loaded.contract_identity == _REDUCED_IDENTITY


def test_reduced_contract_rejects_a_135d_checkpoint_by_identity(tmp_path: Path) -> None:
  checkpoint = tmp_path / "full.pt"
  _save(checkpoint, StateLatentTransformer(DenoiserSettings()), DEFAULT_CONTRACT)
  with pytest.raises(ReplayError, match="129-D state/161-channel token"):
    load_denoiser(
      checkpoint,
      contract=_ReducedContract(),  # type: ignore[arg-type]
      normalizer=_normalizer(_ReducedContract()),
      verify_projection_hashes=False,
      verify_contract_identity=True,
    )


def test_reduced_contract_rejects_a_135d_checkpoint_by_shape(tmp_path: Path) -> None:
  checkpoint = tmp_path / "full.pt"
  _save(checkpoint, StateLatentTransformer(DenoiserSettings()), DEFAULT_CONTRACT)
  contract = _ReducedContract()
  with pytest.raises(ReplayError, match="129-D state/161-channel token"):
    load_denoiser(
      checkpoint,
      contract=contract,  # type: ignore[arg-type]
      normalizer=_normalizer(contract),
      verify_projection_hashes=False,
      verify_contract_identity=False,
    )


def test_backend_contract_layout_accepts_reduced_and_rejects_inconsistent() -> None:
  layout = _require_contract_layout(_ReducedContract())  # type: ignore[arg-type]
  assert layout.state_dimension == 129
  assert layout.token_dimension == 161
  with pytest.raises(ValueError, match="usable model layout"):
    _require_contract_layout(
      _ReducedContract(state_dimension=129, token_dimension=160)  # type: ignore[arg-type]
    )


def test_reduced_planner_rejects_a_167_channel_packet() -> None:
  contract = _ReducedContract()
  schedule = DiffusionSchedule.build()
  planner = LatentPlanner(
    denoiser=lambda tokens, ids: tokens,
    schedule=schedule,
    grid=build_inference_grid(schedule),
    normalizer=_normalizer(contract),
    contract=contract,  # type: ignore[arg-type]
  )
  reduced_packet = _packet(token_dimension=161, contract_identity=_REDUCED_IDENTITY)
  assert len(reduced_packet.clean_tokens[0]) == 161
  full_width_packet = _packet(token_dimension=167, contract_identity=_REDUCED_IDENTITY)
  with pytest.raises(LatentPlanningError, match="token width differs"):
    planner.plan(full_width_packet)


def test_preflight_reports_the_declared_layout(tmp_path: Path) -> None:
  repo_root = Path(diffusion.__file__).resolve().parents[3]
  contract_path = repo_root / diffusion.DEFAULT_CONTRACT_PATH
  report = diffusion.run_preflight(contract_path, artifact_root=tmp_path)
  assert report.layout == {
    "state_dimension": 135,
    "latent_dimension": 32,
    "token_dimension": 167,
    "window_steps": 41,
  }
  assert report.as_dict()["layout"] == report.layout
