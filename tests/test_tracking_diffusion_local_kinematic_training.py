"""Local-kinematic 129-D state ablation: dimension-driven training plumbing.

The ablation removes the two root-translation slices (``root_position`` and
``root_linear_velocity``) from the 135-D hybrid state, so the packed state is
129-D and the state+latent token is 161-D.  These tests pin the positive 129-D
load and the fail-closed behaviour of the training/data path against 135-D
artifacts, checkpoints and normalizers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from mjlab.tasks.tracking.diffusion import DiffusionContract, DiffusionSchedule
from mjlab.tasks.tracking.diffusion.checkpoint import (
  CheckpointError,
  load_checkpoint,
  save_checkpoint,
)
from mjlab.tasks.tracking.diffusion.dataset import (
  DatasetError,
  _pack_state_for_contract,
)
from mjlab.tasks.tracking.diffusion.model import DenoiserSettings
from mjlab.tasks.tracking.diffusion.projection import (
  FeatureStats,
  ProjectionError,
  StateLatentNormalizer,
)
from mjlab.tasks.tracking.diffusion.state import hybrid_to_local_kinematic
from mjlab.tasks.tracking.diffusion.trainer import (
  DiffusionTrainer,
  TrainerError,
  _as_tokens,
)
from mjlab.tasks.tracking.diffusion.training_config import TrainingConfig
from mjlab.tasks.tracking.diffusion.window_dataset import (
  WindowRecord,
  token_cache_identity_key,
)

LOCAL_KINEMATIC = Path(
  "docs/plans/beyondmimic_diffusion_d1_no_mixtures_local_kinematic_contract.yaml"
)
LOCAL_KINEMATIC_VGK8 = Path(
  "docs/plans/beyondmimic_diffusion_d1_no_mixtures_local_kinematic_vgk8_contract.yaml"
)
FULL = Path("docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml")
FULL_VGK8 = Path("docs/plans/beyondmimic_diffusion_d1_no_mixtures_vgk8_contract.yaml")


def _windows(count: int, token_dimension: int) -> Dataset:
  class Windows(Dataset):
    def __init__(self) -> None:
      generator = torch.Generator().manual_seed(17)
      self.tokens = torch.randn((count, 41, token_dimension), generator=generator)

    def __len__(self) -> int:
      return len(self.tokens)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, int]:
      return self.tokens[index], torch.tensor(1.0), index

  return Windows()


class _PointwiseDenoiser(nn.Module):
  """Width-matched stand-in for the frozen transformer."""

  def __init__(self, token_dimension: int) -> None:
    super().__init__()
    self.linear = nn.Linear(token_dimension, token_dimension)

  def forward(self, tokens: Tensor, step_ids: Tensor) -> Tensor:
    del step_ids
    return self.linear(tokens)


def _config() -> TrainingConfig:
  return TrainingConfig(
    epochs=1,
    max_updates=1,
    effective_batch_size=2,
    microbatch_size=2,
    gradient_accumulation_steps=1,
    mixed_precision="fp32",
    warmup_updates=0,
    eval_split="validation",
  )


def _trainer(
  output: Path,
  *,
  contract: DiffusionContract,
  model: nn.Module,
  train: Dataset,
) -> DiffusionTrainer:
  return DiffusionTrainer(
    config=_config(),
    contract=contract,
    schedule=DiffusionSchedule.build(),
    model=model,
    train_dataset=train,
    eval_datasets={"validation": _windows(2, contract.token_dimension)},
    output_dir=output,
    device="cpu",
  )


def test_dataset_packs_simulator_hybrid_state_for_the_129_contract() -> None:
  local = DiffusionContract.from_yaml(LOCAL_KINEMATIC)
  hybrid = np.arange(2 * 41 * 135, dtype=np.float64).reshape(2, 41, 135)
  packed = _pack_state_for_contract(hybrid, local)
  assert packed.shape == (2, 41, 129)
  np.testing.assert_array_equal(packed, hybrid_to_local_kinematic(hybrid))


def test_dataset_keeps_the_frozen_135_contract_layout() -> None:
  full = DiffusionContract.from_yaml(FULL)
  hybrid = np.arange(41 * 135, dtype=np.float64).reshape(41, 135)
  packed = _pack_state_for_contract(hybrid, full)
  np.testing.assert_array_equal(packed, hybrid)


def test_vgk8_local_kinematic_contract_pins_its_own_vae_and_identity() -> None:
  full_vgk8 = DiffusionContract.from_yaml(FULL_VGK8)
  local_vgk8 = DiffusionContract.from_yaml(LOCAL_KINEMATIC_VGK8)
  local_three = DiffusionContract.from_yaml(LOCAL_KINEMATIC)
  assert (local_vgk8.state_dimension, local_vgk8.token_dimension) == (129, 161)
  assert local_vgk8.state_fields == (
    "root_rot6d",
    "root_angular_velocity",
    "body_positions",
    "body_linear_velocities",
  )
  # The ablation keeps the eight-teacher VAE bytes but is a distinct identity
  # from the 135-D VGK8 contract and from the three-teacher ablation.
  assert local_vgk8.vae_sha256 == full_vgk8.vae_sha256
  assert local_vgk8.vae_checkpoint == full_vgk8.vae_checkpoint
  assert local_vgk8.identity_hash() not in {
    full_vgk8.identity_hash(),
    local_three.identity_hash(),
  }


def test_vgk8_local_kinematic_train_config_matches_its_contract() -> None:
  config = TrainingConfig.from_yaml(
    "configs/diffusion/x2_50hz_train_local_kinematic_vgk8.yaml"
  )
  assert config.epochs == 1000
  assert config.effective_batch_size == 512
  assert config.microbatch_size == 128
  assert config.mixed_precision == "bf16"
  assert config.seed == 0

  local = DiffusionContract.from_yaml(LOCAL_KINEMATIC)
  settings = DenoiserSettings.from_contract(local)
  assert settings.token_dimension == 161
  assert settings.sequence_length == local.window_steps
  # The frozen 135-D model default is unchanged.
  assert DenoiserSettings.from_contract(DiffusionContract()).token_dimension == 167


def test_129_contract_trains_one_update_over_161_channel_windows(
  tmp_path: Path,
) -> None:
  contract = DiffusionContract.from_yaml(LOCAL_KINEMATIC)
  trainer = _trainer(
    tmp_path,
    contract=contract,
    model=_PointwiseDenoiser(161),
    train=_windows(4, 161),
  )
  result = trainer.train()
  assert result.global_step == 1
  assert (tmp_path / "checkpoint-last.pt").exists()
  payload = torch.load(
    result.checkpoint_paths["last"], map_location="cpu", weights_only=False
  )
  assert payload["contract_identity"] == contract.identity_hash()
  assert payload["model"]["linear.weight"].shape[0] == 161


def test_as_tokens_rejects_the_135_channel_width_under_the_129_contract() -> None:
  contract = DiffusionContract.from_yaml(LOCAL_KINEMATIC)
  with pytest.raises(TrainerError, match="161"):
    _as_tokens(
      torch.zeros((2, 41, 167)),
      window_steps=contract.window_steps,
      token_dimension=contract.token_dimension,
    )


def test_129_contract_rejects_a_135_projection_normalizer(tmp_path: Path) -> None:
  full = DiffusionContract.from_yaml(FULL)
  local = DiffusionContract.from_yaml(LOCAL_KINEMATIC)
  generator = np.random.default_rng(3)
  normalizer = StateLatentNormalizer.create(
    FeatureStats(4, generator.normal(size=135), np.ones(135)),
    FeatureStats(4, generator.normal(size=32), np.ones(32)),
    contract=full,
  )
  path = tmp_path / "projection.npz"
  normalizer.save(path)
  with pytest.raises(ProjectionError, match="schema mismatch"):
    StateLatentNormalizer.load(path, contract=local)


def test_129_contract_accepts_and_round_trips_its_own_normalizer(
  tmp_path: Path,
) -> None:
  contract = DiffusionContract.from_yaml(LOCAL_KINEMATIC)
  generator = np.random.default_rng(5)
  normalizer = StateLatentNormalizer.create(
    FeatureStats(4, generator.normal(size=129), np.ones(129)),
    FeatureStats(4, generator.normal(size=32), np.ones(32)),
    contract=contract,
  )
  path = tmp_path / "projection.npz"
  normalizer.save(path)
  loaded = StateLatentNormalizer.load(path, contract=contract)
  state = generator.normal(size=(3, 129))
  latent = generator.normal(size=(3, 32))
  assert loaded.make_tokens(state, latent).shape == (3, 161)


def test_129_contract_rejects_a_135_checkpoint(tmp_path: Path) -> None:
  full = DiffusionContract.from_yaml(FULL)
  local = DiffusionContract.from_yaml(LOCAL_KINEMATIC)
  path = tmp_path / "checkpoint-last.pt"
  save_checkpoint(
    path,
    model=_PointwiseDenoiser(167),
    ema=None,
    optimizer=None,
    scaler=None,
    config=_config(),
    contract=full,
    schedule=DiffusionSchedule.build(),
    dataset_identity={},
    global_step=0,
    epoch=0,
    best_validation_loss=None,
    rng_state={"torch": None},
  )
  with pytest.raises(CheckpointError, match="contract identity"):
    load_checkpoint(
      path,
      model=_PointwiseDenoiser(161),
      expected_contract_identity=local.identity_hash(),
    )


def test_129_and_135_token_caches_have_distinct_identity_keys() -> None:
  full = DiffusionContract.from_yaml(FULL)
  local = DiffusionContract.from_yaml(LOCAL_KINEMATIC)
  assert token_cache_identity_key(
    source_hash="a" * 64,
    split="train",
    projection_hash="b" * 64,
    contract_hash=full.identity_hash(),
  ) != token_cache_identity_key(
    source_hash="a" * 64,
    split="train",
    projection_hash="b" * 64,
    contract_hash=local.identity_hash(),
  )


def test_window_record_enforces_the_declared_token_dimension() -> None:
  record = WindowRecord(
    tokens=np.zeros((41, 161), dtype=np.float32),
    split="train",
    group_key="group",
    motion_id="tennis_000",
    phase="clean",
    ou_noise_norm=0.0,
    pair_id="pair",
    start_tick=0,
    token_dimension=161,
  )
  assert record.tokens.shape == (41, 161)
  with pytest.raises(DatasetError, match="finite"):
    WindowRecord(
      tokens=np.zeros((41, 167), dtype=np.float32),
      split="train",
      group_key="group",
      motion_id="tennis_000",
      phase="clean",
      ou_noise_norm=0.0,
      pair_id="pair",
      start_tick=0,
      token_dimension=161,
    )
