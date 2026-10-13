"""Flow checkpoint format, identity and cross-family rejection gates."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from mjlab.tasks.tracking.diffusion import (
  DiffusionContract,
  DiffusionSchedule,
  load_checkpoint,
  save_checkpoint,
)
from mjlab.tasks.tracking.diffusion.checkpoint import CheckpointError
from mjlab.tasks.tracking.diffusion.trainer import ExponentialMovingAverage
from mjlab.tasks.tracking.diffusion.training_config import TrainingConfig
from mjlab.tasks.tracking.flow_matching import (
  FLOW_CHECKPOINT_FORMAT,
  FlowCheckpointError,
  FlowTrainingConfig,
  LinearNoiseProcess,
  identity_sha256,
  load_flow_checkpoint,
  model_identity,
  save_flow_checkpoint,
)


def _identity() -> dict[str, object]:
  return {
    "directory": "synthetic",
    "assignments_hash": "a" * 64,
    "split_coverage": {"train": 2, "validation": 0, "test": 0},
    "projection_hashes": {
      "matrix": "b" * 64,
      "statistics": "c" * 64,
    },
  }


def _save_flow(
  path: Path,
  *,
  model: nn.Module | None = None,
  config: FlowTrainingConfig | None = None,
  process: LinearNoiseProcess | None = None,
  dataset_identity: dict[str, object] | None = None,
) -> nn.Module:
  model = model if model is not None else nn.Linear(3, 2)
  ema = ExponentialMovingAverage(model)
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
  save_flow_checkpoint(
    path,
    model=model,
    ema=ema,
    optimizer=optimizer,
    scaler=None,
    config=config or FlowTrainingConfig(),
    process=process or LinearNoiseProcess(),
    data_contract=DiffusionContract(),
    dataset_identity=dataset_identity or _identity(),
    model_description=model_identity(model),
    global_step=7,
    epoch=3,
    best_validation_loss=0.25,
    rng_state={"torch": torch.get_rng_state()},
    best_metric_split="validation",
    fp32_checked=True,
  )
  return model


def test_flow_checkpoint_round_trip_restores_all_mutable_state(tmp_path: Path) -> None:
  path = tmp_path / "checkpoint.pt"
  model = _save_flow(path)
  assert path.is_file()
  assert not list(tmp_path.glob("*.tmp"))

  restored_model = nn.Linear(3, 2)
  restored_ema = ExponentialMovingAverage(restored_model)
  restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.5)
  state = load_flow_checkpoint(
    path,
    model=restored_model,
    ema=restored_ema,
    optimizer=restored_optimizer,
  )
  assert state.global_step == 7
  assert state.epoch == 3
  assert state.best_validation_loss == pytest.approx(0.25)
  assert state.best_metric_split == "validation"
  assert state.selection_split == "validation"
  assert state.fp32_checked is True
  assert state.family == "x0_linear_flow"
  assert state.process["path"] == "linear_gaussian"
  assert state.process["loss"] == "x0_mse"
  assert state.process["time_encoding"] == "integer_noise_fraction"
  assert state.process_identity == LinearNoiseProcess().identity_hash()
  assert state.data_identity == DiffusionContract().identity_hash()
  assert state.dataset_identity == _identity()
  assert state.statistics_identity == {"matrix": "b" * 64, "statistics": "c" * 64}
  assert state.model_identity_hash == identity_sha256(model_identity(model))
  assert state.model_identity["type"] == "Linear"
  for left, right in zip(model.parameters(), restored_model.parameters(), strict=True):
    assert torch.equal(left, right)
  assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(0.01)


def test_flow_checkpoint_records_family_and_no_diffusion_schedule(
  tmp_path: Path,
) -> None:
  path = tmp_path / "checkpoint.pt"
  _save_flow(path)
  payload = torch.load(path, map_location="cpu", weights_only=False)
  assert payload["format"] == FLOW_CHECKPOINT_FORMAT
  assert payload["family"] == "x0_linear_flow"
  assert "schedule_identity" not in payload
  assert "schedule" not in payload


def test_tampered_process_is_rejected(tmp_path: Path) -> None:
  path = tmp_path / "checkpoint.pt"
  _save_flow(path)
  payload = torch.load(path, map_location="cpu", weights_only=False)
  payload["process"]["path"] = "cosine"
  torch.save(payload, path)
  with pytest.raises(FlowCheckpointError, match="inconsistent"):
    load_flow_checkpoint(path, model=nn.Linear(3, 2))


def test_tampered_config_hash_is_rejected(tmp_path: Path) -> None:
  path = tmp_path / "checkpoint.pt"
  _save_flow(path)
  payload = torch.load(path, map_location="cpu", weights_only=False)
  payload["config"]["max_level"] = 500
  torch.save(payload, path)
  with pytest.raises(FlowCheckpointError, match="config hash"):
    load_flow_checkpoint(path, model=nn.Linear(3, 2))


def test_missing_identity_field_is_rejected(tmp_path: Path) -> None:
  path = tmp_path / "checkpoint.pt"
  _save_flow(path)
  payload = torch.load(path, map_location="cpu", weights_only=False)
  payload.pop("process_identity")
  torch.save(payload, path)
  with pytest.raises(FlowCheckpointError, match="missing required identity"):
    load_flow_checkpoint(path, model=nn.Linear(3, 2))


def test_corrupt_flow_checkpoint_is_rejected(tmp_path: Path) -> None:
  path = tmp_path / "broken.pt"
  path.write_bytes(b"not a torch checkpoint")
  with pytest.raises(FlowCheckpointError, match="read|malformed|unsupported"):
    load_flow_checkpoint(path, model=nn.Linear(3, 2))


def test_flow_loader_rejects_a_diffusion_checkpoint(tmp_path: Path) -> None:
  model = nn.Linear(3, 2)
  path = tmp_path / "diffusion.pt"
  save_checkpoint(
    path,
    model=model,
    ema=None,
    optimizer=None,
    scaler=None,
    config=TrainingConfig(),
    contract=DiffusionContract(),
    schedule=DiffusionSchedule.build(),
    dataset_identity=_identity(),
    global_step=1,
    epoch=1,
    best_validation_loss=None,
    rng_state={"torch": torch.get_rng_state()},
  )
  with pytest.raises(FlowCheckpointError, match="unsupported or malformed flow"):
    load_flow_checkpoint(path, model=nn.Linear(3, 2))


def test_diffusion_loader_rejects_a_flow_checkpoint(tmp_path: Path) -> None:
  path = tmp_path / "flow.pt"
  _save_flow(path)
  with pytest.raises(CheckpointError, match="unsupported or malformed diffusion"):
    load_checkpoint(path, model=nn.Linear(3, 2))


def test_expected_identities_are_enforced_on_load(tmp_path: Path) -> None:
  path = tmp_path / "checkpoint.pt"
  _save_flow(path)
  with pytest.raises(FlowCheckpointError, match="process identity"):
    load_flow_checkpoint(
      path,
      model=nn.Linear(3, 2),
      expected_process_identity=LinearNoiseProcess(max_level=100).identity_hash(),
    )
  changed_contract = DiffusionContract(vae_sha256="d" * 64)
  with pytest.raises(FlowCheckpointError, match="data identity"):
    load_flow_checkpoint(
      path,
      model=nn.Linear(3, 2),
      expected_data_identity=changed_contract.identity_hash(),
    )
