"""Atomic checkpoint and identity regression gates."""

from __future__ import annotations

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
from mjlab.tasks.tracking.diffusion.training_config import TrainingConfig


class TinyEMA:
  """Minimal EMA state holder for checkpoint API tests."""

  def __init__(self) -> None:
    self.value = torch.tensor([2.0])

  def state_dict(self) -> dict[str, torch.Tensor]:
    return {"value": self.value.clone()}

  def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
    self.value.copy_(state["value"])


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


def test_atomic_checkpoint_round_trip_restores_all_mutable_state(tmp_path) -> None:
  model = nn.Linear(3, 2)
  ema = TinyEMA()
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
  config = TrainingConfig(
    epochs=1,
    max_updates=1,
    effective_batch_size=2,
    microbatch_size=1,
    gradient_accumulation_steps=2,
    mixed_precision="fp32",
  )
  contract = DiffusionContract()
  schedule = DiffusionSchedule.build()
  path = tmp_path / "checkpoint.pt"
  rng = {"torch": torch.get_rng_state()}
  save_checkpoint(
    path,
    model=model,
    ema=ema,
    optimizer=optimizer,
    scaler=None,
    config=config,
    contract=contract,
    schedule=schedule,
    dataset_identity=_identity(),
    global_step=7,
    epoch=3,
    best_validation_loss=0.25,
    rng_state=rng,
    best_metric_split="test",
    fp32_checked=True,
  )
  assert path.is_file()
  assert not list(tmp_path.glob("*.tmp"))

  restored_model = nn.Linear(3, 2)
  restored_ema = TinyEMA()
  restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.5)
  state = load_checkpoint(
    path,
    model=restored_model,
    ema=restored_ema,
    optimizer=restored_optimizer,
  )
  assert state.global_step == 7
  assert state.epoch == 3
  assert state.best_validation_loss == pytest.approx(0.25)
  assert state.best_metric_split == "test"
  assert state.fp32_checked is True
  assert state.contract_identity == contract.identity_hash()
  assert state.schedule_identity == schedule.identity_hash()
  assert state.dataset_identity == _identity()
  for left, right in zip(model.parameters(), restored_model.parameters(), strict=True):
    assert torch.equal(left, right)
  assert torch.equal(ema.value, restored_ema.value)
  assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(0.01)


def test_legacy_checkpoint_without_selection_split_is_tolerated(tmp_path) -> None:
  model = nn.Linear(3, 2)
  config = TrainingConfig(
    epochs=1,
    max_updates=1,
    effective_batch_size=1,
    microbatch_size=1,
    gradient_accumulation_steps=1,
    mixed_precision="fp32",
  )
  path = tmp_path / "legacy.pt"
  save_checkpoint(
    path,
    model=model,
    ema=None,
    optimizer=None,
    scaler=None,
    config=config,
    contract=DiffusionContract(),
    schedule=DiffusionSchedule.build(),
    dataset_identity=_identity(),
    global_step=1,
    epoch=1,
    best_validation_loss=None,
    rng_state={"torch": torch.get_rng_state()},
  )
  payload = torch.load(path, map_location="cpu", weights_only=False)
  payload.pop("selection_split")
  torch.save(payload, path)
  state = load_checkpoint(path, model=nn.Linear(3, 2))
  assert state.selection_split is None

  model = nn.Linear(3, 2)
  ema = TinyEMA()
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
  config = TrainingConfig(
    epochs=1,
    max_updates=1,
    effective_batch_size=2,
    microbatch_size=1,
    gradient_accumulation_steps=2,
    mixed_precision="fp32",
  )
  path = tmp_path / "checkpoint.pt"
  save_checkpoint(
    path,
    model=model,
    ema=ema,
    optimizer=optimizer,
    scaler=None,
    config=config,
    contract=DiffusionContract(),
    schedule=DiffusionSchedule.build(),
    dataset_identity=_identity(),
    global_step=1,
    epoch=1,
    best_validation_loss=None,
    rng_state={"torch": torch.get_rng_state()},
  )
  payload = torch.load(path, map_location="cpu", weights_only=False)
  for field, match in (("ema", "EMA"), ("optimizer", "optimizer")):
    tampered = dict(payload)
    tampered[field] = None
    torch.save(tampered, path)
    with pytest.raises(CheckpointError, match=match):
      load_checkpoint(
        path,
        model=nn.Linear(3, 2),
        ema=TinyEMA(),
        optimizer=torch.optim.AdamW(nn.Linear(3, 2).parameters(), lr=0.01),
      )


def test_corrupt_checkpoint_is_rejected(tmp_path) -> None:
  path = tmp_path / "broken.pt"
  path.write_bytes(b"not a torch checkpoint")
  with pytest.raises(CheckpointError, match="read|malformed|unsupported"):
    load_checkpoint(path, model=nn.Linear(3, 2))


def test_checkpoint_config_hash_tampering_is_rejected(tmp_path) -> None:
  model = nn.Linear(3, 2)
  config = TrainingConfig(
    epochs=1,
    max_updates=1,
    effective_batch_size=2,
    microbatch_size=1,
    gradient_accumulation_steps=2,
    mixed_precision="fp32",
  )
  path = tmp_path / "checkpoint.pt"
  save_checkpoint(
    path,
    model=model,
    ema=None,
    optimizer=None,
    scaler=None,
    config=config,
    contract=DiffusionContract(),
    schedule=DiffusionSchedule.build(),
    dataset_identity=_identity(),
    global_step=1,
    epoch=1,
    best_validation_loss=None,
    rng_state={"torch": torch.get_rng_state()},
  )
  payload = torch.load(path, map_location="cpu", weights_only=False)
  payload["config"]["seed"] = 999
  torch.save(payload, path)
  with pytest.raises(CheckpointError, match="config hash"):
    load_checkpoint(path, model=nn.Linear(3, 2))
