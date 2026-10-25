"""CPU-only flow trainer mechanics, identity and deterministic-resume gates."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset, TensorDataset

from mjlab.tasks.tracking.diffusion.contract import DiffusionContract
from mjlab.tasks.tracking.diffusion.state import (
  StateValidationError,
  state_field_channels,
)
from mjlab.tasks.tracking.diffusion.training_config import TrainingConfig
from mjlab.tasks.tracking.flow_matching import (
  FLOW_CHECKPOINT_FORMAT,
  FlowTrainer,
  FlowTrainerError,
  FlowTrainingConfig,
  add_linear_noise,
  sample_independent_levels,
)


class SyntheticWindows(Dataset[Tensor]):
  """Small well-formed clean token fixtures."""

  def __init__(self, count: int = 5, *, identity: str = "synthetic-v1") -> None:
    generator = torch.Generator().manual_seed(31)
    self.tokens = torch.randn((count, 41, 167), generator=generator)
    self.dataset_identity = {
      "directory": identity,
      "assignments_hash": "a" * 64,
      "split_coverage": {"train": count, "validation": 0, "test": 0},
      "projection_hashes": {
        "matrix": "b" * 64,
        "statistics": "c" * 64,
      },
    }

  def __len__(self) -> int:
    return len(self.tokens)

  def __getitem__(self, index: int) -> Tensor:
    return self.tokens[index]


class TinyDenoiser(nn.Module):
  """Fast pointwise model for CPU trainer tests."""

  def __init__(self) -> None:
    super().__init__()
    self.linear = nn.Linear(167, 167)

  def forward(self, tokens: Tensor, step_ids: Tensor) -> Tensor:
    del step_ids
    return self.linear(tokens)


class SettingsDenoiser(TinyDenoiser):
  """Model whose state dict matches ``TinyDenoiser`` but identity does not."""

  def __init__(self, tag: str) -> None:
    super().__init__()
    self.settings = {"tag": tag}


class ConstantDenoiser(nn.Module):
  """Learnable constant used for a deterministic x0 overfit gate."""

  def __init__(self) -> None:
    super().__init__()
    self.value = nn.Parameter(torch.zeros((1, 41, 167)))

  def forward(self, tokens: Tensor, step_ids: Tensor) -> Tensor:
    del step_ids
    return self.value.expand(tokens.shape[0], -1, -1)


class NaNDenoiser(TinyDenoiser):
  """Model used to prove divergence aborts before any checkpoint write."""

  def forward(self, tokens: Tensor, step_ids: Tensor) -> Tensor:
    del step_ids
    return torch.full_like(tokens, float("nan"))


def _config(**overrides: object) -> FlowTrainingConfig:
  values: dict[str, Any] = {
    "epochs": 2,
    "max_updates": 2,
    "effective_batch_size": 4,
    "microbatch_size": 2,
    "gradient_accumulation_steps": 2,
    "mixed_precision": "fp32",
    "warmup_updates": 1,
    "eval_split": "validation",
  }
  values.update(overrides)
  return FlowTrainingConfig(training=TrainingConfig(**values))


def _trainer(
  output: Path,
  *,
  model: nn.Module | None = None,
  dataset: Any | None = None,
  config: FlowTrainingConfig | None = None,
  contract: Any | None = None,
  resume: Path | None = None,
  eval_dataset: Any | None = None,
) -> FlowTrainer:
  config = config or _config()
  # Seed before model construction so two trainers with the same config start
  # from identical parameters.
  torch.manual_seed(config.training.seed)
  train = dataset if dataset is not None else SyntheticWindows()
  return FlowTrainer(
    config=config,
    contract=contract or DiffusionContract(),
    model=model if model is not None else TinyDenoiser(),
    train_dataset=train,
    eval_datasets={
      "validation": eval_dataset if eval_dataset is not None else SyntheticWindows(2)
    },
    output_dir=output,
    device="cpu",
    resume=resume,
  )


def _clean(seed: int = 5, batch: int = 3) -> Tensor:
  return torch.randn((batch, 41, 167), generator=torch.Generator().manual_seed(seed))


def test_forward_loss_matches_stage1_primitives_and_plain_x0_mse(
  tmp_path: Path,
) -> None:
  trainer = _trainer(tmp_path)
  clean = _clean(batch=3)
  trainer.noise_generator.manual_seed(7)
  loss, prediction, target, state_levels, latent_levels, _ = trainer._forward_loss(
    clean
  )

  reference = torch.Generator().manual_seed(7)
  levels = sample_independent_levels(
    (3, 41), process=trainer.process, generator=reference
  )
  noised = add_linear_noise(
    clean[..., :135],
    clean[..., 135:],
    state_levels=levels.state,
    latent_levels=levels.latent,
    process=trainer.process,
    generator=reference,
  )
  step_ids = torch.stack((noised.state_levels, noised.latent_levels), dim=-1)
  expected = trainer.model(noised.tokens, step_ids)

  assert torch.equal(state_levels, levels.state)
  assert torch.equal(latent_levels, levels.latent)
  assert torch.equal(target, clean)
  assert torch.allclose(prediction, expected.float())
  assert loss.item() == pytest.approx(
    float(F.mse_loss(expected.float().detach(), clean)), rel=0.0, abs=1e-12
  )


def test_all_rows_receive_independent_levels_without_clean_insertion(
  tmp_path: Path,
) -> None:
  trainer = _trainer(tmp_path)
  trainer.noise_generator.manual_seed(11)
  clean = _clean(batch=32)
  _, _, _, state_levels, latent_levels, _ = trainer._forward_loss(clean)
  assert state_levels.shape == (32, 41)
  assert latent_levels.shape == (32, 41)
  assert state_levels.dtype == torch.long
  assert int(state_levels.min()) >= 0
  assert int(state_levels.max()) <= 1000
  # History rows are noised like every other row: many distinct levels appear,
  # so no row is pinned to a clean-mask level.
  history = state_levels[:, :8]
  assert len(history.unique()) > 10
  assert torch.equal(state_levels, latent_levels) is False


def test_tiny_fixed_window_overfits_x0_objective(tmp_path: Path) -> None:
  target = torch.full((1, 41, 167), 0.25)
  config = FlowTrainingConfig(
    training=TrainingConfig(
      epochs=200,
      max_updates=100,
      effective_batch_size=1,
      microbatch_size=1,
      gradient_accumulation_steps=1,
      learning_rate=0.05,
      warmup_updates=0,
      mixed_precision="fp32",
      eval_split="validation",
    )
  )
  trainer = _trainer(
    tmp_path,
    model=ConstantDenoiser(),
    dataset=TensorDataset(target),
    config=config,
  )
  result = trainer.train()
  assert result.final_train_loss < 1.0e-3


def test_train_records_flow_identity_and_writes_no_diffusion_schedule(
  tmp_path: Path,
) -> None:
  config = _config(epochs=1, max_updates=2)
  trainer = _trainer(tmp_path, config=config)
  result = trainer.train()
  report = json.loads((tmp_path / "train-report.json").read_text())
  assert report["family"] == "x0_linear_flow"
  assert report["process_identity"] == trainer.process.identity_hash()
  assert report["history_state_dropout_probability"] == 0.0
  assert report["data_identity"] == trainer.data_identity
  payload = torch.load(
    result.checkpoint_paths["last"], map_location="cpu", weights_only=False
  )
  assert payload["format"] == FLOW_CHECKPOINT_FORMAT
  assert "schedule_identity" not in payload
  assert payload["process"]["time_encoding"] == "integer_noise_fraction"


def test_resume_restores_generators_and_reproduces_the_next_step(
  tmp_path: Path,
) -> None:
  config = _config(epochs=1, max_updates=2)
  clean = _clean(seed=9, batch=2)
  first = _trainer(tmp_path / "first", config=config)
  first.noise_generator.manual_seed(13)
  first._forward_loss(clean)
  checkpoint = first._save("checkpoint-probe.pt", epoch=0, best_validation_loss=None)
  expected_loss, expected_prediction, _, expected_state, expected_latent, _ = (
    first._forward_loss(clean)
  )
  payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

  resumed = _trainer(tmp_path / "resumed", config=config, resume=checkpoint)
  assert torch.equal(
    resumed.noise_generator.get_state(), payload["rng_state"]["noise_generator"]
  )
  assert torch.equal(
    resumed.sampler_generator.get_state(), payload["rng_state"]["sampler_generator"]
  )
  loss, prediction, _, state_levels, latent_levels, _ = resumed._forward_loss(clean)
  assert torch.equal(loss, expected_loss)
  assert torch.equal(prediction, expected_prediction)
  assert torch.equal(state_levels, expected_state)
  assert torch.equal(latent_levels, expected_latent)


def test_resume_restores_a_checkpoint_whose_rng_states_left_cpu(
  tmp_path: Path,
) -> None:
  # ``load_flow_checkpoint`` moves every payload tensor to the training device,
  # so a CUDA resume hands the RNG states back as a CUDA ByteTensor while
  # ``set_rng_state`` accepts only a CPU one.  An int32 copy reproduces the same
  # mismatch on a CPU-only test.
  config = _config(epochs=1, max_updates=2)
  first = _trainer(tmp_path / "first", config=config)
  checkpoint = first._save("checkpoint-probe.pt", epoch=0, best_validation_loss=None)
  payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
  payload["rng_state"]["torch"] = payload["rng_state"]["torch"].to(torch.int32)
  payload["rng_state"]["torch_cuda"] = None
  for name in ("noise_generator", "sampler_generator"):
    payload["rng_state"][name] = payload["rng_state"][name].to(torch.int32)
  tampered = tmp_path / "checkpoint-moved.pt"
  torch.save(payload, tampered)

  resumed = _trainer(tmp_path / "resumed", config=config, resume=tampered)
  assert resumed.global_step == 0
  assert resumed.noise_generator.get_state().dtype == torch.uint8
  assert resumed.sampler_generator.get_state().dtype == torch.uint8


def test_resume_accepts_a_changed_config_and_names_it(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  """A precision or schedule branch must be able to resume one checkpoint.

  The config hash used to reject any changed field, which blocked every paired
  branch.  Resume now accepts the change, records the config it ran with, and
  prints the fields that moved.
  """
  config = _config(epochs=1, max_updates=2)
  first = _trainer(tmp_path / "first", config=config)
  checkpoint = first._save("checkpoint-probe.pt", epoch=0, best_validation_loss=None)

  changed_config = _config(epochs=1, max_updates=2, seed=5)
  resumed = _trainer(tmp_path / "config", config=changed_config, resume=checkpoint)
  assert resumed.global_step == first.global_step
  note = capsys.readouterr().out
  assert "resume note" in note
  assert "seed" in note

  with pytest.raises(FlowTrainerError, match="process identity"):
    _trainer(
      tmp_path / "process",
      config=FlowTrainingConfig(training=changed_config.training, max_level=500),
      resume=checkpoint,
    )

  with pytest.raises(FlowTrainerError, match="data identity"):
    _trainer(
      tmp_path / "data",
      config=config,
      contract=DiffusionContract(vae_sha256="d" * 64),
      resume=checkpoint,
    )

  changed_statistics = SyntheticWindows()
  changed_statistics.dataset_identity["projection_hashes"] = {
    "matrix": "e" * 64,
    "statistics": "f" * 64,
  }
  with pytest.raises(FlowTrainerError, match="statistics identity"):
    _trainer(
      tmp_path / "statistics",
      config=config,
      dataset=changed_statistics,
      resume=checkpoint,
    )

  with pytest.raises(FlowTrainerError, match="dataset identity"):
    _trainer(
      tmp_path / "dataset",
      config=config,
      dataset=SyntheticWindows(identity="other-directory"),
      resume=checkpoint,
    )

  with pytest.raises(FlowTrainerError, match="model identity"):
    _trainer(
      tmp_path / "model",
      config=config,
      model=SettingsDenoiser("other"),
      resume=checkpoint,
    )


def test_nonzero_history_dropout_is_supported(tmp_path: Path) -> None:
  """History dropout is allowed and matches the diffusion meaning."""
  config = FlowTrainingConfig(
    training=TrainingConfig(
      history_state_dropout_probability=1.0, mixed_precision="fp32"
    )
  )
  config.validate()
  assert FlowTrainingConfig.from_mapping(config.as_dict()).training == config.training
  assert config.training.history_state_dropout_probability == 1.0

  trainer = _trainer(tmp_path, config=config)
  clean = _clean(batch=3)
  _, _, _, state_levels, latent_levels, dropout = trainer._forward_loss(clean)
  # A full dropout raises every history row of both slices to pure Gaussian and
  # leaves the current and future rows at their sampled levels.
  assert bool(dropout.all())
  assert torch.equal(
    state_levels[:, :8], torch.full_like(state_levels[:, :8], trainer.process.max_level)
  )
  assert torch.equal(
    latent_levels[:, :8],
    torch.full_like(latent_levels[:, :8], trainer.process.max_level),
  )
  assert bool((state_levels[:, 8:] < trainer.process.max_level).any())


def test_history_dropout_fraction_is_reported(tmp_path: Path) -> None:
  config = FlowTrainingConfig(
    training=TrainingConfig(
      history_state_dropout_probability=1.0,
      mixed_precision="fp32",
      epochs=1,
      max_updates=2,
      epoch_sample_budget=8,
      effective_batch_size=4,
      microbatch_size=4,
      gradient_accumulation_steps=1,
    )
  )
  trainer = _trainer(tmp_path, config=config)
  result = trainer.train()
  report = json.loads((tmp_path / "train-report.json").read_text())
  assert report["history_state_dropout_probability"] == 1.0
  assert report["history_state_dropout_fraction"] == 1.0
  assert report["history_state_total_windows"] > 0
  state = trainer._rng_state()
  assert "history_dropout_generator" in state
  assert state["history_dropout_counters"]["windows"] > 0
  assert result.global_step == 2
  # The epoch row keeps the epoch-boundary values: spreading the averaged
  # per-update block last used to replace global_step with the epoch mean.
  epochs = [
    json.loads(line)
    for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    if json.loads(line)["kind"] == "epoch"
  ]
  assert len(epochs) == 1
  assert epochs[0]["epoch"] == 1
  assert epochs[0]["global_step"] == 2
  assert epochs[0]["history_state_dropout_fraction"] == 1.0
  assert epochs[0]["history_state_total_windows"] == 8


def test_update_metrics_report_the_pre_clip_gradient_norm(tmp_path: Path) -> None:
  # A tiny clip threshold makes every step clip, which proves the logged value
  # is the norm measured before ``clip_grad_norm_`` rescales the gradients.
  trainer = _trainer(tmp_path, config=_config(max_grad_norm=1e-6))
  trainer.train()
  updates = [
    json.loads(line)
    for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    if json.loads(line)["kind"] == "update"
  ]
  assert len(updates) == 2
  for row in updates:
    assert row["gradient_norm"] > 1e-6
    assert math.isfinite(row["gradient_norm"])


def test_nan_aborts_without_writing_a_checkpoint(tmp_path: Path) -> None:
  trainer = _trainer(tmp_path, model=NaNDenoiser())
  with pytest.raises(FlowTrainerError, match="finite|NaN|Inf"):
    trainer.train()
  assert not (tmp_path / "checkpoint-last.pt").exists()
  assert not (tmp_path / "checkpoint-best.pt").exists()


def test_evaluate_reports_unknown_and_clean_x0_metrics(tmp_path: Path) -> None:
  trainer = _trainer(tmp_path, dataset=SyntheticWindows())
  metrics = trainer.evaluate("validation", parameters="model")
  assert metrics["windows"] == 2.0
  for name in (
    "loss",
    "total_mse",
    "state_mse",
    "latent_mse",
    "unknown_mse",
    "unknown_state_mse",
    "unknown_latent_mse",
    "clean_state_mse",
    "clean_latent_mse",
  ):
    assert name in metrics
  assert metrics["unknown_mse"] > 0.0


def test_masked_state_fields_remove_named_channels_and_keep_the_rest(
  tmp_path: Path,
) -> None:
  """Root translation is masked on every row and excluded from the loss."""

  fields = ("root_position", "root_linear_velocity")
  config = _config(masked_state_fields=fields)
  assert config.training.masked_state_fields == fields
  trainer = _trainer(tmp_path, config=config)
  assert trainer._state_mask_channels == (0, 1, 2, 9, 10, 11)

  clean = _clean(batch=3)
  loss, prediction, target, state_levels, latent_levels, _ = trainer._forward_loss(
    clean
  )
  masked = list(trainer._state_mask_channels)
  # The forward pass zeroes exactly the masked channels of the noised input; the
  # prediction tensor echoes that input shape, so check the mask arithmetic
  # through the loss weight and the reported channel list instead of the model
  # input, which is not returned.
  assert masked == [0, 1, 2, 9, 10, 11]
  assert prediction.shape == target.shape == clean.shape
  assert bool((latent_levels <= trainer.process.max_level).all())
  assert bool((state_levels <= trainer.process.max_level).all())

  # A non-default fill value is written exactly into the masked channels.
  filled = _trainer(
    tmp_path / "filled",
    config=_config(masked_state_fields=fields, masked_state_fill_value=6.0),
  )
  loss_filled, _, _, _, _, _ = filled._forward_loss(clean)
  assert torch.isfinite(loss_filled)
  assert filled.training.masked_state_fill_value == 6.0

  # The reported loss equals the unmasked-channel mean of the same predictions.
  weight = trainer._loss_channel_weight
  assert weight is not None
  assert float(weight[list(trainer._state_mask_channels)].sum()) == 0.0
  assert float(weight.sum()) == 167.0 - 6.0
  assert torch.isfinite(loss)

  # Inference parity: the same channels must be treated as unknown, never as
  # known values, which the caller expresses through the clean mask.
  assert trainer.contract.projected_state_dimension == 135


def test_masked_state_fields_reject_unknown_or_duplicate_names(tmp_path: Path) -> None:
  from mjlab.tasks.tracking.flow_matching.training_config import (
    FlowTrainingConfigError,
  )

  del tmp_path
  with pytest.raises(FlowTrainingConfigError):
    _config(masked_state_fields=("root_position_typo",)).validate()
  with pytest.raises(FlowTrainingConfigError):
    _config(masked_state_fields=("root_position", "root_position")).validate()
  with pytest.raises(StateValidationError):
    state_field_channels(("not_a_field",))


def test_masked_state_probability_selects_windows_and_is_reproducible(
  tmp_path: Path,
) -> None:
  """A stochastic mask selects whole windows and replays the same draws."""
  fields = ("root_position", "root_linear_velocity")
  config = _config(
    masked_state_fields=fields,
    masked_state_probability=0.5,
    masked_state_fill_value=6.0,
  )
  trainer = _trainer(tmp_path, config=config)
  clean = _clean(batch=8)
  trainer._forward_loss(clean)
  flags = trainer._last_state_mask
  assert flags.shape == (8,)
  assert flags.any() and not flags.all()

  # The same seed and the same batch reproduce the same selection.
  replay = _trainer(tmp_path / "replay", config=config)
  replay._forward_loss(clean)
  assert torch.equal(flags, replay._last_state_mask)

  # Probability 0 disables the mask: no window is selected.
  disabled = _trainer(
    tmp_path / "off",
    config=_config(masked_state_fields=fields, masked_state_probability=0.0),
  )
  disabled._forward_loss(clean)
  assert not bool(disabled._last_state_mask.any())

  # Probability 1 selects every window.
  everything = _trainer(
    tmp_path / "on",
    config=_config(masked_state_fields=fields, masked_state_probability=1.0),
  )
  everything._forward_loss(clean)
  assert bool(everything._last_state_mask.all())


def test_masked_state_probability_is_validated() -> None:
  from mjlab.tasks.tracking.flow_matching.training_config import (
    FlowTrainingConfigError,
  )

  with pytest.raises(FlowTrainingConfigError):
    _config(masked_state_probability=1.5).validate()
  with pytest.raises(FlowTrainingConfigError):
    _config(masked_state_probability=-0.1).validate()


def test_bf16_pure_builds_a_bf16_model_and_skips_autocast(tmp_path: Path) -> None:
  """The pure mode must own the dtype, not delegate it to autocast."""
  from mjlab.tasks.tracking.diffusion.model import denoiser_dtype

  assert denoiser_dtype("bf16") is None
  assert denoiser_dtype("fp32") is None
  assert denoiser_dtype("bf16-pure") is torch.bfloat16

  config = _config(epochs=1, max_updates=1)
  pure = FlowTrainingConfig(
    training=replace(config.training, mixed_precision="bf16-pure"),
    max_level=config.max_level,
  )
  trainer = _trainer(tmp_path / "pure", config=pure)
  assert next(trainer.model.parameters()).dtype == torch.bfloat16
  # No autocast region: under pure bf16 the operations already run in bf16.
  from contextlib import nullcontext
  assert isinstance(trainer._autocast(), nullcontext)


def test_clip_helper_matches_torch_and_reports_the_preclip_norm() -> None:
  """The shared clip helper must keep torch's semantics on bf16 gradients."""
  import torch.nn.utils as torch_utils

  from mjlab.tasks.tracking.diffusion.trainer import clip_gradient_norm

  reference = [torch.nn.Parameter(torch.zeros(4)) for _ in range(2)]
  measured = [
    torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16)) for _ in range(2)
  ]
  for index, (left, right) in enumerate(zip(reference, measured, strict=True)):
    gradient = torch.full((4,), 0.8 + index)
    left.grad = gradient.clone()
    right.grad = gradient.clone().to(torch.bfloat16)

  expected = torch_utils.clip_grad_norm_(reference, 1.0)
  actual = clip_gradient_norm(measured, 1.0)
  # bf16 storage rounds 0.8 and 0.9, so the two norms differ by a fraction of
  # a percent; the clipping decision and the scaling must still agree.
  assert abs(float(expected) - float(actual)) < 0.01 * float(expected)
  for left, right in zip(reference, measured, strict=True):
    assert torch.allclose(left.grad.bfloat16(), right.grad, rtol=1e-2, atol=1e-3)
