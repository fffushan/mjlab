"""Bounded x0 flow-matching trainer for the X2 tracking family.

The trainer consumes the Stage 1 primitives directly: it draws independent
integer state and latent levels for every row (including history), forms
``x_s = (1 - s) * x0 + s * e`` and fits the frozen Transformer with plain
channel-mean x0 MSE.  There is no clean-mask insertion: history rows are noised
and predicted like every other row, which preserves the baseline conditioning
strategy.  Level 0 samples are exactly clean and carry a reconstruction term.

``history_state_dropout_probability`` optionally drops the history pair of a
whole window, exactly as the diffusion trainer does: the first
``current_index`` rows of the state and latent levels are raised to
``process.max_level``, which is pure Gaussian in the linear path, so the model
can no longer read the true history for that window.  The default is 0.0.  The
dropout decision is one draw per window from a dedicated generator whose state
and counters are stored in the checkpoint, and evaluation never applies the
mask, so held-out losses stay comparable across dropout settings.

Optimizer, precision, EMA and gradient-accumulation behavior reuse the diffusion
infrastructure: :class:`~mjlab.tasks.tracking.diffusion.trainer.ExponentialMovingAverage`,
``warmup_cosine_learning_rate`` and the generic dataset-identity, token-batch and
sampling helpers are imported read-only.  No diffusion checkpoint or noise
schedule is constructed or written by this module.
"""

from __future__ import annotations

import json
import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler

from ..diffusion.contract import DiffusionContract
from ..diffusion.noising import clean_mask
from ..diffusion.state import state_field_channels
from ..diffusion.trainer import (
  ExponentialMovingAverage,
  _as_tokens,
  _average_metrics,
  _dataset_identity,
  _dataset_weights,
  _generator,
  _history_dropout_mask,
  _set_seed,
)
from ..diffusion.training_config import warmup_cosine_learning_rate
from .checkpoint import (
  FlowCheckpointError,
  FlowCheckpointState,
  identity_sha256,
  load_flow_checkpoint,
  model_identity,
  save_flow_checkpoint,
)
from .process import (
  LinearNoiseProcess,
  add_linear_noise,
  sample_independent_levels,
)
from .training_config import FlowTrainingConfig


class FlowTrainerError(RuntimeError):
  """The flow training state is invalid or became non-finite."""


@dataclass(frozen=True, slots=True)
class FlowTrainResult:
  """Summary and artifact paths from one bounded flow training invocation."""

  global_step: int
  epochs_completed: int
  best_validation_loss: float | None
  final_train_loss: float
  metrics_path: Path
  checkpoint_paths: dict[str, Path]


def _finite_tensor(value: Tensor, name: str) -> None:
  if not torch.isfinite(value).all():
    raise FlowTrainerError(f"{name} contains NaN or Inf")


def _mean_tensor_or_zero(value: Tensor) -> Tensor:
  if value.numel() == 0:
    return torch.zeros((), device=value.device, dtype=torch.float32)
  return value.mean().float()


def _x0_metrics(
  prediction: Tensor,
  target: Tensor,
  state_levels: Tensor,
  latent_levels: Tensor,
  unknown: Tensor,
  *,
  state_dimension: int,
) -> dict[str, float]:
  """Report plain x0 MSE overall and on the generated (unknown) entries.

  The overall metrics score every row, because flow training noises and
  predicts every row.  The unknown subset isolates the entries that the offline
  sampler actually generates, so a training curve is not misread as generation
  quality.
  """
  squared = (prediction.float() - target.float()).square()
  state = squared[..., :state_dimension]
  latent = squared[..., state_dimension:]
  unknown_batched = unknown.to(device=squared.device).expand(squared.shape[0], -1, -1)
  unknown_state = unknown_batched[..., :state_dimension]
  unknown_latent = unknown_batched[..., state_dimension:]
  clean_state = (state_levels == 0).to(device=squared.device).unsqueeze(-1)
  clean_latent = (latent_levels == 0).to(device=squared.device).unsqueeze(-1)
  values = {
    "loss": _mean_tensor_or_zero(squared),
    "total_mse": _mean_tensor_or_zero(squared),
    "state_mse": _mean_tensor_or_zero(state),
    "latent_mse": _mean_tensor_or_zero(latent),
    "unknown_mse": _mean_tensor_or_zero(squared[unknown_batched]),
    "unknown_state_mse": _mean_tensor_or_zero(state[unknown_state]),
    "unknown_latent_mse": _mean_tensor_or_zero(latent[unknown_latent]),
    "clean_state_mse": _mean_tensor_or_zero(state[clean_state.expand_as(state)]),
    "clean_latent_mse": _mean_tensor_or_zero(latent[clean_latent.expand_as(latent)]),
  }
  return {name: float(value.detach()) for name, value in values.items()}


class FlowTrainer:
  """Train an x0-predicting linear-flow denoiser on clean token windows."""

  def __init__(
    self,
    *,
    config: FlowTrainingConfig,
    contract: DiffusionContract,
    model: nn.Module,
    train_dataset: Any,
    eval_datasets: Mapping[str, Any],
    output_dir: Path,
    device: str | torch.device,
    resume: Path | None = None,
    allow_long_run: bool = False,
  ) -> None:
    if not isinstance(config, FlowTrainingConfig):
      raise TypeError("config must be a FlowTrainingConfig")
    config.validate(allow_long_run=allow_long_run)
    self.config = config
    self.training = config.training
    self.contract = contract
    self.process: LinearNoiseProcess = config.process
    self.model = model
    self.train_dataset = train_dataset
    self.eval_datasets = dict(eval_datasets)
    self.output_dir = Path(output_dir)
    self.output_dir.mkdir(parents=True, exist_ok=True)
    self.device = torch.device(device)
    if self.device.type == "cuda" and not torch.cuda.is_available():
      raise FlowTrainerError("CUDA device requested but CUDA is unavailable")
    self.model.to(self.device)
    self.ema = ExponentialMovingAverage(
      self.model,
      power=self.training.ema_power,
      max_decay=self.training.ema_max_decay,
    )
    self.optimizer = torch.optim.AdamW(
      self.model.parameters(),
      lr=self.training.learning_rate,
      weight_decay=self.training.weight_decay,
    )
    # bfloat16 autocast needs no loss scaler; we store None for resume parity.
    self.scaler: Any | None = None
    self.noise_generator = _generator(self.device, self.training.seed + 1)
    self.sampler_generator = torch.Generator(device="cpu").manual_seed(
      self.training.seed + 2
    )
    self.history_dropout_generator = _generator(self.device, self.training.seed + 3)
    self._history_dropout_windows = 0
    self._history_total_windows = 0
    self.state_mask_generator = _generator(self.device, self.training.seed + 4)
    self._state_mask_windows = 0
    self._state_mask_total_windows = 0
    # Optional state-field mask: the named fields are raised to pure noise on
    # every row and excluded from the loss, so the model cannot read them.
    masked = tuple(self.training.masked_state_fields)
    self._state_mask_channels = tuple(state_field_channels(masked) if masked else ())
    self._loss_channel_weight: Tensor | None = None
    if self._state_mask_channels:
      weight = torch.ones(
        self.contract.token_dimension, dtype=torch.float32, device=self.device
      )
      weight[list(self._state_mask_channels)] = 0.0
      self._loss_channel_weight = weight
    self.identity = _dataset_identity(self.train_dataset)
    self.config_hash = config.sha256()
    self.process_identity = self.process.identity_hash()
    self.data_identity = contract.identity_hash()
    self.statistics_identity = _statistics_identity(self.identity)
    self.model_description = model_identity(self.model)
    self.model_identity_hash = identity_sha256(self.model_description)
    self._unknown_mask = ~clean_mask(
      current_index=self.contract.current_index,
      state_dimension=self.contract.projected_state_dimension,
      token_dimension=self.contract.token_dimension,
      device=self.device,
    )
    self._fp32_checked = False
    self.global_step = 0
    self.start_epoch = 0
    self.best_validation_loss: float | None = None
    self.best_metric_split: str | None = None
    self._last_train_loss: float | None = None
    every = self.training.checkpoint_every
    self._next_periodic = every if every is not None else 0

    _set_seed(self.training.seed)
    if resume is not None:
      try:
        state = load_flow_checkpoint(
          resume,
          model=self.model,
          ema=self.ema,
          optimizer=self.optimizer,
          scaler=self.scaler,
          map_location=self.device,
          expected_process_identity=self.process_identity,
          expected_data_identity=self.data_identity,
        )
      except FlowCheckpointError as exc:
        raise FlowTrainerError(f"resume checkpoint is invalid: {exc}") from exc
      self._validate_resume(state)
      self._restore_rng_state(state.rng_state)
      self.global_step = state.global_step
      self.start_epoch = state.epoch
      self.best_validation_loss = state.best_validation_loss
      self.best_metric_split = state.best_metric_split
      self._fp32_checked = state.fp32_checked
      every = self.training.checkpoint_every
      if every is not None:
        self._next_periodic = (self.global_step // every + 1) * every

    self._updates_per_epoch = self._updates_for_samples(self._epoch_sample_count())
    self._resolved_updates = self.training.validate_resolved_updates(
      self._updates_per_epoch, allow_long_run=allow_long_run
    )
    if self.global_step > self._resolved_updates:
      raise FlowTrainerError("resume checkpoint is beyond the resolved training budget")
    self._write_run_metadata()

  def _epoch_sample_count(self) -> int:
    count = (
      self.training.epoch_sample_budget
      if self.training.epoch_sample_budget is not None
      else len(self.train_dataset)
    )
    if count <= 0:
      raise FlowTrainerError("training dataset or epoch_sample_budget is empty")
    return int(count)

  def _updates_for_samples(self, count: int) -> int:
    micro_batches = math.ceil(count / self.training.microbatch_size)
    return math.ceil(micro_batches / self.training.gradient_accumulation_steps)

  def _write_run_metadata(self) -> None:
    (self.output_dir / "resolved-config.json").write_text(
      json.dumps(
        {"config": self.config.as_dict(), "sha256": self.config_hash},
        sort_keys=True,
        indent=2,
      )
      + "\n",
      encoding="utf-8",
    )
    (self.output_dir / "README.md").write_text(
      "# Flow-matching training run\n\n"
      f"- family: `{self.process.as_dict()['family']}`\n"
      f"- process identity: `{self.process_identity}`\n"
      f"- config SHA256: `{self.config_hash}`\n"
      f"- data contract identity: `{self.data_identity}`\n"
      f"- model identity: `{self.model_identity_hash}`\n"
      f"- dataset identity: `{json.dumps(self.identity, sort_keys=True)}`\n"
      f"- resolved optimizer updates: `{self._resolved_updates}`\n"
      f"- effective batch size: `{self.training.effective_batch_size}`\n"
      "\nThis directory is a bounded engineering artifact; no hardware or quality\n"
      "claim is implied by its presence.\n",
      encoding="utf-8",
    )

  def _validate_resume(self, state: FlowCheckpointState) -> None:
    if state.config_hash != self.config_hash:
      raise FlowTrainerError("resume config identity mismatch")
    if state.family != self.process.as_dict()["family"]:
      raise FlowTrainerError("resume flow family mismatch")
    if state.process_identity != self.process_identity:
      raise FlowTrainerError("resume process identity mismatch")
    if state.data_identity != self.data_identity:
      raise FlowTrainerError("resume data identity mismatch")
    if dict(state.statistics_identity) != dict(self.statistics_identity):
      raise FlowTrainerError("resume statistics identity mismatch")
    if dict(state.dataset_identity) != dict(self.identity):
      raise FlowTrainerError("resume dataset identity mismatch")
    if state.model_identity_hash != self.model_identity_hash:
      raise FlowTrainerError("resume model identity mismatch")

  def _rng_state(self) -> dict[str, Any]:
    return {
      "python": random.getstate(),
      "numpy": np.random.get_state(),
      "torch": torch.get_rng_state(),
      "torch_cuda": torch.cuda.get_rng_state_all()
      if torch.cuda.is_available()
      else None,
      "noise_generator": self.noise_generator.get_state(),
      "sampler_generator": self.sampler_generator.get_state(),
      "history_dropout_generator": self.history_dropout_generator.get_state(),
      "state_mask_generator": self.state_mask_generator.get_state(),
      "history_dropout_counters": {
        "windows": self._history_dropout_windows,
        "total": self._history_total_windows,
      },
      "state_mask_counters": {
        "windows": self._state_mask_windows,
        "total": self._state_mask_total_windows,
      },
    }

  def _restore_rng_state(self, state: Any) -> None:
    if not isinstance(state, Mapping):
      raise FlowTrainerError("resume RNG state is malformed")
    try:
      random.setstate(state["python"])
      np.random.set_state(state["numpy"])
      torch.set_rng_state(state["torch"])
      if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
      self.noise_generator.set_state(state["noise_generator"])
      self.sampler_generator.set_state(state["sampler_generator"])
      if "state_mask_generator" in state:
        self.state_mask_generator.set_state(state["state_mask_generator"])
      elif self._state_mask_channels and float(
        self.training.masked_state_probability
      ) not in (0.0, 1.0):
        raise FlowTrainerError(
          "checkpoint lacks the state-mask RNG state for a stochastic mask run"
        )
      if "history_dropout_generator" in state:
        self.history_dropout_generator.set_state(state["history_dropout_generator"])
      elif float(self.training.history_state_dropout_probability) != 0.0:
        raise FlowTrainerError(
          "checkpoint lacks history dropout RNG state for a non-zero dropout run"
        )
      counters = state.get("history_dropout_counters")
      if isinstance(counters, Mapping):
        self._history_dropout_windows = int(counters.get("windows", 0))
        self._history_total_windows = int(counters.get("total", 0))
      mask_counters = state.get("state_mask_counters")
      if isinstance(mask_counters, Mapping):
        self._state_mask_windows = int(mask_counters.get("windows", 0))
        self._state_mask_total_windows = int(mask_counters.get("total", 0))
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
      if isinstance(exc, FlowTrainerError):
        raise
      raise FlowTrainerError("resume RNG state cannot be restored") from exc

  def _sampler(self) -> Sampler[int] | None:
    weights = _dataset_weights(self.train_dataset, self.training)
    sample_count = self._epoch_sample_count()
    if weights is None and self.training.epoch_sample_budget is None:
      return None
    if weights is None:
      weights = np.ones(len(self.train_dataset), dtype=np.float64)
    if len(weights) != len(self.train_dataset):
      raise FlowTrainerError("sampler weights do not match training dataset")
    if (
      self.training.epoch_sample_budget is None
      and len(weights) > 0
      and np.allclose(weights, weights[0], rtol=0.0, atol=0.0)
    ):
      return None
    return WeightedRandomSampler(
      weights.tolist(),
      num_samples=sample_count,
      replacement=True,
      generator=self.sampler_generator,
    )

  def _loader(self) -> DataLoader[Any]:
    sampler = self._sampler()
    # Window reads are small and scattered in a multi-gigabyte cache.  On a
    # local NVMe a single in-process reader costs most of a step, so an operator
    # may overlap the reads with worker processes.  The default stays 0, which
    # keeps the historical behaviour bit-identical.
    workers = int(os.environ.get("MJLAB_FLOW_DATALOADER_WORKERS", "0"))
    if workers < 0:
      raise FlowTrainerError("MJLAB_FLOW_DATALOADER_WORKERS must be non-negative")
    return DataLoader(
      self.train_dataset,
      batch_size=self.training.microbatch_size,
      sampler=sampler,
      shuffle=sampler is None,
      generator=self.sampler_generator if sampler is None else None,
      num_workers=workers,
      persistent_workers=False,
      prefetch_factor=4 if workers else None,
      drop_last=False,
    )

  def _autocast(self) -> Any:
    if self.training.mixed_precision == "bf16" and self.device.type == "cuda":
      return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()

  def _forward_loss(
    self, clean: Tensor
  ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Noise every row independently and fit plain channel-mean x0 MSE.

    With a non-zero ``history_state_dropout_probability`` the history pair of a
    whole window is raised to ``max_level``, which is pure Gaussian in the
    linear path, so that window cannot read its true history.  The diffusion
    trainer applies the same rule to the same rows.
    """
    clean = clean.to(device=self.device, dtype=torch.float32)
    _finite_tensor(clean, "clean tokens")
    batch = clean.shape[0]
    levels = sample_independent_levels(
      (batch, self.contract.window_steps),
      process=self.process,
      generator=self.noise_generator,
    )
    state_levels = levels.state.to(self.device)
    latent_levels = levels.latent.to(self.device)
    history_dropout = _history_dropout_mask(
      batch,
      float(self.training.history_state_dropout_probability),
      generator=self.history_dropout_generator,
      device=self.device,
    )
    if float(self.training.history_state_dropout_probability) != 0.0:
      history_rows = history_dropout[:, None].expand(-1, self.contract.current_index)
      full_level = self.process.max_level
      state_levels = state_levels.clone()
      latent_levels = latent_levels.clone()
      state_levels[:, : self.contract.current_index] = torch.where(
        history_rows,
        torch.full_like(state_levels[:, : self.contract.current_index], full_level),
        state_levels[:, : self.contract.current_index],
      )
      latent_levels[:, : self.contract.current_index] = torch.where(
        history_rows,
        torch.full_like(latent_levels[:, : self.contract.current_index], full_level),
        latent_levels[:, : self.contract.current_index],
      )
    noised = add_linear_noise(
      clean[..., : self.contract.projected_state_dimension],
      clean[..., self.contract.projected_state_dimension :],
      state_levels=state_levels,
      latent_levels=latent_levels,
      process=self.process,
      generator=self.noise_generator,
    )
    state_mask = _history_dropout_mask(
      batch,
      float(self.training.masked_state_probability)
      if self._state_mask_channels
      else 0.0,
      generator=self.state_mask_generator,
      device=self.device,
    )
    if self._state_mask_channels and bool(state_mask.any()):
      # Replace the named fields on the selected windows with their normalized
      # mean, which is zero.  The token layout is unchanged, the value carries
      # no sample information, and the levels stay per row because the linear
      # path is defined per row rather than per channel.
      channels = list(self._state_mask_channels)
      tokens_override = noised.tokens.clone()
      fill = float(self.training.masked_state_fill_value)
      selected = state_mask[:, None, None]
      replacement = torch.full_like(tokens_override[..., channels], fill)
      tokens_override[..., channels] = torch.where(
        selected, replacement, tokens_override[..., channels]
      )
      noised = replace(noised, tokens=tokens_override)
    step_ids = torch.stack((noised.state_levels, noised.latent_levels), dim=-1)
    if not self._fp32_checked:
      prediction = self.model(noised.tokens, step_ids)
      _finite_tensor(prediction, "FP32 denoiser output")
      self._fp32_checked = True
    else:
      with self._autocast():
        prediction = self.model(noised.tokens, step_ids)
      _finite_tensor(prediction, "denoiser output")
    target = clean
    error = (prediction.float() - target) ** 2
    if self._loss_channel_weight is None:
      # Plain channel-mean x0 MSE; unmasked runs keep the original expression.
      loss = error.mean()
    else:
      # Masked channels carry no input information on any row, so they are
      # excluded from the mean instead of spending capacity on them.
      weight = self._loss_channel_weight
      loss = (error * weight).sum() / (weight.sum() * error.shape[0] * error.shape[1])
    _finite_tensor(loss, "loss")
    self._last_state_mask = state_mask
    return (
      loss,
      prediction.float(),
      target,
      noised.state_levels,
      noised.latent_levels,
      history_dropout,
    )

  def _optimizer_update(
    self, pending: Sequence[Mapping[str, float]], *, update_index: int
  ) -> dict[str, float]:
    if not pending:
      raise FlowTrainerError("optimizer update has no accumulated micro-batches")
    divisor = float(len(pending))
    for parameter in self.model.parameters():
      if parameter.grad is not None:
        parameter.grad.div_(divisor)
        _finite_tensor(parameter.grad, "gradient")
    # ``clip_grad_norm_`` returns the total norm measured BEFORE it rescales
    # the gradients, so this value also shows clipping: a value above
    # ``max_grad_norm`` means the step was scaled down.
    gradient_norm = torch.nn.utils.clip_grad_norm_(
      self.model.parameters(), self.training.max_grad_norm
    )
    _finite_tensor(torch.as_tensor(gradient_norm), "clipped gradient norm")
    learning_rate = warmup_cosine_learning_rate(
      update_index,
      total_updates=self._resolved_updates,
      learning_rate=self.training.learning_rate,
      warmup_updates=self.training.warmup_updates,
    )
    for group in self.optimizer.param_groups:
      group["lr"] = learning_rate
    self.optimizer.step()
    for parameter in self.model.parameters():
      _finite_tensor(parameter, "model parameter")
    self.optimizer.zero_grad(set_to_none=True)
    decay = self.ema.update(self.model)
    self.global_step = update_index + 1
    result = _average_metrics(pending)
    result["learning_rate"] = learning_rate
    result["ema_decay"] = decay
    result["accumulated_microbatches"] = float(len(pending))
    result["global_step"] = float(self.global_step)
    result["gradient_norm"] = float(gradient_norm)
    self._last_train_loss = result["loss"]
    return result

  def _save(self, name: str, *, epoch: int, best_validation_loss: float | None) -> Path:
    path = self.output_dir / name
    save_flow_checkpoint(
      path,
      model=self.model,
      ema=self.ema,
      optimizer=self.optimizer,
      scaler=self.scaler,
      config=self.config,
      process=self.process,
      data_contract=self.contract,
      dataset_identity=self.identity,
      model_description=self.model_description,
      global_step=self.global_step,
      epoch=epoch,
      best_validation_loss=best_validation_loss,
      rng_state=self._rng_state(),
      best_metric_split=self.best_metric_split,
      selection_split=self.best_metric_split,
      fp32_checked=self._fp32_checked,
    )
    return path

  def train(self) -> FlowTrainResult:
    """Run the resolved bounded update budget and atomically save artifacts."""
    metrics_path = self.output_dir / "metrics.jsonl"
    checkpoint_paths: dict[str, Path] = {}
    epochs_completed = self.start_epoch
    mode_before = self.model.training
    self.model.train(True)
    with metrics_path.open("a", encoding="utf-8") as metrics_file:
      stop = False
      epoch = self.start_epoch
      while self.global_step < self._resolved_updates and epoch < self.training.epochs:
        loader = self._loader()
        pending: list[Mapping[str, float]] = []
        epoch_updates: list[Mapping[str, float]] = []
        epoch_dropout_windows = 0
        epoch_total_windows = 0
        self.optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loader):
          if self.global_step >= self._resolved_updates:
            stop = True
            break
          clean = _as_tokens(
            batch,
            window_steps=self.contract.window_steps,
            token_dimension=self.contract.token_dimension,
          )
          (
            loss,
            prediction,
            target,
            state_levels,
            latent_levels,
            history_dropout,
          ) = self._forward_loss(clean)
          dropout_windows = int(history_dropout.sum().item())
          total_windows = int(history_dropout.numel())
          self._history_dropout_windows += dropout_windows
          self._history_total_windows += total_windows
          epoch_dropout_windows += dropout_windows
          epoch_total_windows += total_windows
          state_mask_flags = getattr(self, "_last_state_mask", None)
          if state_mask_flags is not None:
            self._state_mask_windows += int(state_mask_flags.sum().item())
            self._state_mask_total_windows += int(state_mask_flags.numel())
          metrics = _x0_metrics(
            prediction,
            target,
            state_levels,
            latent_levels,
            self._unknown_mask,
            state_dimension=self.contract.projected_state_dimension,
          )
          metrics["history_state_dropout_fraction"] = (
            float(dropout_windows) / float(total_windows) if total_windows else 0.0
          )
          pending.append(metrics)
          loss.backward()
          is_last_batch = batch_index + 1 == len(loader)
          if len(pending) >= self.training.gradient_accumulation_steps or is_last_batch:
            update = self._optimizer_update(pending, update_index=self.global_step)
            epoch_updates.append(update)
            metrics_file.write(
              json.dumps({"kind": "update", "epoch": epoch, **update}, sort_keys=True)
              + "\n"
            )
            metrics_file.flush()
            pending = []
            every = self.training.checkpoint_every
            if every is not None and self.global_step >= self._next_periodic:
              checkpoint_paths["periodic"] = self._save(
                f"checkpoint-iter-{self.global_step:06d}.pt",
                epoch=epoch,
                best_validation_loss=self.best_validation_loss,
              )
              self._next_periodic = (self.global_step // every + 1) * every
        if pending:
          update = self._optimizer_update(pending, update_index=self.global_step)
          epoch_updates.append(update)
          metrics_file.write(
            json.dumps({"kind": "update", "epoch": epoch, **update}, sort_keys=True)
            + "\n"
          )
          metrics_file.flush()
        if stop:
          break
        epochs_completed = epoch + 1
        # The per-update rows carry their own ``global_step`` and
        # ``history_state_dropout_fraction``, so the averaged block is spread
        # FIRST: spreading it last would replace the explicit epoch-boundary
        # values with that epoch's mean.
        epoch_metrics: dict[str, Any] = {
          **_average_metrics(epoch_updates),
          "kind": "epoch",
          "epoch": epochs_completed,
          "global_step": self.global_step,
          "train_loss": self._last_train_loss,
          "history_state_dropout_probability": float(
            self.training.history_state_dropout_probability
          ),
          "history_state_dropout_windows": epoch_dropout_windows,
          "history_state_total_windows": epoch_total_windows,
          "history_state_dropout_fraction": (
            float(epoch_dropout_windows) / float(epoch_total_windows)
            if epoch_total_windows
            else 0.0
          ),
        }
        eval_split = self.training.eval_split
        if eval_split in self.eval_datasets:
          epoch_metrics["eval_split"] = eval_split
          epoch_metrics["eval_model"] = self.evaluate(eval_split, parameters="model")
          ema_values = self.evaluate(eval_split, parameters="ema")
          epoch_metrics["eval_ema"] = ema_values
          validation_loss = None
          if ema_values.get("windows", 0.0) > 0.0:
            validation_loss = ema_values.get("loss")
          if validation_loss is not None and math.isfinite(validation_loss):
            if (
              self.best_validation_loss is None
              or validation_loss < self.best_validation_loss
            ):
              self.best_validation_loss = validation_loss
              self.best_metric_split = eval_split
              checkpoint_paths["best"] = self._save(
                "checkpoint-best.pt",
                epoch=epochs_completed,
                best_validation_loss=self.best_validation_loss,
              )
        metrics_file.write(json.dumps(epoch_metrics, sort_keys=True) + "\n")
        metrics_file.flush()
        checkpoint_paths["last"] = self._save(
          "checkpoint-last.pt",
          epoch=epochs_completed,
          best_validation_loss=self.best_validation_loss,
        )
        epoch += 1
      if self.global_step:
        checkpoint_paths["last"] = self._save(
          "checkpoint-last.pt",
          epoch=epochs_completed,
          best_validation_loss=self.best_validation_loss,
        )
    self.model.train(mode_before)
    summary = {
      "global_step": self.global_step,
      "epochs_completed": epochs_completed,
      "best_validation_loss": self.best_validation_loss,
      "evaluation_split": self.training.eval_split,
      "best_evaluation_split": self.best_metric_split,
      "selection_split": self.best_metric_split,
      "final_train_loss": self._last_train_loss,
      "config_hash": self.config_hash,
      "family": self.process.as_dict()["family"],
      "process_identity": self.process_identity,
      "data_identity": self.data_identity,
      "model_identity": self.model_identity_hash,
      "dataset_identity": self.identity,
      "resolved_updates": self._resolved_updates,
      "effective_batch_size": self.training.effective_batch_size,
      "history_state_dropout_probability": float(
        self.training.history_state_dropout_probability
      ),
      "masked_state_fields": list(self.training.masked_state_fields),
      "masked_state_fill_value": float(self.training.masked_state_fill_value),
      "masked_state_probability": float(self.training.masked_state_probability),
      "masked_state_windows": self._state_mask_windows,
      "masked_state_total_windows": self._state_mask_total_windows,
      "masked_state_window_fraction": (
        float(self._state_mask_windows) / float(self._state_mask_total_windows)
        if self._state_mask_total_windows
        else 0.0
      ),
      "masked_state_channels": list(self._state_mask_channels),
      "masked_state_fraction_of_state": (
        float(len(self._state_mask_channels))
        / float(self.contract.projected_state_dimension)
        if self._state_mask_channels
        else 0.0
      ),
      "history_state_dropout_windows": self._history_dropout_windows,
      "history_state_total_windows": self._history_total_windows,
      "history_state_dropout_fraction": (
        float(self._history_dropout_windows) / float(self._history_total_windows)
        if self._history_total_windows
        else 0.0
      ),
      "resume_reproducibility": {
        "measured": False,
        "residual": None,
        "note": "No uninterrupted/resumed pair was measured by this invocation.",
      },
    }
    (self.output_dir / "train-report.json").write_text(
      json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    if self._last_train_loss is None:
      raise FlowTrainerError("training resolved to zero optimizer updates")
    return FlowTrainResult(
      self.global_step,
      epochs_completed,
      self.best_validation_loss,
      self._last_train_loss,
      metrics_path,
      checkpoint_paths,
    )

  def evaluate(self, split: str, *, parameters: str = "model") -> dict[str, float]:
    """Evaluate raw model or EMA on plain one-step x0 token MSE."""
    if split not in self.eval_datasets:
      raise FlowTrainerError(f"evaluation split {split!r} was not supplied")
    if parameters not in {"model", "ema"}:
      raise FlowTrainerError("parameters must be model or ema")
    dataset = self.eval_datasets[split]
    if len(dataset) == 0:
      return {"loss": 0.0, "windows": 0.0}
    backup = {
      name: value.detach().clone() for name, value in self.model.state_dict().items()
    }
    mode_before = self.model.training
    if parameters == "ema":
      self.ema.copy_to(self.model)
    self.model.eval()
    generator = _generator(self.device, self.training.eval_seed)
    aggregate: list[Mapping[str, float]] = []
    loader = DataLoader(
      dataset, batch_size=self.training.microbatch_size, num_workers=0
    )
    with torch.no_grad():
      for batch in loader:
        clean = _as_tokens(
          batch,
          window_steps=self.contract.window_steps,
          token_dimension=self.contract.token_dimension,
        )
        loss, prediction, target, state_levels, latent_levels = self._forward_eval(
          clean, generator
        )
        _finite_tensor(loss, "evaluation loss")
        aggregate.append(
          _x0_metrics(
            prediction,
            target,
            state_levels,
            latent_levels,
            self._unknown_mask,
            state_dimension=self.contract.projected_state_dimension,
          )
        )
    self.model.load_state_dict(backup, strict=True)
    self.model.train(mode_before)
    result = _average_metrics(aggregate)
    result["windows"] = float(len(dataset))
    return result

  def _forward_eval(
    self, clean: Tensor, generator: torch.Generator
  ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    clean = clean.to(device=self.device, dtype=torch.float32)
    batch = clean.shape[0]
    levels = sample_independent_levels(
      (batch, self.contract.window_steps),
      process=self.process,
      generator=generator,
    )
    noised = add_linear_noise(
      clean[..., : self.contract.projected_state_dimension],
      clean[..., self.contract.projected_state_dimension :],
      state_levels=levels.state.to(self.device),
      latent_levels=levels.latent.to(self.device),
      process=self.process,
      generator=generator,
    )
    step_ids = torch.stack((noised.state_levels, noised.latent_levels), dim=-1)
    prediction = self.model(noised.tokens, step_ids)
    target = clean
    loss = F.mse_loss(prediction.float(), target, reduction="mean")
    return loss, prediction.float(), target, noised.state_levels, noised.latent_levels


def _statistics_identity(identity: Mapping[str, Any]) -> dict[str, str]:
  """Extract the train-only statistics hashes recorded in a dataset identity."""
  projection_hashes = identity.get("projection_hashes", {})
  if not isinstance(projection_hashes, Mapping):
    return {}
  return {str(key): str(value) for key, value in projection_hashes.items()}


__all__ = [
  "FlowTrainResult",
  "FlowTrainer",
  "FlowTrainerError",
]
