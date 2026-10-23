"""CPU-friendly DDPM x0 trainer with accumulation, EMA and resume support."""

from __future__ import annotations

import json
import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler

from .checkpoint import (
  CheckpointError,
  CheckpointState,
  load_checkpoint,
  save_checkpoint,
)
from .contract import DiffusionContract
from .noising import add_independent_noise, clean_mask, sample_levels, x0_target
from .perf_switches import PerfSwitches
from .profiling import StepProfiler
from .schedule import DiffusionSchedule
from .training_config import (
  TrainingConfig,
  ema_decay,
  warmup_cosine_learning_rate,
)


class TrainerError(RuntimeError):
  """The training state is invalid or became non-finite."""


@dataclass(frozen=True, slots=True)
class TrainResult:
  """Summary and artifact paths from one bounded training invocation."""

  global_step: int
  epochs_completed: int
  best_validation_loss: float | None
  final_train_loss: float
  metrics_path: Path
  checkpoint_paths: dict[str, Path]


class ExponentialMovingAverage:
  """Power-law EMA shadow weights used for evaluation and checkpoints."""

  def __init__(
    self,
    model: nn.Module,
    *,
    power: float = 0.75,
    max_decay: float = 0.9999,
  ) -> None:
    if power <= 0.0 or not 0.0 < max_decay < 1.0:
      raise TrainerError("EMA power and max_decay are invalid")
    self.power = float(power)
    self.max_decay = float(max_decay)
    self.updates = 0
    self.shadow = {
      name: value.detach().clone() for name, value in model.state_dict().items()
    }

  def update(self, model: nn.Module, *, foreach: bool = False) -> float:
    """Update the shadow after one optimizer step and return its decay.

    ``foreach`` replaces one multiply and one add per tensor with the batched
    form. The arithmetic per tensor is identical, so the shadows stay
    bit-identical; only the dispatch count falls.
    """
    self.updates += 1
    decay = ema_decay(self.updates, power=self.power, max_decay=self.max_decay)
    with torch.no_grad():
      state = model.state_dict()
      for name in state:
        if name not in self.shadow:
          raise TrainerError(f"model parameter {name!r} appeared after EMA setup")
      if foreach:
        float_names = [n for n, v in state.items() if torch.is_floating_point(v)]
        shadows = [self.shadow[n] for n in float_names]
        sources = [state[n].detach() for n in float_names]
        if shadows:
          torch._foreach_mul_(shadows, decay)
          torch._foreach_add_(shadows, sources, alpha=1.0 - decay)
        for name, value in state.items():
          if not torch.is_floating_point(value):
            self.shadow[name].copy_(value.detach())
      else:
        for name, value in state.items():
          if torch.is_floating_point(value):
            self.shadow[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
          else:
            self.shadow[name].copy_(value.detach())
    return decay

  def copy_to(self, model: nn.Module) -> None:
    """Copy EMA values into ``model`` without changing the shadow."""
    model.load_state_dict(self.shadow, strict=True)

  def state_dict(self) -> dict[str, Any]:
    """Return serializable EMA state."""
    return {
      "power": self.power,
      "max_decay": self.max_decay,
      "updates": self.updates,
      "shadow": {name: value.clone() for name, value in self.shadow.items()},
    }

  def load_state_dict(self, state: Mapping[str, Any]) -> None:
    """Restore EMA state and reject a model-shape mismatch."""
    if not isinstance(state, Mapping) or not isinstance(state.get("shadow"), Mapping):
      raise TrainerError("malformed EMA checkpoint state")
    shadow = state["shadow"]
    if set(shadow) != set(self.shadow):
      raise TrainerError("EMA checkpoint parameter names do not match the model")
    for name, value in shadow.items():
      if not isinstance(value, Tensor) or value.shape != self.shadow[name].shape:
        raise TrainerError(f"EMA checkpoint parameter {name!r} has the wrong shape")
    self.power = float(state.get("power", self.power))
    self.max_decay = float(state.get("max_decay", self.max_decay))
    self.updates = int(state.get("updates", 0))
    self.shadow = {
      name: value.detach().clone().to(self.shadow[name].device)
      for name, value in shadow.items()
    }


def _set_seed(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  try:
    torch.use_deterministic_algorithms(True)
  except (RuntimeError, ValueError):
    # Some backends expose no deterministic implementation for an operation.
    # The measured resume residual is reported by the caller rather than hidden.
    pass


def _generator(device: torch.device, seed: int) -> torch.Generator:
  try:
    result = torch.Generator(device=device)
  except (RuntimeError, TypeError):
    result = torch.Generator(device="cpu")
  return result.manual_seed(seed)


def _finite_tensor(value: Tensor, name: str) -> None:
  if not torch.isfinite(value).all():
    raise TrainerError(f"{name} contains NaN or Inf")


def _history_dropout_mask(
  batch_size: int,
  probability: float,
  *,
  generator: torch.Generator,
  device: torch.device,
) -> Tensor:
  """Sample one history-pair dropout decision per training window."""
  if batch_size < 0:
    raise TrainerError("history dropout batch size must be non-negative")
  if not 0.0 <= probability <= 1.0 or not math.isfinite(probability):
    raise TrainerError("history dropout probability must be finite and in [0, 1]")
  if probability == 0.0:
    return torch.zeros(batch_size, dtype=torch.bool, device=device)
  if probability == 1.0:
    return torch.ones(batch_size, dtype=torch.bool, device=device)
  return (
    torch.rand(
      (batch_size,),
      generator=generator,
      device=device,
      dtype=torch.float32,
    )
    < probability
  )


class _FiniteGuard:
  """Collect finiteness flags and read them in one synchronisation.

  The immediate form of this check costs one device synchronisation per call.
  It is worth that cost for a small number of high-value tensors, such as the
  loss before a backward pass. It is not worth it for the per-parameter sweeps
  after a backward pass and after the optimizer step, which the step profile
  measured at about 170 calls per update. Flags stay on the device until a
  flush, so those sweeps cost one read each while detection stays inside the
  same update.
  """

  def __init__(self) -> None:
    self._pending: list[tuple[str, Tensor]] = []

  def check(self, value: Tensor, name: str) -> None:
    if value.numel() == 0:
      return
    self._pending.append((name, torch.isfinite(value).all()))

  def flush(self) -> None:
    """Read the collected flags. Raise on the first non-finite tensor."""
    if not self._pending:
      return
    pending, self._pending = self._pending, []
    combined = torch.stack([flag.reshape(()) for _, flag in pending]).all()
    if bool(combined):
      return
    # Slow path, entered only after a failure, so it can afford one read per
    # tensor to name the offending value.
    for name, flag in pending:
      if not bool(flag):
        raise TrainerError(f"{name} contains NaN or Inf")
    raise TrainerError("a checked tensor contains NaN or Inf")


def _as_tokens(batch: Any, *, window_steps: int, token_dimension: int) -> Tensor:
  if isinstance(batch, Tensor):
    tokens = batch
  elif isinstance(batch, Mapping):
    if "tokens" not in batch:
      raise TrainerError("dataset mapping batch lacks 'tokens'")
    tokens = batch["tokens"]
  elif isinstance(batch, (tuple, list)) and batch:
    tokens = batch[0]
  else:
    raise TrainerError("dataset must return tokens or (tokens, weight, index)")
  if not isinstance(tokens, Tensor):
    tokens = torch.as_tensor(tokens)
  if tokens.ndim != 3 or tokens.shape[-2:] != (window_steps, token_dimension):
    raise TrainerError(
      f"dataset tokens must have shape [B, {window_steps}, {token_dimension}], "
      f"got {tuple(tokens.shape)}"
    )
  if not tokens.is_floating_point():
    tokens = tokens.float()
  return tokens


def _dataset_identity(dataset: Any) -> dict[str, Any]:
  provided = getattr(dataset, "dataset_identity", None)
  if isinstance(provided, Mapping):
    identity = {str(key): value for key, value in provided.items()}
  else:
    identity = {}
  identity.setdefault("dataset_type", type(dataset).__qualname__)
  try:
    identity.setdefault("window_count", int(len(dataset)))
  except TypeError as exc:
    raise TrainerError("training dataset must implement __len__") from exc

  metadata = getattr(dataset, "metadata", None)
  if isinstance(metadata, Mapping):
    for key in (
      "source_dataset_hash",
      "contract_hash",
      "projection_hash",
      "split",
      "window_count",
    ):
      if key in metadata:
        identity.setdefault(key, metadata[key])
    if "projection_hash" in metadata:
      identity.setdefault(
        "projection_hashes", {"bundle": str(metadata["projection_hash"])}
      )
  index = getattr(dataset, "index", None)
  if index is not None:
    assignments = getattr(index, "assignments", None)
    if assignments is not None and hasattr(assignments, "sha256"):
      identity.setdefault("assignments_hash", str(assignments.sha256()))
    if hasattr(index, "coverage") and callable(index.coverage):
      identity.setdefault("split_coverage", index.coverage())
    store = getattr(index, "store", None)
    if store is not None:
      identity.setdefault("directory", str(getattr(store, "root", "")))
  statistics = getattr(dataset, "statistics", None)
  if statistics is not None:
    projection_hashes = identity.setdefault("projection_hashes", {})
    if isinstance(projection_hashes, dict):
      for name in ("matrix_sha256", "pseudoinverse_sha256", "statistics_sha256"):
        if hasattr(statistics, name):
          projection_hashes[name] = str(getattr(statistics, name))
  return identity


def _dataset_weights(dataset: Any, config: TrainingConfig) -> np.ndarray | None:
  records = getattr(dataset, "records", None)
  if records is not None:
    from .window_dataset import sampling_weights

    weights = sampling_weights(
      records,
      clean_weight=config.clean_weight,
      perturbed_weight=config.perturbed_weight,
    )
  else:
    weights = getattr(dataset, "weights", None)
  if weights is None:
    return None
  values = np.asarray(weights, dtype=np.float64)
  if values.shape != (len(dataset),) or not np.isfinite(values).all():
    raise TrainerError(
      "training sampling weights have the wrong shape or are non-finite"
    )
  if np.any(values < 0.0) or (len(values) and not np.any(values > 0.0)):
    raise TrainerError("training sampling weights must select at least one window")
  return values


def _level_bucket(level: Tensor) -> list[tuple[str, Tensor]]:
  return [
    ("clean", level == 0),
    ("1_9", (level >= 1) & (level < 10)),
    ("10_99", (level >= 10) & (level < 100)),
    ("100_999", (level >= 100) & (level < 1000)),
    ("terminal", level == 1000),
  ]


def _mean_or_zero(value: Tensor) -> float:
  return float(value.mean().item()) if value.numel() else 0.0


def _mean_tensor_or_zero(value: Tensor) -> Tensor:
  if value.numel() == 0:
    return torch.zeros((), device=value.device, dtype=torch.float32)
  return value.mean().float()


def _metrics(
  prediction: Tensor,
  target: Tensor,
  k_state: Tensor,
  k_latent: Tensor,
  unknown: Tensor,
  *,
  state_width: int,
  single_read: bool = False,
) -> dict[str, float]:
  squared = (prediction.float() - target.float()).square()
  state = squared[..., :state_width]
  latent = squared[..., state_width:]
  unknown_batched = unknown.to(device=squared.device).expand(squared.shape[0], -1, -1)
  unknown_sq = squared[unknown_batched]
  unknown_state_mask = unknown_batched[..., :state_width]
  unknown_latent_mask = unknown_batched[..., state_width:]
  pairs: list[tuple[str, Tensor]] = [
    ("loss", _mean_tensor_or_zero(squared)),
    ("total_mse", _mean_tensor_or_zero(squared)),
    ("state_mse", _mean_tensor_or_zero(state)),
    ("latent_mse", _mean_tensor_or_zero(latent)),
    ("unknown_mse", _mean_tensor_or_zero(unknown_sq)),
    ("unknown_state_mse", _mean_tensor_or_zero(state[unknown_state_mask])),
    ("unknown_latent_mse", _mean_tensor_or_zero(latent[unknown_latent_mask])),
  ]
  for prefix, levels, width_slice in (
    ("state", k_state, (..., slice(0, state_width))),
    ("latent", k_latent, (..., slice(state_width, None))),
  ):
    selected_values = squared[width_slice]
    for bucket, mask in _level_bucket(levels):
      expanded = mask.unsqueeze(-1).expand_as(selected_values)
      pairs.append(
        (f"{prefix}_mse_{bucket}", _mean_tensor_or_zero(selected_values[expanded]))
      )
  if single_read:
    # One transfer for the whole metric set. The values are unchanged because
    # only the read is batched, not the reduction.
    scalars = torch.stack([value.reshape(()) for _, value in pairs]).tolist()
    return {
      name: float(scalar) for (name, _), scalar in zip(pairs, scalars, strict=True)
    }
  return {name: float(value.item()) for name, value in pairs}


def _average_metrics(values: Sequence[Mapping[str, float]]) -> dict[str, float]:
  if not values:
    return {}
  names = values[0].keys()
  return {
    name: float(sum(float(item[name]) for item in values) / len(values))
    for name in names
  }


class DiffusionTrainer:
  """Train an x0-predicting denoiser on cached or synthetic token windows."""

  def __init__(
    self,
    *,
    config: TrainingConfig,
    contract: DiffusionContract,
    schedule: DiffusionSchedule,
    model: nn.Module,
    train_dataset: Any,
    eval_datasets: Mapping[str, Any],
    output_dir: Path,
    device: str | torch.device,
    resume: Path | None = None,
    checkpoint_sidecar: Mapping[str, Any] | None = None,
    allow_long_run: bool = False,
  ) -> None:
    if not isinstance(config, TrainingConfig):
      raise TypeError("config must be a TrainingConfig")
    config.validate(allow_long_run=allow_long_run)
    self.config = config
    self.contract = contract
    self.schedule = schedule
    self.model = model
    self.train_dataset = train_dataset
    self.eval_datasets = dict(eval_datasets)
    self.output_dir = Path(output_dir)
    self.output_dir.mkdir(parents=True, exist_ok=True)
    self.device = torch.device(device)
    if self.device.type == "cuda" and not torch.cuda.is_available():
      raise TrainerError("CUDA device requested but CUDA is unavailable")
    self.model.to(self.device)
    self.ema = ExponentialMovingAverage(
      self.model,
      power=self.config.ema_power,
      max_decay=self.config.ema_max_decay,
    )
    self.optimizer = torch.optim.AdamW(
      self.model.parameters(),
      lr=self.config.learning_rate,
      weight_decay=self.config.weight_decay,
    )
    self.scaler: Any | None = None
    self.noise_generator = _generator(self.device, self.config.seed + 1)
    self.sampler_generator = torch.Generator(device="cpu").manual_seed(
      self.config.seed + 2
    )
    self.history_dropout_generator = _generator(self.device, self.config.seed + 3)
    self._history_dropout_windows = 0
    self._history_total_windows = 0
    self.identity = _dataset_identity(self.train_dataset)
    self.config_hash = self.config.sha256()
    self._alpha_bars = self.schedule.alpha_bars_torch(
      device=self.device, dtype=torch.float32
    )
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
    self._resume_state: CheckpointState | None = None
    self._checkpoint_sidecar = (
      dict(checkpoint_sidecar) if checkpoint_sidecar is not None else None
    )
    # First global step at which a periodic checkpoint is due.  Unlike best/last
    # these exist so the validation curve and its optimum stay inspectable.
    every = self.config.checkpoint_every
    self._next_periodic = every if every is not None else 0

    # Inert unless MJLAB_DIFFUSION_PROFILE is set.  Records a bounded window of
    # steady-state steps so the dominant cost of one step can be named.
    self._profiler = StepProfiler.from_environment(self.output_dir)
    self._finite = _FiniteGuard()
    self._perf = PerfSwitches.from_environment()
    # Compiled as a callable over the module, never as the module itself: a
    # compiled module renames its state-dict entries and would break resume.
    self._compiled_model: Any = None

    _set_seed(self.config.seed)
    if resume is not None:
      try:
        state = load_checkpoint(
          resume,
          model=self.model,
          ema=self.ema,
          optimizer=self.optimizer,
          scaler=self.scaler,
          map_location=self.device,
          expected_contract_identity=self.contract.identity_hash(),
        )
      except CheckpointError as exc:
        raise TrainerError(f"resume checkpoint is invalid: {exc}") from exc
      self._validate_resume(state)
      self._restore_rng_state(state.rng_state)
      self.global_step = state.global_step
      self.start_epoch = state.epoch
      self.best_validation_loss = state.best_validation_loss
      self.best_metric_split = state.best_metric_split
      self._fp32_checked = state.fp32_checked
      self._resume_state = state
      every = self.config.checkpoint_every
      if every is not None:
        self._next_periodic = (self.global_step // every + 1) * every

    self._updates_per_epoch = self._updates_for_samples(self._epoch_sample_count())
    self._resolved_updates = self.config.validate_resolved_updates(
      self._updates_per_epoch, allow_long_run=allow_long_run
    )
    if self.global_step > self._resolved_updates:
      raise TrainerError("resume checkpoint is beyond the resolved training budget")
    self._write_run_metadata()

  def _epoch_sample_count(self) -> int:
    count = (
      self.config.epoch_sample_budget
      if self.config.epoch_sample_budget is not None
      else len(self.train_dataset)
    )
    if count <= 0:
      raise TrainerError("training dataset or epoch_sample_budget is empty")
    return int(count)

  def _updates_for_samples(self, count: int) -> int:
    micro_batches = math.ceil(count / self.config.microbatch_size)
    return math.ceil(micro_batches / self.config.gradient_accumulation_steps)

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
      "# Diffusion training run\n\n"
      f"- config SHA256: `{self.config_hash}`\n"
      f"- contract identity: `{self.contract.identity_hash()}`\n"
      f"- schedule identity: `{self.schedule.identity_hash()}`\n"
      f"- dataset identity: `{json.dumps(self.identity, sort_keys=True)}`\n"
      f"- resolved optimizer updates: `{self._resolved_updates}`\n"
      f"- effective batch size: `{self.config.effective_batch_size}`\n"
      "\nThis directory is a bounded engineering artifact; no hardware or quality\n"
      "claim is implied by its presence.\n",
      encoding="utf-8",
    )

  def _validate_resume(self, state: CheckpointState) -> None:
    if state.config_hash != self.config_hash:
      raise TrainerError("resume config identity mismatch")
    if state.contract_identity != self.contract.identity_hash():
      raise TrainerError("resume contract identity mismatch")
    if state.schedule_identity != self.schedule.identity_hash():
      raise TrainerError("resume schedule identity mismatch")
    if dict(state.dataset_identity) != dict(self.identity):
      raise TrainerError("resume dataset identity mismatch")
    expected_projection = self.identity.get("projection_hashes", {})
    if dict(state.projection_hashes) != dict(expected_projection):
      raise TrainerError("resume projection identity mismatch")

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
      "history_dropout_counters": {
        "windows": self._history_dropout_windows,
        "total": self._history_total_windows,
      },
    }

  def _restore_rng_state(self, state: Any) -> None:
    if not isinstance(state, Mapping):
      raise TrainerError("resume RNG state is malformed")
    try:
      random.setstate(state["python"])
      np.random.set_state(state["numpy"])
      torch.set_rng_state(state["torch"])
      if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
      self.noise_generator.set_state(state["noise_generator"])
      self.sampler_generator.set_state(state["sampler_generator"])
      if "history_dropout_generator" in state:
        self.history_dropout_generator.set_state(state["history_dropout_generator"])
      elif self.config.history_state_dropout_probability != 0.0:
        raise TrainerError(
          "checkpoint lacks history dropout RNG state for a non-zero dropout run"
        )
      counters = state.get("history_dropout_counters")
      if isinstance(counters, Mapping):
        self._history_dropout_windows = int(counters.get("windows", 0))
        self._history_total_windows = int(counters.get("total", 0))
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
      raise TrainerError("resume RNG state cannot be restored") from exc

  def _sampler(self, epoch: int) -> Sampler[int] | None:
    weights = _dataset_weights(self.train_dataset, self.config)
    sample_count = self._epoch_sample_count()
    if weights is None and self.config.epoch_sample_budget is None:
      return None
    if weights is None:
      weights = np.ones(len(self.train_dataset), dtype=np.float64)
    if len(weights) != len(self.train_dataset):
      raise TrainerError("sampler weights do not match training dataset")
    # Equal weights without an explicit budget use every window once.  A budget
    # or non-uniform phase weights takes the replacement path so the requested
    # sample count and clean/OU mixture are exact.
    if (
      self.config.epoch_sample_budget is None
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

  def _loader(self, epoch: int) -> DataLoader[Any]:
    sampler = self._sampler(epoch)
    return DataLoader(
      self.train_dataset,
      batch_size=self.config.microbatch_size,
      sampler=sampler,
      shuffle=sampler is None,
      generator=self.sampler_generator if sampler is None else None,
      num_workers=self._perf.loader_workers,
      drop_last=False,
    )

  def _autocast(self) -> Any:
    if self.config.mixed_precision == "bf16" and self.device.type == "cuda":
      return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()

  def _model_forward(self, tokens: Tensor, step_ids: Tensor) -> Tensor:
    """Call the denoiser, optionally through a compiled wrapper.

    The wrapper closes over the module, so the module itself stays untouched
    and its state-dict keys keep their names.
    """
    if not self._perf.compiled_forward:
      return self.model(tokens, step_ids)
    if self._compiled_model is None:
      model = self.model
      self._compiled_model = torch.compile(lambda t, s: model(t, s), dynamic=False)
    return self._compiled_model(tokens, step_ids)

  def _forward_loss(
    self, clean: Tensor
  ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    clean = clean.to(device=self.device, dtype=torch.float32)
    _finite_tensor(clean, "clean tokens")
    batch = clean.shape[0]
    k_state = sample_levels(
      (batch, self.contract.window_steps),
      training_k=self.schedule.training_k,
      generator=self.noise_generator,
      mode=self.config.level_sampling,
      shift_gamma=self.config.level_shift_gamma,
      logit_std=self.config.level_logit_std,
    ).to(self.device)
    k_latent = sample_levels(
      (batch, self.contract.window_steps),
      training_k=self.schedule.training_k,
      generator=self.noise_generator,
      mode=self.config.level_sampling,
      shift_gamma=self.config.level_shift_gamma,
      logit_std=self.config.level_logit_std,
    ).to(self.device)
    history_dropout = _history_dropout_mask(
      batch,
      self.config.history_state_dropout_probability,
      generator=self.history_dropout_generator,
      device=self.device,
    )
    if self.config.history_state_dropout_probability != 0.0:
      history_rows = history_dropout[:, None].expand(-1, self.contract.current_index)
      k_state = k_state.clone()
      k_latent = k_latent.clone()
      k_state[:, : self.contract.current_index] = torch.where(
        history_rows,
        torch.full_like(
          k_state[:, : self.contract.current_index], self.schedule.training_k
        ),
        k_state[:, : self.contract.current_index],
      )
      k_latent[:, : self.contract.current_index] = torch.where(
        history_rows,
        torch.full_like(
          k_latent[:, : self.contract.current_index], self.schedule.training_k
        ),
        k_latent[:, : self.contract.current_index],
      )
    noised = add_independent_noise(
      clean[..., : self.contract.projected_state_dimension],
      clean[..., self.contract.projected_state_dimension :],
      k_state,
      k_latent,
      self._alpha_bars,
      generator=self.noise_generator,
    )
    step_ids = torch.stack((k_state, k_latent), dim=-1)
    if not self._fp32_checked:
      prediction = self._model_forward(noised.tokens, step_ids)
      _finite_tensor(prediction, "FP32 denoiser output")
      self._fp32_checked = True
    else:
      with self._autocast():
        prediction = self._model_forward(noised.tokens, step_ids)
      _finite_tensor(prediction, "denoiser output")
    target = x0_target(clean)
    state_weight = float(self.config.state_loss_weight)
    latent_weight = float(self.config.latent_loss_weight)
    if state_weight == 1.0 and latent_weight == 1.0:
      # Unmodified path: keep the frozen D2 objective byte-identical.
      loss = F.mse_loss(prediction.float(), target, reduction="mean")
    else:
      # Block-weighted replacement for the plain mean.  The denominator keeps
      # the objective on the same per-channel scale as the unmodified loss so
      # learning rates stay comparable across arms.
      difference = (prediction.float() - target).square()
      split = self.contract.projected_state_dimension
      state_squared = difference[..., :split]
      latent_squared = difference[..., split:]
      denominator = state_weight * state_squared.numel() + latent_weight * (
        latent_squared.numel()
      )
      if denominator <= 0.0:
        raise TrainerError("at least one loss weight must be positive")
      loss = (
        state_weight * state_squared.sum() + latent_weight * latent_squared.sum()
      ) / denominator
    _finite_tensor(loss, "loss")
    return loss, prediction.float(), target, k_state, k_latent, history_dropout

  def _optimizer_update(
    self, pending: Sequence[Mapping[str, float]], *, update_index: int
  ) -> dict[str, float]:
    if not pending:
      raise TrainerError("optimizer update has no accumulated micro-batches")
    divisor = float(len(pending))
    if self._perf.finite_combined:
      # One combined check per update instead of one check per parameter. The
      # parameter set is fully covered; only the dispatch count falls.
      for parameter in self.model.parameters():
        if parameter.grad is not None:
          parameter.grad.div_(divisor)
      grads = [p.grad for p in self.model.parameters() if p.grad is not None]
      if grads:
        _finite_tensor(torch.cat([g.reshape(-1) for g in grads]), "stacked gradients")
      self._finite.flush()
    else:
      for parameter in self.model.parameters():
        if parameter.grad is not None:
          parameter.grad.div_(divisor)
          self._finite.check(parameter.grad, "gradient")
    gradient_norm = torch.nn.utils.clip_grad_norm_(
      self.model.parameters(), self.config.max_grad_norm
    )
    self._finite.check(torch.as_tensor(gradient_norm), "clipped gradient norm")
    # One read for every flag above, before the optimizer step: a non-finite
    # gradient must still stop the step rather than reach the weights.
    self._finite.flush()
    learning_rate = warmup_cosine_learning_rate(
      update_index,
      total_updates=self._resolved_updates,
      learning_rate=self.config.learning_rate,
      warmup_updates=self.config.warmup_updates,
    )
    for group in self.optimizer.param_groups:
      group["lr"] = learning_rate
    self.optimizer.step()
    if self._perf.finite_combined:
      parameters = [p.detach() for p in self.model.parameters()]
      if parameters:
        _finite_tensor(
          torch.cat([p.reshape(-1) for p in parameters]), "stacked parameters"
        )
    else:
      for parameter in self.model.parameters():
        self._finite.check(parameter, "model parameter")
      self._finite.flush()
    self.optimizer.zero_grad(set_to_none=True)
    decay = self.ema.update(self.model, foreach=self._perf.ema_foreach)
    self.global_step = update_index + 1
    result = _average_metrics(pending)
    result["learning_rate"] = learning_rate
    result["ema_decay"] = decay
    result["accumulated_microbatches"] = float(len(pending))
    result["global_step"] = float(self.global_step)
    # ``clip_grad_norm_`` returns the norm before clipping, so a value above
    # ``max_grad_norm`` records that this update was scaled down.
    result["gradient_norm"] = float(gradient_norm)
    self._last_train_loss = result["loss"]
    return result

  def _write_checkpoint_sidecar(self, path: Path, name: str) -> None:
    if self._checkpoint_sidecar is None:
      return
    try:
      import yaml

      payload = {
        **self._checkpoint_sidecar,
        "checkpoint": name,
        "global_step": self.global_step,
      }
      temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
      temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
      temporary.replace(path.with_suffix(".yaml"))
    except (ImportError, OSError, TypeError, ValueError) as exc:
      raise TrainerError(
        f"could not write checkpoint sidecar for {path.name}: {exc}"
      ) from exc

  def _save(self, name: str, *, epoch: int, best_validation_loss: float | None) -> Path:
    path = self.output_dir / name
    save_checkpoint(
      path,
      model=self.model,
      ema=self.ema,
      optimizer=self.optimizer,
      scaler=self.scaler,
      config=self.config,
      contract=self.contract,
      schedule=self.schedule,
      dataset_identity=self.identity,
      global_step=self.global_step,
      epoch=epoch,
      best_validation_loss=best_validation_loss,
      rng_state=self._rng_state(),
      best_metric_split=self.best_metric_split,
      selection_split=self.best_metric_split,
      fp32_checked=self._fp32_checked,
    )
    self._write_checkpoint_sidecar(path, name)
    return path

  def train(self) -> TrainResult:
    """Run the resolved bounded update budget and atomically save artifacts."""
    metrics_path = self.output_dir / "metrics.jsonl"
    checkpoint_paths: dict[str, Path] = {}
    epochs_completed = self.start_epoch
    mode_before = self.model.training
    self.model.train(True)
    with metrics_path.open("a", encoding="utf-8") as metrics_file:
      stop = False
      epoch = self.start_epoch
      while self.global_step < self._resolved_updates and epoch < self.config.epochs:
        loader = self._loader(epoch)
        pending: list[Mapping[str, float]] = []
        epoch_updates: list[Mapping[str, float]] = []
        epoch_dropout_windows = 0
        epoch_total_windows = 0
        self.optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loader):
          if self.global_step >= self._resolved_updates:
            stop = True
            break
          self._profiler.on_step_start(self.global_step)
          clean = _as_tokens(
            batch,
            window_steps=self.contract.window_steps,
            token_dimension=self.contract.token_dimension,
          )
          (
            loss,
            prediction,
            target,
            k_state,
            k_latent,
            history_dropout,
          ) = self._forward_loss(clean)
          dropout_windows = int(history_dropout.sum().item())
          total_windows = int(history_dropout.numel())
          self._history_dropout_windows += dropout_windows
          self._history_total_windows += total_windows
          epoch_dropout_windows += dropout_windows
          epoch_total_windows += total_windows
          metrics = _metrics(
            prediction,
            target,
            k_state,
            k_latent,
            self._unknown_mask,
            state_width=self.contract.projected_state_dimension,
            single_read=self._perf.metrics_single_read,
          )
          metrics["history_state_dropout_fraction"] = (
            float(dropout_windows) / float(total_windows) if total_windows else 0.0
          )
          pending.append(metrics)
          loss.backward()
          is_last_batch = batch_index + 1 == len(loader)
          if len(pending) >= self.config.gradient_accumulation_steps or is_last_batch:
            update = self._optimizer_update(pending, update_index=self.global_step)
            epoch_updates.append(update)
            metrics_file.write(
              json.dumps({"kind": "update", "epoch": epoch, **update}, sort_keys=True)
              + "\n"
            )
            metrics_file.flush()
            pending = []
            every = self.config.checkpoint_every
            if every is not None and self.global_step >= self._next_periodic:
              checkpoint_paths["periodic"] = self._save(
                f"checkpoint-iter-{self.global_step:06d}.pt",
                epoch=epoch,
                best_validation_loss=self.best_validation_loss,
              )
              self._next_periodic = (self.global_step // every + 1) * every
            self._profiler.on_step_end()
        if pending:
          # This is defensive: the normal last-batch flush above handles a
          # partial final group, but never discard gradients if a custom loader
          # reports an inconsistent length.
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
          "history_state_dropout_probability": self.config.history_state_dropout_probability,
          "history_state_dropout_windows": epoch_dropout_windows,
          "history_state_total_windows": epoch_total_windows,
          "history_state_dropout_fraction": (
            float(epoch_dropout_windows) / float(epoch_total_windows)
            if epoch_total_windows
            else 0.0
          ),
        }
        validation_loss: float | None = None
        eval_split = self.config.eval_split
        if eval_split in self.eval_datasets:
          raw = self.evaluate(eval_split, parameters="model")
          ema_values = self.evaluate(eval_split, parameters="ema")
          epoch_metrics["eval_split"] = eval_split
          epoch_metrics["eval_model"] = raw
          epoch_metrics["eval_ema"] = ema_values
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
      "best_evaluation_loss": self.best_validation_loss,
      "evaluation_split": self.config.eval_split,
      "best_evaluation_split": self.best_metric_split,
      "selection_split": self.best_metric_split,
      "final_train_loss": self._last_train_loss,
      "config_hash": self.config_hash,
      "dataset_identity": self.identity,
      "resolved_updates": self._resolved_updates,
      "effective_batch_size": self.config.effective_batch_size,
      "history_state_dropout_probability": self.config.history_state_dropout_probability,
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
      raise TrainerError("training resolved to zero optimizer updates")
    self._profiler.finish()
    return TrainResult(
      self.global_step,
      epochs_completed,
      self.best_validation_loss,
      self._last_train_loss,
      metrics_path,
      checkpoint_paths,
    )

  def evaluate(self, split: str, *, parameters: str = "model") -> dict[str, float]:
    """Evaluate raw model or EMA on plain x0 token MSE."""
    if split not in self.eval_datasets:
      raise TrainerError(f"evaluation split {split!r} was not supplied")
    if parameters not in {"model", "ema"}:
      raise TrainerError("parameters must be model or ema")
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
    generator = _generator(self.device, self.config.eval_seed)
    aggregate: list[Mapping[str, float]] = []
    loader = DataLoader(dataset, batch_size=self.config.microbatch_size, num_workers=0)
    with torch.no_grad():
      for batch in loader:
        clean = _as_tokens(
          batch,
          window_steps=self.contract.window_steps,
          token_dimension=self.contract.token_dimension,
        )
        loss, prediction, target, k_state, k_latent = self._forward_eval(
          clean, generator
        )
        _finite_tensor(loss, "evaluation loss")
        aggregate.append(
          _metrics(
            prediction,
            target,
            k_state,
            k_latent,
            self._unknown_mask,
            state_width=self.contract.projected_state_dimension,
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
    k_state = sample_levels(
      (batch, self.contract.window_steps),
      training_k=self.schedule.training_k,
      generator=generator,
    ).to(self.device)
    k_latent = sample_levels(
      (batch, self.contract.window_steps),
      training_k=self.schedule.training_k,
      generator=generator,
    ).to(self.device)
    noised = add_independent_noise(
      clean[..., : self.contract.projected_state_dimension],
      clean[..., self.contract.projected_state_dimension :],
      k_state,
      k_latent,
      self._alpha_bars,
      generator=generator,
    )
    prediction = self.model(noised.tokens, torch.stack((k_state, k_latent), dim=-1))
    target = x0_target(clean)
    loss = F.mse_loss(prediction.float(), target, reduction="mean")
    return loss, prediction.float(), target, k_state, k_latent


__all__ = [
  "DiffusionTrainer",
  "ExponentialMovingAverage",
  "TrainResult",
  "TrainerError",
  "ema_decay",
  "warmup_cosine_learning_rate",
]
