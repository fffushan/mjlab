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

Data parallelism is opt-in through the launcher environment: when ``WORLD_SIZE``
exceeds one, every rank reads ``RANK`` and ``LOCAL_RANK``, joins a process group
(nccl on CUDA, gloo on CPU) and wraps the denoiser in
``DistributedDataParallel``.  The global batch is held fixed by dividing the
microbatch by the world size, gradients are all-reduced once per optimizer
update, every rank reads a disjoint contiguous chunk of one shared epoch
permutation, and rank 0 alone writes metrics, TensorBoard rows and checkpoints.
``WORLD_SIZE == 1`` keeps the single-process behaviour exactly: no process
group, no wrapper, and the historical sampler, metric and checkpoint payloads.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections.abc import Sized
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributed import (
  ReduceOp,
  all_gather_object,
  all_reduce,
  barrier,
  broadcast,
  get_rank,
  get_world_size,
  init_process_group,
  is_available,
  is_initialized,
)
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler

from ..diffusion.contract import DiffusionContract
from ..diffusion.model import denoiser_dtype
from ..diffusion.noising import clean_mask
from ..diffusion.state import HYBRID_STATE, state_field_channels
from ..diffusion.trainer import (
  ExponentialMovingAverage,
  _as_tokens,
  _average_metrics,
  _dataset_identity,
  _dataset_weights,
  _generator,
  _history_dropout_mask,
  _set_seed,
  clip_gradient_norm,
)
from ..diffusion.training_config import (
  config_differences,
  warmup_cosine_learning_rate,
)
from ..tensorboard_logging import open_for
from .checkpoint import (
  FlowCheckpointError,
  FlowCheckpointState,
  identity_sha256,
  load_flow_checkpoint,
  model_identity,
  save_flow_checkpoint,
)
from .diagnostics import (
  ENV_DIRECTORY,
  FlowDiagnostics,
  channel_groups,
  gradient_norms_by_block,
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


_DISTRIBUTED_TIMEOUT_SECONDS = 1800
"""Finite process-group timeout, in seconds.

A rank that dies leaves the survivors blocked inside a collective.  A finite
timeout turns that block into a reported failure; an infinite one turns the job
into a set of cards that are held and never explain why.
"""


@dataclass(frozen=True, slots=True)
class DistributedContext:
  """One process's place in the data-parallel group.

  ``world_size == 1`` means no group was created at all: ``backend`` is ``None``
  and ``device`` is the requested device unchanged, so a single-process run
  cannot observe a difference from the pre-distributed trainer.
  """

  world_size: int
  rank: int
  local_rank: int
  device: torch.device
  backend: str | None

  @property
  def is_distributed(self) -> bool:
    """True when this run spans more than one process."""
    return self.world_size > 1


def _environment_int(name: str) -> int | None:
  raw = os.environ.get(name, "")
  if raw.strip() == "":
    return None
  try:
    return int(raw)
  except ValueError as exc:
    raise FlowTrainerError(f"{name} must be an integer, got {raw!r}") from exc


def initialize_distributed(
  device: str | torch.device, *, timeout_seconds: int = _DISTRIBUTED_TIMEOUT_SECONDS
) -> DistributedContext:
  """Resolve this worker's rank and join the process group when it is shared.

  A launcher exports ``WORLD_SIZE``, ``RANK`` and ``LOCAL_RANK``.  With
  ``WORLD_SIZE`` unset or one the environment is ignored, no group is created
  and the requested device is returned unchanged, so an ordinary run is
  untouched.  Otherwise the backend follows the device (``nccl`` on CUDA,
  ``gloo`` on CPU), every CUDA worker claims ``cuda:LOCAL_RANK`` because each
  rank needs its own card, and the timeout is finite so a dead rank fails the
  survivors instead of hanging them.  Calling this twice in one process reuses
  the existing group, which lets the command line resolve the device first.
  """
  requested = torch.device(device)
  world_size = _environment_int("WORLD_SIZE")
  rank = _environment_int("RANK") or 0
  local_rank = _environment_int("LOCAL_RANK") or 0
  if world_size is None or world_size <= 1:
    return DistributedContext(1, 0, 0, requested, None)
  if rank < 0 or local_rank < 0:
    raise FlowTrainerError("RANK and LOCAL_RANK must be non-negative")
  if rank >= world_size:
    raise FlowTrainerError(f"RANK {rank} is outside the WORLD_SIZE {world_size} group")
  if not is_available():
    raise FlowTrainerError("data parallelism needs torch.distributed")
  backend = "nccl" if requested.type == "cuda" else "gloo"
  if backend == "nccl":
    if not torch.cuda.is_available():
      raise FlowTrainerError(
        f"WORLD_SIZE {world_size} with a CUDA device needs CUDA, which is unavailable"
      )
    if local_rank >= torch.cuda.device_count():
      raise FlowTrainerError(
        f"LOCAL_RANK {local_rank} is outside the "
        f"{torch.cuda.device_count()} visible CUDA device(s)"
      )
    # Each rank owns one card: the module, its gradients and the metric tensors
    # all live on ``cuda:LOCAL_RANK`` rather than on the default device.
    torch.cuda.set_device(local_rank)
    requested = torch.device("cuda", local_rank)
  if is_initialized():
    live = get_world_size()
    if live != world_size:
      raise FlowTrainerError(
        f"the process group has world size {live}, but WORLD_SIZE is {world_size}"
      )
    rank = get_rank()
  else:
    init_process_group(backend=backend, timeout=timedelta(seconds=timeout_seconds))
    rank = get_rank()
  return DistributedContext(world_size, rank, local_rank, requested, backend)


def per_rank_microbatch(
  effective_batch_size: int, gradient_accumulation_steps: int, world_size: int
) -> int:
  """Return the microbatch one rank consumes to keep the global batch fixed.

  Data parallelism must not change the recipe: the global batch is
  ``microbatch * accumulation * world_size``.  A world size that does not divide
  the global batch, and a microbatch below one, are refused here rather than
  rounded, because a rounded batch silently moves the learning rate and warmup
  calibration of every comparison against the single-card run.
  """
  if world_size < 1:
    raise FlowTrainerError(f"world size must be positive, got {world_size}")
  if gradient_accumulation_steps < 1:
    raise FlowTrainerError(
      f"gradient_accumulation_steps must be positive, got {gradient_accumulation_steps}"
    )
  divisor = gradient_accumulation_steps * world_size
  if effective_batch_size % divisor:
    raise FlowTrainerError(
      f"effective_batch_size {effective_batch_size} is not divisible by "
      f"gradient_accumulation_steps * world_size = {divisor}, so the global batch "
      "cannot be held fixed"
    )
  microbatch = effective_batch_size // divisor
  if microbatch < 1:
    raise FlowTrainerError(
      f"world size {world_size} leaves a per-rank microbatch below one for "
      f"effective batch {effective_batch_size} and {gradient_accumulation_steps} "
      "accumulation steps"
    )
  return microbatch


def epoch_shard(
  *,
  dataset_size: int,
  sample_count: int,
  epoch: int,
  seed: int,
  rank: int,
  world_size: int,
) -> Tensor:
  """Return this rank's contiguous chunk of the one shared epoch permutation.

  Every rank builds the same permutation from ``seed + 2 + epoch``, with no rank
  in the seed, and takes a different contiguous chunk of it.  The per-rank
  shards are therefore disjoint and equal length without any communication, and
  no rank can enter an all-reduce an extra time.  Contiguous chunks of a
  permutation keep motions mixed; a stride of ``world_size`` would correlate a
  shard with cache order.  The ``sample_count % world_size`` tail is dropped, and
  it rotates between epochs because the permutation does, so no window is
  starved.  An epoch budget above the dataset length cycles the permutation.
  """
  if dataset_size < 1:
    raise FlowTrainerError("cannot shard a dataset that holds no windows")
  if world_size < 1 or rank < 0 or rank >= world_size:
    raise FlowTrainerError(f"rank {rank} is outside the world size {world_size} group")
  if sample_count < 1:
    raise FlowTrainerError("epoch sample count must be positive")
  permutation = torch.randperm(
    dataset_size, generator=torch.Generator().manual_seed(seed + 2 + epoch)
  )
  repeats = math.ceil(sample_count / dataset_size)
  stream = permutation.repeat(repeats) if repeats > 1 else permutation
  stream = stream[:sample_count]
  shard_size = sample_count // world_size
  if shard_size < 1:
    raise FlowTrainerError(
      f"epoch sample count {sample_count} is smaller than the world size "
      f"{world_size}; every rank needs at least one window"
    )
  start = rank * shard_size
  return stream[start : start + shard_size]


class _ShardSampler(Sampler[int]):
  """Yield window indices from one rank's epoch shard, never from another's.

  ``base`` samples positions inside the shard; every position is remapped
  through ``indices``, so a weighted draw cannot leave the shard and the
  per-rank streams stay disjoint.  With ``base=None`` the shard is yielded once
  in permutation order, which is the distributed form of the unweighted shuffle
  and keeps full per-epoch coverage.
  """

  def __init__(self, indices: Tensor, base: Sampler[int] | None = None) -> None:
    if indices.ndim != 1 or indices.dtype != torch.long:
      raise FlowTrainerError("epoch shard indices must be a 1-D int64 tensor")
    if (
      base is not None and isinstance(base, Sized) and len(base) != int(indices.numel())
    ):
      raise FlowTrainerError(
        "the shard sampler base length disagrees with its shard length"
      )
    self.indices = indices
    self.base = base

  def __iter__(self) -> Iterator[int]:
    if self.base is None:
      yield from (int(index) for index in self.indices.tolist())
      return
    for position in self.base:
      yield int(self.indices[position])

  def __len__(self) -> int:
    return int(self.indices.numel())


@contextmanager
def _metrics_rows(path: Path, *, enabled: bool) -> Iterator[Any]:
  """Open the metric row file on rank 0 and yield ``None`` on every other rank.

  Only rank 0 writes metric rows and TensorBoard events.  The other ranks still
  run every collective in the training loop, so a write gate cannot make the
  ranks diverge.
  """
  if not enabled:
    yield None
    return
  with path.open("a", encoding="utf-8") as handle:
    yield handle


def _finite_tensor(value: Tensor, name: str) -> None:
  if not torch.isfinite(value).all():
    raise FlowTrainerError(f"{name} contains NaN or Inf")


def _mean_tensor_or_zero(value: Tensor) -> Tensor:
  if value.numel() == 0:
    return torch.zeros((), device=value.device, dtype=torch.float32)
  return value.mean().float()


def _lr_horizon_from_environment() -> int | None:
  """Read MJLAB_FLOW_LR_HORIZON_UPDATES, or return None for the run budget."""
  raw = os.environ.get("MJLAB_FLOW_LR_HORIZON_UPDATES", "")
  if raw == "":
    return None
  try:
    value = int(raw)
  except ValueError as exc:
    raise FlowTrainerError(
      f"MJLAB_FLOW_LR_HORIZON_UPDATES must be an integer, got {raw!r}"
    ) from exc
  if value < 1:
    raise FlowTrainerError(
      f"MJLAB_FLOW_LR_HORIZON_UPDATES must be positive, got {value}"
    )
  return value


def _rng_state_tensor(value: Any) -> Tensor:
  """Return one serialized RNG state as the CPU ByteTensor the APIs expect.

  ``load_flow_checkpoint`` moves every tensor in the payload to the training
  device.  ``torch.set_rng_state`` and ``Generator.set_state`` accept only a
  CPU ByteTensor, so resuming a CUDA run used to fail with "resume RNG state
  cannot be restored" even though the saved state was intact.  CPU tests never
  caught it because there the move to the training device is a no-op.
  """
  tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
  if tensor.dtype != torch.uint8:
    tensor = tensor.to(torch.uint8)
  return tensor.detach().to("cpu").contiguous()


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
    # Rank resolution happens before anything claims a device: a distributed run
    # must set its own card and join its group before the parameters, the EMA and
    # the optimizer state land somewhere.
    self.distributed = initialize_distributed(device)
    self.world_size = self.distributed.world_size
    self.rank = self.distributed.rank
    self.local_rank = self.distributed.local_rank
    self.is_master = self.rank == 0
    self.microbatch_size = per_rank_microbatch(
      self.training.effective_batch_size,
      self.training.gradient_accumulation_steps,
      self.world_size,
    )
    self.module = model
    self.train_dataset = train_dataset
    self.eval_datasets = dict(eval_datasets)
    self.output_dir = Path(output_dir)
    self.output_dir.mkdir(parents=True, exist_ok=True)
    self.device = self.distributed.device
    if self.device.type == "cuda" and not torch.cuda.is_available():
      raise FlowTrainerError("CUDA device requested but CUDA is unavailable")
    self.diagnostics = self._build_diagnostics()
    self._lr_horizon_env = _lr_horizon_from_environment()
    self.module.to(self.device)
    # One owner for the parameter dtype.  A pure-bf16 run stores the model in
    # bf16, so the EMA shadow and the optimizer state are created from bf16
    # parameters rather than being cast afterwards.
    pure_dtype = denoiser_dtype(self.training.mixed_precision)
    if pure_dtype is not None:
      self.module.to(pure_dtype)
    # The forward input must follow the parameter dtype.  Under autocast the
    # region casts it; under bf16-pure nothing does, and a fp32 input against
    # bf16 weights raises "mat1 and mat2 must have the same dtype".
    self.model_dtype = next(self.module.parameters()).dtype
    # Data parallelism wraps the module and never the recipe: the EMA shadow,
    # the optimizer state and every checkpoint are built from the unwrapped
    # module, so a saved state keeps the single-process layout.
    self.model: nn.Module = (
      DistributedDataParallel(
        self.module,
        find_unused_parameters=False,
        device_ids=None if self.distributed.backend == "gloo" else [self.local_rank],
      )
      if self.distributed.is_distributed
      else self.module
    )
    self.ema = ExponentialMovingAverage(
      self.module,
      power=self.training.ema_power,
      max_decay=self.training.ema_max_decay,
    )
    self.optimizer = torch.optim.AdamW(
      self.module.parameters(),
      lr=self.training.learning_rate,
      weight_decay=self.training.weight_decay,
    )
    # bfloat16 autocast needs no loss scaler; we store None for resume parity.
    self.scaler: Any | None = None
    self.noise_generator = _generator(self.device, self.training.seed + 1)
    # The sampling stream is the one generator that must differ per rank: two
    # ranks drawing the same stream would make the same weighted decisions.  With
    # one rank this is exactly the historical ``seed + 2``.
    self.sampler_generator = torch.Generator(device="cpu").manual_seed(
      self.training.seed + 2 + self.rank
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
    self.model_description = model_identity(self.module)
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
          model=self.module,
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

  def _epoch_total_samples(self) -> int:
    """Return the number of windows one epoch draws across all ranks."""
    count = (
      self.training.epoch_sample_budget
      if self.training.epoch_sample_budget is not None
      else len(self.train_dataset)
    )
    if count <= 0:
      raise FlowTrainerError("training dataset or epoch_sample_budget is empty")
    return int(count)

  def _epoch_sample_count(self) -> int:
    """Return the number of windows this rank draws in one epoch.

    Every rank draws ``total // world_size``, so all ranks consume the same
    number of micro-batches and enter the same number of all-reduces.  Unequal
    per-rank counts hang the job inside a collective instead of failing, so
    equal length is a construction guarantee rather than an observed property.
    """
    count = self._epoch_total_samples() // self.world_size
    if count <= 0:
      raise FlowTrainerError(
        f"epoch sample count {self._epoch_total_samples()} is smaller than the "
        f"world size {self.world_size}; every rank needs at least one window"
      )
    return int(count)

  def _updates_for_samples(self, count: int) -> int:
    micro_batches = math.ceil(count / self.microbatch_size)
    return math.ceil(micro_batches / self.training.gradient_accumulation_steps)

  def _write_run_metadata(self) -> None:
    """Write the resolved configuration and README once, from rank 0.

    Every rank would otherwise truncate the same two files, so a distributed run
    can leave a half-written record behind.
    """
    if not self.is_master:
      return
    (self.output_dir / "resolved-config.json").write_text(
      json.dumps(
        {"config": self.config.as_dict(), "sha256": self.config_hash},
        sort_keys=True,
        indent=2,
      )
      + "\n",
      encoding="utf-8",
    )
    parallel = (
      f"- data parallelism: `world_size={self.world_size} rank={self.rank}`\n"
      f"- per-rank microbatch: `{self.microbatch_size}`\n"
      if self.distributed.is_distributed
      else ""
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
      f"{parallel}"
      "\nThis directory is a bounded engineering artifact; no hardware or quality\n"
      "claim is implied by its presence.\n",
      encoding="utf-8",
    )

  def _validate_resume(self, state: FlowCheckpointState) -> None:
    # A changed configuration is allowed: the trainer records the config it
    # actually ran with, and names the fields that moved.  Refusing here made
    # every precision or schedule branch impossible.
    changed = config_differences(dict(state.config), self.config.as_dict())
    if changed:
      print(
        f"resume note: {len(changed)} config field(s) differ from the "
        f"checkpoint: {', '.join(changed[:8])}",
        flush=True,
      )
    if state.family != self.process.as_dict()["family"]:
      raise FlowTrainerError("resume flow family mismatch")
    if state.process_identity != self.process_identity:
      raise FlowTrainerError("resume process identity mismatch")
    if state.data_identity != self.data_identity:
      raise FlowTrainerError("resume data identity mismatch")
    if dict(state.statistics_identity) != dict(self.statistics_identity):
      raise FlowTrainerError("resume statistics identity mismatch")
    # A dataset that moved storage is still the same dataset.  Byte-identical
    # data at a local path must resume, so this names the fields that moved
    # instead of refusing.  Already blocked one restart onto local NVMe.
    if dict(state.dataset_identity) != dict(self.identity):
      recorded = dict(state.dataset_identity)
      differing = sorted(
        key
        for key in set(recorded) | set(self.identity)
        if recorded.get(key) != self.identity.get(key)
      )
      print(
        f"resume note: dataset identity differs in {len(differing)} field(s): "
        f"{', '.join(differing[:8])}",
        flush=True,
      )
    if state.model_identity_hash != self.model_identity_hash:
      raise FlowTrainerError("resume model identity mismatch")

  def _rng_state(self) -> dict[str, Any]:
    """Return this process's RNG state, plus the world size it belongs to."""
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
      "world_size": self.world_size,
      "history_dropout_counters": {
        "windows": self._history_dropout_windows,
        "total": self._history_total_windows,
      },
      "state_mask_counters": {
        "windows": self._state_mask_windows,
        "total": self._state_mask_total_windows,
      },
    }

  def _checkpoint_rng_state(self) -> dict[str, Any]:
    """Return the RNG payload of one checkpoint, with one state per rank.

    Rank 0 writes the checkpoint, so a resume of a data-parallel run can only be
    exact if every rank's generators are in the file.  ``rank_generators`` holds
    one entry per rank in rank order, and each entry carries the ``world_size``
    of the group it came from, so a resume at another world size is refused
    instead of restoring a stream that no rank drew.
    """
    state = self._rng_state()
    if not self.distributed.is_distributed:
      return state
    table: list[Any] = [None] * self.world_size
    all_gather_object(table, self._rng_state())
    state["rank_generators"] = table
    return state

  def _restore_rng_state(self, state: Any) -> None:
    if not isinstance(state, Mapping):
      raise FlowTrainerError("resume RNG state is malformed")
    state = self._rank_rng_state(state)
    try:
      random.setstate(state["python"])
      np.random.set_state(state["numpy"])
      torch.set_rng_state(_rng_state_tensor(state["torch"]))
      if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
          [_rng_state_tensor(item) for item in state["torch_cuda"]]
        )
      self.noise_generator.set_state(_rng_state_tensor(state["noise_generator"]))
      self.sampler_generator.set_state(_rng_state_tensor(state["sampler_generator"]))
      if "state_mask_generator" in state:
        self.state_mask_generator.set_state(
          _rng_state_tensor(state["state_mask_generator"])
        )
      elif self._state_mask_channels and float(
        self.training.masked_state_probability
      ) not in (0.0, 1.0):
        raise FlowTrainerError(
          "checkpoint lacks the state-mask RNG state for a stochastic mask run"
        )
      if "history_dropout_generator" in state:
        self.history_dropout_generator.set_state(
          _rng_state_tensor(state["history_dropout_generator"])
        )
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

  def _rank_rng_state(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
    """Select this rank's entry and refuse a changed world size.

    A stream drawn by four ranks cannot be continued by one: the sampler draws
    per rank and the checkpoint holds one state per rank.  A checkpoint without
    the field predates data parallelism, so it describes a world size of one.
    """
    recorded = state.get("world_size", 1)
    try:
      stored = int(recorded)
    except (TypeError, ValueError) as exc:
      raise FlowTrainerError(
        f"resume checkpoint has a malformed world_size {recorded!r}"
      ) from exc
    if stored != self.world_size:
      raise FlowTrainerError(
        f"resume checkpoint was written by world_size {stored}, but this run has "
        f"world_size {self.world_size}; a data-parallel resume needs the same "
        "world size, because every rank owns one sampler and noise stream"
      )
    if not self.distributed.is_distributed:
      return state
    table = state.get("rank_generators")
    if not isinstance(table, (list, tuple)) or len(table) != self.world_size:
      raise FlowTrainerError(
        "resume checkpoint has no per-rank RNG table for its recorded world size"
      )
    entry = table[self.rank]
    if not isinstance(entry, Mapping):
      raise FlowTrainerError(f"resume RNG state for rank {self.rank} is malformed")
    return entry

  def _sampling_weights(self) -> np.ndarray | None:
    """Return per-window sampling weights, or ``None`` for the plain pass.

    ``None`` means "no weighting and no epoch budget", which is the historical
    shuffle path.  Under data parallelism the same condition selects the plain
    shard pass, so an unweighted run keeps full per-epoch coverage instead of
    resampling with replacement.
    """
    weights = _dataset_weights(self.train_dataset, self.training)
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
    return weights

  def _shard_sampler(self, epoch: int, weights: np.ndarray | None) -> Sampler[int]:
    """Return this rank's sampler over its chunk of the shared permutation.

    The chunk is disjoint from every other rank's chunk by construction, so a
    weighted draw can only ever return an index this rank owns.  Weighting stays
    available inside the shard: the sampler runs over shard positions and the
    produced position is remapped to the window it names.
    """
    shard = epoch_shard(
      dataset_size=len(self.train_dataset),
      sample_count=self._epoch_total_samples(),
      epoch=epoch,
      seed=self.training.seed,
      rank=self.rank,
      world_size=self.world_size,
    )
    if weights is None:
      return _ShardSampler(shard)
    local = weights[shard.numpy()]
    if not np.any(local > 0.0):
      raise FlowTrainerError(
        f"the epoch shard of rank {self.rank} holds no positive sampling weight"
      )
    return _ShardSampler(
      shard,
      WeightedRandomSampler(
        local.tolist(),
        num_samples=int(shard.numel()),
        replacement=True,
        generator=self.sampler_generator,
      ),
    )

  def _sampler(self, epoch: int) -> Sampler[int] | None:
    """Return one epoch's sampler for this rank.

    Without data parallelism this is the historical selection, unchanged: the
    unweighted case returns ``None`` so the loader shuffles, and anything else
    is one ``WeightedRandomSampler`` over the whole dataset.  With data
    parallelism the epoch is one permutation shared by every rank and each rank
    draws only from its own contiguous chunk.
    """
    weights = self._sampling_weights()
    if self.distributed.is_distributed:
      return self._shard_sampler(epoch, weights)
    if weights is None:
      return None
    return WeightedRandomSampler(
      weights.tolist(),
      num_samples=self._epoch_sample_count(),
      replacement=True,
      generator=self.sampler_generator,
    )

  @property
  def _lr_horizon(self) -> int:
    """Number of updates over which cosine decays to zero.

    It defaults to the run budget, so ordinary runs are unchanged.  An operator
    may shorten it with MJLAB_FLOW_LR_HORIZON_UPDATES to finish the anneal
    before the stop condition, which separates "the schedule annealed at the
    right time" from "the run simply ended early".
    """
    if self._lr_horizon_env is None:
      return int(self._resolved_updates)
    return min(self._lr_horizon_env, int(self._resolved_updates))

  def _build_diagnostics(self) -> FlowDiagnostics | None:
    """Return a sidecar recorder, or ``None`` when it is not requested.

    The recorder reads its own environment variable, so an ordinary run takes
    the ``None`` path and pays no cost.  Diagnostics stay out of the training
    config on purpose: the config hash is part of the resume identity, and a new
    field there would reject every existing checkpoint.  A distributed run is
    refused instead of silently recording per-rank batches: the replay contract
    reconstructs ONE batch from the recorded window indices in one process, and
    a per-rank subset would not replay.  ``WORLD_SIZE == 1`` is unaffected.
    """
    if self.distributed.is_distributed and os.environ.get(ENV_DIRECTORY):
      raise FlowTrainerError(
        f"{ENV_DIRECTORY} is not supported with data parallelism: the recorded "
        "batch would be one rank's subset, which the single-process replay "
        "cannot reconstruct; unset it or run with WORLD_SIZE=1"
      )
    state_dimension = int(self.contract.projected_state_dimension)
    layout = HYBRID_STATE.slices if state_dimension == HYBRID_STATE.dimension else {}
    return FlowDiagnostics.from_environment(
      state_dimension=state_dimension,
      token_dimension=int(self.contract.token_dimension),
      window_steps=int(self.contract.window_steps),
      level_max=int(self.process.max_level),
      groups=channel_groups(
        layout,
        state_dimension=state_dimension,
        token_dimension=int(self.contract.token_dimension),
      ),
    )

  def _loader(self, epoch: int) -> DataLoader[Any]:
    sampler = self._sampler(epoch)
    # Window reads are small and scattered in a multi-gigabyte cache.  On a
    # local NVMe a single in-process reader costs most of a step, so an operator
    # may overlap the reads with worker processes.  The default stays 0, which
    # keeps the historical behaviour bit-identical.
    workers = int(os.environ.get("MJLAB_FLOW_DATALOADER_WORKERS", "0"))
    if workers < 0:
      raise FlowTrainerError("MJLAB_FLOW_DATALOADER_WORKERS must be non-negative")
    return DataLoader(
      self.train_dataset,
      batch_size=self.microbatch_size,
      sampler=sampler,
      shuffle=sampler is None,
      generator=self.sampler_generator if sampler is None else None,
      num_workers=workers,
      persistent_workers=False,
      prefetch_factor=4 if workers else None,
      drop_last=False,
    )

  def _autocast(self) -> Any:
    # Only plain ``bf16`` casts inside the forward.  ``bf16-pure`` already holds
    # bf16 parameters and every operation runs in bf16, so an autocast region
    # would add a second policy on top of the same dtype.
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
    if self.diagnostics is not None:
      # The level draws already advanced this generator, so the noise state is
      # recorded here, immediately before the only draw the replay must repeat.
      self.diagnostics.note_noise_state(self.noise_generator)
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
    tokens = noised.tokens.to(self.model_dtype)
    if not self._fp32_checked:
      prediction = self.model(tokens, step_ids)
      _finite_tensor(prediction, "FP32 denoiser output")
      self._fp32_checked = True
    else:
      with self._autocast():
        prediction = self.model(tokens, step_ids)
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

  def _gradient_sync(self, finishes_update: bool) -> Any:
    """Return the context that decides whether this backward all-reduces.

    Gradients are reduced once per update, not once per micro-batch: every
    earlier micro-batch runs inside ``no_sync`` and accumulates into the local
    gradients, so the group pays one all-reduce for the whole update instead of
    ``gradient_accumulation_steps`` of them.  The forward belongs inside the same
    context, because the reducer reads the flag when it prepares the backward.
    """
    if finishes_update or not isinstance(self.model, DistributedDataParallel):
      return nullcontext()
    return self.model.no_sync()

  def _reduce_update_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
    """Average the per-rank update metrics so rank 0 logs the group mean.

    ``all_reduce(SUM)`` followed by a division by the world size makes the
    rank-0 row exactly the mean of the per-rank rows.  Only the micro-batch
    metrics are reduced here: the derived fields (learning rate, gradient norm,
    step, ...) are already identical on every rank, so averaging them again
    would multiply them by the world size.
    """
    if not self.distributed.is_distributed or not metrics:
      return metrics
    names = sorted(metrics)
    values = torch.tensor(
      [float(metrics[name]) for name in names],
      dtype=torch.float64,
      device=self.device,
    )
    all_reduce(values, op=ReduceOp.SUM)
    values /= self.world_size
    reduced = dict(metrics)
    for index, name in enumerate(names):
      reduced[name] = float(values[index])
    return reduced

  def _all_reduce_counts(self, values: Sequence[int]) -> list[int]:
    """Sum integer window counters across ranks; a no-op without a group."""
    if not self.distributed.is_distributed or not values:
      return [int(value) for value in values]
    tensor = torch.tensor(
      [int(value) for value in values], dtype=torch.int64, device=self.device
    )
    all_reduce(tensor, op=ReduceOp.SUM)
    return [int(value) for value in tensor.tolist()]

  def _broadcast_validation(
    self, loss: float | None, improved: bool
  ) -> tuple[float | None, bool]:
    """Broadcast rank 0's validation result so every rank selects alike.

    Only rank 0 evaluates, so only rank 0 holds a validation loss.  The other
    ranks adopt the number and the decision instead of re-deriving them, so
    every rank agrees on the best value before any checkpoint is written.
    """
    if not self.distributed.is_distributed:
      return loss, improved
    payload = torch.tensor(
      [float("nan") if loss is None else float(loss), 1.0 if improved else 0.0],
      dtype=torch.float64,
      device=self.device,
    )
    broadcast(payload, src=0)
    value = float(payload[0])
    return (None if math.isnan(value) else value), bool(payload[1])

  def _barrier(self) -> None:
    """Wait for every rank; a no-op without a process group."""
    if self.distributed.is_distributed:
      barrier()

  def _optimizer_update(
    self, pending: Sequence[Mapping[str, float]], *, update_index: int, epoch: int = 0
  ) -> dict[str, float]:
    if not pending:
      raise FlowTrainerError("optimizer update has no accumulated micro-batches")
    divisor = float(len(pending))
    for parameter in self.module.parameters():
      if parameter.grad is not None:
        parameter.grad.div_(divisor)
        _finite_tensor(parameter.grad, "gradient")
    # ``clip_grad_norm_`` returns the total norm measured BEFORE it rescales
    # the gradients, so this value also shows clipping: a value above
    # ``max_grad_norm`` means the step was scaled down.  The reduced gradient is
    # identical on every rank, so this norm is not averaged across ranks.
    gradient_norm = clip_gradient_norm(
      self.module.parameters(), self.training.max_grad_norm
    )
    _finite_tensor(torch.as_tensor(gradient_norm), "clipped gradient norm")
    learning_rate = warmup_cosine_learning_rate(
      update_index,
      total_updates=self._lr_horizon,
      learning_rate=self.training.learning_rate,
      warmup_updates=self.training.warmup_updates,
    )
    # The per-block norms must be read before ``zero_grad`` clears them.
    block_norms = (
      gradient_norms_by_block(self.module) if self.diagnostics is not None else None
    )
    # One number names this update everywhere it appears in the recorder: the
    # count of completed updates when its batches were consumed.  The trigger
    # checkpoint, the capture file and the stored window indices all use it, so
    # the replay offset stays unambiguous.
    step_before = self.global_step
    trigger = None
    if self.diagnostics is not None:
      mean_loss = float(
        sum(float(item["loss"]) for item in pending) / float(len(pending))
      )
      trigger = self.diagnostics.would_trigger(update_index=step_before, loss=mean_loss)
      if trigger is not None and self.diagnostics.checkpoint_allowed():
        # Saved BEFORE the step: this is the weights that produced the batch,
        # so the recorded loss can be replayed and checked against it.
        self._save(
          f"checkpoint-trigger-{step_before:07d}.pt",
          epoch=epoch,
          best_validation_loss=self.best_validation_loss,
        )
        self.diagnostics.checkpoint_written()
    for group in self.optimizer.param_groups:
      group["lr"] = learning_rate
    self.optimizer.step()
    for parameter in self.module.parameters():
      _finite_tensor(parameter, "model parameter")
    self.optimizer.zero_grad(set_to_none=True)
    decay = self.ema.update(self.module)
    self.global_step = update_index + 1
    # One all-reduce per update carries the metric means, so the rank-0 row is
    # the group mean and every rank keeps the same loss for the trigger logic.
    result = self._reduce_update_metrics(_average_metrics(pending))
    result["learning_rate"] = learning_rate
    result["ema_decay"] = decay
    result["accumulated_microbatches"] = float(len(pending))
    result["global_step"] = float(self.global_step)
    result["gradient_norm"] = float(gradient_norm)
    result["lr_horizon_updates"] = float(self._lr_horizon)
    self._last_train_loss = result["loss"]
    if self.diagnostics is not None:
      self.diagnostics.end_update(
        update_index=self.global_step,
        learning_rate=learning_rate,
        loss=result["loss"],
        gradient_norm=float(gradient_norm),
        model=self.module,
        blocks=block_norms,
      )
      if trigger is not None:
        self.diagnostics.dump(update_index=step_before, reason="loss_trigger")
    return result

  def _save(self, name: str, *, epoch: int, best_validation_loss: float | None) -> Path:
    """Write one checkpoint from rank 0 and hold every rank at a barrier.

    The payload stores ``model.module``, so a data-parallel checkpoint is the
    unwrapped module and a single-process resume still loads it.  Every rank
    calls this and waits: a rank that continued would reach the next all-reduce
    while rank 0 is still serialising the state and the RNG table.
    """
    rng_state = self._checkpoint_rng_state()
    path = self.output_dir / name
    if self.is_master:
      save_flow_checkpoint(
        path,
        model=self.module,
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
        rng_state=rng_state,
        best_metric_split=self.best_metric_split,
        selection_split=self.best_metric_split,
        fp32_checked=self._fp32_checked,
      )
    self._barrier()
    return path

  def train(self) -> FlowTrainResult:
    """Run the resolved bounded update budget and atomically save artifacts.

    Every rank runs the same loop and enters the same collectives, so no rank
    can wait in an all-reduce for a batch another rank never sends.  Only rank 0
    writes ``metrics.jsonl``, TensorBoard rows, checkpoints and the run report;
    the other ranks adopt the reduced metrics and the broadcast validation
    result, which keeps ``global_step``, the best value and the RNG streams in
    agreement on every rank.
    """
    metrics_path = self.output_dir / "metrics.jsonl"
    checkpoint_paths: dict[str, Path] = {}
    epochs_completed = self.start_epoch
    mode_before = self.model.training
    self.model.train(True)
    tensorboard = open_for(self.output_dir) if self.is_master else None
    with _metrics_rows(metrics_path, enabled=self.is_master) as metrics_file:

      def emit(row: Mapping[str, Any]) -> None:
        """Append a metric row and mirror it to TensorBoard from rank 0."""
        if metrics_file is None:
          return
        metrics_file.write(json.dumps(row, sort_keys=True) + "\n")
        metrics_file.flush()
        if tensorboard is not None:
          tensorboard.write_row(row, fallback_step=self.global_step)

      stop = False
      epoch = self.start_epoch
      while self.global_step < self._resolved_updates and epoch < self.training.epochs:
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
          clean = _as_tokens(
            batch,
            window_steps=self.contract.window_steps,
            token_dimension=self.contract.token_dimension,
          )
          if self.diagnostics is not None:
            indices = batch[2] if len(batch) > 2 else None
            self.diagnostics.begin_batch(
              update_index=self.global_step, window_indices=indices
            )
          is_last_batch = batch_index + 1 == len(loader)
          # Every rank computes the same branch because the loader length and
          # the accumulation depth are equal on every rank.
          finishes_update = (
            len(pending) + 1 >= self.training.gradient_accumulation_steps
            or is_last_batch
          )
          with self._gradient_sync(finishes_update):
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
            if self.diagnostics is not None:
              self.diagnostics.observe_batch(
                prediction=prediction,
                target=target,
                state_levels=state_levels,
                latent_levels=latent_levels,
                history_dropout=history_dropout,
                state_mask=state_mask_flags,
              )
            pending.append(metrics)
            loss.backward()
          if finishes_update:
            update = self._optimizer_update(
              pending, update_index=self.global_step, epoch=epoch
            )
            epoch_updates.append(update)
            emit({"kind": "update", "epoch": epoch, **update})
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
          update = self._optimizer_update(
            pending, update_index=self.global_step, epoch=epoch
          )
          epoch_updates.append(update)
          emit({"kind": "update", "epoch": epoch, **update})
        if stop:
          break
        epochs_completed = epoch + 1
        # The epoch row is written by rank 0 alone, so its window counters are
        # summed across ranks; the fraction stays a fraction because both sides
        # are summed.
        (
          epoch_dropout_windows,
          epoch_total_windows,
        ) = self._all_reduce_counts((epoch_dropout_windows, epoch_total_windows))
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
        validation_loss: float | None = None
        improved = False
        if eval_split in self.eval_datasets and self.is_master:
          epoch_metrics["eval_split"] = eval_split
          epoch_metrics["eval_model"] = self.evaluate(eval_split, parameters="model")
          ema_values = self.evaluate(eval_split, parameters="ema")
          epoch_metrics["eval_ema"] = ema_values
          if ema_values.get("windows", 0.0) > 0.0:
            validation_loss = ema_values.get("loss")
          if validation_loss is not None and math.isfinite(validation_loss):
            improved = (
              self.best_validation_loss is None
              or validation_loss < self.best_validation_loss
            )
        # Every rank must hold the same best value and the same decision before
        # any save: a rank that kept a local value would write a different
        # selection state into the next checkpoint.
        validation_loss, improved = self._broadcast_validation(
          validation_loss, improved
        )
        if improved and validation_loss is not None:
          self.best_validation_loss = validation_loss
          self.best_metric_split = eval_split
          checkpoint_paths["best"] = self._save(
            "checkpoint-best.pt",
            epoch=epochs_completed,
            best_validation_loss=self.best_validation_loss,
          )
        emit(epoch_metrics)
        if metrics_file is not None:
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
    if tensorboard is not None:
      tensorboard.close()
    # The report is written by rank 0 alone, so its window counters cover the
    # group; the two derived fractions stay fractions because both sides move.
    (
      self._history_dropout_windows,
      self._history_total_windows,
      self._state_mask_windows,
      self._state_mask_total_windows,
    ) = self._all_reduce_counts(
      (
        self._history_dropout_windows,
        self._history_total_windows,
        self._state_mask_windows,
        self._state_mask_total_windows,
      )
    )
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
    if self.distributed.is_distributed:
      summary["world_size"] = self.world_size
      summary["per_rank_microbatch"] = self.microbatch_size
    if self.is_master:
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
      name: value.detach().clone() for name, value in self.module.state_dict().items()
    }
    # Evaluation runs on the unwrapped module: rank 0 alone evaluates, so the
    # forward must not enter the reducer, and the backup is restored by name.
    mode_before = self.module.training
    if parameters == "ema":
      self.ema.copy_to(self.module)
    self.module.eval()
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
    self.module.load_state_dict(backup, strict=True)
    self.module.train(mode_before)
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
    if self.diagnostics is not None:
      # The level draws already advanced this generator, so the noise state is
      # recorded here, immediately before the only draw the replay must repeat.
      self.diagnostics.note_noise_state(self.noise_generator)
    noised = add_linear_noise(
      clean[..., : self.contract.projected_state_dimension],
      clean[..., self.contract.projected_state_dimension :],
      state_levels=levels.state.to(self.device),
      latent_levels=levels.latent.to(self.device),
      process=self.process,
      generator=generator,
    )
    step_ids = torch.stack((noised.state_levels, noised.latent_levels), dim=-1)
    prediction = self.module(noised.tokens.to(self.model_dtype), step_ids)
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
