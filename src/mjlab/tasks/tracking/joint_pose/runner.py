"""Bounded x0 flow-matching runner for the latent-free joint-pose dataset.

The window is ``[41, 68]``.  Rows ``0:8`` are clean conditioning history and
rows ``8:41`` are generated.  Training noises only the generated rows at one
integer level per window (``s = k / max_level``), asks the denoiser for clean
``x0`` and fits a channel-mean MSE on those rows, so the history rows never
contribute to the loss.  With probability ``history_dropout_probability`` the
whole history of a window is replaced by pure Gaussian, which is the same mask
semantics the diffusion and flow trainers use.

``LinearNoiseProcess``, ``FlowGrid``, ``euler_x0_update``,
``ExponentialMovingAverage``, ``warmup_cosine_learning_rate`` and the history
mask helper are imported from the diffusion and flow-matching modules.  Nothing
in those modules is edited and no contract is constructed.

All training randomness comes from dedicated generators: the window index
generator, the level and noise generator, the history-mask generator and the
sampling generator.  Their states are stored in every checkpoint, so a resumed
run continues the same draw sequence.  No global RNG seed is set.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from ..diffusion.trainer import (
  ExponentialMovingAverage,
  _generator,
  _history_dropout_mask,
)
from ..diffusion.training_config import warmup_cosine_learning_rate
from ..flow_matching.process import LinearNoiseProcess, draw_gaussian
from ..flow_matching.sampler import FlowGrid, euler_x0_update
from .config import FlowMatchingTrainerCfg, JointPoseDatasetCfg
from .dataset import JointPoseWindowDataset
from .model import build_denoiser

RESUME_CRITICAL_FIELDS = (
    "micro_batch",
    "gradient_accumulation_steps",
    "effective_batch",
    "learning_rate",
    "weight_decay",
    "warmup_updates",
    "max_grad_norm",
    "scheduler",
    "mixed_precision",
    "loss_reduction",
    "ema_power",
    "ema_max_decay",
    "max_level",
    "history_dropout_probability",
    "final_norm",
    "seed",
    "max_steps",
)
"""Config fields a resume must match, because they change the optimisation.

``max_steps`` is here because the warmup-plus-cosine schedule is planned over the
whole budget: the learning rate at step ``i`` depends on the budget, so resuming
under a different budget would silently re-plan the curve.  Extend a run with a
new configuration instead.
"""

CHECKPOINT_FORMAT = "joint_pose_flow_checkpoint_v1"


class JointPoseFlowRunnerError(RuntimeError):
  """The joint-pose flow runner state is invalid."""


def _finite(value: Tensor, name: str) -> None:
  if not bool(torch.isfinite(value).all()):
    raise JointPoseFlowRunnerError(f"{name} contains NaN or Inf")


class FlowMatchingRunner:
  """Train, checkpoint and sample the joint-pose x0 flow model.

  ``run_dir`` selects the artifact directory.  The default is
  ``<log_root>/<experiment_name>/<timestamp>``.  Passing a directory that
  already holds ``params/`` resumes that run: the recorded dataset identity must
  match the current dataset, and the highest ``model_<step>.pt`` is loaded.
  """

  def __init__(
    self,
    dataset: JointPoseWindowDataset,
    eval_datasets: Mapping[str, JointPoseWindowDataset],
    cfg: FlowMatchingTrainerCfg,
    device: str,
    *,
    run_dir: Path | str | None = None,
    width: int | None = None,
    layers: int | None = None,
    heads: int | None = None,
  ) -> None:
    if not isinstance(cfg, FlowMatchingTrainerCfg):
      raise TypeError("cfg must be a FlowMatchingTrainerCfg")
    if cfg.mixed_precision == "fp16":
      raise JointPoseFlowRunnerError(
        "fp16 autocast is not implemented; use mixed_precision bf16 or none"
      )
    self.cfg = cfg
    self.dataset = dataset
    self.dataset_cfg: JointPoseDatasetCfg = dataset.cfg
    self.eval_datasets = dict(eval_datasets)
    self.device = torch.device(device)
    if self.device.type == "cuda" and not torch.cuda.is_available():
      raise JointPoseFlowRunnerError("CUDA device requested but CUDA is unavailable")
    self.window_steps = self.dataset_cfg.window_steps
    self.past_steps = self.dataset_cfg.past_steps
    self.channels = self.dataset_cfg.channels
    for split, other in self.eval_datasets.items():
      if (other.cfg.window_steps, other.cfg.channels) != (
        self.window_steps,
        self.channels,
      ):
        raise JointPoseFlowRunnerError(
          f"evaluation dataset {split!r} does not match the training window "
          f"[{self.window_steps}, {self.channels}]"
        )
    self.process = LinearNoiseProcess(cfg.max_level)
    self.model = build_denoiser(
      self.dataset_cfg,
      max_level=cfg.max_level,
      width=width,
      layers=layers,
      heads=heads,
      final_norm=cfg.final_norm,
      init_seed=cfg.init_seed,
    ).to(self.device)
    self.ema = ExponentialMovingAverage(
      self.model, power=cfg.ema_power, max_decay=cfg.ema_max_decay
    )
    self.optimizer = torch.optim.AdamW(
      self.model.parameters(),
      lr=cfg.learning_rate,
      weight_decay=cfg.weight_decay,
    )
    self.dataset_identity = dataset.dataset_identity
    self.global_step = 0
    self._index_generator = torch.Generator(device="cpu").manual_seed(cfg.seed + 1)
    self._noise_generator = _generator(self.device, cfg.seed + 2)
    self._history_generator = _generator(self.device, cfg.seed + 3)
    self._sample_generator = _generator(self.device, cfg.seed + 4)
    self._history_dropout_windows = 0
    self._history_total_windows = 0
    self.run_dir = (
      Path(run_dir)
      if run_dir is not None
      else self._fresh_run_dir(cfg.log_root, cfg.experiment_name)
    )
    self._prepare_run_dir()

  # -------------------------------------------------------------- artifacts

  @staticmethod
  def _fresh_run_dir(log_root: str, experiment_name: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = Path(log_root).expanduser() / experiment_name
    candidate = base / stamp
    index = 1
    while candidate.exists():
      candidate = base / f"{stamp}-{index}"
      index += 1
    return candidate

  def _prepare_run_dir(self) -> None:
    """Write the run parameters, or validate them and resume an existing run."""
    params_dir = self.run_dir / "params"
    dataset_path = params_dir / "dataset.json"
    if dataset_path.is_file():
      recorded = json.loads(dataset_path.read_text(encoding="utf-8"))
      if recorded != self.dataset_identity:
        raise JointPoseFlowRunnerError(
          f"{self.run_dir} was started on another dataset; refusing to resume"
        )
      checkpoint = self._latest_checkpoint()
      if checkpoint is not None:
        self.load(checkpoint)
      return
    params_dir.mkdir(parents=True, exist_ok=True)
    (params_dir / "dataset.json").write_text(
      json.dumps(self.dataset_identity, sort_keys=True, indent=2) + "\n",
      encoding="utf-8",
    )
    (params_dir / "trainer.json").write_text(
      json.dumps(self.cfg.as_dict(), sort_keys=True, indent=2) + "\n",
      encoding="utf-8",
    )
    (params_dir / "pointer.txt").write_text(
      f"{self.run_dir.resolve()}\n", encoding="utf-8"
    )

  def _latest_checkpoint(self) -> Path | None:
    latest: tuple[int, Path] | None = None
    for path in self.run_dir.glob("model_*.pt"):
      step_text = path.stem.removeprefix("model_")
      if not step_text.isdigit():
        continue
      step = int(step_text)
      if latest is None or step > latest[0]:
        latest = (step, path)
    return None if latest is None else latest[1]

  def rng_state(self) -> dict[str, Any]:
    """Return the generator states and counters stored in a checkpoint."""
    return {
      "index_generator": self._index_generator.get_state(),
      "noise_generator": self._noise_generator.get_state(),
      "history_generator": self._history_generator.get_state(),
      "sample_generator": self._sample_generator.get_state(),
      "history_dropout_windows": self._history_dropout_windows,
      "history_total_windows": self._history_total_windows,
    }

  def _restore_rng_state(self, state: Any) -> None:
    if not isinstance(state, Mapping):
      raise JointPoseFlowRunnerError("checkpoint RNG state is missing or malformed")
    generators = (
      ("index_generator", self._index_generator),
      ("noise_generator", self._noise_generator),
      ("history_generator", self._history_generator),
      ("sample_generator", self._sample_generator),
    )
    try:
      for name, generator in generators:
        value = state[name]
        generator.set_state(value.cpu() if torch.is_tensor(value) else value)
    except (KeyError, RuntimeError, TypeError) as exc:
      raise JointPoseFlowRunnerError("checkpoint RNG state cannot be restored") from exc
    self._history_dropout_windows = int(state.get("history_dropout_windows", 0))
    self._history_total_windows = int(state.get("history_total_windows", 0))

  def save(self, path: Path | str) -> None:
    """Write model, EMA, optimizer, RNG state and identities to ``path``."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
      "format": CHECKPOINT_FORMAT,
      "global_step": self.global_step,
      "model": self.model.state_dict(),
      "ema": self.ema.state_dict(),
      "optimizer": self.optimizer.state_dict(),
      "rng": self.rng_state(),
      "trainer": self.cfg.as_dict(),
      "dataset_cfg": self.dataset_cfg.as_dict(),
      "dataset_identity": self.dataset_identity,
      "process": self.process.as_dict(),
      "model_settings": asdict(self.model.settings),
    }
    temporary = destination.with_name(f"{destination.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)

  def load(self, path: Path | str) -> None:
    """Restore a checkpoint written by :meth:`save`."""
    source = Path(path)
    try:
      payload = torch.load(source, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, EOFError, ValueError) as exc:
      raise JointPoseFlowRunnerError(
        f"could not read joint-pose flow checkpoint {source}"
      ) from exc
    if not isinstance(payload, Mapping) or payload.get("format") != CHECKPOINT_FORMAT:
      raise JointPoseFlowRunnerError(f"unsupported joint-pose flow checkpoint {source}")
    if payload.get("dataset_identity") != self.dataset_identity:
      raise JointPoseFlowRunnerError(
        f"{source} was trained on another dataset; refusing to load it"
      )
    recorded = payload.get("trainer")
    if isinstance(recorded, Mapping):
      current = self.cfg.as_dict()
      # Only fields that change the optimisation maths are compared.  Checkpoint
      # and logging cadence are operational and may differ on a resume.
      for key in RESUME_CRITICAL_FIELDS:
        if recorded.get(key) != current.get(key):
          raise JointPoseFlowRunnerError(
            f"{source} was trained with {key}={recorded.get(key)!r}, "
            f"current run has {current.get(key)!r}; refusing to resume"
          )

    self.model.load_state_dict(payload["model"], strict=True)
    self.ema.load_state_dict(payload["ema"])
    self.optimizer.load_state_dict(payload["optimizer"])
    self._restore_rng_state(payload.get("rng"))
    self.global_step = int(payload["global_step"])

  # -------------------------------------------------------------- training

  def _autocast(self) -> Any:
    if self.cfg.mixed_precision == "bf16" and self.device.type == "cuda":
      return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()

  def _learning_rate(self, update_index: int, budget: int) -> float:
    rate = float(self.cfg.learning_rate)
    warmup = int(self.cfg.warmup_updates)
    if self.cfg.scheduler == "cosine":
      # budget + 1 keeps progress below 1 on the last update; the frozen helper
      # reaches cos(pi) = 0 when update_index == total_updates - 1, which would
      # spend a final update drawing windows and moving the EMA for no step.
      return warmup_cosine_learning_rate(
        update_index,
        total_updates=budget + 1,
        learning_rate=rate,
        warmup_updates=warmup,
      )
    completed = min(update_index + 1, budget + 1)
    if warmup > 0 and completed <= warmup:
      return rate * completed / warmup
    return rate

  def _noised_batch(self, clean: Tensor) -> tuple[Tensor, Tensor]:
    """Build the model input and its per-row integer level ids.

    Rows ``0:past_steps`` carry level 0, or ``max_level`` on the windows the
    history mask drops, because those rows then hold pure Gaussian.  Rows
    ``past_steps:window_steps`` carry the one level sampled for the window.
    Both level-embedding slots receive the same level: the joint-pose window is
    a single state slice and has no latent slice.
    """
    batch = clean.shape[0]
    future = self.window_steps - self.past_steps
    generator_device = getattr(self._noise_generator, "device", self.device)
    levels = torch.randint(
      0,
      self.process.embedding_count,
      (batch,),
      generator=self._noise_generator,
      device=generator_device,
      dtype=torch.long,
    ).to(self.device)
    fraction = self.process.noise_fractions_torch(
      levels, dtype=torch.float32, device=self.device
    )
    noise = draw_gaussian(
      (batch, future, self.channels),
      generator=self._noise_generator,
      device=self.device,
      dtype=torch.float32,
    )
    noised_future = (1.0 - fraction)[:, None, None] * clean[:, self.past_steps :]
    noised_future = noised_future + fraction[:, None, None] * noise

    history = clean[:, : self.past_steps]
    dropout = _history_dropout_mask(
      batch,
      float(self.cfg.history_dropout_probability),
      generator=self._history_generator,
      device=self.device,
    )
    if float(self.cfg.history_dropout_probability) > 0.0:
      history_noise = draw_gaussian(
        history.shape,
        generator=self._history_generator,
        device=self.device,
        dtype=torch.float32,
      )
      history = torch.where(dropout[:, None, None], history_noise, history)
    self._history_dropout_windows += int(dropout.sum().item())
    self._history_total_windows += int(dropout.numel())

    history_levels = torch.where(
      dropout,
      torch.full_like(levels, self.process.max_level),
      torch.zeros_like(levels),
    )
    row_levels = torch.cat(
      (
        history_levels[:, None].expand(-1, self.past_steps),
        levels[:, None].expand(-1, future),
      ),
      dim=1,
    )
    step_ids = torch.stack((row_levels, row_levels), dim=-1)
    return torch.cat((history, noised_future), dim=1), step_ids

  def _microbatch_loss(self) -> Tensor:
    """Channel-mean x0 MSE on the generated rows of one drawn microbatch."""
    indices = self.dataset.draw_indices(self.cfg.micro_batch, self._index_generator)
    # Gather in ascending order.  A microbatch is a mean over its rows, so the
    # order inside it is irrelevant, but the storage read pattern is not: 512
    # windows drawn at random from a 356 GB memory map cost a page fault each and
    # defeat readahead, which measured 2.7 MB/s on the shared NAS against 74 MB/s
    # for a sequential read of the same file.  Sorting turns the batch into a few
    # ascending runs and lets the page cache work.
    indices = torch.sort(indices).values
    clean = torch.stack([self.dataset[int(index)] for index in indices.tolist()])
    clean = clean.to(device=self.device, dtype=torch.float32)
    _finite(clean, "training window")
    tokens, step_ids = self._noised_batch(clean)
    with self._autocast():
      prediction = self.model(tokens, step_ids)
    # Every generated row has the full channel width, so the channel mean and
    # the token mean of this region are the same number.
    error = prediction.float()[:, self.past_steps :] - clean[:, self.past_steps :]
    loss = error.square().mean()
    _finite(loss, "loss")
    return loss

  def learn(self) -> dict[str, object]:
    """Run the update budget and return the final metrics summary."""
    budget = self.cfg.require_budget()
    if self.global_step >= budget:
      raise JointPoseFlowRunnerError(
        f"the update budget {budget} is already spent at step {self.global_step}"
      )
    accumulation = int(self.cfg.gradient_accumulation_steps)
    metrics_path = self.run_dir / "metrics.jsonl"
    started = time.monotonic()
    mode_before = self.model.training
    self.model.train(True)
    latest: dict[str, object] = {}
    try:
      with metrics_path.open("a", encoding="utf-8") as metrics_file:
        for update_index in range(self.global_step, budget):
          learning_rate = self._learning_rate(update_index, budget)
          for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
          self.optimizer.zero_grad(set_to_none=True)
          losses: list[float] = []
          for _ in range(accumulation):
            loss = self._microbatch_loss()
            loss.backward()
            losses.append(float(loss.detach()))
          divisor = float(len(losses))
          for parameter in self.model.parameters():
            if parameter.grad is not None:
              parameter.grad.div_(divisor)
          gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.max_grad_norm
          )
          _finite(torch.as_tensor(gradient_norm), "clipped gradient norm")
          self.optimizer.step()
          for name, parameter in self.model.named_parameters():
            _finite(parameter.detach(), f"model parameter {name}")
          ema_decay = self.ema.update(self.model)
          self.global_step = update_index + 1
          latest = {
            "kind": "update",
            "step": self.global_step,
            "loss": sum(losses) / len(losses),
            "learning_rate": learning_rate,
            "gradient_norm": float(gradient_norm),
            "ema_decay": ema_decay,
            "accumulated_microbatches": float(divisor),
            "history_dropout_fraction": (
              self._history_dropout_windows / self._history_total_windows
              if self._history_total_windows
              else 0.0
            ),
          }
          every = int(self.cfg.log_every)
          if self.global_step % every == 0 or self.global_step == budget:
            metrics_file.write(json.dumps(latest, sort_keys=True) + "\n")
            metrics_file.flush()
          checkpoint_every = int(self.cfg.checkpoint_every)
          if self.global_step % checkpoint_every == 0 or self.global_step == budget:
            self.save(self.run_dir / f"model_{self.global_step:07d}.pt")
        summary: dict[str, object] = {
          "kind": "summary",
          "run_dir": str(self.run_dir),
          "global_step": self.global_step,
          "max_steps": budget,
          "final_loss": latest.get("loss"),
          "final_learning_rate": latest.get("learning_rate"),
          "final_gradient_norm": latest.get("gradient_norm"),
          "ema_updates": self.ema.updates,
          "history_dropout_fraction": latest.get("history_dropout_fraction", 0.0),
          "channels": self.channels,
          "window_steps": self.window_steps,
          "past_steps": self.past_steps,
          "effective_batch": self.cfg.effective_batch,
          "dataset_identity": self.dataset_identity,
          "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        metrics_file.write(json.dumps(summary, sort_keys=True) + "\n")
        metrics_file.flush()
    finally:
      self.model.train(mode_before)
    return summary

  # -------------------------------------------------------------- sampling

  def sample(self, history: Tensor, steps: int, *, parameters: str = "model") -> Tensor:
    """Generate ``[B, window_steps, channels]`` from clean history.

    ``history`` is ``[past_steps, channels]`` or ``[B, past_steps, channels]``.
    The generated rows start as pure Gaussian at ``max_level`` and descend the
    uniform integer grid in ``steps`` network calls.  The history rows are kept
    at level 0 and are reproduced unchanged in the result.

    Sampling runs in fp32 even when training used bf16 autocast, so an evaluation
    measures a slightly different function from the one trained.  ``parameters``
    selects the live weights or their exponential moving average.
    """
    if not isinstance(history, Tensor):
      raise TypeError("history must be a torch tensor")
    if parameters not in ("model", "ema"):
      raise JointPoseFlowRunnerError("parameters must be model or ema")
    if steps < 1:
      raise JointPoseFlowRunnerError("steps must be positive")
    clean = history.to(device=self.device, dtype=torch.float32)
    squeezed = clean.ndim == 2
    if squeezed:
      clean = clean.unsqueeze(0)
    if clean.ndim != 3 or tuple(clean.shape[1:]) != (self.past_steps, self.channels):
      raise JointPoseFlowRunnerError(
        f"history must be [{self.past_steps}, {self.channels}] or "
        f"[B, {self.past_steps}, {self.channels}], got {tuple(history.shape)}"
      )
    _finite(clean, "history")
    batch = clean.shape[0]
    future = self.window_steps - self.past_steps
    grid = FlowGrid.uniform(calls=steps, max_level=self.process.max_level)
    mode_before = self.model.training
    restored: dict[str, Tensor] | None = None
    if parameters == "ema":
      restored = {
        name: value.detach().clone() for name, value in self.model.state_dict().items()
      }
      self.ema.copy_to(self.model)
    self.model.eval()
    sample = draw_gaussian(
      (batch, future, self.channels),
      generator=self._sample_generator,
      device=self.device,
      dtype=torch.float32,
    )
    history_levels = torch.zeros(
      (batch, self.past_steps), dtype=torch.long, device=self.device
    )
    try:
      with torch.no_grad():
        for source, destination in zip(
          grid.source_levels, grid.destination_levels, strict=True
        ):
          future_levels = torch.full(
            (batch, future), int(source), dtype=torch.long, device=self.device
          )
          row_levels = torch.cat((history_levels, future_levels), dim=1)
          step_ids = torch.stack((row_levels, row_levels), dim=-1)
          x0_hat = self.model(torch.cat((clean, sample), dim=1), step_ids).float()
          sample = euler_x0_update(
            sample,
            x0_hat[:, self.past_steps :],
            source_levels=int(source),
            destination_levels=int(destination),
            process=self.process,
          )
    finally:
      if restored is not None:
        self.model.load_state_dict(restored, strict=True)
      self.model.train(mode_before)
    result = torch.cat((clean, sample), dim=1)
    return result[0] if squeezed else result


__all__ = [
  "CHECKPOINT_FORMAT",
  "FlowMatchingRunner",
  "JointPoseFlowRunnerError",
]
