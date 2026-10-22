"""Configuration for latent-free joint-pose flow matching.

Two dataclasses with named default factories, in the shape the PPO tasks use:
the dataset is described by one object, the optimisation by another, and neither
carries a schema hash.  Identity is a record written next to the checkpoint.

Epoch means one full pass over the training windows.  A run is bounded by
``max_steps`` (optimiser updates), never by a fraction of an epoch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CHANNEL_LAYOUT = (
    ("root_rot6d", 0, 6),
    ("joint_positions", 6, 37),
    ("joint_velocities", 37, 68),
)


class JointPoseFlowConfigError(ValueError):
  """A joint-pose flow configuration is invalid."""


@dataclass(frozen=True)
class JointPoseDatasetCfg:
  """One prepared dataset directory, with its optional reference source.

  ``reference_weight`` is a sampling weight, not a loss weight: a reference
  window is drawn that many times as often as an achieved window.  Every window
  inside a source keeps weight 1.0, so the failure ledger does not reweight
  anything.
  """

  data_dir: str = "/home/fushan/data/mjlab-latent-free-144/data"
  window_steps: int = 41
  past_steps: int = 8
  channels: int = 68
  split: str = "train"
  achieved_dir: str = "offline"
  reference_dir: str = "reference"
  include_reference: bool = True
  reference_weight: float = 30.0

  def __post_init__(self) -> None:
    if self.past_steps < 0 or self.window_steps - self.past_steps - 1 <= 0:
      raise JointPoseFlowConfigError(
        f"window_steps {self.window_steps} leaves no generated rows after "
        f"past_steps {self.past_steps}"
      )
    declared = sum(stop - start for _, start, stop in CHANNEL_LAYOUT)
    if declared != self.channels:
      raise JointPoseFlowConfigError(
        f"channel layout declares {declared} channels but the dataset has "
        f"{self.channels}"
      )
    if CHANNEL_LAYOUT[-1][2] != self.channels:
      raise JointPoseFlowConfigError("channel layout must end at the last channel")
    if self.split not in ("train", "validation", "test"):
      raise JointPoseFlowConfigError(f"unknown split {self.split!r}")
    if not self.include_reference and self.reference_weight != 1.0:
      raise JointPoseFlowConfigError(
        "reference_weight only applies when include_reference is true"
      )
    if self.reference_weight <= 0.0:
      raise JointPoseFlowConfigError("reference_weight must be positive")

  @property
  def future_steps(self) -> int:
    """Rows the model generates: the current row plus every later row.

    The trainer conditions on ``[0, past_steps)`` and generates
    ``[past_steps, window_steps)``, which is ``window_steps - past_steps`` rows.
    """
    return self.window_steps - self.past_steps

  def as_dict(self) -> dict[str, object]:
    return {
        "data_dir": self.data_dir,
        "window_steps": self.window_steps,
        "past_steps": self.past_steps,
        "future_steps": self.future_steps,
        "channels": self.channels,
        "channel_layout": [list(item) for item in CHANNEL_LAYOUT],
        "split": self.split,
        "achieved_dir": self.achieved_dir,
        "reference_dir": self.reference_dir,
        "include_reference": self.include_reference,
        "reference_weight": self.reference_weight,
    }


@dataclass(frozen=True)
class FlowMatchingTrainerCfg:
  """Optimisation surface for the joint-pose flow model.

  ``history_dropout_probability`` keeps the meaning it has in the diffusion and
  flow families: with that probability, a window's whole history pair is raised
  to pure Gaussian so the model cannot read the true history.  At zero the model
  conditions on clean history, which is the deployed planner case.
  """

  experiment_name: str = "x2_joint_pose_flow"
  log_root: str = "logs/flow_matching"
  seed: int = 0

  micro_batch: int = 128
  gradient_accumulation_steps: int = 4
  max_steps: int | None = None

  learning_rate: float = 1.0e-4
  weight_decay: float = 1.0e-3
  warmup_updates: int = 2000
  max_grad_norm: float = 1.0
  scheduler: str = "cosine"
  mixed_precision: str = "bf16"
  loss_reduction: str = "channel_mean"

  ema_power: float = 0.75
  ema_max_decay: float = 0.9999

  max_level: int = 1000
  history_dropout_probability: float = 0.0

  checkpoint_every: int = 10000
  log_every: int = 100

  def __post_init__(self) -> None:
    if self.max_steps is not None and self.max_steps <= 0:
      raise JointPoseFlowConfigError("max_steps must be positive")
    for name in (
        "micro_batch",
        "gradient_accumulation_steps",
        "warmup_updates",
        "max_level",
        "checkpoint_every",
        "log_every",
    ):
      if getattr(self, name) <= 0:
        raise JointPoseFlowConfigError(f"{name} must be positive")
    if self.learning_rate <= 0.0:
      raise JointPoseFlowConfigError("learning_rate must be positive")
    if not 0.0 <= self.history_dropout_probability < 1.0:
      raise JointPoseFlowConfigError(
        "history_dropout_probability must be in [0, 1)"
      )
    if self.scheduler not in ("cosine", "constant"):
      raise JointPoseFlowConfigError(f"unknown scheduler {self.scheduler!r}")
    if self.mixed_precision not in ("bf16", "fp16", "none"):
      raise JointPoseFlowConfigError(
        f"unknown mixed_precision {self.mixed_precision!r}"
      )
    if self.loss_reduction != "channel_mean":
      # Every generated row has the full channel width, so token_mean would give
      # the same number; refusing it keeps the recorded value honest.
      raise JointPoseFlowConfigError(
        "the joint-pose flow loss is always the channel mean; "
        f"loss_reduction must be 'channel_mean', got {self.loss_reduction!r}"
      )
    # A warmup that consumes the whole schedule never reaches the base rate.
    # A single-update run has no schedule to speak of, so it is exempt.
    if (
        self.max_steps is not None
        and self.max_steps > 1
        and self.warmup_updates >= self.max_steps
    ):
      raise JointPoseFlowConfigError(
        f"warmup_updates {self.warmup_updates} must be smaller than max_steps "
        f"{self.max_steps}; otherwise the base learning rate is never reached"
      )

  @property
  def effective_batch(self) -> int:
    return self.micro_batch * self.gradient_accumulation_steps

  def require_budget(self) -> int:
    """Return the update budget, refusing a run that did not state one.

    The flow path must never start an unbounded run from a default.
    """
    if self.max_steps is None:
      raise JointPoseFlowConfigError(
        "max_steps must be set explicitly before training can start"
      )
    return self.max_steps

  def as_dict(self) -> dict[str, object]:
    return {
        "experiment_name": self.experiment_name,
        "log_root": self.log_root,
        "seed": self.seed,
        "micro_batch": self.micro_batch,
        "gradient_accumulation_steps": self.gradient_accumulation_steps,
        "effective_batch": self.effective_batch,
        "max_steps": self.max_steps,
        "learning_rate": self.learning_rate,
        "weight_decay": self.weight_decay,
        "warmup_updates": self.warmup_updates,
        "max_grad_norm": self.max_grad_norm,
        "scheduler": self.scheduler,
        "mixed_precision": self.mixed_precision,
        "loss_reduction": self.loss_reduction,
        "ema_power": self.ema_power,
        "ema_max_decay": self.ema_max_decay,
        "max_level": self.max_level,
        "history_dropout_probability": self.history_dropout_probability,
        "checkpoint_every": self.checkpoint_every,
        "log_every": self.log_every,
    }


@dataclass(frozen=True)
class JointPoseFlowRunCfg:
  """Dataset plus trainer, the pair a run is identified by."""

  dataset: JointPoseDatasetCfg = field(default_factory=JointPoseDatasetCfg)
  trainer: FlowMatchingTrainerCfg = field(default_factory=FlowMatchingTrainerCfg)

  def as_dict(self) -> dict[str, object]:
    return {"dataset": self.dataset.as_dict(), "trainer": self.trainer.as_dict()}


def joint_pose_flow_cfg(
    *, data_dir: str | None = None, max_steps: int | None = None
) -> JointPoseFlowRunCfg:
  """Default run configuration, in the shape of the tracking task factories."""
  dataset = JointPoseDatasetCfg() if data_dir is None else JointPoseDatasetCfg(
      data_dir=data_dir
  )
  trainer = FlowMatchingTrainerCfg() if max_steps is None else FlowMatchingTrainerCfg(
      max_steps=max_steps
  )
  return JointPoseFlowRunCfg(dataset=dataset, trainer=trainer)
