"""Train the latent-free joint-pose x0 flow model.

One run is bounded by an explicit update budget.  ``--trainer.max-steps`` has no
default: a run that does not state a budget is refused before any dataset is
opened.  The three splits of the prepared dataset directory are opened so their
window counts and reference shares are printed before training starts.  The
validation and test splits are passed to the runner as ``eval_datasets``; the
Stage 1 runner does not sample them during training.

The dataset loader and the runner own the semantics of the run.  This module
only builds them from the command line, so there is no YAML, no contract and no
schema hash here.

Usage::

    uv run train-flow --trainer.max-steps 20000 \\
        --dataset.data-dir /path/to/data --device cuda:0
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

import tyro

import mjlab
from mjlab.tasks.tracking.joint_pose import (
  FlowMatchingRunner,
  FlowMatchingTrainerCfg,
  JointPoseDatasetCfg,
  JointPoseDatasetError,
  JointPoseFlowConfigError,
  JointPoseFlowRunnerError,
  JointPoseWindowDataset,
)

SPLITS = ("train", "validation", "test")
"""Split order used for the startup report and for the evaluation datasets."""


@dataclass(frozen=True)
class TrainFlowConfig:
  """Dataset and optimisation surface of one joint-pose flow run."""

  dataset: JointPoseDatasetCfg = field(default_factory=JointPoseDatasetCfg)
  trainer: FlowMatchingTrainerCfg = field(default_factory=FlowMatchingTrainerCfg)
  device: str = "cpu"
  output: str | None = None
  """Run directory. Default: ``<log_root>/<experiment_name>/<timestamp>``."""


def open_datasets(cfg: JointPoseDatasetCfg) -> dict[str, JointPoseWindowDataset]:
  """Open all three splits of one prepared dataset directory."""
  return {split: JointPoseWindowDataset(cfg, split=split) for split in SPLITS}


def window_report(datasets: dict[str, JointPoseWindowDataset]) -> list[str]:
  """Return one line per split, with its window count and reference share."""
  lines = []
  for split in SPLITS:
    dataset = datasets[split]
    lines.append(
      f"[INFO] {split}: {dataset.n_achieved} achieved + "
      f"{dataset.n_reference} reference = {len(dataset)} windows, "
      f"reference share {dataset.reference_share:.3f}"
    )
  return lines


def run_training(cfg: TrainFlowConfig) -> dict[str, object]:
  """Build the datasets and the runner, train, and return the summary."""
  budget = cfg.trainer.require_budget()
  datasets = open_datasets(cfg.dataset)
  for line in window_report(datasets):
    print(line, flush=True)
  print(
    f"[INFO] budget={budget} updates, "
    f"effective batch={cfg.trainer.effective_batch}, "
    f"device={cfg.device}, output={cfg.output or 'timestamped'}",
    flush=True,
  )
  runner = FlowMatchingRunner(
    datasets["train"],
    {split: datasets[split] for split in SPLITS if split != "train"},
    cfg.trainer,
    cfg.device,
    run_dir=cfg.output,
  )
  print(f"[INFO] run directory: {runner.run_dir}", flush=True)
  return runner.learn()


def main(argv: list[str] | None = None) -> int:
  """Run one bounded training job and return a process exit code."""
  args = tyro.cli(TrainFlowConfig, args=argv, config=mjlab.TYRO_FLAGS)
  try:
    summary = run_training(args)
  except (
    JointPoseFlowConfigError,
    JointPoseDatasetError,
    JointPoseFlowRunnerError,
  ) as exc:
    print(f"train-flow: {exc}", file=sys.stderr)
    return 2
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
