"""Stage 4 evaluation of the latent-free joint-pose x0 flow model.

Three readings of one trained checkpoint, all on held-out windows of the
prepared 68-channel dataset.  Every generated window comes from
``FlowMatchingRunner.sample`` conditioned on the recorded history rows; no
training loss and no teacher action is reported here.

1. **Per-split channel-mean x0 error.**  The generated rows after ``past_steps``
   are compared with the recorded rows in the dataset's normalized channel space,
   which is the space the training loss uses.  Two conditioning-only baselines
   are scored on the same rows: ``repeat_last_row`` holds the last history row,
   and ``constant_velocity`` continues the last history step at a constant
   velocity.  The error is not comparable across datasets, only within one table.

2. **Mode distance.**  Root yaw rate, knee flexion and joint speed are the three
   behaviour axes of ``scripts/experiments/replay_survey/analyze_modes.py``.  Each
   axis is summarised by a histogram over generated and reference windows and
   compared with a total-variation distance on shared bin edges.

3. **Replay export.**  ``export_replay_windows`` writes K generated windows as
   tracking motion files for ``scripts/experiments/replay_survey/survey.py``.

Body channels are not recoverable from the 68 channels
------------------------------------------------------
The window contract holds root orientation 6D, joint positions and joint
velocities.  It holds **no root translation**, so world-frame body positions,
body orientations, body linear velocities and body angular velocities cannot be
computed from a generated window.  The export therefore takes those four arrays
from the **nearest reference clip frame**: each generated window is matched to
the reference window whose joint channels are closest, and every row of the
matched reference window supplies the body channels of the same row of the
generated window.  The exported body kinematics are reference kinematics, not
generated kinematics.  ``export_replay_windows`` says this in its own docstring
and writes the same statement, with the per-window match and distance, into
``manifest.json`` next to the exported files.

The generated root 6D channels are not exported either: they are heading-local
to each window's centre and the tracking file takes root orientation from the
body channels.  This is the joint-only substitution the replay pilot already
uses, with the generated joints in place of the recorded ones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence, TypedDict, cast

import numpy as np
import torch

from ..flow_matching.sampler import DEFAULT_CALLS
from .config import CHANNEL_LAYOUT, FlowMatchingTrainerCfg, JointPoseDatasetCfg
from .dataset import JointPoseDatasetError, JointPoseWindowDataset
from .runner import CHECKPOINT_FORMAT, FlowMatchingRunner, JointPoseFlowRunnerError

EXPORT_FORMAT = "mjlab-x2-joint-pose-flow-replay-export-v1"
REFERENCE_MANIFEST_FORMAT = "mjlab-x2-joint-pose-reference-source-v1"

WINDOW_BATCH = 32
"""Windows generated per sampler call; bounds the memory of one batch."""

HISTOGRAM_BINS = 16
"""Bin count of the mode histograms, as in the mode analysis script."""

FPS = 50.0
"""Frame rate of the prepared 68-channel contract."""

_CHANNEL_SLICES = {name: slice(start, stop) for name, start, stop in CHANNEL_LAYOUT}
ROOT_ROT6D = _CHANNEL_SLICES["root_rot6d"]
JOINT_POSITIONS = _CHANNEL_SLICES["joint_positions"]
JOINT_VELOCITIES = _CHANNEL_SLICES["joint_velocities"]
JOINT_CHANNELS = slice(JOINT_POSITIONS.start, JOINT_VELOCITIES.stop)
"""Joint channels: the part a window describes without a heading frame."""

KNEE_JOINT_INDICES = (3, 9)
"""Left and right knee columns of the prepared dataset's joint order.

The order is the robot's joint order, the same order
``scripts/experiments/replay_survey/analyze_modes.py`` names.  The knee axis
averages both knees, so exchanging the two indices cannot change its value.
"""

MODE_AXES: dict[str, str] = {
  "root_yaw_rate": "deg/s",
  "knee_flexion": "deg",
  "joint_speed": "rad/s",
}
"""The three mode axes of the mode analysis script and their units."""

_REPLAY_KEYS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
  "fps",
)
"""Keys ``scripts/experiments/replay_survey/survey.py`` reads from a motion file."""


class JointPoseEvaluationError(ValueError):
  """An evaluation input does not match the contract it claims to follow."""


# ---------------------------------------------------------------- error metrics


def channel_mean_mse(prediction: np.ndarray, target: np.ndarray) -> float:
  """Mean squared error over every channel of every compared row.

  All three channel groups have the same width per row, so the mean over
  elements is the channel mean, exactly as in the training loss.
  """
  predicted = np.asarray(prediction, dtype=np.float64)
  expected = np.asarray(target, dtype=np.float64)
  if predicted.shape != expected.shape:
    raise JointPoseEvaluationError(
      f"prediction {predicted.shape} does not match target {expected.shape}"
    )
  return float(np.square(predicted - expected).mean())


def repeat_last_row_baseline(windows: np.ndarray, *, past_steps: int) -> np.ndarray:
  """Predict every generated row as the last history row.

  The history rows are reproduced unchanged, so the result has the full window
  shape and can be compared with a generated window row for row.
  """
  values = np.asarray(windows, dtype=np.float64)
  if not 1 <= past_steps < values.shape[1]:
    raise JointPoseEvaluationError(
      f"past_steps {past_steps} does not leave generated rows in {values.shape[1]} rows"
    )
  out = np.array(values, dtype=np.float64, copy=True)
  out[:, past_steps:, :] = values[:, past_steps - 1 : past_steps, :]
  return out


def constant_velocity_baseline(windows: np.ndarray, *, past_steps: int) -> np.ndarray:
  """Continue the history at the velocity of its last step.

  Root orientation and joint positions advance by the difference of the last two
  history rows, which is the straight-line continuation of the conditioning
  history.  Joint velocities hold their last value: a constant-velocity
  prediction has no acceleration, so extrapolating a velocity would invent one.
  """
  values = np.asarray(windows, dtype=np.float64)
  if not 2 <= past_steps < values.shape[1]:
    raise JointPoseEvaluationError(
      f"past_steps {past_steps} does not leave two history rows before the "
      f"{values.shape[1]} window rows"
    )
  out = np.array(values, dtype=np.float64, copy=True)
  steps = np.arange(1, values.shape[1] - past_steps + 1, dtype=np.float64)
  last = values[:, past_steps - 1, :]
  velocity = last - values[:, past_steps - 2, :]
  for region in (ROOT_ROT6D, JOINT_POSITIONS):
    slope = velocity[:, region]
    out[:, past_steps:, region] = (
      last[:, None, region] + steps[None, :, None] * slope[:, None, :]
    )
  out[:, past_steps:, JOINT_VELOCITIES] = values[
    :, past_steps - 1 : past_steps, JOINT_VELOCITIES
  ]
  return out


# ----------------------------------------------------------------- mode metrics


def rot6d_matrix(six: np.ndarray) -> np.ndarray:
  """Recover an orthonormal rotation matrix from the two packed 6D columns.

  The channels are z-scored, so after denormalization the two packed columns are
  neither orthogonal nor unit length.  Gram-Schmidt restores a rotation, which is
  the re-orthonormalization the preprocess spec asks for at inference.
  """
  values = np.asarray(six, dtype=np.float64)
  if values.shape[-1] != 6:
    raise JointPoseEvaluationError(
      f"a 6D rotation needs 6 values, got {values.shape[-1]}"
    )
  first = values[..., 0:3]
  second = values[..., 3:6]
  first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
  axis_one = first / np.where(first_norm < 1e-8, 1.0, first_norm)
  projection = np.sum(second * axis_one, axis=-1, keepdims=True)
  orthogonal = second - projection * axis_one
  second_norm = np.linalg.norm(orthogonal, axis=-1, keepdims=True)
  axis_two = orthogonal / np.where(second_norm < 1e-8, 1.0, second_norm)
  axis_three = np.cross(axis_one, axis_two)
  return np.stack((axis_one, axis_two, axis_three), axis=-1)


def yaw_of_matrix(matrix: np.ndarray) -> np.ndarray:
  """Return the heading angle of a rotation whose third column is vertical."""
  values = np.asarray(matrix, dtype=np.float64)
  return np.arctan2(values[..., 1, 0], values[..., 0, 0])


def root_yaw_rate_deg_s(
  windows: np.ndarray, *, past_steps: int, fps: float = FPS
) -> np.ndarray:
  """Mean absolute root yaw rate over the generated rows, in degrees/second.

  Every row of a window is rotated by one constant heading, so a difference of
  two rows is the true relative yaw.  The first difference starts at the last
  history row and the rest are inside the generated region.
  """
  values = np.asarray(windows, dtype=np.float64)
  if not 1 <= past_steps < values.shape[1]:
    raise JointPoseEvaluationError(
      f"past_steps {past_steps} does not leave generated rows in {values.shape[1]} rows"
    )
  yaw = yaw_of_matrix(rot6d_matrix(values[:, past_steps - 1 :, ROOT_ROT6D]))
  step = np.diff(yaw, axis=1)
  wrapped = np.arctan2(np.sin(step), np.cos(step))
  return np.rad2deg(np.abs(wrapped).mean(axis=1)) * float(fps)


def knee_flexion_deg(windows: np.ndarray, *, past_steps: int) -> np.ndarray:
  """Mean knee joint angle over the generated rows, in degrees."""
  values = np.asarray(windows, dtype=np.float64)
  knees = values[:, past_steps:, JOINT_POSITIONS][..., list(KNEE_JOINT_INDICES)]
  return np.rad2deg(knees.mean(axis=(1, 2)))


def joint_speed_rad_s(windows: np.ndarray, *, past_steps: int) -> np.ndarray:
  """Mean absolute joint velocity over the generated rows, in radians/second."""
  values = np.asarray(windows, dtype=np.float64)
  return np.abs(values[:, past_steps:, JOINT_VELOCITIES]).mean(axis=(1, 2))


def mode_values(
  windows: np.ndarray, *, past_steps: int, fps: float = FPS
) -> dict[str, np.ndarray]:
  """Return the three mode axes for one batch of windows, in physical units.

  ``windows`` must already be denormalized; see :class:`ChannelNormalization`.
  """
  return {
    "root_yaw_rate": root_yaw_rate_deg_s(windows, past_steps=past_steps, fps=fps),
    "knee_flexion": knee_flexion_deg(windows, past_steps=past_steps),
    "joint_speed": joint_speed_rad_s(windows, past_steps=past_steps),
  }


def _finite_samples(name: str, values: np.ndarray) -> np.ndarray:
  samples = np.asarray(values, dtype=np.float64).ravel()
  if samples.size == 0:
    raise JointPoseEvaluationError(f"mode axis {name} has no samples")
  if not np.isfinite(samples).all():
    raise JointPoseEvaluationError(f"mode axis {name} has non-finite samples")
  return samples


class HistogramSummary(TypedDict):
  """One side of a histogram comparison."""

  count: int
  mean: float
  quantiles: list[float]
  probability: list[float]


class HistogramDistance(TypedDict):
  """Shared-bin comparison of two one-dimensional samples."""

  bins: int
  range: list[float]
  total_variation: float
  generated: HistogramSummary
  reference: HistogramSummary


class ModeDistance(HistogramDistance):
  """One mode axis: its unit and the histogram distance of that axis."""

  unit: str


class SplitScores(TypedDict):
  """One split of the channel-mean error table."""

  split: str
  windows: int
  metrics: dict[str, float]
  gates: dict[str, bool]


class ExportEntry(TypedDict):
  """One exported motion file."""

  file: str
  rows: int
  reference_window: int
  reference_distance: float


class ExportManifest(TypedDict):
  """The record written next to the exported motion files."""

  format: str
  windows: int
  fps: float
  keys: list[str]
  joint_source: str
  body_source: str
  root_orientation: str
  history_rows: str
  normalization_source: str
  files: list[ExportEntry]
  provenance: dict[str, object]


def _histogram_summary(values: np.ndarray, probability: np.ndarray) -> HistogramSummary:
  quantiles = np.percentile(values, [5.0, 50.0, 95.0])
  return {
    "count": int(values.size),
    "mean": round(float(values.mean()), 6),
    "quantiles": [round(float(value), 6) for value in quantiles],
    "probability": [round(float(value), 6) for value in probability],
  }


def histogram_distance(
  generated: np.ndarray, reference: np.ndarray, *, bins: int = HISTOGRAM_BINS
) -> HistogramDistance:
  """Compare two one-dimensional samples on shared bins.

  Bin edges span the union of both samples, so both histograms count the same
  support.  The distance is the total variation of the two normalized
  histograms: 0 when they are identical and 1 when they share no bin.
  """
  if bins < 2:
    raise JointPoseEvaluationError("histogram bins must be at least 2")
  left = _finite_samples("generated", generated)
  right = _finite_samples("reference", reference)
  low = float(min(left.min(), right.min()))
  high = float(max(left.max(), right.max()))
  edges = np.linspace(low, high, bins + 1)
  left_counts, _ = np.histogram(left, bins=edges)
  right_counts, _ = np.histogram(right, bins=edges)
  left_probability = left_counts / left.size
  right_probability = right_counts / right.size
  distance = 0.5 * np.abs(left_probability - right_probability).sum()
  return {
    "bins": bins,
    "range": [round(low, 6), round(high, 6)],
    "total_variation": round(float(distance), 6),
    "generated": _histogram_summary(left, left_probability),
    "reference": _histogram_summary(right, right_probability),
  }


# ------------------------------------------------------------------ generation


@dataclass(frozen=True)
class ChannelNormalization:
  """The dataset's train-only per-channel mean and standard deviation.

  The prepared windows are z-scored per channel, so a mode axis and an exported
  motion file need this record to return to radians and radians/second.
  """

  mean: np.ndarray
  std: np.ndarray
  source: str = ""

  def __post_init__(self) -> None:
    mean = np.asarray(self.mean, dtype=np.float64)
    std = np.asarray(self.std, dtype=np.float64)
    if mean.shape != std.shape or mean.ndim != 1:
      raise JointPoseEvaluationError(
        f"mean {mean.shape} and std {std.shape} must be 1-D and equal"
      )
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
      raise JointPoseEvaluationError("normalization holds a non-finite value")
    if bool((std <= 0.0).any()):
      raise JointPoseEvaluationError("normalization std must be positive")
    object.__setattr__(self, "mean", mean)
    object.__setattr__(self, "std", std)

  @property
  def channels(self) -> int:
    return int(self.mean.size)

  @classmethod
  def from_data_dir(cls, directory: Path | str) -> "ChannelNormalization":
    """Read ``projection.npz`` from one prepared dataset directory."""
    path = Path(directory).expanduser() / "projection.npz"
    if not path.is_file():
      raise JointPoseEvaluationError(f"dataset statistics not found: {path}")
    with np.load(path, allow_pickle=False) as stats:
      return cls(mean=stats["mean"], std=stats["std"], source=str(path))

  def denormalize(self, windows: np.ndarray) -> np.ndarray:
    """Map normalized windows back to radians, radians/second and rotation 6D."""
    values = np.asarray(windows, dtype=np.float64)
    if values.shape[-1] != self.channels:
      raise JointPoseEvaluationError(
        f"window has {values.shape[-1]} channels but the normalization holds "
        f"{self.channels}"
      )
    return values * self.std + self.mean


def generate_windows(
  runner: FlowMatchingRunner,
  windows: np.ndarray,
  *,
  steps: int,
  batch: int = WINDOW_BATCH,
  parameters: str = "ema",
) -> np.ndarray:
  """Generate the full window of each recorded window from its history rows.

  The history rows of the result are the recorded rows, unchanged, so a
  comparison with the record includes exactly the rows the model generated.

  ``parameters`` defaults to the exponential moving average, which is the set the
  trainer maintains for evaluation.  A report must state which set produced its
  numbers; :func:`parameters_sha256` fingerprints it after generation.
  """
  values = np.asarray(windows, dtype=np.float32)
  if values.ndim != 3:
    raise JointPoseEvaluationError(
      f"windows must be [count, rows, channels], got {values.shape}"
    )
  if values.shape[0] == 0:
    raise JointPoseEvaluationError("no window was selected for generation")
  past = runner.past_steps
  out = np.empty(values.shape, dtype=np.float32)
  out[:, :past] = values[:, :past]
  for start in range(0, values.shape[0], batch):
    stop = min(values.shape[0], start + batch)
    history = torch.from_numpy(np.ascontiguousarray(values[start:stop, :past]))
    generated = runner.sample(history, steps, parameters=parameters)
    out[start:stop] = generated.detach().cpu().numpy().astype(np.float32)
  return out


def sample_generator_sha256(runner: FlowMatchingRunner) -> str:
  """Fingerprint the sampling generator, so a report states its noise stream."""
  state = runner.rng_state()["sample_generator"]
  array = state.cpu().numpy() if torch.is_tensor(state) else np.asarray(state)
  return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def model_parameters_sha256(runner: FlowMatchingRunner) -> str:
  """Fingerprint the denoiser parameters the generated windows came from.

  A fresh training run does not reproduce bit-identically, because the runner
  takes the denoiser's initial weights from the global torch RNG.  Evaluating one
  checkpoint does repeat, so the parameter fingerprint is what ties a table to
  the weights behind it.
  """
  digest = hashlib.sha256()
  for name, value in sorted(runner.model.state_dict().items()):
    digest.update(name.encode("utf-8"))
    array = np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)
    digest.update(array.tobytes())
  return digest.hexdigest()


def _reference_side(
  dataset: JointPoseWindowDataset, max_windows: int | None
) -> np.ndarray:
  """Windows the tracker was asked to follow, or every window without them."""
  if dataset.n_reference:
    available = dataset.n_reference
    first = dataset.n_achieved
  else:
    available = len(dataset)
    first = 0
  count = available if max_windows is None else min(available, max_windows)
  if count <= 0:
    raise JointPoseEvaluationError(f"split {dataset.split} has no window to compare")
  rows = [
    np.asarray(dataset[first + offset], dtype=np.float32) for offset in range(count)
  ]
  return np.stack(rows)


# -------------------------------------------------------------------- reports


def evaluate_split(
  runner: FlowMatchingRunner,
  dataset: JointPoseWindowDataset,
  *,
  steps: int,
  max_windows: int | None = None,
  batch: int = WINDOW_BATCH,
) -> SplitScores:
  """Score generated rows and both baselines on one split.

  Windows are scored in dataset order, so the table does not depend on a random
  draw.  ``max_windows`` bounds the cost; ``None`` scores the whole split.
  """
  past = runner.past_steps
  count = len(dataset) if max_windows is None else min(len(dataset), max_windows)
  if count <= 0:
    raise JointPoseEvaluationError(f"split {dataset.split} has no window to score")
  totals = {"model": 0.0, "repeat_last_row": 0.0, "constant_velocity": 0.0}
  elements = 0
  for start in range(0, count, batch):
    stop = min(count, start + batch)
    target = np.stack(
      [np.asarray(dataset[index], dtype=np.float32) for index in range(start, stop)]
    )
    generated = target[:, past:]
    # Every prediction has the same generated region, so one element count per
    # batch weights the per-batch means into one split mean.
    elements += generated.size
    predictions = {
      "model": generate_windows(runner, target, steps=steps, batch=batch)[:, past:],
      "repeat_last_row": repeat_last_row_baseline(target, past_steps=past)[:, past:],
      "constant_velocity": constant_velocity_baseline(target, past_steps=past)[
        :, past:
      ],
    }
    for name, prediction in predictions.items():
      error = channel_mean_mse(prediction, generated)
      if not np.isfinite(error):
        raise JointPoseEvaluationError(
          f"{name} channel-mean error is non-finite on split {dataset.split}"
        )
      totals[name] += error * generated.size
  metrics = {name: total / elements for name, total in totals.items()}
  model = metrics["model"]
  repeat = metrics["repeat_last_row"]
  velocity = metrics["constant_velocity"]
  return SplitScores(
    split=dataset.split,
    windows=count,
    metrics={
      "model_channel_mean_mse": model,
      "repeat_last_row_channel_mean_mse": repeat,
      "constant_velocity_channel_mean_mse": velocity,
    },
    gates={
      "beats_repeat_last_row": model < repeat,
      "beats_constant_velocity": model < velocity,
    },
  )


def evaluate_modes(
  runner: FlowMatchingRunner,
  dataset: JointPoseWindowDataset,
  *,
  steps: int,
  normalization: ChannelNormalization,
  max_windows: int | None = None,
  batch: int = WINDOW_BATCH,
  fps: float = FPS,
) -> tuple[dict[str, ModeDistance], int]:
  """Compare the mode distribution of generated and reference windows.

  The reference side is the dataset's reference source: the clips the tracker was
  asked to follow.  The generated side is produced from the history rows of those
  same windows, so the two sides see the same conditioning.  A split without a
  reference source compares generated windows with its own record instead.
  """
  truth = _reference_side(dataset, max_windows)
  generated = generate_windows(runner, truth, steps=steps, batch=batch)
  generated_modes = mode_values(
    normalization.denormalize(generated), past_steps=runner.past_steps, fps=fps
  )
  reference_modes = mode_values(
    normalization.denormalize(truth), past_steps=runner.past_steps, fps=fps
  )
  summary: dict[str, ModeDistance] = {}
  for axis, unit in MODE_AXES.items():
    distance = histogram_distance(generated_modes[axis], reference_modes[axis])
    summary[axis] = ModeDistance(
      unit=unit,
      bins=distance["bins"],
      range=distance["range"],
      total_variation=distance["total_variation"],
      generated=distance["generated"],
      reference=distance["reference"],
    )
  return summary, int(truth.shape[0])


@dataclass(frozen=True)
class EvaluationReport:
  """Per-split x0 error, mode distance and the sampling provenance behind them."""

  splits: dict[str, SplitScores]
  modes: dict[str, ModeDistance]
  mode_windows: int
  sampler_calls: int
  model_parameters_sha256: str
  sample_generator_sha256: str
  diagnostics_path: str

  def as_dict(self) -> dict[str, object]:
    return {
      "splits": {name: dict(payload) for name, payload in self.splits.items()},
      "modes": {name: dict(payload) for name, payload in self.modes.items()},
      "mode_windows": self.mode_windows,
      "sampler_calls": self.sampler_calls,
      "model_parameters_sha256": self.model_parameters_sha256,
      "sample_generator_sha256": self.sample_generator_sha256,
      "diagnostics_path": self.diagnostics_path,
    }

  def table(self) -> str:
    """Render the two report tables as markdown."""
    lines = [
      "# X2 joint-pose flow evaluation",
      "",
      f"Sampler calls per window: {self.sampler_calls}. "
      f"Mode windows: {self.mode_windows}.",
      f"Model parameters sha256: `{self.model_parameters_sha256}`.",
      f"Sample generator sha256: `{self.sample_generator_sha256}`.",
      "",
      "Channel-mean x0 MSE is measured on rows "
      "``past_steps:window_steps`` in the dataset's normalized channel space, "
      "which is the space the training loss uses. It is comparable within this "
      "table only.",
      "",
      "| split | windows | model | repeat last row | constant velocity | beats both |",
      "|---|---:|---:|---:|---:|---|",
    ]
    for split, payload in self.splits.items():
      metrics = payload["metrics"]
      gates = payload["gates"]
      lines.append(
        f"| {split} | {payload['windows']} | "
        f"{metrics['model_channel_mean_mse']:.5f} | "
        f"{metrics['repeat_last_row_channel_mean_mse']:.5f} | "
        f"{metrics['constant_velocity_channel_mean_mse']:.5f} | "
        f"{'yes' if all(gates.values()) else 'no'} |"
      )
    lines += [
      "",
      "## Mode distance, generated against reference",
      "",
      "Total variation is 0 for identical histograms and 1 when the two "
      "histograms share no bin.",
      "",
      "| axis | unit | total variation | generated p5 / p50 / p95 | "
      "reference p5 / p50 / p95 |",
      "|---|---|---:|---|---|",
    ]
    for axis, payload in self.modes.items():
      lines.append(
        f"| {axis} | {payload['unit']} | "
        f"{payload['total_variation']:.3f} | "
        f"{_quantile_text(payload['generated'])} | "
        f"{_quantile_text(payload['reference'])} |"
      )
    return "\n".join(lines) + "\n"


def _quantile_text(summary: HistogramSummary) -> str:
  return " / ".join(f"{value:.3f}" for value in summary["quantiles"])


def evaluate_model(
  runner: FlowMatchingRunner,
  eval_datasets: Mapping[str, JointPoseWindowDataset],
  *,
  steps: int,
  normalization: ChannelNormalization | None = None,
  max_windows: int | None = None,
  batch: int = WINDOW_BATCH,
  fps: float = FPS,
  diagnostics_path: Path | str | None = None,
) -> EvaluationReport:
  """Score every evaluation split and the mode distance, then write a report."""
  if not 1 <= steps <= runner.process.max_level:
    raise JointPoseEvaluationError(
      f"steps {steps} must be in 1..{runner.process.max_level}"
    )
  if not eval_datasets:
    raise JointPoseEvaluationError("no evaluation split was supplied")
  resolved = (
    normalization
    if normalization is not None
    else ChannelNormalization.from_data_dir(runner.dataset.root)
  )
  destination = (
    Path(diagnostics_path) if diagnostics_path is not None else Path("<not-written>")
  )
  started = time.monotonic()
  splits = {
    name: evaluate_split(
      runner, dataset, steps=steps, max_windows=max_windows, batch=batch
    )
    for name, dataset in eval_datasets.items()
  }
  # The mode distance is one distribution comparison, so it is reported once:
  # for the first split in the mapping order.
  first_dataset = next(iter(eval_datasets.values()))
  modes, mode_windows = evaluate_modes(
    runner,
    first_dataset,
    steps=steps,
    normalization=resolved,
    max_windows=max_windows,
    batch=batch,
    fps=fps,
  )
  report = EvaluationReport(
    splits=splits,
    modes=modes,
    mode_windows=mode_windows,
    sampler_calls=int(steps),
    model_parameters_sha256=model_parameters_sha256(runner),
    sample_generator_sha256=sample_generator_sha256(runner),
    diagnostics_path=str(destination),
  )
  _write_report(destination, report, elapsed=time.monotonic() - started)
  return report


def _write_report(path: Path, report: EvaluationReport, *, elapsed: float) -> None:
  """Write the JSON report and its markdown table when a path was given."""
  if str(path) == "<not-written>":
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  payload = dict(report.as_dict())
  payload["elapsed_seconds"] = round(elapsed, 3)
  path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  path.with_suffix(".md").write_text(report.table(), encoding="utf-8")


# ------------------------------------------------------------- replay export


@dataclass(frozen=True)
class ReferenceBodyWindows:
  """Body kinematics of candidate reference windows, one row per clip frame.

  ``body_pos_w`` is ``[windows, window_steps, bodies, 3]`` and ``body_quat_w`` is
  ``[windows, window_steps, bodies, 4]``; both velocity arrays follow the same
  layout.  The four arrays come from the tennis clips, not from the 68 channels.
  """

  body_pos_w: np.ndarray
  body_quat_w: np.ndarray
  body_lin_vel_w: np.ndarray
  body_ang_vel_w: np.ndarray
  fps: float

  def __post_init__(self) -> None:
    shapes = {}
    for name in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
      array = np.asarray(getattr(self, name), dtype=np.float32)
      if array.ndim != 4:
        raise JointPoseEvaluationError(
          f"{name} must be [windows, rows, bodies, width], got {array.shape}"
        )
      object.__setattr__(self, name, array)
      shapes[name] = array.shape[:3]
    if len(set(shapes.values())) != 1:
      raise JointPoseEvaluationError(f"body arrays disagree in shape: {shapes}")
    if not np.isfinite(float(self.fps)) or float(self.fps) <= 0.0:
      raise JointPoseEvaluationError("fps must be positive")

  def __len__(self) -> int:
    return int(self.body_pos_w.shape[0])

  @property
  def window_steps(self) -> int:
    return int(self.body_pos_w.shape[1])


def nearest_reference_window(
  generated: np.ndarray, reference: np.ndarray, *, chunk: int = 4096
) -> tuple[np.ndarray, np.ndarray]:
  """Match each generated window to the reference window nearest in joint content.

  The distance is the mean squared difference over the joint position and
  velocity channels of the whole window.  The root 6D channels are left out: they
  are heading-local to each window's centre and do not describe how the window
  moves.  Both arrays must be in the same space.  The reference set is scanned in
  chunks so a large candidate set does not need its full distance matrix.
  """
  query = np.asarray(generated, dtype=np.float64)
  candidates = np.asarray(reference, dtype=np.float64)
  if query.shape[1:] != candidates.shape[1:]:
    raise JointPoseEvaluationError(
      f"generated windows {query.shape[1:]} and reference windows "
      f"{candidates.shape[1:]} do not share a layout"
    )
  if candidates.shape[0] == 0:
    raise JointPoseEvaluationError("no reference window is available to match")
  flat_query = np.ascontiguousarray(query[..., JOINT_CHANNELS]).reshape(
    query.shape[0], -1
  )
  width = flat_query.shape[1]
  query_square = np.square(flat_query).sum(axis=1)
  best_index = np.zeros(query.shape[0], dtype=np.int64)
  best_score = np.full(query.shape[0], np.inf, dtype=np.float64)
  for start in range(0, candidates.shape[0], chunk):
    stop = min(candidates.shape[0], start + chunk)
    block = np.ascontiguousarray(candidates[start:stop, ..., JOINT_CHANNELS]).reshape(
      stop - start, -1
    )
    score = (
      query_square[:, None]
      - 2.0 * flat_query @ block.T
      + np.square(block).sum(axis=1)[None, :]
    )
    local = np.argmin(score, axis=1)
    value = score[np.arange(query.shape[0]), local]
    improved = value < best_score
    best_index = np.where(improved, local + start, best_index)
    best_score = np.where(improved, value, best_score)
  return best_index, np.maximum(best_score, 0.0) / float(width)


def export_replay_windows(
  *,
  generated: np.ndarray,
  reference: np.ndarray,
  reference_body: ReferenceBodyWindows,
  normalization: ChannelNormalization,
  out_dir: Path | str,
  prefix: str = "gen",
  provenance: Mapping[str, object] | None = None,
) -> ExportManifest:
  """Write generated windows as tracking motion files for the replay survey.

  Each file holds the keys ``survey.py`` reads: ``joint_pos``, ``joint_vel``,
  ``body_pos_w``, ``body_quat_w``, ``body_lin_vel_w``, ``body_ang_vel_w`` and
  ``fps``.

  **The body channels are reference body channels, not generated ones.**  The
  68-channel window carries no root translation, so world-frame body kinematics
  cannot be computed from a generated window.  Every generated window is matched
  to the reference window with the nearest joint content
  (:func:`nearest_reference_window`), and each row takes its body channels from
  the same row of that matched reference window.  The root orientation channels
  are not exported at all: the tracking file takes root orientation from
  ``body_quat_w``, which is where this substitution puts it.

  ``joint_pos`` and ``joint_vel`` are the generated channels mapped back to
  radians and radians/second with the dataset's train-only statistics, so the
  history rows and the generated rows share one unit.  ``manifest.json`` records
  the same statement, the per-window match and the match distance.
  """
  query = np.asarray(generated, dtype=np.float32)
  candidates = np.asarray(reference, dtype=np.float32)
  if query.ndim != 3 or candidates.ndim != 3:
    raise JointPoseEvaluationError(
      f"windows must be [count, rows, channels], got {query.shape}"
    )
  if query.shape[0] == 0:
    raise JointPoseEvaluationError("no generated window to export")
  if query.shape[1:] != candidates.shape[1:]:
    raise JointPoseEvaluationError(
      f"generated windows {query.shape[1:]} and reference windows "
      f"{candidates.shape[1:]} do not share a layout"
    )
  if query.shape[2] != normalization.channels:
    raise JointPoseEvaluationError(
      f"windows have {query.shape[2]} channels but the normalization holds "
      f"{normalization.channels}"
    )
  if candidates.shape[0] != len(reference_body):
    raise JointPoseEvaluationError(
      f"{candidates.shape[0]} reference windows but {len(reference_body)} body "
      "windows; an exported body row would be paired with another clip"
    )
  if candidates.shape[1] != reference_body.window_steps:
    raise JointPoseEvaluationError(
      f"reference windows hold {candidates.shape[1]} rows but the body windows "
      f"hold {reference_body.window_steps}"
    )

  matched, distance = nearest_reference_window(query, candidates)
  physical = normalization.denormalize(query)
  directory = Path(out_dir)
  directory.mkdir(parents=True, exist_ok=True)
  entries: list[ExportEntry] = []
  for row in range(query.shape[0]):
    index = int(matched[row])
    name = f"{prefix}{row:06d}.npz"
    np.savez(
      directory / name,
      joint_pos=np.ascontiguousarray(
        physical[row, :, JOINT_POSITIONS], dtype=np.float32
      ),
      joint_vel=np.ascontiguousarray(
        physical[row, :, JOINT_VELOCITIES], dtype=np.float32
      ),
      body_pos_w=reference_body.body_pos_w[index],
      body_quat_w=reference_body.body_quat_w[index],
      body_lin_vel_w=reference_body.body_lin_vel_w[index],
      body_ang_vel_w=reference_body.body_ang_vel_w[index],
      fps=np.array([float(reference_body.fps)], dtype=np.float32),
    )
    entries.append(
      ExportEntry(
        file=name,
        rows=int(query.shape[1]),
        reference_window=index,
        reference_distance=round(float(distance[row]), 6),
      )
    )

  manifest = ExportManifest(
    format=EXPORT_FORMAT,
    windows=len(entries),
    fps=float(reference_body.fps),
    keys=list(_REPLAY_KEYS),
    joint_source=(
      "generated channels 6:37 and 37:68, mapped back to radians and "
      "radians/second with the dataset train-only mean and std"
    ),
    body_source=(
      "nearest reference clip frame: each generated window takes the body "
      "channels of the reference window with the nearest joint content, row "
      "by row. These body kinematics are reference kinematics, not "
      "generated kinematics, because the 68-channel window holds no root "
      "translation."
    ),
    root_orientation=(
      "channels 0:6 are heading-local to each window's centre and are not "
      "exported; a tracking file takes root orientation from body_quat_w"
    ),
    history_rows=(
      "rows before past_steps are the recorded conditioning rows, not generated rows"
    ),
    normalization_source=normalization.source,
    files=entries,
    provenance={} if provenance is None else dict(provenance),
  )
  (directory / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  return manifest


def reference_body_windows(
  data_dir: Path | str,
  tennis_dir: Path | str,
  split: str,
  *,
  max_windows: int | None = None,
) -> ReferenceBodyWindows:
  """Load the body kinematics of one split's reference windows.

  The reference source is packed motion by motion in sorted motion order, and
  each motion packs one window per clip frame from 0, so a reference row maps
  back to ``(motion, start frame)`` from the source manifest.  The mapping is
  verified against the recorded frame and window counts and refused when it does
  not hold, because a shifted mapping would pair a generated window with the
  wrong clip frames.
  """
  root = Path(data_dir).expanduser()
  manifest_path = root / "reference" / "manifest.json"
  if not manifest_path.is_file():
    raise JointPoseEvaluationError(f"reference manifest not found: {manifest_path}")
  payload = json.loads(manifest_path.read_text(encoding="utf-8"))
  if payload.get("format") != REFERENCE_MANIFEST_FORMAT:
    raise JointPoseEvaluationError(
      f"{manifest_path} is {payload.get('format')!r}, expected "
      f"{REFERENCE_MANIFEST_FORMAT!r}"
    )
  window_steps = int(payload["contract"]["window"])
  wanted: list[tuple[str, int]] = []
  for source in payload.get("sources", []):
    if str(source["split"]) != split:
      continue
    frames = int(source["frames"])
    windows = int(source["windows"])
    if windows != frames - window_steps + 1:
      raise JointPoseEvaluationError(
        f"{source['motion']} packs {windows} windows from {frames} frames, so a "
        "reference row cannot be mapped back to a clip frame"
      )
    for start in range(windows):
      if max_windows is not None and len(wanted) >= max_windows:
        break
      wanted.append((str(source["motion"]), start))
  if not wanted:
    raise JointPoseEvaluationError(f"split {split} has no reference window")

  clip_cache: dict[str, dict[str, np.ndarray]] = {}
  clip_fps: dict[str, float] = {}
  rows: list[dict[str, np.ndarray]] = []
  frame_rate: float | None = None
  for motion, start in wanted:
    if motion not in clip_cache:
      path = Path(tennis_dir).expanduser() / motion
      if not path.is_file():
        raise JointPoseEvaluationError(f"reference clip not found: {path}")
      with np.load(path, allow_pickle=False) as clip:
        clip_cache.clear()
        clip_cache[motion] = {
          name: np.asarray(clip[name], dtype=np.float32)
          for name in (
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
          )
        }
        clip_fps[motion] = float(np.asarray(clip["fps"]).ravel()[0])
    rate = clip_fps[motion]
    if frame_rate is None:
      frame_rate = rate
    elif not np.isclose(frame_rate, rate):
      raise JointPoseEvaluationError(
        f"{motion} runs at {rate:g} Hz, earlier clips at {frame_rate:g} Hz"
      )
    window = slice(start, start + window_steps)
    rows.append({name: value[window] for name, value in clip_cache[motion].items()})
  assert frame_rate is not None
  return ReferenceBodyWindows(
    body_pos_w=np.stack([row["body_pos_w"] for row in rows]),
    body_quat_w=np.stack([row["body_quat_w"] for row in rows]),
    body_lin_vel_w=np.stack([row["body_lin_vel_w"] for row in rows]),
    body_ang_vel_w=np.stack([row["body_ang_vel_w"] for row in rows]),
    fps=frame_rate,
  )


# -------------------------------------------------------------------- command


def _declared_values(config_type: type[Any], recorded: object) -> dict[str, Any]:
  """Keep only the declared fields of a dataclass serialized through as_dict."""
  if not isinstance(recorded, Mapping):
    raise JointPoseEvaluationError("checkpoint record is missing or malformed")
  declared = {item.name for item in fields(config_type)}
  values = cast("Mapping[str, Any]", recorded)
  return {key: value for key, value in values.items() if key in declared}


def trainer_cfg_from_checkpoint(
  payload: Mapping[str, object],
) -> FlowMatchingTrainerCfg:
  """Rebuild the optimisation config recorded in a checkpoint."""
  return FlowMatchingTrainerCfg(
    **_declared_values(FlowMatchingTrainerCfg, payload.get("trainer"))
  )


def _model_sizes(payload: Mapping[str, object]) -> dict[str, int | None]:
  """Read the denoiser sizes a checkpoint was trained with."""
  settings = payload.get("model_settings")
  if not isinstance(settings, Mapping):
    return {"width": None, "layers": None, "heads": None}
  recorded = cast("Mapping[str, Any]", settings)
  sizes: dict[str, int | None] = {}
  for size, key in (
    ("width", "width"),
    ("layers", "layers"),
    ("heads", "attention_heads"),
  ):
    value = recorded.get(key)
    sizes[size] = None if value is None else int(value)
  return sizes


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Evaluate one joint-pose flow checkpoint."
  )
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--data-dir", type=str, default=None)
  parser.add_argument("--split", action="append", default=None)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--steps", type=int, default=DEFAULT_CALLS)
  parser.add_argument("--batch", type=int, default=WINDOW_BATCH)
  parser.add_argument("--max-windows", type=int, default=256)
  parser.add_argument("--export-windows", type=int, default=0)
  parser.add_argument("--export-split", default="validation")
  parser.add_argument(
    "--tennis-dir", type=Path, default=Path("/home/fushan/mjlab/data/tennis")
  )
  parser.add_argument("--max-reference-windows", type=int, default=2048)
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  """Evaluate one checkpoint and optionally export replay windows."""
  args = _parse_args(argv)
  checkpoint = args.checkpoint.expanduser()
  try:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
  except (OSError, RuntimeError, EOFError, ValueError) as exc:
    print(f"joint-pose-evaluate: cannot read {checkpoint}: {exc}", file=sys.stderr)
    return 2
  if not isinstance(payload, Mapping) or payload.get("format") != CHECKPOINT_FORMAT:
    print(
      f"joint-pose-evaluate: {checkpoint} is not a flow checkpoint", file=sys.stderr
    )
    return 2
  try:
    dataset_values = _declared_values(JointPoseDatasetCfg, payload.get("dataset_cfg"))
    if args.data_dir is not None:
      dataset_values["data_dir"] = args.data_dir
    dataset_cfg = JointPoseDatasetCfg(**dataset_values)
    trainer_cfg = trainer_cfg_from_checkpoint(payload)
    splits = list(args.split or ("validation", "test"))
    train = JointPoseWindowDataset(dataset_cfg, split="train")
    datasets = {
      name: JointPoseWindowDataset(dataset_cfg, split=name) for name in splits
    }
    runner = FlowMatchingRunner(
      train,
      datasets,
      trainer_cfg,
      args.device,
      run_dir=args.output_dir / "runner",
      **_model_sizes(payload),
    )
    runner.load(checkpoint)
    normalization = ChannelNormalization.from_data_dir(dataset_cfg.data_dir)
    report = evaluate_model(
      runner,
      datasets,
      steps=args.steps,
      normalization=normalization,
      max_windows=args.max_windows if args.max_windows > 0 else None,
      batch=args.batch,
      diagnostics_path=args.output_dir / "evaluation.json",
    )
    print(report.table(), flush=True)
    if args.export_windows > 0:
      export_dataset = datasets.get(args.export_split)
      if export_dataset is None:
        export_dataset = JointPoseWindowDataset(dataset_cfg, split=args.export_split)
      if not export_dataset.n_reference:
        raise JointPoseEvaluationError(
          f"split {args.export_split} has no reference source, so the exported "
          "body channels cannot come from a reference clip"
        )
      candidates = _reference_side(export_dataset, args.max_reference_windows)
      if args.export_windows > candidates.shape[0]:
        raise JointPoseEvaluationError(
          f"--export-windows {args.export_windows} exceeds the "
          f"{candidates.shape[0]} reference windows of split {args.export_split}"
        )
      body = reference_body_windows(
        dataset_cfg.data_dir,
        args.tennis_dir,
        args.export_split,
        max_windows=candidates.shape[0],
      )
      generated = generate_windows(
        runner, candidates[: args.export_windows], steps=args.steps, batch=args.batch
      )
      manifest = export_replay_windows(
        generated=generated,
        reference=candidates,
        reference_body=body,
        normalization=normalization,
        out_dir=args.output_dir / "replay",
        provenance={
          "checkpoint": str(checkpoint),
          "checkpoint_step": int(payload.get("global_step", 0)),
          "dataset_identity_sha256": train.identity_sha256(),
          "sampler_calls": int(args.steps),
          "model_parameters_sha256": report.model_parameters_sha256,
          "sample_generator_sha256": report.sample_generator_sha256,
        },
      )
      destination = args.output_dir / "replay"
      print(
        f"[INFO] exported {manifest['windows']} windows to {destination}", flush=True
      )
  except (
    JointPoseEvaluationError,
    JointPoseDatasetError,
    JointPoseFlowRunnerError,
  ) as exc:
    print(f"joint-pose-evaluate: {exc}", file=sys.stderr)
    return 2
  return 0


if __name__ == "__main__":
  raise SystemExit(main())


__all__ = [
  "ChannelNormalization",
  "EvaluationReport",
  "EXPORT_FORMAT",
  "JointPoseEvaluationError",
  "MODE_AXES",
  "ReferenceBodyWindows",
  "channel_mean_mse",
  "constant_velocity_baseline",
  "evaluate_model",
  "evaluate_modes",
  "evaluate_split",
  "export_replay_windows",
  "generate_windows",
  "histogram_distance",
  "knee_flexion_deg",
  "joint_speed_rad_s",
  "main",
  "mode_values",
  "model_parameters_sha256",
  "nearest_reference_window",
  "reference_body_windows",
  "repeat_last_row_baseline",
  "root_yaw_rate_deg_s",
  "rot6d_matrix",
  "yaw_of_matrix",
]
