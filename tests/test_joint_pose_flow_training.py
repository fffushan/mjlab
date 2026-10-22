"""CPU tests for the joint-pose flow runner: budget, mask, loss region, resume.

The sources are tiny synthetic ``[N, 41, 68]`` float32 arrays under ``tmp_path``,
and the model is the small ``width=32, layers=1, heads=2`` variant.  Every run
uses a three-update budget at most, writes only inside ``tmp_path`` and never
touches a GPU or a prepared dataset.  The runner semantics under test are the
ones the plan fixes: level ids on the history rows, the loss region, and the
dataset identity recorded in a run directory.

The tail of the file covers the Stage 4 evaluation metrics: the channel-mean
error and its two baselines, the three mode axes with their histogram distance,
and the body-channel substitution of the replay export.  Each of those is a pure
function of arrays, so a test needs no runner, no checkpoint and no dataset.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from mjlab.tasks.tracking.diffusion.trainer import _history_dropout_mask
from mjlab.tasks.tracking.joint_pose import (
  FlowMatchingRunner,
  FlowMatchingTrainerCfg,
  JointPoseDatasetCfg,
  JointPoseFlowConfigError,
  JointPoseFlowRunnerError,
  JointPoseWindowDataset,
)
from mjlab.tasks.tracking.joint_pose import runner as runner_module
from mjlab.tasks.tracking.joint_pose.evaluate import (
  ChannelNormalization,
  JointPoseEvaluationError,
  ReferenceBodyWindows,
  channel_mean_mse,
  constant_velocity_baseline,
  export_replay_windows,
  histogram_distance,
  joint_speed_rad_s,
  knee_flexion_deg,
  mode_values,
  nearest_reference_window,
  repeat_last_row_baseline,
  root_yaw_rate_deg_s,
  rot6d_matrix,
  yaw_of_matrix,
)

WINDOW_STEPS = 41
PAST_STEPS = 8
CHANNELS = 68
TRAIN_WINDOWS = 64
REFERENCE_WINDOWS = 8
MICRO_BATCH = 8
SMALL_MODEL = {"width": 32, "layers": 1, "heads": 2}


def _windows(count: int, *, seed: int, offset: float) -> np.ndarray:
  rng = np.random.default_rng(seed)
  values = rng.standard_normal((count, WINDOW_STEPS, CHANNELS)).astype(np.float32)
  return (values + np.float32(offset)).astype(np.float32)


@pytest.fixture
def training_root(tmp_path: Path) -> Path:
  """One achieved source and one reference source of synthetic windows."""
  root = tmp_path / "data"
  for relative, count, seed, offset in (
    ("offline/tokens-train.npy", TRAIN_WINDOWS, 0, 0.0),
    ("reference/train.npy", REFERENCE_WINDOWS, 1, 100.0),
  ):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, _windows(count, seed=seed, offset=offset))
  return root


def _dataset(root: Path) -> JointPoseWindowDataset:
  cfg = JointPoseDatasetCfg(
    data_dir=str(root),
    window_steps=WINDOW_STEPS,
    past_steps=PAST_STEPS,
    channels=CHANNELS,
  )
  return JointPoseWindowDataset(cfg)


def _runner(
  dataset: JointPoseWindowDataset, run_dir: Path, **overrides: object
) -> FlowMatchingRunner:
  settings: dict[str, object] = {
    "max_steps": 3,
    "micro_batch": MICRO_BATCH,
    "gradient_accumulation_steps": 1,
    "warmup_updates": 1,
    "checkpoint_every": 2,
    "log_every": 1,
    "mixed_precision": "none",
    "history_dropout_probability": 0.0,
  }
  settings.update(overrides)
  cfg = FlowMatchingTrainerCfg(**settings)  # type: ignore[arg-type]
  return FlowMatchingRunner(dataset, {}, cfg, "cpu", run_dir=run_dir, **SMALL_MODEL)


class _FixedWindows:
  """Dataset stand-in that returns the same windows in a fixed order."""

  def __init__(self, windows: torch.Tensor) -> None:
    self.windows = windows

  def draw_indices(
    self, count: int, generator: torch.Generator | None = None
  ) -> torch.Tensor:
    return torch.arange(count) % self.windows.shape[0]

  def __getitem__(self, index: int) -> torch.Tensor:
    return self.windows[index]


class _FixedPrediction(torch.nn.Module):
  """Model stand-in whose two halves are set by the test, not by the input."""

  def __init__(self, generated: torch.Tensor, history_value: float) -> None:
    super().__init__()
    self.generated = generated
    self.history_value = history_value
    self.future_offset = 0.0

  def forward(self, tokens: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
    history = torch.full_like(tokens[:, :PAST_STEPS], self.history_value)
    return torch.cat((history, self.generated + self.future_offset), dim=1)


def _constant_mask(value: bool):
  def build(
    batch_size: int,
    probability: float,
    *,
    generator: torch.Generator,
    device: torch.device,
  ) -> torch.Tensor:
    return torch.full((batch_size,), value, dtype=torch.bool, device=device)

  return build


def test_require_budget_refuses_missing_max_steps() -> None:
  """A run without an explicit update budget must not start."""
  cfg = FlowMatchingTrainerCfg()
  assert cfg.max_steps is None
  with pytest.raises(JointPoseFlowConfigError, match="max_steps"):
    cfg.require_budget()
  assert FlowMatchingTrainerCfg(max_steps=7, warmup_updates=1).require_budget() == 7
  # Warmup must fit inside the budget, or the base rate is never reached.
  with pytest.raises(JointPoseFlowConfigError, match="warmup_updates"):
    FlowMatchingTrainerCfg(max_steps=7, warmup_updates=7)
  # A single-update run is exempt: there is no schedule to consume.
  assert FlowMatchingTrainerCfg(max_steps=1, warmup_updates=1).require_budget() == 1


def test_history_mask_replaces_only_history_rows(
  training_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """A masked window drops rows 0:8 and leaves rows 8:41 exactly as they were."""
  device = torch.device("cpu")
  generator = torch.Generator(device=device).manual_seed(0)
  every_window = _history_dropout_mask(4, 1.0, generator=generator, device=device)
  assert bool(every_window.all())

  runner = _runner(
    _dataset(training_root), tmp_path / "run", history_dropout_probability=0.25
  )
  clean = torch.randn(4, WINDOW_STEPS, CHANNELS) * 3.0 + 5.0
  state = runner.rng_state()

  monkeypatch.setattr(runner_module, "_history_dropout_mask", _constant_mask(True))
  masked_tokens, masked_ids = runner._noised_batch(clean)
  runner._restore_rng_state(state)
  monkeypatch.setattr(runner_module, "_history_dropout_mask", _constant_mask(False))
  plain_tokens, plain_ids = runner._noised_batch(clean)

  assert torch.equal(plain_tokens[:, :PAST_STEPS], clean[:, :PAST_STEPS])
  assert not torch.allclose(masked_tokens[:, :PAST_STEPS], clean[:, :PAST_STEPS])
  assert torch.equal(masked_tokens[:, PAST_STEPS:], plain_tokens[:, PAST_STEPS:])
  assert torch.equal(masked_ids[:, PAST_STEPS:], plain_ids[:, PAST_STEPS:])
  assert bool((masked_ids[:, :PAST_STEPS] == runner.process.max_level).all())
  assert bool((plain_ids[:, :PAST_STEPS] == 0).all())


def test_loss_ignores_history_rows(training_root: Path, tmp_path: Path) -> None:
  """Rows 0:8 never enter the reported loss, so their error cannot change it."""
  runner = _runner(
    _dataset(training_root), tmp_path / "run", max_steps=1, micro_batch=4
  )
  windows = torch.randn(4, WINDOW_STEPS, CHANNELS)
  runner.dataset = _FixedWindows(windows)

  prediction = _FixedPrediction(windows[:, PAST_STEPS:], history_value=1.0e6)
  runner.model = prediction
  assert float(runner._microbatch_loss()) == 0.0

  prediction.future_offset = 0.5
  assert float(runner._microbatch_loss()) == pytest.approx(0.25)


def test_runner_smoke_three_updates(training_root: Path, tmp_path: Path) -> None:
  """Three CPU updates train, checkpoint and resume at the same parameters."""
  dataset = _dataset(training_root)
  run_dir = tmp_path / "run"
  runner = _runner(dataset, run_dir)
  before = {
    name: value.detach().clone() for name, value in runner.model.state_dict().items()
  }

  summary = runner.learn()

  assert summary["global_step"] == 3
  assert summary["max_steps"] == 3
  final_loss = summary["final_loss"]
  assert isinstance(final_loss, float)
  assert math.isfinite(final_loss)
  assert final_loss > 0.0
  trained = runner.model.state_dict()
  changed = [
    name for name, value in before.items() if not torch.equal(value, trained[name])
  ]
  assert changed

  checkpoint = run_dir / "model_0000003.pt"
  assert checkpoint.is_file()
  assert (run_dir / "params" / "dataset.json").is_file()
  assert (run_dir / "metrics.jsonl").is_file()

  resumed = _runner(dataset, run_dir)
  assert resumed.global_step == 3
  reloaded = resumed.model.state_dict()
  for name, value in trained.items():
    assert torch.equal(value, reloaded[name])


def test_resume_refuses_changed_dataset_identity(
  training_root: Path, tmp_path: Path
) -> None:
  """A run directory and a checkpoint both reject another dataset identity."""
  dataset = _dataset(training_root)
  run_dir = tmp_path / "run"
  _runner(dataset, run_dir, max_steps=2).learn()
  checkpoint = run_dir / "model_0000002.pt"
  assert checkpoint.is_file()

  source = training_root / "offline" / "tokens-train.npy"
  staged = source.with_name("tokens-train-staged.npy")
  np.save(staged, _windows(TRAIN_WINDOWS + 16, seed=2, offset=0.0))
  os.replace(staged, source)

  changed = _dataset(training_root)
  assert changed.dataset_identity["data_dir"] == dataset.dataset_identity["data_dir"]
  assert (
    changed.dataset_identity["window_count"] != dataset.dataset_identity["window_count"]
  )
  with pytest.raises(JointPoseFlowRunnerError, match="another dataset"):
    _runner(changed, run_dir, max_steps=2)

  other = _runner(changed, tmp_path / "other-run", max_steps=2)
  with pytest.raises(JointPoseFlowRunnerError, match="another dataset"):
    other.load(checkpoint)


# --------------------------------------------------------------------- Stage 4
# Evaluation metrics.  Every function below is a pure function of arrays, so the
# tests build a window whose channel content is known and check the number that
# comes back.

JOINT_CHANNELS = slice(6, 68)


def _mode_windows(
  count: int = 2,
  *,
  yaw_per_row_deg: float = 1.0,
  knee_rad: float = 0.4,
  joint_speed: float = 0.2,
) -> np.ndarray:
  """Windows with one constant yaw rate, one knee angle and one joint speed."""
  windows = np.zeros((count, WINDOW_STEPS, CHANNELS), dtype=np.float64)
  for row in range(WINDOW_STEPS):
    yaw = math.radians(yaw_per_row_deg) * row
    cosine, sine = math.cos(yaw), math.sin(yaw)
    # First rotation-matrix column then second: Rz(yaw) in the packed 6D order.
    windows[:, row, 0:6] = (cosine, sine, 0.0, -sine, cosine, 0.0)
  windows[:, :, 9] = knee_rad
  windows[:, :, 15] = knee_rad
  windows[:, :, 37:68] = joint_speed
  return windows


def _reference_body(count: int, fill: float = 1.0) -> ReferenceBodyWindows:
  """Body kinematics whose position value identifies the reference window."""
  offset = np.arange(count, dtype=np.float32).reshape(-1, 1, 1, 1)
  position = np.full((count, WINDOW_STEPS, 6, 3), fill, dtype=np.float32) + offset
  quaternion = np.zeros((count, WINDOW_STEPS, 6, 4), dtype=np.float32)
  quaternion[..., 0] = 1.0
  velocity = np.zeros((count, WINDOW_STEPS, 6, 3), dtype=np.float32)
  return ReferenceBodyWindows(
    body_pos_w=position,
    body_quat_w=quaternion,
    body_lin_vel_w=velocity,
    body_ang_vel_w=np.zeros_like(velocity),
    fps=50.0,
  )


def test_channel_mean_mse_averages_every_channel() -> None:
  """The error is the mean over all channels and rows, as in the training loss."""
  target = np.zeros((2, 3, CHANNELS))
  assert channel_mean_mse(target + 0.5, target) == pytest.approx(0.25)
  assert channel_mean_mse(target, target) == 0.0
  with pytest.raises(JointPoseEvaluationError, match="does not match"):
    channel_mean_mse(np.zeros((2, 3, 4)), target)


def test_repeat_last_row_baseline_holds_the_last_history_row() -> None:
  """Every generated row repeats history row 7 and the history is untouched."""
  windows = _mode_windows(3)
  windows[:, PAST_STEPS:, 20] = 7.0
  predicted = repeat_last_row_baseline(windows, past_steps=PAST_STEPS)
  assert np.array_equal(predicted[:, :PAST_STEPS], windows[:, :PAST_STEPS])
  held = windows[:, PAST_STEPS - 1, 20][:, None]
  assert np.all(predicted[:, PAST_STEPS:, 20] == held)
  assert np.all(held == 0.0)
  with pytest.raises(JointPoseEvaluationError, match="past_steps"):
    repeat_last_row_baseline(windows, past_steps=0)


def test_constant_velocity_baseline_continues_at_the_last_step() -> None:
  """Positions advance by the last history difference; velocities hold."""
  windows = _mode_windows(2)
  windows[:, :, 20] = np.arange(WINDOW_STEPS) * 0.1
  predicted = constant_velocity_baseline(windows, past_steps=PAST_STEPS)
  future = WINDOW_STEPS - PAST_STEPS
  expected = windows[:, PAST_STEPS - 1, 20][:, None] + 0.1 * np.arange(1, future + 1)
  assert np.allclose(predicted[:, PAST_STEPS:, 20], expected)
  assert np.allclose(predicted[:, PAST_STEPS:, 37:68], 0.2)
  assert np.array_equal(predicted[:, :PAST_STEPS], windows[:, :PAST_STEPS])
  with pytest.raises(JointPoseEvaluationError, match="past_steps"):
    constant_velocity_baseline(windows, past_steps=1)


def test_rot6d_matrix_restores_a_rotation_from_scaled_columns() -> None:
  """Gram-Schmidt repairs the columns the z-score transform distorted."""
  windows = _mode_windows(1, yaw_per_row_deg=10.0)
  packed = windows[0, 5, 0:6].copy()
  packed[0:3] *= 4.0
  packed[3] += 0.6
  matrix = rot6d_matrix(packed)
  assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-6)
  assert yaw_of_matrix(matrix) == pytest.approx(math.radians(50.0), abs=1e-6)
  with pytest.raises(JointPoseEvaluationError, match="6 values"):
    rot6d_matrix(np.zeros(5))


def test_root_yaw_rate_reads_a_constant_turn() -> None:
  """One degree per frame at 50 Hz is 50 degrees per second."""
  windows = _mode_windows(2, yaw_per_row_deg=1.0)
  rate = root_yaw_rate_deg_s(windows, past_steps=PAST_STEPS, fps=50.0)
  assert rate == pytest.approx([50.0, 50.0])


def test_mode_values_report_knee_angle_and_joint_speed() -> None:
  """The knee axis is in degrees and the speed axis in radians per second."""
  windows = _mode_windows(2, knee_rad=0.4, joint_speed=0.2)
  modes = mode_values(windows, past_steps=PAST_STEPS, fps=50.0)
  assert modes["knee_flexion"] == pytest.approx([math.degrees(0.4)] * 2)
  assert modes["joint_speed"] == pytest.approx([0.2] * 2)
  assert modes["root_yaw_rate"] == pytest.approx([50.0] * 2)
  assert knee_flexion_deg(windows, past_steps=PAST_STEPS).shape == (2,)
  assert joint_speed_rad_s(windows, past_steps=PAST_STEPS).shape == (2,)


def test_histogram_distance_separates_identical_and_disjoint_samples() -> None:
  """Total variation is 0 for one distribution and 1 for disjoint supports."""
  same = histogram_distance(np.linspace(0.0, 1.0, 50), np.linspace(0.0, 1.0, 50))
  assert same["total_variation"] == 0.0
  apart = histogram_distance(np.zeros(40), np.full(40, 100.0))
  assert apart["total_variation"] == 1.0
  shifted = histogram_distance(
    np.zeros(100), np.concatenate((np.zeros(50), np.full(50, 100.0)))
  )
  assert shifted["total_variation"] == pytest.approx(0.5)
  with pytest.raises(JointPoseEvaluationError, match="non-finite"):
    histogram_distance(np.array([0.0, np.nan]), np.zeros(3))


def test_nearest_reference_window_matches_joint_content() -> None:
  """Each generated window selects the reference window nearest in the joints."""
  windows = _mode_windows(3)
  windows[1, :, JOINT_CHANNELS] += 3.0
  windows[2, :, JOINT_CHANNELS] -= 3.0
  index, distance = nearest_reference_window(windows, windows)
  assert index.tolist() == [0, 1, 2]
  assert np.allclose(distance, 0.0)
  index, distance = nearest_reference_window(windows[2:3], windows)
  assert index.tolist() == [2]
  assert distance[0] == pytest.approx(0.0)
  with pytest.raises(JointPoseEvaluationError, match="share a layout"):
    nearest_reference_window(windows[:, : WINDOW_STEPS - 1], windows)


def test_export_replay_windows_borrows_body_channels_from_the_reference(
  tmp_path: Path,
) -> None:
  """Exported joints are generated; exported body kinematics are the reference."""
  normalization = ChannelNormalization(
    mean=np.full(CHANNELS, 0.5), std=np.full(CHANNELS, 2.0), source="unit-test"
  )
  reference = np.concatenate(
    (_mode_windows(1, knee_rad=0.4), _mode_windows(1, knee_rad=0.9)), axis=0
  )
  body = _reference_body(2, fill=1.0)
  generated = reference[::-1].copy()

  manifest = export_replay_windows(
    generated=generated,
    reference=reference,
    reference_body=body,
    normalization=normalization,
    out_dir=tmp_path / "replay",
  )

  assert manifest["windows"] == 2
  assert [entry["reference_window"] for entry in manifest["files"]] == [1, 0]
  assert "nearest reference clip frame" in manifest["body_source"]
  assert "no root translation" in manifest["body_source"]
  stored = json.loads((tmp_path / "replay" / "manifest.json").read_text())
  assert stored["files"] == manifest["files"]

  with np.load(tmp_path / "replay" / "gen000000.npz", allow_pickle=False) as payload:
    assert sorted(payload.files) == sorted(
      (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "fps",
      )
    )
    assert payload["joint_pos"].shape == (WINDOW_STEPS, 31)
    assert payload["joint_pos"].dtype == np.float32
    expected_joint = generated[0, :, 6:37] * 2.0 + 0.5
    assert np.allclose(payload["joint_pos"], expected_joint)
    # Row 0 matched reference window 1, whose body position fill is 1 + 1.
    assert payload["body_pos_w"].shape == (WINDOW_STEPS, 6, 3)
    assert float(payload["body_pos_w"][0, 0, 0]) == pytest.approx(2.0)
    assert float(payload["fps"][0]) == pytest.approx(50.0)

  with pytest.raises(JointPoseEvaluationError, match="body windows"):
    export_replay_windows(
      generated=generated,
      reference=reference,
      reference_body=_reference_body(1),
      normalization=normalization,
      out_dir=tmp_path / "refused",
    )


def test_resume_continues_identically(tmp_path: Path, training_root: Path) -> None:
  """A 2+2 run resumed from a checkpoint must match an uninterrupted 4-update run."""
  dataset = _dataset(training_root)
  straight_dir = tmp_path / "straight"
  torch.manual_seed(0)
  straight = _runner(dataset, straight_dir, max_steps=4, checkpoint_every=2)
  straight.learn()
  expected = {k: v.clone() for k, v in straight.model.state_dict().items()}
  checkpoints = sorted(straight_dir.glob("model_*.pt"))
  assert len(checkpoints) >= 2, "expected a mid-run and a final checkpoint"

  # Continue from the mid-run checkpoint under the same budget: the learning-rate
  # schedule is planned over the budget, so a different budget is a different run.
  torch.manual_seed(0)
  resumed = _runner(dataset, tmp_path / "resumed", max_steps=4, checkpoint_every=2)
  resumed.load(checkpoints[0])
  assert resumed.global_step < 4
  resumed.learn()

  for name, value in resumed.model.state_dict().items():
    assert torch.equal(value, expected[name]), f"{name} diverged after resume"


def test_sample_preserves_history_and_honours_parameters(
  tmp_path: Path, training_root: Path
) -> None:
  """Sampling returns a full window, keeps history, and reads the chosen weights."""
  dataset = _dataset(training_root)
  runner = _runner(dataset, tmp_path / "run", max_steps=2, checkpoint_every=1)
  runner.learn()
  history = torch.zeros(3, PAST_STEPS, CHANNELS)
  generated = runner.sample(history, 2)
  assert generated.shape == (3, WINDOW_STEPS, CHANNELS)
  assert torch.equal(generated[:, :PAST_STEPS], history)

  with_ema = runner.sample(history, 2, parameters="ema")
  with_model = runner.sample(history, 2, parameters="model")
  assert with_ema.shape == with_model.shape
  assert not torch.equal(with_ema, with_model), "EMA and live weights must differ"
