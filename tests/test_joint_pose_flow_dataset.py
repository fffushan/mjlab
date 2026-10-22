"""CPU tests for the latent-free joint-pose flow dataset and its configuration.

Every source is a tiny synthetic ``[N, 41, 68]`` float32 array written under
``tmp_path``.  No prepared dataset, NAS path or GPU is touched, and no training
run starts.  The two sources are separated by a large constant offset, so a
window's origin is visible in its values as well as in ``source_of``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mjlab.tasks.tracking.joint_pose import (
  JointPoseDatasetCfg,
  JointPoseDatasetError,
  JointPoseFlowConfigError,
  JointPoseWindowDataset,
)

WINDOW_STEPS = 41
PAST_STEPS = 8
CHANNELS = 68
ACHIEVED_WINDOWS = 64
REFERENCE_WINDOWS = 8
REFERENCE_OFFSET = 100.0
REFERENCE_WEIGHT = 30.0
SAMPLING_DRAWS = 20_000


def _windows(count: int, *, seed: int, offset: float) -> np.ndarray:
  rng = np.random.default_rng(seed)
  values = rng.standard_normal((count, WINDOW_STEPS, CHANNELS)).astype(np.float32)
  return (values + np.float32(offset)).astype(np.float32)


def _write_source(path: Path, windows: np.ndarray) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  np.save(path, windows)


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
  """Two tiny window sources, plus the achieved cache record."""
  root = tmp_path / "data"
  _write_source(
    root / "offline" / "tokens-train.npy",
    _windows(ACHIEVED_WINDOWS, seed=0, offset=0.0),
  )
  _write_source(
    root / "reference" / "train.npy",
    _windows(REFERENCE_WINDOWS, seed=1, offset=REFERENCE_OFFSET),
  )
  (root / "offline" / "tokens-train.json").write_text(
    json.dumps({"bytes": 4096, "segments": 2}), encoding="utf-8"
  )
  return root


def _cfg(root: Path, **overrides: object) -> JointPoseDatasetCfg:
  settings: dict[str, object] = {
    "data_dir": str(root),
    "window_steps": WINDOW_STEPS,
    "past_steps": PAST_STEPS,
    "channels": CHANNELS,
    "reference_weight": REFERENCE_WEIGHT,
  }
  settings.update(overrides)
  return JointPoseDatasetCfg(**settings)  # type: ignore[arg-type]


def test_config_refuses_shape_mismatch() -> None:
  """Channel arithmetic and window arithmetic are both checked."""
  with pytest.raises(JointPoseFlowConfigError, match="channels"):
    JointPoseDatasetCfg(channels=67)
  with pytest.raises(JointPoseFlowConfigError, match="generated rows"):
    JointPoseDatasetCfg(window_steps=8, past_steps=8)
  cfg = _cfg(Path("/tmp/joint-pose-flow-unused"))
  # generated rows are [past_steps, window_steps): 33 of 41, not 32.
  assert cfg.future_steps == WINDOW_STEPS - PAST_STEPS


def test_synthetic_dataset_shape(dataset_root: Path) -> None:
  """The window count, dtype and per-window shape match the two sources."""
  dataset = JointPoseWindowDataset(_cfg(dataset_root))
  assert len(dataset) == ACHIEVED_WINDOWS + REFERENCE_WINDOWS
  assert dataset.n_achieved == ACHIEVED_WINDOWS
  assert dataset.n_reference == REFERENCE_WINDOWS
  assert dataset.current_index == PAST_STEPS

  achieved = dataset[0]
  reference = dataset[dataset.n_achieved]
  for window in (achieved, reference):
    assert isinstance(window, torch.Tensor)
    assert window.shape == (WINDOW_STEPS, CHANNELS)
    assert window.dtype == torch.float32

  assert dataset.source_of(0) == "achieved"
  assert dataset.source_of(len(dataset) - 1) == "reference"
  expected_achieved = _windows(ACHIEVED_WINDOWS, seed=0, offset=0.0)
  expected_reference = _windows(REFERENCE_WINDOWS, seed=1, offset=REFERENCE_OFFSET)
  assert torch.equal(achieved, torch.from_numpy(expected_achieved[0]))
  assert torch.equal(reference, torch.from_numpy(expected_reference[0]))


def test_source_sampling_matches_reference_share(dataset_root: Path) -> None:
  """A fixed generator reproduces the reference share of the draw."""
  dataset = JointPoseWindowDataset(_cfg(dataset_root))
  expected = (
    REFERENCE_WEIGHT
    * REFERENCE_WINDOWS
    / (REFERENCE_WEIGHT * REFERENCE_WINDOWS + ACHIEVED_WINDOWS)
  )
  assert dataset.reference_share == pytest.approx(expected)
  assert dataset.reference_weight == REFERENCE_WEIGHT

  generator = torch.Generator().manual_seed(7)
  indices = dataset.draw_indices(SAMPLING_DRAWS, generator)
  assert indices.shape == (SAMPLING_DRAWS,)
  assert int(indices.min()) >= 0
  assert int(indices.max()) < len(dataset)
  reference_draws = float((indices >= dataset.n_achieved).double().mean())
  assert reference_draws == pytest.approx(expected, abs=0.02)


def test_split_leak_is_refused(dataset_root: Path) -> None:
  """A reference motion assigned to another split stops the dataset."""
  (dataset_root / "assignments.json").write_text(
    json.dumps(
      {
        "policy": "motion",
        "motions": {"walk": "train", "idle": "train", "run": "test"},
      }
    ),
    encoding="utf-8",
  )
  manifest_path = dataset_root / "reference" / "manifest.json"
  manifest_path.write_text(
    json.dumps(
      {
        "format": "joint_pose_reference_v1",
        "sources": [
          {"motion": "walk", "split": "train"},
          {"motion": "run", "split": "train"},
        ],
      }
    ),
    encoding="utf-8",
  )
  with pytest.raises(JointPoseDatasetError, match="crosses the split"):
    JointPoseWindowDataset(_cfg(dataset_root))

  manifest_path.write_text(
    json.dumps(
      {
        "format": "joint_pose_reference_v1",
        "sources": [
          {"motion": "walk", "split": "train"},
          {"motion": "idle", "split": "train"},
        ],
      }
    ),
    encoding="utf-8",
  )
  agree = JointPoseWindowDataset(_cfg(dataset_root))
  assert len(agree) == ACHIEVED_WINDOWS + REFERENCE_WINDOWS
  assert agree.dataset_identity["split_policy"] == "motion"
  assert agree.dataset_identity["reference_source_count"] == 2


def test_identity_detects_rewritten_content(dataset_root: Path) -> None:
  """A cache rewritten in place with the same shape must change the identity.

  Counts and file sizes cannot see this; the sampled content digest must.
  """
  cfg = _cfg(dataset_root)
  before = JointPoseWindowDataset(cfg).identity_sha256()
  path = dataset_root / "offline" / "tokens-train.npy"
  array = np.load(path)
  array[0, 0, 0] = np.float32(array[0, 0, 0] + 1.0)
  np.save(path, array)
  after = JointPoseWindowDataset(cfg).identity_sha256()
  assert after != before
