"""Tests for the X2 tennis-endpoint recovery state pool.

Synthetic tests verify validation, split determinism, sampling, and error
paths. Real-data diagnostic tests (skipped when the dataset is absent)
establish the actual dataset contract for downstream stages: all-body FK
ordering, velocity convention, foot-sphere clearance geometry, and
whole-inventory reproducibility.

All tests follow AGENTS.md: function-style, no test classes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mjlab.tasks.velocity.mdp.tennis_endpoint_pool import (
  EndpointPool,
  _trajectory_split,
)

# ---------------------------------------------------------------------------
# Helpers for synthetic data.
# ---------------------------------------------------------------------------

NUM_JOINTS = 31
NUM_BODIES = 32


def _make_synthetic_npz(
  path: Path,
  num_frames: int = 50,
  fps: float = 50.0,
  joint_pos=None,
  joint_vel=None,
  body_pos_w=None,
  body_quat_w=None,
  body_lin_vel_w=None,
  body_ang_vel_w=None,
  seed: int = 0,
) -> None:
  """Write a structurally valid synthetic NPZ for testing."""
  rng = np.random.default_rng(seed)
  if joint_pos is None:
    joint_pos = rng.uniform(-1, 1, (num_frames, NUM_JOINTS)).astype(np.float32)
  if joint_vel is None:
    joint_vel = rng.uniform(-5, 5, (num_frames, NUM_JOINTS)).astype(np.float32)
  if body_pos_w is None:
    body_pos_w = rng.uniform(-5, 5, (num_frames, NUM_BODIES, 3)).astype(np.float32)
    body_pos_w[:, 0, 2] = np.abs(body_pos_w[:, 0, 2]) + 0.5
  if body_quat_w is None:
    body_quat_w = rng.standard_normal((num_frames, NUM_BODIES, 4)).astype(np.float32)
    body_quat_w[:, :, 0] += 2.0
    body_quat_w /= np.linalg.norm(body_quat_w, axis=-1, keepdims=True)
  if body_lin_vel_w is None:
    body_lin_vel_w = rng.uniform(-2, 2, (num_frames, NUM_BODIES, 3)).astype(np.float32)
  if body_ang_vel_w is None:
    body_ang_vel_w = rng.uniform(-3, 3, (num_frames, NUM_BODIES, 3)).astype(np.float32)
  np.savez(
    path,
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    body_pos_w=body_pos_w,
    body_quat_w=body_quat_w,
    body_lin_vel_w=body_lin_vel_w,
    body_ang_vel_w=body_ang_vel_w,
    fps=np.array([fps], dtype=np.float32),
  )


def _make_dataset_dir(
  tmp_path: Path,
  num_files: int = 10,
  frames_per_file: int = 50,
  prefix: str = "syn_",
) -> Path:
  """Create a directory of synthetic NPZ files."""
  d = tmp_path / "tennis_data"
  d.mkdir()
  for i in range(num_files):
    _make_synthetic_npz(d / f"{prefix}{i:03d}.npz", num_frames=frames_per_file, seed=i)
  return d


# ---------------------------------------------------------------------------
# Real-data fixtures.
# ---------------------------------------------------------------------------

TENNIS_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "tennis"


@pytest.fixture(scope="module")
def real_pool_all():
  """Load the full real dataset once for all real-data tests."""
  if not TENNIS_DATA_DIR.is_dir():
    pytest.skip(f"Tennis dataset not found at {TENNIS_DATA_DIR}")
  return EndpointPool.from_directory(TENNIS_DATA_DIR, split="all")


@pytest.fixture(scope="module")
def x2_model():
  """Compile the X2 MuJoCo model once."""
  from mjlab.asset_zoo.robots.agibot_x2.x2_constants import get_spec

  return get_spec().compile()


@pytest.fixture(scope="module")
def x2_model_and_names(x2_model):
  """Return (model, joint_names, body_names) from the X2 asset."""
  joint_names = tuple(x2_model.joint(i).name for i in range(1, x2_model.njnt))
  body_names = tuple(x2_model.body(i).name for i in range(1, x2_model.nbody))
  return x2_model, joint_names, body_names


# ---------------------------------------------------------------------------
# Trajectory split tests.
# ---------------------------------------------------------------------------


def test_trajectory_split_deterministic():
  t1, v1 = _trajectory_split(144, 0.2, 42)
  t2, v2 = _trajectory_split(144, 0.2, 42)
  assert t1 == t2
  assert v1 == v2


def test_trajectory_split_no_leakage():
  train_ids, val_ids = _trajectory_split(144, 0.2, 42)
  assert set(train_ids).isdisjoint(set(val_ids))
  assert set(train_ids) | set(val_ids) == set(range(144))


def test_trajectory_split_counts():
  train_ids, val_ids = _trajectory_split(144, 0.2, 42)
  # ceil(0.2 * 144) = ceil(28.8) = 29 validation, 115 train.
  assert len(val_ids) == 29
  assert len(train_ids) == 115


def test_trajectory_split_small_dataset():
  """Tiny dataset: 3 files, 0.2 fraction -> ceil(0.6) = 1 validation."""
  train_ids, val_ids = _trajectory_split(3, 0.2, 42)
  assert len(val_ids) == 1
  assert len(train_ids) == 2
  assert set(train_ids).isdisjoint(set(val_ids))


# ---------------------------------------------------------------------------
# Construction and array shape tests.
# ---------------------------------------------------------------------------


def test_from_directory_basic_shapes(tmp_path):
  d = _make_dataset_dir(tmp_path, num_files=10, frames_per_file=50)
  pool = EndpointPool.from_directory(d, last_n_frames=10, split="all")
  assert len(pool) == 100
  assert pool.joint_pos.shape == (100, NUM_JOINTS)
  assert pool.joint_vel.shape == (100, NUM_JOINTS)
  assert pool.root_pos_w.shape == (100, 3)
  assert pool.root_quat_w.shape == (100, 4)
  assert pool.root_lin_vel_w.shape == (100, 3)
  assert pool.root_ang_vel_w.shape == (100, 3)
  assert pool.trajectory_ids.shape == (100,)
  assert pool.frame_indices.shape == (100,)


def test_from_directory_last_n_frames_window(tmp_path):
  """Verify exactly the last N frames are taken, in chronological order."""
  d = _make_dataset_dir(tmp_path, num_files=2, frames_per_file=30)
  pool = EndpointPool.from_directory(d, last_n_frames=10, split="all")

  data = np.load(d / "syn_000.npz")
  expected_jp = data["joint_pos"][-10:, :]
  np.testing.assert_allclose(pool.joint_pos[:10], expected_jp, atol=1e-6)
  np.testing.assert_allclose(
    pool.root_pos_w[:10], data["body_pos_w"][-10:, 0], atol=1e-6
  )

  np.testing.assert_array_equal(pool.frame_indices[:10], np.arange(20, 30))
  np.testing.assert_array_equal(pool.frame_indices[10:], np.arange(20, 30))

  assert np.all(pool.trajectory_ids[:10] == 0)
  assert np.all(pool.trajectory_ids[10:] == 1)


def test_from_directory_last_n_custom(tmp_path):
  d = _make_dataset_dir(tmp_path, num_files=3, frames_per_file=40)
  pool = EndpointPool.from_directory(d, last_n_frames=5, split="all")
  assert len(pool) == 15
  np.testing.assert_array_equal(pool.frame_indices[:5], np.arange(35, 40))


# ---------------------------------------------------------------------------
# Split and sampling tests.
# ---------------------------------------------------------------------------


def test_split_train_vs_validation(tmp_path):
  d = _make_dataset_dir(tmp_path, num_files=10, frames_per_file=50)
  train_pool = EndpointPool.from_directory(d, last_n_frames=10, split="train", seed=42)
  val_pool = EndpointPool.from_directory(
    d, last_n_frames=10, split="validation", seed=42
  )
  all_pool = EndpointPool.from_directory(d, last_n_frames=10, split="all")

  train_tids = set(train_pool.trajectory_ids.tolist())
  val_tids = set(val_pool.trajectory_ids.tolist())
  assert train_tids.isdisjoint(val_tids)
  assert train_tids | val_tids == set(all_pool.trajectory_ids.tolist())

  assert len(val_pool.trajectory_ids) == 20
  assert len(train_pool.trajectory_ids) == 80


def test_split_deterministic_across_constructions(tmp_path):
  d = _make_dataset_dir(tmp_path, num_files=10, frames_per_file=50)
  p1 = EndpointPool.from_directory(d, last_n_frames=10, split="train", seed=42)
  p2 = EndpointPool.from_directory(d, last_n_frames=10, split="train", seed=42)
  np.testing.assert_array_equal(p1.trajectory_ids, p2.trajectory_ids)
  np.testing.assert_array_equal(p1.joint_pos, p2.joint_pos)


def test_sample_indices_uniform(tmp_path):
  d = _make_dataset_dir(tmp_path, num_files=5, frames_per_file=20)
  pool = EndpointPool.from_directory(d, last_n_frames=10, split="all")
  rng = np.random.default_rng(123)
  idx = pool.sample_indices(1000, rng)
  assert idx.shape == (1000,)
  assert idx.min() >= 0
  assert idx.max() < len(pool)
  assert len(set(idx.tolist())) == len(pool)


def test_sample_indices_reproducible(tmp_path):
  d = _make_dataset_dir(tmp_path, num_files=5, frames_per_file=20)
  pool = EndpointPool.from_directory(d, last_n_frames=10, split="all")
  rng1 = np.random.default_rng(99)
  rng2 = np.random.default_rng(99)
  idx1 = pool.sample_indices(100, rng1)
  idx2 = pool.sample_indices(100, rng2)
  np.testing.assert_array_equal(idx1, idx2)


def test_sample_indices_type_validation(tmp_path):
  d = _make_dataset_dir(tmp_path, num_files=2, frames_per_file=20)
  pool = EndpointPool.from_directory(d, last_n_frames=10, split="all")
  rng = np.random.default_rng(0)
  with pytest.raises(ValueError, match="count must be an int"):
    pool.sample_indices(5.0, rng)  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="count must be non-negative"):
    pool.sample_indices(-1, rng)


# ---------------------------------------------------------------------------
# Manifest tests.
# ---------------------------------------------------------------------------


def test_manifest_serializable_and_complete(tmp_path):
  d = _make_dataset_dir(tmp_path, num_files=3, frames_per_file=30)
  pool = EndpointPool.from_directory(d, last_n_frames=10, split="all")
  m = pool.manifest
  json.dumps(m)
  assert m["split"] == "all"
  assert m["num_files_total"] == 3
  assert m["num_files_selected"] == 3
  assert m["num_states"] == 30
  assert m["last_n_frames"] == 10
  assert m["fps"] == 50.0
  # Full inventory includes ALL files, not just selected.
  assert len(m["sources"]) == 3
  for src in m["sources"]:
    assert "sha256" in src and src["sha256"] != ""
    assert src["frame_start"] == 20
    assert src["frame_end"] == 29
    assert src["num_frames"] == 30
  assert len(m["joint_names"]) == NUM_JOINTS
  assert len(m["body_names"]) == NUM_BODIES


def test_manifest_full_inventory_in_every_split(tmp_path):
  """Manifest in train/validation splits must inventory ALL source files."""
  d = _make_dataset_dir(tmp_path, num_files=10, frames_per_file=50)
  train_pool = EndpointPool.from_directory(d, split="train", seed=42)
  val_pool = EndpointPool.from_directory(d, split="validation", seed=42)
  all_pool = EndpointPool.from_directory(d, split="all")

  # All splits must have full source inventory.
  assert len(train_pool.manifest["sources"]) == 10
  assert len(val_pool.manifest["sources"]) == 10
  assert len(all_pool.manifest["sources"]) == 10

  # Source inventory identity is the same across splits.
  train_src = {s["filename"]: s["sha256"] for s in train_pool.manifest["sources"]}
  val_src = {s["filename"]: s["sha256"] for s in val_pool.manifest["sources"]}
  all_src = {s["filename"]: s["sha256"] for s in all_pool.manifest["sources"]}
  assert train_src == val_src == all_src


def test_manifest_split_consistency(tmp_path):
  d = _make_dataset_dir(tmp_path, num_files=10, frames_per_file=50)
  train_pool = EndpointPool.from_directory(d, split="train", seed=42)
  val_pool = EndpointPool.from_directory(d, split="validation", seed=42)
  assert set(train_pool.manifest["train_trajectory_ids"]) == set(
    train_pool.manifest["selected_trajectory_ids"]
  )
  assert set(val_pool.manifest["validation_trajectory_ids"]) == set(
    val_pool.manifest["selected_trajectory_ids"]
  )
  assert set(train_pool.manifest["selected_trajectory_ids"]).isdisjoint(
    set(val_pool.manifest["selected_trajectory_ids"])
  )


# ---------------------------------------------------------------------------
# Immutability tests.
# ---------------------------------------------------------------------------


def test_arrays_are_readonly(tmp_path):
  d = _make_dataset_dir(tmp_path, num_files=2, frames_per_file=20)
  pool = EndpointPool.from_directory(d, last_n_frames=10, split="all")
  with pytest.raises(ValueError, match="read-only"):
    pool.joint_pos[0, 0] = 99.0
  with pytest.raises(ValueError, match="read-only"):
    pool.root_pos_w[0, 0] = 99.0


# ---------------------------------------------------------------------------
# Error path tests.
# ---------------------------------------------------------------------------


def test_empty_directory_raises(tmp_path):
  d = tmp_path / "empty"
  d.mkdir()
  with pytest.raises(ValueError, match="No .npz files"):
    EndpointPool.from_directory(d)


def test_nonexistent_directory_raises(tmp_path):
  with pytest.raises(ValueError, match="Directory does not exist"):
    EndpointPool.from_directory(tmp_path / "nonexistent")


def test_short_trajectory_raises(tmp_path):
  d = _make_dataset_dir(tmp_path, num_files=2, frames_per_file=5)
  with pytest.raises(ValueError, match="need at least 10"):
    EndpointPool.from_directory(d, last_n_frames=10, split="all")


def test_invalid_last_n_frames_type(tmp_path):
  d = _make_dataset_dir(tmp_path, num_files=2, frames_per_file=20)
  with pytest.raises(ValueError, match="last_n_frames must be an int"):
    EndpointPool.from_directory(d, last_n_frames=10.0)  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="last_n_frames must be positive"):
    EndpointPool.from_directory(d, last_n_frames=0)
  with pytest.raises(ValueError, match="last_n_frames must be positive"):
    EndpointPool.from_directory(d, last_n_frames=-1)


def test_invalid_last_n_frames_bool(tmp_path):
  d = _make_dataset_dir(tmp_path, num_files=2, frames_per_file=20)
  with pytest.raises(ValueError, match="last_n_frames must be an int"):
    EndpointPool.from_directory(d, last_n_frames=True)


def test_invalid_validation_fraction(tmp_path):
  d = _make_dataset_dir(tmp_path, num_files=2, frames_per_file=20)
  with pytest.raises(ValueError, match="validation_fraction"):
    EndpointPool.from_directory(d, validation_fraction=0.0)
  with pytest.raises(ValueError, match="validation_fraction"):
    EndpointPool.from_directory(d, validation_fraction=1.0)


def test_invalid_split(tmp_path):
  d = _make_dataset_dir(tmp_path, num_files=2, frames_per_file=20)
  with pytest.raises(ValueError, match="split must be"):
    EndpointPool.from_directory(d, split="test")


def test_missing_keys_raises(tmp_path):
  d = tmp_path / "bad_data"
  d.mkdir()
  rng = np.random.default_rng(0)
  np.savez(
    d / "bad_000.npz",
    joint_pos=rng.uniform(-1, 1, (20, NUM_JOINTS)).astype(np.float32),
    joint_vel=rng.uniform(-1, 1, (20, NUM_JOINTS)).astype(np.float32),
    body_pos_w=rng.uniform(-1, 1, (20, NUM_BODIES, 3)).astype(np.float32),
    body_lin_vel_w=rng.uniform(-1, 1, (20, NUM_BODIES, 3)).astype(np.float32),
    body_ang_vel_w=rng.uniform(-1, 1, (20, NUM_BODIES, 3)).astype(np.float32),
    fps=np.array([50.0], dtype=np.float32),
  )
  with pytest.raises(ValueError, match="missing required NPZ keys"):
    EndpointPool.from_directory(d, last_n_frames=10, split="all")


def test_wrong_shape_raises(tmp_path):
  d = tmp_path / "bad_shape"
  d.mkdir()
  rng = np.random.default_rng(0)
  quat = rng.standard_normal((20, NUM_BODIES, 4)).astype(np.float32)
  quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
  np.savez(
    d / "bad_000.npz",
    joint_pos=rng.uniform(-1, 1, (20, 30)).astype(np.float32),  # wrong count
    joint_vel=rng.uniform(-1, 1, (20, NUM_JOINTS)).astype(np.float32),
    body_pos_w=rng.uniform(-1, 1, (20, NUM_BODIES, 3)).astype(np.float32),
    body_quat_w=quat,
    body_lin_vel_w=rng.uniform(-1, 1, (20, NUM_BODIES, 3)).astype(np.float32),
    body_ang_vel_w=rng.uniform(-1, 1, (20, NUM_BODIES, 3)).astype(np.float32),
    fps=np.array([50.0], dtype=np.float32),
  )
  with pytest.raises(ValueError, match="joint_pos shape"):
    EndpointPool.from_directory(d, last_n_frames=10, split="all")


def test_scalar_joint_pos_raises(tmp_path):
  """A 0-D or 1-D joint_pos must raise ValueError, not IndexError."""
  d = tmp_path / "scalar_data"
  d.mkdir()
  rng = np.random.default_rng(0)
  quat = rng.standard_normal((20, NUM_BODIES, 4)).astype(np.float32)
  quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
  np.savez(
    d / "scalar_000.npz",
    joint_pos=rng.uniform(-1, 1, NUM_JOINTS).astype(np.float32),  # 1-D
    joint_vel=rng.uniform(-1, 1, (20, NUM_JOINTS)).astype(np.float32),
    body_pos_w=rng.uniform(-1, 1, (20, NUM_BODIES, 3)).astype(np.float32),
    body_quat_w=quat,
    body_lin_vel_w=rng.uniform(-1, 1, (20, NUM_BODIES, 3)).astype(np.float32),
    body_ang_vel_w=rng.uniform(-1, 1, (20, NUM_BODIES, 3)).astype(np.float32),
    fps=np.array([50.0], dtype=np.float32),
  )
  with pytest.raises(ValueError, match="joint_pos must be 2-D"):
    EndpointPool.from_directory(d, last_n_frames=10, split="all")


def test_nan_raises(tmp_path):
  d = _make_dataset_dir(tmp_path, num_files=1, frames_per_file=20)
  rng = np.random.default_rng(0)
  jp = rng.uniform(-1, 1, (20, NUM_JOINTS)).astype(np.float32)
  jp[5, 3] = np.nan
  _make_synthetic_npz(d / "syn_000.npz", num_frames=20, joint_pos=jp, seed=0)
  with pytest.raises(ValueError, match="non-finite"):
    EndpointPool.from_directory(d, last_n_frames=10, split="all")


def test_degenerate_quaternion_raises(tmp_path):
  d = tmp_path / "bad_quat"
  d.mkdir()
  rng = np.random.default_rng(0)
  quat = np.ones((20, NUM_BODIES, 4), dtype=np.float32)
  quat[5, 0, :] = [0, 0, 0, 0]
  np.savez(
    d / "bad_000.npz",
    joint_pos=rng.uniform(-1, 1, (20, NUM_JOINTS)).astype(np.float32),
    joint_vel=rng.uniform(-1, 1, (20, NUM_JOINTS)).astype(np.float32),
    body_pos_w=rng.uniform(-1, 1, (20, NUM_BODIES, 3)).astype(np.float32),
    body_quat_w=quat,
    body_lin_vel_w=rng.uniform(-1, 1, (20, NUM_BODIES, 3)).astype(np.float32),
    body_ang_vel_w=rng.uniform(-1, 1, (20, NUM_BODIES, 3)).astype(np.float32),
    fps=np.array([50.0], dtype=np.float32),
  )
  with pytest.raises(ValueError, match="degenerate quaternion"):
    EndpointPool.from_directory(d, last_n_frames=10, split="all")


def test_incompatible_fps_raises(tmp_path):
  d = _make_dataset_dir(tmp_path, num_files=3, frames_per_file=20)
  _make_synthetic_npz(d / "syn_002.npz", num_frames=20, fps=60.0, seed=2)
  with pytest.raises(ValueError, match="Incompatible FPS"):
    EndpointPool.from_directory(d, last_n_frames=10, split="all")


def test_empty_file_raises(tmp_path):
  d = tmp_path / "empty_file"
  d.mkdir()
  quat = np.ones((0, NUM_BODIES, 4), dtype=np.float32)
  np.savez(
    d / "empty_000.npz",
    joint_pos=np.zeros((0, NUM_JOINTS), dtype=np.float32),
    joint_vel=np.zeros((0, NUM_JOINTS), dtype=np.float32),
    body_pos_w=np.zeros((0, NUM_BODIES, 3), dtype=np.float32),
    body_quat_w=quat,
    body_lin_vel_w=np.zeros((0, NUM_BODIES, 3), dtype=np.float32),
    body_ang_vel_w=np.zeros((0, NUM_BODIES, 3), dtype=np.float32),
    fps=np.array([50.0], dtype=np.float32),
  )
  with pytest.raises(ValueError, match="0 frames"):
    EndpointPool.from_directory(d, last_n_frames=10, split="all")


def test_one_file_train_split_raises(tmp_path):
  """One file with train split leaves no training trajectories."""
  d = _make_dataset_dir(tmp_path, num_files=1, frames_per_file=20)
  with pytest.raises(ValueError, match="Selected split 'train' is empty"):
    EndpointPool.from_directory(d, last_n_frames=10, split="train")


def test_float32_overflow_raises(tmp_path):
  """Finite float64 values that overflow float32 must be caught.

  The overflow value is placed at frame 15, which is inside the last-10
  window (frames 10-19) of a 20-frame file.
  """
  d = _make_dataset_dir(tmp_path, num_files=1, frames_per_file=20)
  rng = np.random.default_rng(0)
  jp = rng.uniform(-1, 1, (20, NUM_JOINTS)).astype(np.float64)
  jp[15, 3] = 1e40  # inside the last-10 window
  _make_synthetic_npz(d / "syn_000.npz", num_frames=20, joint_pos=jp, seed=0)
  with pytest.raises(ValueError, match="non-finite"):
    EndpointPool.from_directory(d, last_n_frames=10, split="all")


def test_cross_split_incompatible_fps_in_held_out(tmp_path):
  """A held-out file with different FPS must fail even in the train split."""
  d = _make_dataset_dir(tmp_path, num_files=5, frames_per_file=30)
  # Overwrite file 4 with fps=60. With seed=42 and 5 files, one file goes
  # to validation and 4 to train; the incompatible file could be in either
  # split. Either way, ALL files are validated, so it must fail.
  _make_synthetic_npz(d / "syn_004.npz", num_frames=30, fps=60.0, seed=4)
  with pytest.raises(ValueError, match="Incompatible FPS"):
    EndpointPool.from_directory(d, last_n_frames=10, split="all")


# ---------------------------------------------------------------------------
# Import safety test.
# ---------------------------------------------------------------------------


def test_import_does_not_read_dataset(monkeypatch):
  """Importing the module must not trigger any dataset I/O."""
  import importlib

  import mjlab.tasks.velocity.mdp.tennis_endpoint_pool as pool_mod

  npz_opens: list[str] = []

  original_np_load = np.load

  def tracking_np_load(*args, **kwargs):
    fname = args[0] if args else kwargs.get("file", "")
    if isinstance(fname, str | Path) and str(fname).endswith(".npz"):
      npz_opens.append(str(fname))
    return original_np_load(*args, **kwargs)

  original_path_open = Path.open

  def tracking_path_open(self, *args, **kwargs):
    if str(self).endswith(".npz"):
      npz_opens.append(str(self))
    return original_path_open(self, *args, **kwargs)

  monkeypatch.setattr(np, "load", tracking_np_load)
  monkeypatch.setattr(Path, "open", tracking_path_open)
  importlib.reload(pool_mod)
  monkeypatch.undo()
  assert len(npz_opens) == 0, f"Import opened NPZ files: {npz_opens}"


# ---------------------------------------------------------------------------
# Real-data diagnostic tests (function-style, per AGENTS.md).
# ---------------------------------------------------------------------------


def test_real_dataset_loads_and_counts(real_pool_all):
  assert len(real_pool_all) > 0
  assert real_pool_all.joint_pos.shape[1] == NUM_JOINTS
  assert real_pool_all.root_pos_w.shape[1] == 3
  assert real_pool_all.root_quat_w.shape[1] == 4


def test_real_dataset_split_disjoint(real_pool_all):
  train = EndpointPool.from_directory(TENNIS_DATA_DIR, split="train", seed=42)
  val = EndpointPool.from_directory(TENNIS_DATA_DIR, split="validation", seed=42)
  train_tids = set(train.trajectory_ids.tolist())
  val_tids = set(val.trajectory_ids.tolist())
  assert train_tids.isdisjoint(val_tids)
  assert train_tids | val_tids == set(real_pool_all.trajectory_ids.tolist())


def test_real_dataset_fps(real_pool_all):
  assert real_pool_all.manifest["fps"] == 50.0


def test_real_dataset_quaternion_unit_norm(real_pool_all):
  norms = np.linalg.norm(real_pool_all.root_quat_w, axis=-1)
  assert np.allclose(norms, 1.0, atol=1e-5)


def test_real_dataset_finite(real_pool_all):
  for attr in (
    "joint_pos",
    "joint_vel",
    "root_pos_w",
    "root_quat_w",
    "root_lin_vel_w",
    "root_ang_vel_w",
  ):
    assert np.isfinite(getattr(real_pool_all, attr)).all(), (
      f"{attr} has non-finite values"
    )


def test_real_dataset_model_order(real_pool_all):
  """Model-order names match the X2 model exactly."""
  assert real_pool_all.body_names[0] == "pelvis"
  assert real_pool_all.joint_names[0] == "left_hip_pitch_joint"
  waist_joints = [n for n in real_pool_all.joint_names if n.startswith("waist")]
  assert waist_joints == [
    "waist_yaw_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
  ]


def test_real_dataset_root_velocities_nonzero(real_pool_all):
  """Verify high-velocity states are preserved, not clipped."""
  assert np.max(np.abs(real_pool_all.root_lin_vel_w)) > 0.5
  assert np.max(np.abs(real_pool_all.root_ang_vel_w)) > 0.5
  assert np.max(np.abs(real_pool_all.joint_vel)) > 1.0


def test_real_dataset_manifest_full_inventory(real_pool_all):
  m = real_pool_all.manifest
  json.dumps(m)
  assert m["num_files_total"] == len(m["sources"])
  for src in m["sources"]:
    assert len(src["sha256"]) == 64


def test_real_dataset_sample_indices(real_pool_all):
  rng = np.random.default_rng(0)
  idx = real_pool_all.sample_indices(100, rng)
  assert idx.shape == (100,)
  assert idx.min() >= 0
  assert idx.max() < len(real_pool_all)


def test_real_dataset_all_body_fk_ordering(real_pool_all, x2_model):
  """Verify ALL 32 body FK positions and orientations match the source NPZ.

  This catches joint/body reordering: if joint_pos columns were permuted,
  downstream body FK positions would diverge from the stored body_pos_w.
  """
  import mujoco

  data = mujoco.MjData(x2_model)
  sample_rows = np.linspace(0, len(real_pool_all) - 1, 10, dtype=int)
  max_pos_err = 0.0
  max_quat_err = 0.0

  for row in sample_rows:
    traj_id = int(real_pool_all.trajectory_ids[row])
    frame_idx = int(real_pool_all.frame_indices[row])
    filename = real_pool_all.manifest["sources"][traj_id]["filename"]
    npz_path = TENNIS_DATA_DIR / filename
    with np.load(npz_path) as npz:
      stored_body_pos = npz["body_pos_w"][frame_idx]
      stored_body_quat = npz["body_quat_w"][frame_idx]

    data.qpos[:3] = real_pool_all.root_pos_w[row]
    data.qpos[3:7] = real_pool_all.root_quat_w[row]
    data.qpos[7:] = real_pool_all.joint_pos[row]
    mujoco.mj_kinematics(x2_model, data)

    for bi in range(1, x2_model.nbody):
      fk_pos = data.xpos[bi]
      stored_pos = stored_body_pos[bi - 1]
      pos_err = np.max(np.abs(fk_pos - stored_pos))
      max_pos_err = max(max_pos_err, pos_err)

      fk_quat = data.xquat[bi]
      stored_quat = stored_body_quat[bi - 1]
      dot = abs(np.dot(fk_quat, stored_quat))
      max_quat_err = max(max_quat_err, 1.0 - dot)

  # Matches verified full-dataset bounds: pos ~1.2e-6, quat ~8.3e-8.
  assert max_pos_err < 1e-4, f"FK position error too high: {max_pos_err}"
  assert max_quat_err < 1e-6, f"FK quaternion error too high: {max_quat_err}"


def test_real_dataset_fk_detects_joint_permutation(real_pool_all, x2_model):
  """A permuted joint_pos column must produce FK errors well above tolerance."""
  import mujoco

  data = mujoco.MjData(x2_model)
  # Swap two leg joints with different kinematic chains.
  swapped = real_pool_all.joint_pos.copy()
  swapped_jp = swapped[0].copy()
  swapped_jp[[0, 6]] = swapped_jp[[6, 0]]  # swap left/right hip_pitch

  data.qpos[:3] = real_pool_all.root_pos_w[0]
  data.qpos[3:7] = real_pool_all.root_quat_w[0]
  data.qpos[7:] = swapped_jp
  mujoco.mj_kinematics(x2_model, data)

  traj_id = int(real_pool_all.trajectory_ids[0])
  frame_idx = int(real_pool_all.frame_indices[0])
  filename = real_pool_all.manifest["sources"][traj_id]["filename"]
  with np.load(TENNIS_DATA_DIR / filename) as npz:
    stored_body_pos = npz["body_pos_w"][frame_idx]

  max_err = 0.0
  for bi in range(1, x2_model.nbody):
    err = np.max(np.abs(data.xpos[bi] - stored_body_pos[bi - 1]))
    max_err = max(max_err, err)
  # A swap of left/right hip pitch must cause large FK divergence.
  assert max_err > 0.01, f"Permutation should cause >1cm FK error but got {max_err}"


def test_real_dataset_velocity_convention(real_pool_all, x2_model):
  """Verify root_ang_vel_w is world-frame: R^T @ world_omega as qvel[3:6]
  reproduces world omega via mj_objectVelocity, but writing world_omega
  directly does not (at non-identity orientations)."""
  import mujoco

  data = mujoco.MjData(x2_model)
  # Find a row with non-identity orientation and significant angular velocity.
  best_row = -1
  best_yaw = 0.0
  for row in range(len(real_pool_all)):
    w, x, y, z = real_pool_all.root_quat_w[row]
    # Approximate yaw magnitude from quaternion.
    yaw_mag = abs(z) + abs(y)
    if yaw_mag > best_yaw:
      best_yaw = yaw_mag
      best_row = row

  assert best_row >= 0
  quat = real_pool_all.root_quat_w[best_row]  # wxyz
  world_omega = real_pool_all.root_ang_vel_w[best_row]

  w, x, y, z = quat
  R = np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
  )
  local_omega = R.T @ world_omega

  data.qpos[:3] = real_pool_all.root_pos_w[best_row]
  data.qpos[3:7] = quat
  data.qpos[7:] = real_pool_all.joint_pos[best_row]
  data.qvel[:3] = real_pool_all.root_lin_vel_w[best_row]
  data.qvel[3:6] = local_omega
  data.qvel[6:] = real_pool_all.joint_vel[best_row]
  mujoco.mj_forward(x2_model, data)

  res = np.zeros(6)
  mujoco.mj_objectVelocity(x2_model, data, mujoco.mjtObj.mjOBJ_BODY, 1, res, 0)
  recovered = res[:3]
  err_correct = np.max(np.abs(recovered - world_omega))

  data.qvel[3:6] = world_omega  # WRONG: world omega in body-local slot
  mujoco.mj_forward(x2_model, data)
  mujoco.mj_objectVelocity(x2_model, data, mujoco.mjtObj.mjOBJ_BODY, 1, res, 0)
  recovered_wrong = res[:3]
  err_wrong = np.max(np.abs(recovered_wrong - world_omega))

  assert err_correct < 1e-4, f"Local qvel error too high: {err_correct}"
  assert err_wrong > 0.01, (
    f"World qvel should not match at non-identity orientation: {err_wrong}"
  )


def test_real_dataset_foot_sphere_clearance(real_pool_all, x2_model):
  """Measure actual foot collision-sphere clearance. Report overlap, do not
  assert 'no penetration'. This is diagnostic evidence, not a filter."""
  import mujoco

  data = mujoco.MjData(x2_model)
  foot_geom_indices = []
  for gi in range(x2_model.ngeom):
    gname = x2_model.geom(gi).name
    if gname and "foot" in gname and "collision" in gname:
      if x2_model.geom_size[gi][0] > 0:
        foot_geom_indices.append(gi)

  assert len(foot_geom_indices) == 24

  worst_clearance = float("inf")
  states_below_1mm = 0

  for row in range(len(real_pool_all)):
    data.qpos[:3] = real_pool_all.root_pos_w[row]
    data.qpos[3:7] = real_pool_all.root_quat_w[row]
    data.qpos[7:] = real_pool_all.joint_pos[row]
    mujoco.mj_kinematics(x2_model, data)

    state_below = False
    for gi in foot_geom_indices:
      sz = x2_model.geom_size[gi][0]
      clearance = data.geom_xpos[gi, 2] - sz
      if clearance < worst_clearance:
        worst_clearance = clearance
      if clearance < -0.001:
        state_below = True
    if state_below:
      states_below_1mm += 1

  # Report actual measured values. These are shallow overlaps, not clearance.
  assert worst_clearance < 0.0, (
    f"Expected foot-sphere overlap (parent measured -5.76mm) but got "
    f"{worst_clearance:.4f}m"
  )
  # Report the measured diagnostics. Do NOT filter, clip, lift, or reject.
  assert states_below_1mm > 0, "Expected states with foot overlap"


def test_real_dataset_no_silent_data_modification(real_pool_all):
  """The pool must preserve source data verbatim: loading the NPZ directly
  and comparing to pool arrays for the same trajectory/frame should match."""
  traj_id = int(real_pool_all.trajectory_ids[0])
  frame_idx = int(real_pool_all.frame_indices[0])
  filename = real_pool_all.manifest["sources"][traj_id]["filename"]
  with np.load(TENNIS_DATA_DIR / filename) as npz:
    np.testing.assert_allclose(
      real_pool_all.joint_pos[0],
      npz["joint_pos"][frame_idx],
      atol=1e-5,
    )
    np.testing.assert_allclose(
      real_pool_all.root_pos_w[0],
      npz["body_pos_w"][frame_idx, 0],
      atol=1e-5,
    )
    np.testing.assert_allclose(
      real_pool_all.root_ang_vel_w[0],
      npz["body_ang_vel_w"][frame_idx, 0],
      atol=1e-5,
    )
