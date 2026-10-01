import json
from pathlib import Path

import numpy as np
import pytest

from mjlab.tasks.tracking.diffusion import (
  FeatureStats,
  ProjectionBundle,
  ProjectionError,
)


def test_projection_normalizes_without_added_rows_and_roundtrips() -> None:
  state = np.arange(270, dtype=np.float64).reshape(2, 135) / 10
  latent = np.arange(64, dtype=np.float64).reshape(2, 32) / 5
  state_stats = FeatureStats(2, state.mean(axis=0), np.maximum(state.std(axis=0), 1e-6))
  latent_stats = FeatureStats(
    2, latent.mean(axis=0), np.maximum(latent.std(axis=0), 1e-6)
  )
  bundle = ProjectionBundle.create(state_stats, latent_stats)
  projected = bundle.project_state(state)
  assert projected.shape == (2, 135)
  np.testing.assert_allclose(bundle.inverse_state(projected), state, atol=1e-10)
  np.testing.assert_array_equal(bundle.matrix, np.eye(135, dtype=np.float64))
  np.testing.assert_array_equal(bundle.pseudoinverse, bundle.matrix)


def test_projection_bundle_persistence_checks_hashes(tmp_path) -> None:
  state_stats = FeatureStats(1, np.zeros(135), np.ones(135))
  latent_stats = FeatureStats(1, np.zeros(32), np.ones(32))
  bundle = ProjectionBundle.create(state_stats, latent_stats)
  path = tmp_path / "projection.npz"
  bundle.save(path)
  loaded = ProjectionBundle.load(path)
  np.testing.assert_array_equal(loaded.matrix, bundle.matrix)


@pytest.mark.parametrize("field", ["matrix", "pseudoinverse"])
def test_no_mixture_bundle_rejects_rehashed_nonidentity_transform(
  tmp_path: Path, field: str
) -> None:
  bundle = ProjectionBundle.create(
    FeatureStats(1, np.zeros(135), np.ones(135)),
    FeatureStats(1, np.zeros(32), np.ones(32)),
  )
  path = tmp_path / "scaled.npz"
  bundle.save(path)
  with np.load(path, allow_pickle=False) as saved:
    payload = {name: saved[name] for name in saved.files}
  payload[field] = payload[field].copy()
  payload[field][0, 0] = 6.0
  metadata = json.loads(str(payload["metadata"].item()))
  metadata[f"{field}_sha256"] = bundle._hash(payload[field])
  payload["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
  np.savez_compressed(path, **payload)
  with pytest.raises(ProjectionError, match="identity state transforms"):
    ProjectionBundle.load(path)
