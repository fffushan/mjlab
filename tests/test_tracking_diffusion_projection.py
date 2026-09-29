import numpy as np

from mjlab.tasks.tracking.diffusion import FeatureStats, ProjectionBundle


def test_projection_normalizes_before_emphasis_and_roundtrips() -> None:
  state = np.arange(270, dtype=np.float64).reshape(2, 135) / 10
  latent = np.arange(64, dtype=np.float64).reshape(2, 32) / 5
  state_stats = FeatureStats(2, state.mean(axis=0), np.maximum(state.std(axis=0), 1e-6))
  latent_stats = FeatureStats(
    2, latent.mean(axis=0), np.maximum(latent.std(axis=0), 1e-6)
  )
  bundle = ProjectionBundle.create(state_stats, latent_stats)
  projected = bundle.project_state(state)
  assert projected.shape == (2, 199)
  np.testing.assert_allclose(bundle.inverse_state(projected), state, atol=1e-10)
  assert np.all(bundle.matrix[:64, :15] != bundle.matrix[:64, 15:16])


def test_projection_bundle_persistence_checks_hashes(tmp_path) -> None:
  state_stats = FeatureStats(1, np.zeros(135), np.ones(135))
  latent_stats = FeatureStats(1, np.zeros(32), np.ones(32))
  bundle = ProjectionBundle.create(state_stats, latent_stats)
  path = tmp_path / "projection.npz"
  bundle.save(path)
  loaded = ProjectionBundle.load(path)
  np.testing.assert_array_equal(loaded.matrix, bundle.matrix)
