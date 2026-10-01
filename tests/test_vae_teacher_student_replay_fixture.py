"""Tests for the committed X2 teacher-fidelity recorded-state fixture."""

import json
from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).parent / "fixtures" / "vae_teacher_student"


def test_recorded_state_fixture_contract_and_coverage() -> None:
  manifest = json.loads((FIXTURE / "manifest.json").read_text())
  with np.load(FIXTURE / "recorded_states.npz", allow_pickle=False) as states:
    assert states["vae_reference"].shape == (1103, 68)
    assert states["vae_conditioning"].shape == (1103, 99)
    assert states["student_raw_action"].shape == (1103, 31)
    assert states["student_latent"].shape == (1103, 32)
    assert states["reference_frame"].shape == (1103,)
    assert states["engagement_generation"].shape == (1103,)
    assert states["assembly_wall_ns"].shape == (1103,)
    assert states["run_kind"].shape == (1103,)
    assert np.isfinite(states["vae_reference"]).all()
    assert np.isfinite(states["vae_conditioning"]).all()

    failure = states["run_kind"] == "failure"
    successful = states["run_kind"] == "successful"
    assert int(failure.sum()) == manifest["runs"]["failure"]["rows"] == 110
    assert int(successful.sum()) == manifest["runs"]["successful_r4"]["rows"] == 993
    assert np.array_equal(states["reference_frame"][failure], np.arange(110))
    for generation in (2, 4, 6):
      selected = successful & (states["engagement_generation"] == generation)
      assert int(selected.sum()) == 331
      assert np.array_equal(states["reference_frame"][selected], np.arange(331))
