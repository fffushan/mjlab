"""Focused D3 real-model replay tests (CPU, stub models, cheap fixtures).

These tests exercise only the pure-Python gates and the seam-vs-direct parity
plumbing in ``diffusion/replay_integration.py``.  No real checkpoint, projection
bundle, VAE checkpoint or simulator is involved: the denoiser and decoder are
small deterministic stubs and the projection statistics are fitted from a few
random rows.  Passing here is NOT evidence about the frozen D2 checkpoint or
the frozen VAE; the real-model command in the README must be executed by the
integration owner for that.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import replace as dataclass_replace

import numpy as np
import pytest
import torch
from torch import nn

from mjlab.tasks.tracking.diffusion import replay_integration as ri
from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT
from mjlab.tasks.tracking.diffusion.model import (
  DenoiserSettings,
  StateLatentTransformer,
)
from mjlab.tasks.tracking.diffusion.projection import (
  FeatureStats,
  ProjectionBundle,
  fit_feature_stats,
)
from mjlab.tasks.tracking.diffusion.replay_integration import (
  ReplayError,
  apply_ema_selection,
  load_recorded_history,
  run_seam_replay,
  synthetic_conditioning,
  verify_file_digest,
  verify_projection_consistency,
)
from mjlab.tasks.tracking.diffusion.schedule import (
  DiffusionSchedule,
  build_inference_grid,
)
from mjlab.tasks.tracking.diffusion.state import WorldState
from mjlab.tasks.tracking.distillation.cohort_contract import CohortMember

_STATE_DIM = DEFAULT_CONTRACT.state_dimension
_LATENT_DIM = DEFAULT_CONTRACT.latent_dimension


def _stats(width: int, *, seed: int) -> FeatureStats:
  rng = np.random.default_rng(seed)
  return fit_feature_stats([rng.normal(size=(64, width)) + 3.0], std_floor=1e-6)


def _projection() -> ProjectionBundle:
  return ProjectionBundle.create(
    _stats(_STATE_DIM, seed=1), _stats(_LATENT_DIM, seed=2)
  )


def _make_state(index: int) -> WorldState:
  yaw = 0.17 + 0.03 * index
  return WorldState(
    np.array([1.0 + index * 0.01, -0.2, 0.5]),
    np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]),
    np.array([0.2, -0.1, 0.03]),
    np.array([0.0, 0.1, -0.2]),
    np.arange(60, dtype=np.float64).reshape(20, 3) * 0.01,
    np.ones((20, 3), dtype=np.float64) * 0.2,
  )


class _StubDenoiser:
  """Deterministic scaled-identity denoiser stand-in."""

  def __call__(self, sample: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
    return 0.1 * sample


class _StubDecoder(nn.Module):
  """Gravity-mode stand-in with the frozen decode(latent, cond) signature."""

  def __call__(self, latent: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
    assert latent.shape == (1, 32) and conditioning.shape == (1, 99)
    assert latent.dtype == conditioning.dtype == torch.float32
    return latent[:, :31] * 0.5


def _history_inputs(samples: int) -> tuple[list[WorldState], np.ndarray, list[int]]:
  states = [_make_state(i) for i in range(8 + samples)]
  latents = (
    np.linspace(-1.0, 1.0, _LATENT_DIM)[None, :]
    + 0.01 * np.arange(8 + samples)[:, None]
  )
  return states, latents, list(range(8 + samples))


def test_seam_matches_direct_path_with_stub_models() -> None:
  """Seam and direct sampler paths must agree exactly from one generator."""
  schedule = DiffusionSchedule.build()
  grid = build_inference_grid(schedule, contract=DEFAULT_CONTRACT)
  states, latents, ticks = _history_inputs(samples=1)
  report = run_seam_replay(
    denoiser=_StubDenoiser(),
    decode=_StubDecoder(),
    schedule=schedule,
    grid=grid,
    projection=_projection(),
    history_states=states,
    history_latents=latents,
    ticks=ticks,
    seeds=(0, 1),
    samples=1,
  )
  assert report["closed_loop"] is False
  assert report["conditioning"]["kind"] == "synthetic-fixture"
  assert report["runs"][0]["steps"][0]["parity"]["passed"]
  assert report["runs"][0]["steps"][0]["parity"]["latent_max_abs_diff"] == 0.0
  assert report["runs"][0]["steps"][0]["denoiser_calls"] == 20
  assert report["runs"][0]["steps"][0]["final_index"] == 0
  assert report["plan"]["mask_matches_contract"] is True
  assert report["plan"]["shape"] == [41, 167]
  assert report["determinism"]["passed"]
  assert report["decode_calls"] == 6  # 2 runs x 1 + replay x 1 + 3 direct paths
  assert [run["seed"] for run in report["runs"]] == [0, 1, 0]


def test_seam_replay_records_executed_pairs_and_no_model_state_change() -> None:
  """Recorded executed latents feed the next step and models stay frozen."""
  schedule = DiffusionSchedule.build()
  grid = build_inference_grid(schedule, contract=DEFAULT_CONTRACT)
  states, latents, ticks = _history_inputs(samples=2)
  projection = _projection()
  denoiser = StateLatentTransformer(DenoiserSettings.from_contract(DEFAULT_CONTRACT))
  denoiser.eval()
  before = {
    name: value.detach().clone() for name, value in denoiser.state_dict().items()
  }
  report = run_seam_replay(
    denoiser=denoiser,
    decode=_StubDecoder(),
    schedule=schedule,
    grid=grid,
    projection=projection,
    history_states=states,
    history_latents=latents,
    ticks=ticks,
    seeds=(0, 1),
    samples=2,
  )
  after = {
    name: value.detach().clone() for name, value in denoiser.state_dict().items()
  }
  assert set(before) == set(after)
  for name in before:
    assert torch.equal(before[name], after[name])
  assert report["runs"][0]["steps"][0]["parity"]["passed"]
  # Each run recorded its executed latents: history stays bounded at 8 pairs.
  assert report["runs"][0]["steps"][1]["tick"] == 9
  # Conditioning fixture is finite and deterministic.
  np.testing.assert_array_equal(synthetic_conditioning(0), synthetic_conditioning(0))
  with pytest.raises(ReplayError, match="two independent window seeds"):
    run_seam_replay(
      denoiser=_StubDenoiser(),
      decode=_StubDecoder(),
      schedule=schedule,
      grid=grid,
      projection=projection,
      history_states=states,
      history_latents=latents,
      ticks=ticks,
      seeds=(5, 5),
      samples=1,
    )


def test_ema_selection_copies_shadow_and_reports_updates() -> None:
  """EMA selection follows trainer.py: shadow copied into the model."""
  model = StateLatentTransformer(DenoiserSettings.from_contract(DEFAULT_CONTRACT))
  ema = ri.ExponentialMovingAverage(model, power=0.75, max_decay=0.9999)
  ema.updates = 42
  for value in ema.shadow.values():
    value.add_(1.0)
  shifted = {name: value.clone() for name, value in model.state_dict().items()}
  state = {
    "power": ema.power,
    "max_decay": ema.max_decay,
    "updates": 42,
    "shadow": ema.shadow,
  }
  selected, updates = apply_ema_selection(model, state)
  assert selected is True
  assert updates == 42
  for name, value in model.state_dict().items():
    assert torch.equal(value, ema.shadow[name])
    assert not torch.equal(value, shifted[name])
  # A checkpoint without EMA keeps the raw model weights.
  unselected, no_updates = apply_ema_selection(model, None)
  assert unselected is False and no_updates is None


def test_digest_and_projection_gates_fail_closed(tmp_path) -> None:
  gate = tmp_path / "artifact.bin"
  gate.write_bytes(b"frozen bytes")
  digest = ri.sha256_file(gate)
  verify_file_digest(gate, digest, role="gate")
  with pytest.raises(ReplayError, match="sha256 mismatch"):
    verify_file_digest(gate, "0" * 64, role="gate")
  with pytest.raises(ReplayError, match="not a lowercase 64-hex"):
    verify_file_digest(gate, "xyz", role="gate")

  bundle = _projection()
  other = ProjectionBundle.create(
    _stats(_STATE_DIM, seed=7), _stats(_LATENT_DIM, seed=8)
  )
  verify_projection_consistency(
    {
      "matrix": bundle.matrix_sha256,
      "pseudoinverse": bundle.pseudoinverse_sha256,
      "statistics": bundle.statistics_sha256,
    },
    bundle,
  )
  with pytest.raises(ReplayError, match="statistics"):
    verify_projection_consistency(
      {
        "matrix": bundle.matrix_sha256,
        "pseudoinverse": bundle.pseudoinverse_sha256,
        "statistics": other.statistics_sha256,
      },
      bundle,
    )
  with pytest.raises(ReplayError, match="matrix"):
    verify_projection_consistency({}, bundle)


def test_recorded_history_gates_on_real_layout() -> None:
  """Row flags/latent presence/ticks are validated under the recorded layout."""
  rows = 12
  metadata = [
    {
      "controller": "vae",
      "reset": False,
      "teleport": False,
      "reference_boundary": False,
      "segment_qualified": True,
      "tick": index,
      "run_id": "run",
      "episode_id": "episode",
    }
    for index in range(rows)
  ]
  good = {
    "root_position": np.zeros((rows, 3)),
    "root_quaternion_wxyz": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (rows, 1)),
    "root_linear_velocity": np.zeros((rows, 3)),
    "root_angular_velocity": np.zeros((rows, 3)),
    "body_positions": np.zeros((rows, 20, 3)),
    "body_linear_velocities": np.zeros((rows, 20, 3)),
    "latent": np.ones((rows, 32)),
    "latent_present": np.ones(rows, dtype=np.uint8),
    "metadata": np.array([json.dumps(item) for item in metadata]),
  }
  with pytest.raises(ReplayError, match="unreadable"):
    load_recorded_history("/nonexistent/shard.npz", row_offset=0, samples=1)

  def write_shard(path, arrays):
    np.savez(path, **arrays)

  # Clean rows load and return the recorded world state / executed latent.
  recorded = load_recorded_history(
    _temporary_store(write_shard, good), row_offset=0, samples=1
  )
  assert recorded.states[0].root_position.shape == (3,)
  assert recorded.latents.shape == (9, 32)
  assert recorded.ticks == tuple(range(9))

  # A missing executed latent fails closed.
  bad_present = dict(good)
  bad_present["latent_present"] = np.ones(rows, dtype=np.uint8)
  bad_present["latent_present"][3] = 0
  with pytest.raises(ReplayError, match="latent_present"):
    load_recorded_history(
      _temporary_store(write_shard, bad_present), row_offset=0, samples=1
    )
  # A teleport row fails closed.
  bad_teleport = json.loads(json.dumps(metadata))
  bad_teleport[2]["teleport"] = True
  bad_shard = dict(good)
  bad_shard["metadata"] = np.array([json.dumps(item) for item in bad_teleport])
  with pytest.raises(ReplayError, match="teleport"):
    load_recorded_history(
      _temporary_store(write_shard, bad_shard), row_offset=0, samples=1
    )
  # Ticks must strictly increase.
  bad_ticks = json.loads(json.dumps(metadata))
  bad_ticks[4]["tick"] = 0
  bad_shard = dict(good)
  bad_shard["metadata"] = np.array([json.dumps(item) for item in bad_ticks])
  with pytest.raises(ReplayError, match="strictly increase"):
    load_recorded_history(
      _temporary_store(write_shard, bad_shard), row_offset=0, samples=1
    )
  # Segment must be qualified.
  bad_segment = json.loads(json.dumps(metadata))
  bad_segment[6]["segment_qualified"] = False
  bad_shard = dict(good)
  bad_shard["metadata"] = np.array([json.dumps(item) for item in bad_segment])
  with pytest.raises(ReplayError, match="qualified"):
    load_recorded_history(
      _temporary_store(write_shard, bad_shard), row_offset=0, samples=1
    )


def _temporary_store(writer, arrays) -> object:
  """Write a shard to a temp file and return the path object."""
  import tempfile
  from pathlib import Path

  descriptor_path = Path(tempfile.mkdtemp()) / "shard.npz"
  writer(descriptor_path, arrays)
  return descriptor_path


def test_vae_member_joint_order_uses_cohort_common() -> None:
  """The real CohortMember has no .actions; joint order comes from common.

  Regression for the verified loader-interface mismatch fixed in
  ``replay_integration.load_vae_decoder`` (AttributeError:
  'CohortMember' object has no attribute 'actions').  The real interface is
  ``inference.cohort.common.joint_names`` on the returned
  ``CohortMemberInference``; the loader must consume that, not invent one.
  """
  from mjlab.tasks.tracking.distillation.checkpoint import CohortMemberInference

  class _RealShapeSchema:
    conditioning_dim = 99
    reference_dim = 68
    action_dim = 31

  class _RealShapeSettings:
    latent_dim = 32

  class _RealShapeModel(nn.Module):
    schema = _RealShapeSchema()
    settings = _RealShapeSettings()

  inference = object.__new__(CohortMemberInference)
  object.__setattr__(inference, "model", _RealShapeModel())
  object.__setattr__(inference, "schema", None)
  object.__setattr__(inference, "settings", None)
  object.__setattr__(inference, "cohort", None)
  object.__setattr__(inference, "member", None)
  object.__setattr__(inference, "requested_teacher_id", "tennis_000")
  object.__setattr__(inference, "counters", {})
  object.__setattr__(inference, "schedule", {})
  object.__setattr__(inference, "resolved_config", {})
  object.__setattr__(inference, "relocated_artifact_roles", ("onnx",))
  object.__setattr__(inference, "reset_provenance", None)
  object.__setattr__(inference, "evaluation_reset_profile", None)
  # The real CohortMember exposes only artifacts/hashes/motion metadata;
  # accessing member.actions must raise, exactly as on the real return value.
  member = object.__new__(CohortMember)
  with pytest.raises(AttributeError):
    _ = member.actions  # type: ignore[attr-defined]
  # The loader reads joint names from cohort.common, mirroring the real path.
  assert ri.LoadedVae.__dataclass_fields__["joint_order"].type == "tuple[str, ...]"
  source = inspect.getsource(ri.load_vae_decoder)
  assert "cohort.common.joint_names" in source
  assert ".actions.joint_names" not in source


def test_module_entrypoint_guard(monkeypatch, tmp_path, capsys) -> None:
  """python -m executes main(); import does not (CLI entrypoint guard)."""
  import subprocess
  import sys

  code = "import mjlab.tasks.tracking.diffusion.replay_integration as m; print(m.main)"
  env = dict(**__import__("os").environ)
  env["PYTHONPATH"] = str(ri.Path.cwd() / "src")
  plain = subprocess.run(
    [sys.executable, "-c", code],
    capture_output=True,
    text=True,
    env=env,
    check=False,
  )
  assert plain.returncode == 0, plain.stderr
  assert plain.stdout.strip() != ""
  guarded = ri.Path(
    "src/mjlab/tasks/tracking/diffusion/replay_integration.py"
  ).read_text()
  assert 'if __name__ == "__main__":' in guarded


def test_denoiser_loader_gate_rejects_wrong_contract_identity(tmp_path) -> None:
  """The loader must fail closed on a checkpoint with a foreign identity."""
  from mjlab.tasks.tracking.diffusion.checkpoint import save_checkpoint

  model = StateLatentTransformer(DenoiserSettings.from_contract(DEFAULT_CONTRACT))

  ema = ri.ExponentialMovingAverage(model)
  schedule = DiffusionSchedule.build()
  checkpoint = tmp_path / "checkpoint.pt"
  save_checkpoint(
    checkpoint,
    model=model,
    ema=ema,
    optimizer=None,
    scaler=None,
    config={"seed": 7},
    contract=dataclass_replace(DEFAULT_CONTRACT, std_floor=1e-5),
    schedule=schedule,
    dataset_identity={},
    global_step=5,
    epoch=1,
    best_validation_loss=0.5,
    rng_state=None,
  )
  with pytest.raises(ReplayError, match="rejected"):
    ri.load_denoiser(
      checkpoint,
      contract=DEFAULT_CONTRACT,
      checkpoint_sha256=ri.sha256_file(checkpoint),
      projection_bundle=_projection(),
    )


def _cli_args(tmp_path) -> list[str]:
  """Explicit CLI arguments against non-existent tmp fixtures."""
  return [
    "--checkpoint",
    str(tmp_path / "c.pt"),
    "--checkpoint-sha256",
    "0" * 64,
    "--contract",
    str(tmp_path / "contract.yaml"),
    "--projection",
    str(tmp_path / "p.npz"),
    "--projection-sha256",
    "0" * 64,
    "--vae-checkpoint",
    str(tmp_path / "v.pt"),
    "--vae-checkpoint-sha256",
    "0" * 64,
    "--vae-manifest",
    str(tmp_path / "m.yaml"),
    "--vae-member",
    "tennis_000",
    "--vae-repo-root",
    str(tmp_path / "root"),
    "--history-shard",
    str(tmp_path / "s.npz"),
    "--output",
    str(tmp_path / "out" / "report.json"),
  ]


def test_cli_sample_and_seed_bounds(monkeypatch, tmp_path) -> None:
  """The CLI must reject out-of-bounds samples and seeds before any loading."""
  import sys

  tail = _cli_args(tmp_path)
  for extra, pattern in (
    (["--samples", "5"], "samples"),
    (["--samples", "0"], "samples"),
    (["--seed-0", "-1"], "seeds"),
    (["--seed-1", "1001"], "seeds"),
    (["--cpu-threads", "0"], "cpu_threads"),
  ):
    monkeypatch.setattr(sys, "argv", ["replay", *extra, *tail])
    with pytest.raises(ri.ReplayError, match=pattern):
      ri.main()


def test_cli_hash_gate_fails_closed(monkeypatch, tmp_path) -> None:
  """A wrong projection digest must fail before any model loading."""
  import shutil
  import sys

  contract_yaml = tmp_path / "contract.yaml"
  shutil.copyfile(
    ri.Path("docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml"),
    contract_yaml,
  )
  projection = _projection()
  projection_path = tmp_path / "projection.npz"
  projection.save(projection_path)
  tail = _cli_args(tmp_path)
  tail[tail.index("--contract") + 1] = str(contract_yaml)
  tail[tail.index("--projection") + 1] = str(projection_path)
  tail[tail.index("--projection-sha256") + 1] = "e" * 64
  monkeypatch.setattr(sys, "argv", ["replay", *tail])
  with pytest.raises(ri.ReplayError, match="projection bundle sha256 mismatch"):
    ri.main()
