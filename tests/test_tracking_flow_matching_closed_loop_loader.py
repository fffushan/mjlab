"""Runtime loader tests for the closed-loop flow boundary.

These tests exercise ``replay_integration.load_flow_denoiser`` at the model
loading seam with small synthetic flow checkpoints built from a real
:class:`StateLatentTransformer`.  They check loader wiring only: the actual
linear process and level grid, EMA selection, freezing, and family rejection.
No real checkpoint, simulator or closed-loop claim is involved.
"""

from __future__ import annotations

from dataclasses import replace as dataclass_replace
from typing import Any, cast

import numpy as np
import pytest
import torch

from mjlab.tasks.tracking.diffusion import replay_integration as ri
from mjlab.tasks.tracking.diffusion.checkpoint import save_checkpoint
from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT
from mjlab.tasks.tracking.diffusion.model import (
  DenoiserSettings,
  StateLatentTransformer,
)
from mjlab.tasks.tracking.diffusion.projection import (
  ProjectionBundle,
  StateLatentNormalizer,
  fit_feature_stats,
)
from mjlab.tasks.tracking.diffusion.schedule import DiffusionSchedule
from mjlab.tasks.tracking.diffusion.trainer import ExponentialMovingAverage
from mjlab.tasks.tracking.flow_matching.checkpoint import (
  load_flow_checkpoint,
  model_identity,
  save_flow_checkpoint,
)
from mjlab.tasks.tracking.flow_matching.process import LinearNoiseProcess
from mjlab.tasks.tracking.flow_matching.sampler import FlowGrid

_STATE_DIM = DEFAULT_CONTRACT.state_dimension
_LATENT_DIM = DEFAULT_CONTRACT.latent_dimension


class _AutoEma:
  """Sentinel: build an EMA from the model when the caller supplies none."""


_AUTO_EMA = _AutoEma()


def _stats(width: int, *, seed: int):
  rng = np.random.default_rng(seed)
  return fit_feature_stats([rng.normal(size=(64, width)) + 3.0], std_floor=1e-6)


def _normalizer(contract=DEFAULT_CONTRACT, *, seed: int = 1) -> StateLatentNormalizer:
  return StateLatentNormalizer.create(
    _stats(_STATE_DIM, seed=seed),
    _stats(_LATENT_DIM, seed=seed + 1),
    contract=contract,
  )


def _identity(normalizer: StateLatentNormalizer) -> dict[str, object]:
  """A flow dataset identity whose recorded statistics match ``normalizer``."""
  return {
    "directory": "synthetic",
    "projection_hashes": {"statistics": normalizer.statistics_sha256},
  }


def _build_model(process: LinearNoiseProcess) -> StateLatentTransformer:
  return StateLatentTransformer(
    DenoiserSettings.from_contract(DEFAULT_CONTRACT, training_k=process.max_level)
  )


def _save_flow(
  path,
  *,
  dataset_identity: dict[str, object],
  process: LinearNoiseProcess | None = None,
  ema: ExponentialMovingAverage | None | _AutoEma = _AUTO_EMA,
) -> ExponentialMovingAverage | None:
  """Write a synthetic flow checkpoint and return the EMA that was saved.

  ``ema`` is created from the model when omitted; pass ``None`` explicitly to
  store a checkpoint that carries no EMA shadow.
  """
  process = process if process is not None else LinearNoiseProcess()
  model = _build_model(process)
  if isinstance(ema, _AutoEma):
    ema = ExponentialMovingAverage(model)
  save_flow_checkpoint(
    path,
    model=model,
    ema=ema,
    optimizer=None,
    scaler=None,
    config={"seed": 3},
    process=process,
    data_contract=DEFAULT_CONTRACT,
    dataset_identity=dataset_identity,
    model_description=model_identity(model),
    global_step=9,
    epoch=2,
    best_validation_loss=0.125,
    rng_state=None,
    best_metric_split="validation",
    fp32_checked=True,
  )
  return ema


def test_flow_loader_restores_real_process_grid_and_freezes_model(tmp_path) -> None:
  normalizer = _normalizer()
  checkpoint = tmp_path / "flow.pt"
  _save_flow(checkpoint, dataset_identity=_identity(normalizer))

  loaded = ri.load_flow_denoiser(
    checkpoint, contract=DEFAULT_CONTRACT, normalizer=normalizer
  )

  assert isinstance(loaded.process, LinearNoiseProcess)
  assert loaded.process.max_level == 1000
  assert isinstance(loaded.grid, FlowGrid)
  assert loaded.grid.calls == ri.DEFAULT_CLOSED_LOOP_SOLVER_STEPS == 6
  assert loaded.grid.levels[0] == 1000
  assert loaded.grid.levels[-1] == 0
  # Honest flow-family provenance, no relabelled diffusion identity.
  assert loaded.family == "x0_linear_flow"
  assert loaded.process_identity == loaded.process.identity_hash()
  assert loaded.data_identity == DEFAULT_CONTRACT.identity_hash()
  assert loaded.global_step == 9
  assert loaded.best_validation_loss == pytest.approx(0.125)
  assert loaded.selection_split == "validation"
  assert loaded.fp32_checked is True
  assert loaded.config_seed == 3
  assert loaded.projection_hashes == {"statistics": normalizer.statistics_sha256}
  # No diffusion schedule is fabricated for the flow family.
  assert not hasattr(loaded, "schedule")
  # The loaded model is frozen in eval mode.
  assert loaded.model.training is False
  assert all(not p.requires_grad for p in loaded.model.parameters())


def test_flow_loader_uses_the_checkpoint_max_level(tmp_path) -> None:
  normalizer = _normalizer()
  process = LinearNoiseProcess(max_level=100)
  checkpoint = tmp_path / "flow100.pt"
  _save_flow(checkpoint, process=process, dataset_identity=_identity(normalizer))

  loaded = ri.load_flow_denoiser(
    checkpoint, contract=DEFAULT_CONTRACT, normalizer=normalizer
  )
  assert loaded.process.max_level == 100
  assert loaded.grid.calls == ri.DEFAULT_CLOSED_LOOP_SOLVER_STEPS
  assert loaded.grid.levels[0] == 100
  assert loaded.grid.levels[-1] == 0
  assert loaded.process_identity == process.identity_hash()


def test_flow_loader_takes_an_explicit_solver_step_count(tmp_path) -> None:
  normalizer = _normalizer()
  checkpoint = tmp_path / "flow.pt"
  _save_flow(checkpoint, dataset_identity=_identity(normalizer))

  for steps in (1, 4, 10, 64, 1000):
    loaded = ri.load_flow_denoiser(
      checkpoint, contract=DEFAULT_CONTRACT, normalizer=normalizer, solver_steps=steps
    )
    assert loaded.grid.calls == steps
    assert len(loaded.grid.levels) == steps + 1
    assert loaded.grid.levels[0] == 1000
    assert loaded.grid.levels[-1] == 0
    # The checkpoint process itself is unchanged by the step count.
    assert loaded.process.max_level == 1000


def test_flow_loader_rejects_an_impossible_solver_step_count(tmp_path) -> None:
  normalizer = _normalizer()
  checkpoint = tmp_path / "flow100.pt"
  _save_flow(
    checkpoint,
    process=LinearNoiseProcess(max_level=100),
    dataset_identity=_identity(normalizer),
  )

  # More steps than levels cannot form a strictly descending integer grid.
  with pytest.raises(ri.ReplayError, match="exceed the checkpoint process"):
    ri.load_flow_denoiser(
      checkpoint, contract=DEFAULT_CONTRACT, normalizer=normalizer, solver_steps=101
    )
  with pytest.raises(ri.ReplayError, match="positive integer"):
    ri.load_flow_denoiser(
      checkpoint, contract=DEFAULT_CONTRACT, normalizer=normalizer, solver_steps=0
    )
  # Malformed callers are rejected at runtime, not only by the type checker.
  with pytest.raises(ri.ReplayError, match="must be an integer"):
    ri.load_flow_denoiser(
      checkpoint,
      contract=DEFAULT_CONTRACT,
      normalizer=normalizer,
      solver_steps=cast(Any, 1.5),
    )
  with pytest.raises(ri.ReplayError, match="must be an integer"):
    ri.load_flow_denoiser(
      checkpoint,
      contract=DEFAULT_CONTRACT,
      normalizer=normalizer,
      solver_steps=cast(Any, True),
    )
  # A smaller grid than the process allows is legitimate at any level count.
  loaded = ri.load_flow_denoiser(
    checkpoint, contract=DEFAULT_CONTRACT, normalizer=normalizer, solver_steps=7
  )
  assert loaded.grid.calls == 7
  assert loaded.grid.levels[0] == 100


def test_flow_loader_selects_the_ema_shadow(tmp_path) -> None:
  normalizer = _normalizer()
  checkpoint = tmp_path / "flow.pt"
  model = _build_model(LinearNoiseProcess())
  ema = ExponentialMovingAverage(model)
  ema.updates = 11
  for value in ema.shadow.values():
    value.add_(0.5)
  _save_flow(checkpoint, dataset_identity=_identity(normalizer), ema=ema)

  loaded = ri.load_flow_denoiser(
    checkpoint, contract=DEFAULT_CONTRACT, normalizer=normalizer
  )
  assert loaded.ema_selected is True
  assert loaded.ema_updates == 11
  for name, value in loaded.model.state_dict().items():
    assert torch.equal(value, ema.shadow[name])


def test_flow_loader_reports_a_checkpoint_without_ema(tmp_path) -> None:
  normalizer = _normalizer()
  checkpoint = tmp_path / "flow.pt"
  _save_flow(checkpoint, dataset_identity=_identity(normalizer), ema=None)
  recorded = load_flow_checkpoint(checkpoint, model=_build_model(LinearNoiseProcess()))
  assert recorded.ema_state is None

  loaded = ri.load_flow_denoiser(
    checkpoint, contract=DEFAULT_CONTRACT, normalizer=normalizer
  )
  assert loaded.ema_selected is False
  assert loaded.ema_updates is None


def test_flow_loader_accepts_a_projection_bundle(tmp_path) -> None:
  bundle = ProjectionBundle.create(
    _stats(_STATE_DIM, seed=1), _stats(_LATENT_DIM, seed=2)
  )
  hashes = {
    "matrix": bundle.matrix_sha256,
    "pseudoinverse": bundle.pseudoinverse_sha256,
    "statistics": bundle.statistics_sha256,
  }
  checkpoint = tmp_path / "flow.pt"
  _save_flow(
    checkpoint,
    dataset_identity={"directory": "synthetic", "projection_hashes": hashes},
  )

  loaded = ri.load_flow_denoiser(
    checkpoint, contract=DEFAULT_CONTRACT, projection_bundle=bundle
  )
  assert loaded.projection_hashes == hashes


def test_flow_loader_rejects_a_diffusion_checkpoint(tmp_path) -> None:
  checkpoint = tmp_path / "diffusion.pt"
  model = StateLatentTransformer(DenoiserSettings.from_contract(DEFAULT_CONTRACT))
  save_checkpoint(
    checkpoint,
    model=model,
    ema=None,
    optimizer=None,
    scaler=None,
    config={"seed": 3},
    contract=DEFAULT_CONTRACT,
    schedule=DiffusionSchedule.build(),
    dataset_identity={},
    global_step=1,
    epoch=1,
    best_validation_loss=None,
    rng_state=None,
  )
  with pytest.raises(ri.ReplayError, match="unsupported or malformed"):
    ri.load_flow_denoiser(
      checkpoint, contract=DEFAULT_CONTRACT, normalizer=_normalizer()
    )


def test_diffusion_loader_still_rejects_a_flow_checkpoint(tmp_path) -> None:
  normalizer = _normalizer()
  checkpoint = tmp_path / "flow.pt"
  _save_flow(checkpoint, dataset_identity=_identity(normalizer))
  bundle = ProjectionBundle.create(
    _stats(_STATE_DIM, seed=1), _stats(_LATENT_DIM, seed=2)
  )
  with pytest.raises(ri.ReplayError, match="frozen denoiser checkpoint rejected"):
    ri.load_denoiser(checkpoint, contract=DEFAULT_CONTRACT, projection_bundle=bundle)


def test_flow_loader_fails_closed_on_a_foreign_contract_identity(tmp_path) -> None:
  normalizer = _normalizer()
  checkpoint = tmp_path / "flow.pt"
  _save_flow(checkpoint, dataset_identity=_identity(normalizer))
  changed = dataclass_replace(DEFAULT_CONTRACT, std_floor=1e-5)
  assert changed.identity_hash() != DEFAULT_CONTRACT.identity_hash()
  with pytest.raises(ri.ReplayError, match="rejected"):
    ri.load_flow_denoiser(checkpoint, contract=changed, normalizer=_normalizer(changed))


def test_flow_loader_fails_closed_on_a_normalizer_hash_mismatch(tmp_path) -> None:
  normalizer = _normalizer()
  other = _normalizer(seed=7)
  assert other.statistics_sha256 != normalizer.statistics_sha256
  checkpoint = tmp_path / "flow.pt"
  _save_flow(
    checkpoint,
    dataset_identity={
      "directory": "synthetic",
      "projection_hashes": {"statistics": "0" * 64},
    },
  )
  with pytest.raises(ri.ReplayError, match="statistics"):
    ri.load_flow_denoiser(checkpoint, contract=DEFAULT_CONTRACT, normalizer=normalizer)
  # A matching identity passes the same gate.
  good = tmp_path / "flowgood.pt"
  _save_flow(good, dataset_identity=_identity(other))
  loaded = ri.load_flow_denoiser(good, contract=DEFAULT_CONTRACT, normalizer=other)
  assert loaded.projection_hashes == {"statistics": other.statistics_sha256}
