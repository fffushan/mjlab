"""Device-placement regression tests for offline generation.

``evaluate_generation`` moves the model to the requested device itself.  A CPU
model evaluated with ``device="cuda"`` previously reached the first ``Linear``
and failed with a device mismatch, so these tests assert the placement rather
than only that a run completes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from mjlab.tasks.tracking.diffusion.contract import DiffusionContract
from mjlab.tasks.tracking.diffusion.evaluation import evaluate_generation
from mjlab.tasks.tracking.diffusion.projection import FeatureStats, ProjectionBundle
from mjlab.tasks.tracking.diffusion.schedule import (
  DiffusionSchedule,
  build_inference_grid,
)
from mjlab.tasks.tracking.diffusion.window_dataset import WindowRecord

CONTRACT_PATH = (
  Path(__file__).resolve().parents[1]
  / "docs"
  / "plans"
  / "beyondmimic_diffusion_d0_contract.yaml"
)
CONTRACT = DiffusionContract.from_yaml(CONTRACT_PATH)
SCHEDULE = DiffusionSchedule.from_contract(CONTRACT)


class _TinyDenoiser(torch.nn.Module):
  """Minimal denoiser; the real transformer is exercised by its own module."""

  def __init__(self) -> None:
    super().__init__()
    self.scale = torch.nn.Parameter(torch.zeros(1))

  def forward(self, tokens: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
    if step_ids.shape[-1] != 2:
      raise ValueError("step ids must be [B,T,2]")
    return tokens * (1.0 + self.scale)


def _projection() -> ProjectionBundle:
  state = FeatureStats(
    count=8,
    mean=np.zeros(CONTRACT.state_dimension),
    std=np.ones(CONTRACT.state_dimension),
  )
  latent = FeatureStats(
    count=8,
    mean=np.zeros(CONTRACT.latent_dimension),
    std=np.ones(CONTRACT.latent_dimension),
  )
  return ProjectionBundle.create(state, latent, contract=CONTRACT)


def _records(count: int) -> list[WindowRecord]:
  generator = np.random.default_rng(0)
  return [
    WindowRecord(
      tokens=generator.standard_normal(
        (CONTRACT.window_steps, CONTRACT.token_dimension)
      ).astype(np.float32),
      split="test",
      group_key=f"group:{index}",
      motion_id="tennis_000",
      phase="clean",
      ou_noise_norm=0.0,
      pair_id=f"pair:{index}",
      start_tick=index,
    )
    for index in range(count)
  ]


def _evaluate(model: torch.nn.Module, device: str) -> int:
  report = evaluate_generation(
    model,
    schedule=SCHEDULE,
    grid=build_inference_grid(SCHEDULE, contract=CONTRACT),
    records=_records(2),
    projection=_projection(),
    contract=CONTRACT,
    seed=0,
    device=device,
    max_windows=2,
  )
  return report.windows


def test_evaluate_generation_places_the_model_on_the_requested_device() -> None:
  """A CPU model with device="cuda" used to fail on the first Linear."""
  if not torch.cuda.is_available():
    pytest.skip("requires CUDA to observe model placement")
  model = _TinyDenoiser()
  assert {parameter.device.type for parameter in model.parameters()} == {"cpu"}
  assert _evaluate(model, "cuda:0") == 2
  assert {parameter.device.type for parameter in model.parameters()} == {"cuda"}


def test_evaluate_generation_runs_on_cpu() -> None:
  """The CPU path keeps working and reports the scored window count."""
  model = _TinyDenoiser()
  assert _evaluate(model, "cpu") == 2
  assert {parameter.device.type for parameter in model.parameters()} == {"cpu"}
