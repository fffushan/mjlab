"""Tests for the frozen NumPy schedule and DDIM grid."""

from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from mjlab.tasks.tracking.diffusion import schedule as schedule_module
from mjlab.tasks.tracking.diffusion.contract import DiffusionContract
from mjlab.tasks.tracking.diffusion.schedule import (
  DiffusionSchedule,
  InferenceGrid,
  ScheduleError,
  build_inference_grid,
)


def test_schedule_keeps_torch_out_of_module_scope() -> None:
  source = Path(schedule_module.__file__).read_text()
  tree = ast.parse(source)
  assert not any(
    isinstance(node, ast.Import)
    and any(alias.name == "torch" for alias in node.names)
    or isinstance(node, ast.ImportFrom)
    and node.module == "torch"
    for node in tree.body
  )


def test_schedule_matches_d0_values_and_self_hash() -> None:
  schedule = DiffusionSchedule.build()
  assert schedule.alpha_bars.shape == (1001,)
  assert schedule.betas.shape == (1000,)
  assert schedule.alpha_bars[0] == 1.0
  for index, expected in {
    1: 0.999958715775178,
    50: 0.9920072786842186,
    500: 0.49384359044063775,
    1000: 2.4287669070348542e-09,
  }.items():
    assert math.isclose(
      float(schedule.alpha_bars[index]), expected, rel_tol=4e-16, abs_tol=0.0
    )
  assert math.isclose(
    schedule.terminal_snr, 2.428766912933763e-09, rel_tol=4e-16, abs_tol=0.0
  )
  assert math.isclose(float(schedule.betas.min()), 4.128422482196914e-05, rel_tol=4e-16)
  assert schedule.betas.max() == 0.999
  assert (
    schedule.sha256()
    == hashlib.sha256(
      np.ascontiguousarray(schedule.alpha_bars, dtype=np.float64).tobytes()
    ).hexdigest()
  )


def test_schedule_agrees_with_d0_torch_float64_oracle() -> None:
  schedule = DiffusionSchedule.build()
  x = torch.linspace(0, 1000, 1001, dtype=torch.float64)
  raw = torch.cos(((x / 1000) + 0.008) / 1.008 * math.pi / 2).square()
  raw = raw / raw[0]
  betas = (1.0 - raw[1:] / raw[:-1]).clamp(1.0e-5, 0.999)
  oracle = torch.cat(
    (torch.ones(1, dtype=torch.float64), torch.cumprod(1.0 - betas, 0))
  ).numpy()
  delta = np.abs(schedule.alpha_bars - oracle)
  assert float(delta.max()) <= 1.0e-12
  assert int(delta.argmax()) == 732
  assert float(delta[732]) == pytest.approx(2.7755575615628914e-17)


def test_schedule_save_load_and_tamper_rejection(tmp_path) -> None:
  schedule = DiffusionSchedule.build()
  path = tmp_path / "schedule.npz"
  schedule.save(path)
  loaded = DiffusionSchedule.load(path)
  assert np.array_equal(loaded.betas, schedule.betas)
  assert np.array_equal(loaded.alpha_bars, schedule.alpha_bars)

  with np.load(path, allow_pickle=False) as data:
    tampered_betas = np.array(data["betas"], copy=True)
    tampered_alpha = np.array(data["alpha_bars"], copy=True)
    metadata = data["metadata"]
  tampered_alpha[10] = np.nextafter(tampered_alpha[10], 0.0)
  tampered_path = tmp_path / "tampered.npz"
  np.savez_compressed(
    tampered_path,
    betas=tampered_betas,
    alpha_bars=tampered_alpha,
    metadata=metadata,
  )
  with pytest.raises(ScheduleError, match="schedule"):
    DiffusionSchedule.load(tampered_path)


def test_frozen_inference_grid_and_contract_guard() -> None:
  schedule = DiffusionSchedule.build()
  grid = InferenceGrid.uniform(schedule)
  assert grid.source_ids == tuple(range(1000, 0, -50))
  assert grid.destination_ids == tuple(range(950, -1, -50))
  guarded = build_inference_grid(schedule, contract=DiffusionContract())
  assert guarded == grid
  with pytest.raises(ScheduleError, match="20 updates"):
    InferenceGrid.uniform(schedule, updates=10)
