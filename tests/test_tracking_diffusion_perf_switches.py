"""Focused tests for the environment-gated performance switches.

Every switch must default to current behaviour, because the training arms run
with the same code and set none of these variables. The two value-preserving
switches are tested for exact parity, not for approximate agreement.
"""

from __future__ import annotations

import pytest
import torch

from mjlab.tasks.tracking.diffusion.perf_switches import PerfSwitches
from mjlab.tasks.tracking.diffusion.trainer import (
  TrainerError,
  _finite_tensor,
  _metrics,
)


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
  for name in (
    "MJLAB_DIFFUSION_FINITE_COMBINED",
    "MJLAB_DIFFUSION_METRICS_SINGLE_READ",
    "MJLAB_DIFFUSION_LOADER_WORKERS",
    "MJLAB_DIFFUSION_PERF_LABEL",
  ):
    monkeypatch.delenv(name, raising=False)


def test_all_switches_default_off() -> None:
  switches = PerfSwitches.from_environment()
  assert switches.finite_combined is False
  assert switches.metrics_single_read is False
  assert switches.loader_workers == 0
  assert switches.label == "default"
  assert switches.active is False


def test_switches_read_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("MJLAB_DIFFUSION_FINITE_COMBINED", "1")
  monkeypatch.setenv("MJLAB_DIFFUSION_METRICS_SINGLE_READ", "true")
  monkeypatch.setenv("MJLAB_DIFFUSION_LOADER_WORKERS", "4")
  switches = PerfSwitches.from_environment()
  assert switches.finite_combined is True
  assert switches.metrics_single_read is True
  assert switches.loader_workers == 4
  assert switches.label == "finite+metrics1+workers4"
  assert switches.active is True


def test_explicit_label_wins(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("MJLAB_DIFFUSION_FINITE_COMBINED", "1")
  monkeypatch.setenv("MJLAB_DIFFUSION_PERF_LABEL", "wave1-l1")
  assert PerfSwitches.from_environment().label == "wave1-l1"


def test_negative_worker_count_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("MJLAB_DIFFUSION_LOADER_WORKERS", "-2")
  with pytest.raises(ValueError, match="must not be negative"):
    PerfSwitches.from_environment()


def _metric_inputs(
  seed: int = 0, batch: int = 6, steps: int = 5
) -> tuple[torch.Tensor, ...]:
  generator = torch.Generator().manual_seed(seed)
  prediction = torch.randn(batch, steps, 12, generator=generator)
  target = torch.randn(batch, steps, 12, generator=generator)
  k_state = torch.randint(0, 4, (batch, steps), generator=generator)
  k_latent = torch.randint(0, 1, (batch, steps), generator=generator)
  unknown = torch.rand(batch, steps, 12, generator=generator) > 0.5
  return prediction, target, k_state, k_latent, unknown


def test_single_read_metrics_are_exactly_equal() -> None:
  inputs = _metric_inputs()
  per_item = _metrics(*inputs)
  single = _metrics(*inputs, single_read=True)
  assert per_item.keys() == single.keys()
  for name in per_item:
    assert per_item[name] == single[name], (
      f"{name} differs: {per_item[name]} vs {single[name]}"
    )


def test_single_read_metrics_agree_with_zero_masks() -> None:
  prediction, target, k_state, k_latent, _ = _metric_inputs(seed=3)
  unknown = torch.zeros_like(target, dtype=torch.bool)
  per_item = _metrics(prediction, target, k_state, k_latent, unknown)
  single = _metrics(prediction, target, k_state, k_latent, unknown, single_read=True)
  assert per_item == single


def test_stacked_check_passes_finite_and_names_failures() -> None:
  good = [torch.ones(4), torch.zeros(3)]
  _finite_tensor(torch.cat([t.reshape(-1) for t in good]), "stacked parameters")

  bad = [torch.ones(4), torch.tensor([1.0, float("nan")])]
  with pytest.raises(TrainerError, match="stacked parameters"):
    _finite_tensor(torch.cat([t.reshape(-1) for t in bad]), "stacked parameters")
