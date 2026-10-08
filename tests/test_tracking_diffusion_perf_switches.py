"""Focused tests for the diffusion performance switches.

l1, l4 and l6 are default-on. The environment variables remain explicit
opt-out controls for matched comparisons and rollback.
"""

from __future__ import annotations

import pytest
import torch

from mjlab.tasks.tracking.diffusion.perf_switches import PerfSwitches
from mjlab.tasks.tracking.diffusion.trainer import (
  ExponentialMovingAverage,
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
    "MJLAB_DIFFUSION_EMA_FOREACH",
    "MJLAB_DIFFUSION_COMPILED_FORWARD",
    "MJLAB_DIFFUSION_PERF_LABEL",
  ):
    monkeypatch.delenv(name, raising=False)


def test_adopted_switches_default_on() -> None:
  switches = PerfSwitches.from_environment()
  assert switches.finite_combined is True
  assert switches.metrics_single_read is False
  assert switches.loader_workers == 0
  assert switches.ema_foreach is True
  assert switches.compiled_forward is True
  assert switches.label == "finite+ema4+compiled"
  assert switches.active is True


def test_adopted_switches_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("MJLAB_DIFFUSION_FINITE_COMBINED", "0")
  monkeypatch.setenv("MJLAB_DIFFUSION_EMA_FOREACH", "0")
  monkeypatch.setenv("MJLAB_DIFFUSION_COMPILED_FORWARD", "0")
  switches = PerfSwitches.from_environment()
  assert switches.finite_combined is False
  assert switches.ema_foreach is False
  assert switches.compiled_forward is False
  assert switches.label == "default"
  assert switches.active is False


def test_switches_read_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("MJLAB_DIFFUSION_FINITE_COMBINED", "1")
  monkeypatch.setenv("MJLAB_DIFFUSION_METRICS_SINGLE_READ", "true")
  monkeypatch.setenv("MJLAB_DIFFUSION_LOADER_WORKERS", "4")
  monkeypatch.setenv("MJLAB_DIFFUSION_EMA_FOREACH", "1")
  monkeypatch.setenv("MJLAB_DIFFUSION_COMPILED_FORWARD", "on")
  switches = PerfSwitches.from_environment()
  assert switches.finite_combined is True
  assert switches.metrics_single_read is True
  assert switches.loader_workers == 4
  assert switches.ema_foreach is True
  assert switches.compiled_forward is True
  assert switches.label == "finite+metrics1+workers4+ema4+compiled"
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


def test_ema_foreach_is_bit_identical() -> None:
  """The batched EMA form must reproduce the per-tensor form exactly."""

  def build(seed: int) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(seed)
    module = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    # A non-float buffer exercises the copy branch of both forms.
    module.register_buffer("count", torch.tensor(3, dtype=torch.int64))
    return module, torch.optim.AdamW(module.parameters(), lr=1e-3)

  model_a, optimiser_a = build(0)
  model_b, optimiser_b = build(0)
  ema_a = ExponentialMovingAverage(model_a)
  ema_b = ExponentialMovingAverage(model_b)
  for _ in range(5):
    x = torch.randn(6, 4)
    loss_a = model_a(x).pow(2).mean()
    loss_a.backward()
    optimiser_a.step()
    optimiser_a.zero_grad(set_to_none=True)
    loss_b = model_b(x).pow(2).mean()
    loss_b.backward()
    optimiser_b.step()
    optimiser_b.zero_grad(set_to_none=True)
    decay_a = ema_a.update(model_a, foreach=False)
    decay_b = ema_b.update(model_b, foreach=True)
    assert decay_a == decay_b
  assert ema_a.shadow.keys() == ema_b.shadow.keys()
  for name, value in ema_a.shadow.items():
    assert torch.equal(value, ema_b.shadow[name]), f"{name} differs between forms"


def test_compiled_wrapper_leaves_module_state_dict_unrenamed() -> None:
  """Compiling a callable over a module keeps the module's keys intact.

  This documents the trap: compiling the module itself renames every state-dict
  entry with an ``_orig_mod.`` prefix, which breaks checkpoint resume.
  """
  module = torch.nn.Linear(4, 2)
  before = list(module.state_dict().keys())
  callable_ = torch.compile(lambda x: module(x), dynamic=False)
  callable_(torch.randn(3, 4))
  assert list(module.state_dict().keys()) == before
