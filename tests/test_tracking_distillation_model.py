"""CPU contracts for the M2 conditional VAE and controlled normalizers."""

from __future__ import annotations

import inspect

import pytest
import torch

from mjlab.tasks.tracking.distillation.model import (
  ConditionalVAE,
  ModelValidationError,
  StudentNormalizer,
  VaeLoss,
  vae_loss,
)
from mjlab.tasks.tracking.distillation.vae_config import (
  DEFAULT_SCHEMA,
  DecoderMode,
  ModelSettings,
  make_schema,
  schema_for_mode,
)


def test_loss_sums_joint_and_latent_terms_before_batch_mean() -> None:
  predicted = torch.zeros(2, 31)
  predicted[0, :2] = torch.tensor([1.0, 3.0])
  predicted[1, :2] = torch.tensor([2.0, 2.0])
  teacher = torch.zeros_like(predicted)
  mu = torch.zeros(2, 32)
  mu[0, 0] = 1.0
  mu[1, 1] = 2.0
  logvar = torch.zeros_like(mu)
  result = vae_loss(predicted, teacher, mu, logvar, beta=0.5)

  # reconstruction=(10+8)/2; KL=(1/2+2)/2, with the 0.5 beta applied last.
  assert isinstance(result, VaeLoss)
  assert result.reconstruction.item() == pytest.approx(9.0)
  assert result.kl.item() == pytest.approx(1.25)
  assert result.total.item() == pytest.approx(9.625)


def test_loss_detaches_fixed_teacher_targets_but_keeps_prediction_gradients() -> None:
  predicted = torch.zeros(1, 31, requires_grad=True)
  teacher = torch.ones(1, 31, requires_grad=True)
  mu = torch.ones(1, 32, requires_grad=True)
  logvar = torch.zeros(1, 32, requires_grad=True)
  result = vae_loss(predicted, teacher, mu, logvar)
  result.total.backward()

  assert teacher.grad is None
  assert predicted.grad is not None
  assert mu.grad is not None
  assert logvar.grad is not None

  empty = torch.empty((0, 31))
  with pytest.raises(ModelValidationError, match="non-empty"):
    vae_loss(empty, empty, torch.empty((0, 32)), torch.empty((0, 32)))
  with pytest.raises(ModelValidationError, match="beta"):
    vae_loss(
      torch.zeros(1, 31),
      torch.zeros(1, 31),
      torch.zeros(1, 32),
      torch.zeros(1, 32),
      beta=-1,
    )
  with pytest.raises(ModelValidationError, match="non-finite"):
    vae_loss(
      torch.full((1, 31), float("nan")),
      torch.zeros(1, 31),
      torch.zeros(1, 32),
      torch.zeros(1, 32),
    )


def test_normalizer_moments_freeze_and_atomic_rejection() -> None:
  normalizer = StudentNormalizer(2, eps=1e-5)
  normalizer.update(torch.tensor([[1.0, 3.0], [3.0, 7.0]]))
  assert normalizer.state_dict()["count"].item() == 2.0
  torch.testing.assert_close(normalizer.state_dict()["mean"], torch.tensor([2.0, 5.0]))
  torch.testing.assert_close(normalizer.variance, torch.tensor([1.0, 4.0]))
  before = {name: value.clone() for name, value in normalizer.state_dict().items()}
  with pytest.raises(ModelValidationError, match="non-finite"):
    normalizer.update(torch.tensor([[float("nan"), 4.0]]))
  for name, value in before.items():
    torch.testing.assert_close(normalizer.state_dict()[name], value)
  normalizer.freeze()
  with pytest.raises(RuntimeError, match="frozen"):
    normalizer.update(torch.ones(1, 2))
  assert normalizer.frozen
  normalizer.unfreeze()
  normalizer.update(torch.ones(1, 2))
  assert normalizer.state_dict()["count"].item() == 3.0


def test_normalizer_updates_are_detached_and_repeated_backward_is_safe() -> None:
  normalizer = StudentNormalizer(2)
  source = torch.tensor([[1.0, 3.0], [3.0, 7.0]], requires_grad=True)
  normalizer.update(source)

  for statistic in (normalizer.count, normalizer.mean, normalizer.m2):
    assert statistic.requires_grad is False
    assert statistic.grad_fn is None

  first = torch.tensor([[2.0, 4.0]], requires_grad=True)
  normalizer(first).sum().backward()
  second = torch.tensor([[5.0, 8.0]], requires_grad=True)
  normalizer(second).sum().backward()
  assert source.grad is None
  assert first.grad is not None
  assert second.grad is not None

  normalizer = StudentNormalizer(2)
  samples = torch.tensor([[2.0, -1.0]])
  before = normalizer.state_dict()["count"].clone()
  output = normalizer(samples)
  assert output.shape == samples.shape
  torch.testing.assert_close(normalizer.count, before)


def test_normalizer_epsilon_roundtrips_standalone_and_nested() -> None:
  standalone = StudentNormalizer(2, eps=0.5)
  standalone.update(torch.tensor([[1.0, 2.0], [3.0, 6.0]]))
  restored = StudentNormalizer(2, eps=1e-5)
  restored.load_state_dict(standalone.state_dict())
  assert restored.eps == pytest.approx(0.5)
  samples = torch.tensor([[2.0, 4.0]])
  torch.testing.assert_close(restored(samples), standalone(samples))

  settings = ModelSettings(hidden_dims=(8,))
  model = ConditionalVAE(normalizer_eps=0.125, settings=settings)
  model.reference_normalizer.update(torch.randn(4, 68))
  model.conditioning_normalizer.update(torch.randn(4, 99))
  state = model.state_dict()
  nested = ConditionalVAE(normalizer_eps=1e-5, settings=settings)
  nested.load_state_dict(state)
  assert nested.reference_normalizer.eps == pytest.approx(0.125)
  assert nested.conditioning_normalizer.eps == pytest.approx(0.125)
  reference = torch.randn(2, 68)
  conditioning = torch.randn(2, 99)
  torch.testing.assert_close(
    nested.mean_inference(reference, conditioning),
    model.mean_inference(reference, conditioning),
  )


def make_model() -> ConditionalVAE:
  return ConditionalVAE(
    DEFAULT_SCHEMA,
    ModelSettings(hidden_dims=(16, 8), beta=0.01),
  )


def test_cold_start_normalization_is_finite_for_fixed_seeds() -> None:
  for seed in (0, 1, 2, 7):
    torch.manual_seed(seed)
    model = make_model()
    output = model.sampled_inference(torch.randn(3, 68), torch.randn(3, 99))
    result = vae_loss(output.action, torch.zeros(3, 31), output.mu, output.logvar)
    assert torch.isfinite(result.total)
    result.total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_mean_and_sampled_inference_are_explicit_and_reparameterized() -> None:
  model = make_model()
  reference = torch.randn(3, DEFAULT_SCHEMA.reference_dim)
  conditioning = torch.randn(3, DEFAULT_SCHEMA.conditioning_dim)
  noise = torch.ones(3, 32)
  mean = model.mean_inference(reference, conditioning)
  sampled = model.sampled_inference(reference, conditioning, noise=noise)
  direct = model.forward(reference, conditioning, sample=True, noise=noise)
  assert mean.shape == (3, 31)
  torch.testing.assert_close(sampled.action, direct.action)
  torch.testing.assert_close(
    sampled.latent, sampled.mu + torch.exp(0.5 * sampled.logvar)
  )
  assert inspect.signature(model.decode).parameters.keys() == {"latent", "conditioning"}


def test_reparameterized_sample_has_gradients_and_noise_is_reproducible() -> None:
  model = make_model()
  reference = torch.randn(2, 68, requires_grad=True)
  conditioning = torch.randn(2, 99)
  noise = torch.randn(2, 32)
  first = model.sampled_inference(reference, conditioning, noise=noise)
  second = model.sampled_inference(reference, conditioning, noise=noise)
  torch.testing.assert_close(first.action, second.action)
  first.action.sum().backward()
  assert reference.grad is not None
  assert any(parameter.grad is not None for parameter in model.parameters())


def test_model_normalizer_state_and_schema_identity_roundtrip() -> None:
  model = make_model()
  model.reference_normalizer.update(torch.randn(4, 68))
  model.conditioning_normalizer.update(torch.randn(4, 99))
  state = model.state_dict()
  restored = make_model()
  restored.load_state_dict(state)
  assert restored.schema_metadata == model.schema_metadata
  assert restored.reference_normalizer.frozen is False
  torch.testing.assert_close(
    restored.reference_normalizer.mean, model.reference_normalizer.mean
  )
  with pytest.raises(ModelValidationError, match="schema"):
    ConditionalVAE(
      schema_for_mode(DecoderMode.ANCHOR),
      ModelSettings(hidden_dims=(16, 8)),
    ).load_state_dict(state)
  with pytest.raises(ModelValidationError, match="schema"):
    ConditionalVAE(
      make_schema(DecoderMode.GRAVITY, tuple(f"other_{i}" for i in range(31))),
      ModelSettings(hidden_dims=(16, 8)),
    ).load_state_dict(state)


def test_default_paper_size_cpu_forward_shape() -> None:
  model = ConditionalVAE()
  output = model(
    torch.zeros(1, 68),
    torch.zeros(1, 99),
  )
  assert output.action.shape == (1, 31)
  assert output.mu.shape == output.logvar.shape == output.latent.shape == (1, 32)
