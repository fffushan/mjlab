"""CPU tests for the opt-in device-native tensor policy path.

The tensor methods are an alternative to the NumPy observation/action round
trip, not a replacement: the scalar and NumPy batch paths must stay unchanged.
CUDA parity is authored here but never executed in this lane; set
``MJLAB_D1_TENSOR_CUDA_TESTS=1`` on a GPU host to run those cases.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from tracking_diffusion_fixtures import make_state

from mjlab.tasks.tracking.diffusion import (
  ActionContract,
  FrozenVaePolicy,
  VaeObservation,
  WorldState,
)
from mjlab.tasks.tracking.diffusion.adapter import (
  AdapterError,
  VaeObservationBatch,
  VaeObservationTensorBatch,
  resolve_inference_device,
)
from mjlab.tasks.tracking.diffusion.runtime_x2 import (
  RuntimeX2Error,
  X2MultiMotionEnvironment,
  resolve_policy_device,
)

_CUDA_AUTHORIZED = os.environ.get("MJLAB_D1_TENSOR_CUDA_TESTS") == "1"


class _Schema:
  reference_dim = 68
  conditioning_dim = 99


class _TensorVae(torch.nn.Module):
  """Deterministic fake whose outputs expose reference/conditioning identity."""

  schema = _Schema()

  def __init__(self) -> None:
    super().__init__()
    self.bias = torch.nn.Parameter(torch.zeros(()))
    self.grad_enabled: list[bool] = []

  def encode(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    self.grad_enabled.append(torch.is_grad_enabled())
    mu = reference[:, :1].repeat(1, 32) + self.bias
    # A non-trivial logvar proves the caller uses the mean, not a sampled z.
    return mu, torch.full_like(mu, 4.0)

  def decode(self, latent: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
    self.grad_enabled.append(torch.is_grad_enabled())
    return latent[:, :1].repeat(1, 31) + conditioning[:, :1]


class _NonFiniteVae(torch.nn.Module):
  schema = _Schema()

  def encode(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu = reference[:, :1].repeat(1, 32)
    return mu, torch.zeros_like(mu)

  def decode(self, latent: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
    return torch.full((latent.shape[0], 31), float("nan"))


class _ShortVae(torch.nn.Module):
  schema = _Schema()

  def encode(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu = reference[:, :1].repeat(1, 32)
    return mu, torch.zeros_like(mu)

  def decode(self, latent: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
    return latent[:, :1]


def _contract() -> ActionContract:
  return ActionContract(np.ones(31), np.zeros(31), np.ones(31), np.ones(31))


def _vae_observation_batch(
  reference: np.ndarray, conditioning: np.ndarray
) -> VaeObservationBatch:
  rows = reference.shape[0]
  return VaeObservationBatch(
    WorldState.stack([make_state(index) for index in range(rows)]),
    reference,
    conditioning,
    tuple("motion" for _ in range(rows)),
    np.arange(rows, dtype=np.int64),
    np.zeros(rows),
  )


def test_tensor_vae_batch_matches_numpy_batch_and_returns_posterior_mean() -> None:
  model = _TensorVae()
  vae = FrozenVaePolicy(model, _contract())
  reference = np.linspace(0.0, 1.0, 2 * 68, dtype=np.float32).reshape(2, 68)
  conditioning = np.full((2, 99), 0.25, dtype=np.float32)

  latent_np, clean_np = vae.batch_action(
    _vae_observation_batch(reference, conditioning)
  )
  observation = VaeObservationTensorBatch(
    torch.as_tensor(reference), torch.as_tensor(conditioning)
  )
  latent, clean = vae.batch_action_tensor(observation)

  assert latent.dtype == torch.float32 and clean.dtype == torch.float32
  assert latent.device.type == "cpu" and clean.device.type == "cpu"
  assert latent.shape == (2, 32) and clean.shape == (2, 31)
  mean = torch.as_tensor(reference[:, :1]).repeat(1, 32)
  torch.testing.assert_close(latent, mean, rtol=0.0, atol=0.0)
  np.testing.assert_allclose(latent.numpy(), latent_np, rtol=0.0, atol=0.0)
  np.testing.assert_allclose(clean.numpy(), clean_np, rtol=0.0, atol=0.0)


def test_tensor_vae_batch_runs_frozen_without_gradients() -> None:
  model = _TensorVae()
  vae = FrozenVaePolicy(model, _contract())
  observation = VaeObservationTensorBatch(torch.ones(2, 68), torch.zeros(2, 99))

  latent, clean = vae.batch_action_tensor(observation)

  assert model.grad_enabled == [False, False]
  assert torch.is_grad_enabled() is True
  assert latent.requires_grad is False and clean.requires_grad is False
  assert model.training is False
  assert all(not parameter.requires_grad for parameter in model.parameters())


def test_tensor_observation_refuses_bad_shape_dtype_device_and_nonfinite() -> None:
  valid_reference = torch.zeros(2, 68)
  valid_conditioning = torch.zeros(2, 99)
  with pytest.raises(AdapterError, match="reference"):
    VaeObservationTensorBatch(torch.zeros(2, 67), valid_conditioning)
  with pytest.raises(AdapterError, match="conditioning"):
    VaeObservationTensorBatch(valid_reference, torch.zeros(2, 98))
  with pytest.raises(AdapterError, match="float32"):
    VaeObservationTensorBatch(valid_reference.double(), valid_conditioning)
  with pytest.raises(AdapterError, match="device"):
    VaeObservationTensorBatch(valid_reference, torch.zeros(2, 99, device="meta"))
  with pytest.raises(AdapterError, match="finite"):
    bad = valid_reference.clone()
    bad[0, 0] = float("nan")
    VaeObservationTensorBatch(bad, valid_conditioning)
  with pytest.raises(AdapterError, match="shape"):
    VaeObservationTensorBatch(torch.zeros(0, 68), torch.zeros(0, 99))
  not_tensors = cast(Any, (np.zeros((2, 68)), np.zeros((2, 99))))
  with pytest.raises(AdapterError, match="torch.Tensor"):
    VaeObservationTensorBatch(*not_tensors)
  with pytest.raises(AdapterError, match="row evidence"):
    VaeObservationTensorBatch(
      valid_reference,
      valid_conditioning,
      WorldState.stack([make_state(0)]),
      ("motion",),
    )
  with pytest.raises(AdapterError, match="frames and phases"):
    VaeObservationTensorBatch(
      valid_reference,
      valid_conditioning,
      WorldState.stack([make_state(0), make_state(1)]),
      ("motion", "motion"),
    )


def test_tensor_vae_batch_refuses_non_observation_and_invalid_output() -> None:
  vae = FrozenVaePolicy(_TensorVae(), _contract())
  with pytest.raises(AdapterError, match="VaeObservationTensorBatch"):
    vae.batch_action_tensor(cast(Any, np.zeros((1, 68))))
  observation = VaeObservationTensorBatch(torch.ones(1, 68), torch.zeros(1, 99))
  with pytest.raises(AdapterError, match="non-finite"):
    FrozenVaePolicy(_NonFiniteVae(), _contract()).batch_action_tensor(observation)
  with pytest.raises(AdapterError, match="invalid latent or action"):
    FrozenVaePolicy(_ShortVae(), _contract()).batch_action_tensor(observation)


def test_legacy_cpu_numpy_path_is_unchanged() -> None:
  vae = FrozenVaePolicy(_TensorVae(), _contract())
  latent, action = vae.action(
    VaeObservation(make_state(), np.full(68, 0.5), np.full(99, 0.25), "m", 0, 0.0)
  )
  assert isinstance(latent, np.ndarray) and latent.dtype == np.float64
  assert isinstance(action, np.ndarray) and action.dtype == np.float64
  batch_latent, batch_action = vae.batch_action(
    _vae_observation_batch(np.full((1, 68), 0.5), np.full((1, 99), 0.25))
  )
  assert batch_latent.dtype == np.float64 and batch_action.dtype == np.float64


def test_resolve_inference_device_normalizes_and_fails_closed() -> None:
  assert resolve_inference_device("cpu") == "cpu"
  assert resolve_inference_device(torch.device("cpu")) == "cpu"
  with pytest.raises(AdapterError):
    resolve_inference_device("")
  with pytest.raises(AdapterError, match="cpu or cuda"):
    resolve_inference_device("meta")
  with pytest.raises(AdapterError):
    resolve_inference_device("tpu")
  if torch.cuda.is_available():
    assert resolve_inference_device("cuda:0") == "cuda:0"
  else:
    with pytest.raises(AdapterError, match="CUDA"):
      resolve_inference_device("cuda:0")


def test_resolve_policy_device_defaults_to_cpu_and_tracks_simulation() -> None:
  assert resolve_policy_device("", simulation_device="cuda:0") == "cpu"
  assert resolve_policy_device(" CPU ", simulation_device="cuda:0") == "cpu"
  assert resolve_policy_device("simulation", simulation_device="cpu") == "cpu"
  with pytest.raises(RuntimeX2Error):
    resolve_policy_device("tpu", simulation_device="cpu")
  if torch.cuda.is_available():
    assert resolve_policy_device("cuda:0", simulation_device="cpu") == "cuda:0"
    assert resolve_policy_device("simulation", simulation_device="cuda:0") == "cuda:0"
  else:
    with pytest.raises(RuntimeX2Error, match="CUDA"):
      resolve_policy_device("cuda:0", simulation_device="cpu")


def _tensor_runtime(
  num_envs: int,
  reference: torch.Tensor,
  conditioning: torch.Tensor,
  *,
  policy_device: str = "cpu",
  sim_device: str = "cpu",
) -> X2MultiMotionEnvironment:
  runtime = X2MultiMotionEnvironment.__new__(X2MultiMotionEnvironment)
  runtime.num_envs = num_envs
  runtime.device = sim_device
  runtime.policy_device = torch.device(policy_device)
  runtime._snapshot = lambda: SimpleNamespace(
    packed=SimpleNamespace(reference=reference, conditioning=conditioning)
  )
  runtime.command = SimpleNamespace(
    time_steps=torch.arange(num_envs, dtype=torch.long),
    row_lengths=torch.full((num_envs,), 11, dtype=torch.long),
    motion_ids=torch.zeros(num_envs, dtype=torch.long),
  )
  runtime.teacher_ids = ("motion",)
  runtime._world_state = lambda ids: WorldState.stack(
    make_state(int(index)) for index in np.asarray(ids, dtype=np.int64).tolist()
  )
  return runtime


def test_observe_vae_tensor_batch_gathers_noncontiguous_slots() -> None:
  reference = torch.arange(4 * 68, dtype=torch.float32).reshape(4, 68)
  conditioning = torch.arange(4 * 99, dtype=torch.float32).reshape(4, 99)
  runtime = _tensor_runtime(4, reference, conditioning)

  observation = runtime.observe_vae_tensor_batch(np.asarray([2, 0]))

  assert observation.batch_size == 2
  assert observation.device.type == "cpu"
  torch.testing.assert_close(observation.reference, reference[[2, 0]])
  torch.testing.assert_close(observation.conditioning, conditioning[[2, 0]])
  # The same capture carries the CPU row evidence for the collector's rows.
  assert observation.motion_id == ("motion", "motion")
  np.testing.assert_array_equal(observation.reference_frame, np.asarray([2, 0]))
  assert observation.state.count == 2
  with pytest.raises(RuntimeX2Error):
    runtime.observe_vae_tensor_batch(np.asarray([0, 0]))
  with pytest.raises(RuntimeX2Error):
    runtime.observe_vae_tensor_batch(np.asarray([4]))
  with pytest.raises(RuntimeX2Error):
    runtime.observe_vae_tensor_batch(np.asarray([], dtype=np.int64))


def test_observe_vae_tensor_batch_casts_inputs_to_float32() -> None:
  reference = torch.zeros(2, 68, dtype=torch.float64)
  conditioning = torch.zeros(2, 99, dtype=torch.float64)
  runtime = _tensor_runtime(2, reference, conditioning)

  observation = runtime.observe_vae_tensor_batch(np.asarray([1, 0]))

  # The frozen VAE inference path has always consumed float32 inputs.
  assert observation.reference.dtype == torch.float32
  assert observation.conditioning.dtype == torch.float32
  np.testing.assert_array_equal(observation.reference_frame, np.asarray([1, 0]))
  np.testing.assert_allclose(observation.reference_phase, np.asarray([0.1, 0.0]))


class _FakeSimEnv:
  def __init__(self) -> None:
    self.received: list[torch.Tensor] = []
    self.result = object()

  def step(self, action: torch.Tensor) -> object:
    self.received.append(action)
    return self.result


def _step_runtime(num_envs: int = 3) -> tuple[X2MultiMotionEnvironment, Any]:
  runtime = X2MultiMotionEnvironment.__new__(X2MultiMotionEnvironment)
  runtime.num_envs = num_envs
  runtime.device = "cpu"
  runtime.policy_device = torch.device("cpu")
  runtime.command = SimpleNamespace(
    time_steps=torch.tensor([5, 6, 7], dtype=torch.long)
  )
  fake = _FakeSimEnv()
  runtime.env = fake
  runtime._last_result = None
  return runtime, fake


def test_step_tensor_batch_matches_step_batch_bookkeeping() -> None:
  tensor_runtime, tensor_env = _step_runtime()
  array_runtime, array_env = _step_runtime()

  tensor_result = tensor_runtime.step_tensor_batch(torch.full((3, 31), 0.5))
  array_result = array_runtime.step_batch(np.full((3, 31), 0.5))

  assert tensor_result is tensor_env.result
  assert array_result is array_env.result
  torch.testing.assert_close(
    tensor_runtime._pre_frames, tensor_runtime.command.time_steps
  )
  torch.testing.assert_close(
    tensor_runtime._post_frames, tensor_runtime.command.time_steps
  )
  torch.testing.assert_close(tensor_runtime._pre_frames, array_runtime._pre_frames)
  torch.testing.assert_close(tensor_runtime._post_frames, array_runtime._post_frames)
  received = tensor_env.received[0]
  assert isinstance(received, torch.Tensor)
  assert received.dtype == torch.float32 and received.device.type == "cpu"
  torch.testing.assert_close(received, array_env.received[0])


def test_step_tensor_batch_accepts_noncontiguous_action() -> None:
  runtime, env = _step_runtime()
  noncontiguous = torch.arange(3 * 62, dtype=torch.float32).reshape(3, 62)[:, ::2]
  runtime.step_tensor_batch(noncontiguous)
  torch.testing.assert_close(env.received[0], noncontiguous)


def test_step_tensor_batch_refuses_invalid_actions() -> None:
  runtime, env = _step_runtime()
  with pytest.raises(RuntimeX2Error, match="torch.Tensor"):
    runtime.step_tensor_batch(cast(Any, np.zeros((3, 31))))
  with pytest.raises(RuntimeX2Error, match="shape"):
    runtime.step_tensor_batch(torch.zeros(2, 31))
  with pytest.raises(RuntimeX2Error, match="finite"):
    runtime.step_tensor_batch(torch.full((3, 31), float("nan")))
  assert runtime._last_result is None
  assert env.received == []


@pytest.mark.skipif(
  not _CUDA_AUTHORIZED,
  reason="CUDA tensor tests are authored but not executed in this lane",
)
def test_cuda_device_native_tensor_action_stays_on_gpu() -> None:
  if not torch.cuda.is_available():
    pytest.skip("CUDA is unavailable")
  device = torch.device("cuda:0")
  vae = FrozenVaePolicy(_TensorVae().to(device), _contract())
  observation = VaeObservationTensorBatch(
    torch.full((2, 68), 0.5, device=device),
    torch.full((2, 99), 0.25, device=device),
  )
  latent, clean = vae.batch_action_tensor(observation)
  assert latent.device == device and clean.device == device
  assert latent.dtype == torch.float32 and latent.shape == (2, 32)


@pytest.mark.skipif(
  not _CUDA_AUTHORIZED,
  reason="CUDA tensor tests are authored but not executed in this lane",
)
def test_cuda_observe_moves_reference_to_policy_device() -> None:
  if not torch.cuda.is_available():
    pytest.skip("CUDA is unavailable")
  device = torch.device("cuda:0")
  reference = torch.zeros(2, 68, dtype=torch.float32, device=device)
  conditioning = torch.zeros(2, 99, dtype=torch.float32, device=device)
  runtime = _tensor_runtime(
    2, reference, conditioning, policy_device="cuda:0", sim_device="cuda:0"
  )
  observation = runtime.observe_vae_tensor_batch(np.asarray([0, 1]))
  assert observation.device == device
  assert observation.batch_size == 2
