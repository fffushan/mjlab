"""CPU tests for frozen-policy action/frame adapters."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tracking_diffusion_fixtures import make_state

from mjlab.tasks.tracking.diffusion import (
  ActionContract,
  AdapterError,
  CallbackEnvironmentAdapter,
  FrozenPpoRecoveryPolicy,
  FrozenVaePolicy,
  PpoObservation,
  VaeObservation,
)


class _Schema:
  reference_dim = 68
  conditioning_dim = 99


class _FakeVae(torch.nn.Module):
  schema = _Schema()

  def encode(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ones((reference.shape[0], 32)), torch.zeros((reference.shape[0], 32))

  def decode(self, latent: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
    return torch.cat(
      (latent[:, :1].expand(-1, 31), conditioning[:, :1].expand(-1, 31)), dim=1
    )[:, :31]


def _contract(scale: float = 2.0, offset: float = 1.0) -> ActionContract:
  return ActionContract(
    np.full(31, scale), np.full(31, offset), np.ones(31), np.ones(31)
  )


def test_callback_adapter_delegates_reset_after_trial() -> None:
  calls: list[str] = []
  adapter = CallbackEnvironmentAdapter(
    reset_fn=lambda **_: None,
    vae_observe_fn=lambda: VaeObservation(
      make_state(), np.ones(68), np.ones(99), "motion", 0, 0.0
    ),
    step_fn=lambda _action: None,
    ppo_observe_fn=lambda: PpoObservation(make_state(), np.ones(102), "motion", 0, 0.0),
    switch_fn=lambda: None,
    evidence_fn=lambda result: result,
    trial_reset_fn=lambda: calls.append("reset_after_trial"),
  )
  adapter.reset_after_trial()
  assert calls == ["reset_after_trial"]


def test_vae_passes_mu_to_decoder_and_recovery_previous_action_converts() -> None:
  vae = FrozenVaePolicy(_FakeVae(), _contract())
  observation = VaeObservation(make_state(), np.ones(68), np.ones(99), "motion", 0, 0.0)
  latent, action = vae.action(observation)
  np.testing.assert_allclose(latent, 1.0)
  np.testing.assert_allclose(action, 1.0)

  policy = FrozenPpoRecoveryPolicy(
    lambda observation: np.full((1, 31), 0.25), _contract(scale=4.0, offset=-2.0)
  )
  policy.previous_from_physical(vae.action_contract.to_physical(np.ones(31)))
  np.testing.assert_allclose(policy.previous_action, 1.25)
  ppo_observation = PpoObservation(make_state(), np.zeros(102), "motion", 0, 0.0)
  np.testing.assert_allclose(policy.action(ppo_observation), 0.25)


def test_recovery_inference_receives_converted_endpoint_action() -> None:
  seen: list[np.ndarray] = []
  policy = FrozenPpoRecoveryPolicy(
    lambda observation: seen.append(observation.copy()) or np.zeros((1, 31)),
    _contract(scale=4.0, offset=-2.0),
  )
  endpoint = np.full(31, 2.0)
  policy.handoff_from_physical(endpoint)
  observed = np.zeros(102)
  observed[68:99] = -3.0
  observation = PpoObservation(make_state(), observed, "motion", 0, 0.0)
  policy.action(observation)
  assert seen
  np.testing.assert_allclose(seen[0][0, 68:99], 1.0)
  np.testing.assert_allclose(seen[0][0, 99:102], observed[99:102])


def test_action_metadata_mismatch_is_rejected() -> None:
  contract = _contract()
  with pytest.raises(AdapterError, match="scale"):
    contract.verify_metadata(
      {
        "joint_names": list(contract.joint_order),
        "action_scale": ",".join(["3.0"] * 31),
        "default_joint_pos": ",".join(["1.0"] * 31),
        "joint_stiffness": ",".join(["1.0"] * 31),
        "joint_damping": ",".join(["1.0"] * 31),
      }
    )

  with pytest.raises(AdapterError):
    VaeObservation(make_state(), np.ones(68), np.ones(99), "motion", 0, np.inf)


def test_real_pinned_checkpoint_and_recovery_metadata_bind_on_cpu() -> None:
  import onnx

  recovery_path = (
    "logs/rsl_rl/agibot_x2_velocity/2026-09-26_02-40-45_x2-tennis-recovery-n25/"
    "2026-09-26_02-40-45_x2-tennis-recovery-n25.onnx"
  )
  metadata = {
    entry.key: entry.value for entry in onnx.load(recovery_path).metadata_props
  }
  contract = ActionContract.from_metadata(metadata)
  vae = FrozenVaePolicy.from_checkpoint(
    "logs/distillation/mixed-10k/checkpoint-final.pt",
    contract,
    teacher_id="tennis_000",
  )
  recovery = FrozenPpoRecoveryPolicy.from_onnx(recovery_path, contract)
  assert vae.artifact_sha256 == (
    "69891dffb59af31539388e40041efe30242a2028f2876b746d1f7c5ef44ac117"
  )
  assert recovery.artifact_sha256 == (
    "caf17c38f23ad180829a230a9a3259f14eedd3860dd90a046b8be92d3d047681"
  )
  assert vae.action_contract.joint_order == contract.joint_order
