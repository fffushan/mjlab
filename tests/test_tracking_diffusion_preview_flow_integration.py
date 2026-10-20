"""CPU-only seam test from a preview or legacy reference to the flow token window.

The collection lane derives the live VAE reference from the cohort and the
startup lane assembles the reference in saved schema order.  This test joins
those seams to one small real ``ConditionalVAE``: the assembled 68-D or 130-D
reference must reach the encoder, the clean decoded action and encoder mean must
be the values the collection wrapper reports, and that latent must enter the
unchanged 41x167 flow token window without a width or row change.

No checkpoint, simulator, training run or flow-family source change is used.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from tracking_diffusion_fixtures import make_state

from mjlab.tasks.tracking.diffusion.adapter import ActionContract, FrozenVaePolicy
from mjlab.tasks.tracking.diffusion.closed_loop_runtime_x2 import _assemble_reference
from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT
from mjlab.tasks.tracking.diffusion.handover import (
  HandoverHistory,
  build_reference_free_plan,
)
from mjlab.tasks.tracking.diffusion.projection import (
  FeatureStats,
  ProjectionBundle,
  fit_feature_stats,
)
from mjlab.tasks.tracking.diffusion.sampler import SamplerConditions
from mjlab.tasks.tracking.distillation.adapter import default_schema_for_cohort
from mjlab.tasks.tracking.distillation.model import ConditionalVAE
from mjlab.tasks.tracking.distillation.vae_config import (
  PREVIEW_DIM,
  REFERENCE_DIM,
  REFERENCE_DIM_WITH_PREVIEW,
  REFERENCE_PREVIEW_FIELD,
  ModelSettings,
)
from mjlab.tasks.tracking.flow_matching.closed_loop import FlowSamplingAdapter
from mjlab.tasks.tracking.flow_matching.process import LinearNoiseProcess
from mjlab.tasks.tracking.flow_matching.sampler import FlowGrid, FlowSamplerConfig


def _stats(width: int, *, seed: int) -> FeatureStats:
  rng = np.random.default_rng(seed)
  return fit_feature_stats([rng.normal(size=(64, width)) + 3.0], std_floor=1e-6)


def _projection() -> ProjectionBundle:
  return ProjectionBundle.create(_stats(135, seed=31), _stats(32, seed=32))


class _IdentityFlowDenoiser:
  """Tiny flow network stand-in that preserves the sampled token shape."""

  def __call__(self, sample: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
    assert step_ids.shape == (*sample.shape[:-1], 2)
    return sample


def _cohort(lookahead_s: float, joint_names: tuple[str, ...]) -> Any:
  return SimpleNamespace(
    lookahead_s=lookahead_s,
    actions=SimpleNamespace(joint_names=joint_names, dim=len(joint_names)),
  )


def _reference_values(
  *, preview: bool, joint_count: int
) -> tuple[dict[str, np.ndarray], np.ndarray]:
  command = np.linspace(-1.7, 1.7, 2 * joint_count, dtype=np.float64)
  anchor = np.linspace(0.1, 0.6, 6, dtype=np.float64)
  values = {"command": command, "motion_anchor_ori_b": anchor}
  if preview:
    values[REFERENCE_PREVIEW_FIELD] = np.linspace(
      -0.9, 0.9, PREVIEW_DIM, dtype=np.float64
    )
  return values, np.full(99, 0.25, dtype=np.float64)


@pytest.mark.parametrize(
  ("lookahead_s", "reference_dim", "preview"),
  [
    (0.0, REFERENCE_DIM, False),
    (0.2, REFERENCE_DIM_WITH_PREVIEW, True),
  ],
)
def test_reference_to_vae_latent_keeps_the_unchanged_flow_token_window(
  lookahead_s: float,
  reference_dim: int,
  preview: bool,
) -> None:
  contract = ActionContract(np.ones(31), np.zeros(31), np.ones(31), np.ones(31))
  schema = default_schema_for_cohort(_cohort(lookahead_s, contract.joint_order))
  assert (
    schema.reference_dim,
    schema.conditioning_dim,
    schema.latent_dim,
    schema.action_dim,
  ) == (reference_dim, 99, 32, 31)

  # The assembled reference keeps the saved schema order: q, dq, anchor
  # orientation, then the one 62-wide future sample when the cohort declares it.
  values, conditioning = _reference_values(
    preview=preview, joint_count=len(contract.joint_order)
  )
  reference = _assemble_reference(values, preview=preview)
  assert reference.shape == (reference_dim,)
  if preview:
    np.testing.assert_array_equal(reference[68:], values[REFERENCE_PREVIEW_FIELD])

  model = ConditionalVAE(schema, ModelSettings(hidden_dims=(8,)))
  policy = FrozenVaePolicy(model, contract)
  assert policy.reference_width == reference_dim
  latent, action = policy.latent_and_action(reference, conditioning)
  assert latent.shape == (32,)
  assert action.shape == (31,)

  # The collection wrapper must report the deterministic encoder mean and the
  # action decoded from exactly that latent, not a sample or a re-encode.
  with torch.no_grad():
    mu, _ = model.encode(torch.as_tensor(reference, dtype=torch.float32)[None])
    decoded = model.decode(mu, torch.as_tensor(conditioning, dtype=torch.float32)[None])
  np.testing.assert_allclose(latent, mu[0].numpy(), rtol=0.0, atol=1e-6)
  np.testing.assert_allclose(action, decoded[0].numpy(), rtol=0.0, atol=1e-6)

  projection = _projection()
  history = HandoverHistory()
  history_latents = np.stack([latent + 0.001 * index for index in range(8)])
  for index in range(DEFAULT_CONTRACT.past_steps):
    history.push(make_state(index), history_latents[index])
  plan = build_reference_free_plan(
    history, make_state(8), projection=projection, tick=0
  )

  state_width = DEFAULT_CONTRACT.projected_state_dimension
  token_width = DEFAULT_CONTRACT.token_dimension
  assert (state_width, token_width) == (135, 167)
  assert plan.clean_tokens.shape == (41, token_width)
  assert plan.clean_mask.shape == (41, token_width)
  assert bool(plan.clean_mask[:9, :state_width].all())
  assert not bool(plan.clean_mask[8, state_width:].any())
  assert torch.count_nonzero(plan.clean_tokens[8, state_width:]) == 0
  np.testing.assert_allclose(
    plan.clean_tokens[:8, state_width:].numpy(),
    projection.normalize_latent(history_latents),
    rtol=1e-6,
    atol=1e-6,
  )

  # The unchanged flow sampler consumes the same window: 41 rows and 167
  # channels, with clean state and history-latent entries restored exactly.
  process = LinearNoiseProcess(1000)
  adapter = FlowSamplingAdapter(
    denoiser=_IdentityFlowDenoiser(),
    config=FlowSamplerConfig(
      grid=FlowGrid.uniform(calls=2, max_level=process.max_level),
      process=process,
    ),
    state_dimension=state_width,
  )
  sample, diagnostics = adapter(
    SamplerConditions(plan.clean_tokens, plan.clean_mask),
    torch.Generator().manual_seed(7),
  )
  assert sample.shape == (41, 167)
  assert diagnostics.denoiser_calls == 2
  assert torch.equal(sample[:9, :state_width], plan.clean_tokens[:9, :state_width])
  assert torch.equal(sample[:8, state_width:], plan.clean_tokens[:8, state_width:])
