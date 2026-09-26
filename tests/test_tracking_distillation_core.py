"""Integration checks for the pure M2 packing, replay, and VAE core."""

from __future__ import annotations

import torch
from tracking_distillation_fixtures import make_actor

from mjlab.tasks.tracking.distillation import (
  DEFAULT_SCHEMA,
  ConditionalVAE,
  DecoderMode,
  LabeledReplayBatch,
  LabeledReplayBuffer,
  ModelSettings,
  ObservationSnapshot,
  PackedObservationBatch,
  make_schema,
  pack_observations,
  vae_loss,
)
from mjlab.tasks.tracking.distillation.config import ActorArchitecture
from mjlab.tasks.tracking.distillation.teachers import (
  TeacherBank,
  build_frozen_teacher,
)


def make_snapshot(batch_size: int = 4) -> ObservationSnapshot:
  generator = torch.Generator().manual_seed(19)

  def values(width: int) -> torch.Tensor:
    return torch.randn(batch_size, width, generator=generator)

  return ObservationSnapshot(
    reference_q=values(31),
    reference_dq=values(31),
    anchor_orientation_error=values(6),
    projected_gravity=values(3),
    gyro=values(3),
    relative_joint_q=values(31),
    joint_dq=values(31),
    previous_action=values(31),
  )


def make_replay_batch(
  packed: PackedObservationBatch, teacher_action: torch.Tensor
) -> LabeledReplayBatch:
  batch_size = packed.batch_size
  metadata = torch.arange(batch_size, dtype=torch.int64)
  return LabeledReplayBatch(
    observations=packed,
    teacher_action=teacher_action,
    motion_id=metadata,
    teacher_id=metadata % 2,
    reference_frame=metadata + 10,
    episode_id=metadata + 20,
    collector_iteration=metadata + 30,
  )


def test_gravity_default_and_opt_in_schema_identities() -> None:
  snapshot = make_snapshot(2)
  assert DEFAULT_SCHEMA.mode is DecoderMode.GRAVITY
  assert (DEFAULT_SCHEMA.reference_dim, DEFAULT_SCHEMA.conditioning_dim) == (68, 99)

  expected = {
    DecoderMode.GRAVITY: (99, False),
    DecoderMode.ANCHOR: (102, True),
    DecoderMode.GRAVITY_ANCHOR: (105, True),
  }
  for mode, (conditioning_dim, reference_conditioned) in expected.items():
    schema = make_schema(mode, joint_order=tuple(f"x2_{i}" for i in range(31)))
    packed = pack_observations(snapshot, schema)
    assert packed.reference.shape == (2, 68)
    assert packed.conditioning.shape == (2, conditioning_dim)
    assert schema.decoder_reference_conditioned is reference_conditioned
    assert schema.compatibility_metadata()["mode"] == mode.value
    assert schema.compatibility_metadata()["joint_order"][0] == "x2_0"


def test_raw_replay_to_fixed_teacher_labels_to_tiny_vae_step() -> None:
  torch.manual_seed(23)
  packed = pack_observations(make_snapshot(), DEFAULT_SCHEMA)
  teacher_model = make_actor(24, 31, seed=41)
  teacher_architecture = ActorArchitecture(
    class_name="MLPModel",
    hidden_dims=(8, 4),
    activation="elu",
    obs_normalization=True,
    obs_groups=("actor",),
    distribution_class_name="GaussianDistribution",
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "std_type": "scalar",
      "init_std": 1.0,
    },
    obs_dim=24,
    action_dim=31,
  )
  teacher_bank = TeacherBank(
    [build_frozen_teacher("tiny", teacher_model.state_dict(), teacher_architecture)],
    device="cpu",
  )
  teacher_input = torch.randn(
    packed.batch_size, 24, generator=torch.Generator().manual_seed(7)
  )
  teacher_action = teacher_bank.label(
    torch.zeros(packed.batch_size, dtype=torch.int64), teacher_input
  )
  assert not teacher_action.requires_grad

  replay = LabeledReplayBuffer(capacity=8, schema=DEFAULT_SCHEMA, device="cpu")
  replay.insert(make_replay_batch(packed, teacher_action))
  sampled = replay.sample(
    packed.batch_size,
    replacement=False,
    generator=torch.Generator().manual_seed(5),
  )

  model = ConditionalVAE(
    schema=DEFAULT_SCHEMA,
    settings=ModelSettings(hidden_dims=(16, 8)),
  )
  # Normalization is deliberately explicit and consumes raw replay tensors.
  model.reference_normalizer.update(sampled.reference)
  model.conditioning_normalizer.update(sampled.conditioning)
  before = model.action_head.weight.detach().clone()
  optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

  output = model(
    sampled.reference,
    sampled.conditioning,
    sample=True,
    noise=torch.zeros(sampled.batch_size, 32),
  )
  loss = vae_loss(output.action, sampled.teacher_action, output.mu, output.logvar)
  assert torch.isfinite(loss.total)
  loss.total.backward()
  optimizer.step()

  assert model.action_head.weight.grad is not None
  assert not torch.equal(before, model.action_head.weight.detach())
  assert output.action.shape == (packed.batch_size, 31)
  assert loss.reconstruction.ndim == 0
  assert loss.kl.ndim == 0
