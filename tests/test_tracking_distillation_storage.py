"""CPU tests for bounded, raw labeled replay."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from mjlab.tasks.tracking.distillation.observations import PackedObservationBatch
from mjlab.tasks.tracking.distillation.storage import (
  LabeledReplayBatch,
  LabeledReplayBuffer,
  ReplayValidationError,
)
from mjlab.tasks.tracking.distillation.vae_config import DEFAULT_SCHEMA, schema_for_mode


def make_batch(
  start: int = 0,
  size: int = 1,
  *,
  requires_grad: bool = False,
  dtype: torch.dtype = torch.float32,
) -> LabeledReplayBatch:
  rows = torch.arange(start, start + size, dtype=dtype).reshape(-1, 1)
  reference = rows * 100 + torch.arange(68, dtype=dtype)
  conditioning = rows * 1000 + torch.arange(99, dtype=dtype)
  action = rows * 10 + torch.arange(31, dtype=dtype)
  if requires_grad:
    reference.requires_grad_()
    conditioning.requires_grad_()
    action.requires_grad_()
  observations = PackedObservationBatch(reference, conditioning, DEFAULT_SCHEMA)
  ids = torch.arange(start, start + size, dtype=torch.int64)
  return LabeledReplayBatch(
    observations=observations,
    teacher_action=action,
    motion_id=ids + 10,
    teacher_id=ids + 20,
    reference_frame=ids + 30,
    episode_id=ids + 40,
    collector_iteration=ids + 50,
  )


def ids_in(batch: LabeledReplayBatch) -> list[int]:
  return batch.motion_id.tolist()


def test_partial_wraparound_and_oversized_insert_retain_fifo_tail() -> None:
  buffer = LabeledReplayBuffer(capacity=3, schema=DEFAULT_SCHEMA)
  buffer.insert(make_batch(0, 2))
  buffer.insert(make_batch(2, 2))

  assert buffer.size == 3
  wrapped = buffer.sample(
    3, replacement=False, generator=torch.Generator().manual_seed(1)
  )
  assert sorted(value - 10 for value in ids_in(wrapped)) == [1, 2, 3]
  assert wrapped.teacher_id.tolist() == [
    value + 10 for value in wrapped.motion_id.tolist()
  ]

  buffer.insert(make_batch(4, 5))
  assert buffer.size == 3
  oversized = buffer.sample(
    3, replacement=False, generator=torch.Generator().manual_seed(2)
  )
  assert sorted(value - 10 for value in ids_in(oversized)) == [6, 7, 8]


def test_empty_insert_is_validated_noop_and_empty_sampling_is_explicit() -> None:
  buffer = LabeledReplayBuffer(capacity=2, schema=DEFAULT_SCHEMA)
  buffer.insert(make_batch(size=0, dtype=torch.float64))
  assert buffer.device is None
  assert buffer.dtype is None
  assert buffer.size == 0
  with pytest.raises(ReplayValidationError, match="empty replay"):
    buffer.sample(1)

  buffer.insert(make_batch(0, 1, dtype=torch.float32))
  assert buffer.size == 1

  explicit = LabeledReplayBuffer(
    capacity=2, schema=DEFAULT_SCHEMA, device="cpu:0", dtype=torch.float64
  )
  explicit.insert(make_batch(size=0, dtype=torch.float64))
  assert explicit.device == torch.device("cpu")
  assert explicit.dtype is torch.float64


def test_cpu_device_alias_accepts_cpu_tensor_and_canonicalizes_policy() -> None:
  buffer = LabeledReplayBuffer(capacity=2, schema=DEFAULT_SCHEMA, device="cpu:0")
  buffer.insert(make_batch(0, 1))
  assert buffer.device == torch.device("cpu")


def test_insert_and_sampling_own_detached_nonaliased_copies() -> None:
  source = make_batch(0, 2, requires_grad=True)
  buffer = LabeledReplayBuffer(capacity=3, schema=DEFAULT_SCHEMA)
  buffer.insert(source)
  with torch.no_grad():
    source.observations.reference.fill_(999)
    source.observations.conditioning.fill_(999)
    source.teacher_action.fill_(999)
    source.motion_id.fill_(999)

  sampled = buffer.sample(
    2, replacement=False, generator=torch.Generator().manual_seed(3)
  )
  assert sorted(value - 10 for value in ids_in(sampled)) == [0, 1]
  assert sampled.reference.requires_grad is False
  assert sampled.conditioning.requires_grad is False
  assert sampled.teacher_action.requires_grad is False

  sampled.reference.fill_(1234)
  sampled.motion_id.fill_(1234)
  again = buffer.sample(
    2, replacement=False, generator=torch.Generator().manual_seed(3)
  )
  assert sorted(value - 10 for value in ids_in(again)) == [0, 1]
  assert not torch.equal(again.reference, sampled.reference)


def test_reused_caller_buffers_are_copied_per_insert_and_metadata_stays_aligned() -> (
  None
):
  buffer = LabeledReplayBuffer(capacity=4, schema=DEFAULT_SCHEMA)
  first = make_batch(0, 2)
  buffer.insert(first)
  second = make_batch(2, 2)
  buffer.insert(second)
  second.motion_id.fill_(777)

  sampled = buffer.sample(
    4, replacement=False, generator=torch.Generator().manual_seed(4)
  )
  records = sorted(
    zip(
      sampled.motion_id.tolist(),
      sampled.teacher_id.tolist(),
      sampled.reference_frame.tolist(),
      sampled.episode_id.tolist(),
      sampled.collector_iteration.tolist(),
      strict=True,
    )
  )
  assert records == [(10 + i, 20 + i, 30 + i, 40 + i, 50 + i) for i in range(4)]


def test_seeded_sampling_is_reproducible_and_replacement_is_explicit() -> None:
  buffer = LabeledReplayBuffer(capacity=5, schema=DEFAULT_SCHEMA)
  buffer.insert(make_batch(0, 5))
  first = buffer.sample(12, generator=torch.Generator().manual_seed(9))
  second = buffer.sample(12, generator=torch.Generator().manual_seed(9))
  torch.testing.assert_close(first.motion_id, second.motion_id)
  assert len(set(first.motion_id.tolist())) < 12

  distinct = buffer.sample(
    5, replacement=False, generator=torch.Generator().manual_seed(9)
  )
  assert len(set(distinct.motion_id.tolist())) == 5
  with pytest.raises(ReplayValidationError, match="cannot exceed"):
    buffer.sample(6, replacement=False)


def test_complete_batch_validation_rejects_without_mutating_existing_storage() -> None:
  buffer = LabeledReplayBuffer(capacity=3, schema=DEFAULT_SCHEMA)
  buffer.insert(make_batch(0, 2))
  before = buffer.sample(
    2, replacement=False, generator=torch.Generator().manual_seed(5)
  )

  invalid = make_batch(2, 1)
  invalid.teacher_action[0, 0] = float("nan")
  with pytest.raises(ReplayValidationError, match="non-finite"):
    buffer.insert(invalid)
  after = buffer.sample(
    2, replacement=False, generator=torch.Generator().manual_seed(5)
  )
  torch.testing.assert_close(before.reference, after.reference)
  torch.testing.assert_close(before.motion_id, after.motion_id)

  bad_shape = make_batch(2, 1)
  bad_shape = replace(
    bad_shape,
    observations=PackedObservationBatch(
      bad_shape.reference, torch.empty(1, 98), DEFAULT_SCHEMA
    ),
  )
  with pytest.raises(ReplayValidationError, match="shape"):
    buffer.insert(bad_shape)


def test_shape_schema_device_dtype_and_metadata_rejections() -> None:
  buffer = LabeledReplayBuffer(capacity=2, schema=DEFAULT_SCHEMA)
  with pytest.raises(ReplayValidationError, match="shape"):
    buffer.insert(
      LabeledReplayBatch(
        observations=PackedObservationBatch(
          torch.zeros(1, 67), torch.zeros(1, 99), DEFAULT_SCHEMA
        ),
        teacher_action=torch.zeros(1, 31),
        motion_id=torch.zeros(1, dtype=torch.int64),
        teacher_id=torch.zeros(1, dtype=torch.int64),
        reference_frame=torch.zeros(1, dtype=torch.int64),
        episode_id=torch.zeros(1, dtype=torch.int64),
        collector_iteration=torch.zeros(1, dtype=torch.int64),
      )
    )
  wrong_schema = make_batch(0, 1)
  wrong_schema = replace(
    wrong_schema,
    observations=PackedObservationBatch(
      torch.zeros(1, 68), torch.zeros(1, 102), schema_for_mode("anchor")
    ),
  )
  with pytest.raises(ReplayValidationError, match="schema"):
    buffer.insert(wrong_schema)
  wrong_metadata = replace(
    make_batch(0, 1), motion_id=torch.zeros(1, dtype=torch.int32)
  )
  with pytest.raises(ReplayValidationError, match="int64"):
    buffer.insert(wrong_metadata)


def test_constructor_requires_positive_capacity_and_floating_dtype() -> None:
  with pytest.raises(ReplayValidationError, match="positive"):
    LabeledReplayBuffer(capacity=0, schema=DEFAULT_SCHEMA)
  with pytest.raises(ReplayValidationError, match="floating"):
    LabeledReplayBuffer(capacity=2, schema=DEFAULT_SCHEMA, dtype=torch.int64)
