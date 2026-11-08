"""CPU tests for the balanced per-motion replay buffer.

The batch factory encodes each row's identity in ``reference[:, 0]`` as
``motion_id * 1000 + row`` so retained ordering, eviction, and drawn mixtures
can be asserted exactly instead of only by shape.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import pytest
import torch

from mjlab.tasks.tracking.distillation.balanced_storage import (
  BalancedReplayBuffer,
  MotionCapacityQuota,
  ReplayNotReadyError,
  allocate_motion_capacity_quotas,
)
from mjlab.tasks.tracking.distillation.model import ConditionalVAE
from mjlab.tasks.tracking.distillation.observations import PackedObservationBatch
from mjlab.tasks.tracking.distillation.storage import (
  LabeledReplayBatch,
  LabeledReplayBuffer,
  ReplayBufferProtocol,
  ReplayValidationError,
)
from mjlab.tasks.tracking.distillation.trainer import (
  FreshTrainingData,
  TrainerValidationError,
  VaeDistillationTrainer,
)
from mjlab.tasks.tracking.distillation.training_config import TrainingConfig
from mjlab.tasks.tracking.distillation.vae_config import DEFAULT_SCHEMA, ModelSettings

FRAMES = {0: 12, 1: 12}
CODES = {0: 0, 1: 1}


def make_motion_batch(
  motion_ids: Sequence[int],
  *,
  start: int = 0,
  frames: Sequence[int] | None = None,
  codes: Sequence[int] | None = None,
  iterations: Sequence[int] | None = None,
  dtype: torch.dtype = torch.float32,
) -> LabeledReplayBatch:
  """One batch whose reference/conditioning column 0 tags ``motion*1000 + row``."""
  size = len(motion_ids)
  tags = torch.tensor(
    [motion * 1000 + start + index for index, motion in enumerate(motion_ids)],
    dtype=dtype,
  )
  reference = torch.zeros(size, DEFAULT_SCHEMA.reference_dim, dtype=dtype)
  reference[:, 0] = tags
  reference[:, 1:] = torch.arange(
    DEFAULT_SCHEMA.reference_dim - 1, dtype=dtype
  ).reshape(1, -1)
  conditioning = torch.zeros(size, DEFAULT_SCHEMA.conditioning_dim, dtype=dtype)
  conditioning[:, 0] = tags
  action = tags.reshape(-1, 1) * 0.001
  action = action.expand(-1, DEFAULT_SCHEMA.action_dim).contiguous()
  ids = torch.arange(start, start + size, dtype=torch.int64)
  return LabeledReplayBatch(
    observations=PackedObservationBatch(reference, conditioning, DEFAULT_SCHEMA),
    teacher_action=action,
    motion_id=torch.tensor(list(motion_ids), dtype=torch.int64),
    teacher_id=torch.tensor(
      list(codes) if codes is not None else list(motion_ids), dtype=torch.int64
    ),
    reference_frame=torch.tensor(
      list(frames) if frames is not None else [0] * size, dtype=torch.int64
    ),
    episode_id=ids,
    collector_iteration=torch.tensor(
      list(iterations) if iterations is not None else [1] * size, dtype=torch.int64
    ),
  )


def tags_of(batch: LabeledReplayBatch) -> list[float]:
  return [float(value) for value in batch.reference[:, 0].tolist()]


def make_buffer(
  capacity: int = 8,
  *,
  weights: Mapping[int, float] | None = None,
  teacher_codes: Mapping[int, int] | None = None,
  frame_counts: Mapping[int, int] | None = None,
  device: torch.device | str | None = None,
  dtype: torch.dtype | None = None,
) -> BalancedReplayBuffer:
  return BalancedReplayBuffer(
    capacity,
    DEFAULT_SCHEMA,
    weights if weights is not None else {0: 1.0, 1: 1.0},
    teacher_codes=CODES if teacher_codes is None else teacher_codes,
    frame_counts=FRAMES if frame_counts is None else frame_counts,
    device=device,
    dtype=dtype,
  )


def retained_tags(buffer: BalancedReplayBuffer, motion_id: int) -> list[float]:
  """Tags retained by one motion, read through the public persistence seam."""
  state = buffer.state_dict()
  position = list(state["motion_ids"]).index(motion_id)
  partition = state["partitions"][position]
  storage = state["fields"]
  if storage is None or partition["size"] == 0:
    return []
  size, cursor, quota = partition["size"], partition["next"], partition["quota"]
  start = (cursor - size) % quota
  physical = [(index + start) % quota for index in range(size)]
  base = sum(state["quotas"][:position])
  reference = storage["reference"]
  return [float(reference[base + row, 0]) for row in physical]


def assert_state_equal(left: object, right: object) -> None:
  if isinstance(left, torch.Tensor):
    assert isinstance(right, torch.Tensor)
    assert left.shape == right.shape
    torch.testing.assert_close(left, right)
    return
  if isinstance(left, Mapping):
    assert isinstance(right, Mapping)
    left_fields: dict[str, Any] = {str(name): value for name, value in left.items()}
    right_fields: dict[str, Any] = {str(name): value for name, value in right.items()}
    assert set(left_fields) == set(right_fields)
    for key, value in left_fields.items():
      assert_state_equal(value, right_fields[key])
    return
  if isinstance(left, (list, tuple)):
    assert isinstance(right, (list, tuple))
    assert len(left) == len(right)
    for one, other in zip(left, right, strict=True):
      assert_state_equal(one, other)
    return
  assert left == right


def invalid(value: object) -> Any:
  """Pass a deliberately wrong runtime value through the static type check."""
  return value


def test_quota_allocation_is_deterministic_and_fills_capacity() -> None:
  quotas = allocate_motion_capacity_quotas({0: 1.0, 1: 1.0}, 5)
  assert isinstance(quotas, MotionCapacityQuota)
  assert quotas.motion_ids == (0, 1)
  assert quotas.quotas == (3, 2)
  assert sum(quotas.quotas) == 5
  assert quotas.fractions == pytest.approx((0.6, 0.4))
  assert quotas.quota_for(1) == 2
  assert quotas.as_dict()["total_capacity"] == 5
  # Equal weights break ties toward the lower motion id, so repeated calls and
  # a differently ordered mapping agree.
  assert allocate_motion_capacity_quotas({1: 1.0, 0: 1.0}, 5).quotas == (3, 2)
  assert allocate_motion_capacity_quotas({0: 1.0, 1: 1.0}, 2).quotas == (1, 1)
  assert allocate_motion_capacity_quotas({0: 1.0, 1: 2.0, 2: 3.0}, 5).quotas == (
    1,
    2,
    2,
  )
  assert allocate_motion_capacity_quotas({0: 1.0, 1: 1.0, 2: 1.0}, 4).quotas == (
    2,
    1,
    1,
  )
  weighted = allocate_motion_capacity_quotas({0: 3.0, 1: 1.0}, 10)
  assert weighted.quotas == (7, 3)
  assert all(quota >= 1 for quota in weighted.quotas)


def test_quota_allocation_rejects_impossible_or_invalid_requests() -> None:
  with pytest.raises(ReplayValidationError, match="cannot give each"):
    allocate_motion_capacity_quotas({0: 1.0, 1: 1.0}, 1)
  with pytest.raises(ReplayValidationError, match="non-empty mapping"):
    allocate_motion_capacity_quotas({}, 4)
  for weight in (0.0, -1.0, float("nan"), float("inf")):
    with pytest.raises(ReplayValidationError, match="finite and positive"):
      allocate_motion_capacity_quotas({0: weight, 1: 1.0}, 4)
  with pytest.raises(ReplayValidationError, match="capacity must be"):
    allocate_motion_capacity_quotas({0: 1.0}, 0)
  with pytest.raises(ReplayValidationError, match="capacity must be an integer"):
    allocate_motion_capacity_quotas({0: 1.0}, True)
  with pytest.raises(ReplayValidationError, match="non-negative integers"):
    allocate_motion_capacity_quotas({-1: 1.0}, 2)
  with pytest.raises(ReplayValidationError, match="cannot give each"):
    make_buffer(capacity=1)


def test_independent_fifo_eviction_under_unequal_insert_rates() -> None:
  buffer = make_buffer(capacity=6)
  assert buffer.quota.quotas == (3, 3)
  buffer.insert(make_motion_batch([0, 0], start=0))
  buffer.insert(make_motion_batch([1], start=0))
  assert retained_tags(buffer, 0) == [0.0, 1.0]
  assert retained_tags(buffer, 1) == [1000.0]
  assert buffer.size == 3

  # Motion 0 evicts only its own oldest records.
  buffer.insert(make_motion_batch([0, 0, 0, 0], start=2))
  assert retained_tags(buffer, 0) == [3.0, 4.0, 5.0]
  assert retained_tags(buffer, 1) == [1000.0]

  # Motion 1 keeps its own FIFO order too.
  buffer.insert(make_motion_batch([1, 1, 1], start=1))
  assert retained_tags(buffer, 1) == [1001.0, 1002.0, 1003.0]
  assert retained_tags(buffer, 0) == [3.0, 4.0, 5.0]
  report = buffer.report()
  assert report.retained == 6
  assert [motion.retained for motion in report.motions] == [3, 3]
  assert report.ready


def test_oversized_insert_keeps_only_its_own_latest_rows() -> None:
  buffer = make_buffer(capacity=4)
  assert buffer.quota.quotas == (2, 2)
  buffer.insert(make_motion_batch([1, 1, 1], start=0))
  assert retained_tags(buffer, 1) == [1001.0, 1002.0]

  buffer.insert(make_motion_batch([0] * 5, start=10))
  assert retained_tags(buffer, 0) == [13.0, 14.0]
  assert retained_tags(buffer, 1) == [1001.0, 1002.0]
  assert buffer.size == 4

  # An insertion exactly at quota replaces the partition with those rows.
  buffer.insert(make_motion_batch([1, 1], start=5))
  assert retained_tags(buffer, 1) == [1005.0, 1006.0]


def test_invalid_batch_leaves_every_partition_untouched() -> None:
  buffer = make_buffer(capacity=6)
  buffer.insert(make_motion_batch([0, 0, 1, 1], start=0))
  before = buffer.state_dict()
  invalid_batches = [
    # A bad row after valid rows must not commit the valid prefix.
    make_motion_batch([0, 0, 1, 7], start=0),
    make_motion_batch([0, 0, 1, 1], start=0, codes=[0, 0, 1, 0]),
    make_motion_batch([0, 0, 1, 1], start=0, frames=[0, 0, 0, FRAMES[1]]),
    make_motion_batch([0, 0, 1, -1], start=0),
    make_motion_batch([0, 0, 1, 1], start=0, dtype=torch.float64),
    LabeledReplayBatch(
      observations=PackedObservationBatch(
        make_motion_batch([0]).reference,
        make_motion_batch([0]).conditioning * 2,
        DEFAULT_SCHEMA,
      ),
      teacher_action=make_motion_batch([0]).teacher_action,
      motion_id=torch.tensor([0, 0, 1, 1], dtype=torch.int64),
      teacher_id=torch.tensor([0, 0, 1, 1], dtype=torch.int64),
      reference_frame=torch.zeros(4, dtype=torch.int64),
      episode_id=torch.zeros(4, dtype=torch.int64),
      collector_iteration=torch.zeros(4, dtype=torch.int64),
    ),
  ]
  for invalid in invalid_batches:
    with pytest.raises(ReplayValidationError):
      buffer.insert(invalid)
    assert_state_equal(buffer.state_dict(), before)
  generator = torch.Generator().manual_seed(11)
  expected = tags_of(buffer.sample(4, generator=generator))
  assert expected == tags_of(
    buffer.sample(4, generator=torch.Generator().manual_seed(11))
  )


def test_tiny_capacity_partitions_and_empty_insert_contract() -> None:
  buffer = make_buffer(capacity=2)
  assert buffer.quota.quotas == (1, 1)
  buffer.insert(make_motion_batch([0], start=0))
  buffer.insert(make_motion_batch([1], start=0))
  assert buffer.size == 2
  drawn = buffer.sample(6, generator=torch.Generator().manual_seed(1))
  assert set(drawn.motion_id.tolist()) == {0, 1}
  buffer.insert(make_motion_batch([0], start=5))
  assert retained_tags(buffer, 0) == [5.0]

  empty = make_buffer(capacity=4)
  empty.insert(make_motion_batch([], dtype=torch.float64))
  assert empty.size == 0 and empty.device is None and empty.dtype is None
  assert not empty.ready
  with pytest.raises(ReplayValidationError, match="empty replay buffer"):
    empty.sample(1)
  bad_empty = make_motion_batch([], dtype=torch.float64)
  with pytest.raises(ReplayValidationError, match="reference must have shape"):
    empty.insert(
      LabeledReplayBatch(
        observations=PackedObservationBatch(
          bad_empty.reference[:, :-1], bad_empty.conditioning, DEFAULT_SCHEMA
        ),
        teacher_action=bad_empty.teacher_action,
        motion_id=bad_empty.motion_id,
        teacher_id=bad_empty.teacher_id,
        reference_frame=bad_empty.reference_frame,
        episode_id=bad_empty.episode_id,
        collector_iteration=bad_empty.collector_iteration,
      )
    )


def test_readiness_fails_closed_until_every_motion_has_records() -> None:
  buffer = make_buffer(capacity=8)
  buffer.insert(make_motion_batch([1, 1, 1], start=0))
  assert not buffer.ready
  assert buffer.unready_motion_ids == (0,)
  assert buffer.report().unready_motion_ids == (0,)
  with pytest.raises(ReplayNotReadyError, match=r"\[0\]"):
    buffer.sample(1, generator=torch.Generator().manual_seed(2))
  with pytest.raises(ReplayNotReadyError):
    buffer.sample(3, replacement=False)

  # The initial stratified slot policy populates every motion in one tick.
  buffer.insert(make_motion_batch([0, 1], start=0))
  assert buffer.ready
  assert buffer.unready_motion_ids == ()
  assert set(
    buffer.sample(4, generator=torch.Generator().manual_seed(3)).motion_id.tolist()
  ) == {
    0,
    1,
  }


def test_bad_mapping_or_frame_configuration_is_rejected() -> None:
  with pytest.raises(ReplayValidationError, match="do not cover"):
    make_buffer(teacher_codes={0: 0})
  with pytest.raises(ReplayValidationError, match="do not cover"):
    make_buffer(frame_counts={0: 12, 1: 12, 2: 12})
  with pytest.raises(ReplayValidationError, match="must be distinct"):
    make_buffer(teacher_codes={0: 0, 1: 0})
  with pytest.raises(ReplayValidationError, match="must be non-negative"):
    make_buffer(teacher_codes={0: 0, 1: -1})
  with pytest.raises(ReplayValidationError, match="must be an integer"):
    make_buffer(teacher_codes=invalid({0: 0, 1: 1.0}))
  with pytest.raises(ReplayValidationError, match="must be positive"):
    make_buffer(frame_counts={0: 12, 1: 0})
  with pytest.raises(ReplayValidationError, match="must be a mapping"):
    make_buffer(teacher_codes=invalid([0, 1]))


def test_bad_routing_and_frames_in_a_batch_are_rejected() -> None:
  buffer = make_buffer(capacity=8)
  with pytest.raises(ReplayValidationError, match="no replay partition"):
    buffer.insert(make_motion_batch([0, 4]))
  with pytest.raises(ReplayValidationError, match="no replay partition"):
    buffer.insert(make_motion_batch([0, -1]))
  with pytest.raises(ReplayValidationError, match="teacher ids disagree"):
    buffer.insert(make_motion_batch([0, 1], codes=[1, 1]))
  with pytest.raises(ReplayValidationError, match="teacher ids disagree"):
    buffer.insert(make_motion_batch([0], codes=[-1]))
  with pytest.raises(ReplayValidationError, match="inside each row's own clip"):
    buffer.insert(make_motion_batch([1], frames=[FRAMES[1]]))
  with pytest.raises(ReplayValidationError, match="inside each row's own clip"):
    buffer.insert(make_motion_batch([1], frames=[-1]))
  assert buffer.size == 0
  buffer.insert(make_motion_batch([0, 1], frames=[FRAMES[0] - 1, 0]))
  assert buffer.size == 2
  frames = buffer.sample(
    2, replacement=False, generator=torch.Generator().manual_seed(0)
  ).reference_frame
  assert sorted(frames.tolist()) == [0, FRAMES[0] - 1]


def test_batch_size_one_and_odd_batches_keep_the_configured_mixture() -> None:
  buffer = make_buffer(capacity=8)
  buffer.insert(make_motion_batch([0] * 4 + [1] * 4, start=0))
  generator = torch.Generator().manual_seed(7)
  singles = [tags_of(buffer.sample(1, generator=generator)) for _ in range(400)]
  counts = Counter(int(tag) // 1000 for tag in (row[0] for row in singles))
  assert 120 <= counts[0] <= 280
  assert counts[0] + counts[1] == 400

  odd = [buffer.sample(3, generator=generator) for _ in range(200)]
  odd_counts = Counter(int(tag) // 1000 for batch in odd for tag in tags_of(batch))
  assert 200 <= odd_counts[0] <= 400
  assert odd_counts[0] + odd_counts[1] == 600
  per_draw = {sum(int(tag) // 1000 == 0 for tag in tags_of(batch)) for batch in odd}
  assert per_draw == {1, 2}


def test_draws_are_reproducible_and_next_sample_is_deterministic() -> None:
  first = make_buffer(capacity=8)
  second = make_buffer(capacity=8)
  rows = make_motion_batch([0] * 5 + [1] * 3, start=0)
  first.insert(rows)
  second.insert(rows)
  left = [
    tags_of(first.sample(size, generator=torch.Generator().manual_seed(4)))
    for size in (1, 3, 5, 8)
  ]
  right = [
    tags_of(second.sample(size, generator=torch.Generator().manual_seed(4)))
    for size in (1, 3, 5, 8)
  ]
  assert left == right

  without = [
    tags_of(
      first.sample(size, replacement=False, generator=torch.Generator().manual_seed(5))
    )
    for size in (2, 4)
  ]
  repeat = [
    tags_of(
      second.sample(size, replacement=False, generator=torch.Generator().manual_seed(5))
    )
    for size in (2, 4)
  ]
  assert without == repeat
  for batch_tags in without:
    assert len(set(batch_tags)) == len(batch_tags)

  # Restoring generator state reproduces the next sample exactly.
  generator = torch.Generator().manual_seed(17)
  state = generator.get_state()
  expected = tags_of(first.sample(4, generator=generator))
  generator.set_state(state)
  assert tags_of(first.sample(4, generator=generator)) == expected

  # An explicit generator never consumes the global CPU stream.
  torch.manual_seed(0)
  global_state = torch.get_rng_state()
  first.sample(4, generator=torch.Generator().manual_seed(1))
  assert torch.equal(torch.get_rng_state(), global_state)
  torch.manual_seed(0)
  implicit = tags_of(first.sample(4))
  torch.manual_seed(0)
  assert tags_of(first.sample(4)) == implicit


def test_inserts_and_draws_own_their_tensors() -> None:
  buffer = make_buffer(capacity=8)
  source = make_motion_batch([0, 1], start=0)
  source.reference.requires_grad_()
  buffer.insert(source)
  assert not buffer.is_empty
  assert buffer.retained(0) == 1 and buffer.retained(1) == 1

  # Mutating the caller's tensors (including its metadata) after the insert
  # cannot change the retained rows.
  source.reference.data.fill_(999.0)
  source.motion_id.fill_(-5)
  source.episode_id.fill_(77)
  generator = torch.Generator().manual_seed(3)
  drawn = buffer.sample(4, generator=generator)
  assert not drawn.reference.requires_grad
  assert sorted(tags_of(drawn)) == [0.0, 0.0, 1001.0, 1001.0]
  assert set(drawn.motion_id.tolist()) == {0, 1}
  assert sorted(drawn.episode_id.tolist()) == [0, 0, 1, 1]

  before = buffer.state_dict()
  drawn.reference.data.fill_(12345.0)
  drawn.motion_id.fill_(9)
  assert_state_equal(buffer.state_dict(), before)
  generator = torch.Generator().manual_seed(3)
  assert tags_of(buffer.sample(4, generator=generator)) == tags_of(
    buffer.sample(4, generator=torch.Generator().manual_seed(3))
  )
  assert set(
    buffer.sample(4, generator=torch.Generator().manual_seed(3)).motion_id.tolist()
  ) == {0, 1}


def test_without_replacement_is_distinct_and_refuses_impossible_requests() -> None:
  buffer = make_buffer(capacity=4)
  buffer.insert(make_motion_batch([0, 0, 1, 1], start=0))
  complete = buffer.sample(
    4, replacement=False, generator=torch.Generator().manual_seed(2)
  )
  assert sorted(tags_of(complete)) == [0.0, 1.0, 1002.0, 1003.0]
  with pytest.raises(ReplayValidationError, match="cannot exceed buffer size"):
    buffer.sample(5, replacement=False)

  # Quotas (3, 2): a without-replacement draw of three floors motion 0 at two
  # rows while only one is retained, so the request is refused rather than
  # silently reweighted.
  skewed = make_buffer(capacity=5, weights={0: 4.0, 1: 1.0})
  assert skewed.quota.quotas == (3, 2)
  skewed.insert(make_motion_batch([0], start=0))
  skewed.insert(make_motion_batch([1, 1], start=0))
  assert skewed.ready
  with pytest.raises(ReplayValidationError, match="which retains 1"):
    skewed.sample(3, replacement=False, generator=torch.Generator().manual_seed(1))
  with_replacement = skewed.sample(3, generator=torch.Generator().manual_seed(1))
  assert len(tags_of(with_replacement)) == 3


def test_state_round_trip_preserves_partitions_weights_and_next_sample() -> None:
  buffer = make_buffer(capacity=10)
  buffer.insert(make_motion_batch([0] * 4 + [1] * 2, start=0))
  buffer.insert(make_motion_batch([0] * 3 + [1] * 3, start=10))
  state = buffer.state_dict()
  restored = make_buffer(capacity=10)
  restored.load_state_dict(state)
  assert restored.size == buffer.size
  assert restored.quota.as_dict() == buffer.quota.as_dict()
  assert restored.report().as_dict() == buffer.report().as_dict()
  assert restored.max_valid_segment_id() == buffer.max_valid_segment_id()
  for size in (1, 4, 7):
    assert tags_of(
      restored.sample(size, generator=torch.Generator().manual_seed(size))
    ) == tags_of(buffer.sample(size, generator=torch.Generator().manual_seed(size)))
  # The restored buffer keeps inserting into the same partitions.
  before = retained_tags(buffer, 0)
  restored.insert(make_motion_batch([0], start=99))
  buffer.insert(make_motion_batch([0], start=99))
  assert retained_tags(restored, 0) == retained_tags(buffer, 0)
  assert retained_tags(restored, 0) != before

  # A device policy that disagrees with the checkpoint is refused up front.
  pinned = make_buffer(capacity=10, device="cpu")
  mismatched = dict(state)
  mismatched["device"] = "cuda:0"
  with pytest.raises(ReplayValidationError, match="device does not match"):
    pinned.load_state_dict(mismatched)
  assert pinned.is_empty


def test_state_snapshot_is_deterministic_and_ignores_unwritten_slots() -> None:
  buffer = make_buffer(capacity=8)
  buffer.insert(make_motion_batch([0, 1], start=0))
  # Unwritten ring rows hold undefined memory, so a snapshot must not include
  # them; the same logical state therefore always snapshots identically.
  assert_state_equal(buffer.state_dict(), buffer.state_dict())
  for partition in buffer.state_dict()["partitions"]:
    assert partition["size"] == 1
  # Each motion occupies its own contiguous segment; the rows past its one
  # retained record are outside every live window and must snapshot as zeros.
  reference = buffer.state_dict()["fields"]["reference"]
  assert bool((reference[1:4] == 0).all())
  assert bool((reference[5:8] == 0).all())

  # A snapshot that still carries undefined unwritten rows is accepted, because
  # only the live windows are restored.
  dirty = buffer.state_dict()
  dirty["fields"]["reference"][1:4] = float("nan")
  dirty["fields"]["reference"][5:8] = float("nan")
  restored = make_buffer(capacity=8)
  restored.load_state_dict(dirty)
  assert restored.size == 2
  assert sorted(
    tags_of(
      restored.sample(2, replacement=False, generator=torch.Generator().manual_seed(1))
    )
  ) == [0.0, 1001.0]


def test_state_restore_is_transactional_for_invalid_later_partition() -> None:
  buffer = make_buffer(capacity=8)
  buffer.insert(make_motion_batch([0, 0, 0, 1, 1], start=0))
  good = buffer.state_dict()
  buffer.validate_state(good)

  def corrupted(mutate) -> dict:
    candidate = buffer.state_dict()
    mutate(candidate)
    return candidate

  def wrong_shape(state: dict) -> None:
    state["fields"]["reference"] = state["fields"]["reference"][:1]

  def non_finite(state: dict) -> None:
    state["fields"]["reference"][0, 0] = float("inf")

  def bad_cursor(state: dict) -> None:
    state["partitions"][1]["next"] = state["partitions"][1]["quota"]

  def bad_inserted(state: dict) -> None:
    state["partitions"][1]["inserted"] = 0

  def bad_size(state: dict) -> None:
    state["size"] = 3

  def bad_kind(state: dict) -> None:
    state["kind"] = "single-partition-fifo"

  def bad_version(state: dict) -> None:
    state["version"] = 99

  def bad_capacity(state: dict) -> None:
    state["capacity"] = 16

  def bad_mapping(state: dict) -> None:
    state["motion_ids"] = [1, 0]
    state["partitions"] = list(reversed(state["partitions"]))

  def bad_weights(state: dict) -> None:
    state["weights"] = [1.0, 2.0]

  def bad_codes(state: dict) -> None:
    state["teacher_codes"] = [1, 0]

  def bad_metadata_dtype(state: dict) -> None:
    state["fields"]["episode_id"] = state["fields"]["episode_id"].to(torch.float32)

  def unknown_field(state: dict) -> None:
    state["extra"] = 1

  for mutate in (
    wrong_shape,
    non_finite,
    bad_cursor,
    bad_inserted,
    bad_size,
    bad_kind,
    bad_version,
    bad_capacity,
    bad_mapping,
    bad_weights,
    bad_codes,
    bad_metadata_dtype,
    unknown_field,
  ):
    baseline = buffer.state_dict()
    candidate = corrupted(mutate)
    with pytest.raises(ReplayValidationError):
      buffer.validate_state(candidate)
    with pytest.raises(ReplayValidationError):
      buffer.load_state_dict(candidate)
    assert_state_equal(buffer.state_dict(), baseline)
    assert buffer.size == 5

  # The surviving buffer still holds and samples its own retained rows.
  assert tags_of(
    buffer.sample(4, generator=torch.Generator().manual_seed(6))
  ) == tags_of(buffer.sample(4, generator=torch.Generator().manual_seed(6)))


def test_segment_namespace_seams_are_storage_independent() -> None:
  buffer = make_buffer(capacity=8)
  assert buffer.max_valid_segment_id() is None
  assert buffer.rebase_segment_ids(1, 50) == 0
  buffer.insert(make_motion_batch([0, 0, 1], start=0, iterations=[1, 1, 1]))
  buffer.insert(make_motion_batch([0, 1], start=3, iterations=[2, 2]))
  assert buffer.max_valid_segment_id() == 4
  assert buffer.rebase_segment_ids(1, 100) == 3
  assert buffer.max_valid_segment_id() == 102
  assert buffer.rebase_segment_ids(1, 0) == 0
  assert buffer.rebase_segment_ids(5, 10) == 0
  with pytest.raises(ReplayValidationError, match="rebase arguments"):
    buffer.rebase_segment_ids(1, -4)
  with pytest.raises(ReplayValidationError, match="rebase arguments"):
    buffer.rebase_segment_ids(invalid(1.5), 4)

  fifo = LabeledReplayBuffer(capacity=8, schema=DEFAULT_SCHEMA)
  assert fifo.max_valid_segment_id() is None
  fifo.insert(make_motion_batch([0, 0, 1], start=6))
  assert fifo.max_valid_segment_id() == 8


def test_protocol_conformance_and_trainer_accepts_balanced_replay() -> None:
  fifo = LabeledReplayBuffer(capacity=8, schema=DEFAULT_SCHEMA)
  balanced = make_buffer(capacity=8)
  assert isinstance(fifo, ReplayBufferProtocol)
  assert isinstance(balanced, ReplayBufferProtocol)
  fifo.validate_batch(fifo_batch())
  fifo.insert(fifo_batch())
  balanced.insert(make_motion_batch([0, 1], start=0))
  for buffer in (fifo, balanced):
    with pytest.raises(ReplayValidationError, match="expected torch.float32"):
      buffer.validate_batch(make_motion_batch([0], start=0, dtype=torch.float64))
  with pytest.raises(TrainerValidationError, match="protocol"):
    VaeDistillationTrainer(
      ConditionalVAE(settings=ModelSettings(hidden_dims=(8, 8))),
      invalid(object()),
      TrainingConfig(accumulation_steps=1, minibatch_size=2),
    )

  balanced.insert(make_motion_batch([0, 0, 1, 1], start=0))
  model = ConditionalVAE(settings=ModelSettings(hidden_dims=(8, 8)))
  trainer = VaeDistillationTrainer(
    model, balanced, TrainingConfig(accumulation_steps=2, minibatch_size=4), seed=5
  )
  assert trainer.replay is balanced
  fresh = make_motion_batch([0, 1], start=40)
  trainer.begin_training(FreshTrainingData(fresh, "collector-0"))
  update = trainer.train_update()
  assert update.optimizer_step == 1
  assert update.microbatches == 2
  assert update.samples == 8
  assert torch.isfinite(torch.tensor(update.total_loss))


def fifo_batch() -> LabeledReplayBatch:
  """A single-teacher batch for the FIFO protocol check."""
  batch = make_motion_batch([0, 0])
  return LabeledReplayBatch(
    observations=batch.observations,
    teacher_action=batch.teacher_action,
    motion_id=batch.motion_id,
    teacher_id=torch.zeros_like(batch.motion_id),
    reference_frame=batch.reference_frame,
    episode_id=batch.episode_id,
    collector_iteration=batch.collector_iteration,
  )


def test_normalizers_count_every_fresh_raw_row_once_not_balanced_draws() -> None:
  buffer = make_buffer(capacity=16)
  assert buffer.quota.quotas == (8, 8)
  fresh = make_motion_batch([0, 0, 0, 0, 0, 0, 1, 1], start=0)
  buffer.insert(fresh)
  assert buffer.size == 8
  assert buffer.ready

  model = ConditionalVAE(settings=ModelSettings(hidden_dims=(8, 8)))
  trainer = VaeDistillationTrainer(
    model, buffer, TrainingConfig(accumulation_steps=1, minibatch_size=4), seed=3
  )
  trainer.begin_training(FreshTrainingData(fresh, "collector-0"))
  assert float(model.reference_normalizer.state_dict()["count"]) == fresh.batch_size
  assert float(model.conditioning_normalizer.state_dict()["count"]) == fresh.batch_size
  torch.testing.assert_close(
    model.reference_normalizer.state_dict()["mean"], fresh.reference.mean(dim=0)
  )
  # The replay mixture is the configured one (4 and 4), not the raw 6/2 split.
  draw = buffer.sample(8, generator=torch.Generator().manual_seed(2))
  assert Counter(draw.motion_id.tolist()) == {0: 4, 1: 4}

  # Repeated replay draws never move the student statistics.
  trainer.freeze_normalizers()
  counts = (
    float(model.reference_normalizer.state_dict()["count"]),
    float(model.conditioning_normalizer.state_dict()["count"]),
  )
  for _ in range(200):
    buffer.sample(4, generator=torch.Generator().manual_seed(9))
  assert (
    float(model.reference_normalizer.state_dict()["count"]),
    float(model.conditioning_normalizer.state_dict()["count"]),
  ) == counts
  assert counts == (8.0, 8.0)

  # A fresh identity is consumed exactly once.
  trainer.begin_training()
  with pytest.raises(TrainerValidationError, match="already used"):
    trainer.update_normalizers_from_new_data(fresh, update_id="collector-0")
  assert float(model.reference_normalizer.state_dict()["count"]) == 8


def test_coverage_sampling_forces_one_unknown_mvp_row() -> None:
  base = make_motion_batch([0, 0, 0, 1, 1, 1])
  batch = replace(
    base,
    initialization_kind=torch.tensor([1, 1, 0, 1, 1, 0], dtype=torch.int64),
    segment_initial_reference_frame=torch.zeros(6, dtype=torch.int64),
    segment_age=torch.zeros(6, dtype=torch.int64),
  )
  buffer = BalancedReplayBuffer(
    8,
    DEFAULT_SCHEMA,
    {0: 1.0, 1: 1.0},
    teacher_codes=CODES,
    frame_counts=FRAMES,
    provenance_enabled=True,
    coverage_sampling=True,
  )
  buffer.insert(batch)

  drawn = buffer.sample(4, generator=torch.Generator().manual_seed(3))

  assert drawn.initialization_kind is not None
  assert bool((drawn.initialization_kind == 0).any().item())
  assert buffer.coverage_drawn >= 1
  report = buffer.report().as_dict()
  assert report["coverage_sampling"] is True
  assert report["coverage_drawn"] >= 1
  restored = BalancedReplayBuffer(
    8,
    DEFAULT_SCHEMA,
    {0: 1.0, 1: 1.0},
    teacher_codes=CODES,
    frame_counts=FRAMES,
    provenance_enabled=True,
    coverage_sampling=True,
  )
  restored.load_state_dict(buffer.state_dict())
  assert restored.coverage_drawn == buffer.coverage_drawn


def test_report_exposes_capacity_occupancy_and_coverage() -> None:
  buffer = make_buffer(capacity=8)
  empty = buffer.report()
  assert empty.capacity == 8
  assert empty.retained == 0 and empty.inserted == 0 and empty.drawn == 0
  assert empty.occupancy == 0.0 and not empty.ready
  assert empty.as_dict()["motions"][0]["coverage"] == 0.0

  buffer.insert(make_motion_batch([0, 1, 1], start=0))
  buffer.sample(5, generator=torch.Generator().manual_seed(1))
  report = buffer.report()
  assert report.retained == 3 and report.inserted == 3 and report.drawn == 5
  assert report.occupancy == pytest.approx(3 / 8)
  assert report.ready and report.unready_motion_ids == ()
  by_motion = {motion.motion_id: motion for motion in report.motions}
  assert by_motion[0].retained == 1 and by_motion[0].quota == 4
  assert by_motion[0].coverage == pytest.approx(0.25)
  assert by_motion[0].inserted == 1 and by_motion[1].inserted == 2
  assert by_motion[0].drawn + by_motion[1].drawn == 5
  assert set(report.as_dict()) == {
    "capacity",
    "retained",
    "inserted",
    "drawn",
    "occupancy",
    "ready",
    "unready_motion_ids",
    "motions",
  }


def test_dtype_and_device_policy_is_enforced_before_insertion() -> None:
  buffer = make_buffer(capacity=8, device="cpu", dtype=torch.float64)
  assert buffer.device == torch.device("cpu")
  assert buffer.dtype == torch.float64
  with pytest.raises(ReplayValidationError, match="expected torch.float64"):
    buffer.insert(make_motion_batch([0, 1], start=0))
  assert buffer.is_empty
  buffer.insert(make_motion_batch([0, 1], start=0, dtype=torch.float64))
  assert buffer.size == 2
  assert buffer.sample(
    2, generator=torch.Generator().manual_seed(0)
  ).reference.dtype == (torch.float64)


def test_sampling_rejects_invalid_arguments() -> None:
  buffer = make_buffer(capacity=8)
  buffer.insert(make_motion_batch([0, 1], start=0))
  for size in (0, -1, True, 1.5):
    with pytest.raises(ReplayValidationError, match="positive integer"):
      buffer.sample(invalid(size))
  with pytest.raises(ReplayValidationError, match="must be a bool"):
    buffer.sample(1, replacement=invalid(1))


def test_next_sample_and_next_update_are_deterministic() -> None:
  rows = make_motion_batch([0, 0, 0, 1, 1], start=0)
  first = make_buffer(capacity=8)
  second = make_buffer(capacity=8)
  first.insert(rows)
  second.insert(rows)

  generator = torch.Generator().manual_seed(3)
  state = generator.get_state()
  expected = tags_of(first.sample(4, generator=generator))
  generator.set_state(state)
  assert tags_of(second.sample(4, generator=generator)) == expected

  twin = ConditionalVAE(settings=ModelSettings(hidden_dims=(8, 8)))
  model = ConditionalVAE(settings=ModelSettings(hidden_dims=(8, 8)))
  twin.load_state_dict(model.state_dict())
  config = TrainingConfig(accumulation_steps=2, minibatch_size=4)
  left = VaeDistillationTrainer(model, first, config, seed=13)
  right = VaeDistillationTrainer(twin, second, config, seed=13)
  left.begin_training(FreshTrainingData(rows, "collector-0"))
  right.begin_training(FreshTrainingData(rows, "collector-0"))
  for _ in range(3):
    left_update = left.train_update()
    right_update = right.train_update()
    assert left_update.total_loss == pytest.approx(right_update.total_loss, rel=1e-9)
  assert left.samples_seen == right.samples_seen == 3 * 2 * 4
  for one, other in zip(model.parameters(), twin.parameters(), strict=True):
    torch.testing.assert_close(one, other, rtol=1e-6, atol=1e-8)


def segment_buffer(quota: int, motions: int = 4) -> BalancedReplayBuffer:
  """A balanced buffer of ``motions`` equally weighted segments of ``quota``."""
  return BalancedReplayBuffer(
    motions * quota,
    DEFAULT_SCHEMA,
    {motion: 1.0 for motion in range(motions)},
    teacher_codes={motion: motion for motion in range(motions)},
    frame_counts={motion: FRAMES[0] for motion in range(motions)},
  )


def test_every_drawn_row_comes_from_its_own_segments_live_window() -> None:
  """A draw reads only the live window of each motion's own flat segment.

  Five insertions into a quota-three segment leave rows 2..4 retained, so every
  drawn tag must be one of those rows of its own motion: a row from another
  segment, or one the wrap has already overwritten, would show up here.  A
  second buffer keeps every segment half empty and poisons every row outside
  the live windows, which proves the draw never reads a slot past a cursor.
  """
  motions = 4
  wrapped = segment_buffer(quota=3, motions=motions)
  for start in range(5):
    for motion in range(motions):
      wrapped.insert(make_motion_batch([motion], start=start))
  assert [wrapped.retained(motion) for motion in range(motions)] == [3] * motions
  for seed in range(4):
    drawn = wrapped.sample(64, generator=torch.Generator().manual_seed(seed))
    tags = tags_of(drawn)
    assert bool(torch.tensor(tags).isfinite().all())
    for tag, motion in zip(tags, drawn.motion_id.tolist(), strict=True):
      assert int(tag) // 1000 == motion
      assert int(tag) % 1000 in {2, 3, 4}

  capacity = motions * 4
  partial = segment_buffer(quota=4, motions=motions)
  for start in range(2):
    for motion in range(motions):
      partial.insert(make_motion_batch([motion], start=start))
  state = partial.state_dict()
  live: set[int] = set()
  base = 0
  for partition in state["partitions"]:
    quota, size, cursor = partition["quota"], partition["size"], partition["next"]
    first = (cursor - size) % quota
    live |= {base + (first + row) % quota for row in range(size)}
    base += quota
  outside = [row for row in range(capacity) if row not in live]
  assert len(live) == 8 and len(outside) == 8
  for value in state["fields"].values():
    value[outside] = float("nan") if value.is_floating_point() else -1
  poisoned = segment_buffer(quota=4, motions=motions)
  poisoned.load_state_dict(state)
  for seed in range(4):
    drawn = poisoned.sample(64, generator=torch.Generator().manual_seed(seed))
    tags = tags_of(drawn)
    assert bool(torch.tensor(tags).isfinite().all())
    for tag, motion in zip(tags, drawn.motion_id.tolist(), strict=True):
      assert int(tag) // 1000 == motion
      assert int(tag) % 1000 in {0, 1}


def test_restored_segments_reproduce_the_same_draw_sequence() -> None:
  """A saved and restored buffer draws the same rows for the same seed."""
  buffer = make_buffer(capacity=10)
  buffer.insert(make_motion_batch([0] * 4 + [1] * 2, start=0))
  buffer.insert(make_motion_batch([0] * 2 + [1] * 3, start=10))
  buffer.sample(5, generator=torch.Generator().manual_seed(2))
  state = buffer.state_dict()

  restored = make_buffer(capacity=10)
  restored.load_state_dict(state)
  assert restored.report().as_dict() == buffer.report().as_dict()
  left = torch.Generator().manual_seed(21)
  right = torch.Generator().manual_seed(21)
  for size in (1, 3, 6, 9):
    torch.testing.assert_close(
      restored.sample(size, generator=left).reference,
      buffer.sample(size, generator=right).reference,
    )
  # The restored cursors keep accepting insertions identically too.
  restored.insert(make_motion_batch([0, 1], start=40))
  buffer.insert(make_motion_batch([0, 1], start=40))
  assert retained_tags(restored, 0) == retained_tags(buffer, 0)
  assert retained_tags(restored, 1) == retained_tags(buffer, 1)
  assert restored.report().as_dict() == buffer.report().as_dict()


def phase_provenance_buffer() -> BalancedReplayBuffer:
  """A provenance- and phase-enabled buffer for the checkpoint tallies."""
  return BalancedReplayBuffer(
    8,
    DEFAULT_SCHEMA,
    {0: 1.0, 1: 1.0},
    teacher_codes=CODES,
    frame_counts=FRAMES,
    provenance_enabled=True,
    coverage_sampling=True,
    phase_bins=3,
  )


def test_kind_tallies_and_phase_counters_round_trip_through_a_checkpoint() -> None:
  base = make_motion_batch([0, 0, 0, 1, 1, 1], start=0, frames=[0, 5, 9, 0, 5, 9])
  batch = replace(
    base,
    initialization_kind=torch.tensor([1, 1, 0, 2, 1, 0], dtype=torch.int64),
    segment_initial_reference_frame=torch.zeros(6, dtype=torch.int64),
    segment_age=torch.zeros(6, dtype=torch.int64),
  )
  buffer = phase_provenance_buffer()
  buffer.insert(batch)
  buffer.sample(4, generator=torch.Generator().manual_seed(5))
  before = buffer.report().as_dict()

  restored = phase_provenance_buffer()
  restored.load_state_dict(buffer.state_dict())
  assert restored.report().as_dict() == before
  # The tallies are nonzero, so the round trip above is a real assertion.
  assert (
    sum(
      motion["inserted_by_initialization"].get("unknown", 0)
      for motion in before["motions"]
    )
    == 2
  )
  assert sum(sum(motion["drawn_by_phase_bin"]) for motion in before["motions"]) == 4
  left = torch.Generator().manual_seed(8)
  right = torch.Generator().manual_seed(8)
  torch.testing.assert_close(
    restored.sample(6, generator=left).reference,
    buffer.sample(6, generator=right).reference,
  )


def test_a_pre_flat_ring_checkpoint_is_refused_by_its_version() -> None:
  """The flat rings are checkpoint version 2; version 1 is not readable.

  Version 1 stored one ring per motion per field, so its ``partitions`` carry a
  ``storage`` table and the state has no ``fields`` table.  It is refused on its
  version, which is the change an operator needs to see, and the live buffer is
  untouched.
  """
  buffer = make_buffer(capacity=8)
  buffer.insert(make_motion_batch([0, 1], start=0))
  before = buffer.state_dict()
  legacy = dict(before)
  legacy["version"] = 1
  legacy.pop("fields")
  legacy["partitions"] = [
    {**partition, "storage": None} for partition in before["partitions"]
  ]
  with pytest.raises(ReplayValidationError, match="version is not supported"):
    buffer.load_state_dict(legacy)
  assert_state_equal(buffer.state_dict(), before)
  assert buffer.size == 2
