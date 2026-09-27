from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from mjlab.tasks.tracking.distillation.balanced_storage import BalancedReplayBuffer
from mjlab.tasks.tracking.distillation.observations import PackedObservationBatch
from mjlab.tasks.tracking.distillation.storage import (
  LabeledReplayBatch,
  LabeledReplayBuffer,
  ReplayValidationError,
)
from mjlab.tasks.tracking.distillation.vae_config import DEFAULT_SCHEMA


def batch(
  motion: list[int], *, kind: list[int] | None = None, age: list[int] | None = None
) -> LabeledReplayBatch:
  size = len(motion)
  reference = torch.arange(size * 68, dtype=torch.float32).reshape(size, 68)
  conditioning = torch.arange(size * 99, dtype=torch.float32).reshape(size, 99)
  action = torch.arange(size * 31, dtype=torch.float32).reshape(size, 31)
  ids = torch.tensor(motion, dtype=torch.int64)
  kwargs: dict[str, torch.Tensor] = {}
  if kind is not None:
    assert age is not None
    kwargs = {
      "initialization_kind": torch.tensor(kind, dtype=torch.int64),
      "segment_initial_reference_frame": torch.tensor(
        [10 + value for value in range(size)], dtype=torch.int64
      ),
      "segment_age": torch.tensor(age, dtype=torch.int64),
    }
  return LabeledReplayBatch(
    observations=PackedObservationBatch(reference, conditioning, DEFAULT_SCHEMA),
    teacher_action=action,
    motion_id=ids,
    teacher_id=ids,
    reference_frame=torch.zeros(size, dtype=torch.int64),
    episode_id=torch.arange(size, dtype=torch.int64),
    collector_iteration=torch.zeros(size, dtype=torch.int64),
    **kwargs,
  )


def test_provenance_fifo_sampling_and_round_trip_preserve_alignment() -> None:
  replay = LabeledReplayBuffer(2, DEFAULT_SCHEMA)
  replay.insert(batch([0, 0], kind=[2, 1], age=[0, 7]))
  replay.insert(batch([0], kind=[2], age=[1]))
  drawn = replay.sample(
    2, replacement=False, generator=torch.Generator().manual_seed(4)
  )
  assert drawn.initialization_kind is not None
  assert drawn.segment_initial_reference_frame is not None
  assert drawn.segment_age is not None
  rows = sorted(
    zip(
      drawn.initialization_kind.tolist(),
      drawn.segment_initial_reference_frame.tolist(),
      drawn.segment_age.tolist(),
      strict=True,
    )
  )
  assert rows == [(1, 11, 7), (2, 10, 1)]

  restored = LabeledReplayBuffer(2, DEFAULT_SCHEMA)
  restored.load_state_dict(replay.state_dict())
  again = restored.sample(
    2, replacement=False, generator=torch.Generator().manual_seed(4)
  )
  assert again.initialization_kind is not None
  assert again.segment_age is not None
  torch.testing.assert_close(drawn.initialization_kind, again.initialization_kind)
  torch.testing.assert_close(drawn.segment_age, again.segment_age)


def test_partial_provenance_and_invalid_late_batch_do_not_mutate() -> None:
  replay = LabeledReplayBuffer(3, DEFAULT_SCHEMA)
  replay.insert(batch([0], kind=[2], age=[0]))
  before = replay.state_dict()
  with pytest.raises(ReplayValidationError, match="supplied together"):
    replay.insert(replace(batch([0]), initialization_kind=torch.tensor([2])))
  assert replay.state_dict()["size"] == before["size"]
  with pytest.raises(ReplayValidationError, match="initialization_kind"):
    replay.insert(batch([0], kind=[9], age=[0]))
  assert replay.state_dict()["size"] == before["size"]
  with pytest.raises(ReplayValidationError, match="layout"):
    replay.insert(batch([0]))


def test_balanced_report_separates_initialization_categories() -> None:
  replay = BalancedReplayBuffer(
    4,
    DEFAULT_SCHEMA,
    {0: 1.0, 1: 1.0},
    teacher_codes={0: 0, 1: 1},
    frame_counts={0: 4, 1: 4},
  )
  replay.insert(batch([0, 1], kind=[2, 1], age=[0, 8]))
  report = replay.report().as_dict()
  by_motion = {item["motion_id"]: item for item in report["motions"]}
  assert by_motion[0]["retained_by_initialization"] == {"standing": 1}
  assert by_motion[1]["retained_by_initialization"] == {"reference": 1}
  drawn = replay.sample(4, generator=torch.Generator().manual_seed(5))
  assert drawn.initialization_kind is not None
  assert drawn.initialization_kind.shape == (4,)
  report_after = replay.report().as_dict()
  assert report_after["motions"][0]["drawn_by_initialization"]
