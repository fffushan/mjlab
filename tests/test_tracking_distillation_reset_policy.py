"""Pure CPU gates for the distillation-owned standing-start reset policy."""

from __future__ import annotations

import hashlib
import json
from typing import cast

import pytest
import torch

from mjlab.tasks.tracking.distillation.reset_policy import (
  RESET_POLICY_VERSION,
  ResetPolicy,
  ResetPolicyError,
  canonical_json,
  effective_window_frames,
  effective_windows,
  make_reset_policy,
  reset_policy_digest,
  sample_reset_policy,
)


def test_reference_default_is_a_valid_legacy_noop() -> None:
  policy = ResetPolicy()
  assert policy.kind == "reference"
  assert not policy.enabled
  assert policy.as_dict() == {
    "version": RESET_POLICY_VERSION,
    "kind": "reference",
    "standing_start_fraction": 0.25,
    "standing_start_window_frames": 25,
    "standing_start_frame_zero_fraction": 0.5,
  }


def test_effective_windows_are_clip_local_for_unequal_and_tiny_clips() -> None:
  policy = make_reset_policy(kind="standing-mixture", standing_start_window_frames=25)
  clips = torch.tensor([1, 7, 24, 25, 26, 1000], dtype=torch.int64)
  assert effective_windows(policy, clips).tolist() == [1, 7, 24, 25, 25, 25]
  assert effective_window_frames([1, 3, 2000], 4).tolist() == [1, 3, 4]


def test_standing_sampler_respects_clip_bounds_and_kind() -> None:
  policy = make_reset_policy(
    kind="standing-mixture",
    standing_start_fraction=0.5,
    standing_start_window_frames=25,
    standing_start_frame_zero_fraction=0.0,
  )
  clips = torch.tensor([1, 7, 24, 25, 26, 1000], dtype=torch.int64)
  result = sample_reset_policy(policy, clips, torch.Generator().manual_seed(11))
  assert result.kind.dtype is torch.bool
  assert result.frame.dtype is torch.int64
  assert torch.all(result.frame >= 0)
  for is_standing, frame, clip in zip(
    result.kind.tolist(), result.frame.tolist(), clips.tolist(), strict=True
  ):
    assert frame < (min(25, clip) if is_standing else clip)


def test_seeded_sampling_is_reproducible_and_has_no_global_rng_dependency() -> None:
  policy = make_reset_policy(
    kind="standing-mixture",
    standing_start_fraction=0.3,
    standing_start_frame_zero_fraction=0.4,
  )
  clips = [1, 19, 30, 1000] * 200
  first = sample_reset_policy(policy, clips, torch.Generator().manual_seed(91))
  second = sample_reset_policy(policy, clips, torch.Generator().manual_seed(91))
  assert torch.equal(first.kind, second.kind)
  assert torch.equal(first.frame, second.frame)


def test_realized_frame_zero_probability_includes_uniform_branch() -> None:
  policy = make_reset_policy(
    kind="standing-mixture",
    standing_start_fraction=1.0,
    standing_start_window_frames=4,
    standing_start_frame_zero_fraction=0.25,
  )
  draws = 40_000
  result = sample_reset_policy(
    policy, [100] * draws, torch.Generator().manual_seed(1234)
  )
  assert bool(result.kind.all())
  frequency = float((result.frame == 0).to(torch.float32).mean())
  # Expected p(frame 0 | standing) = .25 + .75 / 4 = .4375.
  assert frequency == pytest.approx(0.4375, abs=0.012)


def test_zero_and_one_fractions_and_frame_zero_branches() -> None:
  all_reference = make_reset_policy(
    kind="standing-mixture",
    standing_start_fraction=0.0,
    standing_start_frame_zero_fraction=1.0,
  )
  result = sample_reset_policy(
    all_reference, [1, 3, 1000], torch.Generator().manual_seed(1)
  )
  assert not bool(result.kind.any())
  assert result.frame.tolist()[0] == 0
  assert 0 <= result.frame.tolist()[1] < 3
  assert 0 <= result.frame.tolist()[2] < 1000

  all_zero_branch = make_reset_policy(
    kind="standing-mixture",
    standing_start_fraction=1.0,
    standing_start_window_frames=3,
    standing_start_frame_zero_fraction=0.0,
  )
  result = sample_reset_policy(
    all_zero_branch, [1, 2, 100], torch.Generator().manual_seed(1)
  )
  assert bool(result.kind.all())
  assert result.frame.tolist()[0] == 0
  assert all(
    frame < min(3, clip)
    for frame, clip in zip(result.frame.tolist(), [1, 2, 100], strict=True)
  )

  all_frame_zero = make_reset_policy(
    kind="standing-mixture",
    standing_start_fraction=1.0,
    standing_start_frame_zero_fraction=1.0,
  )
  result = sample_reset_policy(
    all_frame_zero, [1, 2, 100], torch.Generator().manual_seed(1)
  )
  assert result.frame.tolist() == [0, 0, 0]


def test_disabled_policy_does_not_consume_generator_or_global_rng() -> None:
  policy = ResetPolicy()
  generator = torch.Generator().manual_seed(37)
  before = generator.get_state()
  torch.manual_seed(99)
  global_before = torch.random.get_rng_state()
  result = sample_reset_policy(policy, [1, 24, 1000], generator)
  assert torch.equal(generator.get_state(), before)
  assert torch.equal(torch.random.get_rng_state(), global_before)
  assert result.kind.tolist() == [False, False, False]
  assert result.frame.tolist() == [0, 0, 0]


def test_policy_serialization_and_digest_are_canonical_and_hash_stable() -> None:
  policy = make_reset_policy(
    kind="standing-mixture",
    standing_start_fraction=0.25,
    standing_start_window_frames=25,
    standing_start_frame_zero_fraction=0.5,
  )
  encoded = policy.canonical_json()
  assert encoded == canonical_json(policy)
  assert json.loads(encoded) == policy.as_dict()
  restored = ResetPolicy.from_json(encoded)
  assert restored == policy
  assert restored.digest() == reset_policy_digest(policy)
  assert restored.digest() == hashlib.sha256(encoded.encode()).hexdigest()
  assert ResetPolicy.from_dict(policy.as_dict()).as_dict() == policy.as_dict()


@pytest.mark.parametrize(
  ("field", "value", "message"),
  [
    ("standing_start_fraction", float("nan"), "finite"),
    ("standing_start_fraction", float("inf"), "finite"),
    ("standing_start_fraction", -0.1, "[0, 1]"),
    ("standing_start_fraction", 1.1, "[0, 1]"),
    ("standing_start_frame_zero_fraction", float("nan"), "finite"),
    ("standing_start_frame_zero_fraction", -1.0, "[0, 1]"),
    ("standing_start_frame_zero_fraction", 2.0, "[0, 1]"),
    ("standing_start_window_frames", 0, "positive"),
    ("standing_start_window_frames", -2, "positive"),
    ("standing_start_window_frames", 2.5, "positive integer"),
    ("standing_start_window_frames", True, "positive integer"),
  ],
)
def test_invalid_options_are_rejected_actionably(
  field: str, value: object, message: str
) -> None:
  values = {
    "kind": "standing-mixture",
    "standing_start_fraction": 0.25,
    "standing_start_window_frames": 25,
    "standing_start_frame_zero_fraction": 0.5,
  }
  values[field] = value
  with pytest.raises(ResetPolicyError, match=message):
    make_reset_policy(**values)  # type: ignore[arg-type]


def test_invalid_kind_and_serialized_records_are_rejected() -> None:
  with pytest.raises(ResetPolicyError, match="reference.*standing-mixture"):
    make_reset_policy(kind="other")  # type: ignore[arg-type]
  with pytest.raises(ResetPolicyError, match="unsupported.*version"):
    ResetPolicy.from_dict({**ResetPolicy().as_dict(), "version": 99})
  with pytest.raises(ResetPolicyError, match="missing=.*kind"):
    ResetPolicy.from_dict({"version": 1})
  with pytest.raises(ResetPolicyError, match="unknown"):
    ResetPolicy.from_dict({**ResetPolicy().as_dict(), "unexpected": 1})


def test_sampler_rejects_bad_clip_lengths_and_policy_inputs() -> None:
  policy = make_reset_policy(kind="standing-mixture")
  generator = torch.Generator().manual_seed(3)
  with pytest.raises(ResetPolicyError, match="positive"):
    sample_reset_policy(policy, [1, 0, 3], generator)
  with pytest.raises(ResetPolicyError, match="integer"):
    sample_reset_policy(policy, cast(list[int], [1.0, 3.0]), generator)
  with pytest.raises(ResetPolicyError, match="one-dimensional"):
    sample_reset_policy(policy, torch.ones(2, 1, dtype=torch.int64), generator)
  with pytest.raises(ResetPolicyError, match="torch.Generator"):
    sample_reset_policy(policy, [1, 2], object())  # type: ignore[arg-type]


def test_empty_batch_is_valid_and_does_not_draw() -> None:
  policy = make_reset_policy(kind="standing-mixture")
  generator = torch.Generator().manual_seed(3)
  before = generator.get_state()
  result = sample_reset_policy(policy, [], generator)
  assert result.kind.shape == (0,)
  assert result.frame.shape == (0,)
  assert torch.equal(generator.get_state(), before)
