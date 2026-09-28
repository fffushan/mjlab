"""CPU tests for the opt-in phase-balanced within-motion replay draw.

The default ``phase_bins=0`` path must stay bitwise-identical to the
historical density-proportional draw; every test below pins that first and
then exercises the balanced policy: cell-uniform draws over non-empty cells,
motion-mixture invariance, deterministic ordering, fail-closed
without-replacement semantics, and additive per-phase statistics.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch

from mjlab.tasks.tracking.distillation.balanced_storage import (
  BalancedReplayBuffer,
  ReplayNotReadyError,
)
from mjlab.tasks.tracking.distillation.observations import PackedObservationBatch
from mjlab.tasks.tracking.distillation.storage import (
  LabeledReplayBatch,
  ReplayValidationError,
)
from mjlab.tasks.tracking.distillation.vae_config import DEFAULT_SCHEMA

# One motion with a 100-frame clip: phase cell = frame // 10 with 10 bins, so
# each block of 10 consecutive frames is exactly one cell.
FRAMES = {0: 100}
CODES = {0: 7}

_GOLDEN_PATH = Path(__file__).parent / "tracking_distillation_phase_golden.json"


def _batch(
  motion_ids: list[int],
  frames: list[int],
  *,
  iterations: list[int] | None = None,
  codes: Mapping[int, int] | None = None,
) -> LabeledReplayBatch:
  size = len(motion_ids)
  tags = torch.tensor(frames, dtype=torch.float32)
  reference = torch.zeros(size, DEFAULT_SCHEMA.reference_dim)
  reference[:, 0] = tags
  conditioning = torch.zeros(size, DEFAULT_SCHEMA.conditioning_dim)
  conditioning[:, 0] = tags
  action = tags.reshape(-1, 1).expand(-1, DEFAULT_SCHEMA.action_dim).contiguous()
  routing = codes if codes is not None else CODES
  return LabeledReplayBatch(
    observations=PackedObservationBatch(reference, conditioning, DEFAULT_SCHEMA),
    teacher_action=action,
    motion_id=torch.tensor(motion_ids, dtype=torch.int64),
    teacher_id=torch.tensor([routing[m] for m in motion_ids], dtype=torch.int64),
    reference_frame=torch.tensor(frames, dtype=torch.int64),
    episode_id=torch.arange(size, dtype=torch.int64),
    collector_iteration=torch.tensor(
      iterations if iterations is not None else [1] * size, dtype=torch.int64
    ),
  )


def _buffer(
  capacity: int = 512,
  *,
  phase_bins: int,
  frames: Mapping[int, int] | None = None,
  weights: Mapping[int, float] | None = None,
) -> BalancedReplayBuffer:
  return BalancedReplayBuffer(
    capacity,
    DEFAULT_SCHEMA,
    weights if weights is not None else {0: 1.0},
    teacher_codes=CODES,
    frame_counts=frames if frames is not None else FRAMES,
    phase_bins=phase_bins,
  )


def test_phase_bins_zero_reproduces_the_historical_draw_exactly() -> None:
  """Default-off consumes the generator and returns rows exactly as before."""
  frames = [i % 100 for i in range(64)]
  buffer_default = _buffer(phase_bins=0)
  buffer_default.insert(_batch([0] * len(frames), frames))
  buffer_explicit = _buffer(phase_bins=0)
  buffer_explicit.insert(_batch([0] * len(frames), frames))

  generator_a = torch.Generator().manual_seed(1234)
  generator_b = torch.Generator().manual_seed(1234)
  for _ in range(4):
    left = buffer_default.sample(16, generator=generator_a)
    right = buffer_explicit.sample(16, generator=generator_b)
    torch.testing.assert_close(left.reference, right.reference)
    torch.testing.assert_close(left.reference_frame, right.reference_frame)


def test_phase_bins_unset_reproduces_the_prechange_golden_sequences() -> None:
  """Default-off rows/counters/RNG stream match pre-change golden values.

  The golden file was generated once from the HEAD implementation *before*
  the phase-balanced change (see the plan doc): three multi-motion cases with
  varied weights, batch sizes including one, four draws each of reference
  tags + motion ids, per-partition ``drawn`` counters, and the first words of
  the final generator state.  Any behavioral drift on the default path —
  rows, generator consumption, or counter updates — fails here.
  """
  cases = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))
  assert len(cases) >= 3
  for case in cases:
    weights = {int(k): float(v) for k, v in case["weights"].items()}
    frame_counts = {int(k): int(v) for k, v in case["frame_counts"].items()}
    routing = {motion: 7 + motion for motion in weights}
    buffer = BalancedReplayBuffer(
      case["capacity"],
      DEFAULT_SCHEMA,
      weights,
      teacher_codes=routing,
      frame_counts=frame_counts,
    )
    buffer.insert(_batch(case["motions"], case["frames"], codes=routing))
    generator = torch.Generator().manual_seed(case["seed"])
    for expected in case["draws"]:
      batch = buffer.sample(case["batch_size"], generator=generator)
      observed = [float(v) for v in batch.reference[:, 0].tolist()] + [
        int(v) for v in batch.motion_id.tolist()
      ]
      assert observed == expected
    state = buffer.state_dict()
    assert [p["drawn"] for p in state["partitions"]] == case["drawn_counters"]
    assert [int(v) for v in generator.get_state().tolist()[:8]] == case["rng_state"]


def test_phase_bins_zero_matches_a_buffer_constructed_without_the_option() -> None:
  """The historical constructor call (no argument) is the same object policy."""
  legacy = BalancedReplayBuffer(
    512,
    DEFAULT_SCHEMA,
    {0: 1.0},
    teacher_codes=CODES,
    frame_counts=FRAMES,
  )
  assert legacy.phase_bins == 0
  frames = [i % 100 for i in range(64)]
  legacy.insert(_batch([0] * len(frames), frames))
  generator = torch.Generator().manual_seed(7)
  batch = legacy.sample(32, generator=generator)
  assert batch.batch_size == 32


def test_invalid_phase_bins_is_rejected_at_construction() -> None:
  for bad in (-1, True, 1.5, "4"):
    with pytest.raises(ReplayValidationError):
      _buffer(phase_bins=bad)  # type: ignore[arg-type]


def test_balanced_draw_equalizes_exposure_over_non_empty_cells() -> None:
  """A 10x density skew evens out: every non-empty cell is drawn equally."""
  # Cell 0 holds 50 rows; cells 1..9 hold 5 rows each (95 rows total).
  frames = [0] * 50 + [
    f for cell in range(1, 10) for f in range(cell * 10, cell * 10 + 5)
  ]
  buffer = _buffer(capacity=512, phase_bins=10)
  buffer.insert(_batch([0] * len(frames), frames))
  assert buffer.retained(0) == 95

  generator = torch.Generator().manual_seed(20260927)
  drawn_cells: Counter[int] = Counter()
  draws = 400
  for _ in range(draws):
    batch = buffer.sample(10, generator=generator)
    for frame in batch.reference_frame.tolist():
      drawn_cells[frame // 10] += 1
  total = sum(drawn_cells.values())
  assert total == draws * 10
  # Every non-empty cell receives an equal share in expectation.
  expected = total / 10
  for cell in range(10):
    assert drawn_cells[cell] == pytest.approx(expected, rel=0.08), (
      f"cell {cell} drawn {drawn_cells[cell]} of ~{expected}"
    )
  # Density-proportional would give cell 0 ten times cell 1's share; pin that
  # the balanced draw really departs from it.
  ratio = drawn_cells[0] / drawn_cells[1]
  assert 0.8 < ratio < 1.25


def test_empty_cells_receive_no_rows_and_do_not_break_the_draw() -> None:
  """Coverage cannot be invented: only cells 2 and 7 populated."""
  frames = [20 + i for i in range(6)] + [70 + i for i in range(6)]
  buffer = _buffer(capacity=64, phase_bins=10)
  buffer.insert(_batch([0] * len(frames), frames))
  generator = torch.Generator().manual_seed(11)
  for _ in range(25):
    batch = buffer.sample(8, generator=generator)
    cells = {frame // 10 for frame in batch.reference_frame.tolist()}
    assert cells <= {2, 7}
    assert cells == {2, 7}  # both populated cells always contribute


def test_one_row_draws_spread_over_cells_in_expectation() -> None:
  frames = [0] * 30 + [50 + i for i in range(30)]
  buffer = _buffer(capacity=128, phase_bins=2)
  buffer.insert(_batch([0] * len(frames), frames))
  generator = torch.Generator().manual_seed(3)
  counts = Counter(
    (buffer.sample(1, generator=generator).reference_frame.item() // 50)
    for _ in range(600)
  )
  assert counts[0] == pytest.approx(300, abs=40)
  assert counts[1] == pytest.approx(300, abs=40)


def test_draws_are_deterministic_for_a_generator_state() -> None:
  frames = [i % 100 for i in range(80)]
  buffer = _buffer(capacity=256, phase_bins=10)
  buffer.insert(_batch([0] * len(frames), frames))
  left = torch.Generator().manual_seed(99)
  right = torch.Generator().manual_seed(99)
  for _ in range(6):
    batch_left = buffer.sample(20, generator=left)
    batch_right = buffer.sample(20, generator=right)
    torch.testing.assert_close(batch_left.reference, batch_right.reference)
    torch.testing.assert_close(batch_left.teacher_action, batch_right.teacher_action)


def test_motion_mixture_is_unchanged_by_phase_balancing() -> None:
  """The across-motion mixture stays at the configured weights."""
  frames_0 = [i % 100 for i in range(60)]
  frames_1 = [i % 60 for i in range(60)]
  frames = frames_0 + frames_1
  motions = [0] * 60 + [1] * 60
  buffer = BalancedReplayBuffer(
    512,
    DEFAULT_SCHEMA,
    {0: 1.0, 1: 1.0},
    teacher_codes={0: 7, 1: 8},
    frame_counts={0: 100, 1: 60},
    phase_bins=10,
  )
  buffer.insert(_batch(motions, frames, codes={0: 7, 1: 8}))
  generator = torch.Generator().manual_seed(5)
  motion_counts: Counter[int] = Counter()
  for _ in range(300):
    batch = buffer.sample(10, generator=generator)
    motion_counts.update(batch.motion_id.tolist())
  total = sum(motion_counts.values())
  assert motion_counts[0] == pytest.approx(total / 2, rel=0.08)
  assert motion_counts[1] == pytest.approx(total / 2, rel=0.08)


def test_without_replacement_refuses_an_impossible_cell_request() -> None:
  """Per-cell overflow refuses the whole draw before any counter changes."""
  # Cell 0 holds 2 rows; cells 1..9 hold 20 rows each (within the 100 frames).
  frames = [0, 1] + [
    f for cell in range(1, 10) for f in range(cell * 10, min(cell * 10 + 20, 100))
  ]
  buffer = _buffer(capacity=512, phase_bins=10)
  buffer.insert(_batch([0] * len(frames), frames))
  drawn_before = buffer.stats()[0].drawn
  drawn_cells_before = list(buffer.stats()[0].drawn_by_phase_bin or ())
  generator = torch.Generator().manual_seed(13)
  # 50 rows over 10 non-empty cells: floors of 5 plus one residual row in one
  # cell asks for 6 > 2 from cell 0 -- refused before any mutation.
  with pytest.raises(ReplayValidationError):
    buffer.sample(50, replacement=False, generator=generator)
  stats = buffer.stats()[0]
  assert stats.drawn == drawn_before  # nothing was mutated
  assert list(stats.drawn_by_phase_bin or ()) == drawn_cells_before


def test_without_replacement_residual_never_exceeds_a_small_cell() -> None:
  """The residual allocation itself cannot exceed a small cell.

  Cells sized [1, 100] with count=3: floors of 1 plus residual rows.  The
  exact preflight (ceil(3/2)=2 > 1) refuses the whole draw, so the per-cell
  randperm can never be asked for more rows than the cell retains.
  """
  frames = [0] + [50 + i for i in range(50)]  # cell 0: 1 row, cell 1: 50 rows
  buffer = _buffer(capacity=128, phase_bins=2)
  buffer.insert(_batch([0] * len(frames), frames))
  drawn_before = buffer.stats()[0].drawn
  for seed in range(1, 9):
    with pytest.raises(ReplayValidationError):
      buffer.sample(3, replacement=False, generator=torch.Generator().manual_seed(seed))
  assert buffer.stats()[0].drawn == drawn_before
  # The satisfiable neighbor (2 rows: floor 1 + one residual, max 2 per cell)
  # draws without truncation.
  batch = buffer.sample(
    2, replacement=False, generator=torch.Generator().manual_seed(1)
  )
  assert batch.batch_size == 2


def test_without_replacement_balanced_draw_returns_distinct_rows() -> None:
  frames = [i % 100 for i in range(100)]
  buffer = _buffer(capacity=256, phase_bins=10)
  buffer.insert(_batch([0] * len(frames), frames))
  generator = torch.Generator().manual_seed(17)
  for _ in range(10):
    batch = buffer.sample(20, replacement=False, generator=generator)
    assert len(set(batch.reference_frame.tolist())) == 20


def test_readiness_contract_is_unchanged() -> None:
  buffer = BalancedReplayBuffer(
    64,
    DEFAULT_SCHEMA,
    {0: 1.0, 1: 1.0},
    teacher_codes={0: 7, 1: 8},
    frame_counts={0: 100, 1: 60},
    phase_bins=10,
  )
  buffer.insert(_batch([0, 0], [10, 80]))  # only motion 0 populated
  with pytest.raises(ReplayNotReadyError):
    buffer.sample(4, generator=torch.Generator().manual_seed(1))


def test_per_phase_statistics_are_reported_and_additive() -> None:
  frames = [0] * 50 + [50 + i for i in range(50)]
  buffer = _buffer(capacity=256, phase_bins=2)
  buffer.insert(_batch([0] * len(frames), frames))
  report = buffer.report()
  motion = report.motions[0]
  assert motion.retained_by_phase_bin == (50, 50)
  assert motion.drawn_by_phase_bin == (0, 0)
  assert report.as_dict()["motions"][0]["retained_by_phase_bin"] == [50, 50]

  generator = torch.Generator().manual_seed(23)
  for _ in range(10):
    buffer.sample(8, generator=generator)
  motion = buffer.stats()[0]
  assert sum(motion.drawn_by_phase_bin or ()) == 80
  assert motion.drawn_by_phase_bin == pytest.approx((40, 40), abs=6)


def test_phase_statistics_absent_when_disabled() -> None:
  frames = [i % 100 for i in range(40)]
  buffer = _buffer(capacity=128, phase_bins=0)
  buffer.insert(_batch([0] * len(frames), frames))
  motion = buffer.stats()[0]
  assert motion.retained_by_phase_bin is None
  assert motion.drawn_by_phase_bin is None
  assert "retained_by_phase_bin" not in motion.as_dict()


def test_state_round_trip_keeps_policy_and_future_draws() -> None:
  frames = [i % 100 for i in range(90)]
  buffer = _buffer(capacity=256, phase_bins=10)
  buffer.insert(_batch([0] * len(frames), frames))
  generator = torch.Generator().manual_seed(31)
  buffer.sample(20, generator=generator)  # draw something first
  state = buffer.state_dict()

  restored = _buffer(capacity=256, phase_bins=10)
  restored.load_state_dict(state)
  assert restored.phase_bins == 10
  left = torch.Generator().manual_seed(41)
  right = torch.Generator().manual_seed(41)
  for _ in range(4):
    torch.testing.assert_close(
      restored.sample(12, generator=left).reference,
      buffer.sample(12, generator=right).reference,
    )
  # The stored counters survive the restore.
  assert restored.stats()[0].drawn_by_phase_bin is not None


def test_state_validation_ignores_no_new_fields() -> None:
  """The ring state is policy-agnostic: same required key set as before."""
  frames = [i % 100 for i in range(30)]
  buffer = _buffer(capacity=64, phase_bins=10)
  buffer.insert(_batch([0] * len(frames), frames))
  state = buffer.state_dict()
  assert "phase_bins" not in state  # policy is construction state, not ring state
  # A phase_bins=0 buffer can restore the same ring: only the policy differs.
  unbalanced = _buffer(capacity=64, phase_bins=0)
  unbalanced.load_state_dict(state)
  assert unbalanced.phase_bins == 0


def test_balanced_draw_spreads_rows_over_each_motions_own_cells() -> None:
  """Per-motion per-cell allocation: each motion's cells share its own rows.

  Motion 0's clip has 100 frames (cell = frame // 10); motion 1's clip has 20
  frames (cell = frame // 2 with the same 10 bins).  Equal-motion mixture and
  balanced cells give each motion ~half the batch and each of that motion's
  populated cells an equal share of that half.
  """
  frames_0 = [i % 100 for i in range(60)]
  frames_1 = [i % 20 for i in range(60)]
  motions = [0] * 60 + [1] * 60
  frames = frames_0 + frames_1
  routing = {0: 7, 1: 8}
  buffer = BalancedReplayBuffer(
    512,
    DEFAULT_SCHEMA,
    {0: 1.0, 1: 1.0},
    teacher_codes=routing,
    frame_counts={0: 100, 1: 20},
    phase_bins=10,
  )
  buffer.insert(_batch(motions, frames, codes=routing))

  generator = torch.Generator().manual_seed(77)
  motion_cells: dict[int, Counter[int]] = {0: Counter(), 1: Counter()}
  draws = 300
  for _ in range(draws):
    batch = buffer.sample(20, generator=generator)
    for motion, frame in zip(
      batch.motion_id.tolist(), batch.reference_frame.tolist(), strict=True
    ):
      divisor = 10 if motion == 0 else 2
      motion_cells[motion][frame // divisor] += 1
  # Motion 0's 60 rows populate only cells 0..5 (10 rows each); motion 1's
  # 60 rows populate all 10 cells (6 rows each).  Each motion's own populated
  # cells share that motion's half of the batch.
  populated = {0: set(range(6)), 1: set(range(10))}
  for motion, cells in motion_cells.items():
    total = sum(cells.values())
    assert total == pytest.approx(draws * 10, rel=0.12)
    assert set(cells) <= populated[motion]
    per_cell = total / len(populated[motion])
    for cell in populated[motion]:
      assert cells[cell] == pytest.approx(per_cell, rel=0.15), (
        f"motion {motion} cell {cell}: {cells[cell]} of ~{per_cell}"
      )


def test_phase_cells_use_exact_integer_arithmetic_at_scale() -> None:
  """(frame * bins) // frames stays inside [0, bins) even at float-hostile scale."""
  from mjlab.tasks.tracking.distillation.balanced_storage import _phase_cells

  bins = 10
  for frame_count in (100, 453, 10**9, 10**15, 10**18):
    frames = torch.tensor([0, 1, frame_count // 2, frame_count - 1], dtype=torch.int64)
    cells = _phase_cells(frames, frame_count, bins)
    assert int(cells.min()) >= 0
    assert int(cells.max()) < bins
    # The last frame must land in the last cell, never outside the range.
    assert int(cells[-1]) == bins - 1


def test_enabled_stats_report_zero_cells_for_an_empty_partition() -> None:
  """An enabled-but-empty partition reports zeros instead of omitting the field."""
  buffer = BalancedReplayBuffer(
    64,
    DEFAULT_SCHEMA,
    {0: 1.0, 1: 1.0},
    teacher_codes={0: 7, 1: 8},
    frame_counts={0: 100, 1: 60},
    phase_bins=4,
  )
  buffer.insert(_batch([0, 0], [10, 80]))  # only motion 0 populated
  report = buffer.report().as_dict()
  motion_0, motion_1 = report["motions"]
  assert motion_0["retained_by_phase_bin"] == [1, 0, 0, 1]
  assert motion_1["retained_by_phase_bin"] == [0, 0, 0, 0]
  assert motion_1["drawn_by_phase_bin"] == [0, 0, 0, 0]


def test_trainer_update_draws_through_the_phase_balanced_path() -> None:
  """train_update consumes balanced draws; fresh-row normalizer counting is unchanged."""
  from mjlab.tasks.tracking.distillation.model import ConditionalVAE
  from mjlab.tasks.tracking.distillation.trainer import (
    FreshTrainingData,
    VaeDistillationTrainer,
  )
  from mjlab.tasks.tracking.distillation.training_config import TrainingConfig
  from mjlab.tasks.tracking.distillation.vae_config import ModelSettings

  # Cell 0 holds 6 rows, cell 1 holds 2: the replay is deliberately skewed.
  frames = [0] * 6 + [50, 51]
  buffer = _buffer(capacity=32, phase_bins=2)
  fresh = _batch([0] * len(frames), frames)
  buffer.insert(fresh)

  model = ConditionalVAE(settings=ModelSettings(hidden_dims=(8, 8)))
  trainer = VaeDistillationTrainer(
    model, buffer, TrainingConfig(accumulation_steps=1, minibatch_size=4), seed=3
  )
  trainer.begin_training(FreshTrainingData(fresh, "collector-0"))
  # Normalizers counted the fresh rows exactly once, regardless of draw policy.
  assert float(model.reference_normalizer.state_dict()["count"]) == 8
  assert float(model.conditioning_normalizer.state_dict()["count"]) == 8

  update = trainer.train_update()
  assert update.samples == 4
  counts = trainer.health_check()
  assert counts is not None
  # Repeated balanced draws never move the student statistics.
  reference_count = float(model.reference_normalizer.state_dict()["count"])
  for _ in range(25):
    buffer.sample(4, generator=torch.Generator().manual_seed(9))
  assert float(model.reference_normalizer.state_dict()["count"]) == reference_count

  # The drawn_by_phase_bin counter tracks the balanced exposure.
  stats = buffer.stats()[0]
  assert stats.drawn_by_phase_bin is not None
  assert sum(stats.drawn_by_phase_bin) >= 25 * 4


def _map_values(value: Any) -> Any:
  return value
