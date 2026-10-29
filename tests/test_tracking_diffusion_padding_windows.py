"""Window and split invariants of the D1 uniform-anchor layout.

A rollout writes eight padding rows before its clip rows, so every segment is
``N + 8`` rows long and its anchor grid starts on the clip's first frame.  These
tests pin the four properties that layout depends on: the window arithmetic, the
padding row never being a current row, the split group, and the clean/OU pair
identity.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from tracking_diffusion_fixtures import make_row

from mjlab.tasks.tracking.diffusion.collector import TrialSpec
from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT
from mjlab.tasks.tracking.diffusion.dataset import (
  WindowIndex,
  grouped_split,
  segment_window_scan,
)
from mjlab.tasks.tracking.diffusion.storage import TrajectoryRow

_PADDING = DEFAULT_CONTRACT.padding_steps


def _segment_rows(
  clip_frames: int,
  *,
  seed: int = 0,
  pose: str = "reference",
  phase: str = "clean",
) -> tuple[TrajectoryRow, ...]:
  """One rollout: eight padding rows, then the clip rows at stride one."""
  episode = f"run:motion:{seed}:{pose}:{phase}"
  rows: list[TrajectoryRow] = []
  for tick in range(_PADDING + clip_frames):
    reference_frame = tick - _PADDING
    row = make_row(tick, episode=episode)
    rows.append(
      replace(
        row,
        reference_frame=reference_frame,
        timestamp=reference_frame * DEFAULT_CONTRACT.period_seconds,
        group_key=f"motion:{seed}:{reference_frame}",
        provenance={
          "phase": phase,
          "reference_phase": "0.0",
          "initial_pose": pose,
          "initial_state_mode": "perturbed",
          "pair_id": f"motion:{seed}:{pose}",
          "initial_state_id": f"init:motion:{seed}:{pose}:perturbed",
        },
      )
    )
  return tuple(rows)


def test_a_segment_yields_one_window_per_anchor_and_no_padding_anchor() -> None:
  clip_frames = 60
  rows = _segment_rows(clip_frames)
  scan = segment_window_scan(rows, DEFAULT_CONTRACT)
  starts = np.flatnonzero(scan.eligible_starts)
  # N + 8 rows hold N - 32 full windows, and anchors run 0 .. N - 33.
  assert len(starts) == clip_frames - DEFAULT_CONTRACT.future_steps
  anchors = [
    scan.rows[position + DEFAULT_CONTRACT.current_index].reference_frame
    for position in starts
  ]
  assert anchors == list(range(0, clip_frames - DEFAULT_CONTRACT.future_steps))
  # Every window is 41 rows in one segment at stride one with no tick gap.
  for position in starts:
    window = scan.rows[position : position + DEFAULT_CONTRACT.window_steps]
    assert len(window) == 41
    assert len({row.segment_id for row in window}) == 1
    assert len({row.episode_id for row in window}) == 1
    assert all(
      window[index + 1].tick == window[index].tick + 1
      for index in range(len(window) - 1)
    )


def test_a_padding_row_is_never_a_current_row_or_a_target() -> None:
  clip_frames = 40
  rows = _segment_rows(clip_frames)
  scan = segment_window_scan(rows, DEFAULT_CONTRACT)
  for position in np.flatnonzero(scan.eligible_starts):
    window = scan.rows[position : position + DEFAULT_CONTRACT.window_steps]
    current = window[DEFAULT_CONTRACT.current_index]
    assert current.reference_frame >= 0
    # The padding rows can only appear in the history half of a window.
    for index, row in enumerate(window):
      if row.reference_frame < 0:
        assert index < DEFAULT_CONTRACT.current_index


def test_the_index_skips_a_window_whose_current_row_is_padding() -> None:
  clip_frames = 40
  rows = _segment_rows(clip_frames)
  scan = segment_window_scan(rows, DEFAULT_CONTRACT)

  def window(current_index: int) -> tuple[TrajectoryRow, ...]:
    return rows[current_index - DEFAULT_CONTRACT.current_index :][
      : DEFAULT_CONTRACT.window_steps
    ]

  # The window anchored on clip frame 0 is eligible; the one anchored on the
  # last padding row (reference frame -1) is not.
  assert WindowIndex._eligible(window(DEFAULT_CONTRACT.current_index), DEFAULT_CONTRACT)
  assert not WindowIndex._eligible(
    window(DEFAULT_CONTRACT.current_index - 1), DEFAULT_CONTRACT
  )
  # Rows start at tick 0, so the scan cannot even address a window that would
  # put a padding row on the current slot.
  assert scan.ticks.min() == 0
  assert scan.position(-1, DEFAULT_CONTRACT.window_steps) is None


def _window_rows_for_split(
  clip_frames: int, *, pose: str, seed: int = 0
) -> tuple[TrajectoryRow, ...]:
  rows = _segment_rows(clip_frames, seed=seed, pose=pose)
  anchor = DEFAULT_CONTRACT.current_index
  return rows[anchor : anchor + DEFAULT_CONTRACT.window_steps]


def test_both_initial_poses_of_a_clip_share_one_split_group() -> None:
  reference = _window_rows_for_split(60, pose="reference")
  standing = _window_rows_for_split(60, pose="standing")
  assert reference[0].group_key == standing[0].group_key == "motion:0:0"
  # The pair identity does carry the pose, so the two poses are separate pairs.
  assert reference[0].provenance["pair_id"] != standing[0].provenance["pair_id"]
  assignments = grouped_split(list(reference + standing), contract=DEFAULT_CONTRACT)
  assert assignments.split_for(reference[0].group_key) == assignments.split_for(
    standing[0].group_key
  )


def test_the_clean_ou_pair_identity_carries_the_initial_pose() -> None:
  frames = (("motion", 60),)
  from mjlab.tasks.tracking.diffusion.runtime_x2 import build_trials

  trials = build_trials(frames, seeds=(0,), run_id="run")
  assert {spec.initial_pose for spec in trials} == {"reference", "standing"}
  clean = {(spec.initial_pose, spec.phase): spec for spec in trials}
  reference = clean[("reference", "clean")]
  standing = clean[("standing", "clean")]
  assert reference.pair_id != standing.pair_id
  assert reference.initial_state_id != standing.initial_state_id
  # Clean and OU of one pose are one pair in one initial state.
  assert reference.pair_id == clean[("reference", "ou")].pair_id
  assert reference.initialization_fingerprint() == (
    clean[("reference", "ou")].initialization_fingerprint()
  )
  # An unperturbed attempt is a different initial state, so it can never be
  # admitted against the perturbed clean partner.
  unperturbed = TrialSpec(
    "run",
    "motion",
    0,
    0,
    pair_id=reference.pair_id,
    initial_state_id="init:motion:0:reference:reference",
    initial_state_mode="reference",
  )
  assert unperturbed.initialization_fingerprint() != (
    reference.initialization_fingerprint()
  )


def test_an_unknown_initial_pose_is_refused() -> None:
  import pytest

  with pytest.raises(ValueError, match="initial pose"):
    TrialSpec("run", "motion", 0, 0, initial_pose="floating")


def test_every_path_keys_a_window_on_its_anchor_not_its_first_row() -> None:
  """The split key of a window is its CURRENT row's group, in every consumer.

  The index builder and the statistics fit each derive a window's split group.
  They must agree, so both read the anchor row.  Reading the first row instead
  gives ``anchor - padding_steps``, which is a different group and makes the two
  passes select different windows.
  """
  from mjlab.tasks.tracking.diffusion.dataset import _segment_refs

  rows = _segment_rows(60)
  scan = segment_window_scan(rows, DEFAULT_CONTRACT)
  assignments = grouped_split(list(rows), contract=DEFAULT_CONTRACT)
  refs = _segment_refs(rows, assignments, DEFAULT_CONTRACT)

  starts = np.flatnonzero(scan.eligible_starts)
  assert len(refs) == len(starts)
  for ref, position in zip(refs, starts, strict=True):
    first = scan.rows[int(position)]
    anchor = scan.rows[int(position) + DEFAULT_CONTRACT.current_index]
    assert anchor.reference_frame == first.reference_frame + _PADDING
    # The emitted group is the anchor's, never the history row's.
    assert ref.group_key == anchor.group_key
    if first.reference_frame < 0:
      assert ref.group_key != first.group_key
