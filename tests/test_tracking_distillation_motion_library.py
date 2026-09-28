"""CPU tests for the immutable multi-motion reference library."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from tracking_distillation_fixtures import write_reference_clip

from mjlab.tasks.tracking.distillation.motion_library import (
  BodySelection,
  MotionClip,
  MotionClipSpec,
  MotionLibrary,
  MotionLibraryError,
)

CLIP_FRAMES = (2, 5)
JOINT_DIM = 3
SOURCE_BODIES = 4
TRACKED_INDICES = (0, 1)
SENTINEL_BASE = 1000.0


def clip_specs(
  root: Path, frames: tuple[int, ...] = CLIP_FRAMES
) -> list[MotionClipSpec]:
  """Two unequal clips whose values identify clip, frame, and body."""
  return [
    MotionClipSpec(
      teacher_id=f"tiny_{index:03d}",
      motion_file=write_reference_clip(
        root / f"clip_{index}.npz",
        frames=count,
        joint_dim=JOINT_DIM,
        bodies=SOURCE_BODIES,
        joint_base=SENTINEL_BASE * index,
        body_base=SENTINEL_BASE * index,
      ),
      expected_frames=count,
      expected_fps=50.0,
    )
    for index, count in enumerate(frames)
  ]


@pytest.fixture
def library(tmp_path: Path) -> MotionLibrary:
  return MotionLibrary.from_clips(clip_specs(tmp_path))


@pytest.fixture
def selected_library(library: MotionLibrary) -> MotionLibrary:
  return library.with_body_selection(
    BodySelection(
      indices=TRACKED_INDICES,
      source_body_count=SOURCE_BODIES,
      names=("pelvis", "torso_link"),
    )
  )


def source_array(path: Path, name: str, frames: torch.Tensor | slice) -> torch.Tensor:
  """Read a source NPZ array directly, independently of the library."""
  with np.load(path, allow_pickle=False) as data:
    values = torch.from_numpy(np.asarray(data[name]))
  return values[frames]


def tracked_source(
  library: MotionLibrary, motion_id: int, name: str, frame: int
) -> torch.Tensor:
  """Tracked-body source values for one row, read independently of the library."""
  index = torch.tensor([frame], dtype=torch.long)
  return source_array(library.clips[motion_id].motion_file, name, index)[
    0, list(TRACKED_INDICES)
  ]


def test_clip_identity_records_order_codes_and_source_digests(tmp_path: Path) -> None:
  specs = clip_specs(tmp_path)
  library = MotionLibrary.from_clips(specs)
  assert [
    (clip.motion_id, clip.teacher_id, clip.teacher_code) for clip in library.clips
  ] == [
    (0, "tiny_000", 0),
    (1, "tiny_001", 1),
  ]
  assert [clip.frames for clip in library.clips] == list(CLIP_FRAMES)
  assert [clip.fps for clip in library.clips] == [50.0, 50.0]
  assert [clip.storage_offset for clip in library.clips] == [0, CLIP_FRAMES[0]]
  assert library.num_motions == 2
  assert library.total_frames == sum(CLIP_FRAMES)
  assert library.source_body_count == SOURCE_BODIES
  assert library.joint_dim == JOINT_DIM
  assert library.device == torch.device("cpu")
  assert library.dtype == torch.float32
  assert library.body_selection is None
  for spec, clip in zip(specs, library.clips, strict=True):
    digest = hashlib.sha256(Path(spec.motion_file).read_bytes()).hexdigest()
    assert clip.source_hash == digest
    assert isinstance(clip, MotionClip)
  assert library.clip(1).teacher_id == "tiny_001"
  assert library.ordered_mapping() == library.clips


def test_mapping_digest_tracks_order_not_paths(tmp_path: Path) -> None:
  first = MotionLibrary.from_clips(clip_specs(tmp_path, (2, 5)))
  reordered = MotionLibrary.from_clips(
    [
      MotionClipSpec(
        teacher_id="tiny_001",
        motion_file=first.clips[1].motion_file,
        teacher_code=1,
      ),
      MotionClipSpec(
        teacher_id="tiny_000",
        motion_file=first.clips[0].motion_file,
        teacher_code=0,
      ),
    ]
  )
  assert reordered.mapping_digest() != first.mapping_digest()
  assert [clip.teacher_id for clip in reordered.clips] == ["tiny_001", "tiny_000"]
  assert [clip.motion_id for clip in reordered.clips] == [0, 1]
  # Same sources and order copied byte-for-byte elsewhere: same digest, so a
  # relocated asset is accepted by content digest rather than by path.
  relocated_root = tmp_path / "elsewhere"
  relocated_root.mkdir()
  relocated = MotionLibrary.from_clips(
    [
      MotionClipSpec(
        teacher_id=clip.teacher_id,
        motion_file=shutil.copy2(
          clip.motion_file, relocated_root / f"clip_{clip.motion_id}.npz"
        ),
        teacher_code=clip.teacher_code,
      )
      for clip in first.clips
    ]
  )
  assert relocated.mapping_digest() == first.mapping_digest()


def test_gather_reads_first_and_last_frames_of_each_clip(
  selected_library: MotionLibrary,
) -> None:
  for motion_id, frames in enumerate(CLIP_FRAMES):
    for frame in (0, frames - 1):
      index = torch.tensor([frame], dtype=torch.long)
      local = torch.tensor([motion_id], dtype=torch.long)
      path = selected_library.clips[motion_id].motion_file
      assert torch.equal(
        selected_library.joint_pos(local, index), source_array(path, "joint_pos", index)
      )
      assert torch.equal(
        selected_library.joint_vel(local, index), source_array(path, "joint_vel", index)
      )
      for name in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        assert torch.equal(
          getattr(selected_library, name)(local, index)[0],
          tracked_source(selected_library, motion_id, name, frame),
        )


def test_gather_mixes_and_permutes_row_ids(selected_library: MotionLibrary) -> None:
  motion_ids = torch.tensor([1, 0, 1, 0, 0], dtype=torch.long)
  frames = torch.tensor([4, 1, 0, 0, 1], dtype=torch.long)
  gathered = selected_library.joint_pos(motion_ids, frames)
  for row in range(motion_ids.numel()):
    motion_id = int(motion_ids[row])
    frame = int(frames[row])
    assert torch.equal(
      gathered[row],
      source_array(
        selected_library.clips[motion_id].motion_file,
        "joint_pos",
        torch.tensor([frame]),
      )[0],
    )
  body = selected_library.body_pos_w(motion_ids, frames)
  assert body.shape == (5, len(TRACKED_INDICES), 3)
  # The clips' distinct sentinel bases prove no row read the other clip.
  assert body[0, 0, 0] > SENTINEL_BASE
  assert body[1, 0, 0] < SENTINEL_BASE


def test_metadata_gathers_are_per_row(selected_library: MotionLibrary) -> None:
  motion_ids = torch.tensor([1, 0, 1], dtype=torch.long)
  assert selected_library.frame_counts_for(motion_ids).tolist() == [5, 2, 5]
  assert selected_library.teacher_codes_for(motion_ids).tolist() == [1, 0, 1]
  clamped = selected_library.clamp_local_frames(
    motion_ids, torch.tensor([7, 4, 5], dtype=torch.long)
  )
  assert clamped.tolist() == [4, 1, 4]
  with pytest.raises(MotionLibraryError, match="non-negative"):
    selected_library.clamp_local_frames(
      motion_ids, torch.tensor([-1, 0, 0], dtype=torch.long)
    )


def test_out_of_clip_and_invalid_indexes_are_rejected(
  selected_library: MotionLibrary,
) -> None:
  local = torch.tensor([0], dtype=torch.long)
  # Frame len(clip 0) is the first frame of clip 1 in flat storage: never read.
  with pytest.raises(MotionLibraryError, match="outside its clip"):
    selected_library.joint_pos(local, torch.tensor([CLIP_FRAMES[0]], dtype=torch.long))
  with pytest.raises(MotionLibraryError, match="non-negative"):
    selected_library.joint_pos(local, torch.tensor([-1], dtype=torch.long))
  with pytest.raises(MotionLibraryError, match="motion_ids must be in"):
    selected_library.joint_pos(torch.tensor([2]), torch.tensor([0]))
  with pytest.raises(MotionLibraryError, match="motion_ids must be in"):
    selected_library.joint_pos(torch.tensor([-1]), torch.tensor([0]))
  with pytest.raises(MotionLibraryError, match="disagrees"):
    selected_library.joint_pos(torch.tensor([0, 1]), torch.tensor([0]))
  with pytest.raises(MotionLibraryError, match="integer tensor"):
    selected_library.joint_pos(torch.tensor([0.0]), torch.tensor([0]))
  with pytest.raises(MotionLibraryError, match="one-dimensional"):
    selected_library.joint_pos(torch.zeros(1, 1, dtype=torch.long), local)
  with pytest.raises(MotionLibraryError, match="must be a torch.Tensor"):
    selected_library.joint_pos([0], local)  # type: ignore[arg-type]
  with pytest.raises(MotionLibraryError, match="outside the 2 loaded clips"):
    selected_library.clip(2)
  assert selected_library.flat_index(local, torch.tensor([0])).tolist() == [0]
  assert selected_library.flat_index(torch.tensor([1]), torch.tensor([0])).tolist() == [
    CLIP_FRAMES[0]
  ]


def test_body_queries_require_a_resolved_selection(library: MotionLibrary) -> None:
  with pytest.raises(MotionLibraryError, match="with_body_selection"):
    library.body_pos_w(torch.tensor([0]), torch.tensor([0]))
  with pytest.raises(MotionLibraryError, match="with_body_selection"):
    library.compare_body_references(0, {})


def test_body_selection_is_validated() -> None:
  with pytest.raises(MotionLibraryError, match="at least one"):
    BodySelection(indices=(), source_body_count=SOURCE_BODIES)
  with pytest.raises(MotionLibraryError, match="duplicate"):
    BodySelection(indices=(0, 0), source_body_count=SOURCE_BODIES)
  with pytest.raises(MotionLibraryError, match="outside the"):
    BodySelection(indices=(SOURCE_BODIES,), source_body_count=SOURCE_BODIES)
  with pytest.raises(MotionLibraryError, match="do not align"):
    BodySelection(indices=(0, 1), source_body_count=SOURCE_BODIES, names=("pelvis",))


def test_selection_rejects_a_different_source_body_count(tmp_path: Path) -> None:
  library = MotionLibrary.from_clips(clip_specs(tmp_path))
  with pytest.raises(MotionLibraryError, match="resolved against"):
    library.with_body_selection(
      BodySelection(indices=(0,), source_body_count=SOURCE_BODIES + 1)
    )


def test_returned_tensors_are_copies(selected_library: MotionLibrary) -> None:
  motion_ids = torch.tensor([1, 0], dtype=torch.long)
  frames = torch.tensor([4, 1], dtype=torch.long)
  before = selected_library.joint_pos(motion_ids, frames).clone()
  mutated = selected_library.joint_pos(motion_ids, frames)
  mutated.zero_()
  assert torch.equal(selected_library.joint_pos(motion_ids, frames), before)
  body = selected_library.body_pos_w(motion_ids, frames)
  body.fill_(123.0)
  assert not torch.equal(selected_library.body_pos_w(motion_ids, frames), body)
  counts = selected_library.frame_counts_for(motion_ids)
  counts.zero_()
  assert selected_library.frame_counts_for(motion_ids).tolist() == list(
    [CLIP_FRAMES[1], CLIP_FRAMES[0]]
  )
  assert selected_library.body_selection is not None
  assert selected_library.body_selection.indices == TRACKED_INDICES
  assert selected_library.clips[1].body_indices == TRACKED_INDICES


def test_source_files_are_never_written(selected_library: MotionLibrary) -> None:
  paths = [clip.motion_file for clip in selected_library.clips]
  before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
  selected_library.joint_pos(torch.tensor([0, 1]), torch.tensor([0, 4]))
  selected_library.body_quat_w(torch.tensor([1]), torch.tensor([2]))
  assert {
    path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
  } == before


def test_compare_body_references_reports_export_disagreement(
  selected_library: MotionLibrary,
) -> None:
  clip = selected_library.clips[0]
  expected_arrays = {
    name: source_array(clip.motion_file, name, slice(None))[
      :, list(TRACKED_INDICES)
    ].numpy()
    for name in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
  }
  errors = selected_library.compare_body_references(0, expected_arrays)
  assert errors == {
    "body_pos_w": 0.0,
    "body_quat_w": 0.0,
    "body_lin_vel_w": 0.0,
    "body_ang_vel_w": 0.0,
  }

  shifted = dict(expected_arrays)
  shifted["body_pos_w"] = expected_arrays["body_pos_w"] + 0.5
  errors = selected_library.compare_body_references(0, shifted)
  assert errors["body_pos_w"] == pytest.approx(0.5)
  assert errors["body_quat_w"] == 0.0

  wrong_shape = dict(expected_arrays)
  wrong_shape["body_pos_w"] = expected_arrays["body_pos_w"][:, :1]
  with pytest.raises(MotionLibraryError, match="disagrees with the export shape"):
    selected_library.compare_body_references(0, wrong_shape)
  with pytest.raises(MotionLibraryError, match="no 'body_quat_w' array"):
    selected_library.compare_body_references(0, {"body_pos_w": np.zeros((2, 2, 3))})


def test_malformed_clips_are_rejected(tmp_path: Path) -> None:
  path = write_reference_clip(tmp_path / "clip.npz", frames=3)
  with np.load(path, allow_pickle=False) as data:
    arrays = {name: data[name] for name in data.files}
  good = MotionClipSpec(teacher_id="tiny_000", motion_file=path)

  with pytest.raises(MotionLibraryError, match="at least one clip"):
    MotionLibrary.from_clips([])
  with pytest.raises(MotionLibraryError, match="non-empty teacher id"):
    MotionLibrary.from_clips([MotionClipSpec(teacher_id="", motion_file=path)])
  with pytest.raises(MotionLibraryError, match="duplicate teacher id"):
    MotionLibrary.from_clips([good, good])
  with pytest.raises(MotionLibraryError, match="duplicate teacher code"):
    MotionLibrary.from_clips(
      [good, MotionClipSpec(teacher_id="tiny_001", motion_file=path, teacher_code=0)]
    )
  with pytest.raises(MotionLibraryError, match="non-negative integer teacher code"):
    MotionLibrary.from_clips(
      [MotionClipSpec(teacher_id="a", motion_file=path, teacher_code=-1)]
    )
  with pytest.raises(MotionLibraryError, match="not found"):
    MotionLibrary.from_clips(
      [MotionClipSpec(teacher_id="a", motion_file=tmp_path / "missing.npz")]
    )
  with pytest.raises(MotionLibraryError, match="frames") as error:
    MotionLibrary.from_clips(
      [MotionClipSpec(teacher_id="a", motion_file=path, expected_frames=9)]
    )
  assert "declares 9" in str(error.value)
  with pytest.raises(MotionLibraryError, match="is 50 Hz but the teacher declares"):
    MotionLibrary.from_clips(
      [MotionClipSpec(teacher_id="a", motion_file=path, expected_fps=30.0)]
    )

  truncated = {name: values[:0] for name, values in arrays.items()}
  truncated["fps"] = arrays["fps"]
  empty = tmp_path / "empty.npz"
  np.savez(empty, **truncated)
  with pytest.raises(MotionLibraryError, match="no frames"):
    MotionLibrary.from_clips([MotionClipSpec(teacher_id="a", motion_file=empty)])

  missing = dict(arrays)
  del missing["joint_vel"]
  missing_path = tmp_path / "missing_array.npz"
  np.savez(missing_path, **missing)
  with pytest.raises(MotionLibraryError, match=r"missing arrays \['joint_vel'\]"):
    MotionLibrary.from_clips([MotionClipSpec(teacher_id="a", motion_file=missing_path)])

  no_fps = dict(arrays)
  del no_fps["fps"]
  no_fps_path = tmp_path / "no_fps.npz"
  np.savez(no_fps_path, **no_fps)
  with pytest.raises(MotionLibraryError, match="has no fps array"):
    MotionLibrary.from_clips([MotionClipSpec(teacher_id="a", motion_file=no_fps_path)])

  bad_fps = dict(arrays)
  bad_fps["fps"] = np.array([0.0], dtype=np.float32)
  bad_fps_path = tmp_path / "bad_fps.npz"
  np.savez(bad_fps_path, **bad_fps)
  with pytest.raises(MotionLibraryError, match="non-positive fps"):
    MotionLibrary.from_clips([MotionClipSpec(teacher_id="a", motion_file=bad_fps_path)])

  nonfinite = dict(arrays)
  nonfinite["joint_pos"] = arrays["joint_pos"].copy()
  nonfinite["joint_pos"][0, 0] = np.nan
  nonfinite_path = tmp_path / "nonfinite.npz"
  np.savez(nonfinite_path, **nonfinite)
  with pytest.raises(MotionLibraryError, match="non-finite values"):
    MotionLibrary.from_clips(
      [MotionClipSpec(teacher_id="a", motion_file=nonfinite_path)]
    )

  ragged = dict(arrays)
  ragged["joint_vel"] = arrays["joint_vel"][:-1]
  ragged_path = tmp_path / "ragged.npz"
  np.savez(ragged_path, **ragged)
  with pytest.raises(MotionLibraryError, match="joint_vel carries"):
    MotionLibrary.from_clips([MotionClipSpec(teacher_id="a", motion_file=ragged_path)])

  shaped = dict(arrays)
  shaped["body_pos_w"] = arrays["body_pos_w"][:, :1]
  shaped_path = tmp_path / "shaped.npz"
  np.savez(shaped_path, **shaped)
  with pytest.raises(MotionLibraryError, match="body_quat_w shape"):
    MotionLibrary.from_clips([MotionClipSpec(teacher_id="a", motion_file=shaped_path)])

  other = write_reference_clip(tmp_path / "other.npz", frames=3, joint_dim=4)
  with pytest.raises(MotionLibraryError, match="earlier clips carry"):
    MotionLibrary.from_clips([good, MotionClipSpec(teacher_id="b", motion_file=other)])


@pytest.mark.parametrize("device", ["cpu", "cpu:0"])
def test_device_alias_matches_the_allocated_reference_tensors(
  tmp_path: Path, device: str
) -> None:
  library = MotionLibrary.from_clips(clip_specs(tmp_path), device=device)
  ids = torch.tensor([0, 1], dtype=torch.long)
  assert library.device == ids.device
  assert library.frame_counts_for(ids).tolist() == list(CLIP_FRAMES)
  assert library.joint_pos(ids, torch.tensor([1, 4])).shape == (2, JOINT_DIM)


def test_repr_describes_clips_and_selection(selected_library: MotionLibrary) -> None:
  text = repr(selected_library)
  assert "tiny_000" in text and "tiny_001" in text
  assert "selection=(0, 1)" in text


def test_cpu_copy_moves_storage_to_the_host_and_keeps_identity() -> None:
  """A library that crosses a process boundary must carry host storage.

  A sharded parent builds its replay and cohort identity from a worker's
  library, so the worker sends a copy: sending the live library would try to
  move device tensors through a queue, and a reduced stand-in would not satisfy
  the identity builder's type.
  """
  import torch

  from mjlab.tasks.tracking.distillation.motion_library import (
    BodySelection,
    MotionClip,
    MotionLibrary,
  )

  clips = (
    MotionClip(
      motion_id=0,
      teacher_id="tennis_000",
      teacher_code=0,
      motion_file=Path("a.npz"),
      source_hash="deadbeef",
      frames=4,
      fps=50.0,
      storage_offset=0,
      source_body_count=2,
    ),
  )
  library = MotionLibrary(
    clips=clips,
    device=torch.device("cpu"),
    dtype=torch.float32,
    offsets=torch.tensor([0, 4], dtype=torch.long),
    frame_counts=torch.tensor([4], dtype=torch.long),
    teacher_codes=torch.tensor([0], dtype=torch.long),
    joint_pos=torch.zeros(4, 2, 3),
    joint_vel=torch.ones(4, 2, 3),
    body_pos_w=torch.zeros(4, 2, 3),
    body_quat_w=torch.zeros(4, 2, 4),
    body_lin_vel_w=torch.zeros(4, 2, 3),
    body_ang_vel_w=torch.zeros(4, 2, 3),
    selection=BodySelection(indices=(0, 1), source_body_count=2),
  )

  copy = library.cpu_copy()

  assert isinstance(copy, MotionLibrary)
  assert copy.clips == library.clips
  assert copy.body_selection == library.body_selection
  assert copy.source_body_count == library.source_body_count
  assert copy.device.type == "cpu"
  # Equal values, independent host storage: the copy is what crosses the
  # boundary.  The gather methods are the public API, so this asserts on the
  # storage the copy is about.
  assert copy._joint_vel.device.type == "cpu"
  assert torch.equal(copy._joint_vel, library._joint_vel)
  assert copy._joint_vel.data_ptr() != library._joint_vel.data_ptr()
