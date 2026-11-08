"""CPU tests for clip-sharded distillation cohorts.

A clip-sharded run gives each collection worker a disjoint slice of one cohort's
clips, so a cohort too large for one card is built one slice at a time.  That
touches four seams: the clip id a library carries, the balanced shard plan, the
parent's metadata-only union check, and the shard-scoped teacher bank.  These
tests cover all four without a simulator, and they pin that every one of them is
inert unless a shard plan is supplied.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from test_tracking_distillation_multi_motion import (
  ROBOT_BODIES,
  TRACKED_BODIES,
  make_command,
)
from tracking_distillation_fixtures import build_tiny_cohort, write_reference_clip

from mjlab.scripts import distill
from mjlab.tasks.tracking.distillation.cohort_contract import ResetProvenance
from mjlab.tasks.tracking.distillation.cohort_setup import CohortSetup
from mjlab.tasks.tracking.distillation.config import (
  CohortContract,
  DistillationError,
  load_manifest,
  resolve_cohort,
)
from mjlab.tasks.tracking.distillation.environment import RuntimeSeedProvenance
from mjlab.tasks.tracking.distillation.motion_library import (
  BodySelection,
  ClipMapping,
  MotionClip,
  MotionClipSpec,
  MotionLibrary,
  MotionLibraryError,
)
from mjlab.tasks.tracking.distillation.multi_motion import (
  MotionSlotAllocation,
  plan_multi_motion,
)
from mjlab.tasks.tracking.distillation.reset_policy import ResetPolicy
from mjlab.tasks.tracking.distillation.teachers import (
  TeacherBankError,
  build_cohort_teacher_bank,
)
from mjlab.tasks.tracking.distillation.vae_config import DEFAULT_SCHEMA
from mjlab.tasks.tracking.distillation.worker import (
  WorkerError,
  clip_shard_plan,
  clip_shard_worker_specs,
)

TEACHER_IDS = ("tiny_000", "tiny_001", "tiny_002", "tiny_003", "tiny_004")
BODY_NAMES = ("pelvis", "torso_link")
DEVICES = ("cuda:1", "cuda:2", "cuda:3")
BASE_SEED = 5
STANDING_MIXTURE = ResetPolicy(kind="standing-mixture")


@pytest.fixture(scope="module")
def cohort(tmp_path_factory: pytest.TempPathFactory) -> CohortContract:
  root = tmp_path_factory.mktemp("clip-shards")
  with (
    contextlib.redirect_stdout(io.StringIO()),
    contextlib.redirect_stderr(io.StringIO()),
  ):
    return resolve_cohort(
      load_manifest(build_tiny_cohort(root, teacher_ids=TEACHER_IDS).manifest, root)
    )


def shard_setup(cohort: CohortContract, teacher_ids: tuple[str, ...] = TEACHER_IDS):
  return CohortSetup(
    manifest=cohort.manifest.path,
    repo_root=cohort.manifest.repo_root,
    teacher_ids=teacher_ids,
    task_id=None,
    phase_policy="uniform",
    reset_policy=ResetPolicy(),
    base_seed=BASE_SEED,
  )


# The plan.


def test_plan_covers_every_clip_once_with_balanced_contiguous_slices(
  cohort: CohortContract,
) -> None:
  """Every clip is built by exactly one worker, in cohort order.

  The partition is what the parent's replay is split by and what a resume
  reproduces, so it has to be deterministic: contiguous slices in cohort order,
  differing in size by at most one.  A clip built twice would be counted twice;
  a clip built by nobody would collect no rows while the run still looked
  complete.
  """
  shards = clip_shard_plan(cohort, TEACHER_IDS, DEVICES, envs_per_motion=3)

  assert [shard.worker_index for shard in shards] == [0, 1, 2]
  assert [shard.device for shard in shards] == list(DEVICES)
  built = [teacher_id for shard in shards for teacher_id in shard.teacher_ids]
  assert built == list(TEACHER_IDS), "not the cohort order, or not a cover"
  sizes = [len(shard.teacher_ids) for shard in shards]
  assert sizes == [2, 2, 1]
  assert max(sizes) - min(sizes) <= 1
  # Clip ids are the cohort-wide positions, not each slice's own numbering.
  assert [motion_id for shard in shards for motion_id in shard.motion_ids] == [
    0,
    1,
    2,
    3,
    4,
  ]
  assert shards[1].motion_ids == (2, 3)


def test_plan_sizes_each_shard_from_envs_per_motion(cohort: CohortContract) -> None:
  """Each shard asks for one row budget per clip it owns, like the whole cohort.

  Shard totals therefore differ, which only this layout allows: the run is sized
  ``clips times envs-per-motion`` and the uneven split is the point.
  """
  shards = clip_shard_plan(cohort, TEACHER_IDS, DEVICES, envs_per_motion=4)

  for shard in shards:
    assert shard.num_envs == 4 * len(shard.teacher_ids)
  assert {shard.num_envs for shard in shards} == {8, 4}
  assert sum(shard.num_envs for shard in shards) == 4 * len(TEACHER_IDS)


def test_plan_refuses_impossible_splits(cohort: CohortContract) -> None:
  """Every refusal happens before an environment exists.

  More shards than clips, a repeated device, a repeated clip and a zero row
  budget cannot produce one row per clip on one card per slice, so each is
  refused instead of building a shard that cannot collect or that pays the
  per-member cost twice on one card.
  """
  with pytest.raises(ValueError, match="at least one teacher id"):
    clip_shard_plan(cohort, (), DEVICES, envs_per_motion=1)
  with pytest.raises(WorkerError, match="cannot be split over"):
    clip_shard_plan(cohort, ("tiny_000", "tiny_001"), DEVICES, envs_per_motion=1)
  with pytest.raises(WorkerError, match="envs_per_motion must be a positive"):
    clip_shard_plan(cohort, TEACHER_IDS, DEVICES, envs_per_motion=0)
  with pytest.raises(WorkerError, match="repeats a device"):
    clip_shard_plan(cohort, TEACHER_IDS, ("cuda:1", "cuda:1"), envs_per_motion=1)
  with pytest.raises(ValueError, match="duplicate teacher ids"):
    clip_shard_plan(cohort, ("tiny_000", "tiny_000"), ("cuda:1",), envs_per_motion=1)
  with pytest.raises(ValueError, match="cohort has no teacher"):
    clip_shard_plan(cohort, ("not-a-member",), ("cuda:1",), envs_per_motion=1)


def test_worker_specs_carry_each_shards_clips_and_ids(cohort: CohortContract) -> None:
  """A worker receives its slice by value, including the global clip ids.

  The recipe travels with the spec, so a worker cannot rebuild the whole
  selection from a command line or renumber its clips from zero.
  """
  setup = shard_setup(cohort)
  shards = clip_shard_plan(cohort, TEACHER_IDS, DEVICES, envs_per_motion=2)

  specs = clip_shard_worker_specs(setup, shards)

  assert [spec.num_envs for spec in specs] == [4, 4, 2]
  assert [spec.setup.teacher_ids for spec in specs] == [
    shard.teacher_ids for shard in shards
  ]
  assert specs[1].setup.motion_ids == (("tiny_002", 2), ("tiny_003", 3))
  # Only the selection differs; the rest of the recipe is carried unchanged.
  assert specs[0].setup.base_seed == setup.base_seed
  assert specs[0].setup.phase_policy == setup.phase_policy
  with pytest.raises(ValueError, match="do not align"):
    replace(setup, motion_ids=(("tiny_000", 0),))


# Global clip identity.


def clip_specs(root: Path, motion_ids: tuple[int | None, ...]) -> list[MotionClipSpec]:
  """Three short clips whose ids are given, with distinguishable contents."""
  return [
    MotionClipSpec(
      teacher_id=f"tiny_{index:03d}",
      motion_file=write_reference_clip(
        root / f"clip_{index}.npz",
        frames=2 + index,
        joint_dim=3,
        bodies=4,
        joint_base=1000.0 * index,
        body_base=1000.0 * index,
      ),
      motion_id=motion_id,
    )
    for index, motion_id in enumerate(motion_ids)
  ]


def test_a_library_honours_explicit_clip_ids(tmp_path: Path) -> None:
  """A slice keeps cohort-wide ids, so its rows label themselves globally.

  Every per-row lookup translates the id to its own storage slot, which is what
  lets a shard read only its own tensors while the rows it collects carry the
  ids the parent's replay and identity use.
  """
  library = MotionLibrary.from_clips(clip_specs(tmp_path, (7, 8, 9)))

  assert library.motion_ids == (7, 8, 9)
  assert [clip.motion_id for clip in library.clips] == [7, 8, 9]
  ids = torch.tensor([9, 7], dtype=torch.long)
  frames = torch.tensor([1, 0], dtype=torch.long)
  # Clip lengths are 4 and 2 frames, and storage is ordered by the slice.
  assert library.frame_counts_for(ids).tolist() == [4, 2]
  assert library.flat_index(ids, frames).tolist() == [6, 0]
  assert library.clip(9).frames == 4
  assert library.joint_pos(ids, frames).shape == (2, 3)
  with pytest.raises(MotionLibraryError, match="one of the 3 clip ids"):
    library.slots_for(torch.tensor([0], dtype=torch.long))
  with pytest.raises(MotionLibraryError, match="outside the 3 loaded clips"):
    library.clip(0)


def test_per_step_lookups_gather_without_validating_clip_ids(tmp_path: Path) -> None:
  """Clip-id validation happens where ids enter, not on every simulation step.

  ``slots_for`` is the validating seam, and it runs at construction and in
  ``select_motion_ids``, where a host Boolean conversion costs nothing.  The
  per-row metadata lookups run inside ``snapshot()`` (twice per environment
  step) and in ``lookahead_command``, so they translate and gather on the device
  only: an id this library does not hold is clamped into range there, because a
  sync on that path is what the slice translation must not add.
  """
  library = MotionLibrary.from_clips(clip_specs(tmp_path, (7, 8, 9)))

  with pytest.raises(MotionLibraryError, match="one of the 3 clip ids"):
    library.slots_for(torch.tensor([0], dtype=torch.long))
  # Clamped to slot 0, which is clip 7: no raise, and no host conversion.
  assert library.frame_counts_for(torch.tensor([0], dtype=torch.long)).tolist() == [2]
  assert library.teacher_codes_for(torch.tensor([0], dtype=torch.long)).tolist() == [0]
  assert library.clamp_local_frames(
    torch.tensor([0], dtype=torch.long), torch.tensor([5], dtype=torch.long)
  ).tolist() == [1]


def test_a_library_refuses_duplicate_and_negative_clip_ids(tmp_path: Path) -> None:
  """Two clips behind one row label, or a negative label, cannot be recorded.

  A duplicate id would make a row resolve to a clip the caller never selected,
  and a negative id cannot be written to a replay partition, so both are refused
  when the library loads rather than when the first row is collected.
  """
  with pytest.raises(MotionLibraryError, match="duplicate motion id 3"):
    MotionLibrary.from_clips(clip_specs(tmp_path, (3, 3)))
  with pytest.raises(MotionLibraryError, match="non-negative motion id"):
    MotionLibrary.from_clips(clip_specs(tmp_path, (-1, 3)))
  with pytest.raises(MotionLibraryError, match="integer motion id"):
    MotionLibrary.from_clips(clip_specs(tmp_path, (True, 3)))


def test_plan_labels_rows_with_the_ids_it_was_given(cohort: CohortContract) -> None:
  """The plan honours a clip-id mapping and refuses one that does not align.

  A mapping that omits a selected clip, names an unselected one, or carries a
  negative id means the caller's plan and its selection disagree; rows labelled
  from either would reach the wrong replay partition.
  """
  selected = TEACHER_IDS[:3]
  mapping = {teacher_id: 10 + index for index, teacher_id in enumerate(selected)}
  plan = plan_multi_motion(
    cohort, selected, 6, phase_policy="uniform", motion_ids=mapping
  )
  assert [clip.motion_id for clip in plan.library.clips] == [10, 11, 12]
  # The allocation itself stays positional: it describes this build's selection.
  assert set(plan.slots.row_motion_ids) == {0, 1, 2}

  with pytest.raises(ValueError, match="no entry for selected teacher"):
    plan_multi_motion(
      cohort, TEACHER_IDS[:3], 6, motion_ids={"tiny_000": 0, "tiny_001": 1}
    )
  with pytest.raises(ValueError, match="did not select"):
    plan_multi_motion(
      cohort,
      TEACHER_IDS[:2],
      6,
      motion_ids={"tiny_000": 0, "tiny_001": 1, "tiny_004": 2},
    )
  with pytest.raises(ValueError, match="non-negative"):
    plan_multi_motion(
      cohort, TEACHER_IDS[:2], 6, motion_ids={"tiny_000": -1, "tiny_001": 1}
    )


def test_a_shard_command_labels_rows_with_global_codes_and_ids(
  cohort: CohortContract,
) -> None:
  """Rows carry cohort-wide clip ids and teacher codes, and read their own clip.

  This is the whole point of the clip id: the parent partitions one replay by
  those ids, so a shard that renumbered them would file its rows under another
  clip while every counter still looked right.
  """
  # A three-clip selection over two cards: worker 1 owns the last two clips,
  # whose cohort-wide ids are 1 and 2 while their teacher codes are 2 and 3.
  shards = clip_shard_plan(
    cohort, TEACHER_IDS[1:5], ("cuda:1", "cuda:2"), envs_per_motion=2
  )
  shard = shards[1]
  assert shard.teacher_ids == ("tiny_003", "tiny_004")
  assert shard.motion_ids == (2, 3)
  plan = plan_multi_motion(
    cohort,
    shard.teacher_ids,
    shard.num_envs,
    motion_ids=dict(zip(shard.teacher_ids, shard.motion_ids, strict=True)),
  )
  bank = build_cohort_teacher_bank(cohort, teacher_ids=shard.teacher_ids)
  body_names = tuple(cohort.body_names)
  library = plan.library.with_body_selection(
    BodySelection(
      indices=tuple(range(len(body_names))),
      source_body_count=plan.library.source_body_count,
      names=body_names,
    )
  )
  command, _env = make_command(
    library,
    plan.slots,
    num_envs=shard.num_envs,
    robot_bodies=body_names,
    body_names=body_names,
  )

  assert set(command.motion_ids.tolist()) == {2, 3}
  assert set(command.teacher_codes.tolist()) == {
    bank.code("tiny_003"),
    bank.code("tiny_004"),
  }
  assert set(command.teacher_codes.tolist()) == {3, 4}
  for row, motion_id in enumerate(command.motion_ids.tolist()):
    clip = library.clip(int(motion_id))
    assert int(command.row_lengths[row]) == clip.frames
    assert int(command.teacher_codes[row]) == clip.teacher_code
  # Clip ids are the run's, not this slice's numbering.
  assert set(command.motion_ids.tolist()) != {0, 1}
  # A row never resolves to a clip the shard does not hold.
  with pytest.raises(ValueError, match="one of the 2 clip ids"):
    command.select_motion_ids(
      torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)
    )


# The shard-scoped teacher bank.


def registered_teachers(bank) -> set[str]:
  """Names of the teacher modules the bank actually registers."""
  return set(dict(bank.named_children()))


def test_a_subset_bank_holds_only_its_modules_and_refuses_other_codes(
  cohort: CohortContract,
) -> None:
  """A shard's bank registers its slice but keeps the cohort's code table.

  Registering only the slice is what removes the measured per-member cost from
  the other shards' cards; keeping the code table cohort-wide is what lets a
  shard label rows exactly as the parent and the single-process path do.  A code
  it does not hold must raise, not route to a neighbour or return zeros.
  """
  bank = build_cohort_teacher_bank(cohort, teacher_ids=("tiny_001", "tiny_003"))

  assert bank.teacher_ids == TEACHER_IDS, "the code order must stay cohort-wide"
  assert bank.held_codes == frozenset({1, 3})
  assert bank.code("tiny_003") == 3
  assert bank.teacher(3).teacher_id == "tiny_003"
  assert registered_teachers(bank) == {"teacher_1", "teacher_3"}

  with pytest.raises(TeacherBankError, match="does not hold code 0"):
    bank.teacher(0)
  with pytest.raises(TeacherBankError, match="cannot label routing code 2"):
    bank.label(torch.tensor([1, 2], dtype=torch.long), torch.zeros(2, bank.obs_dim))
  # The codes it does hold are labelled exactly like the whole-cohort bank.
  whole = build_cohort_teacher_bank(cohort)
  codes = torch.tensor([1, 3], dtype=torch.long)
  observations = torch.zeros(2, bank.obs_dim)
  assert torch.equal(bank.label(codes, observations), whole.label(codes, observations))


def test_a_whole_cohort_bank_registers_every_member(cohort: CohortContract) -> None:
  """The default path is unchanged: every member, codes by manifest position.

  A build that owns the whole selection keeps the historical bank, so its codes
  and module names are what every existing checkpoint and report expects.
  """
  bank = build_cohort_teacher_bank(cohort)

  assert bank.teacher_ids == TEACHER_IDS
  assert bank.held_codes == frozenset(range(len(TEACHER_IDS)))
  assert [bank.code(teacher_id) for teacher_id in TEACHER_IDS] == list(range(5))
  assert [bank.teacher(code).teacher_id for code in range(5)] == list(TEACHER_IDS)
  assert registered_teachers(bank) == {f"teacher_{code}" for code in range(5)}


# The parent's union contract.


@dataclass(frozen=True)
class _ShardAsset:
  """Stand-in for the compiled-asset audit, with its per-clip evidence."""

  robot_bodies: tuple[str, ...]
  tracked_bodies: tuple[str, ...]
  clips: tuple[SimpleNamespace, ...]


@dataclass(frozen=True)
class _ShardAudit:
  """Frozen stand-in for the live audit, shaped so ``replace`` applies.

  The merge replaces the slots and the mapping digest of the record it returns,
  so the stand-in has to be the same kind of object the real one is.
  """

  task_id: str
  teacher_ids: tuple[str, ...]
  joint_names: tuple[str, ...]
  control_period_s: float
  actor_terms: tuple[str, ...]
  asset: _ShardAsset
  slots: MotionSlotAllocation
  mapping_digest: str
  phase_policy: str
  additional_gravity_policy: str
  semantic_overrides: tuple[str, ...]
  seed_provenance: RuntimeSeedProvenance | None
  reset_policy: ResetPolicy


def mint_clip(motion_id: int, teacher_id: str, *, frames: int = 4) -> MotionClip:
  """One clip record as a worker's library reports it."""
  return MotionClip(
    motion_id=motion_id,
    teacher_id=teacher_id,
    teacher_code=motion_id,
    motion_file=Path(f"{teacher_id}.npz"),
    source_hash=f"hash-{teacher_id}",
    frames=frames,
    fps=50.0,
    storage_offset=0,
    source_body_count=len(BODY_NAMES),
  )


def worker_library(clips: tuple[MotionClip, ...]) -> MotionLibrary:
  """The reference library a worker of this slice would report, storage and all.

  Built without clip files because the parent never reads a clip's contents: the
  identity and the replay are decided by the clip records alone, which is the
  property these tests pin.  The body selection is resolved, as a live
  environment's library is.
  """
  offsets: list[int] = []
  offset = 0
  for clip in clips:
    offsets.append(offset)
    offset += clip.frames
  bodies = len(BODY_NAMES)
  scalars = torch.zeros(offset, 1)
  body_vectors = torch.zeros(offset, bodies, 3)
  library = MotionLibrary(
    clips=clips,
    device=torch.device("cpu"),
    dtype=torch.float32,
    offsets=torch.tensor(offsets, dtype=torch.long),
    frame_counts=torch.tensor([clip.frames for clip in clips], dtype=torch.long),
    teacher_codes=torch.tensor([clip.teacher_code for clip in clips], dtype=torch.long),
    joint_pos=scalars,
    joint_vel=scalars.clone(),
    body_pos_w=body_vectors,
    body_quat_w=torch.zeros(offset, bodies, 4),
    body_lin_vel_w=body_vectors.clone(),
    body_ang_vel_w=body_vectors.clone(),
  )
  return library.with_body_selection(
    BodySelection(
      indices=tuple(range(bodies)), source_body_count=bodies, names=BODY_NAMES
    )
  )


def shard_description(
  worker_index: int,
  teacher_ids: tuple[str, ...],
  motion_ids: tuple[int, ...],
  *,
  clips: tuple[MotionClip, ...] | None = None,
  schema: Any = "schema",
  routing: dict[int, int] | None = None,
) -> SimpleNamespace:
  """One worker's describe reply for a clip-sharded run, as the parent reads it."""
  built = (
    clips
    if clips is not None
    else tuple(
      mint_clip(motion_id, teacher_id)
      for teacher_id, motion_id in zip(teacher_ids, motion_ids, strict=True)
    )
  )
  derived = BASE_SEED + worker_index * 1_000_003
  return SimpleNamespace(
    worker_index=worker_index,
    num_envs=2 * len(teacher_ids),
    schema=schema,
    library=worker_library(built),
    audit=_ShardAudit(
      task_id="Mjlab-Tracking-Flat-AgiBot-X2",
      teacher_ids=teacher_ids,
      joint_names=("joint_a", "joint_b"),
      control_period_s=0.02,
      actor_terms=("base_lin_vel",),
      asset=_ShardAsset(
        robot_bodies=ROBOT_BODIES,
        tracked_bodies=TRACKED_BODIES,
        clips=tuple(SimpleNamespace(motion_id=motion_id) for motion_id in motion_ids),
      ),
      slots=MotionSlotAllocation(
        weights=tuple(1.0 for _ in teacher_ids),
        counts=tuple(2 for _ in teacher_ids),
        teacher_ids=teacher_ids,
        row_motion_ids=tuple(
          index for index in range(len(teacher_ids)) for _ in range(2)
        ),
      ),
      mapping_digest="mapping",
      phase_policy="uniform",
      additional_gravity_policy="none",
      semantic_overrides=(),
      seed_provenance=RuntimeSeedProvenance(
        requested_seed=derived,
        effective_seed=derived,
        applied_before_construction=True,
      ),
      reset_policy=ResetPolicy(),
    ),
    motion_teacher_codes=(
      routing
      if routing is not None
      else {clip.motion_id: clip.teacher_code for clip in built}
    ),
    sampling_mode="uniform",
    reset_policy_enabled=False,
    reset_provenance=None,
  )


def shard_reset_provenance(
  windows: tuple[int, ...],
  *,
  policy: ResetPolicy = STANDING_MIXTURE,
) -> ResetProvenance:
  """One shard's live reset record, as ``reset_provenance_from_adapter`` writes it.

  ``windows`` is the per-clip window set of the slice that shard owns, which is
  where the two shards of one cohort legitimately differ.
  """
  return ResetProvenance(
    reset_policy=policy,
    effective_windows=windows,
    boundary_semantics={
      "full_reset": "sample standing/reference once per affected row",
      "timer_resample": "reference-state teleport; no standing mixture",
      "reference_wrap": "reference-state teleport; no standing mixture",
      "reset_to_frame": "explicit reference-state teleport; no standing mixture",
    },
    standing_pose={
      "joint_position": [0.0, 0.0],
      "root_height": 0.9,
      "yaw_alignment": "upright yaw from selected reference anchor quaternion",
      "joint_order": list(BODY_NAMES),
    },
    perturbations={
      "pose_range": [0.0] * 12,
      "velocity_range": [0.0] * 12,
      "joint_position_range": [-0.05, 0.05],
    },
    provenance={
      "command_type": "MultiMotionCommand",
      "entity_name": "robot",
      "standing_pose_source": "robot.data.default_joint_pos/default_root_state",
      "yaw_source": "reference anchor quaternion",
    },
  )


def with_reset_provenance(
  descriptions: Any, *, policy: ResetPolicy = STANDING_MIXTURE
) -> Any:
  """Give each shard a live reset record whose windows are its own slice's.

  One window per clip the shard built, which is what a live shard reports: the
  windows are a property of the clip lengths it holds, not of the cohort.
  """
  for description in descriptions:
    description.reset_policy_enabled = True
    description.reset_provenance = shard_reset_provenance(
      (4,) * len(description.library.clips), policy=policy
    )
  return descriptions


def descriptions_for(cohort: CohortContract, shards, **overrides) -> tuple[Any, ...]:
  return tuple(
    shard_description(
      shard.worker_index,
      shard.teacher_ids,
      shard.motion_ids,
      **overrides,
    )
    for shard in shards
  )


def test_the_union_contract_merges_disjoint_shards(cohort: CohortContract) -> None:
  """Disjoint slices merge into one cohort, in cohort order.

  The parent's replay partitions by clip id and its identity orders members by
  it, so the merge has to hand back one record ordered by global id, with the
  per-clip weights the workers allocated rows with.
  """
  shards = clip_shard_plan(cohort, TEACHER_IDS, DEVICES, envs_per_motion=2)
  descriptions = descriptions_for(cohort, shards)

  merged = distill._require_one_shard_contract(
    descriptions, shard_setup(cohort), clip_shards=shards
  )

  assert isinstance(merged.library, ClipMapping)
  assert [clip.motion_id for clip in merged.clips] == [0, 1, 2, 3, 4]
  assert [clip.teacher_id for clip in merged.clips] == list(TEACHER_IDS)
  assert [clip.frames for clip in merged.clips] == [4, 4, 4, 4, 4]
  # Storage placement is recomputed for the merged cohort, so the stand-in
  # records the same offsets a single-process library would have.
  assert [clip.storage_offset for clip in merged.clips] == [0, 4, 8, 12, 16]
  assert merged.replay_weights() == {motion_id: 1.0 for motion_id in range(5)}
  assert merged.frame_counts() == {motion_id: 4 for motion_id in range(5)}
  assert merged.motion_teacher_codes == {motion_id: motion_id for motion_id in range(5)}
  assert merged.library.mapping_digest()


def manual_shards(*entries) -> tuple[Any, ...]:
  """A hand-built plan, for the plans no healthy partition would produce."""
  from mjlab.tasks.tracking.distillation.worker import ClipShard

  return tuple(
    ClipShard(
      worker_index=index,
      device=f"cuda:{index + 1}",
      teacher_ids=teacher_ids,
      motion_ids=motion_ids,
      num_envs=2 * len(teacher_ids),
    )
    for index, (teacher_ids, motion_ids) in enumerate(entries)
  )


def test_the_union_contract_refuses_overlapping_shards(
  cohort: CohortContract,
) -> None:
  """Two workers claiming one clip is refused, not merged twice.

  A clip built twice would put two rows blocks under one replay partition and
  make the identity claim a cohort whose membership is ambiguous.
  """
  shards = manual_shards(
    (("tiny_000", "tiny_001"), (0, 1)),
    (("tiny_000", "tiny_002"), (0, 2)),
  )
  descriptions = [
    shard_description(0, ("tiny_000", "tiny_001"), (0, 1)),
    shard_description(1, ("tiny_000", "tiny_002"), (0, 2)),
  ]

  with pytest.raises(DistillationError, match="assigns clip 'tiny_000' to two"):
    distill._require_one_shard_contract(
      descriptions, shard_setup(cohort), clip_shards=shards
    )


def test_the_union_contract_refuses_a_plan_that_omits_a_clip(
  cohort: CohortContract,
) -> None:
  """A plan missing a requested clip is refused before any row is merged.

  The run would otherwise claim a cohort member no worker built, so its replay
  partition would stay empty while every counter looked complete.
  """
  shards = manual_shards(
    (("tiny_000", "tiny_001"), (0, 1)),
    (("tiny_002", "tiny_003"), (2, 3)),
  )
  descriptions = [
    shard_description(0, ("tiny_000", "tiny_001"), (0, 1)),
    shard_description(1, ("tiny_002", "tiny_003"), (2, 3)),
  ]

  with pytest.raises(DistillationError, match="but the run requested"):
    distill._require_one_shard_contract(
      descriptions, shard_setup(cohort), clip_shards=shards
    )


def test_the_union_contract_refuses_clip_ids_that_are_not_positions(
  cohort: CohortContract,
) -> None:
  """Clip ids must be the cohort's positions, because the identity records them.

  A plan that renumbered a selection from one, or skipped an id, would produce a
  cohort identity whose members are not its ordered positions.
  """
  shards = manual_shards(
    (("tiny_000", "tiny_001"), (1, 2)),
  )
  descriptions = [shard_description(0, ("tiny_000", "tiny_001"), (1, 2))]

  with pytest.raises(DistillationError, match="clip ids are 0..1 in cohort order"):
    distill._require_one_shard_contract(
      descriptions,
      shard_setup(cohort, ("tiny_000", "tiny_001")),
      clip_shards=shards,
    )


def test_the_union_contract_refuses_a_slice_that_is_not_what_the_plan_gave(
  cohort: CohortContract,
) -> None:
  """A worker that built other clips than its slice is refused by name.

  The plan is what the parent partitions the replay and records the identity by,
  so a shard that collected under a different clip set would train its rows
  against another clip's partition.
  """
  shards = clip_shard_plan(cohort, TEACHER_IDS, DEVICES, envs_per_motion=2)
  descriptions = list(descriptions_for(cohort, shards))
  # Worker 1 collected for worker 0's slice.
  descriptions[1] = shard_description(1, shards[0].teacher_ids, shards[0].motion_ids)

  with pytest.raises(DistillationError, match="plan gave it"):
    distill._require_one_shard_contract(
      descriptions, shard_setup(cohort), clip_shards=shards
    )


def test_the_union_contract_refuses_a_shard_with_a_different_schema(
  cohort: CohortContract,
) -> None:
  """The observation schema must agree across shards even when clips differ.

  Every shard packs its own rows, so a shard with a different layout would
  collect rows the run's model cannot consume under the same identity.
  """
  shards = clip_shard_plan(cohort, TEACHER_IDS, DEVICES, envs_per_motion=2)
  descriptions = list(descriptions_for(cohort, shards))
  descriptions[2] = shard_description(
    2, shards[2].teacher_ids, shards[2].motion_ids, schema="other-schema"
  )

  with pytest.raises(DistillationError, match="observation schema"):
    distill._require_one_shard_contract(
      descriptions, shard_setup(cohort), clip_shards=shards
    )


def test_the_union_contract_accepts_shards_with_their_own_reset_windows(
  cohort: CohortContract,
) -> None:
  """Two shards of one cohort report different reset window sets, by design.

  ``effective_windows`` holds one window per clip of the slice a shard owns, so
  a two-worker smoke of one 201-clip selection reported 101 windows on one shard
  and 100 on the other and failed the union check there.  Nothing about the
  cohort's reset contract differed; only the slice did.  Each shard's count is
  recorded in the merged metadata, so the per-slice window set is still
  auditable rather than dropped.
  """
  # Three clips on one shard and two on the other, the shape of 101 and 100.
  shards = manual_shards(
    (("tiny_000", "tiny_001", "tiny_002"), (0, 1, 2)),
    (("tiny_003", "tiny_004"), (3, 4)),
  )
  descriptions = with_reset_provenance(descriptions_for(cohort, shards))
  windows = [
    description.reset_provenance.effective_windows for description in descriptions
  ]
  assert windows == [(4, 4, 4), (4, 4)], "the test needs differing window sets"

  merged = distill._require_one_shard_contract(
    descriptions, shard_setup(cohort), clip_shards=shards
  )

  assert [clip.teacher_id for clip in merged.clips] == list(TEACHER_IDS)
  assert merged.reset_window_counts == (3, 2)


def test_the_union_contract_refuses_a_shard_that_resets_under_another_policy(
  cohort: CohortContract,
) -> None:
  """Only the window set is excused: a different reset policy still fails.

  A shard that resets with another standing fraction collects rows under a
  reset contract the run does not record, which is exactly what the comparison
  exists to refuse.
  """
  shards = manual_shards(
    (("tiny_000", "tiny_001", "tiny_002"), (0, 1, 2)),
    (("tiny_003", "tiny_004"), (3, 4)),
  )
  descriptions = with_reset_provenance(descriptions_for(cohort, shards))
  descriptions[1].reset_provenance = shard_reset_provenance(
    (4, 4),
    policy=ResetPolicy(kind="standing-mixture", standing_start_fraction=0.5),
  )

  with pytest.raises(
    DistillationError, match="live reset provenance.reset_policy"
  ) as error:
    distill._require_one_shard_contract(
      descriptions, shard_setup(cohort), clip_shards=shards
    )

  assert "standing_start_fraction" in str(error.value)


def test_the_whole_cohort_contract_still_requires_one_window_set(
  cohort: CohortContract,
) -> None:
  """Without a plan every shard holds every clip, so its windows must agree.

  The exclusion is scoped to the clip-sharded layout: in the whole-cohort layout
  the same clips have the same lengths on every shard, so a differing window
  set means those shards reset under different contracts.
  """

  def whole(worker_index: int, windows: tuple[int, ...]) -> SimpleNamespace:
    description = shard_description(
      worker_index, TEACHER_IDS, tuple(range(len(TEACHER_IDS)))
    )
    description.reset_policy_enabled = True
    description.reset_provenance = shard_reset_provenance(windows)
    return description

  same = (whole(0, (4, 4, 4, 4, 4)), whole(1, (4, 4, 4, 4, 4)))
  assert distill._require_one_shard_contract(same, shard_setup(cohort)) is same[0]

  other = whole(1, (4, 4, 4, 4, 5))
  with pytest.raises(
    DistillationError, match="live reset provenance.effective_windows"
  ):
    distill._require_one_shard_contract([same[0], other], shard_setup(cohort))


def test_the_union_contract_keeps_the_equality_contract_without_a_plan(
  cohort: CohortContract,
) -> None:
  """Without a plan the check is the historical equality contract.

  Every worker builds the whole selection in that layout, so equal clip sets are
  the correct requirement and no merge happens; a caller that passed a plan by
  mistake would otherwise be able to change the default path's contract.
  """
  shards = clip_shard_plan(cohort, TEACHER_IDS, DEVICES, envs_per_motion=2)
  descriptions = descriptions_for(cohort, shards)

  with pytest.raises(DistillationError, match="reference library clips"):
    distill._require_one_shard_contract(descriptions, shard_setup(cohort))

  # Without a plan every worker builds the whole selection, each on its own
  # seed stratum, which is the layout the equality contract describes.
  same = tuple(
    shard_description(index, TEACHER_IDS, tuple(range(len(TEACHER_IDS))))
    for index in range(len(DEVICES))
  )
  assert distill._require_one_shard_contract(same, shard_setup(cohort)) is same[0]


def test_parent_metadata_covers_the_cohort_without_loading_a_clip(
  cohort: CohortContract,
) -> None:
  """Replay weights, extents, codes and the digest come from the shards alone.

  A clip-sharded parent must never load the cohort's reference arrays or resolve
  every teacher: that is the memory the split exists to save.  So the merged
  metadata has to reproduce, from the workers' reports, exactly what a
  single-process build of the same recipe records -- including the ordered
  mapping digest.
  """
  shards = clip_shard_plan(cohort, TEACHER_IDS, DEVICES, envs_per_motion=2)
  descriptions = []
  for shard in shards:
    plan = plan_multi_motion(
      cohort,
      shard.teacher_ids,
      shard.num_envs,
      motion_ids=dict(zip(shard.teacher_ids, shard.motion_ids, strict=True)),
    )
    description = shard_description(
      shard.worker_index, shard.teacher_ids, shard.motion_ids
    )
    description.library = plan.library.with_body_selection(
      BodySelection(
        indices=tuple(range(len(cohort.body_names))),
        source_body_count=plan.library.source_body_count,
        names=tuple(cohort.body_names),
      )
    )
    description.audit = replace(description.audit, slots=plan.slots)
    description.motion_teacher_codes = {
      clip.motion_id: clip.teacher_code for clip in plan.library.clips
    }
    descriptions.append(description)

  merged = distill._require_one_shard_contract(
    descriptions, shard_setup(cohort), clip_shards=shards
  )
  whole = plan_multi_motion(cohort, TEACHER_IDS, 32)

  assert merged.replay_weights() == {
    clip.motion_id: float(cohort.teacher(clip.teacher_id).entry.sampling_weight)
    for clip in whole.library.clips
  }
  assert merged.frame_counts() == {
    clip.motion_id: int(clip.frames) for clip in whole.library.clips
  }
  assert merged.motion_teacher_codes == {
    clip.motion_id: clip.teacher_code for clip in whole.library.clips
  }
  # The identity a sharded run records is the single-process one.
  assert merged.library.mapping_digest() == whole.library.mapping_digest()


# Resume identity and the default path.


def real_descriptions(cohort: CohortContract, shards) -> list[SimpleNamespace]:
  """Descriptions whose clip records come from the real cohort's slices.

  Only metadata is real here: every library is loaded from its own slice, which
  is what a worker reports, and no reference tensor ever reaches the parent.
  """
  descriptions = []
  for shard in shards:
    plan = plan_multi_motion(
      cohort,
      shard.teacher_ids,
      shard.num_envs,
      motion_ids=dict(zip(shard.teacher_ids, shard.motion_ids, strict=True)),
    )
    description = shard_description(
      shard.worker_index, shard.teacher_ids, shard.motion_ids
    )
    description.num_envs = shard.num_envs
    description.library = plan.library.with_body_selection(
      BodySelection(
        indices=tuple(range(len(cohort.body_names))),
        source_body_count=plan.library.source_body_count,
        names=tuple(cohort.body_names),
      )
    )
    description.audit = replace(description.audit, slots=plan.slots)
    description.motion_teacher_codes = {
      clip.motion_id: clip.teacher_code for clip in plan.library.clips
    }
    descriptions.append(description)
  return descriptions


def cohort_replay(cohort: CohortContract, clips, *, num_motions: int):
  """The parent replay a run would build from these clips, metadata only."""
  from mjlab.tasks.tracking.distillation.balanced_storage import BalancedReplayBuffer
  from mjlab.tasks.tracking.distillation.vae_config import DEFAULT_SCHEMA

  return BalancedReplayBuffer(
    64,
    DEFAULT_SCHEMA,
    {
      clip.motion_id: float(cohort.teacher(clip.teacher_id).entry.sampling_weight)
      for clip in clips
    },
    teacher_codes={clip.motion_id: clip.teacher_code for clip in clips},
    frame_counts={clip.motion_id: int(clip.frames) for clip in clips},
    device=torch.device("cpu"),
    dtype=torch.float32,
  )


def test_a_sharded_run_records_the_single_process_identity(
  cohort: CohortContract,
) -> None:
  """The merged metadata produces the identity a single-process run records.

  This is the contract the whole split rests on: the workers hold the clips, the
  parent holds metadata, and a checkpoint from either layout must describe the
  same cohort -- same ordered mapping digest, same members, same replay policy.
  """
  from mjlab.tasks.tracking.distillation.cohort_contract import (
    cohort_identity_from_parts,
  )
  from mjlab.tasks.tracking.distillation.multi_motion import multi_motion_slots

  shards = clip_shard_plan(cohort, TEACHER_IDS, DEVICES, envs_per_motion=2)
  num_envs = sum(shard.num_envs for shard in shards)
  merged = distill._require_one_shard_contract(
    real_descriptions(cohort, shards), shard_setup(cohort), clip_shards=shards
  )
  single = plan_multi_motion(cohort, TEACHER_IDS, num_envs)
  selection = BodySelection(
    indices=tuple(range(len(cohort.body_names))),
    source_body_count=single.library.source_body_count,
    names=tuple(cohort.body_names),
  )
  seed_provenance = merged.audit.seed_provenance
  assert seed_provenance is not None
  slots = multi_motion_slots(
    cohort,
    TEACHER_IDS,
    num_envs,
    slot_generator=torch.Generator().manual_seed(BASE_SEED),
  )

  sharded_identity = cohort_identity_from_parts(
    cohort=cohort,
    library=merged.library,
    audit=replace(
      merged.audit,
      slots=slots,
      mapping_digest=merged.library.mapping_digest(),
    ),
    replay=cohort_replay(cohort, merged.clips, num_motions=len(TEACHER_IDS)),
    device="cpu",
  )
  single_identity = cohort_identity_from_parts(
    cohort=cohort,
    library=single.library.with_body_selection(selection),
    audit=SimpleNamespace(
      slots=slots,
      mapping_digest=single.library.mapping_digest(),
      phase_policy="uniform",
      seed_provenance=seed_provenance,
      semantic_overrides=(),
    ),
    replay=cohort_replay(cohort, single.library.clips, num_motions=len(TEACHER_IDS)),
    device="cpu",
  )

  assert sharded_identity.mapping_digest == single_identity.mapping_digest
  assert sharded_identity.digest() == single_identity.digest()


def test_the_sharded_parent_builds_no_cohort_teacher_bank(
  cohort: CohortContract, monkeypatch: pytest.MonkeyPatch
) -> None:
  """The clip-sharded parent hands its trainer no teacher at all.

  The parent labels no row in this layout: a shard's own bank labels the rows
  that shard collects, and sharded evaluation runs on the workers.  A bank in
  the parent would resolve one frozen actor per cohort member onto the trainer
  card -- 15.8 MiB each, 50.3 GiB at 3 261 clips -- for a trainer that only
  keeps a teacher frozen.
  """
  shards = clip_shard_plan(cohort, TEACHER_IDS, DEVICES, envs_per_motion=2)
  descriptions = real_descriptions(cohort, shards)
  # The shard stand-ins carry a placeholder schema; the builder needs the one
  # the run's model and replay are built from.
  for description in descriptions:
    description.schema = DEFAULT_SCHEMA

  class _FakePool:
    """Enough of a pool to hand the builder the replies the shards built."""

    def __init__(self, specs: Any) -> None:
      self.specs = specs

    def start(self) -> None:
      pass

    def describe(self) -> tuple[Any, ...]:
      return tuple(descriptions)

    def close(self) -> None:
      pass

  class _StopAtTrainer(Exception):
    """Ends the builder just after the trainer it was given is recorded."""

  received: list[dict[str, Any]] = []

  def record_trainer(*args: Any, **kwargs: Any) -> Any:
    received.append(kwargs)
    raise _StopAtTrainer

  monkeypatch.setattr(distill, "WorkerPool", _FakePool)
  monkeypatch.setattr(distill, "VaeDistillationTrainer", record_trainer)

  with pytest.raises(_StopAtTrainer):
    distill._build_sharded_cohort_runner(
      cohort=cohort,
      setup=shard_setup(cohort),
      teacher_ids=TEACHER_IDS,
      device="cpu",
      num_envs=sum(shard.num_envs for shard in shards),
      worker_devices=DEVICES,
      replay_capacity=64,
      phase_bins=0,
      minibatch_size=2,
      accumulation_steps=1,
      learning_rate=1e-4,
      beta=1.0,
      numerical_controls=False,
      seed=BASE_SEED,
      max_iterations=1,
      bootstrap_steps=0,
      collection_steps=1,
      updates_per_iteration=1,
      teacher_probability=0.5,
      evaluate_every=0,
      evaluation_steps=0,
      rollout_latent="mean",
      clip_shards=shards,
    )

  assert len(received) == 1
  assert received[0]["teacher"] is None


def test_the_resume_identity_round_trips_the_clip_plan(
  cohort: CohortContract,
) -> None:
  """The recorded plan is the partition a resume must reproduce.

  The replay was split by the ids the workers labelled their rows with, so a
  resume that re-partitioned the clips would file its rows differently.  The
  cards stay audit-only, exactly as they are for the other layouts.
  """
  shards = clip_shard_plan(cohort, TEACHER_IDS, DEVICES, envs_per_motion=2)
  identity = distill._workers_identity(
    DEVICES,
    num_envs=sum(shard.num_envs for shard in shards),
    clip_shards=shards,
    envs_per_motion=2,
  )

  plan = identity["clip_sharding"]
  assert plan["envs_per_motion"] == 2
  assert [entry["teacher_ids"] for entry in plan["shards"]] == [
    list(shard.teacher_ids) for shard in shards
  ]
  assert [entry["motion_ids"] for entry in plan["shards"]] == [
    list(shard.motion_ids) for shard in shards
  ]
  assert [entry["num_envs"] for entry in plan["shards"]] == [4, 4, 2]
  # Unequal shard totals are recorded as such instead of as a per-worker average.
  assert identity["envs_per_worker"] is None

  state = SimpleNamespace(
    resolved_config={"provenance_version": 1, "runtime": {"workers": identity}},
    schedule={"max_iterations": 4},
  )
  requested = {"runtime": {"workers": identity}, "schedule": {"max_iterations": 8}}
  audit = distill._check_resume_compatibility(
    state, requested, invariant_keys=("runtime",)
  )
  assert audit["mismatches"] == []

  # A changed partition is refused rather than silently re-sharded.
  reordered = clip_shard_plan(cohort, TEACHER_IDS, DEVICES, envs_per_motion=3)
  changed = distill._workers_identity(
    DEVICES,
    num_envs=sum(shard.num_envs for shard in reordered),
    clip_shards=reordered,
    envs_per_motion=3,
  )
  with pytest.raises(DistillationError, match="runtime"):
    distill._check_resume_compatibility(
      state,
      {"runtime": {"workers": changed}, "schedule": {"max_iterations": 8}},
      invariant_keys=("runtime",),
    )

  # Moving to other cards is audited, not refused.
  moved = distill._workers_identity(
    ("cuda:4", "cuda:5", "cuda:6"),
    num_envs=sum(shard.num_envs for shard in shards),
    clip_shards=shards,
    envs_per_motion=2,
  )
  moved_audit = distill._check_resume_compatibility(
    state,
    {"runtime": {"workers": moved}, "schedule": {"max_iterations": 8}},
    invariant_keys=("runtime",),
  )
  assert moved_audit["mismatches"] == []
  assert moved_audit["device_changed"] == {
    "worker_devices": {
      "stored": list(DEVICES),
      "requested": ["cuda:4", "cuda:5", "cuda:6"],
    }
  }


def test_the_cli_refuses_shard_flags_that_cannot_describe_a_plan(
  capsys: pytest.CaptureFixture[str],
) -> None:
  """The flag's own refusals happen before any manifest is opened.

  ``--shard-clips`` derives the environment total from the clip plan, so an
  explicit ``--num-envs``, a missing device list, and a missing row budget are
  contradictions rather than inputs a run could reconcile.
  """
  assert distill._train(teacher_ids=("tiny_000",), shard_clips=True) == 1
  assert "--shard-clips requires --worker-devices" in capsys.readouterr().err

  assert (
    distill._train(
      teacher_ids=("tiny_000",),
      worker_devices=("cuda:1",),
      shard_clips=True,
    )
    == 1
  )
  assert "--shard-clips requires --envs-per-motion" in capsys.readouterr().err

  assert (
    distill._train(
      teacher_ids=("tiny_000",),
      worker_devices=("cuda:1",),
      shard_clips=True,
      envs_per_motion=2,
      num_envs=4,
    )
    == 1
  )
  assert "--shard-clips refuses --num-envs" in capsys.readouterr().err


def test_an_uneven_total_is_accepted_only_for_clip_sharding() -> None:
  """An environment total that no even split can express is refused by default.

  The whole-cohort layout divides the total across workers, so an uneven one is
  still refused there; a clip-sharded run's plan bounds the totals instead.
  """
  with pytest.raises(ValueError, match="does not split evenly"):
    distill._resolve_worker_devices(("cuda:1", "cuda:2"), num_envs=9)
  assert distill._resolve_worker_devices(
    ("cuda:1", "cuda:2"), num_envs=9, even_split=False
  ) == ("cuda:1", "cuda:2")
  # Repeated CPU devices stay allowed: the CPU merge path has no card to share.
  assert distill._resolve_worker_devices(("cpu", "cpu"), num_envs=8) == ("cpu", "cpu")


def test_a_clip_sharded_default_is_off(cohort: CohortContract) -> None:
  """An unchanged run keeps today's worker layout and worker identity.

  ``_workers_identity`` records the whole-cohort layout when no plan is given,
  and a single-process run records no workers at all, so existing checkpoints
  still compare equal.
  """
  identity = distill._workers_identity(("cuda:1", "cuda:2"), num_envs=8)
  assert identity == {
    "mode": "sharded",
    "devices": ["cuda:1", "cuda:2"],
    "count": 2,
    "envs_per_worker": 4,
    "seed_scheme": 1,
  }
  assert "clip_sharding" not in identity
  single = distill._workers_identity((), num_envs=8)
  assert single["mode"] == "single"
  assert single["envs_per_worker"] == 8
  assert "clip_sharding" not in single
