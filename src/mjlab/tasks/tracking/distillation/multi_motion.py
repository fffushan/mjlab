"""Opt-in multi-motion command, row-slot allocation, and trusted build plan.

Scope: one tracking command term whose environment rows follow different
reference clips from a :class:`MotionLibrary`, the deterministic stratified
allocation that pins each row to a clip, and the pure planning step that the
private environment factory consumes.  The registered single-motion tracking
command is reused unchanged through :class:`SegmentMotionCommand`; nothing here
registers a new task or changes a shared command, manager, or viewer.

Baseline policy (M4): every row keeps its clip for the whole run, phase sampling
is ``uniform`` for training and ``start``/``uniform`` for evaluation, and
adaptive/weighted sampling and standing-start insertion are refused rather than
borrowing one clip's failure bins or default pose for the others.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

from mjlab.managers import CommandTerm
from mjlab.tasks.tracking.distillation.environment import (
  SegmentMotionCommand,
  SegmentMotionCommandCfg,
)
from mjlab.tasks.tracking.distillation.motion_library import (
  BodySelection,
  MotionClipSpec,
  MotionLibrary,
)
from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg, MotionLoader

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.tracking.distillation.config import CohortContract

PhasePolicy = Literal["uniform", "start"]

_PHASE_POLICIES: tuple[PhasePolicy, ...] = ("uniform", "start")
_METRIC_NAMES = (
  "error_anchor_pos",
  "error_anchor_rot",
  "error_anchor_lin_vel",
  "error_anchor_ang_vel",
  "error_body_pos",
  "error_body_rot",
  "error_joint_pos",
  "error_joint_vel",
  "sampling_entropy",
  "sampling_top1_prob",
  "sampling_top1_bin",
  "standing_start",
)


class MultiMotionError(ValueError):
  """A multi-motion request violates the pinned-slot command contract."""


@dataclass(frozen=True, slots=True)
class MotionSlotAllocation:
  """Stratified per-row clip slots: requested weights and realized counts.

  ``counts`` are aligned with ``teacher_ids`` (motion id 0..n-1), and
  ``row_motion_ids`` is the per-environment assignment of those motion ids.
  """

  weights: tuple[float, ...]
  counts: tuple[int, ...]
  teacher_ids: tuple[str, ...]
  row_motion_ids: tuple[int, ...]

  def __post_init__(self) -> None:
    size = len(self.weights)
    if size == 0:
      raise MultiMotionError("a slot allocation needs at least one motion")
    if len(self.counts) != size or len(self.teacher_ids) != size:
      raise MultiMotionError("slot weights, counts, and teacher ids must align")
    if len(set(self.teacher_ids)) != size:
      raise MultiMotionError("slot teacher ids must be unique")
    for weight in self.weights:
      if not math.isfinite(weight) or weight <= 0.0:
        raise MultiMotionError(f"slot weights must be finite and positive: {weight!r}")
    if any(count < 1 for count in self.counts):
      raise MultiMotionError(
        f"every selected motion needs at least one row, got {self.counts}"
      )
    if sum(self.counts) != len(self.row_motion_ids):
      raise MultiMotionError(
        f"slot counts {self.counts} do not cover {len(self.row_motion_ids)} rows"
      )
    if any(not 0 <= motion_id < size for motion_id in self.row_motion_ids):
      raise MultiMotionError(
        f"row motion ids must be in [0, {size}): {self.row_motion_ids}"
      )

  @property
  def num_envs(self) -> int:
    return len(self.row_motion_ids)

  @property
  def fractions(self) -> tuple[float, ...]:
    """Realized row fraction per motion, aligned with ``teacher_ids``."""
    return tuple(count / self.num_envs for count in self.counts)

  def as_dict(self) -> dict[str, Any]:
    """Plain-data form for provenance records and run reports."""
    return {
      "teacher_ids": list(self.teacher_ids),
      "weights": list(self.weights),
      "counts": list(self.counts),
      "fractions": list(self.fractions),
      "row_motion_ids": list(self.row_motion_ids),
      "num_envs": self.num_envs,
    }


@dataclass(frozen=True, slots=True)
class MultiMotionPlan:
  """Selected clips, pinned rows, and phase policy for one multi-motion build."""

  teacher_ids: tuple[str, ...]
  library: MotionLibrary
  slots: MotionSlotAllocation
  phase_policy: PhasePolicy


def stratified_slot_allocation(
  weights: Sequence[float],
  num_envs: int,
  *,
  teacher_ids: Sequence[str],
  generator: torch.Generator | None = None,
) -> MotionSlotAllocation:
  """Allocate one clip per environment row from positive motion weights.

  Counts are floor allocations plus a largest-deficit repair, so they are
  deterministic for a given ``(weights, num_envs)``, sum to ``num_envs``, and
  give every selected motion at least one row.  The row order is a permutation
  drawn from ``generator`` (the global RNG when ``generator`` is ``None``), so
  which row carries which clip is seeded rather than positional.

  Raises:
    MultiMotionError: if the weights are not positive and finite, if
      ``num_envs`` cannot represent every selected motion, or if the teacher
      ids do not align with the weights.
  """
  if isinstance(num_envs, bool) or not isinstance(num_envs, int):
    raise MultiMotionError(f"num_envs must be an integer, got {num_envs!r}")
  values = tuple(float(weight) for weight in weights)
  names = tuple(teacher_ids)
  size = len(values)
  if size == 0:
    raise MultiMotionError("a slot allocation needs at least one motion")
  if len(names) != size:
    raise MultiMotionError(f"got {size} weights but {len(names)} teacher ids")
  if len(set(names)) != size:
    raise MultiMotionError(f"slot teacher ids must be unique: {names}")
  for name, weight in zip(names, values, strict=True):
    if not math.isfinite(weight) or weight <= 0.0:
      raise MultiMotionError(
        f"weight for {name!r} must be finite and positive, got {weight!r}"
      )
  if num_envs < size:
    raise MultiMotionError(
      f"num_envs={num_envs} cannot represent {size} selected motions: at least one "
      "row per motion is required"
    )

  total = math.fsum(values)
  exact = [num_envs * weight / total for weight in values]
  counts = [int(math.floor(value)) for value in exact]
  # A very small weight can floor to zero rows, and every selected motion needs
  # at least one.  Spend leftover rows on the unrepresented motions first, and
  # only take a row from the largest allocation once no leftover row remains.
  leftover = num_envs - sum(counts)
  for index in range(size):
    if counts[index] > 0:
      continue
    if leftover > 0:
      counts[index] = 1
      leftover -= 1
      continue
    donor = max(range(size), key=lambda candidate: (counts[candidate], -candidate))
    if counts[donor] < 2:
      raise MultiMotionError(
        f"no allocation gives each of {size} motions a row out of {num_envs}"
      )
    counts[donor] -= 1
    counts[index] += 1
  for _ in range(leftover):
    best = max(
      range(size),
      key=lambda candidate: (exact[candidate] - counts[candidate], -candidate),
    )
    counts[best] += 1

  flat = [motion_id for motion_id, count in enumerate(counts) for _ in range(count)]
  order = torch.randperm(num_envs, generator=generator).tolist()
  return MotionSlotAllocation(
    weights=values,
    counts=tuple(counts),
    teacher_ids=names,
    row_motion_ids=tuple(flat[position] for position in order),
  )


def plan_multi_motion(
  cohort: CohortContract,
  teacher_ids: Sequence[str],
  num_envs: int,
  *,
  phase_policy: PhasePolicy = "uniform",
  slot_generator: torch.Generator | None = None,
  device: str | torch.device = "cpu",
) -> MultiMotionPlan:
  """Select clips, pin rows, and validate the phase policy before any build.

  Selected teachers are ordered by the manifest, not by the request, so numeric
  motion ids stay stable when a caller reorders its arguments; the ordered
  mapping digest changes only when the cohort's own order changes.

  Raises:
    MultiMotionError: if the selection is empty, duplicated, unknown, or the
      phase policy is not implemented for mixed slots.
    MotionLibraryError: if a selected reference motion cannot be loaded or
      disagrees with the extent or rate its teacher declares.
  """
  if phase_policy not in _PHASE_POLICIES:
    raise MultiMotionError(
      f"multi-motion phase sampling supports {list(_PHASE_POLICIES)}, got "
      f"{phase_policy!r}"
    )
  requested = tuple(teacher_ids)
  if not requested:
    raise MultiMotionError("multi-motion needs at least one teacher id")
  if len(set(requested)) != len(requested):
    raise MultiMotionError(f"duplicate teacher ids {requested}")
  known = {teacher.id for teacher in cohort.teachers}
  unknown = [teacher_id for teacher_id in requested if teacher_id not in known]
  if unknown:
    raise MultiMotionError(f"cohort has no teacher(s) {unknown}")
  wanted = set(requested)
  selected = tuple(teacher for teacher in cohort.teachers if teacher.id in wanted)
  codes = {teacher.id: index for index, teacher in enumerate(cohort.teachers)}
  library = MotionLibrary.from_clips(
    [
      MotionClipSpec(
        teacher_id=teacher.id,
        motion_file=teacher.entry.motion,
        teacher_code=codes[teacher.id],
        expected_frames=teacher.reference.frames,
        expected_fps=teacher.reference.fps,
      )
      for teacher in selected
    ],
    device=device,
  )
  slots = stratified_slot_allocation(
    [teacher.entry.sampling_weight for teacher in selected],
    num_envs,
    teacher_ids=[teacher.id for teacher in selected],
    generator=slot_generator,
  )
  return MultiMotionPlan(
    teacher_ids=tuple(teacher.id for teacher in selected),
    library=library,
    slots=slots,
    phase_policy=phase_policy,
  )


def make_multi_motion_cfg(
  cfg: MotionCommandCfg, plan: MultiMotionPlan
) -> MultiMotionCommandCfg:
  """Copy a trusted registered motion config into the opt-in multi config.

  ``motion_file`` is emptied, because no single clip bounds the mixed rows and
  the authoritative per-row sources are ``plan.library.clips``, and
  ``sampling_mode`` becomes the plan's phase policy.
  """
  if isinstance(cfg, MultiMotionCommandCfg):
    raise MultiMotionError("motion command is already multi-motion")
  values = {field.name: getattr(cfg, field.name) for field in fields(MotionCommandCfg)}
  values["motion_file"] = ""
  values["sampling_mode"] = plan.phase_policy
  return MultiMotionCommandCfg(
    **values,
    library=plan.library,
    slots=plan.slots,
  )


@dataclass(kw_only=True)
class MultiMotionCommandCfg(SegmentMotionCommandCfg):
  """Private config selecting :class:`MultiMotionCommand`."""

  library: MotionLibrary
  """Reference clips, loaded in the plan's manifest order."""

  slots: MotionSlotAllocation
  """Pinned per-row clip assignment and its requested weights."""

  def build(self, env: ManagerBasedRlEnv) -> MultiMotionCommand:
    return MultiMotionCommand(self, env)


class MultiMotionCommand(SegmentMotionCommand):
  """Motion command whose environment rows follow different reference clips.

  Every clip-local query resolves through the row's own clip in the shared
  library, so no scalar clip length bounds a mixed batch: ``time_steps`` are
  clip-local frame indexes, ``row_lengths`` is the per-row bound, and wrap,
  phase sampling, relative-body refreshes, and endpoint lookahead all use the
  row's own clip.  A row keeps its clip for the whole segment; only
  :meth:`select_motion_ids` changes a row's clip, and it resets that row at the
  same time.

  There is deliberately no single ``motion`` loader, so inherited single-clip
  code paths fail loudly instead of silently using one clip's timeline, and the
  single-motion scrubber GUI is unavailable for mixed rows.
  """

  def __init__(self, cfg: MultiMotionCommandCfg, env: ManagerBasedRlEnv) -> None:
    self._validate_cfg(cfg)
    # MotionCommand.__init__ loads one MotionLoader for cfg.motion_file and
    # derives bin counts and the standing-start window from it.  That scalar
    # clip length must never bound mixed rows, so the shared command state is
    # initialized here instead of through a single-clip loader.
    CommandTerm.__init__(self, cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    if len(self.robot.body_names) != cfg.library.source_body_count:
      raise MultiMotionError(
        f"reference clips carry {cfg.library.source_body_count} source bodies but "
        f"the compiled robot has {len(self.robot.body_names)}: the clip body axis "
        "must be the compiled robot body order (Entity.body_names)"
      )
    self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
    self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )
    self.library = cfg.library.with_body_selection(
      BodySelection(
        indices=tuple(int(index) for index in self.body_indexes.tolist()),
        source_body_count=len(self.robot.body_names),
        names=tuple(cfg.body_names),
      )
    )
    self.slot_allocation = cfg.slots
    self.motion_ids = torch.tensor(
      list(cfg.slots.row_motion_ids), dtype=torch.long, device=self.device
    )
    if self.motion_ids.numel() != self.num_envs:
      raise MultiMotionError(
        f"slot allocation covers {self.motion_ids.numel()} rows but the environment "
        f"has {self.num_envs}"
      )

    if not math.isfinite(cfg.lookahead_s) or cfg.lookahead_s < 0.0:
      raise MultiMotionError(
        f"lookahead_s must be finite and non-negative, got {cfg.lookahead_s!r}"
      )
    self._motion_lookahead_steps = torch.tensor(
      [
        0 if cfg.lookahead_s <= 0.0 else max(1, math.ceil(cfg.lookahead_s * clip.fps))
        for clip in self.library.clips
      ],
      dtype=torch.long,
      device=self.device,
    )
    self._motion_bin_counts = torch.tensor(
      [int(clip.frames // (1 / env.step_dt)) + 1 for clip in self.library.clips],
      dtype=torch.long,
      device=self.device,
    )
    self._refresh_row_bindings()

    env_fps = 1.0 / env.step_dt
    mismatched = [
      clip for clip in self.library.clips if not math.isclose(clip.fps, env_fps)
    ]
    if mismatched:
      described = ", ".join(f"{clip.teacher_id}={clip.fps:g}Hz" for clip in mismatched)
      warnings.warn(
        f"Motion trajectory FPS ({described}) differs from environment rate "
        f"({env_fps:g} Hz); each clip still advances one frame per env step.",
        UserWarning,
        stacklevel=2,
      )

    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0
    for name in _METRIC_NAMES:
      self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    self._ghost_model = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)
    self._pending_forward = False
    self._init_boundary_bookkeeping()

  # Public row metadata.

  @property
  def row_lengths(self) -> torch.Tensor:
    """Per-row clip length in frames, ``[num_envs]``."""
    return self._row_lengths

  @property
  def teacher_codes(self) -> torch.Tensor:
    """Per-row frozen-teacher label code, ``[num_envs]``."""
    return self._teacher_codes

  @property
  def row_lookahead_steps(self) -> torch.Tensor:
    """Per-row lookahead offset in frames, derived from the row's clip rate."""
    return self._row_lookahead_steps

  @property
  def row_bin_counts(self) -> torch.Tensor:
    """Per-row phase-bin count computed from the row's own clip length."""
    return self._row_bin_counts

  def select_motion_ids(self, env_ids: torch.Tensor, motion_ids: torch.Tensor) -> None:
    """Pin rows to clips and fully reset exactly those rows.

    This is the only path that changes ``motion_ids``: selection never mutates a
    row mid-segment, because the selected rows are reset through the
    environment's own reset, which resamples their phase, writes their reference
    state, advances their generation once, and refreshes scene state.  Other
    rows keep their clip, frame, and generation.  Do not reset the same rows
    again in the same tick.
    """
    ids = self._row_ids(env_ids, "env_ids")
    if ids.numel() == 0:
      raise MultiMotionError("select_motion_ids needs at least one row")
    new_ids = self._row_ids(
      motion_ids, "motion_ids", limit=self.library.num_motions, unique=False
    )
    if new_ids.numel() != ids.numel():
      raise MultiMotionError(f"got {ids.numel()} rows but {new_ids.numel()} motion ids")
    self.motion_ids[ids] = new_ids
    self._refresh_row_bindings()
    self._env.reset(env_ids=ids)

  # Reference queries.

  @property
  def motion(self) -> MotionLoader:
    """Refuse the single-clip loader: no scalar clip length bounds mixed rows.

    The authoritative per-row references are ``library`` with ``motion_ids``.
    """
    raise MultiMotionError(
      "MultiMotionCommand has no single 'motion' loader; read 'command.library' "
      "with 'command.motion_ids' for per-row references. Multi-motion supports "
      "the 'uniform' and 'start' phase policies only."
    )

  @property
  def lookahead_command(self) -> torch.Tensor:
    """Future joint target per row, clamped inside the row's own clip."""
    if self.cfg.lookahead_s <= 0.0:
      return torch.empty(self.num_envs, 0, dtype=self.library.dtype, device=self.device)
    frames = self.library.clamp_local_frames(
      self.motion_ids, self.time_steps + self.row_lookahead_steps
    )
    return torch.cat(
      [
        self.library.joint_pos(self.motion_ids, frames),
        self.library.joint_vel(self.motion_ids, frames),
      ],
      dim=1,
    )

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.library.joint_pos(self.motion_ids, self.time_steps)

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.library.joint_vel(self.motion_ids, self.time_steps)

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.library.body_pos_w(self.motion_ids, self.time_steps)
      + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.library.body_quat_w(self.motion_ids, self.time_steps)

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.library.body_lin_vel_w(self.motion_ids, self.time_steps)

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.library.body_ang_vel_w(self.motion_ids, self.time_steps)

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self.library.body_pos_w(self.motion_ids, self.time_steps)[
        :, self.motion_anchor_body_index
      ]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.library.body_quat_w(self.motion_ids, self.time_steps)[
      :, self.motion_anchor_body_index
    ]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.library.body_lin_vel_w(self.motion_ids, self.time_steps)[
      :, self.motion_anchor_body_index
    ]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.library.body_ang_vel_w(self.motion_ids, self.time_steps)[
      :, self.motion_anchor_body_index
    ]

  # Command term hooks.

  def _uniform_sampling(self, env_ids: torch.Tensor) -> None:
    """Sample each row's phase uniformly inside that row's own clip."""
    lengths = self.row_lengths[env_ids]
    draws = torch.rand(len(env_ids), dtype=torch.float64, device=self.device)
    self.time_steps[env_ids] = (draws * lengths.to(torch.float64)).to(torch.long)
    self.metrics["sampling_entropy"][:] = 1.0
    self.metrics["sampling_top1_prob"][:] = 1.0 / self.row_bin_counts.to(torch.float32)
    self.metrics["sampling_top1_bin"][:] = 0.5

  def _adaptive_sampling(self, env_ids: torch.Tensor) -> None:
    raise MultiMotionError(
      "adaptive phase sampling is refused for multi-motion: failure bins are "
      "clip-local and are never shared across clips; use 'uniform' or 'start'"
    )

  def _weighted_sampling(self, env_ids: torch.Tensor) -> None:
    raise MultiMotionError(
      "weighted phase sampling is refused for multi-motion; use 'uniform' or 'start'"
    )

  def _select_standing_start_envs(self, env_ids: torch.Tensor) -> torch.Tensor:
    """Return an all-False mask: mixed slots never insert the standing pose.

    ``standing_start_prob`` must be zero for a multi-motion build, so no row is
    initialized from the entity default pose.
    """
    self.metrics["standing_start"][env_ids] = 0.0
    return torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    """Advance each row's clip-local frame and wrap against its own length."""
    if env_ids is None:
      candidate = torch.arange(self.num_envs, device=self.device)
    else:
      candidate = env_ids
    self._wrap_resample_ids = candidate[
      self.time_steps[candidate] + 1 >= self.row_lengths[candidate]
    ]
    try:
      if env_ids is None:
        self.time_steps += 1
      else:
        self.time_steps[env_ids] += 1
      wrap_ids = torch.where(self.time_steps >= self.row_lengths)[0]
      if wrap_ids.numel() > 0:
        self._resample_command(wrap_ids)

      # _resample_command writes qpos/qvel but does not refresh derived
      # quantities; forward() so update_relative_body_poses reads the
      # post-teleport robot anchor instead of the stale pre-resample pose.
      if self._pending_forward:
        self._pending_forward = False
        self._env.sim.forward()
      self.update_relative_body_poses()
    finally:
      self._wrap_resample_ids = torch.empty(0, dtype=torch.long, device=self.device)

  def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
    """Teleport the given rows to an explicit clip-local frame.

    ``frame`` is validated against every selected row's own clip length, so a
    frame index is never read from a neighbouring clip.
    """
    ids = self._row_ids(env_ids, "env_ids")
    if ids.numel() == 0:
      raise MultiMotionError("reset_to_frame needs at least one row")
    if isinstance(frame, bool) or not isinstance(frame, int):
      raise MultiMotionError(f"frame must be an integer, got {frame!r}")
    lengths = self.row_lengths[ids]
    if frame < 0 or bool((lengths <= frame).any()):
      raise MultiMotionError(
        f"frame {frame} is outside the selected rows' clip lengths {lengths.tolist()}"
      )
    super().reset_to_frame(ids, frame)

  def _validate_cfg(self, cfg: MultiMotionCommandCfg) -> None:
    if not isinstance(cfg, MultiMotionCommandCfg):
      raise MultiMotionError(
        f"MultiMotionCommand needs a MultiMotionCommandCfg, got {type(cfg).__name__}"
      )
    if not isinstance(cfg.library, MotionLibrary):
      raise MultiMotionError("multi-motion config needs a MotionLibrary")
    if not isinstance(cfg.slots, MotionSlotAllocation):
      raise MultiMotionError("multi-motion config needs a MotionSlotAllocation")
    if cfg.sampling_mode not in _PHASE_POLICIES:
      raise MultiMotionError(
        f"multi-motion phase sampling supports {list(_PHASE_POLICIES)}, got "
        f"{cfg.sampling_mode!r}"
      )
    if cfg.standing_start_prob != 0.0:
      raise MultiMotionError(
        "standing-start initialization is not implemented for mixed clip slots; "
        f"build with standing_start_prob=0.0, got {cfg.standing_start_prob!r}"
      )
    library_teachers = tuple(clip.teacher_id for clip in cfg.library.clips)
    if library_teachers != cfg.slots.teacher_ids:
      raise MultiMotionError(
        f"slot allocation teachers {cfg.slots.teacher_ids} disagree with the "
        f"library clip order {library_teachers}"
      )

  def _refresh_row_bindings(self) -> None:
    """Recompute every per-row value derived from the pinned motion ids."""
    self._row_lengths = self.library.frame_counts_for(self.motion_ids)
    self._teacher_codes = self.library.teacher_codes_for(self.motion_ids)
    self._row_lookahead_steps = self._motion_lookahead_steps[self.motion_ids]
    self._row_bin_counts = self._motion_bin_counts[self.motion_ids]

  def _row_ids(
    self,
    value: torch.Tensor,
    name: str,
    *,
    limit: int | None = None,
    unique: bool = True,
  ) -> torch.Tensor:
    if limit is None:
      limit = self.num_envs
    if not isinstance(value, torch.Tensor):
      raise MultiMotionError(f"{name} must be a torch.Tensor")
    if value.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
      raise MultiMotionError(f"{name} must be an integer tensor, got {value.dtype}")
    if value.ndim != 1:
      raise MultiMotionError(f"{name} must be one-dimensional")
    if value.device != self.motion_ids.device:
      raise MultiMotionError(
        f"{name} is on {value.device} but the command is on {self.motion_ids.device}"
      )
    ids = value.to(torch.int64)
    if ids.numel() and bool((ids < 0).any()):
      raise MultiMotionError(f"{name} must be non-negative: {ids.tolist()}")
    if ids.numel() and bool((ids >= limit).any()):
      raise MultiMotionError(f"{name} must be in [0, {limit}): {ids.tolist()}")
    if unique and ids.numel() and len(set(ids.tolist())) != ids.numel():
      raise MultiMotionError(f"{name} must not repeat rows: {ids.tolist()}")
    return ids


__all__ = [
  "MultiMotionCommand",
  "MultiMotionCommandCfg",
  "MultiMotionError",
  "MultiMotionPlan",
  "MotionSlotAllocation",
  "PhasePolicy",
  "make_multi_motion_cfg",
  "plan_multi_motion",
  "stratified_slot_allocation",
]
