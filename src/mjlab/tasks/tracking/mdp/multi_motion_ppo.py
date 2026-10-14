"""Plain-file multi-motion PPO command for the X2 teacher task.

Scope: the reference command of the X2 multi-motion PPO teacher.  Its clip list
comes from a plain YAML manifest instead of a frozen teacher cohort, and its
episode boundary is the clip endpoint instead of an implicit reference wrap.
Everything else is inherited: the distillation :class:`MultiMotionCommand`
already owns clip-local queries, pinned row slots, preview clamping, and
library validation, so this module adds no new motion storage and no teacher.
All input clips must use the compiled X2 joint and body order.
The NPZ arrays have no order metadata; shape checks cannot detect column permutations.

PPO differences from the distillation command:

* A row never wraps.  It advances to its clip's last frame and holds that frame
  until the environment resets it, so no environment step teleports a moving
  robot to a different part of its reference.
* A command timer expiry does not resample the reference.  Only an environment
  reset and the clip endpoint end a reference segment.
* A row keeps the frame sampled by its reset through the first command update
  after that reset, because the environment computes that update with ``dt=0``
  and therefore must not skip the frame the policy starts from.  An explicit
  ``reset_to_frame`` teleport is deliberately not protected this way: no
  command update separates it from the next action, so the inherited one-frame
  advance there is the correct continuation.
* :func:`clip_complete` reports the per-row clip endpoint.  The environment
  computes terminations before it advances command frames, so a true entry means
  the last-frame action was just executed; the reset that follows is the only
  teleport of that episode.

The registered configuration stays plain data: clip files, the loaded
:class:`~mjlab.tasks.tracking.distillation.motion_library.MotionLibrary`, and
the row slots live in the runtime ``MultiMotionCommandCfg`` that
:meth:`MultiMotionPpoCommandCfg.build` resolves once, on the environment device,
while setting up the environment.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
import yaml

from mjlab.tasks.tracking.distillation.motion_library import (
  MotionClipSpec,
  MotionLibrary,
)
from mjlab.tasks.tracking.distillation.multi_motion import (
  MultiMotionCommand,
  MultiMotionCommandCfg,
  MultiMotionPlan,
  PhasePolicy,
  make_multi_motion_cfg,
  stratified_slot_allocation,
)
from mjlab.tasks.tracking.distillation.reset_policy import ResetPolicy
from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_FPS_RTOL = 1e-4
"""Relative tolerance for a clip rate, matching the motion library's own."""


class MultiMotionPpoError(ValueError):
  """A multi-motion PPO request violates the reference-command contract."""


def repository_root() -> Path:
  """Return the mjlab checkout root that relative reference paths resolve from."""
  return Path(__file__).resolve().parents[5]


def resolve_motion_files(cfg: MultiMotionPpoCommandCfg) -> tuple[Path, ...]:
  """Return the ordered, resolved reference clip files selected by ``cfg``.

  A nonempty ``motion_manifest`` names a plain YAML file holding a nonempty
  ``motions:`` list.  A nonempty ``motion_file`` selects exactly one clip, which
  is the playback path.  Setting both is ambiguous and rejected, so neither
  source can silently override the other.

  Relative entries resolve from :func:`repository_root`, not from the caller's
  working directory.  Every returned path exists and appears once.

  Raises:
    MultiMotionPpoError: if the source is missing, ambiguous, malformed, empty,
      repeated, or names a file that does not exist.
  """
  if not isinstance(cfg, MultiMotionPpoCommandCfg):
    raise MultiMotionPpoError(
      f"resolve_motion_files needs a MultiMotionPpoCommandCfg, got {type(cfg).__name__}"
    )
  root = repository_root()
  if cfg.motion_manifest and cfg.motion_file:
    raise MultiMotionPpoError(
      "select one reference source: motion_manifest "
      f"{cfg.motion_manifest!r} and motion_file {cfg.motion_file!r} are both set"
    )
  if cfg.motion_manifest:
    entries = _load_manifest_entries(_resolve_path(root, cfg.motion_manifest))
  elif cfg.motion_file:
    entries = (cfg.motion_file,)
  else:
    raise MultiMotionPpoError(
      "the multi-motion PPO command needs a nonempty motion_manifest or motion_file"
    )
  paths = tuple(_resolve_path(root, entry) for entry in entries)
  for path in paths:
    if not path.is_file():
      raise MultiMotionPpoError(f"reference clip not found: {path}")
  if len(set(paths)) != len(paths):
    raise MultiMotionPpoError(
      "the reference clip list repeats a file: "
      f"{[str(path) for path in paths]}. Each clip must be listed once."
    )
  return paths


def clip_complete(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
  """Per-row mask of rows whose final reference frame was just executed.

  The environment computes terminations before it advances command frames, so a
  true entry means the row's last-frame action is the action of this step and
  the reference task for that row is finished.  Register this as a termination
  term with ``time_out=False``: completion is a reference-task boundary, not a
  time limit, and an episode time limit remains a separate timeout.

  The termination manager ORs every term, so a physical failure on the same step
  is reported by its own term.  A caller that needs the success/failure split
  reads this term from :meth:`TerminationManager.get_term` and treats a
  simultaneous failure as the failure.
  """
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, MultiMotionPpoCommand):
    raise MultiMotionPpoError(
      f"clip_complete needs the multi-motion PPO command, but {command_name!r} is "
      f"{type(command).__name__}"
    )
  return command.reference_endpoint


def _resolve_path(root: Path, value: str) -> Path:
  path = Path(value)
  return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_manifest_entries(path: Path) -> tuple[str, ...]:
  """Read the ``motions:`` list from a plain YAML reference manifest."""
  if not path.is_file():
    raise MultiMotionPpoError(f"motion manifest not found: {path}")
  try:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
  except yaml.YAMLError as exc:
    raise MultiMotionPpoError(
      f"motion manifest {path} must be plain YAML without python tags: {exc}"
    ) from exc
  if not isinstance(raw, Mapping):
    raise MultiMotionPpoError(f"motion manifest {path} must be a mapping")
  unknown = sorted(key for key in raw if key != "motions")
  if unknown:
    raise MultiMotionPpoError(
      f"motion manifest {path} has unknown keys {unknown}; only 'motions' is read"
    )
  entries = raw.get("motions")
  if not isinstance(entries, list) or not entries:
    raise MultiMotionPpoError(f"motion manifest {path} needs a nonempty 'motions' list")
  for entry in entries:
    if not isinstance(entry, str) or not entry.strip():
      raise MultiMotionPpoError(
        f"motion manifest {path} entries must be nonempty path strings, got {entry!r}"
      )
  return tuple(entries)


def _clip_specs(paths: Sequence[Path]) -> tuple[MotionClipSpec, ...]:
  """Name every clip by its file stem, which is its whole identity here.

  Clip names travel through the motion library's existing ``teacher_id`` field
  because that field is the library's clip identity; no teacher is loaded from
  it, and the clip-local gathers never consult it.
  """
  specs: list[MotionClipSpec] = []
  names: set[str] = set()
  for path in paths:
    name = path.stem
    if name in names:
      raise MultiMotionPpoError(
        f"clip identity {name!r} is shared by more than one reference file; "
        "every clip needs its own file name"
      )
    names.add(name)
    specs.append(MotionClipSpec(teacher_id=name, motion_file=path))
  return tuple(specs)


def _require_reference_compatibility(
  library: MotionLibrary, env: ManagerBasedRlEnv, entity_name: str
) -> None:
  """Refuse clips that the compiled robot or the control rate cannot run.

  The robot's joint axis supplies the expected joint count (31 for X2).
  This checks width, not joint order. Joint and body order are input preconditions
  because the reference arrays have no order metadata. A clip whose rate differs
  from ``env.step_dt`` is rejected rather than resampled: this version does not
  interpolate reference timing.
  """
  robot_joints = int(env.scene[entity_name].data.joint_pos.shape[1])
  if library.joint_dim != robot_joints:
    raise MultiMotionPpoError(
      f"reference clips carry {library.joint_dim} joints but the compiled robot "
      f"has {robot_joints}: every clip must use the compiled robot's X2 joint order"
    )
  env_fps = 1.0 / env.step_dt
  mismatched = [
    f"{clip.teacher_id}={clip.fps:g}Hz"
    for clip in library.clips
    if not math.isclose(clip.fps, env_fps, rel_tol=_FPS_RTOL)
  ]
  if mismatched:
    raise MultiMotionPpoError(
      f"reference clips {', '.join(mismatched)} do not match the {env_fps:g} Hz "
      "control rate; this version does not interpolate reference timing"
    )


def _phase_policy(cfg: MultiMotionPpoCommandCfg) -> PhasePolicy:
  if cfg.sampling_mode not in ("uniform", "start"):
    raise MultiMotionPpoError(
      "multi-motion PPO phase sampling supports 'uniform' (training) and 'start' "
      f"(playback and evaluation), got {cfg.sampling_mode!r}"
    )
  return cfg.sampling_mode


def _runtime_multi_motion_cfg(
  cfg: MultiMotionPpoCommandCfg, env: ManagerBasedRlEnv
) -> MultiMotionCommandCfg:
  """Load the reference library once and pin equal row counts to each clip."""
  paths = resolve_motion_files(cfg)
  phase_policy = _phase_policy(cfg)
  library = MotionLibrary.from_clips(_clip_specs(paths), device=env.device)
  _require_reference_compatibility(library, env, cfg.entity_name)
  # Equal weights realize equal pinned row counts within rounding, and the
  # allocation refuses a selection that has fewer environments than clips.
  slots = stratified_slot_allocation(
    [1.0] * library.num_motions,
    int(env.num_envs),
    teacher_ids=[clip.teacher_id for clip in library.clips],
  )
  plan = MultiMotionPlan(
    teacher_ids=tuple(clip.teacher_id for clip in library.clips),
    library=library,
    slots=slots,
    phase_policy=phase_policy,
  )
  return make_multi_motion_cfg(cfg, plan, reset_policy=ResetPolicy())


@dataclass(kw_only=True)
class MultiMotionPpoCommandCfg(MotionCommandCfg):
  """Serializable command config for the file-configured multi-motion PPO task.

  The loaded reference arrays and the row slots are deliberately absent: they
  are built once, on the environment device, by :meth:`build`.
  """

  motion_manifest: str = ""
  """Plain YAML file holding the ``motions:`` reference list.

  Relative paths in the manifest and in this field resolve from the mjlab
  checkout root.  The manifest is the training source; an explicitly selected
  ``motion_file`` is the playback alternative, never a second source.
  """

  motion_file: str = ""
  """Singleton reference clip, used only when no manifest is configured."""

  sampling_mode: Literal["adaptive", "uniform", "start", "weighted"] = "uniform"
  """Reference phase sampling: ``uniform`` for training, ``start`` for playback.

  The distillation command supports no other policy for mixed rows.
  """

  def build(self, env: ManagerBasedRlEnv) -> MultiMotionPpoCommand:
    return MultiMotionPpoCommand(_runtime_multi_motion_cfg(self, env), env)


class MultiMotionPpoCommand(MultiMotionCommand):
  """Multi-motion command whose reference segments end at the clip endpoint.

  The runtime config is the resolved :class:`MultiMotionCommandCfg` (library and
  slots included), so this class only replaces the inherited reference-timeline
  lifecycle.  Inherited clip-local queries, preview clamping, metrics, and
  boundary bookkeeping are unchanged, and the finite non-negative ``lookahead_s``
  validation of the shared command still applies.
  """

  @property
  def reference_endpoint(self) -> torch.Tensor:
    """Per-row mask: the row's current clip-local frame is its clip's last frame.

    Because terminations are computed before command frames advance, a true
    entry describes the action of the current step, which used the final frame.
    """
    return self.time_steps >= self.row_lengths - 1

  def _resample(self, env_ids: torch.Tensor) -> None:
    """Refill a timer expiry, and resample only for an environment reset.

    A reset-path resample reaches this method while the reset rows are scoped on
    the inherited ``_reset_resample_ids``; every other caller is a command timer
    expiry, which must not move an in-flight row's reference frame.
    """
    if self._reset_resample_ids.numel() != 0:
      super()._resample(env_ids)
      return
    if len(env_ids) != 0:
      self.time_left[env_ids] = self.time_left[env_ids].uniform_(
        *self.cfg.resampling_time_range
      )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Sample and teleport, then hold the sampled frame for the next update.

    Only an environment reset reaches this method, because timer expiries are
    not resampled and a row never wraps.  Marking the resampled rows pending is
    what keeps the frame the reset sampled: the environment follows a reset with
    a ``dt=0`` command update, which would otherwise advance the fresh frame.
    """
    super()._resample_command(env_ids)
    reset_ids = env_ids[torch.isin(env_ids, self._reset_resample_ids)]
    if reset_ids.numel():
      self._pending_reset_frame_ids = reset_ids.detach().clone()

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    """Advance each row one frame and hold it at its own clip's last frame.

    A row that reached its last frame keeps that frame until an environment
    reset resamples it, so this command never wraps and never teleports a row
    inside an episode.  Rows that a reset just resampled keep the sampled frame
    for this update.
    """
    if env_ids is None:
      candidate = torch.arange(self.num_envs, device=self.device)
    else:
      candidate = env_ids
    pending = candidate[torch.isin(candidate, self._pending_reset_frame_ids)]
    advance = candidate[~torch.isin(candidate, pending)]
    if advance.numel():
      self.time_steps[advance] = torch.minimum(
        self.time_steps[advance] + 1, self.row_lengths[advance] - 1
      )
    try:
      # A reset-path teleport is followed by the environment's own forward(),
      # and this command resamples nothing else, but the shared command's
      # post-teleport rule is kept so relative poses are never read stale.
      if self._pending_forward:
        self._pending_forward = False
        self._env.sim.forward()
      self.update_relative_body_poses()
    finally:
      self._pending_reset_frame_ids = torch.empty(
        0, dtype=torch.long, device=self.device
      )


__all__ = [
  "MultiMotionPpoCommand",
  "MultiMotionPpoCommandCfg",
  "MultiMotionPpoError",
  "clip_complete",
  "repository_root",
  "resolve_motion_files",
]
