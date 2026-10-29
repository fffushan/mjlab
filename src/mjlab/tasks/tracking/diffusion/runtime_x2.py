"""Concrete AgiBot X2 runtime factory for D1 uniform-anchor collection.

This module owns the simulator boundary the offline D1 code deliberately does
not: it builds the registered tracking environment, drives the frozen
version-3 cohort VAE for a whole rollout, and reports physical post-step
evidence.

Design constraints taken from the D1 uniform-anchor plan
------------------------------------------------------

* A shared multi-motion environment may carry many rows.  Rows pin their own
  reference clip and the collector owns every reset boundary.
* One rollout starts at frame 0 of its clip and runs to the last frame.  The
  collector ends it there; nothing continues past the clip end.
* The tracking task's own reference terminations and the reference wrap are
  suspended for active rows, so a rollout can only end on its own endpoint or
  on the task time-out.
* Both initial poses (the reference state at frame 0 and ``STAND_KEYFRAME``)
  receive the same seeded perturbation in the same simulator state, so the
  clean/OU snapshot check has one deterministic reset to verify.
* Eight synthetic ``DEFAULT_ROW`` rows pad every rollout.  They hold the
  standing keyframe and its identity-root kinematics, and are history only.

The module is intentionally not a CLI: :func:`build_x2_runtime` is the
``--runtime-factory`` entry point, and the CLI has already applied its
fail-closed budget and artifact checks before calling it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .adapter import (
  ActionContract,
  AdapterError,
  FrozenVaePolicy,
  PostStepBundle,
  VaeObservation,
  VaeObservationTensorBatch,
  resolve_inference_device,
)
from .collector import DiffusionCollector, TrialSpec
from .contract import DEFAULT_CONTRACT
from .model import ContractDimensions, contract_dimensions
from .qualification import QualificationConfig, StepEvidence
from .state import WorldState

# Frozen D1 design (docs/plans/beyondmimic_diffusion_d1_uniform_anchor_collection.md).
INITIAL_POSES: tuple[str, ...] = ("reference", "standing")
INITIAL_STATE_MODES: tuple[str, ...] = ("perturbed", "reference")
PILOT_SEEDS: tuple[int, ...] = (0, 1, 2)
D1_RUN_ID = "d1-pilot-x2-tennis-mixed-v1"
D1_BULK_RUN_ID = "d1-bulk-x2-tennis-mixed-v1"

# One window needs ``future_steps`` rows after its anchor, so the last anchor of
# a clip of ``N`` frames is ``N - future_steps - 1``.
_ANCHOR_TAIL = DEFAULT_CONTRACT.future_steps

# Frozen three-clip cohort used by both frozen revisions' pilot schedule.
_DEFAULT_COHORT_MANIFEST = "configs/distillation/x2_tennis_mixed.yaml"

# The plan sizes the bulk collection as ``L * 50 * 100`` retained rows
# (~111300 for the three clips).  At the pilot yield of ~108 accepted rows per
# trial that is ~1030 trials, i.e. 57 seeds x 3 phases x 3 clips pairs.
# Parity tolerance for "the OU reset reproduced the clean reset".  The state is
# written, read back through the simulator and round-tripped in single precision,
# so a reproducible reset can still differ in the last bits of a quaternion.
_RESTORE_PARITY_ATOL = 1e-6

BULK_SEED_COUNT = 57

_END_EFFECTOR_BODIES: tuple[str, ...] = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)

_BODY_COUNT = len(DEFAULT_CONTRACT.body_names)


_REFERENCE_TERMINATIONS: tuple[str, ...] = ("anchor_pos", "anchor_ori", "ee_body_pos")


def _yaw_only(quaternion: torch.Tensor) -> torch.Tensor:
  """The yaw-only orientation of a quaternion, as the standing insert writes it."""
  from mjlab.utils.lab_api.math import yaw_quat

  return yaw_quat(quaternion)


def _apply_rpy(quaternion: torch.Tensor, rpy: np.ndarray) -> torch.Tensor:
  """Left-multiply one quaternion batch by a roll/pitch/yaw delta."""
  from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

  angles = torch.as_tensor(
    np.asarray(rpy, dtype=np.float64), dtype=torch.float32, device=quaternion.device
  ).reshape(1, 3)
  delta = quat_from_euler_xyz(angles[:, 0], angles[:, 1], angles[:, 2])
  return quat_mul(delta, quaternion)


def _never_terminate(env: Any, **_: Any) -> torch.Tensor:
  """Always-false termination, used to suspend reference guards."""
  return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


class RuntimeX2Error(RuntimeError):
  """The concrete X2 runtime cannot be built or driven safely."""


@dataclass(frozen=True, slots=True)
class InitialPerturbation:
  """One seeded reset perturbation, in the world and joint frames."""

  root_offset: np.ndarray
  root_rpy: np.ndarray
  joint_offsets: np.ndarray

  def __post_init__(self) -> None:
    object.__setattr__(
      self, "root_offset", np.asarray(self.root_offset, dtype=np.float64).reshape(3)
    )
    object.__setattr__(
      self, "root_rpy", np.asarray(self.root_rpy, dtype=np.float64).reshape(3)
    )
    offsets = np.asarray(self.joint_offsets, dtype=np.float64).reshape(-1)
    if offsets.shape != (DEFAULT_CONTRACT.action_dimension,):
      raise RuntimeX2Error("joint perturbation must cover every actuated joint")
    if not all(
      np.isfinite(value).all() for value in (self.root_offset, self.root_rpy, offsets)
    ):
      raise RuntimeX2Error("initial perturbation must be finite")
    object.__setattr__(self, "joint_offsets", offsets)


@dataclass(frozen=True, slots=True)
class DefaultRow:
  """The synthetic standing row that pads the start of every rollout.

  ``reference`` is the standing keyframe packed in the frozen reference layout.
  ``state`` is that keyframe's kinematics from an identity root.  Only
  ``state.root_position`` comes from the clip, so the fake history adds no
  world-absolute step at frame 0.
  """

  reference: np.ndarray
  state: WorldState


_ANKLE_JOINT = re.compile(r"^(?:left|right)_ankle_(?:pitch|roll)_joint$")
_JOINT_OFFSET_RANGE = (-0.01, 0.01)
_ANKLE_OFFSET_RANGE = (-0.1, 0.1)
_ROOT_OFFSET_RANGES = ((-0.05, 0.05), (-0.05, 0.05), (-0.01, 0.01))
_ROOT_ROTATION_RANGES = ((-0.1, 0.1), (-0.1, 0.1), (-0.2, 0.2))
_REFERENCE_FIELD_WIDTHS: tuple[tuple[str, int], ...] = (
  ("reference_q", DEFAULT_CONTRACT.action_dimension),
  ("reference_dq", DEFAULT_CONTRACT.action_dimension),
  ("anchor_orientation_error", 6),
  # Version-2 cohorts append one future reference frame: joint positions and
  # joint velocities.  A padding row has no future, so its preview is the same
  # standing pose with zero velocity.
  ("motion_lookahead", 2 * DEFAULT_CONTRACT.action_dimension),
)


def initial_perturbation(seed: int, joint_names: Sequence[str]) -> InitialPerturbation:
  """The plan's reset perturbation for one seed.

  The generator depends on the seed alone, so the same seed produces the same
  perturbation for both initial poses and for the clean and OU phase of a pair.
  Velocities stay zero for this build.
  """
  names = tuple(joint_names)
  if len(names) != DEFAULT_CONTRACT.action_dimension:
    raise RuntimeX2Error(
      f"expected {DEFAULT_CONTRACT.action_dimension} joints, got {len(names)}"
    )
  rng = np.random.default_rng(int(seed))
  root_offset = np.asarray(
    [rng.uniform(*bounds) for bounds in _ROOT_OFFSET_RANGES], dtype=np.float64
  )
  root_rpy = np.asarray(
    [rng.uniform(*bounds) for bounds in _ROOT_ROTATION_RANGES], dtype=np.float64
  )
  offsets = np.asarray(
    rng.uniform(*_JOINT_OFFSET_RANGE, size=len(names)), dtype=np.float64
  )
  ankles = [index for index, name in enumerate(names) if _ANKLE_JOINT.match(name)]
  if ankles:
    offsets[ankles] = rng.uniform(*_ANKLE_OFFSET_RANGE, size=len(ankles))
  return InitialPerturbation(root_offset, root_rpy, offsets)


def stand_joint_positions(joint_names: Sequence[str]) -> np.ndarray:
  """``STAND_KEYFRAME`` joint positions in the given joint order."""
  from mjlab.asset_zoo.robots.agibot_x2.x2_constants import STAND_KEYFRAME
  from mjlab.utils.string import resolve_expr

  patterns = STAND_KEYFRAME.joint_pos
  if patterns is None:
    raise RuntimeX2Error("STAND_KEYFRAME declares no joint_pos patterns")
  return np.asarray(
    resolve_expr(patterns, tuple(joint_names), 0.0),
    dtype=np.float64,
  )


def stand_body_positions(model: Any, robot: Any) -> np.ndarray:
  """Body positions of ``STAND_KEYFRAME`` under an identity root pose.

  ``mj_kinematics`` runs on a private data object, so this reads the compiled
  model without stepping, sensing, or writing the running simulation.
  """
  import mujoco

  data = mujoco.MjData(model)
  data.qpos[:] = 0.0
  data.qvel[:] = 0.0
  free = robot.indexing.free_joint_q_adr.detach().cpu().numpy()
  if free.size != 7:
    raise RuntimeX2Error("the X2 robot root is not a free joint")
  data.qpos[int(free[3])] = 1.0
  joints = robot.indexing.joint_q_adr.detach().cpu().numpy()
  data.qpos[joints] = stand_joint_positions(robot.joint_names)
  mujoco.mj_kinematics(model, data)
  # The compiled entity prefixes its body names ("robot/pelvis"), so resolve the
  # contract order through the entity's own name table and model indices.
  entity_names = tuple(robot.body_names)
  entity_ids = robot.indexing.body_ids.detach().cpu().numpy()
  body_ids = [
    int(entity_ids[entity_names.index(name)]) for name in DEFAULT_CONTRACT.body_names
  ]
  return np.asarray(data.xpos[body_ids], dtype=np.float64).reshape(_BODY_COUNT, 3)


def stand_reference(
  reference_names: Sequence[str], joint_positions: np.ndarray
) -> np.ndarray:
  """The standing keyframe packed in the frozen reference layout.

  ``reference_q`` holds the standing joints, ``reference_dq`` and the anchor
  orientation error are zero: the padding row states the standing pose, not a
  measurement of the robot against it.
  """
  widths = dict(_REFERENCE_FIELD_WIDTHS)
  pieces: list[np.ndarray] = []
  for name in reference_names:
    width = widths.get(name)
    if width is None:
      raise RuntimeX2Error(
        f"reference layout field {name!r} has no standing default; the padding "
        "row is defined for reference_q, reference_dq, anchor_orientation_error"
      )
    joints = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
    if name == "reference_q":
      pieces.append(joints.reshape(width))
    elif name == "motion_lookahead":
      # The preview is (q, dq) of the next frame; the padding row holds the
      # standing pose at rest, so the repeated q is the same standing joints.
      pieces.append(np.concatenate((joints, np.zeros_like(joints))).reshape(width))
    else:
      pieces.append(np.zeros(width, dtype=np.float64))
  return np.concatenate(pieces)


def anchor_frames_for(frames: int) -> range:
  """Anchor frames ``0 … N - 33`` for a clip of ``N`` frames.

  The anchor is the current row of a training window, so it needs 32 rows
  behind it for the history (eight padding rows plus the clip opening) and 32
  after it for the future targets.
  """
  if frames < _ANCHOR_TAIL + 1:
    raise RuntimeX2Error(
      f"clip must have at least {_ANCHOR_TAIL + 1} frames, got {frames}"
    )
  return range(0, frames - _ANCHOR_TAIL)


def build_trials(
  motion_frames: Sequence[tuple[str, int]],
  *,
  seeds: Sequence[int],
  run_id: str,
  initial_poses: Sequence[str] = INITIAL_POSES,
  worker: int = 0,
  workers: int = 1,
) -> tuple[TrialSpec, ...]:
  """Build a pair-local clean/OU schedule for the given seeds and poses.

  Each OU item follows its own clean item.  Unrelated pairs may interleave.  The
  old global clean-before-OU ordering protected the single-row snapshot map, but
  the collector checks the matching fingerprint and qualification at each pair
  admission, so that global restriction is not required.

  ``worker``/``workers`` take a disjoint stride slice of whole pairs so several
  processes can collect in parallel.  Each worker still receives every one of
  its own clean trials before any of its own OU trials, which is what the
  collector and the CLI require, and a pair is never split across workers
  because the OU restore needs its clean partner in the same process.

  One rollout covers every anchor of its clip, because it starts at frame 0 and
  runs to the last frame.  The pair identity therefore carries the initial pose,
  which is what makes the two poses of one clip separate clean/OU pairs.  The
  per-row split group is ``motion:seed:anchor`` and is set by the collector, so
  the two poses of a clip cannot land in different splits.
  """
  if workers < 1 or not 0 <= worker < workers:
    raise RuntimeX2Error(
      f"worker must satisfy 0 <= worker < workers, got {worker}/{workers}"
    )
  poses = tuple(initial_poses)
  unknown = [pose for pose in poses if pose not in INITIAL_POSES]
  if not poses or unknown:
    raise RuntimeX2Error(
      f"initial poses must be a non-empty subset of {INITIAL_POSES}, got {poses}"
    )
  if len(set(poses)) != len(poses):
    raise RuntimeX2Error(f"initial poses repeat a value: {poses}")
  for motion_id, frames in motion_frames:
    if frames < _ANCHOR_TAIL + 1:
      raise RuntimeX2Error(
        f"clip {motion_id!r} needs {_ANCHOR_TAIL + 1} frames, has {frames}"
      )
  pairs = [
    (motion_id, seed, pose)
    for motion_id, _ in motion_frames
    for seed in seeds
    for pose in poses
  ]
  clean: list[TrialSpec] = []
  ou: list[TrialSpec] = []
  for motion_id, seed, pose in pairs[worker::workers]:
    pair_id = f"{motion_id}:{seed}:{pose}"
    # The clean and OU trials of one pose share one initial state; a later
    # unperturbed attempt would use its own identity.
    initial_state_id = f"init:{motion_id}:{seed}:{pose}:" + "perturbed"
    clean.append(
      TrialSpec(
        run_id=run_id,
        motion_id=motion_id,
        seed=seed,
        start_frame=0,
        phase="clean",
        group_key=pair_id,
        pair_id=pair_id,
        initial_state_id=initial_state_id,
        initial_pose=pose,
      )
    )
    ou.append(
      TrialSpec(
        run_id=run_id,
        motion_id=motion_id,
        seed=seed,
        start_frame=0,
        phase="ou",
        group_key=pair_id,
        pair_id=pair_id,
        initial_state_id=initial_state_id,
        initial_pose=pose,
      )
    )
  return tuple(item for pair in zip(clean, ou, strict=True) for item in pair)


def resolve_rollout_ids(cohort_ids: Sequence[str], selection: str) -> tuple[str, ...]:
  """Resolve the rollout teacher selection against a cohort manifest.

  An empty ``selection`` means "every member".  A non-empty selection must name
  each member at most once and every name must be a cohort member, so a typo can
  never silently roll out a different clip or a member the checkpoint cannot
  serve.  This is what keeps the checkpoint's cohort identity (which it
  validates) separate from the clips actually rolled out.
  """
  ids = tuple(str(member) for member in cohort_ids)
  if not ids:
    raise RuntimeX2Error("cohort manifest contains no members")
  chosen = tuple(part.strip() for part in selection.split(",") if part.strip())
  rollout = chosen if chosen else ids
  if len(set(rollout)) != len(rollout):
    raise RuntimeX2Error(f"rollout selection repeats a member: {rollout}")
  unknown = [member for member in rollout if member not in ids]
  if unknown:
    raise RuntimeX2Error(f"rollout teachers {unknown} are not members of {ids}")
  return rollout


def build_pilot_trials(
  motion_frames: Sequence[tuple[str, int]],
  *,
  run_id: str = D1_RUN_ID,
  initial_poses: Sequence[str] = INITIAL_POSES,
  worker: int = 0,
  workers: int = 1,
) -> tuple[TrialSpec, ...]:
  """The pilot schedule: seeds 0-2, every initial pose, every anchor."""
  return build_trials(
    motion_frames,
    seeds=PILOT_SEEDS,
    run_id=run_id,
    initial_poses=initial_poses,
    worker=worker,
    workers=workers,
  )


def build_bulk_trials(
  motion_frames: Sequence[tuple[str, int]],
  *,
  seed_count: int = BULK_SEED_COUNT,
  run_id: str = D1_BULK_RUN_ID,
  initial_poses: Sequence[str] = INITIAL_POSES,
  worker: int = 0,
  workers: int = 1,
) -> tuple[TrialSpec, ...]:
  """The bulk schedule: ``seed_count`` seeds times every initial pose."""
  if seed_count <= 0:
    raise RuntimeX2Error("bulk collection needs at least one seed")
  return build_trials(
    motion_frames,
    seeds=tuple(range(seed_count)),
    run_id=run_id,
    initial_poses=initial_poses,
    worker=worker,
    workers=workers,
  )


def resolve_initial_poses(value: str) -> tuple[str, ...]:
  """Resolve ``MJLAB_D1_INITIAL_POSES`` to the poses every clip is rolled out in.

  ``both`` is the plan's schedule; the two single-pose values exist for a
  bounded smoke run and for diagnosing one pose at a time.
  """
  text = value.strip().lower()
  if text in ("", "both"):
    return INITIAL_POSES
  chosen = tuple(part.strip() for part in text.split(",") if part.strip())
  unknown = [pose for pose in chosen if pose not in INITIAL_POSES]
  if not chosen or unknown:
    raise RuntimeX2Error(
      f"MJLAB_D1_INITIAL_POSES must be 'both' or a subset of {INITIAL_POSES}, "
      f"got {value!r}"
    )
  if len(set(chosen)) != len(chosen):
    raise RuntimeX2Error(f"MJLAB_D1_INITIAL_POSES repeats a pose: {chosen}")
  return chosen


def resolve_policy_device(value: str, *, simulation_device: str) -> str:
  """Resolve ``MJLAB_D1_POLICY_DEVICE`` to one concrete inference device.

  ``cpu`` (the default) keeps the frozen CPU policy path.  ``simulation``
  follows the simulator device, so a device-native policy needs no transfer.
  ``cuda[:N]`` names an explicit GPU.  The choice is validated here, before any
  checkpoint or environment is constructed, and the resolved value is recorded
  in the runtime provenance.
  """
  text = value.strip().lower()
  if text == "simulation":
    text = simulation_device
  try:
    return resolve_inference_device(text or "cpu")
  except AdapterError as exc:
    raise RuntimeX2Error(
      f"MJLAB_D1_POLICY_DEVICE={value!r} is not a usable policy device: {exc}"
    ) from exc


def resolve_worker_offset(value: str, worker: int, workers: int) -> int:
  """Resolve ``MJLAB_D1_WORKER_OFFSET`` for one worker.

  A resumed collection must skip the trials a previous run already stored, and
  each worker can stop at a different trial.  An empty value keeps the frozen
  behaviour (offset 0).  One integer applies to every worker; a comma-separated
  list gives one offset per worker and must have exactly ``workers`` entries.
  """
  text = value.strip()
  if not text:
    return 0
  parts = [part.strip() for part in text.split(",")]
  if any(part == "" for part in parts):
    raise RuntimeX2Error("MJLAB_D1_WORKER_OFFSET holds an empty entry")
  if len(parts) == 1:
    raw = parts[0]
  elif len(parts) == workers:
    raw = parts[worker]
  else:
    raise RuntimeX2Error(
      f"MJLAB_D1_WORKER_OFFSET has {len(parts)} entries but MJLAB_D1_WORKERS "
      f"is {workers}; give one value or one value per worker"
    )
  try:
    offset = int(raw)
  except ValueError as exc:
    raise RuntimeX2Error(
      f"MJLAB_D1_WORKER_OFFSET must hold integers, got {raw!r}"
    ) from exc
  if offset < 0:
    raise RuntimeX2Error("MJLAB_D1_WORKER_OFFSET must be non-negative")
  return offset


def _as_numpy(value: Any, width: int | None = None) -> np.ndarray:
  array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
  array = np.asarray(array, dtype=np.float64).reshape(-1)
  if width is not None and array.shape != (width,):
    raise RuntimeX2Error(f"expected {width} values, got {array.shape}")
  if not np.isfinite(array).all():
    raise RuntimeX2Error("runtime state contains non-finite values")
  return array


@dataclass(frozen=True, slots=True)
class InitialStateSnapshot:
  """Deterministic initialization identity for a clean/OU pair.

  Restoration is verification-based: the reset is deterministic in ``seed`` and
  ``start_frame``, so a matching pair must reproduce the captured state exactly.
  A mismatch fails closed instead of silently collecting different data.
  """

  seed: int
  start_frame: int
  signature: tuple[float, ...]

  def digest(self) -> str:
    payload = f"{self.seed}:{self.start_frame}:" + ",".join(
      f"{value:.12g}" for value in self.signature
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class X2MotionEnvironment:
  """One single-motion X2 tracking simulator with the D1 evidence contract."""

  def __init__(
    self,
    cohort: Any,
    teacher_id: str,
    *,
    device: str,
    seed: int,
  ) -> None:
    self.cohort = cohort
    self.teacher_id = teacher_id
    try:
      from mjlab.tasks.tracking.distillation.adapter import make_distillation_adapter
    except ImportError as exc:  # pragma: no cover - mjlab is a hard dependency
      raise RuntimeX2Error(f"mjlab distillation adapter unavailable: {exc}") from exc
    self.adapter = make_distillation_adapter(
      cohort, teacher_id, num_envs=1, device=device, seed=seed
    )
    # The adapter's schema is the cohort's live layout, including a declared
    # reference preview; the scalar observation derives its width from it.
    self._schema = self.adapter.schema
    self._reference_width = int(self._schema.reference_dim)
    self.env = self.adapter.env
    # The D1 contract forbids any mid-trial reset, so the environment may never
    # reset itself: the collector owns every reset boundary.
    self.env.cfg.auto_reset = False
    self.command = self.env.command_manager.get_term("motion")
    self.robot = self.env.scene["robot"]
    self.device = self.env.device
    self.period_seconds = float(self.env.step_dt)
    self.clip_frames = int(self.command.motion.time_step_total)
    self._row_ids = torch.arange(
      self.env.num_envs, dtype=torch.int64, device=self.device
    )

    body_names = tuple(self.robot.body_names)
    missing = [name for name in DEFAULT_CONTRACT.body_names if name not in body_names]
    if missing:
      raise RuntimeX2Error(f"compiled robot is missing contract bodies: {missing}")
    self._body_indexes = torch.tensor(
      [body_names.index(name) for name in DEFAULT_CONTRACT.body_names],
      dtype=torch.long,
      device=self.device,
    )
    command_bodies = tuple(self.command.cfg.body_names)
    try:
      self._ee_indexes = tuple(
        command_bodies.index(name) for name in _END_EFFECTOR_BODIES
      )
    except ValueError as exc:
      raise RuntimeX2Error(
        f"tracked body set lacks an end-effector body: {exc}"
      ) from exc
    self.joint_names = tuple(self.robot.joint_names)

    # The tracking task's own reference guards are suspended for the whole
    # rollout: a rollout is a stability test, and its reference terminations would
    # end it before the last frame is reached.
    self._reference_terms: dict[str, tuple[Any, dict[str, Any]]] = {}
    manager = self.env.termination_manager
    for name in _REFERENCE_TERMINATIONS:
      if name not in manager.active_terms:
        raise RuntimeX2Error(
          f"tracking task is missing the expected termination {name!r}"
        )
      index = manager.active_terms.index(name)
      term_cfg = manager._term_cfgs[index]
      if term_cfg.time_out:
        raise RuntimeX2Error(f"termination {name!r} must not be a time-out term")
      self._reference_terms[name] = (term_cfg.func, dict(term_cfg.params))
    self._set_reference_terminations(False)

    self._last_frame = 0
    self._trial_clock = 0.0
    self._initial: InitialStateSnapshot | None = None
    self._stand_body_positions: np.ndarray | None = None
    self._install_reference_advance_mask()
    self._default_row: DefaultRow | None = None

  # -- lifecycle -----------------------------------------------------------

  def reset(
    self, *, seed: int, initial_pose: str = "reference", perturb: bool = True
  ) -> None:
    """Full reset to frame 0, then the chosen initial pose and perturbation."""
    if initial_pose not in INITIAL_POSES:
      raise RuntimeX2Error(
        f"initial pose must be one of {INITIAL_POSES}, got {initial_pose!r}"
      )
    self.env.reset(seed=seed)
    self.command.reset_to_frame(self._row_ids, 0)
    self._refresh()
    self._remember_default_row()
    self._write_initial_pose(
      initial_pose=initial_pose, seed=int(seed), perturb=bool(perturb)
    )
    self._refresh()
    self._last_frame = 0
    self._trial_clock = 0.0
    self._initial = InitialStateSnapshot(
      seed=int(seed),
      start_frame=0,
      signature=self._initial_signature(),
    )

  def default_row(self) -> DefaultRow:
    """The padded history row of this clip, computed before the first rollout."""
    if self._default_row is None:
      raise RuntimeX2Error("the default row needs one reset before it is readable")
    return self._default_row

  def capture_initial_state(self) -> InitialStateSnapshot:
    if self._initial is None:
      raise RuntimeX2Error("no reset state has been captured yet")
    return self._initial

  def restore_initial_state(self, snapshot: InitialStateSnapshot) -> None:
    """Verify that the reset reproduced the pair's captured initial state.

    The comparison uses the same float32 parity tolerance as the batched path.
    Exact equality is wrong here: the state is written, read back through the
    simulator and round-tripped in single precision, so a reproducible reset can
    still differ in the last bits of one quaternion component.
    """
    current = np.asarray(self._initial_signature(), dtype=np.float64)
    expected = np.asarray(snapshot.signature, dtype=np.float64)
    if current.shape != expected.shape or not np.allclose(
      current, expected, rtol=0.0, atol=_RESTORE_PARITY_ATOL
    ):
      delta = np.abs(current - expected) if current.shape == expected.shape else None
      worst = int(delta.argmax()) if delta is not None and delta.size else -1
      detail = (
        f"; worst component {worst} now {current[worst]!r} expected {expected[worst]!r}"
        f" (max abs delta {float(delta.max()):.6g})"
        if delta is not None and delta.size
        else ""
      )
      raise AdapterError(
        "OU trial could not reproduce its clean partner's initial state; "
        f"expected {snapshot.digest()}{detail}"
      )

  def reset_after_trial(self) -> None:
    """Return the simulator to a clean, non-pending state for the next trial."""
    self.env.reset(env_ids=self._row_ids)

  def close(self) -> None:
    self.adapter.close()

  # -- observations --------------------------------------------------------

  def observe_vae(self) -> VaeObservation:
    return self._vae_observation(self.adapter.snapshot())

  def step(self, action: np.ndarray) -> Any:
    tensor = torch.as_tensor(
      _as_numpy(action, 31), dtype=torch.float32, device=self.device
    )[None]
    return self.adapter.step(tensor)

  def post_step_evidence(self, result: Any) -> PostStepBundle:
    """Convert one post-step distillation snapshot into D1 evidence.

    The observation is built from the same post-step snapshot the action
    produced, so the reported terminal state and the reported guard metrics
    describe one instant.
    """
    snapshot = result.snapshot
    frame = int(snapshot.reference_frame[0].item())
    # ``_last_frame`` is the frame the recorded row acts at.  The reference holds
    # on the clip's last frame, so a post frame equal to the row frame there is
    # expected rather than an unexpected teleport.
    row_frame = self._last_frame
    clip_end = row_frame >= self.clip_frames - 1
    teleport = frame != row_frame + 1 and not (clip_end and frame == row_frame)
    self._last_frame = frame
    observation = self._vae_observation(snapshot)
    evidence = StepEvidence(
      terminal_available=True,
      terminated=bool(result.terminated[0].item()),
      truncated=bool(result.time_outs[0].item()),
      reason=None,
      finite=self._post_step_finite(observation),
      physical_fall=False,
      tracking_rejection=False,
      reset=False,
      teleport=teleport,
      reference_boundary=clip_end,
      clip_ended=clip_end,
    )
    return PostStepBundle(evidence, self._timestamp(), observation)

  # -- internals -----------------------------------------------------------

  def _install_reference_advance_mask(self) -> None:
    """Advance the reference and hold it on the clip's last frame.

    The command's wrap path resamples a start frame and teleports the robot, so
    it must never run during a rollout: a rollout ends on its own endpoint
    instead.  A single-motion command has one clip, so ``clip_frames`` is its
    row length; the batched runtime reads a per-row length instead.
    """
    update = getattr(self.command, "_update_command", None)
    if update is None:
      # A command double without the private update hook cannot wrap, so there is
      # nothing to hold.  The real single-motion command always exposes it.
      return

    def advance(env_ids: torch.Tensor | None = None) -> None:
      ids = self._row_ids if env_ids is None else env_ids
      advancing = self.command.time_steps[ids] + 1 < self.clip_frames
      if bool(advancing.any()):
        update(ids[advancing])

    self.command._update_command = advance

  def _set_reference_terminations(self, enabled: bool) -> None:
    """Enable or suspend the tracking task's reference-dependent terminations.

    mjlab exposes no public term-disable API, so the term functions recorded at
    construction are swapped here.  The always-false replacement keeps the
    ``time_out`` term and the collector's own physical-tilt guard intact.
    """
    manager = self.env.termination_manager
    for name, (func, params) in self._reference_terms.items():
      index = manager.active_terms.index(name)
      term_cfg = manager._term_cfgs[index]
      if enabled:
        term_cfg.func = func
        term_cfg.params = dict(params)
      else:
        term_cfg.func = _never_terminate
        term_cfg.params = {}

  def _timestamp(self) -> float:
    now = self._trial_clock
    self._trial_clock = now + self.period_seconds
    return now

  @staticmethod
  def _post_step_finite(observation: Any) -> bool:
    state = observation.state
    values = (
      state.root_position,
      state.root_quaternion_wxyz,
      state.root_linear_velocity,
      state.root_angular_velocity,
      state.body_positions.reshape(-1),
      state.body_linear_velocities.reshape(-1),
    )
    return all(
      bool(np.isfinite(np.asarray(value, dtype=np.float64)).all()) for value in values
    )

  def _reference_phase(self, frame: int) -> float:
    return float(frame) / float(max(self.clip_frames - 1, 1))

  def _world_state(self) -> WorldState:
    data = self.robot.data
    return WorldState(
      root_position=_as_numpy(data.root_link_pos_w[0], 3),
      root_quaternion_wxyz=_as_numpy(data.root_link_quat_w[0], 4),
      root_linear_velocity=_as_numpy(data.root_link_lin_vel_w[0], 3),
      root_angular_velocity=_as_numpy(data.root_link_ang_vel_w[0], 3),
      body_positions=_as_numpy(
        data.body_link_pos_w[0][self._body_indexes], 3 * _BODY_COUNT
      ).reshape(_BODY_COUNT, 3),
      body_linear_velocities=_as_numpy(
        data.body_link_lin_vel_w[0][self._body_indexes], 3 * _BODY_COUNT
      ).reshape(_BODY_COUNT, 3),
    )

  def _initial_signature(self) -> tuple[float, ...]:
    data = self.robot.data
    values = np.concatenate(
      (
        _as_numpy(data.root_link_pos_w[0], 3),
        _as_numpy(data.root_link_quat_w[0], 4),
        _as_numpy(data.joint_pos[0], len(self.joint_names)),
        _as_numpy(data.joint_vel[0], len(self.joint_names)),
      )
    )
    return tuple(float(np.round(value, 12)) for value in values)

  def _vae_observation(self, snapshot: Any) -> VaeObservation:
    metrics = self._reference_metrics()
    return VaeObservation(
      state=self._world_state(),
      reference=_as_numpy(snapshot.packed.reference[0], self._reference_width),
      conditioning=_as_numpy(snapshot.packed.conditioning[0], 99),
      motion_id=self.teacher_id,
      reference_frame=int(snapshot.reference_frame[0].item()),
      reference_phase=self._reference_phase(int(snapshot.reference_frame[0].item())),
      **metrics,
    )

  def _refresh(self) -> None:
    """Replay the environment's own post-write recompute tail.

    ``reset_to_frame`` and the initial-state write change qpos/qvel after the
    environment's own command compute, so the observation cache, the anchor
    cache and the sensor data are stale until this runs.
    """
    self.env.scene.write_data_to_sim()
    self.env.sim.forward()
    self.env.sim.sense()
    self.command.update_relative_body_poses()
    self.env.obs_buf = self.env.observation_manager.compute(
      update_history=True, env_ids=self._row_ids
    )

  def _remember_default_row(self) -> None:
    """Cache the standing row from the clip's frame-0 reference position."""
    if self._default_row is not None:
      return
    if self._stand_body_positions is None:
      self._stand_body_positions = self._measure_stand_body_positions()
    root_position = _as_numpy(self.command.body_pos_w[0, 0], 3).copy()
    self._default_row = DefaultRow(
      reference=stand_reference(
        self._schema.reference_names, stand_joint_positions(self.joint_names)
      ),
      state=WorldState(
        root_position=root_position,
        root_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        root_linear_velocity=np.zeros(3, dtype=np.float64),
        root_angular_velocity=np.zeros(3, dtype=np.float64),
        body_positions=self._stand_body_positions.copy(),
        body_linear_velocities=np.zeros((_BODY_COUNT, 3), dtype=np.float64),
      ),
    )

  def _measure_stand_body_positions(self) -> np.ndarray:
    """Forward kinematics of ``STAND_KEYFRAME`` from an identity root."""
    if self._stand_body_positions is not None:
      return self._stand_body_positions
    self._stand_body_positions = stand_body_positions(self.env.sim.mj_model, self.robot)
    return self._stand_body_positions

  def _write_initial_pose(self, *, initial_pose: str, seed: int, perturb: bool) -> None:
    """Write the initial pose of one rollout and its seeded perturbation."""
    to_torch = lambda value, width: torch.as_tensor(  # noqa: E731
      np.asarray(value, dtype=np.float64),
      dtype=torch.float32,
      device=self.device,
    ).reshape(1, width)
    data = self.robot.data
    root_pos = to_torch(_as_numpy(data.root_link_pos_w[0], 3), 3)
    root_quat = to_torch(_as_numpy(data.root_link_quat_w[0], 4), 4)
    joint_pos = to_torch(
      _as_numpy(data.joint_pos[0], len(self.joint_names)), len(self.joint_names)
    )
    if initial_pose == "standing":
      origin_z = float(self.env.scene.env_origins[0, 2].item())
      root_pos[0, 2] = float(data.default_root_state[0, 2].item()) + origin_z
      root_quat = _yaw_only(self.command.anchor_quat_w[0:1])
      joint_pos = to_torch(
        stand_joint_positions(self.joint_names), len(self.joint_names)
      )
    if perturb:
      values = initial_perturbation(seed, self.joint_names)
      root_pos = root_pos + to_torch(values.root_offset, 3)
      root_quat = _apply_rpy(root_quat, values.root_rpy)
      joint_pos = joint_pos + to_torch(values.joint_offsets, len(self.joint_names))
    zeros3 = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
    self.robot.write_root_state_to_sim(
      torch.cat((root_pos, root_quat, zeros3, zeros3), dim=1), self._row_ids
    )
    self.robot.write_joint_state_to_sim(
      joint_pos, torch.zeros_like(joint_pos), env_ids=self._row_ids
    )

  def _reference_metrics(self) -> dict[str, float]:
    """The tracking task's own reference-error terms, reused verbatim."""
    from mjlab.utils.lab_api.math import quat_apply_inverse

    command = self.command
    gravity = self.robot.data.gravity_vec_w
    anchor_z_error = float(
      (command.anchor_pos_w[0, 2] - command.robot_anchor_pos_w[0, 2]).item()
    )
    motion_gravity_z = float(
      quat_apply_inverse(command.anchor_quat_w, gravity)[0, 2].item()
    )
    robot_gravity_z = float(
      quat_apply_inverse(command.robot_anchor_quat_w, gravity)[0, 2].item()
    )
    end_effector_z_error = max(
      abs(
        float(
          (
            command.body_pos_relative_w[0, index, 2]
            - command.robot_body_pos_w[0, index, 2]
          ).item()
        )
      )
      for index in self._ee_indexes
    )
    return {
      "anchor_z_error": anchor_z_error,
      "gravity_z_error": motion_gravity_z - robot_gravity_z,
      "end_effector_z_error": end_effector_z_error,
    }


class X2MultiMotionEnvironment:
  """Shared X2 environment exposing the collector's vectorized row protocol."""

  _RESTORE_PARITY_ATOL = _RESTORE_PARITY_ATOL

  def __init__(
    self,
    cohort: Any,
    teacher_ids: Sequence[str],
    *,
    num_envs: int,
    device: str,
    seed: int,
    policy_device: str | torch.device | None = None,
  ) -> None:
    from mjlab.tasks.tracking.distillation.environment import (
      build_multi_motion_environment,
    )

    if num_envs < len(teacher_ids):
      raise RuntimeX2Error("a shared environment needs at least one row per motion")
    self.cohort = cohort
    self.teacher_ids = tuple(teacher_ids)
    self.env = build_multi_motion_environment(
      cohort,
      teacher_ids,
      phase_policy="start",
      num_envs=num_envs,
      device=device,
      seed=seed,
    )
    self.env.cfg.auto_reset = False
    self.num_envs = int(self.env.num_envs)
    self.device = self.env.device
    # Device-native observations are placed where the frozen VAE lives.  The
    # default follows the simulator, which is the no-transfer choice; the
    # runtime factory passes the explicitly validated policy device instead.
    self.policy_device = torch.device(
      device if policy_device is None else policy_device
    )
    self.period_seconds = float(self.env.step_dt)
    self._row_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
    self.command: Any = self.env.command_manager.get_term("motion")
    self.robot = self.env.scene["robot"]
    self._trial_clock = np.zeros(self.num_envs, dtype=np.float64)
    self._pre_frames = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._last_result: Any = None
    self._body_indexes = torch.tensor(
      [self.robot.body_names.index(name) for name in DEFAULT_CONTRACT.body_names],
      dtype=torch.long,
      device=self.device,
    )
    self.joint_names = tuple(self.robot.joint_names)
    from mjlab.tasks.tracking.distillation.adapter import (
      _capture_snapshot,
      _term_slices,
      default_schema_for_cohort,
    )

    self._capture_snapshot = _capture_snapshot
    self._slices = _term_slices(cohort)
    # The batched runtime packs the cohort's live layout, preview included;
    # a preview cohort must not fall back to the preview-blind base schema.
    self._schema = default_schema_for_cohort(cohort)
    self._default_rows: dict[str, DefaultRow] = {}
    self._stand_body_positions: np.ndarray | None = None
    self._install_termination_masks()
    self._install_reference_advance_mask()

  def _install_reference_advance_mask(self) -> None:
    """Advance active references and hold them at their last clip frame.

    The wrap path resamples a new start frame and teleports the robot, so it must
    never run during a rollout: a rollout ends on its own endpoint instead.
    """
    update = self.command._update_command

    def advance(env_ids: torch.Tensor | None = None) -> None:
      candidates = self._row_ids if env_ids is None else env_ids
      advancing = torch.as_tensor(
        self._active_rows, dtype=torch.bool, device=self.device
      )[candidates] & (
        self.command.time_steps[candidates] + 1 < self.command.row_lengths[candidates]
      )
      if bool(advancing.any()):
        update(candidates[advancing])

    self.command._update_command = advance

  def _install_termination_masks(self) -> None:
    """Keep idle rows and every reference guard out of termination.

    The reference guards are suspended for the whole rollout, not only after a
    handoff: the collection is a stability test that must reach the last frame.
    """
    self._reference_terms: list[tuple[Any, dict[str, Any]]] = []
    self._active_rows = np.zeros(self.num_envs, dtype=bool)
    manager = self.env.termination_manager
    for name in _REFERENCE_TERMINATIONS:
      if name not in manager.active_terms:
        raise RuntimeX2Error(f"tracking task is missing termination {name!r}")
    for name in manager.active_terms:
      cfg = manager.get_term_cfg(name)
      original, params = cfg.func, dict(cfg.params)
      if name in _REFERENCE_TERMINATIONS:
        self._reference_terms.append((original, params))

      def masked(
        env: Any,
        _original: Any = original,
        _reference: bool = name in _REFERENCE_TERMINATIONS,
        **kwargs: Any,
      ) -> Any:
        enabled = np.zeros_like(self._active_rows) if _reference else self._active_rows
        return _original(env, **kwargs) & torch.as_tensor(
          enabled, dtype=torch.bool, device=env.device
        )

      cfg.func = masked

  def _refresh(self, ids: torch.Tensor) -> None:
    self.env.scene.write_data_to_sim()
    self.env.sim.forward()
    self.env.sim.sense()
    self.command.update_relative_body_poses()
    self.env.obs_buf = self.env.observation_manager.compute(
      update_history=True, env_ids=ids
    )

  def reset_batch(
    self, specs: Sequence[TrialSpec], *, env_ids: np.ndarray | None = None
  ) -> None:
    if not specs:
      raise RuntimeX2Error("cannot reset an empty trial batch")
    slots = (
      np.arange(len(specs), dtype=np.int64)
      if env_ids is None
      else np.asarray(env_ids, dtype=np.int64).reshape(-1)
    )
    if slots.size != len(specs):
      raise RuntimeX2Error("trial slots and specs must align")
    if np.any(slots >= self.num_envs):
      raise RuntimeX2Error("trial batch exceeds shared environment slots")
    self._active_rows[slots] = True
    ids = torch.as_tensor(slots, dtype=torch.long, device=self.device)
    motion_codes = torch.as_tensor(
      [self.teacher_ids.index(spec.motion_id) for spec in specs],
      dtype=torch.long,
      device=self.device,
    )
    if not torch.equal(self.command.motion_ids[ids], motion_codes):
      self.command.select_motion_ids(ids, motion_codes)
    for slot, spec in zip(slots.tolist(), specs, strict=True):
      self.env.reset(
        seed=spec.seed,
        env_ids=torch.as_tensor([slot], dtype=torch.long, device=self.device),
      )
    for frame in sorted({spec.start_frame for spec in specs}):
      frame_ids = ids[
        torch.as_tensor(
          [spec.start_frame == frame for spec in specs],
          dtype=torch.bool,
          device=self.device,
        )
      ]
      if frame_ids.numel():
        self.command.reset_to_frame(frame_ids, int(frame))
    self._refresh(ids)
    self._write_initial_poses(specs, ids)
    self._refresh(ids)
    self._trial_clock[slots] = 0.0
    self._pre_frames[ids] = self.command.time_steps[ids]

  def default_row(self, motion_id: str) -> DefaultRow:
    """The padded history row of one clip, computed before its first rollout."""
    row = self._default_rows.get(motion_id)
    if row is None:
      raise RuntimeX2Error(
        f"the default row of {motion_id!r} needs one reset before it is readable"
      )
    return row

  def _measure_stand_body_positions(self) -> np.ndarray:
    """Forward kinematics of ``STAND_KEYFRAME`` from an identity root.

    The pose is solved on a private MuJoCo data object, so no live rollout row
    is touched and no device work is scheduled.  The result is a pure function
    of the compiled model and the standing keyframe.
    """
    if self._stand_body_positions is not None:
      return self._stand_body_positions
    self._stand_body_positions = stand_body_positions(self.env.sim.mj_model, self.robot)
    return self._stand_body_positions

  def _write_initial_poses(self, specs: Sequence[TrialSpec], ids: torch.Tensor) -> None:
    """Write each slot's initial pose, its perturbation and its default row."""
    if self._stand_body_positions is None:
      self._measure_stand_body_positions()
    assert self._stand_body_positions is not None
    data = self.robot.data
    stand_joints = torch.as_tensor(
      stand_joint_positions(self.joint_names),
      dtype=torch.float32,
      device=self.device,
    ).repeat(len(specs), 1)
    root_pos = data.root_link_pos_w[ids].clone()
    root_quat = data.root_link_quat_w[ids].clone()
    joint_pos = data.joint_pos[ids].clone()
    for index, spec in enumerate(specs):
      if spec.initial_pose == "standing":
        row = ids[index]
        root_pos[index, 2] = (
          data.default_root_state[row, 2] + self.env.scene.env_origins[row, 2]
        )
        root_quat[index] = _yaw_only(self.command.anchor_quat_w[row : row + 1])[0]
        joint_pos[index] = stand_joints[index]
      if spec.initial_state_mode == "perturbed":
        values = initial_perturbation(spec.seed, self.joint_names)
        root_pos[index] += torch.as_tensor(
          values.root_offset, dtype=torch.float32, device=self.device
        )
        root_quat[index : index + 1] = _apply_rpy(
          root_quat[index : index + 1], values.root_rpy
        )
        joint_pos[index] += torch.as_tensor(
          values.joint_offsets, dtype=torch.float32, device=self.device
        )
    zeros = torch.zeros((len(specs), 3), dtype=torch.float32, device=self.device)
    self.robot.write_root_state_to_sim(
      torch.cat((root_pos, root_quat, zeros, zeros), dim=1), ids
    )
    self.robot.write_joint_state_to_sim(
      joint_pos, torch.zeros_like(joint_pos), env_ids=ids
    )
    for index, spec in enumerate(specs):
      if spec.motion_id in self._default_rows:
        continue
      self._default_rows[spec.motion_id] = DefaultRow(
        reference=stand_reference(
          self._schema.reference_names, stand_joint_positions(self.joint_names)
        ),
        state=WorldState(
          root_position=(
            self.command.body_pos_w[ids[index], 0].detach().cpu().numpy().copy()
          ),
          root_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
          root_linear_velocity=np.zeros(3, dtype=np.float64),
          root_angular_velocity=np.zeros(3, dtype=np.float64),
          body_positions=self._stand_body_positions.copy(),
          body_linear_velocities=np.zeros((_BODY_COUNT, 3), dtype=np.float64),
        ),
      )

  def _snapshot(self) -> Any:
    return self._capture_snapshot(
      self.env,
      cohort=self.cohort,
      slices=self._slices,
      schema=self._schema,
      command=self.command,
      teacher_id="multiple",
      teacher_code=-1,
      motion_id=self.command.motion_ids,
      teacher_codes=self.command.teacher_codes,
    )

  def _world_state(self, ids: np.ndarray | None = None) -> WorldState:
    data = self.robot.data
    select = slice(None) if ids is None else torch.as_tensor(ids, device=self.device)
    return WorldState(
      data.root_link_pos_w[select].detach().cpu().numpy(),
      data.root_link_quat_w[select].detach().cpu().numpy(),
      data.root_link_lin_vel_w[select].detach().cpu().numpy(),
      data.root_link_ang_vel_w[select].detach().cpu().numpy(),
      data.body_link_pos_w[select][:, self._body_indexes].detach().cpu().numpy(),
      data.body_link_lin_vel_w[select][:, self._body_indexes].detach().cpu().numpy(),
    )

  def _reference_metrics(self, ids: np.ndarray) -> dict[str, np.ndarray]:
    from mjlab.utils.lab_api.math import quat_apply_inverse

    tensor_ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
    gravity = self.robot.data.gravity_vec_w[tensor_ids]
    anchor_z = (
      (
        self.command.anchor_pos_w[tensor_ids, 2]
        - self.command.robot_anchor_pos_w[tensor_ids, 2]
      )
      .detach()
      .cpu()
      .numpy()
    )
    motion_gravity = quat_apply_inverse(
      self.command.anchor_quat_w[tensor_ids], gravity
    )[:, 2]
    robot_gravity = quat_apply_inverse(
      self.command.robot_anchor_quat_w[tensor_ids], gravity
    )[:, 2]
    gravity_z = (motion_gravity - robot_gravity).detach().cpu().numpy()
    ee_names = _END_EFFECTOR_BODIES
    command_bodies = tuple(self.command.cfg.body_names)
    ee = [command_bodies.index(name) for name in ee_names]
    end_effector = (
      (
        self.command.body_pos_relative_w[tensor_ids][:, ee, 2]
        - self.command.robot_body_pos_w[tensor_ids][:, ee, 2]
      )
      .abs()
      .max(dim=1)
      .values.detach()
      .cpu()
      .numpy()
    )
    return {
      "anchor_z_error": anchor_z,
      "gravity_z_error": gravity_z,
      "end_effector_z_error": end_effector,
    }

  def observe_vae_batch(self, env_ids: np.ndarray) -> Any:
    ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
    snapshot = self._snapshot()
    frames = self.command.time_steps.detach().cpu().numpy()[ids]
    lengths = self.command.row_lengths.detach().cpu().numpy()[ids]
    metrics = self._reference_metrics(ids)
    from mjlab.tasks.tracking.diffusion.adapter import VaeObservationBatch

    return VaeObservationBatch(
      self._world_state(ids),
      snapshot.packed.reference.detach().cpu().numpy()[ids],
      snapshot.packed.conditioning.detach().cpu().numpy()[ids],
      tuple(
        self.teacher_ids[int(code)]
        for code in self.command.motion_ids.detach().cpu().numpy()[ids]
      ),
      frames,
      frames / np.maximum(lengths - 1, 1),
      **metrics,
    )

  def observe_vae_tensor_batch(self, env_ids: np.ndarray) -> Any:
    """Return device-native VAE inputs plus CPU row evidence for the rows.

    One snapshot capture supplies both halves.  The packed
    reference/conditioning tensors are gathered on the simulator device and
    moved once to :attr:`policy_device` as float32, the dtype the frozen VAE
    inference path has always consumed, so they never make a host round trip.
    The row evidence (state, motion id, reference frame and phase) is derived
    for the collector's rows only; it never carries the packed arrays.  Row
    order matches ``env_ids``, so a caller may pass noncontiguous slots.
    """
    ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
    if ids.size == 0 or np.any(ids < 0) or np.any(ids >= self.num_envs):
      raise RuntimeX2Error("tensor VAE batch ids must be non-empty and in range")
    if np.unique(ids).size != ids.size:
      raise RuntimeX2Error("tensor VAE batch ids must not repeat")
    snapshot = self._snapshot()
    tensor_ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
    reference = snapshot.packed.reference[tensor_ids].to(
      dtype=torch.float32, device=self.policy_device
    )
    conditioning = snapshot.packed.conditioning[tensor_ids].to(
      dtype=torch.float32, device=self.policy_device
    )
    frames = self.command.time_steps.detach().cpu().numpy()[ids]
    lengths = self.command.row_lengths.detach().cpu().numpy()[ids]
    return VaeObservationTensorBatch(
      reference,
      conditioning,
      self._world_state(ids),
      tuple(
        self.teacher_ids[int(code)]
        for code in self.command.motion_ids.detach().cpu().numpy()[ids]
      ),
      frames,
      frames / np.maximum(lengths - 1, 1),
    )

  def set_active_env_ids(self, env_ids: np.ndarray) -> None:
    """Declare the trial-owned rows before the next full-batch physics step."""
    self._active_rows[:] = False
    self._active_rows[np.asarray(env_ids, dtype=np.int64)] = True

  def step_batch(self, action: np.ndarray) -> Any:
    # Physics still advances idle rows.  Their terminal signals are masked
    # because no trial owns or consumes the corresponding evidence.
    self._pre_frames = self.command.time_steps.detach().clone()
    tensor = torch.as_tensor(action, dtype=torch.float32, device=self.device)
    self._last_result = self.env.step(tensor)
    self._post_frames = self.command.time_steps.detach().clone()
    return self._last_result

  def step_tensor_batch(self, action: torch.Tensor) -> Any:
    """Advance physics from a device-native action tensor.

    Mirrors :meth:`step_batch` bookkeeping exactly -- ``_pre_frames``,
    ``_last_result`` and ``_post_frames`` are all updated -- but accepts the
    policy action as a tensor and only casts dtype/device when needed, so a
    simulation-device policy action never passes through NumPy.
    """
    if not isinstance(action, torch.Tensor):
      raise RuntimeX2Error("tensor batch step requires a torch.Tensor action")
    if action.shape != (self.num_envs, 31):
      raise RuntimeX2Error(
        f"tensor batch action must have shape [{self.num_envs}, 31], "
        f"got {tuple(action.shape)}"
      )
    if not torch.isfinite(action).all():
      raise RuntimeX2Error("tensor batch action must be finite")
    self._pre_frames = self.command.time_steps.detach().clone()
    values = action.to(device=self.device, dtype=torch.float32)
    self._last_result = self.env.step(values)
    self._post_frames = self.command.time_steps.detach().clone()
    return self._last_result

  def post_step_evidence_batch(self, env_ids: np.ndarray) -> Any:
    ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
    if self._last_result is None:
      raise RuntimeX2Error("post-step evidence requested before a batch step")
    terminated = self._last_result[2].detach().cpu().numpy().astype(bool)[ids]
    truncated = self._last_result[3].detach().cpu().numpy().astype(bool)[ids]
    pre = self._pre_frames.detach().cpu().numpy()[ids]
    post = self._post_frames.detach().cpu().numpy()[ids]
    lengths = self.command.row_lengths.detach().cpu().numpy()[ids]
    # ``pre`` is the frame the recorded row acts at.  The endpoint row is the
    # row on the clip's last frame; there the reference holds instead of
    # wrapping, so a post frame equal to the pre frame is expected, not a
    # teleport.
    boundary = pre + 1 >= lengths
    teleport = ~boundary & (post != pre + 1) & (post != pre)
    from mjlab.tasks.tracking.diffusion.adapter import PostStepBundleBatch
    from mjlab.tasks.tracking.diffusion.qualification import StepEvidenceBatch

    evidence = StepEvidenceBatch(
      np.ones(ids.size, dtype=bool),
      terminated,
      truncated,
      tuple(None for _ in ids),
      np.isfinite(self.robot.data.root_link_pos_w.detach().cpu().numpy()[ids]).all(
        axis=1
      ),
      np.zeros(ids.size, dtype=bool),
      np.zeros(ids.size, dtype=bool),
      np.zeros(ids.size, dtype=bool),
      teleport,
      boundary,
      boundary,
    )
    timestamp = self._trial_clock[ids].copy()
    self._trial_clock[ids] += self.period_seconds
    return PostStepBundleBatch(
      evidence,
      timestamp,
      self.observe_vae_batch(ids),
    )

  def capture_initial_state_batch(
    self, env_ids: np.ndarray
  ) -> tuple[InitialStateSnapshot, ...]:
    """Capture reset parity in scene-local coordinates.

    The simulator places environment rows on a world-frame grid.  A clean/OU
    pair may use different rows, so absolute root positions include different
    ``scene.env_origins`` even when the physical local state is identical.
    Normalize only this parity signature.  Stored trajectory states remain in
    world coordinates.
    """
    ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
    tensor_ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
    data = self.robot.data
    root_position = (
      data.root_link_pos_w[tensor_ids] - self.env.scene.env_origins[tensor_ids]
    )
    result = []
    for index, row in enumerate(ids.tolist()):
      values = np.concatenate(
        (
          root_position[index].detach().cpu().numpy(),
          data.root_link_quat_w[row].detach().cpu().numpy(),
          data.joint_pos[row].detach().cpu().numpy(),
          data.joint_vel[row].detach().cpu().numpy(),
        )
      )
      result.append(
        InitialStateSnapshot(
          0,
          int(self.command.time_steps[row]),
          tuple(float(np.round(value, 12)) for value in values),
        )
      )
    return tuple(result)

  def restore_initial_state_batch(
    self, env_ids: np.ndarray, snapshots: Sequence[InitialStateSnapshot]
  ) -> None:
    actual = self.capture_initial_state_batch(env_ids)
    if len(actual) != len(snapshots) or any(
      not np.allclose(
        np.asarray(current.signature),
        np.asarray(expected.signature),
        rtol=0.0,
        atol=self._RESTORE_PARITY_ATOL,
      )
      for current, expected in zip(actual, snapshots, strict=True)
    ):
      raise AdapterError("OU rows could not reproduce their clean partner states")

  def reset_after_trial_batch(self, env_ids: np.ndarray) -> None:
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
    self.env.reset(env_ids=ids)
    rows = np.asarray(env_ids, dtype=np.int64).reshape(-1)
    self._active_rows[rows] = False

  def close(self) -> None:
    self.env.close()


class X2DiffusionRuntime:
  """Cross-motion single-row runtime implementing the env protocol."""

  def __init__(
    self,
    cohort: Any,
    teacher_ids: Sequence[str],
    *,
    device: str,
  ) -> None:
    self.cohort = cohort
    self.teacher_ids = tuple(teacher_ids)
    self.device = device
    self._envs: dict[str, X2MotionEnvironment] = {}
    self._active: X2MotionEnvironment | None = None

  def environment(self, motion_id: str) -> X2MotionEnvironment:
    """Build (once) and return the single-motion environment for ``motion_id``."""
    if motion_id not in self._envs:
      if motion_id not in self.teacher_ids:
        raise RuntimeX2Error(f"unknown motion {motion_id!r}")
      self._envs[motion_id] = X2MotionEnvironment(
        self.cohort, motion_id, device=self.device, seed=0
      )
    return self._envs[motion_id]

  @property
  def active(self) -> X2MotionEnvironment:
    if self._active is None:
      raise RuntimeX2Error("no trial is active")
    return self._active

  @property
  def periods(self) -> dict[str, float]:
    return {motion_id: env.period_seconds for motion_id, env in self._envs.items()}

  def reset(
    self,
    *,
    seed: int,
    motion_id: str,
    initial_pose: str = "reference",
    perturb: bool = True,
  ) -> None:
    self._active = self.environment(motion_id)
    self._active.reset(seed=seed, initial_pose=initial_pose, perturb=perturb)

  def observe_vae(self) -> VaeObservation:
    return self.active.observe_vae()

  def step(self, action: np.ndarray) -> Any:
    return self.active.step(action)

  def default_row(self, motion_id: str) -> DefaultRow:
    return self.environment(motion_id).default_row()

  def post_step_evidence(self, result: Any) -> PostStepBundle:
    return self.active.post_step_evidence(result)

  def reset_after_trial(self) -> None:
    if self._active is not None:
      self._active.reset_after_trial()

  def capture_initial_state(self) -> InitialStateSnapshot:
    return self.active.capture_initial_state()

  def restore_initial_state(self, snapshot: InitialStateSnapshot) -> None:
    self.active.restore_initial_state(snapshot)

  def close(self) -> None:
    for env in self._envs.values():
      env.close()
    self._envs.clear()
    self._active = None


def _action_contract_from_onnx(path: str | Path) -> ActionContract:
  """Read the normalized-action and actuator contract from one ONNX asset."""
  try:
    import onnx
  except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeX2Error(f"reading artifact metadata requires onnx: {exc}") from exc
  metadata = {entry.key: entry.value for entry in onnx.load(str(path)).metadata_props}
  return ActionContract.from_metadata(metadata)


def _action_contract_for(cohort: Any, teacher_id: str) -> ActionContract:
  """Read the VAE action contract from the selected teacher's ONNX asset."""
  teacher = cohort.teacher(teacher_id)
  return _action_contract_from_onnx(teacher.onnx.path)


def _configure_policy_precision(policy_device: str) -> None:
  """Pin CUDA frozen-policy inference to the CPU-parity IEEE FP32 mode."""
  if torch.device(policy_device).type == "cuda":
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends(allow_tf32=False)


def build_x2_runtime(request: Any) -> Any:
  """``--runtime-factory`` entry point; returns a verified ``RuntimeBundle``.

  The factory resolves the frozen cohort and pinned artifacts itself and fails
  closed on any identity mismatch.  It never chooses the resource budget: the
  CLI has already validated it and passes it in ``request``.
  """
  import os

  from mjlab.tasks.tracking.distillation.adapter import default_schema_for_cohort
  from mjlab.tasks.tracking.distillation.config import load_manifest, resolve_cohort

  from .runtime import RuntimeBundle

  repo_root = Path(__file__).resolve().parents[5]
  device = os.environ.get("MJLAB_D1_DEVICE", "cuda:0")
  # Batched collection follows the simulation device by default. Scalar and
  # explicit CPU requests remain compatible with historical callers.
  policy_default = "simulation" if request.max_envs > 1 else "cpu"
  policy_setting = os.environ.get("MJLAB_D1_POLICY_DEVICE", policy_default)
  policy_device = resolve_policy_device(policy_setting, simulation_device=device)
  # The selected contract owns the pinned artifacts.  The CLI resolves it from
  # ``--contract`` and passes it in the request; the module default is only a
  # fallback for direct library callers.
  selected_contract = getattr(request, "contract", None)
  if selected_contract is None:
    selected_contract = DEFAULT_CONTRACT
  # The collection runtime records raw world states; the window packer (state.py)
  # owns the declared state width.  Read that width here so a reduced 129-D
  # local-kinematic contract is accepted and reported, and an inconsistent one
  # fails before any environment is built.
  try:
    contract_layout: ContractDimensions = contract_dimensions(selected_contract)
  except ValueError as exc:
    raise RuntimeX2Error(
      f"selected collection contract has no usable state layout: {exc}"
    ) from exc

  # Schedule selection is an explicit operational input, never inferred: the
  # frozen contract's pilot schedule must stay the default.
  from functools import partial

  schedule = os.environ.get("MJLAB_D1_SCHEDULE", "pilot").strip().lower()
  worker = int(os.environ.get("MJLAB_D1_WORKER", "0"))
  workers = int(os.environ.get("MJLAB_D1_WORKERS", "1"))
  if schedule == "pilot":
    seed_count = len(PILOT_SEEDS)
    default_run_id = D1_RUN_ID
  elif schedule == "bulk":
    seed_count = int(os.environ.get("MJLAB_D1_BULK_SEEDS", str(BULK_SEED_COUNT)))
    default_run_id = D1_BULK_RUN_ID
  else:
    raise RuntimeX2Error(
      f"MJLAB_D1_SCHEDULE must be 'pilot' or 'bulk', got {schedule!r}"
    )
  run_id = os.environ.get("MJLAB_D1_RUN_ID", "").strip() or default_run_id
  initial_poses = resolve_initial_poses(
    os.environ.get("MJLAB_D1_INITIAL_POSES", "both")
  )
  base_schedule = (
    partial(build_pilot_trials, run_id=run_id, initial_poses=initial_poses)
    if schedule == "pilot"
    else partial(
      build_bulk_trials,
      seed_count=seed_count,
      run_id=run_id,
      initial_poses=initial_poses,
    )
  )
  schedule_trials = partial(base_schedule, worker=worker, workers=workers)

  # The checkpoint's own cohort manifest resolves VAE members; the rollout
  # selection may be any non-empty subset of those members.  Keeping the two
  # separate is what lets a 12-teacher student be rolled out on three clips
  # without redefining its cohort identity, which the checkpoint validates.
  manifest_path = Path(
    os.environ.get("MJLAB_D1_COHORT_MANIFEST", _DEFAULT_COHORT_MANIFEST)
  )
  if not manifest_path.is_absolute():
    manifest_path = repo_root / manifest_path
  if not manifest_path.is_file():
    raise RuntimeX2Error(f"frozen cohort manifest is missing: {manifest_path}")
  manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
  cohort = resolve_cohort(load_manifest(manifest_path, repo_root=repo_root))
  live_schema = default_schema_for_cohort(cohort)
  cohort_ids = tuple(cohort.teacher_ids)
  rollout_ids = resolve_rollout_ids(
    cohort_ids, os.environ.get("MJLAB_D1_ROLLOUT_TEACHERS", "")
  )
  action_contract = _action_contract_for(cohort, rollout_ids[0])
  if tuple(action_contract.joint_order) != tuple(cohort.actions.joint_names):
    raise RuntimeX2Error("artifact joint order disagrees with the cohort contract")

  qualification = QualificationConfig()
  vae = FrozenVaePolicy.from_checkpoint(
    repo_root / selected_contract.vae_checkpoint,
    action_contract,
    teacher_id=rollout_ids[0],
    cohort_manifest=manifest_path,
    cohort_manifest_sha256=manifest_sha256,
    contract=selected_contract,
    device=policy_device,
  )
  runtime: Any
  if request.max_envs > 1:
    runtime: Any = X2MultiMotionEnvironment(
      cohort,
      rollout_ids,
      num_envs=request.max_envs,
      device=device,
      seed=0,
      policy_device=policy_device,
    )
    batch_runtime = True
  else:
    runtime = X2DiffusionRuntime(cohort, rollout_ids, device=device)
    for teacher_id in rollout_ids:
      env = runtime.environment(teacher_id)
      if env.joint_names != tuple(action_contract.joint_order):
        raise RuntimeX2Error(
          f"compiled joint order for {teacher_id} disagrees with the action contract"
        )
    batch_runtime = False
  # Task registration can enable TF32 globally. Frozen VAE parity with the
  # historical CPU path requires IEEE FP32; pin it after runtime construction.
  _configure_policy_precision(policy_device)

  collector = DiffusionCollector(
    runtime,
    vae,
    vae_actions=action_contract,
    qualification=qualification,
    store=request.store,
  )
  trials = schedule_trials(
    [
      (teacher_id, cohort.teacher(teacher_id).reference.frames)
      for teacher_id in rollout_ids
    ]
  )
  # A resumed collection skips the trials a previous run already stored.  The
  # schedule is deterministic, so a worker's shard is a stable prefix of the
  # work it owes and the offset is the number of trials that worker finished.
  worker_offset = resolve_worker_offset(
    os.environ.get("MJLAB_D1_WORKER_OFFSET", ""), worker, workers
  )
  if worker_offset:
    if worker_offset > len(trials):
      raise RuntimeX2Error(
        f"MJLAB_D1_WORKER_OFFSET {worker_offset} exceeds this worker's "
        f"{len(trials)} scheduled trials"
      )
    trials = trials[worker_offset:]
  if len(trials) > request.max_trials:
    raise RuntimeX2Error(
      f"collection schedule has {len(trials)} trials, budget is {request.max_trials}"
    )

  return RuntimeBundle(
    collector=collector,
    trials=trials,
    runtime_verified=True,
    runtime_provenance={
      "environment": "agibot_x2",
      "max_envs": request.max_envs,
      "max_gpus": request.max_gpus,
      "worker_index": worker,
      "worker_count": workers,
      "worker_offset": worker_offset,
      "device": device,
      "policy_device": policy_device,
      "policy_device_setting": policy_setting,
      "control_hz": int(round(1.0 / qualification.period_seconds)),
      "cohort_manifest": str(manifest_path),
      "cohort_manifest_sha256": manifest_sha256,
      "cohort_ids": list(cohort_ids),
      "rollout_teachers": list(rollout_ids),
      "vae_checkpoint": selected_contract.vae_checkpoint,
      "vae_sha256": selected_contract.vae_sha256,
      "vae_schema": live_schema.compatibility_metadata(),
      "cohort_lookahead_s": float(cohort.lookahead_s),
      "state_dimension": contract_layout.state_dimension,
      "token_dimension": contract_layout.token_dimension,
      "window_steps": contract_layout.window_steps,
      "run_id": run_id,
      "schedule": schedule,
      "seeds": seed_count,
      "initial_poses": list(initial_poses),
      "padding_steps": qualification.padding_steps,
      "trials": len(trials),
      "worker": worker,
      "workers": workers,
      "batched": batch_runtime,
    },
  )


__all__ = [
  "BULK_SEED_COUNT",
  "D1_BULK_RUN_ID",
  "D1_RUN_ID",
  "DefaultRow",
  "INITIAL_POSES",
  "INITIAL_STATE_MODES",
  "InitialPerturbation",
  "InitialStateSnapshot",
  "PILOT_SEEDS",
  "RuntimeX2Error",
  "X2DiffusionRuntime",
  "X2MultiMotionEnvironment",
  "X2MotionEnvironment",
  "anchor_frames_for",
  "build_bulk_trials",
  "build_pilot_trials",
  "build_trials",
  "build_x2_runtime",
  "initial_perturbation",
  "resolve_initial_poses",
  "resolve_policy_device",
  "resolve_rollout_ids",
  "stand_body_positions",
  "stand_joint_positions",
  "stand_reference",
]
