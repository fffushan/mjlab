"""Concrete AgiBot X2 runtime factory for D1 pilot collection.

This module owns the simulator boundary the offline D1 code deliberately does
not: it builds the registered tracking environment, drives the frozen
version-3 cohort VAE and the frozen zero-command recovery policy, and reports
physical post-step evidence.

Design constraints taken from the frozen D0 contract
---------------------------------------------------

* ``max_envs == 1``.  The multi-motion builder cannot represent a three-clip
  cohort in one row, so this factory builds **one single-motion environment per
  clip** and reuses it across that clip's trials.
* No mid-trial reset.  ``auto_reset`` is disabled and every trial start is a
  controlled reset plus a deterministic ``reset_to_frame`` teleport.
* The VAE phase ends at the clip's last reference frame, where control hands to
  the zero-command recovery policy in the *same* simulator state (continuation,
  never a state transplant).
* Physical fall is the tracking task's own root-quantum tilt criterion applied
  in both phases by the collector's qualification tracker; the VAE reference
  guards apply only while the VAE owns control.
* Recovery rows are not training data; no window may cross the handoff.

The module is intentionally not a CLI: :func:`build_x2_runtime` is the
``--runtime-factory`` entry point, and the CLI has already applied its
fail-closed budget and artifact checks before calling it.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .adapter import (
  ActionContract,
  AdapterError,
  FrozenPpoRecoveryPolicy,
  FrozenVaePolicy,
  PostStepBundle,
  PpoObservation,
  VaeObservation,
)
from .collector import DiffusionCollector, TrialSpec
from .contract import DEFAULT_CONTRACT
from .qualification import QualificationConfig, StepEvidence
from .state import WorldState

# Frozen D1 pilot design (docs/plans/beyondmimic_diffusion_d0_contract.yaml).
PHASE_FRACTIONS: tuple[float, ...] = (0.1, 0.5, 0.8)
PILOT_SEEDS: tuple[int, ...] = (0, 1, 2)
D1_RUN_ID = "d1-pilot-x2-tennis-mixed-v1"
D1_BULK_RUN_ID = "d1-bulk-x2-tennis-mixed-v1"

# The plan sizes the bulk collection as ``L * 50 * 100`` retained rows
# (~111300 for the three clips).  At the pilot yield of ~108 accepted rows per
# trial that is ~1030 trials, i.e. 57 seeds x 3 phases x 3 clips pairs.
BULK_SEED_COUNT = 57

_END_EFFECTOR_BODIES: tuple[str, ...] = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)

_BODY_COUNT = len(DEFAULT_CONTRACT.body_names)


_REFERENCE_TERMINATIONS: tuple[str, ...] = ("anchor_pos", "anchor_ori", "ee_body_pos")


def _never_terminate(env: Any, **_: Any) -> torch.Tensor:
  """Always-false termination, used to suspend reference guards."""
  return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


class RuntimeX2Error(RuntimeError):
  """The concrete X2 runtime cannot be built or driven safely."""


def start_frames_for(frames: int) -> tuple[int, ...]:
  """Contract start frames ``floor(fraction * (N - 1))`` for one clip."""
  if frames < 2:
    raise RuntimeX2Error(f"clip must have at least two frames, got {frames}")
  return tuple(
    min(frames - 1, int(math.floor(fraction * (frames - 1))))
    for fraction in PHASE_FRACTIONS
  )


def build_trials(
  motion_frames: Sequence[tuple[str, int]],
  *,
  seeds: Sequence[int],
  run_id: str,
  worker: int = 0,
  workers: int = 1,
) -> tuple[TrialSpec, ...]:
  """Build a clean-then-OU schedule for the given seeds.

  The CLI requires the whole clean phase to precede every OU trial, exactly one
  OU partner per clean pair, and an identical initialization fingerprint per
  pair.  Trial order is therefore deterministic: every clean pair first, then
  the matched OU partners in the same order.

  ``worker``/``workers`` take a disjoint stride slice of whole pairs so several
  processes can collect in parallel.  Each worker still receives every one of
  its own clean trials before any of its own OU trials, which is what the
  collector and the CLI require, and a pair is never split across workers
  because the OU restore needs its clean partner in the same process.

  ``group_key`` is set explicitly to the pair family
  (``motion:seed:start_frame``).  The collector would otherwise synthesize
  ``motion:start_frame``, which carries no seed and collapses every seed of a
  start frame into one split group -- enough for a 27-pair pilot (9 groups,
  empty validation) but not for a dataset that needs a populated validation
  partition.  The explicit key is the family itself, so clean and OU partners
  still share a split and the pair relation cannot leak.
  """
  if workers < 1 or not 0 <= worker < workers:
    raise RuntimeX2Error(
      f"worker must satisfy 0 <= worker < workers, got {worker}/{workers}"
    )
  pairs = [
    (motion_id, seed, start_frame)
    for motion_id, frames in motion_frames
    for seed in seeds
    for start_frame in start_frames_for(frames)
  ]
  clean: list[TrialSpec] = []
  ou: list[TrialSpec] = []
  for motion_id, seed, start_frame in pairs[worker::workers]:
    pair_id = f"{motion_id}:{seed}:{start_frame}"
    # Each seed produces its own perturbed initial state, so the state
    # identity must carry the seed too.
    initial_state_id = f"init:{motion_id}:{seed}:{start_frame}"
    clean.append(
      TrialSpec(
        run_id=run_id,
        motion_id=motion_id,
        seed=seed,
        start_frame=start_frame,
        phase="clean",
        group_key=pair_id,
        pair_id=pair_id,
        initial_state_id=initial_state_id,
      )
    )
    ou.append(
      TrialSpec(
        run_id=run_id,
        motion_id=motion_id,
        seed=seed,
        start_frame=start_frame,
        phase="ou",
        group_key=pair_id,
        pair_id=pair_id,
        initial_state_id=initial_state_id,
      )
    )
  return tuple(clean) + tuple(ou)


def build_pilot_trials(
  motion_frames: Sequence[tuple[str, int]],
  *,
  run_id: str = D1_RUN_ID,
  worker: int = 0,
  workers: int = 1,
) -> tuple[TrialSpec, ...]:
  """The frozen 54-trial pilot schedule (seeds 0-2)."""
  return build_trials(
    motion_frames, seeds=PILOT_SEEDS, run_id=run_id, worker=worker, workers=workers
  )


def build_bulk_trials(
  motion_frames: Sequence[tuple[str, int]],
  *,
  seed_count: int = BULK_SEED_COUNT,
  worker: int = 0,
  workers: int = 1,
) -> tuple[TrialSpec, ...]:
  """The ~100-fold coverage schedule (``seed_count`` seeds per phase/clip)."""
  if seed_count <= 0:
    raise RuntimeX2Error("bulk collection needs at least one seed")
  return build_trials(
    motion_frames,
    seeds=tuple(range(seed_count)),
    run_id=D1_BULK_RUN_ID,
    worker=worker,
    workers=workers,
  )


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

    # The tracking task's own reference guards must apply while the VAE tracks and
    # must be suspended once the zero-command recovery policy owns the trial:
    # a standing policy cannot follow a reference the clip no longer advances.
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
    self._set_reference_terminations(True)

    self._phase = "vae"
    self._last_frame = 0
    self._handoff_frame = 0
    self._trial_clock = 0.0
    self._initial: InitialStateSnapshot | None = None

  # -- lifecycle -----------------------------------------------------------

  def reset(self, *, seed: int, start_frame: int) -> None:
    """Full reset, then a deterministic teleport to the trial start frame."""
    if start_frame < 0 or start_frame >= self.clip_frames:
      raise RuntimeX2Error(
        f"start frame {start_frame} is outside clip length {self.clip_frames}"
      )
    self.env.reset(seed=seed)
    self.command.reset_to_frame(self._row_ids, int(start_frame))
    # Replay the environment's own post-reset recompute tail so the observation
    # cache, history, and sensor data describe the teleported state.  The
    # command's own update is deliberately skipped: ``_update_command``
    # increments ``time_steps`` unconditionally, which would land on
    # ``start_frame + 1``.
    self.env.scene.write_data_to_sim()
    self.env.sim.forward()
    self.env.sim.sense()
    # ``reset_to_frame`` changes ``time_steps`` after the environment's own
    # command compute, so the anchor-relative reference cache (used by the
    # tracking task's ``body_pos_relative_w`` termination and by the D1
    # reference guard) is stale until it is refreshed here.
    self.command.update_relative_body_poses()
    self.env.obs_buf = self.env.observation_manager.compute(
      update_history=True, env_ids=self._row_ids
    )
    self._phase = "vae"
    self._last_frame = int(start_frame)
    self._handoff_frame = int(start_frame)
    self._trial_clock = 0.0
    self._set_reference_terminations(True)
    self._initial = InitialStateSnapshot(
      seed=int(seed),
      start_frame=int(start_frame),
      signature=self._initial_signature(),
    )

  def capture_initial_state(self) -> InitialStateSnapshot:
    if self._initial is None:
      raise RuntimeX2Error("no reset state has been captured yet")
    return self._initial

  def restore_initial_state(self, snapshot: InitialStateSnapshot) -> None:
    current = self._initial_signature()
    if tuple(current) != tuple(snapshot.signature):
      raise AdapterError(
        "OU trial could not reproduce its clean partner's initial state; "
        f"expected {snapshot.digest()}"
      )

  def reset_after_trial(self) -> None:
    """Return the simulator to a clean, non-pending state for the next trial."""
    self.env.reset(env_ids=self._row_ids)
    self._phase = "vae"

  def close(self) -> None:
    self.adapter.close()

  # -- observations --------------------------------------------------------

  def observe_vae(self) -> VaeObservation:
    if self._phase != "vae":
      raise RuntimeX2Error(
        "VAE observation requested while the recovery policy owns control"
      )
    return self._vae_observation(self.adapter.snapshot())

  def observe_ppo(self) -> PpoObservation:
    if self._phase != "ppo":
      raise RuntimeX2Error("recovery observation requested before the handoff")
    return self._ppo_observation(self.adapter.snapshot())

  def switch_to_ppo(self) -> None:
    # The reference has ended: freeze its frame so the recovery phase reports the
    # handoff frame, and suspend the tracking task's reference guards, which the
    # frozen contract marks inactive after the handoff.
    self._handoff_frame = self._last_frame
    self._set_reference_terminations(False)
    self._phase = "ppo"

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
    # The reference drives neither control nor the guard once the recovery policy
    # owns the trial, so its wrap is not an unexpected reference teleport.
    if self._phase == "vae":
      clip_end = frame >= self.clip_frames - 1
      teleport = frame != self._last_frame + 1
      self._last_frame = frame
    else:
      clip_end = False
      teleport = False
    observation = (
      self._vae_observation(snapshot)
      if self._phase == "vae"
      else self._ppo_observation(snapshot)
    )
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
      reference=_as_numpy(snapshot.packed.reference[0], 68),
      conditioning=_as_numpy(snapshot.packed.conditioning[0], 99),
      motion_id=self.teacher_id,
      reference_frame=int(snapshot.reference_frame[0].item()),
      reference_phase=self._reference_phase(int(snapshot.reference_frame[0].item())),
      **metrics,
    )

  def _ppo_observation(self, snapshot: Any) -> PpoObservation:
    frame = self._handoff_frame
    return PpoObservation(
      state=self._world_state(),
      observation=self._recovery_observation(),
      motion_id=self.teacher_id,
      reference_frame=frame,
      reference_phase=self._reference_phase(frame),
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

  def _recovery_observation(self) -> np.ndarray:
    """The frozen recovery policy's 102-D observation.

    Built with the *same* mjlab observation terms the policy was trained with in
    the no-state-estimation velocity task: the IMU gyro and up-vector sensors,
    the **biased** relative joint position (encoder bias included), relative
    joint velocity, the last applied action, and a **zero** standing command
    (never a tracking command).
    """
    from mjlab.envs import mdp as envs_mdp

    joint_count = len(self.joint_names)
    ang_vel = _as_numpy(
      envs_mdp.builtin_sensor(self.env, sensor_name="robot/imu_ang_vel")[0], 3
    )
    gravity = _as_numpy(
      envs_mdp.projected_gravity_from_sensor(
        self.env, sensor_name="robot/imu_upvector"
      )[0],
      3,
    )
    joint_pos = _as_numpy(envs_mdp.joint_pos_rel(self.env, biased=True)[0], joint_count)
    joint_vel = _as_numpy(envs_mdp.joint_vel_rel(self.env)[0], joint_count)
    actions = _as_numpy(envs_mdp.last_action(self.env)[0], joint_count)
    return np.concatenate(
      (ang_vel, gravity, joint_pos, joint_vel, actions, np.zeros(3, dtype=np.float64))
    )


class X2DiffusionRuntime:
  """Cross-motion single-row runtime implementing the collector's env protocol."""

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

  def reset(self, *, seed: int, motion_id: str, start_frame: int) -> None:
    self._active = self.environment(motion_id)
    self._active.reset(seed=seed, start_frame=start_frame)

  def observe_vae(self) -> VaeObservation:
    return self.active.observe_vae()

  def observe_ppo(self) -> PpoObservation:
    return self.active.observe_ppo()

  def step(self, action: np.ndarray) -> Any:
    return self.active.step(action)

  def switch_to_ppo(self) -> None:
    self.active.switch_to_ppo()

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


def _action_contract_for(cohort: Any, teacher_id: str) -> ActionContract:
  teacher = cohort.teacher(teacher_id)
  try:
    import onnx
  except ImportError as exc:  # pragma: no cover - recovery loading needs it too
    raise RuntimeX2Error(f"reading artifact metadata requires onnx: {exc}") from exc
  metadata = {
    entry.key: entry.value for entry in onnx.load(str(teacher.onnx.path)).metadata_props
  }
  return ActionContract.from_metadata(metadata)


def build_x2_runtime(request: Any) -> Any:
  """``--runtime-factory`` entry point; returns a verified ``RuntimeBundle``.

  The factory resolves the frozen cohort and pinned artifacts itself and fails
  closed on any identity mismatch.  It never chooses the resource budget: the
  CLI has already validated it and passes it in ``request``.
  """
  import os

  from mjlab.tasks.tracking.distillation.config import load_manifest, resolve_cohort

  from .runtime import RuntimeBundle

  repo_root = Path(__file__).resolve().parents[5]
  device = os.environ.get("MJLAB_D1_DEVICE", "cuda:0")

  # Schedule selection is an explicit operational input, never inferred: the
  # frozen contract's pilot schedule must stay the default.
  from functools import partial

  schedule = os.environ.get("MJLAB_D1_SCHEDULE", "pilot").strip().lower()
  worker = int(os.environ.get("MJLAB_D1_WORKER", "0"))
  workers = int(os.environ.get("MJLAB_D1_WORKERS", "1"))
  if schedule == "pilot":
    base_schedule = build_pilot_trials
    seed_count = len(PILOT_SEEDS)
  elif schedule == "bulk":
    seed_count = int(os.environ.get("MJLAB_D1_BULK_SEEDS", str(BULK_SEED_COUNT)))
    base_schedule = partial(build_bulk_trials, seed_count=seed_count)
  else:
    raise RuntimeX2Error(
      f"MJLAB_D1_SCHEDULE must be 'pilot' or 'bulk', got {schedule!r}"
    )
  schedule_trials = partial(base_schedule, worker=worker, workers=workers)
  manifest_path = repo_root / "configs/distillation/x2_tennis_mixed.yaml"
  if not manifest_path.is_file():
    raise RuntimeX2Error(f"frozen cohort manifest is missing: {manifest_path}")
  manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
  cohort = resolve_cohort(load_manifest(manifest_path, repo_root=repo_root))
  teacher_ids = tuple(cohort.teacher_ids)
  if len(teacher_ids) != 3:
    raise RuntimeX2Error(
      f"D1 pilot expects the frozen 3-member cohort, found {teacher_ids}"
    )

  action_contract = _action_contract_for(cohort, teacher_ids[0])
  if tuple(action_contract.joint_order) != tuple(cohort.actions.joint_names):
    raise RuntimeX2Error("artifact joint order disagrees with the cohort contract")

  qualification = QualificationConfig()
  vae = FrozenVaePolicy.from_checkpoint(
    repo_root / DEFAULT_CONTRACT.vae_checkpoint,
    action_contract,
    teacher_id=teacher_ids[0],
  )
  ppo = FrozenPpoRecoveryPolicy.from_onnx(
    repo_root / DEFAULT_CONTRACT.recovery_onnx, action_contract
  )

  runtime = X2DiffusionRuntime(cohort, teacher_ids, device=device)
  for teacher_id in teacher_ids:
    env = runtime.environment(teacher_id)
    if env.joint_names != tuple(action_contract.joint_order):
      raise RuntimeX2Error(
        f"compiled joint order for {teacher_id} disagrees with the action contract"
      )

  collector = DiffusionCollector(
    runtime,
    vae,
    ppo,
    vae_actions=action_contract,
    ppo_actions=action_contract,
    qualification=qualification,
    store=request.store,
  )
  trials = schedule_trials(
    [
      (teacher_id, cohort.teacher(teacher_id).reference.frames)
      for teacher_id in teacher_ids
    ]
  )
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
      "device": device,
      "control_hz": int(round(1.0 / qualification.period_seconds)),
      "cohort_manifest": "configs/distillation/x2_tennis_mixed.yaml",
      "cohort_manifest_sha256": manifest_sha256,
      "schedule": schedule,
      "seeds": seed_count,
      "trials": len(trials),
      "worker": worker,
      "workers": workers,
    },
  )


__all__ = [
  "BULK_SEED_COUNT",
  "D1_BULK_RUN_ID",
  "D1_RUN_ID",
  "PHASE_FRACTIONS",
  "PILOT_SEEDS",
  "InitialStateSnapshot",
  "RuntimeX2Error",
  "X2DiffusionRuntime",
  "X2MotionEnvironment",
  "build_bulk_trials",
  "build_pilot_trials",
  "build_trials",
  "build_x2_runtime",
  "start_frames_for",
]
