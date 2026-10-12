"""Concrete AgiBot X2 runtime factory for D1 pilot collection.

This module owns the simulator boundary the offline D1 code deliberately does
not: it builds the registered tracking environment, drives the frozen
version-3 cohort VAE and the frozen zero-command recovery policy, and reports
physical post-step evidence.

Design constraints taken from the frozen D0 contract
---------------------------------------------------

* A shared multi-motion environment may carry many rows.  Rows pin their own
  reference clip and the collector owns every reset boundary.
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
  VaeObservationTensorBatch,
  resolve_inference_device,
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

# Frozen three-clip cohort used by both frozen revisions' pilot schedule.
_DEFAULT_COHORT_MANIFEST = "configs/distillation/x2_tennis_mixed.yaml"

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
  """Build a pair-local clean/OU schedule for the given seeds.

  Each OU item follows its own clean item.  Unrelated pairs may interleave.  The
  old global clean-before-OU ordering protected the single-row snapshot map, but
  the collector now checks the matching fingerprint and qualification at each
  pair admission, so that global restriction is not required.

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
  run_id: str = D1_BULK_RUN_ID,
  worker: int = 0,
  workers: int = 1,
) -> tuple[TrialSpec, ...]:
  """The ~100-fold coverage schedule (``seed_count`` seeds per phase/clip)."""
  if seed_count <= 0:
    raise RuntimeX2Error("bulk collection needs at least one seed")
  return build_trials(
    motion_frames,
    seeds=tuple(range(seed_count)),
    run_id=run_id,
    worker=worker,
    workers=workers,
  )


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


class X2MultiMotionEnvironment:
  """Shared X2 environment exposing the collector's vectorized row protocol."""

  _RESTORE_PARITY_ATOL = 1e-6

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
    self._phase_ppo = np.zeros(self.num_envs, dtype=bool)
    self._handoff_frame = np.zeros(self.num_envs, dtype=np.int64)
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
    )
    from mjlab.tasks.tracking.distillation.vae_config import make_schema

    self._capture_snapshot = _capture_snapshot
    self._slices = _term_slices(cohort)
    self._schema = make_schema(joint_order=cohort.actions.joint_names)
    self._install_termination_masks()
    self._install_reference_advance_mask()

  def _install_reference_advance_mask(self) -> None:
    """Hold recovery frames without invoking the motion wrap/teleport path."""
    update = self.command._update_command

    def advance(env_ids: torch.Tensor | None = None) -> None:
      candidates = self._row_ids if env_ids is None else env_ids
      enabled = torch.as_tensor(
        self._active_rows & self._vae_phase, dtype=torch.bool, device=self.device
      )
      update(candidates[enabled[candidates]])

    self.command._update_command = advance

  def _install_termination_masks(self) -> None:
    """Keep idle physics rows and recovery references out of termination."""
    self._reference_terms: list[tuple[Any, dict[str, Any]]] = []
    self._vae_phase = np.ones(self.num_envs, dtype=bool)
    self._active_rows = np.zeros(self.num_envs, dtype=bool)
    manager = self.env.termination_manager
    for name in _REFERENCE_TERMINATIONS:
      if name not in manager.active_terms:
        raise RuntimeX2Error(f"tracking task is missing termination {name!r}")
    for name in manager.active_terms:
      cfg = manager.get_term_cfg(name)
      original, params = cfg.func, dict(cfg.params)
      is_reference = name in _REFERENCE_TERMINATIONS
      if is_reference:
        self._reference_terms.append((original, params))

      def masked(
        env: Any,
        _original: Any = original,
        _reference: bool = is_reference,
        **kwargs: Any,
      ) -> Any:
        enabled = self._active_rows
        if _reference:
          enabled = enabled & self._vae_phase
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
    self._phase_ppo[slots] = False
    self._vae_phase[slots] = True
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
    self._phase_ppo[slots] = False
    self._vae_phase[slots] = True
    self._handoff_frame[slots] = np.asarray(
      [spec.start_frame for spec in specs], dtype=np.int64
    )
    self._trial_clock[slots] = 0.0
    self._pre_frames[ids] = self.command.time_steps[ids]

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

  def _recovery_observation(self, ids: np.ndarray) -> np.ndarray:
    from mjlab.envs import mdp as envs_mdp

    tensor_ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
    gyro = envs_mdp.builtin_sensor(self.env, sensor_name="robot/imu_ang_vel")[
      tensor_ids
    ]
    gravity = envs_mdp.projected_gravity_from_sensor(
      self.env, sensor_name="robot/imu_upvector"
    )[tensor_ids]
    joint_pos = envs_mdp.joint_pos_rel(self.env, biased=True)[tensor_ids]
    joint_vel = envs_mdp.joint_vel_rel(self.env)[tensor_ids]
    actions = envs_mdp.last_action(self.env)[tensor_ids]
    return (
      torch.cat(
        (gyro, gravity, joint_pos, joint_vel, actions, torch.zeros_like(gyro)), dim=1
      )
      .detach()
      .cpu()
      .numpy()
    )

  def observe_ppo_batch(self, env_ids: np.ndarray) -> Any:
    ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
    frames = self._handoff_frame[ids]
    from mjlab.tasks.tracking.diffusion.adapter import PpoObservationBatch

    return PpoObservationBatch(
      self._world_state(ids),
      self._recovery_observation(ids),
      tuple(
        self.teacher_ids[int(code)]
        for code in self.command.motion_ids.detach().cpu().numpy()[ids]
      ),
      frames,
      frames / np.maximum(self.command.row_lengths.detach().cpu().numpy()[ids] - 1, 1),
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
    vae = ~self._phase_ppo[ids]
    boundary = vae & ((pre + 1 >= lengths) | (post >= lengths - 1))
    teleport = vae & (~boundary) & (post != pre + 1)
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

  def switch_to_ppo_batch(self, env_ids: np.ndarray) -> None:
    ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
    self._phase_ppo[ids] = True
    self._vae_phase[ids] = False
    self._handoff_frame[ids] = self._post_frames.detach().cpu().numpy()[ids]

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
    self._phase_ppo[rows] = False
    self._vae_phase[rows] = True

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
  base_schedule = (
    partial(build_pilot_trials, run_id=run_id)
    if schedule == "pilot"
    else partial(build_bulk_trials, seed_count=seed_count, run_id=run_id)
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
  ppo_action_contract = _action_contract_from_onnx(
    repo_root / selected_contract.recovery_onnx
  )
  ppo = FrozenPpoRecoveryPolicy.from_onnx(
    repo_root / selected_contract.recovery_onnx, ppo_action_contract
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
    ppo,
    vae_actions=action_contract,
    ppo_actions=ppo_action_contract,
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
      "run_id": run_id,
      "schedule": schedule,
      "seeds": seed_count,
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
  "resolve_policy_device",
  "resolve_rollout_ids",
  "PHASE_FRACTIONS",
  "PILOT_SEEDS",
  "InitialStateSnapshot",
  "RuntimeX2Error",
  "X2DiffusionRuntime",
  "X2MultiMotionEnvironment",
  "X2MotionEnvironment",
  "build_bulk_trials",
  "build_pilot_trials",
  "build_trials",
  "build_x2_runtime",
  "start_frames_for",
]
