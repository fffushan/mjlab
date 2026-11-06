"""Capture additions for the X2 nominal juggling E0 evaluation.

Two things are added to the resolved training config, and nothing else:

1. **One read-only contact sensor**, ``ball_any_contact``, over the ball body with
   no secondary filter, so every contact the ball has with anything is visible at
   1 ms resolution. It is what makes "no intervening ground or non-racket contact"
   a direct contact measurement instead of a height proxy.
2. **One recorder term**, ``nominal_trace``, which copies the ball state, the plate
   pose and both contact histories into preallocated buffers, once per control step.

Neither addition changes the physics: contact sensors read contacts that already
exist, and the recorder only reads environment state. The saved training config had
no recorder, and the comparison in :mod:`config_check` refuses to run if anything
else differs from it.

**Contact classification limit.** The non-racket flag is
``any_contact and not racket_contact`` at substep resolution. If the ball touches
the racket and something else in the same 1 ms substep, that substep is attributed
to the racket and the other contact is hidden. A ball caught between the plate and
the floor is a hold or a failure on its own, so the limit is recorded rather than
corrected; there is no per-pair decomposition available from the sensors.

**Terminal state.** The environment computes termination, reward and metrics and
then calls ``record_pre_reset`` before ``_reset_idx``. That hook is where the
terminal row is captured, so no post-reset state can be scored. Ball position and
velocity are read through the same entity accessors as every other control step;
those are derived quantities that ``env.step()`` refreshes with a single
``sim.forward()`` *after* the reset block, so at ``record_pre_reset`` they lag the
integrated state by one 1 ms substep. Termination and reward see the same lag, so
the recorded terminal row and the termination decision are consistent with each
other. The alternative - reading ``qpos``/``qvel`` directly - would be one
substep fresher than the termination decision and is deliberately not used.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from config_check import ANY_CONTACT_SENSOR_NAME, RECORDER_TERM_NAME

from mjlab.asset_zoo.robots.agibot_x2.tennis_constants import (
  FACE_SITE,
)
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.config.agibot_x2.tennis_juggle_env_cfg import (
  BALL_ENTITY,
  BAT_BALL_CONTACT_SENSOR,
)
from mjlab.tasks.velocity.mdp.ball_contact import substep_contact
from mjlab.utils.lab_api.math import matrix_from_quat

ROBOT_ENTITY = "robot"

FLOAT_FIELDS = ("ball_pos_w", "ball_vel_w", "plate_pos_w", "plate_axes_w")
BOOL_FIELDS = ("racket_substeps", "other_substeps", "active")


def add_capture_additions(cfg: ManagerBasedRlEnvCfg) -> ContactSensorCfg:
  """Append the read-only any-contact sensor and the trace recorder to a config.

  Must run before the environment is built: both are part of the scene or the
  manager set, not something that can be attached to a live environment.

  Args:
    cfg: The resolved environment config to modify in place.

  Returns:
    The sensor config that was appended.

  Raises:
    ValueError: If the sensor or the recorder term is already present, or if the
      racket contact sensor's history does not cover exactly one control step.
  """
  sensor_names = [sensor.name for sensor in (cfg.scene.sensors or ())]
  if ANY_CONTACT_SENSOR_NAME in sensor_names:
    raise ValueError(f"{ANY_CONTACT_SENSOR_NAME!r} is already installed")
  if BAT_BALL_CONTACT_SENSOR not in sensor_names:
    raise ValueError(
      f"expected the task's {BAT_BALL_CONTACT_SENSOR!r} sensor; found {sensor_names}"
    )
  racket = next(s for s in cfg.scene.sensors or () if s.name == BAT_BALL_CONTACT_SENSOR)
  if racket.history_length != cfg.decimation:
    raise ValueError(
      f"{BAT_BALL_CONTACT_SENSOR!r} keeps {racket.history_length} substeps but the "
      f"control step is {cfg.decimation}; the recording would not cover one step"
    )

  sensor_cfg = ContactSensorCfg(
    name=ANY_CONTACT_SENSOR_NAME,
    primary=ContactMatch(mode="body", pattern="ball", entity=BALL_ENTITY),
    secondary=None,
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=cfg.decimation,
  )
  cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (sensor_cfg,)

  if RECORDER_TERM_NAME in cfg.recorders:
    raise ValueError(f"recorder term {RECORDER_TERM_NAME!r} is already installed")
  cfg.recorders[RECORDER_TERM_NAME] = RecorderTermCfg(func=NominalTraceRecorder)
  return sensor_cfg


class NominalTraceRecorder(RecorderTerm):
  """Copy one control step of trajectory and contact state into device buffers.

  The first episode of every environment is recorded; automatic-reset episodes are
  not. The buffers are allocated once from the environment's episode length, so a
  whole arm costs one allocation and one host copy at the end, not one copy per
  step.
  """

  def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._num_envs = int(env.num_envs)
    self._max_steps = int(env.max_episode_length)
    self._substeps = int(env.cfg.decimation)
    device = env.device
    self._device = device

    def buffer(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
      return torch.zeros(shape, dtype=dtype, device=device)

    self._ball_pos = buffer(self._max_steps, self._num_envs, 3)
    self._ball_vel = buffer(self._max_steps, self._num_envs, 3)
    self._plate_pos = buffer(self._max_steps, self._num_envs, 3)
    self._plate_axes = buffer(self._max_steps, self._num_envs, 3, 3)
    self._racket = buffer(
      self._max_steps, self._num_envs, self._substeps, dtype=torch.bool
    )
    self._other = buffer(
      self._max_steps, self._num_envs, self._substeps, dtype=torch.bool
    )
    self._active = buffer(self._max_steps, self._num_envs, dtype=torch.bool)

    self._initial_ball_pos = buffer(self._num_envs, 3)
    self._initial_ball_vel = buffer(self._num_envs, 3)
    self._initial_plate_pos = buffer(self._num_envs, 3)
    self._initial_plate_axes = buffer(self._num_envs, 3, 3)
    self._reset_count = buffer(self._num_envs, dtype=torch.long)
    self._pre_step_resets = buffer(self._num_envs, dtype=torch.long)

    self._term_names = list(env.termination_manager.active_terms)
    self._term_flags = buffer(self._num_envs, len(self._term_names), dtype=torch.bool)
    self._ended = buffer(self._num_envs, dtype=torch.bool)
    self._end_step = torch.full((self._num_envs,), -1, dtype=torch.long, device=device)
    self._step = 0

    self._ball = env.scene[BALL_ENTITY]
    self._robot = env.scene[ROBOT_ENTITY]
    self._plate_site_id = int(self._robot.find_sites(FACE_SITE)[0][0])
    self._racket_sensor = env.scene[BAT_BALL_CONTACT_SENSOR]
    self._any_sensor = env.scene[ANY_CONTACT_SENSOR_NAME]

  # Environment hooks.

  def record_post_reset(self, env_ids: torch.Tensor) -> None:
    """Capture the release state produced by the final seeded reset."""
    self._reset_count[env_ids] += 1
    if self._step != 0:
      # A mid-run auto-reset. Those episodes are not scored.
      return
    self._pre_step_resets[env_ids] += 1
    ball_pos, ball_vel, plate_pos, plate_axes = self._sample_state()
    self._initial_ball_pos[env_ids] = ball_pos[env_ids]
    self._initial_ball_vel[env_ids] = ball_vel[env_ids]
    self._initial_plate_pos[env_ids] = plate_pos[env_ids]
    self._initial_plate_axes[env_ids] = plate_axes[env_ids]

  def record_pre_reset(self, env_ids: torch.Tensor) -> None:
    """Capture the terminal row before ``_reset_idx`` overwrites the state."""
    live = env_ids[~self._ended[env_ids]]
    if live.numel() == 0:
      return
    self._write_rows(live, self._step)
    for index, name in enumerate(self._term_names):
      self._term_flags[live, index] = self._env.termination_manager.get_term(name)[live]
    self._end_step[live] = self._step
    self._ended[live] = True

  def record_post_step(self) -> None:
    """Capture the row for every environment still on its first episode."""
    live = (~self._ended).nonzero(as_tuple=False).squeeze(-1)
    if live.numel():
      self._write_rows(live, self._step)
    self._step += 1

  # Accessors used by the evaluator.

  @property
  def steps_recorded(self) -> int:
    return self._step

  @property
  def ended(self) -> torch.Tensor:
    """Per-environment first-episode end step, ``-1`` while it is still running."""
    return self._end_step

  @property
  def termination_terms(self) -> list[str]:
    return list(self._term_names)

  def export(self, steps: int) -> dict[str, np.ndarray]:
    """Copy the recorded episode trace to NumPy arrays.

    Args:
      steps: Number of control steps to keep, usually ``max(end_step) + 1``.

    Returns:
      One host copy per array, truncated once. Per-episode rows end where that
      environment's first episode ended, which is recorded in ``end_step``. World
      positions include ``env_origins``; the origin is recorded alongside them so a
      consumer can work in environment-local coordinates.
    """
    if steps < 1 or steps > self._max_steps:
      raise ValueError(f"steps must be in [1, {self._max_steps}], got {steps}")
    exported: dict[str, np.ndarray] = {
      "ball_pos_w": self._ball_pos[:steps].cpu().numpy(),
      "ball_vel_w": self._ball_vel[:steps].cpu().numpy(),
      "plate_pos_w": self._plate_pos[:steps].cpu().numpy(),
      "plate_axes_w": self._plate_axes[:steps].cpu().numpy(),
      "racket_substeps": self._racket[:steps].cpu().numpy(),
      "other_substeps": self._other[:steps].cpu().numpy(),
      "active": self._active[:steps].cpu().numpy(),
      "initial_ball_pos_w": self._initial_ball_pos.cpu().numpy(),
      "initial_ball_vel_w": self._initial_ball_vel.cpu().numpy(),
      "initial_plate_pos_w": self._initial_plate_pos.cpu().numpy(),
      "initial_plate_axes_w": self._initial_plate_axes.cpu().numpy(),
      "reset_count": self._reset_count.cpu().numpy(),
      "pre_step_resets": self._pre_step_resets.cpu().numpy(),
      "end_step": self._end_step.cpu().numpy(),
      "term_flags": self._term_flags.cpu().numpy(),
      "env_origins": self._env.scene.env_origins.cpu().numpy(),
      "step_dt": np.asarray(self._env.step_dt),
      "substep_dt": np.asarray(self._env.physics_dt),
    }
    return exported

  # Internals.

  def _sample_state(
    self,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ball world pose/velocity and plate world pose for every environment."""
    ball_data = self._ball.data
    robot_data = self._robot.data
    site_pose = robot_data.site_pose_w[:, self._plate_site_id]
    plate_axes = matrix_from_quat(site_pose[:, 3:7])
    return (
      ball_data.root_link_pos_w,
      ball_data.root_link_lin_vel_w,
      site_pose[:, :3],
      plate_axes,
    )

  def _write_rows(self, env_ids: torch.Tensor, row: int) -> None:
    if row >= self._max_steps:
      return
    ball_pos, ball_vel, plate_pos, plate_axes = self._sample_state()
    self._ball_pos[row, env_ids] = ball_pos[env_ids]
    self._ball_vel[row, env_ids] = ball_vel[env_ids]
    self._plate_pos[row, env_ids] = plate_pos[env_ids]
    self._plate_axes[row, env_ids] = plate_axes[env_ids]
    racket = substep_contact(self._racket_sensor.data.force_history)
    self._racket[row, env_ids] = racket[env_ids]
    other = substep_contact(self._any_sensor.data.force_history) & ~racket
    self._other[row, env_ids] = other[env_ids]
    self._active[row, env_ids] = True


def recorder_term(env: ManagerBasedRlEnv) -> NominalTraceRecorder:
  """Return the installed trace recorder term of a built environment."""
  term: Any = env.recorder_manager.get_term(RECORDER_TERM_NAME)
  if not isinstance(term, NominalTraceRecorder):
    raise TypeError(f"recorder term is {type(term).__name__}, not NominalTraceRecorder")
  return term


__all__ = [
  "BOOL_FIELDS",
  "FLOAT_FIELDS",
  "NominalTraceRecorder",
  "ROBOT_ENTITY",
  "add_capture_additions",
  "recorder_term",
]
