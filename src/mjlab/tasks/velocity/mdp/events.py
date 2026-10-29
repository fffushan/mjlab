"""Task-scoped MDP events and the metrics that report their state.

The generic resets live in :mod:`mjlab.envs.mdp.events`
(``reset_root_state_uniform``, ``reset_joints_by_offset``). This module holds
terms that need a task's own geometry: currently the X2 tennis ball's stage-1
release state for the juggling task, plus the logging-only episode metrics that
report what that release sampled.

The release state is the first thing the policy sees, so the sampler and the
metrics that describe it live together: a change to the sampling formula cannot
leave the reported start height or start type stale. The sampler owns the
per-episode facts and the two start metrics read them from its instance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.asset_zoo.robots.agibot_x2.tennis_constants import (
  BALL_RADIUS,
  BALL_SPAWN_CLEARANCE,
  BALL_SPAWN_FACE_NORMAL,
  JUGGLE_DROP_EXTRA_TIME_S,
  JUGGLE_DROP_HEIGHT_RANGE_M,
  JUGGLE_DROP_LATERAL_RANGE_M,
  JUGGLE_DROP_LATERAL_SPEED_RANGE_MPS,
  JUGGLE_PLATE_START_FRACTION,
  JUGGLE_PLATE_START_LATERAL_RANGE_M,
  ball_spawn_init_pos,
)
from mjlab.entity import Entity
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.rewards import ball_vertical_energy
from mjlab.utils.lab_api.math import quat_apply, sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class reset_ball_release(ManagerTermBase):
  """Place the ball in one of the two stage-1 release states.

  Each environment draws one release type with probability
  :data:`JUGGLE_PLATE_START_FRACTION`:

  * **Plate start.** The ball centre is one radius plus
    :data:`BALL_SPAWN_CLEARANCE` above the **actual** post-reset plate surface,
    plus a lateral offset inside that plate's plane, at rest. The policy has to
    strike the ball up from the strings. The plate pose is read after
    ``env.sim.forward()`` so it includes the joint offsets written by the
    preceding robot reset; placing the ball at the *nominal* bed point instead
    would leave the perturbed plate inside the ball, and the first contact would
    eject it with several times the target energy.
  * **Drop start.** The ball is released ``h`` above the **nominal** bed,
    ``t`` seconds before the episode starts, so it arrives with the position and
    velocity a real drop would already have::

      z = bed_point + normal * (h - 0.5 g t^2),   v_z = -g t

    plus a lateral offset and lateral speed. A bounce exists from step zero. The
    drop point stays world-fixed: a ball dropped from a harness does not follow
    the racket, and the release must exercise the same approach whatever the arm
    pose.

  The bed normal is world +z to 6e-6 (see ``BALL_SPAWN_FACE_NORMAL``). The drop
  branch therefore samples its lateral offset and lateral speed on the world x/y
  axes, which is the bed plane to that tolerance, and avoids building a rotation
  that could disagree with the measured normal. The plate branch uses the live
  site's own x/y axes because the perturbed plate can tilt.

  ``g`` is read from the compiled model, the same gravity the energy reward and
  ``ball_not_lifted`` use, so the sampled state and the energy formula cannot
  disagree. With the drop samples, the release energy is exactly ``g * h``:
  ``0.5 (g t)^2 + g (h - 0.5 g t^2) = g h``.

  This is a class rather than a function because the two logging-only start
  metrics need the sampled facts. :attr:`start_height` and :attr:`start_plate`
  are per-environment and are overwritten on every reset of that environment.

  Args:
    asset_cfg: The ball entity, which must be a floating base.
    plate_cfg: The robot's ``face_center`` site; the plate frame is the site
      frame, whose +z is the bed normal.
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    del cfg  # The parameters are read on every call.
    self.start_height = torch.zeros(env.num_envs, device=env.device)
    """Height above the nominal resting state, in metres: 0 for a plate start,
    ``h - 0.5 g t^2`` for a drop start. Public for the start-height metric."""
    self.start_plate = torch.zeros(env.num_envs, device=env.device)
    """1.0 for a plate start, 0.0 for a drop start. Public for the start-type
    metric."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    plate_cfg: SceneEntityCfg,
  ) -> None:
    env_ids = resolve_env_ids(env, env_ids)
    asset: Entity = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    device = env.device

    gravity = abs(float(env.sim.mj_model.opt.gravity[2]))
    plate = torch.rand(num_envs, device=device) < JUGGLE_PLATE_START_FRACTION
    self.start_plate[env_ids] = plate.float()

    drop_height = sample_uniform(
      JUGGLE_DROP_HEIGHT_RANGE_M[0],
      JUGGLE_DROP_HEIGHT_RANGE_M[1],
      (num_envs,),
      device,
    )
    extra_time = sample_uniform(
      JUGGLE_DROP_EXTRA_TIME_S[0],
      JUGGLE_DROP_EXTRA_TIME_S[1],
      (num_envs,),
      device,
    )
    drop_lateral = sample_uniform(
      -JUGGLE_DROP_LATERAL_RANGE_M,
      JUGGLE_DROP_LATERAL_RANGE_M,
      (num_envs, 2),
      device,
    )
    drop_lateral_speed = sample_uniform(
      -JUGGLE_DROP_LATERAL_SPEED_RANGE_MPS,
      JUGGLE_DROP_LATERAL_SPEED_RANGE_MPS,
      (num_envs, 2),
      device,
    )
    plate_lateral = sample_uniform(
      -JUGGLE_PLATE_START_LATERAL_RANGE_M,
      JUGGLE_PLATE_START_LATERAL_RANGE_M,
      (num_envs, 2),
      device,
    )

    # Drop branch, world-fixed: the nominal bed reference plus the actual fall
    # distance along the measured bed normal.
    drop_axial = drop_height - 0.5 * gravity * extra_time.square()
    self.start_height[env_ids] = torch.where(
      plate, torch.zeros_like(drop_axial), drop_axial
    )
    anchor = torch.tensor(ball_spawn_init_pos(0.0), device=device, dtype=torch.float32)
    normal = torch.tensor(BALL_SPAWN_FACE_NORMAL, device=device, dtype=torch.float32)
    drop_position = anchor + normal * drop_axial[:, None]
    drop_position[:, :2] += drop_lateral
    drop_position += env.scene.env_origins[env_ids]

    # Plate branch, live: refresh kinematics first so the site pose includes the
    # robot joint offsets written by the preceding reset events.
    env.sim.forward()
    robot: Entity = env.scene[plate_cfg.name]
    site_pose = robot.data.site_pose_w[env_ids][:, plate_cfg.site_ids].squeeze(1)
    site_position = site_pose[:, :3]
    site_quat = site_pose[:, 3:7]
    local_x = torch.tensor((1.0, 0.0, 0.0), device=device).expand(num_envs, 3)
    local_y = torch.tensor((0.0, 1.0, 0.0), device=device).expand(num_envs, 3)
    local_z = torch.tensor((0.0, 0.0, 1.0), device=device).expand(num_envs, 3)
    plate_x = quat_apply(site_quat, local_x)
    plate_y = quat_apply(site_quat, local_y)
    plate_normal = quat_apply(site_quat, local_z)
    plate_position = site_position + plate_normal * (BALL_RADIUS + BALL_SPAWN_CLEARANCE)
    plate_position = plate_position + plate_x * plate_lateral[:, 0:1]
    plate_position = plate_position + plate_y * plate_lateral[:, 1:2]

    position = torch.where(plate[:, None], plate_position, drop_position)

    linear_velocity = torch.zeros((num_envs, 3), device=device)
    linear_velocity[:, :2] = torch.where(
      plate[:, None], torch.zeros_like(drop_lateral_speed), drop_lateral_speed
    )
    linear_velocity[:, 2] = torch.where(
      plate, torch.zeros_like(extra_time), -gravity * extra_time
    )
    # One release is a placement and the other a fall; neither imparts spin.
    angular_velocity = torch.zeros((num_envs, 3), device=device)

    orientation = torch.zeros((num_envs, 4), device=device)
    orientation[:, 0] = 1.0
    asset.write_root_state_to_sim(
      torch.cat([position, orientation, linear_velocity, angular_velocity], dim=-1),
      env_ids=env_ids,
    )


def _reset_env_ids(env_ids: torch.Tensor | slice | None) -> torch.Tensor | slice:
  return slice(None) if env_ids is None else env_ids


def _release_term(env: ManagerBasedRlEnv, release_event: str) -> reset_ball_release:
  """The live release-event instance, which owns the per-episode start facts."""
  term = env.event_manager.get_term_cfg(release_event).func
  assert isinstance(term, reset_ball_release), (
    f"metric needs a class-based '{release_event}' event term"
  )
  return term


class ball_start_height(ManagerTermBase):
  """Logging-only metric: ball height above its resting state at reset, in metres.

  A plate start reads 0 by construction; a drop start reads the sampled release
  height ``h - 0.5 g t_extra^2``. Install with ``reduce="last"``: the value is
  constant for the episode, so ``mean`` would only add a meaningless average.

  Args:
    release_event: Name of the :class:`reset_ball_release` event term that owns
      the sampled facts.
  """

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    self._start_height = _release_term(env, cfg.params["release_event"]).start_height

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    release_event: str,
  ) -> torch.Tensor:
    del env, release_event  # The sampler writes this tensor at reset.
    return self._start_height


class ball_start_plate(ManagerTermBase):
  """Logging-only metric: 1.0 for a plate start, 0.0 for a drop start.

  The value is recorded by :class:`reset_ball_release` when it samples the
  episode, so it cannot be inferred wrongly from a perturbed plate height.

  Args:
    release_event: Name of the :class:`reset_ball_release` event term that owns
      the sampled facts.
  """

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    self._start_plate = _release_term(env, cfg.params["release_event"]).start_plate

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    release_event: str,
  ) -> torch.Tensor:
    del env, release_event  # The sampler writes this tensor at reset.
    return self._start_plate


class ball_first_lift_time(ManagerTermBase):
  """Logging-only metric: time from reset to the first lift, or -1.0.

  The lift is the first control step whose ball energy reaches ``threshold``;
  the energy is :func:`~mjlab.tasks.velocity.mdp.rewards.ball_vertical_energy`,
  the same formula the reward and ``ball_not_lifted`` use. Time is elapsed
  **control** time counted by this term (``_steps * env.step_dt``), so the first
  sampled step reads 0.0. It is counted here rather than read from
  ``env.episode_length_buf`` because the RL wrapper seeds that buffer with a
  random episode length once per run (``init_at_random_ep_len``), which would
  report a fabricated lift time on the very first step. A drop start crosses at that first step by
  construction, because its release energy is ``g * h`` and ``h`` is above the
  threshold; a plate start records when the policy first lifts the ball. The
  value stays -1.0 if the episode ends before any lift.

  Args:
    ball_cfg: Ball entity.
    ball_rest_z: Ball centre's world z when it rests on the plate, in metres.
    threshold: Energy that counts as lifted, in J/kg.
  """

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    self._step_dt = env.step_dt
    self._ball_cfg: SceneEntityCfg = cfg.params["ball_cfg"]
    self._ball_rest_z: float = cfg.params["ball_rest_z"]
    self._threshold: float = cfg.params["threshold"]
    self._first_lift_s = torch.full((env.num_envs,), -1.0, device=env.device)
    self._steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    ball_cfg: SceneEntityCfg,
    ball_rest_z: float,
    threshold: float,
  ) -> torch.Tensor:
    del ball_cfg, ball_rest_z, threshold  # Read once in ``__init__``.
    energy = ball_vertical_energy(env, self._ball_cfg, self._ball_rest_z)
    elapsed = self._steps.float() * self._step_dt
    self._steps += 1
    lifted = (self._first_lift_s < 0.0) & (energy >= self._threshold)
    self._first_lift_s = torch.where(lifted, elapsed, self._first_lift_s)
    return self._first_lift_s

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    env_ids = _reset_env_ids(env_ids)
    self._first_lift_s[env_ids] = -1.0
    self._steps[env_ids] = 0
