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

import math
from typing import TYPE_CHECKING

import torch

from mjlab.asset_zoo.robots.agibot_x2.tennis_constants import (
  BALL_RADIUS,
  HEAD_JOINT_NAMES,
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
from mjlab.tasks.velocity.mdp.observations import plate_relative_ball_state
from mjlab.tasks.velocity.mdp.rewards import ball_vertical_energy
from mjlab.utils.lab_api.math import quat_apply, sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class reset_ball_release(ManagerTermBase):
  """Place the ball in one of the two stage-1 release states.

  Each environment draws one release type with probability
  ``plate_start_fraction``, which defaults to :data:`JUGGLE_PLATE_START_FRACTION`:

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
    plate_start_fraction: Probability that an environment draws a plate start
      instead of a drop start. Defaults to
      :data:`JUGGLE_PLATE_START_FRACTION`; ``0.0`` gives drop starts only.
    drop_height_range: ``(low, high)`` release height above the bed in metres.
      Defaults to :data:`JUGGLE_DROP_HEIGHT_RANGE_M`; ``(h, h)`` releases every
      drop from one height.
    drop_extra_time_s: ``(low, high)`` extra fall time in seconds. Defaults to
      :data:`JUGGLE_DROP_EXTRA_TIME_S`; ``(0.0, 0.0)`` releases the ball at rest
      vertically.
    drop_lateral_range_m: Half-width of the in-plane drop offset. Defaults to
      :data:`JUGGLE_DROP_LATERAL_RANGE_M`.
    drop_lateral_speed_range_mps: Half-width of the in-plane drop speed. Defaults
      to :data:`JUGGLE_DROP_LATERAL_SPEED_RANGE_MPS`; ``0.0`` drops the ball with
      no lateral speed.
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
    plate_start_fraction: float = JUGGLE_PLATE_START_FRACTION,
    drop_height_range: tuple[float, float] = JUGGLE_DROP_HEIGHT_RANGE_M,
    drop_extra_time_s: tuple[float, float] = JUGGLE_DROP_EXTRA_TIME_S,
    drop_lateral_range_m: float = JUGGLE_DROP_LATERAL_RANGE_M,
    drop_lateral_speed_range_mps: float = JUGGLE_DROP_LATERAL_SPEED_RANGE_MPS,
  ) -> None:
    env_ids = resolve_env_ids(env, env_ids)
    asset: Entity = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    device = env.device

    gravity = abs(float(env.sim.mj_model.opt.gravity[2]))
    plate = torch.rand(num_envs, device=device) < plate_start_fraction
    self.start_plate[env_ids] = plate.float()

    drop_height = sample_uniform(
      drop_height_range[0],
      drop_height_range[1],
      (num_envs,),
      device,
    )
    extra_time = sample_uniform(
      drop_extra_time_s[0],
      drop_extra_time_s[1],
      (num_envs,),
      device,
    )
    drop_lateral = sample_uniform(
      -drop_lateral_range_m,
      drop_lateral_range_m,
      (num_envs, 2),
      device,
    )
    drop_lateral_speed = sample_uniform(
      -drop_lateral_speed_range_mps,
      drop_lateral_speed_range_mps,
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


def reset_head_pose(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  position_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=HEAD_JOINT_NAMES),
) -> None:
  """Put the head at a random pose and make its actuator hold it there.

  For the 29-DOF family the policy cannot command the head, so the head pose is an
  exogenous condition that differs per robot. Writing the joint **state** alone
  would not work: the head is outside every action term, nothing rewrites its
  actuator target during the episode, and the reset zeroes that target to zero
  radians. The PD would then drag the head to 0 instead of to the sampled pose, and
  both the ``joint_pos`` observation and the camera-to-plate conversion would see a
  pose that was never sampled.

  So this term writes the target as well as the state. Because no action term owns
  these joints, the target survives the whole episode and the head stays where it is
  put until the next reset.

  Args:
    env: The environment.
    env_ids: Environments to reset, or None for all.
    position_range: Symmetric half-width of the offset about the default pose, in
      radians, applied to every joint in ``asset_cfg``. ``(0.0, 0.0)`` writes the
      default pose, which is the fixed-pose option.
    asset_cfg: The head joints. Must resolve to the head body's two joints.
  """
  env_ids = resolve_env_ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None

  default = default_joint_pos[env_ids][:, asset_cfg.joint_ids]
  limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
  half = max(abs(position_range[0]), abs(position_range[1]))
  if half == 0.0:
    # A zero half-width is the default pose. The band is not computed here because
    # it can be empty: the model's soft head limits are +-0.27 rad while the default
    # yaw is 0.276, so ``lo`` would pass ``hi`` and the draw would land between them.
    joint_pos = default.clone()
  else:
    # Sample uniformly inside the *feasible* band rather than about the default
    # followed by a clamp. A clamp piles environments onto whichever limit the band
    # ends at, and this band is narrow: the model allows +-0.3 rad for head yaw while
    # its default sits at 0.276, so most of a +-0.5 offset would land on the same
    # number. Uniform over the intersection keeps the poses spread and never
    # oversamples a limit.
    lo = torch.maximum(limits[..., 0], default - half)
    hi = torch.minimum(limits[..., 1], default + half)
    joint_pos = lo + (hi - lo) * torch.rand(default.shape, device=env.device)

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device)

  asset.write_joint_state_to_sim(
    joint_pos.view(len(env_ids), -1),
    torch.zeros_like(joint_pos).view(len(env_ids), -1),
    env_ids=env_ids,
    joint_ids=joint_ids,
  )
  # The half of this term that makes the pose persist. See the docstring.
  asset.set_joint_position_target(joint_pos, joint_ids=joint_ids, env_ids=env_ids)


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


class hold_ball(ManagerTermBase):
  """Hold the ball world-fixed above the plate, then let go of it at rest.

  The deployment condition this emulates: the operator holds the ball over the
  racket, waits, and releases it. The waiting time is not something the policy
  can influence, so the delay is sampled uniformly per environment and the ball
  is pinned at a world-fixed point until that delay elapses.

  **Why a pin and not a weld.** MuJoCo welds are model-level; this scene runs
  thousands of environments with independent release times. Writing the ball's
  ``qpos``/``qvel`` back to the hold point on every control step is per-environment
  and needs no model change. The write lands in the sim state directly, so the
  observation computed later in the same step sees the exact hold point with zero
  velocity.

  **The residual sag is hidden.** A control step is four physics substeps, so the
  ball falls ``g (4 dt)^2 / 2 = 1.96 mm`` and reaches 0.196 m/s *inside* each held
  step, and the next hold write erases it. Reward and termination terms that run
  before the event manager therefore see that sagged state; the energy term is
  gated off while held (:func:`~mjlab.tasks.velocity.mdp.rewards.ball_energy_after_release`)
  so the 0.019 J/kg it contributes is not scored.

  **Release is exact.** The release write sets the sampled hold point and zero
  velocity, so the ball starts from rest at the intended place: no position jump
  and no impulse. Nothing else writes the ball between the release and the next
  ``sim.step()``.

  The hold point uses the *nominal* bed anchor plus the measured bed normal, the
  same convention as :class:`reset_ball_release`'s drop branch: an operator's hand
  is world-fixed, it does not follow a perturbed plate. Height is measured above
  the ball's resting state on the plate, so ``0.40`` is the same 0.40 m the energy
  target uses.

  Args:
    asset_cfg: The ball entity, which must be a floating base.
    hold_height_m: Ball-centre height above its resting state, in metres, or
      ``None`` to follow the height the ball reset sampled. ``None`` is the mode a
      randomised rung uses: ``reset_ball_release`` then owns the single sample, the
      pin follows it, and ``ball_start_height_m`` cannot report a height the ball
      was never held at.
    hold_delay_s: ``(low, high)`` uniform hold duration before release. Include
      ``0.0`` so an immediate release stays in the distribution and stillness
      never becomes a precondition.
    lateral_range_m: Half-width of the world x/y hold offset, in metres.
    offset_speed_mps: Metres of additional world x/y displacement per second of
      hold. The displacement is ``offset_speed_mps * hold_duration`` along one
      uniformly drawn direction, applied at the moment the ball is placed and then
      held fixed, so the plate must travel to the ball rather than follow it.
      ``0.0`` keeps the rung that places the ball over the plate. Coupling the
      distance to the duration makes the required speed constant whatever the
      wait: the target is further away exactly when more time exists to reach it.
    release_event: Name of the :class:`reset_ball_release` term whose sampled
      height is followed when ``hold_height_m`` is ``None``.
    plate_cfg: Racket body, used only to record the waiting alignment at release.
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    self._asset_name = cfg.params["asset_cfg"].name
    self._step_dt = env.step_dt
    num_envs, device = env.num_envs, env.device
    self.hold_point = torch.zeros(num_envs, 3, device=device)
    """World hold point of every environment, in metres."""
    self.release_after_s = torch.zeros(num_envs, device=device)
    """Sampled hold duration of the current episode, in seconds."""
    self.elapsed_s = torch.zeros(num_envs, device=device)
    """Time spent held in the current episode, in seconds."""
    self.held = torch.zeros(num_envs, dtype=torch.bool, device=device)
    """True while the ball is still pinned."""
    self.released = torch.zeros(num_envs, dtype=torch.bool, device=device)
    """True once the current episode has released the ball."""
    self._released_this_step = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self.offset_m = torch.zeros(num_envs, device=device)
    """Sampled world x/y displacement from the nominal hold point, in metres."""
    self.release_alignment_m = torch.zeros(num_envs, device=device)
    """Plate-frame horizontal distance from the plate centre to the ball at the
    instant of release, in metres. Written once per episode, logging only."""
    self._asset_cfg: SceneEntityCfg | None = None
    self._plate_cfg: SceneEntityCfg | None = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    hold_height_m: float | None,
    hold_delay_s: tuple[float, float],
    lateral_range_m: float,
    offset_speed_mps: float = 0.0,
    release_event: str = "reset_ball",
    plate_cfg: SceneEntityCfg | None = None,
  ) -> None:
    env_ids = resolve_env_ids(env, env_ids)
    asset: Entity = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    device = env.device
    self._asset_cfg = asset_cfg
    self._plate_cfg = plate_cfg

    anchor = torch.tensor(ball_spawn_init_pos(0.0), device=device, dtype=torch.float32)
    normal = torch.tensor(BALL_SPAWN_FACE_NORMAL, device=device, dtype=torch.float32)
    anchor = anchor.expand(num_envs, 3)
    normal = normal.expand(num_envs, 3)
    lateral = sample_uniform(-lateral_range_m, lateral_range_m, (num_envs, 2), device)
    if hold_height_m is None:
      # Single authority: the reset sampled the release height, so hold there.
      height = _release_term(env, release_event).start_height[env_ids].clone()
    else:
      height = torch.full((num_envs,), float(hold_height_m), device=device)
    hold_s = sample_uniform(hold_delay_s[0], hold_delay_s[1], (num_envs,), device)
    # The offset draws nothing when it is disabled, so a rung without it consumes
    # exactly the random numbers it consumed before and a seeded replay is
    # unchanged to the bit.
    offset = torch.zeros(num_envs, device=device)
    point = anchor + normal * height[:, None]
    point[:, :2] += lateral
    if offset_speed_mps > 0.0:
      angle = sample_uniform(0.0, 2.0 * math.pi, (num_envs,), device)
      offset = float(offset_speed_mps) * hold_s
      point[:, 0] += offset * torch.cos(angle)
      point[:, 1] += offset * torch.sin(angle)
    point += env.scene.env_origins[env_ids]

    self.hold_point[env_ids] = point
    self.release_after_s[env_ids] = hold_s
    self.offset_m[env_ids] = offset
    self.elapsed_s[env_ids] = 0.0
    self.held[env_ids] = True
    self.released[env_ids] = False
    self._released_this_step[env_ids] = False
    self._pin(env, asset, env_ids)

  def advance(self, env: ManagerBasedRlEnv, asset: Entity) -> None:
    """Advance the hold clock one control step and release the due environments."""
    self._released_this_step.fill_(False)
    if not bool(self.held.any()):
      return
    self.elapsed_s += self._step_dt
    due = self.held & (self.elapsed_s >= self.release_after_s)
    still_held = self.held & ~due
    if bool(still_held.any()):
      self._pin(env, asset, still_held.nonzero(as_tuple=False).squeeze(-1))
    if bool(due.any()):
      due_ids = due.nonzero(as_tuple=False).squeeze(-1)
      self._record_alignment(env, due_ids)
      self._pin(env, asset, due_ids)
      self.held[due_ids] = False
      self.released[due_ids] = True
      self._released_this_step[due_ids] = True
      # Start the post-release clock at the release: every environment then gets
      # the same full window regardless of how long the operator waited.
      env.episode_length_buf[due_ids] = 0

  def _record_alignment(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
    """Store the plate-frame lateral offset at the release instant."""
    if self._plate_cfg is None or self._asset_cfg is None:
      return
    position, _ = plate_relative_ball_state(env, self._plate_cfg, self._asset_cfg)
    distance = torch.linalg.norm(position[:, :2], dim=1)
    self.release_alignment_m[env_ids] = distance[env_ids]

  def _pin(
    self,
    env: ManagerBasedRlEnv,
    asset: Entity,
    env_ids: torch.Tensor,
  ) -> None:
    if env_ids.numel() == 0:
      return
    count = env_ids.numel()
    orientation = torch.zeros((count, 4), device=env.device)
    orientation[:, 0] = 1.0
    zeros = torch.zeros((count, 3), device=env.device)
    asset.write_root_state_to_sim(
      torch.cat([self.hold_point[env_ids], orientation, zeros, zeros], dim=-1),
      env_ids=env_ids,
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Clear only the per-step release flag.

    The sampled hold state is **not** cleared here on purpose.
    ``ManagerBasedRlEnv._reset_idx`` applies the ``reset``-mode events *before* it
    calls ``event_manager.reset(env_ids)``, so a destructive reset would wipe the
    hold that :meth:`__call__` just sampled and every episode would start
    released. The reset-mode call is the authority for the sampled state.
    """
    env_ids = _reset_env_ids(env_ids)
    self._released_this_step[env_ids] = False


def advance_ball_hold(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  hold_event: str,
  asset_cfg: SceneEntityCfg,
) -> None:
  """Per-step companion to :class:`hold_ball`: keep pinning, release when due.

  A ``reset``-mode event only fires on environment resets, so the hold needs a
  second ``step``-mode term to advance every control step. Both terms share one
  state object, the same way the start metrics share
  :class:`reset_ball_release`'s instance.

  ``env_ids`` is the step-mode manager's positional ``None`` and is ignored: the
  term advances every environment and works from its own ``held`` mask.

  Args:
    env_ids: Unused; a ``step``-mode event receives ``None`` for all environments.
    hold_event: Name of the :class:`hold_ball` reset event that owns the state.
    asset_cfg: The ball entity.
  """
  del env_ids
  term = hold_term(env, hold_event)
  term.advance(env, env.scene[asset_cfg.name])


def hold_term(env: ManagerBasedRlEnv, hold_event: str) -> hold_ball:
  """The live :class:`hold_ball` instance, which owns the per-episode hold state."""
  term = env.event_manager.get_term_cfg(hold_event).func
  assert isinstance(term, hold_ball), (
    f"term needs a class-based '{hold_event}' event term"
  )
  return term


class ball_hold_state(ManagerTermBase):
  """Logging-only metrics for the hold phase, read from the hold term.

  ``held`` is 1.0 while the ball is pinned and 0.0 after release, so
  ``reduce="mean"`` reports the held fraction of the episode. ``release_after_s``
  is the sampled hold duration and is constant per episode, so it is logged with
  ``reduce="last"``. ``release_alignment_m`` is the plate-frame horizontal distance
  between the plate centre and the ball at the moment of release: the measured
  answer to "did it get under the ball", which is otherwise only visible by eye. Both are simulator-side facts used for accounting only; the
  actor never sees them.

  Args:
    hold_event: Name of the :class:`hold_ball` event term.
    field: ``"held"``, ``"release_after_s"``, ``"release_alignment_m"`` or
      ``"offset_m"``.
  """

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    term = hold_term(env, cfg.params["hold_event"])
    field = str(cfg.params.get("field", "held"))
    assert field in ("held", "release_after_s", "release_alignment_m", "offset_m"), (
      f"unknown hold field '{field}'"
    )
    # Keep the buffer itself, never a converted copy: ``held`` is a bool tensor, so
    # ``.float()`` allocates and a cached copy would report the values from
    # construction time for the whole run. The conversion happens on read instead.
    self._source = getattr(term, field)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    hold_event: str,
    field: str = "held",
  ) -> torch.Tensor:
    del env, hold_event, field  # The hold term writes this buffer in place.
    return self._source.float()
