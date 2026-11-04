from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.velocity.mdp.ball_contact import step_contact
from mjlab.tasks.velocity.mdp.observations import plate_relative_ball_state
from mjlab.tasks.velocity.mdp.rewards import ball_vertical_energy

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)


def out_of_terrain_bounds(
  env: ManagerBasedRlEnv,
  margin: float = 0.3,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Truncate if robot leaves the generated terrain footprint.

  Returns all-false for non-generator terrains (e.g. plane).
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_type != "generator":
    return torch.zeros(
      (env.num_envs,),
      device=env.device,
      dtype=torch.bool,
    )

  terrain_generator = terrain.cfg.terrain_generator
  if terrain_generator is None or terrain.terrain_origins is None:
    return torch.zeros(
      (env.num_envs,),
      device=env.device,
      dtype=torch.bool,
    )

  asset: Entity = env.scene[asset_cfg.name]
  root_xy_w = asset.data.root_link_pos_w[:, :2]

  # Use the generated grid shape (curriculum mode overrides cfg.num_cols with
  # len(sub_terrains)), and include the flat border around the patch grid.
  num_rows, num_cols = terrain.terrain_origins.shape[:2]
  half_x = 0.5 * (num_rows * terrain_generator.size[0]) + terrain_generator.border_width
  half_y = 0.5 * (num_cols * terrain_generator.size[1]) + terrain_generator.border_width
  limit_x = max(0.0, half_x - margin)
  limit_y = max(0.0, half_y - margin)

  return (root_xy_w[:, 0].abs() > limit_x) | (root_xy_w[:, 1].abs() > limit_y)


def terrain_edge_reached(
  env: ManagerBasedRlEnv,
  threshold_fraction: float = 0.95,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when robot displacement from spawn exceeds sub-terrain size.

  Intended as ``time_out=True`` (successful traversal, not penalized). Skips the first
  2 steps after reset to avoid stale-position triggers.
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_type != "generator":
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  terrain_generator = terrain.cfg.terrain_generator
  if terrain_generator is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  asset: Entity = env.scene[asset_cfg.name]
  displacement = (
    asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  ).abs()

  half_x = terrain_generator.size[0] / 2.0 * threshold_fraction
  half_y = terrain_generator.size[1] / 2.0 * threshold_fraction

  at_edge = (displacement[:, 0] > half_x) | (displacement[:, 1] > half_y)

  # Don't fire on the first 2 steps after reset (position may be stale).
  at_edge &= env.episode_length_buf > 2

  return at_edge


class ball_lost(ManagerTermBase):
  """End the episode when the ball has stayed below a fixed height above the ground.

  The test is the ball's **absolute** world height. It does not involve the plate.

  **Why not the plate frame.** A plate-frame test is defeated by the policy, because
  "below the plate" rotates with the racket. Holding the racket near vertical and
  dropping the ball on the floor keeps the ball on the positive side of that plane,
  so the height never falls below the threshold and the episode runs to the time
  limit with no ball at all. The 30 k Ball3 baseline shows the failure: 3.6 racket
  contacts across a 19.2 s episode, and 94 % of its episodes ended by truncation
  rather than by losing the ball.

  **Why absolute.** No pose of the robot changes the number, so the test cannot be
  satisfied by a gesture. The only way to stay above ``min_world_z`` is to keep the
  ball up. A plane test is also defeated by its own extent: it asks whether the ball
  is above an infinite plane, not above the plate.

  The grace count is unchanged: ``grace_steps`` consecutive control steps below the
  threshold are required, so a ball that briefly passes below the height during a
  strike does not end the episode, and a ball that comes back up restarts the count.

  Args:
    min_world_z: Ball-centre height above the ground, in metres, at or below which
      the ball counts as lost.
    grace_steps: Consecutive control steps the ball may stay lost. ``0`` fires on the
      first lost step.
    ball_cfg: Ball entity.
  """

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    self._lost_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    min_world_z: float,
    grace_steps: int,
    ball_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    ball: Entity = env.scene[ball_cfg.name]
    # The ball's root is its only body, so this is the ball centre's world height.
    lost = ball.data.root_link_pos_w[:, 2] <= min_world_z
    self._lost_steps = torch.where(
      lost, self._lost_steps + 1, torch.zeros_like(self._lost_steps)
    )
    return self._lost_steps >= max(int(grace_steps), 1)

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._lost_steps[env_ids] = 0


def ball_below_plate(
  env: ManagerBasedRlEnv,
  threshold: float,
  ball_radius: float,
  plate_cfg: SceneEntityCfg,
  ball_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """End the episode when the ball has dropped below the plate.

  Args:
    threshold: Height above the plate's top surface below which the episode
      ends, in metres. Negative, so the small penetration a compliant contact
      allows does not fire it. A ball rolling off the edge reaches -20 mm about
      65 ms after leaving the plate.
    ball_radius: The ball's radius; the stored height is to its centre.
    plate_cfg: Racket body; the plate frame is defined by it.
    ball_cfg: Ball entity.

  Returns:
    Boolean tensor of shape ``(num_envs,)``.
  """
  position, _ = plate_relative_ball_state(env, plate_cfg, ball_cfg)
  height = position[:, 2] - ball_radius
  below = height < threshold
  # Don't fire on the first steps after a reset, when the ball's state may still
  # be the one the resetter wrote rather than a stepped one.
  below &= env.episode_length_buf > 2
  return below


class ball_not_lifted(ManagerTermBase):
  """End an episode whose ball was never lifted within the timeout.

  A plate start has no bounce to sustain: if the policy never strikes the ball,
  the ball sits on the strings and nothing else ends the episode, so the whole
  20 s allowance is spent on a state that teaches nothing. This term ends that
  episode after ``timeout_s`` of control time.

  The test is the **peak** energy since reset, not the current energy, so a ball
  that was lifted once and fell back does not fire the term. The peak is
  per-environment and is cleared on every reset, so a stale episode cannot end a
  fresh one.

  **Drop starts are inert by construction.** Their release energy is
  ``g * h`` with ``h >= JUGGLE_DROP_HEIGHT_RANGE_M[0] = 0.15`` m, i.e. at least
  ``9.81 * 0.15 = 1.4715`` J/kg, which is 0.4715 J/kg above
  :data:`JUGGLE_LIFT_ENERGY_THRESHOLD_J_PER_KG`. The peak therefore exceeds the
  threshold on the first control step and the term can never fire. It rejects
  only plate starts where the policy never lifted the ball.

  Timing is elapsed **control** time, counted by this term itself
  (``_steps * env.step_dt``), not read from ``env.episode_length_buf``: the RL
  wrapper overwrites that buffer once per run with a random episode length
  (``init_at_random_ep_len``), so it starts at an arbitrary value up to the time
  limit and a term that trusted it would fire on the first step. ``ball_lost``
  in this module counts its own steps for the same reason. ``env.step_dt`` is
  ``cfg.sim.mujoco.timestep * cfg.decimation``; the physics timestep is 20 times
  smaller and is not the policy's clock.

  Args:
    ball_cfg: Ball entity.
    ball_rest_z: Ball centre's world z when it rests on the plate, in metres.
    threshold: Peak energy that counts as lifted, in J/kg.
    timeout_s: Control time after reset before the term may fire, in seconds.
  """

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    self._step_dt = env.step_dt
    self._peak_energy = torch.zeros(env.num_envs, device=env.device)
    self._steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    ball_cfg: SceneEntityCfg,
    ball_rest_z: float,
    threshold: float,
    timeout_s: float,
  ) -> torch.Tensor:
    energy = ball_vertical_energy(env, ball_cfg, ball_rest_z)
    self._peak_energy = torch.maximum(self._peak_energy, energy)
    elapsed = self._steps.float() * self._step_dt
    self._steps += 1
    return (elapsed >= timeout_s) & (self._peak_energy < threshold)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._peak_energy[env_ids] = 0.0
    self._steps[env_ids] = 0


class ball_no_bounce(ManagerTermBase):
  """End an episode in which the racket has not struck the ball for a while.

  A parked ball is a measured failure mode: in the last ablation the parked arms
  collected about 4.5 reward per second for 19.4 s with 2.5 impacts per episode,
  while the only bouncing arm survived 6.4 s at 1.3 impacts per second. Episode
  reward is ``sum(raw * weight * dt) / max_episode_length_s``, so duration
  multiplies everything the policy collects. Ending an episode whose last impact
  is 2.5 s old caps a permanently parked episode at about 2.5 s after that
  impact and removes the advantage.

  **Impact is a rising edge.** Contact now with no contact on the previous
  control step restarts the timer. Sustained contact does not: a ball resting on
  the plate is sustained contact, and treating it as an impact would make
  parking invisible to this term. The deadline is therefore measured from the
  last impact, not from reset: a carry-up with sustained contact fires at about
  2.6-2.8 s, while a toss at 1.4 s followed by parking fires at about 4.3-4.5 s.

  **Why this cannot pre-empt the other terms.** ``ball_not_lifted`` fires first,
  at its 1.5 s timeout, for a plate start the policy never lifts, because this
  term's deadline is longer. A drop start cannot reach this deadline either.
  Measured on 8 environments with zero action and every ball termination
  removed, a drop start produced 1-4 impacts in 20 s, its contact fraction
  decayed to zero, and the ball ended on the floor. The first contact arrives
  0.2-0.35 s after release, and an inert policy ends through ``ball_dropped``
  about 0.5 s later, long before 2.5 s.

  **A missed bounce does not accumulate tolerance.** A missed bounce ends the
  episode through ``ball_dropped`` 0.5 s later, so 2.5 s of airborne misses
  cannot accrue. 2.5 s is 4.4 flight times of a 0.40 m bounce, and the window
  only expires on sustained contact.

  **Substep history is required.** The term refuses the instantaneous ``found``
  fallback. One sample per control step misses about half of the 9-10 ms
  contacts, so five consecutive misses on a live bounce cycle would falsely end
  the episode. The sensor must keep one control step of force history
  (``history_length == decimation``).

  **What it does not close.** Ground bouncing is already closed by
  ``ball_dropped``: the ball 2 cm below the plate for 0.5 s. A sustained
  minimum-height dribble that keeps producing rising edges remains possible and
  is deliberately not guarded here.

  The clock is counted in this term's own control steps
  (``_steps_since * env.step_dt``) and is zeroed on reset.
  ``env.episode_length_buf`` is not read: the RL wrapper overwrites it with a
  random value at the start of every ``learn()`` (``init_at_random_ep_len``), so
  a term that trusted it would fire on step 1. ``ball_lost`` and
  ``ball_not_lifted`` in this module count their own steps for the same reason.

  Args:
    sensor_name: Scene name of the racket-ball
      :class:`~mjlab.sensor.ContactSensor`.
    timeout_s: Control time without a rising-edge contact before the episode
      ends, in seconds.

  Returns:
    Shape ``(num_envs,)`` bool.
  """

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    self._sensor_name = str(cfg.params["sensor_name"])
    self._step_dt = env.step_dt
    self._substep_dt = env.physics_dt
    sensor: ContactSensor = env.scene[self._sensor_name]
    decimation = round(self._step_dt / self._substep_dt)
    # The metrics may fall back to the instantaneous ``found`` sample; a
    # termination may not, because a missed impact would extend the deadline
    # instead of being harmless.
    if sensor.cfg.history_length != decimation or "force" not in sensor.cfg.fields:
      msg = (
        f"Sensor '{sensor.cfg.name}' has history_length="
        f"{sensor.cfg.history_length} and fields={tuple(sensor.cfg.fields)}, but "
        f"ball_no_bounce needs one control "
        f"step, {decimation}, of force history: the instantaneous 'found' "
        "fallback misses about half of the 9-10 ms contacts, so five "
        "consecutive misses would falsely terminate a live juggle."
      )
      raise ValueError(msg)
    self._prev_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self._steps_since = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def __call__(
    self, env: ManagerBasedRlEnv, sensor_name: str, timeout_s: float
  ) -> torch.Tensor:
    del sensor_name  # Resolved in ``__init__``.
    sensor: ContactSensor = env.scene[self._sensor_name]
    contact = step_contact(sensor.data)
    rising = contact & ~self._prev_contact
    # Clone: environment buffers are written in place, and an uncloned alias
    # made a guard fire every step and froze a term in this repo before.
    self._prev_contact = contact.clone()
    self._steps_since = torch.where(
      rising,
      torch.zeros_like(self._steps_since),
      self._steps_since + 1,
    )
    elapsed = self._steps_since.float() * self._step_dt
    return elapsed >= timeout_s

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._prev_contact[env_ids] = False
    self._steps_since[env_ids] = 0
