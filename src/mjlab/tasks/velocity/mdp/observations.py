from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.asset_zoo.robots.agibot_x2.tennis_constants import FACE_SITE_POS
from mjlab.entity import Entity
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def foot_height(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Per-foot vertical clearance above terrain.

  Returns:
    Tensor of shape [B, F] where F is the number of frames (feet).
  """
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, TerrainHeightSensor), (
    f"foot_height requires a TerrainHeightSensor, got {type(sensor).__name__}"
  )
  return sensor.data.heights


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


##
# Tennis ball, expressed in the plate frame.
#
# These are X2-tennis specific and live here only because the observation is the
# primitive: the reward and termination terms import the same helper, so there is
# exactly one place where the frame convention is defined.
##


def _single_body_id(plate_cfg: SceneEntityCfg) -> int:
  """The one body index a plate config must resolve to."""
  ids = plate_cfg.body_ids
  if isinstance(ids, slice) or len(ids) != 1:
    raise ValueError(f"plate_cfg must resolve to exactly one racket body, got {ids}")
  return ids[0]


def _face_site_pos(
  position_dtype: torch.dtype, device: torch.device | str
) -> torch.Tensor:
  """``FACE_SITE_POS`` as a tensor on the caller's device, ready to broadcast."""
  return torch.tensor(FACE_SITE_POS, dtype=position_dtype, device=device)


def _plate_origin_and_quat(
  env: ManagerBasedRlEnv, plate_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor]:
  """World position and orientation of the plate frame.

  The plate frame shares its orientation with the racket body, so this is just
  that body's pose.
  """
  plate: Entity = env.scene[plate_cfg.name]
  body_id = _single_body_id(plate_cfg)
  return (
    plate.data.body_link_pos_w[:, body_id],
    plate.data.body_link_quat_w[:, body_id],
  )


def plate_relative_ball_state(
  env: ManagerBasedRlEnv,
  plate_cfg: SceneEntityCfg,
  ball_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Ball position and velocity relative to the plate, in the plate frame.

  Args:
    plate_cfg: The racket body inside the robot entity. Must resolve to one body.
    ball_cfg: The ball entity.

  Returns:
    ``(position, velocity)``, each shape ``(num_envs, 3)``. Position is measured
    from the centre of the plate's **top surface** to the ball's **centre**, so a
    ball resting on the strings reads ``z = BALL_RADIUS``. Callers that want the
    gap above the strings subtract the radius. Velocity is the ball's velocity
    relative to the plate, expressed in the plate frame.

  The plate is an axis-aligned ellipse in the racket body frame, so the plate
  frame is the racket body frame translated to ``FACE_SITE_POS``.
  """
  plate: Entity = env.scene[plate_cfg.name]
  ball: Entity = env.scene[ball_cfg.name]
  plate_pos_w, plate_quat_w = _plate_origin_and_quat(env, plate_cfg)
  body_id = _single_body_id(plate_cfg)

  # Position: from the plate's top-surface centre to the ball's centre.
  arm_w = ball.data.root_link_pos_w - plate_pos_w
  position = quat_apply_inverse(plate_quat_w, arm_w) - _face_site_pos(
    position_dtype=arm_w.dtype, device=arm_w.device
  )

  # Velocity: relative to the *material point* of the plate at the ball, not to
  # the racket body origin. The face centre is 0.515 m from that origin, so a
  # 2 rad/s wrist turn moves it about 1 m/s faster than the origin; ignoring the
  # rotation would report a still ball as moving whenever the wrist turns.
  plate_point_vel_w = plate.data.body_link_lin_vel_w[:, body_id] + torch.cross(
    plate.data.body_link_ang_vel_w[:, body_id],
    arm_w,
    dim=1,
  )
  velocity = quat_apply_inverse(
    plate_quat_w, ball.data.root_link_lin_vel_w - plate_point_vel_w
  )
  return position, velocity


def ball_position_in_plate(
  env: ManagerBasedRlEnv,
  plate_cfg: SceneEntityCfg,
  ball_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Ball position relative to the plate centre, in the plate frame."""
  position, _ = plate_relative_ball_state(env, plate_cfg, ball_cfg)
  return position


def ball_velocity_in_plate(
  env: ManagerBasedRlEnv,
  plate_cfg: SceneEntityCfg,
  ball_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Ball velocity relative to the plate, in the plate frame.

  A ball moving with the bat reads zero, which is what "held" means.
  """
  _, velocity = plate_relative_ball_state(env, plate_cfg, ball_cfg)
  return velocity


class ball_position_hold(ManagerTermBase):
  """Ball position in the plate frame, with rejected frames served from a hold.

  The real perception chain does not deliver a measurement on every control step:
  it publishes nothing when the colour detector finds no ball or more than one
  candidate, when it cannot fit a consistent depth, and nothing at all when the
  stream breaks. A deployed controller therefore repeats the last valid position
  until a new one arrives, and the policy must train against that behaviour or it
  will be fitted to a guaranteed-fresh measurement it never has.

  On each step the term either passes the fresh position through or, with
  probability ``rejection_prob``, starts a hold that repeats the last accepted
  value for a number of steps drawn from ``hold_steps``. The held value and the
  hold counter are per environment, so one rejected frame does not freeze another
  environment's measurement, and both are cleared on reset so a new episode cannot
  repeat the previous episode's position.

  The first measurement after a reset is **always** accepted: there is no older
  value to hold yet, and inventing one would feed the policy a fabricated
  position. For the same reason the step after a hold always delivers a fresh
  value. Without that refractory step a rejection probability of 1 would hold one
  value forever, so the refresh the hold model exists to bound could not happen.
  The cost is that an unlimited stream break is not representable: the longest
  stale run is ``hold_steps`` long.

  ``bias_half_width_m`` adds a **constant** offset per environment per episode,
  drawn once at reset and applied to every measurement of that episode. It models a
  biased measurement chain rather than a noisy one: the robot's camera-to-pelvis
  disagreement is a fixed 24.69 mm / 1.24 deg, which zero-mean noise cannot
  represent. The default of 0.0 reproduces the unbiased chain exactly, and the
  offset is drawn per episode so a policy cannot average it out within one.

  Args:
    plate_cfg: The racket body inside the robot entity. Must resolve to one body.
    ball_cfg: The ball entity.
    rejection_prob: Probability per control step that a frame is rejected.
    hold_steps: Inclusive ``(min, max)`` hold length in control steps.
    bias_half_width_m: Half-width of the constant per-episode offset, in metres on
      each plate axis. 0.0 disables it.

  Returns:
    Tensor of shape ``(num_envs, 3)``, the same shape and frame as
    :func:`ball_position_in_plate`.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    half = float(cfg.params.get("bias_half_width_m", 0.0) or 0.0)
    self._bias_half_width_m = half
    self._bias = self._draw_bias(env.num_envs, env.device, half)
    self._held = torch.zeros(env.num_envs, 3, device=env.device)
    self._steps_left = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self._has_value = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self._just_held = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    plate_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    rejection_prob: float,
    hold_steps: tuple[int, int],
    bias_half_width_m: float = 0.0,
  ) -> torch.Tensor:
    del bias_half_width_m  # Drawn once per episode in ``reset``/``__init__``.
    fresh = ball_position_in_plate(env, plate_cfg, ball_cfg) + self._bias
    # One draw per environment per step, so a rejection in one environment says
    # nothing about any other. An environment that is already holding consumes a
    # draw and ignores it.
    rejected = torch.rand(self.num_envs, device=fresh.device) < rejection_prob
    start = rejected & (self._steps_left == 0) & self._has_value & ~self._just_held
    low, high = hold_steps
    drawn = torch.randint(int(low), int(high) + 1, (self.num_envs,), device=env.device)
    self._steps_left = torch.where(start, drawn, self._steps_left)

    holding = self._steps_left > 0
    self._held = torch.where(holding[:, None], self._held, fresh)
    self._has_value = self._has_value | ~holding
    # The step that serves the last held value arms the refractory step.
    self._just_held = holding & (self._steps_left == 1)
    self._steps_left = (self._steps_left - 1).clamp(min=0)
    return torch.where(holding[:, None], self._held, fresh)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._held[env_ids] = 0.0
    self._steps_left[env_ids] = 0
    self._has_value[env_ids] = False
    self._just_held[env_ids] = False
    if self._bias_half_width_m > 0.0:
      n = self._bias[env_ids].shape[0]
      self._bias[env_ids] = self._draw_bias(n, self._bias.device, self._bias_half_width_m)

  @staticmethod
  def _draw_bias(n: int, device, half: float):
    if half <= 0.0:
      return torch.zeros(n, 3, device=device)
    return (torch.rand(n, 3, device=device) * 2.0 - 1.0) * half
