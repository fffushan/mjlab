from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.asset_zoo.robots.agibot_x2.tennis_constants import FACE_SITE_POS
from mjlab.entity import Entity
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
