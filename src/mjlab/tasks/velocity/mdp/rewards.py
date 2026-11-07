from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.asset_zoo.robots.agibot_x2.tennis_constants import BALL_RADIUS, FACE_SITE_POS
from mjlab.entity import Entity
from mjlab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.velocity.mdp.ball_contact import (
  history_length_or_raise,
  step_contact,
  substep_contact,
)
from mjlab.tasks.velocity.mdp.observations import plate_relative_ball_state
from mjlab.tasks.velocity.mdp.terrain_utils import terrain_normal_from_sensors
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_mul
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + z_error
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + xy_error
  return torch.exp(-ang_vel_error / std**2)


class upright:
  """Reward for keeping the base upright.

  Without ``terrain_sensor_names``, penalizes tilt relative to world up (correct for
  flat ground).

  With ``terrain_sensor_names``, penalizes tilt relative to the terrain surface normal.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._terrain_sensor_names: tuple[str, ...] | None = cfg.params.get(
      "terrain_sensor_names"
    )
    self._debug_vis_enabled = True
    self._env = env
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    terrain_sensor_names: tuple[str, ...] | None = None,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    if asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
      body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    else:
      body_quat_w = asset.data.root_link_quat_w  # [B, 4]

    if terrain_sensor_names is not None:
      terrain_normal = terrain_normal_from_sensors(env, terrain_sensor_names)  # [B, 3]
      # Project terrain normal into body frame. When aligned with the terrain surface
      # this should be (0, 0, 1); XY measures tilt.
      target_b = quat_apply_inverse(body_quat_w, terrain_normal)  # [B, 3]
      xy_squared = torch.sum(torch.square(target_b[:, :2]), dim=1)
    else:
      gravity_w = asset.data.gravity_vec_w  # [3]
      projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
      xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

    return torch.exp(-xy_squared / std**2)

  def reset(self, env_ids: torch.Tensor) -> None:
    del env_ids  # Unused.

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self._debug_vis_enabled or self._terrain_sensor_names is None:
      return

    env = self._env
    asset: Entity = env.scene[self._asset_cfg.name]

    env_indices = list(visualizer.get_env_indices(env.num_envs))
    if not env_indices:
      return

    terrain_normal = terrain_normal_from_sensors(env, self._terrain_sensor_names)
    if self._asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, self._asset_cfg.body_ids, :].squeeze(
        1
      )
    else:
      body_quat_w = asset.data.root_link_quat_w
    up_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(
      body_quat_w[:, :3]
    )
    body_up_w = quat_apply(body_quat_w, up_local)

    positions = asset.data.root_link_pos_w.cpu().numpy()
    offset = np.array([0.0, 0.3, 0.0])
    terrain_normal_np = terrain_normal.cpu().numpy()
    body_up_np = body_up_w.cpu().numpy()
    scale = 0.25

    for i in env_indices:
      origin = positions[i] + offset
      # Terrain normal (magenta).
      visualizer.add_arrow(
        start=origin,
        end=origin + terrain_normal_np[i] * scale,
        color=(0.8, 0.2, 0.8, 0.8),
        width=0.01,
      )
      # Body up (orange).
      visualizer.add_arrow(
        start=origin,
        end=origin + body_up_np[i] * scale,
        color=(1.0, 0.5, 0.0, 0.8),
        width=0.01,
      )


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.sum(dim=-1).float()


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)
  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  height_sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  height_sensor = env.scene[height_sensor_name]
  assert isinstance(height_sensor, TerrainHeightSensor), (
    f"feet_clearance requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
  )
  foot_height = height_sensor.data.heights  # [B, F]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, F, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, F]
  delta = torch.abs(foot_height - target_height)  # [B, F]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    height_sensor = env.scene[cfg.params["height_sensor_name"]]
    assert isinstance(height_sensor, TerrainHeightSensor), (
      f"feet_swing_height requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
    )
    num_feet = height_sensor.num_frames
    self.peak_heights = torch.zeros(
      (env.num_envs, num_feet), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
  ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    height_sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    foot_heights = height_sensor.data.heights
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


##
# Tennis ball on the bat.
#


def ball_centred(
  env: ManagerBasedRlEnv,
  plateau: float,
  sigma: float,
  ball_radius: float,
  sink: float,
  plate_cfg: SceneEntityCfg,
  ball_cfg: SceneEntityCfg,
) -> torch.Tensor:
  r"""Reward the ball being near the middle of the plate.

  .. math:: r = \exp(-(\max(0, d - plateau) / sigma)^2)

  where ``d`` is the ball's horizontal distance from the plate centre. The
  plateau keeps the gradient zero inside the tolerance, so the ball-position
  observation noise cannot turn into reward noise where the ball usually sits.

  Args:
    plateau: Radius inside which the reward is exactly 1, in metres.
    sigma: Falloff distance beyond the plateau, in metres.
    ball_radius: The ball's radius; the stored height is to its centre.
    sink: How far below the string plane the ball may be and still count as on the
      plate, in metres. Below that the reward is zero. The horizontal distance alone
      cannot separate a ball on the strings from one that has left the plate
      downward, and paying it there paid full credit for a lost ball.
    plate_cfg: Racket body; the plate frame is defined by it.
    ball_cfg: Ball entity.

  Returns:
    Shape ``(num_envs,)``, between 0 and 1.
  """
  position, _ = plate_relative_ball_state(env, plate_cfg, ball_cfg)
  distance = torch.linalg.norm(position[:, :2], dim=1)
  outside = torch.clamp(distance - plateau, min=0.0)
  centred = torch.exp(-torch.square(outside / sigma))
  height = position[:, 2] - ball_radius
  return torch.where(height > -sink, centred, torch.zeros_like(centred))


def ball_height(
  env: ManagerBasedRlEnv,
  limit: float,
  sigma: float,
  ball_radius: float,
  sink: float,
  plate_cfg: SceneEntityCfg,
  ball_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward the ball being on the strings or up to ``limit`` above them.

  Full credit from ``sink`` below the string plane (a compliant contact lets the
  ball penetrate slightly) up to ``limit``, then a Gaussian falloff with ``sigma``
  on either side. A ball further below the strings than ``sink`` is gone, so it
  decays instead of collecting the clamp's free maximum.

  The clamp used to be one-sided, `max(0, height - limit)`, which paid 1.0 for
  every height below ``limit`` including half a metre under the racket. The
  below-plate case was left to a termination, and that termination is now the
  ball's absolute height, which a ball held above 0.5 m can satisfy while hanging
  under the strings.

  Without this term "throw the ball straight up and keep it centred" would pay as
  well as holding it still.
  """
  position, _ = plate_relative_ball_state(env, plate_cfg, ball_cfg)
  height = position[:, 2] - ball_radius
  above = torch.clamp(height - limit, min=0.0)
  below = torch.clamp(-sink - height, min=0.0)
  return torch.exp(-torch.square((above + below) / sigma))


def ball_in_camera_fov(
  env: ManagerBasedRlEnv,
  plateau: float,
  sigma: float,
  mount_xyz: tuple[float, float, float],
  mount_quat: tuple[float, float, float, float],
  half_angle_x: float,
  half_angle_y: float,
  head_cfg: SceneEntityCfg,
  ball_cfg: SceneEntityCfg,
) -> torch.Tensor:
  r"""Reward the ball being inside the head camera's field of view.

  The ball position on hardware comes from the head stereo pair, so a ball outside
  the frame is a lost ball however well it is balanced. This is a **soft safety
  term**: the ball sits at about a quarter of the half-frame while it is carried, so
  the reward is normally 1 and only falls when the ball leaves the hand or the arm
  swings out of view.

  The camera frame is the head body frame transformed by a constant mount. Its axes
  follow the ROS optical convention, +x right, +y down, **+z forward**; MuJoCo's own
  camera convention is the opposite along z, and getting that sign wrong puts the
  ball behind the camera.

  Angles are converted to an image radius with the **equidistant** model of the live
  fisheye calibration, so an image edge lies at :math:`r/f` and not at
  :math:`\arctan(r/f)`. The normalised offset is the larger of the two per-axis
  ratios, which is a rectangular frame test:

  .. math:: s = \max(|\theta_x| / \text{half\_angle\_x},\ |\theta_y| / \text{half\_angle\_y})

  Args:
    plateau: Normalised image radius that still earns full credit.
    sigma: Falloff beyond the plateau; the reward reaches zero at ``s = plateau + ~2 sigma``.
    mount_xyz: Camera position in the head frame, metres.
    mount_quat: Camera orientation in the head frame, wxyz.
    half_angle_x: Angle from the optical axis to the image edge, horizontal, radians.
    half_angle_y: The same, vertical.
    head_cfg: The head body the camera is mounted on. Must resolve to one body.
    ball_cfg: Ball entity.

  Returns:
    Shape ``(num_envs,)``, between 0 and 1. A ball at or behind the camera plane
    scores 0.
  """
  head: Entity = env.scene[head_cfg.name]
  ball: Entity = env.scene[ball_cfg.name]
  ids = head_cfg.body_ids
  if isinstance(ids, slice) or len(ids) != 1:
    raise ValueError(f"head_cfg must resolve to exactly one head body, got {ids}")
  body_id = ids[0]

  head_pos_w = head.data.body_link_pos_w[:, body_id]
  head_quat_w = head.data.body_link_quat_w[:, body_id]
  mount_xyz_t = torch.tensor(
    mount_xyz, dtype=head_pos_w.dtype, device=head_pos_w.device
  )
  mount_quat_t = torch.tensor(
    mount_quat, dtype=head_quat_w.dtype, device=head_quat_w.device
  )

  cam_pos_w = head_pos_w + quat_apply(head_quat_w, mount_xyz_t.expand_as(head_pos_w))
  cam_quat_w = quat_mul(head_quat_w, mount_quat_t.expand_as(head_quat_w))
  delta_cam = quat_apply_inverse(cam_quat_w, ball.data.root_link_pos_w - cam_pos_w)

  # Optical frame: +z forward. Anything at or behind the camera plane is out.
  forward = delta_cam[:, 2]
  in_front = forward > 1e-6
  safe_forward = torch.where(in_front, forward, torch.ones_like(forward))
  theta_x = torch.atan2(delta_cam[:, 0], safe_forward)
  theta_y = torch.atan2(delta_cam[:, 1], safe_forward)
  radius = torch.maximum(
    torch.abs(theta_x) / half_angle_x, torch.abs(theta_y) / half_angle_y
  )
  outside = torch.clamp(radius - plateau, min=0.0)
  reward = torch.exp(-torch.square(outside / sigma))
  return torch.where(in_front, reward, torch.zeros_like(reward))


def head_joint_motion(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalise head joint velocity, summed over the joints in ``asset_cfg``.

  The head joints are policy outputs, so the field-of-view reward gives the policy a
  reason to hunt with them. Chasing a ball that perception measures with millimetres
  of noise is not useful motion: it costs actuator travel, it smears the frames
  perception needs, and it sets the ball's image moving.

  This penalises **velocity**, not position, so the head can still be aimed
  deliberately and then held.
  """
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


##
# Tennis juggling: the ball bounces off the plate and comes back.
#
# The task is paid by the ball's **vertical energy relative to its resting height
# on the plate** matching the energy of a bounce to a target height, on every
# control step. Energy is the quantity a bounce conserves: a ball at rest on the
# plate has none, a ball rising to the target after a bounce has the target value at
# every point of its flight, and a ball hit too hard has too much. See
# :func:`ball_energy`.
#
# :class:`ball_bounce_apex` and :class:`ball_at_rest` below are an earlier design
# that paid once per bounce and charged for parking the ball. They are kept as
# library terms with their own tests, but the juggling task no longer installs them:
# a per-bounce payment is zero on every non-apex step, so it gives no gradient while
# the ball is in flight, which is most of the cycle.
##


def _plate_body_id(plate_cfg: SceneEntityCfg) -> int:
  """The single racket body ``plate_cfg`` resolves to."""
  body_ids = plate_cfg.body_ids
  if isinstance(body_ids, slice) or len(body_ids) != 1:
    raise ValueError(
      f"plate_cfg must resolve to exactly one racket body, got {body_ids}"
    )
  return body_ids[0]


def _plate_top_z(
  env: ManagerBasedRlEnv,
  plate_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """World z of the centre of the plate's top surface."""
  plate: Entity = env.scene[plate_cfg.name]
  body_id = _plate_body_id(plate_cfg)
  plate_pos_w = plate.data.body_link_pos_w[:, body_id]
  plate_quat_w = plate.data.body_link_quat_w[:, body_id]
  face_offset = torch.tensor(
    FACE_SITE_POS, dtype=plate_pos_w.dtype, device=plate_pos_w.device
  )
  face_pos_w = plate_pos_w + quat_apply(
    plate_quat_w, face_offset.expand_as(plate_pos_w)
  )
  return face_pos_w[:, 2]


def _ball_gap_and_vertical_speed(
  env: ManagerBasedRlEnv,
  plate_cfg: SceneEntityCfg,
  ball_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
  """The gap under the ball, and the ball's world vertical speed.

  The gap is measured along the **world vertical**, from the centre of the plate's
  top surface down to the ball's **lowest point**: a ball resting on the strings
  reads about zero, a positive value is clearance under it. The plate's top surface
  is the racket body pose translated by the rotated ``FACE_SITE_POS``, the same
  point the ball observations measure from.

  The axis is the world's, not the plate normal, because gravity defines a bounce:
  the ball turns around where its world vertical speed changes sign, and its highest
  point is a world-frame fact. The plate tilts while juggling, so its normal is not a
  fixed direction to measure an apex against.

  Args:
    plate_cfg: Racket body. Must resolve to exactly one body.
    ball_cfg: Ball entity.

  Returns:
    ``(gap, vertical_speed)``, each shape ``(num_envs,)``. Positive
    ``vertical_speed`` is upward.
  """
  ball: Entity = env.scene[ball_cfg.name]
  gap = ball.data.root_link_pos_w[:, 2] - _plate_top_z(env, plate_cfg) - BALL_RADIUS
  return gap, ball.data.root_link_lin_vel_w[:, 2]


def ball_vertical_energy(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg,
  ball_rest_z: float,
) -> torch.Tensor:
  r"""The ball's vertical energy per unit mass, in J/kg.

  .. math:: E = \tfrac{1}{2} v_{\mathrm{ball},z}^2 + g \, (z - z_{\mathrm{rest}})

  This is the single definition of :math:`E` for the juggling task's reward, its
  energy metrics and its ``ball_not_lifted`` termination, so the three cannot
  drift apart. The velocity is **vertical only** and the datum is a fixed world
  height, for the anti-loophole reasons in :func:`_ball_energy_and_target`. The
  racket's own velocity never enters. Gravity is read from the compiled model,
  the physics that actually runs, not a literal 9.81.

  Args:
    ball_cfg: Ball entity.
    ball_rest_z: ``z_rest``, the ball centre's absolute world z when the ball
      rests on the plate in the carry pose, in metres.

  Returns:
    Shape ``(num_envs,)``, in J/kg.
  """
  ball: Entity = env.scene[ball_cfg.name]
  height = ball.data.root_link_pos_w[:, 2] - ball_rest_z
  gravity = abs(float(env.sim.mj_model.opt.gravity[2]))
  return 0.5 * torch.square(ball.data.root_link_lin_vel_w[:, 2]) + gravity * height


def _ball_energy_and_target(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg,
  target_height: float,
  ball_rest_z: float,
) -> tuple[torch.Tensor, float]:
  r"""The ball's energy per unit mass, and the energy of the target apex.

  .. math::

    h = z_{\mathrm{ball}} - z_{\mathrm{rest}}, \qquad
    E = \tfrac{1}{2} v_{\mathrm{ball},z}^2 + g \, h, \qquad
    E^* = g \, h_{\mathrm{target}}

  where ``z_rest`` (``ball_rest_z``) is the ball centre's **absolute world
  height** when the ball rests on the plate in the carry pose, so ``h`` is the
  ball's height above that resting state. The juggle spawn is ``h = 0.20`` m with
  zero vertical speed, i.e. ``E = 1.962`` J/kg against a target of ``E^* = 3.924``
  J/kg: the spawn is **below** the target, not on it.

  Four rules keep :math:`E` from being reachable without a bounce. Each is an
  anti-loophole rule, not a stylistic choice:

  * **Vertical only.** With the full speed :math:`\tfrac{1}{2}|\mathbf{v}|^2`, the
    target :math:`g \cdot 0.40` is reached at a speed of 2.80 m/s with no bounce at
    all: a robot walking at 2.80 m/s with a ball sitting still on the racket, or a
    ball rolling across the plate, would earn full credit. The vertical component is
    zero in both cases, so this formula pays nothing for them.
  * **No plate velocity anywhere.** The racket is the control input, so it must not
    be able to pay itself. Subtracting the plate velocity from the ball's would let a
    **descending** racket manufacture energy with the ball untouched: dropping the arm
    at 1 m/s under a resting ball gives :math:`\tfrac{1}{2} \cdot 1.0^2 = 0.5`
    J/kg, 13 % of the target for free, and at about 2.8 m/s it reaches the target
    with the ball never leaving the strings. The racket's motion enters this formula
    only through the motion it actually gives the ball.
  * **The datum is a fixed world height.** With ``h`` measured from the pelvis
    instead, lowering the pelvis 0.20 m while the ball stays put raises ``h`` by
    0.20 m and adds :math:`g \cdot 0.20 = 1.962` J/kg, half of :math:`E^*`, with no
    work done on the ball. The lift term cannot see it, because the plate and the
    pelvis drop together and its excess stays zero. That is free energy, and leg
    motion is not a juggling skill. The cost of the world datum is that gait bob
    moves :math:`E`: 3 cm of bob is :math:`g \cdot 0.03 = 0.29` J/kg, 0.19 sigma.
    Accepted, because the exploit is free energy under any target.
  * **Gravity comes from the compiled model**, ``env.sim.mj_model.opt.gravity[2]``,
    not a literal 9.81. The environment's gravity is the physics that actually runs,
    so :math:`E^*` has to be computed with it; the absolute value is taken because
    MuJoCo stores gravity as a vector with a negative z entry.

  A ball resting on the plate and raised by the racket is a separate loophole: its
  ``h`` equals the world lift, so a 0.40 m lift reads exactly :math:`E^*`.
  :func:`racket_lift` exists to charge for that motion; see its docstring.

  Args:
    ball_cfg: Ball entity.
    target_height: Height of the target apex above the resting state, in metres.
    ball_rest_z: ``z_rest``, the ball centre's absolute world z when the ball rests
      on the plate in the carry pose, in metres.

  Returns:
    ``(energy, target)``: shape ``(num_envs,)`` in J/kg, and a scalar in J/kg.
  """
  energy = ball_vertical_energy(env, ball_cfg, ball_rest_z)
  gravity = abs(float(env.sim.mj_model.opt.gravity[2]))
  return energy, gravity * target_height


def ball_energy(
  env: ManagerBasedRlEnv,
  plate_cfg: SceneEntityCfg,
  ball_cfg: SceneEntityCfg,
  target_height: float,
  sigma: float,
  ball_rest_z: float,
) -> torch.Tensor:
  r"""Reward the ball's energy matching a bounce to ``target_height``.

  .. math:: r = \exp(-((E - E^*) / \sigma)^2)

  with :math:`E` and :math:`E^*` as :func:`_ball_energy_and_target` defines them.
  One for a ball at the top of a bounce of exactly ``target_height``: its energy is
  then :math:`E^*`. The reset state does **not** earn 1.0: the ball spawns at
  :math:`h = 0.20` m, below the 0.40 m target. The reward falls off symmetrically
  in J/kg as the ball carries too little or too much energy.

  **Dense and stateless:** the reward is a function of the current state only, paid
  on every control step, with no per-episode state and no reset hook. A ball resting
  on the plate has :math:`h = 0` and :math:`E \approx 0`, far below :math:`E^*`, so
  parking earns almost nothing; the one way to park at the target is to hold the
  racket 0.40 m high, which :func:`racket_lift` charges for. A per-bounce apex reward
  is the opposite: zero on every step except the apex, so it gives no gradient while
  the ball is in flight.

  Args:
    plate_cfg: Accepted so the ball terms share one parameter dict; the energy
      datum is a fixed world height, so this term does not read the robot.
    ball_cfg: Ball entity.
    target_height: Height of the target apex above the resting state, in metres.
    sigma: Falloff of the Gaussian in energy, in J/kg, the same units as ``E``.
    ball_rest_z: ``z_rest``, the resting ball centre's absolute world z, in metres.

  Returns:
    Shape ``(num_envs,)``, between 0 and 1.
  """
  energy, target = _ball_energy_and_target(env, ball_cfg, target_height, ball_rest_z)
  return torch.exp(-torch.square((energy - target) / sigma))


def ball_energy_value(
  env: ManagerBasedRlEnv,
  plate_cfg: SceneEntityCfg,
  ball_cfg: SceneEntityCfg,
  target_height: float,
  ball_rest_z: float,
) -> torch.Tensor:
  r"""The ball's energy per unit mass :math:`E`, for a metrics term.

  With ``reduce="mean"`` the metric reports the episode mean of :math:`E`, whose
  target is :math:`E^* = g \, h_{\mathrm{target}}`. The value is recomputed from
  the current simulation state, exactly like :func:`ball_energy`, so it cannot read
  a stale flag: there is no reward-term state to go stale.

  Args:
    plate_cfg: Accepted so the ball terms share one parameter dict; the energy
      datum is a fixed world height, so this term does not read the robot.
    ball_cfg: Ball entity.
    target_height: Height of the target apex above the resting state, in metres.
    ball_rest_z: ``z_rest``, the resting ball centre's absolute world z, in metres.

  Returns:
    Shape ``(num_envs,)``, in J/kg.
  """
  energy, _ = _ball_energy_and_target(env, ball_cfg, target_height, ball_rest_z)
  return energy


def ball_energy_error(
  env: ManagerBasedRlEnv,
  plate_cfg: SceneEntityCfg,
  ball_cfg: SceneEntityCfg,
  target_height: float,
  ball_rest_z: float,
) -> torch.Tensor:
  """The distance :math:`|E - E^*|` in J/kg, for a metrics term.

  With ``reduce="mean"`` the metric reports the episode mean distance from the
  target energy, which is the quantity the reward shrinks but cannot report as a
  physical number. Like :func:`ball_energy`, it is recomputed from the current
  state, so it cannot go stale.

  Returns:
    Shape ``(num_envs,)``, in J/kg, zero when the energy is exactly on target.
  """
  energy, target = _ball_energy_and_target(env, ball_cfg, target_height, ball_rest_z)
  return torch.abs(energy - target)


def racket_lift(
  env: ManagerBasedRlEnv,
  plate_cfg: SceneEntityCfg,
  plate_top_rest_z: float,
  deadband: float,
) -> torch.Tensor:
  r"""Metres the plate has been lifted above its carry height, for a penalty.

  .. math::

    \max(0, z_{\mathrm{plate}} - z_{\mathrm{rest}} - d)

  where ``plate_top_rest_z`` is the plate top surface's **absolute world height**
  in the carry pose and ``deadband`` is ``d``.

  This closes the parking loophole that the energy reward leaves open. A ball resting
  on the plate has ``h`` equal to the plate's world lift above the carry pose, so
  holding the racket 0.40 m high makes the ball read exactly :math:`E^*` on every
  step. A term that returned this quantity with weight
  ``-JUGGLE_RACKET_LIFT_WEIGHT`` makes the net reward non-increasing in lift, so a
  partial lift cannot farm the energy reward.

  The measurement is in **world z**, like the energy datum, so a lift is charged
  whether the legs or the arm produce it. A pelvis-relative measurement would miss
  the leg lift that the world energy datum makes visible: raising the whole robot
  0.40 m carries the resting ball to :math:`E^*` and must pay for it, or the pair
  of terms can still be farmed by crouching and standing. The deadband leaves normal
  jostling uncharged.

  Args:
    plate_cfg: Racket body; the plate top surface is defined by it.
    plate_top_rest_z: Absolute world z of the plate top surface in the carry pose,
      in metres.
    deadband: Lift that is not charged, in metres.

  Returns:
    Shape ``(num_envs,)``, non-negative metres. Install it with a **negative**
    weight; the returned value is the excess lift, not the penalty.
  """
  lift = _plate_top_z(env, plate_cfg) - plate_top_rest_z
  return torch.clamp(lift - deadband, min=0.0)


class ball_bounce_apex(ManagerTermBase):
  r"""Pay once per bounce, when the ball's apex is at the target height.

  .. math:: r = \exp(-((h_\mathrm{apex} - h_\mathrm{target}) / \sigma)^2)

  paid on the apex step and 0 on every other step, where :math:`h_\mathrm{apex}` is
  the gap under the ball at the top of the bounce, as
  :func:`_ball_gap_and_vertical_speed` defines it.

  A per-step height reward cannot express a bounce: a ball parked at the target
  height collects it on every step of the episode, which is exactly the behaviour
  this term rules out. Paying once per apex leaves bouncing as the only way to earn
  it again.

  **Apex detection.** An apex is a step where the ball's **world vertical** velocity
  crosses from positive to negative. Gravity defines that axis, so it is the axis the
  ball really turns around on; the plate normal is not, because the plate tilts.

  **Rejecting a parked ball.** A ball resting on the plate has a vertical speed that
  wobbles around zero, so it produces sign crossings of its own. Two gates reject
  them: the apex gap must reach ``min_clearance``, and the rise's **peak upward
  speed** must reach ``min_speed``. The peak, and not the previous step's speed:
  gravity changes the vertical speed by 0.20 m/s over one 0.02 s control step, so the
  last positive sample before a crossing is at most 0.20 m/s whatever the launch was,
  and a ``min_speed`` of 0.30 would never open. The peak is the launch speed the ball
  actually reached.

  With ``min_clearance`` at 3 cm the speed gate is mostly redundant, because a 3 cm
  rise already implies a 0.77 m/s launch. It stays as a cheap guard against a small
  crossing produced by contact jitter.

  Args:
    plate_cfg: Racket body; the plate frame is defined by it.
    ball_cfg: Ball entity.
    target_height: Apex gap that earns exactly 1.0, in metres.
    sigma: Falloff of the Gaussian in the apex gap, in metres.
    min_clearance: Smallest apex gap that can count, in metres.
    min_speed: Smallest peak upward speed of the rise that can count, in m/s.

  Returns:
    Shape ``(num_envs,)``, between 0 and 1, and 0 on every non-apex step.

  Per-environment state, all cleared on reset: the previous step's vertical speed,
  the peak upward speed of the current rise, a flag that marks the first step of an
  episode (no crossing can be read against a stale value), and :attr:`last_bounce`.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg  # No parameters are needed before the first call.
    super().__init__(env)
    self._prev_vertical_speed = torch.zeros(env.num_envs, device=env.device)
    self._peak_rising_speed = torch.zeros(env.num_envs, device=env.device)
    self._first_step = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    self.last_bounce = torch.zeros(env.num_envs, device=env.device)
    """Per-step bounce flag: 1.0 on a counted apex, 0.0 otherwise.

    Public because :func:`ball_bounce_count` reports it as a metrics term. It is
    overwritten on every call, so a metric reads the step the reward manager has
    just computed.
    """

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    plate_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    target_height: float,
    sigma: float,
    min_clearance: float,
    min_speed: float,
  ) -> torch.Tensor:
    gap, vertical_speed = _ball_gap_and_vertical_speed(env, plate_cfg, ball_cfg)
    # The peak includes this step, so a bounce that is already past its apex when
    # the episode starts still carries the speed it rose with.
    peak = torch.maximum(self._peak_rising_speed, torch.clamp(vertical_speed, min=0.0))
    crossed = (self._prev_vertical_speed > 0.0) & (vertical_speed <= 0.0)
    apex = crossed & ~self._first_step
    counted = apex & (gap >= min_clearance) & (peak >= min_speed)
    bell = torch.exp(-torch.square((gap - target_height) / sigma))
    reward = torch.where(counted, bell, torch.zeros_like(bell))
    # An apex ends the rise, so the next bounce starts its own peak. A crossing the
    # gates rejected resets it too: resting jitter must not accumulate a launch.
    self._peak_rising_speed = torch.where(apex, torch.zeros_like(peak), peak)
    # Cloned: the entity buffer this comes from is refreshed in place.
    self._prev_vertical_speed = vertical_speed.clone()
    self._first_step = torch.zeros_like(self._first_step)
    self.last_bounce = counted.float()
    return reward

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._prev_vertical_speed[env_ids] = 0.0
    self._peak_rising_speed[env_ids] = 0.0
    self._first_step[env_ids] = True
    self.last_bounce[env_ids] = 0.0


def ball_bounce_count(
  env: ManagerBasedRlEnv,
  reward_term_name: str = "ball_bounce_apex",
) -> torch.Tensor:
  """Per-step bounce count of :class:`ball_bounce_apex`, for a metrics term.

  One counted apex is 1.0 for one step and 0.0 otherwise, so a metrics entry with
  ``reduce="sum"`` reports bounces per episode::

    metrics = {
      "ball_bounces": MetricsTermCfg(func=mdp.ball_bounce_count, reduce="sum"),
    }

  The flag is read from the reward term, which the reward manager computes before
  the metrics manager on the same step. **Keep the apex reward weight non-zero:**
  ``RewardManager.compute`` skips terms whose weight is 0, so the flag would go
  stale and the count would freeze at whatever it last held.

  Args:
    reward_term_name: Reward manager key of the ``ball_bounce_apex`` term.

  Returns:
    Shape ``(num_envs,)``, 0.0 or 1.0 per environment per step.
  """
  term = env.reward_manager.get_term_cfg(reward_term_name).func
  if not isinstance(term, ball_bounce_apex):
    raise TypeError(
      f"reward term '{reward_term_name}' is not ball_bounce_apex, got "
      f"{type(term).__name__}"
    )
  return term.last_bounce


class ball_at_rest(ManagerTermBase):
  """Indicator that the ball is parked on the plate, for a negative weight.

  Returns 1.0 when the ball has been still on the plate **continuously** for
  ``grace_s`` seconds, and 0.0 otherwise, so the config can give the term a negative
  weight. It exists to remove the "park the ball and stand still" solution that a
  bounce reward otherwise leaves open.

  Still means the gap under the ball is below ``max_clearance`` and its world
  vertical speed is below ``max_speed``. The grace period is what separates a park
  from a bounce: at the bottom of every contact the ball passes through the same low,
  slow state, but only for as long as its speed takes to grow past ``max_speed``
  (about 40 ms at a 2.4 m/s launch, i.e. two control steps), while a parked ball
  stays there for the whole episode.

  The counter is per environment, is cleared on reset, and restarts from zero the
  moment either condition breaks, so one environment cannot hold another above the
  threshold.

  Args:
    plate_cfg: Racket body; the plate frame is defined by it.
    ball_cfg: Ball entity.
    max_clearance: Largest gap under the ball that still counts as resting, in
      metres.
    max_speed: Largest world vertical speed that still counts as resting, in m/s.
    grace_s: Seconds the conditions must hold without a break.

  Returns:
    Shape ``(num_envs,)``, 0.0 or 1.0.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    # Whole control steps: the condition is sampled once per step, so a grace
    # shorter than one step is one step.
    self._grace_steps = max(1, round(float(cfg.params["grace_s"]) / env.step_dt))
    self._rest_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    plate_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    max_clearance: float,
    max_speed: float,
    grace_s: float,
  ) -> torch.Tensor:
    del grace_s  # Resolved to whole control steps in ``__init__``.
    gap, vertical_speed = _ball_gap_and_vertical_speed(env, plate_cfg, ball_cfg)
    resting = (gap < max_clearance) & (torch.abs(vertical_speed) < max_speed)
    self._rest_steps = torch.where(
      resting, self._rest_steps + 1, torch.zeros_like(self._rest_steps)
    )
    return (self._rest_steps >= self._grace_steps).float()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._rest_steps[env_ids] = 0


_BALL_CONTACT_MODES = ("impact_count", "flight_time", "sustained_contact")


def _consecutive_true_runs(
  mask: torch.Tensor, carry: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Consecutive-True counts over one window of substeps, oldest first.

  Args:
    mask: ``[B, H]`` bool substeps.
    carry: ``[B]`` long, consecutive True substeps ending just before the
      window.

  Returns:
    ``run``: ``[B, H]``, the run of True substeps ending at each substep.
    ``end``: ``[B]``, the run ending at the last substep. It is the carry for
      the next window.
  """
  n_envs, h = mask.shape
  idx = torch.arange(h, device=mask.device)
  # Position of the last False at or before each substep; -1 if there is none.
  last_false = torch.where(~mask, idx, -1).cummax(dim=1).values
  run = torch.where(
    mask,
    torch.where(
      last_false >= 0,
      idx - last_false,
      idx + 1 + carry.unsqueeze(1),
    ),
    torch.zeros((n_envs, h), dtype=idx.dtype, device=mask.device),
  )
  return run, run[:, -1]


class ball_contact_metric(ManagerTermBase):
  """Contact-derived ball metrics, selected by ``mode``, for ``metrics`` entries.

  One class because the three metrics share one input, the racket-ball
  :class:`~mjlab.sensor.ContactSensor` named by ``sensor_name``, and their
  per-episode state. Each metrics entry gets its own instance, so the modes do
  not share counters.

  **Substep resolution.** The racket-ball contact lasts about 9-10 ms while the
  control step is 20 ms, so one sample of the sensor's ``found`` field per step
  misses about half of the contacts. When the sensor is configured with
  ``history_length`` equal to the environment's decimation, ``force_history``
  holds every physics substep of the control step. ``contact`` for one step is
  then "any substep in the window had a non-zero contact force in any slot".

  **Modes**

  * ``"impact_count"`` returns 1.0 on a rising edge of the per-step contact flag
    and 0.0 otherwise. A continuous contact is one impact, however many control
    steps it lasts, and a contact that starts and ends between two samples now
    counts once instead of zero. Use ``reduce="sum"`` for impacts per episode.
  * ``"flight_time"`` keeps a running mean of the airborne time that ends at
    each substep-level rising edge, in seconds, accumulated in substeps. A
    contact substep never adds to a flight. The value returned on every step is
    the running mean. Use ``reduce="last"``: the manager's ``"mean"`` divides by
    the episode's step count, which is not the number of flights.
  * ``"sustained_contact"`` returns the longest continuous contact visible in
    the control step, in seconds, accumulated in substeps. A contact that ends
    inside the window still reports its length. Use ``reduce="max"`` for the
    longest single contact of the episode.

  **Limit.** Impact count is a rising edge of the per-control-step flag, so it
  merges two contacts whenever no control step between them is fully airborne:
  in particular, two contacts inside one control step count as one impact.
  Flight time and sustained contact still resolve that gap in substeps.

  **Fallback.** Without ``force_history`` (``history_length == 0``) the metrics
  read the instantaneous ``found`` field at control-step resolution. That is the
  pre-history behaviour and can miss a contact that starts and ends between two
  samples.

  **Reset.** Every counter and the previous contact flag are cleared on episode
  reset. A contact on the first sampled step after reset counts as a rising edge,
  because the episode has no earlier state.

  Args:
    sensor_name: Scene name of the :class:`~mjlab.sensor.ContactSensor`.
    mode: One of ``"impact_count"``, ``"flight_time"`` and
      ``"sustained_contact"``.

  Returns:
    Shape ``(num_envs,)``. The units depend on ``mode``: impacts, seconds, and
    seconds.
  """

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    mode = str(cfg.params["mode"])
    if mode not in _BALL_CONTACT_MODES:
      msg = (
        f"ball_contact_metric mode '{mode}' is unknown. "
        f"Valid modes are {_BALL_CONTACT_MODES}."
      )
      raise ValueError(msg)
    self._mode = mode
    self._sensor_name = str(cfg.params["sensor_name"])
    self._step_dt = env.step_dt
    self._substep_dt = env.physics_dt
    sensor: ContactSensor = env.scene[self._sensor_name]
    history_length_or_raise(
      sensor, self._step_dt, self._substep_dt, "ball_contact_metric"
    )
    self._prev_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self._air_run = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self._contact_run = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self._flight_sum = torch.zeros(env.num_envs, device=env.device)
    self._flight_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def __call__(
    self, env: ManagerBasedRlEnv, sensor_name: str, mode: str
  ) -> torch.Tensor:
    del sensor_name, mode  # Resolved in ``__init__``.
    sensor: ContactSensor = env.scene[self._sensor_name]
    data = sensor.data
    contact = step_contact(data)
    if data.force_history is not None:
      # [B, N, H, 3], H = decimation, so the window is this control step.
      substeps = substep_contact(data.force_history)
      substep_dt = self._substep_dt
    else:
      # One sample per control step: the pre-history fallback.
      substeps = contact.unsqueeze(-1)
      substep_dt = self._step_dt

    rising = contact & ~self._prev_contact
    self._prev_contact = contact.clone()

    if self._mode == "impact_count":
      return rising.float()

    air_carry = self._air_run
    contact_carry = self._contact_run
    air_run, air_end = _consecutive_true_runs(~substeps, air_carry)
    contact_run, contact_end = _consecutive_true_runs(substeps, contact_carry)
    self._air_run = air_end
    self._contact_run = contact_end

    if self._mode == "flight_time":
      # ``air_run`` at substep i-1 is the airborne run ending just before
      # substep i; at substep 0 it is the carry from the previous control step.
      air_before = torch.cat([air_carry.unsqueeze(1), air_run[:, :-1]], dim=1)
      # A rising edge closes a flight. At episode start the carry is zero, so a
      # contact on the first sampled substep has no flight before it.
      rising_substep = substeps & (air_before > 0)
      self._flight_sum = self._flight_sum + (
        air_before.float() * substep_dt * rising_substep
      ).sum(dim=-1)
      self._flight_count = self._flight_count + rising_substep.sum(dim=-1)
      mean = self._flight_sum / self._flight_count.clamp(min=1).float()
      return torch.where(self._flight_count > 0, mean, torch.zeros_like(mean))

    return contact_run.max(dim=-1).values.float() * substep_dt

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._prev_contact[env_ids] = False
    self._air_run[env_ids] = 0
    self._contact_run[env_ids] = 0
    self._flight_sum[env_ids] = 0.0
    self._flight_count[env_ids] = 0


def _held_mask(env: ManagerBasedRlEnv, hold_event: str | None) -> torch.Tensor | None:
  """The per-environment hold flag of a ``hold_ball`` event term, or None.

  Duck-typed on purpose: the hold term lives in :mod:`mjlab.tasks.velocity.mdp.events`,
  which imports this module, so importing it back would be circular. Any term that
  exposes a boolean ``held`` buffer works, and the assertion names the expected
  term rather than silently reading nothing.
  """
  if hold_event is None:
    return None
  term = env.event_manager.get_term_cfg(hold_event).func
  held = getattr(term, "held", None)
  assert held is not None, (
    f"'{hold_event}' must be a hold_ball event term exposing a 'held' buffer"
  )
  return held


def ball_energy_after_release(
  env: ManagerBasedRlEnv,
  plate_cfg: SceneEntityCfg,
  ball_cfg: SceneEntityCfg,
  target_height: float,
  sigma: float,
  ball_rest_z: float,
  hold_event: str = "hold_ball",
) -> torch.Tensor:
  """``ball_energy``, scored only after the ball has been released.

  The formula is unchanged; this term gates it to the post-release phase. While
  the ball is held it is world-fixed with zero velocity, so ``E`` is a per-episode
  constant that the policy cannot influence: at a 0.40 m hold it is exactly
  :math:`E^*` and the term would pay a full 1.0 on every waiting step for doing
  nothing, and with a randomised hold height a high release would be worth more
  than a low one for no reason a skill explains. Gating removes both.

  Args:
    plate_cfg: Accepted so the ball terms share one parameter dict.
    ball_cfg: Ball entity.
    target_height: Target apex above the resting state, in metres.
    sigma: Gaussian falloff in J/kg.
    ball_rest_z: ``z_rest``, the resting ball centre's absolute world z.
    hold_event: Name of the ``hold_ball`` event term that owns the hold state.

  Returns:
    Shape ``(num_envs,)``, between 0 and 1, zero while held.
  """
  reward = ball_energy(env, plate_cfg, ball_cfg, target_height, sigma, ball_rest_z)
  held = _held_mask(env, hold_event)
  if held is None:
    return reward
  return torch.where(held, torch.zeros_like(reward), reward)


def racket_under_ball(
  env: ManagerBasedRlEnv,
  plate_cfg: SceneEntityCfg,
  ball_cfg: SceneEntityCfg,
  hold_event: str = "hold_ball",
  plateau: float = 0.010,
  sigma: float = 0.030,
  level_power: float = 2.0,
  speed_sigma: float = 0.5,
) -> torch.Tensor:
  r"""Reward putting a level, quiet racket under the held ball.

  .. math::

    r = \exp(-(\max(0, d - p) / \sigma)^2) \; \cos^{k}\!\theta \; e^{-(v/\sigma_v)^2}

  with ``d`` the ball's plate-frame horizontal offset from the plate centre,
  :math:`\cos\theta` the world-z component of the plate normal (so a tilted plate
  cannot satisfy the alignment by leaning toward the ball), and ``v`` the ball's
  speed relative to the plate. While the ball is held it does not move, so ``v`` is
  the plate's own speed: this factor is the "wait quietly" part.

  This is the term ``ball_centred`` cannot be: its offset is plate-frame only, so
  tilting the plate toward an elevated ball improves it without placing the racket
  below the ball, and nothing in the reward set specifies a waiting posture. The
  term is gated to the hold phase, so it cannot pay for anything after release.

  Args:
    plate_cfg: Racket body; the plate frame is defined by it.
    ball_cfg: Ball entity.
    hold_event: Name of the ``hold_ball`` event term that owns the hold state.
    plateau: Offset inside which the positioning factor is exactly 1, in metres.
    sigma: Positioning falloff beyond the plateau, in metres.
    level_power: Exponent on the plate normal's world-z component.
    speed_sigma: Scale of the quiet factor, in metres per second.

  Returns:
    Shape ``(num_envs,)``, between 0 and 1, zero after release.
  """
  position, velocity = plate_relative_ball_state(env, plate_cfg, ball_cfg)
  distance = torch.linalg.norm(position[:, :2], dim=1)
  outside = torch.clamp(distance - plateau, min=0.0)
  aligned = torch.exp(-torch.square(outside / sigma))

  plate: Entity = env.scene[plate_cfg.name]
  body_id = plate_cfg.body_ids
  if isinstance(body_id, list):
    body_id = body_id[0]
  normal_w = quat_apply(
    plate.data.body_link_quat_w[:, body_id],
    torch.tensor((0.0, 0.0, 1.0), device=env.device).expand(env.num_envs, 3),
  )
  level = torch.clamp(normal_w[:, 2], min=0.0).pow(level_power)

  speed = torch.linalg.norm(velocity, dim=1)
  quiet = torch.exp(-torch.square(speed / speed_sigma))

  reward = aligned * level * quiet
  held = _held_mask(env, hold_event)
  if held is None:
    return reward
  return torch.where(held, reward, torch.zeros_like(reward))


class ball_lost_penalty(ManagerTermBase):
  """The lost-ball cost that replaces ending the episode when the ball falls.

  ``ball_lost`` ends an episode the moment the ball drops below a fixed world
  height. That is the right detector and the wrong ending: the robot is trained up
  to the instant the ball is lost and never afterwards, so the state that actually
  hurts hardware - the ball gone and the policy left with nothing to juggle - has no
  learned behaviour at all.

  This term prices that state instead. While the ball has stayed below
  ``min_world_z`` for ``grace_steps`` consecutive control steps the term returns 1.0,
  so its weight is a cost per second of being lost. ``ball_no_bounce`` still ends the
  episode 2.5 s after the last strike, which bounds the tail: the cost prices the
  aftermath, and the terminator bounds it.

  The grace is the same consecutive-step count ``ball_lost`` uses, so a ball that
  dips below the threshold during a strike is not charged. The counter is per
  environment and clears on reset.

  Args:
    ball_cfg: Ball entity.
    min_world_z: Ball-centre height above the ground at or below which the ball
      counts as lost, in metres.
    grace_steps: Consecutive control steps below the threshold before charging.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    n, device = env.num_envs, env.device
    self._lost_steps = torch.zeros(n, dtype=torch.long, device=device)
    self.lost = torch.zeros(n, dtype=torch.bool, device=device)
    """True while the ball counts as lost."""
    self.lost_events = torch.zeros(n, device=device)
    """Lost-ball transitions in the current episode, logging only."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    ball_cfg: SceneEntityCfg,
    min_world_z: float,
    grace_steps: int,
  ) -> torch.Tensor:
    ball: Entity = env.scene[ball_cfg.name]
    lost = ball.data.root_link_pos_w[:, 2] <= min_world_z
    self._lost_steps = torch.where(
      lost, self._lost_steps + 1, torch.zeros_like(self._lost_steps)
    )
    now_lost = self._lost_steps >= max(int(grace_steps), 1)
    self.lost_events += (now_lost & ~self.lost).float()
    self.lost = now_lost
    return now_lost.float()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._lost_steps[env_ids] = 0
    self.lost[env_ids] = False
    self.lost_events[env_ids] = 0.0


def lost_ball_state(
  env: ManagerBasedRlEnv, reward_term: str, field: str = "lost"
) -> torch.Tensor:
  """Logging-only read of a :class:`ball_lost_penalty` instance's state."""
  term = env.reward_manager.get_term_cfg(reward_term).func
  assert isinstance(term, ball_lost_penalty), (
    f"'{reward_term}' must be a class-based ball_lost_penalty reward term"
  )
  return getattr(term, field).float()


def pose_quadratic(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  std: float = 0.05,
) -> torch.Tensor:
  """Mean squared joint deviation from the default pose, in units of ``std``.

  The shipped ``pose`` term is ``exp(-mean((q - q*)^2 / std^2))``. It is a
  fine-tolerance term: at a 30 degree deviation on ten joints the exponent is about
  -32, so the value and its gradient are both numerically zero. A policy that has
  learned a twisted posture therefore pays nothing and, more importantly, is pulled
  by nothing -- which is exactly the pose in the S9b2 frame, where the torso is
  yawed and the racket arm is carried across the body.

  This term is the negative log of the same quantity, i.e. the same reference and the
  same tolerance, but with a gradient that never vanishes:

  ``mean_j((q_j - q*_j)^2 / std_j^2)``

  With a negative weight it pulls from any distance. It agrees with ``pose`` near the
  default pose and keeps pulling far away, where ``pose`` has saturated.

  Args:
    asset_cfg: Robot with the joints to score.
    std: Tolerance in radians, one value for every joint. 0.05 matches the shipped
      ``pose`` term's ``std_standing``, so the two terms describe the same target.
  """
  asset: Entity = env.scene[asset_cfg.name]
  current = asset.data.joint_pos[:, asset_cfg.joint_ids]
  reference = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  return torch.mean(torch.square((current - reference) / float(std)), dim=1)
