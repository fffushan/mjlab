"""Camera visibility, shared by the field-of-view reward and the ball stream.

One definition of "the camera can see the ball", so the reward and the measurement
stream cannot drift apart. The geometry is the head body frame transformed by a
constant mount; axes follow the ROS optical convention, +x right, +y down, +z forward.

The real estimator publishes nothing for a frame in which the ball was not detected
(Rule 0 of ``TENNIS-BALL-ESTIMATOR-CONTRACT.md``). The simulator had no equivalent, so
the actor could rely on a ball position at every step. :func:`ball_in_camera_region`
is the gate the stream uses to reproduce that silence.
"""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_mul


def camera_pose_w(
  env,
  head_cfg: SceneEntityCfg,
  mount_xyz: tuple[float, float, float],
  mount_quat: tuple[float, float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
  """World position and orientation of a camera rigidly mounted on one body.

  Args:
    head_cfg: The body the camera is mounted on. Must resolve to exactly one body.
    mount_xyz: Camera position in that body's frame, metres.
    mount_quat: Camera orientation in that body's frame, wxyz.

  Returns:
    ``(position (num_envs, 3), orientation (num_envs, 4))`` in world coordinates.
  """
  head: Entity = env.scene[head_cfg.name]
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
  return cam_pos_w, cam_quat_w


def ball_in_camera_region(
  env,
  ball_cfg: SceneEntityCfg,
  head_cfg: SceneEntityCfg,
  mount_xyz: tuple[float, float, float],
  mount_quat: tuple[float, float, float, float],
  half_angles: tuple[float, float, float, float],
  min_range_m: float = 0.0,
  max_range_m: float = float("inf"),
) -> torch.Tensor:
  """Boolean mask: is the ball inside the usable camera region at this instant?

  A rectangular frame test in the optical frame, plus a straight-line range gate. A
  ball at or behind the camera plane is never visible. The frame is asymmetric, so
  the horizontal and vertical limits are separate for the two sides; the delivered
  image is not centred on the optical axis.

  Args:
    half_angles: ``(left, right, up, down)`` half-angles from the optical axis to the
      usable edge, radians. ``left`` bounds a negative ``theta_x`` and ``right`` a
      positive one; ``up`` bounds a negative ``theta_y`` (the optical +y axis points
      down) and ``down`` a positive one.
    min_range_m: Near limit of the usable region, straight-line from the camera
      centre. The estimator cannot triangulate a ball closer than this.
    max_range_m: Far limit, also straight-line from the camera centre, so the region
      it closes is a sphere and not a forward-depth slab.

  Returns:
    ``(num_envs,)`` boolean tensor, True where the ball is inside the region.
  """
  cam_pos_w, cam_quat_w = camera_pose_w(env, head_cfg, mount_xyz, mount_quat)
  ball: Entity = env.scene[ball_cfg.name]
  delta_cam = quat_apply_inverse(cam_quat_w, ball.data.root_link_pos_w - cam_pos_w)
  left, right, up, down = half_angles

  forward = delta_cam[:, 2]
  in_front = forward > 1e-6
  safe_forward = torch.where(in_front, forward, torch.ones_like(forward))
  theta_x = torch.atan2(delta_cam[:, 0], safe_forward)
  theta_y = torch.atan2(delta_cam[:, 1], safe_forward)

  inside = (
    (theta_x >= -left) & (theta_x <= right) & (theta_y >= -up) & (theta_y <= down)
  )
  range_m = torch.linalg.vector_norm(delta_cam, dim=1)
  in_range = (range_m >= min_range_m) & (range_m <= max_range_m)
  return in_front & inside & in_range
