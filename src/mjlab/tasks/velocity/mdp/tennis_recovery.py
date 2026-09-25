"""Tennis-end recovery: paired reset event and velocity-command specialization.

This module implements the runtime component of the X2 tennis-end recovery
fine-tuning experiment. It provides:

- :class:`TennisRecoveryResetEvent`: a class-based reset event that restores a
  full reference state (joint pos/vel, pelvis world pose and world-frame linear
  and angular velocity) from an endpoint pool for *recovery* environments,
  recentered to the environment origin in XY with the reference height preserved.
  *Retention* environments keep the original reset sequence untouched.

- :class:`TennisRecoveryVelocityCommand`: a velocity-command subclass that
  keeps recovery environments at exactly zero twist across resets, resamples,
  heading updates and curriculum changes, while retention environments use the
  original :class:`UniformVelocityCommand` distribution. The zero command never
  writes physical qvel.

- :class:`TennisRecoveryVelocityCommandCfg`: the command configuration with
  pool/group fields exposed for the factory.

The endpoint pool is loaded lazily at environment instantiation (a local
import at this explicit runtime boundary is acceptable). Tests inject a
synthetic pool via the ``pool`` attribute on the event instance.

Group assignment is fixed for the environment lifetime, seeded and
deterministic. The recovery count is ``round(recovery_fraction * num_envs)``.
For tiny envs where the rounded count equals ``num_envs`` (e.g. 2 envs at 0.8
fraction → 2 recovery, 0 retention), use ``force_mode="retention"`` or
``force_mode="recovery"`` to override.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import torch

from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
)

if TYPE_CHECKING:
  import numpy.typing as npt

  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


# ---------------------------------------------------------------------------
# Structural protocol for the endpoint pool.
# ---------------------------------------------------------------------------


@runtime_checkable
class EndpointPoolProtocol(Protocol):
  """Structural protocol matching the production ``EndpointPool`` interface.

  The production type is ``mjlab.tasks.velocity.mdp.tennis_endpoint_pool
  .EndpointPool``. Tests may supply any object with these attributes; the
  runtime does not require the real loader.
  """

  @property
  def joint_pos(self) -> npt.NDArray[np.float32]: ...

  @property
  def joint_vel(self) -> npt.NDArray[np.float32]: ...

  @property
  def root_pos_w(self) -> npt.NDArray[np.float32]: ...

  @property
  def root_quat_w(self) -> npt.NDArray[np.float32]: ...

  @property
  def root_lin_vel_w(self) -> npt.NDArray[np.float32]: ...

  @property
  def root_ang_vel_w(self) -> npt.NDArray[np.float32]: ...

  @property
  def trajectory_ids(self) -> npt.NDArray[np.int64]: ...

  @property
  def frame_indices(self) -> npt.NDArray[np.int64]: ...

  @property
  def joint_names(self) -> tuple[str, ...]: ...

  @property
  def body_names(self) -> tuple[str, ...]: ...

  @property
  def manifest(self) -> dict[str, Any]: ...

  def __len__(self) -> int: ...

  def sample_indices(
    self, count: int, rng: np.random.Generator
  ) -> "npt.NDArray[np.int64]": ...


# ---------------------------------------------------------------------------
# Group assignment.
# ---------------------------------------------------------------------------


def _assign_recovery_groups(
  num_envs: int,
  recovery_fraction: float,
  seed: int,
  force_mode: str,
  device: str,
) -> torch.Tensor:
  """Return a fixed boolean mask: ``True`` for recovery, ``False`` for retention.

  The recovery count is ``round(recovery_fraction * num_envs)`` (Python's
  banker's rounding). With 5 envs and 0.8 fraction this gives 4 recovery + 1
  retention; with 10 envs, 8 + 2. For tiny envs where the rounded count equals
  ``num_envs`` (e.g. 2 envs at 0.8 → 2 recovery, 0 retention), use
  ``force_mode="retention"`` or ``"recovery"`` to override.

  The assignment is shuffled with a CPU generator seeded by ``seed`` so it is
  deterministic and device-independent.
  """
  if force_mode == "recovery":
    return torch.ones(num_envs, dtype=torch.bool, device=device)
  if force_mode == "retention":
    return torch.zeros(num_envs, dtype=torch.bool, device=device)
  if force_mode != "none":
    raise ValueError(
      f"force_mode must be 'none', 'recovery', or 'retention', got {force_mode!r}"
    )

  if not isinstance(recovery_fraction, int | float):
    raise TypeError(
      f"recovery_fraction must be a float, got {type(recovery_fraction).__name__}"
    )
  if (
    not np.isfinite(recovery_fraction)
    or recovery_fraction < 0.0
    or recovery_fraction > 1.0
  ):
    raise ValueError(
      f"recovery_fraction must be a finite value in [0, 1], got {recovery_fraction}"
    )

  n_recovery = int(round(recovery_fraction * num_envs))
  n_recovery = max(0, min(n_recovery, num_envs))

  gen = torch.Generator(device="cpu")
  gen.manual_seed(seed)
  perm = torch.randperm(num_envs, generator=gen)

  mask = torch.zeros(num_envs, dtype=torch.bool, device="cpu")
  mask[perm[:n_recovery]] = True
  return mask.to(device=device)


# ---------------------------------------------------------------------------
# Recovery reset event.
# ---------------------------------------------------------------------------


class TennisRecoveryResetEvent:
  """Class-based reset event that restores full reference state for recovery envs.

  On each episode reset, for each recovery environment being reset, this event
  samples a random row from the cached endpoint pool and writes:

  - Joint position and velocity (via ``write_joint_state_to_sim``).
  - Pelvis world pose (position + wxyz quaternion) via
    ``write_root_link_pose_to_sim``.
  - Pelvis world-frame linear and angular velocity via
    ``write_root_link_velocity_to_sim``, which internally converts world omega
    to local angular qvel using the inverse root quaternion.

  XY is recentered to the environment origin; the reference height (Z) is
  preserved. Optional yaw augmentation rotates the orientation and world
  velocities consistently, preserving the authored roll/pitch.

  Retention environments are not touched: the original reset events
  (``reset_base``, ``reset_robot_joints``) handle them.

  The pool is loaded lazily on first use. For tests, set ``self.pool`` to a
  synthetic object matching :class:`EndpointPoolProtocol` before calling.
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    self._env = env
    self._device = env.device
    self._num_envs = env.num_envs

    params = cfg.params
    self._asset_cfg: SceneEntityCfg = params["asset_cfg"]
    self._robot: Entity = env.scene[self._asset_cfg.name]

    self._pool_directory = params.get("pool_directory")
    self._last_n_frames = params.get("last_n_frames", 10)
    self._split = params.get("split", "train")
    self._validation_fraction = params.get("validation_fraction", 0.2)
    self._pool_seed = params.get("seed", 42)

    recovery_fraction = params.get("recovery_fraction", 0.8)
    seed = params.get("seed", 42)
    force_mode = params.get("force_mode", "none")
    self.recovery_env_mask = _assign_recovery_groups(
      self._num_envs, recovery_fraction, seed, force_mode, self._device
    )

    self._yaw_aug_range: tuple[float, float] | None = params.get("yaw_aug_range")

    # Pool: lazily loaded on first reset. Tests set this directly.
    self.pool: EndpointPoolProtocol | None = None
    # Cached device tensors.
    self._cached: dict[str, torch.Tensor] | None = None

    # Evaluation-only row override: when set to a tensor of shape [num_envs],
    # the event uses these pool row indices instead of random sampling.
    # This is a narrow evaluation-only override; default training resets
    # use random uniform sampling. Set to None for normal training behavior.
    self.eval_row_indices: torch.Tensor | None = None

    # Per-env provenance: last sampled row index, trajectory ID and frame
    # index for downstream coverage/evaluation metrics. Not in actor
    # observations.
    self.last_row_indices = torch.full(
      (self._num_envs,), -1, dtype=torch.long, device=self._device
    )
    self.last_trajectory_ids = torch.full(
      (self._num_envs,), -1, dtype=torch.long, device=self._device
    )
    self.last_frame_indices = torch.full(
      (self._num_envs,), -1, dtype=torch.long, device=self._device
    )

  def _ensure_pool(self) -> None:
    """Load and cache the pool on first use (lazy production import)."""
    if self._cached is not None:
      return
    if self.pool is None:
      if self._pool_directory is None:
        raise ValueError(
          "TennisRecoveryResetEvent requires either a 'pool' (injected for "
          "tests) or a 'pool_directory' (for production). Neither was provided."
        )
      # Lazy import at this explicit runtime boundary.
      from mjlab.tasks.velocity.mdp.tennis_endpoint_pool import (
        EndpointPool,
      )

      self.pool = EndpointPool.from_directory(
        self._pool_directory,
        last_n_frames=self._last_n_frames,
        split=self._split,
        validation_fraction=self._validation_fraction,
        seed=self._pool_seed,
      )

    pool = self.pool
    assert pool is not None
    n = len(pool.joint_pos)
    if n == 0:
      raise ValueError("Endpoint pool is empty; cannot sample recovery states.")

    # Verify joint name ordering against the entity (one-time safety check).
    entity_joint_names = self._robot.joint_names
    if pool.joint_names != entity_joint_names:
      raise ValueError(
        "Pool joint_names do not match entity joint_names.\n"
        f"  Pool:   {pool.joint_names}\n"
        f"  Entity: {entity_joint_names}\n"
        "This indicates an ordering mismatch that would corrupt the reset state."
      )

    d = self._device
    # torch.as_tensor on a read-only NumPy array produces a warning about
    # negative strides. Copy via torch.from_numpy().clone() to own the memory
    # and avoid the alias warning.
    self._cached = {
      "joint_pos": torch.from_numpy(
        np.array(pool.joint_pos, copy=True, dtype=np.float32)
      ).to(d),
      "joint_vel": torch.from_numpy(
        np.array(pool.joint_vel, copy=True, dtype=np.float32)
      ).to(d),
      "root_pos_w": torch.from_numpy(
        np.array(pool.root_pos_w, copy=True, dtype=np.float32)
      ).to(d),
      "root_quat_w": torch.from_numpy(
        np.array(pool.root_quat_w, copy=True, dtype=np.float32)
      ).to(d),
      "root_lin_vel_w": torch.from_numpy(
        np.array(pool.root_lin_vel_w, copy=True, dtype=np.float32)
      ).to(d),
      "root_ang_vel_w": torch.from_numpy(
        np.array(pool.root_ang_vel_w, copy=True, dtype=np.float32)
      ).to(d),
    }
    self._pool_trajectory_ids = torch.from_numpy(
      np.array(pool.trajectory_ids, copy=True)
    ).to(d)
    self._pool_frame_indices = torch.from_numpy(
      np.array(pool.frame_indices, copy=True)
    ).to(d)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    **kwargs: Any,
  ) -> None:
    """Restore reference state for recovery envs being reset.

    Called by the event manager in ``mode="reset"`` after the original reset
    events (``reset_base``, ``reset_robot_joints``) have fired. Only recovery
    envs within ``env_ids`` are overwritten; retention envs and non-reset envs
    are untouched.
    """
    del env, kwargs  # All state captured in __init__.

    # Return early before pool loading when no recovery envs are being reset.
    # This avoids importing/loading the dataset for forced-retention configs
    # or partial resets that touch only retention envs.
    recovery_ids = env_ids[self.recovery_env_mask[env_ids]]
    if len(recovery_ids) == 0:
      return

    self._ensure_pool()

    n = len(recovery_ids)
    c = self._cached
    assert c is not None

    # Uniform row sampling = uniform trajectory then frame (equal window length).
    # When eval_row_indices is set (evaluation only), use the pre-assigned
    # row indices instead of random sampling. This lets the evaluation use
    # identical held-out rows for source and candidate without relying on
    # global RNG parity after divergent policy behavior.
    if self.eval_row_indices is not None:
      indices = self.eval_row_indices
      if not isinstance(indices, torch.Tensor) or indices.shape != (self._num_envs,):
        raise ValueError("eval_row_indices must be a tensor with shape [num_envs]")
      if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("eval_row_indices must have integer dtype")
      if indices.device != c["joint_pos"].device:
        raise ValueError("eval_row_indices must be on the environment device")
      row_indices = indices[recovery_ids]
      # Validate: no negative indices (would silently select last row),
      # no out-of-bounds indices.
      if (row_indices < 0).any():
        raise ValueError(
          f"eval_row_indices contains negative values; min={row_indices.min().item()}"
        )
      pool_size = len(c["joint_pos"])
      if (row_indices >= pool_size).any():
        raise ValueError(
          f"eval_row_indices contains out-of-bounds values; "
          f"max={row_indices.max().item()}, pool_size={pool_size}"
        )
    else:
      row_indices = torch.randint(0, len(c["joint_pos"]), (n,), device=self._device)

    # Record provenance for downstream coverage/evaluation metrics.
    self.last_row_indices[recovery_ids] = row_indices
    self.last_trajectory_ids[recovery_ids] = self._pool_trajectory_ids[row_indices]
    self.last_frame_indices[recovery_ids] = self._pool_frame_indices[row_indices]

    joint_pos = c["joint_pos"][row_indices].clone()
    joint_vel = c["joint_vel"][row_indices].clone()
    root_pos = c["root_pos_w"][row_indices].clone()
    root_quat = c["root_quat_w"][row_indices].clone()
    root_lin_vel = c["root_lin_vel_w"][row_indices].clone()
    root_ang_vel = c["root_ang_vel_w"][row_indices].clone()

    # Recenter XY to environment origin; preserve reference height (Z).
    env_origins = self._env.scene.env_origins
    root_pos[:, 0] = env_origins[recovery_ids, 0]
    root_pos[:, 1] = env_origins[recovery_ids, 1]

    # Optional yaw augmentation: rotate orientation and world velocities
    # consistently, preserving authored roll/pitch.
    if self._yaw_aug_range is not None:
      yaws = sample_uniform(
        self._yaw_aug_range[0], self._yaw_aug_range[1], (n,), self._device
      )
      zeros = torch.zeros_like(yaws)
      q_yaw = quat_from_euler_xyz(zeros, zeros, yaws)
      root_quat = quat_mul(q_yaw, root_quat)
      root_lin_vel = quat_apply(q_yaw, root_lin_vel)
      root_ang_vel = quat_apply(q_yaw, root_ang_vel)

    # Write root pose first (sets qpos including quaternion), then root
    # velocity (reads quaternion from qpos to convert world omega → local).
    root_pose = torch.cat([root_pos, root_quat], dim=-1)  # [N, 7]
    self._robot.write_root_link_pose_to_sim(root_pose, env_ids=recovery_ids)

    root_velocity = torch.cat([root_lin_vel, root_ang_vel], dim=-1)  # [N, 6]
    self._robot.write_root_link_velocity_to_sim(root_velocity, env_ids=recovery_ids)

    # Write joint state (no clipping: preserve raw reference values).
    self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=recovery_ids)

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """No per-episode state to reset (group mask is fixed for env lifetime)."""
    del env_ids


# ---------------------------------------------------------------------------
# Recovery velocity command.
# ---------------------------------------------------------------------------


class TennisRecoveryVelocityCommand(UniformVelocityCommand):
  """Velocity command that keeps recovery envs at exactly zero twist.

  Recovery environments always receive zero planar velocity and yaw rate,
  regardless of resampling, heading updates, or curriculum changes. The zero
  command never writes physical qvel: the ``init_velocity_prob`` path is
  skipped for recovery envs so the restored reference velocity survives.

  Retention environments use the original :class:`UniformVelocityCommand`
  distribution and curriculum semantics unchanged.
  """

  def __init__(self, cfg: TennisRecoveryVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    # Look up the recovery reset event to share the group mask.
    event_name = cfg.recovery_reset_event_name
    event_cfg = env.event_manager.get_term_cfg(event_name)
    self._recovery_reset: TennisRecoveryResetEvent = event_cfg.func
    self._recovery_env_mask = self._recovery_reset.recovery_env_mask

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    recovery_ids = env_ids[self._recovery_env_mask[env_ids]]
    if len(recovery_ids) > 0:
      self.vel_command_b[recovery_ids, :] = 0.0
      self.vel_command_w[recovery_ids, :] = 0.0
      self.heading_target[recovery_ids] = 0.0
      self.is_heading_env[recovery_ids] = False
      self.is_standing_env[recovery_ids] = True
      self.is_world_env[recovery_ids] = False
      self.is_forward_env[recovery_ids] = False

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    super()._update_command(env_ids)
    # Force recovery envs to zero after any heading/world/standing updates.
    recovery_ids = self._recovery_env_mask.nonzero(as_tuple=False).flatten()
    if len(recovery_ids) > 0:
      self.vel_command_b[recovery_ids, :] = 0.0
      self.vel_command_w[recovery_ids, :] = 0.0

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    assert isinstance(env_ids, torch.Tensor)
    # Call CommandTerm.reset directly to skip UniformVelocityCommand's
    # init_velocity path, which would overwrite recovery envs' restored
    # reference velocity with their (zero) command velocity.
    extras = CommandTerm.reset(self, env_ids)

    # Apply init_velocity_prob only to retention envs.
    if self.cfg.init_velocity_prob > 0.0:
      r = torch.empty(len(env_ids), device=self.device)
      init_ids = env_ids[r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob]
      # Exclude recovery envs: their zero command must not write qvel.
      init_ids = init_ids[~self._recovery_env_mask[init_ids]]
      if len(init_ids) > 0:
        vel_b = torch.zeros(len(init_ids), 6, device=self.device)
        vel_b[:, :2] = self.vel_command_b[init_ids, :2]
        vel_b[:, 5] = self.vel_command_b[init_ids, 2]
        self.robot.write_root_link_velocity_b_to_sim(vel_b, env_ids=init_ids)
    return extras


@dataclass(kw_only=True)
class TennisRecoveryVelocityCommandCfg(UniformVelocityCommandCfg):
  """Configuration for the tennis-recovery velocity command.

  Extends :class:`UniformVelocityCommandCfg` with a reference to the recovery
  reset event name, used to share the fixed group mask.
  """

  recovery_reset_event_name: str = "tennis_recovery_reset"
  """Name of the ``TennisRecoveryResetEvent`` term in the event config, used to
  look up the shared recovery/retention group mask at build time."""

  def build(self, env: ManagerBasedRlEnv) -> TennisRecoveryVelocityCommand:
    return TennisRecoveryVelocityCommand(self, env)
