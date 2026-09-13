"""Replay a retargeted qpos motion onto an mjlab robot and export the
tracking-format motion npz consumed by ``train.py``.

Retargeting pipelines (e.g. UMR / SMPL-X retargeting) emit an npz with a raw
qpos trajectory:

  qpos (T, 7 + N)          floating base pos + quat (wxyz) + joint positions
  fps (1,)
  robot_joint_names (N,)

``train.py``'s ``MotionLoader`` instead expects the processed tracking format:

  joint_pos (T, N), joint_vel (T, N)
  body_pos_w (T, nb, 3), body_quat_w (T, nb, 4)
  body_lin_vel_w (T, nb, 3), body_ang_vel_w (T, nb, 3)
  fps (1,)

This script resamples the trajectory to the environment rate (50 Hz by
default, matching ``timestep=0.005`` and ``decimation=4``), maps the joints by
name onto the mjlab mode-15 model, replays it kinematically through the robot,
and records every link's world frame (the tracking task picks the bodies it
tracks by name from this file).

Example:

  uv run python scripts/retarget_npz_to_tracking_npz.py \
    --input data/qianghuo_smplx_unitree_g1_29dof_mode_15.npz \
    --output data/qianghuo_smplx_unitree_g1_29dof_mode_15_tracking.npz

  uv run python scripts/retarget_npz_to_tracking_npz.py \
    --input data/qianghuo_smplx_agibot_x2.npz \
    --output data/qianghuo_smplx_agibot_x2_tracking.npz \
    --task Mjlab-Tracking-Flat-AgiBot-X2
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
  quat_slerp,
)

RECORD_KEYS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)


def _lerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
  return a * (1.0 - t) + b * t


def resample_qpos(
  qpos: torch.Tensor, input_fps: float, output_fps: float
) -> torch.Tensor:
  """Resample qpos (T, 7 + N) from ``input_fps`` to ``output_fps``.

  Positions and joints are linearly interpolated; the base quaternion is
  slerped. The base velocity convention (wxyz) is preserved.
  """
  frames = qpos.shape[0]
  duration = (frames - 1) / input_fps
  times = torch.arange(0.0, duration, 1.0 / output_fps, device=qpos.device)
  phase = times / duration
  index_0 = (phase * (frames - 1)).floor().long()
  index_1 = torch.minimum(index_0 + 1, torch.tensor(frames - 1))
  blend = (phase * (frames - 1) - index_0).to(qpos.dtype)

  base_pos = _lerp(qpos[index_0, :3], qpos[index_1, :3], blend[:, None])
  base_quat = torch.stack(
    [
      quat_slerp(qpos[index_0[i], 3:7], qpos[index_1[i], 3:7], float(blend[i]))
      for i in range(times.shape[0])
    ]
  )
  joints = _lerp(qpos[index_0, 7:], qpos[index_1, 7:], blend[:, None])
  return torch.cat([base_pos, base_quat, joints], dim=1)


def compute_velocities(
  qpos: torch.Tensor, fps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Finite-difference velocities for base (lin/ang) and joints from qpos."""
  dt = 1.0 / fps
  base_pos = qpos[:, :3]
  base_quat = qpos[:, 3:7]
  joints = qpos[:, 7:]

  base_lin_vel = torch.gradient(base_pos, spacing=dt, dim=0)[0]
  joint_vel = torch.gradient(joints, spacing=dt, dim=0)[0]

  q_prev, q_next = base_quat[:-2], base_quat[2:]
  omega = axis_angle_from_quat(quat_mul(q_next, quat_conjugate(q_prev))) / (2.0 * dt)
  base_ang_vel = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
  return base_lin_vel, base_ang_vel, joint_vel


def replay_and_save(
  qpos: torch.Tensor,
  fps: float,
  joint_names: list[str],
  output: Path,
  device: str,
  task: str,
) -> None:
  """Replay the qpos trajectory through the task's robot and save."""
  output_fps = float(fps)
  env_cfg = load_env_cfg(task)

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  scene = Scene(env_cfg.scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  robot: Entity = scene["robot"]
  # Map retargeted joints (in file order) onto the model's joint ordering.
  robot_joint_indexes = torch.tensor(
    robot.find_joints(joint_names, preserve_order=True)[0], device=device
  )

  base_lin_vel, base_ang_vel, joint_vel = compute_velocities(qpos, output_fps)

  log: dict[str, list[np.ndarray]] = {k: [] for k in RECORD_KEYS}

  pbar = tqdm(total=qpos.shape[0], desc="Replaying frames", unit="frame")
  for t in range(qpos.shape[0]):
    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = qpos[t : t + 1, :3]
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = qpos[t : t + 1, 3:7]
    root_states[:, 7:10] = base_lin_vel[t : t + 1]
    root_states[:, 10:] = base_ang_vel[t : t + 1]
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel_full = robot.data.default_joint_vel.clone()
    joint_pos[:, robot_joint_indexes] = qpos[t : t + 1, 7:]
    joint_vel_full[:, robot_joint_indexes] = joint_vel[t : t + 1]
    robot.write_joint_state_to_sim(joint_pos, joint_vel_full)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
    log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
    log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
    log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
    log["body_lin_vel_w"].append(
      robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy()
    )
    log["body_ang_vel_w"].append(
      robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy()
    )
    pbar.update(1)
  pbar.close()

  print("\nStacking arrays and saving data...")
  data: dict[str, np.ndarray] = {k: np.stack(log[k], axis=0) for k in RECORD_KEYS}
  data["fps"] = np.array([output_fps], dtype=np.float32)
  arrays_any: dict[str, Any] = dict(data)
  np.savez(output, **arrays_any)
  print(f"Saved tracking motion to {output}")
  print(
    f"  frames: {data['joint_pos'].shape[0]} @ {output_fps:.0f} Hz, "
    f"joints: {data['joint_pos'].shape[1]}, bodies: {data['body_pos_w'].shape[1]}"
  )


def main(
  input: Path,
  output: Path,
  task: str = "Mjlab-Tracking-Flat-Unitree-G1-29DOF-Mode-15",
  output_fps: float = 50.0,
  device: str = "cuda:0",
) -> None:
  """Convert a retargeted qpos npz into the tracking-format motion npz.

  Args:
    input: Path to the retargeted npz (qpos, fps, robot_joint_names).
    output: Path where the tracking-format npz will be written.
    task: Registered mjlab task whose robot the motion is replayed on. The
      motion's joints must exist in that robot; a robot trained with more
      joints than the motion provides keeps its keyframe pose for the rest.
    output_fps: Frame rate of the exported motion. Defaults to 50 Hz to match
      the environment rate (timestep=0.005, decimation=4).
    device: Device to use for the replay simulation.
  """
  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING]: CUDA is not available. Falling back to CPU. This may be slow.")
    device = "cpu"

  data = np.load(input, allow_pickle=True)
  for key in ("qpos", "fps", "robot_joint_names"):
    if key not in data:
      raise ValueError(f"{input} has no '{key}' field; is this a retargeted npz?")

  fps = float(np.asarray(data["fps"]).item())
  qpos = torch.from_numpy(np.asarray(data["qpos"], dtype=np.float32))
  joint_names = [str(n) for n in data["robot_joint_names"]]

  print(f"Input: {input}")
  print(
    f"  retargeted frames: {qpos.shape[0]} @ {fps:.0f} Hz, joints: {len(joint_names)}"
  )

  if fps != output_fps:
    print(f"Resampling from {fps:.0f} Hz to {output_fps:.0f} Hz...")
    qpos = resample_qpos(qpos, fps, output_fps)
  else:
    print(f"Keeping native {fps:.0f} Hz frame rate.")

  qpos = qpos.to(device)

  output.parent.mkdir(parents=True, exist_ok=True)
  replay_and_save(qpos, output_fps, joint_names, output, device, task)


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
