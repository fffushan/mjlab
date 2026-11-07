"""Probe the carry-pose velocity task at step 0 and over 100 zero-action steps.

Two separate measurements, reported separately:

* **Step 0, before any physics step.** This is the acceptance measurement: the
  joint positions equal ``BAT_CARRY_POSE``, the entity default root height equals
  ``BAT_CARRY_BASE_HEIGHT``, the self-collision force is zero and the
  ``pose_quadratic`` term is zero on the arm and waist group. It proves the
  default-pose and reward plumbing.
* **100 zero-action steps.** This is a **diagnostic**, not a pass/fail
  criterion. A zero action with ``use_default_offset=True`` is only a PD hold at
  the default pose; it is not a balancing policy. The same recipe is run on the
  existing no-state-estimation velocity task for comparison. Root height, torso
  tilt, per-group pose error, falls and self-collision force are reported.

Probe, not a test: 8 environments, 100 steps, no training, no checkpoint.

Usage::

    .venv/bin/python scripts/experiments/x2_carry_velocity/probe_carry_stand.py

See docs/design/x2_carry_velocity_plan.md section 5.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from mjlab.asset_zoo.robots.agibot_x2.tennis_constants import (
  BAT_CARRY_BASE_HEIGHT,
  BAT_CARRY_POSE,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity import mdp

TASK_ID = "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-CarryVelocity"
BASE_TASK_ID = "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation"

PROBE_ENVS = 8
PROBE_STEPS = 100
PROBE_SEED = 11
DEFAULT_POS_TOLERANCE = 1e-6
COLLISION_THRESHOLD_N = 10.0
"""Same threshold as the ``self_collisions`` reward term."""


def _device() -> str:
  return "cuda:0" if torch.cuda.is_available() else "cpu"


def _build(task_id: str) -> tuple[Any, ManagerBasedRlEnv]:
  cfg = load_env_cfg(task_id)
  cfg.scene.num_envs = PROBE_ENVS
  cfg.seed = PROBE_SEED
  cfg.auto_reset = False
  env = ManagerBasedRlEnv(cfg=cfg, device=_device(), render_mode=None)
  env.reset()
  return cfg, env


def _tilt_deg(quat_wxyz: torch.Tensor) -> torch.Tensor:
  """Torso tilt from upright in degrees, same convention as ``bad_orientation``."""
  x, y = quat_wxyz[:, 1], quat_wxyz[:, 2]
  cos_tilt = torch.clamp(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)
  return torch.rad2deg(torch.acos(cos_tilt))


def _group_joint_ids(robot) -> tuple[torch.Tensor, torch.Tensor]:
  names = robot.joint_names
  leg_head = [
    index
    for index, name in enumerate(names)
    if any(tag in name for tag in ("hip", "knee", "ankle", "head"))
  ]
  arm_waist = [index for index in range(len(names)) if index not in leg_head]
  device = robot.data.joint_pos.device
  return (
    torch.tensor(leg_head, dtype=torch.long, device=device),
    torch.tensor(arm_waist, dtype=torch.long, device=device),
  )


def _pose_error_deg(robot, joint_ids: torch.Tensor) -> torch.Tensor:
  error = (
    robot.data.joint_pos[:, joint_ids] - robot.data.default_joint_pos[:, joint_ids]
  )
  return torch.rad2deg(torch.abs(error))


def _collision_instant_n(env: ManagerBasedRlEnv) -> tuple[float, int]:
  """Instantaneous self-collision force at the current state.

  Reads ``data.force``, not ``data.force_history``: the history buffer is
  zeroed by ``ContactSensor.reset`` and only refilled by ``ContactSensor.update``
  during ``env.step``. At step 0 the history is all zeros, so it cannot measure
  the spawn state.
  """
  data = env.scene["self_collision"].data
  assert data.force is not None
  magnitude = torch.norm(data.force, dim=-1)
  return float(magnitude.max()), int((magnitude > COLLISION_THRESHOLD_N).sum())


def _collision_max_n(env: ManagerBasedRlEnv) -> tuple[float, int]:
  """Maximum self-collision force over the policy step's substep history."""
  data = env.scene["self_collision"].data
  assert data.force_history is not None
  magnitude = torch.norm(data.force_history, dim=-1)
  return float(magnitude.max()), int((magnitude > COLLISION_THRESHOLD_N).sum())


def _report_step_zero(env: ManagerBasedRlEnv) -> list[str]:
  """Acceptance checks at step 0, before any physics step. Returns failures."""
  failures: list[str] = []
  robot = env.scene["robot"]
  names = robot.joint_names
  default = robot.data.default_joint_pos
  missing = sorted(set(names) ^ set(BAT_CARRY_POSE))
  default_error = max(
    abs(float(default[0, names.index(name)]) - value)
    for name, value in BAT_CARRY_POSE.items()
  )

  # The zero-action target comes from the action term, not from the entity's
  # default_joint_pos. With use_default_offset=True they should agree.
  action_term = env.action_manager.get_term("joint_pos")
  offset = action_term.offset
  assert isinstance(offset, torch.Tensor)
  offset_by_name = {
    name: float(offset[0, index]) for index, name in enumerate(action_term.target_names)
  }
  offset_missing = sorted(set(offset_by_name) ^ set(BAT_CARRY_POSE))
  offset_error = max(
    abs(offset_by_name[name] - value) for name, value in BAT_CARRY_POSE.items()
  )

  # The inherited reset_joints_by_offset event clamps the default pose to the
  # soft joint limits. Record which joints the clamp moves.
  device = robot.data.joint_pos.device
  carry = torch.tensor(
    [BAT_CARRY_POSE[name] for name in names], dtype=torch.float32, device=device
  )
  soft = robot.data.soft_joint_pos_limits[0]
  expected_reset = torch.clamp(carry, soft[:, 0], soft[:, 1])
  reset_error = float((robot.data.joint_pos - expected_reset).abs().max())
  clamped = [
    index
    for index in range(len(names))
    if abs(float(expected_reset[index]) - float(carry[index])) > 1e-9
  ]

  root_default = float(robot.data.default_root_state[0, 2])
  measured_height = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  height_offset = measured_height - BAT_CARRY_BASE_HEIGHT
  collision_max, collision_hits = _collision_instant_n(env)

  post_reset_joint_pos = robot.data.joint_pos.clone()
  quad_cfg = env.reward_manager.get_term_cfg("pose_quadratic")
  post_reset_quad = mdp.pose_quadratic(
    env,
    asset_cfg=quad_cfg.params["asset_cfg"],
    std=float(quad_cfg.params["std"]),
  )

  dof_cfg = env.reward_manager.get_term_cfg("dof_pos_limits")
  dof_post_reset = dof_cfg.func(env, **dof_cfg.params)

  # Acceptance pose: write BAT_CARRY_POSE itself, not the clamped reset state.
  robot.write_joint_position_to_sim(default)
  env.scene.write_data_to_sim()
  env.sim.forward()
  default_quad = mdp.pose_quadratic(
    env,
    asset_cfg=quad_cfg.params["asset_cfg"],
    std=float(quad_cfg.params["std"]),
  )
  dof_default = dof_cfg.func(env, **dof_cfg.params)
  joint_pos_default = robot.data.joint_pos[0]
  violation = (soft[:, 0] - joint_pos_default).clamp(min=0.0) + (
    joint_pos_default - soft[:, 1]
  ).clamp(min=0.0)
  violation_joints = [
    (names[index], float(violation[index]))
    for index in range(len(names))
    if float(violation[index]) > 1e-12
  ]

  # Restore the post-reset state so the drift rollout starts where reset put it.
  robot.write_joint_position_to_sim(post_reset_joint_pos)
  env.scene.write_data_to_sim()
  env.sim.forward()

  def check(label: str, ok: bool, detail: str) -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label} -- {detail}")
    if not ok:
      failures.append(label)

  print("=== step 0 (before any physics step) ===")
  check(
    "default_joint_pos equals BAT_CARRY_POSE",
    not missing and default_error <= DEFAULT_POS_TOLERANCE,
    f"{len(names)} joints, max |error|={default_error:.3e} rad, missing={missing}",
  )
  check(
    "zero-action target equals BAT_CARRY_POSE",
    not offset_missing and offset_error <= DEFAULT_POS_TOLERANCE,
    f"action term offset, max |error|={offset_error:.3e} rad, missing={offset_missing}",
  )
  check(
    "post-reset joint positions equal BAT_CARRY_POSE clamped to soft limits",
    reset_error <= DEFAULT_POS_TOLERANCE,
    f"max |error|={reset_error:.3e} rad, clamped joints="
    f"{[names[index] for index in clamped]}",
  )
  for index in clamped:
    name = names[index]
    before = float(carry[index])
    after = float(expected_reset[index])
    lower, upper = float(soft[index, 0]), float(soft[index, 1])
    print(
      f"  INFO  inherited reset clamp: {name} {before:.6f} -> {after:.6f} rad "
      f"({math.degrees(after - before):+.3f} deg), "
      f"soft limits [{lower:.6f}, {upper:.6f}]"
    )
  check(
    "entity default root height equals BAT_CARRY_BASE_HEIGHT",
    abs(root_default - BAT_CARRY_BASE_HEIGHT) <= DEFAULT_POS_TOLERANCE,
    f"default_root_state.z={root_default:.9f} m, expected "
    f"{BAT_CARRY_BASE_HEIGHT:.9f} m",
  )
  print(
    "  INFO  measured reset root height = BAT_CARRY_BASE_HEIGHT + z offset "
    f"-- measured [{float(measured_height.min()):.6f}, "
    f"{float(measured_height.max()):.6f}] m, offset "
    f"[{float(height_offset.min()):.6f}, {float(height_offset.max()):.6f}] m "
    "(config offset is (0.0, 0.02))"
  )
  check(
    "self-collision force at step 0 is zero",
    collision_max == 0.0 and collision_hits == 0,
    f"max force={collision_max:.3e} N, contacts>{COLLISION_THRESHOLD_N:g} N="
    f"{collision_hits}",
  )
  check(
    "pose_quadratic on the arm and waist group is zero at exact BAT_CARRY_POSE",
    float(default_quad.max()) <= 1e-12,
    f"max={float(default_quad.max()):.3e} (raw reward value)",
  )
  weight = float(quad_cfg.weight)
  print(
    "  INFO  inherited behaviour: pose_quadratic at the post-reset clamped "
    f"state = raw {float(post_reset_quad.max()):.6f}, weighted "
    f"{weight * float(post_reset_quad.max()):.6f} per env-step"
  )
  print(
    "  INFO  inherited behaviour: dof_pos_limits at the post-reset clamped "
    f"state = raw {float(dof_post_reset.max()):.3e}, weighted "
    f"{-float(dof_post_reset.max()):.3e} per env-step"
  )
  print(
    "  INFO  inherited behaviour: dof_pos_limits at exact BAT_CARRY_POSE = "
    f"raw {float(dof_default.max()):.6f}, weighted "
    f"{-float(dof_default.max()):.6f} per env-step, from "
    f"{[(name, round(value, 6)) for name, value in violation_joints]}"
  )
  return failures


def _rollout(env: ManagerBasedRlEnv, label: str) -> None:
  """100 zero-action steps. Diagnostic only; no pass/fail verdict."""
  robot = env.scene["robot"]
  torso_id = robot.body_names.index("torso_link")
  leg_head_ids, arm_waist_ids = _group_joint_ids(robot)
  action = torch.zeros(
    (env.num_envs, env.action_manager.total_action_dim), device=env.device
  )

  falls = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  first_fall = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  truncations = 0
  live_env_steps = 0
  height_sum = 0.0
  height_min = float("inf")
  height_max = float("-inf")
  tilt_sum = 0.0
  tilt_max = 0.0
  leg_sum = 0.0
  leg_max = 0.0
  arm_sum = 0.0
  arm_max = 0.0
  collision_max = 0.0
  collision_substeps = 0

  for step in range(1, PROBE_STEPS + 1):
    _, _, terminated, truncated, _ = env.step(action)

    # Accumulate only before an environment's first fall, including the terminal
    # step. Reset environments restart from the carry pose and are excluded.
    standing = first_fall == 0
    if bool(standing.any()):
      height = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
      tilt = _tilt_deg(robot.data.body_link_quat_w[:, torso_id])
      leg_error = _pose_error_deg(robot, leg_head_ids)
      arm_error = _pose_error_deg(robot, arm_waist_ids)
      count = int(standing.sum())
      live_env_steps += count
      height_sum += float(height[standing].sum())
      height_min = min(height_min, float(height[standing].min()))
      height_max = max(height_max, float(height[standing].max()))
      tilt_sum += float(tilt[standing].sum())
      tilt_max = max(tilt_max, float(tilt[standing].max()))
      leg_sum += float(leg_error[standing].mean(dim=1).sum())
      leg_max = max(leg_max, float(leg_error[standing].max()))
      arm_sum += float(arm_error[standing].mean(dim=1).sum())
      arm_max = max(arm_max, float(arm_error[standing].max()))

    current_max, current_substeps = _collision_max_n(env)
    collision_max = max(collision_max, current_max)
    collision_substeps += current_substeps

    newly_fallen = terminated & (first_fall == 0)
    first_fall = torch.where(
      newly_fallen,
      torch.full_like(first_fall, step),
      first_fall,
    )
    falls += terminated.long()
    truncations += int(truncated.sum())
    done = terminated | truncated
    if bool(done.any()):
      env.reset(env_ids=done.nonzero(as_tuple=False).squeeze(-1))

  print(f"=== {PROBE_STEPS}-step zero-action drift: {label} (diagnostic) ===")
  print(
    f"  steps={PROBE_STEPS}/{PROBE_STEPS} truncations={truncations} "
    f"falls={int(falls.sum())} envs_that_fell={int((falls > 0).sum())}/"
    f"{env.num_envs}"
  )
  print(
    f"  first-fall steps: "
    f"{[int(value) if value > 0 else None for value in first_fall.tolist()]}"
  )
  if live_env_steps > 0:
    print(
      f"  root height live env-steps: mean={height_sum / live_env_steps:.6f} m "
      f"min={height_min:.6f} m max={height_max:.6f} m"
    )
    print(
      f"  torso tilt live env-steps: mean={tilt_sum / live_env_steps:.3f} deg "
      f"max={tilt_max:.3f} deg"
    )
    print(
      f"  pose error leg+head live env-steps: "
      f"mean={leg_sum / live_env_steps:.3f} deg max={leg_max:.3f} deg"
    )
    print(
      f"  pose error arm+waist live env-steps: "
      f"mean={arm_sum / live_env_steps:.3f} deg max={arm_max:.3f} deg"
    )
  print(
    f"  self-collision max force={collision_max:.3f} N, "
    f"substeps>{COLLISION_THRESHOLD_N:g} N={collision_substeps}"
  )


def main() -> int:
  _, carry_env = _build(TASK_ID)
  try:
    failures = _report_step_zero(carry_env)
    _rollout(carry_env, "carry-pose task")
  finally:
    carry_env.close()

  _, base_env = _build(BASE_TASK_ID)
  try:
    _rollout(base_env, "base no-state-estimation task (comparison)")
  finally:
    base_env.close()

  print()
  if failures:
    print(f"STEP 0 RESULT: FAIL ({len(failures)} checks)")
    for failure in failures:
      print(f"  - {failure}")
    return 1
  print("STEP 0 RESULT: PASS")
  print("DRIFT RESULT: diagnostic only, not a pass/fail criterion")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
