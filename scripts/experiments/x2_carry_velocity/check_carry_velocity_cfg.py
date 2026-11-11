"""Check the carry-pose velocity config against the resolved values.

Probe, not a test: it builds one environment for the existing no-state-estimation
velocity task and one for the carry-pose task, prints the resolved numbers, and
exits non-zero only if a stated config contract fails. It runs one environment
per task, no training step and no checkpoint.

Usage::

    .venv/bin/python scripts/experiments/x2_carry_velocity/check_carry_velocity_cfg.py

See docs/design/x2_carry_velocity_plan.md section 5.
"""

from __future__ import annotations

import math
import re

import torch

from mjlab.asset_zoo.robots.agibot_x2.tennis_constants import (
  BAT_CARRY_POSE,
  BED_GEOM,
  bed_solref,
  CLAMP_BODY,
  FACE_SITE,
  RACKET_BODY,
)
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.config.agibot_x2.vendor_gains import X2_VENDOR_RL_GAINS
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

TASK_ID = "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-CarryVelocity"
JUGGLE_TASK_ID = (
  "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
  "-Ball3-Pelvis-Nominal-S9c-PoseQuad"
)
BASE_TASK_ID = "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation"
EXPERIMENT_NAME = "agibot_x2_velocity_carry_pose"

EXPECTED_ACTION_DIM = 31
"""Resolved joint actions: 31 actuated joints. A pattern-list length is not an
action count; the 16 actuator groups expand to 31 joints."""

FOOT_SITES = ("left_foot", "right_foot")

PROBE_ENVS = 1
PROBE_SEED = 7
DEFAULT_POS_TOLERANCE = 1e-6


def _device() -> str:
  return "cuda:0" if torch.cuda.is_available() else "cpu"


def _build(task_id: str) -> tuple[ManagerBasedRlEnvCfg, ManagerBasedRlEnv]:
  cfg = load_env_cfg(task_id)
  cfg.scene.num_envs = PROBE_ENVS
  cfg.seed = PROBE_SEED
  env = ManagerBasedRlEnv(cfg=cfg, device=_device(), render_mode=None)
  return cfg, env


def _resolved_joint_names(env: ManagerBasedRlEnv, term_name: str) -> list[str]:
  asset_cfg = env.reward_manager.get_term_cfg(term_name).params["asset_cfg"]
  names = env.scene[asset_cfg.name].joint_names
  if isinstance(asset_cfg.joint_ids, slice):
    return list(names)
  return [names[index] for index in asset_cfg.joint_ids]


def _group_joint_ids(robot) -> tuple[list[int], list[int]]:
  """Leg+head and arm+waist indices, classified by name, not by reward config."""
  names = robot.joint_names
  leg_head = [
    index
    for index, name in enumerate(names)
    if any(tag in name for tag in ("hip", "knee", "ankle", "head"))
  ]
  arm_waist = [index for index in range(len(names)) if index not in leg_head]
  return leg_head, arm_waist


class _Checks:
  def __init__(self) -> None:
    self.failures: list[str] = []

  def check(self, label: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    suffix = f" -- {detail}" if detail else ""
    print(f"  {status}  {label}{suffix}")
    if not ok:
      self.failures.append(label)


def _vendor_gains_by_joint(entity_cfg) -> dict[str, tuple[float, float]]:
  """Per-joint (kp, kd) for every joint the vendor table names.

  mjlab sets gains per actuator group and the vendor table is per joint, so each
  joint takes the gain of the group whose target expressions match it.
  """
  gains: dict[str, tuple[float, float]] = {}
  for group in entity_cfg.articulation.actuators:
    for name in X2_VENDOR_RL_GAINS:
      if any(re.fullmatch(p, name) for p in group.target_names_expr):
        gains[name] = (float(group.stiffness), float(group.damping))
  return gains


def _print_base_fingerprint(env: ManagerBasedRlEnv) -> None:
  reward_manager = env.reward_manager
  rewards = ", ".join(
    f"{name}={reward_manager.get_term_cfg(name).weight:g}"
    for name in sorted(reward_manager.active_terms)
  )
  print("[0] existing no-state-estimation velocity task")
  print(
    f"  actor={tuple(env.observation_manager.group_obs_dim['actor'])} "
    f"critic={tuple(env.observation_manager.group_obs_dim['critic'])} "
    f"actions={env.action_manager.total_action_dim}"
  )
  print(f"  rewards: {rewards}")
  print(f"  curriculum: {sorted(env.cfg.curriculum)}")


def _print_carry_fingerprint(env: ManagerBasedRlEnv) -> None:
  reward_manager = env.reward_manager
  rewards = ", ".join(
    f"{name}={reward_manager.get_term_cfg(name).weight:g}"
    for name in sorted(reward_manager.active_terms)
  )
  print("[0] carry-pose velocity task")
  print(
    f"  actor={tuple(env.observation_manager.group_obs_dim['actor'])} "
    f"critic={tuple(env.observation_manager.group_obs_dim['critic'])} "
    f"actions={env.action_manager.total_action_dim}"
  )
  print(f"  rewards: {rewards}")
  print(f"  curriculum: {sorted(env.cfg.curriculum)}")


def main() -> int:
  checks = _Checks()

  base_cfg, base_env = _build(BASE_TASK_ID)
  try:
    base_actor_dim = tuple(base_env.observation_manager.group_obs_dim["actor"])
    base_action_dim = int(base_env.action_manager.total_action_dim)
    base_pose_names = set(_resolved_joint_names(base_env, "pose"))
    base_pose_params = base_env.reward_manager.get_term_cfg("pose").params
    base_std_tables = {
      group: base_pose_params[group]
      for group in ("std_standing", "std_walking", "std_running")
    }
    _print_base_fingerprint(base_env)
  finally:
    base_env.close()

  cfg, env = _build(TASK_ID)
  try:
    _print_carry_fingerprint(env)
    robot = env.scene["robot"]

    print("[1] entity and compiled spec")
    names = robot.joint_names
    default = robot.data.default_joint_pos[0]
    missing = sorted(set(names) ^ set(BAT_CARRY_POSE))
    max_error = max(
      abs(float(default[names.index(name)]) - value)
      for name, value in BAT_CARRY_POSE.items()
    )
    checks.check(
      "default_joint_pos equals BAT_CARRY_POSE for every joint",
      not missing and max_error <= DEFAULT_POS_TOLERANCE,
      f"{len(names)} joints, max |error|={max_error:.3e} rad, missing={missing}",
    )

    mj_model = env.sim.mj_model
    scene_bodies = {mj_model.body(index).name for index in range(mj_model.nbody)}
    scene_sites = {mj_model.site(index).name for index in range(mj_model.nsite)}
    entity_name = "robot"
    racket_names = (CLAMP_BODY, RACKET_BODY)
    checks.check(
      "racket bodies exist in the compiled spec",
      set(racket_names) <= set(robot.body_names)
      and {f"{entity_name}/{name}" for name in racket_names} <= scene_bodies,
      ", ".join(racket_names),
    )
    checks.check(
      "face site exists in the compiled spec",
      FACE_SITE in robot.site_names and f"{entity_name}/{FACE_SITE}" in scene_sites,
      FACE_SITE,
    )
    checks.check(
      "both velocity foot sites exist in the compiled spec",
      set(FOOT_SITES) <= set(robot.site_names)
      and {f"{entity_name}/{name}" for name in FOOT_SITES} <= scene_sites,
      ", ".join(FOOT_SITES),
    )

    print("[1b] string-bed contact")
    # The bed is geometry here, not a contact body: no ball, and its calibrated solref
    # was fitted at a 1 ms step while this task runs 5 ms. Two arms died with a NaN in the
    # actor observation with no metric warning, so the bits are asserted rather than
    # assumed. The geometry, the face site and the contact parameters stay in the model:
    # only contype/conaffinity/priority move, which is what removes the contact pair.
    bed = [g for g in range(mj_model.ngeom) if BED_GEOM in mj_model.geom(g).name]
    checks.check(
      "the bed geom is still in the compiled spec",
      len(bed) == 1,
      f"matches={[mj_model.geom(g).name for g in bed]}",
    )
    if len(bed) == 1:
      bed_id = bed[0]
      checks.check(
        "the bed generates no contact in the carry task",
        int(mj_model.geom_contype[bed_id]) == 0
        and int(mj_model.geom_conaffinity[bed_id]) == 0,
        f"contype={int(mj_model.geom_contype[bed_id])} "
        f"conaffinity={int(mj_model.geom_conaffinity[bed_id])}",
      )
      checks.check(
        "only the contact bits moved: the bed keeps its calibrated solref",
        tuple(float(v) for v in mj_model.geom_solref[bed_id])
        == tuple(float(v) for v in bed_solref()),
        f"solref={tuple(float(v) for v in mj_model.geom_solref[bed_id])}",
      )
    # Consequence, stated rather than hidden: the bed is the racket's ONLY collider --
    # every other racket geom is visual-only -- so switching the bed off leaves the racket
    # with no contact at all. That is what this task wants (no ball, and the racket is
    # held clear of the ground), but it is a property of the asset, not of this config,
    # so it is measured here and again on the juggle model in section [7].
    racket_geoms = [
      g
      for g in range(mj_model.ngeom)
      if any(
        key in mj_model.geom(g).name for key in (RACKET_BODY, CLAMP_BODY, BED_GEOM)
      )
    ]
    colliding_racket = [
      mj_model.geom(g).name
      for g in racket_geoms
      if int(mj_model.geom_contype[g]) != 0 or int(mj_model.geom_conaffinity[g]) != 0
    ]
    checks.check(
      "the racket has no collider left in the carry task",
      bool(racket_geoms) and not colliding_racket,
      f"{len(racket_geoms)} racket geoms, still colliding={colliding_racket}",
    )

    print("[2] observation and action contract")
    actor_dim = tuple(env.observation_manager.group_obs_dim["actor"])
    action_dim = int(env.action_manager.total_action_dim)
    checks.check(
      "actor observation width equals the existing velocity task",
      actor_dim == base_actor_dim,
      f"carry={actor_dim} base={base_actor_dim}",
    )
    checks.check(
      "action count is 31",
      action_dim == EXPECTED_ACTION_DIM,
      f"carry={action_dim} base={base_action_dim} joints={len(names)}",
    )
    checks.check(
      "action count equals the existing velocity task",
      action_dim == base_action_dim,
      f"carry={action_dim} base={base_action_dim}",
    )

    print("[3] reward joint groups")
    pose_names = _resolved_joint_names(env, "pose")
    quad_names = _resolved_joint_names(env, "pose_quadratic")
    _, arm_waist_ids = _group_joint_ids(robot)
    expected_arm_waist = {names[index] for index in arm_waist_ids}
    checks.check(
      "pose scores all 31 joints",
      set(pose_names) == set(names),
      f"{len(pose_names)} joints",
    )
    checks.check(
      "pose covers the same joints as the base velocity task",
      set(pose_names) == base_pose_names,
      f"carry={len(pose_names)} base={len(base_pose_names)}",
    )
    checks.check(
      "pose_quadratic covers arms and waist",
      set(quad_names) == expected_arm_waist,
      f"{len(quad_names)} joints: {', '.join(quad_names)}",
    )
    checks.check(
      "pose_quadratic is a subset of pose, so the arm pull is added on top",
      set(quad_names) <= set(pose_names),
      f"quadratic={len(quad_names)} pose={len(pose_names)}",
    )
    pose_params = env.reward_manager.get_term_cfg("pose").params
    same_tables = all(
      pose_params[group] == base_std_tables[group]
      for group in ("std_standing", "std_walking", "std_running")
    )
    checks.check(
      "pose keeps the base task's three tolerance tables unchanged",
      same_tables,
      f"sizes={ {g: len(pose_params[g]) for g in ('std_standing', 'std_walking', 'std_running')} }",
    )

    print("[4] command range and reset")
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    expected_ranges = {
      "lin_vel_x": (-0.6, 0.6),
      "lin_vel_y": (-0.3, 0.3),
      "ang_vel_z": (-0.5, 0.5),
      "heading": (-math.pi, math.pi),
    }
    observed_ranges = {
      "lin_vel_x": tuple(twist.ranges.lin_vel_x),
      "lin_vel_y": tuple(twist.ranges.lin_vel_y),
      "ang_vel_z": tuple(twist.ranges.ang_vel_z),
      "heading": tuple(twist.ranges.heading)
      if twist.ranges.heading is not None
      else None,
    }
    checks.check(
      "twist command ranges",
      observed_ranges == expected_ranges,
      f"{observed_ranges}",
    )
    checks.check(
      "rel_standing_envs is 0.25",
      twist.rel_standing_envs == 0.25,
      f"{twist.rel_standing_envs}",
    )
    checks.check(
      "curriculum is empty",
      cfg.curriculum == {},
      f"{sorted(cfg.curriculum)}",
    )
    reset_z = tuple(cfg.events["reset_base"].params["pose_range"]["z"])
    checks.check(
      "reset_base height offset is (0.0, 0.02)",
      reset_z == (0.0, 0.02),
      f"{reset_z}",
    )
    checks.check(
      "experiment name",
      load_rl_cfg(TASK_ID).experiment_name == EXPERIMENT_NAME,
      load_rl_cfg(TASK_ID).experiment_name,
    )

    print("[5] pose_quadratic cost at uniform arm+waist offset")
    quad_cfg = env.reward_manager.get_term_cfg("pose_quadratic")
    quad_asset = quad_cfg.params["asset_cfg"]
    std = float(quad_cfg.params["std"])
    weight = float(quad_cfg.weight)
    base_joint_pos = robot.data.default_joint_pos.clone()
    costs: dict[float, tuple[float, float]] = {}
    for degrees in (0.0, 5.0, 10.0, 20.0):
      joint_pos = base_joint_pos.clone()
      joint_pos[:, quad_asset.joint_ids] += math.radians(degrees)
      robot.write_joint_position_to_sim(joint_pos)
      env.scene.write_data_to_sim()
      env.sim.forward()
      raw = mdp.pose_quadratic(env, asset_cfg=quad_asset, std=std)
      costs[degrees] = (float(raw.mean()), weight * float(raw.mean()))
      print(
        f"  {degrees:>4.0f} deg: raw={costs[degrees][0]:.9f} "
        f"weighted={costs[degrees][1]:.9f}"
      )
    robot.write_joint_position_to_sim(base_joint_pos)
    env.scene.write_data_to_sim()
    env.sim.forward()
    checks.check(
      "cost is zero at zero offset",
      abs(costs[0.0][0]) <= 1e-12,
      f"{costs[0.0][0]:.3e}",
    )
  finally:
    env.close()

  print()
  print("[6] PD gains match the juggle family")
  carry_gains = _vendor_gains_by_joint(cfg.scene.entities["robot"])
  juggle_cfg, juggle_env = _build(JUGGLE_TASK_ID)
  try:
    juggle_gains = _vendor_gains_by_joint(juggle_cfg.scene.entities["robot"])
    juggle_model = juggle_env.sim.mj_model
    juggle_bed = [
      g for g in range(juggle_model.ngeom) if BED_GEOM in juggle_model.geom(g).name
    ]
    juggle_bed_bits = (
      (
        int(juggle_model.geom_contype[juggle_bed[0]]),
        int(juggle_model.geom_conaffinity[juggle_bed[0]]),
      )
      if len(juggle_bed) == 1
      else None
    )
    juggle_colliding_racket = [
      juggle_model.geom(g).name
      for g in range(juggle_model.ngeom)
      if any(
        key in juggle_model.geom(g).name for key in (RACKET_BODY, CLAMP_BODY, BED_GEOM)
      )
      and (
        int(juggle_model.geom_contype[g]) != 0
        or int(juggle_model.geom_conaffinity[g]) != 0
      )
    ]
  finally:
    juggle_env.close()
  differing = {
    name: (carry_gains.get(name), juggle_gains.get(name))
    for name in set(carry_gains) | set(juggle_gains)
    if carry_gains.get(name) != juggle_gains.get(name)
  }
  checks.check(
    "PD gains equal the tennis juggle task on every vendor joint",
    not differing,
    f"{len(carry_gains)} joints, differing={differing}",
  )
  checks.check(
    "shoulder and wrist use the vendor RL gains, not the velocity task's",
    carry_gains.get("right_shoulder_pitch_joint") == (50.0, 3.0)
    and carry_gains.get("right_wrist_yaw_joint") == (20.0, 2.0),
    f"shoulder={carry_gains.get('right_shoulder_pitch_joint')} "
    f"wrist={carry_gains.get('right_wrist_yaw_joint')}",
  )

  print("[7] the change is scoped to the carry task")
  checks.check(
    "the juggle family keeps a colliding bed",
    juggle_bed_bits == (1, 1),
    f"juggle bed contype/conaffinity={juggle_bed_bits}",
  )
  checks.check(
    "the bed is the racket's only collider, so the juggle family still has one",
    juggle_colliding_racket == ["robot/string_bed"],
    f"juggle colliding racket geoms={juggle_colliding_racket}",
  )

  if checks.failures:
    print(f"RESULT: FAIL ({len(checks.failures)} checks)")
    for failure in checks.failures:
      print(f"  - {failure}")
    return 1
  print("RESULT: PASS")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
