"""Probe: the measured right-arm tasks differ from their shipped twins in two servos.

Run:

    python scripts/experiments/x2_right_arm_measured/probe_right_arm_measured.py

What it proves, and why each check is worth the line:

[1] Every stage of both families has a measured twin registered, and the two families
    are generated from the same stage tables, so a stage cannot exist in one and be
    missing from the other.
[2] A measured twin's robot differs from the shipped robot in exactly two servo entries:
    the right shoulder pitch and roll damping. This is the whole change. If a third
    entry ever moves, the rung is not comparable with its pair.
[3] The action-to-joint mapping is identical, taken from the compiled models'
    ``actuator_trnid`` actuator by actuator rather than from the config objects. The
    action vector is ordered by actuator, so a reordering would silently remap every
    checkpoint and the controller's 29-wide action contract. The compiled stiffness of
    every actuator and every effort limit are compared too, and the set of actuators
    whose damping moved must be exactly the two the servo table names.
[4] The compiled armature is identical. The measured data changed damping, not inertia,
    so `dof_armature` must not move.
[5] The values match the fit they claim to come from: `kd = 2 zeta omega_n J` with
    `zeta = 0.70` and `J` the compiled mass-matrix diagonal at the carry pose.
[6] The observation widths and the action count are unchanged, so a shipped checkpoint
    resumes into its measured twin with `strict=True`.
"""

from __future__ import annotations

import contextlib
import io
import math

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.agibot_x2.right_arm_measured import (
  RIGHT_ARM_MEASURED_SERVO,
)
from mjlab.asset_zoo.robots.agibot_x2.tennis_constants import BAT_CARRY_POSE
from mjlab.entity import Entity
from mjlab.tasks.velocity.config.agibot_x2.tennis_juggle_stage_env_cfg import (
  STAGE_DELTAS,
  agibot_x2_tennis_juggle_stage_env_cfg,
)
from mjlab.tasks.velocity.config.agibot_x2.tennis_juggle_target_height_env_cfg import (
  STAGE_TARGET_HEIGHT_RANGE_M,
  agibot_x2_tennis_juggle_target_height_env_cfg,
)

MEASURED_FREQ_HZ = {
  "right_shoulder_pitch_joint": 2.34,
  "right_shoulder_roll_joint": 2.52,
}

MEASURED_ZETA = 0.70
"""The damping ratio the right-arm fit measured on both shoulder axes, racket held."""

EXPECTED_MEASURED = {
  "right_shoulder_pitch_joint": (50.0, 4.81),
  "right_shoulder_roll_joint": (50.0, 4.78),
}

failures: list[str] = []


def check(ok: bool, message: str) -> None:
  if not ok:
    failures.append(message)


def robot_of(cfg) -> Entity:
  return Entity(cfg.scene.entities["robot"])


def servo_table(entity: Entity) -> tuple[list[str], dict[str, tuple[float, float]]]:
  """Resolved joint order, and the ``(kp, kd)`` each joint's actuator carries."""
  order: list[str] = []
  table: dict[str, tuple[float, float]] = {}
  for actuator in entity.actuators:
    for name in actuator.target_names:
      order.append(name)
      table[name] = (actuator.cfg.stiffness, actuator.cfg.damping)
  return order, table


def measured_diagonal(entity: Entity) -> tuple[dict[str, float], object]:
  """Mass-matrix diagonal at the carry pose, per joint, kg m2.

  The fit was made with **the racket held**, so the entity passed in must be the tennis
  robot with the racket welded on. The racket is a large part of the shoulder inertia:
  0.2357 kg m2 against 0.1851 for the bare robot on the pitch axis, which is the
  difference between "kd 4.25" and "kd 4.81".
  """
  model = entity.spec.compile()
  data = mujoco.MjData(model)
  qpos = np.array(model.qpos0, copy=True)
  for name, value in BAT_CARRY_POSE.items():
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint >= 0:
      qpos[model.jnt_qposadr[joint]] = value
  data.qpos[:] = qpos
  mujoco.mj_forward(model, data)
  mass = np.zeros((model.nv, model.nv))
  mujoco.mj_fullM(model, data, mass)
  out = {}
  for name in RIGHT_ARM_MEASURED_SERVO:
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    out[name] = float(mass[model.jnt_dofadr[joint], model.jnt_dofadr[joint]])
  return out, model


#: Config sections that must be identical between a stage and its measured twin. The
#: robot entity is the only thing allowed to differ, and ``scene`` is compared on its
#: non-robot entities below instead.
COMPARED_SECTIONS = (
  "observations",
  "rewards",
  "terminations",
  "metrics",
  "events",
  "commands",
  "actions",
  "curriculum",
  "sim",
  "episode_length_s",
  "decimation",
)


def compare(label: str, base_cfg, measured_cfg) -> None:
  with contextlib.redirect_stdout(io.StringIO()):
    base = robot_of(base_cfg)
    measured = robot_of(measured_cfg)

  order_a, table_a = servo_table(base)
  order_b, table_b = servo_table(measured)
  check(order_a == order_b, f"{label}: the actuator order changed")
  check(len(order_a) == 31, f"{label}: {len(order_a)} actuators, expected 31")

  changed = {
    name: (table_a[name], table_b[name])
    for name in table_a
    if table_a[name] != table_b[name]
  }
  check(
    set(changed) == set(EXPECTED_MEASURED),
    f"{label}: servos that moved are {sorted(changed)}, expected "
    f"{sorted(EXPECTED_MEASURED)}",
  )
  for name, (before, after) in sorted(changed.items()):
    check(
      before[0] == after[0],
      f"{label}: {name} stiffness moved {before[0]} -> {after[0]}, the measurement "
      "was of damping only",
    )
    check(
      after == EXPECTED_MEASURED[name],
      f"{label}: {name} is {after}, expected {EXPECTED_MEASURED[name]}",
    )

  base_model = base.spec.compile()
  measured_model = measured.spec.compile()
  check(
    np.array_equal(base_model.dof_armature, measured_model.dof_armature),
    f"{label}: dof_armature changed, but the measurement was of damping",
  )

  # The action vector is ordered by actuator, and the action-to-joint mapping is what a
  # reorder would break silently. Compare it on the compiled models, actuator by
  # actuator, not on the config objects.
  def action_map(model) -> list[str]:
    return [
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(joint))
      for joint in model.actuator_trnid[:, 0]
    ]

  map_a, map_b = action_map(base_model), action_map(measured_model)
  check(
    map_a == map_b,
    f"{label}: the action-to-joint mapping changed",
  )
  check(
    np.array_equal(
      base_model.actuator_gainprm[:, 0], measured_model.actuator_gainprm[:, 0]
    ),
    f"{label}: a stiffness changed in the compiled model",
  )
  kd_a = -base_model.actuator_biasprm[:, 2]
  kd_b = -measured_model.actuator_biasprm[:, 2]
  moved = [map_b[i] for i in range(len(kd_a)) if abs(kd_a[i] - kd_b[i]) > 1e-12]
  check(
    sorted(moved) == sorted(EXPECTED_MEASURED),
    f"{label}: the compiled model damped {moved}",
  )
  check(
    np.array_equal(base_model.actuator_forcerange, measured_model.actuator_forcerange),
    f"{label}: an effort limit changed",
  )

  # Everything outside the robot must be untouched: a measured rung is the same
  # experiment on a different robot, not a new condition.
  for section in COMPARED_SECTIONS:
    check(
      getattr(base_cfg, section) == getattr(measured_cfg, section),
      f"{label}: {section} differs between the stage and its measured twin",
    )
  extra_a = {name for name in base_cfg.scene.entities if name != "robot"}
  extra_b = {name for name in measured_cfg.scene.entities if name != "robot"}
  check(
    extra_a == extra_b,
    f"{label}: the non-robot scene entities changed from {sorted(extra_a)} to "
    f"{sorted(extra_b)}",
  )


def main() -> int:
  print("check 1  both families have a measured twin for every stage")
  import mjlab.tasks  # noqa: F401  (registration side effect)
  from mjlab.tasks.registry import list_tasks

  registered = sorted(t for t in list_tasks() if "RightArmMeasured" in t)
  expected_count = len(STAGE_DELTAS) + len(STAGE_TARGET_HEIGHT_RANGE_M)
  check(
    len(registered) == expected_count,
    f"{len(registered)} measured tasks registered, expected {expected_count} "
    f"({len(STAGE_DELTAS)} ladder + {len(STAGE_TARGET_HEIGHT_RANGE_M)} target height)",
  )
  print(
    f"  ok   {len(registered)} measured tasks: {len(STAGE_DELTAS)} ladder stages and "
    f"{len(STAGE_TARGET_HEIGHT_RANGE_M)} target-height stages"
  )

  print("check 2  a measured twin moves only the two right shoulder dampings")
  for stage in STAGE_DELTAS:
    with contextlib.redirect_stdout(io.StringIO()):
      base = agibot_x2_tennis_juggle_stage_env_cfg(stage=stage)
      measured = agibot_x2_tennis_juggle_stage_env_cfg(
        stage=stage, right_arm_measured=True
      )
    compare(f"ladder {stage}", base, measured)
  for stage in STAGE_TARGET_HEIGHT_RANGE_M:
    with contextlib.redirect_stdout(io.StringIO()):
      base = agibot_x2_tennis_juggle_target_height_env_cfg(stage=stage)
      measured = agibot_x2_tennis_juggle_target_height_env_cfg(
        stage=stage, right_arm_measured=True
      )
    compare(f"target-height {stage}", base, measured)
  check(not failures, "see the failures above")
  print(
    f"  ok   {len(STAGE_DELTAS) + len(STAGE_TARGET_HEIGHT_RANGE_M)} pairs: two servo "
    "entries, identical actuator order, identical armature"
  )
  print(
    f"  ok   every other config section identical in all "
    f"{len(STAGE_DELTAS) + len(STAGE_TARGET_HEIGHT_RANGE_M)} pairs: "
    f"{', '.join(COMPARED_SECTIONS)}"
  )

  print(
    "check 3  the values match the fit they claim, at the inertia that was measured"
  )
  with contextlib.redirect_stdout(io.StringIO()):
    tennis = robot_of(
      agibot_x2_tennis_juggle_stage_env_cfg(
        stage="s11-tilted-start", right_arm_measured=True
      )
    )
  diagonal, model = measured_diagonal(tennis)
  for joint, (kp, kd) in sorted(RIGHT_ARM_MEASURED_SERVO.items()):
    inertia = diagonal[joint]
    omega = math.sqrt(kp / inertia)
    predicted = 2.0 * MEASURED_ZETA * omega * inertia
    measured_hz = MEASURED_FREQ_HZ[joint]
    sim_hz = omega / (2.0 * math.pi)
    check(
      abs(predicted - kd) < 0.05,
      f"{joint}: kd {kd} against the fit's {predicted:.3f} at J {inertia:.4f}",
    )
    check(
      abs(sim_hz - measured_hz) / measured_hz < 0.10,
      f"{joint}: the fit's frequency agreement slipped, sim {sim_hz:.2f} Hz against "
      f"measured {measured_hz:.2f} Hz",
    )
    print(
      f"  ok   {joint:28s} J {inertia:.4f} kg m2, kp {kp:.1f} -> {sim_hz:.2f} Hz "
      f"(measured {measured_hz:.2f}), kd {kd:.2f} for zeta {MEASURED_ZETA:.2f} "
      f"(the fit gives {predicted:.2f})"
    )
  for joint, (kp_expected, kd_expected) in sorted(RIGHT_ARM_MEASURED_SERVO.items()):
    joint_id = mujoco.mj_name2id(
      tennis_model := model, mujoco.mjtObj.mjOBJ_JOINT, joint
    )
    actuator = next(
      a for a in range(tennis_model.nu) if tennis_model.actuator_trnid[a, 0] == joint_id
    )
    kp_model = -tennis_model.actuator_biasprm[actuator, 1]
    kd_model = -tennis_model.actuator_biasprm[actuator, 2]
    check(
      abs(kp_model - kp_expected) < 1e-9 and abs(kd_model - kd_expected) < 1e-9,
      f"{joint}: the compiled model carries {kp_model}/{kd_model}",
    )
  print("  ok   the compiled MuJoCo actuators carry the same gains")

  if failures:
    print()
    print(f"RESULT: FAIL ({len(failures)})")
    for failure in failures:
      print(f"  - {failure}")
    return 1
  print()
  print("RESULT: PASS")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
