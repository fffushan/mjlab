"""Vendor X2 reference-policy PD gains, and the helper that installs them.

The values are the servo gains the vendor's own reference tracking policy uses,
taken from the X2 RL deploy package v1.1 that ships with the robot::

  x2_references/aimdk-extra-x2_rl_deploy-v1.1/aimdk-extra/x2_rl_deploy/
    x2_rl_deploy_controller/config/motion_control.yaml

``rl_config.kps`` and ``rl_config.kds`` hold one gain pair per entry of that
policy's 29-joint ``seq``. The two head joints are absent from the sequence, so a
task built from this table keeps the mjlab head gains instead of inventing vendor
values for them. The same file also carries a ``joint_default`` table with
*different* values (ankle pitch ``40/4`` there against ``40/2`` in ``rl_config``);
those describe a different control mode and must not be mixed with this one.

Deliberate deviation from the vendor file: the three waist joints are retuned,
because the vendor damping left a sustained high-frequency waist vibration on the
robot. The change is ``waist_yaw`` ``40.1792/2.5579`` -> ``40/4`` and
``waist_pitch``/``waist_roll`` ``200/2`` -> ``200/4``; every other joint keeps its
vendor value. The retune was chosen from two recorded hardware trials
(``/home/agiuser/data/20261008T094713Z_vendor-dr-005-vib`` and
``20261008T124731Z_tune-kd-001``) and removes the measured 20-30 Hz waist
vibration, at the cost of a small waist-yaw position-RMSE increase. It is an
operator-selected hardware fix, not a re-measurement of the vendor servo gains.

mjlab configures gains per actuator group, while the vendor table is per joint,
so :func:`apply_vendor_rl_gains` matches joint names against each group's own
``target_names_expr`` and refuses a group whose joints disagree.
"""

from __future__ import annotations

import re
from copy import deepcopy

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityCfg

# The vendor's `rl_config.kps` / `rl_config.kds`, keyed by joint name and ordered
# as the vendor's own `seq` for review against the source file.
X2_VENDOR_RL_GAINS: dict[str, tuple[float, float]] = {
  "left_hip_pitch_joint": (120.0, 5.0),
  "right_hip_pitch_joint": (120.0, 5.0),
  "waist_yaw_joint": (40.0, 4.0),
  "left_hip_roll_joint": (100.0, 4.0),
  "right_hip_roll_joint": (100.0, 4.0),
  "waist_pitch_joint": (200.0, 4.0),
  "left_hip_yaw_joint": (100.0, 4.0),
  "right_hip_yaw_joint": (100.0, 4.0),
  "waist_roll_joint": (200.0, 4.0),
  "left_knee_joint": (150.0, 5.0),
  "right_knee_joint": (150.0, 5.0),
  "left_shoulder_pitch_joint": (50.0, 3.0),
  "right_shoulder_pitch_joint": (50.0, 3.0),
  "left_ankle_pitch_joint": (40.0, 2.0),
  "right_ankle_pitch_joint": (40.0, 2.0),
  "left_shoulder_roll_joint": (50.0, 3.0),
  "right_shoulder_roll_joint": (50.0, 3.0),
  "left_ankle_roll_joint": (40.0, 2.0),
  "right_ankle_roll_joint": (40.0, 2.0),
  "left_shoulder_yaw_joint": (50.0, 3.0),
  "right_shoulder_yaw_joint": (50.0, 3.0),
  "left_elbow_joint": (50.0, 3.0),
  "right_elbow_joint": (50.0, 3.0),
  "left_wrist_yaw_joint": (20.0, 2.0),
  "right_wrist_yaw_joint": (20.0, 2.0),
  "left_wrist_pitch_joint": (20.0, 2.0),
  "right_wrist_pitch_joint": (20.0, 2.0),
  "left_wrist_roll_joint": (20.0, 2.0),
  "right_wrist_roll_joint": (20.0, 2.0),
}


def _vendor_joints_of(group: BuiltinPositionActuatorCfg) -> tuple[str, ...]:
  """Vendor joints whose name matches the group's own target expressions."""
  return tuple(
    name
    for name in X2_VENDOR_RL_GAINS
    if any(re.fullmatch(pattern, name) for pattern in group.target_names_expr)
  )


def apply_vendor_rl_gains(
  robot_cfg: EntityCfg,
) -> list[tuple[tuple[str, ...], float, float, tuple[str, ...]]]:
  """Replace the robot's PD gains with the vendor reference values, in place.

  Returns one ``(target_names_expr, kp, kd, joints)`` row per changed actuator
  group so a caller can report exactly what changed.

  The articulation is deep-copied first: :func:`~mjlab.asset_zoo.robots.
  agibot_x2.x2_constants.get_x2_robot_cfg` shares one module-level articulation
  between every task, so editing it in place would change unrelated tasks.

  Raises:
    ValueError: if a group spans vendor joints with different gains, or if some
      vendor joint matches no group, which means the group layout and this table
      have diverged.
  """
  articulation = deepcopy(robot_cfg.articulation)
  report: list[tuple[tuple[str, ...], float, float, tuple[str, ...]]] = []
  matched: set[str] = set()

  for group in articulation.actuators:
    if not isinstance(group, BuiltinPositionActuatorCfg):
      raise TypeError(
        f"vendor gains need a position actuator group, got {type(group).__name__}"
      )
    joints = _vendor_joints_of(group)
    if not joints:
      continue
    values = {X2_VENDOR_RL_GAINS[name] for name in joints}
    if len(values) != 1:
      raise ValueError(
        f"actuator group {group.target_names_expr} spans vendor joints with "
        f"different gains: "
        f"{ {name: X2_VENDOR_RL_GAINS[name] for name in joints} }"
      )
    stiffness, damping = values.pop()
    group.stiffness = stiffness
    group.damping = damping
    matched.update(joints)
    report.append((group.target_names_expr, stiffness, damping, joints))

  missing = sorted(set(X2_VENDOR_RL_GAINS) - matched)
  if missing:
    raise ValueError(
      f"no actuator group matches vendor joints {missing}; the robot's actuator "
      f"groups and the vendor table have diverged"
    )

  robot_cfg.articulation = articulation
  return report
