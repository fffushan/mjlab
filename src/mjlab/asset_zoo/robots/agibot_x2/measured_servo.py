"""Measured servo damping for every X2 joint, for the tracking tasks.

Provenance
----------
Two joint-step runs on the real robot, both at the carry pose with instant target
changes of 0.5, 1, 3, 5 and 10 degrees both signs and 3 s holds, at the vendor RL
gains these tasks train with:

* ``x2_docker/docs/reports/REPORT-joint-step-response.md`` -- the seven right-arm
  joints, racket held.
* ``x2_docker/docs/reports/REPORT-sim-vs-robot-servo.md`` -- the other 24 joints:
  legs, waist, left arm, head.

How each ``kd`` was chosen
--------------------------
By the residual rule of the right-arm report's section 4c, applied to all 31
joints with ``x2_docker/tools/joint_servo_kd.py``. The rule holds the simulator's
closed-loop frequency fixed (``kd`` cannot move it) and sweeps ``kd`` alone until
the second-order step response matches the measured steps with the smallest
normalised shape residual. One free parameter, the same objective the two-joint
servo fit uses, so the value is the fit's own answer rather than a formula.

The tool reproduces the published right-arm answers from the raw steps: 4.73 /
4.78 / 4.28 / 3.85 against the published 4.74 / 4.78 / 4.38 / 3.90, with the same
residuals (9.5 %, 10.1 %, 6.2 %, 4.7 % at the shipped ``kd``). That is the reason
to trust it on the other 24 joints. The right-arm entries below are therefore
taken from :data:`RIGHT_ARM_MEASURED_SERVO` itself, not re-derived, so the two
task families cannot drift apart.

Shape residual, shipped against adopted, over the 3 deg and larger steps:

  group            joint            shipped kd  adopted kd  residual  ->  adopted
  legs             hip pitch      5.00 L/5.00 R  14.48/12.80   26.1/23.7 % -> 6.7/7.0 %
                   hip roll       4.00           10.20/ 9.88   23.8/22.7 % -> 9.8/8.6 %
                   hip yaw        4.00            5.25/ 5.87    4.2/ 6.3 % -> 2.3/4.2 %
                   knee           5.00            8.57/ 8.79    9.7/10.1 % -> 3.7/4.3 %
  waist            yaw            4.00            7.55         18.9 %     -> 4.9 %
                   pitch          4.00           34.71         46.5 %     -> 7.3 %
                   roll           4.00           20.55         36.4 %     -> 7.8 %
  left arm         shoulder pitch 3.00            4.93         11.2 %     -> 3.4 %
                   shoulder roll  3.00            4.45          9.1 %     -> 3.5 %
                   shoulder yaw   3.00            3.65          4.0 %     -> 3.1 %
                   elbow          3.00            3.75          4.1 %     -> 2.2 %
  right arm        shoulder pitch 3.00            4.74          9.5 %     -> 2.1 %
  (published)      shoulder roll  3.00            4.78         10.1 %     -> 2.8 %
                   shoulder yaw   3.00            4.38          6.2 %     -> 2.3 %
                   elbow          3.00            3.90          4.7 %     -> 2.9 %

Unchanged, with the reason each one has nothing to buy:

* **Ankles** (``kd`` 2.0): the simulator already describes them, residual 2.1-3.8 %
  at the shipped value, which is the measurement floor.
* **Wrists** (``kd`` 2.0): residual 2.7-6.2 %, and the sweep saves at most 1 % on the
  three that exceed the floor. This repeats the right-arm decision on the left side.
* **Head** (``kd`` 0.11 yaw / 0.05 pitch, the mjlab values): the step test drove the
  head with ``kp`` 3.4 and 1, which produces a degenerate overdamped corner. The
  24-joint report says explicitly not to read those rows as a servo, so the sweep's
  0.28 / 0.11 are not adopted.

How much of this is supported
-----------------------------
Two tiers, and the difference matters:

1. **Reaches the floor** (adopted residual 2.1-4.9 %): the four knees, the four hips'
   yaw, the shoulders and elbows on both sides, and the waist yaw. At these joints a
   second-order servo with the measured damping describes the measured steps as well
   as the two-parameter fit described them.
2. **Compromise** (adopted residual 6.7-9.8 %): hip pitch, hip roll, waist pitch and
   waist roll. The sweep removes 59-84 % of the shape error and the resulting ``kd``
   is the best a single damping value can do, but 7-10 % is two to three times the
   measurement floor. The reason is that the simulator's frequency is wrong at these
   joints and ``kd`` cannot move it: the waist pitch is overdamped on the robot
   (``zeta`` 3.26) and under-damped in the simulator (0.11) at 1.76 Hz, and no damping
   value reconciles that. Treat these four as the best available compromise, not as
   an identified servo.

**The legs carry an extra caveat.** The 24-joint run recorded
``--load-bearing-deg 0``, which states an assumption and not a physical fact: whether
the feet were clear of the floor during the leg steps is **unknown**. If they were
touching, the fitted leg damping and inertia include contact, and the leg ``kd``
values above inherit that. The report lists this as its first open item.

What this is not
----------------
**This is a training-side model, not a deployment gain.** The measurements are what
the robot already does at the vendor ``kd``; the simulator is the side that is
wrong. Sending 14.48 or 34.71 to the robot would be a different, unmeasured change.
The controller takes the RL gains from the exported policy metadata times
``policy.gain_multiplier``, so **a measured-servo policy must never be deployed
without an explicit per-joint hardware gain override**.

Dry friction, which both reports name as the mechanism behind the amplitude-dependent
overshoot (0.9 % at 3 deg against 4.3-7.4 % at 10 deg on the shoulder pitch), is not
modelled here and cannot be: a linear servo cannot produce it.

``randomize_pd_gains`` still scales every ``kd`` by 0.7-1.3 per environment at reset,
so these values are the nominal centre of that range, not a fixed configuration.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import replace

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityCfg

from .right_arm_measured import RIGHT_ARM_MEASURED_SERVO
from .x2_constants import actuator_group_index_by_patterns

#: The wrist group's own target expressions. Both families keep the wrist damping at 2.0,
#: so this group is never split and the patterns stay exact.
WRIST_GROUP_PATTERNS: tuple[str, ...] = (
  ".*_wrist_pitch_joint",
  ".*_wrist_roll_joint",
)

#: The model's joint order, from ``get_spec().joints`` minus the free joint. Actuator
#: groups resolve their regex targets in this order, so a split that follows it leaves
#: the actuator list and the action contract unchanged.
X2_JOINT_ORDER: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_pitch_joint",
  "waist_roll_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_yaw_joint",
  "left_wrist_pitch_joint",
  "left_wrist_roll_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_yaw_joint",
  "right_wrist_pitch_joint",
  "right_wrist_roll_joint",
  "head_yaw_joint",
  "head_pitch_joint",
)

#: Measured servo damping, ``joint -> kd``, for every joint whose value moves. Joints
#: absent from this table keep whatever the task already set: ankles 2.0, wrists 2.0,
#: head 0.11/0.05. ``kp`` is never touched here.
X2_MEASURED_SERVO_KD: dict[str, float] = {
  # Legs. Compromise on the two pitch axes, floor elsewhere.
  "left_hip_pitch_joint": 14.48,
  "right_hip_pitch_joint": 12.80,
  "left_hip_roll_joint": 10.20,
  "right_hip_roll_joint": 9.88,
  "left_hip_yaw_joint": 5.25,
  "right_hip_yaw_joint": 5.87,
  "left_knee_joint": 8.57,
  "right_knee_joint": 8.79,
  # Waist. The pitch and roll are the two largest departures, and both are compromise.
  "waist_yaw_joint": 7.55,
  "waist_pitch_joint": 34.71,
  "waist_roll_joint": 20.55,
  # Left arm: the same rule, on the run that measured it.
  "left_shoulder_pitch_joint": 4.93,
  "left_shoulder_roll_joint": 4.45,
  "left_shoulder_yaw_joint": 3.65,
  "left_elbow_joint": 3.75,
  # Right arm: the published values the tennis tasks already train with.
  **{
    joint: damping
    for joint, (_stiffness, damping) in RIGHT_ARM_MEASURED_SERVO.items()
  },
}


def _joints_of(group: BuiltinPositionActuatorCfg) -> tuple[str, ...]:
  """Joints of ``X2_JOINT_ORDER`` that the group's own patterns match, in model order."""
  return tuple(
    joint
    for joint in X2_JOINT_ORDER
    if any(re.fullmatch(pattern, joint) for pattern in group.target_names_expr)
  )


def group_index_of(robot_cfg: EntityCfg, patterns: tuple[str, ...]) -> int:
  """Index of the one actuator group whose target expressions are exactly ``patterns``.

  A split moves groups, so any parameter that addresses a group by position has to be
  re-resolved after :func:`apply_measured_servo`. ``SceneEntityCfg.actuator_ids`` is such
  a parameter: it indexes the config's group list, not the compiled actuators.

  Args:
    robot_cfg: The robot config, after any split.
    patterns: The group's ``target_names_expr``, in its own order.

  Returns:
    The group's index in ``robot_cfg.articulation.actuators``.

  Raises:
    ValueError: if no group has exactly those patterns, or more than one does.
  """
  articulation = robot_cfg.articulation
  if articulation is None:
    raise ValueError("the robot config has no articulation")
  return actuator_group_index_by_patterns(articulation, patterns)


def retarget_wrist_effort_event(cfg, robot_cfg) -> None:
  """Re-resolve the wrist effort-derating event on ``cfg``, in place.

  ``apply_x2_tracking_robot_randomization`` addresses the wrist group by position,
  because ``SceneEntityCfg.actuator_ids`` indexes the config's group list. A split
  inserts groups before the wrist one, so a stored position then points at whatever
  group moved into it. The event would derate another joint's effort limit and leave
  the wrists untouched, and the ordering assertion would still pass because it
  compares names.

  Call this on every config whose robot articulation was split after the domain
  randomisation was installed. A split cannot change the wrist group's own patterns,
  because the wrists keep ``kd`` 2.0 in both families, so their group is never split.

  Args:
    cfg: The environment config holding the event.
    robot_cfg: The robot config, after any split.
  """
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  event = cfg.events.get("randomize_effort_limits_wrist")
  if event is None:
    return
  index = group_index_of(robot_cfg, WRIST_GROUP_PATTERNS)
  event.params["asset_cfg"] = SceneEntityCfg("robot", actuator_ids=[index])


def apply_measured_servo(
  robot_cfg: EntityCfg,
) -> list[tuple[tuple[str, ...], float, float, float]]:
  """Install the measured servo damping on ``robot_cfg``, in place.

  Must run *after* :func:`~mjlab.tasks.tracking.config.agibot_x2.vendor_gains.
  apply_vendor_rl_gains`, which writes one ``(kp, kd)`` per group and would otherwise
  overwrite these values with the symmetric vendor ones.

  A group whose joints all keep their own damping is left as one group. A group whose
  joints need different damping is split into one group per joint, in the model's joint
  order, which is the order the shared pattern already resolved to -- so the actuator
  count and the action-index order do not change, only the damping values.

  Args:
    robot_cfg: The robot config to modify. Its ``articulation`` is replaced.

  Returns:
    One ``(target_names_expr, kp, kd_before, kd_after)`` row per changed group or
    split-off joint, for reporting.

  Raises:
    ValueError: if a table joint matches no group, if a joint ends up owned by other
      than exactly one group, or if a joint's damping is not what this module says.
  """
  articulation = deepcopy(robot_cfg.articulation)
  if articulation is None:
    raise ValueError("the robot config has no articulation to modify")

  covered = {joint for group in articulation.actuators for joint in _joints_of(group)}
  missing = sorted(set(X2_MEASURED_SERVO_KD) - covered)
  if missing:
    raise ValueError(
      f"no actuator group matches the measured joints {missing}; the robot's "
      f"actuator layout and the measured table have diverged"
    )

  report: list[tuple[tuple[str, ...], float, float, float]] = []
  rebuilt: list[BuiltinPositionActuatorCfg] = []
  for group in articulation.actuators:
    if not isinstance(group, BuiltinPositionActuatorCfg):
      raise TypeError(
        f"the measured servo needs position actuator groups, got {type(group).__name__}"
      )
    joints = _joints_of(group)
    if not joints:
      rebuilt.append(group)
      continue
    damping = {joint: X2_MEASURED_SERVO_KD.get(joint, group.damping) for joint in joints}
    if len(set(damping.values())) == 1:
      value = damping[joints[0]]
      if value != group.damping:
        report.append((group.target_names_expr, group.stiffness, group.damping, value))
        rebuilt.append(replace(group, damping=value))
      else:
        rebuilt.append(group)
      continue
    for joint in joints:
      report.append(
        (group.target_names_expr, group.stiffness, group.damping, damping[joint])
      )
      rebuilt.append(
        replace(
          group,
          target_names_expr=(joint,),
          damping=damping[joint],
        )
      )

  # Every joint of the model must still be owned by exactly one group, and the joints
  # this module claims must carry exactly the damping it states.
  for joint in X2_JOINT_ORDER:
    owners = [group for group in rebuilt if joint in _joints_of(group)]
    if len(owners) != 1:
      raise ValueError(
        f"{joint} is matched by {len(owners)} actuator groups after the split, "
        "expected exactly one"
      )
    expected = X2_MEASURED_SERVO_KD.get(joint)
    if expected is not None and owners[0].damping != expected:
      raise ValueError(
        f"{joint} carries kd {owners[0].damping} after the split, expected {expected}"
      )

  articulation.actuators = tuple(rebuilt)
  robot_cfg.articulation = articulation
  return report


__all__ = [
  "WRIST_GROUP_PATTERNS",
  "X2_JOINT_ORDER",
  "X2_MEASURED_SERVO_KD",
  "apply_measured_servo",
  "group_index_of",
  "retarget_wrist_effort_event",
]
