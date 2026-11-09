"""Measured right-arm servo gains for the X2 tennis tasks.

Provenance
----------
Measured on the real robot ``.234`` on 2026-11-09 by the joint step test in
``x2_docker/tools/joint_step_test.py`` (carry pose, **racket held**, controller stopped,
vendor state ``Develop_MC``): 0.5/1/3/10 deg steps both signs, 3 s holds, at the vendor
RL gains these tasks train with. Analysis: ``x2_docker/docs/reports/
REPORT-joint-step-response.md``, peer audit ``AUDIT-joint-step-model-20261109.md``.

The racket is part of what was measured, so the inertia that belongs in these gains is
the tennis scene's, racket welded on, at the carry pose -- not the bare robot's.

What the data supports, and what it does not
--------------------------------------------
The report's own section 5 is the constraint: a wholesale replacement of the actuator
model is *not* supported, and the one change it does support is that the simulator is
more lightly damped than the robot on the shoulder group. Measured damping ratio 0.70
on both shoulder axes; at the racket-held inertia the shipped gains give 0.437 and
0.439, so 0.8-0.9 % overshoot on the robot against **21.7 % and 21.5 %** in the
simulator at 10 deg. (The report first quoted 16.7 / 18.6 %; those were computed on the
bare robot's inertia and are corrected in its section 0.)

So this module changes **damping only, on the two right shoulder axes**, and keeps the
commanded stiffness. ``kd = 2 zeta sqrt(kp J)`` with ``zeta = 0.70`` and ``J`` the
mass-matrix diagonal in the tennis scene at the carry pose, racket held:

  shoulder_pitch  J 0.2357  kp 50  ->  kd 4.81   (shipped 3.0 gives zeta 0.44)
  shoulder_roll   J 0.2333  kp 50  ->  kd 4.78   (shipped 3.0 gives zeta 0.44)

With the racket included the shoulder frequency already agrees: the simulator gives
2.32 Hz on the pitch axis against 2.34 Hz measured. So on that axis the inertia is
corroborated and damping is the only defect.

**This is a training-side change, not a deployment gain.** The robot already produces
zeta 0.70 at the commanded kd 3.0, so the extra damping in the simulator is imitating
damping the hardware already has. Sending kd 4.81 to the robot would add it a second
time and land near zeta 1.0. The controller takes the RL gains from the exported policy
metadata times ``policy.gain_multiplier`` (`x2_docker``'s ``x2_tennis_hardware.yaml``
states this explicitly), so **a measured-variant policy must never be deployed without
an explicit per-joint hardware gain override**. Nothing from this module has been
exported or deployed.

Deliberately *not* changed, with reasons:

* ``shoulder_yaw``, ``elbow``, ``wrist_*``. Two separate statements, because the peer
  audit is right that they are not the same statement:

  1. The **damping ratio is matchable without knowing the inertia**. For
     ``q'' = a u - a q - b q'`` both ``omega`` and ``zeta`` scale with ``sqrt(alpha)``,
     so ``zeta / omega_n = kd / (2 kp)`` contains neither ``J`` nor ``alpha``. An
     earlier version of this docstring said the residual gap "is the ``(J, alpha)``
     axis, which no damping value can fix". That is true of the **frequency** gap and
     false of ``zeta``. Raising kd on those joints would fix their damping ratio.
  2. It is still not done, for two reasons. The mismatch is small at the shipped gains
     -- 0.72 against 0.93 at the shoulder yaw, 0.85 against 0.93 at the elbow -- and at
     the wrists the shipped gains are already *over*-damped (1.07 and 1.12 against
     0.93), because the simulated wrist frequency is 25-28 % above the measured one.
     Matching ``kd / (2 kp)`` there, the inertia-free route, would raise the wrist kd to
     2.23 and 2.17 and make that over-damping worse (1.19 against 0.93); matching
     ``zeta`` at the simulated inertia would lower it to 1.74 and 1.65, trading a value
     with hardware trials behind it for a 15 % move on a joint whose model is wrong for
     a reason kd cannot address. The two prescriptions disagree by 20-35 % on exactly
     the joints where the frequency gap is largest, which is the signal that the data
     does not choose between them.

  If the yaw and elbow overshoot is judged to matter, the ``zeta``-at-simulated-inertia
  values are kd 3.88 and 3.30, and the report's ratio route gives 4.60 and 4.26. That is
  a decision to take explicitly, not a default.
* Dry friction, which the report names as the mechanism behind the amplitude-dependent
  overshoot but does not fit. Adding an unfitted friction term would be invention.
* Anything on the left arm. The measurement covers the right arm only.

The compiled armature does not move: MuJoCo's ``dof_armature`` comes from each group's
``armature`` field, which this module does not touch.

Why the groups are split per side
---------------------------------
A ``BuiltinPositionActuatorCfg`` holds one ``damping`` for all of its targets, and the
shipped articulation covers both arms with one regex per joint
(``.*_shoulder_pitch_joint``). A right-only damping therefore cannot be expressed in
that layout, so :func:`apply_right_arm_measured_servo` splits the two affected groups
into a left group and a right group, in that order. That order is what the shared group
already resolved to, so the actuator list is unchanged apart from the two damping values
-- which is what keeps every existing checkpoint and the controller's 29-wide action
contract valid.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import replace

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityCfg

#: Measured right-arm servo gains, ``joint -> (kp, kd)``. Only the damping departs from
#: the vendor reference table; ``kp`` repeats the commanded value, so this table is a
#: complete statement of the right arm's servo.
RIGHT_ARM_MEASURED_SERVO: dict[str, tuple[float, float]] = {
  "right_shoulder_pitch_joint": (50.0, 4.81),
  "right_shoulder_roll_joint": (50.0, 4.78),
}

#: ``right joint -> the shipped shared group pattern it must be split out of``. The
#: patterns are the shipped ones, so a layout change upstream fails loudly.
_SPLIT_FROM: dict[str, str] = {
  "right_shoulder_pitch_joint": ".*_shoulder_pitch_joint",
  "right_shoulder_roll_joint": ".*_shoulder_roll_joint",
}

#: The left joint that stays in each split group.
_LEFT_OF: dict[str, str] = {
  "right_shoulder_pitch_joint": "left_shoulder_pitch_joint",
  "right_shoulder_roll_joint": "left_shoulder_roll_joint",
}


def _matches(group: BuiltinPositionActuatorCfg, joint: str) -> bool:
  return any(re.fullmatch(pattern, joint) for pattern in group.target_names_expr)


def apply_right_arm_measured_servo(
  robot_cfg: EntityCfg,
) -> list[tuple[tuple[str, ...], float, float]]:
  """Install the measured right-arm servo on ``robot_cfg``, in place.

  Must run *after* :func:`~mjlab.tasks.tracking.config.agibot_x2.vendor_gains.
  apply_vendor_rl_gains`, which writes one ``(kp, kd)`` per group and would otherwise
  overwrite the right-arm values with the symmetric vendor ones.

  The articulation is deep-copied first, like ``apply_vendor_rl_gains``: the module
  level ``X2_ARTICULATION`` is shared between every task.

  Args:
    robot_cfg: The robot config to modify. Its ``articulation`` is replaced.

  Returns:
    One ``(target_names_expr, kp, kd)`` row per split-off right group, for reporting.

  Raises:
    ValueError: if a pattern this module expects is absent, or if a joint it must set is
      matched by other than exactly one group afterwards, which would mean the split did
      not take.
  """
  articulation = deepcopy(robot_cfg.articulation)
  if articulation is None:
    raise ValueError("the robot config has no articulation to modify")

  pending = dict(_SPLIT_FROM)
  report: list[tuple[tuple[str, ...], float, float]] = []
  rebuilt: list[BuiltinPositionActuatorCfg] = []
  for group in articulation.actuators:
    if not isinstance(group, BuiltinPositionActuatorCfg):
      raise TypeError(
        f"the measured servo needs position actuator groups, got {type(group).__name__}"
      )
    pattern = group.target_names_expr[0] if len(group.target_names_expr) == 1 else None
    right_joint = next((j for j, p in pending.items() if p == pattern), None)
    if right_joint is None:
      rebuilt.append(group)
      continue
    stiffness, damping = RIGHT_ARM_MEASURED_SERVO[right_joint]
    rebuilt.append(replace(group, target_names_expr=(_LEFT_OF[right_joint],)))
    right = replace(
      group, target_names_expr=(right_joint,), stiffness=stiffness, damping=damping
    )
    rebuilt.append(right)
    del pending[right_joint]
    report.append((right.target_names_expr, right.stiffness, right.damping))

  if pending:
    raise ValueError(
      f"the shipped actuator layout no longer has the groups this module splits: "
      f"{sorted(pending.values())}"
    )

  claimed = {*RIGHT_ARM_MEASURED_SERVO, *_LEFT_OF.values()}
  for joint in sorted(claimed):
    owners = [group for group in rebuilt if _matches(group, joint)]
    if len(owners) != 1:
      raise ValueError(
        f"{joint} is matched by {len(owners)} actuator groups after the split, "
        "expected exactly one"
      )
  for joint, (stiffness, damping) in RIGHT_ARM_MEASURED_SERVO.items():
    owner = next(group for group in rebuilt if _matches(group, joint))
    if (owner.stiffness, owner.damping) != (stiffness, damping):
      raise ValueError(
        f"{joint} carries {owner.stiffness}/{owner.damping} after the split, "
        f"expected {stiffness}/{damping}"
      )

  articulation.actuators = tuple(rebuilt)
  robot_cfg.articulation = articulation
  return report


__all__ = ["RIGHT_ARM_MEASURED_SERVO", "apply_right_arm_measured_servo"]
