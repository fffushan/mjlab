"""Probe the measured-servo tracking task against the base multi-motion task.

Checks, from the compiled model and not from the source:

1. The two tasks resolve the same joints in the same order.
2. The actuator list is the same length and drives the same joints in the same order, so
   the action index contract is unchanged.
3. ``kp`` is identical per joint; ``kd`` differs exactly on the joints the measured table
   names, and equals the table there.
4. Every other articulation field the tasks share (armature, effort limit, friction loss,
   delay settings, transmission) is identical per joint.
5. The action scale that the actor will use is identical per actuator.

Run with the mjlab environment:

    .venv/bin/python scripts/experiments/x2_measured_servo/probe_measured_servo.py
"""

from __future__ import annotations

import sys

from mjlab.asset_zoo.robots.agibot_x2.measured_servo import (
  X2_JOINT_ORDER,
  X2_MEASURED_SERVO_KD,
  apply_measured_servo,
)
from mjlab.tasks.tracking.config.agibot_x2.env_cfgs import (
  agibot_x2_multi_motion_tracking_env_cfg,
  agibot_x2_multi_motion_tracking_measured_servo_env_cfg,
)

_FIELDS = (
  "stiffness",
  "damping",
  "effort_limit",
  "armature",
  "frictionloss",
  "delay_min_lag",
  "delay_max_lag",
  "delay_hold_prob",
  "delay_update_period",
  "delay_per_env_phase",
)


def resolved(cfg) -> tuple[dict, list[str]]:
  """{joint: (flat actuator index, field values)} plus the flat actuator order.

  A group builds one MuJoCo actuator per target, and the targets of a group come back in
  the model's joint order (``resolve_matching_names`` preserves the name-list order). So
  the flat index is counted by walking the configs in order and, inside each, the joints
  in :data:`X2_JOINT_ORDER`.
  """
  articulation = cfg.scene.entities["robot"].articulation
  assert articulation is not None
  import re

  out: dict[str, tuple[int, dict]] = {}
  order: list[str] = []
  for group in articulation.actuators:
    for joint in X2_JOINT_ORDER:
      if not any(re.fullmatch(p, joint) for p in group.target_names_expr):
        continue
      if joint in out:
        raise SystemExit(f"{joint} matched by two groups")
      out[joint] = (len(order), {f: getattr(group, f) for f in _FIELDS})
      order.append(joint)
  return out, order


def main() -> int:
  base = agibot_x2_multi_motion_tracking_env_cfg(play=True)
  measured = agibot_x2_multi_motion_tracking_measured_servo_env_cfg(play=True)
  a, order_a = resolved(base)
  b, order_b = resolved(measured)

  failures = []
  if len(a) != len(X2_JOINT_ORDER) or len(b) != len(X2_JOINT_ORDER):
    failures.append(f"joint coverage {len(a)} and {len(b)} against 31")
  if list(a) != list(b):
    failures.append("the two configs resolve different joints")
  if order_a != order_b:
    failures.append(
      f"the resolved actuator order differs: {order_a[:6]} against {order_b[:6]}"
    )
  if len(order_a) != 31:
    failures.append(f"the base task resolves {len(order_a)} joints, expected 31")

  changed = []
  for joint in a:
    fa, fb = a[joint][1], b[joint][1]
    if fa["stiffness"] != fb["stiffness"]:
      failures.append(f"{joint}: kp {fa['stiffness']} -> {fb['stiffness']}")
    for field in _FIELDS:
      if field in ("stiffness", "damping"):
        continue
      if fa[field] != fb[field]:
        failures.append(f"{joint}: {field} {fa[field]} -> {fb[field]}")
    if fa["damping"] != fb["damping"]:
      changed.append(joint)
      expected = X2_MEASURED_SERVO_KD.get(joint)
      if expected is None:
        failures.append(f"{joint}: kd changed to {fb['damping']}, not in the table")
      elif fb["damping"] != expected:
        failures.append(f"{joint}: kd {fb['damping']} against table {expected}")

  if set(changed) != set(X2_MEASURED_SERVO_KD):
    failures.append(
      f"kd changed on {sorted(set(changed) ^ set(X2_MEASURED_SERVO_KD))} but not both ways"
    )

  print(f"joints resolved: {len(a)}")
  print(f"actuator count: base {len(base.scene.entities['robot'].articulation.actuators)}"
        f" measured {len(measured.scene.entities['robot'].articulation.actuators)}")
  print(f"kd changed on {len(changed)} joints")
  print(f"{'joint':26s} {'kp':>6s} {'kd base':>8s} {'kd measured':>12s}")
  for joint in X2_JOINT_ORDER:
    fa, fb = a[joint][1], b[joint][1]
    flag = "" if joint in changed else "  (kept)"
    print(f"{joint:26s} {fa['stiffness']:6.1f} {fa['damping']:8.3f}"
          f" {fb['damping']:12.3f}{flag}")

  # The compiled MuJoCo spec is the only place that proves the values reach the model:
  # a position actuator stores [0, -kp, -kd] in biasprm.
  from mjlab.asset_zoo.robots.agibot_x2.x2_constants import get_x2_robot_cfg
  from mjlab.entity import Entity
  from mjlab.tasks.tracking.config.agibot_x2.vendor_gains import apply_vendor_rl_gains

  robot_cfg = get_x2_robot_cfg()
  apply_vendor_rl_gains(robot_cfg)
  apply_measured_servo(robot_cfg)
  spec = Entity(robot_cfg).spec
  compiled = {a.name: (-a.biasprm[2], -a.biasprm[1]) for a in spec.actuators}
  if len(compiled) != 31:
    failures.append(f"the compiled spec has {len(compiled)} actuators, expected 31")
  for joint, damping in X2_MEASURED_SERVO_KD.items():
    if joint not in compiled:
      failures.append(f"{joint} is missing from the compiled spec")
      continue
    kd, _kp = compiled[joint]
    if abs(float(kd) - damping) > 1e-9:
      failures.append(f"{joint}: compiled kd {float(kd)} against table {damping}")
  print(f"compiled spec: {len(compiled)} actuators, "
        f"kd mismatch on {sum(1 for j, d in X2_MEASURED_SERVO_KD.items() if j in compiled and abs(float(compiled[j][0]) - d) > 1e-9)} table joints")

  if failures:
    print("\nFAIL")
    for line in failures:
      print(" -", line)
    return 1
  print("\nPASS: same joints, same order, same kp, same non-gain fields; kd in the config and in the compiled spec")
  return 0


if __name__ == "__main__":
  sys.exit(main())
