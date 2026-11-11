"""Calibrate the rolling-friction coefficient of a tennis ball on the string bed.

This script fits :data:`~mjlab.asset_zoo.robots.agibot_x2.tennis_constants.
ROLLING_FRICTION`, the third number of MuJoCo's geom friction triple. At
``condim=6`` that number is a rolling-resistance coefficient with length units;
it is not a dimensionless sliding coefficient, and MuJoCo does not derive it
from a normal force and a radius, so it has to be fitted in the simulator that
will use it.

The fixture is deliberately isolated: one horizontal 48-sided plate, 3 mm thick
and 0.3 m in half-extent, plus one sphere ball. The ball comes from
:func:`~mjlab.asset_zoo.robots.agibot_x2.tennis_constants.get_tennis_ball_spec`,
so its mass, measured shell inertia, ``condim``, friction and ``solref`` are the
shipped ones, and the plate takes its sliding, spin, ``solref``, ``solimp`` and
``priority`` from the same module. Only the rolling coefficient is swept. The
plate carries ``priority=1`` like the real string bed, so the plate's row is the
one that governs the contact; the script asserts that from the compiled contact.

The ball is released rolling without slipping: linear speed ``v`` along +x and
angular speed ``v / R`` about +y, which is the zero-slip relation for a contact
point at ``-R z``. A slip column reports ``|v - omega R|`` so the assumption can
be checked instead of trusted.

**The target coast distance is an assumption, not a measurement.** No source
gives a usable rolling-resistance coefficient for a conforming tennis ball at
these speeds (see ``docs/reports/x2_tennis_ball_bat_contact_models.md`` section
3.6). The default target, 0.06 m at a 0.10 m/s release, says "a ball nudged
across a level bed settles well before the rim". A ball's centre starts at the
plate centre, so it has only 0.0835 m of travel across the plate's short axis
before it passes the rim; 0.06 m keeps a 28% margin, and corresponds to an
equivalent dimensionless rolling resistance of about 0.013, inside the plausible
0.01-0.02 band for a ball on a compliant surface. The fitted coefficient
inherits the assumption: it is provisional until a roll-decay measurement of the
real ball on the real racket replaces it.

The coast distance moves by up to 12% with the release convention alone, so a
re-fit must use this script rather than a freshly written fixture.

Usage::

    uv run python scripts/experiments/x2_tennis_roll_friction_calibrate.py \\
        --target-coast-m 0.06 --release-speed 0.10

Omit ``--target-coast-m`` to run the sweep only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.agibot_x2.tennis_constants import (
  BALL_RADIUS,
  BED_HALF_THICKNESS,
  BED_PRIORITY,
  BED_SOLIMP,
  bed_solref,
  CONTACT_CONDIM,
  ROLLING_FRICTION,
  SLIDING_FRICTION,
  SPIN_FRICTION,
  get_tennis_ball_spec,
)
from mjlab.tasks.velocity.config.agibot_x2.tennis_ball_env_cfg import (
  TENNIS_PHYSICS_TIMESTEP,
)

PLATE_HALF_EXTENT = 0.3
"""Half-extent of the square plate, in metres. Wide enough for a 0.15 m coast,
which is longer than the real plate allows: this fixture measures the coefficient,
not whether the ball stays on a racket."""

PLATE_SEGMENTS = 48

DEFAULT_ROLLING_SWEEP = (0.0, 1e-4, 3e-4, ROLLING_FRICTION, 1e-3, 3e-3)
"""Cover the reviewer's measured range, plus the coefficient this file ships.

The shipped row is included so a plain run also reports the coast and
acceleration the current :data:`ROLLING_FRICTION` produces.
"""

SETTLE_TIME = 0.1
"""Seconds simulated at rest before release, to load the resting contact."""

WINDOW = (0.05, 0.45)
"""Deceleration is measured between these times after release, in seconds.

Not from t = 0: the normal force needs a few milliseconds to build up, and a
one-step difference would mix that transient into the deceleration.
"""

STOP_SPEED = 1e-3
"""A roll counts as finished below this speed, in m/s."""

MJ_MIN_MU = 1e-5
"""MuJoCo's floor on every contact friction coefficient (``mjMINMU``).

A zero rolling coefficient therefore reaches the contact as 1e-5, which is why
``friction[2] = 0`` is nearly but not exactly frictionless.
"""

MAX_TIME = 20.0
"""Give up after this long, so a slow tail cannot hang the script."""


@dataclass(frozen=True)
class Roll:
  """One release-and-coast run at a fixed rolling coefficient."""

  rolling_friction: float
  decel: float
  coast: float
  slip: float
  stopped: bool
  off_plate: bool


def build_spec(rolling_friction: float) -> mujoco.MjSpec:
  """The isolated fixture: one plate plus the shipped ball.

  Solver options are the ones the tennis task configures (implicitfast, Newton,
  pyramidal cone, the same tolerances), because the coefficient is fitted for
  that solver and that 1 ms step.
  """
  spec = get_tennis_ball_spec()
  spec.option.timestep = TENNIS_PHYSICS_TIMESTEP
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
  spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
  spec.option.iterations = 100
  spec.option.tolerance = 1e-8
  spec.option.ls_iterations = 50
  spec.option.ls_tolerance = 0.01

  angle = np.linspace(0.0, 2.0 * np.pi, PLATE_SEGMENTS, endpoint=False)
  vertices = np.concatenate(
    [
      np.c_[
        PLATE_HALF_EXTENT * np.cos(angle),
        PLATE_HALF_EXTENT * np.sin(angle),
        np.full(PLATE_SEGMENTS, z),
      ]
      for z in (-BED_HALF_THICKNESS, BED_HALF_THICKNESS)
    ]
  )
  spec.add_mesh(name="plate_mesh", uservert=vertices.ravel().tolist())
  plate = spec.worldbody.add_body(name="plate")
  plate.add_geom(
    name="plate",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="plate_mesh",
    contype=1,
    conaffinity=1,
    condim=CONTACT_CONDIM,
    priority=BED_PRIORITY,
    mass=0.0,
    friction=[SLIDING_FRICTION, SPIN_FRICTION, rolling_friction],
    solref=list(bed_solref()),
    solimp=list(BED_SOLIMP),
  )
  return spec


def roll_once(rolling_friction: float, speed: float) -> Roll:
  """Release the ball rolling without slipping and follow it until it stops."""
  model = build_spec(rolling_friction).compile()
  data = mujoco.MjData(model)
  ball = model.joint("ball_free")
  qadr = int(np.ravel(ball.qposadr)[0])
  vadr = int(np.ravel(ball.dofadr)[0])

  # Rest on the plate, then let the compliant contact take the ball's weight.
  data.qpos[qadr + 2] = BED_HALF_THICKNESS + BALL_RADIUS
  mujoco.mj_forward(model, data)
  for _ in range(round(SETTLE_TIME / model.opt.timestep)):
    mujoco.mj_step(model, data)
  if data.ncon == 0:
    raise RuntimeError("the ball never settled on the plate")
  contact = data.contact[0]
  if contact.dim != CONTACT_CONDIM:
    raise RuntimeError(f"contact dimension is {contact.dim}, expected {CONTACT_CONDIM}")
  # ``mjContact.friction`` holds five numbers: the sliding coefficient twice,
  # the torsional one, and the rolling one twice (one per tangent axis). The
  # geom triple expands into that, so this also pins the slot order.
  expected_friction = [
    SLIDING_FRICTION,
    SLIDING_FRICTION,
    SPIN_FRICTION,
    max(rolling_friction, MJ_MIN_MU),
    max(rolling_friction, MJ_MIN_MU),
  ]
  if not np.allclose(contact.friction, expected_friction):
    raise RuntimeError(
      f"contact friction {contact.friction} is not the plate's row; "
      "priority is not governing this pair"
    )

  # Rolling without slipping: v = omega R about +y for travel along +x, because
  # the contact point sits at -R z from the centre.
  data.qvel[vadr + 0] = speed
  data.qvel[vadr + 4] = speed / BALL_RADIUS
  mujoco.mj_forward(model, data)

  x0 = float(data.qpos[qadr])
  step = model.opt.timestep
  samples: list[tuple[float, float, float]] = []  # (t, v_x, slip)
  x = x0
  stopped = False
  off_plate = False
  travel = 0.0
  for step_index in range(round(MAX_TIME / step) + 1):
    t = step_index * step
    v = float(data.qvel[vadr + 0])
    omega = float(data.qvel[vadr + 4])
    samples.append((t, v, abs(v - omega * BALL_RADIUS)))
    x = float(data.qpos[qadr])
    travel = x - x0
    if v < STOP_SPEED:
      stopped = True
      break
    if travel > PLATE_HALF_EXTENT - BALL_RADIUS:
      off_plate = True
      break
    mujoco.mj_step(model, data)

  # Average deceleration over the measurement window, from the samples that
  # exist. Using the samples' own times keeps the divisor honest if a large
  # coefficient stops the ball before the window closes.
  start, end = WINDOW
  i0 = min(range(len(samples)), key=lambda i: abs(samples[i][0] - start))
  i1 = min(range(len(samples)), key=lambda i: abs(samples[i][0] - end))
  if i1 <= i0:
    raise RuntimeError("the ball stopped before the measurement window opened")
  decel = (samples[i0][1] - samples[i1][1]) / (samples[i1][0] - samples[i0][0])
  slip = max(sample[2] for sample in samples[i0 : i1 + 1])
  return Roll(
    rolling_friction=rolling_friction,
    decel=decel,
    coast=travel,
    slip=slip,
    stopped=stopped,
    off_plate=off_plate,
  )


def report(roll: Roll) -> None:
  note = "stopped" if roll.stopped else f"hit {MAX_TIME:.0f} s"
  if roll.off_plate:
    note = "left the plate"
  print(
    f"  {roll.rolling_friction:9.2e}  {roll.decel:10.5f}  {roll.coast:8.4f}  "
    f"{roll.slip:8.2e}  {note}"
  )


def bisect(target: float, speed: float, tolerance: float = 1e-4) -> Roll:
  """Rolling coefficient whose coast distance is ``target``, by bisection.

  The coast distance falls monotonically as the rolling coefficient rises, so
  bisection needs a bracket, not a derivative. A run that reaches the plate
  edge is still usable as a lower bracket: its true coast is longer than the
  distance it reports.
  """
  low, high = 0.0, 1e-4
  low_roll, high_roll = roll_once(low, speed), roll_once(high, speed)
  while high_roll.coast > target:
    low, low_roll = high, high_roll
    high *= 10.0
    if high > 1.0:
      raise SystemExit(f"no rolling coefficient reaches a {target:.3f} m coast")
    high_roll = roll_once(high, speed)
  print("bracket:")
  report(low_roll)
  report(high_roll)
  print("iterations:")
  for _ in range(60):
    mid = 0.5 * (low + high)
    roll = roll_once(mid, speed)
    coast = f"coast {roll.coast:.5f} m"
    if roll.off_plate:
      coast = f"coast > {roll.coast:.4f} m (left the plate)"
    print(f"  {mid:9.4e} -> {coast}, decel {roll.decel:8.5f} m/s^2")
    if roll.coast > target:
      low = mid
      continue
    if roll.off_plate:
      raise SystemExit("target is beyond the plate; widen PLATE_HALF_EXTENT")
    if abs(roll.coast - target) < tolerance:
      return roll
    high = mid
  raise SystemExit("bisection did not converge")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--rolling",
    type=float,
    nargs="+",
    default=list(DEFAULT_ROLLING_SWEEP),
    help="rolling-friction coefficients to sweep (friction[2], length units)",
  )
  parser.add_argument(
    "--release-speed",
    type=float,
    default=0.10,
    help="initial rolling speed in m/s (default 0.10)",
  )
  parser.add_argument(
    "--target-coast-m",
    type=float,
    default=None,
    help="ASSUMED coast distance to invert by bisection; not a measurement",
  )
  args = parser.parse_args()

  print(
    f"fixture: {PLATE_SEGMENTS}-sided plate, "
    f"{2e3 * BED_HALF_THICKNESS:.1f} mm thick, "
    f"{PLATE_HALF_EXTENT:.2f} m half-extents, condim {CONTACT_CONDIM}, "
    f"priority {BED_PRIORITY}"
  )
  print(
    f"from tennis_constants: sliding {SLIDING_FRICTION}, spin {SPIN_FRICTION}, "
    f"rolling {ROLLING_FRICTION:.2e}, bed solref {bed_solref()}, "
    f"timestep {TENNIS_PHYSICS_TIMESTEP * 1e3:.1f} ms"
  )
  print(
    f"release {args.release_speed:.3f} m/s rolling without slipping; "
    f"deceleration window {WINDOW[0]:.2f}-{WINDOW[1]:.2f} s"
  )
  print("  rolling    decel(m/s^2)   coast(m)      slip(m/s)  note")
  for rolling in args.rolling:
    report(roll_once(rolling, args.release_speed))

  if args.target_coast_m is None:
    return

  print(
    f"\nASSUMED target: a ball released at {args.release_speed:.2f} m/s on a "
    f"level bed coasts {args.target_coast_m:.3f} m. This is a design choice, "
    "not a measurement."
  )
  fitted = bisect(args.target_coast_m, args.release_speed)
  print(
    f"\nROLLING_FRICTION = {fitted.rolling_friction:.4e}  "
    f"# coast {fitted.coast:.4f} m at {args.release_speed:.2f} m/s, "
    f"decel {fitted.decel:.5f} m/s^2 (provisional)"
  )


if __name__ == "__main__":
  main()
