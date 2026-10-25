X2 Tennis Ball Juggling
=======================

Overview
--------

``Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle`` is the
juggling variant of the tennis-ball task: the ball bounces on the string bed and
returns to an apex about 30 cm above the plate, repeatedly. It is the carrying task
(see :doc:`x2_tennis_ball_balance`) with a different ball reward set, a standstill
command band and no command curriculum. Scene, assets, 1 ms physics, observations,
resets, termination, servo gains and domain randomization are the carrying task's,
so the actor width is still 105 and the critic width 123. The config is a copy, not
a wrapper: the two files diff by intent.

Goal
----

Juggling, not carrying. The carrying task pays for a ball held still on the plate;
this task pays for a **bounce that comes back to a target apex**. Version 1 juggles
while standing still: the twist command is pinned to about ±0.10 m/s linear and
±0.05 rad/s yaw, with no command curriculum, since one band has no stage to reach.

Why the reward fires per bounce
-------------------------------

A per-step height reward cannot express a bounce: a ball parked at the target height
would collect it on every control step, and parking is much easier than bouncing.
``ball_bounce_apex`` therefore pays **once per bounce**, on the step where the ball's
world vertical speed crosses from positive to negative, and only if that apex clears
a 30 mm gap and the rise peaked above 0.30 m/s. Every other step of the bounce pays
nothing, so earning the term again requires hitting the ball again.

One payment per cycle is easy to miss and hard to steer, so the rest of the set
shapes the cycle: ``ball_centred`` keeps the ball over the plate, and
``ball_at_rest`` charges for a ball that has stayed low and slow for the grace
period, which rules out parking it.

Reward set
----------

The base task's own reward terms are unchanged. The ball terms are:

.. list-table:: Ball reward terms
   :header-rows: 1

   * - term
     - weight
     - purpose
   * - ``ball_centred``
     - 0.25
     - ball over the plate centre, plateau and Gaussian
   * - ``ball_bounce_apex``
     - 6.0
     - one payment per bounce, Gaussian in the apex gap
   * - ``ball_at_rest``
     - -0.10
     - parking penalty, after a grace period
   * - ``ball_in_fov``
     - 0.2
     - soft camera constraint, unchanged
   * - ``head_joint_motion``
     - -0.05
     - head velocity penalty, unchanged

``ball_height`` is removed. It is flat out to a 100 mm gap, so it pays for a ball
parked on the strings, which is the behaviour this task must not reward.

Centring is a quarter of the carrying task's 1.0 because the ball is in the air for
most of the cycle and cannot be steered there: over one 0.5 s bounce cycle at 50 Hz,
carrying the ball earns about 6 from centring alone, a juggle about 6.0 per apex plus
the same centring near the plate. **These weights are first guesses to be tuned
against the bounce count, not measurements.**

Why 30 cm
---------

At the carry pose the plate top surface sits at world z 1.1194. At a 30 cm gap the
ball centre is 0.30 m above it and the normalised image radius is **0.434** — 27.8
degrees of the 64.0 degree vertical half-frame — inside the 0.60 plateau, so the
field-of-view reward is still exactly **1.000**. At rest the same ball reads 0.271,
and the reward starts to fall above a gap of about 0.41 m. So a 30 cm apex costs
almost nothing in field of view, and it is a real bounce: 30 cm of rise needs a
launch of √(2 · 9.81 · 0.30) = 2.43 m/s. These values are measured on the built
environment at the pinned reset pose, with the camera mount from the tennis constants.

The ``JUGGLE_APEX_TARGET_M`` comment in ``tennis_constants.py`` records 0.889 for this
ratio. That figure is incorrect: 0.889 of the half-frame would be outside the 0.60
plateau and would score about 0.26, not 1.000. The correct figure is 0.434.

Acceptance metric
-----------------

``ball_bounce_count`` is registered as a metrics term with ``reduce="sum"``, so
``Episode_Metrics/ball_bounce_count`` reports **bounces per episode**. That is this
task's acceptance signal: a reward can be collected in unintended ways, a counted
bounce is behaviour. The metric reads the apex term's per-step flag, 1.0 on a counted
apex and 0.0 otherwise. ``RewardManager`` skips zero-weight terms, so the apex weight
must stay non-zero or the count freezes.

Tests
-----

``tests/test_x2_tennis_juggle_task.py`` pins the registration and a real build, the
reward diff against the carrying task term by term, the 105/123 observation widths,
the standstill band, the absent curriculum and the summed metric. Two scripted ball
sequences check behaviour: a ballistic rise and fall, where the apex term pays on
exactly one step while the rest term reads 0, and a parked ball, which earns no apex
payment and is charged after the grace period. The tests do not show that a policy can
juggle: they drive the reward terms with a scripted ball state.

Not implemented
---------------

* **Walking while juggling.** The command is a standstill band. A walking version needs
  a command curriculum, which this task deliberately does not have.
* **A juggle-height curriculum.** The 30 cm apex is a fixed target. There is no ramp
  from a lower apex, and nothing shifts the target during training.
* **Real-robot verification.** None. The target, the weights and the camera geometry
  are simulator-level and unverified on hardware.
