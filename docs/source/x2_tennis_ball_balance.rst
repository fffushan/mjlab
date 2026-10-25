X2 Tennis Ball on the Bat
=========================

Overview
--------

``Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Balance`` is a
variant of the X2 no-state-estimation flat velocity task with the tennis
equipment in the scene: the racket assembly ("bat") rigged to the right wrist
and a free tennis ball spawned on the string bed.

This page describes the task as shipped. **The ball is not observed and not
rewarded yet.** The task wires the assets into a trainable environment so that
the observation and reward work can start from a verified scene.

Because nothing about the interface changed, the actor width is still 102 and the
critic width 117, against 31 actions — the same as the base task. Reusing a base
checkpoint as an initial weight was not tested.

Servo gains
-----------

The task runs the **vendor reference-policy gains**, the same table that
``Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion`` uses. They are imported from
``tasks/tracking/config/agibot_x2/vendor_gains.py``, not copied, so the two
tasks cannot drift apart.

Only ``stiffness``/``damping`` differ between the velocity task and the tracking
task. Joint ranges, effort limits, armature, friction loss and action scale are
byte-identical (a test pins this). 27 of 31 joints change gain:

.. list-table:: Right-arm and waist gains
   :header-rows: 1

   * - group
     - velocity task kp / kd
     - this task (vendor) kp / kd
   * - shoulder pitch / roll / yaw
     - 30 / 50 / 20, kd 1
     - 50 / 50 / 50, kd 3
   * - elbow
     - 50 / 1
     - 50 / 3
   * - wrist yaw
     - 50 / 1
     - 20 / 2
   * - wrist pitch / roll
     - 20 / 1
     - 20 / 2
   * - waist pitch / roll
     - 48.3 / 3.08
     - 200 / 4
   * - waist yaw
     - 40 / 8
     - 40 / 4
   * - hip pitch
     - 100 / 4
     - 120 / 5
   * - hip roll / yaw
     - 100 / 3
     - 100 / 4
   * - ankle pitch
     - 40 / 3
     - 40 / 2
   * - ankle roll
     - 30 / 2
     - 40 / 2
   * - knee, head
     - unchanged
     - unchanged

The reason this matters here, measured on the compiled model in the bat-carry
pose:
``mj_rne`` gravity torque, then ``sag = tau / kp`` per joint, then the resulting
string-bed normal:

.. list-table:: Static behaviour under the bat's own weight
   :header-rows: 1

   * - quantity
     - velocity-task gains
     - vendor gains
   * - string-bed tilt, sum over shoulder, elbow and wrist
     - **19.1 deg**
     - **12.9 deg**
   * - shoulder pitch sag
     - 12.8 deg
     - 7.7 deg
   * - shoulder pitch damping ratio
     - 0.19
     - 0.44
   * - wrist pitch damping ratio
     - 0.56
     - 1.12
   * - wrist roll damping ratio
     - 1.55
     - 3.11

So the vendor table cuts the static sag by 32 % and brings the shoulder out of
the underdamped regime where it would ring after a ball impact. The static sag
is a constant offset the policy can learn to command away; the damping is not.

Scope
-----

**Verified** — asset fidelity, string-bed contact calibration, task contract,
and an environment build.

**Not verified** — ball retention under a trained policy. No test here shows a
ball staying on the bat.

**Not implemented** — ball observation, ball-holding reward, ball-fall
termination.

The robot cannot hold the bat-carry pose with open-loop PD, so retention can
only be shown by a trained policy. The reference asset snapshot documents the
same limitation for its own default pose.

Assets
------

``mjlab.asset_zoo.robots.agibot_x2.tennis_constants`` holds the assembly and the
ball. Meshes live in ``asset_zoo/robots/agibot_x2/tennis_meshes/`` and are
byte-identical to the ``x2_references/x2_tennis`` snapshot; their SHA-256 values
are pinned in ``TENNIS_MESH_SHA256`` and checked by a test.

Racket assembly (part of the robot entity)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The bat is rigid tooling, not a separate entity. The white wrist connector
replaces the right hand, and the clamp plus racket hang off
``right_wrist_roll_link`` as fixed bodies. Rigid attachment is modelled by the
kinematic tree, so the bat's 0.37 kg enters the arm's reflected inertia through
``mj_step`` with no weld equality and no soft constraint.

.. list-table:: Bat bodies
   :header-rows: 1

   * - Body
     - Parent
     - Mass (kg)
     - Role
   * - ``right_wrist_roll_link``
     - ``right_wrist_pitch_link``
     - 0.080
     - Connector; the hand's 0.303847 kg is removed
   * - ``racket_clamp``
     - ``right_wrist_roll_link``
     - 0.090
     - Clamp halves + bore
   * - ``tennis_racket``
     - ``racket_clamp``
     - 0.280
     - Racket + string bed + ``face_center`` site

The bat adds two bodies and **no joints and no actuators**: 31 actuators before
and after.

String bed
~~~~~~~~~~

The collision surface of the bat is a 32-sided, 3 mm thick elliptical plate at
local ``x = 0.515 m``, half-extents 0.157 m by 0.117 m. It is deliberately not
the convex hull of the racket mesh, which would be a solid slab.

Every other tennis geom is visual-only (``contype = 0``, ``conaffinity = 0``).

.. warning::

   The bed geom is named ``string_bed``, not something ending in
   ``_collision``. Two mechanisms would otherwise break it silently:

   * ``FULL_COLLISION`` sets ``disable_other_geoms=True``, which zeroes
     ``contype``/``conaffinity`` on every geom its patterns do not match. The
     bed does not match, so ``TENNIS_BED_COLLISION`` re-enables it afterwards.
     That config carries ``disable_other_geoms=False`` so it cannot undo the
     rest of the robot's collision story.
   * A ``*_collision`` name would also match the ``condim=1``, ``priority=0``
     catch-all in ``FULL_COLLISION`` and lose its priority, friction and solver
     parameters.

Ball
~~~~

The ball is a separate floating entity named ``ball``: one free joint, one
sphere geom, mass 0.057 kg, radius 33.5 mm, resting contact ``solref`` 0.004 /
0.35. The bed carries ``priority = 1``, so the bed's parameters — not the
ball's — govern every bed contact.

Physics timestep
----------------

The task runs at **1 ms with 20 substeps per control step** (50 Hz control),
instead of the base task's 5 ms with 4 substeps.

This is a contact-calibration requirement, not a tuning preference.
``BED_SOLREF = (-100000, -22)`` is MuJoCo's direct stiffness/damping form,
fitted to the user's drop measurements at a 1 ms step. The same numbers feed
energy into the contact at coarser steps:

.. list-table:: Rebound after a 0.30 m drop, ball bottom above the bed
   :header-rows: 1

   * - Physics step
     - Rebound
     - Measured target
   * - 1.0 ms
     - 0.247 m
     - 0.23 m
   * - 2.5 ms
     - 0.254 m
     - 0.23 m
   * - 5.0 ms
     - 1.187 m
     - 0.23 m

At 5 ms the response is not merely wrong, it is unstable. Do not raise
``TENNIS_PHYSICS_TIMESTEP``.

Measured cost: on one laptop RTX 5080 at 1024 environments, 4 PPO iterations,
the tennis task took 1.80 s per iteration against 0.61 s for the base task —
about 2.9 times. This is a single small-scale measurement, not a production
throughput figure. If that cost is unacceptable, the lever is the bed parameter
fit, not the substep count: re-fitting ``BED_SOLREF`` at a coarser step would
change the calibrated rebound.

Resets
------

The ball spawns on the bed, so both entities are pinned to the bat-carry pose:

* ``reset_base`` uses a zero pose range (no ``x``/``y``/``z``/``yaw`` sampling).
  A sampled yaw would rotate the bat and leave the ball behind; a sampled height
  would move the bed through the ball.
* ``reset_ball`` places the ball at ``BALL_SPAWN_LOCAL_POS`` with zero velocity.

The measured spawn point is the ``face_center`` site plus the face normal times
``radius + 1 mm``. ``measure_bat_carry_state`` recomputes it, and a test asserts
the compiled scene still matches the stored constants.

Push disturbances, encoder bias, base-COM offset, foot friction and the
per-episode PD-gain scale are the base velocity task's own terms.

Domain randomization
--------------------

The physics sim-to-real axes are the **multi-motion tracking task's**, not the
velocity task's. The velocity task randomizes only push, foot friction, encoder
bias, base COM and PD gains; the tracking family adds six more, and the ball task
runs all of them.

.. list-table:: Adopted sim-to-real axes
   :header-rows: 1

   * - event
     - range
     - acts on
   * - ``randomize_inertia``
     - mass 0.95–1.05x, COM ±10 mm
     - every body
   * - ``randomize_armature``
     - 0.95–1.05x
     - every joint
   * - ``randomize_effort_limits``
     - 0.95–1.0x (weaker only)
     - every actuator
   * - ``randomize_effort_limits_wrist``
     - 0.95–1.0x on top
     - wrist pitch/roll only
   * - ``randomize_joint_friction``
     - 0.95–1.05x
     - every joint
   * - ``randomize_joint_damping``
     - 0–0.03 N·m·s/rad, scaled per torque class
     - every joint
   * - ``randomize_foot_size``
     - 0.97–1.03x, one scale for both feet
     - the 12 sole spheres per foot
   * - ``randomize_pd_gains``
     - kp and kd 0.7–1.3x, one shared scale per environment
     - every actuator

Two details are load-bearing:

* ``randomize_effort_limits_wrist`` must run **after**
  ``randomize_effort_limits``. Both write the same actuator fields and only the
  last writer takes effect, so the ordering is asserted at config build time.
* ``randomize_pd_gains`` uses ``shared_gain_scale=True``: one scale per
  environment, not per joint. A per-joint scale lets every joint drift
  independently, which contradicts the measured kp/kd pairing.

The tracking family's seventh axis, **observation delay**, is deliberately not
adopted. It changes the actor's input timing contract, which is a separate
decision from randomizing physics.

The definitions are imported, not copied: the terms come from
``make_tracking_env_cfg()`` and are finished by
``apply_x2_tracking_robot_randomization``, which the tracking task also calls.
So ``MJLAB_DR_AXES`` still selects axes at build time, and an axis switched off
simply does not appear. A test pins every shared term against the tracking task.

Measured: with four environments, every adopted axis changes the compiled
model — armature, friction loss, damping, body mass, body inertia, actuator force
ranges and foot geom sizes all differ from the nominal values.

.. warning::

   This is a training-distribution change, not a physics fix. Two runs with the
   same seed will diverge once these axes are on, and a checkpoint trained
   before them was trained against a narrower distribution.

.. note::

   The carry pose is the reference's approved "raised prepare" keep-up pose, not
   ``STAND_KEYFRAME``: hip −0.15 / knee 0.30 / ankle −0.15 rad, pelvis at
   0.669841 m with the soles 2 mm above the ground. The bat is level in this
   pose (face normal within 1e-5 of world +z), so the ball has no slope to roll
   down.

Sensors
-------

``bat_ball_contact`` reports ``found`` and the net force between the racket body
and the ball. ``primary`` is the whole ``tennis_racket`` body, so the sensor
covers the bed and any future racket collision geom. No reward consumes it yet.

Running
-------

.. code-block:: bash

   uv run train \
     Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Balance \
     --env.scene.num-envs 4096

   uv run play Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Balance

The PPO experiment name is ``agibot_x2_velocity_tennis_ball_balance``.

Tests
-----

``tests/test_x2_tennis_ball_task.py`` covers:

* mesh provenance, bat masses and inertias, the hand replacement, and that the
  bat adds no joints or actuators;
* that only the string bed collides, and that the bed keeps its calibrated
  ``contype``, ``condim``, ``priority``, friction, ``solref`` and ``solimp``
  after the collision-config pipeline;
* that the carry-height and spawn constants are still what the current spec
  measures;
* that the ball spawns 1 mm above a horizontal bed, over the plate;
* the four calibrated rebounds at 1 ms with zero lateral drift, and the
  over-rebound at 5 ms;
* the task contract: task id, entities, timestep, substeps, sensor, unchanged
  action/observation/reward/termination/command term sets, pinned resets, and a
  real environment build;
* the servo configuration: kp/kd, joint ranges, effort limits, armature, friction
  loss and action scale all equal to ``Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion``,
  and the vendor gains do not leak into the other tasks;
* the six sim-to-real DR terms plus the wrist effort derating and the shared
  PD-gain scale, each compared term-for-term against the tracking task.

Next step
---------

Add the ball observation and the holding reward. Both change the actor input
width, so they need their own experiment name and a fresh normalizer set. The
ball position is assumed to be provided by an external perception program, so
the observation should be a relative position and velocity in a body frame, with
the noise of that estimate applied explicitly rather than as simulator ground
truth.
