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

**The plate fills the string area exactly.** Measured on the compiled model, the
racket mesh occupies local x 0.000–0.690 m, y ±0.135 m, z ±0.0145 m, and the
frame ring leaves an inner string area of 310 × 230 mm. The plate is 314 × 234 mm,
so it overlaps the inner edge of the frame by about 2 mm per side and no more.

**The simulated racket is not the operator's racket, by decision.**

.. list-table:: Racket geometry, simulated versus measured
   :header-rows: 1

   * - dimension
     - simulated (vendor mesh)
     - operator's real racket
   * - hoop, outer long axis
     - 320 mm
     - **358 mm**
   * - string area, long axis
     - 310 mm
     - **332 mm**
   * - string area, short axis
     - 230 mm
     - **255 mm**
   * - frame height above the strings
     - 14.5 mm
     - not measured

The plate cannot simply be widened to the real 332 × 255 mm: it would protrude
through the simulated frame by 8–10 mm per side and the ball would visibly roll on
nothing. Matching the real racket would need the whole mesh scaled by about 8 %,
which changes the bat's mass properties, the carry pose and the spawn point.

**Decision (2026-10-25): keep the vendor racket.** The operator chose the
original model and accepted the 8 % size difference rather than rescale the bat.
The size difference is small next to the behavioural gap below, and rescaling
would invalidate the measured carry pose for no gain in task fidelity.

**The simulated racket has no rim, also by decision.** The real frame stands
14.5 mm above the strings, and the operator's slope video shows the ball rolling
into that frame at the end of the measured travel and bouncing back onto the bed.
In simulation the plate is the only collidable part of the bat, so the ball leaves
the plate edge and falls with no bounce.

The two failure modes therefore differ in kind: on hardware the frame usually
keeps a ball that reaches the edge, while in simulation reaching the edge loses
it. The chosen mitigation is a **reward that keeps the ball near the plate
centre**, not a rim collision. A centring reward makes the rim's absence
irrelevant, because a policy that holds the ball near the middle never reaches the
edge where the two models disagree. Rim collision would additionally make the
task easier and would need its own asset work, so it stays out.

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
sphere geom, mass 0.0560 kg, radius 32.7 mm, resting contact ``solref`` 0.004 /
0.35. The bed carries ``priority = 1``, so the bed's parameters — not the
ball's — govern every bed contact.

Head camera and head steadying
------------------------------

The ball position on hardware comes from the **head stereo pair**, so a ball outside
the camera's frame is a lost ball however well it is balanced. Two reward terms
cover that, and both are deliberately coarse: perception and ball estimation are a
separate process, and the camera calibration will be refined later.

``ball_in_fov`` (weight **+0.2**) pays for the ball being inside the frame. Each
step the ball's direction is taken in the head camera frame and converted to a
normalised image radius ``s``; full credit inside ``s = 0.6``, Gaussian falloff with
``sigma = 0.25``, zero at the frame edge and zero for anything behind the camera
plane.

.. list-table:: Measured response
   :header-rows: 1

   * - ball moved from the carry pose
     - ``ball_in_fov``
   * - 0 (as carried)
     - 1.000
   * - 0.6 m up
     - 0.514
   * - 0.9 m up
     - 0.14
   * - 1.0 m up
     - 0.040
   * - 0.6 m behind the head
     - 0.000
   * - 1.0 m to the side
     - 1.000

That last row is the honest summary: the stereo pair is a **fisheye**, so the ball
stays inside the frame through a metre of lateral travel and this term rarely bites.
It is a safety term, not a shaping term. If you want the ball kept near the image
*centre* instead — for better pixel accuracy, because fisheye distortion grows
towards the edge — that is a different objective and needs a much smaller plateau,
around 0.3.

The geometry is approximate by intent:

* the mount is the reference snapshot's ``camera_mount_162.json``, which its README
  warns is robot 162's vendor transform rather than a calibration of another unit;
* the mount produces a **ROS optical frame**, ``+z`` along the optical axis with
  ``+x`` right and ``+y`` down, while MuJoCo's own camera convention is ``-z``
  forward. Using the MuJoCo sign would put the ball behind the camera;
* the half-angles come from the live stereo-left calibration under the
  **equidistant** model, so an image edge lies at ``r/f`` rather than
  ``arctan(r/f)``: 85.4° horizontally and 64.0° vertically.

``head_joint_motion`` (weight **−0.05**) penalises the squared velocity of
``head_yaw_joint`` and ``head_pitch_joint``. Without it the field-of-view reward pays
the policy to hunt with the head, chasing a ball that perception measures to a few
millimetres; that costs actuator travel and smears the very frames perception needs.
It is a velocity penalty, not a position penalty, so the head can still be aimed
deliberately and then held.

Both head joints are part of the policy's 31 actions. The reference instead runs a
29-action body policy with an **independent head servo** tracking the measured ball;
this task keeps the head under the policy, which is why it needs both of these terms.

Training curriculum
-------------------

The base velocity task asks for up to 1 m/s in any direction from iteration 0, and
its own curriculum widens those ranges on a step count. Balancing a ball and walking
are separate skills, so this task widens the commands in its own stages and starts
at 0.3 m/s.

.. list-table:: Command stages
   :header-rows: 1

   * - earliest step
     - lin_vel_x
     - lin_vel_y
     - yaw rate
     - intent
   * - 0
     - ±0.3 m/s
     - ±0.3 m/s
     - ±0.15 rad/s
     - creep
   * - 48 000 (iteration 2000)
     - ±0.6 m/s
     - ±0.6 m/s
     - ±0.3 rad/s
     - walk
   * - 120 000 (iteration 5000)
     - ±1.0 m/s
     - ±1.0 m/s
     - ±0.5 rad/s
     - the base task's own stage 0

**The difficulty advances on a step count, and performance can only withhold an
advance.** That is the outcome of three attempts on this task, and the reasons
matter more than the design:

1. *A performance gate with a stand-still first stage.* Rejected. With the robot
   commanded to stand, the ball simply sits on the plate, nothing terminates for the
   whole 20 s episode, and the gate read a full-length episode as mastery. In a
   4096-environment run it promoted to the hardest stage in a single call.
2. *The same gate without the stand-still stage.* Rejected one level down. An
   untrained policy already keeps the ball for 9.1 s at 0.3 m/s and 15.7 s at
   0.6 m/s, and 1.05 s at 1.0 m/s. **Every survivability observable saturates in the
   easy regime**, so no absolute threshold placed there can separate "doing nothing"
   from "doing it well". It again reached 1.0 m/s by iteration 2.
3. *A schedule carrying the difficulty, with a mastery floor as a veto.* Adopted.
   A stage is entered on its scheduled step **and** only if the typical episode at
   the current stage is at least 8 s. Measured, a failing policy runs ~1.05 s
   episodes and a comfortable one several seconds, so the floor means "the ball is
   not being lost immediately". Without it the clock would push a failing policy
   faster.

The floor is a permission check, never a trigger, so the progression is predictable
and the run cannot be gated by a saturated measurement. Advancement is one stage per
call and one-way. The stage, the typical episode, the floor and the next stage's
scheduled step are all logged.

The final stage is the base task's stage 0. Its later widening to 1.5–3.0 m/s is a
locomotion-only objective and is deliberately not adopted. In play mode the final
stage is written directly and the ramp is not registered, because an evaluation
should not start at a crawl.

**Losing the ball does not end the episode immediately.** The ball counts as lost
when its centre is at or below **0.5 m above the ground**, and the episode only ends
once it has stayed there for 25 control steps (0.5 s). The counter is consecutive
and per-environment, is zeroed on every reset, and restarts if the ball comes back
up, so a recovering policy can continue. This is patience, not a
relaxed threshold: an untrained policy loses the ball in about 0.55 s, and without
the grace every episode would end there.

Ball observation, reward and termination
---------------------------------------

The policy has to hold the ball near the middle of the plate while the robot
walks. Three things make that a well-posed problem: the ball must be observable,
it must be paid for, and losing it must end the episode.

**Observation.** A 3-vector position is appended to both groups and a 3-vector
velocity to the critic alone, taking the actor from 102 to 105 and the critic from
117 to 123:

.. list-table:: Ball observation terms
   :header-rows: 1

   * - term
     - content
     - frame
     - actor randomization (half-widths)
   * - ``ball_pos``
     - ball centre relative to the plate's top-surface centre
     - plate
     - ±5, ±5, ±8 mm along x, y, z
   * - ``ball_vel``
     - ball velocity relative to the plate
     - plate
     - none: critic only, uncorrupted

The plate frame is the racket body frame translated to ``FACE_SITE_POS``: x along
the shaft, y across it, z along the plate normal. A ball resting on the strings
therefore reads ``ball_pos = (0, 0, R)``, and a ball carried along by the bat reads
``ball_vel = 0``.

Velocity is measured against the **material point of the plate at the ball's
centre**, not against the racket body origin. The face centre is 0.515 m from that
origin, so a 2 rad/s wrist turn moves it about 1 m/s faster than the origin;
subtracting only the origin's velocity would report a perfectly still ball as
moving whenever the wrist turns. ``tests/test_x2_tennis_ball_task.py`` checks the
``v + ω × r`` form against MuJoCo's own ``site_vel_w`` to 1e-5.

**The three position dimensions are randomized independently.** Every other
observation term in the velocity task uses ``Unoise``, a symmetric uniform
distribution, and this term follows the same idiom: each of the three numbers above
is a half-width, so the standard deviation along that axis is the half-width over
√3. They are unequal because the errors are unequal — a camera measures the two
lateral axes better than the depth axis.

Both terms are tunable without code changes, like every other noise term:

.. code-block:: bash

   uv run python -m mjlab.scripts.train <task> \
     --env.observations.actor.terms.ball-pos.noise.n-min '(-0.01,-0.01,-0.02)' \
     --env.observations.actor.terms.ball-pos.noise.n-max '(0.01,0.01,0.02)'

Each term also has ``clip``, ``scale``, ``delay-min-lag``/``delay-max-lag`` and
``history_length`` available through the same path.

These magnitudes are the **perception budget, not physics**, and they are
placeholders for the real program's error: they must be replaced by measured
accuracy. They are also the reason the centring reward has a plateau — ±5 mm of
position error on a plain Gaussian reward shows up as reward noise exactly where
the ball spends most of its time. The critic sees the same two terms uncorrupted,
like the rest of its privileged set.

The actor has no ball-velocity term: on the real robot the only available velocity
would be a finite difference of the 10 Hz position, which is unusable. It must
therefore infer motion from ``ball_pos`` changing between the frames it sees. The
critic keeps the velocity as privileged information.

The actor's ``ball_pos`` is **delayed**, because the head stereo pipeline publishes
at 10 Hz and a published position is 110 ms old at the median and 141 ms at P95.
One frame covers five 50 Hz control steps, so the actor's lag is sampled from 5 to
9 steps and resampled once per sensor period; the timing comes from the measured
pipeline, not from tuning. The critic's ball terms stay undelayed because the critic
is privileged. Delay changes timing only: the actor stays at 105 and the critic at
123.

The actor can also see a **stale** position, because the detector rejects frames
and the stream can break: it publishes nothing when it finds no ball or several
candidates, cannot fit a consistent depth, or loses the stream. The actor's
position term therefore holds the last valid value for 100-300 ms when a frame is
rejected. The rejection rate is a placeholder (``BALL_DROPOUT_PROB``, 2 % per
control step): the reference implementation was only measured on a clean static
scene (293/293 frames valid), so nothing is known about rejection under motion or
occlusion. Holding changes neither the observation frame nor its width.

**Reward.** Two terms, added to the base task's thirteen:

.. math:: r_\text{centre} = \exp\left(-\left(\frac{\max(0, d - d_0)}{\sigma}\right)^2\right),
          \quad r_\text{height} = \exp\left(-\left(\frac{\max(0, h - h_0)}{\sigma_h}\right)^2\right)

with ``d`` the ball's horizontal distance from the plate centre, ``h`` its height
above the string plane, ``d₀ = 20 mm``, ``σ = 40 mm``, ``h₀ = 100 mm`` and
``σ_h = 40 mm``. Weights are 1.0 and 0.1.

.. list-table:: Centring reward
   :header-rows: 1

   * - d
     - reward
   * - 0–20 mm
     - 1.000
   * - 30 mm
     - 0.939
   * - 45 mm (one coast distance)
     - 0.677
   * - 84 mm (plate edge, short axis)
     - 0.077
   * - 124 mm (plate edge, long axis)
     - 0.001

The plateau is the point of the shape. With ±5 mm of position error, a plain
Gaussian has a slope exactly where the ball spends most of its time, so
measurement error becomes reward noise; inside the plateau the gradient is zero
and noise cannot reach the reward. A hill-shaped reward would also ask for
centimetre precision that the perception cannot resolve.

The height term exists only to remove one loophole: without it, throwing the ball
straight up and staying centred would pay as well as holding it still. It is
deliberately small, and it is flat from the string plane to 100 mm above it, so
the ball sinking into the compliant bed costs nothing.

**Termination.** ``ball_dropped`` ends the episode when the ball's centre is at or
below 0.5 m above the ground. The test is the ball's **absolute** world height and
does not involve the plate: a plate-frame test is defeated by tilting the racket,
because "below the plate" rotates with it, and a ball left on the floor beside a
near-vertical racket stays on the positive side of that plane forever. A carried
ball sits at about 1.15 m, so the margin is 0.65 m; a ball on the floor rests at
0.03 m. The term requires 0.5 s below the threshold, so a ball that dips under it
during a strike is not lost.

**A new experiment name.** The observation width changed, so the observation
normalizer has a different width and runs made before the ball was observed cannot
be resumed. The task therefore trains under
``agibot_x2_velocity_tennis_ball_centre_pos_only`` rather than the previous
``..._balance``.

**Untested: whether it trains.** At the shipped stage 0 the robot is commanded to
stand still, and an untrained policy keeps the ball for about 0.55 s and then loses
it; with the 0.5 s drop grace an episode runs about 1.05 s against a 20 s
allowance. The curriculum's first bar is a 6 s typical episode, so the policy must
first learn to level the racket while standing. Whether PPO reaches that bar is a
training question that no test here answers. Run 15 iterations of a 64-environment
smoke test: the stage stayed at 0 and the typical episode settled at 1.05 s.

Contact model
-------------

The bed–ball contact has five layers plus the ball's own spin inertia. Only the
tangential layers changed when the contact went from three to six dimensions;
the values live in ``tennis_constants.py`` and the status column is evidence
status, not confidence.

.. list-table:: Bed–ball contact layers
   :header-rows: 1

   * - layer
     - value
     - set by
     - status
   * - spring (normal)
     - stiffness 100000 N/m, damping 22 N·s/m
     - ``BED_SOLREF = (-100000.0, -22.0)``, MuJoCo's direct form
     - **fitted** to the four recorded drop rebounds
   * - cushion (normal)
     - ``(0.95, 0.95, 0.001, 0.5, 2.0)``
     - ``BED_SOLIMP``
     - **fitted** in the same four-point fit
   * - sliding friction
     - 0.44
     - ``SLIDING_FRICTION``, ``friction[0]``
     - **measured**: 0.43 ± 0.02 for a tennis ball on a string bed
   * - spin friction
     - 0.005
     - ``SPIN_FRICTION``, ``friction[1]``
     - **assumed**, and inert below ``condim=4``
   * - rolling friction
     - 1.7185e-03
     - ``ROLLING_FRICTION``, ``friction[2]``
     - **fitted** to a measured slope run
   * - ball spin inertia
     - 3.287e-05 kg·m², so I/(mR²) = 0.549
     - ``BALL_DIAGINERTIA``, explicit ball inertia
     - **estimated**: shape ratio measured, values at the ITF lower limit

``CONTACT_CONDIM = 6`` is what makes the last three rows act. ``condim=3``
carries a normal direction and two sliding directions and **no rotation of any
kind**, so with it the bed could not exert a torque on the ball at all and a
ball rolling on the level bed kept exactly the same speed forever. Six
dimensions add torsional friction about the contact normal and rolling friction
about the two tangent axes. MuJoCo geom friction is exactly three numbers, in
the order (sliding, spin, rolling), and the bed and the ball declare the same
triple; the bed's ``priority = 1`` is what makes its row the one that acts.

Three statements matter more than the numbers:

* **The normal direction is unchanged on purpose.** ``BED_SOLREF`` and
  ``BED_SOLIMP`` are untouched, so the four calibrated rebounds still reproduce.
  Re-measured through the test fixture, the simulated coefficient of restitution
  moves by less than 0.003 at every drop height. Making the contact stiffer and
  shorter — option O3 in
  ``docs/reports/x2_tennis_ball_bat_contact_models.md`` — needs a new four-point
  fit and is deliberately not part of this change.
* **The rolling coefficient is fitted to a measurement, through a coast
  distance.** MuJoCo's coefficient is not a dimensionless rolling-resistance
  ratio, so no literature value converts into it. The chain is: a slope run of
  the real ball on the real racket, θ = 0.129 rad, 0.245 m travelled in 0.834 s
  from rest, gives a = 0.705 m/s² and therefore

  .. math:: \mu_r = \frac{\sin\theta - (1+k)\,a/g}{\cos\theta} = 0.0174

  with k = I/(mR²) = 0.549 for this ball. On a level bed that same ball
  decelerates at μ_r g/(1+k) = 0.110 m/s², so a 0.10 m/s release coasts 0.045 m,
  which is 54 % of the ball's usable travel across the plate. That number, not
  μ_r, is what the calibration script inverts: 0.045 m gives
  ``friction[2] = 1.7185e-03``. The measured coast is shorter than the 0.06 m
  placeholder it replaced, so the real racket is grippier than the guess.
* **The dominant uncertainty is the ball's moment of inertia, not the
  measurement.** k enters the formula and does not cancel, so the same slope run
  implies a different coast for each plausible ball: 0.025 m for a uniform solid
  sphere (k = 0.400), 0.045 m for Cross's tennis-ball ratio (k = 0.549), 0.059 m
  for a 3.5 mm wall (k = 0.600) and 0.096 m for an infinitely thin shell
  (k = 0.667). **Every one of those reproduces the measured 0.705 m/s² equally
  well**, so the slope run cannot choose between them. The shipped 0.549 is the
  only value backed by a measurement of a real tennis ball.
* **One known bias in the slope run.** The ball rolls into the raised racket
  frame at the end of the measured travel, and the last frames show it already
  deflected, so ``a = 2L/t²`` slightly over-states the free acceleration. Fitting
  the per-frame positions over the same span gives 0.73 m/s², i.e. a coast a few
  percent above 0.045 m. The coefficient is therefore good to about ±7 % from
  this cause, and a re-run with a longer unobstructed travel would remove it.
* **The tangential change is what makes a rolling ball slow down.** In the
  isolated level-bed fixture a ball released rolling without slipping at
  0.10 m/s keeps 0.56 of its speed after one second with the shipped
  coefficient, and 1.00 of it with the rolling coefficient forced to zero. That
  control is a test: without it, the slowdown could be attributed to the solver
  rather than to the coefficient.

The ball's inertia is ``3.287e-05 kg·m²`` on all three axes, which is
``I/(m R²) = 0.549``. That ratio is Cross's measurement for a tennis ball, and
it is a **shape** property, so it is carried over unchanged while the absolute
value is re-derived at the shipped mass and radius. A uniform solid sphere would
be 0.400, and MuJoCo's default sphere inertia is exactly that, so the body
carries its inertia explicitly and its geom carries zero mass.

The ball itself is the **ITF lower limit** of the Type 2 regulation ball,
65.4 mm diameter and 56.0 g, not a measurement of the ball in hand: the operator
measured roughly 65 mm with a ruler and has no caliper, scale or inertia rig. The
real ball is somewhere in 65.4–68.6 mm and 56.0–59.4 g. This costs nothing in the
normal direction, because MuJoCo's direct ``solref``/``solimp`` form gives a
mass-normalised contact response: measured, the four calibrated rebounds are
identical for 40 g and 57 g at either radius. It costs something in the rolling
fit, which depends on ``I/(m R²)`` and cannot be separated from the friction
coefficient by the slope run alone. Rolling friction acts through the spin degree of freedom, so
the two changes are not independent: measured in the same fixture, the same
coefficient with the solid-sphere inertia coasts 0.137 m instead of 0.150 m.

Physics timestep
----------------

The task runs at **1 ms with 20 substeps per control step** (50 Hz control),
instead of the base task's 5 ms with 4 substeps.

This is a contact-calibration requirement, not a tuning preference.
``BED_SOLREF = (-100000, -22)`` is MuJoCo's direct stiffness/damping form,
fitted to the user's drop measurements at a 1 ms step. The same numbers feed
energy into the contact at coarser steps:

.. list-table:: Rebound height after the four recorded drops, by physics step
   :header-rows: 1

   * - physics step
     - 0.20 m
     - 0.30 m
     - 0.40 m
     - 0.50 m
     - worst error
   * - measured
     - 0.170 m
     - 0.230 m
     - 0.310 m
     - 0.420 m
     - —
   * - 1.0 ms
     - 0.159 m
     - 0.237 m
     - 0.335 m
     - 0.394 m
     - 26 mm
   * - 2.0 ms
     - 0.149 m
     - 0.257 m
     - 0.314 m
     - 0.433 m
     - 27 mm
   * - 2.5 ms
     - 0.154 m
     - 0.213 m
     - 0.321 m
     - 0.413 m
     - 17 mm
   * - 3.5 ms
     - 0.198 m
     - 0.314 m
     - 0.482 m
     - 0.381 m
     - 172 mm
   * - 5.0 ms
     - 0.696 m
     - 0.635 m
     - 0.250 m
     - 0.324 m
     - 526 mm

All rows use the shipped contact parameters, a 32-sided bed, and the task's own
solver settings. The 3.5 ms and 5 ms rows are unusable: a 0.20 m drop rebounds
0.70 m, which is energy from the solver, not from the ball.

**What this table does and does not justify.** It rules out 3.5 ms and above. It
does **not** select 1 ms over 2.0 or 2.5 ms: on these four points 2.5 ms fits
better (17 mm) than 1 ms (26 mm). 1 ms is kept for three other reasons, and the
first two are the deciding ones:

* it is the step the reference asset was calibrated at, so the fitted
  ``solref`` keeps its original meaning;
* it leaves margin above the 3.5 ms failure boundary. The four-point fit is a
  weak criterion — its own tolerance admits any restitution between 0.84 and
  1.00 — and the domain randomization changes armature, damping and effort
  limits, which shifts the effective contact. A step sitting 1 ms from the
  boundary is not where a 30 000-iteration run should sit;
* the fit itself cannot tell the difference, so fidelity buys nothing there.

Moving to 2.5 ms with 8 substeps would cut the physics cost by about 2.5 times.
That is **not tested**: it needs the same table re-measured with the tracking
randomization active before any run relies on it.

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

The PPO experiment name is ``agibot_x2_velocity_tennis_ball_centre_pos_only``.

Tests
-----

``tests/test_x2_tennis_ball_task.py`` covers:

* mesh provenance, bat masses and inertias, the hand replacement, and that the
  bat adds no joints or actuators;
* that only the string bed collides, and that the bed keeps its calibrated
  ``contype``, ``condim``, ``priority``, friction, ``solref`` and ``solimp``
  after the collision-config pipeline;
* that the compiled bed and ball are both six-dimensional and carry the declared
  friction triple, and that the ball body is 0.0560 kg with the shell
  inertia rather than a solid sphere's;
* that a ball rolling without slipping at 0.10 m/s on a level bed keeps between
  20 % and 95 % of its speed after one second, and keeps all of it when the
  rolling coefficient is zeroed;
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
