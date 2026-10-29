X2 Tennis Ball Juggling
=======================

Overview
--------

``Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle`` is the
juggling variant of the tennis-ball task: the ball is dropped onto the string bed and
must be kept bouncing. It is the carrying task (see :doc:`x2_tennis_ball_balance`)
with a different ball reward set, a standstill command band and no command curriculum.
Scene, assets, 1 ms physics, observations, resets, termination, servo gains and domain
randomization are the carrying task's, so the actor width is still 105 and the critic
width 123. The config is a copy, not a wrapper: the two files diff by intent.

Goal
----

Juggling, not carrying. The carrying task pays for a ball held still on the plate; this
task pays for **sustained vertical energy** toward a **0.40 m** target. The operator
chose 0.40 m to match human juggling height. The ball starts 0.20 m above the bed and
falls, so a bounce exists from the first control step. The spawn is 0.20 m below the
target. At sigma 1.5 J/kg the spawn scores 0.1807. At the old sigma of 0.5 J/kg it
scored 2.1e-7 and gave no learning signal. Version 1 juggles while standing still: the
twist command is pinned to about ±0.10 m/s linear and ±0.05 rad/s yaw, with no command
curriculum, since one band has no stage to reach.

Reward structure
----------------

There is no cross-term normalisation. Every term is multiplied by its weight and by the
control step, and the results are summed:

.. math::

   r_t = \sum_i w_i \, r_i \, \Delta t \qquad \Delta t = 0.02\ \mathrm{s}

So the relative influence of two terms is exactly ``weight × term value``. Every
positive term here is bounded in [0, 1]; the penalties are unbounded. All figures below
are per control step.

``tennis_constants.py`` is the single source of truth for every number quoted here.

Velocity tracking
-----------------

.. list-table::
   :header-rows: 1

   * - term
     - weight
     - shape
     - per step
   * - ``track_linear_velocity``
     - 2.0
     - :math:`\exp(-\lVert v_{cmd} - v_{xy}\rVert^2 / 0.25)`, std 0.5 m/s
     - 0.040 at zero error
   * - ``track_angular_velocity``
     - 2.0
     - :math:`\exp(-\lVert\omega_{cmd} - \omega_z\rVert^2 / 0.5)`, std 0.707
     - 0.040 at zero error

These are the largest positive per-step values in the task, and they apply in every
variant that inherits the base velocity task.

**In this task the command band is pinned at ±0.10 m/s**, so a robot that stands still
has an error of about 0.05 m/s and the term reads about 0.99 — roughly **0.040 per step
whatever the robot does with the ball**. That is a near-constant offset. It does not
change the policy gradient, but it does mean the reported total reward cannot be read as
a measure of juggling skill.

If the band is ever widened, this term becomes discriminating and, at weight 2.0, would
rival or exceed the ball terms. Reduce it to about 0.5 at the same time, or velocity
tracking will outvote the ball.

Why a ball-energy reward
------------------------

The first version of this task paid a Gaussian on the **apex gap once per bounce**. It
was measured to fail, for two reasons.

**It was flat where the policies lived.** With a 0.30 m target and a 0.08 m sigma, an
apex of 40 mm scores ``exp(-(0.26/0.08)^2) ≈ 2e-5``. The policies reached 31-71 mm, so
the term paid essentially nothing and carried no gradient. Raising the weight only
scaled a zero.

**A bounce is a poor unit of reward.** One payment per cycle is sparse, and the gates it
needed — a 30 mm clearance and a 0.30 m/s peak rising speed — turned a physics question
into a threshold question.

What replaced it is a **dense, per-step ball energy**. Energy is the state the policy
must hold, and unlike a per-step height it cannot be satisfied by parking the ball at the
target height:

.. math::

   \begin{aligned}
   h &= z_{ball} - z_{rest} \\
   E &= \tfrac{1}{2} v_{z,ball}^{2} + g\,h \\
   r &= \exp\left(-\left(\frac{E - g \cdot 0.40}{\sigma_E}\right)^{2}\right),
   \qquad \sigma_E = 1.5\ \mathrm{J/kg}
   \end{aligned}

where ``z_rest`` is the ball centre's world height when it rests on the plate in the carry
pose (``BALL_SPAWN_LOCAL_POS[2]``, measured). The reference is **fixed in the world**: it
moves neither with the plate nor with the pelvis. ``g`` is read from the simulator, not
hardcoded, so the target matches the gravity in use.

| Ball resting on the plate: ``E ≈ 0``.
| Top of a bounce to height ``h``: ``E = g·h``, so a 0.40 m apex is exactly the target.
| Between impacts: ``E`` is conserved, so the term pays every step of the flight.

Three deliberate restrictions, each closing a way to earn energy without bouncing:

* **Vertical only.** With the full speed, ``½|v|²`` reaches the target at 2.80 m/s: a
  robot walking at 2.80 m/s gives a ball sitting still on the racket that energy, and so
  does a ball rolling across the plate at 2.80 m/s.
* **World vertical, with a datum fixed in the world.** Any *moving* datum can be farmed.
  In the plate frame a descending racket manufactures energy with the ball untouched:
  dropping the arm at 1 m/s under a resting ball gives 0.5 J/kg, 13 % of the target,
  and at about 2.8 m/s it reaches the target outright. With the pelvis as the datum,
  crouching does the same — lowering the pelvis 0.20 m while the ball stays put raises
  ``h`` by 0.20 m and adds 1.962 J/kg, half the target, with no work done on the ball at
  all, and the lift term cannot see it because the plate falls with the body. A fixed
  world reference supports neither trick. The price is that the ball's height now
  includes gait bob: a 3 cm bob is 0.29 J/kg, about 0.19 sigma. That is noise correlated
  with real motion rather than a free reward, and it is accepted deliberately.
* **The racket may not be lifted.** A ball resting on a racket raised 0.40 m reads
  exactly ``E = g·0.40``, so the energy term alone would pay for holding the arm up.
  Hence the second term below.

The target 0.40 m is **not** the height of the drop-in spawn. ``BALL_SPAWN_INIT_POS`` is
0.20 m above the ball rest position, and a test pins that to 1e-9. The episode therefore
starts 0.20 m **below** the target. At spawn ``E = 1.962`` J/kg against ``E* = 3.924``
J/kg, a deficit of 1.962 J/kg. At sigma 1.5 J/kg the spawn scores 0.1807, so the
climb to the target has a gradient. At sigma 0.5 J/kg it scored 2.1e-7 and a run had
no usable signal.

Ball reward set
---------------

.. list-table:: Ball reward terms
   :header-rows: 1

   * - term
     - weight
     - purpose
   * - ``ball_energy``
     - 2.0
     - dense vertical ball energy, target ``g · 0.40``, sigma 1.5 J/kg
   * - ``ball_centred``
     - 0.25
     - horizontal distance from the plate centre; plateau 20 mm, sigma 40 mm
   * - ``racket_lift``
     - ``11.2196`` per m
     - maximum-slope penalty on the plate rising above its nominal world height, 30 mm deadband
   * - ``ball_in_fov``
     - 0.2
     - soft camera constraint, unchanged
   * - ``head_joint_motion``
     - -0.05
     - head velocity penalty, unchanged

``ball_centred`` measures the **horizontal** offset only (``position[:, :2]``), so a
ball bouncing straight up keeps earning it in full. Centring and bouncing are additive,
not competing; the weight is a quarter of the carrying task's 1.0 because the ball spends
most of the cycle in the air.

Why the racket term is not a tidy-up
------------------------------------

Without it the cheapest policy is to raise the arm: park the ball at the 0.40 m target
and collect the full energy reward forever. The weight is *derived*, not chosen, from
the **maximum-slope rule**:

.. code-block:: python

   JUGGLE_RACKET_LIFT_WEIGHT = juggle_racket_lift_weight()  # sigma = 1.5 J/kg
   # 2.0 * sqrt(2) * (9.81 / 1.5) * exp(-0.5) = 11.2196 per metre

The energy Gaussian's steepest slope in lift is
:math:`\sqrt{2}\,(g/\sigma)\,e^{-1/2}` per metre. A lift penalty at least that steep
makes the net reward non-increasing above the deadband. No partial lift can farm the
pair. A lift to the target is strongly negative.

The wider sigma is exactly what makes this rule necessary. With the old sigma of
0.5 J/kg, the old rule ``2.0 / (0.40 - 0.03) = 5.405`` cancelled the reward at the
target. At sigma 1.5 J/kg that rule lets partial lifts near the target profit by about
+0.09 at 0.36 m. The maximum-slope rule closes that farm.

Arm motion itself is not banned, because the penalty is per step and so is the reward:

| one bounce cycle, about 20 steps at the target: ``+2.0 × 20 = +40``
| a 0.10 m strike lasting 5 steps: ``-11.2196 × 0.07 × 5 ≈ -3.9``

The strike costs about a tenth of what it earns. Only *sustained* lifting is charged.

Both ``h`` and the lift use the same fixed world reference, so lifting the plate is
penalised whether the arm or the legs do it, and a crouch cannot buy energy because it
does not change the ball's world height. The two reference constants are absolute world
heights, which is valid only because the tennis tasks pin the base reset at
``BAT_CARRY_BASE_HEIGHT`` with zero yaw; if that pin were ever removed they would need
rethinking.

Why the 40 cm target
--------------------

The operator chose 0.40 m to match human juggling height. At the carry pose the plate
top surface sits at world z 1.1194. A 0.30 m gap puts the ball at a normalised image
radius of **0.434** — 27.8 degrees of the 64.0 degree vertical half-frame — inside the
0.60 plateau, so the field-of-view reward is exactly 1.000 there; at rest the same ball
reads 0.271. The 0.40 m target sits just inside the plateau: the falloff begins just
above 0.41 m of gap (0.595 at 0.41, 0.608 at 0.42). The target has little field-of-view
margin, but it does not fight that term. These values are measured on the built
environment at the pinned reset pose, with the camera mount from the tennis constants.

A 0.40 m apex needs a launch of :math:`\sqrt{2 \cdot 9.81 \cdot 0.40} = 2.80` m/s. That
is a real bounce. The drop-in spawn does not deliver it: the passive rebound measured on
the built environment returns 0.205 m from a 0.20 m drop, because the sagging arm returns
energy during the fall. The passive peak is therefore about 0.20 m below the target.

The 29-DOF variant
------------------

``...-Tennis-Ball-Juggle-29DOF`` is this task with the two head joints out of the action
space: 29 actions, and actor/critic widths of 103/121 against 105/123 here. Everything
else is the same config, produced by ``head_actuated=False``.

**Why.** The head breaks. Two of the three machines we can reach have one head axis that
does not follow its command. On ``.147`` ``head_yaw_joint`` reports ``error_code 1027``
with its position frozen at ``+3.141590118``. On ``.132`` the same joint accepts 2-7 N.m
and delivers about 0 N.m: it travelled 0.057 rad while its target travelled 1.400 rad,
while ``head_pitch`` tracked its own target to 0.014 rad in the same run.

**The asymmetry is deliberate.** The joints stay in ``joint_pos`` and ``joint_vel`` and
leave only the action vector. The camera rides on the head, so its measured pose feeds
the camera-to-plate conversion; a policy that never received the pose would be
unconvertible on the robot. What changes is that the pose becomes an *exogenous
condition* rather than something the policy sets.

**The condition is randomized per episode** over the feasible band of both joints, and
held for the whole episode. Measured offsets from the trained default (yaw 0.2759953,
pitch 0.0617994) are -0.349 rad for ``.132`` yaw, +0.450 for ``.132`` pitch and +0.217
for ``.236`` yaw, so a policy trained on the default alone is fitted to one robot.
``head_joint_motion`` is dropped rather than zeroed: the policy cannot influence it.

Three implementation details are load-bearing, and each has a silent failure mode.
The reset writes the **actuator target**, not only the joint state, because the head is
outside every action term: the reset zeroes that target, nothing rewrites it, and a
state-only write decays back within a few steps — which would silently remove the
randomization by the time the ball is released. The sample is uniform over the
intersection of the soft limits with default +- 0.5 rad, rather than an offset followed
by a clamp, because the soft range for head yaw is narrower than the offset and clamping
piles environments onto one limit. The scale map is keyed by joint pattern and every key
must match, so the two head entries must be dropped along with the joints.

**Measured on the built environment** (64 environments): yaw spans -0.219..+0.325 with
59 of 64 distinct, pitch -0.343..+0.343 with 61 of 64, and one environment at a limit
against the 32 that a clamp would have produced. The held target does not change during
an episode, and ``ball_in_camera_fov`` stays at 0.63-1.00 (mean 0.98) across the sampled
poses, so the randomization does not fight that reward.

**Known limit, and it needs an operator decision.** The model's joint ranges are
``-0.366..0.366`` for head yaw and ``-0.3838..0.3838`` for head pitch, and the *soft*
limits are narrower still at about ``-0.02..0.345``. Both robots measured are outside
them: ``.132`` reads pitch ``+0.5115`` and ``.236`` reads yaw up to ``+0.493``. So this
variant cannot represent two of the three measured head configurations, and the sampled
band covers only the near-default ones. Covering them needs the head range in
``x2_ultra.xml`` widened from measurement. That is a robot-model change and is not made
here: it also changes the limits a deployed controller enforces, because the hardware
config reads ``limits: source: robot_model``.

**Deployment contract.** The exported metadata gains ``action_joint_names``.
``joint_names`` remains the entity's 31-joint list, which is what the observation refers
to, so the two no longer agree and a consumer must map actions through the new field.

Acceptance metrics
------------------

``Episode_Reward/ball_energy`` is the reward as collected. The task adds two **energy**
metrics and three **contact** metrics.

The energy metrics are the episode mean of ``E`` and of ``|E - E*|``. They are this
task's energy acceptance signals, because a reward can be collected in unintended ways
and the energy metric is behaviour.

The contact metrics read the ``bat_ball_contact`` sensor, whose ``force_history``
buffer keeps one control step of physics substeps (``history_length`` equal to
``cfg.decimation``, 20 substeps of 1 ms):

* ``ball_impact_count`` counts rising edges of the per-step contact flag: any
  substep with a non-zero contact force. One continuous contact counts once, and a
  contact that starts and ends inside one 20 ms step also counts once. It uses
  ``reduce="sum"``.
* ``ball_flight_time`` reports the running mean airborne time between impacts, in
  seconds, accumulated in substeps. Contact substeps are excluded. It uses
  ``reduce="last"``.
* ``ball_sustained_contact`` reports the longest continuous contact seen in the
  control step, in seconds, accumulated in substeps. It uses ``reduce="max"``.

All five ball metrics are **logging only**. They change no reward, observation,
termination or curriculum. The contact metrics are the only evidence that can
distinguish juggling from a passive bounce. Substep resolution is needed because the
racket-ball contact lasts about 9-10 ms: a single ``found`` snapshot per 20 ms step
misses about half the contacts. One limit remains: ``ball_impact_count`` is a rising
edge of a per-step flag, so it counts two contacts as one whenever no control step
between them is fully airborne - in particular, two contacts inside one control step.
Flight time and sustained contact still resolve that gap in substeps.

The former ``ball_bounce_count`` metric was removed with the apex term. It read that
term's per-step flag, and ``RewardManager`` skips zero-weight terms, so it would have
frozen at its last value once the apex term stopped being computed.

Tests
-----

``tests/test_x2_tennis_juggle_task.py`` pins the registration and a real build, the
reward diff against the carrying task term by term, the 105/123 observation widths, the
standstill band and the absent curriculum. The energy formula is pinned by direct tests:
a resting ball gives ``E ≈ 0``; the top of a 0.40 m bounce gives exactly the target; the
reward is symmetric about the target; **a ball and plate moving vertically together leave
``E`` unchanged** (the anti-loophole test — a descending racket must not create energy);
the drop-in spawn is 0.20 m below the target and scores 0.1807; no partial lift makes the
energy-plus-lift pair positive; and the lift weight follows the maximum-slope rule, so a
sigma change moves it. A synthetic contact sequence pins the three contact metrics and
their reset. The tests do not show that a policy can juggle: they drive the reward terms
with a scripted ball state.

Not implemented
---------------

* **Walking while juggling.** The command is a standstill band. A walking version needs a
  command curriculum, which this task deliberately does not have.
* **An energy-target curriculum.** The target is a fixed ``g · 0.40``. There is no ramp
  from a lower energy.
* **A lift weight that leaves the strike free.** The maximum-slope weight is about 2x the
  old rule. If it suppresses the upward strike, the lift weight is the first suspect. The
  symptom is a ball that dies while ``racket_lift`` stays near zero.
* **Gait-bob rejection.** None. The ball's world height includes the vertical gait
  oscillation, which is about 0.2 sigma at a 3 cm bob. It is not filtered, because every
  filter tried for this reward so far has reintroduced a datum that can be farmed.
* **Real-robot verification.** None. The target, the weights and the camera geometry are
  simulator-level and unverified on hardware.
* **A validated arm impedance.** The simulated arm returns more impact energy than a passive
  support can. With no policy action at all the ball rebounds to 1.02-1.05 times its drop
  height and bounces indefinitely (34 bounces in 18 s, apexes alternating about 0.21 and
  0.06 m). An operator measurement of the shipped ball and racket gives a hard upper bound of
  0.767 for a *rigidly supported* racket, so no passive surface can return more, and the
  position servo is the only remaining energy source. The contact model is not the cause and
  is left at the reference fit. Consequences: the energy metric alone cannot separate
  juggling from doing nothing, and a target at or below the drop height is free. The contact
  metrics are the intended check. This is deliberately deferred, because the arm's impedance
  has a sim-to-real gap and should be tuned from hardware behaviour rather than fitted in
  simulation.
