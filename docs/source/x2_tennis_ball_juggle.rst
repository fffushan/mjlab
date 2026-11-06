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

**Current resolved configuration (2026-11-05 +08):** ``racket_lift`` has weight
``0.0`` and is skipped. The derivation below describes the earlier penalty design,
not an active penalty in the current tasks. The `nominal six-run comparison
<../reports/x2_tennis_nominal_comparison_20261105.md>`_ uses the saved run settings.

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
     - ``0.0`` (disabled)
     - earlier maximum-slope design: 11.2196 per m; not charged by current configs
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

**The head range is +-0.3 rad for both joints** in ``x2_ultra.xml``, by operator
decision; the operator keeps the real robots inside it. Soft limits are 0.9 of the hard
limits, so the reachable band is ``+-0.27`` and that is what the sampler draws from.

**Measured on the built environment** (64 environments, at +-0.3): yaw spans
-0.201..+0.256 and pitch -0.252..+0.267, 60 of 64 distinct each, none outside +-0.3.
The held target does not change during an episode, and ``ball_in_camera_fov`` reads
1.000 for every environment, so at this range the randomization does not touch that
reward at all. Under the earlier +-0.5 request the band was -0.219..+0.325 and the same
term ranged 0.63-1.00.

Two consequences of the tighter range are worth stating rather than discovering later.
The trained default yaw is 0.2760, which is now *above* the +-0.27 soft limit and only
24 mrad below the hard limit: a zero action on the 31-DOF task settles the target at
0.2661 instead of 0.2760, so a trained policy sees ``joint_pos`` shifted by about 0.01
rad. That is inside the noise a policy already tolerates, but it is not zero. And the
measured robots that motivated this task are now **outside** the model range: ``.132``
reads pitch ``+0.5115`` and ``.236`` reads yaw up to ``+0.493``. Those poses are not
representable, by choice, so a robot parked beyond +-0.3 rad is an operator problem.

The range also applies to the tracking family, which tracks ``head_yaw_link`` and
``head_pitch_link`` as bodies: a reference motion that drives the head beyond +-0.3 is
now clamped. **Not measured** - no motion dataset was checked.

**Deployment contract.** The exported metadata gains ``action_joint_names``.
``joint_names`` remains the entity's 31-joint list, which is what the observation refers
to, so the two no longer agree and a consumer must map actions through the new field.

The Ball3 variant: three timestamped samples
--------------------------------------------

``...-Juggle-29DOF-Ball3`` is the 29-DOF task with the actor's ball input changed from one
delayed plate-frame position to **three timestamped samples**: 12 numbers, interleaved
``[x0, y0, z0, t0, x1, y1, z1, t1, x2, y2, z2, t2]``, newest first, where ``t`` is the age
of that sample in seconds relative to the current control step, clipped at 1.0 s. Widths
become **29 / 112 / 121** against 29 / 103 / 121 for the 29-DOF task; the critic keeps its
clean position and velocity and is unchanged.

**Why the stream, not a delay.** A single delayed position cannot say how stale it is, so
training had to choose one latency model and the two robots broke it: the estimator
measured 18.44 Hz on ``.147`` and 5.40 Hz on ``.132``, where one stall held a value for
10.4 s and a third of the published estimates were 2-3 m ghosts. The term therefore models
the chain itself - a per-environment ring of true positions sampled on a 5-20 Hz schedule,
each acquisition reading one drawn latency (80-250 ms) back, plus a per-episode constant
bias and per-sample noise. A stall is not a state: the sample stays and its age grows, so
the previous hold term is not used here.

**Slots are ordered by capture step, not arrival.** Latency jitter can deliver an older
frame after a newer one; ordering by arrival would put it in slot 0 and make the ages
non-monotonic. Fewer than three samples repeat the oldest slot **with its own age**, so a
duplicate is self-evident. The term counts its own control steps rather than reading
``episode_length_buf``, which rsl_rl seeds with ``randint`` at the start of every
``learn()``: that trap already produced phantom ``ball_not_lifted`` counts here.

**What it buys, measured on CPU.** With noise and bias off, fitting a parabola through the
three samples and extrapolating to the current step gives a median **69 mm** error against
**472 mm** for the freshest sample alone, in single-arc triples. Only 69 of 740 triples
were single-arc: the rest straddle a bounce, where the fit is meaningless. The gain is real
and **conditional**, and the policy has to learn when to trust the fit.

That conditionality has a physical cause worth stating: the ring stores each sample in the
plate frame **of its own capture step**, so the three positions live in three different
moving frames. Ballistic extrapolation is only valid while the plate barely moves between
samples.

**Not supported by the controller.** The deployed adaptor builds one plate-frame position,
so this task is training-only until the provider keeps a short history and reports ages.

The Ball3-Pelvis variant: samples in the pelvis frame
-----------------------------------------------------

``...-Juggle-29DOF-Ball3-Pelvis`` changes only the actor's ball-history frame and its
contract identity. The actor term is ``ball_hist_pelvis``. Action/actor/critic widths
remain **29 / 112 / 121**. The critic, rewards, actions and physics are unchanged.

**Frame and timing.** At capture time, compute
``p_local = R_pelvis.T @ (p_ball_world - p_pelvis_world)``.
Use the pelvis body origin and axes, not its centre of mass.
Store three positions and their capture timestamps, newest capture first.
The output remains ``[x0, y0, z0, age0, x1, y1, z1, age1, x2, y2, z2, age2]``.
Stored positions stay unchanged; their ages advance. Do not reproject old samples
into the current pelvis frame. Noise, latency, stalls and age clipping are unchanged.

Racket motion alone no longer changes the measured ball position. Pelvis motion
still does. The history is not a fixed-world ballistic trajectory. Small movement
commands do not guarantee a stationary pelvis, especially with the existing pushes.
This change adds no estimator, proprioception history, recurrent network or extra feature.

**Controller contract, not implemented.** The deployed adaptor cannot supply this
three-sample input. A future controller must convert each new camera measurement to
pelvis coordinates once, then store the position and capture timestamp.
Exact capture-time conversion needs a fixed camera-to-pelvis transform or capture-time
chain state. Using arrival-time joint state without history is an approximation.
That timing approximation is not modeled by this frame change and is not measured here.

**Validation.** ``tests/test_x2_tennis_pelvis_ball_history.py`` checks the coordinate
transform, unchanged stored samples, and task configuration. Existing Ball3 tasks
retain their plate-frame input. Use the separate task and experiment for new training;
do not assume an old checkpoint is compatible merely because its input width matches.
Learning improvement and controller deployment are **not tested**.

The Nominal variant: the pilot's randomization-free settings
------------------------------------------------------------

``...-Ball3-Pelvis-Nominal`` is the pelvis variant with every randomization axis the
nominal juggling pilot holds fixed switched off. It is built by wrapping the pelvis
task, so the scene, the 1 ms contact physics, the vendor gains, the actuator limits,
the reward weights, the loss and no-bounce terminations and the widths
(29 / 112 / 121) are that task's. The wrapping code is
``tennis_juggle_nominal_env_cfg.py``; the pilot is
``docs/design/x2_tennis_nominal_pilot.md``.

Resolved settings:

* Pushes, physics randomization, and encoder bias: removed.
* Actor corruption: disabled. Ball history also has zero internal position noise,
  bias, latency, and stalls. It keeps three samples at 50 Hz.
* Head, right arm, and waist resets: fixed default pose.
* Ball reset: 0.20 m drop, lateral jitter ±0.005 m, zero initial velocity,
  zero extra fall time, and no plate starts.
* Command: zero linear and angular velocity; no curriculum.
* Energy target: ``MJLAB_JUGGLE_ENERGY_TARGET_HEIGHT_M`` sets the reward and both
  energy metrics together.

Two details are load-bearing. First, **disabling the observation manager's corruption
is not sufficient to make the actor's ball input noise-free**: the history term applies
its own per-sample noise, bias, latency and stalls. The term now takes
``position_noise_m`` (default :data:`BALL_POSITION_NOISE_M`), and this task sets it to
zero with the other stream magnitudes. Second, **zero command ranges alone do not give
a zero command**: ``rel_forward_envs`` lifts a forward environment's ``lin_vel_x`` to at
least 0.3 m/s after sampling the ranges, so the task clears it as well. A third
detail is smaller: a zero-width head reset band writes the default pose directly,
because the model's soft head limits (+-0.27 rad) are narrower than the default yaw
(0.276 rad) and the sampled band would be empty.

A 0.20 m and a 0.40 m arm differ only in ``MJLAB_JUGGLE_ENERGY_TARGET_HEIGHT_M``, which
reaches the reward and both energy metrics together. Unset, the shipped 0.40 m is used.
``tests/test_x2_tennis_nominal_task.py`` pins the resolved settings, the shipped
defaults of the pelvis task, the zero-noise history and the drop placement. Whether a
policy learns under these settings is **not tested** here: the pilot's training runs are
separate, and no hardware claim is made.

Terminations
------------

The base velocity task's fall and timeout terms are unchanged; three ball terms are
added. ``ball_dropped`` is the carrying task's ground test: the ball centre at or below
0.5 m above the ground for 0.5 s, measured in the world frame so tilting the racket
cannot satisfy it. ``ball_not_lifted`` ends a plate start that never reached
1.0 J/kg of lift energy within 1.5 s. ``ball_no_bounce`` ends any episode in which
the racket has produced no **rising-edge impact** for 2.5 s of control time.

``ball_no_bounce`` targets a measured failure mode. In the last ablation the parked
arms collected about 4.5 reward per second for 19.4 s with 2.5 impacts per episode,
while the only bouncing arm survived 6.4 s at 1.3 impacts per second. Episode reward
is ``sum(raw × weight × dt) / max_episode_length_s``, so duration multiplies
everything; ending an episode whose last impact is 2.5 s old removes that advantage.
Only a rising edge restarts the clock: a ball resting on the plate is sustained
contact, not an impact, so parking stays visible to the term.

The deadline is 2.5 s since the **last impact**, not a fixed time in the episode. A
carry-up with sustained contact fires at about 2.6-2.8 s, and a toss at 1.4 s followed
by parking fires at about 4.3-4.5 s, because a release from a resting plate is not a
rising edge: without an intervening impact the clock keeps running from the pre-release
contact, so a plate start that lifts at 1.4 s has about 2.2-2.4 s of allowance left. 2.5 s is 4.4 flight times of a 0.40 m bounce, and
the window only expires on sustained contact: a missed bounce ends the episode through
``ball_dropped`` 0.5 s later, so 2.5 s of airborne misses cannot accrue. The timeout is
also longer than ``ball_not_lifted``'s 1.5 s, so a plate start that is never lifted is
ended by that term first.

A drop start cannot reach this deadline either. Measured on 8 environments with zero
action and every ball termination removed, a drop start produced 1-4 impacts in
20 s, its contact fraction decayed to zero, and the ball ended on the floor. The first
contact arrives 0.2-0.35 s after release, and an inert policy ends through
``ball_dropped`` about 0.5 s later, long before 2.5 s. Ground bouncing is already
closed by ``ball_dropped``. A sustained minimum-height dribble that keeps producing
rising edges remains possible and is deliberately not guarded here.

``Episode_Reward/*`` values and per-episode sums (``ball_impact_count``,
``ball_flight_time``) are not comparable with runs before this change: the reward
denominator is the fixed 20 s cap, and both a shortened episode and a shorter
denominator move those numbers for reasons unrelated to behaviour.

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
* **A validated arm impedance.** The arm's impedance under impact is not validated: the
  simulated arm can return more impact energy than a passive support can. Measured on this
  task's resets with 8 environments, zero action and every ball termination removed, the
  ball produced 1-4 impacts in 20 s, its contact fraction decayed to zero and it ended on
  the floor; it did not bounce indefinitely. An operator measurement of the shipped ball and
  racket gives a hard upper bound of 0.767 for a *rigidly supported* racket, so no passive
  surface can return more, and the position servo is the only remaining energy source. The
  contact model is not the cause and is left at the reference fit. Consequences: the energy
  metric alone cannot separate juggling from doing nothing, and a target at or below the
  drop height is free. The contact metrics are the intended check. This is deliberately
  deferred, because the arm's impedance has a sim-to-real gap and should be tuned from
  hardware behaviour rather than fitted in simulation.

Staged fine-tune rungs
----------------------

A rung is the nominal pilot with **one** physical condition changed, trained by resuming
the previous rung's checkpoint with the same environment count and seed, so a rung
differs from its parent in the input condition only. There is no scheduler: the
checkpoint is the whole handoff, and each run's ``params/env.yaml`` records exactly one
condition.

==========================================  ==============================================
task suffix                                 condition
==========================================  ==============================================
``-Nominal-S1-15Hz``                        ball history at 15 Hz, 0-40 ms delay
``-Nominal-S2-80-100ms``                    15 Hz, 80-100 ms delay
``-Nominal-S3-40-140ms``                    15 Hz, 40-140 ms delay, redrawn per sample
``-Nominal-S3b-80-140ms``                   15 Hz, 80-140 ms delay, redrawn per sample
``-Nominal-S3c-0-140ms``                    15 Hz, 0-140 ms delay, redrawn per sample
``-Nominal-S4-Stalls``                       S3's band plus the assumed stall model
``-Nominal-S4b-Stalls``                      S3's band plus the measured (clip) stall model
``-Nominal-S5-Noise``                        S4b plus +-(10, 10, 15) mm position noise
``-Nominal-S6-Release``                      S5 plus a release-position distribution
``-Nominal-S7a-DelayedRelease``              S6's perception plus a 0-2 s held ball, then release
==========================================  ==============================================

``...-Ball3-Pelvis-Nominal-S1-15Hz`` is the first rung. It is the
nominal pilot with the ball history arriving at
15 Hz with 0 to 40 ms of uniform transport delay, so the estimator samples are
asynchronous with the 50 Hz control clock. Rewards, terminations, resets, the zero
command, the locked head, the action width 29, the actor width 112 and the critic
width 121 are the nominal pilot's. Three samples are still stored, so the actor stays
112 wide and an earlier checkpoint resumes with its optimizer state. The stage code is
``tennis_juggle_stage_env_cfg.py``; the term parameters it moves are ``stream_rate_hz``
and ``stream_latency_s`` on ``ball_hist_pelvis``, and nothing else.

The term quantises both values to whole control steps, which are 20 ms here. The
requested 15 Hz runs one sample every 3 steps, so the realised rate is **16.67 Hz**, and
the 0-40 ms delay becomes 0, 20 or 40 ms with probabilities 0.25 / 0.5 / 0.25. A true
15 Hz is not representable at this control rate; the nearest rates are 16.67 Hz and
12.5 Hz, and this stage takes the faster one. Read a pass here as mildly optimistic, and
run 12.5 Hz if the answer has to be conservative.

``...-Nominal-S2-80-100ms`` keeps S1's rate and moves the delay to 80-100 ms. That is a
different order of magnitude from S1, and it is a target for an improved estimator rather
than a measurement of the current one: the deployed chain costs 26.1 ms of adapter, about
65 ms of stereo inference and 2.2 ms of estimator before transport, and was measured end
to end at 110-141 ms. Latency in this term is **capture to consumption**, not capture to
publish: it includes the frame phase, transport and the control-tick quantisation.

``...-Nominal-S3-40-140ms`` draws the delay from 40 to 140 ms on every acquisition, i.e.
2 to 7 control steps, so the delay changes inside an episode. A fixed delay can be
compensated once with a single learned lead; a varying one has to be read from the
per-sample age that the observation already carries. This is the rung that tests whether
that age is usable, and it is the rung whose condition covers both the best estimator we
could build and the measured 110-141 ms chain.

``...-Nominal-S3b-80-140ms`` is a **sibling** of S3, not its successor: the same wide band
shifted up, mean 110 ms instead of 90 ms. Both resume S2's checkpoint and run the same
budget, so the pair isolates the effect of the band's **centre**. The measured chain's mean
is about 125 ms, so S3b brackets what the current pipeline delivers while S3 brackets what a
better estimator could. ``...-Nominal-S3c-0-140ms`` is the third sibling: the whole band,
mean 70 ms, which also includes zero delay, where the S2 policy collapses to streaks of
five flights. The three curves differ only in the band's centre, so together they say
whether a wide band teaches the policy to use the per-sample age or only to sit on the
band's mean.

``...-Nominal-S4-Stalls`` changes the condition rather than the delay: S3's 40-140 ms band
plus the measured stall model (2 % of acquisitions stall for 0.1-0.6 s). A stall is a longer
gap before the next acquisition, so the newest sample is simply absent and its age keeps
growing - 10 to 30 control steps on top of 2 to 7 steps of transport lag. S3 showed the
policy reads that age; S4 asks whether it stays correct once the ball must be extrapolated
rather than observed. Per the curriculum plan, rate, latency and stalls are separate
substeps, and stalls is the third.

``...-Nominal-S4b-Stalls`` is a sibling of S4 that replaces S4's assumed stall parameters with
measured ones: a recorded hardware clip shows the ball leaving the camera field of view when it
is raised high, for about 0.1 s, on about 10 % of frames. The gap is therefore 5-10 control
steps recurring every ~0.6 s, against S4's 5-30 steps every ~3 s. The worst age the actor sees
falls from 0.74 s to 0.34 s while the fraction of time the newest sample is stale rises from
about 10 % to about 20 %. S4 trained against gaps the camera does not produce and degraded to
0.38 s flights with five drops per 4096 environments; S4b tests the gap that occurs.

``...-Nominal-S5-Noise`` adds measurement error on top of S4b's whole condition: the rate, the
latency band and the stall model all repeat, because a stage carries the complete condition and
not only its own increment. Position noise is per axis and per acquisition, drawn uniformly in
+-10 mm laterally and +-15 mm vertically, so the three history samples carry independent error,
which is what a real estimator gives. The ages stay exact; only the positions are wrong. The bias
is deliberately zero: a per-episode offset is not measurable with the current rig, and a bias
drawn in the wrong direction would train a compensation for a fault that may not exist.

``...-Nominal-S6-Release`` stops varying what the estimator reports and varies what the operator
does: the release. The ball leaves the hand anywhere from rest on the strings to 0.40 m above them,
sampled uniformly, and within +-20 mm of the plate centre in both lateral axes. The target apex
stays 0.40 m, so a release at the target height must be caught and re-bounced rather than lifted,
and a release at zero has no fall energy at all. Everything S5 added still applies. The ranges are
the operator's stated intent; they are not yet a measurement of a real hand release, and the
curriculum plan asks for that measurement before the ranges are treated as final. A stage declares
these with a ``reset`` sub-entry, which the factory applies to ``events["reset_ball"]``; every other
event term must still equal S0's, and the one-delta test enforces that.

``...-Nominal-S7a-DelayedRelease`` is the first rung that changes the episode **structure**
rather than a perception condition. The ball is pinned in world space 0.40 m above its resting
height and released after a uniform 0 to 2 s wait, so the policy must place a level, quiet racket
under it and be ready before the ball moves. S5's whole perception condition still applies: a
deferred release must be seen through the same delayed, stalled, noisy chain, and the actor is
never told that the ball is held.

Three additions come with the hold, and each closes a measured loophole rather than adding skill.
The energy reward is gated to the post-release phase, because a held ball is world-fixed with zero
velocity, so its energy is a per-episode constant the policy cannot influence: at a 0.40 m hold it
equals the target exactly and would pay a full 1.0 on every waiting step for doing nothing.
``racket_under_ball`` pays for a level, quiet racket under the held ball, because ``ball_centred``
is plate-frame horizontal only and a tilted plate can satisfy it without the racket being below the
ball. ``held_ball_contact`` ends an episode in which the racket touches the held ball, so a wait
cannot become an early strike or a preload. The no-bounce deadline is paused while the ball is
held, or a 2 s wait would consume 80 % of its 2.5 s window before the release.

The hold height is fixed at 0.40 m, which is the energy target, so the release energy equals
``E*`` and any change is attributable to waiting. Randomising the hold height changes the drop
energy as well, and is a separate rung.

A rung is trained by resuming the previous rung with the same environment count and
seed, so the two runs differ in the input condition only. S1 and S2 are **trained**;
S1 passed its screen unchanged against the nominal pilot, and S2 passed at its own
condition, where the S1 policy scores 61 qualified flights and 1085 drops in closed loop.
S3 is **not trained**: no result is claimed for it here.
