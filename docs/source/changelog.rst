Upcoming version (not yet released)
===================================

Changed
^^^^^^^

- **Measured right-arm servo, and 23 tasks that use it.**
  ``right_shoulder_pitch_joint`` damping 3.0 -> 4.81 and
  ``right_shoulder_roll_joint`` 3.0 -> 4.78, from the robot's step response with the
  racket held (``zeta`` 0.44 -> 0.70 at the carry-pose inertia). Stiffness is untouched.
  A ``BuiltinPositionActuatorCfg`` holds one damping for all its targets, so the two
  shoulder groups split per side; the actuator order, the action scale and
  ``dof_armature`` do not change. Both the fixed-height ladder and the target-height
  family gain a ``-RightArmMeasured-`` twin for every stage (19 + 4 = 23), generated
  from the stage tables. See ``docs/design/x2_right_arm_measured.md``.
- **Target-height S3: the wide 40-140 ms band on top of the height command.**
  ``...-Ball3-Pelvis-Nominal-S3-TargetHeight`` carries S2's target range and widens the
  ball-history latency band to 40-140 ms, redrawn per acquisition -- the condition that
  made the shipped ladder flat across 0-140 ms. The family's probe check 1b is now
  table-driven over the ported rungs (S2 and S3) and asserts each one twice: it equals
  its ladder stage plus the target machinery across twelve config sections, and it
  differs from its parent in exactly one parameter, the latency band.
- **Target-height S2: the height-conditioned policy at an 80-100 ms perception band.**
  ``...-Ball3-Pelvis-Nominal-S2-TargetHeight`` carries S1a's uniform 0.05-0.40 m target
  range and changes one condition: the ball-history latency band moves from 0-40 ms to
  80-100 ms. The perception ranges are not redefined in this family; a
  ``STAGE_BASE_STAGE`` map points each stage at the ladder's own stage table, so there is
  one definition of every range. The probe asserts the pair is an experiment rather than
  two runs: S2 equals the shipped ``s2-80-100ms`` stage plus the target machinery in
  twelve config sections, and S1a and S2 differ in exactly one parameter,
  ``actor/ball_hist_pelvis/stream_latency_s``.
- **The carry-pose velocity task no longer collides the string bed.** The tennis entity
  keeps its calibrated bed contact in the juggle family, where a ball loads it, but the
  carry task has no ball and runs a 5 ms physics step that the bed's ``solref`` was never
  fitted for. Two arms of that task died with a NaN in the actor observation, at
  iterations 8206 and 15781, with healthy metrics immediately before and no warning in any
  logged term. ``TENNIS_BED_COLLISION_DISABLED`` zeroes ``contype``/``conaffinity`` on
  ``string_bed`` and leaves the geometry, the face site and the ``solref`` in the model.
  The bed is the racket's only collider, so the racket now has no contact at all in that
  task; that is the intended state where the racket is held clear of the ground, and the
  probe asserts it. Not claimed: that this removes the NaN -- no run has completed the
  budget under the change yet.
- **Delayed release is now a training condition, not only a design goal.** The task
  ``...-Ball3-Pelvis-Nominal-S7a-DelayedRelease`` holds the ball in world space 0.40 m
  above its resting height and releases it after a uniform 0-2 s wait, with S5's whole
  perception condition carried, so the policy must place a level, quiet racket under
  the ball and be ready before it moves. The actor is never told that the ball is held.
  Four parts exist because each closes a measured loophole: the energy reward is gated
  to the post-release phase (a held ball is world-fixed, so its energy is a per-episode
  constant the policy cannot influence, and at 0.40 m it equals the target exactly);
  ``racket_under_ball`` pays for a level, quiet racket under the ball while held,
  because ``ball_centred`` is plate-frame horizontal only and a tilted plate satisfies
  it; ``held_ball_contact`` ends an episode in which the racket touches the held ball;
  and the no-bounce deadline is paused while held, or a 2 s wait consumes 80 % of its
  2.5 s window. Measured on four environments with zero action: the pinned ball's
  observed position error is 0.0000 mm and its speed 0.000000 m/s, the energy reward is
  exactly 0 while held and 1.0000 at release, the release has a 0.0000 mm position jump
  and 0.000000 m/s velocity, and at a forced 3.0 s hold ``ball_no_bounce`` never fires
  while the same term without the pause fires at 2.48 s. What is not claimed: that a
  policy learns to wait, and that the 0.40 m hold matches a real hand release.

- Revised the `X2 juggling curriculum implementation plan
  <../design/x2_tennis_nominal_curriculum.md>`_ to use real-robot results as the
  decisive acceptance gate. Controller-contract parity and measured sensor/reset
  conditions precede physical trials. This is a plan, not an implemented change.

- **The two plate-frame rewards are now bounded below.** ``ball_centred`` measures the
  ball's horizontal distance from the plate centre and ``ball_height`` clamped its
  height excursion from above only, so a ball that had left the plate downward scored
  1.0 on both. The pair was safe while "below the plate" was fatal, and that stopped
  being true when ``ball_dropped`` moved to the ball's absolute height: a ball hanging
  half a metre under the strings, kept above 0.5 m world, paid full credit and never
  ended the episode. ``ball_centred`` is now zero when the ball is more than
  ``BALL_SINK_M`` (5 mm) below the string plane, and ``ball_height`` decays on both
  sides of its flat band. Measured at the plate frame: resting on the strings, 1 cm
  above, and 3 mm sunk all score 1.0 / 1.0; 20 mm sunk scores 0.0 / 0.87; 30 cm under
  scores 0.0 / 0.0. ``ball_below_plate``, the dead plate-frame termination that would
  have restored the old contract, is deleted.

- **``ball_dropped`` now tests the ball's absolute height, not its height above the
  plate.** The old test measured the ball in the plate frame, so "below the plate"
  rotated with the racket. A policy could hold the racket near vertical, leave the
  ball on the floor, and keep the ball on the positive side of that plane: the height
  never fell below the threshold and the episode ran to the 20 s limit with no ball at
  all. The 30 k Ball3 baseline shows it, with 3.6 racket contacts across a 19.2 s
  episode and 94 % of episodes ending by truncation instead of by losing the ball. The
  term now ends the episode when the ball centre stays at or below
  ``BALL_LOST_MIN_WORLD_Z_M`` = 0.5 m above the ground for 0.5 s. That number cannot be
  changed by any pose of the robot, so the only way to stay above it is to keep the
  ball up. A carried ball sits at about 1.15 m and a ball on the floor at 0.03 m, so
  the margin is 0.65 m. The plate is no longer an argument of the term, and the
  ``BALL_FALL_THRESHOLD`` constant is replaced by ``BALL_LOST_MIN_WORLD_Z_M``.
  Episodes in the carry and juggle tasks are not comparable with runs before this
  change: long episodes that ended by truncation may have had the ball on the floor.

- **The juggle task's target apex height is now ablatable.** ``MJLAB_JUGGLE_ENERGY_TARGET_HEIGHT_M``
  overrides ``JUGGLE_ENERGY_TARGET_HEIGHT_M`` (0.40 m) for one arm. The reward and both
  energy metrics read the same resolved value, so an arm cannot score against one target
  and be measured against another. Unset leaves the shipped 0.40 m and cannot drift from
  the constants. A lower target is a much easier objective: at 0.20 m the target energy is
  1.962 J/kg against 3.924, and the reward at the 0.20 m drop-start release rises from
  0.18 to 1.0 of the Gaussian maximum, so the arm begins inside the reward's useful range.

- **The tennis juggle tasks now end an episode whose last racket-ball impact is
  2.5 s old.** ``ball_no_bounce`` counts rising edges of the per-step contact flag and
  restarts only on a strike; sustained contact does not restart it, so a ball parked on
  the plate no longer earns to the 20 s cap. In the last ablation the parked arms
  collected about 4.5 reward per second for 19.4 s with 2.5 impacts per episode, while
  the only bouncing arm survived 6.4 s at 1.3 per second; episode reward is divided by
  the 20 s limit, so duration multiplied the parked score. The deadline is 2.5 s since
  the last impact, not a fixed episode time: a carry-up with sustained contact fires at
  about 2.6-2.8 s, and a toss at 1.4 s followed by parking at about 4.3-4.5 s. 2.5 s is
  4.4 flight times of a 0.40 m bounce, and the window only expires on sustained
  contact: a missed bounce ends the episode through ``ball_dropped`` 0.5 s later, so
  2.5 s of airborne misses cannot accrue. The timeout is longer than
  ``ball_not_lifted``'s 1.5 s, so it cannot pre-empt a never-lifted plate start, and an
  inert drop start ends through ``ball_dropped`` long before it. ``Episode_Reward/*``
  values and per-episode sums (``ball_impact_count``, ``ball_flight_time``) are not
  comparable with runs before this change: the reward denominator is the fixed 20 s
  cap, and both a shortened episode and a shorter denominator move those numbers for
  reasons unrelated to behaviour. All three juggle tasks (31-DOF, 29-DOF,
  29-DOF-Ball3) inherit it.

- **The tennis juggling tasks no longer charge for lifting the ball.** ``racket_lift``
  is installed with weight 0, so ``JUGGLE_RACKET_LIFT_WEIGHT = 0.0``. The maximum-slope
  derivation was a correct argument about the reward at one point, but it does not model
  what the policy does with the rest of the episode, and two independent 30k-iteration
  batches agree it suppresses bouncing rather than parking: in the A0-A4 sweep the
  lift-0.0 arm (A3) was the only one that juggled and is the deployed policy, and in the
  stage-1 four-arm batch the charged arms held the ball on the plate (1.95 impacts per
  20 s episode, ``ball_energy`` 0.080 against a 3.92 target, reward 82 earned by
  standing) while weight 0.0 gave 16.7 impacts and 0.423. ``juggle_racket_lift_weight()``
  and the ``MJLAB_JUGGLE_LIFT_WEIGHT`` override are unchanged, so the charged arm can be
  re-run without editing code. Neither setting reaches the energy target: parking the
  ball is not solved by this change, and the acceptance signal stays ``ball_energy`` and
  ``ball_impact_count`` rather than reward.

- Narrowed both X2 head joint ranges in ``x2_ultra.xml`` from ``-0.366..0.366`` (yaw) and
  ``-0.3838..0.3838`` (pitch) to **``-0.3..0.3``**, by operator decision, to bound what a
  head-pose-randomized policy has to represent. Soft limits become ``+-0.27``. Two effects
  to know about: the trained default yaw of 0.2760 now sits above the soft limit, so a zero
  action settles the target at 0.2661 rather than 0.2760 (about 0.01 rad of ``joint_pos``
  shift), and the tracking family, which tracks ``head_yaw_link``/``head_pitch_link`` as
  bodies, will clamp any reference motion that drives the head beyond the new range. The
  motions were not measured.

Removed
^^^^^^^

- **Deleted thirteen ablation task registrations that no active workflow needs.** The four
  velocity torso-IMU and critic-frame tasks
  (``Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Torso-IMU-Critic-{Pelvis,Torso}-{Root,Upvector}``)
  are gone together with the ``imu_source``, ``critic_imu_source`` and
  ``critic_gravity_source`` arguments of ``agibot_x2_flat_velocity_env_cfg`` and the
  task-local ``imu_1`` gyro, velocimeter and ``framezaxis`` sensors. Nine tracking
  ablations are gone: ``-Correlated-DR``,
  ``-Correlated-DR-Reduced-Perturbations-Standing-Start``, ``-Increated-Perturbations``,
  ``-Increated-Perturbations-Vendor-Gains-No-PD-Gain-DR``,
  ``-Correlated-DR-Reduced-Perturbations-{Projected-Gravity,Projected-Gravity-And-Anchor,Vendor-Velocity-Scaling,Pelvis-Anchor,Torso-IMU}``.
  Two factories left with them:
  ``agibot_x2_flat_tracking_observation_ablation_env_cfg`` and
  ``agibot_x2_flat_tracking_increated_perturbations_vendor_gains_no_pd_gain_dr_env_cfg``.
  The ``imu_source`` argument, the ``X2TrackingImuSource`` and
  ``X2TrackingObservationAblation`` aliases and the task-local ``imu_1`` sensors are
  gone from the tracking factories too. The tasks that stay registered in these
  families are the base velocity task, the base tracking task and its
  no-state-estimation variant, ``-Correlated-DR-Reduced-Perturbations`` (the
  distillation ``base_task``), ``-Increated-Perturbations-Vendor-Gains`` (the
  multi-motion teacher) and ``Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion``. The
  factories those remaining tasks build on stay, so
  ``agibot_x2_flat_tracking_correlated_dr_env_cfg``,
  ``agibot_x2_flat_tracking_standing_start_env_cfg`` and
  ``agibot_x2_flat_tracking_increated_perturbations_env_cfg`` still exist. A deleted
  task ID can no longer be loaded, so the runs under its
  ``logs/rsl_rl/<experiment_name>`` directory can no longer be resumed or played from
  the CLI; the checkpoint files are not deleted. Measurements of the removed
  ablations stay in ``docs/source/sim2real_domain_randomization.md``, marked as
  removed.

Added
^^^^^

- **Target-height juggling family, S1 and S1a.** Two tasks,
  ``...-Ball3-Pelvis-Nominal-S1-TargetHeight`` and ``...-S1a-TargetHeight``, take the
  target apex height as an **input** instead of a reward constant. A new command term
  (``TargetHeightCommand``) holds one height per environment in metres and resamples it
  on every episode reset; a new ``target_height`` observation is appended last to both
  groups, divided by 0.40, and is not corrupted because a goal is not a measurement.
  Five consumers read the command instead of a constant: the ``ball_energy`` target and
  sigma (3.75 J/kg per m, so the tolerance stays a constant fraction of the target), the
  ``ball_not_lifted`` and ``ball_first_lift_s`` thresholds (2.5 J/kg per m, so a low
  target cannot classify a perfect juggle as "not lifted"), and the two energy metrics.
  The ball's spawn scales as ``0.20 * h / 0.40``. S1 fixes the target at 0.40 m and is
  the shipped ``s1-15hz`` condition with every rule a no-op at that height; S1a draws
  uniformly from 0.05 to 0.40 m and resumes S1. The actor widens 112 to 113 and the
  critic 121 to 122 with 29 actions, so no ladder checkpoint resumes strictly, and the
  controller's 112-wide path does not apply. ``MJLAB_JUGGLE_TARGET_HEIGHT_RANGE_M``
  (``H`` or ``LO,HI``) pins the range for an acceptance cell, and the target is logged
  as the command metric ``Metrics/target_height/target_height`` so a run records the
  heights it asked for. A CPU probe
  (``scripts/experiments/x2_target_height/probe_target_height.py``) checks all seven
  properties the plan asks for, including that the 0.40 m reward equals the old
  constant-0.40 code path at 35 ball states (exactly in float64, to one float32 ulp of
  the target energy in the shipped dtype), and that the reward peaks at the commanded
  apex from a physics-derived energy. What is not claimed: that a policy uses the
  input. The commanded-against-achieved apex table needs a trained run and is out of
  scope. See ``docs/design/x2_tennis_target_height_plan.md``.

- **Carry-pose velocity task.**
  ``Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-CarryVelocity`` (factory
  ``agibot_x2_flat_velocity_carry_env_cfg``) uses the tennis robot in
  ``BAT_CARRY_POSE`` with the welded racket. It keeps the ``pose`` reward over all 31
  joints and adds a quadratic arm/waist term on top, and uses carry-pose command ranges
  with no curriculum. It starts from scratch because ``default_joint_pos``
  changes the meaning of a zero action. See ``docs/design/x2_carry_velocity_plan.md``.

- **Nominal 15 Hz stage.**
  ``Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF-Ball3-Pelvis-Nominal-S1-15Hz``
  keeps the nominal pilot and moves one input condition: the ball history arrives at
  15 Hz with 0-40 ms of uniform transport delay. Three samples are still stored, so the
  actor stays 112 wide and an earlier checkpoint resumes with its optimizer state. It is
  the first rung of a staged fine-tune: one input condition per rung, each starting from
  the previous rung's checkpoint.

- Added a standalone nominal juggling evaluator under
  ``scripts/experiments/x2_tennis_nominal_eval/``. It records first-episode
  trajectories and substep contact histories for six policies and two zero-action
  controls. Synthetic scoring, recorder and launch-guard checks passed.
  The `completed E0 evaluation <../reports/x2_tennis_nominal_evaluation_20261105.md>`_
  accepted all three 0.40 m policies: 34 consecutive qualified flights in each of
  96 nominal simulation episodes. The 0.20 m policies carried the ball; controls lost it.
  The batch is separate from training and provides no hardware validation.

- Added the `six-run nominal X2 juggling comparison
  <../reports/x2_tennis_nominal_comparison_20261105.md>`_ and an
  `evaluation-gated curriculum plan <../design/x2_tennis_nominal_curriculum.md>`_.
  The analysis compares three seeds at each target. Training curves do not verify
  consecutive juggles. No additional training or policy rollout was started.

- Added ``Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF-Ball3-Pelvis-Nominal``.
  This pilot uses zero movement commands, no pushes or physics randomization,
  noise-free ball history, and centred drop starts. Physical limits and old task
  defaults are unchanged. See :doc:`x2_tennis_ball_juggle` and
  ``docs/design/x2_tennis_nominal_pilot.md``.

- Added ``Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF-Ball3-Pelvis``.
  Its ``ball_hist_pelvis`` actor term stores three capture-time pelvis-relative positions
  and their ages. It reuses the existing stream without an estimator or proprioception
  history. Old tasks are unchanged. Controller support is not implemented; learning
  improvement is not tested. See :doc:`x2_tennis_ball_juggle`.

- Added ``Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF-Ball3``,
  the 29-DOF juggling task whose actor observes **three timestamped ball samples** instead
  of one delayed position: 12 numbers, ``[x, y, z, t]`` per slot, newest first, ages in
  seconds clipped at 1.0 s. Widths 29 / 112 / 121. The term models the estimator's own
  chain (5-20 Hz schedule, 80-250 ms latency, stalls, per-episode bias, per-sample noise)
  rather than applying a fixed delay, so latency becomes an observable input instead of a
  trained assumption; the two robots publish at 18.44 Hz and 5.40 Hz with different
  latencies. Slots are ordered by capture step, duplicates repeat the oldest sample with
  its own age, and the term counts its own control steps because rsl_rl randomises
  ``episode_length_buf`` at the start of every ``learn()``. Measured on CPU: in single-arc
  triples the three-sample fit extrapolates to a median 69 mm against 472 mm for the
  freshest sample alone, but only 69 of 740 triples were single-arc, so the gain is
  conditional. Not supported by the deployed controller. See
  ``docs/source/x2_tennis_ball_juggle.rst``.

- Added ``Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF``,
  the juggling task with the two head joints removed from the action space: 29
  actions and actor/critic widths of 103/121 instead of 105/123.

  The head is fragile on the real robots and regularly breaks. Measured on two of the
  three machines we can reach, one head axis does not follow its command at all: on
  ``.147`` ``head_yaw_joint`` reports ``error_code 1027`` with its position frozen at
  ``+3.141590118``, and on ``.132`` the same joint accepts 2-7 N.m of command and
  delivers ~0 N.m, moving 0.057 rad while its target travelled 1.400 rad. Head pitch
  on ``.132`` tracked its target to 0.014 rad in the same run, so the fault is the
  joint, not the command path.

  The joints stay in ``joint_pos``/``joint_vel`` because the camera rides on the head
  and its measured pose feeds the camera-to-plate conversion. Their pose is instead
  randomized once per episode over the feasible band, and held there for the whole
  episode, so one policy covers robots that park the head differently: measured
  offsets from the trained default are -0.349 rad (``.132`` yaw), +0.450 (``.132``
  pitch) and +0.217 (``.236`` yaw). ``head_joint_motion`` is dropped, because the
  policy can no longer influence it.

  Three details are load-bearing. Holding the sampled pose needs the **actuator
  target** written as well as the joint state: the head is outside every action term,
  so the reset zeroes its target and nothing rewrites it, and a state-only write is
  undone by the PD within a few steps. The sample is uniform over the *feasible* band
  rather than an offset about the default followed by a clamp, because the model's
  soft range for head yaw is narrower than a +-0.5 rad offset and clamping would pile
  half the environments onto one number. The scale map is keyed by joint-name pattern
  and every key must match, so the two head entries have to be dropped with the
  joints or the build fails with "Not all regular expressions are matched".

  The exported ONNX metadata gains ``action_joint_names``: ``joint_names`` is the
  entity's 31-joint list that the observation refers to, and it no longer equals the
  action vector, so a consumer must map actions through the new field. See
  ``docs/source/x2_tennis_ball_juggle.rst``.

- Added ``Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Balance``,
  the X2 no-state-estimation velocity task with the tennis equipment in the
  scene. The racket assembly (white wrist connector, clamp, racket) is rigid
  tooling on the robot: it replaces the right hand and rides
  ``right_wrist_roll_link`` as two fixed bodies, adding 0.37 kg and no joints or
  actuators. The 0.0560 kg tennis ball is a separate floating entity spawned 1 mm
  above the string bed. Meshes are byte-identical to the ``x2_references``
  tennis snapshot, with SHA-256 pins.

  Three properties are load-bearing and each is tested. The string bed is named
  ``string_bed`` so it escapes ``FULL_COLLISION``'s ``condim=1``, ``priority=0``
  catch-all, and a second ``CollisionCfg`` with ``disable_other_geoms=False``
  re-enables it after that pass zeroes unmatched geoms. The bed's negative
  ``solref`` is a 1 ms calibration, so the task runs at 1 ms with 20 substeps
  (50 Hz control) instead of 5 ms with 4; at 5 ms a 0.30 m drop rebounds to
  0.635 m instead of 0.23 m. The robot and ball resets are pinned so the ball
  stays on the bed. The four recorded rebounds reproduce within 20 mm at 1 ms
  with zero lateral drift.

  The ball is **not observed and not rewarded** yet; observations and the
  holding reward are a separate change because they alter the actor input
  width. In-environment retention is therefore **not tested**: open-loop PD
  cannot hold the bat-carry pose, so a trained policy is required. CPU tests and
  one real environment build pass. See ``docs/source/x2_tennis_ball_balance.rst``.

  The task runs the multi-motion tracking task's servo and sim-to-real
  configuration rather than the velocity task's. ``apply_vendor_rl_gains``
  installs the vendor reference-policy kp/kd; joint ranges, effort limits,
  armature, friction loss and action scale were already identical between the two
  tasks. In the bat-carry pose that cuts the static string-bed tilt from the
  bat's weight from **19.1 to 12.9 deg** and raises the shoulder-pitch damping
  ratio from 0.19 to 0.44 — the underdamped shoulder was ringing after a ball
  impact. The six tracking physics DR axes (inertia, armature, effort limits,
  joint friction, joint damping, foot size), the wrist-only effort derating and
  a shared per-environment PD-gain scale are adopted too; observation delay is
  not, because it changes the actor's input timing contract. The tracking task's
  X2-specific DR wiring moved into ``apply_x2_tracking_robot_randomization`` so
  both tasks call one definition, and the tracking tasks' events are unchanged
  by that extraction. Terms are pinned against the tracking task by test.

  Two head-related terms were added for the perception chain. ``ball_in_fov``
  (+0.2) pays for the ball staying inside the head stereo pair's field of view,
  which is where its position comes from on hardware; ``head_joint_motion``
  (−0.05) penalises squared head joint velocity so the field-of-view reward cannot
  pay the policy to hunt with its head. The camera geometry is approximate on
  purpose: the reference snapshot's robot-162 mount, a ROS optical frame (+z
  forward, not MuJoCo's −z), and equidistant half-angles of 85.4° × 64.0° from the
  live stereo-left calibration. Measured, the ball sits at 24° off the optical axis
  while carried and stays in frame through a metre of lateral travel, so
  ``ball_in_fov`` is a safety term rather than a shaping term.

  The tennis task now has its own training procedure. The base task's command
  curriculum is replaced by this task's stages: 0.3 m/s from the start, 0.6 m/s at
  iteration 2000 and 1.0 m/s at iteration 5000 (the base task's own stage 0).
  **The difficulty advances on the step count, not on performance.** Two
  performance-gated designs were measured and rejected: with a stand-still first
  stage the ball simply parks on the plate and every environment times out at 20 s
  together, and without it an untrained policy already survives 9 s at 0.3 m/s and
  16 s at 0.6 m/s while managing 1 s at 1.0 m/s, so any survivability threshold in
  the easy regime is cleared with no skill at all. Performance is therefore used
  only as a veto: a stage is entered on its scheduled step only if the median
  length of the last 100 finished episodes is at least 8 s.
  Losing the ball no longer ends the episode immediately: the ball counts as lost
  at the same 20 mm below the plate, but the episode ends only after it has stayed
  lost for 25 control steps (0.5 s), counted consecutively and restarted if the
  ball comes back. Episodes go from 0.55 s to about 1.05 s, against a 20 s
  allowance.

  The tennis task now teaches the ball. The policy observes ``ball_pos`` and
  ``ball_vel``, two 3-vectors in the plate frame (actor 102 -> 108, critic
  117 -> 123; a new experiment name, ``agibot_x2_velocity_tennis_ball_centre``,
  because the observation normalizer width changed). It is paid to keep the ball
  within a 20 mm plateau of the plate centre with a 40 mm Gaussian tail, and it
  gets a small term against throwing the ball more than 100 mm above the strings.
  The episode ends when the ball drops 20 mm below the plate. Each of the six
  ball-observation dimensions is randomized independently with its own uniform
  half-width (±5, ±5, ±8 mm; ±0.15, ±0.15, ±0.20 m/s), because the perception
  program's depth error and its differenced-velocity error are not alike. Ball
  velocity is
  measured against the plate's material point at the ball, not the racket body
  origin, so a still ball on a turning wrist reads zero; the ``v + ω × r`` form is
  checked against MuJoCo's ``site_vel_w`` to 1e-5.

  The bed–ball contact is now six-dimensional. ``CONTACT_CONDIM = 6`` adds
  torsional (spin) friction and two rolling dimensions to the three the contact
  had before; ``condim=3`` carries no rotation at all, which is why a ball
  rolling on the level bed kept exactly the same speed forever. Friction is now
  the documented triple (sliding 0.44, spin 0.005, rolling 1.7185e-03): the
  sliding value is **measured** for a tennis ball on a string bed and replaces
  the old ball-on-court 0.6, the spin value is a carried-over provisional number
  that is only now active, and the rolling value is **fitted** to a measured
  slope run of the real ball on the real racket. That run — 0.245 m in 0.834 s
  from rest at 0.129 rad — gives a dimensionless rolling resistance of 0.0174,
  which is a 0.045 m coast for a 0.10 m/s release on a level bed, which in turn
  is what ``scripts/experiments/x2_tennis_roll_friction_calibrate.py`` inverts
  to 1.7185e-03. The ball's moment of inertia sets that conversion and cannot be
  separated from it by the slope run: a solid sphere would imply 0.025 m and a
  thin shell 0.096 m. The ball body therefore carries its inertia explicitly, at
  the shell ratio I/(m R²) = 0.549 against the 0.400 of a uniform solid sphere,
  with its geom mass set to zero so no kilogram is counted twice. The ball is
  sized to the ITF lower limit, 65.4 mm and 56.0 g, because the operator could
  not measure the real ball's diameter, mass or inertia to better than a ruler. The normal direction (``solref``, ``solimp``) and the 1 ms step are
  untouched on purpose, so the four recorded rebounds still reproduce and the
  simulated coefficient of restitution moves by less than 0.003 at every drop
  height. Measured in an isolated level-bed fixture, a ball released rolling
  without slipping at 0.10 m/s keeps 0.56 of its speed after one second, and
  1.00 of it with the rolling coefficient zeroed.

- Added ``Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle``,
  which asks for a ball that bounces on the plate and returns to an apex about
  300 mm above it, repeatedly. The task is a **copy** of the carrying task with a
  different ball objective, so the carrying task is unchanged; the robot, racket,
  ball, contact model, observations and terminations are its.

  Height cannot be paid per control step. A ball parked at the apex height would
  collect a per-step height reward for as long as it stayed there, which is the
  carrying behaviour this task exists to replace, so ``ball_bounce_apex`` pays
  **once per bounce** - a Gaussian in the apex clearance, centred at 300 mm with an
  80 mm sigma - at the instant the ball's *world* vertical speed changes sign
  (gravity defines an apex, not the tilting plate normal). A crossing counts only
  if the apex clearance is at least 30 mm and the peak rising speed of that rise
  was at least 0.30 m/s, so a resting ball's jitter cannot pay. Two details are
  load-bearing. The speed gate uses the **peak** speed of the rise, because the
  previous control step's speed is capped by ``g * dt`` = 0.196 m/s and a 0.30 gate
  on it could never open. Clearance is the **gap under the ball**, not the height
  of its centre, because a resting ball's centre already sits one radius above the
  plate, so a 30 mm resting threshold measured from the centre would never fire.
  ``ball_at_rest`` charges a ball parked low and slow for 0.5 s, which removes the
  "park it and stand still" fallback, and ``ball_centred`` is kept at a reduced
  weight so the ball must still land on the strings. ``ball_height`` is gone: it
  was flat to 100 mm and paid for a parked ball.

  Version 1 juggles **standing still**: the twist command is pinned to +-0.10 m/s
  and +-0.05 rad/s and the command curriculum is absent. Walking while juggling is
  a later step. ``Episode_Metrics/ball_bounce_count`` is the acceptance metric,
  because a reward can be gamed and a counter cannot. It reads the reward term's
  state, so the apex weight must stay non-zero or the count goes stale.

  A 300 mm apex costs almost nothing in field of view: measured on the built task
  at the carry pose, the ball sits at 0.434 of the camera half-frame against 0.271
  at rest, and the field-of-view reward stays exactly 1.000. That reward starts to
  fall just above a 0.41 m gap, so 300 mm is comfortable and 400 mm is about the
  practical ceiling for full credit.

  In 100 iterations at 4096 environments the bounce count first falls from 1.09 to
  0.04 and then rises to 1.73 per episode, while the apex quality is still under
  1 % of a perfect apex, so the height target is not learned yet. **That is not
  evidence of juggling.** Tests drive the terms with scripted ball states; no
  physical bounce and no trained policy were validated. See
  ``docs/source/x2_tennis_ball_juggle.rst``.

- Added the offline joint-pose rollout tool at
  ``mjlab.scripts.joint_pose_rollout``. It reads the window geometry, denoiser
  sizes and noise scale from one checkpoint, rolls one reference clip into a
  long generated series in chunks of ``window - past - 1`` frames, and writes a
  ``generated``/``control`` motion pair plus a JSON record of per-chunk
  diagnostics. The generated file takes its body channels from the recorded
  clip rows for the same absolute frames, because the 68-channel window carries
  no root translation. Generation is open loop: no simulator and no robot is
  involved. ``--keyframes current,last`` pins the anchor row and the window's
  last row to the reference, which bounds heading drift at the cost of a
  measured seam at every chunk.
  See ``docs/plans/x2_joint_pose_flow_rollout_eval_plan.md``.

- Added ``scripts/play_reference.py``, a kinematic playback of one tracking
  reference. It writes the file's root pose and joint angles into the X2 model
  every frame and renders, with no physics, no policy and no episode reset, so
  inspection does not stop when a reference would make the robot fall. Playback
  repeats until the window closes and the camera tracks the pelvis. Before
  playing it verifies that forward kinematics reproduces the stored body poses
  and refuses a file that never matches, which separates a scrambled joint order
  from a generated file whose body channels are the recorded clip's.
  ``--check-only`` prints the check and exits.

- Added the opt-in raw X2 joint-pose collector at
  ``scripts/experiments/joint_pose_collection/collect.py``. It rolls out the
  frozen multi-motion PPO teacher on explicit reference clips and writes
  achieved root quaternion (world frame, ``wxyz``), joint position and joint
  velocity with per-trajectory boundaries. It restores the checkpoint's saved
  preview, reads terminal state before manual reset, and refuses an existing
  output directory or non-finite recorded poses. This is raw data collection
  only: no heading alignment, 6D packing, VAE, flow model, reference adapter,
  or controller change. See ``scripts/experiments/joint_pose_collection/README.md``.

- Extended the raw X2 joint-pose collector with a finite multi-seed budget, a
  start/uniform reference-phase option, opt-in interval root-velocity pushes,
  and separate baseline/disturbed scenario labels. Collector schema version 2
  adds per-trajectory run, seed, initial reference frame, scenario, and push
  count/tick evidence. Baseline still removes ``push_robot``; the disturbed
  option reuses the task's existing ``push_by_setting_velocity`` term and
  records configured ranges, not force magnitudes. The default CLI remains one
  seed, start phase, and no pushes.

- Added a non-destructive eligibility gate for raw X2 joint-pose schema2
  collections at ``scripts/experiments/joint_pose_collection/gate.py``. It
  writes a hash-bound JSON index and a retained/rejected report, and never
  edits raw data. A trajectory is kept only when its stored outcome is
  ``complete``, its root quaternion and joint position/velocity samples are
  finite, and it has at least one complete 41-row window (8 history, current,
  32 future). Failed, budget-stopped, non-finite, and too-short trajectories
  are rejected whole; push exposure is not required. No windows are
  materialized, normalized, or wired into training. See
  ``scripts/experiments/joint_pose_collection/README.md``.

- Documented the two-GPU X2 shared-teacher test at
  ``docs/reports/x2_multi_motion_ppo_2gpu_8192env_100iterations.md``. Two ranks
  on one container train 8192 environments each over 148 clips for 100
  iterations. The run is better than the single-GPU reference on every logged
  scalar, and the episode-length collapse reverses. Batch size, GPU count and
  hardware changed together, so the result is not a scaling measurement.

- Documented the existing multi-GPU scaling path for the X2 shared teacher in
  ``docs/plans/x2_multi_motion_ppo_teacher.md``. Environment counts are per GPU;
  gradients and observation statistics synchronize across workers. The task's
  multi-GPU runtime and global reference-frame coverage remain unvalidated.

- Documented the all-tennis X2 PPO test at
  ``docs/reports/x2_multi_motion_ppo_all_tennis_100_iterations.md``. All 148
  references run with 4096 environments for 100 iterations. The reward curve
  becomes less negative, but episodes shorten and height-error terminations
  increase. The report records this limit; teacher quality is not qualified.

- Added the bounded real-GPU X2 PPO smoke driver at
  ``scripts/experiments/x2_multi_motion_ppo_smoke/run.py``. It checks two clips,
  8 environments, 10 PPO updates, exact preview, endpoint/subset resets, and
  checkpoint reload. The local smoke passes; tracking quality is not tested.
  See ``docs/reports/x2_multi_motion_ppo_teacher_gpu_smoke.md``.

- Added ``Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion`` for one shared PPO teacher.
  It reads a motion-list YAML, uses the SCDP six-layer actor/critic shape, and
  gives both networks a configurable future-reference sample (default 0.2 s).
  Clip completion ends the reference task without a wrap teleport. Local
  playback and per-clip evaluation support the task. Ten critical tests cover
  clip isolation, preview, reset boundaries, and evaluator attribution.
  Software checks and a bounded real-runner smoke pass; policy quality is not tested.
  See ``docs/reports/x2_multi_motion_ppo_teacher_implementation.md``.

- Added the lean X2 multi-motion PPO teacher plan at
  ``docs/plans/x2_multi_motion_ppo_teacher.md``. It uses the SCDP six-layer
  network, one future-reference sample, and existing reference components.
  Tests cover only critical clip, preview, episode-boundary, and task behavior.
  Implementation, training, VAE changes, and deployment remain out of scope
  for this planning request.

- Documented the agreed X2 general-teacher and goal-conditioned latent diffusion
  design in ``docs/design/x2_general_teacher_goal_conditioned_latent_diffusion.md``.
  It puts reference preview in the VAE encoder and keeps the decoder reference-free.
  The note separates SCDP paper evidence, proposed settings, and open interfaces.
  It does not authorize implementation, training, or deployment.

- Added an opt-in local-kinematic state ablation for X2 diffusion. The new
  contract
  ``docs/plans/beyondmimic_diffusion_d1_no_mixtures_local_kinematic_contract.yaml``
  removes ``root_position`` [3] and ``root_linear_velocity`` [3] from the
  135-D hybrid state. It keeps ``root_rot6d`` [6], ``root_angular_velocity``
  [3], ``body_positions`` [60] and ``body_linear_velocities`` [60]. The raw and
  projected state is 129-D and the token is 161-D. This is not an absolute
  state: translation is omitted on purpose and the body terms stay
  pelvis-relative. The training input
  ``configs/diffusion/x2_50hz_train_local_kinematic.yaml`` and the collection
  input ``configs/diffusion/x2_50hz_collection_local_kinematic.yaml`` bind the
  ablation to that contract. The diffusion trainer, denoiser settings, token
  records and token cache now follow the contract's declared dimensions, so
  the 135-D defaults are unchanged and a 135-D normalizer, checkpoint or cache
  is refused for the 129-D contract. The 135-D contract identity and defaults
  are unchanged.

- Added looped playback to the one-shot future-pose viewer. It starts paused
  on pose 1 and has Play/Pause, Step, Reset and playback-speed controls.
  Playback changes only the displayed saved pose. It runs no physics,
  neural inference or IK. Normal ``--execute`` generates the result and opens
  this playback viewer in one command. The 32-pose ghost overlay remains available.

- Added the one-shot D3 future-pose inspection tool
  ``scripts/experiments/diffusion_d3_closed_loop/inspect_sequence.py`` and its
  display-only pose helper
  ``scripts/experiments/diffusion_d3_closed_loop/sequence_poses.py``. It executes
  eight real VAE bootstrap transitions, draws one complete window from the
  installed plan and sampler, recovers the 32 predicted future world poses and
  fits display-only joint angles with an isolated CPU MuJoCo model, then shows
  one static Viser scene with minimal controls. ``--replay DIR`` reopens saved
  poses without a neural artifact or a physics step, and ``--no-view``
  generates headlessly. The tool never decodes the diffusion action and never
  steps physics after handover. It is a single-sample visualization and makes no
  matched-evaluation, policy, deployment or hardware claim.

- Added the X2 VGK8 diffusion training pipeline runbook at
  ``docs/source/x2_diffusion_training.rst``. It documents the D1 collection,
  inspect/replay, grouped windowing, train-only statistics, token-cache,
  bounded D2 training, and offline evaluation commands. It records
  ``max-envs 64`` with one worker and one CPU thread per OpenMP and MKL pool as
  the best tested collection setting on the managed RTX 4090 server. This is a
  throughput practice, not a policy-quality or deployment claim.

- Added an opt-in VGK8 diffusion history dropout ablation. The training arm
  drops both state and latent values in the eight history rows with a configured
  probability, using independent ``K=1000`` diffusion noise. The default is
  zero. The D1 contract, dataset identity, token caches and model shape remain
  unchanged. The offline evaluator accepts
  ``--history-condition-dropout-probability`` for matched missing-history
  stress evaluation. This evidence does not claim closed-loop, hardware or
  deployment readiness.

- Difficult-state coverage is now **enabled by default on the cohort path**.
  ``distill train --teacher-ids ...`` selects ``difficult-state-coverage`` when
  ``--coverage-policy`` is omitted, while a single-teacher run resolves to
  ``off`` because it cannot carry the policy. An explicit value always wins, so
  ``--coverage-policy off`` still disables coverage and reproduces or resumes a
  coverage-off run. The fail-closed restore-parity gate is unchanged: a cohort
  launch without ``--coverage-parity-evidence`` is refused before any
  environment is constructed, and the refusal names the remedy.

- Added command-local reuse of validated motion-reference indices. Reuse ends
  when motion/frame tensors are replaced or their version counters change;
  inference tensors use the validating uncached path. Public library queries
  still validate every call. Per-motion collection counters now defer report
  scalar transfers while preserving pre-step mapping checks and owned metadata.

- Added configurable collection-linked VAE distillation budgets.
  ``distill train`` can derive environment counts from selected motions,
  optimizer updates from a draw-to-collection ratio, and replay capacity from
  a retention horizon. Reports and checkpoints include resolved totals and
  live per-motion allocations. Explicit legacy totals remain supported;
  conflicting explicit and derived settings fail before construction.

- Added the bounded D5 numerical/export parity gate under
  ``scripts/experiments/diffusion_d5_parity/``. It checks the active
  135-state/32-latent/167-token and 99-conditioning/31-action contract,
  statistics-only normalization, clean-mask semantics and 20-update DDIM
  output, then exports and reloads ONNX denoiser and VAE-decoder artifacts for
  CPU FP32 comparison. It is one-time frozen-artifact evidence and does not
  claim simulator, deployment, quality or hardware readiness.

- Added the opt-in D5 controller lifecycle and execution bridge in
  ``diffusion/controller.py`` and ``diffusion/controller_driver.py``. The bridge
  preserves eight acknowledged measured bootstrap pairs, performs bounded
  owner-decoder and spawned-planner warmup, requires explicit activation,
  decodes only finite 31-D actions from selected raw 32-D execution rows, and
  commits history only after literal-``True`` execution acknowledgement. The
  CPU integration test uses a real ``BoundedPlannerExecutor`` and the actual
  ``LatentPlanner.process`` callback with synthetic statistics, denoiser and
  decoder inputs. It does not claim 50 Hz timing, physics, export, deployment,
  simulator or hardware readiness.

- Added the CPU-only D5 asynchronous latent runtime facade. It preserves the
  eight measured VAE bootstrap pairs, submits bounded raw 32-D latent planning,
  selects the execution-tick row, decodes it with current finite 99-D
  conditioning, and records state/latent history only after execution
  acknowledgement. The runtime checks originating request provenance and exact
  executor cleanup evidence. Stale, future, incompatible, exhausted and
  non-finite plans fail closed with no action. This integration evidence does
  not claim D5 timing, simulator, export, deployment or hardware readiness.

- Added the concrete D5 model-backed runtime assembly in
  ``diffusion/model_backend.py``. It loads the existing denoiser, projection and
  gravity-mode VAE interfaces, creates the denoiser inside the bounded spawned
  planner worker, decodes only the execution-tick raw 32-D latent with current
  99-D conditioning, and exposes a direct ``build_model_backed_runtime``
  entrypoint. Paper-reproduction loading no longer requires contract-file or
  artifact SHA values supplied by the caller; schema, dimensions, finite values
  and model interfaces remain checked. This is CPU/software wiring only and
  does not claim model timing, simulator, export, deployment or hardware
  readiness.

- Changed diffusion contract loading for reproduction speed. Source YAML and
  artifact hashes remain descriptive metadata but are no longer registry or
  semantic-identity gates. The active schema and 135-state/32-latent/167-token
  dimensional checks remain enforced.

- Added an offline ``distill coverage-inspect`` contract/status surface and a
  family-disjoint fixed-state difficult-state fidelity evaluator. It reports
  teacher-label action RMS/p95 and optional target-position metrics without
  constructing physics, updating normalizers, launching training, or including
  the held-out hardware fixture. The enabled MVP collection path requires a
  passing restore-parity evidence file; multi-step branch continuation remains
  deferred.

- Added the MVP difficult-state coverage execution path. ``distill train`` now
  accepts ``--coverage-policy difficult-state-coverage`` and a matching
  ``--coverage-parity-evidence`` report. The one-step profile captures physical
  terminal-predecessor rows before reset, stores cached 68/99 inputs and routing metadata,
  obtains one frozen-teacher label from the cached 164-D actor view, inserts the
  resulting row into ordinary replay, reports capture/label/insertion counters,
  and preserves the unchanged VAE objective. Branch continuation, runtime
  coverage perturbations and protected replay quotas remain deferred.

- Fixed the CPU difficult-state coverage MVP capture path. It uses an explicit
  policy-input snapshot instead of touching unsupported CPU Warp vector fields.
  Full physics snapshot restoration remains reserved for the deferred branch
  path.

- Added the D1 collection and dataset throughput integration. The statistics fit
  now reads train-only moments in one segment traversal with a bounded jobs
  partition; the storage layer adds a byte-preserving validated shard merge and
  bounded shard-level inspect/replay; the runtime adds a device-native VAE
  tensor path; and the collector adds bounded streaming publication, progress
  records and enforced wall, inactivity and output limits. The CLI exposes the
  worker counts for ``inspect``, ``replay``, ``build`` and ``stats``, and a
  persisted window index can be reused instead of rebuilt; a foreign index is
  refused with a message and exit status 1. The new
  ``scripts/experiments/diffusion_d1_throughput/prepare_dataset.py`` entry point
  validates worker reports, merges, builds the index once and prepares token
  caches without starting training; ``benchmark_matrix.py`` generates a matched
  CPU/GPU VAE and 64/128/256-environment benchmark plan. Managed-server tests
  measure GPU256 collection at 1.53x original throughput, shard merge at 3.14x,
  and eight-worker statistics at 4.85x. Eight-worker token caches are 3.90x
  slower; cache preparation stays at one worker. Collection CLI defaults now
  use 256 environments, streaming and compact counts, with compatible tensor
  dispatch. Batched X2 VAE inference follows simulation and pins IEEE FP32 for
  the validated CPU/CUDA parity. Scalar and explicit CPU/NumPy paths remain.
  Offline validation/index/statistics default to eight bounded workers and
  512-window statistics batches. Existing matching window indices are reused.

- Added the opt-in X2 x0 flow-matching family under
  ``src/mjlab/tasks/tracking/flow_matching/`` and
  ``src/mjlab/scripts/flow_matching.py``. It uses the linear Gaussian path
  ``x_s = (1 - s) * x0 + s * e`` with ``s = k / max_level``, independent integer
  levels per row and per state/latent slice, a plain channel-mean x0 MSE, and an
  x0 Euler sampler that descends from level 1000 to level 0. It reuses the
  unchanged ``StateLatentTransformer``, the D1 read-only token caches and the
  frozen VAE, writes flow-format checkpoints, and never reads or writes a
  diffusion schedule or checkpoint. History dropout is supported and matches
  the diffusion meaning. The new runbook is ``docs/source/x2_flow_matching_training.rst``. This
  is software validation on synthetic CPU fixtures; no training run, simulator,
  controller or hardware use is claimed.

- Added ``--model-family diffusion|flow`` to the D3 closed-loop evaluation
  runner, the parallel driver and the visual diagnostic. The default stays
  ``diffusion``. A flow run loads a flow checkpoint through the flow loader,
  injects the flow Euler sampler into the existing single-row and batched
  continuations, and rejects D4 task guidance during preflight. A flow
  checkpoint sidecar uses the schema
  ``mjlab-d3-flow-matching-checkpoint-v1``; a sidecar/family contradiction
  fails closed. See ``docs/source/x2_flow_matching_training.rst``. This entry is
  CPU software validation with fake loaders: no flow checkpoint, simulator or
  GPU run is claimed.

- Added ``--flow-euler-steps N`` to the same runner, driver and visual
  diagnostic. It sets the flow Euler ODE step count at inference time; the
  count must not exceed the checkpoint process ``max_level``. The report always
  carries the grid actually built. The diffusion family keeps its frozen 20 DDIM
  updates and refuses the flag.

- Changed the closed-loop flow default to **6 Euler steps**
  (``DEFAULT_CLOSED_LOOP_SOLVER_STEPS`` in ``diffusion/replay_integration.py``),
  applied by ``load_flow_denoiser`` and by ``--flow-euler-steps``. Six steps is
  inside the flat closed-loop quality region (6 to 20 steps are statistically
  indistinguishable over 2048 rows per arm; 5 steps is measurably worse), and on
  the X2 robot PC with the TensorRT execution provider and a locked GPU clock it
  is 15.82 ms median against a 20 ms control period, where 10 steps is 24.92 ms
  and 20 steps is 56.83 ms. The flow package's own ``DEFAULT_CALLS`` stays 20
  because it is the offline generation-scoring grid.

- Added the reference-free deploy export for the X2 shared multi-motion teacher.
  ``MultiMotionTrackingOnPolicyRunner``
  (``src/mjlab/tasks/tracking/rl/multi_motion_runner.py``) and
  ``scripts/export_x2_multi_motion_policy.py`` write the raw ``obs -> actions``
  graph with the observation normalizer baked in and the deterministic mean as
  the output. The graph embeds no reference table, so one policy can replay a
  clip the teacher never trained on. The deployed artifact is local and
  git-ignored: ``logs/diagnostics/x2-multi-motion-deploy-20261015/policy.onnx``,
  15909634 bytes, SHA-256
  ``5e6b7c74fdf820024ffc821ffbda9dd5ccc7de944988d91d6d1f031f8fedca1f``, exported
  from ``model_19999.pt``. A torch-versus-onnxruntime parity check runs inside
  the export script. Controller acceptance is not tested, and the X2 loader
  still requires the embedded-reference interface.

- Added the preview-offset metadata key ``motion_lookahead_steps``
  (``src/mjlab/tasks/tracking/rl/export_metadata.py``). The X2 controller
  compiles an observation plan from the export metadata and derives every term
  width itself, but the term name ``motion_lookahead`` is present for a disabled
  preview and an enabled one alike, so the term list cannot distinguish them.
  The export therefore states the offset, and an absent key means a disabled
  preview, which keeps every export written before this key valid. Training
  computes ``max(1, ceil(lookahead_s * fps))``; the export mirrors that formula
  exactly, and a multi-motion command whose rows disagree is refused instead of
  averaged.

- Added ``scripts/export_x2_reference_bundle.py`` and
  ``src/mjlab/tasks/tracking/rl/reference_bundle.py``, which convert motion npz
  clips into the controller's existing ``mjlab-vae-motion`` version 2 JSON plus
  a manifest carrying source digests. Only reference tables are written, never
  weights. The writer fails closed on a frame rate other than exactly 50 Hz, a
  body or joint count mismatch, a non-finite row, a non-unit anchor quaternion,
  a missing required array, an unsafe clip name and a duplicate clip name.

- Added 23 focused tests for the export metadata and the reference bundle. One
  test rebuilds the tables of a deployed bundle and compares ``q``, ``dq`` and
  ``anchor_quat_wxyz`` with tolerance zero, so the writer cannot drift from the
  format the controller already accepts. Student quality, distillation and
  hardware behaviour are not tested here.

- Added the version-2 reference layout for a preview-dependent teacher
  (``PREVIEW_SCHEMA_VERSION`` in
  ``src/mjlab/tasks/tracking/distillation/vae_config.py``). It appends one
  future-reference frame of joint positions and velocities, so the encoder
  reference grows from 68 to 130 while the decoder conditioning stays at 99 and
  the decoder still receives proprioception only. The version-1 layout stays the
  default and is unchanged, so existing checkpoints, normalizer statistics and
  installed bundles keep loading; a version-1 and a version-2 schema refuse each
  other's stored metadata, so a checkpoint cannot cross the boundary silently.

- The student reference layout now follows the cohort's declared ``lookahead_s``.
  ``default_schema_for_cohort`` in ``distillation/adapter.py`` and the setup
  builder in ``distillation/cohort_setup.py`` select the version-2 layout exactly
  when the cohort declares a preview, because an encoder that sees only the
  current reference cannot reproduce a teacher that used a preview. Every
  manifest in ``configs/distillation/`` declares zero, so all eleven existing
  cohorts keep the version-1 layout. A version-2 layout against a teacher
  observation with no ``motion_lookahead`` term fails closed instead of packing a
  student that cannot see what the teacher saw.

- Added ``resolve_manifest_clips`` (``src/mjlab/tasks/tracking/mdp/multi_motion_ppo.py``),
  the one reader of a reference manifest, so a second caller cannot disagree with the
  task about what a manifest declares.

- Added the per-clip embedded-reference teacher export that VAE distillation
  requires. ``MultiMotionTrackingOnPolicyRunner.export_embedded_reference_onnx``
  and ``scripts/export_x2_teacher_reference_onnx.py`` write the original
  tracking interface (``obs, time_step`` in; ``actions`` plus seven reference
  outputs out) with one clip's tables embedded. The weights are the same shared
  teacher in every export, so a cohort covers many clips with one policy while
  the loader still verifies each member's reference against its own clip. This is
  deliberately the opposite shape from the deploy graph, which embeds no reference
  and serves any clip. Measured on the 20k teacher: two clips exported at 226
  observation width, and both pass the loader's own ``_require_onnx_interface``
  check with embedded ``joint_pos``/``joint_vel`` equal to the source clip exactly
  and ``body_quat_w`` equal at 1e-5.

- Fixed the difficult-state coverage gates reading the legacy schema instead of the
  resolved task configuration.  Four sites wrote the version-1 layout down as magic
  numbers: the coverage policy scope, two ``CohortSetup`` guards, and the parity
  evidence validator.  A preview cohort could not enable coverage at all, even with a
  passing restore-parity report, because the validator demanded ``68/99/32/31``.
  ``CoveragePolicy.difficult_state_coverage`` now takes the four dimensions, and the
  training entry point reads them from the cohort through one reader,
  ``CohortSetup.schema_for_cohort``, which the environment builder also uses.  The
  validator compares the evidence with that schema, and the environment guard compares
  the policy's declared scope with it, so a policy and a cohort that disagree are
  refused for any schema rather than for one named layout.  The policy defaults keep
  the version-1 identity byte-identical, and a caller that passes no resolved schema
  keeps the earlier contract exactly, so existing evidence and checkpoints still
  resolve.  Four regression tests, including both directions of a width mismatch.

- Fixed the cohort environment taking its preview from the base task instead of
  from the cohort.  ``make_multi_motion_cfg`` copies the motion command's fields,
  so a preview-blind base task built a narrow actor observation while the same
  cohort's saved teacher configs, and the student reference layout they drive,
  already declared the preview.  The two halves of the lane disagreed about one
  cohort.  ``build_multi_motion_environment`` now applies the cohort's
  ``lookahead_s`` to the live command and records it as a private semantic
  override, in the same way it already handles the reset policy.  An agreeing
  base task records nothing.  Regression tests cover both directions.

- Added the 144-clip multi-motion distillation cohort at
  ``configs/distillation/x2_tennis_multimotion_v1.yaml``. Every entry is the same
  20,000-iteration checkpoint with that clip's reference tables embedded, and the
  clip list is the teacher's own 148-clip training manifest filtered to the 144
  ``single_*`` references (the four ``standing_*`` clips are excluded). The
  manifest is generated from the export report, which records 144 distinct ONNX
  digests and no failures, totalling 2,350,039,928 bytes. ``scripts/export_x2_teacher_reference_onnx.py``
  gained an explicit ``--only`` name filter, because a partial selection from a
  larger manifest must be stated rather than silently inferred.

  Validation is partial by necessity. ``load_manifest`` accepts all 144 entries
  and confirms one shared checkpoint and one shared saved-params pair. Then
  ``resolve_cohort`` resolves the checkpoint, the actor, the control contract, the
  observation contract at the derived 226 width, and the ONNX interface, before it
  stops at the one check that needs the artifacts' own host: the teacher's saved
  params name an absolute NAS reference manifest, so the clip-membership check
  cannot run here. That refusal names the path, which is the designed behaviour.
  Cohort resolution must therefore run where the recorded manifest path exists, and
  training has not started.

Fixed
^^^^^

- Fixed the epoch rows of the diffusion and flow-matching ``metrics.jsonl``
  traces. The averaged per-update block is now spread before the explicit
  fields, instead of after them, so an epoch row keeps the step at the epoch
  boundary and the epoch-wide dropout ratio. Previously those fields were
  silently replaced by the mean of that epoch's per-update values, which made
  the epoch ``global_step`` read roughly half an epoch early and shifted epoch
  scalars in conversions to TensorBoard. Training, checkpoint selection and
  ``train-report.json`` were never affected.

- Fixed diffusion long-run permission propagation from the CLI to the trainer.
  Config serialization no longer rejects a large approved budget. Execution
  still requires ``--allow-long-run`` above 200,000 optimizer updates.
  CPU constructor and config round-trip regressions cover the 400,000-update
  budget without executing a long training run.

- Fixed the one-shot future-pose camera framing. The static scene keeps world
  coordinates, and the camera frames the fitted robot bodies instead of the
  global model extent. A ``Center view`` button restores this camera pose.
  Saved poses and IK results are unchanged.

- Fixed D1 batched X2 collection lifecycle. Idle slots no longer set pending
  reset flags, and early termination marks the correct environment IDs.
  VAE reference guards remain active in mixed VAE/recovery batches. Recovery
  holds the handoff reference without a wrap teleport. Clean trials receive no
  OU noise; only the retained VAE interval of OU trials receives that noise.
  The simulator's manual-reset guard remains enabled.

- Made batched OU restore parity origin-independent. The reset signature now
  subtracts each row's ``scene.env_origins`` and compares with ``rtol=0`` and
  ``atol=1e-6``. World-frame trajectory rows remain unchanged, and larger state
  changes still fail closed.

Changed
^^^^^^^

- Changed the distillation VAE export to take every declared tensor width from
  the checkpoint's own schema instead of fixed 68/99/32/31 values. A version-2
  preview checkpoint now exports an encoder input of ``[1, 130]``; a version-1
  checkpoint keeps ``[1, 68]``. The decoder ONNX output and its contract entry
  are renamed from ``actions`` to ``action``, which is the name the deployed
  controller requires. The D5 parity gate also exports the VAE encoder, and its
  denoiser wrapper copies the final-normalisation ablation and refuses to drop a
  tensor the checkpoint records.

- Changed the exported deployment contract to declare ``schema.lookahead_s`` for
  a version-2 (``motion_lookahead``) reference layout. The value is read from the
  member's saved environment config, so a hand patch is no longer required to
  load a 130-wide bundle; a version-1 contract declares no horizon. A
  version-2 member whose saved config has no ``lookahead_s``, or whose
  ``ceil(lookahead_s * reference_fps)`` is not positive, is refused. The
  checkpoint compatibility metadata is unchanged, so existing checkpoints still
  load.

- Changed the X2 collection and closed-loop startup reference path to accept
  both frozen gravity VAE layouts. Version 1 keeps reference 68 / conditioning
  99 / latent 32 / action 31. Version 2 adds one 62-wide q/dq preview at
  ``t + 0.2 s`` for reference 130, derives it from the cohort declaration, and
  checks it against the loaded model schema. The preview stays clip-local and
  reference-free after handover. The flow token layout is unchanged at 41 rows
  by 167 channels (135 state + 32 latent). Documented the preview experiment
  inputs in ``docs/source/x2_flow_matching_training.rst``. Local CPU tests use a
  synthetic VAE; no preview checkpoint, collection, training or simulator run is
  validated.

- Changed the batched D1 collection report to report resource limits truthfully:
  a run stopped by a wall-clock, inactivity or stored-output budget now writes
  ``partial: true`` and a ``stop_reason``, warns on standard error and exits
  non-zero. Streaming publication and compact counts are enabled for batched
  CLI collection; compatible runtimes use device tensor dispatch. Explicit
  ``--no-*`` options retain legacy paths. The stored-output limit applies with
  streaming. A scalar request that carries a batched-only option is refused
  before any environment is built instead of silently ignoring the option. The
  collect report now records the resolved ``policy_device`` and the requested
  ``policy_device_setting``, and ``--policy-device`` is scoped to the runtime
  factory call instead of leaking into later runs in one process. The batched
  statistics accumulator changes the floating-point reduction order even at
  ``--jobs 1``, so ``statistics_sha256`` can change for any fit through the
  batched path; existing frozen projections and matching token caches are kept,
  not refitted.

- Changed ``diffusion_merge`` to stage and atomically publish a byte-preserving
  merge, refuse an existing output and default the output shard bound to the
  largest input shard. ``--legacy-row-merge`` keeps the historical row-level
  rewrite.

- The queue dispatcher now allows all eight GPUs by default. The optional
  ``ALLOW_GPUS`` and ``EXCLUDE_GPUS`` environment variables still support
  per-server allocation and device-specific exclusions.

- Changed VAE distillation defaults to a 16384-row microbatch and two
  accumulation steps, preserving a 32768-row effective batch. Explicit batch
  options remain supported. Existing checkpoints still require their stored
  microbatch and accumulation settings on resume.

- Reduced valid multi-motion reference-index checks to one combined host
  Boolean decision. Invalid motion/frame inputs retain detailed rejection
  and error precedence. No reference cache or sampling change is introduced.

- Changed diffusion token-cache reuse and preparation. Existing valid caches are
  checked before the build-size ceiling. Missing provenance can use the same
  bounded shard-local worker plan as token materialization. Published caches
  record file identity and skip a repeated full digest when the identity is
  unchanged. Direct dataset and audit validation still force a full digest
  check.

- Changed D5 fallback scheduling so ``max_fallback_ticks`` is a structured
  warning threshold rather than an action limit. The runtime continues to use
  later rows of the accepted latent trajectory until a newer plan replaces it
  or the trajectory reaches its valid end tick. Invalid, stale, reset and
  shutdown conditions remain fail-closed.

- Added D5 startup warmup for the owner-side VAE decoder and the spawned
  diffusion planner. ``AsyncLatentRuntime.warmup`` uses valid bootstrap data,
  waits on the bounded worker completion signal, validates and discards the
  warmup plan, and leaves the controller ready for a fresh active request.
  Warmup does not publish an action or claim controller readiness.

- Changed the D3 closed-loop sampler reports to name the model family. The
  preflight report and the run provenance now state ``model_family``,
  ``sampler_method``, the actual ``inference_grid`` and ``solver_updates``. A
  diffusion run still reports ``ddim_updates: 20`` and start level ``1000``; a
  flow run reports ``euler_x0``, its real process identity and grid, and no
  ``ddim_updates`` field.

- Changed ``Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion`` to register
  ``MultiMotionTrackingOnPolicyRunner`` instead of ``MjlabOnPolicyRunner``, so a
  checkpoint save also writes the deploy export. The new runner subclasses the
  base runner and deliberately does not subclass
  ``MotionTrackingOnPolicyRunner``, whose export embeds one clip's reference
  tables; the multi-motion command refuses a single-clip loader by design. The
  policy graph, the actor and critic layouts and the training path are
  unchanged.

- Changed the X2 multi-motion export to keep the reference outside the graph.
  The reference-free layout is what allows one policy to serve many clips, but
  it also means the reference contract must travel separately. The reference is
  read from the controller's existing ``mjlab-vae-motion`` file path, so no new
  controller reader is required.

- Changed the distillation reference check to accept a teacher whose saved env
  config declares a reference manifest instead of one clip
  (``_saved_reference_clip`` in ``distillation/config.py``). A shared teacher over
  many clips records an empty ``motion_file`` and names its manifest, which the
  check previously refused before any environment existed. The relaxation keeps
  the binding in both directions: a single-clip teacher must still name exactly
  the clip the manifest selects, and a manifest-based teacher may only be paired
  with a clip that manifest lists. A recorded manifest that no longer resolves is
  refused with its path named rather than assumed, and a saved config that declares
  both sources or neither is refused. Every manifest under ``configs/distillation/``
  is single-clip, so all eleven existing cohorts take the unchanged path.


=========
Added
^^^^^

- Added the opt-in D4 sparse task surface on the D3 closed-loop seam. A new
  ``conditioned_continuation`` module wires the finished hard-keyframe
  inpainting and pose-cost guidance components into the frozen reference-free
  continuation through two narrow optional handover hooks; ``evaluate.py``
  gains ``--task-config PATH``, a strict bounded JSON loader under
  ``scripts/experiments/diffusion_d3_closed_loop/`` and a preparation tool that
  extracts verified sparse targets from an existing qualified recording. The
  default run, the eight bootstrap pairs, the measured current conditioning and
  the existing GUI/headless guards are unchanged.

- Added the bounded D3 reference-free X2 diffusion handover seam: owned eight-step state/latent history snapshots, active 167-channel no-mixture conditioning, strict 20-step DDIM/current-latent decoding, contract/hash guards, and an offline frozen D2/VAE replay diagnostic with explicit synthetic-conditioning and no-closed-loop claims. See ``docs/plans/beyondmimic_diffusion_d3_seam.md`` and the replay evaluator under ``scripts/experiments/diffusion_d3_replay/``.

- Added the bounded integration-only X2 closed-loop harness under
  ``scripts/experiments/diffusion_d3_closed_loop/``. Its explicit-path
  preflight loads the frozen contract, no-mixture projection, EMA-selected D2
  checkpoint, gravity VAE cohort member and manifest without constructing an
  environment; only ``--execute`` enters one matched 8-bootstrap/25-tick
  smoke budget. The report uses one bounded full/compact/marker persistence path,
  records actual imported-source versus artifact-root provenance and
  model/normalizer/file integrity, and makes no fake-test or simulator quality
  claim.

- Added the preregistered serial D3 pilot driver
  ``scripts/experiments/diffusion_d3_closed_loop/run_pilot.py``. It fixes the
  ``tennis_000`` seed x start-fraction matrix, the eight-bootstrap/tick budget
  and the deterministic baseline-boundary-safe horizon before any physics,
  runs one serial GPU environment at a time under a whole-batch wall/output
  cap, retains every attempted/partial/fall outcome, stops on software,
  integrity, output or untrusted-runtime failure, and records a manifest plus
  bounded JSONL progress. Its CPU fake tests cover the matrix, the horizon, the
  subprocess capture and the batch stop/continue policy.

- Added an opt-in single-branch Viser browser diagnostic for the D3
  frozen-policy path under
  ``scripts/experiments/diffusion_d3_closed_loop/view.py``. It reuses the
  evaluator's explicit-path frozen loaders and the existing Viser viewer, runs
  one operator-selected ``baseline`` or ``diffusion`` branch (eight real VAE
  bootstrap pairs then reference-free continuation), starts paused with
  pause/resume/single-step/slower playback, renders an explicit diagnostic-only
  status, and disables the stock reference-scrubbing controls. It is a separate
  authority from the unchanged matched-pair gate and makes no acceptance claim.

- Added the versioned X2 diffusion no-mixture representation contract
  ``mjlab-x2-state-latent-d1-no-mixtures-v1``: train-only normalization of 135
  physical hybrid-state values plus the frozen 32-D VAE latent (167-wide tokens,
  41 steps at 50 Hz). The explicit ``migrate-no-mixtures`` CLI rebuilds a
  derived dataset from legacy raw rows while preserving split assignments and
  source provenance; legacy projected bundles remain rejected.

- Added the X2 PPO tracking task ``Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Increated-Perturbations``. It preserves the standing-start X2 task while using the Unitree G1 magnitudes for initial pose/velocity/joint-position randomization and external pushes.

 and evaluator at
  ``scripts/experiments/vae_teacher_student_replay/``. It replays a frozen
  teacher and VAE student on the same recorded 164-D teacher/68-D reference
  and 99-D conditioning states, reports raw-action/target disagreement by
  fixed reference-time windows, validates embedded reference and student
  ONNX parity, and deliberately makes no closed-loop stability claim. The
  compact 1,103-state failure/success fixture and source/model hashes are in
  ``tests/fixtures/vae_teacher_student/``.

- Added an offline X2 tennis reference generator that copies clips 000–020
  (012 is absent) into ``data/tennis/guard040_linear05/`` with 0.5-second
  linear root-position/joint transitions to motion 040's initial guard pose,
  yaw-aligned per endpoint, SLERP root rotation and FK-derived body poses.
  `scripts/render_x2_tennis_guard_gifs.py` renders each clip and the 20-clip
  entry/exit comparison as GIFs.
  Original reference files and frozen teacher assets remain unchanged;
  reference generation does not validate balance or permit using existing
  teachers against changed motion hashes. An offline MuJoCo renderer
  generates one annotated GIF per clip that plays at real time, plus a 20-clip
  transition comparison (2x slow) under `data/tennis/guard040_linear05/renders/`;
  these are kinematic previews, not controller or balance rollouts.

- Added the D1 diffusion integration CLI (`diffusion`) with fail-closed frozen
  artifact preflight, explicit bounded collection via a concrete runtime
  factory, grouped window indexing, train-only statistics persistence, shard
  inspection, and offline action/latent replay diagnostics. Help and offline
  commands do not construct an environment; collection requires `--execute`
  and cannot override the immutable D0 execution record. See
  `docs/plans/beyondmimic_diffusion_d1_implementation.md`.
- Added the non-authorizing 50 Hz collection input at
  `configs/diffusion/x2_50hz_collection.yaml`.

- Added opt-in phase-balanced within-motion replay sampling for the shared
  cohort path. ``distill train --teacher-ids ... --phase-bins N`` (default 0,
  disabled) splits each clip's reference-phase axis into ``N`` cells and
  equalizes every minibatch's within-motion exposure over the clip's non-empty
  cells, correcting retained-row density skew (long-surviving clip segments
  accumulate more replay rows than hard-to-reach phases) without changing the
  configured across-motion mixture. The default keeps the historical
  density-proportional draw bit-for-bit. Balancing is coverage-only: it cannot
  invent coverage for phases never collected and stays difficulty-blind
  (adaptive, loss-aware sampling remains refused on the multi-motion path).
  ``phase_bins`` is recorded in the resolved configuration
  (``replay.phase_bins``) and must be reproduced on resume, with checkpoints
  that predate the option resuming at the documented default 0; the
  single-teacher path refuses the flag. Per-phase telemetry
  (``retained_by_phase_bin``/``drawn_by_phase_bin``) is added to replay reports
  when enabled. See ``docs/source/x2_tennis_distillation.rst``.
- Added opt-in standing-start distillation for the shared cohort path. Real
  deployment engages from standing, whereas reference-frame-only initialization
  never trains or validates that transition. ``distill train`` accepts
  ``--reset-policy standing-mixture`` with configurable
  ``--standing-start-fraction`` (default ``0.25``),
  ``--standing-start-window-frames`` (default ``25``), and
  ``--standing-start-frame-zero-fraction`` (default ``0.5``); the reference
  policy remains the exact default. A singleton cohort uses the Python-literal
  spelling ``--teacher-ids "('tennis_000',)"``. Standing resets use a
  clip-local ``min(25, F_i)`` window and report the realized frame-zero rate
  ``a + (1-a)/W_i`` rather than treating ``a`` as that rate. Fractions are
  proposed starting hyperparameters, not tuned values. Teacher qualification is
  deliberately a manual standalone activity and is not a pipeline gate.
  Student-only ``standing-start`` and ``standing-window`` evaluation profiles
  retain explicit censoring and completed-or-failed denominators. Enabled runs
  use version-3 cohort checkpoints; version-2 reference-only runs remain
  unchanged and cannot be resumed with standing enabled. This work is
  simulation-only and makes no policy-quality, sim2sim, or hardware claim. See
  ``docs/source/x2_tennis_distillation.rst``.

- Added version-3 cohort export. ``distill export --asset-audits`` exports a
  saved version-2/3 M4 cohort checkpoint as one shared student graph pair
  stamped with the cohort identity plus one ``{contract, motion, parity}``
  triple per manifest member in manifest order (the order the deployed
  controller plays them in). Each member is audited by the existing
  ``make_export_audit`` producer against a pinned single-teacher environment of
  that member; audits that describe different compiled robots are refused, and
  a cohort checkpoint with the singular ``--asset-audit`` is refused rather
  than silently reduced to one teacher. Member contracts stay version-2 shaped
  so the C++ deployment parser is unchanged. See
  ``docs/source/x2_vae_export_contract.rst``.

- Added the M4 shared multi-teacher distillation cohort. ``distill train
  --teacher-ids "('tennis_000','tennis_001')"`` trains **one shared conditional
  VAE** over both 50 Hz tennis clips in **one** vectorized simulator: rows keep
  their own reference clip and are labeled by that clip's frozen teacher through
  one frozen teacher bank, actions/observations keep their meanings, and the
  gravity schema (68/99/32/31) is unchanged. Motion and teacher ids stay routing
  metadata and never enter decoder conditioning. Selection is explicit and
  mutually exclusive with the preserved single-teacher ``--teacher-id`` path
  (supplying both is refused before any environment is constructed), and
  omitting both keeps the historical ``tennis_000`` default. Cohort runs use
  stratified fixed environment slots (deterministic largest-remainder allocation
  from the manifest weights, at least one row per selected motion, seeded row
  permutation), uniform phase sampling recorded as an explicit private override,
  and one per-motion balanced replay that cannot let a shorter or easier clip
  evict another. Unknown or repeated teacher ids and an impossible replay
  capacity are refused before any environment is constructed, and a failure
  after the environment exists closes it exactly once. See
  ``docs/source/x2_tennis_distillation.rst``.

- Added version-2 M4 cohort checkpoints, strict cohort resume, and checked
  member selection. ``distill train`` records the ordered member identities with
  every artifact path and digest, per-clip extent/source digest/audited body
  mapping, the ordered mapping digest, the common action/control/observation
  contract, the slot and phase policy, the replay partition policy, resource
  settings, and all RNG state; strict resume reproduces the whole record and
  only the total iteration budget and the save/reporting cadence may differ. A
  version-2 checkpoint is never converted to or from version 1, resume restarts
  the simulator with a segment namespace above every retained replay partition,
  and ``distill evaluate``/``distill play`` accept a single **member** of the
  saved cohort (member artifacts and the common contract validated against the
  live manifest, the full stored cohort identity kept in the report). The
  Viser Checkpoints tab dispatches each discovered artifact to its version's
  loader. Version-1 training resume, inference, playback, and export are
  unchanged.

- Added ``distill evaluate-cohort``: a bounded evaluation that normalizes
  ``--teacher-ids`` into manifest order (repeated ids refused before any
  environment is constructed), pins each selected motion into its own
  single-motion environment, and builds a fresh identically seeded adapter for
  each mode so the student baseline never inherits the teacher rollout's
  simulator state; one student model is loaded once and reused. Each motion
  block reports, separately for the teacher baseline and the shared student, the
  segment and outcome counts, completion/failure denominators, censored-segment
  counts, reference coverage, timeout/step-cap/timer censoring, disaggregation,
  and action-rate metrics with the same seed, phase mode, environment count, and
  step cap. Every reported level is attributed to the motion's cohort identity
  (the stored cohort member's motion id and teacher code, or the explicit
  manifest position for a teacher-only run) rather than to the pinned
  environment's local clip 0, and each block records which source it used.
  Aggregate summaries report an equal-motion ``macro`` view and an explicitly
  ``clip_duration_weighted`` view (weights proportional to ``frames/fps``) with
  the per-motion values, aggregation weights, contributing motion counts, and
  partially covered metrics retained; execution errors are reported per motion
  and mode, make ``complete`` false, and the report states that no aggregate is
  a quality pass.

- Added ``distill train`` reporting of per-motion collection attribution
  (``collection.motion_stats``: per-motion samples, teacher/student steps,
  boundaries, teacher/student disagreement, and reference-frame coverage)
  alongside the existing totals and boundary summary.

- Added the audited ``distill export`` VAE deployment seam. Version-2 bundles
  contain matching encoder/decoder ONNX graphs, frozen exact normalizers, a
  frame-major float32 motion table, full-precision action metadata, explicit
  sensor/anchor provenance, relative paths, and SHA-256 content identities.
  Export refuses malformed source/schema data and refuses to manufacture
  physical evidence from ``declared_unverified`` tensor metadata. See
  ``docs/source/x2_vae_export_contract.rst``.


- Added ``distill play``: interactive playback of a checkpointed distillation
  student in the existing Viser (default) or native viewers. It reuses the
  audited evaluation environment contract (no ``play=True`` overrides, explicit
  ``--sampling-mode {start,uniform}``), loads the checkpoint model-only before
  building the simulator, packs with the checkpoint's saved schema, performs one
  audited seeded reset before the first viewer action, and decodes the
  deterministic mean latent with no trainer, optimizer, replay buffer, or PPO
  runner. The Viser Checkpoints tab discovers ``checkpoint-iter-<N>.pt`` and
  ``checkpoint-final.pt`` and validates every load, refusing an incompatible
  swap. See ``docs/source/x2_tennis_distillation.rst``.

- Added the bounded M3 native single-teacher distillation lifecycle: segment-aware
  collection, pure VAE updates, atomic checkpoint/resume, and ``distill train`` /
  ``distill evaluate`` commands for the existing 50 Hz X2 ``tennis_000`` contract.
  The commands record resource/configuration provenance and distinguish bounded
  implementation smoke evidence from policy quality and hardware readiness; they
  do not launch unbounded production training or student export.

- Added periodic checkpointing to ``distill train`` through
  ``--checkpoint-every N``. The default cadence is 500 completed iterations;
  ``0`` disables periodic saves. Positive ``N`` writes an atomic
  ``checkpoint-iter-<lifetime iteration>.pt`` after every ``N`` completed
  iterations, each one a complete resume input, and reports every checkpoint
  written by the invocation. The cadence is recorded in the checkpoint's
  ``resolved_config`` for audit only and is not a resume invariant, so changing
  it never refuses a resume and a resumed run continues the lifetime filename
  sequence. Training now defaults to ``logs/distillation/<task-id>/`` when
  ``--task-id`` is supplied, or a UTC timestamp directory below
  ``logs/distillation/`` otherwise.

- Added progress reporting to ``distill train`` through ``--progress-every N``.
  The default cadence is 10 completed iterations; ``0`` disables it. A positive
  ``N`` writes a flushed ``[progress]`` line to stderr every ``N`` completed
  iterations with the lifetime iteration, collected samples, last update loss,
  disagreement, elapsed time, measured seconds per iteration, and projected
  ETA. Progress is written to stderr so stdout remains exactly the
  machine-readable JSON report; launchers can capture stderr as ``train.log``.

- Added model-only student evaluation to the M3 distillation CLI. ``distill
  evaluate --mode student --checkpoint PATH`` reconstructs the saved model via
  the new ``InferenceModel``/``load_inference_checkpoint`` API exported from
  ``mjlab.tasks.tracking.distillation``, infers the saved schema and model
  settings, and validates teacher hashes, control contract, and weight
  shape/dtype/finiteness without requiring the checkpoint's optimizer, replay
  buffer, training batch settings, or collector RNG. ``train`` and ``evaluate``
  now also report the requested seed, the resolved seed the environment factory
  applied before startup randomization, and the complete resolved
  runner/trainer/runtime configuration that is stored in the checkpoint.

- Documented specialist versus general tracking teachers, multimodal control,
  and the role of latent diffusion in
  ``docs/design/beyondmimic_general_tracker_and_diffusion.md``. The discussion
  retains specialist supervision for the current experiment and does not
  authorize new training.

- Added the M2 gravity-first latent policy core for BeyondMimic distillation:
  versioned named observation schemas (68/99/32/31 by default, with explicit
  102/105 anchor ablation layouts), pure packing, separate explicit student
  normalizers, conditional VAE mean/sampled inference, exact reconstruction/KL
  loss, and bounded raw labeled replay. The core is CPU-testable and does not
  add simulator collection, DAgger, training, rollout, or student export;
  physical sensor frames and validated cohort joint order remain caller-owned
  future work. See ``docs/source/x2_tennis_distillation.rst``.

- Added the first milestone of the BeyondMimic-style VAE distillation work: a
  typed, versioned teacher manifest (``configs/distillation/x2_tennis.yaml``)
  with compatibility validation, a frozen actor-only ``FrozenTeacher``/
  ``TeacherBank`` with per-teacher checkpoint normalizers and deterministic
  batched routing, and a first-class CPU ``uv run distill validate-teachers``
  command that checks native actions against the original ONNX exports for both
  selected 50 Hz X2 tennis teachers (plus export/checkpoint association and
  embedded reference arrays). The safe saved-configuration loader used by
  ``load_saved_lookahead_s`` moved to ``mjlab.utils.os.load_saved_yaml`` and now
  keeps ``!!python/name`` callables as qualified-name strings; training,
  collection, and student export are later milestones. See
  ``docs/source/x2_tennis_distillation.rst``.

- Added the X2 tennis-end recovery fine-tuning task
  ``Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Recovery``, an
  opt-in variant of the X2 no-state-estimation velocity task that pairs a
  full-state recovery reset event (80% of environments) with zero-velocity
  commands, while retention environments (20%) keep the original reset
  sequence and command distribution. The task loads tennis endpoint states
  from ``data/tennis/*.npz``, restores joint positions/velocities, pelvis
  world pose and world-frame linear/angular velocity, and recenters XY to the
  environment origin while preserving reference height. The source velocity
  checkpoint (``model_19999.pt``) loads strictly via the narrow
  ``TennisRecoveryOnPolicyRunner`` with optional post-load LR override
  (``initial_finetune_lr``). A bounded evaluation CLI provides deterministic
  held-out row selection for source-vs-candidate comparison. See
  ``docs/source/x2_tennis_recovery.rst``.

- Added the X2 velocity torso-IMU ablation grid: four tasks derived from
  ``Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation`` that move the actor onto
  the torso ``imu_1`` gyro and up-vector and enumerate the two frames the
  privileged critic can read — ``-Torso-IMU-Critic-Pelvis-Root``,
  ``-Torso-IMU-Critic-Pelvis-Upvector``, ``-Torso-IMU-Critic-Torso-Root`` and
  ``-Torso-IMU-Critic-Torso-Upvector``. Observation order and widths, rewards,
  terminations, actions, commands, domain randomization, terrain and PPO settings
  are unchanged, each variant has its own PPO directory, and the shipped task is
  untouched, so the four are fresh runs comparable against it. The twist command
  and its velocity rewards are root-referenced while ``upright`` and
  ``body_ang_vel`` are torso-referenced, so a torso actor trades root-frame rate
  observability against directly measuring what is penalized. See
  ``docs/source/sim2real_domain_randomization.md``.

- Added the X2 torso-IMU ablation
  ``Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR-Reduced-Perturbations-Torso-IMU``,
  the reduced-perturbation baseline with the anchor still on ``torso_link`` but
  the ``base_ang_vel`` observations (actor and critic) and the critic's
  ``base_lin_vel`` re-pointed from the pelvis ``imu_0`` site to the torso
  ``imu_1`` site. The observation order, widths, noise, scale, delay, all DR
  axes, rewards, terminations, reset/push bounds and PPO settings are unchanged,
  and the root stays the pelvis; only the IMU source moves, so it is a clean A/B
  against the baseline and requires fresh training in its own PPO directory. The
  ONNX export metadata now also records the per-term sensor site and body so the
  angular-velocity frame is explicit. See
  ``docs/source/sim2real_domain_randomization.md``.

- Added the X2 anchor ablation
  ``Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR-Reduced-Perturbations-Pelvis-Anchor``,
  the reduced-perturbation baseline with ``commands.motion.anchor-body-name``
  set to ``pelvis`` (the simulated root and the frame the robot measures
  directly) instead of ``torso_link``. Rewards, terminations, DR, reset and push
  bounds, PPO settings, critic terms and observation dimensions are unchanged,
  so it trains from scratch in its own PPO experiment directory. See
  ``docs/source/sim2real_domain_randomization.md``.

- Added three additive X2 tracking observation ablations from the selected
  reduced-perturbation ABLATION3 baseline:
  ``...-Projected-Gravity`` (161 actor dimensions),
  ``...-Projected-Gravity-And-Anchor`` (167), and
  ``...-Vendor-Velocity-Scaling`` (164 with measured gyro/joint-velocity
  preprocessing scales). They use separate PPO experiment directories,
  preserve the historical tasks and checkpoint, and require fresh training.
  See ``docs/source/sim2real_domain_randomization.md`` for contracts,
  normalization and future deployment caveats.
- Added the X2 no-state-estimation tracking ablations
  ``Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR`` and
  ``Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR-Reduced-Perturbations``.
  The first correlates Kp/Kd reset scales and joint encoder packet delay; the
  second halves only reset/push perturbation bounds and lengthens push intervals.
  They use independent PPO experiment directories and leave the existing X2
  baseline task and its checkpoints unchanged. See
  ``docs/source/sim2real_domain_randomization.md``.

- The motion-tracking tasks randomize the actuator PD gains (``dr.pd_gains``)
  per episode with matched ``kp_range``/``kd_range`` of ``(0.7, 1.3)`` in
  ``scale`` mode, so policies train against a spread of joint servo stiffness
  and damping for actuator sim-to-real robustness.

- The motion-tracking tasks randomize seven more sim-to-real axes by default:
  link mass/inertia/COM (``dr.pseudo_inertia``), joint armature, actuator effort
  limits (weaker-only, with an extra 0.45x range on the X2 wrist group, which the
  vendor simulator clamps to 2.2 N.m against the URDF's 4.8), joint friction and
  damping, foot geom size, and observation delay. X2 actuator groups also carry a
  0-2 step command delay. ``MJLAB_DR_AXES`` selects a subset at process start so a
  trained checkpoint's sensitivity can be swept one axis at a time (``none``
  reproduces the pre-axis set); unknown names raise. See
  ``docs/source/sim2real_domain_randomization.md``.

- AgiBot X2 Ultra (flagship revision, vendor URDF release v1.3.0) joins the asset
  zoo as ``get_x2_robot_cfg``, together with the motion-tracking tasks
  ``Mjlab-Tracking-Flat-AgiBot-X2`` and its no-state-estimation variant. Actuator
  parameters are derived from AgiBot's PFP joint-module data (rotor inertia and
  reducer ratio) and the peak joint torques of the flagship revision, and
  ``scripts/convert_agibot_x2_to_mjcf.py`` ports a vendor release into an
  mjlab-convention MJCF (URDF body inertials, named collision geoms, mjlab IMU
  sensor names).
- AgiBot X2 Ultra gains a flat-terrain velocity-tracking task
  ``Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation`` (no-state-estimation
  variant only: the actor observes gyro, IMU up-vector, encoders and the twist
  command, but not base linear velocity; the critic keeps it privileged). The
  config injects ``left_foot``/``right_foot`` sites at spec build time for the
  foot height scan and foot slip/clearance rewards, leaving the vendored XML
  byte-identical. It also randomizes the actuator PD gains per episode
  (``dr.pd_gains``, matched ``kp_range``/``kd_range`` of ``(0.7, 1.3)`` in
  ``scale`` mode) on top of the base velocity DR set; command-channel delay is
  inherited from the X2 actuator groups.
- New script ``scripts/record_tracking_policy_commands.py`` records the motor commands
  a tracking checkpoint issues in the training simulator: for every actuated joint it
  logs the reference frame's joint state, the actor observation, the raw and clipped
  actions, the processed position target, the simulator target, pre/post joint state,
  actuator torque, root state, gains, effort limits and timing, plus provenance and a
  nominal-diagnostic override list. It supports an optional deploy-ONNX parity check on
  identical actor inputs and refuses to overwrite an existing recording. See
  ``docs/reports/tracking_policy_command_recording.md``.
- Motion-tracking policies can optionally receive a single future reference frame
  through ``MotionCommandCfg.lookahead_s``.
- The G1 29DOF mode-15 tracking task adds dedicated foot-tracking reward terms
  (``motion_feet_pos``, ``motion_feet_lin_vel``) that grade ankle position and
  swing velocity directly instead of diluting them in the all-bodies mean.
- Tracking task init-frame sampling gains a ``weighted`` mode: with
  ``MotionCommandCfg.init_weight_s > 0``, episode start frames are drawn from
  an exponential distribution over the trajectory (``p(t) ∝ exp(-t/τ)``),
  concentrating training on the beginning of the motion.
- New script ``scripts/evaluate_tracking_policy.py`` evaluates a tracking
  checkpoint against a local motion file without Weights & Biases, reporting
  MPKPE, joint velocity error, end-effector errors, success rate and a
  termination-time histogram over the full trajectory.
- New script ``scripts/retarget_npz_to_tracking_npz.py`` converts a retargeted
  qpos motion (e.g. UMR / SMPL-X retargeting output: ``qpos``, ``fps``,
  ``robot_joint_names``) into the tracking-format npz consumed by ``train.py``.
  It resamples the trajectory to the environment rate (50 Hz by default) and
  replays it kinematically through the G1 29DOF mode-15 robot.

Changed
^^^^^^^

- The D4 task loader now re-verifies recording-bound sources instead of trusting
  the declared label: it re-reads the recording and summary bytes, re-checks their
  digests, rows, recorded contract identity and member/seed/start-frame scenario,
  and re-extracts every target state at its tick before accepting the task. A
  hand-written ``preparation_tool:`` prefix can no longer upgrade a source. Source
  qualification remains in task provenance but is no longer an ``--execute`` gate.
  The preparation tool gained ``--recording-dir`` for actual D3 closed-loop runs
  (the archived ``diffusion-d3-recorded-comparison-*`` layout), keeps raw NPZ
  shards as an honestly ``declared`` weaker layout, and validates its payload with
  the strict loader before an atomic write. Task configs and recording files are
  read through a bounded regular-file reader.

- The opt-in D4 sparse task surface now enforces pure modes (hard keyframes only
  in ``keyframe``, soft pose goals only in ``poseguided``, both in ``combined``)
  and refuses a hard keyframe and a soft goal scheduled on the same tick, since
  the hard row makes the soft cost a constant with zero gradient.

- D4 task runs now score hard keyframes and soft pose goals separately from
  trusted measured observations, including the final budget post and a
  trustworthy fall post, which are reported through a new optional
  post-observation seam. Planning ticks and measured observations are counted
  separately, a next-row metric on a hard-fixed row is labelled
  ``hard_keyframe_row`` instead of ``denoiser_forecast``, and
  ``TaskConditioning.reset()`` plus the new optional reset hook clear the whole
  task ledger before the mandatory fresh bootstrap. Runs without
  ``--task-config`` keep the original D3 provenance: no task block is emitted in
  the preflight report, the run provenance or the bounded marker.

- The D3 reference-free continuation seam accepts two optional D4 task hooks
  (``plan_transform`` and ``sampler_factory``) plus an optional bounded per-step
  task record on ``ContinuationStep``. Both default to ``None``, so every
  existing caller keeps the original plan, sampler, RNG and decode path.

- The D3 reference-free continuation seam also accepts two optional reporting
  lifecycle hooks (``on_reset`` and ``measured_post``). The reset hook runs after
  the seam clears its own history, and the measured-post seam only forwards an
  already-validated trusted successor tick; neither can sample, decode, change
  history or advance the planning count. Both default to ``None``.

- The D3 closed-loop matched-pair runner now defaults its initial/bootstrap
  array matching tolerance to ``1e-3``, the teacher-tracking error scale
  selected by the user, instead of ``1e-6``. This is a raw matched-array
  tolerance, not a universal length unit and not an unguided reference-tracking
  failure criterion; historical ``1e-6`` results remain unchanged.

- ``distill train`` now exposes standing-start options without changing the
  legacy reference-only default. Standing options require the cohort path,
  including the singleton tuple form, and their resolved reset provenance is
  persisted only in version-3 standing cohort checkpoints. ``distill evaluate``
  and ``distill evaluate-cohort`` accept the student-only
  ``--reset-profile standing-start|standing-window`` profiles; explicit
  ``--sampling-mode`` is not combined with these profiles. Teacher and both-mode
  transition evaluations remain intentionally unsupported.

- ``distill train --teacher-id`` now defaults to ``None`` so the plural
  ``--teacher-ids`` selection can be unambiguously distinguished from it. The
  resolved behavior is unchanged: with neither flag the single-teacher path uses
  ``tennis_000`` exactly as before, and ``--teacher-id <id>`` behaves as before.
  Because the shared Tyro configuration uses Python literal syntax for
  collections, the plural form is ``--teacher-ids "('a','b')"`` (a single
  element keeps its trailing comma). ``distill export`` refuses a version-2 M4
  cohort checkpoint explicitly instead of fabricating a single-teacher training
  history; version-1 export is unchanged.

Fixed
^^^^^

- Viser motion scrubbing now aligns the first policy action after ``Start Here``
  (or a paused frame-slider move) with the edited command state. The
  observation cache is refreshed through a scoped
  ``ObservationManager.refresh(env_ids, baseline=...)`` hook that splices the
  recomputed rows for the edited environments only, so unedited environments
  keep their cached rows and their history/delay timelines (no extra tick or lag
  draw). ``Start Here`` captures the environment/frame/selection at click time
  and snapshots the pre-reset cache with
  ``ObservationManager.cached_observations()``, so a partial reset cannot leak
  its whole-batch recompute (and resampled raw noise) into the untouched
  environments; the all-env path refreshes everyone normally. The frame write
  and refresh are queued to the viewer's main loop instead of running on Viser's
  GUI worker thread, and a rejected checkpoint swap keeps the viewer alive
  without installing the returned policy (a loader that mutates captured state
  in place is not rolled back).

- ``load_inference_checkpoint`` now accepts a relocated, byte-identical
  reference motion: a changed ``control_contract['motion']`` path is allowed
  only when the supplied teacher hashes (including the motion digest) match and
  every other contract field agrees. Strict training resume is unchanged.

- ``distill evaluate --mode student`` can now load a checkpoint produced by
  ``distill train``. It previously rebuilt a default student and a
  training-shaped trainer/replay, so loading a train-produced checkpoint failed
  on the saved trainer configuration and then, even with matching settings, on
  the saved collector RNG.

- ``play.py`` and ``evaluate.py`` now restore ``lookahead_s`` from the saved
  ``params/env.yaml``, so watching or evaluating a checkpoint trained with
  lookahead works without manually repeating the flag. The saved YAML uses
  Python-specific tags (``!!python/tuple``), which are now parsed safely.
- The deployment ONNX exporter now reconstructs the saved lookahead setting and
  can convert a state-estimation tracking checkpoint to the no-state-estimation
  observation layout required by the G1 deployment controller.

Changed
^^^^^^^

- ``distill train`` reports no longer embed the per-environment boundary
  arrays. Each iteration's ``collection.boundaries`` is now a compact summary
  by default: per-``reason`` boundary-record and environment-mention counts,
  the observed before/after segment and generation ranges, and the iteration's
  ``ticks``/``records``/``env_mentions`` totals. A boundary record holds four
  full-batch arrays (before/after segment and generation) plus ``env_indices``,
  so a 4096-environment, 20-iteration ``train-report.json`` reached 211 MB and
  the raw arrays are now written only on request with the new
  ``--report-boundaries full`` flag (``summary`` is the default, and an unknown
  value is refused before any environment is constructed). A summary report is
  aggregated per iteration as collection finishes instead of buffering the
  whole run, and every other report entry is unchanged.

- ``distill evaluate`` no longer accepts the trainer-only options
  ``--learning-rate``, ``--beta``, ``--accumulation-steps``,
  ``--minibatch-size`` and ``--replay-capacity``; student evaluation is
  model-only and infers those settings from the checkpoint, so keeping the flags
  would have silently ignored them.

- ``distill train``/``evaluate`` forward ``--seed`` into environment
  construction, so the private environment configuration is seeded before MuJoCo
  startup randomization instead of relying on a post-construction global seed.
  A resume now compares the checkpoint's stored ``resolved_config`` provenance
  against the requested settings and refuses a mismatch, while still allowing
  the explicitly requested total-budget extension of ``--max-iterations``.

- The AgiBot X2 tracking task now tracks the full wrist chain: the body-tracking
  rewards previously supervised ``left/right_wrist_yaw_link`` only, so wrist
  pitch/roll had no reward gradient and stayed passive regardless of reference
  motion. ``body_names`` now also includes ``left/right_wrist_pitch_link`` and
  ``left/right_wrist_roll_link``, and the four body-tracking reward terms
  gained an optional per-body ``body_weights`` parameter; the X2 task uses it
  to grade wrist pitch/roll at 0.3 vs 1.0 for the main body chain, so wrist
  flexion is reproduced without competing with core pose/velocity tracking.

- ``scripts/retarget_npz_to_tracking_npz.py`` takes a ``--task`` argument instead of
  hard-coding the G1 29DOF mode-15 environment, so a retargeted motion can be replayed
  onto any registered tracking robot.
- Bumped ``rsl-rl-lib`` from 5.4.2 to 5.5.0. This update removes the ``logger_type``
  attribute of the ``rsl_rl.utils.Logger``, so code that previously checked
  ``logger.logger_type`` must instead check the type of ``logger.writer``.
- The AgiBot X2 actuator gains are now the gains the real robot commands in its
  native stable stand, measured from the recorded ``/aima/hal/joint/*/command``
  messages of two hardware captures, instead of the reflected-inertia 10 Hz
  template. The hips, knee, ankles, arms and head yaw move to the measured
  values (hip 100/4, knee 150/5, ankle pitch 40/3, wrist pitch/roll 20/1,
  head yaw 3.4/0.114, ...). Relative to the template these are 1.7x softer on the
  hips and 2-6x stiffer on the ankles, elbow, wrist and head yaw. Waist yaw and
  waist pitch/roll keep the template value because the vendor's own two tables
  disagree by 3.75x and 7.5x on them; head pitch keeps it because the measured
  gain contradicts the joint's 0.6 N.m effort limit.

  See ``docs/source/x2_gain_provenance.md`` for the measurement chain, the four
  other vendor gain tables in circulation, and the open evidence gaps (notably:
  the gains are *commanded*, and nothing yet verifies the motor-side loop).
- The X2 action scale is no longer ``0.25 * effort_limit / Kp``. It is now the
  larger of that torque floor and a tracking floor derived from the reference
  motion's p95 deviation from the keyframe, so a joint's gain and its action
  parameterization are no longer coupled. This raises the action scale on six
  groups, where a unit action consequently reaches 27-60% of the torque budget
  (knee 33%, elbow 50%, wrist pitch/roll 60%); the other ten sit at exactly 25%.
- The X2 keyframe is the pose the real robot is measured standing in (hip pitch
  -0.19, knee 0.457, ankle pitch -0.267, shoulder pitch 0.4, elbow -1.2) instead
  of an ad hoc bend, with the soles still flat on the ground. Both X2 tasks share
  the keyframe, so the velocity task's default pose and its action scales (which
  come from the same robot-level table) change with it; its actor and rewards are
  otherwise untouched.
- The tracking tasks' sim-to-real randomization ranges are capped at 5% of
  nominal. The axes they carry exist nowhere else (the velocity tasks randomize
  only the actuator PD gains), so they were the only place a policy could be
  trained against an actuator markedly different from the one it runs on: the
  armature and joint-friction scales go from +-30% and +-50% to +-5%, the effort
  limits from 0.75-1.0x to 0.95-1.0x, and the pseudo-inertia mass range from
  0.90-1.10x to 0.95-1.05x with its COM shift halved to +-1 cm. Two ranges have
  no nominal to take 5% of and are documented rather than scaled: the absolute
  passive-damping range is one tenth of what it was (0.03 N.m.s/rad on a leg
  joint, now all but inert against a derivative gain of 4), and the discrete
  0-1 step observation delay is unchanged. The X2 wrist effort derating drops to
  its 5% residual, which means the vendor simulator's 2.2 N.m wrist clamp is no
  longer trained for; modelling it now requires changing the nominal wrist
  effort limit instead.
- The X2 waist yaw actuator moves to the measured hardware gain (``kp 40``, ``kd 8``,
  from 170.785/10.872) and to the reflected rotor inertia AgiBot's own Isaac actuator
  config carries for it (0.010177520, from the 0.043260 the PFP-96 torque class
  implied). The measured value had been held back because the vendor's own two tables
  disagree 3.75x on that joint (40/8 standing, 150/3 posture hold); a third source
  resolves it, since the vendor's sample policy ships 40.1792/2.5579, which is the
  10 Hz template applied to that armature. The X2 effort limits are now pinned by a
  test to the flagship revision's released joint ratings: the package this differs
  from (`sonic-x2`) is built on the newer revision, whose waist is 36 N.m and wrist 6
  rather than 48 and 4.8. See ``docs/source/x2_gain_provenance.md`` §7.

Fixed
^^^^^

- Capped ``wandb`` below 0.29, which removed the ``start_method`` setting still passed
  by ``rsl-rl-lib`` and crashed training runs launched with ``--logger wandb``.
- ``distribution="gaussian"`` domain randomization now draws an independent value per
  environment. ``sample_gaussian`` ignored its ``size`` argument when ``mean``/``std``
  were tensors, so every environment received the same sample :issue:`1168`.
- ``FlatPatchSamplingCfg(patch_radius=0)`` no longer collapses every patch to the
  sub-terrain center. The edge-exclusion mask sliced ``arr[-0:]``, which is
  ``arr[0:]``, so it cleared the entire valid mask :issue:`1171`.
- The X2 joints no longer inherit the vendored MJCF's blanket
  ``frictionloss="0.3"``: dry friction is now per joint group, at or below 2% of
  each joint's rated torque. The old value was 50% of ``head_pitch``'s effort
  limit, and combined with the friction and effort-limit randomization it could
  reach the joint's whole torque budget and lock it.
- The X2 tracking task's joint-damping randomization is no longer an absolute
  ``(0.0, 0.3) N.m.s/rad`` range for every joint. It scales with each group's
  torque class (``effort_limit / 400``, which reproduces the old value for the
  legs), because on ``head_pitch`` that absolute range was 37% of the joint's
  own derivative-gain authority.


Version 1.6.0 (August 8, 2026)
------------------------------

.. admonition:: Breaking API changes
   :class: attention

   - ``CollisionCfg`` now requires ``contype``, ``conaffinity``, ``condim``,
     and ``priority`` to be explicit instead of silently defaulting to
     MuJoCo's values, and dict values for these fields must cover every
     matched geom (add a catch-all ``".*"`` entry).
   - ``CommandTerm._update_command`` now takes an ``env_ids`` argument:
     ``None`` on the regular per-step update and the reset environment ids
     when called from ``reset()``. Custom command terms must add the
     parameter (construction raises a ``TypeError`` with migration
     instructions otherwise) and scope any per-step state advance, such as
     a motion frame index, to ``env_ids``.
   - ``ViewerConfig`` is now keyword-only; positional construction no
     longer works.

.. admonition:: Highlights
   :class: note

   - Upgraded to MuJoCo and MuJoCo Warp 3.11.
   - Upgraded ``rsl-rl-lib`` to 5.4.2.

Added
^^^^^

- Added ``GeomCfg``, exposed as the ``geoms`` field on ``EntityCfg``, a spec
  editor that matches geoms by name and patches their attributes. Supports
  ``group`` (so a geom can collide without being drawn) and all collision
  attributes; unset attributes are left untouched. Contribution by
  @bd-pmorais.
- Added ``diffuse``, ``specular``, ``ambient``, ``active``, and
  ``attenuation`` fields to ``LightCfg`` for configuring light color and
  falloff. Contribution by @bd-pmorais.
- Added ``random``, ``file``, ``cubefiles``, ``gridsize``, ``gridlayout``,
  ``nchannel``, ``hflip``, and ``vflip`` fields to ``TextureCfg``, so textures
  can be loaded from image files instead of only built-in patterns.
  ``width`` and ``height`` are now optional, since file-based textures take
  their size from the image. Contribution by @bd-pmorais.
- Added light domain randomization functions: ``dr.light_diffuse``,
  ``dr.light_specular``, ``dr.light_ambient``, ``dr.light_attenuation``,
  ``dr.light_cutoff``, and ``dr.light_exponent``. Contribution by @bd-pmorais.
- Added ``reduce="sum"`` to ``MetricsTermCfg`` for reporting the accumulated
  episode total (e.g. episodic reward, total distance traveled) instead of a
  per-step average. Contribution by @bd-mlutter
- Added ``ViewerConfig.geom_group`` and ``ViewerConfig.site_group`` to
  control which geom and site visualization groups the offscreen renderer
  draws. Defaults match MuJoCo's (groups 0 through 2), so rendering is
  unchanged unless configured. Contribution by @bd-mlutter.
- Added ``dr.mat_texid`` to randomize which texture fills a given
  ``mjtTextureRole`` slot (RGB by default) of each selected material,
  sampling uniformly from ``asset_cfg.texture_names``. Contribution by
  @bd-pmorais.

.. figure:: _static/changelog/mat_texid_dr.gif
   :width: 30%

Changed
^^^^^^^

- Bumped ``mujoco`` and ``mujoco-warp`` from 3.10 to 3.11, and regenerated the
  bundled MuJoCo type stubs.
- Bumped ``rsl-rl-lib`` from 5.4.0 to 5.4.2.
- ``CollisionCfg`` and ``GeomCfg`` now share one write path, and mjlab warns
  when a ``GeomCfg`` collision patch is overwritten by a ``CollisionCfg``.
- Changed the default MuJoCo Warp render background to solid black
  (``0, 0, 0, 1``), matching MuJoCo's native renderer. Contribution by
  @bd-pmorais.
- The offscreen renderer now works on a copy of the ``MjModel``, so its
  render-only tweaks (extent, shadows, reflections, offscreen size) no
  longer leak into the shared model. Contribution by @bd-mlutter.
- ``ViewerConfig`` is now keyword-only, with fields grouped and documented.
  Contribution by @bd-mlutter.
- ``ViewerConfig.fovy`` now also applies to the ``ASSET_ROOT`` and
  ``ASSET_BODY`` tracking cameras instead of being silently ignored; leave
  it at ``None`` (the default) to keep the model value. Contribution by
  @bd-mlutter.
- ``auto_reset`` and an explicit ``reset()`` now leave identical command and
  event timer state (:issue:`1133`).

Fixed
^^^^^

- The Viser motion scrubber's Start Here button no longer computes relative
  body poses from stale pre-scrub kinematics, which could spuriously
  terminate the episode on the next step.
- Mid-episode lifting command resamples now refresh kinematics and the
  multi-cube reward cache, so observations and rewards no longer see
  pre-teleport object positions for one step after each resample.
- ``UniformVelocityCommand``'s ``init_velocity_prob`` path no longer writes
  the previous episode's terminal pose back into the sim on reset. It read
  derived kinematics before ``forward()`` ran and rewrote the root pose; it
  now reads only qpos for the orientation and writes only the root velocity.
- ``init_velocity_prob`` now applies on episode reset only. A mid-episode
  timer resample used to also teleport the root velocity, which ran after
  ``step()``'s forward and left velocity observations stale for one step.
- ``Entity.set_joint_position_target`` and its velocity/effort/tendon/site
  siblings now select the outer product when both ``env_ids`` and the element
  ids are tensors, instead of pairing them elementwise.
- ``MotionCommand`` now refreshes kinematics after a timer-expiry resample
  (finite ``resampling_time_range``), matching its wraparound path.
- ``CircularBuffer`` lag retrieval now clamps to the oldest retained frame;
  a lag beyond the buffer length used to wrap around to a newer frame.
- Camera sensor caches are now invalidated after ``sense()``, so a
  pre-sense read with ``clone_data=True`` can no longer pin the previous
  step's frame into the observations (mirrors the raycast fix for
  :issue:`998`).
- ``reset(env_ids=...)`` no longer appends a frame to every env's observation
  history and delay buffers; only the reset envs receive their post-reset
  frame. Previously, each manual partial reset gave the other envs a duplicate
  frame, shortening their effective history and drifting their delay
  schedules.
- ``reset(env_ids=...)`` no longer advances stateful commands in
  environments that were not reset. Previously a partial reset gave every
  environment an extra command update, so with ``auto_reset=False`` a
  ``MotionCommand`` reference motion played at twice the normal speed and
  could teleport non-reset robots via the wraparound resample. The adaptive
  sampling EMA also no longer folds on resets, matching auto-reset
  training dynamics. :issue:`1138`
- Interval event timers are now resampled on episode reset for function-based
  terms, as documented. Previously the countdown carried across episodes, so a
  new episode's first ``push_robot`` in the velocity and tracking tasks could
  fire arbitrarily soon after spawn; with mixed function and class interval
  terms, reset also wrote into the wrong timer slots. Note this changes push
  timing relative to earlier training runs, and an interval term whose range
  exceeds the episode length is now re-armed on every reset and will never
  fire.
- ``RayCastSensorCfg.include_geom_groups`` now raises on values outside
  ``[0, mjNGROUP)`` instead of silently excluding every geom.
- Geoms with a negative group no longer pick up group 5's visibility toggle in
  the Viser viewer.

Version 1.5.3 (July 22, 2026)
-----------------------------

Changed
^^^^^^^

- The Viser reward bar panel's term cap is now configurable via
  ``ViewerConfig.reward_bar_max_terms``, so environments with more than 20
  reward terms can show them all. Defaults to 20, preserving previous behavior.
  :issue:`1079`

Fixed
^^^^^

- Bumped ``pillow`` (12.3.0), ``onnx`` (1.22.0), and ``soupsieve`` (2.9.1) in the
  lockfile to pick up security fixes.
- Fixed raycast sensor debug visualization and observations lagging one step
  behind the sensed hits. ``sense()`` rebinds the hit tensors after the cache had
  already been repopulated by a pre-sense reward read, so ``.data`` returned the
  previous step's hits; the cache is now invalidated after ``postprocess_rays``.
  With ``ray_alignment="yaw"`` this made debug rays appear tilted by the foot's
  per-step motion instead of vertical. :issue:`998`
- Restored ONNX uploads and W&B run metadata for velocity and manipulation
  training when using RSL-RL's current ``WandbLogWriter`` logger name.
- The Viser reward bar panel no longer *silently* drops reward terms beyond
  ``max_terms``; it now emits a warning listing the hidden terms. Previously
  environments with more than 20 reward terms had the overflow disappear from
  the bar panel with no indication. :issue:`1079`
- Fixed the ``terrain_levels_vel`` curriculum promoting every env from level 0
  to level 1 on the initial reset, ignoring ``max_init_terrain_level=0``. Before
  the first step the robot sits at its spawn pose rather than a walked-to
  position, so the distance check was spurious; terrain levels are now frozen on
  that first reset. :issue:`1094`
- Fixed the velocity task's actor ``joint_pos`` observation not being biased by
  the ``encoder_bias`` domain randomization, so the encoder bias only affected
  actions and never the observed joint positions. The actor now observes biased
  joint positions while the critic keeps the true (unbiased) values as
  privileged information, matching the tracking task.
  See `discussion #1065 <https://github.com/mujocolab/mjlab/discussions/1065>`_.
- Hardened ``fit_terrain_normal`` against non-finite raycast hits. A single env
  with a diverged state produced a NaN/Inf covariance that made
  ``torch.linalg.eigh`` raise and abort the whole batch; such rows now fall back
  to the up vector. This stops the hard crash so a diverged env can be reset
  normally; it does not by itself make a diverged env's downstream reward finite.
  :issue:`912`
- Enabled ``obs_normalization`` on the Go1 velocity actor and critic to match
  the other velocity tasks. Without it, extreme-but-finite observations on rough
  terrain drove value/policy divergence that eventually surfaced as a
  ``normal expects all elements of std >= 0.0`` crash. Note that Go1 velocity
  checkpoints trained before this change carry no normalizer buffers and will no
  longer load; retrain from scratch. :issue:`870` :issue:`1044` :issue:`1053`
- Fixed ``ContactSensor`` air-time tracking accumulating float32 sim-clock
  differences, whose quantization error grows with the clock magnitude and made
  ``compute_first_contact`` / ``compute_first_air`` miss touchdowns on long runs.
  The exact float64 substep ``dt`` is now accumulated instead. :issue:`1101`
- Bumped ``mujoco-warp`` to 3.10.0.3, fixing a CUDA 700 illegal memory access in
  ``smooth.crb`` triggered by startup mass domain randomization (via
  ``set_const``) once ``num_envs >= 128`` on consumer Ada GPUs. :issue:`1108`

Version 1.5.2 (July 17, 2026)
-----------------------------

Fixed
^^^^^

- Fixed CUDA illegal memory accesses when domain randomization triggers
  ``set_const`` with multiple environments. ``actuator_acc0`` is now expanded
  per environment before MuJoCo Warp recomputes it.
- Fixed ``MaterialCfg.reflectance`` being ignored when building the MuJoCo
  spec. Contribution by @bd-pmorais.

Version 1.5.1 (July 15, 2026)
-----------------------------

Added
^^^^^

- Added ``MeshCfg``, a spec editor that matches mesh assets by name and edits
  their asset-level attributes. The first attribute is ``maxhullvert``, which
  caps the collision convex hull's vertex count to lower narrowphase cost.
- Added ``SimulationCfg.broadphase`` and ``SimulationCfg.broadphase_filter``
  to configure MuJoCo Warp's broadphase collision algorithm and
  bounding-volume filters.

Changed
^^^^^^^

- Enabled skybox rendering for camera sensors. Contribution by @bd-pmorais.
- Bumped the minimum ``mujoco-warp`` to 3.10.0.2, which fixes ``qfrc_constraint``
  being populated incorrectly across vectorized environments (:issue:`1086`).
  Earlier 3.10.0.x releases are no longer supported.
- Command delay on fusable actuators (ideal PD, DC motor) now applies one shared
  lag per environment across all fused actuators sharing a delay config, matching
  the built-in actuator path, rather than an independent lag per actuator group
  (:issue:`1035`).

Fixed
^^^^^

- Fixed ``TerrainGenerator`` overwriting custom geom names set by sub-terrain
  functions with the default ``terrain_{i}`` name. Only unnamed geoms are now
  auto-named.
- Fixed ``TorchArray`` not expanding world-shared model fields to ``nworld``
  with mujoco_warp 3.10.0.2, which allocates them as real size-1 arrays
  instead of stride-0 broadcast views. Multi-env indexing of fields like
  ``soft_joint_pos_limits`` raised ``IndexError`` during resets (:issue:`1093`).
- Fixed ``mdp.bad_orientation`` returning NaN when float32 rounding in
  ``quat_apply_inverse`` pushed the projected-gravity z-component slightly
  outside ``[-1, 1]``, making ``torch.acos`` return NaN and silently
  suppressing the termination for flipped robots. The argument is now clamped
  to ``[-1, 1]``.
- Fixed a crash when using command delay on ideal PD (or other custom)
  actuators whenever ``num_envs`` differed from the number of delayed targets,
  and fused ideal PD and DC motor actuators sharing a transmission and delay
  config into a single gather, delay, control-law evaluation, and control
  write, removing per-group host overhead (:issue:`1035`).

Version 1.5.0 (June 28, 2026)
-----------------------------

Added
^^^^^

- Added ``reduce="max"`` to ``MetricsTermCfg`` for reporting episode-peak values
  (e.g. peak power, peak contact force) without needing stateful wrapper classes.
- Added ``BuiltinDcMotorActuator``, a native MuJoCo ``<dcmotor>`` wrapper.
  Supports voltage / position / velocity input modes with back-EMF,
  configurable motor constants, and optional integral, slew, inductance,
  thermal, LuGre, and cogging extensions.
- Added ``scale_with_difficulty`` to ``HfRandomUniformTerrainCfg``. When
  enabled, the noise amplitude scales with difficulty (flat at 0, full
  ``noise_range`` at 1) so the terrain progresses in a curriculum. Defaults to
  ``False``, preserving the previous difficulty-independent behavior.
- Added material domain randomization functions for MuJoCo Warp RGB rendering:
  ``dr.mat_emission``, ``dr.mat_specular``, ``dr.mat_shininess``, and
  ``dr.mat_texrepeat``.

Changed
^^^^^^^

- Bumped ``rsl-rl-lib`` from 5.2.0 to 5.4.0.
- Bumped ``mujoco`` and ``mujoco-warp`` to 3.10, both pinned from PyPI. The
  ``py.mujoco.org`` nightly index and the ``mujoco-warp`` git pin are dropped, so
  resolution no longer breaks when nightly wheels are garbage-collected.

  .. warning::

     ``SimulationCfg.ls_parallel`` is deprecated and now ignored, since parallel
     linesearch was removed upstream in MuJoCo Warp. Setting it emits a
     ``DeprecationWarning``; remove it from any ``SimulationCfg`` you construct.
- Curriculum-mode terrain difficulty is now deterministic across rows
  and reaches the configured ``difficulty_range`` endpoints
  (:issue:`1027`).
- Heightfield terrains now color by absolute height with a diverging palette
  (cool below the ground plane, green at ground level, warm above) on a fixed
  scale, replacing the per-patch normalization. Color is now consistent across
  terrains, and low-amplitude terrain such as ``random_rough`` reads as gently
  tinted ground instead of high-contrast noise.
- ``BoxNestedRingsTerrainCfg`` now builds uniform-height concentric ridges
  whose separating gaps widen with difficulty, replacing the random per-ring
  heights. Rings are colored by height (like the other terrains) and the outer
  border matches the ring height.
- Terrain generation no longer prints timing information to stdout.

Fixed
^^^^^

- Fixed domain randomization events that target different ``axes`` of the same
  model field (e.g. two ``dr.geom_size`` events scaling axis 0 and axis 1
  separately) silently clobbering each other. Each event now writes back only
  the axes it targeted, so per-axis events compose (:issue:`1042`).
- Regenerated the bundled MuJoCo type stubs, which had drifted from the
  installed mujoco version. CI now regenerates them and fails if they are
  stale, so they stay in sync going forward. Run ``make stubs`` to update them
  (:issue:`1048`).
- Fixed ``select_gpus`` crashing when ``CUDA_VISIBLE_DEVICES`` contains MIG
  UUIDs instead of numeric indices.
- Fixed pyramid-stairs terrains (``BoxPyramidStairsTerrainCfg``,
  ``BoxInvertedPyramidStairsTerrainCfg``, and ``BoxOpenStairsTerrainCfg``)
  leaving an empty, geometry-free border at difficulty 0, where the step
  height collapses to zero. The flat border frame is now always generated as
  solid geometry flush with the ground (:issue:`1033`).
- Fixed ``HfPerlinNoiseTerrainCfg`` failing to compile at difficulty 0, where
  the target height collapses to zero and MuJoCo rejects the non-positive
  heightfield size.
- Fixed ``BoxRandomGridTerrainCfg`` producing NaN colors (and failing to build)
  at difficulty 0, where the grid height is zero and the color normalization
  divided by zero.
- Fixed the center platform z-fighting with surrounding geometry in
  ``BoxRandomGridTerrainCfg`` (grid cells were left underneath the platform) and
  ``BoxRandomSpreadTerrainCfg`` (the platform duplicated the floor surface).
- Fixed ``BoxNarrowBeamsTerrainCfg`` square platform corners protruding between
  the beams at high difficulty; the platform now shrinks to stay within the
  beams' angular coverage.
- Fixed ``BoxSteppingStonesTerrainCfg`` reconfiguring abruptly at a difficulty
  threshold, where the stone grid re-tiled as its spacing crossed an integer
  boundary, and leaving an oversized gap around the center platform. The grid is
  now difficulty-independent and the platform snaps to it as a clean island.
- Fixed ``train --video``, ``play``, and ``demo`` crashing with ``OpenGL
  platform library not loaded`` on headless Linux hosts that don't pre-set
  ``MUJOCO_GL``. The default is now applied in ``mjlab/__init__.py`` (Linux
  only) so it takes effect before mujoco's GL backend selection runs.
- Fixed motion tracking re-anchoring to a stale robot pose after a mid-episode
  motion resample. ``MotionCommand._update_command`` now calls ``sim.forward()``
  after resampling so relative body poses read the post-teleport state
  (:issue:`1068`).

Version 1.4.0 (May 26, 2026)
----------------------------

Added
^^^^^

- Added ``BuiltinPdActuator``, the implicit-integration version of
  ``IdealPdActuator``. Same interface (position + velocity targets,
  kp/kd gains), but expresses the PD as native MuJoCo ``<position>``
  and ``<velocity>`` elements so the ``implicit`` / ``implicitfast``
  integrators include the kp/kd derivatives in their velocity update.
  The actuator stays stable at gain/timestep combinations where
  explicit Python PD would diverge, which matters when you want to
  run a real motor's stiff on-board PD gains in sim. ``effort_limit``
  is enforced as a sum-clamp on the two PD terms via
  ``jnt_actfrcrange`` (or ``tendon_actfrcrange``). Supported by
  ``dr.pd_gains`` and ``dr.effort_limits``.
- Added ``mdp.projected_gravity_from_sensor``, an observation that derives
  projected gravity from a ``framezaxis`` up-vector sensor (negated) rather
  than from the root body orientation. Unlike ``mdp.projected_gravity``, it
  reflects the sensor's site frame, so it can observe IMU mounting domain
  randomization (e.g. via ``dr.site_quat``). Go1 and G1 ship an
  ``imu_upvector`` sensor for this.
- Added ``DebugVisualizer.add_box`` for drawing an axis-oriented box
  primitive, mirroring ``add_ellipsoid``. Supported by both the native
  and Viser viewers. ``size`` is the box half-extents (:issue:`992`).
- Added ``--log-root`` CLI option to ``train``, ``play``, and ``evaluate``
  scripts for choosing where training logs are stored. Defaults to
  ``logs/rsl_rl`` (unchanged behavior). Useful for directing outputs to a
  scratch disk or shared mount.
- ``RewardManager``, ``TerminationManager``, and ``MetricsManager`` now
  validate that every term function returns a tensor of shape
  ``(num_envs,)`` when evaluated, raising a clear ``ValueError``
  naming the offending term instead of silently broadcasting or crashing
  with an opaque error later during training.
- Added ``ContactSensor.primary_names`` property to expose the resolved
  primary names in the order they appear along the per-contact axis of the
  output tensors. This makes it possible to map a contact-data column back
  to the primary it belongs to (:issue:`914`).
- Added per-world mesh variant support via ``VariantEntityCfg``. Each
  world in a batched simulation can now use a different mesh asset for
  the same logical entity (e.g. world 0 holds a cube, world 1 a
  sphere). Variants are passed as a ``dict[str, Callable]`` of named
  spec callables; the optional ``assignment`` field controls how worlds
  map to variants and accepts ``None`` (uniform), a ``dict[str, float]``
  of per-variant weights, or a custom ``Callable[[int], Sequence[int]]``.
  Mesh-derived constants (collision bounds, body inertials, subtree
  mass, inverse weights) are compiled per-variant and stored as
  per-world arrays in the Warp model, so domain randomization, the
  native viewer, the offscreen renderer, and the Viser viewer all pick
  up the variant assignment automatically. Variants must share the
  same kinematic structure (same bodies, joints, joint types); only
  mesh geoms may differ. Assignment is fixed at simulation init. See
  :ref:`heterogeneous_worlds` for usage. With help from @XiangruiJiang.
- Per-world mesh variants now support per-variant materials and textures.
  Each variant can reference its own named material, which is automatically
  prefixed and scattered via ``geom_matid`` alongside the existing
  ``geom_dataid`` table. Variants without a material get ``matid = -1``.
  Contribution by @omarrayyann.
- Added ``dr.geom_matid`` to randomize which baked material each geom uses
  per environment, sampling uniformly from ``asset_cfg.material_names``.
  Contribution by @bd-pmorais.

Changed
^^^^^^^

- ``Entity`` now raises a clear error at construction when its spec contains
  more than one freejoint. An entity models a single system rooted at one
  body, so it has at most one freejoint; a second one was previously accepted
  silently and only surfaced later as a cryptic shape mismatch when writing
  root state. Model each detached floating body as its own entry in
  ``SceneCfg.entities`` instead.
- Changed ``compute_root_relative_mpkpe`` to re-anchor the reference to the
  robot's root each step, removing yaw drift as well as translation so it
  measures intrinsic body pose error.
- Changed ``compute_joint_velocity_error`` from an L2 norm to a per-joint
  RMS, so it no longer scales with the number of joints.
- Bumped ``mujoco`` to 3.8 and ``mujoco-warp`` to 3.8.0. The ``multiccd``
  enable flag was removed in mujoco 3.8 (it became default-on), so configs
  that listed ``"multiccd"`` in ``MujocoCfg.enableflags`` need to drop it.
- Camera segmentation now matches ``mujoco_warp``'s typed segmentation
  output. ``CameraSensorData.segmentation`` stores ``(object_id,
  object_type)`` pairs in shape ``[B, H, W, 2]`` instead of the previous
  legacy geom-id-only layout. Contribution by @tkelestemur.
- Sped up ``RayCaster`` post-processing by removing boolean-mask indexing
  operations and replacing them with ``masked_fill_`` plus a clamped-distance
  formulation of ``hit_pos_w`` that places misses at the world origin. This
  removes all CUDA syncs from the ray post-process, letting the CPU thread
  proceed while GPU-based sensing runs. Contribution by @bd-pdomanico.
- Bumped ``rsl-rl-lib`` from 5.0.1 to 5.2.0. This brings ``torch.compile`` support for
  PPO and Distillation, and optional std clamping and constant std in
  ``GaussianDistribution``. No code changes required on the mjlab side.
- ``TerrainEntityCfg`` debug visualization sites (environment origins,
  terrain origins, flat patches) are now off by default. Set
  ``debug_vis=True`` to re-enable them. The sites inflated ``nsite`` and
  caused a measurable slowdown in the per-step ``site_local_to_global``
  kernel (:issue:`942`).
- Task package load failures during ``mjlab`` import now print the full
  traceback (and the entry point's module path) to ``stderr`` instead of
  just the exception message, making it easier to pinpoint the source of
  import errors when running commands like ``list-envs`` (:issue:`910`).
  Contribution by @saikishor.
- Clarified ``ContactSensor`` shape conventions: per-contact fields
  (``found``, ``force``, ``torque``, ``dist``, ``pos``, ``normal``,
  ``tangent``) have shape ``[B, P * num_slots, ...]`` while per-primary
  air-time fields (``current_air_time``, ``last_air_time``,
  ``current_contact_time``, ``last_contact_time``) have shape ``[B, P]``,
  where ``P`` is the number of resolved primaries (:issue:`914`).
- Event functions now share a single ``resolve_env_ids`` helper to expand
  ``env_ids=None`` to all environments, replacing five copies of the same
  guard. ``push_by_setting_velocity`` and ``apply_external_force_torque``
  accept ``env_ids=None`` too, so they work as global-time interval terms.
  Documented when to use ``apply_external_force_torque`` (a constant,
  self-managed wrench) versus ``apply_body_impulse`` (transient, automatic
  impulses) versus ``push_by_setting_velocity`` (an instantaneous velocity
  kick).

Fixed
^^^^^

- Removed use of deprecated ``warp-lang`` symbols (``wp.context.runtime``
  and ``wp.context.Device``) that were dropped in newer ``warp-lang``
  releases, causing ``AttributeError: module 'warp' has no attribute
  'context'`` at import/runtime. mjlab now uses
  ``wp.get_cuda_driver_version()`` and ``wp.Device`` instead
  (:issue:`967`). Contribution by @rdeits.
- Fixed the tracking ``evaluate`` script scoring each metric against the
  next motion frame; the reference is now snapshotted before each step to
  match the reward.
- Fixed the tracking end-effector metrics silently scoring zero for an
  unknown body name; they now raise ``ValueError``.
- Fixed ``compute_mpkpe`` measuring root-relative instead of global error;
  it now uses the global reference ``body_pos_w`` (:issue:`1006`).
- Fixed heavy flicker in offscreen training videos on rough-terrain tasks.
  The renderer recomputed its context "neighbor" robots every frame from
  ``env_origins``, which the terrain curriculum mutates on reset, so the
  neighbor set kept changing and robots popped in and out. The neighbor
  set is now computed once and cached (:issue:`979`).
- Fixed command delay only applying to an actuator's position target.
  ``IdealPdActuator`` and ``DcMotorActuator`` also use velocity and effort, which
  arrived undelayed and out of sync; all command targets now share one delay.
  Zero-reference setups are unaffected.
- Fixed duplicate random seeds across nodes in multi-node training. The
  per-process seed offset in ``scripts/train.py`` now uses the global
  ``RANK`` instead of ``LOCAL_RANK``. Contribution by @bd-pdomanico.
- Fixed ``apply_body_impulse`` firing an impulse on the very first step (and
  the first step after every reset) instead of starting with a cooldown as
  documented. The cooldown is now sampled lazily on the first call so impulse
  timing is decorrelated from episode resets (:issue:`973`).
- Fixed ``dr.pd_gains`` and ``dr.effort_limits`` silently no-oping when
  passed an ``Operation`` object (e.g. ``dr.scale``) instead of a string.
  Both functions now accept ``Operation | str`` like every other DR event
  and raise ``ValueError`` for unsupported operations (:issue:`971`).
- Fixed ``ContactSensor`` with ``global_frame=True`` and
  ``reduce`` ∈ {``"none"``, ``"mindist"``, ``"maxforce"``} producing forces
  rotated onto the wrong axis. The contact-frame→world rotation matrix had
  its columns ordered ``[tangent, tangent2, normal]`` instead of
  ``[normal, tangent, tangent2]``, projecting the normal-force component
  onto a tangent direction. Contribution by @bd-pdomanico.
- Fixed ``extras["log"]`` entries written by reward terms (e.g. ``Metrics/*``
  values in velocity tasks) being silently discarded on any step where at
  least one environment resets. ``_reset_idx`` was clearing the dict after
  ``reward_manager.compute()`` had already populated it. The clear now
  happens at the top of ``step()`` and ``reset()`` so that all entries
  survive (:issue:`957`).
- Fixed ``ContactSensor.compute_first_contact`` and ``compute_first_air``
  occasionally missing events when a contact began or ended right at the
  last physics substep of a control step. ``current_contact_time`` /
  ``current_air_time`` accumulate in float32 and can drift a few ULPs past
  ``dt``, but the default ``abs_tol`` of ``1e-8`` sat at the noise floor
  and rejected the comparison. Raised the default to ``1e-6``, which stays
  well below typical control ``dt`` while comfortably covering float32
  accumulation noise (:issue:`933`). Contribution by @paLeziart.
- Fixed ``out_of_terrain_bounds`` using stale terrain dimensions. It read
  ``TerrainGeneratorCfg.num_cols`` directly, which is ignored in curriculum
  mode (the generator uses ``len(sub_terrains)`` columns instead), and it
  did not account for ``border_width``. The termination now reads the
  effective grid shape from ``terrain.terrain_origins`` and includes the
  border in the footprint, so robots no longer reset while still on valid
  terrain (or fail to reset after running off it) (:issue:`923`).
- ``ObservationManager`` now skips observation groups that end up with
  zero active terms (e.g. all terms set to ``None``) with a log message,
  instead of crashing later in ``torch.stack``/``torch.cat``. This lets
  a shared runner config define groups that become empty under certain
  runtime flags (e.g. model-specific terms all disabled for one variant).
  The whole group can still be set to ``None`` to disable it explicitly.
- Fixed a runtime broadcast error in ``ContactSensor`` when combining
  ``num_slots > 1`` with ``track_air_time=True`` and more than one primary.
  Air-time tracking now reduces ``found`` across slots so that a primary is
  considered in contact when any of its slots reports a match (:issue:`914`).
- Updated the ``create_new_task.ipynb`` Colab tutorial to import
  ``XmlActuatorCfg`` instead of the removed ``XmlVelocityActuatorCfg``.
  Added a regression test (``tests/test_notebooks.py``) that parses each
  notebook cell and verifies that every ``from mjlab... import X``
  reference resolves, so future renames in the mjlab public API can't
  silently rot the tutorials (:issue:`913`).
- Fixed ``ObservationManager`` silently sharing a single ``NoiseModelCfg``
  instance across observation groups that declared terms with the same
  name. ``_group_obs_class_instances`` was keyed by term name alone, so
  the last group processed in ``_prepare_terms`` overwrote earlier
  groups' instances. Symptoms included the wrong noise config being
  applied, shared per-episode state for ``NoiseModelWithAdditiveBias``
  (e.g. bias drawn from the wrong ``bias_noise_cfg``), and missed
  ``reset()`` calls for overwritten instances. Instances are now keyed
  by ``(group_name, term_name)`` so each group owns its own noise model.
- Fixed ``CurriculumManager.get_active_iterable_terms`` raising
  ``TypeError`` when a term's state was a dict. The dict branch indexed
  the output list by term name instead of appending to the local ``data``
  list. No in-tree caller currently invokes this method, so the bug was
  latent.

Version 1.3.0 (April 14, 2026)
------------------------------

Added
^^^^^

- Added ``ManagerBasedRlEnvCfg.auto_reset`` flag. When ``True`` (default),
  ``step()`` continues to reset done environments in place and returns the
  post-reset observation. When ``False``, ``step()`` skips the reset block
  and returns the terminal observation directly; the caller must call
  ``reset(env_ids=...)`` for done environments before the next ``step()``
  or a ``RuntimeError`` is raised. Enables access to the true terminal
  state for algorithms that need it. Note that mjlab's bundled ``train.py``
  uses rsl_rl's ``OnPolicyRunner``, which does not drive manual resets, so
  ``auto_reset=False`` is intended for custom training loops (:issue:`900`).
- Added ``ActuatorCfg.viscous_damping`` for passive velocity proportional
  damping (``f = -b·v``), distinct from the PD derivative gain ``damping``
  used by position and velocity actuators. Maps to ``<joint damping>`` for
  JOINT transmission and ``<tendon damping>`` for TENDON transmission.
  Defaults to ``None`` (preserves the XML value).
- Added :class:`~mjlab.managers.RecorderManager` for logging observations,
  actions, or arbitrary environment data during rollouts. Implement a
  :class:`~mjlab.managers.RecorderTerm` subclass and register it in the
  ``recorders`` dict on ``ManagerBasedRlEnvCfg``. The manager provides
  ``record_pre_reset``, ``record_post_reset``, and ``record_post_step``
  lifecycle hooks with no opinion on how data is stored.
- Added :func:`~mjlab.envs.mdp.curriculums.termination_curriculum` for
  scheduling changes to termination term parameters during training,
  matching the existing ``reward_curriculum`` pattern. Both now share a
  single internal engine with init-time validation of stage ordering,
  field existence, and param keys.
- Added ``reduce`` field to ``MetricsTermCfg``. Setting ``reduce="last"``
  reports the value from the final step of the episode rather than the
  episode mean, which is useful for binary success metrics.
- Added :class:`~mjlab.envs.mdp.actions.RelativeJointPositionAction` for
  joint position control relative to the current configuration. The target is
  ``current_pos + action * scale``, so a zero action holds the current
  configuration rather than commanding the default pose.
- Added :func:`~mjlab.envs.mdp.dr.pair_friction` for randomizing geom-pair
  friction overrides (``pair_friction`` in ``mjModel``), with an
  ``isotropic=True`` option that mirrors the symmetric tangent and roll
  axes so single-axis randomization does not leave the paired axis stale.
- Added ``STAIRS_TERRAINS_CFG`` terrain preset for progressive stair
  curriculum training and ``@terrain_preset`` decorator for composing
  terrain configurations from reusable presets.
- Added cartpole balance and swingup tasks (``Mjlab-Cartpole-Balance`` and
  ``Mjlab-Cartpole-Swingup``) with a :ref:`tutorial <tutorial-cartpole>`
  that walks through building an environment from scratch.
- Added :ref:`motion imitation <motion-imitation>` documentation with
  preprocessing instructions. The README now links here instead of the
  BeyondMimic repository, which produced incompatible NPZ files when used
  with mjlab (:issue:`777`).
- Added ``margin``, ``gap``, and ``solmix`` fields to ``CollisionCfg``
  for per geom contact parameter configuration (:issue:`766`).
- NaN guard now captures mocap body poses (``mocap_pos``, ``mocap_quat``)
  when the model has mocap bodies, enabling full state reconstruction in
  the dump viewer for fixed-base entities.
- Implemented ``ActionTermCfg.clip`` for clamping processed actions after
  scale and offset (:issue:`771`).
- Added ``qfrc_actuator`` and ``qfrc_external`` generalized force accessors
  to ``EntityData``. ``qfrc_actuator`` gives actuator forces in joint space
  (projected through the transmission). ``qfrc_external`` recovers the
  generalized force from body external wrenches (``xfrc_applied``)
  (:issue:`776`).
- Added ``RewardBarPanel`` to the Viser viewer, showing horizontal bars for
  each reward term with a running mean over ~1 second (:issue:`800`).
- Added ``per_substep`` flag to ``MetricsTermCfg`` for evaluating metrics
  once per physics substep inside the decimation loop. The per substep
  values are averaged within each environment step, so episode averages
  remain comparable to regular per step metrics.
- Added ``project-instinct/InstinctMJ`` to the research page's list of
  projects built on mjlab.
- Added a Checkpoints tab to the Viser play viewer for hot-swapping
  checkpoints without restarting. Works with local directories and W&B
  runs (:issue:`751`). Contribution by @omarrayyann.
- Added ``"segmentation"`` camera data type for per-pixel geom ID output
  alongside RGB and depth, and a multi-cube goal-conditioned lifting task
  (``Mjlab-Multi-Cube-Seg-Yam``) that uses it (:issue:`862`).
  Contribution by @pthangeda.

Changed
^^^^^^^

- Renamed the ``list_envs`` console script to ``list-envs`` for consistency
  with the other hyphenated entry points (``viz-nan``, ``export-scene``).
  Invoke via ``uv run list-envs``.
- ``ActuatorCfg.armature`` and ``ActuatorCfg.frictionloss`` now default to
  ``None`` instead of ``0.0``. ``None`` preserves the value defined in the
  XML. Previously, builtin actuators would silently overwrite XML joint and
  tendon properties with zero when these fields were not explicitly set.
  To restore the old behavior, pass ``armature=0.0`` or ``frictionloss=0.0``
  explicitly.
- Actuator delay is now configured inline on any ``ActuatorCfg`` subclass
  (e.g. ``BuiltinPositionActuatorCfg(..., delay_min_lag=2, delay_max_lag=5)``)
  instead of wrapping with ``DelayedActuatorCfg``. ``DelayedActuator``,
  ``DelayedActuatorCfg``, and ``DelayedBuiltinActuatorGroup`` are removed.
- Removed ``delay_target`` from ``ActuatorCfg``. Delay now always applies to
  the actuator's ``command_field`` automatically. Multi-target delay
  (``delay_target=("position", "velocity")``) is no longer supported.
- ``XmlPositionActuatorCfg``, ``XmlVelocityActuatorCfg``, ``XmlMotorActuatorCfg``,
  and ``XmlMuscleActuatorCfg`` are replaced by a single ``XmlActuatorCfg`` that auto
  detects the actuator type from XML. Pass ``command_field=...`` to override detection.
- Replaced the viser viewer internals with the ``mjviser`` package. Scene
  creation, mesh conversion, and overlay rendering (contacts, forces,
  inertia, tendons, joints, frames) are now provided by mjviser. The viewer
  exposes a new Visualization tab for overlay controls and a Groups tab for
  geom/site visibility. Debug visualization and warp tensor conversion remain
  in mjlab's ``MjlabViserScene`` subclass (:issue:`839`).
- In curriculum terrain mode, each terrain type now gets exactly one column
  (``num_cols`` is set to ``len(sub_terrains)``). The ``proportion`` field
  now controls robot spawning distribution across columns rather than column
  count. Random mode is unchanged (:issue:`811`).
- ``BoxSteppingStonesTerrainCfg`` stone size now decreases with difficulty,
  interpolating from the large end of ``stone_size_range`` at difficulty 0
  to the small end at difficulty 1 (:issue:`785`).
- Removed deprecated ``TerrainImporter`` and ``TerrainImporterCfg`` aliases.
  Use ``TerrainEntity`` and ``TerrainEntityCfg`` instead (:issue:`667`).
- ``Entity.clear_state()`` is deprecated. Use ``Entity.reset()`` instead.
  ``clear_state`` only zeroed actuator targets without resetting actuator
  internal state (e.g. delay buffers), which could cause stale commands
  after teleporting the robot to a new pose.
- Removed ``EntityData.generalized_force``. The property was bugged (indexed
  free joint DOFs instead of articulated DOFs) and the name was ambiguous.
  Use ``qfrc_actuator`` or ``qfrc_external`` instead (:issue:`776`).
- ``get_wandb_checkpoint_path`` now filters checkpoints server-side via the
  ``pattern`` parameter, avoiding unnecessary pagination and tolerance to
  corrupted metadata (:issue:`898`).

Fixed
^^^^^

- ``train`` and ``play`` now print a top-level usage message when invoked
  with ``-h`` / ``--help`` and no task argument, pointing users at
  ``list-envs`` and ``<TASK> --help`` (:issue:`905`).
- Fixed ghost geom filtering in the Viser viewer. Ghost geoms were selected
  by collision flags, so collision-disabled robot geoms appeared as ghosts.
  The viewer now uses visual alpha to determine which geoms to render.
- Scene now warns when an attached entity or terrain spec has non-default
  ``<option>`` fields (e.g. ``<flag contact="disable"/>``), which are
  silently dropped by ``MjSpec.attach()``. Use ``MujocoCfg`` to set
  simulation options instead (:issue:`885`).
- Fixed ``SceneEntityCfg`` names and IDs ordering mismatch when
  ``preserve_order=False`` (:issue:`876`). Contribution by @jsw7460.
- Fixed ONNX export path resolution in the velocity, manipulation, and
  tracking runners when a parent directory name contains the word
  ``"model"`` (:issue:`867`). Contribution by @gokulp01.
- ``export-scene`` now writes only referenced assets and places them
  correctly under the output directory. Previously, asset keys containing
  path traversal could write files outside the output directory, and all
  spec assets were included regardless of whether the scene XML referenced
  them (:issue:`858`).
- ``electrical_power_cost`` now uses ``qfrc_actuator`` (joint space) instead
  of ``actuator_force`` (actuation space) for mechanical power computation.
  Previously the reward was incorrect for actuators with gear ratios other
  than 1 (:issue:`776`).
- ``create_velocity_actuator`` no longer sets ``ctrllimited=True`` with
  ``inheritrange=1.0``. This caused a ``ValueError`` for continuous joints
  (e.g. wheels) that have no position range defined (:issue:`787`).
- ``write_root_com_velocity_to_sim`` no longer fails with tensor ``env_ids``
  on floating base entities (:issue:`793`).
- Joint limits for unlimited joints are now set to [-inf, inf] instead of
  [0, 0]. Previously the zero range caused incorrect clamping for entities
  with unlimited hinge or slide joints.
- Contact force visualization now copies ``ctrl`` into the CPU ``MjData``
  before calling ``mj_forward``. Actuators that compute torques in Python
  (``DcMotorActuator``, ``IdealPdActuator``) previously showed incorrect
  contact forces because the viewer ran with ``ctrl=0``
  (:issue:`786`).
- ``BoxSteppingStonesTerrainCfg`` no longer creates a large gap around the
  platform. Stones are now only skipped when their center falls inside the
  platform; edges that extend under the platform are allowed since the
  platform covers them (:issue:`785`).
- ``dr.pseudo_inertia`` no longer loads cuSOLVER, eliminating ~4 GB of
  persistent GPU memory overhead. Cholesky and eigendecomposition are now
  computed analytically for the small matrices involved (4x4 and 3x3)
  (:issue:`753`).
- Set terrain geom mass to zero so that the static terrain body does not
  inflate ``stat.meanmass``, which made force arrow visualization invisible
  on rough terrain (:issue:`734`, :issue:`537`).
- Native viewer now syncs ``qpos0`` when domain randomized, fixing incorrect
  body positions after ``dr.joint_default_pos`` randomization
  (:issue:`760`).
- ``command_manager.compute()`` is now called during ``reset()`` so that
  derived command state (e.g. relative body positions in tracking
  environments) is populated before the first observation is returned
  (:issue:`761`).
- ``RayCastSensor`` with ``ray_alignment="yaw"`` or ``"world"`` now correctly
  aligns the frame offset when attached to a site or geom with a local offset
  from its parent body. Previously only ray directions and pattern offsets were
  aligned, causing the frame position to swing with body pitch/roll
  (:issue:`775`).

Version 1.2.0 (March 6, 2026)
-----------------------------

.. admonition:: Breaking API changes
   :class: attention

   - ``randomize_field`` no longer exists. Replace calls with typed functions
     from the new ``dr`` module (e.g. ``dr.geom_friction``, ``dr.body_mass``).
   - ``EventTermCfg`` no longer accepts ``domain_randomization``. The
     ``@requires_model_fields`` decorator on each ``dr`` function takes care
     of field expansion automatically.
   - ``Scene.to_zip()`` is deprecated. Use ``Scene.write(path, zip=True)``.
   - ``RslRlModelCfg`` no longer accepts ``stochastic``, ``init_noise_std``,
     or ``noise_std_type``. Use ``distribution_cfg`` instead
     (e.g. ``{"class_name": "GaussianDistribution", "init_std": 1.0,
     "std_type": "scalar"}``). Existing checkpoints are automatically
     migrated on load.

Added
^^^^^

- Added ``"step"`` event mode that fires every environment step.
- Added ``apply_body_impulse`` event for applying transient external wrenches
  to bodies with configurable duration and optional application point offset.
- ONNX auto-export and metadata attachment for manipulation tasks (lift cube)
  on every checkpoint save, matching the velocity and tracking task behavior.
- Multi-frame ``RayCastSensor``: pass a tuple of ``ObjRef`` to ``frame`` for
  per-site raycasting with independent body exclusion. New properties:
  ``num_frames``, ``num_rays_per_frame``. New ``RayCastData`` fields:
  ``frame_pos_w`` and ``frame_quat_w``.
- ``RingPatternCfg`` ray pattern for concentric ring sampling around each
  frame.
- ``TerrainHeightSensor``, a ``RayCastSensor`` subclass that computes
  per-frame vertical clearance above terrain (``sensor.data.heights``).
  Velocity task configs now use it for ``feet_clearance``,
  ``feet_swing_height``, and ``foot_height``, replacing the previous
  world-Z proxy that was incorrect on rough terrain.
- Cloud training support via `SkyPilot <https://skypilot.readthedocs.io/>`_
  and Lambda Cloud, with documentation covering setup, monitoring, and
  cost management.
- W&B hyperparameter sweep scripts that distribute one agent per GPU
  across a multi-GPU instance.
- Contributing guide with documentation for shared Claude Code commands
  (``/update-mjwarp``, ``/commit-push-pr``).
- Added optional ``ViewerConfig.fovy`` and apply it in native viewer camera
  setup when provided.
- Native viewer now tracks the first non-fixed body by default (matching
  the Viser viewer behavior introduced in
  ``716aaaa58ad7bfaf34d2f771549d461204d1b4ba``).
- New ``dr`` module (``mjlab.envs.mdp.dr``) replacing ``randomize_field``
  with typed per-field domain randomization functions. Each function
  automatically recomputes derived fields via ``set_const``. Highlights:

  - Camera and light randomization: ``dr.cam_fovy``, ``dr.cam_pos``,
    ``dr.cam_quat``, ``dr.cam_intrinsic``, ``dr.light_pos``,
    ``dr.light_dir``. Camera and light names are now supported in
    ``SceneEntityCfg`` (``camera_names`` / ``light_names``).
  - ``dr.pseudo_inertia`` for physics-consistent randomization of
    ``body_mass``, ``body_ipos``, ``body_inertia``, and ``body_iquat``
    via the pseudo-inertia matrix parameterization (Rucker & Wensing
    2022). Replaces the removed ``dr.body_inertia`` /
    ``dr.body_iquat``.
  - ``dr.geom_size`` with automatic recomputation of ``geom_rbound``
    and ``geom_aabb`` for broadphase consistency.
  - ``dr.tendon_armature`` and ``dr.tendon_frictionloss``.
  - ``dr.body_quat``, ``dr.geom_quat``, and ``dr.site_quat`` with RPY
    perturbation composed onto the default quaternion.
  - Extensible ``Operation`` and ``Distribution`` types. Users can define
    custom operations and distributions as class instances and pass them
    anywhere a string is accepted. Built-in instances (``dr.abs``,
    ``dr.scale``, ``dr.add``, ``dr.uniform``, ``dr.log_uniform``,
    ``dr.gaussian``) are exported from the ``dr`` module.
  - ``dr.mat_rgba`` for per-world material color randomization. Tints
    the texture color, useful for randomizing appearance of textured
    surfaces. Material names are now supported in ``SceneEntityCfg``
    (``material_names``).
  - Fixed ``dr.effort_limits`` drifting on repeated randomization.
  - Fixed ``dr.body_com_offset`` not triggering ``set_const``.

- ``export-scene`` CLI script to export any task scene or asset_zoo entity
  (``g1``, ``go1``, ``yam``) to a directory or zip archive for inspection
  and debugging.

- ``yam_lift_cube_vision_env_cfg`` now randomizes cube color (``dr.geom_rgba``)
  on every reset when ``cam_type="rgb"``.

- The native viewer now reflects per-world DR changes to visual model fields
  on each reset. Geom appearance, body and site poses, camera parameters,
  and light positions are all synced from the GPU model before rendering.
  Inertia boxes (press ``I``) and camera frustums (press ``Q``) update
  correctly when the corresponding fields are randomized. See
  :doc:`randomization` for viewer-specific caveats.

- ``MaterialCfg.geom_names_expr`` for assigning materials to geoms by
  name pattern during ``edit_spec``.

- ``TerrainEntityCfg`` now exposes ``textures``, ``materials``, and
  ``lights`` as configurable fields (previously hardcoded). Set
  ``textures=()``, ``materials=()`` to use flat ``dr.geom_rgba``
  instead of the default checker texture.

- ``DebugVisualizer`` now supports ellipsoid visualization via
  ``add_ellipsoid``.

- Interactive velocity joystick sliders in the Viser viewer. Enable the
  joystick under Commands/Twist to override velocity commands with manual
  sliders for ``lin_vel_x``, ``lin_vel_y``, and ``ang_vel_z``
  (`#666 <https://github.com/mujocolab/mjlab/issues/666>`_).
- Per-term debug visualization toggles in the Viser viewer. Individual
  command term visualizers (e.g. velocity arrows) can now be toggled
  independently under Scene/Debug Viz.
- Viewer single-step mode: press RIGHT arrow (native) or click "Step"
  (Viser) to advance exactly one physics step while paused.
- Viewer error recovery: exceptions during stepping now pause the viewer
  and log the traceback instead of crashing the process.
- Native viewer runs forward kinematics while paused, keeping
  perturbation visuals accurate.
- Viewer speed multipliers use clean power-of-2 fractions (1/32x to 1x).

- Visualizers display the realtime factor alongside FPS.

- ``joint_torques_l2`` now respects ``SceneEntityCfg.actuator_ids``,
  allowing penalization of a subset of actuators instead of all of them
  (`#703 <https://github.com/mujocolab/mjlab/pull/703>`_). Contribution by
  `@saikishor <https://github.com/saikishor>`_.

- Terrain is now a proper ``Entity`` subclass (``TerrainEntity``). This
  allows domain randomization functions to target terrain parameters
  (friction, cameras, lights) via ``SceneEntityCfg("terrain", ...)``.
  ``TerrainImporter`` / ``TerrainImporterCfg`` remain as aliases but will be
  deprecated in a future version.
- Added ``upload_model`` option to ``RslRlBaseRunnerCfg`` to control W&B model
  file uploads (``.pt`` and ``.onnx``) while keeping metric logging enabled
  (`#654 <https://github.com/mujocolab/mjlab/pull/654>`_).
- ``Scene.write(output_dir, zip=False)`` exports the scene XML and mesh
  assets to a directory (or zip archive). Replaces ``Scene.to_zip()``.
- ``Entity.write_xml()`` and ``Scene.write()`` now apply XML fixups
  (empty defaults, duplicate nested defaults) and strip buffer textures
  that ``MjSpec.to_xml()`` cannot serialize.
- ``fix_spec_xml`` and ``strip_buffer_textures`` utilities in
  ``mjlab.utils.xml``.

Changed
^^^^^^^

- Native viewer now syncs ``xfrc_applied`` to the render buffer and draws
  arrows for any nonzero applied forces. Mouse perturbation forces are
  converted to ``qfrc_applied`` (generalized joint space) so they coexist
  with programmatic forces on ``xfrc_applied`` without conflict.
- ``ViewerConfig.OriginType.WORLD`` now configures a free camera at the
  specified lookat point instead of auto tracking a body. A new ``AUTO``
  origin type (now the default) preserves the previous auto tracking
  behavior.
- Upgraded ``rsl-rl-lib`` from 4.0.1 to 5.0.1. ``RslRlModelCfg`` now
  uses ``distribution_cfg`` dict instead of ``stochastic`` /
  ``init_noise_std`` / ``noise_std_type``. Existing checkpoints are
  automatically migrated on load.
- Reorganized the Viser Controls tab into a cleaner folder hierarchy:
  Info, Simulation, Commands, Scene (with Environment, Camera, Debug Viz,
  Contacts sub-folders), and Camera Feeds. The Environment folder is
  hidden for single-env tasks and the Commands folder is hidden when no
  command terms are active.
- Viser camera tracking is now enabled by default so the agent stays in
  frame on launch.
- Self collision and illegal contact sensors now use ``history_length`` to
  catch contacts across decimation substeps. Reward and termination functions
  read ``force_history`` with a configurable ``force_threshold``.
- Replaced the single ``scale`` parameter in ``DifferentialIKActionCfg`` with
  separate ``delta_pos_scale`` and ``delta_ori_scale`` for independent scaling
  of position and orientation components.
- Improved offscreen multi environment framing by selecting neighboring
  environments around the focused env instead of first N envs.
- Tuned tracking task viewer defaults for tighter camera framing.
- Disabled shadow casting on the G1 tracking light to avoid duplicate
  stacked shadows when robots are close.

Fixed
^^^^^

- Fixed actuator target resolution for entities whose ``spec_fn`` uses
  internal ``MjSpec.attach(prefix=...)``
  (`#709 <https://github.com/mujocolab/mjlab/issues/709>`_).
- Fixed viewer physics loop starving the renderer by replacing the single
  sim-time budget with a two-clock design (tracked vs actual sim time).
  Physics now self-corrects after overshooting, keeping FPS smooth at all
  speed multipliers.
- Bundled ``ffmpeg`` for ``mediapy`` via ``imageio-ffmpeg``, removing the
  requirement for a system ``ffmpeg`` install. Thanks to
  `@rdeits-bd <https://github.com/rdeits-bd>`_ for the suggestion.
- Fixed ``height_scan`` returning ~0 for missed rays; now defaults to
  ``max_distance``. Replaced ``clip=(-1, 1)`` with ``scale`` normalization
  in the velocity task config. Thanks to `@eufrizz <https://github.com/eufrizz>`_
  for reporting and the initial fix (`#642 <https://github.com/mujocolab/mjlab/pull/642>`_).
- Fixed ghost mesh visualization for fixed-base entities by extending
  ``DebugVisualizer.add_ghost_mesh`` to optionally accept ``mocap_pos`` and
  ``mocap_quat`` (`#645 <https://github.com/mujocolab/mjlab/pull/645>`_).
- Fixed viser viewer crashing on scenes with no mocap bodies by adding
  an ``nmocap`` guard, matching the native viewer behavior.
- Fixed offscreen rendering artifacts in large vectorized scenes by applying
  a render local extent override in ``OffscreenRenderer`` and restoring the
  original extent on close.
- Fixed ``RslRlVecEnvWrapper.unwrapped`` to return the base environment,
  ensuring checkpoint state restore and logging work correctly when wrappers
  such as ``VideoRecorder`` are enabled.

Version 1.1.1 (February 14, 2026)
---------------------------------

Added
^^^^^

- Added reward term visualization to the native viewer (toggle with ``P``) (`#629 <https://github.com/mujocolab/mjlab/pull/629>`_).
- Added ``DifferentialIKAction`` for task-space control via damped
  least-squares IK. Supports weighted position/orientation tracking,
  soft joint-limit avoidance, and null-space posture regularization.
  Includes an interactive viser demo (``scripts/demos/differential_ik.py``) (`#632 <https://github.com/mujocolab/mjlab/pull/632>`_).

Fixed
^^^^^

- Fixed ``play.py`` defaulting to the base rsl-rl ``OnPolicyRunner`` instead
  of ``MjlabOnPolicyRunner``, which caused a ``TypeError`` from an unexpected
  ``cnn_cfg`` keyword argument (`#626 <https://github.com/mujocolab/mjlab/pull/626>`_). Contribution by
  `@griffinaddison <https://github.com/griffinaddison>`_.

Changed
^^^^^^^

- Removed ``body_mass``, ``body_inertia``, ``body_pos``, and ``body_quat``
  from ``FIELD_SPECS`` in domain randomization. These fields have derived
  quantities that require ``set_const`` to recompute; without that call,
  randomizing them silently breaks physics (`#631 <https://github.com/mujocolab/mjlab/pull/631>`_).
- Replaced ``moviepy`` with ``mediapy`` for video recording. ``mediapy``
  handles cloud storage paths (GCS, S3) natively (`#637 <https://github.com/mujocolab/mjlab/pull/637>`_).

.. figure:: _static/changelog/native_reward.png
   :width: 80%

Version 1.1.0 (February 12, 2026)
---------------------------------

Added
^^^^^

- Added RGB and depth camera sensors and BVH-accelerated raycasting (`#597 <https://github.com/mujocolab/mjlab/pull/597>`_).
- Added ``MetricsManager`` for logging custom metrics during training (`#596 <https://github.com/mujocolab/mjlab/pull/596>`_).
- Added terrain visualizer (`#609 <https://github.com/mujocolab/mjlab/pull/609>`_). Contribution by
  `@mktk1117 <https://github.com/mktk1117>`_.

.. figure:: _static/changelog/terrain_visualizer.jpg
   :width: 80%

- Added many new terrains including ``HfDiscreteObstaclesTerrainCfg``,
  ``HfPerlinNoiseTerrainCfg``, ``BoxSteppingStonesTerrainCfg``,
  ``BoxNarrowBeamsTerrainCfg``, ``BoxRandomStairsTerrainCfg``, and
  more. Added flat patch sampling for heightfield terrains (`#542 <https://github.com/mujocolab/mjlab/pull/542>`_, `#581 <https://github.com/mujocolab/mjlab/pull/581>`_).
- Added site group visualization to the Viser viewer (Geoms and Sites
  tabs unified into a single Groups tab) (`#551 <https://github.com/mujocolab/mjlab/pull/551>`_).
- Added ``env_ids`` parameter to ``Entity.write_ctrl_to_sim`` (`#567 <https://github.com/mujocolab/mjlab/pull/567>`_).

Changed
^^^^^^^

- Upgraded ``rsl-rl-lib`` to 4.0.0 and replaced the custom ONNX
  exporter with rsl-rl's built-in ``as_onnx()`` (`#589 <https://github.com/mujocolab/mjlab/pull/589>`_, `#595 <https://github.com/mujocolab/mjlab/pull/595>`_).
- ``sim.forward()`` is now called unconditionally after the decimation
  loop. See :ref:`faq-sim-forward` for details (`#591 <https://github.com/mujocolab/mjlab/pull/591>`_).
- Unnamed freejoints are now automatically named to prevent
  ``KeyError`` during entity init (`#545 <https://github.com/mujocolab/mjlab/pull/545>`_).

Fixed
^^^^^

- Fixed ``randomize_pd_gains`` crash with ``num_envs > 1`` (`#564 <https://github.com/mujocolab/mjlab/pull/564>`_).
- Fixed ``ctrl_ids`` index error with multiple actuated entities (`#573 <https://github.com/mujocolab/mjlab/pull/573>`_).
  Reported by `@bwrooney82 <https://github.com/bwrooney82>`_.
- Fixed Viser viewer rendering textured robots as gray (`#544 <https://github.com/mujocolab/mjlab/pull/544>`_).
- Fixed Viser plane rendering ignoring MuJoCo size parameter (`#540 <https://github.com/mujocolab/mjlab/pull/540>`_).
- Fixed ``HfDiscreteObstaclesTerrainCfg`` spawn height (`#552 <https://github.com/mujocolab/mjlab/pull/552>`_).
- Fixed ``RaycastSensor`` visualization ignoring the all-envs toggle (`#607 <https://github.com/mujocolab/mjlab/pull/607>`_).
  Contribution by `@oxkitsune <https://github.com/oxkitsune>`_.

Version 1.0.0 (January 28, 2026)
--------------------------------

Initial release of mjlab.
