from functools import partial

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import (
  TennisRecoveryOnPolicyRunner,
  VelocityOnPolicyRunner,
)

from .env_cfgs import (
  agibot_x2_flat_velocity_carry_env_cfg,
  agibot_x2_flat_velocity_env_cfg,
)
from .rl_cfg import agibot_x2_velocity_ppo_runner_cfg
from .tennis_ball_env_cfg import agibot_x2_tennis_ball_env_cfg
from .tennis_juggle_env_cfg import agibot_x2_tennis_juggle_env_cfg
from .tennis_juggle_nominal_env_cfg import agibot_x2_tennis_juggle_nominal_env_cfg
from .tennis_juggle_stage_env_cfg import (
  STAGE_DELTAS,
  agibot_x2_tennis_juggle_stage_env_cfg,
)
from .tennis_juggle_target_height_env_cfg import (
  STAGE_TARGET_HEIGHT_RANGE_M,
  agibot_x2_tennis_juggle_target_height_env_cfg,
)
from .tennis_recovery_env_cfg import agibot_x2_tennis_recovery_env_cfg
from .tennis_recovery_rl_cfg import agibot_x2_tennis_recovery_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation",
  env_cfg=agibot_x2_flat_velocity_env_cfg(),
  play_env_cfg=agibot_x2_flat_velocity_env_cfg(play=True),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Carry-pose velocity task: the same no-state-estimation contract, but the robot
# is the tennis entity in BAT_CARRY_POSE with the welded racket, the ``pose``
# reward keeps only the legs and head, and a quadratic arm/waist term holds the
# racket steady. It starts from scratch because ``default_joint_pos`` changes
# what a zero action means. See docs/design/x2_carry_velocity_plan.md.
register_mjlab_task(
  task_id=("Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-CarryVelocity"),
  env_cfg=agibot_x2_flat_velocity_carry_env_cfg(),
  play_env_cfg=agibot_x2_flat_velocity_carry_env_cfg(play=True),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_carry_pose"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# Tennis-ball-on-the-bat velocity task. The base no-state-estimation velocity
# task plus the tennis equipment: the racket assembly rigidly attached to the
# right wrist and a floating tennis ball spawned on the string bed. Physics runs
# at 1 ms so the calibrated negative-solref bed contact stays stable, and the
# robot/ball resets are pinned so the ball starts on the bed in every env. The
# ball is not yet observed or rewarded; that is the next step and it changes the
# actor input width. See docs/source/x2_tennis_ball_balance.rst.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Balance",
  env_cfg=agibot_x2_tennis_ball_env_cfg(),
  play_env_cfg=agibot_x2_tennis_ball_env_cfg(play=True),
  # The ball observation widens the actor from 102 to 105 (position only) and the
  # critic from 117 to 123 (position and velocity), so the observation normalizer
  # has a different width. Runs made before the ball was observed, or with the
  # ball velocity in the actor, cannot be resumed, hence a separate experiment
  # name.
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_tennis_ball_centre_pos_only"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# Juggling variant of the tennis-ball task: the ball bounces on the plate and
# returns to an apex about 30 cm above it, instead of being carried. Same scene,
# observations, resets, termination and domain randomization as the balancing
# task above; the reward set pays per bounce and charges for parking the ball, the
# twist command is pinned to a standstill band, and there is no command
# curriculum. The observation contract is unchanged, so the widths are the
# balancing task's 105/123, but the reward set is not, hence a separate experiment
# name. See tennis_juggle_env_cfg.py and docs/source/x2_tennis_ball_juggle.rst.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle",
  env_cfg=agibot_x2_tennis_juggle_env_cfg(),
  play_env_cfg=agibot_x2_tennis_juggle_env_cfg(play=True),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_tennis_ball_juggle"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# The same juggling task with the two head joints removed from the action space, so a
# 29-DOF policy. The head regularly breaks on the real robots, and two of the three
# machines we can reach already have a head axis that does not follow its command. The
# joints stay in the observation because the camera rides on the head and its measured
# pose is needed for the camera-to-plate conversion, and their pose is randomized per
# episode so one policy covers robots that park the head differently. Widths become
# 103/121 (105/123 minus the two action channels) with 29 actions.
register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
  ),
  env_cfg=agibot_x2_tennis_juggle_env_cfg(head_actuated=False),
  play_env_cfg=agibot_x2_tennis_juggle_env_cfg(play=True, head_actuated=False),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_tennis_ball_juggle_29dof"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# The 29-DOF juggling task with the actor's ball observation widened from one
# delayed plate-frame position (3 numbers) to three timestamped samples (12
# numbers): the estimator's rate, latency, stalls, bias and noise are modelled
# inside the term instead of by a fixed delay buffer. Actor 103 -> 112, critic stays
# 121, actions stay 29. The observation contract and therefore the normalizer width
# changed, so this is a separate task and experiment.
register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF-Ball3"
  ),
  env_cfg=agibot_x2_tennis_juggle_env_cfg(head_actuated=False, ball_history=True),
  play_env_cfg=agibot_x2_tennis_juggle_env_cfg(
    play=True, head_actuated=False, ball_history=True
  ),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_tennis_ball_juggle_29dof_ball3"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# Opt-in pelvis-frame history. The distinct task and term names keep old Ball3
# checkpoints on their original plate-frame contract; widths stay 29 / 112 / 121.
register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
    "-Ball3-Pelvis"
  ),
  env_cfg=agibot_x2_tennis_juggle_env_cfg(
    head_actuated=False, ball_history=True, ball_history_frame="pelvis"
  ),
  play_env_cfg=agibot_x2_tennis_juggle_env_cfg(
    play=True, head_actuated=False, ball_history=True, ball_history_frame="pelvis"
  ),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# Opt-in nominal pilot. The pelvis task with the pushes, the physics/encoder
# randomization, the actor input corruption and the sampled resets removed, a
# centred 0.20 m drop start and a zero command. Widths stay 29 / 112 / 121. The
# target apex height stays on the existing MJLAB_JUGGLE_ENERGY_TARGET_HEIGHT_M
# override. See docs/design/x2_tennis_nominal_pilot.md.
register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
    "-Ball3-Pelvis-Nominal"
  ),
  env_cfg=agibot_x2_tennis_juggle_nominal_env_cfg(),
  play_env_cfg=agibot_x2_tennis_juggle_nominal_env_cfg(play=True),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# Rung S1 of the staged fine-tune. One input condition moves against the nominal pilot,
# so an S0 checkpoint resumes with its optimizer state. The ball history arrives at
# 15 Hz with 0-40 ms of uniform delay, so the estimator samples are asynchronous with
# the 50 Hz control clock; the screen needs no scheduler and no contract change.
register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
    "-Ball3-Pelvis-Nominal-S1-15Hz"
  ),
  env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(),
  play_env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(play=True),
  # The S0 experiment name is deliberate: the run resumes with --agent.load-run
  # pointing at an S0 run directory inside the same log root, so no checkpoint copy is
  # needed.
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# Rung S2 of the staged fine-tune: the S1 rate with the latency a real estimator
# chain has (80-100 ms). Resumes an S1 run's checkpoint, so it keeps S1's
# experiment name and the same 112 / 29 / 121 contract. See
# docs/source/x2_tennis_ball_juggle.rst.
register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
    "-Ball3-Pelvis-Nominal-S2-80-100ms"
  ),
  env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s2-80-100ms"),
  play_env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s2-80-100ms", play=True),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# Rung S3 of the staged fine-tune: 15 Hz with the latency drawn across the whole
# plausible band, 40-140 ms. Resumes an S2 run's checkpoint and keeps the same
# 112 / 29 / 121 contract; the delay varies within an episode, so the policy has to
# use the per-sample age rather than a single learned lead. See
# docs/source/x2_tennis_ball_juggle.rst.
register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
    "-Ball3-Pelvis-Nominal-S3-40-140ms"
  ),
  env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s3-40-140ms"),
  play_env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s3-40-140ms", play=True),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# Rung S3b, a sibling of S3: the same wide band shifted up to 80-140 ms, mean 110 ms,
# which brackets the measured chain's mean instead of a hypothetical better estimator.
# Same contract and the same S2 parent checkpoint as S3, so the pair isolates the
# band's centre.
register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
    "-Ball3-Pelvis-Nominal-S3b-80-140ms"
  ),
  env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s3b-80-140ms"),
  play_env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s3b-80-140ms", play=True),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# Rung S3c, the third sibling of S3: the whole band, 0-140 ms, mean 70 ms. Same
# contract and the same S2 parent checkpoint as S3 and S3b, so the three curves differ
# only in the band's centre and together say whether a wide band teaches the policy to
# use the per-sample age or only to sit on the band's mean.
register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
    "-Ball3-Pelvis-Nominal-S3c-0-140ms"
  ),
  env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s3c-0-140ms"),
  play_env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s3c-0-140ms", play=True),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# Rung S4: S3's condition plus the measured stream stall model. A stall is a longer
# gap before the next acquisition, so the newest sample is absent and the ball must be
# extrapolated from a growing age. Same contract and the same per-seed S3 parent
# checkpoint, so the only change is the stall.
register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
    "-Ball3-Pelvis-Nominal-S4-Stalls"
  ),
  env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s4-stalls"),
  play_env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s4-stalls", play=True),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# Rung S4b, a sibling of S4 with the stall measured from a recorded clip instead of assumed:
# 10 % of acquisitions stall for 0.1-0.2 s, because the ball leaves the camera field of view
# when it is raised. Same contract and the same per-seed S3 parent checkpoint as S4, so the
# pair isolates the stall parameters.
register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
    "-Ball3-Pelvis-Nominal-S4b-Stalls"
  ),
  env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s4b-stalls"),
  play_env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s4b-stalls", play=True),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# Rung S5: S4b's full condition plus per-acquisition position noise of +-(10, 10, 15) mm and
# no bias. The three history samples then carry independent error, which is what a real
# estimator gives, while the ages stay exact. Bias stays zero because it is not measurable
# here and a wrong-direction offset would train a compensation for a fault that may not exist.
register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
    "-Ball3-Pelvis-Nominal-S5-Noise"
  ),
  env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s5-noise"),
  play_env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s5-noise", play=True),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# Rung S6: S5's whole condition plus a release-position distribution - uniform height from
# rest on the strings to 0.40 m above them, and +-20 mm laterally. The target apex stays
# 0.40 m. This is the rung the plan asks for once the real hand release is characterised;
# until it is, the ranges are the operator's stated intent, not a measurement.
register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
    "-Ball3-Pelvis-Nominal-S6-Release"
  ),
  env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s6-release"),
  play_env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s6-release", play=True),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# Rung S7a: the operator holds the ball 0.40 m above the strings and releases it after a
# uniform 0-2 s wait, so the policy must align a level, quiet racket under it and be ready
# before the ball moves. S5's whole perception condition still applies, because a deferred
# release must be seen through the same delayed, stalled, noisy chain. The hold height is
# fixed at the energy target on purpose: the release energy then equals E* and any change
# is attributable to waiting rather than to a different drop energy.
register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
    "-Ball3-Pelvis-Nominal-S7a-DelayedRelease"
  ),
  env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s7a-delayed-release"),
  play_env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(
    stage="s7a-delayed-release", play=True
  ),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# Rung S7b: S7a's hold with the release-height range S6 had. The ball is held at a
# uniform 0.10-0.40 m and released after a 0-2 s wait, so the wait must generalise
# across the geometry an operator chooses. It resumes an S6 checkpoint, not S7a's, so
# S7b - S6 is the hold alone and S7b - S7a is the height range alone.
register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
    "-Ball3-Pelvis-Nominal-S7b-VariableHold"
  ),
  env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s7b-variable-hold"),
  play_env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(
    stage="s7b-variable-hold", play=True
  ),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# S7c: the same held ball and height range as S7b, held for a uniform 0-6 s
# instead of 0-2 s. Only the wait length moves, so S7c - S7b is the duration alone.
register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
    "-Ball3-Pelvis-Nominal-S7c-LongHold"
  ),
  env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s7c-long-hold"),
  play_env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s7c-long-hold", play=True),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# S8: S7c with the ball placed away from the plate by 5 cm per second of hold,
# along a random direction, and held there until release. The plate must travel to
# the ball; the displacement alone distinguishes it from S7c.
register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
    "-Ball3-Pelvis-Nominal-S8-OffsetHold"
  ),
  env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s8-offset-hold"),
  play_env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s8-offset-hold", play=True),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# S9a: a lost ball is priced instead of ending the episode, so the aftermath is
# trained; and the torso rotation rate is priced high enough to outvote the ball
# reward, because the motion that broke a connector was a rate, not a tilt.
register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
    "-Ball3-Pelvis-Nominal-S9a-DropTail"
  ),
  env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s9a-drop-tail"),
  play_env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s9a-drop-tail", play=True),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# S9b: S9a with the tracking family's full sim-to-real randomisation, definitions and
# ranges copied from the validated Multi-Motion task, including the push disturbance.
register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
    "-Ball3-Pelvis-Nominal-S9b-SimToReal"
  ),
  env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s9b-sim-to-real"),
  play_env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(
    stage="s9b-sim-to-real", play=True
  ),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# S9c: S9b plus a quadratic pose penalty beside the saturating exponential one, so a
# twisted posture is pulled back from any distance instead of costing nothing.
register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
    "-Ball3-Pelvis-Nominal-S9c-PoseQuad"
  ),
  env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s9c-pose-quad"),
  play_env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s9c-pose-quad", play=True),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# S9d: the ball starts below the plate and no ball-caused termination exists, so the
# episode runs the full 20 s and the robot learns to stay upright beside a dead ball.
register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
    "-Ball3-Pelvis-Nominal-S9d-DeadBall"
  ),
  env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s9d-dead-ball"),
  play_env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s9d-dead-ball", play=True),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# S10: S9b with the ball stream gated on what the camera can actually see, plus
# the mixed safety cases (ground, never-seen, induced loss), the 30 s budget and
# the no-ball-termination contract. It is no longer a one-delta gate rung; see
# docs/design/x2_tennis_s10_mixed_plan.md.
register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
    "-Ball3-Pelvis-Nominal-S10-VisibleBall"
  ),
  env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s10-visible-ball"),
  play_env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(
    stage="s10-visible-ball", play=True
  ),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# S11: S10 with a tilted start, so a lean becomes a state the policy can act from
# instead of one that only ever appears on the way to a fall.
register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
    "-Ball3-Pelvis-Nominal-S11-TiltedStart"
  ),
  env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(stage="s11-tilted-start"),
  play_env_cfg=agibot_x2_tennis_juggle_stage_env_cfg(
    stage="s11-tilted-start", play=True
  ),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name="agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# The target-height family: a second family, not a rung of the ladder above. The
# target apex is an actor and critic input here, so the observation widens to 113 /
# 122 and no ladder checkpoint resumes strictly. S1 is the shipped `s1-15hz`
# condition with the target fixed at 0.40 m, which reproduces that stage's reward,
# threshold and spawn exactly; S1a draws the target uniformly from 0.05 to 0.40 m and
# resumes S1, so it keeps S1's experiment name and the run is continued with
# --agent.load-run pointing at an S1 run directory inside the same log root.
# 29-DOF only, as the deployed ladder head. See
# docs/design/x2_tennis_target_height_plan.md.
register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
    "-Ball3-Pelvis-Nominal-S1-TargetHeight"
  ),
  env_cfg=agibot_x2_tennis_juggle_target_height_env_cfg(stage="s1"),
  play_env_cfg=agibot_x2_tennis_juggle_target_height_env_cfg(stage="s1", play=True),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name=(
      "agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal_s1_target_height"
    )
  ),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
    "-Ball3-Pelvis-Nominal-S1a-TargetHeight"
  ),
  env_cfg=agibot_x2_tennis_juggle_target_height_env_cfg(stage="s1a"),
  play_env_cfg=agibot_x2_tennis_juggle_target_height_env_cfg(stage="s1a", play=True),
  # S1's experiment name on purpose: S1a resumes an S1 checkpoint with its optimizer
  # state, so both live in one log root and no checkpoint copy is needed.
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name=(
      "agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal_s1_target_height"
    )
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# S2 carries S1a's target range and changes one condition: the perception band moves
# from 0-40 ms to 80-100 ms. Nothing about the target machinery moves, so the pair
# isolates the band's centre on top of a height-conditioned policy.
register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
    "-Ball3-Pelvis-Nominal-S2-TargetHeight"
  ),
  env_cfg=agibot_x2_tennis_juggle_target_height_env_cfg(stage="s2"),
  play_env_cfg=agibot_x2_tennis_juggle_target_height_env_cfg(stage="s2", play=True),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name=(
      "agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal_s1_target_height"
    )
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# Tennis-end recovery fine-tuning task. Builds on the no-state-estimation
# velocity config, replacing the twist command with the recovery-specialized
# command and adding the paired recovery reset event. Recovery environments
# restore a full reference state from the tennis endpoint pool and receive
# zero velocity commands; retention environments keep the original reset
# sequence and command distribution. The source velocity checkpoint loads
# strictly via the narrow TennisRecoveryOnPolicyRunner (same actor/critic
# dimensions, sensor frames, rewards, observations, actions, PD/DR).
# See docs/source/x2_tennis_recovery.rst.
register_mjlab_task(
  task_id=("Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Recovery"),
  env_cfg=agibot_x2_tennis_recovery_env_cfg(),
  play_env_cfg=agibot_x2_tennis_recovery_env_cfg(play=True),
  rl_cfg=agibot_x2_tennis_recovery_ppo_runner_cfg(),
  runner_cls=TennisRecoveryOnPolicyRunner,
)

# S3 keeps S2's target range and widens the perception band to 40-140 ms, redrawn per
# acquisition. In the shipped ladder this was the rung that made the policy flat across
# 0-140 ms; here it also has to hold the height command.
register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
    "-Ball3-Pelvis-Nominal-S3-TargetHeight"
  ),
  env_cfg=agibot_x2_tennis_juggle_target_height_env_cfg(stage="s3"),
  play_env_cfg=agibot_x2_tennis_juggle_target_height_env_cfg(stage="s3", play=True),
  rl_cfg=agibot_x2_velocity_ppo_runner_cfg(
    experiment_name=(
      "agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal_s1_target_height"
    )
  ),
  runner_cls=VelocityOnPolicyRunner,
)

##
# Measured right arm: the whole ladder again, on the measured servo.
#
# One change from each stage's shipped twin: the right shoulder pitch and roll damping
# come from the robot's step response instead of the vendor value (4.25 and 4.46 rad/s
# instead of 3.0, stiffness unchanged). See
# ``mjlab.asset_zoo.robots.agibot_x2.right_arm_measured`` for the measurement, for what
# the data does and does not support, and for why the actuator groups split per side.
#
# The stages are generated from ``STAGE_DELTAS`` rather than written out, so a stage
# cannot exist in one ladder and be missing from the other. Each family gets its own
# experiment name, so a run and its measured twin can never land in the same directory.
##

_RIGHT_ARM_MEASURED_TAG = "RightArmMeasured"
_RIGHT_ARM_STAGE_EXPERIMENT = (
  "agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal_right_arm_measured"
)
_RIGHT_ARM_TARGET_HEIGHT_EXPERIMENT = (
  "agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal_s1_target_height"
  "_right_arm_measured"
)
_STAGE_TASK_PREFIX = (
  "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
  "-Ball3-Pelvis-Nominal"
)


def _measured_stage_suffix(stage: str) -> str:
  """``s9c-pose-quad`` -> ``S9c-Pose-Quad``. Deterministic, no table to maintain."""
  return "-".join(
    token.capitalize() if token[:1].isalpha() else token for token in stage.split("-")
  )


RIGHT_ARM_MEASURED_TASK_IDS: list[str] = []
"""Every registered task that runs the measured right-arm servo.

Exported so the training-task tests can whitelist them by name instead of repeating
the generated task ids in a literal, which would silently drift.
"""


def _register_measured_stage(task_id: str, env_cfg_fn, experiment_name: str) -> None:
  RIGHT_ARM_MEASURED_TASK_IDS.append(task_id)
  register_mjlab_task(
    task_id=task_id,
    env_cfg=env_cfg_fn(),
    play_env_cfg=env_cfg_fn(play=True),
    rl_cfg=agibot_x2_velocity_ppo_runner_cfg(experiment_name=experiment_name),
    runner_cls=VelocityOnPolicyRunner,
  )


for _stage in STAGE_DELTAS:
  _register_measured_stage(
    task_id=f"{_STAGE_TASK_PREFIX}-{_RIGHT_ARM_MEASURED_TAG}-"
    f"{_measured_stage_suffix(_stage)}",
    env_cfg_fn=partial(
      agibot_x2_tennis_juggle_stage_env_cfg, stage=_stage, right_arm_measured=True
    ),
    experiment_name=_RIGHT_ARM_STAGE_EXPERIMENT,
  )

for _target_stage in STAGE_TARGET_HEIGHT_RANGE_M:
  _register_measured_stage(
    task_id=f"{_STAGE_TASK_PREFIX}-{_RIGHT_ARM_MEASURED_TAG}-"
    f"{_measured_stage_suffix(_target_stage)}-TargetHeight",
    env_cfg_fn=partial(
      agibot_x2_tennis_juggle_target_height_env_cfg,
      stage=_target_stage,
      right_arm_measured=True,
    ),
    experiment_name=_RIGHT_ARM_TARGET_HEIGHT_EXPERIMENT,
  )
