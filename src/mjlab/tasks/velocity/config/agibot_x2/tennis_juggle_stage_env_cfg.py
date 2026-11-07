"""AgiBot X2 staged juggling fine-tune configurations.

A **stage** is one input condition changed against the previous stage, nothing
else. The scene, the 1 ms contact physics, the vendor gains, the actuator limits,
the reward set, the terminations, the resets, the zero command, the locked head
and the observation layout stay the previous stage's by construction, so the
actor/critic/action widths do not move and a stage is trained by resuming the
previous stage's checkpoint with its optimizer state, at the same environment
count and seed. A stage exists so one condition can be screened on its own,
without a scheduler and without changing the contract. S10 is the documented
exception: it carries the visibility gate and the mixed safety contract together,
so a pure gate effect is not measurable from it alone.

The stages build on
:func:`~mjlab.tasks.velocity.config.agibot_x2.tennis_juggle_nominal_env_cfg.
agibot_x2_tennis_juggle_nominal_env_cfg`, the trained and evaluated nominal
juggling pilot (S0). That factory is called first and edited afterwards, so
every axis it pins stays pinned.

Stages that exist today:

``s1-15hz``
  S0 with the ball history arriving at 15 Hz with 0-40 ms of uniform transport
  delay, so the estimator samples are asynchronous with the 50 Hz control clock.
  Three samples are still stored, so the actor stays 112 wide.

See ``docs/source/x2_tennis_ball_juggle.rst``.
"""

from __future__ import annotations

import os
from copy import deepcopy

from mjlab.asset_zoo.robots.agibot_x2.tennis_constants import (
  BALL_CASE_INDUCED_SPEED_MPS,
  BALL_CASE_LOSS_TIME_RANGE_S,
  BALL_CASE_SHARES,
  BALL_DROP_GRACE_S,
  BALL_LOST_MIN_WORLD_Z_M,
  BALL_VISIBILITY_DROPOUT_PROB,
  BALL_VISIBLE_HALF_ANGLES_RAD,
  BALL_VISIBLE_MAX_RANGE_M,
  BALL_VISIBLE_MIN_RANGE_M,
  CAMERA_MOUNT_QUAT,
  CAMERA_MOUNT_XYZ,
  JUGGLE_ANKLE_JOINT_PATTERNS,
  JUGGLE_ANKLE_TILT_HALF_RANGE_RAD,
  JUGGLE_BALL_LOST_EVENT_WEIGHT,
  JUGGLE_BALL_LOST_PENALTY_WEIGHT,
  JUGGLE_DEAD_BALL_DROP_RANGE_M,
  JUGGLE_HOLD_DELAY_S,
  JUGGLE_HOLD_HEIGHT_M,
  JUGGLE_HOLD_LATERAL_RANGE_M,
  JUGGLE_HOLD_LONG_DELAY_S,
  JUGGLE_HOLD_OFFSET_SPEED_MPS,
  JUGGLE_POSE_QUAD_STD_RAD,
  JUGGLE_POSE_QUAD_WEIGHT,
  JUGGLE_S10_EPISODE_LENGTH_S,
  JUGGLE_TILT_HALF_RANGE_RAD,
  JUGGLE_TILT_JOINT_PATTERNS,
  JUGGLE_TORSO_ANG_VEL_WEIGHT,
  JUGGLE_WAIT_ALIGN_WEIGHT,
  JUGGLE_WAIT_LEVEL_POWER,
  JUGGLE_WAIT_PLATEAU_M,
  JUGGLE_WAIT_SIGMA_M,
  JUGGLE_WAIT_SPEED_SIGMA_MPS,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.tracking.config.agibot_x2.env_cfgs import (
  apply_x2_tracking_robot_randomization,
)
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.agibot_x2.tennis_ball_env_cfg import (
  BAT_BALL_CONTACT_SENSOR,
)
from mjlab.tasks.velocity.config.agibot_x2.tennis_juggle_nominal_env_cfg import (
  agibot_x2_tennis_juggle_nominal_env_cfg,
)
from mjlab.tasks.velocity.mdp.ball_cases import GROUND, INDUCED, LIVE, NEVER_SEEN

_HOLD_HEIGHT_ENV = "MJLAB_JUGGLE_HOLD_HEIGHT_M"
_HOLD_DELAY_ENV = "MJLAB_JUGGLE_HOLD_DELAY_S"


def _hold_delay(
  default: tuple[float, float] = JUGGLE_HOLD_DELAY_S,
) -> tuple[float, float]:
  """Hold duration range for the delayed-release rung, in seconds.

  Defaults to the stage's own value, so a run that does not set
  ``MJLAB_JUGGLE_HOLD_DELAY_S`` keeps the committed condition. The override exists
  because an acceptance cell must fix the wait instead of sampling it: a height by
  delay sweep sets one pair per cell, and a no-release control pins the wait past the
  horizon. It takes ``SECONDS`` or ``MIN,MAX``.

  This is an evaluation knob, not a training condition. A rung that changes the
  duration belongs in the stage table.
  """
  raw = os.environ.get(_HOLD_DELAY_ENV, "")
  if raw in ("", None):
    return default
  parts = [p.strip() for p in str(raw).split(",") if p.strip()]
  if len(parts) == 1:
    parts = [parts[0], parts[0]]
  if len(parts) != 2:
    raise ValueError(f"{_HOLD_DELAY_ENV} must be SECONDS or MIN,MAX, got {raw!r}")
  low, high = float(parts[0]), float(parts[1])
  if low < 0.0 or high < low:
    raise ValueError(
      f"{_HOLD_DELAY_ENV} must satisfy 0 <= min <= max, got ({low}, {high})"
    )
  return (low, high)


def _hold_height_override() -> float | None:
  """The requested fixed hold height, or ``None`` when nothing was requested."""
  raw = os.environ.get(_HOLD_HEIGHT_ENV, "")
  return float(raw) if raw not in ("", None) else None


def _hold_height(default: float = JUGGLE_HOLD_HEIGHT_M) -> float:
  """Hold height for the fixed-height rung, in metres.

  Defaults to :data:`JUGGLE_HOLD_HEIGHT_M`, so a run that does not set
  ``MJLAB_JUGGLE_HOLD_HEIGHT_M`` is byte-identical to the committed rung. The
  override exists because ``play`` has no ``--env.*`` flags: watching one policy
  released from several heights would otherwise need a source edit per height.
  Both the reset and the hold read this one value, so they cannot disagree and the
  start-height metric stays true.

  This is a viewing and ablation knob, not a training condition. A rung that
  randomises the height belongs in the stage table, not in an environment variable.
  """
  value = _hold_height_override()
  return default if value is None else value


def _hold_height_range(default: tuple[float, float]) -> tuple[float, float]:
  """The reset's height range, collapsed to one value when a height is requested.

  The randomising rungs sample a range per environment. An acceptance cell has to
  fix the height instead of sampling it, and it must fix the same value in the
  reset and in the pin, or ``ball_start_height_m`` would report a height the ball
  was never held at.
  """
  value = _hold_height_override()
  return default if value is None else (value, value)


def _pinned_hold_height() -> float | None:
  """The pinned height, or ``None`` to follow the reset's sample.

  ``None`` is the training condition: one sampler, one authority, so the pin and
  the reported start height cannot disagree.
  """
  return _hold_height_override()


STAGE_DELTAS: dict[str, dict[str, object]] = {
  "s1-15hz": {
    "stream_rate_hz": (15.0, 15.0),
    "stream_latency_s": (0.0, 0.040),
  },
  # S1 at a latency the deployed chain actually has. 80-100 ms is the improved
  # estimator target, not the measured chain: the older chain cost 26.1 ms of
  # adapter, 65 ms of stereo inference and 2.2 ms of estimator before transport,
  # and was measured end to end at 110-141 ms. The rate stays at S1's because a
  # faster estimator is a separate rung.
  "s2-80-100ms": {
    "stream_rate_hz": (15.0, 15.0),
    "stream_latency_s": (0.080, 0.100),
  },
  # S3 spans the whole plausible band on purpose: 40-140 ms is 2 to 7 control
  # steps of lag, drawn per acquisition, so the delay changes inside an episode.
  # A fixed delay can be compensated once; a varying one must be read from the
  # per-sample age that the observation already carries. If this rung works, one
  # policy covers every estimator between the best and the measured chain, which is
  # what the timestamps were for.
  "s3-40-140ms": {
    "stream_rate_hz": (15.0, 15.0),
    "stream_latency_s": (0.040, 0.140),
  },
  # S3b is a sibling of S3, not its successor: the same wide band shifted up, 80-140 ms
  # (mean 110 ms) against S3's 40-140 ms (mean 90 ms). Both resume S2's checkpoint and
  # run the same budget, so the pair isolates the effect of the band's CENTRE. The
  # deployed chain's mean is about 125 ms, so S3b brackets reality while S3 brackets
  # what a good estimator could deliver.
  "s3b-80-140ms": {
    "stream_rate_hz": (15.0, 15.0),
    "stream_latency_s": (0.080, 0.140),
  },
  # S3c is the third sibling: the whole band, 0-140 ms, mean 70 ms. Together with S3
  # (mean 90) and S3b (mean 110) it spans the band centres while keeping the width,
  # so the three curves say whether a wide band makes the policy track the age or
  # makes it settle on the band's mean. It also includes zero delay, where the S2
  # policy collapses to streaks of five flights, so it is the widest stress of the
  # three.
  "s3c-0-140ms": {
    "stream_rate_hz": (15.0, 15.0),
    "stream_latency_s": (0.0, 0.140),
  },
  # S4 changes the condition, not the delay: S3's band plus the measured stall model.
  # A stall is a longer gap before the next acquisition, so its only visible effect is
  # the growing age of the held sample: 10-30 control steps on top of 2-7 steps of
  # transport lag, up to about 0.74 s. S3 proved the policy reads the per-sample age,
  # so S4 asks the next question, whether it stays correct while the newest sample is
  # absent long enough that the ball must be extrapolated rather than observed.
  "s4-stalls": {
    "stream_rate_hz": (15.0, 15.0),
    "stream_latency_s": (0.040, 0.140),
    "stall_prob": 0.02,
    "stall_range_s": (0.1, 0.6),
  },
  # S4b is a sibling of S4 with the stall measured from a recorded hardware clip rather
  # than assumed: the ball leaves the camera field of view when it is raised high, for
  # about 0.1 s, on about 10 % of frames. The gap is therefore 5-10 control steps and it
  # recurs every ~0.6 s, against S4's 5-30 steps every ~3 s. Worst age seen by the actor
  # drops from 0.14 + 0.6 = 0.74 s to 0.14 + 0.2 = 0.34 s, while the fraction of time the
  # newest sample is stale rises from about 10 % to about 20 %. S4 trained the same policy
  # against gaps the camera does not produce, and it degraded to 0.38 s flights with
  # 5 drops per 4096 environments; S4b tests the gap the camera does produce.
  "s4b-stalls": {
    "stream_rate_hz": (15.0, 15.0),
    "stream_latency_s": (0.040, 0.140),
    "stall_prob": 0.10,
    "stall_range_s": (0.1, 0.2),
  },
  # S5 adds measurement error on top of S4b's whole condition: the rate, the latency band
  # and the stall model all repeat, because a stage carries the complete condition and not
  # only its own increment. The noise is per axis and per acquisition, drawn uniformly in
  # +-(10, 10, 15) mm - half a ball radius vertically - so the three history samples carry
  # INDEPENDENT error, which is what a real estimator gives. The ages stay exact; only the
  # positions are wrong.
  #
  # The bias is deliberately zero. A per-episode offset is not measurable with the current
  # rig, and a bias drawn in the wrong direction would teach the actor to compensate a
  # calibration error that does not exist. Noise is a magnitude we can bound from the
  # ball-centre fit; a bias is a direction we cannot.
  "s5-noise": {
    "stream_rate_hz": (15.0, 15.0),
    "stream_latency_s": (0.040, 0.140),
    "stall_prob": 0.10,
    "stall_range_s": (0.1, 0.2),
    "position_noise_m": (0.010, 0.010, 0.015),
    "bias_half_width_m": 0.0,
  },
  # S6 stops varying what the estimator reports and varies what the operator does: the
  # release. The ball leaves the hand anywhere from rest on the strings to 0.40 m above
  # them, uniformly, and anywhere within +-20 mm of the plate centre - the lateral spread
  # the plan asks for once a measured release exists. The target apex stays 0.40 m, so a
  # release at the target height must be caught and re-bounced rather than lifted, and a
  # release at zero has no fall energy at all. Everything S5 added still applies: rate,
  # latency band, stall model and noise.
  #
  # A ``reset`` section is the one key a stage may use to move a reset term; the factory
  # applies it to ``events["reset_ball"]`` and the one-delta test requires every other
  # event term to equal S0's.
  "s6-release": {
    "stream_rate_hz": (15.0, 15.0),
    "stream_latency_s": (0.040, 0.140),
    "stall_prob": 0.10,
    "stall_range_s": (0.1, 0.2),
    "position_noise_m": (0.010, 0.010, 0.015),
    "bias_half_width_m": 0.0,
    "reset": {
      "drop_height_range": (0.0, 0.4),
      "drop_lateral_range_m": 0.02,
    },
  },
  # S7a is the first stage that changes the *episode structure* rather than a
  # perception condition: the ball is held still above the plate and released after a
  # random wait, so the policy must put a level racket under it and be ready before the
  # ball moves. S5's whole perception condition still applies, because a deferred
  # release must be seen through the same delayed, stalled, noisy chain.
  #
  # Three additions come with it, and each is there to stop a measured loophole rather
  # than to add skill:
  #   * the energy reward is gated to the post-release phase, because a held ball is
  #     world-fixed with zero velocity, so `E` is a per-episode constant the policy
  #     cannot influence -- at a 0.40 m hold it is exactly `E*` and would pay a full
  #     1.0 on every waiting step for doing nothing;
  #   * `racket_under_ball` pays for a level, quiet racket under the ball while held,
  #     because `ball_centred` alone is plate-frame horizontal only and a tilted plate
  #     can satisfy it;
  #   * `held_ball_contact` ends an episode in which the racket touches the held ball,
  #     so a wait cannot become an early strike or a preload against the pin.
  # `ball_no_bounce` is paused while held: a 2 s wait would otherwise consume 80 % of
  # its 2.5 s deadline before the ball is released.
  #
  # The hold height is fixed at 0.40 m, which is the energy target, so the release
  # energy equals `E*` and any change is attributable to waiting. Randomising the hold
  # height changes the drop energy as well and is a separate rung.
  "s7a-delayed-release": {
    "stream_rate_hz": (15.0, 15.0),
    "stream_latency_s": (0.040, 0.140),
    "stall_prob": 0.10,
    "stall_range_s": (0.1, 0.2),
    "position_noise_m": (0.010, 0.010, 0.015),
    "bias_half_width_m": 0.0,
    "reset": {
      # S0 already releases every episode as a drop start (plate_start_fraction is 0),
      # so only the height, the extra fall time and the lateral spread move.
      "drop_height_range": (_hold_height(), _hold_height()),
      "drop_lateral_range_m": JUGGLE_HOLD_LATERAL_RANGE_M,
    },
    "hold": {
      "hold_height_m": _hold_height(),
      "hold_delay_s": _hold_delay(),
      "lateral_range_m": JUGGLE_HOLD_LATERAL_RANGE_M,
    },
  },
  # S7b restores the randomisation S7a removed. S6 released the ball at rest from a
  # uniform 0.0-0.40 m; S7a replaced that with a fixed 0.40 m hold, so its difference
  # from S6 mixed the hold with the loss of the height range. S7b keeps S7a's hold and
  # carries the height range, which makes S7b - S6 the hold alone and S7b - S7a the
  # height range alone. It resumes an S6 checkpoint, not S7a's, for the same reason.
  #
  # The floor is 0.10 m, not 0.0: a hold at 0.0 means the ball rests on the strings,
  # which is a plate start rather than something an operator can hold without touching
  # them, and `held_ball_contact` would fire at reset. The 0-0.10 band is expressible
  # as a drop, not as a hold, and that limit is stated rather than hidden.
  #
  # The height is sampled ONCE, by `reset_ball`: `hold_ball` is given
  # `hold_height_m: None` and follows that sample, so the pin and
  # `ball_start_height_m` cannot disagree.
  "s7b-variable-hold": {
    "stream_rate_hz": (15.0, 15.0),
    "stream_latency_s": (0.040, 0.140),
    "stall_prob": 0.10,
    "stall_range_s": (0.1, 0.2),
    "position_noise_m": (0.010, 0.010, 0.015),
    "bias_half_width_m": 0.0,
    "reset": {
      "drop_height_range": _hold_height_range((0.10, 0.40)),
      "drop_lateral_range_m": JUGGLE_HOLD_LATERAL_RANGE_M,
    },
    "hold": {
      "hold_height_m": _pinned_hold_height(),
      "hold_delay_s": _hold_delay(),
      "lateral_range_m": JUGGLE_HOLD_LATERAL_RANGE_M,
    },
  },
  # S7c - S7b isolates the wait length: every other key is S7b's, so the only
  # difference is how long the ball is held before release. 6 s is bounded by the
  # reward accounting, not by the episode: an episode is 20 s and reward is
  # normalised by that maximum, so a mean 3 s wait still leaves 14 s of juggling.
  "s7c-long-hold": {
    "stream_rate_hz": (15.0, 15.0),
    "stream_latency_s": (0.040, 0.140),
    "stall_prob": 0.10,
    "stall_range_s": (0.1, 0.2),
    "position_noise_m": (0.010, 0.010, 0.015),
    "bias_half_width_m": 0.0,
    "reset": {
      "drop_height_range": _hold_height_range((0.10, 0.40)),
      "drop_lateral_range_m": JUGGLE_HOLD_LATERAL_RANGE_M,
    },
    "hold": {
      "hold_height_m": _pinned_hold_height(),
      "hold_delay_s": _hold_delay(JUGGLE_HOLD_LONG_DELAY_S),
      "lateral_range_m": JUGGLE_HOLD_LATERAL_RANGE_M,
    },
  },
  # S8 - S7c isolates the placement: the ball is put down away from the plate by
  # 0.05 m per second of hold, along a random direction, and stays there until the
  # release. The plate has to travel to it. Everything else is S7c's, so a
  # difference in behaviour is the displacement alone.
  "s8-offset-hold": {
    "stream_rate_hz": (15.0, 15.0),
    "stream_latency_s": (0.040, 0.140),
    "stall_prob": 0.10,
    "stall_range_s": (0.1, 0.2),
    "position_noise_m": (0.010, 0.010, 0.015),
    "bias_half_width_m": 0.0,
    "reset": {
      "drop_height_range": _hold_height_range((0.10, 0.40)),
      "drop_lateral_range_m": JUGGLE_HOLD_LATERAL_RANGE_M,
    },
    "hold": {
      "hold_height_m": _pinned_hold_height(),
      "hold_delay_s": _hold_delay(JUGGLE_HOLD_LONG_DELAY_S),
      "lateral_range_m": JUGGLE_HOLD_LATERAL_RANGE_M,
      "offset_speed_mps": JUGGLE_HOLD_OFFSET_SPEED_MPS,
    },
  },
  # S9a - S8 isolates the safety behaviour. A lost ball no longer ends the episode:
  # the tail is charged while the ball is gone and ``ball_no_bounce`` ends it 2.5 s
  # after the last strike. The torso rotation rate is priced high enough to outvote
  # the ball reward, because the motion that broke a connector was a rate, not a
  # tilt, and the shipped weight was 1/550th of the tilt reward.
  "s9a-drop-tail": {
    "stream_rate_hz": (15.0, 15.0),
    "stream_latency_s": (0.040, 0.140),
    "stall_prob": 0.10,
    "stall_range_s": (0.1, 0.2),
    "position_noise_m": (0.010, 0.010, 0.015),
    "bias_half_width_m": 0.0,
    "reset": {
      "drop_height_range": _hold_height_range((0.10, 0.40)),
      "drop_lateral_range_m": JUGGLE_HOLD_LATERAL_RANGE_M,
    },
    "hold": {
      "hold_height_m": _pinned_hold_height(),
      "hold_delay_s": _hold_delay(JUGGLE_HOLD_LONG_DELAY_S),
      "lateral_range_m": JUGGLE_HOLD_LATERAL_RANGE_M,
      "offset_speed_mps": JUGGLE_HOLD_OFFSET_SPEED_MPS,
    },
    "drop_tail": True,
  },
  # S9b - S9a isolates the randomisation: the tracking family's full sim-to-real set,
  # definitions and ranges copied from the validated Multi-Motion task, including the
  # push disturbance. The stage's own condition (stream, hold, offset, drop tail) is
  # unchanged, so a difference in behaviour is the randomisation alone.
  "s9b-sim-to-real": {
    "stream_rate_hz": (15.0, 15.0),
    "stream_latency_s": (0.040, 0.140),
    "stall_prob": 0.10,
    "stall_range_s": (0.1, 0.2),
    "position_noise_m": (0.010, 0.010, 0.015),
    "bias_half_width_m": 0.0,
    "reset": {
      "drop_height_range": _hold_height_range((0.10, 0.40)),
      "drop_lateral_range_m": JUGGLE_HOLD_LATERAL_RANGE_M,
    },
    "hold": {
      "hold_height_m": _pinned_hold_height(),
      "hold_delay_s": _hold_delay(JUGGLE_HOLD_LONG_DELAY_S),
      "lateral_range_m": JUGGLE_HOLD_LATERAL_RANGE_M,
      "offset_speed_mps": JUGGLE_HOLD_OFFSET_SPEED_MPS,
    },
    "drop_tail": True,
    "sim_to_real": True,
  },
  # S9c - S9b isolates the posture: a quadratic penalty beside the saturating
  # exponential pose term, same reference and same 0.05 rad tolerance, weighted so a
  # 30 degree twist costs about a fifth of the whole reward intake.
  "s9c-pose-quad": {
    "stream_rate_hz": (15.0, 15.0),
    "stream_latency_s": (0.040, 0.140),
    "stall_prob": 0.10,
    "stall_range_s": (0.1, 0.2),
    "position_noise_m": (0.010, 0.010, 0.015),
    "bias_half_width_m": 0.0,
    "reset": {
      "drop_height_range": _hold_height_range((0.10, 0.40)),
      "drop_lateral_range_m": JUGGLE_HOLD_LATERAL_RANGE_M,
    },
    "hold": {
      "hold_height_m": _pinned_hold_height(),
      "hold_delay_s": _hold_delay(JUGGLE_HOLD_LONG_DELAY_S),
      "lateral_range_m": JUGGLE_HOLD_LATERAL_RANGE_M,
      "offset_speed_mps": JUGGLE_HOLD_OFFSET_SPEED_MPS,
    },
    "drop_tail": True,
    "sim_to_real": True,
    "pose_quad": True,
  },
  # S9d - the ball starts below the plate, where the racket cannot reach it, and no
  # ball-caused termination exists. The episode runs the full 20 s, so the robot learns
  # to stay upright and square while the ball lies on the floor instead of lunging at a
  # ball it cannot rescue. This is the posture rung the S9b2 screenshot asked for: it
  # must be followed by a re-juggle rung, because a dead ball gives no juggle gradient.
  "s9d-dead-ball": {
    "stream_rate_hz": (15.0, 15.0),
    "stream_latency_s": (0.040, 0.140),
    "stall_prob": 0.10,
    "stall_range_s": (0.1, 0.2),
    "position_noise_m": (0.010, 0.010, 0.015),
    "bias_half_width_m": 0.0,
    "reset": {
      # Negative: the ball starts below the plate, where the racket cannot reach it.
      "drop_height_range": JUGGLE_DEAD_BALL_DROP_RANGE_M,
    },
    "dead_ball": True,
    "sim_to_real": True,
    "pose_quad": True,
  },
  # S10 - S9c with the ball stream gated on what the camera can see, plus the mixed
  # safety cases. The gate is the measurement condition; the mixture, the loss-charge
  # and termination changes and the 30 s budget are the safety condition. That makes S10
  # S9c plus two declared sections, not a one-delta rung, so a pure gate effect is not
  # measurable from S10 alone.
  #
  # It descends from S9c rather than S9b because the posture S9c fixed came from a
  # reward term: dropping ``pose_quadratic`` while resuming an S9c checkpoint would hand
  # back the twist that term removed, so ``pose_quad`` is declared here too.
  "s10-visible-ball": {
    "stream_rate_hz": (15.0, 15.0),
    "stream_latency_s": (0.040, 0.140),
    "stall_prob": 0.10,
    "stall_range_s": (0.1, 0.2),
    "position_noise_m": (0.010, 0.010, 0.015),
    "bias_half_width_m": 0.0,
    "reset": {
      "drop_height_range": _hold_height_range((0.10, 0.40)),
      "drop_lateral_range_m": JUGGLE_HOLD_LATERAL_RANGE_M,
      "case_shares": BALL_CASE_SHARES,
      "loss_time_range_s": BALL_CASE_LOSS_TIME_RANGE_S,
    },
    "hold": {
      "hold_height_m": _pinned_hold_height(),
      "hold_delay_s": _hold_delay(JUGGLE_HOLD_LONG_DELAY_S),
      "lateral_range_m": JUGGLE_HOLD_LATERAL_RANGE_M,
      "offset_speed_mps": JUGGLE_HOLD_OFFSET_SPEED_MPS,
    },
    "drop_tail": True,
    "sim_to_real": True,
    "pose_quad": True,
    "visibility_gate": True,
    "safety_mix": True,
  },
  "s11-tilted-start": {
    "stream_rate_hz": (15.0, 15.0),
    "stream_latency_s": (0.040, 0.140),
    "stall_prob": 0.10,
    "stall_range_s": (0.1, 0.2),
    "position_noise_m": (0.010, 0.010, 0.015),
    "bias_half_width_m": 0.0,
    "reset": {
      "drop_height_range": _hold_height_range((0.10, 0.40)),
      "drop_lateral_range_m": JUGGLE_HOLD_LATERAL_RANGE_M,
      "case_shares": BALL_CASE_SHARES,
      "loss_time_range_s": BALL_CASE_LOSS_TIME_RANGE_S,
    },
    "hold": {
      "hold_height_m": _pinned_hold_height(),
      "hold_delay_s": _hold_delay(JUGGLE_HOLD_LONG_DELAY_S),
      "lateral_range_m": JUGGLE_HOLD_LATERAL_RANGE_M,
      "offset_speed_mps": JUGGLE_HOLD_OFFSET_SPEED_MPS,
    },
    "drop_tail": True,
    "sim_to_real": True,
    "pose_quad": True,
    "visibility_gate": True,
    "safety_mix": True,
    "tilted_start": True,
  },
}
"""The one condition each stage changes, as ``ball_hist_pelvis`` term parameters.

Each entry holds exactly the keys a stage moves, and every other parameter of the
term is the nominal pilot's. A ``reset`` sub-entry moves the ball reset instead: it is
applied to ``events["reset_ball"]``. Nothing else about the reset may change, and no
other event term may be touched. The values are ranges: the term draws one value per
environment from the range on every acquisition. A stage carries the whole
condition, not only its own increment, because the factory applies the entry on top
of S0: a later stage repeats the earlier stage's keys.

The term quantises both to whole control steps, which are 20 ms here. A requested
15 Hz therefore runs one sample every 3 steps, i.e. 16.67 Hz, and a 0-40 ms delay
becomes 0, 20 or 40 ms. 15 Hz is not representable at this control rate: the
neighbours are 16.67 Hz and 12.5 Hz, and S1 takes the faster one, so a pass at S1
is slightly optimistic rather than slightly harsh.
"""


def agibot_x2_tennis_juggle_stage_env_cfg(
  stage: str = "s1-15hz", play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Create a staged fine-tune's AgiBot X2 juggling environment config.

  Args:
    stage: The stage to build. Must be a key of :data:`STAGE_DELTAS`.
    play: If True, applies the nominal pilot's play-mode overrides first.

  Returns:
    A ``ManagerBasedRlEnvCfg`` for the 29-DOF pelvis-history juggling task, equal
    to the nominal pilot's config except for the stage's own term parameters.

  Raises:
    ValueError: If ``stage`` is not a known stage.
  """
  if stage not in STAGE_DELTAS:
    raise ValueError(
      f"unknown stage {stage!r}, known stages are {sorted(STAGE_DELTAS)}"
    )

  delta = dict(STAGE_DELTAS[stage])
  reset_delta = delta.pop("reset", None)
  hold_delta = delta.pop("hold", None)
  # Structural sections are popped before the update below, or they would arrive as
  # unknown parameters of the ball-history observation term.
  drop_tail = bool(delta.pop("drop_tail", False))
  sim_to_real = bool(delta.pop("sim_to_real", False))
  pose_quad = bool(delta.pop("pose_quad", False))
  dead_ball = bool(delta.pop("dead_ball", False))
  visibility_gate = bool(delta.pop("visibility_gate", False))
  tilted_start = bool(delta.pop("tilted_start", False))
  safety_mix = bool(delta.pop("safety_mix", False))
  cfg = agibot_x2_tennis_juggle_nominal_env_cfg(play=play)
  cfg.observations["actor"].terms["ball_hist_pelvis"].params.update(delta)
  if reset_delta:
    cfg.events["reset_ball"].params.update(reset_delta)
  if hold_delta:
    _apply_delayed_release(cfg, hold_delta)
  if drop_tail:
    _apply_drop_tail(cfg, cfg.sim.mujoco.timestep * cfg.decimation)
  # Play mode is a viewing and evaluation view: it keeps the nominal robot, exactly
  # as it already drops the push and the corruption. Randomising it would make the
  # same checkpoint behave differently every launch.
  if sim_to_real and not play:
    _apply_sim_to_real(cfg)
  if pose_quad:
    _apply_pose_quad(cfg)
  if dead_ball:
    _apply_dead_ball(cfg)
  if visibility_gate:
    _apply_visibility_gate(cfg)
  if safety_mix:
    _apply_safety_mix(cfg, play)
  if tilted_start:
    _apply_tilted_start(cfg)
  return cfg


FULL_DR_EVENT_NAMES = (
  # The five the velocity base task carries and the nominal pilot removes.
  "base_com",
  "encoder_bias",
  "foot_friction",
  "push_robot",
  "randomize_pd_gains",
  # The tracking family's physics axes, in their install order.
  "randomize_inertia",
  "randomize_armature",
  "randomize_effort_limits",
  "randomize_joint_friction",
  "randomize_joint_damping",
  "randomize_foot_size",
)
"""Every domain-randomisation term the tracking family runs, by name.

Definitions come from ``make_tracking_env_cfg()``, never from this module, so there is
one definition of every range. ``randomize_effort_limits_wrist`` is absent on purpose:
``apply_x2_tracking_robot_randomization`` adds it after ``randomize_effort_limits``
because both write the same actuator fields and only the last writer takes effect.

The tracking family's observation-delay axis is not here: it changes the actor's input
timing contract, which the ball-history term already owns.
"""


def _apply_sim_to_real(cfg: ManagerBasedRlEnvCfg) -> None:
  """Install the tracking family's full domain randomisation on this config.

  The nominal pilot removes twelve randomisation events so the juggle can be learned
  on an exactly modelled robot. This puts them back, with the definitions and ranges
  the validated ``Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion`` task runs -- that is
  the set with dynamic tennis motions behind it, so nothing here is invented.

  ``push_robot`` is part of the set: it is the disturbance axis, applied as a root
  velocity every 1-3 s, and it is what makes the juggle survive a hand on the ball
  rather than only an exactly modelled one.
  """
  source = make_tracking_env_cfg().events
  for name in FULL_DR_EVENT_NAMES:
    term = source.get(name)
    if term is None:
      # ``MJLAB_DR_AXES`` can switch an axis off; a missing axis is not an error.
      continue
    cfg.events[name] = deepcopy(term)
  apply_x2_tracking_robot_randomization(cfg)


def _apply_dead_ball(cfg: ManagerBasedRlEnvCfg) -> None:
  """Remove every episode end and every reward that involves the ball (S9d).

  The dead-ball start places the ball below the plate, where the racket cannot reach
  it, so the ball is lost from the first step. In that situation the three
  ball-caused terminations are harmful, not merely useless:

  * ``ball_no_bounce`` and ``ball_not_lifted`` end the episode a second or two in, so
    the robot never spends time standing while a ball lies on the floor -- which is
    the behaviour this rung exists to teach;
  * ``held_ball_contact`` has no held phase to protect.

  What remains is ``fell_over`` (70 degree tilt) and ``time_out`` (the full 20 s), so
  the only way an episode ends early is by falling. That is the signal we want.

  The ball rewards go with them. ``ball_lost_penalty`` in particular must not stay: it
  charges per step while the ball is lost, and a fall would then *end* the charge, so
  the policy would learn that falling is cheaper than standing upright with a dead
  ball. ``ball_energy``, ``ball_centred``, ``ball_in_fov`` and ``racket_under_ball``
  are all identically zero here and only cost evaluation.

  Policy, critic, action widths and the observation layout do not change, so this rung
  resumes an S9c checkpoint with its optimizer state.
  """
  for name in (
    "ball_dropped",
    "ball_no_bounce",
    "ball_not_lifted",
    "held_ball_contact",
  ):
    cfg.terminations.pop(name, None)
  for name in (
    "ball_energy",
    "ball_centred",
    "ball_in_fov",
    "racket_under_ball",
    "ball_lost_penalty",
  ):
    cfg.rewards.pop(name, None)


def _apply_tilted_start(cfg: ManagerBasedRlEnvCfg) -> None:
  """Start the robot already leaning, so a tilt can be acted from (S11).

  Every earlier rung resets the robot upright in the carry pose with zero velocity, so a
  tilt could only be reached transiently by a push and became terminal at 70 degrees.
  The policy therefore never had a tilted state to recover from: on hardware it stands
  calmly but cannot come back once it leans.

  Two reset terms are inserted **after** the carry-pose reset and **before** the ball
  reset, so the ball is placed relative to the resulting pose:

  * waist, hip and knee joints get a uniform offset of plus or minus 15 degrees;
  * both ankles get a uniform offset of plus or minus 30 degrees.

  :func:`reset_joints_by_offset` clamps each sampled value to the joint's soft limits, so
  a joint with less than 15 degrees of travel in one direction saturates there. The
  realised distribution is therefore truncated by the machine, not by the constant.

  Nothing else changes: rewards, terminations, metrics, the mixture, the gate, the
  observation layout and the 30 s budget are S10's.
  """

  def tilt_term(patterns, half_range):
    return EventTermCfg(
      mode="reset",
      func=mdp.reset_joints_by_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=patterns),
        "position_range": (-half_range, half_range),
        "velocity_range": (0.0, 0.0),
      },
    )

  body = tilt_term(JUGGLE_TILT_JOINT_PATTERNS, JUGGLE_TILT_HALF_RANGE_RAD)
  ankle = tilt_term(JUGGLE_ANKLE_JOINT_PATTERNS, JUGGLE_ANKLE_TILT_HALF_RANGE_RAD)

  events: dict = {}
  for name, term in cfg.events.items():
    events[name] = term
    if name == "reset_robot_juggle_joints":
      events["reset_tilt_body"] = body
      events["reset_tilt_ankle"] = ankle
  if "reset_tilt_body" not in events:
    raise ValueError("tilted start needs a reset_robot_juggle_joints term to follow")
  cfg.events = events


def _apply_visibility_gate(cfg: ManagerBasedRlEnvCfg) -> None:
  """Make the ball stream refuse frames the camera could not have measured (S10).

  The estimator publishes nothing for a frame in which the ball was not detected, and
  the simulator had no equivalent: ``ball_history_samples`` wrote the true ball position
  at every step, so the actor could rely on a measurement that the robot cannot make.
  This stage turns the gate on. It changes only what the actor measures; the reward
  manager and the metrics keep reading the true state.

  Geometry is the delivered image, not the raw sensor: a 480x288 rectified pinhole with
  half-angles 65.7 / 67.2 / 53.9 / 54.1 degrees and a 0.1-1.0 m straight-line range gate
  about the left camera centre. The mount is the training value, and the measurement
  stays unbiased: the estimator's measured +8.3 % range overestimate is not modelled.

  The dropout probability stays at its default of zero, so this stage is pure geometry.
  A measured 4.8-5.2 % of frames are rejected as ``ambiguous`` inside the region; that is
  recorded in the plan and deliberately left out here, so the stage moves one thing.
  """
  cfg.observations["actor"].terms["ball_hist_pelvis"].params.update(
    visibility_head_cfg=SceneEntityCfg("robot", body_names=("head_pitch_link",)),
    visibility_mount_xyz=CAMERA_MOUNT_XYZ,
    visibility_mount_quat=CAMERA_MOUNT_QUAT,
    visibility_half_angles=BALL_VISIBLE_HALF_ANGLES_RAD,
    visibility_min_range_m=BALL_VISIBLE_MIN_RANGE_M,
    visibility_max_range_m=BALL_VISIBLE_MAX_RANGE_M,
    visibility_dropout_prob=BALL_VISIBILITY_DROPOUT_PROB,
  )


def _apply_pose_quad(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add a quadratic pose penalty beside the saturating exponential one (S9c).

  ``pose`` is ``exp(-mean((q-q*)^2/std^2))`` with ``std = 0.05`` rad. At the
  deviations a twisted juggle posture reaches -- 30 degrees on ten joints -- the
  exponent is about -32, so both the value and its gradient are numerically zero and
  the policy is neither charged nor pulled back. ``pose_quadratic`` is the negative
  log of the same quantity: same reference, same tolerance, gradient at any distance.

  Weight -0.04 is sized against the measured reward budget. Positive intake is
  +0.351 per second, so a term worth 1.28 per step at that posture is about 18 % of
  everything the policy earns: enough to pull the torso back, not enough to stop the
  juggle. The existing exponential term stays, so nothing is removed.
  """
  cfg.rewards["pose_quadratic"] = RewardTermCfg(
    func=mdp.pose_quadratic,
    weight=JUGGLE_POSE_QUAD_WEIGHT,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "std": JUGGLE_POSE_QUAD_STD_RAD,
    },
  )


def _apply_drop_tail(cfg: ManagerBasedRlEnvCfg, step_dt: float) -> None:
  """Replace the drop termination with a lost-ball cost (S9a).

  ``ball_lost`` ends the episode the instant the ball falls. That trains the robot up
  to the loss and never afterwards, so the state that damaged hardware - ball gone,
  nothing left to juggle - had no learned behaviour. Removing the termination and
  pricing the state keeps the episode alive for the aftermath, which ``ball_no_bounce``
  bounds at 2.5 s after the last strike.

  Two changes, both ball-level or whole-body: no new sensor, no per-joint term.

  * ``ball_lost_penalty`` charges while the ball is lost, at the weight the measured
    reward budget supports.
  * ``body_ang_vel`` on the torso goes from -0.05 to -0.5. At the shipped weight the
    term was worth 1/550th of ``upright``, so no rotation rate was ever discouraged.
  """
  cfg.terminations.pop("ball_dropped", None)
  cfg.rewards["ball_lost_penalty"] = RewardTermCfg(
    func=mdp.ball_lost_penalty,
    weight=JUGGLE_BALL_LOST_PENALTY_WEIGHT,
    params={
      "ball_cfg": SceneEntityCfg("ball"),
      "min_world_z": BALL_LOST_MIN_WORLD_Z_M,
      "grace_steps": round(BALL_DROP_GRACE_S / step_dt),
    },
  )
  cfg.rewards["body_ang_vel"].weight = JUGGLE_TORSO_ANG_VEL_WEIGHT
  cfg.metrics["ball_lost_events"] = MetricsTermCfg(
    func=mdp.lost_ball_state,
    params={"reward_term": "ball_lost_penalty", "field": "lost_events"},
    reduce="max",
  )


def _apply_safety_mix(cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
  """Add the mixed safety cases and the no-ball-termination contract (S10).

  Four pieces, all reachable from the stage flag:

  * the case draw itself is declared in the stage's ``reset`` section, so
    ``reset_ball`` receives ``case_shares`` and ``loss_time_range_s`` and stores
    the per-episode case that every other term reads;
  * ``induce_loss`` is a step event that applies one horizontal impulse at the
    sampled time to an induced episode only;
  * ``ball_no_bounce`` and ``ball_not_lifted`` are removed, so a lost or dead ball
    cannot discard the standing interval. ``held_ball_contact`` stays and is
    masked to the live cases in the termination itself;
  * the per-step ``ball_lost_penalty`` is replaced by the one-shot
    ``ball_lost_event`` at weight :data:`JUGGLE_BALL_LOST_EVENT_WEIGHT` (0.0), and
    the episode budget becomes :data:`JUGGLE_S10_EPISODE_LENGTH_S`.

  The four logging metrics report the realized episode-start shares beside the
  declared ones. ``ball_lost_events`` is removed with the per-step term it
  reported; the one-shot event is a reward output, not a per-step state.

  Play mode keeps the infinite episode length that every play config uses, but
  keeps the mixture itself so an evaluation sees the same cases as training.
  """
  if not play:
    cfg.episode_length_s = JUGGLE_S10_EPISODE_LENGTH_S

  ball_cfg = SceneEntityCfg("ball")
  cfg.events["induce_loss"] = EventTermCfg(
    mode="step",
    func=mdp.induce_loss,
    params={
      "asset_cfg": ball_cfg,
      "release_event": "reset_ball",
      "hold_event": "hold_ball",
      "speed_mps": BALL_CASE_INDUCED_SPEED_MPS,
    },
  )

  for name in ("ball_no_bounce", "ball_not_lifted"):
    cfg.terminations.pop(name, None)

  cfg.rewards.pop("ball_lost_penalty", None)
  cfg.rewards["ball_lost_event"] = RewardTermCfg(
    func=mdp.ball_lost_event,
    weight=JUGGLE_BALL_LOST_EVENT_WEIGHT,
    params={
      "ball_cfg": ball_cfg,
      "min_world_z": BALL_LOST_MIN_WORLD_Z_M,
      "release_event": "reset_ball",
    },
  )

  cfg.metrics.pop("ball_lost_events", None)
  for name, case in (
    ("ball_case_live", LIVE),
    ("ball_case_ground", GROUND),
    ("ball_case_never_seen", NEVER_SEEN),
    ("ball_case_induced", INDUCED),
  ):
    cfg.metrics[name] = MetricsTermCfg(
      func=mdp.ball_case_share,
      params={"release_event": "reset_ball", "case": case},
      reduce="mean",
    )


def _apply_delayed_release(cfg: ManagerBasedRlEnvCfg, hold: dict[str, object]) -> None:
  """Wire the held-ball phase into a config that would otherwise start it released.

  The reset term still runs and places the ball at the hold height; :class:`hold_ball`
  is registered **after** it and pins the ball at the sampled hold point, so the two
  agree by construction and the order cannot matter more than one lateral sample.
  """
  ball_cfg = SceneEntityCfg("ball")
  cfg.events["hold_ball"] = EventTermCfg(
    mode="reset",
    func=mdp.hold_ball,
    params={
      "asset_cfg": ball_cfg,
      # The waiting-alignment record needs the plate frame at the release instant.
      "plate_cfg": SceneEntityCfg("robot", body_names=("tennis_racket",)),
      **hold,
    },
  )
  cfg.events["advance_ball_hold"] = EventTermCfg(
    mode="step",
    func=mdp.advance_ball_hold,
    params={"hold_event": "hold_ball", "asset_cfg": ball_cfg},
  )

  # Energy is scored only once the ball is free: a held ball cannot be given energy,
  # so paying for its current energy is paying for nothing.
  cfg.rewards["ball_energy"].func = mdp.ball_energy_after_release
  cfg.rewards["ball_energy"].params["hold_event"] = "hold_ball"

  cfg.rewards["racket_under_ball"] = RewardTermCfg(
    func=mdp.racket_under_ball,
    weight=JUGGLE_WAIT_ALIGN_WEIGHT,
    params={
      "plate_cfg": SceneEntityCfg("robot", body_names=("tennis_racket",)),
      "ball_cfg": ball_cfg,
      "hold_event": "hold_ball",
      "plateau": JUGGLE_WAIT_PLATEAU_M,
      "sigma": JUGGLE_WAIT_SIGMA_M,
      "level_power": JUGGLE_WAIT_LEVEL_POWER,
      "speed_sigma": JUGGLE_WAIT_SPEED_SIGMA_MPS,
    },
  )

  cfg.terminations["held_ball_contact"] = TerminationTermCfg(
    func=mdp.held_ball_contact,
    params={
      "sensor_name": BAT_BALL_CONTACT_SENSOR,
      "hold_event": "hold_ball",
    },
  )
  cfg.terminations["ball_no_bounce"].params["hold_event"] = "hold_ball"

  cfg.metrics["ball_held"] = MetricsTermCfg(
    func=mdp.ball_hold_state,
    params={"hold_event": "hold_ball", "field": "held"},
    reduce="mean",
  )
  cfg.metrics["ball_hold_delay_s"] = MetricsTermCfg(
    func=mdp.ball_hold_state,
    params={"hold_event": "hold_ball", "field": "release_after_s"},
    reduce="last",
  )
  cfg.metrics["ball_release_alignment_m"] = MetricsTermCfg(
    func=mdp.ball_hold_state,
    params={"hold_event": "hold_ball", "field": "release_alignment_m"},
    reduce="last",
  )
  cfg.metrics["ball_hold_offset_m"] = MetricsTermCfg(
    func=mdp.ball_hold_state,
    params={"hold_event": "hold_ball", "field": "offset_m"},
    reduce="last",
  )


__all__ = [
  "STAGE_DELTAS",
  "agibot_x2_tennis_juggle_stage_env_cfg",
]
