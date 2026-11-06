"""AgiBot X2 staged juggling fine-tune configurations.

A **stage** is one input condition changed against the previous stage, nothing
else. The scene, the 1 ms contact physics, the vendor gains, the actuator limits,
the reward set, the terminations, the resets, the zero command, the locked head
and the observation layout stay the previous stage's by construction, so the
actor/critic/action widths do not move and a stage is trained by resuming the
previous stage's checkpoint with its optimizer state, at the same environment
count and seed. A stage exists so one condition can be screened on its own,
without a scheduler and without changing the contract.

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

from mjlab.asset_zoo.robots.agibot_x2.tennis_constants import (
  JUGGLE_HOLD_DELAY_S,
  JUGGLE_HOLD_LONG_DELAY_S,
  JUGGLE_HOLD_HEIGHT_M,
  JUGGLE_HOLD_LATERAL_RANGE_M,
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
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.agibot_x2.tennis_ball_env_cfg import (
  BAT_BALL_CONTACT_SENSOR,
)
from mjlab.tasks.velocity.config.agibot_x2.tennis_juggle_nominal_env_cfg import (
  agibot_x2_tennis_juggle_nominal_env_cfg,
)

_HOLD_HEIGHT_ENV = "MJLAB_JUGGLE_HOLD_HEIGHT_M"


def _hold_height() -> float:
  """Hold height for the delayed-release rung, in metres.

  Defaults to :data:`JUGGLE_HOLD_HEIGHT_M`, so a run that does not set
  ``MJLAB_JUGGLE_HOLD_HEIGHT_M`` is byte-identical to the committed rung. The
  override exists because ``play`` has no ``--env.*`` flags: watching one policy
  released from several heights would otherwise need a source edit per height.
  Both the reset and the hold read this one value, so they cannot disagree and the
  start-height metric stays true.

  This is a viewing and ablation knob, not a training condition. A rung that
  randomises the height belongs in the stage table, not in an environment variable.
  """
  raw = os.environ.get(_HOLD_HEIGHT_ENV, "")
  return float(raw) if raw not in ("", None) else JUGGLE_HOLD_HEIGHT_M


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
      "hold_delay_s": JUGGLE_HOLD_DELAY_S,
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
      "drop_height_range": (0.10, 0.40),
      "drop_lateral_range_m": JUGGLE_HOLD_LATERAL_RANGE_M,
    },
    "hold": {
      "hold_height_m": None,
      "hold_delay_s": JUGGLE_HOLD_DELAY_S,
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
      "drop_height_range": (0.10, 0.40),
      "drop_lateral_range_m": JUGGLE_HOLD_LATERAL_RANGE_M,
    },
    "hold": {
      "hold_height_m": None,
      "hold_delay_s": JUGGLE_HOLD_LONG_DELAY_S,
      "lateral_range_m": JUGGLE_HOLD_LATERAL_RANGE_M,
    },
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
  cfg = agibot_x2_tennis_juggle_nominal_env_cfg(play=play)
  cfg.observations["actor"].terms["ball_hist_pelvis"].params.update(delta)
  if reset_delta:
    cfg.events["reset_ball"].params.update(reset_delta)
  if hold_delta:
    _apply_delayed_release(cfg, hold_delta)
  return cfg


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


__all__ = [
  "STAGE_DELTAS",
  "agibot_x2_tennis_juggle_stage_env_cfg",
]
