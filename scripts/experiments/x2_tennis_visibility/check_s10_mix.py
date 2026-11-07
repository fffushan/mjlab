"""Check the second half of stage S10: the mixed safety contract.

The seven cases of ``docs/design/x2_tennis_s10_mixed_plan.md`` section 4, plus a
check that a stage without the mixture still receives no case parameters. Each case
prints its own numbers and the script exits non-zero if any case fails. It is a
**probe**: 8 environments, at most 1500 control steps, no training, no checkpoint,
no hardware.

The probe holds the robot at its post-reset stance for the 1500-step rollout. The
stage is a fine-tune rung and the zero action is not a standing policy: without
the hold the untrained robot falls in about 1.3 s, which would terminate the
live episodes before the budget and would move the head camera during the
never-seen check. The stance hold changes no stage parameter; it only keeps the
robot where the reset put it so the episode budget and the stream geometry are not
confounded by a fall. The dead-case placement, the reward masks, the hold mask and
the induced-loss event all run unchanged.

The rollout's cases are assigned explicitly, two per case, by calling the same
``reset_ball`` and ``hold_ball`` terms with a one-hot ``case_shares``. The realized
share check draws from the production shares separately, so the forced assignment
cannot bias it.

Usage::

    .venv/bin/python \\
        scripts/experiments/x2_tennis_visibility/check_s10_mix.py
"""

from __future__ import annotations

from typing import Any

import torch

from mjlab.asset_zoo.robots.agibot_x2.tennis_constants import (
  BALL_CASE_GROUND_AHEAD_M,
  BALL_CASE_GROUND_LATERAL_RANGE_M,
  BALL_CASE_INDUCED_SPEED_MPS,
  BALL_CASE_LOSS_TIME_RANGE_S,
  BALL_CASE_NEVER_SEEN_BEHIND_M,
  BALL_CASE_SHARES,
  BALL_RADIUS,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.config.agibot_x2.tennis_juggle_stage_env_cfg import (
  agibot_x2_tennis_juggle_stage_env_cfg,
)
from mjlab.tasks.velocity.mdp.ball_cases import (
  CASES,
  GROUND,
  INDUCED,
  LIVE,
  NEVER_SEEN,
  case_of,
)
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.torch import configure_torch_backends

STAGE = "s10-visible-ball"
PRE_MIX_STAGE = "s9b-sim-to-real"

PROBE_ENVS = 8
PROBE_SEED = 11
PROBE_STEPS = 1500
"""The stage budget: 30 s at the 0.02 s control step."""

SHARE_ROUNDS = 400
"""Reset rounds used for the realized-share check: 3200 episode starts."""

SHARE_TOLERANCE = 0.05
"""Allowed absolute deviation of a realized share, in probability."""

BALL_SLICE = slice(100, 112)
"""The actor's twelve ball-history numbers: ``ball_hist_pelvis`` is last."""

PASS = "PASS"
FAIL = "FAIL"

GROUP_NAMES = ("live", "ground", "never_seen", "induced")


def _device() -> str:
  return "cuda:0" if torch.cuda.is_available() else "cpu"


def _build_env() -> tuple[Any, ManagerBasedRlEnv]:
  cfg = agibot_x2_tennis_juggle_stage_env_cfg(stage=STAGE)
  cfg.scene.num_envs = PROBE_ENVS
  cfg.seed = PROBE_SEED
  cfg.auto_reset = False
  env = ManagerBasedRlEnv(cfg=cfg, device=_device(), render_mode=None)
  return cfg, env


def _reset_term(env: ManagerBasedRlEnv):
  return env.event_manager.get_term_cfg("reset_ball")


def _hold_term(env: ManagerBasedRlEnv):
  return env.event_manager.get_term_cfg("hold_ball")


def _case_shares(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Draw ``SHARE_ROUNDS`` production resets and report the realized shares.

  Also collects the loss times drawn for the induced cases, so the timing range is
  a distribution rather than the two samples of the rollout.
  """
  term = _reset_term(env)
  counts = torch.zeros(len(CASES), dtype=torch.long, device=env.device)
  induced_times: list[torch.Tensor] = []
  for _ in range(SHARE_ROUNDS):
    term.func(env, None, **term.params)
    cases = term.func.case
    assert cases is not None
    counts += torch.bincount(cases, minlength=len(CASES))
    loss_at = term.func.loss_at_s
    assert loss_at is not None
    induced_times.append(loss_at[cases == INDUCED].clone())
  total = int(counts.sum())
  realized = counts.float() / float(total)
  return realized, torch.cat(induced_times)


def _force_cases(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Assign two environments per case through the production reset terms."""
  ids = torch.arange(PROBE_ENVS, device=env.device, dtype=torch.int)
  groups = (
    (LIVE, ids[0:2]),
    (GROUND, ids[2:4]),
    (NEVER_SEEN, ids[4:6]),
    (INDUCED, ids[6:8]),
  )
  reset = _reset_term(env)
  hold = _hold_term(env)
  for case, group in groups:
    shares = tuple(1.0 if index == case else 0.0 for index in CASES)
    params = dict(reset.params)
    params["case_shares"] = shares
    reset.func(env, group, **params)
    # Fix the wait to zero so one control step is one post-release step. This is
    # the same knob MJLAB_JUGGLE_HOLD_DELAY_S exposes, passed directly so the
    # probe does not depend on when the stage module was imported.
    hold_params = dict(hold.params)
    hold_params["hold_delay_s"] = (0.0, 0.0)
    hold.func(env, group, **hold_params)
  expected = torch.empty(PROBE_ENVS, dtype=torch.long, device=env.device)
  for case, group in groups:
    expected[group] = case
  cases = case_of(env, "reset_ball")
  assert cases is not None
  if not torch.equal(cases, expected):
    raise RuntimeError(f"forced cases {cases.tolist()} != expected {expected.tolist()}")
  # The reset-time observation still holds the pre-forcing ball. Clear it, or the
  # first acquisition could admit a frame the forced placement never produced.
  env.observation_manager.reset()
  env.sim.forward()
  return cases


def _placement_errors(
  env: ManagerBasedRlEnv, cases: torch.Tensor
) -> tuple[float, float, float, float]:
  """Ground and never-seen placement errors, in metres.

  Returns ``(ground_ahead, ground_lateral, ground_z, never_behind)``. The two z
  errors are folded into one maximum absolute value.
  """
  robot = env.scene["robot"]
  ball = env.scene["ball"]
  pelvis = robot.data.root_link_pos_w
  pelvis_quat = robot.data.root_link_quat_w
  delta = ball.data.root_link_pos_w - pelvis
  local = quat_apply_inverse(pelvis_quat, delta)
  ground = cases == GROUND
  never = cases == NEVER_SEEN
  ground_ahead = float((local[ground, 0] - BALL_CASE_GROUND_AHEAD_M).abs().max())
  ground_lateral = float(
    torch.clamp(
      local[ground, 1].abs() - BALL_CASE_GROUND_LATERAL_RANGE_M, min=0.0
    ).max()
  )
  ground_z = float((ball.data.root_link_pos_w[ground, 2] - BALL_RADIUS).abs().max())
  never_behind = float((local[never, 0] + BALL_CASE_NEVER_SEEN_BEHIND_M).abs().max())
  never_z = float((ball.data.root_link_pos_w[never, 2] - BALL_RADIUS).abs().max())
  return ground_ahead, ground_lateral, max(ground_z, never_z), never_behind


def _hold_stance(robot: Any, root_pose: torch.Tensor, joint_pos: torch.Tensor) -> None:
  """Put the robot back at its post-reset stance after a control step."""
  n = root_pose.shape[0]
  robot.write_root_link_pose_to_sim(root_pose)
  robot.write_root_link_velocity_to_sim(torch.zeros((n, 6), device=root_pose.device))
  robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
  robot.set_joint_position_target(joint_pos)


def main() -> int:
  configure_torch_backends()
  torch.set_grad_enabled(False)
  print(
    f"S10 mixed-case checks; stage {STAGE}, {PROBE_ENVS} envs, seed {PROBE_SEED}, "
    f"{PROBE_STEPS} zero-action control steps"
  )
  print(
    f"declared shares {BALL_CASE_SHARES}, loss times "
    f"{BALL_CASE_LOSS_TIME_RANGE_S} s, budget 30 s"
  )
  print()

  results: list[tuple[str, bool]] = []
  cfg, env = _build_env()
  try:
    env.reset()

    # 1. Realized shares.
    print("[1] realized episode-start shares over production resets")
    realized, induced_times = _case_shares(env)
    share_ok = True
    for name, got, want in zip(
      GROUP_NAMES, realized.tolist(), BALL_CASE_SHARES, strict=True
    ):
      delta = got - want
      ok = abs(delta) <= SHARE_TOLERANCE
      share_ok = share_ok and ok
      print(
        f"  {name:11s} {got:.4f}  declared {want:.4f}  delta {delta:+.4f}  "
        f"{PASS if ok else FAIL}"
      )
    print(
      f"  total starts {SHARE_ROUNDS * PROBE_ENVS}, induced loss draws "
      f"{induced_times.numel()}, sampled range [{float(induced_times.min()):.3f}, "
      f"{float(induced_times.max()):.3f}] s"
    )
    print(
      f"  -> {PASS if share_ok else FAIL}: every share within +-{SHARE_TOLERANCE:.2f}"
    )
    results.append(("realized shares", share_ok))

    # Static contract checks before the rollout.
    print("[2] no ball-caused terminations besides the masked hold contact")
    terminations = sorted(cfg.terminations)
    absent = (
      "ball_no_bounce" not in cfg.terminations
      and "ball_not_lifted" not in cfg.terminations
      and "ball_dropped" not in cfg.terminations
    )
    hold_present = "held_ball_contact" in cfg.terminations
    print(f"  terminations {terminations}")
    print(
      f"  ball_no_bounce absent: {'ball_no_bounce' not in cfg.terminations}; "
      f"ball_not_lifted absent: {'ball_not_lifted' not in cfg.terminations}; "
      f"ball_dropped absent: {'ball_dropped' not in cfg.terminations}; "
      f"held_ball_contact present: {hold_present}"
    )
    term_ok = absent and hold_present
    print(f"  -> {PASS if term_ok else FAIL}")
    results.append(("no ball terminations", term_ok))

    print("[3] live-only stage keeps the mixture off")
    pre_cfg = agibot_x2_tennis_juggle_stage_env_cfg(stage=PRE_MIX_STAGE)
    pre_params = pre_cfg.events["reset_ball"].params
    mixture_keys = {"case_shares", "loss_time_range_s"} & set(pre_params)
    print(
      f"  {PRE_MIX_STAGE!r} reset_ball mixture keys {sorted(mixture_keys)}, "
      f"case_shares in S10 {cfg.events['reset_ball'].params['case_shares']}"
    )
    pre_ok = not mixture_keys and "case_shares" in cfg.events["reset_ball"].params
    print(
      f"  -> {PASS if pre_ok else FAIL}: the case draw exists only in the "
      f"declaring stage"
    )
    results.append(("mixture declared only in S10", pre_ok))

    # Force two environments per case, then release the live holds.
    cases = _force_cases(env)
    reset = _reset_term(env)
    hold = _hold_term(env)
    held_at_reset = hold.func.held.clone()
    dead = (cases == GROUND) | (cases == NEVER_SEEN)
    live = (cases == LIVE) | (cases == INDUCED)
    never = cases == NEVER_SEEN
    print("[4] forced case placement and hold mask")
    ground_ahead, ground_lateral, ball_z, never_behind = _placement_errors(env, cases)
    print(
      f"  cases {cases.tolist()}; held at reset {held_at_reset.int().tolist()}; "
      f"dead held false {bool((~held_at_reset[dead]).all())}"
    )
    print(
      f"  ground: ahead error {ground_ahead:.2e} m, lateral beyond "
      f"{BALL_CASE_GROUND_LATERAL_RANGE_M:.2f} m by {ground_lateral:.2e} m; "
      f"ground/never z error {ball_z:.2e} m (radius {BALL_RADIUS:.4f} m); "
      f"never behind error {never_behind:.2e} m"
    )
    place_ok = (
      ground_ahead < 1e-4
      and ground_lateral == 0.0
      and ball_z < 1e-5
      and never_behind < 1e-4
      and bool((~held_at_reset[dead]).all())
      and bool(held_at_reset[live].all())
    )
    print(
      f"  -> {PASS if place_ok else FAIL}: dead starts rest on the floor and are "
      f"not held"
    )
    results.append(("case placement and hold mask", place_ok))

    # Step-0 reward snapshot: the hold is still pinned here, so the hold-phase
    # reward is observable for the live cases.
    reward_names = ("ball_energy", "ball_centred", "ball_in_fov", "racket_under_ball")
    live_max = {
      name: torch.zeros(PROBE_ENVS, device=env.device) for name in reward_names
    }
    dead_max = {
      name: torch.zeros(PROBE_ENVS, device=env.device) for name in reward_names
    }
    reward_manager = env.reward_manager
    for name in reward_names:
      term = reward_manager.get_term_cfg(name)
      value = term.func(env, **term.params)
      live_max[name] = torch.maximum(
        live_max[name], torch.where(live, value, torch.zeros_like(value))
      )
      dead_max[name] = torch.maximum(
        dead_max[name], torch.abs(torch.where(dead, value, torch.zeros_like(value)))
      )

    # Release the live holds one step early so the 1500-step rollout starts at the
    # post-release clock and the live episodes reach exactly the 1500-step timeout.
    env.event_manager.apply(mode="step", dt=env.step_dt)
    print(
      f"[5] release and step 0 ball rewards: held after release "
      f"{hold.func.held.int().tolist()}"
    )
    print(
      "  live  "
      + "  ".join(
        f"{name} {float(live_max[name][live].max()):.4f}" for name in reward_names
      )
    )
    print(
      "  dead  "
      + "  ".join(
        f"{name} {float(dead_max[name][dead].max()):.2e}" for name in reward_names
      )
    )
    step0_ok = all(
      float(live_max[name][live].max()) > 0.0
      for name in ("ball_centred", "ball_in_fov", "racket_under_ball")
    )
    results.append(("step-0 live ball rewards non-zero", step0_ok))

    # 1500-step rollout.
    robot = env.scene["robot"]
    root_pose = robot.data.root_link_pose_w.clone()
    joint_pos = robot.data.joint_pos.clone()
    action = torch.zeros(
      (env.num_envs, env.action_manager.total_action_dim), device=env.device
    )
    lost_term = reward_manager.get_term_cfg("ball_lost_event")
    induce = env.event_manager.get_term_cfg("induce_loss").func
    prev_applied = induce.applied.clone()
    timeouts = torch.zeros(PROBE_ENVS, dtype=torch.long, device=env.device)
    term_steps = torch.zeros(PROBE_ENVS, dtype=torch.long, device=env.device)
    impulse_count = torch.zeros(PROBE_ENVS, dtype=torch.long, device=env.device)
    impulse_steps = torch.zeros(PROBE_ENVS, dtype=torch.long, device=env.device)
    impulse_speed = torch.zeros(PROBE_ENVS, device=env.device)
    lost_counts = torch.zeros(PROBE_ENVS, device=env.device)
    held_dead_true = 0
    never_stream_max = 0.0
    unexpected: list[int] = []

    for step in range(1, PROBE_STEPS + 1):
      obs, _, terminated, truncated, _ = env.step(action)
      _hold_stance(robot, root_pose, joint_pos)
      env.sim.forward()

      for name in reward_names:
        term = reward_manager.get_term_cfg(name)
        value = term.func(env, **term.params)
        live_max[name] = torch.maximum(
          live_max[name], torch.where(live, value, torch.zeros_like(value))
        )
        dead_max[name] = torch.maximum(
          dead_max[name], torch.abs(torch.where(dead, value, torch.zeros_like(value)))
        )

      lost_counts += lost_term.func(env, **lost_term.params)
      applied = induce.applied
      rising = applied & ~prev_applied
      impulse_count += rising.long()
      impulse_steps = torch.where(
        rising, torch.full_like(impulse_steps, step), impulse_steps
      )
      speed = torch.linalg.norm(env.scene["ball"].data.root_link_vel_w[:, :2], dim=1)
      impulse_speed = torch.where(rising, speed, impulse_speed)
      prev_applied = applied.clone()

      held_contact = env.termination_manager.get_term("held_ball_contact")
      held_dead_true += int(held_contact[dead].sum())
      never_stream_max = max(
        never_stream_max, float(obs["actor"][never, BALL_SLICE].abs().max())
      )

      timeouts = torch.where(
        truncated & (timeouts == 0), torch.full_like(timeouts, step), timeouts
      )
      term_steps = torch.where(
        terminated & (term_steps == 0),
        torch.full_like(term_steps, step),
        term_steps,
      )
      if bool(terminated.any()):
        unexpected.append(step)
        break

    # 6. Budget.
    print("[6] episode budget")
    print(
      f"  episode_length_s {cfg.episode_length_s}, env.max_episode_length "
      f"{env.max_episode_length}, steps {PROBE_STEPS}"
    )
    print(
      f"  first timeout step live {timeouts[live].tolist()}, dead "
      f"{timeouts[dead].tolist()}, all {timeouts.tolist()}"
    )
    print(
      f"  non-timeout terminations {unexpected or 0}; steps recorded "
      f"{int((timeouts > 0).sum())} of {PROBE_ENVS}"
    )
    budget_ok = (
      cfg.episode_length_s == 30.0
      and env.max_episode_length == PROBE_STEPS
      and not unexpected
      and bool((timeouts[live] == PROBE_STEPS).all())
      and bool((timeouts[dead] == PROBE_STEPS).all())
    )
    print(
      f"  -> {PASS if budget_ok else FAIL}: every live episode reached the "
      f"1500-step ceiling"
    )
    results.append(("episode budget", budget_ok))

    # 7. Induced loss.
    print("[7] induced loss")
    induced = cases == INDUCED
    impulse_times = impulse_steps.float() * env.step_dt
    remaining = (PROBE_STEPS - impulse_steps.float()) * env.step_dt
    print(
      f"  impulses per induced env {impulse_count[induced].tolist()}; "
      f"times {[round(float(v), 3) for v in impulse_times[induced].tolist()]} s; "
      f"remaining {[round(float(v), 3) for v in remaining[induced].tolist()]} s"
    )
    print(
      f"  sampled loss_at {[round(float(v), 3) for v in reset.func.loss_at_s[induced].tolist()]} s; "
      f"speed at impulse {[round(float(v), 3) for v in impulse_speed[induced].tolist()]} m/s"
    )
    induce_ok = (
      bool((impulse_count[induced] == 1).all())
      and bool((impulse_count[~induced] == 0).all())
      and bool((impulse_times[induced] >= BALL_CASE_LOSS_TIME_RANGE_S[0]).all())
      and bool((impulse_times[induced] <= BALL_CASE_LOSS_TIME_RANGE_S[1]).all())
      and bool((remaining[induced] >= 10.0).all())
      and bool(
        (impulse_speed[induced] - BALL_CASE_INDUCED_SPEED_MPS).abs().max() < 1e-3
      )
    )
    print(
      f"  -> {PASS if induce_ok else FAIL}: one {BALL_CASE_INDUCED_SPEED_MPS} m/s "
      f"impulse, 5-20 s, at least 10 s left"
    )
    results.append(("induced loss", induce_ok))

    # 8. Held-contact mask.
    print("[8] held-contact mask")
    print(
      f"  held_ball_contact true on GROUND/NEVER_SEEN over {PROBE_STEPS} steps: "
      f"{held_dead_true}; on live/induced: "
      f"{int(env.termination_manager.get_term('held_ball_contact')[live].sum())} "
      f"at the final step"
    )
    mask_ok = held_dead_true == 0
    print(f"  -> {PASS if mask_ok else FAIL}: never true for a dead start")
    results.append(("held-contact mask", mask_ok))

    # 9. Dead-case and live ball rewards.
    print("[9] ball rewards: dead zero, live non-zero")
    any_dead = False
    any_live = False
    for name in reward_names:
      dead_value = float(dead_max[name][dead].max())
      live_value = float(live_max[name][live].max())
      any_dead = any_dead or dead_value != 0.0
      any_live = any_live or live_value > 0.0
      print(f"  {name:19s} dead max |r| {dead_value:.2e}   live max r {live_value:.4f}")
    reward_ok = not any_dead and any_live
    print(
      f"  -> {PASS if reward_ok else FAIL}: every dead value is exactly zero and "
      f"a live episode earns ball reward"
    )
    results.append(("dead-case rewards", reward_ok))

    # 10. Never-seen stream.
    print("[10] never-seen stream")
    print(
      f"  actor ball-history max |value| on NEVER_SEEN over {PROBE_STEPS} steps: "
      f"{never_stream_max:.2e}"
    )
    never_ok = never_stream_max == 0.0
    print(f"  -> {PASS if never_ok else FAIL}: all twelve numbers stayed zero")
    results.append(("never-seen stream", never_ok))

    # 11. Loss charge.
    print("[11] loss charge")
    print(
      f"  ball_lost_event fires live {lost_counts[live].tolist()}, induced "
      f"{lost_counts[induced].tolist()}, ground "
      f"{lost_counts[cases == GROUND].tolist()}, never "
      f"{lost_counts[never].tolist()}"
    )
    charge_ok = (
      bool((lost_counts[dead] == 0).all())
      and bool((lost_counts[live] <= 1).all())
      and bool((lost_counts[induced] <= 1).all())
      and bool((lost_counts[induced] >= 1).all())
    )
    print(
      f"  -> {PASS if charge_ok else FAIL}: at most once per live episode and "
      f"never for a dead start"
    )
    results.append(("loss charge", charge_ok))
  finally:
    env.close()

  print()
  print("summary")
  for name, ok in results:
    print(f"  {PASS if ok else FAIL}  {name}")
  failed = [name for name, ok in results if not ok]
  print()
  print("ALL CHECKS PASSED" if not failed else f"FAILED: {', '.join(failed)}")
  return 0 if not failed else 1


if __name__ == "__main__":
  raise SystemExit(main())
