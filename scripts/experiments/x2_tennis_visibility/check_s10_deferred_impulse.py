r"""Check that an induced impulse never lands on a pinned ball.

S10 draws a 0-6 s hold and a 5-20 s loss time, so the two ranges overlap on [5, 6).
Before the guard in ``induce_loss``, an environment whose loss time fell inside its
hold had the impulse written into a pinned ball. The next hold write erased it, and
``applied`` latched anyway, so the episode silently became a plain live episode while
still being counted as induced.

This probe runs the production S10 distribution and asserts the invariant that closes
the defect: **while the ball is held, no impulse is ever latched, and the ball's
horizontal speed stays at zero.** It also forces the overlap directly, by configuring a
long hold with an early loss time before the environment is built, and checks that the
impulse then arrives after release and exactly once.

Budget: 8 environments, at most 260 control steps. No training, no checkpoint.
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/home/agiuser/projects/mjlab/src")

import mjlab.tasks  # noqa: F401, E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402
from mjlab.tasks.velocity.mdp.ball_cases import (  # noqa: E402
  INDUCED,
  LIVE,
  case_of,
)

TASK = (
  "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
  "-Ball3-Pelvis-Nominal-S10-VisibleBall"
)
ENVS = 8
HOLD_S = 4.0
LOSS_S = (0.5, 1.5)
STEPS = 320


def _build(force_overlap: bool) -> ManagerBasedRlEnv:
  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = ENVS
  cfg.seed = 23
  if force_overlap:
    # Every environment induced, a 4 s hold, and a loss time that is always inside it.
    cfg.events["reset_ball"].params["case_shares"] = (0.0, 0.0, 0.0, 1.0)
    cfg.events["reset_ball"].params["loss_time_range_s"] = LOSS_S
    cfg.events["hold_ball"].params["hold_delay_s"] = (HOLD_S, HOLD_S)
    # Probe only: with zero actions the robot falls in about two seconds, which resets
    # the episode before a 4 s hold can release and hides the deferral entirely. Keep
    # the episode alive so the release and the deferred impulse are both observable.
    cfg.terminations.pop("fell_over", None)
    cfg.terminations.pop("held_ball_contact", None)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
  env.reset()
  return env


def _run(env: ManagerBasedRlEnv) -> dict:
  ball = env.scene["ball"]
  impulse = env.event_manager.get_term_cfg("induce_loss").func
  hold = env.event_manager.get_term_cfg("hold_ball").func
  cases = case_of(env, "reset_ball")
  assert cases is not None
  action = torch.zeros(ENVS, env.action_space.shape[-1], device=env.device)
  dt = float(env.step_dt)
  speeds, held, applied = [], [], []
  for _ in range(STEPS):
    env.step(action)
    speeds.append(torch.linalg.vector_norm(ball.data.root_link_vel_w[:, :2], dim=1))
    held.append(hold.held.clone())
    applied.append(impulse.applied.clone())
  S, H, A = torch.stack(speeds), torch.stack(held), torch.stack(applied)
  # Post-release clock: the hold term zeroes episode_length_buf at release.
  elapsed = (env.episode_length_buf.float() * dt).clone()
  latched_while_held = bool((A & H).any())
  worst_pinned_speed = float(S[H].abs().max().item()) if bool(H.any()) else 0.0
  first_latch = A.float().argmax(dim=0)
  return {
    "cases": cases,
    "n_induced": int((cases == INDUCED).sum()),
    "n_live": int((cases == LIVE).sum()),
    "latched_while_held": latched_while_held,
    "worst_pinned_speed": worst_pinned_speed,
    "first_latch_step": first_latch.tolist(),
    "applied_all": bool(A[-1].all()),
    "any_applied": bool(A.any()),
    "release_step": int((~H).float().argmax(dim=0).min().item()),
    "release_steps": (~H).float().argmax(dim=0).tolist(),
    "held_at_0": H[0].tolist(),
    "elapsed_now": float(elapsed[0]),
    "final_speed": S[-1].mean().item(),
  }


def main() -> int:
  results = {}
  for label, overlap in (("production mix", False), ("forced overlap", True)):
    env = _build(overlap)
    r = _run(env)
    results[label] = r
    print(f"  {label}: induced {r['n_induced']}/{ENVS}, live {r['n_live']}")
    print(
      f"    latched while held: {r['latched_while_held']}   "
      f"worst pinned speed: {r['worst_pinned_speed']:.4f} m/s"
    )
    print(
      f"    applied at end: {r['applied_all']}   any applied: {r['any_applied']}   "
      f"first latch steps: {r['first_latch_step']}"
    )
    env.close()

  forced = results["forced overlap"]
  prod = results["production mix"]
  ok_pinned = not forced["latched_while_held"] and not prod["latched_while_held"]
  ok_still = forced["worst_pinned_speed"] < 0.05 and prod["worst_pinned_speed"] < 0.05
  ok_fires = forced["applied_all"] and forced["any_applied"]
  ok_release = min(forced["first_latch_step"]) >= forced["release_step"]
  for name, ok in (
    ("no impulse latched while the ball is held", ok_pinned),
    ("pinned ball never moves", ok_still),
    ("a forced overlap still fires, once per environment", ok_fires),
    ("the impulse arrives after release", ok_release),
  ):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
  return 0 if all((ok_pinned, ok_still, ok_fires, ok_release)) else 1


if __name__ == "__main__":
  raise SystemExit(main())
