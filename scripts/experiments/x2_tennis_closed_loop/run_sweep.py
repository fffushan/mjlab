"""One closed-loop latency cell for an X2 juggling checkpoint.

One invocation is **one latency cell**. It builds the task's environment, sets the
actor's ball-history stream latency in process, removes the ``time_out``
termination and runs the policy closed loop for a fixed horizon. Every other
termination stays installed: inside this loop a termination *is* the failure
signal, so none may be disabled.

It reports two independent signals.

**A. Streak (recorder).** The trace recorder records one episode per environment
and stops at that environment's first reset, so each environment contributes
exactly one streak. Per environment the runner reports the qualified flights in
that streak, the apex scatter, the contact durations, the lateral landing
positions and the term that ended it. An environment that never terminates inside
the horizon is **censored**, not a failure, and is reported as censored.

**B. Resets (``env.reset_buf``).** Every step's reset flags are summed per
environment over the whole horizon. Resets per minute is the robustness number
that stays meaningful when signal A censors every environment.

**Lateral landing error.** For every racket contact in a streak, the ball position
at that contact is expressed in the plate frame. The runner reports the sample
standard deviation of the in-plane ``(x, y)`` coordinates and the distribution of
the radial fraction of the modeled plate half-extent (``1.0`` is the bed edge).

Neither signal is a calibration of the qualified-flight thresholds, and neither is
a hardware claim. The two limits that matter for reading the numbers are in
``LIMITS`` and in ``README.md``: the recorder gives one streak per environment, and
the stream latency is quantised to whole 20 ms control steps.

This directory imports the sibling E0 evaluator (``../x2_tennis_nominal_eval``)
read-only for the recorder, the scoring rule and the checkpoint loader. It
modifies no existing file.

Usage, one cell (``README.md`` has the full task id):

  uv run python scripts/experiments/x2_tennis_closed_loop/run_sweep.py \
    --task "$TASK" --checkpoint /path/to/model_6998.pt \
    --latency-min 0.08 --latency-max 0.10 --target 0.40 \
    --horizon-s 180 --num-envs 32 \
    --output-dir logs/diagnostics/x2_cl/lat_0.08_0.10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_EVAL_DIR = _HERE.parent / "x2_tennis_nominal_eval"
"""The sibling E0 evaluator, used read-only."""

BALL_HISTORY_TERM = "ball_hist_pelvis"
"""Actor observation term whose stream latency this runner overrides."""

STREAM_LATENCY_KEY = "stream_latency_s"
"""Parameter of :data:`BALL_HISTORY_TERM` that the latency cell sets."""

REMOVED_TERMINATION = "time_out"
"""The horizon cap. Removing it makes an episode end on physics or not at all."""

REWARD_TARGET_PATHS = (
  ("rewards", "ball_energy"),
  ("metrics", "ball_energy"),
  ("metrics", "ball_energy_error"),
)
"""Terms that carry the reward's target apex height, all read by the task."""

TARGET_ENV_VAR = "MJLAB_JUGGLE_ENERGY_TARGET_HEIGHT_M"
"""The task registry reads the reward target from this variable at import time."""

LIMITS = (
  "The recorder stops recording an environment after that environment's first "
  "reset, so one environment gives exactly one streak. A second streak in the same "
  "environment is never recorded, whatever the horizon.",
  "The stream latency is quantised by the observation term to whole control steps "
  "of 20 ms (round(latency / step_dt)), so a cell's measured lag is the rounded "
  "value, not the requested one.",
  "Censoring is not success. A censored environment only says the policy survived "
  "the remaining horizon; it says nothing about how long it would have survived.",
  "Resets per minute counts every termination on the whole horizon, including the "
  "auto-reset episodes the recorder does not score.",
  "The qualified-flight thresholds are the frozen plan values, not a calibration "
  "against this sweep, and the lateral spread is the spread of the recorded contact "
  "samples, not of the controlled landing target.",
  "This is a simulation measurement with no hardware, deployment or sim-to-real claim.",
)


def _ensure_eval_path() -> None:
  """Put the sibling E0 evaluator on ``sys.path`` so its modules import by name."""
  if not _EVAL_DIR.is_dir():
    raise SystemExit(f"missing the sibling E0 evaluator directory: {_EVAL_DIR}")
  if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", required=True, help="Task id to build.")
  parser.add_argument(
    "--checkpoint",
    type=Path,
    required=True,
    help="Policy checkpoint. With --zero-action it is recorded but not loaded.",
  )
  parser.add_argument(
    "--latency-min", type=float, required=True, help="Lower stream latency, seconds."
  )
  parser.add_argument(
    "--latency-max", type=float, required=True, help="Upper stream latency, seconds."
  )
  parser.add_argument(
    "--target", type=float, required=True, help="Target apex height, metres."
  )
  parser.add_argument(
    "--horizon-s",
    type=float,
    default=180.0,
    help="Simulated seconds per run. Also the episode length, since time_out is "
    "removed.",
  )
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument(
    "--zero-action",
    action="store_true",
    help="Hold the default joint pose instead of loading the policy. Smoke test "
    "only: a zero-action policy cannot juggle.",
  )
  return parser.parse_args(argv)


def _check_args(args: argparse.Namespace) -> None:
  """Refuse argument combinations that cannot mean anything."""
  if not 0.0 <= args.latency_min <= args.latency_max:
    raise SystemExit(
      f"latency range must satisfy 0 <= min <= max, got "
      f"({args.latency_min}, {args.latency_max})"
    )
  if not 0.0 < args.horizon_s:
    raise SystemExit(f"--horizon-s must be positive, got {args.horizon_s}")
  if args.num_envs < 1:
    raise SystemExit(f"--num-envs must be positive, got {args.num_envs}")


def _configure_latency(
  env_cfg: Any, latency_min: float, latency_max: float, step_dt: float
) -> dict[str, Any]:
  """Point the actor's ball-history stream at this cell's latency range.

  Args:
    env_cfg: The resolved config, edited in place.
    latency_min: Lower bound of the per-acquisition latency, seconds.
    latency_max: Upper bound, seconds.
    step_dt: Control step, seconds. The term rounds each drawn latency by it.

  Returns:
    The requested range, the value the config now holds, and the range in control
    steps that the term can actually produce.

  Raises:
    AssertionError: If the task has no such actor term or parameter. This runner
      overrides nothing else about perception, so a missing term means the task is
      not the one it claims to be.
  """
  terms = env_cfg.observations["actor"].terms
  assert BALL_HISTORY_TERM in terms, (
    f"actor has no term {BALL_HISTORY_TERM!r}; its terms are {sorted(terms)}. This "
    "runner only overrides the pelvis ball-history stream latency."
  )
  from mjlab.asset_zoo.robots.agibot_x2.tennis_constants import BALL_TRUE_HISTORY_S

  params = terms[BALL_HISTORY_TERM].params
  assert params is not None and STREAM_LATENCY_KEY in params, (
    f"term {BALL_HISTORY_TERM!r} has no {STREAM_LATENCY_KEY!r} parameter; its keys "
    f"are {sorted(params or {})}"
  )
  if float(latency_max) >= BALL_TRUE_HISTORY_S:
    # The term clamps the capture step to its ring instead of refusing, so a cell
    # longer than the ring would serve a sample NEWER than the one requested while
    # this runner reported the requested value.
    raise SystemExit(
      f"latency {latency_max} s reaches the term's true-history ring "
      f"({BALL_TRUE_HISTORY_S} s), where the capture step wraps; refusing to "
      "measure a condition the term cannot represent"
    )
  params[STREAM_LATENCY_KEY] = (float(latency_min), float(latency_max))
  resolved = tuple(float(value) for value in params[STREAM_LATENCY_KEY])
  return {
    "term": BALL_HISTORY_TERM,
    "parameter": STREAM_LATENCY_KEY,
    "requested_s": [float(latency_min), float(latency_max)],
    "resolved_s": list(resolved),
    "quantised_control_steps": [
      int(round(latency_min / step_dt)),
      int(round(latency_max / step_dt)),
    ],
    "step_dt_s": float(step_dt),
  }


def _configure_horizon(env_cfg: Any, horizon_s: float) -> dict[str, Any]:
  """Remove the horizon cap and make the episode length the horizon.

  ``episode_length_s`` is set to the horizon although no termination reads it any
  more, so ``max_episode_length`` - which sizes the recorder buffers - is consistent
  with how long the loop actually runs.

  Args:
    env_cfg: The resolved config, edited in place.
    horizon_s: Simulated seconds to run.

  Returns:
    The terminations left installed and whether ``time_out`` was present.

  Raises:
    AssertionError: If ``time_out`` is still installed after the removal.
  """
  removed = env_cfg.terminations.pop(REMOVED_TERMINATION, None)
  env_cfg.episode_length_s = float(horizon_s)
  assert REMOVED_TERMINATION not in env_cfg.terminations, (
    f"{REMOVED_TERMINATION!r} is still installed; the horizon would cap every "
    "episode and hide the physics terminations"
  )
  return {
    "removed_time_out": removed is not None,
    "installed": sorted(env_cfg.terminations),
  }


def _check_reward_target(env_cfg: Any, target_m: float) -> dict[str, float]:
  """Fail unless the task's reward and metric targets already equal ``target_m``.

  The reward reads its target from :data:`TARGET_ENV_VAR` at import time, so the
  variable must be set before the process starts. Measuring against one target
  while the task rewards another would make the cell uninterpretable.

  Args:
    env_cfg: The resolved config.
    target_m: The target passed to :func:`scoring.score_episode`.

  Returns:
    The resolved target per term.

  Raises:
    SystemExit: If any consumer resolves a different target.
  """
  managers = {"rewards": env_cfg.rewards, "metrics": env_cfg.metrics}
  resolved: dict[str, float] = {}
  for group, name in REWARD_TARGET_PATHS:
    params = managers[group][name].params
    resolved[f"{group}.{name}"] = float(params["target_height"])
  for name, value in resolved.items():
    if abs(value - target_m) > 1e-9:
      raise SystemExit(
        f"{name}.target_height resolved to {value}, not {target_m:g}. Export "
        f"{TARGET_ENV_VAR}={target_m:g} before this process starts."
      )
  return resolved


def _sample_at_or_after(substep: int, substeps_per_step: int, steps: int) -> int:
  """Control-step sample index at or just after a physics-substep boundary.

  Sample ``k`` is the state at the end of control step ``k``, i.e. at substep
  ``(k+1) * substeps_per_step``. A contact that starts inside step ``k`` is
  therefore reported by sample ``k``. This is the same sample rule the scoring
  module uses for its takeoff and return-plate reads, so a contact's lateral
  position and its radial return test are read from one sample.
  """
  h = substeps_per_step
  return min(max((substep + h - 1) // h - 1, 0), steps - 1)


def _racket_contact_durations(trace: Any) -> np.ndarray:
  """Duration in seconds of every racket contact segment in a streak."""
  from scoring import contact_segments, flatten_substep_flags

  segments = contact_segments(flatten_substep_flags(trace.racket_substeps))
  if len(segments) == 0:
    return np.zeros(0, dtype=np.float64)
  return (segments[:, 1] - segments[:, 0]) * trace.substep_dt


def _contact_lateral(
  trace: Any, geometry: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Ball position at every racket contact, in the plate frame.

  Args:
    trace: One recorded streak.
    geometry: The modeled bed, for the radial fraction.

  Returns:
    ``(x, y, radial)``, one entry per racket contact segment. ``radial`` is the
    fraction of the plate half-extent, so ``1.0`` is the bed edge.
  """
  from scoring import (
    contact_segments,
    flatten_substep_flags,
    plate_local_positions,
    plate_radial_fraction,
  )

  segments = contact_segments(flatten_substep_flags(trace.racket_substeps))
  local = plate_local_positions(trace)
  indices = np.asarray(
    [
      _sample_at_or_after(int(start), trace.substeps_per_step, trace.steps)
      for start, _ in segments
    ],
    dtype=np.int64,
  )
  x = local[indices, 0]
  y = local[indices, 1]
  return x, y, plate_radial_fraction(x, y, geometry)


def _stats(values: Any) -> dict[str, Any] | None:
  """Count, mean, sample standard deviation and percentiles, or ``None`` if empty.

  The standard deviation uses ``ddof=1`` because the values are a sample of one
  cell's contacts, not a population. ``None`` for ``sd`` means fewer than two
  values.
  """
  array = np.asarray(list(values), dtype=np.float64)
  if array.size == 0:
    return None
  return {
    "count": int(array.size),
    "mean": float(array.mean()),
    "sd": float(array.std(ddof=1)) if array.size > 1 else None,
    "min": float(array.min()),
    "p50": float(np.percentile(array, 50)),
    "p95": float(np.percentile(array, 95)),
    "max": float(array.max()),
  }


def _stack(records: list[dict[str, Any]], key: str) -> list[float]:
  """Concatenate one per-environment list field over the whole cell."""
  return [value for record in records for value in record[key]]


def _plain(value: Any) -> Any:
  """Convert NumPy values into plain Python so ``json`` can write them."""
  if isinstance(value, dict):
    return {str(key): _plain(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_plain(item) for item in value]
  if isinstance(value, np.ndarray):
    return _plain(value.tolist())
  if isinstance(value, np.generic):
    return value.item()
  return value


def _episode_record(
  trace: Any,
  environment: int,
  censored: bool,
  resets: int,
  target_m: float,
  rest_z_m: float,
  geometry: Any,
) -> dict[str, Any]:
  """Score one streak and collect its lateral landing record.

  Args:
    trace: One recorded environment streak.
    environment: Environment index.
    censored: True when that environment never terminated inside the horizon.
    resets: Resets of that environment over the whole horizon.
    target_m: Target apex height above the rest datum.
    rest_z_m: World height of the ball's resting centre on the bed.
    geometry: The modeled bed.

  Returns:
    The per-environment record of both signals plus the lateral landing data.
  """
  from scoring import score_episode

  scored = score_episode(trace, target_m=target_m, rest_z_m=rest_z_m, geometry=geometry)
  x, y, radial = _contact_lateral(trace, geometry)
  fired = sorted(name for name, value in scored["termination_terms"].items() if value)
  return {
    "environment": environment,
    "censored": bool(censored),
    "steps": int(trace.steps),
    "duration_s": float(trace.duration_s),
    "termination_term": fired[0] if fired else None,
    "termination_terms": dict(scored["termination_terms"]),
    "resets": int(resets),
    "qualified_flights": int(scored["qualified_flights"]),
    "longest_qualified_sequence": int(scored["longest_qualified_sequence"]),
    "flight_candidates": int(scored["flight_candidates"]),
    "racket_contact_edges": int(scored["racket_contact_edges"]),
    "hold_count": int(scored["hold_count"]),
    "nonracket_contact_substeps": int(scored["nonracket_contact_substeps"]),
    "rejected_by_criterion": dict(scored["rejected_by_criterion"]),
    "apex_above_rest_m": scored["qualified_apex_above_rest_m"],
    "qualified_flight_durations_s": scored["qualified_flight_durations_s"],
    "qualified_return_radial_fraction": scored[
      "qualified_return_plate_radial_fraction"
    ],
    "lateral_x_m": x.tolist(),
    "lateral_y_m": y.tolist(),
    "radial_fraction": radial.tolist(),
    "racket_contact_durations_s": _racket_contact_durations(trace).tolist(),
  }


def _aggregate(
  records: list[dict[str, Any]], resets_total: int, simulated_s: float
) -> dict[str, Any]:
  """Cell aggregates over both signals and the lateral landing error.

  Args:
    records: One :func:`_episode_record` per recorded environment.
    resets_total: Resets summed over every environment and every step.
    simulated_s: Simulated seconds actually run, the reset-rate denominator.

  Returns:
    The aggregates the README documents. ``None`` means "not measurable in this
    cell", never zero: an empty denominator or a single sample is reported as
    ``None`` rather than as a confidence.
  """
  uncensored = [record for record in records if not record["censored"]]
  censored = [record for record in records if record["censored"]]
  return {
    "resets_total": int(resets_total),
    "resets_per_min": (
      float(resets_total) / (simulated_s / 60.0) if simulated_s > 0.0 else None
    ),
    "resets_per_min_denominator_s": float(simulated_s),
    "recorded_envs": len(records),
    "censored_envs": len(censored),
    "uncensored_envs": len(uncensored),
    "censored_env_indices": [record["environment"] for record in censored],
    "termination_term_counts": {
      name: sum(1 for record in uncensored if record["termination_term"] == name)
      for name in sorted(
        {
          record["termination_term"]
          for record in uncensored
          if record["termination_term"]
        }
      )
    },
    "streak_qualified_flights_uncensored": _stats(
      [record["qualified_flights"] for record in uncensored]
    ),
    "streak_longest_sequence_uncensored": _stats(
      [record["longest_qualified_sequence"] for record in uncensored]
    ),
    "streak_qualified_flights_all_envs": _stats(
      [record["qualified_flights"] for record in records]
    ),
    "streak_longest_sequence_all_envs": _stats(
      [record["longest_qualified_sequence"] for record in records]
    ),
    "apex_above_rest_m": _stats(_stack(records, "apex_above_rest_m")),
    "lateral_x_m": _stats(_stack(records, "lateral_x_m")),
    "lateral_y_m": _stats(_stack(records, "lateral_y_m")),
    "radial_fraction": _stats(_stack(records, "radial_fraction")),
    "racket_contact_durations_s": _stats(_stack(records, "racket_contact_durations_s")),
    "qualified_flight_durations_s": _stats(
      _stack(records, "qualified_flight_durations_s")
    ),
    "qualified_return_radial_fraction": _stats(
      _stack(records, "qualified_return_radial_fraction")
    ),
  }


def run(args: argparse.Namespace) -> dict[str, Any]:
  """Run one latency cell on the policy's closed loop and write ``result.json``."""
  _check_args(args)
  output_dir: Path = args.output_dir
  if output_dir.exists() and any(output_dir.iterdir()):
    raise SystemExit(f"refuse to overwrite a non-empty cell directory: {output_dir}")
  output_dir.mkdir(parents=True, exist_ok=True)

  _ensure_eval_path()
  started = time.monotonic()
  started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

  import torch
  from evaluate_arm import EPISODE_SEED, _load_policy
  from recorder import add_capture_additions, recorder_term
  from scoring import EpisodeTrace, PlateGeometry

  import mjlab.tasks  # noqa: F401  (populates the task registry)
  from mjlab.asset_zoo.robots.agibot_x2.tennis_constants import (
    BALL_RADIUS,
    BED_HALF_LENGTH,
    BED_HALF_WIDTH,
    JUGGLE_ENERGY_BALL_REST_Z,
  )
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg

  env_cfg = load_env_cfg(args.task)
  step_dt = float(env_cfg.sim.mujoco.timestep * env_cfg.decimation)
  latency = _configure_latency(env_cfg, args.latency_min, args.latency_max, step_dt)
  print(
    f"latency override applied: {latency['term']}.{latency['parameter']} = "
    f"{latency['resolved_s']} s "
    f"({latency['quantised_control_steps']} control steps of {step_dt:g} s)",
    flush=True,
  )
  horizon = _configure_horizon(env_cfg, args.horizon_s)
  print(
    f"time_out removed: {horizon['removed_time_out']}; terminations kept: "
    f"{horizon['installed']}",
    flush=True,
  )
  resolved_targets = _check_reward_target(env_cfg, args.target)

  env_cfg.scene.num_envs = args.num_envs
  if env_cfg.scene.terrain is not None:
    # Scene._add_terrain copies scene.num_envs into the terrain config at build
    # time; mirror it so the built scene matches the recorded num_envs.
    env_cfg.scene.terrain.num_envs = args.num_envs
  env_cfg.seed = EPISODE_SEED

  add_capture_additions(env_cfg)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=args.device))
  unwrapped = env.unwrapped

  policy = None
  if args.zero_action:
    print("policy: zero action (default joint pose held by the PD controller)")
  else:
    policy, _ = _load_policy(env, args.device, args.checkpoint, args.task)
    if bool(policy.training):
      raise SystemExit("policy is in training mode; deterministic mean is required")
    print(f"policy: {type(policy).__name__} from {args.checkpoint}")

  # One seed for every cell, applied after the policy load, so the cells differ in
  # the latency condition only and not in their release states.
  env.seed(EPISODE_SEED)
  obs, _ = env.reset()

  recorder = recorder_term(unwrapped)
  num_envs = int(unwrapped.num_envs)
  action_dim = int(unwrapped.action_manager.total_action_dim)
  max_steps = int(unwrapped.max_episode_length)
  zero_actions = torch.zeros((num_envs, action_dim), device=unwrapped.device)
  resets = torch.zeros(num_envs, dtype=torch.long, device=unwrapped.device)

  steps_run = 0
  report_every = max(1, max_steps // 10)
  loop_started = time.monotonic()
  for step in range(max_steps):
    with torch.no_grad():
      actions = zero_actions if policy is None else policy(obs)
    if not bool(torch.isfinite(actions).all()):
      raise SystemExit(f"non-finite action at step {step}")
    obs, _, _, _ = env.step(actions)
    resets += unwrapped.reset_buf.long()
    steps_run += 1
    if (step + 1) % report_every == 0:
      print(
        f"step {step + 1}/{max_steps} resets={int(resets.sum().item())} "
        f"elapsed={time.monotonic() - loop_started:.1f}s",
        flush=True,
      )
  rollout_s = round(time.monotonic() - loop_started, 3)

  raw = recorder.export(max_steps)
  steps_recorded = int(recorder.steps_recorded)
  term_names = list(recorder.termination_terms)
  physics_dt = float(unwrapped.physics_dt)
  env.close()

  finite_fields = ("ball_pos_w", "ball_vel_w", "plate_pos_w", "plate_axes_w")
  bad = [name for name in finite_fields if not bool(np.isfinite(raw[name]).all())]
  if bad:
    raise SystemExit(f"non-finite values in the recorded trace: {bad}")

  geometry = PlateGeometry(
    half_length_m=float(BED_HALF_LENGTH),
    half_width_m=float(BED_HALF_WIDTH),
    ball_radius_m=float(BALL_RADIUS),
  )
  rest_z_m = float(JUGGLE_ENERGY_BALL_REST_Z)
  end_step = raw["end_step"]
  resets_per_env = resets.cpu().numpy()
  if int(steps_recorded) < 1:
    # A cell with no recorded step would otherwise be written out as an empty but
    # successful result, and an all-censored cell would then read as zero censored.
    raise SystemExit(
      "the recorder exported no step for any environment; refusing to write an empty cell"
    )

  records: list[dict[str, Any]] = []
  for index in range(num_envs):
    censored = bool(int(end_step[index]) < 0)
    steps = steps_recorded if censored else int(end_step[index]) + 1
    steps = min(steps, max_steps)
    if steps < 1:
      continue
    trace = EpisodeTrace(
      ball_pos_w=raw["ball_pos_w"][:steps, index],
      ball_vel_w=raw["ball_vel_w"][:steps, index],
      plate_pos_w=raw["plate_pos_w"][:steps, index],
      plate_axes_w=raw["plate_axes_w"][:steps, index],
      racket_substeps=raw["racket_substeps"][:steps, index],
      other_substeps=raw["other_substeps"][:steps, index],
      termination={
        name: bool(raw["term_flags"][index, position])
        for position, name in enumerate(term_names)
      },
      env_origin_w=raw["env_origins"][index],
      step_dt=float(raw["step_dt"]),
      substep_dt=float(raw["substep_dt"]),
    )
    records.append(
      _episode_record(
        trace,
        environment=index,
        censored=censored,
        resets=int(resets_per_env[index]),
        target_m=args.target,
        rest_z_m=rest_z_m,
        geometry=geometry,
      )
    )

  simulated_s = steps_run * step_dt
  resets_total = int(resets_per_env.sum())
  # The recorder counts every reset independently of env.reset_buf. The two counts
  # agree when the pre-step resets (the wrapper's construction reset and the seeded
  # reset) are added back, which is a direct check on signal B.
  recorder_resets = int(raw["reset_count"].sum())
  pre_step_resets = int(raw["pre_step_resets"].sum())

  result: dict[str, Any] = {
    "task": args.task,
    "checkpoint": str(args.checkpoint),
    "policy_kind": "zero_action" if args.zero_action else "checkpoint",
    "device": args.device,
    "latency": latency,
    "horizon_s": float(args.horizon_s),
    "num_envs": num_envs,
    "steps_run": steps_run,
    "steps_recorded": steps_recorded,
    "simulated_s": float(simulated_s),
    "step_dt_s": step_dt,
    "physics_dt_s": physics_dt,
    "episode_seed": EPISODE_SEED,
    "terminations": horizon["installed"],
    "removed_time_out": horizon["removed_time_out"],
    "target_m": float(args.target),
    "resolved_reward_targets_m": resolved_targets,
    "rest_z_m": rest_z_m,
    "plate_geometry": {
      "half_length_m": float(BED_HALF_LENGTH),
      "half_width_m": float(BED_HALF_WIDTH),
      "ball_radius_m": float(BALL_RADIUS),
    },
    "reset_count_check": {
      "reset_buf_total": resets_total,
      "recorder_total": recorder_resets,
      "pre_step_resets_total": pre_step_resets,
      "agree": recorder_resets - pre_step_resets == resets_total,
    },
    "environments": records,
    "aggregates": _aggregate(records, resets_total, simulated_s),
    "rollout_wall_s": rollout_s,
    "wall_time_s": round(time.monotonic() - started, 3),
    "started_utc": started_utc,
    "limits": list(LIMITS),
  }
  (output_dir / "result.json").write_text(json.dumps(_plain(result), indent=2) + "\n")

  summary = {
    "task": args.task,
    "checkpoint": str(args.checkpoint),
    "policy_kind": result["policy_kind"],
    "latency_resolved_s": latency["resolved_s"],
    "latency_quantised_control_steps": latency["quantised_control_steps"],
    "target_m": float(args.target),
    "horizon_s": float(args.horizon_s),
    "num_envs": num_envs,
    "steps_run": steps_run,
    "output": str(output_dir / "result.json"),
    **result["aggregates"],
  }
  print(json.dumps(_plain(summary), indent=2), flush=True)
  return result


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  try:
    run(args)
  except SystemExit:
    raise
  except BaseException:
    traceback.print_exc()
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
