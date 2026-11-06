"""Run one E0 evaluation arm of the X2 nominal juggling study.

One arm is one policy (or one zero-action control) on one target height, run as 32
parallel environments for exactly one 20 s episode each. Exactly 32 scored episodes
per arm and eight arms give the 256 episodes the plan authorises.

**What this arm does not vary.** The environment config comes from the task
registry and is compared field by field against the arm's saved ``params/env.yaml``
before anything is built. Only the environment count, the episode seed, one
read-only contact sensor and one recorder term may differ. Rewards, terminations,
events, physics, actuators, observations and the action space stay the training
config's.

**Target override.** The energy target is read from
``MJLAB_JUGGLE_ENERGY_TARGET_HEIGHT_M`` when the task registry is imported, so the
variable must already be set when the process starts. Every mjlab import therefore
happens inside :func:`run_arm` after that check, and a mismatch between the
variable, the requested target and the three resolved target fields aborts the arm.

**Determinism.** Every arm reseeds the environment with one shared episode seed
*after* the policy is loaded and then resets once. Policy construction draws random
initial weights, so without that reseed the two control arms would start from
different release states than the trained ones. :mod:`summarize_batch` checks the
recorded release states afterwards.

Usage (the batch wrapper sets the target variable for you):

  MJLAB_JUGGLE_ENERGY_TARGET_HEIGHT_M=0.40 \
  uv run python scripts/experiments/x2_tennis_nominal_eval/evaluate_arm.py \
    --arm t040-s7 --target 0.40 \
    --checkpoint /path/to/model_4999.pt \
    --saved-env-yaml /path/to/env.yaml \
    --output-dir logs/diagnostics/tennis_nominal_eval_20261105/arms/t040-s7
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  # The sibling modules are imported by name; keeping this directory first makes
  # the import path the same whether this file runs as a script or with -m.
  sys.path.insert(0, str(_HERE))

TASK_ID = (
  "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
  "-Ball3-Pelvis-Nominal"
)
TARGET_ENV_VAR = "MJLAB_JUGGLE_ENERGY_TARGET_HEIGHT_M"
EXPECTED_ACTOR_DIM = 112
EXPECTED_CRITIC_DIM = 121
EXPECTED_ACTION_DIM = 29
EPISODE_SEED = 20261105
"""The one episode seed shared by all eight arms."""

NUM_ENVS = 32
"""Scored episodes per arm: one episode per environment."""

EPISODE_LENGTH_S = 20.0
"""The plan's per-episode cap. The training config already resolves to this."""

LIMITS = (
  "Terminal rows are captured in record_pre_reset, so no post-reset state is "
  "scored. The ball and plate accessors used there are derived quantities that "
  "env.step() refreshes only after the reset block, so they lag the integrated "
  "state by one 1 ms substep, the same lag the termination decision sees.",
  "Ball position and velocity are sampled once per control step (20 ms). A "
  "ballistic apex between samples is under-read by at most g*dt^2/8 = 0.5 mm, and "
  "the takeoff reference can be up to one control step late, which makes the "
  "measured rise a conservative lower bound.",
  "Racket contact and any-contact flags are per 1 ms physics substep. Non-racket "
  "contact is 'any contact and not racket contact' per substep: a ball touching the "
  "racket and something else in the same substep is attributed to the racket.",
  "32 episodes per arm are reset samples of one policy, not 32 independent "
  "policies. The episode count is a screening budget, not a reliability estimate.",
  "Zero action means a zero position-policy action offset, i.e. the default joint "
  "pose held by the PD controller. It is not zero torque.",
  "This is a simulation measurement. It says nothing about hardware behaviour, and "
  "the qualified-flight thresholds are the plan's provisional values.",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--arm", required=True, help="Arm name, e.g. t040-s7.")
  parser.add_argument(
    "--task",
    default=TASK_ID,
    help=(
      "Task id to evaluate. Defaults to the nominal pilot. A staged rung must pass "
      "its own task id, because the saved env.yaml is validated against the loaded "
      "config and a rung changes the ball-history condition on purpose."
    ),
  )
  parser.add_argument(
    "--target", type=float, required=True, help="Target apex height, metres."
  )
  parser.add_argument(
    "--checkpoint",
    type=Path,
    help="Final policy checkpoint (model_4999.pt). Omit for a zero-action control.",
  )
  parser.add_argument(
    "--zero-action",
    action="store_true",
    help="Hold the default joint pose instead of loading a policy.",
  )
  parser.add_argument(
    "--saved-env-yaml",
    type=Path,
    required=True,
    help="The arm's saved training params/env.yaml, used for config validation.",
  )
  parser.add_argument(
    "--output-dir", type=Path, required=True, help="Arm output directory."
  )
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument(
    "--num-envs",
    type=int,
    default=NUM_ENVS,
    help="Scored episodes. Only the CPU pipeline self-check may lower it.",
  )
  parser.add_argument(
    "--episode-length-s",
    type=float,
    default=EPISODE_LENGTH_S,
    help="Episode cap. Only the CPU pipeline self-check may lower it; never raise it.",
  )
  parser.add_argument(
    "--cpu-self-check",
    action="store_true",
    help="Run the whole pipeline on CPU with a shortened episode to verify the "
    "artifact path. Requires --zero-action --device cpu. Not a measurement.",
  )
  parser.add_argument(
    "--expect-source-root",
    default="",
    help="Directory the imported mjlab package must live under. Defaults to the "
    "first PYTHONPATH entry that contains an mjlab package.",
  )
  return parser.parse_args(argv)


def _expected_source_root(explicit: str) -> Path | None:
  """Resolve the directory the imported ``mjlab`` package must be under."""
  if explicit:
    return Path(explicit).resolve()
  for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    if entry and (Path(entry) / "mjlab" / "__init__.py").is_file():
      return Path(entry).resolve()
  return None


def _check_arm_scope(args: argparse.Namespace) -> None:
  """Refuse any arm scope other than the plan's, unless it is the CPU self-check.

  The self-check exists to exercise the artifact path on a machine without a GPU.
  It is a shortened CPU rollout with zero action, and its ``arm_metadata.json``
  carries ``self_check: true`` so no reader can mistake it for a scored arm.
  """
  if args.cpu_self_check:
    if not args.zero_action or args.device != "cpu":
      raise SystemExit("--cpu-self-check requires --zero-action and --device cpu")
    if not 0.0 < args.episode_length_s <= EPISODE_LENGTH_S:
      raise SystemExit(
        f"--episode-length-s must be in (0, {EPISODE_LENGTH_S}], got "
        f"{args.episode_length_s}"
      )
    if not 0 < args.num_envs <= NUM_ENVS:
      raise SystemExit(f"--num-envs must be in (0, {NUM_ENVS}], got {args.num_envs}")
    return
  if args.num_envs != NUM_ENVS:
    raise SystemExit(
      f"the plan fixes {NUM_ENVS} episodes per arm, got --num-envs {args.num_envs}"
    )
  if abs(args.episode_length_s - EPISODE_LENGTH_S) > 1e-9:
    raise SystemExit(
      f"the plan fixes a {EPISODE_LENGTH_S} s episode cap, got "
      f"--episode-length-s {args.episode_length_s}"
    )


def _check_target_environment(target: float) -> None:
  """Fail unless the process environment carries exactly this arm's target.

  Only the energy target may be exported. The other ``MJLAB_JUGGLE_*`` variables
  configure perception and reward ablations; a stray one would silently change the
  task the checkpoint was trained on.
  """
  raw = os.environ.get(TARGET_ENV_VAR, "")
  if raw == "":
    raise SystemExit(
      f"{TARGET_ENV_VAR} must be set to {target:g} before this process starts: the "
      "task registry reads it at import time."
    )
  if abs(float(raw) - target) > 1e-9:
    raise SystemExit(f"{TARGET_ENV_VAR}={raw} does not match --target {target:g}")
  stray = sorted(
    name for name in os.environ if name.startswith("MJLAB_") and name != TARGET_ENV_VAR
  )
  if stray:
    raise SystemExit(f"unexpected MJLAB_* overrides in the environment: {stray}")


def _verify_checkpoint_dims(checkpoint: Path) -> dict[str, Any]:
  """Return the actor and critic widths held in a checkpoint."""
  import torch

  payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
  actor = payload["actor_state_dict"]
  critic = payload.get("critic_state_dict") or {}
  actor_in = int(actor["mlp.0.weight"].shape[1])
  weight_layers = sorted(
    (name for name in actor if name.endswith(".weight")),
    key=lambda name: int(name.split(".")[1]),
  )
  actor_out = int(actor[weight_layers[-1]].shape[0])
  critic_in = int(critic["mlp.0.weight"].shape[1]) if critic else -1
  return {
    "file": str(checkpoint),
    "size_bytes": checkpoint.stat().st_size,
    "iteration": payload.get("iter"),
    "actor_input_dim": actor_in,
    "actor_output_dim": actor_out,
    "critic_input_dim": critic_in,
    "has_observation_normalizer": "obs_normalizer._mean" in actor,
  }


def _normalizer_snapshot(policy: Any) -> dict[str, Any]:
  """Copy normalizer values for direct comparison; no hashes or checksums."""
  normalizer = getattr(policy, "obs_normalizer", None)
  if normalizer is None:
    return {"kind": "absent"}
  snapshot: dict[str, Any] = {"kind": type(normalizer).__name__}
  count = getattr(normalizer, "count", None)
  if count is not None:
    snapshot["count"] = int(count.item())
  for name in ("_mean", "_var"):
    value = getattr(normalizer, name, None)
    if value is not None:
      snapshot[name] = value.detach().cpu().tolist()
  snapshot["training"] = bool(getattr(normalizer, "training", True))
  return snapshot


def _obs_width(obs: Any, group: str) -> int:
  """Width of one observation group in a batched observation dict."""
  return int(obs[group].shape[-1])


def _config_report(
  env_cfg: Any, saved_yaml: Path, extra_allowed: dict[str, str] | None = None
) -> dict[str, Any]:
  """Validate the runtime config against the saved training config.

  Args:
    env_cfg: The resolved runtime config, before the capture additions.
    saved_yaml: The run's own ``params/env.yaml``.
    extra_allowed: Additional allowed differences. Only the CPU self-check passes
      one, so its shortened episode is reported as the self-check's own deviation
      instead of silently passing.

  Returns:
    The comparison report from :func:`config_check.compare_env_config`.
  """
  from dataclasses import asdict

  from config_check import ALLOWED_RUNTIME_DIFFERENCES, compare_env_config

  from mjlab.utils.os import load_saved_yaml

  allowed = dict(ALLOWED_RUNTIME_DIFFERENCES)
  allowed.update(extra_allowed or {})
  return compare_env_config(
    asdict(env_cfg), load_saved_yaml(saved_yaml), allowed=allowed
  )


def _load_policy(
  env: Any, device: str, checkpoint: Path, task: str = TASK_ID
) -> tuple[Any, Any]:
  """Load the checkpoint's actor and return its deterministic inference policy."""
  from dataclasses import asdict

  import torch

  from mjlab.rl import MjlabOnPolicyRunner
  from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

  torch.set_grad_enabled(False)
  agent_cfg = load_rl_cfg(task)
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)
  policy.eval()
  return policy, agent_cfg


def _episode_trace(raw: dict[str, Any], env_index: int, end_step: int, term_names):
  """Slice one environment's first episode out of the arm-wide recorder buffers."""
  from scoring import EpisodeTrace

  steps = end_step + 1
  termination = {
    name: bool(raw["term_flags"][env_index, index])
    for index, name in enumerate(term_names)
  }
  return EpisodeTrace(
    ball_pos_w=raw["ball_pos_w"][:steps, env_index],
    ball_vel_w=raw["ball_vel_w"][:steps, env_index],
    plate_pos_w=raw["plate_pos_w"][:steps, env_index],
    plate_axes_w=raw["plate_axes_w"][:steps, env_index],
    racket_substeps=raw["racket_substeps"][:steps, env_index],
    other_substeps=raw["other_substeps"][:steps, env_index],
    termination=termination,
    env_origin_w=raw["env_origins"][env_index],
  )


def _csv_row(episode: dict[str, Any]) -> dict[str, Any]:
  """Flatten one scored episode into CSV columns."""
  row: dict[str, Any] = {
    key: value
    for key, value in episode.items()
    if isinstance(value, (int, float, bool, str))
  }
  for name, fired in episode["termination_terms"].items():
    row[f"termination_{name}"] = bool(fired)
  row["qualified_flight_durations_s"] = ";".join(
    f"{value:.4f}" for value in episode["qualified_flight_durations_s"]
  )
  row["qualified_apex_above_rest_m"] = ";".join(
    f"{value:.4f}" for value in episode["qualified_apex_above_rest_m"]
  )
  return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  if not rows:
    path.write_text("\n")
    return
  fieldnames: list[str] = []
  for row in rows:
    for key in row:
      if key not in fieldnames:
        fieldnames.append(key)
  with path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def _write_plot(
  output_dir: Path,
  raw: dict[str, Any],
  end_steps: list[int],
  episodes: list[dict[str, Any]],
  rest_z_m: float,
  arm: str,
) -> str:
  """Write one small figure: ball height traces and qualified-flight counts."""
  try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
  except Exception as error:  # pragma: no cover - optional dependency path
    return f"plot skipped: {error}"

  figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 3.6), dpi=130)
  step_dt = float(raw["step_dt"])
  for env_index in range(min(4, len(end_steps))):
    steps = end_steps[env_index] + 1
    time_s = (np.arange(steps) + 1) * step_dt
    height = (
      raw["ball_pos_w"][:steps, env_index, 2]
      - raw["env_origins"][env_index, 2]
      - rest_z_m
    )
    left.plot(time_s, height, linewidth=0.9, label=f"env {env_index}")
    contact = raw["racket_substeps"][:steps, env_index].any(axis=1)
    left.plot(
      time_s[contact],
      height[contact],
      linestyle="none",
      marker=".",
      markersize=2.5,
      color="black",
    )
  left.set_xlabel("time since reset (s)")
  left.set_ylabel("ball height above rest datum (m)")
  left.set_title("ball height, black dots = racket contact")
  left.grid(alpha=0.25)
  left.legend(fontsize=6, loc="upper right")

  counts = [episode["longest_qualified_sequence"] for episode in episodes]
  right.bar(range(len(counts)), counts)
  right.set_xlabel("environment (scored episode)")
  right.set_ylabel("longest qualified sequence")
  right.set_title(f"{arm}: longest qualified flights per episode")
  right.grid(alpha=0.25, axis="y")
  figure.tight_layout()
  figure.savefig(output_dir / "trajectories.png")
  plt.close(figure)
  return "trajectories.png"


def run_arm(args: argparse.Namespace) -> dict[str, Any]:
  """Run one arm, write its artifacts, and return its metadata record."""
  started = time.monotonic()
  output_dir: Path = args.output_dir
  if output_dir.exists() and any(output_dir.iterdir()):
    raise SystemExit(f"refuse to overwrite a non-empty arm directory: {output_dir}")
  output_dir.mkdir(parents=True, exist_ok=True)

  metadata: dict[str, Any] = {
    "arm": args.arm,
    "target_m": args.target,
    "task": args.task,
    "mode": "zero_action" if args.zero_action else "policy",
    "self_check": bool(args.cpu_self_check),
    "checkpoint": str(args.checkpoint) if args.checkpoint else None,
    "saved_env_yaml": str(args.saved_env_yaml),
    "episode_seed": EPISODE_SEED,
    "num_envs": args.num_envs,
    "episode_length_s": args.episode_length_s,
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "limits": list(LIMITS),
    "status": "running",
  }

  def write_metadata() -> None:
    metadata["elapsed_s"] = round(time.monotonic() - started, 3)
    (output_dir / "arm_metadata.json").write_text(
      json.dumps(metadata, indent=2, default=str) + "\n"
    )

  try:
    _check_arm_scope(args)
    _check_target_environment(args.target)
    if not args.zero_action and args.checkpoint is None:
      raise SystemExit("a policy arm needs --checkpoint")
    if args.zero_action and args.checkpoint is not None:
      raise SystemExit("a zero-action control must not load a checkpoint")

    import numpy as np
    import torch

    import mjlab
    from mjlab.asset_zoo.robots.agibot_x2.tennis_constants import (
      BALL_RADIUS,
      BED_HALF_LENGTH,
      BED_HALF_WIDTH,
      JUGGLE_ENERGY_BALL_REST_Z,
    )

    expected_root = _expected_source_root(args.expect_source_root)
    source_path = Path(mjlab.__file__).resolve()
    metadata["source"] = {
      "mjlab_file": str(source_path),
      "expected_root": str(expected_root) if expected_root else None,
      "pythonpath": os.environ.get("PYTHONPATH", ""),
    }
    if expected_root is not None and expected_root not in source_path.parents:
      raise SystemExit(
        f"mjlab resolved to {source_path}, which is not under {expected_root}"
      )

    import mjlab.tasks  # noqa: F401  (populates the task registry)
    from mjlab.tasks.registry import load_env_cfg

    env_cfg = load_env_cfg(args.task)
    if args.cpu_self_check:
      env_cfg.episode_length_s = args.episode_length_s
    elif abs(float(env_cfg.episode_length_s) - EPISODE_LENGTH_S) > 1e-9:
      raise SystemExit(
        f"the task's episode length is {env_cfg.episode_length_s}, not the plan's "
        f"{EPISODE_LENGTH_S}"
      )
    env_cfg.scene.num_envs = args.num_envs
    if env_cfg.scene.terrain is not None:
      # Scene._add_terrain copies scene.num_envs into the terrain config at build
      # time; mirror it here so the validation report shows the size the
      # environment will actually use.
      env_cfg.scene.terrain.num_envs = args.num_envs
    env_cfg.seed = EPISODE_SEED

    targets = {
      "rewards.ball_energy": env_cfg.rewards["ball_energy"].params["target_height"],
      "metrics.ball_energy": env_cfg.metrics["ball_energy"].params["target_height"],
      "metrics.ball_energy_error": env_cfg.metrics["ball_energy_error"].params[
        "target_height"
      ],
    }
    metadata["resolved_targets"] = {
      name: float(value) for name, value in targets.items()
    }
    for name, value in targets.items():
      if abs(float(value) - args.target) > 1e-9:
        raise SystemExit(
          f"{name}.target_height resolved to {value}, not {args.target:g}"
        )

    extra_allowed = (
      {"episode_length_s": "CPU self-check shortens the episode; not a scored arm"}
      if args.cpu_self_check
      else None
    )
    report = _config_report(env_cfg, args.saved_env_yaml, extra_allowed)
    metadata["config_check"] = report
    if not report["ok"]:
      raise SystemExit(
        "runtime config differs from the saved training config: "
        + json.dumps(report["unexpected_differences"][:5], indent=2)
      )

    if args.checkpoint is not None:
      checkpoint_dims = _verify_checkpoint_dims(args.checkpoint)
      metadata["checkpoint_dims"] = checkpoint_dims
      if not checkpoint_dims["has_observation_normalizer"]:
        raise SystemExit("checkpoint is missing the trained observation normalizer")
      expected = {
        "actor_input_dim": EXPECTED_ACTOR_DIM,
        "actor_output_dim": EXPECTED_ACTION_DIM,
        "critic_input_dim": EXPECTED_CRITIC_DIM,
      }
      for name, value in expected.items():
        if checkpoint_dims[name] != value:
          raise SystemExit(
            f"checkpoint {name} is {checkpoint_dims[name]}, expected {value}"
          )

    from recorder import add_capture_additions, recorder_term

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    add_capture_additions(env_cfg)
    metadata["installed_capture"] = {
      "scene_sensors": [sensor.name for sensor in env_cfg.scene.sensors],
      "recorders": sorted(env_cfg.recorders),
    }
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=args.device))
    unwrapped = env.unwrapped

    policy = None
    normalizer_before = None
    if args.zero_action:
      metadata["policy"] = {"kind": "zero_action"}
    else:
      policy, agent_cfg = _load_policy(env, args.device, args.checkpoint, args.task)
      normalizer_before = _normalizer_snapshot(policy)
      metadata["policy"] = {
        "kind": "checkpoint",
        "class": type(policy).__name__,
        "training_mode": bool(policy.training),
        "clip_actions": agent_cfg.clip_actions,
        "normalizer_before": normalizer_before,
      }
      if agent_cfg.clip_actions is not None:
        raise SystemExit("E0 requires the saved unclipped action contract")
      if bool(policy.training):
        raise SystemExit("policy is in training mode; deterministic mean is required")

    # The final, authoritative reset. Seeding happens after the policy load so the
    # trained and zero-action arms sample identical release states.
    env.seed(EPISODE_SEED)
    obs, _ = env.reset()

    action_dim = int(unwrapped.action_manager.total_action_dim)
    metadata["runtime_dims"] = {
      "actor_obs": _obs_width(obs, "actor"),
      "critic_obs": _obs_width(obs, "critic"),
      "actions": action_dim,
      "num_envs": int(unwrapped.num_envs),
      "max_episode_length": int(unwrapped.max_episode_length),
      "step_dt": float(unwrapped.step_dt),
      "physics_dt": float(unwrapped.physics_dt),
    }
    resolved = (
      metadata["runtime_dims"]["actor_obs"],
      metadata["runtime_dims"]["critic_obs"],
      action_dim,
    )
    if resolved != (EXPECTED_ACTOR_DIM, EXPECTED_CRITIC_DIM, EXPECTED_ACTION_DIM):
      raise SystemExit(
        "resolved dimensions differ from the trained contract: "
        f"actor {resolved[0]}, critic {resolved[1]}, actions {resolved[2]}"
      )

    recorder = recorder_term(unwrapped)
    max_steps = int(unwrapped.max_episode_length)
    zero_actions = torch.zeros(
      (int(unwrapped.num_envs), action_dim), device=unwrapped.device
    )

    nonfinite_actions = 0
    nonfinite_observations = 0
    loop_started = time.monotonic()
    for _ in range(max_steps):
      with torch.no_grad():
        actions = zero_actions if policy is None else policy(obs)
      if not bool(torch.isfinite(actions).all()):
        nonfinite_actions += 1
      for group in ("actor", "critic"):
        if not bool(torch.isfinite(obs[group]).all()):
          nonfinite_observations += 1
      if nonfinite_actions or nonfinite_observations:
        metadata["nonfinite_action_steps"] = nonfinite_actions
        metadata["nonfinite_observation_steps"] = nonfinite_observations
        np.savez_compressed(output_dir / "trace_raw.npz", **recorder.export(max_steps))
        raise SystemExit("non-finite action/observation; partial trace preserved")
      obs, _, _, _ = env.step(actions)
      if bool((recorder.ended >= 0).all()):
        # Every scored first episode has ended. Later steps would only advance
        # auto-reset episodes that are not scored, so stop here.
        break
    metadata["rollout_s"] = round(time.monotonic() - loop_started, 3)
    metadata["nonfinite_action_steps"] = nonfinite_actions
    metadata["nonfinite_observation_steps"] = nonfinite_observations

    if policy is not None:
      normalizer_after = _normalizer_snapshot(policy)
      metadata["policy"]["normalizer_after"] = normalizer_after
      if normalizer_after != normalizer_before:
        raise SystemExit("the observation normalizer changed during the rollout")

    raw = recorder.export(max_steps)

    end_step = raw["end_step"]
    if int(end_step.min()) < 0:
      unfinished = int((end_step < 0).sum())
      raise SystemExit(
        f"{unfinished} environments had no completed first episode within "
        f"{max_steps} steps; the 20 s cap did not fire"
      )
    np.savez_compressed(output_dir / "trace_raw.npz", **raw)

    finiteness = {
      name: bool(np.isfinite(raw[name]).all())
      for name in ("ball_pos_w", "ball_vel_w", "plate_pos_w", "plate_axes_w")
    }
    metadata["finite_traces"] = finiteness
    if not all(finiteness.values()):
      raise SystemExit("non-finite values in the recorded trace; trace preserved")

    from scoring import (
      PLAN_THRESHOLDS,
      PlateGeometry,
      score_episode,
      summarize_arm,
    )

    bed = PlateGeometry(
      half_length_m=float(BED_HALF_LENGTH),
      half_width_m=float(BED_HALF_WIDTH),
      ball_radius_m=float(BALL_RADIUS),
    )
    rest_z_m = float(JUGGLE_ENERGY_BALL_REST_Z)
    term_names = recorder.termination_terms
    episodes: list[dict[str, Any]] = []
    for env_index in range(int(unwrapped.num_envs)):
      trace = _episode_trace(raw, env_index, int(end_step[env_index]), term_names)
      scored = score_episode(
        trace, target_m=args.target, rest_z_m=rest_z_m, geometry=bed
      )
      scored["environment"] = env_index
      episodes.append(scored)

    thresholds = PLAN_THRESHOLDS
    summary = summarize_arm(episodes, target_m=args.target, thresholds=thresholds)
    summary["thresholds"] = {
      "min_airborne_s": thresholds.min_airborne_s,
      "max_contact_s": thresholds.max_contact_s,
      "apex_target_tolerance_m": thresholds.apex_target_tolerance_m,
      "rise_fraction_of_target": thresholds.rise_fraction_of_target,
      "min_sequence_length": thresholds.min_sequence_length,
      "min_passing_episodes": thresholds.min_passing_episodes,
      "episodes_per_arm": thresholds.episodes_per_arm,
    }

    (output_dir / "episodes.json").write_text(
      json.dumps(episodes, indent=2, default=str) + "\n"
    )
    _write_csv(output_dir / "episodes.csv", [_csv_row(row) for row in episodes])
    (output_dir / "arm_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    metadata["plot"] = _write_plot(
      output_dir,
      raw,
      [int(value) for value in end_step],
      episodes,
      rest_z_m,
      args.arm,
    )
    metadata["geometry"] = {
      "rest_z_m": rest_z_m,
      "bed_half_length_m": float(BED_HALF_LENGTH),
      "bed_half_width_m": float(BED_HALF_WIDTH),
      "ball_radius_m": float(BALL_RADIUS),
      "env_origin_z": [float(value) for value in raw["env_origins"][:, 2]],
    }
    metadata["termination_terms"] = term_names
    metadata["initial_state"] = {
      "ball_pos_w": raw["initial_ball_pos_w"].tolist(),
      "ball_vel_w": raw["initial_ball_vel_w"].tolist(),
      "pre_step_resets_min": int(raw["pre_step_resets"].min()),
      "pre_step_resets_max": int(raw["pre_step_resets"].max()),
      "reset_count_min": int(raw["reset_count"].min()),
      "reset_count_max": int(raw["reset_count"].max()),
      "steps_recorded": int(recorder.steps_recorded),
    }
    if not bool((raw["pre_step_resets"] == 2).all()):
      raise SystemExit(
        "expected exactly two pre-step resets per environment (the wrapper's "
        f"construction reset and the seeded reset), got "
        f"{int(raw['pre_step_resets'].min())}"
      )
    metadata["gate_passed"] = bool(summary["gate_passed"])
    metadata["status"] = "ok"
    env.close()
    write_metadata()
    return metadata
  except BaseException as error:
    metadata["status"] = "failed"
    metadata["error"] = f"{type(error).__name__}: {error}"
    metadata["traceback"] = traceback.format_exc()
    write_metadata()
    raise


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  try:
    metadata = run_arm(args)
  except SystemExit:
    raise
  except BaseException:
    traceback.print_exc()
    return 1
  print(
    json.dumps(
      {
        "arm": metadata["arm"],
        "status": metadata["status"],
        "gate_passed": metadata.get("gate_passed"),
        "elapsed_s": metadata.get("elapsed_s"),
      }
    ),
    flush=True,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
