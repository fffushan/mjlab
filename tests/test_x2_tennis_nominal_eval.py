"""Focused synthetic tests for the X2 nominal-evaluation scoring rules.

The scored quantity is a qualified racket-to-racket flight, so the rules that are
easy to get silently wrong are tested here on hand-built traces with no simulator:

1. A real ballistic sequence is counted, and the arm gate passes on it.
2. The initial release fall is not a flight, because no racket contact precedes it.
3. Contact chatter and a sustained hold are rejected for the right reasons.
4. A contact that spans a control-step boundary is one contact, not two.
5. A flight that ends with the episode is not counted, and the return test uses the
   recorded plate frame rather than world axes.

No GPU, no policy and no training run is involved. The traces are synthetic ball
states, so a passing test says the scoring rule matches the plan's definition. It
says nothing about what any policy does.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Any

import numpy as np
import pytest

SCORING_PATH = (
  pathlib.Path(__file__).resolve().parents[1]
  / "scripts"
  / "experiments"
  / "x2_tennis_nominal_eval"
  / "scoring.py"
)

STEP_DT = 0.02
SUBSTEPS = 20
GRAVITY = 9.81
REST_Z = 1.1530848027869869
BALL_RADIUS = 0.0327
TARGET_M = 0.40


_NAME = "nominal_scoring_under_test"
_CONFIG_NAME = "nominal_config_check_under_test"
_RECORDER_NAME = "nominal_recorder_under_test"
_MODULE_DIR = SCORING_PATH.parent


def _module():
  spec = importlib.util.spec_from_file_location(_NAME, SCORING_PATH)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  # Registered before execution so the module's dataclasses can resolve their own
  # string annotations, which they look up through ``sys.modules``.
  sys.modules[_NAME] = module
  spec.loader.exec_module(module)
  return module


def _load(name: str, path: pathlib.Path):
  spec = importlib.util.spec_from_file_location(name, path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


def _config_module():
  return _load(_CONFIG_NAME, _MODULE_DIR / "config_check.py")


def _recorder_module():
  if str(_MODULE_DIR) not in sys.path:
    # ``recorder`` imports its sibling ``config_check`` by module name.
    sys.path.insert(0, str(_MODULE_DIR))
  return _load(_RECORDER_NAME, _MODULE_DIR / "recorder.py")


def _rotation_z(angle_rad: float) -> np.ndarray:
  """Plate-to-world rotation for a plate yawed about the world z axis."""
  cos, sin = np.cos(angle_rad), np.sin(angle_rad)
  x_axis = np.array([cos, sin, 0.0])
  y_axis = np.array([-sin, cos, 0.0])
  normal = np.array([0.0, 0.0, 1.0])
  return np.stack([x_axis, y_axis, normal], axis=1)


def _build_trace(
  *,
  steps: int = 1000,
  apex_m: float = 0.41,
  contact_windows: list[tuple[int, int]] | None = None,
  lateral_world: np.ndarray | None = None,
  plate_yaw_rad: float = 0.0,
  termination: dict[str, bool] | None = None,
):
  """Build a trace from explicit racket-contact windows, in substeps.

  Between two windows the ball follows a ballistic arc that starts at rest height
  with the upward speed needed to reach ``apex_m`` above rest. Inside a window the
  ball rests on the plate. The contact list defaults to a repeated bounce pattern
  whose flight matches ``apex_m``.
  """
  module = _module()
  n_substeps = steps * SUBSTEPS
  if contact_windows is None:
    speed = float(np.sqrt(2 * GRAVITY * apex_m))
    flight_substeps = int(round(2 * speed / GRAVITY / 0.001))
    contact_substeps = 8
    period = flight_substeps + contact_substeps
    contact_windows = []
    start = 0
    while start + contact_substeps <= n_substeps:
      contact_windows.append((start, start + contact_substeps))
      start += period

  # One sample per millisecond boundary: index j is the state at time j ms.
  times = np.arange(n_substeps + 1, dtype=np.float64) * 0.001
  z = np.full(times.shape, REST_Z)
  vz = np.zeros(times.shape)
  speed = float(np.sqrt(2 * GRAVITY * apex_m))
  for index, (_, end) in enumerate(contact_windows):
    next_start = (
      contact_windows[index + 1][0]
      if index + 1 < len(contact_windows)
      else n_substeps + 1
    )
    window = slice(end, next_start)
    elapsed = times[window] - times[end]
    z[window] = REST_Z + speed * elapsed - 0.5 * GRAVITY * elapsed**2
    vz[window] = speed - GRAVITY * elapsed

  flags = np.zeros(n_substeps, dtype=bool)
  for start, end in contact_windows:
    flags[start:end] = True

  samples = np.arange(steps, dtype=np.int64)
  sample_times_ms = (samples + 1) * SUBSTEPS
  ball_pos = np.zeros((steps, 3))
  ball_pos[:, 2] = z[sample_times_ms]
  ball_vel = np.zeros((steps, 3))
  ball_vel[:, 2] = vz[sample_times_ms]
  if lateral_world is not None:
    ball_pos[:, :2] = np.asarray(lateral_world, dtype=np.float64)

  plate_pos = np.tile(np.array([0.0, 0.0, REST_Z - BALL_RADIUS]), (steps, 1))
  plate_axes = np.tile(_rotation_z(plate_yaw_rad), (steps, 1, 1))
  trace = module.EpisodeTrace(
    ball_pos_w=ball_pos + (lateral_world is not None) * 0.0,
    ball_vel_w=ball_vel,
    plate_pos_w=plate_pos,
    plate_axes_w=plate_axes,
    racket_substeps=flags.reshape(steps, SUBSTEPS),
    other_substeps=np.zeros((steps, SUBSTEPS), dtype=bool),
    termination=dict(termination or {"time_out": True}),
    env_origin_w=np.zeros(3),
  )
  return module, trace, contact_windows


def _score(trace, *, target_m: float = TARGET_M):
  module = _module()
  return module.score_episode(
    trace,
    target_m=target_m,
    rest_z_m=REST_Z,
    geometry=module.PlateGeometry(ball_radius_m=BALL_RADIUS),
  )


def test_ballistic_sequence_is_qualified_and_passes_the_gate() -> None:
  module, trace, windows = _build_trace()
  episode = _score(trace)

  assert episode["racket_contact_edges"] == len(windows)
  assert episode["hold_count"] == 0
  assert episode["qualified_flights"] == len(windows) - 1
  assert episode["longest_qualified_sequence"] == len(windows) - 1
  assert episode["rejected_by_criterion"] == dict.fromkeys(module.CRITERION_ORDER, 0)
  assert episode["episode_gate_pass"]

  apex = episode["qualified_apex_above_rest_m"]
  assert min(apex) == pytest.approx(0.41, abs=0.01)
  assert min(episode["qualified_takeoff_rise_m"]) > 0.5 * TARGET_M
  assert max(episode["qualified_return_plate_radial_fraction"]) < 1.0

  summary = module.summarize_arm(
    [episode] * 32,
    target_m=TARGET_M,
  )
  assert summary["episodes"] == 32
  assert summary["gate_episode_count_matches_plan"]
  assert summary["gate_passed"]
  assert summary["losses"] == 0
  assert summary["falls"] == 0

  # One episode is not enough evidence for a 26/32 gate.
  short = module.summarize_arm([episode] * 26, target_m=TARGET_M)
  assert not short["gate_episode_count_matches_plan"]
  assert not short["gate_passed"]


def test_incomplete_short_return_contact_is_not_qualified() -> None:
  _, trace, _ = _build_trace(steps=30, contact_windows=[(0, 8), (590, 600)])
  result = _score(trace)
  assert result["qualified_flights"] == 0
  assert result["rejected_by_criterion"]["1_incomplete_return_contact"] == 1


def test_matched_reset_checks_plate_and_origins() -> None:
  module = _module()
  arrays = {
    "initial_ball_pos_w": np.zeros((32, 3)),
    "initial_ball_vel_w": np.zeros((32, 3)),
    "initial_plate_pos_w": np.zeros((32, 3)),
    "initial_plate_axes_w": np.tile(np.eye(3), (32, 1, 1)),
    "env_origins": np.zeros((32, 3)),
  }
  other = {key: value.copy() for key, value in arrays.items()}
  assert module.compare_matched_initial_states({"a": arrays, "b": other})["matched"]
  other["initial_plate_pos_w"][0, 2] = 0.01
  assert not module.compare_matched_initial_states({"a": arrays, "b": other})["matched"]


def test_normalizer_comparison_preserves_all_values() -> None:
  from types import SimpleNamespace

  import torch

  evaluator = _load("nominal_eval_under_test", _MODULE_DIR / "evaluate_arm.py")
  normalizer = SimpleNamespace(
    count=torch.tensor(7),
    _mean=torch.tensor([[1.0, 2.0]]),
    _var=torch.tensor([[3.0, 4.0]]),
    training=False,
  )
  policy = SimpleNamespace(obs_normalizer=normalizer)
  before = evaluator._normalizer_snapshot(policy)
  normalizer._mean = torch.tensor([[2.0, 1.0]])  # equal sum, different values
  assert evaluator._normalizer_snapshot(policy) != before


def test_duplicate_refusal_preserves_existing_output(tmp_path) -> None:
  import os
  import subprocess

  stage = tmp_path / "stage"
  out = stage / "out"
  out.mkdir(parents=True)
  (out / "exit-status").write_text("0\n")
  (out / "progress.jsonl").write_text("original\n")
  jobs = tmp_path / "jobs"
  exp = "agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal"
  for target in ("t020", "t040"):
    for seed in (7, 17, 27):
      arm = f"{target}-s{seed}"
      stamp = "2026-11-04_23-35-39" if seed == 27 else "2026-11-04_23-35-40"
      run = jobs / arm / "run" / exp / f"{stamp}_{arm}"
      (run / "params").mkdir(parents=True)
      (run / "model_4999.pt").write_text("stub")
      (run / "params/env.yaml").write_text("stub")
  env = dict(
    os.environ,
    STAGE=str(stage),
    OUT=str(out),
    JOBS=str(jobs),
    CODE=str(_MODULE_DIR),
    SRC=str(SCORING_PATH.parents[3] / "src"),
    PY=sys.executable,
  )
  result = subprocess.run(
    ["bash", str(_MODULE_DIR / "run_batch.sh")],
    env=env,
    capture_output=True,
    timeout=10,
  )
  assert result.returncode == 12, result.stderr.decode()
  assert (out / "exit-status").read_text() == "0\n"
  assert (out / "progress.jsonl").read_text() == "original\n"


def test_initial_release_fall_is_not_a_flight() -> None:
  # The ball is released at t=0, falls, lands once at 600 ms, and nothing follows.
  _, trace, _ = _build_trace(contact_windows=[(600, 608)])
  episode = _score(trace)

  assert episode["racket_contact_edges"] == 1
  assert episode["flight_candidates"] == 0
  assert episode["qualified_flights"] == 0
  assert episode["longest_qualified_sequence"] == 0
  assert not episode["episode_gate_pass"]
  # No termination term fired, so this episode also cannot pass the gate on a
  # technicality such as an unrecorded loss.
  assert not episode["loss"] and not episode["fall"]


def test_chatter_is_rejected_as_a_short_flight() -> None:
  # Two contacts 60 ms apart: a 60 ms airborne gap, well under the 150 ms floor.
  _, trace, _ = _build_trace(
    apex_m=0.02, contact_windows=[(0, 8), (68, 76), (136, 144)]
  )
  episode = _score(trace)

  assert episode["racket_contact_edges"] == 3
  assert episode["flight_candidates"] == 2
  assert episode["qualified_flights"] == 0
  assert episode["rejected_by_criterion"]["2_airborne_too_short"] == 2


def test_sustained_hold_is_rejected_and_counted() -> None:
  # A 120 ms contact, then a long flight, then another 120 ms contact.
  _, trace, _ = _build_trace(contact_windows=[(0, 120), (700, 820)])
  episode = _score(trace)

  assert episode["hold_count"] == 2
  assert episode["longest_racket_contact_s"] == pytest.approx(0.12)
  assert episode["qualified_flights"] == 0
  assert episode["rejected_by_criterion"]["6_bounding_contact_too_long"] == 1
  assert not episode["episode_gate_pass"]


def test_low_flight_fails_the_half_target_rise() -> None:
  # A 250 ms airborne gap clears the duration floor but rises only ~76 mm.
  _, trace, _ = _build_trace(apex_m=0.076, contact_windows=[(0, 8), (300, 308)])
  episode = _score(trace)

  assert episode["flight_candidates"] == 1
  assert episode["qualified_flights"] == 0
  assert episode["rejected_by_criterion"]["4_rise_below_half_target"] == 1


def test_contact_carries_across_a_control_step_boundary() -> None:
  module, trace, _ = _build_trace(contact_windows=[(17, 24), (2000, 2008)])
  episode = _score(trace)

  flat = module.flatten_substep_flags(trace.racket_substeps)
  segments = module.contact_segments(flat)
  # The first contact starts in substep 17 (control step 0) and ends in substep 23
  # (control step 1). Flattened, it is one 7 ms contact, not a 3 ms and a 4 ms one.
  assert segments[0].tolist() == [17, 24]
  assert episode["racket_contact_edges"] == 2
  assert episode["racket_contact_time_s"] == pytest.approx(15 * 0.001)
  assert episode["longest_racket_contact_s"] == pytest.approx(0.008)

  # The same carry rule must also see a 60 ms contact that spans three steps as one
  # hold, which is the case a per-step maximum cannot resolve.
  _, held, _ = _build_trace(contact_windows=[(30, 90), (2000, 2008)])
  held_episode = _score(held)
  assert module.contact_segments(module.flatten_substep_flags(held.racket_substeps))[
    0
  ].tolist() == [30, 90]
  assert held_episode["hold_count"] == 1
  assert held_episode["longest_racket_contact_s"] == pytest.approx(0.06)


def test_flight_that_ends_the_episode_is_not_counted() -> None:
  # A single contact at 300 ms: the launch flight runs to the last recorded step and
  # no returning contact exists in the episode, so nothing is qualified.
  _, trace, _ = _build_trace(steps=60, contact_windows=[(300, 308)])
  episode = _score(trace)

  assert episode["steps"] == 60
  assert episode["racket_contact_edges"] == 1
  assert episode["flight_candidates"] == 0
  assert episode["qualified_flights"] == 0
  # A last-sample index clamp keeps the scoring inside the recorded episode.
  assert len(episode["candidate_flight_durations_s"]) == 0


def test_return_test_uses_the_plate_frame_not_world_axes() -> None:
  # Plate yawed 90 deg: its long axis is world +y. A ball returning 0.150 m along
  # world +y is inside the plate ellipse (0.150 / 0.157 = 0.96) but outside any
  # world-x test (0.150 / 0.117 = 1.28), so the verdict separates the two rules.
  _, inside, _ = _build_trace(
    lateral_world=np.array([0.0, 0.150]), plate_yaw_rad=np.pi / 2
  )
  inside_episode = _score(inside)
  assert inside_episode["qualified_flights"] > 0
  assert max(inside_episode["qualified_return_plate_radial_fraction"]) < 1.0

  # 0.200 m along world +y is outside the plate's long half-axis of 0.157 m.
  _, outside, _ = _build_trace(
    lateral_world=np.array([0.0, 0.200]), plate_yaw_rad=np.pi / 2
  )
  outside_episode = _score(outside)
  assert outside_episode["qualified_flights"] == 0
  assert (
    outside_episode["rejected_by_criterion"]["7_return_outside_plate"]
    == outside_episode["flight_candidates"]
  )
  assert outside_episode["flight_candidates"] > 0


def test_nonracket_contact_breaks_a_flight() -> None:
  _, trace, windows = _build_trace(contact_windows=[(0, 8), (600, 608), (1200, 1208)])
  other = trace.other_substeps.copy()
  # A ground/body touch in the middle of the first flight only.
  other[30, 10] = True
  module = _module()
  patched = module.EpisodeTrace(
    ball_pos_w=trace.ball_pos_w,
    ball_vel_w=trace.ball_vel_w,
    plate_pos_w=trace.plate_pos_w,
    plate_axes_w=trace.plate_axes_w,
    racket_substeps=trace.racket_substeps,
    other_substeps=other,
    termination=dict(trace.termination),
    env_origin_w=trace.env_origin_w,
  )
  episode = _score(patched)

  assert episode["flight_candidates"] == len(windows) - 1
  assert episode["rejected_by_criterion"]["2_nonracket_contact_in_flight"] == 1
  assert episode["nonracket_contact_substeps"] == 1
  # The other flight is unaffected, so the longest run is 1 of 2.
  assert episode["longest_qualified_sequence"] == 1


def test_trace_rejects_misaligned_arrays() -> None:
  module, trace, _ = _build_trace(steps=10, contact_windows=[(0, 8)])
  with pytest.raises(ValueError, match="racket_substeps"):
    module.EpisodeTrace(
      ball_pos_w=trace.ball_pos_w,
      ball_vel_w=trace.ball_vel_w,
      plate_pos_w=trace.plate_pos_w,
      plate_axes_w=trace.plate_axes_w,
      racket_substeps=trace.racket_substeps[:, :10],
      other_substeps=trace.other_substeps,
    )


def test_config_comparison_allows_only_the_declared_differences() -> None:
  module = _config_module()
  saved: dict[str, Any] = {
    "decimation": 20,
    "scene": {
      "num_envs": 4096,
      "sensors": [{"name": "bat_ball_contact", "history_length": 20}],
    },
    "seed": 7,
    "recorders": {},
    "rewards": {"ball_energy": {"params": {"target_height": 0.4}}},
  }
  runtime: dict[str, Any] = {
    "decimation": 20,
    "scene": {
      "num_envs": 32,
      "sensors": [
        {"name": "bat_ball_contact", "history_length": 20},
        {"name": "ball_any_contact"},
      ],
    },
    "seed": 20261105,
    "recorders": {"nominal_trace": {"func": "<class>"}},
    "rewards": {"ball_energy": {"params": {"target_height": 0.4}}},
  }

  report = module.compare_env_config(runtime, saved)
  assert report["ok"]
  assert report["capture_additions"]["appended_sensor"] == "ball_any_contact"
  assert report["capture_additions"]["recorder_terms"] == ["nominal_trace"]
  unexpected = {row["path"] for row in report["unexpected_differences"]}
  assert unexpected == set()
  allowed = {row["path"] for row in report["differences"]}
  assert allowed == {"scene.num_envs", "seed"}

  # A different reward target is the failure this check exists to catch: an arm run
  # against the wrong energy target would score a task nobody trained.
  runtime["rewards"]["ball_energy"]["params"]["target_height"] = 0.2
  report = module.compare_env_config(runtime, saved)
  assert not report["ok"]
  assert {row["path"] for row in report["unexpected_differences"]} == {
    "rewards.ball_energy.params.target_height"
  }

  # And an unexpected extra sensor is refused rather than silently accepted.
  runtime["rewards"]["ball_energy"]["params"]["target_height"] = 0.4
  runtime["scene"]["sensors"].append({"name": "something_else"})
  with pytest.raises(ValueError, match="plus at most one appended capture sensor"):
    module.compare_env_config(runtime, saved)


def test_recorder_captures_the_terminal_step_and_release_state() -> None:
  # A real environment on CPU, 2 environments, a 3 s episode: this exercises the
  # hook order (record_pre_reset before _reset_idx, record_post_step after), the
  # 1 ms contact windows and the environment-origin offsets. It is not a policy
  # evaluation and uses zero action on both environments.
  import torch

  import mjlab.tasks  # noqa: F401  (task registry)
  from mjlab.asset_zoo.robots.agibot_x2.tennis_constants import (
    BALL_SPAWN_LOCAL_POS,
    JUGGLE_ENERGY_BALL_REST_Z,
  )
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg

  recorder_module = _recorder_module()
  task = (
    "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
    "-Ball3-Pelvis-Nominal"
  )
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = 2
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.num_envs = 2
  cfg.seed = 20261105
  cfg.episode_length_s = 3.0
  recorder_module.add_capture_additions(cfg)

  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cpu"))
  recorder = recorder_module.recorder_term(env.unwrapped)
  env.seed(20261105)
  env.reset()

  steps = int(env.unwrapped.max_episode_length)
  zero = torch.zeros((2, env.unwrapped.action_manager.total_action_dim))
  for _ in range(steps):
    env.step(zero)
    if bool((recorder.ended >= 0).all()):
      break

  raw = recorder.export(steps)
  env.close()

  assert raw["racket_substeps"].shape == (steps, 2, 20)
  assert raw["plate_axes_w"].shape == (steps, 2, 3, 3)
  # The wrapper resets once at construction and this script resets once after
  # seeding, so the recorded release state is the second one. Later resets are
  # auto-resets of episodes that are not scored.
  assert raw["pre_step_resets"].tolist() == [2, 2]
  assert (raw["reset_count"] >= raw["pre_step_resets"]).all()

  # Both environments end their first episode inside the budget, at the same step.
  assert (raw["end_step"] >= 0).all()
  assert len(set(raw["end_step"].tolist())) == 1
  ended = int(raw["end_step"][0])
  assert 0 < ended < steps - 1
  term_names = recorder.termination_terms
  flags = dict(zip(term_names, raw["term_flags"][0].tolist(), strict=True))
  assert flags["ball_dropped"], f"expected a lost ball, got {flags}"
  assert not flags["time_out"]

  # The release state is the nominal one: 0.20 m above the ball's rest height, with
  # the +-5 mm lateral jitter, and identical in environment-local coordinates.
  origin = raw["env_origins"]
  local_initial = raw["initial_ball_pos_w"] - origin
  assert local_initial[:, 2] == pytest.approx(
    JUGGLE_ENERGY_BALL_REST_Z + 0.20, abs=1e-4
  )
  lateral = local_initial[:, :2] - np.asarray(BALL_SPAWN_LOCAL_POS[:2])
  assert float(np.abs(lateral).max()) <= 0.005 + 1e-5
  assert float(np.abs(raw["env_origins"]).max()) > 0.5  # origins are not all zero
  # Both environments start from the same nominal release, up to the two
  # independent +-5 mm lateral jitter draws.
  assert local_initial[0] == pytest.approx(local_initial[1], abs=0.011)

  # The racket struck the ball, and the ball then touched something that is not the
  # racket: the direct non-racket contact channel works. The two are disjoint by
  # construction.
  assert raw["racket_substeps"].any()
  assert raw["other_substeps"].any()
  assert not (raw["racket_substeps"] & raw["other_substeps"]).any()

  # Scoring the recorded episode must see the dropped ball as a loss and must not
  # invent a qualified flight from the single release bounce.
  scoring = _module()
  trace = scoring.EpisodeTrace(
    ball_pos_w=raw["ball_pos_w"][: ended + 1, 0],
    ball_vel_w=raw["ball_vel_w"][: ended + 1, 0],
    plate_pos_w=raw["plate_pos_w"][: ended + 1, 0],
    plate_axes_w=raw["plate_axes_w"][: ended + 1, 0],
    racket_substeps=raw["racket_substeps"][: ended + 1, 0],
    other_substeps=raw["other_substeps"][: ended + 1, 0],
    termination=flags,
    env_origin_w=origin[0],
  )
  episode = scoring.score_episode(
    trace,
    target_m=0.40,
    rest_z_m=JUGGLE_ENERGY_BALL_REST_Z,
    geometry=scoring.PlateGeometry(),
  )
  assert episode["loss"]
  assert episode["racket_contact_edges"] >= 1
  assert episode["qualified_flights"] == 0
  assert not episode["episode_gate_pass"]
