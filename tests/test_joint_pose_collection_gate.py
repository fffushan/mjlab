"""Focused CPU tests for the raw joint-pose eligibility gate.

The fixtures write small schema-version 2 collection manifests and NPZ files
that mirror the raw collector's layout.  No simulator, teacher checkpoint, or
MuJoCo model is built.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pytest

GATE_SCRIPT = (
  pathlib.Path(__file__).resolve().parents[1]
  / "scripts"
  / "experiments"
  / "joint_pose_collection"
  / "gate.py"
)
JOINT_NAMES = ("joint_a", "joint_b")
JOINT_DIM = len(JOINT_NAMES)
STEP_DT = 0.02
FRAMES = 500
STATE_MARKER = 4242.0


def _gate_module():
  """Load the gate script, which is not an importable package."""
  spec = importlib.util.spec_from_file_location(
    "joint_pose_gate_under_test", GATE_SCRIPT
  )
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


@dataclass(frozen=True)
class _Traj:
  """One synthetic raw trajectory for the gate fixtures."""

  rows: int
  outcome: str = "complete"
  nonfinite: str | None = None
  push_ticks: tuple[int, ...] = ()


def _write_clip_npz(
  path: Path,
  specs: list[_Traj],
  *,
  seed: int,
  run_index: int,
  scenario: str,
  tag: int,
) -> None:
  """Write one synthetic schema2 NPZ file."""
  rows = sum(spec.rows for spec in specs)
  root_quat = np.zeros((rows, 4), dtype=np.float32)
  root_quat[:, 0] = 1.0
  joint_pos = np.full((rows, JOINT_DIM), STATE_MARKER + tag, dtype=np.float32)
  joint_vel = np.full((rows, JOINT_DIM), -STATE_MARKER - tag, dtype=np.float32)
  tick = np.zeros(rows, dtype=np.int32)
  time_s = np.zeros(rows, dtype=np.float32)
  env_index = np.zeros(rows, dtype=np.int32)
  offsets = np.zeros(len(specs) + 1, dtype=np.int64)
  push_offsets = np.zeros(len(specs) + 1, dtype=np.int64)
  push_ticks: list[int] = []
  push_env_index: list[int] = []
  push_count = np.zeros(len(specs), dtype=np.int32)
  outcomes = np.empty(len(specs), dtype="U8")
  steps = np.zeros(len(specs), dtype=np.int32)
  initial_frames = np.zeros(len(specs), dtype=np.int32)
  trajectory_seed = np.zeros(len(specs), dtype=np.int64)
  trajectory_run = np.zeros(len(specs), dtype=np.int32)
  trajectory_scenario = np.empty(len(specs), dtype="U9")
  trajectory_env = np.zeros(len(specs), dtype=np.int32)
  for index, spec in enumerate(specs):
    start = int(offsets[index])
    stop = start + spec.rows
    offsets[index + 1] = stop
    tick[start:stop] = np.arange(spec.rows, dtype=np.int32)
    time_s[start:stop] = tick[start:stop].astype(np.float32) * np.float32(STEP_DT)
    trajectory_env[index] = index
    env_index[start:stop] = trajectory_env[index]
    outcomes[index] = spec.outcome
    steps[index] = spec.rows - 1
    trajectory_seed[index] = seed
    trajectory_run[index] = run_index
    trajectory_scenario[index] = scenario
    if spec.nonfinite == "root_quat_wxyz":
      root_quat[start, 0] = np.nan
    elif spec.nonfinite == "joint_pos":
      joint_pos[start, 0] = np.nan
    elif spec.nonfinite == "joint_vel":
      joint_vel[start, 0] = np.nan
    elif spec.nonfinite is not None:
      raise AssertionError(f"unknown nonfinite field {spec.nonfinite!r}")
    push_count[index] = len(spec.push_ticks)
    push_offsets[index + 1] = push_offsets[index] + len(spec.push_ticks)
    push_ticks.extend(spec.push_ticks)
    push_env_index.extend([int(trajectory_env[index])] * len(spec.push_ticks))
  np.savez(
    path,
    root_quat_wxyz=root_quat,
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    tick=tick,
    time_s=time_s,
    env_index=env_index,
    trajectory_offsets=offsets,
    trajectory_env_index=trajectory_env,
    trajectory_outcome=outcomes,
    trajectory_steps=steps,
    trajectory_initial_frame=initial_frames,
    trajectory_seed=trajectory_seed,
    trajectory_run_index=trajectory_run,
    trajectory_scenario=trajectory_scenario,
    trajectory_push_count=push_count,
    push_offsets=push_offsets,
    push_tick=np.asarray(push_ticks, dtype=np.int32),
    push_env_index=np.asarray(push_env_index, dtype=np.int32),
  )


def _write_collection(
  root: Path,
  motions: dict[str, list[_Traj]],
  *,
  seed: int = 100,
  num_seeds: int = 1,
  sampling_mode: str = "start",
  scenario: str = "baseline",
  max_steps: int | None = None,
  tag: int = 0,
  manifest_name: str = "manifest.json",
) -> Path:
  """Write one complete schema2 collection and return its manifest path."""
  trajectory_counts = {len(specs) for specs in motions.values()}
  assert len(trajectory_counts) == 1, "all clips must have the same env count"
  num_envs = trajectory_counts.pop()
  root.mkdir(parents=True, exist_ok=True)
  (root / "trajectories").mkdir(exist_ok=True)
  clips: list[dict[str, Any]] = []
  for run_index in range(num_seeds):
    run_seed = seed + run_index
    for clip_index, (motion_name, specs) in enumerate(motions.items()):
      output = (
        f"trajectories/run{run_index:03d}_seed{run_seed:03d}_"
        f"{clip_index:03d}_{Path(motion_name).stem}_{scenario}.npz"
      )
      path = root / output
      _write_clip_npz(
        path,
        specs,
        seed=run_seed,
        run_index=run_index,
        scenario=scenario,
        tag=tag * 100 + clip_index,
      )
      counts = {"complete": 0, "failure": 0, "budget": 0}
      for spec in specs:
        counts[spec.outcome] += 1
      clips.append(
        {
          "index": clip_index,
          "run_index": run_index,
          "seed": run_seed,
          "scenario": scenario,
          "disturbance": (None if scenario == "baseline" else {"kind": "synthetic"}),
          "file": f"/data/{motion_name}",
          "file_name": motion_name,
          "frames": FRAMES,
          "fps": 50.0,
          "step_budget": FRAMES if max_steps is None else max_steps,
          "budget_source": ("clip_length" if max_steps is None else "max_steps"),
          "sampling_mode": sampling_mode,
          "output": output,
          "control_dt_s": STEP_DT,
          "joint_names": list(JOINT_NAMES),
          "rows": sum(spec.rows for spec in specs),
          "trajectories": len(specs),
          "pushes_observed": sum(len(spec.push_ticks) for spec in specs),
          "outcomes": counts,
        }
      )
  manifest = {
    "schema_version": 2,
    "kind": "x2_joint_pose_collection",
    "complete": True,
    "created_utc": "2026-01-01T00:00:00+00:00",
    "task": "synthetic",
    "checkpoint": {"file": "/data/model.pt"},
    "collection": {
      "num_envs": num_envs,
      "seed": seed,
      "num_seeds": num_seeds,
      "seeds": [seed + index for index in range(num_seeds)],
      "num_clips": len(motions),
      "num_seed_runs": num_seeds,
      "num_clip_rollouts": len(clips),
      "total_trajectories": len(clips) * num_envs,
      "max_steps": max_steps,
      "device": "cpu",
      "lookahead_s": 0.2,
      "actor_corruption": True,
      "push_robot_event": scenario == "disturbed",
      "sampling_mode": sampling_mode,
      "standing_start_prob": 0.0,
      "auto_reset": False,
      "scenario": scenario,
      "disturbance": (None if scenario == "baseline" else {"kind": "synthetic"}),
    },
    "frame": {},
    "joint_names": list(JOINT_NAMES),
    "control_dt_s": STEP_DT,
    "units": {},
    "sample_timing": "synthetic",
    "arrays": {},
    "clips": clips,
  }
  manifest_path = root / manifest_name
  manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
  return manifest_path


def _mutate_npz(path: Path, mutate: Callable[[dict[str, np.ndarray]], None]) -> None:
  with np.load(path) as archive:
    payload = {name: np.array(archive[name]) for name in archive.files}
  mutate(payload)
  np.savez(path, **cast(dict[str, Any], payload))


def _edit_manifest(path: Path, edit: Callable[[dict[str, Any]], None]) -> None:
  document = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
  edit(document)
  path.write_text(json.dumps(document), encoding="utf-8")


def _run_config(module: Any, manifest: Path, output: Path) -> Any:
  return module.GateConfig(manifest=[manifest], output_dir=output)


def _output_path(manifest: Path, output: str) -> Path:
  return manifest.parent / output


def test_cli_help_and_complete_finite_trajectory(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  module = _gate_module()
  with pytest.raises(SystemExit) as help_exit:
    module.parse_config(["--help"])
  assert help_exit.value.code == 0
  captured = capsys.readouterr()
  help_text = captured.out + captured.err
  assert "--manifest" in help_text
  assert "--output-dir" in help_text

  manifest = _write_collection(
    tmp_path / "collection", {"clip_a.npz": [_Traj(rows=41)]}
  )
  npz_path = _output_path(
    manifest,
    "trajectories/run000_seed100_000_clip_a_baseline.npz",
  )
  manifest_before = manifest.read_bytes()
  npz_before = npz_path.read_bytes()
  output = tmp_path / "out"
  config = module.parse_config(
    ["--manifest", str(manifest), "--output-dir", str(output)]
  )
  module.main(config)

  report = json.loads((output / "eligibility.json").read_text(encoding="utf-8"))
  totals = report["totals"]
  assert report["kind"] == "x2_joint_pose_eligibility"
  assert report["schema_version"] == 1
  assert totals["trajectories"] == 1
  assert totals["accepted_trajectories"] == 1
  assert totals["rejected_trajectories"] == 0
  assert totals["candidate_windows"] == 1
  assert totals["retained_windows"] == 1
  assert totals["rejected_windows"] == 0
  assert (
    report["inputs"][0]["manifest_sha256"]
    == hashlib.sha256(manifest_before).hexdigest()
  )
  assert report["files"][0]["sha256"] == hashlib.sha256(npz_before).hexdigest()
  assert report["files"][0]["group"] == "start/baseline"
  assert report["files"][0]["trajectory_records"][0]["window_start_range"] == {
    "start": 0,
    "stop": 1,
  }
  assert report["window_contract"]["history_frames"] == 8
  assert report["window_contract"]["current_frames"] == 1
  assert report["window_contract"]["future_frames"] == 32
  assert report["window_contract"]["window_frames"] == 41
  assert report["policy"]["outcome_requirement"] == "complete"
  assert manifest.read_bytes() == manifest_before
  assert npz_path.read_bytes() == npz_before


def test_window_boundary_40_41_42_rows(tmp_path: Path) -> None:
  module = _gate_module()
  manifest = _write_collection(
    tmp_path / "collection",
    {"clip.npz": [_Traj(rows=41), _Traj(rows=42), _Traj(rows=40)]},
  )
  result = module.run_gate(_run_config(module, manifest, tmp_path / "out"))

  records = result.report["files"][0]["trajectory_records"]
  assert [
    (record["rows"], record["candidate_windows"], record["status"])
    for record in records
  ] == [
    (41, 1, "accepted"),
    (42, 2, "accepted"),
    (40, 0, "rejected"),
  ]
  assert records[0]["window_start_range"] == {"start": 0, "stop": 1}
  assert records[1]["window_start_range"] == {"start": 41, "stop": 43}
  assert records[2]["reason"] == "too_short"
  assert records[2]["retained_windows"] == 0
  totals = result.report["totals"]
  assert totals["candidate_windows"] == 3
  assert totals["retained_windows"] == 3
  assert totals["rejected_windows"] == 0
  assert totals["accepted_trajectories"] == 2
  assert totals["rejected_trajectories"] == 1
  assert result.report["by_reason"][2] == {
    "reason": "too_short",
    "trajectories": 1,
    "candidate_windows": 0,
    "rejected_windows": 0,
  }


def test_failed_trajectory_is_excluded_whole_even_with_long_prefix(
  tmp_path: Path,
) -> None:
  module = _gate_module()
  manifest = _write_collection(
    tmp_path / "collection",
    {
      "clip.npz": [
        _Traj(rows=200, outcome="failure"),
        _Traj(rows=41, outcome="failure"),
        _Traj(rows=41),
      ]
    },
  )
  result = module.run_gate(_run_config(module, manifest, tmp_path / "out"))
  records = result.report["files"][0]["trajectory_records"]

  assert records[0]["status"] == "rejected"
  assert records[0]["reason"] == "outcome_non_complete"
  assert records[0]["candidate_windows"] == 160
  assert records[0]["retained_windows"] == 0
  assert records[1]["status"] == "rejected"
  assert records[1]["reason"] == "outcome_non_complete"
  assert records[1]["candidate_windows"] == 1
  assert records[1]["retained_windows"] == 0
  assert records[2]["status"] == "accepted"
  totals = result.report["totals"]
  assert totals["candidate_windows"] == 162
  assert totals["retained_windows"] == 1
  assert totals["rejected_windows"] == 161
  assert totals["reasons"]["outcome_non_complete"] == 2


def test_budget_is_excluded_and_unknown_outcome_is_malformed(
  tmp_path: Path,
) -> None:
  module = _gate_module()
  manifest = _write_collection(
    tmp_path / "budget",
    {"clip.npz": [_Traj(rows=60, outcome="budget")]},
  )
  result = module.run_gate(_run_config(module, manifest, tmp_path / "budget_out"))
  record = result.report["files"][0]["trajectory_records"][0]
  assert record["status"] == "rejected"
  assert record["reason"] == "outcome_non_complete"
  assert record["candidate_windows"] == 20
  assert record["retained_windows"] == 0

  unknown = _write_collection(tmp_path / "unknown", {"clip.npz": [_Traj(rows=60)]})
  unknown_npz = _output_path(
    unknown, "trajectories/run000_seed100_000_clip_baseline.npz"
  )

  def set_unknown(payload: dict[str, np.ndarray]) -> None:
    payload["trajectory_outcome"] = np.asarray(["unknown"], dtype="U8")

  _mutate_npz(unknown_npz, set_unknown)
  _edit_manifest(
    unknown,
    lambda document: document["clips"][0].__setitem__(
      "outcomes", {"complete": 0, "failure": 0, "budget": 0}
    ),
  )
  output = tmp_path / "unknown_out"
  with pytest.raises(ValueError, match="unknown trajectory outcome"):
    module.run_gate(_run_config(module, unknown, output))
  assert not output.exists()


def test_complete_disturbed_retains_zero_and_nonzero_push_trajectories(
  tmp_path: Path,
) -> None:
  module = _gate_module()
  manifest = _write_collection(
    tmp_path / "collection",
    {
      "clip.npz": [
        _Traj(rows=60, push_ticks=(10, 20)),
        _Traj(rows=41, push_ticks=()),
      ]
    },
    sampling_mode="uniform",
    scenario="disturbed",
  )
  result = module.run_gate(_run_config(module, manifest, tmp_path / "out"))
  records = result.report["files"][0]["trajectory_records"]
  assert [record["status"] for record in records] == ["accepted", "accepted"]
  assert [record["retained_windows"] for record in records] == [20, 1]
  assert result.report["files"][0]["group"] == "uniform/disturbed"
  assert result.report["totals"]["rejected_trajectories"] == 0


@pytest.mark.parametrize("field", ["root_quat_wxyz", "joint_pos", "joint_vel"])
def test_nonfinite_required_state_excludes_complete_trajectory(
  tmp_path: Path, field: str
) -> None:
  module = _gate_module()
  manifest = _write_collection(
    tmp_path / "collection",
    {"clip.npz": [_Traj(rows=60, nonfinite=field)]},
  )
  result = module.run_gate(_run_config(module, manifest, tmp_path / "out"))
  record = result.report["files"][0]["trajectory_records"][0]
  assert record["status"] == "rejected"
  assert record["reason"] == "nonfinite_state"
  assert record["candidate_windows"] == 20
  assert record["retained_windows"] == 0
  assert record["rejected_windows"] == 20
  assert result.report["totals"]["retained_windows"] == 0


def test_rejection_precedence_is_disjoint(tmp_path: Path) -> None:
  module = _gate_module()
  manifest = _write_collection(
    tmp_path / "collection",
    {
      "clip.npz": [
        _Traj(rows=60, outcome="failure", nonfinite="joint_pos"),
        _Traj(rows=5, nonfinite="joint_vel"),
        _Traj(rows=5),
      ]
    },
  )
  result = module.run_gate(_run_config(module, manifest, tmp_path / "out"))
  records = result.report["files"][0]["trajectory_records"]
  assert [record["reason"] for record in records] == [
    "outcome_non_complete",
    "nonfinite_state",
    "too_short",
  ]


def test_windows_do_not_cross_trajectory_boundaries(tmp_path: Path) -> None:
  module = _gate_module()
  manifest = _write_collection(
    tmp_path / "collection",
    {"clip.npz": [_Traj(rows=42), _Traj(rows=5), _Traj(rows=41)]},
  )
  result = module.run_gate(_run_config(module, manifest, tmp_path / "out"))
  records = result.report["files"][0]["trajectory_records"]
  assert records[0]["window_start_range"] == {"start": 0, "stop": 2}
  assert records[1]["status"] == "rejected"
  assert records[1]["row_start"] == 42
  assert records[1]["row_stop"] == 47
  assert records[2]["window_start_range"] == {"start": 47, "stop": 48}
  starts = {
    start
    for record in records
    if record["window_start_range"] is not None
    for start in range(
      record["window_start_range"]["start"],
      record["window_start_range"]["stop"],
    )
  }
  assert starts == {0, 1, 47}
  assert not starts.intersection(range(42, 47))


def test_multiple_manifests_keep_seed_and_group_identities(
  tmp_path: Path,
) -> None:
  module = _gate_module()
  start = _write_collection(
    tmp_path / "start",
    {
      "clip_a.npz": [_Traj(rows=41), _Traj(rows=42)],
      "clip_b.npz": [_Traj(rows=41), _Traj(rows=42)],
    },
    seed=100,
    num_seeds=2,
    tag=1,
  )
  uniform = _write_collection(
    tmp_path / "uniform",
    {
      "clip_a.npz": [_Traj(rows=41), _Traj(rows=42)],
      "clip_b.npz": [_Traj(rows=41), _Traj(rows=42)],
    },
    seed=100,
    sampling_mode="uniform",
    tag=2,
  )
  disturbed = _write_collection(
    tmp_path / "disturbed",
    {
      "clip_a.npz": [_Traj(rows=41, outcome="failure"), _Traj(rows=42)],
      "clip_b.npz": [_Traj(rows=41, outcome="failure"), _Traj(rows=42)],
    },
    seed=100,
    sampling_mode="uniform",
    scenario="disturbed",
    tag=3,
  )
  output = tmp_path / "out"
  config = module.parse_config(
    [
      "--manifest",
      str(start),
      str(uniform),
      str(disturbed),
      "--output-dir",
      str(output),
    ]
  )
  result = module.run_gate(config)
  report = result.report
  totals = report["totals"]

  assert [entry["group"] for entry in report["inputs"]] == [
    "start/baseline",
    "uniform/baseline",
    "uniform/disturbed",
  ]
  assert len(report["files"]) == 8
  assert len({entry["path"] for entry in report["files"]}) == 8
  start_files = [
    entry for entry in report["files"] if entry["group"] == "start/baseline"
  ]
  assert [entry["run_index"] for entry in start_files] == [0, 0, 1, 1]
  assert [entry["seed"] for entry in start_files] == [100, 100, 101, 101]
  assert all(entry["scenario"] == "baseline" for entry in report["files"][:6])
  groups = {entry["group"]: entry for entry in report["by_group"]}
  assert groups["start/baseline"]["retained_windows"] == 12
  assert groups["uniform/baseline"]["retained_windows"] == 6
  assert groups["uniform/disturbed"]["retained_windows"] == 4
  assert groups["uniform/disturbed"]["rejected_windows"] == 2
  assert totals["candidate_windows"] == 24
  assert totals["retained_windows"] == 22
  assert totals["rejected_windows"] == 2
  assert totals["accepted_trajectories"] == 14
  assert totals["rejected_trajectories"] == 2
  assert sum(entry["retained_windows"] for entry in report["by_group"]) == 22
  assert sum(entry["retained_windows"] for entry in report["by_motion"]) == 22
  assert sum(entry["trajectories"] for entry in report["by_reason"]) == 2
  assert report["count_identities"]["windows"]["checked"] is True
  assert report["totals"]["candidate_windows"] == (
    report["totals"]["retained_windows"] + report["totals"]["rejected_windows"]
  )


def test_duplicate_inputs_and_sources_are_refused(tmp_path: Path) -> None:
  module = _gate_module()
  manifest = _write_collection(tmp_path / "collection", {"clip.npz": [_Traj(rows=41)]})
  output = tmp_path / "out"
  with pytest.raises(ValueError, match="duplicate manifest input"):
    module.run_gate(module.GateConfig(manifest=[manifest, manifest], output_dir=output))
  assert not output.exists()

  copy = manifest.parent / "manifest_copy.json"
  copy.write_bytes(manifest.read_bytes())
  with pytest.raises(ValueError, match="duplicate manifest contents"):
    module.run_gate(module.GateConfig(manifest=[manifest, copy], output_dir=output))
  assert not output.exists()

  _edit_manifest(
    copy, lambda document: document.__setitem__("created_utc", "different")
  )
  with pytest.raises(ValueError, match="duplicate collection NPZ source"):
    module.run_gate(module.GateConfig(manifest=[manifest, copy], output_dir=output))
  assert not output.exists()


def test_refuses_existing_output_and_output_inside_collection(
  tmp_path: Path,
) -> None:
  module = _gate_module()
  manifest = _write_collection(tmp_path / "collection", {"clip.npz": [_Traj(rows=41)]})
  output = tmp_path / "out"
  module.run_gate(_run_config(module, manifest, output))
  with pytest.raises(FileExistsError, match="already exists"):
    module.run_gate(_run_config(module, manifest, output))

  inside = manifest.parent / "eligibility"
  with pytest.raises(ValueError, match="inside input collection"):
    module.run_gate(_run_config(module, manifest, inside))
  assert not inside.exists()


def test_refuses_unsupported_incomplete_and_missing_inputs(
  tmp_path: Path,
) -> None:
  module = _gate_module()
  manifest = _write_collection(tmp_path / "schema", {"clip.npz": [_Traj(rows=41)]})
  _edit_manifest(manifest, lambda document: document.__setitem__("schema_version", 1))
  with pytest.raises(ValueError, match="unsupported manifest schema_version"):
    module.run_gate(_run_config(module, manifest, tmp_path / "schema_out"))

  incomplete = _write_collection(
    tmp_path / "incomplete", {"clip.npz": [_Traj(rows=41)]}
  )
  _edit_manifest(incomplete, lambda document: document.__setitem__("complete", False))
  with pytest.raises(ValueError, match="manifest is not complete"):
    module.run_gate(_run_config(module, incomplete, tmp_path / "incomplete_out"))

  missing = _write_collection(tmp_path / "missing", {"clip.npz": [_Traj(rows=41)]})
  missing_npz = _output_path(
    missing, "trajectories/run000_seed100_000_clip_baseline.npz"
  )
  missing_npz.unlink()
  with pytest.raises(FileNotFoundError, match="collection NPZ not found"):
    module.run_gate(_run_config(module, missing, tmp_path / "missing_out"))


def test_refuses_clip_path_escape_and_absolute_output(tmp_path: Path) -> None:
  module = _gate_module()
  escaped = _write_collection(tmp_path / "escaped", {"clip.npz": [_Traj(rows=41)]})
  _edit_manifest(
    escaped,
    lambda document: document["clips"][0].__setitem__("output", "../outside.npz"),
  )
  with pytest.raises(ValueError, match="escapes the collection directory"):
    module.run_gate(_run_config(module, escaped, tmp_path / "escaped_out"))

  absolute = _write_collection(tmp_path / "absolute", {"clip.npz": [_Traj(rows=41)]})
  _edit_manifest(
    absolute,
    lambda document: document["clips"][0].__setitem__("output", "/tmp/outside.npz"),
  )
  with pytest.raises(ValueError, match="relative file path"):
    module.run_gate(_run_config(module, absolute, tmp_path / "absolute_out"))


def test_hash_counts_and_ranges_round_trip_without_state_copies(
  tmp_path: Path,
) -> None:
  module = _gate_module()
  manifest = _write_collection(
    tmp_path / "collection",
    {"clip.npz": [_Traj(rows=41), _Traj(rows=60)]},
  )
  npz_path = _output_path(manifest, "trajectories/run000_seed100_000_clip_baseline.npz")
  manifest_before = manifest.read_bytes()
  npz_before = npz_path.read_bytes()
  result = module.run_gate(_run_config(module, manifest, tmp_path / "out"))
  report = result.report
  report_text = result.path.read_text(encoding="utf-8")

  assert (
    report["inputs"][0]["manifest_sha256"]
    == hashlib.sha256(manifest_before).hexdigest()
  )
  assert report["files"][0]["sha256"] == hashlib.sha256(npz_before).hexdigest()
  assert manifest.read_bytes() == manifest_before
  assert npz_path.read_bytes() == npz_before
  assert str(STATE_MARKER) not in report_text
  assert len(report_text) < 20_000
  assert list((tmp_path / "out").iterdir()) == [result.path]
  records = report["files"][0]["trajectory_records"]
  assert sum(record["candidate_windows"] for record in records) == 21
  for record in records:
    if record["status"] == "accepted":
      window_range = record["window_start_range"]
      assert window_range is not None
      assert (
        window_range["stop"] - window_range["start"] == (record["candidate_windows"])
      )
      assert record["retained_windows"] == record["candidate_windows"]
    else:
      assert record["window_start_range"] is None
  totals = report["totals"]
  assert totals["candidate_windows"] == 21
  assert totals["retained_windows"] == 21
  assert totals["rejected_windows"] == 0


def test_gate_module_does_not_import_simulator_packages() -> None:
  gate_path = str(GATE_SCRIPT)
  code = (
    "import importlib.util, sys\n"
    f"spec = importlib.util.spec_from_file_location('g', {gate_path!r})\n"
    "module = importlib.util.module_from_spec(spec)\n"
    "sys.modules[spec.name] = module\n"
    "spec.loader.exec_module(module)\n"
    "for forbidden in ('mjlab', 'torch', 'mujoco'):\n"
    "    assert forbidden not in sys.modules, forbidden\n"
    "print('ok')\n"
  )
  completed = subprocess.run(
    [sys.executable, "-c", code],
    capture_output=True,
    text=True,
    timeout=60,
    check=False,
  )
  assert completed.returncode == 0, completed.stderr
  assert completed.stdout.strip() == "ok"


def _bad_joint_shape(payload: dict[str, np.ndarray]) -> None:
  payload["joint_pos"] = payload["joint_pos"][:, :1]


def _bad_offsets_start(payload: dict[str, np.ndarray]) -> None:
  payload["trajectory_offsets"] = payload["trajectory_offsets"] + 1


def _bad_offsets_order(payload: dict[str, np.ndarray]) -> None:
  payload["trajectory_offsets"] = np.asarray([0, 42, 42], dtype=np.int64)


def _bad_offsets_end(payload: dict[str, np.ndarray]) -> None:
  offsets = payload["trajectory_offsets"]
  offsets[-1] = offsets[-1] - 1
  payload["trajectory_offsets"] = offsets


def _bad_offsets_dtype(payload: dict[str, np.ndarray]) -> None:
  payload["trajectory_offsets"] = payload["trajectory_offsets"].astype(np.float64)


def _bad_tick(payload: dict[str, np.ndarray]) -> None:
  payload["tick"] = payload["tick"] + 1


def _bad_time(payload: dict[str, np.ndarray]) -> None:
  payload["time_s"] = payload["time_s"] + 1.0


def _bad_steps(payload: dict[str, np.ndarray]) -> None:
  payload["trajectory_steps"] = payload["trajectory_steps"] + 1


def _bad_seed(payload: dict[str, np.ndarray]) -> None:
  payload["trajectory_seed"] = payload["trajectory_seed"] + 1


def _bad_scenario(payload: dict[str, np.ndarray]) -> None:
  payload["trajectory_scenario"] = np.full(
    payload["trajectory_scenario"].shape, "disturbed", dtype="U9"
  )


def _bad_initial_frame(payload: dict[str, np.ndarray]) -> None:
  payload["trajectory_initial_frame"] = np.full(
    payload["trajectory_initial_frame"].shape, FRAMES, dtype=np.int32
  )


def _bad_env_index(payload: dict[str, np.ndarray]) -> None:
  payload["env_index"] = payload["env_index"] + 1


def _bad_env_identity(payload: dict[str, np.ndarray]) -> None:
  payload["trajectory_env_index"] = np.asarray([0, 0], dtype=np.int32)


def _bad_push_count(payload: dict[str, np.ndarray]) -> None:
  payload["trajectory_push_count"] = payload["trajectory_push_count"] + 1


@pytest.mark.parametrize(
  ("mutate", "match"),
  [
    (_bad_joint_shape, "joint_pos must have shape"),
    (_bad_offsets_start, "trajectory_offsets must start at 0"),
    (_bad_offsets_order, "trajectory_offsets must increase"),
    (_bad_offsets_end, "trajectory_offsets ends at"),
    (_bad_offsets_dtype, "trajectory_offsets must have an integer dtype"),
    (_bad_tick, "tick slice is not the contiguous sequence"),
    (_bad_time, "time_s is inconsistent"),
    (_bad_steps, "trajectory_steps does not match"),
    (_bad_seed, "trajectory_seed does not match"),
    (_bad_scenario, "trajectory_scenario does not match"),
    (_bad_initial_frame, "trajectory_initial_frame is outside"),
    (_bad_env_index, "env_index does not match"),
    (_bad_env_identity, "duplicate environment rows"),
    (_bad_push_count, "trajectory_push_count does not match push_offsets"),
  ],
)
def test_refuses_malformed_npz_contract(
  tmp_path: Path,
  mutate: Callable[[dict[str, np.ndarray]], None],
  match: str,
) -> None:
  module = _gate_module()
  manifest = _write_collection(
    tmp_path / "collection",
    {"clip.npz": [_Traj(rows=41), _Traj(rows=42)]},
  )
  npz_path = _output_path(manifest, "trajectories/run000_seed100_000_clip_baseline.npz")
  _mutate_npz(npz_path, mutate)
  output = tmp_path / "out"
  with pytest.raises(ValueError, match=match):
    module.run_gate(_run_config(module, manifest, output))
  assert not output.exists()


def test_refuses_manifest_npz_provenance_mismatches(tmp_path: Path) -> None:
  module = _gate_module()

  def expect_refusal(
    name: str, edit: Callable[[dict[str, Any]], None], match: str
  ) -> None:
    manifest = _write_collection(tmp_path / name, {"clip.npz": [_Traj(rows=41)]})
    _edit_manifest(manifest, edit)
    output = tmp_path / f"{name}_out"
    with pytest.raises(ValueError, match=match):
      module.run_gate(_run_config(module, manifest, output))
    assert not output.exists()

  expect_refusal(
    "rows",
    lambda document: document["clips"][0].__setitem__(
      "rows", document["clips"][0]["rows"] + 1
    ),
    "manifest rows .* does not match NPZ rows",
  )
  expect_refusal(
    "outcomes",
    lambda document: document["clips"][0]["outcomes"].__setitem__("complete", 0),
    "manifest outcomes .* does not match NPZ outcomes",
  )
  expect_refusal(
    "joints",
    lambda document: document["clips"][0].__setitem__("joint_names", ["other"]),
    "joint_names does not match manifest joint order",
  )
  expect_refusal(
    "timing",
    lambda document: document["clips"][0].__setitem__("control_dt_s", 0.01),
    "control_dt_s .* does not match manifest control_dt_s",
  )
  expect_refusal(
    "seed",
    lambda document: document["clips"][0].__setitem__("seed", 99),
    "does not match seed",
  )
  expect_refusal(
    "total",
    lambda document: document["collection"].__setitem__("total_trajectories", 2),
    "total_trajectories .* !=",
  )
  expect_refusal(
    "schema",
    lambda document: document.__setitem__("schema_version", "2"),
    "schema_version must be an integer",
  )


@pytest.mark.parametrize("field", ["joint_names", "control_dt_s"])
def test_refuses_incompatible_input_manifests(tmp_path: Path, field: str) -> None:
  module = _gate_module()
  first = _write_collection(tmp_path / "first", {"clip.npz": [_Traj(rows=41)]})
  second = _write_collection(tmp_path / "second", {"clip.npz": [_Traj(rows=42)]}, tag=2)

  def change(document: dict[str, Any]) -> None:
    if field == "joint_names":
      names = list(reversed(document["joint_names"]))
      document["joint_names"] = names
      document["clips"][0]["joint_names"] = names
    else:
      document["control_dt_s"] = 0.04
      document["clips"][0]["control_dt_s"] = 0.04
      document["clips"][0]["fps"] = 25.0

  _edit_manifest(second, change)
  if field == "control_dt_s":
    data = _output_path(second, "trajectories/run000_seed100_000_clip_baseline.npz")
    _mutate_npz(data, lambda a: a.__setitem__("time_s", a["time_s"] * 2))
  output = tmp_path / "out"
  with pytest.raises(ValueError, match="across input manifests"):
    module.run_gate(module.GateConfig(manifest=[first, second], output_dir=output))
  assert not output.exists()
