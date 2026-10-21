"""Build a non-destructive eligibility index for raw X2 joint-pose data.

This gate reads schema-version 2 collection manifests and their NPZ files.  It
never edits, deletes, or rewrites raw data.  A trajectory is eligible only when
its stored outcome is ``complete``, its required root-orientation and joint
samples are finite, and it has at least one complete 41-row window: 8 history
rows, the current row, and 32 future rows.

The gate applies whole-trajectory decisions.  A failed or budget-stopped
trajectory is rejected completely, and a complete trajectory with any
non-finite required sample is rejected completely.  No prefix or local window
is salvaged.  Push exposure is not required.

Example:

  PYTHONPATH="$PWD/src" uv run --no-sync python \\
    scripts/experiments/joint_pose_collection/gate.py \\
    --manifest logs/joint_pose_collection/start/manifest.json \\
               logs/joint_pose_collection/uniform/manifest.json \\
    --output-dir logs/joint_pose_eligibility/batch12
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np
import tyro

INDEX_SCHEMA_VERSION = 1
INDEX_KIND = "x2_joint_pose_eligibility"
INDEX_FILENAME = "eligibility.json"
COLLECTION_SCHEMA_VERSION = 2
COLLECTION_KIND = "x2_joint_pose_collection"

OUTCOME_COMPLETE = "complete"
OUTCOME_FAILURE = "failure"
OUTCOME_BUDGET = "budget"
OUTCOMES = (OUTCOME_COMPLETE, OUTCOME_FAILURE, OUTCOME_BUDGET)

SCENARIO_BASELINE = "baseline"
SCENARIO_DISTURBED = "disturbed"
SCENARIOS = (SCENARIO_BASELINE, SCENARIO_DISTURBED)
SAMPLING_MODES = ("start", "uniform")

HISTORY_FRAMES = 8
CURRENT_FRAMES = 1
FUTURE_FRAMES = 32
WINDOW_FRAMES = HISTORY_FRAMES + CURRENT_FRAMES + FUTURE_FRAMES
"""Fixed window length: 8 history + current + 32 future, with no padding."""

REASON_OUTCOME = "outcome_non_complete"
REASON_NONFINITE = "nonfinite_state"
REASON_TOO_SHORT = "too_short"
REJECTION_REASONS = (REASON_OUTCOME, REASON_NONFINITE, REASON_TOO_SHORT)
"""Disjoint primary reasons, evaluated in this precedence order."""

REQUIRED_ARRAYS = (
  "root_quat_wxyz",
  "joint_pos",
  "joint_vel",
  "tick",
  "time_s",
  "env_index",
  "trajectory_offsets",
  "trajectory_env_index",
  "trajectory_outcome",
  "trajectory_steps",
  "trajectory_initial_frame",
  "trajectory_seed",
  "trajectory_run_index",
  "trajectory_scenario",
  "trajectory_push_count",
  "push_offsets",
  "push_tick",
  "push_env_index",
)


@dataclass(frozen=True)
class GateConfig:
  """One explicit eligibility evaluation over existing collection manifests."""

  manifest: list[Path]
  """One or more complete schema-version 2 collection manifest files."""

  output_dir: Path
  """New directory for the eligibility index. An existing path is refused."""


@dataclass(frozen=True)
class GateResult:
  """The published eligibility index and its path."""

  report: dict[str, Any]
  path: Path


@dataclass(frozen=True)
class ClipInfo:
  """Validated provenance for one NPZ file entry in a manifest."""

  index: int
  run_index: int
  seed: int
  scenario: str
  sampling_mode: str
  motion_file: str
  motion_name: str
  output: str
  path: Path
  frames: int
  fps: float
  control_dt_s: float
  joint_names: tuple[str, ...]
  rows: int
  trajectories: int
  pushes_observed: int
  outcomes: dict[str, int]


@dataclass(frozen=True)
class ManifestInfo:
  """A validated manifest and the fields the gate depends on."""

  input_index: int
  path: Path
  sha256: str
  size_bytes: int
  schema_version: int
  kind: str
  task: str
  group: str
  sampling_mode: str
  scenario: str
  control_dt_s: float
  joint_names: tuple[str, ...]
  num_envs: int
  seeds: tuple[int, ...]
  clips: tuple[ClipInfo, ...]


def parse_config(argv: Sequence[str] | None = None) -> GateConfig:
  """Parse the public CLI into a gate configuration."""
  return tyro.cli(
    GateConfig,
    args=None if argv is None else list(argv),
    description=(
      "Build a non-destructive eligibility index for raw X2 joint-pose "
      "schema2 collections."
    ),
  )


def validate_config(config: GateConfig) -> tuple[Path, tuple[Path, ...]]:
  """Resolve paths and refuse a duplicate or mutating input/output layout."""
  if not config.manifest:
    raise ValueError("at least one --manifest is required")
  output_dir = config.output_dir.expanduser().resolve()
  if output_dir.exists():
    raise FileExistsError(f"output directory already exists: {output_dir}")
  manifests: list[Path] = []
  seen: set[Path] = set()
  for raw_path in config.manifest:
    path = raw_path.expanduser().resolve()
    if not path.is_file():
      raise FileNotFoundError(f"collection manifest not found: {path}")
    if path in seen:
      raise ValueError(f"duplicate manifest input: {path}")
    seen.add(path)
    manifests.append(path)
  for path in manifests:
    collection_dir = path.parent.resolve()
    try:
      output_dir.relative_to(collection_dir)
    except ValueError:
      continue
    raise ValueError(
      f"output directory {output_dir} is inside input collection "
      f"{collection_dir}; choose a separate directory"
    )
  return output_dir, tuple(manifests)


def run_gate(config: GateConfig) -> GateResult:
  """Validate all inputs, then publish one new eligibility index."""
  output_dir, report = _build_report(config)
  _publish_index(output_dir, report)
  return GateResult(report=report, path=output_dir / INDEX_FILENAME)


def main(config: GateConfig | None = None) -> None:
  """Run the gate from the parsed command line."""
  if config is None:
    config = parse_config()
  result = run_gate(config)
  totals = result.report["totals"]
  print(f"[INFO] eligibility index: {result.path}")
  print(
    f"[INFO] trajectories: accepted={totals['accepted_trajectories']} "
    f"rejected={totals['rejected_trajectories']} total={totals['trajectories']}"
  )
  print(
    f"[INFO] windows: retained={totals['retained_windows']} "
    f"rejected={totals['rejected_windows']} "
    f"candidate={totals['candidate_windows']}"
  )


def _fail(source: Path | str, message: str) -> NoReturn:
  raise ValueError(f"{source}: {message}")


def _reject_json_constant(value: str) -> NoReturn:
  raise ValueError(f"non-finite JSON constant {value!r}")


def _require_keys(
  source: Path, document: dict[str, Any], name: str, keys: Sequence[str]
) -> None:
  missing = [key for key in keys if key not in document]
  if missing:
    _fail(source, f"{name} is missing required field(s): {', '.join(missing)}")


def _require_object(source: Path, value: Any, name: str) -> dict[str, Any]:
  if not isinstance(value, dict):
    _fail(source, f"{name} must be a JSON object")
  return cast(dict[str, Any], value)


def _require_list(source: Path, value: Any, name: str) -> list[Any]:
  if not isinstance(value, list):
    _fail(source, f"{name} must be a JSON list")
  return cast(list[Any], value)


def _require_int(
  source: Path, value: Any, name: str, *, minimum: int | None = None
) -> int:
  if isinstance(value, bool) or not isinstance(value, int):
    _fail(source, f"{name} must be an integer, got {value!r}")
  if minimum is not None and value < minimum:
    _fail(source, f"{name} must be at least {minimum}, got {value}")
  return value


def _require_float(
  source: Path, value: Any, name: str, *, positive: bool = False
) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    _fail(source, f"{name} must be a number, got {value!r}")
  number = float(value)
  if not np.isfinite(number):
    _fail(source, f"{name} must be finite, got {value!r}")
  if positive and number <= 0.0:
    _fail(source, f"{name} must be positive, got {number}")
  return number


def _require_string(source: Path, value: Any, name: str) -> str:
  if not isinstance(value, str) or not value:
    _fail(source, f"{name} must be a non-empty string, got {value!r}")
  return value


def _read_manifest(path: Path) -> tuple[str, int, dict[str, Any]]:
  """Read, hash, and parse one manifest from the same byte buffer."""
  payload = path.read_bytes()
  try:
    text = payload.decode("utf-8")
  except UnicodeDecodeError as exc:
    raise ValueError(f"{path}: manifest is not valid UTF-8") from exc
  try:
    document = json.loads(text, parse_constant=_reject_json_constant)
  except ValueError as exc:
    raise ValueError(f"{path}: manifest is not valid JSON: {exc}") from exc
  if not isinstance(document, dict):
    _fail(path, "manifest root must be a JSON object")
  return (
    hashlib.sha256(payload).hexdigest(),
    len(payload),
    cast(dict[str, Any], document),
  )


def _resolve_clip_path(source: Path, collection_dir: Path, output: str) -> Path:
  relative = Path(output)
  if relative.is_absolute() or not relative.parts:
    _fail(source, f"clip output must be a relative file path, got {output!r}")
  if ".." in relative.parts:
    _fail(source, f"clip output escapes the collection directory: {output!r}")
  resolved = (collection_dir / relative).resolve()
  try:
    resolved.relative_to(collection_dir)
  except ValueError:
    _fail(source, f"clip output escapes the collection directory: {output!r}")
  if not resolved.is_file():
    raise FileNotFoundError(f"collection NPZ not found: {resolved}")
  return resolved


def _validated_joint_names(source: Path, value: Any) -> tuple[str, ...]:
  items = _require_list(source, value, "joint_names")
  if not items:
    _fail(source, "joint_names must not be empty")
  names: list[str] = []
  for index, item in enumerate(items):
    names.append(_require_string(source, item, f"joint_names[{index}]"))
  if len(set(names)) != len(names):
    _fail(source, "joint_names contains duplicate names")
  return tuple(names)


def _validated_outcomes(source: Path, value: Any) -> dict[str, int]:
  document = _require_object(source, value, "clip outcomes")
  if set(document) != set(OUTCOMES):
    _fail(
      source,
      f"clip outcomes keys must be {list(OUTCOMES)}, got {sorted(document)}",
    )
  return {
    outcome: _require_int(
      source, document[outcome], f"clip outcomes.{outcome}", minimum=0
    )
    for outcome in OUTCOMES
  }


def _validated_clip(
  source: Path,
  collection_dir: Path,
  position: int,
  raw_clip: Any,
  *,
  manifest_joint_names: tuple[str, ...],
  manifest_control_dt_s: float,
  collection: dict[str, Any],
  seeds: tuple[int, ...],
  num_clips: int,
  max_steps: int | None,
) -> ClipInfo:
  clip = _require_object(source, raw_clip, f"clips[{position}]")
  _require_keys(
    source,
    clip,
    f"clips[{position}]",
    (
      "index",
      "run_index",
      "seed",
      "scenario",
      "sampling_mode",
      "file",
      "file_name",
      "frames",
      "fps",
      "step_budget",
      "budget_source",
      "output",
      "control_dt_s",
      "joint_names",
      "rows",
      "trajectories",
      "pushes_observed",
      "outcomes",
    ),
  )
  clip_index = _require_int(
    source, clip["index"], f"clips[{position}].index", minimum=0
  )
  if clip_index >= num_clips:
    _fail(
      source,
      f"clips[{position}].index {clip_index} is outside num_clips={num_clips}",
    )
  run_index = _require_int(
    source, clip["run_index"], f"clips[{position}].run_index", minimum=0
  )
  if run_index >= len(seeds):
    _fail(
      source,
      f"clips[{position}].run_index {run_index} has no seed entry",
    )
  clip_seed = _require_int(source, clip["seed"], f"clips[{position}].seed", minimum=0)
  if clip_seed != seeds[run_index]:
    _fail(
      source,
      f"clips[{position}].seed {clip_seed} does not match seed "
      f"{seeds[run_index]} for run_index {run_index}",
    )
  scenario = _require_string(source, clip["scenario"], f"clips[{position}].scenario")
  if scenario != collection["scenario"]:
    _fail(
      source,
      f"clips[{position}].scenario {scenario!r} does not match "
      f"collection.scenario {collection['scenario']!r}",
    )
  sampling_mode = _require_string(
    source, clip["sampling_mode"], f"clips[{position}].sampling_mode"
  )
  if sampling_mode != collection["sampling_mode"]:
    _fail(
      source,
      f"clips[{position}].sampling_mode {sampling_mode!r} does not match "
      f"collection.sampling_mode {collection['sampling_mode']!r}",
    )
  motion_file = _require_string(source, clip["file"], f"clips[{position}].file")
  motion_name = _require_string(
    source, clip["file_name"], f"clips[{position}].file_name"
  )
  frames = _require_int(source, clip["frames"], f"clips[{position}].frames", minimum=1)
  fps = _require_float(source, clip["fps"], f"clips[{position}].fps", positive=True)
  step_budget = _require_int(
    source, clip["step_budget"], f"clips[{position}].step_budget", minimum=1
  )
  budget_source = _require_string(
    source, clip["budget_source"], f"clips[{position}].budget_source"
  )
  if budget_source == "clip_length":
    if step_budget != frames:
      _fail(
        source,
        f"clips[{position}] clip_length budget {step_budget} != frames {frames}",
      )
  elif budget_source == "max_steps":
    if max_steps is None or step_budget != max_steps:
      _fail(
        source,
        f"clips[{position}] max_steps budget {step_budget} != "
        f"collection.max_steps {max_steps!r}",
      )
    if step_budget >= frames:
      _fail(
        source,
        f"clips[{position}] max_steps budget {step_budget} must be below "
        f"frames {frames}",
      )
  else:
    _fail(
      source,
      f"clips[{position}].budget_source must be 'clip_length' or 'max_steps', "
      f"got {budget_source!r}",
    )
  if max_steps is not None and max_steps < 1:
    _fail(source, f"collection.max_steps must be positive, got {max_steps}")
  output = _require_string(source, clip["output"], f"clips[{position}].output")
  clip_path = _resolve_clip_path(source, collection_dir, output)
  clip_control_dt_s = _require_float(
    source,
    clip["control_dt_s"],
    f"clips[{position}].control_dt_s",
    positive=True,
  )
  if clip_control_dt_s != manifest_control_dt_s:
    _fail(
      source,
      f"clips[{position}].control_dt_s {clip_control_dt_s} does not match "
      f"manifest control_dt_s {manifest_control_dt_s}",
    )
  joint_names = _validated_joint_names(source, clip["joint_names"])
  if joint_names != manifest_joint_names:
    _fail(
      source,
      f"clips[{position}].joint_names does not match manifest joint order",
    )
  rows = _require_int(source, clip["rows"], f"clips[{position}].rows", minimum=0)
  trajectories = _require_int(
    source,
    clip["trajectories"],
    f"clips[{position}].trajectories",
    minimum=1,
  )
  if rows < trajectories:
    _fail(
      source,
      f"clips[{position}] has rows={rows} below trajectories={trajectories}",
    )
  pushes_observed = _require_int(
    source,
    clip["pushes_observed"],
    f"clips[{position}].pushes_observed",
    minimum=0,
  )
  outcomes = _validated_outcomes(source, clip["outcomes"])
  return ClipInfo(
    index=clip_index,
    run_index=run_index,
    seed=clip_seed,
    scenario=scenario,
    sampling_mode=sampling_mode,
    motion_file=motion_file,
    motion_name=motion_name,
    output=output,
    path=clip_path,
    frames=frames,
    fps=fps,
    control_dt_s=clip_control_dt_s,
    joint_names=joint_names,
    rows=rows,
    trajectories=trajectories,
    pushes_observed=pushes_observed,
    outcomes=outcomes,
  )


def _load_manifest(input_index: int, path: Path) -> ManifestInfo:
  sha256, size_bytes, manifest = _read_manifest(path)
  _require_keys(
    path,
    manifest,
    "manifest",
    (
      "schema_version",
      "kind",
      "complete",
      "task",
      "collection",
      "joint_names",
      "control_dt_s",
      "clips",
    ),
  )
  schema_version = _require_int(path, manifest["schema_version"], "schema_version")
  if schema_version != COLLECTION_SCHEMA_VERSION:
    _fail(
      path,
      f"unsupported manifest schema_version {schema_version}; "
      f"only {COLLECTION_SCHEMA_VERSION} is supported",
    )
  kind = _require_string(path, manifest["kind"], "kind")
  if kind != COLLECTION_KIND:
    _fail(path, f"unexpected manifest kind {kind!r}")
  complete = manifest["complete"]
  if not isinstance(complete, bool):
    _fail(path, f"complete must be a boolean, got {complete!r}")
  if not complete:
    error = manifest.get("error")
    suffix = "" if error is None else f": {error}"
    _fail(path, f"manifest is not complete{suffix}")
  task = _require_string(path, manifest["task"], "task")
  joint_names = _validated_joint_names(path, manifest["joint_names"])
  control_dt_s = _require_float(
    path, manifest["control_dt_s"], "control_dt_s", positive=True
  )
  collection = _require_object(path, manifest["collection"], "collection")
  _require_keys(
    path,
    collection,
    "collection",
    (
      "num_envs",
      "seed",
      "num_seeds",
      "seeds",
      "num_clips",
      "num_seed_runs",
      "num_clip_rollouts",
      "total_trajectories",
      "max_steps",
      "sampling_mode",
      "scenario",
      "push_robot_event",
      "disturbance",
    ),
  )
  num_envs = _require_int(
    path, collection["num_envs"], "collection.num_envs", minimum=1
  )
  base_seed = _require_int(path, collection["seed"], "collection.seed", minimum=0)
  num_seeds = _require_int(
    path, collection["num_seeds"], "collection.num_seeds", minimum=1
  )
  seeds_raw = _require_list(path, collection["seeds"], "collection.seeds")
  if len(seeds_raw) != num_seeds:
    _fail(
      path,
      f"collection.seeds has {len(seeds_raw)} entries, expected {num_seeds}",
    )
  seeds = tuple(
    _require_int(path, value, f"collection.seeds[{index}]", minimum=0)
    for index, value in enumerate(seeds_raw)
  )
  if len(set(seeds)) != len(seeds):
    _fail(path, "collection.seeds contains duplicate seed values")
  expected_seeds = tuple(base_seed + index for index in range(num_seeds))
  if seeds != expected_seeds:
    _fail(
      path,
      f"collection.seeds {list(seeds)} is not the finite schedule "
      f"{list(expected_seeds)}",
    )
  num_clips = _require_int(
    path, collection["num_clips"], "collection.num_clips", minimum=1
  )
  num_seed_runs = _require_int(
    path, collection["num_seed_runs"], "collection.num_seed_runs", minimum=1
  )
  if num_seed_runs != num_seeds:
    _fail(
      path,
      f"collection.num_seed_runs {num_seed_runs} != num_seeds {num_seeds}",
    )
  num_clip_rollouts = _require_int(
    path,
    collection["num_clip_rollouts"],
    "collection.num_clip_rollouts",
    minimum=1,
  )
  total_trajectories = _require_int(
    path,
    collection["total_trajectories"],
    "collection.total_trajectories",
    minimum=1,
  )
  max_steps_raw = collection["max_steps"]
  max_steps = (
    None
    if max_steps_raw is None
    else _require_int(path, max_steps_raw, "collection.max_steps", minimum=1)
  )
  sampling_mode = _require_string(
    path, collection["sampling_mode"], "collection.sampling_mode"
  )
  if sampling_mode not in SAMPLING_MODES:
    _fail(
      path,
      f"collection.sampling_mode must be one of {list(SAMPLING_MODES)}, "
      f"got {sampling_mode!r}",
    )
  scenario = _require_string(path, collection["scenario"], "collection.scenario")
  if scenario not in SCENARIOS:
    _fail(
      path,
      f"collection.scenario must be one of {list(SCENARIOS)}, got {scenario!r}",
    )
  push_robot_event = collection["push_robot_event"]
  if not isinstance(push_robot_event, bool):
    _fail(
      path,
      f"collection.push_robot_event must be a boolean, got {push_robot_event!r}",
    )
  disturbance = collection["disturbance"]
  if scenario == SCENARIO_BASELINE:
    if push_robot_event:
      _fail(path, "baseline scenario must not enable push_robot_event")
    if disturbance is not None:
      _fail(path, "baseline scenario must have disturbance=null")
  else:
    if not push_robot_event:
      _fail(path, "disturbed scenario must enable push_robot_event")
    _require_object(path, disturbance, "collection.disturbance")
  clips_raw = _require_list(path, manifest["clips"], "clips")
  if len(clips_raw) != num_clip_rollouts:
    _fail(
      path,
      f"clips has {len(clips_raw)} entries, expected num_clip_rollouts "
      f"{num_clip_rollouts}",
    )
  if num_clip_rollouts != num_clips * num_seeds:
    _fail(
      path,
      f"num_clip_rollouts {num_clip_rollouts} != num_clips {num_clips} * "
      f"num_seeds {num_seeds}",
    )
  if total_trajectories != num_clip_rollouts * num_envs:
    _fail(
      path,
      f"total_trajectories {total_trajectories} != num_clip_rollouts "
      f"{num_clip_rollouts} * num_envs {num_envs}",
    )
  collection_dir = path.parent.resolve()
  clips = tuple(
    _validated_clip(
      path,
      collection_dir,
      position,
      raw_clip,
      manifest_joint_names=joint_names,
      manifest_control_dt_s=control_dt_s,
      collection=collection,
      seeds=seeds,
      num_clips=num_clips,
      max_steps=max_steps,
    )
    for position, raw_clip in enumerate(clips_raw)
  )
  if any(clip.trajectories != num_envs for clip in clips):
    _fail(
      path,
      f"every clip must contain num_envs trajectories ({num_envs})",
    )
  if sum(clip.trajectories for clip in clips) != total_trajectories:
    _fail(
      path,
      f"clip trajectories sum to {sum(clip.trajectories for clip in clips)}, "
      f"expected total_trajectories={total_trajectories}",
    )
  combos = {(clip.run_index, clip.index) for clip in clips}
  if len(combos) != len(clips):
    _fail(path, "clips contains a duplicate (run_index, index) entry")
  expected_combos = {
    (run_index, clip_index)
    for run_index in range(num_seeds)
    for clip_index in range(num_clips)
  }
  if combos != expected_combos:
    _fail(
      path,
      "clips does not cover every (run_index, index) pair exactly once",
    )
  outputs = [clip.output for clip in clips]
  if len(set(outputs)) != len(outputs):
    _fail(path, "clips contains a duplicate output path")
  group = f"{sampling_mode}/{scenario}"
  return ManifestInfo(
    input_index=input_index,
    path=path,
    sha256=sha256,
    size_bytes=size_bytes,
    schema_version=schema_version,
    kind=kind,
    task=task,
    group=group,
    sampling_mode=sampling_mode,
    scenario=scenario,
    control_dt_s=control_dt_s,
    joint_names=joint_names,
    num_envs=num_envs,
    seeds=seeds,
    clips=clips,
  )


def _read_npz(path: Path) -> tuple[str, int, dict[str, np.ndarray]]:
  """Hash and load one NPZ from one byte buffer, once per array."""
  payload = path.read_bytes()
  sha256 = hashlib.sha256(payload).hexdigest()
  try:
    archive = np.load(io.BytesIO(payload), allow_pickle=False)
  except Exception as exc:
    raise ValueError(f"{path}: cannot open collection NPZ: {exc}") from exc
  with archive:
    missing = [name for name in REQUIRED_ARRAYS if name not in archive.files]
    if missing:
      _fail(
        path,
        f"collection NPZ is missing required array(s): {', '.join(missing)}",
      )
    try:
      arrays = {name: archive[name] for name in REQUIRED_ARRAYS}
    except Exception as exc:
      raise ValueError(f"{path}: cannot read collection NPZ array: {exc}") from exc
  return sha256, len(payload), arrays


def _require_integer_vector(
  source: Path, name: str, array: np.ndarray, *, length: int | None = None
) -> None:
  if array.ndim != 1:
    _fail(source, f"{name} must be one-dimensional, got shape {array.shape}")
  if length is not None and array.shape[0] != length:
    _fail(source, f"{name} has {array.shape[0]} entries, expected {length}")
  if not np.issubdtype(array.dtype, np.integer):
    _fail(source, f"{name} must have an integer dtype, got {array.dtype}")


def _validate_arrays(
  clip: ClipInfo,
  num_envs: int,
  path: Path,
  arrays: dict[str, np.ndarray],
) -> tuple[
  np.ndarray,
  np.ndarray,
  np.ndarray,
  np.ndarray,
  np.ndarray,
  np.ndarray,
  np.ndarray,
]:
  """Validate one NPZ contract and return the arrays needed for decisions."""
  root_quat = arrays["root_quat_wxyz"]
  joint_pos = arrays["joint_pos"]
  joint_vel = arrays["joint_vel"]
  if root_quat.ndim != 2 or root_quat.shape[1] != 4:
    _fail(
      path,
      f"root_quat_wxyz must have shape [rows, 4], got {root_quat.shape}",
    )
  rows = int(root_quat.shape[0])
  if rows < 1:
    _fail(path, "root_quat_wxyz must contain at least one row")
  joint_dim = len(clip.joint_names)
  for name, array in (("joint_pos", joint_pos), ("joint_vel", joint_vel)):
    if array.shape != (rows, joint_dim):
      _fail(
        path,
        f"{name} must have shape {(rows, joint_dim)}, got {array.shape}",
      )
  for name, array in (
    ("root_quat_wxyz", root_quat),
    ("joint_pos", joint_pos),
    ("joint_vel", joint_vel),
  ):
    if not np.issubdtype(array.dtype, np.floating):
      _fail(path, f"{name} must have a floating dtype, got {array.dtype}")
  tick = arrays["tick"]
  time_s = arrays["time_s"]
  env_index = arrays["env_index"]
  _require_integer_vector(path, "tick", tick, length=rows)
  _require_integer_vector(path, "env_index", env_index, length=rows)
  if time_s.ndim != 1 or time_s.shape[0] != rows:
    _fail(
      path,
      f"time_s must have shape [{rows}], got {time_s.shape}",
    )
  if not np.issubdtype(time_s.dtype, np.floating):
    _fail(path, f"time_s must have a floating dtype, got {time_s.dtype}")
  outcomes = arrays["trajectory_outcome"]
  _require_integer_vector(path, "trajectory_offsets", arrays["trajectory_offsets"])
  _require_integer_vector(path, "trajectory_env_index", arrays["trajectory_env_index"])
  if outcomes.ndim != 1 or outcomes.dtype.kind != "U":
    _fail(
      path,
      f"trajectory_outcome must be a one-dimensional unicode array, got "
      f"shape {outcomes.shape} and dtype {outcomes.dtype}",
    )
  trajectories = int(outcomes.shape[0])
  if trajectories != clip.trajectories:
    _fail(
      path,
      f"trajectory_outcome has {trajectories} entries, expected {clip.trajectories}",
    )
  for name in (
    "trajectory_steps",
    "trajectory_initial_frame",
    "trajectory_seed",
    "trajectory_run_index",
    "trajectory_push_count",
  ):
    _require_integer_vector(path, name, arrays[name], length=trajectories)
  scenario_array = arrays["trajectory_scenario"]
  if scenario_array.ndim != 1 or scenario_array.shape[0] != trajectories:
    _fail(
      path,
      f"trajectory_scenario has shape {scenario_array.shape}, expected "
      f"[{trajectories}]",
    )
  if scenario_array.dtype.kind != "U":
    _fail(
      path,
      f"trajectory_scenario must be a unicode array, got {scenario_array.dtype}",
    )
  _require_integer_vector(
    path, "push_offsets", arrays["push_offsets"], length=trajectories + 1
  )
  _require_integer_vector(path, "push_tick", arrays["push_tick"])
  _require_integer_vector(path, "push_env_index", arrays["push_env_index"])
  offsets = arrays["trajectory_offsets"].astype(np.int64, copy=False)
  if offsets.shape != (trajectories + 1,):
    _fail(
      path,
      f"trajectory_offsets must have shape [{trajectories + 1}], got {offsets.shape}",
    )
  if int(offsets[0]) != 0:
    _fail(path, f"trajectory_offsets must start at 0, got {int(offsets[0])}")
  if np.any(np.diff(offsets) < 1):
    _fail(path, "trajectory_offsets must increase by at least one row")
  if int(offsets[-1]) != rows:
    _fail(
      path,
      f"trajectory_offsets ends at {int(offsets[-1])}, expected rows={rows}",
    )
  if clip.rows != rows:
    _fail(
      path,
      f"manifest rows {clip.rows} does not match NPZ rows {rows}",
    )
  computed_steps = np.diff(offsets) - 1
  steps = arrays["trajectory_steps"].astype(np.int64, copy=False)
  if np.any(steps < 0):
    _fail(path, "trajectory_steps must be non-negative")
  if not np.array_equal(steps, computed_steps):
    _fail(path, "trajectory_steps does not match trajectory_offsets")
  env_rows = arrays["trajectory_env_index"]
  if np.any(env_rows < 0) or np.any(env_rows >= num_envs):
    _fail(path, "trajectory_env_index is outside the collection environment range")
  if len(set(int(value) for value in env_rows)) != trajectories:
    _fail(path, "trajectory_env_index contains duplicate environment rows")
  if np.any(arrays["trajectory_seed"] != clip.seed):
    _fail(path, "trajectory_seed does not match manifest clip seed")
  if np.any(arrays["trajectory_run_index"] != clip.run_index):
    _fail(path, "trajectory_run_index does not match manifest clip run_index")
  if any(str(value) != clip.scenario for value in scenario_array):
    _fail(path, "trajectory_scenario does not match manifest clip scenario")
  initial_frames = arrays["trajectory_initial_frame"]
  if np.any(initial_frames < 0) or np.any(initial_frames >= clip.frames):
    _fail(path, "trajectory_initial_frame is outside the source clip range")
  push_offsets = arrays["push_offsets"].astype(np.int64, copy=False)
  if int(push_offsets[0]) != 0:
    _fail(path, "push_offsets must start at 0")
  if np.any(np.diff(push_offsets) < 0):
    _fail(path, "push_offsets must not decrease")
  if int(push_offsets[-1]) != int(arrays["push_tick"].shape[0]):
    _fail(path, "push_offsets does not end at the push_tick length")
  push_counts = np.diff(push_offsets)
  trajectory_push_count = arrays["trajectory_push_count"].astype(np.int64, copy=False)
  if not np.array_equal(trajectory_push_count, push_counts):
    _fail(path, "trajectory_push_count does not match push_offsets")
  if int(push_counts.sum()) != clip.pushes_observed:
    _fail(
      path,
      f"manifest pushes_observed {clip.pushes_observed} does not match "
      f"NPZ push count {int(push_counts.sum())}",
    )
  if clip.scenario == SCENARIO_BASELINE and clip.pushes_observed != 0:
    _fail(path, "baseline scenario must not contain recorded pushes")
  outcome_counts = {outcome: 0 for outcome in OUTCOMES}
  for value in outcomes:
    label = str(value)
    if label not in outcome_counts:
      _fail(path, f"unknown trajectory outcome {label!r}")
    outcome_counts[label] += 1
  if outcome_counts != clip.outcomes:
    _fail(
      path,
      f"manifest outcomes {clip.outcomes} does not match NPZ outcomes {outcome_counts}",
    )
  push_tick = arrays["push_tick"]
  push_env_index = arrays["push_env_index"]
  for index in range(trajectories):
    start = int(offsets[index])
    stop = int(offsets[index + 1])
    expected_ticks = np.arange(stop - start, dtype=tick.dtype)
    if not np.array_equal(tick[start:stop], expected_ticks):
      _fail(
        path,
        f"trajectory {index} tick slice is not the contiguous sequence "
        f"0..{stop - start - 1}",
      )
    expected_env = np.full(stop - start, int(env_rows[index]), dtype=env_index.dtype)
    if not np.array_equal(env_index[start:stop], expected_env):
      _fail(
        path,
        f"trajectory {index} env_index does not match trajectory_env_index",
      )
    expected_time = tick[start:stop].astype(np.float64) * clip.control_dt_s
    if not np.allclose(time_s[start:stop], expected_time, rtol=1e-5, atol=1e-7):
      _fail(
        path,
        f"trajectory {index} time_s is inconsistent with tick and control_dt_s",
      )
    push_start = int(push_offsets[index])
    push_stop = int(push_offsets[index + 1])
    if push_stop > push_start:
      trajectory_pushes = push_tick[push_start:push_stop]
      if np.any(trajectory_pushes < 0) or np.any(trajectory_pushes > int(steps[index])):
        _fail(
          path,
          f"trajectory {index} push_tick is outside its recorded tick range",
        )
      expected_push_env = np.full(
        push_stop - push_start, int(env_rows[index]), dtype=push_env_index.dtype
      )
      if not np.array_equal(push_env_index[push_start:push_stop], expected_push_env):
        _fail(
          path,
          f"trajectory {index} push_env_index does not match its environment row",
        )
  return root_quat, joint_pos, joint_vel, outcomes, offsets, env_rows, steps


def _trajectory_records(
  arrays: tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
  ],
) -> list[dict[str, Any]]:
  """Evaluate each trajectory once and return compact range/count records."""
  root_quat, joint_pos, joint_vel, outcomes, offsets, env_rows, _ = arrays
  records: list[dict[str, Any]] = []
  for index in range(len(outcomes)):
    start = int(offsets[index])
    stop = int(offsets[index + 1])
    trajectory_rows = stop - start
    outcome = str(outcomes[index])
    if outcome != OUTCOME_COMPLETE:
      reason: str | None = REASON_OUTCOME
    elif not (
      np.isfinite(root_quat[start:stop]).all()
      and np.isfinite(joint_pos[start:stop]).all()
      and np.isfinite(joint_vel[start:stop]).all()
    ):
      reason = REASON_NONFINITE
    elif trajectory_rows < WINDOW_FRAMES:
      reason = REASON_TOO_SHORT
    else:
      reason = None
    candidate_windows = max(trajectory_rows - WINDOW_FRAMES + 1, 0)
    retained_windows = candidate_windows if reason is None else 0
    rejected_windows = candidate_windows - retained_windows
    status = "accepted" if reason is None else "rejected"
    window_start_range = (
      None if reason is not None else {"start": start, "stop": start + retained_windows}
    )
    records.append(
      {
        "trajectory_index": index,
        "env_index": int(env_rows[index]),
        "row_start": start,
        "row_stop": stop,
        "rows": trajectory_rows,
        "steps": trajectory_rows - 1,
        "outcome": outcome,
        "status": status,
        "reason": reason,
        "candidate_windows": candidate_windows,
        "retained_windows": retained_windows,
        "rejected_windows": rejected_windows,
        "window_start_range": window_start_range,
      }
    )
  return records


def _new_totals() -> dict[str, Any]:
  return {
    "trajectories": 0,
    "accepted_trajectories": 0,
    "rejected_trajectories": 0,
    "trajectories_with_candidate_windows": 0,
    "candidate_windows": 0,
    "retained_windows": 0,
    "rejected_windows": 0,
    "outcomes": {outcome: 0 for outcome in OUTCOMES},
    "reasons": {reason: 0 for reason in REJECTION_REASONS},
  }


def _add_record(totals: dict[str, Any], record: dict[str, Any]) -> None:
  totals["trajectories"] += 1
  totals["candidate_windows"] += int(record["candidate_windows"])
  totals["retained_windows"] += int(record["retained_windows"])
  totals["rejected_windows"] += int(record["rejected_windows"])
  totals["outcomes"][str(record["outcome"])] += 1
  if record["status"] == "accepted":
    totals["accepted_trajectories"] += 1
  else:
    totals["rejected_trajectories"] += 1
    totals["reasons"][str(record["reason"])] += 1
  if int(record["candidate_windows"]) > 0:
    totals["trajectories_with_candidate_windows"] += 1


def _file_record(
  info: ManifestInfo,
  clip: ClipInfo,
  sha256: str,
  size_bytes: int,
  records: list[dict[str, Any]],
) -> dict[str, Any]:
  totals = _new_totals()
  for record in records:
    _add_record(totals, record)
  return {
    "input_index": info.input_index,
    "output": clip.output,
    "path": str(clip.path),
    "sha256": sha256,
    "bytes": size_bytes,
    "motion": clip.motion_name,
    "motion_file": clip.motion_file,
    "group": info.group,
    "sampling_mode": clip.sampling_mode,
    "scenario": clip.scenario,
    "run_index": clip.run_index,
    "seed": clip.seed,
    "rows": clip.rows,
    "trajectories": clip.trajectories,
    **totals,
    "trajectory_records": records,
  }


def _build_report(config: GateConfig) -> tuple[Path, dict[str, Any]]:
  output_dir, manifest_paths = validate_config(config)
  infos: list[ManifestInfo] = []
  seen_manifest_hashes: set[str] = set()
  for input_index, path in enumerate(manifest_paths):
    info = _load_manifest(input_index, path)
    if info.sha256 in seen_manifest_hashes:
      raise ValueError(f"duplicate manifest contents: {path}")
    seen_manifest_hashes.add(info.sha256)
    if infos and (
      info.joint_names != infos[0].joint_names
      or info.control_dt_s != infos[0].control_dt_s
    ):
      _fail(path, "joint order or control timing differs across input manifests")
    infos.append(info)

  files: list[dict[str, Any]] = []
  input_records: list[dict[str, Any]] = []
  totals = _new_totals()
  total_rows = 0
  seen_sources: dict[Path, str] = {}
  seen_source_hashes: dict[str, str] = {}
  group_totals: dict[str, dict[str, Any]] = {}
  motion_totals: dict[str, dict[str, Any]] = {}
  reason_totals: dict[str, dict[str, Any]] = {
    reason: {
      "reason": reason,
      "trajectories": 0,
      "candidate_windows": 0,
      "rejected_windows": 0,
    }
    for reason in REJECTION_REASONS
  }

  for info in infos:
    input_totals = _new_totals()
    for clip in info.clips:
      if clip.path in seen_sources:
        raise ValueError(
          f"duplicate collection NPZ source: {clip.path} "
          f"(also from {seen_sources[clip.path]})"
        )
      sha256, size_bytes, arrays = _read_npz(clip.path)
      if sha256 in seen_source_hashes:
        raise ValueError(
          f"duplicate collection NPZ contents: {clip.path} "
          f"(also from {seen_source_hashes[sha256]})"
        )
      seen_sources[clip.path] = str(info.path)
      seen_source_hashes[sha256] = str(clip.path)
      validated = _validate_arrays(clip, info.num_envs, clip.path, arrays)
      records = _trajectory_records(validated)
      del arrays, validated
      file_record = _file_record(info, clip, sha256, size_bytes, records)
      files.append(file_record)
      total_rows += clip.rows
      _add_file_totals(input_totals, file_record)
      _add_file_totals(totals, file_record)
      group = group_totals.setdefault(
        info.group,
        {
          "group": info.group,
          "input_indexes": [],
          "files": 0,
          **_new_totals(),
        },
      )
      if info.input_index not in group["input_indexes"]:
        group["input_indexes"].append(info.input_index)
      group["files"] += 1
      _add_file_totals(group, file_record)
      motion = motion_totals.setdefault(
        clip.motion_name,
        {
          "motion": clip.motion_name,
          "source_files": [],
          "groups": [],
          "files": 0,
          **_new_totals(),
        },
      )
      if clip.motion_file not in motion["source_files"]:
        motion["source_files"].append(clip.motion_file)
      if info.group not in motion["groups"]:
        motion["groups"].append(info.group)
      motion["files"] += 1
      _add_file_totals(motion, file_record)
      for record in records:
        if record["status"] != "rejected":
          continue
        reason_entry = reason_totals[str(record["reason"])]
        reason_entry["trajectories"] += 1
        reason_entry["candidate_windows"] += int(record["candidate_windows"])
        reason_entry["rejected_windows"] += int(record["rejected_windows"])
    input_records.append(
      {
        "input_index": info.input_index,
        "manifest": str(info.path),
        "manifest_sha256": info.sha256,
        "manifest_bytes": info.size_bytes,
        "schema_version": info.schema_version,
        "kind": info.kind,
        "complete": True,
        "group": info.group,
        "sampling_mode": info.sampling_mode,
        "scenario": info.scenario,
        "task": info.task,
        "control_dt_s": info.control_dt_s,
        "joint_names": list(info.joint_names),
        "num_envs": info.num_envs,
        "seeds": list(info.seeds),
        "npz_files": len(info.clips),
        **_copy_totals(input_totals),
      }
    )

  _check_totals(totals)
  totals_record = {
    "inputs": len(input_records),
    "files": len(files),
    "rows": total_rows,
    **_copy_totals(totals),
  }
  report = {
    "schema_version": INDEX_SCHEMA_VERSION,
    "kind": INDEX_KIND,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "policy": _policy_document(),
    "window_contract": _window_contract_document(),
    "inputs": input_records,
    "files": files,
    "totals": totals_record,
    "by_group": list(group_totals.values()),
    "by_motion": list(motion_totals.values()),
    "by_reason": list(reason_totals.values()),
    "count_identities": {
      "trajectories": {
        "accepted": totals["accepted_trajectories"],
        "rejected": totals["rejected_trajectories"],
        "total": totals["trajectories"],
        "checked": True,
      },
      "windows": {
        "retained": totals["retained_windows"],
        "rejected": totals["rejected_windows"],
        "candidate": totals["candidate_windows"],
        "checked": True,
      },
    },
  }
  return output_dir, report


def _add_file_totals(target: dict[str, Any], file_record: dict[str, Any]) -> None:
  target["trajectories"] += int(file_record["trajectories"])
  target["accepted_trajectories"] += int(file_record["accepted_trajectories"])
  target["rejected_trajectories"] += int(file_record["rejected_trajectories"])
  target["trajectories_with_candidate_windows"] += int(
    file_record["trajectories_with_candidate_windows"]
  )
  target["candidate_windows"] += int(file_record["candidate_windows"])
  target["retained_windows"] += int(file_record["retained_windows"])
  target["rejected_windows"] += int(file_record["rejected_windows"])
  for outcome in OUTCOMES:
    target["outcomes"][outcome] += int(file_record["outcomes"][outcome])
  for reason in REJECTION_REASONS:
    target["reasons"][reason] += int(file_record["reasons"][reason])


def _copy_totals(totals: dict[str, Any]) -> dict[str, Any]:
  return {
    key: (dict(value) if isinstance(value, dict) else value)
    for key, value in totals.items()
  }


def _check_totals(totals: dict[str, Any]) -> None:
  if (
    totals["accepted_trajectories"] + totals["rejected_trajectories"]
    != totals["trajectories"]
  ):
    raise RuntimeError("internal error: trajectory counts do not reconcile")
  if (
    totals["retained_windows"] + totals["rejected_windows"]
    != totals["candidate_windows"]
  ):
    raise RuntimeError("internal error: window counts do not reconcile")
  if sum(totals["reasons"].values()) != totals["rejected_trajectories"]:
    raise RuntimeError("internal error: rejection reasons do not reconcile")
  if sum(totals["outcomes"].values()) != totals["trajectories"]:
    raise RuntimeError("internal error: outcome counts do not reconcile")


def _policy_document() -> dict[str, Any]:
  return {
    "name": "complete_finite_raw_41row",
    "collection_schema_version": COLLECTION_SCHEMA_VERSION,
    "outcome_requirement": OUTCOME_COMPLETE,
    "rejection_reasons_in_precedence_order": [
      {
        "reason": REASON_OUTCOME,
        "condition": (
          "stored trajectory_outcome is failure or budget; the whole "
          "trajectory is excluded"
        ),
      },
      {
        "reason": REASON_NONFINITE,
        "condition": (
          "any value in root_quat_wxyz, joint_pos, or joint_vel inside the "
          "trajectory row range is non-finite; the whole trajectory is excluded"
        ),
      },
      {
        "reason": REASON_TOO_SHORT,
        "condition": (
          f"fewer than {WINDOW_FRAMES} rows, so no complete "
          f"{WINDOW_FRAMES}-row window exists"
        ),
      },
    ],
    "conservative_behavior": [
      "A failed or budget-stopped trajectory is rejected as a whole; no "
      "prefix or suffix is salvaged.",
      "A complete trajectory with any non-finite required raw sample is "
      "rejected as a whole; no local windows around corrupt samples are "
      "salvaged.",
      "A complete trajectory shorter than one window is a valid raw record "
      "but is rejected with reason too_short.",
      "Push exposure is not required. Complete disturbed trajectories are "
      "retained with or without pushes.",
      "Outcome labels are stored labels. A failure is not reclassified from "
      "clip-end length, and a complete label is not proof of perfect tracking.",
    ],
    "excluded_checks": [
      "No height, tilt, contact, tracking-error, quaternion-normalization, or "
      "push-severity threshold is applied.",
    ],
  }


def _window_contract_document() -> dict[str, Any]:
  return {
    "history_frames": HISTORY_FRAMES,
    "current_frames": CURRENT_FRAMES,
    "future_frames": FUTURE_FRAMES,
    "window_frames": WINDOW_FRAMES,
    "row_span": "Each window covers raw rows [start, start + 41).",
    "candidate_count_formula": "max(trajectory_rows - 41 + 1, 0)",
    "window_start_range_semantics": (
      "Half-open [start, stop) range of global NPZ row indices s; each s "
      "denotes the 41 raw rows [s, s + 41)."
    ),
    "padding": "none",
    "cross_trajectory_windows": False,
    "materialized": False,
  }


def _publish_index(output_dir: Path, report: dict[str, Any]) -> None:
  """Create the new output directory and publish a completed index."""
  output_dir.mkdir(parents=True, exist_ok=False)
  temporary = output_dir / f"{INDEX_FILENAME}.tmp"
  temporary.write_text(
    json.dumps(report, separators=(",", ":")) + "\n", encoding="utf-8"
  )
  os.replace(temporary, output_dir / INDEX_FILENAME)


if __name__ == "__main__":
  main()
