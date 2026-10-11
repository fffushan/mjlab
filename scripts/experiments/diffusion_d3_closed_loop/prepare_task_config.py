#!/usr/bin/env python3
"""Prepare a D4 sparse-task config from an existing qualified baseline recording.

The tool extracts only the *chosen sparse* target world states from an existing
qualified recording.  It never samples a dense motion, never fetches a reference
at a control tick and never synthesizes a state: each target is copied verbatim
from the recorded row whose absolute control tick was selected.  Two recording
layouts are accepted, exclusively:

* ``--recording-dir`` reads an actual D3 closed-loop run (``summary.json`` plus
  ``ticks.jsonl``).  Only ``baseline`` rows are used, the summary must be a
  completed paired non-synthetic closed-loop attempt whose baseline branch is
  the frozen VAE reference path, and the requested member/seed/start-frame must
  match the recorded scenario.  This is the layout of the archived
  ``diffusion-d3-recorded-comparison-*`` baselines.
* ``--source-shard`` reads a raw D1 NPZ shard written by the frozen collector,
  optionally bound to its store manifest.  That layout carries weaker recorded
  provenance, so its output is honestly declared ``declared`` rather than
  ``verified``.  The loader preserves that qualification in the run report,
  but it does not block execution.

Qualification is checked, not assumed: the payload is validated by the strict
task loader -- including a re-read of the recording bytes, the target states and
the scenario -- before it is written atomically to a new path.  A scenario
mismatch is refused: no transform is implemented here.

``--diagnostic`` instead writes a clearly labeled ``synthetic_fixture`` /
``unverified`` config.  It can never claim qualified physical provenance; the
qualification remains visible in the task report even when the run is executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from mjlab.tasks.tracking.diffusion.conditioned_continuation import (
  DEFAULT_FRAME,
  DEFAULT_TICK_ORIGIN,
  TASK_SCHEMA,
)
from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT, DiffusionContract

# The strict loader is the single source of truth for the task format and for
# source re-verification; this tool never writes a payload it does not accept.
_LOADER_NAME = "mjlab_d4_sparse_task_config"
_LOADER_PATH = Path(__file__).resolve().with_name("task_config.py")


def _loader() -> Any:
  existing = sys.modules.get(_LOADER_NAME)
  if existing is not None:
    return existing
  spec = spec_from_file_location(_LOADER_NAME, _LOADER_PATH)
  if spec is None or spec.loader is None:
    raise PreparationError(f"task loader cannot be imported: {_LOADER_PATH}")
  module = module_from_spec(spec)
  sys.modules[_LOADER_NAME] = module
  spec.loader.exec_module(module)
  return module


_STATE_KEYS = (
  "root_position",
  "root_quaternion_wxyz",
  "root_linear_velocity",
  "root_angular_velocity",
  "body_positions",
  "body_linear_velocities",
)
_STORE_FORMAT = "mjlab-x2-diffusion-raw-v1"
_EXPECTED_FPS = 50.0


class PreparationError(ValueError):
  """A preparation request or its source recording failed verification."""


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _ticks_argument(value: str) -> tuple[int, ...]:
  if not value:
    return ()
  ticks: list[int] = []
  for item in value.split(","):
    text = item.strip()
    if not text:
      raise PreparationError("tick list contains an empty entry")
    try:
      ticks.append(int(text))
    except ValueError as exc:
      raise PreparationError(f"tick {text!r} is not an integer control tick") from exc
  return tuple(ticks)


def _load_rows(shard: Path) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
  try:
    with np.load(shard, allow_pickle=False) as data:
      arrays = {key: np.asarray(data[key]) for key in (*_STATE_KEYS, "metadata")}
  except (OSError, KeyError, ValueError) as exc:
    raise PreparationError(f"recording shard {shard} is unreadable: {exc}") from exc
  rows = int(arrays["root_position"].shape[0])
  if rows <= 0:
    raise PreparationError(f"recording shard {shard} has no rows")
  metadata: list[dict[str, Any]] = []
  for index in range(rows):
    try:
      item = json.loads(str(arrays["metadata"][index]))
    except json.JSONDecodeError as exc:
      raise PreparationError(f"row {index} metadata is not valid JSON") from exc
    if not isinstance(item, dict):
      raise PreparationError(f"row {index} metadata is not a mapping")
    metadata.append(item)
  return arrays, metadata


def build_task_payload(
  *,
  lookup: Callable[[int, str], dict[str, Any]],
  member: str,
  seed: int,
  start_frame: int,
  mode: str,
  keyframe_ticks: tuple[int, ...],
  goal_ticks: tuple[int, ...],
  strength: float,
  weights: dict[str, float] | None,
  contract: DiffusionContract,
  source: dict[str, Any],
  verification: dict[str, Any],
) -> dict[str, Any]:
  """Assemble the task payload from already-verified recorded rows."""
  target = lookup

  payload: dict[str, Any] = {
    "schema": TASK_SCHEMA,
    "mode": mode,
    "frame": DEFAULT_FRAME,
    "tick_origin": DEFAULT_TICK_ORIGIN,
    "control_period_seconds": contract.period_seconds,
    "contract_identity": contract.identity_hash(),
    "body_order": list(contract.body_names),
    "source": {
      **source,
      "member": member,
      "seed": int(seed),
      "start_frame": int(start_frame),
      "verification": verification["verification"],
      "verification_detail": {
        key: value for key, value in verification.items() if key != "verification"
      },
    },
    "strength": float(strength),
    "keyframes": [target(tick, "keyframe") for tick in keyframe_ticks],
    "goals": [target(tick, "goal") for tick in goal_ticks],
  }
  if weights is not None:
    payload["weights"] = dict(weights)
  return payload


def recording_lookup(
  recording_dir: Path, *, contract: DiffusionContract
) -> tuple[Callable[[int, str], dict[str, Any]], dict[str, Any]]:
  """Return a target lookup over an actual D3 recording's baseline rows."""
  loader = _loader()
  ticks_path = recording_dir / "ticks.jsonl"
  summary_path = recording_dir / "summary.json"
  if not ticks_path.is_file() or not summary_path.is_file():
    raise PreparationError(
      f"recording directory {recording_dir} needs summary.json and ticks.jsonl"
    )
  try:
    recording = loader.read_d3_recording(
      ticks_path, summary_path=summary_path, contract=contract
    )
  except loader.TaskConfigError as exc:
    raise PreparationError(f"recording {recording_dir} is not usable: {exc}") from exc

  def lookup(tick: int, label: str) -> dict[str, Any]:
    try:
      state = recording.state_at(tick)
      row = recording.rows[int(tick)]
    except loader.TaskConfigError as exc:
      raise PreparationError(f"selected {label}: {exc}") from exc
    return {
      "tick": int(tick),
      "state": {
        key: np.asarray(getattr(state, key), dtype=np.float64).tolist()
        for key in _STATE_KEYS
      },
      "provenance": (
        f"{ticks_path}#branch=baseline,tick={int(tick)},segment={row.segment_id}"
      ),
    }

  source = {
    "kind": "baseline_recording",
    "format": "d3_closed_loop_recording",
    "qualification": recording.qualification,
    "verification": "preparation_tool:d3_recording+" + ",".join(recording.checks),
    "provenance": (
      "explicit sparse rows selected from a re-read D3 closed-loop baseline "
      "recording; digest and structural qualification checked, not an "
      "independent proof of physical execution"
    ),
    "path": str(ticks_path),
    "sha256": recording.ticks_sha256,
    "summary_path": str(summary_path),
    "summary_sha256": recording.summary_sha256,
    "member": recording.scenario.get("member"),
    "seed": recording.scenario.get("seed"),
    "start_frame": recording.scenario.get("start_frame"),
  }
  detail = {
    "checks": list(recording.checks),
    "scenario": dict(recording.scenario),
    "ticks_sha256": recording.ticks_sha256,
    "summary_sha256": recording.summary_sha256,
  }
  return lookup, {"source": source, "detail": detail}


def npz_lookup(
  source_shard: Path, *, store_root: Path | None, contract: DiffusionContract
) -> tuple[Callable[[int, str], dict[str, Any]], dict[str, Any]]:
  """Return a target lookup over a raw D1 NPZ shard (declared provenance)."""
  arrays, metadata = _load_rows(source_shard)
  rows_by_tick = {int(item["tick"]): index for index, item in enumerate(metadata)}
  digest = _sha256(source_shard)
  binding: dict[str, Any] | None = None
  if store_root is not None:
    loader = _loader()
    try:
      binding = loader.verify_store_binding(store_root, source_shard, contract)
    except loader.TaskConfigError as exc:
      raise PreparationError(f"store binding failed: {exc}") from exc

  def lookup(tick: int, label: str) -> dict[str, Any]:
    if tick not in rows_by_tick:
      raise PreparationError(
        f"selected {label} tick {tick} is not present in the recording"
      )
    index = rows_by_tick[tick]
    item = metadata[index]
    if item.get("segment_qualified") is not True:
      raise PreparationError(f"{label} tick {tick} row is not a qualified segment")
    if item.get("reset") or item.get("teleport") or item.get("reference_boundary"):
      raise PreparationError(f"{label} tick {tick} row is not a clean measured row")
    return {
      "tick": tick,
      "state": {
        key: np.asarray(arrays[key][index], dtype=np.float64).tolist()
        for key in _STATE_KEYS
      },
      "provenance": (
        f"{source_shard}#row={index},tick={tick},segment={item.get('segment_id')}"
      ),
    }

  return lookup, {
    "source": {
      "kind": "baseline_recording",
      "format": "d1_npz_shard",
      "qualification": "declared",
      "verification": "preparation_tool:npz_shard",
      "provenance": (
        "explicit sparse rows selected from a raw NPZ recording shard; its "
        "recorded provenance is weaker than a D3 recording, so it stays declared"
      ),
      "path": str(source_shard),
      "sha256": digest,
    },
    "detail": {"store_binding": binding, "format": "d1_npz_shard"},
  }


def prepare(
  *,
  source_shard: Path | None = None,
  recording_dir: Path | None = None,
  source_sha256: str | None,
  store_root: Path | None,
  member: str,
  seed: int,
  start_frame: int,
  mode: str,
  keyframe_ticks: tuple[int, ...],
  goal_ticks: tuple[int, ...],
  strength: float,
  weights: dict[str, float] | None,
  contract: DiffusionContract,
  diagnostic: bool,
) -> dict[str, Any]:
  """Verify the source recording and return a task payload plus its checks."""
  if (source_shard is None) == (recording_dir is None):
    raise PreparationError(
      "exactly one of --recording-dir or --source-shard must be given"
    )
  if recording_dir is not None:
    if diagnostic:
      raise PreparationError(
        "--diagnostic applies to a raw shard source; a D3 recording is verified"
      )
    lookup, checks = recording_lookup(recording_dir, contract=contract)
    scenario = dict(checks["detail"].get("scenario") or {})
    mismatches = [
      f"{name} {scenario.get(name)!r} != {expected!r}"
      for name, expected in (
        ("member", member),
        ("seed", seed),
        ("start_frame", start_frame),
      )
      if str(scenario.get(name)) != str(expected)
    ]
    if mismatches:
      raise PreparationError(
        "recorded scenario does not match the requested member/seed/start-frame: "
        + "; ".join(mismatches)
        + " (no source transform is implemented in this tool)"
      )
  else:
    assert source_shard is not None
    if not source_shard.is_file():
      raise PreparationError(f"source shard is not a readable file: {source_shard}")
    digest = _sha256(source_shard)
    if source_sha256 is not None and digest != source_sha256:
      raise PreparationError(
        f"source shard sha256 mismatch: expected {source_sha256}, got {digest}"
      )
    arrays, metadata = _load_rows(source_shard)
    ticks: list[int] = []
    for index, item in enumerate(metadata):
      if item.get("controller") != "vae":
        raise PreparationError(
          f"row {index} controller is {item.get('controller')!r}, expected 'vae'"
        )
      if item.get("segment_qualified") is not True:
        raise PreparationError(f"row {index} segment is not qualified")
      if item.get("reset") or item.get("teleport") or item.get("reference_boundary"):
        raise PreparationError(f"row {index} is not a clean measured row")
      if not np.isclose(
        float(item.get("fps", 0.0)), _EXPECTED_FPS, atol=1e-5, rtol=0.0
      ):
        raise PreparationError(f"row {index} is not a 50 Hz recording")
      ticks.append(int(item["tick"]))
    for earlier, later in zip(ticks, ticks[1:], strict=False):
      if earlier >= later:
        raise PreparationError("recorded ticks must strictly increase")
    row_provenance = metadata[0].get("provenance") or {}
    if not isinstance(row_provenance, dict):
      raise PreparationError("row provenance must be a string mapping")
    scenario = {
      "motion_id": str(metadata[0].get("motion_id", "")),
      "initial_seed": row_provenance.get("initial_seed"),
      "initial_start_frame": row_provenance.get("initial_start_frame"),
      "first_tick": ticks[0],
      "last_tick": ticks[-1],
    }
    mismatches = []
    if scenario["motion_id"] != member:
      mismatches.append(f"motion_id {scenario['motion_id']!r} != {member!r}")
    if str(scenario["initial_seed"]) != str(seed):
      mismatches.append(f"initial_seed {scenario['initial_seed']!r} != {seed!r}")
    if str(scenario["initial_start_frame"]) != str(start_frame):
      mismatches.append(
        f"initial_start_frame {scenario['initial_start_frame']!r} != {start_frame!r}"
      )
    if mismatches and not diagnostic:
      raise PreparationError(
        "recorded scenario does not match the requested member/seed/start-frame: "
        + "; ".join(mismatches)
        + " (no source transform is implemented in this tool)"
      )
    lookup, checks = npz_lookup(source_shard, store_root=store_root, contract=contract)
    if diagnostic:
      checks["source"]["kind"] = "synthetic_fixture"
      checks["source"]["qualification"] = "unverified"
      checks["source"]["verification"] = "preparation_tool:diagnostic"
      checks["source"]["provenance"] = (
        "hand-selected rows from an explicit recording shard; diagnostic only, "
        "not a qualified physical claim"
      )
    checks["detail"] = {
      **checks["detail"],
      "scenario": scenario,
      "scenario_mismatch": list(mismatches),
      "declared_sha256": source_sha256,
      "row_checks": {
        "controller": "vae",
        "segment_qualified": True,
        "clean_rows_only": True,
        "strictly_increasing_ticks": True,
        "fps": _EXPECTED_FPS,
      },
    }
  return build_task_payload(
    lookup=lookup,
    member=member,
    seed=seed,
    start_frame=start_frame,
    mode=mode,
    keyframe_ticks=keyframe_ticks,
    goal_ticks=goal_ticks,
    strength=strength,
    weights=weights,
    contract=contract,
    source=checks["source"],
    verification={
      "verification": checks["source"]["verification"],
      **checks["detail"],
    },
  )


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description=(
      "Extract chosen sparse D4 task targets from an existing qualified baseline "
      "recording.  This tool never samples motion and never runs a simulator."
    )
  )
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument(
    "--recording-dir",
    type=Path,
    default=None,
    help="an actual D3 closed-loop run directory (summary.json + ticks.jsonl)",
  )
  source.add_argument(
    "--source-shard",
    type=Path,
    default=None,
    help="a raw D1 NPZ shard (weaker recorded provenance; declared, not verified)",
  )
  parser.add_argument("--source-sha256", default=None)
  parser.add_argument("--store-root", type=Path, default=None)
  parser.add_argument("--contract", type=Path, default=None)
  parser.add_argument("--member", required=True)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--start-frame", type=int, required=True)
  parser.add_argument(
    "--mode", choices=("keyframe", "poseguided", "combined"), required=True
  )
  parser.add_argument("--keyframe-ticks", default="")
  parser.add_argument("--goal-ticks", default="")
  parser.add_argument("--strength", type=float, default=0.0)
  parser.add_argument("--root-position-weight", type=float, default=None)
  parser.add_argument("--root-orientation-weight", type=float, default=None)
  parser.add_argument("--body-positions-weight", type=float, default=None)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
    "--diagnostic",
    action="store_true",
    help="write a clearly unverified diagnostic config instead of a verified one",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  args = _parser().parse_args(argv)
  contract = (
    DEFAULT_CONTRACT
    if args.contract is None
    else DiffusionContract.from_yaml(args.contract)
  )
  weights: dict[str, float] | None = None
  if args.mode in {"poseguided", "combined"}:
    if None in (
      args.root_position_weight,
      args.root_orientation_weight,
      args.body_positions_weight,
    ):
      print("preparation failed: pose modes require all three explicit weights")
      return 1
    weights = {
      "root_position": float(args.root_position_weight),
      "root_orientation": float(args.root_orientation_weight),
      "body_positions": float(args.body_positions_weight),
    }
  try:
    loader = _loader()
    payload = prepare(
      source_shard=args.source_shard,
      recording_dir=args.recording_dir,
      source_sha256=args.source_sha256,
      store_root=args.store_root,
      member=args.member,
      seed=args.seed,
      start_frame=args.start_frame,
      mode=args.mode,
      keyframe_ticks=_ticks_argument(args.keyframe_ticks),
      goal_ticks=_ticks_argument(args.goal_ticks),
      strength=float(args.strength),
      weights=weights,
      contract=contract,
      diagnostic=bool(args.diagnostic),
    )
    # The strict loader must accept the exact bytes about to be written; a
    # preparation run can never emit an invalid or unverifiable task file.
    spec = loader.write_task_config(
      payload,
      args.output,
      contract=contract,
      expected_scenario=loader.ScenarioExpectation(
        member=args.member, seed=args.seed, start_frame=args.start_frame
      ),
    )
  except (PreparationError, ValueError, OSError) as exc:
    print(f"preparation failed: {exc}")
    return 1
  print(
    json.dumps(
      {
        "ok": True,
        "output": str(args.output),
        "config_sha256": spec.config_sha256,
        "qualification": spec.source.qualification,
        "verification": spec.source.verification,
        "source": spec.source.provenance,
        "keyframe_ticks": list(spec.keyframe_ticks),
        "goal_ticks": list(spec.goal_ticks),
      },
      indent=1,
      sort_keys=True,
    )
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
