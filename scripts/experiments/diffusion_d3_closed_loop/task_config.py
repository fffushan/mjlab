#!/usr/bin/env python3
"""Strict, bounded loader for the D4 opt-in sparse-task JSON.

The task file is a *separately versioned input*, not a revision of the frozen
learned contract.  It declares its own schema version, the frozen contract
identity it was authored against, the frozen body order, the control period, the
world frame and the absolute-tick origin, and then carries at most 64 scheduled
targets.  Nothing in it is executable: there are no dynamic imports, callables,
pickle payloads, target latents or encoder outputs, so a task file can only
describe sparse world-space targets.

Every rejection happens while loading, i.e. before any environment or runtime is
constructed: unknown or missing fields, duplicate JSON keys, duplicate or
too-densely-scheduled target ticks, boolean/non-integral ticks, non-finite or
ill-shaped states, legacy schemas, contract/body-order/period mismatches and
fabricated "verified" physical provenance on manual or synthetic sources.

The qualification of a recording-bound source is **recomputed here**, never
trusted from the file: the declared recording files are re-read with their
digests checked, their rows re-qualified, every target state re-extracted at its
tick, and the recorded member/seed/start-frame re-compared with the actual run
arguments.  A hand-written ``preparation_tool:`` prefix cannot therefore upgrade
a source, and a missing, modified or mismatched recording is refused.  Digest and
declared-record checks prove the bytes and the structural qualification of the
recording; they are not a cryptographic or independent authentication that the
recording reflects physical execution.

Targets are sparse by construction: the loader enforces the same 10-tick minimum
spacing as the component seams (0.2 s at the frozen 50 Hz) and a hard bound of 64
targets, so a dense reference-tracking schedule cannot be expressed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from mjlab.tasks.tracking.diffusion.conditioned_continuation import (
  DEFAULT_FRAME,
  DEFAULT_TICK_ORIGIN,
  MODES,
  TASK_SCHEMA,
  SparseTaskSpec,
  TaskConditioningError,
  TaskSource,
)
from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT, DiffusionContract
from mjlab.tasks.tracking.diffusion.inpainting import (
  TimedStateKeyframe,
)
from mjlab.tasks.tracking.diffusion.pose_costs import (
  PoseCostError,
  PoseCostWeights,
  TimedPoseGoal,
)
from mjlab.tasks.tracking.diffusion.state import StateValidationError, WorldState

MAX_TASK_BYTES = 1024 * 1024
MAX_TARGETS = 64
MINIMUM_TARGET_SPACING = 10
# Bounded reader for the declared recording files and the task config itself.
MAX_SOURCE_BYTES = 64 * 1024 * 1024
_RECORDING_SCHEMA = "mjlab-d3-closed-loop-evaluation-v1"
_RECORDING_FPS = 50.0
_STORE_FORMAT = "mjlab-x2-diffusion-raw-v1"
_D3_FORMAT = "d3_closed_loop_recording"
_NPZ_FORMAT = "d1_npz_shard"

_STATE_FIELDS = (
  "root_position",
  "root_quaternion_wxyz",
  "root_linear_velocity",
  "root_angular_velocity",
  "body_positions",
  "body_linear_velocities",
)

_TOP_FIELDS = frozenset(
  {
    "schema",
    "mode",
    "frame",
    "tick_origin",
    "control_period_seconds",
    "contract_identity",
    "body_order",
    "source",
    "strength",
    "weights",
    "keyframes",
    "goals",
  }
)
_TARGET_FIELDS = frozenset({"tick", "state", "provenance"})
_SOURCE_FIELDS = frozenset(
  {
    "kind",
    "qualification",
    "verification",
    "verification_detail",
    "provenance",
    "format",
    "path",
    "sha256",
    "summary_path",
    "summary_sha256",
    "store_root",
    "store_manifest_sha256",
    "member",
    "seed",
    "start_frame",
  }
)
_WEIGHT_FIELDS = frozenset(
  {"root_position", "root_orientation", "body_positions", "body_coefficients"}
)
# A task target is a world-space state only.  These names are rejected with an
# explicit message instead of a generic unknown-field error.
_FORBIDDEN_TARGET_FIELDS = (
  "latent",
  "latents",
  "vae_latent",
  "encoder",
  "encoding",
  "action",
  "actions",
  "reference",
  "reference_frame",
  "reward",
)


class TaskConfigError(ValueError):
  """A D4 task config file failed a strict load-time gate."""


@dataclass(frozen=True, slots=True)
class ScenarioExpectation:
  """The scenario the run will actually execute, for source re-binding."""

  member: str | None = None
  seed: int | None = None
  start_frame: int | None = None


def bounded_read(path: Path | str, limit: int, what: str) -> bytes:
  """Read a regular file with a hard bound, without trusting its size first."""
  resolved = Path(path)
  try:
    stat = resolved.stat()
  except OSError as exc:
    raise TaskConfigError(f"{what} {resolved} is not readable: {exc}") from exc
  if not resolved.is_file():
    raise TaskConfigError(f"{what} {resolved} is not a regular file")
  if stat.st_size > limit:
    raise TaskConfigError(
      f"{what} {resolved} is {stat.st_size} bytes, above the {limit}-byte bound"
    )
  try:
    with resolved.open("rb") as handle:
      data = handle.read(limit + 1)
  except OSError as exc:
    raise TaskConfigError(f"{what} {resolved} is unreadable: {exc}") from exc
  if len(data) > limit:
    raise TaskConfigError(f"{what} {resolved} exceeds the {limit}-byte bound")
  return data


@dataclass(frozen=True, slots=True)
class RecordingRow:
  """One qualified baseline row of a D3 closed-loop recording."""

  tick: int
  sim_time_s: float
  segment_id: str
  phase: str
  state: WorldState
  post_tick: int | None
  post_sim_time_s: float | None
  post_segment_id: str | None
  post_state: WorldState | None


@dataclass(frozen=True, slots=True)
class Recording:
  """A re-read, structurally qualified D3 baseline recording."""

  ticks_path: str
  ticks_sha256: str
  summary_path: str | None
  summary_sha256: str | None
  rows: Mapping[int, RecordingRow]
  scenario: Mapping[str, Any]
  checks: tuple[str, ...]
  qualification: str

  def state_at(self, tick: int) -> WorldState:
    """Return the measured baseline state at one absolute tick."""
    row = self.rows.get(int(tick))
    if row is None:
      raise TaskConfigError(
        f"recording {self.ticks_path} has no baseline row at tick {tick}"
      )
    return row.state


def _no_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
  result: dict[str, Any] = {}
  for key, value in pairs:
    if key in result:
      raise TaskConfigError(f"duplicate JSON key {key!r} is not accepted")
    result[key] = value
  return result


def _field_error(
  where: str,
  payload: object,
  allowed: frozenset[str],
  *,
  optional: frozenset[str] = frozenset(),
) -> None:
  if not isinstance(payload, Mapping):
    raise TaskConfigError(f"{where} must be a JSON object")
  if not all(isinstance(key, str) for key in payload):
    raise TaskConfigError(f"{where} must have string keys")
  payload = cast(Mapping[str, Any], payload)
  forbidden = sorted(set(payload) & set(_FORBIDDEN_TARGET_FIELDS))
  if forbidden:
    raise TaskConfigError(
      f"{where} must describe a world-space target only; {forbidden} is not a "
      "task input (no target latent and no encoder output)"
    )
  unknown = sorted(set(payload) - allowed)
  if unknown:
    raise TaskConfigError(f"{where} has unknown fields {unknown}")
  missing = sorted(allowed - set(payload) - set(optional))
  if missing:
    raise TaskConfigError(f"{where} is missing fields {missing}")


def _tick(value: object, where: str) -> int:
  if isinstance(value, bool) or not isinstance(value, int):
    raise TaskConfigError(f"{where} must be an integer control tick")
  if value < 0:
    raise TaskConfigError(f"{where} must be non-negative")
  return int(value)


def _number(value: object, where: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise TaskConfigError(f"{where} must be a finite number")
  number = float(value)
  if not np.isfinite(number):
    raise TaskConfigError(f"{where} must be finite")
  return number


def _text(value: object, where: str) -> str:
  if not isinstance(value, str) or not value.strip():
    raise TaskConfigError(f"{where} must be a non-empty string")
  return value


def _state(payload: object, where: str) -> WorldState:
  if not isinstance(payload, Mapping):
    raise TaskConfigError(f"{where} must be a mapping of the six physical arrays")
  _field_error(where, payload, frozenset(_STATE_FIELDS))
  payload = cast(Mapping[str, Any], payload)
  arrays: dict[str, np.ndarray] = {}
  for name in _STATE_FIELDS:
    value = payload[name]
    if not isinstance(value, list):
      raise TaskConfigError(f"{where}.{name} must be a JSON array")
    try:
      array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
      raise TaskConfigError(f"{where}.{name} is not numeric: {exc}") from exc
    if not np.isfinite(array).all():
      raise TaskConfigError(f"{where}.{name} contains non-finite values")
    arrays[name] = array
  try:
    world = WorldState(**arrays)
  except StateValidationError as exc:
    raise TaskConfigError(f"{where} is not a valid world state: {exc}") from exc
  if world.root_position.ndim != 1:
    raise TaskConfigError(f"{where} must be one unbatched measurement")
  return world


def _targets(
  payload: object, where: str, *, contract: DiffusionContract
) -> tuple[dict[str, Any], ...]:
  if not isinstance(payload, list):
    raise TaskConfigError(f"{where} must be a JSON array")
  targets: list[dict[str, Any]] = []
  for index, item in enumerate(payload):
    item_where = f"{where}[{index}]"
    _field_error(item_where, item, _TARGET_FIELDS)
    item = cast(Mapping[str, Any], item)  # validated by _field_error
    targets.append(
      {
        "tick": _tick(item["tick"], f"{item_where}.tick"),
        "provenance": _text(item["provenance"], f"{item_where}.provenance"),
        "state": _state(item["state"], f"{item_where}.state"),
      }
    )
  ticks = [target["tick"] for target in targets]
  if len(set(ticks)) != len(ticks):
    raise TaskConfigError(f"{where} contains duplicate target ticks")
  for earlier, later in zip(sorted(ticks), sorted(ticks)[1:], strict=False):
    if later - earlier < MINIMUM_TARGET_SPACING:
      raise TaskConfigError(
        f"{where} targets {earlier} and {later} are closer than the "
        f"{MINIMUM_TARGET_SPACING}-tick minimum spacing"
      )
  return tuple(targets)


def _recording_summary_checks(
  summary_bytes: bytes, *, contract: DiffusionContract
) -> dict[str, Any]:
  """Validate a recorded D3 summary and return its recorded scenario."""
  try:
    summary = json.loads(summary_bytes.decode("utf-8"))
  except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise TaskConfigError(f"recording summary is not valid JSON: {exc}") from exc
  if not isinstance(summary, Mapping):
    raise TaskConfigError("recording summary must be a JSON object")
  if summary.get("schema") != _RECORDING_SCHEMA:
    raise TaskConfigError(
      f"recording summary schema must be {_RECORDING_SCHEMA!r}, got "
      f"{summary.get('schema')!r}"
    )
  if summary.get("status") != "complete" or summary.get("paired") is not True:
    raise TaskConfigError("recording is not a complete paired attempt")
  if summary.get("valid") is not True:
    raise TaskConfigError("recording summary is not valid")
  if summary.get("closed_loop") is not True or summary.get("synthetic") is not False:
    raise TaskConfigError(
      "recording must be real closed-loop evidence (closed_loop true, synthetic false)"
    )
  if summary.get("errors") not in (None, []):
    raise TaskConfigError(
      f"recording summary reports errors: {summary.get('errors')!r}"
    )
  provenance = summary.get("provenance")
  if not isinstance(provenance, Mapping):
    raise TaskConfigError("recording summary has no provenance mapping")
  # No integrity gate: the runs in this checkout write `provenance.integrity: null`,
  # for every family, so requiring it rejected every possible task source.  The
  # recording is still checked below for real closed-loop evidence, the frozen
  # contract identity, the same VAE and checkpoint contract, the control period,
  # eight bootstrap pairs, the clip rate and a reference-only baseline branch.
  vae = provenance.get("vae")
  if not isinstance(vae, Mapping) or vae.get("sha256") != contract.vae_sha256:
    raise TaskConfigError("recording VAE differs from the frozen contract")
  checkpoint = provenance.get("checkpoint")
  recorded_identity = (
    None if not isinstance(checkpoint, Mapping) else checkpoint.get("contract_identity")
  )
  if recorded_identity not in (
    contract.identity_hash(),
    replace(contract, contract_sha256=None).identity_hash(),
  ):
    raise TaskConfigError(
      "recording was recorded against a different frozen contract identity"
    )
  budget = summary.get("budget")
  if not isinstance(budget, Mapping):
    raise TaskConfigError("recording summary has no budget mapping")
  period = budget.get("period_seconds")
  if not isinstance(period, (int, float)) or not np.isclose(
    float(period), contract.period_seconds, rtol=0.0, atol=1e-12
  ):
    raise TaskConfigError(
      f"recording control period {period!r} disagrees with the frozen contract"
    )
  if budget.get("bootstrap_pairs") != contract.past_steps:
    raise TaskConfigError("recording was not collected with eight bootstrap pairs")
  environment = provenance.get("environment")
  if not isinstance(environment, Mapping):
    raise TaskConfigError("recording summary has no environment provenance")
  if not np.isclose(
    float(environment.get("clip_fps", 0.0)), _RECORDING_FPS, rtol=0.0, atol=1e-5
  ):
    raise TaskConfigError("recording is not a 50 Hz clip")
  branches = summary.get("branches")
  baseline = branches.get("baseline") if isinstance(branches, Mapping) else None
  if not isinstance(baseline, Mapping):
    raise TaskConfigError("recording summary has no baseline branch")
  counters = baseline.get("counters")
  if not isinstance(counters, Mapping) or int(counters.get("denoiser", -1)) != 0:
    raise TaskConfigError(
      "recording baseline branch is not the frozen VAE reference path "
      "(expected zero denoiser calls)"
    )
  baseline_provenance = baseline.get("provenance")
  if not isinstance(baseline_provenance, Mapping):
    raise TaskConfigError("recording baseline branch has no provenance mapping")
  recorded_body_order = baseline_provenance.get("body_order")
  if recorded_body_order is not None:
    if tuple(recorded_body_order) != contract.body_names:
      raise TaskConfigError("recording body order differs from the frozen contract")
  elif not (
    baseline_provenance.get("live_contract_audited") is True
    and baseline_provenance.get("factory_trusted") is True
  ):
    # Historical D3 recordings bind order through the audited frozen contract
    # and trusted factory, but did not persist a separate body_order list.
    raise TaskConfigError("recording has no explicit or audited-contract body order")
  if baseline.get("errors") not in (None, []):
    raise TaskConfigError("recording baseline branch reports errors")
  if (
    baseline_provenance.get("synthetic") is not False
    or baseline_provenance.get("physical_runtime") != "mjlab"
  ):
    raise TaskConfigError(
      "recording baseline branch is not marked as real mjlab measurement data"
    )
  if baseline_provenance.get("untrusted_terminal_evidence") is not False:
    raise TaskConfigError(
      "recording baseline branch carries untrusted terminal evidence"
    )
  return {
    "member": environment.get("member"),
    "seed": environment.get("seed"),
    "start_frame": environment.get("start_frame"),
    "clip_frames": environment.get("clip_frames"),
    "contract_identity": str(recorded_identity),
    "completed_ticks": _tick(
      baseline.get("completed_ticks"), "baseline completed ticks"
    ),
    "attempted_ticks": _tick(
      baseline.get("attempted_ticks"), "baseline attempted ticks"
    ),
    "encoder_calls": _tick(counters.get("encoder"), "baseline encoder calls"),
    "decoder_calls": _tick(counters.get("decoder"), "baseline decoder calls"),
  }


def read_d3_recording(
  ticks_path: str | Path,
  *,
  summary_path: str | Path | None = None,
  contract: DiffusionContract = DEFAULT_CONTRACT,
) -> Recording:
  """Re-read and structurally qualify a real D3 closed-loop recording.

  Only ``baseline`` rows are read: a diffusion row can never become a verified
  VAE-baseline target.  The recorded summary must be a completed, paired,
  closed-loop, non-synthetic attempt whose baseline branch is the frozen VAE
  reference path (zero denoiser calls), and its recorded contract identity,
  control period, 50 Hz clip and scenario are re-read here.  Baseline rows must
  be contiguous in tick, segment and control period with terminal evidence, and
  every post/pre hand-over must agree.
  """
  ticks_file = Path(ticks_path)
  ticks_bytes = bounded_read(ticks_file, MAX_SOURCE_BYTES, "recording ticks file")
  checks = ["ticks_sha256"]
  scenario: dict[str, Any] = {}
  summary_sha: str | None = None
  summary_file = None if summary_path is None else Path(summary_path)
  if summary_file is not None:
    summary_bytes = bounded_read(
      summary_file, MAX_SOURCE_BYTES, "recording summary file"
    )
    summary_sha = hashlib.sha256(summary_bytes).hexdigest()
    scenario = _recording_summary_checks(summary_bytes, contract=contract)
    checks.extend(
      [
        "summary_sha256",
        "summary_contract_identity",
        "summary_pairing_and_closed_loop",
        "summary_baseline_vae_path",
      ]
    )
  rows: dict[int, RecordingRow] = {}
  try:
    text = ticks_bytes.decode("utf-8")
  except UnicodeDecodeError as exc:
    raise TaskConfigError(f"recording ticks file is not UTF-8: {exc}") from exc
  for index, line in enumerate(text.splitlines()):
    if not line.strip():
      continue
    try:
      payload = json.loads(line)
    except json.JSONDecodeError as exc:
      raise TaskConfigError(f"recording line {index} is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
      raise TaskConfigError(f"recording line {index} is not a JSON object")
    if payload.get("branch") != "baseline":
      # A diffusion row is never a verified VAE-baseline source.
      continue
    tick_payload = payload.get("tick")
    if not isinstance(tick_payload, Mapping):
      raise TaskConfigError(f"recording line {index} has no tick mapping")
    tick = _tick(tick_payload.get("tick"), f"recording line {index} tick")
    if tick in rows:
      raise TaskConfigError(f"recording contains duplicate baseline tick {tick}")
    phase = tick_payload.get("phase")
    if phase not in {"bootstrap", "baseline"}:
      raise TaskConfigError(
        f"recording tick {tick} phase {phase!r} is not a VAE baseline phase"
      )
    if tick_payload.get("completed") is not True:
      raise TaskConfigError(f"recording tick {tick} is not a completed row")
    if tick_payload.get("stop_reason") is not None:
      raise TaskConfigError(
        f"recording tick {tick} carries stop reason {tick_payload.get('stop_reason')!r}"
      )
    state = _state(tick_payload.get("state"), f"recording tick {tick} state")
    post = tick_payload.get("post")
    post_state = post_tick = post_time = post_segment = None
    if isinstance(post, Mapping) and post.get("state") is not None:
      post_state = _state(post.get("state"), f"recording tick {tick} post state")
      post_tick = _tick(post.get("tick"), f"recording tick {tick} post tick")
      post_time = _number(post.get("sim_time_s"), f"recording tick {tick} post time")
      post_segment = str(post.get("segment_id"))
    rows[tick] = RecordingRow(
      tick=tick,
      sim_time_s=_number(tick_payload.get("sim_time_s"), f"recording tick {tick} time"),
      segment_id=str(tick_payload.get("segment_id")),
      phase=str(phase),
      state=state,
      post_tick=post_tick,
      post_sim_time_s=post_time,
      post_segment_id=post_segment,
      post_state=post_state,
    )
  if not rows:
    raise TaskConfigError("recording contains no baseline rows")
  ordered = [rows[tick] for tick in sorted(rows)]
  if summary_file is not None:
    if ordered[0].tick != 0 or not np.isclose(ordered[0].sim_time_s, 0.0, atol=1e-9):
      raise TaskConfigError("recording does not start at absolute tick/time zero")
    for key in ("completed_ticks", "attempted_ticks", "encoder_calls", "decoder_calls"):
      if scenario[key] != len(ordered):
        raise TaskConfigError(f"recording full tick coverage disagrees with {key}")
  for row in ordered:
    expected_phase = "bootstrap" if row.tick < contract.past_steps else "baseline"
    if row.phase != expected_phase:
      raise TaskConfigError(f"recording phase disagrees with absolute tick {row.tick}")
    if row.post_state is None or row.post_tick != row.tick + 1:
      raise TaskConfigError(
        f"recording tick {row.tick} has no contiguous terminal post"
      )
    if row.post_segment_id != row.segment_id or not np.isclose(
      float(row.post_sim_time_s or 0.0) - row.sim_time_s,
      contract.period_seconds,
      rtol=0.0,
      atol=1e-9,
    ):
      raise TaskConfigError(
        f"recording tick {row.tick} has an invalid post clock/segment"
      )
  for earlier, later in zip(ordered, ordered[1:], strict=False):
    if later.tick != earlier.tick + 1:
      raise TaskConfigError(
        f"recording baseline rows are not contiguous: ticks {earlier.tick} then "
        f"{later.tick}"
      )
    if later.segment_id != earlier.segment_id:
      raise TaskConfigError(
        f"recording segment changes between ticks {earlier.tick} and {later.tick}"
      )
    if not np.isclose(
      later.sim_time_s - earlier.sim_time_s,
      contract.period_seconds,
      rtol=0.0,
      atol=1e-9,
    ):
      raise TaskConfigError(
        f"recording control period is discontinuous between ticks "
        f"{earlier.tick} and {later.tick}"
      )
    if earlier.post_tick != later.tick or earlier.post_state is None:
      raise TaskConfigError(
        f"recording tick {earlier.tick} has no trusted post for tick {later.tick}"
      )
    if not np.isclose(
      float(earlier.post_sim_time_s or 0.0),
      later.sim_time_s,
      rtol=0.0,
      atol=1e-9,
    ):
      raise TaskConfigError(
        f"recording post time disagrees with the next measured tick at {later.tick}"
      )
    if earlier.post_segment_id != later.segment_id:
      raise TaskConfigError(
        f"recording post segment disagrees with the next measured tick at {later.tick}"
      )
    difference = float(
      np.max(
        np.abs(
          np.concatenate(
            [
              np.asarray(getattr(earlier.post_state, name), dtype=np.float64).reshape(
                -1
              )
              - np.asarray(getattr(later.state, name), dtype=np.float64).reshape(-1)
              for name in _STATE_FIELDS
            ]
          )
        )
      )
    )
    if not np.isfinite(difference) or difference > 1e-9:
      raise TaskConfigError(
        f"recording post state at tick {earlier.post_tick} disagrees with the "
        f"measured row at tick {later.tick}"
      )
  checks.extend(["baseline_rows_only", "tick_contiguity", "post_pre_consistency"])
  # A recording is trusted only for what was actually re-read here; ticks
  # without a summary stay declared rather than verified.
  qualification = "verified" if summary_file is not None else "declared"
  return Recording(
    ticks_path=str(ticks_file),
    ticks_sha256=hashlib.sha256(ticks_bytes).hexdigest(),
    summary_path=None if summary_file is None else str(summary_file),
    summary_sha256=summary_sha,
    rows=rows,
    scenario=scenario,
    checks=tuple(checks),
    qualification=qualification,
  )


def verify_store_binding(
  store_root: str | Path, shard: str | Path, contract: DiffusionContract
) -> dict[str, Any]:
  """Re-read a raw recording store manifest and bind one shard to it."""
  root = Path(store_root)
  manifest_bytes = bounded_read(
    root / "manifest.json", MAX_SOURCE_BYTES, "store manifest file"
  )
  try:
    payload = json.loads(manifest_bytes.decode("utf-8"))
  except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise TaskConfigError(f"store manifest is unreadable: {exc}") from exc
  if not isinstance(payload, Mapping) or payload.get("format") != _STORE_FORMAT:
    raise TaskConfigError("store manifest is not a frozen raw recording store")
  if payload.get("contract_hash") != contract.identity_hash():
    raise TaskConfigError(
      "store manifest contract hash differs from the active contract"
    )
  contract_payload = payload.get("contract")
  body_names = (
    tuple(contract_payload.get("body_names", ()))
    if isinstance(contract_payload, Mapping)
    else ()
  )
  if body_names != contract.body_names:
    raise TaskConfigError("store manifest body order differs from the active contract")
  shard_path = Path(shard)
  digest = hashlib.sha256(
    bounded_read(shard_path, MAX_SOURCE_BYTES, "shard file")
  ).hexdigest()
  entries = payload.get("shards")
  if not isinstance(entries, list):
    raise TaskConfigError("store manifest has no shard list")
  for entry in entries:
    if isinstance(entry, Mapping) and entry.get("file") == shard_path.name:
      if entry.get("sha256") != digest:
        raise TaskConfigError(
          f"shard {shard_path.name} digest disagrees with the store manifest"
        )
      return {
        "store_root": str(root),
        "store_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "contract_hash": str(payload["contract_hash"]),
        "shard_listed": True,
      }
  raise TaskConfigError(f"shard {shard_path.name} is not listed in the store manifest")


def _weights(payload: object) -> PoseCostWeights:
  if not isinstance(payload, Mapping):
    raise TaskConfigError("weights must be a mapping")
  _field_error(
    "weights", payload, _WEIGHT_FIELDS, optional=frozenset({"body_coefficients"})
  )
  payload = cast(Mapping[str, Any], payload)
  coefficients = payload.get("body_coefficients")
  if isinstance(coefficients, list):
    coefficients = [
      _number(value, f"weights.body_coefficients[{index}]")
      for index, value in enumerate(coefficients)
    ]
  elif isinstance(coefficients, Mapping):
    coefficients = {
      _text(name, "weights.body_coefficients key"): _number(
        value, f"weights.body_coefficients[{name}]"
      )
      for name, value in coefficients.items()
    }
  elif coefficients is not None:
    raise TaskConfigError("weights.body_coefficients must be a list or mapping")
  try:
    return PoseCostWeights(
      root_position=_number(payload["root_position"], "weights.root_position"),
      root_orientation=_number(payload["root_orientation"], "weights.root_orientation"),
      body_positions=_number(payload["body_positions"], "weights.body_positions"),
      body_coefficients=coefficients,
    )
  except PoseCostError as exc:
    raise TaskConfigError(f"weights are invalid: {exc}") from exc


def _source(payload: object) -> TaskSource:
  if not isinstance(payload, Mapping):
    raise TaskConfigError("source must be a mapping")
  if not all(isinstance(key, str) for key in payload):
    raise TaskConfigError("source must have string keys")
  payload = cast(Mapping[str, Any], payload)
  unknown = sorted(set(payload) - _SOURCE_FIELDS)
  if unknown:
    raise TaskConfigError(f"source has unknown fields {unknown}")
  for name in ("kind", "qualification", "provenance"):
    if name not in payload:
      raise TaskConfigError(f"source is missing field {name!r}")
  seed = payload.get("seed")
  start_frame = payload.get("start_frame")
  detail = payload.get("verification_detail")
  if detail is not None and not isinstance(detail, Mapping):
    raise TaskConfigError("source.verification_detail must be a mapping")
  try:
    return TaskSource(
      kind=_text(payload["kind"], "source.kind"),
      qualification=_text(payload["qualification"], "source.qualification"),
      provenance=_text(payload["provenance"], "source.provenance"),
      verification=str(payload.get("verification") or "loader_structural_only"),
      path=None if payload.get("path") is None else str(payload["path"]),
      format=None if payload.get("format") is None else str(payload["format"]),
      sha256=None if payload.get("sha256") is None else str(payload["sha256"]),
      summary_path=(
        None if payload.get("summary_path") is None else str(payload["summary_path"])
      ),
      summary_sha256=(
        None
        if payload.get("summary_sha256") is None
        else str(payload["summary_sha256"])
      ),
      member=None if payload.get("member") is None else str(payload["member"]),
      seed=None if seed is None else _tick(seed, "source.seed"),
      start_frame=(
        None if start_frame is None else _tick(start_frame, "source.start_frame")
      ),
      verification_detail=None if detail is None else dict(detail),
    )
  except TaskConditioningError as exc:
    raise TaskConfigError(f"source is invalid: {exc}") from exc


def _contract_identity_matches(declared: str, contract: DiffusionContract) -> bool:
  """Accept the active identity or the same identity without the YAML digest.

  A task file binds the *contract*, not one YAML file: accepting both the active
  identity and the identity of the same contract with ``contract_sha256``
  cleared keeps the schema, dimensions, body order, control period and artifact
  identities pinned while letting one example file serve both the frozen YAML
  and the in-package default contract.
  """
  if declared == contract.identity_hash():
    return True
  return declared == replace(contract, contract_sha256=None).identity_hash()


def source_checks_for_targets(
  payload: Mapping[str, Any],
  targets: Sequence[Mapping[str, Any]],
  *,
  contract: DiffusionContract,
  expected_scenario: ScenarioExpectation | None = None,
) -> dict[str, Any]:
  """Re-read a recording-bound source and require the declared targets to match.

  This is the only path that may return ``verified``: the declared bytes are
  re-read, their digests re-checked, the recorded scenario re-compared with the
  actual run arguments and every target state re-extracted at its tick.  A
  declared ``verified`` that cannot be reproduced here is refused.
  """
  kind = payload.get("kind")
  declared = payload.get("qualification")
  path = payload.get("path")
  sha256 = payload.get("sha256")
  layout = payload.get("format")
  if kind != "baseline_recording":
    if declared == "verified":
      raise TaskConfigError(
        "only a re-readable baseline recording can be verified; a manual or "
        "synthetic source must stay unverified"
      )
    return {"qualification": "unverified", "checks": ["structural_only"]}
  if not isinstance(path, str) or not isinstance(sha256, str):
    raise TaskConfigError(
      "a recording-bound source requires the recording path and its digest"
    )
  if layout == _NPZ_FORMAT:
    return _check_npz_source(payload, targets, contract=contract)
  if layout not in (None, _D3_FORMAT):
    raise TaskConfigError(f"unsupported recording source format {layout!r}")
  ticks_path = Path(path)
  digest = hashlib.sha256(
    bounded_read(ticks_path, MAX_SOURCE_BYTES, "recording ticks file")
  ).hexdigest()
  if digest != sha256:
    raise TaskConfigError(
      f"declared recording digest disagrees with the file on disk: {digest} != {sha256}"
    )
  summary_path = payload.get("summary_path")
  summary_sha256 = payload.get("summary_sha256")
  if summary_path is not None:
    if not isinstance(summary_sha256, str):
      raise TaskConfigError("a declared recording summary requires its recorded digest")
    actual_summary = hashlib.sha256(
      bounded_read(summary_path, MAX_SOURCE_BYTES, "recording summary file")
    ).hexdigest()
    if actual_summary != summary_sha256:
      raise TaskConfigError(
        "declared recording summary digest disagrees with the file on disk"
      )
  recording = read_d3_recording(
    ticks_path,
    summary_path=summary_path,
    contract=contract,
  )
  if recording.ticks_sha256 != sha256 or (
    summary_path is not None and recording.summary_sha256 != summary_sha256
  ):
    raise TaskConfigError("recording inputs changed during source verification")
  if summary_path is None:
    raise TaskConfigError(
      "a recording-bound task requires its recorded summary for re-verification"
    )
  scenario = recording.scenario
  if expected_scenario is not None:
    mismatches: list[str] = []
    for name, expected in (
      ("member", expected_scenario.member),
      ("seed", expected_scenario.seed),
      ("start_frame", expected_scenario.start_frame),
    ):
      if expected is None:
        continue
      if str(scenario.get(name)) != str(expected):
        mismatches.append(f"{name} {scenario.get(name)!r} != {expected!r}")
    if mismatches:
      raise TaskConfigError(
        "recording scenario does not match the requested run: " + "; ".join(mismatches)
      )
  for target in targets:
    tick = int(target["tick"])
    measured = recording.state_at(tick)
    declared_state = target["state"]
    difference = float(
      np.max(
        np.abs(
          np.concatenate(
            [
              np.asarray(getattr(measured, name), dtype=np.float64).reshape(-1)
              - np.asarray(getattr(declared_state, name), dtype=np.float64).reshape(-1)
              for name in _STATE_FIELDS
            ]
          )
        )
      )
    )
    if not np.isfinite(difference) or difference > 1e-9:
      raise TaskConfigError(
        f"target at tick {tick} does not match the measured baseline state "
        f"(max abs difference {difference})"
      )
  return {
    "qualification": "verified",
    "checks": list(recording.checks) + ["target_states", "scenario_binding"],
    "ticks_path": recording.ticks_path,
    "ticks_sha256": recording.ticks_sha256,
    "summary_path": recording.summary_path,
    "summary_sha256": recording.summary_sha256,
    "scenario": dict(scenario),
    "target_ticks": [int(target["tick"]) for target in targets],
    "authentication": (
      "recorded bytes and structural qualification re-read here; not a "
      "cryptographic or independent proof of physical execution"
    ),
  }


def _check_npz_source(
  payload: Mapping[str, Any],
  targets: Sequence[Mapping[str, Any]],
  *,
  contract: DiffusionContract,
) -> dict[str, Any]:
  """Re-read a raw NPZ shard, re-checking its recorded rows and target states.

  The shard digest, each recorded row's controller/segment/clean flags and every
  declared target state are re-read here, so a modified target is refused.  The
  layout still carries weaker recorded provenance than a D3 closed-loop run (no
  paired summary, no closed-loop claim), so the achieved qualification stays
  ``declared`` and the ``--execute`` gate refuses it.
  """
  import io

  path = str(payload["path"])
  sha256 = str(payload["sha256"])
  raw = bounded_read(Path(path), MAX_SOURCE_BYTES, "shard file")
  digest = hashlib.sha256(raw).hexdigest()
  if digest != sha256:
    raise TaskConfigError(
      f"declared shard digest disagrees with the file on disk: {digest} != {sha256}"
    )
  keys = (*_STATE_FIELDS, "metadata")
  try:
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
      arrays = {key: np.asarray(data[key]) for key in keys}
  except (OSError, KeyError, ValueError) as exc:
    raise TaskConfigError(f"shard {path} is unreadable: {exc}") from exc
  rows = int(arrays["root_position"].shape[0])
  checked: list[int] = []
  for target in targets:
    match = re.search(r"#row=(\d+),tick=(\d+)", str(target["provenance"]))
    if match is None:
      raise TaskConfigError(
        "an NPZ-bound target provenance must record '#row=<index>,tick=<tick>'"
      )
    index, tick = int(match.group(1)), int(match.group(2))
    if tick != int(target["tick"]) or not 0 <= index < rows:
      raise TaskConfigError(
        f"target provenance row/tick does not match its declared tick {target['tick']}"
      )
    try:
      metadata = json.loads(str(arrays["metadata"][index]))
    except json.JSONDecodeError as exc:
      raise TaskConfigError(f"shard row {index} metadata is not JSON") from exc
    if not isinstance(metadata, Mapping):
      raise TaskConfigError(f"shard row {index} metadata is not a mapping")
    if (
      metadata.get("controller") != "vae"
      or metadata.get("segment_qualified") is not True
    ):
      raise TaskConfigError(f"shard row {index} is not a qualified VAE row")
    if (
      metadata.get("reset")
      or metadata.get("teleport")
      or metadata.get("reference_boundary")
    ):
      raise TaskConfigError(f"shard row {index} is not a clean measured row")
    if int(metadata.get("tick", -1)) != tick:
      raise TaskConfigError(f"shard row {index} recorded tick disagrees with {tick}")
    measured = WorldState(
      **{
        name: np.asarray(arrays[name][index], dtype=np.float64)
        for name in _STATE_FIELDS
      }
    )
    difference = float(
      np.max(
        np.abs(
          np.concatenate(
            [
              np.asarray(getattr(measured, name), dtype=np.float64).reshape(-1)
              - np.asarray(getattr(target["state"], name), dtype=np.float64).reshape(-1)
              for name in _STATE_FIELDS
            ]
          )
        )
      )
    )
    if not np.isfinite(difference) or difference > 1e-9:
      raise TaskConfigError(
        f"target at tick {tick} does not match the recorded shard row "
        f"(max abs difference {difference})"
      )
    checked.append(tick)
  return {
    "qualification": "declared",
    "checks": ["sha256", "row_qualification", "target_states"],
    "shard_path": path,
    "shard_sha256": digest,
    "target_ticks": checked,
    "authentication": (
      "recorded bytes and row qualification re-read here; not a cryptographic "
      "or independent proof of physical execution"
    ),
  }


def parse_task_config(
  payload: object,
  *,
  contract: DiffusionContract,
  config_sha256: str,
  expected_scenario: ScenarioExpectation | None = None,
  require_verified: bool = False,
) -> SparseTaskSpec:
  """Validate an already-decoded task payload against the active contract."""
  if not isinstance(payload, Mapping):
    raise TaskConfigError("task config must be a JSON object")
  _field_error("task config", payload, _TOP_FIELDS, optional=frozenset({"weights"}))
  payload = cast(Mapping[str, Any], payload)
  if payload["schema"] != TASK_SCHEMA:
    raise TaskConfigError(
      f"task schema must be {TASK_SCHEMA!r}, got {payload['schema']!r}"
    )
  if payload["frame"] != DEFAULT_FRAME:
    raise TaskConfigError(
      f"task frame must be {DEFAULT_FRAME!r}, got {payload['frame']!r}"
    )
  if payload["tick_origin"] != DEFAULT_TICK_ORIGIN:
    raise TaskConfigError(
      f"task tick_origin must be {DEFAULT_TICK_ORIGIN!r}, got "
      f"{payload['tick_origin']!r}"
    )
  period = _number(payload["control_period_seconds"], "control_period_seconds")
  if not np.isclose(period, contract.period_seconds, rtol=0.0, atol=1e-12):
    raise TaskConfigError(
      f"task control_period_seconds {period} disagrees with the active contract "
      f"{contract.period_seconds}"
    )
  identity = _text(payload["contract_identity"], "contract_identity")
  if not _contract_identity_matches(identity, contract):
    raise TaskConfigError(
      "task contract_identity was authored against a different frozen contract"
    )
  body_order = payload["body_order"]
  if not isinstance(body_order, list) or tuple(body_order) != contract.body_names:
    raise TaskConfigError("task body_order differs from the frozen contract body order")
  mode = _text(payload["mode"], "mode")
  if mode not in MODES:
    raise TaskConfigError(f"task mode must be one of {list(MODES)}")
  source = _source(payload["source"])
  strength = _number(payload["strength"], "strength")
  if strength < 0.0:
    raise TaskConfigError("strength must be non-negative")
  weights = None if payload.get("weights") is None else _weights(payload["weights"])
  keyframes_payload = _targets(payload["keyframes"], "keyframes", contract=contract)
  goals_payload = _targets(payload["goals"], "goals", contract=contract)
  total = len(keyframes_payload) + len(goals_payload)
  if total > MAX_TARGETS:
    raise TaskConfigError(
      f"a sparse task carries at most {MAX_TARGETS} targets, got {total}"
    )
  try:
    keyframes = tuple(
      TimedStateKeyframe(
        tick=item["tick"], state=item["state"], provenance=item["provenance"]
      )
      for item in keyframes_payload
    )
    goals = tuple(
      TimedPoseGoal(
        tick=item["tick"], state=item["state"], provenance=item["provenance"]
      )
      for item in goals_payload
    )
  except (TaskConditioningError, PoseCostError) as exc:
    raise TaskConfigError(f"task targets are invalid: {exc}") from exc
  # The declared source qualification is recomputed from the recording bytes; a
  # hand-written "verified" label can never upgrade a source by itself.
  checks = source_checks_for_targets(
    source.as_dict(),
    [*keyframes_payload, *goals_payload],
    contract=contract,
    expected_scenario=expected_scenario,
  )
  achieved = str(checks["qualification"])
  if source.qualification == "verified" and achieved != "verified":
    raise TaskConfigError(
      "the source declares 'verified' but its recording cannot be re-read and "
      "re-checked here"
    )
  if require_verified and achieved != "verified":
    raise TaskConfigError(
      "this run requires a verified recording source; the task source is only "
      f"{achieved!r}"
    )
  verified_source = replace(
    source,
    qualification=achieved,
    verification=(
      "loader_recheck:" + ",".join(str(check) for check in checks["checks"])
      if achieved == "verified"
      else source.verification
    ),
    verification_detail={
      **(source.verification_detail or {}),
      "loader_recheck": checks,
    },
  )
  try:
    return SparseTaskSpec(
      mode=mode,
      keyframes=keyframes,
      goals=goals,
      weights=weights,
      strength=strength,
      source=verified_source,
      config_sha256=config_sha256,
    )
  except (TaskConditioningError, PoseCostError) as exc:
    raise TaskConfigError(f"task targets are invalid: {exc}") from exc


def load_task_config(
  path: str | Path,
  *,
  contract: DiffusionContract = DEFAULT_CONTRACT,
  expected_scenario: ScenarioExpectation | None = None,
  require_verified: bool = False,
) -> SparseTaskSpec:
  """Read, bound-check and strictly validate one D4 task config file.

  The file is read through the bounded regular-file reader, so an oversized or
  non-regular path cannot allocate arbitrarily even outside the evaluation CLI.
  """
  source = Path(path)
  raw = bounded_read(source, MAX_TASK_BYTES, "task config")
  if not raw:
    raise TaskConfigError(f"task config {source} is empty")
  try:
    text = raw.decode("utf-8")
  except UnicodeDecodeError as exc:
    raise TaskConfigError(f"task config is not valid UTF-8: {exc}") from exc
  try:
    payload = json.loads(text, object_pairs_hook=_no_duplicate_keys)
  except json.JSONDecodeError as exc:
    raise TaskConfigError(f"task config is not valid JSON: {exc}") from exc
  digest = hashlib.sha256(raw).hexdigest()
  return parse_task_config(
    payload,
    contract=contract,
    config_sha256=digest,
    expected_scenario=expected_scenario,
    require_verified=require_verified,
  )


def write_task_config(
  payload: Mapping[str, Any],
  path: str | Path,
  *,
  contract: DiffusionContract = DEFAULT_CONTRACT,
  expected_scenario: ScenarioExpectation | None = None,
) -> SparseTaskSpec:
  """Validate a new payload and only then write it atomically to a new path.

  The strict loader (including the recording re-check) must accept the exact
  bytes that will be written, so a preparation run can never emit an invalid,
  unverifiable or oversized task file.
  """
  destination = Path(path)
  if destination.exists():
    raise TaskConfigError(f"refusing to overwrite existing output {destination}")
  encoded = (json.dumps(payload, indent=1, sort_keys=True) + "\n").encode("utf-8")
  if not encoded or len(encoded) > MAX_TASK_BYTES:
    raise TaskConfigError(
      f"task config must be a non-empty file of at most {MAX_TASK_BYTES} bytes"
    )
  spec = parse_task_config(
    json.loads(encoded.decode("utf-8"), object_pairs_hook=_no_duplicate_keys),
    contract=contract,
    config_sha256=hashlib.sha256(encoded).hexdigest(),
    expected_scenario=expected_scenario,
  )
  destination.parent.mkdir(parents=True, exist_ok=True)
  handle, temporary = tempfile.mkstemp(
    dir=str(destination.parent), prefix=f".{destination.name}.", suffix=".tmp"
  )
  try:
    with os.fdopen(handle, "wb") as stream:
      stream.write(encoded)
    os.replace(temporary, destination)
  except OSError as exc:
    raise TaskConfigError(
      f"task config {destination} could not be written: {exc}"
    ) from exc
  finally:
    if os.path.exists(temporary):
      os.unlink(temporary)
  return spec


def main(argv: Sequence[str] | None = None) -> int:
  """Validate a task config without constructing anything (diagnostic only)."""
  import argparse

  parser = argparse.ArgumentParser(
    description="Strictly validate a D4 sparse-task config (no execution)."
  )
  parser.add_argument("--task-config", type=Path, required=True)
  parser.add_argument(
    "--contract",
    type=Path,
    default=None,
    help="active contract YAML; default: the in-package frozen default contract",
  )
  parser.add_argument("--member", default=None, help="expected recorded member")
  parser.add_argument("--seed", type=int, default=None)
  parser.add_argument("--start-frame", type=int, default=None)
  args = parser.parse_args(argv)
  contract = (
    DEFAULT_CONTRACT
    if args.contract is None
    else DiffusionContract.from_yaml(args.contract)
  )
  try:
    task = load_task_config(
      args.task_config,
      contract=contract,
      expected_scenario=ScenarioExpectation(
        member=args.member, seed=args.seed, start_frame=args.start_frame
      ),
    )
  except (TaskConfigError, ValueError) as exc:
    print(f"task config rejected: {exc}")
    return 1
  print(
    json.dumps(
      {
        "ok": True,
        "schema": task.schema_version,
        "mode": task.mode,
        "config_sha256": task.config_sha256,
        "keyframe_ticks": list(task.keyframe_ticks),
        "goal_ticks": list(task.goal_ticks),
        "strength": task.strength,
        "source": task.source.as_dict(),
      },
      indent=1,
      sort_keys=True,
    )
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
