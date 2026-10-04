#!/usr/bin/env python3
"""Bounded, explicit-path D3 closed-loop evaluation entry point.

The command has two modes.  Without ``--execute`` it performs the frozen
contract/model/manifest preflight and constructs no simulator.  With
``--execute`` it runs exactly one matched VAE-baseline/reference-free diffusion
pair through the public :func:`run_matched_pair` seam and the private X2
runtime factory.  This is a measurement harness, not a policy-quality verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import torch
from torch import Tensor

from mjlab.tasks.tracking.diffusion.closed_loop import (
  MIN_OUTPUT_CAP_BYTES,
  EvaluationBudget,
  EvaluationReport,
  persist_evaluation_report,
  run_matched_pair,
)
from mjlab.tasks.tracking.diffusion.conditioned_continuation import (
  SparseTaskSpec,
  TaskConditioning,
  build_conditioned_continuation,
)
from mjlab.tasks.tracking.diffusion.handover import ReferenceFreeContinuation
from mjlab.tasks.tracking.diffusion.replay_integration import (
  LoadedDenoiser,
  LoadedVae,
  ReplayError,
  load_active_contract,
  load_denoiser,
  load_projection_bundle,
  load_vae_decoder,
  sha256_file,
  verify_file_digest,
)
from mjlab.tasks.tracking.distillation.config import load_manifest, resolve_cohort

_MAX_CONTINUATION_STEPS = 250
_MAX_WALL_SECONDS = 1200.0
_MAX_OUTPUT_BYTES = 128 * 1024 * 1024
_TRAINING_BUDGET_UPDATES = 104_000
_BOOTSTRAP_PAIRS = 8
_FROZEN_DDIM_STEPS = 20
_FROZEN_START_LEVEL = 1000
_FROZEN_CONTROL_PERIOD = 0.02

# The strict D4 task loader lives beside this CLI as an experiment tool.  It is
# loaded by absolute path so the CLI keeps working both as a script and as an
# imported module, and so the loaded source is fingerprinted explicitly.
_TASK_LOADER_NAME = "mjlab_d4_sparse_task_config"
_TASK_LOADER_PATH = Path(__file__).resolve().with_name("task_config.py")


def _task_loader() -> Any:
  """Import the sibling D4 task loader by absolute path exactly once."""
  existing = sys.modules.get(_TASK_LOADER_NAME)
  if existing is not None:
    return existing
  if not _TASK_LOADER_PATH.is_file():
    raise ClosedLoopCliError(f"task loader is not readable: {_TASK_LOADER_PATH}")
  spec = importlib.util.spec_from_file_location(_TASK_LOADER_NAME, _TASK_LOADER_PATH)
  if spec is None or spec.loader is None:
    raise ClosedLoopCliError(f"task loader cannot be imported: {_TASK_LOADER_PATH}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[_TASK_LOADER_NAME] = module
  spec.loader.exec_module(module)
  return module


class ClosedLoopCliError(ValueError):
  """A closed-loop CLI request failed a preflight gate."""


@dataclass(frozen=True, slots=True)
class PreparedEvaluation:
  """All non-simulator artifacts validated by preflight."""

  args: argparse.Namespace
  repo_root: Path
  contract: Any
  projection: Any
  denoiser: LoadedDenoiser
  vae: LoadedVae
  clip_frames: int
  clip_fps: float
  manifest_sha256: str
  teacher_hashes: Mapping[str, str]
  file_expectations: Mapping[str, tuple[Path, str]]
  source_fingerprints: Mapping[str, Any]
  integrity_before: Mapping[str, Any]
  device: torch.device
  task: SparseTaskSpec | None = None
  task_source_files: Mapping[str, tuple[Path, str]] = field(default_factory=dict)


def _json_value(value: Any) -> Any:
  """Convert model metadata into bounded JSON-compatible values."""
  if isinstance(value, Tensor):
    return {
      "dtype": str(value.dtype),
      "shape": list(value.shape),
      "values_sha256": hashlib.sha256(
        value.detach().to(device="cpu").contiguous().numpy().tobytes()
      ).hexdigest(),
    }
  if isinstance(value, np.ndarray):
    return {
      "dtype": str(value.dtype),
      "shape": list(value.shape),
      "values_sha256": hashlib.sha256(
        np.ascontiguousarray(value).tobytes()
      ).hexdigest(),
    }
  if isinstance(value, Mapping):
    return {
      str(key): _json_value(item)
      for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
    }
  if isinstance(value, (list, tuple)):
    return [_json_value(item) for item in value]
  if isinstance(value, (str, int, float, bool)) or value is None:
    return value
  return repr(value)


def _update_digest(digest: "hashlib._Hash", value: Any) -> None:
  """Hash tensors and non-tensor extra state without dropping metadata."""
  if isinstance(value, Tensor):
    array = value.detach().to(device="cpu").contiguous().numpy()
    digest.update(b"tensor\0")
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(array.tobytes())
    return
  if isinstance(value, np.ndarray):
    array = np.ascontiguousarray(value)
    digest.update(b"ndarray\0")
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return
  if isinstance(value, Mapping):
    digest.update(b"mapping\0")
    for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
      _update_digest(digest, str(key))
      _update_digest(digest, item)
    return
  if isinstance(value, (list, tuple)):
    digest.update(b"sequence\0")
    for item in value:
      _update_digest(digest, item)
    return
  digest.update(type(value).__name__.encode())
  digest.update(json.dumps(value, sort_keys=True, default=repr).encode())


def _state_fingerprint(module: Any) -> dict[str, Any]:
  """Fingerprint a frozen module, including ``_extra_state`` entries."""
  state_method = getattr(module, "state_dict", None)
  if not callable(state_method):
    raise ClosedLoopCliError("frozen model does not expose state_dict()")
  state = state_method()
  if not isinstance(state, Mapping):
    raise ClosedLoopCliError("frozen model state_dict() is not a mapping")
  digest = hashlib.sha256()
  tensor_keys: list[str] = []
  non_tensor_keys: list[str] = []
  extra_state: dict[str, Any] = {}
  for key, value in sorted(state.items(), key=lambda pair: str(pair[0])):
    name = str(key)
    if isinstance(value, Tensor):
      tensor_keys.append(name)
    else:
      non_tensor_keys.append(name)
      if name.endswith("_extra_state"):
        extra_state[name] = _json_value(value)
    _update_digest(digest, name)
    _update_digest(digest, value)
  return {
    "state_dict_sha256": digest.hexdigest(),
    "tensor_keys": tensor_keys,
    "non_tensor_keys": non_tensor_keys,
    "non_tensor_extra_state": extra_state,
  }


def _array_digest(value: Any) -> str:
  array = np.ascontiguousarray(np.asarray(value))
  return hashlib.sha256(array.tobytes()).hexdigest()


def _normalizer_fingerprint(normalizer: Any) -> dict[str, Any]:
  result = _state_fingerprint(normalizer)
  result["dimension"] = int(getattr(normalizer, "dimension", -1))
  result["frozen"] = bool(getattr(normalizer, "frozen", False))
  result["eps"] = float(getattr(normalizer, "eps", float("nan")))
  return result


def _integrity_snapshot(prepared: PreparedEvaluation) -> dict[str, Any]:
  """Capture files, models, projection stats and VAE normalizers."""
  files: dict[str, Any] = {}
  for role, (path, expected) in prepared.file_expectations.items():
    actual = sha256_file(path)
    files[role] = {
      "path": str(path),
      "expected_sha256": expected,
      "actual_sha256": actual,
      "matches_expected": actual == expected,
    }
  projection = prepared.projection
  vae_model = cast(Any, prepared.vae.model)
  return {
    "files": files,
    "models": {
      "diffusion": _state_fingerprint(prepared.denoiser.model),
      "vae": _state_fingerprint(vae_model),
    },
    "normalizers": {
      "projection_state_stats": {
        "statistics_sha256": projection.statistics_sha256,
        "mean_sha256": _array_digest(projection.state_stats.mean),
        "std_sha256": _array_digest(projection.state_stats.std),
      },
      "projection_latent_stats": {
        "mean_sha256": _array_digest(projection.latent_stats.mean),
        "std_sha256": _array_digest(projection.latent_stats.std),
      },
      "vae_reference": _normalizer_fingerprint(vae_model.reference_normalizer),
      "vae_conditioning": _normalizer_fingerprint(vae_model.conditioning_normalizer),
    },
  }


_SOURCE_MODULES = (
  "mjlab.tasks.tracking.diffusion.closed_loop_types",
  "mjlab.tasks.tracking.diffusion.closed_loop_runtime_x2",
  "mjlab.tasks.tracking.diffusion.closed_loop",
  "mjlab.tasks.tracking.diffusion.handover",
  "mjlab.tasks.tracking.diffusion.replay_integration",
  "mjlab.tasks.tracking.diffusion.contract",
)


def _git_root(path: Path) -> Path | None:
  """Find the checkout containing an actual imported source file."""
  for candidate in (path.parent, *path.parents):
    if (candidate / ".git").exists():
      return candidate
  return None


def _git_head(root: Path) -> str | None:
  try:
    result = subprocess.run(
      ["git", "rev-parse", "HEAD"],
      cwd=root,
      check=False,
      capture_output=True,
      text=True,
      timeout=5,
    )
  except (OSError, subprocess.SubprocessError):
    return None
  head = result.stdout.strip()
  return head if result.returncode == 0 and head else None


def _source_fingerprints() -> dict[str, Any]:
  """Fingerprint the executed CLI and files actually imported by this process."""
  paths: dict[str, dict[str, str]] = {}
  roots: dict[str, str | None] = {}
  cli_path = Path(__file__).resolve()
  if not cli_path.is_file():
    raise ClosedLoopCliError(f"executed CLI source cannot be identified: {cli_path}")
  paths["__executed_cli__"] = {
    "path": str(cli_path),
    "sha256": sha256_file(cli_path),
  }
  source_paths = [cli_path]
  loader = sys.modules.get(_TASK_LOADER_NAME)
  loader_file = None if loader is None else getattr(loader, "__file__", None)
  if isinstance(loader_file, str) and loader_file:
    loader_path = Path(loader_file).resolve()
    paths[_TASK_LOADER_NAME] = {
      "path": str(loader_path),
      "sha256": sha256_file(loader_path),
    }
    source_paths.append(loader_path)
  module_names: set[str] = set(_SOURCE_MODULES)
  for name in sys.modules:
    if name.startswith(
      ("mjlab.tasks.tracking.diffusion.", "mjlab.tasks.tracking.distillation.")
    ):
      module_names.add(name)
  for module_name in sorted(module_names):
    try:
      module = importlib.import_module(module_name)
    except Exception as exc:
      raise ClosedLoopCliError(
        f"imported source module {module_name!r} cannot be loaded: {exc}"
      ) from exc
    filename = getattr(module, "__file__", None)
    if not isinstance(filename, str) or not filename:
      raise ClosedLoopCliError(
        f"imported source module {module_name!r} has no identifiable __file__"
      )
    path = Path(filename).resolve()
    if path.suffix != ".py" or not path.is_file():
      raise ClosedLoopCliError(
        f"imported source module {module_name!r} is not a readable Python file: {path}"
      )
    paths[module_name] = {"path": str(path), "sha256": sha256_file(path)}
    source_paths.append(path)
  for path in source_paths:
    root = _git_root(path)
    if root is None:
      raise ClosedLoopCliError(f"source checkout cannot be identified for {path}")
    roots[str(root)] = _git_head(root)
  return {
    "executed_cli": paths["__executed_cli__"],
    "modules": {
      key: value for key, value in paths.items() if key != "__executed_cli__"
    },
    "source_checkouts": roots,
  }


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description=(
      "Run one bounded matched VAE-baseline/reference-free D3 diffusion pair. "
      "Without --execute this is a simulator-free frozen-artifact preflight."
    )
  )
  parser.add_argument(
    "--contract", type=Path, required=True, help="active contract YAML"
  )
  parser.add_argument(
    "--contract-sha256", required=True, help="expected SHA256 of the contract file"
  )
  parser.add_argument(
    "--checkpoint", type=Path, required=True, help="frozen D2 checkpoint"
  )
  parser.add_argument(
    "--checkpoint-sha256", required=True, help="expected SHA256 of the D2 checkpoint"
  )
  parser.add_argument(
    "--projection", type=Path, required=True, help="projection bundle"
  )
  parser.add_argument(
    "--projection-sha256",
    required=True,
    help="expected SHA256 of the projection bundle",
  )
  parser.add_argument(
    "--vae-checkpoint",
    "--vae",
    dest="vae_checkpoint",
    type=Path,
    required=True,
    help="frozen gravity-mode VAE cohort checkpoint",
  )
  parser.add_argument(
    "--vae-checkpoint-sha256",
    "--vae-sha256",
    dest="vae_checkpoint_sha256",
    required=True,
    help="expected SHA256 of the VAE checkpoint",
  )
  parser.add_argument(
    "--manifest",
    "--vae-manifest",
    dest="manifest",
    type=Path,
    required=True,
    help="frozen VAE cohort manifest",
  )
  parser.add_argument(
    "--manifest-sha256",
    "--vae-manifest-sha256",
    dest="manifest_sha256",
    required=True,
    help="expected SHA256 of the cohort manifest",
  )
  parser.add_argument(
    "--repo-root",
    "--vae-repo-root",
    dest="repo_root",
    type=Path,
    required=True,
    help="repository root resolving manifest artifacts",
  )
  parser.add_argument("--member", "--vae-member", dest="member", required=True)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--start-frame", type=int, required=True)
  parser.add_argument(
    "--continuation-steps",
    type=int,
    default=25,
    help="post-bootstrap evaluation ticks per branch (1..250; default: 25)",
  )
  parser.add_argument(
    "--max-wall-seconds",
    type=float,
    default=_MAX_WALL_SECONDS,
    help="wall-clock budget in seconds (0 < value <= 1200; default: 1200)",
  )
  parser.add_argument(
    "--max-output-bytes",
    type=int,
    default=_MAX_OUTPUT_BYTES,
    help="summary/tick output cap (0 < value <= 128 MiB)",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    required=True,
    help="new output directory under logs/diagnostics/...",
  )
  parser.add_argument(
    "--device",
    default="cuda:0",
    help="torch device for the one-environment pair (default: cuda:0)",
  )
  parser.add_argument(
    "--task-config",
    type=Path,
    default=None,
    help=(
      "optional D4 sparse-task JSON (hard keyframes and/or soft pose goals). "
      "Without it the run is the unchanged unguided reference-free D3 pair."
    ),
  )
  parser.add_argument(
    "--execute",
    action="store_true",
    help="construct the simulator and execute the bounded pair",
  )
  return parser


def build_parser() -> argparse.ArgumentParser:
  """Expose the parser for focused CLI tests and documentation tooling."""
  return _parser()


def _validate_request(args: argparse.Namespace) -> tuple[Path, torch.device]:
  if isinstance(args.seed, bool) or not isinstance(args.seed, int) or args.seed < 0:
    raise ClosedLoopCliError("seed must be a nonnegative integer")
  if (
    isinstance(args.start_frame, bool)
    or not isinstance(args.start_frame, int)
    or args.start_frame < 0
  ):
    raise ClosedLoopCliError("start-frame must be a nonnegative integer")
  if (
    isinstance(args.continuation_steps, bool)
    or not isinstance(args.continuation_steps, int)
    or not 1 <= args.continuation_steps <= _MAX_CONTINUATION_STEPS
  ):
    raise ClosedLoopCliError(
      f"continuation-steps must be an integer in [1, {_MAX_CONTINUATION_STEPS}]"
    )
  if (
    isinstance(args.max_wall_seconds, bool)
    or not np.isfinite(float(args.max_wall_seconds))
    or not 0.0 < float(args.max_wall_seconds) <= _MAX_WALL_SECONDS
  ):
    raise ClosedLoopCliError(
      f"max-wall-seconds must be finite and in (0, {_MAX_WALL_SECONDS}]"
    )
  if (
    isinstance(args.max_output_bytes, bool)
    or not isinstance(args.max_output_bytes, int)
    or not MIN_OUTPUT_CAP_BYTES <= args.max_output_bytes <= _MAX_OUTPUT_BYTES
  ):
    raise ClosedLoopCliError(
      f"max-output-bytes must be an integer in [{MIN_OUTPUT_CAP_BYTES}, "
      f"{_MAX_OUTPUT_BYTES}]"
    )
  output_dir = args.output_dir.resolve()
  if output_dir.exists():
    raise ClosedLoopCliError(
      f"output-dir must be new and absent; refusing existing path {output_dir}"
    )
  repo_root = args.repo_root.resolve()
  if not repo_root.is_dir():
    raise ClosedLoopCliError(f"repo-root is not a directory: {repo_root}")
  task_config = getattr(args, "task_config", None)
  if task_config is not None:
    task_path = Path(task_config)
    if not task_path.is_file():
      raise ClosedLoopCliError(f"task-config is not a readable file: {task_path}")
    size = task_path.stat().st_size
    if size <= 0 or size > (1 << 20):
      raise ClosedLoopCliError(
        f"task-config must be a non-empty file of at most 1 MiB, got {size} bytes"
      )
  try:
    device = torch.device(args.device)
    torch.empty(0, device=device)
  except (RuntimeError, ValueError) as exc:
    raise ClosedLoopCliError(
      f"requested device is unavailable: {args.device!r}"
    ) from exc
  return repo_root, device


def _require_frozen_contract(contract: Any, expected_sha256: str) -> None:
  if contract.schema_version != "mjlab-x2-state-latent-d1-no-mixtures-v1":
    raise ClosedLoopCliError(
      "D3 closed-loop requires the active no-mixture v1 contract; "
      f"got {contract.schema_version!r}"
    )
  expected = {
    "control_hz": 50,
    "state_dimension": 135,
    "body_count": 20,
    "latent_dimension": 32,
    "action_dimension": 31,
    "past_steps": 8,
    "current_index": 8,
    "future_steps": 32,
    "window_steps": 41,
    "projected_state_dimension": 135,
    "token_dimension": 167,
    "projection_rows": 0,
  }
  for name, value in expected.items():
    if getattr(contract, name) != value:
      raise ClosedLoopCliError(
        f"frozen contract field {name} must be {value!r}, got {getattr(contract, name)!r}"
      )
  if not np.isclose(contract.period_seconds, _FROZEN_CONTROL_PERIOD):
    raise ClosedLoopCliError("frozen contract control period must be 0.02 seconds")
  if contract.contract_sha256 != expected_sha256:
    raise ClosedLoopCliError(
      "contract embedded digest disagrees with the explicit contract hash"
    )


def _prepare(args: argparse.Namespace) -> PreparedEvaluation:
  repo_root, device = _validate_request(args)
  contract_path = args.contract.resolve()
  checkpoint_path = args.checkpoint.resolve()
  projection_path = args.projection.resolve()
  vae_path = args.vae_checkpoint.resolve()
  manifest_path = args.manifest.resolve()
  file_expectations = {
    "contract": (contract_path, str(args.contract_sha256)),
    "diffusion_checkpoint": (checkpoint_path, str(args.checkpoint_sha256)),
    "projection": (projection_path, str(args.projection_sha256)),
    "vae_checkpoint": (vae_path, str(args.vae_checkpoint_sha256)),
    "vae_manifest": (manifest_path, str(args.manifest_sha256)),
  }
  try:
    for role, (path, expected) in file_expectations.items():
      verify_file_digest(path, expected, role=role)
    contract = load_active_contract(contract_path)
  except (ReplayError, OSError, ValueError) as exc:
    raise ClosedLoopCliError(str(exc)) from exc
  _require_frozen_contract(contract, str(args.contract_sha256))
  try:
    manifest = load_manifest(manifest_path, repo_root=repo_root)
    cohort = resolve_cohort(manifest)
    teacher = cohort.teacher(args.member)
  except Exception as exc:
    raise ClosedLoopCliError(f"VAE manifest/member preflight failed: {exc}") from exc
  try:
    projection = load_projection_bundle(projection_path, contract=contract)
    denoiser = load_denoiser(
      checkpoint_path,
      contract=contract,
      checkpoint_sha256=str(args.checkpoint_sha256),
      projection_bundle=projection,
    )
    vae = load_vae_decoder(
      vae_path,
      manifest_path=manifest_path,
      repo_root=repo_root,
      member=args.member,
      contract=contract,
      vae_checkpoint_sha256=str(args.vae_checkpoint_sha256),
    )
  except (ReplayError, OSError, ValueError) as exc:
    raise ClosedLoopCliError(str(exc)) from exc
  if denoiser.schedule.training_k != _FROZEN_START_LEVEL:
    raise ClosedLoopCliError("frozen diffusion schedule must use training level 1000")
  if denoiser.grid.updates != _FROZEN_DDIM_STEPS:
    raise ClosedLoopCliError("frozen diffusion inference grid must use 20 DDIM updates")
  if denoiser.global_step > _TRAINING_BUDGET_UPDATES:
    raise ClosedLoopCliError(
      f"checkpoint selected step {denoiser.global_step} exceeds the frozen "
      f"{_TRAINING_BUDGET_UPDATES}-update training budget"
    )
  if (
    vae.reference_dim != 68
    or vae.conditioning_dim != 99
    or vae.latent_dim != 32
    or vae.action_dim != 31
  ):
    raise ClosedLoopCliError("frozen VAE dimensions are not 68/99/32/31")
  clip_frames = int(teacher.reference.frames)
  clip_fps = float(teacher.reference.fps)
  if clip_frames < 2 or args.start_frame >= clip_frames:
    raise ClosedLoopCliError(
      f"start-frame {args.start_frame} is outside selected {args.member} clip "
      f"[0, {clip_frames})"
    )
  if not np.isclose(clip_fps, 50.0, atol=1.0e-5, rtol=0.0):
    raise ClosedLoopCliError(f"selected motion must be 50 Hz, got {clip_fps}")
  task: SparseTaskSpec | None = None
  if getattr(args, "task_config", None) is not None:
    # Strict task validation happens here, before any runtime or environment is
    # constructed; a malformed, unverifiable or scenario-mismatched task file can
    # never reach the simulator.  Under --execute an unverified source is also
    # refused, so only a re-read recording can drive a physical attempt.
    loader = _task_loader()
    try:
      task = cast(
        SparseTaskSpec,
        loader.load_task_config(
          Path(args.task_config),
          contract=contract,
          expected_scenario=loader.ScenarioExpectation(
            member=args.member, seed=args.seed, start_frame=args.start_frame
          ),
          require_verified=bool(args.execute),
        ),
      )
    except (ValueError, OSError) as exc:
      raise ClosedLoopCliError(f"task-config rejected: {exc}") from exc
    detail = task.source.verification_detail or {}
    # Only fresh loader checks supply integrity paths, never caller-authored
    # verification detail.  The receipt is nested under loader_recheck.
    source_checks = detail.get("loader_recheck", {})
    if not isinstance(source_checks, Mapping):
      raise ClosedLoopCliError("task source has no fresh verification receipt")
    file_expectations["task_config"] = (
      Path(args.task_config).resolve(),
      task.config_sha256,
    )
    for role, path_key, sha_key in (
      ("task_source_ticks", "ticks_path", "ticks_sha256"),
      ("task_source_summary", "summary_path", "summary_sha256"),
      ("task_source_shard", "shard_path", "shard_sha256"),
    ):
      path_value = source_checks.get(path_key)
      sha_value = source_checks.get(sha_key)
      if isinstance(path_value, str) and isinstance(sha_value, str):
        file_expectations[role] = (Path(path_value), sha_value)
  source_fingerprints = _source_fingerprints()
  prepared = PreparedEvaluation(
    args=args,
    repo_root=repo_root,
    contract=contract,
    projection=projection,
    denoiser=denoiser,
    vae=vae,
    clip_frames=clip_frames,
    clip_fps=clip_fps,
    manifest_sha256=manifest.sha256,
    teacher_hashes=dict(teacher.hashes),
    file_expectations=file_expectations,
    source_fingerprints=source_fingerprints,
    integrity_before={},
    device=device,
    task=task,
    task_source_files={
      role: value
      for role, value in file_expectations.items()
      if role.startswith("task_")
    },
  )
  integrity = _integrity_snapshot(prepared)
  for role, info in integrity.get("files", {}).items():
    if info.get("matches_expected") is False:
      raise ClosedLoopCliError(f"preflight input changed while loading: {role}")
  object.__setattr__(prepared, "integrity_before", integrity)
  return prepared


def _task_config_report(task: SparseTaskSpec | None) -> dict[str, Any] | None:
  """Bounded preflight view of an opt-in task config, with no target states.

  ``None`` for a run without ``--task-config`` so the report and marker keep the
  original D3 provenance: a no-task run must not acquire task metadata.
  """
  if task is None:
    return None
  return {
    "provided": True,
    "schema": task.schema_version,
    "mode": task.mode,
    "config_sha256": task.config_sha256,
    "keyframe_count": len(task.keyframes),
    "goal_count": len(task.goals),
    "keyframe_ticks": list(task.keyframe_ticks),
    "goal_ticks": list(task.goal_ticks),
    "guidance_strength": task.strength,
    "weights": None
    if task.weights is None
    else {
      "root_position": task.weights.root_position,
      "root_orientation": task.weights.root_orientation,
      "body_positions": task.weights.body_positions,
    },
    "source": task.source.as_dict(),
    "effect": "hard keyframes and/or soft pose-cost guidance on the diffusion branch",
  }


def _preflight_report(prepared: PreparedEvaluation) -> dict[str, Any]:
  args = prepared.args
  report: dict[str, Any] = {
    "schema": "mjlab-d3-closed-loop-cli-preflight-v1",
    "ok": True,
    "execution": "not-run; pass --execute to construct the simulator",
    "contract": {
      "path": str(args.contract.resolve()),
      "sha256": args.contract_sha256,
      "schema_version": prepared.contract.schema_version,
      "identity_hash": prepared.contract.identity_hash(),
      "token_dimension": prepared.contract.token_dimension,
      "past_steps": prepared.contract.past_steps,
      "future_steps": prepared.contract.future_steps,
    },
    "checkpoint": {
      "path": str(args.checkpoint.resolve()),
      "sha256": args.checkpoint_sha256,
      "selected_step": prepared.denoiser.global_step,
      "training_budget_updates": _TRAINING_BUDGET_UPDATES,
      "ema_selected": prepared.denoiser.ema_selected,
      "ema_updates": prepared.denoiser.ema_updates,
    },
    "projection": {
      "path": str(args.projection.resolve()),
      "sha256": args.projection_sha256,
      "matrix_sha256": prepared.projection.matrix_sha256,
      "pseudoinverse_sha256": prepared.projection.pseudoinverse_sha256,
      "statistics_sha256": prepared.projection.statistics_sha256,
    },
    "vae": {
      "path": str(args.vae_checkpoint.resolve()),
      "sha256": args.vae_checkpoint_sha256,
      "manifest": str(args.manifest.resolve()),
      "manifest_sha256": prepared.manifest_sha256,
      "member": args.member,
      "reference_dimension": prepared.vae.reference_dim,
      "conditioning_dimension": prepared.vae.conditioning_dim,
      "latent_dimension": prepared.vae.latent_dim,
      "action_dimension": prepared.vae.action_dim,
    },
    "motion": {
      "frames": prepared.clip_frames,
      "fps": prepared.clip_fps,
      "start_frame": args.start_frame,
    },
    "budget": {
      "bootstrap_pairs": _BOOTSTRAP_PAIRS,
      "continuation_steps": args.continuation_steps,
      "max_wall_seconds": args.max_wall_seconds,
      "max_output_bytes": args.max_output_bytes,
    },
    "sampler": {
      "ddim_updates": _FROZEN_DDIM_STEPS,
      "start_level": _FROZEN_START_LEVEL,
      "eta": 0.0,
      "guidance_strength": 0.0,
      "conditioning": "measured runtime gravity + cached actor q/dq/gyro/previous action (99-D)",
      "vae_convention": "reference 68-D, conditioning 99-D, raw mean latent 32-D",
    },
    "device": str(prepared.device),
    "artifact_root": str(prepared.repo_root),
    "source_provenance": dict(prepared.source_fingerprints),
    "frozen_file_integrity": dict(prepared.integrity_before["files"]),
  }
  task_report = _task_config_report(getattr(prepared, "task", None))
  if task_report is not None:
    report["task"] = task_report
    report["sampler"]["guidance_strength"] = task_report["guidance_strength"]
    report["sampler"]["guidance_backend"] = (
      "pose_cost_input_vjp" if task_report["guidance_strength"] > 0 else "none"
    )
  return report


def _runtime_factory(prepared: PreparedEvaluation):
  from mjlab.tasks.tracking.diffusion.closed_loop_runtime_x2 import (
    build_closed_loop_runtime,
  )

  args = prepared.args

  def factory() -> Any:
    return build_closed_loop_runtime(
      manifest_path=args.manifest,
      repo_root=prepared.repo_root,
      member=args.member,
      device=str(prepared.device),
      seed=args.seed,
    )

  return factory


def _continuation_factory(
  prepared: PreparedEvaluation, conditioning: TaskConditioning | None = None
):
  from mjlab.tasks.tracking.diffusion.sampler import SamplerConfig

  args = prepared.args
  task = getattr(prepared, "task", None)
  if task is not None and conditioning is None:
    raise ClosedLoopCliError(
      "a task config requires its task conditioning to be constructed first"
    )
  sampler_config = SamplerConfig(
    prepared.denoiser.grid,
    eta=0.0,
    guidance_strength=0.0,
    initial_noise_scale=1.0,
    start_level=_FROZEN_START_LEVEL,
  )

  def decode(latent: Tensor, conditioning_tensor: Tensor) -> Tensor:
    with torch.no_grad():
      return prepared.vae.model.decode(latent, conditioning_tensor)

  def factory() -> ReferenceFreeContinuation:
    generator = torch.Generator(device=prepared.device)
    generator.manual_seed(args.seed)
    return build_conditioned_continuation(
      conditioning=conditioning,
      denoiser=prepared.denoiser.model,
      decode=decode,
      schedule=prepared.denoiser.schedule,
      grid=prepared.denoiser.grid,
      projection=prepared.projection,
      contract=prepared.contract,
      device=prepared.device,
      generator=generator,
      sampler_config=sampler_config,
    )

  return factory


def _provenance(prepared: PreparedEvaluation, started: float) -> dict[str, Any]:
  args = prepared.args
  source_provenance = _source_fingerprints()
  provenance: dict[str, Any] = {
    "cli": str(Path(__file__).resolve()),
    "artifact_root": str(prepared.repo_root),
    "source_provenance": source_provenance,
    "environment": {
      "artifact_root": str(prepared.repo_root),
      "cwd": str(Path.cwd()),
      "python": platform.python_version(),
      "torch": torch.__version__,
      "numpy": np.__version__,
      "device": str(prepared.device),
      "member": args.member,
      "seed": args.seed,
      "start_frame": args.start_frame,
      "clip_frames": prepared.clip_frames,
      "clip_fps": prepared.clip_fps,
      "control_period_seconds": _FROZEN_CONTROL_PERIOD,
    },
    "conditioning": {
      "source": "real mjlab runtime projected_gravity_b plus cached actor base_ang_vel/joint_pos/joint_vel/previous_applied_action",
      "dimension": 99,
      "reference_source": "real mjlab motion command reference first 62 values plus motion_anchor_ori_b",
      "reference_dimension": 68,
      "continuation_reference": "absent after enter_reference_free",
    },
    "checkpoint": {
      "path": str(args.checkpoint.resolve()),
      "sha256": args.checkpoint_sha256,
      "selected_step": prepared.denoiser.global_step,
      "training_budget_updates": _TRAINING_BUDGET_UPDATES,
      "selection_split": prepared.denoiser.selection_split,
      "best_validation_loss": prepared.denoiser.best_validation_loss,
      "ema_selected": prepared.denoiser.ema_selected,
      "ema_updates": prepared.denoiser.ema_updates,
      "fp32_checked": prepared.denoiser.fp32_checked,
      "config_seed": prepared.denoiser.config_seed,
      "contract_identity": prepared.denoiser.contract_identity,
      "config_hash": prepared.denoiser.config_hash,
      "projection_hashes": dict(prepared.denoiser.projection_hashes),
    },
    "projection": {
      "path": str(args.projection.resolve()),
      "sha256": args.projection_sha256,
      "matrix_sha256": prepared.projection.matrix_sha256,
      "pseudoinverse_sha256": prepared.projection.pseudoinverse_sha256,
      "statistics_sha256": prepared.projection.statistics_sha256,
    },
    "vae": {
      "path": str(args.vae_checkpoint.resolve()),
      "sha256": args.vae_checkpoint_sha256,
      "manifest": str(args.manifest.resolve()),
      "manifest_sha256": prepared.manifest_sha256,
      "member": args.member,
      "member_artifact_hashes": dict(prepared.teacher_hashes),
      "relocated_artifact_roles": list(prepared.vae.relocated_artifact_roles),
      "joint_order": list(prepared.vae.joint_order),
    },
    "sampler": {
      "inference_grid": prepared.denoiser.grid.as_dict(),
      "ddim_updates": _FROZEN_DDIM_STEPS,
      "start_level": _FROZEN_START_LEVEL,
      "eta": 0.0,
      "guidance_strength": 0.0,
      "initial_noise_scale": 1.0,
    },
    "budget": {
      "bootstrap_pairs": _BOOTSTRAP_PAIRS,
      "continuation_steps": args.continuation_steps,
      "max_wall_seconds": args.max_wall_seconds,
      "max_output_bytes": args.max_output_bytes,
      "started_monotonic": started,
    },
    "frozen_file_integrity": dict(prepared.integrity_before["files"]),
  }
  task_report = _task_config_report(getattr(prepared, "task", None))
  if task_report is not None:
    provenance["task"] = task_report
    provenance["sampler"]["guidance_strength"] = task_report["guidance_strength"]
    provenance["sampler"]["guidance_backend"] = (
      "pose_cost_input_vjp" if task_report["guidance_strength"] > 0 else "none"
    )
  return provenance


def _mark_report_partial(report: EvaluationReport, message: str) -> None:
  """Make a software/integrity failure visible to status and CLI dispatch."""
  object.__setattr__(report, "status", "partial")
  if message not in report.errors:
    object.__setattr__(report, "errors", (*report.errors, message))


def _report_exit_code(report: EvaluationReport) -> int:
  """Return nonzero for software/report failures, not for valid physical falls."""
  if report.status != "complete" or not report.valid or report.errors:
    return 1
  if report.output.get("error"):
    return 1
  for key in ("integrity", "postintegrity", "after_run_integrity"):
    integrity = report.provenance.get(key)
    if isinstance(integrity, Mapping):
      if (
        integrity.get("error")
        or integrity.get("unchanged") is False
        or integrity.get("passed") is False
      ):
        return 1
  execution = report.provenance.get("execution")
  if isinstance(execution, Mapping):
    if (
      execution.get("within_wall_budget") is False
      or execution.get("within_budget") is False
      or execution.get("wall_status") in {"exceeded", "failed", "over_budget"}
    ):
      return 1
  return 0


def _stdout_payload(report: EvaluationReport, output_dir: Path) -> dict[str, Any]:
  """Print the bounded persisted representation, never an uncapped fallback."""
  summary = output_dir / "summary.json"
  try:
    payload = json.loads(summary.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
      return payload
  except (OSError, json.JSONDecodeError):
    pass
  payload = report.as_dict()
  output = payload.get("output", {})
  representation = output.get("representation") if isinstance(output, Mapping) else None
  if representation != "full":
    branches = payload.get("branches", {})
    if isinstance(branches, dict):
      for branch in branches.values():
        if isinstance(branch, dict):
          branch["ticks"] = []
          branch["ticks_omitted"] = True
  return payload


def _execute(prepared: PreparedEvaluation) -> EvaluationReport:
  args = prepared.args
  prepared.denoiser.model.to(device=prepared.device)
  prepared.denoiser.model.eval()
  vae_model = cast(Any, prepared.vae.model)
  vae_model.to(device=prepared.device)
  vae_model.eval()
  started = time.monotonic()
  output_dir = args.output_dir.resolve()
  task = getattr(prepared, "task", None)
  task_conditioning = (
    None
    if task is None
    else TaskConditioning(
      task, projection=prepared.projection, contract=prepared.contract
    )
  )
  report = run_matched_pair(
    _runtime_factory(prepared),
    model=vae_model,
    continuation_factory=_continuation_factory(prepared, task_conditioning),
    seed=args.seed,
    start_frame=args.start_frame,
    budget=EvaluationBudget(
      max_steps=args.continuation_steps,
      max_sim_time_s=None,
      bootstrap_pairs=_BOOTSTRAP_PAIRS,
      period_seconds=_FROZEN_CONTROL_PERIOD,
      output_cap_bytes=args.max_output_bytes,
    ),
    device=prepared.device,
    latent_normalizer=prepared.projection.normalize_latent,
    output_dir=output_dir,
    provenance=_provenance(prepared, started),
  )
  elapsed = time.monotonic() - started
  provenance = dict(report.provenance)
  try:
    after = _integrity_snapshot(prepared)
    unchanged = prepared.integrity_before == after
    integrity: dict[str, Any] = {
      "before": dict(prepared.integrity_before),
      "after": after,
      "unchanged": unchanged,
    }
    if not unchanged:
      integrity["error"] = "frozen model/normalizer/file integrity changed"
  except Exception as exc:
    integrity = {
      "before": dict(prepared.integrity_before),
      "after": None,
      "unchanged": False,
      "error": str(exc),
    }
  provenance["integrity"] = integrity
  try:
    provenance["source_provenance"] = _source_fingerprints()
    source_error: str | None = None
  except Exception as exc:
    source_error = str(exc)
    provenance["source_provenance_error"] = source_error
  within_wall_budget = elapsed <= args.max_wall_seconds
  provenance["execution"] = {
    "wall_seconds": elapsed,
    "max_wall_seconds": args.max_wall_seconds,
    "within_wall_budget": within_wall_budget,
    "source_provenance_error": source_error,
    "partial_outcomes_retained": True,
  }
  provenance["runtime_counters"] = {
    branch: None if branch_report is None else dict(branch_report.counters)
    for branch, branch_report in (
      ("baseline", report.baseline),
      ("diffusion", report.diffusion),
    )
  }
  if task_conditioning is not None:
    stop_reason = getattr(report.diffusion, "stop_reason", None)
    task_conditioning.close(stop_reason=stop_reason)
    try:
      provenance["task"] = task_conditioning.provenance()
    except ValueError as exc:
      provenance["task"] = {"error": str(exc)}
      _mark_report_partial(report, f"task provenance failed: {exc}")
  object.__setattr__(report, "provenance", provenance)
  if integrity.get("error"):
    _mark_report_partial(report, f"post-run integrity failed: {integrity['error']}")
  if not within_wall_budget:
    _mark_report_partial(report, "wall-clock budget exceeded")
  if source_error:
    _mark_report_partial(report, f"source provenance failed: {source_error}")
  persist_evaluation_report(report, output_dir)
  return report


def main(argv: Sequence[str] | None = None) -> int:
  parser = _parser()
  args = parser.parse_args(argv)
  try:
    prepared = _prepare(args)
    if not args.execute:
      print(json.dumps(_preflight_report(prepared), indent=1, sort_keys=True))
      return 0
    report = _execute(prepared)
  except (ClosedLoopCliError, ReplayError, OSError, ValueError) as exc:
    print(f"D3 closed-loop preflight/execution failed: {exc}", file=sys.stderr)
    return 1
  print(
    json.dumps(
      _stdout_payload(report, args.output_dir.resolve()), indent=1, sort_keys=True
    )
  )
  return _report_exit_code(report)


if __name__ == "__main__":
  raise SystemExit(main())
