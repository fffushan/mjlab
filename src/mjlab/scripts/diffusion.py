"""Fail-closed D1 diffusion data and collection command line.

The default and all offline commands operate without constructing an mjlab
environment.  A collection run is an explicit extension point: a runtime
factory must build the concrete environment and frozen policies and return a
:class:`RuntimeBundle`.  This keeps the simulator-specific adapter owned by
the deployment/runtime integration while making the operational safety checks
executable here.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence, cast

from mjlab.tasks.tracking.diffusion.model import denoiser_dtype
from mjlab.tasks.tracking.diffusion.runtime import CollectionRequest, RuntimeBundle

DEFAULT_CONTRACT_PATH = Path(
  "docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml"
)
DEFAULT_CONFIG_PATH = Path("configs/diffusion/x2_50hz_collection.yaml")
# Hard collection ceilings the CLI will not exceed.  The *pilot* budget (54
# trials / 13500 transitions / 15 min) is the frozen D0 contract and
# configs/diffusion/x2_50hz_collection.yaml; these are the outer limits, raised
# for the post-pilot bulk collection after the pilot's budgets were reviewed.
MAX_COLLECTION_TRIALS = 5_000
# Sanity ceiling for the declared environment budget.  There is deliberately no
# semantic constraint here any more.  The collector used to fail closed unless
# max_envs was 1, because exactly one environment identity was carried
# end-to-end.  A shared multi-motion environment removes that reason: a local
# measurement on 2026-10-06 held 30 reference clips and 64 parallel rows in one
# environment of 4.3 GiB, against 36.9 GiB for 30 single-motion environments.
# The runtime still builds one environment per runtime identity and reports the
# budget it was given, so a larger declared budget is not yet a promise that
# that many environments are created.
MAX_COLLECTION_ENVS = 4_096
# Per-trial no-progress tolerance.  The CLI default stays at 120 s, but a longer
# explicit value is allowed: a shared-storage stall of about ten minutes stopped a
# 2565-trial collection at 47.8% (2026-10-06), and --max-wall-minutes already
# bounds a genuine hang.
MAX_MONITOR_INACTIVITY_SECONDS = 1_800


@dataclass(frozen=True)
class PreflightReport:
  """Serializable result of contract and pinned-artifact validation."""

  ok: bool
  contract_sha256: str | None
  artifacts: dict[str, dict[str, Any]]
  errors: tuple[str, ...]
  layout: dict[str, int] | None = None

  def as_dict(self) -> dict[str, Any]:
    return {
      "ok": self.ok,
      "contract_sha256": self.contract_sha256,
      "artifacts": self.artifacts,
      "errors": list(self.errors),
      "layout": self.layout,
    }


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _load_contract(path: Path) -> Any:
  from mjlab.tasks.tracking.diffusion.contract import DiffusionContract

  return DiffusionContract.from_yaml(path)


def run_preflight(
  contract_path: str | Path = DEFAULT_CONTRACT_PATH,
  *,
  artifact_root: str | Path = ".",
) -> PreflightReport:
  """Validate the immutable D0 contract and exact selected artifact bytes."""
  contract_file = Path(contract_path)
  root = Path(artifact_root)
  artifacts: dict[str, dict[str, Any]] = {}
  errors: list[str] = []
  contract_sha: str | None = None
  layout: dict[str, int] | None = None
  try:
    contract = _load_contract(contract_file)
    contract_sha = contract.contract_sha256
  except Exception as exc:  # report a CLI failure, never hide a bad manifest
    return PreflightReport(False, None, {}, (f"contract: {exc}",))
  # Report the declared state/latent/token layout so a reduced 129-D
  # local-kinematic contract is visible, and reject an inconsistent one before
  # any artifact or model is loaded.
  from mjlab.tasks.tracking.diffusion.model import contract_dimensions

  try:
    dimensions = contract_dimensions(contract)
  except ValueError as exc:
    errors.append(f"layout: {exc}")
  else:
    layout = {
      "state_dimension": dimensions.state_dimension,
      "latent_dimension": dimensions.latent_dimension,
      "token_dimension": dimensions.token_dimension,
      "window_steps": dimensions.window_steps,
    }

  identities = (
    ("vae", contract.vae_checkpoint, contract.vae_sha256),
    ("recovery", contract.recovery_onnx, contract.recovery_sha256),
    ("x2_xml", contract.state_xml, contract.state_xml_sha256),
    (
      "physical_fall_source",
      contract.physical_fall_source,
      contract.physical_fall_source_sha256,
    ),
    (
      "vae_tracking_rejection_source",
      contract.vae_tracking_rejection_source,
      contract.vae_tracking_rejection_source_sha256,
    ),
  )
  for role, relative, expected in identities:
    path = root / relative
    entry: dict[str, Any] = {
      "path": str(path),
      "expected_sha256": expected,
      "exists": path.is_file(),
    }
    if path.is_file():
      actual = _sha256(path)
      entry["sha256"] = actual
      entry["matches"] = actual == expected
      if actual != expected:
        errors.append(f"{role}: sha256 mismatch for {path}")
    else:
      entry["sha256"] = None
      entry["matches"] = False
      errors.append(f"{role}: missing pinned artifact {path}")
    artifacts[role] = entry
  return PreflightReport(not errors, contract_sha, artifacts, tuple(errors), layout)


def _load_factory(spec: str) -> Callable[[CollectionRequest], RuntimeBundle]:
  if ":" not in spec:
    raise ValueError("runtime factory must use MODULE:FUNCTION syntax")
  module_name, function_name = spec.split(":", 1)
  if not module_name or not function_name:
    raise ValueError("runtime factory must use MODULE:FUNCTION syntax")
  module = importlib.import_module(module_name)
  factory = getattr(module, function_name, None)
  if not callable(factory):
    raise TypeError(f"runtime factory is not callable: {spec}")
  return cast(Callable[[CollectionRequest], RuntimeBundle], factory)


def _json_write(path: Path, payload: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _yaml_write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
  """Write a small YAML artifact without exposing a partial file."""
  try:
    import yaml

    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
      yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8"
    )
    temporary.replace(path)
  except (ImportError, OSError, TypeError, ValueError) as exc:
    raise ValueError(f"could not write YAML artifact {path}: {exc}") from exc


def _repository_root(path: Path) -> Path:
  """Find the checkout root without serializing a machine-local path."""
  path = path.resolve()
  for candidate in (path.parent, *path.parents):
    if (candidate / "src" / "mjlab").is_dir() and (candidate / "configs").is_dir():
      return candidate
  raise ValueError(f"could not identify mjlab repository root from {path}")


def _repo_relative(path: Path, root: Path) -> str:
  """Return a repository-relative artifact reference."""
  path = path.resolve()
  try:
    return path.relative_to(root).as_posix()
  except ValueError as exc:
    raise ValueError(
      f"artifact path is outside repository root {root}: {path}"
    ) from exc


def _resolve_repo_path(value: object, root: Path, *, field: str) -> Path:
  if not isinstance(value, str) or not value:
    raise ValueError(f"contract field {field!r} must be a non-empty path")
  path = Path(value)
  return path.resolve() if path.is_absolute() else (root / path).resolve()


def _diffusion_artifact_metadata(
  *,
  args: argparse.Namespace,
  contract: Any,
  contract_payload: Mapping[str, Any],
  source: Any,
  output_dir: Path,
  config_path: Path,
  seed: int,
) -> dict[str, Any]:
  """Build portable run and checkpoint metadata from resolved training inputs."""
  root = _repository_root(Path(args.contract))
  vae_payload = contract_payload.get("vae")
  if not isinstance(vae_payload, Mapping):
    raise ValueError("contract must contain a vae mapping for checkpoint metadata")
  manifest_path = _resolve_repo_path(
    vae_payload.get("cohort_manifest"), root, field="vae.cohort_manifest"
  )
  vae_path = _resolve_repo_path(
    vae_payload.get("checkpoint", contract.vae_checkpoint),
    root,
    field="vae.checkpoint",
  )
  projection_path = Path(source.directory).resolve() / "offline" / "projection.npz"
  if not projection_path.is_file():
    raise ValueError(f"training projection bundle is missing: {projection_path}")
  try:
    import yaml

    manifest_payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
  except (ImportError, OSError, TypeError, ValueError) as exc:
    raise ValueError(
      f"could not read VAE cohort manifest {manifest_path}: {exc}"
    ) from exc
  cohort_members: list[str] = []
  declared_members = vae_payload.get("cohort_ids")
  if isinstance(declared_members, list):
    cohort_members = [str(value) for value in declared_members if str(value)]
  if not cohort_members and isinstance(manifest_payload, Mapping):
    teachers = manifest_payload.get("teachers")
    if isinstance(teachers, list):
      cohort_members = [
        str(item["id"])
        for item in teachers
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
      ]
  if not cohort_members:
    raise ValueError(f"VAE cohort manifest has no teacher members: {manifest_path}")
  selected_members = [str(value) for value in args.motion_id if str(value)]
  default_member = selected_members[0] if selected_members else cohort_members[0]
  sidecar_root = os.path.relpath(root, output_dir.resolve())
  artifact_paths = {
    "contract": _repo_relative(Path(args.contract), root),
    "projection": _repo_relative(projection_path, root),
    "vae_checkpoint": _repo_relative(vae_path, root),
    "manifest": _repo_relative(manifest_path, root),
    "training_config": _repo_relative(config_path, root),
  }
  run_manifest = {
    "schema": "mjlab-diffusion-run-v1",
    "paths_relative_to": "repository_root",
    "artifacts": artifact_paths,
    "cohort_members": cohort_members,
    "training": {"seed": seed},
  }
  checkpoint_sidecar = {
    "schema": "mjlab-d3-diffusion-checkpoint-v1",
    "repo_root": sidecar_root,
    **artifact_paths,
    "cohort_members": cohort_members,
    "member": default_member,
    "seed": seed,
    "start_frame": 45,
    "run_manifest": "run-manifest.yaml",
  }
  return {"run_manifest": run_manifest, "checkpoint_sidecar": checkpoint_sidecar}


def _trial_fingerprint(spec: Any) -> tuple[Any, ...]:
  return (
    getattr(spec, "run_id", None),
    getattr(spec, "motion_id", None),
    getattr(spec, "seed", None),
    getattr(spec, "start_frame", None),
    getattr(spec, "pair_id", None),
    getattr(spec, "initial_state_id", None),
  )


def _ou_pair_block_reason(clean_result: Any) -> str | None:
  """Why one clean pair cannot run its OU counterpart, or ``None`` if it can.

  A qualified clean partner is enough.  A missing clip-end handoff is a
  ``vae_only`` outcome, not a block: the perturbation axis needs a clean rollout
  that survived its window, not an endpoint audit.
  """
  if clean_result is None:
    return "clean partner was not collected"
  qualification = clean_result.qualification
  if not qualification.qualified:
    return f"clean partner did not qualify ({qualification.reason or 'failed'})"
  return None


def _validate_trial_pairs(trials: tuple[Any, ...]) -> str | None:
  """Require each OU trial to follow its own clean partner.

  The former global phase check protected a process-wide snapshot map.  The
  collector now checks each pair fingerprint and qualification at admission,
  so unrelated clean and OU pairs may interleave.
  """
  clean: dict[str, Any] = {}
  seen_ou: set[str] = set()
  for spec in trials:
    phase = getattr(spec, "phase", None)
    pair_id = getattr(spec, "pair_id", "")
    initial_state_id = getattr(spec, "initial_state_id", "")
    if phase == "clean":
      if not pair_id or not initial_state_id or pair_id in clean:
        return "clean trials require unique pair and initial-state identities"
      clean[pair_id] = spec
    elif phase == "ou":
      if not pair_id or not initial_state_id:
        return "OU trials require a clean pair and initial-state identity"
      partner = clean.get(pair_id)
      if partner is None:
        return "every OU trial must follow its clean pair"
      if pair_id in seen_ou:
        return "each clean pair may have only one OU trial"
      if _trial_fingerprint(spec) != _trial_fingerprint(partner):
        return "OU trial does not match its clean initialization fingerprint"
      seen_ou.add(pair_id)
    else:
      return "trial phase must be clean or ou"
  if set(clean) != seen_ou:
    return "complete clean/OU pair matrix is required before collection"
  return None


def _validate_runtime_bundle(
  bundle: RuntimeBundle,
  *,
  max_envs: int | None = None,
  max_gpus: int | None = None,
  contract: Any | None = None,
) -> str | None:
  if not bundle.runtime_verified:
    return "runtime factory is not a verified concrete X2 runtime"
  provenance = bundle.runtime_provenance or {}
  if provenance.get("environment") != "agibot_x2":
    return "runtime provenance must identify agibot_x2"
  for name in ("max_envs", "max_gpus"):
    if name not in provenance:
      return f"runtime provenance must report honored {name}"
  if max_envs is not None and str(provenance["max_envs"]) != str(max_envs):
    return "runtime did not honor the requested max_envs budget"
  if max_gpus is not None and str(provenance["max_gpus"]) != str(max_gpus):
    return "runtime did not honor the requested max_gpus budget"

  from mjlab.tasks.tracking.diffusion.adapter import FrozenVaePolicy
  from mjlab.tasks.tracking.diffusion.collector import DiffusionCollector

  collector = bundle.collector
  if not isinstance(collector, DiffusionCollector):
    return "runtime collector must be the concrete DiffusionCollector type"
  if not isinstance(collector.vae, FrozenVaePolicy):
    return "runtime VAE binding is not a FrozenVaePolicy"
  if collector.vae.artifact_path is None or collector.vae.artifact_sha256 is None:
    return "runtime VAE binding has no frozen artifact identity"
  from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT

  selected_contract = contract if contract is not None else DEFAULT_CONTRACT
  # D1 drives the frozen VAE for the whole rollout, so the VAE is the only
  # policy artifact the collector binds.
  for name, policy, expected_hash in (
    ("VAE", collector.vae, selected_contract.vae_sha256),
  ):
    path = policy.artifact_path
    if not isinstance(path, Path) or not path.is_file():
      return f"runtime {name} binding artifact path is not a file"
    actual_hash = _sha256(path)
    if actual_hash != policy.artifact_sha256 or actual_hash != expected_hash:
      return f"runtime {name} binding artifact bytes are not the pinned contract asset"
    metadata = policy.artifact_metadata
    try:
      policy.action_contract.verify_metadata(metadata, require_actuator_metadata=True)
    except Exception as exc:
      return f"runtime {name} action metadata binding is invalid: {exc}"
  if collector.vae_actions.joint_order != collector.vae.action_contract.joint_order:
    return "runtime collector action bindings disagree with policy joint order"
  import numpy as np

  for collector_contract, policy_contract in (
    (collector.vae_actions, collector.vae.action_contract),
  ):
    if not all(
      np.allclose(
        getattr(collector_contract, name),
        getattr(policy_contract, name),
        atol=1.0e-3,
        rtol=0.0,
      )
      for name in ("scale", "offset", "stiffness", "damping")
    ):
      return "runtime collector action bindings disagree with policy metadata"
  if not callable(getattr(collector, "collect", None)):
    return "runtime collector has no callable collect method"
  import inspect

  if "remaining_transitions" not in inspect.signature(collector.collect).parameters:
    return "runtime collector does not expose a bounded transition interface"
  return None


def _common(parser: argparse.ArgumentParser) -> None:
  parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
  parser.add_argument("--artifact-root", type=Path, default=Path("."))


def _command_preflight(args: argparse.Namespace) -> int:
  report = run_preflight(args.contract, artifact_root=args.artifact_root)
  print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
  return 0 if report.ok else 1


def _load_collection_config(path: Path) -> dict[str, Any]:
  """Load the non-authorizing collection input without constructing a runtime."""
  try:
    import yaml
  except ImportError as exc:  # pragma: no cover - project environments include yaml
    raise RuntimeError("collection config requires PyYAML") from exc
  payload = yaml.safe_load(path.read_text())
  if not isinstance(payload, dict) or payload.get("schema_version") != (
    "mjlab-x2-diffusion-d1-collection-v1"
  ):
    raise ValueError("invalid D1 collection config schema")
  if payload.get("execution_authorized") is not False:
    raise ValueError("collection config must remain non-authorizing")
  pilot = payload.get("pilot")
  if isinstance(pilot, dict):
    try:
      declared_envs = int(cast(Any, pilot.get("max_envs")))
    except (TypeError, ValueError) as exc:
      raise ValueError(
        "collection config pilot.max_envs must be a positive integer"
      ) from exc
    if declared_envs < 1:
      raise ValueError("collection config pilot.max_envs must be a positive integer")
  return payload


def _validate_resource_budgets(args: argparse.Namespace) -> str | None:
  limits = (
    ("max_trials", args.max_trials, 1, MAX_COLLECTION_TRIALS),
    ("max_envs", args.max_envs, 1, MAX_COLLECTION_ENVS),
    ("max_gpus", args.max_gpus, 1, 1),
    ("max_wall_minutes", args.max_wall_minutes, 1, 720),
    (
      "monitor_inactivity_seconds",
      args.monitor_inactivity_seconds,
      1,
      MAX_MONITOR_INACTIVITY_SECONDS,
    ),
    ("max_output_mib", args.max_output_mib, 1, 4096),
  )
  for name, value, lower, upper in limits:
    if value < lower or value > upper:
      return f"{name} must be in [{lower}, {upper}]"
  return None


def _validate_streaming_controls(args: argparse.Namespace) -> str | None:
  """Refuse an explicit speed option that the collector cannot honour."""
  if args.max_rows_per_append is not None:
    if args.max_rows_per_append < 1:
      return "max_rows_per_append must be one or more"
    if args.stream_rows is False:
      return "max_rows_per_append requires --stream-rows"
  if args.compact_results and args.stream_rows is False:
    return "compact_results requires --stream-rows"
  return None


def _batched_only_option(args: argparse.Namespace) -> str | None:
  """Name the first requested batched-only option, or ``None``.

  The X2 runtime factory selects the batched runtime above one environment.
  A scalar request cannot honour these options, so the CLI refuses them
  instead of accepting a speed request and silently ignoring it.
  """
  for flag, requested in (
    ("--stream-rows", args.stream_rows),
    ("--compact-results", args.compact_results),
    ("--max-rows-per-append", args.max_rows_per_append is not None),
    ("--tensor-dispatch", args.tensor_dispatch),
    ("--progress", args.progress is not None),
  ):
    if requested:
      return flag
  return None


def _scalar_control_error(flag: str) -> str:
  return (
    f"{flag} is honoured only by the batched runtime; raise --max-envs above 1 "
    "or drop the option"
  )


@contextlib.contextmanager
def _policy_device_environment(setting: str | None) -> Iterator[None]:
  """Scope ``MJLAB_D1_POLICY_DEVICE`` to one runtime-factory call.

  The factory resolves the policy device once, while it constructs the frozen
  policies.  The override must not outlive that call: a later collection in the
  same process would otherwise inherit the earlier device instead of the
  documented ``cpu`` default.
  """
  if setting is None:
    yield
    return
  previous = os.environ.get("MJLAB_D1_POLICY_DEVICE")
  os.environ["MJLAB_D1_POLICY_DEVICE"] = setting
  try:
    yield
  finally:
    if previous is None:
      os.environ.pop("MJLAB_D1_POLICY_DEVICE", None)
    else:
      os.environ["MJLAB_D1_POLICY_DEVICE"] = previous


def _rollout_summary_payload(qualification: Any) -> dict[str, Any]:
  """The endpoint verdict and the per-rollout stability measurements."""
  return {
    "qualified": qualification.qualified,
    "label": qualification.label,
    "reason": qualification.reason,
    "steps": qualification.steps,
    "min_tilt_degrees": qualification.min_tilt_degrees,
    "max_anchor_z_error_m": qualification.max_anchor_z_error_m,
    "max_end_effector_z_error_m": qualification.max_end_effector_z_error_m,
    "fall_frames": qualification.fall_frames,
  }


@contextlib.contextmanager
def _schedule_environment(
  seeds: int | None, initial_poses: str | None
) -> Iterator[None]:
  """Scope the schedule overrides to the factory call that reads them.

  The runtime factory builds the trial schedule itself, so the CLI cannot pass
  these values as arguments.  Restoring the previous environment keeps a later
  collection in the same process on its own configuration.
  """
  overrides = {
    "MJLAB_D1_BULK_SEEDS": None if seeds is None else str(seeds),
    "MJLAB_D1_INITIAL_POSES": initial_poses,
  }
  previous = {name: os.environ.get(name) for name in overrides}
  try:
    for name, value in overrides.items():
      if value is not None:
        os.environ[name] = value
    yield
  finally:
    for name, value in previous.items():
      if value is None:
        os.environ.pop(name, None)
      else:
        os.environ[name] = value


def _batched_outcome_payload(result: Any) -> dict[str, Any]:
  """Serialize one batched trial outcome for full rows or compact summaries."""
  rows = getattr(result, "rows", None)
  if rows is None:
    row_count = int(result.row_count)
    accepted_rows = int(result.accepted_row_count)
    verification_rows = int(result.verification_row_count)
  else:
    row_count = len(rows)
    accepted_rows = len(result.accepted_rows)
    verification_rows = len(result.verification_rows)
  return {
    "run_id": result.spec.run_id,
    "motion_id": result.spec.motion_id,
    "phase": result.spec.phase,
    "initial_pose": result.spec.initial_pose,
    "initial_state_mode": result.spec.initial_state_mode,
    "rows": row_count,
    "accepted_rows": accepted_rows,
    "verification_rows": verification_rows,
    "stopped_reason": result.stopped_reason,
    "store_files": list(result.store_files),
    **_rollout_summary_payload(result.qualification),
  }


def _progress_callback(path: Path | None) -> Callable[[Any], None] | None:
  """Append one JSON progress record per event, flushed as it happens."""
  if path is None:
    return None
  path.parent.mkdir(parents=True, exist_ok=True)
  handle = path.open("a", encoding="utf-8")

  def report(event: Any) -> None:
    handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")
    handle.flush()

  return report


def _command_inspect(args: argparse.Namespace) -> int:
  from mjlab.tasks.tracking.diffusion.offline_validation import inspect_store

  if args.jobs < 1:
    print("inspect: jobs must be one or more", file=sys.stderr)
    return 2
  contract = _load_contract(args.contract)
  payload = inspect_store(args.store, contract=contract, jobs=args.jobs)
  print(json.dumps(payload, indent=2, sort_keys=True))
  return 0


def _command_replay(args: argparse.Namespace) -> int:
  from mjlab.tasks.tracking.diffusion.offline_validation import replay_store

  if args.jobs < 1:
    print("replay: jobs must be one or more", file=sys.stderr)
    return 2
  contract = _load_contract(args.contract)
  payload = replay_store(args.store, contract=contract, jobs=args.jobs)
  print(json.dumps(payload, indent=2, sort_keys=True))
  return 0 if payload["ok"] else 1


def _load_assignments(path: Path) -> Any:
  from mjlab.tasks.tracking.diffusion.dataset import SplitAssignments

  return SplitAssignments.from_dict(json.loads(path.read_text()))


def _command_build(args: argparse.Namespace) -> int:
  from mjlab.tasks.tracking.diffusion.dataset import (
    build_window_index,
    grouped_split,
  )
  from mjlab.tasks.tracking.diffusion.storage import AppendOnlyShardStore

  if args.jobs < 1:
    print("build: jobs must be one or more", file=sys.stderr)
    return 2
  contract = _load_contract(args.contract)
  store = AppendOnlyShardStore(args.store, contract=contract)
  assignments = grouped_split(store.iter_rows(), contract=contract)
  index, method = build_window_index(
    store, assignments, jobs=args.jobs, contract=contract
  )
  _json_write(args.assignments, assignments.as_dict())
  payload = {
    "contract_sha256": contract.contract_sha256,
    "assignments": str(args.assignments),
    "jobs": args.jobs,
    "method": method,
    "groups": assignments.coverage(),
    "windows": index.coverage(),
    "total_windows": len(index),
  }
  _json_write(args.report, payload)
  print(json.dumps(payload, indent=2, sort_keys=True))
  return 0


def _command_stats(args: argparse.Namespace) -> int:
  from mjlab.tasks.tracking.diffusion.dataset import (
    DatasetError,
    WindowIndex,
    fit_training_statistics,
    load_window_index,
  )
  from mjlab.tasks.tracking.diffusion.storage import AppendOnlyShardStore

  if args.jobs < 1:
    print("stats: jobs must be one or more", file=sys.stderr)
    return 2
  if args.batch_windows is not None and args.batch_windows < 1:
    print("stats: batch-windows must be one or more", file=sys.stderr)
    return 2
  contract = _load_contract(args.contract)
  store = AppendOnlyShardStore(args.store, contract=contract)
  assignments = _load_assignments(args.assignments)
  index_path = args.index
  if index_path is None:
    candidate = args.assignments.parent / "window_index.npz"
    if candidate.is_file():
      index_path = candidate
  if index_path is not None:
    # Reuse the persisted, source-bound index instead of rebuilding it.  The
    # loader refuses an index that does not describe this exact store and
    # assignment set, so a stale artifact cannot change the dataset definition.
    try:
      index = load_window_index(
        index_path, store=store, assignments=assignments, contract=contract
      )
    except (DatasetError, OSError, ValueError) as exc:
      print(f"stats: refusing the persisted window index: {exc}", file=sys.stderr)
      return 1
    index_source = str(index_path)
  else:
    index = WindowIndex.build(store, assignments, contract=contract)
    index_source = "rebuilt"
  statistics_kwargs = (
    {} if args.batch_windows is None else {"batch_windows": args.batch_windows}
  )
  bundle = fit_training_statistics(
    index,
    contract=contract,
    jobs=args.jobs,
    **statistics_kwargs,
  )
  bundle.save(args.bundle)
  payload = {
    "contract_sha256": contract.contract_sha256,
    "bundle": str(args.bundle),
    "index_source": index_source,
    "jobs": args.jobs,
    "batch_windows": args.batch_windows,
    "windows": index.coverage(),
    "train_windows_used": len(index.refs_for("train")),
    "matrix_sha256": bundle.matrix_sha256,
    "statistics_sha256": bundle.statistics_sha256,
  }
  _json_write(args.report, payload)
  print(json.dumps(payload, indent=2, sort_keys=True))
  return 0


def _command_build_token_cache(args: argparse.Namespace) -> int:
  """Prepare or validate dataset-owned token caches with a split report."""
  from mjlab.tasks.tracking.diffusion.token_cache_lifecycle import (
    CacheProvenanceError,
    parse_splits,
    prepare_token_cache,
    validate_token_cache_read_only,
  )
  from mjlab.tasks.tracking.diffusion.window_dataset import (
    _DEFAULT_MAX_CACHE_GIB,
    DatasetSource,
    TokenCacheBusyError,
    WindowCacheError,
    _source_dataset_hash,
    dataset_token_cache_path,
    recover_cache_lock,
  )

  try:
    requested = parse_splits(args.splits)
  except (ValueError, TypeError) as exc:
    print(f"build-token-cache: {exc}", file=sys.stderr)
    return 1
  ceiling = _DEFAULT_MAX_CACHE_GIB if args.max_cache_gib is None else args.max_cache_gib
  payload: dict[str, object] = {
    "dataset": str(args.dataset_dir),
    "mode": "validate-only" if args.validate_only else "build",
    "splits": {},
    "complete": False,
  }
  entries = cast(dict[str, object], payload["splits"])
  try:
    source = DatasetSource(Path(args.dataset_dir), Path(args.contract))
    loaded = source.load()
    source_hash = _source_dataset_hash(source, loaded)
  except (OSError, RuntimeError, ValueError, TypeError) as exc:
    payload["error"] = str(exc)
    print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
    return 1

  if args.recover_lock:
    recovered: list[dict[str, object]] = []
    for split in requested:
      target = dataset_token_cache_path(
        source, loaded, split, motion_ids=args.motion, source_hash=source_hash
      )
      try:
        removed = recover_cache_lock(target)
        recovered.append({"split": split, "removed": bool(removed)})
      except TokenCacheBusyError as exc:
        recovered.append({"split": split, "removed": False, "error": str(exc)})
      except WindowCacheError as exc:
        recovered.append({"split": split, "removed": False, "error": str(exc)})
    payload["recovered_locks"] = recovered

  failed = False

  def ready_entry(item: object) -> bool:
    if not isinstance(item, dict):
      return False
    return cast(dict[str, object], item).get("status") in {
      "built",
      "reused",
      "ready",
    }

  for split in requested:
    started = time.perf_counter()
    entry: dict[str, object] = {"status": "failed", "seconds": 0.0}
    try:
      if args.validate_only:
        prepared = validate_token_cache_read_only(
          source,
          loaded,
          split,
          motion_ids=args.motion,
          source_hash=source_hash,
        )
        status = "ready"
      else:
        prepared = prepare_token_cache(
          source,
          loaded,
          split,
          motion_ids=args.motion,
          max_cache_gib=ceiling,
          jobs=args.jobs,
          source_hash=source_hash,
          lock_timeout=args.lock_timeout,
        )
        status = prepared.cache_status
      entry.update(
        {
          "status": status,
          "path": str(prepared.path),
          "provenance_path": str(
            prepared.path.with_name(f"{prepared.path.stem}.provenance.json")
          ),
          "windows": prepared.count,
          "bytes": prepared.path.stat().st_size,
          "provenance": prepared.provenance_status,
        }
      )
      print(
        f"token cache {split}: {status} windows={prepared.count} "
        f"in {time.perf_counter() - started:.1f}s",
        flush=True,
      )
    except TokenCacheBusyError as exc:
      failed = True
      entry.update({"status": "busy", "error": str(exc)})
      print(f"token cache {split}: busy: {exc}", file=sys.stderr, flush=True)
    except (
      CacheProvenanceError,
      WindowCacheError,
      OSError,
      RuntimeError,
      ValueError,
      TypeError,
    ) as exc:
      failed = True
      entry.update(
        {"status": "missing" if args.validate_only else "failed", "error": str(exc)}
      )
      print(
        f"token cache {split}: {entry['status']}: {exc}", file=sys.stderr, flush=True
      )
    entry["seconds"] = round(time.perf_counter() - started, 3)
    entries[split] = entry
    # A report is persisted after each builder split.  Validate-only never
    # writes a report file because it is a strictly read-only operation.
    if args.json is not None and not args.validate_only:
      payload["complete"] = (
        not failed
        and len(entries) == len(requested)
        and all(ready_entry(item) for item in entries.values())
      )
      _json_write(args.json, payload)

  payload["complete"] = (
    not failed
    and len(entries) == len(requested)
    and all(ready_entry(item) for item in entries.values())
  )
  print(json.dumps(payload, indent=2, sort_keys=True))
  return 0 if payload["complete"] else 1


def _command_build_index(args: argparse.Namespace) -> int:
  """Build and persist the window index that belongs to a dataset.

  The index is derived metadata, so the dataset owns it.  This command writes
  the artifact, reads it back, and refuses a success exit unless the reloaded
  index reproduces the built one.
  """
  import time

  from mjlab.tasks.tracking.diffusion.dataset import (
    build_window_index,
    load_window_index,
    save_window_index,
    store_manifest_sha256,
    window_index_path,
  )
  from mjlab.tasks.tracking.diffusion.storage import AppendOnlyShardStore

  contract = _load_contract(args.contract)
  dataset = Path(args.dataset_dir)
  store = AppendOnlyShardStore(dataset / "store", contract=contract)
  assignments = _load_assignments(dataset / "offline" / "assignments.json")

  started = time.perf_counter()
  index, method = build_window_index(
    store, assignments, jobs=args.jobs, contract=contract
  )
  built_seconds = time.perf_counter() - started

  # Refuse to persist an index whose coverage disagrees with the dataset
  # manifest: this is the check that caught a partition that silently dropped
  # windows, and it must run before anything is written.
  manifest_path = dataset / "offline" / "dataset.json"
  if manifest_path.is_file():
    recorded = {
      str(key): int(value)
      for key, value in json.loads(manifest_path.read_text(encoding="utf-8"))[
        "windows"
      ].items()
    }
    actual = index.coverage()
    if recorded != actual:
      print(
        "refusing to persist a window index whose coverage differs from "
        f"dataset.json: expected {recorded}, got {actual}",
        file=sys.stderr,
      )
      return 1

  path = window_index_path(dataset)
  record = save_window_index(
    path,
    index,
    store_manifest_sha256=store_manifest_sha256(store),
    assignments_sha256=assignments.sha256(),
  )
  reloaded = load_window_index(
    path, store=store, assignments=assignments, contract=contract
  )
  payload = {
    "contract_sha256": contract.contract_sha256,
    "dataset": str(dataset),
    "jobs": args.jobs,
    "method": method,
    "refs": len(index),
    "coverage": index.coverage(),
    "seconds": round(built_seconds, 3),
    "reload_matches": reloaded.refs == index.refs,
    **record,
  }
  if args.json is not None:
    _json_write(args.json, payload)
  print(json.dumps(payload, indent=2, sort_keys=True))
  return 0 if payload["reload_matches"] else 1


def _command_collect(args: argparse.Namespace) -> int:
  if not args.execute:
    print("collect requires --execute; no runtime was launched", file=sys.stderr)
    return 2
  if not args.runtime_factory:
    print("collect requires --runtime-factory MODULE:FUNCTION", file=sys.stderr)
    return 2
  budget_error = _validate_resource_budgets(args)
  if budget_error is not None:
    print(budget_error, file=sys.stderr)
    return 2
  if args.seeds is not None and args.seeds < 1:
    print("--seeds must be one or more", file=sys.stderr)
    return 2
  streaming_error = _validate_streaming_controls(args)
  if streaming_error is not None:
    print(streaming_error, file=sys.stderr)
    return 2
  batched_only = _batched_only_option(args)
  if batched_only is not None and args.max_envs <= 1:
    # Refuse before any environment, checkpoint or store work: the scalar
    # runtime cannot honour a batched publication, progress or tensor option.
    print(_scalar_control_error(batched_only), file=sys.stderr)
    return 2
  collection_config: dict[str, Any] = {}
  if args.config is not None:
    try:
      collection_config = _load_collection_config(args.config)
    except (OSError, RuntimeError, ValueError) as exc:
      print(f"collection config: {exc}", file=sys.stderr)
      return 2
  report = run_preflight(args.contract, artifact_root=args.artifact_root)
  if not report.ok:
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True), file=sys.stderr)
    return 1

  from mjlab.tasks.tracking.diffusion.storage import AppendOnlyShardStore

  contract = _load_contract(args.contract)
  store = AppendOnlyShardStore(args.output, contract=contract, max_total_rows=None)
  trial_specs: tuple[Any, ...] = ()
  request = CollectionRequest(
    contract,
    store,
    args.output,
    trial_specs,
    args.max_trials,
    # No separate transition cap: the schedule and the clip lengths determine how
    # many transitions a collection produces.
    None,
    args.max_envs,
    args.max_gpus,
    args.max_wall_minutes,
    args.monitor_inactivity_seconds,
    args.max_output_mib,
  )
  started = time.monotonic()
  # The factory reads the policy-device setting once, when it builds the frozen
  # policies, so the override is scoped to this call and cannot leak into a
  # later collection in the same process.
  # The schedule size is a collection input, so an explicit CLI value wins and the
  # config file is the declared default.  Without this fallback ``schedule.seeds``
  # is inert and the runtime seed default silently applies.
  declared = collection_config.get("schedule")
  declared = declared if isinstance(declared, dict) else {}
  schedule_seeds = args.seeds
  if schedule_seeds is None:
    raw_seeds = declared.get("seeds", collection_config.get("seeds"))
    if raw_seeds is not None:
      try:
        schedule_seeds = int(raw_seeds)
      except (TypeError, ValueError):
        schedule_seeds = 0
      if schedule_seeds < 1:
        print("collection config: schedule.seeds must be a positive integer", file=sys.stderr)
        return 2
  if schedule_seeds is not None and schedule_seeds < 1:
    print("--seeds must be one or more", file=sys.stderr)
    return 2

  schedule_poses = args.initial_poses
  if schedule_poses is None:
    raw_poses = declared.get("initial_poses")
    if isinstance(raw_poses, (list, tuple)) and raw_poses:
      schedule_poses = ",".join(str(pose) for pose in raw_poses)
    elif isinstance(raw_poses, str) and raw_poses.strip():
      schedule_poses = raw_poses

  with (
    _policy_device_environment(args.policy_device),
    _schedule_environment(schedule_seeds, schedule_poses),
  ):
    bundle = _load_factory(args.runtime_factory)(request)
  if not isinstance(bundle, RuntimeBundle):
    raise TypeError("runtime factory must return RuntimeBundle")
  runtime_error = _validate_runtime_bundle(
    bundle, max_envs=args.max_envs, max_gpus=args.max_gpus, contract=contract
  )
  if runtime_error is not None:
    print(runtime_error, file=sys.stderr)
    return 2
  provenance = bundle.runtime_provenance or {}
  batch_runtime = bool(provenance.get("batched", False))
  if batched_only is not None and not batch_runtime:
    # Backstop for a factory that ignores max_envs: refuse the option instead of
    # accepting it and running the scalar path without it.
    print(_scalar_control_error(batched_only), file=sys.stderr)
    return 2
  if len(bundle.trials) == 0:
    raise ValueError("runtime factory returned no trial specifications")
  if len(bundle.trials) > args.max_trials:
    raise ValueError("runtime factory returned more trials than the requested budget")
  pair_error = _validate_trial_pairs(bundle.trials)
  if pair_error is not None:
    print(pair_error, file=sys.stderr)
    return 2
  if batch_runtime:
    from mjlab.tasks.tracking.diffusion.collection_controls import (
      CollectionControls,
      CollectionLimits,
      limits_from_request,
    )

    if args.stream_rows is None:
      args.stream_rows = True
    if args.compact_results is None:
      args.compact_results = args.stream_rows
    if args.tensor_dispatch is None:
      from mjlab.tasks.tracking.diffusion.collection_controls import (
        missing_tensor_methods,
      )

      args.tensor_dispatch = not missing_tensor_methods(
        getattr(bundle.collector, "env", None),
        getattr(bundle.collector, "vae", None),
      )
    # A supplied progress path stays explicit; the final report always has counts.
    # Wall-clock and inactivity limits apply to every batched run: the request
    # always carries a budget, so the report must state whether it bound the run.
    # The stored-output limit needs streaming publication to be measurable.
    requested_limits = limits_from_request(request)
    limits = CollectionLimits(
      wall_seconds=requested_limits.wall_seconds,
      inactivity_seconds=requested_limits.inactivity_seconds,
      max_output_bytes=(
        requested_limits.max_output_bytes if args.stream_rows else None
      ),
    )
    controls = CollectionControls(
      limits=limits,
      stream_rows=args.stream_rows,
      compact_results=args.compact_results,
      max_rows_per_append=args.max_rows_per_append,
      tensor_dispatch=args.tensor_dispatch,
      progress=_progress_callback(args.progress),
    )
    batch = bundle.collector.collect_batched(
      bundle.trials,
      slots=args.max_envs,
      controls=controls,
    )
    transitions = (
      batch.progress.transitions
      if batch.progress is not None
      else sum(result.qualification.steps for result in batch.results)
    )
    ou_skipped = len(batch.skipped_ou_pairs)
    ou_attempted = sum(result.spec.phase == "ou" for result in batch.results)
    batched_results = [_batched_outcome_payload(result) for result in batch.results]
    payload = {
      "contract_sha256": contract.contract_sha256,
      "trials_attempted": len(batched_results),
      "control_transitions": transitions,
      "ou_trials_attempted": ou_attempted,
      "ou_trials_skipped": ou_skipped,
      "abandoned_trials": [
        {"episode_id": episode_id, "reason": reason}
        for episode_id, reason in batch.abandoned_trials
      ],
      "retried_pairs": list(batch.retried_pairs),
      "partial": batch.partial,
      "stop_reason": batch.stop_reason,
      "unstarted_trials": batch.unstarted_trials,
      "stored_bytes": batch.stored_bytes,
      "output_limit_exceeded": batch.output_limit_exceeded,
      "policy_device": provenance.get("policy_device"),
      "policy_device_setting": provenance.get("policy_device_setting"),
      "streaming": {
        "stream_rows": args.stream_rows,
        "compact_results": args.compact_results,
        "max_rows_per_append": args.max_rows_per_append,
        "tensor_dispatch": args.tensor_dispatch,
      },
      "progress": None if batch.progress is None else asdict(batch.progress),
      "results": batched_results,
      "batched_store_files": list(batch.store_files),
      "note": "pilot collection only; this command does not authorize bulk collection or training",
    }
    _json_write(args.report, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if batch.partial:
      print(
        "collect: batched run is partial "
        f"(stop_reason={batch.stop_reason!r}, "
        f"{batch.unstarted_trials} trials unstarted); do not merge this store",
        file=sys.stderr,
      )
      return 1
    return 0
  results: list[dict[str, Any]] = []
  transitions = 0
  clean_results: dict[str, Any] = {}
  partial_reason: str | None = None
  ou_attempted = 0
  ou_skipped = 0
  for spec in bundle.trials:
    if len(results) >= args.max_trials:
      break
    if time.monotonic() - started >= args.max_wall_minutes * 60:
      partial_reason = "wall_clock_limit"
      results.append({"phase": "monitor", "reason": partial_reason})
      break
    if spec.phase == "ou":
      clean_result = clean_results.get(spec.pair_id)
      block = _ou_pair_block_reason(clean_result)
      if block is not None:
        # Skip only this pair and keep collecting the others.  A stop here would
        # discard every remaining OU trial because one clean partner failed.
        ou_skipped += 1
        results.append(
          {
            "phase": "monitor",
            "reason": "ou_skipped_clean_partner",
            "detail": block,
            "pair_id": spec.pair_id,
            "clean_partner": None
            if clean_result is None
            else _rollout_summary_payload(clean_result.qualification),
          }
        )
        continue
      ou_attempted += 1
    trial_started = time.monotonic()
    result = bundle.collector.collect(spec)
    trial_elapsed = time.monotonic() - trial_started
    transitions += result.qualification.steps
    if spec.phase == "clean":
      clean_results[spec.pair_id] = result
    if (
      result.qualification.steps == 0
      or trial_elapsed >= args.monitor_inactivity_seconds
    ):
      partial_reason = "inactivity_limit"
      results.append({"phase": "monitor", "reason": partial_reason})
      break
    results.append(
      {
        "run_id": spec.run_id,
        "motion_id": spec.motion_id,
        "phase": spec.phase,
        "initial_pose": spec.initial_pose,
        "initial_state_mode": spec.initial_state_mode,
        "rows": len(result.rows),
        "accepted_rows": len(result.accepted_rows),
        "verification_rows": len(result.verification_rows),
        "stopped_reason": result.stopped_reason,
        "store_files": list(result.store_files),
        **_rollout_summary_payload(result.qualification),
      }
    )
    output_bytes = sum(
      path.stat().st_size for path in args.output.glob("**/*") if path.is_file()
    )
    if output_bytes > args.max_output_mib * 1024 * 1024:
      partial_reason = "output_size_limit"
      results.append({"phase": "monitor", "reason": partial_reason})
      break

  if partial_reason is None and ou_skipped > 0 and ou_attempted == 0:
    # Every candidate pair was skipped, so the perturbation axis never ran.
    partial_reason = "ou_phase_unavailable"
  payload = {
    "contract_sha256": contract.contract_sha256,
    "trials_attempted": len(results),
    "control_transitions": transitions,
    "ou_trials_attempted": ou_attempted,
    "ou_trials_skipped": ou_skipped,
    "partial": partial_reason is not None,
    "stop_reason": partial_reason,
    "policy_device": provenance.get("policy_device"),
    "policy_device_setting": provenance.get("policy_device_setting"),
    "results": results,
    "note": "pilot collection only; this command does not authorize bulk collection or training",
  }
  _json_write(args.report, payload)
  print(json.dumps(payload, indent=2, sort_keys=True))
  return 0


def _build_seeded_model(
  settings: Any, seed: int, constructor: Callable[[Any], Any]
) -> Any:
  """Construct a model only after applying the trainer's deterministic seed."""
  from mjlab.tasks.tracking.diffusion.trainer import _set_seed

  _set_seed(seed)
  return constructor(settings)


def _attach_dataset_identity(
  dataset: Any,
  *,
  base_identity: dict[str, Any],
  split: str,
  motion_ids: tuple[str, ...] | None,
  limit: int | None = None,
) -> Any:
  """Attach the same full source identity used by cached datasets."""
  dataset.dataset_identity = {
    **base_identity,
    "split": split,
    "motion_ids": None if motion_ids is None else list(motion_ids),
    "limit": limit,
  }
  return dataset


def _checkpoint_motion_ids(identity: Any) -> tuple[str, ...] | None:
  """Read and normalize the motion filter persisted in a checkpoint."""
  raw_motion_ids = identity.get("motion_ids")
  if raw_motion_ids is None:
    return None
  if not isinstance(raw_motion_ids, (list, tuple)):
    raise ValueError("checkpoint motion identity is malformed")
  motion_ids = tuple(sorted({str(value) for value in raw_motion_ids}))
  if any(not value for value in motion_ids):
    raise ValueError("checkpoint motion identity contains an empty id")
  return motion_ids


def _command_train(args: argparse.Namespace) -> int:
  """Build cached D1 windows and run one explicitly bounded D2 training job."""
  try:
    import math

    if args.device == "cuda:0":
      os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch

    from mjlab.tasks.tracking.diffusion.contract import DiffusionContract
    from mjlab.tasks.tracking.diffusion.model import (
      DenoiserSettings,
      StateLatentTransformer,
    )
    from mjlab.tasks.tracking.diffusion.schedule import DiffusionSchedule
    from mjlab.tasks.tracking.diffusion.token_cache_lifecycle import (
      prepare_token_cache,
      validate_token_cache_read_only,
    )
    from mjlab.tasks.tracking.diffusion.trainer import DiffusionTrainer
    from mjlab.tasks.tracking.diffusion.training_config import (
      TrainingConfig,
      TrainingConfigError,
    )
    from mjlab.tasks.tracking.diffusion.window_dataset import (
      _DEFAULT_MAX_CACHE_GIB,
      DatasetSource,
      TokenWindowDataset,
      _projection_identity,
      _source_dataset_hash,
      load_window_records,
    )

    config = TrainingConfig.from_yaml(args.config)
    try:
      import yaml

      raw_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
      raise ValueError(f"could not read training config contract field: {exc}") from exc
    if isinstance(raw_config, dict) and raw_config.get("contract") is not None:
      declared_contract = Path(str(raw_config["contract"]))
      if not declared_contract.is_absolute():
        declared_contract = Path.cwd() / declared_contract
      if declared_contract.resolve() != Path(args.contract).resolve():
        raise ValueError(
          "training config contract path disagrees with --contract: "
          f"{raw_config['contract']!r}"
        )
    if args.max_updates is None and args.epochs is None and config.max_updates is None:
      raise TrainingConfigError(
        "train requires an explicit --max-updates, --epochs, or config max_updates"
      )
    if args.max_updates is not None:
      config = replace(config, max_updates=args.max_updates)
    if args.epochs is not None:
      config = replace(config, epochs=args.epochs)
    config.validate(allow_long_run=args.allow_long_run)

    if args.cache_tokens and args.prepare_token_cache and args.cache_jobs < 1:
      raise ValueError("--cache-jobs must be one or more")
    if args.cache_tokens and args.prepare_token_cache and args.cache_lock_timeout < 0:
      raise ValueError("--cache-lock-timeout must be non-negative")

    contract = DiffusionContract.from_yaml(args.contract)
    try:
      contract_payload = yaml.safe_load(args.contract.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
      raise ValueError(f"could not read contract metadata: {exc}") from exc
    if not isinstance(contract_payload, Mapping):
      raise ValueError("diffusion contract metadata must be a YAML mapping")
    source = DatasetSource.resolve(args.dataset_dir, contract_path=args.contract)
    loaded = source.load()
    schedule = DiffusionSchedule.from_contract(contract)
    selected_motions = (
      tuple(sorted({str(value) for value in args.motion_id}))
      if args.motion_id
      else None
    )
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_metadata = _diffusion_artifact_metadata(
      args=args,
      contract=contract,
      contract_payload=contract_payload,
      source=source,
      output_dir=output_dir,
      config_path=args.config,
      seed=config.seed,
    )
    _yaml_write_atomic(
      output_dir / "run-manifest.yaml", artifact_metadata["run_manifest"]
    )

    source_hash = _source_dataset_hash(source, loaded)
    projection_hash = _projection_identity(loaded.projection)
    base_identity = {
      "directory": str(source.directory),
      "assignments_hash": loaded.assignments.sha256(),
      "split_coverage": loaded.index.coverage(),
      "source_dataset_hash": source_hash,
      "contract_hash": contract.identity_hash(),
      "projection_hash": projection_hash,
      "projection_hashes": {
        "matrix": loaded.projection.matrix_sha256,
        "pseudoinverse": loaded.projection.pseudoinverse_sha256,
        "statistics": loaded.projection.statistics_sha256,
      },
    }

    def make_dataset(split: str) -> tuple[Any, list[Any], Path | None]:
      if args.cache_tokens:
        # A normal training run is a read-only cache consumer.  Building is an
        # explicit preparation action, so a missing cache cannot trigger a
        # hidden worker pool or a partial cache publication.
        if args.require_prepared_cache and not args.prepare_token_cache:
          prepared = validate_token_cache_read_only(
            source,
            loaded,
            split,
            motion_ids=selected_motions,
            source_hash=source_hash,
          )
        else:
          prepared = prepare_token_cache(
            source,
            loaded,
            split,
            motion_ids=selected_motions,
            max_cache_gib=(
              _DEFAULT_MAX_CACHE_GIB
              if args.max_cache_gib is None
              else args.max_cache_gib
            ),
            jobs=args.cache_jobs,
            source_hash=source_hash,
            lock_timeout=args.cache_lock_timeout,
          )
        provenance = list(prepared.provenance)
        dataset = TokenWindowDataset(
          prepared.path,
          provenance,
          source=source,
          split=split,
          motion_ids=selected_motions,
          validated_cache=prepared.cache,
        )
        return dataset, provenance, prepared.path

      records = load_window_records(
        loaded,
        split,
        motion_ids=selected_motions,
      )
      if records:
        values = torch.from_numpy(np.stack([record.tokens for record in records]))
      else:
        values = torch.empty(
          (0, contract.window_steps, contract.token_dimension), dtype=torch.float32
        )
      dataset = _attach_dataset_identity(
        torch.utils.data.TensorDataset(values),
        base_identity=base_identity,
        split=split,
        motion_ids=selected_motions,
      )
      dataset.records = records
      return dataset, records, None

    train_dataset, train_records, train_cache = make_dataset("train")
    validation_dataset, _, validation_cache = make_dataset("validation")
    test_dataset, _, test_cache = make_dataset("test")
    if len(train_dataset) == 0:
      raise ValueError("training split is empty after motion filtering")

    micro_batches = math.ceil(len(train_dataset) / config.microbatch_size)
    updates_per_epoch = math.ceil(micro_batches / config.gradient_accumulation_steps)
    resolved_updates = config.validate_resolved_updates(
      updates_per_epoch, allow_long_run=args.allow_long_run
    )
    settings = DenoiserSettings.from_contract(contract, training_k=schedule.training_k)
    model = _build_seeded_model(settings, config.seed, StateLatentTransformer)
    pure_dtype = denoiser_dtype(config.mixed_precision)
    if pure_dtype is not None:
      model = model.to(pure_dtype)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
      raise RuntimeError("CUDA device requested but CUDA is unavailable")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    cache_paths = [
      path for path in (train_cache, validation_cache, test_cache) if path is not None
    ]
    cache_bytes = sum(path.stat().st_size for path in cache_paths if path.is_file())
    train_identity = getattr(train_dataset, "dataset_identity", None)
    if not isinstance(train_identity, dict):
      train_identity = {
        **base_identity,
        "split": "train",
        "motion_ids": None if selected_motions is None else list(selected_motions),
        "limit": None,
      }
    preamble = {
      "config_sha256": config.sha256(),
      "contract_identity": contract.identity_hash(),
      "dataset_identity": train_identity,
      "cache_paths": [str(path) for path in cache_paths],
      "cache_bytes": cache_bytes,
      "planned_optimizer_updates": resolved_updates,
      "updates_per_epoch": updates_per_epoch,
      "effective_batch_size": config.effective_batch_size,
      "evaluation_split": config.eval_split,
      "device": str(device),
      "device_name": device_name,
      "train_windows": len(train_records),
    }
    print(json.dumps(preamble, indent=2, sort_keys=True))
    trainer = DiffusionTrainer(
      config=config,
      contract=contract,
      schedule=schedule,
      model=model,
      train_dataset=train_dataset,
      eval_datasets={
        "validation": validation_dataset,
        "test": test_dataset,
      },
      output_dir=output_dir,
      device=device,
      resume=args.resume,
      checkpoint_sidecar=artifact_metadata["checkpoint_sidecar"],
      allow_long_run=args.allow_long_run,
    )
    result = trainer.train()
    payload = {
      **preamble,
      "global_step": result.global_step,
      "epochs_completed": result.epochs_completed,
      "best_validation_loss": result.best_validation_loss,
      "best_evaluation_loss": result.best_validation_loss,
      "evaluation_split": config.eval_split,
      "final_train_loss": result.final_train_loss,
      "metrics_path": str(result.metrics_path),
      "checkpoint_paths": {
        name: str(path) for name, path in result.checkpoint_paths.items()
      },
      "run_manifest": str(output_dir / "run-manifest.yaml"),
      "checkpoint_sidecars": {
        name: str(path.with_suffix(".yaml"))
        for name, path in result.checkpoint_paths.items()
      },
      "note": "bounded engineering training; no model-quality claim",
    }
    if args.json is not None:
      _json_write(args.json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
  except (OSError, RuntimeError, ValueError, TypeError) as exc:
    print(f"train: {exc}", file=sys.stderr)
    return 1


def _command_evaluate_offline(args: argparse.Namespace) -> int:
  """Load a run checkpoint and score held-out windows without a simulator."""
  try:
    import torch

    from mjlab.tasks.tracking.diffusion.checkpoint import load_checkpoint
    from mjlab.tasks.tracking.diffusion.contract import DiffusionContract
    from mjlab.tasks.tracking.diffusion.evaluation import evaluate_generation
    from mjlab.tasks.tracking.diffusion.model import (
      DenoiserSettings,
      StateLatentTransformer,
    )
    from mjlab.tasks.tracking.diffusion.schedule import (
      DiffusionSchedule,
      build_inference_grid,
    )
    from mjlab.tasks.tracking.diffusion.trainer import ExponentialMovingAverage
    from mjlab.tasks.tracking.diffusion.training_config import (
      TrainingConfig,
      TrainingConfigError,
    )
    from mjlab.tasks.tracking.diffusion.window_dataset import (
      DatasetSource,
      _source_dataset_hash,
      load_window_records,
    )

    run_dir = Path(args.run)
    checkpoint_path = (
      Path(args.checkpoint)
      if args.checkpoint is not None
      else run_dir / "checkpoint-best.pt"
    )
    if not checkpoint_path.is_file():
      fallback = run_dir / "checkpoint-last.pt"
      raise FileNotFoundError(
        f"checkpoint not found: {checkpoint_path}; pass --checkpoint {fallback} explicitly"
      )
    contract = DiffusionContract.from_yaml(args.contract)
    schedule = DiffusionSchedule.from_contract(contract)
    settings = DenoiserSettings.from_contract(contract, training_k=schedule.training_k)
    model = StateLatentTransformer(settings)
    ema = ExponentialMovingAverage(model)
    device = torch.device(args.device)
    state = load_checkpoint(
      checkpoint_path,
      model=model,
      ema=ema,
      map_location=device,
      expected_contract_identity=contract.identity_hash(),
    )
    recorded_split = state.selection_split
    if recorded_split is None:
      recorded_split = getattr(state, "best_metric_split", None)
    is_last_checkpoint = checkpoint_path.name == "checkpoint-last.pt"
    if recorded_split == "test" and not is_last_checkpoint:
      field = (
        "selection_split" if state.selection_split is not None else "best_metric_split"
      )
      fallback = run_dir / "checkpoint-last.pt"
      raise ValueError(
        f"checkpoint {field}='test' is reserved for the final report; "
        f"pass --checkpoint {fallback} explicitly"
      )
    if state.contract_identity != contract.identity_hash():
      raise ValueError(
        "checkpoint contract identity differs from the active no-mixture "
        "representation; legacy projected checkpoints require explicit migration and retraining"
      )
    if state.schedule_identity != schedule.identity_hash():
      raise ValueError("checkpoint schedule identity mismatch")
    dataset_path = state.dataset_identity.get("directory")
    if not isinstance(dataset_path, str) or not dataset_path:
      raise ValueError("checkpoint does not record a dataset directory")
    source = DatasetSource.resolve(dataset_path, contract_path=args.contract)
    loaded = source.load()
    if Path(dataset_path).resolve() != source.directory.resolve():
      raise ValueError("checkpoint dataset directory could not be resolved")
    expected_identity = state.dataset_identity
    actual_projection_hashes = {
      "matrix": loaded.projection.matrix_sha256,
      "pseudoinverse": loaded.projection.pseudoinverse_sha256,
      "statistics": loaded.projection.statistics_sha256,
    }
    if expected_identity.get("source_dataset_hash") != _source_dataset_hash(
      source, loaded
    ):
      raise ValueError("checkpoint dataset source identity mismatch")
    if expected_identity.get("assignments_hash") != loaded.assignments.sha256():
      raise ValueError("checkpoint assignments identity mismatch")
    if expected_identity.get("split_coverage") != loaded.index.coverage():
      raise ValueError("checkpoint split coverage mismatch")
    if expected_identity.get("contract_hash") != contract.identity_hash():
      raise ValueError("checkpoint dataset contract identity mismatch")
    if expected_identity.get("projection_hashes") != actual_projection_hashes:
      raise ValueError("checkpoint projection identity mismatch")
    try:
      config = TrainingConfig.from_mapping(state.config)
    except TrainingConfigError:
      # Checkpoints written before F1 stored ``eval_split=test``.  That legacy
      # field is only used here for the evaluation seed; it never authorizes a
      # new trainer configuration or model-selection decision.
      if state.selection_split is not None or state.config.get("eval_split") != "test":
        raise
      config = TrainingConfig.from_mapping({**state.config, "eval_split": "validation"})
    motion_ids = _checkpoint_motion_ids(expected_identity)
    records = load_window_records(
      loaded,
      args.split,
      motion_ids=motion_ids,
      limit=args.max_windows,
    )
    if state.ema_state is not None:
      ema.copy_to(model)
    # Keep the CLI explicit about device placement; ``evaluate_generation``
    # also enforces it at the API boundary.
    model.to(device)
    diagnostics_path = (
      run_dir / f"generation-{args.sampler}-{args.split}.json"
      if args.start_level == 1000
      else run_dir
      / f"generation-{args.sampler}-start-{args.start_level}-{args.split}.json"
    )
    reverse_diagnostics_path = (
      run_dir / f"reverse-diagnostic-{args.split}.json"
      if args.reverse_diagnostic
      else None
    )
    pathwise_transition_path = (
      Path(args.pathwise_output)
      if args.pathwise_output is not None
      else run_dir / f"pathwise-transition-diagnostic-{args.split}.json"
      if args.pathwise_transition_diagnostic
      else None
    )
    report = evaluate_generation(
      model,
      schedule=schedule,
      grid=build_inference_grid(schedule, contract=contract),
      records=records,
      projection=loaded.projection,
      contract=contract,
      seed=config.eval_seed if args.seed is None else args.seed,
      device=device,
      max_windows=args.max_windows,
      diagnostics_path=diagnostics_path,
      split=args.split,
      sampler_kind=args.sampler,
      start_level=args.start_level,
      history_condition_dropout_probability=args.history_condition_dropout_probability,
      reverse_diagnostics_path=reverse_diagnostics_path,
      pathwise_transition_diagnostics_path=pathwise_transition_path,
    )
    payload = report.as_dict()
    payload["sampler"] = args.sampler
    payload["start_level"] = args.start_level
    payload["selection_split"] = state.selection_split
    payload["best_metric_split"] = getattr(state, "best_metric_split", None)
    payload["test_selection_provenance"] = recorded_split == "test"
    payload["pathwise_transition_diagnostic"] = (
      str(pathwise_transition_path) if pathwise_transition_path is not None else None
    )
    payload["checkpoint"] = str(checkpoint_path)
    payload["dataset"] = str(source.directory)
    payload["device"] = str(device)
    if device.type == "cuda":
      payload["device_name"] = torch.cuda.get_device_name(device)
    if report.windows == 0 and args.max_windows != 0:
      print(
        f"evaluate-offline: split {args.split!r}: no windows were selected",
        file=sys.stderr,
      )
      return 1
    if args.json is not None:
      _json_write(args.json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
  except (OSError, RuntimeError, ValueError, TypeError) as exc:
    print(f"evaluate-offline: {exc}", file=sys.stderr)
    return 1


def _command_migrate_no_mixtures(args: argparse.Namespace) -> int:
  """Explicitly rebuild a derived no-mixture dataset from a D0 raw store."""
  try:
    from mjlab.tasks.tracking.diffusion.migration import rebuild_no_mixture_dataset

    payload = rebuild_no_mixture_dataset(
      args.source_directory,
      args.destination_directory,
      source_contract=args.source_contract,
      contract_path=args.contract,
    )
    if args.json is not None:
      _json_write(args.json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
  except (OSError, RuntimeError, ValueError, TypeError) as exc:
    print(f"migrate-no-mixtures: {exc}", file=sys.stderr)
    return 1


def _command_audit(args: argparse.Namespace) -> int:
  """Run identity and token/provenance diagnostics against a D1 dataset."""
  try:
    from mjlab.tasks.tracking.diffusion.evaluation import audit_dataset
    from mjlab.tasks.tracking.diffusion.window_dataset import DatasetSource

    source = DatasetSource.resolve(args.dataset_dir, contract_path=args.contract)
    payload = audit_dataset(source)
    if args.json is not None:
      _json_write(args.json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("ok") is True else 1
  except (OSError, RuntimeError, ValueError, TypeError) as exc:
    payload = {"ok": False, "dataset": str(args.dataset_dir), "error": str(exc)}
    if args.json is not None:
      _json_write(args.json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="diffusion",
    description="D1 frozen-X2 diffusion preflight, collection and dataset tools.",
  )
  commands = parser.add_subparsers(dest="command", required=False)

  preflight = commands.add_parser(
    "preflight", help="check frozen contract and artifact bytes"
  )
  _common(preflight)
  preflight.set_defaults(handler=_command_preflight)

  inspect = commands.add_parser("inspect", help="inspect a bounded shard store")
  _common(inspect)
  inspect.add_argument("--store", type=Path, required=True)
  inspect.add_argument(
    "--jobs", type=int, default=8, help="bounded shard-validation workers (default: 8)"
  )
  inspect.set_defaults(handler=_command_inspect)

  replay = commands.add_parser("replay", help="run offline row arithmetic diagnostics")
  _common(replay)
  replay.add_argument("--store", type=Path, required=True)
  replay.add_argument(
    "--jobs", type=int, default=8, help="bounded replay workers (default: 8)"
  )
  replay.set_defaults(handler=_command_replay)

  build = commands.add_parser("build", help="index eligible windows and grouped splits")
  _common(build)
  build.add_argument("--store", type=Path, required=True)
  build.add_argument("--assignments", type=Path, required=True)
  build.add_argument("--report", type=Path, required=True)
  build.add_argument(
    "--jobs", type=int, default=8, help="bounded window-index workers (default: 8)"
  )
  build.set_defaults(handler=_command_build)

  stats = commands.add_parser("stats", help="fit train-only projection statistics")
  _common(stats)
  stats.add_argument("--store", type=Path, required=True)
  stats.add_argument("--assignments", type=Path, required=True)
  stats.add_argument("--bundle", type=Path, required=True)
  stats.add_argument("--report", type=Path, required=True)
  stats.add_argument(
    "--jobs", type=int, default=8, help="bounded shard-range workers (default: 8)"
  )
  stats.add_argument(
    "--batch-windows",
    type=int,
    default=512,
    help="windows per statistics update (default: 512)",
  )
  stats.add_argument(
    "--index",
    type=Path,
    help=(
      "reuse this persisted window index instead of rebuilding it; the loader "
      "refuses an index that does not describe the given store and assignments"
    ),
  )
  stats.set_defaults(handler=_command_stats)

  collect = commands.add_parser(
    "collect", help="run an explicitly authorized runtime factory"
  )
  _common(collect)
  collect.add_argument(
    "--execute", action="store_true", help="required operational opt-in"
  )
  collect.add_argument("--runtime-factory", type=str)
  collect.add_argument("--config", type=Path)
  collect.add_argument("--output", type=Path, required=True)
  collect.add_argument("--report", type=Path, required=True)
  collect.add_argument(
    "--seeds",
    type=int,
    default=None,
    help=(
      "seed count for a bulk schedule; overrides MJLAB_D1_BULK_SEEDS for this "
      "run.  The seeds are 0..N-1 and are shared by both initial poses."
    ),
  )
  collect.add_argument(
    "--initial-poses",
    choices=("reference", "standing", "both"),
    default=None,
    help=(
      "initial poses to roll out: both (the plan's schedule), or one pose for a "
      "bounded smoke run; overrides MJLAB_D1_INITIAL_POSES for this run"
    ),
  )
  collect.add_argument("--max-trials", type=int, default=1)
  collect.add_argument("--max-envs", type=int, default=256)
  collect.add_argument("--max-gpus", type=int, default=1)
  collect.add_argument("--max-wall-minutes", type=int, default=1)
  collect.add_argument("--monitor-inactivity-seconds", type=int, default=120)
  collect.add_argument("--max-output-mib", type=int, default=512)
  collect.add_argument(
    "--stream-rows",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="bounded streaming is on for batched collection; --no-stream-rows opts out",
  )
  collect.add_argument(
    "--compact-results",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="compact batched counts are on with streaming; --no-compact-results opts out",
  )
  collect.add_argument(
    "--max-rows-per-append",
    type=int,
    default=None,
    help="bound one store append in rows; requires --stream-rows",
  )
  collect.add_argument(
    "--tensor-dispatch",
    action=argparse.BooleanOptionalAction,
    default=None,
    help=(
      "device-native dispatch is on for compatible batched runtimes; "
      "--no-tensor-dispatch uses NumPy"
    ),
  )
  collect.add_argument(
    "--policy-device",
    type=str,
    default=None,
    help=(
      "VAE device: cpu, simulation, or cuda[:N]; default follows simulation "
      "for batched X2 collection and stays cpu for scalar collection"
    ),
  )
  collect.add_argument(
    "--progress",
    type=Path,
    help="append one JSON progress record per event to this file",
  )
  collect.set_defaults(handler=_command_collect)

  train = commands.add_parser("train", help="run explicitly bounded D2 training")
  train.add_argument("--dataset-dir", type=Path, required=True)
  train.add_argument("--config", type=Path, required=True)
  train.add_argument("--out", type=Path, required=True)
  train.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
  train.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
  budget = train.add_mutually_exclusive_group()
  budget.add_argument("--max-updates", type=int)
  budget.add_argument("--epochs", type=int)
  train.add_argument("--motion-id", action="append", default=[])
  train.add_argument("--resume", type=Path)
  train.add_argument(
    "--cache-tokens",
    action=argparse.BooleanOptionalAction,
    default=True,
  )
  train.add_argument(
    "--max-cache-gib",
    type=float,
    default=None,
    help=(
      "ceiling for the float32 token cache, in GiB.  The default is the "
      "module constant window_dataset._DEFAULT_MAX_CACHE_GIB.  A cache larger "
      "than the ceiling is refused before it is built."
    ),
  )
  train.add_argument("--allow-long-run", action="store_true")
  train.add_argument(
    "--prepare-token-cache",
    action="store_true",
    help=(
      "build missing token caches and provenance before training; this is the "
      "default behaviour, kept for explicit launch records"
    ),
  )
  train.add_argument(
    "--require-prepared-cache",
    action="store_true",
    help=(
      "fail instead of building a missing token cache; use when the dataset "
      "caches were prepared in advance"
    ),
  )
  train.add_argument(
    "--cache-jobs",
    type=int,
    default=1,
    help="worker processes for explicit token-cache preparation",
  )
  train.add_argument(
    "--cache-lock-timeout",
    type=float,
    default=1800.0,
    help=(
      "bounded writer-lock wait when several runs prepare the same cache; the "
      "losers reuse the published cache instead of failing"
    ),
  )
  train.add_argument("--json", type=Path)
  train.set_defaults(handler=_command_train)

  index = commands.add_parser(
    "build-index", help="build and persist the dataset window index"
  )
  index.add_argument("--dataset-dir", type=Path, required=True)
  index.add_argument("--contract", type=Path, required=True)
  index.add_argument(
    "--jobs", type=int, default=8, help="bounded index workers (default: 8)"
  )
  index.add_argument("--json", type=Path)
  index.set_defaults(handler=_command_build_index)

  token_cache = commands.add_parser(
    "build-token-cache",
    help="build or validate the dataset-owned token cache",
  )
  token_cache.add_argument("--dataset-dir", type=Path, required=True)
  token_cache.add_argument("--contract", type=Path, required=True)
  token_cache.add_argument(
    "--splits",
    default="train,validation,test",
    help="comma-separated split names",
  )
  token_cache.add_argument(
    "--motion",
    action="append",
    default=None,
    help="limit the cache to these motion ids; repeatable",
  )
  token_cache.add_argument(
    "--max-cache-gib",
    type=float,
    default=None,
    help="ceiling for a single cache; the module constant applies by default",
  )
  token_cache.add_argument(
    "--jobs",
    type=int,
    default=1,
    help="worker processes; 1 builds the cache in this process",
  )
  token_cache.add_argument(
    "--recover-lock",
    action="store_true",
    help=(
      "remove a stale writer lock for the requested splits before starting; "
      "refuses while the recorded owner process is alive"
    ),
  )
  token_cache.add_argument(
    "--lock-timeout",
    type=float,
    default=30.0,
    help="bounded writer-lock wait in seconds",
  )
  token_cache.add_argument(
    "--validate-only",
    action="store_true",
    help="validate cache and provenance only; never create locks or write files",
  )
  token_cache.add_argument("--json", type=Path)
  token_cache.set_defaults(handler=_command_build_token_cache)

  evaluate = commands.add_parser(
    "evaluate-offline", help="score a checkpoint without constructing a simulator"
  )
  evaluate.add_argument("--run", type=Path, required=True)
  evaluate.add_argument(
    "--split", choices=("train", "validation", "test"), default="test"
  )
  evaluate.add_argument("--checkpoint", type=Path)
  evaluate.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
  evaluate.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
  evaluate.add_argument("--max-windows", type=int)
  evaluate.add_argument(
    "--sampler",
    choices=("ddim", "ddpm"),
    default="ddim",
    help="reverse sampler; ddpm runs all 1000 adjacent training levels",
  )
  evaluate.add_argument(
    "--start-level",
    type=int,
    choices=tuple(range(50, 1001, 50)),
    default=1000,
    help="DDIM initialization level; 950 skips the 1000-level call",
  )
  evaluate.add_argument(
    "--reverse-diagnostic",
    action="store_true",
    help="record correctly-noised versus evolving x0 errors per DDIM level",
  )
  evaluate.add_argument(
    "--pathwise-transition-diagnostic",
    action="store_true",
    help="record matched-epsilon DDIM transition errors by feature group",
  )
  evaluate.add_argument(
    "--pathwise-output",
    type=Path,
    help="output path for the opt-in pathwise transition report",
  )
  evaluate.add_argument(
    "--history-condition-dropout-probability",
    type=float,
    default=0.0,
    help=(
      "hide both state and latent history rows with this probability during "
      "offline generation stress evaluation"
    ),
  )
  evaluate.add_argument("--seed", type=int)
  evaluate.add_argument("--json", type=Path)
  evaluate.set_defaults(handler=_command_evaluate_offline)

  migrate = commands.add_parser(
    "migrate-no-mixtures",
    help="explicitly rebuild a no-mixture dataset from a legacy D0 raw store",
  )
  migrate.add_argument("--source-directory", type=Path, required=True)
  migrate.add_argument("--destination-directory", type=Path, required=True)
  migrate.add_argument(
    "--source-contract",
    type=Path,
    default=Path("docs/plans/beyondmimic_diffusion_d0_contract.yaml"),
  )
  migrate.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
  migrate.add_argument("--json", type=Path)
  migrate.set_defaults(handler=_command_migrate_no_mixtures)

  audit = commands.add_parser("audit", help="audit a D1 dataset offline")
  audit.add_argument("--dataset-dir", type=Path, required=True)
  audit.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
  audit.add_argument("--json", type=Path)
  audit.set_defaults(handler=_command_audit)
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  """Run the CLI and return a process exit code."""
  parser = build_parser()
  args = parser.parse_args(argv)
  if args.command is None:
    parser.print_help()
    return 0
  return int(args.handler(args))


if __name__ == "__main__":
  raise SystemExit(main())
