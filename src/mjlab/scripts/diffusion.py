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
import hashlib
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence, cast

from mjlab.tasks.tracking.diffusion.runtime import CollectionRequest, RuntimeBundle

DEFAULT_CONTRACT_PATH = Path("docs/plans/beyondmimic_diffusion_d0_contract.yaml")
DEFAULT_CONFIG_PATH = Path("configs/diffusion/x2_50hz_collection.yaml")
# Hard collection ceilings the CLI will not exceed.  The *pilot* budget (54
# trials / 13500 transitions / 15 min) is the frozen D0 contract and
# configs/diffusion/x2_50hz_collection.yaml; these are the outer limits, raised
# for the post-pilot bulk collection after the pilot's budgets were reviewed.
MAX_COLLECTION_TRIALS = 5_000
MAX_COLLECTION_TRANSITIONS = 600_000


@dataclass(frozen=True)
class PreflightReport:
  """Serializable result of contract and pinned-artifact validation."""

  ok: bool
  contract_sha256: str | None
  artifacts: dict[str, dict[str, Any]]
  errors: tuple[str, ...]

  def as_dict(self) -> dict[str, Any]:
    return {
      "ok": self.ok,
      "contract_sha256": self.contract_sha256,
      "artifacts": self.artifacts,
      "errors": list(self.errors),
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
  try:
    contract = _load_contract(contract_file)
    contract_sha = contract.contract_sha256
  except Exception as exc:  # report a CLI failure, never hide a bad manifest
    return PreflightReport(False, None, {}, (f"contract: {exc}",))

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
  return PreflightReport(not errors, contract_sha, artifacts, tuple(errors))


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
  """Require a complete clean phase before matching OU partners."""
  clean: dict[str, Any] = {}
  seen_ou: set[str] = set()
  saw_ou = False
  for spec in trials:
    phase = getattr(spec, "phase", None)
    pair_id = getattr(spec, "pair_id", "")
    initial_state_id = getattr(spec, "initial_state_id", "")
    if phase == "clean":
      if saw_ou:
        return "the complete clean phase must precede every OU trial"
      if not pair_id or not initial_state_id or pair_id in clean:
        return "clean trials require unique pair and initial-state identities"
      clean[pair_id] = spec
    elif phase == "ou":
      saw_ou = True
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
  if seen_ou and set(clean) != seen_ou:
    return "complete clean/OU pair matrix is required before collection"
  return None


def _validate_runtime_bundle(
  bundle: RuntimeBundle, *, max_envs: int | None = None, max_gpus: int | None = None
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

  from mjlab.tasks.tracking.diffusion.adapter import (
    FrozenPpoRecoveryPolicy,
    FrozenVaePolicy,
  )
  from mjlab.tasks.tracking.diffusion.collector import DiffusionCollector

  collector = bundle.collector
  if not isinstance(collector, DiffusionCollector):
    return "runtime collector must be the concrete DiffusionCollector type"
  if not isinstance(collector.vae, FrozenVaePolicy):
    return "runtime VAE binding is not a FrozenVaePolicy"
  if not isinstance(collector.ppo, FrozenPpoRecoveryPolicy):
    return "runtime PPO binding is not a FrozenPpoRecoveryPolicy"
  if collector.vae.artifact_path is None or collector.vae.artifact_sha256 is None:
    return "runtime VAE binding has no frozen artifact identity"
  if collector.ppo.artifact_path is None or collector.ppo.artifact_sha256 is None:
    return "runtime PPO binding has no frozen artifact identity"
  from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT

  for name, policy, expected_hash in (
    ("VAE", collector.vae, DEFAULT_CONTRACT.vae_sha256),
    ("PPO", collector.ppo, DEFAULT_CONTRACT.recovery_sha256),
  ):
    path = policy.artifact_path
    if not isinstance(path, Path) or not path.is_file():
      return f"runtime {name} binding artifact path is not a file"
    actual_hash = _sha256(path)
    if actual_hash != policy.artifact_sha256 or actual_hash != expected_hash:
      return f"runtime {name} binding artifact bytes are not the pinned D0 asset"
    metadata = policy.artifact_metadata
    try:
      policy.action_contract.verify_metadata(metadata, require_actuator_metadata=True)
    except Exception as exc:
      return f"runtime {name} action metadata binding is invalid: {exc}"
  if (
    collector.vae_actions.joint_order != collector.vae.action_contract.joint_order
    or collector.ppo_actions.joint_order != collector.ppo.action_contract.joint_order
  ):
    return "runtime collector action bindings disagree with policy joint order"
  import numpy as np

  for collector_contract, policy_contract in (
    (collector.vae_actions, collector.vae.action_contract),
    (collector.ppo_actions, collector.ppo.action_contract),
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
  if isinstance(pilot, dict) and pilot.get("max_envs") != 1:
    raise ValueError(
      "collection config pilot.max_envs must be 1 until environment identity is carried"
    )
  return payload


def _validate_resource_budgets(args: argparse.Namespace) -> str | None:
  limits = (
    ("max_trials", args.max_trials, 1, MAX_COLLECTION_TRIALS),
    (
      "max_control_transitions",
      args.max_control_transitions,
      1,
      MAX_COLLECTION_TRANSITIONS,
    ),
    ("max_envs", args.max_envs, 1, 8),
    ("max_gpus", args.max_gpus, 1, 1),
    ("max_wall_minutes", args.max_wall_minutes, 1, 720),
    ("monitor_inactivity_seconds", args.monitor_inactivity_seconds, 1, 120),
    ("max_output_mib", args.max_output_mib, 1, 4096),
  )
  for name, value, lower, upper in limits:
    if value < lower or value > upper:
      return f"{name} must be in [{lower}, {upper}]"
  if args.max_envs != 1:
    return (
      "max_envs must be 1 until collector environment identity is carried end-to-end"
    )
  return None


def _command_inspect(args: argparse.Namespace) -> int:
  from mjlab.tasks.tracking.diffusion.storage import AppendOnlyShardStore

  contract = _load_contract(args.contract)
  store = AppendOnlyShardStore(args.store, contract=contract)
  rows = tuple(store.iter_rows())
  controllers = {
    name: sum(row.controller == name for row in rows) for name in ("vae", "ppo")
  }
  payload = {
    "contract_sha256": contract.contract_sha256,
    "store": str(args.store),
    "shards": store.shard_count,
    "rows": store.row_count,
    "controllers": controllers,
    "unique_identities": len({row.identity for row in rows}) == len(rows),
  }
  print(json.dumps(payload, indent=2, sort_keys=True))
  return 0


def _command_replay(args: argparse.Namespace) -> int:
  import numpy as np

  from mjlab.tasks.tracking.diffusion.storage import AppendOnlyShardStore

  contract = _load_contract(args.contract)
  store = AppendOnlyShardStore(args.store, contract=contract)
  rows = tuple(store.iter_rows())
  errors: list[str] = []
  for row in rows:
    if not np.allclose(
      row.clean_action + row.ou_noise,
      row.executed_action,
      atol=1.0e-5,
      rtol=1.0e-5,
    ):
      errors.append(f"{row.identity}: executed action arithmetic mismatch")
    if row.controller == "vae" and row.latent is None:
      errors.append(f"{row.identity}: VAE row has no latent")
    if row.controller == "ppo" and row.latent is not None:
      errors.append(f"{row.identity}: PPO row has a latent")
  payload = {
    "ok": not errors,
    "rows": len(rows),
    "errors": errors,
    "note": "offline shard replay diagnostics; no simulator was launched",
  }
  print(json.dumps(payload, indent=2, sort_keys=True))
  return 0 if not errors else 1


def _load_assignments(path: Path) -> Any:
  from mjlab.tasks.tracking.diffusion.dataset import SplitAssignments

  return SplitAssignments.from_dict(json.loads(path.read_text()))


def _command_build(args: argparse.Namespace) -> int:
  from mjlab.tasks.tracking.diffusion.dataset import build_dataset
  from mjlab.tasks.tracking.diffusion.storage import AppendOnlyShardStore

  contract = _load_contract(args.contract)
  store = AppendOnlyShardStore(args.store, contract=contract)
  index, assignments = build_dataset(store, contract=contract)
  _json_write(args.assignments, assignments.as_dict())
  payload = {
    "contract_sha256": contract.contract_sha256,
    "assignments": str(args.assignments),
    "groups": assignments.coverage(),
    "windows": index.coverage(),
    "total_windows": len(index),
  }
  _json_write(args.report, payload)
  print(json.dumps(payload, indent=2, sort_keys=True))
  return 0


def _command_stats(args: argparse.Namespace) -> int:
  from mjlab.tasks.tracking.diffusion.dataset import (
    WindowIndex,
    fit_training_statistics,
  )
  from mjlab.tasks.tracking.diffusion.storage import AppendOnlyShardStore

  contract = _load_contract(args.contract)
  store = AppendOnlyShardStore(args.store, contract=contract)
  assignments = _load_assignments(args.assignments)
  index = WindowIndex.build(store, assignments, contract=contract)
  bundle = fit_training_statistics(index, contract=contract)
  bundle.save(args.bundle)
  payload = {
    "contract_sha256": contract.contract_sha256,
    "bundle": str(args.bundle),
    "windows": index.coverage(),
    "train_windows_used": len(index.refs_for("train")),
    "matrix_sha256": bundle.matrix_sha256,
    "statistics_sha256": bundle.statistics_sha256,
  }
  _json_write(args.report, payload)
  print(json.dumps(payload, indent=2, sort_keys=True))
  return 0


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
  if args.config is not None:
    try:
      _load_collection_config(args.config)
    except (OSError, RuntimeError, ValueError) as exc:
      print(f"collection config: {exc}", file=sys.stderr)
      return 2
  report = run_preflight(args.contract, artifact_root=args.artifact_root)
  if not report.ok:
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True), file=sys.stderr)
    return 1

  from mjlab.tasks.tracking.diffusion.storage import AppendOnlyShardStore

  contract = _load_contract(args.contract)
  store = AppendOnlyShardStore(
    args.output, contract=contract, max_total_rows=args.max_control_transitions
  )
  trial_specs: tuple[Any, ...] = ()
  request = CollectionRequest(
    contract,
    store,
    args.output,
    trial_specs,
    args.max_trials,
    args.max_control_transitions,
    args.max_envs,
    args.max_gpus,
    args.max_wall_minutes,
    args.monitor_inactivity_seconds,
    args.max_output_mib,
  )
  started = time.monotonic()
  bundle = _load_factory(args.runtime_factory)(request)
  if not isinstance(bundle, RuntimeBundle):
    raise TypeError("runtime factory must return RuntimeBundle")
  runtime_error = _validate_runtime_bundle(
    bundle, max_envs=args.max_envs, max_gpus=args.max_gpus
  )
  if runtime_error is not None:
    print(runtime_error, file=sys.stderr)
    return 2
  if len(bundle.trials) == 0:
    raise ValueError("runtime factory returned no trial specifications")
  if len(bundle.trials) > args.max_trials:
    raise ValueError("runtime factory returned more trials than the requested budget")
  pair_error = _validate_trial_pairs(bundle.trials)
  if pair_error is not None:
    print(pair_error, file=sys.stderr)
    return 2
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
            else {
              "qualified": clean_result.qualification.qualified,
              "label": clean_result.qualification.label,
              "reason": clean_result.qualification.reason,
              "handoff_step": clean_result.qualification.handoff_step,
            },
          }
        )
        continue
      ou_attempted += 1
    remaining = args.max_control_transitions - transitions
    trial_started = time.monotonic()
    result = bundle.collector.collect(spec, remaining_transitions=remaining)
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
    if transitions > args.max_control_transitions:
      raise RuntimeError("control-transition budget exceeded by runtime factory")
    results.append(
      {
        "run_id": spec.run_id,
        "motion_id": spec.motion_id,
        "phase": spec.phase,
        "qualified": result.qualification.qualified,
        "label": result.qualification.label,
        "reason": result.qualification.reason,
        "rows": len(result.rows),
        "accepted_rows": len(result.accepted_rows),
        "verification_rows": len(result.verification_rows),
        "stopped_reason": result.stopped_reason,
        "store_files": list(result.store_files),
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

    from mjlab.tasks.tracking.diffusion.checkpoint import CheckpointError
    from mjlab.tasks.tracking.diffusion.contract import DiffusionContract
    from mjlab.tasks.tracking.diffusion.model import (
      DenoiserSettings,
      StateLatentTransformer,
    )
    from mjlab.tasks.tracking.diffusion.schedule import DiffusionSchedule
    from mjlab.tasks.tracking.diffusion.trainer import DiffusionTrainer
    from mjlab.tasks.tracking.diffusion.training_config import (
      TrainingConfig,
      TrainingConfigError,
    )
    from mjlab.tasks.tracking.diffusion.window_dataset import (
      DatasetSource,
      TokenWindowDataset,
      _projection_identity,
      _source_dataset_hash,
      build_token_cache,
      iter_window_provenance,
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

    contract = DiffusionContract.from_yaml(args.contract)
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
        if selected_motions:
          suffix = hashlib.sha256(
            "\\0".join(sorted(selected_motions)).encode("utf-8")
          ).hexdigest()[:12]
        else:
          suffix = "all"
        path = output_dir / f"tokens-{split}-{suffix}.npy"
        cache_path, cache_count = build_token_cache(
          source,
          split,
          path,
          motion_ids=selected_motions,
        )
        provenance = list(
          iter_window_provenance(
            loaded,
            split,
            motion_ids=selected_motions,
          )
        )
        if len(provenance) != cache_count:
          raise RuntimeError(
            f"token cache count changed while collecting {split} provenance"
          )
        dataset = TokenWindowDataset(
          cache_path,
          provenance,
          source=source,
          split=split,
          motion_ids=selected_motions,
        )
        return dataset, provenance, cache_path

      records = load_window_records(
        loaded,
        split,
        motion_ids=selected_motions,
      )
      if records:
        values = torch.from_numpy(np.stack([record.tokens for record in records]))
      else:
        values = torch.empty((0, 41, 231), dtype=torch.float32)
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
      "note": "bounded engineering training; no model-quality claim",
    }
    if args.json is not None:
      _json_write(args.json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
  except (CheckpointError, OSError, RuntimeError, ValueError, TypeError) as exc:
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
      raise ValueError("checkpoint contract identity mismatch")
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
    diagnostics_path = run_dir / f"generation-{args.split}.json"
    report = evaluate_generation(
      model,
      schedule=schedule,
      grid=build_inference_grid(schedule, contract=contract),
      records=records,
      projection=loaded.projection,
      contract=contract,
      seed=config.eval_seed,
      device=device,
      max_windows=args.max_windows,
      diagnostics_path=diagnostics_path,
      split=args.split,
    )
    payload = report.as_dict()
    payload["selection_split"] = state.selection_split
    payload["best_metric_split"] = getattr(state, "best_metric_split", None)
    payload["test_selection_provenance"] = recorded_split == "test"
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
  inspect.set_defaults(handler=_command_inspect)

  replay = commands.add_parser("replay", help="run offline row arithmetic diagnostics")
  _common(replay)
  replay.add_argument("--store", type=Path, required=True)
  replay.set_defaults(handler=_command_replay)

  build = commands.add_parser("build", help="index eligible windows and grouped splits")
  _common(build)
  build.add_argument("--store", type=Path, required=True)
  build.add_argument("--assignments", type=Path, required=True)
  build.add_argument("--report", type=Path, required=True)
  build.set_defaults(handler=_command_build)

  stats = commands.add_parser("stats", help="fit train-only projection statistics")
  _common(stats)
  stats.add_argument("--store", type=Path, required=True)
  stats.add_argument("--assignments", type=Path, required=True)
  stats.add_argument("--bundle", type=Path, required=True)
  stats.add_argument("--report", type=Path, required=True)
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
  collect.add_argument("--max-trials", type=int, default=1)
  collect.add_argument("--max-control-transitions", type=int, default=250)
  collect.add_argument("--max-envs", type=int, default=1)
  collect.add_argument("--max-gpus", type=int, default=1)
  collect.add_argument("--max-wall-minutes", type=int, default=1)
  collect.add_argument("--monitor-inactivity-seconds", type=int, default=120)
  collect.add_argument("--max-output-mib", type=int, default=512)
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
  train.add_argument("--allow-long-run", action="store_true")
  train.add_argument("--json", type=Path)
  train.set_defaults(handler=_command_train)

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
  evaluate.add_argument("--json", type=Path)
  evaluate.set_defaults(handler=_command_evaluate_offline)

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
