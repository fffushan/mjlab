"""Atomic diffusion checkpoints with strict training identities."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


class CheckpointError(ValueError):
  """A checkpoint is malformed or incompatible with the requested run."""


_FORMAT = "mjlab-x2-diffusion-checkpoint-v1"


@dataclass(frozen=True, slots=True)
class CheckpointState:
  """Restored checkpoint metadata and state dictionaries."""

  model_state: Mapping[str, Any]
  ema_state: Mapping[str, Any] | None
  optimizer_state: Mapping[str, Any] | None
  scaler_state: Mapping[str, Any] | None
  config: Mapping[str, Any]
  config_hash: str
  contract_identity: str
  schedule_identity: str
  projection_hashes: Mapping[str, str]
  dataset_identity: Mapping[str, Any]
  global_step: int
  epoch: int
  best_validation_loss: float | None
  best_metric_split: str | None
  selection_split: str | None
  rng_state: Any
  fp32_checked: bool
  torch_version: str
  cuda_version: str | None


def _config_payload(config: Any) -> tuple[dict[str, Any], str]:
  if hasattr(config, "as_dict") and callable(config.as_dict):
    values = dict(config.as_dict())
  elif isinstance(config, Mapping):
    values = dict(config)
  else:
    raise CheckpointError("config must provide as_dict() or be a mapping")
  if hasattr(config, "sha256") and callable(config.sha256):
    digest = str(config.sha256())
  else:
    digest = hashlib.sha256(
      json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
  return values, digest


def _identity(value: Any, name: str) -> str:
  method = getattr(value, "identity_hash", None)
  if not callable(method):
    method = getattr(value, "sha256", None)
  if not callable(method):
    raise CheckpointError(f"{name} must provide identity_hash() or sha256()")
  digest = str(method())
  if len(digest) != 64 or any(
    character not in "0123456789abcdef" for character in digest
  ):
    raise CheckpointError(f"{name} identity is not a lowercase SHA256 digest")
  return digest


def _state_dict(value: Any, name: str) -> Mapping[str, Any] | None:
  if value is None:
    return None
  method = getattr(value, "state_dict", None)
  if not callable(method):
    raise CheckpointError(f"{name} must provide state_dict()")
  result = method()
  if not isinstance(result, Mapping):
    raise CheckpointError(f"{name}.state_dict() must return a mapping")
  return result


def _jsonable_identity(value: Mapping[str, Any]) -> dict[str, Any]:
  try:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    result = json.loads(encoded)
  except (TypeError, ValueError) as exc:
    raise CheckpointError("dataset identity must be JSON serialisable") from exc
  if not isinstance(result, dict):  # pragma: no cover - json object guard
    raise CheckpointError("dataset identity must be a mapping")
  return result


_SELECTION_SPLITS = frozenset(("train", "validation", "test"))


def _validate_split_field(value: object, name: str) -> None:
  """Reject selection-split provenance that is not a known split name."""
  if value is None:
    return
  if not isinstance(value, str):
    raise CheckpointError(f"{name} must be a string or None")
  if value not in _SELECTION_SPLITS:
    raise CheckpointError(
      f"{name} must be one of train, validation or test, got {value!r}"
    )


def save_checkpoint(
  path: str | Path,
  *,
  model: Any,
  ema: Any,
  optimizer: Any,
  scaler: Any,
  config: Any,
  contract: Any,
  schedule: Any,
  dataset_identity: Mapping[str, Any],
  global_step: int,
  epoch: int,
  best_validation_loss: float | None,
  rng_state: Any,
  best_metric_split: str | None = None,
  selection_split: str | None = None,
  fp32_checked: bool = False,
) -> None:
  """Atomically write model, optimizer, EMA, RNG and identity state."""
  if global_step < 0 or epoch < 0:
    raise CheckpointError("global_step and epoch must be non-negative")
  best_value: float | None = None
  if best_validation_loss is not None:
    try:
      best_value = float(best_validation_loss)
    except (TypeError, ValueError) as exc:
      raise CheckpointError("best_validation_loss must be finite or None") from exc
    if not torch.isfinite(torch.tensor(best_value)):
      raise CheckpointError("best_validation_loss must be finite or None")
  if not isinstance(fp32_checked, bool):
    raise CheckpointError("fp32_checked must be a boolean")
  _validate_split_field(best_metric_split, "best_metric_split")
  if selection_split is None:
    selection_split = best_metric_split
  else:
    _validate_split_field(selection_split, "selection_split")
    if best_metric_split is not None and selection_split != best_metric_split:
      raise CheckpointError(
        "selection_split contradicts best_metric_split; a checkpoint may not "
        "record two different selection splits"
      )
  config_values, config_hash = _config_payload(config)
  contract_identity = _identity(contract, "contract")
  schedule_identity = _identity(schedule, "schedule")
  identity_values = _jsonable_identity(dataset_identity)
  projection_hashes = identity_values.get("projection_hashes", {})
  if not isinstance(projection_hashes, Mapping):
    projection_hashes = {}
  projection_hashes = {str(key): str(value) for key, value in projection_hashes.items()}
  model_state = _state_dict(model, "model")
  if model_state is None:
    raise CheckpointError("model cannot be None")
  optimizer_state = _state_dict(optimizer, "optimizer")
  scaler_state = _state_dict(scaler, "scaler")
  ema_state = _state_dict(ema, "ema")
  payload: dict[str, Any] = {
    "format": _FORMAT,
    "model": model_state,
    "ema": ema_state,
    "optimizer": optimizer_state,
    "scaler": scaler_state,
    "config": config_values,
    "config_hash": config_hash,
    "contract_identity": contract_identity,
    "schedule_identity": schedule_identity,
    "projection_hashes": projection_hashes,
    "dataset_identity": identity_values,
    "global_step": int(global_step),
    "epoch": int(epoch),
    "best_validation_loss": best_value,
    "best_metric_split": best_metric_split,
    "selection_split": selection_split,
    "rng_state": rng_state,
    "fp32_checked": fp32_checked,
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
  }
  destination = Path(path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  temporary: Path | None = None
  try:
    descriptor, temporary_name = tempfile.mkstemp(
      dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    with os.fdopen(descriptor, "wb") as handle:
      torch.save(payload, handle)
      handle.flush()
      os.fsync(handle.fileno())
    os.replace(temporary, destination)
    temporary = None
  except (OSError, RuntimeError) as exc:
    raise CheckpointError(
      f"could not atomically write checkpoint {destination}"
    ) from exc
  finally:
    if temporary is not None:
      temporary.unlink(missing_ok=True)


def _require_mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
  value = payload.get(name)
  if not isinstance(value, Mapping):
    raise CheckpointError(f"checkpoint field {name!r} must be a mapping")
  return value


def load_checkpoint(
  path: str | Path,
  *,
  model: Any,
  ema: Any | None = None,
  optimizer: Any | None = None,
  scaler: Any | None = None,
  map_location: str | torch.device = "cpu",
) -> CheckpointState:
  """Load a checkpoint and optionally restore supplied mutable components."""
  source = Path(path)
  try:
    payload = torch.load(source, map_location=map_location, weights_only=False)
  except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError) as exc:
    raise CheckpointError(f"could not read checkpoint {source}") from exc
  if not isinstance(payload, Mapping) or payload.get("format") != _FORMAT:
    raise CheckpointError("unsupported or malformed diffusion checkpoint")
  required = {
    "model",
    "config",
    "config_hash",
    "contract_identity",
    "schedule_identity",
    "projection_hashes",
    "dataset_identity",
    "global_step",
    "epoch",
    "best_validation_loss",
    "best_metric_split",
    "rng_state",
    "fp32_checked",
    "torch_version",
    "cuda_version",
  }
  if not required.issubset(payload):
    raise CheckpointError("checkpoint is missing required identity/state fields")
  model_state = _require_mapping(payload, "model")
  config = _require_mapping(payload, "config")
  projection_hashes_raw = _require_mapping(payload, "projection_hashes")
  dataset_identity = _require_mapping(payload, "dataset_identity")
  config_hash = str(payload["config_hash"])
  calculated_config_hash = hashlib.sha256(
    json.dumps(dict(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
  ).hexdigest()
  if config_hash != calculated_config_hash:
    raise CheckpointError("checkpoint config hash mismatch")
  try:
    global_step = int(payload["global_step"])
    epoch = int(payload["epoch"])
  except (TypeError, ValueError) as exc:
    raise CheckpointError("checkpoint step and epoch are invalid") from exc
  if global_step < 0 or epoch < 0:
    raise CheckpointError("checkpoint step and epoch must be non-negative")

  ema_state = payload.get("ema")
  if ema is not None and not isinstance(ema_state, Mapping):
    raise CheckpointError(
      "checkpoint EMA state is required when an EMA object is supplied"
    )
  if ema_state is not None and not isinstance(ema_state, Mapping):
    raise CheckpointError("checkpoint EMA state must be a mapping or None")
  optimizer_state = payload.get("optimizer")
  if optimizer is not None and not isinstance(optimizer_state, Mapping):
    raise CheckpointError(
      "checkpoint optimizer state is required when an optimizer is supplied"
    )
  if optimizer_state is not None and not isinstance(optimizer_state, Mapping):
    raise CheckpointError("checkpoint optimizer state must be a mapping or None")
  scaler_state = payload.get("scaler")
  if scaler_state is not None and not isinstance(scaler_state, Mapping):
    raise CheckpointError("checkpoint scaler state must be a mapping or None")
  best = payload["best_validation_loss"]
  if best is not None:
    try:
      best = float(best)
    except (TypeError, ValueError) as exc:
      raise CheckpointError("checkpoint best validation loss is invalid") from exc
    if not torch.isfinite(torch.tensor(best)):
      raise CheckpointError("checkpoint best validation loss is non-finite")

  best_metric_split = payload["best_metric_split"]
  _validate_split_field(best_metric_split, "checkpoint best_metric_split")
  selection_split = payload.get("selection_split")
  if selection_split is not None:
    _validate_split_field(selection_split, "checkpoint selection_split")
    if best_metric_split is not None and selection_split != best_metric_split:
      raise CheckpointError(
        "checkpoint selection split contradicts its best metric split"
      )
  fp32_checked = payload["fp32_checked"]
  if not isinstance(fp32_checked, bool):
    raise CheckpointError("checkpoint fp32_checked state is invalid")

  load_model = getattr(model, "load_state_dict", None)
  if not callable(load_model):
    raise CheckpointError("model must provide load_state_dict()")
  try:
    load_model(model_state)
    if ema is not None and ema_state is not None:
      ema.load_state_dict(ema_state)
    if optimizer is not None and optimizer_state is not None:
      optimizer.load_state_dict(optimizer_state)
    if scaler is not None and scaler_state is not None:
      scaler.load_state_dict(scaler_state)
  except (RuntimeError, ValueError, TypeError, KeyError) as exc:
    raise CheckpointError(
      f"checkpoint state does not fit supplied objects: {source}"
    ) from exc

  return CheckpointState(
    model_state=model_state,
    ema_state=ema_state,
    optimizer_state=optimizer_state,
    scaler_state=scaler_state,
    config=config,
    config_hash=config_hash,
    contract_identity=str(payload["contract_identity"]),
    schedule_identity=str(payload["schedule_identity"]),
    projection_hashes={str(k): str(v) for k, v in projection_hashes_raw.items()},
    dataset_identity=dict(dataset_identity),
    global_step=global_step,
    epoch=epoch,
    best_validation_loss=best,
    best_metric_split=best_metric_split,
    selection_split=selection_split,
    rng_state=payload["rng_state"],
    fp32_checked=fp32_checked,
    torch_version=str(payload["torch_version"]),
    cuda_version=None
    if payload["cuda_version"] is None
    else str(payload["cuda_version"]),
  )


__all__ = ["CheckpointError", "CheckpointState", "load_checkpoint", "save_checkpoint"]
