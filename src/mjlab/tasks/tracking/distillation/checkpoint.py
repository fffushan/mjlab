"""Versioned, atomic persistence for the bounded distillation lifecycle.

Only validated tensor/plain-data state is serialized.  Live simulator objects,
teachers, and callables never cross this boundary.  Loading validates all
compatibility contracts before mutating the supplied trainer or replay.

Two checkpoint versions exist and neither is reinterpreted as the other:

* ``version 1`` (``mjlab-m3-distillation``) is the single-teacher M3 training
  checkpoint.  Its resume path (:func:`load_checkpoint`) stays byte-strict on
  the stored teacher hashes and control contract, and
  :func:`load_inference_checkpoint` restores only the saved model schema,
  settings, weights, and normalizers for model-only evaluation.
* ``version 2`` (``mjlab-m4-cohort-distillation``) adds the ordered cohort
  identity (every member's artifact digests, clip extent and audited body
  mapping, the common contract, the slot/phase policy, the per-motion replay
  partition policy, and the resource/seed settings) plus the live replay policy
  declaration.  :func:`load_cohort_checkpoint` is the strict multi-teacher
  resume, and :func:`load_cohort_member_inference` is the checked model-only
  selection of one saved cohort member.

A checkpoint is therefore a training artifact: it legitimately carries the
optimizer, raw replay, collector RNG, and trainer minibatch/replay settings.
No loader ever converts a version-1 checkpoint into a version-2 cohort run or
the other way round.
"""

from __future__ import annotations

import copy
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mjlab.tasks.tracking.distillation.cohort_contract import (
  BALANCED_REPLAY_KIND,
  COHORT_ARTIFACT_ROLES,
  FIFO_REPLAY_KIND,
  CohortContractError,
  CohortIdentity,
  CohortMember,
  ReplayPolicy,
  require_member_matches,
  require_same_cohort,
  require_same_replay_policy,
)
from mjlab.tasks.tracking.distillation.collector import DAggerCollector
from mjlab.tasks.tracking.distillation.config import CohortContract
from mjlab.tasks.tracking.distillation.model import ConditionalVAE
from mjlab.tasks.tracking.distillation.storage import (
  ReplayBufferProtocol,
  ReplayValidationError,
)
from mjlab.tasks.tracking.distillation.trainer import (
  VaeDistillationTrainer,
)
from mjlab.tasks.tracking.distillation.vae_config import ModelSettings, VaeSchema

CHECKPOINT_VERSION = 1
COHORT_CHECKPOINT_VERSION = 2
_CHECKPOINT_KIND = "mjlab-m3-distillation"
_COHORT_CHECKPOINT_KIND = "mjlab-m4-cohort-distillation"
_KIND_DESCRIPTIONS = {
  _CHECKPOINT_KIND: "version-1 single-teacher training checkpoint",
  _COHORT_CHECKPOINT_KIND: "version-2 multi-teacher cohort checkpoint",
}
_STATE_FIELDS = frozenset(
  {
    "kind",
    "version",
    "schema",
    "model_settings",
    "trainer_config",
    "model",
    "optimizer",
    "trainer_counters",
    "counters",
    "schedule",
    "resolved_config",
    "replay",
    "rng",
    "resume_restarts_simulator",
  }
)
_CHECKPOINT_IDENTITY_FIELDS = frozenset({"teacher_hashes", "control_contract"})
_COHORT_IDENTITY_FIELDS = frozenset({"cohort", "cohort_digest", "replay_policy"})
_INFERENCE_FIELDS = ("schema", "model_settings", "model")


class CheckpointValidationError(ValueError):
  """A checkpoint is malformed or incompatible with the live components."""


@dataclass(frozen=True, slots=True)
class LifecycleState:
  """State returned after a validated checkpoint load."""

  counters: dict[str, int]
  schedule: dict[str, Any]
  resolved_config: dict[str, Any]
  teacher_hashes: dict[str, str]
  control_contract: dict[str, Any]
  resume_restarts_simulator: bool = True


@dataclass(frozen=True, slots=True)
class CohortLifecycleState:
  """State returned after a validated multi-teacher cohort checkpoint load."""

  counters: dict[str, int]
  schedule: dict[str, Any]
  resolved_config: dict[str, Any]
  cohort: CohortIdentity
  replay_policy: ReplayPolicy
  resume_restarts_simulator: bool = True


@dataclass(frozen=True, slots=True)
class InferenceModel:
  """Model-only artifact reconstructed from a training checkpoint.

  The saved optimizer, replay, RNG, and trainer settings are deliberately not
  represented: model-only evaluation must not require them.  ``schema`` and
  ``settings`` are the *saved* identities used to rebuild ``model``.
  """

  model: ConditionalVAE
  schema: VaeSchema
  settings: ModelSettings
  counters: dict[str, int]
  schedule: dict[str, Any]
  resolved_config: dict[str, Any]
  teacher_hashes: dict[str, str]
  control_contract: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CohortMemberInference:
  """Model-only artifact plus the full trained cohort identity it belongs to.

  ``member`` is the saved identity of the requested cohort member and
  ``cohort`` is the whole stored cohort, so an evaluation/playback/export
  report can retain the complete provenance of the shared student while it pins
  one member.  ``relocated_artifact_roles`` names the artifact roles whose
  stored path differs from the live manifest: each of them was accepted only
  because its content digest matched.
  """

  model: ConditionalVAE
  schema: VaeSchema
  settings: ModelSettings
  cohort: CohortIdentity
  member: CohortMember
  requested_teacher_id: str
  counters: dict[str, int]
  schedule: dict[str, Any]
  resolved_config: dict[str, Any]
  relocated_artifact_roles: tuple[str, ...] = ()

  @property
  def teacher_ids(self) -> tuple[str, ...]:
    """Every member of the trained cohort, in the stored order."""
    return self.cohort.teacher_ids

  @property
  def motion_id(self) -> int:
    return self.member.motion_id

  @property
  def teacher_code(self) -> int:
    return self.member.teacher_code

  @property
  def artifact_hashes(self) -> dict[str, str]:
    """Content digests the member selection verified, keyed by artifact role."""
    return dict(self.member.hashes)


def _plain(value: Any, name: str) -> Any:
  """Reject executable/non-portable checkpoint metadata."""
  if value is None or isinstance(value, (str, int, float, bool)):
    return value
  if isinstance(value, (list, tuple)):
    return [_plain(item, name) for item in value]
  if isinstance(value, Mapping):
    if any(not isinstance(key, str) for key in value):
      raise CheckpointValidationError(f"{name} has a non-string key")
    return {key: _plain(item, name) for key, item in value.items()}
  raise CheckpointValidationError(f"{name} contains unsupported value {type(value)!r}")


def _mapping(value: Any, name: str) -> dict[str, Any]:
  if not isinstance(value, Mapping):
    raise CheckpointValidationError(f"{name} must be a mapping")
  result = _plain(value, name)
  assert isinstance(result, dict)
  return result


def _hashes(value: Any, name: str) -> dict[str, str]:
  result = _mapping(value, name)
  if any(
    not isinstance(key, str) or not isinstance(item, str)
    for key, item in result.items()
  ):
    raise CheckpointValidationError(f"{name} must map strings to strings")
  return result


def _schema_metadata(value: Any, name: str) -> dict[str, Any]:
  if isinstance(value, VaeSchema):
    return value.compatibility_metadata()
  return _mapping(value, name)


def _model_settings(metadata: Any) -> ModelSettings:
  """Rebuild the *saved* model settings; never a hard-coded architecture."""
  if not isinstance(metadata, Mapping) or set(metadata) != {
    "latent_dim",
    "hidden_dims",
    "activation",
    "beta",
  }:
    raise CheckpointValidationError("checkpoint model settings are malformed")
  hidden_dims = metadata["hidden_dims"]
  if not isinstance(hidden_dims, (list, tuple)) or not hidden_dims:
    raise CheckpointValidationError(
      "checkpoint model settings hidden_dims must be a non-empty sequence"
    )
  if any(
    not isinstance(width, int) or isinstance(width, bool) or width <= 0
    for width in hidden_dims
  ):
    raise CheckpointValidationError(
      "checkpoint model settings hidden_dims must be positive integers"
    )
  try:
    return ModelSettings(
      latent_dim=metadata["latent_dim"],
      hidden_dims=tuple(hidden_dims),
      activation=metadata["activation"],
      beta=metadata["beta"],
    )
  except ValueError as exc:
    raise CheckpointValidationError(
      f"checkpoint model settings are invalid: {exc}"
    ) from exc


def _check_inference_control_contract(
  stored: Mapping[str, Any],
  expected: Mapping[str, Any],
  *,
  motion_hashes_verified: bool,
) -> None:
  """Compare an inference control contract, allowing a relocated motion artifact.

  ``control_contract['motion']`` records the absolute path of the reference file
  a checkpoint was produced from.  An inference-only load may run from a
  byte-identical copy at a different path, which changes only that entry.  The
  relocation is accepted only when the caller supplied teacher hashes that
  matched the checkpoint and include the motion artifact: that digest is the
  evidence the reference content is identical, and every other contract field
  must still agree exactly.  Training resume (:func:`load_checkpoint`) does not
  use this relaxation and stays byte-for-byte strict.
  """
  if stored == expected:
    return
  only_motion_differs = (
    set(stored) == set(expected)
    and "motion" in stored
    and "motion" in expected
    and all(stored[key] == expected[key] for key in stored if key != "motion")
  )
  if only_motion_differs and motion_hashes_verified:
    return
  if only_motion_differs:
    raise CheckpointValidationError(
      "control contract motion path differs from the checkpoint and the motion "
      "content identity was not verified by a matching motion artifact hash"
    )
  raise CheckpointValidationError("control contract does not match checkpoint")


def _trainer_config(trainer: VaeDistillationTrainer) -> dict[str, Any]:
  config = trainer.config
  return {
    "learning_rate": config.learning_rate,
    "beta": config.beta,
    "accumulation_steps": config.accumulation_steps,
    "minibatch_size": config.minibatch_size,
    "latent_mode": config.latent_mode,
  }


def _global_rng_state() -> torch.Tensor:
  return torch.random.get_rng_state().detach().clone()


def _validate_rng_state(value: Any, name: str) -> torch.Tensor:
  """Return the state as the CPU uint8 vector PyTorch's RNG APIs demand.

  ``torch.load(map_location=...)`` may place a saved state on another device,
  and a checkpoint may store the state under a wider dtype holding the same
  bytes.  Both ``torch.random.set_rng_state`` and
  ``torch.Generator.set_state`` reject anything but a CPU ``uint8`` vector, so
  normalize here and return the normalized tensor for the caller to install
  instead of the raw payload tensor.  Reinterpreting another dtype as bytes is
  lossless; the probe below still rejects any payload that is not a real
  generator state.
  """
  if not isinstance(value, torch.Tensor) or value.ndim != 1:
    raise CheckpointValidationError(f"{name} must be a uint8 vector")
  try:
    state = value.detach().to("cpu").contiguous()
    if state.dtype != torch.uint8:
      state = state.view(torch.uint8)
    state = state.clone()
    probe = torch.Generator(device="cpu")
    probe.set_state(state)
  except (RuntimeError, TypeError, ValueError) as exc:
    raise CheckpointValidationError(f"{name} has an invalid generator state") from exc
  return state


def _require_envelope(
  payload: Any, *, kind: str, version: int, other_kind: str, other_loader: str
) -> None:
  """Require one exact checkpoint kind/version and name the sibling loader.

  A checkpoint of the other version is reported together with the loader that
  reads it, so a version-1 single-teacher artifact is never silently accepted
  as a multi-teacher cohort checkpoint, or the reverse.
  """
  if not isinstance(payload, Mapping):
    raise CheckpointValidationError("checkpoint root must be a mapping")
  if payload.get("kind") == kind and payload.get("version") == version:
    return
  if payload.get("kind") == other_kind:
    raise CheckpointValidationError(
      f"checkpoint is a {_KIND_DESCRIPTIONS[other_kind]}; read it with "
      f"{other_loader} instead"
    )
  raise CheckpointValidationError("unsupported or malformed checkpoint version")


def _payload(
  trainer: VaeDistillationTrainer,
  replay: ReplayBufferProtocol,
  *,
  counters: Mapping[str, int],
  schedule: Mapping[str, Any],
  resolved_config: Mapping[str, Any],
  collector: DAggerCollector | None,
) -> dict[str, Any]:
  """The trainer/model/replay/RNG half shared by both checkpoint versions."""
  if trainer.replay is not replay:
    raise CheckpointValidationError(
      "trainer and replay must be the same ownership instance"
    )
  clean_counters = _mapping(counters, "counters")
  if any(
    not isinstance(value, int) or isinstance(value, bool)
    for value in clean_counters.values()
  ):
    raise CheckpointValidationError("counters must contain integers")
  clean_schedule = _mapping(schedule, "schedule")
  clean_config = _mapping(resolved_config, "resolved_config")
  rng: dict[str, Any] = {
    "global_cpu": _global_rng_state(),
    "trainer": trainer.generator_states(),
  }
  if collector is not None:
    rng["collector"] = collector.generator_state()
  return {
    "schema": trainer.model.schema_metadata,
    "model_settings": trainer.model.settings.to_metadata(),
    "trainer_config": _trainer_config(trainer),
    "model": {
      key: value.detach().clone()
      if isinstance(value, torch.Tensor)
      else copy.deepcopy(value)
      for key, value in trainer.model.state_dict().items()
    },
    "optimizer": copy.deepcopy(trainer.optimizer.state_dict()),
    "trainer_counters": {
      "optimizer_steps": trainer.optimizer_steps,
      "microbatches_seen": trainer.microbatches_seen,
      "samples_seen": trainer.samples_seen,
      "normalization_update_ids": list(trainer.normalization_update_ids),
    },
    "counters": clean_counters,
    "schedule": clean_schedule,
    "resolved_config": clean_config,
    "replay": replay.state_dict(),
    "rng": rng,
    "resume_restarts_simulator": True,
  }


def _single_teacher_payload(
  trainer: VaeDistillationTrainer,
  replay: ReplayBufferProtocol,
  *,
  counters: Mapping[str, int],
  schedule: Mapping[str, Any],
  resolved_config: Mapping[str, Any],
  teacher_hashes: Mapping[str, str],
  control_contract: Mapping[str, Any],
  collector: DAggerCollector | None,
) -> dict[str, Any]:
  """The version-1 payload: shared state plus the single-teacher identity."""
  payload = _payload(
    trainer,
    replay,
    counters=counters,
    schedule=schedule,
    resolved_config=resolved_config,
    collector=collector,
  )
  payload["kind"] = _CHECKPOINT_KIND
  payload["version"] = CHECKPOINT_VERSION
  payload["teacher_hashes"] = _hashes(teacher_hashes, "teacher_hashes")
  payload["control_contract"] = _mapping(control_contract, "control_contract")
  return payload


def _atomic_save(payload: Mapping[str, Any], destination: Path) -> Path:
  """Write one payload to a temporary sibling, then replace the destination."""
  destination.parent.mkdir(parents=True, exist_ok=True)
  fd, temporary = tempfile.mkstemp(
    prefix=f".{destination.name}.", dir=destination.parent
  )
  try:
    with os.fdopen(fd, "wb") as handle:
      torch.save(dict(payload), handle)
      handle.flush()
      os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return destination
  except Exception:
    try:
      os.unlink(temporary)
    except FileNotFoundError:
      pass
    raise


def _validate_state_payload(
  payload: Any,
  trainer: VaeDistillationTrainer,
  replay: ReplayBufferProtocol,
  *,
  kind: str,
  version: int,
  identity_fields: frozenset[str],
  collector: DAggerCollector | None,
) -> dict[str, Any]:
  """Validate the trainer/model/replay/RNG half shared by both versions."""
  other_kind = _COHORT_CHECKPOINT_KIND if kind == _CHECKPOINT_KIND else _CHECKPOINT_KIND
  other_loader = (
    "load_cohort_checkpoint" if kind == _CHECKPOINT_KIND else "load_checkpoint"
  )
  _require_envelope(
    payload,
    kind=kind,
    version=version,
    other_kind=other_kind,
    other_loader=other_loader,
  )
  required = _STATE_FIELDS | identity_fields
  if set(payload) != required:
    raise CheckpointValidationError(
      "checkpoint fields are missing or unknown: missing "
      f"{sorted(required - set(payload))}, unknown {sorted(set(payload) - required)}"
    )
  if (
    payload["schema"] != trainer.model.schema_metadata
    or payload["schema"] != replay.schema.compatibility_metadata()
  ):
    raise CheckpointValidationError(
      "checkpoint schema does not match live model/replay"
    )
  if payload["model_settings"] != trainer.model.settings.to_metadata():
    raise CheckpointValidationError("checkpoint model settings do not match live model")
  if payload["trainer_config"] != _trainer_config(trainer):
    raise CheckpointValidationError(
      "checkpoint trainer configuration does not match live trainer"
    )
  counters = _mapping(payload["counters"], "checkpoint counters")
  if any(
    not isinstance(value, int) or isinstance(value, bool) for value in counters.values()
  ):
    raise CheckpointValidationError("checkpoint counters must contain integers")
  _mapping(payload["schedule"], "checkpoint schedule")
  _mapping(payload["resolved_config"], "checkpoint resolved_config")
  model_state = payload["model"]
  if not isinstance(model_state, Mapping):
    raise CheckpointValidationError("checkpoint model state is invalid")
  expected_model_keys = set(trainer.model.state_dict())
  if set(model_state) != expected_model_keys:
    raise CheckpointValidationError(
      "checkpoint model parameters do not match live model"
    )
  for key, value in model_state.items():
    current = trainer.model.state_dict()[key]
    if isinstance(current, torch.Tensor):
      if not isinstance(value, torch.Tensor):
        raise CheckpointValidationError(f"checkpoint model tensor {key!r} is missing")
      if value.shape != current.shape or value.dtype != current.dtype:
        raise CheckpointValidationError(
          f"checkpoint model tensor {key!r} is incompatible"
        )
      if not torch.isfinite(value).all().item():
        raise CheckpointValidationError(
          f"checkpoint model tensor {key!r} is non-finite"
        )
    elif value != current:
      raise CheckpointValidationError(
        f"checkpoint model metadata {key!r} is incompatible"
      )
  tc = _mapping(payload["trainer_counters"], "trainer_counters")
  required_tc = {
    "optimizer_steps",
    "microbatches_seen",
    "samples_seen",
    "normalization_update_ids",
  }
  if set(tc) != required_tc or any(
    not isinstance(tc[key], int) or isinstance(tc[key], bool)
    for key in required_tc - {"normalization_update_ids"}
  ):
    raise CheckpointValidationError("checkpoint trainer counters are invalid")
  if not isinstance(tc["normalization_update_ids"], list):
    raise CheckpointValidationError("checkpoint normalization identities are invalid")
  if len(set(map(repr, tc["normalization_update_ids"]))) != len(
    tc["normalization_update_ids"]
  ):
    raise CheckpointValidationError(
      "checkpoint normalization identities are duplicated"
    )
  rng = payload["rng"]
  if not isinstance(rng, Mapping) or set(rng) != (
    {"global_cpu", "trainer"} | ({"collector"} if collector is not None else set())
  ):
    raise CheckpointValidationError(
      "checkpoint RNG streams do not match live components"
    )
  # Restore must install these normalized states, not the raw payload ones:
  # a checkpoint read with a CUDA ``map_location`` carries CUDA tensors here.
  clean_rng: dict[str, Any] = {
    "global_cpu": _validate_rng_state(rng["global_cpu"], "global RNG state")
  }
  trainer_states = rng["trainer"]
  if not isinstance(trainer_states, Mapping) or set(trainer_states) != {
    "replay",
    "latent",
  }:
    raise CheckpointValidationError("checkpoint trainer RNG state is invalid")
  clean_rng["trainer"] = {
    "replay": _validate_rng_state(trainer_states["replay"], "replay RNG state"),
    "latent": _validate_rng_state(trainer_states["latent"], "latent RNG state"),
  }
  if collector is not None:
    clean_rng["collector"] = _validate_rng_state(
      rng["collector"], "collector RNG state"
    )
  try:
    # The public, storage-independent validation seam: no replay implementation's
    # private ring layout is read here.
    replay.validate_state(payload["replay"])
  except ReplayValidationError as exc:
    raise CheckpointValidationError(str(exc)) from exc
  if not isinstance(payload["optimizer"], Mapping):
    raise CheckpointValidationError("checkpoint optimizer state is invalid")
  clean = dict(payload)
  clean["rng"] = clean_rng
  return clean


def _validate_payload(
  payload: Any,
  trainer: VaeDistillationTrainer,
  replay: ReplayBufferProtocol,
  *,
  expected_teacher_hashes: Mapping[str, str] | None,
  expected_control_contract: Mapping[str, Any] | None,
  collector: DAggerCollector | None,
) -> dict[str, Any]:
  """Validate one version-1 checkpoint, including its stored identity."""
  clean = _validate_state_payload(
    payload,
    trainer,
    replay,
    kind=_CHECKPOINT_KIND,
    version=CHECKPOINT_VERSION,
    identity_fields=_CHECKPOINT_IDENTITY_FIELDS,
    collector=collector,
  )
  hashes = _hashes(clean["teacher_hashes"], "checkpoint teacher_hashes")
  if expected_teacher_hashes is not None and hashes != _hashes(
    expected_teacher_hashes, "expected_teacher_hashes"
  ):
    raise CheckpointValidationError("teacher artifact hashes do not match checkpoint")
  contract = _mapping(clean["control_contract"], "checkpoint control_contract")
  if expected_control_contract is not None and contract != _mapping(
    expected_control_contract, "expected_control_contract"
  ):
    raise CheckpointValidationError("control contract does not match checkpoint")
  return clean


def _validate_cohort_payload(
  payload: Any,
  trainer: VaeDistillationTrainer,
  replay: ReplayBufferProtocol,
  *,
  expected_cohort: CohortIdentity,
  collector: DAggerCollector | None,
) -> tuple[dict[str, Any], ReplayPolicy]:
  """Validate one multi-teacher cohort checkpoint before any mutation.

  The stored cohort record and its digest, the recorded replay partition policy
  (against the live buffer and against the cohort record), and the saved
  schema's joint order are checked, and then the whole stored cohort must be
  reproduced by ``expected_cohort``: the same ordered members, artifact digests
  and paths, clip extents, audited body mapping, common contract, slot/phase
  policy, replay policy, and resource/seed settings.
  """
  if not isinstance(expected_cohort, CohortIdentity):
    raise CheckpointValidationError(
      "a strict cohort resume needs the live CohortIdentity to compare against"
    )
  # The M4 replay layout is required before anything else is inspected, so a
  # version-2 checkpoint is never reported against a buffer it cannot use.
  live_policy = _live_replay_policy(replay)
  clean = _validate_state_payload(
    payload,
    trainer,
    replay,
    kind=_COHORT_CHECKPOINT_KIND,
    version=COHORT_CHECKPOINT_VERSION,
    identity_fields=_COHORT_IDENTITY_FIELDS,
    collector=collector,
  )
  try:
    stored = CohortIdentity.from_dict(clean["cohort"], "checkpoint cohort")
  except CohortContractError as exc:
    raise CheckpointValidationError(
      f"checkpoint cohort record is invalid: {exc}"
    ) from exc
  digest = clean["cohort_digest"]
  if not isinstance(digest, str) or digest != stored.digest():
    raise CheckpointValidationError(
      "checkpoint cohort digest does not match its own cohort record"
    )
  try:
    stored_policy = ReplayPolicy.from_dict(
      clean["replay_policy"], "checkpoint replay_policy"
    )
  except CohortContractError as exc:
    raise CheckpointValidationError(
      f"checkpoint replay policy is invalid: {exc}"
    ) from exc
  if stored.replay.as_dict() != stored_policy.as_dict():
    raise CheckpointValidationError(
      "checkpoint cohort record and replay policy describe different replay partitions"
    )
  try:
    require_same_replay_policy(stored_policy, live_policy)
    require_same_cohort(stored, expected_cohort)
  except CohortContractError as exc:
    raise CheckpointValidationError(f"cohort resume refused: {exc}") from exc
  schema = clean["schema"]
  joint_order = schema.get("joint_order") if isinstance(schema, Mapping) else None
  if joint_order is None or tuple(joint_order) != stored.common.joint_names:
    raise CheckpointValidationError(
      "checkpoint schema joint order disagrees with its cohort contract"
    )
  return clean, stored_policy


def _restore_state(
  clean: Mapping[str, Any],
  trainer: VaeDistillationTrainer,
  replay: ReplayBufferProtocol,
  collector: DAggerCollector | None,
) -> None:
  """Install one validated payload; any failure rolls every owned state back.

  Model, optimizer, every replay partition, the trainer counters, and every RNG
  stream (including the collector's) are installed together, so a failure that
  happens after some of them were restored -- a rejected later partition or an
  optimizer state Adam refuses -- leaves the previous state intact.
  """
  old_model = {
    key: value.detach().clone()
    if isinstance(value, torch.Tensor)
    else copy.deepcopy(value)
    for key, value in trainer.model.state_dict().items()
  }
  old_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
  old_replay = replay.state_dict()
  old_counters = (
    trainer.optimizer_steps,
    trainer.microbatches_seen,
    trainer.samples_seen,
    set(trainer._normalization_update_ids),  # noqa: SLF001
  )
  old_rng = trainer.generator_states()
  old_global = _global_rng_state()
  old_collector = None if collector is None else collector.generator_state()
  try:
    trainer.model.load_state_dict(clean["model"], strict=True)
    trainer.optimizer.load_state_dict(clean["optimizer"])
    replay.load_state_dict(clean["replay"])
    tc = clean["trainer_counters"]
    trainer.optimizer_steps = tc["optimizer_steps"]
    trainer.microbatches_seen = tc["microbatches_seen"]
    trainer.samples_seen = tc["samples_seen"]
    trainer._normalization_update_ids = set(tc["normalization_update_ids"])  # noqa: SLF001
    trainer.set_generator_states(clean["rng"]["trainer"])
    torch.random.set_rng_state(clean["rng"]["global_cpu"])
    if collector is not None:
      collector.set_generator_state(clean["rng"]["collector"])
    # Clear post-step poison only as the last step of a fully validated
    # restore.  A failure here rolls the model/optimizer/replay/RNG back and
    # therefore leaves both the previous state and the poison intact.
    trainer.clear_poison_after_validated_restore()
  except Exception as exc:
    trainer.model.load_state_dict(old_model, strict=True)
    trainer.optimizer.load_state_dict(old_optimizer)
    replay.load_state_dict(old_replay)
    trainer.optimizer_steps, trainer.microbatches_seen, trainer.samples_seen, ids = (
      old_counters
    )
    trainer._normalization_update_ids = ids  # noqa: SLF001
    trainer.set_generator_states(old_rng)
    torch.random.set_rng_state(old_global)
    if collector is not None and old_collector is not None:
      collector.set_generator_state(old_collector)
    raise CheckpointValidationError(
      f"checkpoint restore failed before completion: {exc}"
    ) from exc


def save_checkpoint(
  path: str | os.PathLike[str],
  trainer: VaeDistillationTrainer,
  replay: ReplayBufferProtocol,
  *,
  counters: Mapping[str, int] | None = None,
  schedule: Mapping[str, Any] | None = None,
  resolved_config: Mapping[str, Any] | None = None,
  teacher_hashes: Mapping[str, str] | None = None,
  control_contract: Mapping[str, Any] | None = None,
  collector: DAggerCollector | None = None,
) -> Path:
  """Atomically save a validated CPU-safe version-1 lifecycle checkpoint.

  A poisoned trainer (post-step failure without a validated restore) is
  refused through :meth:`VaeDistillationTrainer.assert_healthy` before any file
  is created, so a corrupt in-memory state can never be persisted.
  """
  trainer.assert_healthy()
  _require_fifo_replay(replay, "save_checkpoint")
  payload = _single_teacher_payload(
    trainer,
    replay,
    counters=counters or {},
    schedule=schedule or {},
    resolved_config=resolved_config or {},
    teacher_hashes=teacher_hashes or {},
    control_contract=control_contract or {},
    collector=collector,
  )
  return _atomic_save(payload, Path(path))


def save_cohort_checkpoint(
  path: str | os.PathLike[str],
  trainer: VaeDistillationTrainer,
  replay: ReplayBufferProtocol,
  *,
  cohort: CohortIdentity,
  counters: Mapping[str, int] | None = None,
  schedule: Mapping[str, Any] | None = None,
  resolved_config: Mapping[str, Any] | None = None,
  collector: DAggerCollector | None = None,
) -> Path:
  """Atomically save a version-2 multi-teacher cohort checkpoint.

  The checkpoint records the ordered cohort identity (member digests, clip
  extents, audited body mapping, common contract, slot/phase policy, replay
  partition policy, and resource/seed settings), its content digest, and the
  live replay policy, so a strict resume can require all of them.  A poisoned
  trainer and a cohort that disagrees with the live replay or the trained schema
  are refused before any file is created.
  """
  trainer.assert_healthy()
  if not isinstance(cohort, CohortIdentity):
    raise CheckpointValidationError(
      "save_cohort_checkpoint needs the live CohortIdentity to record"
    )
  policy = _live_replay_policy(replay)
  try:
    require_same_replay_policy(cohort.replay, policy)
  except CohortContractError as exc:
    raise CheckpointValidationError(
      f"cohort record does not describe the live replay buffer: {exc}"
    ) from exc
  joint_order = trainer.model.schema_metadata.get("joint_order")
  if tuple(joint_order or ()) != cohort.common.joint_names:
    raise CheckpointValidationError(
      "the trained model's schema joint order disagrees with the cohort's saved "
      "joint order"
    )
  payload = _payload(
    trainer,
    replay,
    counters=counters or {},
    schedule=schedule or {},
    resolved_config=resolved_config or {},
    collector=collector,
  )
  payload["kind"] = _COHORT_CHECKPOINT_KIND
  payload["version"] = COHORT_CHECKPOINT_VERSION
  payload["cohort"] = cohort.as_dict()
  payload["cohort_digest"] = cohort.digest()
  payload["replay_policy"] = policy.as_dict()
  return _atomic_save(payload, Path(path))


def _live_replay_policy(replay: ReplayBufferProtocol) -> ReplayPolicy:
  """Replay policy of the live buffer, refusing anything but the M4 layout."""
  try:
    policy = ReplayPolicy.from_buffer(replay)
  except CohortContractError as exc:
    raise CheckpointValidationError(str(exc)) from exc
  if policy.kind != BALANCED_REPLAY_KIND:
    raise CheckpointValidationError(
      "a version-2 cohort checkpoint requires the per-motion balanced replay "
      f"buffer, got {policy.kind!r}"
    )
  return policy


def _require_fifo_replay(replay: ReplayBufferProtocol, caller: str) -> ReplayPolicy:
  """Require the single-partition FIFO layout a version-1 checkpoint stores.

  Without this a version-1 checkpoint could silently record the per-motion
  balanced replay layout, which carries no version-1 replay partition identity
  for a resume to compare.
  """
  try:
    policy = ReplayPolicy.from_buffer(replay)
  except CohortContractError as exc:
    raise CheckpointValidationError(str(exc)) from exc
  if policy.kind != FIFO_REPLAY_KIND:
    raise CheckpointValidationError(
      f"{caller} writes version-1 single-teacher checkpoints backed by the "
      f"single-partition FIFO replay buffer, got {policy.kind!r}; use "
      "save_cohort_checkpoint for the per-motion balanced replay buffer"
    )
  return policy


def _load_payload(
  path: str | os.PathLike[str], map_location: str | torch.device
) -> Any:
  try:
    return torch.load(path, map_location=map_location, weights_only=True)
  except Exception as exc:
    raise CheckpointValidationError(f"could not load checkpoint: {exc}") from exc


def _build_inference_model(
  payload: Mapping[str, Any],
  *,
  device: str | torch.device,
  expected_schema: VaeSchema | Mapping[str, Any] | None,
) -> tuple[ConditionalVAE, VaeSchema, ModelSettings]:
  """Rebuild the saved schema/settings and a validated model-only instance.

  Shared by the version-1 model-only loader and the version-2 checked cohort
  member loader: both infer the trained architecture from the checkpoint instead
  of a caller-supplied default, both validate every weight key, shape, dtype, and
  finiteness, and both move only the model/normalizers to ``device``.
  """
  missing = [name for name in _INFERENCE_FIELDS if name not in payload]
  if missing:
    raise CheckpointValidationError(
      f"checkpoint is missing inference fields {sorted(missing)}"
    )
  try:
    schema = VaeSchema.from_metadata(payload["schema"])
  except ValueError as exc:
    raise CheckpointValidationError(f"checkpoint schema is invalid: {exc}") from exc
  settings = _model_settings(payload["model_settings"])
  if (
    expected_schema is not None
    and _schema_metadata(expected_schema, "expected_schema")
    != schema.compatibility_metadata()
  ):
    raise CheckpointValidationError(
      "checkpoint schema does not match the expected schema"
    )
  model_state = payload["model"]
  if not isinstance(model_state, Mapping):
    raise CheckpointValidationError("checkpoint model state is invalid")
  model = ConditionalVAE(schema, settings)
  current_state = model.state_dict()
  if set(model_state) != set(current_state):
    raise CheckpointValidationError(
      "checkpoint weights do not match the saved model schema/settings"
    )
  for key, value in model_state.items():
    current = current_state[key]
    if isinstance(current, torch.Tensor):
      if not isinstance(value, torch.Tensor):
        raise CheckpointValidationError(f"checkpoint model tensor {key!r} is missing")
      if value.shape != current.shape or value.dtype != current.dtype:
        raise CheckpointValidationError(
          f"checkpoint model tensor {key!r} is incompatible"
        )
      if not torch.isfinite(value).all().item():
        raise CheckpointValidationError(
          f"checkpoint model tensor {key!r} is non-finite"
        )
    elif value != current:
      raise CheckpointValidationError(
        f"checkpoint model metadata {key!r} is incompatible"
      )
  try:
    model.load_state_dict(dict(model_state), strict=True)
  except Exception as exc:
    raise CheckpointValidationError(
      f"checkpoint weights could not be restored: {exc}"
    ) from exc
  model.to(device)
  model.eval()
  return model, schema, settings


def load_checkpoint(
  path: str | os.PathLike[str],
  trainer: VaeDistillationTrainer,
  replay: ReplayBufferProtocol,
  *,
  expected_teacher_hashes: Mapping[str, str] | None = None,
  expected_control_contract: Mapping[str, Any] | None = None,
  collector: DAggerCollector | None = None,
  map_location: str | torch.device = "cpu",
) -> LifecycleState:
  """Validate then restore version-1 state; failures roll all state back."""
  payload = _load_payload(path, map_location)
  clean = _validate_payload(
    payload,
    trainer,
    replay,
    expected_teacher_hashes=expected_teacher_hashes,
    expected_control_contract=expected_control_contract,
    collector=collector,
  )
  _restore_state(clean, trainer, replay, collector)
  return LifecycleState(
    counters=dict(clean["counters"]),
    schedule=dict(clean["schedule"]),
    resolved_config=dict(clean["resolved_config"]),
    teacher_hashes=dict(clean["teacher_hashes"]),
    control_contract=dict(clean["control_contract"]),
    resume_restarts_simulator=True,
  )


def load_cohort_checkpoint(
  path: str | os.PathLike[str],
  trainer: VaeDistillationTrainer,
  replay: ReplayBufferProtocol,
  *,
  expected_cohort: CohortIdentity,
  collector: DAggerCollector | None = None,
  map_location: str | torch.device = "cpu",
) -> CohortLifecycleState:
  """Validate then restore version-2 cohort state; failures roll state back.

  ``expected_cohort`` is the live identity of the cohort the caller built, so a
  resume refuses a checkpoint whose ordered members, artifact digests, clip
  extents, body mapping, common contract, slot/phase policy, replay policy, or
  resource/seed settings differ.  A version-1 checkpoint is never converted.
  """
  payload = _load_payload(path, map_location)
  clean, policy = _validate_cohort_payload(
    payload,
    trainer,
    replay,
    expected_cohort=expected_cohort,
    collector=collector,
  )
  stored = CohortIdentity.from_dict(clean["cohort"], "checkpoint cohort")
  _restore_state(clean, trainer, replay, collector)
  return CohortLifecycleState(
    counters=dict(clean["counters"]),
    schedule=dict(clean["schedule"]),
    resolved_config=dict(clean["resolved_config"]),
    cohort=stored,
    replay_policy=policy,
    resume_restarts_simulator=True,
  )


def load_inference_checkpoint(
  path: str | os.PathLike[str],
  *,
  device: str | torch.device = "cpu",
  expected_schema: VaeSchema | Mapping[str, Any] | None = None,
  expected_teacher_hashes: Mapping[str, str] | None = None,
  expected_control_contract: Mapping[str, Any] | None = None,
) -> InferenceModel:
  """Rebuild a model-only artifact from a version-1 training checkpoint.

  The saved schema and model settings are inferred, so the actual trained
  architecture is restored instead of a caller-supplied default.  Schema joint
  order, weight keys, shapes, dtypes, and finiteness, plus the teacher hashes
  and control contract, are validated before the model is returned.  Raw
  tensors are always read on CPU and only the model/normalizers are allocated
  on ``device``; no optimizer or replay buffer is constructed at all.  A
  version-2 cohort checkpoint is read with :func:`load_cohort_member_inference`.
  """
  payload = _load_payload(path, "cpu")
  _require_envelope(
    payload,
    kind=_CHECKPOINT_KIND,
    version=CHECKPOINT_VERSION,
    other_kind=_COHORT_CHECKPOINT_KIND,
    other_loader="load_cohort_member_inference",
  )
  required = {
    "kind",
    "version",
    *_INFERENCE_FIELDS,
    "teacher_hashes",
    "control_contract",
  }
  missing = required - set(payload)
  if missing:
    raise CheckpointValidationError(
      f"checkpoint is missing inference fields {sorted(missing)}"
    )
  hashes = _hashes(payload["teacher_hashes"], "checkpoint teacher_hashes")
  expected_hashes = (
    None
    if expected_teacher_hashes is None
    else _hashes(expected_teacher_hashes, "expected_teacher_hashes")
  )
  if expected_hashes is not None and hashes != expected_hashes:
    raise CheckpointValidationError("teacher artifact hashes do not match checkpoint")
  contract = _mapping(payload["control_contract"], "checkpoint control_contract")
  if expected_control_contract is not None:
    _check_inference_control_contract(
      contract,
      _mapping(expected_control_contract, "expected_control_contract"),
      # A matching motion digest is only evidence when both the checkpoint and
      # the caller actually recorded one.  A missing entry cannot prove that a
      # relocated path holds identical content, so it never authorizes a move.
      motion_hashes_verified=(
        expected_hashes is not None
        and "motion" in hashes
        and "motion" in expected_hashes
      ),
    )
  model, schema, settings = _build_inference_model(
    payload, device=device, expected_schema=expected_schema
  )
  counters = _mapping(payload.get("counters", {}), "checkpoint counters")
  if any(
    not isinstance(value, int) or isinstance(value, bool) for value in counters.values()
  ):
    raise CheckpointValidationError("checkpoint counters must contain integers")
  schedule = _mapping(payload.get("schedule", {}), "checkpoint schedule")
  resolved_config = _mapping(
    payload.get("resolved_config", {}), "checkpoint resolved_config"
  )
  # Drop the raw file contents so no training optimizer/replay tensors outlive
  # this call; only the requested model/normalizers were moved to the device.
  del payload
  return InferenceModel(
    model=model,
    schema=schema,
    settings=settings,
    counters=counters,
    schedule=schedule,
    resolved_config=resolved_config,
    teacher_hashes=hashes,
    control_contract=contract,
  )


def load_cohort_member_inference(
  path: str | os.PathLike[str],
  cohort: CohortContract,
  teacher_id: str,
  *,
  device: str | torch.device = "cpu",
  expected_schema: VaeSchema | Mapping[str, Any] | None = None,
) -> CohortMemberInference:
  """Load one checked member of a saved version-2 cohort for inference only.

  The requested member must exist in the stored cohort; its artifact digests,
  clip extent, saved reference digest, and the common action/control/
  observation contract must match the live manifest; and the stored cohort
  digest must match its own record.  A byte-identical *relocated* artifact is
  accepted by its content digest and reported in ``relocated_artifact_roles``,
  while every changed artifact fails on its digest.  Strict resume is untouched
  by this relaxation.

  Only the saved schema, model settings, weights, and normalizers are restored:
  no optimizer, replay buffer, trainer, or simulator is constructed, and the
  returned object keeps the whole stored cohort identity as provenance.  Loading
  the same checkpoint for another member returns identical model parameters.
  """
  payload = _load_payload(path, "cpu")
  _require_envelope(
    payload,
    kind=_COHORT_CHECKPOINT_KIND,
    version=COHORT_CHECKPOINT_VERSION,
    other_kind=_CHECKPOINT_KIND,
    other_loader="load_inference_checkpoint",
  )
  required = {
    "cohort",
    "cohort_digest",
    "replay_policy",
    *_INFERENCE_FIELDS,
  }
  missing = required - set(payload)
  if missing:
    raise CheckpointValidationError(
      f"checkpoint is missing inference fields {sorted(missing)}"
    )
  try:
    stored = CohortIdentity.from_dict(payload["cohort"], "checkpoint cohort")
  except CohortContractError as exc:
    raise CheckpointValidationError(
      f"checkpoint cohort record is invalid: {exc}"
    ) from exc
  digest = payload["cohort_digest"]
  if not isinstance(digest, str) or digest != stored.digest():
    raise CheckpointValidationError(
      "checkpoint cohort digest does not match its own cohort record"
    )
  try:
    policy = ReplayPolicy.from_dict(
      payload["replay_policy"], "checkpoint replay_policy"
    )
  except CohortContractError as exc:
    raise CheckpointValidationError(
      f"checkpoint replay policy is invalid: {exc}"
    ) from exc
  if stored.replay.as_dict() != policy.as_dict():
    raise CheckpointValidationError(
      "checkpoint cohort record and replay policy describe different replay partitions"
    )
  try:
    member = require_member_matches(stored, cohort, teacher_id)
  except CohortContractError as exc:
    raise CheckpointValidationError(f"cohort member selection refused: {exc}") from exc
  model, schema, settings = _build_inference_model(
    payload, device=device, expected_schema=expected_schema
  )
  if tuple(schema.joint_order) != stored.common.joint_names:
    raise CheckpointValidationError(
      "checkpoint schema joint order disagrees with its cohort contract"
    )
  live_paths = cohort.teacher(teacher_id).entry.paths()
  relocated = tuple(
    role
    for role in COHORT_ARTIFACT_ROLES
    if str(live_paths[role]) != member.artifacts.get(role)
  )
  counters = _mapping(payload.get("counters", {}), "checkpoint counters")
  if any(
    not isinstance(value, int) or isinstance(value, bool) for value in counters.values()
  ):
    raise CheckpointValidationError("checkpoint counters must contain integers")
  schedule = _mapping(payload.get("schedule", {}), "checkpoint schedule")
  resolved_config = _mapping(
    payload.get("resolved_config", {}), "checkpoint resolved_config"
  )
  del payload
  return CohortMemberInference(
    model=model,
    schema=schema,
    settings=settings,
    cohort=stored,
    member=member,
    requested_teacher_id=teacher_id,
    counters=counters,
    schedule=schedule,
    resolved_config=resolved_config,
    relocated_artifact_roles=relocated,
  )


__all__ = [
  "CHECKPOINT_VERSION",
  "COHORT_CHECKPOINT_VERSION",
  "CheckpointValidationError",
  "CohortLifecycleState",
  "CohortMemberInference",
  "InferenceModel",
  "LifecycleState",
  "load_checkpoint",
  "load_cohort_checkpoint",
  "load_cohort_member_inference",
  "load_inference_checkpoint",
  "save_checkpoint",
  "save_cohort_checkpoint",
]
