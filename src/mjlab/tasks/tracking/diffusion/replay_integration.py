"""Bounded offline real-model replay for the D3 handover seam.

Loads the frozen D2 denoiser checkpoint (EMA-selected, contract-checked), the
active no-mixture normalization statistics, the frozen X2 VAE cohort member decoder
and bounded recorded D1 pilot rows, then compares the
:class:`ReferenceFreeContinuation` seam against a direct existing-sampler +
denormalize + frozen-decoder path from identical generator state and input.

This module is deliberately simulator-free and claims no closed-loop, stability,
training-quality, real-world-conditioning or 50 Hz evidence: generated latents
are supplied to the frozen decoder as data and are never executed.  D1 recorded
rows carry no measured 99-D gravity conditioning, so the decoder conditioning is
a deterministic labeled synthetic fixture; the seam-vs-direct comparison is
independent of the conditioning values.  Every path is an explicit argument; no
absolute path is embedded in this library module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, cast

import numpy as np
import torch
from torch import Tensor, nn

from mjlab.tasks.tracking.distillation.vae_config import (
  ACTION_DIM,
  CONDITIONING_DIMS,
  LATENT_DIM,
  REFERENCE_DIM,
  REFERENCE_DIM_WITH_PREVIEW,
)

from ..flow_matching.checkpoint import (
  FLOW_CHECKPOINT_FORMAT,
  FlowCheckpointError,
  _restore_process,
  load_flow_checkpoint,
)
from ..flow_matching.process import FlowMatchingError, LinearNoiseProcess
from ..flow_matching.sampler import FlowGrid
from .checkpoint import CheckpointError, load_checkpoint
from .contract import DEFAULT_CONTRACT, ContractError, DiffusionContract
from .handover import (
  ContinuationPlan,
  ReferenceFreeContinuation,
)
from .model import (
  DenoiserSettings,
  StateLatentTransformer,
  contract_dimensions,
)
from .noising import clean_mask as build_clean_mask
from .projection import (
  ProjectionBundle,
  ProjectionError,
  StateLatentNormalizer,
  StateLatentTransform,
  StateLatentTransformLike,
)
from .sampler import SamplerConditions, SamplerConfig, sample_trajectory
from .schedule import DiffusionSchedule, InferenceGrid, build_inference_grid
from .state import WorldState
from .trainer import ExponentialMovingAverage

# Closed-loop Euler step count for the flow family.  Measured on 2026-10-15:
# closed-loop falls over 6 to 20 steps are statistically indistinguishable,
# while the cost is linear in the step count, and on the robot PC (Orin NX,
# TensorRT, clock locked) 6 steps is 15.82 ms against a 20 ms control period.
# The flow package's own ``DEFAULT_CALLS`` stays the offline reference grid of 20.
DEFAULT_CLOSED_LOOP_SOLVER_STEPS = 6

_SEAM_VS_DIRECT_TOLERANCE = 1.0e-6
_MAX_SAMPLES = 4
_MAX_SEED = 1000
_MAX_THREADS = 64
_SYNTHETIC_CONDITIONING_AMPLITUDE = 0.3
_SYNTHETIC_CONDITIONING_WAVENUMBER = 0.05


class ReplayError(ValueError):
  """A replay input, gate or invariant failed closed."""


def sha256_file(path: str | Path) -> str:
  """Stream a file through SHA256, failing closed on unreadable paths."""
  digest = hashlib.sha256()
  try:
    with Path(path).open("rb") as handle:
      for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
  except OSError as exc:
    raise ReplayError(f"could not read {path} for hashing: {exc}") from exc
  return digest.hexdigest()


def _is_sha256(value: object) -> bool:
  return (
    isinstance(value, str)
    and len(value) == 64
    and all(character in "0123456789abcdef" for character in value)
  )


def verify_file_digest(path: str | Path, expected: str, *, role: str) -> None:
  """Fail closed unless ``path`` hashes exactly to the expected digest."""
  if not _is_sha256(expected):
    raise ReplayError(
      f"{role} sha256 argument is not a lowercase 64-hex digest: {expected!r}"
    )
  actual = sha256_file(path)
  if actual != expected:
    raise ReplayError(
      f"{role} sha256 mismatch for {path}: expected {expected}, got {actual}"
    )


def load_active_contract(path: str | Path) -> DiffusionContract:
  """Load the active dimensional contract without a file-hash gate."""
  try:
    return DiffusionContract.from_yaml(path)
  except (ContractError, OSError) as exc:
    raise ReplayError(f"active diffusion contract rejected: {exc}") from exc


def verify_projection_consistency(
  checkpoint_hashes: Mapping[str, str], bundle: ProjectionBundle
) -> None:
  """Fail closed unless the checkpoint recorded exactly this bundle."""
  expected = {
    "matrix": bundle.matrix_sha256,
    "pseudoinverse": bundle.pseudoinverse_sha256,
    "statistics": bundle.statistics_sha256,
  }
  for key, value in expected.items():
    recorded = checkpoint_hashes.get(key)
    if recorded != value:
      raise ReplayError(
        f"checkpoint projection hash {key!r} differs from the supplied bundle: "
        f"checkpoint {recorded!r}, bundle {value!r}"
      )


def verify_normalization_consistency(
  checkpoint_hashes: Mapping[str, str], normalizer: StateLatentTransform
) -> None:
  """Fail closed unless the checkpoint recorded the supplied statistics."""
  recorded = checkpoint_hashes.get("statistics")
  if recorded != normalizer.statistics_sha256:
    raise ReplayError(
      "checkpoint normalization statistics differ from the supplied normalizer: "
      f"checkpoint {recorded!r}, normalizer {normalizer.statistics_sha256!r}"
    )


def apply_ema_selection(
  model: nn.Module, ema_state: Mapping[str, Any] | None
) -> tuple[bool, int | None]:
  """Restore the evaluation-selected weights into ``model``.

  Follows the training/evaluation convention in ``diffusion/trainer.py``
  (evaluation runs on the EMA shadow): a checkpoint that carries an EMA shadow
  has it copied into the model.  A checkpoint without EMA keeps its raw model
  weights and is reported as such.
  """
  if ema_state is None:
    return False, None
  ema = ExponentialMovingAverage(model)
  ema.load_state_dict(ema_state)
  ema.copy_to(model)
  return True, int(ema.updates)


@dataclass(frozen=True, slots=True)
class LoadedDenoiser:
  """Frozen contract-checked denoiser with its checkpoint provenance."""

  model: StateLatentTransformer
  schedule: DiffusionSchedule
  grid: InferenceGrid
  global_step: int
  config_hash: str
  contract_identity: str
  projection_hashes: dict[str, str]
  best_validation_loss: float | None
  selection_split: str | None
  ema_selected: bool
  ema_updates: int | None
  fp32_checked: bool
  config_seed: int | None


def build_schedule_from_contract(contract: DiffusionContract) -> DiffusionSchedule:
  """Build the frozen schedule from the active contract representation."""
  return DiffusionSchedule.from_contract(contract)


def load_denoiser(
  checkpoint_path: str | Path,
  *,
  contract: DiffusionContract,
  projection_bundle: StateLatentTransformLike | None = None,
  normalizer: StateLatentTransform | None = None,
  checkpoint_sha256: str | None = None,
  verify_projection_hashes: bool = True,
  verify_contract_identity: bool = True,
) -> LoadedDenoiser:
  """Load the D2 denoiser with optional reproduction hash checks.

  The model, checkpoint schema and tensor dimensions are always validated.  A
  caller may omit file and normalizer digests for fast paper reproduction when
  the artifact files are supplied explicitly.
  """
  if projection_bundle is not None and normalizer is not None:
    raise ReplayError("pass only one of projection_bundle or normalizer")
  transforms = normalizer if normalizer is not None else projection_bundle
  if transforms is None:
    raise ReplayError("a state/latent normalizer is required")
  if transforms.contract.identity_hash() != contract.identity_hash():
    raise ReplayError("normalizer contract identity differs from active contract")
  if checkpoint_sha256 is not None:
    verify_file_digest(checkpoint_path, checkpoint_sha256, role="diffusion checkpoint")
  if not isinstance(verify_projection_hashes, bool):
    raise ReplayError("verify_projection_hashes must be a boolean")
  if not isinstance(verify_contract_identity, bool):
    raise ReplayError("verify_contract_identity must be a boolean")
  try:
    dimensions = contract_dimensions(contract)
  except ValueError as exc:
    raise ReplayError(
      f"active diffusion contract has no usable model layout: {exc}"
    ) from exc
  model = StateLatentTransformer(DenoiserSettings.from_contract(contract))
  try:
    state = load_checkpoint(
      checkpoint_path,
      model=model,
      expected_contract_identity=(
        contract.identity_hash() if verify_contract_identity else None
      ),
    )
  except CheckpointError as exc:
    # A checkpoint trained for a different state width fails here (contract
    # identity or tensor shapes).  Name the active layout so a 135-D artifact
    # supplied for a reduced 129-D contract fails clearly instead of silently.
    raise ReplayError(
      f"frozen denoiser checkpoint rejected for the {dimensions.label} active "
      f"contract: {exc}"
    ) from exc
  if verify_projection_hashes:
    if isinstance(transforms, StateLatentNormalizer):
      verify_normalization_consistency(state.projection_hashes, transforms)
    else:
      verify_projection_consistency(
        state.projection_hashes, cast(ProjectionBundle, transforms)
      )
  ema_selected, ema_updates = apply_ema_selection(model, state.ema_state)
  model.eval()
  for parameter in model.parameters():
    parameter.requires_grad_(False)
  schedule = DiffusionSchedule.from_contract(contract)
  grid = build_inference_grid(schedule, contract=contract)
  seed = state.config.get("seed")
  return LoadedDenoiser(
    model=model,
    schedule=schedule,
    grid=grid,
    global_step=state.global_step,
    config_hash=state.config_hash,
    contract_identity=state.contract_identity,
    projection_hashes=dict(state.projection_hashes),
    best_validation_loss=state.best_validation_loss,
    selection_split=state.selection_split,
    ema_selected=ema_selected,
    ema_updates=ema_updates,
    fp32_checked=state.fp32_checked,
    config_seed=None if seed is None else int(seed),
  )


def _read_flow_process(path: str | Path) -> LinearNoiseProcess:
  """Read the linear process a flow checkpoint records before model construction.

  ``load_flow_checkpoint`` restores state into a caller-built model, so the
  model's level-embedding width must be known first.  The checkpoint is read
  once for this identity; :func:`load_flow_checkpoint` then revalidates the
  same process mapping, so a tampered process cannot reach the runtime.
  """
  try:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
  except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError) as exc:
    raise ReplayError(f"could not read flow checkpoint {path}: {exc}") from exc
  if (
    not isinstance(payload, Mapping) or payload.get("format") != FLOW_CHECKPOINT_FORMAT
  ):
    raise ReplayError(
      f"flow checkpoint {path} is unsupported or malformed for the flow family"
    )
  try:
    raw = _restore_process(payload)
    return LinearNoiseProcess(int(raw["max_level"]))
  except (FlowCheckpointError, KeyError, TypeError, ValueError) as exc:
    raise ReplayError(
      f"flow checkpoint {path} records no valid linear process: {exc}"
    ) from exc


@dataclass(frozen=True, slots=True)
class LoadedFlowDenoiser:
  """Frozen flow checkpoint with its real process, level grid and provenance.

  Unlike :class:`LoadedDenoiser` this record carries the flow family's
  :class:`LinearNoiseProcess` and :class:`FlowGrid`; it has no diffusion
  schedule, because the flow family defines none.
  """

  model: StateLatentTransformer
  process: LinearNoiseProcess
  grid: FlowGrid
  family: str
  process_identity: str
  data_identity: str
  model_identity_hash: str
  global_step: int
  config_hash: str
  projection_hashes: dict[str, str]
  best_validation_loss: float | None
  selection_split: str | None
  ema_selected: bool
  ema_updates: int | None
  fp32_checked: bool
  config_seed: int | None


def _flow_solver_grid(process: LinearNoiseProcess, solver_steps: int) -> FlowGrid:
  """Build the flow solver grid for an explicit Euler step count.

  The step count is an inference-time choice, not a checkpoint property: every
  step count descends the same trained level scale.
  :data:`DEFAULT_CLOSED_LOOP_SOLVER_STEPS` steps is the closed-loop default.  A
  request larger than the checkpoint process ``max_level`` cannot form a
  strictly descending integer grid, so it fails closed here instead of silently
  reusing the default.
  """
  if isinstance(solver_steps, bool) or not isinstance(solver_steps, int):
    raise ReplayError("flow solver steps must be an integer")
  if solver_steps < 1:
    raise ReplayError("flow solver steps must be a positive integer")
  if solver_steps > process.max_level:
    raise ReplayError(
      f"flow solver steps {solver_steps} exceed the checkpoint process "
      f"max_level {process.max_level}: a strictly descending integer grid "
      "cannot be formed"
    )
  try:
    return FlowGrid.uniform(calls=solver_steps, max_level=process.max_level)
  except FlowMatchingError as exc:
    raise ReplayError(
      f"flow solver steps {solver_steps} do not form a valid grid at "
      f"max_level {process.max_level}: {exc}"
    ) from exc


def load_flow_denoiser(
  checkpoint_path: str | Path,
  *,
  contract: DiffusionContract,
  projection_bundle: StateLatentTransformLike | None = None,
  normalizer: StateLatentTransform | None = None,
  checkpoint_sha256: str | None = None,
  verify_projection_hashes: bool = True,
  verify_contract_identity: bool = True,
  solver_steps: int = DEFAULT_CLOSED_LOOP_SOLVER_STEPS,
) -> LoadedFlowDenoiser:
  """Load the x0 flow denoiser with its exact linear process and level grid.

  Mirrors :func:`load_denoiser` at the runtime boundary but consumes flow
  artifacts: the model is built with the checkpoint's own ``max_level``, the
  state is restored by :func:`load_flow_checkpoint` with the process and data
  identities enforced, the same EMA selection and freezing run, and the same
  projection/normalizer hash consistency is checked against the checkpoint's
  recorded projection hashes.  The returned :class:`LoadedFlowDenoiser` exposes
  the trained :class:`LinearNoiseProcess` and a ``solver_steps``-call
  :class:`FlowGrid` that ends at clean level 0.  No diffusion schedule is built
  or implied.  A diffusion checkpoint is rejected by the flow format check; a
  flow checkpoint is rejected by :func:`load_denoiser` by the diffusion format
  check.

  ``solver_steps`` selects the Euler step count (default
  :data:`DEFAULT_CLOSED_LOOP_SOLVER_STEPS`).  The loaded record always reports
  the grid actually built, so a run never claims the default when another count
  was requested.
  """
  if projection_bundle is not None and normalizer is not None:
    raise ReplayError("pass only one of projection_bundle or normalizer")
  transforms = normalizer if normalizer is not None else projection_bundle
  if transforms is None:
    raise ReplayError("a state/latent normalizer is required")
  if transforms.contract.identity_hash() != contract.identity_hash():
    raise ReplayError("normalizer contract identity differs from active contract")
  if checkpoint_sha256 is not None:
    verify_file_digest(checkpoint_path, checkpoint_sha256, role="flow checkpoint")
  if not isinstance(verify_projection_hashes, bool):
    raise ReplayError("verify_projection_hashes must be a boolean")
  if not isinstance(verify_contract_identity, bool):
    raise ReplayError("verify_contract_identity must be a boolean")
  try:
    dimensions = contract_dimensions(contract)
  except ValueError as exc:
    raise ReplayError(
      f"active diffusion contract has no usable model layout: {exc}"
    ) from exc
  process = _read_flow_process(checkpoint_path)
  model = StateLatentTransformer(
    DenoiserSettings.from_contract(contract, training_k=process.max_level)
  )
  try:
    state = load_flow_checkpoint(
      checkpoint_path,
      model=model,
      expected_process_identity=process.identity_hash(),
      expected_data_identity=(
        contract.identity_hash() if verify_contract_identity else None
      ),
    )
  except FlowCheckpointError as exc:
    raise ReplayError(
      f"frozen flow checkpoint rejected for the {dimensions.label} active "
      f"contract: {exc}"
    ) from exc
  if verify_projection_hashes:
    if isinstance(transforms, StateLatentNormalizer):
      verify_normalization_consistency(state.statistics_identity, transforms)
    else:
      verify_projection_consistency(
        state.statistics_identity, cast(ProjectionBundle, transforms)
      )
  ema_selected, ema_updates = apply_ema_selection(model, state.ema_state)
  _freeze(model)
  grid = _flow_solver_grid(process, solver_steps)
  seed = state.config.get("seed")
  return LoadedFlowDenoiser(
    model=model,
    process=process,
    grid=grid,
    family=state.family,
    process_identity=state.process_identity,
    data_identity=state.data_identity,
    model_identity_hash=state.model_identity_hash,
    global_step=state.global_step,
    config_hash=state.config_hash,
    projection_hashes=dict(state.statistics_identity),
    best_validation_loss=state.best_validation_loss,
    selection_split=state.selection_split,
    ema_selected=ema_selected,
    ema_updates=ema_updates,
    fp32_checked=state.fp32_checked,
    config_seed=None if seed is None else int(seed),
  )


def load_projection_bundle(
  projection_path: str | Path, *, contract: DiffusionContract
) -> ProjectionBundle:
  """Load the historical projection bundle for legacy replay paths."""
  try:
    return ProjectionBundle.load(projection_path, contract=contract)
  except Exception as exc:
    raise ReplayError(f"projection bundle rejected: {exc}") from exc


def load_normalizer_bundle(
  normalizer_path: str | Path, *, contract: DiffusionContract
) -> StateLatentNormalizer:
  """Load state and latent statistics without loading a projection matrix."""
  try:
    return StateLatentNormalizer.load(normalizer_path, contract=contract)
  except (OSError, ProjectionError, ValueError) as exc:
    raise ReplayError(f"normalizer bundle rejected: {exc}") from exc


class VaeDecodeModule(Protocol):
  """The frozen decoder surface the replay needs: decode(latent, conditioning)."""

  def decode(self, latent: Tensor, conditioning: Tensor) -> Tensor: ...


def _freeze(module: Any) -> Any:
  """Switch a loaded model to eval mode and freeze its parameters."""
  module.eval()
  for parameter in module.parameters():
    parameter.requires_grad_(False)
  return module


@dataclass(frozen=True, slots=True)
class LoadedVae:
  """Frozen VAE cohort member plus its verified provenance."""

  model: VaeDecodeModule
  member_id: str
  checkpoint_sha256: str | None
  manifest_sha256: str
  relocated_artifact_roles: tuple[str, ...]
  joint_order: tuple[str, ...]
  conditioning_dim: int
  reference_dim: int
  latent_dim: int
  action_dim: int


def load_vae_decoder(
  vae_checkpoint: str | Path,
  *,
  manifest_path: str | Path,
  repo_root: str | Path,
  member: str,
  contract: DiffusionContract,
  vae_checkpoint_sha256: str | None = None,
  device: str | torch.device = "cpu",
) -> LoadedVae:
  """Load a gravity-mode VAE member without requiring artifact digests.

  The cohort loader still validates the saved model schema and the requested
  member.  File digests are optional reproduction metadata rather than a gate.
  """
  if vae_checkpoint_sha256 is not None:
    verify_file_digest(
      vae_checkpoint, vae_checkpoint_sha256, role="frozen VAE checkpoint"
    )
  if vae_checkpoint_sha256 is not None and vae_checkpoint_sha256 != contract.vae_sha256:
    raise ReplayError(
      "frozen VAE checkpoint digest does not match the active contract's "
      f"pinned VAE identity {contract.vae_sha256}"
    )
  from mjlab.tasks.tracking.distillation.checkpoint import (
    load_cohort_member_inference,
  )
  from mjlab.tasks.tracking.distillation.config import load_manifest, resolve_cohort

  try:
    manifest_sha256 = sha256_file(manifest_path)
    manifest = load_manifest(manifest_path, repo_root=repo_root)
    cohort = resolve_cohort(manifest)
    inference = load_cohort_member_inference(
      vae_checkpoint,
      cohort,
      member,
      device=device,
    )
  except Exception as exc:
    raise ReplayError(
      f"frozen VAE cohort member {member!r} could not be loaded from "
      f"manifest {manifest_path}: {exc}"
    ) from exc
  model = inference.model
  schema = model.schema
  settings = model.settings
  preview = float(cohort.lookahead_s) > 0.0
  expected_reference_dim = REFERENCE_DIM_WITH_PREVIEW if preview else REFERENCE_DIM
  mode = getattr(schema, "mode", None)
  mode_name = getattr(mode, "value", mode)
  if mode_name is not None and mode_name != "gravity":
    raise ReplayError(
      "frozen VAE does not implement a supported gravity decoder mode: "
      f"got {mode_name!r}"
    )
  if (
    schema.reference_dim not in (REFERENCE_DIM, REFERENCE_DIM_WITH_PREVIEW)
    or schema.conditioning_dim != CONDITIONING_DIMS["gravity"]
    or settings.latent_dim != LATENT_DIM
    or schema.action_dim != ACTION_DIM
  ):
    raise ReplayError(
      "frozen VAE does not implement a supported gravity schema "
      "(reference 68 or 130 / conditioning 99 / latent 32 / action 31): got "
      f"{schema.reference_dim}/{schema.conditioning_dim}/"
      f"{settings.latent_dim}/{schema.action_dim}"
    )
  if schema.reference_dim != expected_reference_dim:
    raise ReplayError(
      "frozen VAE reference width disagrees with the live cohort layout: "
      f"model {schema.reference_dim}, cohort {expected_reference_dim} "
      f"(lookahead_s={float(cohort.lookahead_s)!r})"
    )
  _freeze(model)
  return LoadedVae(
    model=model,
    member_id=member,
    checkpoint_sha256=vae_checkpoint_sha256,
    manifest_sha256=manifest_sha256,
    relocated_artifact_roles=tuple(inference.relocated_artifact_roles),
    joint_order=tuple(inference.cohort.common.joint_names),
    conditioning_dim=int(schema.conditioning_dim),
    reference_dim=int(schema.reference_dim),
    latent_dim=int(settings.latent_dim),
    action_dim=int(schema.action_dim),
  )


@dataclass(frozen=True, slots=True)
class RecordedHistory:
  """Bounded measured rows used as the executed-pair history source."""

  states: tuple[WorldState, ...]
  latents: np.ndarray
  ticks: tuple[int, ...]
  run_id: str
  episode_id: str
  shard_sha256: str
  row_indexes: tuple[int, ...]


def load_recorded_history(
  shard_path: str | Path,
  *,
  row_offset: int,
  samples: int,
  contract: DiffusionContract = DEFAULT_CONTRACT,
) -> RecordedHistory:
  """Read ``past_steps + samples`` clean recorded rows from a D1 shard.

  Rows are raw recorded store rows (world states plus the executed VAE
  latent) read under the recorded-row layout.  Rows with any reset, teleport
  or reference-boundary flag, a missing executed latent, an unqualified
  segment, a non-VAE controller, or non-increasing ticks are rejected.
  """
  if row_offset < 0:
    raise ReplayError("history row offset must be non-negative")
  needed = contract.past_steps + samples
  keys = (
    "root_position",
    "root_quaternion_wxyz",
    "root_linear_velocity",
    "root_angular_velocity",
    "body_positions",
    "body_linear_velocities",
    "latent",
    "latent_present",
    "metadata",
  )
  try:
    with np.load(shard_path, allow_pickle=False) as data:
      arrays = {key: np.asarray(data[key]) for key in keys}
  except (OSError, KeyError, ValueError) as exc:
    raise ReplayError(
      f"recorded history shard {shard_path} is unreadable: {exc}"
    ) from exc
  rows = int(arrays["root_position"].shape[0])
  if row_offset + needed > rows:
    raise ReplayError(
      f"history shard {shard_path} has {rows} rows; rows "
      f"[{row_offset}, {row_offset + needed}) are required"
    )
  present = np.asarray(arrays["latent_present"][row_offset : row_offset + needed])
  if not bool((present == 1).all()):
    raise ReplayError(
      "history rows must all carry an executed VAE latent (latent_present == 1)"
    )
  metadata: list[dict[str, Any]] = []
  for index in range(row_offset, row_offset + needed):
    try:
      item = json.loads(str(arrays["metadata"][index]))
    except json.JSONDecodeError as exc:
      raise ReplayError(f"history row {index} metadata is not valid JSON") from exc
    if not isinstance(item, dict):
      raise ReplayError(f"history row {index} metadata is not a mapping")
    if item.get("controller") != "vae":
      raise ReplayError(
        f"history row {index} controller is {item.get('controller')!r}, expected 'vae'"
      )
    if item.get("reset") or item.get("teleport") or item.get("reference_boundary"):
      raise ReplayError(
        f"history row {index} is not a clean measured row "
        "(reset/teleport/reference_boundary flag set)"
      )
    if item.get("segment_qualified") is not True:
      raise ReplayError(f"history row {index} segment is not qualified")
    metadata.append(item)
  ticks = tuple(int(item["tick"]) for item in metadata)
  # Pairwise check: a 1-tick history must also pass, hence no strict zip.
  for earlier, later in zip(ticks, ticks[1:], strict=False):
    if earlier >= later:
      raise ReplayError(f"history ticks must strictly increase, got {ticks}")
  states = tuple(
    WorldState(
      arrays["root_position"][index],
      arrays["root_quaternion_wxyz"][index],
      arrays["root_linear_velocity"][index],
      arrays["root_angular_velocity"][index],
      arrays["body_positions"][index],
      arrays["body_linear_velocities"][index],
    )
    for index in range(row_offset, row_offset + needed)
  )
  latents = np.asarray(
    arrays["latent"][row_offset : row_offset + needed], dtype=np.float64
  )
  if (
    latents.shape != (needed, contract.latent_dimension)
    or not np.isfinite(latents).all()
  ):
    raise ReplayError(
      "recorded latents must be finite with shape "
      f"[{needed}, {contract.latent_dimension}]"
    )
  return RecordedHistory(
    states=states,
    latents=latents,
    ticks=ticks,
    run_id=str(metadata[0].get("run_id", "")),
    episode_id=str(metadata[0].get("episode_id", "")),
    shard_sha256=sha256_file(shard_path),
    row_indexes=tuple(range(row_offset, row_offset + needed)),
  )


def synthetic_conditioning(step_index: int, *, dimension: int = 99) -> np.ndarray:
  """Deterministic labeled synthetic decoder-conditioning fixture.

  D1 recorded rows do not contain the measured 99-D gravity conditioning and
  reconstructing it from body poses is forbidden.  The seam-vs-direct parity
  comparison does not depend on the conditioning values, so a fixed finite
  sinusoid is used and every report labels it ``synthetic-fixture``.
  """
  indexes = np.arange(dimension, dtype=np.float64)
  values = _SYNTHETIC_CONDITIONING_AMPLITUDE * np.sin(
    _SYNTHETIC_CONDITIONING_WAVENUMBER * indexes + float(step_index)
  )
  if values.shape != (dimension,) or not np.isfinite(values).all():
    raise ReplayError("synthetic conditioning fixture is malformed")
  return values


def compare_samples(
  seam_latent: np.ndarray,
  seam_action: np.ndarray,
  direct_latent: np.ndarray,
  direct_action: np.ndarray,
  *,
  tolerance: float = _SEAM_VS_DIRECT_TOLERANCE,
) -> dict[str, Any]:
  """Compare seam and direct latent/action vectors."""
  latent_diff = float(np.max(np.abs(seam_latent - direct_latent)))
  action_diff = float(np.max(np.abs(seam_action - direct_action)))
  worst = max(latent_diff, action_diff)
  return {
    "latent_max_abs_diff": latent_diff,
    "action_max_abs_diff": action_diff,
    "tolerance": float(tolerance),
    "passed": bool(worst <= tolerance),
  }


def _decode_vector(
  decode: Any, latent_raw: np.ndarray, conditioning: np.ndarray
) -> np.ndarray:
  """Decode one raw latent with the frozen decoder and validate the output."""
  latent_tensor = torch.from_numpy(np.ascontiguousarray(latent_raw, dtype=np.float32))[
    None
  ]
  conditioning_tensor = torch.from_numpy(
    np.ascontiguousarray(conditioning, dtype=np.float32)
  )[None]
  action = decode(latent_tensor, conditioning_tensor)
  if not isinstance(action, Tensor):
    raise ReplayError("decoder must return a torch tensor")
  vector = action.detach().to(device="cpu", dtype=torch.float32).numpy().reshape(-1)
  if (
    vector.shape != (DEFAULT_CONTRACT.action_dimension,)
    or not np.isfinite(vector).all()
  ):
    raise ReplayError("decoder action must be finite with shape [31]")
  return vector


def _vector_summary(values: np.ndarray) -> dict[str, float]:
  return {
    "min": float(np.min(values)),
    "max": float(np.max(values)),
    "mean_abs": float(np.mean(np.abs(values))),
    "max_abs": float(np.max(np.abs(values))),
  }


def _direct_sample(
  *,
  plan: ContinuationPlan,
  denoiser: Any,
  schedule: DiffusionSchedule,
  grid: InferenceGrid,
  projection: ProjectionBundle,
  contract: DiffusionContract,
  generator_state: Any,
  decode: Any,
  conditioning: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
  """Run the existing sampler + denormalize + frozen-decoder path directly.

  Starts from the exact generator state captured before the seam step and
  consumes the same draws, so any difference is a real seam discrepancy.
  """
  generator = torch.Generator()
  generator.set_state(generator_state)
  sample, diagnostics = sample_trajectory(
    denoiser,
    schedule=schedule,
    conditions=SamplerConditions(plan.clean_tokens, plan.clean_mask),
    config=SamplerConfig(grid),
    generator=generator,
  )
  normalized = (
    sample[contract.current_index, contract.projected_state_dimension :]
    .detach()
    .to(device="cpu", dtype=torch.float32)
    .numpy()
    .astype(np.float64)
  )
  latent = projection.denormalize_latent(normalized)
  action = _decode_vector(decode, latent, conditioning)
  return latent, action, int(diagnostics.denoiser_calls)


@dataclass(frozen=True, slots=True)
class ReplayRunResult:
  """One bounded window's seam and direct-path records."""

  label: str
  seed: int
  steps: tuple[dict[str, Any], ...]


def run_seam_replay(
  *,
  denoiser: Any,
  decode: Any,
  schedule: DiffusionSchedule,
  grid: InferenceGrid,
  projection: ProjectionBundle,
  history_states: Sequence[WorldState],
  history_latents: np.ndarray,
  ticks: Sequence[int],
  contract: DiffusionContract = DEFAULT_CONTRACT,
  seeds: tuple[int, int] = (0, 1),
  samples: int = 1,
  tolerance: float = _SEAM_VS_DIRECT_TOLERANCE,
) -> dict[str, Any]:
  """Compare the seam against the direct path for two seeds plus a replay.

  Each run uses a fresh generator seeded with the given seed and the same
  recorded 8-pair history.  After every seam step the executed latent is
  recorded into that run's own continuation history (the documented
  lifecycle), and a direct sampler path from the pre-step generator snapshot
  must reproduce the same latent and action within ``tolerance``.  The third
  run repeats the first seed and must reproduce window zero's latents and
  actions exactly; no recorded history is mutated across runs.
  """
  past = contract.past_steps
  if len(seeds) != 2 or seeds[0] == seeds[1]:
    raise ReplayError("exactly two independent window seeds are required")
  if len(history_states) != past + samples:
    raise ReplayError(
      f"expected {past + samples} recorded states, got {len(history_states)}"
    )
  if history_latents.shape != (past + samples, contract.latent_dimension):
    raise ReplayError(
      "recorded latents must have shape "
      f"[{past + samples}, {contract.latent_dimension}]"
    )
  if len(ticks) != past + samples:
    raise ReplayError(f"expected {past + samples} ticks, got {len(ticks)}")

  decode_calls = 0

  def counted_decode(latent_tensor: Tensor, conditioning_tensor: Tensor) -> Tensor:
    nonlocal decode_calls
    decode_calls += 1
    return decode(latent_tensor, conditioning_tensor)

  def seam_decode_and_count(
    latent_tensor: Tensor, conditioning_tensor: Tensor
  ) -> Tensor:
    return counted_decode(latent_tensor, conditioning_tensor)

  plan_report: dict[str, Any] | None = None
  runs: list[ReplayRunResult] = []
  all_seeds = (int(seeds[0]), int(seeds[1]), int(seeds[0]))
  # Executed latents/actions per run, kept only for the same-seed comparison.
  executed: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}

  for run_index, seed in enumerate(all_seeds):
    label = "window" if run_index < 2 else "same-seed-replay"
    generator = torch.Generator()
    generator.manual_seed(seed)
    continuation = ReferenceFreeContinuation(
      denoiser=denoiser,
      decode=seam_decode_and_count,
      schedule=schedule,
      grid=grid,
      projection=cast(StateLatentTransform, projection),
      contract=contract,
      device="cpu",
      generator=generator,
    )
    history = continuation.history
    for index in range(past):
      history.push(history_states[index], history_latents[index])
    steps: list[dict[str, Any]] = []
    executed[run_index] = []
    for step_index in range(samples):
      current_state = history_states[past + step_index]
      tick = int(ticks[past + step_index])
      conditioning = synthetic_conditioning(step_index)
      generator_state = generator.get_state()
      seam = continuation.step(current_state, conditioning, tick=tick)
      plan = continuation.plan(current_state, tick=tick)
      assert isinstance(plan, ContinuationPlan)
      if run_index == 0 and step_index == 0:
        expected_mask = build_clean_mask(
          current_index=contract.current_index,
          state_dimension=contract.projected_state_dimension,
          token_dimension=contract.token_dimension,
          device=torch.device("cpu"),
        )
        plan_report = {
          "shape": list(plan.clean_tokens.shape),
          "mask_shape": list(plan.clean_mask.shape),
          "mask_known_entries": int(plan.clean_mask.sum().item()),
          "mask_matches_contract": bool(torch.equal(plan.clean_mask, expected_mask)),
          "tokens_finite": bool(torch.isfinite(plan.clean_tokens).all().item()),
        }
      direct_latent, direct_action, direct_calls = _direct_sample(
        plan=plan,
        denoiser=denoiser,
        schedule=schedule,
        grid=grid,
        projection=projection,
        contract=contract,
        generator_state=generator_state,
        decode=counted_decode,
        conditioning=conditioning,
      )
      parity = compare_samples(
        seam.executed_latent,
        seam.action,
        direct_latent,
        direct_action,
        tolerance=tolerance,
      )
      if not parity["passed"]:
        raise ReplayError(
          f"seam-vs-direct parity failed for seed {seed} step {step_index}: {parity}"
        )
      steps.append(
        {
          "seed": seed,
          "step_index": step_index,
          "tick": tick,
          "denoiser_calls": int(seam.denoiser_calls),
          "direct_path_denoiser_calls": direct_calls,
          "final_index": int(seam.final_index),
          "parity": parity,
          "executed_latent_summary": _vector_summary(seam.executed_latent),
          "action_summary": _vector_summary(seam.action),
        }
      )
      executed[run_index].append((seam.executed_latent, seam.action))
      continuation.record_executed(current_state, seam.executed_latent)
    runs.append(ReplayRunResult(label=label, seed=seed, steps=tuple(steps)))

  first_latents = np.stack([pair[0] for pair in executed[0]])
  first_actions = np.stack([pair[1] for pair in executed[0]])
  replay_latents = np.stack([pair[0] for pair in executed[2]])
  replay_actions = np.stack([pair[1] for pair in executed[2]])
  determinism = {
    "latent_max_abs_diff": float(np.max(np.abs(first_latents - replay_latents))),
    "action_max_abs_diff": float(np.max(np.abs(first_actions - replay_actions))),
    "seed_repeat": True,
    "passed": bool(
      np.array_equal(first_latents, replay_latents)
      and np.array_equal(first_actions, replay_actions)
    ),
  }
  if not determinism["passed"]:
    raise ReplayError(f"same-seed replay is not deterministic: {determinism}")
  return {
    "closed_loop": False,
    "conditioning": {
      "kind": "synthetic-fixture",
      "dimension": 99,
      "formula": ("0.3 * sin(0.05 * index + step_index); deterministic finite fixture"),
      "reason": (
        "D1 recorded rows carry no measured 99-D gravity conditioning; the "
        "seam-vs-direct parity comparison is independent of the values"
      ),
    },
    "history_source": {
      "kind": "recorded-d1-pilot-rows",
      "states_and_latents": "measured recorded states with executed VAE latents",
    },
    "runs": [
      {"run": run.label, "seed": run.seed, "steps": list(run.steps)} for run in runs
    ],
    "determinism": determinism,
    "plan": plan_report,
    "decode_calls": decode_calls,
  }


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Bounded offline real-model D3 seam replay: EMA-selected frozen "
      "denoiser + frozen VAE decoder, seam-vs-direct parity, two seeds plus "
      "a same-seed replay. Not a closed-loop evaluation."
    )
  )
  parser.add_argument(
    "--checkpoint", type=Path, required=True, help="frozen D2 diffusion checkpoint"
  )
  parser.add_argument(
    "--checkpoint-sha256", required=True, help="expected diffusion checkpoint digest"
  )
  parser.add_argument(
    "--contract", type=Path, required=True, help="active no-mixture contract YAML"
  )
  parser.add_argument(
    "--projection", type=Path, required=True, help="active projection bundle npz"
  )
  parser.add_argument(
    "--projection-sha256",
    required=True,
    help="expected projection bundle digest (sha256 of the exact .npz file)",
  )
  parser.add_argument(
    "--vae-checkpoint", type=Path, required=True, help="frozen VAE cohort checkpoint"
  )
  parser.add_argument(
    "--vae-checkpoint-sha256", required=True, help="expected frozen VAE digest"
  )
  parser.add_argument(
    "--vae-manifest", type=Path, required=True, help="distillation cohort manifest"
  )
  parser.add_argument(
    "--vae-member", required=True, help="cohort member id (e.g. tennis_000)"
  )
  parser.add_argument(
    "--vae-repo-root",
    type=Path,
    required=True,
    help="root against which the manifest's relative artifact paths resolve",
  )
  parser.add_argument(
    "--history-shard",
    type=Path,
    required=True,
    help="recorded D1 store shard npz supplying measured state/executed latent rows",
  )
  parser.add_argument(
    "--history-row-offset",
    type=int,
    default=0,
    help="first recorded row index to use",
  )
  parser.add_argument("--output", type=Path, required=True, help="report JSON path")
  parser.add_argument("--seed-0", type=int, default=0, help="first window seed")
  parser.add_argument("--seed-1", type=int, default=1, help="second window seed")
  parser.add_argument(
    "--samples",
    type=int,
    default=1,
    help="continuation steps per window (bounded, at most 4)",
  )
  parser.add_argument("--cpu-threads", type=int, default=1, help="torch CPU thread cap")
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  if (
    args.seed_0 < 0
    or args.seed_1 < 0
    or args.seed_0 > _MAX_SEED
    or args.seed_1 > _MAX_SEED
  ):
    raise ReplayError(f"seeds must be within [0, {_MAX_SEED}]")
  if args.samples < 1 or args.samples > _MAX_SAMPLES:
    raise ReplayError(f"samples must be within [1, {_MAX_SAMPLES}]")
  if args.cpu_threads < 1 or args.cpu_threads > _MAX_THREADS:
    raise ReplayError(f"cpu_threads must be within [1, {_MAX_THREADS}]")
  torch.set_num_threads(args.cpu_threads)

  contract = load_active_contract(args.contract)
  if contract.projection_rows != 0:
    raise ReplayError(
      "the active contract must be a no-mixture state contract; legacy "
      "projected contracts are not accepted"
    )
  verify_file_digest(args.projection, args.projection_sha256, role="projection bundle")
  projection = load_projection_bundle(args.projection, contract=contract)
  denoiser = load_denoiser(
    args.checkpoint,
    contract=contract,
    checkpoint_sha256=args.checkpoint_sha256,
    projection_bundle=cast(StateLatentTransform, projection),
  )
  vae = load_vae_decoder(
    args.vae_checkpoint,
    manifest_path=args.vae_manifest,
    repo_root=args.vae_repo_root,
    member=args.vae_member,
    contract=contract,
    vae_checkpoint_sha256=args.vae_checkpoint_sha256,
  )
  history = load_recorded_history(
    args.history_shard,
    row_offset=args.history_row_offset,
    samples=args.samples,
    contract=contract,
  )

  def frozen_decode(latent: Tensor, conditioning: Tensor) -> Tensor:
    decoder = vae.model.decode
    with torch.no_grad():
      return decoder(latent, conditioning)

  report = run_seam_replay(
    denoiser=denoiser.model,
    decode=frozen_decode,
    schedule=denoiser.schedule,
    grid=denoiser.grid,
    projection=projection,
    history_states=history.states,
    history_latents=history.latents,
    ticks=history.ticks,
    contract=contract,
    seeds=(args.seed_0, args.seed_1),
    samples=args.samples,
  )
  report["provenance"] = {
    "checkpoint": {
      "path": str(args.checkpoint),
      "sha256": args.checkpoint_sha256,
      "global_step": denoiser.global_step,
      "config_hash": denoiser.config_hash,
      "contract_identity": denoiser.contract_identity,
      "selection_split": denoiser.selection_split,
      "best_validation_loss": denoiser.best_validation_loss,
      "ema_selected": denoiser.ema_selected,
      "ema_updates": denoiser.ema_updates,
      "ema_convention": (
        "evaluation runs on the EMA shadow (diffusion/trainer.py); the "
        "checkpoint EMA shadow was copied into the model before replay"
        if denoiser.ema_selected
        else "checkpoint carries no EMA shadow; raw model weights were used"
      ),
      "fp32_checked": denoiser.fp32_checked,
      "config_seed": denoiser.config_seed,
    },
    "projection": {
      "path": str(args.projection),
      "sha256": args.projection_sha256,
      "statistics_sha256": projection.statistics_sha256,
      "checkpoint_projection_hashes": denoiser.projection_hashes,
    },
    "contract": {
      "path": str(args.contract),
      "schema_version": contract.schema_version,
      "contract_sha256": contract.contract_sha256,
      "identity_hash": contract.identity_hash(),
    },
    "vae": {
      "checkpoint": str(args.vae_checkpoint),
      "checkpoint_sha256": vae.checkpoint_sha256,
      "manifest": str(args.vae_manifest),
      "manifest_sha256": vae.manifest_sha256,
      "member": vae.member_id,
      "relocated_artifact_roles": list(vae.relocated_artifact_roles),
      "joint_order": list(vae.joint_order),
      "reference_dim": vae.reference_dim,
      "conditioning_dim": vae.conditioning_dim,
      "latent_dim": vae.latent_dim,
      "action_dim": vae.action_dim,
    },
    "history": {
      "shard": str(args.history_shard),
      "sha256": history.shard_sha256,
      "row_offset": args.history_row_offset,
      "row_indexes": list(history.row_indexes),
      "run_id": history.run_id,
      "episode_id": history.episode_id,
      "ticks": list(history.ticks),
    },
    "sampling": {
      "grid": denoiser.grid.as_dict(),
      "eta": 0.0,
      "guidance_strength": 0.0,
      "start_level": 1000,
      "seeds": [args.seed_0, args.seed_1],
      "same_seed_replay": args.seed_0,
      "samples_per_window": args.samples,
      "tolerance": _SEAM_VS_DIRECT_TOLERANCE,
      "cpu_threads": args.cpu_threads,
    },
    "claims": {
      "closed_loop": False,
      "real_models": True,
      "synthetic_conditioning_fixture": True,
      "generated_latents_executed": False,
      "latents_reported_as": "supplied to the frozen decoder as data",
    },
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  temporary = args.output.with_suffix(".json.tmp")
  temporary.write_text(json.dumps(report, indent=1) + "\n")
  temporary.replace(args.output)


__all__ = [
  "DEFAULT_CLOSED_LOOP_SOLVER_STEPS",
  "LoadedDenoiser",
  "LoadedFlowDenoiser",
  "LoadedVae",
  "RecordedHistory",
  "ReplayError",
  "ReplayRunResult",
  "apply_ema_selection",
  "load_active_contract",
  "load_denoiser",
  "load_flow_denoiser",
  "load_normalizer_bundle",
  "load_projection_bundle",
  "load_recorded_history",
  "load_vae_decoder",
  "run_seam_replay",
  "sha256_file",
  "synthetic_conditioning",
  "verify_file_digest",
  "verify_normalization_consistency",
  "verify_projection_consistency",
]

if __name__ == "__main__":
  main()
