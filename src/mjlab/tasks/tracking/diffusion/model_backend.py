"""Concrete D5 model-backed planner and runtime assembly.

This module connects the accepted CPU D5 facade to the existing frozen-model
loaders.  It does not duplicate diffusion sampling, normalization or VAE
math.  The planner worker receives only paths and scalar settings, creates its
own denoiser inside the spawned process, and returns raw 32-D latent rows.
The owner process loads the VAE decoder and decodes only the selected row with
current 99-D conditioning.

This is a paper-reproduction path.  Artifact SHA arguments are optional.  The
loader still rejects missing files, malformed checkpoints, incompatible
schemas and non-finite or wrong-shaped tensors.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor

from .async_latent_runtime import AsyncLatentRuntime
from .contract import DiffusionContract
from .latent_planning import LatentPlanner
from .planner_executor import BoundedPlannerExecutor
from .projection import ProjectionBundle
from .replay_integration import (
  LoadedVae,
  load_active_contract,
  load_denoiser,
  load_projection_bundle,
  load_vae_decoder,
)
from .scheduler import Plan, PlannerRequest


class ModelBackendError(ValueError):
  """A model-backed D5 input or compatibility check is invalid."""


@dataclass(frozen=True, slots=True)
class ModelBackendSpec:
  """Explicit paths and scalar settings needed to build one D5 backend.

  Paths are intentionally plain data so this value can cross the spawn
  boundary.  A missing path fails during backend construction; no standing,
  zero-action or synthetic model is substituted.
  """

  contract_path: str | Path
  diffusion_checkpoint: str | Path
  projection_bundle: str | Path
  vae_checkpoint: str | Path
  vae_manifest: str | Path
  repo_root: str | Path
  vae_member: str
  planner_device: str = "cpu"
  decoder_device: str = "cpu"
  model_identity: str | None = None
  vae_identity: str | None = None
  verify_projection_hashes: bool = False

  def __post_init__(self) -> None:
    for name in (
      "contract_path",
      "diffusion_checkpoint",
      "projection_bundle",
      "vae_checkpoint",
      "vae_manifest",
      "repo_root",
    ):
      value = getattr(self, name)
      if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ModelBackendError(f"{name} must be a non-empty path")
      object.__setattr__(self, name, str(value))
    if not isinstance(self.vae_member, str) or not self.vae_member.strip():
      raise ModelBackendError("vae_member must be a non-empty string")
    for name in ("planner_device", "decoder_device"):
      value = getattr(self, name)
      if not isinstance(value, str) or not value.strip():
        raise ModelBackendError(f"{name} must be a non-empty device string")
    if self.model_identity is not None and (
      not isinstance(self.model_identity, str) or not self.model_identity.strip()
    ):
      raise ModelBackendError("model_identity must be a non-empty string or None")
    if self.vae_identity is not None and (
      not isinstance(self.vae_identity, str) or not self.vae_identity.strip()
    ):
      raise ModelBackendError("vae_identity must be a non-empty string or None")
    if not isinstance(self.verify_projection_hashes, bool):
      raise ModelBackendError("verify_projection_hashes must be a boolean")

  @property
  def resolved_model_identity(self) -> str:
    """Stable local identity without requiring a checkpoint digest."""

    if self.model_identity is not None:
      return self.model_identity
    return f"diffusion:{Path(self.diffusion_checkpoint)}"

  @property
  def resolved_vae_identity(self) -> str:
    """Stable local identity for the selected decoder member."""

    if self.vae_identity is not None:
      return self.vae_identity
    return f"vae:{Path(self.vae_checkpoint)}:{self.vae_member}"

  def validate_paths(self) -> None:
    """Reject missing explicit inputs before a worker or decoder is started."""

    for name in (
      "contract_path",
      "diffusion_checkpoint",
      "projection_bundle",
      "vae_checkpoint",
      "vae_manifest",
    ):
      path = Path(getattr(self, name))
      if not path.is_file():
        raise ModelBackendError(f"{name} does not exist as a file: {path}")
    root = Path(self.repo_root)
    if not root.is_dir():
      raise ModelBackendError(f"repo_root does not exist as a directory: {root}")


@dataclass(frozen=True, slots=True)
class ModelBackend:
  """Loaded owner-side VAE plus the worker-side planner factory."""

  spec: ModelBackendSpec
  contract: DiffusionContract
  projection: ProjectionBundle
  vae: LoadedVae

  @property
  def model_identity(self) -> str:
    return self.spec.resolved_model_identity

  @property
  def vae_identity(self) -> str:
    return self.spec.resolved_vae_identity

  @property
  def decoder(self) -> Callable[[Tensor, Tensor], Tensor]:
    """Return the owner-process decoder callback used by D5."""

    return self.vae.model.decode

  def planner_factory(self) -> "ModelPlannerFactory":
    """Return a pickleable factory that initializes the planner in the worker."""

    return ModelPlannerFactory(self.spec)


@dataclass(frozen=True, slots=True)
class ModelPlannerFactory:
  """Spawn-safe factory for the concrete denoiser planner."""

  spec: ModelBackendSpec

  def __call__(self) -> Callable[[PlannerRequest], Plan]:
    self.spec.validate_paths()
    contract = load_active_contract(self.spec.contract_path)
    projection = load_projection_bundle(
      self.spec.projection_bundle,
      contract=contract,
    )
    loaded = load_denoiser(
      self.spec.diffusion_checkpoint,
      contract=contract,
      projection_bundle=projection,
      checkpoint_sha256=None,
      verify_projection_hashes=self.spec.verify_projection_hashes,
      verify_contract_identity=False,
    )
    try:
      device = torch.device(self.spec.planner_device)
      model = loaded.model.to(device=device)
    except (RuntimeError, ValueError) as exc:
      raise ModelBackendError(
        f"planner model could not move to {self.spec.planner_device!r}"
      ) from exc
    planner = LatentPlanner(
      denoiser=model,
      schedule=loaded.schedule,
      grid=loaded.grid,
      projection=projection,
      contract=contract,
      device=device,
      vae_identity=self.spec.resolved_vae_identity,
      model_identity=self.spec.resolved_model_identity,
    )
    return planner.process


def load_model_backend(spec: ModelBackendSpec) -> ModelBackend:
  """Load the owner-side contract, projection bundle and VAE decoder."""

  if not isinstance(spec, ModelBackendSpec):
    raise ModelBackendError("spec must be a ModelBackendSpec")
  spec.validate_paths()
  contract = load_active_contract(spec.contract_path)
  projection = load_projection_bundle(spec.projection_bundle, contract=contract)
  try:
    decoder_device = torch.device(spec.decoder_device)
    torch.empty(0, device=decoder_device)
  except (RuntimeError, ValueError) as exc:
    raise ModelBackendError(f"invalid decoder device {spec.decoder_device!r}") from exc
  vae = load_vae_decoder(
    spec.vae_checkpoint,
    manifest_path=spec.vae_manifest,
    repo_root=spec.repo_root,
    member=spec.vae_member,
    contract=contract,
    vae_checkpoint_sha256=None,
    device=decoder_device,
  )
  return ModelBackend(spec, contract, projection, vae)


def build_model_backed_runtime(
  spec: ModelBackendSpec,
  *,
  generator: torch.Generator,
  clock: Callable[[], float] = time.monotonic,
  max_fallback_ticks: int = 4,
  max_source_age_seconds: float = 0.5,
  max_future_skew_seconds: float = 0.0,
  expected_source_frame: int | None = None,
) -> AsyncLatentRuntime:
  """Construct the concrete D5 runtime from explicit model paths.

  The VAE is loaded in the owner process.  The denoiser and projection bundle
  are loaded inside the executor's spawned worker by ``ModelPlannerFactory``.
  """

  backend = load_model_backend(spec)
  executor = BoundedPlannerExecutor(backend.planner_factory())
  try:
    return AsyncLatentRuntime(
      executor,
      projection=backend.projection,
      decode=backend.decoder,
      generator=generator,
      contract=backend.contract,
      max_fallback_ticks=max_fallback_ticks,
      max_source_age_seconds=max_source_age_seconds,
      max_future_skew_seconds=max_future_skew_seconds,
      clock=clock,
      device=spec.decoder_device,
      vae_identity=backend.vae_identity,
      model_identity=backend.model_identity,
      expected_source_frame=expected_source_frame,
    )
  except BaseException:
    executor.shutdown()
    raise


__all__ = [
  "ModelBackend",
  "ModelBackendError",
  "ModelBackendSpec",
  "ModelPlannerFactory",
  "build_model_backed_runtime",
  "load_model_backend",
]
