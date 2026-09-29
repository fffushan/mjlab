"""Frozen X2 state-latent diffusion data contract.

This module intentionally contains no simulator or model code.  It is the
small, serialisable identity shared by collection, shard storage and offline
preprocessing.  The defaults mirror the frozen D0 contract; callers may only
construct a contract from a mapping when all identity fields are present.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, cast


class ContractError(ValueError):
  """A contract or artifact manifest is incomplete or incompatible."""


_SCHEMA = "mjlab-x2-state-latent-d0-v1"
_D0_CONTRACT_SHA = "42694cdd3acd72afd2d234a85ca29e9c7a646d64df831021c8dd43e55f3f73e7"
_VAE_SHA = "69891dffb59af31539388e40041efe30242a2028f2876b746d1f7c5ef44ac117"
_RECOVERY_SHA = "caf17c38f23ad180829a230a9a3259f14eedd3860dd90a046b8be92d3d047681"
_XML_SHA = "ae0dcbceda3ef74e029e3bf4a9ea24ea8cc58ab44f1518e635653fc79f928125"
_PHYSICAL_FALL_SHA = "3c0d64bdc9aa69d8fd372f343e47ab6f9da59a77d2efd8e3d860eff2c1ebf68a"
_VAE_TRACKING_REJECTION_SHA = (
  "5f88576b5995408a744cde451c7e43982d2f03284cfeebb8856969ca3dc76019"
)


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
  """Content identity recorded in a dataset manifest."""

  role: str
  path: str
  sha256: str

  def __post_init__(self) -> None:
    if not self.role or not self.path:
      raise ContractError("artifact role and path must be non-empty")
    if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
      raise ContractError(f"{self.role} sha256 must be a lowercase hex digest")

  def as_dict(self) -> dict[str, str]:
    return {"role": self.role, "path": self.path, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class DiffusionContract:
  """Strict D1 subset of the frozen X2 diffusion contract."""

  schema_version: str = _SCHEMA
  contract_sha256: str | None = None
  control_hz: int = 50
  period_seconds: float = 0.02
  state_dimension: int = 135
  body_count: int = 20
  latent_dimension: int = 32
  action_dimension: int = 31
  past_steps: int = 8
  current_index: int = 8
  future_steps: int = 32
  window_steps: int = 41
  projected_state_dimension: int = 199
  token_dimension: int = 231
  split_seed: int = 42
  split_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)
  std_floor: float = 1.0e-6
  projection_seed: int = 42
  projection_rows: int = 64
  projection_rcond: float = 1.0e-12
  vae_checkpoint: str = "logs/distillation/mixed-10k/checkpoint-final.pt"
  vae_sha256: str = _VAE_SHA
  recovery_onnx: str = (
    "logs/rsl_rl/agibot_x2_velocity/2026-09-26_02-40-45_x2-tennis-recovery-n25/"
    "2026-09-26_02-40-45_x2-tennis-recovery-n25.onnx"
  )
  recovery_sha256: str = _RECOVERY_SHA
  state_xml: str = "src/mjlab/asset_zoo/robots/agibot_x2/xmls/x2_ultra.xml"
  state_xml_sha256: str = _XML_SHA
  physical_fall_source: str = "src/mjlab/envs/mdp/terminations.py"
  physical_fall_source_sha256: str = _PHYSICAL_FALL_SHA
  vae_tracking_rejection_source: str = "src/mjlab/tasks/tracking/mdp/terminations.py"
  vae_tracking_rejection_source_sha256: str = _VAE_TRACKING_REJECTION_SHA
  body_names: tuple[str, ...] = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "left_wrist_pitch_link",
    "left_wrist_roll_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
    "right_wrist_pitch_link",
    "right_wrist_roll_link",
    "head_yaw_link",
    "head_pitch_link",
  )

  def __post_init__(self) -> None:
    if self.schema_version != _SCHEMA:
      raise ContractError(f"unsupported schema_version {self.schema_version!r}")
    if self.current_index != self.past_steps:
      raise ContractError("current_index must equal past_steps")
    if self.window_steps != self.past_steps + 1 + self.future_steps:
      raise ContractError("window dimensions are inconsistent")
    if (
      len(self.body_names) != self.body_count
      or self.state_dimension != 15 + 6 * self.body_count
    ):
      raise ContractError("ordered body names and state dimension are inconsistent")
    if self.projected_state_dimension != self.state_dimension + self.projection_rows:
      raise ContractError("projection dimension is inconsistent")
    if self.token_dimension != self.projected_state_dimension + self.latent_dimension:
      raise ContractError("token dimension is inconsistent")
    if self.control_hz <= 0 or self.period_seconds <= 0 or self.std_floor <= 0:
      raise ContractError(
        "control period and standard-deviation floor must be positive"
      )
    if len(self.split_fractions) != 3 or abs(sum(self.split_fractions) - 1.0) > 1e-12:
      raise ContractError("split fractions must contain three values summing to one")
    if any(value <= 0 for value in self.split_fractions):
      raise ContractError("split fractions must be positive")
    ArtifactIdentity("vae", self.vae_checkpoint, self.vae_sha256)
    ArtifactIdentity("recovery", self.recovery_onnx, self.recovery_sha256)
    ArtifactIdentity("x2_xml", self.state_xml, self.state_xml_sha256)
    ArtifactIdentity(
      "physical_fall_source",
      self.physical_fall_source,
      self.physical_fall_source_sha256,
    )
    ArtifactIdentity(
      "vae_tracking_rejection_source",
      self.vae_tracking_rejection_source,
      self.vae_tracking_rejection_source_sha256,
    )

  @classmethod
  def from_mapping(cls, payload: Mapping[str, Any]) -> "DiffusionContract":
    """Build a contract from a strict, already-decoded D0 mapping.

    The YAML contains many D2 fields.  They are deliberately ignored here,
    while all D1 identity values are checked against the frozen defaults.
    """
    if not isinstance(payload, Mapping) or payload.get("schema_version") != _SCHEMA:
      raise ContractError("payload is not the frozen D0 schema")
    state = payload.get("state", {})
    window = payload.get("window", {})
    prep = payload.get("preprocessing", {})
    proj = prep.get("projection", {})
    dims = payload.get("vae", {}).get("dimensions", {})
    try:
      fractions = tuple(float(x) for x in prep["split_fractions"])
      if len(fractions) != 3:
        raise ValueError("split_fractions must contain three values")
      contract = cls(
        schema_version=str(payload["schema_version"]),
        control_hz=int(payload["control"]["frequency_hz"]),
        period_seconds=float(payload["control"]["period_seconds"]),
        state_dimension=int(state["raw_dimension"]),
        body_count=len(state["body_order"]),
        latent_dimension=int(dims["latent"]),
        action_dimension=int(dims["action"]),
        past_steps=int(window["past_steps"]),
        current_index=int(window["current_index"]),
        future_steps=int(window["future_steps"]),
        window_steps=int(window["total_steps"]),
        projected_state_dimension=int(proj["projected_dimension"]),
        token_dimension=int(proj["token_dimension"]),
        split_seed=int(prep["split_seed"]),
        split_fractions=(fractions[0], fractions[1], fractions[2]),
        std_floor=float(prep["std_floor"]),
        projection_seed=int(proj["seed"]),
        projection_rows=int(proj["gaussian_rows"]),
        projection_rcond=float(proj["pseudoinverse_rcond"]),
        vae_checkpoint=str(payload["vae"]["checkpoint"]),
        vae_sha256=str(payload["vae"]["sha256"]),
        recovery_onnx=str(payload["recovery"]["onnx"]),
        recovery_sha256=str(payload["recovery"]["sha256"]),
        state_xml=str(state["xml"]),
        state_xml_sha256=str(state["xml_sha256"]),
        physical_fall_source=str(payload["qualification"]["physical_fall"]["source"]),
        physical_fall_source_sha256=str(
          payload["qualification"]["physical_fall"]["source_sha256"]
        ),
        vae_tracking_rejection_source=str(
          payload["qualification"]["vae_tracking_rejection"]["source"]
        ),
        vae_tracking_rejection_source_sha256=str(
          payload["qualification"]["vae_tracking_rejection"]["source_sha256"]
        ),
        body_names=tuple(str(x) for x in state["body_order"]),
      )
    except (KeyError, TypeError, ValueError) as exc:
      raise ContractError(f"malformed D0 payload: {exc}") from exc
    if (
      contract.vae_sha256 != _VAE_SHA
      or contract.recovery_sha256 != _RECOVERY_SHA
      or contract.state_xml_sha256 != _XML_SHA
      or contract.physical_fall_source_sha256 != _PHYSICAL_FALL_SHA
      or contract.vae_tracking_rejection_source_sha256 != _VAE_TRACKING_REJECTION_SHA
    ):
      raise ContractError("D0 artifact identities do not match the selected assets")
    return contract

  @classmethod
  def from_yaml(cls, path: str | Path) -> "DiffusionContract":
    """Load and validate the frozen YAML without making it executable config."""
    try:
      import yaml
    except ImportError as exc:  # pragma: no cover - project environments include yaml
      raise ContractError("PyYAML is required to read the frozen D0 manifest") from exc
    source = Path(path).read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != _D0_CONTRACT_SHA:
      raise ContractError(
        f"frozen D0 contract hash mismatch: expected {_D0_CONTRACT_SHA}, got {digest}"
      )
    payload = yaml.safe_load(source)
    contract = cls.from_mapping(payload)
    return replace(contract, contract_sha256=digest)

  def as_dict(self) -> dict[str, Any]:
    return {
      "schema_version": self.schema_version,
      "contract_sha256": self.contract_sha256,
      "control_hz": self.control_hz,
      "period_seconds": self.period_seconds,
      "state_dimension": self.state_dimension,
      "body_count": self.body_count,
      "latent_dimension": self.latent_dimension,
      "action_dimension": self.action_dimension,
      "past_steps": self.past_steps,
      "current_index": self.current_index,
      "future_steps": self.future_steps,
      "window_steps": self.window_steps,
      "projected_state_dimension": self.projected_state_dimension,
      "token_dimension": self.token_dimension,
      "split_seed": self.split_seed,
      "split_fractions": list(self.split_fractions),
      "std_floor": self.std_floor,
      "projection_seed": self.projection_seed,
      "projection_rows": self.projection_rows,
      "projection_rcond": self.projection_rcond,
      "vae_checkpoint": self.vae_checkpoint,
      "vae_sha256": self.vae_sha256,
      "recovery_onnx": self.recovery_onnx,
      "recovery_sha256": self.recovery_sha256,
      "state_xml": self.state_xml,
      "state_xml_sha256": self.state_xml_sha256,
      "physical_fall_source": self.physical_fall_source,
      "physical_fall_source_sha256": self.physical_fall_source_sha256,
      "vae_tracking_rejection_source": self.vae_tracking_rejection_source,
      "vae_tracking_rejection_source_sha256": self.vae_tracking_rejection_source_sha256,
      "body_names": list(self.body_names),
    }

  def identity_hash(self) -> str:
    encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()

  def validate_artifacts(
    self, artifacts: Mapping[str, ArtifactIdentity | Mapping[str, Any]]
  ) -> None:
    """Fail closed unless the pinned VAE and recovery identities are present."""
    for role, path, digest in (
      ("vae", self.vae_checkpoint, self.vae_sha256),
      ("recovery", self.recovery_onnx, self.recovery_sha256),
      ("x2_xml", self.state_xml, self.state_xml_sha256),
      (
        "physical_fall_source",
        self.physical_fall_source,
        self.physical_fall_source_sha256,
      ),
      (
        "vae_tracking_rejection_source",
        self.vae_tracking_rejection_source,
        self.vae_tracking_rejection_source_sha256,
      ),
    ):
      item = artifacts.get(role)
      if isinstance(item, Mapping):
        try:
          fields = cast("Mapping[str, object]", item)
          item = ArtifactIdentity(
            str(fields["role"]), str(fields["path"]), str(fields["sha256"])
          )
        except (KeyError, TypeError, ValueError) as exc:
          raise ContractError(f"invalid {role} artifact identity") from exc
      if (
        not isinstance(item, ArtifactIdentity)
        or item.path != path
        or item.sha256 != digest
      ):
        raise ContractError(f"artifact {role!r} does not match the frozen identity")


DEFAULT_CONTRACT = DiffusionContract()

__all__ = ["ArtifactIdentity", "ContractError", "DEFAULT_CONTRACT", "DiffusionContract"]
