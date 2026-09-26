"""Versioned schemas and model defaults for the M2 latent policy core.

The schema is deliberately independent of a simulator.  A caller supplies a
single, named observation snapshot and :func:`pack_observations` in
``observations.py`` performs the only feature ordering operation.  Frame names
are declarations from the teacher contract; physical site/frame verification
is intentionally not implied by this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

SCHEMA_VERSION = 1
JOINT_DIM = 31
REFERENCE_DIM = 68
ACTION_DIM = 31
LATENT_DIM = 32
CONDITIONING_DIMS = {"gravity": 99, "anchor": 102, "gravity_anchor": 105}

DEFAULT_JOINT_ORDER = tuple(f"joint_{index:02d}" for index in range(JOINT_DIM))


class DecoderMode(str, Enum):
  """Explicit decoder conditioning schemas supported by M2."""

  GRAVITY = "gravity"
  ANCHOR = "anchor"
  GRAVITY_ANCHOR = "gravity_anchor"


@dataclass(frozen=True, slots=True)
class FrameIdentity:
  """A declared feature frame and its unresolved physical-verification status."""

  name: str
  verification: str = "declared_unverified"

  def to_metadata(self) -> dict[str, str]:
    return {"name": self.name, "verification": self.verification}

  @classmethod
  def from_metadata(cls, value: Any) -> FrameIdentity:
    if not isinstance(value, dict) or set(value) != {"name", "verification"}:
      raise ValueError("frame metadata must contain name and verification")
    name, verification = value["name"], value["verification"]
    if not isinstance(name, str) or not isinstance(verification, str):
      raise ValueError("frame metadata values must be strings")
    return cls(name=name, verification=verification)


@dataclass(frozen=True, slots=True)
class FeatureSpec:
  """One ordered raw feature in a reference or decoder batch."""

  name: str
  dimension: int
  frame: FrameIdentity

  def to_metadata(self) -> dict[str, Any]:
    return {
      "name": self.name,
      "dimension": self.dimension,
      "frame": self.frame.to_metadata(),
    }

  @classmethod
  def from_metadata(cls, value: Any) -> FeatureSpec:
    if not isinstance(value, dict) or set(value) != {"name", "dimension", "frame"}:
      raise ValueError("feature metadata must contain name, dimension and frame")
    name, dimension = value["name"], value["dimension"]
    if not isinstance(name, str) or not isinstance(dimension, int) or dimension <= 0:
      raise ValueError("feature name and positive integer dimension are required")
    return cls(
      name=name, dimension=dimension, frame=FrameIdentity.from_metadata(value["frame"])
    )


@dataclass(frozen=True, slots=True)
class VaeSchema:
  """Immutable, versioned tensor layout for one decoder mode.

  The reference layout is shared by all modes.  In particular, the default
  gravity decoder has no reference feature in ``conditioning_fields``; the
  reference can influence it only through the latent produced by the encoder.
  """

  mode: DecoderMode
  joint_order: tuple[str, ...] = DEFAULT_JOINT_ORDER
  version: int = SCHEMA_VERSION
  reference_fields: tuple[FeatureSpec, ...] = ()
  conditioning_fields: tuple[FeatureSpec, ...] = ()
  latent_dim: int = LATENT_DIM
  action_dim: int = ACTION_DIM

  def __post_init__(self) -> None:
    if self.version != SCHEMA_VERSION:
      raise ValueError(f"unsupported schema version {self.version}")
    if len(self.joint_order) != JOINT_DIM or any(not name for name in self.joint_order):
      raise ValueError(f"joint_order must contain {JOINT_DIM} non-empty names")
    if len(set(self.joint_order)) != len(self.joint_order):
      raise ValueError("joint_order names must be unique")
    if self.latent_dim != LATENT_DIM or self.action_dim != ACTION_DIM:
      raise ValueError("M2 schemas require latent_dim=32 and action_dim=31")
    if self.reference_dim != REFERENCE_DIM:
      raise ValueError(f"reference layout must have width {REFERENCE_DIM}")
    expected = CONDITIONING_DIMS[self.mode.value]
    if self.conditioning_dim != expected:
      raise ValueError(
        f"{self.mode.value} conditioning layout must have width {expected}, "
        f"got {self.conditioning_dim}"
      )
    if self.reference_names != (
      "reference_q",
      "reference_dq",
      "anchor_orientation_error",
    ):
      raise ValueError("reference fields are not in the required q, dq, anchor order")
    if self.mode is DecoderMode.GRAVITY:
      expected_names = (
        "projected_gravity",
        "gyro",
        "relative_joint_q",
        "joint_dq",
        "previous_action",
      )
    elif self.mode is DecoderMode.ANCHOR:
      expected_names = (
        "anchor_orientation_error",
        "gyro",
        "relative_joint_q",
        "joint_dq",
        "previous_action",
      )
    else:
      expected_names = (
        "projected_gravity",
        "anchor_orientation_error",
        "gyro",
        "relative_joint_q",
        "joint_dq",
        "previous_action",
      )
    if self.conditioning_names != expected_names:
      raise ValueError(f"{self.mode.value} fields are not in the required order")

  @property
  def reference_names(self) -> tuple[str, ...]:
    return tuple(field.name for field in self.reference_fields)

  @property
  def conditioning_names(self) -> tuple[str, ...]:
    return tuple(field.name for field in self.conditioning_fields)

  @property
  def reference_dim(self) -> int:
    return sum(field.dimension for field in self.reference_fields)

  @property
  def conditioning_dim(self) -> int:
    return sum(field.dimension for field in self.conditioning_fields)

  @property
  def decoder_input_dim(self) -> int:
    return self.latent_dim + self.conditioning_dim

  @property
  def decoder_reference_conditioned(self) -> bool:
    return self.mode is not DecoderMode.GRAVITY

  def compatibility_metadata(self) -> dict[str, Any]:
    """Return JSON-compatible metadata sufficient to reject schema collisions."""
    return {
      "schema_version": self.version,
      "mode": self.mode.value,
      "joint_order": list(self.joint_order),
      "reference_fields": [field.to_metadata() for field in self.reference_fields],
      "conditioning_fields": [
        field.to_metadata() for field in self.conditioning_fields
      ],
      "reference_dim": self.reference_dim,
      "conditioning_dim": self.conditioning_dim,
      "latent_dim": self.latent_dim,
      "action_dim": self.action_dim,
      "decoder_reference_conditioned": self.decoder_reference_conditioned,
    }

  def assert_compatible(self, metadata: Any) -> None:
    """Raise when serialized metadata is not exactly this schema identity."""
    if metadata != self.compatibility_metadata():
      raise ValueError(
        f"schema metadata does not match {self.mode.value} version {self.version}"
      )

  @classmethod
  def from_metadata(cls, metadata: Any) -> VaeSchema:
    if not isinstance(metadata, dict):
      raise ValueError("schema metadata must be a mapping")
    required = {
      "schema_version",
      "mode",
      "joint_order",
      "reference_fields",
      "conditioning_fields",
      "reference_dim",
      "conditioning_dim",
      "latent_dim",
      "action_dim",
      "decoder_reference_conditioned",
    }
    if set(metadata) != required:
      raise ValueError("schema metadata has missing or unknown fields")
    try:
      mode = DecoderMode(metadata["mode"])
      joint_order = tuple(metadata["joint_order"])
      reference_fields = tuple(
        FeatureSpec.from_metadata(value) for value in metadata["reference_fields"]
      )
      conditioning_fields = tuple(
        FeatureSpec.from_metadata(value) for value in metadata["conditioning_fields"]
      )
      schema = cls(
        mode=mode,
        joint_order=joint_order,
        version=metadata["schema_version"],
        reference_fields=reference_fields,
        conditioning_fields=conditioning_fields,
        latent_dim=metadata["latent_dim"],
        action_dim=metadata["action_dim"],
      )
    except (KeyError, TypeError, ValueError) as exc:
      raise ValueError("invalid serialized VAE schema metadata") from exc
    if metadata["reference_dim"] != schema.reference_dim:
      raise ValueError("serialized reference_dim disagrees with fields")
    if metadata["conditioning_dim"] != schema.conditioning_dim:
      raise ValueError("serialized conditioning_dim disagrees with fields")
    if (
      metadata["decoder_reference_conditioned"] != schema.decoder_reference_conditioned
    ):
      raise ValueError("serialized reference-conditioning flag disagrees with mode")
    return schema


@dataclass(frozen=True, slots=True)
class ModelSettings:
  """M2 defaults shared by the next model component."""

  latent_dim: int = LATENT_DIM
  hidden_dims: tuple[int, ...] = (2048, 1024, 512)
  activation: str = "ELU"
  beta: float = 0.01

  def __post_init__(self) -> None:
    if self.latent_dim != LATENT_DIM:
      raise ValueError("M2 latent dimension must be 32")
    if not self.hidden_dims or any(
      not isinstance(width, int) or width <= 0 for width in self.hidden_dims
    ):
      raise ValueError("hidden_dims must contain positive integers")
    if self.activation != "ELU":
      raise ValueError("M2 uses ELU activations")
    if not isinstance(self.beta, (int, float)) or self.beta < 0:
      raise ValueError("beta must be finite and non-negative")
    if not math.isfinite(float(self.beta)):
      raise ValueError("beta must be finite and non-negative")

  def to_metadata(self) -> dict[str, Any]:
    return {
      "latent_dim": self.latent_dim,
      "hidden_dims": list(self.hidden_dims),
      "activation": self.activation,
      "beta": self.beta,
    }


VAEModelConfig = ModelSettings
DEFAULT_MODEL_SETTINGS = ModelSettings()


def _field(name: str, dimension: int, frame: str) -> FeatureSpec:
  return FeatureSpec(name, dimension, FrameIdentity(frame))


def make_schema(
  mode: DecoderMode | str = DecoderMode.GRAVITY,
  joint_order: tuple[str, ...] = DEFAULT_JOINT_ORDER,
) -> VaeSchema:
  """Build one immutable layout, with gravity as the default mode."""
  selected = DecoderMode(mode)
  reference_fields = (
    _field("reference_q", JOINT_DIM, "reference"),
    _field("reference_dq", JOINT_DIM, "reference"),
    _field("anchor_orientation_error", 6, "anchor"),
  )
  common = (
    _field("gyro", 3, "imu"),
    _field("relative_joint_q", JOINT_DIM, "joint"),
    _field("joint_dq", JOINT_DIM, "joint"),
    _field("previous_action", ACTION_DIM, "action"),
  )
  if selected is DecoderMode.GRAVITY:
    conditioning_fields = (_field("projected_gravity", 3, "root"), *common)
  elif selected is DecoderMode.ANCHOR:
    conditioning_fields = (reference_fields[2], *common)
  else:
    conditioning_fields = (
      _field("projected_gravity", 3, "root"),
      reference_fields[2],
      *common,
    )
  return VaeSchema(
    mode=selected,
    joint_order=tuple(joint_order),
    reference_fields=reference_fields,
    conditioning_fields=conditioning_fields,
  )


DEFAULT_SCHEMA = make_schema()
"""The gravity-first default schema (68 reference / 99 conditioning)."""


def schema_for_mode(mode: DecoderMode | str) -> VaeSchema:
  """Return a fresh immutable schema for an explicitly selected mode."""
  return make_schema(mode)


__all__ = [
  "ACTION_DIM",
  "CONDITIONING_DIMS",
  "DEFAULT_JOINT_ORDER",
  "DEFAULT_SCHEMA",
  "DEFAULT_MODEL_SETTINGS",
  "DecoderMode",
  "FeatureSpec",
  "FrameIdentity",
  "JOINT_DIM",
  "LATENT_DIM",
  "ModelSettings",
  "REFERENCE_DIM",
  "SCHEMA_VERSION",
  "VAEModelConfig",
  "VaeSchema",
  "make_schema",
  "schema_for_mode",
]
