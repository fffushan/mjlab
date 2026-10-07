"""Pure, validated policy for difficult-state VAE coverage.

The policy is deliberately plain data: it does not import a simulator, consume a
random stream while being constructed, or carry teacher/latent outcomes.  The
``off`` instance is the compatibility boundary used by legacy collectors.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

CoverageMode = Literal["off", "balance-recovery"]
CoverageTier = Literal["baseline", "moderate", "strong"]
RootDirection = Literal["sagittal", "lateral", "yaw", "horizontal_translation"]
ExcitationKind = Literal["root_only", "joint_chain_only", "coupled"]

POLICY_VERSION = 1
POLICY_KIND = "x2-vae-balance-recovery-p0-v1"


class CoveragePolicyError(ValueError):
  """Raised when coverage policy data is malformed or unsupported."""


def _finite(value: object, name: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise CoveragePolicyError(f"{name} must be a finite number")
  result = float(value)
  if not math.isfinite(result):
    raise CoveragePolicyError(f"{name} must be finite")
  return result


def _probability(value: object, name: str) -> float:
  result = _finite(value, name)
  if not 0.0 <= result <= 1.0:
    raise CoveragePolicyError(f"{name} must be in [0, 1]")
  return result


def _positive_int(value: object, name: str, *, zero: bool = False) -> int:
  if isinstance(value, bool) or not isinstance(value, int):
    raise CoveragePolicyError(f"{name} must be an integer")
  if value < (0 if zero else 1):
    raise CoveragePolicyError(f"{name} must be {'nonnegative' if zero else 'positive'}")
  return value


def _tuple_names(value: object, name: str) -> tuple[str, ...]:
  if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
    raise CoveragePolicyError(f"{name} must be a sequence of names")
  result = tuple(value)
  if not all(isinstance(item, str) and item for item in result):
    raise CoveragePolicyError(f"{name} must contain non-empty names")
  if len(set(result)) != len(result):
    raise CoveragePolicyError(f"{name} contains duplicate names")
  return result


def _mapping_numbers(value: object, name: str) -> tuple[tuple[str, float], ...]:
  if isinstance(value, Mapping):
    items = list(value.items())
  elif isinstance(value, (tuple, list)):
    items = []
    for item in value:
      if not isinstance(item, (tuple, list)) or len(item) != 2:
        raise CoveragePolicyError(f"{name} must contain name/value pairs")
      items.append((item[0], item[1]))
  else:
    raise CoveragePolicyError(f"{name} must be a mapping")
  normalized: list[tuple[str, float]] = []
  for key, item in items:
    if not isinstance(key, str) or not key:
      raise CoveragePolicyError(f"{name} keys must be non-empty strings")
    if key in {entry[0] for entry in normalized}:
      raise CoveragePolicyError(f"{name} keys must be unique")
    normalized.append((key, _finite(item, f"{name}.{key}")))
  result = tuple(sorted(normalized))
  if any(item < 0 for _, item in result):
    raise CoveragePolicyError(f"{name} cannot contain negative magnitudes")
  return result


def _validate_weights(
  values: object, name: str, expected: set[str]
) -> tuple[tuple[str, float], ...]:
  if not isinstance(values, (tuple, list)):
    raise CoveragePolicyError(f"{name} must be a sequence of name/probability pairs")
  normalized: list[tuple[str, float]] = []
  for item in values:
    if not isinstance(item, (tuple, list)) or len(item) != 2:
      raise CoveragePolicyError(f"{name} must contain name/probability pairs")
    key, raw = item
    if not isinstance(key, str) or key in {entry[0] for entry in normalized}:
      raise CoveragePolicyError(f"{name} names must be unique strings")
    probability = _probability(raw, f"{name}.{key}")
    if probability <= 0.0:
      raise CoveragePolicyError(f"{name}.{key} must be positive")
    normalized.append((key, probability))
  if {key for key, _ in normalized} != expected:
    raise CoveragePolicyError(f"{name} is incomplete or contains unknown names")
  if not math.isclose(sum(value for _, value in normalized), 1.0, abs_tol=1e-9):
    raise CoveragePolicyError(f"{name} must sum to 1")
  return tuple(sorted(normalized))


def _validate_scope(scope: object) -> tuple[tuple[str, Any], ...]:
  if not isinstance(scope, (tuple, list)):
    raise CoveragePolicyError("scope must be a sequence of pairs")
  result: list[tuple[str, Any]] = []
  for item in scope:
    if (
      not isinstance(item, (tuple, list))
      or len(item) != 2
      or not isinstance(item[0], str)
    ):
      raise CoveragePolicyError("scope must contain unique string-keyed pairs")
    if item[0] in {key for key, _ in result}:
      raise CoveragePolicyError("scope keys must be unique")
    result.append((item[0], item[1]))
  return tuple(result)


@dataclass(frozen=True, slots=True)
class RootEnvelope:
  position_m: tuple[tuple[str, float], ...]
  rotation_rad: tuple[tuple[str, float], ...]
  linear_velocity_m_s: tuple[tuple[str, float], ...]
  angular_velocity_rad_s: tuple[tuple[str, float], ...]

  def __post_init__(self) -> None:
    expected = {
      "position_m": {"x", "y", "z"},
      "rotation_rad": {"roll", "pitch", "yaw"},
      "linear_velocity_m_s": {"x", "y", "z"},
      "angular_velocity_rad_s": {"roll", "pitch", "yaw"},
    }
    for field_name, values in self.as_dict().items():
      if set(values) != expected[field_name]:
        raise CoveragePolicyError(f"{field_name} axes are incomplete or unknown")
      _mapping_numbers(values, field_name)

  @classmethod
  def from_dict(cls, value: object, name: str) -> RootEnvelope:
    if not isinstance(value, Mapping):
      raise CoveragePolicyError(f"{name} must be a mapping")
    required = (
      "position_m",
      "rotation_rad",
      "linear_velocity_m_s",
      "angular_velocity_rad_s",
    )
    if set(value) != set(required):
      raise CoveragePolicyError(f"{name} fields must be {required}")
    axes = {
      "position_m": {"x", "y", "z"},
      "rotation_rad": {"roll", "pitch", "yaw"},
      "linear_velocity_m_s": {"x", "y", "z"},
      "angular_velocity_rad_s": {"roll", "pitch", "yaw"},
    }
    fields = []
    for key in required:
      field = value[key]
      if not isinstance(field, Mapping) or set(field) != axes[key]:
        raise CoveragePolicyError(f"{name}.{key} axes are malformed")
      fields.append(_mapping_numbers(field, f"{name}.{key}"))
    return cls(*fields)

  def as_dict(self) -> dict[str, dict[str, float]]:
    return {
      key: dict(value)
      for key, value in (
        ("position_m", self.position_m),
        ("rotation_rad", self.rotation_rad),
        ("linear_velocity_m_s", self.linear_velocity_m_s),
        ("angular_velocity_rad_s", self.angular_velocity_rad_s),
      )
    }


@dataclass(frozen=True, slots=True)
class TriggerPolicy:
  thresholds: tuple[tuple[str, float], ...]
  rearm_low_fraction: float = 0.75
  low_dwell_steps: int = 5
  cooldown_steps: int = 25
  max_threshold_candidates_per_segment: int = 2

  def __post_init__(self) -> None:
    allowed = {
      "reference_relative_root_gravity_angle_rad",
      "reference_relative_anchor_gravity_angle_rad",
      "reference_relative_anchor_rotation_angle_rad",
      "root_angular_velocity_error_rad_s",
      "selected_legs_waist_joint_position_rms_rad",
      "selected_legs_waist_joint_velocity_rms_rad_s",
    }
    normalized: list[tuple[str, float]] = []
    for item in self.thresholds:
      if not isinstance(item, (tuple, list)) or len(item) != 2:
        raise CoveragePolicyError("trigger thresholds must be name/value pairs")
      name, value = item
      if not isinstance(name, str) or name in {entry[0] for entry in normalized}:
        raise CoveragePolicyError("trigger threshold names must be unique strings")
      normalized.append((name, _finite(value, f"thresholds.{name}")))
    if set(name for name, _ in normalized) != allowed:
      raise CoveragePolicyError(
        "trigger thresholds are incomplete or contain unknown metrics"
      )
    object.__setattr__(self, "thresholds", tuple(sorted(normalized)))
    if any(value <= 0 for _, value in self.thresholds):
      raise CoveragePolicyError("trigger thresholds must be positive")
    object.__setattr__(
      self,
      "rearm_low_fraction",
      _probability(self.rearm_low_fraction, "rearm_low_fraction"),
    )
    object.__setattr__(
      self, "low_dwell_steps", _positive_int(self.low_dwell_steps, "low_dwell_steps")
    )
    object.__setattr__(
      self,
      "cooldown_steps",
      _positive_int(self.cooldown_steps, "cooldown_steps", zero=True),
    )
    object.__setattr__(
      self,
      "max_threshold_candidates_per_segment",
      _positive_int(
        self.max_threshold_candidates_per_segment,
        "max_threshold_candidates_per_segment",
      ),
    )

  def as_dict(self) -> dict[str, Any]:
    return {
      "thresholds": dict(self.thresholds),
      "rearm_low_fraction": self.rearm_low_fraction,
      "low_dwell_steps": self.low_dwell_steps,
      "cooldown_steps": self.cooldown_steps,
      "max_threshold_candidates_per_segment": self.max_threshold_candidates_per_segment,
    }


@dataclass(frozen=True, slots=True)
class RuntimePolicy:
  per_segment_opportunity_probability: float = 0.50
  max_events_per_segment: int = 1
  target_phase_bins: int = 5
  min_segment_age_steps: int = 10
  min_steps_remaining: int = 5

  def __post_init__(self) -> None:
    object.__setattr__(
      self,
      "per_segment_opportunity_probability",
      _probability(
        self.per_segment_opportunity_probability, "per_segment_opportunity_probability"
      ),
    )
    for field_name in (
      "max_events_per_segment",
      "target_phase_bins",
      "min_segment_age_steps",
      "min_steps_remaining",
    ):
      object.__setattr__(
        self, field_name, _positive_int(getattr(self, field_name), field_name)
      )

  def as_dict(self) -> dict[str, Any]:
    return {
      "per_segment_opportunity_probability": self.per_segment_opportunity_probability,
      "max_events_per_segment": self.max_events_per_segment,
      "target_phase_bins": self.target_phase_bins,
      "min_segment_age_steps": self.min_segment_age_steps,
      "min_steps_remaining": self.min_steps_remaining,
    }


# These are tuples instead of dicts so the frozen policy is deeply immutable.
_DEFAULT_CHAINS = (
  (
    "left_sagittal",
    ("left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint"),
  ),
  (
    "right_sagittal",
    ("right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint"),
  ),
  ("left_lateral", ("left_hip_roll_joint", "left_ankle_roll_joint")),
  ("right_lateral", ("right_hip_roll_joint", "right_ankle_roll_joint")),
  ("waist", ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint")),
)


@dataclass(frozen=True, slots=True)
class CoveragePolicy:
  """Canonical P0 settings and identity for one collection run."""

  mode: CoverageMode = "off"
  version: int = POLICY_VERSION
  scope: tuple[tuple[str, Any], ...] = ()
  tier_probabilities: tuple[tuple[str, float], ...] = ()
  root_reset_envelopes: tuple[tuple[str, RootEnvelope], ...] = ()
  reset_excitation_probabilities: tuple[tuple[str, float], ...] = ()
  root_direction_probabilities: tuple[tuple[str, float], ...] = ()
  joint_chain_probabilities: tuple[tuple[str, float], ...] = ()
  joint_chains: tuple[tuple[str, tuple[str, ...]], ...] = ()
  joint_position_envelopes_rad: tuple[
    tuple[str, tuple[tuple[str, float], ...]], ...
  ] = ()
  joint_velocity_delta_envelopes_rad_s: tuple[
    tuple[str, tuple[tuple[str, float], ...]], ...
  ] = ()
  runtime: RuntimePolicy = RuntimePolicy()
  triggers: TriggerPolicy = TriggerPolicy(
    (
      ("reference_relative_root_gravity_angle_rad", 0.17453292519943295),
      ("reference_relative_anchor_gravity_angle_rad", 0.17453292519943295),
      ("reference_relative_anchor_rotation_angle_rad", 0.2617993877991494),
      ("root_angular_velocity_error_rad_s", 1.0),
      ("selected_legs_waist_joint_position_rms_rad", 0.20),
      ("selected_legs_waist_joint_velocity_rms_rad_s", 1.0),
    )
  )

  def __post_init__(self) -> None:
    if self.mode not in ("off", "balance-recovery"):
      raise CoveragePolicyError(f"invalid coverage mode {self.mode!r}")
    if isinstance(self.version, bool) or self.version != POLICY_VERSION:
      raise CoveragePolicyError(f"unsupported coverage policy version {self.version}")
    object.__setattr__(self, "scope", _validate_scope(self.scope))
    if self.mode == "off":
      if any(
        (
          self.scope,
          self.tier_probabilities,
          self.root_reset_envelopes,
          self.reset_excitation_probabilities,
          self.root_direction_probabilities,
          self.joint_chain_probabilities,
          self.joint_chains,
          self.joint_position_envelopes_rad,
          self.joint_velocity_delta_envelopes_rad_s,
        )
      ):
        raise CoveragePolicyError("off policy cannot carry enabled coverage settings")
      return
    object.__setattr__(
      self,
      "tier_probabilities",
      _validate_weights(
        self.tier_probabilities,
        "tier_probabilities",
        {"baseline", "moderate", "strong"},
      ),
    )
    roots = (
      tuple(self.root_reset_envelopes)
      if isinstance(self.root_reset_envelopes, (tuple, list))
      else ()
    )
    if {name for name, _ in roots} != {"baseline", "moderate", "strong"} or len(
      roots
    ) != 3:
      raise CoveragePolicyError("all three reset tiers are required exactly once")
    if len({name for name, _ in roots}) != len(roots) or any(
      not isinstance(value, RootEnvelope) for _, value in roots
    ):
      raise CoveragePolicyError("reset envelopes must be unique RootEnvelope values")
    object.__setattr__(self, "root_reset_envelopes", tuple(sorted(roots)))
    object.__setattr__(
      self,
      "reset_excitation_probabilities",
      _validate_weights(
        self.reset_excitation_probabilities,
        "reset_excitation_probabilities",
        {"root_only", "joint_chain_only", "coupled"},
      ),
    )
    object.__setattr__(
      self,
      "root_direction_probabilities",
      _validate_weights(
        self.root_direction_probabilities,
        "root_direction_probabilities",
        {"sagittal", "lateral", "yaw", "horizontal_translation"},
      ),
    )
    object.__setattr__(
      self,
      "joint_chain_probabilities",
      _validate_weights(
        self.joint_chain_probabilities,
        "joint_chain_probabilities",
        {"left_sagittal", "right_sagittal", "left_lateral", "right_lateral", "waist"},
      ),
    )
    chains = (
      tuple(self.joint_chains) if isinstance(self.joint_chains, (tuple, list)) else ()
    )
    if {name for name, _ in chains} != {
      "left_sagittal",
      "right_sagittal",
      "left_lateral",
      "right_lateral",
      "waist",
    } or len({name for name, _ in chains}) != len(chains):
      raise CoveragePolicyError("joint chains are incomplete or duplicated")
    for name, joints in chains:
      if (
        not isinstance(name, str)
        or not isinstance(joints, (tuple, list))
        or not joints
        or len(set(joints)) != len(joints)
        or not all(isinstance(joint, str) and joint for joint in joints)
      ):
        raise CoveragePolicyError(f"invalid joint chain {name!r}")
    object.__setattr__(
      self,
      "joint_chains",
      tuple(sorted((name, tuple(joints)) for name, joints in chains)),
    )
    expected_tiers = {"baseline", "moderate", "strong"}
    positions = (
      tuple(self.joint_position_envelopes_rad)
      if isinstance(self.joint_position_envelopes_rad, (tuple, list))
      else ()
    )
    velocities = (
      tuple(self.joint_velocity_delta_envelopes_rad_s)
      if isinstance(self.joint_velocity_delta_envelopes_rad_s, (tuple, list))
      else ()
    )
    if {name for name, _ in positions} != expected_tiers or {
      name for name, _ in velocities
    } != expected_tiers:
      raise CoveragePolicyError("joint envelope tiers are incomplete or unknown")
    object.__setattr__(
      self,
      "joint_position_envelopes_rad",
      tuple(
        sorted(
          (name, _mapping_numbers(values, f"joint_position_envelopes_rad.{name}"))
          for name, values in positions
        )
      ),
    )
    object.__setattr__(
      self,
      "joint_velocity_delta_envelopes_rad_s",
      tuple(
        sorted(
          (
            name,
            _mapping_numbers(values, f"joint_velocity_delta_envelopes_rad_s.{name}"),
          )
          for name, values in velocities
        )
      ),
    )

  @property
  def enabled(self) -> bool:
    return self.mode != "off"

  @property
  def identity(self) -> str:
    return hashlib.sha256(self.canonical_json().encode()).hexdigest()

  def as_dict(self) -> dict[str, Any]:
    return {
      "version": self.version,
      "contract_id": POLICY_KIND,
      "mode": self.mode,
      "scope": dict(self.scope),
      "tier_probabilities": dict(self.tier_probabilities),
      "root_reset_envelopes": {k: v.as_dict() for k, v in self.root_reset_envelopes},
      "reset_excitation_probabilities": dict(self.reset_excitation_probabilities),
      "root_direction_probabilities": dict(self.root_direction_probabilities),
      "joint_chain_probabilities": dict(self.joint_chain_probabilities),
      "joint_chains": {k: list(v) for k, v in self.joint_chains},
      "joint_position_envelopes_rad": {
        k: dict(v) for k, v in self.joint_position_envelopes_rad
      },
      "joint_velocity_delta_envelopes_rad_s": {
        k: dict(v) for k, v in self.joint_velocity_delta_envelopes_rad_s
      },
      "runtime": self.runtime.as_dict(),
      "triggers": self.triggers.as_dict(),
    }

  def canonical_json(self) -> str:
    return json.dumps(
      self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    )

  @classmethod
  def off(cls) -> CoveragePolicy:
    return cls()

  @classmethod
  def balance_recovery(cls) -> CoveragePolicy:
    def root(
      position: dict[str, float],
      rotation: dict[str, float],
      linear: dict[str, float],
      angular: dict[str, float],
    ) -> RootEnvelope:
      return RootEnvelope.from_dict(
        {
          "position_m": position,
          "rotation_rad": rotation,
          "linear_velocity_m_s": linear,
          "angular_velocity_rad_s": angular,
        },
        "root",
      )

    common_pos = {"x": 0.025, "y": 0.025, "z": 0.005}
    common_rot = {"roll": 0.05, "pitch": 0.05, "yaw": 0.10}
    common_lin = {"x": 0.25, "y": 0.25, "z": 0.10}
    common_ang = {"roll": 0.26, "pitch": 0.26, "yaw": 0.39}
    roots = {
      "baseline": root(common_pos, common_rot, common_lin, common_ang),
      "moderate": root(
        {"x": 0.05, "y": 0.05, "z": 0.005},
        {"roll": 0.15, "pitch": 0.15, "yaw": 0.10},
        {"x": 0.5, "y": 0.5, "z": 0.1},
        {"roll": 0.75, "pitch": 0.75, "yaw": 0.39},
      ),
      "strong": root(
        {"x": 0.075, "y": 0.075, "z": 0.005},
        {"roll": 0.3, "pitch": 0.3, "yaw": 0.2},
        {"x": 1.0, "y": 1.0, "z": 0.1},
        {"roll": 1.5, "pitch": 1.5, "yaw": 0.78},
      ),
    }
    return cls(
      mode="balance-recovery",
      scope=(
        ("cohort_manifest", "configs/distillation/x2_tennis_mixed_v3.yaml"),
        (
          "manifest_sha256",
          "4b73aca9450d8186bdf2c12ab375899ec4d1420e7efe9e705aa160fa260b057a",
        ),
        ("root_body", "pelvis"),
        ("anchor_body", "torso_link"),
        ("reference_dim", 68),
        ("conditioning_dim", 99),
        ("latent_dim", 32),
        ("action_dim", 31),
        ("reference_fps", 50),
      ),
      tier_probabilities=(("baseline", 0.6), ("moderate", 0.3), ("strong", 0.1)),
      root_reset_envelopes=tuple(roots.items()),
      reset_excitation_probabilities=(
        ("root_only", 0.4),
        ("joint_chain_only", 0.4),
        ("coupled", 0.2),
      ),
      root_direction_probabilities=(
        ("sagittal", 0.4),
        ("lateral", 0.4),
        ("yaw", 0.1),
        ("horizontal_translation", 0.1),
      ),
      joint_chain_probabilities=(
        ("left_sagittal", 0.25),
        ("right_sagittal", 0.25),
        ("left_lateral", 0.15),
        ("right_lateral", 0.15),
        ("waist", 0.2),
      ),
      joint_chains=_DEFAULT_CHAINS,
      joint_position_envelopes_rad=(
        ("baseline", (("all", 0.05),)),
        (
          "moderate",
          (
            ("hip_pitch", 0.1),
            ("knee", 0.1),
            ("ankle_pitch", 0.08),
            ("hip_roll", 0.08),
            ("ankle_roll", 0.08),
            ("waist_pitch_roll", 0.08),
            ("waist_yaw", 0.05),
          ),
        ),
        (
          "strong",
          (
            ("hip_pitch", 0.2),
            ("knee", 0.2),
            ("ankle_pitch", 0.15),
            ("hip_roll", 0.15),
            ("ankle_roll", 0.15),
            ("waist_pitch_roll", 0.15),
            ("waist_yaw", 0.1),
          ),
        ),
      ),
      joint_velocity_delta_envelopes_rad_s=(
        ("baseline", (("all", 0.0),)),
        ("moderate", (("selected_chain", 0.5),)),
        ("strong", (("selected_chain", 1.0), ("knee", 1.5))),
      ),
    )

  @classmethod
  def from_dict(cls, payload: object) -> CoveragePolicy:
    if not isinstance(payload, Mapping):
      raise CoveragePolicyError("coverage policy must be a mapping")
    allowed = set(cls().as_dict()) | {"contract_id"}
    unknown = set(payload) - allowed
    if unknown:
      raise CoveragePolicyError(f"unknown coverage policy fields: {sorted(unknown)}")
    mode = payload.get("mode", "off")
    if mode == "off":
      if payload in (
        {"mode": "off"},
        {"version": POLICY_VERSION, "mode": "off"},
        cls.off().as_dict(),
      ):
        return cls.off()
      # Off remains strict: accepting an enabled-looking partial policy could
      # accidentally change RNG/checkpoint semantics.
      raise CoveragePolicyError("off policy must contain only mode/version")
    if mode != "balance-recovery":
      raise CoveragePolicyError(f"invalid coverage mode {mode!r}")
    # The enabled contract is immutable P0 data; reject partial or altered data.
    expected = cls.balance_recovery()
    candidate = dict(payload)
    candidate.pop("contract_id", None)
    expected_payload = expected.as_dict()
    expected_payload.pop("contract_id", None)
    try:
      matches = canonical_json(candidate) == canonical_json(expected_payload)
    except (TypeError, ValueError) as exc:
      raise CoveragePolicyError(
        "enabled policy contains non-finite or non-serializable data"
      ) from exc
    if not matches:
      raise CoveragePolicyError(
        "enabled policy must exactly match confirmed P0 settings"
      )
    return expected

  @classmethod
  def from_json(cls, text: str) -> CoveragePolicy:
    try:
      return cls.from_dict(json.loads(text))
    except (json.JSONDecodeError, TypeError) as exc:
      raise CoveragePolicyError("coverage policy JSON is invalid") from exc


def canonical_json(value: object) -> str:
  """Canonical JSON helper shared by policy/metadata identity code."""
  serializer = getattr(value, "as_dict", None)
  payload = serializer() if callable(serializer) else value
  return json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
  )


def weighted_choice(rng: Any, weighted: Sequence[tuple[str, float]]) -> str:
  """Choose one named item without changing state when ``weighted`` is empty."""
  if not weighted:
    raise CoveragePolicyError("cannot choose from an empty weighted sequence")
  total = sum(_finite(weight, "weight") for _, weight in weighted)
  if total <= 0:
    raise CoveragePolicyError("weighted choices need positive total weight")
  draw = float(rng.random()) * total
  for name, weight in weighted:
    draw -= weight
    if draw < 0:
      return name
  return weighted[-1][0]


def choose_tier(policy: CoveragePolicy, rng: Any) -> CoverageTier:
  if not policy.enabled:
    return "baseline"
  return cast(CoverageTier, weighted_choice(rng, policy.tier_probabilities))


def choose_excitation(policy: CoveragePolicy, rng: Any) -> ExcitationKind:
  if not policy.enabled:
    return "root_only"
  return cast(
    ExcitationKind, weighted_choice(rng, policy.reset_excitation_probabilities)
  )


def choose_root_direction(
  policy: CoveragePolicy, rng: Any, *, runtime: bool = False
) -> RootDirection:
  if not policy.enabled:
    return "sagittal"
  choices = policy.root_direction_probabilities
  if runtime:
    choices = (("sagittal", 0.45), ("lateral", 0.45), ("yaw", 0.10))
  return cast(RootDirection, weighted_choice(rng, choices))


def choose_joint_chain(policy: CoveragePolicy, rng: Any) -> str:
  if not policy.enabled:
    return "left_sagittal"
  return weighted_choice(rng, policy.joint_chain_probabilities)


def draw_uniform(rng: Any, magnitude: float) -> float:
  """Symmetric draw; callers use this to make the baseline draw explicit."""
  return float(rng.uniform(-magnitude, magnitude))


# Explicit aliases make the narrow runtime/replay handoff discoverable without
# coupling either caller to the concrete classmethod spelling.
load_coverage_policy = CoveragePolicy.from_dict
make_coverage_policy = CoveragePolicy.balance_recovery


def coverage_policy_identity(policy: CoveragePolicy) -> str:
  return policy.identity
