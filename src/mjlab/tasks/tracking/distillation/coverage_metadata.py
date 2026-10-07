"""Owned, outcome-independent provenance for difficult-state replay rows.

Metadata is intentionally separate from model tensors.  It identifies the source
trajectory, family, branch and trigger so runtime and replay can share one stable
row contract without exposing teacher outcomes or latent values to the model.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

CoverageSource = Literal[
  "student_difficult", "teacher_continuation", "finite_post_failure"
]
CoverageTrigger = Literal[
  "threshold_crossing", "scheduled", "physical_termination", "manual"
]
COVERAGE_METADATA_VERSION = 1


class CoverageMetadataError(ValueError):
  """Raised when row provenance is malformed or not safely owned."""


def _required_text(value: object, name: str) -> str:
  if not isinstance(value, str) or not value:
    raise CoverageMetadataError(f"{name} must be a non-empty string")
  return value


def _finite(value: object, name: str, *, nonnegative: bool = False) -> float:
  if (
    isinstance(value, bool)
    or not isinstance(value, (int, float))
    or not math.isfinite(float(value))
  ):
    raise CoverageMetadataError(f"{name} must be finite")
  result = float(value)
  if nonnegative and result < 0:
    raise CoverageMetadataError(f"{name} must be nonnegative")
  return result


def _integer(value: object, name: str, *, nonnegative: bool = True) -> int:
  if (
    isinstance(value, bool) or not isinstance(value, int) or (nonnegative and value < 0)
  ):
    raise CoverageMetadataError(
      f"{name} must be a {'nonnegative' if nonnegative else ''} integer"
    )
  return value


def _canonical(payload: object) -> str:
  return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_family_id(*, motion_id: str, trajectory_id: str) -> str:
  """Stable family namespace shared by a student trajectory and its branches."""
  return hashlib.sha256(
    _canonical(
      {
        "domain": "x2-coverage-family-v1",
        "motion_id": motion_id,
        "trajectory_id": trajectory_id,
      }
    ).encode()
  ).hexdigest()[:32]


def stable_row_id(
  *, family_id: str, source_kind: CoverageSource, frame: int, ordinal: int
) -> str:
  """Derive a stable row ID from source identity, never from model output."""
  return hashlib.sha256(
    _canonical(
      {
        "family_id": family_id,
        "source_kind": source_kind,
        "frame": frame,
        "ordinal": ordinal,
      }
    ).encode()
  ).hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class CoverageRowMetadata:
  """Immutable provenance for one fresh or archived row."""

  row_id: str
  family_id: str
  source_kind: CoverageSource
  motion_id: str
  source_branch: Literal["main", "teacher_branch"] = "main"
  branch_id: str | None = None
  source_frame: int = 0
  phase_bin: int = 0
  severity_score: float = 0.0
  severity_band: Literal["nominal", "moderate", "strong"] = "nominal"
  perturbation_tier: Literal["baseline", "moderate", "strong"] = "baseline"
  trigger: CoverageTrigger = "manual"
  trigger_reason: str = ""
  branch_age_steps: int = 0
  terminal: bool = False
  first_failure: bool = False
  event_frame: int | None = None

  def __post_init__(self) -> None:
    _required_text(self.row_id, "row_id")
    _required_text(self.family_id, "family_id")
    _required_text(self.motion_id, "motion_id")
    if self.source_kind not in (
      "student_difficult",
      "teacher_continuation",
      "finite_post_failure",
    ):
      raise CoverageMetadataError(f"invalid source_kind {self.source_kind!r}")
    if self.source_branch not in ("main", "teacher_branch"):
      raise CoverageMetadataError(f"invalid source_branch {self.source_branch!r}")
    if (
      self.source_kind == "teacher_continuation"
      and self.source_branch != "teacher_branch"
    ):
      raise CoverageMetadataError(
        "teacher_continuation rows must identify a teacher branch"
      )
    if self.source_branch == "teacher_branch" and not self.branch_id:
      raise CoverageMetadataError("teacher branch rows require branch_id")
    if self.phase_bin < 0 or self.source_frame < 0 or self.branch_age_steps < 0:
      raise CoverageMetadataError("frame, phase and branch age must be nonnegative")
    _finite(self.severity_score, "severity_score", nonnegative=True)
    if self.severity_band not in ("nominal", "moderate", "strong"):
      raise CoverageMetadataError(f"invalid severity_band {self.severity_band!r}")
    if self.perturbation_tier not in ("baseline", "moderate", "strong"):
      raise CoverageMetadataError(
        f"invalid perturbation_tier {self.perturbation_tier!r}"
      )
    if self.trigger not in (
      "threshold_crossing",
      "scheduled",
      "physical_termination",
      "manual",
    ):
      raise CoverageMetadataError(f"invalid trigger {self.trigger!r}")
    if self.event_frame is not None and self.event_frame < 0:
      raise CoverageMetadataError("event_frame must be nonnegative")

  def as_dict(self) -> dict[str, Any]:
    return {
      "version": COVERAGE_METADATA_VERSION,
      "row_id": self.row_id,
      "family_id": self.family_id,
      "source_kind": self.source_kind,
      "motion_id": self.motion_id,
      "source_branch": self.source_branch,
      "branch_id": self.branch_id,
      "source_frame": self.source_frame,
      "phase_bin": self.phase_bin,
      "severity_score": self.severity_score,
      "severity_band": self.severity_band,
      "perturbation_tier": self.perturbation_tier,
      "trigger": self.trigger,
      "trigger_reason": self.trigger_reason,
      "branch_age_steps": self.branch_age_steps,
      "terminal": self.terminal,
      "first_failure": self.first_failure,
      "event_frame": self.event_frame,
    }

  @property
  def digest(self) -> str:
    return hashlib.sha256(_canonical(self.as_dict()).encode()).hexdigest()

  @classmethod
  def from_dict(cls, payload: object) -> CoverageRowMetadata:
    if not isinstance(payload, Mapping):
      raise CoverageMetadataError("row metadata must be a mapping")
    if payload.get("version") != COVERAGE_METADATA_VERSION:
      raise CoverageMetadataError("unsupported row metadata version")
    expected = set(cls.__dataclass_fields__) | {"version"}
    unknown = set(payload) - expected
    missing = {field for field in cls.__dataclass_fields__ if field not in payload}
    if unknown or missing:
      raise CoverageMetadataError(
        f"row metadata fields mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}"
      )
    values = dict(payload)
    values.pop("version")
    return cls(**values)


@dataclass(frozen=True, slots=True)
class CoverageBatchMetadata:
  """Owned metadata batch with exactly-once row IDs and stable serialization."""

  rows: tuple[CoverageRowMetadata, ...]
  batch_id: str = ""

  def __post_init__(self) -> None:
    if not isinstance(self.rows, tuple):
      object.__setattr__(self, "rows", tuple(self.rows))
    if not all(isinstance(row, CoverageRowMetadata) for row in self.rows):
      raise CoverageMetadataError("rows must contain CoverageRowMetadata")
    ids = [row.row_id for row in self.rows]
    if len(set(ids)) != len(ids):
      raise CoverageMetadataError("row IDs must be unique within a batch")
    if self.batch_id:
      _required_text(self.batch_id, "batch_id")
    else:
      object.__setattr__(
        self,
        "batch_id",
        hashlib.sha256(
          _canonical([row.as_dict() for row in self.rows]).encode()
        ).hexdigest()[:32],
      )

  def __len__(self) -> int:
    return len(self.rows)

  def __iter__(self):
    return iter(self.rows)

  def as_dict(self) -> dict[str, Any]:
    return {
      "version": COVERAGE_METADATA_VERSION,
      "batch_id": self.batch_id,
      "rows": [row.as_dict() for row in self.rows],
    }

  def canonical_json(self) -> str:
    return _canonical(self.as_dict())

  @property
  def digest(self) -> str:
    return hashlib.sha256(self.canonical_json().encode()).hexdigest()

  @classmethod
  def from_dict(cls, payload: object) -> CoverageBatchMetadata:
    if (
      not isinstance(payload, Mapping)
      or payload.get("version") != COVERAGE_METADATA_VERSION
    ):
      raise CoverageMetadataError("unsupported batch metadata")
    if set(payload) != {"version", "batch_id", "rows"} or not isinstance(
      payload["rows"], Sequence
    ):
      raise CoverageMetadataError("batch metadata fields are malformed")
    batch_id = payload["batch_id"]
    if not isinstance(batch_id, str) or not batch_id:
      raise CoverageMetadataError("batch_id must be a non-empty string")
    return cls(
      tuple(CoverageRowMetadata.from_dict(row) for row in payload["rows"]),
      batch_id,
    )


def make_row_metadata(
  *,
  motion_id: str,
  trajectory_id: str,
  source_kind: CoverageSource,
  frame: int,
  ordinal: int,
  **kwargs: Any,
) -> CoverageRowMetadata:
  """Construct metadata while deriving IDs from the owned source identity."""
  family_id = stable_family_id(motion_id=motion_id, trajectory_id=trajectory_id)
  return CoverageRowMetadata(
    row_id=stable_row_id(
      family_id=family_id, source_kind=source_kind, frame=frame, ordinal=ordinal
    ),
    family_id=family_id,
    source_kind=source_kind,
    motion_id=motion_id,
    source_frame=frame,
    **kwargs,
  )


def own_metadata(rows: Iterable[CoverageRowMetadata]) -> CoverageBatchMetadata:
  """Copy an iterable into an immutable batch before it crosses a runtime seam."""
  return CoverageBatchMetadata(tuple(rows))


def compose_optional_metadata(
  rows: Sequence[CoverageRowMetadata] | None,
) -> CoverageBatchMetadata | None:
  """Compatibility adapter: ``None`` keeps legacy batch constructors unchanged."""
  return None if rows is None else own_metadata(rows)

def namespace_coverage_rows(
  rows: Sequence[CoverageRowMetadata], *, namespace: str
) -> tuple[CoverageRowMetadata, ...]:
  """Return rows whose local identifiers carry a namespace.

  Two shards can derive the same local identifier from the same clip, so a merged
  store needs one namespace per reply.  The parent applies this function once per
  reply, before any insert, and the result is unique across workers and across a
  resume.  The function raises on a repeated namespaced row identifier.
  """
  prefix = _required_text(namespace, "namespace")
  if "/" in prefix or prefix != prefix.strip():
    raise CoverageMetadataError("namespace must be a token without a slash")
  namespaced: list[CoverageRowMetadata] = []
  seen: set[str] = set()
  for row in rows:
    updated = replace(
      row,
      row_id=f"{prefix}/{row.row_id}",
      family_id=f"{prefix}/{row.family_id}",
      branch_id=None if row.branch_id is None else f"{prefix}/{row.branch_id}",
    )
    if updated.row_id in seen:
      raise CoverageMetadataError("namespaced row id is not unique")
    seen.add(updated.row_id)
    namespaced.append(updated)
  return tuple(namespaced)


def namespace_coverage_batch(
  metadata: CoverageBatchMetadata, *, namespace: str
) -> CoverageBatchMetadata:
  """Namespace a whole batch.

  ``CoverageBatchMetadata`` derives its own batch id from the rows it holds, so this
  function namespaces the rows only and lets the metadata compute the new id.  Do not
  prefix the stored batch id by hand: the metadata rejects an id that its own rows do
  not produce.
  """
  return CoverageBatchMetadata(rows=namespace_coverage_rows(metadata.rows, namespace=namespace))
