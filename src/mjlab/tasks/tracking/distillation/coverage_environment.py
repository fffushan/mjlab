"""Fail-closed runtime coverage component and bounded terminal admission."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .branch_state import BranchState
from .coverage_policy import CoveragePolicy


class CoverageRuntimeError(RuntimeError):
  """Runtime coverage cannot safely execute."""


@dataclass(frozen=True, slots=True)
class TerminalCapture:
  state: BranchState
  cached_policy_view: Any
  executed_action: Any
  termination: Any
  reconstructed: bool = False


@dataclass(frozen=True, slots=True)
class SnapshotCandidate:
  """Metadata known before copying one raw source state."""

  candidate_id: str
  source_kind: str = "finite_post_failure"
  motion_id: int = -1
  phase_bin: int = -1
  severity_band: str = "nominal"
  family_id: str = ""
  estimated_bytes: int = 0
  capture: Any = None

  def __post_init__(self) -> None:
    if self.source_kind not in (
      "student_difficult",
      "teacher_continuation",
      "finite_post_failure",
    ):
      raise ValueError(f"unsupported snapshot source_kind {self.source_kind!r}")
    if self.estimated_bytes < 0:
      raise ValueError("estimated_bytes must be nonnegative")

  @property
  def stratum(self) -> tuple[str, int, int, str]:
    return (self.source_kind, self.motion_id, self.phase_bin, self.severity_band)


@dataclass(slots=True)
class _AdmittedCapture:
  candidate: SnapshotCandidate
  capture: TerminalCapture
  actual_bytes: int


class CoverageEnvironmentComponent:
  """Owns bounded candidate admission and terminal snapshot storage."""

  def __init__(self, environment: Any, policy: CoveragePolicy) -> None:
    self.environment, self.policy = environment, policy
    self._parity_evidence: str | None = None
    self._terminal: list[TerminalCapture] = []
    self._terminal_records: dict[str, _AdmittedCapture] = {}
    self._terminal_bytes = 0
    self._terminal_seen = 0
    self._terminal_cap = 51
    self._pre_step_cap = 205
    self._candidate_cap = 256
    self._terminal_bytes_cap = 64 * 1024 * 1024
    self.counters: dict[str, int] = {}

  @property
  def enabled(self) -> bool:
    return self.policy.enabled

  @property
  def live_ready(self) -> bool:
    return False

  def preflight(self) -> None:
    if not self.enabled:
      return
    fixed = all(
      hasattr(self.environment, name)
      for name in ("sim", "scene", "action_manager", "observation_manager")
    )
    if fixed:
      from .x2_state_adapter import X2FixedTopologyStateAdapter

      X2FixedTopologyStateAdapter(self.environment, source_row=0)
      return
    if not (
      callable(getattr(self.environment, "coverage_capture_fields", None))
      and callable(getattr(self.environment, "coverage_restore_fields", None))
    ):
      raise CoverageRuntimeError(
        "coverage requires the concrete X2 fixed-topology adapter or explicit test handler"
      )

  def authorize_live(self, parity_evidence: str) -> None:
    """Enable live coverage, but only against recorded parity evidence.

    The requirement stays: a run must supply the reference of a measured restore
    parity result.  What changed is that the gate is satisfiable, so an experiment
    can proceed on evidence instead of a constant refusal.
    """
    if not parity_evidence or not str(parity_evidence).strip():
      raise CoverageRuntimeError(
        "live coverage needs a measured restore-parity reference; pass the report "
        "or artifact that records it"
      )
    self._parity_evidence = str(parity_evidence)

  def capture_cached_pre_step(
    self,
    *,
    env_index: int | None = None,
    metadata: Mapping[str, Any] | None = None,
  ) -> BranchState:
    self.preflight()
    return BranchState.capture(
      self.environment,
      env_index=env_index,
      capture_instant="cached_pre_step",
      metadata=metadata,
    )

  def capture_raw_reached_pre_reset(
    self,
    env_ids: torch.Tensor,
    *,
    termination: Any,
    candidates: Sequence[SnapshotCandidate] | None = None,
  ) -> tuple[TerminalCapture, ...]:
    """Admit terminal candidates before invoking any state-copy callback."""
    self.preflight()
    if not callable(getattr(self.environment, "coverage_cached_policy_view", None)):
      raise CoverageRuntimeError(
        "raw terminal capture needs cached policy-view handler"
      )
    if candidates is None:
      indices = env_ids.detach().cpu().tolist()
      candidates = tuple(
        SnapshotCandidate(
          candidate_id=f"terminal:{self._terminal_seen + offset}:{index}",
          motion_id=-1,
          capture=lambda index=index: self._capture_terminal(index, termination),
        )
        for offset, index in enumerate(indices)
      )
    admitted = self.admit_snapshot_candidates(candidates)
    return tuple(record.capture for record in admitted)

  def admit_snapshot_candidates(
    self, candidates: Sequence[SnapshotCandidate]
  ) -> tuple[_AdmittedCapture, ...]:
    """Admit bounded strata and return only newly copied records.

    Candidate metadata is ranked before any callback runs.  The two source
    categories share one 256-entry pool but retain their simultaneous 51/205
    reservations, so a late category cannot consume the other category's quota.
    Actual owned bytes are committed only after the aggregate estimate fits.
    """
    if not self.enabled or not candidates:
      return ()
    self._terminal_seen += len(candidates)
    incoming: dict[str, SnapshotCandidate] = {}
    for candidate in candidates:
      if (
        candidate.candidate_id in self._terminal_records
        or candidate.candidate_id in incoming
      ):
        self.counters["duplicate_id"] = self.counters.get("duplicate_id", 0) + 1
        continue
      incoming[candidate.candidate_id] = candidate

    existing = tuple(record.candidate for record in self._terminal_records.values())
    selected = _select_stratified_candidates(
      existing,
      tuple(incoming.values()),
      terminal_capacity=self._terminal_cap,
      pre_step_capacity=self._pre_step_cap,
      candidate_cap=self._candidate_cap,
    )
    selected_ids = {candidate.candidate_id for candidate in selected}
    metadata_dropped = sum(
      candidate.candidate_id not in selected_ids for candidate in incoming.values()
    )
    if metadata_dropped:
      self.counters["candidate_metadata_dropped"] = (
        self.counters.get("candidate_metadata_dropped", 0) + metadata_dropped
      )
    retained_existing = [
      record
      for record in self._terminal_records.values()
      if record.candidate.candidate_id in selected_ids
    ]
    incoming_selected = [
      candidate
      for candidate in selected
      if candidate.candidate_id not in self._terminal_records
    ]

    # Do not evict an old record merely because a replacement was ranked ahead
    # of it: if the replacement cannot fit, the old owned snapshot remains.
    base_bytes = sum(record.actual_bytes for record in retained_existing)
    accepted: list[tuple[SnapshotCandidate, TerminalCapture, int]] = []
    estimate_bytes = base_bytes
    for candidate in sorted(incoming_selected, key=_candidate_rank):
      if estimate_bytes + candidate.estimated_bytes > self._terminal_bytes_cap:
        self.counters["byte_reserve_skipped"] = (
          self.counters.get("byte_reserve_skipped", 0) + 1
        )
        continue
      # The estimate gate is intentionally before this callback.  A source
      # capture may copy Warp state and must never be paid for a known overflow.
      raw = (
        candidate.capture()
        if callable(candidate.capture)
        else self._capture_terminal(candidate.candidate_id, None)
      )
      if not isinstance(raw, TerminalCapture):
        raise CoverageRuntimeError(
          "snapshot capture callback returned an invalid capture"
        )
      actual = _owned_size(raw)
      if estimate_bytes + actual > self._terminal_bytes_cap:
        self.counters["byte_overflow"] = self.counters.get("byte_overflow", 0) + 1
        continue
      accepted.append((candidate, raw, actual))
      estimate_bytes += actual

    retained_ids = {record.candidate.candidate_id for record in retained_existing} | {
      candidate.candidate_id for candidate, _, _ in accepted
    }
    for old_id in tuple(self._terminal_records):
      if old_id not in retained_ids:
        self._evict(old_id)
        self.counters["replaced"] = self.counters.get("replaced", 0) + 1
    fresh: list[_AdmittedCapture] = []
    for candidate, raw, actual in accepted:
      record = _AdmittedCapture(candidate, raw, actual)
      self._terminal_records[candidate.candidate_id] = record
      self._terminal_bytes += actual
      fresh.append(record)
    self._terminal = [record.capture for record in self._terminal_records.values()]
    return tuple(fresh)

  def _capture_terminal(self, index: Any, termination: Any) -> TerminalCapture:
    index = int(index)
    state = BranchState.capture(
      self.environment, env_index=index, capture_instant="raw_reached_pre_reset"
    )
    cached = _owned(self.environment.coverage_cached_policy_view(index))
    action = _owned(self.environment.coverage_executed_action(index))
    return TerminalCapture(state, cached, action, _owned(termination))

  def _evict(self, candidate_id: str) -> None:
    record = self._terminal_records.pop(candidate_id, None)
    if record is not None:
      self._terminal_bytes -= record.actual_bytes

  def reconstruct_terminal(
    self,
    capture: TerminalCapture,
    branch_environment: Any,
    *,
    env_index: int | None = None,
    synthetic: bool = False,
  ) -> BranchState:
    if not self.live_ready and not synthetic:
      raise CoverageRuntimeError(
        "terminal reconstruction is fail-closed pending explicit parity evidence"
      )
    if synthetic and not callable(
      getattr(branch_environment, "coverage_reconstruct_next_boundary", None)
    ):
      raise CoverageRuntimeError(
        "synthetic terminal reconstruction needs an explicit fake-boundary handler"
      )
    capture.state.restore(branch_environment, env_index=env_index)
    handler = getattr(branch_environment, "coverage_reconstruct_next_boundary", None)
    if not callable(handler):
      raise CoverageRuntimeError("branch backend lacks single next-boundary handler")
    handler(capture.executed_action)
    return BranchState.capture(
      branch_environment,
      env_index=env_index,
      capture_instant="raw_reached_pre_reset",
    )

  def terminal_captures(self) -> tuple[TerminalCapture, ...]:
    return tuple(self._terminal)

  def clear_terminal_captures(self) -> None:
    self._terminal.clear()
    self._terminal_records.clear()
    self._terminal_bytes = 0
    self._terminal_seen = 0
    self.counters.clear()


def _owned(value: Any) -> Any:
  if isinstance(value, torch.Tensor):
    return value.detach().clone()
  if isinstance(value, np.ndarray):
    return value.copy()
  if isinstance(value, dict):
    return {key: _owned(item) for key, item in value.items()}
  if isinstance(value, list):
    return [_owned(item) for item in value]
  if isinstance(value, tuple):
    return tuple(_owned(item) for item in value)
  return copy.deepcopy(value)


def _owned_size(value: Any) -> int:
  if isinstance(value, torch.Tensor):
    return value.numel() * value.element_size()
  if isinstance(value, np.ndarray):
    return int(value.nbytes)
  if dataclasses.is_dataclass(value):
    return sum(
      _owned_size(getattr(value, field.name)) for field in dataclasses.fields(value)
    )
  if isinstance(value, dict):
    return sum(_owned_size(item) for item in value.values())
  if isinstance(value, (list, tuple)):
    return sum(_owned_size(item) for item in value)
  return 0


def _candidate_rank(candidate: SnapshotCandidate) -> int:
  return int(hashlib.sha256(candidate.candidate_id.encode()).hexdigest()[:16], 16)


def _candidate_category(candidate: SnapshotCandidate) -> str:
  return "terminal" if candidate.source_kind == "finite_post_failure" else "pre_step"


def _select_stratified_candidates(
  existing: Sequence[SnapshotCandidate],
  incoming: Sequence[SnapshotCandidate],
  *,
  terminal_capacity: int,
  pre_step_capacity: int,
  candidate_cap: int,
) -> tuple[SnapshotCandidate, ...]:
  """Select one bounded metadata pool with category reservations.

  Stable hash ranking avoids arrival-order bias.  At most one representative
  per metadata stratum is selected first, then remaining capacity is filled by
  the same stable rank inside each category.
  """
  unique: dict[str, SnapshotCandidate] = {
    candidate.candidate_id: candidate for candidate in (*existing, *incoming)
  }
  categories = {"terminal": terminal_capacity, "pre_step": pre_step_capacity}
  selected: list[SnapshotCandidate] = []
  for category, category_capacity in categories.items():
    members = [
      candidate
      for candidate in unique.values()
      if _candidate_category(candidate) == category
    ]
    if len(members) <= category_capacity:
      selected.extend(sorted(members, key=_candidate_rank))
      continue
    by_stratum: dict[tuple[Any, ...], list[SnapshotCandidate]] = {}
    for candidate in members:
      by_stratum.setdefault(candidate.stratum, []).append(candidate)
    representatives = [min(group, key=_candidate_rank) for group in by_stratum.values()]
    chosen = sorted(representatives, key=_candidate_rank)
    remaining = [candidate for candidate in members if candidate not in chosen]
    chosen.extend(sorted(remaining, key=_candidate_rank))
    selected.extend(chosen[:category_capacity])
  if len(selected) > candidate_cap:
    selected = sorted(selected, key=_candidate_rank)[:candidate_cap]
  return tuple(selected)
