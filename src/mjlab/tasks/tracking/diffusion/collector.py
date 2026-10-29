"""Frozen-policy rollout collection for the D1 offline dataset."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Sequence, cast

import numpy as np
import torch

from .adapter import (
  ActionContract,
  AdapterError,
  FrozenVaePolicy,
  PostStepBundle,
)
from .collection_controls import (
  STOP_OUTPUT_SIZE,
  STOP_TRANSITION_BUDGET,
  BoundedRowPublisher,
  CollectionControls,
  CollectionLimitWatcher,
  CollectionProgress,
  CollectionProgressTracker,
  CollectionSummary,
  missing_tensor_methods,
  stored_output_bytes,
)
from .policies import BatchedOUProcess, OUProcess, make_policy_action
from .qualification import (
  QualificationConfig,
  QualificationResult,
  QualificationTracker,
  parse_step_evidence,
  parse_step_evidence_batch,
)
from .storage import (
  AppendOnlyShardStore,
  TerminalEvidence,
  TrajectoryRow,
  validate_monotonic_timestamps,
)

_INITIAL_POSES: tuple[str, ...] = ("reference", "standing")
_INITIAL_STATE_MODES: tuple[str, ...] = ("perturbed", "reference")


class CollectorError(ValueError):
  """A collection lifecycle or row-assembly step is invalid."""


def _qualified_row_partition(
  rows: Sequence[TrajectoryRow],
  qualification: QualificationResult,
) -> tuple[
  tuple[TrajectoryRow, ...], tuple[TrajectoryRow, ...], tuple[TrajectoryRow, ...]
]:
  """Split one rollout's rows into accepted, verification and persisted rows.

  A rollout is one endpoint verdict, so a qualified rollout accepts every row,
  padding included, and a failed one accepts none.  Persisted rows keep row
  order and carry the accepted flag, so the store receives exactly the rows a
  caller sees in the result.
  """
  accepted_rows: list[TrajectoryRow] = (
    [replace(row, segment_qualified=True) for row in rows]
    if qualification.qualified
    else []
  )
  accepted_by_identity: dict[Any, TrajectoryRow] = {}
  for row in accepted_rows:
    accepted_by_identity.setdefault(row.identity, row)
  verification = tuple(row for row in rows if row.identity not in accepted_by_identity)
  persisted = tuple(accepted_by_identity.get(row.identity, row) for row in rows)
  return tuple(accepted_rows), verification, persisted


def _step_tensor_batch(
  env: Any,
  vae: FrozenVaePolicy,
  *,
  observation: Any,
  active_ids: np.ndarray,
  clean: np.ndarray,
  latent: np.ndarray,
  noise: np.ndarray,
) -> np.ndarray:
  """Run one batched step through the optional device tensor path.

  Every active row is a VAE row, so the caller passes the single device-native
  observation captured for this step and no second snapshot or CPU
  reference/conditioning copy is made.  The seeded NumPy OU noise is copied to
  that device, and the composed action returns to the host only because every
  active row records it.  ``clean`` and ``latent`` receive the CPU copies of the
  VAE device tensors.  The scalar NumPy path remains the reference
  implementation.
  """
  action_tensor = getattr(vae, "batch_action_tensor", None)
  if not callable(action_tensor):
    raise CollectorError("tensor dispatch lacks a VAE action method")
  latent_tensor, clean_tensor = cast(Any, action_tensor)(observation)
  if not isinstance(clean_tensor, torch.Tensor) or not isinstance(
    latent_tensor, torch.Tensor
  ):
    raise CollectorError("tensor VAE must return device tensors")
  if tuple(clean_tensor.shape) != (int(active_ids.size), 31) or tuple(
    latent_tensor.shape
  ) != (int(active_ids.size), 32):
    raise CollectorError("tensor VAE produced an invalid latent or action")
  if not bool(torch.isfinite(clean_tensor).all()) or not bool(
    torch.isfinite(latent_tensor).all()
  ):
    raise CollectorError("tensor VAE produced non-finite values")
  device = clean_tensor.device
  executed_tensor = torch.zeros(
    (clean.shape[0], 31), dtype=torch.float64, device=device
  )
  executed_tensor[active_ids] = clean_tensor.to(torch.float64) + torch.as_tensor(
    noise[active_ids], dtype=torch.float64, device=device
  )
  clean[active_ids] = clean_tensor.detach().to("cpu").numpy().astype(np.float64)
  latent[active_ids] = latent_tensor.detach().to("cpu").numpy().astype(np.float64)
  executed = executed_tensor.detach().to("cpu").numpy().astype(np.float64)
  if not np.isfinite(executed[active_ids]).all():
    raise CollectorError("batched policy produced non-finite actions")
  set_active = getattr(env, "set_active_env_ids", None)
  if callable(set_active):
    set_active(active_ids)
  step_tensor = getattr(env, "step_tensor_batch", None)
  if not callable(step_tensor):
    raise CollectorError("tensor dispatch lacks a step method")
  step_tensor(executed_tensor)
  return executed


@dataclass(frozen=True, slots=True)
class TrialSpec:
  """Deterministic trial identity and initialization provenance."""

  run_id: str
  motion_id: str
  seed: int
  start_frame: int
  phase: str = "clean"
  group_key: str = ""
  episode_id: str = ""
  pair_id: str = ""
  initial_state_id: str = ""
  initial_pose: str = "reference"
  initial_state_mode: str = "perturbed"

  def __post_init__(self) -> None:
    if not self.run_id or not self.motion_id or self.start_frame < 0:
      raise CollectorError("trial identity and start frame are invalid")
    if self.phase not in {"clean", "ou"}:
      raise CollectorError("trial phase must be clean or ou")
    if self.initial_pose not in _INITIAL_POSES:
      raise CollectorError(
        f"initial pose must be one of {_INITIAL_POSES}, got {self.initial_pose!r}"
      )
    if self.initial_state_mode not in _INITIAL_STATE_MODES:
      raise CollectorError(
        "initial state mode must be one of "
        f"{_INITIAL_STATE_MODES}, got {self.initial_state_mode!r}"
      )
    if self.episode_id == "":
      object.__setattr__(
        self,
        "episode_id",
        f"{self.run_id}:{self.motion_id}:{self.seed}:{self.start_frame}:"
        f"{self.initial_pose}:{self.initial_state_mode}:{self.phase}",
      )

  @property
  def perturb(self) -> bool:
    return self.initial_state_mode == "perturbed"

  def initialization_fingerprint(self) -> tuple[str, str, int, int, str, str]:
    """Immutable identity shared by clean and OU partners."""
    return (
      self.run_id,
      self.motion_id,
      self.seed,
      self.start_frame,
      self.pair_id,
      self.initial_state_id,
    )


@dataclass(frozen=True, slots=True)
class _InitialSnapshot:
  """A clean initial state and the identity/result that authorized its OU pair."""

  fingerprint: tuple[str, str, int, int, str, str]
  value: Any
  qualification: QualificationResult | None = None


@dataclass(frozen=True, slots=True)
class _PreStepRow:
  """CPU row evidence taken from one row of a pre-step observation batch.

  Only the fields a persisted row needs are kept, so the device tensor path can
  share one snapshot capture without copying the packed VAE reference and
  conditioning tensors back to the host.
  """

  state: Any
  motion_id: str
  reference_frame: int
  reference_phase: float


@dataclass(frozen=True, slots=True)
class CollectionResult:
  """Rows, verification trace and qualification for one attempted trial."""

  spec: TrialSpec
  qualification: QualificationResult
  rows: tuple[TrajectoryRow, ...]
  accepted_rows: tuple[TrajectoryRow, ...]
  verification_rows: tuple[TrajectoryRow, ...]
  store_files: tuple[str, ...] = ()
  stopped_reason: str | None = None

  @property
  def failed_attempt(self) -> bool:
    return not self.qualification.qualified


# Compact publication replaces rows with counts for the same trial verdict.
TrialOutcome = CollectionResult | CollectionSummary


@dataclass(frozen=True, slots=True)
class BatchedCollectionResult:
  """Results from one slot-scheduled collection and its store publications.

  ``stop_reason`` is set only when a limit or transition cap stopped scheduled
  work; ``partial`` reports exactly that.  ``unstarted_trials`` counts the
  admitted-queue trials that never ran because the run stopped early.  With
  streaming controls the results are full rows, or
  :class:`~mjlab.tasks.tracking.diffusion.collection_controls.CollectionSummary`
  records when the caller requested compact results.
  """

  results: tuple[TrialOutcome, ...]
  store_files: tuple[str, ...] = ()
  skipped_ou_pairs: tuple[str, ...] = ()
  stop_reason: str | None = None
  progress: CollectionProgress | None = None
  unstarted_trials: int = 0
  abandoned_trials: tuple[tuple[str, str], ...] = ()
  retried_pairs: tuple[str, ...] = ()

  @property
  def partial(self) -> bool:
    return self.stop_reason is not None

  @property
  def stored_bytes(self) -> int | None:
    """Measured stored output from the final progress record, if any."""
    return None if self.progress is None else self.progress.stored_bytes

  @property
  def output_limit_exceeded(self) -> bool:
    """True when the stored-output limit is the recorded stop reason."""
    return self.stop_reason == STOP_OUTPUT_SIZE


class TrialSlotScheduler:
  """Replace completed rows while enforcing each pair's clean admission."""

  def __init__(self, trials: Sequence[TrialSpec], slots: int) -> None:
    if slots <= 0:
      raise CollectorError("slot scheduler needs a positive slot count")
    self._pending = list(trials)
    self._slots = int(slots)
    self._active: dict[int, TrialSpec] = {}
    self._completed_clean: dict[str, TrialOutcome] = {}
    self._blocked_pairs: set[str] = set()
    self._results: dict[str, TrialOutcome] = {}
    self._skipped_ou_pairs: list[str] = []
    self._abandoned: dict[str, str] = {}
    self._retried_pairs: list[str] = []

  @property
  def active(self) -> dict[int, TrialSpec]:
    return dict(self._active)

  @property
  def pending(self) -> tuple[TrialSpec, ...]:
    return tuple(self._pending)

  @property
  def skipped_ou_pairs(self) -> tuple[str, ...]:
    return tuple(self._skipped_ou_pairs)

  @property
  def abandoned_trials(self) -> tuple[tuple[str, str], ...]:
    return tuple(self._abandoned.items())

  @property
  def retried_pairs(self) -> tuple[str, ...]:
    """Pairs re-run from the reference initial pose after a failure (D9)."""
    return tuple(self._retried_pairs)

  def abandon(self, slot: int, *, reason: str) -> TrialSpec:
    """Retire an admitted trial that cannot run and skip the rest of its pair.

    The frozen parity guard proves that this row cannot reproduce its clean
    partner's initial state.  Retiring only that pair keeps the rest of the
    schedule running instead of ending the whole collection; the pair and the
    reason stay visible in the run report.
    """
    spec = self._active.pop(slot, None)
    if spec is None:
      raise CollectorError(f"slot {slot} is not active")
    pair_id = spec.pair_id
    if pair_id:
      self._blocked_pairs.add(pair_id)
      kept: list[TrialSpec] = []
      skipped = 0
      for pending in self._pending:
        if pending.pair_id == pair_id and pending.phase == "ou":
          skipped += 1
        else:
          kept.append(pending)
      self._pending = kept
      if spec.phase == "ou":
        skipped += 1
      self._skipped_ou_pairs.extend([pair_id] * max(skipped, 1))
    self._abandoned[spec.episode_id] = reason
    return spec

  @property
  def results(self) -> tuple[TrialOutcome, ...]:
    return tuple(self._results.values())

  def _admissible(self, spec: TrialSpec) -> bool:
    if spec.phase == "clean":
      return not spec.pair_id or spec.pair_id not in self._completed_clean
    if spec.phase != "ou" or not spec.pair_id:
      return False
    clean = self._completed_clean.get(spec.pair_id)
    if clean is None:
      return False
    if not clean.qualification.qualified:
      return False
    return spec.pair_id not in self._blocked_pairs

  def fill(self) -> dict[int, TrialSpec]:
    """Fill empty slots and return only the newly admitted trials."""
    started: dict[int, TrialSpec] = {}
    for slot in range(self._slots):
      if slot in self._active:
        continue
      while True:
        candidate_index = next(
          (index for index, spec in enumerate(self._pending) if self._admissible(spec)),
          None,
        )
        if candidate_index is not None:
          spec = self._pending.pop(candidate_index)
          self._active[slot] = spec
          started[slot] = spec
          break
        blocked_index = next(
          (
            index
            for index, spec in enumerate(self._pending)
            if spec.phase == "ou"
            and spec.pair_id in self._completed_clean
            and not self._completed_clean[spec.pair_id].qualification.qualified
          ),
          None,
        )
        if blocked_index is None:
          break
        blocked = self._pending.pop(blocked_index)
        self._blocked_pairs.add(blocked.pair_id)
        self._skipped_ou_pairs.append(blocked.pair_id)
    return started

  def complete(
    self, slot: int, result: TrialOutcome, *, fill: bool = True
  ) -> dict[int, TrialSpec]:
    spec = self._active.pop(slot, None)
    if spec is None:
      raise CollectorError(f"slot {slot} is not active")
    if result.spec != spec:
      raise CollectorError("slot result does not match its admitted trial")
    self._results[spec.episode_id] = result
    if spec.phase == "clean" and spec.pair_id:
      self._completed_clean[spec.pair_id] = result
      if not result.qualification.qualified:
        self._blocked_pairs.add(spec.pair_id)
        self._enqueue_reference_retry(spec)
    return self.fill() if fill else {}

  def _enqueue_reference_retry(self, spec: TrialSpec) -> None:
    """Re-run a failed perturbed pair once from the unperturbed initial pose.

    Plan decision D9.  The retry is atomic over the pair: a new pair identity and
    a new initial-state identity keep its clean and OU rows apart from the
    failed attempt, while the group key stays the same so both attempts land in
    one split component.  A pair that already used the reference pose is not
    retried again.
    """
    if spec.initial_state_mode != "perturbed" or not spec.initial_pose:
      return
    retry_pair = f"{spec.pair_id}:retry"
    if retry_pair in self._completed_clean or any(
      pending.pair_id == retry_pair for pending in self._pending
    ):
      return
    retry_state = f"{spec.initial_state_id}:retry"
    retries = [
      replace(
        spec,
        phase=phase,
        pair_id=retry_pair,
        initial_state_id=retry_state,
        initial_state_mode="reference",
        episode_id="",
      )
      for phase in ("clean", "ou")
    ]
    self._pending.extend(retries)
    self._retried_pairs.append(spec.pair_id)

  def finish(self, *, allow_pending: bool = False) -> tuple[TrialOutcome, ...]:
    """Close the schedule and return its results.

    A schedule that never met its pair-local admission rule is an error unless
    the caller stopped the run early and accepts the unstarted trials.
    """
    if self._active:
      raise CollectorError("slot scheduler cannot finish with active trials")
    if self._pending and not allow_pending:
      raise CollectorError(
        "slot scheduler has trials that never met the pair-local admission rule"
      )
    return self.results


# Descriptive alias used by callers that name rows rather than slots.
SlotScheduler = TrialSlotScheduler


class DiffusionCollector:
  """Collect one VAE rollout per trial, from clip frame 0 to its last frame.

  The collector deliberately owns no automatic reset.  ``reset_after_trial`` is
  called only after every post-step evidence object has been converted into a
  row, so a reset state cannot masquerade as terminal survival.  Every rollout
  emits ``padding_steps`` synthetic history rows first, so its window index can
  place an anchor on the clip's own first frame.
  """

  def __init__(
    self,
    env: Any,
    vae: FrozenVaePolicy,
    *,
    vae_actions: ActionContract | None = None,
    qualification: QualificationConfig | None = None,
    store: AppendOnlyShardStore | None = None,
  ) -> None:
    self.env = env
    self.vae = vae
    self.vae_actions = vae_actions or vae.action_contract
    self.qualification = qualification or QualificationConfig()
    self.store = store
    self._initial_snapshots: dict[tuple[str, str], _InitialSnapshot] = {}
    self._padding_latents: dict[str, np.ndarray] = {}

  def _row_group_key(self, spec: TrialSpec, reference_frame: int) -> str:
    """The split group of one row: the clip, the seed and the anchor frame.

    The anchor is the row's own clip frame.  Both initial poses of a clip share
    this key, so they can never land in different splits, while the clean/OU
    pair identity carries the pose that separates two rollouts of one anchor.
    """
    return f"{spec.motion_id}:{spec.seed}:{int(reference_frame)}"

  def _row_provenance(
    self,
    spec: TrialSpec,
    *,
    reference_phase: str,
    finite: bool,
    reason: str,
  ) -> dict[str, str]:
    """The provenance every row of one rollout carries."""
    return {
      "phase": spec.phase,
      "reference_phase": reference_phase,
      "initial_seed": str(spec.seed),
      "initial_pose": spec.initial_pose,
      "initial_state_mode": spec.initial_state_mode,
      "pair_id": spec.pair_id,
      "initial_state_id": spec.initial_state_id,
      "group_key_generated": "true",
      "post_step_finite": str(finite),
      "post_step_reason": reason,
    }

  def _padding_rows(
    self, spec: TrialSpec, *, env_id: int = 0
  ) -> tuple[TrajectoryRow, ...]:
    """The synthetic history rows that pad the start of one rollout.

    The runtime supplies the standing keyframe and its identity-root state, so
    the padding is a pure function of the clip.  Its latent is the frozen
    encoder mean of that reference, encoded once per clip.
    """
    count = self.qualification.padding_steps
    if count == 0:
      return ()
    default = self.env.default_row(spec.motion_id)
    latent = self._padding_latents.get(spec.motion_id)
    if latent is None:
      latent = self.vae.encode_reference(default.reference)
      self._padding_latents[spec.motion_id] = latent
    zeros = np.zeros(31, dtype=np.float64)
    rows: list[TrajectoryRow] = []
    for index in range(count):
      reference_frame = index - count
      rows.append(
        TrajectoryRow(
          run_id=spec.run_id,
          env_id=env_id,
          episode_id=spec.episode_id,
          segment_id=f"{spec.episode_id}:vae",
          tick=index,
          timestamp=reference_frame * self.qualification.period_seconds,
          motion_id=spec.motion_id,
          reference_frame=reference_frame,
          controller="vae",
          state=default.state,
          previous_executed_action=zeros,
          clean_action=zeros,
          ou_noise=zeros,
          executed_action=zeros,
          latent=latent,
          terminal_evidence=TerminalEvidence(True, False, False, None, False, False),
          segment_qualified=False,
          fps=1.0 / self.qualification.period_seconds,
          group_key=self._row_group_key(spec, reference_frame),
          provenance=self._row_provenance(
            spec,
            reference_phase=f"{0.0:.9g}",
            finite=True,
            reason="padding_row",
          ),
        )
      )
    return tuple(rows)

  def collect(
    self, spec: TrialSpec, *, remaining_transitions: int | None = None
  ) -> CollectionResult:
    """Run one attempt without executing beyond the remaining transition cap."""
    if remaining_transitions is not None and remaining_transitions < 0:
      raise CollectorError("remaining transition budget must be non-negative")
    snapshot_key = (spec.pair_id, spec.initial_state_id)
    snapshot: _InitialSnapshot | None = None
    if spec.phase == "ou":
      if not spec.pair_id or not spec.initial_state_id:
        raise CollectorError("OU trial lacks clean pair/initial-state identity")
      snapshot = self._initial_snapshots.get(snapshot_key)
      if snapshot is None:
        raise CollectorError("OU trial snapshot was not captured by its clean partner")
      if snapshot.fingerprint != spec.initialization_fingerprint():
        raise CollectorError("OU trial identity disagrees with its clean partner")
      if snapshot.qualification is None:
        raise CollectorError("OU trial clean partner has no completed qualification")
      # A qualified clean partner is enough.  Requiring a *handoff* here conflated
      # the endpoint audit with the perturbation axis: the OU axis only needs a
      # clean rollout that survived its window, which is exactly what a
      # ``vae_only`` outcome means (see docs/plans/beyondmimic_diffusion_reproduction.md
      # and the paper's supplementary "Diffusion Dataset Collection").
      # Admission is pair-local.  The old global clean-before-OU rule protected
      # this process-local snapshot map by making every clean snapshot available
      # before any OU trial.  The snapshot and qualification checks above protect
      # the actual dependency, so unrelated pairs can overlap safely.
      if not snapshot.qualification.qualified:
        raise CollectorError(
          "OU trial is blocked because its clean partner did not qualify"
        )
    elif spec.phase == "clean":
      if spec.pair_id:
        if not spec.initial_state_id:
          raise CollectorError("clean trial lacks an initial-state snapshot identity")
        if snapshot_key in self._initial_snapshots:
          raise CollectorError("clean trial would overwrite an existing pair snapshot")
    try:
      self.env.reset(
        seed=spec.seed,
        motion_id=spec.motion_id,
        initial_pose=spec.initial_pose,
        perturb=spec.perturb,
      )
      if spec.phase == "ou":
        restore = getattr(self.env, "restore_initial_state", None)
        if not callable(restore):
          raise CollectorError("OU trial cannot restore its clean initial snapshot")
        assert snapshot is not None
        restore(snapshot.value)
      elif spec.pair_id:
        capture = getattr(self.env, "capture_initial_state", None)
        if not callable(capture):
          raise CollectorError("real adapter must provide initial-state snapshots")
        captured = capture()
        self._initial_snapshots[snapshot_key] = _InitialSnapshot(
          spec.initialization_fingerprint(), captured
        )
      ou = OUProcess(seed=spec.seed) if spec.phase == "ou" else None
      tracker = QualificationTracker(self.qualification)
      rows: list[TrajectoryRow] = list(self._padding_rows(spec))
      previous = np.zeros(31, dtype=np.float64)
      stopped_reason: str | None = None
    except BaseException:
      self.env.reset_after_trial()
      raise
    try:
      while not tracker.done:
        if remaining_transitions is not None and tracker.steps >= remaining_transitions:
          stopped_reason = "control_transition_budget"
          break
        observation = self.env.observe_vae()
        latent, clean = self.vae.action(observation)
        noise = ou.sample() if ou is not None else np.zeros(31, dtype=np.float64)
        action = make_policy_action(clean, noise)
        segment_id = f"{spec.episode_id}:vae"
        state = observation.state
        motion_id = observation.motion_id
        reference_frame = observation.reference_frame
        reference_phase = observation.reference_phase
        row_latent = latent

        result = self.env.step(action.executed.copy())
        raw_bundle = self.env.post_step_evidence(result)
        if isinstance(raw_bundle, PostStepBundle):
          bundle = raw_bundle
        elif isinstance(raw_bundle, dict) and "timestamp" in raw_bundle:
          bundle = PostStepBundle(
            raw_bundle.get("evidence", raw_bundle),
            raw_bundle["timestamp"],
            raw_bundle.get("observation", raw_bundle.get("post_observation")),
          )
        else:
          raise CollectorError("post-step evidence must include a physical timestamp")
        evidence = parse_step_evidence(bundle.evidence)
        final_evidence = tracker.observe(evidence, bundle.observation)
        row = TrajectoryRow(
          run_id=spec.run_id,
          env_id=0,
          episode_id=spec.episode_id,
          segment_id=segment_id,
          tick=self.qualification.padding_steps + tracker.steps - 1,
          timestamp=bundle.timestamp,
          motion_id=motion_id,
          reference_frame=reference_frame,
          controller="vae",
          state=state,
          previous_executed_action=previous,
          clean_action=action.clean,
          ou_noise=action.ou_noise,
          executed_action=action.executed,
          latent=row_latent,
          terminal_evidence=final_evidence.terminal,
          segment_qualified=False,
          reset=final_evidence.reset,
          teleport=final_evidence.teleport,
          # There is no control handoff any more, so the clip-end row is
          # ordinary training data: it is the last future target of the last
          # window of its segment.  The row keeps its terminal evidence.
          reference_boundary=False,
          fps=1.0 / self.qualification.period_seconds,
          group_key=self._row_group_key(spec, reference_frame),
          provenance=self._row_provenance(
            spec,
            reference_phase=f"{reference_phase:.9g}",
            finite=evidence.finite,
            reason=evidence.reason or "",
          ),
        )
        rows.append(row)
        previous = action.executed.copy()
      validate_monotonic_timestamps(rows)
      qualification = tracker.finish()
      if spec.phase == "clean" and spec.pair_id:
        current = self._initial_snapshots[(spec.pair_id, spec.initial_state_id)]
        self._initial_snapshots[(spec.pair_id, spec.initial_state_id)] = replace(
          current, qualification=qualification
        )
      accepted, verification, persisted = _qualified_row_partition(rows, qualification)
      if self.store is not None:
        store_files = self.store.append(persisted)
      else:
        store_files = ()
      return CollectionResult(
        spec,
        qualification,
        tuple(rows),
        accepted,
        verification,
        store_files,
        stopped_reason,
      )
    finally:
      self.env.reset_after_trial()

  def collect_batched(
    self,
    specs: Sequence[TrialSpec],
    *,
    slots: int | None = None,
    remaining_transitions: int | None = None,
    controls: CollectionControls | None = None,
  ) -> BatchedCollectionResult:
    """Collect independent trial slots with batched policy and evidence work.

    The scalar :meth:`collect` path remains the reference implementation.  This
    path only accepts an environment that exposes the explicit batch protocol.
    Every row is produced by the frozen VAE: there is no policy handoff, so the
    batch carries one policy for all slots.

    ``controls`` is optional and backward compatible.  Without it the call
    keeps full rows, appends once after the schedule, checks no limit and
    reports no progress.  With it a caller can coalesce finished-trial rows
    into bounded appends while keeping the pending buffer small, keep compact
    per-trial accounting, receive progress records, and stop before more physics
    work when a wall-clock, inactivity or stored-output limit is spent.
    """
    if not specs:
      return BatchedCollectionResult(())
    if remaining_transitions is not None and remaining_transitions < 0:
      raise CollectorError("remaining transition budget must be non-negative")
    controls = controls or CollectionControls()
    if controls.stream_rows and self.store is None:
      raise CollectorError("streaming row publication needs a collector store")
    if controls.tensor_dispatch:
      missing_tensor = missing_tensor_methods(self.env, self.vae)
      if missing_tensor:
        raise CollectorError(f"tensor dispatch is unavailable: {list(missing_tensor)}")
    measured_output = controls.output_bytes
    if measured_output is None and (
      controls.progress is not None
      or (controls.limits is not None and controls.limits.max_output_bytes is not None)
    ):
      measured_output = stored_output_bytes(self.store)
    if (
      controls.limits is not None
      and controls.limits.max_output_bytes is not None
      and measured_output is None
    ):
      raise CollectorError("output-size limit needs a measurable store output")
    watcher = CollectionLimitWatcher(
      controls.limits, output_bytes=measured_output, clock=controls.clock
    )
    reporter = CollectionProgressTracker(
      controls.progress,
      # Share the watcher's cached measurement: the store is walked once per
      # publication instead of once per progress event.
      output_bytes=(
        watcher.measured_output_bytes if measured_output is not None else None
      ),
    )
    publisher = (
      BoundedRowPublisher(self.store, max_rows_per_append=controls.max_rows_per_append)
      if controls.stream_rows
      else None
    )
    batch_size = int(slots or getattr(self.env, "num_envs", 0))
    if batch_size <= 0:
      raise CollectorError("batched collection needs a positive environment slot count")
    required = (
      "reset_batch",
      "observe_vae_batch",
      "default_row",
      "step_batch",
      "post_step_evidence_batch",
      "capture_initial_state_batch",
      "restore_initial_state_batch",
      "reset_after_trial_batch",
    )
    missing = [name for name in required if not callable(getattr(self.env, name, None))]
    if missing:
      raise CollectorError(f"environment does not implement batch protocol: {missing}")

    from .qualification import BatchedQualificationTracker

    scheduler = TrialSlotScheduler(specs, batch_size)
    tracker = BatchedQualificationTracker(batch_size, self.qualification)
    ou = BatchedOUProcess([0] * batch_size)
    previous = np.zeros((batch_size, 31), dtype=np.float64)
    active_specs: list[TrialSpec | None] = [None] * batch_size
    slot_rows: list[list[TrajectoryRow]] = [[] for _ in range(batch_size)]
    slot_latents = np.zeros((batch_size, 32), dtype=np.float64)
    snapshots: dict[tuple[str, str], Any] = {}
    result_map: dict[str, TrialOutcome] = {}
    persisted: list[TrajectoryRow] = []
    transitions = 0
    stop_reason: str | None = None
    # Scheduled OU trials per initial state: the snapshot of a pair is released
    # as soon as its last scheduled OU trial restored it.
    remaining_ou = Counter(
      (spec.pair_id, spec.initial_state_id) for spec in specs if spec.phase == "ou"
    )

    def _ids(values: Sequence[int]) -> np.ndarray:
      return np.asarray(tuple(values), dtype=np.int64)

    def _state_row(state: Any, index: int) -> Any:
      if state.root_position.ndim == 1:
        return state
      return type(state)(
        *(
          np.asarray(getattr(state, field))[index].copy()
          for field in (
            "root_position",
            "root_quaternion_wxyz",
            "root_linear_velocity",
            "root_angular_velocity",
            "body_positions",
            "body_linear_velocities",
          )
        )
      )

    def _row_evidence(observation: Any, local_index: int) -> _PreStepRow:
      return _PreStepRow(
        _state_row(observation.state, local_index),
        observation.motion_id[local_index],
        int(observation.reference_frame[local_index]),
        float(observation.reference_phase[local_index]),
      )

    def _finish_slot(slot: int, stopped_reason: str | None = None) -> TrialOutcome:
      spec = active_specs[slot]
      if spec is None:
        raise CollectorError(f"slot {slot} has no active trial")
      validate_monotonic_timestamps(slot_rows[slot])
      qualification = tracker.finish(slot)
      if stopped_reason is not None and qualification.qualified:
        qualification = replace(
          qualification, qualified=False, label="incomplete", reason=stopped_reason
        )
      rows = tuple(slot_rows[slot])
      # The slot buffer never outlives the publication of its trial.
      slot_rows[slot] = []
      accepted, verification, persisted_rows = _qualified_row_partition(
        rows, qualification
      )
      if publisher is None:
        persisted.extend(persisted_rows)
        files: tuple[str, ...] = ()
      else:
        before = publisher.stored_rows
        files = publisher.add(persisted_rows)
        written = publisher.stored_rows - before
        if written:
          reporter.note_append(rows=written)
        if files:
          watcher.note_publication()
      if controls.compact_results:
        result: TrialOutcome = CollectionSummary(
          spec,
          qualification,
          len(rows),
          len(accepted),
          len(verification),
          files,
          stopped_reason,
        )
      else:
        result = CollectionResult(
          spec,
          qualification,
          rows,
          accepted,
          verification,
          files,
          stopped_reason,
        )
      result_map[spec.episode_id] = result
      reporter.note_trial(steps=qualification.steps, accepted_rows=len(accepted))
      watcher.note_trial_completed()
      return result

    def _admit(started: dict[int, TrialSpec]) -> None:
      if not started:
        return
      slots_now = tuple(sorted(started))
      ids = _ids(slots_now)
      trial_specs = tuple(started[slot] for slot in slots_now)
      set_active = getattr(self.env, "set_active_env_ids", None)
      if callable(set_active):
        set_active(_ids(tuple(sorted(scheduler.active))))
      self.env.reset_batch(trial_specs, env_ids=ids)
      tracker.reset(ids)
      previous[ids] = 0.0
      ou.reset(ids, [spec.seed for spec in trial_specs])
      for slot, spec in started.items():
        active_specs[slot] = spec
        slot_rows[slot] = list(self._padding_rows(spec, env_id=slot))
        slot_latents[slot] = 0.0
        key = (spec.pair_id, spec.initial_state_id)
        if spec.phase == "clean":
          captured = self.env.capture_initial_state_batch(np.asarray([slot]))
          snapshots[key] = (
            captured[0] if isinstance(captured, (tuple, list)) else captured
          )
        else:
          if key not in snapshots:
            raise CollectorError(
              "OU trial was admitted without its clean partner snapshot"
            )
          try:
            self.env.restore_initial_state_batch(np.asarray([slot]), (snapshots[key],))
          except AdapterError as exc:
            # One pair that cannot reproduce its clean partner state is retired
            # instead of ending the whole worker.  The pair and the exact reason
            # stay in the run report, so the loss is visible and counted.
            scheduler.abandon(slot, reason=str(exc))
            active_specs[slot] = None
            slot_rows[slot] = []
            slot_latents[slot] = 0.0
            continue
          remaining_ou[key] -= 1
          if remaining_ou[key] <= 0:
            # No scheduled OU trial still needs this initial state.
            snapshots.pop(key, None)

    _admit(scheduler.fill())
    while scheduler.active:
      if controls.limits is not None:
        stop_reason = watcher.stop_reason()
        if stop_reason is not None:
          break
      active_ids = _ids(tuple(sorted(scheduler.active)))
      # The VAE observation supplies both the inference input and the row
      # identity/state evidence.  Tensor dispatch captures it once on the
      # policy device and carries the CPU row metadata with it, so the packed
      # reference/conditioning never make a host round trip for inference.
      if controls.tensor_dispatch:
        vae_observation = self.env.observe_vae_tensor_batch(active_ids)
        required_evidence = ("state", "motion_id", "reference_frame", "reference_phase")
        missing = [
          name
          for name in required_evidence
          if getattr(vae_observation, name, None) is None
        ]
        if missing:
          raise CollectorError(
            f"tensor observation lacks collection row evidence: {', '.join(missing)}"
          )
      else:
        vae_observation = self.env.observe_vae_batch(active_ids)
      clean = np.zeros((batch_size, 31), dtype=np.float64)
      latent = np.zeros((batch_size, 32), dtype=np.float64)
      noise = np.zeros((batch_size, 31), dtype=np.float64)
      if not controls.tensor_dispatch:
        latent_values, clean_values = self.vae.batch_action(vae_observation)
        clean[active_ids] = clean_values
        latent[active_ids] = latent_values
      # The OU perturbation runs on every step of an OU rollout.
      ou_ids = active_ids[
        np.asarray(
          [
            active_specs[int(slot)] is not None
            and active_specs[int(slot)].phase == "ou"
            for slot in active_ids
          ],
          dtype=bool,
        )
      ]
      if ou_ids.size:
        noise[ou_ids] = ou.sample(ou_ids.tolist())
      if controls.tensor_dispatch:
        executed = _step_tensor_batch(
          self.env,
          self.vae,
          observation=vae_observation,
          active_ids=active_ids,
          clean=clean,
          latent=latent,
          noise=noise,
        )
      else:
        executed = clean + noise
        if not np.isfinite(executed).all():
          raise CollectorError("batched policy produced non-finite actions")
        set_active = getattr(self.env, "set_active_env_ids", None)
        if callable(set_active):
          set_active(active_ids)
        self.env.step_batch(executed)
      bundle = self.env.post_step_evidence_batch(active_ids)
      if bundle.observation is None:
        raise CollectorError("batched post-step evidence lacks an observation")
      evidence = parse_step_evidence_batch(bundle.evidence)
      if evidence.batch_size != active_ids.size:
        raise CollectorError("batched evidence rows do not match active slots")
      parsed = tracker.observe(active_ids, evidence, bundle.observation)
      position = {int(slot): index for index, slot in enumerate(active_ids.tolist())}
      vae_position = {
        int(slot): index for index, slot in enumerate(active_ids.tolist())
      }
      for slot in active_ids.tolist():
        local = position[slot]
        if vae_observation is None:
          raise CollectorError("missing pre-step observation for an active slot")
        pre_row = _row_evidence(vae_observation, vae_position[slot])
        terminal = TerminalEvidence(
          bool(parsed.terminal_available[local]),
          bool(parsed.terminated[local]),
          bool(parsed.truncated[local]),
          parsed.reason[local],
          bool(parsed.physical_fall[local]),
          bool(parsed.tracking_rejection[local]),
        )
        spec = active_specs[slot]
        assert spec is not None
        row = TrajectoryRow(
          run_id=spec.run_id,
          env_id=slot,
          episode_id=spec.episode_id,
          segment_id=f"{spec.episode_id}:vae",
          tick=self.qualification.padding_steps + int(tracker.steps[slot] - 1),
          timestamp=float(bundle.timestamp[local]),
          motion_id=pre_row.motion_id,
          reference_frame=pre_row.reference_frame,
          controller="vae",
          state=pre_row.state,
          previous_executed_action=previous[slot].copy(),
          clean_action=clean[slot].copy(),
          ou_noise=noise[slot].copy(),
          executed_action=executed[slot].copy(),
          latent=latent[slot].copy(),
          terminal_evidence=terminal,
          segment_qualified=False,
          reset=bool(parsed.reset[local]),
          teleport=bool(parsed.teleport[local]),
          # The clip-end row is ordinary training data: it is the last future
          # target of the last window of its segment, and no control handoff
          # follows it.
          reference_boundary=False,
          fps=1.0 / self.qualification.period_seconds,
          group_key=self._row_group_key(spec, pre_row.reference_frame),
          provenance=self._row_provenance(
            spec,
            reference_phase=f"{pre_row.reference_phase:.9g}",
            finite=bool(parsed.finite[local]),
            reason=parsed.reason[local] or "",
          ),
        )
        slot_rows[slot].append(row)
        previous[slot] = executed[slot].copy()

      done = active_ids[tracker.done[active_ids]]
      transitions += int(active_ids.size)
      capped = (
        remaining_transitions is not None and transitions >= remaining_transitions
      )
      if capped:
        done = active_ids
      replacement_specs: dict[int, TrialSpec] = {}
      finished: list[TrialOutcome] = []
      for slot in done.tolist():
        stopped = (
          STOP_TRANSITION_BUDGET if capped and not tracker.endpoint[slot] else None
        )
        result = _finish_slot(slot, stopped)
        finished.append(result)
        started = scheduler.complete(slot, result, fill=not capped)
        self.env.reset_after_trial_batch(np.asarray([slot], dtype=np.int64))
        active_specs[slot] = None
        replacement_specs.update(started)
      for result in finished:
        reporter.emit(
          event="trial",
          stop_reason=result.stopped_reason,
          episode_id=result.spec.episode_id,
          files=result.store_files,
        )
      if capped:
        stop_reason = STOP_TRANSITION_BUDGET
        break
      _admit(replacement_specs)

    if scheduler.active:
      # A limit or the optional transition cap ends every remaining slot without
      # adding a post-stop simulator step; the rows already recorded stay valid.
      tail_reason = stop_reason or STOP_TRANSITION_BUDGET
      for slot in tuple(sorted(scheduler.active)):
        result = _finish_slot(slot, tail_reason)
        scheduler.complete(slot, result, fill=False)
        self.env.reset_after_trial_batch(np.asarray([slot], dtype=np.int64))
        active_specs[slot] = None
        reporter.emit(
          event="trial",
          stop_reason=tail_reason,
          episode_id=result.spec.episode_id,
          files=result.store_files,
        )
      stop_reason = tail_reason
    # A limit that stopped nothing is not a stop: the schedule completed.
    cut_trials = sum(
      result.stopped_reason is not None for result in result_map.values()
    )
    if stop_reason is not None and cut_trials == 0 and not scheduler.pending:
      stop_reason = None
    # No scheduled OU trial can use an initial state after the run stops.
    snapshots.clear()
    scheduler.finish(allow_pending=stop_reason is not None)
    ordered = tuple(
      result_map[spec.episode_id] for spec in specs if spec.episode_id in result_map
    )
    store_files: tuple[str, ...] = ()
    if publisher is not None:
      before = publisher.stored_rows
      final_files = publisher.flush()
      written = publisher.stored_rows - before
      if written:
        reporter.note_append(rows=written)
        watcher.note_publication()
      store_files = tuple(publisher.files)
      if final_files and ordered:
        last = ordered[-1]
        ordered = (
          *ordered[:-1],
          replace(last, store_files=tuple(last.store_files) + final_files),
        )
    elif self.store is not None and persisted:
      store_files = self.store.append(persisted)
      reporter.note_append(rows=len(persisted))
      watcher.note_publication()
      if store_files and ordered:
        last = ordered[-1]
        ordered = (*ordered[:-1], replace(last, store_files=store_files))
    # The final publication can push the store over the declared budget after
    # the schedule already ended.  Keep that stop verdict instead of reporting
    # a completed collection.
    if (
      stop_reason is None
      and controls.limits is not None
      and controls.limits.max_output_bytes is not None
    ):
      overage = watcher.output_stop()
      if overage is not None:
        stop_reason = overage
    unstarted_trials = len(scheduler.pending) if stop_reason is not None else 0
    progress = reporter.emit(event="final", stop_reason=stop_reason, files=store_files)
    return BatchedCollectionResult(
      ordered,
      store_files,
      scheduler.skipped_ou_pairs,
      stop_reason,
      progress,
      unstarted_trials,
      scheduler.abandoned_trials,
      scheduler.retried_pairs,
    )


__all__ = [
  "BatchedCollectionResult",
  "CollectionResult",
  "CollectionSummary",
  "CollectorError",
  "DiffusionCollector",
  "SlotScheduler",
  "TrialOutcome",
  "TrialSlotScheduler",
  "TrialSpec",
]
