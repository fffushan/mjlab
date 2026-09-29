"""Single-environment frozen-policy collection and handoff lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .adapter import (
  ActionContract,
  DiffusionEnvironment,
  FrozenPpoRecoveryPolicy,
  FrozenVaePolicy,
  PostStepBundle,
)
from .policies import OUProcess, make_policy_action
from .qualification import (
  QualificationConfig,
  QualificationResult,
  QualificationTracker,
  parse_step_evidence,
)
from .storage import (
  AppendOnlyShardStore,
  TrajectoryRow,
  validate_monotonic_timestamps,
)


class CollectorError(ValueError):
  """A collection lifecycle or policy handoff is invalid."""


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

  def __post_init__(self) -> None:
    if not self.run_id or not self.motion_id or self.start_frame < 0:
      raise CollectorError("trial identity and start frame are invalid")
    if self.phase not in {"clean", "ou"}:
      raise CollectorError("trial phase must be clean or ou")
    if self.episode_id == "":
      object.__setattr__(
        self,
        "episode_id",
        f"{self.run_id}:{self.motion_id}:{self.seed}:{self.start_frame}:{self.phase}",
      )

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


class DiffusionCollector:
  """Collect a bounded VAE interval and, at a clip boundary, PPO recovery.

  The collector deliberately owns no automatic reset.  ``reset_after_trial`` is
  called only after every post-step evidence object has been converted into a
  row, so a reset state cannot masquerade as terminal survival.
  """

  def __init__(
    self,
    env: DiffusionEnvironment,
    vae: FrozenVaePolicy,
    ppo: FrozenPpoRecoveryPolicy,
    *,
    vae_actions: ActionContract | None = None,
    ppo_actions: ActionContract | None = None,
    qualification: QualificationConfig | None = None,
    store: AppendOnlyShardStore | None = None,
  ) -> None:
    self.env = env
    self.vae = vae
    self.ppo = ppo
    self.vae_actions = vae_actions or vae.action_contract
    self.ppo_actions = ppo_actions or ppo.action_contract
    self.qualification = qualification or QualificationConfig()
    self.store = store
    self._initial_snapshots: dict[tuple[str, str], _InitialSnapshot] = {}
    self._ou_phase_started = False

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
      if not snapshot.qualification.qualified:
        raise CollectorError(
          "OU trial is blocked because its clean partner did not qualify"
        )
    elif spec.phase == "clean":
      if self._ou_phase_started:
        raise CollectorError("clean trial cannot start after the OU phase")
      if spec.pair_id:
        if not spec.initial_state_id:
          raise CollectorError("clean trial lacks an initial-state snapshot identity")
        if snapshot_key in self._initial_snapshots:
          raise CollectorError("clean trial would overwrite an existing pair snapshot")
    if spec.phase == "ou":
      self._ou_phase_started = True
    try:
      self.env.reset(
        seed=spec.seed, motion_id=spec.motion_id, start_frame=spec.start_frame
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
      self.ppo.reset()
      ou = OUProcess(seed=spec.seed) if spec.phase == "ou" else None
      tracker = QualificationTracker(self.qualification)
      rows: list[TrajectoryRow] = []
      controller = "vae"
      previous = np.zeros(31, dtype=np.float64)
      stopped_reason: str | None = None
    except BaseException:
      self.env.reset_after_trial()
      raise
    try:
      for _ in range(self.qualification.control_steps):
        if remaining_transitions is not None and tracker.steps >= remaining_transitions:
          stopped_reason = "control_transition_budget"
          break
        if controller == "vae":
          observation = self.env.observe_vae()
          latent, clean = self.vae.action(observation)
          noise = (
            ou.sample()
            if ou is not None and tracker.vae_steps < self.qualification.retained_steps
            else np.zeros(31, dtype=np.float64)
          )
          action = make_policy_action(clean, noise)
          segment_id = f"{spec.episode_id}:vae"
          state = observation.state
          motion_id = observation.motion_id
          reference_frame = observation.reference_frame
          reference_phase = observation.reference_phase
          row_latent = latent
        else:
          observation = self.env.observe_ppo()
          clean = self.ppo.action(observation)
          action = make_policy_action(clean)
          segment_id = f"{spec.episode_id}:ppo"
          state = observation.state
          motion_id = observation.motion_id
          reference_frame = observation.reference_frame
          reference_phase = observation.reference_phase
          row_latent = None

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
        final_evidence = tracker.observe(controller, evidence, bundle.observation)
        row = TrajectoryRow(
          run_id=spec.run_id,
          env_id=0,
          episode_id=spec.episode_id,
          segment_id=segment_id,
          tick=tracker.steps - 1,
          timestamp=bundle.timestamp,
          motion_id=motion_id,
          reference_frame=reference_frame,
          controller=controller,
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
          reference_boundary=(
            final_evidence.reference_boundary or final_evidence.clip_ended
          ),
          fps=1.0 / self.qualification.period_seconds,
          # Keep the stable row key while marking it as a generated split label.
          group_key=spec.group_key or f"{spec.motion_id}:{spec.start_frame}",
          provenance={
            "phase": spec.phase,
            "reference_phase": f"{reference_phase:.9g}",
            "initial_seed": str(spec.seed),
            "initial_start_frame": str(spec.start_frame),
            "pair_id": spec.pair_id,
            "initial_state_id": spec.initial_state_id,
            "group_key_generated": "true" if not spec.group_key else "false",
            "post_step_finite": str(evidence.finite),
            "post_step_reason": evidence.reason or "",
          },
        )
        rows.append(row)
        previous = action.executed.copy()

        if tracker.failure is not None:
          break
        if controller == "vae" and (evidence.reference_boundary or evidence.clip_ended):
          physical = self.vae_actions.to_physical(action.executed)
          self.ppo.handoff_from_physical(physical)
          self.env.switch_to_ppo()
          controller = "ppo"
        elif evidence.terminated or evidence.truncated:
          break
      validate_monotonic_timestamps(rows)
      qualification = tracker.finish()
      if spec.phase == "clean" and spec.pair_id:
        current = self._initial_snapshots[(spec.pair_id, spec.initial_state_id)]
        self._initial_snapshots[(spec.pair_id, spec.initial_state_id)] = replace(
          current, qualification=qualification
        )
      accepted = (
        tuple(
          replace(row, segment_qualified=True)
          for index, row in enumerate(rows)
          if row.controller == "vae"
          and sum(existing.controller == "vae" for existing in rows[: index + 1])
          <= self.qualification.retained_steps
        )
        if qualification.qualified
        else ()
      )
      accepted_ids = {row.identity for row in accepted}
      verification = tuple(row for row in rows if row.identity not in accepted_ids)
      persisted = tuple(
        next(
          (
            accepted_row
            for accepted_row in accepted
            if accepted_row.identity == row.identity
          ),
          row,
        )
        for row in rows
      )
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


__all__ = [
  "CollectionResult",
  "CollectorError",
  "DiffusionCollector",
  "TrialSpec",
]
