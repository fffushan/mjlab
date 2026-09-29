"""Small CPU fixtures for the D1 offline-core tests."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from mjlab.tasks.tracking.diffusion import TerminalEvidence, TrajectoryRow, WorldState


def make_state(index: int = 0, *, angular_velocity: float | None = None) -> WorldState:
  yaw = 0.17 + 0.03 * index
  return WorldState(
    np.array([1.0 + index * 0.01, -0.2, 0.5]),
    np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]),
    np.array([0.2, -0.1, 0.03]),
    np.full(3, angular_velocity)
    if angular_velocity is not None
    else np.array([0.0, 0.1, -0.2]),
    np.arange(60, dtype=np.float64).reshape(20, 3) * 0.01,
    np.ones((20, 3), dtype=np.float64) * 0.2,
  )


def make_row(
  index: int,
  *,
  group: str = "group",
  episode: str = "episode",
  run: str = "run",
  env: int = 0,
  timestamp: float | None = None,
  provenance: Mapping[str, str] | None = None,
  state: WorldState | None = None,
) -> TrajectoryRow:
  action = np.linspace(-0.2, 0.2, 31) + index * 1e-3
  return TrajectoryRow(
    run_id=run,
    env_id=env,
    episode_id=episode,
    segment_id="segment",
    tick=index,
    timestamp=index * 0.02 if timestamp is None else timestamp,
    motion_id="motion",
    reference_frame=index,
    controller="vae",
    state=make_state(index) if state is None else state,
    previous_executed_action=action,
    clean_action=action + 0.01,
    ou_noise=np.full(31, 0.01),
    executed_action=(action + 0.01) + np.full(31, 0.01),
    latent=np.linspace(-1.0, 1.0, 32) + index * 1e-3,
    terminal_evidence=TerminalEvidence(True, False, False),
    segment_qualified=True,
    group_key=group,
    provenance={} if provenance is None else provenance,
  )
