"""Lightweight interactive playback for a checkpointed distillation student.

The play surface reuses the audited distillation environment and the existing
Viser/native viewers.  It builds the same trusted environment contract as
``distill evaluate`` (the registered non-``play`` task, so corruption, events,
and episode semantics stay those the teacher was trained under), wraps it in a
minimal ``EnvProtocol`` facade so a viewer can step it, and evaluates the saved
student with deterministic mean-latent inference.  No trainer, optimizer,
replay buffer, collector, or PPO runner is constructed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch

from mjlab.tasks.tracking.distillation.adapter import (
  DistillationEnvironmentAdapter,
  DistillationSnapshot,
)
from mjlab.tasks.tracking.distillation.model import ConditionalVAE

_FINAL_CHECKPOINT_NAME = "checkpoint-final.pt"
_ITERATION_CHECKPOINT = re.compile(r"^checkpoint-iter-(\d+)\.pt$")


def discover_distillation_checkpoints(directory: Path | str) -> list[Path]:
  """Return a directory's distillation checkpoints, oldest iteration first.

  ``distill train`` writes ``checkpoint-iter-<lifetime>.pt`` plus the final
  ``checkpoint-final.pt``.  The PPO ``model_<N>.pt`` sort key does not apply
  here, so only this naming scheme is recognized: iterative checkpoints sort by
  their numeric lifetime iteration and the final checkpoint is placed last.
  """
  root = Path(directory)
  if not root.is_dir():
    return []
  keyed: list[tuple[int, int, Path]] = []
  for path in root.iterdir():
    if not path.is_file():
      continue
    key = _checkpoint_sort_key(path.name)
    if key is not None:
      keyed.append((key[0], key[1], path))
  keyed.sort(key=lambda entry: (entry[0], entry[1]))
  return [path for _, _, path in keyed]


def _checkpoint_sort_key(name: str) -> tuple[int, int] | None:
  if name == _FINAL_CHECKPOINT_NAME:
    return (1, 0)
  match = _ITERATION_CHECKPOINT.match(name)
  if match is None:
    return None
  return (0, int(match.group(1)))


class DistillationPlayPolicy:
  """Deterministic mean-latent student policy for interactive playback.

  The callable consumes one :class:`DistillationSnapshot` (the object returned
  by :meth:`DistillationPlayEnvironment.get_observations`) and decodes the
  Gaussian mean latent, matching the canonical evaluation path.  It performs no
  latent sampling, normalizer update, or training step.  The parameter is typed
  ``Any`` because the shared viewer only guarantees that the object it passes
  back is the one ``get_observations`` returned.
  """

  def __init__(self, model: ConditionalVAE) -> None:
    self._model = model

  @property
  def model(self) -> ConditionalVAE:
    return self._model

  def __call__(self, obs: Any) -> torch.Tensor:
    snapshot = obs
    assert isinstance(snapshot, DistillationSnapshot)
    packed = snapshot.packed
    with torch.no_grad():
      action = self._model.mean_inference(packed.reference, packed.conditioning)
    if not torch.isfinite(action).all().item():
      raise ValueError("student playback produced a non-finite action")
    return action.detach().clone()

  def reset(self) -> None:
    """Keep the viewer's reset contract; mean-latent playback has no state."""


class DistillationPlayEnvironment:
  """``EnvProtocol`` facade over the audited distillation adapter.

  The viewer steps ``get_observations``/``step``/``reset`` through the same
  adapter used by collection and evaluation, so reference conditioning, noise,
  delay, history, and boundary-event consumption are unchanged.  ``unwrapped``
  stays the real ``ManagerBasedRlEnv`` so the shared Viser/native overlays and
  the motion scrubber keep operating on the live simulation.
  """

  def __init__(self, adapter: DistillationEnvironmentAdapter) -> None:
    self._adapter = adapter
    # The viewer's EnvProtocol declares ``num_envs`` as a plain attribute.
    self.num_envs = adapter.env.num_envs

  @property
  def adapter(self) -> DistillationEnvironmentAdapter:
    return self._adapter

  @property
  def device(self) -> torch.device | str:
    return self._adapter.env.device

  @property
  def cfg(self) -> Any:
    return self._adapter.env.cfg

  @property
  def unwrapped(self) -> Any:
    return self._adapter.env

  def get_observations(self) -> DistillationSnapshot:
    """Return the aligned snapshot a policy consumes for the next action."""
    return self._adapter.snapshot()

  def step(self, actions: torch.Tensor) -> Any:
    return self._adapter.step(actions)

  def reset(self) -> DistillationSnapshot:
    return self._adapter.reset()

  def close(self) -> None:
    self._adapter.close()


__all__ = [
  "DistillationPlayEnvironment",
  "DistillationPlayPolicy",
  "discover_distillation_checkpoints",
]
