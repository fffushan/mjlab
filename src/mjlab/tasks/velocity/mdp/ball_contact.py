"""Racket-ball contact derived once, so a metric and a termination agree.

The racket-ball contact is only 9-10 ms long while the control step is 20 ms,
so a single ``found`` sample per step misses about half the contacts. The
sensor therefore keeps one control step of physics substeps in
``force_history``, and every consumer must read contact through these helpers
rather than re-implement "did the racket strike the ball".

Two resolutions are exposed because the consumers differ. ``substep_contact``
keeps the substep flag for metrics that measure flight or contact time in
substeps; ``step_contact`` collapses the same data to one flag per control
step, which is what an impact counter or a termination needs.
"""

from __future__ import annotations

import torch

from mjlab.sensor import ContactData, ContactSensor

__all__ = ["history_length_or_raise", "step_contact", "substep_contact"]


def substep_contact(force_history: torch.Tensor) -> torch.Tensor:
  """Contact flags for every substep in the sensor's force history.

  Args:
    force_history: ``[B, N, H, 3]`` contact forces, newest substep at index 0.

  Returns:
    ``[B, H]`` bool, oldest substep first.
  """
  force_mag = torch.linalg.vector_norm(force_history, dim=-1)  # [B, N, H]
  # A non-zero force in any slot is a contact. A zero-force touch has nothing to
  # see in the force channel and is indistinguishable from no contact here.
  contact = torch.any(force_mag > 0.0, dim=1)  # [B, H], newest first
  return contact.flip(-1)


def step_contact(sensor_data: ContactData) -> torch.Tensor:
  """Contact flag for the control step the sensor data belongs to.

  With ``force_history`` the flag is the substep history of this control step,
  so a contact that starts and ends between two ``found`` samples still counts.
  Without it the flag falls back to the instantaneous ``found`` field, which is
  the pre-history behaviour.

  Args:
    sensor_data: The racket-ball contact sensor's current data.

  Returns:
    ``[B]`` bool, one flag per control step.
  """
  if sensor_data.force_history is not None:
    return substep_contact(sensor_data.force_history).any(dim=-1)
  assert sensor_data.found is not None, (
    "ball_contact_metric needs the sensor's 'force' history or 'found' field"
  )
  return torch.any(sensor_data.found > 0, dim=-1)


def history_length_or_raise(
  sensor: ContactSensor, step_dt: float, substep_dt: float, caller: str
) -> None:
  """Check that the sensor's history is one control step, or absent.

  Args:
    sensor: The racket-ball contact sensor.
    step_dt: Control step, in seconds.
    substep_dt: Physics substep, in seconds.
    caller: Term name used in the error message, so the failure names the term
      that required the history.

  Raises:
    ValueError: If ``history_length`` is neither 0 (fallback) nor the
      decimation.
  """
  decimation = round(step_dt / substep_dt)
  history_length = sensor.cfg.history_length
  if history_length not in (0, decimation):
    msg = (
      f"Sensor '{sensor.cfg.name}' has history_length={history_length}, but "
      f"{caller} needs 0 (fallback) or one control step, {decimation}."
    )
    raise ValueError(msg)
