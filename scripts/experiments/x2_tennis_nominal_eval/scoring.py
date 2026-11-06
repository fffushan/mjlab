"""Qualified-flight scoring for the X2 nominal juggling E0 evaluation.

The plan (``docs/design/x2_tennis_nominal_curriculum.md``) fixes seven conditions
for a qualified racket-to-racket flight. This module applies exactly those
conditions to one recorded episode trace and summarises an arm. It is pure NumPy,
so the rules can be tested on synthetic traces without a simulator or a GPU.

**Temporal alignment.** The scoring depends on it, so it is stated once here.

* Substep contact flags are 1 ms physics substeps. The window recorded for control
  step ``k`` covers substeps ``k*H .. k*H+H-1``, oldest first, so concatenating the
  windows in step order gives one chronological 1 ms series. A contact that spans a
  control-step boundary is therefore one run and one duration, not two. This is the
  same carry rule the task's ``ball_sustained_contact`` metric uses, and it is why
  contact time is never capped at the 20 ms control step.
* Ball, plate and velocity samples are taken once per control step, after that
  step's last substep. Sample ``k`` is the state at ``(k+1)*step_dt``.
* Therefore the takeoff reference and the flight apex are read off a 20 ms grid.
  A ballistic apex between two samples is under-read by at most
  ``g*dt**2/8 = 0.5 mm``. The takeoff reference is the sample at the end of the
  control step that contained the last contact substep, which is 0-20 ms after
  release; that makes the measured rise a **conservative lower bound** on the true
  rise, because the reference height is already part of the way up.
* The return-plate test uses the ball sample at the end of the control step that
  contains the returning contact's first substep, i.e. 0-20 ms after the strike.
  Only the in-plane position is used, and the in-plane speed there is small, so the
  plate-coordinate result is insensitive to that offset. The vertical position is
  not used by that test.

**No threshold was tuned against a policy result.** :data:`PLAN_THRESHOLDS` copies
the plan's provisional numbers verbatim. They must stay frozen for the whole E0
run: changing them after seeing a success rate would make the run uninterpretable.

**What this module does not do.** It does not judge whether the sim matches the
real robot, and it does not turn 32 reset samples into 32 independent policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

STEP_DT_S = 0.02
"""Control step of the nominal task: 1 ms physics x 20 substeps."""

SUBSTEPS_PER_STEP = 20
SUBSTEP_DT_S = STEP_DT_S / SUBSTEPS_PER_STEP

LOSS_TERM = "ball_dropped"
"""Termination that means the ball was lost (height or its 0.5 s grace)."""

FALL_TERM = "fell_over"
"""Termination that means the robot fell."""

CRITERION_ORDER = (
  "1_missing_next_contact",
  "1_incomplete_return_contact",
  "2_airborne_too_short",
  "2_nonracket_contact_in_flight",
  "3_no_upward_then_downward",
  "4_rise_below_half_target",
  "5_apex_outside_target_band",
  "6_bounding_contact_too_long",
  "7_return_outside_plate",
)
"""The plan's seven conditions, split where one condition has two parts."""


@dataclass(frozen=True)
class PlanThresholds:
  """The plan's provisional qualified-flight thresholds, frozen before E0.

  Attributes:
    min_airborne_s: Minimum airborne interval between two racket contacts.
    max_contact_s: Longest continuous contact either bounding flight may have. A
      contact that lasts longer is a hold indicator.
    apex_target_tolerance_m: Half-width of the accepted world-datum apex band
      around the arm's target height.
    rise_fraction_of_target: Minimum rise above the actual takeoff position, as a
      fraction of the arm's target height.
    min_sequence_length: Consecutive qualified flights an episode needs to count
      towards the gate.
    min_passing_episodes: Episodes of the 32 that must reach the gate.
    episodes_per_arm: Scored episodes per arm. One per environment.
  """

  min_airborne_s: float = 0.15
  max_contact_s: float = 0.05
  apex_target_tolerance_m: float = 0.05
  rise_fraction_of_target: float = 0.5
  min_sequence_length: int = 10
  min_passing_episodes: int = 26
  episodes_per_arm: int = 32


PLAN_THRESHOLDS = PlanThresholds()


@dataclass(frozen=True)
class PlateGeometry:
  """The modeled string bed, in the plate (``face_center`` site) frame.

  The collision bed is a 32-sided ellipse inscribed in these half-axes. The
  inscribed polygon is 0.48 % smaller than the ellipse at its edge midpoints,
  which is far below the 20 ms position resolution, so the exact ellipse is used
  and the difference is recorded here rather than corrected.

  Attributes:
    half_length_m: Bed half-length along the plate x axis (the shaft).
    half_width_m: Bed half-width along the plate y axis.
    ball_radius_m: Ball radius, for the record only: a sphere touching a plane
      puts its centre directly above the contact point, so the in-plane test needs
      no radius allowance.
  """

  half_length_m: float = 0.157
  half_width_m: float = 0.117
  ball_radius_m: float = 0.0327


@dataclass(frozen=True)
class EpisodeTrace:
  """One scored episode, recorded by the trace recorder.

  All arrays are per control step, in world coordinates and SI units, and the
  first axis is the control step. Contact arrays are per physics substep, oldest
  substep first inside a control step.

  Attributes:
    ball_pos_w: ``[K, 3]`` ball centre position, world.
    ball_vel_w: ``[K, 3]`` ball linear velocity, world.
    plate_pos_w: ``[K, 3]`` centre of the plate top surface, world.
    plate_axes_w: ``[K, 3, 3]`` plate frame axes in world, columns x, y, normal.
    racket_substeps: ``[K, H]`` True where the racket touched the ball.
    other_substeps: ``[K, H]`` True where anything touched the ball and the racket
      did not. See the recorder for the resolution limit of that derivation.
    termination: Per-term termination flags for the episode's final step.
    env_origin_w: ``[3]`` environment-origin offset added to every world position.
    step_dt: Control step in seconds.
    substep_dt: Physics substep in seconds.
  """

  ball_pos_w: np.ndarray
  ball_vel_w: np.ndarray
  plate_pos_w: np.ndarray
  plate_axes_w: np.ndarray
  racket_substeps: np.ndarray
  other_substeps: np.ndarray
  termination: dict[str, bool] = field(default_factory=dict)
  env_origin_w: np.ndarray = field(default_factory=lambda: np.zeros(3))
  step_dt: float = STEP_DT_S
  substep_dt: float = SUBSTEP_DT_S

  def __post_init__(self) -> None:
    self.validate()

  def validate(self) -> None:
    steps = int(self.ball_pos_w.shape[0])
    if steps == 0:
      raise ValueError("an episode trace needs at least one control step")
    for name in ("ball_pos_w", "ball_vel_w", "plate_pos_w"):
      array = getattr(self, name)
      if array.shape != (steps, 3):
        raise ValueError(f"{name} must be [{steps}, 3], got {array.shape}")
    if self.plate_axes_w.shape != (steps, 3, 3):
      raise ValueError(
        f"plate_axes_w must be [{steps}, 3, 3], got {self.plate_axes_w.shape}"
      )
    substeps = int(round(self.step_dt / self.substep_dt))
    for name in ("racket_substeps", "other_substeps"):
      array = getattr(self, name)
      if array.shape != (steps, substeps):
        raise ValueError(f"{name} must be [{steps}, {substeps}], got {array.shape}")
      if array.dtype != np.bool_:
        raise ValueError(f"{name} must be boolean, got {array.dtype}")

  @property
  def steps(self) -> int:
    return int(self.ball_pos_w.shape[0])

  @property
  def substeps_per_step(self) -> int:
    return int(round(self.step_dt / self.substep_dt))

  @property
  def duration_s(self) -> float:
    return self.steps * self.step_dt

  def local(self, world: np.ndarray) -> np.ndarray:
    """Remove the environment-origin offset from a world position or velocity."""
    return np.asarray(world, dtype=np.float64) - np.asarray(
      self.env_origin_w, dtype=np.float64
    )


def flatten_substep_flags(window_flags: np.ndarray) -> np.ndarray:
  """Flatten per-control-step substep flags into one chronological series.

  Args:
    window_flags: ``[K, H]`` boolean flags, oldest substep first inside a step.

  Returns:
    ``[K*H]`` boolean series in physics-substep order. A contact that runs from
    the last substeps of one control step into the first substeps of the next is
    continuous here, which is what carries contact duration across steps.
  """
  array = np.asarray(window_flags)
  if array.ndim != 2:
    raise ValueError(f"expected [steps, substeps], got {array.shape}")
  return array.reshape(-1)


def contact_segments(flat_flags: np.ndarray) -> np.ndarray:
  """Runs of True in a chronological boolean series.

  Args:
    flat_flags: ``[M]`` boolean series.

  Returns:
    ``[n, 2]`` integer array of ``(start, end_exclusive)`` substep indices. Empty
    when the series is all False.
  """
  series = np.asarray(flat_flags, dtype=bool)
  if series.size == 0:
    return np.zeros((0, 2), dtype=np.int64)
  padded = np.concatenate(([False], series, [False]))
  edges = np.flatnonzero(padded[1:] != padded[:-1])
  return edges.reshape(-1, 2).astype(np.int64)


def plate_local_positions(trace: EpisodeTrace) -> np.ndarray:
  """Ball positions in the plate frame, ``[K, 3]``.

  ``R`` is the plate-to-world rotation, so ``R.T @ (p_ball - p_plate)`` is the
  ball position in plate coordinates for each step. The actual recorded plate
  orientation is used, not an assumed upright plate.
  """
  offset = trace.ball_pos_w - trace.plate_pos_w
  return np.einsum("kji,kj->ki", trace.plate_axes_w, offset)


def plate_radial_fraction(x: np.ndarray, y: np.ndarray, geometry: PlateGeometry):
  """Radial position inside the modeled bed ellipse; 1.0 is the bed edge."""
  return np.sqrt(
    (np.asarray(x) / geometry.half_length_m) ** 2
    + (np.asarray(y) / geometry.half_width_m) ** 2
  )


def _samples_strictly_inside(
  first_substep: int, last_substep: int, substeps_per_step: int, steps: int
) -> np.ndarray:
  """Control-step sample indices whose sample time lies strictly inside a window.

  Sample ``k`` is the state at ``(k+1) * step_dt``, i.e. at substep
  ``(k+1) * H``. It is inside ``(first, last)`` in substep units when
  ``first < (k+1) * H < last``. Both sides are integers, so the test is exact and
  has no floating-point edge cases.

  Args:
    first_substep: Window start, in substeps, exclusive.
    last_substep: Window end, in substeps, exclusive.
    substeps_per_step: Physics substeps per control step.
    steps: Control steps available in the episode, used to clamp the result.

  Returns:
    Sorted sample indices, possibly empty.
  """
  h = substeps_per_step
  candidates = np.arange(steps, dtype=np.int64)
  times = (candidates + 1) * h
  return candidates[(times > first_substep) & (times < last_substep)]


def _last_sample_at_or_before(substep: int, substeps_per_step: int, steps: int) -> int:
  """Index of the last control-step sample at or before a substep boundary."""
  h = substeps_per_step
  return min(max(substep // h - 1, 0), steps - 1)


def _first_sample_at_or_after(substep: int, substeps_per_step: int, steps: int) -> int:
  """Index of the first control-step sample at or after a substep boundary."""
  h = substeps_per_step
  return min(max((substep + h - 1) // h - 1, 0), steps - 1)


def _has_upward_then_downward(vertical_velocity: np.ndarray) -> bool:
  """True when the series rises above zero and later falls below it."""
  rising = np.flatnonzero(vertical_velocity > 0.0)
  if rising.size == 0:
    return False
  falling = np.flatnonzero(vertical_velocity < 0.0)
  return bool(falling.size and falling[-1] > rising[0])


def score_episode(
  trace: EpisodeTrace,
  *,
  target_m: float,
  rest_z_m: float,
  geometry: PlateGeometry | None = None,
  thresholds: PlanThresholds = PLAN_THRESHOLDS,
) -> dict[str, Any]:
  """Score one episode against the plan's qualified-flight rule.

  Args:
    trace: The recorded episode. It must contain exactly one episode: the recorder
      stops recording an environment after its first reset, so a flight can never
      cross a reset.
    target_m: The arm's target apex height above the rest datum, in metres.
    rest_z_m: World height of the ball's resting centre on the bed, in the same
      environment-local frame as the trace after the origin is removed.
    geometry: The modeled bed. Defaults to the shipped tennis constants.
    thresholds: The frozen plan thresholds.

  Returns:
    A flat record of per-episode facts and distributions. Flights are counted
    from racket contact to racket contact, so the initial release fall - which
    starts with no preceding racket contact - is never a candidate.
  """
  bed = PlateGeometry() if geometry is None else geometry
  h = trace.substeps_per_step
  steps = trace.steps

  racket_flat = flatten_substep_flags(trace.racket_substeps)
  other_flat = flatten_substep_flags(trace.other_substeps)
  racket_segments = contact_segments(racket_flat)

  ball_z_local = trace.local(trace.ball_pos_w)[:, 2]
  ball_vz_local = trace.ball_vel_w[:, 2]
  local_plate = plate_local_positions(trace)
  radial = plate_radial_fraction(local_plate[:, 0], local_plate[:, 1], bed)

  rejected = dict.fromkeys(CRITERION_ORDER, 0)
  candidate_durations: list[float] = []
  candidate_apex: list[float] = []
  qualified_durations: list[float] = []
  qualified_apex: list[float] = []
  qualified_rise: list[float] = []
  qualified_radial: list[float] = []
  qualified_takeoff: list[float] = []

  longest_run = 0
  current_run = 0
  longest_candidate_run = 0
  current_candidate_run = 0

  for index in range(len(racket_segments) - 1):
    start_a, end_a = (int(value) for value in racket_segments[index])
    start_b, end_b = (int(value) for value in racket_segments[index + 1])
    del start_a  # The launching contact's start does not enter the rule.

    airborne_substeps = start_b - end_a
    airborne_s = airborne_substeps * trace.substep_dt
    candidate_durations.append(airborne_s)

    window_ids = _samples_strictly_inside(end_a, start_b, h, steps)
    if window_ids.size:
      candidate_apex.append(float(ball_z_local[window_ids].max() - rest_z_m))

    # Criterion 6 first: a bounding contact that is a hold disqualifies the flight
    # in either direction, and reporting it as a contact problem is clearer than
    # reporting it as a missing next contact.
    contact_a_s = (end_a - int(racket_segments[index][0])) * trace.substep_dt
    contact_b_s = (end_b - start_b) * trace.substep_dt

    failure = ""
    if end_b == len(racket_flat) and racket_flat[-1]:
      failure = "1_incomplete_return_contact"
    elif (
      contact_a_s > thresholds.max_contact_s or contact_b_s > thresholds.max_contact_s
    ):
      failure = "6_bounding_contact_too_long"
    elif airborne_s < thresholds.min_airborne_s:
      failure = "2_airborne_too_short"
    elif bool(other_flat[end_a:start_b].any()):
      failure = "2_nonracket_contact_in_flight"
    elif window_ids.size == 0 or not _has_upward_then_downward(
      ball_vz_local[window_ids]
    ):
      failure = "3_no_upward_then_downward"

    if not failure:
      takeoff_id = _first_sample_at_or_after(end_a, h, steps)
      takeoff_z = float(ball_z_local[takeoff_id])
      apex_z = float(ball_z_local[window_ids].max())
      rise_m = apex_z - takeoff_z
      apex_above_rest = apex_z - rest_z_m
      if rise_m < thresholds.rise_fraction_of_target * target_m:
        failure = "4_rise_below_half_target"
      elif abs(apex_above_rest - target_m) > thresholds.apex_target_tolerance_m:
        failure = "5_apex_outside_target_band"
      else:
        return_id = _first_sample_at_or_after(start_b, h, steps)
        if radial[return_id] > 1.0:
          failure = "7_return_outside_plate"

    if failure:
      rejected[failure] += 1
      longest_run = max(longest_run, current_run)
      current_run = 0
      longest_candidate_run = max(longest_candidate_run, current_candidate_run)
      current_candidate_run = 0
      continue

    takeoff_id = _first_sample_at_or_after(end_a, h, steps)
    return_id = _first_sample_at_or_after(start_b, h, steps)
    apex_z = float(ball_z_local[window_ids].max())
    qualified_durations.append(airborne_s)
    qualified_apex.append(apex_z - rest_z_m)
    qualified_rise.append(apex_z - float(ball_z_local[takeoff_id]))
    qualified_radial.append(float(radial[return_id]))
    qualified_takeoff.append(float(ball_z_local[takeoff_id] - rest_z_m))
    current_run += 1
    current_candidate_run += 1

  longest_run = max(longest_run, current_run)
  longest_candidate_run = max(longest_candidate_run, current_candidate_run)

  racket_contact_lengths = (
    (racket_segments[:, 1] - racket_segments[:, 0]) * trace.substep_dt
    if len(racket_segments)
    else np.zeros(0)
  )
  holds = int((racket_contact_lengths > thresholds.max_contact_s).sum())

  termination = dict(trace.termination)
  physical_terms = [name for name, fired in termination.items() if fired]
  timed_out = bool(termination.get("time_out", False))
  loss = bool(termination.get(LOSS_TERM, False))
  fall = bool(termination.get(FALL_TERM, False))
  terminated = any(name != "time_out" for name in physical_terms)

  reached_run = longest_run >= thresholds.min_sequence_length
  gate_pass = reached_run and holds == 0 and not loss and not fall

  return {
    "steps": trace.steps,
    "duration_s": trace.duration_s,
    "terminated": terminated,
    "timed_out": timed_out,
    "termination_terms": termination,
    "loss": loss,
    "fall": fall,
    "racket_contact_edges": int(len(racket_segments)),
    "racket_contact_time_s": float(racket_flat.sum() * trace.substep_dt),
    "longest_racket_contact_s": float(
      racket_contact_lengths.max() if len(racket_contact_lengths) else 0.0
    ),
    "hold_count": holds,
    "nonracket_contact_substeps": int(other_flat.sum()),
    "flight_candidates": int(len(candidate_durations)),
    "qualified_flights": int(len(qualified_durations)),
    "longest_qualified_sequence": int(longest_run),
    "longest_candidate_sequence": int(longest_candidate_run),
    "rejected_by_criterion": rejected,
    "candidate_flight_durations_s": candidate_durations,
    "candidate_apex_above_rest_m": candidate_apex,
    "qualified_flight_durations_s": qualified_durations,
    "qualified_apex_above_rest_m": qualified_apex,
    "qualified_takeoff_rise_m": qualified_rise,
    "qualified_takeoff_above_rest_m": qualified_takeoff,
    "qualified_return_plate_radial_fraction": qualified_radial,
    "reached_run_of_min_sequence": bool(reached_run),
    "episode_gate_pass": bool(gate_pass),
  }


def _describe(values: list[float]) -> dict[str, float] | None:
  """Min / quartiles / max of a list, or ``None`` when it is empty."""
  if not values:
    return None
  array = np.asarray(values, dtype=np.float64)
  return {
    "count": int(array.size),
    "min": float(array.min()),
    "p25": float(np.percentile(array, 25)),
    "median": float(np.median(array)),
    "p75": float(np.percentile(array, 75)),
    "max": float(array.max()),
    "mean": float(array.mean()),
  }


def summarize_arm(
  episodes: list[dict[str, Any]],
  *,
  target_m: float,
  thresholds: PlanThresholds = PLAN_THRESHOLDS,
) -> dict[str, Any]:
  """Aggregate scored episodes into the arm's report and gate decision.

  The denominator is the recorded episode count, not the number of resets the
  environment performed, so automatic-reset episodes cannot inflate or dilute a
  rate.

  Args:
    episodes: One :func:`score_episode` record per scored episode.
    target_m: The arm's target apex height.
    thresholds: The frozen plan thresholds.

  Returns:
    Counts, distributions and the gate result. Rates use ``len(episodes)`` as the
    denominator and are reported with that denominator alongside them.
  """
  count = len(episodes)
  if count == 0:
    raise ValueError("an arm summary needs at least one scored episode")

  term_names = sorted({name for row in episodes for name in row["termination_terms"]})
  term_counts = {
    name: sum(1 for row in episodes if row["termination_terms"].get(name, False))
    for name in term_names
  }
  rejected: dict[str, int] = dict.fromkeys(CRITERION_ORDER, 0)
  for row in episodes:
    for name, value in row["rejected_by_criterion"].items():
      rejected[name] += int(value)

  durations = [
    value for row in episodes for value in row["qualified_flight_durations_s"]
  ]
  apexes = [value for row in episodes for value in row["qualified_apex_above_rest_m"]]
  rises = [value for row in episodes for value in row["qualified_takeoff_rise_m"]]
  radial = [
    value for row in episodes for value in row["qualified_return_plate_radial_fraction"]
  ]
  candidate_durations = [
    value for row in episodes for value in row["candidate_flight_durations_s"]
  ]

  gate_passes = sum(1 for row in episodes if row["episode_gate_pass"])
  reached = sum(1 for row in episodes if row["reached_run_of_min_sequence"])

  return {
    "target_m": target_m,
    "episodes": count,
    "episode_denominator": count,
    "duration_s": _describe([row["duration_s"] for row in episodes]),
    "timed_out": sum(1 for row in episodes if row["timed_out"]),
    "terminated": sum(1 for row in episodes if row["terminated"]),
    "termination_terms": term_counts,
    "losses": sum(1 for row in episodes if row["loss"]),
    "falls": sum(1 for row in episodes if row["fall"]),
    "episodes_with_qualified_flight": sum(
      1 for row in episodes if row["qualified_flights"] > 0
    ),
    "qualified_flights_total": sum(row["qualified_flights"] for row in episodes),
    "qualified_flights_per_episode": _describe(
      [row["qualified_flights"] for row in episodes]
    ),
    "longest_qualified_sequence": _describe(
      [row["longest_qualified_sequence"] for row in episodes]
    ),
    "longest_candidate_sequence": _describe(
      [row["longest_candidate_sequence"] for row in episodes]
    ),
    "racket_contact_edges": _describe(
      [row["racket_contact_edges"] for row in episodes]
    ),
    "racket_contact_time_s": _describe(
      [row["racket_contact_time_s"] for row in episodes]
    ),
    "longest_racket_contact_s": _describe(
      [row["longest_racket_contact_s"] for row in episodes]
    ),
    "episodes_with_hold": sum(1 for row in episodes if row["hold_count"] > 0),
    "nonracket_contact_substeps": _describe(
      [row["nonracket_contact_substeps"] for row in episodes]
    ),
    "rejected_by_criterion": rejected,
    "candidate_flight_durations_s": _describe(candidate_durations),
    "qualified_flight_durations_s": _describe(durations),
    "qualified_apex_above_rest_m": _describe(apexes),
    "qualified_takeoff_rise_m": _describe(rises),
    "qualified_return_plate_radial_fraction": _describe(radial),
    "episodes_reaching_min_sequence": reached,
    "episodes_passing_gate": gate_passes,
    "gate_threshold": {
      "min_sequence_length": thresholds.min_sequence_length,
      "min_passing_episodes": thresholds.min_passing_episodes,
      "episodes_per_arm": thresholds.episodes_per_arm,
    },
    "gate_episode_count_matches_plan": count == thresholds.episodes_per_arm,
    "gate_passed": (
      count == thresholds.episodes_per_arm
      and gate_passes >= thresholds.min_passing_episodes
    ),
  }


def compare_matched_initial_states(
  arms: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
  """Check that every arm started from the same recorded release states.

  All eight arms use one episode seed and no other reset randomization, so their
  recorded initial ball positions must agree exactly. Any disagreement means the
  matched-reset claim is false and the arm comparison is not like for like.

  Args:
    arms: Arm name to recorded ball, plate-pose and environment-origin arrays.

  Returns:
    The maximum absolute deviation per array across arms, and whether they agree
    to ``1e-6`` in the same recorded world frame (including matched origins).
  """
  names = sorted(arms)
  if not names:
    return {"arms": [], "matched": False, "reason": "no arms"}
  reference = arms[names[0]]
  worst = dict.fromkeys(
    (
      "initial_ball_pos_w",
      "initial_ball_vel_w",
      "initial_plate_pos_w",
      "initial_plate_axes_w",
      "env_origins",
    ),
    0.0,
  )
  for name in names[1:]:
    for key in worst:
      first = np.asarray(reference[key], dtype=np.float64)
      other = np.asarray(arms[name][key], dtype=np.float64)
      if first.shape != other.shape:
        return {
          "arms": names,
          "matched": False,
          "reason": f"{name} {key} shape {other.shape} != {first.shape}",
        }
      if not np.isfinite(first).all() or not np.isfinite(other).all():
        return {"arms": names, "matched": False, "reason": f"non-finite {name} {key}"}
      worst[key] = max(worst[key], float(np.abs(first - other).max()))
  tolerance = 1e-6
  return {
    "arms": names,
    "max_abs_deviation": worst,
    "tolerance": tolerance,
    "matched": all(value <= tolerance for value in worst.values()),
  }


__all__ = [
  "CRITERION_ORDER",
  "FALL_TERM",
  "LOSS_TERM",
  "EpisodeTrace",
  "PLAN_THRESHOLDS",
  "PlanThresholds",
  "PlateGeometry",
  "STEP_DT_S",
  "SUBSTEPS_PER_STEP",
  "SUBSTEP_DT_S",
  "compare_matched_initial_states",
  "contact_segments",
  "flatten_substep_flags",
  "plate_local_positions",
  "plate_radial_fraction",
  "score_episode",
  "summarize_arm",
]
