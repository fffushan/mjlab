"""Headless parallel (many-row) D3 diffusion diagnostic.

This is the batched sibling of
``scripts/experiments/diffusion_d3_closed_loop/view.py``.  It reproduces the
viewer's ``--branch diffusion`` arm without a viewer, on many environment rows
at once, and reports how many rows fell.

The arm is the same one the viewer runs:

* every row is reset to the checkpoint sidecar's configured member and start
  frame, so the initial pose is the configured pose of the configured motion;
* the frozen VAE drives exactly ``_BOOTSTRAP_PAIRS`` measured bootstrap ticks,
  and each row records the pair ``(state_t, latent_t)`` of the latent that was
  decoded on ``state_t``;
* handover freezes the reference and disables the reference guards for the whole
  batch, and the reference-free continuation then samples with the frozen
  20-step DDIM grid and decodes the current latent row with that tick's 99-D
  gravity conditioning;
* a row stops on the first ``physical_fall`` (root tilt above the fall
  threshold), reset, teleport, terminal flag, timeout, or continuity break.

Authority: this command is a bounded diagnostic.  It runs one branch, makes no
matched-pair comparison and claims no acceptance.  Every row runs one member
from one start frame, so rows differ only through the environment's own per-row
domain randomization and through their own sampler noise.  They are not
independent initial conditions.

``--parity`` adds the evidence that the batched path is the single-row path:
it runs the real single-row runtime and ``VisualDiagnosticPlayback`` for the
same member, start frame and seed, then compares the applied actions per tick.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate  # noqa: E402  (sibling CLI module, imported the way view.py does)

from mjlab.tasks.tracking.diffusion.parallel_closed_loop_x2 import (  # noqa: E402
  BatchedReferenceFreeContinuation,
  build_parallel_closed_loop_runtime,
  cohort_members,
  tilt_degrees_batch,
)
from mjlab.tasks.tracking.diffusion.sampler import SamplerConfig  # noqa: E402

_BOOTSTRAP_PAIRS = 8
_DEFAULT_ENVS = 256
_DEFAULT_FALL_DEGREES = 70.0
_DEFAULT_PARITY_TICKS = 200
_CONTROL_PERIOD = 0.02
_TIME_TOLERANCE = 1.0e-8
_MAX_STEPS = evaluate._MAX_CONTINUATION_STEPS
_DEFAULT_OUTPUT_ROOT = Path("logs/diagnostics/diffusion-d3-parallel")


class ParallelDiagnosticError(ValueError):
  """A parallel diagnostic request or result is invalid."""


def build_parser() -> argparse.ArgumentParser:
  parser = evaluate.build_parser()
  parser.description = (
    "Run the D3 diffusion arm headless on many environment rows and report per "
    "row falls.  Diagnostic only; no matched acceptance."
  )
  parser.add_argument(
    "--members",
    default="configured",
    help=(
      "comma-separated cohort members, the word 'all' for every cohort member, "
      "or 'configured' for the checkpoint sidecar member (default: configured)"
    ),
  )
  parser.add_argument(
    "--num-envs",
    type=int,
    default=_DEFAULT_ENVS,
    help=f"environment rows per member (default: {_DEFAULT_ENVS})",
  )
  parser.add_argument(
    "--fall-degrees",
    type=float,
    default=_DEFAULT_FALL_DEGREES,
    help=(
      "root tilt above which a row counts as a physical fall "
      f"(default: {_DEFAULT_FALL_DEGREES}; the viewer uses the same value)"
    ),
  )
  parser.add_argument(
    "--steps",
    type=int,
    default=None,
    help=(
      "post-bootstrap continuation ticks per row (default: the remaining clip, "
      "that is clip_frames - 2 - start_frame - 8)"
    ),
  )
  parser.add_argument(
    "--parity",
    action="store_true",
    help="also run the single-row viewer path and compare applied actions",
  )
  parser.add_argument(
    "--parity-rows",
    type=int,
    default=8,
    help=(
      "rows of the parity batch (default: 8, maximum: 12).  The task draws "
      "per-row reset perturbations; above 12 rows its stratified draw changes "
      "row zero's realization, so a one-row comparison stops being a "
      "same-realization comparison"
    ),
  )
  parser.add_argument(
    "--parity-ticks",
    type=int,
    default=_DEFAULT_PARITY_TICKS,
    help="ticks compared in --parity (default: 200)",
  )
  parser.add_argument(
    "--summary-name",
    default="summary.json",
    help="file name of the per-member summary inside --output-dir",
  )
  return parser


def _validate_options(args: Any) -> None:
  if isinstance(args.num_envs, bool) or args.num_envs < 1 or args.num_envs > 4096:
    raise ParallelDiagnosticError("num-envs must be an integer in [1, 4096]")
  if not np.isfinite(float(args.fall_degrees)) or not (
    0.0 < float(args.fall_degrees) < 180.0
  ):
    raise ParallelDiagnosticError("fall-degrees must be finite and in (0, 180)")
  if args.steps is not None and not (
    1 <= args.steps <= _MAX_STEPS
  ):
    raise ParallelDiagnosticError(f"steps must be in [1, {_MAX_STEPS}]")
  if isinstance(args.parity_ticks, bool) or args.parity_ticks < 1:
    raise ParallelDiagnosticError("parity-ticks must be a positive integer")
  if isinstance(args.parity_rows, bool) or not 1 <= args.parity_rows <= 12:
    raise ParallelDiagnosticError(
      "parity-rows must be an integer in [1, 12]; above 12 rows the task's "
      "stratified reset draw changes row zero's realization, so the row-zero "
      "comparison would compare two different realizations"
    )


def _resolve_members(args: Any, prepared: Any) -> tuple[str, ...]:
  configured = str(prepared.args.member)
  if args.members == "configured":
    return (configured,)
  manifest = Path(args.manifest)
  root = Path(prepared.repo_root)
  available = cohort_members(manifest, root)
  if args.members == "all":
    return available
  requested = tuple(
    item.strip() for item in str(args.members).split(",") if item.strip()
  )
  if not requested:
    raise ParallelDiagnosticError("--members selected no member")
  unknown = [item for item in requested if item not in available]
  if unknown:
    raise ParallelDiagnosticError(
      f"members {unknown} are outside the cohort {list(available)}"
    )
  return requested


def _steps_for(args: Any, clip_frames: int, start_frame: int) -> int:
  if args.steps is not None:
    budget = int(args.steps)
  else:
    budget = int(clip_frames) - 2 - int(start_frame) - _BOOTSTRAP_PAIRS
  if budget < 1:
    raise ParallelDiagnosticError(
      f"clip {clip_frames} from frame {start_frame} leaves no continuation tick"
    )
  if budget > _MAX_STEPS:
    budget = _MAX_STEPS
  return budget


def _prepare_models(prepared: Any, device: torch.device) -> None:
  """Place the frozen denoiser and VAE on the run device, in eval mode."""
  prepared.denoiser.model.to(device=device)
  prepared.denoiser.model.eval()
  prepared.vae.model.to(device=device)
  prepared.vae.model.eval()


def _vae_actions(
  model: Any, observation: Any, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
  """Return the frozen VAE action and the decoded latent mean for every row."""
  reference = np.ascontiguousarray(observation.reference, dtype=np.float32)
  conditioning = np.ascontiguousarray(observation.conditioning, dtype=np.float32)
  with torch.no_grad():
    mu, logvar = model.encode(torch.from_numpy(reference).to(device=device))
    action = model.decode(mu, torch.from_numpy(conditioning).to(device=device))
  if not isinstance(mu, torch.Tensor) or not isinstance(action, torch.Tensor):
    raise ParallelDiagnosticError("the frozen VAE must return tensors")
  if not torch.isfinite(mu).all().item() or not torch.isfinite(logvar).all().item():
    raise ParallelDiagnosticError("the frozen VAE encode is not finite")
  actions = action.detach().to(device="cpu", dtype=torch.float32).numpy()
  latents = mu.detach().to(device="cpu", dtype=torch.float32).numpy()
  if actions.shape != (observation.rows, 31):
    raise ParallelDiagnosticError(
      f"VAE actions must have shape [{observation.rows}, 31], got {actions.shape}"
    )
  if latents.shape != (observation.rows, 32):
    raise ParallelDiagnosticError(
      f"VAE latents must have shape [{observation.rows}, 32], got {latents.shape}"
    )
  return (
    np.array(actions, dtype=np.float64, copy=True),
    np.array(latents, dtype=np.float64, copy=True),
  )


def _global_stop_reasons(
  transition: Any, pre: Any, *, reference_free: bool
) -> str | None:
  """Return a batch-wide continuity stop reason, or None."""
  post = transition.observation
  if post.tick != pre.tick + 1:
    return "tick_discontinuity"
  if post.segment_id != pre.segment_id:
    return "segment_change"
  if not np.isclose(
    post.sim_time_s - pre.sim_time_s, _CONTROL_PERIOD, atol=_TIME_TOLERANCE
  ):
    return "period_discontinuity"
  if reference_free and (
    post.reference is not None or post.reference_frame is not None
  ):
    return "reference_leakage"
  return None


def _row_stop_reason(
  transition: Any,
  tilt: np.ndarray,
  row: int,
  fall_degrees: float,
) -> str | None:
  """Return one row's stop reason, in the single-row playback's order."""
  if bool(np.asarray(transition.reset)[row]):
    return "reset"
  if bool(np.asarray(transition.teleport)[row]):
    return "teleport"
  if float(tilt[row]) > float(fall_degrees):
    return "physical_fall"
  if bool(np.asarray(transition.terminated)[row]):
    return "terminated"
  if bool(np.asarray(transition.truncated)[row]):
    return "timeout"
  return None


def run_member(
  prepared: Any,
  *,
  member: str,
  args: Any,
  device: torch.device,
  record_row: int | None = None,
) -> dict[str, Any]:
  """Run one member on ``--num-envs`` rows and return its measurements.

  ``record_row`` also returns the applied actions and the tilt of one row, so
  the parity check compares the production loop instead of a copy of it.
  """
  runtime = build_parallel_closed_loop_runtime(
    manifest_path=args.manifest,
    repo_root=prepared.repo_root,
    member=member,
    device=str(device),
    seed=int(prepared.args.seed),
    num_envs=int(args.num_envs),
  )
  rows = int(runtime.num_envs)
  started = time.perf_counter()
  result: dict[str, Any] = {
    "member": member,
    "rows": rows,
    "device": str(device),
    "seed": int(prepared.args.seed),
    "start_frame": int(prepared.args.start_frame),
    "clip_frames": int(runtime.clip_frames),
    "bootstrap_pairs": _BOOTSTRAP_PAIRS,
    "fall_degrees": float(args.fall_degrees),
    "diagnostic_only": True,
    "matched_acceptance": False,
  }
  try:
    model = prepared.vae.model
    _prepare_models(prepared, device)
    budget = _steps_for(args, runtime.clip_frames, prepared.args.start_frame)
    result["steps_budget"] = budget
    if record_row is not None and not 0 <= int(record_row) < rows:
      raise ParallelDiagnosticError(
        f"record row {record_row} is outside the {rows}-row batch"
      )
    row_actions: list[list[float]] = []
    row_tilts: list[float] = []
    observation = runtime.reset(
      seed=int(prepared.args.seed), start_frame=int(prepared.args.start_frame)
    )
    first_conditioning = (
      None
      if record_row is None
      else np.array(observation.conditioning[int(record_row)], copy=True)
    )
    first_reference = (
      None
      if (record_row is None or observation.reference is None)
      else np.array(observation.reference[int(record_row)], copy=True)
    )
    active = np.arange(rows, dtype=np.int64)
    first_fall_tick = np.full(rows, -1, dtype=np.int64)
    stop_reason: list[str | None] = [None] * rows
    max_tilt = np.zeros(rows, dtype=np.float64)
    bootstrap_states: list[Any] = []
    bootstrap_latents: list[np.ndarray] = []

    # Phase 1: the frozen VAE bootstrap.  Every row advances through the
    # reference clip and records the latent that was decoded on its own state.
    for _ in range(_BOOTSTRAP_PAIRS):
      actions, latents = _vae_actions(model, observation, device)
      pre = observation
      if record_row is not None:
        row_actions.append([float(value) for value in actions[int(record_row)]
                            .reshape(-1)])
      transition = runtime.step(actions)
      tilt = tilt_degrees_batch(transition.observation.state)
      max_tilt = np.maximum(max_tilt, tilt)
      if record_row is not None:
        row_tilts.append(float(tilt[int(record_row)]))
      pre_states = pre.state
      post = transition.observation
      stopped = np.zeros(rows, dtype=bool)
      reason = _global_stop_reasons(transition, pre, reference_free=False)
      for row in range(rows):
        row_reason = _row_stop_reason(transition, tilt, row, args.fall_degrees)
        if row_reason == "physical_fall":
          first_fall_tick[row] = post.tick
        if row_reason is None and reason is not None:
          row_reason = reason
        if row_reason is not None:
          stop_reason[row] = row_reason if row_reason != "physical_fall" else "physical_fall"
          stopped[row] = True
      active = active[~stopped[active]]
      bootstrap_states.append(pre_states)
      bootstrap_latents.append(latents)
      observation = post

    # Handover: freeze the reference for the batch and rebuild each row history.
    continuation = BatchedReferenceFreeContinuation(
      denoiser=prepared.denoiser.model,
      decode=lambda latent, conditioning: prepared.vae.model.decode(
        latent, conditioning
      ),
      schedule=prepared.denoiser.schedule,
      projection=prepared.projection,
      contract=prepared.contract,
      device=device,
      generator=_seeded_generator(device, int(prepared.args.seed)),
      sampler_config=SamplerConfig(
        prepared.denoiser.grid,
        eta=0.0,
        guidance_strength=0.0,
        initial_noise_scale=1.0,
        start_level=evaluate._FROZEN_START_LEVEL,
      ),
      rows=rows,
    )
    for index in range(_BOOTSTRAP_PAIRS):
      continuation.record_executed(
        np.arange(rows, dtype=np.int64),
        bootstrap_states[index],
        bootstrap_latents[index],
      )
    handed = runtime.enter_reference_free()
    if handed.reference is not None or handed.reference_frame is not None:
      raise ParallelDiagnosticError("handover leaked a reference into the batch")
    observation = handed

    # Phase 2: reference-free continuation.
    evaluation_steps = np.zeros(rows, dtype=np.int64)
    ticks_run = 0
    wall_budget = float(prepared.args.max_wall_seconds)
    partial = False
    for _ in range(budget):
      if active.size == 0:
        break
      if time.perf_counter() - started >= wall_budget:
        partial = True
        break
      pre = observation
      step = continuation.step(
        pre.state, pre.conditioning, tick=int(pre.tick), rows=active
      )
      actions = np.zeros((rows, 31), dtype=np.float64)
      actions[step.rows] = step.actions
      if record_row is not None and int(record_row) in set(
        int(row) for row in step.rows
      ):
        position = int(np.flatnonzero(step.rows == int(record_row))[0])
        row_actions.append(
          [float(value) for value in step.actions[position].reshape(-1)]
        )
      transition = runtime.step(actions)
      post = transition.observation
      tilt = tilt_degrees_batch(post.state)
      max_tilt = np.maximum(max_tilt, tilt)
      if record_row is not None:
        row_tilts.append(float(tilt[int(record_row)]))
      ticks_run += 1
      global_reason = _global_stop_reasons(transition, pre, reference_free=True)
      keep: list[int] = []
      for position, row in enumerate(step.rows):
        local = _row_stop_reason(transition, tilt, int(row), args.fall_degrees)
        evaluation_steps[row] += 1
        if local == "physical_fall":
          first_fall_tick[row] = int(post.tick)
        if local is None and global_reason is not None:
          local = global_reason
        if local is None and evaluation_steps[row] >= budget:
          local = "step_budget"
        if local is not None:
          stop_reason[int(row)] = local
          continue
        keep.append(position)
      if keep:
        kept = np.asarray(keep, dtype=np.int64)
        continuation.record_executed(
          step.rows[kept],
          pre.state.select(step.rows[kept]),
          step.executed_latents[kept],
        )
      active = step.rows[np.asarray(keep, dtype=np.int64)] if keep else np.empty(
        0, dtype=np.int64
      )
      observation = post

    if partial:
      # The wall budget ended the sweep; unfinished rows are not measurements.
      for row in range(rows):
        if stop_reason[row] is None:
          stop_reason[row] = "wall_budget"
    else:
      for row in range(rows):
        if stop_reason[row] is not None:
          continue
        if int(evaluation_steps[row]) >= budget:
          stop_reason[row] = "step_budget"
        else:
          # The row was still running when the loop ended, so its outcome is
          # unknown rather than a completed budget.
          stop_reason[row] = "incomplete_run"
          partial = True

    falls = first_fall_tick >= 0
    fall_ticks = first_fall_tick[falls]
    stops: dict[str, int] = {}
    for reason in stop_reason:
      label = reason if reason is not None else "unknown"
      stops[label] = stops.get(label, 0) + 1
    result.update(
      {
        "ticks_run": int(ticks_run),
        "wall_seconds": float(time.perf_counter() - started),
        "partial": bool(partial),
        "fall_count": int(falls.sum()),
        "fall_rate": float(falls.mean()),
        "fall_first_tick": {
          "min": None if fall_ticks.size == 0 else int(fall_ticks.min()),
          "median": None if fall_ticks.size == 0 else int(np.median(fall_ticks)),
          "max": None if fall_ticks.size == 0 else int(fall_ticks.max()),
          "histogram": (
            {}
            if fall_ticks.size == 0
            else {
              str(int(bucket)): int(count)
              for bucket, count in zip(
                *np.unique((fall_ticks // 25) * 25, return_counts=True), strict=True
              )
            }
          ),
        },
        "fall_first_ticks": [int(value) for value in fall_ticks.tolist()],
        "stop_counts": stops,
        "max_tilt_degrees": {
          "max": float(max_tilt.max()),
          "median": float(np.median(max_tilt)),
        },
        "denoiser_calls": int(continuation.denoiser_calls),
        "history_sizes": continuation.history_sizes(),
      }
    )
    result["provenance"] = dict(runtime.provenance)
    result["initial_signature_row0_sha256"] = hashlib.sha256(
      np.ascontiguousarray(runtime.initial_signature()[0], dtype=np.float64).tobytes()
    ).hexdigest()
    if record_row is not None:
      result["row_series"] = {
        "row": int(record_row),
        "actions": row_actions,
        "tilts": row_tilts,
        "stop_reason": stop_reason[int(record_row)],
      }
      result["row0_first_conditioning"] = (
        None if first_conditioning is None else first_conditioning.tolist()
      )
      result["row0_first_reference"] = (
        None if first_reference is None else first_reference.tolist()
      )
    return result
  finally:
    runtime.close()


def _require_plain_contract(prepared: Any) -> None:
  """The parity path mirrors the viewer's plain diffusion branch only."""
  if getattr(prepared, "task", None) is not None:
    raise ParallelDiagnosticError(
      "--parity does not support a task config; run it without --task-config"
    )


def _seeded_generator(device: torch.device, seed: int) -> torch.Generator:
  generator = torch.Generator(device=device)
  generator.manual_seed(int(seed))
  return generator


def run_single_row_reference(
  prepared: Any, *, member: str, args: Any, device: torch.device
) -> dict[str, Any]:
  """Run the real single-row viewer path headless and record its actions."""
  from mjlab.tasks.tracking.diffusion.closed_loop_runtime_x2 import (
    build_closed_loop_runtime,
  )
  from mjlab.tasks.tracking.diffusion.visual_playback import (
    VisualDiagnosticPlayback,
  )

  runtime = build_closed_loop_runtime(
    manifest_path=args.manifest,
    repo_root=prepared.repo_root,
    member=member,
    device=str(device),
    seed=int(prepared.args.seed),
  )
  try:
    # The viewer moves both frozen models onto its device before use; mirror it.
    _prepare_models(prepared, device)
    budget = _steps_for(args, runtime.clip_frames, prepared.args.start_frame)
    playback = VisualDiagnosticPlayback(
      runtime=runtime,
      mode="diffusion",
      model=prepared.vae.model,
      continuation_factory=evaluate._continuation_factory(prepared),
      seed=int(prepared.args.seed),
      start_frame=int(prepared.args.start_frame),
      max_steps=budget,
      fall_degrees=float(args.fall_degrees),
      period_seconds=_CONTROL_PERIOD,
      device=device,
    )
    actions: list[list[float]] = []
    tilts: list[float] = []
    try:
      observation = playback.reset()
      first_conditioning = np.array(observation.conditioning, copy=True)
      first_reference = (
        None
        if observation.reference is None
        else np.array(observation.reference, copy=True)
      )
      while True:
        action = playback.act(observation)
        actions.append(
          [
            float(value)
            for value in np.asarray(
              action.detach().cpu().numpy(), dtype=np.float64
            ).reshape(-1)
          ]
        )
        playback.advance(action)
        tilts.append(
          float(np.asarray(tilt_degrees_batch(playback.observe().state)).reshape(-1)[0])
        )
        if playback.stopped:
          break
        observation = playback.observe()
        if len(actions) > _MAX_STEPS + _BOOTSTRAP_PAIRS + 8:
          break
      status = playback.status()
    finally:
      playback.close()
    return {
      "member": member,
      "rows": 1,
      "start_frame": int(prepared.args.start_frame),
      "clip_frames": int(runtime.clip_frames),
      "steps_budget": budget,
      "actions": actions,
      "tilts": tilts,
      "stop_reason": status.get("stop_reason"),
      "stop_tick": status.get("tick"),
      "phase": status.get("phase"),
      "initial_signature_digest": runtime.provenance.get(
        "initial_state_signature_digest"
      ),
      "first_conditioning": first_conditioning.tolist(),
      "first_reference": (
        None if first_reference is None else first_reference.tolist()
      ),
      "denoiser_calls": status.get("counters", {}).get("denoiser"),
    }
  finally:
    runtime.close()


def _series_difference(left: Any, right: Any) -> float | None:
  """Return the maximum absolute difference of two optional float series."""
  if left is None or right is None:
    return None
  first = np.asarray(left, dtype=np.float64).reshape(-1)
  second = np.asarray(right, dtype=np.float64).reshape(-1)
  if first.shape != second.shape or first.size == 0:
    return None
  return float(np.max(np.abs(first - second)))


def compare_parity(
  prepared: Any, *, member: str, args: Any, device: torch.device
) -> dict[str, Any]:
  """Compare row zero of a small batched batch with the single-row path.

  The parity batch is deliberately small.  The tracking task draws a per-row
  reset perturbation (RSI root pose, velocity, and joint positions) and
  stratifies that draw once the batch is large, so row zero of a wide batch is
  a different realization from a one-row run.  Below the stratification
  boundary the two share one realization, which is what makes a tick-by-tick
  comparison meaningful.
  """
  _require_plain_contract(prepared)
  reference = run_single_row_reference(prepared, member=member, args=args, device=device)
  scoped = argparse.Namespace(**vars(args))
  scoped.num_envs = int(args.parity_rows)
  measured = run_member(
    prepared, member=member, args=scoped, device=device, record_row=0
  )
  series = measured.pop("row_series", {"actions": [], "tilts": [], "stop_reason": None})
  ticks = min(
    len(reference["actions"]), len(series["actions"]), int(args.parity_ticks)
  )
  differences = [
    float(
      np.max(
        np.abs(
          np.asarray(reference["actions"][index], dtype=np.float64)
          - np.asarray(series["actions"][index], dtype=np.float64)
        )
      )
    )
    for index in range(ticks)
  ]
  bootstrap_differences = differences[: min(ticks, _BOOTSTRAP_PAIRS)]
  return {
      "member": member,
      "device": str(device),
      "rows": int(measured["rows"]),
      "parity_rows": int(args.parity_rows),
      "ticks_compared": int(ticks),
      "bootstrap_ticks_compared": int(len(bootstrap_differences)),
      "sign_convention": "applied normalized action, both paths, per tick",
      "max_abs_action_difference": 0.0 if not differences else max(differences),
      "max_abs_action_difference_bootstrap": (
        0.0 if not bootstrap_differences else max(bootstrap_differences)
      ),
      "median_abs_action_difference": (
        0.0 if not differences else float(np.median(differences))
      ),
      "tolerance_used": 1.0e-6,
      "ticks_above_tolerance": int(
        sum(1 for value in differences if value > 1.0e-6)
      ),
      "first_tick_action_difference": 0.0 if not differences else differences[0],
      "second_tick_action_difference": (
        None if len(differences) < 2 else differences[1]
      ),
      "single_row_initial_signature_sha256": reference[
        "initial_signature_digest"
      ],
      "batched_row_zero_initial_signature_sha256": measured.get(
        "initial_signature_row0_sha256"
      ),
      "initial_world_signature_matched": bool(
        reference["initial_signature_digest"] is not None
        and reference["initial_signature_digest"]
        == measured.get("initial_signature_row0_sha256")
      ),
      "first_conditioning_max_abs_difference": _series_difference(
        reference.get("first_conditioning"), measured.get("row0_first_conditioning")
      ),
      "first_reference_max_abs_difference": _series_difference(
        reference.get("first_reference"), measured.get("row0_first_reference")
      ),
      "initial_realization_matched": bool(
        _series_difference(
          reference.get("first_conditioning"), measured.get("row0_first_conditioning")
        )
        == 0.0
        and _series_difference(
          reference.get("first_reference"), measured.get("row0_first_reference")
        )
        == 0.0
      ),
      "note_on_initial_state": (
        "the raw world signature includes each row's environment origin, so it "
        "differs by a constant in a wider batch; the conditioning and reference "
        "vectors are origin independent, so equality at tick zero there is the "
        "test that both paths start from one realization"
      ),
      "first_tick_above_tolerance": next(
        (index for index, value in enumerate(differences) if value > 1.0e-6), None
      ),
      "single_row_ticks_recorded": len(reference["actions"]),
      "batched_rows_recorded": len(series["actions"]),
      "single_row_stop_reason": reference["stop_reason"],
      "single_row_stop_tick": reference["stop_tick"],
      "batched_row_zero_stop_reason": series["stop_reason"],
      "single_row_final_tilt_degrees": (
        reference["tilts"][-1] if reference["tilts"] else None
      ),
      "batched_row_zero_final_tilt_degrees": (
        series["tilts"][-1] if series["tilts"] else None
      ),
      "interpretation": (
        "both paths apply the same arm; a nonzero difference is floating-point "
        "divergence between a batched kernel and a single-row kernel, not a "
        "different arm, and it grows in a contact-rich rollout"
      ),
    }


def _write_json(path: Path, payload: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
  temporary.replace(path)


def _stamp() -> str:
  return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def main(argv: list[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  try:
    _validate_options(args)
    prepared = evaluate._prepare(args)
  except (
    evaluate.ClosedLoopCliError,
    evaluate.ReplayError,
    ParallelDiagnosticError,
    OSError,
    ValueError,
  ) as exc:
    print(f"parallel D3 diagnostic preflight failed: {exc}", file=sys.stderr)
    return 1
  members = _resolve_members(args, prepared)
  if not args.execute:
    print(
      json.dumps(
        {
          "diagnostic_only": True,
          "matched_acceptance": False,
          "authority": "headless-parallel-diagnostic",
          "execute": "not-run; pass --execute to build environments and run",
          "members": list(members),
          "num_envs": int(args.num_envs),
          "configured_member": str(prepared.args.member),
          "configured_start_frame": int(prepared.args.start_frame),
          "fall_degrees": float(args.fall_degrees),
          "parity": bool(args.parity),
          "checkpoint": str(args.checkpoint),
        },
        indent=1,
        sort_keys=True,
      )
    )
    return 0

  device = prepared.device
  results: list[dict[str, Any]] = []
  parity: dict[str, Any] | None = None
  output_root = (
    Path(args.output_dir) if args.output_dir is not None else _DEFAULT_OUTPUT_ROOT
  )
  run_dir = output_root / f"parallel-{_stamp()}"
  exit_code = 0
  for member in members:
    try:
      per_member = (
        prepared
        if member == str(prepared.args.member)
        else _prepared_for_member(args, prepared, member)
      )
      if args.parity and parity is None:
        parity = compare_parity(
          per_member, member=member, args=args, device=device
        )
        _write_json(run_dir / f"parity-{member}.json", parity)
      measured = run_member(per_member, member=member, args=args, device=device)
      results.append(measured)
      _write_json(run_dir / f"member-{member}.json", measured)
      print(
        f"[{member}] rows={measured['rows']} steps={measured['steps_budget']} "
        f"falls={measured['fall_count']}/{measured['rows']} "
        f"({measured['fall_rate']:.3%}) stops={measured['stop_counts']} "
        f"wall={measured['wall_seconds']:.1f}s",
        flush=True,
      )
    except Exception as exc:  # keep the remaining members and report the failure
      exit_code = 1
      import traceback

      failure = {
        "member": member,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc()[-4000:],
      }
      results.append(failure)
      _write_json(run_dir / f"member-{member}.json", failure)
      print(f"[{member}] failed: {failure['error']}", file=sys.stderr, flush=True)

  summary = {
    "diagnostic_only": True,
    "matched_acceptance": False,
    "authority": "headless-parallel-diagnostic",
    "checkpoint": str(args.checkpoint),
    "contract": str(prepared.contract.schema_version),
    "device": str(device),
    "num_envs": int(args.num_envs),
    "fall_degrees": float(args.fall_degrees),
    "configured_member": str(prepared.args.member),
    "configured_start_frame": int(prepared.args.start_frame),
    "members": list(members),
    "results": results,
    "parity": parity,
    "row_semantics": (
      "every row shares one member and one start frame; rows differ only by the "
      "environment's per-row domain randomization and by their own sampler noise"
    ),
  }
  _write_json(run_dir / str(args.summary_name), summary)
  total_rows = sum(int(item.get("rows", 0)) for item in results)
  total_falls = sum(int(item.get("fall_count", 0)) for item in results)
  print(
    f"[summary] rows={total_rows} falls={total_falls} "
    f"({(total_falls / total_rows) if total_rows else 0.0:.3%}) "
    f"evidence={run_dir}",
    flush=True,
  )
  return exit_code


def _prepared_for_member(args: Any, prepared: Any, member: str) -> Any:
  """Re-prepare the same inputs for one other cohort member."""
  scoped = argparse.Namespace(**vars(args))
  scoped.member = member
  return evaluate._prepare(scoped)


if __name__ == "__main__":
  raise SystemExit(main())
