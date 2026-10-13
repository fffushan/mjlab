#!/usr/bin/env python3
"""Opt-in browser Viser diagnostic for one bounded D3 frozen-policy branch.

This entry point sits beside ``evaluate.py`` and reuses its explicit-path
frozen loaders and preflight gates, but it is a *separate* authority: it runs
exactly one operator-selected branch (``baseline`` or ``diffusion``) so the
operator can watch the robot move in the browser.  It never instantiates a
second branch, never performs the matched-pair comparison, and never claims the
1e-3 acceptance gate.

Without ``--execute`` the command loads the frozen contract/checkpoint/
projection/VAE/manifest and prints a JSON preflight report; it constructs no
environment, no viewer and no server.  ``--execute`` constructs one private
runtime and opens a localhost-bound Viser server.

Only Python, torch, the frozen loaders and the existing Viser viewer are used.
No training, deployment, remote, hardware or reference-mutating control is
involved.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
_FALL_DEGREES = 70.0
_FROZEN_CONTROL_PERIOD = 0.02


def _load_evaluate() -> Any:
  """Import the sibling evaluator without depending on the process cwd.

  The evaluator is the reviewed owner of the explicit-path loaders and frozen
  gates; reusing it avoids a second, drifting copy.  It is loaded under a
  private module name so an unrelated ``evaluate`` package cannot shadow it and
  importing it executes no simulator or viewer code.
  """
  spec = importlib.util.spec_from_file_location(
    "mjlab_d3_closed_loop_evaluate", _HERE / "evaluate.py"
  )
  if spec is None or spec.loader is None:
    raise ImportError(f"cannot load D3 evaluator beside view.py: {_HERE}")
  module = importlib.util.module_from_spec(spec)
  # Register before execution so dataclass string-annotation resolution can
  # find the module in ``sys.modules``.
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


evaluate = _load_evaluate()


def build_parser() -> ArgumentParser:
  """Return the evaluator parser extended with explicit viewer options."""
  parser = evaluate.build_parser()
  parser.description = (
    "Open a localhost Viser diagnostic for one bounded D3 frozen-policy branch "
    "(baseline or diffusion).  Without --execute this is a simulator-free "
    "frozen-artifact preflight; with --execute it constructs one environment "
    "and one local viewer server."
  )
  parser.add_argument(
    "--branch",
    choices=["baseline", "diffusion"],
    default="diffusion",
    help="single diagnostic branch to run (default: diffusion)",
  )
  actions = getattr(parser, "_option_string_actions", {})
  execute_action = actions.get("--execute")
  if execute_action is not None:
    execute_action.help = "construct one private environment and open the local viewer"
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Viser bind host (default: 127.0.0.1; localhost only)",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Viser bind port (default: 8080)",
  )
  parser.add_argument(
    "--frame-rate",
    type=float,
    default=60.0,
    help="viewer render frame rate in Hz (default: 60)",
  )
  return parser


def _validate_viewer_options(args: Any) -> None:
  if not isinstance(args.host, str) or not args.host:
    raise evaluate.ClosedLoopCliError("host must be a non-empty string")
  if (
    isinstance(args.port, bool)
    or not isinstance(args.port, int)
    or not (1 <= args.port <= 65535)
  ):
    raise evaluate.ClosedLoopCliError("port must be an integer in [1, 65535]")
  if (
    isinstance(args.frame_rate, bool)
    or not np.isfinite(float(args.frame_rate))
    or float(args.frame_rate) <= 0.0
  ):
    raise evaluate.ClosedLoopCliError("frame-rate must be finite and positive")


def _preflight_report(prepared: Any) -> dict[str, Any]:
  args = prepared.args
  report: dict[str, Any] = {
    "schema": "mjlab-d3-visual-diagnostic-preflight-v1",
    "ok": True,
    "authority": "single-branch-visual-diagnostic",
    "diagnostic_only": True,
    "matched_acceptance": False,
    "execution": (
      "not-run; pass --execute to construct one environment and open the viewer"
    ),
    "branch": args.branch,
    "viewer": {
      "backend": "viser",
      "host": args.host,
      "port": args.port,
      "url": f"http://{_display_host(args.host)}:{args.port}",
      "frame_rate_hz": float(args.frame_rate),
    },
    "budget": {
      "bootstrap_pairs": evaluate._BOOTSTRAP_PAIRS,
      "max_steps": args.continuation_steps,
      "max_wall_seconds": args.max_wall_seconds,
      "fall_degrees": _FALL_DEGREES,
    },
    "sampler": {
      "ddim_updates": evaluate._FROZEN_DDIM_STEPS,
      "start_level": evaluate._FROZEN_START_LEVEL,
      "eta": 0.0,
      "guidance_strength": 0.0,
    },
    "motion": {
      "frames": prepared.clip_frames,
      "fps": prepared.clip_fps,
      "start_frame": args.start_frame,
    },
    "vae": {
      "member": args.member,
    },
    "checkpoint": {
      "selected_step": prepared.denoiser.global_step,
      "ema_selected": prepared.denoiser.ema_selected,
    },
    "device": str(prepared.device),
    "artifact_root": str(prepared.repo_root),
    "artifact_integrity_checks": "disabled",
    "notes": [
      "one branch only; no baseline comparison and no matched acceptance",
      "reference-free continuation follows eight real VAE bootstrap pairs",
      "the viewer starts paused and never mutates the reference or physical state",
    ],
  }
  task_report = evaluate._task_config_report(getattr(prepared, "task", None))
  if task_report is not None:
    report["task"] = task_report
  return report


def _display_host(host: str) -> str:
  return "localhost" if host in {"0.0.0.0", "::", ""} else host


def _write_status(prepared: Any, playback: Any, task_conditioning: Any = None) -> None:
  """Best-effort bounded status record; never masks the run result."""
  args = prepared.args
  output_dir = Path(args.output_dir).resolve()
  try:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
      "schema": "mjlab-d3-visual-diagnostic-status-v1",
      "diagnostic_only": True,
      "matched_acceptance": False,
      "branch": args.branch,
      "member": args.member,
      "seed": args.seed,
      "start_frame": args.start_frame,
      "device": str(prepared.device),
      "limits": {
        "max_steps": args.continuation_steps,
        "max_wall_seconds": args.max_wall_seconds,
      },
      "status": playback.status(),
    }
    if task_conditioning is not None:
      payload["task"] = task_conditioning.provenance()
    (output_dir / "viewer-status.json").write_text(
      json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
  except OSError as exc:
    print(f"[WARN] could not write viewer status: {exc}", file=sys.stderr)


def _run_viewer(prepared: Any) -> int:
  from mjlab.tasks.tracking.diffusion.visual_playback import (
    D3DiagnosticEnv,
    D3DiagnosticPolicy,
    D3DiagnosticViewer,
    VisualDiagnosticPlayback,
  )

  args = prepared.args
  # The session wall budget must cover setup/model/device load, rendering and
  # paused time, inference/stepping and explicit resets.  Capture the deadline
  # origin here, before any model load, and pass it to the playback.
  session_start = time.perf_counter()
  runtime = None
  playback = None
  viewer = None
  server = None
  task_conditioning = None
  task_conditioning_closed = False
  task_conditioning_used = False
  try:
    import viser

    prepared.denoiser.model.to(device=prepared.device)
    prepared.denoiser.model.eval()
    vae_model = prepared.vae.model
    vae_model.to(device=prepared.device)
    vae_model.eval()

    runtime = evaluate._runtime_factory(prepared)()
    continuation_factory = None
    if args.branch == "diffusion":
      task_spec = getattr(prepared, "task", None)
      if task_spec is None:
        continuation_factory = evaluate._continuation_factory(prepared)
      else:
        from mjlab.tasks.tracking.diffusion.conditioned_continuation import (
          TaskConditioning,
        )

        task_conditioning = TaskConditioning(
          task_spec,
          projection=prepared.projection,
          contract=prepared.contract,
        )

        def build_task_continuation() -> Any:
          nonlocal task_conditioning, task_conditioning_used
          if task_conditioning_used:
            task_conditioning.close(stop_reason="viewer_reset")
            task_conditioning = TaskConditioning(
              task_spec,
              projection=prepared.projection,
              contract=prepared.contract,
            )
          task_conditioning_used = True
          factory = evaluate._continuation_factory(prepared, task_conditioning)
          return factory()

        continuation_factory = build_task_continuation
    render_env = getattr(runtime, "env", None)
    playback = VisualDiagnosticPlayback(
      runtime=runtime,
      mode=args.branch,
      model=vae_model,
      continuation_factory=continuation_factory,
      seed=args.seed,
      start_frame=args.start_frame,
      max_steps=args.continuation_steps,
      max_wall_seconds=args.max_wall_seconds,
      session_start=session_start,
      fall_degrees=_FALL_DEGREES,
      period_seconds=_FROZEN_CONTROL_PERIOD,
      device=prepared.device,
    )
    playback.reset()

    env = D3DiagnosticEnv(playback, render_env=render_env)
    policy = D3DiagnosticPolicy(playback)
    server = viser.ViserServer(host=args.host, port=args.port, label="mjlab-d3-visual")
    print(
      f"[INFO] D3 visual diagnostic ({args.branch}) ready at "
      f"http://{_display_host(args.host)}:{args.port} "
      f"(diagnostic-only; not a matched acceptance)",
      file=sys.stderr,
      flush=True,
    )
    viewer = D3DiagnosticViewer(
      env, policy, playback, viser_server=server, frame_rate=args.frame_rate
    )
    viewer.run()
    if task_conditioning is not None:
      task_conditioning.close(stop_reason=playback.stop_reason)
      task_conditioning_closed = True
    _write_status(prepared, playback, task_conditioning)
    return 0
  except Exception as exc:
    print(f"D3 visual diagnostic failed: {exc}", file=sys.stderr)
    return 1
  finally:
    if task_conditioning is not None and not task_conditioning_closed:
      try:
        task_conditioning.close(
          stop_reason=None if playback is None else playback.stop_reason
        )
        task_conditioning_closed = True
      except Exception as exc:  # noqa: BLE001 - cleanup must not mask the run result
        print(f"[WARN] task conditioning close failed: {exc}", file=sys.stderr)
    if viewer is not None:
      try:
        viewer.close()
      except Exception as exc:  # noqa: BLE001 - cleanup must not mask failure
        print(f"[WARN] viewer close failed: {exc}", file=sys.stderr)
    if server is not None:
      try:
        server.stop()
      except Exception as exc:  # noqa: BLE001 - cleanup must not mask failure
        print(f"[WARN] server stop failed: {exc}", file=sys.stderr)
    # The playback owns the runtime and closes it exactly once.  If the
    # playback was never constructed (e.g. its constructor raised), the
    # runtime is still owned by this function and must be closed here.
    if playback is not None:
      try:
        playback.close()
      except Exception as exc:  # noqa: BLE001 - cleanup must not mask failure
        print(f"[WARN] playback close failed: {exc}", file=sys.stderr)
    elif runtime is not None:
      try:
        runtime.close()
      except Exception as exc:  # noqa: BLE001 - cleanup must not mask failure
        print(f"[WARN] runtime close failed: {exc}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  try:
    _validate_viewer_options(args)
    prepared = evaluate._prepare(args)
  except (
    evaluate.ClosedLoopCliError,
    evaluate.ReplayError,
    OSError,
    ValueError,
  ) as exc:
    print(f"D3 visual diagnostic preflight failed: {exc}", file=sys.stderr)
    return 1
  if not args.execute:
    print(json.dumps(_preflight_report(prepared), indent=1, sort_keys=True))
    return 0
  return _run_viewer(prepared)


if __name__ == "__main__":
  raise SystemExit(main())
