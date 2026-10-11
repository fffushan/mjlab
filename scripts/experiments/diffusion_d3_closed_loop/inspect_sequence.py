#!/usr/bin/env python3
"""One-shot D3 diffusion future-pose inspection.

This experiment entry point generates exactly one complete future-pose sample
from the frozen D3 diffusion stack and shows the fitted poses in a browser:

1. without ``--execute`` it is a simulator-free preflight of the frozen
   artifacts (the same explicit-path loaders as ``evaluate.py``);
2. with ``--execute`` one environment executes exactly eight real VAE bootstrap
   transitions, then the reference is frozen by ``enter_reference_free``;
3. one complete 41x167 sample is drawn directly from the installed plan and
   sampler.  ``continuation.step`` is deliberately never used: no diffusion
   action is decoded and no physics step is executed after handover;
4. the 32 predicted future world poses are recovered and fitted by one isolated
   CPU inverse-kinematics call;
5. a small NPZ plus the compiled MuJoCo model are saved. The viewer plays the
   fitted poses in a loop, with pause, single-step, reset and all-ghosts controls.

``--no-view`` generates the artifacts headlessly.  ``--replay DIR`` reopens a
saved directory without loading neural artifacts and without stepping physics.

The pose recovery and the IK solver live in the sibling ``sequence_poses.py``
helper, which is imported lazily so preflight and replay do not need it.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from mjlab.tasks.tracking.diffusion.closed_loop import _vae_action

_HERE = Path(__file__).resolve().parent

# The strict D4 task loader and the frozen-artifact preflight are owned by the
# sibling evaluator; it is loaded by absolute path so this script keeps working
# from any working directory without a second, drifting copy of the loaders.
_EVALUATE_NAME = "mjlab_d3_closed_loop_evaluate"
_EVALUATE_PATH = _HERE / "evaluate.py"

# The pose/IK helper is a local experiment module owned by a separate lane.  It
# is loaded lazily so preflight and replay never import it.
_SEQUENCE_POSES_NAME = "mjlab_d3_sequence_poses"
_SEQUENCE_POSES_PATH = _HERE / "sequence_poses.py"

_BOOTSTRAP_TRANSITIONS = 8
_FUTURE_STEPS = 32
_MAX_IK_ITERATIONS = 30
_TIME_TOLERANCE = 1.0e-8

_NPZ_NAME = "future-poses.npz"
_MODEL_NAME = "ghost-model.mjb"
_STATUS_NAME = "inspect-status.json"

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8080

_BODY_POINT_COLOR = (0.90, 0.25, 0.20, 1.0)
_BODY_POINT_RADIUS = 0.012


class InspectSequenceError(ValueError):
  """The one-shot future-pose inspection request or a generation step failed."""


def _load_evaluate() -> Any:
  """Import the sibling D3 evaluator by absolute path exactly once."""
  existing = sys.modules.get(_EVALUATE_NAME)
  if existing is not None:
    return existing
  if not _EVALUATE_PATH.is_file():
    raise InspectSequenceError(f"D3 evaluator is not readable: {_EVALUATE_PATH}")
  spec = importlib.util.spec_from_file_location(_EVALUATE_NAME, _EVALUATE_PATH)
  if spec is None or spec.loader is None:
    raise InspectSequenceError(f"D3 evaluator cannot be imported: {_EVALUATE_PATH}")
  module = importlib.util.module_from_spec(spec)
  # Register before execution so dataclass string-annotation resolution can
  # find the module in ``sys.modules``.
  sys.modules[_EVALUATE_NAME] = module
  spec.loader.exec_module(module)
  return module


evaluate = _load_evaluate()


def _sequence_poses() -> Any:
  """Import the sibling pose/IK helper by absolute path exactly once."""
  existing = sys.modules.get(_SEQUENCE_POSES_NAME)
  if existing is not None:
    return existing
  if not _SEQUENCE_POSES_PATH.is_file():
    raise InspectSequenceError(
      f"sequence pose helper is not readable: {_SEQUENCE_POSES_PATH}"
    )
  spec = importlib.util.spec_from_file_location(
    _SEQUENCE_POSES_NAME, _SEQUENCE_POSES_PATH
  )
  if spec is None or spec.loader is None:
    raise InspectSequenceError(
      f"sequence pose helper cannot be imported: {_SEQUENCE_POSES_PATH}"
    )
  module = importlib.util.module_from_spec(spec)
  sys.modules[_SEQUENCE_POSES_NAME] = module
  spec.loader.exec_module(module)
  return module


# Parsing and preflight.


def build_parser() -> argparse.ArgumentParser:
  """Return the evaluator parser extended with the inspection options."""
  parser = evaluate.build_parser()
  actions = getattr(parser, "_option_string_actions", {})
  checkpoint_action = actions.get("--checkpoint")
  if checkpoint_action is not None:
    # ``--replay`` reopens saved poses and names no frozen artifact, so the
    # checkpoint gate is enforced after parsing instead of by argparse.
    checkpoint_action.required = False
  parser.description = (
    "Generate exactly one D3 diffusion future-pose sample and replay fitted poses. "
    "Without --execute this is a simulator-free preflight; --replay reopens "
    "a previously saved directory with no neural artifact and no physics step."
  )
  parser.epilog = (
    "This tool samples once. Inherited evaluator budget flags do not apply. "
    "The saved-pose viewer stays open until Ctrl+C."
  )
  for option in ("--continuation-steps", "--max-wall-seconds", "--max-output-bytes"):
    actions[option].help = "evaluator-only option; ignored by this one-shot tool"
  parser.add_argument(
    "--replay",
    type=Path,
    default=None,
    help="saved inspection directory to reopen instead of generating a sample",
  )
  parser.add_argument(
    "--no-view",
    action="store_true",
    help="generate or load the poses without opening a Viser server",
  )
  parser.add_argument(
    "--host",
    default=_DEFAULT_HOST,
    help=f"Viser bind host (default: {_DEFAULT_HOST}; localhost only)",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=_DEFAULT_PORT,
    help=f"Viser bind port (default: {_DEFAULT_PORT})",
  )
  return parser


def _path_text(value: Any) -> str | None:
  """Return a resolved absolute path string, or ``None`` for an absent path."""
  return None if value is None else str(Path(value).resolve())


def _preflight_report(prepared: Any) -> dict[str, Any]:
  args = prepared.args
  report: dict[str, Any] = {
    "schema": "mjlab-d3-future-pose-inspection-preflight-v1",
    "ok": True,
    "authority": "one-shot-future-pose-inspection",
    "diagnostic_only": True,
    "execution": "not-run; pass --execute to generate the future-pose sample",
    "member": args.member,
    "seed": args.seed,
    "start_frame": args.start_frame,
    "device": str(prepared.device),
    "artifact_root": str(prepared.repo_root),
    "motion": {"frames": prepared.clip_frames, "fps": prepared.clip_fps},
    "checkpoint": {
      "path": _path_text(args.checkpoint),
      "selected_step": prepared.denoiser.global_step,
    },
    "contract": {
      "path": _path_text(args.contract),
      "window_steps": prepared.contract.window_steps,
      "token_dimension": prepared.contract.token_dimension,
      "body_count": prepared.contract.body_count,
    },
    "budget": {
      "bootstrap_transitions": _BOOTSTRAP_TRANSITIONS,
      "future_steps": _FUTURE_STEPS,
      "diffusion_action_executed": False,
      "physics_after_handover": False,
    },
    "output_dir": str(Path(args.output_dir).resolve()),
    "viewer": {
      "enabled": not args.no_view,
      "backend": "viser",
      "host": args.host,
      "port": args.port,
    },
  }
  if not args.no_view:
    _validate_viewer_options(args)
  task_report = evaluate._task_config_report(getattr(prepared, "task", None))
  if task_report is not None:
    report["task"] = task_report
  return report


def _validate_viewer_options(args: argparse.Namespace) -> None:
  if not isinstance(args.host, str) or not args.host:
    raise InspectSequenceError("host must be a non-empty string")
  if (
    isinstance(args.port, bool)
    or not isinstance(args.port, int)
    or not 1 <= args.port <= 65535
  ):
    raise InspectSequenceError("port must be an integer in [1, 65535]")


# One-step VAE bootstrap.


def _to_numpy(value: Any) -> np.ndarray:
  """Convert a torch tensor, a warp bridge array or a sequence to NumPy."""
  if isinstance(value, Tensor):
    return value.detach().to(device="cpu").numpy()
  tensor = getattr(value, "_tensor", None)
  if isinstance(tensor, Tensor):
    return tensor.detach().to(device="cpu").numpy()
  return np.asarray(value)


def _vae_latent_action(
  model: Any, observation: Any, *, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
  """Reuse the evaluator's deterministic VAE mean/action implementation."""
  return _vae_action(model, observation, {"encoder": 0, "decoder": 0}, device=device)


def _bootstrap_stop_reason(transition: Any, pre: Any, *, period: float) -> str | None:
  """Return why one bootstrap transition is not an admissible measured step.

  Physical termination (including a fall) is delegated to the runtime's own
  termination evidence; this guard only rejects transplanted, truncated or
  discontinuous post-step data that cannot be paired with the executed latent.
  """
  if bool(getattr(transition, "reset", False)):
    return "reset"
  if bool(getattr(transition, "teleport", False)):
    return "teleport"
  if not bool(getattr(transition, "terminal_available", True)):
    return "missing_terminal_evidence"
  post = transition.observation
  if bool(getattr(transition, "terminated", False)):
    return "terminated"
  if bool(getattr(transition, "truncated", False)):
    return "truncated"
  if bool(getattr(transition, "reference_boundary", False)):
    return "reference_boundary"
  if post.tick != pre.tick + 1:
    return "tick_discontinuity"
  if post.segment_id != pre.segment_id:
    return "segment_change"
  if not np.isclose(
    post.sim_time_s - pre.sim_time_s, period, atol=_TIME_TOLERANCE, rtol=0.0
  ):
    return "period_discontinuity"
  return None


def _execute_bootstrap(
  runtime: Any,
  model: Any,
  *,
  seed: int,
  start_frame: int,
  device: torch.device,
) -> tuple[Any, list[tuple[Any, np.ndarray]]]:
  """Execute exactly eight real VAE transitions, then freeze the reference.

  Returns the reference-free handover observation and the eight measured
  ``(state, latent)`` pairs the continuation history must be rebuilt from.
  """
  period = float(runtime.period_seconds)
  observation = runtime.reset(seed=int(seed), start_frame=int(start_frame))
  pairs: list[tuple[Any, np.ndarray]] = []
  for index in range(_BOOTSTRAP_TRANSITIONS):
    latent, action = _vae_latent_action(model, observation, device=device)
    pre_state = copy.deepcopy(observation.state)
    transition = runtime.step(np.array(action, copy=True))
    reason = _bootstrap_stop_reason(transition, observation, period=period)
    if reason is not None:
      raise InspectSequenceError(
        f"VAE bootstrap transition {index + 1}/{_BOOTSTRAP_TRANSITIONS} "
        f"stopped: {reason}"
      )
    pairs.append((pre_state, latent))
    observation = transition.observation
  handed = runtime.enter_reference_free()
  if handed.reference is not None or handed.reference_frame is not None:
    raise InspectSequenceError(
      "reference-free handover still exposes a motion reference"
    )
  return handed, pairs


# One complete sample, pose recovery and one isolated IK call.


def _sampler_config(prepared: Any) -> Any:
  """Return the frozen unguided sampler config of the closed-loop evaluator.

  The 20-update DDIM grid, ``eta=0``, zero base guidance, unit initial noise
  and start level 1000 mirror ``evaluate._continuation_factory``.  Task
  guidance is installed by the task sampler factory, never by this base config.
  """
  from mjlab.tasks.tracking.diffusion.sampler import SamplerConfig

  return SamplerConfig(
    prepared.denoiser.grid,
    eta=0.0,
    guidance_strength=0.0,
    initial_noise_scale=1.0,
    start_level=evaluate._FROZEN_START_LEVEL,
  )


def _sample_complete_sequence(
  *,
  prepared: Any,
  continuation: Any,
  conditioning: Any,
  current_state: Any,
  tick: int,
  generator: torch.Generator,
) -> tuple[np.ndarray, Any]:
  """Draw one complete 41x167 sample from the installed plan and sampler.

  ``continuation.plan`` applies any installed task plan transform (hard
  keyframe inpainting) and the task sampler factory supplies the guided
  sampler.  ``continuation.step`` is never called, so the diffusion action is
  never decoded and no physics step is executed.
  """
  from mjlab.tasks.tracking.diffusion.sampler import (
    SamplerConditions,
    sample_trajectory,
  )

  plan = continuation.plan(current_state, tick=int(tick))
  conditions = SamplerConditions(plan.clean_tokens, plan.clean_mask)
  config = _sampler_config(prepared)
  denoiser = prepared.denoiser.model
  schedule = prepared.denoiser.schedule
  sampler = None
  if conditioning is not None and conditioning.active:
    sampler = conditioning.sampler_factory(plan, current_state)
  if sampler is None:
    sample, diagnostics = sample_trajectory(
      denoiser,
      schedule=schedule,
      conditions=conditions,
      config=config,
      generator=generator,
    )
  else:
    sample, diagnostics = sampler(
      denoiser,
      schedule=schedule,
      conditions=conditions,
      config=config,
      generator=generator,
    )
  if not isinstance(sample, Tensor):
    raise InspectSequenceError("the sampler must return a torch tensor")
  array = _to_numpy(sample).astype(np.float64)
  if array.ndim == 3 and array.shape[0] == 1:
    array = array[0]
  expected = (prepared.contract.window_steps, prepared.contract.token_dimension)
  if array.shape != expected or not np.isfinite(array).all():
    raise InspectSequenceError(
      f"sampled sequence must be finite with shape {expected}, got {array.shape}"
    )
  return np.ascontiguousarray(array), diagnostics


def _future_poses(
  sample: np.ndarray, *, projection: Any, current_state: Any, contract: Any
) -> dict[str, np.ndarray]:
  """Recover the 32 predicted future world poses through the helper API."""
  helper = _sequence_poses()
  result = helper.recover_future_poses(
    sample, projection=projection, current_state=current_state, contract=contract
  )
  if not isinstance(result, Mapping):
    raise InspectSequenceError("recover_future_poses must return a mapping")
  body_count = contract.body_count
  return {
    "root_positions": _checked_array(
      result.get("root_positions"), "root_positions", (_FUTURE_STEPS, 3)
    ),
    "root_rotations": _checked_array(
      result.get("root_rotations"), "root_rotations", (_FUTURE_STEPS, 3, 3)
    ),
    "body_positions": _checked_array(
      result.get("body_positions"), "body_positions", (_FUTURE_STEPS, body_count, 3)
    ),
  }


def _checked_array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
  """Return a finite float64 array of exactly ``shape`` or fail closed."""
  array = np.asarray(value, dtype=np.float64)
  if array.shape != shape or not np.isfinite(array).all():
    raise InspectSequenceError(
      f"{name} must be a finite array of shape {shape}, got {array.shape}"
    )
  return np.ascontiguousarray(array)


def _isolated_model_context(
  runtime: Any,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
  """Copy the runtime MuJoCo model for CPU IK and read the measured center qpos.

  The returned model is an isolated deep copy: holding it cannot mutate the
  live simulator model.  Collision geoms are made fully transparent so the
  ghost renderer shows only visual geometry, exactly like the tracking
  command's ghost model.  The qpos addresses and the returned center qpos are
  expressed in this copied model's layout.
  """
  env = getattr(runtime, "env", None)
  if env is None:
    raise InspectSequenceError("runtime does not expose an environment")
  sim = getattr(env, "sim", None)
  if sim is None:
    raise InspectSequenceError("runtime environment does not expose a simulator")
  source = getattr(sim, "mj_model", None)
  if source is None:
    raise InspectSequenceError("runtime does not expose a compiled MuJoCo model")
  model = copy.deepcopy(source)
  for geom in range(model.ngeom):
    if model.geom_contype[geom] != 0 or model.geom_conaffinity[geom] != 0:
      model.geom_rgba[geom, 3] = 0.0
  robot = env.scene["robot"]
  free_q_adr = _to_numpy(robot.indexing.free_joint_q_adr).reshape(-1).astype(np.int64)
  joint_q_adr = _to_numpy(robot.indexing.joint_q_adr).reshape(-1).astype(np.int64)
  if free_q_adr.shape != (7,):
    raise InspectSequenceError("the free joint must have 7 qpos addresses")
  if joint_q_adr.shape[0] != len(runtime.joint_order):
    raise InspectSequenceError(
      "the robot joint qpos address array must cover every robot joint"
    )
  qpos = _to_numpy(sim.data.qpos)
  if qpos.ndim == 2:
    qpos = qpos[0]
  if qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
    raise InspectSequenceError(
      f"measured center qpos must be finite with shape ({model.nq},), got {qpos.shape}"
    )
  return model, free_q_adr, joint_q_adr, np.array(qpos, dtype=np.float64, copy=True)


def _fit_future_poses(
  *,
  model: Any,
  initial_qpos: np.ndarray,
  free_joint_qpos_adr: np.ndarray,
  joint_qpos_adr: np.ndarray,
  body_names: Sequence[str],
  poses: Mapping[str, np.ndarray],
  max_iterations: int = _MAX_IK_ITERATIONS,
) -> dict[str, np.ndarray]:
  """Fit the whole future sequence in one isolated IK call through the helper."""
  helper = _sequence_poses()
  result = helper.fit_pose_sequence(
    model,
    initial_qpos=initial_qpos,
    free_joint_qpos_adr=free_joint_qpos_adr,
    joint_qpos_adr=joint_qpos_adr,
    body_names=body_names,
    root_positions=poses["root_positions"],
    root_rotations=poses["root_rotations"],
    body_positions=poses["body_positions"],
    max_iterations=int(max_iterations),
  )
  if not isinstance(result, Mapping):
    raise InspectSequenceError("fit_pose_sequence must return a mapping")
  converged = np.asarray(result.get("converged"), dtype=bool).reshape(-1)
  iterations = np.asarray(result.get("iterations")).reshape(-1)
  if converged.shape != (_FUTURE_STEPS,):
    raise InspectSequenceError("fit converged flags must have shape [32]")
  if iterations.shape != (_FUTURE_STEPS,) or not np.issubdtype(
    iterations.dtype, np.integer
  ):
    raise InspectSequenceError("fit iteration counts must be integers of shape [32]")
  return {
    "qpos": _checked_array(result.get("qpos"), "qpos", (_FUTURE_STEPS, model.nq)),
    "body_errors": _checked_array(
      result.get("body_errors"),
      "body_errors",
      (_FUTURE_STEPS, len(body_names)),
    ),
    "converged": converged,
    "iterations": iterations.astype(np.int64),
  }


# Artifacts and replay.


@dataclass(frozen=True, slots=True)
class FuturePoseDisplay:
  """The saved future-pose sequence every viewer and replay needs."""

  qpos: np.ndarray
  root_positions: np.ndarray
  body_positions: np.ndarray
  body_errors: np.ndarray
  converged: np.ndarray
  iterations: np.ndarray
  center_qpos: np.ndarray
  plan_tick: int
  member: str
  seed: int
  start_frame: int

  @property
  def future_ticks(self) -> list[int]:
    """Absolute control ticks of the 32 future steps."""
    return [self.plan_tick + offset for offset in range(1, _FUTURE_STEPS + 1)]

  def summary(self) -> dict[str, Any]:
    """Bounded, serializable view of the saved sequence."""
    errors = np.asarray(self.body_errors, dtype=np.float64)
    return {
      "schema": "mjlab-d3-future-pose-inspection-display-v1",
      "authority": "one-shot-future-pose-inspection",
      "diagnostic_only": True,
      "member": self.member,
      "seed": self.seed,
      "start_frame": self.start_frame,
      "plan_tick": self.plan_tick,
      "future_steps": int(len(self.qpos)),
      "future_ticks": self.future_ticks,
      "center_root_position": self.center_qpos[:3].tolist(),
      "ik": {
        "converged": int(np.count_nonzero(self.converged)),
        "max_body_error_m": float(errors.max()),
        "mean_body_error_m": float(errors.mean()),
      },
    }


def _save_model(model: Any, path: Path) -> None:
  """Write the isolated compiled MuJoCo model for replay."""
  import mujoco

  mujoco.mj_saveModel(model, str(path))


def _save_display(
  display: FuturePoseDisplay, model: Any, output_dir: Path
) -> dict[str, str]:
  """Write the small NPZ pose bundle, the model and the run status record."""
  output_dir.mkdir(parents=True, exist_ok=True)
  npz_path = output_dir / _NPZ_NAME
  np.savez(
    npz_path,
    qpos=display.qpos,
    root_positions=display.root_positions,
    body_positions=display.body_positions,
    body_errors=display.body_errors,
    converged=display.converged,
    iterations=display.iterations,
    center_qpos=display.center_qpos,
    plan_tick=np.int64(display.plan_tick),
    member=np.array(display.member),
    seed=np.int64(display.seed),
    start_frame=np.int64(display.start_frame),
  )
  model_path = output_dir / _MODEL_NAME
  _save_model(model, model_path)
  return {"poses": str(npz_path), "model": str(model_path)}


def _replay_paths(directory: Path) -> dict[str, Path]:
  """Return the required saved files of a replay directory or fail closed."""
  paths = {
    "poses": directory / _NPZ_NAME,
    "model": directory / _MODEL_NAME,
  }
  if not directory.is_dir():
    raise InspectSequenceError(f"replay directory is not a directory: {directory}")
  for role in ("poses", "model"):
    if not paths[role].is_file():
      raise InspectSequenceError(f"replay directory has no {role}: {paths[role]}")
  return paths


def _load_display(paths: Mapping[str, Path]) -> FuturePoseDisplay:
  """Load the saved display arrays; no neural artifact and no physics is used."""
  with np.load(paths["poses"]) as stored:
    missing = [
      name
      for name in (
        "qpos",
        "root_positions",
        "body_positions",
        "body_errors",
        "converged",
        "iterations",
        "center_qpos",
        "plan_tick",
      )
      if name not in stored
    ]
    if missing:
      raise InspectSequenceError(f"saved pose bundle is missing fields {missing}")
    qpos = np.asarray(stored["qpos"], dtype=np.float64)
    root_positions = np.asarray(stored["root_positions"], dtype=np.float64)
    body_positions = np.asarray(stored["body_positions"], dtype=np.float64)
    body_errors = np.asarray(stored["body_errors"], dtype=np.float64)
    converged = np.asarray(stored["converged"], dtype=bool)
    iterations = np.asarray(stored["iterations"], dtype=np.int64)
    center_qpos = np.asarray(stored["center_qpos"], dtype=np.float64)
    plan_tick = int(np.asarray(stored["plan_tick"]).reshape(-1)[0])
    member = (
      str(np.asarray(stored["member"]).reshape(-1)[0])
      if "member" in stored
      else "unknown"
    )
    seed = int(np.asarray(stored["seed"]).reshape(-1)[0]) if "seed" in stored else -1
    start_frame = (
      int(np.asarray(stored["start_frame"]).reshape(-1)[0])
      if "start_frame" in stored
      else -1
    )
  if center_qpos.ndim != 1 or qpos.shape != (_FUTURE_STEPS, center_qpos.size):
    raise InspectSequenceError(
      f"saved qpos must have shape [{_FUTURE_STEPS}, nq], got {qpos.shape}"
    )
  body_count = body_positions.shape[1] if body_positions.ndim == 3 else 0
  if (
    root_positions.shape != (_FUTURE_STEPS, 3)
    or body_count == 0
    or body_errors.shape != (_FUTURE_STEPS, body_count)
    or converged.shape != (_FUTURE_STEPS,)
    or iterations.shape != (_FUTURE_STEPS,)
  ):
    raise InspectSequenceError("saved pose bundle arrays have inconsistent shapes")
  return FuturePoseDisplay(
    qpos=qpos,
    root_positions=root_positions,
    body_positions=body_positions,
    body_errors=body_errors,
    converged=converged,
    iterations=iterations,
    center_qpos=center_qpos,
    plan_tick=plan_tick,
    member=member,
    seed=seed,
    start_frame=start_frame,
  )


def _replay_preflight(paths: Mapping[str, Path]) -> dict[str, Any]:
  return {
    "schema": "mjlab-d3-future-pose-inspection-replay-preflight-v1",
    "ok": True,
    "authority": "one-shot-future-pose-inspection",
    "diagnostic_only": True,
    "execution": "not-run; pass --execute to open the saved-pose viewer",
    "replay": {
      "directory": str(paths["poses"].parent),
      "poses": str(paths["poses"]),
      "model": str(paths["model"]),
      "neural_artifacts_loaded": False,
      "physics_steps": 0,
    },
  }


# Generation.


def _task_conditioning(prepared: Any) -> Any:
  """Build the D4 task conditioning for a prepared task config, or ``None``."""
  task = getattr(prepared, "task", None)
  if task is None:
    return None
  from mjlab.tasks.tracking.diffusion.conditioned_continuation import (
    TaskConditioning,
  )

  return TaskConditioning(
    task, projection=prepared.projection, contract=prepared.contract
  )


def _status_payload(
  prepared: Any,
  display: FuturePoseDisplay,
  *,
  diagnostics: Any,
  artifacts: Mapping[str, str],
  task_active: bool,
) -> dict[str, Any]:
  errors = np.asarray(display.body_errors, dtype=np.float64)
  payload: dict[str, Any] = {
    "schema": "mjlab-d3-future-pose-inspection-status-v1",
    "authority": "one-shot-future-pose-inspection",
    "diagnostic_only": True,
    "paired_acceptance": False,
    "member": display.member,
    "seed": display.seed,
    "start_frame": display.start_frame,
    "device": str(prepared.device),
    "checkpoint": {
      "path": _path_text(prepared.args.checkpoint),
      "selected_step": prepared.denoiser.global_step,
    },
    "plan_tick": display.plan_tick,
    "future_ticks": display.future_ticks,
    "bootstrap_transitions": _BOOTSTRAP_TRANSITIONS,
    "diffusion_action_executed": False,
    "physics_after_handover": False,
    "sampler": {
      "ddim_updates": evaluate._FROZEN_DDIM_STEPS,
      "start_level": evaluate._FROZEN_START_LEVEL,
      "eta": 0.0,
      "guidance_strength": 0.0,
      "task_conditioning_active": bool(task_active),
      "denoiser_calls": int(getattr(diagnostics, "denoiser_calls", 0)),
    },
    "ik": {
      "max_iterations": _MAX_IK_ITERATIONS,
      "solver_calls": 1,
      "converged": int(np.count_nonzero(display.converged)),
      "max_body_error_m": float(errors.max()),
      "mean_body_error_m": float(errors.mean()),
    },
    "artifacts": dict(artifacts),
    "output_dir": str(Path(prepared.args.output_dir).resolve()),
  }
  task_report = evaluate._task_config_report(getattr(prepared, "task", None))
  if task_report is not None:
    payload["task"] = task_report
  return payload


def _write_status(status: Mapping[str, Any], output_dir: Path) -> Path:
  output_dir.mkdir(parents=True, exist_ok=True)
  status_path = output_dir / _STATUS_NAME
  status_path.write_text(
    json.dumps(status, indent=1, sort_keys=True) + "\n", encoding="utf-8"
  )
  return status_path


def generate(prepared: Any) -> dict[str, Any]:
  """Generate, save and (unless ``--no-view``) display one future-pose sample."""
  args = prepared.args
  output_dir = Path(args.output_dir).resolve()
  generator = torch.Generator(device=prepared.device)
  generator.manual_seed(int(args.seed))
  runtime: Any = None
  conditioning: Any = None
  try:
    prepared.denoiser.model.to(device=prepared.device)
    prepared.denoiser.model.eval()
    vae_model = prepared.vae.model
    vae_model.to(device=prepared.device)
    vae_model.eval()
    runtime = evaluate._runtime_factory(prepared)()
    observation, pairs = _execute_bootstrap(
      runtime,
      vae_model,
      seed=args.seed,
      start_frame=args.start_frame,
      device=prepared.device,
    )
    conditioning = _task_conditioning(prepared)
    continuation = evaluate._continuation_factory(prepared, conditioning)()
    for state, latent in pairs:
      continuation.record_executed(state, latent)
    sample, diagnostics = _sample_complete_sequence(
      prepared=prepared,
      continuation=continuation,
      conditioning=conditioning,
      current_state=observation.state,
      tick=observation.tick,
      generator=generator,
    )
    poses = _future_poses(
      sample,
      projection=prepared.projection,
      current_state=observation.state,
      contract=prepared.contract,
    )
    model, free_q_adr, joint_q_adr, center_qpos = _isolated_model_context(runtime)
    fitted = _fit_future_poses(
      model=model,
      initial_qpos=center_qpos,
      free_joint_qpos_adr=free_q_adr,
      joint_qpos_adr=joint_q_adr,
      body_names=prepared.contract.body_names,
      poses=poses,
    )
    display = FuturePoseDisplay(
      qpos=fitted["qpos"],
      root_positions=poses["root_positions"],
      body_positions=poses["body_positions"],
      body_errors=fitted["body_errors"],
      converged=fitted["converged"],
      iterations=fitted["iterations"],
      center_qpos=center_qpos,
      plan_tick=int(observation.tick),
      member=str(args.member),
      seed=int(args.seed),
      start_frame=int(args.start_frame),
    )
    artifacts = _save_display(display, model, output_dir)
  finally:
    if conditioning is not None:
      try:
        conditioning.close(stop_reason=None)
      except Exception as exc:  # noqa: BLE001 - cleanup must not mask the result
        print(f"[WARN] task conditioning close failed: {exc}", file=sys.stderr)
    # The runtime is owned here and closed exactly once, on success or failure.
    if runtime is not None:
      try:
        runtime.close()
      except Exception as exc:  # noqa: BLE001 - cleanup must not mask the result
        print(f"[WARN] runtime close failed: {exc}", file=sys.stderr)
  status = _status_payload(
    prepared,
    display,
    diagnostics=diagnostics,
    artifacts=artifacts,
    task_active=conditioning is not None and conditioning.active,
  )
  _write_status(status, output_dir)
  if not args.no_view:
    _validate_viewer_options(args)
    _open_viewer(
      display, model_path=Path(artifacts["model"]), host=args.host, port=args.port
    )
  return status


# Saved-pose Viser playback.


def _viewer_status_html(
  display: FuturePoseDisplay, index: int, all_ghosts: bool
) -> str:
  errors = np.asarray(display.body_errors[index - 1], dtype=np.float64)
  center = np.asarray(display.center_qpos[:3], dtype=np.float64)
  return f"""
      <div style="font-size:0.85em;line-height:1.3;padding:0 1em 0.5em 1em;">
        <strong>FUTURE-POSE INSPECTION</strong> (single sample, not a matched
        acceptance)<br/>
        <strong>Member:</strong> {display.member} | <strong>seed:</strong>
        {display.seed} | <strong>start frame:</strong> {display.start_frame}<br/>
        <strong>Measured center root:</strong>
        [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}] m<br/>
        <strong>Plan tick:</strong> {display.plan_tick} (absolute) |
        <strong>Future tick:</strong> {display.plan_tick + index} |
        <strong>Index:</strong> {index}/{len(display.qpos)}<br/>
        <strong>IK error:</strong> max {float(errors.max()):.4f} m, mean
        {float(errors.mean()):.4f} m | <strong>converged:</strong>
        {bool(display.converged[index - 1])} | <strong>iterations:</strong>
        {int(display.iterations[index - 1])}<br/>
        <strong>View:</strong> {"all 32 ghosts" if all_ghosts else f"future pose {index}"}
      </div>
    """


def _open_viewer(
  display: FuturePoseDisplay, *, model_path: Path, host: str, port: int
) -> None:
  """Play fitted poses in a browser; no simulation or model inference occurs."""
  import mujoco
  import viser

  model = mujoco.MjModel.from_binary_path(str(model_path))
  if display.center_qpos.shape != (model.nq,) or display.qpos.shape != (
    _FUTURE_STEPS,
    model.nq,
  ):
    raise InspectSequenceError(
      f"saved qpos layout does not match the saved model (nq={model.nq})"
    )
  data = mujoco.MjData(model)
  data.qpos[:] = display.center_qpos
  mujoco.mj_forward(model, data)

  server = viser.ViserServer(host=host, port=port, label="mjlab-d3-future-poses")
  try:
    tick = _build_static_scene(display, model, data, server)
    print(f"[INFO] future-pose playback: http://{host}:{port} (Ctrl+C to close)")
    while True:
      tick(time.perf_counter())
      time.sleep(0.01)
  except KeyboardInterrupt:
    pass
  finally:
    server.stop()


def _camera_frame(display: FuturePoseDisplay, model: Any) -> tuple[np.ndarray, float]:
  """Frame fitted robot bodies, not the floor or raw IK target outliers."""
  import mujoco

  data = mujoco.MjData(model)
  moving = np.flatnonzero(model.body_weldid != 0)
  positions = []
  for qpos in (display.center_qpos, *display.qpos):
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    positions.append(data.xpos[moving].copy())
  points = np.concatenate(positions)
  lower, upper = points.min(axis=0) - 0.2, points.max(axis=0) + 0.2
  return (lower + upper) * 0.5, float(np.linalg.norm(upper - lower) * 0.5)


def _build_static_scene(
  display: FuturePoseDisplay, model: Any, data: Any, server: Any
) -> Callable[[float], None]:
  """Build saved-pose playback; the returned timer never steps physics."""
  import mujoco

  from mjlab.viewer.viser.scene import MjlabViserScene

  scene = MjlabViserScene(server, model, num_envs=1)
  scene.debug_visualization_enabled = True
  scene.camera_tracking_enabled = False
  look_at, radius = _camera_frame(display, model)
  lock = threading.RLock()

  def _center_camera(client: Any) -> None:
    fov = np.deg2rad(45.0)
    direction = np.array([-1.0, -1.0, 0.5])
    direction /= np.linalg.norm(direction)
    distance = max(1.5, 1.15 * radius / np.sin(fov * 0.5))
    client.camera.up_direction = (0.0, 0.0, 1.0)
    client.camera.fov = fov
    client.camera.position = look_at + direction * distance
    client.camera.look_at = look_at

  def _center_clients() -> None:
    for client in server.get_clients().values():
      _center_camera(client)

  state: dict[str, Any] = {
    "index": 1,
    "all_ghosts": False,
    "opacity": 0.35,
    "body_points": False,
    "playing": False,
    "speed": 0.25,
    "next_tick": 0.0,
    "syncing": False,
  }

  def _render() -> None:
    index = int(state["index"])
    scene.clear()
    # Single-pose mode shows a solid fitted robot, not an overlay on the center.
    data.qpos[:] = (
      display.center_qpos if state["all_ghosts"] else display.qpos[index - 1]
    )
    mujoco.mj_kinematics(model, data)
    if state["all_ghosts"]:
      for ghost, qpos in enumerate(display.qpos):
        scene.add_ghost_mesh(
          qpos, model, alpha=float(state["opacity"]), label=f"future_{ghost + 1}"
        )
    if state["body_points"]:
      for point in display.body_positions[index - 1]:
        scene.add_sphere(point, _BODY_POINT_RADIUS, _BODY_POINT_COLOR)
    scene.update_from_mjdata(data)
    for handle in scene._ghost_handles.values():
      handle.opacity = float(state["opacity"])
    status.content = _viewer_status_html(display, index, bool(state["all_ghosts"]))
    status.content += (
      f"<p>{'Playing (loop)' if state['playing'] else 'Paused'} | "
      f"{state['speed']:g}x | future offset +{index * 0.02:.2f} s</p>"
    )

  status = server.gui.add_html("")
  with server.gui.add_folder("Future poses"):
    play = server.gui.add_button("Play")
    step = server.gui.add_button("Step")
    reset = server.gui.add_button("Reset")
    speed = server.gui.add_slider(
      "Playback speed",
      min=0.1,
      max=1.0,
      step=0.05,
      initial_value=0.25,
      hint="1x = 50 poses/s.",
    )
    center_view = server.gui.add_button("Center view")
    all_ghosts = server.gui.add_checkbox("All ghosts", initial_value=False)
    index_slider = server.gui.add_slider(
      "Future index",
      min=1,
      max=len(display.qpos),
      step=1,
      initial_value=1,
      hint="Seek pauses playback. Absolute tick = plan tick + index.",
    )
    opacity_slider = server.gui.add_slider(
      "Ghost opacity", min=0.05, max=1.0, step=0.05, initial_value=0.35
    )
    body_points = server.gui.add_checkbox("Raw body points", initial_value=False)

  def _sync_render() -> None:
    # Viser invokes callbacks synchronously for server-side value assignments.
    state["syncing"] = True
    try:
      index_slider.value = state["index"]
      all_ghosts.value = state["all_ghosts"]
      play.label = "Pause" if state["playing"] else "Play"
      opacity_slider.disabled = not state["all_ghosts"]
    finally:
      state["syncing"] = False
    _render()

  def _pause_at(index: int) -> None:
    state.update(index=index, playing=False, all_ghosts=False)
    _sync_render()

  @play.on_click
  def _(_: Any) -> None:
    with lock:
      state["playing"] = not state["playing"]
      state["all_ghosts"] = False
      state["next_tick"] = time.perf_counter() + 0.02 / state["speed"]
      _sync_render()

  @step.on_click
  def _(_: Any) -> None:
    with lock:
      _pause_at(state["index"] % len(display.qpos) + 1)

  @reset.on_click
  def _(_: Any) -> None:
    with lock:
      _pause_at(1)

  @speed.on_update
  def _(_: Any) -> None:
    with lock:
      state["speed"] = float(speed.value)
      state["next_tick"] = time.perf_counter() + 0.02 / state["speed"]
      _sync_render()

  @center_view.on_click
  def _(_: Any) -> None:
    _center_clients()

  @all_ghosts.on_update
  def _(_: Any) -> None:
    with lock:
      if state["syncing"]:
        return
      state.update(all_ghosts=bool(all_ghosts.value), playing=False)
      _sync_render()

  @index_slider.on_update
  def _(_: Any) -> None:
    with lock:
      if not state["syncing"]:
        _pause_at(int(index_slider.value))

  @opacity_slider.on_update
  def _(_: Any) -> None:
    with lock:
      state["opacity"] = float(opacity_slider.value)
      _render()

  @body_points.on_update
  def _(_: Any) -> None:
    with lock:
      state["body_points"] = bool(body_points.value)
      _render()

  def tick(now: float) -> None:
    with lock:
      if not state["playing"] or now < state["next_tick"]:
        return
      state["index"] = state["index"] % len(display.qpos) + 1
      # Do not skip poses or burst through them after a slow render.
      state["next_tick"] = now + 0.02 / state["speed"]
      _sync_render()

  server.on_client_connect(_center_camera)
  with lock:
    _sync_render()
  _center_clients()
  return tick


# Entry point.


def _replay(args: argparse.Namespace) -> int:
  paths = _replay_paths(Path(args.replay).resolve())
  if not args.execute:
    print(json.dumps(_replay_preflight(paths), indent=1, sort_keys=True))
    return 0
  display = _load_display(paths)
  if args.no_view:
    print(json.dumps(display.summary(), indent=1, sort_keys=True))
    return 0
  _validate_viewer_options(args)
  _open_viewer(display, model_path=paths["model"], host=args.host, port=args.port)
  return 0


def main(argv: Sequence[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  try:
    if args.replay is not None:
      return _replay(args)
    if args.checkpoint is None:
      raise InspectSequenceError(
        "--checkpoint is required unless --replay names a saved directory"
      )
    output_explicit = args.output_dir is not None
    prepared = evaluate._prepare(args)
    if not output_explicit:
      label = (
        f"{prepared.args.member}-start{prepared.args.start_frame}"
        f"-seed{prepared.args.seed}"
      )
      prepared.args.output_dir = (
        prepared.repo_root / "logs/diagnostics/diffusion-d3-future-pose" / label
      )
    if not args.execute:
      print(json.dumps(_preflight_report(prepared), indent=1, sort_keys=True))
      return 0
    status = generate(prepared)
    print(json.dumps(status, indent=1, sort_keys=True))
    return 0
  except (
    InspectSequenceError,
    evaluate.ClosedLoopCliError,
    evaluate.ReplayError,
    OSError,
    ValueError,
  ) as exc:
    print(f"D3 future-pose inspection failed: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
