"""Opt-in single-branch Viser diagnostic for the frozen D3 policy path.

This module is a deliberately *separate* authority from the matched-pair
headless evaluator in :mod:`mjlab.tasks.tracking.diffusion.closed_loop`.  It
drives exactly one operator-selected branch (``baseline`` or ``diffusion``)
through the existing :class:`~.closed_loop_types.ClosedLoopRuntime` boundary and
the existing frozen :class:`~.handover.ReferenceFreeContinuation`, so an
operator can watch the robot move.

What it deliberately does NOT do:

* it never instantiates a second branch or a paired comparison, so it cannot
  and does not claim the 1e-3 matched-acceptance gate;
* it never relaxes, renormalizes or relabels the frozen contract (167 token
  channels, 20 DDIM updates, start level 1000, ``eta=0``, zero guidance, FP32,
  raw mean-latent decode);
* it never reads the motion reference after handover and never resamples the
  command during reference-free continuation.

Every status surface is labelled ``diagnostic_only`` / ``matched_acceptance:
false``.  A single branch has no separate baseline comparison, so diffusion
continuation is exposed without pretending a paired evaluation passed.

The playback core is simulator-free and viewer-free: it accepts a runtime and
a VAE, and computes one control tick per ``act``/``advance`` pair.  The thin
:class:`D3DiagnosticEnv`/:class:`D3DiagnosticPolicy` facades expose it through
the shared ``EnvProtocol``/``PolicyProtocol`` used by
:class:`~mjlab.viewer.viser.viewer.ViserPlayViewer`, and
:class:`D3DiagnosticViewer` reuses that viewer while defaulting to paused and
disabling the stock reference-scrub controls that would mutate command state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor
from typing_extensions import override

from mjlab.viewer.viser.viewer import UpdateReason, ViserPlayViewer

from .closed_loop_types import ClosedLoopRuntime, D3Observation, D3Transition
from .state import WorldState

_BOOTSTRAP_PAIRS = 8
_ACTION_WIDTH = 31
_LATENT_WIDTH = 32
_CONDITIONING_WIDTH = 99
_REFERENCE_WIDTH = 68
_DEFAULT_PERIOD = 0.02
_DEFAULT_FALL_DEGREES = 70.0
_TIME_TOLERANCE = 1.0e-8

_MODE_BASELINE = "baseline"
_MODE_DIFFUSION = "diffusion"
_MODES = (_MODE_BASELINE, _MODE_DIFFUSION)

_PHASE_TRACKING = "tracking"
_PHASE_BOOTSTRAP = "bootstrap"
_PHASE_REFERENCE_FREE = "reference_free"

# Reference-mutating GUI payloads the stock viewer would otherwise apply.  The
# diagnostic rejects them so a scrubber/seek can never reintroduce a reference
# or teleport the physical state during reference-free continuation.
_UNSAFE_PAYLOAD_TYPES = frozenset({"motion_scrub", "gui_reset"})

# Stop reasons whose post-step observation must never replace the last
# trustworthy cached pose: the terminal/reset/teleport evidence is unavailable
# or transplanted, so the measured post pose is not admissible evidence.
_UNTRUSTED_POSE_REASONS = frozenset({"reset", "teleport", "missing_terminal_evidence"})


class VisualPlaybackError(ValueError):
  """A visual diagnostic request or runtime observation is invalid."""


def tilt_degrees(state: WorldState) -> float:
  """Return root-up tilt from the world quaternion, independent of yaw."""
  q = np.asarray(state.root_quaternion_wxyz, dtype=np.float64)
  norm = float(np.linalg.norm(q))
  if not np.isfinite(norm) or norm < 1.0e-12:
    return float("inf")
  w, x, y, z = q / norm
  up_z = 1.0 - 2.0 * (x * x + y * y)
  return float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0))))


@dataclass(frozen=True, slots=True)
class BootstrapPair:
  """One measured pre-action state paired with the VAE latent decoded on it."""

  state: WorldState
  latent: np.ndarray


@dataclass(slots=True)
class _PendingAction:
  """The action computed for the current observation, awaiting the step."""

  kind: str
  pre_state: WorldState
  latent: np.ndarray
  denoiser_calls: int = 0


def _owned_state(state: WorldState) -> WorldState:
  """Snapshot the six physical arrays so later runtime writes cannot alias."""
  return WorldState(
    root_position=np.array(state.root_position, dtype=np.float64, copy=True),
    root_quaternion_wxyz=np.array(
      state.root_quaternion_wxyz, dtype=np.float64, copy=True
    ),
    root_linear_velocity=np.array(
      state.root_linear_velocity, dtype=np.float64, copy=True
    ),
    root_angular_velocity=np.array(
      state.root_angular_velocity, dtype=np.float64, copy=True
    ),
    body_positions=np.array(state.body_positions, dtype=np.float64, copy=True),
    body_linear_velocities=np.array(
      state.body_linear_velocities, dtype=np.float64, copy=True
    ),
  )


class VisualDiagnosticPlayback:
  """One bounded, single-branch D3 diagnostic owned by exactly one writer.

  ``baseline`` runs the frozen VAE reference-tracking path for at most
  ``max_steps`` ticks.  ``diffusion`` runs exactly eight measured VAE
  bootstrap pairs, then hands ownership to the frozen reference-free
  continuation without resetting or advancing physics.  Every accepted
  transition is bounded by ``max_steps`` post-bootstrap ticks and an optional
  wall budget; a boundary, fall, terminal flag or runtime error stops the run
  and leaves the final pose visible until the viewer resets or closes.
  """

  def __init__(
    self,
    *,
    runtime: ClosedLoopRuntime,
    mode: str,
    model: Any,
    seed: int,
    start_frame: int,
    max_steps: int,
    continuation_factory: Any = None,
    max_sim_time_s: float | None = None,
    max_wall_seconds: float | None = None,
    session_start: float | None = None,
    fall_degrees: float = _DEFAULT_FALL_DEGREES,
    period_seconds: float = _DEFAULT_PERIOD,
    device: str | torch.device = "cpu",
  ) -> None:
    if mode not in _MODES:
      raise VisualPlaybackError(f"mode must be one of {_MODES}, got {mode!r}")
    if not callable(getattr(model, "encode", None)) or not callable(
      getattr(model, "decode", None)
    ):
      raise VisualPlaybackError("model must expose encode(reference) and decode()")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
      raise VisualPlaybackError("max_steps must be a nonnegative integer")
    if mode == _MODE_DIFFUSION and not callable(continuation_factory):
      raise VisualPlaybackError("diffusion mode requires a continuation_factory")
    if not np.isfinite(float(period_seconds)) or float(period_seconds) <= 0:
      raise VisualPlaybackError("period_seconds must be finite and positive")
    self._runtime = runtime
    self._mode = mode
    self._model = model
    self._seed = int(seed)
    self._start_frame = int(start_frame)
    self._max_steps = int(max_steps)
    self._continuation_factory: Any = (
      continuation_factory if mode == _MODE_DIFFUSION else None
    )
    self._max_sim_time_s = max_sim_time_s
    self._max_wall_seconds = max_wall_seconds
    # The session wall budget is an absolute monotonic deadline fixed at
    # execute setup, so rendering/paused/idle time, inference, stepping and
    # explicit resets all count against it.  A user reset never extends it.
    self._session_start = (
      time.perf_counter() if session_start is None else float(session_start)
    )
    self._fall_degrees = float(fall_degrees)
    self._period = float(period_seconds)
    self._device = torch.device(device)
    self._member = str(getattr(runtime, "provenance", {}).get("member", "unknown"))

    self._closed = False
    self._observation: D3Observation | None = None
    self._phase = _PHASE_TRACKING if mode == _MODE_BASELINE else _PHASE_BOOTSTRAP
    self._pending: _PendingAction | None = None
    self._pairs: list[BootstrapPair] = []
    self._continuation: Any = None
    self._bootstrap_completed = 0
    self._evaluation_steps = 0
    self._encoder_calls = 0
    self._decoder_calls = 0
    self._denoiser_calls = 0
    self._sim_origin = 0.0
    self._stopped = True
    self._stop_reason: str | None = None

  # Read-only inspection.

  @property
  def mode(self) -> str:
    return self._mode

  @property
  def device(self) -> torch.device:
    return self._device

  @property
  def stopped(self) -> bool:
    return self._stopped

  @property
  def stop_reason(self) -> str | None:
    return self._stop_reason

  @property
  def runtime(self) -> ClosedLoopRuntime:
    return self._runtime

  def observe(self) -> D3Observation:
    """Return the cached owned observation without advancing or recomputing."""
    if self._observation is None:
      raise VisualPlaybackError("observe() requires reset() first")
    return self._observation

  @property
  def session_deadline(self) -> float | None:
    """Absolute monotonic deadline for the session, or None when unbounded."""
    if self._max_wall_seconds is None:
      return None
    return self._session_start + float(self._max_wall_seconds)

  @property
  def session_expired(self) -> bool:
    """Whether the fixed session wall budget has elapsed, paused or not."""
    deadline = self.session_deadline
    if deadline is None:
      return False
    return time.perf_counter() >= deadline

  def enforce_session_budget(self) -> bool:
    """Stop with ``wall_budget`` once the session deadline has passed.

    Returns True when the session is over.  This is checked before any new
    action and from the viewer tick/run lifecycle (including while paused), so
    idle/inference/reset time cannot exceed the fixed session budget.
    """
    if self.session_expired:
      self._stop("wall_budget")
      return True
    return False

  # Lifecycle.

  def reset(self) -> D3Observation:
    """Reset the runtime and clear all owned history and generation state.

    A diffusion reset performs a brand-new legitimate eight-step bootstrap; a
    previous continuation is dropped so generations are never mixed.
    """
    self._pending = None
    self._pairs = []
    self._continuation = None
    self._bootstrap_completed = 0
    self._evaluation_steps = 0
    self._encoder_calls = 0
    self._decoder_calls = 0
    self._denoiser_calls = 0
    self._stopped = False
    self._stop_reason = None
    self._phase = _PHASE_TRACKING if self._mode == _MODE_BASELINE else _PHASE_BOOTSTRAP
    try:
      observation = self._runtime.reset(seed=self._seed, start_frame=self._start_frame)
    except Exception as exc:
      self._stopped = True
      self._stop_reason = f"reset_error: {exc}"
      raise
    self._observation = observation
    self._sim_origin = float(observation.sim_time_s)
    return observation

  def close(self) -> None:
    """Release owned history and close the runtime exactly once."""
    self._pending = None
    self._pairs = []
    self._continuation = None
    if not self._closed:
      self._closed = True
      self._runtime.close()

  # Stepping.

  def act(self, observation: D3Observation) -> Tensor:
    """Compute the action for the current observation; never advances physics."""
    if self._stopped:
      return self._zero_action()
    if self.enforce_session_budget():
      # No new action may be computed once the session deadline has passed.
      return self._zero_action()
    if observation is not self._observation:
      raise VisualPlaybackError("act() requires the current owned observation")
    if self._phase in (_PHASE_TRACKING, _PHASE_BOOTSTRAP):
      return self._act_vae(observation)
    return self._act_reference_free(observation)

  def advance(self, action: Any) -> None:
    """Apply the pending action, update history, and evaluate stop conditions."""
    if self._stopped:
      return
    if self.enforce_session_budget():
      # Inference may have crossed the deadline after act() began. Do not
      # apply its result or append history once the session has expired.
      self._pending = None
      return
    pending = self._pending
    self._pending = None
    if pending is None:
      self._stop("missing_action")
      return
    values = np.asarray(action, dtype=np.float64).reshape(-1)
    if values.shape != (_ACTION_WIDTH,) or not np.isfinite(values).all():
      self._stop("invalid_action")
      return
    pre_observation = self._observation
    if pre_observation is None:
      self._stop("missing_observation")
      return
    try:
      transition = self._runtime.step(np.array(values, copy=True))
    except Exception as exc:
      self._stop(f"runtime_error: {exc}")
      return
    reason = self._transition_stop_reason(transition, pre_observation)
    if reason is not None:
      if reason not in _UNTRUSTED_POSE_REASONS:
        # Keep the final trustworthy pose of a fall/boundary/terminal tick.  A
        # reset/teleport/unavailable-terminal pose is not admissible evidence.
        self._observation = transition.observation
      self._stop(reason)
      return
    self._observation = transition.observation
    self._accept(pending, transition)

  # Internal stepping helpers.

  def _act_vae(self, observation: D3Observation) -> Tensor:
    if observation.reference is None:
      self._stop("missing_reference")
      return self._zero_action()
    try:
      latent, action = self._vae_action(observation)
    except Exception as exc:
      self._stop(f"inference_error: {exc}")
      return self._zero_action()
    self._pending = _PendingAction(
      kind="vae", pre_state=observation.state, latent=latent
    )
    return torch.as_tensor(action, dtype=torch.float32)

  def _act_reference_free(self, observation: D3Observation) -> Tensor:
    if self._continuation is None:
      self._stop("handover_error: continuation is not established")
      return self._zero_action()
    try:
      step = self._continuation.step(
        observation.state, observation.conditioning, tick=observation.tick
      )
      action = np.asarray(step.action, dtype=np.float64).reshape(-1)
      latent = np.asarray(step.executed_latent, dtype=np.float64).reshape(-1)
      calls = int(step.denoiser_calls)
      if action.shape != (_ACTION_WIDTH,) or not np.isfinite(action).all():
        raise VisualPlaybackError("continuation action must be finite shape [31]")
      if latent.shape != (_LATENT_WIDTH,) or not np.isfinite(latent).all():
        raise VisualPlaybackError("continuation latent must be finite shape [32]")
      if calls < 0:
        raise VisualPlaybackError("continuation denoiser_calls must be nonnegative")
    except Exception as exc:
      self._stop(f"continuation_error: {exc}")
      return self._zero_action()
    self._pending = _PendingAction(
      kind="continuation",
      pre_state=observation.state,
      latent=latent,
      denoiser_calls=calls,
    )
    return torch.as_tensor(action, dtype=torch.float32)

  def _vae_action(self, observation: D3Observation) -> tuple[np.ndarray, np.ndarray]:
    """Decode the deterministic VAE mean, mirroring the frozen evaluator path."""
    reference = np.ascontiguousarray(observation.reference, dtype=np.float32)
    conditioning = np.ascontiguousarray(observation.conditioning, dtype=np.float32)
    if reference.shape != (_REFERENCE_WIDTH,):
      raise VisualPlaybackError("observation reference must have shape [68]")
    if conditioning.shape != (_CONDITIONING_WIDTH,):
      raise VisualPlaybackError("observation conditioning must have shape [99]")
    reference_tensor = torch.from_numpy(reference).to(device=self._device)[None]
    conditioning_tensor = torch.from_numpy(conditioning).to(device=self._device)[None]
    self._encoder_calls += 1
    with torch.no_grad():
      encoded = self._model.encode(reference_tensor)
    if not isinstance(encoded, tuple) or len(encoded) != 2:
      raise VisualPlaybackError("VAE encode must return (mu, logvar)")
    mu, logvar = encoded
    if not isinstance(mu, Tensor) or not isinstance(logvar, Tensor):
      raise VisualPlaybackError("VAE encode outputs must be tensors")
    if mu.shape != (1, _LATENT_WIDTH) or logvar.shape != (1, _LATENT_WIDTH):
      raise VisualPlaybackError("VAE encode outputs must have shape [1, 32]")
    if not torch.isfinite(mu).all().item() or not torch.isfinite(logvar).all().item():
      raise VisualPlaybackError("VAE encode outputs contain non-finite values")
    self._decoder_calls += 1
    with torch.no_grad():
      action = self._model.decode(mu.detach().clone(), conditioning_tensor)
    if not isinstance(action, Tensor):
      raise VisualPlaybackError("VAE decode must return a torch tensor")
    action_np = (
      action.detach().to(device="cpu", dtype=torch.float32).numpy().reshape(-1)
    )
    if action_np.shape != (_ACTION_WIDTH,) or not np.isfinite(action_np).all():
      raise VisualPlaybackError("decoded action must be finite shape [31]")
    mu_np = (
      mu.detach().to(device="cpu", dtype=torch.float32).numpy()[0].astype(np.float64)
    )
    return mu_np, action_np.astype(np.float64)

  def _transition_stop_reason(
    self, transition: D3Transition, pre: D3Observation
  ) -> str | None:
    reset = bool(getattr(transition, "reset", False))
    teleport = bool(getattr(transition, "teleport", False))
    terminal_available = bool(getattr(transition, "terminal_available", True))
    if reset or teleport:
      if not terminal_available:
        return "missing_terminal_evidence"
      return "reset" if reset else "teleport"
    if not terminal_available:
      # Untrusted post-step data cannot establish a fall or an executed pair.
      # The cached trusted pose is preserved by advance().
      return "missing_terminal_evidence"
    post = transition.observation
    if tilt_degrees(post.state) > self._fall_degrees:
      return "physical_fall"
    if getattr(transition, "terminated", False):
      return "terminated"
    if getattr(transition, "truncated", False):
      return "timeout"
    if post.tick != pre.tick + 1:
      return "tick_discontinuity"
    if post.segment_id != pre.segment_id:
      return "segment_change"
    if not np.isclose(
      post.sim_time_s - pre.sim_time_s,
      self._period,
      atol=_TIME_TOLERANCE,
    ):
      return "period_discontinuity"
    if self._phase == _PHASE_REFERENCE_FREE:
      if post.reference is not None or post.reference_frame is not None:
        return "reference_leakage"
    elif getattr(transition, "reference_boundary", False):
      return "reference_boundary"
    return None

  def _accept(self, pending: _PendingAction, transition: D3Transition) -> None:
    if pending.kind == "vae":
      if self._mode == _MODE_BASELINE:
        self._evaluation_steps += 1
      else:
        self._pairs.append(BootstrapPair(pending.pre_state, pending.latent))
        self._bootstrap_completed += 1
        if self._bootstrap_completed == _BOOTSTRAP_PAIRS:
          self._handover()
    else:
      try:
        self._continuation.record_executed(pending.pre_state, pending.latent)
      except Exception as exc:
        self._stop(f"history_record_error: {exc}")
        return
      self._denoiser_calls += pending.denoiser_calls
      self._decoder_calls += 1
      self._evaluation_steps += 1
    self._check_budgets()

  def _handover(self) -> None:
    """Freeze the reference and rebuild history for reference-free continuation."""
    if self._continuation_factory is None:
      self._stop("handover_error: no continuation factory")
      return
    try:
      continuation = self._continuation_factory()
      for pair in self._pairs:
        continuation.record_executed(pair.state, pair.latent)
      handed = self._runtime.enter_reference_free()
    except Exception as exc:
      self._stop(f"handover_error: {exc}")
      return
    if handed.reference is not None or handed.reference_frame is not None:
      self._stop("reference_leakage")
      return
    self._continuation = continuation
    self._phase = _PHASE_REFERENCE_FREE
    self._observation = handed

  def _check_budgets(self) -> None:
    if self._evaluation_steps >= self._max_steps:
      self._stop("step_budget")
      return
    if self._max_sim_time_s is not None:
      observation = self._observation
      if observation is not None:
        elapsed = observation.sim_time_s - self._sim_origin
        if elapsed >= float(self._max_sim_time_s) - _TIME_TOLERANCE:
          self._stop("time_budget")

  def _stop(self, reason: str) -> None:
    if not self._stopped:
      self._stopped = True
      self._stop_reason = reason

  def _zero_action(self) -> Tensor:
    return torch.zeros(_ACTION_WIDTH, dtype=torch.float32)

  def status(self) -> dict[str, Any]:
    """Return a serializable, explicitly diagnostic-only status snapshot."""
    observation = self._observation
    counters = {
      "encoder": self._encoder_calls,
      "decoder": self._decoder_calls,
      "denoiser": self._denoiser_calls,
    }
    return {
      "diagnostic_only": True,
      "matched_acceptance": False,
      "authority": "single-branch-visual-diagnostic",
      "branch": self._mode,
      "member": self._member,
      "phase": self._phase,
      "stopped": self._stopped,
      "stop_reason": self._stop_reason,
      "tick": None if observation is None else int(observation.tick),
      "sim_time_s": None if observation is None else float(observation.sim_time_s),
      "tilt_degrees": (
        None if observation is None else tilt_degrees(observation.state)
      ),
      "pelvis_height_m": (
        None if observation is None else float(observation.state.root_position[2])
      ),
      "bootstrap_completed": self._bootstrap_completed,
      "bootstrap_required": (_BOOTSTRAP_PAIRS if self._mode == _MODE_DIFFUSION else 0),
      "evaluation_steps": self._evaluation_steps,
      "max_steps": self._max_steps,
      "counters": counters,
    }


class D3DiagnosticEnv:
  """``EnvProtocol`` facade so the shared viewer can drive the playback.

  Physics stepping is always delegated to the playback; ``unwrapped`` returns
  the real mjlab environment so the shared scene/overlays keep rendering live
  kinematics.  ``get_observations`` returns the cached owned observation and
  never recomputes an encoder input or samples new noise.
  """

  def __init__(self, playback: VisualDiagnosticPlayback, *, render_env: Any = None):
    self._playback = playback
    self._render_env = render_env
    self.num_envs = 1

  @property
  def playback(self) -> VisualDiagnosticPlayback:
    return self._playback

  @property
  def device(self) -> torch.device:
    return self._playback.device

  @property
  def cfg(self) -> Any:
    if self._render_env is None:
      raise VisualPlaybackError("no render environment is attached")
    return self._render_env.cfg

  @property
  def unwrapped(self) -> Any:
    if self._render_env is None:
      raise VisualPlaybackError("no render environment is attached")
    return self._render_env

  def get_observations(self) -> D3Observation:
    return self._playback.observe()

  def step(self, actions: Any) -> tuple[Any, ...]:
    self._playback.advance(actions)
    return ()

  def reset(self) -> D3Observation:
    return self._playback.reset()

  def close(self) -> None:
    self._playback.close()


class D3DiagnosticPolicy:
  """``PolicyProtocol`` facade delegating one action to the playback."""

  def __init__(self, playback: VisualDiagnosticPlayback):
    self._playback = playback

  def __call__(self, obs: Any) -> Tensor:
    # ``obs`` is the owned D3Observation returned by the env facade's
    # ``get_observations``; the annotation stays ``Any`` to satisfy the shared
    # ``PolicyProtocol`` without leaking the diagnostic type into it.
    return self._playback.act(obs)

  def reset(self) -> None:
    """No-op: the env facade's reset already rebuilt playback state."""


class D3DiagnosticViewer(ViserPlayViewer):
  """Shared Viser viewer with diagnostic status, default pause and safe controls.

  Reuses the stock scene, camera and playback controls.  It (1) starts paused,
  (2) renders an explicit diagnostic-only status header, and (3) locally
  disables the stock motion scrubbing/seek/reference-reset controls, which
  would otherwise mutate physical command state or reintroduce a reference
  after reference-free handover.  No global viewer or task default is changed.
  """

  def __init__(
    self,
    env: D3DiagnosticEnv,
    policy: D3DiagnosticPolicy,
    playback: VisualDiagnosticPlayback,
    *,
    viser_server: Any,
    frame_rate: float = 60.0,
  ) -> None:
    super().__init__(env, policy, frame_rate=frame_rate, viser_server=viser_server)
    self._playback = playback
    # Default to paused before any frame is rendered; ``setup`` re-syncs the
    # pause button label.  Rendering and physics never advance unattended.
    self.pause()

  @override
  def setup(self) -> None:
    super().setup()
    self.pause()
    self._sync_ui_state()

  @override
  def _sync_ui_state(self) -> None:
    super()._sync_ui_state()
    self._disable_reference_controls()

  @override
  def _execute_step(self) -> bool:
    if self._playback.stopped:
      self._pause_for_stop()
      return False
    step_ok = super()._execute_step()
    if step_ok and self._playback.stopped:
      # The accepted step reached a terminal/budget stop: pause, sync the
      # status/button, and force one final scene refresh of the real pose.
      self._pause_for_stop()
    return step_ok

  def _pause_for_stop(self) -> None:
    """Pause on a stop and request one final render of the stopped pose.

    No extra physics or inference runs; this only updates UI state and forces
    the paused render policy to submit the final real scene at least once.
    """
    self.pause()
    reasons = getattr(self, "_pending_update_reasons", None)
    if reasons is not None:
      reasons.add(UpdateReason.ACTION)
    scene = getattr(self, "_scene", None)
    if scene is not None and hasattr(scene, "needs_update"):
      scene.needs_update = True
    self._sync_ui_state()

  @override
  def is_running(self) -> bool:
    """Run until the fixed session wall budget is exhausted.

    ``BaseViewer.run`` re-checks this every loop iteration, so the session
    ends at the deadline even while paused/idle; the launcher then closes the
    server, viewer and env in its cleanup.
    """
    if self._playback.enforce_session_budget():
      return False
    return True

  @override
  def _handle_custom_action(self, action: Any, payload: Any) -> bool:
    if isinstance(payload, Mapping) and payload.get("type") in _UNSAFE_PAYLOAD_TYPES:
      # Reject reference scrubbing/seek: physical command state and the
      # reference-free continuation must not be mutated by stock GUI controls.
      return True
    return super()._handle_custom_action(action, payload)

  def _disable_reference_controls(self) -> None:
    env = getattr(self.env, "unwrapped", None)
    manager = getattr(env, "command_manager", None)
    get_term = getattr(manager, "get_term", None)
    if not callable(get_term):
      return
    try:
      command = get_term("motion")
    except (KeyError, TypeError):
      return
    disable = getattr(command, "_set_scrubber_disabled", None)
    if callable(disable):
      disable(True)

  @override
  def _update_status_display(self) -> None:
    playback = self._playback.status()
    status = self.get_status()
    tilt = playback["tilt_degrees"]
    pelvis = playback["pelvis_height_m"]
    tick = playback["tick"]
    sim_time = playback["sim_time_s"]
    stop_line = ""
    if playback["stopped"]:
      stop_line = (
        '<br/><span style="color:#e74c3c;"><strong>STOPPED:</strong> '
        f"{playback['stop_reason']}</span>"
      )
    error_line = ""
    if status.last_error:
      first_line = status.last_error.strip().splitlines()[-1]
      error_line = (
        f'<br/><span style="color:#e74c3c;"><strong>Error:</strong> {first_line}</span>'
      )
    self._status_html.content = f"""
      <div style="font-size: 0.85em; line-height: 1.25; padding: 0 1em 0.5em 1em;">
        <strong style="color:#e67e22;">DIAGNOSTIC ONLY — single branch, NOT a
        matched acceptance</strong><br/>
        <strong>Branch:</strong> {playback["branch"]}
        (<code>{playback["member"]}</code>)<br/>
        <strong>Phase:</strong> {playback["phase"]}<br/>
        <strong>Tick:</strong> {tick} | <strong>Sim:</strong>
        {_format_optional(sim_time, ".2f", "s")}<br/>
        <strong>Tilt:</strong> {_format_optional(tilt, ".1f", "°")} |
        <strong>Pelvis:</strong> {_format_optional(pelvis, ".3f", "m")}<br/>
        <strong>Bootstrap:</strong> {playback["bootstrap_completed"]}/
        {playback["bootstrap_required"]} | <strong>Eval:</strong>
        {playback["evaluation_steps"]}/{playback["max_steps"]}<br/>
        <strong>Playback:</strong> {"Paused" if status.paused else "Running"}
        ({status.speed_label}){stop_line}{error_line}
      </div>
      """


def _format_optional(value: float | None, spec: str, unit: str) -> str:
  if value is None:
    return "—"
  return f"{value:{spec}}{unit}"


__all__ = [
  "BootstrapPair",
  "D3DiagnosticEnv",
  "D3DiagnosticPolicy",
  "D3DiagnosticViewer",
  "VisualDiagnosticPlayback",
  "VisualPlaybackError",
  "tilt_degrees",
]
