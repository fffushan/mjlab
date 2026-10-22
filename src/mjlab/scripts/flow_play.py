"""Run one x0 flow-matching X2 policy in closed loop.

Usage::

    uv run python -m mjlab.scripts.flow_play \
        --checkpoint-file run/checkpoint-best.pt \
        --motion-file reference.npz

The run directory holds a sidecar named after the checkpoint
(``checkpoint-best.pt`` -> ``checkpoint-best.yaml``).  The sidecar names the VAE,
the projection bundle, the control contract, the cohort manifest and the member,
so the player needs no other flag.  Recorded absolute paths are relocated by
suffix when absent, which lets a copied run directory work on another machine.

The player adds no artifact digest.  It loads the VAE model-only and the flow
denoiser without projection or contract identity checks, because playback needs
weights and dimensions, not reproduction proof.  It synthesises a one-member
cohort in a temporary directory so the unchanged runtime builder can be reused.

The reference motion file may be any clip the runtime loader accepts.  Its
basename replaces the cohort clip, and a temporary one-clip reference manifest
keeps the saved teacher parameters paired with their single declared reference.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_STEPS = 300
_BOOTSTRAP_PAIRS = 8
_DEFAULT_FLOW_STEPS = 6
_CONTROL_PERIOD_S = 0.02
_FALL_TILT_DEGREES = 70.0
_FALL_HEIGHT_M = 0.25
_FALL_DEBOUNCE_STEPS = 5

# Suffix anchors tried in order when a recorded absolute path is absent.
_RELOCATION_ANCHORS = ("logs/", "data/", "configs/", "src/", "scripts/")


class FlowPlayError(RuntimeError):
  """One user-facing player failure."""


def _relocate(recorded: str | Path, repo_root: Path, *, role: str) -> Path:
  """Return an existing file for a recorded path, relocating by suffix."""
  path = Path(recorded)
  if path.is_absolute():
    if path.is_file():
      return path
  else:
    candidate = (repo_root / path).resolve()
    if candidate.is_file():
      return candidate
    path = candidate

  text = str(path)
  attempts: list[Path] = []
  for anchor in _RELOCATION_ANCHORS:
    index = text.find(anchor)
    if index == -1:
      continue
    candidate = (repo_root / text[index:]).resolve()
    if candidate.is_file():
      return candidate
    attempts.append(candidate)
  tried = ", ".join(str(item) for item in attempts) if attempts else "no suffix match"
  raise FlowPlayError(
    f"{role} not found: recorded {str(path)!r} is absent and relocation under "
    f"{repo_root} failed ({tried})"
  )


def _read_yaml(path: Path, *, role: str) -> dict[str, Any]:
  try:
    raw = yaml.safe_load(path.read_text())
  except yaml.YAMLError as exc:
    raise FlowPlayError(f"{role} {path} is not plain YAML: {exc}") from exc
  if not isinstance(raw, dict):
    raise FlowPlayError(f"{role} {path} must be a mapping")
  return raw


def _sidecar_for(checkpoint: Path, override: Path | None) -> Path:
  if override is not None:
    path = override.resolve()
    if not path.is_file():
      raise FlowPlayError(f"sidecar not found: {path}")
    return path
  path = checkpoint.with_suffix(".yaml")
  if not path.is_file():
    raise FlowPlayError(
      f"checkpoint sidecar not found: {path}. Put a sidecar named after the "
      "checkpoint beside it, or pass --checkpoint-config."
    )
  return path


def _sidecar_value(sidecar: dict[str, Any], key: str, path: Path) -> str:
  value = sidecar.get(key)
  if not isinstance(value, str) or not value:
    raise FlowPlayError(f"sidecar {path} has no usable {key!r}")
  return value


def _cohort_entry(
  manifest_path: Path, member: str
) -> tuple[dict[str, Any], dict[str, Any]]:
  """Return the recorded cohort manifest and one teacher entry."""
  raw = _read_yaml(manifest_path, role="cohort manifest")
  teachers = raw.get("teachers")
  if not isinstance(teachers, list):
    raise FlowPlayError(f"cohort manifest {manifest_path} has no teacher list")
  for entry in teachers:
    if isinstance(entry, dict) and entry.get("id") == member:
      return raw, entry
  raise FlowPlayError(f"cohort manifest {manifest_path} has no member {member!r}")


def _write_env_config(workspace: Path, recorded: Path, motion_manifest: Path) -> Path:
  """Copy the saved teacher environment config and repoint its reference.

  The saved file carries ``!!python/tuple`` and ``!!python/name`` tags, so this
  edits the recorded text instead of round-tripping it through a YAML dumper.
  Every other byte stays as the teacher wrote it.
  """
  text = recorded.read_text()
  replaced, count = re.subn(
    r"(?m)^(\s*)motion_manifest:.*$",
    lambda match: f"{match.group(1)}motion_manifest: {motion_manifest}",
    text,
  )
  if count != 1:
    raise FlowPlayError(
      f"saved env config {recorded} declares {count} motion_manifest lines; "
      "exactly one is required"
    )
  path = workspace / "env.yaml"
  path.write_text(replaced)
  return path


def _synth_cohort(
  workspace: Path,
  *,
  cohort: dict[str, Any],
  entry: dict[str, Any],
  clip: Path,
  repo_root: Path,
  member: str,
) -> Path:
  """Write a one-member cohort that pairs the saved teacher with the user clip."""
  motion_manifest = workspace / "motions.yaml"
  motion_manifest.write_text(yaml.safe_dump({"motions": [str(clip)]}, sort_keys=False))
  env_config = _write_env_config(
    workspace,
    _relocate(entry["env_config"], repo_root, role="teacher env config"),
    motion_manifest,
  )
  agent_config = workspace / "agent.yaml"
  shutil.copyfile(
    _relocate(entry["agent_config"], repo_root, role="teacher agent config"),
    agent_config,
  )
  record = {
    "version": 1,
    "name": f"{cohort.get('name', 'cohort')}-play",
    "robot": cohort.get("robot"),
    "base_task": cohort.get("base_task"),
    "teachers": [
      {
        "id": member,
        "checkpoint": str(
          _relocate(entry["checkpoint"], repo_root, role="teacher checkpoint")
        ),
        "motion": str(clip),
        "env_config": str(env_config),
        "agent_config": str(agent_config),
        "onnx": str(_relocate(entry["onnx"], repo_root, role="teacher ONNX export")),
        "sampling_weight": 1.0,
      }
    ],
  }
  path = workspace / "cohort.yaml"
  path.write_text(yaml.safe_dump(record, sort_keys=False))
  return path


def _load_vae_model_only(path: Path, device: torch.device):
  """Build the frozen VAE from its own saved schema; no manifest, no digest."""
  from mjlab.tasks.tracking.distillation.checkpoint import (
    _build_inference_model,
    _load_payload,
  )

  payload = _load_payload(path, "cpu")
  model, schema, settings = _build_inference_model(
    payload, device=device, expected_schema=None
  )
  model.to(device)
  model.eval()
  for parameter in model.parameters():
    parameter.requires_grad_(False)
  return model, schema, settings


def _load_flow(path: Path, contract: Any, projection: Any, steps: int):
  from mjlab.tasks.tracking.diffusion.replay_integration import load_flow_denoiser

  return load_flow_denoiser(
    path,
    contract=contract,
    checkpoint_sha256=None,
    projection_bundle=projection,
    verify_projection_hashes=False,
    solver_steps=steps,
  )


class _EpisodeDriver:
  """Advance one runtime episode: reference bootstrap, then reference-free flow.

  The bootstrap encodes the live reference into latents and decodes them back to
  actions, which seeds the continuation with executed history.  After the
  handover the flow model supplies every action.
  """

  def __init__(
    self,
    runtime: Any,
    vae_model: Any,
    continuation_factory: Any,
    *,
    device: torch.device,
    seed: int,
    start_frame: int,
    bootstrap_pairs: int = _BOOTSTRAP_PAIRS,
  ) -> None:
    self._runtime = runtime
    self._vae = vae_model
    self._continuation_factory = continuation_factory
    self._device = device
    self._seed = seed
    self._start_frame = start_frame
    self._bootstrap_pairs = bootstrap_pairs
    self._continuation: Any = None
    self._current: Any = None
    self._bootstrap_left = 0
    self.steps = 0
    self.episodes = 0
    self.falls = 0
    self.last_reason: str | None = None
    self.last_error: str | None = None
    self._fall_frames = 0
    self._begin()

  def _begin(self) -> None:
    self._current = self._runtime.reset(seed=self._seed, start_frame=self._start_frame)
    self._continuation = self._continuation_factory()
    self._bootstrap_left = self._bootstrap_pairs
    self.steps = 0
    self._fall_frames = 0
    self.last_reason = None

  def _physical_fall(self) -> bool:
    """Report a sustained fall using tilt and root height.

    Reference-free terminations are disabled by the runtime, so a fallen robot
    would otherwise lie on the ground for the rest of the session.
    """
    state = self._current.state
    quat = np.asarray(state.root_quaternion_wxyz, dtype=np.float64).reshape(-1)
    height = float(np.asarray(state.root_position, dtype=np.float64).reshape(-1)[2])
    tilt = float(np.degrees(2.0 * np.arccos(min(1.0, abs(quat[0])))))
    if tilt > _FALL_TILT_DEGREES or height < _FALL_HEIGHT_M:
      self._fall_frames += 1
    else:
      self._fall_frames = 0
    return self._fall_frames >= _FALL_DEBOUNCE_STEPS

  def _end_episode(self, reason: str) -> None:
    self.last_reason = reason
    self._begin()
    self.episodes += 1

  def _bootstrap_action(self) -> tuple[Any, np.ndarray]:
    reference = torch.from_numpy(
      np.ascontiguousarray(self._current.reference, dtype=np.float32)
    ).to(self._device)[None]
    conditioning = torch.from_numpy(
      np.ascontiguousarray(self._current.conditioning, dtype=np.float32)
    ).to(self._device)[None]
    with torch.no_grad():
      mu, _ = self._vae.encode(reference)
      action = self._vae.decode(mu, conditioning)
    latent = mu.detach().cpu().numpy().reshape(-1)
    return latent, action.detach().cpu().numpy().reshape(-1)

  def step(self) -> None:
    """Advance one control step; a finished episode restarts automatically."""
    if self._bootstrap_left > 0:
      latent, action = self._bootstrap_action()
      self._continuation.record_executed(copy.deepcopy(self._current.state), latent)
      transition = self._runtime.step(action)
      self.steps += 1
      self._bootstrap_left -= 1
      if transition.terminated or transition.truncated:
        self._end_episode(transition.reason or "reset_during_bootstrap")
        return
      if self._bootstrap_left == 0:
        self._current = self._runtime.enter_reference_free()
      return

    step = self._continuation.step(
      copy.deepcopy(self._current.state),
      np.array(self._current.conditioning, copy=True),
      tick=int(self._current.tick),
    )
    transition = self._runtime.step(step.action)
    self.steps += 1
    if transition.terminated or transition.truncated:
      self._end_episode(transition.reason or "reset")
      return
    self._current = transition.observation
    if self._physical_fall():
      self.falls += 1
      self._end_episode("fall")


def _build_viewer(env: Any, driver: _EpisodeDriver):
  """Native viewer whose step hook advances the runtime instead of the policy."""
  import traceback

  from mjlab.viewer import NativeMujocoViewer
  from mjlab.viewer.base import VerbosityLevel

  class _FlowPlayerViewer(NativeMujocoViewer):
    def _execute_step(self) -> bool:
      try:
        driver.step()
        self._step_count += 1
        self._stats_steps += 1
        return True
      except Exception:
        self._last_error = traceback.format_exc()
        driver.last_error = self._last_error
        self.log(f"[ERROR] {self._last_error}", VerbosityLevel.SILENT)
        self.pause()
        return False

  return _FlowPlayerViewer(env=env, policy=lambda obs: obs)


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="flow_play",
    description="Run one x0 flow-matching X2 policy in closed loop.",
  )
  parser.add_argument(
    "--checkpoint-file", type=Path, required=True, help="flow policy checkpoint"
  )
  parser.add_argument(
    "--motion-file", type=Path, required=True, help="reference motion npz"
  )
  parser.add_argument(
    "--checkpoint-config",
    type=Path,
    default=None,
    help="sidecar YAML (default: checkpoint stem with .yaml)",
  )
  parser.add_argument(
    "--vae", type=Path, default=None, help="override the VAE checkpoint"
  )
  parser.add_argument(
    "--projection", type=Path, default=None, help="override the projection bundle"
  )
  parser.add_argument(
    "--contract", type=Path, default=None, help="override the control contract"
  )
  parser.add_argument(
    "--repo-root", type=Path, default=_REPO_ROOT, help="artifact root"
  )
  parser.add_argument(
    "--start-frame", type=int, default=0, help="clip frame the episode starts at"
  )
  parser.add_argument(
    "--steps", type=int, default=_DEFAULT_STEPS, help="scored control steps"
  )
  parser.add_argument("--seed", type=int, default=0, help="reset seed")
  parser.add_argument(
    "--flow-steps", type=int, default=_DEFAULT_FLOW_STEPS, help="Euler ODE steps"
  )
  parser.add_argument("--device", default="cuda:0", help="torch device")
  parser.add_argument(
    "--viewer",
    action="store_true",
    help="open the native MuJoCo viewer instead of a bounded headless run",
  )
  parser.add_argument(
    "--viewer-frames",
    type=int,
    default=None,
    help="close the viewer after this many frames (default: run until closed)",
  )
  parser.add_argument(
    "--json", action="store_true", help="print the full matched-pair report"
  )
  return parser


def _summary(report: Any) -> dict[str, Any]:
  branch = report.diffusion
  if branch is None:
    return {"error": "the matched pair produced no diffusion branch"}
  return {
    "outcome": branch.outcome,
    "stop_reason": branch.stop_reason,
    "survival_observed": bool(branch.survival_observed),
    "steps_completed": int(branch.evaluation_completed),
    "steps_attempted": int(branch.evaluation_attempted),
    "bootstrap_pairs": int(branch.bootstrap_completed),
    "simulation_duration_s": round(float(branch.simulation_duration_s), 3),
    "peak_tilt_degrees": branch.peak_tilt_degrees,
    "minimum_pelvis_height_m": branch.minimum_pelvis_height_m,
    "errors": list(branch.errors),
    "baseline_outcome": None if report.baseline is None else report.baseline.outcome,
  }


def _run(args: argparse.Namespace) -> dict[str, Any]:
  repo_root = args.repo_root.resolve()
  checkpoint = args.checkpoint_file.resolve()
  clip = args.motion_file.resolve()
  if not checkpoint.is_file():
    raise FlowPlayError(f"checkpoint not found: {checkpoint}")
  if not clip.is_file():
    raise FlowPlayError(f"motion file not found: {clip}")

  sidecar_path = _sidecar_for(checkpoint, args.checkpoint_config)
  sidecar = _read_yaml(sidecar_path, role="sidecar")
  member = _sidecar_value(sidecar, "member", sidecar_path)

  from mjlab.tasks.tracking.diffusion.replay_integration import (
    load_active_contract,
    load_projection_bundle,
  )

  def resolve(override: Path | None, key: str, role: str) -> Path:
    recorded = (
      override if override is not None else _sidecar_value(sidecar, key, sidecar_path)
    )
    return _relocate(recorded, repo_root, role=role)

  contract = load_active_contract(
    resolve(args.contract, "contract", "control contract")
  )
  projection = load_projection_bundle(
    resolve(args.projection, "projection", "projection bundle"), contract=contract
  )
  vae_path = resolve(args.vae, "vae_checkpoint", "VAE checkpoint")
  manifest_path = resolve(None, "manifest", "cohort manifest")
  cohort, entry = _cohort_entry(manifest_path, member)
  device = torch.device(args.device)

  vae_model, schema, settings = _load_vae_model_only(vae_path, device)
  flow = _load_flow(checkpoint, contract, projection, args.flow_steps)

  from mjlab.tasks.tracking.diffusion.closed_loop import (
    EvaluationBudget,
    run_matched_pair,
  )
  from mjlab.tasks.tracking.diffusion.closed_loop_runtime_x2 import (
    build_closed_loop_runtime,
  )
  from mjlab.tasks.tracking.diffusion.conditioned_continuation import (
    build_conditioned_continuation,
  )
  from mjlab.tasks.tracking.flow_matching.closed_loop import FlowSamplingAdapter
  from mjlab.tasks.tracking.flow_matching.sampler import FlowSamplerConfig

  workspace = Path(tempfile.mkdtemp(prefix="flow-play-"))
  try:
    synth_manifest = _synth_cohort(
      workspace,
      cohort=cohort,
      entry=entry,
      clip=clip,
      repo_root=repo_root,
      member=member,
    )

    def runtime_factory() -> Any:
      return build_closed_loop_runtime(
        manifest_path=synth_manifest,
        repo_root=repo_root,
        member=member,
        device=str(device),
        seed=args.seed,
        compare_reference_arrays=False,
      )

    sampler = FlowSamplingAdapter(
      denoiser=flow.model,
      config=FlowSamplerConfig(grid=flow.grid, process=flow.process),
      state_dimension=contract.projected_state_dimension,
    )

    def decode(latent: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
      with torch.no_grad():
        return vae_model.decode(latent, conditioning)

    def continuation_factory() -> Any:
      generator = torch.Generator(device=device)
      generator.manual_seed(args.seed)
      return build_conditioned_continuation(
        conditioning=None,
        denoiser=flow.model,
        decode=decode,
        projection=projection,
        contract=contract,
        device=device,
        generator=generator,
        sampler=sampler,
      )

    flow.model.to(device)
    flow.model.eval()

    if args.viewer:
      print(
        "viewer note: the reference ghost is frozen by design; after the "
        "reference-free handover the flow model supplies the motion, and the "
        "episode restarts after a fall",
        flush=True,
      )
      runtime = runtime_factory()
      try:
        driver = _EpisodeDriver(
          runtime,
          vae_model,
          continuation_factory,
          device=device,
          seed=args.seed,
          start_frame=args.start_frame,
        )
        viewer = _build_viewer(runtime.env, driver)
        viewer.run(num_steps=args.viewer_frames)
      finally:
        runtime.close()
      return {
        "mode": "viewer",
        "control_steps": driver.steps,
        "episodes": driver.episodes,
        "falls": driver.falls,
        "last_reset_reason": driver.last_reason,
        "last_error": driver.last_error,
        "errors": [] if driver.last_error is None else [driver.last_error],
      }

    report = run_matched_pair(
      runtime_factory,
      model=vae_model,
      continuation_factory=continuation_factory,
      seed=args.seed,
      start_frame=args.start_frame,
      budget=EvaluationBudget(
        max_steps=args.steps,
        bootstrap_pairs=_BOOTSTRAP_PAIRS,
        period_seconds=_CONTROL_PERIOD_S,
      ),
      device=device,
      latent_normalizer=projection.normalize_latent,
    )
  finally:
    shutil.rmtree(workspace, ignore_errors=True)

  if args.json:
    return report.as_dict()
  summary = _summary(report)
  summary.update(
    {
      "checkpoint": str(checkpoint),
      "motion_file": str(clip),
      "sidecar": str(sidecar_path),
      "vae": str(vae_path),
      "member": member,
      "vae_reference_dim": int(schema.reference_dim),
      "vae_latent_dim": int(settings.latent_dim),
      "start_frame": int(args.start_frame),
      "seed": int(args.seed),
    }
  )
  return summary


def main(argv: list[str] | None = None) -> int:
  args = _parser().parse_args(argv)
  try:
    payload = _run(args)
  except FlowPlayError as exc:
    print(f"flow_play: {exc}", file=sys.stderr)
    return 2
  except Exception as exc:  # model or runtime failure
    print(f"flow_play failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 1
  print(json.dumps(payload, indent=2, sort_keys=True))
  if isinstance(payload, dict) and payload.get("errors"):
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
