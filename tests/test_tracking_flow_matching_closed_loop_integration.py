"""CPU-only tests for ``--model-family`` dispatch in the D3 closed-loop CLIs.

These tests exercise the parser, the checkpoint-sidecar family gate, preflight
dispatch, the flow guidance rejection and the family-aware report fields.  They
load no real checkpoint, construct no simulator and claim no physical evidence.
The frozen loaders are replaced by small deterministic fakes; the flow process,
grid and projection are real objects, so the reported metadata is real.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pytest
import torch

_EXPERIMENT_DIR = (
  Path(__file__).parents[1] / "scripts/experiments/diffusion_d3_closed_loop"
)


def _load_sibling(name: str, filename: str) -> Any:
  """Load one sibling CLI module and register it under ``name``.

  ``evaluate_parallel.py`` imports the evaluator as ``import evaluate``, so the
  evaluator is registered under that exact name before the driver loads.  Both
  drivers then share one module object and a monkeypatch reaches both.
  """
  spec = importlib.util.spec_from_file_location(name, _EXPERIMENT_DIR / filename)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


evaluate = _load_sibling("evaluate", "evaluate.py")
evaluate_parallel = _load_sibling("evaluate_parallel", "evaluate_parallel.py")
view = _load_sibling("mjlab_d3_closed_loop_view_integration_test", "view.py")

from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT  # noqa: E402
from mjlab.tasks.tracking.diffusion.handover import (  # noqa: E402
  ReferenceFreeContinuation,
)
from mjlab.tasks.tracking.diffusion.parallel_closed_loop_x2 import (  # noqa: E402
  BatchedReferenceFreeContinuation,
)
from mjlab.tasks.tracking.diffusion.projection import (  # noqa: E402
  ProjectionBundle,
  fit_feature_stats,
)
from mjlab.tasks.tracking.diffusion.schedule import (  # noqa: E402
  DiffusionSchedule,
  InferenceGrid,
)
from mjlab.tasks.tracking.flow_matching.process import LinearNoiseProcess  # noqa: E402
from mjlab.tasks.tracking.flow_matching.sampler import FlowGrid  # noqa: E402

_FLOW_SCHEMA = "mjlab-d3-flow-matching-checkpoint-v1"
_DIFFUSION_SCHEMA = "mjlab-d3-diffusion-checkpoint-v1"


class _FlowDenoiserModel:
  """Callable stand-in for the loaded flow network."""

  def __call__(self, sample: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
    return sample


class _DiffusionDenoiserModel:
  """Callable stand-in for the loaded diffusion network."""

  def __call__(self, sample: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
    return sample


class _Decoder:
  """Decoder stand-in with the frozen ``decode(latent, conditioning)`` surface."""

  def decode(self, latent: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
    return latent[..., :31]


def _stats(width: int, *, seed: int) -> Any:
  rng = np.random.default_rng(seed)
  return fit_feature_stats([rng.normal(size=(64, width)) + 3.0], std_floor=1e-6)


def _projection() -> ProjectionBundle:
  return ProjectionBundle.create(_stats(135, seed=11), _stats(32, seed=12))


def _flow_record(solver_steps: int = 20) -> SimpleNamespace:
  process = LinearNoiseProcess(1000)
  return SimpleNamespace(
    model=_FlowDenoiserModel(),
    process=process,
    grid=FlowGrid.uniform(calls=solver_steps, max_level=process.max_level),
    family="x0_linear_flow",
    process_identity=process.identity_hash(),
    data_identity="b" * 64,
    model_identity_hash="c" * 64,
    global_step=400000,
    config_hash="d" * 64,
    projection_hashes={},
    best_validation_loss=0.5,
    selection_split="validation",
    ema_selected=True,
    ema_updates=7,
    fp32_checked=True,
    config_seed=3,
  )


def _diffusion_record() -> SimpleNamespace:
  schedule = DiffusionSchedule.build()
  return SimpleNamespace(
    model=_DiffusionDenoiserModel(),
    schedule=schedule,
    grid=InferenceGrid.uniform(schedule),
    global_step=100000,
    config_hash="d" * 64,
    contract_identity="b" * 64,
    projection_hashes={},
    best_validation_loss=0.4,
    selection_split="validation",
    ema_selected=True,
    ema_updates=5,
    fp32_checked=True,
    config_seed=1,
  )


def _vae_record() -> SimpleNamespace:
  return SimpleNamespace(
    model=_Decoder(),
    reference_dim=68,
    conditioning_dim=99,
    latent_dim=32,
    action_dim=31,
    relocated_artifact_roles=(),
    joint_order=tuple(f"joint-{index}" for index in range(31)),
  )


def _checkpoint(tmp_path: Path, schema: str) -> Path:
  """Write one placeholder checkpoint with its required sidecar YAML."""
  run_dir = tmp_path / "run" / "seed0"
  run_dir.mkdir(parents=True, exist_ok=True)
  checkpoint = run_dir / "checkpoint-best.pt"
  checkpoint.write_bytes(b"placeholder")
  (run_dir / "checkpoint-best.yaml").write_text(
    "\n".join(
      [
        f"schema: {schema}",
        f"repo_root: {tmp_path}",
        "contract: contract.yaml",
        "projection: projection.npz",
        "vae_checkpoint: vae.pt",
        "manifest: manifest.yaml",
        "member: vgk000",
        "seed: 3",
        "start_frame: 7",
      ]
    )
    + "\n",
    encoding="utf-8",
  )
  return checkpoint


def _request(
  tmp_path: Path,
  checkpoint: Path,
  family: str | None,
  extra: Sequence[str] = (),
) -> Namespace:
  argv = [
    "--checkpoint",
    str(checkpoint),
    "--device",
    "cpu",
    "--output-dir",
    str(tmp_path / "out"),
    *extra,
  ]
  if family is not None:
    argv = ["--model-family", family, *argv]
  return evaluate.build_parser().parse_args(argv)


def _patch_flow_loaders(
  monkeypatch: pytest.MonkeyPatch, record: SimpleNamespace
) -> list[str]:
  """Patch every frozen loader and record which denoiser loader ran."""
  calls: list[str] = []
  teacher = SimpleNamespace(reference=SimpleNamespace(frames=100, fps=50.0), hashes={})
  monkeypatch.setattr(evaluate, "load_active_contract", lambda path: DEFAULT_CONTRACT)
  monkeypatch.setattr(
    evaluate, "load_manifest", lambda path, repo_root: SimpleNamespace(sha256="e" * 64)
  )
  monkeypatch.setattr(
    evaluate,
    "resolve_cohort",
    lambda manifest: SimpleNamespace(teacher=lambda member: teacher),
  )
  monkeypatch.setattr(evaluate, "load_projection_bundle", lambda *a, **k: _projection())
  monkeypatch.setattr(evaluate, "load_vae_decoder", lambda *a, **k: _vae_record())

  def flow_loader(*args: Any, **kwargs: Any) -> SimpleNamespace:
    calls.append("flow")
    if "solver_steps" in kwargs:
      return _flow_record(int(kwargs["solver_steps"]))
    return record

  def diffusion_loader(*args: Any, **kwargs: Any) -> SimpleNamespace:
    calls.append("diffusion")
    return record

  monkeypatch.setattr(evaluate, "load_flow_denoiser", flow_loader)
  monkeypatch.setattr(evaluate, "load_denoiser", diffusion_loader)
  monkeypatch.setattr(evaluate, "_source_fingerprints", lambda: {"modules": {}})
  return calls


def test_parser_selects_the_model_family_and_defaults_to_diffusion() -> None:
  parser = evaluate.build_parser()
  default = parser.parse_args(["--checkpoint", "checkpoint.pt"])
  assert default.model_family == "diffusion"
  flow = parser.parse_args(["--checkpoint", "checkpoint.pt", "--model-family", "flow"])
  assert flow.model_family == "flow"
  with pytest.raises(SystemExit):
    parser.parse_args(["--checkpoint", "checkpoint.pt", "--model-family", "ddim"])


def test_flow_sidecar_requires_the_flow_family(tmp_path: Path) -> None:
  checkpoint = _checkpoint(tmp_path, _FLOW_SCHEMA)
  default = _request(tmp_path, checkpoint, None)
  with pytest.raises(evaluate.ClosedLoopCliError, match="does not declare the"):
    evaluate._validate_request(default)

  flow = _request(tmp_path, checkpoint, "flow")
  evaluate._validate_request(flow)
  assert flow.member == "vgk000"
  assert flow.seed == 3
  assert flow.start_frame == 7


def test_diffusion_sidecar_rejects_the_flow_family(tmp_path: Path) -> None:
  checkpoint = _checkpoint(tmp_path, _DIFFUSION_SCHEMA)
  flow = _request(tmp_path, checkpoint, "flow")
  with pytest.raises(evaluate.ClosedLoopCliError, match="does not declare the"):
    evaluate._validate_request(flow)


def test_flow_family_accepts_hard_keyframes_and_refuses_soft_pose_goals() -> None:
  """The family supports hard keyframes; PoseCost guidance stays DDIM-only."""
  from types import SimpleNamespace

  keyframes_only = SimpleNamespace(goals=(), strength=0.0)
  evaluate._require_task_family_support("flow", keyframes_only)
  evaluate._require_task_family_support("diffusion", keyframes_only)

  with_goals = SimpleNamespace(goals=(object(),), strength=0.0)
  with pytest.raises(evaluate.ClosedLoopCliError, match="hard keyframes only"):
    evaluate._require_task_family_support("flow", with_goals)
  evaluate._require_task_family_support("diffusion", with_goals)

  strength_only = SimpleNamespace(goals=(), strength=0.5)
  with pytest.raises(evaluate.ClosedLoopCliError, match="hard keyframes only"):
    evaluate._require_task_family_support("flow", strength_only)


@pytest.mark.parametrize("reference_dim", [68, 130])
def test_flow_prepare_dispatches_the_flow_loader_and_reports_euler_x0(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reference_dim: int
) -> None:
  checkpoint = _checkpoint(tmp_path, _FLOW_SCHEMA)
  calls = _patch_flow_loaders(monkeypatch, _flow_record())
  vae = _vae_record()
  vae.reference_dim = reference_dim
  monkeypatch.setattr(evaluate, "load_vae_decoder", lambda *a, **k: vae)
  prepared = evaluate._prepare(_request(tmp_path, checkpoint, "flow"))
  assert calls == ["flow"]
  assert prepared.family == "flow"

  report = evaluate._preflight_report(prepared)
  sampler = report["sampler"]
  assert sampler["model_family"] == "flow"
  assert sampler["sampler_method"] == "euler_x0"
  assert (
    sampler["solver_updates"]
    == (prepared.denoiser.grid.calls)
    == evaluate.DEFAULT_CLOSED_LOOP_SOLVER_STEPS
    == 6
  )
  assert sampler["inference_grid"] == prepared.denoiser.grid.as_dict()
  assert sampler["start_level"] == prepared.denoiser.process.max_level == 1000
  assert sampler["process_identity"] == prepared.denoiser.process_identity
  assert "ddim_updates" not in json.dumps(report)

  provenance = evaluate._provenance(prepared, 0.0)
  assert provenance["sampler"]["sampler_method"] == "euler_x0"
  assert provenance["sampler"]["solver_updates"] == 6
  assert provenance["checkpoint"]["process_family"] == "x0_linear_flow"
  assert provenance["checkpoint"]["data_identity"] == prepared.denoiser.data_identity
  assert "contract_identity" not in provenance["checkpoint"]
  assert report["vae"]["reference_dimension"] == reference_dim
  assert f"reference {reference_dim}-D" in sampler["vae_convention"]
  assert provenance["conditioning"]["reference_dimension"] == reference_dim
  assert ("motion_lookahead" in provenance["conditioning"]["reference_source"]) == (
    reference_dim == 130
  )


def test_diffusion_prepare_keeps_the_default_loader_and_ddim_reporting(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  checkpoint = _checkpoint(tmp_path, _DIFFUSION_SCHEMA)
  calls = _patch_flow_loaders(monkeypatch, _diffusion_record())
  prepared = evaluate._prepare(_request(tmp_path, checkpoint, None))
  assert calls == ["diffusion"]
  assert prepared.family == "diffusion"

  sampler = evaluate._preflight_report(prepared)["sampler"]
  assert sampler["model_family"] == "diffusion"
  assert sampler["sampler_method"] == "ddim"
  assert sampler["solver_updates"] == evaluate._FROZEN_DDIM_STEPS
  assert sampler["ddim_updates"] == evaluate._FROZEN_DDIM_STEPS
  assert sampler["start_level"] == evaluate._FROZEN_START_LEVEL
  assert "start_distribution" not in sampler


def test_flow_continuation_factory_injects_the_flow_sampler(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  checkpoint = _checkpoint(tmp_path, _FLOW_SCHEMA)
  _patch_flow_loaders(monkeypatch, _flow_record())
  prepared = evaluate._prepare(_request(tmp_path, checkpoint, "flow"))
  adapter = evaluate._flow_sampling_adapter(prepared)
  assert adapter.config.grid.levels == prepared.denoiser.grid.levels
  assert adapter.config.process.identity_hash() == (
    prepared.denoiser.process.identity_hash()
  )
  assert adapter.state_dimension == prepared.contract.projected_state_dimension

  continuation = evaluate._continuation_factory(prepared)()
  assert isinstance(continuation, ReferenceFreeContinuation)
  # No diffusion schedule, grid or SamplerConfig may reach a flow continuation:
  # the constructor fails closed when any of them is passed with a sampler.
  assert continuation.schedule is None


def test_diffusion_continuation_factory_keeps_the_schedule(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  checkpoint = _checkpoint(tmp_path, _DIFFUSION_SCHEMA)
  record = _diffusion_record()
  _patch_flow_loaders(monkeypatch, record)
  prepared = evaluate._prepare(_request(tmp_path, checkpoint, None))
  continuation = evaluate._continuation_factory(prepared)()
  assert continuation.schedule is record.schedule


def test_parallel_flow_continuation_takes_the_injected_sampler(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  checkpoint = _checkpoint(tmp_path, _FLOW_SCHEMA)
  _patch_flow_loaders(monkeypatch, _flow_record())
  prepared = evaluate._prepare(_request(tmp_path, checkpoint, "flow"))
  continuation = evaluate_parallel._build_continuation(prepared, torch.device("cpu"), 2)
  assert isinstance(continuation, BatchedReferenceFreeContinuation)
  assert continuation.rows == 2
  assert continuation.schedule is None


def test_flow_solver_step_count_reaches_the_loader_and_the_report(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  checkpoint = _checkpoint(tmp_path, _FLOW_SCHEMA)
  _patch_flow_loaders(monkeypatch, _flow_record())
  prepared = evaluate._prepare(
    _request(tmp_path, checkpoint, "flow", ["--flow-euler-steps", "9"])
  )
  sampler = evaluate._preflight_report(prepared)["sampler"]
  assert sampler["solver_updates"] == 9
  assert sampler["inference_grid"]["calls"] == 9
  assert sampler["inference_grid"]["levels"][0] == 1000
  assert sampler["inference_grid"]["levels"][-1] == 0
  adapter = evaluate._flow_sampling_adapter(prepared)
  assert adapter.config.grid.calls == 9


def test_default_flow_solver_step_count_is_the_closed_loop_choice(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  checkpoint = _checkpoint(tmp_path, _FLOW_SCHEMA)
  _patch_flow_loaders(monkeypatch, _flow_record())
  prepared = evaluate._prepare(_request(tmp_path, checkpoint, "flow"))
  assert (
    evaluate._preflight_report(prepared)["sampler"]["solver_updates"]
    == evaluate.DEFAULT_CLOSED_LOOP_SOLVER_STEPS
    == 6
  )


def test_flow_solver_steps_are_refused_for_the_diffusion_family(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  checkpoint = _checkpoint(tmp_path, _DIFFUSION_SCHEMA)
  _patch_flow_loaders(monkeypatch, _diffusion_record())
  args = _request(tmp_path, checkpoint, None, ["--flow-euler-steps", "9"])
  with pytest.raises(evaluate.ClosedLoopCliError, match="applies to --model-family"):
    evaluate._validate_request(args)
  flow_checkpoint = _checkpoint(tmp_path / "flow", _FLOW_SCHEMA)
  with pytest.raises(evaluate.ClosedLoopCliError, match="positive integer"):
    evaluate._validate_request(
      _request(tmp_path / "flow", flow_checkpoint, "flow", ["--flow-euler-steps", "0"])
    )


def test_parallel_preflight_reports_the_model_family(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  checkpoint = _checkpoint(tmp_path, _FLOW_SCHEMA)
  _patch_flow_loaders(monkeypatch, _flow_record())
  prepared = evaluate._prepare(_request(tmp_path, checkpoint, "flow"))
  monkeypatch.setattr(evaluate, "_prepare", lambda args: prepared)
  code = evaluate_parallel.main(
    [
      "--checkpoint",
      str(checkpoint),
      "--output-dir",
      str(tmp_path / "parallel"),
      "--device",
      "cpu",
    ]
  )
  assert code == 0
  payload = json.loads(capsys.readouterr().out)
  assert payload["model_family"] == "flow"
  assert payload["execute"].startswith("not-run")
  assert payload["checkpoint"] == str(checkpoint)
  assert payload["sampler"]["solver_updates"] == 6
  assert payload["sampler"]["inference_grid"] == prepared.denoiser.grid.as_dict()
  assert payload["sampler"]["process_identity"] == prepared.denoiser.process_identity


def test_sequence_inspector_rejects_flow_before_loading(
  monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  inspector = _load_sibling("flow_inspector_rejection_test", "inspect_sequence.py")

  def unexpected_load(args: Any) -> Any:
    pytest.fail("flow sequence inspection must not load artifacts")

  monkeypatch.setattr(inspector.evaluate, "_prepare", unexpected_load)
  assert inspector.main(["--model-family", "flow", "--checkpoint", "missing.pt"]) == 1
  assert "does not support --model-family flow" in capsys.readouterr().err


def test_viewer_flow_preflight_labels_the_family_without_a_ddim_grid(
  tmp_path: Path,
) -> None:
  prepared = Namespace(
    args=Namespace(
      branch="diffusion",
      host="127.0.0.1",
      port=8080,
      frame_rate=60.0,
      continuation_steps=5,
      max_wall_seconds=120.0,
      start_frame=7,
      member="vgk000",
      seed=3,
      contract=tmp_path / "contract.yaml",
      vae_checkpoint=tmp_path / "vae.pt",
      output_dir=tmp_path / "run",
    ),
    contract=DEFAULT_CONTRACT,
    clip_frames=100,
    clip_fps=50.0,
    family="flow",
    denoiser=_flow_record(),
    device=torch.device("cpu"),
    repo_root=tmp_path,
    source_fingerprints={},
  )
  report = view._preflight_report(prepared)
  assert report["model_family"] == "flow"
  assert report["sampler"]["sampler_method"] == "euler_x0"
  assert "ddim_updates" not in report["sampler"]
