"""Tests for the X2 joint-pose controller bundle exporter.

The checkpoint is a tiny ``width=32, layers=1, heads=2`` denoiser with the real
``[41, 68]`` window, and the dataset is only a synthetic ``projection.npz``.
Every export runs on CPU.  The tests check the descriptor schema, the sampler
ladder, the asset digests, the EMA/model selection and each refusal the exporter
promises.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper

from mjlab.tasks.tracking.flow_matching.sampler import FlowGrid
from mjlab.tasks.tracking.joint_pose.config import (
  FlowMatchingTrainerCfg,
  JointPoseDatasetCfg,
)
from mjlab.tasks.tracking.joint_pose.model import build_denoiser
from mjlab.tasks.tracking.joint_pose.runner import CHECKPOINT_FORMAT

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/export_x2_joint_pose_bundle.py"
_SMALL_MODEL = {"width": 32, "layers": 1, "heads": 2}
_MAX_LEVEL = 1000
_STEPS = 6


@pytest.fixture(scope="module")
def export_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("x2_joint_pose_bundle_export", _SCRIPT)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1 << 20), b""):
      digest.update(block)
  return digest.hexdigest()


def _write_projection(data_dir: Path, *, length: int = 68) -> None:
  rng = np.random.default_rng(7)
  mean = rng.standard_normal(length).astype(np.float32)
  std = np.abs(rng.standard_normal(length)).astype(np.float32) + 0.5
  np.savez(
    data_dir / "projection.npz",
    mean=mean,
    std=std,
    channels=np.arange(length, dtype=np.int32),
  )


def _write_tracker(path: Path, *, lookahead: str = "10") -> None:
  observation = np.zeros((226, 31), dtype=np.float32)
  observation[0, 0] = 1.0
  graph = helper.make_graph(
    [helper.make_node("MatMul", ["obs", "weight"], ["actions"])],
    "synthetic_tracker",
    [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 226])],
    [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 31])],
    [
      helper.make_tensor(
        "weight",
        TensorProto.FLOAT,
        [226, 31],
        observation.reshape(-1).tolist(),
      )
    ],
  )
  model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
  model.ir_version = 9
  if lookahead is not None:
    item = model.metadata_props.add()
    item.key = "motion_lookahead_steps"
    item.value = lookahead
  onnx.save(model, path)


def _checkpoint_payload(
  data_dir: Path, *, final_norm: bool = False, with_ema: bool = True
) -> dict[str, object]:
  dataset = JointPoseDatasetCfg(data_dir=str(data_dir))
  model = build_denoiser(
    dataset,
    max_level=_MAX_LEVEL,
    final_norm=final_norm,
    init_seed=123,
    **_SMALL_MODEL,
  )
  state = {name: value.detach().clone() for name, value in model.state_dict().items()}
  trainer = FlowMatchingTrainerCfg(
    max_steps=2,
    warmup_updates=1,
    max_level=_MAX_LEVEL,
    mixed_precision="none",
    final_norm=final_norm,
  )
  payload: dict[str, object] = {
    "format": CHECKPOINT_FORMAT,
    "global_step": 7000,
    "model": state,
    "trainer": trainer.as_dict(),
    "dataset_cfg": dataset.as_dict(),
    "dataset_identity": {
      "dataset_type": "JointPoseWindowDataset",
      "data_dir": str(data_dir),
      "window_count": 3,
      "channels": 68,
      "window_steps": 41,
      "past_steps": 8,
    },
    "model_settings": asdict(model.settings),
    "process": {"max_level": _MAX_LEVEL},
  }
  if with_ema:
    payload["ema"] = {
      "power": 0.75,
      "max_decay": 0.9999,
      "updates": 7000,
      "shadow": {
        name: value + 1.0e-3 if value.is_floating_point() else value.clone()
        for name, value in state.items()
      },
    }
  return payload


def _inputs(
  tmp_path: Path,
  *,
  final_norm: bool = False,
  with_ema: bool = True,
  projection_length: int = 68,
  token_dimension: int | None = None,
  sequence_length: int | None = None,
  tracker_lookahead: str = "10",
) -> dict[str, Path | dict[str, object]]:
  data_dir = tmp_path / "data"
  data_dir.mkdir(parents=True)
  _write_projection(data_dir, length=projection_length)
  payload = _checkpoint_payload(data_dir, final_norm=final_norm, with_ema=with_ema)
  settings = dict(payload["model_settings"])  # type: ignore[arg-type]
  if token_dimension is not None:
    settings["token_dimension"] = token_dimension
  if sequence_length is not None:
    settings["sequence_length"] = sequence_length
  payload["model_settings"] = settings
  checkpoint = tmp_path / "model_0007000.pt"
  torch.save(payload, checkpoint)
  tracker = tmp_path / "policy.onnx"
  _write_tracker(tracker, lookahead=tracker_lookahead)
  return {
    "checkpoint": checkpoint,
    "data_dir": data_dir,
    "tracker": tracker,
    "payload": payload,
  }


def _cli(
  inputs: dict[str, Path | dict[str, object]],
  output: Path,
  *args: str,
) -> list[str]:
  return [
    "--checkpoint",
    str(inputs["checkpoint"]),
    "--data-dir",
    str(inputs["data_dir"]),
    "--tracker-onnx",
    str(inputs["tracker"]),
    "--output-dir",
    str(output),
    *args,
  ]


def test_export_writes_descriptor_and_assets(
  export_tool: ModuleType, tmp_path: Path
) -> None:
  inputs = _inputs(tmp_path)
  output = tmp_path / "bundle"
  assert export_tool.main(_cli(inputs, output, "--max-level", "1000")) == 0
  assert {path.name for path in output.iterdir()} == {
    "SHA256SUMS",
    "denoiser.onnx",
    "descriptor.json",
    "export-report.json",
    "reference_stats.json",
    "tracker.onnx",
  }

  descriptor = json.loads((output / "descriptor.json").read_text())
  assert set(descriptor) == {
    "family",
    "schema_version",
    "task",
    "dimensions",
    "channel_layout",
    "sampler",
    "regeneration",
    "assets",
    "defaults",
  }
  assert descriptor["family"] == "joint_pose_tracking"
  assert descriptor["schema_version"] == 1
  assert set(descriptor["task"]) == {
    "denoiser_identity",
    "tracker_identity",
    "dataset_identity",
    "source_revision_x2",
    "source_revision_mjlab",
    "mode",
  }
  assert descriptor["task"]["denoiser_identity"] == "model_0007000@ema"
  assert descriptor["task"]["source_revision_x2"] == "unspecified"
  assert descriptor["task"]["mode"] == "generated_reference"
  assert descriptor["dimensions"] == {
    "channels": 68,
    "window": 41,
    "past": 8,
    "lookahead_steps": 10,
    "joint": 31,
    "action": 31,
    "observation": 226,
    "control_period_seconds": 0.02,
  }
  assert descriptor["channel_layout"] == {
    "root_rot6d": [0, 6],
    "joint_positions": [6, 37],
    "joint_velocities": [37, 68],
  }
  assert descriptor["sampler"] == {
    "method": "euler_x0",
    "max_level": _MAX_LEVEL,
    "steps": _STEPS,
    "levels": list(FlowGrid.uniform(calls=_STEPS, max_level=_MAX_LEVEL).levels),
  }
  assert descriptor["regeneration"] == {"period_ticks": 16}
  assert descriptor["defaults"] == {
    "max_source_age_seconds": 0.5,
    "history_gap_ticks": 2,
    "exhaustion_fallback_ticks": 20,
    "yaw_jump_warn_deg": 15.0,
  }
  assert descriptor["sampler"]["levels"] == [1000, 833, 667, 500, 333, 167, 0]

  assert set(descriptor["assets"]) == {
    "denoiser_onnx",
    "reference_stats_json",
    "tracker_onnx",
  }
  for asset_name, asset in descriptor["assets"].items():
    assert set(asset) == {"path", "sha256"}
    path = output / asset["path"]
    assert path.is_file(), asset_name
    assert asset["sha256"] == _sha256(path)
  assert descriptor["assets"]["denoiser_onnx"]["path"] == "denoiser.onnx"
  assert descriptor["assets"]["reference_stats_json"]["path"] == (
    "reference_stats.json"
  )
  assert descriptor["assets"]["tracker_onnx"]["path"] == "tracker.onnx"
  assert descriptor["task"]["tracker_identity"] == _sha256(output / "tracker.onnx")

  checksums = {}
  for line in (output / "SHA256SUMS").read_text().splitlines():
    digest, name = line.split("  ", 1)
    checksums[name] = digest
  assert checksums == {
    "denoiser.onnx": _sha256(output / "denoiser.onnx"),
    "reference_stats.json": _sha256(output / "reference_stats.json"),
    "tracker.onnx": _sha256(output / "tracker.onnx"),
  }

  stats = json.loads((output / "reference_stats.json").read_text())
  # The controller parses this asset with a strict key set: exactly mean and std.
  assert set(stats) == {"mean", "std"}
  projection = np.load(inputs["data_dir"] / "projection.npz")  # type: ignore[operator]
  np.testing.assert_array_equal(
    np.asarray(stats["mean"], dtype=np.float32), projection["mean"]
  )
  np.testing.assert_array_equal(
    np.asarray(stats["std"], dtype=np.float32), projection["std"]
  )

  report = json.loads((output / "export-report.json").read_text())
  assert report["reference_stats_json"]["format"] == ("mjlab-x2-joint-pose-stats-v1")
  assert report["reference_stats_json"]["channels"] == 68
  assert report["reference_stats_json"]["source"]["projection_npz_sha256"] == (
    _sha256(inputs["data_dir"] / "projection.npz")  # type: ignore[operator]
  )
  assert report["checkpoint_sha256"] == _sha256(inputs["checkpoint"])  # type: ignore[arg-type]
  assert report["parameters"] == "ema"
  assert report["ema_available"] is True
  assert report["final_norm"] is False
  assert report["final_norm_keys"] == []
  assert report["onnx_bytes"] == (output / "denoiser.onnx").stat().st_size
  assert report["verification"]["onnx_parity_verified"] is True
  assert report["verification"]["parity_max_abs_difference"] <= 1.0e-4

  model = onnx.load(str(output / "denoiser.onnx"))
  assert [item.name for item in model.graph.input] == [
    "noisy_tokens",
    "step_ids",
  ]
  assert [item.name for item in model.graph.output] == ["x0_hat"]
  assert model.opset_import[0].version == 18

  import onnxruntime as ort

  session = ort.InferenceSession(
    str(output / "denoiser.onnx"), providers=["CPUExecutionProvider"]
  )
  values = session.run(
    ["x0_hat"],
    {
      "noisy_tokens": np.zeros((2, 41, 68), dtype=np.float32),
      "step_ids": np.zeros((2, 41, 2), dtype=np.int64),
    },
  )
  assert np.asarray(values[0]).shape == (2, 41, 68)


def test_export_selects_ema_by_default_and_raw_model_on_request(
  export_tool: ModuleType, tmp_path: Path
) -> None:
  inputs = _inputs(tmp_path)
  ema_output = tmp_path / "ema"
  model_output = tmp_path / "raw"
  assert export_tool.main(_cli(inputs, ema_output)) == 0
  assert export_tool.main(_cli(inputs, model_output, "--parameters", "model")) == 0
  ema_descriptor = json.loads((ema_output / "descriptor.json").read_text())
  model_descriptor = json.loads((model_output / "descriptor.json").read_text())
  assert ema_descriptor["task"]["denoiser_identity"].endswith("@ema")
  assert model_descriptor["task"]["denoiser_identity"].endswith("@model")
  ema_report = json.loads((ema_output / "export-report.json").read_text())
  model_report = json.loads((model_output / "export-report.json").read_text())
  assert ema_report["parameters"] == "ema"
  assert model_report["parameters"] == "model"
  assert ema_report["parameters_sha256"] != model_report["parameters_sha256"]


def test_export_supports_final_norm_checkpoint(
  export_tool: ModuleType, tmp_path: Path
) -> None:
  inputs = _inputs(tmp_path, final_norm=True)
  output = tmp_path / "final_norm"
  assert export_tool.main(_cli(inputs, output)) == 0
  descriptor = json.loads((output / "descriptor.json").read_text())
  report = json.loads((output / "export-report.json").read_text())
  assert descriptor["task"]["denoiser_identity"] == "model_0007000@ema@finalnorm"
  assert report["final_norm"] is True
  assert report["final_norm_based_on"] == "ema.shadow"
  assert report["final_norm_keys"] == [
    "final_norm.bias",
    "final_norm.weight",
  ]
  assert report["verification"]["onnx_parity_verified"] is True


def test_refuses_wrong_token_dimension(
  export_tool: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  inputs = _inputs(tmp_path, token_dimension=67)
  output = tmp_path / "bundle"
  assert export_tool.main(_cli(inputs, output)) != 0
  assert "token_dimension" in capsys.readouterr().err
  assert not (output / "descriptor.json").exists()


def test_refuses_wrong_sequence_length(
  export_tool: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  inputs = _inputs(tmp_path, sequence_length=40)
  output = tmp_path / "bundle"
  assert export_tool.main(_cli(inputs, output)) != 0
  assert "sequence_length" in capsys.readouterr().err
  assert not (output / "descriptor.json").exists()


def test_refuses_wrong_projection_length(
  export_tool: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  inputs = _inputs(tmp_path, projection_length=67)
  output = tmp_path / "bundle"
  assert export_tool.main(_cli(inputs, output)) != 0
  assert "expected 68/68" in capsys.readouterr().err
  assert not (output / "descriptor.json").exists()


def test_refuses_wrong_tracker_lookahead(
  export_tool: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  inputs = _inputs(tmp_path, tracker_lookahead="9")
  output = tmp_path / "bundle"
  assert export_tool.main(_cli(inputs, output)) != 0
  assert "motion_lookahead_steps" in capsys.readouterr().err
  assert not (output / "descriptor.json").exists()


def test_refuses_missing_ema_when_requested(
  export_tool: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  inputs = _inputs(tmp_path, with_ema=False)
  output = tmp_path / "bundle"
  assert export_tool.main(_cli(inputs, output)) != 0
  assert "no 'ema' state" in capsys.readouterr().err
  assert not (output / "descriptor.json").exists()


def test_missing_onnxruntime_still_exports(
  export_tool: ModuleType,
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
) -> None:
  inputs = _inputs(tmp_path)
  output = tmp_path / "bundle"
  monkeypatch.setattr(export_tool, "_ort", None)
  monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
  assert export_tool.main(_cli(inputs, output)) == 0
  report = json.loads((output / "export-report.json").read_text())
  assert report["verification"]["onnxruntime_available"] is False
  assert report["verification"]["onnx_parity_verified"] is False
  assert "NOT verified" in capsys.readouterr().err


def test_refuses_existing_descriptor(
  export_tool: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  inputs = _inputs(tmp_path)
  output = tmp_path / "bundle"
  output.mkdir()
  existing = output / "descriptor.json"
  existing.write_text('{"sentinel": true}\n')
  assert export_tool.main(_cli(inputs, output)) != 0
  assert "already holds" in capsys.readouterr().err
  assert existing.read_text() == '{"sentinel": true}\n'
