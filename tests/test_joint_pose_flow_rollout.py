"""CPU tests for the joint-pose flow window codec and the chunked rollout.

The checkpoint fixture is a tiny ``width=32, layers=1, heads=2`` denoiser and
the stub sampler returns recorded windows, so chunk accounting and re-anchoring
are checked without a trained model.  No test opens a dataset directory.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from mjlab.scripts.joint_pose_rollout import (
  MOTION_KEYS,
  SUBSTITUTION_STATEMENT,
  write_pair,
)
from mjlab.tasks.tracking.joint_pose.config import JointPoseDatasetCfg
from mjlab.tasks.tracking.joint_pose.evaluate import ChannelNormalization
from mjlab.tasks.tracking.joint_pose.model import build_denoiser
from mjlab.tasks.tracking.joint_pose.rollout import (
  KEYFRAME_CURRENT,
  KEYFRAME_LAST,
  ChunkRecord,
  ClipArrays,
  RolloutError,
  RolloutResult,
  encode_window,
  joint_pose_flow,
  matrix_from_quat,
  parse_keyframes,
  quat_from_6d,
  quat_from_yaw,
  quat_mul,
  rollout,
  rotate_rows,
  yaw_of,
)
from mjlab.tasks.tracking.joint_pose.runner import CHECKPOINT_FORMAT

_PREPARE_SCRIPT = (
  Path(__file__).resolve().parents[1]
  / "scripts/experiments/joint_space_prepare/prepare.py"
)
_REAL = {"past_steps": 8, "window_steps": 41, "channels": 68}
_TOY = {"past_steps": 4, "window_steps": 20, "channels": 68}
_SMALL_MODEL = {"width": 32, "layers": 1, "heads": 2}
_MAX_LEVEL = 1000
_JOINTS = 31
_FPS = 50.0
_M3_KEYS = (
  "checkpoint global_step parameters schema stride chunk_count projection clip seed "
  "keyframes "
  "start_frame target_frames rows steps stop_reason chunks substitution"
).split()


@pytest.fixture(scope="module")
def prepare_tool() -> ModuleType:
  """Load ``prepare.py`` the way the bundle tests load their script."""
  spec = importlib.util.spec_from_file_location("joint_space_prepare", _PREPARE_SCRIPT)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def _synthetic_clip(rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  rng = np.random.default_rng(0)
  quat = quat_from_yaw(-1.2 + 0.03 * np.arange(rows, dtype=np.float64))
  for axis, angle in ((1, -0.18), (0, 0.25)):
    tilt = np.zeros((rows, 4))
    tilt[:, (0, axis + 1)] = (np.cos(angle / 2), np.sin(angle / 2))
    quat = quat_mul(quat, tilt)
  pos = (0.4 * rng.standard_normal((rows, _JOINTS))).astype(np.float32)
  vel = (0.5 * rng.standard_normal((rows, _JOINTS))).astype(np.float32)
  return quat, pos, vel


def _identity(channels: int) -> ChannelNormalization:
  return ChannelNormalization(mean=np.zeros(channels), std=np.ones(channels))


def _chunk_count(window_steps: int, past_steps: int, frames: int) -> int:
  stride = window_steps - past_steps - 1
  return (max(0, frames - window_steps) + stride - 1) // stride + 1


def _encode_windows(
  clip: ClipArrays, first: int, count: int, *, past_steps: int, window_steps: int
) -> np.ndarray:
  stride = window_steps - past_steps - 1
  return np.stack(
    [
      encode_window(
        clip.root_quat_wxyz,
        clip.joint_pos,
        clip.joint_vel,
        first + stride * index,
        past_steps=past_steps,
        window_steps=window_steps,
      )
      for index in range(count)
    ]
  )


class _StubSampler:
  """Return recorded windows and check their history, or zero the future."""

  def __init__(
    self,
    *,
    past_steps: int,
    window_steps: int,
    channels: int,
    windows: np.ndarray | None = None,
  ) -> None:
    self.past_steps = past_steps
    self.window_steps = window_steps
    self.channels = channels
    self.windows = windows
    self.calls = 0

  def sample(
    self,
    history: torch.Tensor,
    *,
    steps: int,
    generator: torch.Generator,
    pinned: dict[int, torch.Tensor] | None = None,
  ) -> torch.Tensor:
    self.pinned_rows = () if pinned is None else tuple(sorted(pinned))
    self.calls += 1
    if self.windows is None:
      future = torch.zeros(
        (self.window_steps - self.past_steps, self.channels), dtype=history.dtype
      )
      window = torch.cat((history, future), dim=0)
      for row, values in (pinned or {}).items():
        window[row] = values
      return window
    assert self.calls <= self.windows.shape[0], "more chunks than recorded windows"
    window = self.windows[self.calls - 1]
    np.testing.assert_allclose(
      history.numpy(), window[: self.past_steps], atol=1.0e-4, rtol=0.0
    )
    for row, values in (pinned or {}).items():
      np.testing.assert_allclose(window[row], values.numpy(), atol=1.0e-4, rtol=0.0)
    return torch.from_numpy(window.copy())


def _payload(
  *,
  data_dir: str,
  past_steps: int = _REAL["past_steps"],
  window_steps: int = _REAL["window_steps"],
) -> dict[str, object]:
  dataset = JointPoseDatasetCfg(
    data_dir=data_dir, past_steps=past_steps, window_steps=window_steps
  )
  model = build_denoiser(dataset, max_level=_MAX_LEVEL, init_seed=123, **_SMALL_MODEL)
  state = {name: value.detach().clone() for name, value in model.state_dict().items()}
  # A shadow that differs from the live weights, so a test can tell which set
  # the sampler loaded.
  shadow = {name: value * 2.0 for name, value in state.items()}
  return {
    "format": CHECKPOINT_FORMAT,
    "global_step": 7000,
    "model": state,
    "trainer": {"max_level": _MAX_LEVEL},
    "dataset_cfg": dataset.as_dict(),
    "model_settings": asdict(model.settings),
    "ema": {"power": 0.75, "max_decay": 0.9999, "updates": 7000, "shadow": shadow},
  }


def test_encode_window_matches_prepare_segment_windows(
  prepare_tool: ModuleType,
) -> None:
  quat, pos, vel = _synthetic_clip(150)
  past = _REAL["past_steps"]
  window = _REAL["window_steps"]
  assert (prepare_tool.PAST, prepare_tool.WINDOW) == (past, window)
  assert (prepare_tool.CHANNELS, prepare_tool.JOINTS) == (_REAL["channels"], _JOINTS)
  starts = np.arange(0, 150 - window + 1, 7, dtype=np.int64)
  recorded = prepare_tool.segment_windows(quat, pos, vel, starts)
  for index, start in enumerate(starts):
    encoded = encode_window(
      quat, pos, vel, int(start), past_steps=past, window_steps=window
    )
    difference = np.max(np.abs(encoded - recorded[index]))
    assert encoded.shape == (window, _REAL["channels"])
    assert difference <= 1.0e-6, f"start {start}: {difference:.3g}"


def test_centre_row_has_zero_yaw_and_keeps_the_clip_tilt() -> None:
  quat, pos, vel = _synthetic_clip(120)
  past = _REAL["past_steps"]
  start = 16
  window = encode_window(
    quat, pos, vel, start, past_steps=past, window_steps=_REAL["window_steps"]
  )
  centre = quat_from_6d(window[past, :6])
  assert abs(float(yaw_of(centre))) <= 1.0e-6
  heading = quat_from_yaw(-yaw_of(quat[start + past]))
  expected = matrix_from_quat(heading)[None] @ matrix_from_quat(quat[start + past])
  assert np.max(np.abs(matrix_from_quat(centre) - expected)) <= 1.0e-6
  # The heading was removed, so the residual is the clip's nonzero tilt.
  assert np.max(np.abs(expected[..., 2, 0])) > 1.0e-2


def test_rotate_rows_round_trip_is_identity() -> None:
  quat, pos, vel = _synthetic_clip(90)
  rows = encode_window(quat, pos, vel, 3, past_steps=8, window_steps=41)
  first = quat_from_yaw(0.37)
  second = quat_mul(quat_from_yaw(-1.1), np.array([np.cos(0.1), 0.0, np.sin(0.1), 0.0]))
  back = quat_from_yaw(-(yaw_of(first) + yaw_of(second)))
  turned = rotate_rows(rotate_rows(rows, first), second)
  restored = rotate_rows(turned, back)
  assert np.max(np.abs(restored - rows)) <= 1.0e-6
  assert np.array_equal(turned[:, 6:], rows[:, 6:])


def test_toy_schema_encodes_and_rolls(tmp_path: Path) -> None:
  past, window = _TOY["past_steps"], _TOY["window_steps"]
  quat, pos, vel = _synthetic_clip(150)
  encoded = encode_window(quat, pos, vel, 0, past_steps=past, window_steps=window)
  assert encoded.shape == (window, _TOY["channels"])
  checkpoint = tmp_path / "toy.pt"
  torch.save(
    _payload(data_dir=str(tmp_path / "absent"), past_steps=past, window_steps=window),
    checkpoint,
  )
  sampler = joint_pose_flow(checkpoint)
  assert (sampler.past_steps, sampler.window_steps) == (past, window)
  clip = ClipArrays(root_quat_wxyz=quat, joint_pos=pos, joint_vel=vel, fps=_FPS)
  result = rollout(
    clip=clip,
    sampler=sampler,
    normalization=_identity(_TOY["channels"]),
    start_frame=0,
    target_frames=100,
    steps=1,
    generator=torch.Generator().manual_seed(0),
  )
  assert len(result.chunks) == _chunk_count(window, past, 100) == 7
  assert len(result.joint_pos) == 100


@pytest.mark.parametrize(
  ("geometry", "target_frames"),
  [pytest.param(_REAL, 200, id="real"), pytest.param(_TOY, 100, id="toy")],
)
def test_stub_reconstruction_replays_the_clip(
  geometry: dict[str, int], target_frames: int
) -> None:
  past = geometry["past_steps"]
  window = geometry["window_steps"]
  channels = geometry["channels"]
  start = 3
  quat, pos, vel = _synthetic_clip(start + target_frames + window + 4)
  clip = ClipArrays(root_quat_wxyz=quat, joint_pos=pos, joint_vel=vel, fps=_FPS)
  count = _chunk_count(window, past, target_frames)
  sampler = _StubSampler(
    past_steps=past,
    window_steps=window,
    channels=channels,
    windows=_encode_windows(clip, start, count, past_steps=past, window_steps=window),
  )
  result = rollout(
    clip=clip,
    sampler=sampler,
    normalization=_identity(channels),
    start_frame=start,
    target_frames=target_frames,
    steps=1,
    generator=torch.Generator().manual_seed(0),
  )
  stride = window - past - 1
  assert sampler.calls == count
  assert len(result.chunks) == count
  assert [chunk.index for chunk in result.chunks] == list(range(count))
  assert [chunk.start_frame for chunk in result.chunks] == [
    start + stride * index for index in range(count)
  ]
  expected = clip.joint_pos[start : start + target_frames]
  assert np.max(np.abs(result.joint_pos - expected)) <= 1.0e-5
  velocity = np.abs(result.joint_vel - clip.joint_vel[start : start + target_frames])
  assert np.max(velocity) <= 1.0e-5
  assert result.chunks[0].overlap_rms_rad is None
  for chunk in result.chunks[1:]:
    assert chunk.overlap_rms_rad <= 1.0e-6
  for chunk in result.chunks:
    assert chunk.joint_drift_rms_rad <= 1.0e-5


def test_parse_keyframes_resolves_names() -> None:
  assert parse_keyframes("none") == ()
  assert parse_keyframes("") == ()
  assert parse_keyframes("current") == (KEYFRAME_CURRENT,)
  assert parse_keyframes("last") == (KEYFRAME_LAST,)
  assert parse_keyframes("last,current") == (KEYFRAME_CURRENT, KEYFRAME_LAST)
  assert parse_keyframes(" CURRENT , last ") == (KEYFRAME_CURRENT, KEYFRAME_LAST)
  with pytest.raises(RolloutError, match="unknown keyframes"):
    parse_keyframes("middle")
  with pytest.raises(RolloutError, match="repeats a name"):
    parse_keyframes("last,last")


@pytest.mark.parametrize(
  ("geometry", "target_frames"),
  [pytest.param(_REAL, 200, id="real"), pytest.param(_TOY, 100, id="toy")],
)
def test_keyframes_pin_the_reference_rows(
  geometry: dict[str, int], target_frames: int
) -> None:
  past = geometry["past_steps"]
  window = geometry["window_steps"]
  channels = geometry["channels"]
  start = 3
  quat, pos, vel = _synthetic_clip(start + target_frames + window + 4)
  clip = ClipArrays(root_quat_wxyz=quat, joint_pos=pos, joint_vel=vel, fps=_FPS)
  count = _chunk_count(window, past, target_frames)
  sampler = _StubSampler(
    past_steps=past,
    window_steps=window,
    channels=channels,
    windows=_encode_windows(clip, start, count, past_steps=past, window_steps=window),
  )
  result = rollout(
    clip=clip,
    sampler=sampler,
    normalization=_identity(channels),
    start_frame=start,
    target_frames=target_frames,
    steps=1,
    generator=torch.Generator().manual_seed(0),
    keyframes="current,last",
  )
  assert sampler.pinned_rows == (past, window - 1)
  expected = clip.joint_pos[start : start + target_frames]
  assert np.max(np.abs(result.joint_pos - expected)) <= 1.0e-5
  stride = window - past - 1
  for chunk in result.chunks:
    assert past in chunk.pinned
    assert set(chunk.pinned) <= {past, window - 1}
  # The first chunk has no previous value to snap away from.
  assert result.chunks[0].seam_rms_rad is None
  for chunk in result.chunks[1:]:
    assert chunk.seam_rms_rad is not None and chunk.seam_rms_rad <= 1.0e-5
  # The final chunk can run past the clip end, and then only the current row exists.
  last = result.chunks[-1]
  frame = start + stride * last.index + window - 1
  if frame >= clip.rows:
    assert last.pinned == (past,)
  else:
    assert last.pinned == (past, window - 1)


def test_keyframes_default_pins_nothing() -> None:
  geometry = _REAL
  past, window = geometry["past_steps"], geometry["window_steps"]
  quat, pos, vel = _synthetic_clip(120)
  clip = ClipArrays(root_quat_wxyz=quat, joint_pos=pos, joint_vel=vel, fps=_FPS)
  sampler = _StubSampler(
    past_steps=past,
    window_steps=window,
    channels=geometry["channels"],
    windows=_encode_windows(
      clip, 0, _chunk_count(window, past, 60), past_steps=past, window_steps=window
    ),
  )
  result = rollout(
    clip=clip,
    sampler=sampler,
    normalization=_identity(geometry["channels"]),
    start_frame=0,
    target_frames=60,
    steps=1,
    generator=torch.Generator().manual_seed(0),
  )
  assert sampler.pinned_rows == ()
  assert all(chunk.pinned == () for chunk in result.chunks)
  assert all(chunk.seam_rms_rad is None for chunk in result.chunks)


def test_sampler_honours_pinned_rows(tmp_path: Path) -> None:
  checkpoint = tmp_path / "model_0007000.pt"
  torch.save(_payload(data_dir=str(tmp_path / "absent")), checkpoint)
  sampler = joint_pose_flow(checkpoint)
  history = torch.from_numpy(
    np.linspace(
      -1.0, 1.0, sampler.past_steps * sampler.channels, dtype=np.float32
    ).reshape(sampler.past_steps, sampler.channels)
  )
  row = sampler.window_steps - 1
  pinned_value = torch.linspace(-2.0, 2.0, sampler.channels)
  seeded = torch.Generator().manual_seed(5)
  plain = sampler.sample(history, steps=2, generator=seeded.manual_seed(5))
  sampled = sampler.sample(
    history, steps=2, generator=seeded.manual_seed(5), pinned={row: pinned_value}
  )
  torch.testing.assert_close(sampled[row], pinned_value, atol=1.0e-6, rtol=0.0)
  assert not torch.equal(plain[row], sampled[row])
  torch.testing.assert_close(
    sampled[: sampler.past_steps], history, atol=1.0e-6, rtol=0.0
  )


def test_chunk_count_and_file_rows_follow_the_formula() -> None:
  quat, pos, vel = _synthetic_clip(520)
  clip = ClipArrays(root_quat_wxyz=quat, joint_pos=pos, joint_vel=vel, fps=_FPS)
  for geometry, frames, expected in [(_REAL, 453, 14), (_REAL, 41, 1), (_TOY, 100, 7)]:
    past, window = geometry["past_steps"], geometry["window_steps"]
    sampler = _StubSampler(
      past_steps=past, window_steps=window, channels=geometry["channels"]
    )
    result = rollout(
      clip=clip,
      sampler=sampler,
      normalization=_identity(geometry["channels"]),
      start_frame=0,
      target_frames=frames,
      steps=1,
      generator=torch.Generator().manual_seed(0),
    )
    coverage = (expected - 1) * (window - past - 1) + window
    assert len(result.chunks) == sampler.calls == expected
    assert len(result.joint_pos) == min(frames, coverage)


def test_joint_pose_flow_refuses_bad_payloads(tmp_path: Path) -> None:
  cases = [
    ("format", "format", "other", "not a joint_pose_flow_checkpoint_v1 checkpoint"),
    ("settings", "model_settings", None, "no 'model_settings' mapping"),
    ("tokens", "model_settings.token_dimension", 67, "model settings describe"),
    ("rows", "model_settings.sequence_length", 40, "model settings describe"),
  ]
  base = _payload(data_dir=str(tmp_path / "absent"))
  for name, dotted, value, message in cases:
    payload = copy.deepcopy(base)
    target = payload
    for part in dotted.split(".")[:-1]:
      target = target[part]
    if value is None:
      target.pop(dotted.split(".")[-1])
    else:
      target[dotted.split(".")[-1]] = value
    checkpoint = tmp_path / f"{name}.pt"
    torch.save(payload, checkpoint)
    with pytest.raises(RolloutError, match=message):
      joint_pose_flow(checkpoint)


def test_explicit_ema_without_a_shadow_is_refused(tmp_path: Path) -> None:
  payload = _payload(data_dir=str(tmp_path / "absent"))
  payload.pop("ema")
  checkpoint = tmp_path / "no-ema.pt"
  torch.save(payload, checkpoint)
  with pytest.raises(RolloutError, match="no EMA shadow"):
    joint_pose_flow(checkpoint, parameters="ema")


def test_auto_falls_back_to_the_live_weights(tmp_path: Path) -> None:
  payload = _payload(data_dir=str(tmp_path / "absent"))
  payload.pop("ema")
  checkpoint = tmp_path / "no-ema.pt"
  torch.save(payload, checkpoint)
  sampler = joint_pose_flow(checkpoint)
  assert sampler.parameters == "model"


def test_max_level_falls_back_to_the_embedding_count(tmp_path: Path) -> None:
  payload = _payload(data_dir=str(tmp_path / "absent"))
  payload["trainer"].pop("max_level")
  checkpoint = tmp_path / "no-max-level.pt"
  torch.save(payload, checkpoint)
  sampler = joint_pose_flow(checkpoint)
  assert sampler.max_level == _MAX_LEVEL


def test_channel_layout_mismatch_is_refused(tmp_path: Path) -> None:
  payload = _payload(data_dir=str(tmp_path / "absent"))
  layout = [list(item) for item in payload["dataset_cfg"]["channel_layout"]]
  layout[0] = [layout[0][0], 0, 3]
  payload["dataset_cfg"]["channel_layout"] = layout
  checkpoint = tmp_path / "bad-layout.pt"
  torch.save(payload, checkpoint)
  with pytest.raises(RolloutError, match="channel_layout"):
    joint_pose_flow(checkpoint)


def test_ema_selection_uses_the_shadow_without_mutating_the_payload(
  tmp_path: Path,
) -> None:
  payload = _payload(data_dir=str(tmp_path / "absent"))
  checkpoint = tmp_path / "model_0007000.pt"
  torch.save(payload, checkpoint)
  ema = joint_pose_flow(checkpoint, parameters="ema")
  live = joint_pose_flow(checkpoint, parameters="model")
  history = torch.from_numpy(
    np.linspace(-1.0, 1.0, ema.past_steps * ema.channels, dtype=np.float32).reshape(
      ema.past_steps, ema.channels
    )
  )
  selected = ema.sample(history, steps=2, generator=torch.Generator().manual_seed(11))
  plain = live.sample(history, steps=2, generator=torch.Generator().manual_seed(11))
  assert not torch.equal(selected, plain)
  reloaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
  assert all(
    torch.equal(value, payload["model"][name])
    for name, value in reloaded["model"].items()
  )
  data_dir = tmp_path / "absent"
  checkpoint = tmp_path / "model_0007000.pt"
  torch.save(_payload(data_dir=str(data_dir)), checkpoint)
  sampler = joint_pose_flow(checkpoint)
  assert not data_dir.exists()
  expected = {
    **_REAL,
    **_SMALL_MODEL,
    "data_dir": str(data_dir),
    "final_norm": False,
    "max_level": _MAX_LEVEL,
  }
  assert sampler.scheme == expected
  assert (sampler.global_step, sampler.parameters) == (7000, "ema")


def test_sampler_is_bit_identical_for_one_seed(tmp_path: Path) -> None:
  checkpoint = tmp_path / "model_0007000.pt"
  torch.save(_payload(data_dir=str(tmp_path / "absent")), checkpoint)
  first = joint_pose_flow(checkpoint)
  second = joint_pose_flow(checkpoint)
  past, channels = first.past_steps, first.channels
  history = torch.from_numpy(
    np.linspace(-1.0, 1.0, past * channels, dtype=np.float32).reshape(past, channels)
  )
  left = first.sample(history, steps=2, generator=torch.Generator().manual_seed(11))
  right = second.sample(history, steps=2, generator=torch.Generator().manual_seed(11))
  other = first.sample(history, steps=2, generator=torch.Generator().manual_seed(12))
  assert torch.equal(left, right)
  assert not torch.equal(left, other)


@pytest.fixture()
def written_pair(tmp_path: Path) -> dict[str, object]:
  """Write both motion files from a hand-built rollout result."""
  rows, start, target = 70, 4, 64
  quat, pos, vel = _synthetic_clip(rows)
  span = slice(start, start + target)
  generated_pos = pos[span].copy()
  generated_pos[8:] = pos[span][8:] + np.float32(0.01)
  rng = np.random.default_rng(11)
  clip = {
    "joint_pos": pos,
    "joint_vel": vel,
    "body_quat_w": np.repeat(quat.astype(np.float32)[:, None, :], 3, axis=1),
    "fps": np.array([_FPS], dtype=np.float32),
  }
  for name in ("body_pos_w", "body_lin_vel_w", "body_ang_vel_w"):
    clip[name] = rng.standard_normal((rows, 3, 3)).astype(np.float32)
  result = RolloutResult(
    root_quat_wxyz=quat[span],
    joint_pos=generated_pos,
    joint_vel=vel[span],
    chunks=(ChunkRecord(0, 0, 0.0, None, 0.0, 0.0),),
    stop_reason="target",
    steps=1,
  )
  record = {
    "checkpoint": "model_0007000.pt",
    "global_step": 7000,
    "parameters": "ema",
    "schema": {"past_steps": 8, "window_steps": 41, "channels": 68},
    "projection": str(tmp_path / "projection.npz"),
    "clip": "clip.npz",
    "seed": 0,
    "keyframes": "none",
  }
  out_dir = tmp_path / "out"
  write_pair(
    out_dir=out_dir,
    clip=clip,
    result=result,
    start_frame=start,
    target_frames=target,
    record=record,
  )
  return {
    "clip": clip,
    "result": result,
    "out_dir": out_dir,
    "start_frame": start,
    "target_frames": target,
  }


def test_write_pair_substitutes_body_channels(written_pair: dict[str, object]) -> None:
  clip, result = written_pair["clip"], written_pair["result"]
  out_dir = written_pair["out_dir"]
  start, target = written_pair["start_frame"], written_pair["target_frames"]
  assert len(MOTION_KEYS) == 7
  with (
    np.load(out_dir / "generated.npz") as generated,
    np.load(out_dir / "control.npz") as control,
  ):
    assert set(generated.files) == set(control.files) == set(MOTION_KEYS)
    for key in MOTION_KEYS:
      assert generated[key].dtype == control[key].dtype == np.float32
      if key == "fps":
        assert generated[key].shape == control[key].shape == (1,)
        continue
      assert generated[key].shape[0] == target
      recorded = np.asarray(clip[key])[start : start + target].astype(np.float32)
      np.testing.assert_array_equal(control[key], recorded)
      if key.startswith("body_"):
        np.testing.assert_array_equal(generated[key], control[key])
    np.testing.assert_array_equal(generated["joint_pos"], result.joint_pos)
    np.testing.assert_array_equal(generated["joint_vel"], result.joint_vel)
    np.testing.assert_array_equal(
      generated["joint_pos"][:8], np.asarray(clip["joint_pos"])[start : start + 8]
    )


def test_rollout_json_records_every_promised_field(
  written_pair: dict[str, object],
) -> None:
  result = written_pair["result"]
  record = json.loads((written_pair["out_dir"] / "rollout.json").read_text())
  assert set(record) >= set(_M3_KEYS)
  assert record["rows"] == record["target_frames"] == written_pair["target_frames"]
  assert record["stride"] == 32
  assert record["chunk_count"] == len(result.chunks)
  assert len(record["chunks"]) == len(result.chunks)
  assert {"past_steps", "window_steps", "channels"} <= set(record["schema"])
  assert record["substitution"] == SUBSTITUTION_STATEMENT
  diagnostics = {"anchor_yaw_deg", "overlap_rms_rad", "joint_drift_rms_rad"}
  assert all(diagnostics | {"start_frame"} <= set(chunk) for chunk in record["chunks"])
  keys = set(record) | set(record["schema"])
  keys |= {key for chunk in record["chunks"] for key in chunk}
  assert [key for key in keys if "sha" in key or "digest" in key] == []
