"""CPU coverage for the preview-aware collection reference path.

The frozen gravity layout has two supported reference widths: 68 (version 1,
no preview) and 130 (version 2, one 62-wide q/dq preview sample).  This lane
checks that the collection boundary derives the live width from the cohort,
that it never accepts an arbitrary width, and that the packed preview stays on
the row's own clip after the frame advances, clamps at the clip end, and after
a subset reset.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
import yaml
from tracking_diffusion_fixtures import make_state

from mjlab.tasks.tracking.diffusion.adapter import (
  ActionContract,
  AdapterError,
  FrozenVaePolicy,
  VaeObservation,
  VaeObservationBatch,
  VaeObservationTensorBatch,
)
from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT
from mjlab.tasks.tracking.diffusion.runtime_x2 import (
  RuntimeX2Error,
  X2MotionEnvironment,
  X2MultiMotionEnvironment,
  resolve_rollout_ids,
)
from mjlab.tasks.tracking.diffusion.state import WorldState
from mjlab.tasks.tracking.distillation import adapter as distillation_adapter
from mjlab.tasks.tracking.distillation import environment as distillation_environment
from mjlab.tasks.tracking.distillation.adapter import default_schema_for_cohort
from mjlab.tasks.tracking.distillation.observations import (
  ObservationSnapshot,
  pack_observations,
)
from mjlab.tasks.tracking.distillation.vae_config import (
  JOINT_DIM,
  PREVIEW_DIM,
  PREVIEW_SCHEMA_VERSION,
  REFERENCE_DIM,
  REFERENCE_DIM_WITH_PREVIEW,
  VaeSchema,
  make_schema,
)
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionLoader

_PREVIEW_SECONDS = 0.2
_PREVIEW_STEPS = 10
_MULTI_MOTION_MANIFEST = "configs/distillation/x2_tennis_multimotion_v1.yaml"


def _cohort(*, lookahead_s: float) -> Any:
  """The two cohort fields the live schema derivation reads."""
  return SimpleNamespace(
    lookahead_s=lookahead_s,
    actions=SimpleNamespace(
      joint_names=tuple(f"joint_{index:02d}" for index in range(JOINT_DIM)),
      dim=JOINT_DIM,
    ),
  )


def _contract() -> ActionContract:
  return ActionContract(np.ones(31), np.zeros(31), np.ones(31), np.ones(31))


class _RecordingVae(torch.nn.Module):
  """Fake frozen student that records the reference width it receives."""

  def __init__(self, schema: VaeSchema) -> None:
    super().__init__()
    self.schema = schema
    self.bias = torch.nn.Parameter(torch.zeros(()))
    self.reference_widths: list[int] = []

  def encode(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    self.reference_widths.append(int(reference.shape[1]))
    mu = reference[:, :1].repeat(1, 32) + self.bias
    return mu, torch.zeros_like(mu)

  def decode(self, latent: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
    return latent[:, :1].repeat(1, 31) + conditioning[:, :1]


@pytest.mark.parametrize("lookahead_s, width", [(0.0, 68), (0.2, 130)])
def test_collection_constructors_use_the_cohort_schema(
  monkeypatch: pytest.MonkeyPatch, lookahead_s: float, width: int
) -> None:
  cohort = _cohort(lookahead_s=lookahead_s)
  cohort.observations = SimpleNamespace(terms=())
  command = SimpleNamespace(
    motion=SimpleNamespace(time_step_total=11),
    cfg=SimpleNamespace(body_names=DEFAULT_CONTRACT.body_names),
  )
  env = SimpleNamespace(
    cfg=SimpleNamespace(auto_reset=True),
    num_envs=1,
    device="cpu",
    step_dt=0.02,
    command_manager=SimpleNamespace(get_term=lambda _name: command),
    scene={
      "robot": SimpleNamespace(
        body_names=DEFAULT_CONTRACT.body_names,
        joint_names=cohort.actions.joint_names,
      )
    },
    termination_manager=SimpleNamespace(
      active_terms=["anchor_pos", "anchor_ori", "ee_body_pos"],
      _term_cfgs=[SimpleNamespace(time_out=False, func=None, params={})] * 3,
    ),
  )
  adapter = SimpleNamespace(schema=default_schema_for_cohort(cohort), env=env)
  monkeypatch.setattr(
    distillation_adapter, "make_distillation_adapter", lambda *a, **kw: adapter
  )
  monkeypatch.setattr(
    distillation_environment, "build_multi_motion_environment", lambda *a, **kw: env
  )
  # Only the simulator and unchanged termination machinery are mocked. Both
  # real constructors must wire their schema; assigning it in this test would
  # conceal a return to the preview-blind constructor.
  monkeypatch.setattr(
    X2MotionEnvironment, "_set_reference_terminations", lambda *a: None
  )
  monkeypatch.setattr(
    X2MultiMotionEnvironment, "_install_termination_masks", lambda *a: None
  )
  monkeypatch.setattr(
    X2MultiMotionEnvironment, "_install_reference_advance_mask", lambda *a: None
  )
  single = X2MotionEnvironment(cohort, "single_000", device="cpu", seed=0)
  batch = X2MultiMotionEnvironment(
    cohort, ("single_000",), num_envs=1, device="cpu", seed=0
  )
  assert single._reference_width == batch._schema.reference_dim == width
  assert single._schema == batch._schema == adapter.schema
  if lookahead_s:
    assert batch._schema.version == PREVIEW_SCHEMA_VERSION
    assert batch._schema.reference_names[-1] == "motion_lookahead"
    assert batch._schema.reference_fields[-1].dimension == PREVIEW_DIM


def _packed_snapshot(reference_width: int, rows: int = 1) -> SimpleNamespace:
  reference = torch.arange(rows * reference_width, dtype=torch.float32).reshape(
    rows, reference_width
  )
  return SimpleNamespace(
    packed=SimpleNamespace(
      reference=reference,
      conditioning=torch.zeros(rows, 99),
    ),
    reference_frame=torch.arange(rows, dtype=torch.long),
  )


def test_scalar_vae_observation_derives_the_cohort_reference_width() -> None:
  runtime = X2MotionEnvironment.__new__(X2MotionEnvironment)
  runtime._schema = default_schema_for_cohort(_cohort(lookahead_s=_PREVIEW_SECONDS))
  runtime._reference_width = runtime._schema.reference_dim
  runtime.adapter = SimpleNamespace(schema=runtime._schema)
  runtime.teacher_id = "single_000"
  runtime.clip_frames = 11
  runtime._world_state = lambda: make_state()
  runtime._reference_metrics = lambda: {
    "anchor_z_error": 0.0,
    "gravity_z_error": 0.0,
    "end_effector_z_error": 0.0,
  }

  preview = runtime._vae_observation(_packed_snapshot(REFERENCE_DIM_WITH_PREVIEW))
  assert preview.reference.shape == (REFERENCE_DIM_WITH_PREVIEW,)
  np.testing.assert_array_equal(
    preview.reference, np.arange(REFERENCE_DIM_WITH_PREVIEW)
  )

  # A packed reference that disagrees with the cohort schema fails closed.
  with pytest.raises(RuntimeX2Error, match="expected 130"):
    runtime._vae_observation(_packed_snapshot(REFERENCE_DIM))


def test_frozen_vae_policy_accepts_preview_and_rejects_width_mismatch() -> None:
  preview_schema = default_schema_for_cohort(_cohort(lookahead_s=_PREVIEW_SECONDS))
  model = _RecordingVae(preview_schema)
  policy = FrozenVaePolicy(model, _contract())
  assert policy.reference_width == REFERENCE_DIM_WITH_PREVIEW

  latent, clean = policy.action(
    VaeObservation(
      make_state(),
      np.arange(REFERENCE_DIM_WITH_PREVIEW, dtype=np.float64),
      np.full(99, 0.25),
      "single_000",
      0,
      0.0,
    )
  )
  assert latent.shape == (32,) and clean.shape == (31,)
  assert model.reference_widths == [REFERENCE_DIM_WITH_PREVIEW]

  batch = VaeObservationBatch(
    make_state(),
    np.arange(REFERENCE_DIM_WITH_PREVIEW, dtype=np.float64)[None],
    np.full((1, 99), 0.25),
    ("single_000",),
    np.zeros(1, dtype=np.int64),
    np.zeros(1),
  )
  latent_batch, clean_batch = policy.batch_action(batch)
  assert latent_batch.shape == (1, 32) and clean_batch.shape == (1, 31)

  tensor_batch = VaeObservationTensorBatch(
    torch.zeros(1, REFERENCE_DIM_WITH_PREVIEW), torch.zeros(1, 99)
  )
  latent_tensor, clean_tensor = policy.batch_action_tensor(tensor_batch)
  assert latent_tensor.shape == (1, 32) and clean_tensor.shape == (1, 31)
  assert model.reference_widths == [REFERENCE_DIM_WITH_PREVIEW] * 3

  legacy = FrozenVaePolicy(_RecordingVae(make_schema()), _contract())
  with pytest.raises(AdapterError, match="disagrees with the frozen VAE schema"):
    legacy.action(
      VaeObservation(
        make_state(),
        np.zeros(REFERENCE_DIM_WITH_PREVIEW),
        np.zeros(99),
        "single_000",
        0,
        0.0,
      )
    )
  with pytest.raises(AdapterError, match="disagrees with the frozen VAE schema"):
    policy.action(
      VaeObservation(
        make_state(),
        np.zeros(REFERENCE_DIM),
        np.zeros(99),
        "single_000",
        0,
        0.0,
      )
    )

  legacy_batch = VaeObservationBatch(
    make_state(),
    np.zeros((1, REFERENCE_DIM_WITH_PREVIEW)),
    np.zeros((1, 99)),
    ("single_000",),
    np.zeros(1, dtype=np.int64),
    np.zeros(1),
  )
  with pytest.raises(AdapterError, match="disagrees with the frozen VAE schema"):
    legacy.batch_action(legacy_batch)
  with pytest.raises(AdapterError, match="disagrees with the frozen VAE schema"):
    legacy.batch_action_tensor(
      VaeObservationTensorBatch(
        torch.zeros(1, REFERENCE_DIM_WITH_PREVIEW), torch.zeros(1, 99)
      )
    )
  preview_batch = VaeObservationBatch(
    make_state(),
    np.zeros((1, REFERENCE_DIM)),
    np.zeros((1, 99)),
    ("single_000",),
    np.zeros(1, dtype=np.int64),
    np.zeros(1),
  )
  with pytest.raises(AdapterError, match="disagrees with the frozen VAE schema"):
    policy.batch_action(preview_batch)
  with pytest.raises(AdapterError, match="disagrees with the frozen VAE schema"):
    policy.batch_action_tensor(
      VaeObservationTensorBatch(torch.zeros(1, REFERENCE_DIM), torch.zeros(1, 99))
    )


def _batch_runtime(reference: torch.Tensor, conditioning: torch.Tensor) -> Any:
  rows = reference.shape[0]
  runtime = X2MultiMotionEnvironment.__new__(X2MultiMotionEnvironment)
  runtime.num_envs = rows
  runtime.device = "cpu"
  runtime.policy_device = torch.device("cpu")
  runtime._schema = default_schema_for_cohort(_cohort(lookahead_s=_PREVIEW_SECONDS))
  runtime._snapshot = lambda: SimpleNamespace(
    packed=SimpleNamespace(reference=reference, conditioning=conditioning)
  )
  runtime.command = SimpleNamespace(
    time_steps=torch.arange(rows, dtype=torch.long),
    row_lengths=torch.full((rows,), 11, dtype=torch.long),
    motion_ids=torch.zeros(rows, dtype=torch.long),
  )
  runtime.teacher_ids = ("single_000",)
  runtime._world_state = lambda ids: WorldState.stack(
    make_state(int(index)) for index in np.asarray(ids, dtype=np.int64).tolist()
  )
  runtime._reference_metrics = lambda ids: {
    "anchor_z_error": np.zeros(len(ids)),
    "gravity_z_error": np.zeros(len(ids)),
    "end_effector_z_error": np.zeros(len(ids)),
  }
  return runtime


def test_batched_runtime_accepts_a_cohort_preview_reference() -> None:
  reference = torch.arange(2 * REFERENCE_DIM_WITH_PREVIEW, dtype=torch.float32).reshape(
    2, REFERENCE_DIM_WITH_PREVIEW
  )
  conditioning = torch.full((2, 99), 0.25)
  runtime = _batch_runtime(reference, conditioning)
  assert runtime._schema.reference_dim == REFERENCE_DIM_WITH_PREVIEW

  observation = runtime.observe_vae_batch(np.asarray([1, 0]))
  assert observation.reference.shape == (2, REFERENCE_DIM_WITH_PREVIEW)
  np.testing.assert_allclose(observation.reference, reference[[1, 0]].numpy())

  policy = FrozenVaePolicy(
    _RecordingVae(default_schema_for_cohort(_cohort(lookahead_s=_PREVIEW_SECONDS))),
    _contract(),
  )
  latent, clean = policy.batch_action(observation)
  assert latent.shape == (2, 32) and clean.shape == (2, 31)

  tensor_observation = runtime.observe_vae_tensor_batch(np.asarray([1, 0]))
  assert tensor_observation.reference.shape == (2, REFERENCE_DIM_WITH_PREVIEW)
  latent_tensor, clean_tensor = policy.batch_action_tensor(tensor_observation)
  assert latent_tensor.shape == (2, 32) and clean_tensor.shape == (2, 31)


def _write_motion(
  path: Path, *, offset: float, frames: int = 15, fps: float = 50.0
) -> tuple[np.ndarray, np.ndarray]:
  """Write one synthetic clip whose frames are identifiable by sentinel values."""
  joint_pos = offset + np.arange(frames * JOINT_DIM, dtype=np.float32).reshape(
    frames, JOINT_DIM
  )
  joint_vel = (
    100.0
    + offset
    + np.arange(frames * JOINT_DIM, dtype=np.float32).reshape(frames, JOINT_DIM)
  )
  np.savez(
    path,
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    body_pos_w=np.zeros((frames, 1, 3), dtype=np.float32),
    body_quat_w=np.tile(
      np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (frames, 1, 1)
    ),
    body_lin_vel_w=np.zeros((frames, 1, 3), dtype=np.float32),
    body_ang_vel_w=np.zeros((frames, 1, 3), dtype=np.float32),
    fps=np.array([fps], dtype=np.float32),
  )
  return joint_pos, joint_vel


def _lookahead_command(path: Path, time_steps: list[int]) -> MotionCommand:
  """Build the real lookahead property without constructing a simulator."""
  command = object.__new__(MotionCommand)
  command_any = cast(Any, command)
  command_any._env = SimpleNamespace(
    device="cpu", num_envs=len(time_steps), step_dt=1.0 / 50.0
  )
  command_any.motion = MotionLoader(str(path), torch.tensor([0]))
  command_any.time_steps = torch.tensor(time_steps, dtype=torch.long)
  command_any.cfg = SimpleNamespace(lookahead_s=_PREVIEW_SECONDS)
  command_any.lookahead_steps = max(
    1, math.ceil(_PREVIEW_SECONDS * command_any.motion.fps)
  )
  return command


def _snapshot_from(commands: list[MotionCommand]) -> ObservationSnapshot:
  current_q = torch.stack([command.joint_pos[0] for command in commands])
  current_dq = torch.stack([command.joint_vel[0] for command in commands])
  preview = torch.stack([command.lookahead_command[0] for command in commands])
  rows = len(commands)
  zeros = torch.zeros(rows, JOINT_DIM)
  return ObservationSnapshot(
    reference_q=current_q,
    reference_dq=current_dq,
    anchor_orientation_error=torch.zeros(rows, 6),
    projected_gravity=torch.zeros(rows, 3),
    gyro=torch.zeros(rows, 3),
    relative_joint_q=zeros.clone(),
    joint_dq=zeros.clone(),
    previous_action=zeros.clone(),
    motion_lookahead=preview,
  )


def test_preview_packing_is_clip_local_at_normal_end_and_subset_reset(
  tmp_path: Path,
) -> None:
  clip_0 = tmp_path / "clip_0.npz"
  clip_1 = tmp_path / "clip_1.npz"
  positions_0, velocities_0 = _write_motion(clip_0, offset=0.0)
  positions_1, velocities_1 = _write_motion(clip_1, offset=10_000.0)

  # Row 0 is a normal frame; row 1 starts at its own clip's last frame.
  normal = _lookahead_command(clip_0, [0])
  at_end = _lookahead_command(clip_1, [14])
  assert normal.lookahead_steps == _PREVIEW_STEPS
  snapshot = _snapshot_from([normal, at_end])
  preview = pack_observations(snapshot, make_schema(preview=True))

  assert preview.reference.shape == (2, REFERENCE_DIM_WITH_PREVIEW)
  assert preview.conditioning.shape == (2, 99)
  torch.testing.assert_close(
    preview.reference[[0], :JOINT_DIM], torch.from_numpy(positions_0[[0]])
  )
  torch.testing.assert_close(
    preview.reference[1, :JOINT_DIM], torch.from_numpy(positions_1[14])
  )
  torch.testing.assert_close(
    preview.reference[[0], JOINT_DIM : 2 * JOINT_DIM],
    torch.from_numpy(velocities_0[[0]]),
  )
  torch.testing.assert_close(
    preview.reference[1, JOINT_DIM : 2 * JOINT_DIM],
    torch.from_numpy(velocities_1[14]),
  )
  torch.testing.assert_close(
    preview.reference[:, 2 * JOINT_DIM : 2 * JOINT_DIM + 6],
    snapshot.anchor_orientation_error,
  )
  # The normal row previews t + 0.2 s inside its own clip.
  torch.testing.assert_close(
    preview.reference[[0], REFERENCE_DIM : REFERENCE_DIM + JOINT_DIM],
    torch.from_numpy(positions_0[[_PREVIEW_STEPS]]),
  )
  torch.testing.assert_close(
    preview.reference[[0], REFERENCE_DIM + JOINT_DIM :],
    torch.from_numpy(velocities_0[[_PREVIEW_STEPS]]),
  )
  # The end row clamps to its own last frame and never reads another clip.
  torch.testing.assert_close(
    preview.reference[1, REFERENCE_DIM : REFERENCE_DIM + JOINT_DIM],
    torch.from_numpy(positions_1[14]),
  )
  torch.testing.assert_close(
    preview.reference[[1], REFERENCE_DIM + JOINT_DIM :],
    torch.from_numpy(velocities_1[[14]]),
  )

  # A subset reset moves only row 1 back inside its own clip.
  reset_row = _lookahead_command(clip_1, [2])
  reset_snapshot = _snapshot_from([normal, reset_row])
  reset = pack_observations(reset_snapshot, make_schema(preview=True))
  torch.testing.assert_close(reset.reference[:1], preview.reference[:1])
  torch.testing.assert_close(
    reset.reference[1, REFERENCE_DIM : REFERENCE_DIM + JOINT_DIM],
    torch.from_numpy(positions_1[2 + _PREVIEW_STEPS]),
  )
  torch.testing.assert_close(
    reset.reference[[1], REFERENCE_DIM + JOINT_DIM :],
    torch.from_numpy(velocities_1[[2 + _PREVIEW_STEPS]]),
  )

  # The version-1 layout stays 68 wide and ignores the declared preview.
  legacy = pack_observations(snapshot, make_schema())
  assert legacy.reference.shape == (2, REFERENCE_DIM)
  torch.testing.assert_close(legacy.reference[:, :JOINT_DIM], snapshot.reference_q)


def test_non_contiguous_manifest_ids_remain_selectable() -> None:
  payload = yaml.safe_load(Path(_MULTI_MOTION_MANIFEST).read_text())
  member_ids = tuple(str(entry["id"]) for entry in payload["teachers"])
  assert len(member_ids) == 144
  numbers = [int(member.split("_")[-1]) for member in member_ids]
  # The manifest skips clip numbers, so a range-based selector would drop members.
  assert numbers != list(range(numbers[0], numbers[0] + len(numbers)))

  assert resolve_rollout_ids(member_ids, "") == member_ids
  assert resolve_rollout_ids(member_ids, f"{member_ids[0]},{member_ids[-1]}") == (
    member_ids[0],
    member_ids[-1],
  )
  assert resolve_rollout_ids(member_ids, ",".join(member_ids)) == member_ids
