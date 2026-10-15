"""CPU tests for the single-teacher snapshot adapter boundary."""

from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace
from typing import Any

import mujoco
import pytest
import torch

from mjlab.tasks.tracking.distillation.adapter import (
  DistillationEnvironmentAdapter,
  default_schema_for_cohort,
  validate_live_contract,
)
from mjlab.tasks.tracking.distillation.config import (
  DistillationError,
  load_manifest,
  resolve_cohort,
)
from mjlab.tasks.tracking.distillation.environment import (
  SegmentMotionCommand,
  build_distillation_environment,
)
from mjlab.tasks.tracking.distillation.vae_config import (
  PREVIEW_SCHEMA_VERSION,
  SCHEMA_VERSION,
  make_schema,
)


class FakeSegmentMotionCommand(SegmentMotionCommand):
  _body_pos_w: torch.Tensor
  _body_quat_w: torch.Tensor
  _robot_body_pos_w: torch.Tensor
  _robot_body_quat_w: torch.Tensor
  _anchor_pos_w: torch.Tensor
  _robot_anchor_pos_w: torch.Tensor

  @property
  def body_pos_w(self) -> torch.Tensor:
    return self._body_pos_w

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self._body_quat_w

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self._robot_body_pos_w

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self._robot_body_quat_w

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return self._anchor_pos_w

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self._robot_anchor_pos_w


_TERM_NAMES_WIDTHS = (
  ("command", 62),
  ("motion_anchor_ori_b", 6),
  ("base_ang_vel", 3),
  ("joint_pos", 31),
  ("joint_vel", 31),
  ("actions", 31),
)


_BASE_TERM_NAMES_WIDTHS = (
  ("command", 62),
  ("motion_anchor_ori_b", 6),
  ("base_ang_vel", 3),
  ("joint_pos", 31),
  ("joint_vel", 31),
  ("actions", 31),
)


def _term_names_widths(lookahead_width: int) -> tuple[tuple[str, int], ...]:
  """The X2 actor layout, with the reference preview the teacher observed inserted."""
  if lookahead_width == 0:
    return _BASE_TERM_NAMES_WIDTHS
  return (
    ("command", 62),
    ("motion_lookahead", lookahead_width),
    *_BASE_TERM_NAMES_WIDTHS[1:],
  )


def make_fake_adapter(
  lookahead_width: int = 0, lookahead_s: float | None = None
) -> tuple[DistillationEnvironmentAdapter, SimpleNamespace]:
  layouts = _term_names_widths(lookahead_width)
  terms = tuple(SimpleNamespace(name=name, width=width) for name, width in layouts)
  observations = SimpleNamespace(
    terms=terms,
    names=tuple(name for name, _ in layouts),
  )
  actions = SimpleNamespace(
    dim=31,
    joint_names=tuple(f"joint_{index:02d}" for index in range(31)),
  )
  # The real contract always declares one cross-member preview duration, and zero keeps
  # the frozen version-1 reference layout.
  cohort = SimpleNamespace(
    observations=observations,
    actions=actions,
    lookahead_s=(
      (0.2 if lookahead_width else 0.0) if lookahead_s is None else lookahead_s
    ),
  )

  command: Any = object.__new__(FakeSegmentMotionCommand)
  command.cfg = SimpleNamespace(entity_name="robot")
  command.time_steps = torch.tensor([4, 5], dtype=torch.long)
  command.segment_ids = torch.tensor([2, 3], dtype=torch.long)
  command.generation_ids = torch.tensor([7, 8], dtype=torch.long)
  command._body_pos_w = torch.zeros(2, 1, 3)
  command._body_quat_w = torch.tensor([[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]])
  command._robot_body_pos_w = torch.zeros(2, 1, 3)
  command._robot_body_quat_w = command._body_quat_w.clone()
  command._anchor_pos_w = torch.zeros(2, 3)
  command._robot_anchor_pos_w = torch.zeros(2, 3)
  robot = SimpleNamespace(
    data=SimpleNamespace(
      projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    )
  )

  total_width = sum(width for _, width in layouts)
  actor = torch.arange(2 * total_width, dtype=torch.float32).reshape(2, total_width)
  env = SimpleNamespace(
    num_envs=2,
    device=torch.device("cpu"),
    command_manager=SimpleNamespace(get_term=lambda name: command),
    scene={"robot": robot},
    get_observations=lambda: {"actor": actor},
  )
  adapter = DistillationEnvironmentAdapter(
    env,
    cohort,  # type: ignore[arg-type]
    teacher=object(),  # type: ignore[arg-type]
    audit=object(),  # type: ignore[arg-type]
  )
  return adapter, env


def test_real_saved_live_observation_semantic_mismatches_are_rejected() -> None:
  cohort = resolve_cohort(
    load_manifest("configs/distillation/x2_tennis.yaml", repo_root=".")
  )
  output = io.StringIO()
  with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
    env = build_distillation_environment(cohort, "tennis_000", num_envs=1, device="cpu")
  try:
    term = env.observation_manager.get_term_cfg("actor", "joint_pos")
    original_scale = term.scale
    term.scale = (2.0,)
    with pytest.raises(ValueError, match="scale"):
      validate_live_contract(env, cohort)
    term.scale = original_scale

    term.params["biased"] = False
    with pytest.raises(ValueError, match="parameters"):
      validate_live_contract(env, cohort)
    term.params["biased"] = True

    sensor_term = env.observation_manager.get_term_cfg("actor", "base_ang_vel")
    original_sensor_name = sensor_term.params["sensor_name"]
    sensor_term.params["sensor_name"] = "robot/imu_lin_vel"
    with pytest.raises(ValueError, match="parameters"):
      validate_live_contract(env, cohort)
    sensor_term.params["sensor_name"] = original_sensor_name

    original_func = term.func
    term.func = lambda *_args, **_kwargs: torch.zeros(1, 31)
    with pytest.raises(ValueError, match="function"):
      validate_live_contract(env, cohort)
    term.func = original_func

    term.delay_update_period = 2
    with pytest.raises(ValueError, match="delay_update_period"):
      validate_live_contract(env, cohort)
    term.delay_update_period = 1

    env.observation_manager.cfg["actor"].enable_corruption = False
    with pytest.raises(ValueError, match="corruption"):
      validate_live_contract(env, cohort)
    env.observation_manager.cfg["actor"].enable_corruption = True

    model = env.sim.mj_model
    original_object_type = int(model.sensor_objtype[0])
    model.sensor_objtype[0] = int(mujoco.mjtObj.mjOBJ_BODY)
    with pytest.raises(ValueError, match="attached to a site"):
      validate_live_contract(env, cohort)
    model.sensor_objtype[0] = original_object_type

    original_site_quat = model.site_quat[0].copy()
    model.site_quat[0] = (0.0, 1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="non-identity"):
      validate_live_contract(env, cohort)
    model.site_quat[0] = original_site_quat
  finally:
    env.close()


def test_adapter_exposes_reset_and_pre_step_boundary_events() -> None:
  cohort = resolve_cohort(
    load_manifest("configs/distillation/x2_tennis.yaml", repo_root=".")
  )
  output = io.StringIO()
  with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
    env = build_distillation_environment(cohort, "tennis_000", num_envs=1, device="cpu")
  try:
    audit = validate_live_contract(env, cohort)
    adapter = DistillationEnvironmentAdapter(
      env,
      cohort,
      teacher=object(),  # type: ignore[arg-type]
      audit=audit,
    )
    reset_snapshot = adapter.reset(seed=11)
    assert reset_snapshot.boundary_events is not None
    assert reset_snapshot.boundary_events.interrupted.tolist() == [True]
    step = adapter.step(torch.zeros(1, 31))
    assert step.events is not None
    assert step.events.pre_generation.shape == (1,)
    assert step.events.reasons == ("unavailable",)
  finally:
    env.close()

  _, env = make_fake_adapter()
  terms = tuple(
    SimpleNamespace(name=name, width=width) for name, width in _TERM_NAMES_WIDTHS
  )
  cohort = SimpleNamespace(
    observations=SimpleNamespace(
      terms=terms, names=tuple(name for name, _ in _TERM_NAMES_WIDTHS)
    ),
    actions=SimpleNamespace(
      dim=31, joint_names=tuple(f"joint_{index:02d}" for index in range(31))
    ),
  )
  reversed_schema = make_schema(
    joint_order=tuple(f"joint_{index:02d}" for index in reversed(range(31)))
  )
  with pytest.raises(ValueError, match="joint_order"):
    DistillationEnvironmentAdapter(
      env,
      cohort,  # type: ignore[arg-type]
      teacher=object(),  # type: ignore[arg-type]
      audit=object(),  # type: ignore[arg-type]
      schema=reversed_schema,
    )


def test_snapshot_reuses_cached_actor_observation_and_owns_tensors() -> None:
  adapter, env = make_fake_adapter()
  first = adapter.snapshot()
  source = env.get_observations()["actor"]
  source[0, 0] = -1000.0
  assert first.teacher_observation[0, 0].item() == 0.0
  assert first.features.reference_q[0, 0].item() == 0.0
  assert first.features.previous_action.shape == (2, 31)
  assert first.reference_frame.tolist() == [4, 5]
  assert first.segment_id.tolist() == [2, 3]
  assert first.generation_id.tolist() == [7, 8]
  assert first.metrics is not None
  assert first.metrics.root_relative_pose_error.shape == (2,)
  assert first.metrics.heading_error.shape == (2,)


def test_physical_metrics_keep_root_relative_pose_separate_from_heading() -> None:
  adapter, env = make_fake_adapter()
  command = env.command_manager.get_term("motion")
  command._body_pos_w = torch.tensor(
    [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
  )
  command._robot_body_pos_w = torch.tensor(
    [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
  )
  command._body_quat_w = torch.tensor(
    [
      [[0.7071068, 0.0, 0.0, 0.7071068]],
      [[0.7071068, 0.7071068, 0.0, 0.0]],
    ]
  )
  command._robot_body_quat_w = torch.tensor(
    [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]]
  )
  metrics = adapter.snapshot().metrics
  assert metrics is not None
  assert metrics.root_relative_pose_error[0] > 0.5
  torch.testing.assert_close(
    metrics.heading_error[0], torch.tensor(torch.pi / 2), atol=1e-5, rtol=0
  )
  torch.testing.assert_close(
    metrics.heading_error[1], torch.tensor(0.0), atol=1e-5, rtol=0
  )


def test_snapshot_has_shared_feature_values_without_second_group_compute() -> None:
  adapter, env = make_fake_adapter()
  calls = 0

  original = env.get_observations

  def cached_observations() -> dict[str, torch.Tensor]:
    nonlocal calls
    calls += 1
    return original()

  env.get_observations = cached_observations
  snapshot = adapter.snapshot()
  assert calls == 1
  # Reference q/dq and previous action are slices of the one owned actor copy.
  torch.testing.assert_close(
    snapshot.features.reference_q, snapshot.teacher_observation[:, :31]
  )
  torch.testing.assert_close(
    snapshot.features.reference_dq, snapshot.teacher_observation[:, 31:62]
  )
  torch.testing.assert_close(
    snapshot.features.previous_action, snapshot.teacher_observation[:, 133:164]
  )


def test_a_cohort_without_a_preview_keeps_the_frozen_version_one_layout() -> None:
  """Every cohort written before the preview layout must be unaffected."""
  adapter, _ = make_fake_adapter()
  assert adapter.schema.version == SCHEMA_VERSION
  assert (adapter.schema.reference_dim, adapter.schema.conditioning_dim) == (68, 99)
  assert adapter.snapshot().packed.reference.shape == (2, 68)


def test_a_preview_cohort_packs_the_teacher_preview_into_the_reference() -> None:
  """The student must see the future reference the teacher used."""
  adapter, _ = make_fake_adapter(lookahead_width=62)
  assert adapter.schema.version == PREVIEW_SCHEMA_VERSION
  assert adapter.schema.reference_dim == 130
  # The decoder stays proprioception-only at 99.
  assert adapter.schema.conditioning_dim == 99

  snapshot = adapter.snapshot()
  reference = snapshot.packed.reference
  assert reference.shape == (2, 130)
  # The teacher layout is command, motion_lookahead, motion_anchor_ori_b, ... and the
  # packed reference is q, dq, anchor, preview, so the preview moves to the end.
  actor = torch.arange(2 * 226, dtype=torch.float32).reshape(2, 226)
  torch.testing.assert_close(reference[:, 68:], actor[:, 62:124])
  torch.testing.assert_close(reference[:, 62:68], actor[:, 124:130])
  # The version-1 prefix is unmoved.
  torch.testing.assert_close(reference[:, :62], actor[:, :62])


def test_a_version_two_declaration_without_the_preview_term_fails_closed() -> None:
  """A teacher observation that declares no preview cannot satisfy the new layout."""
  adapter, _ = make_fake_adapter(lookahead_width=0, lookahead_s=0.2)
  assert adapter.schema.version == PREVIEW_SCHEMA_VERSION
  with pytest.raises(DistillationError, match="motion_lookahead"):
    adapter.snapshot()


def test_the_cohort_preview_duration_selects_the_schema_version() -> None:
  """The derivation rule is one place, so it cannot disagree with the observation."""
  cohort = SimpleNamespace(
    actions=SimpleNamespace(joint_names=tuple(f"joint_{i:02d}" for i in range(31))),
    lookahead_s=0.0,
  )
  assert default_schema_for_cohort(cohort).version == SCHEMA_VERSION
  cohort.lookahead_s = 0.2
  assert default_schema_for_cohort(cohort).version == PREVIEW_SCHEMA_VERSION
