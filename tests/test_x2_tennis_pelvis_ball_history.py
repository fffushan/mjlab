"""Tests for the pelvis-frame ball history (``ball_hist_pelvis``).

The Ball3 term stores three timestamped ball samples. The ``-Ball3-Pelvis`` task
changes one thing: the frame those samples are stored in. Two claims are
load-bearing and are tested here on synthetic states.

1. **The transform.** ``ball_position_in_pelvis`` measures the ball from the pelvis
   body origin, in the pelvis body axes. Racket motion alone must not change it,
   which is the point of not using the plate frame.
2. **Capture-time frames.** A stored sample is never re-expressed in a later frame.
   When the pelvis moves, the samples already in the history keep their values, only
   their ages advance, and the next capture uses the new pose.

The third test is a config contract check: the new task's term name, frame and body
must be the ones documented, and the older Ball3 task must keep ``ball_hist`` in the
plate frame. The existing tests in ``test_x2_tennis_juggle_task.py`` stay the
regression net for the stream itself (schedule, ages, its own clock).

What these tests do **not** show: that the pelvis frame trains a better policy. No
policy is trained or run here.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl.exporter_utils import get_base_metadata
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.agibot_x2.tennis_juggle_env_cfg import (
  BALL_ENTITY,
  agibot_x2_tennis_juggle_env_cfg,
)
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz

TASK_ID = "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle"
BALL3_TASK_ID = f"{TASK_ID}-29DOF-Ball3"
BALL3_PELVIS_TASK_ID = f"{TASK_ID}-29DOF-Ball3-Pelvis"

STEP_DT = 0.02
"""The tennis control step (20 substeps of 1 ms), which the term reads from the env."""

PELVIS_BODY = 0
"""Body index of the pelvis in the stub scene: the root link."""

NOISE_BOUND_M = 0.01
"""Above the per-axis half-widths of ``BALL_POSITION_NOISE_M`` (5-8 mm)."""


class _StubPelvisEnv:
  """The pelvis and ball state the pelvis-frame terms read, and nothing else.

  Driving the terms directly keeps these tests off the physics path: the claims are
  about a frame change and a stored value, so a scripted pose is enough, and it is
  the only way to move the pelvis between two acquisitions.
  """

  def __init__(self, num_envs: int = 1):
    self.num_envs = num_envs
    self.device = "cpu"
    self.step_dt = STEP_DT
    # The term treats a repeated value as a re-entry and serves its cached output,
    # so a test has to advance this exactly as the env would.
    self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
    self.pelvis_pos = torch.zeros(num_envs, 3)
    self.pelvis_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(num_envs, 4).clone()
    self.ball_pos = torch.zeros(num_envs, 3)
    # A racket body, present only to prove the pelvis frame ignores it.
    self.racket_pos = torch.zeros(num_envs, 3)
    self.racket_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(num_envs, 4).clone()
    self.scene = {
      "robot": SimpleNamespace(
        data=SimpleNamespace(body_link_pos_w=None, body_link_quat_w=None)
      ),
      BALL_ENTITY: SimpleNamespace(data=SimpleNamespace(root_link_pos_w=self.ball_pos)),
    }
    self.place()

  def place(
    self,
    pelvis_pos: torch.Tensor | None = None,
    pelvis_quat: torch.Tensor | None = None,
    ball_pos: torch.Tensor | None = None,
    racket_pos: torch.Tensor | None = None,
    racket_quat: torch.Tensor | None = None,
  ) -> None:
    """Overwrite any subset of the scripted state, then rebuild the scene tensors."""
    if pelvis_pos is not None:
      self.pelvis_pos = pelvis_pos
    if pelvis_quat is not None:
      self.pelvis_quat = pelvis_quat
    if ball_pos is not None:
      self.ball_pos = ball_pos
    if racket_pos is not None:
      self.racket_pos = racket_pos
    if racket_quat is not None:
      self.racket_quat = racket_quat
    self.scene["robot"].data.body_link_pos_w = torch.stack(
      [self.pelvis_pos, self.racket_pos], dim=1
    )
    self.scene["robot"].data.body_link_quat_w = torch.stack(
      [self.pelvis_quat, self.racket_quat], dim=1
    )
    self.scene[BALL_ENTITY].data.root_link_pos_w = self.ball_pos


def _pelvis_history():
  """The history term in the pelvis frame on a fixed, zero-latency schedule.

  One acquisition per control step with no latency, stall or noise schedule jitter
  makes the slot contents predictable step by step. The per-sample noise stays on:
  it is part of the term, and the assertions that matter compare a served sample
  with itself, which noise cannot disturb.
  """
  env = cast(Any, _StubPelvisEnv())
  params = {
    "pelvis_cfg": SceneEntityCfg("robot", body_ids=[PELVIS_BODY]),
    "ball_cfg": SceneEntityCfg(BALL_ENTITY),
    "reference_frame": "pelvis",
    "stream_rate_hz": (50.0, 50.0),
    "stream_latency_s": (0.0, 0.0),
    "stall_prob": 0.0,
    "stall_range_s": (0.0, 0.0),
  }
  term = mdp.ball_history_samples(
    ObservationTermCfg(func=mdp.ball_history_samples, params=params), env
  )
  return env, term, params


def _step(term, env, params) -> torch.Tensor:
  """One control step: advance the episode clock, then read the term."""
  env.episode_length_buf += 1
  return term(env, **params)


def test_the_pelvis_frame_follows_the_pelvis_pose_and_not_the_racket() -> None:
  """A translated and rotated pelvis, then a racket that must change nothing."""
  env = cast(Any, _StubPelvisEnv())
  pelvis_cfg = SceneEntityCfg("robot", body_ids=[PELVIS_BODY])
  ball_cfg = SceneEntityCfg(BALL_ENTITY)

  # No rotation at the origin-relative pose: the pelvis frame is the world offset.
  env.place(
    pelvis_pos=torch.tensor([[0.0, 0.0, 0.80]]),
    ball_pos=torch.tensor([[0.40, 0.20, 1.05]]),
  )
  got = mdp.ball_position_in_pelvis(env, pelvis_cfg, ball_cfg)
  assert torch.allclose(got, torch.tensor([[0.40, 0.20, 0.25]]), atol=1e-6)

  # Turned 90 deg about z: world +y maps to pelvis +x, and world +x to pelvis -y.
  env.place(
    pelvis_pos=torch.tensor([[1.0, 0.0, 0.80]]),
    pelvis_quat=quat_from_euler_xyz(
      torch.zeros(1), torch.zeros(1), torch.tensor([math.pi / 2])
    ),
    ball_pos=torch.tensor([[1.00, 0.25, 1.05]]),
  )
  got = mdp.ball_position_in_pelvis(env, pelvis_cfg, ball_cfg)
  assert torch.allclose(got, torch.tensor([[0.25, 0.00, 0.25]]), atol=1e-6)

  # An arbitrary rotation, checked against the forward rotation: rotating the result
  # back by the pelvis orientation must recover the world offset. This catches a
  # transposed or hand-flipped transform on any axis, not only yaw.
  quat = quat_from_euler_xyz(
    torch.tensor([0.3]), torch.tensor([-0.4]), torch.tensor([1.1])
  )
  env.place(
    pelvis_pos=torch.tensor([[-0.2, 0.5, 0.7]]),
    pelvis_quat=quat,
    ball_pos=torch.tensor([[0.6, -0.1, 1.3]]),
  )
  got = mdp.ball_position_in_pelvis(env, pelvis_cfg, ball_cfg)
  assert torch.allclose(
    quat_apply(quat, got), torch.tensor([[0.8, -0.6, 0.6]]), atol=1e-6
  )

  # The racket is not an argument of this transform, so moving it cannot change the
  # result. In the plate frame the racket's own pose is folded into the ball input.
  before = got.clone()
  env.place(
    racket_pos=torch.tensor([[0.4, -0.3, 1.2]]),
    racket_quat=quat_from_euler_xyz(
      torch.tensor([0.4]), torch.tensor([0.2]), torch.tensor([0.5])
    ),
  )
  assert torch.equal(mdp.ball_position_in_pelvis(env, pelvis_cfg, ball_cfg), before)


def test_a_stored_sample_keeps_its_capture_pose_and_only_its_age_advances() -> None:
  """The robot walks; the samples it captured must not be re-expressed."""
  env, term, params = _pelvis_history()
  env.place(
    pelvis_pos=torch.tensor([[0.0, 0.0, 0.8]]),
    ball_pos=torch.tensor([[0.5, 0.2, 1.1]]),
  )
  # Three control steps at the standing pose: one acquisition each, so the history
  # is full and slot 0 is the sample captured at the third step.
  captured = [_step(term, env, params) for _ in range(3)]
  newest = captured[2][:, :3].clone()
  assert torch.allclose(newest, torch.tensor([[0.5, 0.2, 0.3]]), atol=NOISE_BOUND_M)

  # The robot walks 1 m along +x and turns 90 deg. The ball stays where it was.
  env.place(
    pelvis_pos=torch.tensor([[1.0, 0.0, 0.8]]),
    pelvis_quat=quat_from_euler_xyz(
      torch.zeros(1), torch.zeros(1), torch.tensor([math.pi / 2])
    ),
  )
  after = _step(term, env, params)

  # The sample captured at the old pose is served byte for byte from slot 1. A term
  # that re-expressed stored samples in the current frame would move it by 0.42 m.
  assert torch.equal(after[:, 4:7], newest)
  # Its age advanced by one control step instead of the sample being refreshed.
  assert after[:, 7].item() == pytest.approx(captured[2][:, 3].item() + STEP_DT)
  # The new capture uses the new pose: R^T (ball - pelvis) = (0.2, 0.5, 0.3), not
  # the old frame's (0.5, 0.2, 0.3).
  assert torch.allclose(
    after[:, :3], torch.tensor([[0.2, 0.5, 0.3]]), atol=NOISE_BOUND_M
  )
  assert (after[:, :3] - newest).abs().max() > 0.1


def test_the_pelvis_task_swaps_the_history_frame_only() -> None:
  """The new task differs from Ball3 in the term's name and frame, not its stream."""
  assert BALL3_TASK_ID in list_tasks()
  assert BALL3_PELVIS_TASK_ID in list_tasks()
  pelvis_cfg = load_env_cfg(BALL3_PELVIS_TASK_ID)
  ball3_cfg = load_env_cfg(BALL3_TASK_ID)

  actor = pelvis_cfg.observations["actor"].terms
  assert "ball_hist" not in actor
  term = actor["ball_hist_pelvis"]
  assert term.func is mdp.ball_history_samples
  assert term.params["reference_frame"] == "pelvis"
  assert term.params["pelvis_cfg"].body_names == ("pelvis",)
  assert "plate_cfg" not in term.params
  # The term applies its own corruption, so the manager must not add any.
  assert term.noise is None
  assert term.delay_max_lag == 0

  # The stream magnitudes, the bias and the noise are Ball3's: only the frame moved.
  # Ball3's params stay byte-for-byte what they were, so the frame there is the
  # term's documented default, not a new recorded key.
  legacy = ball3_cfg.observations["actor"].terms["ball_hist"]
  assert "reference_frame" not in legacy.params
  assert legacy.params["plate_cfg"].body_names == ("tennis_racket",)
  for key in (
    "bias_half_width_m",
    "stream_rate_hz",
    "stream_latency_s",
    "stall_prob",
    "stall_range_s",
  ):
    assert term.params[key] == legacy.params[key], key

  # Every other term is the same term, so the widths cannot have changed.
  assert set(actor) - {"ball_hist_pelvis"} == set(
    ball3_cfg.observations["actor"].terms
  ) - {"ball_hist"}
  for name in set(actor) - {"ball_hist_pelvis"}:
    assert actor[name].func is ball3_cfg.observations["actor"].terms[name].func, name
    assert actor[name].params == ball3_cfg.observations["actor"].terms[name].params, (
      name
    )
  assert pelvis_cfg.observations["critic"] == ball3_cfg.observations["critic"]
  assert pelvis_cfg.actions == ball3_cfg.actions
  assert pelvis_cfg.rewards == ball3_cfg.rewards

  # The older tasks still build the plate-frame term, so their checkpoints keep
  # their meaning. An unset frame argument is the shipped behaviour.
  for legacy_cfg in (
    agibot_x2_tennis_juggle_env_cfg(head_actuated=False, ball_history=True),
    agibot_x2_tennis_juggle_env_cfg(),
  ):
    terms = legacy_cfg.observations["actor"].terms
    if "ball_hist" in terms:
      assert "reference_frame" not in terms["ball_hist"].params
    else:
      assert "ball_pos" in terms


def test_an_invalid_history_frame_setting_is_refused() -> None:
  """Do not silently ignore the requested frame or accept a frame typo."""
  with pytest.raises(ValueError, match="ball_history_frame"):
    agibot_x2_tennis_juggle_env_cfg(
      head_actuated=False, ball_history=True, ball_history_frame="torso"
    )
  with pytest.raises(ValueError, match="requires ball_history=True"):
    agibot_x2_tennis_juggle_env_cfg(ball_history_frame="pelvis")


@pytest.mark.slow
def test_the_pelvis_term_builds_in_the_task_and_keeps_the_widths() -> None:
  """The pelvis body must resolve, and the actor output must stay 12 numbers wide."""
  cfg = load_env_cfg(BALL3_PELVIS_TASK_ID)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg, device="cpu", num_envs=1)
  try:
    env.reset()
    om = env.observation_manager
    pelvis = om.get_term_cfg("actor", "ball_hist_pelvis").params["pelvis_cfg"]
    assert len(pelvis.body_ids) == 1, "the term must resolve exactly one pelvis body"
    observations = cast("dict[str, torch.Tensor]", env.obs_buf)
    assert observations["actor"].shape[-1] == 112
    assert observations["critic"].shape[-1] == 121
    assert env.action_manager.total_action_dim == 29
    metadata = get_base_metadata(env, run_path="pelvis-history-test")
    assert cast(list[str], metadata["observation_names"])[-1] == "ball_hist_pelvis"
  finally:
    env.close()
