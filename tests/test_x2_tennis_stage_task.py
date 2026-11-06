"""Tests for the staged fine-tune's rungs.

A stage is the nominal juggling pilot (S0) with one input condition moved. The risk
a stage carries is that it silently moves a second thing, which would break the
resume from the previous checkpoint with its optimizer state. Five claims are
tested, each over every rung declared in ``STAGE_DELTAS``:

1. **The stage's stream.** The resolved ball-history parameters are exactly the
   stage's declared rate and latency range.
2. **The one-delta guarantee.** Every other ball-history parameter equals the S0
   factory's value for that key.
3. **The contract is unchanged.** Actor 112, critic 121, 29 actions, and the head
   joints are not in the action term.
4. **Registration.** The task loads and reports the S0 experiment name.
5. **An unknown stage is refused.**

What these tests do **not** show: that a policy learns under these settings. The
rung is not trained here.
"""

from __future__ import annotations

from typing import cast

import pytest
import torch

from mjlab.asset_zoo.robots.agibot_x2.tennis_constants import HEAD_JOINT_NAMES
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.agibot_x2.tennis_juggle_nominal_env_cfg import (
  agibot_x2_tennis_juggle_nominal_env_cfg,
)
from mjlab.tasks.velocity.config.agibot_x2.tennis_juggle_stage_env_cfg import (
  STAGE_DELTAS,
  agibot_x2_tennis_juggle_stage_env_cfg,
)

STAGE_TASK_ID = (
  "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF"
  "-Ball3-Pelvis-Nominal-S1-15Hz"
)

S0_EXPERIMENT_NAME = "agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal"

STAGE_KEYS = ("stream_rate_hz", "stream_latency_s")
"""The two ball-history parameters every stage declares."""

STAGE_MOVABLE_KEYS = (
  *STAGE_KEYS,
  "stall_prob",
  "stall_range_s",
  "position_noise_m",
  "bias_half_width_m",
)
"""Every ball-history parameter a stage may move: the stream, its stall model and its
measurement error. A stage that moves any of these must declare it."""

STAGES = sorted(STAGE_DELTAS)
"""Every declared rung. A new entry must satisfy the checks below without edits."""


##
# 1. The stage's stream.
##


@pytest.mark.parametrize("stage", STAGES)
def test_each_stage_resolves_its_declared_stream(stage: str) -> None:
  params = (
    agibot_x2_tennis_juggle_stage_env_cfg(stage=stage)
    .observations["actor"]
    .terms["ball_hist_pelvis"]
    .params
  )
  for key in STAGE_KEYS:
    assert params[key] == STAGE_DELTAS[stage][key], key


##
# 2. The one-delta guarantee.
##


@pytest.mark.parametrize("stage", STAGES)
def test_the_stage_changes_nothing_but_the_declared_condition(stage: str) -> None:
  """A stage that moved an undeclared parameter could not resume from its parent.

  A stage may move the stream and, since S4, the stall model - but only the keys it
  declares. Every undeclared key is compared by value, including the nested
  ``SceneEntityCfg`` objects, so a retargeted pelvis body or a stall that appeared
  without being declared would fail here.
  """
  declared = STAGE_DELTAS[stage]
  stage_cfg = agibot_x2_tennis_juggle_stage_env_cfg(stage=stage)
  nominal = agibot_x2_tennis_juggle_nominal_env_cfg()
  stage = stage_cfg
  stage_params = stage.observations["actor"].terms["ball_hist_pelvis"].params
  nominal_params = nominal.observations["actor"].terms["ball_hist_pelvis"].params

  assert set(stage_params) == set(nominal_params)
  for key in STAGE_KEYS:
    assert stage_params[key] != nominal_params[key], key
  # A stage may move the stall model, but only by declaring it: a stall key it does
  # not declare must still equal the nominal pilot's.
  for key in STAGE_MOVABLE_KEYS:
    if key in declared:
      continue
    assert stage_params[key] == nominal_params[key], key
  stage_rest = {k: v for k, v in stage_params.items() if k not in STAGE_MOVABLE_KEYS}
  nominal_rest = {
    k: v for k, v in nominal_params.items() if k not in STAGE_MOVABLE_KEYS
  }
  assert stage_rest == nominal_rest

  # The term's own parameters are not the only way a stage could move a second
  # condition. Compare everything a stage could reach without editing the stage
  # module: a reward weight, a termination, a metric, the command, a reset, the
  # critic group or the corruption switch. A rung that changed one of these would
  # look like a stream experiment and would not be one.
  #
  # A stage that declares a ``hold`` section changes the episode structure, so it is
  # allowed a fixed, named set of additions - and nothing else. The whitelist is
  # written out here rather than read from the factory, so adding a second reward or
  # termination change to the hold path fails until this list is deliberately edited.
  hold_declared = "hold" in declared
  assert stage.actions == nominal.actions
  assert stage.commands == nominal.commands
  if not hold_declared:
    for name in ("rewards", "terminations", "metrics"):
      assert getattr(stage, name) == getattr(nominal, name), name
  else:
    # Rewards: only the energy term may be gated, and only one term may be added.
    assert set(stage.rewards) == set(nominal.rewards) | {"racket_under_ball"}
    for name in nominal.rewards:
      if name == "ball_energy":
        continue
      assert stage.rewards[name] == nominal.rewards[name], name
    gated = stage.rewards["ball_energy"]
    assert gated.func is mdp.ball_energy_after_release
    assert gated.weight == nominal.rewards["ball_energy"].weight
    gated_params = dict(gated.params)
    assert gated_params.pop("hold_event") == "hold_ball"
    assert gated_params == nominal.rewards["ball_energy"].params
    assert stage.rewards["racket_under_ball"].params["hold_event"] == "hold_ball"

    # Terminations: one added term, and the hold pause on the no-bounce deadline.
    assert set(stage.terminations) == set(nominal.terminations) | {"held_ball_contact"}
    for name in nominal.terminations:
      if name == "ball_no_bounce":
        continue
      assert stage.terminations[name] == nominal.terminations[name], name
    contact = stage.terminations["held_ball_contact"]
    assert contact.func is mdp.held_ball_contact
    assert contact.params["hold_event"] == "hold_ball"
    paused = dict(stage.terminations["ball_no_bounce"].params)
    assert paused.pop("hold_event") == "hold_ball"
    assert paused == nominal.terminations["ball_no_bounce"].params

    # Metrics: the two hold facts, logging only.
    assert set(stage.metrics) == set(nominal.metrics) | {
      "ball_held",
      "ball_hold_delay_s",
      "ball_release_alignment_m",
    }
    for name in nominal.metrics:
      assert stage.metrics[name] == nominal.metrics[name], name

  # Events: a stage may declare a ``reset`` section, which moves ``events["reset_ball"]``
  # and nothing else. Every other event term must still equal S0's, and inside
  # ``reset_ball`` only the declared keys may differ - an undeclared reset change would
  # alter what the episode starts from and could not be read as a stream experiment.
  hold_events = {"hold_ball", "advance_ball_hold"}
  if hold_declared:
    assert set(stage.events) == set(nominal.events) | hold_events
    assert stage.events["hold_ball"].mode == "reset"
    assert stage.events["advance_ball_hold"].mode == "step"
    # The hold beat must be sampled after the reset that seats the ball.
    order = list(stage.events)
    assert order.index("hold_ball") > order.index("reset_ball")
  else:
    assert set(stage.events) == set(nominal.events)
  reset_declared = declared.get("reset", {})
  for name in stage.events:
    if name in hold_events:
      continue
    if name != "reset_ball":
      assert stage.events[name] == nominal.events[name], name
      continue
    stage_reset = stage.events["reset_ball"].params
    nominal_reset = nominal.events["reset_ball"].params
    assert set(stage_reset) == set(nominal_reset)
    for key in stage_reset:
      if key in reset_declared:
        assert stage_reset[key] == reset_declared[key], key
        assert stage_reset[key] != nominal_reset[key], key
      else:
        assert stage_reset[key] == nominal_reset[key], key
  assert stage.observations["critic"] == nominal.observations["critic"]

  stage_actor = stage.observations["actor"]
  nominal_actor = nominal.observations["actor"]
  assert stage_actor.enable_corruption == nominal_actor.enable_corruption
  assert set(stage_actor.terms) == set(nominal_actor.terms)
  for name in stage_actor.terms:
    if name == "ball_hist_pelvis":
      continue
    assert stage_actor.terms[name] == nominal_actor.terms[name], name


##
# 3. The contract is unchanged.
##


def test_the_stage_keeps_the_widths_and_locks_the_head() -> None:
  cfg = load_env_cfg(STAGE_TASK_ID)
  num_envs = 4
  cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg, device="cpu", num_envs=num_envs)
  try:
    env.reset()
    observations = cast("dict[str, torch.Tensor]", env.obs_buf)
    assert observations["actor"].shape[-1] == 112
    assert observations["critic"].shape[-1] == 121
    assert env.action_manager.total_action_dim == 29

    # The head stays an observation (the camera rides on it) and leaves the action
    # space, so the head pose is the fixed default the nominal pilot resets to.
    names = list(env.scene["robot"].joint_names)
    head_ids = [names.index(name) for name in HEAD_JOINT_NAMES]
    action_ids = env.action_manager.get_term("joint_pos").target_ids.tolist()
    assert set(head_ids).isdisjoint(action_ids)
  finally:
    env.close()


##
# 4. Registration.
##


def test_the_stage_is_registered_against_the_s0_experiment() -> None:
  assert STAGE_TASK_ID in list_tasks()
  params = (
    load_env_cfg(STAGE_TASK_ID).observations["actor"].terms["ball_hist_pelvis"].params
  )
  assert params["stream_rate_hz"] == (15.0, 15.0)
  # Same experiment name as S0 on purpose: the run resumes with --agent.load-run
  # pointing at an S0 run directory inside the same log root.
  assert load_rl_cfg(STAGE_TASK_ID).experiment_name == S0_EXPERIMENT_NAME


##
# 5. An unknown stage is refused.
##


def test_an_unknown_stage_is_refused() -> None:
  with pytest.raises(ValueError, match="s9-nope"):
    agibot_x2_tennis_juggle_stage_env_cfg(stage="s9-nope")
