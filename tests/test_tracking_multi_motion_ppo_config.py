"""Configuration contracts for the X2 multi-motion PPO teacher task.

The tests pin the registered task, the six-layer actor and critic, the preview
wiring in both observation groups, the endpoint termination, and the unchanged
single-motion vendor-gain task.  No MuJoCo model is compiled, so live
environment evidence stays with the monitored smoke run.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.config.agibot_x2.env_cfgs import (
  agibot_x2_flat_tracking_increated_perturbations_vendor_gains_env_cfg,
)
from mjlab.tasks.tracking.mdp import MotionCommandCfg, motion_lookahead
from mjlab.tasks.tracking.mdp.multi_motion_ppo import (
  MultiMotionPpoCommandCfg,
  clip_complete,
)
from mjlab.tasks.tracking.rl import (
  MOTION_LOOKAHEAD_STEPS_KEY,
  MotionTrackingOnPolicyRunner,
  MultiMotionTrackingOnPolicyRunner,
  motion_export_metadata,
  resolve_lookahead_steps,
)
from mjlab.utils.os import dump_yaml, load_saved_yaml

TASK = "Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion"
SINGLE_MOTION_TASK = (
  "Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Increated-Perturbations-"
  "Vendor-Gains"
)
JOINT_DIM = 31
CONTROL_HZ = 50.0
PREVIEW_STEPS = 10
HIDDEN_DIMS = (1024, 1024, 1024, 1024, 512, 128)
EXPECTED_ACTOR_WIDTH = 226

# Declared width of every actor observation term of the no-state-estimation X2
# actor at JOINT_DIM joints: the command and the preview each carry one frame of
# reference q and dq, the anchor orientation error contributes the first two
# matrix rows, and the remaining terms are joint- or sensor-wide.
ACTOR_TERM_WIDTHS = {
  "command": 2 * JOINT_DIM,
  "motion_lookahead": 2 * JOINT_DIM,
  "motion_anchor_ori_b": 6,
  "base_ang_vel": 3,
  "joint_pos": JOINT_DIM,
  "joint_vel": JOINT_DIM,
  "actions": JOINT_DIM,
}

# The privileged critic keeps its base terms and reads the preview second, in
# the same place the actor does.
CRITIC_TERM_ORDER = (
  "command",
  "motion_lookahead",
  "motion_anchor_pos_b",
  "motion_anchor_ori_b",
  "body_pos",
  "body_ori",
  "base_lin_vel",
  "base_ang_vel",
  "joint_pos",
  "joint_vel",
  "actions",
)


def test_multi_motion_task_settings_preview_and_terminations(tmp_path: Path) -> None:
  assert TASK in list_tasks()
  cfg = load_env_cfg(TASK)
  play_cfg = load_env_cfg(TASK, play=True)
  agent = load_rl_cfg(TASK)
  assert isinstance(agent, RslRlOnPolicyRunnerCfg)

  # The teacher has no single clip to embed, so its runner writes the reference-free
  # graph. It must stay a base-runner subclass: inheriting the single-motion export
  # would try to embed one clip's tables, which this command refuses.
  runner = load_runner_cls(TASK)
  assert runner is MultiMotionTrackingOnPolicyRunner
  assert issubclass(runner, MjlabOnPolicyRunner)
  assert not issubclass(runner, MotionTrackingOnPolicyRunner)

  # The paper network for both networks, and the base PPO settings elsewhere.
  base_agent = load_rl_cfg(SINGLE_MOTION_TASK)
  assert isinstance(base_agent, RslRlOnPolicyRunnerCfg)
  assert agent.actor.hidden_dims == HIDDEN_DIMS
  assert agent.critic.hidden_dims == HIDDEN_DIMS
  assert agent.actor.activation == agent.critic.activation == "elu"
  assert agent.actor.obs_normalization and agent.critic.obs_normalization
  assert agent.actor.distribution_cfg == {
    "class_name": "GaussianDistribution",
    "init_std": 1.0,
    "std_type": "scalar",
  }
  assert replace(agent.actor, hidden_dims=base_agent.actor.hidden_dims) == (
    base_agent.actor
  )
  assert replace(agent.critic, hidden_dims=base_agent.critic.hidden_dims) == (
    base_agent.critic
  )
  assert (
    replace(agent.algorithm, learning_rate=base_agent.algorithm.learning_rate)
    == base_agent.algorithm
  )
  assert agent.algorithm.learning_rate == 3.0e-4
  assert agent.num_steps_per_env == base_agent.num_steps_per_env == 24
  assert agent.max_iterations == 100_000
  assert cfg.scene.num_envs == 4096

  # The command is the plain-data multi-motion config: one manifest, uniform
  # phases for training, start phases for play, and no standing insert.
  motion = cfg.commands["motion"]
  assert isinstance(motion, MultiMotionPpoCommandCfg)
  assert not hasattr(motion, "library")
  assert (motion.motion_manifest, motion.motion_file) == ("", "")
  assert motion.lookahead_s == 0.2
  assert round(motion.lookahead_s * CONTROL_HZ) == PREVIEW_STEPS
  assert motion.standing_start_prob == 0.0
  assert motion.sampling_mode == "uniform"
  play_motion = play_cfg.commands["motion"]
  assert isinstance(play_motion, MultiMotionPpoCommandCfg)
  assert play_motion.sampling_mode == "start"
  assert play_motion.lookahead_s == motion.lookahead_s

  # Both groups read the same clean future sample; the actor's term order and
  # declared widths give the 226-wide no-state-estimation input.
  actor_terms = cfg.observations["actor"].terms
  critic_terms = cfg.observations["critic"].terms
  for terms in (actor_terms, critic_terms):
    lookahead = terms["motion_lookahead"]
    assert lookahead.func is motion_lookahead
    assert lookahead.params == {"command_name": "motion"}
    assert lookahead.noise is None
    assert (lookahead.delay_min_lag, lookahead.delay_max_lag) == (0, 0)
    assert lookahead.history_length == 0
  assert list(actor_terms) == list(ACTOR_TERM_WIDTHS)
  assert sum(ACTOR_TERM_WIDTHS.values()) == EXPECTED_ACTOR_WIDTH
  assert list(critic_terms) == list(CRITIC_TERM_ORDER)

  # The clip endpoint is a reference-task boundary; physical failures and the
  # time limit keep their own terms.
  assert cfg.terminations["clip_complete"].func is clip_complete
  assert cfg.terminations["clip_complete"].time_out is False
  assert set(cfg.terminations) == {
    "time_out",
    "anchor_pos",
    "anchor_ori",
    "ee_body_pos",
    "clip_complete",
  }
  assert cfg.terminations["time_out"].time_out is True

  # Everything else is the vendor-gain base task.
  base = agibot_x2_flat_tracking_increated_perturbations_vendor_gains_env_cfg()
  base_motion = base.commands["motion"]
  assert isinstance(base_motion, MotionCommandCfg)
  assert cfg.actions == base.actions
  assert cfg.rewards == base.rewards
  assert cfg.events == base.events
  assert cfg.scene.entities == base.scene.entities
  assert motion.anchor_body_name == base_motion.anchor_body_name
  assert motion.body_names == base_motion.body_names

  # The registered config stays plain data: it survives the asdict dump the
  # training entry point writes to params/env.yaml.
  dump_yaml(tmp_path / "env.yaml", asdict(cfg))
  saved = load_saved_yaml(tmp_path / "env.yaml")
  assert saved["commands"]["motion"]["motion_manifest"] == ""
  assert saved["commands"]["motion"]["lookahead_s"] == 0.2
  assert saved["commands"]["motion"]["standing_start_prob"] == 0.0
  assert saved["scene"]["num_envs"] == 4096


def test_single_motion_vendor_gain_task_is_unchanged() -> None:
  base = agibot_x2_flat_tracking_increated_perturbations_vendor_gains_env_cfg()
  motion = base.commands["motion"]
  assert isinstance(motion, MotionCommandCfg)
  assert not isinstance(motion, MultiMotionPpoCommandCfg)
  assert motion.lookahead_s == 0.0
  assert motion.standing_start_prob == 0.3
  assert "motion_lookahead" not in base.observations["critic"].terms
  assert "clip_complete" not in base.terminations

  # Registering the multi-motion task left the existing task as it was.
  assert load_runner_cls(SINGLE_MOTION_TASK) is MotionTrackingOnPolicyRunner
  registered = load_env_cfg(SINGLE_MOTION_TASK)
  assert registered.commands["motion"] == motion
  assert registered.observations["actor"].terms == base.observations["actor"].terms
  assert registered.observations["critic"].terms == base.observations["critic"].terms
  assert registered.terminations == base.terminations
  assert registered.scene.num_envs == base.scene.num_envs == 1


def test_deploy_export_declares_the_preview_the_actor_observes() -> None:
  """The reader compiles a width from metadata, so the export must state the offset.

  The term name is present for a disabled and an enabled preview alike, so the term
  list cannot distinguish them. Without the declared offset the reader resolves the
  preview to width zero and refuses the 226-wide graph.
  """
  motion_cfg = load_env_cfg(TASK).commands["motion"]
  assert motion_cfg.lookahead_s == 0.2

  steps = resolve_lookahead_steps(motion_cfg.lookahead_s, CONTROL_HZ)
  assert steps == PREVIEW_STEPS

  metadata = motion_export_metadata(motion_cfg, steps)
  assert metadata["anchor_body_name"] == "torso_link"
  assert metadata["body_names"] == list(motion_cfg.body_names)
  assert metadata[MOTION_LOOKAHEAD_STEPS_KEY] == steps

  width = sum(ACTOR_TERM_WIDTHS.values())
  assert width == EXPECTED_ACTOR_WIDTH
  # The layout the reader would have compiled without the declaration.
  assert width - ACTOR_TERM_WIDTHS["motion_lookahead"] == 164


def test_the_specialist_layout_still_declares_a_disabled_preview() -> None:
  """Existing 164-wide exports must stay compilable once the key exists."""
  motion_cfg = load_env_cfg(SINGLE_MOTION_TASK).commands["motion"]
  assert resolve_lookahead_steps(motion_cfg.lookahead_s, CONTROL_HZ) == 0
  metadata = motion_export_metadata(motion_cfg, 0)
  assert metadata[MOTION_LOOKAHEAD_STEPS_KEY] == 0
  assert sum(ACTOR_TERM_WIDTHS.values()) - ACTOR_TERM_WIDTHS["motion_lookahead"] == 164
