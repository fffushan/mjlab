"""Critical CLI dispatch for the multi-motion PPO teacher.

The environment, vector-env wrapper, and runner are mocked: these tests pin the
training input dispatch, the saved reference record and its resume gate, the
playback singleton, the manual-reset evaluation settings, the evaluation loop's
reset/observation/attribution contract, and the completion/failure/timeout
attribution.  No MuJoCo model is compiled, so live environment evidence stays
with the monitored smoke run.
"""

from __future__ import annotations

import importlib.util
import pathlib
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from tracking_distillation_fixtures import write_reference_clip

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.termination_manager import TerminationManager, TerminationTermCfg
from mjlab.scripts import train as train_script
from mjlab.scripts.play import (
  omit_multi_motion_scrubber,
  select_multi_motion_playback_clip,
)
from mjlab.scripts.train import (
  MOTION_RECORD_FILENAME,
  TrainConfig,
  motion_record_differences,
  run_train,
)
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.distillation.motion_library import (
  MotionClipSpec,
  MotionLibrary,
)
from mjlab.tasks.tracking.distillation.multi_motion import stratified_slot_allocation
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.multi_motion_ppo import (
  MultiMotionPpoCommand,
  MultiMotionPpoCommandCfg,
  clip_complete,
)
from mjlab.utils.os import dump_yaml, load_saved_yaml

TASK = "Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion"
SINGLE_MOTION_TASK = (
  "Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Increated-Perturbations-"
  "Vendor-Gains"
)
JOINT_DIM = 31
CLIP_FRAMES = (12, 20)
PREVIEW_S = 0.2
TEST_ENVS = 8

EVALUATION_SCRIPT = (
  pathlib.Path(__file__).resolve().parents[1]
  / "scripts"
  / "evaluate_tracking_policy.py"
)


def _evaluation_module():
  """Load the local evaluator script, which is not an importable package."""
  spec = importlib.util.spec_from_file_location(
    "evaluate_tracking_policy_under_test", EVALUATION_SCRIPT
  )
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _write_clips(root: Path) -> tuple[Path, ...]:
  """Write two reference clips with distinct content and unequal lengths."""
  clips = []
  for index, frames in enumerate(CLIP_FRAMES):
    clip = root / "data" / f"clip_{index:03d}_agibot_x2_tracking.npz"
    write_reference_clip(
      clip, frames=frames, joint_dim=JOINT_DIM, bodies=4, joint_base=float(index)
    )
    clips.append(clip)
  return tuple(clips)


def _write_manifest(path: Path, clips: tuple[Path, ...]) -> Path:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
    "motions:\n" + "".join(f"  - {clip}\n" for clip in clips),
    encoding="utf-8",
  )
  return path


class _FakeCommand(MultiMotionPpoCommand):
  """Stand-in for the built command, carrying only the record's inputs.

  It subclasses the real command so the type guard in ``run_train`` sees the
  registered command type; no inherited command state is read.
  """

  def __init__(self, clips: tuple[Path, ...]) -> None:
    self.library = MotionLibrary.from_clips(
      [MotionClipSpec(teacher_id=clip.stem, motion_file=clip) for clip in clips],
      device="cpu",
    )
    self.slot_allocation = stratified_slot_allocation(
      [1.0] * len(clips), TEST_ENVS, teacher_ids=[clip.stem for clip in clips]
    )
    self.cfg = MotionCommandCfg(
      motion_file="",
      anchor_body_name="torso_link",
      body_names=("pelvis",),
      entity_name="robot",
      lookahead_s=PREVIEW_S,
      resampling_time_range=(1e9, 1e9),
    )


class _FakeEnv:
  def __init__(self, command: _FakeCommand) -> None:
    self.command_manager = SimpleNamespace(get_term=lambda name: command)
    self.closed = False

  def close(self) -> None:
    self.closed = True


class _FakeRunner:
  def __init__(self, env, train_cfg, log_dir, device, **kwargs) -> None:
    self.kwargs = kwargs
    self.loaded: str | None = None

  def add_git_repo_to_log(self, path) -> None:
    return None

  def load(self, path) -> None:
    self.loaded = path

  def learn(self, num_learning_iterations, init_at_random_ep_len) -> None:
    return None


@pytest.fixture
def fake_training(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
  """Replace the environment, wrapper, and runner of ``run_train``."""
  clips = _write_clips(tmp_path)
  command = _FakeCommand(clips)
  env = _FakeEnv(command)
  runners: list[_FakeRunner] = []

  def runner_cls(env_, train_cfg, log_dir, device, **kwargs) -> _FakeRunner:
    runner = _FakeRunner(env_, train_cfg, log_dir, device, **kwargs)
    runners.append(runner)
    return runner

  monkeypatch.setattr(train_script, "ManagerBasedRlEnv", lambda *args, **kwargs: env)
  monkeypatch.setattr(
    train_script, "RslRlVecEnvWrapper", lambda env_, clip_actions=None: env_
  )
  monkeypatch.setattr(train_script, "load_runner_cls", lambda task_id: runner_cls)
  monkeypatch.setattr(train_script, "add_wandb_tags", lambda tags: None)
  monkeypatch.setattr(train_script, "configure_torch_backends", lambda: None)
  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
  return SimpleNamespace(clips=clips, command=command, env=env, runners=runners)


def test_train_dispatch_saves_reference_record_and_rejects_changed_resume(
  tmp_path: Path, fake_training
) -> None:
  clips = fake_training.clips
  manifest = _write_manifest(tmp_path / "configs" / "pilot.yaml", clips)
  cfg = TrainConfig.from_task(TASK)
  motion_cmd = cfg.env.commands["motion"]
  assert isinstance(motion_cmd, MultiMotionPpoCommandCfg)
  motion_cmd.motion_manifest = str(manifest)

  log_dir = tmp_path / "logs" / "multi" / "2026-01-01_00-00-00"
  run_train(TASK, cfg, log_dir)

  # A manifest alone is a valid input: no single-file gate, no W&B motion
  # artifact, and no registry_name keyword for the standard runner.
  assert fake_training.runners[0].kwargs == {}
  record = load_saved_yaml(log_dir / "params" / MOTION_RECORD_FILENAME)
  assert record["source"] == str(manifest)
  assert record["preview_s"] == PREVIEW_S
  assert [entry["file"] for entry in record["clips"]] == [
    str(clip.resolve()) for clip in clips
  ]
  assert [entry["frames"] for entry in record["clips"]] == list(CLIP_FRAMES)
  assert record["mapping_digest"] == fake_training.command.library.mapping_digest()
  assert sum(record["allocation"]["counts"]) == TEST_ENVS

  # The single-motion path keeps its own input gate and registry keyword.
  single = TrainConfig.from_task(SINGLE_MOTION_TASK)
  single_motion = single.env.commands["motion"]
  assert isinstance(single_motion, MotionCommandCfg)
  single_motion.motion_file = str(clips[0])
  run_train(SINGLE_MOTION_TASK, single, tmp_path / "logs" / "single" / "run")
  assert fake_training.runners[1].kwargs == {"registry_name": None}

  # A checkpoint that resolved another preview is refused, and a changed clip
  # list is reported as a motion-list difference.
  assert motion_record_differences(record, record) == []
  assert motion_record_differences(dict(record, mapping_digest="other"), record) != []

  previous = log_dir.parent / "previous"
  (previous / "params").mkdir(parents=True)
  (previous / "model_0.pt").write_bytes(b"")
  dump_yaml(previous / "params" / MOTION_RECORD_FILENAME, dict(record, preview_s=0.5))
  cfg.agent.resume = True
  cfg.agent.load_run = "previous"
  cfg.agent.load_checkpoint = "model_0.pt"
  with pytest.raises(ValueError, match="preview_s"):
    run_train(TASK, cfg, tmp_path / "logs" / "multi" / "2026-01-02_00-00-00")


def test_play_singleton_and_manual_reset_evaluation_settings(tmp_path: Path) -> None:
  clips = _write_clips(tmp_path)

  # Playback plays one explicit clip: the training/default manifest is cleared
  # and the file becomes the reference source.
  play_cfg = load_env_cfg(TASK, play=True)
  play_motion = play_cfg.commands["motion"]
  assert isinstance(play_motion, MultiMotionPpoCommandCfg)
  play_motion.motion_manifest = "configs/pilot.yaml"
  assert select_multi_motion_playback_clip(play_motion, str(clips[0])) == clips[0]
  assert play_motion.motion_manifest == ""
  assert play_motion.motion_file == str(clips[0])
  assert play_motion.sampling_mode == "start"
  with pytest.raises(ValueError, match="--motion-file"):
    select_multi_motion_playback_clip(play_motion, None)
  with pytest.raises(FileNotFoundError):
    select_multi_motion_playback_clip(play_motion, str(tmp_path / "missing.npz"))

  # The viewer's single-clip scrubber cannot be built for this command, so the
  # play entry point replaces that one panel builder and keeps the rest.
  class _ScrubberManager:
    def create_gui(self, *_args, **_kwargs):
      raise AssertionError("the multi-motion scrubber must not be built")

  class _ScrubberEnv:
    def __init__(self) -> None:
      self.command_manager = _ScrubberManager()

  scrubber_env = _ScrubberEnv()
  omit_multi_motion_scrubber(cast(ManagerBasedRlEnv, scrubber_env))
  assert scrubber_env.command_manager.create_gui(None, lambda: 0) is None

  # The evaluation branch restores the trained preview, plays one clip from its
  # first frame, and reads terminal metrics before any reset.
  saved_cfg = load_env_cfg(TASK, play=False)
  saved_cfg.commands["motion"].lookahead_s = PREVIEW_S
  checkpoint = tmp_path / "run" / "model_100.pt"
  checkpoint.parent.mkdir(parents=True)
  checkpoint.write_bytes(b"")
  dump_yaml(checkpoint.parent / "params" / "env.yaml", asdict(saved_cfg))

  eval_cfg = load_env_cfg(TASK, play=False)
  eval_motion = eval_cfg.commands["motion"]
  assert isinstance(eval_motion, MultiMotionPpoCommandCfg)
  eval_motion.lookahead_s = 0.0
  module = _evaluation_module()
  settings = module.configure_multi_motion_eval_env(
    eval_cfg, clips[0], checkpoint, num_envs=4, episode_length_s=0.24, seed=7
  )

  assert eval_cfg.auto_reset is False
  assert eval_cfg.episode_length_s == 0.24
  assert eval_cfg.scene.num_envs == 4
  assert eval_cfg.seed == 7
  assert eval_cfg.observations["actor"].enable_corruption is True
  assert eval_motion.motion_manifest == ""
  assert eval_motion.motion_file == str(clips[0])
  assert eval_motion.sampling_mode == "start"
  assert eval_motion.lookahead_s == PREVIEW_S
  assert settings == {
    "motion_file": str(clips[0]),
    "num_envs": 4,
    "episode_length_s": 0.24,
    "seed": 7,
    "lookahead_s": PREVIEW_S,
    "sampling_mode": "start",
    "auto_reset": False,
    "actor_corruption": True,
    "push_robot_event": False,
  }


def test_outcome_attribution_prefers_failure_over_completion() -> None:
  module = _evaluation_module()
  complete = torch.tensor([True, False, False, True, False])
  failed = torch.tensor([True, True, False, False, False])
  timed_out = torch.tensor([False, False, True, True, False])
  dones = torch.tensor([True, True, True, True, False])

  outcomes = module.classify_finished_rows(dones, complete, failed, timed_out)

  # A simultaneous failure wins over completion, completion wins over a
  # simultaneous time limit, an early time limit is not a success, and a row
  # that is still running has no outcome.
  assert outcomes == ["failure", "failure", "timeout", "complete", None]

  with pytest.raises(RuntimeError, match="without a completion"):
    module.classify_finished_rows(
      torch.tensor([True]),
      torch.tensor([False]),
      torch.tensor([False]),
      torch.tensor([False]),
    )


# Scripted multi-motion evaluation scenario: three rows, three steps.

EVAL_STEP_DT = 0.02
EVAL_ROWS = 3
EVAL_CLIP_FRAMES = 3  # Last frame index of the scripted clip.
TRACKING_POSE_ERROR = 0.5
TRACKING_VEL_ERROR = 2.0
RESET_POISON = 100.0
START_MARKER = 3.0
STEP_MARKER = 1.0
RESET_MARKER = 2.0


class _ScriptedCommand(MultiMotionPpoCommand):
  """Scripted command state; a subclass so the real clip_complete term applies.

  The reference, robot, and clip-length quantities the loop reads are
  properties of the real command, so this stand-in replaces them with scripted
  tensors: the reference error stays constant, and a reset can write a pose that
  would show up in any metric read after the reset.
  """

  def __init__(self) -> None:
    # ``num_envs`` and ``device`` are read-only properties of the command.
    self._env = cast(Any, SimpleNamespace(num_envs=EVAL_ROWS, device="cpu"))
    self.time_steps = torch.zeros(EVAL_ROWS, dtype=torch.long)
    self.body_pos_relative_w = torch.zeros(EVAL_ROWS, 1, 3)
    self.reference_vel = torch.zeros(EVAL_ROWS, 1)
    self.robot_pose = torch.zeros(EVAL_ROWS, 1, 3)
    self.robot_vel = torch.zeros(EVAL_ROWS, 1)
    self.clip_lengths = torch.full((EVAL_ROWS,), EVAL_CLIP_FRAMES, dtype=torch.long)
    self.cfg = MotionCommandCfg(
      motion_file="",
      anchor_body_name="torso_link",
      body_names=("pelvis",),
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
    )

  @property
  def row_lengths(self) -> torch.Tensor:
    return self.clip_lengths

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.reference_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot_pose

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot_vel


def _scripted_failure(env) -> torch.Tensor:
  """Physical-failure term: the scripted row mask of the current step."""
  return env.failure_script[env.step_index].clone()


def _scripted_timeout(env) -> torch.Tensor:
  """Time-limit term: never fires in this scenario."""
  return env.timeout_script[env.step_index].clone()


class _ScriptedEvaluationEnv:
  """Three-step environment with the real step, termination, and reset contract.

  It mirrors the parts of ``ManagerBasedRlEnv`` the evaluation loop depends on:
  terminations are computed before the command frames advance, every done row
  must be reset before the next step while auto-reset is off, and a reset
  replaces the observation buffer.  A reset also writes a poison robot state, so
  any metric read after a reset reports the poison instead of the tracking
  error.
  """

  def __init__(self) -> None:
    self.num_envs = EVAL_ROWS
    self.device = "cpu"
    self.step_dt = EVAL_STEP_DT
    self.command = _ScriptedCommand()
    self.unwrapped = self
    self.command_manager = SimpleNamespace(get_term=lambda name: self.command)
    # Row 0 fails on the first two steps, row 1 fails together with its clip
    # completion on the last step, and row 2 completes cleanly on the last step.
    self.failure_script = (
      torch.tensor([True, False, False]),
      torch.tensor([True, False, False]),
      torch.tensor([False, True, False]),
    )
    self.timeout_script = (torch.zeros(EVAL_ROWS, dtype=torch.bool),) * 3
    self.termination_manager = TerminationManager(
      {
        "time_out": TerminationTermCfg(func=_scripted_timeout, time_out=True),
        "anchor_pos": TerminationTermCfg(func=_scripted_failure, time_out=False),
        "clip_complete": TerminationTermCfg(func=clip_complete, time_out=False),
      },
      cast(Any, self),
    )
    self.step_index = 0
    self.pending = torch.zeros(EVAL_ROWS, dtype=torch.bool)
    self.resets: list[list[int]] = []
    self.obs = {"marker": torch.full((EVAL_ROWS,), START_MARKER)}

  def get_observations(self) -> dict[str, torch.Tensor]:
    return self.obs

  def reset(self, *, env_ids: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict]:
    self.resets.append([int(index) for index in env_ids.tolist()])
    self.pending[env_ids] = False
    self.command.time_steps[env_ids] = 0
    self.command.robot_pose[env_ids] = RESET_POISON
    self.command.robot_vel[env_ids] = RESET_POISON
    self.obs = {"marker": torch.full((EVAL_ROWS,), RESET_MARKER)}
    return self.obs, {}

  def step(
    self, actions: torch.Tensor
  ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
    del actions
    if bool(self.pending.any()):
      raise RuntimeError(
        "a done row was stepped without the manual reset the environment requires"
      )
    # Physics result, then terminations, then the command frame advance, which
    # is the order ManagerBasedRlEnv.step uses.  Only the x offset is written, so
    # the reference/robot position error is exactly TRACKING_POSE_ERROR.
    self.command.robot_pose[:, :, 0] = TRACKING_POSE_ERROR
    self.command.robot_vel[:] = TRACKING_VEL_ERROR
    dones = self.termination_manager.compute()
    self.command.time_steps = torch.minimum(
      self.command.time_steps + 1, self.command.row_lengths - 1
    )
    self.pending |= dones
    self.step_index += 1
    self.obs = {"marker": torch.full((EVAL_ROWS,), STEP_MARKER)}
    return self.obs, torch.zeros(EVAL_ROWS), dones.to(torch.long), {}


def test_evaluation_loop_resets_done_rows_and_counts_first_episodes_only() -> None:
  module = _evaluation_module()
  env = _ScriptedEvaluationEnv()
  seen: list[float] = []

  def policy(obs) -> torch.Tensor:
    seen.append(float(obs["marker"][0]))
    return torch.zeros(EVAL_ROWS, 1)

  outcome = module.run_multi_motion_evaluation(env, env.command, policy)

  # Row 0 fails after one control period, its repeated failure while inactive is
  # never counted again, and the final-frame failure of row 1 beats the clip
  # completion it receives on the same step. Row 2 completes cleanly.
  assert outcome["completed"] == 1
  assert outcome["failed"] == 2
  assert outcome["timed_out"] == 0
  assert outcome["steps_taken"] == 3
  assert outcome["failure_time_s"]["min"] == pytest.approx(EVAL_STEP_DT)
  assert outcome["failure_time_s"]["max"] == pytest.approx(3 * EVAL_STEP_DT)
  assert outcome["failure_phase_fraction"]["min"] == pytest.approx(0.0)
  assert outcome["failure_phase_fraction"]["max"] == pytest.approx(1.0)

  # Terminal metrics are read before the reset writes its poison pose, and the
  # inactive rows keep contributing nothing.
  assert outcome["root_relative_mpkpe_m"] == pytest.approx(TRACKING_POSE_ERROR)
  assert outcome["joint_velocity_error"] == pytest.approx(TRACKING_VEL_ERROR)

  # Every row the environment marked done is reset, including the inactive row
  # that failed again, before the next step.
  assert env.resets == [[0], [0], [1, 2]]

  # The policy reads the observation buffer the reset produced, not the stale
  # one from the step that ended the previous episode.
  assert seen == [START_MARKER, RESET_MARKER, RESET_MARKER]
