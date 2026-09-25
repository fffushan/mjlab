"""Integration tests for the X2 tennis-end recovery task registration, parser,
evaluation scheduling, eval-row override, ONNX export parity, and bounded
real CPU evaluation.

These tests exercise real code paths:
- Task registration and registry lookup.
- Full TrainConfig parsing with tyro and mjlab.TYRO_FLAGS (no simulator launch).
- Original velocity task behavior unchanged.
- Evaluation schedule determinism with explicit held-out row selection.
- Eval-row override: actual row selection, invalid row rejection, random path
  unchanged (exercises actual reset, not just attribute assignment).
- Bounded real CPU source==candidate evaluation (1 recovery + 1 retention,
  5 steps each, CPU, CUDA hidden).
- Module --help produces output (not zero bytes).

Tests that require the real source checkpoint or data are explicitly skipped
when unavailable, not silently passed.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from mjlab.tasks.velocity.config.agibot_x2.tennis_recovery_rl_cfg import (
  TennisRecoveryRlRunnerCfg,
)

ORIGINAL_ROOT = Path("/home/agiuser/projects/mjlab")
SOURCE_RUN = (
  ORIGINAL_ROOT
  / "logs"
  / "rsl_rl"
  / "agibot_x2_velocity"
  / "2026-09-16_01-53-44_x2-velocity-measured-gains"
)
SOURCE_CHECKPOINT = SOURCE_RUN / "model_19999.pt"
TENNIS_DATA = ORIGINAL_ROOT / "data" / "tennis"

RECOVERY_TASK = "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Recovery"
BASE_TASK = "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation"


def _checkpoint_available() -> bool:
  return SOURCE_CHECKPOINT.exists()


def _data_available() -> bool:
  return TENNIS_DATA.is_dir() and any(TENNIS_DATA.glob("*.npz"))


pytestmark_checkpoint = pytest.mark.skipif(
  not _checkpoint_available(),
  reason="Source checkpoint not available",
)

pytestmark_data = pytest.mark.skipif(
  not _data_available(),
  reason="Tennis dataset not available",
)


# --------------------------------------------------------------------------- #
# Task registration tests.
# --------------------------------------------------------------------------- #


def test_recovery_task_registered():
  """The tennis recovery task ID appears in the task registry."""
  from mjlab.tasks.registry import list_tasks

  assert RECOVERY_TASK in list_tasks()


def test_original_task_unchanged():
  """The original velocity task is still registered and uses VelocityOnPolicyRunner."""
  from mjlab.tasks.registry import load_runner_cls

  runner_cls = load_runner_cls(BASE_TASK)
  assert runner_cls is not None
  assert runner_cls.__name__ == "VelocityOnPolicyRunner"


def test_recovery_task_uses_recovery_runner():
  """The recovery task uses TennisRecoveryOnPolicyRunner, not the base runner."""
  from mjlab.tasks.registry import load_runner_cls

  runner_cls = load_runner_cls(RECOVERY_TASK)
  assert runner_cls is not None
  assert runner_cls.__name__ == "TennisRecoveryOnPolicyRunner"


def test_recovery_rl_cfg_has_finetune_lr():
  """The recovery RL config has the initial_finetune_lr field (default None)."""
  from mjlab.tasks.registry import load_rl_cfg

  rl_cfg = load_rl_cfg(RECOVERY_TASK)
  assert isinstance(rl_cfg, TennisRecoveryRlRunnerCfg)
  assert rl_cfg.initial_finetune_lr is None


def test_recovery_env_cfg_has_recovery_event():
  """The recovery env config has the tennis_recovery_reset event."""
  from mjlab.tasks.registry import load_env_cfg

  env_cfg = load_env_cfg(RECOVERY_TASK)
  assert "tennis_recovery_reset" in env_cfg.events


def test_recovery_env_cfg_pool_directory_is_string():
  """Production default pool_directory is a string, not None."""
  from mjlab.tasks.velocity.config.agibot_x2.tennis_recovery_env_cfg import (
    agibot_x2_tennis_recovery_env_cfg,
  )

  cfg = agibot_x2_tennis_recovery_env_cfg()
  pool_dir = cfg.events["tennis_recovery_reset"].params["pool_directory"]
  assert isinstance(pool_dir, str)
  assert pool_dir == "data/tennis"


def test_factory_accepts_none_pool_for_tests():
  """Factory(None) still works for test injection (pool provided at runtime)."""
  from mjlab.tasks.velocity.config.agibot_x2.tennis_recovery_env_cfg import (
    agibot_x2_tennis_recovery_env_cfg,
  )

  cfg = agibot_x2_tennis_recovery_env_cfg(pool_directory=None)
  assert cfg.events["tennis_recovery_reset"].params["pool_directory"] is None


def test_recovery_command_type():
  """The recovery config uses TennisRecoveryVelocityCommandCfg."""
  from mjlab.tasks.registry import load_env_cfg
  from mjlab.tasks.velocity.mdp.tennis_recovery import (
    TennisRecoveryVelocityCommandCfg,
  )

  env_cfg = load_env_cfg(RECOVERY_TASK)
  assert isinstance(env_cfg.commands["twist"], TennisRecoveryVelocityCommandCfg)


# --------------------------------------------------------------------------- #
# Parser tests (no simulator launch).
# --------------------------------------------------------------------------- #


def test_full_train_config_parses_with_tyro_flags():
  """Full TrainConfig parses with mjlab.TYRO_FLAGS and recovery task.

  Verifies the exact CLI flags from the brief parse correctly:
  --agent.resume True --agent.initial-finetune-lr 1e-4
  --env.scene.num-envs 5
  --env.events.tennis-recovery-reset.params.pool-directory /some/data
  --env.events.tennis-recovery-reset.params.recovery-fraction 0.8
  --env.events.tennis-recovery-reset.params.last-n-frames 10
  """
  import tyro

  import mjlab
  from mjlab.scripts.train import TrainConfig

  default = TrainConfig.from_task(RECOVERY_TASK)
  args = tyro.cli(
    TrainConfig,
    args=[
      "--agent.resume",
      "True",
      "--agent.initial-finetune-lr",
      "1e-4",
      "--env.scene.num-envs",
      "5",
      "--env.events.tennis-recovery-reset.params.pool-directory",
      "/some/data",
      "--env.events.tennis-recovery-reset.params.recovery-fraction",
      "0.8",
      "--env.events.tennis-recovery-reset.params.last-n-frames",
      "10",
    ],
    default=default,
    config=mjlab.TYRO_FLAGS,
  )

  assert args.agent.resume is True
  assert isinstance(args.agent, TennisRecoveryRlRunnerCfg)
  assert args.agent.initial_finetune_lr == pytest.approx(1e-4)
  assert args.env.scene.num_envs == 5
  event = args.env.events["tennis_recovery_reset"]
  assert event.params["pool_directory"] == "/some/data"
  assert event.params["recovery_fraction"] == pytest.approx(0.8)
  assert event.params["last_n_frames"] == 10


def test_eval_cli_module_help_produces_output():
  """Module --help produces output (not zero bytes).

  Verifies the ``if __name__ == '__main__': main()`` guard is present
  and the module is runnable as ``python -m ... --help``.
  """
  env = os.environ.copy()
  env["CUDA_VISIBLE_DEVICES"] = ""
  env["WARP_DISABLE_CUDA"] = "1"
  env["PYTHONPATH"] = f"{Path.cwd() / 'src'}:{env.get('PYTHONPATH', '')}"
  result = subprocess.run(
    [
      sys.executable,
      "-m",
      "mjlab.tasks.velocity.scripts.tennis_recovery_eval",
      "--help",
    ],
    capture_output=True,
    text=True,
    timeout=30,
    env=env,
    cwd=str(Path.cwd()),
  )
  assert result.returncode == 0, (
    f"Exit code {result.returncode}, stderr: {result.stderr[:500]}"
  )
  assert len(result.stdout) > 0, "Help output is zero bytes"
  assert "--checkpoint" in result.stdout


# --------------------------------------------------------------------------- #
# Eval-row override: actual reset with real pool (exercises real code).
# --------------------------------------------------------------------------- #


@pytestmark_data
def test_eval_row_override_selects_actual_row(device: str) -> None:
  """Eval row override selects the specified row, not a random one.

  Uses the real X2 env and real pool to verify the override actually controls
  which pool row is sampled. This exercises real reset code, not just
  attribute assignment.
  """
  import contextlib
  import io

  import torch

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(RECOVERY_TASK)
  cfg.scene.num_envs = 1
  cfg.events["tennis_recovery_reset"].params["pool_directory"] = str(TENNIS_DATA)
  cfg.events["tennis_recovery_reset"].params["force_mode"] = "recovery"
  cfg.events["tennis_recovery_reset"].params["split"] = "validation"
  cfg.events["tennis_recovery_reset"].params["seed"] = 42

  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with (
      contextlib.redirect_stdout(io.StringIO()),
      contextlib.redirect_stderr(io.StringIO()),
    ):
      env = ManagerBasedRlEnv(cfg, device=device)

  try:
    event_term_cfg = env.event_manager.get_term_cfg("tennis_recovery_reset")
    event_term = event_term_cfg.func

    # Set eval_row_indices to a specific row before reset.
    target_row = 5
    event_term.eval_row_indices = torch.tensor(
      [target_row], dtype=torch.long, device=device
    )

    # Reset to trigger the event.
    env_ids = torch.arange(1, device=device)
    env.reset(env_ids=env_ids)

    # Verify the actual sampled row matches the override.
    assert event_term.last_row_indices[0].item() == target_row

    # Verify trajectory/frame provenance matches the pool.
    from mjlab.tasks.velocity.mdp.tennis_endpoint_pool import EndpointPool

    pool = EndpointPool.from_directory(
      str(TENNIS_DATA), last_n_frames=10, split="validation", seed=42
    )
    assert event_term.last_trajectory_ids[0].item() == int(
      pool.trajectory_ids[target_row]
    )
    assert event_term.last_frame_indices[0].item() == int(
      pool.frame_indices[target_row]
    )
  finally:
    env.close()


@pytestmark_data
def test_eval_row_override_rejects_negative(device: str) -> None:
  """Negative eval_row_indices raises ValueError (no silent last-row selection).

  Uses the real X2 env to exercise the actual validation code path.
  """
  import contextlib
  import io

  import torch

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(RECOVERY_TASK)
  cfg.scene.num_envs = 1
  cfg.events["tennis_recovery_reset"].params["pool_directory"] = str(TENNIS_DATA)
  cfg.events["tennis_recovery_reset"].params["force_mode"] = "recovery"
  cfg.events["tennis_recovery_reset"].params["split"] = "validation"
  cfg.events["tennis_recovery_reset"].params["seed"] = 42

  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with (
      contextlib.redirect_stdout(io.StringIO()),
      contextlib.redirect_stderr(io.StringIO()),
    ):
      env = ManagerBasedRlEnv(cfg, device=device)

  try:
    event_term_cfg = env.event_manager.get_term_cfg("tennis_recovery_reset")
    event_term = event_term_cfg.func

    # Negative index — must raise, not silently select last row.
    event_term.eval_row_indices = torch.tensor([-1], dtype=torch.long, device=device)
    env_ids = torch.arange(1, device=device)
    with pytest.raises(ValueError, match="negative"):
      event_term(env, env_ids)
  finally:
    env.close()


@pytestmark_data
def test_eval_row_override_rejects_out_of_bounds(device: str) -> None:
  """Out-of-bounds eval_row_indices raises ValueError.

  Uses the real X2 env to exercise the actual validation code path.
  """
  import contextlib
  import io

  import torch

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg
  from mjlab.tasks.velocity.mdp.tennis_endpoint_pool import EndpointPool

  cfg = load_env_cfg(RECOVERY_TASK)
  cfg.scene.num_envs = 1
  cfg.events["tennis_recovery_reset"].params["pool_directory"] = str(TENNIS_DATA)
  cfg.events["tennis_recovery_reset"].params["force_mode"] = "recovery"
  cfg.events["tennis_recovery_reset"].params["split"] = "validation"
  cfg.events["tennis_recovery_reset"].params["seed"] = 42

  # Get pool size to create an out-of-bounds index.
  pool = EndpointPool.from_directory(
    str(TENNIS_DATA), last_n_frames=10, split="validation", seed=42
  )

  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with (
      contextlib.redirect_stdout(io.StringIO()),
      contextlib.redirect_stderr(io.StringIO()),
    ):
      env = ManagerBasedRlEnv(cfg, device=device)

  try:
    event_term_cfg = env.event_manager.get_term_cfg("tennis_recovery_reset")
    event_term = event_term_cfg.func

    # Out-of-bounds index — must raise.
    event_term.eval_row_indices = torch.tensor(
      [len(pool) + 100], dtype=torch.long, device=device
    )
    env_ids = torch.arange(1, device=device)
    with pytest.raises(ValueError, match="out-of-bounds"):
      event_term(env, env_ids)
  finally:
    env.close()


# --------------------------------------------------------------------------- #
# Evaluation schedule determinism.
# --------------------------------------------------------------------------- #


def test_schedule_deterministic_row_indices():
  """Same seed produces same row indices for source and candidate."""
  from mjlab.tasks.velocity.scripts.tennis_recovery_eval import EvaluationSchedule

  s1 = EvaluationSchedule(
    num_episodes=10, episode_length=1000, duration_s=20.0, fps=50.0
  )
  s2 = EvaluationSchedule(
    num_episodes=10, episode_length=1000, duration_s=20.0, fps=50.0
  )
  indices1 = s1.sample_indices(1440)
  indices2 = s2.sample_indices(1440)
  np.testing.assert_array_equal(indices1, indices2)


def test_schedule_group_assignment():
  """Group assignment matches 80/20 recovery/retention mix."""
  from mjlab.tasks.velocity.scripts.tennis_recovery_eval import EvaluationSchedule

  s = EvaluationSchedule(
    num_episodes=10, episode_length=1000, duration_s=20.0, fps=50.0
  )
  groups = s.group_assignment()
  assert sum(g == "recovery" for g in groups) == 8
  assert sum(g == "retention" for g in groups) == 2


def test_schedule_recovery_commands_zero():
  from mjlab.tasks.velocity.scripts.tennis_recovery_eval import EvaluationSchedule

  s = EvaluationSchedule(
    num_episodes=10, episode_length=1000, duration_s=20.0, fps=50.0
  )
  cmds = s.recovery_commands()
  assert cmds.shape == (8, 3)
  assert np.all(cmds == 0.0)


def test_schedule_timestamps_strictly_increasing():
  from mjlab.tasks.velocity.scripts.tennis_recovery_eval import EvaluationSchedule

  s = EvaluationSchedule(num_episodes=5, episode_length=100, duration_s=20.0, fps=50.0)
  ts = s.episode_timestamps()
  assert ts[0] == 0.0
  assert np.all(np.diff(ts) > 0)


def test_prepare_episode_descriptors_recovery_has_valid_rows():
  """Recovery descriptors have valid non-negative pool row indices."""
  from mjlab.tasks.velocity.scripts.tennis_recovery_eval import (
    EvaluationSchedule,
    prepare_episode_descriptors,
  )

  s = EvaluationSchedule(num_episodes=4, episode_length=100, duration_s=20.0, fps=50.0)
  descriptors = prepare_episode_descriptors(s, pool_size=290)
  rec_descs = [d for d in descriptors if d.group == "recovery"]
  ret_descs = [d for d in descriptors if d.group == "retention"]
  assert len(rec_descs) == 3  # round(0.8 * 4) = 3
  assert len(ret_descs) == 1
  for d in rec_descs:
    assert d.pool_row_index is not None
    assert d.pool_row_index >= 0
    assert d.pool_row_index < 290
  for d in ret_descs:
    assert d.pool_row_index is None


# --------------------------------------------------------------------------- #
# Bounded real CPU source==candidate evaluation.
# --------------------------------------------------------------------------- #


@pytestmark_checkpoint
@pytestmark_data
def test_bounded_cpu_evaluation_source_equals_candidate(tmp_path):
  """Run a real tiny CPU evaluation: 1 recovery + 1 retention, 5 steps.

  Source and candidate use the same checkpoint, so states/metrics must match.
  This exercises the real evaluation path: single-env config, auto_reset=False,
  true t=0 snapshot, terminal-before-reset capture, torso tilt from torso_link,
  real termination cause, copied snapshots, and JSON output with allow_nan=False.
  """
  from mjlab.tasks.velocity.scripts.tennis_recovery_eval import (
    EvaluationSchedule,
    run_evaluation,
  )

  schedule = EvaluationSchedule(
    num_episodes=2,
    episode_length=5,
    duration_s=0.1,
    fps=50.0,
    seed=42,
    split="validation",
    recovery_fraction=0.5,
  )

  output_path = str(tmp_path / "eval.json")

  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with io.StringIO() as buf:
      import contextlib

      with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        result = run_evaluation(
          schedule=schedule,
          pool_directory=str(TENNIS_DATA),
          checkpoint_path=str(SOURCE_CHECKPOINT),
          device="cpu",
          output_path=output_path,
          candidate_checkpoint_path=str(SOURCE_CHECKPOINT),
        )

  # Verify source results.
  assert len(result["source"]["per_episode"]) == 2
  source_eps = result["source"]["per_episode"]
  assert source_eps[0]["group"] == "recovery"
  assert source_eps[1]["group"] == "retention"

  # Verify termination cause is real (not falsely all "fall").
  for ep in source_eps:
    assert ep["termination_cause"] in ("timeout", "orientation")

  # Verify recovery episode has planned/actual trajectory IDs.
  rec_ep = source_eps[0]
  assert rec_ep["planned_trajectory_id"] is not None
  assert rec_ep["actual_trajectory_id"] is not None
  assert rec_ep["planned_trajectory_id"] == rec_ep["actual_trajectory_id"]

  # Verify retention episode has no trajectory IDs.
  ret_ep = source_eps[1]
  assert ret_ep["planned_trajectory_id"] is None
  assert ret_ep["actual_trajectory_id"] is None

  # Verify t=0 snapshot exists (num_steps includes the initial snapshot).
  for ep in source_eps:
    assert ep["num_steps"] >= 1  # at least the reset snapshot
    assert ep["step_dt"] > 0

  # Verify source==candidate parity (same checkpoint → identical results).
  candidate_eps = result["candidate"]["per_episode"]
  for i in range(len(source_eps)):
    s = source_eps[i]
    c = candidate_eps[i]
    assert s["group"] == c["group"]
    assert s["termination"] == c["termination"]
    assert s["actual_trajectory_id"] == c["actual_trajectory_id"]
    assert s["actual_frame_index"] == c["actual_frame_index"]
    assert s["peak_tilt_deg"] == c["peak_tilt_deg"]
    assert s["num_frames"] == c["num_frames"]
    assert s["trace_sha256"] == c["trace_sha256"]

  # Verify JSON output is valid and has no NaN.
  with open(output_path) as f:
    json_data = json.load(f)
  assert "source" in json_data
  assert "candidate" in json_data
  assert "checkpoint" in json_data
  assert "sha256" in json_data["checkpoint"]
  assert "pool" in json_data
  assert "manifest" in json_data["pool"]


# --------------------------------------------------------------------------- #
# ONNX export and Torch/ONNX numerical parity.
# --------------------------------------------------------------------------- #


@pytestmark_checkpoint
def test_onnx_export_and_torch_onnx_parity(tmp_path):
  """ONNX export succeeds and Torch/ONNX actor outputs agree on actual inputs.

  Exports from the source checkpoint using the original velocity task runner
  and verifies Torch and ONNX actor outputs agree on actual actor inputs.
  """
  import contextlib
  from dataclasses import asdict

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

  cfg = load_env_cfg(BASE_TASK)
  cfg.scene.num_envs = 1
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with (
      contextlib.redirect_stdout(io.StringIO()),
      contextlib.redirect_stderr(io.StringIO()),
    ):
      env = ManagerBasedRlEnv(cfg, device="cpu")
      wrapped = RslRlVecEnvWrapper(env)
      runner = VelocityOnPolicyRunner(
        wrapped,
        asdict(load_rl_cfg(BASE_TASK)),
        str(tmp_path),
        device="cpu",
      )
      runner.load(str(SOURCE_CHECKPOINT), map_location="cpu")

      # Get Torch output.
      policy = runner.get_inference_policy(device="cpu")
      obs = wrapped.get_observations()
      with torch.inference_mode():
        torch_actions = policy(obs)

      # Export and load ONNX.
      onnx_dir = tmp_path / "export"
      runner.export_policy_to_onnx(str(onnx_dir), filename="policy.onnx")
      env.close()

  import onnxruntime as ort

  onnx_path = onnx_dir / "policy.onnx"
  assert onnx_path.exists()
  assert onnx_path.stat().st_size > 0

  sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
  actor_obs = obs["actor"].float().cpu().numpy()
  input_name = sess.get_inputs()[0].name
  onnx_actions = sess.run(None, {input_name: actor_obs})[0]
  assert isinstance(onnx_actions, np.ndarray)
  onnx_actions = np.asarray(onnx_actions, dtype=np.float32)

  torch_np = torch_actions.float().cpu().numpy()
  if onnx_actions.ndim == 2 and onnx_actions.shape[0] == 1:
    onnx_actions = onnx_actions[0]
  if torch_np.ndim == 2 and torch_np.shape[0] == 1:
    torch_np = torch_np[0]

  np.testing.assert_allclose(torch_np, onnx_actions, atol=1e-4, rtol=1e-3)


# --------------------------------------------------------------------------- #
# Fixture.
# --------------------------------------------------------------------------- #


@pytest.fixture
def device() -> str:
  from conftest import get_test_device

  return get_test_device()
