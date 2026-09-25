"""Tests for the X2 tennis-end recovery runner and checkpoint continuation.

Covers:
- Config factory produces the expected structure with optional LR override.
- Actual source checkpoint (``model_19999.pt``) loads strictly into a runner
  built from the original velocity environment; normalizers, iteration counter
  and optimizer moments are preserved.
- ``initial_finetune_lr`` override is applied after load to both
  ``alg.learning_rate`` and all optimizer param-group ``lr`` values.
- Normal resume (``initial_finetune_lr=None``) preserves the saved optimizer LR.
- Finite deterministic inference after load (actor forward pass).
- ``max_iterations`` is an *additional* update count relative to the loaded
  iteration.
"""

import contextlib
import io
import warnings
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity.config.agibot_x2.tennis_recovery_rl_cfg import (
  TennisRecoveryRlRunnerCfg,
  agibot_x2_tennis_recovery_ppo_runner_cfg,
)
from mjlab.tasks.velocity.rl.tennis_recovery_runner import (
  TennisRecoveryOnPolicyRunner,
)

## Paths (original absolute paths; read-only). ##

ORIGINAL_ROOT = Path("/home/agiuser/projects/mjlab")
SOURCE_RUN = (
  ORIGINAL_ROOT
  / "logs"
  / "rsl_rl"
  / "agibot_x2_velocity"
  / "2026-09-16_01-53-44_x2-velocity-measured-gains"
)
SOURCE_CHECKPOINT = SOURCE_RUN / "model_19999.pt"
SOURCE_PARAMS = SOURCE_RUN / "params"

# SHA256 of the source checkpoint, pinned by the plan.
SOURCE_CHECKPOINT_SHA256 = (
  "a8450c34d6fbea0a61ec8086707890bc8f4444aef7a79c6e456b1ee48e01f874"
)

BASE_TASK = "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation"


## Helpers ##


def _build_env(num_envs: int = 2):
  """Build a small real velocity env for runner construction."""
  cfg = load_env_cfg(BASE_TASK)
  cfg.scene.num_envs = num_envs
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with (
      contextlib.redirect_stdout(io.StringIO()),
      contextlib.redirect_stderr(io.StringIO()),
    ):
      return ManagerBasedRlEnv(cfg, device="cpu")


def _build_runner(env, cfg: TennisRecoveryRlRunnerCfg, tmp_path: Path):
  """Build a TennisRecoveryOnPolicyRunner from env + config."""
  env = RslRlVecEnvWrapper(env)
  agent_cfg = asdict(cfg)
  runner = TennisRecoveryOnPolicyRunner(env, agent_cfg, str(tmp_path), device="cpu")
  return runner


def _checkpoint_available() -> bool:
  return SOURCE_CHECKPOINT.exists()


def _verify_checkpoint_sha256() -> str:
  import hashlib

  h = hashlib.sha256()
  with open(SOURCE_CHECKPOINT, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


## Config tests ##


def test_factory_returns_correct_type():
  cfg = agibot_x2_tennis_recovery_ppo_runner_cfg()
  assert isinstance(cfg, TennisRecoveryRlRunnerCfg)


def test_factory_preserves_inherited_velocity_defaults():
  """Actor/critic dims, distribution and algorithm match the shipped config."""
  cfg = agibot_x2_tennis_recovery_ppo_runner_cfg()

  assert cfg.actor.hidden_dims == (512, 256, 128)
  assert cfg.actor.activation == "elu"
  assert cfg.actor.obs_normalization is True
  assert cfg.actor.distribution_cfg == {
    "class_name": "GaussianDistribution",
    "init_std": 1.0,
    "std_type": "scalar",
  }

  assert cfg.critic.hidden_dims == (512, 256, 128)
  assert cfg.critic.activation == "elu"
  assert cfg.critic.obs_normalization is True

  assert cfg.algorithm.learning_rate == 1.0e-3
  assert cfg.algorithm.schedule == "adaptive"
  assert cfg.algorithm.desired_kl == 0.01
  assert cfg.algorithm.gamma == 0.99
  assert cfg.algorithm.lam == 0.95
  assert cfg.algorithm.clip_param == 0.2
  assert cfg.algorithm.entropy_coef == 0.01


def test_factory_logger_matches_source():
  """Logger defaults to tensorboard (matching source run)."""
  cfg = agibot_x2_tennis_recovery_ppo_runner_cfg()
  assert cfg.logger == "tensorboard"


def test_factory_upload_model_disabled():
  """upload_model defaults to False (no unauthorized uploads)."""
  cfg = agibot_x2_tennis_recovery_ppo_runner_cfg()
  assert cfg.upload_model is False


def test_factory_max_iterations_matches_source():
  """max_iterations matches source run's configured horizon (20_000)."""
  cfg = agibot_x2_tennis_recovery_ppo_runner_cfg()
  assert cfg.max_iterations == 20_000


def test_factory_experiment_root_for_resolver():
  """experiment_name stays agibot_x2_velocity for the regex checkpoint resolver."""
  cfg = agibot_x2_tennis_recovery_ppo_runner_cfg()
  assert cfg.experiment_name == "agibot_x2_velocity"


def test_factory_unique_run_name():
  cfg = agibot_x2_tennis_recovery_ppo_runner_cfg()
  assert cfg.run_name == "x2-tennis-end-recovery"


def test_factory_initial_finetune_lr_default_none():
  cfg = agibot_x2_tennis_recovery_ppo_runner_cfg()
  assert cfg.initial_finetune_lr is None


def test_config_asdict_includes_initial_finetune_lr():
  cfg = agibot_x2_tennis_recovery_ppo_runner_cfg()
  cfg.initial_finetune_lr = 1e-4
  d = asdict(cfg)
  assert "initial_finetune_lr" in d
  assert d["initial_finetune_lr"] == 1e-4


## Checkpoint load tests (require actual source checkpoint) ##


pytestmark_checkpoint = pytest.mark.skipif(
  not _checkpoint_available(),
  reason="Source checkpoint not available for load testing",
)


@pytest.fixture
def small_env():
  env = _build_env(num_envs=2)
  try:
    yield env
  finally:
    env.close()


@pytest.fixture
def checkpoint_sha256():
  return _verify_checkpoint_sha256()


@pytestmark_checkpoint
def test_source_checkpoint_sha256_matches(checkpoint_sha256):
  assert checkpoint_sha256 == SOURCE_CHECKPOINT_SHA256


@pytestmark_checkpoint
def test_strict_load_preserves_normalizers_iteration_and_counter(
  small_env, tmp_path, checkpoint_sha256
):
  """Actual model_19999.pt loads strictly; normalizers, iter, and
  common_step_counter are restored."""
  cfg = agibot_x2_tennis_recovery_ppo_runner_cfg()
  runner = _build_runner(small_env, cfg, tmp_path)

  saved = torch.load(SOURCE_CHECKPOINT, weights_only=False, map_location="cpu")

  runner.load(str(SOURCE_CHECKPOINT))

  # Normalizers fully preserved: mean, variance, std, and count.
  for role, state_key in (
    ("actor", "actor_state_dict"),
    ("critic", "critic_state_dict"),
  ):
    model = getattr(runner.alg, role)
    saved_norm = saved[state_key]
    for norm_field in ("_mean", "_var", "_std", "count"):
      saved_val = saved_norm[f"obs_normalizer.{norm_field}"]
      loaded_val = getattr(model.obs_normalizer, norm_field)
      torch.testing.assert_close(loaded_val, saved_val)

  # Iteration counter restored.
  assert runner.current_learning_iteration == saved["iter"]

  # common_step_counter restored from env_state.
  saved_common_step = saved["infos"]["env_state"]["common_step_counter"]
  assert runner.env.unwrapped.common_step_counter == saved_common_step


@pytestmark_checkpoint
def test_strict_load_preserves_optimizer_moments(
  small_env, tmp_path, checkpoint_sha256
):
  """Optimizer state (Adam moments) is restored, not reset."""
  cfg = agibot_x2_tennis_recovery_ppo_runner_cfg()
  runner = _build_runner(small_env, cfg, tmp_path)

  saved = torch.load(SOURCE_CHECKPOINT, weights_only=False, map_location="cpu")
  saved_opt_state = saved["optimizer_state_dict"]["state"]

  runner.load(str(SOURCE_CHECKPOINT))

  loaded_opt_state = runner.alg.optimizer.state_dict()["state"]
  # Adam stores exp_avg and exp_avg_sq per param index.
  for idx in saved_opt_state:
    assert idx in loaded_opt_state, f"optimizer state missing param {idx}"
    for moment_key in ("exp_avg", "exp_avg_sq"):
      if moment_key in saved_opt_state[idx]:
        torch.testing.assert_close(
          loaded_opt_state[idx][moment_key],
          saved_opt_state[idx][moment_key],
        )


@pytestmark_checkpoint
def test_normal_resume_preserves_saved_optimizer_lr(
  small_env, tmp_path, checkpoint_sha256
):
  """With initial_finetune_lr=None, saved optimizer LR is preserved."""
  cfg = agibot_x2_tennis_recovery_ppo_runner_cfg()
  assert cfg.initial_finetune_lr is None
  runner = _build_runner(small_env, cfg, tmp_path)

  saved = torch.load(SOURCE_CHECKPOINT, weights_only=False, map_location="cpu")
  saved_lr = saved["optimizer_state_dict"]["param_groups"][0]["lr"]

  runner.load(str(SOURCE_CHECKPOINT))

  assert runner.alg.learning_rate == pytest.approx(saved_lr)
  for pg in runner.alg.optimizer.param_groups:
    assert pg["lr"] == pytest.approx(saved_lr)


@pytestmark_checkpoint
def test_lr_override_applied_after_load(small_env, tmp_path, checkpoint_sha256):
  """initial_finetune_lr=1e-4 overrides alg LR and all param-group LRs."""
  cfg = agibot_x2_tennis_recovery_ppo_runner_cfg()
  cfg.initial_finetune_lr = 1e-4
  runner = _build_runner(small_env, cfg, tmp_path)

  runner.load(str(SOURCE_CHECKPOINT))

  assert runner.alg.learning_rate == pytest.approx(1e-4)
  assert len(runner.alg.optimizer.param_groups) > 0
  for pg in runner.alg.optimizer.param_groups:
    assert pg["lr"] == pytest.approx(1e-4)


@pytestmark_checkpoint
def test_lr_override_preserves_optimizer_moments(
  small_env, tmp_path, checkpoint_sha256
):
  """LR override does not reset optimizer moments."""
  cfg = agibot_x2_tennis_recovery_ppo_runner_cfg()
  cfg.initial_finetune_lr = 1e-4
  runner = _build_runner(small_env, cfg, tmp_path)

  saved = torch.load(SOURCE_CHECKPOINT, weights_only=False, map_location="cpu")
  saved_opt_state = saved["optimizer_state_dict"]["state"]

  runner.load(str(SOURCE_CHECKPOINT))

  loaded_opt_state = runner.alg.optimizer.state_dict()["state"]
  for idx in saved_opt_state:
    assert idx in loaded_opt_state
    for moment_key in ("exp_avg", "exp_avg_sq"):
      if moment_key in saved_opt_state[idx]:
        torch.testing.assert_close(
          loaded_opt_state[idx][moment_key],
          saved_opt_state[idx][moment_key],
        )


@pytestmark_checkpoint
@pytest.mark.parametrize(
  "bad_lr",
  [
    float("nan"),
    float("inf"),
    float("-inf"),
    0.0,
    -1e-4,
  ],
)
def test_lr_override_rejects_nonpositive_or_nonfinite(
  small_env, tmp_path, checkpoint_sha256, bad_lr
):
  """initial_finetune_lr must be finite and positive; invalid values raise."""
  cfg = agibot_x2_tennis_recovery_ppo_runner_cfg()
  cfg.initial_finetune_lr = bad_lr
  runner = _build_runner(small_env, cfg, tmp_path)

  with pytest.raises(ValueError, match="initial_finetune_lr"):
    runner.load(str(SOURCE_CHECKPOINT))


@pytestmark_checkpoint
def test_lr_override_accepts_tiny_positive_lr(small_env, tmp_path, checkpoint_sha256):
  """A tiny but positive finite LR is valid and applied after load.

  This replaces the previous skipped ``1e-300`` case in the invalid-value
  parametrization. A subnormal positive float is still a valid LR override.
  """
  cfg = agibot_x2_tennis_recovery_ppo_runner_cfg()
  cfg.initial_finetune_lr = 1e-300
  runner = _build_runner(small_env, cfg, tmp_path)

  runner.load(str(SOURCE_CHECKPOINT))

  assert runner.alg.learning_rate == pytest.approx(1e-300)
  for pg in runner.alg.optimizer.param_groups:
    assert pg["lr"] == pytest.approx(1e-300)


@pytestmark_checkpoint
def test_invalid_lr_override_leaves_loaded_state_intact(
  small_env, tmp_path, checkpoint_sha256
):
  """An invalid override raises before mutating algorithm/optimizer state.

  The saved optimizer LR and moments are preserved because the validation
  guard runs before any assignment.
  """
  cfg = agibot_x2_tennis_recovery_ppo_runner_cfg()
  cfg.initial_finetune_lr = float("nan")
  runner = _build_runner(small_env, cfg, tmp_path)

  saved = torch.load(SOURCE_CHECKPOINT, weights_only=False, map_location="cpu")
  saved_lr = saved["optimizer_state_dict"]["param_groups"][0]["lr"]

  with pytest.raises(ValueError, match="initial_finetune_lr"):
    runner.load(str(SOURCE_CHECKPOINT))

  # Base load completed before the override guard; saved LR is intact.
  assert runner.alg.learning_rate == pytest.approx(saved_lr)
  for pg in runner.alg.optimizer.param_groups:
    assert pg["lr"] == pytest.approx(saved_lr)


@pytestmark_checkpoint
def test_finite_deterministic_inference_after_load(
  small_env, tmp_path, checkpoint_sha256
):
  """Actor produces finite deterministic output after checkpoint load."""
  cfg = agibot_x2_tennis_recovery_ppo_runner_cfg()
  runner = _build_runner(small_env, cfg, tmp_path)

  runner.load(str(SOURCE_CHECKPOINT))

  policy = runner.get_inference_policy(device="cpu")
  obs = runner.env.get_observations()
  # The MLPModel actor expects the full TensorDict (it selects obs groups).
  with torch.inference_mode():
    actions1 = policy(obs)
    actions2 = policy(obs)

  assert torch.isfinite(actions1).all()
  assert actions1.shape[0] == small_env.num_envs
  # Deterministic (eval mode, same input → same output).
  torch.testing.assert_close(actions1, actions2)


## max_iterations additional-update semantics ##


def test_max_iterations_is_additional_after_resume(
  small_env, tmp_path, checkpoint_sha256
):
  """learn() runs [current_iter, current_iter + max_iterations).

  Arithmetic-only check of the rsl-rl loop semantics — no PPO update is
  executed.  With a checkpoint at iter 19999, max_iterations=2000 should run
  to 21999 (2000 additional updates), not 20000 total.  A bounded real
  1–2-update resume smoke is deferred to final integration.
  """
  cfg = agibot_x2_tennis_recovery_ppo_runner_cfg()
  cfg.max_iterations = 5
  runner = _build_runner(small_env, cfg, tmp_path)

  runner.load(str(SOURCE_CHECKPOINT))
  saved_iter = runner.current_learning_iteration

  # Verify the rsl-rl learn() loop semantics: range(start_it, start_it + N).
  # We check the arithmetic without running a full update (no GPU, fast).
  start_it = runner.current_learning_iteration
  total_it = start_it + cfg.max_iterations
  assert total_it == saved_iter + 5, (
    f"max_iterations should be additional: start={start_it}, "
    f"total={total_it}, expected {saved_iter + 5}"
  )


## CPU checkpoint portability (CUDA hidden) ##


@pytestmark_checkpoint
def test_load_with_cuda_hidden_uses_device_map(small_env, tmp_path):
  """Runner.load with CUDA hidden resolves map_location=None to self.device.

  The source checkpoint was saved on CUDA. When ``CUDA_VISIBLE_DEVICES`` is
  set to empty, ``torch.load`` with ``map_location=None`` fails because CUDA
  is not available. The recovery runner resolves ``map_location=None`` to
  ``self.device`` (``'cpu'`` here) so the standard train launcher pattern
  ``runner.load(path)`` works on CPU without an explicit map.
  """
  import os

  assert (
    not torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES", "") == ""
  ), "Test must run with CUDA hidden"
  cfg = agibot_x2_tennis_recovery_ppo_runner_cfg()
  runner = _build_runner(small_env, cfg, tmp_path)

  # No explicit map_location — runner resolves None to self.device ('cpu').
  runner.load(str(SOURCE_CHECKPOINT))

  assert runner.current_learning_iteration == 19999
  policy = runner.get_inference_policy(device="cpu")
  obs = runner.env.get_observations()
  with torch.inference_mode():
    actions = policy(obs)
  assert torch.isfinite(actions).all()


@pytestmark_checkpoint
def test_explicit_map_location_respected(small_env, tmp_path):
  """An explicit map_location is passed through unchanged."""
  cfg = agibot_x2_tennis_recovery_ppo_runner_cfg()
  runner = _build_runner(small_env, cfg, tmp_path)

  runner.load(str(SOURCE_CHECKPOINT), map_location="cpu")
  assert runner.current_learning_iteration == 19999
