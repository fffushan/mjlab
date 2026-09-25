"""Final regression gates for real recovery evaluation and deployment export."""

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import get_base_metadata, list_to_csv_str
from mjlab.tasks import registry
from mjlab.tasks.velocity.mdp.tennis_endpoint_pool import EndpointPool
from mjlab.tasks.velocity.mdp.tennis_recovery import TennisRecoveryResetEvent
from mjlab.tasks.velocity.scripts.tennis_recovery_eval import (
  EpisodeDescriptor,
  EvaluationSchedule,
  _run_single_episode,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("MJLAB_TENNIS_DATA_DIR", str(ROOT / "data/tennis")))
CHECKPOINT = Path(
  os.environ.get(
    "MJLAB_SOURCE_CHECKPOINT",
    str(
      ROOT
      / "logs/rsl_rl/agibot_x2_velocity"
      / "2026-09-16_01-53-44_x2-velocity-measured-gains/model_19999.pt"
    ),
  )
)
BASE = "Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation"
RECOVERY = BASE + "-Tennis-Recovery"
requires_assets = pytest.mark.skipif(
  not CHECKPOINT.is_file() or not DATA.is_dir(),
  reason="Local checkpoint/tennis data unavailable; set explicit asset environment paths",
)


def test_eval_rejects_requested_cadence_different_from_task():
  schedule = EvaluationSchedule(1, 1, 0.04, 25.0, recovery_fraction=1.0)
  descriptor = EpisodeDescriptor(0, "recovery", 0, 0, 0, 42)
  with pytest.raises(ValueError, match="does not match task step_dt"):
    _run_single_episode(
      descriptor, schedule, "unused.pt", "unused", "cpu", RECOVERY, BASE
    )


@pytest.mark.parametrize(
  ("indices", "message"),
  [
    (torch.tensor([0.0]), "integer dtype"),
    (torch.tensor([0, 1]), "shape"),
    (torch.empty(1, dtype=torch.long, device="meta"), "environment device"),
  ],
)
def test_eval_row_override_rejects_bad_tensor_before_state_writes(indices, message):
  # Exercise the reset implementation's early validation, without a simulator:
  # anything that reaches the state-write path would fail on absent robot state.
  event = object.__new__(TennisRecoveryResetEvent)
  event._num_envs = 1
  event._device = "cpu"
  event.recovery_env_mask = torch.ones(1, dtype=torch.bool)
  event._cached = {"joint_pos": torch.zeros(1, 31)}
  event.eval_row_indices = indices
  with pytest.raises(ValueError, match=message):
    event(cast(ManagerBasedRlEnv, None), torch.tensor([0]))


@requires_assets
def test_orientation_failure_on_last_step_wins_over_timeout(monkeypatch):
  pool = EndpointPool.from_directory(DATA, split="validation")
  descriptor = EpisodeDescriptor(
    0, "recovery", 0, int(pool.trajectory_ids[0]), int(pool.frame_indices[0]), 42
  )
  original = registry.load_env_cfg

  def load_with_strict_orientation(*args, **kwargs):
    cfg = original(*args, **kwargs)
    cfg.terminations["fell_over"].params["limit_angle"] = 0.0
    return cfg

  monkeypatch.setattr(registry, "load_env_cfg", load_with_strict_orientation)
  result = _run_single_episode(
    descriptor,
    EvaluationSchedule(1, 1, 0.02, 50.0, recovery_fraction=1.0),
    str(CHECKPOINT),
    str(DATA),
    "cpu",
    RECOVERY,
    BASE,
  )
  assert result["num_steps"] == 1
  assert result["termination_cause"] == "orientation"
  assert not result["metrics"].settled


@requires_assets
def test_recovery_production_export_preserves_controller_metadata(tmp_path):
  import onnxruntime as ort

  cfg = registry.load_env_cfg(BASE)
  cfg.scene.num_envs = 1
  cfg.seed = 42
  base_env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    expected = get_base_metadata(base_env, "ignored")
  finally:
    base_env.close()

  cfg = registry.load_env_cfg(RECOVERY)
  cfg.scene.num_envs = 1
  cfg.seed = 42
  cfg.events["tennis_recovery_reset"].params["pool_directory"] = str(DATA)
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    wrapped = RslRlVecEnvWrapper(env)
    agent = registry.load_rl_cfg(RECOVERY)
    assert agent.upload_model is False
    runner_cls = registry.load_runner_cls(RECOVERY)
    assert runner_cls is not None
    runner = runner_cls(wrapped, asdict(agent), str(tmp_path), device="cpu")
    runner.load(str(CHECKPOINT))
    policy = runner.get_inference_policy(device="cpu")
    obs = wrapped.get_observations()
    with torch.inference_mode():
      torch_actions = policy(obs).cpu().numpy()
    # This is the actual save/export path that attaches deployment metadata.
    saved = tmp_path / "model_test.pt"
    runner.save(str(saved))
    _, _, onnx_path = runner._get_export_paths(str(saved))
    options = cast(Any, ort).SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
      str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    metadata = session.get_modelmeta().custom_metadata_map
    for name, value in expected.items():
      if name == "run_path":
        continue
      encoded = list_to_csv_str(value) if isinstance(value, list) else str(value)
      assert metadata[name] == encoded, name
    output = session.run(
      None, {session.get_inputs()[0].name: obs["actor"].cpu().numpy()}
    )[0]
    assert isinstance(output, np.ndarray)
    np.testing.assert_allclose(
      np.asarray(output, dtype=np.float32),
      np.asarray(torch_actions, dtype=np.float32),
      atol=1e-4,
      rtol=1e-3,
    )
  finally:
    env.close()
