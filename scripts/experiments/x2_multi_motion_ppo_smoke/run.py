"""Bounded real-GPU smoke for the shared X2 PPO task; not a quality evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.scripts import train as train_script
from mjlab.tasks.tracking.mdp.multi_motion_ppo import MultiMotionPpoCommand

TASK = "Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion"
ENVS = 8
UPDATES = 10


def progress(stage: str, **values) -> None:
  print(json.dumps({"progress": {"stage": stage, **values}}), flush=True)


def finite_observations(obs) -> None:
  for name, tensor in obs.items():
    assert bool(torch.isfinite(tensor).all()), f"nonfinite observation: {name}"


def memory_record() -> dict:
  return {
    "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  args = parser.parse_args()
  if args.output_dir.exists():
    raise FileExistsError(f"Refuse to overwrite smoke: {args.output_dir}")
  args.output_dir.mkdir(parents=True)
  result = {"task": TASK, "num_envs": ENVS, "ppo_updates": UPDATES, "passed": False}
  started = time.monotonic()
  instances = []
  source = {}
  try:
    progress("gpu_preflight")
    assert torch.cuda.is_available()
    result["gpu"] = torch.cuda.get_device_name(0)
    result["pid"] = os.getpid()
    cgroup = Path("/proc/self/cgroup").read_text().split("0::", 1)[1].strip()
    cap = (
      (Path("/sys/fs/cgroup") / cgroup.lstrip("/") / "memory.max").read_text().strip()
    )
    result["host_memory_cap_bytes"] = cap
    assert cap != "max" and int(cap) <= 12 * 1024**3, cap

    cfg = train_script.TrainConfig.from_task(TASK)
    cfg.env.scene.num_envs = ENVS
    cfg.env.commands["motion"].motion_manifest = str(args.manifest.resolve())
    cfg.agent.max_iterations = UPDATES
    cfg.agent.save_interval = 5
    cfg.agent.seed = 7
    cfg.agent.logger = "tensorboard"
    cfg.agent.upload_model = False
    files = train_script.resolve_motion_files(cfg.env.commands["motion"])
    assert len(files) == 2
    for path in files:
      with np.load(path, allow_pickle=False) as data:
        source[path] = {key: data[key].copy() for key in ("joint_pos", "joint_vel")}
        result.setdefault("clips", []).append(
          {
            "path": str(path),
            "frames": int(data["joint_pos"].shape[0]),
            "fps": float(data["fps"].reshape(-1)[0]),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
          }
        )
    selector = train_script.load_runner_cls
    assert selector(TASK) is MjlabOnPolicyRunner

    class SmokeRunner(MjlabOnPolicyRunner):
      def __init__(self, *runner_args, **runner_kwargs):
        super().__init__(*runner_args, **runner_kwargs)
        instances.append(self)
        self._initial_actor = {
          name: value.detach().clone()
          for name, value in self.alg.get_policy().named_parameters()
        }
        progress("real_runner_built", **memory_record())

      def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        assert num_learning_iterations == UPDATES
        raw = self.env.unwrapped
        command = raw.command_manager.get_term("motion")
        assert isinstance(command, MultiMotionPpoCommand)
        obs = self.env.get_observations()
        finite_observations(obs)
        joint_dim = int(raw.scene["robot"].data.joint_pos.shape[1])
        body_count = len(command.cfg.body_names)
        actor_width = 4 * joint_dim + 6 + 3 + 3 * joint_dim
        critic_width = 4 * joint_dim + 3 + 6 + 9 * body_count + 3 + 3 + 3 * joint_dim
        result["observation_shapes"] = {
          key: list(value.shape) for key, value in obs.items()
        }
        assert tuple(obs["actor"].shape) == (ENVS, actor_width), result[
          "observation_shapes"
        ]
        assert tuple(obs["critic"].shape) == (ENVS, critic_width), result[
          "observation_shapes"
        ]
        for group in ("actor", "critic"):
          assert "motion_lookahead" in raw.observation_manager.active_terms[group]
        counts = torch.bincount(command.motion_ids, minlength=2).cpu().tolist()
        assert counts == [4, 4], counts
        result["allocation"] = counts

        def check_preview():
          expected = []
          for row in range(ENVS):
            motion = int(command.motion_ids[row])
            frame = min(
              int(command.time_steps[row]) + 10, int(command.row_lengths[row]) - 1
            )
            clip = source[command.library.clips[motion].motion_file]
            expected.append(
              np.concatenate([clip["joint_pos"][frame], clip["joint_vel"][frame]])
            )
          expected_tensor = torch.as_tensor(
            np.asarray(expected), device=command.device, dtype=torch.float32
          )
          torch.testing.assert_close(
            command.lookahead_command, expected_tensor, rtol=0, atol=0
          )

        check_preview()
        result["preview_exact"] = True
        old_sampling = command.cfg.sampling_mode
        command.cfg.sampling_mode = "start"
        raw.reset(seed=7)
        endpoint_row = torch.tensor([0], device=command.device)
        command.reset_to_frame(endpoint_row, int(command.row_lengths[0]) - 1)
        raw.sim.forward()
        command.update_relative_body_poses()
        raw.sim.sense()
        check_preview()
        _, _, done, _ = self.env.step(torch.zeros(ENVS, 31, device=command.device))
        assert bool(done[0])
        assert bool(raw.termination_manager.get_term("clip_complete")[0])
        result["forced_endpoint_done"] = True

        before_frames = command.time_steps.clone()
        before_joints = raw.scene["robot"].data.joint_pos.clone()
        before_generations = command.generation_ids.clone()
        subset = torch.tensor([1], device=command.device)
        raw.reset(env_ids=subset)
        other = torch.arange(ENVS, device=command.device) != 1
        assert int(command.time_steps[1]) == 0
        assert torch.equal(command.time_steps[other], before_frames[other])
        assert torch.equal(command.generation_ids[other], before_generations[other])
        torch.testing.assert_close(
          raw.scene["robot"].data.joint_pos[other], before_joints[other], rtol=0, atol=0
        )
        assert int(command.generation_ids[1]) == int(before_generations[1]) + 1
        result["subset_reset_isolated"] = True
        command.cfg.sampling_mode = old_sampling
        raw.reset(seed=7)
        progress("runtime_probes_pass", **memory_record())

        losses = []
        rollout = {"steps": 0, "done_rows": 0, "clip_complete_rows": 0}
        original_step = self.env.step
        original_update = self.alg.update

        def checked_step(actions):
          assert bool(torch.isfinite(actions).all())
          values = original_step(actions)
          finite_observations(values[0])
          assert bool(torch.isfinite(values[1]).all())
          rollout["steps"] += 1
          rollout["done_rows"] += int(values[2].sum())
          rollout["clip_complete_rows"] += int(
            raw.termination_manager.get_term("clip_complete").sum()
          )
          return values

        def checked_update():
          values = original_update()
          assert all(math.isfinite(float(value)) for value in values.values()), values
          losses.append({name: float(value) for name, value in values.items()})
          progress("ppo_update", current=len(losses), total=UPDATES, loss=losses[-1])
          return values

        with (
          patch.object(self.env, "step", side_effect=checked_step),
          patch.object(self.alg, "update", side_effect=checked_update),
        ):
          super().learn(num_learning_iterations, init_at_random_ep_len)
        assert len(losses) == UPDATES
        assert rollout["steps"] == UPDATES * 24
        actor = self.alg.get_policy()
        delta = max(
          float((parameter.detach() - self._initial_actor[name]).abs().max())
          for name, parameter in actor.named_parameters()
        )
        assert math.isfinite(delta) and delta > 0
        assert all(
          bool(torch.isfinite(value).all()) for value in actor.state_dict().values()
        )
        result.update(
          losses=losses, rollout=rollout, actor_parameter_max_abs_change=delta
        )

        checkpoint = args.output_dir / "model_9.pt"
        assert checkpoint.is_file()
        frozen_obs = self.env.get_observations().clone()
        finite_observations(frozen_obs)
        with torch.inference_mode():
          action_before = self.get_inference_policy()(frozen_obs).detach().clone()
        saved_actor = {
          name: value.detach().clone() for name, value in actor.state_dict().items()
        }
        optimizer_steps = [
          int(state["step"])
          for state in self.alg.optimizer.state.values()
          if "step" in state
        ]
        assert optimizer_steps and min(optimizer_steps) > 0
        with torch.no_grad():
          next(
            parameter for parameter in actor.parameters() if parameter.ndim == 2
          ).add_(0.5)
        self.load(str(checkpoint), map_location=command.device)
        assert all(
          torch.equal(value, saved_actor[name])
          for name, value in actor.state_dict().items()
        )
        with torch.inference_mode():
          action_after = self.get_inference_policy()(frozen_obs).detach().clone()
        torch.testing.assert_close(action_after, action_before, rtol=0, atol=0)
        result.update(
          checkpoint=str(checkpoint),
          checkpoint_reload_exact=True,
          inference_action_max_abs_reload_error=float(
            (action_after - action_before).abs().max()
          ),
          optimizer_steps_range=[min(optimizer_steps), max(optimizer_steps)],
        )
        progress("checkpoint_reload_pass", **memory_record())

    progress("construct_real_x2_env", clips=result["clips"])
    with patch.object(train_script, "load_runner_cls", return_value=SmokeRunner):
      train_script.run_train(TASK, cfg, args.output_dir)
    for clip in result["clips"]:
      assert (
        hashlib.sha256(Path(clip["path"]).read_bytes()).hexdigest() == clip["sha256"]
      )
    result.update(passed=True, elapsed_s=time.monotonic() - started, **memory_record())
    progress("smoke_pass", elapsed_s=result["elapsed_s"])
  except BaseException as exc:
    result.update(
      error=repr(exc),
      traceback=traceback.format_exc(),
      elapsed_s=time.monotonic() - started,
    )
    raise
  finally:
    for runner in instances:
      runner.env.close()
    (args.output_dir / "smoke-result.json").write_text(
      json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
  main()
