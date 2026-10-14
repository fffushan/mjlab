"""Script to train RL agent with RSL-RL."""

import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.multi_motion_ppo import (
  MultiMotionPpoCommand,
  MultiMotionPpoCommandCfg,
  resolve_motion_files,
)
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import (
  dump_yaml,
  get_checkpoint_path,
  get_wandb_checkpoint_path,
  load_saved_yaml,
)
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlBaseRunnerCfg
  registry_name: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  log_root: str = "logs/rsl_rl"
  """Root directory under which experiment logs are written."""
  torchrunx_log_dir: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


MOTION_RECORD_FILENAME = "resolved_motions.yaml"
"""Name of the resolved-reference record written under the run's ``params``."""


def resolved_motion_record(
  motion_cfg: MultiMotionPpoCommandCfg, command: MultiMotionPpoCommand
) -> dict[str, Any]:
  """Plain-data record of the reference clips and preview one run resolved.

  ``mapping_digest`` covers the ordered clip identity of the loaded library
  (source digest, frame count, and rate of every clip), which is the part a
  resume has to reproduce; the clip entries keep the record readable.
  """
  return {
    "source": motion_cfg.motion_manifest or motion_cfg.motion_file,
    "preview_s": float(command.cfg.lookahead_s),
    "mapping_digest": command.library.mapping_digest(),
    "clips": [
      {
        "name": clip.teacher_id,
        "file": str(clip.motion_file),
        "frames": int(clip.frames),
        "fps": float(clip.fps),
      }
      for clip in command.library.clips
    ],
    "allocation": command.slot_allocation.as_dict(),
  }


def _clip_files(record: dict[str, Any]) -> list[str]:
  """Return the ordered clip files a resolved-motion record names."""
  return [str(entry["file"]) for entry in record.get("clips", [])]


def motion_record_differences(
  saved: dict[str, Any], current: dict[str, Any]
) -> list[str]:
  """Return the reference differences that resume must not accept silently."""
  differences: list[str] = []
  if saved.get("preview_s") != current.get("preview_s"):
    differences.append(
      f"preview_s: saved {saved.get('preview_s')!r} != "
      f"current {current.get('preview_s')!r}"
    )
  if saved.get("mapping_digest") != current.get("mapping_digest"):
    differences.append(
      f"motion list: saved {_clip_files(saved)} "
      f"(digest {saved.get('mapping_digest')!r}) != "
      f"current {_clip_files(current)} (digest {current.get('mapping_digest')!r})"
    )
  return differences


def check_resume_motion_record(resume_path: Path, current: dict[str, Any]) -> None:
  """Refuse a resume whose saved reference record disagrees with this run.

  Training writes the record next to the checkpoint, so a local resume has one.
  A checkpoint obtained without its run files (for example a W&B download) has
  none: the mismatch cannot be checked then, and the run warns instead of
  guessing.
  """
  record_path = resume_path.parent / "params" / MOTION_RECORD_FILENAME
  if not record_path.exists():
    print(
      f"[WARN] No saved {MOTION_RECORD_FILENAME} next to {resume_path.name}; the "
      "motion list and preview of the resumed run cannot be verified."
    )
    return
  differences = motion_record_differences(load_saved_yaml(record_path), current)
  if differences:
    raise ValueError(
      f"resume reference mismatch ({record_path}): "
      + "; ".join(differences)
      + ". Resume with the same motion list and preview as the checkpoint."
    )


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  registry_name: str | None = None

  # Check if this is a tracking task by checking for motion command. The
  # multi-motion teacher is its own case: its reference source is a manifest
  # clip list, so neither the single-file gate nor the W&B motion artifact
  # below applies to it.
  raw_motion_cmd = cfg.env.commands.get("motion")
  multi_motion_cmd = (
    raw_motion_cmd if isinstance(raw_motion_cmd, MultiMotionPpoCommandCfg) else None
  )
  is_tracking_task = (
    isinstance(raw_motion_cmd, MotionCommandCfg) and multi_motion_cmd is None
  )

  if multi_motion_cmd is not None:
    resolved_clips = resolve_motion_files(multi_motion_cmd)
    print(
      f"[INFO] Multi-motion teacher: {len(resolved_clips)} reference clips from "
      f"{multi_motion_cmd.motion_manifest or multi_motion_cmd.motion_file}"
    )
  elif is_tracking_task:
    motion_cmd = raw_motion_cmd
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check if motion_file is already set (e.g., via CLI --env.commands.motion.motion-file).
    if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
      print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")
    elif cfg.registry_name:
      # Download from WandB registry.
      registry_name = cast(str, cfg.registry_name)
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      raise ValueError(
        "For tracking tasks, provide either:\n"
        "  --registry-name your-org/motions/motion-name (download from WandB)\n"
        "  --env.commands.motion.motion-file /path/to/motion.npz (local file)"
      )

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  if cfg.agent.resume:
    if cfg.wandb_run_path is not None:
      # Load checkpoint from W&B.
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      if rank == 0:
        run_id = resume_path.parent.name
        checkpoint_name = resume_path.name
        cached_str = "cached" if was_cached else "downloaded"
        print(
          f"[INFO]: Loading checkpoint from W&B: {checkpoint_name} "
          f"(run: {run_id}, {cached_str})"
        )
    else:
      # Load checkpoint from local filesystem.
      resume_path = get_checkpoint_path(
        log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
      )

  # Record the resolved reference clips and preview, and refuse a resume whose
  # checkpoint resolved a different motion list or preview.  The record is plain
  # data: the loaded library stays out of the serializable configuration.
  if multi_motion_cmd is not None:
    command = env.command_manager.get_term("motion")
    if not isinstance(command, MultiMotionPpoCommand):
      raise RuntimeError(
        "the multi-motion task registered a "
        f"{type(command).__name__} command instead of MultiMotionPpoCommand"
      )
    record = resolved_motion_record(multi_motion_cmd, command)
    if resume_path is not None:
      check_resume_motion_record(resume_path, record)
    if rank == 0:
      dump_yaml(log_dir / "params" / MOTION_RECORD_FILENAME, record)

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  runner_kwargs = {}
  if is_tracking_task:
    # Only the single-clip runner accepts the registry name; the multi-motion
    # teacher runs the standard runner, which has no such argument.
    runner_kwargs["registry_name"] = registry_name

  # Write config files before runner creation, since the runner mutates agent_cfg
  # in-place (e.g., injecting non-serializable objects).
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  add_wandb_tags(cfg.agent.wandb_tags)
  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)

  # Create log directory once before launching workers.
  log_root_path = (Path(args.log_root) / args.agent.experiment_name).resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir)


def main():
  maybe_print_top_level_help("train")

  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
