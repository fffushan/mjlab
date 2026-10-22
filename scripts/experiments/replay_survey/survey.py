"""One batched trackability pass over prepared segments.

Each batch pins one environment to one reference, so a batch of N segments costs
one environment build instead of N.  Every row is replayed from frame 0 with the
same frozen teacher, and its first-episode outcome is recorded together with the
termination term that fired.

The pass writes one result file per batch.  Completed batches are skipped on a
restart, so a pass can be resumed without repeating work.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WORKTREE = Path(os.environ.get("MJLAB_WORKTREE", HERE.parents[2]))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(WORKTREE / "scripts/experiments/joint_pose_collection"))

import build_refs_all as refs  # noqa: E402
import collect as collector  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import (  # noqa: E402
  load_env_cfg,
  load_rl_cfg,
  load_runner_cls,
)
from mjlab.tasks.tracking.mdp.commands import load_saved_lookahead_s  # noqa: E402
from mjlab.tasks.tracking.mdp.multi_motion_ppo import (  # noqa: E402
  MultiMotionPpoCommandCfg,
)
from mjlab.utils.torch import configure_torch_backends  # noqa: E402

DATA = Path("/home/fushan/data/mjlab-latent-free-144/data")
CLIP_COMPLETE = collector.CLIP_COMPLETE_TERM


def plan_batches(table: list[dict], lanes: int, lane: int, batch: int) -> list[list[dict]]:
  """Order segments by length, then deal them to lanes.

  Dealing from a length-sorted list keeps each lane's length mix similar and
  keeps each batch's longest clip close to its shortest, which bounds the idle
  steps a batch spends on rows that already finished.
  """
  ordered = sorted(table, key=lambda e: (-int(e["rows"]), int(e["segment"])))
  mine = [e for position, e in enumerate(ordered) if position % lanes == lane]
  return [mine[start : start + batch] for start in range(0, len(mine), batch)]


def firing_terms(termination) -> list[str | None]:
  """Name the first physical termination term that fired for each row."""
  names = [
      name
      for name in termination.active_terms
      if name != CLIP_COMPLETE and not termination.get_term_cfg(name).time_out
  ]
  if not names:
    return [None] * termination.num_envs
  masks = {name: termination.get_term(name).bool() for name in names}
  out: list[str | None] = []
  for row in range(termination.num_envs):
    hit = next((name for name in names if bool(masks[name][row])), None)
    out.append(hit)
  return out


def run_batch(args, batch_entries: list[dict], work: Path, q, dq) -> list[dict]:
  """Replay one batch and return one result row per segment."""
  work.mkdir(parents=True, exist_ok=True)
  ref_dir = work / "refs"
  ref_dir.mkdir(exist_ok=True)
  clip_cache: dict[str, dict[str, np.ndarray]] = {}
  src_cache: dict[str, np.ndarray] = {}
  paths: list[Path] = []
  expect_rows: dict[str, int] = {}
  for entry in batch_entries:
    clip = refs.load_clip_arrays(entry["motion"], clip_cache)
    init = int(refs.initial_frame_table(entry["source"], src_cache)[int(entry["trajectory_index"])])
    path, rows = refs.write_reference(
        out_dir=ref_dir, entry=entry, q=q, dq=dq, clip=clip, init=init
    )
    paths.append(path)
    expect_rows[path.name] = rows

  manifest = work / "manifest.yaml"
  # The reference manifest is plain YAML with one 'motions' list.
  manifest.write_text(
      "motions:\n" + "".join(f"  - {p}\n" for p in paths), encoding="utf-8"
  )
  n = len(paths)

  env_cfg = load_env_cfg(args.task, play=False)
  motion_cmd = env_cfg.commands.get("motion")
  if not isinstance(motion_cmd, MultiMotionPpoCommandCfg):
    raise RuntimeError("survey needs the multi-motion tracking task")
  lookahead_s = load_saved_lookahead_s(args.checkpoint_file)
  if lookahead_s is None:
    raise RuntimeError("checkpoint has no saved params/env.yaml")
  motion_cmd.motion_manifest = str(manifest)
  motion_cmd.motion_file = ""
  motion_cmd.sampling_mode = "start"
  motion_cmd.standing_start_prob = 0.0
  motion_cmd.lookahead_s = lookahead_s
  env_cfg.observations["actor"].enable_corruption = True
  env_cfg.events.pop("push_robot", None)
  env_cfg.scene.num_envs = n
  fps = 50.0
  env_cfg.episode_length_s = (max(expect_rows.values()) + 2) / fps
  env_cfg.seed = args.seed
  env_cfg.auto_reset = False

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  agent_cfg = load_rl_cfg(args.task)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=args.device)
  runner.load(str(args.checkpoint_file), map_location=args.device)
  policy = runner.get_inference_policy(device=args.device)
  del runner

  unwrapped = env.unwrapped
  command = unwrapped.command_manager.get_term("motion")
  termination = unwrapped.termination_manager
  env_ids = torch.arange(n, device=args.device)
  command.select_motion_ids(env_ids, env_ids.clone())

  obs = env.get_observations()
  if hasattr(obs, "keys"):
    obs = obs["obs"] if "obs" in obs.keys() else obs
  outcomes: list[str | None] = [None] * n
  steps = [0] * n
  terms: list[str | None] = [None] * n
  active = torch.ones(n, dtype=torch.bool, device=args.device)
  budget = max(expect_rows.values()) + 2
  for step in range(1, budget + 1):
    with torch.no_grad():
      actions = policy(obs)
    obs, _rew, dones, _extras = env.step(actions)
    # Every done row must be reset before the next step, including rows whose
    # recorded trajectory already ended: they keep simulating a fresh episode.
    # Only rows that are still active are classified into the ledger.
    done_now = dones.bool()
    if bool(done_now.any()):
      complete = termination.get_term(CLIP_COMPLETE)
      failed = collector.physical_failure_mask(termination)
      timed_out = termination.time_outs
      classified = collector.classify_finished_rows(
          done_now & active, complete, failed, timed_out
      )
      names = firing_terms(termination)
      for row, outcome in enumerate(classified):
        if outcome is None:
          continue
        outcomes[row] = outcome
        steps[row] = step
        terms[row] = names[row] if outcome == collector.OUTCOME_FAILURE else None
        active[row] = False
      reset_ids = done_now.nonzero(as_tuple=False).squeeze(-1)
      if reset_ids.numel():
        unwrapped.reset(env_ids=reset_ids)
        obs = env.get_observations()
        if hasattr(obs, "keys"):
          obs = obs["obs"] if "obs" in obs.keys() else obs
    if not bool(active.any()):
      break

  rows_out: list[dict] = []
  for row, entry in enumerate(batch_entries):
    path = paths[row]
    rows_out.append({
        "segment": int(entry["segment"]),
        "group": entry["group"],
        "motion": entry["motion"],
        "source": entry["source"],
        "trajectory_index": int(entry["trajectory_index"]),
        "ref_rows": int(expect_rows[path.name]),
        "windows": int(entry["windows"]),
        "outcome": outcomes[row] or collector.OUTCOME_BUDGET,
        "steps": int(steps[row]),
        "term": terms[row],
    })
  shutil.rmtree(ref_dir, ignore_errors=True)
  return rows_out


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint-file", type=Path, required=True)
  ap.add_argument("--output-dir", type=Path, required=True)
  ap.add_argument("--task", default="Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion")
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--batch", type=int, default=128)
  ap.add_argument("--lanes", type=int, default=4)
  ap.add_argument("--lane", type=int, default=0)
  ap.add_argument("--seed", type=int, default=500)
  ap.add_argument("--max-batches", type=int, default=0)
  args = ap.parse_args()

  configure_torch_backends()
  table = json.loads((DATA / "store" / "manifest.json").read_text())["table"]
  batches = plan_batches(table, args.lanes, args.lane, args.batch)
  if args.max_batches:
    batches = batches[: args.max_batches]
  q = np.load(DATA / "store" / "joint_pos.npy", mmap_mode="r")
  dq = np.load(DATA / "store" / "joint_vel.npy", mmap_mode="r")
  args.output_dir.mkdir(parents=True, exist_ok=True)
  print(f"[INFO] lane {args.lane}/{args.lanes}: {len(batches)} batches of {args.batch}", flush=True)
  started = time.time()
  for index, entries in enumerate(batches):
    result_path = args.output_dir / f"batch{index:05d}.json"
    if result_path.is_file():
      continue
    work = args.output_dir / f"work{index:05d}"
    rows = run_batch(args, entries, work, q, dq)
    result_path.write_text(json.dumps({"lane": args.lane, "batch": index, "rows": rows}))
    shutil.rmtree(work, ignore_errors=True)
    if index % 10 == 0:
      done = sum(r["outcome"] != "complete" for r in rows)
      print(
          f"[INFO] batch {index + 1}/{len(batches)} rows={len(rows)} failed={done} "
          f"elapsed={time.time() - started:.0f}s",
          flush=True,
      )


if __name__ == "__main__":
  main()
