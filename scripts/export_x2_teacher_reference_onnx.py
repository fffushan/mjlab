"""Export per-clip embedded-reference ONNX teachers for VAE distillation.

A distillation teacher is a different artifact from a deploy policy.  The distillation
loader requires the original tracking interface (``obs, time_step`` in, ``actions`` plus
seven reference outputs out) and it reads a teacher's reference tables out of the ONNX.
The reference-free deploy graph that the controller consumes is therefore refused here,
and one export serves exactly one clip.

The weights are the same shared teacher in every export; only the embedded tables differ.
That is what lets a cohort cover many clips with one policy while the loader still verifies
each member's reference against its own clip.

A clip is also needed to build the environment that supplies the metadata: the joint order,
the defaults, the action scale, the observation terms and the anchor body.

Usage:
  uv run python scripts/export_x2_teacher_reference_onnx.py \\
    --checkpoint <run>/model_19999.pt \\
    --motion-manifest logs/diagnostics/<run>/motions.yaml \\
    --name-pattern '^(single_[0-9]+)' \\
    --out-dir <bundle>/teachers
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import onnx
import onnxruntime as ort

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts.play import select_multi_motion_playback_clip
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp.multi_motion_ppo import resolve_manifest_clips

DEFAULT_TASK = "Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion"
EXPECTED_OUTPUTS = [
  "actions",
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
]


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", required=True, type=Path)
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument("--motion-manifest", type=Path, help="Reference manifest.")
  source.add_argument(
    "--clip", type=Path, action="append", help="One clip, repeatable."
  )
  parser.add_argument("--out-dir", required=True, type=Path)
  parser.add_argument(
    "--name-pattern",
    default=None,
    help="Regex whose first group is the clip name; defaults to the file stem.",
  )
  parser.add_argument("--task", default=DEFAULT_TASK)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--run-name", default="local")
  parser.add_argument("--parity-samples", type=int, default=2)
  return parser


def select_clips(args: argparse.Namespace) -> dict[str, Path]:
  """Map clip names to clips, refusing a name that collides or is unsafe."""
  if args.motion_manifest is not None:
    paths: Sequence[Path] = resolve_manifest_clips(str(args.motion_manifest))
  else:
    paths = tuple(args.clip)
  pattern = re.compile(args.name_pattern) if args.name_pattern else None
  clips: dict[str, Path] = {}
  for path in paths:
    if not path.is_file():
      raise SystemExit(f"clip not found: {path}")
    if pattern is None:
      name = path.stem
    else:
      match = pattern.search(path.name)
      if match is None or match.groups() == ():
        raise SystemExit(f"name pattern did not capture a name in {path.name!r}")
      name = match.group(1)
    if Path(name).name != name or not name:
      raise SystemExit(f"clip name {name!r} is not a safe file stem")
    if name in clips:
      raise SystemExit(f"duplicate clip name {name!r}")
    clips[name] = path
  if not clips:
    raise SystemExit("no clips were selected")
  return clips


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1 << 20), b""):
      digest.update(block)
  return digest.hexdigest()


def verify_export(onnx_path: Path, obs_width: int, samples: int) -> list[str]:
  """Check the loader's stated requirements, and parity against the exported graph."""
  problems: list[str] = []
  session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
  inputs = {value.name: value.shape for value in session.get_inputs()}
  outputs = [value.name for value in session.get_outputs()]

  if inputs != {"obs": [1, obs_width], "time_step": [1, 1]}:
    problems.append(
      f"inputs must be exactly obs/time_step with a {obs_width} obs, got {inputs}"
    )
  if outputs != EXPECTED_OUTPUTS:
    problems.append(f"outputs must be the embedded-reference set, got {outputs}")
  if problems:
    return problems

  # A zero observation and a zero step must produce finite reference rows.
  metadata = {
    entry.key: entry.value for entry in onnx.load(str(onnx_path)).metadata_props
  }
  for key in (
    "joint_names",
    "default_joint_pos",
    "joint_stiffness",
    "joint_damping",
    "action_scale",
    "observation_names",
    "observation_terms_scale",
    "command_names",
    "anchor_body_name",
    "body_names",
    "motion_lookahead_steps",
  ):
    if key not in metadata:
      problems.append(f"metadata is missing {key!r}")

  for _ in range(max(1, samples)):
    values = session.run(
      outputs,
      {
        "obs": np.zeros((1, obs_width), np.float32),
        "time_step": np.zeros((1, 1), np.float32),
      },
    )
    for name, value in zip(outputs, values, strict=True):
      if not np.isfinite(value).all():
        problems.append(f"output {name!r} is not finite at time step 0")
  return problems


def export(args: argparse.Namespace) -> dict:
  clips = select_clips(args)
  first = clips[sorted(clips)[0]]

  env_cfg = load_env_cfg(args.task, play=True)
  agent_cfg = load_rl_cfg(args.task)
  select_multi_motion_playback_clip(env_cfg.commands["motion"], str(first))
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(args.task)
  assert runner_cls is not None, f"task {args.task!r} registers no runner class"
  runner = runner_cls(wrapped, asdict(agent_cfg), device=args.device)
  runner.load(
    str(args.checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=args.device,
  )
  obs_width = int(runner.alg.get_policy().as_onnx(verbose=False).input_size)
  args.out_dir.mkdir(parents=True, exist_ok=True)

  entries: list[dict] = []
  failures: list[str] = []
  for name in sorted(clips):
    target = args.out_dir / f"{name}.onnx"
    runner.export_embedded_reference_onnx(
      clips[name], str(args.out_dir), target.name, args.run_name
    )
    problems = verify_export(target, obs_width, args.parity_samples)
    if problems:
      failures.extend(f"{name}: {problem}" for problem in problems)
    entries.append(
      {
        "name": name,
        "onnx": target.name,
        "motion": str(clips[name]),
        "onnx_sha256": sha256_file(target),
        "bytes": target.stat().st_size,
        "problems": problems,
      }
    )
    print(
      f"  {name:22} {target.stat().st_size:>10} B  {'ok' if not problems else 'FAILED'}"
    )

  report = {
    "task": args.task,
    "checkpoint": str(args.checkpoint),
    "checkpoint_sha256": sha256_file(args.checkpoint),
    "observation_width": obs_width,
    "clips": entries,
    "failures": failures,
  }
  (args.out_dir / "export-report.json").write_text(json.dumps(report, indent=2) + "\n")
  return report


def main() -> None:
  args = build_parser().parse_args()
  report = export(args)
  print(
    f"\n{len(report['clips'])} clips, observation width {report['observation_width']}"
  )
  if report["failures"]:
    raise SystemExit(
      "export validation failed:\n  " + "\n  ".join(report["failures"][:20])
    )
  print("[OK] every export carries the embedded-reference interface")


if __name__ == "__main__":
  main()
