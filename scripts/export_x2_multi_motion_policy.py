"""Export the X2 multi-motion tracking teacher as a deploy ONNX plus contract metadata.

The export is the raw ``obs -> actions`` graph. The reference is deliberately not embedded,
because one policy serves many clips and the point of the split is to replay clips the
teacher never trained on. Write the reference separately with
``scripts/export_x2_reference_bundle.py``.

A clip is still needed here to build the environment that supplies the metadata: the joint
order, the defaults, the action scale, the observation term list and the anchor body. The
clip choice does not affect the exported weights or the actor layout.

Usage:
  uv run python scripts/export_x2_multi_motion_policy.py \\
    --checkpoint <run>/model_19999.pt \\
    --motion-file data/tennis/single_000_zhanghongyu_agibot_x2_tracking.npz \\
    --out <bundle-dir>/policy.onnx
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts.play import select_multi_motion_playback_clip
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

DEFAULT_TASK = "Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion"


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", required=True, type=Path)
  parser.add_argument(
    "--motion-file",
    required=True,
    type=Path,
    help="One clip, used only to build the environment that supplies the metadata.",
  )
  parser.add_argument(
    "--out", required=True, type=Path, help="Output policy.onnx path."
  )
  parser.add_argument("--task", default=DEFAULT_TASK)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--run-name", default="local")
  parser.add_argument("--parity-samples", type=int, default=4)
  return parser


def export(args: argparse.Namespace) -> dict:
  env_cfg = load_env_cfg(args.task, play=True)
  agent_cfg = load_rl_cfg(args.task)
  clip = select_multi_motion_playback_clip(
    env_cfg.commands["motion"], str(args.motion_file)
  )

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

  onnx_path = runner.export_deploy_onnx(
    str(args.out.parent), args.out.name, args.run_name
  )
  problems = check_parity(runner, onnx_path, args.parity_samples)
  report = {
    "task": args.task,
    "checkpoint": str(args.checkpoint),
    "clip_used_for_metadata": str(clip),
    "onnx": str(onnx_path),
    "problems": problems,
  }
  (onnx_path.parent / "export-report.json").write_text(
    json.dumps(report, indent=2) + "\n"
  )
  return {"onnx": onnx_path, "report": report}


def check_parity(runner, onnx_path: Path, samples: int) -> list[str]:
  """Compare the exported graph against the policy it came from, and report problems."""
  problems: list[str] = []
  session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
  inputs = {value.name: value for value in session.get_inputs()}
  outputs = [value.name for value in session.get_outputs()]

  if sorted(inputs) != ["obs"]:
    problems.append(f"unexpected ONNX inputs {sorted(inputs)}")
  if outputs != ["actions"]:
    problems.append(f"unexpected ONNX outputs {outputs}")

  obs_shape = inputs["obs"].shape
  if len(obs_shape) != 2 or obs_shape[0] != 1:
    problems.append(f"unexpected obs shape {obs_shape}")
  width = int(obs_shape[1])

  torch_model = runner.alg.get_policy().as_onnx(verbose=False)
  torch_model.eval()
  rng = np.random.default_rng(0)
  for _ in range(max(1, samples)):
    obs = rng.standard_normal((1, width)).astype(np.float32)
    with torch.no_grad():
      reference = torch_model(torch.from_numpy(obs)).detach().numpy()
    exported = session.run(["actions"], {"obs": obs})[0]
    if exported.shape != reference.shape:
      problems.append(
        f"action shape differs: onnx {exported.shape} vs torch {reference.shape}"
      )
      continue
    drift = float(np.max(np.abs(exported - reference)))
    if drift > 1e-4:
      problems.append(f"action parity drift {drift:.3e} exceeds 1e-4")

  action_shape = session.get_outputs()[0].shape
  if len(action_shape) == 2 and int(action_shape[0]) != 1:
    problems.append(f"unexpected actions shape {action_shape}")
  return problems


def main() -> None:
  args = build_parser().parse_args()
  result = export(args)
  report = result["report"]
  print(json.dumps({k: v for k, v in report.items() if k != "problems"}, indent=2))
  if report["problems"]:
    raise SystemExit("export validation failed:\n  " + "\n  ".join(report["problems"]))
  print("[OK] exported ONNX matches the checkpoint policy")


if __name__ == "__main__":
  main()
