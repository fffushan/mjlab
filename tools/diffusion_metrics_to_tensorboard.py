#!/usr/bin/env python3
"""Convert mjlab diffusion `metrics.jsonl` traces into TensorBoard event files.

The diffusion trainer appends one JSON record per optimizer update and one per
epoch; it does not write TensorBoard events itself.  This rewrites those traces
as scalars so the loss curves can be inspected in the normal TensorBoard UI.

The run directories are read-only inputs: events are written to a separate
output root, so no run artifact is modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

# Metric name -> TensorBoard tag, for the per-update training trace.
UPDATE_SCALARS = {
  "loss": "train/loss",
  "state_mse": "train/mse/state",
  "latent_mse": "train/mse/latent",
  "unknown_mse": "train/mse/unknown",
  "unknown_state_mse": "train/mse/unknown_state",
  "unknown_latent_mse": "train/mse/unknown_latent",
  "state_mse_1_9": "train/horizon/state_1_9",
  "state_mse_10_99": "train/horizon/state_10_99",
  "state_mse_100_999": "train/horizon/state_100_999",
  "latent_mse_1_9": "train/horizon/latent_1_9",
  "latent_mse_10_99": "train/horizon/latent_10_99",
  "latent_mse_100_999": "train/horizon/latent_100_999",
  "state_mse_clean": "train/phase/state_clean",
  "state_mse_terminal": "train/phase/state_terminal",
  "latent_mse_clean": "train/phase/latent_clean",
  "latent_mse_terminal": "train/phase/latent_terminal",
  "learning_rate": "optim/learning_rate",
  "ema_decay": "optim/ema_decay",
}

# Metrics inside the per-epoch evaluation blocks (model and EMA weights).
EVAL_SCALARS = {
  "loss": "loss",
  "total_mse": "mse/total",
  "unknown_mse": "mse/unknown",
  "state_mse": "mse/state",
  "latent_mse": "mse/latent",
  "state_mse_clean": "phase/state_clean",
  "state_mse_terminal": "phase/state_terminal",
  "latent_mse_clean": "phase/latent_clean",
  "latent_mse_terminal": "phase/latent_terminal",
  "windows": "meta/windows",
}


def convert_run(run_dir: Path, out_dir: Path, stride: int) -> dict[str, int]:
  metrics = run_dir / "metrics.jsonl"
  if not metrics.exists():
    return {"rows": 0, "updates": 0, "epochs": 0}
  writer = SummaryWriter(log_dir=str(out_dir / run_dir.name))
  counts = {"rows": 0, "updates": 0, "epochs": 0, "bad": 0}
  with metrics.open() as handle:
    for line in handle:
      counts["rows"] += 1
      try:
        row = json.loads(line)
      except json.JSONDecodeError:
        counts["bad"] += 1
        continue
      step = int(row.get("global_step", 0))
      if row.get("kind") == "update":
        counts["updates"] += 1
        if stride > 1 and counts["updates"] % stride:
          continue
        for key, tag in UPDATE_SCALARS.items():
          value = row.get(key)
          if value is not None:
            writer.add_scalar(tag, float(value), step)
      else:
        counts["epochs"] += 1
        if "train_loss" in row:
          writer.add_scalar("train/epoch_mean_loss", float(row["train_loss"]), step)
        writer.add_scalar("train/epoch", float(row.get("epoch", 0)), step)
        for block in ("eval_model", "eval_ema"):
          values = row.get(block) or {}
          for key, tag in EVAL_SCALARS.items():
            if values.get(key) is not None:
              writer.add_scalar(
                f"val_{block.split('_')[1]}/{tag}", float(values[key]), step
              )
  writer.flush()
  writer.close()
  return counts


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--runs-root", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--run", action="append", default=None)
  parser.add_argument(
    "--stride",
    type=int,
    default=4,
    help="log every Nth optimizer update (evaluation rows are never skipped)",
  )
  args = parser.parse_args()

  names = args.run or sorted(
    path.name for path in args.runs_root.iterdir() if (path / "metrics.jsonl").exists()
  )
  total = 0
  for name in names:
    counts = convert_run(args.runs_root / name, args.out, max(1, args.stride))
    total += counts["rows"]
    print(
      "%-30s rows=%-7d updates=%-7d epochs=%-5d bad=%d"
      % (
        name,
        counts["rows"],
        counts["updates"],
        counts["epochs"],
        counts.get("bad", 0),
      )
    )
  print(f"written to {args.out} ({total} rows total, stride {args.stride})")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
