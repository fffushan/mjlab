#!/bin/bash
# run_flow_3261.sh - data-parallel launcher for the 3 261-clip X2 flow run.
#
# Usage:
#   scripts/run_flow_3261.sh \
#     --gpus 0,1,2,3,4,5,6,7 \
#     --dataset-dir /tmp/d1-3261-local \
#     --config configs/flow_matching/x2_50hz_train.yaml \
#     --contract docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml \
#     --out logs/flow/flow-3261-ddp \
#     --max-updates 1000000
#
# One process per card, launched through torchrunx the same way
# `scripts/train.py` launches a multi-GPU run.  The global batch is held fixed:
# every rank consumes
#
#   effective_batch_size // (gradient_accumulation_steps * number_of_ranks)
#
# windows per micro-batch.  The script prints that value before it launches and
# refuses when the division is not exact, or when its own arithmetic disagrees
# with `per_rank_microbatch` in the trainer.  A rounded batch would silently
# move the learning-rate and warmup calibration of every single-card comparison.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  sed -n '2,15p' "$0"
  exit 2
}

GPUS=""
DATASET_DIR=""
CONFIG=""
CONTRACT=""
OUT=""
MAX_UPDATES=""

while [ $# -gt 0 ]; do
  case "$1" in
    --gpus)        GPUS="${2:-}"; shift 2 ;;
    --gpus=*)      GPUS="${1#--gpus=}"; shift ;;
    --dataset-dir) DATASET_DIR="${2:-}"; shift 2 ;;
    --dataset-dir=*) DATASET_DIR="${1#--dataset-dir=}"; shift ;;
    --config)      CONFIG="${2:-}"; shift 2 ;;
    --config=*)    CONFIG="${1#--config=}"; shift ;;
    --contract)    CONTRACT="${2:-}"; shift 2 ;;
    --contract=*)  CONTRACT="${1#--contract=}"; shift ;;
    --out)         OUT="${2:-}"; shift 2 ;;
    --out=*)       OUT="${1#--out=}"; shift ;;
    --max-updates) MAX_UPDATES="${2:-}"; shift 2 ;;
    --max-updates=*) MAX_UPDATES="${1#--max-updates=}"; shift ;;
    -h|--help)     usage ;;
    *) echo "[error] unknown argument: $1" >&2; usage ;;
  esac
done

for name in GPUS DATASET_DIR CONFIG CONTRACT OUT MAX_UPDATES; do
  if [ -z "${!name}" ]; then
    echo "[error] --$(echo "$name" | tr 'A-Z_' 'a-z-') is required" >&2
    usage
  fi
done

case "$MAX_UPDATES" in
  ''|*[!0-9]*) echo "[error] --max-updates must be a positive integer: $MAX_UPDATES" >&2; exit 2 ;;
esac
if [ "$MAX_UPDATES" -le 0 ]; then
  echo "[error] --max-updates must be positive: $MAX_UPDATES" >&2
  exit 2
fi
if [ ! -f "$CONFIG" ]; then
  echo "[error] config not found: $CONFIG" >&2
  exit 2
fi
if [ ! -f "$CONTRACT" ]; then
  echo "[error] contract not found: $CONTRACT" >&2
  exit 2
fi
if [ ! -x "$PYTHON" ]; then
  echo "[error] python not executable: $PYTHON (set PYTHON=...)" >&2
  exit 2
fi

# --------------------------------------------------------- rank and batch math
IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
RANKS="${#GPU_ARRAY[@]}"
if [ "$RANKS" -lt 1 ]; then
  echo "[error] --gpus lists no device: $GPUS" >&2
  exit 2
fi
for device in "${GPU_ARRAY[@]}"; do
  case "$device" in
    ''|*[!0-9]*) echo "[error] --gpus entries must be non-negative integers: $GPUS" >&2; exit 2 ;;
  esac
done

# The recipe comes from the config, not from this script: one loader for both
# fields, so the printed value is the value the trainer will derive.
read -r EFFECTIVE ACCUM < <("$PYTHON" - "$CONFIG" <<'PY'
import sys

from mjlab.tasks.tracking.flow_matching import FlowTrainingConfig

config = FlowTrainingConfig.from_yaml(sys.argv[1])
print(config.training.effective_batch_size, config.training.gradient_accumulation_steps)
PY
)
if ! [[ "$EFFECTIVE" =~ ^[0-9]+$ && "$ACCUM" =~ ^[0-9]+$ ]]; then
  echo "[error] could not read effective_batch_size and gradient_accumulation_steps" >&2
  echo "        from $CONFIG" >&2
  exit 2
fi

DIVISOR=$(( ACCUM * RANKS ))
if [ "$DIVISOR" -lt 1 ]; then
  echo "[error] gradient_accumulation_steps * ranks is zero" >&2
  exit 2
fi
SHELL_MICROBATCH=$(( EFFECTIVE / DIVISOR ))
if [ $(( SHELL_MICROBATCH * DIVISOR )) -ne "$EFFECTIVE" ]; then
  echo "[error] effective_batch_size $EFFECTIVE is not divisible by" >&2
  echo "        gradient_accumulation_steps * ranks = $ACCUM * $RANKS = $DIVISOR" >&2
  echo "        the global batch cannot be held fixed; change --gpus or the config" >&2
  exit 2
fi
if [ "$SHELL_MICROBATCH" -lt 1 ]; then
  echo "[error] $RANKS ranks leave a per-rank microbatch below one for" >&2
  echo "        effective batch $EFFECTIVE and $ACCUM accumulation steps" >&2
  exit 2
fi

TRAINER_MICROBATCH="$("$PYTHON" - "$EFFECTIVE" "$ACCUM" "$RANKS" <<'PY'
import sys

from mjlab.tasks.tracking.flow_matching import per_rank_microbatch

print(
  per_rank_microbatch(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
)
PY
)"
if [ "$TRAINER_MICROBATCH" != "$SHELL_MICROBATCH" ]; then
  echo "[error] per-rank microbatch disagrees: shell $SHELL_MICROBATCH," \
       "trainer $TRAINER_MICROBATCH" >&2
  exit 2
fi

echo "[ddp] ranks              : $RANKS (gpus $GPUS)"
echo "[ddp] global batch       : $EFFECTIVE ($ACCUM accumulation steps)"
echo "[ddp] per-rank microbatch: $TRAINER_MICROBATCH"
echo "[ddp] dataset            : $DATASET_DIR"
echo "[ddp] output             : $OUT"

# ------------------------------------------------------------------- launch
export CUDA_VISIBLE_DEVICES="$GPUS"
export FLOW_DDP_RANKS="$RANKS"
if [ -z "${TORCHRUNX_LOG_DIR:-}" ]; then
  export TORCHRUNX_LOG_DIR="$OUT/torchrunx"
fi
mkdir -p "$OUT"

exec "$PYTHON" - train \
  --dataset-dir "$DATASET_DIR" \
  --config "$CONFIG" \
  --contract "$CONTRACT" \
  --out "$OUT" \
  --device cuda:0 \
  --max-updates "$MAX_UPDATES" \
  --allow-long-run \
  --json "$OUT/train.json" <<'PY'
"""Launch the flow CLI on every worker through torchrunx.

The trainer initialises its own process group, so ``backend=None`` here: the
launcher only sets RANK, LOCAL_RANK, WORLD_SIZE and the rendezvous address.
"""
import logging
import os
import sys

import torchrunx

from mjlab.scripts.flow_matching import main

logging.basicConfig(level=logging.INFO)
print(
  f"[INFO] launching {os.environ['FLOW_DDP_RANKS']} flow workers",
  flush=True,
)
launcher = torchrunx.Launcher(
  hostnames=["localhost"],
  workers_per_host=int(os.environ["FLOW_DDP_RANKS"]),
  backend=None,
  copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY
  + ("MJLAB*", "PYTHONPATH", "CUBLAS*"),
)
result = launcher.run(main, sys.argv[1:])
codes = [code for host in result.results.values() for code in host]
raise SystemExit(max((int(code or 0) for code in codes), default=0))
PY
