#!/bin/bash
# Post-training evaluation for one diffusion wave, as a single idempotent command.
#
#   run_wave_eval.sh <wave> <train-dir> <contract> <reference-arm> <arm>...
#
# Stages, each with a marker so a re-run skips what is already done:
#   0 intrinsic residual + train-pool k-NN baseline on the wave dataset  (CPU)
#   1 paired generation metrics across every arm, identical windows      (GPU)
#   2 generated spread vs residual bias for every arm                    (GPU)
#   3 wave summary: curves, budget verdict, per-motion sign consistency  (CPU)
#
# Bounds are env-tunable: PAIRED_WINDOWS, DIV_WINDOWS, SEEDS, DEVICE, DRY_RUN=1.
# DRY_RUN verifies every input and prints the commands without running anything.
set -u
W=/home/fushan/mjlab-diffvae
PY=/home/fushan/mjlab/.venv/bin/python
wave=${1:?wave name}; train=${2:?training dir}; contract=${3:?contract path}
reference=${4:?reference arm}; shift 4
arms=("$@")
OUT=$W/logs/diagnostics/$wave-evaluation
PAIRED_WINDOWS=${PAIRED_WINDOWS:-12000}
DIV_WINDOWS=${DIV_WINDOWS:-4000}
SEEDS=${SEEDS:-0,1,2,3}
DEVICE=${DEVICE:-cuda:0}
DRY_RUN=${DRY_RUN:-0}
cd "$W" || exit 1
export PYTHONPATH="$W/src"
mkdir -p "$OUT"

problems=""
for a in "${arms[@]}"; do
  [ -s "$train/$a/checkpoint-best.pt" ] || problems="$problems $a:no-checkpoint-best"
  [ -s "$train/$a/train-report.json" ] || problems="$problems $a:no-report"
done
[ -s "$contract" ] || problems="$problems contract-missing"
if [ -n "$problems" ]; then echo "INPUT PROBLEM:$problems"; exit 2; fi

# The raw checkpoint file is a dict; the library wraps it in a state object.  Accept
# either shape rather than assuming one.
dataset=$("$PY" -c "
import sys, torch
state = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
identity = state.get('dataset_identity') if isinstance(state, dict) else getattr(state, 'dataset_identity', None)
print((identity or {}).get('directory', '') if isinstance(identity, dict) else '')
" "$train/${arms[0]}/checkpoint-best.pt")
[ -n "$dataset" ] || { echo "ABORT: checkpoint records no dataset directory"; exit 2; }
echo "wave=$wave train=$train reference=$reference"
echo "dataset=$dataset"
echo "arms=${arms[*]} paired_windows=$PAIRED_WINDOWS div_windows=$DIV_WINDOWS seeds=$SEEDS"

paired_args=()
for a in "${arms[@]}"; do paired_args+=(--arm "$a=logs/diffusion/$wave/$a:$contract"); done

stage0_cmd="$PY scripts/experiments/diffusion_future_predictability/analyze.py --dataset-dir $dataset --contract $contract --cache-dir logs/diffusion/$wave/${arms[0]} --pool-split train --neighbours 16 --max-queries 1200 --out $OUT/intrinsic-knn.json"
stage1_cmd="$PY scripts/experiments/diffusion_paired_eval/paired_eval.py ${paired_args[*]} --split validation --seed 0 --device $DEVICE --checkpoint checkpoint-best.pt --max-windows $PAIRED_WINDOWS --out $OUT/paired.json"
stage2_cmd="$PY scripts/experiments/diffusion_generated_diversity/analyze.py ${paired_args[*]} --split validation --seeds $SEEDS --device $DEVICE --max-windows $DIV_WINDOWS --cross-check --intrinsic-residual $OUT/intrinsic-knn.json --out $OUT/diversity.json"

if [ "$DRY_RUN" = 1 ]; then
  echo "--- stage 0 (CPU, reference only)"; echo "$stage0_cmd"
  echo "--- stage 1 (GPU, paired across arms)"; echo "$stage1_cmd"
  echo "--- stage 2 (GPU, spread vs bias per arm)"; echo "$stage2_cmd"
  echo "all inputs present; nothing was run"
  exit 0
fi

run_stage() { # marker command
  if [ -s "$OUT/$1.done" ]; then echo "skip $1 (already done)"; return 0; fi
  echo "=== $1 start $(date -u +%FT%TZ)"
  eval "$2" > "$OUT/$1.log" 2>&1
  rc=$?
  echo "=== $1 exit=$rc $(date -u +%FT%TZ)" | tee -a "$OUT/$1.log"
  if [ "$rc" -eq 0 ]; then date -u +%FT%TZ > "$OUT/$1.done"; fi
  return $rc
}

run_stage intrinsic-knn "$stage0_cmd" || echo "WARN: intrinsic baseline failed; diversity will report null spread ratios"
run_stage paired "$stage1_cmd" || { echo "ABORT: paired evaluation failed"; exit 3; }
run_stage diversity "$stage2_cmd" || { echo "ABORT: diversity measurement failed"; exit 4; }

summary_args=()
for a in "${arms[@]}"; do summary_args+=(--run "$a=$train/$a"); done
run_stage summary "$PY scripts/experiments/diffusion_wave_summary/report.py ${summary_args[*]} --paired $OUT/paired.json --reference $reference --out $OUT/summary.json" || { echo "ABORT: summary failed"; exit 5; }

echo "=== evaluation complete: $OUT"
ls -la "$OUT" | tail -8
