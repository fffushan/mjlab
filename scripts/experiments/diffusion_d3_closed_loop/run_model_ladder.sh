#!/bin/bash
# Run the D3 parallel fall sweep for several checkpoints in waves.
#
# One wave is one checkpoint.  Inside a wave every listed motion gets its own
# process on its own GPU, so a wave takes about as long as one motion.  With a
# single GPU the motions run one after another instead, which is the laptop
# case.
#
# Usage:
#   run_model_ladder.sh [options] <label>:<checkpoint.pt> [<label>:<checkpoint.pt> ...]
#
# Options:
#   --python PATH        interpreter, one path only, no arguments
#                        (default: .venv/bin/python when it exists, else python)
#   --driver PATH        evaluate_parallel.py (default: beside this script)
#   --output-root DIR    evidence root (default: logs/diagnostics/diffusion-d3-models)
#   --motions LIST       comma-separated cohort members (default: vgk000..vgk007)
#   --gpus LIST          comma-separated GPU ids (default: 0,1,2,3,4,5,6,7)
#   --num-envs N         environment rows per motion (default: 256)
#   --steps N            override the per-motion tick budget (smoke runs)
#   --extra "ARGS"       extra driver arguments, for example "--parity --parity-rows 8"
#   --summarize          run summarize_models.py on the evidence root afterwards
#   --dry-run            print the commands without running them
#
# Environment: PYTHONPATH is honoured and must point at the source tree when the
# code lives outside the checked-out tree (the reduced-contract worktree case).

set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python)}
DRIVER="$HERE/evaluate_parallel.py"
OUTPUT_ROOT=${OUTPUT_ROOT:-logs/diagnostics/diffusion-d3-models}
MOTIONS=${MOTIONS:-vgk000,vgk001,vgk002,vgk003,vgk004,vgk005,vgk006,vgk007}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NUM_ENVS=${NUM_ENVS:-256}
STEPS=${STEPS:-}
EXTRA=${EXTRA:-}
DRY_RUN=0
SUMMARIZE=0
ENTRIES=()

while [ $# -gt 0 ]; do
  case "$1" in
    --python) PYTHON="$2"; shift 2 ;;
    --driver) DRIVER="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --motions) MOTIONS="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --num-envs) NUM_ENVS="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --extra) EXTRA="$2"; shift 2 ;;
    --summarize) SUMMARIZE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) ENTRIES+=("$1"); shift ;;
  esac
done

if [ "${#ENTRIES[@]}" -eq 0 ]; then
  echo "no <label>:<checkpoint> argument given; see --help" >&2
  exit 2
fi

IFS=',' read -r -a MOTION_LIST <<< "$MOTIONS"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
GPU_COUNT=${#GPU_LIST[@]}
if [ "$GPU_COUNT" -eq 0 ]; then
  echo "no GPU given" >&2
  exit 2
fi

STEP_ARG=()
if [ -n "$STEPS" ]; then
  STEP_ARG=(--steps "$STEPS")
fi
EXTRA_ARG=()
if [ -n "$EXTRA" ]; then
  # shellcheck disable=SC2206  # intentional word splitting for a small argument list
  EXTRA_ARG=($EXTRA)
fi

echo "=== host $(hostname), $(date '+%H:%M:%S %Z') ==="
echo "=== interpreter $PYTHON ==="
echo "=== driver $DRIVER ==="
echo "=== ${#ENTRIES[@]} wave(s), ${#MOTION_LIST[@]} motion(s) each, $NUM_ENVS rows, ${GPU_COUNT} GPU(s) ==="

overall=0
for entry in "${ENTRIES[@]}"; do
  label="${entry%%:*}"
  checkpoint="${entry#*:}"
  if [ "$label" = "$checkpoint" ]; then
    echo "entry '$entry' is not <label>:<checkpoint>" >&2
    exit 2
  fi
  echo "=== WAVE $label  $(date '+%H:%M:%S') ==="
  if [ ! -f "$checkpoint" ]; then
    if [ "$DRY_RUN" -eq 1 ]; then
      echo "  (dry-run: checkpoint not checked)"
    else
      echo "  MISSING checkpoint $checkpoint"
      overall=1
      continue
    fi
  fi
  mkdir -p "$OUTPUT_ROOT/$label"
  pids=()
  wave=0
  index=0
  for motion in "${MOTION_LIST[@]}"; do
    gpu="${GPU_LIST[$((index % GPU_COUNT))]}"
    index=$((index + 1))
    command=(
      env "CUDA_VISIBLE_DEVICES=$gpu" "$PYTHON" "$DRIVER"
      --checkpoint "$checkpoint"
      --members "$motion"
      --num-envs "$NUM_ENVS"
      --execute
      --output-dir "$OUTPUT_ROOT/$label/$motion"
      "${STEP_ARG[@]}" "${EXTRA_ARG[@]}"
    )
    log="$OUTPUT_ROOT/$label/$motion.run.log"
    if [ "$DRY_RUN" -eq 1 ]; then
      echo "  CUDA_VISIBLE_DEVICES=$gpu ${command[*]:3} > $log"
      continue
    fi
    if [ "$GPU_COUNT" -eq 1 ]; then
      # One card: run the motions one after another so they do not share VRAM.
      if ! "${command[@]}" > "$log" 2>&1; then
        wave=1
        echo "  FAILED $motion (see $log)"
        tail -3 "$log"
      fi
    else
      nohup "${command[@]}" > "$log" 2>&1 &
      pids+=($!)
    fi
  done
  if [ "$GPU_COUNT" -ne 1 ] && [ "${#pids[@]}" -gt 0 ]; then
    for i in "${!pids[@]}"; do
      if ! wait "${pids[$i]}"; then
        wave=1
        echo "  FAILED ${MOTION_LIST[$i]} (see $OUTPUT_ROOT/$label/${MOTION_LIST[$i]}.run.log)"
      fi
    done
  fi
  [ "$wave" -ne 0 ] && overall=1
  if [ "$DRY_RUN" -eq 0 ]; then
    for motion in "${MOTION_LIST[@]}"; do
      grep -E "^\[${motion}\]" "$OUTPUT_ROOT/$label/$motion.run.log" 2>/dev/null | tail -1
    done
  fi
  echo "=== WAVE $label exit=$wave ==="
done

if [ "$SUMMARIZE" -eq 1 ] && [ "$DRY_RUN" -eq 0 ]; then
  "$PYTHON" "$HERE/summarize_models.py" || true
fi

echo "=== ladder done, overall exit $overall, $(date '+%H:%M:%S') ==="
exit "$overall"
