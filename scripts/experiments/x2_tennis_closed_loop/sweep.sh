#!/bin/bash
# Run one closed-loop latency cell of the X2 juggling sweep per cell directory.
#
# Thin loop only: every cell is one run_sweep.py invocation, one output directory
# (result.json, run.log, exit-status) and nothing else. This script applies no wall
# bound, no GPU check and no retry; the caller owns those. Bound the whole sweep
# externally and check GPU occupancy before starting it.
#
# A cell is one MIN:MAX pair of stream latencies in seconds. The observation term
# quantises latency to whole control steps of 20 ms, so cells inside one rounding
# bin are the same condition; see README.md.
#
# Usage, from the repository root:
#   bash scripts/experiments/x2_tennis_closed_loop/sweep.sh \
#     --task TASK --checkpoint PATH --target 0.40 \
#     --cells "0.00:0.00 0.04:0.04 0.08:0.08" --output-root DIR
#
# Options:
#   --task TASK           required, mjlab task id
#   --checkpoint PATH     required, policy checkpoint
#   --target METRES       required, target apex height. Export
#                         MJLAB_JUGGLE_ENERGY_TARGET_HEIGHT_M unless it is 0.40.
#   --cells "MIN:MAX ..." required, one cell per space-separated pair, seconds
#   --output-root DIR     required, one subdirectory per cell
#   --horizon-s SECONDS   default 180
#   --num-envs N          default 32
#   --device DEVICE       default cuda:0
#
# Exit status:
#   0    every cell finished and wrote result.json
#   10   bad arguments or a missing required option
#   11   a cell directory already exists; nothing is overwritten
#   *    the failing run_sweep.py status; the sweep stops at that cell
# A cell directory left behind by a failed cell is kept on purpose: it holds the
# log that explains the failure, and rerunning the sweep refuses to clobber it.

set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNNER=$SCRIPT_DIR/run_sweep.py

HORIZON_S=180
NUM_ENVS=32
DEVICE=cuda:0
TASK=""
CHECKPOINT=""
TARGET=""
CELLS=""
OUT_ROOT=""

usage() {
  sed -n '2,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
  esac
  if [ $# -lt 2 ]; then
    echo "missing value for $1" >&2
    exit 10
  fi
  case "$1" in
    --task) TASK=$2 ;;
    --checkpoint) CHECKPOINT=$2 ;;
    --target) TARGET=$2 ;;
    --cells) CELLS=$2 ;;
    --output-root) OUT_ROOT=$2 ;;
    --horizon-s) HORIZON_S=$2 ;;
    --num-envs) NUM_ENVS=$2 ;;
    --device) DEVICE=$2 ;;
    *)
      echo "unknown argument: $1" >&2
      exit 10
      ;;
  esac
  shift 2
done

for name in TASK CHECKPOINT TARGET CELLS OUT_ROOT; do
  if [ -z "${!name}" ]; then
    echo "missing required option for $name" >&2
    exit 10
  fi
done

if [ ! -f "$RUNNER" ]; then
  echo "missing runner: $RUNNER" >&2
  exit 10
fi

mkdir -p "$OUT_ROOT" || exit 10

for cell in $CELLS; do
  case "$cell" in
    *:*) ;;
    *)
      echo "cell must be MIN:MAX seconds, got '$cell'" >&2
      exit 10
      ;;
  esac
  MIN=${cell%%:*}
  MAX=${cell##*:}
  if [ -z "$MIN" ] || [ -z "$MAX" ] || [ "$MAX" = "$cell" ]; then
    echo "cell must be MIN:MAX seconds, got '$cell'" >&2
    exit 10
  fi

  OUT=$OUT_ROOT/lat_${MIN}_${MAX}
  if [ -e "$OUT" ]; then
    echo "refuse to overwrite existing cell directory: $OUT" >&2
    exit 11
  fi
  mkdir -p "$OUT" || exit 10
  # The log is staged beside the cell directory and moved in afterwards, because a
  # file created inside it before the runner starts would trip the runner's own
  # refusal to write into a non-empty directory.
  CELL_LOG=$OUT_ROOT/.lat_${MIN}_${MAX}.log

  echo "CELL-START $(date '+%F %T %Z') cell=$cell out=$OUT"
  uv run python "$RUNNER" \
    --task "$TASK" \
    --checkpoint "$CHECKPOINT" \
    --latency-min "$MIN" \
    --latency-max "$MAX" \
    --target "$TARGET" \
    --horizon-s "$HORIZON_S" \
    --num-envs "$NUM_ENVS" \
    --device "$DEVICE" \
    --output-dir "$OUT" >"$CELL_LOG" 2>&1
  status=$?
  mv "$CELL_LOG" "$OUT/run.log"
  echo "$status" >"$OUT/exit-status"
  echo "CELL-END $(date '+%F %T %Z') cell=$cell status=$status"
  if [ "$status" -ne 0 ]; then
    echo "cell $cell failed with status $status; stopping the sweep" >&2
    exit "$status"
  fi
done

echo "SWEEP-DONE $(date '+%F %T %Z') cells=$(echo "$CELLS" | wc -w) root=$OUT_ROOT"
