#!/bin/bash
# Run the eight E0 evaluation arms sequentially under one wall-clock bound.
#
# E0 is 256 deterministic episodes: six final policies (0.20/0.40 m x seeds
# 7/17/27) and one zero-action control per target, 32 episodes each. The whole
# batch, including interpreter and CUDA startup, is bounded by BATCH_WALL_S.
# There is no per-arm timeout and no automatic rerun.
#
# What this script guarantees, and how to read the result:
#   * One GPU, rechecked for occupancy immediately before any arm runs. A busy GPU
#     fails the batch closed with exit 10; nothing is launched.
#   * A single owned batch. A second invocation refuses to start while the first
#     holds the lock (exit 11), and no process outside this batch is ever killed.
#   * A fresh output directory. Existing artifacts are never overwritten (exit 12).
#   * Per-arm logs, per-arm exit status, and one batch exit status.
#   * On the wall limit the owned python process is stopped and the artifacts of
#     the arms that already finished are kept.
#
# Distinguishing a transport failure from an evaluator failure: the batch writes
# <out>/exit-status on every exit path it can observe. If that file is missing, the
# batch never reached its cleanup - the SSH session, the host, or the container died
# - and the run status is UNKNOWN, not success. A present exit-status of 143 means
# TERM was received (including the wall limit); outer timeout returns 124 on expiry.
# Read <out>/arm-exit-status/<arm> and <out>/logs/<arm>.log for per-arm evidence.
#
# Exit status:
#   0    all eight arms finished and the summary was written
#   10   preflight refused: the requested GPU was busy, or a required path is absent
#   11   duplicate guard: another batch holds the lock, or a stale lock is present
#   12   refuse to overwrite existing artifacts
#   143  inner batch received TERM (wall timeout's outer status is 124)
#   20   at least one arm failed
#   21   the arms' recorded release states disagree, so they are not comparable
#   22   the final summary could not be written
# A preflight refusal also appends a line to <stage>/preflight-fail, because the
# refusal may happen before <out>/exit-status can be written safely.
#
# Usage (the parent runs the launch command through its monitor):
#   GPU=0 bash run_batch.sh
# Environment overrides, all defaulted: STAGE, OUT, CODE, SRC, JOBS, PY, GPU,
# BATCH_WALL_S.

set -u

STAGE=${STAGE:-/mnt/fushan/mjlab-juggle-eval-20261105}
CODE=${CODE:-$STAGE/code}
OUT=${OUT:-$STAGE/out}
SRC=${SRC:-/mnt/fushan/mjlab-juggle-nominal-20261104/src}
JOBS=${JOBS:-/mnt/fushan/mjlab-juggle-nominal-20261104/jobs}
PY=${PY:-/home/fushan/mjlab/.venv/bin/python}
GPU=${GPU:-0}
BATCH_WALL_S=${BATCH_WALL_S:-1800}
case "$BATCH_WALL_S" in
  ''|*[!0-9]*) echo "invalid batch wall limit" >&2; exit 10 ;;
esac
if [ "$BATCH_WALL_S" -lt 1 ] || [ "$BATCH_WALL_S" -gt 1800 ]; then
  echo "batch wall limit must be within the authorized 1..1800 seconds" >&2
  exit 10
fi

EXP=agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal
STAMP_S7_S17=2026-11-04_23-35-40
STAMP_S27=2026-11-04_23-35-39
ARM_NAMES="t020-s7 t020-s17 t020-s27 t040-s7 t040-s17 t040-s27 zero-t020 zero-t040"

LOCK=$STAGE/batch.lock
PROGRESS=$OUT/progress.jsonl
ARM_PID=""
DONE=0
OUTPUT_OWNED=0

# One wall-clock bound for the whole batch. The inner invocation does the work; the
# outer one only owns the limit, so no per-arm timer can reset the budget.
if [ "${1:-}" != "--inner" ]; then
  echo "BATCH-START $(date '+%F %T %Z') stage=$STAGE gpu=$GPU wall_s=$BATCH_WALL_S"
  exec timeout --signal=TERM --kill-after=60 "$BATCH_WALL_S" bash "$0" --inner
fi

fail_closed() {
  # Preflight refusal: recorded outside the output directory, which may not be
  # writable yet or may belong to a previous run.
  local rc=$1
  shift
  echo "PREFLIGHT-FAIL $*" >&2
  printf '%s rc=%s %s\n' "$(date -u '+%F %T')" "$rc" "$*" >> "$STAGE/preflight-fail" \
    2> /dev/null
  exit "$rc"
}

progress() {
  # One JSON line per event, appended and echoed so a log follower sees it too.
  local line
  line="{\"utc\":\"$(date -u '+%F %T')\",$1}"
  printf '%s\n' "$line" | tee -a "$PROGRESS"
}

finish() {
  local rc=$?
  [ "$DONE" = "1" ] && return
  DONE=1
  trap - EXIT TERM INT
  if [ -n "$ARM_PID" ] && kill -0 "$ARM_PID" 2> /dev/null; then
    kill -TERM "$ARM_PID" 2> /dev/null
    wait "$ARM_PID" 2> /dev/null
  fi
  if [ -f "$LOCK/pid" ] && [ "$(cat "$LOCK/pid" 2> /dev/null)" = "$$" ]; then
    rm -rf "$LOCK"
  fi
  if [ "$OUTPUT_OWNED" = "1" ] && [ -d "$OUT" ]; then
    progress "\"event\":\"batch-end\",\"rc\":$rc"
    printf '%s\n' "$rc" > "$OUT/exit-status"
  fi
  if [ "$rc" = "0" ]; then
    echo "BATCH-RESULT ok rc=0"
  else
    echo "BATCH-RESULT failed rc=$rc"
  fi
  exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# arm_run_dir prints the training run directory of a policy arm.
arm_run_dir() {
  case "$1" in
    *-s27) printf '%s/%s/run/%s/%s_%s\n' "$JOBS" "$1" "$EXP" "$STAMP_S27" "$1" ;;
    *) printf '%s/%s/run/%s/%s_%s\n' "$JOBS" "$1" "$EXP" "$STAMP_S7_S17" "$1" ;;
  esac
}

# arm_saved_env_yaml prints the saved training config an arm validates against. The
# two zero-action controls have no training run, so each validates against the first
# policy arm of its own target: the configs differ only in seed, which is an allowed
# difference.
arm_saved_env_yaml() {
  case "$1" in
    zero-t020) printf '%s/params/env.yaml\n' "$(arm_run_dir t020-s7)" ;;
    zero-t040) printf '%s/params/env.yaml\n' "$(arm_run_dir t040-s7)" ;;
    *) printf '%s/params/env.yaml\n' "$(arm_run_dir "$1")" ;;
  esac
}

arm_target() {
  case "$1" in
    t020-* | zero-t020) printf '%s\n' 0.20 ;;
    t040-* | zero-t040) printf '%s\n' 0.40 ;;
  esac
}

# Preflight: paths. Nothing is written before these pass.
for path in "$PY" "$CODE/evaluate_arm.py" "$CODE/summarize_batch.py" \
  "$CODE/recorder.py" "$CODE/scoring.py" "$CODE/config_check.py" \
  "$SRC/mjlab/__init__.py"; do
  if [ ! -e "$path" ]; then
    fail_closed 10 "missing: $path"
  fi
done
if [ ! -x "$PY" ]; then
  fail_closed 10 "interpreter is not executable: $PY"
fi

# Preflight: every checkpoint and every saved config must exist and be non-empty.
for arm in $ARM_NAMES; do
  case "$arm" in
    zero-*) ;;
    *)
      ckpt="$(arm_run_dir "$arm")/model_4999.pt"
      if [ ! -s "$ckpt" ]; then
        fail_closed 10 "missing checkpoint: $ckpt"
      fi
      ;;
  esac
  saved=$(arm_saved_env_yaml "$arm")
  if [ ! -s "$saved" ]; then
    fail_closed 10 "missing saved config: $saved"
  fi
done

# Preflight: refuse to overwrite a previous run.
if [ -e "$OUT/exit-status" ] || [ -e "$PROGRESS" ]; then
  fail_closed 12 "refusing to overwrite an existing batch in $OUT"
fi
mkdir -p "$OUT/logs" "$OUT/arms" "$OUT/arm-exit-status" \
  || fail_closed 10 "cannot create the output tree under $OUT"

# Preflight: one batch only. mkdir is the atomic test-and-set.
if ! mkdir "$LOCK" 2> /dev/null; then
  holder=$(cat "$LOCK/pid" 2> /dev/null || echo unknown)
  if [ "$holder" != "unknown" ] && kill -0 "$holder" 2> /dev/null; then
    fail_closed 11 "another batch is running with pid $holder"
  fi
  fail_closed 11 "stale lock at $LOCK (holder pid $holder is not alive); remove it by hand after checking that no batch is running"
fi
printf '%s\n' "$$" > "$LOCK/pid"
date -u '+%F %T' > "$LOCK/started"
# Recheck under the lock. A prior batch may have finished after the first check.
if [ -e "$OUT/exit-status" ] || [ -e "$PROGRESS" ]; then
  fail_closed 12 "refusing to overwrite an existing batch in $OUT"
fi
OUTPUT_OWNED=1

# Preflight: the requested GPU must be idle now, checked immediately before any arm
# runs. Fail closed: a busy GPU is a refusal, not something to wait out.
if ! command -v nvidia-smi > /dev/null 2>&1; then
  fail_closed 10 "nvidia-smi is not available"
fi
gpu_row=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits | awk -F', ' -v gpu="$GPU" '$1 == gpu {print}')
if [ -z "$gpu_row" ]; then
  fail_closed 10 "GPU $GPU is not present on this host"
fi
mem_used=$(printf '%s' "$gpu_row" | awk -F', ' '{print $2}')
util=$(printf '%s' "$gpu_row" | awk -F', ' '{print $3}')
gpu_uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
  | awk -F', ' -v gpu="$GPU" '$1 == gpu {print $2}')
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name \
  --format=csv,noheader | awk -F', ' -v uuid="$gpu_uuid" '$1 == uuid {print}')
if [ -n "$apps" ]; then
  fail_closed 10 "GPU $GPU already has compute processes: $(printf '%s' "$apps")"
fi
if [ "$mem_used" -gt 512 ] || [ "$util" -gt 0 ]; then
  fail_closed 10 "GPU $GPU is not idle: ${mem_used} MiB used, ${util}% util"
fi
echo "PREFLIGHT-OK gpu=$GPU memory_used=${mem_used}MiB util=${util}pct"

progress "\"event\":\"preflight-ok\",\"gpu\":$GPU,\"wall_s\":$BATCH_WALL_S"

run_arm() {
  local arm=$1 target=$2
  local log=$OUT/logs/$arm.log
  local saved rc
  local -a extra=()
  saved=$(arm_saved_env_yaml "$arm")
  if [ "${arm#zero-}" != "$arm" ]; then
    extra+=(--zero-action)
  else
    extra+=(--checkpoint "$(arm_run_dir "$arm")/model_4999.pt")
  fi

  progress "\"event\":\"arm-start\",\"arm\":\"$arm\",\"target\":$target"
  (
    # Only the arm's energy target may reach the config. Anything else inherited
    # from the login shell would silently change the task.
    for name in $(env | sed -n 's/^\(MJLAB_[A-Z0-9_]*\)=.*/\1/p'); do
      unset "$name"
    done
    export MJLAB_JUGGLE_ENERGY_TARGET_HEIGHT_M="$target"
    export PYTHONPATH="$SRC"
    export PYTHONUNBUFFERED=1
    export MPLCONFIGDIR="$STAGE/cache/mpl"
    export CUDA_VISIBLE_DEVICES="$GPU"
    mkdir -p "$MPLCONFIGDIR"
    echo "[$arm] start $(date '+%F %T %Z') host $(hostname) gpu $GPU target $target"
    exec "$PY" "$CODE/evaluate_arm.py" \
      --arm "$arm" \
      --target "$target" \
      --saved-env-yaml "$saved" \
      --output-dir "$OUT/arms/$arm" \
      --device cuda:0 \
      --expect-source-root "$SRC" \
      ${extra[@]+"${extra[@]}"}
  ) >> "$log" 2>&1 &
  ARM_PID=$!
  wait "$ARM_PID"
  rc=$?
  ARM_PID=""
  printf '%s\n' "$rc" > "$OUT/arm-exit-status/$arm"
  progress "\"event\":\"arm-end\",\"arm\":\"$arm\",\"rc\":$rc"
  if [ "$rc" != "0" ]; then
    echo "[$arm] FAILED rc=$rc, see $log" >&2
  fi
  return 0
}

batch_rc=0
for arm in $ARM_NAMES; do
  run_arm "$arm" "$(arm_target "$arm")"
  status=$(cat "$OUT/arm-exit-status/$arm" 2> /dev/null || echo 99)
  if [ "$status" != "0" ] && [ "$batch_rc" = "0" ]; then
    batch_rc=20
  fi
done

progress "\"event\":\"summary-start\""
"$PY" "$CODE/summarize_batch.py" --batch-dir "$OUT" 2>&1 | tee "$OUT/logs/summary.log"
summary_rc=${PIPESTATUS[0]}
progress "\"event\":\"summary-end\",\"rc\":$summary_rc"
if [ "$summary_rc" = "21" ] && [ "$batch_rc" = "0" ]; then
  batch_rc=21
elif [ "$summary_rc" != "0" ] && [ "$batch_rc" = "0" ]; then
  batch_rc=22
fi

exit "$batch_rc"
