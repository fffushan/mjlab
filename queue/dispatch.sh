#!/usr/bin/env bash
# GPU queue dispatcher: runs pending jobs from jobs.tsv on free GPUs, at most
# MAX_CONCURRENT at a time and never more than ONE job per GPU.
# Never evicts a running job; only uses GPUs with no compute process.
set -uo pipefail
cd /home/fushan/mjlab

QUEUE_DIR="${QUEUE_DIR:-/home/fushan/mjlab/queue}"
JOBS="$QUEUE_DIR/jobs.tsv"
STATE_DIR="$QUEUE_DIR/state"
LOG="$QUEUE_DIR/queue.log"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
MEM_FREE_MIB="${MEM_FREE_MIB:-1000}"
# GPUs to never use (e.g. a device with degraded DMA: see queue.log notes).
EXCLUDE_GPUS="${EXCLUDE_GPUS:-}"
LAUNCH_ROOT="$QUEUE_DIR/runs"
LOG_ROOT="${LOG_ROOT:-/home/fushan/mjlab/runs}"
NUM_ENVS="${NUM_ENVS:-8192}"

mkdir -p "$STATE_DIR" "$LAUNCH_ROOT"
exec >> "$LOG" 2>&1
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

exec 9>"$QUEUE_DIR/dispatch.lock"
if ! flock -n 9; then log "another dispatcher holds the lock; exiting"; exit 0; fi

[ -f "$JOBS" ] || { log "FATAL: missing $JOBS"; exit 1; }

set_state() { echo "$2" > "$STATE_DIR/$1.state"; }
state_of() { local f="$STATE_DIR/$1.state"; [ -f "$f" ] && cat "$f" || echo pending; }
session_alive() { tmux has-session -t "$1" 2>/dev/null; }
batch_dir_of() { cat "$QUEUE_DIR/batch.$1" 2>/dev/null; }
gpu_mem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' '; }

# A GPU is free when the driver reports no compute process and almost no memory used.
gpu_free() {
  local gpu="$1" uuid mem busy
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$gpu" | sed 's/[[:space:]]*$//')
  mem=$(gpu_mem "$gpu")
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sed 's/[[:space:]]*$//')
  [ "${mem:-99999}" -lt "$MEM_FREE_MIB" ] && ! grep -qxF "$uuid" <<<"$busy"
}

gpu_excluded() {
  local gpu="$1" e
  for e in ${EXCLUDE_GPUS//,/ }; do [ "$e" = "$gpu" ] && return 0; done
  return 1
}

list_free_gpus() {
  local n i
  n=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  for ((i = 0; i < n; i++)); do
    gpu_excluded "$i" && continue
    gpu_free "$i" && echo "$i"
  done
}

launch_job() {
  local id="$1" session="$2" command="$3" gpu="$4"
  local bdir="$LAUNCH_ROOT/$id-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$bdir"
  cat > "$bdir/run_one.sh" <<'HDR'
#!/usr/bin/env bash
set -uo pipefail
cd /home/fushan/mjlab
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export GPU_ID=__GPU__
export NUM_ENVS=__ENVS__
export LOG_ROOT=__LOG_ROOT__
echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu=__GPU__"
HDR
  sed -i "s|__GPU__|$gpu|g; s|__ENVS__|$NUM_ENVS|g; s|__LOG_ROOT__|$LOG_ROOT|g" "$bdir/run_one.sh"
  printf '%s\n' "$command" >> "$bdir/run_one.sh"
  cat >> "$bdir/run_one.sh" <<'FTR'
status=$?
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ) exit_status=$status"
FTR
  printf 'echo $status > "%s/exit.status"\nexit $status\n' "$bdir" >> "$bdir/run_one.sh"
  chmod +x "$bdir/run_one.sh"
  bash -n "$bdir/run_one.sh" || { log "job $id: runner syntax check FAILED"; return 1; }
  printf '%s\t%s\t%s\n' "$gpu" "$session" "$command" >> "$bdir/jobs.tsv"
  echo "$bdir" > "$QUEUE_DIR/batch.$id"
  tmux new-session -d -s "$session" "bash $bdir/run_one.sh > $bdir/stdout.log 2>&1"
  return 0
}

# Launch the first pending job on exactly this GPU. Returns 0 when it started.
launch_one_on_gpu() {
  local gpu="$1" row id session command
  gpu_excluded "$gpu" && return 1
  gpu_free "$gpu" || return 1
  for row in "${ROWS[@]}"; do
    id=$(cut -f1 <<<"$row")
    session=$(cut -f2 <<<"$row")
    command=$(cut -f3 <<<"$row")
    [ "$(state_of "$id")" = pending ] || continue
    session_alive "$session" && { log "job $id: session $session exists; skipping"; continue; }
    launch_job "$id" "$session" "$command" "$gpu" || return 1
    set_state "$id" running
    log "job $id started on GPU $gpu (session=$session)"
    return 0
  done
  return 1
}

log "dispatcher start: max_concurrent=$MAX_CONCURRENT poll=${POLL_SECONDS}s num_envs=$NUM_ENVS log_root=$LOG_ROOT exclude_gpus=${EXCLUDE_GPUS:-none}"

while true; do
  mapfile -t ROWS < <(grep -v '^[[:space:]]*#' "$JOBS" | grep -v '^[[:space:]]*$')
  [ "${#ROWS[@]}" -eq 0 ] && { log "jobs.tsv empty; retrying"; sleep "$POLL_SECONDS"; continue; }
  running=0
  pending=0
  for row in "${ROWS[@]}"; do
    id=$(cut -f1 <<<"$row"); session=$(cut -f2 <<<"$row")
    case "$(state_of "$id")" in
      running)
        if session_alive "$session"; then
          running=$((running + 1))
        else
          bdir=$(batch_dir_of "$id"); code="unknown"
          [ -n "$bdir" ] && [ -f "$bdir/exit.status" ] && code=$(cat "$bdir/exit.status")
          if [ "$code" = "0" ]; then set_state "$id" done; log "job $id finished OK ($bdir)"
          else set_state "$id" failed; log "job $id FAILED exit=$code ($bdir)"; fi
        fi
        ;;
      pending) pending=$((pending + 1)) ;;
    esac
  done


  if [ "$running" -lt "$MAX_CONCURRENT" ]; then
    declare -A assigned=()
    for gpu in $(list_free_gpus); do
      [ "$running" -lt "$MAX_CONCURRENT" ] || break
      [ -n "${assigned[$gpu]:-}" ] && continue
      if launch_one_on_gpu "$gpu"; then
        running=$((running + 1))
        assigned[$gpu]=1
        # Wait until this GPU actually shows work before scanning again, so the
        # same device can never be handed to a second job in this round.
        for _ in $(seq 1 18); do
          sleep 10
          [ "$(gpu_mem "$gpu")" -gt "$MEM_FREE_MIB" ] && break
          gpu_free "$gpu" || break
        done
        sleep 5
      fi
    done
  fi

  sleep "$POLL_SECONDS"
done
