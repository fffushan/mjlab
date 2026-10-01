#!/usr/bin/env bash
# Local monitor for the remote GPU task queue (the mjlab queue dispatcher).
#
#   queue_status.sh [OPTIONS] 'ssh -p PORT user@host'
#
# The remote is given as the ssh command you would paste, quoted as ONE argument:
#   queue_status.sh 'ssh -p 31505 fushan@10.14.64.37'
# The leading 'ssh' is optional; without -p the port defaults to 22.
#
# Queue mutations:
#   --add JOB_ID REMOTE_SPEC -- RUN_TASK_ARG...
#   --remove JOB_ID REMOTE_SPEC
# Everything after the add separator is forwarded to run_task.sh unchanged.
#
# The queue itself lives on the remote server in ~/mjlab/queue/:
#   dispatch.sh   the polling dispatcher (runs in tmux session 'queue-dispatch')
#   jobs.tsv      one job per row: id<TAB>session<TAB>command
#   status.sh     queue table + GPU occupancy
#   stop.sh       stop queueing (running jobs keep going)
#   state/        per-job state files (pending|running|done|failed)
#   queue.log     dispatcher log
#
# This wrapper is read-only by default: it just SSHes in and runs the remote
# status.sh, so it works even if the queue files were never copied locally.
#
#   --add ID REMOTE -- RUN_TASK_ARG...
#                append a job running /home/fushan/mjlab/run_task.sh with exactly
#                the forwarded arguments; its session is train_ID
#   --remove ID  remove a non-running job row; state/batch artifacts are retained
#   --stop       run the remote stop.sh instead (idempotent: safe to re-run)
#   --jobs       also print the remote jobs.tsv (the queued job list)
#   --tail N     also print the last N lines of the remote queue.log
#   --dispatch   also print dispatcher/tmux/process status
#
# Exit status: 0 ok, 2 usage error, 3 SSH/scan failure.

set -uo pipefail

MODE=status
SHOW_JOBS=0
SHOW_DISPATCH=0
LOG_TAIL=0
RESTART_EXCLUDE=""
ADD_ID=""
ADD_ARGS=()
ADD_COMMAND_B64=""
REMOVE_ID=""
REMOTE_SPEC=""
REMOTE=""
PORT=22
SCRIPT_NAME=$(basename "$0")

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS] 'ssh -p PORT user@host'

  REMOTE SPEC   The ssh command to the server, quoted as one argument:
                  'ssh -p 31505 fushan@10.14.64.37'
                The leading 'ssh' is optional; without -p the port defaults to 22.

  --add ID REMOTE -- ARGS...
                Add a job. ARGS are forwarded verbatim to run_task.sh.
                REMOTE is the quoted SSH spec; -- separates it from ARGS.
  --remove ID   Remove a non-running job row. Running jobs are refused.
  --stop        Run the remote queue stop.sh (running jobs keep going).
  --restart     Restart only the dispatcher; running jobs keep going.
  --exclude-gpus LIST  Dispatcher exclusion list for --restart (use 'none' for all GPUs).
  --tail N      Also print the last N lines of the remote queue.log.
  --dispatch    Also print tmux/process status for the dispatcher.

Read-only by default: prints the remote queue table + GPU occupancy.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --add)
      if [ "$MODE" != status ]; then
        echo "error: --add cannot be combined with another queue action" >&2; exit 2
      fi
      if [ $# -lt 4 ] || [ "$4" != "--" ]; then
        echo "error: --add syntax is --add ID 'ssh ...' -- RUN_TASK_ARG..." >&2; exit 2
      fi
      MODE=add
      ADD_ID="$2"
      REMOTE_SPEC="$3"
      ADD_ARGS=("${@:5}")
      # The remaining arguments belong to run_task.sh, not this wrapper.
      shift $(( $# )) ;;
    --remove)
      if [ "$MODE" != status ]; then
        echo "error: --remove cannot be combined with another queue action" >&2; exit 2
      fi
      if [ $# -lt 3 ]; then
        echo "error: --remove needs a job ID and a remote spec" >&2; exit 2
      fi
      MODE=remove; REMOVE_ID="$2"; shift 2 ;;
    --stop)
      if [ "$MODE" != status ]; then
        echo "error: --stop cannot be combined with another queue action" >&2; exit 2
      fi
      MODE=stop; shift ;;
    --restart)
      if [ "$MODE" != status ]; then
        echo "error: --restart cannot be combined with another queue action" >&2; exit 2
      fi
      MODE=restart; shift ;;
    --exclude-gpus)
      if [ "$MODE" != restart ] || [ $# -lt 2 ]; then
        echo "error: --exclude-gpus is only valid after --restart" >&2; exit 2
      fi
      RESTART_EXCLUDE="$2"
      shift 2 ;;
    --jobs) SHOW_JOBS=1; shift ;;
    --tail)
      LOG_TAIL="${2:?--tail needs a number of lines}"
      if ! [[ "$LOG_TAIL" =~ ^[0-9]+$ ]]; then
        echo "error: --tail wants a number of lines, got '$LOG_TAIL'" >&2
        exit 2
      fi
      shift 2 ;;
    --dispatch) SHOW_DISPATCH=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -p|-p[0-9]*|-p\ *)
      if [ -z "$REMOTE_SPEC" ]; then
        REMOTE_SPEC="$*"; shift $#
      else
        echo "error: unexpected argument '$1'" >&2; exit 2
      fi ;;
    -*) echo "error: unknown option '$1'" >&2
        echo "hint: the remote goes in ONE quoted argument, e.g. 'ssh -p 31505 user@host'" >&2
        exit 2 ;;
    *) if [ -n "$REMOTE_SPEC" ]; then
         echo "error: unexpected extra argument '$1' (quote the whole ssh command as one argument)" >&2
         exit 2
       fi
       REMOTE_SPEC="$1"; shift ;;
  esac
done

if [ "$MODE" = add ]; then
  if ! [[ "$ADD_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "error: invalid job ID '$ADD_ID'" >&2; exit 2
  fi
  ADD_COMMAND="bash /home/fushan/mjlab/run_task.sh"
  for arg in "${ADD_ARGS[@]}"; do
    printf -v quoted_arg '%q' "$arg"
    ADD_COMMAND+=" $quoted_arg"
  done
  ADD_COMMAND_B64=$(printf '%s' "$ADD_COMMAND" | base64 | tr -d '\n')
elif [ "$MODE" = remove ]; then
  if [ -z "$REMOVE_ID" ] || [[ "$REMOVE_ID" == *$'\t'* || "$REMOVE_ID" == *$'\n'* ]]; then
    echo "error: --remove needs a non-empty job ID without tabs/newlines" >&2; exit 2
  fi
  if ! [[ "$REMOVE_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "error: invalid job ID '$REMOVE_ID'" >&2; exit 2
  fi
fi

if [ -z "$REMOTE_SPEC" ]; then usage >&2; exit 2; fi

# ---------------------------------------------------------------------------
# Parse the pasted ssh command:  [ssh] [-p PORT] user@host
# Accepted forms (quoted as a single argument):
#   'ssh -p 31505 fushan@10.14.64.37'   'ssh -p31505 fushan@10.14.64.37'
#   'ssh fushan@10.14.64.37'            'fushan@10.14.64.37'   (port 22)
# ---------------------------------------------------------------------------
read -r -a SPEC_TOKENS <<< "$REMOTE_SPEC"
if [ "${#SPEC_TOKENS[@]}" -eq 0 ]; then
  echo "error: empty remote spec" >&2; exit 2
fi
spec_i=0
if [ "${SPEC_TOKENS[0]:-}" = "ssh" ]; then spec_i=1; fi
while [ "$spec_i" -lt "${#SPEC_TOKENS[@]}" ]; do
  tok="${SPEC_TOKENS[$spec_i]}"
  case "$tok" in
    -p)
      if [ $((spec_i + 1)) -ge "${#SPEC_TOKENS[@]}" ]; then
        echo "error: '-p' in remote spec needs a port value" >&2; exit 2
      fi
      PORT="${SPEC_TOKENS[$((spec_i + 1))]}"; spec_i=$((spec_i + 2)) ;;
    -p*)
      PORT="${tok#-p}"; spec_i=$((spec_i + 1)) ;;
    -*)
      echo "error: unsupported ssh option '$tok' in remote spec (only -p is handled)" >&2
      echo "hint: pass just 'ssh -p PORT user@host'" >&2; exit 2 ;;
    *)
      if [ -n "$REMOTE" ]; then
        echo "error: unexpected token '$tok' in remote spec" >&2; exit 2
      fi
      REMOTE="$tok"; spec_i=$((spec_i + 1)) ;;
  esac
done

if [ -z "$REMOTE" ]; then
  echo "error: no destination found in remote spec '$REMOTE_SPEC'" >&2; exit 2
fi
if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
  echo "error: invalid port '$PORT' in remote spec" >&2; exit 2
fi

SSH_CMD=(ssh -p "$PORT" -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE")

# ---------------------------------------------------------------------------
# The remote side: one SSH round trip.  Fails loudly if the queue is missing
# (e.g. server recreated after a reboot) so we never paper over that.
# ---------------------------------------------------------------------------
REMOTE_SCRIPT='
set -uo pipefail
QUEUE="${QUEUE:-$HOME/mjlab/queue}"
if [ ! -d "$QUEUE" ]; then
  echo "error: queue dir not found on remote: $QUEUE" >&2
  exit 3
fi
cd "$HOME/mjlab"

case "${MODE:-status}" in
  status)
    if [ -x "$QUEUE/status.sh" ]; then
      bash "$QUEUE/status.sh"
    else
      echo "queue dir exists but status.sh is missing: $QUEUE" >&2
      exit 3
    fi
    ;;
  add)
    if [ ! -f "$QUEUE/jobs.tsv" ]; then
      echo "error: jobs.tsv is missing: $QUEUE/jobs.tsv" >&2
      exit 3
    fi
    if ! ADD_COMMAND=$(printf '%s' "$ADD_COMMAND_B64" | base64 -d); then
      echo "error: could not decode forwarded run_task.sh arguments" >&2
      exit 3
    fi
    if [ -z "$ADD_COMMAND" ]; then
      echo "error: empty run_task.sh command" >&2
      exit 2
    fi
    session="train_$ADD_ID"
    exec 9>"$QUEUE/jobs.tsv.lock"
    flock -x 9
    if awk -F "\\t" -v id="$ADD_ID" "\$1 == id {found=1} END {exit !found}" "$QUEUE/jobs.tsv"; then
      echo "error: job ID already exists in jobs.tsv: $ADD_ID" >&2
      exit 4
    fi
    if [ -e "$QUEUE/state/$ADD_ID.state" ] || [ -e "$QUEUE/batch.$ADD_ID" ]; then
      echo "error: job ID has existing queue state/artifacts: $ADD_ID" >&2
      exit 4
    fi
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "error: tmux session already exists: $session" >&2
      exit 4
    fi
    printf "%s\t%s\t%s\n" "$ADD_ID" "$session" "$ADD_COMMAND" >> "$QUEUE/jobs.tsv"
    echo "job appended: $ADD_ID (session=$session)"
    bash "$QUEUE/status.sh"
    ;;
  remove)
    if [ ! -f "$QUEUE/jobs.tsv" ]; then
      echo "error: jobs.tsv is missing: $QUEUE/jobs.tsv" >&2
      exit 3
    fi
    state=$(cat "$QUEUE/state/$REMOVE_ID.state" 2>/dev/null || echo pending)
    case "$state" in
      running|started)
        echo "error: refusing to remove active job $REMOVE_ID (state=$state); running jobs are not stopped" >&2
        exit 4
        ;;
    esac
    exec 9>"$QUEUE/jobs.tsv.lock"
    flock -x 9
    tmp=$(mktemp "$QUEUE/jobs.tsv.remove.XXXXXX") || exit 3
    if ! awk -F "\\t" -v id="$REMOVE_ID" "BEGIN { found=0 } \$1 == id { found=1; next } { print } END { exit !found }" \
      "$QUEUE/jobs.tsv" > "$tmp"; then
      rm -f "$tmp"
      echo "error: job ID not found in jobs.tsv: $REMOVE_ID" >&2
      exit 4
    fi
    mv "$tmp" "$QUEUE/jobs.tsv"
    echo "job removed from queue: $REMOVE_ID (state=$state; artifacts retained)"
    bash "$QUEUE/status.sh"
    ;;
  restart)
    if [ ! -f "$QUEUE/dispatch.sh" ]; then
      echo "error: dispatch.sh is missing: $QUEUE/dispatch.sh" >&2
      exit 3
    fi
    if tmux has-session -t queue-dispatch 2>/dev/null; then
      tmux kill-session -t queue-dispatch
      sleep 1
    fi
    EXCLUDE_GPUS="$RESTART_EXCLUDE" tmux new-session -d -s queue-dispatch "EXCLUDE_GPUS=$RESTART_EXCLUDE bash $QUEUE/dispatch.sh"
    echo "dispatcher restarted (exclude_gpus=${RESTART_EXCLUDE:-none}; running jobs untouched)"
    ;;
  stop)
    if [ -x "$QUEUE/stop.sh" ]; then
      bash "$QUEUE/stop.sh"
    else
      echo "queue dir exists but stop.sh is missing: $QUEUE" >&2
      exit 3
    fi
    ;;
  *)
    echo "error: unsupported queue action: ${MODE:-status}" >&2
    exit 2
    ;;
esac

if [ "${SHOW_JOBS:-0}" = "1" ]; then
  echo
  echo "=== jobs.tsv ==="
  if [ -f "$QUEUE/jobs.tsv" ]; then
    cat "$QUEUE/jobs.tsv"
  else
    echo "(no jobs.tsv)"
  fi
fi

if [ "${LOG_TAIL:-0}" -gt 0 ] 2>/dev/null; then
  echo
  echo "=== queue.log (last $LOG_TAIL) ==="
  if [ -f "$QUEUE/queue.log" ]; then
    tail -n "$LOG_TAIL" "$QUEUE/queue.log"
  else
    echo "(no queue.log)"
  fi
fi

if [ "${SHOW_DISPATCH:-0}" = "1" ]; then
  echo
  echo "=== dispatcher ==="
  if tmux has-session -t queue-dispatch 2>/dev/null; then
    echo "tmux session queue-dispatch: ALIVE"
  else
    echo "tmux session queue-dispatch: not running"
  fi
  echo "--- training procs ---"
  ps -eo pid,etime,args 2>/dev/null | grep -E "\.venv/bin/train|run_task" | grep -v grep | cut -c1-160 || true
fi
'

# ---------------------------------------------------------------------------
# Run it
# ---------------------------------------------------------------------------
if ! "${SSH_CMD[@]}" "MODE=$MODE SHOW_JOBS=$SHOW_JOBS LOG_TAIL=$LOG_TAIL SHOW_DISPATCH=$SHOW_DISPATCH RESTART_EXCLUDE=$RESTART_EXCLUDE ADD_ID=$ADD_ID ADD_COMMAND_B64=$ADD_COMMAND_B64 REMOVE_ID=$REMOVE_ID bash -s" <<< "$REMOTE_SCRIPT"; then
  echo "error: remote command failed (SSH reachable? queue present?)" >&2
  exit 3
fi
