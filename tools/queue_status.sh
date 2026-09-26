#!/usr/bin/env bash
# Local monitor for the remote GPU task queue (the mjlab queue dispatcher).
#
#   queue_status.sh [OPTIONS] 'ssh -p PORT user@host'
#
# The remote is given as the ssh command you would paste, quoted as ONE argument:
#   queue_status.sh 'ssh -p 31505 fushan@10.14.64.37'
# The leading 'ssh' is optional; without -p the port defaults to 22.
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
#   --stop        run the remote stop.sh instead (idempotent: safe to re-run)
#   --jobs        also print the remote jobs.tsv (the queued job list)
#   --tail N      also print the last N lines of the remote queue.log
#   --dispatch    also print dispatcher/tmux/process status
#
# Exit status: 0 ok, 2 usage error, 3 SSH/scan failure.

set -uo pipefail

MODE=status
SHOW_JOBS=0
SHOW_DISPATCH=0
LOG_TAIL=0
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

  --stop        Run the remote queue stop.sh (stop queueing; running jobs keep
                going). Idempotent -- safe to re-run.
  --jobs        Also print the remote jobs.tsv (current queued job list).
  --tail N      Also print the last N lines of the remote queue.log.
  --dispatch    Also print tmux/process status for the dispatcher.

Read-only by default: prints the remote queue table + GPU occupancy.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --stop) MODE=stop; shift ;;
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
  stop)
    if [ -x "$QUEUE/stop.sh" ]; then
      bash "$QUEUE/stop.sh"
    else
      echo "queue dir exists but stop.sh is missing: $QUEUE" >&2
      exit 3
    fi
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
if ! "${SSH_CMD[@]}" "MODE=$MODE SHOW_JOBS=$SHOW_JOBS LOG_TAIL=$LOG_TAIL SHOW_DISPATCH=$SHOW_DISPATCH bash -s" <<< "$REMOTE_SCRIPT"; then
  echo "error: remote command failed (SSH reachable? queue present?)" >&2
  exit 3
fi
