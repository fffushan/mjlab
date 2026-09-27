#!/usr/bin/env bash
set -uo pipefail
QUEUE_DIR="${QUEUE_DIR:-/home/fushan/mjlab/queue}"
cd /home/fushan/mjlab
echo "dispatcher: $(tmux has-session -t queue-dispatch 2>/dev/null && echo ALIVE || echo 'not running')   max_concurrent=${MAX_CONCURRENT:-8}"
printf '%-16s %-8s %-22s %-4s %s\n' JOB STATE SESSION GPU PROGRESS
while IFS=$'\t' read -r id session command; do
  case "$id" in ''|'#'*) continue;; esac
  st=$(cat "$QUEUE_DIR/state/$id.state" 2>/dev/null || echo pending)
  bdir=$(cat "$QUEUE_DIR/batch.$id" 2>/dev/null || true)
  gpu="-"; prog="-"
  if [ -n "$bdir" ] && [ -d "$bdir" ]; then
    gpu=$(cut -f1 "$bdir/jobs.tsv" 2>/dev/null | head -1)
    [ -f "$bdir/stdout.log" ] && prog=$(tr -d '\r' < "$bdir/stdout.log" | grep -oE 'Learning iteration [0-9]+/[0-9]+' | tail -1)
    [ -f "$bdir/exit.status" ] && prog="exit $(cat "$bdir/exit.status")"
  fi
  printf '%-16s %-8s %-22s %-4s %s\n' "$id" "$st" "$session" "${gpu:--}" "${prog:--}"
done < "$QUEUE_DIR/jobs.tsv"
echo
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
