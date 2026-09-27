#!/usr/bin/env bash
# Stop the dispatcher. Running training jobs keep going.
set -uo pipefail
tmux kill-session -t queue-dispatch 2>/dev/null && echo "dispatcher stopped" || echo "dispatcher not running"
echo "running training jobs were left untouched:"
tmux ls 2>/dev/null | grep -E 'tennis' || true
