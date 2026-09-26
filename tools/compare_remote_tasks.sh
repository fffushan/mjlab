#!/usr/bin/env bash
# Compare training runs on a remote GPU server with local mjlab logs, and optionally
# fetch what is missing.
#
#   compare_remote_tasks.sh [OPTIONS] 'ssh -p PORT user@host'
#
# The remote is given as the ssh command you would paste, quoted as ONE argument:
#   compare_remote_tasks.sh --fetch 'ssh -p 31505 fushan@10.14.64.37'
# The leading 'ssh' is optional; without -p the port defaults to 22.
#
# Matching is by SHA-256 of the LATEST model_*.pt in each run, so local renames
# never produce false "missing" entries, and an advanced (still-training) remote
# run is correctly reported as out of sync.
#
# Remote scan roots:   ~/output/<task>/checkpoints/<config>/<run>/
#                      ~/mjlab/logs/rsl_rl/<config>/<run>/
#                      ~/mjlab/runs/<config>/<run>/
# Local destination:   ~/projects/mjlab/logs/rsl_rl/<config>/<dir>/
#
# Fetch copies only the essentials: latest model_*.pt, *.onnx, events.out.*,
# params/*.  Destination keeps the remote TASK name (not a timestamp).

set -euo pipefail

MODE=list
FORCE=0
SHOW_ALL=0
REMOTE_SPEC=""
REMOTE=""
PORT=22
# Local logs root (override with LOCAL_RSL_RL=... to test or use another checkout).
LOCAL_RSL_RL="${LOCAL_RSL_RL:-$HOME/projects/mjlab/logs/rsl_rl}"
SCRIPT_NAME=$(basename "$0")

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS] 'ssh -p PORT user@host'

  REMOTE SPEC   The ssh command to the server, quoted as one argument:
                  'ssh -p 31505 fushan@10.14.64.37'
                The leading 'ssh' is optional; without -p the port defaults to 22.
  --all         Also show runs already present locally (matched by model checksum)
  --fetch       Fetch the missing runs (rsync: latest .pt, .onnx, events, params)
  --force       With --fetch, overwrite an existing destination directory
  --only PATTERN  Restrict to runs whose task name OR config matches PATTERN
                  (case-insensitive). A plain string matches ANYWHERE in the name
                  or config, e.g. 'agibot_x2_velocity', 'qianghuo-torso-imu-anchor';
                  wildcards are honoured when present, e.g. 'torso-critic-*'.

Output is grouped by remote source and config, e.g.

  runs / agibot_x2_velocity
    2026-09-23_00-28-00                            2        INCOMPLETE

Exit status: 0 ok, 2 usage error, 3 remote scan failure, 4 fetch failure.
EOF
}

ONLY_PATTERN=""
while [ $# -gt 0 ]; do
  case "$1" in
    --all) SHOW_ALL=1; shift ;;
    --fetch) MODE=fetch; shift ;;
    --force) FORCE=1; shift ;;
    --only) ONLY_PATTERN="${2:?--only needs a pattern}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    -p|-p[0-9]*|-p\ *)
      # A remote spec that starts with '-p' (quoted as one argument) is unambiguous
      # even without the 'ssh' prefix: '-p 31505 fushan@10.14.64.37',
      # '-p31505 fushan@10.14.64.37', or a bare '-p 31505 user@host' tail.
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
if [ ! -d "$LOCAL_RSL_RL" ]; then
  echo "error: local logs directory not found: $LOCAL_RSL_RL" >&2
  exit 2
fi

SSH_CMD=(ssh -p "$PORT" -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE")

# Latest model_*.pt iteration in a run dir, or "" (guarded: ls fails when no match).
latest_iter_of() {
  ls "$1"/model_*.pt 2>/dev/null | sed 's/.*model_//; s/\.pt$//' | sort -n | tail -1 || true
}

# ---------------------------------------------------------------------------
# Remote scan (one SSH round trip).  Roots are declared in REMOTE_ROOTS inside the
# remote script below: output/ (job tasks), mjlab/logs/rsl_rl/ and mjlab/runs/
# (direct mjlab training runs).  Output per run, tab separated:
#   source  task  config  run  state  latest_iter  sha256  relpath
# state: COMPLETE | INCOMPLETE | NO-CHECKPOINTS
# "task" is the root's top-level name (for flat roots: the run dir name);
# "relpath" is the run dir relative to the remote $HOME, used to fetch.
# ---------------------------------------------------------------------------
remote_scan() {
  "${SSH_CMD[@]}" 'bash -s' <<'REMOTE_SCRIPT'
set -uo pipefail

latest_iter_of() {
  ls "$1"/model_*.pt 2>/dev/null | sed 's/.*model_//; s/\.pt$//' | sort -n | tail -1 || true
}

scan_run() {  # $1 = run dir; prints "state<TAB>iter<TAB>sha"
  local rundir="$1" latest f sha state
  latest=$(latest_iter_of "$rundir")
  if [ -z "$latest" ]; then
    printf 'NO-CHECKPOINTS\t-\t-\n'
    return 0
  fi
  f="$rundir/model_${latest}.pt"
  sha=$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1)
  if [ "$latest" -ge 29999 ]; then state=COMPLETE; else state=INCOMPLETE; fi
  printf '%s\t%s\t%s\n' "$state" "$latest" "${sha:-}"
}

# layout=task : <task>/checkpoints/<config>/<run>   (job-system output)
scan_task_root() {  # $1=source label, $2=absolute root
  local taskdir cfgbase cfgdir cfg rundir run
  for taskdir in "$2"/*/; do
    [ -d "$taskdir" ] || continue
    cfgbase="$taskdir"checkpoints
    [ -d "$cfgbase" ] || continue
    for cfgdir in "$cfgbase"/*/; do
      [ -d "$cfgdir" ] || continue
      cfg=$(basename "$cfgdir")
      for rundir in "$cfgdir"*/; do
        [ -d "$rundir" ] || continue
        run=$(basename "$rundir")
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$1" "$(basename "$taskdir")" "$cfg" "$run" "$(scan_run "$rundir")" "${rundir#"$HOME"/}"
      done
    done
  done
}

# layout=flat : <config>/<run>                      (direct mjlab training runs)
scan_flat_root() {  # $1=source label, $2=absolute root
  local cfgdir cfg rundir run
  for cfgdir in "$2"/*/; do
    [ -d "$cfgdir" ] || continue
    cfg=$(basename "$cfgdir")
    for rundir in "$cfgdir"*/; do
      [ -d "$rundir" ] || continue
      run=$(basename "$rundir")
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$1" "$run" "$cfg" "$run" "$(scan_run "$rundir")" "${rundir#"$HOME"/}"
    done
  done
}

# Remote roots to scan: "label|home-relative path|layout".  Add new locations here.
REMOTE_ROOTS=(
  "output|output|task"
  "rsl_rl|mjlab/logs/rsl_rl|flat"
  "runs|mjlab/runs|flat"
)
for root_spec in "${REMOTE_ROOTS[@]}"; do
  IFS='|' read -r label relpath layout <<< "$root_spec"
  root="$HOME/$relpath"
  if [ ! -d "$root" ]; then continue; fi
  case "$layout" in
    task) scan_task_root "$label" "$root" ;;
    flat) scan_flat_root "$label" "$root" ;;
    *) printf 'warning: unknown remote layout %s\n' "$layout" >&2 ;;
  esac
done
REMOTE_SCRIPT
}

# ---------------------------------------------------------------------------
# Local scan: sha -> "config/run (iter)" map of every local run with checkpoints
# ---------------------------------------------------------------------------
LOCAL_MAP=$(mktemp)
MISSING_TSV=$(mktemp)
DISPLAY_TSV=$(mktemp)
cleanup() { rm -f "$LOCAL_MAP" "$MISSING_TSV" "$DISPLAY_TSV"; }
trap cleanup EXIT

while IFS= read -r -d '' rundir; do
  latest=$(latest_iter_of "$rundir")
  if [ -z "$latest" ]; then continue; fi
  sha=$(sha256sum "$rundir/model_${latest}.pt" 2>/dev/null | cut -d' ' -f1) || continue
  if [ -z "$sha" ]; then continue; fi
  label="${rundir#"$LOCAL_RSL_RL"/}"
  printf '%s\t%s\t%s\n' "$sha" "$label" "$latest"
done < <(find "$LOCAL_RSL_RL" -mindepth 2 -maxdepth 2 -type d -print0 2>/dev/null) > "$LOCAL_MAP"

# ---------------------------------------------------------------------------
# Compare and report
# ---------------------------------------------------------------------------
REMOTE_SCAN=$(remote_scan) || { echo "error: remote scan failed (SSH reachable?)" >&2; exit 3; }
if [ -z "$REMOTE_SCAN" ]; then
  echo "No runs found on remote."
  exit 0
fi

# Match --only (glob, case-insensitive) against the task name or the config. Empty = all.
only_want() {  # $1=task name, $2=config
  [ -z "$ONLY_PATTERN" ] && return 0
  local pat lowered_task lowered_cfg
  pat=$(printf '%s' "$ONLY_PATTERN" | tr '[:upper:]' '[:lower:]')
  # Plain text matches anywhere, so --only 'qianghuo-torso-imu-anchor' still finds
  # 'qianghuo-torso-imu-anchor_20260924145851' without the user writing wildcards.
  case "$ONLY_PATTERN" in
    *'*'*|*'?'*|*'['*) ;;      # already a glob: use it verbatim
    *) pat="*$pat*" ;;         # plain text: substring match
  esac
  lowered_task=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
  lowered_cfg=$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')
  # shellcheck disable=SC2254
  case "$lowered_task" in $pat) return 0 ;; esac
  # shellcheck disable=SC2254
  case "$lowered_cfg" in $pat) return 0 ;; esac
  return 1
}

# Render rows grouped by source/config so every run is attributable to its experiment.
render_display() {
  LC_ALL=C sort -t$'\t' -k1,1 -k2,2 -k3,3 "$1" | awk -F'\t' '
    {
      grp = $1 " / " $2
      if (grp != prev) { printf "\n%s\n", grp; prev = grp }
      printf "  %-52s %-8s %-14s %s\n", $3, $4, $5, $6
    }'
}

total=0; synced=0; missing=0
: > "$DISPLAY_TSV"
while IFS=$'\t' read -r source task cfg run state iter sha rel; do
  if [ -z "$source" ]; then continue; fi
  only_want "$task" "$cfg" || continue
  total=$((total+1))
  match=""
  if [ "$sha" != "-" ] && [ -n "$sha" ]; then
    match=$(awk -F'\t' -v s="$sha" '$1 == s {print $2; exit}' "$LOCAL_MAP")
  fi

  if [ -n "$match" ]; then
    synced=$((synced+1))
    if [ "$SHOW_ALL" = "1" ]; then
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$source" "$cfg" "$task" "$iter" "$state" "= local: $match" >> "$DISPLAY_TSV"
    fi
  else
    missing=$((missing+1))
    if [ "$source" = "output" ]; then destname="$task"; else destname="$run"; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$source" "$task" "$cfg" "$run" "$iter" "$state" "$sha" "$rel" "$destname" >> "$MISSING_TSV"
    if [ "$state" = "NO-CHECKPOINTS" ]; then note="(no checkpoints)"; else note=""; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$source" "$cfg" "$task" "$iter" "$state" "$note" >> "$DISPLAY_TSV"
  fi
done <<< "$REMOTE_SCAN"

if [ -s "$DISPLAY_TSV" ]; then
  render_display "$DISPLAY_TSV"
elif [ -n "$ONLY_PATTERN" ]; then
  echo "No runs match --only '$ONLY_PATTERN' (plain text matches anywhere; wildcards also allowed)."
else
  echo "No runs to show."
fi

echo
echo "Remote runs: $total total, $synced synced locally, $missing missing."

# ---------------------------------------------------------------------------
# Fetch mode
# ---------------------------------------------------------------------------
if [ "$MODE" != "fetch" ]; then exit 0; fi

if [ ! -s "$MISSING_TSV" ]; then
  echo "Nothing to fetch."
  exit 0
fi

echo
echo "Fetching missing runs (latest .pt, .onnx, events, params)..."
fail=0; fetched=0; skipped=0
declare -A FETCHED_SHA=()
while IFS=$'\t' read -r source task cfg run state iter sha rel destname; do
  if [ "$state" = "NO-CHECKPOINTS" ]; then
    echo "  SKIP  $destname (no checkpoints)"
    skipped=$((skipped+1))
    continue
  fi

  # The same model can be reachable from two remote roots: fetch it only once.
  if [ -n "$sha" ] && [ "$sha" != "-" ] && [ -n "${FETCHED_SHA[$sha]:-}" ]; then
    echo "  SKIP  $destname (identical model already fetched from ${FETCHED_SHA[$sha]})"
    skipped=$((skipped+1))
    continue
  fi

  # relpath comes from the remote scan, so fetch never re-derives the layout.
  srcroot="$rel"
  destdir="$LOCAL_RSL_RL/$cfg/$destname"

  if [ -e "$destdir" ] && [ "$FORCE" = "0" ]; then
    echo "  SKIP  $destname -> $cfg/$destname exists (use --force to overwrite)"
    skipped=$((skipped+1))
    continue
  fi

  # Resolve the latest checkpoint again at fetch time (may have advanced since scan).
  latest=$("${SSH_CMD[@]}" "ls '$srcroot'/model_*.pt 2>/dev/null | sed 's/.*model_//; s/\.pt\$//' | sort -n | tail -1" < /dev/null || true)
  if [ -z "$latest" ]; then
    echo "  SKIP  $destname (no checkpoints at fetch time)"
    skipped=$((skipped+1))
    continue
  fi

  mkdir -p "$(dirname "$destdir")"
  staging=$(mktemp -d "${destdir}.staging.XXXXXX")
  if rsync -a -e "ssh -p $PORT" \
      --include="*/" \
      --include="model_${latest}.pt" \
      --include="*.onnx" \
      --include="events.out.*" \
      --include="params/**" \
      --exclude="*" \
      "$REMOTE:$srcroot/" "$staging/" < /dev/null; then
    if [ -e "$destdir" ]; then rm -rf "$destdir"; fi
    mv "$staging" "$destdir"
    echo "  OK    $destname -> $cfg/$destname (model_${latest}.pt)"
    fetched=$((fetched+1))
    if [ -n "$sha" ] && [ "$sha" != "-" ]; then FETCHED_SHA[$sha]="$source/$destname"; fi
  else
    rm -rf "$staging"
    echo "  FAIL  $destname" >&2
    fail=$((fail+1))
  fi
done < "$MISSING_TSV"

echo
echo "Fetch done: $fetched fetched, $skipped skipped, $fail failed."
if [ "$fail" -ne 0 ]; then exit 4; fi
