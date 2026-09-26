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
# Runs are identified by the SHA-256 of their LATEST model_*.pt, never by name:
# the listing prints a 12-char checksum prefix, and --sha selects exactly those
# runs.  Local renames therefore never produce false "missing" entries, and an
# advanced (still-training) remote run is correctly reported as out of sync.
#
# Remote scan roots:   ~/output/<task>/checkpoints/<config>/<run>/
#                      ~/mjlab/logs/rsl_rl/<config>/<run>/
#                      ~/mjlab/runs/<config>/<run>/
# Local destination:   ~/projects/mjlab/logs/rsl_rl/<config>/<dir>/
#
# Fetch copies only the essentials: latest model_*.pt, *.onnx, events.out.*,
# params/*.  Destination keeps the remote TASK name (not a timestamp).
#
# A checksum match proves the checkpoint is local, not that the run is complete:
# a renamed or hand-copied run can be missing params/.  --fetch therefore also
# restores params/ for such runs, and --params does only that repair.

set -euo pipefail

MODE=list
FORCE=0
SHOW_ALL=0
COMPLETE_ONLY=0
# A run that has saved a checkpoint but has not touched its log for this long is
# treated as finished; a running job appends to the log about every 2 seconds.
STALL_MIN=10
STALL_SEC=600
REMOTE_SPEC=""
REMOTE=""
PORT=22
# Local logs root (override with LOCAL_RSL_RL=... to test or use another checkout).
LOCAL_RSL_RL="${LOCAL_RSL_RL:-$HOME/projects/mjlab/logs/rsl_rl}"
SCRIPT_NAME=$(basename "$0")

# Paths get the same colour ls uses for directories (bold blue); everything else
# stays default.  Only on a terminal, so piping into grep/awk stays clean, and
# NO_COLOR turns it off explicitly.
if [ -n "${NO_COLOR:-}" ] || [ ! -t 1 ]; then
  C_PATH=""; C_OFF=""
else
  C_PATH=$'\033[01;34m'; C_OFF=$'\033[0m'
fi

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS] 'ssh -p PORT user@host'

  REMOTE SPEC   The ssh command to the server, quoted as one argument:
                  'ssh -p 31505 fushan@10.14.64.37'
                The leading 'ssh' is optional; without -p the port defaults to 22.
  --sha HASH    Select the run(s) whose latest model_*.pt SHA-256 begins with HASH
                (6-64 hex chars, case-insensitive; may be repeated). The listing
                prints that prefix. This is the ONLY selector: run or config names
                are never used to choose what to fetch.
  --all         Also show runs already present locally (matched by model checksum)
  --stall MIN   Minutes without log activity before a run counts as finished
                (default 10; a live run logs about every 2 seconds).
  --complete    Show only FINISHED runs (checkpoint saved, log idle >= --stall)
                that are not local yet: the work still to fetch. Combines with
                --fetch to pull them all. Alias: --done.
  --fetch       With --sha fetch exactly the selected run(s); without --sha, every
                run missing locally. Pulls latest .pt, .onnx, events, params/, and
                restores params/ for synced runs that lack it.
  --params      Only restore missing params/ for the selected run(s); without
                --sha, for every synced run missing params/.
  --force       With --fetch, overwrite an existing destination directory

Output is grouped by remote source and config; column 1 is the checksum prefix you
pass to --sha.  State comes from log activity, never from an iteration target:

  runs / agibot_x2_velocity
    a1b2c3d4e5f6  2026-09-23_00-28-00                     2        RUNNING

Exit status: 0 ok, 2 usage error, 3 remote scan failure, 4 fetch failure.
EOF
}

SHA_SELS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --all) SHOW_ALL=1; shift ;;
    --fetch) MODE=fetch; shift ;;
    --params) MODE=params; shift ;;
    --force) FORCE=1; shift ;;
    --complete|--done) COMPLETE_ONLY=1; shift ;;
    --stall)
      stall_arg="${2:?--stall needs minutes}"
      if ! [[ "$stall_arg" =~ ^[0-9]+$ ]]; then
        echo "error: --stall wants a whole number of minutes, got '$stall_arg'" >&2
        exit 2
      fi
      STALL_MIN="$stall_arg"; STALL_SEC=$(( STALL_MIN * 60 )); shift 2 ;;
    --sha)
      sha_arg="${2:?--sha needs a checksum}"
      if ! [[ "${sha_arg,,}" =~ ^[0-9a-f]{6,64}$ ]]; then
        echo "error: --sha wants 6-64 hexadecimal characters, got '$sha_arg'" >&2
        exit 2
      fi
      SHA_SELS+=("${sha_arg,,}"); shift 2 ;;
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
#   source  task  config  run  state  latest_iter  sha256  idle_sec  relpath
# state: RUNNING (log active within STALL_SEC) | DONE (log idle >= STALL_SEC)
#        | NO-CHECKPOINTS (nothing saved yet, so never started)
# "task" is the root's top-level name (for flat roots: the run dir name);
# "idle_sec" is the age of the run's newest activity (log, else checkpoint);
# "relpath" is the run dir relative to the remote $HOME, used to fetch.
# ---------------------------------------------------------------------------
remote_scan() {
  "${SSH_CMD[@]}" "STALL_SEC=$STALL_SEC bash -s" <<'REMOTE_SCRIPT'
set -uo pipefail

latest_iter_of() {
  ls "$1"/model_*.pt 2>/dev/null | sed 's/.*model_//; s/\.pt$//' | sort -n | tail -1 || true
}

scan_run() {  # $1 = run dir; prints "state<TAB>iter<TAB>sha<TAB>idle_sec"
  local rundir="$1" latest f sha state newest now idle
  latest=$(latest_iter_of "$rundir")
  if [ -z "$latest" ]; then
    # Nothing saved yet: not started, or died before the first checkpoint.
    printf 'NO-CHECKPOINTS\t-\t-\t-\n'
    return 0
  fi
  f="$rundir/model_${latest}.pt"
  sha=$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1)
  # A live job appends to its tfevents log every iteration (~2 s), so the newest
  # activity timestamp decides: log if present, otherwise the newest checkpoint.
  newest=$(find "$rundir" -maxdepth 1 \
             \( -name 'events.out.tfevents.*' -o -name 'model_*.pt' \) \
             -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
  if [ -z "$newest" ]; then
    idle=999999
  else
    now=$(date +%s)
    idle=$(( now - ${newest%.*} ))
    if [ "$idle" -lt 0 ]; then idle=0; fi
  fi
  if [ "$idle" -lt "${STALL_SEC:-600}" ]; then state=RUNNING; else state=DONE; fi
  printf '%s\t%s\t%s\t%s\n' "$state" "$latest" "${sha:-}" "$idle"
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
SYNCED_TSV=$(mktemp)
DISPLAY_TSV=$(mktemp)
cleanup() { rm -f "$LOCAL_MAP" "$MISSING_TSV" "$SYNCED_TSV" "$DISPLAY_TSV"; }
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

# Select runs by checkpoint checksum prefix; no --sha selects everything.
sha_want() {  # $1 = the run's checkpoint checksum ('-' when it has no checkpoints)
  if [ "${#SHA_SELS[@]}" -eq 0 ]; then return 0; fi
  local s="${1,,}" sel
  if [ -z "$s" ] || [ "$s" = "-" ]; then return 1; fi
  for sel in "${SHA_SELS[@]}"; do
    if [[ "$s" == "$sel"* ]]; then return 0; fi
  done
  return 1
}

# Render rows grouped by source/config; column 1 is the checksum prefix.
render_display() {
  LC_ALL=C sort -t$'\t' -k1,1 -k2,2 -k4,4 "$1" | awk -F'\t' -v cp="$C_PATH" -v co="$C_OFF" '
    {
      grp = $1 " / " $2
      if (grp != prev) { printf "\n%s%s / %s%s\n", cp, $1, $2, co; prev = grp }
      pad = 50 - length($4); if (pad < 0) pad = 0
      printf "  %-12s %s%s%s%*s %-8s %-14s %s\n", $3, cp, $4, co, pad, "", $5, $6, $7
    }'
}

total=0; synced=0; missing=0; noparams=0
# A --sha selection always shows its row, whether synced or missing.
if [ "${#SHA_SELS[@]}" -gt 0 ]; then SHOW_ALL=1; fi
: > "$DISPLAY_TSV"
: > "$SYNCED_TSV"
while IFS=$'\t' read -r source task cfg run state iter sha idle rel; do
  if [ -z "$source" ]; then continue; fi
  sha_want "$sha" || continue
  sha12="-"
  if [ -n "$sha" ] && [ "$sha" != "-" ]; then sha12="${sha:0:12}"; fi
  match=""
  if [ -n "$sha" ] && [ "$sha" != "-" ]; then
    match=$(awk -F'\t' -v s="$sha" '$1 == s {print $2; exit}' "$LOCAL_MAP")
  fi
  # --complete: finished remote runs (log idle >= --stall) that are not here yet.
  if [ "$COMPLETE_ONLY" = "1" ]; then
    if [ "$state" != "DONE" ]; then continue; fi
    if [ -n "$match" ]; then continue; fi
  fi
  total=$((total+1))

  if [ -n "$match" ]; then
    synced=$((synced+1))
    # Identical checkpoint locally, but the run may still lack params/ (renamed or
    # hand-copied). Record it so --params / --fetch can restore just those files.
    lacks=0
    if [ ! -f "$LOCAL_RSL_RL/$match/params/agent.yaml" ]; then
      lacks=1; noparams=$((noparams+1))
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$source" "$cfg" "$run" "$rel" "$match" "$lacks" "$sha" >> "$SYNCED_TSV"

    if [ "$SHOW_ALL" = "1" ]; then
      note="= local: ${C_PATH}${match}${C_OFF}"
      if [ "$lacks" = "1" ]; then note="$note (no params/)"; fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$source" "$cfg" "$sha12" "$task" "$iter" "$state" "$note" >> "$DISPLAY_TSV"
    fi
  else
    missing=$((missing+1))
    if [ "$source" = "output" ]; then destname="$task"; else destname="$run"; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$source" "$task" "$cfg" "$run" "$iter" "$state" "$sha" "$rel" "$destname" >> "$MISSING_TSV"
    # NO-CHECKPOINTS already says it in the state column: no note is appended.
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$source" "$cfg" "$sha12" "$task" "$iter" "$state" "" >> "$DISPLAY_TSV"
  fi
done <<< "$REMOTE_SCAN"

if [ -s "$DISPLAY_TSV" ]; then
  render_display "$DISPLAY_TSV"
elif [ "$COMPLETE_ONLY" = "1" ]; then
  echo "Nothing to do: every finished run is already local."
elif [ "${#SHA_SELS[@]}" -gt 0 ]; then
  echo "No run has a latest checkpoint matching --sha ${SHA_SELS[*]}."
else
  echo "No runs to show."
fi

echo
echo "Remote runs: $total total, $synced synced locally, $missing missing."
if [ "$COMPLETE_ONLY" = "1" ]; then
  echo "  (--complete: finished runs not yet local)"
fi
if [ "$noparams" -gt 0 ]; then
  echo "  ($noparams synced run(s) have no params/ locally; --params restores them)"
fi
echo "  (DONE = no log activity for ${STALL_MIN} min; RUNNING = still logging; NO-CHECKPOINTS = nothing saved yet)"

# ---------------------------------------------------------------------------
# Fetch / params-repair modes
# ---------------------------------------------------------------------------
if [ "$MODE" = "list" ]; then exit 0; fi

fail=0
if [ "$MODE" = "fetch" ]; then
  if [ ! -s "$MISSING_TSV" ]; then
    echo
    if [ "${#SHA_SELS[@]}" -gt 0 ] && [ -s "$SYNCED_TSV" ]; then
      while IFS=$'\t' read -r s_source s_cfg s_run s_rel s_match s_lacks s_sha; do
        if [ -z "$s_source" ]; then continue; fi
        echo "Already local, nothing to fetch: ${s_sha:0:12}  ${C_PATH}${s_match}${C_OFF}"
      done < "$SYNCED_TSV"
    elif [ "$COMPLETE_ONLY" = "1" ]; then
      echo "Nothing to do: every finished run is already local."
    else
      echo "Nothing missing to fetch."
    fi
  else
    echo
    echo "Fetching missing runs (latest .pt, .onnx, events, params)..."
    fetched=0; skipped=0
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
        echo "  SKIP  $destname -> ${C_PATH}${cfg}/${destname}${C_OFF} exists (use --force to overwrite)"
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
        echo "  OK    ${sha:0:12} $destname -> ${C_PATH}${cfg}/${destname}${C_OFF} (model_${latest}.pt)"
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
  fi
fi

# ---------------------------------------------------------------------------
# params repair (both --fetch and --params): the run's checkpoint is already
# local under its matched directory, only params/ is missing.  --ignore-existing
# means nothing already present is ever overwritten.
# ---------------------------------------------------------------------------
pwant=$(awk -F'\t' '$6 == "1" {n++} END {print n+0}' "$SYNCED_TSV" 2>/dev/null || echo 0)
pfix=0; pnone=0; pfail=0
if [ "$pwant" -gt 0 ]; then
  echo
  echo "Restoring params/ for $pwant synced run(s)..."
  while IFS=$'\t' read -r source cfg run rel match lacks sha; do
    if [ -z "$source" ]; then continue; fi
    if [ "$lacks" != "1" ]; then continue; fi
    localdir="$LOCAL_RSL_RL/$match"
    if rsync -a --ignore-existing -e "ssh -p $PORT" \
        "$REMOTE:$rel/params/" "$localdir/params/" < /dev/null; then
      echo "  OK    ${sha:0:12} $cfg/$run -> ${C_PATH}${match}/params/${C_OFF}"
      pfix=$((pfix+1))
    else
      # Distinguish "remote has no params/" from a real transfer failure.
      if "${SSH_CMD[@]}" "test -d '$rel/params'" < /dev/null 2>/dev/null; then
        echo "  FAIL  $cfg/$run -> ${C_PATH}${match}/params/${C_OFF}" >&2
        pfail=$((pfail+1))
      else
        echo "  NONE  $cfg/$run: remote has no params/ to restore"
        pnone=$((pnone+1))
      fi
    fi
  done < "$SYNCED_TSV"
  echo
  echo "Params: $pfix restored, $pnone unavailable remotely, $pfail failed."
fi

if [ "$fail" -ne 0 ] || [ "$pfail" -ne 0 ]; then exit 4; fi
