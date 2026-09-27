#!/bin/bash
# run_task.sh - one-line launcher for an mjlab training task (dev container or managed job).
#
# Usage (this is the whole 启动命令 field):
#   bash /home/fushan/mjlab/run_task.sh <task> <max-iterations> [motion-file] [extra train args...]
#
#   # velocity / locomotion tasks - no motion file
#   bash /home/fushan/mjlab/run_task.sh Mjlab-Velocity-Flat-Unitree-Go1 30000
#
#   # tracking tasks - motion file REQUIRED
#   #   a bare name is resolved against  $PWD, <repo>, <repo>/data
#   bash /home/fushan/mjlab/run_task.sh Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation 30000 qianghuo.npz
#
# Alternatives to the positional argument:  --motion <file>  or  MOTION_FILE=<file>
#
# Knobs (set in the task form's 环境变量 field):
#   NUM_ENVS=8192     parallel environments
#   GPU_ID=0          visible GPU index (single GPU)
#   LOG_ROOT=<dir>    override the log/checkpoint root
#
# Behaviour: single GPU; checkpoints -> $GENIE_ML_CKPT_OUTPUT_DIR (platform auto-upload);
# run dir symlinked under $GENIE_ML_TENSORBOARD_LOG_DIR; JIT caches kept on the NFS.
set -euo pipefail

REPO=/home/fushan/mjlab
TASK="${1:-}"; ITERS="${2:-}"
if [ -z "$TASK" ] || [ -z "$ITERS" ]; then
  sed -n '2,24p' "$0"; exit 2
fi
shift 2

# ------------------------------------------------------- task shorthands
# The X2 ablation task ids are long enough to invite typos in a platform form.
# Names not listed here pass through untouched, so a full registered task id
# still works.
declare -A TASK_ALIASES=(
  [x2vel]="Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation"
  [x2vel-torso-critic-pelvis-root]="Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Torso-IMU-Critic-Pelvis-Root"
  [x2vel-torso-critic-pelvis-upvector]="Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Torso-IMU-Critic-Pelvis-Upvector"
  [x2vel-torso-critic-torso-root]="Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Torso-IMU-Critic-Torso-Root"
  [x2vel-torso-critic-torso-upvector]="Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Torso-IMU-Critic-Torso-Upvector"
)
if [ -n "${TASK_ALIASES[$TASK]:-}" ]; then
  echo "[alias] $TASK -> ${TASK_ALIASES[$TASK]}"
  TASK="${TASK_ALIASES[$TASK]}"
fi
EXTRA=("$@")

NUM_ENVS="${NUM_ENVS:-8192}"
GPU_ID="${GPU_ID:-0}"

# ---------------------------------------------------------------- motion file
MOTION="${MOTION_FILE:-}"
if [ -z "$MOTION" ] && [ ${#EXTRA[@]} -gt 0 ]; then
  case "${EXTRA[0]}" in
    --motion)   MOTION="${EXTRA[1]:-}"; EXTRA=("${EXTRA[@]:2}") ;;
    --motion=*) MOTION="${EXTRA[0]#--motion=}"; EXTRA=("${EXTRA[@]:1}") ;;
    --*)        ;;
    *)          MOTION="${EXTRA[0]}"; EXTRA=("${EXTRA[@]:1}") ;;
  esac
fi

resolve_motion() {
  local p="$1" c
  for c in "$p" "$REPO/$p" "$REPO/data/$p"; do
    [ -f "$c" ] && { realpath "$c"; return 0; }
  done
  return 1
}

MOTION_ARG=""
if [ -n "$MOTION" ]; then
  if MOTION_ARG="$(resolve_motion "$MOTION")"; then
    :
  else
    echo "[error] motion file not found: $MOTION"
    echo "        searched: \$PWD, $REPO, $REPO/data"
    echo "        available: $(ls "$REPO"/data/*.npz 2>/dev/null | xargs -r -n1 basename | tr '\n' ' ')"
    exit 2
  fi
fi

# tracking tasks cannot start without one - fail fast instead of burning a slot
case "$TASK" in
  *[Tt]racking*)
    if [ -z "$MOTION_ARG" ] && ! printf '%s\n' "${EXTRA[@]:-}" | grep -q -- "--env.commands.motion.motion-file"; then
      echo "[error] '$TASK' is a tracking task: a motion file is required."
      echo "        e.g. bash $0 $TASK $ITERS qianghuo.npz"
      case "$TASK" in
        *X2*|*x2*) echo "        X2 tracking -> use: $REPO/data/qianghuo_smplx_agibot_x2_tracking.npz" ;;
        *G1*|*g1*) echo "        G1 tracking -> use: $REPO/data/qianghuo_smplx_unitree_g1_29dof_mode_15_tracking.npz" ;;
      esac
      echo "        available: $(ls "$REPO"/data/*.npz 2>/dev/null | xargs -r -n1 basename | tr '\n' ' ')"
      exit 2
    fi ;;
esac

# ----------------------------------------------------------------- platform IO
CKPT="${LOG_ROOT:-${GENIE_ML_CKPT_OUTPUT_DIR:-$REPO/runs}}"
TB="${GENIE_ML_TENSORBOARD_LOG_DIR:-$CKPT}"
mkdir -p "$CKPT"
export PYTHONUNBUFFERED=1

case " ${XDG_CACHE_HOME:-$HOME/.cache}" in
  *" /home/fushan/"*) ;;
  *) [ -w /home/fushan ] && export XDG_CACHE_HOME=/home/fushan/.cache ;;
esac
mkdir -p "${XDG_CACHE_HOME:-$HOME/.cache}" 2>/dev/null || true

if [ "$TB" != "$CKPT" ]; then
  mkdir -p "$TB" 2>/dev/null || true
  ln -sfn "$CKPT" "$TB/mjlab" 2>/dev/null || echo "[tb] could not link into $TB"
fi

# ---------------------------------------------------------------------- libEGL
EGL=""
for c in /lib/x86_64-linux-gnu/libEGL.so.1 /usr/lib/x86_64-linux-gnu/libEGL.so.1; do
  [ -e "$c" ] && EGL="$c" && break
done
if [ -z "$EGL" ]; then
  echo "[egl] libEGL missing from this image - attempting install"
  if command -v apt-get >/dev/null 2>&1; then
    { sudo -n apt-get update -qq && sudo -n apt-get install -yqq libegl1 libegl-mesa0 libgbm1; } 2>/dev/null \
      || { apt-get update -qq && apt-get install -yqq libegl1 libegl-mesa0 libgbm1; } 2>/dev/null || true
  fi
  [ -e /usr/lib/x86_64-linux-gnu/libEGL.so.1 ] && EGL=/usr/lib/x86_64-linux-gnu/libEGL.so.1
fi
if [ -z "$EGL" ]; then
  echo "[egl] install unavailable - trying the vendored copy in $REPO/.egl-fallback"
  export LD_LIBRARY_PATH="$REPO/.egl-fallback${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  if "$REPO/.venv/bin/python" -c "import OpenGL.EGL" 2>/dev/null; then
    echo "[egl] vendored copy works"
  else
    echo "[egl] WARNING: still no working libEGL - 'import mjlab' may fail"
  fi
fi

# --------------------------------------------------------------------- summary
cd "$REPO"
echo "==================== mjlab task ===================="
echo "task         : $TASK"
echo "iterations   : $ITERS"
echo "num_envs     : $NUM_ENVS"
echo "gpu          : visible device $GPU_ID"
echo "motion file  : ${MOTION_ARG:-(none)}"
if [ -n "$MOTION_ARG" ]; then
  "$REPO/.venv/bin/python" - "$MOTION_ARG" <<'PY' 2>/dev/null || echo "  (could not inspect motion file)"
import sys, numpy as np
d = np.load(sys.argv[1], allow_pickle=True)
fps = d["fps"][0] if "fps" in d else "?"
jp  = d["joint_pos"].shape if "joint_pos" in d else "?"
bp  = d["body_pos_w"].shape if "body_pos_w" in d else "?"
n   = int(d["joint_pos"].shape[0]) if "joint_pos" in d else max(len(v) for v in d.values())
print(f"  fps={fps}  frames={n}  joint_pos={jp}  body_pos_w={bp}")
print(f"  keys={list(d.keys())[:10]}")
PY
fi
echo "checkpoints  : $CKPT"
echo "tensorboard  : $TB"
echo "python       : $("$REPO/.venv/bin/python" -V 2>&1)"
echo "driver       : $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo 'nvidia-smi missing')"
echo "glibc        : $(ldd --version 2>/dev/null | head -1)"
echo "gpu count    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
echo "--- platform env ---"
env | grep -E '^GENIE_ML' | sort || echo "(none)"
echo "===================================================="

ARGS=(
  --env.scene.num-envs "$NUM_ENVS"
  --agent.max-iterations "$ITERS"
  --gpu-ids "[$GPU_ID]"
  --agent.logger tensorboard
  --log-root "$CKPT"
)
if [ -n "$MOTION_ARG" ]; then
  ARGS+=(--env.commands.motion.motion-file "$MOTION_ARG")
fi

exec "$REPO/.venv/bin/train" "$TASK" "${ARGS[@]}" "${EXTRA[@]}"
