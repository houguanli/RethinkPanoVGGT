#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="$(cd "$SCRIPT_DIR/.." && pwd)"

EXP_NAME="${EXP_NAME:-server_naive_fullerp_vggtomega_2p_12h_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-$BASELINE/logs/$EXP_NAME}"
NOHUP_LOG="${NOHUP_LOG:-$BASELINE/logs/nohup_${EXP_NAME}.log}"
PID_FILE="$OUT/runner.pid"

mkdir -p "$OUT" "$(dirname "$NOHUP_LOG")"
if [[ -s "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE")"
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "[naive-fullerp-server] run already active: pid=$old_pid output=$OUT"
    exit 1
  fi
fi

nohup env \
  PYTHON="${PYTHON:-python}" \
  NPROC_PER_NODE="${NPROC_PER_NODE:-4}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  PANOVGGT_ROOT="${PANOVGGT_ROOT:-}" \
  VGGT_OMEGA_CKPT="${VGGT_OMEGA_CKPT:-}" \
  EXP_NAME="$EXP_NAME" \
  OUT="$OUT" \
  RESOLUTIONS="${RESOLUTIONS:-384,512,1024,2048}" \
  STAGE_DURATION_MINUTES="${STAGE_DURATION_MINUTES:-180}" \
  BUILD_INDEXES="${BUILD_INDEXES:-1}" \
  bash "$SCRIPT_DIR/run_naive_fullerp_vggtomega_progressive.sh" \
  >"$NOHUP_LOG" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$PID_FILE"

echo "[naive-fullerp-server] started pid=$pid"
echo "[naive-fullerp-server] output=$OUT"
echo "[naive-fullerp-server] log=$NOHUP_LOG"
