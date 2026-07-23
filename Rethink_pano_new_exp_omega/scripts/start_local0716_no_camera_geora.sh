#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_ID="${RUN_ID:-local0716_no_camera_geora_20260724_001}"
RUN_OUT="${RUN_OUT:-$PROJECT_ROOT/logs/$RUN_ID}"
NOHUP_LOG="${NOHUP_LOG:-$PROJECT_ROOT/logs/nohup_${RUN_ID}.log}"
PID_FILE="${PID_FILE:-$RUN_OUT/launcher.pid}"

if [[ "${1:-}" == "--background" ]]; then
  mkdir -p "$RUN_OUT"
  if [[ -s "$PID_FILE" ]]; then
    existing_pid="$(cat "$PID_FILE")"
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
      echo "already running: pid=$existing_pid log=$NOHUP_LOG"
      exit 0
    fi
  fi
  nohup "$0" >"$NOHUP_LOG" 2>&1 </dev/null &
  launcher_pid=$!
  echo "$launcher_pid" >"$PID_FILE"
  echo "started: pid=$launcher_pid log=$NOHUP_LOG output=$RUN_OUT"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHON="${PYTHON:-/home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export PANOVGGT_ROOT="${PANOVGGT_ROOT:-/mnt/e/PanoVGGT_minimal_datasets/datasets}"
export BASE_CHECKPOINT="${BASE_CHECKPOINT:-/home/aoki/RethinkPanoVGGT_omega_compare_methods_only/ckpt/VGGT-Omega/vggt_omega_1b_512.pt}"
export WARMUP_CHECKPOINT="${WARMUP_CHECKPOINT:-/home/aoki/RethinkPanoVGGT_omega_multipano_work/Rethink_pano_new_exp_omega/logs/local_full_warmup_no_depthres_20260716_002_warmup_3h/last.pt}"
export SKIP_WARMUP=1
export SKIP_DATA_CHECK="${SKIP_DATA_CHECK:-1}"
export RUN_OUT

cd "$PROJECT_ROOT"
exec bash scripts/run_ablation_4xrtx5000.sh no_camera_geora
