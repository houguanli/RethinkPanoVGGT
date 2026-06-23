#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29641}"
CONFIG="panocity_paired_4xrtx5000_full_finetune_6h"
OUT="$BASELINE/logs/$CONFIG"
RUN_LOG="$OUT/train_6h_console.log"

mkdir -p "$OUT"

{
  echo "[baseline-full] started $(date --iso-8601=seconds)"
  echo "[baseline-full] baseline=$BASELINE"
  echo "[baseline-full] config=$CONFIG"
  echo "[baseline-full] python=$PYTHON"
  echo "[baseline-full] nproc_per_node=$NPROC_PER_NODE"
  echo "[baseline-full] master_port=$MASTER_PORT"
} | tee -a "$RUN_LOG"

cd "$BASELINE"
PYTHONPATH="$BASELINE${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" -m torch.distributed.run \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$MASTER_PORT" \
  training/launch.py --config "$CONFIG" \
  2>&1 | tee -a "$RUN_LOG"

echo "[baseline-full] finished $(date --iso-8601=seconds)" | tee -a "$RUN_LOG"
