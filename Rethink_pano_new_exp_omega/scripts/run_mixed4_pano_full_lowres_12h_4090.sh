#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUNA="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-/home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python}"
CONFIG="${CONFIG:-configs/single_pano_4090_mixed4_pano_full_lowres_luna_after_full_warmup_9h.yaml}"
OUT="$LUNA/logs/mixed4_pano_full_lowres_luna_after_full_warmup_4090_12h"
SEQ_LOG="$LUNA/logs/mixed4_pano_full_lowres_luna_after_full_warmup_4090_12h_sequence.log"

mkdir -p "$OUT" "$(dirname "$SEQ_LOG")"

{
  echo "[mixed4] started $(date --iso-8601=seconds)"
  echo "[mixed4] luna=$LUNA"
  echo "[mixed4] config=$CONFIG"
  echo "[mixed4] python=$PYTHON"
  echo "[mixed4] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
} | tee -a "$SEQ_LOG"

cd "$LUNA"
PYTHONPATH="$LUNA${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" training/train_pano_omega.py --config "$CONFIG" \
  2>&1 | tee -a "$OUT/train_12h.log"

echo "[mixed4] finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
