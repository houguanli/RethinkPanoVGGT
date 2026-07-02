#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUNA="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-/home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python}"
PANOVGGT_ROOT="${PANOVGGT_ROOT:-/mnt/e/PanoVGGT_minimal_datasets/datasets}"
PANOCITY_ROOT="${PANOCITY_ROOT:-$PANOVGGT_ROOT/Panocity}"
CONFIG="${CONFIG:-configs/single_pano_4090_official_mixed4_pano_after_current_12h.yaml}"
WAIT_SESSION="${WAIT_SESSION:-mixed4_pano_4090_12h}"
PREV_CKPT="${PREV_CKPT:-$LUNA/logs/mixed4_pano_full_lowres_luna_after_full_warmup_4090_12h/last.pt}"
OUT="$LUNA/logs/official_mixed4_pano_after_current_4090_12h"
SEQ_LOG="$LUNA/logs/official_mixed4_pano_after_current_4090_12h_sequence.log"

mkdir -p "$OUT" "$(dirname "$SEQ_LOG")"

{
  echo "[official-mixed4] queued $(date --iso-8601=seconds)"
  echo "[official-mixed4] luna=$LUNA"
  echo "[official-mixed4] config=$CONFIG"
  echo "[official-mixed4] python=$PYTHON"
  echo "[official-mixed4] panovggt_root=$PANOVGGT_ROOT"
  echo "[official-mixed4] panocity_root=$PANOCITY_ROOT"
  echo "[official-mixed4] wait_session=$WAIT_SESSION"
  echo "[official-mixed4] prev_ckpt=$PREV_CKPT"
  echo "[official-mixed4] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
} | tee -a "$SEQ_LOG"

cd "$LUNA"
echo "[official-mixed4] building mixed4 official indexes before timed training $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
PYTHONPATH="$LUNA${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" scripts/build_mixed4_official_indexes.py \
  --root "$PANOVGGT_ROOT" \
  --train-fraction 0.95 \
  --seed 42 \
  2>&1 | tee -a "$SEQ_LOG"
echo "[official-mixed4] index ready $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"

while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
  echo "[official-mixed4] waiting for tmux session $WAIT_SESSION to finish $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
  sleep 300
done

while [[ ! -s "$PREV_CKPT" ]]; do
  echo "[official-mixed4] waiting for previous checkpoint $PREV_CKPT $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
  sleep 300
done

echo "[official-mixed4] timed 12h training started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
PYTHONPATH="$LUNA${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" training/train_pano_omega.py --config "$CONFIG" \
  2>&1 | tee -a "$OUT/train_12h.log"

echo "[official-mixed4] finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
