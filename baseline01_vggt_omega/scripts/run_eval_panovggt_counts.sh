#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-/home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python}"
CONFIG="${CONFIG:-mixed4_pano_vggtomega_multiview_groups_4090_12h}"
CHECKPOINT="${CHECKPOINT:-logs/local_vggtomega_multiview_groups_20260717_0210/ckpts/checkpoint.pt}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/e/PanoVGGT_minimal_datasets/datasets}"
TRAIN_LOSS_CSV="${TRAIN_LOSS_CSV:-logs/local_vggtomega_multiview_groups_20260717_0210/loss.csv}"
OUT="${OUT:-logs/local_vggtomega_multiview_groups_20260717_0210/eval_full_anchor_panovggt_counts}"
DATASETS="${DATASETS:-all}"
LIMIT_PER_DATASET="${LIMIT_PER_DATASET:-0}"
DEVICE="${DEVICE:-auto}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
PRED_DEPTH_SCALE="${PRED_DEPTH_SCALE:-}"
FOREGROUND="${FOREGROUND:-0}"

mkdir -p "$OUT"
rm -f \
  "$OUT/eval_stdout.log" \
  "$OUT/eval_stderr.log" \
  "$OUT/eval.pid" \
  "$OUT/validation_mixed4_full_anchor_panovggt_counts_summary.json" \
  "$OUT/per_sample.csv" \
  "$OUT/camera_pairs.csv"

EVAL_CMD=(
  "$PYTHON_BIN" scripts/evaluate_mixed4_depth_checkpoint.py
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --dataset-root "$DATASET_ROOT" \
  --output "$OUT/validation_mixed4_full_anchor_panovggt_counts_summary.json" \
  --per-sample-csv "$OUT/per_sample.csv" \
  --camera-pair-csv "$OUT/camera_pairs.csv" \
  --train-loss-csv "$TRAIN_LOSS_CSV" \
  --datasets "$DATASETS" \
  --limit-per-dataset "$LIMIT_PER_DATASET" \
  --pano-count-policy panovggt \
  --device "$DEVICE" \
  --amp-dtype "$AMP_DTYPE"
)
if [[ -n "$PRED_DEPTH_SCALE" ]]; then
  EVAL_CMD+=(--pred-depth-scale "$PRED_DEPTH_SCALE")
fi
EVAL_CMD+=(--no-progress)

if [[ "$FOREGROUND" == "1" ]]; then
  echo "$$" > "$OUT/eval.pid"
  "${EVAL_CMD[@]}" > "$OUT/eval_stdout.log" 2> "$OUT/eval_stderr.log"
  echo "[eval] foreground finished"
else
  nohup "${EVAL_CMD[@]}" > "$OUT/eval_stdout.log" 2> "$OUT/eval_stderr.log" &
  echo "$!" > "$OUT/eval.pid"
  echo "[eval] pid=$(cat "$OUT/eval.pid")"
fi
echo "[eval] out=$OUT"
