#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUNA="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$LUNA/.." && pwd)"
BASELINE="$ROOT/baseline01_vggt_omega"

PYTHON="${PYTHON:-/home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-29671}"
LUNA_PORT="${LUNA_PORT:-29672}"

BASE_CONFIG="panocity_paired_full_lowres_full_warmup_4090_3h_for_luna"
BASE_OUT="$BASELINE/logs/$BASE_CONFIG"
BASE_CKPT="$BASE_OUT/ckpts/checkpoint.pt"

LUNA_CONFIG="configs/single_pano_4090_panocity_paired_full_lowres_luna_after_full_warmup_9h.yaml"
LUNA_OUT="$LUNA/logs/panocity_paired_full_lowres_luna_after_full_warmup_4090_9h"
SEQ_LOG="$LUNA/logs/panocity_paired_full_lowres_stage_4090_12h_sequence.log"

mkdir -p "$(dirname "$SEQ_LOG")" "$BASE_OUT" "$LUNA_OUT"

{
  echo "[sequence] started $(date --iso-8601=seconds)"
  echo "[sequence] root=$ROOT"
  echo "[sequence] baseline_config=$BASE_CONFIG"
  echo "[sequence] luna_config=$LUNA_CONFIG"
  echo "[sequence] python=$PYTHON"
  echo "[sequence] cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
} | tee -a "$SEQ_LOG"

cd "$BASELINE"
echo "[sequence] stage1 baseline full warmup low384 started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$BASE_PORT" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  PYTHONPATH="$BASELINE${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" training/launch.py --config "$BASE_CONFIG" \
  2>&1 | tee -a "$BASE_OUT/train_3h_console.log"
echo "[sequence] stage1 baseline full warmup low384 finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"

if [[ ! -s "$BASE_CKPT" ]]; then
  echo "[sequence] missing baseline checkpoint: $BASE_CKPT" | tee -a "$SEQ_LOG"
  exit 1
fi

cd "$LUNA"
echo "[sequence] stage2/3 LUNA residual low384 continuation started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$LUNA_PORT" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  PYTHONPATH="$LUNA${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" training/train_pano_omega.py --config "$LUNA_CONFIG" \
  2>&1 | tee -a "$LUNA_OUT/train_9h.log"
echo "[sequence] stage2/3 LUNA residual low384 continuation finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"

if [[ ! -s "$LUNA_OUT/last.pt" ]]; then
  echo "[sequence] missing LUNA checkpoint: $LUNA_OUT/last.pt" | tee -a "$SEQ_LOG"
  exit 1
fi

echo "[sequence] validation val100 started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
"$PYTHON" scripts/evaluate_depth_checkpoint.py \
  --config "$LUNA_CONFIG" \
  --checkpoint "$LUNA_OUT/last.pt" \
  --output "$LUNA_OUT/validation_val100_summary.json" \
  --per-sample-csv "$LUNA_OUT/validation_val100_per_sample.csv" \
  --train-loss-csv "$LUNA_OUT/loss.csv" \
  --device cuda \
  --split val \
  --curriculum-bins all \
  --limit 100 \
  --hard-limit 0 \
  --num-workers 2 \
  --seed 123 \
  --no-progress \
  2>&1 | tee "$LUNA_OUT/validation_val100_console.log"
echo "[sequence] validation val100 finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"

echo "[sequence] plotting loss curves $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
"$PYTHON" scripts/plot_loss_csv.py \
  --run "warmup=$BASE_OUT/loss.csv" \
  --run "luna=$LUNA_OUT/loss.csv@3" \
  --metric auto \
  --x elapsed_hours \
  --smooth-method rolling_median \
  --rolling-window 401 \
  --resample-seconds 10 \
  --clip-quantile 0.98 \
  --raw-alpha 0.10 \
  --out "$LUNA_OUT/loss_curve_full12h_lowres_smoothed_robust.png" \
  --title "PanoCity full-lowres 3h warmup + 9h LUNA robust smoothed loss" \
  2>&1 | tee -a "$SEQ_LOG"

echo "[sequence] finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
