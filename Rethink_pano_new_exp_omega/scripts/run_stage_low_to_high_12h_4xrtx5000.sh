#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUNA="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$LUNA/.." && pwd)"
BASELINE="$ROOT/baseline01_vggt_omega"

PYTHON="${PYTHON:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BASE_PORT="${BASE_PORT:-29681}"
LUNA_PORT="${LUNA_PORT:-29682}"

BASE_CONFIG="panocity_paired_low_to_high_4xrtx5000_full_warmup_3h_for_luna"
BASE_OUT="$BASELINE/logs/$BASE_CONFIG"
BASE_CKPT="$BASE_OUT/ckpts/checkpoint.pt"

LUNA_CONFIG="configs/single_pano_rtx5000x4_panocity_paired_low_to_high_luna_after_full_warmup_9h.yaml"
LUNA_OUT="$LUNA/logs/panocity_paired_low_to_high_4xrtx5000_luna_after_full_warmup_9h"
SEQ_LOG="$LUNA/logs/panocity_paired_low_to_high_4xrtx5000_stage_12h_sequence.log"

mkdir -p "$(dirname "$SEQ_LOG")" "$BASE_OUT" "$LUNA_OUT"

{
  echo "[sequence] started $(date --iso-8601=seconds)"
  echo "[sequence] root=$ROOT"
  echo "[sequence] baseline_config=$BASE_CONFIG"
  echo "[sequence] luna_config=$LUNA_CONFIG"
  echo "[sequence] python=$PYTHON"
  echo "[sequence] nproc_per_node=$NPROC_PER_NODE"
} | tee -a "$SEQ_LOG"

cd "$BASELINE"
echo "[sequence] stage1 baseline full warmup low384 started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
PYTHONPATH="$BASELINE${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" -m torch.distributed.run \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$BASE_PORT" \
  training/launch.py --config "$BASE_CONFIG" \
  2>&1 | tee -a "$BASE_OUT/train_3h_console.log"
echo "[sequence] stage1 baseline full warmup low384 finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"

if [[ ! -s "$BASE_CKPT" ]]; then
  echo "[sequence] missing baseline checkpoint: $BASE_CKPT" | tee -a "$SEQ_LOG"
  exit 1
fi

cd "$LUNA"
echo "[sequence] stage2/3 LUNA low384-to-high512 started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
PYTHONPATH="$LUNA${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" -m torch.distributed.run \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$LUNA_PORT" \
  training/train_pano_omega.py --config "$LUNA_CONFIG" \
  2>&1 | tee -a "$LUNA_OUT/train_9h.log"
echo "[sequence] stage2/3 LUNA low384-to-high512 finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"

if [[ -s "$BASE_OUT/loss.csv" && -s "$LUNA_OUT/loss.csv" ]]; then
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
    --out "$LUNA_OUT/loss_curve_full12h_low_to_high_smoothed_robust.png" \
    --title "PanoCity low-to-high 3h warmup + 9h LUNA robust smoothed loss" \
    2>&1 | tee -a "$SEQ_LOG"
fi

echo "[sequence] finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
