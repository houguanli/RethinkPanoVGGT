#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUNA="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$LUNA/.." && pwd)"
BASELINE="$ROOT/baseline01_vggt_omega"

PYTHON="${PYTHON:-/home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29673}"

CONFIG="configs/single_pano_4090_panocity_paired_full_lowres_luna_continue_to_9h.yaml"
BASE_OUT="$BASELINE/logs/panocity_paired_full_lowres_full_warmup_4090_3h_for_luna"
PREV_LUNA_OUT="$LUNA/logs/panocity_paired_full_lowres_luna_after_full_warmup_4090_9h"
OUT="$LUNA/logs/panocity_paired_full_lowres_luna_continue_to_9h_4090_2h"
SEQ_LOG="$LUNA/logs/panocity_paired_full_lowres_luna_continue_to_9h_4090_sequence.log"

mkdir -p "$OUT" "$(dirname "$SEQ_LOG")"

PREV_LUNA_OFFSET_HOURS="$("$PYTHON" - <<PY
import csv
path = "$PREV_LUNA_OUT/loss.csv"
last = 0.0
with open(path, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        try:
            last = float(row.get("elapsed_seconds") or 0.0)
        except ValueError:
            pass
print(f"{3.0 + last / 3600.0:.6f}")
PY
)"

{
  echo "[continue] started $(date --iso-8601=seconds)"
  echo "[continue] root=$ROOT"
  echo "[continue] config=$CONFIG"
  echo "[continue] previous_luna_offset_hours=$PREV_LUNA_OFFSET_HOURS"
  echo "[continue] python=$PYTHON"
  echo "[continue] cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
} | tee -a "$SEQ_LOG"

cd "$LUNA"
LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  PYTHONPATH="$LUNA${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" training/train_pano_omega.py --config "$CONFIG" \
  2>&1 | tee -a "$OUT/train_continue_2h.log"

if [[ ! -s "$OUT/last.pt" ]]; then
  echo "[continue] missing checkpoint: $OUT/last.pt" | tee -a "$SEQ_LOG"
  exit 1
fi

"$PYTHON" scripts/evaluate_depth_checkpoint.py \
  --config "$CONFIG" \
  --checkpoint "$OUT/last.pt" \
  --output "$OUT/validation_val100_summary.json" \
  --per-sample-csv "$OUT/validation_val100_per_sample.csv" \
  --train-loss-csv "$OUT/loss.csv" \
  --device cuda \
  --split val \
  --curriculum-bins all \
  --limit 100 \
  --hard-limit 0 \
  --num-workers 0 \
  --seed 123 \
  --no-progress \
  2>&1 | tee "$OUT/validation_val100_console.log"

"$PYTHON" scripts/plot_loss_csv.py \
  --run "warmup=$BASE_OUT/loss.csv" \
  --run "luna_7h=$PREV_LUNA_OUT/loss.csv@3" \
  --run "luna_continue=$OUT/loss.csv@$PREV_LUNA_OFFSET_HOURS" \
  --metric auto \
  --x elapsed_hours \
  --smooth-method rolling_median \
  --rolling-window 401 \
  --resample-seconds 10 \
  --clip-quantile 0.98 \
  --raw-alpha 0.10 \
  --out "$OUT/loss_curve_full12h_lowres_continued_smoothed_robust.png" \
  --title "PanoCity full-lowres continued to 9h LUNA robust smoothed loss" \
  2>&1 | tee -a "$SEQ_LOG"

echo "[continue] finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
