#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUNA="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$LUNA/.." && pwd)"
BASELINE="$ROOT/baseline01_vggt_omega"

PYTHON="${PYTHON:-/home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-29691}"
LUNA_PORT="${LUNA_PORT:-29692}"
PANOVGGT_ROOT="${PANOVGGT_ROOT:-/mnt/e/PanoVGGT_minimal_datasets/datasets}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/home/aoki/RethinkPanoVGGT_omega/ckpt/vggt_omega_1b_512.pt}"

BASE_CONFIG="mixed4_pano_weighted_officialscale_full_warmup_4090_3h_for_luna"
BASE_OUT="$BASELINE/logs/$BASE_CONFIG"
BASE_CKPT="$BASE_OUT/ckpts/checkpoint.pt"
CALIB_JSON="$BASE_OUT/depth_scale_calibration.json"
CALIB_LOG="$BASE_OUT/depth_scale_calibration.log"
CALIB_SAMPLES_PER_DATASET="${CALIB_SAMPLES_PER_DATASET:-8}"
CALIB_MAX_PIXELS_PER_SAMPLE="${CALIB_MAX_PIXELS_PER_SAMPLE:-50000}"
NUM_YAW="${NUM_YAW:-4}"
EVAL_LIMIT_PER_DATASET="${EVAL_LIMIT_PER_DATASET:-0}"
EVAL_SUFFIX="valtestfull"
if [[ "$EVAL_LIMIT_PER_DATASET" != "0" ]]; then
  EVAL_SUFFIX="valtest${EVAL_LIMIT_PER_DATASET}"
fi

LUNA_CONFIG="configs/multipano_4090_mixed4_pano_weighted_officialscale_luna_after_full_warmup_9h.yaml"
LUNA_OUT="$LUNA/logs/mixed4_pano_weighted_officialscale_multipano_after_full_warmup_4090_9h"
SEQ_LOG="$LUNA/logs/mixed4_pano_weighted_officialscale_multipano_stage_4090_12h_sequence.log"

mkdir -p "$(dirname "$SEQ_LOG")"

if [[ "${CLEAN_OUTPUT:-0}" == "1" ]]; then
  rm -rf "$BASE_OUT" "$LUNA_OUT" "$SEQ_LOG" "$LUNA/logs/debug_mixed4_pano_weighted_officialscale_multipano_after_full_warmup_4090_9h"
fi
mkdir -p "$BASE_OUT" "$LUNA_OUT"

{
  echo "[sequence] started $(date --iso-8601=seconds)"
  echo "[sequence] root=$ROOT"
  echo "[sequence] baseline_config=$BASE_CONFIG"
  echo "[sequence] luna_config=$LUNA_CONFIG"
  echo "[sequence] python=$PYTHON"
  echo "[sequence] cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "[sequence] panovggt_root=$PANOVGGT_ROOT"
  echo "[sequence] base_checkpoint=$BASE_CHECKPOINT"
  echo "[sequence] calibration_samples_per_dataset=$CALIB_SAMPLES_PER_DATASET"
  echo "[sequence] num_yaw=$NUM_YAW"
  echo "[sequence] eval_limit_per_dataset=$EVAL_LIMIT_PER_DATASET"
} | tee -a "$SEQ_LOG"

echo "[sequence] building/checking mixed4 official indexes under $PANOVGGT_ROOT $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
cd "$LUNA"
PYTHONPATH="$LUNA${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" scripts/build_mixed4_official_indexes.py \
  --root "$PANOVGGT_ROOT" \
  --train-fraction 0.95 \
  --seed 42 \
  2>&1 | tee -a "$SEQ_LOG"

PRED_DEPTH_SCALE="${PRED_DEPTH_SCALE:-}"
if [[ "${SKIP_CALIBRATION:-0}" != "1" ]]; then
  echo "[sequence] calibrating unified pred_depth_scale $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
  cd "$BASELINE"
  PYTHONPATH="$BASELINE${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" scripts/calibrate_mixed4_depth_scale.py \
    --root "$PANOVGGT_ROOT" \
    --checkpoint "$BASE_CHECKPOINT" \
    --output "$CALIB_JSON" \
    --samples-per-dataset "$CALIB_SAMPLES_PER_DATASET" \
    --max-pixels-per-sample "$CALIB_MAX_PIXELS_PER_SAMPLE" \
    --image-size 384 \
    --patch-size 16 \
    --num-yaw "$NUM_YAW" \
    --pitch-degrees -15.0 \
    --fov-degrees 75.0 \
    --depth-max-m 80.0 \
    --device cuda \
    --print-scale \
    2>&1 | tee "$CALIB_LOG"
  PRED_DEPTH_SCALE="$("$PYTHON" - "$CALIB_JSON" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(payload["recommended_pred_depth_scale"])
PY
)"
elif [[ -z "$PRED_DEPTH_SCALE" && -s "$CALIB_JSON" ]]; then
  PRED_DEPTH_SCALE="$("$PYTHON" - "$CALIB_JSON" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(payload["recommended_pred_depth_scale"])
PY
)"
fi

if [[ -z "$PRED_DEPTH_SCALE" ]]; then
  echo "[sequence] PRED_DEPTH_SCALE is empty; run calibration or set PRED_DEPTH_SCALE explicitly." | tee -a "$SEQ_LOG"
  exit 1
fi
echo "[sequence] unified pred_depth_scale=$PRED_DEPTH_SCALE" | tee -a "$SEQ_LOG"

cd "$BASELINE"
echo "[sequence] stage1 baseline full warmup weighted official-scale low384 started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$BASE_PORT" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  PYTHONPATH="$BASELINE${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" training/launch.py --config "$BASE_CONFIG" \
    model.checkpoint_path="$BASE_CHECKPOINT" \
    loss.depth.pred_depth_scale="$PRED_DEPTH_SCALE" \
    loss.depth.mode=log_huber \
    ++loss.depth.depth_scale_alignment=sample_lstsq \
    ++loss.depth.depth_scale_alignment_min=0.05 \
    ++loss.depth.depth_scale_alignment_max=50.0 \
  2>&1 | tee -a "$BASE_OUT/train_3h_console.log"
echo "[sequence] stage1 baseline full warmup weighted official-scale low384 finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"

if [[ ! -s "$BASE_CKPT" ]]; then
  echo "[sequence] missing baseline checkpoint: $BASE_CKPT" | tee -a "$SEQ_LOG"
  exit 1
fi

cd "$LUNA"
echo "[sequence] stage2/3 multi-pano LUNA weighted official-scale low384 started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$LUNA_PORT" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  PYTHONPATH="$LUNA${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" training/train_pano_omega.py --config "$LUNA_CONFIG" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --checkpoint "$BASE_CKPT" \
    --pred-depth-scale "$PRED_DEPTH_SCALE" \
    --depth-loss-mode log_huber \
    --depth-scale-alignment sample_lstsq \
    --depth-scale-alignment-min 0.05 \
    --depth-scale-alignment-max 50.0 \
  2>&1 | tee -a "$LUNA_OUT/train_9h.log"
echo "[sequence] stage2/3 multi-pano LUNA weighted official-scale low384 finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"

if [[ ! -s "$LUNA_OUT/last.pt" ]]; then
  echo "[sequence] missing LUNA checkpoint: $LUNA_OUT/last.pt" | tee -a "$SEQ_LOG"
  exit 1
fi

echo "[sequence] mixed4 validation $EVAL_SUFFIX started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
"$PYTHON" scripts/evaluate_mixed4_depth_checkpoint.py \
  --config "$LUNA_CONFIG" \
  --checkpoint "$LUNA_OUT/last.pt" \
  --output "$LUNA_OUT/validation_mixed4_by_dataset_${EVAL_SUFFIX}_summary.json" \
  --per-sample-csv "$LUNA_OUT/validation_mixed4_by_dataset_${EVAL_SUFFIX}_per_sample.csv" \
  --train-loss-csv "$LUNA_OUT/loss.csv" \
  --device cuda \
  --limit-per-dataset "$EVAL_LIMIT_PER_DATASET" \
  --num-workers 2 \
  --seed 123 \
  --no-progress \
  2>&1 | tee "$LUNA_OUT/validation_mixed4_by_dataset_${EVAL_SUFFIX}_console.log"
echo "[sequence] mixed4 validation $EVAL_SUFFIX finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"

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
  --out "$LUNA_OUT/loss_curve_full12h_weighted_officialscale_multipano_smoothed_robust.png" \
  --title "Mixed4 weighted official-scale 3h full warmup + 9h multi-pano LUNA robust smoothed loss" \
  2>&1 | tee -a "$SEQ_LOG"

echo "[sequence] finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
