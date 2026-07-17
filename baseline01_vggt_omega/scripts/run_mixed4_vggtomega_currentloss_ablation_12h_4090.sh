#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$BASELINE/.." && pwd)"
LUNA="$ROOT/Rethink_pano_new_exp_omega"

PYTHON="${PYTHON:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29731}"
PANOVGGT_ROOT="${PANOVGGT_ROOT:-$(cd "$ROOT/.." && pwd)/panovggt}"
BAD_SAMPLE_LIST="${BAD_SAMPLE_LIST:-$LUNA/configs/structured3d_bad_scenes.txt}"
CONFIG="${CONFIG:-mixed4_pano_vggtomega_multiview_groups_4090_12h}"
EXP_NAME="${EXP_NAME:-mixed4_pano_vggtomega_panovggt_counts_4090_12h}"
OUT="${OUT:-$BASELINE/logs/$EXP_NAME}"
SEQ_LOG="$OUT/run_12h_then_full_eval.log"
CKPT="$OUT/ckpts/checkpoint.pt"
CALIB_JSON="$OUT/depth_scale_calibration.json"
CALIB_LOG="$OUT/depth_scale_calibration.log"
CALIB_SAMPLES_PER_DATASET="${CALIB_SAMPLES_PER_DATASET:-8}"
CALIB_MAX_PIXELS_PER_SAMPLE="${CALIB_MAX_PIXELS_PER_SAMPLE:-50000}"
EVAL_LIMIT_PER_DATASET="${EVAL_LIMIT_PER_DATASET:-0}"
EVAL_OUT="${EVAL_OUT:-$OUT/eval_full_anchor_panovggt_counts}"
USE_CALIBRATED_DEPTH_SCALE="${USE_CALIBRATED_DEPTH_SCALE:-0}"
LOSS_DEPTH_MODE="${LOSS_DEPTH_MODE:-}"

if [[ "${CLEAN_OUTPUT:-0}" == "1" ]]; then
  rm -rf "$OUT" "$EVAL_OUT"
fi
mkdir -p "$OUT" "$EVAL_OUT"

{
  echo "[ablation] started $(date --iso-8601=seconds)"
  echo "[ablation] baseline=$BASELINE"
  echo "[ablation] config=$CONFIG"
  echo "[ablation] exp_name=$EXP_NAME"
  echo "[ablation] python=$PYTHON"
  echo "[ablation] cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "[ablation] panovggt_root=$PANOVGGT_ROOT"
  echo "[ablation] bad_sample_list=$BAD_SAMPLE_LIST"
  echo "[ablation] eval_out=$EVAL_OUT"
  echo "[ablation] eval_limit_per_dataset=$EVAL_LIMIT_PER_DATASET"
  echo "[ablation] use_calibrated_depth_scale=$USE_CALIBRATED_DEPTH_SCALE"
} | tee -a "$SEQ_LOG"

echo "[ablation] building/checking mixed4 official indexes $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
cd "$LUNA"
PYTHONPATH="$LUNA${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" scripts/build_mixed4_official_indexes.py \
  --root "$PANOVGGT_ROOT" \
  --train-fraction 0.95 \
  --seed 42 \
  --bad-scene-list "$BAD_SAMPLE_LIST" \
  2>&1 | tee -a "$SEQ_LOG"

PRED_DEPTH_SCALE="${PRED_DEPTH_SCALE:-}"
if [[ "$USE_CALIBRATED_DEPTH_SCALE" == "1" && "${SKIP_CALIBRATION:-0}" != "1" ]]; then
  echo "[ablation] calibrating unified pred_depth_scale $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
  cd "$BASELINE"
  PYTHONPATH="$BASELINE${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" scripts/calibrate_mixed4_depth_scale.py \
    --root "$PANOVGGT_ROOT" \
    --output "$CALIB_JSON" \
    --samples-per-dataset "$CALIB_SAMPLES_PER_DATASET" \
    --max-pixels-per-sample "$CALIB_MAX_PIXELS_PER_SAMPLE" \
    --image-size 384 \
    --patch-size 16 \
    --num-yaw 8 \
    --pitch-degrees -15.0 \
    --fov-degrees 75.0 \
    --depth-max-m 80.0 \
    --bad-sample-list "$BAD_SAMPLE_LIST" \
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
elif [[ "$USE_CALIBRATED_DEPTH_SCALE" == "1" && -z "$PRED_DEPTH_SCALE" && -s "$CALIB_JSON" ]]; then
  PRED_DEPTH_SCALE="$("$PYTHON" - "$CALIB_JSON" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(payload["recommended_pred_depth_scale"])
PY
)"
fi

TRAIN_OVERRIDES=()
if [[ -n "$PRED_DEPTH_SCALE" ]]; then
  TRAIN_OVERRIDES+=(loss.depth.pred_depth_scale="$PRED_DEPTH_SCALE")
  echo "[ablation] override pred_depth_scale=$PRED_DEPTH_SCALE" | tee -a "$SEQ_LOG"
else
  echo "[ablation] pred_depth_scale=from config" | tee -a "$SEQ_LOG"
fi
if [[ -n "$LOSS_DEPTH_MODE" ]]; then
  TRAIN_OVERRIDES+=(loss.depth.mode="$LOSS_DEPTH_MODE")
  echo "[ablation] override loss.depth.mode=$LOSS_DEPTH_MODE" | tee -a "$SEQ_LOG"
else
  echo "[ablation] loss.depth.mode=from config" | tee -a "$SEQ_LOG"
fi

echo "[ablation] training pure VGGTOmega PanoVGGT-count baseline started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
cd "$BASELINE"
LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  PYTHONPATH="$BASELINE${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" training/launch.py --config "$CONFIG" \
    "${TRAIN_OVERRIDES[@]}" \
  2>&1 | tee -a "$OUT/train_12h_console.log"
echo "[ablation] training finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"

if [[ ! -s "$CKPT" ]]; then
  echo "[ablation] missing checkpoint: $CKPT" | tee -a "$SEQ_LOG"
  exit 1
fi

echo "[ablation] full mixed4 test eval started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
PYTHON="$PYTHON" \
CONFIG="$CONFIG" \
CHECKPOINT="$CKPT" \
DATASET_ROOT="$PANOVGGT_ROOT" \
TRAIN_LOSS_CSV="$OUT/loss.csv" \
OUT="$EVAL_OUT" \
LIMIT_PER_DATASET="$EVAL_LIMIT_PER_DATASET" \
DEVICE=cuda \
AMP_DTYPE=bfloat16 \
PRED_DEPTH_SCALE="$PRED_DEPTH_SCALE" \
FOREGROUND=1 \
  bash scripts/run_eval_panovggt_counts.sh \
  2>&1 | tee "$EVAL_OUT/run_eval_panovggt_counts_console.log"
echo "[ablation] full mixed4 test eval finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"

echo "[ablation] plotting loss curve $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
"$PYTHON" scripts/plot_loss_curve.py \
  --loss-csv "$OUT/loss.csv" \
  --output "$OUT/loss_curve_smoothed.png" \
  --smooth 401 \
  2>&1 | tee -a "$SEQ_LOG"

echo "[ablation] finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
