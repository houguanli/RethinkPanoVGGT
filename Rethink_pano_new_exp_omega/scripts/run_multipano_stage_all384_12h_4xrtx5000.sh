#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUNA="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$LUNA/.." && pwd)"
BASELINE="$ROOT/baseline01_vggt_omega"

PYTHON="${PYTHON:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "${PANOVGGT_ROOT:-}" ]]; then
  if [[ -d /mnt/e/PanoVGGT_minimal_datasets/datasets ]]; then
    PANOVGGT_ROOT="/mnt/e/PanoVGGT_minimal_datasets/datasets"
  else
    PANOVGGT_ROOT="$(cd "$ROOT/.." && pwd)/panovggt"
  fi
fi
if [[ -z "${BASE_CHECKPOINT:-}" ]]; then
  for candidate in \
    /whitehole/AOKI/vggt-omega/ckpt/vggt_omega_1b_512.pt \
    /whitehole/AOKI/RethinkPanoVGGT_omega/ckpt/vggt_omega_1b_512.pt \
    /home/aoki/RethinkPanoVGGT_omega_compare_methods_only/ckpt/VGGT-Omega/vggt_omega_1b_512.pt \
    /home/aoki/RethinkPanoVGGT_omega/ckpt/vggt_omega_1b_512.pt \
    "$LUNA/ckpt/vggt_omega_1b_512.pt"; do
    if [[ -s "$candidate" ]]; then
      BASE_CHECKPOINT="$candidate"
      break
    fi
  done
fi
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/whitehole/AOKI/vggt-omega/ckpt/vggt_omega_1b_512.pt}"
WARMUP_INIT_CHECKPOINT="${WARMUP_INIT_CHECKPOINT:-$BASE_CHECKPOINT}"
if [[ ! -s "$WARMUP_INIT_CHECKPOINT" ]]; then
  WARMUP_INIT_CHECKPOINT="$BASE_CHECKPOINT"
fi
USE_SINGLE_DEPTH_SCALE_INIT="${USE_SINGLE_DEPTH_SCALE_INIT:-1}"
SINGLE_WEIGHTED_PRED_DEPTH_SCALE="${SINGLE_WEIGHTED_PRED_DEPTH_SCALE:-2.827019691467285}"

WARMUP_CONFIG="${WARMUP_CONFIG:-configs/multipano_rtx5000x4_mixed4_pano_omega_warmup_3h_for_luna.yaml}"
WARMUP_OUT="${WARMUP_OUT:-$LUNA/logs/mixed4_pano_omega_multipano_warmup_3h_for_luna}"
WARMUP_CKPT="$WARMUP_OUT/last.pt"
CALIB_JSON="$WARMUP_OUT/depth_scale_calibration.json"
CALIB_LOG="$WARMUP_OUT/depth_scale_calibration.log"
CALIB_SAMPLES_PER_DATASET="${CALIB_SAMPLES_PER_DATASET:-8}"
CALIB_MAX_PIXELS_PER_SAMPLE="${CALIB_MAX_PIXELS_PER_SAMPLE:-50000}"
NUM_YAW="${NUM_YAW:-4}"

LUNA_CONFIG="${LUNA_CONFIG:-configs/multipano_rtx5000x4_mixed4_pano_all384_luna_after_full_warmup_9h.yaml}"
LUNA_OUT="${LUNA_OUT:-$LUNA/logs/mixed4_pano_all384_4xrtx5000_multipano_after_full_warmup_9h}"
SEQ_LOG="${SEQ_LOG:-$LUNA/logs/mixed4_pano_all384_4xrtx5000_multipano_stage_12h_sequence.log}"
EXTRA_TRAIN_ARGS_ARRAY=()
if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_TRAIN_ARGS_ARRAY=($EXTRA_TRAIN_ARGS)
fi

mkdir -p "$(dirname "$SEQ_LOG")"
if [[ "${CLEAN_OUTPUT:-0}" == "1" ]]; then
  rm -rf "$WARMUP_OUT" "$LUNA_OUT" "$SEQ_LOG" "$LUNA/logs/debug_mixed4_pano_all384_4xrtx5000_multipano_after_full_warmup_9h"
fi
mkdir -p "$WARMUP_OUT" "$LUNA_OUT"

{
  echo "[sequence] started $(date --iso-8601=seconds)"
  echo "[sequence] root=$ROOT"
  echo "[sequence] warmup_config=$WARMUP_CONFIG"
  echo "[sequence] warmup_checkpoint=$WARMUP_CKPT"
  echo "[sequence] luna_config=$LUNA_CONFIG"
  echo "[sequence] python=$PYTHON"
  echo "[sequence] nproc_per_node=$NPROC_PER_NODE"
  echo "[sequence] pytorch_cuda_alloc_conf=$PYTORCH_CUDA_ALLOC_CONF"
  echo "[sequence] panovggt_root=$PANOVGGT_ROOT"
  echo "[sequence] base_checkpoint=$BASE_CHECKPOINT"
  echo "[sequence] warmup_init_checkpoint=$WARMUP_INIT_CHECKPOINT"
  echo "[sequence] use_single_depth_scale_init=$USE_SINGLE_DEPTH_SCALE_INIT"
  echo "[sequence] calibration_samples_per_dataset=$CALIB_SAMPLES_PER_DATASET"
  echo "[sequence] num_yaw=$NUM_YAW"
  echo "[sequence] extra_train_args=${EXTRA_TRAIN_ARGS:-}"
} | tee -a "$SEQ_LOG"

if [[ "${SKIP_INDEX_BUILD:-0}" != "1" ]]; then
  echo "[sequence] building mixed4 official indexes under $PANOVGGT_ROOT $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
  PYTHONPATH="$LUNA${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" "$LUNA/scripts/build_mixed4_official_indexes.py" \
    --root "$PANOVGGT_ROOT" \
    --train-fraction 0.95 \
    --seed 42 \
    2>&1 | tee -a "$SEQ_LOG"
else
  echo "[sequence] mixed4 index rebuild skipped because SKIP_INDEX_BUILD=1" | tee -a "$SEQ_LOG"
fi

PRED_DEPTH_SCALE="${PRED_DEPTH_SCALE:-}"
if [[ "$USE_SINGLE_DEPTH_SCALE_INIT" == "1" ]]; then
  PRED_DEPTH_SCALE="${PRED_DEPTH_SCALE:-$SINGLE_WEIGHTED_PRED_DEPTH_SCALE}"
  SKIP_CALIBRATION="${SKIP_CALIBRATION:-1}"
fi
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

cd "$LUNA"
WARMUP_CKPT_PREEXISTING=0
if [[ -s "$WARMUP_CKPT" && "${FORCE_WARMUP:-0}" != "1" ]]; then
  WARMUP_CKPT_PREEXISTING=1
  echo "[sequence] stage1 multi-pano Omega warmup skipped; existing checkpoint found: $WARMUP_CKPT" | tee -a "$SEQ_LOG"
else
  echo "[sequence] stage1 multi-pano Omega warmup low384 started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
  WARMUP_DURATION_ARGS=()
  if [[ -n "${WARMUP_MAX_DURATION_MINUTES:-}" ]]; then
    WARMUP_DURATION_ARGS=(--max-duration-minutes "$WARMUP_MAX_DURATION_MINUTES")
    echo "[sequence] warmup_duration_override=${WARMUP_DURATION_ARGS[*]}" | tee -a "$SEQ_LOG"
  fi
  PYTHONPATH="$LUNA${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NPROC_PER_NODE" \
    training/train_pano_omega.py --config "$WARMUP_CONFIG" \
    "${WARMUP_DURATION_ARGS[@]}" \
    --output-dir "$WARMUP_OUT" \
    --tensorboard-dir "$WARMUP_OUT/tensorboard" \
    --debug-dir "$WARMUP_OUT/debug" \
    --dataset-root "$PANOVGGT_ROOT" \
    --checkpoint "$WARMUP_INIT_CHECKPOINT" \
    --pred-depth-scale "$PRED_DEPTH_SCALE" \
    --depth-loss-mode log_huber \
    --depth-scale-alignment none \
    --depth-scale-diagnostics-alignment sample_l1_depth_weighted \
    --depth-scale-alignment-min 0.05 \
    --depth-scale-alignment-max 1000000.0 \
    --no-inherit-checkpoint-training-defaults \
    "${EXTRA_TRAIN_ARGS_ARRAY[@]}" \
    2>&1 | tee -a "$WARMUP_OUT/train_3h.log"
  echo "[sequence] stage1 multi-pano Omega warmup low384 finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
fi

if [[ ! -s "$WARMUP_CKPT" ]]; then
  echo "[sequence] missing warmup checkpoint: $WARMUP_CKPT" | tee -a "$SEQ_LOG"
  exit 1
fi

if [[ -s "$LUNA_OUT/loss.csv" && ! -s "$LUNA_OUT/last.pt" && "${PRESERVE_CRASHED_LUNA:-1}" == "1" ]]; then
  CRASHED_OUT="${LUNA_OUT}_crashed_$(date +%Y%m%d_%H%M%S)"
  echo "[sequence] preserving incomplete LUNA output: $LUNA_OUT -> $CRASHED_OUT" | tee -a "$SEQ_LOG"
  mv "$LUNA_OUT" "$CRASHED_OUT"
  mkdir -p "$LUNA_OUT"
fi
echo "[sequence] stage2/3 multi-pano LUNA all384 started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
LUNA_DURATION_ARGS=()
if [[ -n "${LUNA_MAX_DURATION_MINUTES:-}" ]]; then
  LUNA_DURATION_ARGS=(--max-duration-minutes "$LUNA_MAX_DURATION_MINUTES")
elif [[ "$WARMUP_CKPT_PREEXISTING" == "1" ]]; then
  LUNA_DURATION_ARGS=(--max-duration-minutes 720.0)
fi
if [[ "${#LUNA_DURATION_ARGS[@]}" -gt 0 ]]; then
  echo "[sequence] luna_duration_override=${LUNA_DURATION_ARGS[*]}" | tee -a "$SEQ_LOG"
fi
PYTHONPATH="$LUNA${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  training/train_pano_omega.py --config "$LUNA_CONFIG" \
  "${LUNA_DURATION_ARGS[@]}" \
  --base-checkpoint "$WARMUP_INIT_CHECKPOINT" \
  --output-dir "$LUNA_OUT" \
  --tensorboard-dir "$LUNA_OUT/tensorboard" \
  --debug-dir "$LUNA_OUT/debug" \
  --dataset-root "$PANOVGGT_ROOT" \
  --checkpoint "$WARMUP_CKPT" \
  --pred-depth-scale "$PRED_DEPTH_SCALE" \
  --depth-loss-mode log_huber \
  --depth-scale-alignment none \
  --depth-scale-diagnostics-alignment sample_l1_depth_weighted \
  --depth-scale-alignment-min 0.05 \
  --depth-scale-alignment-max 1000000.0 \
  --no-inherit-checkpoint-training-defaults \
  "${EXTRA_TRAIN_ARGS_ARRAY[@]}" \
  2>&1 | tee -a "$LUNA_OUT/train_luna.log"
echo "[sequence] stage2/3 multi-pano LUNA all384 finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"

if [[ -s "$WARMUP_OUT/loss.csv" && -s "$LUNA_OUT/loss.csv" ]]; then
  "$PYTHON" scripts/plot_loss_csv.py \
    --run "warmup=$WARMUP_OUT/loss.csv" \
    --run "luna=$LUNA_OUT/loss.csv@3" \
    --metric auto \
    --x elapsed_hours \
    --smooth-method rolling_median \
    --rolling-window 401 \
    --resample-seconds 10 \
    --clip-quantile 0.98 \
    --raw-alpha 0.10 \
    --out "$LUNA_OUT/loss_curve_all384_multipano_smoothed_robust.png" \
    --title "Mixed4 all384 3h multi-pano Omega warmup + 9h LUNA robust smoothed loss" \
    2>&1 | tee -a "$SEQ_LOG"
fi

if [[ "${RUN_VALIDATION:-1}" == "1" ]]; then
  if [[ ! -s "$LUNA_OUT/last.pt" ]]; then
    echo "[sequence] missing LUNA checkpoint for validation: $LUNA_OUT/last.pt" | tee -a "$SEQ_LOG"
    exit 1
  fi
  if [[ -z "${VALIDATION_GPUS:-}" ]]; then
    VALIDATION_GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  fi
  VALIDATION_OUT="${VALIDATION_OUT:-$LUNA_OUT/eval_full_4gpu}"
  VALIDATION_LIMIT_PER_DATASET="${VALIDATION_LIMIT_PER_DATASET:-0}"
  VALIDATION_DATASETS="${VALIDATION_DATASETS:-all}"
  echo "[sequence] mixed4 4GPU validation started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
  env \
    PYTHON="$PYTHON" \
    GPUS="$VALIDATION_GPUS" \
    CONFIG="$LUNA_CONFIG" \
    DATASET_ROOT="$PANOVGGT_ROOT" \
    LUNA_OUT="$LUNA_OUT" \
    CHECKPOINT="$LUNA_OUT/last.pt" \
    TRAIN_LOSS_CSV="$LUNA_OUT/loss.csv" \
    EVAL_OUT="$VALIDATION_OUT" \
    LIMIT_PER_DATASET="$VALIDATION_LIMIT_PER_DATASET" \
    EVAL_DATASETS="$VALIDATION_DATASETS" \
    bash "$LUNA/scripts/run_multipano_mixed4_eval_4gpu.sh" \
    2>&1 | tee -a "$SEQ_LOG"
  VALIDATION_SUMMARY="$VALIDATION_OUT/validation_mixed4_by_dataset_valtestfull_summary.json"
  if [[ ! -s "$VALIDATION_SUMMARY" ]]; then
    echo "[sequence] validation summary missing after eval: $VALIDATION_SUMMARY" | tee -a "$SEQ_LOG"
    exit 1
  fi
  echo "[sequence] validation summary=$VALIDATION_SUMMARY" | tee -a "$SEQ_LOG"
  echo "[sequence] mixed4 4GPU validation finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
else
  echo "[sequence] validation skipped because RUN_VALIDATION=0" | tee -a "$SEQ_LOG"
fi

echo "[sequence] finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
