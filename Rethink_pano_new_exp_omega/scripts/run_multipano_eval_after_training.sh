#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUNA="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-/home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python}"
CONFIG="${CONFIG:-configs/multipano_rtx5000x4_mixed4_pano_all384_luna_after_full_warmup_9h.yaml}"
DATASET_ROOT="${DATASET_ROOT:-}"
LUNA_OUT="${LUNA_OUT:-logs/mixed4_pano_all384_4xrtx5000_multipano_after_full_warmup_9h}"
CHECKPOINT="${CHECKPOINT:-$LUNA_OUT/last.pt}"
TRAIN_LOSS_CSV="${TRAIN_LOSS_CSV:-$LUNA_OUT/loss.csv}"
EVAL_OUT="${EVAL_OUT:-$LUNA_OUT/eval_after_training}"
LIMIT_PER_DATASET="${LIMIT_PER_DATASET:-0}"
EVAL_DATASETS="${EVAL_DATASETS:-all}"
NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-2}"
WAIT_FOR_CHECKPOINT="${WAIT_FOR_CHECKPOINT:-1}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-0}"
POLL_SECONDS="${POLL_SECONDS:-60}"
FORCE_EVAL="${FORCE_EVAL:-0}"
SAMPLE_POLICY="${SAMPLE_POLICY:-anchor}"
PANO_COUNT_POLICY="${PANO_COUNT_POLICY:-panovggt}"
DATASET_PANO_COUNTS="${DATASET_PANO_COUNTS:-}"

SUMMARY_JSON="$EVAL_OUT/validation_mixed4_by_dataset_valtestfull_summary.json"

detect_gpus() {
  "$PYTHON" - <<'PY'
import torch
count = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(",".join(str(i) for i in range(count)))
PY
}

if [[ -z "${GPUS:-}" ]]; then
  GPUS="$(detect_gpus)"
fi

mkdir -p "$EVAL_OUT"
{
  echo "[eval-after-training] started $(date --iso-8601=seconds)"
  echo "[eval-after-training] luna=$LUNA"
  echo "[eval-after-training] config=$CONFIG"
  echo "[eval-after-training] dataset_root=${DATASET_ROOT:-<config>}"
  echo "[eval-after-training] luna_out=$LUNA_OUT"
  echo "[eval-after-training] checkpoint=$CHECKPOINT"
  echo "[eval-after-training] train_loss_csv=$TRAIN_LOSS_CSV"
  echo "[eval-after-training] eval_out=$EVAL_OUT"
  echo "[eval-after-training] gpus=${GPUS:-<none>}"
  echo "[eval-after-training] limit_per_dataset=$LIMIT_PER_DATASET"
  echo "[eval-after-training] datasets=$EVAL_DATASETS"
  echo "[eval-after-training] sample_policy=$SAMPLE_POLICY"
  echo "[eval-after-training] pano_count_policy=$PANO_COUNT_POLICY"
} | tee "$EVAL_OUT/eval_after_training.log"

if [[ -z "$GPUS" ]]; then
  echo "[eval-after-training] no CUDA GPU is visible; refusing to run 1B checkpoint eval on CPU." | tee -a "$EVAL_OUT/eval_after_training.log"
  exit 1
fi

if [[ -s "$SUMMARY_JSON" && "$FORCE_EVAL" != "1" ]]; then
  echo "[eval-after-training] existing summary found, skipping: $SUMMARY_JSON" | tee -a "$EVAL_OUT/eval_after_training.log"
  exit 0
fi

start_ts="$(date +%s)"
while [[ ! -s "$CHECKPOINT" ]]; do
  if [[ "$WAIT_FOR_CHECKPOINT" != "1" ]]; then
    echo "[eval-after-training] checkpoint is missing: $CHECKPOINT" | tee -a "$EVAL_OUT/eval_after_training.log"
    exit 1
  fi
  now_ts="$(date +%s)"
  elapsed=$((now_ts - start_ts))
  if [[ "$WAIT_TIMEOUT_SECONDS" -gt 0 && "$elapsed" -ge "$WAIT_TIMEOUT_SECONDS" ]]; then
    echo "[eval-after-training] timed out waiting for checkpoint after ${elapsed}s: $CHECKPOINT" | tee -a "$EVAL_OUT/eval_after_training.log"
    exit 1
  fi
  echo "[eval-after-training] waiting for checkpoint (${elapsed}s): $CHECKPOINT" | tee -a "$EVAL_OUT/eval_after_training.log"
  sleep "$POLL_SECONDS"
done

echo "[eval-after-training] checkpoint ready; launching mixed4 eval $(date --iso-8601=seconds)" | tee -a "$EVAL_OUT/eval_after_training.log"
env \
  PYTHON="$PYTHON" \
  GPUS="$GPUS" \
  CONFIG="$CONFIG" \
  DATASET_ROOT="$DATASET_ROOT" \
  LUNA_OUT="$LUNA_OUT" \
  CHECKPOINT="$CHECKPOINT" \
  TRAIN_LOSS_CSV="$TRAIN_LOSS_CSV" \
  EVAL_OUT="$EVAL_OUT" \
  LIMIT_PER_DATASET="$LIMIT_PER_DATASET" \
  EVAL_DATASETS="$EVAL_DATASETS" \
  NUM_WORKERS_PER_GPU="$NUM_WORKERS_PER_GPU" \
  SAMPLE_POLICY="$SAMPLE_POLICY" \
  PANO_COUNT_POLICY="$PANO_COUNT_POLICY" \
  DATASET_PANO_COUNTS="$DATASET_PANO_COUNTS" \
  bash "$LUNA/scripts/run_multipano_mixed4_eval_4gpu.sh" \
  2>&1 | tee -a "$EVAL_OUT/eval_after_training.log"

if [[ ! -s "$SUMMARY_JSON" ]]; then
  echo "[eval-after-training] eval finished but summary is missing: $SUMMARY_JSON" | tee -a "$EVAL_OUT/eval_after_training.log"
  exit 1
fi

echo "[eval-after-training] finished $(date --iso-8601=seconds)" | tee -a "$EVAL_OUT/eval_after_training.log"
echo "[eval-after-training] summary=$SUMMARY_JSON" | tee -a "$EVAL_OUT/eval_after_training.log"
