#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WARMUP_CHECKPOINT="${1:-${WARMUP_CHECKPOINT:-}}"
if [[ -z "$WARMUP_CHECKPOINT" || ! -s "$WARMUP_CHECKPOINT" ]]; then
  echo "usage: bash scripts/run_server_luna_9h_10p8w384.sh /absolute/path/to/warmup/last.pt" >&2
  exit 2
fi

if [[ "${PYTHON:-}" == */* ]]; then
  RESOLVED_PYTHON="$PYTHON"
else
  RESOLVED_PYTHON="$(command -v "${PYTHON:-python}" || true)"
fi
if [[ -z "$RESOLVED_PYTHON" || ! -x "$RESOLVED_PYTHON" ]]; then
  echo "[luna-9h] python executable not found; activate the conda environment or set PYTHON=/absolute/path/to/python" >&2
  exit 2
fi

CONFIG="configs/multipano_rtx5000x4_panocity_luna_memory_probe_10p_8w384_fov90_pitch30_erp1024x512.yaml"
DATASET_ROOT="${DATASET_ROOT:-/whitehole/AOKI/panovggt}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LUNA_OUT="${LUNA_OUT:-logs/server_panocity_luna_9h_10p_8w384_fov90_pitch30_${RUN_TAG}}"
RUN_VALIDATION="${RUN_VALIDATION:-1}"

echo "[luna-9h] training output=$LUNA_OUT"
env \
  PYTHON="$RESOLVED_PYTHON" \
  MEMORY_PROBE_CONFIG="$CONFIG" \
  DATASET_ROOT="$DATASET_ROOT" \
  OUTPUT_DIR="$LUNA_OUT" \
  DURATION_MINUTES="${LUNA_DURATION_MINUTES:-540}" \
  NPROC_PER_NODE="${NPROC_PER_NODE:-4}" \
  BASE_CHECKPOINT="${BASE_CHECKPOINT:-}" \
  bash scripts/run_server_luna_memory_probe_10p8w384.sh "$WARMUP_CHECKPOINT"

if [[ "$RUN_VALIDATION" != "1" ]]; then
  echo "[luna-9h] validation skipped because RUN_VALIDATION=$RUN_VALIDATION"
  exit 0
fi
if [[ ! -s "$LUNA_OUT/last.pt" ]]; then
  echo "[luna-9h] training completed without last.pt: $LUNA_OUT/last.pt" >&2
  exit 1
fi

echo "[luna-9h] launching automatic Panocity anchor evaluation"
env \
  PYTHON="$RESOLVED_PYTHON" \
  GPUS="${VALIDATION_GPUS:-0,1,2,3}" \
  CONFIG="$CONFIG" \
  DATASET_ROOT="$DATASET_ROOT" \
  LUNA_OUT="$LUNA_OUT" \
  CHECKPOINT="$LUNA_OUT/last.pt" \
  TRAIN_LOSS_CSV="$LUNA_OUT/loss.csv" \
  EVAL_OUT="${VALIDATION_OUT:-$LUNA_OUT/eval_panocity_anchor_10p8w384_4gpu}" \
  EVAL_DATASETS=panocity \
  LIMIT_PER_DATASET="${VALIDATION_LIMIT_PER_DATASET:-0}" \
  SAMPLE_POLICY=anchor \
  PANO_COUNT_POLICY=panovggt \
  CAMERA_EVAL_MAX_PANOS=10 \
  WINDOW_SIZE=384 \
  NUM_YAW=4 \
  PRINT_EACH_SAMPLE="${PRINT_EACH_SAMPLE:-1}" \
  BASE_CHECKPOINT_OVERRIDE="${BASE_CHECKPOINT:-}" \
  bash scripts/run_multipano_eval_after_training.sh

