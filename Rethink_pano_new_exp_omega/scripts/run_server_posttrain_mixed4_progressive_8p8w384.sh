#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LUNA_OUT="${1:-${LUNA_OUT:-}}"
if [[ -z "$LUNA_OUT" || ! -d "$LUNA_OUT" ]]; then
  echo "usage: bash scripts/run_server_posttrain_mixed4_progressive_8p8w384.sh /path/to/luna_output" >&2
  exit 2
fi
if [[ ! -s "$LUNA_OUT/loss.csv" ]]; then
  echo "[posttrain] loss CSV not found: $LUNA_OUT/loss.csv" >&2
  exit 2
fi

if [[ "${PYTHON:-}" == */* ]]; then
  RESOLVED_PYTHON="$PYTHON"
else
  RESOLVED_PYTHON="$(command -v "${PYTHON:-python}" || true)"
fi
if [[ -z "$RESOLVED_PYTHON" || ! -x "$RESOLVED_PYTHON" ]]; then
  echo "[posttrain] python executable not found; activate conda or set PYTHON=/absolute/path/to/python" >&2
  exit 2
fi

CONFIG="configs/multipano_rtx5000x4_mixed4_pano_all384_luna_after_full_warmup_9h.yaml"
DATASET_ROOT="${DATASET_ROOT:-/whitehole/AOKI/panovggt}"
POSTTRAIN_RUN_VALIDATION="${POSTTRAIN_RUN_VALIDATION:-1}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-}"
if [[ -z "$BASE_CHECKPOINT" ]]; then
  for candidate in \
    /whitehole/AOKI/vggt-omega/ckpt/vggt_omega_1b_512.pt \
    /whitehole/AOKI/RethinkPanoVGGT_multipano/ckpt/vggt_omega_1b_512.pt \
    /whitehole/AOKI/RethinkPanoVGGT_omega/ckpt/vggt_omega_1b_512.pt; do
    if [[ -s "$candidate" ]]; then
      BASE_CHECKPOINT="$candidate"
      break
    fi
  done
fi

LOSS_PLOT="$LUNA_OUT/loss_curve_luna_9h_smoothed_robust.png"
echo "[posttrain] generating smoothed loss plot: $LOSS_PLOT"
"$RESOLVED_PYTHON" scripts/plot_loss_csv.py \
  --run "luna=$LUNA_OUT/loss.csv" \
  --metric auto \
  --x elapsed_hours \
  --smooth-method rolling_median \
  --rolling-window 401 \
  --resample-seconds 10 \
  --clip-quantile 0.98 \
  --raw-alpha 0.10 \
  --out "$LOSS_PLOT" \
  --title "Mixed4 progressive 2-to-8 pano, 8-window LUNA 9h"
if [[ ! -s "$LOSS_PLOT" ]]; then
  echo "[posttrain] loss plot was not created: $LOSS_PLOT" >&2
  exit 1
fi

if [[ "$POSTTRAIN_RUN_VALIDATION" != "1" ]]; then
  echo "[posttrain] validation skipped because POSTTRAIN_RUN_VALIDATION=$POSTTRAIN_RUN_VALIDATION"
  exit 0
fi
if [[ ! -s "$LUNA_OUT/last.pt" ]]; then
  echo "[posttrain] checkpoint not found: $LUNA_OUT/last.pt" >&2
  exit 2
fi
if [[ -z "$BASE_CHECKPOINT" || ! -s "$BASE_CHECKPOINT" ]]; then
  echo "[posttrain] Omega foundation checkpoint not found; set BASE_CHECKPOINT=/absolute/path/vggt_omega_1b_512.pt" >&2
  exit 2
fi

echo "[posttrain] launching automatic mixed4 anchor evaluation"
env \
  PYTHON="$RESOLVED_PYTHON" \
  GPUS="${VALIDATION_GPUS:-0,1,2,3}" \
  CONFIG="$CONFIG" \
  DATASET_ROOT="$DATASET_ROOT" \
  LUNA_OUT="$LUNA_OUT" \
  CHECKPOINT="$LUNA_OUT/last.pt" \
  TRAIN_LOSS_CSV="$LUNA_OUT/loss.csv" \
  EVAL_OUT="${VALIDATION_OUT:-$LUNA_OUT/eval_mixed4_anchor_traincaps_8w384_4gpu}" \
  EVAL_DATASETS=all \
  LIMIT_PER_DATASET="${VALIDATION_LIMIT_PER_DATASET:-0}" \
  SAMPLE_POLICY=anchor \
  PANO_COUNT_POLICY=config \
  DATASET_PANO_COUNTS="panocity:8,matterport3d:3,stanford2d3ds:3,structured3d:3" \
  CAMERA_EVAL_MAX_PANOS=8 \
  WINDOW_SIZE=384 \
  NUM_YAW=4 \
  PRINT_EACH_SAMPLE="${PRINT_EACH_SAMPLE:-1}" \
  BASE_CHECKPOINT_OVERRIDE="$BASE_CHECKPOINT" \
  bash scripts/run_multipano_eval_after_training.sh
