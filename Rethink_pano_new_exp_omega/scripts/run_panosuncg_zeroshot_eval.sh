#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/whitehole/AOKI/panovggt}"
CONFIG="${CONFIG:-configs/multipano_rtx5000x4_mixed4_pano_low_to_high_luna_after_full_warmup_9h.yaml}"
RUN_OR_CKPT="${1:-${TRAIN_OUT:-}}"

if [[ -n "$RUN_OR_CKPT" && -d "$RUN_OR_CKPT" ]]; then
  CHECKPOINT="$RUN_OR_CKPT/last.pt"
elif [[ -n "$RUN_OR_CKPT" ]]; then
  CHECKPOINT="$RUN_OR_CKPT"
else
  CHECKPOINT="$(find logs -mindepth 2 -maxdepth 3 -type f -name last.pt -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
fi

if [[ -z "${CHECKPOINT:-}" || ! -s "$CHECKPOINT" ]]; then
  echo "[panosuncg-eval] no nonempty last.pt found; pass a training output directory or checkpoint" >&2
  echo "usage: bash scripts/run_panosuncg_zeroshot_eval.sh [TRAIN_OUT_OR_LAST_PT]" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "[panosuncg-eval] config not found: $CONFIG" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[panosuncg-eval] python not executable: $PYTHON_BIN" >&2
  exit 2
fi

CHECKPOINT="$(realpath "$CHECKPOINT")"
RUN_DIR="$(dirname "$CHECKPOINT")"
if [[ "$(basename "$RUN_DIR")" == "ckpts" ]]; then
  RUN_DIR="$(dirname "$RUN_DIR")"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/eval_panosuncg_zeroshot_da2}"

echo "[panosuncg-eval] checkpoint=$CHECKPOINT"
echo "[panosuncg-eval] config=$CONFIG"
echo "[panosuncg-eval] dataset_root=$DATASET_ROOT"
echo "[panosuncg-eval] output_dir=$OUTPUT_DIR"
echo "[panosuncg-eval] protocol=DA2 official split + full ERP + per-image median scale"

exec "$PYTHON_BIN" scripts/evaluate_panosuncg_zeroshot.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --device "${DEVICE:-cuda}" \
  --amp-dtype "${AMP_DTYPE:-bfloat16}" \
  --window-size "${WINDOW_SIZE:-384}" \
  --num-yaw "${NUM_YAW:-6}" \
  --pitch-degrees="${PITCH_DEGREES:--55,-15,55}" \
  --fov-degrees "${FOV_DEGREES:-75}" \
  --limit "${LIMIT:-0}" \
  --progress-every "${PROGRESS_EVERY:-10}" \
  --resume
