#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${RUN_OUT:-}" ]]; then
  echo "[naive-fullerp-eval] RUN_OUT must point to the completed progressive training directory" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON:-/home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python}"
CONFIG="${CONFIG:-mixed4_naive_fullerp_vggtomega_scalealigned_2pano}"
DATASET_ROOT="${DATASET_ROOT:-${PANOVGGT_ROOT:-/mnt/e/PanoVGGT_minimal_datasets/datasets}}"
CHECKPOINT="${CHECKPOINT:-}"
if [[ -z "$CHECKPOINT" && -s "$RUN_OUT/last_successful_checkpoint.txt" ]]; then
  CHECKPOINT="$(<"$RUN_OUT/last_successful_checkpoint.txt")"
fi
if [[ ! -s "$CHECKPOINT" ]]; then
  echo "[naive-fullerp-eval] checkpoint not found: ${CHECKPOINT:-unset}" >&2
  exit 2
fi

OUT="${EVAL_OUT:-$RUN_OUT/eval_naive_fullerp_2p384}"
LIMIT_PER_DATASET="${LIMIT_PER_DATASET:-0}"
DATASETS="${DATASETS:-all}"
EVAL_IMG_SIZE="${EVAL_IMG_SIZE:-384}"
FOREGROUND="${FOREGROUND:-0}"
RESUME="${RESUME:-0}"
mkdir -p "$OUT"

cmd=(
  "$PYTHON_BIN" scripts/evaluate_mixed4_depth_checkpoint.py
  --config "$CONFIG"
  --checkpoint "$CHECKPOINT"
  --dataset-root "$DATASET_ROOT"
  --output "$OUT/validation_naive_fullerp_summary.json"
  --per-sample-csv "$OUT/per_sample.csv"
  --camera-pair-csv "$OUT/camera_pairs.csv"
  --datasets "$DATASETS"
  --limit-per-dataset "$LIMIT_PER_DATASET"
  --sample-policy anchor
  --pano-count-policy config
  --img-size "$EVAL_IMG_SIZE"
  --camera-eval-max-panos 0
  --device auto
  --amp-dtype bfloat16
  --fail-fast
)
if [[ "$RESUME" == "1" ]]; then
  cmd+=(--resume)
else
  rm -f "$OUT/per_sample.csv" "$OUT/camera_pairs.csv" "$OUT/validation_naive_fullerp_summary.json"
fi

if [[ -n "${VGGT_OMEGA_CKPT:-}" ]]; then
  export VGGT_OMEGA_CKPT
else
  unset VGGT_OMEGA_CKPT
fi
echo "[naive-fullerp-eval] checkpoint=$CHECKPOINT"
echo "[naive-fullerp-eval] output=$OUT"
echo "[naive-fullerp-eval] representation=full_erp_no_window_split img=${EVAL_IMG_SIZE}x$((EVAL_IMG_SIZE / 2)) pano_policy=config"
if [[ "$FOREGROUND" == "1" ]]; then
  "${cmd[@]}" 2>&1 | tee "$OUT/eval_console.log"
else
  nohup "${cmd[@]}" >"$OUT/eval_stdout.log" 2>"$OUT/eval_stderr.log" &
  echo "$!" > "$OUT/eval.pid"
  echo "[naive-fullerp-eval] pid=$(<"$OUT/eval.pid")"
fi
