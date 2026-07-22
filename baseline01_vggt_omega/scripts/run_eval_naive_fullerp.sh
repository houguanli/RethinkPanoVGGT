#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${RUN_OUT:-}" ]]; then
  marker="$({
    find "$ROOT/logs" -mindepth 2 -maxdepth 2 -name last_successful_checkpoint.txt -printf '%T@ %h\n' 2>/dev/null || true
  } | sort -nr | while read -r _ candidate; do
    if grep -q 'representation=full_erp_no_window_split' "$candidate/run_manifest.log" 2>/dev/null; then
      printf '%s\n' "$candidate"
      break
    fi
  done)"
  RUN_OUT="${marker:-}"
fi
if [[ -z "$RUN_OUT" || ! -d "$RUN_OUT" ]]; then
  echo "[naive-fullerp-eval] no completed naive full-ERP run found under $ROOT/logs" >&2
  echo "[naive-fullerp-eval] set RUN_OUT only when evaluating a non-standard output directory" >&2
  exit 2
fi

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -x /home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python ]]; then
  PYTHON_BIN=/home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python
else
  PYTHON_BIN=python
fi
CONFIG="${CONFIG:-mixed4_naive_fullerp_vggtomega_scalealigned_2pano}"
if [[ -z "${DATASET_ROOT:-}" ]]; then
  DATASET_ROOT="${PANOVGGT_ROOT:-}"
fi
if [[ -z "$DATASET_ROOT" ]]; then
  for candidate in /whitehole/AOKI/panovggt /mnt/e/PanoVGGT_minimal_datasets/datasets; do
    if [[ -d "$candidate" ]]; then
      DATASET_ROOT="$candidate"
      break
    fi
  done
fi
if [[ -z "$DATASET_ROOT" || ! -d "$DATASET_ROOT" ]]; then
  echo "[naive-fullerp-eval] mixed4 dataset root not found" >&2
  exit 2
fi
CHECKPOINT="${CHECKPOINT:-}"
if [[ -z "$CHECKPOINT" && -s "$RUN_OUT/last_successful_checkpoint.txt" ]]; then
  CHECKPOINT="$(<"$RUN_OUT/last_successful_checkpoint.txt")"
fi
if [[ ! -s "$CHECKPOINT" ]]; then
  echo "[naive-fullerp-eval] checkpoint not found: ${CHECKPOINT:-unset}" >&2
  exit 2
fi

OUT="${EVAL_OUT:-$RUN_OUT/eval_naive_fullerp_panovggt_counts_384}"
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
  --pano-count-policy panovggt
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

if [[ -z "${VGGT_OMEGA_CKPT:-}" ]]; then
  for candidate in \
    /whitehole/AOKI/vggt-omega/ckpt/vggt_omega_1b_512.pt \
    /whitehole/AOKI/RethinkPanoVGGT_omega/ckpt/vggt_omega_1b_512.pt \
    /home/aoki/RethinkPanoVGGT_omega_compare_methods_only/ckpt/VGGT-Omega/vggt_omega_1b_512.pt \
    "$ROOT/ckpt/vggt_omega_1b_512.pt"; do
    if [[ -s "$candidate" ]]; then
      VGGT_OMEGA_CKPT="$candidate"
      break
    fi
  done
fi
if [[ -z "${VGGT_OMEGA_CKPT:-}" || ! -s "$VGGT_OMEGA_CKPT" ]]; then
  echo "[naive-fullerp-eval] base VGGT-Omega checkpoint not found" >&2
  exit 2
fi
export VGGT_OMEGA_CKPT
echo "[naive-fullerp-eval] checkpoint=$CHECKPOINT"
echo "[naive-fullerp-eval] dataset_root=$DATASET_ROOT"
echo "[naive-fullerp-eval] output=$OUT"
echo "[naive-fullerp-eval] representation=full_erp_no_window_split img=${EVAL_IMG_SIZE}x$((EVAL_IMG_SIZE / 2))"
echo "[naive-fullerp-eval] pano_counts=panocity:10,matterport3d:3,stanford2d3ds:3,structured3d:3"
if [[ "$FOREGROUND" == "1" ]]; then
  "${cmd[@]}" 2>&1 | tee "$OUT/eval_console.log"
else
  nohup "${cmd[@]}" >"$OUT/eval_stdout.log" 2>"$OUT/eval_stderr.log" &
  echo "$!" > "$OUT/eval.pid"
  echo "[naive-fullerp-eval] pid=$(<"$OUT/eval.pid")"
fi
