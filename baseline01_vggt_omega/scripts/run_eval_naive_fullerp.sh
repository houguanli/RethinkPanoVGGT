#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

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
CHECKPOINT="${1:-${CHECKPOINT:-}}"
if [[ ! -s "$CHECKPOINT" ]]; then
  echo "usage: bash scripts/run_eval_naive_fullerp.sh /absolute/path/to/checkpoint.pt" >&2
  echo "[naive-fullerp-eval] checkpoint not found: ${CHECKPOINT:-unset}" >&2
  exit 2
fi
CHECKPOINT="$(readlink -f "$CHECKPOINT")"
STAGE_OUT="$(dirname "$(dirname "$CHECKPOINT")")"
RUN_OUT="${RUN_OUT:-$STAGE_OUT}"

LIMIT_PER_DATASET="${LIMIT_PER_DATASET:-0}"
DATASETS="${DATASETS:-all}"
EVAL_IMG_SIZE="${EVAL_IMG_SIZE:-384}"
FOREGROUND="${FOREGROUND:-0}"
RESUME="${RESUME:-0}"
PANO_COUNT_POLICY="${PANO_COUNT_POLICY:-auto}"
DATASET_PANO_COUNTS_OVERRIDE="${DATASET_PANO_COUNTS_OVERRIDE:-}"
PANO_PROTOCOL_LABEL=""
if [[ "$PANO_COUNT_POLICY" == "auto" ]]; then
  if [[ "$CHECKPOINT" =~ _10p/ ]]; then
    PANO_COUNT_POLICY=panovggt
    PANO_PROTOCOL_LABEL=panovggt_counts
  elif [[ "$CHECKPOINT" =~ _6p/ ]]; then
    PANO_COUNT_POLICY=config
    DATASET_PANO_COUNTS_OVERRIDE=panocity:6,matterport3d:3,stanford2d3ds:3,structured3d:3
    PANO_PROTOCOL_LABEL=6p3p_counts
  else
    PANO_COUNT_POLICY=config
    PANO_PROTOCOL_LABEL=legacy_2p_counts
  fi
fi
if [[ "$PANO_COUNT_POLICY" != "config" && "$PANO_COUNT_POLICY" != "panovggt" ]]; then
  echo "[naive-fullerp-eval] invalid PANO_COUNT_POLICY=$PANO_COUNT_POLICY" >&2
  exit 2
fi
PANO_PROTOCOL_LABEL="${PANO_PROTOCOL_LABEL:-$PANO_COUNT_POLICY}"
OUT="${EVAL_OUT:-$RUN_OUT/eval_naive_fullerp_${PANO_PROTOCOL_LABEL}_384}"
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
  --pano-count-policy "$PANO_COUNT_POLICY"
  --img-size "$EVAL_IMG_SIZE"
  --camera-eval-max-panos 0
  --device auto
  --amp-dtype bfloat16
  --fail-fast
)
if [[ -n "$DATASET_PANO_COUNTS_OVERRIDE" ]]; then
  cmd+=(--dataset-pano-counts "$DATASET_PANO_COUNTS_OVERRIDE")
fi
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
if [[ -n "$DATASET_PANO_COUNTS_OVERRIDE" ]]; then
  echo "[naive-fullerp-eval] pano_counts=$DATASET_PANO_COUNTS_OVERRIDE"
elif [[ "$PANO_COUNT_POLICY" == "panovggt" ]]; then
  echo "[naive-fullerp-eval] pano_counts=panocity:10,matterport3d:3,stanford2d3ds:3,structured3d:3"
else
  echo "[naive-fullerp-eval] pano_counts=2/2/2/2 (matched to the legacy 2-pano training checkpoint)"
fi
if [[ "$FOREGROUND" == "1" ]]; then
  "${cmd[@]}" 2>&1 | tee "$OUT/eval_console.log"
else
  nohup "${cmd[@]}" >"$OUT/eval_stdout.log" 2>"$OUT/eval_stderr.log" &
  echo "$!" > "$OUT/eval.pid"
  echo "[naive-fullerp-eval] pid=$(<"$OUT/eval.pid")"
fi
