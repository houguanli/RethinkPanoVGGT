#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUNA="$(cd "$SCRIPT_DIR/.." && pwd)"

# The server mirrors /home/aoki below this storage root.  Override any path
# explicitly when a deployment differs from that convention.
export AOKI_STORAGE_ROOT="${AOKI_STORAGE_ROOT:-/whitehole/AOKI}"

first_file() {
  local candidate
  for candidate in "$@"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

first_dir() {
  local candidate
  for candidate in "$@"; do
    if [[ -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

if [[ -z "${PYTHON:-}" ]]; then
  PYTHON="$(first_file \
    "$AOKI_STORAGE_ROOT/miniconda3/envs/RethinkPanoVGGT_omega/bin/python" \
    /home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python \
    "$(command -v python || true)" || true)"
fi

if [[ -z "${DATASET_ROOT:-}" ]]; then
  DATASET_ROOT="$(first_dir \
    "$AOKI_STORAGE_ROOT/PanoVGGT_minimal_datasets/datasets" \
    /mnt/e/PanoVGGT_minimal_datasets/datasets \
    "$AOKI_STORAGE_ROOT/datasets" || true)"
fi

RUN_ROOT_REL="RethinkPanoVGGT_omega_multipano_work/Rethink_pano_new_exp_omega"
LOCAL_RUN_ROOT="$LUNA/logs"
SERVER_RUN_ROOT="$AOKI_STORAGE_ROOT/$RUN_ROOT_REL/logs"
LUNA_RUN_NAME="local_full_warmup_no_depthres_20260716_002_luna_9h"
WARMUP_RUN_NAME="local_full_warmup_no_depthres_20260716_002_warmup_3h"

if [[ -z "${CHECKPOINT:-}" ]]; then
  CHECKPOINT="$(first_file \
    "$LOCAL_RUN_ROOT/$LUNA_RUN_NAME/last.pt" \
    "$SERVER_RUN_ROOT/$LUNA_RUN_NAME/last.pt" || true)"
fi
if [[ -z "${WARMUP_CHECKPOINT:-}" ]]; then
  WARMUP_CHECKPOINT="$(first_file \
    "$LOCAL_RUN_ROOT/$WARMUP_RUN_NAME/last.pt" \
    "$SERVER_RUN_ROOT/$WARMUP_RUN_NAME/last.pt" || true)"
fi
if [[ -z "${FOUNDATION_CHECKPOINT:-}" ]]; then
  FOUNDATION_CHECKPOINT="$(first_file \
    "$AOKI_STORAGE_ROOT/RethinkPanoVGGT_omega/ckpt/vggt_omega_1b_512.pt" \
    "$AOKI_STORAGE_ROOT/vggt-omega/ckpt/vggt_omega_1b_512.pt" \
    /home/aoki/RethinkPanoVGGT_omega/ckpt/vggt_omega_1b_512.pt \
    /home/aoki/RethinkPanoVGGT_omega_compare_methods_only/ckpt/VGGT-Omega/vggt_omega_1b_512.pt || true)"
fi

CONFIG="${CONFIG:-configs/multipano_rtx5000x4_mixed4_pano_all384_luna_after_full_warmup_9h.yaml}"
TRAIN_LOSS_CSV="${TRAIN_LOSS_CSV:-$(dirname "${CHECKPOINT:-/missing}")/loss.csv}"
EVAL_OUT="${EVAL_OUT:-$(dirname "${CHECKPOINT:-$LUNA/logs/$LUNA_RUN_NAME/last.pt}")/eval_server_full_pipeline_4gpu}"
GPUS="${GPUS:-0,1,2,3}"
LIMIT_PER_DATASET="${LIMIT_PER_DATASET:-0}"
EVAL_DATASETS="${EVAL_DATASETS:-all}"
NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-2}"
SAMPLE_POLICY="${SAMPLE_POLICY:-anchor}"
PANO_COUNT_POLICY="${PANO_COUNT_POLICY:-panovggt}"
CAMERA_EVAL_MAX_PANOS="${CAMERA_EVAL_MAX_PANOS:-3}"
WINDOW_SIZE="${WINDOW_SIZE:-384}"
NUM_YAW="${NUM_YAW:-4}"
CHECK_ONLY="${CHECK_ONLY:-0}"

failures=0
require_file() {
  local label="$1" path="$2"
  if [[ -n "$path" && -s "$path" ]]; then
    echo "[server-0716][ok] $label=$path"
  else
    echo "[server-0716][missing] $label=${path:-<unset>}" >&2
    failures=$((failures + 1))
  fi
}

require_dir() {
  local label="$1" path="$2"
  if [[ -n "$path" && -d "$path" ]]; then
    echo "[server-0716][ok] $label=$path"
  else
    echo "[server-0716][missing] $label=${path:-<unset>}" >&2
    failures=$((failures + 1))
  fi
}

echo "[server-0716] preflight $(date --iso-8601=seconds)"
echo "[server-0716] luna=$LUNA"
echo "[server-0716] storage_root=$AOKI_STORAGE_ROOT"
echo "[server-0716] gpus=$GPUS"
require_file python "$PYTHON"
require_dir dataset_root "$DATASET_ROOT"
for dataset in Panocity Matterport3D Stanford2D3DS Structured3D; do
  require_dir "dataset:$dataset" "$DATASET_ROOT/$dataset"
done
require_file config "$LUNA/$CONFIG"
require_file luna_delta "$CHECKPOINT"
require_file warmup_delta "$WARMUP_CHECKPOINT"
require_file foundation "$FOUNDATION_CHECKPOINT"
require_file train_loss_csv "$TRAIN_LOSS_CSV"
if [[ "$failures" -ne 0 ]]; then
  echo "[server-0716] preflight failed with $failures missing path(s)" >&2
  exit 2
fi

"$PYTHON" - "$CHECKPOINT" "$WARMUP_CHECKPOINT" "$FOUNDATION_CHECKPOINT" <<'PY'
import sys
from pathlib import Path
import torch

luna_path, warmup_path, foundation_path = map(Path, sys.argv[1:4])
for label, path, expected_format in (
    ("luna", luna_path, "trainable_delta"),
    ("warmup", warmup_path, "trainable_delta"),
):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    actual = payload.get("checkpoint_format") if isinstance(payload, dict) else None
    if actual != expected_format or "model_delta" not in payload:
        raise RuntimeError(f"{label} checkpoint is not a trainable delta: {path}")
    print(
        f"[server-0716][ok] {label}_checkpoint format={actual} "
        f"step={payload.get('step')} delta_keys={len(payload['model_delta'])}"
    )
print(f"[server-0716][ok] foundation_checkpoint bytes={foundation_path.stat().st_size}")
PY

if [[ "$CHECK_ONLY" == "1" || "${1:-}" == "--check-only" ]]; then
  echo "[server-0716] preflight complete; evaluation not started"
  exit 0
fi

mkdir -p "$EVAL_OUT"
{
  echo "[server-0716] launching full pipeline eval $(date --iso-8601=seconds)"
  echo "[server-0716] checkpoint=$CHECKPOINT"
  echo "[server-0716] warmup_checkpoint=$WARMUP_CHECKPOINT"
  echo "[server-0716] foundation_checkpoint=$FOUNDATION_CHECKPOINT"
  echo "[server-0716] dataset_root=$DATASET_ROOT"
  echo "[server-0716] eval_out=$EVAL_OUT"
  echo "[server-0716] protocol=anchor panovggt_counts window=${WINDOW_SIZE} num_yaw=${NUM_YAW}"
} | tee "$EVAL_OUT/server_0716_launcher.log"

env \
  AOKI_STORAGE_ROOT="$AOKI_STORAGE_ROOT" \
  PYTHON="$PYTHON" \
  GPUS="$GPUS" \
  CONFIG="$CONFIG" \
  DATASET_ROOT="$DATASET_ROOT" \
  LUNA_OUT="$(dirname "$CHECKPOINT")" \
  CHECKPOINT="$CHECKPOINT" \
  TRAIN_LOSS_CSV="$TRAIN_LOSS_CSV" \
  EVAL_OUT="$EVAL_OUT" \
  LIMIT_PER_DATASET="$LIMIT_PER_DATASET" \
  EVAL_DATASETS="$EVAL_DATASETS" \
  NUM_WORKERS_PER_GPU="$NUM_WORKERS_PER_GPU" \
  SAMPLE_POLICY="$SAMPLE_POLICY" \
  PANO_COUNT_POLICY="$PANO_COUNT_POLICY" \
  CAMERA_EVAL_MAX_PANOS="$CAMERA_EVAL_MAX_PANOS" \
  WINDOW_SIZE="$WINDOW_SIZE" \
  NUM_YAW="$NUM_YAW" \
  PRINT_EACH_SAMPLE="${PRINT_EACH_SAMPLE:-1}" \
  ERP_LATITUDE_LIMIT_DEG="${ERP_LATITUDE_LIMIT_DEG:-75}" \
  bash "$LUNA/scripts/run_multipano_mixed4_eval_4gpu.sh" \
  2>&1 | tee -a "$EVAL_OUT/server_0716_launcher.log"

SUMMARY="$EVAL_OUT/validation_mixed4_by_dataset_valtestfull_summary.json"
if [[ ! -s "$SUMMARY" ]]; then
  echo "[server-0716] evaluation ended without summary: $SUMMARY" >&2
  exit 1
fi
echo "[server-0716] complete summary=$SUMMARY"
