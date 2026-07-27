#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python}"
CONFIG="${CONFIG:-configs/multipano_rtx5000x4_panocity_luna_memory_probe_10p_8w_fov120_erp1024x512.yaml}"
DATASET_ROOT="${DATASET_ROOT:-/whitehole/AOKI/panovggt}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
WARMUP_CHECKPOINT="${1:-${WARMUP_CHECKPOINT:-}}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/server_luna_memory_probe_10p_8w_fov120_erp1024x512_${RUN_TAG}}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "$WARMUP_CHECKPOINT" || ! -s "$WARMUP_CHECKPOINT" ]]; then
  echo "usage: bash scripts/run_server_luna_memory_probe_10p8w512.sh /absolute/path/to/warmup/last.pt" >&2
  exit 2
fi
if [[ -z "$BASE_CHECKPOINT" ]]; then
  for candidate in \
    /whitehole/AOKI/vggt-omega/ckpt/vggt_omega_1b_512.pt \
    /whitehole/AOKI/RethinkPanoVGGT_omega/ckpt/vggt_omega_1b_512.pt \
    /home/aoki/RethinkPanoVGGT_omega/ckpt/vggt_omega_1b_512.pt; do
    if [[ -s "$candidate" ]]; then
      BASE_CHECKPOINT="$candidate"
      break
    fi
  done
fi
if [[ -z "$BASE_CHECKPOINT" || ! -s "$BASE_CHECKPOINT" ]]; then
  echo "[memory-probe] foundation checkpoint not found; set BASE_CHECKPOINT=/absolute/path/vggt_omega_1b_512.pt" >&2
  exit 2
fi
if [[ ! -d "$DATASET_ROOT/Panocity" ]]; then
  echo "[memory-probe] Panocity dataset not found under $DATASET_ROOT" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "[memory-probe] python not executable: $PYTHON" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/train_console.log"
{
  echo "[memory-probe] started $(date --iso-8601=seconds)"
  echo "[memory-probe] config=$CONFIG"
  echo "[memory-probe] checkpoint=$WARMUP_CHECKPOINT"
  echo "[memory-probe] base_checkpoint=$BASE_CHECKPOINT"
  echo "[memory-probe] dataset_root=$DATASET_ROOT"
  echo "[memory-probe] output_dir=$OUTPUT_DIR"
  echo "[memory-probe] nproc_per_node=$NPROC_PER_NODE"
  echo "[memory-probe] input=10 panos x ERP 1024x512"
  echo "[memory-probe] sampler=8 windows/pano x 512x512, FoV=120deg; total=80 windows/sample"
  echo "[memory-probe] duration_minutes=${DURATION_MINUTES:-10}"
} | tee "$LOG"

set +e
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  training/train_pano_omega.py \
  --config "$CONFIG" \
  --dataset-root "$DATASET_ROOT" \
  --checkpoint "$WARMUP_CHECKPOINT" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --tensorboard-dir "$OUTPUT_DIR/tensorboard" \
  --debug-dir "$OUTPUT_DIR/debug" \
  --max-duration-minutes "${DURATION_MINUTES:-10}" \
  --no-inherit-checkpoint-training-defaults \
  2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e

if [[ "$status" -ne 0 ]]; then
  echo "[memory-probe] failed with status=$status; output preserved at $OUTPUT_DIR" | tee -a "$LOG"
  exit "$status"
fi
echo "[memory-probe] completed $(date --iso-8601=seconds)" | tee -a "$LOG"
