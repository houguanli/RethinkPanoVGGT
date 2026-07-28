#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WARMUP_CHECKPOINT="${1:-${WARMUP_CHECKPOINT:-}}"
if [[ -z "$WARMUP_CHECKPOINT" || ! -s "$WARMUP_CHECKPOINT" ]]; then
  echo "usage: bash scripts/run_server_luna_9h_mixed4_progressive_8p8w384.sh /absolute/path/to/warmup/last.pt" >&2
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

CONFIG="configs/multipano_rtx5000x4_mixed4_pano_all384_luna_after_full_warmup_9h.yaml"
DATASET_ROOT="${DATASET_ROOT:-/whitehole/AOKI/panovggt}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LUNA_OUT="${LUNA_OUT:-logs/server_mixed4_luna_9h_progressive_2to8p_8w384_fov90_pitch30_${RUN_TAG}}"
LUNA_RUN_VALIDATION="${LUNA_RUN_VALIDATION:-1}"
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
if [[ -z "$BASE_CHECKPOINT" || ! -s "$BASE_CHECKPOINT" ]]; then
  echo "[luna-9h] Omega foundation checkpoint not found; set BASE_CHECKPOINT=/absolute/path/vggt_omega_1b_512.pt" >&2
  exit 2
fi

mkdir -p "$LUNA_OUT"
LOG="$LUNA_OUT/train_9h.log"
{
  echo "[luna-9h] started $(date --iso-8601=seconds)"
  echo "[luna-9h] config=$CONFIG"
  echo "[luna-9h] warmup_checkpoint=$WARMUP_CHECKPOINT"
  echo "[luna-9h] base_checkpoint=$BASE_CHECKPOINT"
  echo "[luna-9h] dataset_root=$DATASET_ROOT"
  echo "[luna-9h] output=$LUNA_OUT"
  echo "[luna-9h] curriculum=0-1h:2p,1-3h:4p,3-6h:6p,6-9h:8p"
  echo "[luna-9h] dataset_caps=panocity:8,matterport3d:3,stanford2d3ds:3,structured3d:3"
  echo "[luna-9h] windows=4 yaw x pitch(-30,+30), 384x384, FoV=90deg"
  echo "[luna-9h] run_validation=$LUNA_RUN_VALIDATION"
} | tee "$LOG"

PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$RESOLVED_PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE:-4}" \
  training/train_pano_omega.py \
  --config "$CONFIG" \
  --dataset-root "$DATASET_ROOT" \
  --checkpoint "$WARMUP_CHECKPOINT" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --output-dir "$LUNA_OUT" \
  --tensorboard-dir "$LUNA_OUT/tensorboard" \
  --debug-dir "$LUNA_OUT/debug" \
  --max-duration-minutes "${LUNA_DURATION_MINUTES:-540}" \
  --dataset-pano-max-counts "panocity:8,matterport3d:3,stanford2d3ds:3,structured3d:3" \
  --window-size 384 \
  --num-yaw 4 \
  --pitch-degrees=-30,30 \
  --fov-degrees 90 \
  --pano-height 512 \
  --pano-width 1024 \
  --no-inherit-checkpoint-training-defaults \
  2>&1 | tee -a "$LOG"

env \
  PYTHON="$RESOLVED_PYTHON" \
  DATASET_ROOT="$DATASET_ROOT" \
  BASE_CHECKPOINT="$BASE_CHECKPOINT" \
  POSTTRAIN_RUN_VALIDATION="$LUNA_RUN_VALIDATION" \
  VALIDATION_GPUS="${VALIDATION_GPUS:-0,1,2,3}" \
  VALIDATION_LIMIT_PER_DATASET="${VALIDATION_LIMIT_PER_DATASET:-0}" \
  PRINT_EACH_SAMPLE="${PRINT_EACH_SAMPLE:-1}" \
  bash scripts/run_server_posttrain_mixed4_progressive_8p8w384.sh "$LUNA_OUT" \
  2>&1 | tee -a "$LOG"
