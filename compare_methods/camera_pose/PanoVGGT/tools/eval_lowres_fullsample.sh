#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

DATASETS_ROOT="${1:-${PANOVGGT_DATASETS_ROOT:-/mnt/e/PanoVGGT_minimal_datasets/datasets}}"
CKPT="${2:-${PANOVGGT_LOWRES_CKPT:-checkpoints/model_lowres.pt}}"
CASE="${3:-mono}"
SPLIT="${PANOVGGT_EVAL_SPLIT:-test_final}"
JSON_ROOT="${PANOVGGT_EVAL_JSON_ROOT:-outputs/eval_lowres_${SPLIT}_${CASE}_fullsample}"

case "${CASE}" in
  mono|single)
    FRAMES_PANOCITY="${PANOVGGT_EVAL_FRAMES_PANOCITY:-1}"
    FRAMES_INDOOR="${PANOVGGT_EVAL_FRAMES_INDOOR:-1}"
    ;;
  multi)
    FRAMES_PANOCITY="${PANOVGGT_EVAL_FRAMES_PANOCITY:-10}"
    FRAMES_INDOOR="${PANOVGGT_EVAL_FRAMES_INDOOR:-3}"
    ;;
  *)
    echo "Usage: $0 [DATASETS_ROOT] [LOWRES_CKPT] [mono|multi]" >&2
    exit 2
    ;;
esac

python -m evaluation.eval_allpano \
  --datasets_root "${DATASETS_ROOT}" \
  --ckpt "${CKPT}" \
  --split "${SPLIT}" \
  --frames_panocity "${FRAMES_PANOCITY}" \
  --frames_matterport "${FRAMES_INDOOR}" \
  --frames_stanford "${FRAMES_INDOOR}" \
  --frames_structured3d "${FRAMES_INDOOR}" \
  --eval_unit sample \
  --sample_stride 1 \
  --depth_align irls-absrel \
  --depth_lat_min -15 \
  --depth_lat_max 60 \
  --no_pointcloud \
  --json_root "${JSON_ROOT}"
