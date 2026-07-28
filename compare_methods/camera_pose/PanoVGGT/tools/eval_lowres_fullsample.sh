#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

DATASETS_ROOT="${1:-${PANOVGGT_DATASETS_ROOT:-/mnt/e/PanoVGGT_minimal_datasets/datasets}}"
CKPT="${2:-${PANOVGGT_LOWRES_CKPT:-checkpoints/model_lowres.pt}}"
CASE="${3:-mono}"
SPLIT="${PANOVGGT_EVAL_SPLIT:-val}"
JSON_ROOT="${PANOVGGT_EVAL_JSON_ROOT:-outputs/eval_lowres_${SPLIT}_${CASE}_fullsample}"

case "${CASE}" in
  mono|single)
    FRAMES=1
    ;;
  multi)
    FRAMES=3
    ;;
  *)
    echo "Usage: $0 [DATASETS_ROOT] [LOWRES_CKPT] [mono|multi]" >&2
    exit 2
    ;;
esac

python -m evaluation.eval_allpano \
  --datasets_root "${DATASETS_ROOT}" \
  --panocity_root "${DATASETS_ROOT}/PanoCity" \
  --matterport_root "${DATASETS_ROOT}/Matterport3D" \
  --stanford_root "${DATASETS_ROOT}/Stanford2D3DS" \
  --structured3d_root "${DATASETS_ROOT}/Structured3D" \
  --ckpt "${CKPT}" \
  --split "${SPLIT}" \
  --frames_panocity "${FRAMES}" \
  --frames_matterport "${FRAMES}" \
  --frames_stanford "${FRAMES}" \
  --frames_structured3d "${FRAMES}" \
  --eval_unit sample \
  --sample_stride 1 \
  --depth_align irls-absrel \
  --depth_lat_min -15 \
  --depth_lat_max 60 \
  --no_pointcloud \
  --json_root "${JSON_ROOT}"
