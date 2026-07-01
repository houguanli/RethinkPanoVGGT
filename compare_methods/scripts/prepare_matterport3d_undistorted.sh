#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PANO_DIR="${REPO_ROOT}/compare_methods/camera_pose/PanoVGGT"

PYTHON_BIN="${PYTHON:-python}"
RAW_ROOT="${MATTERPORT3D_RAW_ROOT:-/mnt/e/PanoVGGT_minimal_datasets/datasets/Matterport3D_raw/v1/scans}"
OUT_ROOT="${MATTERPORT3D_OUT_ROOT:-/mnt/e/PanoVGGT_minimal_datasets/datasets/Matterport3D}"
SPLIT_DIR="${MATTERPORT3D_SPLIT_DIR:-${PANO_DIR}/training/data/splits/matterport3d}"
SPLITS_TEXT="${MATTERPORT3D_SPLITS:-val test}"
HOLE_FILL="${MATTERPORT3D_HOLE_FILL:-small}"
SMALL_HOLE_AREA="${MATTERPORT3D_SMALL_HOLE_AREA:-18000}"
DEPTH_MAX_M="${MATTERPORT3D_DEPTH_MAX_M:-10}"
OUT_H="${MATTERPORT3D_OUT_H:-1024}"
OUT_W="${MATTERPORT3D_OUT_W:-2048}"

read -r -a SPLITS <<< "${SPLITS_TEXT}"

echo "[matterport3d] raw_root=${RAW_ROOT}"
echo "[matterport3d] out_root=${OUT_ROOT}"
echo "[matterport3d] splits=${SPLITS[*]}"
echo "[matterport3d] depth=undistorted hole_fill=${HOLE_FILL} depth_max_m=${DEPTH_MAX_M}"

exec "${PYTHON_BIN}" "${PANO_DIR}/tools/prepare_matterport3d_raw.py" \
  --raw-root "${RAW_ROOT}" \
  --out-root "${OUT_ROOT}" \
  --split-dir "${SPLIT_DIR}" \
  --splits "${SPLITS[@]}" \
  --out-h "${OUT_H}" \
  --out-w "${OUT_W}" \
  --depth-source undistorted \
  --hole-fill "${HOLE_FILL}" \
  --small-hole-area "${SMALL_HOLE_AREA}" \
  --depth-max-m "${DEPTH_MAX_M}" \
  "$@"
