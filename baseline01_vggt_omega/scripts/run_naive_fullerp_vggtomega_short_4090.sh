#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RESOLUTIONS="${RESOLUTIONS:-384,512,1024,2048}"
export STAGE_DURATION_MINUTES="${STAGE_DURATION_MINUTES:-4}"
export BUILD_INDEXES="${BUILD_INDEXES:-0}"
export NAIVE_FULLERP_CONFIG="${NAIVE_FULLERP_CONFIG:-mixed4_naive_fullerp_vggtomega_scalealigned_2pano}"

exec bash "$SCRIPT_DIR/run_naive_fullerp_vggtomega_progressive.sh"
