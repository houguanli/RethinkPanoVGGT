#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/aoki/RethinkPanoVGGT_omega_single_pano_reconstruction/Rethink_pano_new_exp_omega"
PYTHON="/home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python"
CONFIG="configs/single_pano_4090_panocity_paired_100_luna_residual_dense_12h.yaml"
OUT="logs/panocity_paired_100_luna_residual_dense_final_4090_12h"

cd "$ROOT"
mkdir -p "$OUT"

{
  echo "[run] started $(date --iso-8601=seconds)"
  echo "[run] config=$CONFIG"
  echo "[run] python=$PYTHON"
  echo "[run] output=$OUT"
} | tee -a "$OUT/run_12h_sequence.log"

set +e
CUDA_VISIBLE_DEVICES=0 "$PYTHON" training/train_pano_omega.py --config "$CONFIG" \
  2>&1 | tee -a "$OUT/train_12h.log"
status=${PIPESTATUS[0]}
set -e

echo "[run] finished $(date --iso-8601=seconds) status=$status" | tee -a "$OUT/run_12h_sequence.log"
exit "$status"
