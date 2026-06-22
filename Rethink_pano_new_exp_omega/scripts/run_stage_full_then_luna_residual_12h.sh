#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/aoki/RethinkPanoVGGT_omega_single_pano_reconstruction"
BASELINE="$ROOT/baseline01_vggt_omega"
LUNA="$ROOT/Rethink_pano_new_exp_omega"
PYTHON="/home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python"

BASE_CONFIG="panocity_paired_100_full_warmup_4090_3h_for_luna"
BASE_OUT="$BASELINE/logs/panocity_paired_100_full_warmup_4090_3h_for_luna"
BASE_CKPT="$BASE_OUT/ckpts/checkpoint.pt"

LUNA_CONFIG="configs/single_pano_4090_panocity_paired_100_luna_after_full_warmup_9h.yaml"
LUNA_OUT="$LUNA/logs/panocity_paired_100_luna_after_full_warmup_4090_9h"
SEQ_LOG="$LUNA/logs/panocity_paired_100_stage_full_then_luna_residual_4090_12h_sequence.log"

mkdir -p "$(dirname "$SEQ_LOG")" "$BASE_OUT" "$LUNA_OUT"

{
  echo "[sequence] started $(date --iso-8601=seconds)"
  echo "[sequence] baseline_config=$BASE_CONFIG"
  echo "[sequence] luna_config=$LUNA_CONFIG"
  echo "[sequence] python=$PYTHON"
} | tee -a "$SEQ_LOG"

cd "$BASELINE"
echo "[sequence] stage1 baseline full warmup started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29621 CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$BASELINE${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" training/launch.py --config "$BASE_CONFIG" \
  2>&1 | tee -a "$BASE_OUT/train_3h_console.log"
echo "[sequence] stage1 baseline full warmup finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"

if [[ ! -s "$BASE_CKPT" ]]; then
  echo "[sequence] missing baseline checkpoint: $BASE_CKPT" | tee -a "$SEQ_LOG"
  exit 1
fi
echo "[sequence] baseline checkpoint ready: $BASE_CKPT" | tee -a "$SEQ_LOG"

cd "$LUNA"
echo "[sequence] stage2/3 LUNA residual continuation started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29622 CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$LUNA${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" training/train_pano_omega.py --config "$LUNA_CONFIG" \
  2>&1 | tee -a "$LUNA_OUT/train_9h.log"
echo "[sequence] stage2/3 LUNA residual continuation finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
echo "[sequence] finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
