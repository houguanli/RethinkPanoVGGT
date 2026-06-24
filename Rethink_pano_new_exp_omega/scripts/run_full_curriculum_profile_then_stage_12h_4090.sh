#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/aoki/RethinkPanoVGGT_omega_single_pano_reconstruction"
BASELINE="$ROOT/baseline01_vggt_omega"
LUNA="$ROOT/Rethink_pano_new_exp_omega"
PYTHON="${PYTHON:-/home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/f/panovggt/PanoCity_paired}"
PROFILE_WORKERS="${PROFILE_WORKERS:-12}"
FORCE_PROFILE="${FORCE_PROFILE:-1}"

MANIFEST="$DATA_ROOT/metadata/panocity_manifest.jsonl"
SUMMARY="$DATA_ROOT/metadata/panocity_manifest_summary.json"
PROFILE_LOG="$DATA_ROOT/metadata/profile_panocity_manifest_full.log"

BASE_CONFIG="panocity_paired_full_curriculum_full_warmup_4090_3h_for_luna"
BASE_OUT="$BASELINE/logs/panocity_paired_full_curriculum_full_warmup_4090_3h_for_luna"
BASE_CKPT="$BASE_OUT/ckpts/checkpoint.pt"

LUNA_CONFIG="configs/single_pano_4090_panocity_paired_full_curriculum_luna_after_full_warmup_9h.yaml"
LUNA_OUT="$LUNA/logs/panocity_paired_full_curriculum_luna_after_full_warmup_4090_9h"
SEQ_LOG="$LUNA/logs/panocity_paired_full_curriculum_profile_then_stage_4090_12h_sequence.log"

mkdir -p "$DATA_ROOT/metadata" "$(dirname "$SEQ_LOG")" "$BASE_OUT" "$LUNA_OUT"

{
  echo "[sequence] started $(date --iso-8601=seconds)"
  echo "[sequence] data_root=$DATA_ROOT"
  echo "[sequence] manifest=$MANIFEST"
  echo "[sequence] baseline_config=$BASE_CONFIG"
  echo "[sequence] luna_config=$LUNA_CONFIG"
  echo "[sequence] python=$PYTHON"
  echo "[sequence] profile_workers=$PROFILE_WORKERS"
  echo "[sequence] force_profile=$FORCE_PROFILE"
} | tee -a "$SEQ_LOG"

if [[ "$FORCE_PROFILE" == "1" || ! -s "$MANIFEST" || ! -s "$SUMMARY" ]]; then
  echo "[sequence] profile started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
  cd "$ROOT"
  "$PYTHON" "$LUNA/scripts/profile_panocity_manifest.py" \
    --root "$DATA_ROOT" \
    --workers "$PROFILE_WORKERS" \
    --output-jsonl "$MANIFEST" \
    --summary-json "$SUMMARY" \
    2>&1 | tee "$PROFILE_LOG"
  echo "[sequence] profile finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
else
  echo "[sequence] profile skipped; existing manifest found $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
fi

if [[ ! -s "$MANIFEST" || ! -s "$SUMMARY" ]]; then
  echo "[sequence] missing manifest or summary after profile" | tee -a "$SEQ_LOG"
  exit 1
fi

"$PYTHON" - <<PY | tee -a "$SEQ_LOG"
import json
summary = json.load(open("$SUMMARY", "r", encoding="utf-8"))
print("[sequence] manifest_total_entries=", summary.get("total_entries"))
print("[sequence] manifest_quality_bins=", summary.get("quality_bins"))
print("[sequence] manifest_valid_ratio=", summary.get("metrics", {}).get("valid_ratio"))
PY

cd "$BASELINE"
echo "[sequence] stage1 baseline full warmup started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29651 CUDA_VISIBLE_DEVICES=0 \
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
LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29652 CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$LUNA${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" training/train_pano_omega.py --config "$LUNA_CONFIG" \
  2>&1 | tee -a "$LUNA_OUT/train_9h.log"
echo "[sequence] stage2/3 LUNA residual continuation finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
echo "[sequence] finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
