#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$BASELINE/.." && pwd)"
LUNA="$ROOT/Rethink_pano_new_exp_omega"

PYTHON="${PYTHON:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29911}"
PANOVGGT_ROOT="${PANOVGGT_ROOT:-/mnt/e/PanoVGGT_minimal_datasets/datasets}"
VGGT_OMEGA_CKPT="${VGGT_OMEGA_CKPT:-/home/aoki/RethinkPanoVGGT_omega/ckpt/vggt_omega_1b_512.pt}"
BAD_SAMPLE_LIST="${BAD_SAMPLE_LIST:-$LUNA/configs/structured3d_bad_scenes.txt}"

LORA_CONFIG="${LORA_CONFIG:-mixed4_pano_vggtomega_scalealigned_2pano_384_lora_9h}"
EXP_NAME="${EXP_NAME:-local_vggtomega_lora_only_12h_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-$BASELINE/logs/$EXP_NAME}"
LORA_ROOT="$OUT/lora_12h"
EVAL_OUT="${EVAL_OUT:-$OUT/eval_full_anchor_panovggt_counts}"
SEQ_LOG="$OUT/run_lora_only_12h_then_eval.log"
SKIP_INDEX_BUILD="${SKIP_INDEX_BUILD:-1}"
WINDOWS_PER_PANO="${WINDOWS_PER_PANO:-4}"
LORA_STAGE_PLAN="${LORA_STAGE_PLAN:-2:180,4:180,6:180,8:180}"
EVAL_LIMIT_PER_DATASET="${EVAL_LIMIT_PER_DATASET:-0}"

if [[ "${CLEAN_OUTPUT:-0}" == "1" ]]; then
  rm -rf "$OUT"
fi
mkdir -p "$LORA_ROOT" "$EVAL_OUT"

{
  echo "[baseline-lora-only] started $(date --iso-8601=seconds)"
  echo "[baseline-lora-only] baseline=$BASELINE"
  echo "[baseline-lora-only] config=$LORA_CONFIG"
  echo "[baseline-lora-only] exp_name=$EXP_NAME"
  echo "[baseline-lora-only] python=$PYTHON"
  echo "[baseline-lora-only] cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "[baseline-lora-only] panovggt_root=$PANOVGGT_ROOT"
  echo "[baseline-lora-only] vggt_omega_ckpt=$VGGT_OMEGA_CKPT"
  echo "[baseline-lora-only] stage_plan=$LORA_STAGE_PLAN"
  echo "[baseline-lora-only] windows_per_pano=$WINDOWS_PER_PANO"
  echo "[baseline-lora-only] eval_out=$EVAL_OUT"
} | tee -a "$SEQ_LOG"

if [[ "$SKIP_INDEX_BUILD" != "1" ]]; then
  cd "$LUNA"
  PYTHONPATH="$LUNA${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" scripts/build_mixed4_official_indexes.py \
    --root "$PANOVGGT_ROOT" \
    --train-fraction 0.95 \
    --seed 42 \
    --bad-scene-list "$BAD_SAMPLE_LIST" \
    2>&1 | tee -a "$SEQ_LOG"
else
  echo "[baseline-lora-only] mixed4 index rebuild skipped" | tee -a "$SEQ_LOG"
fi

run_train() {
  local out_dir="$1"
  local port="$2"
  shift 2
  mkdir -p "$out_dir"
  cd "$BASELINE"
  LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$port" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    PANOVGGT_ROOT="$PANOVGGT_ROOT" VGGT_OMEGA_CKPT="$VGGT_OMEGA_CKPT" \
    PYTHONPATH="$BASELINE${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" training/launch.py --config "$LORA_CONFIG" "$@" \
    2>&1 | tee "$out_dir/train_console.log"
}

IFS=',' read -r -a STAGES <<< "$LORA_STAGE_PLAN"
PREV_CKPT=""
FINAL_LORA_OUT=""
stage_index=0
for stage in "${STAGES[@]}"; do
  stage_index=$((stage_index + 1))
  pano_count="${stage%%:*}"
  minutes="${stage#*:}"
  max_img=$((pano_count * WINDOWS_PER_PANO))
  stage_name="stage${stage_index}_${pano_count}p_${minutes}m"
  stage_out="$LORA_ROOT/$stage_name"
  FINAL_LORA_OUT="$stage_out"
  resume_args=()
  if [[ -n "$PREV_CKPT" ]]; then
    resume_args+=(
      "checkpoint.resume_checkpoint_path=$PREV_CKPT"
      "checkpoint.resume_optimizer=false"
      "checkpoint.resume_training_progress=false"
      "checkpoint.resume_scaler=false"
    )
  fi
  echo "[baseline-lora-only] $stage_name started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
  run_train "$stage_out" "$((MASTER_PORT + stage_index))" \
    "exp_name=${EXP_NAME}_${stage_name}" \
    "logging.log_dir=$stage_out" \
    "checkpoint.save_dir=$stage_out/ckpts" \
    "max_duration_minutes=$minutes" \
    "checkpoint.save_step_freq=0" \
    "max_img_per_gpu=$max_img" \
    "data.train.common_config.img_nums=[$max_img,$max_img]" \
    "data.train.common_config.max_img_per_gpu=$max_img" \
    "data.train.dataset.dataset_configs.0.pano_min_count=$pano_count" \
    "data.train.dataset.dataset_configs.0.pano_max_count=$pano_count" \
    "data.train.dataset.dataset_configs.0.windows_per_pano=$WINDOWS_PER_PANO" \
    "${resume_args[@]}"
  PREV_CKPT="$stage_out/ckpts/checkpoint.pt"
  if [[ ! -s "$PREV_CKPT" ]]; then
    echo "[baseline-lora-only] missing checkpoint after $stage_name: $PREV_CKPT" | tee -a "$SEQ_LOG"
    exit 1
  fi
  echo "[baseline-lora-only] $stage_name finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
done

echo "[baseline-lora-only] PanoVGGT-count eval started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
PYTHON="$PYTHON" \
CONFIG="$LORA_CONFIG" \
CHECKPOINT="$PREV_CKPT" \
DATASET_ROOT="$PANOVGGT_ROOT" \
TRAIN_LOSS_CSV="$FINAL_LORA_OUT/loss.csv" \
OUT="$EVAL_OUT" \
LIMIT_PER_DATASET="$EVAL_LIMIT_PER_DATASET" \
DEVICE=cuda \
AMP_DTYPE=bfloat16 \
FOREGROUND=1 \
  bash "$BASELINE/scripts/run_eval_panovggt_counts.sh" \
  2>&1 | tee "$EVAL_OUT/run_eval_panovggt_counts_console.log"

echo "[baseline-lora-only] finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
