#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$BASELINE/.." && pwd)"
LUNA="$ROOT/Rethink_pano_new_exp_omega"

PYTHON="${PYTHON:-/home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29811}"
PANOVGGT_ROOT="${PANOVGGT_ROOT:-/mnt/e/PanoVGGT_minimal_datasets/datasets}"
DEFAULT_VGGT_OMEGA_CKPT="/home/aoki/RethinkPanoVGGT_omega_compare_methods_only/ckpt/VGGT-Omega/vggt_omega_1b_512.pt"
if [[ ! -s "$DEFAULT_VGGT_OMEGA_CKPT" ]]; then
  DEFAULT_VGGT_OMEGA_CKPT="/home/aoki/RethinkPanoVGGT_omega/ckpt/vggt_omega_1b_512.pt"
fi
VGGT_OMEGA_CKPT="${VGGT_OMEGA_CKPT:-$DEFAULT_VGGT_OMEGA_CKPT}"
BAD_SAMPLE_LIST="${BAD_SAMPLE_LIST:-$LUNA/configs/structured3d_bad_scenes.txt}"

WARMUP_CONFIG="${WARMUP_CONFIG:-mixed4_pano_vggtomega_scalealigned_2pano_384_full_warmup_3h}"
LORA_CONFIG="${LORA_CONFIG:-mixed4_pano_vggtomega_scalealigned_2pano_384_lora_9h}"
EXP_NAME="${EXP_NAME:-local_vggtomega_2pfull1h_lora2h_pano2to8_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-$BASELINE/logs/$EXP_NAME}"
WARMUP_OUT="$OUT/warmup_1h"
LORA_ROOT="$OUT/lora_2h"
EVAL_OUT="${EVAL_OUT:-$OUT/eval_full_anchor_panovggt_counts}"
SEQ_LOG="$OUT/run_2pfull1h_lora2h_then_eval.log"
SKIP_INDEX_BUILD="${SKIP_INDEX_BUILD:-1}"
WINDOWS_PER_PANO="${WINDOWS_PER_PANO:-4}"
WARMUP_MINUTES="${WARMUP_MINUTES:-60}"
LORA_STAGE_PLAN="${LORA_STAGE_PLAN:-2:30,4:30,6:30,8:30}"
EVAL_LIMIT_PER_DATASET="${EVAL_LIMIT_PER_DATASET:-0}"

if [[ "${CLEAN_OUTPUT:-0}" == "1" ]]; then
  rm -rf "$OUT"
fi
mkdir -p "$WARMUP_OUT" "$LORA_ROOT" "$EVAL_OUT"

{
  echo "[baseline-light-lora] started $(date --iso-8601=seconds)"
  echo "[baseline-light-lora] baseline=$BASELINE"
  echo "[baseline-light-lora] warmup_config=$WARMUP_CONFIG"
  echo "[baseline-light-lora] lora_config=$LORA_CONFIG"
  echo "[baseline-light-lora] exp_name=$EXP_NAME"
  echo "[baseline-light-lora] python=$PYTHON"
  echo "[baseline-light-lora] cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "[baseline-light-lora] panovggt_root=$PANOVGGT_ROOT"
  echo "[baseline-light-lora] vggt_omega_ckpt=$VGGT_OMEGA_CKPT"
  echo "[baseline-light-lora] warmup_minutes=$WARMUP_MINUTES"
  echo "[baseline-light-lora] lora_stage_plan=$LORA_STAGE_PLAN"
  echo "[baseline-light-lora] eval_out=$EVAL_OUT"
  echo "[baseline-light-lora] eval_limit_per_dataset=$EVAL_LIMIT_PER_DATASET"
} | tee -a "$SEQ_LOG"

if [[ "$SKIP_INDEX_BUILD" != "1" ]]; then
  echo "[baseline-light-lora] rebuilding mixed4 official indexes $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
  cd "$LUNA"
  PYTHONPATH="$LUNA${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" scripts/build_mixed4_official_indexes.py \
    --root "$PANOVGGT_ROOT" \
    --train-fraction 0.95 \
    --seed 42 \
    --bad-scene-list "$BAD_SAMPLE_LIST" \
    2>&1 | tee -a "$SEQ_LOG"
else
  echo "[baseline-light-lora] mixed4 index rebuild skipped" | tee -a "$SEQ_LOG"
fi

run_train() {
  local config_name="$1"
  local out_dir="$2"
  local port="$3"
  shift 3
  mkdir -p "$out_dir"
  cd "$BASELINE"
  LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$port" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    PANOVGGT_ROOT="$PANOVGGT_ROOT" VGGT_OMEGA_CKPT="$VGGT_OMEGA_CKPT" \
    PYTHONPATH="$BASELINE${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" training/launch.py --config "$config_name" "$@" \
    2>&1 | tee "$out_dir/train_console.log"
}

echo "[baseline-light-lora] 2p full warmup started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
run_train "$WARMUP_CONFIG" "$WARMUP_OUT" "$MASTER_PORT" \
  "exp_name=${EXP_NAME}_warmup_1h" \
  "logging.log_dir=$WARMUP_OUT" \
  "checkpoint.save_dir=$WARMUP_OUT/ckpts" \
  "max_duration_minutes=$WARMUP_MINUTES" \
  "checkpoint.save_step_freq=0"
echo "[baseline-light-lora] 2p full warmup finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"

PREV_CKPT="$WARMUP_OUT/ckpts/checkpoint.pt"
if [[ ! -s "$PREV_CKPT" ]]; then
  echo "[baseline-light-lora] missing warmup checkpoint: $PREV_CKPT" | tee -a "$SEQ_LOG"
  exit 1
fi

IFS=',' read -r -a STAGES <<< "$LORA_STAGE_PLAN"
stage_index=0
FINAL_LORA_OUT=""
for stage in "${STAGES[@]}"; do
  stage_index=$((stage_index + 1))
  pano_count="${stage%%:*}"
  minutes="${stage#*:}"
  max_img=$((pano_count * WINDOWS_PER_PANO))
  stage_name="stage${stage_index}_${pano_count}p_${minutes}m"
  stage_out="$LORA_ROOT/$stage_name"
  FINAL_LORA_OUT="$stage_out"
  echo "[baseline-light-lora] LoRA $stage_name started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
  run_train "$LORA_CONFIG" "$stage_out" "$((MASTER_PORT + stage_index))" \
    "exp_name=${EXP_NAME}_lora_${stage_name}" \
    "logging.log_dir=$stage_out" \
    "checkpoint.save_dir=$stage_out/ckpts" \
    "checkpoint.resume_checkpoint_path=$PREV_CKPT" \
    "checkpoint.resume_optimizer=false" \
    "checkpoint.resume_training_progress=false" \
    "checkpoint.resume_scaler=false" \
    "max_duration_minutes=$minutes" \
    "checkpoint.save_step_freq=0" \
    "max_img_per_gpu=$max_img" \
    "data.train.common_config.img_nums=[$max_img,$max_img]" \
    "data.train.common_config.max_img_per_gpu=$max_img" \
    "data.train.dataset.dataset_configs.0.pano_min_count=$pano_count" \
    "data.train.dataset.dataset_configs.0.pano_max_count=$pano_count" \
    "data.train.dataset.dataset_configs.0.windows_per_pano=$WINDOWS_PER_PANO"
  PREV_CKPT="$stage_out/ckpts/checkpoint.pt"
  if [[ ! -s "$PREV_CKPT" ]]; then
    echo "[baseline-light-lora] missing LoRA checkpoint after $stage_name: $PREV_CKPT" | tee -a "$SEQ_LOG"
    exit 1
  fi
  echo "[baseline-light-lora] LoRA $stage_name finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
done

FINAL_CKPT="$PREV_CKPT"
FINAL_LOSS_CSV="$FINAL_LORA_OUT/loss.csv"
echo "[baseline-light-lora] final_lora_checkpoint=$FINAL_CKPT" | tee -a "$SEQ_LOG"

echo "[baseline-light-lora] PanoVGGT-count eval started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
PYTHON="$PYTHON" \
CONFIG="$LORA_CONFIG" \
CHECKPOINT="$FINAL_CKPT" \
DATASET_ROOT="$PANOVGGT_ROOT" \
TRAIN_LOSS_CSV="$FINAL_LOSS_CSV" \
OUT="$EVAL_OUT" \
LIMIT_PER_DATASET="$EVAL_LIMIT_PER_DATASET" \
DEVICE=cuda \
AMP_DTYPE=bfloat16 \
FOREGROUND=1 \
  bash "$BASELINE/scripts/run_eval_panovggt_counts.sh" \
  2>&1 | tee "$EVAL_OUT/run_eval_panovggt_counts_console.log"
echo "[baseline-light-lora] PanoVGGT-count eval finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"

echo "[baseline-light-lora] finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
