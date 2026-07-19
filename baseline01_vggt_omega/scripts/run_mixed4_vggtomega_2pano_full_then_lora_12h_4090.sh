#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$BASELINE/.." && pwd)"
LUNA="$ROOT/Rethink_pano_new_exp_omega"

PYTHON="${PYTHON:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29761}"
PANOVGGT_ROOT="${PANOVGGT_ROOT:-/mnt/e/PanoVGGT_minimal_datasets/datasets}"
DEFAULT_VGGT_OMEGA_CKPT="/home/aoki/RethinkPanoVGGT_omega_compare_methods_only/ckpt/VGGT-Omega/vggt_omega_1b_512.pt"
if [[ ! -s "$DEFAULT_VGGT_OMEGA_CKPT" ]]; then
  DEFAULT_VGGT_OMEGA_CKPT="/home/aoki/RethinkPanoVGGT_omega/ckpt/vggt_omega_1b_512.pt"
fi
VGGT_OMEGA_CKPT="${VGGT_OMEGA_CKPT:-$DEFAULT_VGGT_OMEGA_CKPT}"
BAD_SAMPLE_LIST="${BAD_SAMPLE_LIST:-$LUNA/configs/structured3d_bad_scenes.txt}"

WARMUP_CONFIG="${WARMUP_CONFIG:-mixed4_pano_vggtomega_scalealigned_2pano_384_full_warmup_3h}"
LORA_CONFIG="${LORA_CONFIG:-mixed4_pano_vggtomega_scalealigned_2pano_384_lora_9h}"
EXP_NAME="${EXP_NAME:-local_vggtomega_2pano_full3h_lora9h_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-$BASELINE/logs/$EXP_NAME}"
WARMUP_OUT="$OUT/warmup_3h"
LORA_OUT="$OUT/lora_9h"
EVAL_OUT="${EVAL_OUT:-$OUT/eval_full_anchor_panovggt_counts}"
SEQ_LOG="$OUT/run_2pano_full3h_lora9h_then_eval.log"
EVAL_LIMIT_PER_DATASET="${EVAL_LIMIT_PER_DATASET:-0}"

if [[ "${CLEAN_OUTPUT:-0}" == "1" ]]; then
  rm -rf "$OUT"
fi
mkdir -p "$WARMUP_OUT" "$LORA_OUT" "$EVAL_OUT"

{
  echo "[baseline-2p-lora] started $(date --iso-8601=seconds)"
  echo "[baseline-2p-lora] baseline=$BASELINE"
  echo "[baseline-2p-lora] warmup_config=$WARMUP_CONFIG"
  echo "[baseline-2p-lora] lora_config=$LORA_CONFIG"
  echo "[baseline-2p-lora] exp_name=$EXP_NAME"
  echo "[baseline-2p-lora] python=$PYTHON"
  echo "[baseline-2p-lora] cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "[baseline-2p-lora] panovggt_root=$PANOVGGT_ROOT"
  echo "[baseline-2p-lora] vggt_omega_ckpt=$VGGT_OMEGA_CKPT"
  echo "[baseline-2p-lora] bad_sample_list=$BAD_SAMPLE_LIST"
  echo "[baseline-2p-lora] eval_out=$EVAL_OUT"
  echo "[baseline-2p-lora] eval_limit_per_dataset=$EVAL_LIMIT_PER_DATASET"
} | tee -a "$SEQ_LOG"

echo "[baseline-2p-lora] building/checking mixed4 official indexes $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
cd "$LUNA"
PYTHONPATH="$LUNA${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" scripts/build_mixed4_official_indexes.py \
  --root "$PANOVGGT_ROOT" \
  --train-fraction 0.95 \
  --seed 42 \
  --bad-scene-list "$BAD_SAMPLE_LIST" \
  2>&1 | tee -a "$SEQ_LOG"

echo "[baseline-2p-lora] stage1 full 2pano warmup started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
cd "$BASELINE"
LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  PANOVGGT_ROOT="$PANOVGGT_ROOT" VGGT_OMEGA_CKPT="$VGGT_OMEGA_CKPT" \
  PYTHONPATH="$BASELINE${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" training/launch.py --config "$WARMUP_CONFIG" \
    exp_name="${EXP_NAME}_warmup_3h" \
    logging.log_dir="$WARMUP_OUT" \
    checkpoint.save_dir="$WARMUP_OUT/ckpts" \
  2>&1 | tee "$WARMUP_OUT/train_console.log"
echo "[baseline-2p-lora] stage1 warmup finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"

WARMUP_CKPT="$WARMUP_OUT/ckpts/checkpoint.pt"
if [[ ! -s "$WARMUP_CKPT" ]]; then
  echo "[baseline-2p-lora] missing warmup checkpoint: $WARMUP_CKPT" | tee -a "$SEQ_LOG"
  exit 1
fi

LORA_STAGE_PLAN="${LORA_STAGE_PLAN:-2:135,4:135,6:135,8:135}"
WINDOWS_PER_PANO="${WINDOWS_PER_PANO:-4}"
echo "[baseline-2p-lora] stage2 LoRA residual plan=$LORA_STAGE_PLAN started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
IFS=',' read -r -a STAGES <<< "$LORA_STAGE_PLAN"
PREV_CKPT="$WARMUP_CKPT"
stage_index=0
FINAL_LORA_OUT=""
for stage in "${STAGES[@]}"; do
  stage_index=$((stage_index + 1))
  pano_count="${stage%%:*}"
  minutes="${stage#*:}"
  max_img=$((pano_count * WINDOWS_PER_PANO))
  stage_name="stage${stage_index}_${pano_count}p_${minutes}m"
  stage_out="$LORA_OUT/$stage_name"
  mkdir -p "$stage_out"
  FINAL_LORA_OUT="$stage_out"
  echo "[baseline-2p-lora] LoRA $stage_name started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
  cd "$BASELINE"
  LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$((MASTER_PORT + stage_index))" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    PANOVGGT_ROOT="$PANOVGGT_ROOT" VGGT_OMEGA_CKPT="$VGGT_OMEGA_CKPT" \
    PYTHONPATH="$BASELINE${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" training/launch.py --config "$LORA_CONFIG" \
      exp_name="${EXP_NAME}_lora_${stage_name}" \
      logging.log_dir="$stage_out" \
      checkpoint.save_dir="$stage_out/ckpts" \
      checkpoint.resume_checkpoint_path="$PREV_CKPT" \
      checkpoint.resume_optimizer=false \
      checkpoint.resume_training_progress=false \
      checkpoint.resume_scaler=false \
      max_duration_minutes="$minutes" \
      max_img_per_gpu="$max_img" \
      data.train.common_config.img_nums="[$max_img,$max_img]" \
      data.train.common_config.max_img_per_gpu="$max_img" \
      data.train.dataset.dataset_configs.0.pano_min_count="$pano_count" \
      data.train.dataset.dataset_configs.0.pano_max_count="$pano_count" \
      data.train.dataset.dataset_configs.0.windows_per_pano="$WINDOWS_PER_PANO" \
    2>&1 | tee "$stage_out/train_console.log"
  PREV_CKPT="$stage_out/ckpts/checkpoint.pt"
  if [[ ! -s "$PREV_CKPT" ]]; then
    echo "[baseline-2p-lora] missing LoRA checkpoint after $stage_name: $PREV_CKPT" | tee -a "$SEQ_LOG"
    exit 1
  fi
  echo "[baseline-2p-lora] LoRA $stage_name finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
done
echo "[baseline-2p-lora] stage2 LoRA finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"

LORA_CKPT="$PREV_CKPT"

echo "[baseline-2p-lora] plotting loss curves $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
"$PYTHON" scripts/plot_loss_curve.py \
  --loss-csv "$WARMUP_OUT/loss.csv" \
  --output "$WARMUP_OUT/loss_curve_smoothed.png" \
  --smooth 401 \
  2>&1 | tee -a "$SEQ_LOG" || true
"$PYTHON" scripts/plot_loss_curve.py \
  --loss-csv "$FINAL_LORA_OUT/loss.csv" \
  --output "$LORA_OUT/loss_curve_smoothed.png" \
  --smooth 401 \
  2>&1 | tee -a "$SEQ_LOG" || true

echo "[baseline-2p-lora] full mixed4 PanoVGGT-count eval started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
PYTHON="$PYTHON" \
CONFIG="$LORA_CONFIG" \
CHECKPOINT="$LORA_CKPT" \
DATASET_ROOT="$PANOVGGT_ROOT" \
TRAIN_LOSS_CSV="$FINAL_LORA_OUT/loss.csv" \
OUT="$EVAL_OUT" \
LIMIT_PER_DATASET="$EVAL_LIMIT_PER_DATASET" \
DEVICE=cuda \
AMP_DTYPE=bfloat16 \
FOREGROUND=1 \
  bash scripts/run_eval_panovggt_counts.sh \
  2>&1 | tee "$EVAL_OUT/run_eval_panovggt_counts_console.log"
echo "[baseline-2p-lora] full mixed4 PanoVGGT-count eval finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"

echo "[baseline-2p-lora] finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
