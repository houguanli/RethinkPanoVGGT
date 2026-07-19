#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$BASELINE/.." && pwd)"
LUNA="$ROOT/Rethink_pano_new_exp_omega"

PYTHON="${PYTHON:-/home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29791}"
PANOVGGT_ROOT="${PANOVGGT_ROOT:-/mnt/e/PanoVGGT_minimal_datasets/datasets}"
DEFAULT_VGGT_OMEGA_CKPT="/home/aoki/RethinkPanoVGGT_omega_compare_methods_only/ckpt/VGGT-Omega/vggt_omega_1b_512.pt"
if [[ ! -s "$DEFAULT_VGGT_OMEGA_CKPT" ]]; then
  DEFAULT_VGGT_OMEGA_CKPT="/home/aoki/RethinkPanoVGGT_omega/ckpt/vggt_omega_1b_512.pt"
fi
VGGT_OMEGA_CKPT="${VGGT_OMEGA_CKPT:-$DEFAULT_VGGT_OMEGA_CKPT}"
BAD_SAMPLE_LIST="${BAD_SAMPLE_LIST:-$LUNA/configs/structured3d_bad_scenes.txt}"

WARMUP_CONFIG="${WARMUP_CONFIG:-mixed4_pano_vggtomega_scalealigned_2pano_384_full_warmup_3h}"
LORA_CONFIG="${LORA_CONFIG:-mixed4_pano_vggtomega_scalealigned_2pano_384_lora_9h}"
EXP_NAME="${EXP_NAME:-local_vggtomega_lora_8p_smoke_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-$BASELINE/logs/$EXP_NAME}"
WARMUP_OUT="$OUT/warmup_short"
LORA_OUT="$OUT/lora_8p_smoke"
SEQ_LOG="$OUT/run_lora_8p_smoke.log"
SKIP_INDEX_BUILD="${SKIP_INDEX_BUILD:-1}"
WINDOWS_PER_PANO="${WINDOWS_PER_PANO:-4}"
SMOKE_WARMUP_BATCHES="${SMOKE_WARMUP_BATCHES:-0}"
SMOKE_LORA_BATCHES="${SMOKE_LORA_BATCHES:-1}"

if [[ "${CLEAN_OUTPUT:-1}" == "1" ]]; then
  rm -rf "$OUT"
fi
mkdir -p "$WARMUP_OUT" "$LORA_OUT"

{
  echo "[baseline-lora-smoke] started $(date --iso-8601=seconds)"
  echo "[baseline-lora-smoke] baseline=$BASELINE"
  echo "[baseline-lora-smoke] warmup_config=$WARMUP_CONFIG"
  echo "[baseline-lora-smoke] lora_config=$LORA_CONFIG"
  echo "[baseline-lora-smoke] exp_name=$EXP_NAME"
  echo "[baseline-lora-smoke] python=$PYTHON"
  echo "[baseline-lora-smoke] cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "[baseline-lora-smoke] panovggt_root=$PANOVGGT_ROOT"
  echo "[baseline-lora-smoke] vggt_omega_ckpt=$VGGT_OMEGA_CKPT"
  echo "[baseline-lora-smoke] windows_per_pano=$WINDOWS_PER_PANO"
  echo "[baseline-lora-smoke] warmup_batches=$SMOKE_WARMUP_BATCHES"
  echo "[baseline-lora-smoke] lora_batches=$SMOKE_LORA_BATCHES"
} | tee -a "$SEQ_LOG"

if [[ "$SKIP_INDEX_BUILD" != "1" ]]; then
  echo "[baseline-lora-smoke] rebuilding mixed4 official indexes $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
  cd "$LUNA"
  PYTHONPATH="$LUNA${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" scripts/build_mixed4_official_indexes.py \
    --root "$PANOVGGT_ROOT" \
    --train-fraction 0.95 \
    --seed 42 \
    --bad-scene-list "$BAD_SAMPLE_LIST" \
    2>&1 | tee -a "$SEQ_LOG"
else
  echo "[baseline-lora-smoke] mixed4 index rebuild skipped" | tee -a "$SEQ_LOG"
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

echo "[baseline-lora-smoke] short 2p full warmup started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
run_train "$WARMUP_CONFIG" "$WARMUP_OUT" "$MASTER_PORT" \
  "exp_name=${EXP_NAME}_warmup_short" \
  "logging.log_dir=$WARMUP_OUT" \
  "checkpoint.save_dir=$WARMUP_OUT/ckpts" \
  "limit_train_batches=$SMOKE_WARMUP_BATCHES" \
  "max_epochs=1" \
  "max_duration_minutes=0" \
  "logging.progress_log_every=1" \
  "checkpoint.save_step_freq=1"
echo "[baseline-lora-smoke] short warmup finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"

WARMUP_CKPT="$WARMUP_OUT/ckpts/checkpoint.pt"
if [[ ! -s "$WARMUP_CKPT" ]]; then
  echo "[baseline-lora-smoke] missing warmup checkpoint: $WARMUP_CKPT" | tee -a "$SEQ_LOG"
  exit 1
fi

PANO_COUNT=8
MAX_IMG=$((PANO_COUNT * WINDOWS_PER_PANO))
echo "[baseline-lora-smoke] LoRA 8p/$MAX_IMG-window smoke started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
run_train "$LORA_CONFIG" "$LORA_OUT" "$((MASTER_PORT + 1))" \
  "exp_name=${EXP_NAME}_lora_8p_smoke" \
  "logging.log_dir=$LORA_OUT" \
  "checkpoint.save_dir=$LORA_OUT/ckpts" \
  "checkpoint.resume_checkpoint_path=$WARMUP_CKPT" \
  "checkpoint.resume_optimizer=false" \
  "checkpoint.resume_training_progress=false" \
  "checkpoint.resume_scaler=false" \
  "limit_train_batches=$SMOKE_LORA_BATCHES" \
  "max_epochs=1" \
  "max_duration_minutes=0" \
  "logging.progress_log_every=1" \
  "checkpoint.save_step_freq=1" \
  "max_img_per_gpu=$MAX_IMG" \
  "data.train.common_config.img_nums=[$MAX_IMG,$MAX_IMG]" \
  "data.train.common_config.max_img_per_gpu=$MAX_IMG" \
  "data.train.dataset.dataset_configs.0.pano_min_count=$PANO_COUNT" \
  "data.train.dataset.dataset_configs.0.pano_max_count=$PANO_COUNT" \
  "data.train.dataset.dataset_configs.0.windows_per_pano=$WINDOWS_PER_PANO"
echo "[baseline-lora-smoke] LoRA 8p smoke finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"

echo "[baseline-lora-smoke] finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
