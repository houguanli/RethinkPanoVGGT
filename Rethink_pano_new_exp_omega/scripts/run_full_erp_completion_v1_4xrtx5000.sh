#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/whitehole/AOKI/RethinkPanoVGGT_omega_multipano_work_pro/Rethink_pano_new_exp_omega}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PANOVGGT_ROOT="${PANOVGGT_ROOT:-/whitehole/AOKI/panovggt}"
FOUNDATION_CHECKPOINT="${FOUNDATION_CHECKPOINT:-/whitehole/AOKI/vggt-omega/ckpt/vggt_omega_1b_512.pt}"
# Set this to the retained FoV75 A milestone when it is available on the server.
# Otherwise the canonical warm-up starts directly from the Omega foundation.
WARMUP_INIT_CHECKPOINT="${WARMUP_INIT_CHECKPOINT:-$FOUNDATION_CHECKPOINT}"
RUN_NAME="${RUN_NAME:-full_erp_completion_v1_4xrtx5000_2h8h2h}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
WARMUP_MINUTES="${WARMUP_MINUTES:-120}"
COMPLETION_MAIN_MINUTES="${COMPLETION_MAIN_MINUTES:-480}"
COMPLETION_REFINE_MINUTES="${COMPLETION_REFINE_MINUTES:-120}"
NUM_WORKERS="${NUM_WORKERS:-2}"
EVAL_WORKERS_PER_GPU="${EVAL_WORKERS_PER_GPU:-2}"
CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-2000}"
CONFIG="${CONFIG:-configs/multipano_rtx5000x4_mixed4_omega_canonical_warmup_2h.yaml}"

export CUDA_VISIBLE_DEVICES
export SKIP_INDEX_BUILD="${SKIP_INDEX_BUILD:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

WARMUP_OUTPUT="${PROJECT_ROOT}/logs/${RUN_NAME}_omega_warmup_2h"
MAIN_OUTPUT="${PROJECT_ROOT}/logs/${RUN_NAME}_completion_main_8h"
REFINE_OUTPUT="${PROJECT_ROOT}/logs/${RUN_NAME}_completion_refine_2h"
QUICK_EVAL="${PROJECT_ROOT}/logs/${RUN_NAME}_eval_quick20_full_erp_4gpu"
FULL_EVAL="${PROJECT_ROOT}/logs/${RUN_NAME}_eval_full8833_anchor_full_erp_4gpu"
PREVIEW="${PROJECT_ROOT}/logs/${RUN_NAME}_preview_panocity_test0"
PIPELINE_LOG="${PROJECT_ROOT}/logs/${RUN_NAME}_pipeline.log"
LOSS_ANALYSIS="${PROJECT_ROOT}/logs/${RUN_NAME}_loss_analysis"

cd "$PROJECT_ROOT"
mkdir -p "${PROJECT_ROOT}/logs" "$WARMUP_OUTPUT" "$MAIN_OUTPUT" "$REFINE_OUTPUT" "$QUICK_EVAL" "$FULL_EVAL" "$PREVIEW" "$LOSS_ANALYSIS"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

require_file() { [[ -s "$1" ]] || { echo "[ERROR] missing/empty file: $1"; exit 2; }; }
require_dir() { [[ -d "$1" ]] || { echo "[ERROR] missing directory: $1"; exit 2; }; }
require_dir "$PANOVGGT_ROOT"
require_file "$FOUNDATION_CHECKPOINT"
require_file "$WARMUP_INIT_CHECKPOINT"
require_file "$CONFIG"

IFS=',' read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#GPU_LIST[@]}" -ne "$NPROC_PER_NODE" ]]; then
  echo "[ERROR] CUDA_VISIBLE_DEVICES exposes ${#GPU_LIST[@]} GPUs but NPROC_PER_NODE=$NPROC_PER_NODE"
  exit 2
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  detected="$(nvidia-smi -L | wc -l)"
  (( detected >= NPROC_PER_NODE )) || { echo "[ERROR] detected only $detected GPUs"; exit 2; }
  nvidia-smi -L
fi

echo "[PIPELINE] root=$PROJECT_ROOT run=$RUN_NAME"
echo "[PIPELINE] dataset=$PANOVGGT_ROOT foundation=$FOUNDATION_CHECKPOINT init=$WARMUP_INIT_CHECKPOINT"
echo "[PIPELINE] GPUs=$CUDA_VISIBLE_DEVICES schedule=${WARMUP_MINUTES}m+${COMPLETION_MAIN_MINUTES}m+${COMPLETION_REFINE_MINUTES}m"
echo "[PIPELINE] checkpoint_every_steps=$CHECKPOINT_EVERY_STEPS; eval=quick20+full8833"

if [[ ! -s "$WARMUP_OUTPUT/last.pt" ]]; then
  echo "[STAGE 1/6] four-GPU canonical Omega warm-up"
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" \
    training/train_pano_omega.py \
    --config "$CONFIG" --dataset-root "$PANOVGGT_ROOT" \
    --base-checkpoint "$FOUNDATION_CHECKPOINT" --checkpoint "$WARMUP_INIT_CHECKPOINT" \
    --output-dir "$WARMUP_OUTPUT" --tensorboard-dir "$WARMUP_OUTPUT/tensorboard" \
    --debug-dir "$WARMUP_OUTPUT/debug" --max-duration-minutes "$WARMUP_MINUTES" \
    --num-workers "$NUM_WORKERS" --save-every-steps "$CHECKPOINT_EVERY_STEPS" --progress-bar \
    --no-inherit-checkpoint-training-defaults \
    2>&1 | tee -a "$WARMUP_OUTPUT/train.log"
fi
require_file "$WARMUP_OUTPUT/last.pt"

if [[ ! -s "$MAIN_OUTPUT/last.pt" ]]; then
  echo "[STAGE 2/6] four-GPU frozen-Omega completion main"
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" \
    training/train_erp_completion.py \
    --config "$CONFIG" --dataset-root "$PANOVGGT_ROOT" \
    --omega-checkpoint "$WARMUP_OUTPUT/last.pt" --base-checkpoint "$FOUNDATION_CHECKPOINT" \
    --output-dir "$MAIN_OUTPUT" --duration-minutes "$COMPLETION_MAIN_MINUTES" \
    --stage main --num-workers "$NUM_WORKERS" --save-every "$CHECKPOINT_EVERY_STEPS" --progress-bar \
    2>&1 | tee -a "$MAIN_OUTPUT/train.log"
fi
require_file "$MAIN_OUTPUT/last.pt"

if [[ ! -s "$REFINE_OUTPUT/last.pt" ]]; then
  echo "[STAGE 3/6] four-GPU boundary/polar refinement"
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" \
    training/train_erp_completion.py \
    --config "$CONFIG" --dataset-root "$PANOVGGT_ROOT" \
    --omega-checkpoint "$WARMUP_OUTPUT/last.pt" --base-checkpoint "$FOUNDATION_CHECKPOINT" \
    --output-dir "$REFINE_OUTPUT" --resume "$MAIN_OUTPUT/last.pt" \
    --duration-minutes "$COMPLETION_REFINE_MINUTES" --stage refine --lr 5e-5 \
    --num-workers "$NUM_WORKERS" --save-every "$CHECKPOINT_EVERY_STEPS" --progress-bar \
    2>&1 | tee -a "$REFINE_OUTPUT/train.log"
fi
require_file "$REFINE_OUTPUT/last.pt"

echo "[ANALYSIS] generating smoothed loss curves and health report"
"$PYTHON_BIN" scripts/analyze_full_erp_training_losses.py \
  --series "omega_warmup=$WARMUP_OUTPUT/loss.csv" \
  --series "completion_main=$MAIN_OUTPUT/loss.csv" \
  --series "completion_refine=$REFINE_OUTPUT/loss.csv" \
  --output-dir "$LOSS_ANALYSIS"

if [[ ! -s "$QUICK_EVAL/validation_mixed4_by_dataset_valtestfull_summary.json" ]]; then
  echo "[STAGE 4/6] four-shard learned full-ERP quick eval"
  GPUS="$CUDA_VISIBLE_DEVICES" PYTHON="$PYTHON_BIN" CONFIG="$CONFIG" \
    DATASET_ROOT="$PANOVGGT_ROOT" CHECKPOINT="$WARMUP_OUTPUT/last.pt" \
    ERP_COMPLETION_CHECKPOINT="$REFINE_OUTPUT/last.pt" TRAIN_LOSS_CSV="$REFINE_OUTPUT/loss.csv" \
    EVAL_OUT="$QUICK_EVAL" LIMIT_PER_DATASET=20 EVAL_DATASETS=all \
    NUM_WORKERS_PER_GPU="$EVAL_WORKERS_PER_GPU" SAMPLE_POLICY=anchor PANO_COUNT_POLICY=panovggt \
    RESUME=1 bash scripts/run_multipano_mixed4_eval_4gpu.sh
fi

if [[ ! -s "$PREVIEW/pred_range_depth_erp_completed.png" ]]; then
  echo "[STAGE 5/6] preview"
  CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" "$PYTHON_BIN" scripts/reconstruct_pano_omega.py \
    --dataset-root "$PANOVGGT_ROOT" --dataset-format pano_minimal --minimal-datasets panocity \
    --dataset-split test --sample-index 0 --checkpoint "$WARMUP_OUTPUT/last.pt" \
    --erp-completion-checkpoint "$REFINE_OUTPUT/last.pt" --output-dir "$PREVIEW" \
    --device cuda --pano-height 512 --pano-width 1024 --window-size 384 --num-yaw 4 \
    --pitch-degrees=-15 --fov-degrees 75 --fov-x-degrees 75 --fov-y-degrees 75
fi

SUMMARY="$FULL_EVAL/validation_mixed4_by_dataset_valtestfull_summary.json"
if [[ ! -s "$SUMMARY" ]]; then
  echo "[STAGE 6/6] formal learned full-ERP 8,833-set eval on four shards"
  GPUS="$CUDA_VISIBLE_DEVICES" PYTHON="$PYTHON_BIN" CONFIG="$CONFIG" \
    DATASET_ROOT="$PANOVGGT_ROOT" CHECKPOINT="$WARMUP_OUTPUT/last.pt" \
    ERP_COMPLETION_CHECKPOINT="$REFINE_OUTPUT/last.pt" TRAIN_LOSS_CSV="$REFINE_OUTPUT/loss.csv" \
    EVAL_OUT="$FULL_EVAL" LIMIT_PER_DATASET=0 EVAL_DATASETS=all \
    NUM_WORKERS_PER_GPU="$EVAL_WORKERS_PER_GPU" SAMPLE_POLICY=anchor PANO_COUNT_POLICY=panovggt \
    CAMERA_EVAL_MAX_PANOS=3 RESUME=1 bash scripts/run_multipano_mixed4_eval_4gpu.sh
fi
require_file "$SUMMARY"
"$PYTHON_BIN" scripts/validate_eval_cardinality.py "$SUMMARY"
echo "[COMPLETE] summary=$SUMMARY preview=$PREVIEW/pred_range_depth_erp_completed.png"
