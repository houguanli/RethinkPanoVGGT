#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python}"
DATASET_ROOT="${PANOVGGT_ROOT:-/mnt/e/PanoVGGT_minimal_datasets/datasets}"
FOUNDATION_CHECKPOINT="${FOUNDATION_CHECKPOINT:-/home/aoki/RethinkPanoVGGT_omega_compare_methods_only/ckpt/VGGT-Omega/vggt_omega_1b_512.pt}"
A_MILESTONE_CHECKPOINT="${A_MILESTONE_CHECKPOINT:-${PROJECT_ROOT}/logs/m1_fovxy_core_ab_20260823_fov75x75/last.pt}"
A_MILESTONE_SUMMARY="${A_MILESTONE_SUMMARY:-${PROJECT_ROOT}/logs/m1_fovxy_core_ab_20260823_fov75x75/eval_full8833_anchor_canonical_pitch15_fov75x75/summary.json}"
RUN_NAME="${RUN_NAME:-full_erp_completion_v1_2h8h2h_20260824}"
WARMUP_MINUTES="${WARMUP_MINUTES:-120}"
COMPLETION_MAIN_MINUTES="${COMPLETION_MAIN_MINUTES:-480}"
COMPLETION_REFINE_MINUTES="${COMPLETION_REFINE_MINUTES:-120}"
NUM_WORKERS="${NUM_WORKERS:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
export SKIP_INDEX_BUILD="${SKIP_INDEX_BUILD:-1}"

CONFIG="configs/multipano_4090_mixed4_omega_canonical_warmup_2h.yaml"
WARMUP_OUTPUT="${PROJECT_ROOT}/logs/${RUN_NAME}_omega_warmup_2h"
MAIN_OUTPUT="${PROJECT_ROOT}/logs/${RUN_NAME}_completion_main_8h"
REFINE_OUTPUT="${PROJECT_ROOT}/logs/${RUN_NAME}_completion_refine_2h"
QUICK_EVAL="${PROJECT_ROOT}/logs/${RUN_NAME}_eval_quick20_full_erp"
FULL_EVAL="${PROJECT_ROOT}/logs/${RUN_NAME}_eval_full8833_anchor_full_erp"
PREVIEW="${PROJECT_ROOT}/logs/${RUN_NAME}_preview_panocity_test0"
PREFLIGHT="${PROJECT_ROOT}/logs/${RUN_NAME}_completion_preflight"
PIPELINE_LOG="${PROJECT_ROOT}/logs/${RUN_NAME}_pipeline.log"

mkdir -p "${WARMUP_OUTPUT}" "${MAIN_OUTPUT}" "${REFINE_OUTPUT}" "${QUICK_EVAL}" "${FULL_EVAL}" "${PREVIEW}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1
cd "${PROJECT_ROOT}"

wait_for_idle_gpu() {
  local consecutive=0
  local memory_used utilization
  while (( consecutive < 2 )); do
    IFS=',' read -r memory_used utilization < <(
      nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -n 1
    )
    memory_used="${memory_used//[[:space:]]/}"
    utilization="${utilization//[[:space:]]/}"
    if (( memory_used <= 1024 && utilization <= 10 )); then
      consecutive=$((consecutive + 1))
    else
      consecutive=0
      echo "[WAIT] GPU busy: memory=${memory_used}MiB utilization=${utilization}%"
    fi
    (( consecutive >= 2 )) || sleep 30
  done
}

for required in "${PYTHON_BIN}" "${DATASET_ROOT}" "${FOUNDATION_CHECKPOINT}" "${A_MILESTONE_CHECKPOINT}"; do
  [[ -e "${required}" ]] || { echo "[ERROR] missing ${required}"; exit 2; }
done

echo "[WAIT] preserving A as M1 milestone; waiting for its formal 8833-set summary"
while [[ ! -s "${A_MILESTONE_SUMMARY}" ]]; do
  sleep 60
done
"${PYTHON_BIN}" scripts/validate_eval_cardinality.py "${A_MILESTONE_SUMMARY}"
# Avoid racing a previous evaluator during its final file flush/preview.
while pgrep -af "scripts/evaluate_mixed4_depth_checkpoint.py" | grep -v "$$" >/dev/null; do
  echo "[WAIT] previous evaluator is still exiting"
  sleep 30
done
# Another user pipeline was already queued behind the M1 evaluator. Give it a
# deterministic head start, then require two consecutive idle checks.
sleep "${POST_M1_GRACE_SECONDS:-90}"
wait_for_idle_gpu

if [[ ! -s "${PREFLIGHT}/last.pt" ]]; then
  echo "[PREFLIGHT] one real frozen-Omega completion update"
  mkdir -p "${PREFLIGHT}"
  "${PYTHON_BIN}" training/train_erp_completion.py \
    --config "${CONFIG}" --omega-checkpoint "${A_MILESTONE_CHECKPOINT}" \
    --base-checkpoint "${FOUNDATION_CHECKPOINT}" --output-dir "${PREFLIGHT}" \
    --duration-minutes 5 --max-steps 1 --stage main --head-width 8 \
    --height 128 --width 256 --num-workers 0 2>&1 | tee -a "${PREFLIGHT}/train.log"
fi

if [[ ! -s "${WARMUP_OUTPUT}/last.pt" ]]; then
  echo "[STAGE 1/5] canonical full-Omega warm-up (${WARMUP_MINUTES} minutes)"
  "${PYTHON_BIN}" training/train_pano_omega.py \
    --config "${CONFIG}" \
    --dataset-root "${DATASET_ROOT}" \
    --base-checkpoint "${FOUNDATION_CHECKPOINT}" \
    --checkpoint "${A_MILESTONE_CHECKPOINT}" \
    --output-dir "${WARMUP_OUTPUT}" \
    --tensorboard-dir "${WARMUP_OUTPUT}/tensorboard" \
    --debug-dir "${WARMUP_OUTPUT}/debug" \
    --max-duration-minutes "${WARMUP_MINUTES}" \
    --num-workers "${NUM_WORKERS}" \
    --no-inherit-checkpoint-training-defaults 2>&1 | tee -a "${WARMUP_OUTPUT}/train.log"
fi
test -s "${WARMUP_OUTPUT}/last.pt"

if [[ ! -s "${MAIN_OUTPUT}/last.pt" ]]; then
  echo "[STAGE 2/5] frozen-Omega remaining-band main training (${COMPLETION_MAIN_MINUTES} minutes)"
  "${PYTHON_BIN}" training/train_erp_completion.py \
    --config "${CONFIG}" \
    --omega-checkpoint "${WARMUP_OUTPUT}/last.pt" \
    --base-checkpoint "${FOUNDATION_CHECKPOINT}" \
    --output-dir "${MAIN_OUTPUT}" \
    --duration-minutes "${COMPLETION_MAIN_MINUTES}" \
    --stage main --num-workers "${NUM_WORKERS}" 2>&1 | tee -a "${MAIN_OUTPUT}/train.log"
fi
test -s "${MAIN_OUTPUT}/last.pt"

if [[ ! -s "${REFINE_OUTPUT}/last.pt" ]]; then
  echo "[STAGE 3/5] boundary/polar refinement (${COMPLETION_REFINE_MINUTES} minutes)"
  "${PYTHON_BIN}" training/train_erp_completion.py \
    --config "${CONFIG}" \
    --omega-checkpoint "${WARMUP_OUTPUT}/last.pt" \
    --base-checkpoint "${FOUNDATION_CHECKPOINT}" \
    --output-dir "${REFINE_OUTPUT}" \
    --resume "${MAIN_OUTPUT}/last.pt" \
    --duration-minutes "${COMPLETION_REFINE_MINUTES}" \
    --stage refine --lr 5e-5 --num-workers "${NUM_WORKERS}" 2>&1 | tee -a "${REFINE_OUTPUT}/train.log"
fi
test -s "${REFINE_OUTPUT}/last.pt"

if [[ ! -s "${QUICK_EVAL}/summary.json" ]]; then
  echo "[STAGE 4/5] learned full-ERP quick evaluation"
  "${PYTHON_BIN}" scripts/evaluate_mixed4_depth_checkpoint.py \
    --config "${CONFIG}" --dataset-root "${DATASET_ROOT}" \
    --checkpoint "${WARMUP_OUTPUT}/last.pt" \
    --erp-completion-checkpoint "${REFINE_OUTPUT}/last.pt" \
    --output "${QUICK_EVAL}/summary.json" --per-sample-csv "${QUICK_EVAL}/per_sample.csv" \
    --camera-pair-csv "${QUICK_EVAL}/camera_pairs.csv" --progress-file "${QUICK_EVAL}/progress.json" \
    --datasets all --limit-per-dataset 20 --sample-policy anchor --pano-count-policy panovggt \
    --camera-eval-max-panos 3 --window-size 384 --num-yaw 4 --pitch-degrees=-15 \
    --fov-degrees 75 --device cuda --num-workers 0 --amp-dtype bfloat16 --seed 123 \
    --resume --no-print-each-sample
fi

if [[ ! -s "${PREVIEW}/pred_range_depth_erp_completed.png" ]]; then
  "${PYTHON_BIN}" scripts/reconstruct_pano_omega.py \
    --dataset-root "${DATASET_ROOT}" --dataset-format pano_minimal --minimal-datasets panocity \
    --dataset-split test --sample-index 0 --checkpoint "${WARMUP_OUTPUT}/last.pt" \
    --erp-completion-checkpoint "${REFINE_OUTPUT}/last.pt" --output-dir "${PREVIEW}" \
    --device cuda --pano-height 512 --pano-width 1024 --window-size 384 --num-yaw 4 \
    --pitch-degrees=-15 --fov-degrees 75 --fov-x-degrees 75 --fov-y-degrees 75
fi

if [[ ! -s "${FULL_EVAL}/summary.json" ]]; then
  echo "[STAGE 5/5] formal learned full-ERP 8833-set evaluation"
  "${PYTHON_BIN}" scripts/evaluate_mixed4_depth_checkpoint.py \
    --config "${CONFIG}" --dataset-root "${DATASET_ROOT}" \
    --checkpoint "${WARMUP_OUTPUT}/last.pt" \
    --erp-completion-checkpoint "${REFINE_OUTPUT}/last.pt" \
    --output "${FULL_EVAL}/summary.json" --per-sample-csv "${FULL_EVAL}/per_sample.csv" \
    --camera-pair-csv "${FULL_EVAL}/camera_pairs.csv" --progress-file "${FULL_EVAL}/progress.json" \
    --progress-every 25 --datasets all --limit-per-dataset 0 --sample-policy anchor \
    --pano-count-policy panovggt --camera-eval-max-panos 3 --window-size 384 --num-yaw 4 \
    --pitch-degrees=-15 --fov-degrees 75 --device cuda --num-workers 0 --amp-dtype bfloat16 \
    --seed 123 --resume --no-print-each-sample
fi
"${PYTHON_BIN}" scripts/validate_eval_cardinality.py "${FULL_EVAL}/summary.json"
echo "[COMPLETE] learned full-ERP v1 training/evaluation finished: ${FULL_EVAL}/summary.json"
