#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python}"
DATASET_ROOT="${PANOVGGT_ROOT:-/mnt/e/PanoVGGT_minimal_datasets/datasets}"
FOUNDATION_CHECKPOINT="${FOUNDATION_CHECKPOINT:-/home/aoki/RethinkPanoVGGT_omega_compare_methods_only/ckpt/VGGT-Omega/vggt_omega_1b_512.pt}"
COMMON_PARENT_CHECKPOINT="${COMMON_PARENT_CHECKPOINT:-/home/aoki/RethinkPanoVGGT_omega_multipano_work/Rethink_pano_new_exp_omega/logs/local_tail3_low_to_high_12h_20260718_002_warmup_3h/last.pt}"
RUN_PREFIX="${RUN_PREFIX:-m1_fovxy_core_ab_20260823}"
TRAIN_MINUTES="${TRAIN_MINUTES:-180}"
# Full indexed anchor sets (216/891/1662/6064 = 8833 sets) are the default for
# promotion decisions. Set a positive value explicitly for a quick diagnostic.
EVAL_LIMIT_PER_DATASET="${EVAL_LIMIT_PER_DATASET:-0}"
# A=75x75 matches the VGGT-Omega perspective input distribution and is the M1
# milestone. The wider-FoV treatment is opt-in so a formal A run never spends
# another full 8833-set pass on B unless explicitly requested.
RUN_B_ARM="${RUN_B_ARM:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
export SKIP_INDEX_BUILD="${SKIP_INDEX_BUILD:-1}"

CONFIG_A="configs/multipano_4090_mixed4_m1_fov75x75_pitch15_luna_3h.yaml"
CONFIG_B="configs/multipano_4090_mixed4_m1_fov95x75_pitch15_luna_3h.yaml"
OUTPUT_A="${PROJECT_ROOT}/logs/${RUN_PREFIX}_fov75x75"
OUTPUT_B="${PROJECT_ROOT}/logs/${RUN_PREFIX}_fov95x75"
SEQUENCE_LOG="${PROJECT_ROOT}/logs/${RUN_PREFIX}_sequence.log"

mkdir -p "$(dirname "${SEQUENCE_LOG}")" "${OUTPUT_A}" "${OUTPUT_B}"
exec > >(tee -a "${SEQUENCE_LOG}") 2>&1

for required_path in "${PYTHON_BIN}" "${DATASET_ROOT}" "${FOUNDATION_CHECKPOINT}" "${COMMON_PARENT_CHECKPOINT}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "[ERROR] required path missing: ${required_path}"
    exit 2
  fi
done

cd "${PROJECT_ROOT}"
echo "[INFO] M1 A/B parent=${COMMON_PARENT_CHECKPOINT} train_minutes=${TRAIN_MINUTES} eval_limit=${EVAL_LIMIT_PER_DATASET}"

run_training() {
  local arm="$1"
  local config="$2"
  local output="$3"
  if [[ -s "${output}/last.pt" ]]; then
    echo "[INFO] ${arm} training already complete: ${output}/last.pt"
    return
  fi
  echo "[STAGE] train ${arm}"
  "${PYTHON_BIN}" training/train_pano_omega.py \
    --config "${config}" \
    --dataset-root "${DATASET_ROOT}" \
    --base-checkpoint "${FOUNDATION_CHECKPOINT}" \
    --checkpoint "${COMMON_PARENT_CHECKPOINT}" \
    --output-dir "${output}" \
    --tensorboard-dir "${output}/tensorboard" \
    --debug-dir "${output}/debug" \
    --max-duration-minutes "${TRAIN_MINUTES}" \
    --num-workers 0 \
    2>&1 | tee -a "${output}/train.log"
  test -s "${output}/last.pt"
}

run_eval() {
  local arm="$1"
  local config="$2"
  local output="$3"
  local eval_scope="full8833_anchor"
  if (( EVAL_LIMIT_PER_DATASET > 0 )); then
    eval_scope="limit${EVAL_LIMIT_PER_DATASET}"
  fi
  local eval_dir="${output}/eval_${eval_scope}_canonical_pitch15_fov75x75"
  mkdir -p "${eval_dir}"
  if [[ ! -s "${eval_dir}/summary.json" ]]; then
    echo "[STAGE] canonical eval ${arm}"
    "${PYTHON_BIN}" scripts/evaluate_mixed4_depth_checkpoint.py \
      --config "${config}" \
      --dataset-root "${DATASET_ROOT}" \
      --checkpoint "${output}/last.pt" \
      --output "${eval_dir}/summary.json" \
      --per-sample-csv "${eval_dir}/per_sample.csv" \
      --camera-pair-csv "${eval_dir}/camera_pairs.csv" \
      --progress-file "${eval_dir}/progress.json" \
      --progress-every 25 \
      --train-loss-csv "${output}/loss.csv" \
      --datasets all \
      --limit-per-dataset "${EVAL_LIMIT_PER_DATASET}" \
      --sample-policy anchor \
      --pano-count-policy panovggt \
      --camera-eval-max-panos 3 \
      --window-size 384 \
      --num-yaw 4 \
      --pitch-degrees=-15 \
      --fov-degrees 75 \
      --device cuda \
      --num-workers 0 \
      --amp-dtype bfloat16 \
      --seed 123 \
      --resume \
      --no-print-each-sample
  fi

  if (( EVAL_LIMIT_PER_DATASET == 0 )); then
    "${PYTHON_BIN}" scripts/validate_eval_cardinality.py "${eval_dir}/summary.json"
  fi

  local preview_dir="${output}/preview_canonical_panocity_test0"
  if [[ ! -s "${preview_dir}/pred_z_depth_erp_splat.png" ]]; then
    echo "[STAGE] preview ${arm}"
    "${PYTHON_BIN}" scripts/reconstruct_pano_omega.py \
      --dataset-root "${DATASET_ROOT}" \
      --dataset-format pano_minimal \
      --minimal-datasets panocity \
      --dataset-split test \
      --sample-index 0 \
      --checkpoint "${output}/last.pt" \
      --output-dir "${preview_dir}" \
      --device cuda \
      --pano-height 512 \
      --pano-width 1024 \
      --window-size 384 \
      --num-yaw 4 \
      --pitch-degrees=-15 \
      --fov-degrees 75 \
      --fov-x-degrees 75 \
      --fov-y-degrees 75 \
      --fit-pred-depth-scale-from-target
  fi
}

run_training A_fov75x75 "${CONFIG_A}" "${OUTPUT_A}"
run_eval A_fov75x75 "${CONFIG_A}" "${OUTPUT_A}"

if [[ "${RUN_B_ARM}" == "1" ]]; then
  run_training B_fov95x75 "${CONFIG_B}" "${OUTPUT_B}"
  run_eval B_fov95x75 "${CONFIG_B}" "${OUTPUT_B}"
else
  echo "[SKIP] B_fov95x75 is opt-in (set RUN_B_ARM=1 to run it)."
fi

echo "[COMPLETE] M1 milestone training, canonical eval, and previews finished."
