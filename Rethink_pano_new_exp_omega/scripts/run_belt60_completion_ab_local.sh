#!/usr/bin/env bash
# Crop-only pilot: matched updates, identical teacher/head initialization.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python}"
DATASET_ROOT="${PANOVGGT_ROOT:-/mnt/e/PanoVGGT_minimal_datasets/datasets}"
FOUNDATION_CHECKPOINT="${FOUNDATION_CHECKPOINT:-/home/aoki/RethinkPanoVGGT_omega_compare_methods_only/ckpt/VGGT-Omega/vggt_omega_1b_512.pt}"
OMEGA_CHECKPOINT="${OMEGA_CHECKPOINT:-${PROJECT_ROOT}/logs/full_erp_completion_v1_2h8h2h_20260824_omega_warmup_2h/last.pt}"
RUN_NAME="${RUN_NAME:-belt60_completion_ab_20260924}"
OUTPUT="${PROJECT_ROOT}/logs/${RUN_NAME}"
STEPS="${STEPS:-300}"
RUN_FULL_EVAL="${RUN_FULL_EVAL:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
mkdir -p "${OUTPUT}"
exec 9>"${OUTPUT}/pipeline.lock"
flock -n 9 || { echo "[ERROR] This run already has a live launcher."; exit 2; }
exec > >(tee -a "${OUTPUT}/pipeline.log") 2>&1
monitor_pid=""
cleanup() {
  local code=$?
  if [[ -n "${monitor_pid}" ]]; then kill "${monitor_pid}" 2>/dev/null || true; fi
  if (( code != 0 )); then echo "[FAILED] exit=${code}; preserving outputs; no automatic restart"; fi
}
trap cleanup EXIT
[[ ! -e "${OUTPUT}/COMPLETE" ]] || { echo "[COMPLETE] Already finished; no rerun."; exit 0; }
for required in "${FOUNDATION_CHECKPOINT}" "${OMEGA_CHECKPOINT}"; do
  test -s "${required}"
done
(
  while true; do
    date -u +%FT%TZ
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    free -b
    sleep 30
  done
) >> "${OUTPUT}/resources.log" 2>&1 &
monitor_pid=$!
BASE_CONFIG="configs/multipano_4090_mixed4_omega_canonical_warmup_2h.yaml"
BELT_CONFIG="configs/multipano_4090_mixed4_belt60_completion.yaml"

train_head() {
  local arm="$1" config="$2" cutoff="$3" steps="$4"
  local dest="${OUTPUT}/${arm}"
  mkdir -p "${dest}"
  if [[ -s "${dest}/last.pt" && -s "${dest}/status.json" ]]; then
    "${PYTHON_BIN}" -c 'import json,sys; s=json.load(open(sys.argv[1])); assert s["state"]=="completed" and s["step"]==int(sys.argv[2]),s' "${dest}/status.json" "${steps}"
    return
  fi
  [[ ! -s "${dest}/loss.csv" && ! -e "${dest}/interrupted.pt" ]] || {
    echo "[ERROR] Partial training exists in ${dest}; inspect before manual resume"; return 2;
  }
  echo "[STAGE] ${arm} config=${config} cutoff=${cutoff} steps=${steps}"
  "${PYTHON_BIN}" training/train_erp_completion.py \
    --config "${config}" --dataset-root "${DATASET_ROOT}" \
    --omega-checkpoint "${OMEGA_CHECKPOINT}" --base-checkpoint "${FOUNDATION_CHECKPOINT}" \
    --output-dir "${dest}" --duration-minutes 120 --max-steps "${steps}" \
    --core-latitude-degrees "${cutoff}" --head-width 32 --height 256 --width 512 \
    --stage main --lr 2e-4 --seed 57 --num-workers 0 --save-every 2000 \
    2>&1 | tee -a "${dest}/train.log"
  "${PYTHON_BIN}" -c 'import json,sys; s=json.load(open(sys.argv[1])); assert s["state"]=="completed" and s["step"]==int(sys.argv[2]),s' "${dest}/status.json" "${steps}"
}

evaluate() {
  local arm="$1" name="$2"
  shift 2
  local dest="${OUTPUT}/${arm}/${name}"
  mkdir -p "${dest}"
  [[ ! -s "${dest}/summary.json" ]] || return 0
  echo "[STAGE] ${arm}/${name}"
  "${PYTHON_BIN}" scripts/evaluate_mixed4_depth_checkpoint.py \
    --config "${BASE_CONFIG}" --dataset-root "${DATASET_ROOT}" \
    --checkpoint "${OMEGA_CHECKPOINT}" --erp-completion-checkpoint "${OUTPUT}/${arm}/last.pt" \
    --output "${dest}/summary.json" --per-sample-csv "${dest}/per_sample.csv" \
    --camera-pair-csv "${dest}/camera_pairs.csv" --progress-file "${dest}/progress.json" \
    --sample-policy anchor --camera-eval-max-panos 3 --device cuda --num-workers 0 \
    --amp-dtype bfloat16 --seed 123 --resume --no-print-each-sample "$@" \
    2>&1 | tee -a "${dest}/eval.log"
}

# Real, full-width training preflight before spending time on the comparison.
train_head preflight_belt60 "${BELT_CONFIG}" 60 1
train_head A_canonical "${BASE_CONFIG}" 90 "${STEPS}"
train_head B_belt60 "${BELT_CONFIG}" 60 "${STEPS}"
"${PYTHON_BIN}" scripts/analyze_full_erp_training_losses.py \
  --series "A=${OUTPUT}/A_canonical/loss.csv" --series "B=${OUTPUT}/B_belt60/loss.csv" \
  --output-dir "${OUTPUT}/loss_analysis"
for arm in A_canonical B_belt60; do
  # Diagnostic only: the same 20 unseen sets/dataset, TWO panos per set.
  evaluate "${arm}" diagnostic80_two_panos --datasets all --limit-per-dataset 20 \
    --pano-count-policy config --eval-max-panos 2
  dest="${OUTPUT}/${arm}/preview_panocity_test0"
  if [[ ! -s "${dest}/pred_range_depth_erp_completed.png" ]]; then
    "${PYTHON_BIN}" scripts/reconstruct_pano_omega.py \
      --dataset-root "${DATASET_ROOT}" --dataset-format pano_minimal --minimal-datasets panocity \
      --dataset-split test --sample-index 0 --checkpoint "${OMEGA_CHECKPOINT}" \
      --erp-completion-checkpoint "${OUTPUT}/${arm}/last.pt" --output-dir "${dest}" --device cuda
  fi
done
if [[ "${RUN_FULL_EVAL}" == 1 ]]; then
  # Resource gate: do NOT silently reduce the official Panocity 10-pano input.
  evaluate B_belt60 capacity_probe_10panos --datasets panocity --limit-per-dataset 1 --pano-count-policy panovggt
  for arm in A_canonical B_belt60; do
    evaluate "${arm}" full8833 --datasets all --limit-per-dataset 0 --pano-count-policy panovggt
    "${PYTHON_BIN}" scripts/validate_eval_cardinality.py "${OUTPUT}/${arm}/full8833/summary.json"
  done
fi
date -u +%FT%TZ > "${OUTPUT}/COMPLETE"
echo "[COMPLETE] ${OUTPUT}"
