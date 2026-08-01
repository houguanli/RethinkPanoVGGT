#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_ROOT="${1:-/mnt/e/PanoVGGT_minimal_datasets/datasets/PanoSUNCG_zeroshot}"
RESULTS_ROOT="${2:-${DATASET_ROOT}/eval_results/bifuse_pi3}"
METHODS="${METHODS:-bifusepp,pi3}"
TASKS="${TASKS:-depth,camera}"
DEPTH_LIMIT="${DEPTH_LIMIT:-0}"
CAMERA_LIMIT="${CAMERA_LIMIT:-0}"
FACE_SIZE="${FACE_SIZE:-196}"
RESUME="${RESUME:-1}"

CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "${CONDA_BIN}" ]]; then
  echo "[error] conda is not available on PATH" >&2
  exit 1
fi
CONDA_BASE="$(cd "$(dirname "${CONDA_BIN}")/.." && pwd)"
CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[error] conda.sh not found at ${CONDA_SH}" >&2
  exit 1
fi
source "${CONDA_SH}"

IFS=',' read -r -a method_list <<< "${METHODS}"
IFS=',' read -r -a task_list <<< "${TASKS}"
for method in "${method_list[@]}"; do
  case "${method}" in
    bifusepp) env_name="cmp_bifusepp" ;;
    pi3) env_name="cmp_pi3" ;;
    *) echo "[error] unknown method: ${method}" >&2; exit 2 ;;
  esac
  for task in "${task_list[@]}"; do
    extra_args=()
    if [[ "${task}" == "depth" ]]; then
      extra_args+=(--limit "${DEPTH_LIMIT}")
    elif [[ "${task}" == "camera" ]]; then
      extra_args+=(--max-trajectories "${CAMERA_LIMIT}")
    else
      echo "[error] unknown task: ${task}" >&2
      exit 2
    fi
    if [[ "${RESUME}" == "1" ]]; then
      extra_args+=(--resume)
    fi
    output_dir="${RESULTS_ROOT}/${method}/${task}"
    echo "[run] method=${method} task=${task} env=${env_name} output=${output_dir}"
    conda run --no-capture-output -n "${env_name}" \
      env PYTHONPATH="${REPO_ROOT}/compare_methods/camera_pose:${PYTHONPATH:-}" \
      python "${REPO_ROOT}/compare_methods/camera_pose/panosuncg_zeroshot.py" \
        --method "${method}" \
        --task "${task}" \
        --dataset-root "${DATASET_ROOT}" \
        --output-dir "${output_dir}" \
        --face-size "${FACE_SIZE}" \
        "${extra_args[@]}"
  done
done
