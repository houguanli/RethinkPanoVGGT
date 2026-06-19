#!/usr/bin/env bash
set -euo pipefail

METHODS="${1:-panovggt,reloc3r,vggt_omega,dap,panda}"
STAGE="${STAGE:-both}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_UNSUPPORTED="${ALLOW_UNSUPPORTED:-0}"
WALLTIME_HOURS="${WALLTIME_HOURS:-}"
FINETUNE_TIMEOUT_SECONDS="${FINETUNE_TIMEOUT_SECONDS:-}"
MIN_WSL_C_FREE_GB="${MIN_WSL_C_FREE_GB:-20}"
ALLOW_LOW_WSL_C_SPACE="${ALLOW_LOW_WSL_C_SPACE:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPARE_ROOT="${ROOT}/compare_methods"

resolve_conda_sh() {
  local conda_bin conda_root candidate
  if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
    printf '%s\n' "${CONDA_SH}"
    return 0
  fi
  conda_bin="${CONDA_EXE:-}"
  if [[ -z "${conda_bin}" ]]; then
    conda_bin="$(command -v conda || true)"
  fi
  if [[ -n "${conda_bin}" ]]; then
    conda_bin="$(readlink -f "${conda_bin}")"
    conda_root="$(cd "$(dirname "${conda_bin}")/.." && pwd)"
    candidate="${conda_root}/etc/profile.d/conda.sh"
    if [[ ! -f "${candidate}" && "$(basename "${conda_root}")" == "condabin" ]]; then
      candidate="$(cd "${conda_root}/.." && pwd)/etc/profile.d/conda.sh"
    fi
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  fi
  for candidate in \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "${HOME}"/.conda/*/etc/profile.d/conda.sh; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  echo "Cannot find conda.sh. Set CONDA_SH=/path/to/etc/profile.d/conda.sh or put conda on PATH." >&2
  return 1
}

source "$(resolve_conda_sh)"

ensure_wsl_cuda_library_path() {
  if [[ ! -d /usr/lib/wsl/lib ]]; then
    return 0
  fi
  case ":${LD_LIBRARY_PATH:-}:" in
    *:/usr/lib/wsl/lib:*) ;;
    *) export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" ;;
  esac
}

ensure_wsl_cuda_library_path

check_wsl_host_space() {
  if [[ "${ALLOW_LOW_WSL_C_SPACE}" == "1" ]]; then
    return 0
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  if [[ ! -d /mnt/c ]]; then
    return 0
  fi
  if [[ "${STAGE}" != "deep_smoke" && "${STAGE}" != "finetune" && "${STAGE}" != "both" ]]; then
    return 0
  fi
  local avail_kb avail_gb
  avail_kb="$(df -Pk /mnt/c 2>/dev/null | awk 'NR==2 {print $4}')"
  if [[ -z "${avail_kb}" ]]; then
    return 0
  fi
  avail_gb="$((avail_kb / 1024 / 1024))"
  if (( avail_gb < MIN_WSL_C_FREE_GB )); then
    cat >&2 <<EOF
Refusing to run ${STAGE}: /mnt/c has only ${avail_gb}GB free.
WSL2 may need host C: space to grow/flush its virtual disk; low space can show up
as TensorBoard I/O errors, SIGBUS, or WSL CreateInstance/E_UNEXPECTED failures.
Free at least ${MIN_WSL_C_FREE_GB}GB on C:, or set ALLOW_LOW_WSL_C_SPACE=1 to override.
EOF
    exit 3
  fi
}

check_wsl_host_space

resolve_finetune_timeout_seconds() {
  if [[ -n "${FINETUNE_TIMEOUT_SECONDS}" ]]; then
    printf '%s\n' "${FINETUNE_TIMEOUT_SECONDS}"
    return 0
  fi
  if [[ -n "${WALLTIME_HOURS}" ]]; then
    awk -v hours="${WALLTIME_HOURS}" 'BEGIN {
      if (hours <= 0) {
        print "WALLTIME_HOURS must be positive" > "/dev/stderr"
        exit 2
      }
      printf "%d\n", hours * 3600
    }'
  fi
}

env_for_method() {
  case "$1" in
    panovggt|panovggt_camera|panovggt_depth) echo cmp_panovggt;;
    reloc3r) echo cmp_reloc3r;;
    vggt_omega|vggt_omega_camera|vggt_omega_depth) echo cmp_vggt_omega;;
    dap) echo cmp_dap;;
    panda) echo cmp_panda;;
    *) echo "unknown";;
  esac
}

IFS=',' read -ra METHOD_ARRAY <<< "${METHODS}"
finetune_timeout_seconds="$(resolve_finetune_timeout_seconds)"
for method in "${METHOD_ARRAY[@]}"; do
  env_name="$(env_for_method "${method}")"
  if [[ "${env_name}" == "unknown" ]]; then
    echo "Unknown method: ${method}" >&2
    exit 2
  fi
  echo "[pipeline] method=${method} env=${env_name} stage=${STAGE} finetune_timeout_seconds=${finetune_timeout_seconds:-none}"
  conda activate "${env_name}"
  args=(--method "${method}" --stage "${STAGE}")
  if [[ "${DRY_RUN}" == "1" ]]; then
    args+=(--dry-run)
  fi
  if [[ "${ALLOW_UNSUPPORTED}" == "1" ]]; then
    args+=(--allow-unsupported-finetune)
  fi
  if [[ -n "${finetune_timeout_seconds}" ]]; then
    args+=(--finetune-timeout-seconds "${finetune_timeout_seconds}")
  fi
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" python -m compare_methods.common.run_compare_method "${args[@]}"
  conda deactivate
done
