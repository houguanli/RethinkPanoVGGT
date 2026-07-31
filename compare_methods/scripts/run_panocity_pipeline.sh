#!/usr/bin/env bash
set -euo pipefail

METHODS="${1:-panovggt,reloc3r,vggt_omega,dap,panda}"
STAGE="${STAGE:-both}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_UNSUPPORTED="${ALLOW_UNSUPPORTED:-0}"
WALLTIME_HOURS="${WALLTIME_HOURS:-12}"
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
    panovggt|panovggt_single|panovggtsingle|panovggt_multi|panovggtmulti|panovggtcamera|panovggt_camera|panovggtdepth|panovggt_depth|panovggt_multi_camera|panovggt_multi_depth) echo cmp_panovggt;;
    reloc3r|relo3r) echo cmp_reloc3r;;
    bifusepp|bifuse++) echo cmp_bifusepp;;
    pi3) echo cmp_pi3;;
    vggt_omega|vggtomega|vggt_omega_single|vggtomegasingle|vggt_omega_multi|vggtomegamulti|vggt_omega_camera|vggtomegacamera|vggt_omega_depth|vggtomegadepth|vggt_omega_multi_camera|vggt_omega_multi_depth) echo cmp_vggt_omega;;
    dap) echo cmp_dap;;
    panda) echo cmp_panda;;
    *) echo "unknown";;
  esac
}

resolve_installed_env() {
  local canonical="$1"
  local aliases=("${canonical}")
  case "${canonical}" in
    cmp_panovggt) aliases+=("cmppanovggt");;
    cmp_reloc3r) aliases+=("cmpreloc3r");;
    cmp_bifusepp) aliases+=("cmpbifusepp");;
    cmp_pi3) aliases+=("cmppi3");;
    cmp_vggt_omega) aliases+=("cmpvggtomega");;
    cmp_dap) aliases+=("cmpdap");;
    cmp_panda) aliases+=("cmppanda");;
  esac

  local env_name
  for env_name in "${aliases[@]}"; do
    if conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
      printf '%s\n' "${env_name}"
      return 0
    fi
  done
  echo "Cannot find conda env '${canonical}'. Tried: ${aliases[*]}" >&2
  return 1
}

asset_size() {
  local path="$1"
  if command -v stat >/dev/null 2>&1; then
    stat -c '%s bytes' "${path}" 2>/dev/null || wc -c < "${path}"
  else
    wc -c < "${path}"
  fi
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    return 1
  fi
  echo "[asset] ${label}: ${path} ($(asset_size "${path}"))"
}

require_method_relative_file() {
  local cwd="$1"
  local rel_path="$2"
  local label="$3"
  local resolved
  resolved="$(cd "${cwd}" && readlink -f "${rel_path}")"
  require_file "${resolved}" "${label}"
  echo "[asset] ${label} from cwd=${cwd}: ${rel_path} -> ${resolved}"
}

check_method_assets() {
  local method="$1"
  case "${method}" in
    panovggt|panovggtcamera|panovggt_camera)
      require_method_relative_file "${COMPARE_ROOT}/camera_pose/PanoVGGT" "../../../ckpt/PanoVGGT/model.pt" "PanoVGGT checkpoint"
      ;;
    panovggtdepth|panovggt_depth)
      require_method_relative_file "${COMPARE_ROOT}/depth_geometry/PanoVGGT" "../../../ckpt/PanoVGGT/model.pt" "PanoVGGT checkpoint"
      ;;
    reloc3r|relo3r)
      require_method_relative_file "${COMPARE_ROOT}/camera_pose/Reloc3r" "../../../ckpt/Reloc3r-512/Reloc3r-512.pth" "Reloc3r checkpoint"
      ;;
    bifusepp|bifuse++)
      require_method_relative_file "${COMPARE_ROOT}/camera_pose/BiFusePlusPlus" "../../../ckpt/BiFusePlusPlus/pretrain/supervised_pretrain.pkl" "BiFuse++ supervised checkpoint"
      ;;
    pi3)
      require_method_relative_file "${COMPARE_ROOT}/camera_pose/Pi3" "../../../ckpt/Pi3/model.safetensors" "Pi3 checkpoint"
      ;;
    vggt_omega|vggtomega|vggt_omega_camera|vggtomegacamera)
      require_method_relative_file "${COMPARE_ROOT}/camera_pose/VGGT-Omega" "../../../ckpt/VGGT-Omega/vggt_omega_1b_512.pt" "VGGT-Omega checkpoint"
      ;;
    vggt_omega_depth|vggtomegadepth)
      require_method_relative_file "${COMPARE_ROOT}/depth_geometry/VGGT-Omega" "../../../ckpt/VGGT-Omega/vggt_omega_1b_512.pt" "VGGT-Omega checkpoint"
      ;;
    dap)
      require_method_relative_file "${COMPARE_ROOT}/depth_geometry/DAP" "../../../ckpt/DAP/model.pth" "DAP checkpoint"
      ;;
    panda)
      require_method_relative_file "${COMPARE_ROOT}/depth_geometry/PanDA" "../../../ckpt/PanDA/panda_small.pth" "PanDA checkpoint"
      ;;
  esac
}

print_env_context() {
  local env_name="$1"
  conda run --no-capture-output -n "${env_name}" python - <<'PY'
import os
import sys

print(
    "[env] active={} prefix={} python={}".format(
        os.environ.get("CONDA_DEFAULT_ENV", ""),
        os.environ.get("CONDA_PREFIX", ""),
        sys.executable,
    ),
    flush=True,
)
PY
}

IFS=',' read -ra METHOD_ARRAY <<< "${METHODS}"
finetune_timeout_seconds="$(resolve_finetune_timeout_seconds)"
for method in "${METHOD_ARRAY[@]}"; do
  method="$(echo "${method}" | xargs)"
  canonical_env_name="$(env_for_method "${method}")"
  if [[ "${canonical_env_name}" == "unknown" ]]; then
    echo "Unknown method: ${method}" >&2
    exit 2
  fi
  env_name="$(resolve_installed_env "${canonical_env_name}")"
  if [[ "${STAGE}" != "smoke" ]]; then
    check_method_assets "${method}"
  fi
  echo "[pipeline] method=${method} env=${env_name} canonical_env=${canonical_env_name} stage=${STAGE} finetune_timeout_seconds=${finetune_timeout_seconds:-none}"
  print_env_context "${env_name}"
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
  conda run --no-capture-output -n "${env_name}" \
    env PYTHONPATH="${ROOT}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 \
    python -m compare_methods.common.run_compare_method "${args[@]}"
done
