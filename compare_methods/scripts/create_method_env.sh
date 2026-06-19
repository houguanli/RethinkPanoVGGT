#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:-}"
if [[ -z "${METHOD}" ]]; then
  echo "Usage: $0 <method|all>"
  echo "Methods: panovggt_camera panovggt_depth reloc3r vggt_omega_camera vggt_omega_depth dap panda"
  exit 2
fi

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

create_env() {
  local method="$1"
  local env_name python_version method_dir
  local -a reqs
  case "${method}" in
    panovggt_camera) env_name=cmp_panovggt; python_version=3.11; method_dir="${COMPARE_ROOT}/camera_pose/PanoVGGT"; reqs=("${method_dir}/requirements.txt");;
    panovggt_depth) env_name=cmp_panovggt; python_version=3.11; method_dir="${COMPARE_ROOT}/depth_geometry/PanoVGGT"; reqs=("${method_dir}/requirements.txt");;
    reloc3r) env_name=cmp_reloc3r; python_version=3.11; method_dir="${COMPARE_ROOT}/camera_pose/Reloc3r"; reqs=("${method_dir}/requirements.txt" "${method_dir}/requirements_optional.txt");;
    vggt_omega_camera) env_name=cmp_vggt_omega; python_version=3.10; method_dir="${COMPARE_ROOT}/camera_pose/VGGT-Omega"; reqs=("${method_dir}/requirements.txt");;
    vggt_omega_depth) env_name=cmp_vggt_omega; python_version=3.10; method_dir="${COMPARE_ROOT}/depth_geometry/VGGT-Omega"; reqs=("${method_dir}/requirements.txt");;
    dap) env_name=cmp_dap; python_version=3.12; method_dir="${COMPARE_ROOT}/depth_geometry/DAP"; reqs=("${method_dir}/requirements.txt");;
    panda) env_name=cmp_panda; python_version=3.10; method_dir="${COMPARE_ROOT}/depth_geometry/PanDA"; reqs=("${method_dir}/requirements.txt");;
    *) echo "Unknown method: ${method}" >&2; return 2;;
  esac

  if conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
    echo "[env] ${env_name} already exists"
  else
    echo "[env] creating ${env_name} python=${python_version}"
    conda create -n "${env_name}" "python=${python_version}" -y
  fi

  conda activate "${env_name}"
  python -m pip install --upgrade pip

  if [[ "${method}" == "reloc3r" ]]; then
    conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
  fi

  for req in "${reqs[@]}"; do
    if [[ -s "${req}" ]]; then
      python -m pip install -r "${req}"
    fi
  done
  if [[ "${method}" == vggt_omega_* ]]; then
    python -m pip install -e "${method_dir}"
  fi
  python - <<'PY'
import sys
print("python", sys.version)
PY
  conda deactivate
}

if [[ "${METHOD}" == "all" ]]; then
  for m in panovggt_camera reloc3r vggt_omega_camera dap panda; do
    create_env "${m}"
  done
else
  create_env "${METHOD}"
fi
