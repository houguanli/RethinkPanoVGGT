#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:-}"
if [[ -z "${METHOD}" ]]; then
  echo "Usage: $0 <method|all>"
  echo "Methods: panovggt panovggt_single panovggt_multi panovggt_camera panovggt_depth reloc3r bifusepp pi3 vggt_omega vggt_omega_single vggt_omega_multi vggt_omega_camera vggt_omega_depth dap panda all"
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

check_submodules() {
  if [[ -f "${ROOT}/.gitmodules" ]]; then
    echo "[git] initializing submodules"
    git -C "${ROOT}" submodule update --init --recursive
  else
    echo "[git] no git submodules declared"
  fi
}

post_install_check() {
  local method="$1"
  local method_dir="$2"
  echo "[env] validating imports for ${method}"
  case "${method}" in
    panovggt_camera|panovggt_depth)
      (cd "${method_dir}" && PYTHONPATH="${COMPARE_ROOT}:${method_dir}:${PYTHONPATH:-}" python - <<'PY'
import cv2  # noqa: F401
import hydra  # noqa: F401
import iopath  # noqa: F401
import omegaconf  # noqa: F401
import safetensors  # noqa: F401
import torch  # noqa: F401
import torchvision  # noqa: F401
import fvcore  # noqa: F401
import panovggt  # noqa: F401
from training.data.datasets.panocity_paired import PanoCityPairedDataset  # noqa: F401
print("ok panovggt imports")
PY
      )
      ;;
    reloc3r)
      (cd "${method_dir}" && PYTHONPATH="${COMPARE_ROOT}:${method_dir}:${PYTHONPATH:-}" python - <<'PY'
import cv2  # noqa: F401
import open3d  # noqa: F401
import PIL  # noqa: F401
import torch  # noqa: F401
import torchvision  # noqa: F401
import croco  # noqa: F401
import reloc3r  # noqa: F401
from reloc3r.datasets.panocity import PanoCityReloc3r  # noqa: F401
print("ok reloc3r imports")
PY
      )
      ;;
    bifusepp)
      (cd "${method_dir}" && python ../bifusepp_inference.py --check-only)
      ;;
    pi3)
      (cd "${method_dir}" && python ../pi3_inference.py --check-only)
      ;;
    vggt_omega_camera|vggt_omega_depth)
      (cd "${method_dir}" && PYTHONPATH="${COMPARE_ROOT}:${method_dir}:${PYTHONPATH:-}" python - <<'PY'
import cv2  # noqa: F401
import safetensors  # noqa: F401
import torch  # noqa: F401
import torchvision  # noqa: F401
import vggt_omega  # noqa: F401
from common.panocity_paired import PanoCityDepthTorchDataset  # noqa: F401
from panocity import PanoCityOmegaDataset  # noqa: F401
print("ok vggt_omega imports")
PY
      )
      ;;
    dap)
      (cd "${method_dir}" && PYTHONPATH="${COMPARE_ROOT}:${method_dir}:${PYTHONPATH:-}" python - <<'PY'
import cv2  # noqa: F401
import mmengine  # noqa: F401
import open3d  # noqa: F401
import pyexr  # noqa: F401
import safetensors  # noqa: F401
import tensorboardX  # noqa: F401
import torch  # noqa: F401
import torchvision  # noqa: F401
import networks.dap  # noqa: F401
from datasets.panocity import PanoCity  # noqa: F401
print("ok dap imports")
PY
      )
      ;;
    panda)
      (cd "${method_dir}" && PYTHONPATH="${COMPARE_ROOT}:${method_dir}:${PYTHONPATH:-}" python - <<'PY'
import cv2  # noqa: F401
import mmengine  # noqa: F401
import open3d  # noqa: F401
import safetensors  # noqa: F401
import tensorboardX  # noqa: F401
import torch  # noqa: F401
import torchvision  # noqa: F401
import networks.panda  # noqa: F401
from datasets.panocity import PanoCity  # noqa: F401
print("ok panda imports")
PY
      )
      ;;
  esac
}

create_env() {
  local method="$1"
  local env_name python_version method_dir check_method
  local -a reqs
  case "${method}" in
    panovggt_camera|panovggtcamera) check_method=panovggt_camera; env_name=cmp_panovggt; python_version=3.11; method_dir="${COMPARE_ROOT}/camera_pose/PanoVGGT"; reqs=("${method_dir}/requirements.txt");;
    panovggt_depth|panovggtdepth) check_method=panovggt_depth; env_name=cmp_panovggt; python_version=3.11; method_dir="${COMPARE_ROOT}/depth_geometry/PanoVGGT"; reqs=("${method_dir}/requirements.txt");;
    reloc3r|relo3r) check_method=reloc3r; env_name=cmp_reloc3r; python_version=3.11; method_dir="${COMPARE_ROOT}/camera_pose/Reloc3r"; reqs=("${method_dir}/requirements.txt" "${method_dir}/requirements_optional.txt");;
    bifusepp|bifuse++) check_method=bifusepp; env_name=cmp_bifusepp; python_version=3.11; method_dir="${COMPARE_ROOT}/camera_pose/BiFusePlusPlus"; reqs=("${COMPARE_ROOT}/environments/bifusepp_requirements.txt");;
    pi3) check_method=pi3; env_name=cmp_pi3; python_version=3.11; method_dir="${COMPARE_ROOT}/camera_pose/Pi3"; reqs=("${COMPARE_ROOT}/environments/pi3_requirements.txt");;
    vggt_omega_camera|vggtomegacamera) check_method=vggt_omega_camera; env_name=cmp_vggt_omega; python_version=3.10; method_dir="${COMPARE_ROOT}/camera_pose/VGGT-Omega"; reqs=("${method_dir}/requirements.txt");;
    vggt_omega_depth|vggtomegadepth) check_method=vggt_omega_depth; env_name=cmp_vggt_omega; python_version=3.10; method_dir="${COMPARE_ROOT}/depth_geometry/VGGT-Omega"; reqs=("${method_dir}/requirements.txt");;
    dap) check_method=dap; env_name=cmp_dap; python_version=3.12; method_dir="${COMPARE_ROOT}/depth_geometry/DAP"; reqs=("${method_dir}/requirements.txt");;
    panda) check_method=panda; env_name=cmp_panda; python_version=3.10; method_dir="${COMPARE_ROOT}/depth_geometry/PanDA"; reqs=("${method_dir}/requirements.txt");;
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

  if [[ "${check_method}" == "reloc3r" ]]; then
    conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
  fi

  for req in "${reqs[@]}"; do
    if [[ -s "${req}" ]]; then
      python -m pip install -r "${req}"
    fi
  done
  if [[ "${check_method}" == vggt_omega_* ]]; then
    python -m pip install -e "${method_dir}"
  fi
  if [[ "${check_method}" == "pi3" ]]; then
    python -m pip install -e "${method_dir}"
  fi
  post_install_check "${check_method}" "${method_dir}"
  python - <<'PY'
import sys
print("python", sys.version)
PY
  conda deactivate
}

check_submodules

if [[ "${METHOD}" == "all" ]]; then
  for m in panovggt_camera reloc3r bifusepp pi3 vggt_omega_camera dap panda; do
    create_env "${m}"
  done
elif [[ "${METHOD}" == "panovggt" || "${METHOD}" == "panovggt_single" || "${METHOD}" == "panovggtsingle" || "${METHOD}" == "panovggt_multi" || "${METHOD}" == "panovggtmulti" ]]; then
  create_env panovggt_camera
elif [[ "${METHOD}" == "vggt_omega" || "${METHOD}" == "vggtomega" || "${METHOD}" == "vggt_omega_single" || "${METHOD}" == "vggtomegasingle" || "${METHOD}" == "vggt_omega_multi" || "${METHOD}" == "vggtomegamulti" ]]; then
  create_env vggt_omega_camera
else
  create_env "${METHOD}"
fi
