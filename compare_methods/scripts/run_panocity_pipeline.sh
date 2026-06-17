#!/usr/bin/env bash
set -euo pipefail

METHODS="${1:-panovggt_camera,reloc3r,vggt_omega_camera,dap,panda}"
STAGE="${STAGE:-both}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_UNSUPPORTED="${ALLOW_UNSUPPORTED:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPARE_ROOT="${ROOT}/compare_methods"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"

env_for_method() {
  case "$1" in
    panovggt_camera|panovggt_depth) echo cmp_panovggt;;
    reloc3r) echo cmp_reloc3r;;
    vggt_omega_camera|vggt_omega_depth) echo cmp_vggt_omega;;
    dap) echo cmp_dap;;
    panda) echo cmp_panda;;
    *) echo "unknown";;
  esac
}

IFS=',' read -ra METHOD_ARRAY <<< "${METHODS}"
for method in "${METHOD_ARRAY[@]}"; do
  env_name="$(env_for_method "${method}")"
  if [[ "${env_name}" == "unknown" ]]; then
    echo "Unknown method: ${method}" >&2
    exit 2
  fi
  echo "[pipeline] method=${method} env=${env_name} stage=${STAGE}"
  conda activate "${env_name}"
  args=(--method "${method}" --stage "${STAGE}")
  if [[ "${DRY_RUN}" == "1" ]]; then
    args+=(--dry-run)
  fi
  if [[ "${ALLOW_UNSUPPORTED}" == "1" ]]; then
    args+=(--allow-unsupported-finetune)
  fi
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" python -m compare_methods.common.run_compare_method "${args[@]}"
  conda deactivate
done
