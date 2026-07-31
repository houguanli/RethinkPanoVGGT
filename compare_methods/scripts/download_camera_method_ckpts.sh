#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:-all}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

download_bifusepp() {
  local target_dir bundle
  target_dir="${ROOT}/ckpt/BiFusePlusPlus"
  bundle="${target_dir}/pretrained_bundle.zip"
  mkdir -p "${target_dir}"
  if [[ ! -f "${target_dir}/pretrain/supervised_pretrain.pkl" || ! -f "${target_dir}/pretrain/selfsupervised_pretrain.pkl" ]]; then
    python -c 'import gdown' >/dev/null 2>&1 || {
      echo "gdown is required. Activate cmp_bifusepp or install compare_methods/environments/bifusepp_requirements.txt." >&2
      return 1
    }
    python -m gdown --fuzzy \
      'https://drive.google.com/file/d/1ZeQrCt4HQrZ3KGdROzqxWdqB4zz1EkTG/view?usp=sharing' \
      -O "${bundle}"
    python -m zipfile -e "${bundle}" "${target_dir}"
    rm -f "${bundle}"
  fi
  test "$(stat -c %s "${target_dir}/pretrain/supervised_pretrain.pkl")" -gt 200000000
  test "$(stat -c %s "${target_dir}/pretrain/selfsupervised_pretrain.pkl")" -gt 250000000
  echo "[checkpoint] BiFuse++: ${target_dir}/pretrain"
}

download_pi3() {
  local target_dir checkpoint
  target_dir="${ROOT}/ckpt/Pi3"
  checkpoint="${target_dir}/model.safetensors"
  mkdir -p "${target_dir}"
  if [[ ! -f "${checkpoint}" || "$(stat -c %s "${checkpoint}")" -ne 3834909248 ]]; then
    curl -L --fail --retry 5 --retry-delay 3 --continue-at - \
      -o "${checkpoint}" \
      'https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors'
  fi
  test "$(stat -c %s "${checkpoint}")" -eq 3834909248
  echo "[checkpoint] Pi3 (CC BY-NC 4.0 weights): ${checkpoint}"
}

case "${METHOD}" in
  bifusepp|bifuse++ ) download_bifusepp ;;
  pi3 ) download_pi3 ;;
  all ) download_bifusepp; download_pi3 ;;
  * ) echo "Usage: $0 [bifusepp|pi3|all]" >&2; exit 2 ;;
esac
