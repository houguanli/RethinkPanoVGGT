#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-cmp_dap}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python - <<'PY'
import open3d
import torch
import torchvision
import utils3d

print("open3d", open3d.__version__)
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("utils3d", utils3d.__file__)
PY
