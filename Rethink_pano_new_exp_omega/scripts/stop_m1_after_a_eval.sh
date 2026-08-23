#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUMMARY="${PROJECT_ROOT}/logs/m1_fovxy_core_ab_20260823_fov75x75/eval_full8833_anchor_canonical_pitch15_fov75x75/summary.json"
PYTHON_BIN="${PYTHON_BIN:-/home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python}"

while [[ ! -s "${SUMMARY}" ]]; do
  sleep 15
done
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/validate_eval_cardinality.py" "${SUMMARY}"

# Let the A evaluator close its CSV handles before stopping the old A/B wrapper.
while pgrep -af "evaluate_mixed4_depth_checkpoint.py.*m1_fovxy_core_ab_20260823_fov75x75" >/dev/null; do
  sleep 2
done

for cmdline in /proc/[0-9]*/cmdline; do
  [[ -r "${cmdline}" ]] || continue
  command_line="$(tr '\0' ' ' < "${cmdline}")"
  pid="${cmdline#/proc/}"
  pid="${pid%/cmdline}"
  if [[ "${command_line}" == *"run_m1_fovxy_core_ablation_local.sh"* ]]; then
    kill "${pid}" 2>/dev/null || true
  elif [[ "${command_line}" == *"evaluate_mixed4_depth_checkpoint.py"*"m1_fovxy_core_ab_20260823_fov95x75"* ]]; then
    kill "${pid}" 2>/dev/null || true
  fi
done
echo "[COMPLETE] A milestone preserved; B formal evaluation suppressed."
