#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
cd "$project_root"

run_dir="${1:-outputs/pano_omega_luna_1h_20260524}"
pred_depth_scale="${2:-1.0}"
mkdir -p "$run_dir"

nohup scripts/run_luna_1h_train.sh "$run_dir" "$pred_depth_scale" > "$run_dir/train_stdout.log" 2>&1 &
pid="$!"
echo "$pid" > "$run_dir/train.pid"
echo "$run_dir"
echo "$pid"
