#!/usr/bin/env bash
set -euo pipefail

source /home/aoki/miniconda3/etc/profile.d/conda.sh

cd /home/aoki/RethinkPanoVGGT_omega/Rethink_pano_new_exp_omega

run_dir="${1:-outputs/pano_omega_luna_1h_$(date +%Y%m%d_%H%M%S)}"
pred_depth_scale="${2:-1.0}"
mkdir -p "$run_dir"

cat > "$run_dir/run_command.txt" <<EOF
conda run -n RethinkPanoVGGT_omega python training/launch.py \\
  --dataset-root ../dataset \\
  --checkpoint ../ckpt/vggt_omega_1b_512.pt \\
  --output-dir "$run_dir" \\
  --device cuda \\
  --epochs 999 \\
  --max-steps 100000 \\
  --max-duration-minutes 60 \\
  --batch-size 1 \\
  --num-workers 0 \\
  --pred-depth-scale "$pred_depth_scale" \\
  --save-last \\
  --log-csv "$run_dir/loss.csv" \\
  --loss-plot "$run_dir/loss_curve.png"
EOF

exec conda run -n RethinkPanoVGGT_omega python training/launch.py \
  --dataset-root ../dataset \
  --checkpoint ../ckpt/vggt_omega_1b_512.pt \
  --output-dir "$run_dir" \
  --device cuda \
  --epochs 999 \
  --max-steps 100000 \
  --max-duration-minutes 60 \
  --batch-size 1 \
  --num-workers 0 \
  --pred-depth-scale "$pred_depth_scale" \
  --save-last \
  --log-csv "$run_dir/loss.csv" \
  --loss-plot "$run_dir/loss_curve.png"
