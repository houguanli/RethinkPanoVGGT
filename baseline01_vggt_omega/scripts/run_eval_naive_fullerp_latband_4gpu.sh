#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "$#" -ne 1 || ! -s "$1" ]]; then
  echo "usage: bash scripts/run_eval_naive_fullerp_latband_4gpu.sh /absolute/path/to/checkpoint.pt" >&2
  exit 2
fi
CHECKPOINT="$(readlink -f "$1")"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
elif [[ -x /home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python ]]; then
  PYTHON_BIN=/home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python
else
  echo "[latband-eval] Python not found; activate the vggt-omega conda environment" >&2
  exit 2
fi

DATASET_ROOT=""
for candidate in /whitehole/AOKI/panovggt /mnt/e/PanoVGGT_minimal_datasets/datasets; do
  if [[ -d "$candidate" ]]; then
    DATASET_ROOT="$candidate"
    break
  fi
done
if [[ -z "$DATASET_ROOT" ]]; then
  echo "[latband-eval] mixed4 dataset root not found" >&2
  exit 2
fi

if [[ -z "${VGGT_OMEGA_CKPT:-}" ]]; then
  for candidate in \
    /whitehole/AOKI/vggt-omega/ckpt/vggt_omega_1b_512.pt \
    /whitehole/AOKI/RethinkPanoVGGT_omega/ckpt/vggt_omega_1b_512.pt \
    /home/aoki/RethinkPanoVGGT_omega_compare_methods_only/ckpt/VGGT-Omega/vggt_omega_1b_512.pt \
    "$ROOT/ckpt/vggt_omega_1b_512.pt"; do
    if [[ -s "$candidate" ]]; then
      VGGT_OMEGA_CKPT="$candidate"
      break
    fi
  done
fi
if [[ -z "${VGGT_OMEGA_CKPT:-}" || ! -s "$VGGT_OMEGA_CKPT" ]]; then
  echo "[latband-eval] base VGGT-Omega checkpoint not found" >&2
  exit 2
fi
export VGGT_OMEGA_CKPT

CONFIG=mixed4_naive_fullerp_vggtomega_scalealigned_2pano
GPU_LIST=(0 1 2 3)
NUM_SHARDS=${#GPU_LIST[@]}
IMAGE_WIDTH=1024
LATITUDE_MIN=-10
LATITUDE_MAX=75
CHECKPOINT_DIR="$(dirname "$CHECKPOINT")"
if [[ "$(basename "$CHECKPOINT_DIR")" == "ckpts" ]]; then
  STAGE_OUT="$(dirname "$CHECKPOINT_DIR")"
else
  STAGE_OUT="$CHECKPOINT_DIR"
fi
OUT="$STAGE_OUT/eval_naive_fullerp_panovggt_counts_1024_lat_m10_p75_4gpu"
mkdir -p "$OUT/shards"
rm -f \
  "$OUT"/shards/shard_*.json \
  "$OUT"/shards/shard_*.csv \
  "$OUT"/shards/shard_*_camera_pairs.csv \
  "$OUT"/shards/shard_*.log \
  "$OUT/per_sample.csv" \
  "$OUT/camera_pairs.csv" \
  "$OUT/validation_naive_fullerp_lat_m10_p75_summary.json"

echo "[latband-eval] checkpoint=$CHECKPOINT"
echo "[latband-eval] config=$CONFIG"
echo "[latband-eval] dataset_root=$DATASET_ROOT"
echo "[latband-eval] gpus=0,1,2,3"
echo "[latband-eval] representation=full_erp_no_split image=1024x512"
echo "[latband-eval] pano_counts=panocity:10,matterport3d:3,stanford2d3ds:3,structured3d:3"
echo "[latband-eval] latitude_band=[-10,+75] degrees; IRLS scale is fitted inside this mask"
echo "[latband-eval] output=$OUT"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

for rank in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$rank]}"
  shard_prefix="$OUT/shards/shard_${rank}"
  echo "[latband-eval] launching shard=$rank/$NUM_SHARDS gpu=$gpu"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" scripts/evaluate_mixed4_depth_checkpoint.py \
      --config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --dataset-root "$DATASET_ROOT" \
      --output "${shard_prefix}.json" \
      --per-sample-csv "${shard_prefix}.csv" \
      --camera-pair-csv "${shard_prefix}_camera_pairs.csv" \
      --datasets all \
      --limit-per-dataset 0 \
      --sample-policy anchor \
      --pano-count-policy panovggt \
      --img-size "$IMAGE_WIDTH" \
      --camera-eval-max-panos 0 \
      --latitude-min-deg "$LATITUDE_MIN" \
      --latitude-max-deg "$LATITUDE_MAX" \
      --num-shards "$NUM_SHARDS" \
      --shard-rank "$rank" \
      --device cuda \
      --amp-dtype bfloat16 \
      --fail-fast \
      >"${shard_prefix}.log" 2>&1 &
  PIDS+=("$!")
done

while true; do
  alive=0
  rows=0
  for rank in "${!GPU_LIST[@]}"; do
    pid="${PIDS[$rank]}"
    if kill -0 "$pid" 2>/dev/null; then
      alive=1
    fi
    csv_path="$OUT/shards/shard_${rank}.csv"
    if [[ -s "$csv_path" ]]; then
      count=$(($(wc -l < "$csv_path") - 1))
      (( count > 0 )) && rows=$((rows + count))
    fi
  done
  echo "[latband-eval] streamed_samples=$rows active=$alive"
  [[ "$alive" -eq 1 ]] || break
  sleep 30
done

failed=0
for rank in "${!GPU_LIST[@]}"; do
  if ! wait "${PIDS[$rank]}"; then
    echo "[latband-eval] shard $rank failed; inspect $OUT/shards/shard_${rank}.log" >&2
    tail -n 60 "$OUT/shards/shard_${rank}.log" >&2 || true
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

"$PYTHON_BIN" scripts/merge_naive_fullerp_eval_shards.py \
  --shard-json "$OUT"/shards/shard_*.json \
  --shard-csv "$OUT"/shards/shard_?.csv \
  --shard-camera-csv "$OUT"/shards/shard_?_camera_pairs.csv \
  --output "$OUT/validation_naive_fullerp_lat_m10_p75_summary.json" \
  --per-sample-csv "$OUT/per_sample.csv" \
  --camera-pair-csv "$OUT/camera_pairs.csv" \
  >"$OUT/merge.log" 2>&1

echo "[latband-eval] complete"
echo "[latband-eval] summary=$OUT/validation_naive_fullerp_lat_m10_p75_summary.json"
