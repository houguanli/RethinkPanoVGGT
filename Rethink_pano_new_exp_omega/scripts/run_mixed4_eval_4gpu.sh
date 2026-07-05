#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUNA="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-python}"
GPUS="${GPUS:-0,1,2,3}"
CONFIG="${CONFIG:-configs/single_pano_rtx5000x4_mixed4_pano_low_to_high_luna_after_full_warmup_9h.yaml}"
LUNA_OUT="${LUNA_OUT:-logs/mixed4_pano_low_to_high_4xrtx5000_luna_after_full_warmup_9h}"
CHECKPOINT="${CHECKPOINT:-$LUNA_OUT/last.pt}"
TRAIN_LOSS_CSV="${TRAIN_LOSS_CSV:-$LUNA_OUT/loss.csv}"
EVAL_OUT="${EVAL_OUT:-$LUNA_OUT/eval_full_4gpu}"
LIMIT_PER_DATASET="${LIMIT_PER_DATASET:-0}"
EVAL_DATASETS="${EVAL_DATASETS:-all}"
NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-2}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
SEED="${SEED:-123}"

mkdir -p "$EVAL_OUT/shards"

EVAL_CHECKPOINT="$CHECKPOINT"
if [[ -n "${BASE_CHECKPOINT_OVERRIDE:-}" ]]; then
  EVAL_CHECKPOINT="$EVAL_OUT/checkpoint_with_eval_base.pt"
  "$PYTHON" - "$CHECKPOINT" "$BASE_CHECKPOINT_OVERRIDE" "$EVAL_CHECKPOINT" <<'PY'
import sys
import torch

source, base, target = sys.argv[1:4]
payload = torch.load(source, map_location="cpu", weights_only=False)
if isinstance(payload, dict):
    payload = dict(payload)
    payload["base_checkpoint"] = base
    ckpt_args = payload.get("args")
    if isinstance(ckpt_args, dict):
        ckpt_args = dict(ckpt_args)
        ckpt_args["checkpoint"] = base
        payload["args"] = ckpt_args
torch.save(payload, target)
print(f"[eval-4gpu] wrote checkpoint with overridden base: {target}")
PY
fi

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
NUM_SHARDS="${#GPU_LIST[@]}"
if [[ "$NUM_SHARDS" -lt 1 ]]; then
  echo "[eval-4gpu] GPUS is empty" >&2
  exit 1
fi

{
  echo "[eval-4gpu] started $(date --iso-8601=seconds)"
  echo "[eval-4gpu] luna=$LUNA"
  echo "[eval-4gpu] config=$CONFIG"
  echo "[eval-4gpu] checkpoint=$EVAL_CHECKPOINT"
  echo "[eval-4gpu] train_loss_csv=$TRAIN_LOSS_CSV"
  echo "[eval-4gpu] eval_out=$EVAL_OUT"
  echo "[eval-4gpu] gpus=$GPUS num_shards=$NUM_SHARDS"
  echo "[eval-4gpu] datasets=$EVAL_DATASETS"
  echo "[eval-4gpu] limit_per_dataset=$LIMIT_PER_DATASET"
} | tee "$EVAL_OUT/eval_4gpu.log"

cd "$LUNA"
PIDS=()
for rank in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$rank]}"
  shard_json="$EVAL_OUT/shards/shard_${rank}.json"
  shard_csv="$EVAL_OUT/shards/shard_${rank}.csv"
  shard_log="$EVAL_OUT/shards/shard_${rank}.log"
  echo "[eval-4gpu] launching shard $rank/$NUM_SHARDS on gpu=$gpu" | tee -a "$EVAL_OUT/eval_4gpu.log"
  (
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$LUNA${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" scripts/evaluate_mixed4_depth_checkpoint.py \
      --config "$CONFIG" \
      --checkpoint "$EVAL_CHECKPOINT" \
      --output "$shard_json" \
      --per-sample-csv "$shard_csv" \
      --train-loss-csv "$TRAIN_LOSS_CSV" \
      --datasets "$EVAL_DATASETS" \
      --limit-per-dataset "$LIMIT_PER_DATASET" \
      --device cuda \
      --num-workers "$NUM_WORKERS_PER_GPU" \
      --amp-dtype "$AMP_DTYPE" \
      --seed "$SEED" \
      --num-shards "$NUM_SHARDS" \
      --shard-rank "$rank" \
      --no-progress
  ) >"$shard_log" 2>&1 &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "[eval-4gpu] at least one shard failed; inspect $EVAL_OUT/shards/shard_*.log" | tee -a "$EVAL_OUT/eval_4gpu.log"
  exit "$status"
fi

echo "[eval-4gpu] merging shards $(date --iso-8601=seconds)" | tee -a "$EVAL_OUT/eval_4gpu.log"
PYTHONPATH="$LUNA${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" scripts/merge_mixed4_eval_shards.py \
  --shard-json "$EVAL_OUT"/shards/shard_*.json \
  --shard-csv "$EVAL_OUT"/shards/shard_*.csv \
  --output "$EVAL_OUT/validation_mixed4_by_dataset_valtestfull_summary.json" \
  --per-sample-csv "$EVAL_OUT/validation_mixed4_by_dataset_valtestfull_per_sample.csv" \
  --train-loss-csv "$TRAIN_LOSS_CSV" \
  > "$EVAL_OUT/validation_mixed4_by_dataset_valtestfull_console.log" 2>&1

echo "[eval-4gpu] finished $(date --iso-8601=seconds)" | tee -a "$EVAL_OUT/eval_4gpu.log"
