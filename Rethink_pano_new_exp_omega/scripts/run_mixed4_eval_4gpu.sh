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
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
PROGRESS_INTERVAL_SECONDS="${PROGRESS_INTERVAL_SECONDS:-30}"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"

mkdir -p "$EVAL_OUT/shards" "$EVAL_OUT/progress"
rm -f "$EVAL_OUT"/progress/shard_*.json

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
  echo "[eval-4gpu] progress_interval_seconds=$PROGRESS_INTERVAL_SECONDS"
} | tee "$EVAL_OUT/eval_4gpu.log"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L 2>/dev/null | sed 's/^/[eval-4gpu][gpu] /' | tee -a "$EVAL_OUT/eval_4gpu.log" || true
fi

cd "$LUNA"
PIDS=()
trap 'echo "[eval-4gpu] interrupted; terminating shard processes" | tee -a "$EVAL_OUT/eval_4gpu.log"; for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done' INT TERM
for rank in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$rank]}"
  shard_json="$EVAL_OUT/shards/shard_${rank}.json"
  shard_csv="$EVAL_OUT/shards/shard_${rank}.csv"
  shard_log="$EVAL_OUT/shards/shard_${rank}.log"
  progress_file="$EVAL_OUT/progress/shard_${rank}.json"
  echo "[eval-4gpu] launching shard $rank/$NUM_SHARDS on gpu=$gpu" | tee -a "$EVAL_OUT/eval_4gpu.log"
  "$PYTHON" - "$progress_file" "$rank" "$NUM_SHARDS" "$gpu" <<'PY'
import json
import os
import sys
import time

path, rank, num_shards, gpu = sys.argv[1:5]
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "state": "launched",
            "shard_rank": int(rank),
            "num_shards": int(num_shards),
            "cuda_visible_devices": str(gpu),
            "processed_samples": 0,
            "updated_at": time.time(),
        },
        handle,
    )
PY
  (
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 PYTHONPATH="$LUNA${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" scripts/evaluate_mixed4_depth_checkpoint.py \
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
      --progress-file "$progress_file" \
      --progress-every "$PROGRESS_EVERY" \
      --no-progress
  ) >"$shard_log" 2>&1 &
  child_pid="$!"
  PIDS+=("$child_pid")
  "$PYTHON" - "$progress_file" "$child_pid" <<'PY'
import json
import sys
import time

path, pid = sys.argv[1:3]
try:
    payload = json.load(open(path, "r", encoding="utf-8"))
except Exception:
    payload = {}
payload["pid"] = int(pid)
payload["updated_at"] = time.time()
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False)
PY
  echo "[eval-4gpu] shard $rank pid=$child_pid log=$shard_log progress=$progress_file" | tee -a "$EVAL_OUT/eval_4gpu.log"
done

if [[ "$SHOW_PROGRESS" == "1" ]]; then
  while true; do
    alive=0
    for pid in "${PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        alive=1
        break
      fi
    done
    "$PYTHON" - "$EVAL_OUT/progress" "$NUM_SHARDS" <<'PY' | tee -a "$EVAL_OUT/eval_4gpu.log"
import datetime as _dt
import glob
import json
import os
import sys
import time

progress_dir = sys.argv[1]
num_shards = int(sys.argv[2])
now = time.time()
items = []
for rank in range(num_shards):
    path = os.path.join(progress_dir, f"shard_{rank}.json")
    if not os.path.exists(path):
        items.append(f"rank{rank}:missing")
        continue
    try:
        payload = json.load(open(path, "r", encoding="utf-8"))
    except Exception as exc:
        items.append(f"rank{rank}:bad_progress:{exc}")
        continue
    state = payload.get("state", "?")
    run = payload.get("run", "-")
    done = int(payload.get("processed_samples", 0) or 0)
    total = int(payload.get("shard_samples", 0) or 0)
    gpu = payload.get("cuda_visible_devices", "")
    pid = payload.get("pid", "")
    age = now - float(payload.get("updated_at", now) or now)
    loss = payload.get("last_loss")
    loss_text = f" loss={float(loss):.4g}" if isinstance(loss, (float, int)) else ""
    items.append(f"rank{rank}@gpu{gpu}:pid={pid} {state} {run} {done}/{total} age={age:.0f}s{loss_text}")
stamp = _dt.datetime.now().isoformat(timespec="seconds")
print(f"[eval-4gpu][progress] {stamp} " + " | ".join(items))
PY
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
        | sed 's/^/[eval-4gpu][gpu-proc] /' \
        | tee -a "$EVAL_OUT/eval_4gpu.log" || true
    fi
    [[ "$alive" -eq 0 ]] && break
    sleep "$PROGRESS_INTERVAL_SECONDS"
  done
fi

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
