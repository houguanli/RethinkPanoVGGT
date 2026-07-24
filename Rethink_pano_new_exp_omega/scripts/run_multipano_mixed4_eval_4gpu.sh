#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUNA="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-/home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python}"
GPUS="${GPUS:-0,1,2,3}"
CONFIG="${CONFIG:-configs/multipano_rtx5000x4_mixed4_pano_all384_luna_after_full_warmup_9h.yaml}"
DATASET_ROOT="${DATASET_ROOT:-}"
LUNA_OUT="${LUNA_OUT:-logs/mixed4_pano_all384_4xrtx5000_multipano_after_full_warmup_9h}"
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
PROGRESS_STYLE="${PROGRESS_STYLE:-bar}"
PROGRESS_BAR_WIDTH="${PROGRESS_BAR_WIDTH:-32}"
SHOW_GPU_PROC="${SHOW_GPU_PROC:-0}"
SAMPLE_POLICY="${SAMPLE_POLICY:-anchor}"
PANO_COUNT_POLICY="${PANO_COUNT_POLICY:-panovggt}"
DATASET_PANO_COUNTS="${DATASET_PANO_COUNTS:-}"
CAMERA_EVAL_MAX_PANOS="${CAMERA_EVAL_MAX_PANOS:-3}"
NUM_YAW="${NUM_YAW:-8}"
ERP_LATITUDE_LIMIT_DEG="${ERP_LATITUDE_LIMIT_DEG:-75}"

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
  echo "[eval-4gpu] dataset_root=${DATASET_ROOT:-<config>}"
  echo "[eval-4gpu] checkpoint=$EVAL_CHECKPOINT"
  echo "[eval-4gpu] train_loss_csv=$TRAIN_LOSS_CSV"
  echo "[eval-4gpu] eval_out=$EVAL_OUT"
  echo "[eval-4gpu] gpus=$GPUS num_shards=$NUM_SHARDS"
  echo "[eval-4gpu] datasets=$EVAL_DATASETS"
  echo "[eval-4gpu] limit_per_dataset=$LIMIT_PER_DATASET"
  echo "[eval-4gpu] sample_policy=$SAMPLE_POLICY"
  echo "[eval-4gpu] pano_count_policy=$PANO_COUNT_POLICY"
  echo "[eval-4gpu] camera_eval_max_panos=$CAMERA_EVAL_MAX_PANOS"
  echo "[eval-4gpu] num_yaw=$NUM_YAW"
  echo "[eval-4gpu] erp_latitude_limit_deg=$ERP_LATITUDE_LIMIT_DEG"
  echo "[eval-4gpu] progress_interval_seconds=$PROGRESS_INTERVAL_SECONDS"
  echo "[eval-4gpu] progress_style=$PROGRESS_STYLE"
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
now = time.time()
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "state": "launched",
            "shard_rank": int(rank),
            "num_shards": int(num_shards),
            "cuda_visible_devices": str(gpu),
            "processed_samples": 0,
            "started_at": now,
            "updated_at": now,
        },
        handle,
    )
PY
  (
    dataset_root_args=()
    if [[ -n "$DATASET_ROOT" ]]; then
      dataset_root_args=(--dataset-root "$DATASET_ROOT")
    fi
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 PYTHONPATH="$LUNA${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" scripts/evaluate_mixed4_depth_checkpoint.py \
      --config "$CONFIG" \
      "${dataset_root_args[@]}" \
      --checkpoint "$EVAL_CHECKPOINT" \
      --output "$shard_json" \
      --per-sample-csv "$shard_csv" \
      --train-loss-csv "$TRAIN_LOSS_CSV" \
      --datasets "$EVAL_DATASETS" \
      --limit-per-dataset "$LIMIT_PER_DATASET" \
      --sample-policy "$SAMPLE_POLICY" \
      --pano-count-policy "$PANO_COUNT_POLICY" \
      --dataset-pano-counts "$DATASET_PANO_COUNTS" \
      --camera-eval-max-panos "$CAMERA_EVAL_MAX_PANOS" \
      --num-yaw "$NUM_YAW" \
      --erp-latitude-limit-deg "$ERP_LATITUDE_LIMIT_DEG" \
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
  PROGRESS_PRINTED=0
  while true; do
    alive=0
    for pid in "${PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        alive=1
        break
      fi
    done
    progress_line="$("$PYTHON" - "$EVAL_OUT/progress" "$NUM_SHARDS" "$PROGRESS_BAR_WIDTH" "$EVAL_OUT/eval_4gpu.log" <<'PY'
import datetime as _dt
import json
import os
import sys
import time

progress_dir = sys.argv[1]
num_shards = int(sys.argv[2])
bar_width = int(sys.argv[3])
log_path = sys.argv[4]
now = time.time()
items = []
processed_total = 0
sample_total = 0
active_runs = []
started_at_values = []
last_losses = []
states = []
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
    processed_total += done
    sample_total += total
    if run not in ("", "-"):
        active_runs.append(str(run))
    if isinstance(loss, (float, int)):
        last_losses.append(float(loss))
    states.append(str(state))
    started_at = payload.get("started_at")
    if isinstance(started_at, (float, int)):
        started_at_values.append(float(started_at))
stamp = _dt.datetime.now().isoformat(timespec="seconds")
detail = f"[eval-4gpu][progress] {stamp} " + " | ".join(items)
with open(log_path, "a", encoding="utf-8") as handle:
    handle.write(detail + "\n")
run_label = "-"
if active_runs:
    unique_runs = sorted(set(active_runs))
    run_label = unique_runs[0] if len(unique_runs) == 1 else "mixed:" + ",".join(unique_runs[:3])
ratio = (processed_total / sample_total) if sample_total > 0 else 0.0
filled = min(bar_width, max(0, int(round(ratio * bar_width))))
bar = "#" * filled + " " * (bar_width - filled)
percent = ratio * 100.0
elapsed = now - min(started_at_values) if started_at_values else 0.0
rate = processed_total / elapsed if elapsed > 0 and processed_total > 0 else 0.0
remaining = (sample_total - processed_total) / rate if rate > 0 and sample_total > processed_total else 0.0
eta_text = f"{remaining/60:.1f}m" if remaining >= 60 else f"{remaining:.0f}s"
loss_text = f", loss={last_losses[-1]:.4g}" if last_losses else ""
state_text = "done" if states and all(state == "done" for state in states) else ("init" if not sample_total else "running")
print(
    f"Eval {run_label}: {percent:5.1f}%|{bar}| {processed_total}/{sample_total} "
    f"[{elapsed/60:.1f}m<{eta_text}, {rate:.2f} samples/s, state={state_text}{loss_text}]"
)
PY
)"
    if [[ -t 1 && "$PROGRESS_STYLE" == "bar" ]]; then
      printf '\r\033[K%s' "$progress_line"
      PROGRESS_PRINTED=1
    else
      echo "$progress_line"
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
      gpu_proc_text="$(nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
      if [[ -n "$gpu_proc_text" ]]; then
        while IFS= read -r line; do
          [[ -n "$line" ]] || continue
          echo "[eval-4gpu][gpu-proc] $line" >> "$EVAL_OUT/eval_4gpu.log"
          if [[ "$SHOW_GPU_PROC" == "1" ]]; then
            if [[ "$PROGRESS_PRINTED" == "1" ]]; then
              printf '\n'
              PROGRESS_PRINTED=0
            fi
            echo "[eval-4gpu][gpu-proc] $line"
          fi
        done <<< "$gpu_proc_text"
      fi
    fi
    [[ "$alive" -eq 0 ]] && break
    sleep "$PROGRESS_INTERVAL_SECONDS"
  done
  if [[ "$PROGRESS_PRINTED" == "1" ]]; then
    printf '\n'
  fi
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
MERGE_CONSOLE_LOG="$EVAL_OUT/validation_mixed4_by_dataset_valtestfull_console.log"
SHARD_JSON_FILES=("$EVAL_OUT"/shards/shard_[0-9]*.json)
SHARD_CSV_FILES=()
for shard_csv in "$EVAL_OUT"/shards/shard_[0-9]*.csv; do
  [[ "$shard_csv" == *_camera_pairs.csv ]] && continue
  SHARD_CSV_FILES+=("$shard_csv")
done
SHARD_CAMERA_CSV_FILES=("$EVAL_OUT"/shards/shard_[0-9]*_camera_pairs.csv)
merge_status=0
PYTHONPATH="$LUNA${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" scripts/merge_mixed4_eval_shards.py \
  --shard-json "${SHARD_JSON_FILES[@]}" \
  --shard-csv "${SHARD_CSV_FILES[@]}" \
  --shard-camera-csv "${SHARD_CAMERA_CSV_FILES[@]}" \
  --output "$EVAL_OUT/validation_mixed4_by_dataset_valtestfull_summary.json" \
  --per-sample-csv "$EVAL_OUT/validation_mixed4_by_dataset_valtestfull_per_sample.csv" \
  --camera-pair-csv "$EVAL_OUT/validation_mixed4_by_dataset_valtestfull_camera_pairs.csv" \
  --train-loss-csv "$TRAIN_LOSS_CSV" \
  > "$MERGE_CONSOLE_LOG" 2>&1 || merge_status=$?
if [[ "$merge_status" -ne 0 ]]; then
  echo "[eval-4gpu] shard merge failed with status=$merge_status; tail of $MERGE_CONSOLE_LOG:" | tee -a "$EVAL_OUT/eval_4gpu.log"
  tail -n 100 "$MERGE_CONSOLE_LOG" | sed 's/^/[eval-4gpu][merge] /' | tee -a "$EVAL_OUT/eval_4gpu.log"
  exit "$merge_status"
fi

SUMMARY_JSON="$EVAL_OUT/validation_mixed4_by_dataset_valtestfull_summary.json"
PER_SAMPLE_CSV="$EVAL_OUT/validation_mixed4_by_dataset_valtestfull_per_sample.csv"
if [[ ! -s "$SUMMARY_JSON" ]]; then
  echo "[eval-4gpu] missing merged summary: $SUMMARY_JSON" | tee -a "$EVAL_OUT/eval_4gpu.log"
  exit 1
fi
if [[ ! -s "$PER_SAMPLE_CSV" ]]; then
  echo "[eval-4gpu] missing merged per-sample CSV: $PER_SAMPLE_CSV" | tee -a "$EVAL_OUT/eval_4gpu.log"
  exit 1
fi

echo "[eval-4gpu] finished $(date --iso-8601=seconds)" | tee -a "$EVAL_OUT/eval_4gpu.log"
