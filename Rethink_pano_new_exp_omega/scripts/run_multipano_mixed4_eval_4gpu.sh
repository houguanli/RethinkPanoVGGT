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
ERP_COMPLETION_CHECKPOINT="${ERP_COMPLETION_CHECKPOINT:-}"
TRAIN_LOSS_CSV="${TRAIN_LOSS_CSV:-$LUNA_OUT/loss.csv}"
EVAL_OUT="${EVAL_OUT:-$LUNA_OUT/eval_full_4gpu}"
LIMIT_PER_DATASET="${LIMIT_PER_DATASET:-0}"
EVAL_DATASETS="${EVAL_DATASETS:-all}"
NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-2}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
SEED="${SEED:-123}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1}"
PROGRESS_INTERVAL_SECONDS="${PROGRESS_INTERVAL_SECONDS:-30}"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"
PROGRESS_STYLE="${PROGRESS_STYLE:-bar}"
PROGRESS_BAR_WIDTH="${PROGRESS_BAR_WIDTH:-32}"
SHOW_GPU_PROC="${SHOW_GPU_PROC:-0}"
SAMPLE_POLICY="${SAMPLE_POLICY:-anchor}"
PANO_COUNT_POLICY="${PANO_COUNT_POLICY:-panovggt}"
DATASET_PANO_COUNTS="${DATASET_PANO_COUNTS:-}"
CAMERA_EVAL_MAX_PANOS="${CAMERA_EVAL_MAX_PANOS:-3}"
WINDOW_SIZE="${WINDOW_SIZE:-384}"
NUM_YAW="${NUM_YAW:-4}"
PRINT_EACH_SAMPLE="${PRINT_EACH_SAMPLE:-1}"
RESUME="${RESUME:-1}"

if [[ "$LIMIT_PER_DATASET" == "0" && "$EVAL_DATASETS" == "all" ]]; then
  if [[ "$SAMPLE_POLICY" != "anchor" ]]; then
    echo "[eval-4gpu] formal 8,833-set eval requires SAMPLE_POLICY=anchor" >&2
    exit 2
  fi
  if [[ "$PANO_COUNT_POLICY" != "panovggt" || -n "$DATASET_PANO_COUNTS" ]]; then
    echo "[eval-4gpu] formal 8,833-set eval requires PANO_COUNT_POLICY=panovggt and no DATASET_PANO_COUNTS override" >&2
    exit 2
  fi
fi

mkdir -p "$EVAL_OUT/shards" "$EVAL_OUT/progress" "$EVAL_OUT/dataset_summaries"
rm -f "$EVAL_OUT"/progress/shard_*.json
rm -f "$EVAL_OUT"/shards/shard_*_stanford2d3ds.json \
  "$EVAL_OUT"/shards/shard_*_stanford2d3ds.csv \
  "$EVAL_OUT"/shards/shard_*_stanford2d3ds_camera_pairs.csv \
  "$EVAL_OUT"/shards/shard_*_matterport3d.json \
  "$EVAL_OUT"/shards/shard_*_matterport3d.csv \
  "$EVAL_OUT"/shards/shard_*_matterport3d_camera_pairs.csv \
  "$EVAL_OUT"/shards/shard_*_structured3d.json \
  "$EVAL_OUT"/shards/shard_*_structured3d.csv \
  "$EVAL_OUT"/shards/shard_*_structured3d_camera_pairs.csv \
  "$EVAL_OUT"/shards/shard_*_panocity.json \
  "$EVAL_OUT"/shards/shard_*_panocity.csv \
  "$EVAL_OUT"/shards/shard_*_panocity_camera_pairs.csv
rm -f "$EVAL_OUT"/dataset_summaries/*_summary.json

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
  echo "[eval-4gpu] erp_completion_checkpoint=${ERP_COMPLETION_CHECKPOINT:-<disabled>}"
  echo "[eval-4gpu] train_loss_csv=$TRAIN_LOSS_CSV"
  echo "[eval-4gpu] eval_out=$EVAL_OUT"
  echo "[eval-4gpu] gpus=$GPUS num_shards=$NUM_SHARDS"
  echo "[eval-4gpu] datasets=$EVAL_DATASETS"
  echo "[eval-4gpu] limit_per_dataset=$LIMIT_PER_DATASET"
  echo "[eval-4gpu] sample_policy=$SAMPLE_POLICY"
  echo "[eval-4gpu] pano_count_policy=$PANO_COUNT_POLICY"
  echo "[eval-4gpu] camera_eval_max_panos=$CAMERA_EVAL_MAX_PANOS"
  echo "[eval-4gpu] window_size=$WINDOW_SIZE"
  echo "[eval-4gpu] num_yaw=$NUM_YAW"
  echo "[eval-4gpu] print_each_sample=$PRINT_EACH_SAMPLE"
  echo "[eval-4gpu] resume=$RESUME"
  echo "[eval-4gpu] progress_interval_seconds=$PROGRESS_INTERVAL_SECONDS"
  echo "[eval-4gpu] progress_style=$PROGRESS_STYLE"
} | tee "$EVAL_OUT/eval_4gpu.log"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L 2>/dev/null | sed 's/^/[eval-4gpu][gpu] /' | tee -a "$EVAL_OUT/eval_4gpu.log" || true
fi

cd "$LUNA"
PIDS=()
sample_output_args=(--print-each-sample)
if [[ "$PRINT_EACH_SAMPLE" != "1" ]]; then
  sample_output_args=(--no-print-each-sample)
fi
resume_args=()
if [[ "$RESUME" == "1" ]]; then
  resume_args=(--resume)
fi
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
    completion_args=()
    if [[ -n "$ERP_COMPLETION_CHECKPOINT" ]]; then
      completion_args=(--erp-completion-checkpoint "$ERP_COMPLETION_CHECKPOINT")
    fi
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 PYTHONPATH="$LUNA${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" scripts/evaluate_mixed4_depth_checkpoint.py \
      --config "$CONFIG" \
      "${dataset_root_args[@]}" \
      --checkpoint "$EVAL_CHECKPOINT" \
      "${completion_args[@]}" \
      --output "$shard_json" \
      --per-sample-csv "$shard_csv" \
      --train-loss-csv "$TRAIN_LOSS_CSV" \
      --datasets "$EVAL_DATASETS" \
      --limit-per-dataset "$LIMIT_PER_DATASET" \
      --sample-policy "$SAMPLE_POLICY" \
      --pano-count-policy "$PANO_COUNT_POLICY" \
      --dataset-pano-counts "$DATASET_PANO_COUNTS" \
      --camera-eval-max-panos "$CAMERA_EVAL_MAX_PANOS" \
      --window-size "$WINDOW_SIZE" \
      --num-yaw "$NUM_YAW" \
      --device cuda \
      --num-workers "$NUM_WORKERS_PER_GPU" \
      --amp-dtype "$AMP_DTYPE" \
      --seed "$SEED" \
      --num-shards "$NUM_SHARDS" \
      --shard-rank "$rank" \
      --progress-file "$progress_file" \
      --progress-every "$PROGRESS_EVERY" \
      "${resume_args[@]}" \
      "${sample_output_args[@]}" \
      --no-progress
  ) > >(
    tee "$shard_log" |
      awk -v prefix="[eval-4gpu][shard=$rank] " '/^\[EVAL-SAMPLE\]/{print prefix $0; fflush()}'
  ) 2>&1 &
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

merge_ready_dataset_snapshots() {
  local dataset rank ready output merge_log json_path csv_path camera_path
  local -a jsons csvs camera_csvs
  for dataset in stanford2d3ds matterport3d structured3d panocity; do
    output="$EVAL_OUT/dataset_summaries/${dataset}_summary.json"
    [[ -s "$output" ]] && continue
    ready=1
    jsons=()
    csvs=()
    camera_csvs=()
    for rank in "${!GPU_LIST[@]}"; do
      json_path="$EVAL_OUT/shards/shard_${rank}_${dataset}.json"
      csv_path="$EVAL_OUT/shards/shard_${rank}_${dataset}.csv"
      camera_path="$EVAL_OUT/shards/shard_${rank}_${dataset}_camera_pairs.csv"
      jsons+=("$json_path")
      csvs+=("$csv_path")
      camera_csvs+=("$camera_path")
      if [[ ! -s "$json_path" || ! -s "$csv_path" || ! -s "$camera_path" ]]; then
        ready=0
      fi
    done
    [[ "$ready" -eq 1 ]] || continue
    merge_log="$EVAL_OUT/dataset_summaries/${dataset}_merge.log"
    echo "[eval-4gpu] merging completed dataset=$dataset $(date --iso-8601=seconds)" | tee -a "$EVAL_OUT/eval_4gpu.log"
    if PYTHONPATH="$LUNA${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" scripts/merge_mixed4_eval_shards.py \
      --shard-json "${jsons[@]}" \
      --shard-csv "${csvs[@]}" \
      --shard-camera-csv "${camera_csvs[@]}" \
      --output "$output" \
      --per-sample-csv "$EVAL_OUT/dataset_summaries/${dataset}_per_sample.csv" \
      --camera-pair-csv "$EVAL_OUT/dataset_summaries/${dataset}_camera_pairs.csv" \
      --train-loss-csv "$TRAIN_LOSS_CSV" \
      > "$merge_log" 2>&1; then
      echo "[eval-4gpu] dataset summary ready: $output" | tee -a "$EVAL_OUT/eval_4gpu.log"
    else
      echo "[eval-4gpu] dataset merge failed: $dataset; inspect $merge_log" | tee -a "$EVAL_OUT/eval_4gpu.log"
      tail -n 40 "$merge_log" | tee -a "$EVAL_OUT/eval_4gpu.log"
      return 1
    fi
  done
}

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
latest_samples = []
states = []

def sample_metric_text(payload):
    fields = (
        ("loss", "last_loss"),
        ("depth", "last_loss_depth"),
        ("camera", "last_loss_camera"),
        ("raw_absrel", "last_depth_abs_rel"),
        ("irls_absrel", "last_depth_irls_abs_rel"),
        ("delta1", "last_depth_irls_delta_1p25"),
        ("t_med_deg", "last_camera_pose_translation_deg_median"),
        ("r_med_deg", "last_camera_pose_rotation_deg_median"),
    )
    parts = []
    for label, key in fields:
        if label in {"t_med_deg", "r_med_deg"} and int(payload.get("last_camera_pose_pair_count", 0) or 0) <= 0:
            continue
        value = payload.get(key)
        if isinstance(value, (float, int)):
            parts.append(f"{label}={float(value):.4g}")
    return " ".join(parts)

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
    updated_at = float(payload.get("updated_at", now) or now)
    age = now - updated_at
    metrics_text = sample_metric_text(payload)
    metrics_text = f" {metrics_text}" if metrics_text else ""
    items.append(f"rank{rank}@gpu{gpu}:pid={pid} {state} {run} {done}/{total} age={age:.0f}s{metrics_text}")
    processed_total += done
    sample_total += total
    if run not in ("", "-"):
        active_runs.append(str(run))
    if payload.get("last_dataset_index") is not None:
        latest_samples.append((updated_at, payload))
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
latest_text = ""
if latest_samples:
    latest_payload = max(latest_samples, key=lambda item: item[0])[1]
    latest_metrics = sample_metric_text(latest_payload)
    if latest_metrics:
        latest_text = ", " + latest_metrics.replace(" ", ", ")
state_text = "done" if states and all(state == "done" for state in states) else ("init" if not sample_total else "running")
print(
    f"Eval {run_label}: {percent:5.1f}%|{bar}| {processed_total}/{sample_total} "
    f"[{elapsed/60:.1f}m<{eta_text}, {rate:.2f} samples/s, state={state_text}{latest_text}]"
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
    merge_ready_dataset_snapshots
    [[ "$alive" -eq 0 ]] && break
    sleep "$PROGRESS_INTERVAL_SECONDS"
  done
  if [[ "$PROGRESS_PRINTED" == "1" ]]; then
    printf '\n'
  fi
else
  while true; do
    alive=0
    for pid in "${PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        alive=1
        break
      fi
    done
    merge_ready_dataset_snapshots
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

merge_ready_dataset_snapshots

echo "[eval-4gpu] merging shards $(date --iso-8601=seconds)" | tee -a "$EVAL_OUT/eval_4gpu.log"
SHARD_JSONS=()
SHARD_CSVS=()
SHARD_CAMERA_CSVS=()
for rank in "${!GPU_LIST[@]}"; do
  shard_json="$EVAL_OUT/shards/shard_${rank}.json"
  shard_csv="$EVAL_OUT/shards/shard_${rank}.csv"
  shard_camera_csv="$EVAL_OUT/shards/shard_${rank}_camera_pairs.csv"
  for required_file in "$shard_json" "$shard_csv" "$shard_camera_csv"; do
    if [[ ! -s "$required_file" ]]; then
      echo "[eval-4gpu] missing shard output before merge: $required_file" | tee -a "$EVAL_OUT/eval_4gpu.log"
      exit 1
    fi
  done
  SHARD_JSONS+=("$shard_json")
  SHARD_CSVS+=("$shard_csv")
  SHARD_CAMERA_CSVS+=("$shard_camera_csv")
done

MERGE_LOG="$EVAL_OUT/validation_mixed4_by_dataset_valtestfull_console.log"
if ! PYTHONPATH="$LUNA${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" scripts/merge_mixed4_eval_shards.py \
  --shard-json "${SHARD_JSONS[@]}" \
  --shard-csv "${SHARD_CSVS[@]}" \
  --shard-camera-csv "${SHARD_CAMERA_CSVS[@]}" \
  --output "$EVAL_OUT/validation_mixed4_by_dataset_valtestfull_summary.json" \
  --per-sample-csv "$EVAL_OUT/validation_mixed4_by_dataset_valtestfull_per_sample.csv" \
  --camera-pair-csv "$EVAL_OUT/validation_mixed4_by_dataset_valtestfull_camera_pairs.csv" \
  --train-loss-csv "$TRAIN_LOSS_CSV" \
  > "$MERGE_LOG" 2>&1; then
  echo "[eval-4gpu] shard merge failed; tail of $MERGE_LOG:" | tee -a "$EVAL_OUT/eval_4gpu.log"
  tail -n 80 "$MERGE_LOG" | tee -a "$EVAL_OUT/eval_4gpu.log"
  exit 1
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

if [[ "$LIMIT_PER_DATASET" == "0" && "$EVAL_DATASETS" == "all" ]]; then
  "$PYTHON" scripts/validate_eval_cardinality.py "$SUMMARY_JSON" | tee -a "$EVAL_OUT/eval_4gpu.log"
fi

echo "[eval-4gpu] finished $(date --iso-8601=seconds)" | tee -a "$EVAL_OUT/eval_4gpu.log"
