#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_panosuncg_0716_native4_4gpu.sh \
    --checkpoint /absolute/path/to/0716/last.pt \
    --dataset-root /absolute/path/to/PanoSUNCG_zeroshot \
    --output-dir /absolute/path/to/eval_results/0716_native4 \
    [options]

Required:
  --checkpoint PATH        0716 full-pipeline checkpoint (trainable-delta chain)
  --dataset-root PATH      PanoSUNCG_zeroshot root or its parent datasets root
  --output-dir PATH        New or resumable evaluation output directory

Options:
  --config PATH            Training config used to rebuild the checkpoint
  --gpus LIST              Comma-separated physical GPU ids (default: 0,1,2,3)
  --python PATH            Python executable
  --storage-root PATH      Mirrored server storage root (default: /AOKI/whitehole)
  --foundation-checkpoint PATH
                            Optional explicit VGGT-Omega foundation checkpoint
  --limit N                Depth smoke-test limit; 0 means all 3944 (default: 0)
  --max-trajectories N     Camera smoke-test limit; 0 means all 118 (default: 0)
  --frames-per-trajectory N
                            Camera frames per trajectory (default: 5)
  --depth-only             Skip camera-center evaluation
  --check-only             Validate paths and print the resolved protocol
  -h, --help               Show this help

The depth stage uses one independent resumable shard per GPU, then verifies and
merges all shard CSVs. Camera evaluation starts on the first listed GPU after
depth merging. Re-running the same command resumes both stages.
EOF
}

CHECKPOINT=""
DATASET_ROOT=""
OUTPUT_DIR=""
CONFIG="$ROOT/configs/multipano_rtx5000x4_mixed4_pano_all384_luna_after_full_warmup_9h.yaml"
GPUS="0,1,2,3"
PYTHON_BIN="${PYTHON_BIN:-}"
STORAGE_ROOT="${AOKI_STORAGE_ROOT:-/AOKI/whitehole}"
FOUNDATION_CHECKPOINT="${VGGT_OMEGA_CKPT:-}"
LIMIT=0
MAX_TRAJECTORIES=0
FRAMES_PER_TRAJECTORY=5
DEPTH_ONLY=0
CHECK_ONLY=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --storage-root) STORAGE_ROOT="$2"; shift 2 ;;
    --foundation-checkpoint) FOUNDATION_CHECKPOINT="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --max-trajectories) MAX_TRAJECTORIES="$2"; shift 2 ;;
    --frames-per-trajectory) FRAMES_PER_TRAJECTORY="$2"; shift 2 ;;
    --depth-only) DEPTH_ONLY=1; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[panosuncg-native4] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$CHECKPOINT" || -z "$DATASET_ROOT" || -z "$OUTPUT_DIR" ]]; then
  echo "[panosuncg-native4] --checkpoint, --dataset-root and --output-dir are required" >&2
  usage >&2
  exit 2
fi

first_python() {
  local candidate
  for candidate in \
    "$PYTHON_BIN" \
    "$STORAGE_ROOT/home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python" \
    "$STORAGE_ROOT/miniconda3/envs/RethinkPanoVGGT_omega/bin/python" \
    /home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python \
    "$(command -v python || true)"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

PYTHON_BIN="$(first_python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "[panosuncg-native4] Python not found; use --python" >&2
  exit 2
fi

CHECKPOINT="$(readlink -f "$CHECKPOINT")"
CONFIG="$(readlink -f "$CONFIG")"
DATASET_ROOT="$(readlink -f "$DATASET_ROOT")"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(readlink -f "$OUTPUT_DIR")"

ZERO_ROOT=""
for candidate in "$DATASET_ROOT" "$DATASET_ROOT/PanoSUNCG_zeroshot"; do
  if [[ -s "$candidate/panosuncg_da2_split.txt" \
        && -d "$candidate/PanoSUNCG/rotated" \
        && -d "$candidate/PanoSUNCG/labels" ]]; then
    ZERO_ROOT="$candidate"
    break
  fi
done
if [[ -z "$ZERO_ROOT" && "$(basename "$DATASET_ROOT")" == "rotated" ]]; then
  candidate="$(dirname "$(dirname "$DATASET_ROOT")")"
  if [[ -s "$candidate/panosuncg_da2_split.txt" \
        && -d "$candidate/PanoSUNCG/labels" ]]; then
    ZERO_ROOT="$candidate"
  fi
fi

failures=0
require_file() {
  local label="$1" path="$2"
  if [[ -s "$path" ]]; then
    echo "[panosuncg-native4][ok] $label=$path"
  else
    echo "[panosuncg-native4][missing] $label=$path" >&2
    failures=$((failures + 1))
  fi
}
require_dir() {
  local label="$1" path="$2"
  if [[ -d "$path" ]]; then
    echo "[panosuncg-native4][ok] $label=$path"
  else
    echo "[panosuncg-native4][missing] $label=$path" >&2
    failures=$((failures + 1))
  fi
}

require_file python "$PYTHON_BIN"
require_file checkpoint "$CHECKPOINT"
require_file config "$CONFIG"
if [[ -z "$ZERO_ROOT" ]]; then
  echo "[panosuncg-native4][missing] expected panosuncg_da2_split.txt, PanoSUNCG/rotated and PanoSUNCG/labels under $DATASET_ROOT" >&2
  failures=$((failures + 1))
else
  require_file split "$ZERO_ROOT/panosuncg_da2_split.txt"
  require_dir rotated "$ZERO_ROOT/PanoSUNCG/rotated"
  require_dir labels "$ZERO_ROOT/PanoSUNCG/labels"
fi
if [[ -n "$FOUNDATION_CHECKPOINT" ]]; then
  FOUNDATION_CHECKPOINT="$(readlink -f "$FOUNDATION_CHECKPOINT")"
  require_file foundation_checkpoint "$FOUNDATION_CHECKPOINT"
fi
if [[ "$failures" -ne 0 ]]; then
  exit 2
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
if [[ "${#GPU_ARRAY[@]}" -lt 1 ]]; then
  echo "[panosuncg-native4] --gpus must contain at least one GPU id" >&2
  exit 2
fi
for gpu in "${GPU_ARRAY[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "[panosuncg-native4] invalid GPU id: $gpu" >&2
    exit 2
  fi
done

export AOKI_STORAGE_ROOT="$STORAGE_ROOT"
if [[ -n "$FOUNDATION_CHECKPOINT" ]]; then
  export VGGT_OMEGA_CKPT="$FOUNDATION_CHECKPOINT"
fi

echo "[panosuncg-native4] repo=$ROOT"
echo "[panosuncg-native4] checkpoint=$CHECKPOINT"
echo "[panosuncg-native4] config=$CONFIG"
echo "[panosuncg-native4] dataset=$ZERO_ROOT"
echo "[panosuncg-native4] output=$OUTPUT_DIR"
echo "[panosuncg-native4] GPUs=$GPUS"
echo "[panosuncg-native4] depth_protocol=4x384, yaw=4, pitch=-15, FOV=75, direct-window weighted 10-step scale-only IRLS"
echo "[panosuncg-native4] camera_protocol=5 evenly spaced frames, per-trajectory Sim(3)"

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  "$PYTHON_BIN" - "$CHECKPOINT" <<'PY'
import sys
from pathlib import Path
import torch

path = Path(sys.argv[1])
try:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
except (TypeError, RuntimeError):
    payload = torch.load(path, map_location="cpu", weights_only=False)
args = payload.get("args", {}) if isinstance(payload, dict) else {}
actual = {
    "checkpoint_format": payload.get("checkpoint_format"),
    "step": payload.get("step"),
    "window_size": args.get("window_size"),
    "num_yaw": args.get("num_yaw"),
    "pitch_degrees": args.get("pitch_degrees"),
    "fov_degrees": args.get("fov_degrees"),
}
print(f"[panosuncg-native4][checkpoint] {actual}")
expected = (384, 4, "-15", 75.0)
observed = (
    actual["window_size"],
    actual["num_yaw"],
    str(actual["pitch_degrees"]),
    float(actual["fov_degrees"]),
)
if observed != expected:
    raise RuntimeError(f"Checkpoint sampler mismatch: expected {expected}, got {observed}")
PY
  echo "[panosuncg-native4] check complete; evaluation not started"
  exit 0
fi

exec 9>"$OUTPUT_DIR/.launcher.lock"
if ! flock -n 9; then
  echo "[panosuncg-native4] another launcher holds $OUTPUT_DIR/.launcher.lock" >&2
  exit 3
fi

DEPTH_ROOT="$OUTPUT_DIR/depth_native4"
SHARD_ROOT="$DEPTH_ROOT/shards"
mkdir -p "$SHARD_ROOT"
NUM_SHARDS="${#GPU_ARRAY[@]}"
PIDS=()

cleanup() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

for rank in "${!GPU_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[$rank]}"
  shard_dir="$SHARD_ROOT/shard_$(printf '%02d' "$rank")"
  mkdir -p "$shard_dir"
  echo "[panosuncg-native4] launch depth shard=$rank/$NUM_SHARDS gpu=$gpu"
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" scripts/evaluate_panosuncg_native4.py \
      --config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --dataset-root "$ZERO_ROOT" \
      --output-dir "$shard_dir" \
      --device cuda \
      --amp-dtype bfloat16 \
      --limit "$LIMIT" \
      --num-shards "$NUM_SHARDS" \
      --shard-rank "$rank" \
      --resume \
      --progress-every 10 \
      >"$shard_dir/run.log" 2>&1 &
  PIDS+=("$!")
done

while true; do
  alive=0
  rows=0
  for rank in "${!GPU_ARRAY[@]}"; do
    if kill -0 "${PIDS[$rank]}" 2>/dev/null; then
      alive=1
    fi
    shard_csv="$SHARD_ROOT/shard_$(printf '%02d' "$rank")/per_sample_metrics.csv"
    if [[ -s "$shard_csv" ]]; then
      count=$(($(wc -l < "$shard_csv") - 1))
      if (( count > 0 )); then rows=$((rows + count)); fi
    fi
  done
  echo "[panosuncg-native4] depth streamed_samples=$rows active=$alive"
  [[ "$alive" -eq 1 ]] || break
  sleep "${MONITOR_SECONDS:-30}"
done

failed=0
for rank in "${!GPU_ARRAY[@]}"; do
  if ! wait "${PIDS[$rank]}"; then
    shard_dir="$SHARD_ROOT/shard_$(printf '%02d' "$rank")"
    echo "[panosuncg-native4] depth shard $rank failed: $shard_dir/run.log" >&2
    tail -n 80 "$shard_dir/run.log" >&2 || true
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

SHARD_DIRS=()
for rank in "${!GPU_ARRAY[@]}"; do
  SHARD_DIRS+=("$SHARD_ROOT/shard_$(printf '%02d' "$rank")")
done
"$PYTHON_BIN" scripts/merge_panosuncg_native4_shards.py \
  --shard-dir "${SHARD_DIRS[@]}" \
  --output-dir "$DEPTH_ROOT" \
  >"$DEPTH_ROOT/merge.log" 2>&1
echo "[panosuncg-native4] depth complete: $DEPTH_ROOT/metrics_summary.json"

if [[ "$DEPTH_ONLY" -eq 0 ]]; then
  CAMERA_OUT="$OUTPUT_DIR/camera_pose"
  mkdir -p "$CAMERA_OUT"
  camera_gpu="${GPU_ARRAY[0]}"
  echo "[panosuncg-native4] launch camera evaluation gpu=$camera_gpu"
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="$camera_gpu" \
  PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" scripts/evaluate_panosuncg_0716_camera_centers.py \
      --config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --dataset-root "$ZERO_ROOT" \
      --output-dir "$CAMERA_OUT" \
      --frames-per-trajectory "$FRAMES_PER_TRAJECTORY" \
      --max-trajectories "$MAX_TRAJECTORIES" \
      --device cuda \
      --amp-dtype bfloat16 \
      --resume \
      --progress-every 5 \
      >"$CAMERA_OUT/run.log" 2>&1
  echo "[panosuncg-native4] camera complete: $CAMERA_OUT/camera_center_summary.json"
fi

echo "[panosuncg-native4] all requested stages completed"
