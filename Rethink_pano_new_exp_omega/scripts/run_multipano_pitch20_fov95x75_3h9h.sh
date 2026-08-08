#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUNA="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-/home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
EVAL_GPUS="${EVAL_GPUS:-$CUDA_VISIBLE_DEVICES}"
export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WARMUP_CONFIG="${WARMUP_CONFIG:-configs/multipano_4090_mixed4_pitch20_fov95x75_omega_warmup_3h.yaml}"
LUNA_CONFIG="${LUNA_CONFIG:-configs/multipano_4090_mixed4_pano_weighted_officialscale_luna_after_full_warmup_9h.yaml}"
RUN_NAME="${RUN_NAME:-mixed4_pitch20_fov95x75_$(date +%Y%m%d_%H%M%S)}"
WARMUP_OUT="${WARMUP_OUT:-$LUNA/logs/${RUN_NAME}_warmup_3h}"
LUNA_OUT="${LUNA_OUT:-$LUNA/logs/${RUN_NAME}_luna_9h}"
EVAL_OUT="${EVAL_OUT:-$LUNA_OUT/eval_mixed4}"
SEQ_LOG="${SEQ_LOG:-$LUNA/logs/${RUN_NAME}_sequence.log}"

WARMUP_MINUTES="${WARMUP_MINUTES:-180}"
LUNA_MINUTES="${LUNA_MINUTES:-540}"
VALIDATION_LIMIT_PER_DATASET="${VALIDATION_LIMIT_PER_DATASET:-0}"
RUN_VALIDATION="${RUN_VALIDATION:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DATASET_PANO_MAX_COUNTS="${DATASET_PANO_MAX_COUNTS:-panocity:8,matterport3d:3,stanford2d3ds:3,structured3d:3}"
CHECK_ONLY="${CHECK_ONLY:-0}"
WARMUP_EXTRA_TRAIN_ARGS_ARRAY=()
if [[ -n "${WARMUP_EXTRA_TRAIN_ARGS:-}" ]]; then
  # Intended for bounded smoke/debug overrides such as --dataset-max-samples.
  # shellcheck disable=SC2206
  WARMUP_EXTRA_TRAIN_ARGS_ARRAY=($WARMUP_EXTRA_TRAIN_ARGS)
fi
LUNA_EXTRA_TRAIN_ARGS_ARRAY=()
if [[ -n "${LUNA_EXTRA_TRAIN_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  LUNA_EXTRA_TRAIN_ARGS_ARRAY=($LUNA_EXTRA_TRAIN_ARGS)
fi

is_mixed4_root() {
  local root="$1" dataset
  [[ -d "$root" ]] || return 1
  for dataset in Panocity Matterport3D Stanford2D3DS Structured3D; do
    [[ -d "$root/$dataset" ]] || return 1
  done
}

if [[ -z "${PANOVGGT_ROOT:-}" ]]; then
  for candidate in \
    /mnt/f/PanoVGGT_minimal_datasets/datasets \
    /mnt/e/PanoVGGT_minimal_datasets/datasets \
    /whitehole/AOKI/PanoVGGT_minimal_datasets/datasets \
    /whitehole/AOKI/datasets; do
    if is_mixed4_root "$candidate"; then
      PANOVGGT_ROOT="$candidate"
      break
    fi
  done
fi
PANOVGGT_ROOT="${PANOVGGT_ROOT:-}"

if [[ -z "${BASE_CHECKPOINT:-}" ]]; then
  for candidate in \
    /home/aoki/RethinkPanoVGGT_omega_compare_methods_only/ckpt/VGGT-Omega/vggt_omega_1b_512.pt \
    /home/aoki/RethinkPanoVGGT_omega/ckpt/vggt_omega_1b_512.pt \
    /whitehole/AOKI/vggt-omega/ckpt/vggt_omega_1b_512.pt \
    /whitehole/AOKI/RethinkPanoVGGT_omega/ckpt/vggt_omega_1b_512.pt; do
    if [[ -s "$candidate" ]]; then
      BASE_CHECKPOINT="$candidate"
      break
    fi
  done
fi
BASE_CHECKPOINT="${BASE_CHECKPOINT:-}"

failures=0
require_file() {
  local label="$1" path="$2"
  if [[ -n "$path" && -s "$path" ]]; then
    echo "[preflight][ok] $label=$path"
  else
    echo "[preflight][missing] $label=${path:-<unset>}" >&2
    failures=$((failures + 1))
  fi
}
require_dir() {
  local label="$1" path="$2"
  if [[ -n "$path" && -d "$path" ]]; then
    echo "[preflight][ok] $label=$path"
  else
    echo "[preflight][missing] $label=${path:-<unset>}" >&2
    failures=$((failures + 1))
  fi
}

echo "[preflight] $(date --iso-8601=seconds)"
require_file python "$PYTHON"
require_file warmup_config "$LUNA/$WARMUP_CONFIG"
require_file luna_config "$LUNA/$LUNA_CONFIG"
require_file foundation_checkpoint "$BASE_CHECKPOINT"
require_dir mixed4_root "$PANOVGGT_ROOT"
for dataset in Panocity Matterport3D Stanford2D3DS Structured3D; do
  require_dir "dataset:$dataset" "$PANOVGGT_ROOT/$dataset"
done
if [[ "$failures" -ne 0 ]]; then
  echo "[preflight] failed with $failures missing path(s)" >&2
  exit 2
fi

cd "$LUNA"
"$PYTHON" - "$WARMUP_CONFIG" "$LUNA_CONFIG" <<'PY'
import sys
from training.train_pano_omega import parse_args

for path in sys.argv[1:]:
    args = parse_args(["--config", path])
    print(
        f"[preflight][config] {path} pitch={args.pitch_degrees} "
        f"fov={args.fov_x_degrees}x{args.fov_y_degrees} yaw={args.num_yaw} "
        f"duration={args.max_duration_minutes}m distributed={args.distributed}"
    )
    if str(args.pitch_degrees) != "-20":
        raise SystemExit(f"Unexpected pitch in {path}: {args.pitch_degrees}")
    if float(args.fov_x_degrees) != 95.0 or float(args.fov_y_degrees) != 75.0:
        raise SystemExit(f"Unexpected FoV in {path}")
PY

if [[ "$CHECK_ONLY" == "1" || "${1:-}" == "--check-only" ]]; then
  echo "[preflight] complete; training not started"
  exit 0
fi

mkdir -p "$WARMUP_OUT" "$LUNA_OUT" "$EVAL_OUT" "$(dirname "$SEQ_LOG")"
printf '%s\n' "$$" > "$LUNA_OUT/pipeline.pid"
{
  echo "[sequence] started $(date --iso-8601=seconds)"
  echo "[sequence] run_name=$RUN_NAME"
  echo "[sequence] root=$LUNA"
  echo "[sequence] python=$PYTHON"
  echo "[sequence] cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "[sequence] nproc_per_node=$NPROC_PER_NODE"
  echo "[sequence] dataset_root=$PANOVGGT_ROOT"
  echo "[sequence] foundation=$BASE_CHECKPOINT"
  echo "[sequence] geometry=pitch:-20,fov:95x75,yaw:4,window:384"
  echo "[sequence] schedule=${WARMUP_MINUTES}m_omega_warmup+${LUNA_MINUTES}m_luna"
  echo "[sequence] warmup_out=$WARMUP_OUT"
  echo "[sequence] luna_out=$LUNA_OUT"
  echo "[sequence] eval_out=$EVAL_OUT"
  echo "[sequence] warmup_extra_train_args=${WARMUP_EXTRA_TRAIN_ARGS:-}"
  echo "[sequence] luna_extra_train_args=${LUNA_EXTRA_TRAIN_ARGS:-}"
} | tee -a "$SEQ_LOG"

if [[ "${SKIP_INDEX_BUILD:-0}" != "1" ]]; then
  echo "[sequence] checking mixed4 indexes $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
  PYTHONPATH="$LUNA${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" scripts/build_mixed4_official_indexes.py \
      --root "$PANOVGGT_ROOT" --train-fraction 0.95 --seed 42 \
      2>&1 | tee -a "$SEQ_LOG"
fi

WARMUP_CKPT="$WARMUP_OUT/last.pt"
if [[ -s "$WARMUP_CKPT" && "${FORCE_WARMUP:-0}" != "1" ]]; then
  echo "[sequence] warm-up checkpoint exists; skipping stage 1: $WARMUP_CKPT" | tee -a "$SEQ_LOG"
else
  echo "[sequence] stage1 Omega warm-up started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
  PYTHONPATH="$LUNA${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" \
      training/train_pano_omega.py --config "$WARMUP_CONFIG" \
      --dataset-root "$PANOVGGT_ROOT" \
      --checkpoint "$BASE_CHECKPOINT" \
      --output-dir "$WARMUP_OUT" \
      --tensorboard-dir "$WARMUP_OUT/tensorboard" \
      --debug-dir "$WARMUP_OUT/debug" \
      --max-duration-minutes "$WARMUP_MINUTES" \
      --num-workers "$NUM_WORKERS" \
      --dataset-pano-max-counts "$DATASET_PANO_MAX_COUNTS" \
      --no-inherit-checkpoint-training-defaults \
      "${WARMUP_EXTRA_TRAIN_ARGS_ARRAY[@]}" \
      2>&1 | tee -a "$WARMUP_OUT/train.log"
  echo "[sequence] stage1 Omega warm-up finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
fi
require_file warmup_checkpoint "$WARMUP_CKPT"
if [[ ! -s "$WARMUP_CKPT" ]]; then
  exit 1
fi

LUNA_CKPT="$LUNA_OUT/last.pt"
if [[ -s "$LUNA_CKPT" && "${FORCE_LUNA:-0}" != "1" ]]; then
  echo "[sequence] LUNA checkpoint exists; skipping stage 2: $LUNA_CKPT" | tee -a "$SEQ_LOG"
else
  echo "[sequence] stage2 LUNA branch started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
  PYTHONPATH="$LUNA${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" \
      training/train_pano_omega.py --config "$LUNA_CONFIG" \
      --base-checkpoint "$BASE_CHECKPOINT" \
      --checkpoint "$WARMUP_CKPT" \
      --dataset-root "$PANOVGGT_ROOT" \
      --output-dir "$LUNA_OUT" \
      --tensorboard-dir "$LUNA_OUT/tensorboard" \
      --debug-dir "$LUNA_OUT/debug" \
      --max-duration-minutes "$LUNA_MINUTES" \
      --num-workers "$NUM_WORKERS" \
      --dataset-pano-max-counts "$DATASET_PANO_MAX_COUNTS" \
      --no-inherit-checkpoint-training-defaults \
      "${LUNA_EXTRA_TRAIN_ARGS_ARRAY[@]}" \
      2>&1 | tee -a "$LUNA_OUT/train.log"
  echo "[sequence] stage2 LUNA branch finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
fi
require_file luna_checkpoint "$LUNA_CKPT"
if [[ ! -s "$LUNA_CKPT" ]]; then
  exit 1
fi

if [[ "$RUN_VALIDATION" == "1" ]]; then
  echo "[sequence] mixed4 eval started $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
  env \
    PYTHON="$PYTHON" \
    GPUS="$EVAL_GPUS" \
    CONFIG="$LUNA_CONFIG" \
    DATASET_ROOT="$PANOVGGT_ROOT" \
    LUNA_OUT="$LUNA_OUT" \
    CHECKPOINT="$LUNA_CKPT" \
    TRAIN_LOSS_CSV="$LUNA_OUT/loss.csv" \
    EVAL_OUT="$EVAL_OUT" \
    LIMIT_PER_DATASET="$VALIDATION_LIMIT_PER_DATASET" \
    WINDOW_SIZE=384 \
    NUM_YAW=4 \
    bash scripts/run_multipano_mixed4_eval_4gpu.sh \
    2>&1 | tee -a "$SEQ_LOG"
  require_file eval_summary "$EVAL_OUT/validation_mixed4_by_dataset_valtestfull_summary.json"
  if [[ ! -s "$EVAL_OUT/validation_mixed4_by_dataset_valtestfull_summary.json" ]]; then
    exit 1
  fi
  echo "[sequence] mixed4 eval finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
fi

if [[ -s "$WARMUP_OUT/loss.csv" && -s "$LUNA_OUT/loss.csv" ]]; then
  "$PYTHON" scripts/plot_loss_csv.py \
    --run "warmup=$WARMUP_OUT/loss.csv" \
    --run "luna=$LUNA_OUT/loss.csv@3" \
    --metric auto --x elapsed_hours --smooth-method rolling_median \
    --rolling-window 401 --resample-seconds 10 --clip-quantile 0.98 \
    --raw-alpha 0.10 --out "$LUNA_OUT/loss_curve_3h9h.png" \
    --title "Pitch -20, FoV 95x75: 3h Omega warm-up + 9h LUNA" \
    2>&1 | tee -a "$SEQ_LOG"
fi

echo "[sequence] finished $(date --iso-8601=seconds)" | tee -a "$SEQ_LOG"
