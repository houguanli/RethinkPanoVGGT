#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$BASELINE/.." && pwd)"
LUNA="$ROOT/Rethink_pano_new_exp_omega"

if [[ -z "${PYTHON:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON="python"
  elif [[ -x /home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python ]]; then
    PYTHON="/home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python"
  else
    echo "[naive-fullerp] no Python interpreter found; activate the vggt-omega conda environment or set PYTHON" >&2
    exit 1
  fi
fi
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29841}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "${PANOVGGT_ROOT:-}" ]]; then
  if [[ -d /mnt/e/PanoVGGT_minimal_datasets/datasets ]]; then
    PANOVGGT_ROOT="/mnt/e/PanoVGGT_minimal_datasets/datasets"
  else
    PANOVGGT_ROOT="$(cd "$ROOT/.." && pwd)/panovggt"
  fi
fi
if [[ -z "${VGGT_OMEGA_CKPT:-}" ]]; then
  for candidate in \
    /whitehole/AOKI/vggt-omega/ckpt/vggt_omega_1b_512.pt \
    /whitehole/AOKI/RethinkPanoVGGT_omega/ckpt/vggt_omega_1b_512.pt \
    /home/aoki/RethinkPanoVGGT_omega_compare_methods_only/ckpt/VGGT-Omega/vggt_omega_1b_512.pt \
    /home/aoki/RethinkPanoVGGT_omega/ckpt/vggt_omega_1b_512.pt \
    "$BASELINE/ckpt/vggt_omega_1b_512.pt"; do
    if [[ -s "$candidate" ]]; then
      VGGT_OMEGA_CKPT="$candidate"
      break
    fi
  done
fi
VGGT_OMEGA_CKPT="${VGGT_OMEGA_CKPT:-/whitehole/AOKI/vggt-omega/ckpt/vggt_omega_1b_512.pt}"

CONFIG="${NAIVE_FULLERP_CONFIG:-mixed4_naive_fullerp_vggtomega_scalealigned_2pano}"
CONFIG="${CONFIG%.yaml}"
if [[ "$CONFIG" == */* || ! -f "$BASELINE/training/config/$CONFIG.yaml" ]]; then
  echo "[naive-fullerp] invalid baseline config: $CONFIG" >&2
  echo "[naive-fullerp] expected: $BASELINE/training/config/$CONFIG.yaml" >&2
  exit 1
fi
EXP_NAME="${EXP_NAME:-local_naive_fullerp_vggtomega_2to10p_progressive_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-$BASELINE/logs/$EXP_NAME}"
STAGE_PLAN="${STAGE_PLAN:-384:2,512:6,1024:10}"
STAGE_DURATION_MINUTES="${STAGE_DURATION_MINUTES:-4}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1000000}"
START_CHECKPOINT="${START_CHECKPOINT:-}"
BUILD_INDEXES="${BUILD_INDEXES:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_IMG_SIZE="${EVAL_IMG_SIZE:-384}"
EVAL_LIMIT_PER_DATASET="${EVAL_LIMIT_PER_DATASET:-0}"

mkdir -p "$OUT"
cd "$BASELINE"

{
  echo "[naive-fullerp] started $(date --iso-8601=seconds)"
  echo "[naive-fullerp] config=$CONFIG"
  echo "[naive-fullerp] output=$OUT"
  echo "[naive-fullerp] python=$PYTHON"
  echo "[naive-fullerp] nproc_per_node=$NPROC_PER_NODE"
  echo "[naive-fullerp] cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "[naive-fullerp] panovggt_root=$PANOVGGT_ROOT"
  echo "[naive-fullerp] vggt_omega_ckpt=$VGGT_OMEGA_CKPT"
  echo "[naive-fullerp] representation=full_erp_no_window_split"
  echo "[naive-fullerp] stage_plan=$STAGE_PLAN"
  echo "[naive-fullerp] stage_duration_minutes=$STAGE_DURATION_MINUTES"
  echo "[naive-fullerp] normalize_scene_scale=true"
  echo "[naive-fullerp] depth_scale_alignment=sample_log_median"
  echo "[naive-fullerp] start_checkpoint=${START_CHECKPOINT:-none}"
} | tee "$OUT/run_manifest.log"

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[naive-fullerp] preflight passed" | tee -a "$OUT/run_manifest.log"
  exit 0
fi

if [[ "$BUILD_INDEXES" == "1" ]]; then
  echo "[naive-fullerp] building/checking mixed4 indexes $(date --iso-8601=seconds)" | tee -a "$OUT/run_manifest.log"
  PYTHONPATH="$LUNA${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" "$LUNA/scripts/build_mixed4_official_indexes.py" \
    --root "$PANOVGGT_ROOT" \
    --train-fraction 0.95 \
    --seed 42 \
    --bad-scene-list "$LUNA/configs/structured3d_bad_scenes.txt" \
    2>&1 | tee -a "$OUT/run_manifest.log"
fi

previous_checkpoint="$START_CHECKPOINT"
stage_index=0
for stage_spec in ${STAGE_PLAN//,/ }; do
  stage_index=$((stage_index + 1))
  resolution="${stage_spec%%:*}"
  pano_count="${stage_spec##*:}"
  if [[ ! "$resolution" =~ ^[0-9]+$ || ! "$pano_count" =~ ^[0-9]+$ ]]; then
    echo "[naive-fullerp] invalid stage spec: $stage_spec" | tee -a "$OUT/run_manifest.log"
    exit 2
  fi
  if (( pano_count <= 2 )); then
    dataset_pano_counts="panocity:${pano_count},matterport3d:${pano_count},stanford2d3ds:${pano_count},structured3d:${pano_count}"
  else
    dataset_pano_counts="panocity:${pano_count},matterport3d:3,stanford2d3ds:3,structured3d:3"
  fi
  stage_out="$OUT/stage_${stage_index}_${resolution}x$((resolution / 2))_${pano_count}p"
  stage_port=$((MASTER_PORT + stage_index - 1))
  mkdir -p "$stage_out"
  overrides=(
    "exp_name=${EXP_NAME}_stage${stage_index}_${resolution}"
    "img_size=$resolution"
    "max_img_per_gpu=$pano_count"
    "data.train.common_config.fix_img_num=$pano_count"
    "data.train.common_config.img_nums=[$pano_count,$pano_count]"
    "data.train.common_config.max_img_per_gpu=$pano_count"
    "data.train.dataset.dataset_configs.0.pano_min_count=2"
    "data.train.dataset.dataset_configs.0.pano_max_count=$pano_count"
    "logging.log_dir=$stage_out"
    "checkpoint.save_dir=$stage_out/ckpts"
    "max_duration_minutes=$STAGE_DURATION_MINUTES"
    "limit_train_batches=$LIMIT_TRAIN_BATCHES"
  )
  if [[ -n "$previous_checkpoint" ]]; then
    overrides+=(
      "checkpoint.resume_checkpoint_path=$previous_checkpoint"
      "checkpoint.resume_optimizer=true"
      "checkpoint.resume_training_progress=false"
      "checkpoint.resume_scaler=true"
    )
  fi

  echo "[naive-fullerp] stage=$stage_index resolution=${resolution}x$((resolution / 2)) max_panos=$pano_count dataset_pano_counts=$dataset_pano_counts started $(date --iso-8601=seconds)" \
    | tee -a "$OUT/run_manifest.log"
  if (( NPROC_PER_NODE > 1 )); then
    launcher=(
      "$PYTHON" -m torch.distributed.run
      --standalone
      "--nproc_per_node=$NPROC_PER_NODE"
      "--master_port=$stage_port"
      training/launch.py
    )
  else
    export LOCAL_RANK="${LOCAL_RANK:-0}"
    export RANK="${RANK:-0}"
    export WORLD_SIZE="${WORLD_SIZE:-1}"
    launcher=("$PYTHON" training/launch.py)
  fi
  set +e
  MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$stage_port" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  PANOVGGT_ROOT="$PANOVGGT_ROOT" VGGT_OMEGA_CKPT="$VGGT_OMEGA_CKPT" \
  DATASET_PANO_COUNTS="$dataset_pano_counts" \
  PYTHONPATH="$BASELINE${PYTHONPATH:+:$PYTHONPATH}" \
  "${launcher[@]}" --config "$CONFIG" "${overrides[@]}" \
    2>&1 | tee "$stage_out/train_console.log"
  status=${PIPESTATUS[0]}
  set -e

  if [[ "$status" -ne 0 ]]; then
    if grep -Eiq "CUDA out of memory|torch.OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" "$stage_out/train_console.log"; then
      echo "[naive-fullerp] stage=$stage_index resolution=$resolution OOM; keeping $previous_checkpoint" \
        | tee -a "$OUT/run_manifest.log"
      break
    fi
    echo "[naive-fullerp] stage=$stage_index resolution=$resolution failed status=$status" \
      | tee -a "$OUT/run_manifest.log"
    exit "$status"
  fi

  stage_checkpoint="$stage_out/ckpts/checkpoint.pt"
  if [[ ! -s "$stage_checkpoint" ]]; then
    echo "[naive-fullerp] missing checkpoint: $stage_checkpoint" | tee -a "$OUT/run_manifest.log"
    exit 1
  fi
  previous_checkpoint="$stage_checkpoint"
  printf '%s\n' "$previous_checkpoint" > "$OUT/last_successful_checkpoint.txt"
  echo "[naive-fullerp] stage=$stage_index resolution=$resolution complete checkpoint=$stage_checkpoint" \
    | tee -a "$OUT/run_manifest.log"
done

echo "[naive-fullerp] finished $(date --iso-8601=seconds)" | tee -a "$OUT/run_manifest.log"

if [[ "$RUN_EVAL" == "1" && -n "$previous_checkpoint" ]]; then
  echo "[naive-fullerp] launching full-ERP eval checkpoint=$previous_checkpoint" | tee -a "$OUT/run_manifest.log"
  RUN_OUT="$OUT" \
  CONFIG="$CONFIG" \
  DATASET_ROOT="$PANOVGGT_ROOT" \
  VGGT_OMEGA_CKPT="$VGGT_OMEGA_CKPT" \
  EVAL_IMG_SIZE="$EVAL_IMG_SIZE" \
  LIMIT_PER_DATASET="$EVAL_LIMIT_PER_DATASET" \
  FOREGROUND=1 \
    bash "$SCRIPT_DIR/run_eval_naive_fullerp.sh" "$previous_checkpoint" --foreground 2>&1 | tee -a "$OUT/run_manifest.log"
fi
