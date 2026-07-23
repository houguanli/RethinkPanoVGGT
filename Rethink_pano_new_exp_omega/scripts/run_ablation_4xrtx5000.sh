#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-}"
if [[ -z "$VARIANT" ]]; then
  echo "usage: $0 {full|no_patch_bank|no_geora|shuffle_patch_bank}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

case "$VARIANT" in
  full)
    CONFIG="configs/ablation_rtx5000x4_full.yaml"
    RUN_NAME="ablation_rtx5000x4_full"
    ;;
  no_patch_bank)
    CONFIG="configs/ablation_rtx5000x4_no_patch_bank.yaml"
    RUN_NAME="ablation_rtx5000x4_no_patch_bank"
    ;;
  no_geora)
    CONFIG="configs/ablation_rtx5000x4_no_geora.yaml"
    RUN_NAME="ablation_rtx5000x4_no_geora"
    ;;
  shuffle_patch_bank)
    CONFIG="configs/ablation_rtx5000x4_shuffle_patch_bank.yaml"
    RUN_NAME="ablation_rtx5000x4_shuffle_patch_bank"
    ;;
  *)
    echo "unknown variant: $VARIANT" >&2
    exit 2
    ;;
esac

PYTHON="${PYTHON:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
PRED_DEPTH_SCALE="${PRED_DEPTH_SCALE:-2.827019691467285}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "${PANOVGGT_ROOT:-}" ]]; then
  if [[ -d /mnt/e/PanoVGGT_minimal_datasets/datasets ]]; then
    PANOVGGT_ROOT="/mnt/e/PanoVGGT_minimal_datasets/datasets"
  else
    PANOVGGT_ROOT="$(cd "$REPO_ROOT/.." && pwd)/panovggt"
  fi
fi
if [[ ! -d "$PANOVGGT_ROOT" ]]; then
  echo "missing PanoVGGT dataset root: $PANOVGGT_ROOT" >&2
  exit 1
fi

if [[ -z "${BASE_CHECKPOINT:-}" ]]; then
  for candidate in \
    /whitehole/AOKI/vggt-omega/ckpt/vggt_omega_1b_512.pt \
    /whitehole/AOKI/RethinkPanoVGGT_omega/ckpt/vggt_omega_1b_512.pt \
    /home/aoki/RethinkPanoVGGT_omega_compare_methods_only/ckpt/VGGT-Omega/vggt_omega_1b_512.pt \
    /home/aoki/RethinkPanoVGGT_omega/ckpt/vggt_omega_1b_512.pt \
    "$PROJECT_ROOT/ckpt/vggt_omega_1b_512.pt"; do
    if [[ -s "$candidate" ]]; then
      BASE_CHECKPOINT="$candidate"
      break
    fi
  done
fi
if [[ -z "${BASE_CHECKPOINT:-}" || ! -s "$BASE_CHECKPOINT" ]]; then
  echo "set BASE_CHECKPOINT to vggt_omega_1b_512.pt" >&2
  exit 1
fi

WARMUP_CONFIG="${WARMUP_CONFIG:-configs/multipano_rtx5000x4_mixed4_pano_omega_warmup_3h_for_luna.yaml}"
WARMUP_OUT="${WARMUP_OUT:-$PROJECT_ROOT/logs/ablation_rtx5000x4_shared_warmup}"
WARMUP_CHECKPOINT="${WARMUP_CHECKPOINT:-$WARMUP_OUT/last.pt}"
RUN_OUT="${RUN_OUT:-$PROJECT_ROOT/logs/$RUN_NAME}"
EXTRA_TRAIN_ARGS_ARRAY=()
if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_TRAIN_ARGS_ARRAY=($EXTRA_TRAIN_ARGS)
fi

cd "$PROJECT_ROOT"
if [[ "${SKIP_DATA_CHECK:-0}" != "1" ]]; then
  PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" scripts/check_ablation_data.py \
    --config "$CONFIG" \
    --dataset-root "$PANOVGGT_ROOT"
fi

if [[ ! -s "$WARMUP_CHECKPOINT" ]]; then
  if [[ "${SKIP_WARMUP:-0}" == "1" ]]; then
    echo "missing shared warmup checkpoint: $WARMUP_CHECKPOINT" >&2
    exit 1
  fi
  mkdir -p "$WARMUP_OUT"
  WARMUP_DURATION_ARGS=()
  if [[ -n "${WARMUP_MAX_DURATION_MINUTES:-}" ]]; then
    WARMUP_DURATION_ARGS=(--max-duration-minutes "$WARMUP_MAX_DURATION_MINUTES")
  fi
  PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NPROC_PER_NODE" \
    training/train_pano_omega.py \
    --config "$WARMUP_CONFIG" \
    "${WARMUP_DURATION_ARGS[@]}" \
    --dataset-root "$PANOVGGT_ROOT" \
    --checkpoint "$BASE_CHECKPOINT" \
    --output-dir "$WARMUP_OUT" \
    --tensorboard-dir "$WARMUP_OUT/tensorboard" \
    --debug-dir "$WARMUP_OUT/debug" \
    --pred-depth-scale "$PRED_DEPTH_SCALE" \
    --no-inherit-checkpoint-training-defaults \
    "${EXTRA_TRAIN_ARGS_ARRAY[@]}" \
    2>&1 | tee -a "$WARMUP_OUT/train.log"
fi

if [[ ! -s "$WARMUP_CHECKPOINT" ]]; then
  echo "shared warmup did not produce: $WARMUP_CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$RUN_OUT"
ABLATION_DURATION_ARGS=()
if [[ -n "${ABLATION_MAX_DURATION_MINUTES:-}" ]]; then
  ABLATION_DURATION_ARGS=(--max-duration-minutes "$ABLATION_MAX_DURATION_MINUTES")
fi
PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  training/train_pano_omega.py \
  --config "$CONFIG" \
  "${ABLATION_DURATION_ARGS[@]}" \
  --dataset-root "$PANOVGGT_ROOT" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --checkpoint "$WARMUP_CHECKPOINT" \
  --output-dir "$RUN_OUT" \
  --tensorboard-dir "$RUN_OUT/tensorboard" \
  --debug-dir "$RUN_OUT/debug" \
  --pred-depth-scale "$PRED_DEPTH_SCALE" \
  --no-inherit-checkpoint-training-defaults \
  "${EXTRA_TRAIN_ARGS_ARRAY[@]}" \
  2>&1 | tee -a "$RUN_OUT/train.log"
