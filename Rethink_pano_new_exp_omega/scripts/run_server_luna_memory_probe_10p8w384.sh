#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python}"
# Do not honor the generic CONFIG variable: long-running server shells often
# retain it from another experiment. Use the probe-specific override instead.
CONFIG="${MEMORY_PROBE_CONFIG:-configs/multipano_rtx5000x4_panocity_luna_memory_probe_10p_8w384_fov90_pitch30_erp1024x512.yaml}"
DATASET_ROOT="${DATASET_ROOT:-/whitehole/AOKI/panovggt}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
WARMUP_CHECKPOINT="${1:-${WARMUP_CHECKPOINT:-}}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/server_luna_memory_probe_10p_8w384_fov90_pitch30_erp1024x512_${RUN_TAG}}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "$WARMUP_CHECKPOINT" || ! -s "$WARMUP_CHECKPOINT" ]]; then
  echo "usage: bash scripts/run_server_luna_memory_probe_10p8w384.sh /absolute/path/to/warmup/last.pt" >&2
  exit 2
fi
if [[ "$PYTHON" != */* ]]; then
  PYTHON="$(command -v "$PYTHON" || true)"
fi
if [[ -z "$PYTHON" || ! -x "$PYTHON" ]]; then
  echo "[memory-probe] python executable not found; activate the conda environment or set PYTHON=/absolute/path/to/python" >&2
  exit 2
fi

$PYTHON - "$CONFIG" <<'PY'
import sys
import yaml

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
data = config.get("data") or {}
sampler = config.get("sampler") or {}
stages = (config.get("optimization") or {}).get("training_stages") or []
stage = stages[0] if len(stages) == 1 else {}
pitch_values = [value.strip() for value in str(sampler.get("pitch_degrees", "")).split(",") if value.strip()]
checks = {
    "data.pano_min_count": (data.get("pano_min_count"), 10),
    "data.pano_max_count": (data.get("pano_max_count"), 10),
    "sampler.window_size": (sampler.get("window_size"), 384),
    "sampler.num_yaw": (sampler.get("num_yaw"), 4),
    "sampler.pitch_count": (len(pitch_values), 2),
    "stage.pano_min_count": (stage.get("pano_min_count"), 10),
    "stage.pano_max_count": (stage.get("pano_max_count"), 10),
    "stage.window_size": (stage.get("window_size"), 384),
}
errors = [f"{key}={actual!r}, expected {expected!r}" for key, (actual, expected) in checks.items() if actual != expected]
if errors:
    raise SystemExit("[memory-probe] invalid 10p/8-window probe config: " + "; ".join(errors))
print(f"[memory-probe] preflight verified: 10 panos x {sampler['num_yaw']} yaw x {len(pitch_values)} pitch = 80 views/sample")
PY

# Warmup checkpoints use trainable_delta format. Read the recorded foundation
# reference so a relocated server checkout can resolve it automatically.
FOUNDATION_REFERENCE="$($PYTHON - "$WARMUP_CHECKPOINT" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
args = payload.get("args") or {}
if not isinstance(args, dict):
    args = vars(args)
for value in (
    payload.get("foundation_checkpoint"),
    payload.get("base_checkpoint"),
    args.get("base_checkpoint"),
    args.get("checkpoint"),
):
    if value:
        print(value)
        break
PY
)"

if [[ -z "$BASE_CHECKPOINT" ]]; then
  candidates=(
    /whitehole/AOKI/vggt-omega/ckpt/vggt_omega_1b_512.pt \
    /whitehole/AOKI/RethinkPanoVGGT_omega/ckpt/vggt_omega_1b_512.pt \
    /whitehole/AOKI/RethinkPanoVGGT-omega/ckpt/vggt_omega_1b_512.pt \
    /whitehole/AOKI/RethinkPanoVGGT_omega_compare_methods_only/ckpt/VGGT-Omega/vggt_omega_1b_512.pt \
    /whitehole/AOKI/RethinkPanoVGGT_multipano/ckpt/vggt_omega_1b_512.pt \
    "$ROOT/../ckpt/VGGT-Omega/vggt_omega_1b_512.pt" \
    "$ROOT/../ckpt/vggt_omega_1b_512.pt" \
    "$ROOT/ckpt/vggt_omega_1b_512.pt" \
    /home/aoki/RethinkPanoVGGT_omega/ckpt/vggt_omega_1b_512.pt
  )
  if [[ "$FOUNDATION_REFERENCE" == /home/aoki/* ]]; then
    candidates=("/whitehole/AOKI/${FOUNDATION_REFERENCE#/home/aoki/}" "${candidates[@]}")
  fi
  for candidate in "${candidates[@]}"; do
    if [[ -s "$candidate" ]]; then
      BASE_CHECKPOINT="$candidate"
      break
    fi
  done
fi
if [[ -z "$BASE_CHECKPOINT" ]]; then
  checkpoint_name="${FOUNDATION_REFERENCE##*/}"
  checkpoint_name="${checkpoint_name:-vggt_omega_1b_512.pt}"
  BASE_CHECKPOINT="$(find /whitehole/AOKI -maxdepth 6 -type f -name "$checkpoint_name" -size +100M -print -quit 2>/dev/null || true)"
fi
if [[ -z "$BASE_CHECKPOINT" || ! -s "$BASE_CHECKPOINT" ]]; then
  echo "[memory-probe] warmup checkpoint is a trainable_delta, not a standalone model." >&2
  echo "[memory-probe] recorded foundation: ${FOUNDATION_REFERENCE:-<missing>}" >&2
  echo "[memory-probe] foundation checkpoint was not found under /whitehole/AOKI." >&2
  echo "[memory-probe] set BASE_CHECKPOINT=/absolute/path/vggt_omega_1b_512.pt and rerun the same command." >&2
  exit 2
fi
if [[ ! -d "$DATASET_ROOT/Panocity" ]]; then
  echo "[memory-probe] Panocity dataset not found under $DATASET_ROOT" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

# The delta records a /home/aoki/... foundation path. Mirror that path below a
# temporary storage root so recursive checkpoint loading resolves the server file.
if [[ "$FOUNDATION_REFERENCE" == /home/aoki/* ]]; then
  foundation_relative="${FOUNDATION_REFERENCE#/home/aoki/}"
  checkpoint_root="$OUTPUT_DIR/checkpoint_root"
  mkdir -p "$checkpoint_root/$(dirname "$foundation_relative")"
  ln -sfn "$BASE_CHECKPOINT" "$checkpoint_root/$foundation_relative"
  export AOKI_STORAGE_ROOT="$checkpoint_root"
fi

LOG="$OUTPUT_DIR/train_console.log"
{
  echo "[memory-probe] started $(date --iso-8601=seconds)"
  echo "[memory-probe] config=$CONFIG"
  echo "[memory-probe] checkpoint=$WARMUP_CHECKPOINT"
  echo "[memory-probe] checkpoint_foundation_reference=${FOUNDATION_REFERENCE:-<missing>}"
  echo "[memory-probe] base_checkpoint=$BASE_CHECKPOINT"
  echo "[memory-probe] aoki_storage_root=${AOKI_STORAGE_ROOT:-<unset>}"
  echo "[memory-probe] dataset_root=$DATASET_ROOT"
  echo "[memory-probe] output_dir=$OUTPUT_DIR"
  echo "[memory-probe] nproc_per_node=$NPROC_PER_NODE"
  echo "[memory-probe] input=10 panos x ERP 1024x512"
  echo "[memory-probe] sampler=4 yaw x pitch(-30,+30), 384x384, FoV=90deg; total=8 windows/pano, 80 windows/sample"
  echo "[memory-probe] duration_minutes=${DURATION_MINUTES:-10}"
} | tee "$LOG"

set +e
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  training/train_pano_omega.py \
  --config "$CONFIG" \
  --dataset-root "$DATASET_ROOT" \
  --checkpoint "$WARMUP_CHECKPOINT" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --tensorboard-dir "$OUTPUT_DIR/tensorboard" \
  --debug-dir "$OUTPUT_DIR/debug" \
  --max-duration-minutes "${DURATION_MINUTES:-10}" \
  --no-inherit-checkpoint-training-defaults \
  2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e

if [[ "$status" -ne 0 ]]; then
  echo "[memory-probe] failed with status=$status; output preserved at $OUTPUT_DIR" | tee -a "$LOG"
  exit "$status"
fi
echo "[memory-probe] completed $(date --iso-8601=seconds)" | tee -a "$LOG"
