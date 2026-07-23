#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/e/PanoVGGT_minimal_datasets/datasets/Structured3D}"
PARTS="${PARTS:-00 01 02 03 04 05 06 07 08 09 10 11 12 13 14}"
JOBS="${JOBS:-5}"
KEEP_ZIPS="${KEEP_ZIPS:-0}"
REBUILD_CACHE="${REBUILD_CACHE:-1}"
MIXED4_ROOT="${MIXED4_ROOT:-$(dirname "$ROOT")}"
BASE_URL="${BASE_URL:-https://zju-kjl-jointlab-azure.kujiale.com/Structured3D}"

DOWNLOAD_DIR="$ROOT/.downloads"
LOG_DIR="$ROOT/cache"
mkdir -p "$DOWNLOAD_DIR" "$LOG_DIR"

download_one() {
  local part="$1"
  local zip="$DOWNLOAD_DIR/Structured3D_panorama_${part}.zip"
  local done_marker="$zip.ok"
  local url="$BASE_URL/Structured3D_panorama_${part}.zip"

  if [[ -f "$done_marker" && -s "$zip" ]]; then
    echo "[download] part $part already complete"
    return 0
  fi

  echo "[download] part $part -> $zip"
  curl -L --fail --retry 20 --retry-delay 10 --retry-all-errors -C - -o "$zip" "$url"
  unzip -tq "$zip" >/dev/null
  touch "$done_marker"
  echo "[download] part $part verified"
}

extract_one() {
  local part="$1"
  local zip="$DOWNLOAD_DIR/Structured3D_panorama_${part}.zip"
  local done_marker="$zip.extracted"

  if [[ -f "$done_marker" ]]; then
    echo "[extract] part $part already extracted"
    return 0
  fi
  if [[ ! -s "$zip" ]]; then
    echo "[extract] missing zip for part $part: $zip" >&2
    return 1
  fi

  local first_entry
  first_entry="$(zipinfo -1 "$zip" | head -n 1 || true)"
  echo "[extract] part $part first_entry=$first_entry"
  if [[ "$first_entry" == Structured3D/* ]]; then
    unzip -n "$zip" -d "$(dirname "$ROOT")"
  else
    unzip -n "$zip" -d "$ROOT"
  fi
  touch "$done_marker"
  if [[ "$KEEP_ZIPS" != "1" ]]; then
    rm -f "$zip"
  fi
  echo "[extract] part $part complete"
}

export ROOT DOWNLOAD_DIR BASE_URL KEEP_ZIPS
export -f download_one

echo "[start] $(date -Is)"
echo "[config] ROOT=$ROOT"
echo "[config] PARTS=$PARTS"
echo "[config] JOBS=$JOBS"
echo "[config] KEEP_ZIPS=$KEEP_ZIPS"
echo "[config] REBUILD_CACHE=$REBUILD_CACHE"
echo "[space-before]"
df -h "$ROOT" || true

printf '%s\n' $PARTS | xargs -n 1 -P "$JOBS" bash -lc 'download_one "$@"' _

echo "[download-all-complete] $(date -Is)"
for part in $PARTS; do
  extract_one "$part"
done

echo "[scene-count] $(find "$ROOT" -maxdepth 1 -type d -name 'scene_*' | wc -l)"
echo "[space-after-extract]"
df -h "$ROOT" || true

if [[ "$REBUILD_CACHE" == "1" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  echo "[cache] rebuilding mixed4 indexes under $MIXED4_ROOT"
  python "$SCRIPT_DIR/build_mixed4_official_indexes.py" --root "$MIXED4_ROOT" --structured3d-train-source auto --seed 42
fi

echo "[done] $(date -Is)"
