#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${1:-/mnt/f/panovggt/PanoCity_paired}"
OUTPUT_ROOT="${2:-/home/aoki/panovggt/PanoCity_paired}"
COUNT="${3:-100}"

RGB_SRC="${SOURCE_ROOT}/rgb"
DEPTH_SRC="${SOURCE_ROOT}/depth"
RGB_OUT="${OUTPUT_ROOT}/rgb"
DEPTH_OUT="${OUTPUT_ROOT}/depth"
MANIFEST="${OUTPUT_ROOT}/subset_manifest.txt"

if [[ ! -d "${RGB_SRC}" || ! -d "${DEPTH_SRC}" ]]; then
  echo "Expected rgb/ and depth/ under ${SOURCE_ROOT}" >&2
  exit 1
fi

if [[ "$(readlink -f "${SOURCE_ROOT}")" == "$(readlink -f "${OUTPUT_ROOT}")" ]]; then
  echo "Refusing to overwrite source root: ${OUTPUT_ROOT}" >&2
  exit 1
fi

mkdir -p "${RGB_OUT}" "${DEPTH_OUT}"
rm -f "${RGB_OUT}"/* "${DEPTH_OUT}"/*
: > "${MANIFEST}"

copied=0
while IFS= read -r rgb_path; do
  name="$(basename "${rgb_path}")"
  token="${name%%_*}"
  depth_path=""
  candidates=(
    "${DEPTH_SRC}/${name}"
    "${DEPTH_SRC}/${name/_rgb_/_depth_}"
    "${DEPTH_SRC}/${token}_depth_${token}.png"
    "${DEPTH_SRC}/${token}_pano_${token}.png"
  )
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      depth_path="${candidate}"
      break
    fi
  done
  if [[ -z "${depth_path}" ]]; then
    continue
  fi

  cp -p "${rgb_path}" "${RGB_OUT}/${name}"
  cp -p "${depth_path}" "${DEPTH_OUT}/$(basename "${depth_path}")"
  printf '%s\t%s\n' "${rgb_path}" "${depth_path}" >> "${MANIFEST}"
  copied=$((copied + 1))
  if [[ "${copied}" -ge "${COUNT}" ]]; then
    break
  fi
done < <(find "${RGB_SRC}" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | shuf)

if [[ "${copied}" -ne "${COUNT}" ]]; then
  echo "Only copied ${copied} pairs; requested ${COUNT}." >&2
  exit 1
fi

echo "copied_pairs=${copied}"
echo "source_root=${SOURCE_ROOT}"
echo "output_root=${OUTPUT_ROOT}"
echo "manifest=${MANIFEST}"
