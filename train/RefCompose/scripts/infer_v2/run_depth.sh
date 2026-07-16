#!/usr/bin/env bash
# Depth Anything 3 for generated_image/image_gen.png (or override via env/args).
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

INPUT_IMAGE="${INPUT_IMAGE:-generated_image/image_gen.png}"
OUTPUT_DEPTH="${OUTPUT_DEPTH:-}"

args=()
if [[ -n "${OUTPUT_DEPTH}" ]]; then
  args+=(-o "${OUTPUT_DEPTH}")
fi

if [[ -d "${INPUT_IMAGE}" ]]; then
  exec python stage2_depth.py -i "${INPUT_IMAGE}" --batch "${args[@]}" "$@"
fi

exec python stage2_depth.py -i "${INPUT_IMAGE}" "${args[@]}" "$@"
