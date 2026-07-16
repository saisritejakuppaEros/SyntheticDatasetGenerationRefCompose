#!/usr/bin/env bash
# Sweep guidance_scale for stage3_lora_infer.py (1 … 4.5 by default).
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

args=(--no-canvas_depth_coarse_augment)

if [[ -n "${LORA_PATH:-}" ]]; then
  args+=(--lora_path "${LORA_PATH}")
fi
if [[ -n "${OUTPUT:-}" ]]; then
  args+=(--output "${OUTPUT}")
fi
if [[ -n "${SEED:-}" ]]; then
  args+=(--seed "${SEED}")
fi

exec python run_guidance_sweep.py "${args[@]}" "$@"
