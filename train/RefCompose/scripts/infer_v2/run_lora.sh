#!/usr/bin/env bash
# FLUX.2 + canvas/depth LoRA for infer_v2 (black canvas + DA3 depth + image_gen.txt).
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LORA_PATH="${LORA_PATH:-}"

args=()
# Sharp depth from stage2_depth.py (skip training-style random blur/noise).
args+=(--no-canvas_depth_coarse_augment)

if [[ -n "${LORA_PATH}" ]]; then
  args+=(--lora_path "${LORA_PATH}")
fi
if [[ -n "${CANVAS_IMAGE:-}" ]]; then
  args+=(--canvas_image "${CANVAS_IMAGE}")
fi
if [[ -n "${DEPTH_IMAGE:-}" ]]; then
  args+=(--depth_image "${DEPTH_IMAGE}")
fi
if [[ -n "${PROMPT_FILE:-}" ]]; then
  args+=(--prompt-file "${PROMPT_FILE}")
fi
if [[ -n "${OUTPUT:-}" ]]; then
  args+=(--output "${OUTPUT}")
fi
if [[ -n "${SEED:-}" ]]; then
  args+=(--seed "${SEED}")
fi
if [[ -n "${STEPS:-}" ]]; then
  args+=(--num_inference_steps "${STEPS}")
fi
if [[ "${LORA_FIRST_DEPTH_STEPS:-0}" != "0" ]]; then
  args+=(--lora_first_depth_steps "${LORA_FIRST_DEPTH_STEPS}")
fi

exec python stage3_lora_infer.py "${args[@]}" "$@"
