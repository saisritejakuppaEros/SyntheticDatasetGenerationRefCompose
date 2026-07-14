#!/usr/bin/env bash
# Launch flux2_imagegen.py across 6 GPUs in parallel.
#
# GPU mapping (1-indexed physical GPU ids):
#   shard 0 -> GPU 1
#   shard 1 -> GPU 2
#   shard 2 -> GPU 3
#   shard 3 -> GPU 4
#   shard 4 -> GPU 5
#   shard 5 -> GPU 6
#
# To use GPUs 2-7 instead, change GPUS=(2 3 4 5 6 7) below.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NUM_SHARDS=6
GPUS=(1 2 3 4 5 6)

MANIFEST="${MANIFEST:-outputs/theme_prompts/manifest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/theme_images}"
EXTRA_ARGS=("$@")

PIDS=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPUS[$shard]}"
  echo "Starting shard ${shard}/${NUM_SHARDS} on GPU ${gpu} ..."
  CUDA_VISIBLE_DEVICES="$gpu" python flux2_imagegen.py \
    --manifest "$MANIFEST" \
    --output_dir "$OUTPUT_DIR" \
    --shard_id "$shard" \
    --num_shards "$NUM_SHARDS" \
    --device "cuda:0" \
    "${EXTRA_ARGS[@]}" &
  PIDS+=($!)
done

echo "Waiting for ${#PIDS[@]} shards ..."
for pid in "${PIDS[@]}"; do
  wait "$pid"
done

echo "All shards finished."
