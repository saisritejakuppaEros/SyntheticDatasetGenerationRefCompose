#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODEL_DIR="${MODEL_DIR:-/home/jovyan/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev/snapshots/26afe3a78bb242c0a8bb181dcc8937bb16e5c66c}"

MANIFEST_PATH="${MANIFEST_PATH:-/mnt/data0/teja/research_multiref/synthetic_dataset_generation/v2/outputs/dataset_manifest.jsonl}"

DEPTH_KEEP_PROB="${DEPTH_KEEP_PROB:-0.5}"
CANVAS_KEEP_PROB="${CANVAS_KEEP_PROB:-0.5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output/lora_checkpoints_v2_manifest_v1}"

mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/logs"

COMMON_ARGS=(
  --pretrained_model_name_or_path "$MODEL_DIR"
  --dataset_type manifest
  --manifest_path "$MANIFEST_PATH"
  --depth_keep_prob "$DEPTH_KEEP_PROB"
  --canvas_keep_prob "$CANVAS_KEEP_PROB"
  --prompt_warmup_steps 500
  --spatial_column None
  --cond_size 512
  --noise_size 1280
  --unified_train_width 1280
  --unified_train_height 720
  --canvas_unified_resize cover
  --ranks 32
  --network_alphas 32
  --lora_num 2
  --output_dir "$OUTPUT_DIR"
  --logging_dir "$OUTPUT_DIR/logs"
  --mixed_precision bf16
  --gradient_checkpointing
  --learning_rate 1e-4
  --train_batch_size 1
  --gradient_accumulation_steps 4
  --num_train_epochs 100
  --checkpointing_steps 1000
  --validation_steps 500
  --validation_num_samples 4
  --validation_loss_micro_batch_size 1
  --validation_inference_steps 28
  --validation_samples_subdir validation_samples
  --guidance_scale 1.0
  --max_sequence_length 512
  --text_encoder_out_layers 10 20 30
  --dataloader_num_workers 4
)

# NCCL settings (must be set before PyTorch / accelerate launch)
export NCCL_NVLS_ENABLE=0
export NCCL_TREE_THRESHOLD=0
export NCCL_NET_GDR_LEVEL=0
export NCCL_P2P_LEVEL=SYS
export NCCL_SHM_DISABLE=0
export NCCL_ALGO=Ring
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=WARN

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"

accelerate launch --num_processes "$NUM_PROCESSES" --mixed_precision bf16 \
  train.py "${COMMON_ARGS[@]}"
