# Getting Started

This guide covers environment setup and the fastest path to a working train → infer loop.

## Prerequisites

- Linux machine with NVIDIA GPU (CUDA)
- Python 3.10+
- Git
- Hugging Face account (for FLUX model weights)

## Clone and Install

```bash
git clone <repo-url> RefCompose
cd RefCompose
```

### Training (FLUX.2 — canvas + depth)

```bash
cd scripts/train
pip install -r requirements-flux2.txt
```

Core packages: `torch`, `diffusers>=0.32`, `transformers`, `accelerate`, `safetensors`, `logzero`.

### Depth estimation (Depth Anything 3)

```bash
cd scripts/infer_v2
pip install -r requirements-da3.txt
```

Or install DA3 from source:

```bash
git clone https://github.com/ByteDance-Seed/Depth-Anything-3
cd Depth-Anything-3
pip install -e .
```

### Dataset preparation

```bash
pip install ultralytics pandas opencv-python-headless pyarrow tqdm pillow
```

YOLO weights (`yolo26n.pt`) are downloaded automatically by Ultralytics on first run.

## Download Base Models

| Model | Use case | Hugging Face ID |
|-------|----------|-----------------|
| FLUX.2-dev | Canvas/depth training & inference | `black-forest-labs/FLUX.2-dev` |
| FLUX.1-dev | Spatial/subject/style JSONL training | `black-forest-labs/FLUX.1-dev` |
| DA3-BASE | Depth maps | `depth-anything/DA3-BASE` |
| Qwen3-VL | Image captioning (stage 6) | per model card |
| YOLO26n | Object detection | auto via ultralytics |

Set environment variables to point at local snapshots:

```bash
export FLUX2_MODEL=/path/to/FLUX.2-dev/snapshot
export LORA_PATH=/path/to/checkpoint/lora.safetensors
```

## Quick Start: Inference Only

If you already have a trained LoRA checkpoint:

```bash
cd scripts/infer_v2

# 1. Generate a base interior image (FLUX.2 text-to-image)
CUDA_VISIBLE_DEVICES=0 ./run.sh

# 2. Estimate depth from the generated image
CUDA_VISIBLE_DEVICES=0 ./run_depth.sh

# 3. Run canvas + depth LoRA refinement
CUDA_VISIBLE_DEVICES=0 ./run_lora.sh
```

Outputs land in `outputs/` and `generated_image/` (or `methodology_image/` depending on script defaults).

## Quick Start: Full Pipeline

```bash
# 1. Prepare dataset (v2 example)
cd scripts/dataset_prep_v2
python extract_hd_crops.py --images-dir /path/to/tars
python yolo_boxes.py
python stage5_depth_generation.py

# 2. Train
cd ../train
CANVAS_DATA_ROOTS=/path/to/dataset_prep_v2/output ./train_flux2_lora.sh

# 3. Infer
cd ../infer_v2
LORA_PATH=/path/to/output/lora_checkpoints_v2/checkpoint-XXXX/lora.safetensors ./run_lora.sh
```

## Multi-GPU Training

Use Accelerate with the provided config:

```bash
cd scripts/train
accelerate launch --config_file default_config.yaml train.py [args...]
```

`default_config.yaml` is set for 2-GPU local training with `fp16` mixed precision. Override `gpu_ids` and `num_processes` as needed.

## Verify Installation

```bash
# Training smoke tests
cd scripts/train/testing
python test_forward_pass.py
python test_dataloader.py

# Depth smoke test
cd ../../infer_v2
python stage2_depth.py -i /path/to/test.png
```

## Common Issues

| Problem | Fix |
|---------|-----|
| `check_min_version("0.32.0")` fails | Upgrade diffusers: `pip install -U diffusers` |
| CUDA OOM during training | Lower `--unified_train_width/height`, enable `--gradient_checkpointing`, reduce `--ranks` |
| YOLO runs on CPU | Set `YOLO_DEVICE=cuda` or ensure CUDA is available |
| FLUX model path not found | Set `FLUX2_MODEL` or `--pretrained_model_name_or_path` to a local snapshot directory |
| Depth PNG format mismatch | Use 16-bit grayscale `I;16` PNGs; see `load_depth_image_as_rgb_pil` in training code |

## Next Steps

- [Architecture](architecture.md) — understand how conditioning works
- [Dataset Preparation](dataset-preparation.md) — build training data
- [Training](training.md) — configure and run training
- [Inference](inference.md) — generate and refine images
