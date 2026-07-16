# Configuration Reference

Complete reference for CLI flags, environment variables, data formats, and config files.

---

## Environment Variables

| Variable | Used in | Description |
|----------|---------|-------------|
| `FLUX2_MODEL` | `infer_v2/`, `train_flux2_lora.sh` | Path to FLUX.2-dev HuggingFace snapshot |
| `LORA_PATH` | `infer_v2/run_lora.sh`, `stage3_lora_infer.py` | Path to trained `lora.safetensors` |
| `CUDA_VISIBLE_DEVICES` | All GPU scripts | GPU device selection |
| `CANVAS_DATA_ROOTS` | `train_flux2_lora.sh` | Comma-separated dataset output directories |
| `CANVAS_DATA_ROOT_1`, `CANVAS_DATA_ROOT_2` | `train_flux2_lora.sh` | Individual dataset roots |
| `DEPTH_KEEP_PROB` | `train_flux2_lora.sh` | Training depth keep probability |
| `CANVAS_KEEP_PROB` | `train_flux2_lora.sh` | Training canvas keep probability |
| `VARIANTS` | `infer_v2/run.sh` | Number of T2I variants |
| `SEED` | `infer_v2/run.sh` | Base random seed |
| `STEPS` | `infer_v2/run.sh` | Inference denoise steps |
| `GUIDANCE` | `infer_v2/run.sh` | CFG guidance scale |
| `WIDTH`, `HEIGHT` | `infer_v2/run.sh` | Output dimensions |
| `LORA_FIRST_DEPTH_STEPS` | `infer_v2/run_lora.sh` | Depth-priority denoise steps |
| `YOLO_DEVICE` | `dataset_prep/stage2_yolo_inferences.py` | `cuda` or `cpu` for YOLO |

---

## Accelerate Config

**File:** `scripts/train/default_config.yaml`

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
mixed_precision: fp16
num_processes: 2
gpu_ids: all
main_process_port: 14121
```

Override for single GPU:

```yaml
num_processes: 1
gpu_ids: "0"
mixed_precision: bf16
```

---

## Training CLI (`train.py`)

### Model

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--pretrained_model_name_or_path` | str | — | FLUX.1 or FLUX.2 model path |
| `--pretrained_lora_path` | str | None | Resume from existing LoRA |
| `--revision` | str | None | HF model revision |
| `--variant` | str | None | Model variant (e.g. fp16) |

### Dataset Mode

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dataset_type` | str | `jsonl` | `jsonl` or `canvas` |
| `--train_data_dir` | str | — | JSONL file or folder (jsonl mode) |
| `--csv_path` | str | — | Single CSV (canvas mode) |
| `--canvas_data_roots` | str | — | Comma-separated output directories |

### JSONL Columns

| Flag | Default | Description |
|------|---------|-------------|
| `--spatial_column` | `None` | Conditioning image column |
| `--subject_column` | `image` | Subject image column |
| `--target_column` | `image` | Target image column |
| `--caption_column` | `caption_left,caption_right` | Prompt column(s) |

### Canvas Columns

| Flag | Default | Description |
|------|---------|-------------|
| `--canvas_column` | `canvas_path` | Pre-rendered canvas path |
| `--canvas_target_column` | `image_path` | Target image path |
| `--canvas_prompt_column` | `prompt` | Caption column |
| `--canvas_image_root` | — | Prefix for relative paths |
| `--depth_image_root` | — | Depth PNG directory |

### Canvas Behavior

| Flag | Default | Description |
|------|---------|-------------|
| `--canvas_conditioning` | — | `bbox_multiview` or other modes |
| `--canvas_bbox_min_side` | 0 | Min bbox side to include |
| `--canvas_multiview_match_min_side` | 200 | Min side for multiview match |
| `--canvas_multiview_prob` | 0.5 | Multiview vs direct crop probability |
| `--canvas_background` | `black` | `black` or `scaled` |
| `--canvas_unified_resize` | — | `cover` or `contain` |
| `--canvas_balanced_concat_sampling` | true | Equal sampling across roots |
| `--canvas_augment` | true | Enable canvas augmentations |
| `--depth_keep_prob` | 0.5 | Real depth probability |
| `--canvas_keep_prob` | 0.5 | Real canvas probability |
| `--canvas_depth_coarse_augment` | true | Depth blur/noise/patch aug |
| `--canvas_depth_coarse_blur_prob` | 0.7 | Blur probability |
| `--canvas_depth_coarse_noise_prob` | 0.4 | Noise probability |
| `--canvas_depth_coarse_patch_prob` | 0.3 | Patch dropout probability |

### Resolution

| Flag | Default | Description |
|------|---------|-------------|
| `--cond_size` | 512 | Condition image longest side |
| `--noise_size` | 1280 | Target max side (jsonl) |
| `--unified_train_width` | — | Canvas mode width |
| `--unified_train_height` | — | Canvas mode height |
| `--test_h`, `--test_w` | 1024 | Validation image size |

### LoRA

| Flag | Default | Description |
|------|---------|-------------|
| `--lora_num` | 1 | Number of cond LoRA blocks |
| `--ranks` | — | LoRA rank |
| `--network_alphas` | — | LoRA alpha |

### Optimization

| Flag | Default | Description |
|------|---------|-------------|
| `--learning_rate` | — | AdamW LR |
| `--train_batch_size` | 1 | Batch size |
| `--gradient_accumulation_steps` | 1 | Grad accumulation |
| `--num_train_epochs` | — | Total epochs |
| `--max_train_steps` | — | Alternative to epochs |
| `--lr_scheduler` | — | Scheduler type |
| `--lr_warmup_steps` | — | Warmup steps |
| `--prompt_warmup_steps` | — | Prompt encoder warmup |
| `--mixed_precision` | — | `no`, `fp16`, `bf16` |
| `--gradient_checkpointing` | false | Save VRAM |
| `--seed` | — | Random seed |

### Checkpointing & Validation

| Flag | Default | Description |
|------|---------|-------------|
| `--output_dir` | — | Checkpoint output directory |
| `--logging_dir` | — | TensorBoard logs |
| `--checkpointing_steps` | — | Save every N steps |
| `--validation_steps` | — | Validate every N steps |
| `--validation_num_samples` | — | Batches to validate |
| `--validation_inference_steps` | 28 | Denoise steps in validation |
| `--validation_samples_subdir` | — | PNG output subfolder |
| `--num_validation_images` | — | Images per validation (jsonl) |
| `--validation_prompt` | — | Fixed validation prompt (jsonl) |
| `--spatial_test_images` | — | Validation spatial images |
| `--subject_test_images` | — | Validation subject images |

### Text Encoding (FLUX.2)

| Flag | Default | Description |
|------|---------|-------------|
| `--max_sequence_length` | 512 | Max token length |
| `--text_encoder_out_layers` | — | Mistral hidden layer indices |

---

## Inference CLI

### `flux2_heritage_infer.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | `HERITAGE_INTERIOR_PROMPT` | Generation prompt |
| `--prompt-file` | — | Load prompt from file |
| `--variants` | 4 | Number of outputs |
| `--single` | false | Single image mode |
| `--seed` | 42 | Base seed |
| `--seed-step` | 1 | Seed increment per variant |
| `--num-inference-steps` | 50 | Denoise steps |
| `--guidance-scale` | 4.0 | CFG |
| `--width` | 1280 | Output width |
| `--height` | 720 | Output height |
| `-o` / `--output` | `outputs/heritage_interior.png` | Output path template |

### `stage2_depth.py`

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | `generated_image/image_gen.png` | Input image or folder |
| `-o` / `--output` | auto | Output path or directory |
| `--batch` | false | Process all images in folder |
| `--model` | `depth-anything/DA3-BASE` | DA3 model ID |
| `--process-res` | 504 | DA3 processing resolution |

### `stage3_lora_infer.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--lora_path` | `LORA_PATH` env | LoRA checkpoint |
| `--canvas` | `black_canvas.jpg` | Canvas input |
| `--depth` | `image_gen_depth.png` | Depth input |
| `--prompt-file` | `image_gen.txt` | Prompt file |
| `--lora_first_depth_steps` | 0 | Depth-priority steps |
| `--no-canvas_depth_coarse_augment` | false | Skip depth degradation |
| `--guidance-scale` | 4.0 | CFG |
| `--num-inference-steps` | 50 | Denoise steps |
| `--seed` | 42 | Random seed |
| `-o` / `--output` | `outputs/image_gen_lora.png` | Output path |

### `lora_server.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 8767 | HTTP port |
| `--host` | `0.0.0.0` | Bind address |
| `--lora_path` | env default | LoRA checkpoint |

---

## Data Formats

### JSONL Training Record

```json
{
  "source": "relative/path/to/condition.png",
  "target": "relative/path/to/target.png",
  "caption": "Text description of the target image."
}
```

### YOLO Detection CSV

```csv
image_path,x1,y1,x2,y2,confidence,class_id,class_name
images/abc123.png,100,200,300,400,0.87,0,person
images/abc123.png,500,100,700,350,0.92,56,chair
```

### Depth PNG

- Format: PNG, mode `I;16`
- Values: uint16 in [0, 65535]
- One file per image, same basename as RGB source
- Relative depth (not metric); percentile-normalized per image

### Caption Text File

- One file per image: `image_captions/{id}.txt`
- Plain text, ~100 words
- Dense visual description suitable as a text-to-image prompt
- No meta-framing ("The image shows…")

### LoRA Checkpoint

- Format: `safetensors`
- Filename: `lora.safetensors`
- Location: `checkpoint-{step}/lora.safetensors`
- Keys: LoRA weight matrices for attention Q/K/V projections

---

## Shell Script Environment

### `train_flux2_lora.sh`

```bash
MODEL_DIR=...                    # FLUX.2-dev path
CANVAS_DATA_ROOTS=root1,root2    # Dataset outputs
DEPTH_KEEP_PROB=0.5
CANVAS_KEEP_PROB=0.5
OUTPUT_DIR=./output/lora_checkpoints_v2
CUDA_VISIBLE_DEVICES=1
```

### `infer_v2/run.sh`

```bash
VARIANTS=4
SEED=42
STEPS=50
GUIDANCE=4.0
WIDTH=1280
HEIGHT=720
FLUX2_MODEL=...
```

### `infer_v2/run_lora.sh`

```bash
LORA_PATH=...
LORA_FIRST_DEPTH_STEPS=0
CUDA_VISIBLE_DEVICES=0
```

---

## Requirements Files

| File | Packages |
|------|----------|
| `scripts/train/requirements-flux2.txt` | torch, diffusers≥0.32, transformers, accelerate, safetensors, logzero |
| `scripts/infer_v2/requirements-da3.txt` | depth-anything-3 |

---

## Gitignore Patterns

From `.gitignore`:

```
*.pyc
train/output
dataset_prep_v2/output/
dataset_prep_v2/yolo26n.pt
infer/flux_server/outputs_*
```

Output directories and downloaded weights are excluded from version control.
