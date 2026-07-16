# Training

RefCompose supports two training modes in `scripts/train/train.py`:

1. **JSONL mode** — FLUX.1 spatial/subject/style control (classic EasyControl-style)
2. **Canvas mode** — FLUX.2 canvas + depth composition (primary RefCompose workflow)

## Environment

```bash
cd scripts/train
pip install -r requirements-flux2.txt
```

Recommended: Python 3.10+, PyTorch 2.1+, CUDA, `diffusers>=0.32`.

## Canvas Training (FLUX.2)

### One-Command Start

```bash
cd scripts/train
CANVAS_DATA_ROOTS=/path/to/output ./train_flux2_lora.sh
```

Default script settings (`train_flux2_lora.sh`):

| Parameter | Value |
|-----------|-------|
| Base model | FLUX.2-dev |
| Dataset type | `canvas` |
| LoRA rank | 32 |
| LoRA count | 2 (canvas + depth) |
| Unified resolution | 1280×720 (cover crop) |
| Mixed precision | bf16 |
| Learning rate | 1e-4 |
| Batch size | 1 |

### Manual Invocation

```bash
python train.py \
  --pretrained_model_name_or_path /path/to/FLUX.2-dev \
  --dataset_type canvas \
  --canvas_data_roots /path/to/dataset/output \
  --canvas_conditioning bbox_multiview \
  --depth_keep_prob 0.5 \
  --canvas_keep_prob 0.5 \
  --lora_num 2 \
  --ranks 32 \
  --network_alphas 32 \
  --unified_train_width 1280 \
  --unified_train_height 720 \
  --canvas_unified_resize cover \
  --output_dir ./output/lora_checkpoints_v2 \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --learning_rate 1e-4 \
  --num_train_epochs 1000 \
  --checkpointing_steps 1000 \
  --validation_steps 100
```

### Multiple Dataset Roots

```bash
CANVAS_DATA_ROOTS="/data/root1,/data/root2" ./train_flux2_lora.sh
```

Each root must have the standard `output/` layout. Balanced sampling draws equally from each root regardless of CSV length (`--canvas_balanced_concat_sampling`, default on).

### Checkpoints

Saved under `--output_dir`:

```
output/lora_checkpoints_v2/
├── checkpoint-1000/
│   └── lora.safetensors
├── checkpoint-2000/
│   └── lora.safetensors
└── logs/                  # TensorBoard
```

### Validation

During training, dataloader validation runs every `--validation_steps`:

- Samples real batches from the training dataloader
- Runs partial denoise (`--validation_inference_steps`, default 28)
- Saves PNGs to `--validation_samples_subdir`
- Logs validation loss

Implementation: `src/flux2_dataloader_validation.py`

---

## JSONL Training (FLUX.1)

For spatial (pose/canny/depth), subject, or style control on FLUX.1-dev.

### Data Format

JSONL with one JSON object per line:

```json
{"source": "./path/to/condition.png", "caption": "A girl wearing a green coat.", "target": "./path/to/target.png"}
```

Examples: `scripts/train/examples/pose.jsonl`, `subject.jsonl`, `style.jsonl`

| Column | Role |
|--------|------|
| `source` | Conditioning image (spatial or subject) |
| `target` | Ground-truth output image |
| `caption` | Text prompt |

### Shell Scripts

```bash
# Spatial / pose control
bash train_spatial.sh

# Subject control
bash train_subject.sh

# Style control
bash train_style.sh
```

Each script sets column flags:

| Script | `--spatial_column` | `--subject_column` |
|--------|-------------------|-------------------|
| `train_spatial.sh` | `source` | `None` |
| `train_subject.sh` | `None` | `source` |
| `train_style.sh` | `source` | `None` |

### Multi-GPU

```bash
accelerate launch --config_file default_config.yaml train.py [args...]
```

---

## Key Training Parameters

### Resolution

| Flag | Default | Description |
|------|---------|-------------|
| `--cond_size` | 512 | Conditioning image longest side |
| `--noise_size` | 1280 | Target image max side (JSONL mode) |
| `--unified_train_width` | — | Canvas mode width (multiple of 16) |
| `--unified_train_height` | — | Canvas mode height (multiple of 16) |
| `--canvas_unified_resize` | `cover` | `cover` or `contain` |

### LoRA

| Flag | Default | Description |
|------|---------|-------------|
| `--ranks` | 128 | LoRA rank |
| `--network_alphas` | 128 | LoRA alpha scaling |
| `--lora_num` | 1 | Number of cond LoRA blocks (auto-bumped to 2 with depth) |

### Canvas Augmentation

| Flag | Default | Description |
|------|---------|-------------|
| `--depth_keep_prob` | 0.5 | Probability of using real depth (vs black) |
| `--canvas_keep_prob` | 0.5 | Probability of using real canvas (vs black) |
| `--canvas_depth_coarse_augment` | true | Blur/noise/patch dropout on depth |
| `--canvas_multiview_prob` | 0.5 | Use multiview crop vs direct bbox crop |

### Optimization

| Flag | Default | Description |
|------|---------|-------------|
| `--learning_rate` | 1e-4 | AdamW learning rate |
| `--train_batch_size` | 1 | Fixed at 1 for variable resolutions |
| `--gradient_accumulation_steps` | 1 | Effective batch size multiplier |
| `--mixed_precision` | — | `bf16`, `fp16`, or `no` |
| `--gradient_checkpointing` | false | Trade compute for VRAM |
| `--num_train_epochs` | — | Total epochs |
| `--checkpointing_steps` | — | Save interval (steps) |

---

## Source Modules

| Module | Role |
|--------|------|
| `train.py` | Main training loop, argparse, Accelerate integration |
| `src/canvas_dataset.py` | `CanvasSceneDataset`, augmentations, collate |
| `src/canvas_bbox_compose.py` | On-the-fly canvas composition |
| `src/flux2_transformer_cond.py` | Conditional FLUX.2 transformer |
| `src/layers_flux2.py` | LoRA attention processors |
| `src/flux2_train_helpers.py` | Latent encoding utilities |
| `src/flux2_dataloader_validation.py` | Validation denoise + loss |
| `src/prompt_helper.py` | FLUX.2 text encoding (Mistral3/Pixtral) |
| `src/jsonl_datasets.py` | JSONL dataset for FLUX.1 mode |
| `src/lora_helper.py` | LoRA load/save for FLUX.1 inference |
| `src/pipeline.py` | Custom `FluxPipeline` for FLUX.1 |
| `src/transformer_flux.py` | FLUX.1 transformer wrapper |

---

## FLUX.1 Inference

After JSONL training, use the custom pipeline:

```python
import torch
from PIL import Image
from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.lora_helper import set_single_lora, set_multi_lora

def clear_cache(transformer):
    for _, attn_processor in transformer.attn_processors.items():
        attn_processor.bank_kv.clear()

device = "cuda"
base_path = "black-forest-labs/FLUX.1-dev"
pipe = FluxPipeline.from_pretrained(base_path, torch_dtype=torch.bfloat16)
transformer = FluxTransformer2DModel.from_pretrained(
    base_path, subfolder="transformer", torch_dtype=torch.bfloat16
)
pipe.transformer = transformer
pipe.to(device)

set_single_lora(pipe.transformer, "path/to/lora.safetensors", lora_weights=[1], cond_size=512)

control_image = Image.open("condition.png")
image = pipe(
    "your prompt",
    height=1024, width=1024,
    guidance_scale=3.5,
    num_inference_steps=25,
    spatial_images=[control_image],  # or subject_images=[...]
    subject_images=[],
    cond_size=512,
).images[0]
clear_cache(pipe.transformer)
image.save("output.png")
```

For multi-LoRA (e.g. subject + inpainting):

```python
set_multi_lora(pipe.transformer, [path1, path2], lora_weights=[[1], [1]], cond_size=512)
```

---

## Testing

```bash
cd scripts/train/testing
python test_forward_pass.py          # Model forward smoke test
python test_dataloader.py            # Dataloader batch shapes
python test_validation_loop_smoke.py # Validation loop
python test_training_flow_documented.py
```

---

## Dataset Preparation Helper

`prepare_dataset.py` wraps an external layout tool to produce bbox CSVs:

```bash
python prepare_dataset.py \
  --layout_command "python /path/to/generate_layout.py ..." \
  --merge_csv ./detections.csv \
  --output_csv ./dataset_with_layout.csv
```

---

## Next Steps

- [Inference](inference.md) — use trained checkpoints with `stage3_lora_infer.py`
- [Configuration](configuration.md) — full CLI reference
