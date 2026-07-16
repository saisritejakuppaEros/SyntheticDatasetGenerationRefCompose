# RefCompose Documentation

RefCompose is the official codebase for **reference-guided image composition** with diffusion models. The project trains LoRA adapters on FLUX (1.x and 2.x) so a model can generate images conditioned on spatial layouts, subject references, style cues, and canvas/depth maps.

## What RefCompose Does

The pipeline has four main stages:

1. **Dataset preparation** — Extract images, detect objects (YOLO), generate multiview crops, build layout canvases, estimate depth, and caption scenes.
2. **Training** — Fine-tune FLUX/FLUX.2 transformers with LoRA on canvas + depth conditioning (or classic JSONL spatial/subject/style control).
3. **Inference** — Generate heritage interiors, run depth estimation, and apply trained LoRA for canvas-guided refinement.
4. **Object editing** — Experimental scripts for replacing or adding objects in generated scenes (Grounding DINO + compositing).

## Repository Layout

```
RefCompose/
├── docs/                    # This documentation
├── scripts/
│   ├── dataset_prep/      # v1 dataset pipeline (parquet → images → YOLO → multiview → canvas → depth → captions)
│   ├── dataset_prep_v2/   # v2 pipeline (tar archives → HD crops → YOLO → depth)
│   ├── train/             # FLUX / FLUX.2 LoRA training
│   ├── infer_v2/          # FLUX.2 inference (heritage, depth, LoRA, server)
│   └── infer/             # Object replacement / compositing experiments
└── readme.md
```

## Documentation Index

| Document | Description |
|----------|-------------|
| [Getting Started](getting-started.md) | Environment setup, dependencies, and quick-start commands |
| [Architecture](architecture.md) | Model design: token streams, attention masks, LoRA, flow-matching loss |
| [Dataset Preparation](dataset-preparation.md) | End-to-end data pipeline (stages 1–6, v1 vs v2) |
| [Training](training.md) | JSONL and canvas training modes, configs, validation |
| [Inference](inference.md) | Heritage generation, depth, LoRA inference, server API |
| [API Reference](api-reference.md) | Key Python modules and functions |
| [Configuration](configuration.md) | CLI flags, environment variables, data formats |

## Typical Workflow

```mermaid
flowchart LR
    A[Raw data] --> B[dataset_prep / v2]
    B --> C[output/ tree]
    C --> D[train.py]
    D --> E[lora.safetensors]
    E --> F[stage3_lora_infer.py]
    F --> G[Refined image]
```

**Canvas + depth workflow (recommended):**

1. Run `dataset_prep_v2` (or `dataset_prep`) to produce `output/images`, `bbox_results/`, `depth/`, `image_captions/`, `multiview_out/`.
2. Train with `scripts/train/train_flux2_lora.sh` pointing `--canvas_data_roots` at your output folder(s).
3. Infer with `scripts/infer_v2/stage3_lora_infer.py` using a canvas image, depth map, and prompt.

**Classic control workflow (FLUX.1):**

1. Prepare a JSONL file with `source`, `target`, and `caption` columns (see `scripts/train/examples/`).
2. Train with `train_spatial.sh`, `train_subject.sh`, or `train_style.sh`.
3. Run inference via `FluxPipeline` and `lora_helper.py` (see [Training](training.md#flux1-inference)).

## Requirements at a Glance

| Component | Python | Key deps |
|-----------|--------|----------|
| Training (FLUX.2) | 3.10+ | PyTorch 2.1+, diffusers ≥0.32, accelerate |
| Training (FLUX.1) | 3.10+ | PyTorch 2.5+, diffusers, accelerate |
| Depth (DA3) | 3.10+ | `depth-anything-3`, OpenCV |
| Dataset prep | 3.10+ | ultralytics, pandas, PIL |
| LoRA server | 3.10+ | fastapi, uvicorn |

See [Getting Started](getting-started.md) for install commands.

## Related Reading

- In-repo architecture deep-dive: `scripts/train/doc.md`
- Training quick guide: `scripts/train/readme.md`
- Inference quick start: `scripts/infer_v2/start.md`
