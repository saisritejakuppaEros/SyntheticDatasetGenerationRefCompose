# API Reference

This document describes the main Python modules in RefCompose. Paths are relative to `scripts/train/src/` unless noted.

---

## `canvas_dataset.py`

CSV-driven dataset for FLUX.2 canvas training.

### Constants

```python
CANVAS_DATA_ROOT_CSV = Path("bbox_results") / "yolo26_detections.csv"
```

### Functions

| Function | Description |
|----------|-------------|
| `parse_canvas_data_roots(raw)` | Parse comma/semicolon/newline-separated dataset root paths |
| `resolve_canvas_data_paths(root)` | Return dict with `csv`, `images`, `depth`, `captions`, `multiview` paths |
| `balanced_sampling_weights_for_concat(ds)` | Equal-probability weights for `ConcatDataset` |
| `resize_cover_pil(im, out_w, out_h)` | Aspect-preserving cover crop resize |
| `resize_contain_letterbox_pil(im, out_w, out_h, fill)` | Letterbox resize with padding |
| `pil_to_model_tensor(pil)` | PIL → `[-1, 1]` CHW tensor |
| `load_depth_image_as_rgb_pil(path)` | Load 16-bit depth PNG as RGB PIL for VAE encoding |
| `coarse_degrade_depth_rgb_pil(depth_rgb, args)` | Training-style depth blur/noise/patch dropout |
| `make_canvas_train_dataset(args, accelerator)` | Factory for `CanvasSceneDataset` or `CanvasConcatDataset` |
| `collate_fn_canvas(examples)` | Batch collation for DataLoader |

### Classes

#### `CanvasSceneDataset`

PyTorch `Dataset` for one dataset root.

**Per-sample outputs:**

| Key | Type | Description |
|-----|------|-------------|
| `pixel_values` | Tensor | Target RGB latent input |
| `subject_pixel_values` | Tensor | Canvas cond pixels |
| `cond_pixel_values` | Tensor | Depth cond pixels (if depth enabled) |
| `prompt` | str | Caption text |
| `main_latent_ids` | Tensor | FLUX.2 latent position IDs |

#### `CanvasConcatDataset`

Wraps multiple `CanvasSceneDataset` instances; proxies augmentation toggles to all children.

---

## `canvas_bbox_compose.py`

On-the-fly canvas composition from YOLO CSV rows.

### Constants

```python
CANVAS_W = 1920
CANVAS_H = 1080
```

### Functions

| Function | Description |
|----------|-------------|
| `prepare_groups_from_flat_rows(rows, ...)` | Group CSV rows by image, assign `crop_no` |
| `compose_bbox_multiview_canvas(...)` | Paste bbox crops onto canvas |
| `build_canvas_for_image_rel(rel, ...)` | Full canvas build for one image |
| `prompt_for_group(group, caption_root)` | Load caption text for image group |
| `map_rows_xyxy_to_cover_crop_space(...)` | Map bbox coords to cover-crop viewport |
| `cover_layout(w, h, out_w, out_h)` | Compute cover-crop scale and offset |
| `letterbox_layout(w, h, out_w, out_h)` | Compute letterbox scale and offset |

---

## `flux2_transformer_cond.py`

Conditional FLUX.2 transformer with canvas/depth token streams.

### Classes

| Class | Description |
|-------|-------------|
| `CondFlux2AttnProcessor` | Splits attention output into main + cond after joint attention |
| `CondFlux2ParallelSelfAttnProcessor` | Parallel attention variant with cond split |
| `Flux2TransformerBlockCond` | Double-stream block with cond branch |
| `Flux2SingleTransformerBlockCond` | Single-stream block with cond support |
| `Flux2Transformer2DModelCond` | Full transformer model |

**Forward signature highlights:**

```python
def forward(
    self,
    hidden_states,           # main latent tokens
    cond_hidden_states=None, # canvas/depth tokens
    encoder_hidden_states,   # text tokens
    timestep,
    ...
) -> Transformer2DModelOutput
```

---

## `layers_flux2.py`

LoRA processors for FLUX.2 attention blocks.

### Classes

| Class | Description |
|-------|-------------|
| `LoRALinearLayerFlux2` | Low-rank linear with per-block token masking |
| `MultiDoubleStreamBlockFlux2LoraProcessor` | LoRA on double-stream block Q/K/V |
| `MultiSingleStreamBlockFlux2LoraProcessor` | LoRA on single-stream block Q/K/V |

**Token mask logic:** Each LoRA index `i` masks to canvas block `i` in the sequence, zeroing all other positions before the low-rank projection.

---

## `layers.py` (FLUX.1)

LoRA processors for FLUX.1 — same pattern as `layers_flux2.py` but for the original FLUX architecture.

| Class | Description |
|-------|-------------|
| `LoRALinearLayer` | Base LoRA linear layer |
| `MultiDoubleStreamBlockLoraProcessor` | Double-stream LoRA |
| `MultiSingleStreamBlockLoraProcessor` | Single-stream LoRA |

---

## `lora_helper.py` (FLUX.1)

Load and attach LoRA weights for inference.

| Function | Description |
|----------|-------------|
| `load_safetensors(path)` | Load safetensors dict |
| `get_lora_rank(checkpoint)` | Infer rank from checkpoint keys |
| `set_single_lora(transformer, path, lora_weights, cond_size)` | Attach one LoRA |
| `set_multi_lora(transformer, paths, lora_weights, cond_size)` | Attach multiple LoRAs |
| `unset_lora(transformer)` | Remove all LoRA processors |

---

## `pipeline.py` (FLUX.1)

Custom `FluxPipeline` with spatial/subject image conditioning.

### Class: `FluxPipeline`

Extends `DiffusionPipeline` with:

```python
def __call__(
    self,
    prompt,
    height, width,
    spatial_images=None,   # list of PIL images
    subject_images=None,   # list of PIL images
    cond_size=512,
    guidance_scale=3.5,
    num_inference_steps=25,
    ...
) -> FluxPipelineOutput
```

### Utilities

| Function | Description |
|----------|-------------|
| `calculate_shift(image_seq_len, ...)` | FLUX shift schedule parameter |
| `prepare_latent_image_ids_2(h, w, device, dtype)` | 2D latent position IDs |
| `resize_position_encoding(...)` | Interpolate position encodings for cond resize |

---

## `transformer_flux.py` (FLUX.1)

Wrapper around FLUX.1 transformer with custom attention processor slots for LoRA.

---

## `flux2_train_helpers.py`

| Function | Description |
|----------|-------------|
| `encode_flux2_latents(vae, pixel_values, ...)` | VAE encode to packed latents |
| `prepare_subject_latent_ids(latents_bchw, scale)` | Position IDs for cond latents |
| `unpack_main_latents(packed_pred, main_latent_ids)` | Extract main-stream prediction from packed output |

---

## `flux2_dataloader_validation.py`

Training-time validation via dataloader sampling.

| Function | Description |
|----------|-------------|
| `_denoise_one(...)` | Single validation denoise step loop |
| `_compute_flow_matching_loss_batch(...)` | Flow-matching loss on a batch |
| `run_dataloader_validation(...)` | Full validation pass: loss + PNG dumps |
| `_tensor_m11_chw_to_pil(t)` | Denormalize tensor to PIL for saving |

Used by both `train.py` (validation) and `stage3_lora_infer.py` (inference).

---

## `prompt_helper.py`

Text encoding for FLUX.1 and FLUX.2.

| Function | Description |
|----------|-------------|
| `encode_prompts_flux2(tokenizer, text_encoder, prompts, ...)` | FLUX.2 Mistral3/Pixtral encoding |
| `encode_prompt(text_encoders, prompt, ...)` | FLUX.1 CLIP + T5 encoding |
| `load_text_encoders(args, class_one, class_two)` | Load text encoder pair |

---

## `jsonl_datasets.py`

JSONL dataset for FLUX.1 training.

| Function | Description |
|----------|-------------|
| `make_train_dataset(args, accelerator, tokenizer)` | Build HuggingFace-style dataset from JSONL |
| `collate_fn(examples)` | Batch collation |
| `load_image_safely(path, size, root_dir)` | Load and resize image |
| `get_random_resolution(min, max, multiple)` | Random target size (multiple of 16) |
| `multiple_16(num)` | Round to nearest multiple of 16 |

---

## Inference Scripts (`scripts/infer_v2/`)

### `stage3_lora_infer.py`

| Function | Description |
|----------|-------------|
| `print_lora_checkpoint_before_load(lora_path)` | Log checkpoint path before loading |
| `load_models(...)` | Load VAE, text encoder, transformer, scheduler |
| `run_inference(...)` | Full canvas + depth LoRA pipeline |

Constants:

```python
UNIFIED_WIDTH = 1280
UNIFIED_HEIGHT = 720
```

### `stage2_depth.py`

| Function | Description |
|----------|-------------|
| `_relative_depth_to_u16_hw(depth_hw)` | Float depth → uint16 PNG values |
| `_infer_depth_u16(model, device, path)` | Run DA3 on one image |

### `flux2_heritage_infer.py`

| Function | Description |
|----------|-------------|
| `load_flux2_pipeline(model_path, torch_dtype)` | Load with CPU offload |
| `generate_one(pipe, prompt, ...)` | Single T2I generation |
| `HERITAGE_INTERIOR_PROMPT` | Built-in heritage interior prompt string |

### `lora_server.py`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Run LoRA inference on a folder of inputs |
| `/health` | GET | Server health check |

Request body: `{ "folder": "/path/to/inputs" }`

---

## Dataset Prep Scripts

### `dataset_prep/stage4_canva.py`

Builds 1920×1080 canvas PNGs from YOLO CSV + multiview crops.

### `dataset_prep/stage6_image_caption_qwen3vl.py`

Qwen3-VL captioning; outputs one `.txt` per image.

### `dataset_prep_v2/extract_hd_crops.py`

Extract from tar archives → 1280×720 cover crop PNGs + captions.

### `dataset_prep_v2/yolo_boxes.py`

CLI YOLO detection → CSV.

---

## Testing (`scripts/train/testing/`)

| Script | Tests |
|--------|-------|
| `test_forward_pass.py` | Transformer forward with dummy tensors |
| `test_dataloader.py` | `CanvasSceneDataset` batch shapes |
| `test_validation_loop_smoke.py` | Validation denoise loop |
| `test_training_flow_documented.py` | End-to-end training flow |
