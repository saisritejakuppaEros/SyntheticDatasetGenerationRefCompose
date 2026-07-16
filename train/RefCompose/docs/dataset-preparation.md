# Dataset Preparation

RefCompose includes two dataset preparation pipelines under `scripts/dataset_prep/` (v1) and `scripts/dataset_prep_v2/` (v2). Both produce a common **output tree** consumed by canvas training.

## Expected Output Layout

After preparation, each dataset root should contain:

```
output/
├── images/                          # Target RGB scenes
│   └── {id}.png
├── bbox_results/
│   └── yolo26_detections.csv        # One row per detection
├── depth/
│   └── {id}.png                     # 16-bit grayscale depth (I;16)
├── image_captions/
│   └── {id}.txt                     # Text-to-image prompt per image
└── multiview_out/                   # Optional multiview crops
    └── {image_stem}/
        └── {crop_no}/
            └── {viewno}.png
```

Training reads this layout via `--canvas_data_roots` (see [Training](training.md)).

## CSV Format (YOLO Detections)

`yolo26_detections.csv` has one row per bounding box:

| Column | Description |
|--------|-------------|
| `image_path` | Relative path to source image (e.g. `images/abc123.png`) |
| `x1`, `y1`, `x2`, `y2` | Bounding box in image pixel coordinates |
| `confidence` | Detection confidence |
| `class_id` / `class_name` | YOLO class (optional) |

Multiple rows per `image_path` are grouped and composited into a single canvas.

---

## Pipeline v1 (`scripts/dataset_prep/`)

Designed for parquet shards and a full multiview + canvas pipeline.

### Stage 1: Extract Images from Parquet

**Script:** `stage1_extract_images_from_parquet.py`

Extracts image bytes from Hugging Face-style parquet files into `images/`.

```bash
python stage1_extract_images_from_parquet.py \
  --parquet-root /path/to/parquet \
  --out-dir output/images
```

Handles `image.bytes` dict columns and raw byte columns; auto-detects format from magic bytes.

### Stage 2: YOLO Object Detection

**Script:** `stage2_yolo_inferences.py`

Runs Ultralytics YOLO on all images; writes `output/bbox_results/yolo26_detections.csv`.

```bash
# Edit paths in script or use yolo_boxes.py CLI variant
python stage2_yolo_inferences.py
```

**CLI variant (v2 folder):** `yolo_boxes.py` with `--images-dir`, `--output-csv`, `--conf`, `--model`.

### Stage 2b: Bbox Size Statistics (optional)

**Script:** `stage2_bbox_size_stats.py`

Analyzes detection box sizes to tune `--min-side` filters for later stages.

### Stage 3: Multiview Generation

**Script:** `stage3_multiview.py` (Qwen Image Edit)

Filters YOLO boxes by minimum side length, crops objects, and generates multiple viewing angles using Qwen + multi-angle LoRA.

```bash
python stage3_multiview.py \
  --csv output/bbox_results/yolo26_detections.csv \
  --images-dir output/images \
  --out-dir output/multiview_out
```

**Alternative:** `stage3_hunyuan_multiview.py` — Hunyuan-based multiview variant.

Default azimuths: front, quarter views, sides, back (8 views per crop).

### Stage 4: Canvas Composition

**Script:** `stage4_canva.py`

Builds one 1920×1080 canvas per image by pasting bbox crops (from multiview or original image crop) onto a black or scaled background.

```bash
python stage4_canva.py \
  --csv output/bbox_results/yolo26_detections.csv \
  --images-dir output/images \
  --multiview-dir output/multiview_out \
  --out-dir output/canvases
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--min-side` | 0 | Minimum bbox side to include (0 = all) |
| `--multiview-prob` | 0.5 | Probability to use multiview vs direct crop |
| `--background` | `black` | `black` or `scaled` (letterboxed full frame) |
| `--workers` | 8 | Parallel image processing threads |

Paste order: `crop_no` 1, 2, … first; `crop_no == 0` on top.

### Stage 5: Depth Generation

**Script:** `stage5_depth_generation.py`

Runs Depth Anything 3 (DA3-BASE) on RGB images; writes 16-bit depth PNGs to `output/depth/`.

```bash
python stage5_depth_generation.py \
  --images-dir output/images \
  --out-dir output/depth
```

Requires DA3 package (see [Getting Started](getting-started.md)).

### Stage 6: Image Captioning

**Script:** `stage6_image_caption_qwen3vl.py`

Captions each image with Qwen3-VL; writes standalone ~100-word text-to-image prompts.

```bash
python stage6_image_caption_qwen3vl.py \
  --image-dir output/images \
  --out-dir output/image_captions
```

Supports multi-GPU sharding via `--index-start` / `--index-end` and `--print-shard-ranges`.

### Utilities

| Script | Purpose |
|--------|---------|
| `extract_hd_crops.py` | Extract HD crops from images |
| `yolo_boxes.py` | CLI YOLO detection (also in v2) |

---

## Pipeline v2 (`scripts/dataset_prep_v2/`)

Streamlined pipeline for tar-based archives with 1280×720 cover crops.

### Extract HD Crops

**Script:** `extract_hd_crops.py`

Extracts images from `.tar` archives, resizes with cover crop to 1280×720, writes flat outputs:

```
output/images/{unq_id}.png
output/image_captions/{unq_id}.txt
```

```bash
python extract_hd_crops.py --images-dir /path/to/tars
```

`unq_id` = `{tar_stem}_{image_stem}` for cross-archive uniqueness.

### YOLO Detection

**Script:** `yolo_boxes.py`

Same as v1 CLI; defaults to `output/images` and `output/bbox_results/yolo26_detections.csv`.

```bash
python yolo_boxes.py --model yolo26n.pt --conf 0.25
```

### Depth Generation

**Script:** `stage5_depth_generation.py`

Identical to v1 stage 5; writes `output/depth/{unq_id}.png`.

---

## v1 vs v2 Comparison

| Feature | v1 (`dataset_prep`) | v2 (`dataset_prep_v2`) |
|---------|---------------------|------------------------|
| Input | Parquet shards | Tar archives + metadata.jsonl |
| Target resolution | Variable → 1920×1080 canvas | 1280×720 cover crop |
| Multiview | Yes (stage 3) | No (direct bbox crops) |
| Canvas stage | Explicit `stage4_canva.py` | Built at train time via `canvas_bbox_compose.py` |
| Captioning | Qwen3-VL stage 6 | From metadata during extract |

Both produce compatible `output/` trees for `--canvas_data_roots`.

---

## Canvas Building at Train Time

When `dataset_type=canvas`, training does **not** require pre-rendered canvas PNGs. `CanvasSceneDataset` calls `build_canvas_for_image_rel()` from `canvas_bbox_compose.py` on the fly:

1. Group CSV rows by `image_path`
2. For each bbox, sample multiview or crop from source image
3. Paste onto 1920×1080 canvas (black or scaled background)
4. Cover-crop canvas + target to unified training size (default 1280×720)

Pre-rendered canvases from stage 4 can still be used if stored in the CSV `canvas_path` column.

---

## Data Quality Tips

1. **YOLO confidence** — Start at 0.25; raise if too many false positives.
2. **Min side filter** — Use `stage2_bbox_size_stats.py` to pick `--min-side` for multiview (v1).
3. **Depth format** — Always 16-bit grayscale PNG (`I;16`), values in [0, 65535]. False-color RGB depth will break training.
4. **Captions** — Prompts should be dense visual descriptions without meta-framing ("The image shows…").
5. **Balanced roots** — When merging multiple dataset roots, use similar scene diversity per root.

## Next Steps

- [Training](training.md) — point `--canvas_data_roots` at your `output/` folder
- [Architecture](architecture.md) — how canvas tokens are used in the model
