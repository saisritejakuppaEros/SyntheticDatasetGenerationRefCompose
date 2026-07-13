#!/usr/bin/env python3
"""
synthetic_ref_image.py

Stage 3 of the pipeline: takes bbox_annotation.py's output (per-theme
metadata with a bounding box for each of the theme's objects, located inside
the full scene image) and turns every detected box into a clean, isolated,
high-resolution *reference image* of just that object:

  - Crops the object out of the scene using its bbox (with a little padding
    so the model gets some context and isn't fed a razor-tight sliver).
  - Feeds that crop into FLUX.2 as a reference/conditioning image and
    re-generates it as a neutral, front-facing, non-tilted product shot on a
    pure white background, at 1024x1024.
  - Writes one file per detected object, organized by theme:

        outputs/reference_images/<theme>/bbox1.jpg
        outputs/reference_images/<theme>/bbox2.jpg
        ...
        outputs/reference_images/<theme>/bbox1.json   (sidecar metadata)

    The numeric suffix matches the object's position in that theme's
    `objects` list from the bbox metadata, so bbox1.jpg is always the same
    object across runs.

Typical usage:

    CUDA_VISIBLE_DEVICES=0 python synthetic_ref_image.py

    # Single theme, smaller GPU:
    CUDA_VISIBLE_DEVICES=0 python synthetic_ref_image.py --theme action --quantized

    # Re-run everything, including files that already exist:
    CUDA_VISIBLE_DEVICES=0 python synthetic_ref_image.py --no_skip_existing

Notes:
  - Requires bbox_annotation.py to have already been run (reads its metadata
    output, does not touch theme_images).
  - Uses FLUX.2's image-conditioning input (`image=[...]`) the same way
    FLUX.1 Kontext-style pipelines do: the reference crop steers subject
    identity/appearance while the prompt steers pose/background/framing.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch
from PIL import Image, ImageOps


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BBOX_METADATA_DIR = SCRIPT_DIR / "outputs/bbox_annotations/metadata"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs/reference_images"
DEFAULT_DEBUG_CROP_DIR = SCRIPT_DIR / "outputs/reference_images/_debug_crops"

REF_SIZE = 1024                # final output resolution (square)
DEFAULT_PAD_FRAC = 0.12        # pad the tight bbox this much (of box w/h) before feeding as reference
DEFAULT_REF_FEED_SIZE = 1024   # resolution the cropped reference is resized to before conditioning
MIN_CROP_SIDE_PX = 8           # sanity floor; smaller than this and we skip the box

NEUTRAL_PROMPT_TEMPLATE = (
    "Studio product photograph of {label}, a single isolated object centered "
    "on a seamless pure white background. Neutral front-facing view, "
    "camera-facing, perfectly upright, symmetrical, not tilted, not rotated, "
    "no dramatic angle, no perspective distortion. Even soft studio "
    "lighting, no harsh shadows, no reflections, no props, no other "
    "objects, no hands, no text, no watermark, no border, no frame. Sharp "
    "focus, crisp clean edges, high resolution e-commerce catalog photo."
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Turn bbox detections into isolated, neutral, white-background reference images via FLUX.2."
    )
    p.add_argument(
        "--bbox_metadata_dir",
        type=str,
        default=str(DEFAULT_BBOX_METADATA_DIR),
        help="Directory containing bbox_annotation.py output JSON files (read-only).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Root directory for generated reference images (organized by theme subfolder).",
    )
    p.add_argument(
        "--model_id",
        type=str,
        default="black-forest-labs/FLUX.2-dev",
        help="HF model id for Flux2Pipeline. Use 'diffusers/FLUX.2-dev-bnb-4bit' with --quantized for consumer GPUs.",
    )
    p.add_argument(
        "--quantized",
        action="store_true",
        help="Load the 4-bit (bnb) quantized transformer + text encoder instead of full bf16. Auto-enables CPU offload.",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--num_inference_steps", type=int, default=4)
    p.add_argument(
        "--pad_frac",
        type=float,
        default=DEFAULT_PAD_FRAC,
        help="Extra padding around each tight bbox before cropping, as a fraction of box width/height.",
    )
    p.add_argument(
        "--ref_feed_size",
        type=int,
        default=DEFAULT_REF_FEED_SIZE,
        help="Resolution the padded/square crop is resized to before being fed as the reference image.",
    )
    p.add_argument(
        "--min_score",
        type=float,
        default=0.0,
        help="Skip detections with bbox_score below this (0 = keep all found boxes).",
    )
    p.add_argument("--seed", type=int, default=42, help="Base seed; per-object seed derived from this + id.")
    p.add_argument("--theme", type=str, default=None, help="Process only this theme (metadata filename stem).")
    p.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip bbox outputs whose reference image already exists (default: True).",
    )
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    p.add_argument(
        "--save_debug_crops",
        action="store_true",
        help="Also save the raw padded crop (pre-generation) next to the output, for QA.",
    )
    p.add_argument("--jpg_quality", type=int, default=95)
    return p.parse_args()


# ----------------------------------------------------------------------------
# Cropping helpers
# ----------------------------------------------------------------------------

def pad_bbox(bbox, img_w, img_h, pad_frac):
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin
    h = ymax - ymin
    pad_x = w * pad_frac
    pad_y = h * pad_frac
    return [
        max(0.0, xmin - pad_x),
        max(0.0, ymin - pad_y),
        min(float(img_w), xmax + pad_x),
        min(float(img_h), ymax + pad_y),
    ]


def crop_object(image: Image.Image, bbox, pad_frac: float):
    img_w, img_h = image.size
    xmin, ymin, xmax, ymax = pad_bbox(bbox, img_w, img_h, pad_frac)
    xmin, ymin, xmax, ymax = [int(round(v)) for v in (xmin, ymin, xmax, ymax)]
    xmax = max(xmax, xmin + 1)
    ymax = max(ymax, ymin + 1)
    if (xmax - xmin) < MIN_CROP_SIDE_PX or (ymax - ymin) < MIN_CROP_SIDE_PX:
        raise ValueError(f"bbox too small after clamping: {(xmin, ymin, xmax, ymax)}")
    return image.crop((xmin, ymin, xmax, ymax))


def square_pad_white(image: Image.Image) -> Image.Image:
    """Pad (not stretch) the crop onto a white square canvas, centered."""
    w, h = image.size
    side = max(w, h)
    return ImageOps.pad(image, (side, side), color=(255, 255, 255), centering=(0.5, 0.5))


# ----------------------------------------------------------------------------
# Pipeline (mirrors flux2_imagegen.py's loader, plus reference conditioning)
# ----------------------------------------------------------------------------

def build_pipeline(args):
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    if args.quantized:
        from diffusers import Flux2Pipeline, Flux2Transformer2DModel
        from transformers import Mistral3ForConditionalGeneration

        repo_id = args.model_id if "bnb-4bit" in args.model_id else "diffusers/FLUX.2-dev-bnb-4bit"
        print(f"Loading quantized FLUX.2 ({repo_id}) ...")

        transformer = Flux2Transformer2DModel.from_pretrained(
            repo_id, subfolder="transformer", torch_dtype=torch_dtype, device_map="auto",
        )
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            repo_id, subfolder="text_encoder", torch_dtype=torch_dtype, device_map="auto",
        )
        pipe = Flux2Pipeline.from_pretrained(
            repo_id, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch_dtype,
        )
        pipe.enable_model_cpu_offload()
    else:
        from diffusers import Flux2Pipeline

        print(f"Loading pipeline {args.model_id} (dtype={args.dtype}) ...")
        pipe = Flux2Pipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype)
        pipe.to(args.device)

    return pipe


def object_seed(base_seed, sample_id):
    return (base_seed + sum(ord(c) for c in sample_id)) % (2**31 - 1)


def generate_reference_image(pipe, ref_crop: Image.Image, label: str, args, seed: int) -> Image.Image:
    prompt = NEUTRAL_PROMPT_TEMPLATE.format(label=label)

    ref_square = square_pad_white(ref_crop).resize(
        (args.ref_feed_size, args.ref_feed_size), Image.LANCZOS
    )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        image=[ref_square],          # FLUX.2 reference/conditioning image(s)
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        height=REF_SIZE,
        width=REF_SIZE,
        generator=generator,
    )
    return result.images[0]


# ----------------------------------------------------------------------------
# Per-theme processing
# ----------------------------------------------------------------------------

def load_bbox_metadata(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_theme(meta_path: Path, pipe, args):
    meta = load_bbox_metadata(meta_path)
    theme = meta.get("theme", meta_path.stem)

    source_image_path = Path(meta.get("source_image") or "")
    if not source_image_path.is_file():
        raise FileNotFoundError(f"[{theme}] source scene image not found: {source_image_path}")

    scene_image = Image.open(source_image_path).convert("RGB")

    objects = meta.get("bbox_information", {}).get("objects", [])
    if not objects:
        raise ValueError(f"[{theme}] no bbox_information.objects in {meta_path}")

    theme_out_dir = Path(args.output_dir) / theme
    theme_out_dir.mkdir(parents=True, exist_ok=True)

    n_done, n_skipped, n_failed = 0, 0, 0

    for idx, obj_entry in enumerate(objects, start=1):
        object_name = obj_entry.get("object", f"object_{idx}")
        out_img_path = theme_out_dir / f"bbox{idx}.jpg"
        out_meta_path = theme_out_dir / f"bbox{idx}.json"

        if args.skip_existing and out_img_path.is_file():
            print(f"[{theme}] bbox{idx} ({object_name}): SKIP, already exists")
            n_skipped += 1
            continue

        if not obj_entry.get("found"):
            print(f"[{theme}] bbox{idx} ({object_name}): SKIP, not detected in scene")
            n_skipped += 1
            continue

        score = obj_entry.get("score")
        if score is not None and score < args.min_score:
            print(f"[{theme}] bbox{idx} ({object_name}): SKIP, score {score:.3f} < {args.min_score}")
            n_skipped += 1
            continue

        try:
            bbox = obj_entry.get("bbox")
            if not bbox:
                raise ValueError("missing bbox")

            crop = crop_object(scene_image, bbox, args.pad_frac)

            if args.save_debug_crops:
                debug_path = theme_out_dir / f"bbox{idx}_crop.jpg"
                square_pad_white(crop).save(debug_path, quality=args.jpg_quality)

            sample_id = f"{theme}_bbox{idx}"
            seed = object_seed(args.seed, sample_id)

            ref_image = generate_reference_image(pipe, crop, object_name, args, seed)
            if ref_image.size != (REF_SIZE, REF_SIZE):
                ref_image = ref_image.resize((REF_SIZE, REF_SIZE), Image.LANCZOS)

            ref_image.save(out_img_path, quality=args.jpg_quality)

            with open(out_meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "theme": theme,
                        "bbox_index": idx,
                        "object": object_name,
                        "detected_label": obj_entry.get("label"),
                        "detection_score": score,
                        "source_bbox_metadata": str(meta_path.resolve()),
                        "source_scene_image": str(source_image_path.resolve()),
                        "source_bbox": bbox,
                        "pad_frac": args.pad_frac,
                        "ref_feed_size": args.ref_feed_size,
                        "output_size": [REF_SIZE, REF_SIZE],
                        "model_id": args.model_id,
                        "guidance_scale": args.guidance_scale,
                        "num_inference_steps": args.num_inference_steps,
                        "seed": seed,
                        "output_image": str(out_img_path.resolve()),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            n_done += 1
            print(f"[{theme}] bbox{idx} ({object_name}): saved -> {out_img_path.name}")

        except Exception as e:
            n_failed += 1
            print(f"[{theme}] bbox{idx} ({object_name}): ERROR: {e}", file=sys.stderr)
            traceback.print_exc()
            continue

    return n_done, n_skipped, n_failed


def main():
    args = parse_args()
    bbox_metadata_dir = Path(args.bbox_metadata_dir)

    if not bbox_metadata_dir.is_dir():
        print(f"ERROR: bbox metadata dir not found: {bbox_metadata_dir}", file=sys.stderr)
        sys.exit(1)

    meta_paths = sorted(bbox_metadata_dir.glob("*.json"))
    if args.theme:
        meta_paths = [bbox_metadata_dir / f"{args.theme}.json"]
        if not meta_paths[0].is_file():
            print(f"ERROR: bbox metadata not found for theme: {args.theme}", file=sys.stderr)
            sys.exit(1)

    if not meta_paths:
        print(f"ERROR: no bbox metadata JSON files found in {bbox_metadata_dir}", file=sys.stderr)
        sys.exit(1)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    pipe = build_pipeline(args)

    total_done, total_skipped, total_failed = 0, 0, 0
    t_start = time.time()

    for meta_path in meta_paths:
        try:
            n_done, n_skipped, n_failed = process_theme(meta_path, pipe, args)
            total_done += n_done
            total_skipped += n_skipped
            total_failed += n_failed
        except Exception as e:
            total_failed += 1
            print(f"[{meta_path.stem}] ERROR: {e}", file=sys.stderr)
            traceback.print_exc()
            continue

    elapsed = time.time() - t_start
    print(
        f"DONE. generated={total_done} skipped={total_skipped} failed={total_failed} "
        f"output_dir={args.output_dir} elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()