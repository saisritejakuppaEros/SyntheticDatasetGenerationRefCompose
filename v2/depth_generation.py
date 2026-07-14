#!/usr/bin/env python3
"""
depth_generation.py

Run monocular depth estimation on theme images and save depth maps.

Uses Depth Anything V2 via the Hugging Face depth-estimation pipeline:
https://huggingface.co/docs/transformers/en/tasks/monocular_depth_estimation

Typical usage:

    CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python depth_generation.py

    # Single sample:
    CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python depth_generation.py --theme beach_day_000035

    # Re-run everything, including files that already exist:
    CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python depth_generation.py --no_skip_existing
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import pipeline


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_IMAGES_DIR = SCRIPT_DIR / "outputs/theme_images/images"
DEFAULT_SOURCE_METADATA_DIR = SCRIPT_DIR / "outputs/theme_images/metadata"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs/depth"
DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-base-hf"


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate monocular depth maps for theme images."
    )
    p.add_argument(
        "--source_images_dir",
        type=str,
        default=str(DEFAULT_SOURCE_IMAGES_DIR),
        help="Directory containing source theme PNG images.",
    )
    p.add_argument(
        "--source_metadata_dir",
        type=str,
        default=str(DEFAULT_SOURCE_METADATA_DIR),
        help="Directory containing source theme metadata JSON files.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Root directory for generated depth images, raw arrays, and metadata.",
    )
    p.add_argument(
        "--model_id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Hugging Face checkpoint for depth estimation.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string.",
    )
    p.add_argument(
        "--theme",
        type=str,
        default=None,
        help="Process only this sample (image filename stem, e.g. beach_day_000035).",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip samples whose depth outputs already exist (default: True).",
    )
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    return p.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_pipeline_device(device: str):
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested but unavailable: {device}")
        if ":" in device:
            return int(device.split(":", 1)[1])
        return 0
    return -1


def collect_image_paths(args) -> list[Path]:
    source_images_dir = Path(args.source_images_dir)
    if not source_images_dir.is_dir():
        raise FileNotFoundError(f"source images dir not found: {source_images_dir}")

    if args.theme:
        image_path = source_images_dir / f"{args.theme}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"source image not found for sample: {args.theme}")
        return [image_path]

    image_paths = sorted(source_images_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"no PNG images found in {source_images_dir}")
    return image_paths


def process_sample(image_path: Path, pipe, args) -> tuple[int, int]:
    sample_id = image_path.stem
    source_metadata_path = Path(args.source_metadata_dir) / f"{sample_id}.json"
    source_metadata = load_json(source_metadata_path) if source_metadata_path.is_file() else {}

    out_img_dir = Path(args.output_dir) / "images"
    out_raw_dir = Path(args.output_dir) / "raw"
    out_meta_dir = Path(args.output_dir) / "metadata"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_raw_dir.mkdir(parents=True, exist_ok=True)
    out_meta_dir.mkdir(parents=True, exist_ok=True)

    out_img_path = out_img_dir / f"{sample_id}.png"
    out_raw_path = out_raw_dir / f"{sample_id}.npy"
    out_meta_path = out_meta_dir / f"{sample_id}.json"

    if (
        args.skip_existing
        and out_img_path.is_file()
        and out_raw_path.is_file()
        and out_meta_path.is_file()
    ):
        print(f"[{sample_id}] SKIP, depth already exists")
        return 0, 1

    image = Image.open(image_path).convert("RGB")
    predictions = pipe(image)

    predicted_depth = predictions["predicted_depth"]
    if hasattr(predicted_depth, "detach"):
        depth_array = predicted_depth.detach().cpu().numpy().astype(np.float32)
    else:
        depth_array = np.asarray(predicted_depth, dtype=np.float32)

    depth_vis = predictions["depth"]
    if not isinstance(depth_vis, Image.Image):
        depth_min = depth_array.min()
        depth_max = depth_array.max()
        depth_norm = (depth_array - depth_min) / max(depth_max - depth_min, 1e-8)
        depth_vis = Image.fromarray((depth_norm * 255).astype(np.uint8))

    depth_vis.save(out_img_path)
    np.save(out_raw_path, depth_array)

    depth_meta = {
        "id": sample_id,
        "theme": source_metadata.get("theme"),
        "depth_type": "relative",
        "model_id": args.model_id,
        "source_image": str(image_path.resolve()),
        "source_metadata": str(source_metadata_path.resolve())
        if source_metadata_path.is_file()
        else None,
        "image_width": image.width,
        "image_height": image.height,
        "depth_shape": list(depth_array.shape),
        "depth_min": float(depth_array.min()),
        "depth_max": float(depth_array.max()),
        "output_depth_image": str(out_img_path.resolve()),
        "output_depth_raw": str(out_raw_path.resolve()),
    }

    with open(out_meta_path, "w", encoding="utf-8") as f:
        json.dump(depth_meta, f, indent=2, ensure_ascii=False)

    print(f"[{sample_id}] saved depth -> {out_img_path.name}, {out_raw_path.name}")
    return 1, 0


def main():
    args = parse_args()

    try:
        image_paths = collect_image_paths(args)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Samples to process: {len(image_paths)}")
    for image_path in image_paths:
        print(f"  - {image_path.stem}")

    pipeline_device = resolve_pipeline_device(args.device)
    print(f"Loading depth model: {args.model_id} on device={args.device}")
    pipe = pipeline(
        "depth-estimation",
        model=args.model_id,
        device=pipeline_device,
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    total_done, total_skipped = 0, 0
    total_failed = 0
    t_start = time.time()

    for image_path in image_paths:
        sample_id = image_path.stem
        try:
            n_done, n_skipped = process_sample(image_path, pipe, args)
            total_done += n_done
            total_skipped += n_skipped
        except Exception as e:
            total_failed += 1
            print(f"[{sample_id}] ERROR: {e}", file=sys.stderr)
            continue

    elapsed = time.time() - t_start
    print(
        f"DONE. generated={total_done} skipped={total_skipped} failed={total_failed} "
        f"output_dir={args.output_dir} elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()