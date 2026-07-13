#!/usr/bin/env python3
"""
canvas_generation.py

Stage 4 of the pipeline: composite reference images onto a white canvas at
their detected bounding-box locations.

For each sample:
  - Create a white canvas matching the original scene dimensions (from bbox metadata).
  - Place each reference image (from outputs/reference_images) at its bbox position
    (from outputs/bbox_annotations).
  - Resize each reference to its bbox size and paste in z-order: largest bbox first,
    smallest bbox on top.
  - Near-white reference backgrounds are made transparent before compositing.

Outputs:
  - outputs/canvas/images/<sample_id>.png
  - outputs/canvas/metadata/<sample_id>.json

Typical usage:

    python canvas_generation.py

    # Single sample:
    python canvas_generation.py --theme home_office_000024

    # Re-run everything, including files that already exist:
    python canvas_generation.py --no_skip_existing
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BBOX_METADATA_DIR = SCRIPT_DIR / "outputs/bbox_annotations/metadata"
DEFAULT_REFERENCE_DIR = SCRIPT_DIR / "outputs/reference_images"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs/canvas"

DEFAULT_WHITE_THRESHOLD = 240


def parse_args():
    p = argparse.ArgumentParser(
        description="Composite reference images onto white canvases using bbox metadata."
    )
    p.add_argument(
        "--bbox_metadata_dir",
        type=str,
        default=str(DEFAULT_BBOX_METADATA_DIR),
        help="Directory containing bbox_annotation.py output JSON files.",
    )
    p.add_argument(
        "--reference_dir",
        type=str,
        default=str(DEFAULT_REFERENCE_DIR),
        help="Root directory for generated reference images (organized by sample-id subfolder).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Root directory for generated canvas images and metadata.",
    )
    p.add_argument(
        "--theme",
        type=str,
        default=None,
        help="Process only this sample (bbox metadata filename stem, e.g. home_office_000024).",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip samples whose canvas image already exists (default: True).",
    )
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    p.add_argument(
        "--white_threshold",
        type=int,
        default=DEFAULT_WHITE_THRESHOLD,
        help="RGB channels >= this value are treated as transparent reference background.",
    )
    return p.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bbox_area(bbox: list[float]) -> float:
    xmin, ymin, xmax, ymax = bbox
    return max(0.0, xmax - xmin) * max(0.0, ymax - ymin)


def reference_to_rgba(image: Image.Image, white_threshold: int) -> Image.Image:
    rgba = image.convert("RGBA")
    arr = np.array(rgba, copy=True)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    white_mask = (r >= white_threshold) & (g >= white_threshold) & (b >= white_threshold)
    arr[white_mask, 3] = 0
    return Image.fromarray(arr, mode="RGBA")


def paste_rgba(canvas: Image.Image, layer: Image.Image, xy: tuple[int, int]) -> Image.Image:
    base = canvas.convert("RGBA")
    base.paste(layer, xy, layer)
    return base.convert("RGB")


def collect_placements(
    bbox_meta: dict,
    reference_dir: Path,
) -> list[dict]:
    sample_id = bbox_meta["id"]
    sample_ref_dir = reference_dir / sample_id
    objects = bbox_meta.get("bbox_information", {}).get("objects", [])

    placements = []
    for idx, obj_entry in enumerate(objects, start=1):
        if not obj_entry.get("found"):
            continue

        bbox = obj_entry.get("bbox")
        if not bbox:
            continue

        ref_meta_path = sample_ref_dir / f"bbox{idx}.json"
        ref_img_path = sample_ref_dir / f"bbox{idx}.jpg"

        if ref_meta_path.is_file():
            ref_meta = load_json(ref_meta_path)
            ref_img_path = Path(ref_meta.get("output_image") or ref_img_path)
        elif not ref_img_path.is_file():
            continue

        if not ref_img_path.is_file():
            continue

        placements.append(
            {
                "bbox_index": idx,
                "object": obj_entry.get("object"),
                "detected_label": obj_entry.get("label"),
                "detection_score": obj_entry.get("score"),
                "source_bbox": bbox,
                "bbox_area": bbox_area(bbox),
                "reference_image": str(ref_img_path.resolve()),
                "reference_metadata": str(ref_meta_path.resolve()) if ref_meta_path.is_file() else None,
            }
        )

    placements.sort(key=lambda p: p["bbox_area"], reverse=True)
    for z_order, placement in enumerate(placements, start=1):
        placement["z_order"] = z_order

    return placements


def build_canvas(
    bbox_meta: dict,
    placements: list[dict],
    args,
) -> Image.Image:
    image_w = int(bbox_meta.get("image_width") or 0)
    image_h = int(bbox_meta.get("image_height") or 0)
    if image_w <= 0 or image_h <= 0:
        raise ValueError("bbox metadata missing image_width/image_height")

    canvas = Image.new("RGB", (image_w, image_h), color=(255, 255, 255))

    for placement in placements:
        bbox = placement["source_bbox"]
        xmin, ymin, xmax, ymax = [int(round(v)) for v in bbox]
        target_w = max(1, xmax - xmin)
        target_h = max(1, ymax - ymin)

        ref_image = Image.open(placement["reference_image"]).convert("RGB")
        ref_rgba = reference_to_rgba(
            ref_image.resize((target_w, target_h), Image.LANCZOS),
            white_threshold=args.white_threshold,
        )
        canvas = paste_rgba(canvas, ref_rgba, (xmin, ymin))

    return canvas


def process_sample(meta_path: Path, args) -> tuple[int, int]:
    bbox_meta = load_json(meta_path)
    sample_id = bbox_meta.get("id", meta_path.stem)
    theme = bbox_meta.get("theme", "unknown")

    out_img_dir = Path(args.output_dir) / "images"
    out_meta_dir = Path(args.output_dir) / "metadata"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_meta_dir.mkdir(parents=True, exist_ok=True)

    out_img_path = out_img_dir / f"{sample_id}.png"
    out_meta_path = out_meta_dir / f"{sample_id}.json"

    if args.skip_existing and out_img_path.is_file() and out_meta_path.is_file():
        print(f"[{sample_id}] SKIP, canvas already exists")
        return 0, 1

    placements = collect_placements(bbox_meta, Path(args.reference_dir))
    if not placements:
        print(f"[{sample_id}] SKIP, no reference placements found")
        return 0, 1

    canvas = build_canvas(bbox_meta, placements, args)
    canvas.save(out_img_path)

    canvas_meta = {
        "id": sample_id,
        "theme": theme,
        "canvas_width": canvas.size[0],
        "canvas_height": canvas.size[1],
        "canvas_background": "white",
        "white_threshold": args.white_threshold,
        "source_bbox_metadata": str(meta_path.resolve()),
        "source_scene_image": bbox_meta.get("source_image"),
        "source_theme_metadata": bbox_meta.get("source_metadata"),
        "num_placements": len(placements),
        "placements": placements,
        "output_image": str(out_img_path.resolve()),
    }

    with open(out_meta_path, "w", encoding="utf-8") as f:
        json.dump(canvas_meta, f, indent=2, ensure_ascii=False)

    print(
        f"[{sample_id}] saved canvas with {len(placements)} placements -> {out_img_path.name}"
    )
    return 1, 0


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
            print(f"ERROR: bbox metadata not found for sample: {args.theme}", file=sys.stderr)
            sys.exit(1)

    if not meta_paths:
        print(f"ERROR: no bbox metadata JSON files found in {bbox_metadata_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Samples to process: {len(meta_paths)}")
    for meta_path in meta_paths:
        print(f"  - {meta_path.stem}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    total_done, total_skipped = 0, 0
    total_failed = 0
    t_start = time.time()

    for meta_path in meta_paths:
        sample_id = meta_path.stem
        try:
            n_done, n_skipped = process_sample(meta_path, args)
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
