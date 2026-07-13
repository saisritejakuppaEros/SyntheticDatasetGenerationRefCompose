#!/usr/bin/env python3
"""
bbox_visualize.py

Draw bounding boxes from annotated theme metadata onto the source image.
Use this for smoke-testing bbox_annotation output without re-running GDINO.

Typical usage:

    ../.venv/bin/python bbox_visualize.py --theme action --show

    # All themes, save overlays only:
    ../.venv/bin/python bbox_visualize.py
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_METADATA_DIR = SCRIPT_DIR / "outputs/bbox_annotations/metadata"
DEFAULT_DEBUG_DIR = SCRIPT_DIR / "outputs/bbox_annotations/debug_bboxes"

BOX_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Draw bbox overlays from theme metadata for debug/smoke tests."
    )
    p.add_argument(
        "--metadata_dir",
        type=str,
        default=str(DEFAULT_METADATA_DIR),
        help="Directory containing annotated metadata JSON files.",
    )
    p.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Single metadata JSON path (overrides --metadata_dir/--theme).",
    )
    p.add_argument(
        "--theme",
        type=str,
        default=None,
        help="Process only this theme (metadata filename stem).",
    )
    p.add_argument(
        "--debug_dir",
        type=str,
        default=str(DEFAULT_DEBUG_DIR),
        help="Where to write overlay PNGs.",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Open each overlay in the default image viewer after saving.",
    )
    p.add_argument(
        "--line_width",
        type=int,
        default=3,
        help="BBox stroke width in pixels.",
    )
    return p.parse_args()


def load_font(size: int = 16):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        font_path = Path(path)
        if font_path.is_file():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def bbox_items_from_metadata(metadata: dict) -> list[dict]:
    top_level = metadata.get("bbox_information") or {}
    if top_level.get("objects"):
        items = []
        for obj in top_level["objects"]:
            xyxy = obj.get("xyxy")
            if xyxy is None and obj.get("bbox"):
                bbox = obj["bbox"]
                xyxy = {
                    "xmin": bbox[0],
                    "ymin": bbox[1],
                    "xmax": bbox[2],
                    "ymax": bbox[3],
                }
            items.append(
                {
                    "object": obj.get("object"),
                    "bbox_xyxy": xyxy,
                    "bbox_score": obj.get("score"),
                    "bbox_label": obj.get("label"),
                    "bbox_found": obj.get("found", False),
                }
            )
        return items

    return []


def draw_bbox_overlay(
    image: Image.Image,
    bbox_items: list[dict],
    line_width: int = 3,
) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    font = load_font(16)
    small_font = load_font(13)

    missing = []
    found = 0

    for idx, item in enumerate(bbox_items):
        obj = item.get("object", "object")
        color = BOX_COLORS[idx % len(BOX_COLORS)]

        if not item.get("bbox_found") or not item.get("bbox_xyxy"):
            missing.append(obj)
            continue

        found += 1
        box = item["bbox_xyxy"]
        xy = (box["xmin"], box["ymin"], box["xmax"], box["ymax"])
        draw.rectangle(xy, outline=color, width=line_width)

        score = item.get("bbox_score")
        label = f"{obj}"
        if score is not None:
            label = f"{obj} ({score:.2f})"

        text_x = box["xmin"]
        text_y = max(0, box["ymin"] - 20)
        text_bbox = draw.textbbox((text_x, text_y), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((text_x, text_y), label, fill="white", font=font)

    summary = f"found {found}/{len(bbox_items)}"
    if missing:
        summary += f" | missing: {', '.join(missing)}"

    summary_bbox = draw.textbbox((8, 8), summary, font=small_font)
    draw.rectangle(summary_bbox, fill="black")
    draw.text((8, 8), summary, fill="white", font=small_font)

    return overlay


def visualize_metadata_file(
    meta_path: Path,
    debug_dir: Path,
    show: bool = False,
    line_width: int = 3,
) -> Path:
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    theme = metadata.get("theme", meta_path.stem)
    image_path = Path(metadata.get("source_image") or metadata.get("output_image") or "")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found for {theme}: {image_path}")

    bbox_items = bbox_items_from_metadata(metadata)
    if not bbox_items:
        raise ValueError(f"No bbox data in metadata for {theme}")

    image = Image.open(image_path).convert("RGB")
    overlay = draw_bbox_overlay(image, bbox_items, line_width=line_width)

    debug_dir.mkdir(parents=True, exist_ok=True)
    out_path = debug_dir / f"{theme}_bboxes.png"
    overlay.save(out_path)

    if show:
        overlay.show(title=f"{theme} bboxes")

    return out_path


def main():
    args = parse_args()
    debug_dir = Path(args.debug_dir)

    if args.metadata:
        meta_paths = [Path(args.metadata)]
        if not meta_paths[0].is_file():
            print(f"ERROR: metadata not found: {meta_paths[0]}", file=sys.stderr)
            sys.exit(1)
    else:
        metadata_dir = Path(args.metadata_dir)
        if not metadata_dir.is_dir():
            print(f"ERROR: metadata dir not found: {metadata_dir}", file=sys.stderr)
            sys.exit(1)

        meta_paths = sorted(metadata_dir.glob("*.json"))
        if args.theme:
            meta_paths = [metadata_dir / f"{args.theme}.json"]
            if not meta_paths[0].is_file():
                print(f"ERROR: metadata not found for theme: {args.theme}", file=sys.stderr)
                sys.exit(1)

    n_ok, n_failed = 0, 0
    for meta_path in meta_paths:
        theme = meta_path.stem
        try:
            out_path = visualize_metadata_file(
                meta_path,
                debug_dir=debug_dir,
                show=args.show,
                line_width=args.line_width,
            )
            n_ok += 1
            print(f"[{theme}] saved {out_path}")
        except Exception as e:
            n_failed += 1
            print(f"[{theme}] ERROR: {e}", file=sys.stderr)

    print(f"DONE. saved={n_ok} failed={n_failed} debug_dir={debug_dir}")


if __name__ == "__main__":
    main()
