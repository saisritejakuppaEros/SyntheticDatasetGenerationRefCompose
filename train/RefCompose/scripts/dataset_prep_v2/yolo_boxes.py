#!/usr/bin/env python3
"""
Run Ultralytics YOLO on all images under ``output/images`` and write detections to
``output/bbox_results/yolo26_detections.csv``.

Paths default to this script's ``output/`` folder. Override with CLI flags.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def list_images_flat(image_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for suf in IMAGE_SUFFIXES:
        paths.extend(sorted(image_dir.glob(f"*{suf}")))
    return paths


def main() -> None:
    here = Path(__file__).resolve().parent
    default_images = here / "output" / "images"
    default_csv = here / "output" / "bbox_results" / "yolo26_detections.csv"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26n.pt",
        help="Ultralytics weights (e.g. yolo26n.pt)",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=default_images,
        help="Folder of input images (flat; default: <script>/output/images)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=default_csv,
        help="Output CSV path (default: .../output/bbox_results/yolo26_detections.csv)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum confidence threshold",
    )
    args = parser.parse_args()

    image_dir = args.images_dir.resolve()
    output_csv = args.output_csv.resolve()
    os.makedirs(output_csv.parent, exist_ok=True)

    _device_pref = os.environ.get("YOLO_DEVICE", "cuda").lower()
    if _device_pref in ("cuda", "gpu") and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = YOLO(args.model)
    image_paths = list_images_flat(image_dir)
    if not image_paths:
        print(f"No images under {image_dir} ({IMAGE_SUFFIXES})", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(image_paths)} images, running {args.model} on {device!r}...")

    data: list[dict] = []
    skipped = 0

    for im_path in tqdm(image_paths):
        im_str = str(im_path)
        if not im_path.is_file():
            skipped += 1
            tqdm.write(f"[skip] missing: {im_str}", file=sys.stderr)
            continue
        try:
            if im_path.stat().st_size == 0:
                skipped += 1
                tqdm.write(f"[skip] empty file: {im_str}", file=sys.stderr)
                continue
        except OSError as e:
            skipped += 1
            tqdm.write(f"[skip] stat {im_str}: {e}", file=sys.stderr)
            continue

        try:
            results = model(im_str, conf=args.conf, device=device, verbose=False)
        except cv2.error as e:
            skipped += 1
            tqdm.write(f"[skip] OpenCV {im_str}: {e}", file=sys.stderr)
            continue
        except Exception as e:  # noqa: BLE001
            skipped += 1
            tqdm.write(f"[skip] {im_str}: {e}", file=sys.stderr)
            continue

        for result in results:
            if result.boxes is not None and len(result.boxes):
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    cls_name = model.names[cls_id]
                    conf = float(box.conf.item())
                    xyxy = box.xyxy[0].tolist()

                    data.append(
                        {
                            "image_path": im_path.name,
                            "image_abspath": str(im_path.resolve()),
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": conf,
                            "x1": xyxy[0],
                            "y1": xyxy[1],
                            "x2": xyxy[2],
                            "y2": xyxy[3],
                        }
                    )

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

    print(f"\nDone. {len(df)} detections saved to {output_csv}")
    if skipped:
        print(f"Skipped {skipped} unreadable / failed image(s).")
    if len(df):
        print(f"Unique images (basename): {df['image_path'].nunique()}")
        print(f"Unique images (abspath): {df['image_abspath'].nunique()}")


if __name__ == "__main__":
    main()
