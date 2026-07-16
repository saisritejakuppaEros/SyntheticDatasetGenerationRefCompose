#!/usr/bin/env python3
"""
Depth Anything 3 (DA3-BASE) for ``dataset_prep_v2``.

Reads RGB images from ``output/images`` (same basename as ``unq_id.png``), runs
``DepthAnything3`` from Hugging Face ``depth-anything/da3-base``, and writes
``output/depth/{unq_id}.png`` (16-bit relative depth, same stem as the input).

Requires the Depth Anything 3 package:

  git clone https://github.com/ByteDance-Seed/Depth-Anything-3
  cd Depth-Anything-3
  pip install -e .

See also: https://huggingface.co/depth-anything/DA3-BASE
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "OpenCV (cv2) is required for resizing and saving 16-bit depth PNGs. "
        "Install with: pip install opencv-python-headless"
    ) from e

try:
    import torch
    from depth_anything_3.api import DepthAnything3
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Missing depth_anything_3 or torch. Install DA3 from source:\n"
        "  git clone https://github.com/ByteDance-Seed/Depth-Anything-3\n"
        "  cd Depth-Anything-3 && pip install -e .\n"
        "Then re-run this script."
    ) from e


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(images_dir: Path) -> list[Path]:
    paths = []
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            paths.append(p)
    return paths


def depth_to_uint16(depth: np.ndarray) -> np.ndarray:
    """Per-image min–max normalization to uint16 (relative depth, not metric)."""
    d = np.asarray(depth, dtype=np.float64)
    lo, hi = float(d.min()), float(d.max())
    if hi - lo < 1e-8:
        return np.zeros(d.shape, dtype=np.uint16)
    return ((d - lo) / (hi - lo) * 65535.0).astype(np.uint16)


def resize_depth(depth: np.ndarray, out_wh: tuple[int, int]) -> np.ndarray:
    """``depth`` is (H, W); ``out_wh`` is (width, height) for cv2."""
    w, h = out_wh
    if depth.shape[1] == w and depth.shape[0] == h:
        return depth
    return cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)


def optional_rgb_preview(depth_hw: np.ndarray) -> np.ndarray:
    """uint8 BGR three-channel preview (turbo colormap) for quick visual checks."""
    d = depth_hw.astype(np.float64)
    lo, hi = d.min(), d.max()
    if hi - lo < 1e-8:
        g = np.zeros_like(d, dtype=np.uint8)
    else:
        g = ((d - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(g, cv2.COLORMAP_TURBO)


def main() -> None:
    root = Path(__file__).resolve().parent
    default_images = root / "output" / "images"
    default_depth = root / "output" / "depth"

    parser = argparse.ArgumentParser(description="Generate depth PNGs with Depth Anything 3.")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=default_images,
        help=f"Folder of input images (default: {default_images}).",
    )
    parser.add_argument(
        "--depth-dir",
        type=Path,
        default=default_depth,
        help=f"Output folder for depth maps (default: {default_depth}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="depth-anything/da3-base",
        help="Hugging Face model id for DepthAnything3.from_pretrained.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Number of images per model.inference call. Use 1 when images have different "
            "aspect ratios (DA3 may center-crop batches to a common size). "
            "Increase only if your images share the same processed resolution."
        ),
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="DA3 internal processing resolution (longest side bound, see DA3 docs).",
    )
    parser.add_argument(
        "--process-res-method",
        type=str,
        default="upper_bound_resize",
        help="Resize policy passed to model.inference (default: upper_bound_resize).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device, e.g. cuda:0 or cpu. Default: cuda if available else cpu.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip writing if <stem>.png already exists in --depth-dir.",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Also save float32 raw depth (resized to input resolution) as <stem>.npy.",
    )
    parser.add_argument(
        "--rgb-preview",
        action="store_true",
        help="Also save an 8-bit BGR turbo preview as <stem>_preview.png.",
    )
    args = parser.parse_args()

    images_dir: Path = args.images_dir
    depth_dir: Path = args.depth_dir
    if not images_dir.is_dir():
        raise SystemExit(f"Images directory does not exist: {images_dir}")

    depth_dir.mkdir(parents=True, exist_ok=True)

    paths = list_images(images_dir)
    if not paths:
        raise SystemExit(f"No images found under {images_dir} (supported: {sorted(IMAGE_SUFFIXES)})")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DepthAnything3.from_pretrained(args.model)
    model = model.to(device=device)
    model.eval()

    def process_batch(batch_paths: list[Path]) -> None:
        str_paths = [str(p) for p in batch_paths]
        orig_sizes: list[tuple[int, int]] = []
        for p in batch_paths:
            with Image.open(p) as im:
                w, h = im.size
            orig_sizes.append((w, h))

        prediction = model.inference(
            str_paths,
            export_dir=None,
            process_res=args.process_res,
            process_res_method=args.process_res_method,
        )
        depths = prediction.depth  # [N, H, W] float32
        for i, src in enumerate(batch_paths):
            stem = src.stem
            out_png = depth_dir / f"{stem}.png"
            if args.skip_existing and out_png.is_file():
                continue

            d = np.asarray(depths[i], dtype=np.float32)
            d = resize_depth(d, orig_sizes[i])
            u16 = depth_to_uint16(d)
            if not cv2.imwrite(str(out_png), u16):
                raise RuntimeError(f"Failed to write depth PNG: {out_png}")

            if args.save_npy:
                npy_path = depth_dir / f"{stem}.npy"
                np.save(npy_path, d.astype(np.float32))

            if args.rgb_preview:
                prev = optional_rgb_preview(d)
                prev_path = depth_dir / f"{stem}_preview.png"
                if not cv2.imwrite(str(prev_path), prev):
                    raise RuntimeError(f"Failed to write preview: {prev_path}")

    bs = max(1, int(args.batch_size))
    for start in tqdm(range(0, len(paths), bs), desc="batches"):
        batch = paths[start : start + bs]
        process_batch(batch)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
