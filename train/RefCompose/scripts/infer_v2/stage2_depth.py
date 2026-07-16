#!/usr/bin/env python3
"""
Monocular relative depth from a raster image using Depth Anything 3 (DA3-BASE).

Requires: pip install -r requirements-da3.txt
Hub: https://huggingface.co/depth-anything/DA3-BASE

Depth PNG format matches training consumption in ``train/src/canvas_dataset.py``:
``load_depth_image_as_rgb_pil`` — **16-bit grayscale** ``I;16`` with linear values in
``[0, 65535]`` (not false-color RGB).

Examples::

  CUDA_VISIBLE_DEVICES=3 python stage2_depth.py

  python stage2_depth.py -i generated_image/image_gen.png

  python stage2_depth.py -i generated_image -o generated_image/depth

  python stage2_depth.py -i outputs --batch
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_INPUT = _SCRIPT_DIR / "generated_image" / "image_gen.png"
_DEFAULT_MODEL = "depth-anything/DA3-BASE"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _relative_depth_to_u16_hw(depth_hw: np.ndarray) -> np.ndarray:
    """Map relative depth (float) to uint16 [0, 65535] for ``I;16`` PNG."""
    d = depth_hw.astype(np.float32)
    lo, hi = float(np.percentile(d, 2.0)), float(np.percentile(d, 98.0))
    if hi <= lo + 1e-6:
        lo, hi = float(d.min()), float(d.max())
    if hi <= lo + 1e-6:
        hi = lo + 1.0
    x = (d - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)
    return np.round(x * 65535.0).astype(np.uint16)


def _import_depth_anything3():
    try:
        from depth_anything_3.api import DepthAnything3

        return DepthAnything3
    except ImportError as e:
        print(
            "Missing package depth_anything_3. Install with:\n"
            f"  pip install -r {_SCRIPT_DIR / 'requirements-da3.txt'}\n",
            file=sys.stderr,
        )
        raise e


def _infer_depth_u16(
    model,
    device: torch.device,
    input_path: Path,
    *,
    process_res: int = 504,
) -> tuple[np.ndarray, int, int, np.ndarray]:
    """Return ``(u16_hw, orig_w, orig_h, depth_up_float32)`` for one image."""
    input_path = Path(input_path).resolve()
    with Image.open(input_path) as pil_in:
        pil_in = pil_in.convert("RGB")
        orig_w, orig_h = pil_in.size

    prediction = model.inference(
        [str(input_path)],
        process_res=process_res,
        process_res_method="upper_bound_resize",
        export_dir=None,
    )
    depth = np.asarray(prediction.depth[0], dtype=np.float32)

    depth_t = torch.from_numpy(depth)[None, None]
    depth_up = (
        F.interpolate(depth_t, size=(orig_h, orig_w), mode="bilinear", align_corners=False)[
            0, 0
        ]
        .cpu()
        .numpy()
    )
    u16 = _relative_depth_to_u16_hw(depth_up)
    return u16, orig_w, orig_h, depth_up


def _iter_batch_inputs(scan_root: Path) -> list[Path]:
    """All raster images under ``scan_root``, excluding ``.../depth/...`` and ``*_depth.*``."""
    scan_root = Path(scan_root).resolve()
    if not scan_root.is_dir():
        raise NotADirectoryError(f"Not a directory: {scan_root}")

    out: list[Path] = []
    for p in sorted(scan_root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        try:
            rel = p.relative_to(scan_root)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] == "depth":
            continue
        stem = p.stem.lower()
        if stem.endswith("_depth") or stem.endswith("_depth_raw"):
            continue
        out.append(p)
    return out


def run_depth(
    input_path: Path,
    *,
    output_depth_png: Path | None = None,
    model_id: str = _DEFAULT_MODEL,
    save_raw_npy: bool = False,
    process_res: int = 504,
) -> Path:
    DepthAnything3 = _import_depth_anything3()

    input_path = Path(input_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    if output_depth_png is None:
        output_depth_png = input_path.with_name(f"{input_path.stem}_depth.png")
    else:
        output_depth_png = Path(output_depth_png).expanduser().resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading DA3 model {model_id!r} on {device} …")
    model = DepthAnything3.from_pretrained(model_id)
    model = model.to(device=device)

    u16, orig_w, orig_h, depth_up = _infer_depth_u16(
        model, device, input_path, process_res=process_res
    )

    out_img = Image.fromarray(u16, mode="I;16")
    output_depth_png.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(output_depth_png, compress_level=6)
    print(
        f"Saved depth PNG I;16 {orig_w}x{orig_h} (linear 0..65535) → {output_depth_png}"
    )

    if save_raw_npy:
        raw_path = output_depth_png.with_name(f"{input_path.stem}_depth_raw.npy")
        np.save(raw_path, depth_up.astype(np.float32))
        print(f"Saved raw depth array → {raw_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_depth_png


def run_depth_batch(
    scan_root: Path,
    depth_out_root: Path,
    *,
    model_id: str = _DEFAULT_MODEL,
    save_raw_npy: bool = False,
    process_res: int = 504,
    skip_existing: bool = False,
) -> list[Path]:
    """Process every image under ``scan_root``; write mirrored paths under ``depth_out_root``."""
    DepthAnything3 = _import_depth_anything3()
    scan_root = Path(scan_root).expanduser().resolve()
    depth_out_root = Path(depth_out_root).expanduser().resolve()

    inputs = _iter_batch_inputs(scan_root)
    if not inputs:
        print(f"No images found under {scan_root}", file=sys.stderr)
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading DA3 model {model_id!r} on {device} … ({len(inputs)} image(s))")
    model = DepthAnything3.from_pretrained(model_id)
    model = model.to(device=device)

    saved: list[Path] = []
    for i, input_path in enumerate(inputs, start=1):
        rel = input_path.relative_to(scan_root)
        out_png = depth_out_root / rel.parent / f"{input_path.stem}_depth.png"
        if skip_existing and out_png.is_file():
            print(f"[{i}/{len(inputs)}] skip (exists): {out_png}")
            continue

        print(f"[{i}/{len(inputs)}] {input_path} → {out_png}")
        u16, orig_w, orig_h, depth_up = _infer_depth_u16(
            model, device, input_path, process_res=process_res
        )

        out_img = Image.fromarray(u16, mode="I;16")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        out_img.save(out_png, compress_level=6)
        print(f"    Saved I;16 {orig_w}x{orig_h} → {out_png}")
        saved.append(out_png)

        if save_raw_npy:
            raw_path = out_png.with_name(f"{input_path.stem}_depth_raw.npy")
            np.save(raw_path, depth_up.astype(np.float32))
            print(f"    Saved raw → {raw_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return saved


def main() -> int:
    p = argparse.ArgumentParser(description="Depth Anything 3 — relative depth PNG (I;16).")
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        help=f"Input image or directory (default: {_DEFAULT_INPUT}).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output depth PNG (single file) or output root (batch). Default: <stem>_depth.png beside input.",
    )
    p.add_argument(
        "--batch",
        action="store_true",
        help="Treat --input as a directory and process all images under it.",
    )
    p.add_argument("--model", default=os.environ.get("DA3_MODEL", _DEFAULT_MODEL))
    p.add_argument("--process-res", type=int, default=504)
    p.add_argument("--save-raw-npy", action="store_true", help="Also write <stem>_depth_raw.npy")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    input_path = args.input.expanduser().resolve()
    use_batch = args.batch or input_path.is_dir()

    if use_batch:
        if not input_path.is_dir():
            print(f"--batch requires a directory; got file: {input_path}", file=sys.stderr)
            return 2
        depth_root = (
            args.output.expanduser().resolve()
            if args.output is not None
            else input_path / "depth"
        )
        print(f"Batch mode: scan={input_path}\n         depth_out={depth_root}")
        run_depth_batch(
            input_path,
            depth_root,
            model_id=args.model,
            save_raw_npy=args.save_raw_npy,
            process_res=args.process_res,
            skip_existing=args.skip_existing,
        )
        return 0

    run_depth(
        input_path,
        output_depth_png=args.output,
        model_id=args.model,
        save_raw_npy=args.save_raw_npy,
        process_res=args.process_res,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
