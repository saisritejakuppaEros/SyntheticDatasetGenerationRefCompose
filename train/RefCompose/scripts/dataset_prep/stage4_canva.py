#!/usr/bin/env python3
"""
One 1920×1080 canvas per CSV ``image_path``. The CSV has **one row per detection** (each row
is a bbox / "crop"); **all rows** for that image are composited (see ``--min-side``: default
0 keeps every bbox in the CSV, not only large boxes like stage3).

Source frames are **letterboxed** into the canvas (uniform scale, centered, black bars) so
aspect ratio is preserved; bbox corners use the same mapping.

For each row, pixels come either from multiview (when available and selected) or from the
**original image** at ``Path(image-path) / image_path`` via ``Image.crop((x1,y1,x2,y2))``
using CSV coordinates (after clamping), then scaled to the on-canvas bbox and pasted.

If no multiview files exist for that ``crop_no``, or loading fails, or ``--multiview-prob``
skips: use the CSV bbox crop from the original image and overlay it (on black by default).

Paste order: ``crop_no`` 1,2,... first, then ``crop_no == 0`` on top.

``--background black`` (default): black canvas. ``--background scaled``: full frame **letterboxed**
(uniform scale, centered) into 1920×1080, then crops on top (aspect preserved).

Images are processed in parallel with ``ThreadPoolExecutor`` (``--workers``). Each image uses
its own ``random.Random`` so runs are reproducible when ``--seed`` is set.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

CANVAS_W = 1920
CANVAS_H = 1080


def load_and_filter_detections(
    csv_path: Path,
    min_side: float,
    multiview_match_min_side: float,
) -> pd.DataFrame:
    """Order rows like stage3 and assign ``crop_no`` per image (all kept rows).

    ``mv_crop_no`` matches stage3 folder layout: cumcount among rows with w,h >
    ``multiview_match_min_side`` only, so paths ``multiview_dir/stem/{mv_crop_no}/`` stay
    aligned even when ``--min-side`` is 0 and many small CSV boxes are included.
    """
    df = pd.read_csv(csv_path)
    for col in ("x1", "y1", "x2", "y2", "image_path", "class_name", "confidence"):
        if col not in df.columns:
            raise SystemExit(f"CSV missing required column: {col}")
    df = df.copy()
    df["w"] = df["x2"] - df["x1"]
    df["h"] = df["y2"] - df["y1"]
    if min_side > 0:
        df = df[(df["w"] > min_side) & (df["h"] > min_side)]
    else:
        df = df[(df["w"] > 0) & (df["h"] > 0)]
    df = df.sort_values(
        by=["image_path", "confidence", "x1", "y1", "x2", "y2"],
        ascending=[True, False, True, True, True, True],
    )
    df["crop_no"] = df.groupby("image_path", sort=False).cumcount()
    m = (df["w"] > multiview_match_min_side) & (df["h"] > multiview_match_min_side)
    df["mv_crop_no"] = pd.NA
    if m.any():
        df.loc[m, "mv_crop_no"] = df.loc[m].groupby("image_path", sort=False).cumcount()
    return df


def clamp_xyxy(
    x1: float, y1: float, x2: float, y2: float, iw: int, ih: int
) -> tuple[int, int, int, int]:
    x1i = int(round(x1))
    y1i = int(round(y1))
    x2i = int(round(x2))
    y2i = int(round(y2))
    x1i = max(0, min(x1i, iw - 1))
    y1i = max(0, min(y1i, ih - 1))
    x2i = max(x1i + 1, min(x2i, iw))
    y2i = max(y1i + 1, min(y2i, ih))
    return x1i, y1i, x2i, y2i


def letterbox_layout(iw: int, ih: int) -> tuple[float, int, int, int, int]:
    """Uniform scale to fit iw×ih inside CANVAS; returns scale, off_x, off_y, nw, nh."""
    if iw < 1 or ih < 1:
        raise ValueError(f"letterbox_layout: invalid image size {iw}x{ih}")
    scale = min(CANVAS_W / iw, CANVAS_H / ih)
    nw = max(1, min(CANVAS_W, int(round(iw * scale))))
    nh = max(1, min(CANVAS_H, int(round(ih * scale))))
    off_x = (CANVAS_W - nw) // 2
    off_y = (CANVAS_H - nh) // 2
    return scale, off_x, off_y, nw, nh


def scale_box_to_canvas_letterbox(
    x1: int, y1: int, x2: int, y2: int, scale: float, off_x: int, off_y: int
) -> tuple[int, int, int, int]:
    X1 = off_x + int(round(x1 * scale))
    Y1 = off_y + int(round(y1 * scale))
    X2 = off_x + int(round(x2 * scale))
    Y2 = off_y + int(round(y2 * scale))
    X1 = max(0, min(X1, CANVAS_W - 1))
    Y1 = max(0, min(Y1, CANVAS_H - 1))
    X2 = max(X1 + 1, min(X2, CANVAS_W))
    Y2 = max(Y1 + 1, min(Y2, CANVAS_H))
    return X1, Y1, X2, Y2


def list_multiview_pngs(multiview_dir: Path, stem: str, crop_no: int) -> list[Path]:
    d = multiview_dir / stem / str(crop_no)
    if not d.is_dir():
        return []
    pngs = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    return sorted(pngs)


def load_crop_multiview_random(paths: list[Path], rng: random.Random) -> Image.Image | None:
    if not paths:
        return None
    p = rng.choice(paths)
    try:
        return Image.open(p).convert("RGBA")
    except OSError:
        return None


def _rng_for_image(rel: str, base_seed: int | None) -> random.Random:
    h = int(hashlib.md5(rel.encode()).hexdigest()[:8], 16)
    return random.Random((base_seed if base_seed is not None else 0) ^ h)


def load_crop_source(full: Image.Image, x1: int, y1: int, x2: int, y2: int) -> Image.Image:
    return full.crop((x1, y1, x2, y2)).convert("RGBA")


def paste_crop_in_box(canvas: Image.Image, crop: Image.Image, x1: int, y1: int, x2: int, y2: int) -> None:
    tw, th = x2 - x1, y2 - y1
    if tw < 1 or th < 1:
        return
    resized = crop.resize((tw, th), Image.Resampling.LANCZOS)
    if resized.mode == "RGBA":
        canvas.paste(resized, (x1, y1), resized)
    else:
        canvas.paste(resized.convert("RGB"), (x1, y1))


def compose_one_image(
    rel: str,
    g: pd.DataFrame,
    images_dir: Path,
    multiview_dir: Path,
    out_dir: Path,
    min_side: float,
    multiview_prob: float,
    background: str,
    base_seed: int | None,
) -> tuple[str, int, int]:
    """Returns ``(status, used_multiview, used_source)`` where status is
    ``ok`` | ``skip_missing`` | ``skip_unreadable``."""
    stem = Path(rel).stem
    src = images_dir / rel
    rng = _rng_for_image(rel, base_seed)
    if not src.is_file():
        return "skip_missing", 0, 0
    try:
        full = Image.open(src).convert("RGB")
    except OSError:
        return "skip_unreadable", 0, 0

    iw, ih = full.size
    scale, off_x, off_y, nw, nh = letterbox_layout(iw, ih)
    rows = sorted(g.itertuples(index=False), key=lambda r: int(r.crop_no))
    others = [r for r in rows if int(r.crop_no) != 0]
    primary = [r for r in rows if int(r.crop_no) == 0]

    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (0, 0, 0))
    if background == "scaled":
        bg = full.resize((nw, nh), Image.Resampling.LANCZOS).convert("RGB")
        canvas.paste(bg, (off_x, off_y))

    used_multiview = 0
    used_source_crop = 0

    def paste_one(row) -> None:
        nonlocal used_multiview, used_source_crop
        x1, y1, x2, y2 = clamp_xyxy(row.x1, row.y1, row.x2, row.y2, iw, ih)
        if min_side > 0 and ((x2 - x1) <= min_side or (y2 - y1) <= min_side):
            return
        if x2 <= x1 or y2 <= y1:
            return
        X1, Y1, X2, Y2 = scale_box_to_canvas_letterbox(x1, y1, x2, y2, scale, off_x, off_y)

        crop: Image.Image | None = None
        mvn = getattr(row, "mv_crop_no", pd.NA)
        if pd.isna(mvn):
            mv_paths: list[Path] = []
        else:
            mv_paths = list_multiview_pngs(multiview_dir, stem, int(mvn))
        if mv_paths and rng.random() < multiview_prob:
            crop = load_crop_multiview_random(mv_paths, rng)
            if crop is not None:
                used_multiview += 1
        if crop is None:
            crop = load_crop_source(full, x1, y1, x2, y2)
            used_source_crop += 1
        paste_crop_in_box(canvas, crop, X1, Y1, X2, Y2)

    for row in others:
        paste_one(row)
    for row in primary:
        paste_one(row)

    out_path = out_dir / f"{stem}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return "ok", used_multiview, used_source_crop


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSV bboxes from originals; overlay multiview when present else bbox crop."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="YOLO detections CSV (same as stage2/stage3).",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        required=True,
        help="Root folder for originals; each CSV image_path is opened as image-path / image_path.",
    )
    parser.add_argument(
        "--multiview-dir",
        type=Path,
        default=Path("output/multiview_out"),
        help="Stage3 output root: multiview_dir / stem / crop_no / N.png",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory: one PNG per image: out_dir / stem.png",
    )
    parser.add_argument(
        "--min-side",
        type=float,
        default=0.0,
        help="If >0, keep boxes with w,h > this. If 0, composite every CSV bbox row (many crops).",
    )
    parser.add_argument(
        "--multiview-match-min-side",
        type=float,
        default=200.0,
        help="Match stage3 multiview dirs: only rows with w,h > this get mv_crop_no for lookups.",
    )
    parser.add_argument(
        "--multiview-prob",
        type=float,
        default=0.5,
        help="When multiview PNGs exist for that crop_no: probability to use one vs CSV bbox crop.",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (optional).")
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Process at most this many distinct images after filtering.",
    )
    parser.add_argument(
        "--background",
        choices=("black", "scaled"),
        default="black",
        help="Canvas base: black, or full image letterboxed into 1920×1080 then crops overlaid on top.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(8, min(32, (os.cpu_count() or 4) * 2)),
        help="Thread pool size for parallel per-image compositing (1 = sequential).",
    )
    args = parser.parse_args()

    if not (0.0 <= args.multiview_prob <= 1.0):
        raise SystemExit("--multiview-prob must be in [0, 1]")
    if args.min_side < 0:
        raise SystemExit("--min-side must be >= 0")
    if args.multiview_match_min_side < 0:
        raise SystemExit("--multiview-match-min-side must be >= 0")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")
    images_dir = args.image_path.resolve()
    if not images_dir.is_dir():
        raise SystemExit(f"image-path not found: {images_dir}")

    multiview_dir = args.multiview_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_filter_detections(
        csv_path, args.min_side, args.multiview_match_min_side
    )
    if len(df) == 0:
        print("No rows after filtering; nothing to do.")
        return

    groups = list(df.groupby("image_path", sort=False))
    if args.max_images is not None:
        groups = groups[: args.max_images]

    task_args = [
        (
            rel,
            g,
            images_dir,
            multiview_dir,
            out_dir,
            args.min_side,
            args.multiview_prob,
            args.background,
            args.seed,
        )
        for rel, g in groups
    ]

    used_multiview = 0
    used_source_crop = 0
    images_written = 0
    skipped_missing = 0
    skipped_unreadable = 0

    def handle_result(status: str, mv: int, src: int) -> None:
        nonlocal used_multiview, used_source_crop, images_written
        nonlocal skipped_missing, skipped_unreadable
        if status == "ok":
            images_written += 1
            used_multiview += mv
            used_source_crop += src
        elif status == "skip_missing":
            skipped_missing += 1
        else:
            skipped_unreadable += 1

    if args.workers <= 1:
        for t in tqdm(task_args, desc="canvas"):
            handle_result(*compose_one_image(*t))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(compose_one_image, *t) for t in task_args]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="canvas"):
                handle_result(*fut.result())

    skipped = skipped_missing + skipped_unreadable
    print(
        f"Done. wrote {images_written} images under {out_dir} (workers={args.workers}) "
        f"(multiview layers: {used_multiview}, bbox crops from originals: {used_source_crop}, "
        f"skipped: {skipped} missing={skipped_missing} unreadable={skipped_unreadable})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()


# python stage4_canva.py \
#   --csv output/bbox_results/yolo26_detections.csv \
#   --image-path output/images \
#   --multiview-dir output/multiview_out \
#   --out-dir output/canvas_out \
#   --min-side 0 \
#   --multiview-match-min-side 200 \
#   --background black \
#   --max-images 10 \
#   --seed 42