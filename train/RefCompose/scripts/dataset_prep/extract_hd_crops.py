#!/usr/bin/env python3
"""
Extract images from dataset .tar archives, resize+crop to 1280x720 (16:9) with no
letterboxing: scale so the image fully covers the target, then center-crop.

Default: process only the first tar (sorted by name) for a quick test.
Use --all to process every .tar in the images directory.

Output layout: <output_dir>/<tar_stem>/<original_basename_without_ext>.png
"""

from __future__ import annotations

import argparse
import io
import sys
import tarfile
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm

TARGET_W = 1280
TARGET_H = 720

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}


def crop_hd_cover(img: Image.Image) -> Image.Image:
    """
    Scale image so it fully covers TARGET_W x TARGET_H, then center-crop.
    No black bars on left/right (or top/bottom): excess is cropped, not padded.
    """
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")

    w, h = img.size
    if w <= 0 or h <= 0:
        raise ValueError("invalid image size")

    # Scale so both dimensions are >= target (same as max(target_w/w, target_h/h))
    scale = max(TARGET_W / w, TARGET_H / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    left = (new_w - TARGET_W) // 2
    top = (new_h - TARGET_H) // 2
    right = left + TARGET_W
    bottom = top + TARGET_H
    return resized.crop((left, top, right, bottom))


def is_image_member(name: str) -> bool:
    p = Path(name)
    if not p.name or p.name.startswith("."):
        return False
    return p.suffix.lower() in IMAGE_SUFFIXES


def _collect_image_members(tf: tarfile.TarFile) -> list[tarfile.TarInfo]:
    out: list[tarfile.TarInfo] = []
    for member in tf.getmembers():
        if not member.isfile():
            continue
        norm = member.name.lstrip("./")
        if is_image_member(norm):
            out.append(member)
    return out


def _member_dest_paths(member: tarfile.TarInfo, out_dir: Path) -> tuple[str, Path]:
    norm = member.name.lstrip("./")
    base = Path(norm).name
    stem = Path(base).stem
    return norm, out_dir / f"{stem}.png"


def _decode_crop_save_png(
    data: bytes | None,
    dest: Path,
    tar_name: str,
    norm: str,
) -> bool:
    if data is None:
        return False
    try:
        with Image.open(io.BytesIO(data)) as im:
            out = crop_hd_cover(im)
            out.save(dest, format="PNG", optimize=True)
        return True
    except Exception as e:  # noqa: BLE001
        tqdm.write(f"[warn] {tar_name} :: {norm}: {e}", file=sys.stderr)
        return False


def _process_one_member(
    tf: tarfile.TarFile,
    member: tarfile.TarInfo,
    out_dir: Path,
    tar_name: str,
) -> bool:
    norm, dest = _member_dest_paths(member, out_dir)
    try:
        f = tf.extractfile(member)
        if f is None:
            return False
        data = f.read()
    except Exception as e:  # noqa: BLE001
        tqdm.write(f"[warn] {tar_name} :: {norm}: {e}", file=sys.stderr)
        return False
    return _decode_crop_save_png(data, dest, tar_name, norm)


def _process_tar_sequential(
    tf: tarfile.TarFile,
    tar_path: Path,
    out_dir: Path,
    image_members: list[tarfile.TarInfo],
    *,
    limit_images: int | None,
    inner_leave: bool,
    inner_position: int,
) -> tuple[int, int]:
    tar_stem = tar_path.stem
    desc = f"{tar_stem} imgs"
    ok = 0
    skipped = 0

    if limit_images is not None:
        with tqdm(
            total=limit_images,
            desc=desc,
            unit="img",
            leave=inner_leave,
            position=inner_position,
        ) as pbar:
            for member in image_members:
                if ok >= limit_images:
                    break
                if _process_one_member(tf, member, out_dir, tar_path.name):
                    ok += 1
                    pbar.update(1)
                else:
                    skipped += 1
                pbar.set_postfix(ok=ok, skip=skipped, refresh=False)
    else:
        for member in tqdm(
            image_members,
            total=len(image_members),
            desc=desc,
            unit="img",
            leave=inner_leave,
            position=inner_position,
        ):
            if _process_one_member(tf, member, out_dir, tar_path.name):
                ok += 1
            else:
                skipped += 1

    return ok, skipped


def _process_tar_parallel(
    tf: tarfile.TarFile,
    tar_path: Path,
    out_dir: Path,
    image_members: list[tarfile.TarInfo],
    *,
    workers: int,
    inner_leave: bool,
    inner_position: int,
) -> tuple[int, int]:
    """Tar reads stay on one thread; workers decode/crop/save PNG in parallel."""
    tar_stem = tar_path.stem
    desc = f"{tar_stem} imgs"
    total = len(image_members)
    ok = 0
    skipped = 0
    pbar_lock = threading.Lock()
    max_inflight = max(workers * 2, workers + 2)

    def drain_completed(pending: set, pbar: tqdm) -> None:
        nonlocal ok, skipped
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for fut in done:
            pending.remove(fut)
            success = fut.result()
            with pbar_lock:
                if success:
                    ok += 1
                else:
                    skipped += 1
                pbar.update(1)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        with tqdm(
            total=total,
            desc=desc,
            unit="img",
            leave=inner_leave,
            position=inner_position,
        ) as pbar:
            pending: set = set()
            for member in image_members:
                norm, dest = _member_dest_paths(member, out_dir)
                try:
                    fh = tf.extractfile(member)
                    data = None if fh is None else fh.read()
                except Exception as e:  # noqa: BLE001
                    tqdm.write(f"[warn] {tar_path.name} :: {norm}: {e}", file=sys.stderr)
                    data = None

                fut = ex.submit(
                    _decode_crop_save_png,
                    data,
                    dest,
                    tar_path.name,
                    norm,
                )
                pending.add(fut)
                while len(pending) >= max_inflight:
                    drain_completed(pending, pbar)

            while pending:
                drain_completed(pending, pbar)

    return ok, skipped


def process_tar(
    tar_path: Path,
    output_root: Path,
    *,
    limit_images: int | None = None,
    workers: int = 16,
    inner_leave: bool = False,
    inner_position: int = 0,
) -> tuple[int, int]:
    """
    Returns (ok_count, skip_count).

    Uses thread pool for decode/crop/save when workers > 1 and limit_images is None.
    With --limit-images, runs single-threaded so “N successful outputs” follows tar order.
    """
    tar_stem = tar_path.stem
    out_dir = output_root / tar_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r:*") as tf:
        image_members = _collect_image_members(tf)
        if limit_images is not None or workers <= 1:
            return _process_tar_sequential(
                tf,
                tar_path,
                out_dir,
                image_members,
                limit_images=limit_images,
                inner_leave=inner_leave,
                inner_position=inner_position,
            )
        return _process_tar_parallel(
            tf,
            tar_path,
            out_dir,
            image_members,
            workers=workers,
            inner_leave=inner_leave,
            inner_position=inner_position,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("/mnt/data0/teja/multiref_image/dataset/images"),
        help="Directory containing .tar archives",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "hd_1280x720",
        help="Root output directory (tar stem becomes subfolder)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every .tar in images-dir (default: first tar only, for testing)",
    )
    parser.add_argument(
        "--limit-images",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N images per tar (for quick smoke tests)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        metavar="N",
        help="Thread pool size for decode/crop/save (ignored with --limit-images; use 1 to disable)",
    )
    args = parser.parse_args()

    images_dir = args.images_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tars = sorted(images_dir.glob("*.tar"))
    if not tars:
        print(f"No .tar files in {images_dir}", file=sys.stderr)
        sys.exit(1)

    if args.all:
        to_run = tars
        tqdm.write(f"Processing {len(to_run)} tar file(s) -> {output_dir}")
    else:
        to_run = [tars[0]]
        tqdm.write(
            f"TEST MODE: processing first tar only: {to_run[0].name}\n"
            f"  (pass --all to process all {len(tars)} archives)\n"
            f"  output -> {output_dir / to_run[0].stem}/"
        )

    limit = args.limit_images
    use_outer = len(to_run) > 1
    outer_iter = (
        tqdm(
            to_run,
            desc="Archives",
            unit="tar",
            position=0,
            leave=True,
        )
        if use_outer
        else to_run
    )

    for i, tp in enumerate(outer_iter):
        inner_leave = not use_outer or (i == len(to_run) - 1)
        o, s = process_tar(
            tp,
            output_dir,
            limit_images=limit,
            workers=max(1, args.workers),
            inner_leave=inner_leave,
            inner_position=1 if use_outer else 0,
        )
        tqdm.write(f"  {tp.name}: wrote {o} png(s), skipped {s}")


if __name__ == "__main__":
    main()
