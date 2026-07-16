#!/usr/bin/env python3
"""
Extract images from dataset .tar archives, resize+crop to 1280x720 (16:9) with no
letterboxing, and write flat outputs under ``output/``:

  output/images/{unq_id}.png
  output/image_captions/{unq_id}.txt

``unq_id`` is ``{tar_stem}_{image_stem}`` (e.g. ``0000_aue06l4Q9pg``) so ids are unique
across archives. Captions are taken from ``metadata.jsonl`` (``file_name`` + ``text``).

By default **all** ``*.tar`` archives under ``--images-dir`` are processed (full
dataset). Use ``--first-tar-only`` for a quick smoke test on the first archive only.

``--limit-total N`` caps how many images are written **across all archives** (e.g.
50k from the whole dataset). ``--limit-images N`` caps each archive separately
(usually for debugging one tar).
"""

from __future__ import annotations

import argparse
import io
import json
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


class _WriteCap:
    """Allow at most ``limit`` successful PNG writes (thread-safe)."""

    __slots__ = ("_limit", "_n", "_lock")

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._n = 0
        self._lock = threading.Lock()

    def try_reserve(self) -> bool:
        with self._lock:
            if self._n >= self._limit:
                return False
            self._n += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self._n > 0:
                self._n -= 1


def load_metadata_captions(metadata_path: Path) -> dict[tuple[str, str], str]:
    """Map (tar_stem, image_stem) -> caption text from HuggingFace-style jsonl."""
    out: dict[tuple[str, str], str] = {}
    if not metadata_path.is_file():
        return out
    with metadata_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            fn = obj.get("file_name") or ""
            text = obj.get("text")
            if text is None:
                text = ""
            parts = Path(fn).parts
            if len(parts) >= 3 and parts[0] == "images":
                tar_s, base = parts[1], parts[2]
            elif len(parts) >= 2:
                tar_s, base = parts[-2], parts[-1]
            else:
                continue
            stem = Path(base).stem
            out[(tar_s, stem)] = str(text)
    return out


def crop_hd_cover(img: Image.Image) -> Image.Image:
    """Scale image so it fully covers TARGET_W x TARGET_H, then center-crop."""
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")

    w, h = img.size
    if w <= 0 or h <= 0:
        raise ValueError("invalid image size")

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


def _member_paths(
    member: tarfile.TarInfo,
    tar_stem: str,
    output_root: Path,
    caption_index: dict[tuple[str, str], str],
) -> tuple[str, str, Path, Path, str | None]:
    norm = member.name.lstrip("./")
    base = Path(norm).name
    stem = Path(base).stem
    unq_id = f"{tar_stem}_{stem}"
    dest_img = output_root / "images" / f"{unq_id}.png"
    dest_txt = output_root / "image_captions" / f"{unq_id}.txt"
    caption = caption_index.get((tar_stem, stem))
    return norm, unq_id, dest_img, dest_txt, caption


def _decode_crop_save_png(
    data: bytes | None,
    dest_img: Path,
    dest_txt: Path,
    caption: str | None,
    tar_name: str,
    norm: str,
    *,
    write_empty_caption_if_missing: bool,
    write_cap: _WriteCap | None = None,
) -> bool:
    if data is None:
        return False
    try:
        with Image.open(io.BytesIO(data)) as im:
            out = crop_hd_cover(im)
        if write_cap is not None and not write_cap.try_reserve():
            return False
        try:
            dest_img.parent.mkdir(parents=True, exist_ok=True)
            out.save(dest_img, format="PNG", optimize=True)
            dest_txt.parent.mkdir(parents=True, exist_ok=True)
            if caption is not None:
                dest_txt.write_text(caption, encoding="utf-8")
            elif write_empty_caption_if_missing:
                dest_txt.write_text("", encoding="utf-8")
        except Exception:
            if write_cap is not None:
                write_cap.release()
            raise
        return True
    except Exception as e:  # noqa: BLE001
        tqdm.write(f"[warn] {tar_name} :: {norm}: {e}", file=sys.stderr)
        return False


def _process_one_member(
    tf: tarfile.TarFile,
    member: tarfile.TarInfo,
    tar_stem: str,
    output_root: Path,
    caption_index: dict[tuple[str, str], str],
    tar_name: str,
    *,
    write_empty_caption_if_missing: bool,
) -> bool:
    norm, _unq_id, dest_img, dest_txt, caption = _member_paths(
        member, tar_stem, output_root, caption_index
    )
    try:
        f = tf.extractfile(member)
        if f is None:
            return False
        data = f.read()
    except Exception as e:  # noqa: BLE001
        tqdm.write(f"[warn] {tar_name} :: {norm}: {e}", file=sys.stderr)
        return False
    return _decode_crop_save_png(
        data,
        dest_img,
        dest_txt,
        caption,
        tar_name,
        norm,
        write_empty_caption_if_missing=write_empty_caption_if_missing,
        write_cap=None,
    )


def _process_tar_sequential(
    tf: tarfile.TarFile,
    tar_path: Path,
    tar_stem: str,
    output_root: Path,
    caption_index: dict[tuple[str, str], str],
    image_members: list[tarfile.TarInfo],
    *,
    limit_images: int | None,
    inner_leave: bool,
    inner_position: int,
    write_empty_caption_if_missing: bool,
) -> tuple[int, int]:
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
                if _process_one_member(
                    tf,
                    member,
                    tar_stem,
                    output_root,
                    caption_index,
                    tar_path.name,
                    write_empty_caption_if_missing=write_empty_caption_if_missing,
                ):
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
            if _process_one_member(
                tf,
                member,
                tar_stem,
                output_root,
                caption_index,
                tar_path.name,
                write_empty_caption_if_missing=write_empty_caption_if_missing,
            ):
                ok += 1
            else:
                skipped += 1

    return ok, skipped


def _process_tar_parallel(
    tf: tarfile.TarFile,
    tar_path: Path,
    tar_stem: str,
    output_root: Path,
    caption_index: dict[tuple[str, str], str],
    image_members: list[tarfile.TarInfo],
    *,
    workers: int,
    inner_leave: bool,
    inner_position: int,
    write_empty_caption_if_missing: bool,
    limit_images: int | None = None,
) -> tuple[int, int]:
    desc = f"{tar_stem} imgs"
    total = len(image_members)
    ok = 0
    skipped = 0
    pbar_lock = threading.Lock()
    max_inflight = max(workers * 2, workers + 2)
    write_cap: _WriteCap | None = (
        _WriteCap(limit_images) if limit_images is not None else None
    )

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
                with pbar_lock:
                    if limit_images is not None and ok >= limit_images:
                        break
                norm, _u, dest_img, dest_txt, caption = _member_paths(
                    member, tar_stem, output_root, caption_index
                )
                try:
                    fh = tf.extractfile(member)
                    data = None if fh is None else fh.read()
                except Exception as e:  # noqa: BLE001
                    tqdm.write(f"[warn] {tar_path.name} :: {norm}: {e}", file=sys.stderr)
                    data = None

                fut = ex.submit(
                    _decode_crop_save_png,
                    data,
                    dest_img,
                    dest_txt,
                    caption,
                    tar_path.name,
                    norm,
                    write_empty_caption_if_missing=write_empty_caption_if_missing,
                    write_cap=write_cap,
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
    caption_index: dict[tuple[str, str], str],
    *,
    limit_images: int | None = None,
    workers: int = 16,
    inner_leave: bool = False,
    inner_position: int = 0,
    write_empty_caption_if_missing: bool = True,
) -> tuple[int, int]:
    tar_stem = tar_path.stem
    (output_root / "images").mkdir(parents=True, exist_ok=True)
    (output_root / "image_captions").mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r:*") as tf:
        image_members = _collect_image_members(tf)
        if workers <= 1:
            return _process_tar_sequential(
                tf,
                tar_path,
                tar_stem,
                output_root,
                caption_index,
                image_members,
                limit_images=limit_images,
                inner_leave=inner_leave,
                inner_position=inner_position,
                write_empty_caption_if_missing=write_empty_caption_if_missing,
            )
        return _process_tar_parallel(
            tf,
            tar_path,
            tar_stem,
            output_root,
            caption_index,
            image_members,
            workers=workers,
            inner_leave=inner_leave,
            inner_position=inner_position,
            write_empty_caption_if_missing=write_empty_caption_if_missing,
            limit_images=limit_images,
        )


def main() -> None:
    here = Path(__file__).resolve().parent
    default_out = here / "output"
    default_images = Path("/mnt/data0/teja/multiref_image/dataset/images")
    default_meta = Path("/mnt/data0/teja/multiref_image/dataset/metadata.jsonl")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=default_images,
        help="Directory containing .tar archives",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_out,
        help="Root folder; writes output/images and output/image_captions",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=default_meta,
        help="metadata.jsonl with file_name and text fields",
    )
    parser.add_argument(
        "--skip-caption-if-missing",
        action="store_true",
        help="Do not write .txt when no metadata line matches (default: write empty .txt)",
    )
    parser.add_argument(
        "--first-tar-only",
        action="store_true",
        help="Process only the first .tar (smoke test). Default: all archives.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="[Redundant] Process every .tar; this is already the default.",
    )
    parser.add_argument(
        "--limit-total",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N successful images in total across all archives (e.g. 50000)",
    )
    parser.add_argument(
        "--limit-images",
        type=int,
        default=None,
        metavar="N",
        help="Max successful images per .tar (optional; combine with --limit-total)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        metavar="N",
        help="Thread pool size for decode/crop/save (tar reads stay serial; use 1 to force single-threaded)",
    )
    args = parser.parse_args()

    images_dir = args.images_dir.resolve()
    output_root = args.output_root.resolve()
    metadata_path = args.metadata.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    caption_index = load_metadata_captions(metadata_path)
    tqdm.write(
        f"Loaded {len(caption_index)} caption(s) from {metadata_path}",
    )

    tars = sorted(images_dir.glob("*.tar"))
    if not tars:
        print(f"No .tar files in {images_dir}", file=sys.stderr)
        sys.exit(1)

    if args.first_tar_only:
        to_run = [tars[0]]
        tqdm.write(
            f"FIRST TAR ONLY: {to_run[0].name} "
            f"(use default without --first-tar-only for all {len(tars)} archives)\n"
            f"  output -> {output_root / 'images'}/"
        )
    else:
        to_run = tars
        if args.all:
            tqdm.write(
                f"Processing {len(to_run)} tar file(s) -> {output_root / 'images'} "
                f"(--all is optional; all archives are processed by default)"
            )
        else:
            tqdm.write(
                f"Processing {len(to_run)} tar file(s) -> {output_root / 'images'}"
            )

    per_tar_limit = args.limit_images
    total_limit = args.limit_total
    write_empty = not args.skip_caption_if_missing
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

    global_written = 0
    for i, tp in enumerate(outer_iter):
        inner_leave = not use_outer or (i == len(to_run) - 1)
        eff_limit: int | None
        if total_limit is not None:
            remaining = total_limit - global_written
            if remaining <= 0:
                tqdm.write(f"Reached --limit-total {total_limit}; stopping.")
                break
            if per_tar_limit is not None:
                eff_limit = min(per_tar_limit, remaining)
            else:
                eff_limit = remaining
        else:
            eff_limit = per_tar_limit

        o, s = process_tar(
            tp,
            output_root,
            caption_index,
            limit_images=eff_limit,
            workers=max(1, args.workers),
            inner_leave=inner_leave,
            inner_position=1 if use_outer else 0,
            write_empty_caption_if_missing=write_empty,
        )
        global_written += o
        tqdm.write(f"  {tp.name}: wrote {o} png(s), skipped {s} (running total: {global_written})")
        if total_limit is not None and global_written >= total_limit:
            tqdm.write(f"Done: --limit-total {total_limit} reached.")
            break


if __name__ == "__main__":
    main()
