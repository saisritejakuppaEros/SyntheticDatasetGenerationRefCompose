#!/usr/bin/env python3
"""
Extract image bytes from Parquet shards under dataset_preparation into images/.

Handles common layouts:
  - Column `image`: dict with key `bytes` (Hugging Face datasets)
  - Any column whose cell is raw `bytes`
  - Optional id column for filenames (e.g. `id`, `__index_level_0__`)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def _ext_from_magic(b: bytes) -> str:
    if len(b) >= 3 and b[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if len(b) >= 8 and b[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return ".webp"
    if len(b) >= 6 and (b[:6] in (b"GIF87a", b"GIF89a")):
        return ".gif"
    return ".bin"


def _normalize_image_cell(cell) -> bytes | None:
    if cell is None:
        return None
    if isinstance(cell, (bytes, bytearray, memoryview)):
        return bytes(cell)
    if isinstance(cell, dict):
        b = cell.get("bytes")
        if isinstance(b, (bytes, bytearray, memoryview)):
            return bytes(b)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract images from Parquet files to a folder.")
    parser.add_argument(
        "--parquet-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to search recursively for *.parquet (default: this script's folder).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "images",
        help="Output directory for image files (default: dataset_preparation/images).",
    )
    parser.add_argument(
        "--image-column",
        type=str,
        default="image",
        help="Name of the image column (default: image). If missing, first bytes-like column is used.",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default=None,
        help="Optional column for stable filenames (e.g. id). If unset, uses global row index.",
    )
    args = parser.parse_args()

    all_parquet = sorted(args.parquet_root.rglob("*.parquet"))
    if not all_parquet:
        raise SystemExit(f"No .parquet files under {args.parquet_root}")

    # Hugging Face / git-lfs style checkouts often use symlinks to ../blobs/<hash>.
    # Skip missing or broken symlinks so pyarrow does not raise FileNotFoundError.
    parquet_files: list[Path] = []
    for p in all_parquet:
        if p.is_file():
            parquet_files.append(p)
        else:
            print(f"Skipping unreadable path (missing blob or broken symlink): {p}")

    if not parquet_files:
        raise SystemExit(
            "No readable .parquet files found. If these are symlinks to blobs/, "
            "finish the dataset download (e.g. `git lfs pull`) or copy real parquet files here."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    global_idx = 0
    default_image_col = args.image_column
    for pq_path in parquet_files:
        table = pq.read_table(pq_path)
        names = table.column_names

        col_name = default_image_col
        if col_name in names:
            img_col = table.column(col_name)
        else:
            # pick first column that yields bytes from first row
            img_col = None
            for n in names:
                c = table.column(n)
                if len(c) == 0:
                    continue
                if _normalize_image_cell(c[0].as_py()) is not None:
                    img_col = c
                    col_name = n
                    break
            if img_col is None:
                raise SystemExit(
                    f"No image column found in {pq_path}; columns: {names}. "
                    "Pass --image-column explicitly."
                )

        id_col = None
        if args.id_column and args.id_column in names:
            id_col = table.column(args.id_column)

        nrows = table.num_rows
        stem = pq_path.stem
        for i in range(nrows):
            raw = _normalize_image_cell(img_col[i].as_py())
            if raw is None:
                continue

            if id_col is not None:
                fid = id_col[i].as_py()
                base = f"{fid}"
            else:
                base = f"{stem}_{global_idx:08d}"

            ext = _ext_from_magic(raw)
            out_path = args.out_dir / f"{base}{ext}"
            # avoid clobbering same id across shards
            if out_path.exists() and id_col is not None:
                out_path = args.out_dir / f"{base}_{stem}{ext}"

            out_path.write_bytes(raw)
            global_idx += 1

        print(f"{pq_path.name}: wrote {nrows} rows (column {col_name!r})")

    print(f"Done. Output: {args.out_dir} ({global_idx} images total).")


if __name__ == "__main__":
    main()


