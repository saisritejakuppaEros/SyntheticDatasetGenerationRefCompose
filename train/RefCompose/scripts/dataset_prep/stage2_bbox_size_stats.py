#!/usr/bin/env python3
"""
Summarize YOLO detection CSV: bbox pixel sizes and how many pass a min-side filter
(same rule as stage3_multiview: width > T and height > T).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="BBox size stats for yolo26_detections.csv")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent / "output/bbox_results/yolo26_detections.csv",
        help="Path to detections CSV",
    )
    parser.add_argument(
        "--min-side",
        type=float,
        default=200.0,
        help="Threshold for 'large enough' box (strictly greater on both w and h).",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    required = ("x1", "y1", "x2", "y2", "image_path")
    for c in required:
        if c not in df.columns:
            raise SystemExit(f"Missing column: {c}")

    n = len(df)
    df = df.copy()
    df["w"] = df["x2"] - df["x1"]
    df["h"] = df["y2"] - df["y1"]
    df["min_side"] = df[["w", "h"]].min(axis=1)
    df["max_side"] = df[["w", "h"]].max(axis=1)
    df["area"] = df["w"] * df["h"]

    t = args.min_side
    gt_both = (df["w"] > t) & (df["h"] > t)
    ge_both = (df["w"] >= t) & (df["h"] >= t)

    n_gt = int(gt_both.sum())
    n_ge = int(ge_both.sum())
    images_total = df["image_path"].nunique()
    images_with_gt = df.loc[gt_both, "image_path"].nunique()

    print("=== Bbox size stats ===")
    print(f"CSV: {args.csv.resolve()}")
    print(f"Total detection rows: {n:,}")
    print(f"Unique images: {images_total:,}")
    print()
    print(f"Threshold: min_side = {t} (stage3 rule: w > {t} AND h > {t})")
    print(f"  Boxes with w > {t} AND h > {t}:     {n_gt:,}  ({100.0 * n_gt / n:.2f}% of all boxes)")
    print(f"  Boxes with w >= {t} AND h >= {t}:  {n_ge:,}  ({100.0 * n_ge / n:.2f}% of all boxes)")
    print(f"  Images with at least one such box: {images_with_gt:,} / {images_total:,}")
    print()

    # Distribution of min(w,h) — shows how many are "almost" large enough
    qs = [0, 10, 25, 50, 100, 150, 200, 250, 300, 400, 500]
    print("min(w, h) percentiles (pixels):")
    for q in (5, 25, 50, 75, 95, 99):
        print(f"  p{q}: {df['min_side'].quantile(q / 100.0):.1f}")
    print()

    print("Counts by min(w,h) bucket (all boxes):")
    bins = [-1, 0, 50, 100, 150, 200, 250, 300, 400, 1e9]
    labels = ["<=0", "(0,50]", "(50,100]", "(100,150]", "(150,200]", "(200,250]", "(250,300]", "(300,400]", ">400"]
    df["_bucket"] = pd.cut(df["min_side"], bins=bins, labels=labels)
    counts = df["_bucket"].value_counts().sort_index()
    for lab, c in counts.items():
        print(f"  {lab}: {int(c):,}")
    print()

    if "class_name" in df.columns:
        top = (
            df.loc[gt_both]
            .groupby("class_name", dropna=False)
            .size()
            .sort_values(ascending=False)
            .head(15)
        )
        print(f"Top class_name among boxes with w>{t} and h>{t}:")
        for name, c in top.items():
            print(f"  {name}: {int(c):,}")
        print()

    print("width / height summary (all boxes, pixels):")
    for col, label in [("w", "width"), ("h", "height")]:
        s = df[col]
        print(f"  {label}: mean={s.mean():.1f}  std={s.std():.1f}  min={s.min():.1f}  max={s.max():.1f}")


if __name__ == "__main__":
    main()
