#!/usr/bin/env python3
"""Smoke test for v2 manifest JSONL dataloader."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

_TRAIN_DIR = Path(__file__).resolve().parent.parent
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from src.manifest_dataset import collate_fn_canvas, make_manifest_train_dataset


def save_normalized_chw(t: torch.Tensor, path: Path) -> None:
    x = (t.detach().float().cpu() * 0.5 + 0.5).clamp(0, 1)
    save_image(x, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest_path",
        type=str,
        default="/mnt/data0/teja/research_multiref/synthetic_dataset_generation/v2/outputs/dataset_manifest.jsonl",
    )
    parser.add_argument("--dataset_index", type=int, default=0)
    parser.add_argument("--unified_train_width", type=int, default=1280)
    parser.add_argument("--unified_train_height", type=int, default=720)
    parser.add_argument("--cond_size", type=int, default=512)
    parser.add_argument("--noise_size", type=int, default=1280)
    args = parser.parse_args()

    args.dataset_type = "manifest"
    args.canvas_random_target_resolution = False
    args.canvas_unified_resize = "cover"
    args.depth_keep_prob = 1.0
    args.canvas_keep_prob = 1.0
    args.canvas_depth_coarse_augment = False
    args.canvas_augment = False

    out_dir = Path(__file__).resolve().parent / "manifest_dataloader_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = make_manifest_train_dataset(args)
    print(f"manifest dataset length: {len(ds)}")
    if not (0 <= args.dataset_index < len(ds)):
        raise SystemExit(f"dataset_index out of range: {args.dataset_index}")

    sample = ds[args.dataset_index]
    batch = collate_fn_canvas([sample])
    print("prompt preview:", batch["prompts"][0][:160], "...")
    print("pixel_values:", tuple(batch["pixel_values"].shape))
    print("subject_pixel_values:", tuple(batch["subject_pixel_values"].shape))
    print("cond_pixel_values:", tuple(batch["cond_pixel_values"].shape))

    save_normalized_chw(batch["pixel_values"][0], out_dir / "target.png")
    save_normalized_chw(batch["subject_pixel_values"][0], out_dir / "canvas.png")
    save_normalized_chw(batch["cond_pixel_values"][0], out_dir / "depth.png")
    print(f"saved previews under {out_dir}")


if __name__ == "__main__":
    main()
